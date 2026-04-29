//! Shared Anthropic Messages API -> OpenAI Chat Completions request conversion.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

use crate::services::openai_anthropic_bridge::{
    ANTHROPIC_SERVER_BLOCKS_EXT, ANTHROPIC_THINKING_EXT,
};

pub type ModelTransform = fn(&str) -> String;

#[derive(Clone, Copy, Debug)]
pub struct AnthropicToOpenAIConfig {
    pub default_model: &'static str,
    pub preserve_stream: bool,
    pub model_transform: Option<ModelTransform>,
    pub include_reasoning_content: bool,
    pub require_non_empty_reasoning_content: bool,
    pub stringify_other_tool_result_content: bool,
    /// When true, `tool_result` blocks containing images emit an OpenAI
    /// multimodal content array (`[{type:text,...},{type:image_url,...}]`).
    /// When false, image parts are stripped and only text is kept — the
    /// legacy behavior, safe for providers that reject array content in
    /// `role: tool` messages.
    pub tool_result_supports_multimodal: bool,
    pub fallback_tool_arguments_json: &'static str,
}

/// True if a tool definition is an Anthropic server-side built-in
/// (`web_search_*`, `code_execution_*`, `computer_*`, `bash_*`,
/// `text_editor_*`, `web_fetch_*`, `mcp_*`). These have no `input_schema`
/// and can't be executed by an OpenAI-compatible upstream, so we drop them
/// during the bridge to avoid 400s on the empty schema.
pub(crate) fn is_anthropic_server_tool(tool: &Value) -> bool {
    let Some(t) = tool.get("type").and_then(|v| v.as_str()) else {
        return false;
    };
    !matches!(t, "" | "custom" | "function")
}

/// Convert an Anthropic tool's `input_schema` into an OpenAI-compatible
/// `parameters` object. Strict OpenAI validators reject `{}` or a schema
/// without `type` ("schema must be a JSON Schema of type 'object'"), so we
/// always emit a usable object schema.
pub(crate) fn tool_parameters_from_input_schema(input_schema: Option<&Value>) -> Value {
    let default = || json!({"type": "object", "properties": {}});
    match input_schema {
        None | Some(Value::Null) => default(),
        Some(Value::Object(map)) if map.is_empty() => default(),
        Some(v) => {
            if v.get("type").is_none() {
                let mut owned = v.clone();
                if let Some(obj) = owned.as_object_mut() {
                    obj.insert("type".to_string(), json!("object"));
                }
                owned
            } else {
                v.clone()
            }
        }
    }
}

pub fn convert_anthropic_to_openai_request(
    body: &Value,
    config: &AnthropicToOpenAIConfig,
) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = body.get("system") {
        // System prompts are the most common Anthropic cache target. Keep
        // cache_control (including ttl: "1h") on the OpenAI side via the
        // structured array form when any system block carries it; OpenRouter
        // and other Anthropic-aware gateways pass it through.
        match system {
            Value::String(s) if !s.is_empty() => {
                messages.push(json!({"role": "system", "content": s}));
            }
            Value::Array(blocks) => {
                let mut parts: Vec<Value> = Vec::new();
                let mut has_cache_control = false;
                for block in blocks {
                    let Some(text) = block.get("text").and_then(|t| t.as_str()) else {
                        continue;
                    };
                    let mut part = json!({"type": "text", "text": text});
                    if let Some(cc) = block.get("cache_control") {
                        part["cache_control"] = cc.clone();
                        has_cache_control = true;
                    }
                    parts.push(part);
                }
                if !parts.is_empty() {
                    let content = if has_cache_control {
                        Value::Array(parts)
                    } else {
                        Value::String(
                            parts
                                .iter()
                                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                                .collect::<Vec<_>>()
                                .join("\n"),
                        )
                    };
                    messages.push(json!({"role": "system", "content": content}));
                }
            }
            _ => {}
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match msg.get("content") {
                Some(Value::String(text)) => {
                    let mut out = json!({"role": role, "content": text});
                    if config.include_reasoning_content
                        && config.require_non_empty_reasoning_content
                    {
                        ensure_assistant_reasoning_content(&mut out);
                    }
                    messages.push(out);
                }
                Some(Value::Array(blocks)) => {
                    convert_content_blocks(blocks, role, &mut messages, config);
                }
                _ => {
                    let mut out = json!({"role": role, "content": ""});
                    if config.include_reasoning_content
                        && config.require_non_empty_reasoning_content
                    {
                        ensure_assistant_reasoning_content(&mut out);
                    }
                    messages.push(out);
                }
            }
        }
    }

    let raw_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(config.default_model);
    let model = config
        .model_transform
        .map(|transform| transform(raw_model))
        .unwrap_or_else(|| raw_model.to_string());

    let mut req = json!({
        "model": model,
        "messages": messages,
        "stream": if config.preserve_stream {
            body.get("stream").cloned().unwrap_or(json!(false))
        } else {
            json!(false)
        },
    });

    if let Some(mt) = body.get("max_tokens") {
        req["max_tokens"] = mt.clone();
    }
    if let Some(t) = body.get("temperature") {
        req["temperature"] = t.clone();
    }
    if let Some(tp) = body.get("top_p") {
        req["top_p"] = tp.clone();
    }
    if let Some(ss) = body.get("stop_sequences") {
        req["stop"] = ss.clone();
    }

    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .filter(|t| !is_anthropic_server_tool(t))
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or_default(),
                        "description": tool.get("description").cloned().unwrap_or(json!("")),
                        "parameters": tool_parameters_from_input_schema(tool.get("input_schema")),
                    }
                })
            })
            .collect();
        if !openai_tools.is_empty() {
            req["tools"] = Value::Array(openai_tools);
        }
    }

    if let Some(tc) = body.get("tool_choice") {
        // Anthropic disable_parallel_tool_use → OpenAI parallel_tool_calls:false
        if tc.get("disable_parallel_tool_use") == Some(&json!(true)) {
            req["parallel_tool_calls"] = json!(false);
        }
        match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => {
                req["tool_choice"] = json!("auto");
            }
            Some("any") => {
                req["tool_choice"] = json!("required");
            }
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
    }

    req
}

pub(crate) fn ensure_assistant_reasoning_content(message: &mut Value) {
    if message.get("role").and_then(|r| r.as_str()) != Some("assistant") {
        return;
    }
    if message
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .is_some_and(|s| !s.is_empty())
    {
        return;
    }

    let fallback = match message.get("content") {
        Some(Value::String(text)) if !text.is_empty() => text.clone(),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .or_else(|| part.get("input_text"))
                        .and_then(|v| v.as_str())
                })
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                " ".to_string()
            } else {
                text
            }
        }
        _ => " ".to_string(),
    };
    message["reasoning_content"] = Value::String(fallback);
}

pub(crate) fn ensure_assistant_reasoning_content_in_chat_request(request: &mut Value) {
    if let Some(messages) = request.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for message in messages {
            ensure_assistant_reasoning_content(message);
        }
    }
}

fn convert_content_blocks(
    blocks: &[Value],
    role: &str,
    messages: &mut Vec<Value>,
    config: &AnthropicToOpenAIConfig,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, Value)> = Vec::new();
    // Full thinking / redacted_thinking blocks (with signature / data) for
    // round-trip preservation. These travel on an extension field on the
    // OpenAI assistant message; see ANTHROPIC_THINKING_EXT.
    let mut anthropic_thinking_blocks: Vec<Value> = Vec::new();
    // Anthropic server-tool blocks (web_search_tool_result, code_execution_tool_result,
    // server_tool_use, etc.) — opaque JSON we round-trip via the OpenAI
    // extension so server-side tool output isn't lost across the bridge.
    let mut anthropic_server_blocks: Vec<Value> = Vec::new();
    // Parallel structured form: same text content, but each part keeps its
    // `cache_control` (including the `ttl: "1h"` field) so OpenAI-shape
    // upstreams that pass-through the annotation (OpenRouter, Anthropic-via-
    // OpenAI-shape gateways) can still hit the prompt cache. Used only when
    // at least one block carries cache_control; otherwise we keep the flat
    // string form for legacy callers.
    let mut text_parts_with_cache: Vec<Value> = Vec::new();
    let mut any_text_has_cache_control = false;

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        // Server-tool blocks travel verbatim on an extension field. They
        // must not be merged into text_parts or tool_calls because they
        // describe Anthropic-side built-in work, not OpenAI tool calls.
        if matches!(
            block_type,
            "server_tool_use"
                | "web_search_tool_result"
                | "code_execution_tool_result"
                | "web_fetch_tool_result"
                | "mcp_tool_use"
                | "mcp_tool_result"
        ) {
            anthropic_server_blocks.push(block.clone());
            continue;
        }
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                    let mut part = json!({"type": "text", "text": text});
                    if let Some(cc) = block.get("cache_control") {
                        part["cache_control"] = cc.clone();
                        any_text_has_cache_control = true;
                    }
                    text_parts_with_cache.push(part);
                }
            }
            "thinking" => {
                if config.include_reasoning_content
                    && let Some(thinking) = block
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .or_else(|| block.get("text").and_then(|t| t.as_str()))
                {
                    thinking_parts.push(thinking.to_string());
                }
                // Capture the full block (with signature) regardless of
                // whether reasoning_content is being surfaced. Without this,
                // a continuation through this bridge loses the cryptographic
                // signature and Anthropic 400s on the next turn.
                let mut entry = json!({"type": "thinking"});
                if let Some(text) = block.get("thinking").cloned() {
                    entry["thinking"] = text;
                }
                if let Some(sig) = block.get("signature").cloned() {
                    entry["signature"] = sig;
                }
                anthropic_thinking_blocks.push(entry);
            }
            "redacted_thinking" => {
                let mut entry = json!({"type": "redacted_thinking"});
                if let Some(data) = block.get("data").cloned() {
                    entry["data"] = data;
                }
                anthropic_thinking_blocks.push(entry);
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(json!({}));

                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input)
                            .unwrap_or_else(|_| config.fallback_tool_arguments_json.to_string()),
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let content = convert_tool_result_content(block.get("content"), config);
                tool_results.push((tool_use_id, content));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        for (tool_use_id, content) in tool_results {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }));
        }
    } else if !tool_calls.is_empty() {
        // Per OpenAI spec, content must be null (not "") when tool_calls is
        // present without text. Strict OpenAI-compatible providers reject the
        // empty-string form. When any text block has cache_control, surface
        // the structured array so the annotation passes through to the
        // upstream (OpenRouter forwards it on Anthropic models).
        let content = if text_parts.is_empty() {
            Value::Null
        } else if any_text_has_cache_control {
            Value::Array(text_parts_with_cache.clone())
        } else {
            Value::String(text_parts.join("\n"))
        };
        let mut msg = json!({"role": role, "content": content, "tool_calls": tool_calls});
        if config.include_reasoning_content {
            if role == "assistant" && config.require_non_empty_reasoning_content {
                if !thinking_parts.is_empty() {
                    msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
                }
                ensure_assistant_reasoning_content(&mut msg);
            } else if !thinking_parts.is_empty() {
                msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
            }
        }
        if role == "assistant" && !anthropic_thinking_blocks.is_empty() {
            msg[ANTHROPIC_THINKING_EXT] = Value::Array(anthropic_thinking_blocks);
        }
        if role == "assistant" && !anthropic_server_blocks.is_empty() {
            msg[ANTHROPIC_SERVER_BLOCKS_EXT] = Value::Array(anthropic_server_blocks);
        }
        messages.push(msg);
    } else {
        let content = if any_text_has_cache_control {
            Value::Array(text_parts_with_cache)
        } else {
            Value::String(text_parts.join("\n"))
        };
        let mut msg = json!({"role": role, "content": content});
        if config.include_reasoning_content {
            if role == "assistant" && config.require_non_empty_reasoning_content {
                if !thinking_parts.is_empty() {
                    msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
                }
                ensure_assistant_reasoning_content(&mut msg);
            } else if !thinking_parts.is_empty() {
                msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
            }
        }
        if role == "assistant" && !anthropic_thinking_blocks.is_empty() {
            msg[ANTHROPIC_THINKING_EXT] = Value::Array(anthropic_thinking_blocks);
        }
        if role == "assistant" && !anthropic_server_blocks.is_empty() {
            msg[ANTHROPIC_SERVER_BLOCKS_EXT] = Value::Array(anthropic_server_blocks);
        }
        messages.push(msg);
    }
}

/// Convert an Anthropic `tool_result.content` into OpenAI `role: tool`
/// message content. Emits a plain string when all parts are text, or a
/// multimodal array when image parts are present and the config opts in.
fn convert_tool_result_content(content: Option<&Value>, config: &AnthropicToOpenAIConfig) -> Value {
    let parts = match content {
        Some(Value::String(s)) => return Value::String(s.clone()),
        Some(Value::Array(parts)) => parts,
        Some(v) if config.stringify_other_tool_result_content => {
            return Value::String(v.to_string());
        }
        _ => return Value::String(String::new()),
    };

    if config.tool_result_supports_multimodal && parts.iter().any(is_image_block) {
        let openai_parts: Vec<Value> = parts.iter().filter_map(convert_tool_result_part).collect();
        if !openai_parts.is_empty() {
            return Value::Array(openai_parts);
        }
    }

    let text = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    Value::String(text)
}

fn is_image_block(part: &Value) -> bool {
    part.get("type").and_then(|t| t.as_str()) == Some("image")
}

/// Translate a single Anthropic `tool_result` part into its OpenAI shape.
/// Returns `None` for unknown/malformed blocks so they are skipped rather
/// than breaking the whole conversion.
fn convert_tool_result_part(part: &Value) -> Option<Value> {
    match part.get("type").and_then(|t| t.as_str())? {
        "text" => {
            let text = part.get("text").and_then(|t| t.as_str())?;
            Some(json!({"type": "text", "text": text}))
        }
        "image" => {
            let source = part.get("source")?;
            match source.get("type").and_then(|t| t.as_str())? {
                "base64" => {
                    let data = source.get("data").and_then(|t| t.as_str())?;
                    // Prefer the explicit schema field, but keep the prior
                    // best-effort behavior for buggy producers so images don't
                    // disappear entirely on the bridge.
                    let media = source
                        .get("media_type")
                        .and_then(|t| t.as_str())
                        .or_else(|| sniff_base64_image_media_type(data))
                        .unwrap_or("image/png");
                    Some(json!({
                        "type": "image_url",
                        "image_url": {"url": format!("data:{media};base64,{data}")},
                    }))
                }
                "url" => {
                    let url = source.get("url").and_then(|t| t.as_str())?;
                    Some(json!({
                        "type": "image_url",
                        "image_url": {"url": url},
                    }))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn sniff_base64_image_media_type(data: &str) -> Option<&'static str> {
    // Only the first 12 decoded bytes are needed to identify any of the four
    // magic patterns below. 16 base64 chars decode to 12 bytes and are always
    // 4-aligned, so decoding just that prefix avoids allocating the full
    // image buffer (can be MBs) every time we sniff. `get` rather than direct
    // slice so a non-ASCII (and therefore invalid) input fails sniffing
    // cleanly instead of panicking on a codepoint boundary.
    let prefix = data.get(..16).unwrap_or(data);
    let bytes = BASE64.decode(prefix).ok()?;
    match bytes.as_slice() {
        [0x89, b'P', b'N', b'G', ..] => Some("image/png"),
        [0xFF, 0xD8, 0xFF, ..] => Some("image/jpeg"),
        [b'G', b'I', b'F', b'8', ..] => Some("image/gif"),
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => Some("image/webp"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AnthropicToOpenAIConfig {
        AnthropicToOpenAIConfig {
            default_model: "test-model",
            preserve_stream: false,
            model_transform: None,
            include_reasoning_content: false,
            require_non_empty_reasoning_content: false,
            stringify_other_tool_result_content: false,
            tool_result_supports_multimodal: true,
            fallback_tool_arguments_json: "{}",
        }
    }

    #[test]
    fn disable_parallel_tool_use_maps_to_parallel_tool_calls_false() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["parallel_tool_calls"], json!(false));
        assert_eq!(req["tool_choice"], json!("auto"));
    }

    #[test]
    fn tool_choice_without_disable_parallel_omits_field() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"}
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert!(req.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn tool_result_preserves_image_blocks_as_multimodal() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "Screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
                    ]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let tool_msg = &req["messages"][0];
        assert_eq!(tool_msg["role"], "tool");
        let content = tool_msg["content"].as_array().expect("multimodal array");
        assert_eq!(content.len(), 2);
        assert_eq!(
            content[0],
            json!({"type": "text", "text": "Screenshot taken"})
        );
        assert_eq!(
            content[1],
            json!({
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            })
        );
    }

    #[test]
    fn tool_result_missing_media_type_keeps_image_with_sniffed_or_default_media() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        // JPEG magic bytes: FF D8 FF
                        {"type": "image", "source": {"type": "base64", "data": "/9j/"}}
                    ]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let content = req["messages"][0]["content"]
            .as_array()
            .expect("multimodal array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0],
            json!({
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/"},
            })
        );
    }

    #[test]
    fn tool_result_url_source_image_preserved_as_image_url() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}
                    ]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let content = req["messages"][0]["content"]
            .as_array()
            .expect("multimodal array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0],
            json!({
                "type": "image_url",
                "image_url": {"url": "https://example.com/x.png"},
            })
        );
    }

    #[test]
    fn tool_result_drops_images_when_multimodal_disabled() {
        let config = AnthropicToOpenAIConfig {
            tool_result_supports_multimodal: false,
            ..test_config()
        };
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "Screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "data": "abc"}}
                    ]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &config);
        assert_eq!(req["messages"][0]["content"], "Screenshot taken");
    }

    #[test]
    fn cache_control_ttl_preserved_on_text_blocks_via_structured_content() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "long reusable preamble", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                    {"type": "text", "text": "the actual question"}
                ]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let parts = req["messages"][0]["content"]
            .as_array()
            .expect("array form");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["cache_control"]["ttl"], "1h");
        assert!(parts[1].get("cache_control").is_none());
    }

    #[test]
    fn cache_control_absent_keeps_legacy_string_content_form() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"}
                ]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["messages"][0]["content"], "hello\nworld");
    }

    #[test]
    fn cache_control_preserved_on_anthropic_system_array_with_ttl() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "system": [
                {"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
            ],
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let system_msg = req["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "system")
            .expect("system message present");
        let parts = system_msg["content"].as_array().expect("array form");
        assert_eq!(parts[0]["cache_control"]["ttl"], "1h");
    }

    #[test]
    fn anthropic_thinking_blocks_round_trip_through_typed_path() {
        // Even when the bridge isn't the typed `openai_anthropic_bridge` path,
        // thinking signatures must survive the round trip — otherwise routes
        // that go through anthropic_chat_request.rs lose them and break
        // multi-turn extended thinking on Claude 4.
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Pondering.", "signature": "SIG_TYPED"},
                    {"type": "redacted_thinking", "data": "BLOB_TYPED"},
                    {"type": "text", "text": "Here we go."}
                ]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let assistant = &req["messages"][0];
        let blocks = assistant[ANTHROPIC_THINKING_EXT]
            .as_array()
            .expect("typed-path must surface thinking blocks on assistant message");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["signature"], "SIG_TYPED");
        assert_eq!(blocks[0]["thinking"], "Pondering.");
        assert_eq!(blocks[1]["data"], "BLOB_TYPED");
    }

    #[test]
    fn anthropic_thinking_blocks_not_attached_for_non_assistant_role() {
        // User-role messages can technically include `thinking` blocks in
        // synthetic histories, but propagating them onto a `role: user`
        // OpenAI message would confuse providers. Only assistant gets the
        // extension.
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "thinking", "thinking": "stray", "signature": "x"},
                    {"type": "text", "text": "hi"}
                ]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert!(
            req["messages"][0].get(ANTHROPIC_THINKING_EXT).is_none(),
            "extension must only attach to assistant messages"
        );
    }

    #[test]
    fn tool_result_pure_text_stays_string() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [{"type": "text", "text": "ok"}]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["messages"][0]["content"], "ok");
    }

    #[test]
    fn drops_anthropic_server_side_tools() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "ls", "description": "list", "input_schema": {"type": "object"}}
            ]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let tools = req["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "ls");
    }

    #[test]
    fn omits_tools_field_when_only_server_side_tools_present() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "search the web"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert!(req.get("tools").is_none());
    }

    #[test]
    fn missing_input_schema_emits_valid_object_schema() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "noop", "description": "no args"}]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let params = &req["tools"][0]["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert!(params.get("properties").is_some());
    }

    #[test]
    fn empty_input_schema_emits_valid_object_schema() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "noop", "description": "no args", "input_schema": {}}]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let params = &req["tools"][0]["function"]["parameters"];
        assert_eq!(params["type"], "object");
    }

    #[test]
    fn input_schema_without_type_gets_type_object_added() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "ls",
                "description": "list",
                "input_schema": {"properties": {"path": {"type": "string"}}}
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let params = &req["tools"][0]["function"]["parameters"];
        assert_eq!(params["type"], "object");
        assert_eq!(params["properties"]["path"]["type"], "string");
    }

    #[test]
    fn explicit_custom_tool_type_is_kept() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "custom",
                "name": "ls",
                "description": "list",
                "input_schema": {"type": "object"}
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let tools = req["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "ls");
    }
}
