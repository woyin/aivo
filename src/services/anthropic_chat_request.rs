//! Shared Anthropic Messages API -> OpenAI Chat Completions request conversion.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

use crate::services::openai_anthropic_bridge::{
    ANTHROPIC_SERVER_BLOCKS_EXT, ANTHROPIC_THINKING_EXT,
};

/// Report describing what [`hoist_anthropic_system_messages`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemHoistReport {
    /// `role:"system"` entries removed from `messages`.
    pub hoisted_messages: usize,
    /// Text blocks appended to the top-level `system` field as a result.
    pub hoisted_blocks: usize,
}

/// Normalize an Anthropic request body in place by hoisting stray
/// `role:"system"` messages into the top-level `system` field.
///
/// Anthropic's Messages API only permits `user`/`assistant` roles inside
/// `messages`; the system prompt is a top-level `system` field (string or an
/// array of content blocks). Some clients — Claude Code among them — emit a
/// `role:"system"` entry *inside* `messages`, often alongside a correct
/// top-level `system` field. Strict upstreams (real Anthropic, DeepSeek's
/// `/anthropic` endpoint) 400 with
/// "messages[N].role: unknown variant `system`, expected `user` or `assistant`",
/// and the Anthropic→OpenAI bridge would otherwise forward it as a second,
/// mid-conversation system message.
///
/// Hoisted content is appended *after* the existing system (document order —
/// Claude Code emits the skills/commands catalog as a trailing system block) as
/// plain `{type:"text"}` blocks: `cache_control` and other extras are stripped
/// so the repair can't push a request past Anthropic's 4-breakpoint
/// cache_control limit. Returns `None` when nothing was hoisted, leaving a
/// clean request's `system` field untouched.
pub fn hoist_anthropic_system_messages(body: &mut Value) -> Option<SystemHoistReport> {
    let mut hoisted_text: Vec<String> = Vec::new();
    let mut hoisted_messages = 0usize;

    {
        let messages = body.get_mut("messages").and_then(|m| m.as_array_mut())?;
        messages.retain(|msg| {
            if msg.get("role").and_then(|r| r.as_str()) != Some("system") {
                return true;
            }
            hoisted_messages += 1;
            hoisted_text.extend(extract_system_text(msg.get("content")));
            false
        });
    }

    if hoisted_messages == 0 {
        return None;
    }

    let hoisted_blocks = hoisted_text.len();

    // Existing system blocks first (verbatim, preserving cache_control), then
    // the hoisted text appended as plain blocks.
    let mut combined: Vec<Value> = match body.get("system") {
        Some(Value::String(s)) if !s.is_empty() => vec![json!({"type": "text", "text": s})],
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };
    combined.extend(
        hoisted_text
            .into_iter()
            .map(|t| json!({"type": "text", "text": t})),
    );

    // Nothing usable (e.g. an empty system message hoisted onto no existing
    // system): drop the now-removed messages but leave `system` as-is.
    if combined.is_empty() {
        return Some(SystemHoistReport {
            hoisted_messages,
            hoisted_blocks: 0,
        });
    }

    let has_extra_fields = combined.iter().any(|b| {
        b.as_object()
            .is_some_and(|o| o.keys().any(|k| k != "type" && k != "text"))
    });
    if has_extra_fields {
        body["system"] = Value::Array(combined);
    } else {
        let text = combined
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            body["system"] = Value::String(text);
        }
    }

    Some(SystemHoistReport {
        hoisted_messages,
        hoisted_blocks,
    })
}

fn extract_system_text(content: Option<&Value>) -> Vec<String> {
    match content {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) if !s.is_empty() => Some(s.clone()),
                Value::Object(_) => p
                    .get("text")
                    .and_then(|t| t.as_str())
                    .filter(|t| !t.is_empty())
                    .map(|t| t.to_string()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

pub type ModelTransform = fn(&str) -> String;

#[derive(Clone, Copy, Debug)]
pub struct AnthropicToOpenAIConfig<'a> {
    pub default_model: &'a str,
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

    let stream = if config.preserve_stream {
        body.get("stream").cloned().unwrap_or(json!(false))
    } else {
        json!(false)
    };
    let mut req = json!({
        "model": model,
        "messages": messages,
        "stream": stream,
    });
    // OpenAI-compatible providers only emit `usage` in stream chunks when
    // `stream_options.include_usage` is set. Without it, xAI/DeepSeek/etc.
    // close the stream with no usage event, the bridge writes `message_delta`
    // with zero tokens, and Claude Code logs all-zero usage — making the
    // model invisible to `aivo stats` (which filters zero-token rows).
    if stream == json!(true) {
        req["stream_options"] = json!({"include_usage": true});
    }

    if let Some(mt) = body.get("max_tokens") {
        req["max_tokens"] = mt.clone();
    }
    // o-series / forced-reasoning models reject sampling params outright —
    // forwarding them turns into upstream 400s.
    if !crate::services::model_metadata::rejects_temperature(&model) {
        if let Some(t) = body.get("temperature") {
            req["temperature"] = t.clone();
        }
        if let Some(tp) = body.get("top_p") {
            req["top_p"] = tp.clone();
        }
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
            // Anthropic "none" forbids tool calls; dropping it would let the
            // OpenAI default (auto) call a tool the caller explicitly banned.
            Some("none") => {
                req["tool_choice"] = json!("none");
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
    // Image blocks in regular (non-tool_result) messages → OpenAI image_url
    // parts; without this a pasted image silently vanishes on the bridge.
    let mut image_parts: Vec<Value> = Vec::new();

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
            "image" => {
                // Same translation as tool_result images (base64 → data URL).
                if let Some(part) = convert_tool_result_part(block) {
                    image_parts.push(part);
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
        // Text sharing the message (interrupt notes, system reminders) must
        // follow the tool messages — a user message between an assistant
        // tool_calls turn and its results breaks pairing on strict upstreams
        // (DeepSeek).
        if !text_parts.is_empty() || !image_parts.is_empty() {
            let content = if !image_parts.is_empty() {
                let mut parts = text_parts_with_cache;
                parts.extend(image_parts);
                Value::Array(parts)
            } else if any_text_has_cache_control {
                Value::Array(text_parts_with_cache)
            } else {
                Value::String(text_parts.join("\n"))
            };
            messages.push(json!({"role": role, "content": content}));
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
        let content = if !image_parts.is_empty() {
            let mut parts = text_parts_with_cache;
            parts.extend(image_parts);
            Value::Array(parts)
        } else if any_text_has_cache_control {
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

    fn test_config() -> AnthropicToOpenAIConfig<'static> {
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

    fn streaming_config() -> AnthropicToOpenAIConfig<'static> {
        AnthropicToOpenAIConfig {
            preserve_stream: true,
            ..test_config()
        }
    }

    #[test]
    fn tool_choice_none_maps_to_openai_none() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "none"}
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["tool_choice"], "none");
    }

    #[test]
    fn user_message_image_block_becomes_image_url_part() {
        let body = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": "aGk="
                    }}
                ]
            }],
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What is this?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,aGk=");
    }

    #[test]
    fn image_only_user_message_survives() {
        let body = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [{"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": "aGk="
                }}]
            }],
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "image_url");
    }

    #[test]
    fn text_only_message_keeps_string_content() {
        let body = json!({
            "model": "m",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hi"}]
            }],
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["messages"][0]["content"], "hi");
    }

    #[test]
    fn streaming_request_sets_include_usage() {
        // Without stream_options.include_usage, xAI / DeepSeek / OpenRouter
        // never emit a usage event in the SSE stream, so `aivo run claude`
        // backed by an OpenAI-shape upstream logs zero-token assistant turns.
        let body = json!({
            "model": "grok-4.3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let req = convert_anthropic_to_openai_request(&body, &streaming_config());
        assert_eq!(req["stream"], json!(true));
        assert_eq!(req["stream_options"], json!({"include_usage": true}));
    }

    #[test]
    fn non_streaming_request_omits_stream_options() {
        let body = json!({
            "model": "grok-4.3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        });
        let req = convert_anthropic_to_openai_request(&body, &streaming_config());
        assert_eq!(req["stream"], json!(false));
        assert!(req.get("stream_options").is_none());
    }

    #[test]
    fn forced_non_streaming_omits_stream_options_even_if_client_streams() {
        let body = json!({
            "model": "grok-4.3",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["stream"], json!(false));
        assert!(req.get("stream_options").is_none());
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
    fn tool_result_sibling_text_survives_and_follows_tool_messages() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Request interrupted by user]"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}
                ]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        let msgs = req["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        // Tool message first regardless of block order — a user message
        // before it would detach the result from its tool_calls turn.
        assert_eq!(msgs[0]["role"], "tool");
        assert_eq!(msgs[0]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "[Request interrupted by user]");
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

#[cfg(test)]
mod system_hoist_tests {
    use super::*;

    #[test]
    fn clean_request_returns_none_and_leaves_body_untouched() {
        let mut body = json!({
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ]
        });
        let before = body.clone();
        assert_eq!(hoist_anthropic_system_messages(&mut body), None);
        assert_eq!(body, before);
    }

    #[test]
    fn returns_none_when_no_messages_array() {
        let mut with_system = json!({"system": "x"});
        assert_eq!(hoist_anthropic_system_messages(&mut with_system), None);
        let mut empty = json!({});
        assert_eq!(hoist_anthropic_system_messages(&mut empty), None);
    }

    #[test]
    fn hoists_into_string_system_in_append_order() {
        let mut body = json!({
            "system": "You are Claude.",
            "messages": [
                {"role": "user", "content": "do a thing"},
                {"role": "system", "content": "Available skills: review, security-review."}
            ]
        });
        let report = hoist_anthropic_system_messages(&mut body).expect("hoisted");
        assert_eq!(
            report,
            SystemHoistReport {
                hoisted_messages: 1,
                hoisted_blocks: 1
            }
        );
        assert_eq!(
            body["system"],
            json!("You are Claude.\nAvailable skills: review, security-review.")
        );
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": "do a thing"}])
        );
    }

    #[test]
    fn sets_system_when_absent() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"}
            ]
        });
        let report = hoist_anthropic_system_messages(&mut body).expect("hoisted");
        assert_eq!(
            report,
            SystemHoistReport {
                hoisted_messages: 1,
                hoisted_blocks: 1
            }
        );
        assert_eq!(body["system"], json!("You are helpful."));
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    // The exact shape captured from the gateway: a role:system message with
    // array content (cache_control) alongside an array-form top-level system.
    #[test]
    fn observed_payload_preserves_array_system_and_strips_hoisted_cache_control() {
        let mut body = json!({
            "max_tokens": 32000,
            "messages": [
                {"role": "user", "content": [{"text": "<system-reminder>..."}]},
                {"role": "system", "content": [
                    {"cache_control": {"type": "ephemeral"}, "text": "- review: view a pull request", "type": "text"}
                ]}
            ],
            "system": [
                {"cache_control": {"type": "ephemeral"}, "text": "You are Claude Code, Anthropic's official CLI.", "type": "text"}
            ]
        });
        let report = hoist_anthropic_system_messages(&mut body).expect("hoisted");
        assert_eq!(
            report,
            SystemHoistReport {
                hoisted_messages: 1,
                hoisted_blocks: 1
            }
        );

        let msgs = body["messages"].as_array().expect("messages array");
        assert_eq!(msgs.len(), 1);
        assert!(!msgs.iter().any(|m| m["role"] == "system"));

        assert_eq!(
            body["system"],
            json!([
                {"cache_control": {"type": "ephemeral"}, "text": "You are Claude Code, Anthropic's official CLI.", "type": "text"},
                {"type": "text", "text": "- review: view a pull request"}
            ])
        );
        let breakpoints = body["system"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b.get("cache_control").is_some())
            .count();
        assert_eq!(breakpoints, 1);
    }

    #[test]
    fn merges_multiple_system_messages_in_order() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "q"},
                {"role": "system", "content": [{"type": "text", "text": "B"}, {"type": "text", "text": "C"}]}
            ]
        });
        let report = hoist_anthropic_system_messages(&mut body).expect("hoisted");
        assert_eq!(
            report,
            SystemHoistReport {
                hoisted_messages: 2,
                hoisted_blocks: 3
            }
        );
        assert_eq!(body["system"], json!("A\nB\nC"));
        assert_eq!(body["messages"], json!([{"role": "user", "content": "q"}]));
    }

    #[test]
    fn empty_system_message_dropped_without_inventing_system() {
        let mut body = json!({
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hi"}
            ]
        });
        let report = hoist_anthropic_system_messages(&mut body).expect("hoisted");
        assert_eq!(
            report,
            SystemHoistReport {
                hoisted_messages: 1,
                hoisted_blocks: 0
            }
        );
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
        assert!(body.get("system").is_none());
    }
}
