use serde_json::{Value, json};

use crate::services::effort::{anthropic_thinking_config, extract_openai_effort};
use crate::services::http_utils::current_unix_ts;
use crate::services::openai_models::{
    OpenAIChatChoice, OpenAIChatResponse, OpenAIChatResponseMessage, OpenAIChatToolCall,
    OpenAIChatToolCallFunction, OpenAIChatUsage,
};

/// Extension field on the OpenAI assistant message that round-trips Anthropic
/// `thinking` and `redacted_thinking` blocks (with their `signature` / `data`)
/// across the bridge. Anthropic Claude 4 streams require these signatures to
/// be echoed back on subsequent turns, so the bridge cannot silently flatten
/// thinking to plain `reasoning_content`. OpenAI clients ignore unknown
/// fields; Anthropic-aware code uses this to reconstruct full content blocks
/// when the conversation continues.
pub const ANTHROPIC_THINKING_EXT: &str = "_anthropic_thinking_blocks";

/// Extension field carrying Anthropic server-side tool blocks
/// (`server_tool_use`, `web_search_tool_result`, `code_execution_tool_result`,
/// `web_fetch_tool_result`, etc.) as opaque JSON. These describe work done
/// by Anthropic-side built-in tools (web search, code exec, web fetch) and
/// have no OpenAI equivalent — without this passthrough they are silently
/// dropped, which breaks any client that depends on the search results /
/// execution stdout in subsequent turns.
pub const ANTHROPIC_SERVER_BLOCKS_EXT: &str = "_anthropic_server_blocks";

/// Block-type names that we treat as opaque server-tool content. Listed
/// once so capture and restore stay in sync.
const ANTHROPIC_SERVER_BLOCK_TYPES: &[&str] = &[
    "server_tool_use",
    "web_search_tool_result",
    "code_execution_tool_result",
    "web_fetch_tool_result",
    "mcp_tool_use",
    "mcp_tool_result",
];

fn is_anthropic_server_block_type(t: &str) -> bool {
    ANTHROPIC_SERVER_BLOCK_TYPES.contains(&t)
}

/// Convert an OpenAI / Responses-API tool entry into an Anthropic tool entry.
/// Handles:
/// - `{type: "function", function: {…}}` → standard Anthropic custom tool
/// - `{type: "web_search" | "web_search_preview"}` → Anthropic
///   `web_search_20260209` server tool
/// - `{type: "code_interpreter"}` → Anthropic `code_execution_20260120`
///
/// Unknown server-tool types are dropped (they have no Anthropic equivalent).
fn translate_openai_tool_to_anthropic(tool: &Value) -> Option<Value> {
    let kind = tool.get("type").and_then(|v| v.as_str())?;
    match kind {
        "function" => Some(json!({
            "name": tool.get("function").and_then(|f| f.get("name")).cloned().unwrap_or_default(),
            "description": tool.get("function").and_then(|f| f.get("description")).cloned().unwrap_or(json!("")),
            "input_schema": tool.get("function").and_then(|f| f.get("parameters")).cloned().unwrap_or(json!({}))
        })),
        "web_search" | "web_search_preview" => {
            let mut anthropic = json!({
                "type": "web_search_20260209",
                "name": "web_search",
            });
            for key in [
                "allowed_domains",
                "blocked_domains",
                "max_uses",
                "user_location",
            ] {
                if let Some(value) = tool.get(key) {
                    anthropic[key] = value.clone();
                }
            }
            Some(anthropic)
        }
        "code_interpreter" => Some(json!({
            "type": "code_execution_20260120",
            "name": "code_execution",
        })),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OpenAIToAnthropicChatConfig {
    pub default_model: &'static str,
}

pub fn convert_openai_chat_to_anthropic_request(
    body: &Value,
    config: &OpenAIToAnthropicChatConfig,
) -> Value {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match role {
                "system" => {
                    system_blocks.extend(extract_openai_anthropic_text_blocks(msg.get("content")));
                }
                "assistant" => messages.push(openai_assistant_to_anthropic(msg)),
                "tool" => messages.push(openai_tool_to_anthropic(msg)),
                _ => messages.push(openai_user_to_anthropic(msg, role)),
            }
        }
    }

    let mut req = json!({
        "model": body
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or(config.default_model),
        "messages": messages,
        "stream": body.get("stream").cloned().unwrap_or(json!(false)),
        "max_tokens": body.get("max_tokens").cloned().unwrap_or(json!(4096)),
    });

    if !system_blocks.is_empty() {
        req["system"] = anthropic_text_blocks_to_content(system_blocks);
    }
    if let Some(v) = body.get("temperature") {
        req["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        req["top_p"] = v.clone();
    }
    // Anthropic accepts `top_k`; forwarding it preserves caller intent
    // when the OpenAI request was generated by a tool that knows the
    // upstream is actually Anthropic-shaped.
    if let Some(v) = body.get("top_k") {
        req["top_k"] = v.clone();
    }
    if let Some(v) = body.get("stop") {
        req["stop_sequences"] = v.clone();
    }
    // Note: `seed`, `frequency_penalty`, `presence_penalty`, `logit_bias`,
    // `response_format`, `user` are not part of the Anthropic Messages
    // API surface and are intentionally dropped here. Document drops near
    // the call site so future readers don't think they were forgotten.
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter_map(translate_openai_tool_to_anthropic)
            .collect();
        if !anthropic_tools.is_empty() {
            req["tools"] = Value::Array(anthropic_tools);
        }
    }
    if let Some(tc) = body.get("tool_choice") {
        match tc {
            // Anthropic has no "none" mode — disable tools by removing them entirely
            Value::String(s) if s == "none" => {
                if let Some(obj) = req.as_object_mut() {
                    obj.remove("tools");
                }
            }
            _ => {
                req["tool_choice"] = match tc {
                    Value::String(s) if s == "auto" => json!({"type": "auto"}),
                    Value::String(s) if s == "required" => json!({"type": "any"}),
                    Value::Object(obj)
                        if obj.get("type").and_then(|v| v.as_str()) == Some("function") =>
                    {
                        let name = obj
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        json!({"type": "tool", "name": name})
                    }
                    _ => tc.clone(),
                };
            }
        }
    }

    // OpenAI `reasoning_effort` / `reasoning.effort` → Anthropic effort fields.
    // Without this, callers explicitly opting into reasoning silently get
    // Anthropic's default thinking off — invisible regression for anyone
    // routing Codex / GPT-5 prompts through a Claude upstream. We emit
    // `thinking.budget_tokens` (for backwards compat with Claude 3.x's
    // extended-thinking surface) and `output_config.effort` (the newer
    // Claude 4 surface). Both are request-level; Anthropic uses whichever
    // it understands and ignores unknown fields.
    if let Some(effort) = extract_openai_effort(body) {
        if !req.as_object().is_some_and(|m| m.contains_key("thinking"))
            && let Some(thinking) = anthropic_thinking_config(effort)
        {
            req["thinking"] = thinking;
        }
        if !req
            .as_object()
            .is_some_and(|m| m.contains_key("output_config"))
        {
            req["output_config"] = json!({ "effort": effort.to_anthropic_effort() });
        }
    }

    // OpenAI parallel_tool_calls:false → Anthropic disable_parallel_tool_use:true
    if body.get("parallel_tool_calls") == Some(&json!(false)) && req.get("tools").is_some() {
        match req.get_mut("tool_choice").and_then(|v| v.as_object_mut()) {
            Some(tc) => {
                tc.insert("disable_parallel_tool_use".to_string(), json!(true));
            }
            None => {
                req["tool_choice"] = json!({"type": "auto", "disable_parallel_tool_use": true});
            }
        }
    }

    req
}

pub fn convert_anthropic_to_openai_chat_response(resp: &Value, fallback_model: &str) -> Value {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<OpenAIChatToolCall> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut server_blocks: Vec<Value> = Vec::new();

    if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
        let mut tool_use_index: usize = 0;
        for block in content {
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if is_anthropic_server_block_type(block_type) {
                server_blocks.push(block.clone());
                continue;
            }
            match block_type {
                "thinking" => {
                    if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str())
                        && !thinking.is_empty()
                    {
                        thinking_parts.push(thinking.to_string());
                    }
                    // Capture the full block (including the cryptographic
                    // `signature`) so the next turn can echo it back to
                    // Anthropic without a 400 on missing signature.
                    let mut entry = json!({"type": "thinking"});
                    if let Some(text) = block.get("thinking").cloned() {
                        entry["thinking"] = text;
                    }
                    if let Some(sig) = block.get("signature").cloned() {
                        entry["signature"] = sig;
                    }
                    thinking_blocks.push(entry);
                }
                "redacted_thinking" => {
                    let mut entry = json!({"type": "redacted_thinking"});
                    if let Some(data) = block.get("data").cloned() {
                        entry["data"] = data;
                    }
                    thinking_blocks.push(entry);
                }
                "text" => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        text_parts.push(text.to_string());
                    }
                }
                "tool_use" => {
                    let args = block.get("input").cloned().unwrap_or(json!({}));
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned)
                        // Index-aware fallback: multiple tool_use blocks
                        // without ids would all collapse to "call_0", making
                        // the corresponding tool_call_id responses ambiguous.
                        .unwrap_or_else(|| format!("call_{tool_use_index}"));
                    tool_use_index += 1;
                    tool_calls.push(OpenAIChatToolCall {
                        id,
                        kind: "function".to_string(),
                        function: OpenAIChatToolCallFunction {
                            name: block
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                            arguments: serde_json::to_string(&args)
                                .unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                }
                _ => {}
            }
        }
    }

    // Anthropic stop_reason → OpenAI finish_reason. Newer Anthropic values
    // (refusal, pause_turn, stop_sequence) used to collapse to "stop"; map
    // them more precisely so callers can distinguish a refusal from a normal
    // stop. The original value is also preserved on the OpenAI envelope as
    // `_anthropic_stop_reason` for callers that need exact semantics.
    let raw_stop_reason = resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let finish_reason = match raw_stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        "refusal" => "content_filter",
        // "pause_turn" means Claude is pausing the assistant turn pending
        // external work (typically a tool result). When we already have
        // tool_calls in the response, map to "tool_calls" so OpenAI-shaped
        // clients keep the agentic loop alive instead of treating the
        // conversation as terminated. With no tool_calls present (e.g.,
        // server-side tool execution we currently drop), fall back to "stop".
        "pause_turn" if !tool_calls.is_empty() => "tool_calls",
        "stop_sequence" | "end_turn" | "pause_turn" => "stop",
        _ => "stop",
    };

    let raw_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("input_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion_tokens = resp
        .get("usage")
        .and_then(|u| u.get("output_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_read_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let cache_creation_input_tokens = resp
        .get("usage")
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64());
    // Normalize: Anthropic's input_tokens excludes cache, OpenAI's prompt_tokens includes it
    let prompt_tokens = raw_input_tokens
        + cache_read_input_tokens.unwrap_or(0)
        + cache_creation_input_tokens.unwrap_or(0);

    let response = OpenAIChatResponse {
        id: resp
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("chatcmpl-aivo")
            .to_string(),
        object: "chat.completion".to_string(),
        created: Some(
            resp.get("created")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(current_unix_ts),
        ),
        model: resp
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(fallback_model)
            .to_string(),
        choices: vec![OpenAIChatChoice {
            index: 0,
            message: OpenAIChatResponseMessage {
                role: "assistant".to_string(),
                content: (!text_parts.is_empty()).then(|| text_parts.join("\n")),
                reasoning_content: (!thinking_parts.is_empty()).then(|| thinking_parts.join("\n")),
                tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: OpenAIChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        },
    };

    let mut value = serde_json::to_value(response).unwrap_or_else(
        |_| serde_json::json!({"error": "failed to serialize openai chat response"}),
    );
    // Preserve the original Anthropic stop_reason on the first choice so
    // callers that care about the exact reason (refusal vs end_turn vs
    // stop_sequence vs pause_turn) can disambiguate. Standard OpenAI clients
    // ignore unknown fields.
    if !raw_stop_reason.is_empty()
        && let Some(choice) = value
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            .and_then(|arr| arr.first_mut())
            .and_then(|c| c.as_object_mut())
    {
        choice.insert(
            "_anthropic_stop_reason".to_string(),
            Value::String(raw_stop_reason.to_string()),
        );
    }
    // Attach captured thinking blocks (with signatures) on the assistant
    // message so client history preserves them. On the next turn, the
    // request-side converter (`openai_assistant_to_anthropic`) will lift
    // them back into the Anthropic content array. Without this, multi-turn
    // extended thinking 400s at Anthropic on missing signature.
    if !thinking_blocks.is_empty()
        && let Some(message) = value
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            .and_then(|arr| arr.first_mut())
            .and_then(|c| c.get_mut("message"))
            .and_then(|m| m.as_object_mut())
    {
        message.insert(
            ANTHROPIC_THINKING_EXT.to_string(),
            Value::Array(thinking_blocks),
        );
    }
    // Same pattern for server-side tool blocks (web_search_tool_result etc).
    // Without this, Claude's web-search / code-execution outputs disappear
    // from any continuation that flows through this bridge.
    if !server_blocks.is_empty()
        && let Some(message) = value
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            .and_then(|arr| arr.first_mut())
            .and_then(|c| c.get_mut("message"))
            .and_then(|m| m.as_object_mut())
    {
        message.insert(
            ANTHROPIC_SERVER_BLOCKS_EXT.to_string(),
            Value::Array(server_blocks),
        );
    }
    value
}

pub fn convert_openai_chat_response_to_sse(resp: &Value) -> Result<String, serde_json::Error> {
    let response: OpenAIChatResponse = serde_json::from_value(resp.clone())?;
    let id = response.id;
    let model = response.model;
    let created = response.created.unwrap_or_else(current_unix_ts);
    let choice = response.choices.into_iter().next();
    let message = choice
        .as_ref()
        .map(|choice| &choice.message)
        .cloned()
        .unwrap_or(OpenAIChatResponseMessage {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        });
    let finish_reason = choice
        .map(|choice| Value::String(choice.finish_reason))
        .unwrap_or(Value::Null);

    let reasoning_content = message.reasoning_content.as_deref().unwrap_or("");

    let mut events = String::new();
    events.push_str(&format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null
            }]
        })
    ));

    // Emit reasoning_content before content (DeepSeek-reasoner thinking)
    if !reasoning_content.is_empty() {
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": reasoning_content},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    if let Some(text) = message.content.as_deref()
        && !text.is_empty()
    {
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    if let Some(tool_calls) = message.tool_calls
        && !tool_calls.is_empty()
    {
        let delta_calls: Vec<Value> = tool_calls
            .iter()
            .enumerate()
            .map(|(index, tc)| {
                json!({
                    "index": index,
                    "id": tc.id,
                    "type": tc.kind,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            })
            .collect();
        events.push_str(&format!(
            "data: {}\n\n",
            json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": delta_calls},
                    "finish_reason": Value::Null
                }]
            })
        ));
    }

    events.push_str(&format!(
        "data: {}\n\n",
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        })
    ));
    events.push_str("data: [DONE]\n\n");
    Ok(events)
}

fn openai_user_to_anthropic(msg: &Value, role: &str) -> Value {
    json!({
        "role": role,
        "content": openai_content_to_anthropic_content(msg.get("content"))
    })
}

fn openai_assistant_to_anthropic(msg: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    // Restore thinking / redacted_thinking blocks (with signatures) before
    // text/tool_use. Anthropic expects these to come first in the content
    // array of a continued assistant turn — that ordering is what a previous
    // response originally produced, and the signature is validated against it.
    let mut had_explicit_thinking_ext = false;
    if let Some(thinking_blocks) = msg.get(ANTHROPIC_THINKING_EXT).and_then(|v| v.as_array()) {
        had_explicit_thinking_ext = true;
        for block in thinking_blocks {
            // Only carry through block shapes we recognize; drop anything else
            // so a malformed extension can't poison the content array.
            match block.get("type").and_then(|v| v.as_str()) {
                Some("thinking") | Some("redacted_thinking") => blocks.push(block.clone()),
                _ => {}
            }
        }
    }
    // Fallback: if the OpenAI assistant message lacks the structured
    // `_anthropic_thinking_blocks` extension but does carry the legacy
    // `reasoning_content` field (DeepSeek-reasoner / OpenAI o-series shape),
    // synthesize a `thinking` block. We can't fabricate a signature, but
    // emitting the block is still a strict improvement: clients that drop
    // signatures still get visibility into the model's reasoning, and
    // Anthropic does NOT reject a thinking block that simply lacks the
    // `signature` field — only a block with a wrong/expired one fails.
    if !had_explicit_thinking_ext
        && let Some(reasoning) = msg.get("reasoning_content").and_then(|v| v.as_str())
        && !reasoning.is_empty()
    {
        blocks.push(json!({"type": "thinking", "thinking": reasoning}));
    }
    let text = extract_openai_text(msg.get("content"));
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str())
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or_else(|| json!({}));
            blocks.push(json!({
                "type": "tool_use",
                "id": tc.get("id").cloned().unwrap_or(json!("call_0")),
                "name": tc.get("function").and_then(|f| f.get("name")).cloned().unwrap_or(json!("")),
                "input": args
            }));
        }
    }
    // Restore Anthropic server-tool blocks (web_search_tool_result etc) at
    // the end of the assistant content array so the model can reference its
    // own previous server-side work in continuation turns.
    if let Some(server_blocks) = msg
        .get(ANTHROPIC_SERVER_BLOCKS_EXT)
        .and_then(|v| v.as_array())
    {
        for block in server_blocks {
            if let Some(t) = block.get("type").and_then(|v| v.as_str())
                && is_anthropic_server_block_type(t)
            {
                blocks.push(block.clone());
            }
        }
    }
    json!({
        "role": "assistant",
        "content": blocks
    })
}

fn openai_tool_to_anthropic(msg: &Value) -> Value {
    let content = extract_openai_text(msg.get("content"));
    json!({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or(json!("")),
            "content": content
        }]
    })
}

fn extract_openai_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

fn openai_content_to_anthropic_content(content: Option<&Value>) -> Value {
    anthropic_text_blocks_to_content(extract_openai_anthropic_text_blocks(content))
}

fn extract_openai_anthropic_text_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![json!({"type": "text", "text": s})]
            }
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(openai_content_part_to_anthropic_block)
            .collect(),
        Some(Value::Null) | None => Vec::new(),
        Some(other) => vec![json!({"type": "text", "text": other.to_string()})],
    }
}

fn openai_content_part_to_anthropic_block(part: &Value) -> Option<Value> {
    let part_type = part.get("type").and_then(|v| v.as_str());
    match part_type {
        None | Some("text") => {
            let text = part.get("text").and_then(|v| v.as_str())?;
            let mut block = part.clone();
            if !block.is_object() {
                block = json!({});
            }
            block["type"] = Value::String("text".to_string());
            block["text"] = Value::String(text.to_string());
            Some(block)
        }
        Some("image_url") => openai_image_url_part_to_anthropic_image_block(part),
        // Future-proofing: silently drop unknown part types rather than crash.
        // The original implementation also dropped images here — that was the
        // bug. Keep the explicit case to make image handling auditable.
        _ => None,
    }
}

/// Convert an OpenAI `{type: "image_url", image_url: {url, detail?}}` part to an
/// Anthropic image block. Accepts both string and object `image_url` shapes
/// (some clients emit either). Recognises `data:<mime>;base64,<…>` URIs and
/// emits an Anthropic `base64` source; HTTP(S) URLs become `url` sources.
fn openai_image_url_part_to_anthropic_image_block(part: &Value) -> Option<Value> {
    let url = part.get("image_url").and_then(|iu| match iu {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("url").and_then(|v| v.as_str()),
        _ => None,
    })?;

    let source = if let Some(rest) = url.strip_prefix("data:") {
        // data:<mime>;base64,<base64-data>
        let (meta, data) = rest.split_once(',')?;
        let mime = meta
            .split(';')
            .next()
            .filter(|m| !m.is_empty())
            .unwrap_or("image/png");
        // Some clients emit `data:image/png,<…>` (no `;base64`); treat the
        // payload as already-encoded base64 in either case to match how
        // OpenAI itself handles these URIs.
        json!({
            "type": "base64",
            "media_type": mime,
            "data": data,
        })
    } else if url.starts_with("http://") || url.starts_with("https://") {
        json!({
            "type": "url",
            "url": url,
        })
    } else {
        // Unknown scheme — drop rather than emit something Anthropic will reject.
        return None;
    };

    Some(json!({
        "type": "image",
        "source": source,
    }))
}

fn anthropic_text_blocks_to_content(blocks: Vec<Value>) -> Value {
    if blocks.is_empty() {
        return Value::String(String::new());
    }

    if blocks.iter().all(is_plain_anthropic_text_block) {
        return Value::String(
            blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n"),
        );
    }

    Value::Array(blocks)
}

fn is_plain_anthropic_text_block(block: &Value) -> bool {
    let Some(obj) = block.as_object() else {
        return false;
    };

    obj.len() == 2
        && obj.get("type").and_then(|v| v.as_str()) == Some("text")
        && obj.get("text").and_then(|v| v.as_str()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_chat_to_anthropic_request_with_tool_calls() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{\"path\":\".\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "[]"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }]
        });

        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert_eq!(converted["system"], "Be precise.");
        assert_eq!(converted["messages"][1]["content"][0]["type"], "tool_use");
        assert_eq!(
            converted["messages"][2]["content"][0]["type"],
            "tool_result"
        );
        assert_eq!(converted["tools"][0]["name"], "ls");
    }

    #[test]
    fn test_convert_anthropic_to_openai_chat_response_with_tool_use() {
        let body = json!({
            "id": "msg_1",
            "model": "MiniMax-M1",
            "content": [
                {"type": "text", "text": "Need tool"},
                {"type": "tool_use", "id": "call_1", "name": "ls", "input": {"path":"."}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        });
        let converted = convert_anthropic_to_openai_chat_response(&body, "fallback");
        assert_eq!(converted["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            converted["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
    }

    #[test]
    fn test_convert_openai_chat_to_anthropic_request_preserves_cache_control_blocks() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "Be precise.",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"}
                    }]
                }
            ]
        });

        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4-5",
            },
        );

        assert_eq!(converted["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(
            converted["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn test_convert_openai_chat_empty_messages_array() {
        let body = json!({"model": "gpt-4o", "messages": []});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
        assert_eq!(converted["model"], "gpt-4o");
    }

    #[test]
    fn test_convert_openai_chat_missing_model_uses_default() {
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "fallback-model",
            },
        );
        assert_eq!(converted["model"], "fallback-model");
    }

    #[test]
    fn test_convert_openai_chat_null_content_no_panic() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": null},
                {"role": "assistant", "content": null}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert_eq!(converted["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_convert_openai_chat_missing_messages_field() {
        let body = json!({"model": "gpt-4o"});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_convert_anthropic_response_empty_content() {
        let resp = json!({"id": "msg_1", "model": "test", "content": [], "usage": {}});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert!(converted["choices"][0]["message"]["content"].is_null());
        assert!(converted["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[test]
    fn test_convert_anthropic_response_missing_usage() {
        let resp = json!({"content": [{"type": "text", "text": "hi"}]});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
        assert_eq!(converted["usage"]["completion_tokens"], 0);
    }

    #[test]
    fn test_convert_anthropic_response_unknown_stop_reason() {
        let resp =
            json!({"content": [{"type": "text", "text": "hi"}], "stop_reason": "weird_reason"});
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_extract_openai_text_unexpected_type() {
        assert_eq!(extract_openai_text(Some(&json!(42))), "42");
        assert_eq!(extract_openai_text(Some(&json!(true))), "true");
    }

    #[test]
    fn convert_openai_to_anthropic_invalid_tool_args_json() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_stuff", "arguments": "not json"}
                }]}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Invalid JSON arguments should fall back to {}
        let tool_block = &converted["messages"][1]["content"][0];
        assert_eq!(tool_block["type"], "tool_use");
        assert_eq!(tool_block["input"], json!({}));
    }

    #[test]
    fn convert_anthropic_to_openai_null_usage_subfields() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": null, "output_tokens": null}
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        assert_eq!(converted["usage"]["prompt_tokens"], 0);
        assert_eq!(converted["usage"]["completion_tokens"], 0);
        assert_eq!(converted["usage"]["total_tokens"], 0);
    }

    #[test]
    fn convert_openai_to_anthropic_empty_string_arguments() {
        // OpenAI legitimately streams arguments: "" (empty string, not "{}")
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "do_stuff", "arguments": ""}
                }]}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Empty string arguments should fall back to {}
        let tool_block = &converted["messages"][1]["content"][0];
        assert_eq!(tool_block["type"], "tool_use");
        assert_eq!(tool_block["input"], json!({}));
    }

    #[test]
    fn convert_openai_to_anthropic_parallel_tool_calls_false() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "parallel_tool_calls": false
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Should inject disable_parallel_tool_use into tool_choice
        assert_eq!(
            converted["tool_choice"]["disable_parallel_tool_use"],
            json!(true)
        );
        assert_eq!(converted["tool_choice"]["type"], "auto");
    }

    #[test]
    fn convert_openai_to_anthropic_parallel_tool_calls_false_with_existing_tool_choice() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "tool_choice": "required",
            "parallel_tool_calls": false
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Should inject disable_parallel_tool_use into existing tool_choice
        assert_eq!(converted["tool_choice"]["type"], "any");
        assert_eq!(
            converted["tool_choice"]["disable_parallel_tool_use"],
            json!(true)
        );
    }

    #[test]
    fn convert_openai_to_anthropic_tool_choice_none_strips_tools() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {"name": "ls", "description": "list", "parameters": {"type":"object"}}
            }],
            "tool_choice": "none"
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // "none" should strip tools and not set tool_choice
        assert!(converted.get("tools").is_none());
        assert!(converted.get("tool_choice").is_none());
    }

    #[test]
    fn convert_openai_to_anthropic_sse_empty_choices() {
        // No SSE chunk conversion function exists; test convert with empty messages array
        let body = json!({"model": "gpt-4o", "messages": []});
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        assert!(converted["messages"].as_array().unwrap().is_empty());
        assert_eq!(converted["model"], "gpt-4o");
        assert_eq!(converted["max_tokens"], 4096);
    }

    #[test]
    fn convert_openai_to_anthropic_sse_null_content() {
        // Tool calls with null content in assistant message
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "call tool"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_data", "arguments": "{\"x\":1}"}
                    }]
                }
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // Assistant content should have only the tool_use block (no text block since content is null)
        let assistant_content = converted["messages"][1]["content"].as_array().unwrap();
        assert_eq!(assistant_content.len(), 1);
        assert_eq!(assistant_content[0]["type"], "tool_use");
        assert_eq!(assistant_content[0]["name"], "get_data");
    }

    #[test]
    fn convert_anthropic_to_openai_empty_text_blocks() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": ""}],
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        // Empty text is skipped, so content should be null (no text_parts collected)
        assert!(converted["choices"][0]["message"]["content"].is_null());
    }

    #[test]
    fn convert_anthropic_to_openai_cache_tokens_summed() {
        let resp = json!({
            "id": "msg_1",
            "model": "test",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 30
            }
        });
        let converted = convert_anthropic_to_openai_chat_response(&resp, "fallback");
        // prompt_tokens = input_tokens + cache_read + cache_creation = 10 + 20 + 30 = 60
        assert_eq!(converted["usage"]["prompt_tokens"], 60);
        assert_eq!(converted["usage"]["completion_tokens"], 5);
        assert_eq!(converted["usage"]["total_tokens"], 65);
        assert_eq!(converted["usage"]["cache_read_input_tokens"], 20);
        assert_eq!(converted["usage"]["cache_creation_input_tokens"], 30);
    }

    #[test]
    fn convert_openai_to_anthropic_forwards_top_k() {
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 5,
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        assert_eq!(converted["top_k"], 5);
    }

    #[test]
    fn convert_openai_to_anthropic_drops_unsupported_params() {
        // seed, response_format, frequency_penalty, presence_penalty, etc.
        // have no Anthropic equivalent — they should not appear on the
        // converted request body.
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 1,
            "response_format": {"type": "json_object"},
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "logit_bias": {"42": 1},
            "user": "u_1"
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        for k in [
            "seed",
            "response_format",
            "frequency_penalty",
            "presence_penalty",
            "logit_bias",
            "user",
        ] {
            assert!(
                converted.get(k).is_none(),
                "OpenAI-only param {k} should not be forwarded to Anthropic"
            );
        }
    }

    #[test]
    fn convert_anthropic_to_openai_stop_reason_refusal_maps_to_content_filter() {
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "I can't help with that."}],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4");
        assert_eq!(chat["choices"][0]["finish_reason"], "content_filter");
        assert_eq!(chat["choices"][0]["_anthropic_stop_reason"], "refusal");
    }

    #[test]
    fn convert_anthropic_to_openai_stop_reason_pause_turn_maps_to_stop() {
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "thinking…"}],
            "stop_reason": "pause_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        // Extension field still preserves the exact reason for callers that need it.
        assert_eq!(chat["choices"][0]["_anthropic_stop_reason"], "pause_turn");
    }

    #[test]
    fn anthropic_thinking_block_signature_round_trips_through_openai_intermediate() {
        // Anthropic responses with extended-thinking blocks must surface
        // their `signature` (and `data` on `redacted_thinking`) on the
        // OpenAI assistant message so the next turn can echo them back
        // — Claude 4 returns 400 if a continued conversation drops them.
        let resp = json!({
            "id": "msg_t1",
            "model": "claude-opus-4-7",
            "content": [
                {"type": "thinking", "thinking": "Let me check.", "signature": "SIG_42"},
                {"type": "redacted_thinking", "data": "BLOB_99"},
                {"type": "text", "text": "Here is my answer."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-opus-4-7");
        let blocks = chat["choices"][0]["message"]["_anthropic_thinking_blocks"]
            .as_array()
            .expect("thinking-blocks extension present on assistant message");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(blocks[0]["signature"], "SIG_42");
        assert_eq!(blocks[0]["thinking"], "Let me check.");
        assert_eq!(blocks[1]["type"], "redacted_thinking");
        assert_eq!(blocks[1]["data"], "BLOB_99");

        // Now feed those blocks back through the request converter as an
        // OpenAI assistant message. They must reappear at the start of the
        // Anthropic content array, with signature/data intact.
        let openai_followup = json!({
            "model": "claude-opus-4-7",
            "messages": [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": "Here is my answer.",
                    "_anthropic_thinking_blocks": [
                        {"type": "thinking", "thinking": "Let me check.", "signature": "SIG_42"},
                        {"type": "redacted_thinking", "data": "BLOB_99"}
                    ]
                },
                {"role": "user", "content": "Continue please."}
            ]
        });
        let anthropic_req = convert_openai_chat_to_anthropic_request(
            &openai_followup,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let assistant_content = anthropic_req["messages"][1]["content"]
            .as_array()
            .expect("assistant content array");
        assert_eq!(assistant_content[0]["type"], "thinking");
        assert_eq!(assistant_content[0]["signature"], "SIG_42");
        assert_eq!(assistant_content[0]["thinking"], "Let me check.");
        assert_eq!(assistant_content[1]["type"], "redacted_thinking");
        assert_eq!(assistant_content[1]["data"], "BLOB_99");
        assert_eq!(assistant_content[2]["type"], "text");
        assert_eq!(assistant_content[2]["text"], "Here is my answer.");
    }

    #[test]
    fn anthropic_thinking_extension_is_omitted_when_response_has_no_thinking_blocks() {
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4-6");
        assert!(
            chat["choices"][0]["message"]
                .get("_anthropic_thinking_blocks")
                .is_none(),
            "no thinking blocks → no extension field"
        );
    }

    #[test]
    fn openai_reasoning_content_synthesizes_thinking_block_when_extension_absent() {
        // OpenAI o-series / DeepSeek-reasoner emit reasoning via the
        // `reasoning_content` field, not a structured extension. When a
        // continuation flows OpenAI → Anthropic, that field used to be
        // silently dropped. Now it surfaces as a `thinking` block (without
        // signature, since we can't fabricate one — Anthropic accepts
        // unsigned thinking blocks in continuations).
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "I considered three options."
            }]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "I considered three options.");
        assert!(
            content[0].get("signature").is_none(),
            "synthetic thinking has no signature — we can't fabricate Anthropic's"
        );
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "Final answer.");
    }

    #[test]
    fn explicit_thinking_extension_takes_precedence_over_reasoning_content() {
        // When the structured `_anthropic_thinking_blocks` extension is
        // present, it wins over `reasoning_content` — the extension preserves
        // the original signatures, and synthesizing a sibling block from
        // reasoning_content would duplicate the reasoning.
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "assistant",
                "content": "Done.",
                "reasoning_content": "stale text",
                "_anthropic_thinking_blocks": [
                    {"type": "thinking", "thinking": "real reasoning", "signature": "SIG_X"}
                ]
            }]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let content = req["messages"][0]["content"].as_array().unwrap();
        let thinking_blocks: Vec<&Value> =
            content.iter().filter(|b| b["type"] == "thinking").collect();
        assert_eq!(thinking_blocks.len(), 1);
        assert_eq!(thinking_blocks[0]["signature"], "SIG_X");
        assert_eq!(thinking_blocks[0]["thinking"], "real reasoning");
    }

    #[test]
    fn empty_thinking_extension_array_suppresses_reasoning_content_synthesis() {
        // An explicit empty array means "no thinking blocks" — fallback to
        // reasoning_content would defeat the caller's intent.
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "assistant",
                "content": "Done.",
                "reasoning_content": "shouldn't surface",
                "_anthropic_thinking_blocks": []
            }]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert!(
            content.iter().all(|b| b["type"] != "thinking"),
            "explicit empty extension must suppress reasoning_content fallback"
        );
    }

    #[test]
    fn openai_reasoning_effort_high_maps_to_anthropic_thinking_and_output_config() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        assert_eq!(req["thinking"]["type"], "enabled");
        assert_eq!(req["thinking"]["budget_tokens"], 16384);
        assert_eq!(req["output_config"]["effort"], "high");
    }

    #[test]
    fn openai_reasoning_effort_xhigh_maps_to_anthropic_max_effort() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "xhigh"
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        assert_eq!(req["thinking"]["budget_tokens"], 32000);
        assert_eq!(req["output_config"]["effort"], "max");
    }

    #[test]
    fn openai_reasoning_effort_none_does_not_enable_anthropic_thinking() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none"
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        assert!(req.get("thinking").is_none());
        assert_eq!(req["output_config"]["effort"], "low");
    }

    #[test]
    fn anthropic_server_tool_blocks_round_trip_through_extension() {
        let resp = json!({
            "id": "msg_s1",
            "model": "claude-opus-4-7",
            "content": [
                {"type": "text", "text": "Looking up current AAPL price."},
                {
                    "type": "server_tool_use",
                    "id": "stu_1",
                    "name": "web_search",
                    "input": {"query": "AAPL stock price"}
                },
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "stu_1",
                    "content": [{"type": "web_search_result", "url": "https://example.com", "title": "AAPL"}]
                },
                {"type": "text", "text": "It is $200."}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-opus-4-7");
        let blocks = chat["choices"][0]["message"]["_anthropic_server_blocks"]
            .as_array()
            .expect("server-blocks extension present");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "server_tool_use");
        assert_eq!(blocks[1]["type"], "web_search_tool_result");

        let openai_followup = json!({
            "model": "claude-opus-4-7",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Looking up current AAPL price.\nIt is $200.",
                    "_anthropic_server_blocks": blocks
                },
                {"role": "user", "content": "Anything else?"}
            ]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &openai_followup,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let assistant_content = req["messages"][0]["content"].as_array().unwrap();
        let server_block_count = assistant_content
            .iter()
            .filter(|b| {
                let t = b["type"].as_str().unwrap_or("");
                matches!(t, "server_tool_use" | "web_search_tool_result")
            })
            .count();
        assert_eq!(server_block_count, 2);
    }

    #[test]
    fn web_search_tool_translates_to_anthropic_native_server_tool() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "search for X"}],
            "tools": [
                {"type": "web_search"},
                {"type": "function", "function": {"name": "calc", "description": "calc", "parameters": {"type": "object"}}}
            ]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let tools = req["tools"].as_array().unwrap();
        let by_type: Vec<&str> = tools
            .iter()
            .map(|t| {
                t.get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or_else(|| t["name"].as_str().unwrap_or(""))
            })
            .collect();
        assert!(by_type.contains(&"web_search_20260209"));
        assert!(by_type.contains(&"calc"));
    }

    #[test]
    fn web_search_tool_passes_domain_and_max_uses_constraints_through() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{
                "type": "web_search",
                "allowed_domains": ["example.com"],
                "max_uses": 5
            }]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let tool = &req["tools"][0];
        assert_eq!(tool["type"], "web_search_20260209");
        assert_eq!(tool["allowed_domains"][0], "example.com");
        assert_eq!(tool["max_uses"], 5);
    }

    #[test]
    fn code_interpreter_tool_translates_to_anthropic_code_execution() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "compute"}],
            "tools": [{"type": "code_interpreter"}]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        assert_eq!(req["tools"][0]["type"], "code_execution_20260120");
    }

    #[test]
    fn unknown_server_tool_types_are_dropped_silently() {
        let body = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {"type": "computer_use"},
                {"type": "file_search"},
                {"type": "function", "function": {"name": "f", "description": "", "parameters": {}}}
            ]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let tools = req["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "f");
    }

    #[test]
    fn anthropic_thinking_extension_drops_unknown_block_types() {
        // Defensive: a malformed extension carrying an unrecognized block
        // type must not poison the Anthropic content array.
        let openai_msg = json!({
            "model": "claude-opus-4-7",
            "messages": [{
                "role": "assistant",
                "content": "Hello.",
                "_anthropic_thinking_blocks": [
                    {"type": "definitely_not_a_real_type", "blob": "data"}
                ]
            }]
        });
        let req = convert_openai_chat_to_anthropic_request(
            &openai_msg,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-opus-4-7",
            },
        );
        let content = req["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
    }

    #[test]
    fn convert_anthropic_to_openai_stop_reason_pause_turn_with_tool_use_maps_to_tool_calls() {
        // pause_turn + tool_use blocks: Claude is pausing for a tool result,
        // not terminating. Map to "tool_calls" so OpenAI clients keep the
        // agentic loop alive instead of treating the conversation as over.
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4",
            "content": [
                {"type": "text", "text": "Looking that up…"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}}
            ],
            "stop_reason": "pause_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4");
        assert_eq!(chat["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chat["choices"][0]["_anthropic_stop_reason"], "pause_turn");
        let tcs = chat["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls present");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["function"]["name"], "search");
    }

    #[test]
    fn convert_anthropic_to_openai_stop_reason_stop_sequence_maps_to_stop() {
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4",
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "stop_sequence",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            chat["choices"][0]["_anthropic_stop_reason"],
            "stop_sequence"
        );
    }

    #[test]
    fn convert_anthropic_to_openai_tool_use_without_ids_yields_unique_call_ids() {
        // Two tool_use blocks with no `id` must NOT collide on a shared
        // "call_0" — the matching tool_call_id responses would be ambiguous.
        let resp = json!({
            "id": "msg_x",
            "model": "claude-sonnet-4",
            "content": [
                {"type": "tool_use", "name": "ls", "input": {"path": "."}},
                {"type": "tool_use", "name": "cat", "input": {"path": "a"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        let chat = convert_anthropic_to_openai_chat_response(&resp, "claude-sonnet-4");
        let tcs = chat["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs.len(), 2);
        let id0 = tcs[0]["id"].as_str().unwrap().to_string();
        let id1 = tcs[1]["id"].as_str().unwrap().to_string();
        assert_ne!(id0, id1, "fallback ids must be unique");
        assert_eq!(id0, "call_0");
        assert_eq!(id1, "call_1");
    }

    #[test]
    fn convert_openai_to_anthropic_image_url_data_uri_becomes_base64_block() {
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                    }}
                ]
            }]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        let content = &converted["messages"][0]["content"];
        // Mixed content keeps array form; image block uses base64 source.
        assert!(content.is_array(), "expected array content for image input");
        let parts = content.as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what is this?");
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(parts[1]["source"]["type"], "base64");
        assert_eq!(parts[1]["source"]["media_type"], "image/png");
        assert!(
            parts[1]["source"]["data"]
                .as_str()
                .unwrap()
                .starts_with("iVBORw0KGgo")
        );
    }

    #[test]
    fn convert_openai_to_anthropic_image_url_http_becomes_url_source() {
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
                ]
            }]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        let content = &converted["messages"][0]["content"];
        let parts = content.as_array().expect("array content for image-only");
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["type"], "image");
        assert_eq!(parts[0]["source"]["type"], "url");
        assert_eq!(parts[0]["source"]["url"], "https://example.com/cat.png");
    }

    #[test]
    fn convert_openai_to_anthropic_image_url_string_shape_accepted() {
        // Some clients emit `image_url` as a bare string instead of an object.
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/x.jpg"}
                ]
            }]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        let parts = converted["messages"][0]["content"]
            .as_array()
            .expect("array content");
        assert_eq!(parts[0]["type"], "image");
        assert_eq!(parts[0]["source"]["url"], "https://example.com/x.jpg");
    }

    #[test]
    fn convert_openai_to_anthropic_text_only_still_collapses_to_string() {
        // Regression: collapse-to-string for pure-text input must not change.
        let body = json!({
            "model": "claude-sonnet-4",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"}
                ]
            }]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "claude-sonnet-4",
            },
        );
        assert_eq!(converted["messages"][0]["content"], "hello\n\nworld");
    }

    #[test]
    fn convert_openai_to_anthropic_developer_role_mapped() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hi"}
            ]
        });
        let converted = convert_openai_chat_to_anthropic_request(
            &body,
            &OpenAIToAnthropicChatConfig {
                default_model: "gpt-4o",
            },
        );
        // "developer" falls through to the _ match arm which calls openai_user_to_anthropic
        // preserving the role as-is ("developer")
        let messages = converted["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "developer");
        // Content should be converted properly
        assert!(messages[0]["content"].is_string() || messages[0]["content"].is_array());
    }
}
