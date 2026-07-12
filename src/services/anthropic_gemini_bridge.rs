//! Direct Anthropic ↔ Gemini converters for both registry edges. One hop
//! (vs. the two-hop Chat pivot) preserves the extended-thinking budget and
//! round-trips the thought signature; forward streams natively, reverse fakes one SSE event.

use std::collections::{HashMap, VecDeque};

use serde_json::{Value, json};

use crate::services::bridge_defaults::BRIDGE_DEFAULT_ANTHROPIC_MAX_TOKENS;
use crate::services::effort::{
    CanonicalEffort, extract_anthropic_effort, gemini_thinking_config, gemini_uses_thinking_level,
};
use crate::services::http_utils::{self, SseLineBuffer, sse_event};
use crate::services::openai_gemini_bridge::{
    SKIP_THOUGHT_SIGNATURE_PLACEHOLDER, extract_part_call_id, pop_pending_tool_call_id,
    queue_pending_tool_call_id, sanitize_schema_for_gemini, synthesize_tool_call_id,
    take_pending_tool_call_id, uniquify_tool_call_id,
};

pub struct AnthropicToGeminiConfig<'a> {
    /// Fallback model; decides the thinking surface (numeric budget vs. `thinking_level`).
    pub default_model: &'a str,
}

pub struct GeminiToAnthropicConfig<'a> {
    pub model: &'a str,
}

// ── request: Anthropic /v1/messages → Gemini generateContent ──────────────

pub fn convert_anthropic_to_gemini_request(
    body: &Value,
    config: &AnthropicToGeminiConfig,
) -> Value {
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(config.default_model);

    // Gemini matches functionResponse by name; Anthropic tool_result carries only id — learn id→name.
    let mut tool_names_by_id: HashMap<String, String> = HashMap::new();
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            if let Some(Value::Array(blocks)) = msg.get("content").map(|c| c.to_owned()) {
                for block in &blocks {
                    if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                        let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        if !id.is_empty() {
                            tool_names_by_id.insert(id.to_string(), name.to_string());
                        }
                    }
                }
            }
        }
    }

    let mut contents: Vec<Value> = Vec::new();
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let gemini_role = if role == "assistant" { "model" } else { "user" };
            let parts = match msg.get("content") {
                Some(Value::String(text)) if !text.is_empty() => vec![json!({ "text": text })],
                Some(Value::String(_)) => Vec::new(),
                Some(Value::Array(blocks)) => {
                    anthropic_blocks_to_gemini_parts(blocks, &tool_names_by_id)
                }
                _ => Vec::new(),
            };
            if !parts.is_empty() {
                contents.push(json!({ "role": gemini_role, "parts": parts }));
            }
        }
    }
    if contents.is_empty() {
        contents.push(json!({ "role": "user", "parts": [{ "text": "" }] }));
    }

    let mut request = json!({ "contents": contents });

    let system_text = extract_anthropic_system_text(body.get("system"));
    if !system_text.is_empty() {
        request["systemInstruction"] = json!({ "parts": [{ "text": system_text }] });
    }

    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let declarations: Vec<Value> = tools
            .iter()
            .filter(|t| !is_anthropic_server_tool(t))
            .map(|tool| {
                json!({
                    "name": tool.get("name").cloned().unwrap_or_default(),
                    "description": tool.get("description").cloned().unwrap_or(json!("")),
                    "parameters": sanitize_schema_for_gemini(
                        tool.get("input_schema").unwrap_or(&json!({"type": "object", "properties": {}}))
                    )
                })
            })
            .collect();
        if !declarations.is_empty() {
            request["tools"] = json!([{ "functionDeclarations": declarations }]);
        }
    }

    if let Some(tc) = body.get("tool_choice") {
        let cfg = match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => Some(json!({ "mode": "AUTO" })),
            Some("any") => Some(json!({ "mode": "ANY" })),
            Some("none") => Some(json!({ "mode": "NONE" })),
            Some("tool") => tc
                .get("name")
                .and_then(|n| n.as_str())
                .map(|name| json!({ "mode": "ANY", "allowedFunctionNames": [name] })),
            _ => None,
        };
        if let Some(cfg) = cfg {
            request["toolConfig"] = json!({ "functionCallingConfig": cfg });
        }
    }

    let mut generation = serde_json::Map::new();
    if let Some(v) = body.get("max_tokens") {
        generation.insert("maxOutputTokens".to_string(), v.clone());
    }
    if !crate::services::model_metadata::rejects_temperature(model) {
        if let Some(v) = body.get("temperature") {
            generation.insert("temperature".to_string(), v.clone());
        }
        if let Some(v) = body.get("top_p") {
            generation.insert("topP".to_string(), v.clone());
        }
        if let Some(v) = body.get("top_k") {
            generation.insert("topK".to_string(), v.clone());
        }
    }
    if let Some(v) = body.get("stop_sequences") {
        generation.insert("stopSequences".to_string(), v.clone());
    }
    // Extended thinking → thinkingConfig (only the direct edge preserves it).
    if let Some(effort) = extract_anthropic_effort(body)
        && let Some(cfg) = gemini_thinking_config(effort, gemini_uses_thinking_level(model))
    {
        generation.insert("thinkingConfig".to_string(), cfg);
    }
    if !generation.is_empty() {
        request["generationConfig"] = Value::Object(generation);
    }

    request
}

fn extract_anthropic_system_text(system: Option<&Value>) -> String {
    match system {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

fn is_anthropic_server_tool(tool: &Value) -> bool {
    crate::services::anthropic_chat_request::is_anthropic_server_tool(tool)
}

/// Thinking-block signature rides on the first `functionCall` part as `thoughtSignature`.
fn anthropic_blocks_to_gemini_parts(
    blocks: &[Value],
    tool_names_by_id: &HashMap<String, String>,
) -> Vec<Value> {
    let thinking_signature = blocks.iter().find_map(|b| {
        (b.get("type").and_then(|t| t.as_str()) == Some("thinking"))
            .then(|| b.get("signature").and_then(|s| s.as_str()))
            .flatten()
            .map(str::to_string)
    });

    let mut parts: Vec<Value> = Vec::new();
    let mut function_call_seen = false;
    for block in blocks {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str())
                    && !text.is_empty()
                {
                    parts.push(json!({ "text": text }));
                }
            }
            Some("image") => {
                if let Some(part) = anthropic_image_to_gemini_part(block) {
                    parts.push(part);
                }
            }
            Some("tool_use") => {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args = block.get("input").cloned().unwrap_or(json!({}));
                let mut part = json!({ "functionCall": { "id": id, "name": name, "args": args } });
                if !function_call_seen {
                    part["thoughtSignature"] = Value::String(
                        thinking_signature
                            .clone()
                            .unwrap_or_else(|| SKIP_THOUGHT_SIGNATURE_PLACEHOLDER.to_string()),
                    );
                    function_call_seen = true;
                }
                parts.push(part);
            }
            Some("tool_result") => {
                let id = block
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let name = tool_names_by_id
                    .get(id)
                    .filter(|n| !n.is_empty())
                    .cloned()
                    .unwrap_or_else(|| id.to_string());
                let response = anthropic_tool_result_to_gemini_response(block.get("content"));
                parts.push(json!({
                    "functionResponse": { "id": id, "name": name, "response": response }
                }));
            }
            // thinking/redacted_thinking: no Gemini payload; signature captured above.
            _ => {}
        }
    }
    parts
}

fn anthropic_image_to_gemini_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    match source.get("type").and_then(|t| t.as_str()) {
        Some("base64") => {
            let mime = source.get("media_type").and_then(|v| v.as_str())?;
            let data = source.get("data").and_then(|v| v.as_str())?;
            Some(json!({ "inlineData": { "mimeType": mime, "data": data } }))
        }
        Some("url") => {
            let uri = source.get("url").and_then(|v| v.as_str())?;
            Some(json!({ "fileData": { "fileUri": uri } }))
        }
        _ => None,
    }
}

fn anthropic_tool_result_to_gemini_response(content: Option<&Value>) -> Value {
    let text = match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    serde_json::from_str::<Value>(&text)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({ "content": text }))
}

// ── response: Gemini generateContent → Anthropic message ──────────────────

pub fn convert_gemini_to_anthropic_response(
    resp: &Value,
    config: &GeminiToAnthropicConfig,
) -> Value {
    let candidate = resp
        .get("candidates")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let parts = candidate
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut thinking_text = String::new();
    let mut thinking_signature: Option<String> = None;
    let mut body_text = String::new();
    let mut tool_uses: Vec<Value> = Vec::new();
    for (index, part) in parts.iter().enumerate() {
        if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str())
            && !sig.is_empty()
            && thinking_signature.is_none()
        {
            thinking_signature = Some(sig.to_string());
        }
        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
            let is_thought = part
                .get("thought")
                .and_then(|t| t.as_bool())
                .unwrap_or(false);
            let target = if is_thought {
                &mut thinking_text
            } else {
                &mut body_text
            };
            if !target.is_empty() {
                target.push('\n');
            }
            target.push_str(text);
        }
        if let Some(fc) = part.get("functionCall") {
            let id = fc
                .get("id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("toolu_{index}"));
            tool_uses.push(json!({
                "type": "tool_use",
                "id": id,
                "name": fc.get("name").cloned().unwrap_or(json!("")),
                "input": fc.get("args").cloned().unwrap_or(json!({})),
            }));
        }
    }

    // Anthropic content order: thinking, then text, then tool_use.
    let mut content: Vec<Value> = Vec::new();
    if !thinking_text.is_empty() {
        let mut block = json!({ "type": "thinking", "thinking": thinking_text });
        if let Some(sig) = &thinking_signature {
            block["signature"] = json!(sig);
        }
        content.push(block);
    }
    if !body_text.is_empty() {
        content.push(json!({ "type": "text", "text": body_text }));
    }
    content.extend(tool_uses.iter().cloned());
    if content.is_empty() {
        content.push(json!({ "type": "text", "text": "" }));
    }

    let raw_finish = candidate
        .get("finishReason")
        .and_then(|v| v.as_str())
        .unwrap_or("STOP");
    let stop_reason = gemini_finish_to_anthropic(raw_finish, !tool_uses.is_empty());

    json!({
        "id": resp.get("responseId").cloned().unwrap_or_else(|| json!(http_utils::gen_id("msg"))),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": config.model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": gemini_usage_to_anthropic(resp.get("usageMetadata")),
    })
}

fn gemini_finish_to_anthropic(reason: &str, has_tool_use: bool) -> &'static str {
    match reason {
        "MAX_TOKENS" => "max_tokens",
        _ if has_tool_use => "tool_use",
        _ => "end_turn",
    }
}

fn gemini_usage_to_anthropic(usage: Option<&Value>) -> Value {
    let n = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    // Anthropic output_tokens includes thinking; Gemini splits candidates/thoughts — fold them.
    let output = n("candidatesTokenCount").saturating_add(n("thoughtsTokenCount"));
    let mut out = json!({
        "input_tokens": n("promptTokenCount"),
        "output_tokens": output,
    });
    let cached = usage
        .and_then(|u| u.get("cachedContentTokenCount"))
        .and_then(|v| v.as_u64());
    if let Some(v) = cached {
        out["cache_read_input_tokens"] = json!(v);
    }
    out
}

// ── reverse edge: Gemini client / Anthropic upstream ──────────────────────
// Non-streaming only; emulates streaming as one SSE event.

/// `model` comes from the request path; Gemini carries no model in the body.
pub fn convert_gemini_to_anthropic_request(body: &Value, model: &str) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    // Gemini omits tool ids and matches by name; synthesize + correlate (empty ids 400).
    let mut tool_call_id_counts: HashMap<String, usize> = HashMap::new();
    let mut pending_tool_calls: HashMap<String, VecDeque<String>> = HashMap::new();
    if let Some(contents) = body.get("contents").and_then(|c| c.as_array()) {
        for turn in contents {
            let role = if turn.get("role").and_then(|r| r.as_str()) == Some("model") {
                "assistant"
            } else {
                "user"
            };
            let parts = turn.get("parts").and_then(|p| p.as_array());
            let blocks = parts
                .map(|p| {
                    gemini_parts_to_anthropic_blocks(
                        p,
                        &mut tool_call_id_counts,
                        &mut pending_tool_calls,
                    )
                })
                .unwrap_or_default();
            if !blocks.is_empty() {
                messages.push(json!({ "role": role, "content": blocks }));
            }
        }
    }

    let mut request = json!({
        "model": model,
        "messages": messages,
        "max_tokens": body
            .get("generationConfig")
            .and_then(|g| g.get("maxOutputTokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(BRIDGE_DEFAULT_ANTHROPIC_MAX_TOKENS),
    });

    let system = body
        .get("systemInstruction")
        .and_then(|s| s.get("parts"))
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .filter(|s| !s.is_empty());
    if let Some(system) = system {
        request["system"] = json!(system);
    }

    if let Some(decls) = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|groups| {
            groups
                .iter()
                .filter_map(|g| g.get("functionDeclarations").and_then(|d| d.as_array()))
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
        })
        .filter(|d| !d.is_empty())
    {
        let tools: Vec<Value> = decls
            .iter()
            .map(|d| {
                json!({
                    "name": d.get("name").cloned().unwrap_or_default(),
                    "description": d.get("description").cloned().unwrap_or(json!("")),
                    "input_schema": d
                        .get("parameters")
                        .cloned()
                        .unwrap_or(json!({ "type": "object", "properties": {} })),
                })
            })
            .collect();
        request["tools"] = Value::Array(tools);
    }

    if let Some(fcc) = body
        .get("toolConfig")
        .and_then(|t| t.get("functionCallingConfig"))
    {
        let choice = match fcc.get("mode").and_then(|m| m.as_str()) {
            Some("AUTO") => Some(json!({ "type": "auto" })),
            Some("NONE") => Some(json!({ "type": "none" })),
            Some("ANY") => Some(
                fcc.get("allowedFunctionNames")
                    .and_then(|n| n.as_array())
                    .and_then(|n| n.first())
                    .and_then(|n| n.as_str())
                    .map(|name| json!({ "type": "tool", "name": name }))
                    .unwrap_or_else(|| json!({ "type": "any" })),
            ),
            _ => None,
        };
        if let Some(choice) = choice {
            request["tool_choice"] = choice;
        }
    }

    if let Some(gen_cfg) = body.get("generationConfig") {
        if let Some(v) = gen_cfg.get("temperature") {
            request["temperature"] = v.clone();
        }
        if let Some(v) = gen_cfg.get("topP") {
            request["top_p"] = v.clone();
        }
        if let Some(v) = gen_cfg.get("topK") {
            request["top_k"] = v.clone();
        }
        if let Some(v) = gen_cfg.get("stopSequences") {
            request["stop_sequences"] = v.clone();
        }
        if let Some(budget) = gemini_thinking_config_to_budget(gen_cfg.get("thinkingConfig")) {
            request["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
        }
    }

    request
}

/// `thinkingBudget` passes through; `thinking_level` maps via canonical effort tiers.
fn gemini_thinking_config_to_budget(cfg: Option<&Value>) -> Option<u64> {
    let cfg = cfg?;
    if let Some(budget) = cfg.get("thinkingBudget").and_then(|v| v.as_u64()) {
        return Some(budget);
    }
    let level = cfg.get("thinking_level").and_then(|v| v.as_str())?;
    let effort = CanonicalEffort::from_gemini_thinking_level(level)?;
    effort.to_anthropic_budget_tokens()
}

fn gemini_parts_to_anthropic_blocks(
    parts: &[Value],
    tool_call_id_counts: &mut HashMap<String, usize>,
    pending_tool_calls: &mut HashMap<String, VecDeque<String>>,
) -> Vec<Value> {
    let mut blocks: Vec<Value> = Vec::new();
    for part in parts {
        if let Some(fc) = part.get("functionCall") {
            let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let id = extract_part_call_id(fc)
                .map(|id| uniquify_tool_call_id(id.to_string(), tool_call_id_counts))
                .unwrap_or_else(|| synthesize_tool_call_id(name, tool_call_id_counts));
            queue_pending_tool_call_id(pending_tool_calls, name, id.clone());
            blocks.push(json!({
                "type": "tool_use",
                "id": id,
                "name": fc.get("name").cloned().unwrap_or_else(|| json!("")),
                "input": fc.get("args").cloned().unwrap_or(json!({})),
            }));
            continue;
        }
        if let Some(fr) = part.get("functionResponse") {
            let name = fr.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let id = extract_part_call_id(fr)
                .and_then(|explicit| {
                    take_pending_tool_call_id(pending_tool_calls, name, explicit)
                        .or_else(|| pop_pending_tool_call_id(pending_tool_calls, name))
                        .or_else(|| Some(explicit.to_string()))
                })
                .or_else(|| pop_pending_tool_call_id(pending_tool_calls, name))
                .unwrap_or_else(|| synthesize_tool_call_id(name, tool_call_id_counts));
            let content = fr
                .get("response")
                .map(anthropic_tool_result_text)
                .unwrap_or_default();
            blocks.push(json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": content,
            }));
            continue;
        }
        if let Some(inline) = part.get("inlineData") {
            if let (Some(mime), Some(data)) = (
                inline.get("mimeType").and_then(|v| v.as_str()),
                inline.get("data").and_then(|v| v.as_str()),
            ) {
                blocks.push(json!({
                    "type": "image",
                    "source": { "type": "base64", "media_type": mime, "data": data },
                }));
            }
            continue;
        }
        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
            if part
                .get("thought")
                .and_then(|t| t.as_bool())
                .unwrap_or(false)
            {
                let mut block = json!({ "type": "thinking", "thinking": text });
                if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                    block["signature"] = json!(sig);
                }
                blocks.push(block);
            } else {
                blocks.push(json!({ "type": "text", "text": text }));
            }
        }
    }
    blocks
}

fn anthropic_tool_result_text(response: &Value) -> String {
    match response {
        Value::String(s) => s.clone(),
        Value::Object(map) => match map.get("content") {
            Some(Value::String(s)) => s.clone(),
            _ => response.to_string(),
        },
        _ => response.to_string(),
    }
}

pub fn convert_anthropic_to_gemini_response(resp: &Value) -> Value {
    let mut parts: Vec<Value> = Vec::new();
    let mut has_tool_use = false;
    if let Some(content) = resp.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        parts.push(json!({ "text": text }));
                    }
                }
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(|t| t.as_str()) {
                        let mut part = json!({ "text": text, "thought": true });
                        if let Some(sig) = block.get("signature").and_then(|v| v.as_str()) {
                            part["thoughtSignature"] = json!(sig);
                        }
                        parts.push(part);
                    }
                }
                Some("tool_use") => {
                    has_tool_use = true;
                    parts.push(json!({
                        "functionCall": {
                            "id": block.get("id").cloned().unwrap_or_else(|| json!("")),
                            "name": block.get("name").cloned().unwrap_or_else(|| json!("")),
                            "args": block.get("input").cloned().unwrap_or(json!({})),
                        }
                    }));
                }
                _ => {}
            }
        }
    }

    let finish = match resp.get("stop_reason").and_then(|s| s.as_str()) {
        Some("max_tokens") => "MAX_TOKENS",
        _ => "STOP",
    };
    let _ = has_tool_use; // Gemini uses STOP for tool calls.

    json!({
        "candidates": [{
            "content": { "role": "model", "parts": parts },
            "finishReason": finish,
            "index": 0,
        }],
        "usageMetadata": anthropic_usage_to_gemini(resp.get("usage")),
        "modelVersion": resp.get("model").cloned().unwrap_or(json!("")),
        "responseId": resp.get("id").cloned().unwrap_or_else(|| json!(http_utils::gen_id("resp"))),
    })
}

fn anthropic_usage_to_gemini(usage: Option<&Value>) -> Value {
    let n = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    let prompt = n("input_tokens");
    let candidates = n("output_tokens");
    let mut out = json!({
        "promptTokenCount": prompt,
        "candidatesTokenCount": candidates,
        "totalTokenCount": prompt.saturating_add(candidates),
    });
    let cached = usage
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    if let Some(v) = cached {
        out["cachedContentTokenCount"] = json!(v);
    }
    out
}

// ── streaming: Gemini SSE → Anthropic SSE ─────────────────────────────────

/// Native adapter; Gemini emits complete `GenerateContentResponse` chunks as `data:` lines.
pub struct GeminiToAnthropicStreamConverter {
    buf: SseLineBuffer,
    model: String,
    message_id: String,
    started: bool,
    finished: bool,
    block_count: usize,
    thinking_idx: Option<usize>,
    thinking_signature: Option<String>,
    text_idx: Option<usize>,
    saw_tool_use: bool,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    stop_reason: &'static str,
}

impl GeminiToAnthropicStreamConverter {
    pub fn new(model: &str) -> Self {
        Self {
            buf: SseLineBuffer::new(),
            model: model.to_string(),
            message_id: http_utils::gen_id("msg"),
            started: false,
            finished: false,
            block_count: 0,
            thinking_idx: None,
            thinking_signature: None,
            text_idx: None,
            saw_tool_use: false,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            stop_reason: "end_turn",
        }
    }

    pub fn push_bytes(&mut self, chunk: &[u8]) -> anyhow::Result<String> {
        let mut out = String::new();
        for line in self.buf.push_chunk(chunk)? {
            if let Some(data) = http_utils::sse_data_payload(&line)
                && data != "[DONE]"
                && let Ok(value) = serde_json::from_str::<Value>(data)
            {
                self.process_chunk(&value, &mut out);
            }
        }
        Ok(out)
    }

    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        self.ensure_started(&mut out);
        self.close_thinking(&mut out);
        self.close_text(&mut out);
        if !self.finished {
            let mut usage = json!({ "output_tokens": self.output_tokens });
            usage["input_tokens"] = json!(self.input_tokens);
            if let Some(v) = self.cache_read_input_tokens {
                usage["cache_read_input_tokens"] = json!(v);
            }
            out.push_str(&sse_event(
                "message_delta",
                &json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": self.stop_reason, "stop_sequence": null },
                    "usage": usage,
                }),
            ));
            out.push_str(&sse_event(
                "message_stop",
                &json!({ "type": "message_stop" }),
            ));
            self.finished = true;
        }
        out
    }

    fn process_chunk(&mut self, value: &Value, out: &mut String) {
        self.ensure_started(out);
        let parts = value
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        for part in &parts {
            if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str())
                && !sig.is_empty()
            {
                self.thinking_signature = Some(sig.to_string());
            }
            if let Some(text) = part.get("text").and_then(|v| v.as_str())
                && !text.is_empty()
            {
                if part
                    .get("thought")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false)
                {
                    self.emit_thinking_delta(text, out);
                } else {
                    self.emit_text_delta(text, out);
                }
            }
            if let Some(fc) = part.get("functionCall") {
                self.emit_tool_use(fc, out);
            }
        }

        if let Some(usage) = value.get("usageMetadata") {
            let n = |k: &str| usage.get(k).and_then(|v| v.as_u64());
            if let Some(v) = n("promptTokenCount") {
                self.input_tokens = v;
            }
            let candidates = n("candidatesTokenCount").unwrap_or(0);
            let thoughts = n("thoughtsTokenCount").unwrap_or(0);
            if candidates + thoughts > 0 {
                self.output_tokens = candidates.saturating_add(thoughts);
            }
            if let Some(v) = n("cachedContentTokenCount") {
                self.cache_read_input_tokens = Some(v);
            }
        }
        if let Some(reason) = value
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("finishReason"))
            .and_then(|v| v.as_str())
        {
            self.stop_reason = gemini_finish_to_anthropic(reason, self.saw_tool_use);
        }
    }

    fn ensure_started(&mut self, out: &mut String) {
        if self.started {
            return;
        }
        self.started = true;
        out.push_str(&sse_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": self.input_tokens, "output_tokens": 0 },
                },
            }),
        ));
    }

    fn emit_thinking_delta(&mut self, text: &str, out: &mut String) {
        if self.thinking_idx.is_none() {
            let idx = self.block_count;
            self.block_count += 1;
            self.thinking_idx = Some(idx);
            out.push_str(&sse_event(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "thinking", "thinking": "" },
                }),
            ));
        }
        out.push_str(&sse_event(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": self.thinking_idx,
                "delta": { "type": "thinking_delta", "thinking": text },
            }),
        ));
    }

    fn emit_text_delta(&mut self, text: &str, out: &mut String) {
        self.close_thinking(out);
        if self.text_idx.is_none() {
            let idx = self.block_count;
            self.block_count += 1;
            self.text_idx = Some(idx);
            out.push_str(&sse_event(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "text", "text": "" },
                }),
            ));
        }
        out.push_str(&sse_event(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": self.text_idx,
                "delta": { "type": "text_delta", "text": text },
            }),
        ));
    }

    fn emit_tool_use(&mut self, fc: &Value, out: &mut String) {
        self.close_thinking(out);
        self.close_text(out);
        let idx = self.block_count;
        self.block_count += 1;
        self.saw_tool_use = true;
        let id = fc
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| http_utils::gen_id("toolu"));
        out.push_str(&sse_event(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": id,
                    "name": fc.get("name").cloned().unwrap_or(json!("")),
                    "input": {},
                },
            }),
        ));
        let args = fc.get("args").cloned().unwrap_or(json!({}));
        out.push_str(&sse_event(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string()),
                },
            }),
        ));
        out.push_str(&sse_event(
            "content_block_stop",
            &json!({ "type": "content_block_stop", "index": idx }),
        ));
    }

    fn close_thinking(&mut self, out: &mut String) {
        if let Some(idx) = self.thinking_idx.take() {
            if let Some(sig) = self.thinking_signature.take() {
                out.push_str(&sse_event(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "signature_delta", "signature": sig },
                    }),
                ));
            }
            out.push_str(&sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": idx }),
            ));
        }
    }

    fn close_text(&mut self, out: &mut String) {
        if let Some(idx) = self.text_idx.take() {
            out.push_str(&sse_event(
                "content_block_stop",
                &json!({ "type": "content_block_stop", "index": idx }),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_maps_system_messages_tools_and_thinking() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "system": "be terse",
            "max_tokens": 256,
            "thinking": { "type": "enabled", "budget_tokens": 16384 },
            "messages": [{ "role": "user", "content": "hi" }],
            "tools": [{
                "name": "get_weather",
                "description": "weather",
                "input_schema": { "type": "object", "properties": { "city": { "type": "string" } } }
            }],
        });
        let out = convert_anthropic_to_gemini_request(
            &body,
            &AnthropicToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        assert_eq!(out["contents"][0]["role"], "user");
        assert_eq!(out["contents"][0]["parts"][0]["text"], "hi");
        assert_eq!(out["systemInstruction"]["parts"][0]["text"], "be terse");
        assert_eq!(
            out["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert_eq!(out["generationConfig"]["maxOutputTokens"], 256);
        assert!(out["generationConfig"]["thinkingConfig"]["thinkingBudget"].is_number());
    }

    #[test]
    fn forward_request_maps_adaptive_thinking_via_output_config() {
        // Adaptive thinking + output_config.effort must still map to thinkingConfig.
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [{ "role": "user", "content": "hi" }],
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "high" },
        });
        let out = convert_anthropic_to_gemini_request(
            &body,
            &AnthropicToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        assert!(
            out["generationConfig"]["thinkingConfig"]["thinkingBudget"].is_number(),
            "adaptive thinking should map to a thinkingConfig: {out}"
        );
    }

    #[test]
    fn request_round_trips_tool_use_signature_and_result_name() {
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [
                { "role": "assistant", "content": [
                    { "type": "thinking", "thinking": "hmm", "signature": "SIG123" },
                    { "type": "tool_use", "id": "call_1", "name": "lookup", "input": { "q": "x" } }
                ]},
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "call_1", "content": "42" }
                ]},
            ],
        });
        let out = convert_anthropic_to_gemini_request(
            &body,
            &AnthropicToGeminiConfig {
                default_model: "gemini-2.5-pro",
            },
        );
        let model_turn = &out["contents"][0];
        assert_eq!(model_turn["role"], "model");
        assert_eq!(model_turn["parts"][0]["functionCall"]["name"], "lookup");
        assert_eq!(model_turn["parts"][0]["thoughtSignature"], "SIG123");
        let user_turn = &out["contents"][1];
        assert_eq!(
            user_turn["parts"][0]["functionResponse"]["name"], "lookup",
            "functionResponse name resolved from the tool_use id"
        );
    }

    #[test]
    fn response_maps_thought_text_and_function_call() {
        let resp = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "text": "reasoning", "thought": true, "thoughtSignature": "SIG" },
                    { "text": "answer" },
                    { "functionCall": { "id": "call_9", "name": "f", "args": { "a": 1 } } }
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3 }
        });
        let out = convert_gemini_to_anthropic_response(
            &resp,
            &GeminiToAnthropicConfig {
                model: "gemini-2.5-pro",
            },
        );
        assert_eq!(out["type"], "message");
        assert_eq!(out["content"][0]["type"], "thinking");
        assert_eq!(out["content"][0]["signature"], "SIG");
        assert_eq!(out["content"][1]["type"], "text");
        assert_eq!(out["content"][1]["text"], "answer");
        assert_eq!(out["content"][2]["type"], "tool_use");
        assert_eq!(out["content"][2]["name"], "f");
        assert_eq!(out["stop_reason"], "tool_use");
        assert_eq!(out["usage"]["input_tokens"], 10);
        assert_eq!(out["usage"]["output_tokens"], 8, "candidates + thoughts");
    }

    #[test]
    fn stream_emits_anthropic_event_sequence_with_signature() {
        let mut conv = GeminiToAnthropicStreamConverter::new("gemini-2.5-pro");
        let chunk1 = concat!(
            r#"data: {"candidates":[{"content":{"parts":[{"text":"think","thought":true,"thoughtSignature":"S"}]}}]}"#,
            "\n\n"
        );
        let chunk2 = concat!(
            r#"data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}"#,
            "\n\n"
        );
        let chunk3 = concat!(
            r#"data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}"#,
            "\n\n"
        );
        let mut out = conv.push_bytes(chunk1.as_bytes()).unwrap();
        out.push_str(&conv.push_bytes(chunk2.as_bytes()).unwrap());
        out.push_str(&conv.push_bytes(chunk3.as_bytes()).unwrap());
        out.push_str(&conv.finish());

        assert!(out.contains("event: message_start"), "{out}");
        assert!(out.contains("thinking_delta"), "{out}");
        assert!(out.contains("signature_delta"), "{out}");
        assert!(out.contains(r#""signature":"S""#), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("hello"), "{out}");
        assert!(out.contains("event: message_stop"), "{out}");
        let ti = out.find("thinking_delta").unwrap();
        let si = out.find("signature_delta").unwrap();
        let xi = out.find("text_delta").unwrap();
        assert!(
            ti < si && si < xi,
            "thinking closes (with signature) before text: {out}"
        );
    }

    #[test]
    fn stream_emits_tool_use_block() {
        let mut conv = GeminiToAnthropicStreamConverter::new("gemini-2.5-pro");
        let chunk = concat!(
            r#"data: {"candidates":[{"content":{"parts":[{"functionCall":{"id":"c1","name":"do","args":{"x":1}}}]},"finishReason":"STOP"}]}"#,
            "\n\n"
        );
        let mut out = conv.push_bytes(chunk.as_bytes()).unwrap();
        out.push_str(&conv.finish());
        assert!(out.contains(r#""type":"tool_use""#), "{out}");
        assert!(out.contains("input_json_delta"), "{out}");
        assert!(out.contains(r#""stop_reason":"tool_use""#), "{out}");
    }

    // ── reverse edge: Gemini client / Anthropic upstream ──

    #[test]
    fn reverse_request_maps_contents_system_tools_and_thinking() {
        let body = json!({
            "systemInstruction": { "parts": [{ "text": "be terse" }] },
            "contents": [
                { "role": "user", "parts": [{ "text": "hi" }] },
                { "role": "model", "parts": [
                    { "text": "reasoning", "thought": true, "thoughtSignature": "SIG" },
                    { "functionCall": { "id": "c1", "name": "lookup", "args": { "q": "x" } } }
                ]},
                { "role": "user", "parts": [
                    { "functionResponse": { "id": "c1", "name": "lookup", "response": { "content": "42" } } }
                ]},
            ],
            "tools": [{ "functionDeclarations": [{
                "name": "lookup", "description": "d",
                "parameters": { "type": "object", "properties": {} }
            }]}],
            "generationConfig": {
                "maxOutputTokens": 512, "temperature": 0.5,
                "thinkingConfig": { "thinkingBudget": 2048 }
            }
        });
        let out = convert_gemini_to_anthropic_request(&body, "claude-sonnet-4-5");
        assert_eq!(out["model"], "claude-sonnet-4-5");
        assert_eq!(out["max_tokens"], 512);
        assert_eq!(out["system"], "be terse");
        assert_eq!(out["messages"][0]["content"][0]["text"], "hi");
        let model_turn = &out["messages"][1];
        assert_eq!(model_turn["role"], "assistant");
        assert_eq!(model_turn["content"][0]["type"], "thinking");
        assert_eq!(model_turn["content"][0]["signature"], "SIG");
        assert_eq!(model_turn["content"][1]["type"], "tool_use");
        assert_eq!(model_turn["content"][1]["name"], "lookup");
        assert_eq!(out["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(out["messages"][2]["content"][0]["tool_use_id"], "c1");
        assert_eq!(out["messages"][2]["content"][0]["content"], "42");
        assert_eq!(out["tools"][0]["name"], "lookup");
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["thinking"]["budget_tokens"], 2048);
    }

    #[test]
    fn reverse_request_defaults_max_tokens_when_absent() {
        let body = json!({ "contents": [{ "role": "user", "parts": [{ "text": "hi" }] }] });
        let out = convert_gemini_to_anthropic_request(&body, "claude-sonnet-4-5");
        assert_eq!(out["max_tokens"], BRIDGE_DEFAULT_ANTHROPIC_MAX_TOKENS);
    }

    #[test]
    fn reverse_request_synthesizes_and_correlates_tool_ids() {
        // Parallel Gemini calls (no ids) → distinct non-empty ids, responses correlated.
        let body = json!({
            "contents": [
                { "role": "user", "parts": [{ "text": "weather?" }] },
                { "role": "model", "parts": [
                    { "functionCall": { "name": "get_weather", "args": { "city": "SF" } } },
                    { "functionCall": { "name": "get_weather", "args": { "city": "LA" } } },
                ]},
                { "role": "user", "parts": [
                    { "functionResponse": { "name": "get_weather", "response": { "t": 60 } } },
                    { "functionResponse": { "name": "get_weather", "response": { "t": 75 } } },
                ]},
            ],
        });
        let out = convert_gemini_to_anthropic_request(&body, "claude-sonnet-4-5");
        let calls = &out["messages"][1]["content"];
        let id0 = calls[0]["id"].as_str().unwrap();
        let id1 = calls[1]["id"].as_str().unwrap();
        assert!(!id0.is_empty() && !id1.is_empty(), "ids must be non-empty");
        assert_ne!(id0, id1, "parallel calls need distinct ids");
        let results = &out["messages"][2]["content"];
        assert_eq!(results[0]["tool_use_id"].as_str().unwrap(), id0);
        assert_eq!(results[1]["tool_use_id"].as_str().unwrap(), id1);
    }

    #[test]
    fn reverse_response_maps_content_blocks_and_usage() {
        let resp = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                { "type": "thinking", "thinking": "hmm", "signature": "S" },
                { "type": "text", "text": "answer" },
                { "type": "tool_use", "id": "t1", "name": "f", "input": { "a": 1 } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2 }
        });
        let out = convert_anthropic_to_gemini_response(&resp);
        let parts = &out["candidates"][0]["content"]["parts"];
        assert_eq!(parts[0]["thought"], true);
        assert_eq!(parts[0]["thoughtSignature"], "S");
        assert_eq!(parts[1]["text"], "answer");
        assert_eq!(parts[2]["functionCall"]["name"], "f");
        assert_eq!(out["candidates"][0]["finishReason"], "STOP");
        assert_eq!(out["usageMetadata"]["promptTokenCount"], 10);
        assert_eq!(out["usageMetadata"]["candidatesTokenCount"], 5);
        assert_eq!(out["usageMetadata"]["cachedContentTokenCount"], 2);
    }

    #[test]
    fn reverse_response_max_tokens_finish() {
        let resp = json!({
            "content": [{ "type": "text", "text": "cut" }],
            "stop_reason": "max_tokens",
        });
        let out = convert_anthropic_to_gemini_response(&resp);
        assert_eq!(out["candidates"][0]["finishReason"], "MAX_TOKENS");
    }
}
