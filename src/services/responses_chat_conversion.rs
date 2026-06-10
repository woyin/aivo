//! Responses API ↔ Chat Completions conversion logic
//!
//! Converts between OpenAI Responses API format and Chat Completions format.
//! Used by the ResponsesToChatRouter and ServeRouter to bridge clients that
//! speak the Responses API with providers that only support Chat Completions.
use crate::services::codex_model_map::map_model_for_codex_cli;
use crate::services::http_utils::{self, current_unix_ts};
use crate::services::model_names::select_model_for_provider_attempt;
use crate::services::openai_models::{
    OpenAIChatRequest, ResponsesResponse,
    convert_chat_to_responses_request as convert_typed_chat_to_responses,
    convert_responses_to_chat_response as convert_typed_responses_to_chat,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::responses_to_chat_router::ResponsesToChatRouterConfig;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Returns true if the body uses OpenAI Responses API format
/// (has "input" array, no "messages" array)
pub fn is_responses_api_format(body: &Value) -> bool {
    body.get("input").and_then(|v| v.as_array()).is_some() && body.get("messages").is_none()
}

pub(crate) fn cap_token_value(v: &Value, cap: Option<u64>) -> Value {
    if let Some(limit) = cap {
        http_utils::parse_token_u64(v)
            .map(|n| {
                if n == 0 {
                    json!(n)
                } else {
                    json!(n.min(limit))
                }
            })
            .unwrap_or(v.clone())
    } else {
        v.clone()
    }
}

pub(crate) fn apply_max_tokens_cap_to_fields(body: &mut Value, cap: Option<u64>, fields: &[&str]) {
    for field in fields {
        if let Some(v) = body.get(*field).cloned() {
            body[*field] = cap_token_value(&v, cap);
        }
    }
}

/// Cap `reasoning.effort` values that most models don't support (e.g. `xhigh` → `high`).
pub(crate) fn cap_reasoning_effort(body: &mut Value) {
    if let Some(effort) = body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|e| e.as_str())
    {
        if effort.eq_ignore_ascii_case("xhigh") {
            body["reasoning"]["effort"] = json!("high");
        }
    } else if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str())
        && effort.eq_ignore_ascii_case("xhigh")
    {
        body["reasoning_effort"] = json!("high");
    }
}

/// Ensure every text-bearing content part in `input` messages has a `text` field.
///
/// The Responses API rejects `output_text` and `input_text` parts that are
/// missing `text`.  Codex CLI can echo back content parts from a previous
/// response where `text` was absent or null; this guard adds an empty string
/// so the upstream API accepts the request.
pub(crate) fn sanitize_input_content(body: &mut Value) {
    let Some(input) = body.get_mut("input").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for item in input.iter_mut() {
        if item.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let Some(parts) = item.get_mut("content").and_then(|c| c.as_array_mut()) else {
            continue;
        };
        for part in parts.iter_mut() {
            let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match part_type {
                "output_text" | "input_text" | ""
                    if !part.get("text").is_some_and(|t| t.is_string()) =>
                {
                    part["text"] = json!("");
                }
                _ => {}
            }
        }
    }
}

/// Converts an OpenAI Responses API request body to Chat Completions format.
///
/// Handles all input item types:
/// - `message` → role/content message
/// - `function_call` → assistant message with tool_calls
/// - `function_call_output` → tool message
///
/// Also converts tool format (Responses API has no `function` wrapper;
/// Chat Completions requires `{type, function: {name, description, parameters}}`).
pub fn convert_responses_to_chat_request(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
) -> Value {
    let mut messages: Vec<Value> = vec![];

    // System message from "instructions" field
    if let Some(instructions) = body.get("instructions").and_then(|v| v.as_str())
        && !instructions.is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }

    // A DeepSeek thinking-mode turn arrives split across Responses items
    // (reasoning + message + one function_call per parallel call); re-merge
    // them into ONE Chat assistant message, else strict upstreams reject the
    // split (tool results detached from their call, reasoning_content dropped).
    // `current_assistant` indexes the open turn; non-assistant items close it.
    let mut pending_reasoning: String = String::new();
    let mut current_assistant: Option<usize> = None;

    // Convert "input" array items
    if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("reasoning") => {
                    let text = extract_reasoning_text(item);
                    if !text.is_empty() {
                        if pending_reasoning.is_empty() {
                            pending_reasoning = text;
                        } else {
                            pending_reasoning.push('\n');
                            pending_reasoning.push_str(&text);
                        }
                    }
                }
                Some("message") => {
                    // Validate role - only allow valid OpenAI chat completion roles
                    let role = item
                        .get("role")
                        .and_then(|v| v.as_str())
                        .filter(|r| matches!(*r, "system" | "user" | "assistant" | "tool"))
                        .unwrap_or("user");
                    // Vision/file inputs (input_image, input_file) must be
                    // preserved when bridging to Chat Completions; falling back
                    // to text-only here silently dropped them. The helper
                    // collapses to a string when no non-text part is present so
                    // the wire format is unchanged for the common case.
                    let content = convert_responses_content_to_chat(item.get("content"));
                    if role == "assistant" {
                        // Fold text emitted alongside tool calls into the open
                        // turn instead of splitting it into its own message.
                        let idx = open_assistant_turn(&mut messages, &mut current_assistant);
                        fold_assistant_content(&mut messages[idx]["content"], content);
                        flush_reasoning_into(
                            &mut messages[idx],
                            &mut pending_reasoning,
                            item,
                            config.requires_reasoning_content,
                        );
                    } else {
                        current_assistant = None;
                        messages.push(json!({"role": role, "content": content}));
                    }
                }
                Some("function_call") => {
                    // Use call_id as the Chat Completions tool_calls[].id so it matches
                    // the corresponding function_call_output.call_id → tool_call_id.
                    // Fall back to id only if call_id is absent.
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("call_0");
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let arguments = item
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let tool_call = json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                    // Append to the open turn so parallel calls and post-narration calls share one message.
                    let idx = open_assistant_turn(&mut messages, &mut current_assistant);
                    let msg = &mut messages[idx];
                    match msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                        Some(arr) => arr.push(tool_call),
                        None => msg["tool_calls"] = json!([tool_call]),
                    }
                    flush_reasoning_into(
                        msg,
                        &mut pending_reasoning,
                        item,
                        config.requires_reasoning_content,
                    );
                }
                Some("function_call_output") => {
                    current_assistant = None;
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    messages
                        .push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
                }
                None => {
                    // Simple string input
                    if let Some(s) = item.as_str() {
                        current_assistant = None;
                        messages.push(json!({"role": "user", "content": s}));
                    }
                }
                _ => {}
            }
        }
    }

    // Convert tools: filter non-function, convert format
    let tools: Vec<Value> = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"))
                .map(convert_tool_to_chat_format)
                .collect()
        })
        .unwrap_or_default();

    // Apply model name transform (e.g. openai/ prefix for OpenRouter)
    // Skip transform when using Copilot — model names pass through unchanged
    // If actual_model is set, use that (it was set by environment injector)
    let selected_model = select_model_for_provider_attempt(
        &config.target_base_url,
        body.get("model").and_then(|v| v.as_str()),
        config.actual_model.as_deref(),
        config.target_protocol,
    );
    let model = if config.copilot_token_manager.is_none() {
        if config.target_protocol == ProviderProtocol::Openai {
            Value::String(super::responses_to_chat_router::transform_model_str(
                &selected_model,
                &config.target_base_url,
                config.model_prefix.as_deref(),
            ))
        } else {
            Value::String(selected_model)
        }
    } else {
        Value::String(selected_model)
    };

    let mut chat = json!({
        "model": model,
        "messages": messages,
        "stream": false,  // request non-streaming for simpler response handling
    });

    if !tools.is_empty() {
        chat["tools"] = Value::Array(tools);
    }
    if let Some(v) = body
        .get("max_output_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        chat["max_tokens"] = cap_token_value(v, config.max_tokens_cap);
    }
    // Dropped for models that reject sampling params (o-series etc.) —
    // forwarding them turns into upstream 400s.
    let rejects_sampling = chat["model"]
        .as_str()
        .is_some_and(crate::services::model_metadata::rejects_temperature);
    if !rejects_sampling {
        for field in ["temperature", "top_p"] {
            if let Some(v) = body.get(field) {
                chat[field] = v.clone();
            }
        }
    }

    // Copy reasoning fields
    if let Some(reasoning) = body.get("reasoning").and_then(|r| r.as_object()) {
        if let Some(effort) = reasoning.get("effort").and_then(|e| e.as_str()) {
            chat["reasoning_effort"] = json!(effort);
        }
    } else if let Some(effort) = body.get("reasoning_effort").and_then(|e| e.as_str()) {
        chat["reasoning_effort"] = json!(effort);
    }
    cap_reasoning_effort(&mut chat);

    chat
}

/// Copies `reasoning_content` from a source item onto a Chat Completions message.
/// Falls back to a single-space sentinel when the provider requires a non-empty value.
fn attach_reasoning_content(msg: &mut Value, source: &Value, requires: bool) {
    let rc = source
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| requires.then(|| " ".to_string()));
    if let Some(rc) = rc {
        msg["reasoning_content"] = json!(rc);
    }
}

/// Returns the index of the open assistant turn, opening a fresh assistant
/// message (null content) if none is active.
fn open_assistant_turn(messages: &mut Vec<Value>, current: &mut Option<usize>) -> usize {
    if let Some(idx) = *current {
        return idx;
    }
    messages.push(json!({"role": "assistant", "content": null}));
    let idx = messages.len() - 1;
    *current = Some(idx);
    idx
}

/// Drains buffered standalone-`reasoning` text onto an open assistant turn.
/// Appends to reasoning already collected for the turn (so a reasoning item
/// arriving mid-turn isn't carried to a later message) and upgrades a
/// single-space sentinel to real text. With nothing buffered, falls back to
/// the item's own non-standard `reasoning_content` field (or the sentinel the
/// provider requires).
fn flush_reasoning_into(msg: &mut Value, pending: &mut String, source: &Value, requires: bool) {
    if pending.is_empty() {
        if msg.get("reasoning_content").is_none() {
            attach_reasoning_content(msg, source, requires);
        }
        return;
    }
    let rc = std::mem::take(pending);
    match msg.get("reasoning_content").and_then(|v| v.as_str()) {
        Some(existing) if !existing.is_empty() && existing != " " => {
            msg["reasoning_content"] = json!(format!("{existing}\n{rc}"));
        }
        _ => msg["reasoning_content"] = json!(rc),
    }
}

/// Folds assistant text into an open turn's `content`: two strings append with
/// a newline, and a null/empty existing value adopts the addition. Assistant
/// turns here are text-only, so the exotic cases where either side is a
/// multimodal array keep the existing value and drop the addition.
fn fold_assistant_content(existing: &mut Value, addition: Value) {
    match existing {
        Value::String(s) if !s.is_empty() => {
            if let Some(add) = addition.as_str().filter(|a| !a.is_empty()) {
                *existing = Value::String(format!("{s}\n{add}"));
            }
        }
        Value::Null | Value::String(_) => *existing = addition,
        _ => {}
    }
}

/// Extract reasoning text from a standard Responses-API `type:"reasoning"` item.
/// Canonical shape is `summary: [{type:"summary_text", text}]`; some upstreams
/// place the trace in `content[*].text` or a bare `text` field, so accept both.
/// `encrypted_content` is opaque and provider-specific — skip it.
fn extract_reasoning_text(item: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    let collect = |arr: &Value, out: &mut Vec<String>| {
        if let Some(items) = arr.as_array() {
            for part in items {
                if let Some(s) = part.as_str() {
                    if !s.is_empty() {
                        out.push(s.to_string());
                    }
                } else if let Some(text) = part.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(text.to_string());
                } else if let Some(text) = part.get("reasoning").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(text.to_string());
                }
            }
        }
    };
    if let Some(summary) = item.get("summary") {
        collect(summary, &mut parts);
    }
    if let Some(content) = item.get("content") {
        collect(content, &mut parts);
    }
    if let Some(text) = item.get("text").and_then(|v| v.as_str())
        && !text.is_empty()
    {
        parts.push(text.to_string());
    }
    parts.join("\n")
}

/// Convert a Responses-API content value (string or array of `input_text` /
/// `input_image` / `input_file` parts, etc.) into a Chat Completions content
/// value. Returns a `String` when every part is text (preserves the existing
/// wire format), and an array of `{type: ...}` content parts when any
/// multimodal part is present.
///
/// Recognised Responses-API parts:
/// - `input_text` / bare text → `{type: "text", text}`
/// - `input_image` with `image_url` (string or {url}) → `{type: "image_url",
///   image_url: {url, detail?}}`. Both http(s) URLs and `data:` URIs are
///   passed through unchanged — Chat Completions accepts both.
/// - `input_file` → inlined as `{type: "text", text: "[attached file: <name>]"}`
///   so the model still gets a turn-shaped reference instead of a silent drop.
///   (Chat Completions has no native file part. Plan default — see review notes.)
pub fn convert_responses_content_to_chat(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(s)) => Value::String(s.clone()),
        Some(Value::Array(parts)) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(responses_content_part_to_chat_part)
                .collect();
            if converted
                .iter()
                .all(|p| p.get("type").and_then(|v| v.as_str()) == Some("text"))
            {
                let joined = converted
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|v| v.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                Value::String(joined)
            } else {
                Value::Array(converted)
            }
        }
        Some(Value::Object(obj)) => Value::String(
            obj.get("text")
                .and_then(|v| v.as_str())
                .or_else(|| obj.get("content").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string(),
        ),
        _ => Value::String(String::new()),
    }
}

fn responses_content_part_to_chat_part(part: &Value) -> Option<Value> {
    if let Some(s) = part.as_str() {
        return Some(json!({"type": "text", "text": s}));
    }
    let part_type = part.get("type").and_then(|v| v.as_str());
    match part_type {
        // Most Responses-API text variants funnel through `text`/`content`.
        Some("input_text") | Some("text") | Some("output_text") | None => part
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| part.get("content").and_then(|v| v.as_str()))
            .map(|t| json!({"type": "text", "text": t})),
        Some("input_image") => {
            // Responses API: image_url is a string OR { url, detail? }.
            let (url, detail) = match part.get("image_url") {
                Some(Value::String(s)) => (Some(s.as_str()), None),
                Some(Value::Object(o)) => (
                    o.get("url").and_then(|v| v.as_str()),
                    o.get("detail").cloned(),
                ),
                _ => (None, None),
            };
            url.map(|u| {
                let mut iu = serde_json::Map::new();
                iu.insert("url".to_string(), Value::String(u.to_string()));
                if let Some(d) = detail {
                    iu.insert("detail".to_string(), d);
                }
                json!({"type": "image_url", "image_url": Value::Object(iu)})
            })
        }
        Some("input_file") => {
            // Chat Completions has no native file content part, so inline a
            // text reference. Default per fix-plan; can be revisited.
            let name = part
                .get("filename")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("name").and_then(|v| v.as_str()))
                .unwrap_or("file");
            Some(json!({"type": "text", "text": format!("[attached file: {name}]")}))
        }
        _ => part
            .get("text")
            .and_then(|v| v.as_str())
            .map(|t| json!({"type": "text", "text": t})),
    }
}

/// Extracts text from a content value (handles string, array of content parts)
pub fn extract_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                Value::String(s) => Some(s.clone()),
                _ => p
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| p.get("content").and_then(|v| v.as_str()))
                    .map(String::from),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Object(obj)) => obj
            .get("text")
            .and_then(|v| v.as_str())
            .or_else(|| obj.get("content").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string(),
        _ => String::new(),
    }
}

/// Converts a tool from Responses API format to Chat Completions format.
///
/// Responses API: `{type, name, description, parameters}`
/// Chat Completions: `{type, function: {name, description, parameters}}`
pub fn convert_tool_to_chat_format(tool: &Value) -> Value {
    // Already in Chat Completions format (has "function" wrapper)
    if tool.get("function").is_some() {
        return tool.clone();
    }
    let mut func = serde_json::Map::new();
    for field in ["name", "description", "parameters", "strict"] {
        if let Some(v) = tool.get(field) {
            func.insert(field.to_string(), v.clone());
        }
    }
    json!({"type": "function", "function": Value::Object(func)})
}

/// Parses a provider HTTP response body as either a JSON chat completion
/// (stream:false) or an SSE chat completion stream (stream:true).
/// Returns a unified non-streaming chat completion JSON.
pub fn parse_provider_response(text: &str) -> anyhow::Result<Value> {
    // Try JSON first (non-streaming response)
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return Ok(v);
    }
    // Fallback: provider returned SSE despite stream:false
    Ok(accumulate_chat_sse(text))
}

/// Reads an SSE chat completions stream and returns a synthesized non-streaming response.
pub fn accumulate_chat_sse(text: &str) -> Value {
    let mut content = String::new();
    let mut reasoning_content = String::new();
    // (id, name, accumulated_args)
    let mut tool_calls_acc: Vec<(String, String, String)> = Vec::new();
    let mut finish_reason = String::from("stop");

    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<Value>(data) {
                let choice = &chunk["choices"][0];
                let delta = &choice["delta"];

                if let Some(c) = delta["content"].as_str() {
                    content.push_str(c);
                }
                if let Some(rc) = delta["reasoning_content"].as_str() {
                    reasoning_content.push_str(rc);
                }
                if let Some(tcs) = delta["tool_calls"].as_array() {
                    for tc in tcs {
                        let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                        while tool_calls_acc.len() <= idx {
                            tool_calls_acc.push((String::new(), String::new(), String::new()));
                        }
                        if let Some(id) = tc["id"].as_str()
                            && !id.is_empty()
                        {
                            tool_calls_acc[idx].0 = id.to_string();
                        }
                        if let Some(name) = tc["function"]["name"].as_str()
                            && !name.is_empty()
                        {
                            tool_calls_acc[idx].1.push_str(name);
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            tool_calls_acc[idx].2.push_str(args);
                        }
                    }
                }
                if let Some(fr) = choice["finish_reason"].as_str()
                    && !fr.is_empty()
                {
                    finish_reason = fr.to_string();
                }
            }
        }
    }

    if !tool_calls_acc.is_empty() {
        let tcs: Vec<Value> = tool_calls_acc
            .iter()
            .enumerate()
            .map(|(i, (id, name, args))| {
                json!({
                    "id": if id.is_empty() { format!("call_{}", i) } else { id.clone() },
                    "type": "function",
                    "function": {"name": name, "arguments": args}
                })
            })
            .collect();
        let mut msg = json!({"role": "assistant", "content": null, "tool_calls": tcs});
        if !reasoning_content.is_empty() {
            msg["reasoning_content"] = json!(reasoning_content);
        }
        json!({"choices": [{"message": msg, "finish_reason": "tool_calls"}]})
    } else {
        let mut msg = json!({"role": "assistant", "content": content});
        if !reasoning_content.is_empty() {
            msg["reasoning_content"] = json!(reasoning_content);
        }
        json!({"choices": [{"message": msg, "finish_reason": finish_reason}]})
    }
}

/// Converts a Chat Completions non-streaming response to Responses API SSE events.
///
/// Codex CLI parses these SSE events to display output and handle tool calls.
/// Handles both text responses and tool call responses.
///
/// Key correctness requirements from the OpenAI Responses API spec:
/// - `object` must be "response" (not "realtime.response")
/// - All sub-events must include `response_id`
/// - Function call items need a `call_id` (= Chat Completions tc.id) separate
///   from `id` (a fresh item identifier); Codex puts `call_id` in the
///   follow-up `function_call_output.call_id` field
pub fn convert_chat_response_to_responses_sse(
    chat: &Value,
    requires_reasoning_content: bool,
    original_model: &str,
) -> String {
    let resp_id = gen_id("resp");
    let created_at = current_unix_ts();
    // Map model name for Codex CLI compatibility
    let codex_model = map_model_for_codex_cli(original_model);
    let mut sse = String::new();
    let mut output_items: Vec<Value> = Vec::new();

    // response.created — required opening event
    sse.push_str(&sse_event(
        "response.created",
        &json!({
            "type": "response.created",
            "response": {
                "id": resp_id, "object": "response",
                "model": codex_model,
                "created_at": created_at, "status": "in_progress", "output": []
            }
        }),
    ));

    let (content, tool_calls, reasoning_content) = extract_chat_response_payload(chat);

    // Pass reasoning_content through to function_call output items so subsequent requests
    // can round-trip it back. Auto-detected from provider response; no config flag needed.
    // For providers that require a non-empty value even when none was returned (requires_reasoning_content),
    // fall back to content or a single-space sentinel.
    let reasoning_for_tool = if !reasoning_content.is_empty() {
        reasoning_content.clone()
    } else if requires_reasoning_content {
        if !content.is_empty() {
            content.clone()
        } else {
            " ".to_string() // single-space sentinel satisfies non-empty requirement
        }
    } else {
        String::new()
    };

    // Emit a standard `type:"reasoning"` output item before any message or
    // function_call items. Codex CLI parses output items with typed structs,
    // so a stray `reasoning_content` field tucked onto a function_call would
    // be stripped on the round-trip; only a real reasoning item survives.
    // Provider quirk fixes still rely on the non-standard field below — emit
    // both so legacy paths and Codex both work.
    let mut next_output_index: usize = 0;
    if !reasoning_content.is_empty() {
        let rs_id = gen_id("rs");
        sse.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": resp_id, "output_index": next_output_index,
                "item": {
                    "id": rs_id, "type": "reasoning",
                    "summary": []
                }
            }),
        ));
        sse.push_str(&sse_event(
            "response.reasoning_summary_part.added",
            &json!({
                "type": "response.reasoning_summary_part.added",
                "response_id": resp_id, "item_id": rs_id,
                "output_index": next_output_index, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}
            }),
        ));
        sse.push_str(&sse_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "response_id": resp_id, "item_id": rs_id,
                "output_index": next_output_index, "summary_index": 0,
                "delta": reasoning_content
            }),
        ));
        sse.push_str(&sse_event(
            "response.reasoning_summary_text.done",
            &json!({
                "type": "response.reasoning_summary_text.done",
                "response_id": resp_id, "item_id": rs_id,
                "output_index": next_output_index, "summary_index": 0,
                "text": reasoning_content
            }),
        ));
        sse.push_str(&sse_event(
            "response.reasoning_summary_part.done",
            &json!({
                "type": "response.reasoning_summary_part.done",
                "response_id": resp_id, "item_id": rs_id,
                "output_index": next_output_index, "summary_index": 0,
                "part": {"type": "summary_text", "text": reasoning_content}
            }),
        ));
        let reasoning_done_item = json!({
            "id": rs_id, "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reasoning_content}]
        });
        sse.push_str(&sse_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": resp_id, "output_index": next_output_index,
                "item": reasoning_done_item.clone()
            }),
        ));
        output_items.push(reasoning_done_item);
        next_output_index += 1;
    }

    if !tool_calls.is_empty() {
        // Tool call response — each tool call becomes a function_call output item
        for (offset, tc) in tool_calls.iter().enumerate() {
            let i = next_output_index + offset;
            // call_id = the Chat Completions tool call ID (referenced in tool results)
            let call_id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("call_0");
            // item_id = a fresh item identifier within this response
            let item_id = gen_id("fc");
            let tc_name = tc["function"]["name"].as_str().unwrap_or("");
            let tc_args = tc["function"]["arguments"].as_str().unwrap_or("{}");

            sse.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": resp_id, "output_index": i,
                    "item": {
                        "id": item_id, "call_id": call_id,
                        "type": "function_call", "status": "in_progress",
                        "name": tc_name, "arguments": ""
                    }
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": resp_id, "output_index": i,
                    "item_id": item_id, "delta": tc_args
                }),
            ));
            sse.push_str(&sse_event(
                "response.function_call_arguments.done",
                &json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": resp_id, "output_index": i,
                    "item_id": item_id, "arguments": tc_args
                }),
            ));

            // Build done_item with reasoning_content if the provider returned any
            let mut done_item = json!({
                "id": item_id, "call_id": call_id,
                "type": "function_call", "status": "completed",
                "name": tc_name, "arguments": tc_args
            });
            if !reasoning_for_tool.is_empty() {
                done_item["reasoning_content"] = json!(reasoning_for_tool.clone());
            }
            sse.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": resp_id, "output_index": i,
                    "item": done_item
                }),
            ));
            let mut output_item = json!({
                "id": item_id, "call_id": call_id,
                "type": "function_call", "status": "completed",
                "name": tc_name, "arguments": tc_args
            });
            if !reasoning_for_tool.is_empty() {
                output_item["reasoning_content"] = json!(reasoning_for_tool.clone());
            }
            output_items.push(output_item);
        }
    } else {
        // Text message response
        let msg_id = gen_id("msg");
        let i = next_output_index;

        sse.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": resp_id, "output_index": i,
                "item": {
                    "id": msg_id, "type": "message",
                    "status": "in_progress", "role": "assistant", "content": []
                }
            }),
        ));
        // Output text part (always present, even if empty)
        sse.push_str(&sse_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": i, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        if !content.is_empty() {
            sse.push_str(&sse_event(
                "response.output_text.delta",
                &json!({
                    "type": "response.output_text.delta",
                    "response_id": resp_id, "item_id": msg_id,
                    "output_index": i, "content_index": 0, "delta": content
                }),
            ));
        }
        sse.push_str(&sse_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": i, "content_index": 0, "text": content
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": i, "content_index": 0,
                "part": {"type": "output_text", "text": content}
            }),
        ));

        // Reasoning lives in the standalone reasoning item, not here: a
        // `reasoning` part inside `message.content` makes Codex.app reject the
        // whole message ("Unexpected content item in agent message").
        let content_parts =
            vec![json!({"type": "output_text", "text": content, "annotations": []})];
        let done_item = json!({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": content_parts
        });
        sse.push_str(&sse_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": resp_id, "output_index": i, "item": done_item
            }),
        ));
        output_items.push(json!({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": content_parts
        }));
    }

    // response.completed — required closing event with full output array
    let mut response = json!({
        "id": resp_id, "object": "response",
        "model": codex_model,
        "created_at": created_at, "status": "completed",
        "output": output_items
    });
    if let Some(usage) = chat_usage_to_responses_usage(chat) {
        response["usage"] = usage;
    }
    sse.push_str(&sse_event(
        "response.completed",
        &json!({
            "type": "response.completed",
            "response": response
        }),
    ));

    sse
}

/// Incrementally converts a streaming Chat Completions response (SSE chunks with
/// `delta.content` / `delta.reasoning_content` / `delta.tool_calls`) into
/// Responses API SSE events, so output reaches Codex as it's produced instead of
/// arriving in one blob after the turn finishes.
///
/// Emits the same event sequence and `response.completed` output shape as the
/// buffered `convert_chat_response_to_responses_sse`, so a round-trip through the
/// streaming path is indistinguishable from the buffered one to Codex.
pub struct ResponsesStreamConverter {
    pending: Vec<u8>,
    resp_id: String,
    created_at: u64,
    codex_model: String,
    requires_reasoning_content: bool,
    created_emitted: bool,
    finished: bool,
    next_output_index: usize,
    reasoning: Option<StreamItem>,
    message: Option<StreamItem>,
    tools: Vec<StreamToolCall>,
    usage: Option<Value>,
}

struct StreamItem {
    id: String,
    output_index: usize,
    text: String,
}

struct StreamToolCall {
    chat_index: u64,
    output_index: usize,
    item_id: String,
    call_id: String,
    name: String,
    args: String,
}

impl ResponsesStreamConverter {
    pub fn new(original_model: &str, requires_reasoning_content: bool) -> Self {
        Self {
            pending: Vec::new(),
            resp_id: gen_id("resp"),
            created_at: current_unix_ts(),
            codex_model: map_model_for_codex_cli(original_model),
            requires_reasoning_content,
            created_emitted: false,
            finished: false,
            next_output_index: 0,
            reasoning: None,
            message: None,
            tools: Vec::new(),
            usage: None,
        }
    }

    /// Feeds a network chunk of the upstream SSE body, returning any Responses API
    /// SSE to forward to the client. Buffers partial lines across calls.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> String {
        let mut out = String::new();
        self.ensure_created(&mut out);
        self.pending.extend_from_slice(chunk);
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.pending[..pos]).into_owned();
            self.pending.drain(..=pos);
            self.process_line(line.trim_end_matches('\r'), &mut out);
        }
        // OOM backstop: a newline-less upstream can't be converted anyway;
        // drop the oversized partial line instead of buffering it forever.
        if self.pending.len() > crate::services::http_utils::MAX_SSE_PENDING_BYTES {
            self.pending = Vec::new();
        }
        out
    }

    /// Flushes any buffered trailing line and emits the closing `.done` events
    /// plus `response.completed`.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        self.ensure_created(&mut out);
        if !self.pending.is_empty() {
            let line = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.process_line(line.trim_end_matches('\r'), &mut out);
        }
        if self.finished {
            return out;
        }
        self.finished = true;

        // Match the buffered converter: when neither text nor tool calls were
        // produced, still emit an (empty) message item so Codex sees a turn.
        if self.message.is_none() && self.tools.is_empty() {
            self.start_message(&mut out);
        }

        let reasoning_text = self.reasoning.as_ref().map(|r| r.text.clone());
        let reasoning_for_tool = match &reasoning_text {
            Some(t) if !t.is_empty() => t.clone(),
            _ if self.requires_reasoning_content => {
                let msg_text = self.message.as_ref().map(|m| m.text.as_str()).unwrap_or("");
                if msg_text.is_empty() {
                    " ".to_string()
                } else {
                    msg_text.to_string()
                }
            }
            _ => String::new(),
        };

        let mut output_items: Vec<(usize, Value)> = Vec::new();

        if let Some(reasoning) = &self.reasoning {
            let text = reasoning.text.clone();
            out.push_str(&sse_event(
                "response.reasoning_summary_text.done",
                &json!({
                    "type": "response.reasoning_summary_text.done",
                    "response_id": self.resp_id, "item_id": reasoning.id,
                    "output_index": reasoning.output_index, "summary_index": 0,
                    "text": text
                }),
            ));
            out.push_str(&sse_event(
                "response.reasoning_summary_part.done",
                &json!({
                    "type": "response.reasoning_summary_part.done",
                    "response_id": self.resp_id, "item_id": reasoning.id,
                    "output_index": reasoning.output_index, "summary_index": 0,
                    "part": {"type": "summary_text", "text": text}
                }),
            ));
            let item = json!({
                "id": reasoning.id, "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}]
            });
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": reasoning.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((reasoning.output_index, item));
        }

        if let Some(message) = &self.message {
            let text = message.text.clone();
            out.push_str(&sse_event(
                "response.output_text.done",
                &json!({
                    "type": "response.output_text.done",
                    "response_id": self.resp_id, "item_id": message.id,
                    "output_index": message.output_index, "content_index": 0, "text": text
                }),
            ));
            out.push_str(&sse_event(
                "response.content_part.done",
                &json!({
                    "type": "response.content_part.done",
                    "response_id": self.resp_id, "item_id": message.id,
                    "output_index": message.output_index, "content_index": 0,
                    "part": {"type": "output_text", "text": text}
                }),
            ));
            // Reasoning stays in the standalone reasoning item; a `reasoning`
            // part in `message.content` makes Codex.app reject the message.
            let content_parts =
                vec![json!({"type": "output_text", "text": text, "annotations": []})];
            let item = json!({
                "id": message.id, "type": "message", "status": "completed",
                "role": "assistant", "content": content_parts
            });
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": message.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((message.output_index, item));
        }

        for tool in &self.tools {
            out.push_str(&sse_event(
                "response.function_call_arguments.done",
                &json!({
                    "type": "response.function_call_arguments.done",
                    "response_id": self.resp_id, "output_index": tool.output_index,
                    "item_id": tool.item_id, "arguments": tool.args
                }),
            ));
            let mut item = json!({
                "id": tool.item_id, "call_id": tool.call_id,
                "type": "function_call", "status": "completed",
                "name": tool.name, "arguments": tool.args
            });
            if !reasoning_for_tool.is_empty() {
                item["reasoning_content"] = json!(reasoning_for_tool.clone());
            }
            out.push_str(&sse_event(
                "response.output_item.done",
                &json!({
                    "type": "response.output_item.done",
                    "response_id": self.resp_id, "output_index": tool.output_index,
                    "item": item.clone()
                }),
            ));
            output_items.push((tool.output_index, item));
        }

        output_items.sort_by_key(|(idx, _)| *idx);
        let output: Vec<Value> = output_items.into_iter().map(|(_, item)| item).collect();
        let mut response = json!({
            "id": self.resp_id, "object": "response",
            "model": self.codex_model,
            "created_at": self.created_at, "status": "completed",
            "output": output
        });
        if let Some(usage) = &self.usage {
            response["usage"] = usage.clone();
        }
        out.push_str(&sse_event(
            "response.completed",
            &json!({"type": "response.completed", "response": response}),
        ));
        out
    }

    fn ensure_created(&mut self, out: &mut String) {
        if self.created_emitted {
            return;
        }
        self.created_emitted = true;
        out.push_str(&sse_event(
            "response.created",
            &json!({
                "type": "response.created",
                "response": {
                    "id": self.resp_id, "object": "response",
                    "model": self.codex_model,
                    "created_at": self.created_at, "status": "in_progress", "output": []
                }
            }),
        ));
    }

    fn process_line(&mut self, line: &str, out: &mut String) {
        let Some(data) = line.strip_prefix("data: ") else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            return;
        };
        if chunk.get("usage").is_some_and(|u| !u.is_null())
            && let Some(usage) = chat_usage_to_responses_usage(&chunk)
        {
            self.usage = Some(usage);
        }
        let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
            return;
        };
        for choice in choices {
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str())
                && !reasoning.is_empty()
            {
                self.push_reasoning_delta(reasoning, out);
            }
            if let Some(content) = delta.get("content").and_then(|v| v.as_str())
                && !content.is_empty()
            {
                self.push_content_delta(content, out);
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    self.push_tool_call_delta(tc, out);
                }
            }
        }
    }

    fn push_reasoning_delta(&mut self, delta: &str, out: &mut String) {
        if self.reasoning.is_none() {
            let id = gen_id("rs");
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            out.push_str(&sse_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "response_id": self.resp_id, "output_index": output_index,
                    "item": {"id": id, "type": "reasoning", "summary": []}
                }),
            ));
            out.push_str(&sse_event(
                "response.reasoning_summary_part.added",
                &json!({
                    "type": "response.reasoning_summary_part.added",
                    "response_id": self.resp_id, "item_id": id,
                    "output_index": output_index, "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""}
                }),
            ));
            self.reasoning = Some(StreamItem {
                id,
                output_index,
                text: String::new(),
            });
        }
        let reasoning = self.reasoning.as_mut().unwrap();
        reasoning.text.push_str(delta);
        out.push_str(&sse_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "response_id": self.resp_id, "item_id": reasoning.id,
                "output_index": reasoning.output_index, "summary_index": 0,
                "delta": delta
            }),
        ));
    }

    fn start_message(&mut self, out: &mut String) {
        if self.message.is_some() {
            return;
        }
        let id = gen_id("msg");
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        out.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": self.resp_id, "output_index": output_index,
                "item": {
                    "id": id, "type": "message",
                    "status": "in_progress", "role": "assistant", "content": []
                }
            }),
        ));
        out.push_str(&sse_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": self.resp_id, "item_id": id,
                "output_index": output_index, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        self.message = Some(StreamItem {
            id,
            output_index,
            text: String::new(),
        });
    }

    fn push_content_delta(&mut self, delta: &str, out: &mut String) {
        self.start_message(out);
        let message = self.message.as_mut().unwrap();
        message.text.push_str(delta);
        out.push_str(&sse_event(
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "response_id": self.resp_id, "item_id": message.id,
                "output_index": message.output_index, "content_index": 0, "delta": delta
            }),
        ));
    }

    fn push_tool_call_delta(&mut self, tc: &Value, out: &mut String) {
        let chat_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
        let name_fragment = tc
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str());
        let id_fragment = tc.get("id").and_then(|v| v.as_str());
        let args_fragment = tc
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str());

        let pos = self.tools.iter().position(|t| t.chat_index == chat_index);
        let pos = match pos {
            Some(p) => {
                // Late-arriving id/name fragments still update the slot.
                if let Some(id) = id_fragment.filter(|s| !s.is_empty()) {
                    self.tools[p].call_id = id.to_string();
                }
                if let Some(name) = name_fragment.filter(|s| !s.is_empty()) {
                    self.tools[p].name = name.to_string();
                }
                p
            }
            None => {
                let output_index = self.next_output_index;
                self.next_output_index += 1;
                let item_id = gen_id("fc");
                let call_id = id_fragment.filter(|s| !s.is_empty()).unwrap_or("call_0");
                let name = name_fragment.unwrap_or("");
                out.push_str(&sse_event(
                    "response.output_item.added",
                    &json!({
                        "type": "response.output_item.added",
                        "response_id": self.resp_id, "output_index": output_index,
                        "item": {
                            "id": item_id, "call_id": call_id,
                            "type": "function_call", "status": "in_progress",
                            "name": name, "arguments": ""
                        }
                    }),
                ));
                self.tools.push(StreamToolCall {
                    chat_index,
                    output_index,
                    item_id,
                    call_id: call_id.to_string(),
                    name: name.to_string(),
                    args: String::new(),
                });
                self.tools.len() - 1
            }
        };

        if let Some(args) = args_fragment.filter(|s| !s.is_empty()) {
            let tool = &mut self.tools[pos];
            tool.args.push_str(args);
            let item_id = tool.item_id.clone();
            let output_index = tool.output_index;
            out.push_str(&sse_event(
                "response.function_call_arguments.delta",
                &json!({
                    "type": "response.function_call_arguments.delta",
                    "response_id": self.resp_id, "output_index": output_index,
                    "item_id": item_id, "delta": args
                }),
            ));
        }
    }
}

fn chat_usage_to_responses_usage(chat: &Value) -> Option<Value> {
    let usage = chat.get("usage")?;

    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let total_tokens = usage.get("total_tokens").cloned().unwrap_or_else(|| {
        let input = input_tokens.as_u64().unwrap_or(0);
        let output = output_tokens.as_u64().unwrap_or(0);
        json!(input.saturating_add(output))
    });

    // Map OpenAI chat-completion's `prompt_tokens_details.cached_tokens` (and
    // Anthropic's `cache_read_input_tokens`) to the Responses API shape.
    // Some clients (recent OpenAI SDKs) crash on `usage.input_tokens_details.
    // cached_tokens` being absent, so emit a zeroed object even when the
    // upstream didn't return cache info.
    let cached_tokens = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .or_else(|| usage.get("cache_read_input_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));
    let reasoning_tokens = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(0));

    let mut response_usage = json!({
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": cached_tokens,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
        "total_tokens": total_tokens
    });

    if let Some(value) = usage.get("cache_read_input_tokens").cloned() {
        response_usage["cache_read_input_tokens"] = value;
    }
    if let Some(value) = usage.get("cache_creation_input_tokens").cloned() {
        response_usage["cache_creation_input_tokens"] = value;
    }

    Some(response_usage)
}

/// Extracts assistant text, tool calls, and reasoning content from provider chat completion payloads.
/// Handles multi-choice payloads and common non-standard envelopes.
fn extract_chat_response_payload(chat: &Value) -> (String, Vec<Value>, String) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();

    if let Some(choices) = chat.get("choices").and_then(|c| c.as_array()) {
        for choice in choices {
            let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
            let text = extract_message_text(&message);
            if !text.is_empty() {
                text_parts.push(text);
            }
            // Extract reasoning_content if present (Moonshot, etc.)
            if let Some(reasoning) = message.get("reasoning_content").and_then(|r| r.as_str())
                && !reasoning.is_empty()
            {
                reasoning_parts.push(reasoning.to_string());
            }
            if let Some(tcs) = message.get("tool_calls").and_then(|t| t.as_array()) {
                tool_calls.extend(tcs.iter().cloned());
            }
        }
    }

    // Fallback: Responses API-style output payloads from some providers.
    if text_parts.is_empty() && tool_calls.is_empty() {
        let output_items = chat
            .get("output")
            .or_else(|| chat.get("response").and_then(|r| r.get("output")))
            .and_then(|v| v.as_array());

        if let Some(items) = output_items {
            for item in items {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("message") => {
                        let text = extract_content_text(item.get("content"));
                        if !text.is_empty() {
                            text_parts.push(text);
                        }
                    }
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .or_else(|| item.get("id").and_then(|v| v.as_str()))
                            .unwrap_or("call_0");
                        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}");
                        tool_calls.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments
                            }
                        }));
                    }
                    Some("output_text") => {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Fallback envelopes seen from some OpenAI-compatible providers
    if text_parts.is_empty() {
        if let Some(text) = chat
            .get("result")
            .and_then(|r| r.get("response"))
            .and_then(|v| v.as_str())
        {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("response").and_then(|v| v.as_str()) {
            text_parts.push(text.to_string());
        } else if let Some(text) = chat.get("output_text").and_then(|v| v.as_str()) {
            text_parts.push(text.to_string());
        }
    }

    (
        text_parts.join("\n"),
        tool_calls,
        reasoning_parts.join("\n"),
    )
}

fn extract_message_text(message: &Value) -> String {
    extract_content_text(message.get("content"))
}

fn sse_event(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_default()
    )
}

/// Generates a unique ID using an atomic counter + timestamp
fn gen_id(prefix: &str) -> String {
    let n = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}_{:06}", prefix, current_unix_ts(), n % 1_000_000)
}

// =============================================================================
// CHAT COMPLETIONS → RESPONSES API CONVERSION
// =============================================================================

/// Converts an OpenAI Chat Completions request body to a Responses API request body.
/// Delegates to the typed converter in `openai_models` to avoid duplicating conversion logic.
pub fn convert_chat_to_responses_request(body: &Value) -> Value {
    let Ok(typed): Result<OpenAIChatRequest, _> = serde_json::from_value(body.clone()) else {
        return json!({"model": "gpt-4o", "input": [], "stream": false});
    };
    let mut resp = serde_json::to_value(convert_typed_chat_to_responses(&typed))
        .unwrap_or_else(|_| json!({"model": "gpt-4o", "input": [], "stream": false}));
    // Force non-streaming for the fallback path
    resp["stream"] = json!(false);
    resp
}

/// Converts a Responses API JSON response to Chat Completions format.
/// Delegates to the typed converter in `openai_models` to avoid duplicating conversion logic.
pub fn convert_responses_json_to_chat(resp: &Value) -> Value {
    // Handle both direct response and wrapped {"response": ...} format
    let inner = resp
        .get("response")
        .filter(|r| r.is_object())
        .unwrap_or(resp);

    let Ok(typed): Result<ResponsesResponse, _> = serde_json::from_value(inner.clone()) else {
        return json!({"choices": [], "usage": {}});
    };
    serde_json::to_value(convert_typed_responses_to_chat(&typed))
        .unwrap_or_else(|_| json!({"choices": [], "usage": {}}))
}

/// Streaming inverse of [`ResponsesStreamConverter`]: feeds upstream Responses
/// API SSE and emits Chat Completions SSE, so a Chat Completions client (e.g.
/// omp) can drive a model that only accepts `/v1/responses` (gpt-5.x with
/// reasoning + tools) and still get incremental tokens. Buffers partial lines.
pub struct ResponsesToChatStreamConverter {
    pending: Vec<u8>,
    id: String,
    created: u64,
    model: String,
    include_usage: bool,
    role_emitted: bool,
    finished: bool,
    /// function_call item_id → its chat `tool_calls` array index.
    tool_index: std::collections::HashMap<String, u64>,
    next_tool_index: u64,
    finish_reason: &'static str,
    usage: Option<Value>,
}

impl ResponsesToChatStreamConverter {
    pub fn new(original_model: &str, include_usage: bool) -> Self {
        Self {
            pending: Vec::new(),
            id: gen_id("chatcmpl"),
            created: current_unix_ts(),
            model: original_model.to_string(),
            include_usage,
            role_emitted: false,
            finished: false,
            tool_index: std::collections::HashMap::new(),
            next_tool_index: 0,
            finish_reason: "stop",
            usage: None,
        }
    }

    /// Feed a network chunk of the upstream Responses SSE; returns Chat
    /// Completions SSE to forward. Partial trailing lines buffer across calls.
    pub fn push_bytes(&mut self, chunk: &[u8]) -> String {
        let mut out = String::new();
        self.pending.extend_from_slice(chunk);
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.pending[..pos]).into_owned();
            self.pending.drain(..=pos);
            self.process_line(line.trim_end_matches('\r'), &mut out);
        }
        // OOM backstop: a newline-less upstream can't be converted anyway;
        // drop the oversized partial line instead of buffering it forever.
        if self.pending.len() > crate::services::http_utils::MAX_SSE_PENDING_BYTES {
            self.pending = Vec::new();
        }
        out
    }

    /// Flush any buffered line, then emit the terminal `finish_reason` chunk, an
    /// optional usage-only chunk, and `data: [DONE]`.
    pub fn finish(&mut self) -> String {
        let mut out = String::new();
        if !self.pending.is_empty() {
            let line = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.process_line(line.trim_end_matches('\r'), &mut out);
        }
        if self.finished {
            return out;
        }
        self.finished = true;
        out.push_str(&self.chunk(json!({}), Some(self.finish_reason)));
        if self.include_usage
            && let Some(usage) = &self.usage
        {
            out.push_str(&data_line(&json!({
                "id": self.id, "object": "chat.completion.chunk",
                "created": self.created, "model": self.model,
                "choices": [], "usage": usage,
            })));
        }
        out.push_str("data: [DONE]\n\n");
        out
    }

    fn process_line(&mut self, line: &str, out: &mut String) {
        let Some(data) = line.strip_prefix("data: ") else {
            return;
        };
        if data == "[DONE]" {
            return;
        }
        let Ok(ev) = serde_json::from_str::<Value>(data) else {
            return;
        };
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "response.output_text.delta" => {
                if let Some(d) = ev
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    self.emit_delta(json!({ "content": d }), out);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                if let Some(d) = ev
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    self.emit_delta(json!({ "reasoning_content": d }), out);
                }
            }
            "response.output_item.added" => {
                let item = ev.get("item");
                if item.and_then(|i| i.get("type")).and_then(|t| t.as_str())
                    == Some("function_call")
                {
                    self.start_tool_call(item.unwrap(), out);
                }
            }
            "response.function_call_arguments.delta" => {
                let item_id = ev.get("item_id").and_then(|v| v.as_str()).unwrap_or("");
                if let (Some(&idx), Some(d)) = (
                    self.tool_index.get(item_id),
                    ev.get("delta").and_then(|v| v.as_str()),
                ) {
                    self.emit_delta(
                        json!({ "tool_calls": [{ "index": idx, "function": { "arguments": d } }] }),
                        out,
                    );
                }
            }
            "response.completed" => {
                self.usage = ev
                    .get("response")
                    .and_then(|r| r.get("usage"))
                    .filter(|u| !u.is_null())
                    .map(responses_usage_to_chat_usage);
            }
            _ => {}
        }
    }

    fn start_tool_call(&mut self, item: &Value, out: &mut String) {
        let item_id = item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let call_id = item
            .get("call_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| item.get("id").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = item.get("arguments").and_then(|v| v.as_str()).unwrap_or("");
        let idx = self.next_tool_index;
        self.next_tool_index += 1;
        self.tool_index.insert(item_id, idx);
        self.finish_reason = "tool_calls";
        self.emit_delta(
            json!({ "tool_calls": [{
                "index": idx, "id": call_id, "type": "function",
                "function": { "name": name, "arguments": args }
            }] }),
            out,
        );
    }

    /// Emit one chat chunk carrying `delta`, prefixing the assistant role on the
    /// first chunk (mirrors OpenAI's stream).
    fn emit_delta(&mut self, mut delta: Value, out: &mut String) {
        if !self.role_emitted {
            self.role_emitted = true;
            if let Some(obj) = delta.as_object_mut() {
                obj.insert("role".to_string(), json!("assistant"));
            }
        }
        out.push_str(&self.chunk(delta, None));
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        data_line(&json!({
            "id": self.id, "object": "chat.completion.chunk",
            "created": self.created, "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        }))
    }
}

fn data_line(v: &Value) -> String {
    format!("data: {}\n\n", serde_json::to_string(v).unwrap_or_default())
}

/// Map a Responses API `usage` object to the Chat Completions shape.
fn responses_usage_to_chat_usage(usage: &Value) -> Value {
    let num = |obj: &Value, key: &str| obj.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
    let input = num(usage, "input_tokens");
    let output = num(usage, "output_tokens");
    let total = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| input.saturating_add(output));
    let cached = usage
        .get("input_tokens_details")
        .map(|d| num(d, "cached_tokens"))
        .unwrap_or(0);
    let reasoning = usage
        .get("output_tokens_details")
        .map(|d| num(d, "reasoning_tokens"))
        .unwrap_or(0);
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": total,
        "prompt_tokens_details": { "cached_tokens": cached },
        "completion_tokens_details": { "reasoning_tokens": reasoning },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── is_responses_api_format ────────────────────────────────────────────────

    #[test]
    fn test_is_responses_api_format_detected() {
        assert!(is_responses_api_format(
            &json!({"input": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_chat_completions_not_detected() {
        assert!(!is_responses_api_format(
            &json!({"messages": [{"role": "user", "content": "hi"}]})
        ));
    }

    #[test]
    fn test_is_responses_api_format_has_both_not_detected() {
        // If both "input" and "messages" present, treat as Chat Completions
        assert!(!is_responses_api_format(&json!({
            "input": [],
            "messages": []
        })));
    }

    // ── extract_content_text ───────────────────────────────────────────────────

    #[test]
    fn test_extract_content_text_string() {
        assert_eq!(
            extract_content_text(Some(&json!("hello world"))),
            "hello world"
        );
    }

    #[test]
    fn test_extract_content_text_parts_array() {
        let content = json!([
            {"type": "input_text", "text": "list"},
            {"type": "input_text", "text": "files"}
        ]);
        assert_eq!(extract_content_text(Some(&content)), "list\nfiles");
    }

    #[test]
    fn test_extract_content_text_none() {
        assert_eq!(extract_content_text(None), "");
    }

    #[test]
    fn test_extract_content_text_empty_array() {
        assert_eq!(extract_content_text(Some(&json!([]))), "");
    }

    #[test]
    fn test_extract_content_text_null() {
        assert_eq!(extract_content_text(Some(&json!(null))), "");
    }

    // ── convert_tool_to_chat_format ────────────────────────────────────────────

    #[test]
    fn test_convert_tool_format_responses_api_to_chat() {
        let tool = json!({
            "type": "function",
            "name": "shell",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {}}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["type"], "function");
        assert_eq!(converted["function"]["name"], "shell");
        assert_eq!(converted["function"]["description"], "Run a shell command");
        assert!(converted.get("name").is_none()); // moved into "function" wrapper
    }

    #[test]
    fn test_convert_tool_format_already_chat_format() {
        let tool = json!({
            "type": "function",
            "function": {"name": "shell", "description": "..."}
        });
        let converted = convert_tool_to_chat_format(&tool);
        assert_eq!(converted["function"]["name"], "shell");
    }

    // ── convert_responses_to_chat_request ─────────────────────────────────────

    #[test]
    fn test_convert_request_simple_message() {
        let body = json!({
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "list files"}]}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://ai-gateway.vercel.sh/v1".to_string(),
                api_key: "sk-test".to_string(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );

        assert_eq!(chat["model"], "gpt-5.2-codex");
        assert_eq!(chat["stream"], false);
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "list files");
    }

    #[test]
    fn test_convert_request_instructions_become_system_message() {
        let body = json!({
            "model": "gpt-4",
            "instructions": "You are a helpful assistant.",
            "input": [{"type": "message", "role": "user", "content": "hi"}]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn test_convert_request_tool_call_items() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "list files"},
                {"type": "function_call", "id": "fc_item_1", "call_id": "call_abc", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_abc", "output": "file1.txt\nfile2.txt"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["tool_calls"][0]["id"], "call_abc");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_abc");
        assert_eq!(msgs[2]["content"], "file1.txt\nfile2.txt");
    }

    #[test]
    fn test_convert_request_parallel_function_calls_coalesce_into_one_assistant_message() {
        // Codex emits parallel tool calls back as multiple consecutive
        // `function_call` items followed by their `function_call_output`s.
        // Chat Completions requires a single assistant message carrying all
        // parallel tool_calls, immediately followed by one tool message per
        // tool_call_id — otherwise OpenAI strict validators reject with
        // "An assistant message with 'tool_calls' must be followed by tool
        // messages responding to each 'tool_call_id'."
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "message", "role": "user", "content": "do two things"},
                {"type": "function_call", "call_id": "call_a", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call", "call_id": "call_b", "name": "shell", "arguments": "{\"cmd\":\"pwd\"}"},
                {"type": "function_call_output", "call_id": "call_a", "output": "files"},
                {"type": "function_call_output", "call_id": "call_b", "output": "/tmp"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 4, "user + 1 assistant + 2 tool messages");
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        let tool_calls = msgs[1]["tool_calls"].as_array().unwrap();
        assert_eq!(
            tool_calls.len(),
            2,
            "parallel calls share one assistant msg"
        );
        assert_eq!(tool_calls[0]["id"], "call_a");
        assert_eq!(tool_calls[1]["id"], "call_b");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_a");
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["tool_call_id"], "call_b");
    }

    fn default_test_config() -> ResponsesToChatRouterConfig {
        ResponsesToChatRouterConfig {
            target_base_url: "https://example.com/v1".to_string(),
            api_key: String::new(),
            target_protocol: ProviderProtocol::Openai,
            target_path_variant: None,
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
            aivo_prefix_models: Vec::new(),
        }
    }

    #[test]
    fn test_convert_request_reasoning_item_attaches_to_following_function_call() {
        // Codex emits `type:"reasoning"` items immediately before the
        // function_call they belong to. The converter must lift the summary
        // text onto the assistant tool_call message as `reasoning_content`,
        // or deepseek-thinking 400s with "must be passed back to the API".
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "step by step plan"}]
                },
                {"type": "function_call", "call_id": "call_x", "name": "shell", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_x");
        assert_eq!(msgs[0]["reasoning_content"], "step by step plan");
    }

    #[test]
    fn test_convert_request_strips_sampling_for_rejecting_models() {
        // o3 rejects temperature/top_p — forwarding them 400s upstream.
        let body = json!({
            "model": "o3",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "top_p": 0.9
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        assert!(chat.get("temperature").is_none());
        assert!(chat.get("top_p").is_none());

        // Normal models keep them.
        let body = json!({
            "model": "deepseek-chat",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.2
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        assert_eq!(chat["temperature"], 0.2);
    }

    #[test]
    fn test_convert_request_reasoning_item_attaches_to_following_assistant_message() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "reasoned"}]
                },
                {"type": "message", "role": "assistant", "content": "final answer"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], "final answer");
        assert_eq!(msgs[0]["reasoning_content"], "reasoned");
    }

    #[test]
    fn test_convert_request_multiple_reasoning_items_join_with_newline() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "first\nsecond");
    }

    #[test]
    fn test_convert_request_reasoning_only_attaches_to_first_following_turn() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "trace"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                {"type": "message", "role": "assistant", "content": "follow up"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        // First assistant turn carries reasoning; later assistant turn must not inherit it.
        assert_eq!(msgs[0]["reasoning_content"], "trace");
        assert_eq!(msgs[2]["role"], "assistant");
        assert!(msgs[2].get("reasoning_content").is_none());
    }

    #[test]
    fn test_convert_request_reasoning_falls_back_to_content_array() {
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {
                    "type": "reasoning",
                    "content": [{"type": "reasoning_text", "text": "fallback"}]
                },
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "fallback");
    }

    #[test]
    fn test_convert_request_reasoning_attaches_to_first_of_parallel_tool_calls() {
        // Parallel function_calls coalesce into one assistant message. The
        // buffered reasoning attaches to the coalesced message via the first
        // function_call only — subsequent appends must not overwrite it.
        let body = json!({
            "model": "deepseek-reasoner",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "shared trace"}]},
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1);
        let tool_calls = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(msgs[0]["reasoning_content"], "shared trace");
    }

    #[test]
    fn test_convert_request_merges_assistant_text_alongside_tool_call() {
        // Codex replays a content+tool_calls turn as separate message and
        // function_call items; they must re-merge so strict upstreams accept it.
        let body = json!({
            "model": "deepseek-thinking",
            "input": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                {"type": "message", "role": "assistant", "content": "Checking that file."},
                {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "one assistant turn + one tool result");
        assert_eq!(msgs[0]["role"], "assistant");
        assert_eq!(msgs[0]["content"], "Checking that file.");
        assert_eq!(msgs[0]["reasoning_content"], "why");
        let tool_calls = msgs[0]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "c1");
        assert_eq!(msgs[1]["role"], "tool");
        assert_eq!(msgs[1]["tool_call_id"], "c1");
    }

    #[test]
    fn test_convert_request_does_not_coalesce_tool_calls_across_user_boundary() {
        // A user message between two tool calls closes the first assistant turn;
        // the second call must start a fresh assistant message.
        let body = json!({
            "model": "deepseek-thinking",
            "input": [
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "a", "output": "r"},
                {"type": "message", "role": "user", "content": "next"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"}
            ]
        });
        let chat = convert_responses_to_chat_request(&body, &default_test_config());
        let msgs = chat["messages"].as_array().unwrap();
        let assistants: Vec<_> = msgs.iter().filter(|m| m["role"] == "assistant").collect();
        assert_eq!(assistants.len(), 2);
        assert_eq!(assistants[0]["tool_calls"][0]["id"], "a");
        assert_eq!(assistants[1]["tool_calls"][0]["id"], "b");
    }

    #[test]
    fn test_convert_response_sse_emits_standard_reasoning_item_before_tool_calls() {
        // Codex CLI parses output items with typed structs; reasoning must
        // travel as a standalone `type:"reasoning"` item so it survives the
        // round-trip. The legacy `function_call.reasoning_content` field is
        // still emitted (Codex ignores it), driving aivo's own self-bridged
        // path when no separate reasoning item is read.
        let chat = json!({
            "id": "chatcmpl-r",
            "choices": [{
                "message": {
                    "content": null,
                    "reasoning_content": "let me think...",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "run", "arguments": "{}"}
                    }]
                }
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });

        let sse = convert_chat_response_to_responses_sse(&chat, false, "deepseek-reasoner");

        // The first `output_item.added` is a reasoning item with summary text,
        // at output_index 0. function_call comes after at output_index 1.
        assert!(
            sse.contains("\"type\":\"reasoning\""),
            "expected standalone reasoning item in SSE stream"
        );
        assert!(sse.contains("response.reasoning_summary_text.delta"));
        assert!(sse.contains("response.reasoning_summary_text.done"));
        assert!(sse.contains("\"text\":\"let me think...\""));

        // function_call follows at output_index 1
        assert!(
            sse.contains("\"output_index\":1") && sse.contains("\"type\":\"function_call\""),
            "function_call must appear at output_index 1 after the reasoning item"
        );
    }

    #[test]
    fn test_convert_response_sse_no_reasoning_item_when_empty() {
        let chat = json!({
            "id": "chatcmpl-nr",
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4");
        assert!(
            !sse.contains("response.reasoning_summary_text"),
            "no reasoning events expected when message has no reasoning_content"
        );
        // Text message item still appears at output_index 0
        assert!(sse.contains("\"output_index\":0") && sse.contains("\"type\":\"message\""));
    }

    #[test]
    fn test_convert_response_sse_message_content_excludes_reasoning_part() {
        // A `reasoning` part inside message.content makes Codex.app drop the
        // whole message; reasoning must only ride the standalone reasoning item.
        let chat = json!({
            "choices": [{"message": {
                "content": "Once upon a time.",
                "reasoning_content": "the user wants a story"
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "deepseek-reasoner");
        assert!(sse.contains("response.reasoning_summary_text.delta"));

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        let message = output.iter().find(|i| i["type"] == "message").unwrap();
        let content = message["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "output_text");
        assert!(!content.iter().any(|p| p["type"] == "reasoning"));
    }

    #[test]
    fn test_convert_request_function_call_without_call_id_falls_back_to_id() {
        let body = json!({
            "model": "gpt-4",
            "input": [
                {"type": "function_call", "id": "call_legacy", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_legacy", "output": "ok"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["tool_calls"][0]["id"], "call_legacy");
        assert_eq!(msgs[1]["tool_call_id"], "call_legacy");
    }

    #[test]
    fn test_convert_request_filters_non_function_tools() {
        let body = json!({
            "model": "gpt-4",
            "input": [],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {}},
                {"type": "computer_use"},
                {"type": "web_search"}
            ]
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let tools = chat["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "shell");
    }

    #[test]
    fn test_convert_request_openrouter_transforms_model() {
        let body = json!({"model": "gpt-5.2-codex", "input": []});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        assert_eq!(chat["model"], "openai/gpt-5.2-codex");
    }

    #[test]
    fn test_convert_request_caps_max_output_tokens() {
        let body = json!({
            "model": "gpt-4o",
            "input": [],
            "max_output_tokens": 12000
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        assert_eq!(chat["max_tokens"], 8192);
    }

    #[test]
    fn test_convert_request_caps_string_max_output_tokens() {
        let body = json!({
            "model": "gpt-4o",
            "input": [],
            "max_output_tokens": "12000"
        });
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        assert_eq!(chat["max_tokens"], 8192);
    }

    #[test]
    fn test_apply_max_tokens_cap_to_fields_caps_chat_completions_fields() {
        let mut body = json!({
            "max_tokens": 10000,
            "max_output_tokens": 9000
        });
        apply_max_tokens_cap_to_fields(&mut body, Some(8192), &["max_tokens", "max_output_tokens"]);
        assert_eq!(body["max_tokens"], 8192);
        assert_eq!(body["max_output_tokens"], 8192);
    }

    #[test]
    fn test_apply_max_tokens_cap_to_fields_caps_numeric_string_fields() {
        let mut body = json!({
            "max_tokens": "10000",
            "max_output_tokens": "9000"
        });
        apply_max_tokens_cap_to_fields(&mut body, Some(8192), &["max_tokens", "max_output_tokens"]);
        assert_eq!(body["max_tokens"], 8192);
        assert_eq!(body["max_output_tokens"], 8192);
    }

    // ── convert_chat_response_to_responses_sse ─────────────────────────────────

    #[test]
    fn test_convert_response_text_contains_required_events() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 90
            },
            "choices": [{"message": {"role": "assistant", "content": "Here are your files."}}]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("event: response.created\n"));
        assert!(sse.contains("event: response.output_text.delta\n"));
        assert!(sse.contains("event: response.output_text.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("Here are your files."));
        assert!(sse.contains("\"cache_read_input_tokens\":90"));
    }

    #[test]
    fn test_convert_response_tool_call_contains_required_events() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("event: response.output_item.added\n"));
        assert!(sse.contains("event: response.function_call_arguments.delta\n"));
        assert!(sse.contains("event: response.function_call_arguments.done\n"));
        assert!(sse.contains("event: response.output_item.done\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("call_abc"));
        assert!(sse.contains("shell"));
    }

    #[test]
    fn test_convert_response_empty_content_no_delta_event() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": ""}}]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(!sse.contains("response.output_text.delta"));
        assert!(sse.contains("response.output_text.done"));
    }

    #[test]
    fn test_convert_response_joins_text_from_multiple_choices() {
        let chat = json!({
            "choices": [
                {"message": {"role": "assistant", "content": "Hello"}},
                {"message": {"role": "assistant", "content": "world"}}
            ]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("Hello\\nworld"));
    }

    #[test]
    fn test_convert_response_supports_content_array_parts() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]
                }
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("Hello\\nworld"));
    }

    #[test]
    fn test_convert_response_supports_result_response_envelope() {
        let chat = json!({
            "result": {"response": "Hello from envelope"}
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("Hello from envelope"));
    }

    #[test]
    fn test_convert_response_supports_responses_output_message() {
        let chat = json!({
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello from output"}]
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("Hello from output"));
    }

    #[test]
    fn test_convert_response_supports_responses_output_function_call() {
        let chat = json!({
            "response": {
                "output": [{
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"ls\"}"
                }]
            }
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("\"call_id\":\"call_123\""));
        assert!(sse.contains("\"name\":\"shell\""));
    }

    #[test]
    fn test_convert_response_uses_correct_object_type() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("\"object\":\"response\""));
        assert!(!sse.contains("realtime.response"));
    }

    #[test]
    fn test_convert_response_includes_response_id() {
        let chat = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("\"response_id\""));
    }

    #[test]
    fn test_convert_response_tool_call_has_call_id() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{"id": "call_abc123", "type": "function",
                                    "function": {"name": "shell", "arguments": "{}"}}]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("\"call_id\":\"call_abc123\""));
    }

    // ── SSE accumulator ────────────────────────────────────────────────────────

    #[test]
    fn test_accumulate_chat_sse_text_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_accumulate_chat_sse_tool_call_response() {
        let sse = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        let tcs = result["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs[0]["id"], "call_x");
        assert_eq!(tcs[0]["function"]["name"], "shell");
        assert!(
            tcs[0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("ls")
        );
    }

    #[test]
    fn test_parse_provider_response_json() {
        let json_text = r#"{"choices":[{"message":{"role":"assistant","content":"hi"}}]}"#;
        let result = parse_provider_response(json_text).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_parse_provider_response_sse_fallback() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\ndata: [DONE]\n";
        let result = parse_provider_response(sse).unwrap();
        assert_eq!(result["choices"][0]["message"]["content"], "hi");
    }

    #[test]
    fn test_convert_request_copilot_skips_model_transform() {
        let body = json!({"model": "gpt-4o", "input": []});
        let config = ResponsesToChatRouterConfig {
            target_base_url: String::new(),
            api_key: String::new(),
            target_protocol: ProviderProtocol::Openai,
            target_path_variant: None,
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
            aivo_prefix_models: Vec::new(),
        };
        let chat = convert_responses_to_chat_request(&body, &config);
        assert_eq!(chat["model"], "gpt-4o");
    }

    #[test]
    fn test_convert_response_sse_empty_choices_no_panic() {
        let chat = json!({"choices": []});
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_response_sse_missing_choices_no_panic() {
        let chat = json!({});
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_request_missing_model_uses_default() {
        let body = json!({"input": [{"type": "message", "role": "user", "content": "hi"}]});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        assert!(chat.get("model").is_some());
    }

    #[test]
    fn test_convert_request_empty_input() {
        let body = json!({"model": "gpt-4o", "input": []});
        let chat = convert_responses_to_chat_request(
            &body,
            &ResponsesToChatRouterConfig {
                target_base_url: "https://example.com/v1".to_string(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            },
        );
        let msgs = chat["messages"].as_array().unwrap();
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_extract_chat_response_payload_null_message() {
        let chat = json!({"choices": [{"message": null}]});
        let (text, tool_calls, reasoning) = extract_chat_response_payload(&chat);
        assert!(text.is_empty());
        assert!(tool_calls.is_empty());
        assert!(reasoning.is_empty());
    }

    #[test]
    fn test_extract_chat_response_payload_output_text_item() {
        let chat = json!({
            "output": [{"type": "output_text", "text": "hello from output_text"}]
        });
        let (text, tool_calls, _) = extract_chat_response_payload(&chat);
        assert_eq!(text, "hello from output_text");
        assert!(tool_calls.is_empty());
    }

    #[test]
    fn test_accumulate_chat_sse_empty_input() {
        let result = accumulate_chat_sse("");
        assert!(
            result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn test_accumulate_chat_sse_only_done() {
        let result = accumulate_chat_sse("data: [DONE]\n");
        assert!(
            result["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .is_empty()
        );
    }

    #[test]
    fn test_parse_provider_response_empty_string() {
        let result = parse_provider_response("");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_provider_response_malformed_json() {
        let result = parse_provider_response("{not valid json}");
        assert!(result.is_err() || result.unwrap().is_object());
    }

    #[test]
    fn test_chat_usage_to_responses_usage_missing() {
        let chat = json!({"choices": []});
        assert!(chat_usage_to_responses_usage(&chat).is_none());
    }

    #[test]
    fn test_chat_usage_to_responses_usage_present() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 8
            }
        });
        let usage = chat_usage_to_responses_usage(&chat).unwrap();
        assert_eq!(usage["input_tokens"], 10);
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["cache_read_input_tokens"], 8);
    }

    #[test]
    fn test_accumulate_chat_sse_malformed_json_skipped() {
        let sse = "data: {invalid json!!!}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\
                   data: not even close to json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\
                   data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\
                   data: [DONE]\n";
        let result = accumulate_chat_sse(sse);
        assert_eq!(result["choices"][0]["message"]["content"], "Hello world");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn test_convert_chat_response_to_responses_sse_null_usage() {
        let chat = json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": null,
                "total_tokens": null
            }
        });
        let sse = convert_chat_response_to_responses_sse(&chat, false, "gpt-4o");
        assert!(sse.contains("event: response.created\n"));
        assert!(sse.contains("event: response.completed\n"));
        assert!(sse.contains("\"input_tokens\""));
        assert!(sse.contains("\"output_tokens\""));
        assert!(sse.contains("hi"));
    }

    #[test]
    fn test_convert_responses_to_chat_actual_model_override() {
        let body = json!({
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "hello"}]
        });
        let config = ResponsesToChatRouterConfig {
            target_base_url: "https://example.com/v1".to_string(),
            api_key: String::new(),
            target_protocol: ProviderProtocol::Openai,
            target_path_variant: None,
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: Some("kimi-k2.5".to_string()),
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
            aivo_prefix_models: Vec::new(),
        };
        let chat = convert_responses_to_chat_request(&body, &config);
        assert_eq!(chat["model"], "kimi-k2.5");
    }

    #[test]
    fn test_extract_chat_response_payload_no_choices_no_output() {
        let chat = json!({"id": "chatcmpl-123", "object": "chat.completion"});
        let (text, tool_calls, reasoning) = extract_chat_response_payload(&chat);
        assert!(
            text.is_empty(),
            "text should be empty when no choices/output"
        );
        assert!(
            tool_calls.is_empty(),
            "tool_calls should be empty when no choices/output"
        );
        assert!(
            reasoning.is_empty(),
            "reasoning should be empty when no choices/output"
        );
    }

    #[test]
    fn test_chat_usage_to_responses_usage_null_tokens() {
        let chat = json!({
            "usage": {
                "prompt_tokens": null,
                "completion_tokens": 5,
                "total_tokens": 5
            }
        });
        let usage = chat_usage_to_responses_usage(&chat).expect("usage should be Some");
        assert!(usage["input_tokens"].is_null());
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["total_tokens"], 5);
    }

    // ── convert_chat_to_responses_request ─────────────────────────────────────

    #[test]
    fn chat_to_responses_simple_message() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024
        });
        let req = convert_chat_to_responses_request(&body);
        assert_eq!(req["model"], "gpt-4o");
        assert_eq!(req["instructions"], "You are helpful.");
        assert_eq!(req["max_output_tokens"], 1024);
        assert_eq!(req["stream"], false);
        let input = req["input"].as_array().unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
    }

    #[test]
    fn chat_to_responses_tool_calls() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"loc\":\"SF\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"}
            ]
        });
        let req = convert_chat_to_responses_request(&body);
        let input = req["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_1");
        assert_eq!(input[1]["name"], "get_weather");
        assert_eq!(input[2]["type"], "function_call_output");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["output"], "Sunny");
    }

    // ── cap_reasoning_effort ──────────────────────────────────────────────────

    #[test]
    fn cap_reasoning_effort_clamps_xhigh_chat() {
        let mut body = json!({"reasoning_effort": "xhigh"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_clamps_xhigh_responses() {
        let mut body = json!({"reasoning": {"effort": "xhigh"}});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_passes_through_high() {
        let mut body = json!({"reasoning_effort": "high"});
        cap_reasoning_effort(&mut body);
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[test]
    fn cap_reasoning_effort_noop_when_absent() {
        let mut body = json!({"model": "x"});
        cap_reasoning_effort(&mut body);
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn chat_to_responses_tools_converted() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "shell", "description": "Run cmd", "parameters": {}}}
            ]
        });
        let req = convert_chat_to_responses_request(&body);
        let tools = req["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "shell");
    }

    // ── convert_responses_json_to_chat ────────────────────────────────────────

    #[test]
    fn responses_json_to_chat_text() {
        let resp = json!({
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello!"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["id"], "resp_123");
        assert_eq!(chat["model"], "gpt-4o");
        assert_eq!(chat["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(chat["choices"][0]["finish_reason"], "stop");
        assert_eq!(chat["usage"]["prompt_tokens"], 10);
        assert_eq!(chat["usage"]["completion_tokens"], 5);
    }

    #[test]
    fn responses_json_to_chat_tool_calls() {
        let resp = json!({
            "id": "resp_456",
            "model": "gpt-4o",
            "output": [
                {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{\"path\":\"test.rs\"}"}
            ]
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["choices"][0]["finish_reason"], "tool_calls");
        let tcs = chat["choices"][0]["message"]["tool_calls"]
            .as_array()
            .unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["id"], "c1");
        assert_eq!(tcs[0]["function"]["name"], "read_file");
    }

    #[test]
    fn responses_json_to_chat_wrapped_response() {
        let resp = json!({
            "response": {
                "id": "resp_789",
                "model": "gpt-4o",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}
                ]
            }
        });
        let chat = convert_responses_json_to_chat(&resp);
        assert_eq!(chat["id"], "resp_789");
        assert_eq!(chat["choices"][0]["message"]["content"], "Hi");
    }

    #[test]
    fn responses_content_to_chat_text_only_collapses_to_string() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "world"}
        ])));
        assert_eq!(v, Value::String("hello\nworld".to_string()));
    }

    #[test]
    fn responses_content_to_chat_string_passthrough() {
        let v = convert_responses_content_to_chat(Some(&json!("plain string")));
        assert_eq!(v, Value::String("plain string".to_string()));
    }

    #[test]
    fn responses_content_to_chat_input_image_data_uri_preserved() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "what is this?"},
            {"type": "input_image", "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgo=",
                "detail": "high"
            }}
        ])));
        let arr = v.as_array().expect("array shape when image present");
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "what is this?");
        assert_eq!(arr[1]["type"], "image_url");
        assert_eq!(
            arr[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo="
        );
        assert_eq!(arr[1]["image_url"]["detail"], "high");
    }

    #[test]
    fn responses_content_to_chat_input_image_string_url_accepted() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_image", "image_url": "https://example.com/x.jpg"}
        ])));
        let arr = v.as_array().unwrap();
        assert_eq!(arr[0]["type"], "image_url");
        assert_eq!(arr[0]["image_url"]["url"], "https://example.com/x.jpg");
    }

    #[test]
    fn responses_content_to_chat_input_file_inlined_as_text_reference() {
        let v = convert_responses_content_to_chat(Some(&json!([
            {"type": "input_text", "text": "look at this:"},
            {"type": "input_file", "filename": "report.pdf"}
        ])));
        // Both parts are text after conversion (file collapses to a text
        // reference) so the output collapses to a single string.
        assert_eq!(
            v,
            Value::String("look at this:\n[attached file: report.pdf]".to_string())
        );
    }

    // ── ResponsesStreamConverter ───────────────────────────────────────────────

    /// Collects every `event:` line emitted across the chunks + finish, in order.
    fn collect_events(sse: &str) -> Vec<String> {
        sse.lines()
            .filter_map(|l| l.strip_prefix("event: ").map(str::to_string))
            .collect()
    }

    fn chat_chunk_line(delta: Value) -> Vec<u8> {
        format!(
            "data: {}\n\n",
            json!({"choices": [{"index": 0, "delta": delta}]})
        )
        .into_bytes()
    }

    #[test]
    fn stream_converter_emits_reasoning_then_text_in_order() {
        let mut c = ResponsesStreamConverter::new("deepseek-reasoner", false);
        let mut sse = String::new();
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({"reasoning_content": "thin"}))));
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({"reasoning_content": "king"}))));
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({"content": "Hel"}))));
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({"content": "lo"}))));
        sse.push_str(&c.finish());

        let events = collect_events(&sse);
        // Opening event first, completed last.
        assert_eq!(events.first().unwrap(), "response.created");
        assert_eq!(events.last().unwrap(), "response.completed");
        // Reasoning item is opened before the message item.
        let reasoning_added = events
            .iter()
            .position(|e| e == "response.output_item.added")
            .unwrap();
        let first_text_delta = events
            .iter()
            .position(|e| e == "response.output_text.delta")
            .unwrap();
        let reasoning_delta = events
            .iter()
            .position(|e| e == "response.reasoning_summary_text.delta")
            .unwrap();
        assert!(reasoning_added < reasoning_delta);
        assert!(reasoning_delta < first_text_delta);

        // Deltas are streamed (two of each), not collapsed into one blob.
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.reasoning_summary_text.delta")
                .count(),
            2
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.output_text.delta")
                .count(),
            2
        );

        // response.completed carries the assembled text + reasoning items.
        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["summary"][0]["text"], "thinking");
        assert_eq!(output[1]["type"], "message");
        // Message content is output_text only — a reasoning part would make
        // Codex.app drop the message.
        let content = output[1]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["text"], "Hello");
        assert!(!content.iter().any(|p| p["type"] == "reasoning"));
    }

    #[test]
    fn stream_converter_streams_tool_call_arguments_incrementally() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let mut sse = String::new();
        // First fragment carries id + name; later fragments only arguments.
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({
            "tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"ci"}}]
        }))));
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({
            "tool_calls": [{"index": 0, "function": {"arguments": "ty\":\"SF\"}"}}]
        }))));
        sse.push_str(&c.finish());

        let events = collect_events(&sse);
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.function_call_arguments.delta")
                .count(),
            2,
            "argument fragments should stream as separate deltas"
        );
        // Exactly one function_call item opened.
        assert_eq!(
            events
                .iter()
                .filter(|e| *e == "response.output_item.added")
                .count(),
            1
        );

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let item = &completed["response"]["output"][0];
        assert_eq!(item["type"], "function_call");
        assert_eq!(item["call_id"], "call_abc");
        assert_eq!(item["name"], "get_weather");
        assert_eq!(item["arguments"], "{\"city\":\"SF\"}");
    }

    #[test]
    fn stream_converter_handles_split_data_lines_across_chunks() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let line = chat_chunk_line(json!({"content": "hi"}));
        // Split the SSE line mid-way to exercise the pending-buffer reassembly.
        let (a, b) = line.split_at(line.len() / 2);
        let mut sse = String::new();
        sse.push_str(&c.push_bytes(a));
        sse.push_str(&c.push_bytes(b));
        sse.push_str(&c.finish());

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        assert_eq!(
            completed["response"]["output"][0]["content"][0]["text"],
            "hi"
        );
    }

    #[test]
    fn stream_converter_maps_usage_into_completed_event() {
        let mut c = ResponsesStreamConverter::new("deepseek-chat", false);
        let mut sse = String::new();
        sse.push_str(&c.push_bytes(&chat_chunk_line(json!({"content": "x"}))));
        // Trailing usage-only chunk (stream_options.include_usage).
        sse.push_str(&c.push_bytes(
            format!(
                "data: {}\n\n",
                json!({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
            )
            .as_bytes(),
        ));
        sse.push_str(&c.push_bytes(b"data: [DONE]\n\n"));
        sse.push_str(&c.finish());

        let completed = sse
            .split("event: response.completed\ndata: ")
            .nth(1)
            .unwrap();
        let completed: Value = serde_json::from_str(completed.trim()).unwrap();
        let usage = &completed["response"]["usage"];
        assert_eq!(usage["input_tokens"], 10);
        assert_eq!(usage["output_tokens"], 5);
        assert_eq!(usage["total_tokens"], 15);
    }

    // ── ResponsesToChatStreamConverter ─────────────────────────────────────────

    /// Round-trip through both streaming converters: a chat SSE stream → Responses
    /// SSE (existing converter) → chat SSE (new converter) must preserve content,
    /// reasoning, tool calls, and usage. The existing converter is the oracle.
    #[test]
    fn responses_to_chat_stream_roundtrip() {
        let chat_sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        // chat → responses (oracle), then responses → chat (unit under test).
        let mut to_resp = ResponsesStreamConverter::new("gpt-5.4", false);
        let mut responses_sse = to_resp.push_bytes(chat_sse.as_bytes());
        responses_sse.push_str(&to_resp.finish());

        let mut to_chat = ResponsesToChatStreamConverter::new("gpt-5.4", true);
        let mut back = to_chat.push_bytes(responses_sse.as_bytes());
        back.push_str(&to_chat.finish());

        assert!(
            back.trim_end().ends_with("data: [DONE]"),
            "must terminate the stream"
        );
        let acc = accumulate_chat_sse(&back);
        let msg = &acc["choices"][0]["message"];
        assert_eq!(msg["reasoning_content"], "think");
        assert_eq!(acc["choices"][0]["finish_reason"], "tool_calls");
        let tc = &msg["tool_calls"][0];
        assert_eq!(tc["function"]["name"], "shell");
        assert_eq!(tc["function"]["arguments"], "{\"cmd\":\"ls\"}");
        assert_eq!(tc["id"], "call_1");

        // usage rides the dedicated trailing chunk (include_usage = true).
        let usage_line = back
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .find(|c| c.get("usage").is_some_and(|u| !u.is_null()))
            .expect("a usage chunk");
        assert_eq!(usage_line["usage"]["prompt_tokens"], 10);
        assert_eq!(usage_line["usage"]["completion_tokens"], 5);
        assert_eq!(usage_line["usage"]["total_tokens"], 15);
    }

    /// A plain text turn yields content + a `stop` finish and a clean terminator.
    #[test]
    fn responses_to_chat_stream_text_only() {
        let responses_sse = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
        );
        let mut c = ResponsesToChatStreamConverter::new("gpt-5.4", false);
        let mut out = c.push_bytes(responses_sse.as_bytes());
        out.push_str(&c.finish());
        let acc = accumulate_chat_sse(&out);
        assert_eq!(acc["choices"][0]["message"]["content"], "hi");
        assert_eq!(acc["choices"][0]["finish_reason"], "stop");
        // include_usage = false → no usage chunk emitted.
        assert!(!out.contains("\"usage\""));
    }
}
