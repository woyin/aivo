/**
 * Responses API ↔ Chat Completions conversion logic
 *
 * Converts between OpenAI Responses API format and Chat Completions format.
 * Used by the ResponsesToChatRouter and ServeRouter to bridge clients that
 * speak the Responses API with providers that only support Chat Completions.
 */
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

    // Convert "input" array items
    if let Some(input) = body.get("input").and_then(|v| v.as_array()) {
        for item in input {
            match item.get("type").and_then(|v| v.as_str()) {
                Some("message") => {
                    // Validate role - only allow valid OpenAI chat completion roles
                    let role = item
                        .get("role")
                        .and_then(|v| v.as_str())
                        .filter(|r| matches!(*r, "system" | "user" | "assistant" | "tool"))
                        .unwrap_or("user");
                    let content = extract_content_text(item.get("content"));
                    let mut msg = json!({"role": role, "content": content});
                    if role == "assistant" {
                        attach_reasoning_content(&mut msg, item, config.requires_reasoning_content);
                    }
                    messages.push(msg);
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
                    let mut msg = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]
                    });
                    attach_reasoning_content(&mut msg, item, config.requires_reasoning_content);
                    messages.push(msg);
                }
                Some("function_call_output") => {
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                    messages
                        .push(json!({"role": "tool", "tool_call_id": call_id, "content": output}));
                }
                None => {
                    // Simple string input
                    if let Some(s) = item.as_str() {
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
    for field in ["temperature", "top_p"] {
        if let Some(v) = body.get(field) {
            chat[field] = v.clone();
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

    if !tool_calls.is_empty() {
        // Tool call response — each tool call becomes a function_call output item
        for (i, tc) in tool_calls.iter().enumerate() {
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
        let has_reasoning = !reasoning_content.is_empty();

        sse.push_str(&sse_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": resp_id, "output_index": 0,
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
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        if !content.is_empty() {
            sse.push_str(&sse_event(
                "response.output_text.delta",
                &json!({
                    "type": "response.output_text.delta",
                    "response_id": resp_id, "item_id": msg_id,
                    "output_index": 0, "content_index": 0, "delta": content
                }),
            ));
        }
        sse.push_str(&sse_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": 0, "content_index": 0, "text": content
            }),
        ));
        sse.push_str(&sse_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "response_id": resp_id, "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": content}
            }),
        ));

        // Reasoning part (if present)
        if has_reasoning {
            sse.push_str(&sse_event(
                "response.content_part.added",
                &json!({
                    "type": "response.content_part.added",
                    "response_id": resp_id, "item_id": msg_id,
                    "output_index": 0, "content_index": 1,
                    "part": {"type": "reasoning", "reasoning": ""}
                }),
            ));
            sse.push_str(&sse_event(
                "response.content_part.done",
                &json!({
                    "type": "response.content_part.done",
                    "response_id": resp_id, "item_id": msg_id,
                    "output_index": 0, "content_index": 1,
                    "part": {"type": "reasoning", "reasoning": reasoning_content}
                }),
            ));
        }

        let mut content_parts =
            vec![json!({"type": "output_text", "text": content, "annotations": []})];
        if has_reasoning {
            content_parts.push(json!({"type": "reasoning", "reasoning": reasoning_content}));
        }
        let done_item = json!({
            "id": msg_id, "type": "message", "status": "completed",
            "role": "assistant",
            "content": content_parts
        });
        sse.push_str(&sse_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": resp_id, "output_index": 0, "item": done_item
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
        json!(input + output)
    });

    let mut response_usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: Some(8192),
                responses_api_supported: None,
                is_starter: false,
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
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: None,
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
                copilot_token_manager: None,
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
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
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: false,
            actual_model: Some("kimi-k2.5".to_string()),
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
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
}
