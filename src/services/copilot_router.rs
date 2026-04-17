//! CopilotRouter: HTTP proxy for routing Claude Code requests through GitHub Copilot.
//!
//! Receives Anthropic Messages API requests from Claude Code, converts them
//! directly to OpenAI Responses API format, forwards to the Copilot API, and
//! converts the response back to Anthropic format.

use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, convert_anthropic_to_openai_request,
};
use crate::services::anthropic_chat_response::{
    OpenAIToAnthropicConfig, UsageValueMode, convert_openai_to_anthropic_message,
};
use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INITIATOR_HEADER, COPILOT_INTEGRATION_ID,
    COPILOT_OPENAI_INTENT, CopilotTokenManager,
};
use crate::services::http_utils;

#[derive(Clone)]
pub struct CopilotRouterConfig {
    pub github_token: String,
}

pub struct CopilotRouter {
    config: CopilotRouterConfig,
}

struct CopilotRouterState {
    token_manager: Arc<CopilotTokenManager>,
    client: reqwest::Client,
}

impl CopilotRouter {
    pub fn new(config: CopilotRouterConfig) -> Self {
        Self { config }
    }

    pub async fn start_background(&self) -> Result<(u16, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let state = CopilotRouterState {
            token_manager: Arc::new(CopilotTokenManager::new(self.config.github_token.clone())),
            client: http_utils::router_http_client(),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_text_router(listener, Arc::new(state), handle_copilot_request).await
        });
        Ok((port, handle))
    }
}

async fn handle_copilot_request(request: String, state: Arc<CopilotRouterState>) -> String {
    if http_utils::is_post_path(&request, &["/v1/messages", "/messages"]) {
        match handle_messages(&request, &state).await {
            Ok(r) => r,
            Err(e) => http_utils::http_error_response(500, &e.to_string()),
        }
    } else {
        http_utils::http_error_response(404, "Not found")
    }
}

async fn handle_messages(request: &str, state: &CopilotRouterState) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    let is_streaming = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-sonnet-4-20250514")
        .to_string();

    let (copilot_token, api_endpoint) = state.token_manager.get_token().await?;
    let initiator = http_utils::copilot_initiator_from_anthropic(&body);
    let base = api_endpoint.trim_end_matches('/');

    // Try Responses API first (supports newer models like gpt-5.4).
    // No cross-request caching — API support is model-specific and models
    // can change mid-session (e.g. Claude Code fast mode).
    let responses_req = anthropic_to_responses(&body);
    let (s, b) = copilot_post(
        &state.client,
        &format!("{base}/v1/responses"),
        &copilot_token,
        initiator,
        &responses_req,
    )
    .await?;

    if s == 200 {
        let resp_value: Value = serde_json::from_str(&b)?;
        let anthropic_resp = responses_to_anthropic(&resp_value, &model);
        return format_anthropic_response(&anthropic_resp, is_streaming);
    }

    // Fall back to Chat Completions if Responses API doesn't support this model
    if is_responses_fallback(s, &b) {
        let openai_req = anthropic_to_openai_chat(&body);
        let (status, resp_body) = copilot_post(
            &state.client,
            &format!("{base}/chat/completions"),
            &copilot_token,
            initiator,
            &openai_req,
        )
        .await?;

        if status != 200 {
            let message = explain_copilot_error(&resp_body);
            return Ok(http_utils::http_error_response(status, &message));
        }

        let openai_resp: Value = serde_json::from_str(&resp_body)?;
        let anthropic_resp = openai_to_anthropic(&openai_resp, &model)?;
        return format_anthropic_response(&anthropic_resp, is_streaming);
    }

    // Non-fallback error from Responses API — return it directly
    let message = explain_copilot_error(&b);
    Ok(http_utils::http_error_response(s, &message))
}

async fn copilot_post(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    initiator: &str,
    body: &Value,
) -> Result<(u16, String)> {
    let resp = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", CONTENT_TYPE_JSON)
        .header("Editor-Version", COPILOT_EDITOR_VERSION)
        .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
        .header("Openai-Intent", COPILOT_OPENAI_INTENT)
        .header(COPILOT_INITIATOR_HEADER, initiator)
        .json(body)
        .send()
        .await?;
    let status = resp.status().as_u16();
    let body = resp.text().await?;
    Ok((status, body))
}

/// Should we fall back from Responses API to Chat Completions?
fn is_responses_fallback(status: u16, body: &str) -> bool {
    // Endpoint not available
    if matches!(status, 404 | 405) {
        return true;
    }
    // Model only available on chat/completions
    if status == 400
        && let Some(code) = extract_copilot_error_code(body)
    {
        return code == "unsupported_api_for_model";
    }
    false
}

fn extract_copilot_error_code(body: &str) -> Option<&str> {
    // Fast path: check for the code string without full JSON parsing
    if body.contains("unsupported_api_for_model") {
        return Some("unsupported_api_for_model");
    }
    None
}

fn format_anthropic_response(anthropic_resp: &Value, is_streaming: bool) -> Result<String> {
    if is_streaming {
        let sse = anthropic_to_sse(anthropic_resp);
        Ok(http_utils::http_response(200, "text/event-stream", &sse))
    } else {
        let json = serde_json::to_string(anthropic_resp)?;
        Ok(http_utils::http_json_response(200, &json))
    }
}

fn explain_copilot_error(resp_body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(resp_body).ok();
    let outer_message = parsed
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(|v| v.get("message"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let nested = outer_message.and_then(|message| serde_json::from_str::<Value>(message).ok());
    let nested_error = nested.as_ref().and_then(|v| v.get("error"));
    let nested_code = nested_error
        .and_then(|v| v.get("code"))
        .and_then(|v| v.as_str());
    let nested_message = nested_error
        .and_then(|v| v.get("message"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    if nested_code == Some("model_max_prompt_tokens_exceeded") {
        let detail = nested_message.unwrap_or("prompt token count exceeds the model limit");
        return format!(
            "GitHub Copilot rejected the Claude Code request because the prompt is too large for the selected model ({detail}). Claude Code includes a large built-in system and tool prompt, so this can fail even on a short message like \"hi\". Use a provider/model with a larger context window, or use `aivo chat`/`aivo codex` instead of Claude Code for Copilot-backed sessions."
        );
    }

    if nested_code == Some("unsupported_api_for_model") {
        let detail = nested_message.unwrap_or("the selected model is not available on Copilot");
        return format!(
            "GitHub Copilot rejected the selected model ({detail}). Switch to a supported model with `/model`, or relaunch `aivo claude --model claude-sonnet-4`."
        );
    }

    nested_message
        .or(outer_message)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| resp_body.trim().to_string())
}

fn anthropic_to_openai_chat(body: &Value) -> Value {
    convert_anthropic_to_openai_request(
        body,
        &AnthropicToOpenAIConfig {
            default_model: "claude-sonnet-4-20250514",
            preserve_stream: false,
            model_transform: Some(copilot_model_name),
            include_reasoning_content: false,
            require_non_empty_reasoning_content: false,
            stringify_other_tool_result_content: false,
            tool_result_supports_multimodal: true,
            fallback_tool_arguments_json: "",
        },
    )
}

fn copilot_model_name(model: &str) -> String {
    crate::services::model_names::copilot_model_name(model)
}

fn openai_to_anthropic(resp: &Value, model: &str) -> Result<Value> {
    Ok(convert_openai_to_anthropic_message(
        resp,
        &OpenAIToAnthropicConfig {
            fallback_id: "msg_copilot",
            model,
            include_created: false,
            usage_value_mode: UsageValueMode::PreserveJson,
        },
    )?)
}

fn anthropic_to_responses(body: &Value) -> Value {
    let mut input = Vec::new();
    let mut instructions = String::new();
    let mut fc_counter = 0u64;

    // System prompt → instructions
    if let Some(system) = body.get("system") {
        if let Some(s) = system.as_str() {
            instructions = s.to_string();
        } else if let Some(arr) = system.as_array() {
            for item in arr {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    if !instructions.is_empty() {
                        instructions.push('\n');
                    }
                    instructions.push_str(text);
                }
            }
        }
    }

    // Messages → input items
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");

            if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
                // Simple string content
                if role == "assistant" {
                    input.push(json!({
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": s}]
                    }));
                } else {
                    input.push(json!({"type": "message", "role": role, "content": s}));
                }
                continue;
            }

            // Array content — may contain text, tool_use, or tool_result blocks
            let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
                continue;
            };

            let mut user_text = String::new();
            let mut asst_parts: Vec<Value> = Vec::new();
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                    "text" => {
                        let text = block["text"].as_str().unwrap_or("");
                        if role == "assistant" {
                            asst_parts.push(json!({"type": "output_text", "text": text}));
                        } else if !text.is_empty() {
                            if !user_text.is_empty() {
                                user_text.push('\n');
                            }
                            user_text.push_str(text);
                        }
                    }
                    "tool_use" => {
                        if !asst_parts.is_empty() {
                            input.push(json!({
                                "type": "message", "role": "assistant",
                                "content": asst_parts
                            }));
                            asst_parts = Vec::new();
                        }
                        fc_counter += 1;
                        let arguments = serde_json::to_string(&block["input"]).unwrap_or_default();
                        input.push(json!({
                            "type": "function_call",
                            "id": format!("fc_{fc_counter}"),
                            "call_id": block["id"],
                            "name": block["name"],
                            "arguments": arguments
                        }));
                    }
                    "tool_result" => {
                        input.push(json!({
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": extract_tool_result_text(block)
                        }));
                    }
                    _ => {}
                }
            }
            if !user_text.is_empty() {
                input.push(json!({"type": "message", "role": role, "content": user_text}));
            }
            if !asst_parts.is_empty() {
                input.push(json!({
                    "type": "message", "role": "assistant",
                    "content": asst_parts
                }));
            }
        }
    }

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("claude-sonnet-4-20250514");

    let mut req = json!({
        "model": crate::services::model_names::copilot_model_name(model),
        "input": input,
        "stream": false
    });

    if !instructions.is_empty() {
        req["instructions"] = json!(instructions);
    }
    if let Some(v) = body.get("max_tokens") {
        req["max_output_tokens"] = v.clone();
    }
    if let Some(v) = body.get("temperature") {
        req["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        req["top_p"] = v.clone();
    }

    // Tools: input_schema → parameters
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let converted: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description").unwrap_or(&json!("")),
                    "parameters": t.get("input_schema").unwrap_or(&json!({}))
                })
            })
            .collect();
        req["tools"] = json!(converted);
    }

    // tool_choice: auto→auto, any→required, tool→{type:"function",function:{name}}
    if let Some(tc) = body.get("tool_choice") {
        match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => req["tool_choice"] = json!("auto"),
            Some("any") => req["tool_choice"] = json!("required"),
            Some("tool") => {
                if let Some(name) = tc.get("name") {
                    req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
    }

    req
}

fn extract_tool_result_text(block: &Value) -> String {
    if let Some(s) = block.get("content").and_then(|c| c.as_str()) {
        return s.to_string();
    }
    if let Some(arr) = block.get("content").and_then(|c| c.as_array()) {
        return arr
            .iter()
            .filter_map(|item| item.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
    }
    String::new()
}

fn responses_to_anthropic(resp: &Value, model: &str) -> Value {
    let mut content = Vec::new();
    let mut has_tool_use = false;

    if let Some(output) = resp.get("output").and_then(|o| o.as_array()) {
        for item in output {
            match item.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "message" => {
                    if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                        for part in parts {
                            if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                                let text = part["text"].as_str().unwrap_or("");
                                if !text.is_empty() {
                                    content.push(json!({"type": "text", "text": text}));
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    has_tool_use = true;
                    let call_id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool_call");
                    let args_str = item["arguments"].as_str().unwrap_or("{}");
                    let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
                    content.push(json!({
                        "type": "tool_use",
                        "id": call_id,
                        "name": item["name"],
                        "input": input
                    }));
                }
                _ => {}
            }
        }
    }

    let mut usage = json!({"input_tokens": 0, "output_tokens": 0});
    if let Some(u) = resp.get("usage") {
        if let Some(v) = u.get("input_tokens") {
            usage["input_tokens"] = v.clone();
        }
        if let Some(v) = u.get("output_tokens") {
            usage["output_tokens"] = v.clone();
        }
        if let Some(v) = u.get("cache_read_input_tokens") {
            usage["cache_read_input_tokens"] = v.clone();
        }
        if let Some(v) = u.get("cache_creation_input_tokens") {
            usage["cache_creation_input_tokens"] = v.clone();
        }
    }

    json!({
        "id": resp.get("id").and_then(|i| i.as_str()).unwrap_or("msg_copilot"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": if has_tool_use { "tool_use" } else { "end_turn" },
        "stop_sequence": null,
        "usage": usage
    })
}

fn anthropic_to_sse(anthropic: &Value) -> String {
    let mut events = String::new();

    let input_tokens = anthropic["usage"]["input_tokens"].as_i64().unwrap_or(0);
    let output_tokens = anthropic["usage"]["output_tokens"].as_i64().unwrap_or(0);

    events.push_str(&format!(
        "event: message_start\ndata: {}\n\n",
        json!({
            "type": "message_start",
            "message": {
                "id": anthropic["id"],
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic["model"],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        })
    ));

    if let Some(content) = anthropic.get("content").and_then(|c| c.as_array()) {
        for (idx, block) in content.iter().enumerate() {
            match block.get("type").and_then(|t| t.as_str()).unwrap_or("text") {
                "text" => {
                    let text = block["text"].as_str().unwrap_or("");
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
                    ));
                    if !text.is_empty() {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": text}})
                        ));
                    }
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                "tool_use" => {
                    events.push_str(&format!(
                        "event: content_block_start\ndata: {}\n\n",
                        json!({
                            "type": "content_block_start", "index": idx,
                            "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
                        })
                    ));
                    let input_str = serde_json::to_string(&block["input"]).unwrap_or_default();
                    if input_str != "{}" {
                        events.push_str(&format!(
                            "event: content_block_delta\ndata: {}\n\n",
                            json!({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": input_str}})
                        ));
                    }
                    events.push_str(&format!(
                        "event: content_block_stop\ndata: {}\n\n",
                        json!({"type": "content_block_stop", "index": idx})
                    ));
                }
                _ => {}
            }
        }
    }

    events.push_str(&format!(
        "event: message_delta\ndata: {}\n\n",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic["stop_reason"], "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        })
    ));
    events.push_str("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n");

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_names::copilot_model_name;

    #[test]
    fn test_copilot_model_name_strips_date_and_converts_dots() {
        assert_eq!(
            copilot_model_name("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
        assert_eq!(
            copilot_model_name("claude-sonnet-4-6-20250603"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-opus-4-6-20250210"),
            "claude-opus-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-haiku-4-5-20250501"),
            "claude-haiku-4.5"
        );
    }

    #[test]
    fn test_copilot_model_name_converts_dots() {
        assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
        assert_eq!(copilot_model_name("claude-haiku-4-5"), "claude-haiku-4.5");
        assert_eq!(copilot_model_name("claude-opus-4-5"), "claude-opus-4.5");
        assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
    }

    // --- anthropic_to_responses tests ---

    #[test]
    fn test_anthropic_to_responses_basic() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"}
            ]
        });
        let result = anthropic_to_responses(&body);
        assert_eq!(result["model"], "claude-sonnet-4");
        assert_eq!(result["instructions"], "You are helpful.");
        assert_eq!(result["max_output_tokens"], 1024);
        assert_eq!(result["stream"], false);
        let input = result["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "Hello");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"], "How are you?");
    }

    #[test]
    fn test_anthropic_to_responses_system_array() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "System prompt."}],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = anthropic_to_responses(&body);
        assert_eq!(result["instructions"], "System prompt.");
    }

    #[test]
    fn test_anthropic_to_responses_tool_use() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "SF"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 72°F"}
                ]}
            ]
        });
        let result = anthropic_to_responses(&body);
        let input = result["input"].as_array().unwrap();
        // user message + assistant text + function_call + function_call_output
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "What's the weather?");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "toolu_1");
        assert_eq!(input[2]["name"], "get_weather");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "toolu_1");
        assert_eq!(input[3]["output"], "Sunny, 72°F");
    }

    #[test]
    fn test_anthropic_to_responses_tools() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
            }]
        });
        let result = anthropic_to_responses(&body);
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(tools[0]["parameters"]["type"], "object");
    }

    #[test]
    fn test_anthropic_to_responses_empty_messages() {
        let body = json!({"model": "claude-sonnet-4", "max_tokens": 1024, "messages": []});
        let result = anthropic_to_responses(&body);
        assert!(result["input"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_anthropic_to_responses_multi_text_user_joined() {
        let body = json!({
            "model": "claude-sonnet-4",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "First paragraph"},
                {"type": "text", "text": "Second paragraph"}
            ]}]
        });
        let result = anthropic_to_responses(&body);
        let input = result["input"].as_array().unwrap();
        assert_eq!(
            input.len(),
            1,
            "multi-text blocks should be joined into one message"
        );
        assert_eq!(input[0]["content"], "First paragraph\nSecond paragraph");
    }

    #[test]
    fn test_anthropic_to_responses_missing_messages() {
        let body = json!({"model": "claude-sonnet-4", "max_tokens": 1024});
        let result = anthropic_to_responses(&body);
        assert!(result["input"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_tool_choice_auto() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "auto"}
        });
        let req = anthropic_to_responses(&body);
        assert_eq!(req["tool_choice"], json!("auto"));
    }

    #[test]
    fn test_tool_choice_any() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test", "description": "test", "input_schema": {}}],
            "tool_choice": {"type": "any"}
        });
        let req = anthropic_to_responses(&body);
        assert_eq!(req["tool_choice"], json!("required"));
    }

    #[test]
    fn test_tool_choice_specific_tool() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read_file", "description": "read", "input_schema": {}}],
            "tool_choice": {"type": "tool", "name": "read_file"}
        });
        let req = anthropic_to_responses(&body);
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_tool_choice_not_present() {
        let body =
            json!({"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]});
        let req = anthropic_to_responses(&body);
        assert!(req.get("tool_choice").is_none());
    }

    // --- responses_to_anthropic tests ---

    #[test]
    fn test_responses_to_anthropic_text() {
        let resp = json!({
            "id": "resp_xxx",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result = responses_to_anthropic(&resp, "gpt-5.4");
        assert_eq!(result["id"], "resp_xxx");
        assert_eq!(result["model"], "gpt-5.4");
        assert_eq!(result["content"][0]["type"], "text");
        assert_eq!(result["content"][0]["text"], "Hello!");
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 5);
    }

    #[test]
    fn test_responses_to_anthropic_tool_call() {
        let resp = json!({
            "id": "resp_yyy",
            "output": [{
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": "{\"path\":\"test.rs\"}"
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result = responses_to_anthropic(&resp, "gpt-5.4");
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["content"][0]["type"], "tool_use");
        assert_eq!(result["content"][0]["id"], "call_abc");
        assert_eq!(result["content"][0]["name"], "read_file");
        assert_eq!(result["content"][0]["input"]["path"], "test.rs");
    }

    #[test]
    fn test_responses_to_anthropic_empty_output() {
        let resp = json!({"id": "resp_zzz", "output": []});
        let result = responses_to_anthropic(&resp, "gpt-5.4");
        assert_eq!(result["model"], "gpt-5.4");
        assert_eq!(result["stop_reason"], "end_turn");
        assert!(result["content"].as_array().unwrap().is_empty());
    }

    // --- SSE tests ---

    #[test]
    fn test_anthropic_to_sse_text() {
        let anthropic = json!({
            "id": "resp_xxx", "model": "gpt-5.4",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let sse = anthropic_to_sse(&anthropic);
        assert!(sse.contains("event: message_start"));
        assert!(sse.contains("event: content_block_start"));
        assert!(sse.contains("\"text\":\"Hi!\""));
        assert!(sse.contains("event: content_block_stop"));
        assert!(sse.contains("event: message_delta"));
        assert!(sse.contains("event: message_stop"));
    }

    #[test]
    fn test_anthropic_to_sse_tool_use() {
        let anthropic = json!({
            "id": "resp_xxx", "model": "gpt-5.4",
            "content": [{"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "test.rs"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let sse = anthropic_to_sse(&anthropic);
        assert!(sse.contains("\"type\":\"tool_use\""));
        assert!(sse.contains("\"name\":\"read_file\""));
        assert!(sse.contains("input_json_delta"));
        assert!(sse.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_anthropic_to_sse_empty_content() {
        let anthropic = json!({
            "id": "resp_xxx", "model": "gpt-5.4", "content": [],
            "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0}
        });
        let sse = anthropic_to_sse(&anthropic);
        assert!(sse.contains("event: message_start"));
        assert!(sse.contains("event: message_stop"));
    }

    // --- Misc tests ---

    #[test]
    fn test_extract_body() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"key\":\"val\"}"
        );
    }

    #[test]
    fn test_extract_body_missing_separator() {
        assert!(http_utils::extract_request_body("POST /v1/messages HTTP/1.1").is_err());
    }

    #[test]
    fn test_error_response() {
        let resp = http_utils::http_error_response(500, "test error");
        assert!(resp.contains("500"));
        assert!(resp.contains("test error"));
    }

    #[test]
    fn test_explain_copilot_error_for_prompt_limit() {
        let body = json!({
            "error": {
                "message": "{\"error\":{\"message\":\"prompt token count of 13524 exceeds the limit of 12288\",\"code\":\"model_max_prompt_tokens_exceeded\"}}\n"
            }
        })
        .to_string();
        let message = explain_copilot_error(&body);
        assert!(message.contains("GitHub Copilot rejected the Claude Code request"));
        assert!(message.contains("13524"));
    }

    #[test]
    fn test_explain_copilot_error_unwraps_nested_message() {
        let body = json!({
            "error": {
                "message": "{\"error\":{\"message\":\"plain nested error\",\"code\":\"other_code\"}}\n"
            }
        })
        .to_string();
        assert_eq!(explain_copilot_error(&body), "plain nested error");
    }

    #[test]
    fn test_explain_copilot_error_for_unsupported_api_model() {
        let body = json!({
            "error": {
                "message": "{\"error\":{\"message\":\"model \\\"gpt-5.1-codex-mini\\\" is not accessible via the /chat/completions endpoint\",\"code\":\"unsupported_api_for_model\"}}\n"
            }
        })
        .to_string();
        let message = explain_copilot_error(&body);
        assert!(message.contains("rejected the selected model"));
    }

    #[test]
    fn test_explain_copilot_error_plain_text_body() {
        assert_eq!(
            explain_copilot_error("Something went wrong"),
            "Something went wrong"
        );
    }

    #[test]
    fn test_explain_copilot_error_empty_body() {
        assert_eq!(explain_copilot_error(""), "");
    }

    #[test]
    fn test_explain_copilot_error_malformed_json() {
        assert_eq!(
            explain_copilot_error("{not valid json}"),
            "{not valid json}"
        );
    }

    #[test]
    fn test_explain_copilot_error_empty_message() {
        let body = json!({"error": {"message": ""}}).to_string();
        let result = explain_copilot_error(&body);
        assert!(!result.is_empty());
    }

    #[test]
    fn explain_copilot_error_nested_json_no_error_key() {
        let nested = json!({"status": "bad", "detail": "something broke"}).to_string();
        let body = json!({"error": {"message": nested}}).to_string();
        let result = explain_copilot_error(&body);
        assert_eq!(result, nested);
    }
}
