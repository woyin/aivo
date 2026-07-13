use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_route_pipeline::inject_chat_completions_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::effort::gpt5_chat_completions_rejects_tools_with_none_reasoning;
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::model_names::{
    copilot_model_name, requires_max_completion_tokens, transform_model_for_openrouter,
};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    build_google_stream_generate_content_url, convert_gemini_to_openai_chat_response,
    convert_openai_chat_to_gemini_request,
};
use crate::services::responses_chat_conversion::{
    ResponsesStreamConverter, convert_chat_to_responses_request, convert_responses_json_to_chat,
};
use crate::services::serve_stream_converters::{
    AnthropicToOpenAIStreamConverter, GeminiToOpenAIStreamConverter,
};

#[derive(Clone)]
pub(crate) struct UpstreamRequestContext {
    pub(crate) client: reqwest::Client,
    pub(crate) upstream_base_url: String,
    pub(crate) upstream_api_key: String,
    pub(crate) is_copilot: bool,
    pub(crate) is_openrouter: bool,
    pub(crate) is_starter: bool,
    pub(crate) copilot_tokens: Option<Arc<CopilotTokenManager>>,
    /// Usage accounting is on — streamed OpenAI requests must ask for the
    /// trailing usage chunk or the sniffer records zero for the turn.
    pub(crate) accounting: bool,
}

impl UpstreamRequestContext {
    /// Conditionally attaches device fingerprint headers for starter endpoint requests.
    fn with_device_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if !self.is_starter {
            return builder;
        }
        device_fingerprint::with_starter_headers(builder)
    }
}

pub(crate) enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    Streaming {
        status: u16,
        content_type: String,
        body: Box<StreamingBody>,
    },
}

pub(crate) enum StreamingBody {
    Upstream(reqwest::Response),
    Anthropic {
        upstream: reqwest::Response,
        converter: AnthropicToOpenAIStreamConverter,
    },
    Gemini {
        upstream: reqwest::Response,
        converter: GeminiToOpenAIStreamConverter,
    },
    Responses {
        source: Box<StreamingBody>,
        converter: ResponsesStreamConverter,
    },
}

impl RouterResponse {
    pub(crate) fn buffered(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Self::Buffered {
            status,
            content_type: content_type.to_string(),
            body,
        }
    }
}

pub(crate) async fn send_anthropic_chat(
    body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let (fallback_model, anthropic_req) = build_anthropic_request(body, client_wants_stream);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/messages");
    let response = context
        .with_device_headers(
            context
                .client
                .post(&url)
                .header("x-api-key", context.upstream_api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", CONTENT_TYPE_JSON),
        )
        .json(&anthropic_req)
        .send_logged()
        .await?;

    finalize_anthropic_response(response, client_wants_stream, &fallback_model).await
}

pub(crate) async fn send_gemini_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gemini-2.5-pro")
        .to_string();

    let gemini_req = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );

    let url = if client_wants_stream {
        build_google_stream_generate_content_url(&context.upstream_base_url, &model)
    } else {
        build_google_generate_content_url(&context.upstream_base_url, &model)
    };
    let response = context
        .with_device_headers(
            context
                .client
                .post(&url)
                .header("x-goog-api-key", context.upstream_api_key.as_str())
                .header("Content-Type", CONTENT_TYPE_JSON),
        )
        .json(&gemini_req)
        .send_logged()
        .await?;

    finalize_gemini_response(response, client_wants_stream, &model).await
}

pub(crate) async fn send_openai_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    normalize_openai_request_model(body, context.is_openrouter, context.is_copilot);
    migrate_max_tokens_for_reasoning_models(body);
    strip_non_function_tools(body);
    inject_include_usage_for_accounting(body, context.accounting);
    // Surface OpenAI's GPT-5.4 Chat Completions restriction (no tools when
    // reasoning_effort is "none") with a clear local 400 instead of letting
    // the upstream reject and producing a generic error the user has to
    // decode. The Responses API lifts this restriction; only Chat
    // Completions is affected here.
    if gpt5_chat_completions_rejects_tools_with_none_reasoning(body) {
        let body = serde_json::to_vec(&serde_json::json!({
            "error": {
                "message": "GPT-5.4+ Chat Completions does not support tools with reasoning_effort: \"none\". Switch to a higher effort or use the Responses API.",
                "type": "invalid_request_error",
                "code": "tools_require_reasoning_effort"
            }
        }))?;
        return Ok(RouterResponse::buffered(400, "application/json", body));
    }

    // Inception Mercury doesn't reliably stream `tool_calls`; the model narrates
    // tool intent in `delta.content` instead. Force a non-streamed upstream call
    // when tools are present — `finalize_openai_response` will buffer the JSON
    // and re-emit it as SSE so the inbound client still sees a stream.
    disable_stream_for_inception_with_tools(body, &context.upstream_base_url);

    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/chat/completions");
    let initiator = if context.is_copilot {
        Some(http_utils::copilot_initiator_from_openai(body))
    } else {
        None
    };
    let mut req = http_utils::authorized_openai_post(
        &context.client,
        &url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        initiator,
    )
    .await?;
    if context.is_copilot && http_utils::body_requests_vision(body) {
        req = req.header("Copilot-Vision-Request", "true");
    }

    let response = context
        .with_device_headers(req)
        .json(&*body)
        .send_logged()
        .await?;
    finalize_openai_response(response, client_wants_stream).await
}

/// When usage accounting is on, streamed OpenAI upstreams only emit the
/// trailing usage chunk if asked; without this the sniffer records zero
/// tokens for the turn. Client-provided stream_options win.
fn inject_include_usage_for_accounting(body: &mut Value, accounting: bool) {
    if accounting
        && body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        && body.get("stream_options").is_none()
    {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
}

/// True when a Copilot `/chat/completions` error demands the Responses API.
pub(crate) fn copilot_requires_responses_api(body: &[u8]) -> bool {
    let s = String::from_utf8_lossy(body).to_lowercase();
    s.contains("unsupported_api_for_model")
        || (s.contains("/responses") && s.contains("chat/completions"))
}

/// Sends a Copilot chat request as a non-streamed `/responses` call, converting
/// the result back to Chat Completions (re-emitting SSE when a stream was asked).
/// Self-contained: normalizes the model itself, so callers may pass a raw body.
pub(crate) async fn send_copilot_responses(
    chat_body: &Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let mut chat_body = chat_body.clone();
    normalize_openai_request_model(&mut chat_body, false, true);
    strip_non_function_tools(&mut chat_body);
    let responses_body = convert_chat_to_responses_request(&chat_body);
    // Bare path, not a URL: Copilot's endpoint comes from the token exchange, and
    // a URL built from the "copilot" sentinel base wouldn't parse to `/responses`.
    let initiator = http_utils::copilot_initiator_from_openai(&chat_body);
    let mut req = http_utils::authorized_openai_post(
        &context.client,
        "/v1/responses",
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        Some(initiator),
    )
    .await?;
    let responses_value = serde_json::to_value(&responses_body)?;
    if http_utils::body_requests_vision(&responses_value) {
        req = req.header("Copilot-Vision-Request", "true");
    }

    let response = context
        .with_device_headers(req)
        .json(&responses_body)
        .send_logged()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    let text = response.text().await?;
    if status != 200 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            text.into_bytes(),
        ));
    }

    let responses_json: Value = serde_json::from_str(&text)?;
    let chat_json = convert_responses_json_to_chat(&responses_json);
    if client_wants_stream {
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&chat_json)?.into_bytes(),
        ));
    }
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        serde_json::to_vec(&chat_json)?,
    ))
}

pub(crate) async fn send_openai_embeddings(
    body: &Value,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    let url = http_utils::build_target_url(&context.upstream_base_url, "/v1/embeddings");
    let req = http_utils::authorized_openai_post(
        &context.client,
        &url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        None,
    )
    .await?;

    let response = context
        .with_device_headers(req)
        .json(body)
        .send_logged()
        .await?;
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);
    let body_bytes = response.bytes().await?.to_vec();
    Ok(RouterResponse::buffered(status, &content_type, body_bytes))
}

fn build_anthropic_request(body: &Value, client_wants_stream: bool) -> (String, Value) {
    let fallback_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("claude-sonnet-4-5")
        .to_string();

    let mut body_with_cache = body.clone();
    // Only inject cache_control for Claude models — other providers don't
    // honor it (e.g. Gemini uses a different caching model) and strict ones
    // reject the unknown field outright.
    if body_with_cache
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("claude"))
    {
        inject_chat_completions_cache_control(&mut body_with_cache);
    }

    let mut anthropic_req = convert_openai_chat_to_anthropic_request(
        &body_with_cache,
        &OpenAIToAnthropicChatConfig {
            default_model: "claude-sonnet-4-5",
        },
    );
    anthropic_req["stream"] = json!(client_wants_stream);

    (fallback_model, anthropic_req)
}

/// Rename the legacy `max_tokens` field to `max_completion_tokens` when the
/// target model is in OpenAI's reasoning family (o-series / GPT-5+ / Codex).
/// The Chat Completions API rejects `max_tokens` on those models with a 400.
/// If `max_completion_tokens` is already present, the legacy field is removed
/// to avoid the upstream rejecting both being set.
fn migrate_max_tokens_for_reasoning_models(body: &mut Value) {
    let model = match body.get("model").and_then(|v| v.as_str()) {
        Some(m) => m.to_string(),
        None => return,
    };
    if !requires_max_completion_tokens(&model) {
        return;
    }
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let legacy = obj.remove("max_tokens");
    if obj.contains_key("max_completion_tokens") {
        return;
    }
    if let Some(value) = legacy {
        obj.insert("max_completion_tokens".to_string(), value);
    }
}

/// OpenAI Chat Completions only accepts `tools[].type == "function"`. Server
/// tools like `{type:"web_search"}` (Anthropic/Responses-native) reach this
/// passthrough when a model is served over an OpenAI-compatible gateway — e.g.
/// a `claude-*` model on a third-party endpoint — and 400 ("expected function").
/// Drop them so the request succeeds; the Anthropic/Gemini bridges (separate
/// paths) still translate these server tools natively.
pub(crate) fn strip_non_function_tools(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
    }
    if body
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|a| a.is_empty())
        && let Some(obj) = body.as_object_mut()
    {
        obj.remove("tools");
        // A `tool_choice` with no tools 400s the OpenAI Chat upstream.
        obj.remove("tool_choice");
    }
}

pub(crate) fn disable_stream_for_inception_with_tools(body: &mut Value, upstream_base_url: &str) {
    let url_matches = upstream_base_url.contains("inceptionlabs.ai");
    let model_matches = body
        .get("model")
        .and_then(|v| v.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("mercury"));
    if !url_matches && !model_matches {
        return;
    }
    let has_tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    if !has_tools {
        return;
    }
    body["stream"] = json!(false);
    if let Some(obj) = body.as_object_mut() {
        obj.remove("stream_options");
    }
}

fn normalize_openai_request_model(body: &mut Value, is_openrouter: bool, is_copilot: bool) {
    if is_openrouter {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(transform_model_for_openrouter);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    } else if is_copilot {
        let normalized = body
            .get("model")
            .and_then(|v| v.as_str())
            .map(copilot_model_name);
        if let Some(n) = normalized {
            body["model"] = json!(n);
        }
    }
}

async fn finalize_anthropic_response(
    response: reqwest::Response,
    client_wants_stream: bool,
    fallback_model: &str,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Anthropic {
                upstream: response,
                converter: AnthropicToOpenAIStreamConverter::new(fallback_model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let anthropic_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_anthropic_to_openai_chat_response(&anthropic_resp, fallback_model);

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            CONTENT_TYPE_JSON,
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn finalize_gemini_response(
    response: reqwest::Response,
    client_wants_stream: bool,
    model: &str,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type: "text/event-stream".to_string(),
            body: Box::new(StreamingBody::Gemini {
                upstream: response,
                converter: GeminiToOpenAIStreamConverter::new(model),
            }),
        });
    }

    let resp_body = response.text().await?;
    let gemini_resp: Value = serde_json::from_str(&resp_body)?;
    let openai_resp = convert_gemini_to_openai_chat_response(&gemini_resp, model);

    if client_wants_stream {
        Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ))
    } else {
        Ok(RouterResponse::buffered(
            200,
            CONTENT_TYPE_JSON,
            openai_resp.to_string().into_bytes(),
        ))
    }
}

async fn finalize_openai_response(
    response: reqwest::Response,
    client_wants_stream: bool,
) -> Result<RouterResponse> {
    let status = response.status().as_u16();
    let content_type = http_utils::response_content_type(&response);

    if status >= 400 {
        return Ok(RouterResponse::buffered(
            status,
            &content_type,
            response.bytes().await?.to_vec(),
        ));
    }

    if client_wants_stream && content_type.contains("text/event-stream") {
        return Ok(RouterResponse::Streaming {
            status,
            content_type,
            body: Box::new(StreamingBody::Upstream(response)),
        });
    }

    let resp_body = response.text().await?;

    if client_wants_stream && let Ok(openai_resp) = serde_json::from_str::<Value>(&resp_body) {
        return Ok(RouterResponse::buffered(
            200,
            "text/event-stream",
            convert_openai_chat_response_to_sse(&openai_resp)?.into_bytes(),
        ));
    }

    Ok(RouterResponse::buffered(
        status,
        &content_type,
        resp_body.into_bytes(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::Response as HttpResponse;
    use serde_json::json;

    fn mock_response(
        status: u16,
        content_type: &str,
        body: impl Into<String>,
    ) -> reqwest::Response {
        HttpResponse::builder()
            .status(status)
            .header("content-type", content_type)
            .body(body.into())
            .unwrap()
            .into()
    }

    #[test]
    fn include_usage_injected_only_for_accounted_streaming() {
        let mut body = json!({"model": "m", "stream": true});
        inject_include_usage_for_accounting(&mut body, true);
        assert_eq!(body["stream_options"], json!({"include_usage": true}));

        // Client-provided stream_options win.
        let mut body =
            json!({"model": "m", "stream": true, "stream_options": {"include_usage": false}});
        inject_include_usage_for_accounting(&mut body, true);
        assert_eq!(body["stream_options"], json!({"include_usage": false}));

        // No accounting or no stream → untouched.
        let mut body = json!({"model": "m", "stream": true});
        inject_include_usage_for_accounting(&mut body, false);
        assert!(body.get("stream_options").is_none());
        let mut body = json!({"model": "m", "stream": false});
        inject_include_usage_for_accounting(&mut body, true);
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn copilot_requires_responses_api_detects_gpt5_and_codex_redirects() {
        // Exact gpt-5.4 tools + reasoning_effort rejection from Copilot.
        let gpt5 = br#"{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.4 in /v1/chat/completions. Please use /v1/responses instead.","code":"invalid_request_body"}}"#;
        assert!(copilot_requires_responses_api(gpt5));
        // Codex-family "not accessible" / unsupported_api_for_model.
        let codex = br#"{"error":{"message":"model 'gpt-5.3-codex' is not accessible via the /chat/completions endpoint","code":"unsupported_api_for_model"}}"#;
        assert!(copilot_requires_responses_api(codex));
        // Unrelated 400s must not trigger the fallback.
        assert!(!copilot_requires_responses_api(
            br#"{"error":{"message":"invalid request: missing model"}}"#
        ));
        assert!(!copilot_requires_responses_api(b"model not found"));
    }

    fn sample_openai_chat_response() -> String {
        json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from upstream"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        })
        .to_string()
    }

    fn sample_anthropic_response() -> String {
        json!({
            "id": "msg_1",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "Hello from anthropic"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 4}
        })
        .to_string()
    }

    fn sample_gemini_response() -> String {
        json!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello from gemini"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}
        })
        .to_string()
    }

    #[test]
    fn normalize_openai_request_model_rewrites_openrouter_model_names() {
        let mut body = json!({"model": "claude-sonnet-4-6"});
        normalize_openai_request_model(&mut body, true, false);
        assert_eq!(body["model"], "anthropic/claude-sonnet-4.6");
    }

    #[test]
    fn normalize_openai_request_model_rewrites_copilot_model_names() {
        let mut body = json!({"model": "claude-sonnet-4-6-20250603"});
        normalize_openai_request_model(&mut body, false, true);
        assert_eq!(body["model"], "claude-sonnet-4.6");
    }

    #[test]
    fn build_anthropic_request_sets_stream_flag_and_model() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"}
            ]
        });

        let (fallback_model, request) = build_anthropic_request(&body, true);

        assert_eq!(fallback_model, "claude-sonnet-4-5");
        assert_eq!(request["model"], "claude-sonnet-4-5");
        assert_eq!(request["stream"], true);
        assert_eq!(request["system"][0]["text"], "Be precise.");
        assert_eq!(request["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(request["messages"][0]["content"][0]["text"], "Hello");
        assert_eq!(
            request["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn build_anthropic_request_skips_cache_control_for_non_claude_models() {
        let body = json!({
            "model": "MiniMax-M1",
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Hello"}
            ]
        });

        let (_, request) = build_anthropic_request(&body, false);

        assert_eq!(request["system"], "Be precise.");
        assert_eq!(request["messages"][0]["content"], "Hello");
    }

    #[tokio::test]
    async fn finalize_openai_response_converts_json_to_sse_when_streaming_requested() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_openai_chat_response());

        let result = finalize_openai_response(response, true).await.unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("\"content\":\"Hello from upstream\""));
                assert!(sse.contains("data: [DONE]"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[tokio::test]
    async fn finalize_openai_response_buffers_errors() {
        let response = mock_response(404, CONTENT_TYPE_JSON, r#"{"error":"missing"}"#);

        let result = finalize_openai_response(response, false).await.unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                assert_eq!(status, 404);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(String::from_utf8(body).unwrap(), r#"{"error":"missing"}"#);
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered error"),
        }
    }

    #[tokio::test]
    async fn finalize_anthropic_response_converts_json_to_openai_chat() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_anthropic_response());

        let result = finalize_anthropic_response(response, false, "claude-sonnet-4-5")
            .await
            .unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let json: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(
                    json["choices"][0]["message"]["content"],
                    "Hello from anthropic"
                );
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered response"),
        }
    }

    #[tokio::test]
    async fn finalize_gemini_response_converts_json_to_sse_when_streaming_requested() {
        let response = mock_response(200, CONTENT_TYPE_JSON, sample_gemini_response());

        let result = finalize_gemini_response(response, true, "gemini-2.5-pro")
            .await
            .unwrap();

        match result {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("\"content\":\"Hello from gemini\""));
                assert!(sse.contains("data: [DONE]"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[tokio::test]
    async fn finalize_openai_response_preserves_upstream_event_streams() {
        let response = mock_response(200, "text/event-stream", "data: [DONE]\n\n");

        let result = finalize_openai_response(response, true).await.unwrap();

        match result {
            RouterResponse::Streaming {
                status,
                content_type,
                ..
            } => {
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
            }
            RouterResponse::Buffered { .. } => panic!("expected streaming response"),
        }
    }

    #[test]
    fn build_anthropic_request_missing_model_uses_default() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let (fallback_model, request) = build_anthropic_request(&body, true);

        assert_eq!(fallback_model, "claude-sonnet-4-5");
        assert_eq!(request["model"], "claude-sonnet-4-5");
    }

    #[test]
    fn build_anthropic_request_non_stream() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let (_fallback_model, request) = build_anthropic_request(&body, false);

        assert_eq!(request["stream"], false);
    }

    #[test]
    fn normalize_openai_request_model_no_op_when_neither_flag() {
        let mut body = json!({"model": "claude-sonnet-4-6"});
        normalize_openai_request_model(&mut body, false, false);
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn normalize_openai_request_model_no_op_missing_model_field() {
        let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        normalize_openai_request_model(&mut body, true, false);
        // No crash, and no model field is inserted
        assert!(body.get("model").is_none());
    }

    #[test]
    fn migrate_max_tokens_renames_for_reasoning_model() {
        let mut body = json!({"model": "gpt-5", "max_tokens": 4096});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 4096);
    }

    #[test]
    fn migrate_max_tokens_renames_for_o_series() {
        let mut body = json!({"model": "o3-mini", "max_tokens": 2048});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 2048);
    }

    #[test]
    fn migrate_max_tokens_preserves_non_reasoning_field() {
        let mut body = json!({"model": "gpt-4o", "max_tokens": 4096});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert_eq!(body["max_tokens"], 4096);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn migrate_max_tokens_does_not_overwrite_existing_new_field() {
        let mut body = json!({
            "model": "gpt-5",
            "max_tokens": 4096,
            "max_completion_tokens": 8192,
        });
        migrate_max_tokens_for_reasoning_models(&mut body);
        // Drop the legacy field, keep the explicit new field intact.
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], 8192);
    }

    #[test]
    fn migrate_max_tokens_handles_prefixed_reasoning_model_name() {
        let mut body = json!({"model": "openai/gpt-5-codex", "max_tokens": 1024});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert_eq!(body["max_completion_tokens"], 1024);
    }

    #[test]
    fn migrate_max_tokens_no_op_when_field_absent() {
        let mut body = json!({"model": "gpt-5"});
        migrate_max_tokens_for_reasoning_models(&mut body);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn strip_non_function_tools_drops_server_tools() {
        // `{type:"web_search"}` alongside function tools 400s an OpenAI Chat
        // endpoint ("expected function"); it must be dropped, function tools kept.
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "web_search"}
            ]
        });
        strip_non_function_tools(&mut body);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], "read_file");
    }

    #[test]
    fn strip_non_function_tools_removes_empty_tools_key() {
        let mut body = json!({"model": "x", "tools": [{"type": "web_search"}]});
        strip_non_function_tools(&mut body);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn strip_non_function_tools_drops_orphaned_tool_choice() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [{"type": "web_search"}],
            "tool_choice": "required"
        });
        strip_non_function_tools(&mut body);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn strip_non_function_tools_keeps_tool_choice_when_function_tools_survive() {
        let mut body = json!({
            "model": "claude-sonnet-4-6",
            "tools": [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "web_search"}
            ],
            "tool_choice": "auto"
        });
        strip_non_function_tools(&mut body);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn disable_stream_for_inception_flips_stream_when_tools_present() {
        let mut body = json!({
            "model": "mercury-2",
            "stream": true,
            "stream_options": {"include_usage": true},
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], false);
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn disable_stream_for_inception_no_op_for_other_providers() {
        let mut body = json!({
            "model": "gpt-4o",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://api.openai.com/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_no_op_when_no_tools_field() {
        let mut body = json!({"model": "mercury-2", "stream": true});
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_no_op_when_tools_empty() {
        let mut body = json!({"model": "mercury-2", "stream": true, "tools": []});
        disable_stream_for_inception_with_tools(&mut body, "https://api.inceptionlabs.ai/v1/");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn disable_stream_for_inception_matches_by_model_name() {
        let mut body = json!({
            "model": "inception/mercury",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://openrouter.ai/api/v1/");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn disable_stream_for_inception_matches_mercury_edit() {
        let mut body = json!({
            "model": "Mercury-Coder-Small",
            "stream": true,
            "tools": [{"type": "function", "function": {"name": "Bash"}}]
        });
        disable_stream_for_inception_with_tools(&mut body, "https://example.com/v1/");
        assert_eq!(body["stream"], false);
    }
}
