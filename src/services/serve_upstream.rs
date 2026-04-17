use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_route_pipeline::inject_chat_completions_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::http_utils;
use crate::services::model_names::{copilot_model_name, transform_model_for_openrouter};
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    build_google_stream_generate_content_url, convert_gemini_to_openai_chat_response,
    convert_openai_chat_to_gemini_request,
};
use crate::services::serve_responses::OpenAIToResponsesStreamConverter;
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
        converter: OpenAIToResponsesStreamConverter,
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
        .send()
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
        .send()
        .await?;

    finalize_gemini_response(response, client_wants_stream, &model).await
}

pub(crate) async fn send_openai_chat(
    body: &mut Value,
    client_wants_stream: bool,
    context: &UpstreamRequestContext,
) -> Result<RouterResponse> {
    normalize_openai_request_model(body, context.is_openrouter, context.is_copilot);

    let url = http_utils::build_chat_completions_url(&context.upstream_base_url);
    let initiator = if context.is_copilot {
        Some(http_utils::copilot_initiator_from_openai(body))
    } else {
        None
    };
    let req = http_utils::authorized_openai_post(
        &context.client,
        &url,
        context.upstream_api_key.as_str(),
        context.copilot_tokens.as_deref(),
        initiator,
    )
    .await?;

    let response = context.with_device_headers(req).json(&*body).send().await?;
    finalize_openai_response(response, client_wants_stream).await
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

    let response = context.with_device_headers(req).json(body).send().await?;
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
}
