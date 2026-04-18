/**
 * Anthropic-to-OpenAI router service
 *
 * Acts as an HTTP proxy that accepts Anthropic-format requests and routes them
 * to OpenAI-compatible providers (like Cloudflare Workers AI), handling the
 * required request and response transformations.
 *
 * Flow:
 * Anthropic /v1/messages → Router → OpenAI /v1/chat/completions
 */
use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::device_fingerprint;

use crate::services::anthropic_chat_request::{
    AnthropicToOpenAIConfig, convert_anthropic_to_openai_request,
};
use crate::services::anthropic_chat_response::{
    OpenAIToAnthropicConfig, UsageValueMode, convert_openai_to_anthropic_message,
};
use crate::services::anthropic_route_pipeline::{
    CacheControlPatch, RequestContext, RequestPatch, inject_chat_completions_cache_control,
};
use crate::services::http_utils::{self, router_http_client};
use crate::services::model_names::{
    infer_provider_name_from_model, is_gateway_style_endpoint, select_model_for_provider_attempt,
};
use crate::services::openai_anthropic_bridge::convert_openai_chat_response_to_sse;
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    convert_gemini_to_openai_chat_response, convert_openai_chat_to_gemini_request,
    openai_chat_model,
};
use crate::services::openai_models::{
    OpenAIChatChunk, OpenAIChatRequest, ResponsesResponse,
    convert_chat_to_responses_request as convert_typed_chat_to_responses_request,
    convert_responses_to_chat_response as convert_typed_responses_to_chat_response,
    stringify_message_content as stringify_typed_message_content,
};
use crate::services::protocol_fallback::{
    AttemptOutcome, commit_protocol_switch, protocol_candidates,
};
use crate::services::provider_protocol::{ProviderProtocol, is_protocol_mismatch};

#[derive(Clone)]
pub struct AnthropicToOpenAIRouterConfig {
    /// The target OpenAI-compatible provider base URL (e.g., Cloudflare)
    pub target_base_url: String,
    /// API key for the target provider
    pub target_api_key: String,
    /// The upstream protocol spoken by the provider.
    pub target_protocol: ProviderProtocol,
    /// Optional model prefix to add (e.g., "@cf/" for Cloudflare)
    pub model_prefix: Option<String>,
    /// Whether the provider requires `reasoning_content` on assistant tool-call turns (e.g., Moonshot)
    pub requires_reasoning_content: bool,
    /// Cap applied to `max_tokens` before forwarding to the provider.
    /// Use for providers with hard limits (e.g., DeepSeek: 8192).
    pub max_tokens_cap: Option<u64>,
    /// Whether this is the aivo starter provider (requires device fingerprint headers).
    pub is_starter: bool,
}

pub struct AnthropicToOpenAIRouter {
    config: AnthropicToOpenAIRouterConfig,
}

struct AnthropicToOpenAIRouterState {
    config: Arc<AnthropicToOpenAIRouterConfig>,
    client: reqwest::Client,
    active_protocol: Arc<AtomicU8>,
    /// Set to true after a native Anthropic attempt returns a protocol mismatch,
    /// so we don't waste a round-trip on every subsequent Claude request.
    native_anthropic_failed: Arc<AtomicBool>,
    /// Set to true when the provider rejects `anthropic-beta` headers (e.g. Bedrock, Vertex AI).
    /// Once learned, the header is stripped from all future requests.
    beta_header_rejected: Arc<AtomicBool>,
}

enum RouterResponse {
    Buffered {
        status: u16,
        content_type: String,
        body: Vec<u8>,
    },
    /// Already streamed to socket — nothing to write.
    AlreadyStreamed,
}

impl AnthropicToOpenAIRouter {
    pub fn new(config: AnthropicToOpenAIRouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set ANTHROPIC_BASE_URL.
    pub async fn start_background(
        &self,
    ) -> Result<(u16, Arc<AtomicU8>, tokio::task::JoinHandle<Result<()>>)> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let active_protocol = Arc::new(AtomicU8::new(self.config.target_protocol.to_u8()));
        let state = AnthropicToOpenAIRouterState {
            config: Arc::new(self.config.clone()),
            client: router_http_client(),
            active_protocol: active_protocol.clone(),
            native_anthropic_failed: Arc::new(AtomicBool::new(false)),
            beta_header_rejected: Arc::new(AtomicBool::new(false)),
        };
        let handle = tokio::spawn(async move { run_router(listener, state).await });
        Ok((port, active_protocol, handle))
    }
}

async fn run_router(
    listener: tokio::net::TcpListener,
    state: AnthropicToOpenAIRouterState,
) -> Result<()> {
    loop {
        let (mut socket, _) = listener.accept().await?;
        let config = state.config.clone();
        let client = state.client.clone();
        let active_protocol = state.active_protocol.clone();
        let native_anthropic_failed = state.native_anthropic_failed.clone();
        let beta_header_rejected = state.beta_header_rejected.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let request_bytes = match http_utils::read_full_request(&mut socket).await {
                Ok(b) => b,
                Err(err) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
            };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();

            if !http_utils::is_post_path(&request, &["/v1/messages", "/messages"]) {
                let not_found =
                    http_utils::http_response(404, CONTENT_TYPE_JSON, "{\"error\":\"Not found\"}");
                let _ = socket.write_all(not_found.as_bytes()).await;
                return;
            }

            let response = match handle_anthropic_to_upstream(
                &request,
                &config,
                &client,
                &active_protocol,
                &native_anthropic_failed,
                &beta_header_rejected,
                &mut socket,
            )
            .await
            {
                Ok(response) => response,
                Err(e) => {
                    let error = http_utils::http_error_response(500, &e.to_string());
                    let _ = socket.write_all(error.as_bytes()).await;
                    return;
                }
            };

            let _ = write_router_response(&mut socket, response).await;
        });
    }
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
) -> Result<()> {
    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            http_utils::write_buffered_response(socket, status, &content_type, &body).await?;
        }
        RouterResponse::AlreadyStreamed => {}
    }
    Ok(())
}

/// Apply an optional prefix to a model name, skipping if the prefix is already present.
fn apply_model_prefix(model: &str, prefix: Option<&str>) -> String {
    match prefix {
        Some(p) if !model.starts_with(p) => format!("{}{}", p, model),
        _ => model.to_string(),
    }
}

/// Build the standard headers for native Anthropic requests.
fn build_native_anthropic_headers(
    passthrough_headers: &HeaderMap,
    api_key: &str,
) -> Result<HeaderMap> {
    let mut headers = passthrough_headers.clone();
    headers.insert("x-api-key", HeaderValue::from_str(api_key)?);
    headers.insert("Content-Type", HeaderValue::from_static(CONTENT_TYPE_JSON));
    headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
    Ok(headers)
}

/// Try sending the request in native Anthropic format to the upstream's `/v1/messages`.
/// Returns `Some(response)` on success or non-mismatch error, `None` on protocol mismatch.
async fn try_native_anthropic(
    body: &Value,
    config: &AnthropicToOpenAIRouterConfig,
    client: &reqwest::Client,
    passthrough_headers: &HeaderMap,
    native_anthropic_failed: &AtomicBool,
    beta_header_rejected: &AtomicBool,
) -> Result<Option<RouterResponse>> {
    if native_anthropic_failed.load(Ordering::Relaxed) {
        return Ok(None);
    }

    let mut native_body = body.clone();
    let ctx = RequestContext {
        upstream_base_url: &config.target_base_url,
    };
    CacheControlPatch.patch_json("messages", &mut native_body, &ctx)?;

    let url = http_utils::build_target_url(&config.target_base_url, "/v1/messages");
    let headers = build_native_anthropic_headers(passthrough_headers, &config.target_api_key)?;

    let is_starter = config.is_starter;
    let response = device_fingerprint::maybe_with_starter_headers(
        client.post(&url).headers(headers).json(&native_body),
        is_starter,
    )
    .send()
    .await?;

    let status_code = response.status().as_u16();
    if is_protocol_mismatch(status_code) {
        native_anthropic_failed.store(true, Ordering::Relaxed);
        return Ok(None);
    }

    // Detect beta header rejection: if a 400 mentions beta-related terms,
    // learn to strip anthropic-beta and retry immediately.
    if status_code == 400 && !beta_header_rejected.load(Ordering::Relaxed) {
        let content_type = http_utils::response_content_type(&response);
        let response_body = response.bytes().await?;
        let body_str = String::from_utf8_lossy(&response_body);

        if http_utils::is_beta_header_rejection(&body_str) {
            beta_header_rejected.store(true, Ordering::Relaxed);
            eprintln!("  • Provider rejected anthropic-beta header — retrying without it");

            let mut retry_headers =
                build_native_anthropic_headers(passthrough_headers, &config.target_api_key)?;
            http_utils::strip_beta_headers(&mut retry_headers);

            let retry_response = device_fingerprint::maybe_with_starter_headers(
                client.post(&url).headers(retry_headers).json(&native_body),
                is_starter,
            )
            .send()
            .await?;

            let retry_status = retry_response.status().as_u16();
            if is_protocol_mismatch(retry_status) {
                native_anthropic_failed.store(true, Ordering::Relaxed);
                return Ok(None);
            }

            let retry_ct = http_utils::response_content_type(&retry_response);
            let retry_body = retry_response.bytes().await?;
            return Ok(Some(RouterResponse::Buffered {
                status: retry_status,
                content_type: retry_ct,
                body: retry_body.to_vec(),
            }));
        }

        return Ok(Some(RouterResponse::Buffered {
            status: status_code,
            content_type,
            body: response_body.to_vec(),
        }));
    }

    let content_type = http_utils::response_content_type(&response);
    let response_body = response.bytes().await?;
    Ok(Some(RouterResponse::Buffered {
        status: status_code,
        content_type,
        body: response_body.to_vec(),
    }))
}

/// Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions
async fn handle_anthropic_to_upstream(
    request: &str,
    config: &Arc<AnthropicToOpenAIRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    native_anthropic_failed: &Arc<AtomicBool>,
    beta_header_rejected: &Arc<AtomicBool>,
    socket: &mut tokio::net::TcpStream,
) -> Result<RouterResponse> {
    let mut passthrough_headers = http_utils::extract_passthrough_headers(request)?;
    if beta_header_rejected.load(Ordering::Relaxed) {
        http_utils::strip_beta_headers(&mut passthrough_headers);
    }
    let body_str = http_utils::extract_request_body(request)?;

    let body: Value = serde_json::from_str(body_str)?;

    // If the model is Claude, try native Anthropic first — preserves model name and prompt caching.
    let model_is_claude = body
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("claude"));

    if model_is_claude
        && let Some(response) = try_native_anthropic(
            &body,
            config,
            client,
            &passthrough_headers,
            native_anthropic_failed,
            beta_header_rejected,
        )
        .await?
    {
        return Ok(response);
    }

    let mut simplified = anthropic_to_openai(&body, config.requires_reasoning_content)?;
    // Only inject cache_control for Claude models — other providers don't support it
    // and strict ones (e.g. Cloudflare Workers AI) reject unknown fields / array content.
    if model_is_claude {
        inject_chat_completions_cache_control(&mut simplified);
    }
    // Map Anthropic thinking config to OpenAI reasoning_effort
    if let Some(thinking) = body.get("thinking")
        && thinking.get("type").and_then(|t| t.as_str()) == Some("enabled")
    {
        simplified["reasoning_effort"] = json!("high");
    }
    cap_max_tokens_field(&mut simplified, config.max_tokens_cap);
    let requested_stream = simplified
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let candidates = protocol_candidates(active_protocol);
    let mut last_response: Option<RouterResponse> = None;

    for (attempt, protocol) in candidates.into_iter().enumerate() {
        let mut req_body = simplified.clone();
        let mut attempt_headers = passthrough_headers.clone();
        prepare_gateway_model_metadata(&mut req_body, &mut attempt_headers, config, protocol);

        // Apply model prefix for OpenAI protocol
        if protocol == ProviderProtocol::Openai
            && let Some(model) = req_body.get_mut("model")
            && let Some(model_str) = model.as_str()
        {
            *model = Value::String(apply_model_prefix(
                model_str,
                config.model_prefix.as_deref(),
            ));
        }

        let outcome: AttemptOutcome<RouterResponse> = match protocol {
            ProviderProtocol::Google => {
                req_body["stream"] = json!(false);
                let model = openai_chat_model(&req_body, "gemini-2.5-pro");
                let google_body = convert_openai_chat_to_gemini_request(
                    &req_body,
                    &OpenAIToGeminiConfig {
                        default_model: "gemini-2.5-pro",
                    },
                );
                let url = build_google_generate_content_url(&config.target_base_url, &model);
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("x-goog-api-key", config.target_api_key.as_str())
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&google_body),
                    config.is_starter,
                )
                .send()
                .await?;

                let status_code = response.status().as_u16();
                let response_body = response.text().await?;
                if is_protocol_mismatch(status_code) {
                    AttemptOutcome::Mismatch {
                        status: status_code,
                        body: response_body,
                    }
                } else if status_code >= 400 {
                    AttemptOutcome::Success(RouterResponse::Buffered {
                        status: status_code,
                        content_type: CONTENT_TYPE_JSON.to_string(),
                        body: response_body.into_bytes(),
                    })
                } else {
                    let google_response: Value = serde_json::from_str(&response_body)?;
                    let openai_response =
                        convert_gemini_to_openai_chat_response(&google_response, &model);
                    let r = if requested_stream {
                        let openai_sse = convert_openai_chat_response_to_sse(&openai_response)?;
                        let anthropic_sse = convert_openai_sse_to_anthropic(&openai_sse, 200)?;
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "text/event-stream".to_string(),
                            body: anthropic_sse.into_bytes(),
                        }
                    } else {
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: CONTENT_TYPE_JSON.to_string(),
                            body: convert_openai_to_anthropic(&openai_response.to_string(), 200)?
                                .into_bytes(),
                        }
                    };
                    AttemptOutcome::Success(r)
                }
            }
            ProviderProtocol::ResponsesApi => {
                let mut responses_body = convert_chat_to_responses_request(&req_body)?;
                responses_body["stream"] = json!(false);
                let url = build_responses_url(&config.target_base_url);
                let response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("Authorization", format!("Bearer {}", config.target_api_key))
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&responses_body),
                    config.is_starter,
                )
                .send()
                .await?;

                let status_code = response.status().as_u16();
                let response_body = response.text().await?;
                if is_protocol_mismatch(status_code) || status_code == 400 {
                    AttemptOutcome::Mismatch {
                        status: status_code,
                        body: response_body,
                    }
                } else if status_code >= 400 {
                    AttemptOutcome::Success(RouterResponse::Buffered {
                        status: status_code,
                        content_type: CONTENT_TYPE_JSON.to_string(),
                        body: response_body.into_bytes(),
                    })
                } else {
                    let resp: Value = serde_json::from_str(&response_body)?;
                    let openai_response = convert_responses_to_chat_response(&resp)?;
                    let r = if requested_stream {
                        let openai_sse = convert_openai_chat_response_to_sse(&openai_response)?;
                        let anthropic_sse = convert_openai_sse_to_anthropic(&openai_sse, 200)?;
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: "text/event-stream".to_string(),
                            body: anthropic_sse.into_bytes(),
                        }
                    } else {
                        RouterResponse::Buffered {
                            status: 200,
                            content_type: CONTENT_TYPE_JSON.to_string(),
                            body: convert_openai_to_anthropic(&openai_response.to_string(), 200)?
                                .into_bytes(),
                        }
                    };
                    AttemptOutcome::Success(r)
                }
            }
            _ => {
                // OpenAI or Anthropic — use chat completions endpoint
                let url = http_utils::build_chat_completions_url(&config.target_base_url);
                let mut response = device_fingerprint::maybe_with_starter_headers(
                    client
                        .post(&url)
                        .headers(attempt_headers)
                        .header("Authorization", format!("Bearer {}", config.target_api_key))
                        .header("Content-Type", CONTENT_TYPE_JSON)
                        .json(&req_body),
                    config.is_starter,
                )
                .send()
                .await?;

                let status_code = response.status().as_u16();
                if is_protocol_mismatch(status_code) {
                    AttemptOutcome::Mismatch {
                        status: status_code,
                        body: String::new(),
                    }
                } else {
                    let is_streaming = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .map(|ct| ct.contains("text/event-stream"))
                        .unwrap_or(false);

                    // Stream OpenAI SSE → Anthropic SSE directly to socket
                    if status_code == 200 && is_streaming {
                        use tokio::io::AsyncWriteExt;
                        let headers =
                            http_utils::http_chunked_response_head(200, "text/event-stream");
                        socket.write_all(headers.as_bytes()).await?;
                        let mut converter = OpenAIStreamConverter::new();
                        while let Some(chunk) = response.chunk().await? {
                            let converted = converter.push_bytes(&chunk)?;
                            if !converted.is_empty() {
                                let formatted = http_utils::format_http_chunk(converted.as_bytes());
                                socket.write_all(&formatted).await?;
                            }
                        }
                        let tail = converter.finish()?;
                        if !tail.is_empty() {
                            let formatted = http_utils::format_http_chunk(tail.as_bytes());
                            socket.write_all(&formatted).await?;
                        }
                        socket.write_all(b"0\r\n\r\n").await?;
                        commit_protocol_switch(active_protocol, protocol, attempt);
                        return Ok(RouterResponse::AlreadyStreamed);
                    }

                    let response_body = response.text().await?;
                    // Detect Responses API validation errors leaking through
                    // Chat Completions — treat as protocol mismatch so the
                    // ResponsesApi candidate gets a chance.
                    if status_code == 400 && is_responses_api_error(&response_body) {
                        AttemptOutcome::Mismatch {
                            status: status_code,
                            body: response_body,
                        }
                    } else {
                        let r = if status_code == 200 && response_body.starts_with("data:") {
                            let anthropic_sse =
                                convert_openai_sse_to_anthropic(&response_body, status_code)?;
                            RouterResponse::Buffered {
                                status: 200,
                                content_type: "text/event-stream".to_string(),
                                body: anthropic_sse.into_bytes(),
                            }
                        } else {
                            let anthropic_response =
                                convert_openai_to_anthropic(&response_body, status_code)?;
                            RouterResponse::Buffered {
                                status: status_code,
                                content_type: CONTENT_TYPE_JSON.to_string(),
                                body: anthropic_response.into_bytes(),
                            }
                        };
                        AttemptOutcome::Success(r)
                    }
                }
            }
        };

        match outcome {
            AttemptOutcome::Success(r) => {
                commit_protocol_switch(active_protocol, protocol, attempt);
                return Ok(r);
            }
            AttemptOutcome::ProviderError { status, .. }
            | AttemptOutcome::Mismatch { status, .. } => {
                last_response = Some(RouterResponse::Buffered {
                    status,
                    content_type: CONTENT_TYPE_JSON.to_string(),
                    body: format!("{{\"error\":\"Protocol mismatch ({})\"}}", status).into_bytes(),
                });
            }
        }
    }

    // All candidates exhausted — return last error
    Ok(last_response.unwrap_or(RouterResponse::Buffered {
        status: 503,
        content_type: CONTENT_TYPE_JSON.to_string(),
        body: b"{\"error\":\"No compatible protocol found\"}".to_vec(),
    }))
}

fn anthropic_to_openai(body: &Value, requires_reasoning_content: bool) -> Result<Value> {
    let mut req = convert_anthropic_to_openai_request(
        body,
        &AnthropicToOpenAIConfig {
            default_model: "gpt-4o",
            preserve_stream: true,
            model_transform: None,
            include_reasoning_content: true,
            require_non_empty_reasoning_content: requires_reasoning_content,
            stringify_other_tool_result_content: true,
            tool_result_supports_multimodal: true,
            fallback_tool_arguments_json: "{}",
        },
    );
    let mut typed_req: OpenAIChatRequest = serde_json::from_value(req)
        .context("failed to convert anthropic request to typed OpenAI request")?;
    stringify_typed_message_content(&mut typed_req);
    req = serde_json::to_value(typed_req).context("failed to serialize typed OpenAI request")?;
    Ok(req)
}

/// Flatten any array-valued `content` fields to plain strings.
/// Strict OpenAI-compatible providers (e.g. Cloudflare Workers AI) reject
/// the multi-part content arrays that the standard OpenAI API accepts.
#[cfg(test)]
fn stringify_message_content(req: &mut Value) {
    let Ok(mut typed_req) = serde_json::from_value::<OpenAIChatRequest>(req.clone()) else {
        return;
    };
    stringify_typed_message_content(&mut typed_req);
    *req = serde_json::to_value(typed_req).expect("typed openai request should serialize");
}

/// Build /v1/responses URL from a base URL.
fn build_responses_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{}/responses", base)
    } else {
        format!("{}/v1/responses", base)
    }
}

/// Detect 400 errors that indicate the provider speaks Responses API, not Chat Completions.
fn is_responses_api_error(body: &str) -> bool {
    body.contains("\"input[") || body.contains("begins with 'fc")
}

/// Convert OpenAI Chat Completions request → Responses API request.
fn convert_chat_to_responses_request(openai_req: &Value) -> Result<Value> {
    let openai_req: OpenAIChatRequest = serde_json::from_value(openai_req.clone())
        .context("failed to parse openai chat request for responses conversion")?;
    serde_json::to_value(convert_typed_chat_to_responses_request(&openai_req))
        .context("failed to serialize responses request")
}

/// Convert Responses API response → OpenAI Chat Completions response.
fn convert_responses_to_chat_response(resp: &Value) -> Result<Value> {
    let response: ResponsesResponse =
        serde_json::from_value(resp.clone()).context("failed to parse responses API response")?;
    serde_json::to_value(convert_typed_responses_to_chat_response(&response))
        .context("failed to serialize openai chat response")
}

fn prepare_gateway_model_metadata(
    simplified: &mut Value,
    passthrough_headers: &mut HeaderMap,
    config: &AnthropicToOpenAIRouterConfig,
    protocol: ProviderProtocol,
) {
    let requested_model = simplified
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let selected_model = select_model_for_provider_attempt(
        &config.target_base_url,
        Some(&requested_model),
        None,
        protocol,
    );
    simplified["model"] = Value::String(selected_model);

    if is_gateway_style_endpoint(&config.target_base_url)
        && !passthrough_headers.contains_key("x-provider")
        && let Some(provider) = infer_provider_name_from_model(&requested_model)
        && let Ok(value) = HeaderValue::from_str(&provider)
    {
        passthrough_headers.insert("x-provider", value);
    }
}

fn cap_max_tokens_field(body: &mut Value, cap: Option<u64>) {
    let Some(limit) = cap else {
        return;
    };
    if let Some(mt) = body.get("max_tokens").and_then(http_utils::parse_token_u64)
        && mt > limit
    {
        body["max_tokens"] = json!(limit);
    }
}

/// Convert OpenAI /v1/chat/completions response to Anthropic /v1/messages format
fn convert_openai_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    // If error status, return as-is
    if status_code >= 400 {
        return Ok(response_body.to_string());
    }

    let openai_resp: Value = serde_json::from_str(response_body)?;
    let anthropic_resp = convert_openai_to_anthropic_message(
        &openai_resp,
        &OpenAIToAnthropicConfig {
            fallback_id: "msg_default",
            model: openai_resp
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown"),
            include_created: true,
            usage_value_mode: UsageValueMode::CoerceU64,
        },
    )?;

    Ok(anthropic_resp.to_string())
}

#[derive(Default)]
struct StreamToolBlock {
    anthropic_idx: usize,
    id: String,
    name: String,
    opened: bool,
    pending_args: String,
}

fn append_sse_event(output: &mut String, event: &str, data: Value) {
    output.push_str(&format!("event: {event}\ndata: {data}\n\n"));
}

fn ensure_message_start(
    output: &mut String,
    started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) {
    if *started {
        return;
    }
    let mut usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": 0
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    append_sse_event(
        output,
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage
            }
        }),
    );
    *started = true;
}

#[allow(clippy::too_many_arguments)]
fn emit_tool_delta(
    output: &mut String,
    block_count: &mut usize,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    openai_idx: usize,
    id: Option<&str>,
    name: Option<&str>,
    args_fragment: Option<&str>,
    saw_tool_use: &mut bool,
) {
    let block = tool_blocks.entry(openai_idx).or_insert_with(|| {
        let idx = *block_count;
        *block_count += 1;
        StreamToolBlock {
            anthropic_idx: idx,
            ..Default::default()
        }
    });

    if let Some(v) = id
        && !v.is_empty()
    {
        block.id = v.to_string();
    }
    if let Some(v) = name
        && !v.is_empty()
    {
        block.name = v.to_string();
    }

    if let Some(fragment) = args_fragment
        && !fragment.is_empty()
    {
        if block.opened {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragment
                    }
                }),
            );
        } else {
            block.pending_args.push_str(fragment);
        }
    }

    if !block.opened && !block.name.is_empty() {
        if block.id.is_empty() {
            block.id = format!("toolu_{}", uuid_simple());
        }
        append_sse_event(
            output,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": block.anthropic_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name
                }
            }),
        );
        block.opened = true;
        *saw_tool_use = true;

        if !block.pending_args.is_empty() {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": block.pending_args
                    }
                }),
            );
            block.pending_args.clear();
        }
    }
}

fn map_openai_finish_reason(reason: &str) -> &'static str {
    match reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_stream_message(
    output: &mut String,
    message_started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    thinking_block_idx: &mut Option<usize>,
    text_block_idx: &mut Option<usize>,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    stop_reason: &str,
) {
    ensure_message_start(
        output,
        message_started,
        message_id,
        model,
        input_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    );

    if let Some(idx) = thinking_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({"type": "content_block_stop", "index": idx}),
        );
    }

    if let Some(idx) = text_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    let mut ordered_tool_idxs = tool_blocks
        .values()
        .filter(|b| b.opened)
        .map(|b| b.anthropic_idx)
        .collect::<Vec<_>>();
    ordered_tool_idxs.sort_unstable();
    for idx in ordered_tool_idxs {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    let mut usage = json!({
        "output_tokens": output_tokens
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    append_sse_event(
        output,
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": usage
        }),
    );
    append_sse_event(
        output,
        "message_stop",
        json!({
            "type": "message_stop"
        }),
    );
}

struct OpenAIStreamConverter {
    pending: Vec<u8>,
    message_started: bool,
    finished: bool,
    block_count: usize,
    thinking_block_idx: Option<usize>,
    text_block_idx: Option<usize>,
    tool_blocks: HashMap<usize, StreamToolBlock>,
    message_id: String,
    model: String,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    saw_tool_use: bool,
}

impl OpenAIStreamConverter {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            message_started: false,
            finished: false,
            block_count: 0,
            thinking_block_idx: None,
            text_block_idx: None,
            tool_blocks: HashMap::new(),
            message_id: "msg".to_string(),
            model: "claude".to_string(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            saw_tool_use: false,
        }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.extend_from_slice(chunk);

        let mut output = String::new();
        while let Some(pos) = self.pending.iter().position(|&b| b == b'\n') {
            let line = String::from_utf8_lossy(&self.pending[..pos]).into_owned();
            self.pending.drain(..=pos);
            self.process_line(line.trim_end_matches('\r'), &mut output)?;
        }

        Ok(output)
    }

    fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        if !self.pending.is_empty() {
            let line = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            self.process_line(line.trim_end_matches('\r'), &mut output)?;
        }

        if !self.finished && self.message_started {
            let fallback_stop = if self.saw_tool_use {
                "tool_use"
            } else {
                "end_turn"
            };
            finalize_stream_message(
                &mut output,
                &mut self.message_started,
                &self.message_id,
                &self.model,
                self.input_tokens,
                self.output_tokens,
                self.cache_read_input_tokens,
                self.cache_creation_input_tokens,
                &mut self.thinking_block_idx,
                &mut self.text_block_idx,
                &mut self.tool_blocks,
                fallback_stop,
            );
            self.finished = true;
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let fallback_stop = if self.saw_tool_use {
                    "tool_use"
                } else {
                    "end_turn"
                };
                finalize_stream_message(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.output_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                    &mut self.thinking_block_idx,
                    &mut self.text_block_idx,
                    &mut self.tool_blocks,
                    fallback_stop,
                );
                self.finished = true;
            }
            return Ok(());
        }

        let chunk = match serde_json::from_str::<OpenAIChatChunk>(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };

        if let Some(v) = chunk.id.as_deref()
            && !v.is_empty()
        {
            self.message_id = v.to_string();
        }
        if let Some(v) = chunk.model.as_deref()
            && !v.is_empty()
        {
            self.model = v.to_string();
        }
        if let Some(usage) = chunk.usage {
            if let Some(v) = usage.prompt_tokens {
                self.input_tokens = v;
            }
            if let Some(v) = usage.completion_tokens {
                self.output_tokens = v;
            }
            if let Some(v) = usage.cache_read_input_tokens {
                self.cache_read_input_tokens = Some(v);
            }
            if let Some(v) = usage.cache_creation_input_tokens {
                self.cache_creation_input_tokens = Some(v);
            }
        }

        for choice in chunk.choices {
            let delta = choice.delta;

            // DeepSeek-reasoner: emit reasoning_content as Anthropic thinking blocks
            if let Some(thinking) = delta.reasoning_content.as_deref()
                && !thinking.is_empty()
            {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                if self.thinking_block_idx.is_none() {
                    let idx = self.block_count;
                    self.block_count += 1;
                    self.thinking_block_idx = Some(idx);
                    append_sse_event(
                        output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "thinking",
                                "thinking": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.thinking_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": thinking
                        }
                    }),
                );
            }

            if let Some(text) = delta.content.as_deref()
                && !text.is_empty()
            {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                // Close thinking block before starting text block
                if let Some(thinking_idx) = self.thinking_block_idx.take() {
                    append_sse_event(
                        output,
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": thinking_idx}),
                    );
                }
                if self.text_block_idx.is_none() {
                    let idx = self.block_count;
                    self.block_count += 1;
                    self.text_block_idx = Some(idx);
                    append_sse_event(
                        output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "text",
                                "text": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.text_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "text_delta",
                            "text": text
                        }
                    }),
                );
            }

            if let Some(function_call) = delta.function_call {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                emit_tool_delta(
                    output,
                    &mut self.block_count,
                    &mut self.tool_blocks,
                    0,
                    function_call.id.as_deref(),
                    function_call.name.as_deref(),
                    function_call.arguments.as_deref(),
                    &mut self.saw_tool_use,
                );
            }

            if let Some(tool_calls) = delta.tool_calls {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                for tc in tool_calls {
                    let openai_idx = tc.index.unwrap_or(0) as usize;
                    emit_tool_delta(
                        output,
                        &mut self.block_count,
                        &mut self.tool_blocks,
                        openai_idx,
                        tc.id.as_deref(),
                        tc.function.as_ref().and_then(|f| f.name.as_deref()),
                        tc.function.as_ref().and_then(|f| f.arguments.as_deref()),
                        &mut self.saw_tool_use,
                    );
                }
            }

            if !self.finished
                && let Some(finish_reason) = choice.finish_reason.as_deref()
                && !finish_reason.is_empty()
            {
                finalize_stream_message(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.output_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                    &mut self.thinking_block_idx,
                    &mut self.text_block_idx,
                    &mut self.tool_blocks,
                    map_openai_finish_reason(finish_reason),
                );
                self.finished = true;
            }
        }

        Ok(())
    }
}

/// Convert OpenAI SSE streaming response to Anthropic SSE format.
fn convert_openai_sse_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    if status_code >= 400 {
        return Ok(format!("data: {}\n\ndata: [DONE]\n\n", response_body));
    }

    let mut converter = OpenAIStreamConverter::new();
    let mut sse_output = converter.push_bytes(response_body.as_bytes())?;
    sse_output.push_str(&converter.finish()?);
    Ok(sse_output)
}

/// Generate a collision-resistant unique ID using a monotonic counter + timestamp.
fn uuid_simple() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!(
        "{:x}{:x}{:x}",
        duration.as_secs(),
        duration.subsec_nanos(),
        count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_to_anthropic_uses_response_model_and_created() {
        let openai_resp = r#"{
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 15
            }
        }"#;

        let result = convert_openai_to_anthropic(openai_resp, 200).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-123");
        assert_eq!(parsed["model"], "gpt-4");
        assert_eq!(parsed["created"], 1700000000);
        assert_eq!(parsed["usage"]["input_tokens"], 10);
        assert_eq!(parsed["usage"]["output_tokens"], 5);
        assert_eq!(parsed["usage"]["cache_read_input_tokens"], 90);
        assert_eq!(parsed["usage"]["cache_creation_input_tokens"], 15);
    }

    #[test]
    fn anthropic_to_openai_preserves_tool_result_images_through_typed_roundtrip() {
        // Regression for P0-1 Tier B: the typed round-trip in
        // `anthropic_to_openai` (OpenAIChatRequest → stringify →
        // OpenAIChatRequest) used to strip image_url parts. Images in
        // tool_result should now reach the upstream request intact.
        let body = json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_screenshot",
                    "content": [
                        {"type": "text", "text": "Screenshot of the home page."},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}}
                    ]
                }]
            }]
        });
        let req = anthropic_to_openai(&body, false).expect("conversion succeeds");
        let tool_msg = &req["messages"][0];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["tool_call_id"], "toolu_screenshot");
        let content = tool_msg["content"]
            .as_array()
            .expect("multimodal content array survives round-trip");
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Screenshot of the home page.");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "data:image/png;base64,iVBORw0KGgo"
        );
    }

    #[test]
    fn test_anthropic_to_openai_preserves_fields_and_tools() {
        let body = json!({
            "model": "gpt-4o-mini",
            "system": [{"type": "text", "text": "You are helpful."}],
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "hello"}]
            }],
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "tools": [{
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}
            }],
            "tool_choice": {"type": "tool", "name": "read_file"},
            "stream": true
        });

        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(req["model"], "gpt-4o-mini");
        assert_eq!(req["stream"], true);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
        assert_eq!(req["max_tokens"], 128);
        assert_eq!(req["temperature"], 0.2);
        assert_eq!(req["top_p"], 0.9);
        assert_eq!(req["stop"][0], "END");
        assert_eq!(req["tools"][0]["type"], "function");
        assert_eq!(req["tools"][0]["function"]["name"], "read_file");
        assert_eq!(
            req["tool_choice"],
            json!({"type": "function", "function": {"name": "read_file"}})
        );
    }

    #[test]
    fn test_cap_max_tokens_field_caps_numeric_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 12000});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 8192);
    }

    #[test]
    fn test_cap_max_tokens_field_caps_numeric_string_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": "12000"});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 8192);
    }

    #[test]
    fn test_cap_max_tokens_field_ignores_non_numeric_string_value() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": "oops"});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], "oops");
    }

    #[test]
    fn test_anthropic_to_openai_maps_thinking_to_reasoning_content_for_tool_calls() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect files first."},
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(
            messages[0]["reasoning_content"],
            "Need to inspect files first."
        );
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_anthropic_to_openai_sets_reasoning_content_for_assistant_tool_calls_without_thinking() {
        let body = json!({
            "model": "kimi-k2.5",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, true).unwrap();
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["reasoning_content"], " ");
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["name"],
            "list_files"
        );
    }

    #[test]
    fn test_prepare_gateway_model_metadata_preserves_gateway_claude_model() {
        let config = AnthropicToOpenAIRouterConfig {
            target_base_url: "https://api.ai.example-gateway.net/endpoint".to_string(),
            target_api_key: "test".to_string(),
            target_protocol: ProviderProtocol::Openai,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(&mut body, &mut headers, &config, config.target_protocol);

        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(
            headers.get("x-provider").and_then(|v| v.to_str().ok()),
            Some("anthropic")
        );
    }

    #[test]
    fn test_prepare_gateway_model_metadata_remaps_plain_openai_endpoint() {
        let config = AnthropicToOpenAIRouterConfig {
            target_base_url: "https://api.openai.com/v1".to_string(),
            target_api_key: "test".to_string(),
            target_protocol: ProviderProtocol::Openai,
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            is_starter: false,
        };
        let mut body = json!({"model": "claude-sonnet-4-6"});
        let mut headers = HeaderMap::new();

        prepare_gateway_model_metadata(&mut body, &mut headers, &config, config.target_protocol);

        assert_eq!(body["model"], "gpt-4o");
        assert!(headers.get("x-provider").is_none());
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_text() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4,\"cache_read_input_tokens\":90,\"cache_creation_input_tokens\":15}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("event: message_start"));
        assert!(result.contains("\"type\":\"text_delta\""));
        assert!(result.contains("\"text\":\"hello \""));
        assert!(result.contains("\"text\":\"world\""));
        assert!(result.contains("\"stop_reason\":\"end_turn\""));
        assert!(result.contains("\"cache_read_input_tokens\":90"));
        assert!(result.contains("\"cache_creation_input_tokens\":15"));
        assert!(result.contains("event: message_stop"));
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_split_tool_calls() {
        let sse = "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"list_files\"}}]},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("\"type\":\"tool_use\""));
        assert!(result.contains("\"id\":\"call_1\""));
        assert!(result.contains("\"name\":\"list_files\""));
        assert!(result.contains("\"type\":\"input_json_delta\""));
        assert!(result.contains("\"partial_json\":\"{\\\"path\\\":\\\".\\\"}\""));
        assert!(result.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_openai_stream_converter_handles_split_chunks() {
        let mut converter = OpenAIStreamConverter::new();
        let mut output = String::new();

        output.push_str(
            &converter
                .push_bytes(b"data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hel")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"lo\"},\"finish_reason\":null}]}\n")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n")
                .unwrap(),
        );
        output.push_str(&converter.push_bytes(b"data: [DONE]\n").unwrap());
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"text\":\"hello\""));
        assert!(output.contains("\"text\":\" world\""));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert_eq!(output.matches("event: message_stop").count(), 1);
    }

    #[test]
    fn test_model_prefix() {
        // Test the production helper directly to catch regressions in the real code path
        assert_eq!(
            apply_model_prefix("glm-4.7-flash", Some("@cf/")),
            "@cf/glm-4.7-flash"
        );
        // Prefix already present — must not double-add
        assert_eq!(
            apply_model_prefix("@cf/llama-3.1-8b", Some("@cf/")),
            "@cf/llama-3.1-8b"
        );
        // No prefix configured
        assert_eq!(apply_model_prefix("llama-3.1-8b", None), "llama-3.1-8b");
    }

    #[test]
    fn test_anthropic_to_openai_keeps_content_on_tool_call_messages() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {"path": "."}}
                ]
            }]
        });

        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();

        // content must be present (null) for strict providers like Cloudflare Workers AI
        assert!(
            messages[0].get("content").is_some(),
            "assistant tool_call message must retain content field"
        );
        assert!(messages[0]["tool_calls"].is_array());
    }

    #[test]
    fn test_stringify_message_content_flattens_arrays() {
        let mut req = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]},
                {"role": "assistant", "content": "already a string"},
                {"role": "user", "content": ["plain", "strings"]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}
            ]
        });

        stringify_message_content(&mut req);
        let messages = req["messages"].as_array().unwrap();

        assert_eq!(messages[0]["content"], "hello\nworld");
        assert_eq!(messages[1]["content"], "already a string");
        assert_eq!(messages[2]["content"], "plain\nstrings");
        // Array carrying a non-text kind stays as an array — flattening
        // would silently erase the payload, and strict providers should
        // fail loudly rather than receive corrupted data.
        assert!(messages[3]["content"].is_array());
        assert_eq!(messages[3]["content"][0]["type"], "image_url");
    }

    #[test]
    fn test_convert_openai_to_anthropic_error_status_passthrough() {
        let error_body = r#"{"error":{"message":"rate limited"}}"#;
        let result = convert_openai_to_anthropic(error_body, 429).unwrap();
        // Error responses should be passed through as-is
        assert!(result.contains("rate limited"));
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_error_status_passthrough() {
        let error_body = r#"{"error":"upstream down"}"#;
        let result = convert_openai_sse_to_anthropic(error_body, 502).unwrap();
        assert!(result.contains("upstream down"));
        assert!(result.contains("data: "));
    }

    #[test]
    fn test_convert_openai_to_anthropic_empty_body() {
        let result = convert_openai_to_anthropic("", 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_openai_to_anthropic_malformed_json() {
        let result = convert_openai_to_anthropic("{not valid}", 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_anthropic_to_openai_empty_messages() {
        let body = json!({
            "model": "gpt-4o",
            "messages": []
        });
        let req = anthropic_to_openai(&body, false).unwrap();
        let messages = req["messages"].as_array().unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_cap_max_tokens_field_no_cap() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 12000});
        cap_max_tokens_field(&mut req, None);
        assert_eq!(req["max_tokens"], 12000);
    }

    #[test]
    fn test_cap_max_tokens_field_under_cap() {
        let mut req = json!({"model": "gpt-4o", "max_tokens": 4096});
        cap_max_tokens_field(&mut req, Some(8192));
        assert_eq!(req["max_tokens"], 4096);
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_empty_sse() {
        let result = convert_openai_sse_to_anthropic("", 200).unwrap();
        // Empty input → no events emitted (converter never started)
        assert!(result.is_empty());
    }

    #[test]
    fn test_openai_stream_converter_malformed_json_in_data_line() {
        let mut converter = OpenAIStreamConverter::new();
        // Malformed JSON should be silently skipped, not error
        let output = converter
            .push_bytes(b"data: {invalid json}\ndata: [DONE]\n")
            .unwrap();
        let tail = converter.finish().unwrap();
        // Should not panic and should produce empty/minimal output
        let _ = output;
        let _ = tail;
    }

    #[test]
    fn test_stringify_message_content_null_content() {
        let mut req = json!({
            "messages": [{"role": "assistant", "content": null}]
        });
        stringify_message_content(&mut req);
        // null content should remain unchanged (not crash)
        let messages = req["messages"].as_array().unwrap();
        assert!(messages[0]["content"].is_null());
    }

    #[test]
    fn is_responses_api_error_detects_input_bracket() {
        // The detector looks for literal `"input[` in the raw body string,
        // matching how OpenAI formats Responses API validation errors.
        assert!(is_responses_api_error(
            r#"{"error":{"message":"Invalid value for \"input[0].content\""}}"#
        ));
    }

    #[test]
    fn is_responses_api_error_detects_fc_prefix() {
        assert!(is_responses_api_error(
            r#"{"error":"tool call id begins with 'fc' which is reserved"}"#
        ));
    }

    #[test]
    fn is_responses_api_error_false_for_normal_error() {
        assert!(!is_responses_api_error(r#"{"error":"rate limited"}"#));
    }

    #[test]
    fn build_responses_url_with_v1_suffix() {
        let url = build_responses_url("https://api.openai.com/v1");
        assert_eq!(url, "https://api.openai.com/v1/responses");
        // Must not produce /v1/v1/responses
        assert!(!url.contains("/v1/v1"));
    }

    #[test]
    fn build_responses_url_without_v1_suffix() {
        let url = build_responses_url("https://api.example.com");
        assert_eq!(url, "https://api.example.com/v1/responses");
    }

    #[test]
    fn convert_openai_sse_to_anthropic_done_only() {
        // A stream consisting of only the [DONE] sentinel should produce
        // a minimal valid Anthropic SSE stream (message_start + message_stop).
        let result = convert_openai_sse_to_anthropic("data: [DONE]\n", 200).unwrap();
        assert!(
            result.contains("event: message_start"),
            "must emit message_start"
        );
        assert!(
            result.contains("event: message_stop"),
            "must emit message_stop"
        );
        assert!(
            result.contains("\"stop_reason\":\"end_turn\""),
            "must have a stop_reason"
        );
    }
}
