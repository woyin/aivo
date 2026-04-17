/**
 * Responses-to-Chat router service
 *
 * Acts as an HTTP proxy that accepts OpenAI Responses API requests and forwards
 * them to upstreams that may only support Chat Completions or other protocols.
 *
 * 1. Tool filtering: strips built-in tool types (computer_use, file_search,
 *    web_search, code_interpreter) that most non-OpenAI providers reject.
 *
 * 2. Responses API conversion: clients like Codex CLI use `/v1/responses`
 *    with `input` items. Providers that only support `/v1/chat/completions`
 *    need a full request/response conversion. This router handles that automatically.
 *
 * Conversion logic (Responses API ↔ Chat Completions) lives in
 * `responses_chat_conversion.rs` and is re-exported here for backwards compatibility.
 */
use crate::constants::CONTENT_TYPE_JSON;
use crate::services::anthropic_route_pipeline::inject_chat_completions_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::http_utils::{self};
use crate::services::model_names::select_model_for_provider_attempt;
use crate::services::openai_anthropic_bridge::{
    OpenAIToAnthropicChatConfig, convert_anthropic_to_openai_chat_response,
    convert_openai_chat_response_to_sse, convert_openai_chat_to_anthropic_request,
};
use crate::services::openai_gemini_bridge::{
    OpenAIToGeminiConfig, build_google_generate_content_url,
    convert_gemini_to_openai_chat_response, convert_openai_chat_to_gemini_request,
    openai_chat_model,
};
use crate::services::protocol_fallback::{
    AttemptOutcome, classify_attempt, commit_protocol_switch, protocol_candidates,
};
use crate::services::provider_protocol::{ProviderProtocol, is_protocol_mismatch};
use crate::services::responses_chat_conversion;
use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

// Re-export public conversion functions used by other modules
pub use responses_chat_conversion::{
    convert_chat_response_to_responses_sse, convert_responses_to_chat_request,
    is_responses_api_format, parse_provider_response,
};

// Internal re-exports used within this router
use responses_chat_conversion::{
    apply_max_tokens_cap_to_fields, cap_reasoning_effort, sanitize_input_content,
};

#[derive(Clone)]
pub struct ResponsesToChatRouterConfig {
    pub target_base_url: String,
    pub api_key: String,
    pub target_protocol: ProviderProtocol,
    pub copilot_token_manager: Option<Arc<CopilotTokenManager>>,
    /// Optional model prefix to add (e.g., "@cf/" for Cloudflare)
    pub model_prefix: Option<String>,
    /// Whether the provider requires a non-empty `reasoning_content` sentinel on assistant
    /// tool-call turns even when the provider returned no reasoning text (e.g., Moonshot).
    /// Auto-detection handles the normal case: if the provider returns `reasoning_content`
    /// in a response it is always round-tripped, regardless of this flag.
    pub requires_reasoning_content: bool,
    /// The actual model name to use with the provider (e.g., "kimi-k2.5" while Codex CLI sees "gpt-4o")
    pub actual_model: Option<String>,
    /// Cap applied to `max_tokens` / `max_output_tokens` before forwarding to the provider.
    /// Use for providers with hard limits (e.g., DeepSeek: 8192).
    pub max_tokens_cap: Option<u64>,
    /// Persisted Responses API support state: None = unknown, Some(true) = supported,
    /// Some(false) = not supported.  Avoids a wasted probe request on every session.
    pub responses_api_supported: Option<bool>,
    /// Whether this is the aivo starter provider (requires device fingerprint headers).
    pub is_starter: bool,
}

pub struct ResponsesToChatRouter {
    config: ResponsesToChatRouterConfig,
}

enum ForwardedChatResponse {
    Success(Value),
    HttpError { status: u16, body: String },
}

struct ResponsesToChatRouterState {
    config: Arc<ResponsesToChatRouterConfig>,
    client: Arc<reqwest::Client>,
    active_protocol: Arc<AtomicU8>,
    /// Tri-state: 0 = unknown, 1 = supported, 2 = not supported
    responses_api_supported: Arc<AtomicU8>,
}

impl ResponsesToChatRouter {
    pub fn new(config: ResponsesToChatRouterConfig) -> Self {
        Self { config }
    }

    /// Binds to a random available port and starts the router in the background.
    /// Returns the actual port number so callers can set OPENAI_BASE_URL.
    pub async fn start_background(
        &self,
    ) -> Result<(
        u16,
        Arc<AtomicU8>,
        Arc<AtomicU8>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let active_protocol = Arc::new(AtomicU8::new(self.config.target_protocol.to_u8()));
        let initial_responses = match self.config.responses_api_supported {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        };
        let responses_api_supported = Arc::new(AtomicU8::new(initial_responses));
        let state = ResponsesToChatRouterState {
            config: Arc::new(self.config.clone()),
            client: Arc::new(http_utils::router_http_client()),
            active_protocol: active_protocol.clone(),
            responses_api_supported: responses_api_supported.clone(),
        };
        let handle = tokio::spawn(async move {
            http_utils::run_streaming_router(
                listener,
                Arc::new(state),
                handle_router_request_streaming,
            )
            .await
        });
        Ok((port, active_protocol, responses_api_supported, handle))
    }
}

async fn handle_router_request_streaming(
    request: String,
    state: Arc<ResponsesToChatRouterState>,
    mut socket: tokio::net::TcpStream,
) {
    use tokio::io::AsyncWriteExt;
    let response = handle_router_request(request, &state, &mut socket).await;
    if let Some(response) = response {
        let _ = socket.write_all(response.as_bytes()).await;
    }
}

/// Returns `Some(response)` for buffered responses, `None` if the handler
/// already streamed the response directly to the socket.
async fn handle_router_request(
    request: String,
    state: &ResponsesToChatRouterState,
    socket: &mut tokio::net::TcpStream,
) -> Option<String> {
    let path = http_utils::extract_request_path(&request);

    let is_api_path = matches!(
        path.as_str(),
        "/responses" | "/v1/responses" | "/chat/completions" | "/v1/chat/completions"
    );

    if is_api_path {
        match handle_api_request(
            &path,
            &request,
            &state.config,
            state.client.as_ref(),
            &state.active_protocol,
            &state.responses_api_supported,
            socket,
        )
        .await
        {
            Ok(r) => r,
            Err(_) => Some(http_utils::http_error_response(
                500,
                "Internal Server Error",
            )),
        }
    } else {
        match forward_request(&path, &request, &state.config, state.client.as_ref()).await {
            Ok(r) => Some(r),
            Err(_) => Some(http_utils::http_error_response(502, "Bad Gateway")),
        }
    }
}

/// Routes the request based on body format:
/// - Responses API format (has "input" array): convert ↔ Chat Completions
/// - Chat Completions format: filter non-function tools and forward
///
/// Returns `None` if the response was streamed directly to the socket.
async fn handle_api_request(
    path: &str,
    request: &str,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    responses_api_supported: &Arc<AtomicU8>,
    socket: &mut tokio::net::TcpStream,
) -> Result<Option<String>> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    if is_responses_api_format(&body) {
        // When the upstream supports the Responses API natively, forward directly
        // to preserve IDs and avoid lossy Chat Completions round-trip conversion.
        let current = ProviderProtocol::from_u8(active_protocol.load(Ordering::Relaxed));
        if current == ProviderProtocol::Openai
            && let Some(result) =
                try_responses_api_passthrough(&body, config, client, responses_api_supported).await
        {
            return Ok(Some(result?));
        }
        Ok(Some(
            handle_responses_api_via_chat(path, &body, config, client, active_protocol).await?,
        ))
    } else {
        // For streaming Chat Completions, stream directly from upstream to client
        if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            && stream_chat_completions(&body, config, client, active_protocol, socket)
                .await
                .is_ok()
        {
            return Ok(None); // already streamed to socket
        }
        Ok(Some(
            handle_chat_completions_with_filter(path, &body, config, client, active_protocol)
                .await?,
        ))
    }
}

// =============================================================================
// RESPONSES API PATH: passthrough or convert
// =============================================================================

/// Tries to forward a Responses API request directly to the upstream `/v1/responses`
/// endpoint. Returns `Some(Ok(response))` on success or non-protocol HTTP errors,
/// `None` if the upstream doesn't support the Responses API (404/405/415), allowing
/// fallback to Chat Completions conversion.
async fn try_responses_api_passthrough(
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    responses_api_supported: &Arc<AtomicU8>,
) -> Option<Result<String>> {
    if responses_api_supported.load(Ordering::Relaxed) == 2 {
        return None;
    }

    let mut body = body.clone();
    // Don't filter_tools here — the upstream Responses API supports all tool types
    // (computer_use_preview, web_search_preview, code_interpreter, etc.).
    // Tool filtering is only needed for the Chat Completions conversion path.

    // Strip Chat Completions-only parameters that the Responses API doesn't accept
    if let Some(obj) = body.as_object_mut() {
        obj.remove("stream_options");
    }
    // Cap reasoning effort: xhigh is not supported by most models
    cap_reasoning_effort(&mut body);
    // Ensure all message content parts have a `text` field — the Responses API
    // rejects `output_text` / `input_text` parts that are missing it.
    sanitize_input_content(&mut body);
    apply_max_tokens_cap_to_fields(&mut body, config.max_tokens_cap, &["max_output_tokens"]);
    apply_selected_model(&mut body, config.as_ref(), ProviderProtocol::Openai);

    let target_url = build_target_url(&config.target_base_url, "/v1/responses");
    let req = http_utils::authorized_openai_post(
        client,
        &target_url,
        &config.api_key,
        config.copilot_token_manager.as_deref(),
        None,
    )
    .await
    .ok()?;
    let response =
        device_fingerprint::maybe_with_starter_headers(req.json(&body), config.is_starter)
            .send()
            .await
            .ok()?;

    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(CONTENT_TYPE_JSON)
        .to_string();
    let response_body = response.text().await.ok()?;

    if status != 200 {
        if responses_api_supported.load(Ordering::Relaxed) == 0 || is_protocol_mismatch(status) {
            // Still probing or clear protocol mismatch — mark unsupported, fall through
            responses_api_supported.store(2, Ordering::Relaxed);
            return None;
        }
        // Known supported but got an error — return it to the client
        return Some(Ok(http_utils::http_response(
            status,
            &content_type,
            &response_body,
        )));
    }

    // Only validate on the first probe (unknown state).  Once confirmed,
    // skip the scan over the full response body on every subsequent request.
    if responses_api_supported.load(Ordering::Relaxed) != 1 {
        let looks_like_responses_api = if content_type.contains("text/event-stream") {
            response_body.contains("response.completed")
        } else {
            response_body.contains("\"object\":\"response\"")
                || response_body.contains("\"object\": \"response\"")
        };

        if !looks_like_responses_api {
            responses_api_supported.store(2, Ordering::Relaxed);
            return None;
        }

        responses_api_supported.store(1, Ordering::Relaxed);
    }
    Some(Ok(http_utils::http_response(
        status,
        &content_type,
        &response_body,
    )))
}

/// Handles Responses API requests by converting to Chat Completions format,
/// forwarding to the provider, and converting the response back to Responses
/// API SSE format that the Codex CLI expects.
async fn handle_responses_api_via_chat(
    _path: &str,
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
) -> Result<String> {
    // Extract original model before conversion
    let original_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o")
        .to_string();

    // Create a config copy with the model pinned to avoid protocol-based transformation
    // before we know which protocol the fallback loop will select.
    let mut chat_config = (**config).clone();
    chat_config.actual_model = Some(original_model.clone());
    let chat_body = convert_responses_to_chat_request(body, &chat_config);
    let chat_response =
        match forward_openai_chat_request(&chat_body, config, client, false, active_protocol)
            .await?
        {
            ForwardedChatResponse::Success(value) => value,
            ForwardedChatResponse::HttpError { status, body } => {
                return Ok(http_utils::http_json_response(status, &body));
            }
        };
    let sse = convert_chat_response_to_responses_sse(
        &chat_response,
        config.requires_reasoning_content,
        &original_model,
    );

    Ok(http_utils::http_response(200, "text/event-stream", &sse))
}

// =============================================================================
// CHAT COMPLETIONS PATH: streaming passthrough
// =============================================================================

/// Applies shared request transforms (tool filtering, token caps, model selection).
fn prepare_chat_completions_body(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    protocol: ProviderProtocol,
) -> Value {
    let mut body = body.clone();
    filter_tools(&mut body);
    apply_max_tokens_cap_to_fields(
        &mut body,
        config.max_tokens_cap,
        &["max_tokens", "max_output_tokens"],
    );
    apply_selected_model(&mut body, config, protocol);
    body
}

/// Streams a Chat Completions request directly from upstream to the client socket,
/// forwarding SSE chunks in real time so reasoning_content appears progressively.
async fn stream_chat_completions(
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    socket: &mut tokio::net::TcpStream,
) -> Result<()> {
    // Only stream for OpenAI protocol (the common case for DeepSeek, etc.)
    let protocol = ProviderProtocol::from_u8(active_protocol.load(Ordering::Relaxed));
    if protocol != ProviderProtocol::Openai {
        anyhow::bail!("streaming passthrough only for OpenAI protocol");
    }

    let body = prepare_chat_completions_body(body, config, protocol);

    let target_url = build_target_url(&config.target_base_url, "/v1/chat/completions");
    let initiator = if config.copilot_token_manager.is_some() {
        Some(http_utils::copilot_initiator_from_openai(&body))
    } else {
        None
    };
    let req = http_utils::authorized_openai_post(
        client,
        &target_url,
        &config.api_key,
        config.copilot_token_manager.as_deref(),
        initiator,
    )
    .await?;
    let response =
        device_fingerprint::maybe_with_starter_headers(req.json(&body), config.is_starter)
            .send()
            .await?;

    let status = response.status().as_u16();
    if status != 200 {
        anyhow::bail!("upstream returned {status}");
    }

    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();

    http_utils::write_streaming_response(socket, 200, &content_type, response).await
}

// =============================================================================
// CHAT COMPLETIONS PATH: filter tools and forward (buffered)
// =============================================================================

async fn handle_chat_completions_with_filter(
    _path: &str,
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
) -> Result<String> {
    let body = prepare_chat_completions_body(
        body,
        config,
        ProviderProtocol::from_u8(active_protocol.load(Ordering::Relaxed)),
    );
    let requested_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let chat_response =
        match forward_openai_chat_request(&body, config, client, requested_stream, active_protocol)
            .await?
        {
            ForwardedChatResponse::Success(value) => value,
            ForwardedChatResponse::HttpError { status, body } => {
                return Ok(http_utils::http_json_response(status, &body));
            }
        };
    if requested_stream {
        let sse = convert_openai_chat_response_to_sse(&chat_response)?;
        Ok(http_utils::http_response(200, "text/event-stream", &sse))
    } else {
        Ok(http_utils::http_json_response(
            200,
            &serde_json::to_string(&chat_response)?,
        ))
    }
}

// =============================================================================
// PASSTHROUGH
// =============================================================================

/// Forwards a request as-is to the target provider (for non-API paths)
async fn forward_request(
    path: &str,
    request: &str,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
) -> Result<String> {
    let body_str = http_utils::extract_request_body(request)?;

    let target_url = build_target_url(&config.target_base_url, path);

    let mut req = http_utils::authorized_openai_post(
        client,
        &target_url,
        &config.api_key,
        config.copilot_token_manager.as_deref(),
        None,
    )
    .await?;

    if !body_str.is_empty() {
        req = req.body(body_str.to_string());
    }

    let response = device_fingerprint::maybe_with_starter_headers(req, config.is_starter)
        .send()
        .await?;
    http_utils::buffered_reqwest_to_http_response(response).await
}

async fn forward_openai_chat_request(
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    force_non_streaming: bool,
    active_protocol: &Arc<AtomicU8>,
) -> Result<ForwardedChatResponse> {
    let candidates = protocol_candidates(active_protocol);
    let mut last_status = 0u16;
    let mut last_body = String::new();

    for (attempt, protocol) in candidates.into_iter().enumerate() {
        match forward_chat_for_protocol(
            protocol,
            body,
            config.as_ref(),
            client,
            force_non_streaming,
        )
        .await?
        {
            AttemptOutcome::Success(value) => {
                commit_protocol_switch(active_protocol, protocol, attempt);
                return Ok(ForwardedChatResponse::Success(value));
            }
            AttemptOutcome::ProviderError { status, body } => {
                return Ok(ForwardedChatResponse::HttpError { status, body });
            }
            AttemptOutcome::Mismatch { status, body } => {
                last_status = status;
                last_body = body;
            }
        }
    }

    Ok(ForwardedChatResponse::HttpError {
        status: last_status,
        body: last_body,
    })
}

async fn forward_chat_for_protocol(
    protocol: ProviderProtocol,
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
    force_non_streaming: bool,
) -> Result<AttemptOutcome<Value>> {
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
            forward_openai_protocol(body, config, client).await
        }
        ProviderProtocol::Anthropic => {
            forward_anthropic_protocol(body, config, client, force_non_streaming).await
        }
        ProviderProtocol::Google => forward_google_protocol(body, config, client).await,
    }
}

async fn forward_openai_protocol(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
) -> Result<AttemptOutcome<Value>> {
    let target_url = build_target_url(&config.target_base_url, "/v1/chat/completions");
    let initiator = if config.copilot_token_manager.is_some() {
        Some(http_utils::copilot_initiator_from_openai(body))
    } else {
        None
    };
    let req = http_utils::authorized_openai_post(
        client,
        &target_url,
        &config.api_key,
        config.copilot_token_manager.as_deref(),
        initiator,
    )
    .await?;
    let response =
        device_fingerprint::maybe_with_starter_headers(req.json(body), config.is_starter)
            .send()
            .await?;

    let status = response.status().as_u16();
    let body_text = response.text().await?;
    let parsed = if status == 200 {
        Some(parse_provider_response(&body_text)?)
    } else {
        None
    };
    let result = classify_attempt(status, body_text, parsed);

    // If Copilot rejected the model specifically because /chat/completions
    // is unsupported for it, fall back to /responses.
    if config.copilot_token_manager.is_some()
        && let AttemptOutcome::ProviderError {
            body: ref error_body,
            ..
        } = result
        && (error_body.contains("unsupported_api_for_model")
            || (error_body.contains("not support") && error_body.contains("chat/completions")))
        && let Ok(fallback) = try_copilot_responses_fallback(body, config, client).await
    {
        return Ok(fallback);
    }

    Ok(result)
}

/// Converts a Chat Completions request to Responses API format and sends it to
/// Copilot's /responses endpoint. Returns the response converted back to Chat
/// Completions format.
async fn try_copilot_responses_fallback(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
) -> Result<AttemptOutcome<Value>> {
    let responses_body = responses_chat_conversion::convert_chat_to_responses_request(body);
    let target_url = build_target_url(&config.target_base_url, "/v1/responses");
    let req = http_utils::authorized_openai_post(
        client,
        &target_url,
        &config.api_key,
        config.copilot_token_manager.as_deref(),
        None,
    )
    .await?;
    let response = device_fingerprint::maybe_with_starter_headers(
        req.json(&responses_body),
        config.is_starter,
    )
    .send()
    .await?;

    let status = response.status().as_u16();
    let body_text = response.text().await?;
    if status == 200 {
        let resp_value: Value = serde_json::from_str(&body_text)?;
        let chat_value = responses_chat_conversion::convert_responses_json_to_chat(&resp_value);
        Ok(AttemptOutcome::Success(chat_value))
    } else {
        Ok(AttemptOutcome::ProviderError {
            status,
            body: body_text,
        })
    }
}

async fn forward_anthropic_protocol(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
    force_non_streaming: bool,
) -> Result<AttemptOutcome<Value>> {
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

    let mut anthropic_body = convert_openai_chat_to_anthropic_request(
        &body_with_cache,
        &OpenAIToAnthropicChatConfig {
            default_model: "claude-sonnet-4-5",
        },
    );
    if force_non_streaming {
        anthropic_body["stream"] = json!(false);
    }

    let target_url = build_target_url(&config.target_base_url, "/v1/messages");
    let response = device_fingerprint::maybe_with_starter_headers(
        client
            .post(&target_url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("x-api-key", config.api_key.as_str())
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&anthropic_body),
        config.is_starter,
    )
    .send()
    .await?;

    let status_code = response.status().as_u16();
    let response_text = response.text().await?;
    let parsed = if status_code == 200 {
        let anthropic_response: Value = serde_json::from_str(&response_text)?;
        Some(convert_anthropic_to_openai_chat_response(
            &anthropic_response,
            body.get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("gpt-4o"),
        ))
    } else {
        None
    };
    Ok(classify_attempt(status_code, response_text, parsed))
}

async fn forward_google_protocol(
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
) -> Result<AttemptOutcome<Value>> {
    let google_body = convert_openai_chat_to_gemini_request(
        body,
        &OpenAIToGeminiConfig {
            default_model: "gemini-2.5-pro",
        },
    );
    let model = openai_chat_model(body, "gemini-2.5-pro");
    let target_url = build_google_generate_content_url(&config.target_base_url, &model);
    let response = device_fingerprint::maybe_with_starter_headers(
        client
            .post(&target_url)
            .header("x-goog-api-key", config.api_key.as_str())
            .header("Content-Type", CONTENT_TYPE_JSON)
            .json(&google_body),
        config.is_starter,
    )
    .send()
    .await?;

    let status_code = response.status().as_u16();
    let response_text = response.text().await?;
    let parsed = if status_code == 200 {
        let google_response: Value = serde_json::from_str(&response_text)?;
        Some(convert_gemini_to_openai_chat_response(
            &google_response,
            &model,
        ))
    } else {
        None
    };
    Ok(classify_attempt(status_code, response_text, parsed))
}

// =============================================================================
// URL HELPERS
// =============================================================================

/// Constructs target URL, avoiding /v1 duplication when base already ends with /v1
fn build_target_url(base_url: &str, path: &str) -> String {
    http_utils::build_target_url(base_url, path)
}

// =============================================================================
// TOOL FILTERING
// =============================================================================

/// Removes non-"function" tools from the request body.
/// If the tools array becomes empty, removes the key entirely.
fn filter_tools(body: &mut Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        tools.retain(|t| t.get("type").and_then(|v| v.as_str()) == Some("function"));
        if tools.is_empty() {
            body.as_object_mut().map(|o| o.remove("tools"));
        }
    }
}

// =============================================================================
// MODEL TRANSFORM
// =============================================================================

/// For OpenRouter, prefixes model with "openai/" if not already namespaced.
/// Also applies a custom prefix (e.g., "@cf/" for Cloudflare) if configured.
fn transform_model(body: &mut Value, base_url: &str, model_prefix: Option<&str>) {
    if let Some(model) = body["model"].as_str().map(String::from) {
        let transformed = transform_model_str(&model, base_url, model_prefix);
        if transformed != model {
            body["model"] = Value::String(transformed);
        }
    }
}

pub(crate) fn transform_model_str(
    model: &str,
    base_url: &str,
    model_prefix: Option<&str>,
) -> String {
    // First apply custom prefix (e.g., "@cf/" for Cloudflare)
    let with_prefix = if let Some(prefix) = model_prefix {
        if !model.starts_with(prefix) {
            format!("{}{}", prefix, model)
        } else {
            model.to_string()
        }
    } else {
        model.to_string()
    };

    // Then apply OpenRouter prefix if needed
    if base_url.contains("openrouter") && !with_prefix.contains('/') {
        format!("openai/{}", with_prefix)
    } else {
        with_prefix
    }
}

fn apply_selected_model(
    body: &mut Value,
    config: &ResponsesToChatRouterConfig,
    protocol: ProviderProtocol,
) {
    if config.copilot_token_manager.is_some() {
        // For Copilot, only apply model name normalization (e.g. claude-sonnet-4-6 → claude-sonnet-4.6)
        if let Some(model) = body.get("model").and_then(|v| v.as_str()) {
            let copilot_name = crate::services::model_names::copilot_model_name(model);
            if copilot_name != model {
                body["model"] = Value::String(copilot_name);
            }
        }
        return;
    }

    let selected_model = select_model_for_provider_attempt(
        &config.target_base_url,
        body.get("model").and_then(|v| v.as_str()),
        config.actual_model.as_deref(),
        protocol,
    );
    body["model"] = Value::String(selected_model);

    if protocol == ProviderProtocol::Openai {
        transform_model(
            body,
            &config.target_base_url,
            config.model_prefix.as_deref(),
        );
    }
}

// Conversion logic (Responses API ↔ Chat Completions) has been extracted to
// responses_chat_conversion.rs and is re-exported at the top of this file.

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── HTTP body extraction ────────────────────────────────────────────────────

    #[test]
    fn test_extract_request_body_normal() {
        let req = "POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"model\":\"gpt-4\"}";
        assert_eq!(
            http_utils::extract_request_body(req).unwrap(),
            "{\"model\":\"gpt-4\"}"
        );
    }

    #[test]
    fn test_extract_request_body_missing_separator_returns_error() {
        let req = "POST /v1/chat/completions HTTP/1.1";
        assert!(http_utils::extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short_request_no_panic() {
        assert!(http_utils::extract_request_body("AB").is_err());
    }

    // ── Tool filtering ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_tools_removes_non_function() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [
                {"type": "function", "function": {"name": "my_fn"}},
                {"type": "computer_use"},
                {"type": "file_search"},
                {"type": "web_search"},
                {"type": "code_interpreter"}
            ]
        });

        filter_tools(&mut body);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
    }

    #[test]
    fn test_filter_tools_all_non_function_removes_key() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [{"type": "computer_use"}, {"type": "web_search"}]
        });
        filter_tools(&mut body);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_filter_tools_already_function_only_unchanged() {
        let mut body = json!({
            "model": "gpt-4",
            "tools": [
                {"type": "function", "function": {"name": "fn1"}},
                {"type": "function", "function": {"name": "fn2"}}
            ]
        });
        filter_tools(&mut body);
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_filter_tools_no_tools_key_is_noop() {
        let mut body = json!({"model": "gpt-4", "messages": []});
        filter_tools(&mut body);
        assert!(body.get("tools").is_none());
        assert_eq!(body["model"], "gpt-4");
    }

    // ── Model transform ────────────────────────────────────────────────────────

    #[test]
    fn test_transform_model_openrouter_adds_prefix() {
        let mut body = json!({"model": "gpt-4o"});
        transform_model(&mut body, "https://openrouter.ai/api/v1", None);
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn test_transform_model_openrouter_already_prefixed() {
        let mut body = json!({"model": "openai/gpt-4o"});
        transform_model(&mut body, "https://openrouter.ai/api/v1", None);
        assert_eq!(body["model"], "openai/gpt-4o");
    }

    #[test]
    fn test_transform_model_non_openrouter_passthrough() {
        let mut body = json!({"model": "gpt-4o"});
        transform_model(&mut body, "https://ai-gateway.vercel.sh/v1", None);
        assert_eq!(body["model"], "gpt-4o");
    }

    #[test]
    fn test_transform_model_cloudflare_prefix() {
        let mut body = json!({"model": "glm-4.7-flash"});
        transform_model(
            &mut body,
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1",
            Some("@cf/"),
        );
        assert_eq!(body["model"], "@cf/glm-4.7-flash");
    }

    #[test]
    fn test_transform_model_cloudflare_prefix_already_present() {
        let mut body = json!({"model": "@cf/llama-3.1-8b"});
        transform_model(
            &mut body,
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1",
            Some("@cf/"),
        );
        assert_eq!(body["model"], "@cf/llama-3.1-8b");
    }

    // ── URL building ───────────────────────────────────────────────────────────

    #[test]
    fn test_build_target_url_strips_v1_duplication() {
        let url = build_target_url("https://ai-gateway.vercel.sh/v1", "/v1/responses");
        assert_eq!(url, "https://ai-gateway.vercel.sh/v1/responses");
    }

    #[test]
    fn test_build_target_url_no_v1_in_path() {
        let url = build_target_url("https://ai-gateway.vercel.sh/v1", "/responses");
        assert_eq!(url, "https://ai-gateway.vercel.sh/v1/responses");
    }

    #[test]
    fn test_build_target_url_base_no_v1() {
        let url = build_target_url("https://api.example.com", "/v1/responses");
        assert_eq!(url, "https://api.example.com/v1/responses");
    }
}
