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
use crate::services::anthropic_chat_request::ensure_assistant_reasoning_content_in_chat_request;
use crate::services::anthropic_route_pipeline::inject_chat_completions_cache_control;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::device_fingerprint;
use crate::services::http_debug::LoggedSend;
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
use crate::services::provider_protocol::{
    PathVariant, ProviderProtocol, classify_failed_attempt, decode_route, is_endpoint_missing,
};
use crate::services::responses_chat_conversion;
use anyhow::Result;
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

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
    /// Persisted path-variant pin from a prior launch. When set, the router
    /// skips re-probing the alternate variant.
    pub target_path_variant: Option<PathVariant>,
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
    /// Cap applied to `max_tokens` / `max_output_tokens` before forwarding to the provider,
    /// for providers that reject values above a fixed output ceiling.
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
    /// Flipped to `true` once any request returns a non-error response. Read
    /// by `persist_runtime_discoveries` to gate protocol pinning.
    request_succeeded: Arc<AtomicBool>,
    /// Flipped to `true` when the cascade observes an authoritative response
    /// (2xx success or a 4xx with a parseable LLM-API error envelope). Used
    /// by `persist_runtime_discoveries` to persist `claude_path_variant`
    /// even when no 2xx was seen — the path responded, so it's safe to
    /// remember which variant won.
    saw_authoritative_response: Arc<AtomicBool>,
    /// Flipped to `true` when an upstream returns an error envelope matching
    /// the `requires_reasoning_content` quirk. Persisted to `ApiKey` so future
    /// launches enable strict mode without hardcoding the host.
    learned_requires_reasoning: Arc<AtomicBool>,
    /// Consecutive non-2xx responses against the active route. After
    /// `CONSECUTIVE_FAILURES_BEFORE_RESET`, the active route is reset so the
    /// next request re-probes from the configured default — recovers when an
    /// upstream changes shape mid-session.
    consecutive_failures: Arc<AtomicU8>,
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
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (listener, port) = http_utils::bind_local_listener().await?;
        let initial_route = crate::services::provider_protocol::encode_route(
            self.config.target_protocol,
            self.config
                .target_path_variant
                .unwrap_or(PathVariant::Default),
        );
        let active_protocol = Arc::new(AtomicU8::new(initial_route));
        let initial_responses = match self.config.responses_api_supported {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        };
        let responses_api_supported = Arc::new(AtomicU8::new(initial_responses));
        let request_succeeded = Arc::new(AtomicBool::new(false));
        let saw_authoritative_response = Arc::new(AtomicBool::new(false));
        let learned_requires_reasoning = Arc::new(AtomicBool::new(false));
        let consecutive_failures = Arc::new(AtomicU8::new(0));
        let state = ResponsesToChatRouterState {
            config: Arc::new(self.config.clone()),
            client: Arc::new(http_utils::router_http_client()),
            active_protocol: active_protocol.clone(),
            responses_api_supported: responses_api_supported.clone(),
            request_succeeded: request_succeeded.clone(),
            saw_authoritative_response: saw_authoritative_response.clone(),
            learned_requires_reasoning: learned_requires_reasoning.clone(),
            consecutive_failures,
        };
        let handle = tokio::spawn(async move {
            http_utils::run_streaming_router(
                listener,
                Arc::new(state),
                handle_router_request_streaming,
            )
            .await
        });
        Ok((
            port,
            active_protocol,
            responses_api_supported,
            request_succeeded,
            saw_authoritative_response,
            learned_requires_reasoning,
            handle,
        ))
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
        // Track success/failure for the auto-clear-on-N-failures heuristic.
        // Streamed responses (where `response` is None) bypass this — they
        // already updated `request_succeeded` from inside the streamer.
        let succeeded = response_is_2xx(&response);
        crate::services::protocol_fallback::record_request_outcome(
            state.active_protocol.as_ref(),
            state.consecutive_failures.as_ref(),
            state.config.target_protocol,
            state
                .config
                .target_path_variant
                .unwrap_or(PathVariant::Default),
            succeeded,
        );
        let _ = socket.write_all(response.as_bytes()).await;
    }
}

/// Quick check on a buffered HTTP/1.1 response string: status starts at byte
/// offset 9 (after "HTTP/1.1 "). 2xx → success.
fn response_is_2xx(http_response: &str) -> bool {
    http_response
        .as_bytes()
        .get(9..12)
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<u16>().ok())
        .is_some_and(|status| (200..300).contains(&status))
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
            &state.request_succeeded,
            &state.saw_authoritative_response,
            &state.learned_requires_reasoning,
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
#[allow(clippy::too_many_arguments)]
async fn handle_api_request(
    path: &str,
    request: &str,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    responses_api_supported: &Arc<AtomicU8>,
    request_succeeded: &Arc<AtomicBool>,
    saw_authoritative_response: &Arc<AtomicBool>,
    learned_requires_reasoning: &Arc<AtomicBool>,
    socket: &mut tokio::net::TcpStream,
) -> Result<Option<String>> {
    let body_str = http_utils::extract_request_body(request)?;
    let body: Value = serde_json::from_str(body_str)?;

    if is_responses_api_format(&body) {
        // When the upstream supports the Responses API natively, forward directly
        // to preserve IDs and avoid lossy Chat Completions round-trip conversion.
        let (current, variant) = decode_route(active_protocol.load(Ordering::Relaxed));
        if current == ProviderProtocol::Openai
            && let Some(result) = try_responses_api_passthrough(
                variant,
                &body,
                config,
                client,
                responses_api_supported,
                request_succeeded,
            )
            .await
        {
            return Ok(Some(result?));
        }
        Ok(Some(
            handle_responses_api_via_chat(
                path,
                &body,
                config,
                client,
                active_protocol,
                request_succeeded,
                saw_authoritative_response,
                learned_requires_reasoning,
            )
            .await?,
        ))
    } else {
        // For streaming Chat Completions, stream directly from upstream to client
        if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            && stream_chat_completions(
                &body,
                config,
                client,
                active_protocol,
                request_succeeded,
                socket,
            )
            .await
            .is_ok()
        {
            return Ok(None); // already streamed to socket
        }
        Ok(Some(
            handle_chat_completions_with_filter(
                path,
                &body,
                config,
                client,
                active_protocol,
                request_succeeded,
                saw_authoritative_response,
                learned_requires_reasoning,
            )
            .await?,
        ))
    }
}

// =============================================================================
// RESPONSES API PATH: passthrough or convert
// =============================================================================

/// Decision returned by `classify_responses_passthrough_error` for a non-200
/// response received while probing or using the native Responses API endpoint.
#[derive(Debug, PartialEq, Eq)]
enum ResponsesPassthroughDecision {
    /// Path is genuinely missing — latch `responses_api_supported = 2` and fall
    /// through to chat conversion.
    LatchUnsupported,
    /// Transient error during the unknown-state probe — leave state at 0 so the
    /// next request re-probes; fall through to chat for this request.
    Reprobe,
    /// Already-known supported (or non-mismatch) — surface the error to the client.
    PassError,
}

/// Decide what to do with a non-200 response from `/v1/responses` based on the
/// current `responses_api_supported` state byte. The state encoding is:
///   0 = unknown (probing)
///   1 = supported
///   2 = unsupported
///
/// Only `(unknown, endpoint-missing)` should latch unsupported. Every other
/// non-200 in the unknown state should re-probe — a transient 429/5xx must not
/// permanently disable Responses API for the key.
fn classify_responses_passthrough_error(state: u8, status: u16) -> ResponsesPassthroughDecision {
    let unknown = state == 0;
    if unknown {
        if is_endpoint_missing(status) {
            ResponsesPassthroughDecision::LatchUnsupported
        } else {
            ResponsesPassthroughDecision::Reprobe
        }
    } else {
        ResponsesPassthroughDecision::PassError
    }
}

/// Tries to forward a Responses API request directly to the upstream `/v1/responses`
/// endpoint. Returns `Some(Ok(response))` on success or non-protocol HTTP errors,
/// `None` if the upstream doesn't support the Responses API (404/405/415), allowing
/// fallback to Chat Completions conversion.
async fn try_responses_api_passthrough(
    variant: PathVariant,
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    responses_api_supported: &Arc<AtomicU8>,
    request_succeeded: &Arc<AtomicBool>,
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

    let target_url = build_target_url(&config.target_base_url, variant.apply("/v1/responses"));
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
            .send_logged()
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
        match classify_responses_passthrough_error(
            responses_api_supported.load(Ordering::Relaxed),
            status,
        ) {
            ResponsesPassthroughDecision::LatchUnsupported => {
                responses_api_supported.store(2, Ordering::Relaxed);
                return None;
            }
            ResponsesPassthroughDecision::Reprobe => return None,
            ResponsesPassthroughDecision::PassError => {
                return Some(Ok(http_utils::http_response(
                    status,
                    &content_type,
                    &response_body,
                )));
            }
        }
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
    request_succeeded.store(true, Ordering::Relaxed);
    Some(Ok(http_utils::http_response(
        status,
        &content_type,
        &response_body,
    )))
}

/// Handles Responses API requests by converting to Chat Completions format,
/// forwarding to the provider, and converting the response back to Responses
/// API SSE format that the Codex CLI expects.
#[allow(clippy::too_many_arguments)]
async fn handle_responses_api_via_chat(
    _path: &str,
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    request_succeeded: &Arc<AtomicBool>,
    saw_authoritative_response: &Arc<AtomicBool>,
    learned_requires_reasoning: &Arc<AtomicBool>,
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
    let chat_response = match forward_openai_chat_request(
        &chat_body,
        config,
        client,
        false,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
    )
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
    if config.requires_reasoning_content {
        ensure_assistant_reasoning_content_in_chat_request(&mut body);
    }
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
    request_succeeded: &Arc<AtomicBool>,
    socket: &mut tokio::net::TcpStream,
) -> Result<()> {
    // Only stream for OpenAI protocol (the common case for DeepSeek, etc.)
    let (protocol, variant) = decode_route(active_protocol.load(Ordering::Relaxed));
    if protocol != ProviderProtocol::Openai {
        anyhow::bail!("streaming passthrough only for OpenAI protocol");
    }

    let body = prepare_chat_completions_body(body, config, protocol);

    let target_url = build_target_url(
        &config.target_base_url,
        variant.apply("/v1/chat/completions"),
    );
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
            .send_logged()
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

    request_succeeded.store(true, Ordering::Relaxed);
    http_utils::write_streaming_response(socket, 200, &content_type, response).await
}

// =============================================================================
// CHAT COMPLETIONS PATH: filter tools and forward (buffered)
// =============================================================================

#[allow(clippy::too_many_arguments)]
async fn handle_chat_completions_with_filter(
    _path: &str,
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    active_protocol: &Arc<AtomicU8>,
    request_succeeded: &Arc<AtomicBool>,
    saw_authoritative_response: &Arc<AtomicBool>,
    learned_requires_reasoning: &Arc<AtomicBool>,
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

    let chat_response = match forward_openai_chat_request(
        &body,
        config,
        client,
        requested_stream,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
    )
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
        .send_logged()
        .await?;
    http_utils::buffered_reqwest_to_http_response(response).await
}

#[allow(clippy::too_many_arguments)]
async fn forward_openai_chat_request(
    body: &Value,
    config: &Arc<ResponsesToChatRouterConfig>,
    client: &reqwest::Client,
    force_non_streaming: bool,
    active_protocol: &Arc<AtomicU8>,
    request_succeeded: &Arc<AtomicBool>,
    saw_authoritative_response: &Arc<AtomicBool>,
    learned_requires_reasoning: &Arc<AtomicBool>,
) -> Result<ForwardedChatResponse> {
    // Openai and ResponsesApi both route through `forward_openai_protocol`,
    // so one is a byte-identical duplicate of the other. Drop the non-active.
    let active_proto = decode_route(active_protocol.load(Ordering::Relaxed)).0;
    let candidates: Vec<(ProviderProtocol, PathVariant)> = protocol_candidates(active_protocol)
        .into_iter()
        .filter(|(proto, _)| {
            !matches!(
                (active_proto, *proto),
                (ProviderProtocol::Openai, ProviderProtocol::ResponsesApi)
                    | (ProviderProtocol::ResponsesApi, ProviderProtocol::Openai)
            )
        })
        .collect();
    let mut first_error: Option<(u16, String)> = None;
    let mut body_for_attempts = body.clone();
    let mut retried_with_strict = false;
    let mut idx = 0;

    while idx < candidates.len() {
        let (protocol, variant) = candidates[idx];
        let attempt = idx;
        match forward_chat_for_protocol(
            protocol,
            variant,
            &body_for_attempts,
            config.as_ref(),
            client,
            force_non_streaming,
        )
        .await?
        {
            AttemptOutcome::Success(value) => {
                commit_protocol_switch(active_protocol, protocol, variant, attempt);
                request_succeeded.store(true, Ordering::Relaxed);
                saw_authoritative_response.store(true, Ordering::Relaxed);
                return Ok(ForwardedChatResponse::Success(value));
            }
            AttemptOutcome::Mismatch {
                status,
                body: response_body,
            } => {
                // A 4xx whose body is a recognizable LLM-API error envelope is
                // a semantic rejection — bail out instead of probing other
                // protocols that will return the same rejection.
                let classification = classify_failed_attempt(status, &response_body);
                if classification.is_terminal
                    || classification.is_semantic_rejection
                    || first_error.is_none()
                {
                    first_error = Some((status, response_body));
                }
                if classification.is_semantic_rejection {
                    saw_authoritative_response.store(true, Ordering::Relaxed);
                    if classification.quirk_hint == Some("requires_reasoning_content") {
                        learned_requires_reasoning.store(true, Ordering::Relaxed);
                        if !retried_with_strict && !config.requires_reasoning_content {
                            retried_with_strict = true;
                            body_for_attempts = body.clone();
                            ensure_assistant_reasoning_content_in_chat_request(
                                &mut body_for_attempts,
                            );
                            continue;
                        }
                    }
                    if attempt > 0 {
                        commit_protocol_switch(active_protocol, protocol, variant, attempt);
                    }
                    break;
                }
                // Skip the fast-bail at attempt 0: a 401/403 there often means
                // "this host rejected the protocol's auth header shape" rather
                // than "your key is bad" (e.g. cross-protocol gateways). Probe
                // at least one fallback before believing the upstream.
                if classification.is_terminal && attempt > 0 {
                    // The path answered (with an error, but it answered) —
                    // pin it in memory so retry storms from codex/claude
                    // don't re-probe the wrong chat/completions paths every
                    // time. Don't flip request_succeeded: we still don't
                    // want to *persist* the pin to disk based on a 5xx.
                    commit_protocol_switch(active_protocol, protocol, variant, attempt);
                    break;
                }
            }
        }
        idx += 1;
    }

    let (status, body) = first_error.unwrap_or_default();
    Ok(ForwardedChatResponse::HttpError { status, body })
}

async fn forward_chat_for_protocol(
    protocol: ProviderProtocol,
    variant: PathVariant,
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
    force_non_streaming: bool,
) -> Result<AttemptOutcome<Value>> {
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
            forward_openai_protocol(variant, body, config, client).await
        }
        ProviderProtocol::Anthropic => {
            forward_anthropic_protocol(variant, body, config, client, force_non_streaming).await
        }
        ProviderProtocol::Google => forward_google_protocol(body, config, client).await,
    }
}

async fn forward_openai_protocol(
    variant: PathVariant,
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
) -> Result<AttemptOutcome<Value>> {
    let target_url = build_target_url(
        &config.target_base_url,
        variant.apply("/v1/chat/completions"),
    );
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
            .send_logged()
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
        && let AttemptOutcome::Mismatch {
            body: ref error_body,
            ..
        } = result
        && (error_body.contains("unsupported_api_for_model")
            || (error_body.contains("not support") && error_body.contains("chat/completions")))
        && let Ok(fallback) = try_copilot_responses_fallback(variant, body, config, client).await
    {
        return Ok(fallback);
    }

    Ok(result)
}

/// Converts a Chat Completions request to Responses API format and sends it to
/// Copilot's /responses endpoint. Returns the response converted back to Chat
/// Completions format.
async fn try_copilot_responses_fallback(
    variant: PathVariant,
    body: &Value,
    config: &ResponsesToChatRouterConfig,
    client: &reqwest::Client,
) -> Result<AttemptOutcome<Value>> {
    let responses_body = responses_chat_conversion::convert_chat_to_responses_request(body);
    let target_url = build_target_url(&config.target_base_url, variant.apply("/v1/responses"));
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
    .send_logged()
    .await?;

    let status = response.status().as_u16();
    let body_text = response.text().await?;
    if status == 200 {
        let resp_value: Value = serde_json::from_str(&body_text)?;
        let chat_value = responses_chat_conversion::convert_responses_json_to_chat(&resp_value);
        Ok(AttemptOutcome::Success(chat_value))
    } else {
        Ok(AttemptOutcome::Mismatch {
            status,
            body: body_text,
        })
    }
}

async fn forward_anthropic_protocol(
    variant: PathVariant,
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

    let target_url = build_target_url(&config.target_base_url, variant.apply("/v1/messages"));
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
    .send_logged()
    .await?;

    let status_code = response.status().as_u16();
    let response_text = response.text().await?;
    if status_code == 200 {
        let anthropic_response: Value = serde_json::from_str(&response_text)?;
        let openai_response = convert_anthropic_to_openai_chat_response(
            &anthropic_response,
            body.get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("gpt-4o"),
        );
        return Ok(AttemptOutcome::Success(openai_response));
    }
    // Anthropic error bodies wrap the OpenAI-shape `{"error":{...}}` in an
    // outer `{"type":"error", "error":{...}}` envelope. Codex/openai clients
    // can't parse that and fall back to generic "high demand" messages —
    // strip the wrap so the real upstream message reaches the user.
    let normalized = strip_anthropic_error_wrap(&response_text);
    Ok(classify_attempt::<Value>(status_code, normalized, None))
}

fn strip_anthropic_error_wrap(body: &str) -> String {
    let Ok(parsed) = serde_json::from_str::<Value>(body) else {
        return body.to_string();
    };
    if parsed.get("type").and_then(|v| v.as_str()) == Some("error")
        && let Some(inner) = parsed.get("error")
    {
        return json!({ "error": inner }).to_string();
    }
    body.to_string()
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
    .send_logged()
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

    #[test]
    fn response_is_2xx_recognises_status_line() {
        assert!(response_is_2xx("HTTP/1.1 200 OK\r\n\r\n{}"));
        assert!(response_is_2xx("HTTP/1.1 204 No Content\r\n"));
        assert!(!response_is_2xx("HTTP/1.1 502 Bad Gateway\r\n"));
        assert!(!response_is_2xx("HTTP/1.1 429 Too Many Requests\r\n"));
        assert!(!response_is_2xx("garbage"));
    }

    #[test]
    fn classify_passthrough_error_latches_only_on_endpoint_missing_when_unknown() {
        for status in [404, 405, 415, 501] {
            assert_eq!(
                classify_responses_passthrough_error(0, status),
                ResponsesPassthroughDecision::LatchUnsupported,
                "endpoint-missing status {status} should latch when state is unknown",
            );
        }
    }

    #[test]
    fn classify_passthrough_error_reprobes_on_transient_when_unknown() {
        // Transient errors (rate-limit, 5xx, server timeout) on the first probe
        // must NOT latch the key as unsupported. The next request should re-probe.
        for status in [400, 401, 403, 408, 422, 429, 500, 502, 503, 504] {
            assert_eq!(
                classify_responses_passthrough_error(0, status),
                ResponsesPassthroughDecision::Reprobe,
                "transient status {status} should re-probe when state is unknown",
            );
        }
    }

    #[test]
    fn classify_passthrough_error_passes_error_to_client_when_known_supported() {
        // Once we've confirmed the upstream supports Responses API, any non-200
        // is the upstream's real answer — surface it to the client unchanged.
        for status in [400, 401, 403, 404, 422, 429, 500, 502, 503] {
            assert_eq!(
                classify_responses_passthrough_error(1, status),
                ResponsesPassthroughDecision::PassError,
                "status {status} should pass through when state is known-supported",
            );
        }
    }

    #[test]
    fn classify_passthrough_error_passes_error_when_known_unsupported() {
        // State == 2 means we've already given up on /v1/responses for this key,
        // so the chat-fallback path should be in use; this branch only fires
        // when something weird sends us back through the passthrough function.
        // Either way, treat as PassError (not Reprobe) so we don't accidentally
        // re-enable.
        assert_eq!(
            classify_responses_passthrough_error(2, 500),
            ResponsesPassthroughDecision::PassError,
        );
    }

    #[test]
    fn strip_anthropic_error_wrap_unwraps_minimax_500_body() {
        // Real body observed from minimax /v1/messages on a plan/billing error.
        let body = r#"{"type":"error","error":{"type":"api_error","message":"your current token plan not support model, MiniMax-M2.7 (2061)"},"request_id":"abc"}"#;
        let stripped = strip_anthropic_error_wrap(body);
        let parsed: Value = serde_json::from_str(&stripped).unwrap();
        // Outer envelope is gone; codex-readable shape: { error: { type, message } }.
        assert!(parsed.get("type").is_none());
        assert_eq!(
            parsed["error"]["message"],
            "your current token plan not support model, MiniMax-M2.7 (2061)"
        );
        assert_eq!(parsed["error"]["type"], "api_error");
    }

    #[test]
    fn strip_anthropic_error_wrap_passes_through_openai_shape() {
        let body = r#"{"error":{"message":"bad key","type":"invalid_request"}}"#;
        // No outer `"type":"error"` wrap → returned unchanged.
        assert_eq!(strip_anthropic_error_wrap(body), body);
    }

    #[test]
    fn strip_anthropic_error_wrap_passes_through_invalid_json() {
        let body = "<html>504 Gateway Timeout</html>";
        assert_eq!(strip_anthropic_error_wrap(body), body);
    }

    #[test]
    fn prepare_chat_completions_body_adds_reasoning_for_plain_assistant_in_strict_mode() {
        let config = ResponsesToChatRouterConfig {
            target_base_url: "https://api.example.com".to_string(),
            api_key: "sk-test".to_string(),
            target_protocol: ProviderProtocol::Openai,
            target_path_variant: None,
            copilot_token_manager: None,
            model_prefix: None,
            requires_reasoning_content: true,
            actual_model: None,
            max_tokens_cap: None,
            responses_api_supported: None,
            is_starter: false,
        };
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": "OK, continuing."}]
        });

        let prepared = prepare_chat_completions_body(&body, &config, ProviderProtocol::Openai);
        let messages = prepared["messages"].as_array().unwrap();
        assert_eq!(messages[0]["reasoning_content"], "OK, continuing.");
    }

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
