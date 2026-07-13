//! Serve Router — exposes a local OpenAI-compatible HTTP API.
//!
//! Clients send OpenAI-format requests; this router transforms them to whatever
//! protocol the active upstream provider requires, forwards them, and returns
//! OpenAI-format responses.

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::commands::models::fetch_models;
use crate::constants::CONTENT_TYPE_JSON;
use crate::services::copilot_auth::CopilotTokenManager;
use crate::services::http_utils::{self, router_http_client_with_timeout};
use crate::services::log_store::{LogEvent, LogStore};
use crate::services::model_list_response;
use crate::services::protocol_fallback::{
    FirstError, MismatchDirective, commit_protocol_switch, mismatch_directive, protocol_candidates,
    record_request_outcome,
};
use crate::services::provider_protocol::{
    PathVariant, ProviderProtocol, classify_failed_attempt, is_protocol_mismatch,
};
use crate::services::request_log::RequestLogger;
use crate::services::responses_chat_conversion::ResponsesStreamConverter;
use crate::services::responses_to_chat_router::{
    ResponsesToChatRouterConfig, collect_custom_tool_names, convert_chat_response_to_responses_sse,
    convert_responses_to_chat_request,
};
use crate::services::route_cache::{RouteCache, RouteSlot};
use crate::services::serve_responses::{
    convert_chat_response_to_responses_json, convert_chat_sse_to_responses_sse,
};
use crate::services::serve_upstream::{
    RouterResponse, StreamingBody, UpstreamRequestContext, copilot_requires_responses_api,
    send_anthropic_chat, send_copilot_responses, send_gemini_chat, send_openai_chat,
    send_openai_embeddings,
};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::usage_stats_store::RunTokenTally;

use std::sync::LazyLock;

static HEALTH_RESPONSE: LazyLock<Vec<u8>> = LazyLock::new(|| {
    json!({"status": "ok", "version": crate::version::VERSION})
        .to_string()
        .into_bytes()
});

/// A random 32-char alphanumeric bearer token for a serve/endpoint instance.
pub(crate) fn random_auth_token() -> String {
    use rand::Rng;
    rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

pub struct ServeRouterConfig {
    pub upstream_base_url: String,
    pub upstream_api_key: String,
    pub upstream_protocol: ProviderProtocol,
    pub is_copilot: bool,
    pub is_openrouter: bool,
    pub is_starter: bool,
    pub is_joycode: bool,
    pub cors: bool,
    pub timeout: u64,
    pub auth_token: Option<String>,
    /// Snapshot of model aliases taken at startup. The router rewrites
    /// `body["model"]` against this map before any protocol-specific handler
    /// runs; clients that POST `{"model": "<alias>", ...}` reach the real
    /// upstream model. Edits to aliases after launch require a restart.
    pub aliases: HashMap<String, String>,
}

impl ServeRouterConfig {
    /// Build the config from a resolved key, deriving the upstream URL +
    /// protocol/provider flags from its provider profile. The caller supplies
    /// the serving knobs (cors/timeout/auth/aliases). Shared by `aivo serve` and
    /// the plugin loopback endpoint.
    pub(crate) fn from_key(
        key: &ApiKey,
        cors: bool,
        timeout: u64,
        auth_token: Option<String>,
        aliases: HashMap<String, String>,
    ) -> Self {
        use crate::services::provider_profile::{
            provider_profile_for_key, resolve_starter_base_url,
        };
        let profile = provider_profile_for_key(key);
        Self {
            upstream_base_url: resolve_starter_base_url(&key.base_url),
            upstream_api_key: key.key.as_str().to_string(),
            upstream_protocol: profile.default_protocol,
            is_copilot: profile.serve_flags.is_copilot,
            is_openrouter: profile.serve_flags.is_openrouter,
            is_starter: profile.serve_flags.is_starter,
            is_joycode: crate::services::joycode_auth::is_joycode_key(&key.base_url),
            cors,
            timeout,
            auth_token,
            aliases,
        }
    }
}

pub struct ServeRouter {
    config: ServeRouterConfig,
    key: ApiKey,
    log_store: LogStore,
    logger: Option<RequestLogger>,
    failover_keys: Vec<ApiKey>,
    /// When set, buffered 2xx responses have their token usage recorded against
    /// `key` in stats. Off by default (plain `aivo serve` doesn't account); the
    /// plugin endpoint opts in via `with_usage_accounting`.
    usage_sink: Option<SessionStore>,
    /// Tool label for accounted requests (the plugin name); `None` → "serve".
    usage_tool: Option<String>,
    /// Per-run token tally for the plugin endpoint, so the run's finished log row
    /// carries timestamped tokens (windowable by `aivo stats --since`). Fed at the
    /// same point as `usage_sink`; `None` for plain `aivo serve`.
    run_tally: Option<Arc<RunTokenTally>>,
    /// Suppress the router's progress lines (protocol auto-switch, failover) on
    /// stderr. `aivo code` runs this router in-process behind a raw-mode TUI, so
    /// stray `eprintln!`s would corrupt the screen / land in the prompt box.
    quiet: bool,
    /// Caller-owned route cache for the primary upstream. When set, the serve
    /// learns into it instead of a throwaway one, so the owner can seed a known
    /// protocol and read confirmed routes back. `aivo code` shares one across its
    /// per-turn serves.
    seed_route_cache: Option<Arc<RouteCache>>,
}

struct ServeState {
    config: Arc<ServeRouterConfig>,
    client: reqwest::Client,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
    /// Per-model learned protocol routes (in-memory only — `aivo serve` doesn't
    /// persist routes yet). Replaces the old single per-process pin so a
    /// multi-model gateway key learns a route per model instead of thrashing
    /// one scalar.
    route_cache: Arc<RouteCache>,
    log_store: LogStore,
    logger: Option<RequestLogger>,
    failover_keys: Arc<Vec<FailoverEntry>>,
    shutdown: Arc<tokio::sync::Notify>,
    usage_sink: Option<SessionStore>,
    usage_tool: Option<String>,
    run_tally: Option<Arc<RunTokenTally>>,
    /// Mirror of `ServeRouter::quiet` — suppresses stderr progress lines.
    quiet: bool,
}

struct FailoverEntry {
    config: Arc<ServeRouterConfig>,
    key: ApiKey,
    copilot_tokens: Option<Arc<CopilotTokenManager>>,
    /// Shared across failover attempts so a route learned during one failover
    /// carries to the next request instead of being re-probed every time.
    route_cache: Arc<RouteCache>,
}

impl ServeRouter {
    pub fn new(config: ServeRouterConfig, key: ApiKey, log_store: LogStore) -> Self {
        Self {
            config,
            key,
            log_store,
            logger: None,
            failover_keys: Vec::new(),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
            seed_route_cache: None,
        }
    }

    /// Use a caller-owned route cache so the learned protocol can be seeded and
    /// read back. Used by `aivo code` to share one across its per-turn serves.
    pub fn with_route_cache(mut self, cache: Arc<RouteCache>) -> Self {
        self.seed_route_cache = Some(cache);
        self
    }

    pub fn with_logger(mut self, logger: Option<RequestLogger>) -> Self {
        self.logger = logger;
        self
    }

    /// Silence the router's stderr progress lines (protocol auto-switch,
    /// failover). Set by `aivo code`, whose TUI owns the terminal.
    pub fn quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    pub fn with_failover_keys(mut self, keys: Vec<ApiKey>) -> Self {
        self.failover_keys = keys;
        self
    }

    /// Record token usage of buffered 2xx responses against the bound key,
    /// labeling them with `tool` in logs. Used by the plugin endpoint so a
    /// coding-agent plugin routing through the loopback gets token/cost stats.
    pub fn with_usage_accounting(mut self, store: SessionStore, tool: String) -> Self {
        self.usage_sink = Some(store);
        self.usage_tool = Some(tool);
        self
    }

    /// Also fold accounted usage into a per-run tally, so the plugin run's
    /// finished log row carries timestamped tokens for `aivo stats --since`.
    pub fn with_run_tally(mut self, tally: Arc<RunTokenTally>) -> Self {
        self.run_tally = Some(tally);
        self
    }

    /// Binds to the port eagerly (propagates "address already in use" immediately),
    /// then spawns the accept loop in the background and returns the join handle
    /// and a shutdown notifier.
    pub async fn start_background(
        self,
        host: &str,
        port: u16,
    ) -> Result<(
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
    )> {
        let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
        Ok(self.spawn_on(listener))
    }

    /// Like `start_background`, but also returns the actually-bound port — for
    /// `port: 0` (OS-assigned), which the plugin endpoint uses to avoid clashing
    /// with a user's own `aivo serve`.
    pub async fn start_background_with_addr(
        self,
        host: &str,
        port: u16,
    ) -> Result<(
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
        u16,
    )> {
        let listener = tokio::net::TcpListener::bind(format!("{}:{}", host, port)).await?;
        let bound = listener.local_addr()?.port();
        let (handle, shutdown) = self.spawn_on(listener);
        Ok((handle, shutdown, bound))
    }

    /// Build the shared state and spawn the accept loop over an already-bound
    /// listener.
    fn spawn_on(
        self,
        listener: tokio::net::TcpListener,
    ) -> (
        tokio::task::JoinHandle<Result<()>>,
        Arc<tokio::sync::Notify>,
    ) {
        let copilot_tokens = if self.config.is_copilot {
            Some(Arc::new(CopilotTokenManager::new(
                self.config.upstream_api_key.clone(),
            )))
        } else {
            None
        };

        let initial_protocol = self.config.upstream_protocol;
        let timeout = self.config.timeout;
        // A caller-owned cache (shared across `aivo code` turns) wins; otherwise
        // each run gets a fresh throwaway cache seeded at the upstream protocol.
        let route_cache = self.seed_route_cache.unwrap_or_else(|| {
            Arc::new(RouteCache::new(
                "serve",
                initial_protocol,
                std::collections::BTreeMap::new(),
            ))
        });

        // Failover handlers re-parse the RAW request, so each failover config
        // needs the alias map too — an aliased model name would otherwise
        // reach the failover upstream unresolved and 400.
        let failover_aliases = self.config.aliases.clone();
        let failover_entries: Vec<FailoverEntry> = self
            .failover_keys
            .into_iter()
            .map(|fk| {
                let profile = crate::services::provider_profile::provider_profile_for_key(&fk);
                let is_copilot = profile.serve_flags.is_copilot;
                let protocol = profile.default_protocol;
                let ct = if is_copilot {
                    Some(Arc::new(CopilotTokenManager::new(
                        fk.key.as_str().to_string(),
                    )))
                } else {
                    None
                };
                FailoverEntry {
                    config: Arc::new(ServeRouterConfig {
                        upstream_base_url:
                            crate::services::provider_profile::resolve_starter_base_url(
                                &fk.base_url,
                            ),
                        upstream_api_key: fk.key.as_str().to_string(),
                        upstream_protocol: protocol,
                        is_copilot,
                        is_openrouter: profile.serve_flags.is_openrouter,
                        is_starter: profile.serve_flags.is_starter,
                        is_joycode: crate::services::joycode_auth::is_joycode_key(&fk.base_url),
                        cors: false,
                        timeout,
                        auth_token: None,
                        aliases: failover_aliases.clone(),
                    }),
                    key: fk,
                    copilot_tokens: ct,
                    route_cache: Arc::new(RouteCache::new(
                        "serve",
                        protocol,
                        std::collections::BTreeMap::new(),
                    )),
                }
            })
            .collect();

        let shutdown = Arc::new(tokio::sync::Notify::new());

        let state = Arc::new(ServeState {
            config: Arc::new(self.config),
            client: router_http_client_with_timeout(timeout),
            key: self.key,
            copilot_tokens,
            route_cache,
            log_store: self.log_store,
            logger: self.logger,
            failover_keys: Arc::new(failover_entries),
            shutdown: shutdown.clone(),
            usage_sink: self.usage_sink,
            usage_tool: self.usage_tool,
            run_tally: self.run_tally,
            quiet: self.quiet,
        });

        (tokio::spawn(run_accept_loop(listener, state)), shutdown)
    }
}

async fn run_accept_loop(listener: tokio::net::TcpListener, state: Arc<ServeState>) -> Result<()> {
    let semaphore = Arc::new(tokio::sync::Semaphore::new(100));
    let cors = state.config.cors;
    let cors_extra = if cors {
        http_utils::cors_header_block()
    } else {
        ""
    };
    // Accept both forms Claude Code sends: `Authorization: Bearer <token>`
    // (when ANTHROPIC_AUTH_TOKEN is set) and `x-api-key: <token>` (when
    // ANTHROPIC_API_KEY is set). The two Arcs share lifetime with the loop;
    // both are None when no auth_token is configured.
    let expected_bearer: Option<Arc<str>> = state
        .config
        .auth_token
        .as_ref()
        .map(|t| Arc::from(format!("Bearer {}", t)));
    let expected_token: Option<Arc<str>> = state
        .config
        .auth_token
        .as_ref()
        .map(|t| Arc::from(t.as_str()));

    loop {
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = state.shutdown.notified() => {
                // Wait for in-flight requests to finish (max 5s)
                let _ = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    semaphore.acquire_many(100),
                ).await;
                return Ok(());
            }
        };
        // Transient accept errors (ECONNABORTED on client reset, EMFILE under
        // fd pressure) must not kill the server; back off briefly and keep
        // accepting.
        let (mut socket, peer_addr) = match accept {
            Ok(pair) => pair,
            Err(e) => {
                if !state.quiet {
                    eprintln!("  \u{26a0} accept error (continuing): {e}");
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                continue;
            }
        };
        let peer_ip = peer_addr.ip().to_string();
        let state = state.clone();
        let permit = match semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => continue, // semaphore closed during shutdown
        };
        let expected_bearer = expected_bearer.clone();
        let expected_token = expected_token.clone();

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let _permit = permit;
            let read_result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                http_utils::read_full_request(&mut socket),
            )
            .await;

            let request_bytes = match read_result {
                Ok(Ok(b)) => b,
                Ok(Err(err)) => {
                    let response = http_utils::http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
                Err(_) => {
                    let _ = socket
                        .write_all(
                            http_utils::http_error_response(408, "Request read timed out")
                                .as_bytes(),
                        )
                        .await;
                    return;
                }
            };

            let request = String::from_utf8_lossy(&request_bytes).into_owned();

            // Handle OPTIONS preflight for CORS
            if cors && request.starts_with("OPTIONS ") {
                let head =
                    http_utils::http_response_head_with_extra(204, "text/plain", 0, cors_extra);
                let _ = socket.write_all(head.as_bytes()).await;
                return;
            }

            let path = http_utils::extract_request_path(&request);
            let path_no_query = path.split('?').next().unwrap_or(&path);

            // Auth check (skip /health). Claude Code sends `Authorization:
            // Bearer <token>` for ANTHROPIC_AUTH_TOKEN and `x-api-key: <token>`
            // for ANTHROPIC_API_KEY; accept either against the configured token.
            if let (Some(bearer), Some(token)) = (&expected_bearer, &expected_token)
                && path_no_query != "/health"
            {
                let headers_end = request.find("\r\n\r\n").unwrap_or(request.len());
                let head = &request[..headers_end];
                let auth_header = http_utils::header_value(head, "Authorization");
                let api_key_header = http_utils::header_value(head, "x-api-key");
                let bearer_match = auth_header == Some(&**bearer);
                let api_key_match = api_key_header == Some(&**token);
                if !bearer_match && !api_key_match {
                    let _ = socket
                        .write_all(
                            http_utils::http_error_response(
                                401,
                                "Invalid or missing auth token (expected Authorization: Bearer or x-api-key)",
                            )
                            .as_bytes(),
                        )
                        .await;
                    return;
                }
            }

            let request_start = std::time::Instant::now();

            // Extract model from request body for logging (best-effort)
            let log_model = http_utils::extract_request_body(&request)
                .ok()
                .and_then(|body| serde_json::from_str::<Value>(body).ok())
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from));

            let result = match path_no_query {
                "/health" => Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
                    HEALTH_RESPONSE.clone(),
                )),
                "/v1/models" | "/models" => handle_models(&state).await,
                "/v1/chat/completions" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_chat_with_failover(&request, &state).await
                    }
                }
                "/v1/responses" | "/responses" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_responses_with_failover(&request, &state).await
                    }
                }
                "/v1/embeddings" => {
                    if !request.starts_with("POST ") {
                        Ok(RouterResponse::buffered(
                            405,
                            CONTENT_TYPE_JSON,
                            br#"{"error":{"message":"Method not allowed"}}"#.to_vec(),
                        ))
                    } else {
                        handle_embeddings_with_failover(&request, &state).await
                    }
                }
                _ => Ok(RouterResponse::buffered(
                    404,
                    CONTENT_TYPE_JSON,
                    br#"{"error":{"message":"Not found"}}"#.to_vec(),
                )),
            };

            let response_status = match &result {
                Ok(RouterResponse::Buffered { status, .. }) => *status,
                Ok(RouterResponse::Streaming { status, .. }) => *status,
                Err(_) => 500,
            };

            let accounting = state.usage_sink.is_some();
            // Peek token usage off a buffered 2xx body before it's moved to the
            // socket; streaming bodies are sniffed as they're forwarded below.
            let buffered_usage = if accounting {
                match &result {
                    Ok(RouterResponse::Buffered { status, body, .. })
                        if (200..300).contains(status) =>
                    {
                        parse_token_usage(body)
                    }
                    _ => None,
                }
            } else {
                None
            };

            let stream_usage = match result {
                Ok(response) => {
                    write_router_response(&mut socket, response, cors_extra, accounting)
                        .await
                        .unwrap_or(None)
                }
                Err(e) => {
                    let _ = socket
                        .write_all(http_utils::http_error_response(500, &e.to_string()).as_bytes())
                        .await;
                    None
                }
            };
            let usage = buffered_usage.or(stream_usage);

            if let (Some(store), Some(u)) = (&state.usage_sink, &usage) {
                let _ = store
                    .record_tokens(
                        &state.key.id,
                        state.usage_tool.as_deref(),
                        log_model.as_deref(),
                        u.prompt,
                        u.completion,
                        u.cache_read,
                        u.cache_creation,
                    )
                    .await;
                // Same totals into the per-run tally, so the finished log row is
                // windowable by `aivo stats --since` (lifetime stats aren't).
                if let Some(tally) = &state.run_tally {
                    tally.add(u.prompt, u.completion, u.cache_read, u.cache_creation);
                }
            }

            // Log request (non-blocking, non-fatal)
            let latency_ms = request_start.elapsed().as_millis();
            let method = request
                .split_whitespace()
                .next()
                .unwrap_or("GET")
                .to_string();

            if let Some(ref logger) = state.logger {
                logger
                    .log(crate::services::request_log::RequestLogEntry {
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        method: method.clone(),
                        path: path_no_query.to_string(),
                        model: log_model.clone(),
                        status: response_status,
                        latency_ms: latency_ms as u64,
                        ip: peer_ip.clone(),
                    })
                    .await;
            }

            let _ = state
                .log_store
                .append(LogEvent {
                    source: "serve".to_string(),
                    kind: "serve_request".to_string(),
                    key_id: Some(state.key.id.clone()),
                    key_name: Some(state.key.display_name().to_string()),
                    base_url: Some(state.key.base_url.clone()),
                    tool: Some(
                        state
                            .usage_tool
                            .clone()
                            .unwrap_or_else(|| "serve".to_string()),
                    ),
                    model: log_model,
                    status_code: Some(response_status as i64),
                    duration_ms: Some(latency_ms as i64),
                    input_tokens: usage.as_ref().map(|u| u.prompt as i64),
                    output_tokens: usage.as_ref().map(|u| u.completion as i64),
                    cache_read_input_tokens: usage.as_ref().map(|u| u.cache_read as i64),
                    cache_creation_input_tokens: usage.as_ref().map(|u| u.cache_creation as i64),
                    title: Some(format!("{method} {path_no_query}")),
                    payload_json: Some(json!({
                        "method": method,
                        "path": path_no_query,
                        "ip": peer_ip,
                    })),
                    ..Default::default()
                })
                .await;
        });
    }
}

async fn handle_models(state: &ServeState) -> Result<RouterResponse> {
    // JoyCode has a custom models endpoint.
    if state.config.is_joycode {
        let ctx = upstream_context(state);
        return crate::services::joycode_router::send_joycode_models(&ctx).await;
    }
    let models = fetch_models(&state.client, &state.key).await?;
    // Local cache instance: lazy one-time disk read, and this endpoint
    // already pays a network fetch per call.
    let cache = crate::services::models_cache::ModelsCache::new();
    let cache_base = crate::commands::models::model_cache_key_for_key(&state.key);
    let mut entries = Vec::with_capacity(models.len() + state.config.aliases.len());
    for id in models {
        let limits =
            crate::services::model_metadata::resolve_limits(&cache, Some(&cache_base), &id).await;
        entries.push(model_list_response::ModelListEntry {
            id,
            owned_by: "aivo".to_string(),
            limits,
        });
    }
    let mut alias_names: Vec<&String> = state.config.aliases.keys().collect();
    alias_names.sort();
    for name in alias_names {
        // Aliases inherit the limits of the model they resolve to.
        let limits = match state.config.aliases.get(name) {
            Some(target) => {
                crate::services::model_metadata::resolve_limits(&cache, Some(&cache_base), target)
                    .await
            }
            None => Default::default(),
        };
        entries.push(model_list_response::ModelListEntry {
            id: name.clone(),
            owned_by: "aivo-alias".to_string(),
            limits,
        });
    }
    let resp = model_list_response::build_models_response_body(entries);
    Ok(RouterResponse::buffered(
        200,
        CONTENT_TYPE_JSON,
        resp.to_string().into_bytes(),
    ))
}

/// Token counts pulled from a response `usage` block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenUsage {
    pub(crate) prompt: u64,
    pub(crate) completion: u64,
    pub(crate) cache_read: u64,
    pub(crate) cache_creation: u64,
}

impl TokenUsage {
    fn is_zero(&self) -> bool {
        *self == TokenUsage::default()
    }

    /// Per-field max. Merges partial usage from successive stream events —
    /// Anthropic reports input in `message_start` and output in `message_delta`,
    /// and providers send cumulative counts, so the max is the final total.
    fn merge_max(&mut self, other: &TokenUsage) {
        self.prompt = self.prompt.max(other.prompt);
        self.completion = self.completion.max(other.completion);
        self.cache_read = self.cache_read.max(other.cache_read);
        self.cache_creation = self.cache_creation.max(other.cache_creation);
    }
}

/// Pull a `TokenUsage` out of any provider's response JSON object: OpenAI chat
/// (`usage` with `prompt_tokens`/`completion_tokens`), Responses (`usage` with
/// `input_tokens`/`output_tokens`, or nested under `response`), Anthropic
/// (`usage`, or nested under `message`), or Gemini (`usageMetadata`). Returns
/// `None` when there's no usage or it's all zero.
pub(crate) fn extract_usage_from_value(v: &Value) -> Option<TokenUsage> {
    if let Some(u) = v
        .get("usage")
        .or_else(|| v.get("message").and_then(|m| m.get("usage")))
        .or_else(|| v.get("response").and_then(|r| r.get("usage")))
    {
        let num = |k: &str| u.get(k).and_then(|x| x.as_u64());
        // details/hit-style cached counts are ⊂ the prompt figure; Anthropic-named
        // fields are disjoint from `input_tokens` and get added back.
        let details_cached = crate::services::openai_models::extract_cached_prompt_tokens(u)
            .or_else(|| {
                u.get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|x| x.as_u64())
            });
        let anthropic_read = num("cache_read_input_tokens");
        let cache_creation = num("cache_creation_input_tokens").unwrap_or(0);
        let prompt = match num("prompt_tokens") {
            Some(p) => p,
            None if details_cached.is_some() => num("input_tokens").unwrap_or(0),
            None => num("input_tokens").unwrap_or(0) + anthropic_read.unwrap_or(0) + cache_creation,
        };
        let usage = TokenUsage {
            prompt,
            completion: num("completion_tokens")
                .or_else(|| num("output_tokens"))
                .unwrap_or(0),
            cache_read: details_cached.or(anthropic_read).unwrap_or(0),
            cache_creation,
        };
        return (!usage.is_zero()).then_some(usage);
    }
    if let Some(um) = v.get("usageMetadata") {
        let n = |k: &str| um.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
        let usage = TokenUsage {
            prompt: n("promptTokenCount"),
            completion: n("candidatesTokenCount"),
            cache_read: n("cachedContentTokenCount"),
            cache_creation: 0,
        };
        return (!usage.is_zero()).then_some(usage);
    }
    None
}

/// Extract token usage from a buffered JSON response body.
pub(crate) fn parse_token_usage(body: &[u8]) -> Option<TokenUsage> {
    if let Ok(v) = serde_json::from_slice::<Value>(body) {
        return extract_usage_from_value(&v);
    }
    // Buffered SSE body — the Responses-via-chat path returns
    // text/event-stream even when buffered, so usage rides on `data:` lines
    // instead of a JSON envelope. Without this, those turns account zero.
    let mut sniffer = StreamUsageSniffer::new(true);
    sniffer.observe(body);
    sniffer.observe(b"\n");
    sniffer.finish()
}

/// Accumulates token usage from a forwarded SSE stream by scanning `data:` lines
/// for any provider's usage event. A no-op when `enabled` is false (native
/// launches don't account usage). `finish()` yields the merged per-field max.
pub(crate) struct StreamUsageSniffer {
    enabled: bool,
    pending: String,
    usage: TokenUsage,
    seen: bool,
}

impl StreamUsageSniffer {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pending: String::new(),
            usage: TokenUsage::default(),
            seen: false,
        }
    }

    /// Feed a raw upstream chunk (native provider SSE bytes).
    pub(crate) fn observe(&mut self, chunk: &[u8]) {
        if !self.enabled {
            return;
        }
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        // Parse complete lines; keep any trailing partial line buffered. Usage
        // only rides on `data:` lines, so skip everything else.
        while let Some(nl) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=nl).collect();
            let Some(json) = http_utils::sse_data_payload(line.trim()) else {
                continue;
            };
            if json.is_empty() || json == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(json)
                && let Some(u) = extract_usage_from_value(&v)
            {
                self.usage.merge_max(&u);
                self.seen = true;
            }
        }
        // Sniffing is best-effort: a pathological newline-less stream must not
        // grow this buffer without bound, so give up rather than hold it.
        if self.pending.len() > http_utils::MAX_SSE_PENDING_BYTES {
            self.pending = String::new();
            self.enabled = false;
        }
    }

    pub(crate) fn finish(self) -> Option<TokenUsage> {
        (self.enabled && self.seen).then_some(self.usage)
    }

    /// True when usage accounting is on — gates request-side `include_usage`
    /// injection so an OpenAI chat stream emits a usage chunk to sniff.
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
    }
}

fn parse_json_body(body_str: &str) -> std::result::Result<Value, RouterResponse> {
    serde_json::from_str(body_str).map_err(|e| {
        RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            json!({"error":{"message": format!("Invalid JSON: {}", e)}})
                .to_string()
                .into_bytes(),
        )
    })
}

/// Rewrites `body["model"]` against the alias snapshot, in place. No-op when
/// the field is missing, non-string, empty, or not an alias. Cycles are
/// detected by `resolve_alias_in_memory` and leave the original value.
fn apply_alias(body: &mut Value, aliases: &HashMap<String, String>) {
    if aliases.is_empty() {
        return;
    }
    let Some(model) = body.get("model").and_then(|v| v.as_str()) else {
        return;
    };
    if let Some(resolved) =
        crate::cli_args::resolve_alias_in_memory(aliases, Some(model.to_string()))
        && resolved != model
    {
        body["model"] = Value::String(resolved);
    }
}

async fn handle_chat(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    if !body.get("messages").is_some_and(|v| v.is_array()) {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: messages"}}"#.to_vec(),
        ));
    }

    apply_alias(&mut body, &state.config.aliases);
    handle_chat_body(body, state).await
}

async fn handle_responses(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    if body.get("input").is_none() {
        return Ok(RouterResponse::buffered(
            400,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Missing required field: input"}}"#.to_vec(),
        ));
    }
    apply_alias(&mut body, &state.config.aliases);
    let original_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4o")
        .to_string();
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Use `actual_model` to pin the model name to the raw user-supplied value.  The config's
    // `target_protocol` is snapshotted here, before `handle_chat_body` runs the fallback loop;
    // if the loop switches protocol, any protocol-based model-name transformation done by
    // `convert_responses_to_chat_request` would have used the wrong protocol.  Setting
    // `actual_model` causes `select_model_for_protocol` to return it verbatim, so the model
    // field in `chat_body` is always the original string and `handle_chat_body` transforms it
    // for the protocol that is actually selected.
    let mut config = responses_router_config(state, resolve_slot(&body, state).current().0);
    config.actual_model = Some(original_model.clone());
    let custom_tools = collect_custom_tool_names(&body);
    let mut chat_body = convert_responses_to_chat_request(&body, &config);
    chat_body["stream"] = json!(client_wants_stream);
    let chat_response = handle_chat_body(chat_body, state).await?;
    convert_chat_response_for_responses_route(
        chat_response,
        client_wants_stream,
        &original_model,
        custom_tools,
    )
}

/// Returns true if the status code should trigger failover.
/// - 401/403: auth failure (key revoked, expired, or lacks model access)
/// - 429: rate limited
/// - 5xx: server errors
fn is_failover_status(status: u16) -> bool {
    matches!(status, 401 | 403 | 429) || (500..600).contains(&status)
}

/// Builds a temporary ServeState from a FailoverEntry, sharing the client.
/// Logger is intentionally omitted — failover attempts are not individually logged.
fn failover_state(
    entry: &FailoverEntry,
    client: &reqwest::Client,
    log_store: &LogStore,
    quiet: bool,
) -> ServeState {
    ServeState {
        config: entry.config.clone(), // Arc clone — O(1) atomic increment
        client: client.clone(),
        key: entry.key.clone(),
        copilot_tokens: entry.copilot_tokens.clone(),
        route_cache: entry.route_cache.clone(),
        log_store: log_store.clone(),
        logger: None,
        failover_keys: Arc::new(Vec::new()),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        usage_sink: None,
        usage_tool: None,
        run_tally: None,
        quiet,
    }
}

/// Generates a failover wrapper around a handler function.
/// Tries the primary handler, then falls through to failover keys on 429/5xx
/// buffered responses. Streaming responses are never retried.
macro_rules! impl_with_failover {
    ($name:ident, $handler:ident) => {
        async fn $name(request: &str, state: &ServeState) -> Result<RouterResponse> {
            let response = $handler(request, state).await?;
            if state.failover_keys.is_empty() {
                return Ok(response);
            }

            let status = match &response {
                RouterResponse::Buffered { status, .. } => *status,
                RouterResponse::Streaming { .. } => return Ok(response),
            };

            if !is_failover_status(status) {
                return Ok(response);
            }

            if !state.quiet {
                eprintln!(
                    "  \u{21bb} Primary key returned {}; trying failover keys...",
                    status
                );
            }
            for entry in state.failover_keys.iter() {
                let fstate = failover_state(entry, &state.client, &state.log_store, state.quiet);
                if let Ok(resp) = $handler(request, &fstate).await {
                    let s = match &resp {
                        RouterResponse::Buffered { status, .. } => *status,
                        RouterResponse::Streaming { .. } => 200,
                    };
                    if !is_failover_status(s) {
                        if !state.quiet {
                            eprintln!(
                                "  \u{2713} Failover to {} succeeded",
                                entry.key.display_name()
                            );
                        }
                        return Ok(resp);
                    }
                }
            }
            Ok(response)
        }
    };
}

impl_with_failover!(handle_chat_with_failover, handle_chat);
impl_with_failover!(handle_responses_with_failover, handle_responses);
impl_with_failover!(handle_embeddings_with_failover, handle_embeddings);

async fn handle_embeddings(request: &str, state: &ServeState) -> Result<RouterResponse> {
    let body_str = http_utils::extract_request_body(request)?;
    let mut body = match parse_json_body(body_str) {
        Ok(v) => v,
        Err(r) => return Ok(r),
    };

    apply_alias(&mut body, &state.config.aliases);
    let protocol = resolve_slot(&body, state).current().0;
    if !matches!(
        protocol,
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
    ) {
        return Ok(RouterResponse::buffered(
            501,
            CONTENT_TYPE_JSON,
            br#"{"error":{"message":"Embeddings not supported with this provider"}}"#.to_vec(),
        ));
    }

    send_openai_embeddings(&body, &upstream_context(state)).await
}

/// Resolve the per-model route slot for a request body's `model` field.
fn resolve_slot(body: &Value, state: &ServeState) -> Arc<RouteSlot> {
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("");
    state.route_cache.resolve(model)
}

async fn handle_chat_body(body: Value, state: &ServeState) -> Result<RouterResponse> {
    let client_wants_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let slot = resolve_slot(&body, state);

    // Copilot has a fixed OpenAI protocol, but its reasoning models (gpt-5.x)
    // reject tools + reasoning_effort on /chat/completions and require /responses.
    // Pin that switch once it's observed so later turns skip the wasted 400.
    if state.config.is_copilot {
        let mut body = body;
        let (protocol, variant) = slot.current();
        let ctx = upstream_context(state);
        if protocol == ProviderProtocol::ResponsesApi {
            return send_copilot_responses(&body, client_wants_stream, &ctx).await;
        }
        let result = handle_chat_openai(&mut body, client_wants_stream, state).await?;
        let redirected = matches!(&result, RouterResponse::Buffered { status: 400, body: e, .. }
            if copilot_requires_responses_api(e));
        if redirected {
            commit_protocol_switch(
                slot.route_atom(),
                ProviderProtocol::ResponsesApi,
                variant,
                1,
            );
            return send_copilot_responses(&body, client_wants_stream, &ctx).await;
        }
        return Ok(result);
    }

    // JoyCode uses a fixed OpenAI-compatible protocol with custom headers/signing.
    if state.config.is_joycode {
        let mut body = body;
        let ctx = upstream_context(state);
        return crate::services::joycode_router::send_joycode_chat(
            &mut body, client_wants_stream, &ctx,
        )
        .await;
    }

    // Skip fallback for openrouter — fixed protocol.
    if state.config.is_openrouter {
        let mut body = body;
        return match slot.current().0 {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body, client_wants_stream, state).await
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body, client_wants_stream, state).await
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body, client_wants_stream, state).await
            }
        };
    }

    // Protocol-only candidates: serve's upstream senders own their URLs, so
    // path variants don't apply, and ResponsesApi shares the OpenAI chat
    // handler — keep only the first candidate per handler family.
    let mut seen_openai_family = false;
    let candidates: Vec<ProviderProtocol> = protocol_candidates(slot.route_atom())
        .into_iter()
        .filter(|(_, variant)| *variant == PathVariant::Default)
        .map(|(protocol, _)| protocol)
        .filter(|protocol| {
            let openai_family = matches!(
                protocol,
                ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
            );
            !openai_family || !std::mem::replace(&mut seen_openai_family, true)
        })
        .collect();

    let mut first_error: FirstError<RouterResponse> = FirstError::new();
    let mut success: Option<RouterResponse> = None;
    for (attempt, protocol) in candidates.into_iter().enumerate() {
        let mut body_clone = body.clone();
        let response = match protocol {
            ProviderProtocol::Anthropic => {
                handle_chat_anthropic(&body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Google => {
                handle_chat_gemini(&mut body_clone, client_wants_stream, state).await?
            }
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                handle_chat_openai(&mut body_clone, client_wants_stream, state).await?
            }
        };

        let status = match &response {
            RouterResponse::Buffered { status, .. } => *status,
            // Streaming is only produced when the upstream returned 200 (see each handle_chat_* handler);
            // a protocol mismatch (404/405/415) always results in a Buffered error response.
            RouterResponse::Streaming { .. } => 200,
        };

        if !is_protocol_mismatch(status) {
            commit_protocol_switch(slot.route_atom(), protocol, PathVariant::Default, attempt);
            slot.confirm();
            if attempt > 0 && !state.quiet {
                eprintln!("  \u{2022} Protocol auto-switched to {}", protocol.as_str());
            }
            success = Some(response);
            break;
        }

        let body_text = match &response {
            RouterResponse::Buffered { body, .. } => String::from_utf8_lossy(body).into_owned(),
            RouterResponse::Streaming { .. } => String::new(),
        };
        let classification = classify_failed_attempt(status, &body_text);
        first_error.record_with(&classification, || response);
        match mismatch_directive(
            attempt,
            &classification,
            &slot,
            protocol,
            PathVariant::Default,
            None,
        ) {
            MismatchDirective::Bail => break,
            MismatchDirective::RetrySameCandidate | MismatchDirective::NextCandidate => {}
        }
    }

    let (seed_protocol, seed_variant) = slot.seed_route();
    record_request_outcome(
        slot.route_atom(),
        slot.failures_atom(),
        seed_protocol,
        seed_variant,
        success.is_some(),
    );
    if let Some(response) = success {
        return Ok(response);
    }
    Ok(first_error.take().unwrap_or(RouterResponse::buffered(
        503,
        CONTENT_TYPE_JSON,
        br#"{"error":{"message":"No compatible protocol found"}}"#.to_vec(),
    )))
}

fn responses_router_config(
    state: &ServeState,
    target_protocol: ProviderProtocol,
) -> ResponsesToChatRouterConfig {
    ResponsesToChatRouterConfig {
        target_base_url: state.config.upstream_base_url.clone(),
        api_key: state.config.upstream_api_key.clone(),
        target_protocol,
        target_path_variant: None,
        copilot_token_manager: state.copilot_tokens.clone(),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: state.config.is_starter,
        aivo_prefix_models: Vec::new(),
    }
}

fn upstream_context(state: &ServeState) -> UpstreamRequestContext {
    UpstreamRequestContext {
        client: state.client.clone(),
        upstream_base_url: state.config.upstream_base_url.clone(),
        upstream_api_key: state.config.upstream_api_key.clone(),
        is_copilot: state.config.is_copilot,
        is_openrouter: state.config.is_openrouter,
        is_starter: state.config.is_starter,
        is_joycode: state.config.is_joycode,
        copilot_tokens: state.copilot_tokens.clone(),
        accounting: state.usage_sink.is_some(),
    }
}

async fn handle_chat_anthropic(
    body: &Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_anthropic_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_gemini(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_gemini_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn handle_chat_openai(
    body: &mut Value,
    client_wants_stream: bool,
    state: &ServeState,
) -> Result<RouterResponse> {
    send_openai_chat(body, client_wants_stream, &upstream_context(state)).await
}

async fn write_router_response(
    socket: &mut tokio::net::TcpStream,
    response: RouterResponse,
    extra_headers: &str,
    sniff_usage: bool,
) -> Result<Option<TokenUsage>> {
    use tokio::io::AsyncWriteExt;

    let mut sniffer = StreamUsageSniffer::new(sniff_usage);
    match response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_response_head_with_extra(
                status,
                &content_type,
                body.len(),
                extra_headers,
            );
            socket.write_all(headers.as_bytes()).await?;
            socket.write_all(&body).await?;
        }
        RouterResponse::Streaming {
            status,
            content_type,
            body,
        } => {
            let headers = http_utils::http_chunked_response_head_with_extra(
                status,
                &content_type,
                extra_headers,
            );
            socket.write_all(headers.as_bytes()).await?;

            match *body {
                StreamingBody::Upstream(mut upstream) => {
                    while let Some(chunk) = upstream.chunk().await? {
                        sniffer.observe(&chunk);
                        write_chunk(socket, &chunk).await?;
                    }
                }
                StreamingBody::Anthropic {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        sniffer.observe(&chunk);
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
                StreamingBody::Gemini {
                    mut upstream,
                    mut converter,
                } => {
                    while let Some(chunk) = upstream.chunk().await? {
                        sniffer.observe(&chunk);
                        let mapped = converter.push_bytes(&chunk)?;
                        if !mapped.is_empty() {
                            write_chunk(socket, mapped.as_bytes()).await?;
                        }
                    }
                    let tail = converter.finish()?;
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
                StreamingBody::Responses {
                    source,
                    mut converter,
                } => {
                    match *source {
                        StreamingBody::Upstream(mut upstream) => {
                            while let Some(chunk) = upstream.chunk().await? {
                                sniffer.observe(&chunk);
                                let mapped = converter.push_bytes(&chunk);
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Anthropic {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                sniffer.observe(&chunk);
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes());
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes());
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Gemini {
                            mut upstream,
                            converter: mut openai_converter,
                        } => {
                            while let Some(chunk) = upstream.chunk().await? {
                                sniffer.observe(&chunk);
                                let openai = openai_converter.push_bytes(&chunk)?;
                                if !openai.is_empty() {
                                    let mapped = converter.push_bytes(openai.as_bytes());
                                    if !mapped.is_empty() {
                                        write_chunk(socket, mapped.as_bytes()).await?;
                                    }
                                }
                            }
                            let openai_tail = openai_converter.finish()?;
                            if !openai_tail.is_empty() {
                                let mapped = converter.push_bytes(openai_tail.as_bytes());
                                if !mapped.is_empty() {
                                    write_chunk(socket, mapped.as_bytes()).await?;
                                }
                            }
                        }
                        StreamingBody::Responses { .. } => {
                            anyhow::bail!("nested responses stream sources are not supported");
                        }
                    }

                    let tail = converter.finish();
                    if !tail.is_empty() {
                        write_chunk(socket, tail.as_bytes()).await?;
                    }
                }
            }

            socket.write_all(b"0\r\n\r\n").await?;
        }
    }

    Ok(sniffer.finish())
}

async fn write_chunk(socket: &mut tokio::net::TcpStream, chunk: &[u8]) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    let formatted = http_utils::format_http_chunk(chunk);
    if formatted.is_empty() {
        return Ok(());
    }
    socket.write_all(&formatted).await?;
    Ok(())
}

fn convert_chat_response_for_responses_route(
    chat_response: RouterResponse,
    client_wants_stream: bool,
    original_model: &str,
    custom_tools: HashSet<String>,
) -> Result<RouterResponse> {
    match chat_response {
        RouterResponse::Buffered {
            status,
            content_type,
            body,
        } => {
            if status >= 400 {
                return Ok(RouterResponse::buffered(status, &content_type, body));
            }

            if client_wants_stream {
                let sse = if content_type.contains("text/event-stream") {
                    convert_chat_sse_to_responses_sse(
                        std::str::from_utf8(&body)?,
                        original_model,
                        &custom_tools,
                    )
                } else {
                    let chat_json: Value = serde_json::from_slice(&body)?;
                    convert_chat_response_to_responses_sse(
                        &chat_json,
                        false,
                        original_model,
                        &custom_tools,
                    )
                };
                Ok(RouterResponse::buffered(
                    200,
                    "text/event-stream",
                    sse.into_bytes(),
                ))
            } else {
                let chat_json: Value = serde_json::from_slice(&body)?;
                let response_json = convert_chat_response_to_responses_json(
                    &chat_json,
                    original_model,
                    &custom_tools,
                )?;
                Ok(RouterResponse::buffered(
                    200,
                    CONTENT_TYPE_JSON,
                    serde_json::to_vec(&response_json)?,
                ))
            }
        }
        RouterResponse::Streaming {
            status,
            content_type: _,
            body,
        } => {
            if !client_wants_stream {
                anyhow::bail!(
                    "internal error: responses route received streaming body for non-streaming request"
                );
            }

            Ok(RouterResponse::Streaming {
                status,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Responses {
                    source: body,
                    converter: ResponsesStreamConverter::new(original_model, false)
                        .with_custom_tools(custom_tools),
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ApiKey;
    use http::Response as HttpResponse;
    use serde_json::json;

    #[test]
    fn usage_sniffer_disables_on_oversized_partial_line() {
        let mut sniffer = StreamUsageSniffer::new(true);
        // A newline-less stream larger than the cap must disable sniffing
        // instead of buffering forever.
        let big = vec![b'x'; http_utils::MAX_SSE_PENDING_BYTES + 1];
        sniffer.observe(&big);
        assert!(!sniffer.is_enabled());
        sniffer.observe(b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n");
        assert!(sniffer.finish().is_none());
    }

    fn test_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://example.com/v1".to_string(),
            None,
            "secret".to_string(),
        )
    }

    fn test_state(protocol: ProviderProtocol) -> ServeState {
        ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://example.com/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: protocol,
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
                is_joycode: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            client: http_utils::router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                protocol,
                std::collections::BTreeMap::new(),
            )),
            log_store: LogStore::new(std::env::temp_dir()),
            logger: None,
            failover_keys: Arc::new(Vec::new()),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
        }
    }

    fn mock_reqwest_response(
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
    fn convert_chat_response_for_responses_route_maps_buffered_json() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from router"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, serde_json::to_vec(&chat).unwrap()),
            false,
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let json: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, CONTENT_TYPE_JSON);
                assert_eq!(json["object"], "response");
                assert_eq!(json["output"][0]["content"][0]["text"], "Hello from router");
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered response"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_maps_streaming_sse() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, "text/event-stream", chat_sse.as_bytes().to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                let sse = String::from_utf8(body).unwrap();
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                assert!(sse.contains("event: response.created"));
                assert!(sse.contains("\"delta\":\"Hel\""));
                assert!(sse.contains("event: response.completed"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_rejects_streaming_non_stream_requests() {
        let response = convert_chat_response_for_responses_route(
            RouterResponse::Streaming {
                status: 200,
                content_type: "text/event-stream".to_string(),
                body: Box::new(StreamingBody::Upstream(mock_reqwest_response(
                    200,
                    "text/event-stream",
                    "data: [DONE]\n\n",
                ))),
            },
            false,
            "gpt-4o",
            HashSet::new(),
        );

        assert!(response.is_err());
    }

    #[test]
    fn responses_router_config_uses_slot_protocol() {
        let state = test_state(ProviderProtocol::Google);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);

        assert_eq!(config.target_protocol, ProviderProtocol::Google);
        assert_eq!(config.target_base_url, "https://example.com/v1");
        assert_eq!(config.api_key, "secret");
    }

    #[test]
    fn upstream_context_copies_router_flags() {
        let state = ServeState {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://openrouter.ai/api/v1".to_string(),
                upstream_api_key: "secret".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_openrouter: true,
                is_starter: false,
                is_joycode: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            client: http_utils::router_http_client(),
            key: test_key(),
            copilot_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
            log_store: LogStore::new(std::env::temp_dir()),
            logger: None,
            failover_keys: Arc::new(Vec::new()),
            shutdown: Arc::new(tokio::sync::Notify::new()),
            usage_sink: None,
            usage_tool: None,
            run_tally: None,
            quiet: false,
        };

        let context = upstream_context(&state);
        assert!(context.is_openrouter);
        assert!(!context.is_copilot);
        assert_eq!(context.upstream_base_url, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn format_http_chunk_adds_hex_prefix_and_trailer() {
        assert_eq!(http_utils::format_http_chunk(b"hello"), b"5\r\nhello\r\n");
        assert!(http_utils::format_http_chunk(b"").is_empty());
    }

    #[test]
    fn convert_chat_response_for_responses_route_passes_error_status_through() {
        let error_body = br#"{"error":{"message":"rate limited"}}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(429, CONTENT_TYPE_JSON, error_body.to_vec()),
            false,
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, body, .. } => {
                assert_eq!(status, 429);
                assert_eq!(body, error_body);
            }
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn convert_chat_response_for_responses_route_passes_500_through() {
        let error_body = br#"{"error":"internal"}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(500, CONTENT_TYPE_JSON, error_body.to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, .. } => assert_eq!(status, 500),
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn format_http_chunk_large_payload() {
        let data = vec![b'x'; 256];
        let chunk = http_utils::format_http_chunk(&data);
        // 256 = 0x100
        assert!(chunk.starts_with(b"100\r\n"));
        assert!(chunk.ends_with(b"\r\n"));
    }

    #[test]
    fn format_http_chunk_single_byte() {
        let chunk = http_utils::format_http_chunk(b"a");
        assert_eq!(chunk, b"1\r\na\r\n");
    }

    #[test]
    fn responses_router_config_anthropic_protocol() {
        let state = test_state(ProviderProtocol::Anthropic);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);
        assert_eq!(config.target_protocol, ProviderProtocol::Anthropic);
    }

    #[test]
    fn convert_chat_response_for_responses_route_buffered_json_to_stream() {
        let chat = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "streamed text"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}
        });

        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, serde_json::to_vec(&chat).unwrap()),
            true, // client wants stream
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered {
                status,
                content_type,
                body,
            } => {
                assert_eq!(status, 200);
                assert_eq!(content_type, "text/event-stream");
                let sse = String::from_utf8(body).unwrap();
                assert!(sse.contains("event: response.created"));
                assert!(sse.contains("streamed text"));
                assert!(sse.contains("event: response.completed"));
            }
            RouterResponse::Streaming { .. } => panic!("expected buffered SSE"),
        }
    }

    // ── Failover tests ────────────────────────────────────────────────────

    #[test]
    fn is_failover_status_triggers_on_auth_errors() {
        assert!(is_failover_status(401));
        assert!(is_failover_status(403));
    }

    #[test]
    fn is_failover_status_triggers_on_rate_limit() {
        assert!(is_failover_status(429));
    }

    #[test]
    fn is_failover_status_triggers_on_server_errors() {
        assert!(is_failover_status(500));
        assert!(is_failover_status(502));
        assert!(is_failover_status(503));
        assert!(is_failover_status(504));
        assert!(is_failover_status(599));
    }

    #[test]
    fn is_failover_status_does_not_trigger_on_success() {
        assert!(!is_failover_status(200));
        assert!(!is_failover_status(201));
        assert!(!is_failover_status(204));
    }

    #[test]
    fn is_failover_status_does_not_trigger_on_client_errors() {
        // Client errors that indicate a bad request — retrying with a different
        // key won't help.
        assert!(!is_failover_status(400));
        assert!(!is_failover_status(404));
        assert!(!is_failover_status(405));
        assert!(!is_failover_status(422));
    }

    #[test]
    fn failover_state_builds_from_entry() {
        let entry = FailoverEntry {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://backup.example.com/v1".to_string(),
                upstream_api_key: "backup-key".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
                is_joycode: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            key: test_key(),
            copilot_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        assert_eq!(
            state.config.upstream_base_url,
            "https://backup.example.com/v1"
        );
        assert_eq!(state.config.upstream_api_key, "backup-key");
        assert!(state.logger.is_none());
        assert!(state.failover_keys.is_empty());
    }

    #[test]
    fn failover_state_shares_arc_config() {
        let config = Arc::new(ServeRouterConfig {
            upstream_base_url: "https://backup.example.com/v1".to_string(),
            upstream_api_key: "key".to_string(),
            upstream_protocol: ProviderProtocol::Openai,
            is_copilot: false,
            is_openrouter: false,
            is_starter: false,
                is_joycode: false,
            cors: false,
            timeout: 300,
            auth_token: None,
            aliases: HashMap::new(),
        });

        let entry = FailoverEntry {
            config: config.clone(),
            key: test_key(),
            copilot_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        // Arc should be a clone of the same allocation, not a new copy
        assert!(Arc::ptr_eq(&entry.config, &state.config));
    }

    #[test]
    fn failover_state_does_not_cascade() {
        let entry = FailoverEntry {
            config: Arc::new(ServeRouterConfig {
                upstream_base_url: "https://backup.example.com/v1".to_string(),
                upstream_api_key: "key".to_string(),
                upstream_protocol: ProviderProtocol::Openai,
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
                is_joycode: false,
                cors: false,
                timeout: 300,
                auth_token: None,
                aliases: HashMap::new(),
            }),
            key: test_key(),
            copilot_tokens: None,
            route_cache: Arc::new(RouteCache::new(
                "serve",
                ProviderProtocol::Openai,
                std::collections::BTreeMap::new(),
            )),
        };

        let client = http_utils::router_http_client();
        let state = failover_state(&entry, &client, &LogStore::new(std::env::temp_dir()), false);

        // Failover state should have no failover keys (no cascading)
        assert!(state.failover_keys.is_empty());
        // Logger should be disabled on failover attempts
        assert!(state.logger.is_none());
    }

    #[test]
    fn health_response_is_valid_json() {
        let json: Value = serde_json::from_slice(&HEALTH_RESPONSE).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["version"].is_string());
    }

    #[test]
    fn health_response_is_stable() {
        // LazyLock should return the same bytes every time
        let a = HEALTH_RESPONSE.clone();
        let b = HEALTH_RESPONSE.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn is_failover_status_boundary_499() {
        // 499 is a non-standard client error — should NOT trigger failover
        assert!(!is_failover_status(499));
    }

    #[test]
    fn is_failover_status_boundary_600() {
        // 600 is outside the 5xx range — should NOT trigger failover
        assert!(!is_failover_status(600));
    }

    #[test]
    fn convert_chat_response_for_responses_route_malformed_json_body() {
        // A non-JSON body with status 200 should fail to parse and return an error
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, b"not valid json".to_vec()),
            false,
            "gpt-4o",
            HashSet::new(),
        );
        assert!(response.is_err());
    }

    #[test]
    fn convert_chat_response_for_responses_route_empty_body_non_stream() {
        // An empty body with status 200 should fail to parse and return an error
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(200, CONTENT_TYPE_JSON, Vec::new()),
            false,
            "gpt-4o",
            HashSet::new(),
        );
        assert!(response.is_err());
    }

    #[test]
    fn convert_chat_response_for_responses_route_error_stream_passthrough() {
        // A 400 error response passes through unchanged even when client wants stream
        let error_body = br#"{"error":{"message":"bad request"}}"#;
        let response = convert_chat_response_for_responses_route(
            RouterResponse::buffered(400, CONTENT_TYPE_JSON, error_body.to_vec()),
            true,
            "gpt-4o",
            HashSet::new(),
        )
        .unwrap();

        match response {
            RouterResponse::Buffered { status, body, .. } => {
                assert_eq!(status, 400);
                assert_eq!(body, error_body);
            }
            _ => panic!("expected buffered error passthrough"),
        }
    }

    #[test]
    fn responses_router_config_openai_protocol() {
        let state = test_state(ProviderProtocol::Openai);
        let slot = state.route_cache.resolve("some-model");
        let config = responses_router_config(&state, slot.current().0);

        assert_eq!(config.target_protocol, ProviderProtocol::Openai);
        assert_eq!(config.target_base_url, "https://example.com/v1");
        assert_eq!(config.api_key, "secret");
        assert!(config.copilot_token_manager.is_none());
        assert!(config.model_prefix.is_none());
        assert!(!config.requires_reasoning_content);
        assert!(config.actual_model.is_none());
        assert!(config.max_tokens_cap.is_none());
        assert!(config.responses_api_supported.is_none());
    }

    #[test]
    fn apply_alias_rewrites_known_alias() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);
        let mut body = json!({"model": "fast", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "gpt-4o-mini");
    }

    #[test]
    fn apply_alias_passes_through_unknown_model() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);
        let mut body = json!({"model": "claude-sonnet-4-6", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "claude-sonnet-4-6");
    }

    #[test]
    fn apply_alias_follows_chain() {
        let aliases = HashMap::from([
            ("fast".to_string(), "quick".to_string()),
            ("quick".to_string(), "gpt-4o-mini".to_string()),
        ]);
        let mut body = json!({"model": "fast"});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "gpt-4o-mini");
    }

    #[test]
    fn apply_alias_no_op_when_field_missing_or_empty() {
        let aliases = HashMap::from([("fast".to_string(), "gpt-4o-mini".to_string())]);

        let mut body = json!({"messages": []});
        apply_alias(&mut body, &aliases);
        assert!(body.get("model").is_none());

        let mut body = json!({"model": "", "messages": []});
        apply_alias(&mut body, &aliases);
        assert_eq!(body["model"], "");
    }

    #[test]
    fn apply_alias_no_op_when_alias_map_empty() {
        let mut body = json!({"model": "fast"});
        apply_alias(&mut body, &HashMap::new());
        assert_eq!(body["model"], "fast");
    }

    #[test]
    fn parse_token_usage_openai_shape() {
        let body = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 8}
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt, u.completion, u.cache_read), (30, 12, 8));
    }

    #[test]
    fn parse_token_usage_buffered_sse_body() {
        // Responses-via-chat returns text/event-stream even on the buffered
        // path; usage rides on a data: line, not a JSON envelope.
        let body = "event: response.completed\n\
                    data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":21,\"output_tokens\":7}}}\n\n\
                    data: [DONE]\n";
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt, u.completion), (21, 7));
    }

    #[test]
    fn parse_token_usage_responses_shape() {
        let body = json!({
            "object": "response",
            "usage": {"input_tokens": 100, "output_tokens": 40}
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt, u.completion), (100, 40));
    }

    #[test]
    fn parse_token_usage_anthropic_shape_folds_cache_into_prompt() {
        let body = json!({
            "usage": {
                "input_tokens": 61, "output_tokens": 32,
                "cache_read_input_tokens": 5000, "cache_creation_input_tokens": 120
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!(
            (u.prompt, u.completion, u.cache_read, u.cache_creation),
            (5181, 32, 5000, 120)
        );
    }

    #[test]
    fn parse_token_usage_deepseek_hit_tokens_as_cache_read() {
        // DeepSeek without the OpenAI-style details block: hit tokens ⊂ prompt.
        let body = json!({
            "usage": {
                "prompt_tokens": 5000, "completion_tokens": 100,
                "prompt_cache_hit_tokens": 4800, "prompt_cache_miss_tokens": 200
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt, u.cache_read), (5000, 4800));
    }

    #[test]
    fn parse_token_usage_responses_cached_subset_not_double_added() {
        let body = json!({
            "object": "response",
            "usage": {
                "input_tokens": 1000, "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": 800}
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt, u.cache_read), (1000, 800));
    }

    #[test]
    fn parse_token_usage_none_when_absent_or_zero() {
        assert!(parse_token_usage(br#"{"choices":[]}"#).is_none());
        assert!(parse_token_usage(b"not json").is_none());
        let zero = json!({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}).to_string();
        assert!(parse_token_usage(zero.as_bytes()).is_none());
    }

    #[test]
    fn sniffer_disabled_is_noop() {
        let mut s = StreamUsageSniffer::new(false);
        s.observe(b"data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n");
        assert!(s.finish().is_none());
    }

    #[test]
    fn sniffer_openai_chat_final_usage_chunk() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        s.observe(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":8}}}\n\n");
        s.observe(b"data: [DONE]\n\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt, u.completion, u.cache_read), (30, 12, 8));
    }

    #[test]
    fn sniffer_anthropic_merges_start_and_delta() {
        // Anthropic splits input (message_start) and output (message_delta); its
        // disjoint cache counts fold into the inclusive prompt (100+20+5).
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":20,\"cache_creation_input_tokens\":5,\"output_tokens\":1}}}\n\n");
        s.observe(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!(
            (u.prompt, u.completion, u.cache_read, u.cache_creation),
            (125, 42, 20, 5)
        );
    }

    #[test]
    fn sniffer_responses_completed_event() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":80,\"output_tokens\":25,\"input_tokens_details\":{\"cached_tokens\":10}}}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt, u.completion, u.cache_read), (80, 25, 10));
    }

    #[test]
    fn sniffer_gemini_usage_metadata() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"usageMetadata\":{\"promptTokenCount\":70,\"candidatesTokenCount\":18,\"cachedContentTokenCount\":12}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt, u.completion, u.cache_read), (70, 18, 12));
    }

    #[test]
    fn sniffer_reassembles_usage_line_split_across_chunks() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"usage\":{\"prompt_tokens\":11,");
        s.observe(b"\"completion_tokens\":7}}\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt, u.completion), (11, 7));
    }
}
