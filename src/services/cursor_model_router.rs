//! Local HTTP→ACP compatibility router for Cursor-backed tools.
//!
//! Spawned by `launch_runtime` when the active key is a cursor sentinel.
//! Translates Anthropic/OpenAI/Responses-shaped HTTP requests into
//! `cursor-agent acp` session prompts so Claude/Codex/Pi/OpenCode can target
//! Cursor models without speaking ACP directly. One ACP session is reused
//! across the launched tool's requests (per-router) and serialized behind a
//! Mutex because cursor-agent doesn't accept concurrent prompts on a session.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CURSOR_ACP_SENTINEL, CursorAcpSession};
use crate::services::http_debug::{self, DebugEntry, Phase, redact_headers};
use crate::services::http_utils::{
    self, bind_local_listener, cors_header_block, extract_request_body, extract_request_path,
    format_http_chunk, http_chunked_response_head_with_extra, http_response_head_with_extra,
};
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::ApiKey;

/// Configuration handed to the router at spawn time.
#[derive(Clone)]
pub struct CursorRouterConfig {
    pub key: ApiKey,
    pub workspace_cwd: String,
    /// Disk-backed model list cache shared with `aivo models` / the picker.
    /// When `Some`, `/v1/models` checks the cache before spawning
    /// `cursor-agent models` — saving ~1-3 s on every launch where a fresh
    /// entry exists from a prior `aivo models cursor` or picker invocation.
    pub models_cache: Option<ModelsCache>,
    /// Number of ACP sessions to pre-open at router startup. Capped at
    /// [`MAX_POOL_SESSIONS`]. Pick `2` for tools that fan out paired requests
    /// (e.g. Claude Code's main + subagent burst), `1` for sequential tools
    /// (pi, codex), `0` in tests that don't want to spawn `cursor-agent`.
    pub prewarm_count: usize,
}

pub struct CursorModelRouter {
    config: CursorRouterConfig,
}

/// Soft cap on concurrent ACP sessions; cursor-agent serializes prompts
/// per session, so paired turns (main + subagent) need separate slots.
const MAX_POOL_SESSIONS: usize = 3;

/// Default prewarm when the caller doesn't override. Two sessions covers
/// Claude Code's paired main+subagent burst; tools without that pattern
/// override this via `CursorRouterConfig::prewarm_count` to avoid spawning
/// a second cursor-agent that the tool would never use.
pub const DEFAULT_POOL_PREWARM: usize = 2;

/// One lazily-opened ACP session. Held inside `Arc<Mutex<...>>` so HTTP
/// handlers can take an owned lock for the full turn duration; the outer pool
/// just tracks the slot Arcs.
type SessionSlot = Arc<Mutex<Option<CursorAcpSession>>>;

struct RouterState {
    config: CursorRouterConfig,
    cached_models: Mutex<Option<Vec<String>>>,
    /// Pool of ACP session slots, grown lazily up to [`MAX_POOL_SESSIONS`].
    /// Each slot holds at most one session; concurrent requests fan out into
    /// idle slots and only block when every slot is busy.
    pool: Mutex<Vec<SessionSlot>>,
    /// Count of prewarm tasks still opening sessions. `acquire_session_slot`
    /// consults this to wait on an in-flight prewarmed slot rather than
    /// expanding the pool past `prewarm_count` and orphaning a process. The
    /// counter is incremented synchronously in `spawn_prewarm` (before any
    /// task is spawned) so an early request arriving between
    /// `start_background` returning and the prewarm task starting still
    /// sees `prewarming > 0`.
    prewarming: AtomicUsize,
}

impl CursorModelRouter {
    pub fn new(config: CursorRouterConfig) -> Self {
        Self { config }
    }

    /// Bind a random localhost port and start the router in the background.
    /// Also kicks off an ACP session pre-warm so the launched tool's first
    /// HTTP request doesn't pay Node.js startup + ACP
    /// initialize/authenticate/session/new serially before the first prompt
    /// can stream.
    pub async fn start_background(&self) -> Result<(u16, JoinHandle<Result<()>>)> {
        let (listener, port) = bind_local_listener().await?;
        let prewarm = self.config.prewarm_count;
        let state = Arc::new(RouterState {
            config: self.config.clone(),
            cached_models: Mutex::new(None),
            pool: Mutex::new(Vec::new()),
            prewarming: AtomicUsize::new(0),
        });
        spawn_prewarm(state.clone(), prewarm);
        let handle = tokio::spawn(run_router(listener, state));
        Ok((port, handle))
    }
}

/// Reserve a session slot for the current HTTP turn. Prefers an idle slot;
/// when prewarm is still in flight, blocks on one of the existing slots so we
/// don't expand past `prewarm_count` and orphan a cursor-agent process;
/// otherwise expands the pool up to [`MAX_POOL_SESSIONS`]; finally falls
/// back to waiting on the first slot when the cap is hit. The returned guard
/// pins the slot for the caller's exclusive use until dropped.
async fn acquire_session_slot(
    state: &RouterState,
) -> tokio::sync::OwnedMutexGuard<Option<CursorAcpSession>> {
    let existing: Vec<SessionSlot> = state.pool.lock().await.clone();
    for slot in &existing {
        if let Ok(guard) = slot.clone().try_lock_owned() {
            return guard;
        }
    }
    // Every existing slot is busy. If a prewarm task is still opening one of
    // them, wait for it to finish instead of expanding the pool — otherwise
    // we'd spawn a second cursor-agent that the prewarmed one would
    // duplicate. Race the lock_owned() futures across all existing slots so
    // whichever frees first (prewarm completion OR request return) wins;
    // pinning to slot 0 would deadlock a second concurrent acquirer when
    // slot 1 finishes first.
    if state.prewarming.load(Ordering::SeqCst) > 0 && !existing.is_empty() {
        return wait_for_any_slot(existing).await;
    }
    {
        let mut pool = state.pool.lock().await;
        // Re-sweep under the pool mutex before expanding: a prewarm or
        // another acquirer may have released between the initial try_lock
        // pass above and now. Without this, prewarming dropping to 0
        // milliseconds before we read it would still let us expand past a
        // freshly-freed slot and orphan a cursor-agent.
        for slot in pool.iter() {
            if let Ok(guard) = slot.clone().try_lock_owned() {
                return guard;
            }
        }
        if pool.len() < MAX_POOL_SESSIONS {
            let slot: SessionSlot = Arc::new(Mutex::new(None));
            pool.push(slot.clone());
            // The new slot has no other waiters, so this resolves immediately.
            return slot.lock_owned().await;
        }
    }
    // Cap reached — block on the first slot. Round-robin between slots would
    // be marginally fairer but the launched tool rarely keeps all three
    // pegged, so a deterministic pick keeps the code simple.
    existing[0].clone().lock_owned().await
}

/// Race `lock_owned()` across every slot, returning whichever lock resolves
/// first. Pending futures are dropped (and their internal waiters cancelled)
/// when the winner returns, so the loser slots remain available for the next
/// acquirer.
async fn wait_for_any_slot(
    slots: Vec<SessionSlot>,
) -> tokio::sync::OwnedMutexGuard<Option<CursorAcpSession>> {
    use futures::future::FutureExt;
    let mut futures: Vec<_> = slots.into_iter().map(|s| s.lock_owned().boxed()).collect();
    let (guard, _, _rest) = futures::future::select_all(futures.drain(..)).await;
    guard
}

/// Ensure the slot holds an open session, opening one on demand. Equivalent
/// to the previous lazy-open pattern but factored out so every handler reuses
/// the same retry-and-context wrapping.
async fn ensure_session_open(
    guard: &mut tokio::sync::OwnedMutexGuard<Option<CursorAcpSession>>,
    state: &RouterState,
    requested_model: Option<&str>,
) -> Result<()> {
    if guard.is_some() {
        return Ok(());
    }
    let opened = CursorAcpSession::open(
        &state.config.key,
        requested_model,
        &state.config.workspace_cwd,
    )
    .await
    .context("open cursor-agent ACP session")?;
    **guard = Some(opened);
    Ok(())
}

/// Pre-opens up to `count` pool slots concurrently so paired HTTP requests
/// from the launched tool find ready ACP sessions instead of paying Node.js
/// startup + initialize + authenticate + session/new serially. Slots beyond
/// `count` are still opened on demand the first time a request lands on
/// them. `count` is capped at [`MAX_POOL_SESSIONS`].
///
/// The `prewarming` counter is bumped synchronously *before* each task is
/// spawned, so an HTTP request that lands during the small window between
/// `start_background` returning and the task actually starting will still
/// see `prewarming > 0` and choose to wait on the prewarmed slot instead of
/// expanding the pool.
fn spawn_prewarm(state: Arc<RouterState>, count: usize) {
    let target = count.min(MAX_POOL_SESSIONS);
    for slot_idx in 0..target {
        state.prewarming.fetch_add(1, Ordering::SeqCst);
        spawn_single_prewarm(state.clone(), slot_idx);
    }
}

fn spawn_single_prewarm(state: Arc<RouterState>, slot_idx: usize) {
    tokio::spawn(async move {
        // Drop guard so the counter decrements on any exit path, including
        // panics inside CursorAcpSession::open.
        let _decrement = PrewarmCounterGuard {
            state: state.clone(),
        };
        // Reserve the slot at the requested index, growing the pool to fit
        // if necessary. Concurrent requests that arrive before the prewarm
        // finishes find the slot already in the pool and wait on its lock.
        let slot: SessionSlot = {
            let mut pool = state.pool.lock().await;
            while pool.len() <= slot_idx {
                pool.push(Arc::new(Mutex::new(None)));
            }
            pool[slot_idx].clone()
        };
        let mut guard = slot.lock_owned().await;
        if guard.is_some() {
            return;
        }
        match CursorAcpSession::open(&state.config.key, None, &state.config.workspace_cwd).await {
            Ok(session) => {
                *guard = Some(session);
            }
            Err(e) => {
                // Don't fail the launch — the next HTTP request retries
                // open() on demand and surfaces the error via JSON 502 then.
                eprintln!("aivo: cursor ACP session pre-warm (slot {slot_idx}) failed: {e:#}");
            }
        }
    });
}

/// Decrements `RouterState::prewarming` on drop. Used by `spawn_single_prewarm`
/// so the counter is bookkept correctly across both Ok and Err exit paths,
/// and even if `CursorAcpSession::open` panics.
struct PrewarmCounterGuard {
    state: Arc<RouterState>,
}

impl Drop for PrewarmCounterGuard {
    fn drop(&mut self) {
        self.state.prewarming.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn run_router(listener: TcpListener, state: Arc<RouterState>) -> Result<()> {
    http_utils::run_streaming_router(listener, state, |request, state, socket| async move {
        handle_request(request, state, socket).await;
    })
    .await
}

async fn handle_request(request: String, state: Arc<RouterState>, mut socket: TcpStream) {
    let path = extract_request_path(&request);
    let path = path.split('?').next().unwrap_or("").to_string();
    let method = request.split_whitespace().next().unwrap_or("").to_string();

    let log_id = log_inbound(&method, &path, &request).await;
    let started = Instant::now();

    // Collapse `/v1/...` (Claude/Anthropic style) and the unversioned `/...`
    // (Codex/Pi style) to a single canonical form so each handler is named
    // once below. Mirrors `commands::normalize_base_url`'s suffix-stripping
    // on outbound URLs; here we strip the same prefix from inbound paths.
    let canonical = strip_v1_prefix(&path);

    let (status, summary) = if method == "POST"
        && let Some(generate) = parse_gemini_generate_path(&path)
    {
        handle_gemini_generate(&mut socket, &state, &request, &generate).await
    } else {
        match (method.as_str(), canonical) {
            ("OPTIONS", _) => {
                let _ = write_cors_preflight(&mut socket).await;
                (204, None)
            }
            // Claude Code probes the router root with HEAD on launch and treats a
            // 404 as a configuration error. Reply 200 so the probe succeeds; GET
            // gets the same treatment for parity with the OpenAI/Anthropic SDKs.
            ("HEAD" | "GET", "/" | "") => {
                let _ = write_root_probe_response(&mut socket).await;
                (200, None)
            }
            ("GET", "/models") => write_models(&mut socket, &state).await,
            ("POST", "/chat/completions") => {
                handle_openai_chat(&mut socket, &state, &request).await
            }
            ("POST", "/messages") => handle_anthropic_messages(&mut socket, &state, &request).await,
            ("POST", "/responses") => handle_responses(&mut socket, &state, &request).await,
            _ => {
                // Echo the client's *original* path back in the 404 so debugging
                // shows what they actually sent, not the canonical form.
                let _ = write_not_found(&mut socket, &path).await;
                (404, None)
            }
        }
    };

    log_outbound(
        log_id,
        &method,
        &path,
        status,
        summary,
        started.elapsed().as_millis() as u64,
    )
    .await;
}

// === Read-only endpoints ===

async fn write_models(socket: &mut TcpStream, state: &RouterState) -> (u16, Option<String>) {
    let models = match cached_models(state).await {
        Ok(m) => m,
        Err(err) => {
            let msg = format!("cursor-agent models failed: {err}");
            let _ = write_json_error(socket, 502, &msg).await;
            return (502, Some(msg));
        }
    };
    let body = build_models_response_body(&models);
    let summary = http_debug::global().is_some().then(|| body.to_string());
    let _ = write_json_response(socket, 200, &body).await;
    (200, summary)
}

/// Builds the `/v1/models` response body that satisfies both OpenAI's standard
/// `{"object":"list","data":[...]}` consumers AND codex 0.132+'s
/// `codex_models_manager`, whose `ModelsResponse` expects `{"models": [<ModelInfo>...]}`
/// with each entry carrying the full strongly-typed codex `ModelInfo` shape
/// (see `codex-rs/protocol/src/openai_models.rs`). Missing any required field
/// makes codex spam `failed to decode models response: missing field "<x>"` on
/// every interaction.
///
/// We emit both `data` (OpenAI) and `models` (codex) twin arrays. Codex's
/// `ModelsResponse` doesn't `deny_unknown_fields`, so the extra `data` / `object`
/// keys are ignored by codex; OpenAI-style consumers ignore the unknown
/// `models` field in turn.
fn build_models_response_body(models: &[String]) -> Value {
    let openai_entries: Vec<Value> = models
        .iter()
        .map(|id| json!({"id": id, "object": "model", "owned_by": CURSOR_ACP_SENTINEL}))
        .collect();
    let codex_entries: Vec<Value> = models.iter().map(|id| codex_model_info(id)).collect();
    json!({
        "object": "list",
        "data": openai_entries,
        "models": codex_entries,
    })
}

/// Minimal valid `ModelInfo` for codex. Every field serde considers required
/// (no `#[serde(default)]`, no `Option` with default) is present; optional
/// fields are omitted so codex's defaults apply.
fn codex_model_info(id: &str) -> Value {
    json!({
        "slug": id,
        "display_name": id,
        "description": null,
        "supported_reasoning_levels": [],
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "availability_nux": null,
        "upgrade": null,
        "base_instructions": "",
        "supports_reasoning_summaries": false,
        "support_verbosity": false,
        "default_verbosity": null,
        "apply_patch_tool_type": null,
        "truncation_policy": {"mode": "tokens", "limit": 100_000},
        "supports_parallel_tool_calls": false,
        "experimental_supported_tools": [],
    })
}

async fn cached_models(state: &RouterState) -> Result<Vec<String>> {
    let mut guard = state.cached_models.lock().await;
    if let Some(existing) = guard.as_ref() {
        return Ok(existing.clone());
    }
    // Consult the disk cache before paying the cursor-agent subprocess spawn.
    // The cache key matches what `aivo models cursor` and the model picker
    // write, so a warm entry from either populates the router transparently.
    let disk_cache = state.config.models_cache.as_ref();
    let cache_key = cursor_acp::cursor_models_cache_identity(&state.config.key);
    if let Some(cache) = disk_cache
        && let Some(cached) = cache.get(&cache_key).await
        && !cached.is_empty()
    {
        *guard = Some(cached.clone());
        return Ok(cached);
    }
    let models = cursor_acp::list_cursor_models(&state.config.key).await?;
    if let Some(cache) = disk_cache {
        cache.set(&cache_key, models.clone()).await;
    }
    *guard = Some(models.clone());
    Ok(models)
}

/// Best-effort: when an HTTP turn fails (client disconnect, write failure, or
/// upstream ACP error), notify cursor-agent to stop generating so we don't
/// keep burning tokens after the launched tool has gone away. The session
/// stays alive for subsequent requests on the same router.
async fn cancel_session_on_error<T>(session: &CursorAcpSession, outcome: &Result<T>) {
    if outcome.is_err() {
        let _ = session.cancel().await;
    }
}

/// Per-protocol inputs after the request body has been parsed and reduced
/// to a flat ACP prompt. The protocol-specific differences live in the
/// closures passed to [`run_turn`] (SSE writer + non-stream body builder).
struct ParsedTurn {
    stream_flag: bool,
    requested_model: Option<String>,
    image_blocks: Vec<Value>,
    prompt: String,
}

/// Shared scaffold for OpenAI / Anthropic / Responses / Gemini turns;
/// per-protocol bits live in the SSE-writer and body-builder closures.
async fn run_turn<S, A>(
    socket: &mut TcpStream,
    state: &RouterState,
    parsed: ParsedTurn,
    response_model_fallback: &str,
    stream_sse: S,
    aggregate_body: A,
) -> Result<Option<String>>
where
    S: AsyncFnOnce(
        &mut TcpStream,
        &mut crate::services::acp_client::PromptStream,
        &str,
        u64,
    ) -> Result<String>,
    A: FnOnce(&AggregatedTurn, &str, u64) -> Value,
{
    let ParsedTurn {
        stream_flag,
        requested_model,
        image_blocks,
        prompt,
    } = parsed;

    let mut guard = acquire_session_slot(state).await;
    ensure_session_open(&mut guard, state, requested_model.as_deref()).await?;
    let session = guard.as_mut().expect("session opened above");

    if let Some(model) = &requested_model {
        let _ = session.set_model(model).await;
    }

    if !image_blocks.is_empty() && !session.supports_image_prompts() {
        return Err(anyhow!(image_capability_error()));
    }

    let response_model = session
        .model_id()
        .map(str::to_string)
        .or(requested_model)
        .unwrap_or_else(|| response_model_fallback.to_string());

    let input_tokens = estimate_tokens(&prompt);
    let blocks = cursor_acp::assemble_prompt_blocks(&prompt, image_blocks);
    let mut stream = session
        .prompt_with_blocks(blocks)
        .await
        .context("cursor-agent session/prompt")?;

    let outcome: Result<Option<String>> = if stream_flag {
        stream_sse(socket, &mut stream, &response_model, input_tokens)
            .await
            .map(Some)
    } else {
        match aggregate_prompt_stream(&mut stream).await {
            Ok(aggregated) => {
                let summary = aggregated.content.clone();
                let body = aggregate_body(&aggregated, &response_model, input_tokens);
                write_json_response(socket, 200, &body).await?;
                Ok(Some(summary))
            }
            Err(e) => Err(e),
        }
    };
    cancel_session_on_error(session, &outcome).await;
    outcome
}

/// Walks the anyhow error chain looking for an io::Error of the broken-pipe
/// or connection-reset variety. Used by the dispatcher to log status 499
/// (nginx's "client closed request" convention) instead of a misleading 502
/// when the launched tool closed the socket mid-stream.
fn is_client_disconnect(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
            )
        })
    })
}

/// Resolves the status code logged for a failed turn. Broken pipes get 499
/// so logs distinguish client-initiated disconnects (expected, harmless) from
/// real upstream failures (502).
fn status_for_handler_error(err: &anyhow::Error) -> u16 {
    if is_client_disconnect(err) { 499 } else { 502 }
}

// === OpenAI chat completions ===

async fn handle_openai_chat(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> (u16, Option<String>) {
    match run_openai_chat(socket, state, request).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            // Errors that surface *before* we've sent any bytes get a clean JSON
            // 502; errors after the stream head is on the wire just close the
            // socket (the client sees a truncated SSE stream). Broken pipes
            // get 499 in the log so post-mortems can tell client disconnects
            // apart from real upstream failures.
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

async fn run_openai_chat(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value =
        serde_json::from_str(body_str).context("parse OpenAI chat completion request body")?;
    if body.get("messages").and_then(Value::as_array).is_none() {
        return Err(anyhow!("`messages` array is required"));
    }
    let parsed = ParsedTurn {
        stream_flag: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        requested_model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        image_blocks: extract_openai_image_blocks(&body)?,
        prompt: reduce_openai_request_to_prompt(&body),
    };
    if parsed.prompt.trim().is_empty() && parsed.image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }
    run_turn(
        socket,
        state,
        parsed,
        CURSOR_ACP_SENTINEL,
        stream_openai_chat_sse,
        openai_completion_body,
    )
    .await
}

/// Streams Cursor session/update events into the socket as an OpenAI
/// `chat.completion.chunk` SSE feed, terminating with `data: [DONE]`. Returns
/// the aggregated assistant text so the dispatcher can log it.
async fn stream_openai_chat_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let chat_id = new_chat_completion_id();
    let created = current_unix_timestamp();

    // Many OpenAI clients expect a leading `delta.role = "assistant"` chunk.
    let role_chunk =
        openai_chunk_frame(&chat_id, created, model, json!({"role": "assistant"}), None);
    write_sse_chunk(socket, &role_chunk).await?;

    let mut finish_reason = "stop".to_string();
    let mut aggregated = String::new();
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    aggregated.push_str(text);
                    let chunk = openai_chunk_frame(
                        &chat_id,
                        created,
                        model,
                        json!({"content": text}),
                        None,
                    );
                    write_sse_chunk(socket, &chunk).await?;
                } else {
                    // Non-text updates (tool_call_update, plan, thought, …)
                    // get a keepalive comment line. cursor-agent can spend
                    // 10+ s on internal work; OpenAI-SDK clients (pi 6.26
                    // observed dropping after ~9 s of silence) treat the
                    // stream as stalled and reconnect. Comment lines are
                    // ignored per the SSE spec but reset client idle timers.
                    write_sse_chunk(socket, SSE_KEEPALIVE).await?;
                }
            }
            PromptEvent::Done(result) => {
                match result {
                    Ok(_) => {}
                    Err(err) => {
                        finish_reason = format!("error:{}", err.code);
                    }
                }
                break;
            }
        }
    }

    let final_chunk = openai_chunk_frame(
        &chat_id,
        created,
        model,
        json!({}),
        Some(finish_reason.as_str()),
    );
    write_sse_chunk(socket, &final_chunk).await?;

    // Emit a usage-only chunk per OpenAI's stream_options=include_usage
    // convention. Modern SDKs accept it unconditionally; older ones ignore
    // unrecognized fields. Without this, Codex/Pi see prompt_tokens=0 and
    // can't display context-window usage.
    let output_tokens = estimate_tokens(&aggregated);
    let usage_chunk = openai_usage_chunk(&chat_id, created, model, input_tokens, output_tokens);
    write_sse_chunk(socket, &usage_chunk).await?;
    write_sse_chunk(socket, "data: [DONE]\n\n").await?;
    write_chunk_terminator(socket).await?;
    Ok(aggregated)
}

fn openai_usage_chunk(
    id: &str,
    created: i64,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> String {
    let payload = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens.saturating_add(completion_tokens),
        },
    });
    format!("data: {payload}\n\n")
}

struct AggregatedTurn {
    content: String,
    reasoning: String,
}

async fn aggregate_prompt_stream(
    stream: &mut crate::services::acp_client::PromptStream,
) -> Result<AggregatedTurn> {
    let mut content = String::new();
    let mut reasoning = String::new();
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    content.push_str(text);
                } else if let Some(text) = extract_agent_thought(&value) {
                    reasoning.push_str(text);
                }
            }
            PromptEvent::Done(result) => {
                result
                    .map_err(|e| anyhow!(e))
                    .context("cursor-agent session/prompt failed")?;
                break;
            }
        }
    }
    Ok(AggregatedTurn { content, reasoning })
}

fn openai_completion_body(turn: &AggregatedTurn, model: &str, input_tokens: u64) -> Value {
    let mut message = json!({"role": "assistant", "content": turn.content});
    if !turn.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(turn.reasoning.clone());
    }
    let completion_tokens = estimate_tokens(&turn.content);
    json!({
        "id": new_chat_completion_id(),
        "object": "chat.completion",
        "created": current_unix_timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": input_tokens.saturating_add(completion_tokens),
        },
    })
}

fn openai_chunk_frame(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> String {
    let payload = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    format!("data: {payload}\n\n")
}

fn extract_agent_text(value: &Value) -> Option<&str> {
    let update = value.get("update")?;
    if update.get("sessionUpdate").and_then(Value::as_str) != Some("agent_message_chunk") {
        return None;
    }
    update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
}

fn extract_agent_thought(value: &Value) -> Option<&str> {
    let update = value.get("update")?;
    if update.get("sessionUpdate").and_then(Value::as_str) != Some("agent_thought_chunk") {
        return None;
    }
    update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
}

/// Maximum characters per tool-result transcript entry — anything beyond is
/// truncated with a marker. Tool outputs (large file contents, command output)
/// would otherwise blow past Cursor's context.
const TOOL_RESULT_CHAR_LIMIT: usize = 4000;
/// Maximum characters per tool-schema description.
const TOOL_DESCRIPTION_CHAR_LIMIT: usize = 240;

/// Reduces an OpenAI chat-completions request body to a flat ACP prompt.
/// Preserves tool schemas as a compact "Available tools" header and turns
/// `tool_calls` (assistant) and `role=tool` messages into transcript markers so
/// the downstream agent sees what the original tool was doing.
pub(crate) fn reduce_openai_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_openai_tools_list(tools)
    {
        parts.push(block);
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let text = extract_openai_message_text(msg.get("content"));
        match role {
            "system" | "developer" => {
                if !text.trim().is_empty() {
                    parts.push(format!("System: {text}"));
                }
            }
            "user" => {
                if !text.trim().is_empty() {
                    parts.push(format!("User: {text}"));
                }
            }
            "assistant" => {
                if !text.trim().is_empty() {
                    parts.push(format!("Assistant: {text}"));
                }
                if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        if let Some(line) = format_openai_tool_call(call) {
                            parts.push(line);
                        }
                    }
                }
            }
            "tool" => {
                let name = msg
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| msg.get("tool_call_id").and_then(Value::as_str))
                    .unwrap_or("tool");
                parts.push(format_tool_result_block(name, &text));
            }
            other => {
                if !text.trim().is_empty() {
                    parts.push(format!("{other}: {text}"));
                }
            }
        }
    }
    parts.join("\n\n")
}

// === Anthropic messages ===

async fn handle_anthropic_messages(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> (u16, Option<String>) {
    match run_anthropic_messages(socket, state, request).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

async fn run_anthropic_messages(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value =
        serde_json::from_str(body_str).context("parse Anthropic messages request body")?;
    let stream_flag = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);

    // Title-gen short-circuit. Claude Code fires this in parallel with every
    // real turn — forwarding to cursor costs 60-100 s of full-model time.
    // Skip the transcript reduction + image-block walk on this path; only
    // the first user message is needed to derive the title.
    if is_title_generation_request(&body) {
        let model = requested_model.unwrap_or_else(|| CURSOR_ACP_SENTINEL.to_string());
        let user_text = extract_first_user_text(&body).unwrap_or_default();
        let title = build_title_from_user_text(&user_text);
        return short_circuit_title_response(
            socket,
            &model,
            &title,
            stream_flag,
            estimate_tokens(&user_text),
        )
        .await;
    }

    let parsed = ParsedTurn {
        stream_flag,
        requested_model,
        image_blocks: extract_anthropic_image_blocks(&body)?,
        prompt: reduce_anthropic_request_to_prompt(&body),
    };
    if parsed.prompt.trim().is_empty() && parsed.image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }

    run_turn(
        socket,
        state,
        parsed,
        CURSOR_ACP_SENTINEL,
        stream_anthropic_sse,
        anthropic_message_body,
    )
    .await
}

/// Detects Claude Code's title-generation subagent request by its
/// system-prompt signature. Matching three distinct fragments keeps false
/// positives very unlikely — a coding request would have to coincidentally
/// contain all three to be misclassified.
fn is_title_generation_request(body: &Value) -> bool {
    let system_text = extract_anthropic_system_text(body.get("system"));
    system_text.contains("Generate a concise")
        && system_text.contains("sentence-case title")
        && system_text.contains("Return JSON")
}

/// Pulls a reasonable conversation title out of the user-visible messages.
/// Falls back to a static label only when the body carries no usable text.
#[cfg(test)]
fn build_title_from_anthropic_body(body: &Value) -> String {
    build_title_from_user_text(&extract_first_user_text(body).unwrap_or_default())
}

fn build_title_from_user_text(user_text: &str) -> String {
    if user_text.trim().is_empty() {
        "Coding session".to_string()
    } else {
        compose_short_title(user_text)
    }
}

fn extract_first_user_text(body: &Value) -> Option<String> {
    let messages = body.get("messages").and_then(Value::as_array)?;
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) == Some("user") {
            let text = collect_anthropic_text(msg.get("content"));
            if !text.trim().is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn collect_anthropic_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut acc = String::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(t) = block.get("text").and_then(Value::as_str)
                {
                    if !acc.is_empty() {
                        acc.push(' ');
                    }
                    acc.push_str(t);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

/// Truncates a free-form prompt to a 3-7 word title, breaking on word
/// boundaries and capping at ~60 visible chars to match Claude Code's UI
/// expectations for the session label.
fn compose_short_title(raw: &str) -> String {
    let trimmed = raw.trim();
    let first_line = trimmed.lines().next().unwrap_or("").trim();
    let words: Vec<&str> = first_line.split_whitespace().take(7).collect();
    if words.is_empty() {
        return "Coding session".to_string();
    }
    let mut title = words.join(" ");
    if title.chars().count() > 60 {
        let truncated: String = title.chars().take(60).collect();
        let cut = truncated
            .rfind(' ')
            .map(|i| truncated[..i].to_string())
            .unwrap_or(truncated);
        title = cut;
    }
    title
}

/// Emits a hardcoded Anthropic response with a JSON `{"title":"..."}` body,
/// skipping any cursor work. Supports both streaming and one-shot modes so
/// Claude Code sees a normal `/v1/messages` reply.
async fn short_circuit_title_response(
    socket: &mut TcpStream,
    model: &str,
    title: &str,
    stream_flag: bool,
    input_tokens: u64,
) -> Result<Option<String>> {
    let json_content = json!({"title": title}).to_string();
    if !stream_flag {
        let turn = AggregatedTurn {
            content: json_content.clone(),
            reasoning: String::new(),
        };
        let body = anthropic_message_body(&turn, model, input_tokens);
        write_json_response(socket, 200, &body).await?;
        return Ok(Some(json_content));
    }

    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let message_id = new_anthropic_message_id();
    let output_tokens = estimate_tokens(&json_content);

    write_sse_chunk(
        socket,
        &sse_named_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &anthropic_content_block_start(0, AnthropicBlockKind::Text),
    )
    .await?;
    write_sse_chunk(socket, &anthropic_text_delta(0, &json_content)).await?;
    write_sse_chunk(socket, &anthropic_content_block_stop(0)).await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event("message_stop", &json!({"type": "message_stop"})),
    )
    .await?;
    write_chunk_terminator(socket).await?;
    Ok(Some(json_content))
}

async fn stream_anthropic_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let message_id = new_anthropic_message_id();
    write_sse_chunk(
        socket,
        &sse_named_event(
            "message_start",
            &json!({
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            }),
        ),
    )
    .await?;
    // Content blocks open lazily so Claude sees a clean interleaving of
    // `thinking` (cursor's reasoning + tool-call titles) and `text` (the
    // agent's user-visible message). Each transition closes the current
    // block and starts the next one at a fresh index — the protocol allows
    // multiple blocks per message and Claude Code's UI uses block type to
    // pick its renderer (collapsible "Cogitated…" panel vs. message bubble).
    let mut block_state = AnthropicBlockState::default();
    let mut stop_reason = "end_turn";
    let mut output_tokens: u64 = 0;
    let mut aggregated = String::new();
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    aggregated.push_str(text);
                    output_tokens = output_tokens.saturating_add(estimate_tokens(text));
                    block_state
                        .ensure_kind(socket, AnthropicBlockKind::Text)
                        .await?;
                    write_sse_chunk(socket, &anthropic_text_delta(block_state.index(), text))
                        .await?;
                } else if let Some(thought) = extract_agent_thought(&value) {
                    block_state
                        .ensure_kind(socket, AnthropicBlockKind::Thinking)
                        .await?;
                    write_sse_chunk(
                        socket,
                        &anthropic_thinking_delta(block_state.index(), thought),
                    )
                    .await?;
                } else if let Some(marker) = extract_tool_call_marker(&value) {
                    // Surface cursor's tool-call titles as inline thinking
                    // text. Claude Code shows them inside the "Cogitated…"
                    // panel — without this, the user sees no progress at all
                    // while cursor runs (or tries to run) tools, and the
                    // status indicator can stall for tens of seconds.
                    block_state
                        .ensure_kind(socket, AnthropicBlockKind::Thinking)
                        .await?;
                    write_sse_chunk(
                        socket,
                        &anthropic_thinking_delta(block_state.index(), &marker),
                    )
                    .await?;
                }
                // available_commands_update / session_info_update etc. are
                // pure protocol overhead and intentionally dropped.
            }
            PromptEvent::Done(result) => {
                if result.is_err() {
                    stop_reason = "error";
                }
                break;
            }
        }
    }

    block_state.close(socket).await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event("message_stop", &json!({"type": "message_stop"})),
    )
    .await?;
    write_chunk_terminator(socket).await?;
    Ok(aggregated)
}

/// Builds a single SSE frame in `event: <name>\ndata: <json>\n\n` form, used
/// by Anthropic and Responses streams alike.
fn sse_named_event(name: &str, payload: &Value) -> String {
    format!("event: {name}\ndata: {payload}\n\n")
}

/// SSE comment line. Per the SSE spec, lines starting with `:` are ignored by
/// the client but reset its idle-disconnect timer. Used by the streaming
/// handlers when cursor-agent emits a non-text update (tool_call, plan,
/// thought) — without a periodic byte on the wire, OpenAI-SDK clients
/// interpret the silence as a stalled stream and reconnect mid-turn.
const SSE_KEEPALIVE: &str = ": keepalive\n\n";

/// Anthropic content-block kind tracked by [`AnthropicBlockState`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AnthropicBlockKind {
    Text,
    Thinking,
}

/// Lazy block-opener for `stream_anthropic_sse`. Tracks the current block's
/// index and type so cursor's interleaved messages and thoughts get rendered
/// as alternating `text` and `thinking` blocks. Each transition closes the
/// current block and starts a fresh one at the next index, which is what
/// Anthropic's protocol requires (and what Claude Code's UI uses to pick
/// between the "Cogitated…" panel and the inline message bubble).
#[derive(Default)]
struct AnthropicBlockState {
    current: Option<(u32, AnthropicBlockKind)>,
    next_index: u32,
}

impl AnthropicBlockState {
    fn index(&self) -> u32 {
        self.current.map(|(i, _)| i).unwrap_or(0)
    }

    async fn ensure_kind(
        &mut self,
        socket: &mut TcpStream,
        kind: AnthropicBlockKind,
    ) -> Result<()> {
        if let Some((_, current)) = self.current
            && current == kind
        {
            return Ok(());
        }
        if let Some((idx, _)) = self.current.take() {
            write_sse_chunk(socket, &anthropic_content_block_stop(idx)).await?;
        }
        let idx = self.next_index;
        self.next_index += 1;
        write_sse_chunk(socket, &anthropic_content_block_start(idx, kind)).await?;
        self.current = Some((idx, kind));
        Ok(())
    }

    async fn close(&mut self, socket: &mut TcpStream) -> Result<()> {
        if let Some((idx, _)) = self.current.take() {
            write_sse_chunk(socket, &anthropic_content_block_stop(idx)).await?;
        }
        Ok(())
    }
}

fn anthropic_content_block_start(index: u32, kind: AnthropicBlockKind) -> String {
    let body = match kind {
        AnthropicBlockKind::Text => json!({"type": "text", "text": ""}),
        AnthropicBlockKind::Thinking => json!({"type": "thinking", "thinking": ""}),
    };
    sse_named_event(
        "content_block_start",
        &json!({
            "type": "content_block_start",
            "index": index,
            "content_block": body,
        }),
    )
}

fn anthropic_content_block_stop(index: u32) -> String {
    sse_named_event(
        "content_block_stop",
        &json!({"type": "content_block_stop", "index": index}),
    )
}

fn anthropic_text_delta(index: u32, text: &str) -> String {
    sse_named_event(
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }),
    )
}

fn anthropic_thinking_delta(index: u32, thinking: &str) -> String {
    sse_named_event(
        "content_block_delta",
        &json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        }),
    )
}

/// Renders a cursor `tool_call` / `tool_call_update` event as a one-line
/// "[kind] title" string suitable for streaming into a thinking block. Falls
/// back to `None` for events that carry no human-readable hint (we drop
/// those silently so the thinking panel stays clean).
fn extract_tool_call_marker(value: &Value) -> Option<String> {
    let update = value.get("update")?;
    let kind = update.get("sessionUpdate").and_then(Value::as_str)?;
    if kind != "tool_call" && kind != "tool_call_update" {
        return None;
    }
    let title = update.get("title").and_then(Value::as_str);
    let status = update.get("status").and_then(Value::as_str);
    let tool_kind = update.get("kind").and_then(Value::as_str);
    match (title, status, tool_kind) {
        (Some(t), Some(s), _) => Some(format!("\n[{s}] {t}\n")),
        (Some(t), None, _) => Some(format!("\n[tool] {t}\n")),
        (None, Some(s), Some(k)) => Some(format!("\n[{k} → {s}]\n")),
        (None, _, Some(k)) => Some(format!("\n[tool: {k}]\n")),
        _ => None,
    }
}

fn anthropic_message_body(turn: &AggregatedTurn, model: &str, input_tokens: u64) -> Value {
    let mut content_blocks = Vec::new();
    if !turn.content.is_empty() {
        content_blocks.push(json!({"type": "text", "text": turn.content}));
    }
    json!({
        "id": new_anthropic_message_id(),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": estimate_tokens(&turn.content),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    })
}

/// Reduces an Anthropic `/v1/messages` request body to a flat ACP prompt.
/// Preserves `tools` as an "Available tools" header and surfaces `tool_use` /
/// `tool_result` blocks as transcript markers so multi-turn tool loops keep
/// their context when forwarded to Cursor.
pub(crate) fn reduce_anthropic_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_anthropic_tools_list(tools)
    {
        parts.push(block);
    }
    let system_text = extract_anthropic_system_text(body.get("system"));
    if !system_text.trim().is_empty() {
        parts.push(format!("System: {system_text}"));
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let label = match role {
            "user" => "User",
            "assistant" => "Assistant",
            other => other,
        };
        for entry in flatten_anthropic_message_blocks(label, msg.get("content")) {
            parts.push(entry);
        }
    }
    parts.join("\n\n")
}

/// Anthropic accepts `system` as a string or an array of text-typed blocks.
fn extract_anthropic_system_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut acc = String::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(t) = block.get("text").and_then(Value::as_str)
                {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(t);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

/// Walks one Anthropic message and yields a transcript line per logical block.
/// Plain text accumulates under `User:` / `Assistant:`; tool_use / tool_result
/// blocks become their own entries so the downstream agent sees the loop.
fn flatten_anthropic_message_blocks(label: &str, content: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(content) = content else {
        return out;
    };
    let mut buffer = String::new();
    let flush = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.trim().is_empty() {
            out.push(format!("{label}: {buf}"));
        }
        buf.clear();
    };
    match content {
        Value::String(s) if !s.trim().is_empty() => {
            out.push(format!("{label}: {s}"));
        }
        Value::Array(blocks) => {
            for block in blocks {
                let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            if !buffer.is_empty() {
                                buffer.push('\n');
                            }
                            buffer.push_str(t);
                        }
                    }
                    "tool_use" => {
                        flush(&mut buffer, &mut out);
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let args = block.get("input").cloned().unwrap_or(Value::Null);
                        out.push(format_tool_call_line(name, &args));
                    }
                    "tool_result" => {
                        flush(&mut buffer, &mut out);
                        let name = block
                            .get("tool_use_id")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let result_text = extract_anthropic_tool_result_text(block.get("content"));
                        out.push(format_tool_result_block(name, &result_text));
                    }
                    "image" | "document" => {
                        flush(&mut buffer, &mut out);
                        out.push(format!("[{kind} attachment omitted]"));
                    }
                    _ => {}
                }
            }
            flush(&mut buffer, &mut out);
        }
        _ => {}
    }
    out
}

fn extract_anthropic_tool_result_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut acc = String::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(t) = block.get("text").and_then(Value::as_str)
                {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(t);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

fn new_anthropic_message_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let salt = current_unix_timestamp_micros();
    format!("msg_cur{n:x}{salt:x}")
}

// === Responses API (Codex) ===

async fn handle_responses(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> (u16, Option<String>) {
    match run_responses(socket, state, request).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

async fn run_responses(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value = serde_json::from_str(body_str).context("parse Responses request body")?;
    let parsed = ParsedTurn {
        stream_flag: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        requested_model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        image_blocks: extract_responses_image_blocks(&body)?,
        prompt: reduce_responses_request_to_prompt(&body),
    };
    if parsed.prompt.trim().is_empty() && parsed.image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }
    run_turn(
        socket,
        state,
        parsed,
        CURSOR_ACP_SENTINEL,
        stream_responses_sse,
        responses_completion_body,
    )
    .await
}

async fn stream_responses_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let resp_id = new_responses_id();
    let msg_id = new_anthropic_message_id();
    let created = current_unix_timestamp();

    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.created",
            &json!({
                "type": "response.created",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "model": model,
                    "created_at": created,
                    "status": "in_progress",
                    "output": [],
                },
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "response_id": resp_id,
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.content_part.added",
            &json!({
                "type": "response.content_part.added",
                "response_id": resp_id,
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""},
            }),
        ),
    )
    .await?;

    let mut full_text = String::new();
    let mut errored = false;
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    full_text.push_str(text);
                    write_sse_chunk(
                        socket,
                        &sse_named_event(
                            "response.output_text.delta",
                            &json!({
                                "type": "response.output_text.delta",
                                "response_id": resp_id,
                                "item_id": msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": text,
                            }),
                        ),
                    )
                    .await?;
                } else {
                    // Keep the stream alive during non-text updates; see the
                    // OpenAI streamer for the rationale.
                    write_sse_chunk(socket, SSE_KEEPALIVE).await?;
                }
            }
            PromptEvent::Done(result) => {
                if result.is_err() {
                    errored = true;
                }
                break;
            }
        }
    }

    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "response_id": resp_id,
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": full_text,
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "response_id": resp_id,
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": full_text},
            }),
        ),
    )
    .await?;
    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "response_id": resp_id,
                "output_index": 0,
                "item": {
                    "id": msg_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": full_text, "annotations": []}],
                },
            }),
        ),
    )
    .await?;
    let final_status = if errored { "failed" } else { "completed" };
    write_sse_chunk(
        socket,
        &sse_named_event(
            "response.completed",
            &json!({
                "type": "response.completed",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "model": model,
                    "created_at": created,
                    "status": final_status,
                    "output": [{
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": full_text, "annotations": []}],
                    }],
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": estimate_tokens(&full_text),
                        "total_tokens": input_tokens.saturating_add(estimate_tokens(&full_text)),
                    },
                },
            }),
        ),
    )
    .await?;
    write_chunk_terminator(socket).await?;
    Ok(full_text)
}

fn responses_completion_body(turn: &AggregatedTurn, model: &str, input_tokens: u64) -> Value {
    let msg_id = new_anthropic_message_id();
    let output_tokens = estimate_tokens(&turn.content);
    json!({
        "id": new_responses_id(),
        "object": "response",
        "created_at": current_unix_timestamp(),
        "model": model,
        "status": "completed",
        "output": [{
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": turn.content, "annotations": []}],
        }],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens.saturating_add(output_tokens),
        },
    })
}

/// Reduces a Responses-API request body to a flat ACP prompt. Honors the
/// top-level `instructions` field as a system prefix, formats the `tools`
/// schema list, and walks the `input` array's typed items — including
/// `function_call` / `function_call_output` — so Codex tool loops keep their
/// context when forwarded to Cursor.
pub(crate) fn reduce_responses_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_responses_tools_list(tools)
    {
        parts.push(block);
    }
    let instructions = body
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !instructions.trim().is_empty() {
        parts.push(format!("System: {instructions}"));
    }
    match body.get("input") {
        Some(Value::String(s)) if !s.trim().is_empty() => {
            parts.push(format!("User: {s}"));
        }
        Some(Value::Array(items)) => {
            for item in items {
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                match item_type {
                    "function_call" => {
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let args = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .map(parse_loose_json)
                            .unwrap_or(Value::Null);
                        parts.push(format_tool_call_line(name, &args));
                    }
                    "function_call_output" => {
                        let name = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let output = item.get("output").and_then(Value::as_str).unwrap_or("");
                        parts.push(format_tool_result_block(name, output));
                    }
                    "reasoning" => {
                        // Codex emits its own chain-of-thought summary; drop it
                        // to keep prompts small. Cursor will produce its own.
                    }
                    "message" | "" => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        let label = match role {
                            "system" | "developer" => "System",
                            "user" => "User",
                            "assistant" => "Assistant",
                            other => other,
                        };
                        let text = extract_responses_item_text(item.get("content"));
                        if !text.trim().is_empty() {
                            parts.push(format!("{label}: {text}"));
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    parts.join("\n\n")
}

fn extract_responses_item_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => {
            let mut acc = String::new();
            for block in blocks {
                let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
                if (kind == "input_text" || kind == "output_text" || kind == "text")
                    && let Some(text) = block.get("text").and_then(Value::as_str)
                {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(text);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

fn parse_loose_json(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

// === Gemini generateContent (gemini-cli) ===

/// Parsed metadata for an incoming Gemini-protocol request.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GeminiGenerate {
    model: String,
    stream: bool,
}

/// Extracts the model name and stream flag from a Gemini-API path.
///
/// Accepts `/v1beta/models/<model>:generateContent`,
/// `/v1beta/models/<model>:streamGenerateContent`, and the `/v1/models/...` /
/// `/models/...` variants gemini-cli sometimes emits depending on the base URL
/// it was given. Returns `None` for non-Gemini paths so the dispatcher can
/// fall through to the canonical OpenAI/Anthropic/Responses handlers.
fn parse_gemini_generate_path(path: &str) -> Option<GeminiGenerate> {
    // Find the `/models/` segment; anything before it is the version prefix.
    let after_models = path
        .strip_prefix("/v1beta/models/")
        .or_else(|| path.strip_prefix("/v1/models/"))
        .or_else(|| path.strip_prefix("/models/"))?;
    let (model, action) = after_models.split_once(':')?;
    if model.is_empty() {
        return None;
    }
    let stream = match action {
        "streamGenerateContent" => true,
        "generateContent" => false,
        _ => return None,
    };
    Some(GeminiGenerate {
        model: model.to_string(),
        stream,
    })
}

async fn handle_gemini_generate(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
    generate: &GeminiGenerate,
) -> (u16, Option<String>) {
    match run_gemini_generate(socket, state, request, generate).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

async fn run_gemini_generate(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
    generate: &GeminiGenerate,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value =
        serde_json::from_str(body_str).context("parse Gemini generateContent request body")?;
    let parsed = ParsedTurn {
        stream_flag: generate.stream,
        requested_model: Some(generate.model.clone()),
        image_blocks: extract_gemini_image_blocks(&body)?,
        prompt: reduce_gemini_request_to_prompt(&body),
    };
    if parsed.prompt.trim().is_empty() && parsed.image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }
    run_turn(
        socket,
        state,
        parsed,
        &generate.model,
        stream_gemini_sse,
        gemini_response_body,
    )
    .await
}

/// Reduces a Gemini `GenerateContentRequest` body to a flat ACP prompt.
/// Mirrors `reduce_anthropic_request_to_prompt` in shape: tool schemas become an
/// "Available tools" header, `systemInstruction` becomes a `System:` block, and
/// `functionCall`/`functionResponse` parts become inline transcript markers so
/// multi-turn tool loops survive the round trip.
pub(crate) fn reduce_gemini_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_gemini_tools_list(tools)
    {
        parts.push(block);
    }
    let system_text = extract_gemini_system_text(body.get("systemInstruction"));
    if !system_text.trim().is_empty() {
        parts.push(format!("System: {system_text}"));
    }
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let label = match role {
            "model" => "Assistant",
            "user" => "User",
            other => other,
        };
        for entry in flatten_gemini_content_parts(label, content.get("parts")) {
            parts.push(entry);
        }
    }
    parts.join("\n\n")
}

fn extract_gemini_system_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.clone(),
        Value::Object(_) => {
            let parts = value
                .get("parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut acc = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(text);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

fn flatten_gemini_content_parts(label: &str, parts: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(parts) = parts.and_then(Value::as_array) else {
        return out;
    };
    let mut buffer = String::new();
    let flush = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.trim().is_empty() {
            out.push(format!("{label}: {buf}"));
        }
        buf.clear();
    };
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            if !buffer.is_empty() {
                buffer.push('\n');
            }
            buffer.push_str(text);
            continue;
        }
        if let Some(call) = part.get("functionCall") {
            flush(&mut buffer, &mut out);
            let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
            let args = call.get("args").cloned().unwrap_or(Value::Null);
            out.push(format_tool_call_line(name, &args));
            continue;
        }
        if let Some(resp) = part.get("functionResponse") {
            flush(&mut buffer, &mut out);
            let name = resp.get("name").and_then(Value::as_str).unwrap_or("tool");
            let result_text = extract_gemini_function_response_text(resp.get("response"));
            out.push(format_tool_result_block(name, &result_text));
            continue;
        }
        if part.get("inlineData").is_some() || part.get("fileData").is_some() {
            flush(&mut buffer, &mut out);
            out.push("[binary attachment omitted]".to_string());
        }
    }
    flush(&mut buffer, &mut out);
    out
}

fn extract_gemini_function_response_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn format_gemini_tools_list(tools: &[Value]) -> Option<String> {
    let mut lines = Vec::new();
    for tool in tools {
        let Some(declarations) = tool.get("functionDeclarations").and_then(Value::as_array) else {
            continue;
        };
        for decl in declarations {
            let Some(name) = decl.get("name").and_then(Value::as_str) else {
                continue;
            };
            let description = decl
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            lines.push(format_tool_schema_line(name, description));
        }
    }
    finalize_tools_block(lines)
}

/// Builds the non-streaming `GenerateContentResponse` JSON body.
fn gemini_response_body(turn: &AggregatedTurn, model: &str, input_tokens: u64) -> Value {
    let completion_tokens = estimate_tokens(&turn.content);
    let total_tokens = input_tokens.saturating_add(completion_tokens);
    let parts = if turn.content.is_empty() {
        json!([{"text": ""}])
    } else {
        json!([{"text": turn.content}])
    };
    json!({
        "candidates": [{
            "content": {"parts": parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": total_tokens,
        },
        "modelVersion": model,
    })
}

/// Streams Cursor session updates as a Gemini `streamGenerateContent` SSE feed.
/// Each `data:` line is a partial `GenerateContentResponse`; the final frame
/// carries `finishReason` and `usageMetadata`. Unlike OpenAI's SSE, Gemini
/// streams have no `[DONE]` marker — the stream just ends when the chunked
/// body closes.
async fn stream_gemini_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let mut aggregated = String::new();
    let mut output_tokens: u64 = 0;
    let mut finish_reason = "STOP";
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    aggregated.push_str(text);
                    output_tokens = output_tokens.saturating_add(estimate_tokens(text));
                    let frame = gemini_stream_text_frame(model, text);
                    write_sse_chunk(socket, &frame).await?;
                } else {
                    // Same rationale as the OpenAI handler: cursor can spend
                    // 10+ s on internal work between text deltas. Emit a
                    // comment line to keep idle SDK timers alive.
                    write_sse_chunk(socket, SSE_KEEPALIVE).await?;
                }
            }
            PromptEvent::Done(result) => {
                if result.is_err() {
                    finish_reason = "OTHER";
                }
                break;
            }
        }
    }
    let total_tokens = input_tokens.saturating_add(output_tokens);
    let final_frame = gemini_stream_final_frame(
        model,
        finish_reason,
        input_tokens,
        output_tokens,
        total_tokens,
    );
    write_sse_chunk(socket, &final_frame).await?;
    write_chunk_terminator(socket).await?;
    Ok(aggregated)
}

fn gemini_stream_text_frame(model: &str, text: &str) -> String {
    let payload = json!({
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "index": 0,
        }],
        "modelVersion": model,
    });
    format!("data: {payload}\n\n")
}

fn gemini_stream_final_frame(
    model: &str,
    finish_reason: &str,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
) -> String {
    let payload = json!({
        "candidates": [{
            "content": {"parts": [], "role": "model"},
            "finishReason": finish_reason,
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": total_tokens,
        },
        "modelVersion": model,
    });
    format!("data: {payload}\n\n")
}

// === Shared prompt-reduction helpers ===

/// Extracts plain text from OpenAI `messages[].content`. Accepts a bare string
/// or an array of typed parts (`{type: "text", text: "..."}`); image parts are
/// replaced with a placeholder marker so the agent knows something was elided.
fn extract_openai_message_text(content: Option<&Value>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => {
            let mut acc = String::new();
            for part in parts {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "text" | "input_text" | "output_text" => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            if !acc.is_empty() {
                                acc.push('\n');
                            }
                            acc.push_str(t);
                        }
                    }
                    "image_url" | "input_image" | "image" => {
                        if !acc.is_empty() {
                            acc.push('\n');
                        }
                        acc.push_str("[image attachment omitted]");
                    }
                    _ => {}
                }
            }
            acc
        }
        _ => String::new(),
    }
}

/// Extract inline image content parts from an incoming HTTP request and
/// return them as ACP image content blocks ready for `session/prompt`.
///
/// Per-protocol functions walk the protocol's message/content shape, decode
/// base64 data URLs / Anthropic source blocks / Gemini `inlineData`, and
/// return `Vec<Value>` of `{type: "image", mimeType, data}` blocks. Remote
/// URLs and file-id references are not supported and produce a clear error
/// rather than a silent drop — the router doesn't fetch URLs on behalf of
/// callers (security + latency boundary).
///
/// Caller is responsible for the capability check
/// ([`crate::services::cursor_acp::ensure_image_attachments_supported`])
/// after the session is open.
fn extract_openai_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(out);
    };
    for msg in messages {
        collect_openai_style_image_blocks(msg.get("content"), &mut out)?;
    }
    Ok(out)
}

fn extract_responses_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(input) = body.get("input").and_then(Value::as_array) else {
        return Ok(out);
    };
    for item in input {
        collect_openai_style_image_blocks(item.get("content"), &mut out)?;
    }
    Ok(out)
}

fn extract_anthropic_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(out);
    };
    for msg in messages {
        let Some(blocks) = msg.get("content").and_then(Value::as_array) else {
            continue;
        };
        for block in blocks {
            if block.get("type").and_then(Value::as_str) != Some("image") {
                continue;
            }
            let source = block.get("source").ok_or_else(|| {
                anyhow!("cursor: anthropic image block missing `source` — cannot forward")
            })?;
            let src_type = source.get("type").and_then(Value::as_str).unwrap_or("");
            match src_type {
                "base64" => {
                    let mime = source
                        .get("media_type")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("cursor: anthropic image missing media_type"))?;
                    let data = source
                        .get("data")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("cursor: anthropic image missing data"))?;
                    out.push(cursor_acp::image_block_from_inline(mime, data));
                }
                "url" => {
                    return Err(anyhow!(
                        "cursor: remote image URLs are not supported — send the image inline as base64"
                    ));
                }
                other => {
                    return Err(anyhow!(
                        "cursor: unknown anthropic image source type `{other}`"
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn extract_gemini_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return Ok(out);
    };
    for content in contents {
        let Some(parts) = content.get("parts").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if let Some(inline) = part.get("inlineData") {
                let (mime, data) = read_gemini_inline_pair(inline)?;
                if mime.starts_with("image/") {
                    out.push(cursor_acp::image_block_from_inline(mime, data));
                }
                continue;
            }
            if let Some(file) = part.get("fileData") {
                let mime = file
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .or_else(|| file.get("mime_type").and_then(Value::as_str))
                    .unwrap_or("");
                if mime.starts_with("image/") {
                    return Err(anyhow!(
                        "cursor: remote image fileData is not supported — send the image inline as base64"
                    ));
                }
            }
        }
    }
    Ok(out)
}

/// Walks an OpenAI/Responses-style `content` array (array of typed parts)
/// and pushes any image parts onto `out` as ACP image blocks. Returns Err
/// for unsupported shapes (remote URLs, file ids).
fn collect_openai_style_image_blocks(content: Option<&Value>, out: &mut Vec<Value>) -> Result<()> {
    let Some(Value::Array(parts)) = content else {
        return Ok(());
    };
    for part in parts {
        let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
        if !matches!(kind, "image_url" | "input_image" | "image") {
            continue;
        }
        // OpenAI ships the payload under either `image_url` (chat completions)
        // or `image_url`/`source`/data fields (Responses). Both `string`
        // ("data:..." directly) and `{url: "..."}` forms occur in the wild.
        let url_field = part
            .get("image_url")
            .or_else(|| part.get("url"))
            .or_else(|| part.get("data"));
        let url_str = match url_field {
            Some(Value::String(s)) => Some(s.as_str()),
            Some(Value::Object(_)) => url_field.and_then(|v| v.get("url")).and_then(Value::as_str),
            _ => None,
        };
        let Some(url) = url_str else {
            if part.get("file_id").is_some() {
                return Err(anyhow!(
                    "cursor: image references by file_id are not supported — inline the image as a data URL"
                ));
            }
            return Err(anyhow!("cursor: image part missing url/data field"));
        };
        let (mime, data) = parse_data_url(url).ok_or_else(|| {
            anyhow!(
                "cursor: remote image URLs are not supported — send the image inline as a data URL (data:image/...;base64,...)"
            )
        })?;
        out.push(cursor_acp::image_block_from_inline(&mime, &data));
    }
    Ok(())
}

fn read_gemini_inline_pair(inline: &Value) -> Result<(&str, &str)> {
    let mime = inline
        .get("mimeType")
        .and_then(Value::as_str)
        .or_else(|| inline.get("mime_type").and_then(Value::as_str))
        .ok_or_else(|| anyhow!("cursor: gemini inlineData missing mimeType"))?;
    let data = inline
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("cursor: gemini inlineData missing data"))?;
    Ok((mime, data))
}

/// Error returned when a request includes image parts but the live
/// cursor-agent session doesn't advertise `promptCapabilities.image`.
fn image_capability_error() -> &'static str {
    "cursor: this session's cursor-agent does not advertise promptCapabilities.image; remove the image from the request"
}

/// Parse a `data:<mime>;base64,<payload>` URL into `(mime, base64_payload)`.
/// Returns None for non-data URLs (http, gs://, file ids, etc.) so the
/// caller can produce a "remote URLs not supported" error. Also returns
/// None for non-base64 data URLs (e.g. URL-encoded text) — those would
/// require decoding and re-encoding, and image clients never use them.
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    let mut parts: Vec<&str> = meta.split(';').collect();
    let has_base64 = parts.iter().any(|p| p.eq_ignore_ascii_case("base64"));
    if !has_base64 {
        return None;
    }
    parts.retain(|p| !p.eq_ignore_ascii_case("base64"));
    let mime = parts
        .first()
        .copied()
        .filter(|s| !s.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    Some((mime, payload.to_string()))
}

fn format_openai_tool_call(call: &Value) -> Option<String> {
    let function = call.get("function")?;
    let name = function.get("name").and_then(Value::as_str)?;
    let args_value = match function.get("arguments") {
        Some(Value::String(s)) => parse_loose_json(s),
        Some(other) => other.clone(),
        None => Value::Null,
    };
    Some(format_tool_call_line(name, &args_value))
}

fn format_openai_tools_list(tools: &[Value]) -> Option<String> {
    let mut lines = Vec::new();
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        lines.push(format_tool_schema_line(name, description));
    }
    finalize_tools_block(lines)
}

fn format_anthropic_tools_list(tools: &[Value]) -> Option<String> {
    let mut lines = Vec::new();
    for tool in tools {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = tool
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        lines.push(format_tool_schema_line(name, description));
    }
    finalize_tools_block(lines)
}

fn format_responses_tools_list(tools: &[Value]) -> Option<String> {
    let mut lines = Vec::new();
    for tool in tools {
        // Responses-API function tools may be flat (`{type, name, description}`)
        // or chat-shaped (`{type:"function", function:{name,description}}`).
        let inner = tool.get("function").unwrap_or(tool);
        let Some(name) = inner.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = inner
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        lines.push(format_tool_schema_line(name, description));
    }
    finalize_tools_block(lines)
}

fn format_tool_schema_line(name: &str, description: &str) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        format!("- {name}")
    } else {
        let collapsed = collapse_whitespace(trimmed);
        let snippet = truncate_chars(&collapsed, TOOL_DESCRIPTION_CHAR_LIMIT);
        format!("- {name}: {snippet}")
    }
}

fn finalize_tools_block(lines: Vec<String>) -> Option<String> {
    if lines.is_empty() {
        None
    } else {
        let mut block = String::from("Available tools:\n");
        block.push_str(&lines.join("\n"));
        Some(block)
    }
}

fn format_tool_call_line(name: &str, args: &Value) -> String {
    let args_text = match args {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if args_text.is_empty() {
        format!("[Tool call] {name}()")
    } else {
        let snippet = truncate_chars(&args_text, TOOL_RESULT_CHAR_LIMIT);
        format!("[Tool call] {name}({snippet})")
    }
}

fn format_tool_result_block(name: &str, text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        format!("[Tool result for {name}]")
    } else {
        let snippet = truncate_chars(trimmed, TOOL_RESULT_CHAR_LIMIT);
        format!("[Tool result for {name}]\n{snippet}")
    }
}

fn collapse_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_space = false;
    for ch in input.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

fn truncate_chars(input: &str, max: usize) -> String {
    if input.chars().count() <= max {
        return input.to_string();
    }
    let mut out: String = input.chars().take(max).collect();
    out.push_str("…[truncated]");
    out
}

fn new_responses_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let salt = current_unix_timestamp_micros();
    format!("resp_cur{n:x}{salt:x}")
}

// === Low-level response writers ===

async fn write_json_response(socket: &mut TcpStream, status: u16, body: &Value) -> Result<()> {
    let body_str = body.to_string();
    let head = http_response_head_with_extra(
        status,
        "application/json",
        body_str.len(),
        cors_header_block(),
    );
    socket.write_all(head.as_bytes()).await?;
    socket.write_all(body_str.as_bytes()).await?;
    Ok(())
}

async fn write_json_error(socket: &mut TcpStream, status: u16, message: &str) -> Result<()> {
    write_json_response(
        socket,
        status,
        &json!({
            "error": {
                "message": message,
                "type": "cursor_model_router_error",
            }
        }),
    )
    .await
}

async fn write_cors_preflight(socket: &mut TcpStream) -> Result<()> {
    let head = http_response_head_with_extra(204, "text/plain", 0, cors_header_block());
    socket.write_all(head.as_bytes()).await?;
    Ok(())
}

async fn write_root_probe_response(socket: &mut TcpStream) -> Result<()> {
    write_json_response(
        socket,
        200,
        &json!({"router": "cursor", "endpoints": [
            "/v1/models", "/models",
            "/v1/chat/completions", "/chat/completions",
            "/v1/messages", "/messages",
            "/v1/responses", "/responses",
            "/v1beta/models/<model>:generateContent",
            "/v1beta/models/<model>:streamGenerateContent",
        ]}),
    )
    .await
}

async fn write_not_found(socket: &mut TcpStream, path: &str) -> Result<()> {
    write_json_response(
        socket,
        404,
        &json!({
            "error": {
                "message": format!("Unknown path `{path}`. Cursor router accepts /v1/models, /models, /v1/chat/completions, /chat/completions, /v1/messages, /messages, /v1/responses, /responses, /v1beta/models/<model>:generateContent, /v1beta/models/<model>:streamGenerateContent."),
                "type": "cursor_model_router_not_found",
            }
        }),
    )
    .await
}

async fn write_sse_chunk(socket: &mut TcpStream, sse_frame: &str) -> Result<()> {
    let chunk = format_http_chunk(sse_frame.as_bytes());
    socket.write_all(&chunk).await?;
    Ok(())
}

async fn write_chunk_terminator(socket: &mut TcpStream) -> Result<()> {
    socket.write_all(b"0\r\n\r\n").await?;
    Ok(())
}

/// Collapses incoming request paths so each handler has a single canonical
/// arm. Strips a leading `/v1` only when it's followed by `/` (or is the
/// whole path), so weird suffixes like `/v1bogus` aren't silently
/// re-interpreted as a known endpoint.
fn strip_v1_prefix(path: &str) -> &str {
    match path.strip_prefix("/v1") {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => rest,
        _ => path,
    }
}

fn new_chat_completion_id() -> String {
    // Same shape as the OpenAI API: `chatcmpl-` plus a short random suffix.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let salt = current_unix_timestamp_micros();
    format!("chatcmpl-cur{n:x}{salt:x}")
}

/// Rough byte-based token estimate (~4 chars/token). Cursor's session/prompt
/// result doesn't carry usage, so without this Claude Code's status-bar
/// context-percentage indicator sticks at 0 % regardless of how much was
/// actually streamed. Empty inputs return 0 so we never inflate `total_tokens`
/// to 1 for blank deltas.
fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(4) as u64
    }
}

fn current_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn current_unix_timestamp_micros() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0)
}

// === Debug logging ===
//
// The router writes raw HTTP to a TcpStream, so it can't piggyback on
// `LoggedSend` (which wraps reqwest). Mirror the schema by emitting paired
// `Phase::Request` / `Phase::Response` entries against the global
// `HttpDebugLogger`. The schema matches what other routers write so `aivo
// logs` and downstream JSONL consumers don't need a special case.

const CURSOR_ROUTER_URL_BASE: &str = "cursor-router://localhost";
const REQUEST_BODY_MAX: usize = 64 * 1024;
const RESPONSE_BODY_MAX: usize = 64 * 1024;

fn new_router_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("cur-router-{n:x}")
}

fn parse_request_headers(request: &str) -> BTreeMap<String, String> {
    let header_end = match request.find("\r\n\r\n") {
        Some(pos) => pos,
        None => return BTreeMap::new(),
    };
    request[..header_end]
        .lines()
        .skip(1) // request line
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect()
}

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = s[..max].to_string();
        out.push_str("…[truncated]");
        out
    }
}

async fn log_inbound(method: &str, path: &str, request: &str) -> Option<String> {
    let logger = http_debug::global()?;
    let id = new_router_request_id();
    let headers = parse_request_headers(request);
    let body = extract_request_body(request)
        .ok()
        .map(|b| truncate_for_log(b, REQUEST_BODY_MAX));
    logger
        .log(DebugEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            id: id.clone(),
            phase: Phase::Request,
            method: method.to_string(),
            url: format!("{CURSOR_ROUTER_URL_BASE}{path}"),
            status: None,
            duration_ms: None,
            request_headers: redact_headers(&headers),
            request_body: body,
            response_headers: BTreeMap::new(),
            response_body: None,
            error: None,
        })
        .await;
    Some(id)
}

async fn log_outbound(
    id: Option<String>,
    method: &str,
    path: &str,
    status: u16,
    summary: Option<String>,
    duration_ms: u64,
) {
    let (Some(logger), Some(id)) = (http_debug::global(), id) else {
        return;
    };
    let body = summary.map(|s| truncate_for_log(&s, RESPONSE_BODY_MAX));
    logger
        .log(DebugEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            id,
            phase: Phase::Response,
            method: method.to_string(),
            url: format!("{CURSOR_ROUTER_URL_BASE}{path}"),
            status: Some(status),
            duration_ms: Some(duration_ms),
            request_headers: BTreeMap::new(),
            request_body: None,
            response_headers: BTreeMap::new(),
            response_body: body,
            error: None,
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use zeroize::Zeroizing;

    fn fake_key() -> ApiKey {
        ApiKey {
            id: "cursor-test".to_string(),
            name: "cursor".to_string(),
            base_url: CURSOR_ACP_SENTINEL.to_string(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            routing_schema_version: 0,
            key: Zeroizing::new("cursor-login".to_string()),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn state_with_models(models: Vec<&str>) -> Arc<RouterState> {
        Arc::new(RouterState {
            config: CursorRouterConfig {
                key: fake_key(),
                workspace_cwd: "/tmp".to_string(),
                // Tests pin the in-memory cache directly via `cached_models`;
                // skip disk plumbing so they don't touch `~/.config/aivo/`.
                models_cache: None,
                prewarm_count: 0,
            },
            cached_models: Mutex::new(Some(models.into_iter().map(String::from).collect())),
            pool: Mutex::new(Vec::new()),
            prewarming: AtomicUsize::new(0),
        })
    }

    async fn round_trip(state: Arc<RouterState>, request: &str) -> String {
        let (listener, port) = bind_local_listener().await.unwrap();
        let handle = tokio::spawn(run_router(listener, state));

        let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).await.unwrap();
        handle.abort();
        String::from_utf8(buf).unwrap()
    }

    #[tokio::test]
    async fn get_v1_models_returns_openai_shaped_list() {
        let state = state_with_models(vec!["composer-2.5", "claude-sonnet-4-6"]);
        let response = round_trip(
            state,
            "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(response.contains("200 OK"));
        let body = response.split("\r\n\r\n").nth(1).unwrap();
        let parsed: Value = serde_json::from_str(body).unwrap();
        assert_eq!(parsed["object"], "list");
        let data = parsed["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "composer-2.5");
        assert_eq!(data[0]["owned_by"], "cursor");
    }

    #[tokio::test]
    async fn options_preflight_returns_cors() {
        let state = state_with_models(vec![]);
        let response = round_trip(
            state,
            "OPTIONS /v1/models HTTP/1.1\r\nOrigin: x\r\nConnection: close\r\n\r\n",
        )
        .await;
        assert!(response.contains("204 No Content"));
        assert!(response.contains("Access-Control-Allow-Origin: *"));
    }

    #[test]
    fn models_response_body_satisfies_both_openai_and_codex_consumers() {
        // Regression: codex 0.132+ parses /models as `ModelsResponse { models }`
        // with each entry strictly-typed (codex-rs/protocol/src/openai_models.rs::
        // ModelInfo). The OpenAI-style `{"object":"list","data":[...]}` shape
        // alone makes codex spam `failed to decode models response: missing
        // field "models"`/`"slug"`/etc. on every interaction. We emit both
        // shapes as twin arrays.
        let body = build_models_response_body(&["composer-2.5".to_string(), "auto".to_string()]);

        // OpenAI side: object="list", data=[{id, object, owned_by}, ...]
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("data must be an array");
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "composer-2.5");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "cursor");

        // Codex side: every field codex considers required (no `#[serde(default)]`,
        // no `Option` with default attribute) must be present. Verify the most
        // load-bearing ones; missing any of these is what produced the
        // production error.
        let models = body["models"].as_array().expect("models must be an array");
        assert_eq!(models.len(), 2);
        let first = &models[0];
        for required in [
            "slug",
            "display_name",
            "description",
            "supported_reasoning_levels",
            "shell_type",
            "visibility",
            "supported_in_api",
            "priority",
            "availability_nux",
            "upgrade",
            "base_instructions",
            "supports_reasoning_summaries",
            "support_verbosity",
            "default_verbosity",
            "apply_patch_tool_type",
            "truncation_policy",
            "supports_parallel_tool_calls",
            "experimental_supported_tools",
        ] {
            assert!(
                first.get(required).is_some(),
                "codex requires field `{required}`; codex models-manager will refuse the response otherwise: {first:?}"
            );
        }
        assert_eq!(first["slug"], "composer-2.5");
        assert_eq!(first["truncation_policy"]["mode"], "tokens");
    }

    #[tokio::test]
    async fn cached_models_serves_from_disk_cache_without_spawning_cursor_agent() {
        // A warm entry under `cursor_models_cache_identity(&key)` (the same key
        // `aivo models cursor` and the picker write) must be returned without
        // shelling out to `cursor-agent`. The test would otherwise hang or
        // error trying to spawn cursor-agent in the sandbox.
        let dir = tempfile::tempdir().unwrap();
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        let key = fake_key();
        let cache_key = cursor_acp::cursor_models_cache_identity(&key);
        cache
            .set(
                &cache_key,
                vec!["composer-2.5".to_string(), "claude-sonnet-4-6".to_string()],
            )
            .await;

        let state = Arc::new(RouterState {
            config: CursorRouterConfig {
                key,
                workspace_cwd: "/tmp".to_string(),
                models_cache: Some(cache),
                prewarm_count: 0,
            },
            // In-memory cache deliberately empty so the lookup falls through
            // to the disk-backed branch.
            cached_models: Mutex::new(None),
            pool: Mutex::new(Vec::new()),
            prewarming: AtomicUsize::new(0),
        });

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            round_trip(
                state,
                "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
            ),
        )
        .await
        .expect("disk-cache path must not spawn cursor-agent");
        assert!(response.contains("200 OK"));
        let body = response.split("\r\n\r\n").nth(1).unwrap();
        let parsed: Value = serde_json::from_str(body).unwrap();
        let data = parsed["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "composer-2.5");
    }

    #[tokio::test]
    async fn acquire_session_slot_grows_pool_when_existing_slot_is_busy() {
        let state = state_with_models(vec![]);
        // Hold slot 0 by acquiring its guard and not dropping it. The next
        // acquire should append a new slot instead of blocking.
        let busy_guard = acquire_session_slot(&state).await;
        assert_eq!(state.pool.lock().await.len(), 1);

        let second = acquire_session_slot(&state).await;
        assert_eq!(state.pool.lock().await.len(), 2);
        drop(second);

        // Re-acquiring without releasing slot 0 should reuse slot 1, not grow.
        let reused = acquire_session_slot(&state).await;
        assert_eq!(state.pool.lock().await.len(), 2);
        drop(reused);
        drop(busy_guard);
    }

    #[tokio::test]
    async fn acquire_session_slot_caps_pool_and_then_waits_on_first() {
        let state = state_with_models(vec![]);
        let mut guards = Vec::new();
        for _ in 0..MAX_POOL_SESSIONS {
            guards.push(acquire_session_slot(&state).await);
        }
        assert_eq!(state.pool.lock().await.len(), MAX_POOL_SESSIONS);

        // Cap reached. A further acquire should block. Race a release of slot 0
        // against the queued acquire and confirm it unblocks (rather than
        // growing past the cap).
        let state_clone = state.clone();
        let pending = tokio::spawn(async move { acquire_session_slot(&state_clone).await });
        // Yield a couple of times to let the spawned task park on the mutex.
        for _ in 0..4 {
            tokio::task::yield_now().await;
        }
        assert!(!pending.is_finished(), "pending acquire must block at cap");
        drop(guards.remove(0));
        let _granted = tokio::time::timeout(std::time::Duration::from_secs(1), pending)
            .await
            .expect("pending acquire should unblock once slot 0 frees")
            .expect("spawn join");
        assert_eq!(
            state.pool.lock().await.len(),
            MAX_POOL_SESSIONS,
            "pool must not grow past cap when waiting on existing slot"
        );
    }

    #[tokio::test]
    async fn head_and_get_root_return_200_for_probes() {
        let state = state_with_models(vec![]);
        for verb in ["HEAD", "GET"] {
            let response = round_trip(
                state.clone(),
                &format!("{verb} / HTTP/1.1\r\nConnection: close\r\n\r\n"),
            )
            .await;
            assert!(
                response.contains("200 OK"),
                "{verb} / must succeed; got: {response}"
            );
        }
    }

    #[tokio::test]
    async fn unknown_path_returns_404_with_hint() {
        let state = state_with_models(vec![]);
        let response = round_trip(state, "GET /nope HTTP/1.1\r\nConnection: close\r\n\r\n").await;
        assert!(response.contains("404"));
        assert!(response.contains("/nope"));
        assert!(response.contains("/v1/responses"));
    }

    #[test]
    fn is_client_disconnect_detects_broken_pipe_through_anyhow_context() {
        let raw = std::io::Error::from(std::io::ErrorKind::BrokenPipe);
        let wrapped: anyhow::Error = anyhow::Error::new(raw).context("cursor-agent session/prompt");
        assert!(is_client_disconnect(&wrapped));
        assert_eq!(status_for_handler_error(&wrapped), 499);
    }

    #[test]
    fn is_client_disconnect_detects_connection_reset_and_aborted() {
        for kind in [
            std::io::ErrorKind::ConnectionReset,
            std::io::ErrorKind::ConnectionAborted,
        ] {
            let err: anyhow::Error = anyhow::Error::new(std::io::Error::from(kind));
            assert!(
                is_client_disconnect(&err),
                "kind {kind:?} should map to 499"
            );
        }
    }

    #[test]
    fn is_client_disconnect_returns_false_for_unrelated_io_errors() {
        let err: anyhow::Error =
            anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
        assert!(!is_client_disconnect(&err));
        assert_eq!(status_for_handler_error(&err), 502);
    }

    #[test]
    fn is_client_disconnect_returns_false_for_non_io_errors() {
        let err = anyhow::anyhow!("upstream rejected request");
        assert!(!is_client_disconnect(&err));
        assert_eq!(status_for_handler_error(&err), 502);
    }

    #[test]
    fn anthropic_thinking_delta_has_thinking_delta_payload() {
        let frame = anthropic_thinking_delta(2, "reasoning text");
        assert!(frame.starts_with("event: content_block_delta\n"));
        let data = frame
            .split_once("data: ")
            .and_then(|(_, s)| s.split("\n\n").next())
            .unwrap();
        let parsed: Value = serde_json::from_str(data).unwrap();
        assert_eq!(parsed["index"], 2);
        assert_eq!(parsed["delta"]["type"], "thinking_delta");
        assert_eq!(parsed["delta"]["thinking"], "reasoning text");
    }

    #[test]
    fn anthropic_text_delta_has_text_delta_payload() {
        let frame = anthropic_text_delta(0, "hello");
        let data = frame
            .split_once("data: ")
            .and_then(|(_, s)| s.split("\n\n").next())
            .unwrap();
        let parsed: Value = serde_json::from_str(data).unwrap();
        assert_eq!(parsed["index"], 0);
        assert_eq!(parsed["delta"]["type"], "text_delta");
        assert_eq!(parsed["delta"]["text"], "hello");
    }

    #[test]
    fn anthropic_content_block_start_emits_correct_inner_shape() {
        let text = anthropic_content_block_start(0, AnthropicBlockKind::Text);
        assert!(text.contains("\"type\":\"text\""));
        assert!(text.contains("\"text\":\"\""));
        let thinking = anthropic_content_block_start(1, AnthropicBlockKind::Thinking);
        assert!(thinking.contains("\"type\":\"thinking\""));
        assert!(thinking.contains("\"thinking\":\"\""));
    }

    #[test]
    fn extract_tool_call_marker_formats_known_event_shapes() {
        let pending = json!({
            "update": {"sessionUpdate": "tool_call", "kind": "search",
                       "status": "pending", "title": "Find sum.js"},
        });
        let s = extract_tool_call_marker(&pending).unwrap();
        assert!(s.contains("pending"));
        assert!(s.contains("Find sum.js"));

        let no_title = json!({
            "update": {"sessionUpdate": "tool_call_update",
                       "kind": "read", "status": "in_progress"},
        });
        let s = extract_tool_call_marker(&no_title).unwrap();
        assert!(s.contains("read"));
        assert!(s.contains("in_progress"));

        // Non-tool events are ignored.
        let irrelevant = json!({
            "update": {"sessionUpdate": "session_info_update"},
        });
        assert!(extract_tool_call_marker(&irrelevant).is_none());
    }

    #[tokio::test]
    async fn acquire_session_slot_waits_for_prewarm_instead_of_expanding() {
        // Regression: when a prewarm task is mid-`CursorAcpSession::open` it
        // holds the slot's lock. The previous `try_lock_owned` path
        // immediately gave up and expanded the pool, spawning a second
        // cursor-agent that the prewarmed one would duplicate. With the fix,
        // an arriving request must wait on the prewarmed slot when
        // `prewarming > 0`, not append a new slot.
        let state = state_with_models(vec![]);
        let slot: SessionSlot = Arc::new(Mutex::new(None));
        state.pool.lock().await.push(slot.clone());
        // Simulate the prewarm task holding the slot's lock and the
        // prewarming counter being non-zero.
        let held = slot.clone().lock_owned().await;
        state.prewarming.fetch_add(1, Ordering::SeqCst);

        let acquire_state = state.clone();
        let mut acquire = Box::pin(async move { acquire_session_slot(&acquire_state).await });

        // While the prewarm slot is held, acquire must NOT complete (it
        // should be parked on the slot's lock) AND must NOT expand the
        // pool.
        let parked = tokio::time::timeout(std::time::Duration::from_millis(50), &mut acquire)
            .await
            .is_err();
        assert!(
            parked,
            "acquire should block, not expand, while prewarm holds the slot"
        );
        assert_eq!(
            state.pool.lock().await.len(),
            1,
            "pool must stay at 1 slot — expanding would orphan a cursor-agent"
        );

        // Release the prewarm's hold; acquire should now resolve with the
        // existing slot.
        state.prewarming.fetch_sub(1, Ordering::SeqCst);
        drop(held);
        let guard = tokio::time::timeout(std::time::Duration::from_secs(1), acquire)
            .await
            .expect("acquire should resolve once prewarm releases");
        assert!(guard.is_none(), "slot's inner session is still empty");
        assert_eq!(state.pool.lock().await.len(), 1);
    }

    #[tokio::test]
    async fn acquire_session_slot_picks_whichever_prewarm_slot_frees_first() {
        // With two prewarmed slots both locked, two concurrent acquirers
        // must each land on a separate slot. Pinning to existing[0] would
        // deadlock the second acquirer if slot[1] freed first.
        let state = state_with_models(vec![]);
        let slot_a: SessionSlot = Arc::new(Mutex::new(None));
        let slot_b: SessionSlot = Arc::new(Mutex::new(None));
        {
            let mut pool = state.pool.lock().await;
            pool.push(slot_a.clone());
            pool.push(slot_b.clone());
        }
        let held_a = slot_a.clone().lock_owned().await;
        let held_b = slot_b.clone().lock_owned().await;
        state.prewarming.fetch_add(2, Ordering::SeqCst);

        let s1 = state.clone();
        let s2 = state.clone();
        let acquire_1 = tokio::spawn(async move { acquire_session_slot(&s1).await });
        let acquire_2 = tokio::spawn(async move { acquire_session_slot(&s2).await });

        // Free slot B first: one acquirer wakes.
        drop(held_b);
        // Then slot A: the other acquirer wakes.
        drop(held_a);
        state.prewarming.store(0, Ordering::SeqCst);

        let r1 = tokio::time::timeout(std::time::Duration::from_secs(1), acquire_1).await;
        let r2 = tokio::time::timeout(std::time::Duration::from_secs(1), acquire_2).await;
        assert!(r1.is_ok() && r2.is_ok(), "both acquirers must resolve");
        assert_eq!(
            state.pool.lock().await.len(),
            2,
            "pool size unchanged — no expansion happened"
        );
    }

    #[tokio::test]
    async fn spawn_prewarm_increments_counter_synchronously() {
        // The counter must reflect pending prewarms BEFORE the spawned
        // tasks run, so an early request that arrives in the gap between
        // `start_background` returning and the prewarm task starting
        // observes `prewarming > 0` and picks the wait-on-slot branch.
        let state = state_with_models(vec![]);
        spawn_prewarm(state.clone(), 2);
        // No `.await` since the increment is synchronous.
        assert_eq!(state.prewarming.load(Ordering::SeqCst), 2);
        // Drain any background work and let the PrewarmCounterGuard drop
        // so we don't leave state behind for sibling tests.
        for _ in 0..32 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn spawn_prewarm_reserves_requested_number_of_slots() {
        // Open `count` slot Arcs in the pool before any HTTP request lands,
        // so two paired requests both find a slot already in the pool (even
        // if the underlying CursorAcpSession::open is still in flight). We
        // can't run the real open() in unit tests — it would spawn a real
        // cursor-agent binary — so the assertion is about pool *shape*:
        // requesting 2 prewarms must yield at least 2 slot Arcs.
        let state = state_with_models(vec![]);
        spawn_prewarm(state.clone(), 2);
        // Yield enough that the prewarm tasks have time to register their
        // slots in the outer pool Mutex (the slot reservation is the very
        // first await they hit).
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
        let n = state.pool.lock().await.len();
        assert!(
            n >= 2,
            "prewarm should reserve >= 2 slot Arcs in the pool, saw {n}"
        );
    }

    #[test]
    fn anthropic_block_state_defaults_to_no_block_and_index_zero() {
        let s = AnthropicBlockState::default();
        assert!(s.current.is_none());
        assert_eq!(s.next_index, 0);
        // Without a current block the helper exposes index 0 as a safe
        // default so deltas have something to address before the first
        // ensure_kind. The dispatcher always calls ensure_kind first.
        assert_eq!(s.index(), 0);
    }

    #[test]
    fn sse_keepalive_uses_comment_prefix_per_spec() {
        // The SSE spec says lines starting with `:` are comments — clients
        // ignore them but their idle-timer logic still counts them as
        // "the connection is alive". Two trailing newlines terminate the
        // message frame.
        assert!(SSE_KEEPALIVE.starts_with(':'));
        assert!(SSE_KEEPALIVE.ends_with("\n\n"));
        assert!(!SSE_KEEPALIVE.contains("data:"));
    }

    #[test]
    fn strip_v1_prefix_collapses_only_real_v1_segments() {
        assert_eq!(strip_v1_prefix("/v1/chat/completions"), "/chat/completions");
        assert_eq!(strip_v1_prefix("/v1/models"), "/models");
        assert_eq!(strip_v1_prefix("/v1/"), "/");
        // Bare `/v1` collapses to empty so the root probe handler catches it.
        assert_eq!(strip_v1_prefix("/v1"), "");
        // Don't accidentally re-interpret `/v1bogus` as `/bogus`.
        assert_eq!(strip_v1_prefix("/v1bogus"), "/v1bogus");
        // Anything without the prefix passes through unchanged.
        assert_eq!(strip_v1_prefix("/chat/completions"), "/chat/completions");
        assert_eq!(strip_v1_prefix("/"), "/");
        assert_eq!(strip_v1_prefix(""), "");
    }

    #[tokio::test]
    async fn unversioned_post_paths_route_to_the_same_handlers() {
        // The handlers themselves need an ACP session, which we can't spawn
        // in unit tests. Assert that the *router* dispatches the unversioned
        // form to the right handler by checking the body parsing path: a body
        // missing `messages` should yield "messages array is required" (the
        // OpenAI handler's error) for both /v1/chat/completions and
        // /chat/completions, *not* the router 404 message.
        let state = state_with_models(vec![]);
        let body = r#"{"model": "composer-2.5"}"#;
        for path in [
            "/v1/chat/completions",
            "/chat/completions",
            "/v1/messages",
            "/messages",
            "/v1/responses",
            "/responses",
        ] {
            let request = format!(
                "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
                len = body.len()
            );
            let response = round_trip(state.clone(), &request).await;
            assert!(
                !response.contains("cursor_model_router_not_found"),
                "{path} unexpectedly routed to the 404 handler: {response}"
            );
        }
    }

    #[test]
    fn reduce_openai_request_joins_role_labels_and_renders_tool_results() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": ""},
                {"role": "tool", "name": "read_file", "content": "file body"},
                {"role": "user", "content": "Explain X."},
            ],
        });
        let prompt = reduce_openai_request_to_prompt(&body);
        assert_eq!(
            prompt,
            "System: You are helpful.\n\nUser: Hi\n\n[Tool result for read_file]\nfile body\n\nUser: Explain X."
        );
    }

    #[test]
    fn reduce_openai_request_emits_available_tools_header_and_tool_calls() {
        let body = json!({
            "tools": [
                {"type": "function", "function": {
                    "name": "read_file",
                    "description": "Read a file from disk.",
                }},
                {"type": "function", "function": {
                    "name": "write_file",
                    "description": "  Write\n  content  to disk.",
                }},
            ],
            "messages": [
                {"role": "user", "content": "Update the readme."},
                {"role": "assistant", "content": "Reading first.", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"README.md\"}",
                    }},
                ]},
                {"role": "tool", "name": "read_file", "content": "# Title"},
            ],
        });
        let prompt = reduce_openai_request_to_prompt(&body);
        assert!(prompt.starts_with(
            "Available tools:\n- read_file: Read a file from disk.\n- write_file: Write content to disk."
        ));
        assert!(prompt.contains("Assistant: Reading first."));
        assert!(prompt.contains("[Tool call] read_file({\"path\":\"README.md\"})"));
        assert!(prompt.contains("[Tool result for read_file]\n# Title"));
    }

    #[test]
    fn reduce_openai_request_extracts_text_from_content_arrays() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                    {"type": "text", "text": "Now explain."},
                ]},
            ],
        });
        let prompt = reduce_openai_request_to_prompt(&body);
        assert_eq!(
            prompt,
            "User: Look at this:\n[image attachment omitted]\nNow explain."
        );
    }

    #[test]
    fn extract_image_blocks_inline_base64_each_protocol() {
        let openai = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
                ]},
            ],
        });
        assert_eq!(
            extract_openai_image_blocks(&openai).unwrap(),
            vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
        );

        let responses = json!({
            "input": [
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/jpeg;base64,QUJD"},
                ]},
            ],
        });
        assert_eq!(
            extract_responses_image_blocks(&responses).unwrap(),
            vec![json!({"type": "image", "mimeType": "image/jpeg", "data": "QUJD"})]
        );

        let anthropic = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "see"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA="}},
                ]},
            ],
        });
        assert_eq!(
            extract_anthropic_image_blocks(&anthropic).unwrap(),
            vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
        );

        let gemini = json!({
            "contents": [
                {"role": "user", "parts": [
                    {"text": "look"},
                    {"inlineData": {"mimeType": "image/jpeg", "data": "ZZZ="}},
                ]},
            ],
        });
        assert_eq!(
            extract_gemini_image_blocks(&gemini).unwrap(),
            vec![json!({"type": "image", "mimeType": "image/jpeg", "data": "ZZZ="})]
        );
    }

    #[test]
    fn extract_image_blocks_rejects_remote_urls() {
        // Remote URLs would need fetching — refuse rather than silently drop
        // or call out to the network on behalf of the launched tool.
        let openai = json!({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ]}],
        });
        assert!(extract_openai_image_blocks(&openai).is_err());

        let anthropic = json!({
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
            ]}],
        });
        assert!(extract_anthropic_image_blocks(&anthropic).is_err());

        let gemini = json!({
            "contents": [{"role": "user", "parts": [
                {"fileData": {"mimeType": "image/png", "fileUri": "gs://x/y"}},
            ]}],
        });
        assert!(extract_gemini_image_blocks(&gemini).is_err());
    }

    #[test]
    fn extract_image_blocks_empty_for_text_only_bodies() {
        assert!(
            extract_openai_image_blocks(&json!({
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap()
            .is_empty()
        );
        assert!(
            extract_anthropic_image_blocks(&json!({
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            }))
            .unwrap()
            .is_empty()
        );
        // Non-image inline data (PDF) must not show up as an image block —
        // images are gated on mime starting with `image/`.
        assert!(
            extract_gemini_image_blocks(&json!({
                "contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": "application/pdf", "data": "AAA="}},
                ]}],
            }))
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn parse_data_url_recognizes_base64_and_skips_other_forms() {
        assert_eq!(
            parse_data_url("data:image/png;base64,AAA="),
            Some(("image/png".to_string(), "AAA=".to_string()))
        );
        // Case-insensitive token.
        assert_eq!(
            parse_data_url("data:image/jpeg;BASE64,ZZZ"),
            Some(("image/jpeg".to_string(), "ZZZ".to_string()))
        );
        // Charset param before base64.
        assert_eq!(
            parse_data_url("data:image/svg+xml;charset=utf-8;base64,QUJD"),
            Some(("image/svg+xml".to_string(), "QUJD".to_string()))
        );
        // URL-encoded text (no base64 token) — refused.
        assert_eq!(parse_data_url("data:text/plain,hello"), None);
        // Not a data URL at all.
        assert_eq!(parse_data_url("https://example.com/x.png"), None);
    }

    #[test]
    fn openai_chunk_frame_shapes_choices_and_delta() {
        let frame = openai_chunk_frame(
            "chatcmpl-test",
            1234567890,
            "composer-2.5",
            json!({"content": "hello"}),
            None,
        );
        assert!(frame.starts_with("data: "));
        assert!(frame.ends_with("\n\n"));
        let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert_eq!(parsed["model"], "composer-2.5");
        assert_eq!(parsed["choices"][0]["delta"]["content"], "hello");
        assert!(parsed["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn openai_chunk_frame_emits_finish_reason_when_provided() {
        let frame = openai_chunk_frame(
            "chatcmpl-test",
            1234567890,
            "composer-2.5",
            json!({}),
            Some("stop"),
        );
        let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn openai_completion_body_surfaces_content_and_reasoning() {
        let turn = AggregatedTurn {
            content: "answer".to_string(),
            reasoning: "thinking".to_string(),
        };
        let body = openai_completion_body(&turn, "composer-2.5", 42);
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["content"], "answer");
        assert_eq!(
            body["choices"][0]["message"]["reasoning_content"],
            "thinking"
        );
        assert_eq!(body["usage"]["prompt_tokens"], 42);
        // "answer" is 6 chars → (6+3)/4 = 2 tokens.
        assert_eq!(body["usage"]["completion_tokens"], 2);
        assert_eq!(body["usage"]["total_tokens"], 44);
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn extract_agent_text_picks_only_agent_message_chunks() {
        let msg = json!({
            "sessionId": "s",
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hi"}},
        });
        assert_eq!(extract_agent_text(&msg), Some("Hi"));

        let thought = json!({
            "sessionId": "s",
            "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "..."}},
        });
        assert_eq!(extract_agent_text(&thought), None);
        assert_eq!(extract_agent_thought(&thought), Some("..."));
    }

    #[test]
    fn reduce_anthropic_request_preserves_tool_use_and_tool_result_blocks() {
        let body = json!({
            "model": "claude-sonnet-4-6",
            "system": "Be terse.",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Hello!"},
                    {"type": "tool_use", "id": "t", "name": "lookup", "input": {"q": "x"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": [
                        {"type": "text", "text": "result body"},
                    ]},
                    {"type": "text", "text": "What's 2+2?"},
                ]},
            ],
        });
        let prompt = reduce_anthropic_request_to_prompt(&body);
        assert_eq!(
            prompt,
            "System: Be terse.\n\nUser: Hi\n\nAssistant: Hello!\n\n[Tool call] lookup({\"q\":\"x\"})\n\n[Tool result for t]\nresult body\n\nUser: What's 2+2?"
        );
    }

    #[test]
    fn reduce_anthropic_request_emits_available_tools_header_and_system_blocks() {
        let body = json!({
            "system": [
                {"type": "text", "text": "Always be terse."},
                {"type": "text", "text": "Cite sources."},
            ],
            "tools": [
                {"name": "search", "description": "Search the web.", "input_schema": {}},
                {"name": "edit", "description": "Edit a file."},
            ],
            "messages": [
                {"role": "user", "content": "Find something."},
            ],
        });
        let prompt = reduce_anthropic_request_to_prompt(&body);
        assert!(prompt.starts_with(
            "Available tools:\n- search: Search the web.\n- edit: Edit a file.\n\nSystem: Always be terse.\nCite sources."
        ));
        assert!(prompt.ends_with("User: Find something."));
    }

    #[test]
    fn reduce_anthropic_request_marks_image_attachments() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "See this:"},
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                    {"type": "text", "text": "Thoughts?"},
                ]},
            ],
        });
        let prompt = reduce_anthropic_request_to_prompt(&body);
        assert_eq!(
            prompt,
            "User: See this:\n\n[image attachment omitted]\n\nUser: Thoughts?"
        );
    }

    #[test]
    fn reduce_anthropic_request_handles_string_content() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Just a string"},
            ],
        });
        assert_eq!(
            reduce_anthropic_request_to_prompt(&body),
            "User: Just a string"
        );
    }

    #[test]
    fn anthropic_event_uses_sse_event_data_pair() {
        let frame = sse_named_event(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
        );
        assert!(frame.starts_with("event: content_block_delta\n"));
        assert!(frame.contains("data: "));
        assert!(frame.ends_with("\n\n"));
    }

    #[test]
    fn anthropic_message_body_emits_text_content_block_and_usage() {
        let turn = AggregatedTurn {
            content: "hello world".to_string(),
            reasoning: String::new(),
        };
        let body = anthropic_message_body(&turn, "claude-sonnet-4-6", 100);
        assert_eq!(body["type"], "message");
        assert_eq!(body["role"], "assistant");
        assert_eq!(body["model"], "claude-sonnet-4-6");
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["content"][0]["type"], "text");
        assert_eq!(body["content"][0]["text"], "hello world");
        assert_eq!(body["usage"]["input_tokens"], 100);
        // "hello world" is 11 chars → (11+3)/4 = 3 tokens.
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
        assert_eq!(body["usage"]["cache_read_input_tokens"], 0);
    }

    #[test]
    fn reduce_responses_request_accepts_string_input() {
        let body = json!({
            "model": "gpt-5",
            "instructions": "Be terse.",
            "input": "Hi there",
        });
        assert_eq!(
            reduce_responses_request_to_prompt(&body),
            "System: Be terse.\n\nUser: Hi there"
        );
    }

    #[test]
    fn reduce_responses_request_handles_array_input_with_typed_blocks() {
        let body = json!({
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "Be helpful."}]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "Question A."},
                    {"type": "input_image", "image_url": "..."},
                ]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Sure."}]},
            ],
        });
        assert_eq!(
            reduce_responses_request_to_prompt(&body),
            "System: Be helpful.\n\nUser: Question A.\n\nAssistant: Sure."
        );
    }

    #[test]
    fn reduce_responses_request_renders_tool_schemas_and_function_calls() {
        let body = json!({
            "tools": [
                {"type": "function", "name": "shell", "description": "Run a shell command."},
                {"type": "function", "function": {
                    "name": "patch",
                    "description": "Apply a unified diff.",
                }},
            ],
            "instructions": "Be terse.",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "Edit README."},
                ]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"type": "function_call", "name": "shell", "call_id": "c1",
                 "arguments": "{\"command\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "README.md\n"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "Done."},
                ]},
            ],
        });
        let prompt = reduce_responses_request_to_prompt(&body);
        assert!(prompt.starts_with(
            "Available tools:\n- shell: Run a shell command.\n- patch: Apply a unified diff."
        ));
        assert!(prompt.contains("System: Be terse."));
        assert!(prompt.contains("User: Edit README."));
        assert!(!prompt.contains("thinking"));
        assert!(prompt.contains("[Tool call] shell({\"command\":\"ls\"})"));
        assert!(prompt.contains("[Tool result for c1]\nREADME.md"));
        assert!(prompt.ends_with("Assistant: Done."));
    }

    #[test]
    fn tool_schema_line_truncates_long_descriptions() {
        let desc = "x".repeat(400);
        let line = format_tool_schema_line("tool", &desc);
        assert!(line.starts_with("- tool: "));
        assert!(line.ends_with("…[truncated]"));
        assert!(line.chars().count() < desc.len());
    }

    #[test]
    fn tool_result_block_truncates_huge_outputs() {
        let huge = "x".repeat(10_000);
        let block = format_tool_result_block("read_file", &huge);
        assert!(block.starts_with("[Tool result for read_file]\n"));
        assert!(block.ends_with("…[truncated]"));
        let body_chars = block.chars().count();
        assert!(body_chars < huge.len());
    }

    #[test]
    fn parse_request_headers_lowercases_names_and_skips_request_line() {
        let req = "POST /v1/chat/completions HTTP/1.1\r\n\
                   Host: localhost\r\n\
                   Content-Type: application/json\r\n\
                   X-Custom-Header: value\r\n\
                   \r\n\
                   {}";
        let headers = parse_request_headers(req);
        assert_eq!(headers.get("host"), Some(&"localhost".to_string()));
        assert_eq!(
            headers.get("content-type"),
            Some(&"application/json".to_string())
        );
        assert_eq!(headers.get("x-custom-header"), Some(&"value".to_string()));
        // The request line should never show up as a header.
        assert!(!headers.contains_key("post /v1/chat/completions http/1.1"));
    }

    #[test]
    fn truncate_for_log_marks_overflow() {
        let s = "x".repeat(100);
        let out = truncate_for_log(&s, 50);
        assert!(out.starts_with(&"x".repeat(50)));
        assert!(out.ends_with("…[truncated]"));
        // Under-cap inputs are returned verbatim.
        assert_eq!(truncate_for_log("short", 50), "short");
    }

    #[test]
    fn responses_completion_body_wraps_output_text_block_and_usage() {
        let turn = AggregatedTurn {
            content: "answer text".to_string(),
            reasoning: String::new(),
        };
        let body = responses_completion_body(&turn, "gpt-5", 80);
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["type"], "message");
        assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(body["output"][0]["content"][0]["text"], "answer text");
        assert_eq!(body["usage"]["input_tokens"], 80);
        // "answer text" is 11 chars → (11+3)/4 = 3 tokens.
        assert_eq!(body["usage"]["output_tokens"], 3);
        assert_eq!(body["usage"]["total_tokens"], 83);
    }

    #[test]
    fn title_gen_detector_matches_claude_code_signature() {
        // Real prompt observed in production logs.
        let body = json!({
            "model": "claude-haiku-4-5",
            "system": "You are Claude Code, Anthropic's official CLI for Claude.\n\
                Generate a concise, sentence-case title (3-7 words) that \
                captures the main topic or goal of this coding session. \
                Return JSON with a single \"title\" field.",
            "messages": [{"role": "user", "content": "fix the login bug"}],
        });
        assert!(is_title_generation_request(&body));
    }

    #[test]
    fn title_gen_detector_rejects_regular_coding_prompts() {
        let body = json!({
            "system": "You are Claude Code, a coding assistant. Help the user fix bugs.",
            "messages": [{"role": "user", "content": "What's broken?"}],
        });
        assert!(!is_title_generation_request(&body));
    }

    #[test]
    fn title_gen_detector_handles_array_system_blocks() {
        let body = json!({
            "system": [
                {"type": "text", "text": "You are Claude Code."},
                {"type": "text", "text": "Generate a concise, sentence-case title for the session. Return JSON with a single 'title' field."},
            ],
            "messages": [],
        });
        assert!(is_title_generation_request(&body));
    }

    #[test]
    fn build_title_uses_first_user_message_and_truncates() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Help me fix the login button on the mobile homepage"},
                {"role": "assistant", "content": "Sure, let's look."},
            ],
        });
        let title = build_title_from_anthropic_body(&body);
        // Capped at 7 words.
        assert_eq!(title.split_whitespace().count(), 7);
        assert!(title.starts_with("Help me fix the login button"));
    }

    #[test]
    fn build_title_falls_back_to_coding_session_when_empty() {
        assert_eq!(
            build_title_from_anthropic_body(&json!({"messages": []})),
            "Coding session"
        );
        assert_eq!(
            build_title_from_anthropic_body(&json!({})),
            "Coding session"
        );
    }

    #[test]
    fn compose_short_title_breaks_on_word_boundary_when_long() {
        let raw = "thisisaverylongwordthatshouldgetbrokenat the boundary somewhere along the way";
        let title = compose_short_title(raw);
        assert!(title.chars().count() <= 60);
        assert!(!title.ends_with(' '));
    }

    #[tokio::test]
    async fn title_gen_short_circuit_returns_json_title_without_cursor() {
        // Bind a router with no cursor session at all — if the short-circuit
        // works, the response will arrive without acquire_session_slot ever
        // hitting CursorAcpSession::open (which would hang here since there's
        // no real cursor-agent binary in tests).
        let state = state_with_models(vec![]);
        let body = json!({
            "model": "claude-haiku-4-5",
            "stream": false,
            "system": "You are Claude Code, Anthropic's official CLI for Claude. \
                Generate a concise, sentence-case title (3-7 words) that captures \
                the main topic. Return JSON with a single \"title\" field.",
            "messages": [{"role": "user", "content": "investigate the auth race condition"}],
        });
        let body_bytes = body.to_string();
        let request = format!(
            "POST /v1/messages HTTP/1.1\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {len}\r\n\
             Connection: close\r\n\r\n{body_bytes}",
            len = body_bytes.len()
        );
        // round_trip times out at the OS level if no response arrives. Wrap in
        // a tokio::time::timeout so a hang surfaces as a test failure instead
        // of a 60 s wait.
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            round_trip(state, &request),
        )
        .await
        .expect("title-gen must short-circuit without opening a cursor session");
        assert!(response.contains("200 OK"));
        let body = response.split("\r\n\r\n").nth(1).unwrap();
        let parsed: Value = serde_json::from_str(body).unwrap();
        // Content is itself JSON: {"title":"..."}.
        let text = parsed["content"][0]["text"].as_str().unwrap();
        let inner: Value = serde_json::from_str(text).unwrap();
        assert!(
            inner["title"]
                .as_str()
                .is_some_and(|t| t.starts_with("investigate the auth"))
        );
    }

    #[test]
    fn estimate_tokens_rounds_up_and_handles_empty() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("a"), 1);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens(&"x".repeat(100)), 25);
    }

    #[test]
    fn openai_usage_chunk_includes_prompt_and_completion_tokens() {
        let frame = openai_usage_chunk("chatcmpl-test", 42, "composer-2.5", 120, 30);
        let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed["object"], "chat.completion.chunk");
        assert!(parsed["choices"].as_array().unwrap().is_empty());
        assert_eq!(parsed["usage"]["prompt_tokens"], 120);
        assert_eq!(parsed["usage"]["completion_tokens"], 30);
        assert_eq!(parsed["usage"]["total_tokens"], 150);
    }

    #[test]
    fn parse_gemini_generate_path_accepts_known_prefixes_and_actions() {
        assert_eq!(
            parse_gemini_generate_path("/v1beta/models/gemini-2.5-pro:generateContent"),
            Some(GeminiGenerate {
                model: "gemini-2.5-pro".to_string(),
                stream: false,
            })
        );
        assert_eq!(
            parse_gemini_generate_path("/v1beta/models/gemini-2.5-pro:streamGenerateContent"),
            Some(GeminiGenerate {
                model: "gemini-2.5-pro".to_string(),
                stream: true,
            })
        );
        assert_eq!(
            parse_gemini_generate_path("/v1/models/composer-2.5:generateContent"),
            Some(GeminiGenerate {
                model: "composer-2.5".to_string(),
                stream: false,
            })
        );
        assert_eq!(
            parse_gemini_generate_path("/models/composer-2.5:generateContent"),
            Some(GeminiGenerate {
                model: "composer-2.5".to_string(),
                stream: false,
            })
        );
        // Wrong action verb falls through so the unknown-path 404 fires.
        assert_eq!(
            parse_gemini_generate_path("/v1beta/models/x:countTokens"),
            None
        );
        // Non-Gemini paths are not falsely matched.
        assert_eq!(parse_gemini_generate_path("/v1/messages"), None);
        assert_eq!(parse_gemini_generate_path("/v1/chat/completions"), None);
    }

    #[test]
    fn reduce_gemini_request_renders_system_user_and_tool_history() {
        let body = json!({
            "systemInstruction": {"parts": [{"text": "Be terse."}]},
            "tools": [{"functionDeclarations": [
                {"name": "read_file", "description": "Read a file."},
            ]}],
            "contents": [
                {"role": "user", "parts": [{"text": "Look at config.toml."}]},
                {"role": "model", "parts": [
                    {"text": "Reading."},
                    {"functionCall": {"name": "read_file", "args": {"path": "config.toml"}}},
                ]},
                {"role": "user", "parts": [{"functionResponse": {
                    "name": "read_file",
                    "response": {"output": "# title"},
                }}]},
            ],
        });
        let prompt = reduce_gemini_request_to_prompt(&body);
        assert!(prompt.starts_with("Available tools:\n- read_file: Read a file."));
        assert!(prompt.contains("System: Be terse."));
        assert!(prompt.contains("User: Look at config.toml."));
        assert!(prompt.contains("Assistant: Reading."));
        assert!(prompt.contains("[Tool call] read_file"));
        assert!(prompt.contains("[Tool result for read_file]"));
    }

    #[test]
    fn gemini_response_body_shapes_candidates_and_usage() {
        let turn = AggregatedTurn {
            content: "hello".to_string(),
            reasoning: String::new(),
        };
        let body = gemini_response_body(&turn, "composer-2.5", 12);
        let candidate = &body["candidates"][0];
        assert_eq!(candidate["content"]["role"], "model");
        assert_eq!(candidate["content"]["parts"][0]["text"], "hello");
        assert_eq!(candidate["finishReason"], "STOP");
        assert_eq!(body["usageMetadata"]["promptTokenCount"], 12);
        assert_eq!(body["modelVersion"], "composer-2.5");
    }

    #[test]
    fn gemini_stream_frames_have_no_done_marker_and_carry_usage_on_final() {
        let text_frame = gemini_stream_text_frame("composer-2.5", "delta");
        assert!(text_frame.starts_with("data: "));
        let parsed: Value = serde_json::from_str(
            text_frame
                .trim_start_matches("data: ")
                .trim_end_matches("\n\n"),
        )
        .unwrap();
        assert_eq!(
            parsed["candidates"][0]["content"]["parts"][0]["text"],
            "delta"
        );
        // Intermediate frames carry no finishReason — clients buffer text until
        // the final frame announces STOP.
        assert!(parsed["candidates"][0]["finishReason"].is_null());

        let final_frame = gemini_stream_final_frame("composer-2.5", "STOP", 100, 50, 150);
        let parsed: Value = serde_json::from_str(
            final_frame
                .trim_start_matches("data: ")
                .trim_end_matches("\n\n"),
        )
        .unwrap();
        assert_eq!(parsed["candidates"][0]["finishReason"], "STOP");
        assert_eq!(parsed["usageMetadata"]["totalTokenCount"], 150);
    }

    #[tokio::test]
    async fn gemini_generate_path_routes_to_gemini_handler() {
        // Mirrors `unversioned_post_paths_route_to_the_same_handlers`: a body
        // missing `contents` triggers the handler's "reduced prompt is empty"
        // error, which is the proof the dispatcher reached the Gemini handler
        // (and not the catch-all 404).
        let state = state_with_models(vec![]);
        for path in [
            "/v1beta/models/gemini-2.5-pro:generateContent",
            "/v1beta/models/gemini-2.5-pro:streamGenerateContent",
            "/v1/models/composer-2.5:generateContent",
        ] {
            let body = r#"{"contents": []}"#;
            let request = format!(
                "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
                len = body.len()
            );
            let response = round_trip(state.clone(), &request).await;
            assert!(
                !response.contains("cursor_model_router_not_found"),
                "{path} unexpectedly routed to the 404 handler: {response}"
            );
            assert!(
                response.contains("reduced prompt is empty"),
                "{path} should have reached the Gemini handler's empty-prompt error: {response}"
            );
        }
    }
}
