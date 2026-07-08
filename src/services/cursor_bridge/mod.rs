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
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

use self::mcp::{BridgeSession, McpBridge, ToolUseIdStyle};
use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CURSOR_ACP_SENTINEL, CursorAcpSession};
use crate::services::http_debug::{self, DebugEntry, Phase, redact_headers};
use crate::services::http_utils::{
    self, bind_local_listener, cors_header_block, extract_request_body, extract_request_path,
    format_http_chunk, http_response_head_with_extra,
};
use crate::services::model_list_response;

mod anthropic;
mod gemini;
pub mod mcp;
mod openai_chat;
mod responses;

use self::anthropic::handle_anthropic_messages;
use self::gemini::{handle_gemini_generate, parse_gemini_generate_path};
use self::openai_chat::handle_openai_chat;
use self::responses::handle_responses;

#[cfg(test)]
mod tests;
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
    /// When `Some`, pre-open one ACP session with the MCP bridge attached up
    /// front so the first bridged turn doesn't pay cursor-agent's ~16 s
    /// `session/new` cost. The id-style must match the protocol the launched
    /// tool uses (e.g. `OpenAi` for codex's `/responses`); a prewarmed slot
    /// is only consumed by the matching `run_*_bridged_fresh` path. `None`
    /// disables the optimization. Distinct from `prewarm_count`, which
    /// pre-opens MCP-less pool sessions for non-tool turns.
    pub mcp_prewarm_id_style: Option<ToolUseIdStyle>,
    /// When `Some`, every request (except the CORS preflight and root probe)
    /// must carry `Authorization: Bearer <t>` or `x-api-key: <t>`. Native-tool
    /// launches leave this `None` (they reach the router over trusted local env
    /// with the `aivo-cursor` placeholder); the plugin endpoint sets it so the
    /// loopback proxy is bearer-gated like the plain-key `ServeRouter`.
    pub expected_token: Option<String>,
}

pub struct CursorModelRouter {
    config: CursorRouterConfig,
}

/// Soft cap on concurrent ACP sessions; cursor-agent serializes prompts
/// per session, so paired turns (main + subagent) need separate slots.
pub(crate) const MAX_POOL_SESSIONS: usize = 3;

/// One lazily-opened ACP session. Held inside `Arc<Mutex<...>>` so HTTP
/// handlers can take an owned lock for the full turn duration; the outer pool
/// just tracks the slot Arcs.
pub(crate) type SessionSlot = Arc<Mutex<Option<CursorAcpSession>>>;

pub(crate) struct RouterState {
    config: CursorRouterConfig,
    cached_models: Mutex<Option<Vec<String>>>,
    /// HTTP MCP server that surfaces claude-cli's `/v1/messages` tools to
    /// cursor-agent's model. Shared across all Anthropic-shaped turns;
    /// other protocols (OpenAI, Responses, Gemini) ignore it because their
    /// tool schemas already flatten cleanly into the text prompt.
    mcp_bridge: Arc<McpBridge>,
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
    /// Prewarm slot for the bridged-fresh path. Holds at most one ready
    /// (BridgeSession, CursorAcpSession) pair plus an `in_flight` flag that
    /// callers wait on so a `/responses` arriving 4 s after router boot
    /// blocks on the in-progress prewarm instead of opening a duplicate
    /// cursor-agent. Behind `Arc` so refill tasks scheduled from
    /// `&RouterState` can hold the slot without threading
    /// `Arc<RouterState>` through every call site.
    mcp_prewarmed: Arc<Mutex<McpPrewarmSlot>>,
}

pub(crate) struct McpPrewarmedSession {
    bridge_session: Arc<Mutex<BridgeSession>>,
    acp: CursorAcpSession,
}

pub(crate) struct McpPrewarmSlot {
    /// The ready prewarmed pair, if a prewarm has completed and nothing has
    /// consumed it yet.
    ready: Option<McpPrewarmedSession>,
    /// True between the moment a prewarm task is scheduled and the moment
    /// it completes (success OR failure). Set under-lock by the scheduler
    /// (start_background, take_mcp_prewarmed) so a concurrent take sees
    /// in_flight=true even before the task's first lock acquisition.
    in_flight: bool,
    /// Fired by a finishing prewarm task. `take_mcp_prewarmed` subscribes
    /// before re-checking the slot so a notify that races the check still
    /// wakes the waiter.
    completed: Arc<Notify>,
}

impl McpPrewarmSlot {
    fn new() -> Self {
        Self {
            ready: None,
            in_flight: false,
            completed: Arc::new(Notify::new()),
        }
    }
}

/// Maximum time `take_mcp_prewarmed` waits on an in-flight prewarm before
/// giving up and cold-pathing. Observed prewarm latency is ~17-18 s
/// (cursor-agent `session/new`); 25 s leaves a margin without blocking the
/// user indefinitely when the prewarm task wedges.
pub(crate) const MCP_PREWARM_WAIT_TIMEOUT: Duration = Duration::from_secs(25);

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
        let mcp_bridge = McpBridge::start_background().await?;
        let prewarm = self.config.prewarm_count;
        let mcp_prewarmed = Arc::new(Mutex::new(McpPrewarmSlot::new()));
        let state = Arc::new(RouterState {
            config: self.config.clone(),
            cached_models: Mutex::new(None),
            mcp_bridge: mcp_bridge.clone(),
            pool: Mutex::new(Vec::new()),
            prewarming: AtomicUsize::new(0),
            mcp_prewarmed: mcp_prewarmed.clone(),
        });
        spawn_prewarm(state.clone(), prewarm);
        if let Some(id_style) = self.config.mcp_prewarm_id_style {
            // Mark in_flight=true under-lock BEFORE the background task
            // starts so a request arriving in the first few seconds sees
            // the prewarm and waits instead of cold-pathing.
            mcp_prewarmed.lock().await.in_flight = true;
            spawn_mcp_prewarm_task(
                self.config.key.clone(),
                self.config.workspace_cwd.clone(),
                mcp_bridge,
                mcp_prewarmed,
                id_style,
            );
        }
        let handle = tokio::spawn(run_router(listener, state));
        Ok((port, handle))
    }
}

/// Background task that does the actual prewarm work. The caller must set
/// `slot.in_flight = true` under the slot mutex BEFORE invoking this — that
/// way a request arriving between the schedule and the task's first lock
/// acquisition still observes `in_flight=true` and waits. The task itself
/// only clears `in_flight` (and notifies waiters) on completion, success or
/// failure. The first action the task takes is the long-running
/// `open_session_for_prewarm` + `CursorAcpSession::open_with_mcp`.
pub(crate) fn spawn_mcp_prewarm_task(
    key: ApiKey,
    workspace_cwd: String,
    mcp_bridge: Arc<McpBridge>,
    slot: Arc<Mutex<McpPrewarmSlot>>,
    id_style: ToolUseIdStyle,
) {
    tokio::spawn(async move {
        let (bridge_session, mcp_url) = mcp_bridge.open_session_for_prewarm(id_style).await;
        let bridge_id = bridge_session.lock().await.id.clone();
        let result =
            CursorAcpSession::open_with_mcp(&key, None, &workspace_cwd, Some(&mcp_url)).await;
        let notify = {
            let mut guard = slot.lock().await;
            match result {
                Ok(acp) => {
                    guard.ready = Some(McpPrewarmedSession {
                        bridge_session,
                        acp,
                    });
                }
                Err(e) => {
                    eprintln!("aivo: mcp prewarm failed: {e:#}");
                    // Drop the orphaned bridge session before clearing the
                    // flag so a concurrent take that wakes on the notify
                    // doesn't race with the cleanup.
                    drop(guard);
                    mcp_bridge.drop_session(&bridge_id).await;
                    let mut guard = slot.lock().await;
                    guard.in_flight = false;
                    let notify = guard.completed.clone();
                    drop(guard);
                    notify.notify_waiters();
                    return;
                }
            }
            guard.in_flight = false;
            guard.completed.clone()
        };
        notify.notify_waiters();
    });
}

pub(crate) enum TakeOutcome {
    Got(McpPrewarmedSession),
    Wait,
    Cold,
}

/// Try to consume the prewarmed slot. Returns `Some(slot)` when one is
/// ready (or becomes ready while waiting); the slot is removed atomically.
/// On a hit, schedules a refill so subsequent bridged-fresh turns also
/// avoid the cold `session/new`. If the slot is empty but a prewarm is
/// in flight, blocks on [`MCP_PREWARM_WAIT_TIMEOUT`] for that prewarm to
/// finish rather than racing it with a duplicate cold session.
pub(crate) async fn take_mcp_prewarmed(state: &RouterState) -> Option<McpPrewarmedSession> {
    loop {
        // Subscribe to the completion notify BEFORE checking the flags so a
        // notify_waiters that fires between the check and the await still
        // wakes us — the classic tokio::sync::Notify subscribe-then-recheck
        // pattern.
        let notify = state.mcp_prewarmed.lock().await.completed.clone();
        let notified = notify.notified();
        tokio::pin!(notified);
        // Enroll the waker before the recheck — Notified only registers on
        // first poll, so notify_waiters() fired in the gap would be lost.
        notified.as_mut().enable();

        let outcome = {
            let mut guard = state.mcp_prewarmed.lock().await;
            if let Some(slot) = guard.ready.take() {
                // Refilling: set in_flight under-lock so a concurrent take
                // sees an in-flight refill rather than (None, !in_flight).
                if state.config.mcp_prewarm_id_style.is_some() {
                    guard.in_flight = true;
                }
                TakeOutcome::Got(slot)
            } else if guard.in_flight {
                TakeOutcome::Wait
            } else {
                TakeOutcome::Cold
            }
        };

        match outcome {
            TakeOutcome::Got(slot) => {
                if let Some(id_style) = state.config.mcp_prewarm_id_style {
                    spawn_mcp_prewarm_task(
                        state.config.key.clone(),
                        state.config.workspace_cwd.clone(),
                        state.mcp_bridge.clone(),
                        state.mcp_prewarmed.clone(),
                        id_style,
                    );
                }
                return Some(slot);
            }
            TakeOutcome::Cold => return None,
            TakeOutcome::Wait => {
                if tokio::time::timeout(MCP_PREWARM_WAIT_TIMEOUT, notified.as_mut())
                    .await
                    .is_err()
                {
                    // Prewarm overran our budget — fall through to cold path.
                    // The in-flight prewarm continues in the background and
                    // its eventual completion fills the slot for the next
                    // caller.
                    return None;
                }
                // Loop to re-check — the prewarm either filled `ready` (we
                // take it next iteration) or failed (we cold-path next
                // iteration when in_flight=false and ready=None).
            }
        }
    }
}

/// Reserve a session slot for the current HTTP turn. Prefers an idle slot;
/// when prewarm is still in flight, blocks on one of the existing slots so we
/// don't expand past `prewarm_count` and orphan a cursor-agent process;
/// otherwise expands the pool up to [`MAX_POOL_SESSIONS`]; finally falls
/// back to waiting on the first slot when the cap is hit. The returned guard
/// pins the slot for the caller's exclusive use until dropped.
pub(crate) async fn acquire_session_slot(
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
    let all_slots = {
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
        // Cap reached and all busy — race locks over the FULL pool (not the
        // stale entry-time `existing`, which can be empty on a cold start).
        pool.clone()
    };
    wait_for_any_slot(all_slots).await
}

/// Race `lock_owned()` across every slot, returning whichever lock resolves
/// first. Pending futures are dropped (and their internal waiters cancelled)
/// when the winner returns, so the loser slots remain available for the next
/// acquirer.
pub(crate) async fn wait_for_any_slot(
    slots: Vec<SessionSlot>,
) -> tokio::sync::OwnedMutexGuard<Option<CursorAcpSession>> {
    use futures::future::FutureExt;
    assert!(!slots.is_empty(), "select_all panics on empty input");
    let mut futures: Vec<_> = slots.into_iter().map(|s| s.lock_owned().boxed()).collect();
    let (guard, _, _rest) = futures::future::select_all(futures.drain(..)).await;
    guard
}

/// Ensure the slot holds an open session, opening one on demand. Equivalent
/// to the previous lazy-open pattern but factored out so every handler reuses
/// the same retry-and-context wrapping.
pub(crate) async fn ensure_session_open(
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
pub(crate) fn spawn_prewarm(state: Arc<RouterState>, count: usize) {
    let target = count.min(MAX_POOL_SESSIONS);
    for slot_idx in 0..target {
        state.prewarming.fetch_add(1, Ordering::SeqCst);
        spawn_single_prewarm(state.clone(), slot_idx);
    }
}

pub(crate) fn spawn_single_prewarm(state: Arc<RouterState>, slot_idx: usize) {
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
pub(crate) struct PrewarmCounterGuard {
    state: Arc<RouterState>,
}

impl Drop for PrewarmCounterGuard {
    fn drop(&mut self) {
        self.state.prewarming.fetch_sub(1, Ordering::SeqCst);
    }
}

pub(crate) async fn run_router(listener: TcpListener, state: Arc<RouterState>) -> Result<()> {
    http_utils::run_streaming_router(listener, state, |request, state, socket| async move {
        handle_request(request, state, socket).await;
    })
    .await
}

pub(crate) async fn handle_request(
    request: String,
    state: Arc<RouterState>,
    mut socket: TcpStream,
) {
    let path = extract_request_path(&request);
    let path = path.split('?').next().unwrap_or("").to_string();
    let method = request.split_whitespace().next().unwrap_or("").to_string();

    let log_id = log_inbound(&method, &path, &request).await;
    let started = Instant::now();

    // Token gate, shared by the plugin endpoint and native-tool launches
    // (both inject a per-launch token). The CORS preflight and the root probe
    // stay open — clients send them before any auth and Claude Code treats a
    // non-200 root as a config error.
    if let Some(expected) = state.config.expected_token.as_deref() {
        let is_open = method == "OPTIONS"
            || ((method == "HEAD" || method == "GET")
                && matches!(strip_v1_prefix(&path), "/" | ""));
        if !is_open && !http_utils::request_loopback_authorized(&request, expected) {
            let msg = "Invalid or missing auth token (expected Authorization: Bearer or x-api-key)";
            let _ = write_json_error(&mut socket, 401, msg).await;
            log_outbound(
                log_id,
                &method,
                &path,
                401,
                None,
                started.elapsed().as_millis() as u64,
            )
            .await;
            return;
        }
    }

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

pub(crate) async fn write_models(
    socket: &mut TcpStream,
    state: &RouterState,
) -> (u16, Option<String>) {
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

pub(crate) fn build_models_response_body(models: &[String]) -> Value {
    model_list_response::build_models_response_body_for_owner(models, CURSOR_ACP_SENTINEL)
}

pub(crate) async fn cached_models(state: &RouterState) -> Result<Vec<String>> {
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
pub(crate) async fn cancel_session_on_error<T>(session: &CursorAcpSession, outcome: &Result<T>) {
    if outcome.is_err() {
        let _ = session.cancel().await;
    }
}

/// Per-protocol inputs after the request body has been parsed and reduced
/// to a flat ACP prompt. The protocol-specific differences live in the
/// closures passed to [`run_turn`] (SSE writer + non-stream body builder).
pub(crate) struct ParsedTurn {
    stream_flag: bool,
    requested_model: Option<String>,
    image_blocks: Vec<Value>,
    prompt: String,
}

/// Shared scaffold for OpenAI / Anthropic / Responses / Gemini turns;
/// per-protocol bits live in the SSE-writer and body-builder closures.
pub(crate) async fn run_turn<S, A>(
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
    let mut stream = match session.prompt_with_blocks(blocks).await {
        Ok(s) => s,
        Err(e) => {
            // A failed prompt write means a dead cursor-agent child — evict so
            // the next request reopens instead of 502ing on the corpse forever.
            *guard = None;
            return Err(e).context("cursor-agent session/prompt");
        }
    };

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
    // Evict on any non-disconnect failure (an unhealthy child would 502 every
    // future request on this slot); a client disconnect leaves it usable.
    if let Err(e) = &outcome
        && !is_client_disconnect(e)
    {
        *guard = None;
    }
    outcome
}

/// Walks the anyhow error chain looking for an io::Error of the broken-pipe
/// or connection-reset variety. Used by the dispatcher to log status 499
/// (nginx's "client closed request" convention) instead of a misleading 502
/// when the launched tool closed the socket mid-stream.
pub(crate) fn is_client_disconnect(err: &anyhow::Error) -> bool {
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

/// True for a malformed request body (JSON that didn't parse) — map to 400 so
/// SDKs fail fast instead of retry-looping on a 5xx.
fn is_bad_request(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<serde_json::Error>().is_some())
}

/// Resolves the status code logged for a failed turn. Broken pipes get 499
/// so logs distinguish client-initiated disconnects (expected, harmless) from
/// malformed requests (400) and real upstream failures (502).
pub(crate) fn status_for_handler_error(err: &anyhow::Error) -> u16 {
    if is_client_disconnect(err) {
        499
    } else if is_bad_request(err) {
        400
    } else {
        502
    }
}

/// Normalized ACP `stopReason`. Each protocol adapter maps this onto its own
/// finish-reason vocab so a truncated / refused turn isn't reported as clean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AcpStop {
    EndTurn,
    MaxTokens,
    Refusal,
}

/// Normalize a `session/prompt` result's `stopReason`; unknown/absent (and
/// `cancelled`/`end_turn`) fold to [`AcpStop::EndTurn`].
pub(crate) fn acp_stop_from_result(result: &Value) -> AcpStop {
    match result.get("stopReason").and_then(Value::as_str) {
        Some("max_tokens") | Some("max_turn_requests") => AcpStop::MaxTokens,
        Some("refusal") => AcpStop::Refusal,
        _ => AcpStop::EndTurn,
    }
}

pub(crate) struct AggregatedTurn {
    content: String,
    reasoning: String,
}

pub(crate) async fn aggregate_prompt_stream(
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

pub(crate) fn extract_agent_text(value: &Value) -> Option<&str> {
    let update = value.get("update")?;
    if update.get("sessionUpdate").and_then(Value::as_str) != Some("agent_message_chunk") {
        return None;
    }
    update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
}

pub(crate) fn extract_agent_thought(value: &Value) -> Option<&str> {
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
pub(crate) const TOOL_RESULT_CHAR_LIMIT: usize = 4000;
/// Maximum characters per tool-schema description.
pub(crate) const TOOL_DESCRIPTION_CHAR_LIMIT: usize = 240;

/// Builds a single SSE frame in `event: <name>\ndata: <json>\n\n` form, used
/// by Anthropic and Responses streams alike.
pub(crate) fn sse_named_event(name: &str, payload: &Value) -> String {
    format!("event: {name}\ndata: {payload}\n\n")
}

/// SSE comment line. Per the SSE spec, lines starting with `:` are ignored by
/// the client but reset its idle-disconnect timer. Used by the streaming
/// handlers when cursor-agent emits a non-text update (tool_call, plan,
/// thought) — without a periodic byte on the wire, OpenAI-SDK clients
/// interpret the silence as a stalled stream and reconnect mid-turn.
pub(crate) const SSE_KEEPALIVE: &str = ": keepalive\n\n";

/// Renders a cursor `tool_call` / `tool_call_update` event as a one-line
/// "[kind] title — summary" string for the thinking block. Returns `None`
/// for events with no human-readable hint so the panel stays clean.
pub(crate) fn extract_tool_call_marker(value: &Value) -> Option<String> {
    let update = value.get("update")?;
    let kind = update.get("sessionUpdate").and_then(Value::as_str)?;
    if kind != "tool_call" && kind != "tool_call_update" {
        return None;
    }
    let title = update.get("title").and_then(Value::as_str);
    let status = update.get("status").and_then(Value::as_str);
    let tool_kind = update.get("kind").and_then(Value::as_str);
    let summary = format_tool_call_summary(update);
    let suffix = summary
        .as_deref()
        .map(|s| format!(" — {s}"))
        .unwrap_or_default();
    match (title, status, tool_kind, summary.as_deref()) {
        (Some(t), Some(s), _, _) => Some(format!("\n[{s}] {t}{suffix}\n")),
        (Some(t), None, _, _) => Some(format!("\n[tool] {t}{suffix}\n")),
        (None, Some(s), Some(k), _) => Some(format!("\n[{k} → {s}]{suffix}\n")),
        (None, _, Some(k), _) => Some(format!("\n[tool: {k}]{suffix}\n")),
        // Bare completion (status + rawOutput) — surface the summary even
        // when the matching tool_call start carried only a generic title.
        (None, Some(s), None, Some(_)) => Some(format!("\n[{s}]{suffix}\n")),
        _ => None,
    }
}

/// One-line summary of a `tool_call_update.rawOutput` payload or a `content`
/// array carrying diff blocks. Covers the shapes cursor-agent emits
/// (search/grep, read, execute, list/glob, edit/write — verified 2026-05-24
/// against cursor-agent 2026.05.20-2b5dd59).
///
/// Edit/write tools emit `update.content = [{type: "diff", path, oldText,
/// newText}]` instead of populating `rawOutput`; without the diff fallback
/// these completions were silently dropped (no marker shown after the
/// `[pending] Edit File` start).
pub(crate) fn format_tool_call_summary(update: &Value) -> Option<String> {
    let mut bits: Vec<String> = Vec::new();
    if let Some(raw) = update.get("rawOutput").and_then(Value::as_object) {
        if let Some(n) = raw.get("totalMatches").and_then(Value::as_u64) {
            bits.push(format!("{n} match{}", if n == 1 { "" } else { "es" }));
        }
        if let Some(n) = raw.get("totalFiles").and_then(Value::as_u64) {
            bits.push(format!("{n} file{}", if n == 1 { "" } else { "s" }));
        }
        if raw.get("truncated").and_then(Value::as_bool) == Some(true) {
            bits.push("truncated".to_string());
        }
        if let Some(code) = raw.get("exitCode").and_then(Value::as_i64) {
            bits.push(format!("exit {code}"));
        }
        if let Some(content) = raw.get("content").and_then(Value::as_str) {
            let lines = content.lines().count().max(1);
            bits.push(format!("{lines} line{}", if lines == 1 { "" } else { "s" }));
        }
    }
    if let Some(content) = update.get("content").and_then(Value::as_array) {
        for block in content {
            if block.get("type").and_then(Value::as_str) != Some("diff") {
                continue;
            }
            let path = block.get("path").and_then(Value::as_str).unwrap_or("");
            let added = block
                .get("newText")
                .and_then(Value::as_str)
                .map(|s| s.lines().count())
                .unwrap_or(0);
            let removed = block
                .get("oldText")
                .and_then(Value::as_str)
                .map(|s| s.lines().count())
                .unwrap_or(0);
            let label = pathlike_basename(path).unwrap_or(path);
            if label.is_empty() {
                bits.push(format!("+{added} -{removed}"));
            } else {
                bits.push(format!("{label} +{added} -{removed}"));
            }
        }
    }
    if bits.is_empty() {
        None
    } else {
        Some(bits.join(", "))
    }
}

/// Returns the final path segment of a `/`-separated path, or `None` for
/// empty input. Used only by [`format_tool_call_summary`] to keep diff
/// markers short ("scratch.txt +2 -1" instead of the full path).
fn pathlike_basename(path: &str) -> Option<&str> {
    if path.is_empty() {
        return None;
    }
    Some(path.rsplit('/').next().unwrap_or(path))
}

// === Shared prompt-reduction helpers ===

/// Extracts plain text from OpenAI `messages[].content`. Accepts a bare string
/// or an array of typed parts (`{type: "text", text: "..."}`); image parts are
/// replaced with a placeholder marker so the agent knows something was elided.
pub(crate) fn extract_openai_message_text(content: Option<&Value>) -> String {
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
pub(crate) fn extract_openai_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(out);
    };
    for msg in messages {
        collect_openai_style_image_blocks(msg.get("content"), &mut out)?;
    }
    Ok(out)
}

pub(crate) fn extract_responses_image_blocks(body: &Value) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let Some(input) = body.get("input").and_then(Value::as_array) else {
        return Ok(out);
    };
    for item in input {
        collect_openai_style_image_blocks(item.get("content"), &mut out)?;
    }
    Ok(out)
}

pub(crate) fn extract_anthropic_image_blocks(body: &Value) -> Result<Vec<Value>> {
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

pub(crate) fn extract_gemini_image_blocks(body: &Value) -> Result<Vec<Value>> {
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
pub(crate) fn collect_openai_style_image_blocks(
    content: Option<&Value>,
    out: &mut Vec<Value>,
) -> Result<()> {
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

pub(crate) fn read_gemini_inline_pair(inline: &Value) -> Result<(&str, &str)> {
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
pub(crate) fn image_capability_error() -> &'static str {
    "cursor: this session's cursor-agent does not advertise promptCapabilities.image; remove the image from the request"
}

/// Parse a `data:<mime>;base64,<payload>` URL into `(mime, base64_payload)`.
/// Returns None for non-data URLs (http, gs://, file ids, etc.) so the
/// caller can produce a "remote URLs not supported" error. Also returns
/// None for non-base64 data URLs (e.g. URL-encoded text) — those would
/// require decoding and re-encoding, and image clients never use them.
pub(crate) fn parse_data_url(url: &str) -> Option<(String, String)> {
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

pub(crate) fn parse_loose_json(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

pub(crate) fn format_openai_tool_call(call: &Value) -> Option<String> {
    let function = call.get("function")?;
    let name = function.get("name").and_then(Value::as_str)?;
    let args_value = match function.get("arguments") {
        Some(Value::String(s)) => parse_loose_json(s),
        Some(other) => other.clone(),
        None => Value::Null,
    };
    Some(format_tool_call_line(name, &args_value))
}

pub(crate) fn format_openai_tools_list(tools: &[Value]) -> Option<String> {
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

pub(crate) fn format_anthropic_tools_list(tools: &[Value]) -> Option<String> {
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

pub(crate) fn format_responses_tools_list(tools: &[Value]) -> Option<String> {
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

pub(crate) fn format_tool_schema_line(name: &str, description: &str) -> String {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        format!("- {name}")
    } else {
        let collapsed = collapse_whitespace(trimmed);
        let snippet = truncate_chars(&collapsed, TOOL_DESCRIPTION_CHAR_LIMIT);
        format!("- {name}: {snippet}")
    }
}

pub(crate) fn finalize_tools_block(lines: Vec<String>) -> Option<String> {
    if lines.is_empty() {
        None
    } else {
        let mut block = String::from("Available tools:\n");
        block.push_str(&lines.join("\n"));
        Some(block)
    }
}

pub(crate) fn format_tool_call_line(name: &str, args: &Value) -> String {
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

pub(crate) fn format_tool_result_block(name: &str, text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        format!("[Tool result for {name}]")
    } else {
        let snippet = truncate_chars(trimmed, TOOL_RESULT_CHAR_LIMIT);
        format!("[Tool result for {name}]\n{snippet}")
    }
}

pub(crate) fn collapse_whitespace(input: &str) -> String {
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

pub(crate) fn truncate_chars(input: &str, max: usize) -> String {
    if input.chars().count() <= max {
        return input.to_string();
    }
    let mut out: String = input.chars().take(max).collect();
    out.push_str("…[truncated]");
    out
}

pub(crate) fn new_responses_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let salt = current_unix_timestamp_micros();
    format!("resp_cur{n:x}{salt:x}")
}

// === Low-level response writers ===

pub(crate) async fn write_json_response(
    socket: &mut TcpStream,
    status: u16,
    body: &Value,
) -> Result<()> {
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

pub(crate) async fn write_json_error(
    socket: &mut TcpStream,
    status: u16,
    message: &str,
) -> Result<()> {
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

pub(crate) async fn write_cors_preflight(socket: &mut TcpStream) -> Result<()> {
    let head = http_response_head_with_extra(204, "text/plain", 0, cors_header_block());
    socket.write_all(head.as_bytes()).await?;
    Ok(())
}

pub(crate) async fn write_root_probe_response(socket: &mut TcpStream) -> Result<()> {
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

pub(crate) async fn write_not_found(socket: &mut TcpStream, path: &str) -> Result<()> {
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

pub(crate) async fn write_sse_chunk(socket: &mut TcpStream, sse_frame: &str) -> Result<()> {
    let chunk = format_http_chunk(sse_frame.as_bytes());
    socket.write_all(&chunk).await?;
    Ok(())
}

pub(crate) async fn write_chunk_terminator(socket: &mut TcpStream) -> Result<()> {
    socket.write_all(b"0\r\n\r\n").await?;
    Ok(())
}

/// Collapses incoming request paths so each handler has a single canonical
/// arm. Strips a leading `/v1` only when it's followed by `/` (or is the
/// whole path), so weird suffixes like `/v1bogus` aren't silently
/// re-interpreted as a known endpoint.
pub(crate) fn strip_v1_prefix(path: &str) -> &str {
    match path.strip_prefix("/v1") {
        Some(rest) if rest.is_empty() || rest.starts_with('/') => rest,
        _ => path,
    }
}

pub(crate) fn new_chat_completion_id() -> String {
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
pub(crate) fn estimate_tokens(text: &str) -> u64 {
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(4) as u64
    }
}

/// Best-effort JSON-output enforcement: cursor-agent/ACP has no
/// structured-output control, so a client's `response_format` / `text.format`
/// / `responseSchema` is otherwise dropped silently. Returns the instruction to
/// append to the prompt, or `None`. Mirrors composer-api's `appendJsonConstraint`.
pub(crate) fn json_output_constraint(body: &Value) -> Option<String> {
    if let Some(line) = body
        .get("response_format")
        .and_then(json_constraint_from_format)
    {
        return Some(line);
    }
    if let Some(line) = body
        .get("text")
        .and_then(|text| text.get("format"))
        .and_then(json_constraint_from_format)
    {
        return Some(line);
    }
    let cfg = body.get("generationConfig")?;
    let wants_json = cfg
        .get("responseMimeType")
        .and_then(Value::as_str)
        .is_some_and(|mime| mime.eq_ignore_ascii_case("application/json"));
    wants_json.then(|| json_constraint_line(cfg.get("responseSchema")))
}

/// Map an OpenAI/Responses `{type, schema?}` format object to a constraint
/// line. `None` for `type: "text"` or an unknown/missing type.
fn json_constraint_from_format(format: &Value) -> Option<String> {
    match format.get("type").and_then(Value::as_str)? {
        "json_object" => Some(json_constraint_line(None)),
        // Chat nests schema under `json_schema.schema`; Responses uses `format.schema`.
        "json_schema" => {
            let schema = format
                .get("schema")
                .or_else(|| format.get("json_schema").and_then(|j| j.get("schema")));
            Some(json_constraint_line(schema))
        }
        _ => None,
    }
}

fn json_constraint_line(schema: Option<&Value>) -> String {
    match schema {
        Some(schema) if !schema.is_null() => format!(
            "OUTPUT CONSTRAINT: Respond with a single valid JSON value and nothing else — no prose, no markdown, no code fences. The JSON must conform to this schema: {}",
            serde_json::to_string(schema).unwrap_or_default()
        ),
        _ => "OUTPUT CONSTRAINT: Respond with a single valid JSON object and nothing else — no prose, no markdown, no code fences."
            .to_string(),
    }
}

/// Append [`json_output_constraint`] to a reduced prompt when JSON was
/// requested. `has_images` keeps an image-only turn (empty text) carrying the
/// constraint; a turn with neither text nor images stays empty so the adapters'
/// empty-prompt guard still rejects it.
pub(crate) fn append_json_output_constraint(
    prompt: String,
    body: &Value,
    has_images: bool,
) -> String {
    if prompt.trim().is_empty() && !has_images {
        return prompt;
    }
    let Some(constraint) = json_output_constraint(body) else {
        return prompt;
    };
    if prompt.trim().is_empty() {
        constraint
    } else {
        format!("{prompt}\n\n{constraint}")
    }
}

pub(crate) fn current_unix_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub(crate) fn current_unix_timestamp_micros() -> u128 {
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

pub(crate) const CURSOR_ROUTER_URL_BASE: &str = "cursor-router://localhost";
pub(crate) const REQUEST_BODY_MAX: usize = 64 * 1024;
pub(crate) const RESPONSE_BODY_MAX: usize = 64 * 1024;

pub(crate) fn new_router_request_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("cur-router-{n:x}")
}

pub(crate) fn parse_request_headers(request: &str) -> BTreeMap<String, String> {
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

pub(crate) fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut split = max;
    while split > 0 && !s.is_char_boundary(split) {
        split -= 1;
    }
    let mut out = s[..split].to_string();
    out.push_str("…[truncated]");
    out
}

pub(crate) async fn log_inbound(method: &str, path: &str, request: &str) -> Option<String> {
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

pub(crate) async fn log_outbound(
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
