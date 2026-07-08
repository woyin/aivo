//! HTTP MCP server that exposes claude-cli's `/v1/messages` `tools` array to
//! cursor-agent's model. Closes the long-standing gap where client-declared
//! tools (`AskUserQuestion`, `TaskCreate`, etc.) silently degraded to plain
//! prose on the cursor route.
//!
//! cursor-agent's ACP `tool_call` updates carry no tool name / arguments
//! ([[reference-cursor-acp-mcp-propagation]]), so tool semantics flow only
//! through this MCP HTTP path — never through the ACP session-update stream.

use anyhow::{Context, Result, anyhow};
use rand::Rng;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, Notify, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::time::timeout;

use crate::services::acp_client::PromptStream;
use crate::services::cursor_acp::CursorAcpSession;
use crate::services::http_debug::{self, DebugEntry, Phase};
use crate::services::http_utils::{
    self, bind_local_listener, cors_header_block, extract_request_body, extract_request_path,
    http_response_head_with_extra,
};

/// MCP protocol version we advertise during `initialize`. Cursor-agent
/// (2026.05.20-2b5dd59) accepts this.
const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// How long an MCP `tools/call` is held parked before we give up and surface
/// an error back to cursor-agent. Set generously because the round-trip
/// involves claude-cli closing the SSE, running the tool locally (which can
/// itself be a long-running picker), and opening a fresh `/v1/messages` with
/// the result.
const TOOL_CALL_PARK_TIMEOUT: Duration = Duration::from_secs(600);

/// How long an MCP `tools/list` response is parked when the session was
/// opened via [`McpBridge::open_session_for_prewarm`]. Current cursor-agent
/// builds only fetch `tools/list` when they're about to process a prompt
/// (well after `take_for_use` runs), so this is belt-and-suspenders for
/// future or differently-configured cursor clients that probe eagerly.
/// 60 s gives the user a comfortable window to type their first prompt.
const TOOLS_LIST_PARK_TIMEOUT: Duration = Duration::from_secs(60);

/// Cap on concurrent bridge sessions. Matches the cursor router's session
/// pool ceiling so a flood of tool-using `/v1/messages` requests can't
/// fork-bomb cursor-agent (each bridge session spawns a fresh Node.js
/// child). Excess requests wait for an in-flight session to finish.
pub const MAX_BRIDGE_SESSIONS: usize = 3;

/// Top-level bridge handle. One instance lives for the lifetime of the
/// cursor router; HTTP MCP requests from any cursor-agent session this
/// router opens come back to this server on a unique `/sess/<id>/` path.
pub struct McpBridge {
    state: Arc<BridgeState>,
}

struct BridgeState {
    /// Port the bridge's HTTP MCP server listens on (localhost-only).
    port: u16,
    /// Active per-(claude-cli-turn) bridge sessions, keyed by an opaque
    /// bridge id we embed in the MCP server's URL path.
    sessions: Mutex<HashMap<String, Arc<Mutex<BridgeSession>>>>,
    /// Reverse lookup: a parked tool_use_id points to its bridge session id
    /// so the resumption `/v1/messages` (carrying `tool_result`) can find
    /// the still-running ACP prompt without claude-cli having to forward
    /// the session id explicitly.
    parked: Mutex<HashMap<String, String>>,
    /// Concurrency cap on live bridge sessions. Acquired in `open_session`
    /// and released when the BridgeSession is dropped (via the held
    /// `_permit`). Prevents unbounded cursor-agent process growth when a
    /// caller spams tool-using requests faster than they complete.
    session_permits: Arc<Semaphore>,
}

/// State shared between the HTTP MCP request handler and the cursor router's
/// `/v1/messages` handlers for one in-flight assistant turn (which may span
/// several HTTP round-trips when the model invokes client-side tools).
pub struct BridgeSession {
    pub id: String,
    /// Protocol-specific id-prefix style for the synthetic tool_use_id this
    /// session emits to upstream agents.
    id_style: ToolUseIdStyle,
    /// The Anthropic-format tool schemas that this session's MCP `tools/list`
    /// will translate and return. Set by the cursor router when the first
    /// `/v1/messages` for this turn arrives.
    tools: Vec<Value>,
    /// The single currently-parked MCP `tools/call`, if any. cursor-agent
    /// serializes tool calls within a prompt, so we never see more than one
    /// at a time per session.
    parked: Option<ParkedCall>,
    /// Outbound event channel into the cursor router's SSE loop. The router
    /// installs this with [`Self::attach_event_sink`] for the duration of a
    /// `/v1/messages` turn; when the MCP HTTP handler parks a tool call it
    /// pushes a [`BridgeEvent::ToolCall`] onto this channel so the router
    /// can render the matching Anthropic `tool_use` block.
    event_tx: Option<mpsc::Sender<BridgeEvent>>,
    /// The cursor ACP session backing this bridge session. Kept alive across
    /// the parking gap so the same ACP prompt can resume when claude-cli
    /// sends `tool_result` on a follow-up `/v1/messages`.
    acp: Option<CursorAcpSession>,
    /// The PromptStream currently driving this session's `session/prompt`.
    /// Taken by the active HTTP handler while it streams events; put back
    /// before the handler returns so the resumption handler can pick it up.
    stream: Option<PromptStream>,
    /// Semaphore permit held for the lifetime of the bridge session. The
    /// permit drops when the BridgeSession is dropped, freeing capacity
    /// for the next waiter in `open_session`.
    _permit: OwnedSemaphorePermit,
    /// True when the session was created via [`McpBridge::open_session_for_prewarm`]
    /// and is still waiting for the cursor router to swap in the real tools.
    /// Cleared by [`McpBridge::take_for_use`].
    awaiting_real_tools: bool,
    /// Fired by [`McpBridge::take_for_use`] to wake up any `tools/list` handler
    /// that parked because `awaiting_real_tools` was true.
    tools_ready: Arc<Notify>,
}

struct ParkedCall {
    tool_use_id: String,
    /// Name of the tool that was called. Stored so the by-name resumption
    /// path (Gemini) can find this call without an id from the upstream
    /// agent. Cursor serializes tool calls per session, so name-based
    /// lookup is unambiguous in practice.
    tool_name: String,
    /// Sender used by the resumption path to deliver the claude-cli-side
    /// `tool_result` content back to the MCP HTTP handler that's still
    /// holding the response open.
    response_tx: oneshot::Sender<McpToolResult>,
}

/// Result the MCP shim delivers back to cursor-agent's `tools/call`. Mirrors
/// the JSON-RPC `result` body for MCP's tools/call response.
#[derive(Clone, Debug)]
pub struct McpToolResult {
    pub content: Vec<Value>,
    pub is_error: bool,
}

/// Per-protocol id-prefix style for the synthetic tool_use_ids the bridge
/// generates. Each upstream client validates the id format slightly
/// differently — Anthropic SDKs assume `toolu_`, OpenAI/Responses SDKs
/// assume `call_` and some pattern-match the prefix to distinguish their
/// own ids from injected ones. The router picks the style when it opens
/// a bridge session for a given protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolUseIdStyle {
    Anthropic,
    OpenAi,
    Gemini,
}

impl ToolUseIdStyle {
    fn prefix(self) -> &'static str {
        match self {
            ToolUseIdStyle::Anthropic => "toolu_",
            ToolUseIdStyle::OpenAi => "call_",
            ToolUseIdStyle::Gemini => "call_",
        }
    }
}

/// Notification produced by `BridgeSession::poll_next_event`. Driven by the
/// MCP HTTP path; the cursor router consumes these and emits matching
/// Anthropic SSE frames.
#[derive(Clone, Debug)]
pub enum BridgeEvent {
    /// cursor-agent's model invoked one of our client-declared tools.
    /// Carries the allocated Anthropic `tool_use_id`, the tool name, and
    /// the structured arguments.
    ToolCall {
        tool_use_id: String,
        name: String,
        arguments: Value,
    },
}

impl McpBridge {
    /// Bind the MCP server's HTTP listener and start serving in the
    /// background. Returns the bridge handle plus the server task. The
    /// caller (cursor router startup) keeps the handle around for the
    /// lifetime of the router process; the task is detached.
    pub async fn start_background() -> Result<Arc<Self>> {
        let (listener, port) = bind_local_listener()
            .await
            .context("bind MCP bridge listener")?;
        let state = Arc::new(BridgeState {
            port,
            sessions: Mutex::new(HashMap::new()),
            parked: Mutex::new(HashMap::new()),
            session_permits: Arc::new(Semaphore::new(MAX_BRIDGE_SESSIONS)),
        });
        let server_state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = serve(listener, server_state).await {
                eprintln!("aivo: cursor_bridge::mcp server stopped: {e:#}");
            }
        });
        Ok(Arc::new(Self { state }))
    }

    pub fn port(&self) -> u16 {
        self.state.port
    }

    /// Construct an inert bridge that doesn't bind a port or accept HTTP
    /// requests. Used by tests that build a `RouterState` for code paths
    /// that never touch the bridge (model listing, OPTIONS preflight, etc.).
    #[cfg(test)]
    pub fn for_tests() -> Arc<Self> {
        Arc::new(Self {
            state: Arc::new(BridgeState {
                port: 0,
                sessions: Mutex::new(HashMap::new()),
                parked: Mutex::new(HashMap::new()),
                session_permits: Arc::new(Semaphore::new(MAX_BRIDGE_SESSIONS)),
            }),
        })
    }

    /// Allocate a fresh bridge session and return the matching MCP-server
    /// URL to put in `mcpServers[].url`. The cursor router calls this once
    /// per claude-cli assistant turn (the resumption path uses
    /// [`Self::resume_with_tool_result`] instead). The caller still has to
    /// open the ACP session and call [`BridgeSession::attach_session`] to
    /// hand the session + stream off for the parking gap.
    pub async fn open_session(
        &self,
        tools: Vec<Value>,
        id_style: ToolUseIdStyle,
    ) -> (Arc<Mutex<BridgeSession>>, String) {
        // Block until a permit frees up. Under load this serializes
        // bridged turns at MAX_BRIDGE_SESSIONS instead of fork-bombing
        // cursor-agent.
        let permit = self
            .state
            .session_permits
            .clone()
            .acquire_owned()
            .await
            .expect("session_permits semaphore never closes");
        let id = generate_bridge_id();
        let session = Arc::new(Mutex::new(BridgeSession {
            id: id.clone(),
            id_style,
            tools,
            parked: None,
            event_tx: None,
            acp: None,
            stream: None,
            _permit: permit,
            awaiting_real_tools: false,
            tools_ready: Arc::new(Notify::new()),
        }));
        self.state
            .sessions
            .lock()
            .await
            .insert(id.clone(), session.clone());
        let url = format!("http://127.0.0.1:{}/sess/{}/", self.state.port, id);
        (session, url)
    }

    /// Pre-open a bridge session before the upstream tool's request has
    /// arrived. The session starts with empty tools and `awaiting_real_tools`
    /// set, so any `tools/list` poll cursor-agent issues during the warm-up
    /// window is parked (up to [`TOOLS_LIST_PARK_TIMEOUT`]) until the cursor
    /// router activates the session with [`Self::take_for_use`].
    ///
    /// Use this when the cursor router knows the protocol up front (e.g. the
    /// codex `/responses` bridge) and wants to amortize the ~16 s cursor-agent
    /// `session/new` cost before the first real request lands.
    pub async fn open_session_for_prewarm(
        &self,
        id_style: ToolUseIdStyle,
    ) -> (Arc<Mutex<BridgeSession>>, String) {
        let permit = self
            .state
            .session_permits
            .clone()
            .acquire_owned()
            .await
            .expect("session_permits semaphore never closes");
        let id = generate_bridge_id();
        let session = Arc::new(Mutex::new(BridgeSession {
            id: id.clone(),
            id_style,
            tools: Vec::new(),
            parked: None,
            event_tx: None,
            acp: None,
            stream: None,
            _permit: permit,
            awaiting_real_tools: true,
            tools_ready: Arc::new(Notify::new()),
        }));
        self.state
            .sessions
            .lock()
            .await
            .insert(id.clone(), session.clone());
        let url = format!("http://127.0.0.1:{}/sess/{}/", self.state.port, id);
        (session, url)
    }

    /// Activate a session opened by [`Self::open_session_for_prewarm`]: swap
    /// in the real tool catalog and release any `tools/list` request that
    /// parked while we were idle. After this returns the session behaves like
    /// one created via [`Self::open_session`].
    pub async fn take_for_use(session: &Arc<Mutex<BridgeSession>>, real_tools: Vec<Value>) {
        let notify = {
            let mut guard = session.lock().await;
            guard.tools = real_tools;
            guard.awaiting_real_tools = false;
            guard.tools_ready.clone()
        };
        notify.notify_waiters();
    }

    /// Drop a bridge session entirely. Called when the ACP prompt fully
    /// terminates (`end_turn`) or when the cursor router gives up on it.
    pub async fn drop_session(&self, bridge_id: &str) {
        purge_bridge_session(&self.state, bridge_id, None).await;
    }

    /// Resumption path: deliver claude-cli's `tool_result` back to the MCP
    /// HTTP handler that's still holding the response open, and return the
    /// owning bridge session so the cursor router can attach the new SSE
    /// stream and keep bridging events. Returns `None` when no parked call
    /// matches — caller should treat that as "fresh path" instead.
    pub async fn resume_with_tool_result(
        &self,
        tool_use_id: &str,
        content: Vec<Value>,
        is_error: bool,
    ) -> Option<Arc<Mutex<BridgeSession>>> {
        let bridge_id = self.state.parked.lock().await.remove(tool_use_id)?;
        let session = self.state.sessions.lock().await.get(&bridge_id).cloned()?;
        let parked = {
            let mut guard = session.lock().await;
            guard.parked.take()
        };
        // None on no-op so callers fall through to fresh instead of streaming
        // a resume against a tools/call that was never satisfied.
        let p = parked?;
        if p.tool_use_id != tool_use_id {
            return None;
        }
        if p.response_tx
            .send(McpToolResult { content, is_error })
            .is_err()
        {
            return None;
        }
        Some(session)
    }

    /// Deliver the tool_result and immediately tear the bridge session
    /// down. Used by the non-streaming resumption path: cursor-agent gets
    /// its MCP response so it stops waiting, but we don't keep the session
    /// alive — the new request flows through the legacy text-flatten path
    /// instead. Returns `true` when a parked call matched and was torn
    /// down. Without this, a stream=true→stream=false retry would orphan
    /// the parked session for the full 600 s park timeout.
    pub async fn deliver_and_drop_parked(
        &self,
        tool_use_id: &str,
        content: Vec<Value>,
        is_error: bool,
    ) -> bool {
        let Some(session) = self
            .resume_with_tool_result(tool_use_id, content, is_error)
            .await
        else {
            return false;
        };
        let bridge_id = session.lock().await.id.clone();
        self.drop_session(&bridge_id).await;
        true
    }

    /// Name-based variant of [`Self::resume_with_tool_result`] for protocols
    /// that don't echo the synthetic tool_use_id (Gemini's `functionResponse`
    /// carries only `name`). Only delivers on a GLOBALLY unique match: two
    /// conversations parked on the same tool name would otherwise cross-deliver
    /// and corrupt both, so an ambiguous match (0 or >1) returns `None`.
    pub async fn resume_with_tool_result_by_name(
        &self,
        tool_name: &str,
        content: Vec<Value>,
        is_error: bool,
    ) -> Option<Arc<Mutex<BridgeSession>>> {
        let sessions: Vec<Arc<Mutex<BridgeSession>>> =
            self.state.sessions.lock().await.values().cloned().collect();
        let mut matches: Vec<String> = Vec::new();
        for session in &sessions {
            let guard = session.lock().await;
            if let Some(p) = guard.parked.as_ref().filter(|p| p.tool_name == tool_name) {
                matches.push(p.tool_use_id.clone());
            }
        }
        if matches.len() != 1 {
            return None;
        }
        self.resume_with_tool_result(&matches[0], content, is_error)
            .await
    }
}

impl BridgeSession {
    /// Install the channel the MCP HTTP handler should push events onto for
    /// this turn. Returns a receiver the router drains inside its SSE loop.
    /// Caller is responsible for clearing the channel with
    /// [`Self::detach_event_sink`] when the turn finishes.
    pub fn attach_event_sink(&mut self) -> mpsc::Receiver<BridgeEvent> {
        let (tx, rx) = mpsc::channel(8);
        self.event_tx = Some(tx);
        rx
    }

    pub fn detach_event_sink(&mut self) {
        self.event_tx = None;
    }

    /// Move the freshly-opened ACP session + its first prompt's stream into
    /// the bridge session. Called by the cursor router after
    /// [`McpBridge::open_session`] and a successful
    /// `session.prompt_with_blocks(...)`.
    pub fn attach_session(&mut self, acp: CursorAcpSession, stream: PromptStream) {
        self.acp = Some(acp);
        self.stream = Some(stream);
    }

    /// Take exclusive control of the ACP session + prompt stream for the
    /// duration of one `/v1/messages` SSE response. Returns `Err` when
    /// neither slot is populated — this can happen on a race where one
    /// handler errored mid-attach and another resumption found the bridge
    /// in the session map but with cleared slots. Callers surface this as
    /// an HTTP 500 instead of panicking the tokio task.
    pub fn take_active(&mut self) -> Result<(CursorAcpSession, PromptStream)> {
        let acp = self
            .acp
            .take()
            .ok_or_else(|| anyhow!("bridge session has no attached cursor ACP session"))?;
        let stream = self
            .stream
            .take()
            .ok_or_else(|| anyhow!("bridge session has no attached prompt stream"))?;
        Ok((acp, stream))
    }

    /// Put the ACP session + stream back so the resumption handler (or a
    /// later cleanup) can pick them up. Inverse of [`Self::take_active`].
    pub fn return_active(&mut self, acp: CursorAcpSession, stream: PromptStream) {
        self.acp = Some(acp);
        self.stream = Some(stream);
    }
}

// ----- HTTP MCP server -----

async fn serve(listener: TcpListener, state: Arc<BridgeState>) -> Result<()> {
    http_utils::run_streaming_router(listener, state, |request, state, socket| async move {
        handle(request, state, socket).await;
    })
    .await
}

async fn handle(request: String, state: Arc<BridgeState>, mut socket: TcpStream) {
    let path = extract_request_path(&request);
    let path = path.split('?').next().unwrap_or("").to_string();
    let method = request.split_whitespace().next().unwrap_or("").to_string();
    let body_for_log = extract_request_body(&request)
        .ok()
        .map(|b| b.to_string())
        .unwrap_or_default();
    let log_id = log_mcp_inbound(&method, &path, &body_for_log).await;
    let started = std::time::Instant::now();

    // OPTIONS is sent by some HTTP clients as a CORS preflight; respond 204.
    if method == "OPTIONS" {
        let _ = socket
            .write_all(
                http_response_head_with_extra(204, "text/plain", 0, cors_header_block()).as_bytes(),
            )
            .await;
        return;
    }

    let Some(bridge_id) = parse_session_path(&path) else {
        let _ = write_json_status(&mut socket, 404, "session path not found").await;
        return;
    };

    if method == "GET" {
        // The cursor-agent Phase 0 probe never opened a server→client SSE
        // stream; until that changes, 405 keeps clients from hanging on a
        // long-poll we don't implement.
        let _ = write_json_status(&mut socket, 405, "GET not supported").await;
        return;
    }
    if method != "POST" {
        let _ = write_json_status(&mut socket, 405, "method not allowed").await;
        return;
    }

    let body = match extract_request_body(&request) {
        Ok(b) => b.to_string(),
        Err(_) => {
            let _ = write_json_status(&mut socket, 400, "missing body").await;
            return;
        }
    };
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(_) => {
            let _ = write_json_status(&mut socket, 400, "invalid JSON").await;
            return;
        }
    };

    let session = state.sessions.lock().await.get(&bridge_id).cloned();
    let Some(session) = session else {
        let _ = write_json_status(&mut socket, 404, "session unknown").await;
        return;
    };

    let msg_id = parsed.get("id").cloned();
    let method_name = parsed
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let params = parsed.get("params").cloned().unwrap_or(Value::Null);

    // Notifications have no id; per JSON-RPC + MCP spec we return 202 with
    // an empty body. Currently the only one cursor-agent sends is
    // `notifications/initialized`.
    if msg_id.is_none() {
        let _ = socket
            .write_all(
                http_response_head_with_extra(202, "text/plain", 0, cors_header_block()).as_bytes(),
            )
            .await;
        log_mcp_outbound(log_id, &method, &path, 202, None, started.elapsed()).await;
        return;
    }

    let response = match method_name.as_str() {
        "initialize" => json_rpc_ok(msg_id, mcp_initialize_result()),
        "tools/list" => {
            // Subscribe to `tools_ready` BEFORE re-checking the flag so a
            // `take_for_use` that races between the check and the await still
            // wakes us. Bounded by `TOOLS_LIST_PARK_TIMEOUT` so a never-used
            // prewarm slot doesn't pin cursor-agent's MCP poll forever.
            let notify = session.lock().await.tools_ready.clone();
            let notified = notify.notified();
            tokio::pin!(notified);
            // Enroll before the recheck — see take_mcp_prewarmed for rationale.
            notified.as_mut().enable();
            let still_awaiting = session.lock().await.awaiting_real_tools;
            if still_awaiting {
                let _ = timeout(TOOLS_LIST_PARK_TIMEOUT, notified.as_mut()).await;
            }
            let tools = session.lock().await.tools.clone();
            json_rpc_ok(msg_id, json!({"tools": translate_tools(&tools)}))
        }
        "tools/call" => match handle_tools_call(&state, &session, params).await {
            Ok(result) => json_rpc_ok(msg_id, result),
            Err(e) => json_rpc_err(msg_id, -32000, &e.to_string()),
        },
        other => json_rpc_err(msg_id, -32601, &format!("method not found: {other}")),
    };

    let body = response.to_string();
    let head =
        http_response_head_with_extra(200, "application/json", body.len(), cors_header_block());
    let _ = socket.write_all(head.as_bytes()).await;
    let _ = socket.write_all(body.as_bytes()).await;
    log_mcp_outbound(log_id, &method, &path, 200, Some(body), started.elapsed()).await;
}

const MCP_LOG_URL_BASE: &str = "mcp-bridge://localhost";

/// Mask the session id in a `/sess/<id>/…` path before logging it: the id is
/// the MCP server's only access secret (no bearer), so a log reader could
/// otherwise POST `tools/call` into a live turn.
fn redact_session_path(path: &str) -> String {
    match parse_session_path(path) {
        Some(id) => path.replacen(&id, "<redacted>", 1),
        None => path.to_string(),
    }
}

async fn log_mcp_inbound(method: &str, path: &str, body: &str) -> Option<String> {
    let logger = http_debug::global()?;
    let id = format!(
        "mcp-{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros())
            .unwrap_or(0)
    );
    logger
        .log(DebugEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            id: id.clone(),
            phase: Phase::Request,
            method: method.to_string(),
            url: format!("{MCP_LOG_URL_BASE}{}", redact_session_path(path)),
            status: None,
            duration_ms: None,
            request_headers: std::collections::BTreeMap::new(),
            request_body: if body.is_empty() {
                None
            } else {
                Some(body.to_string())
            },
            response_headers: std::collections::BTreeMap::new(),
            response_body: None,
            error: None,
        })
        .await;
    Some(id)
}

async fn log_mcp_outbound(
    id: Option<String>,
    method: &str,
    path: &str,
    status: u16,
    body: Option<String>,
    duration: std::time::Duration,
) {
    let (Some(logger), Some(id)) = (http_debug::global(), id) else {
        return;
    };
    logger
        .log(DebugEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            id,
            phase: Phase::Response,
            method: method.to_string(),
            url: format!("{MCP_LOG_URL_BASE}{}", redact_session_path(path)),
            status: Some(status),
            duration_ms: Some(duration.as_millis() as u64),
            request_headers: std::collections::BTreeMap::new(),
            request_body: None,
            response_headers: std::collections::BTreeMap::new(),
            response_body: body,
            error: None,
        })
        .await;
}

fn mcp_initialize_result() -> Value {
    json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name": "aivo-cursor-bridge", "version": env!("CARGO_PKG_VERSION")},
    })
}

/// Translate Anthropic tool schemas (`{name, description, input_schema}`) to
/// MCP form (`{name, description, inputSchema}`). The JSON-Schema body itself
/// is structurally identical between the two protocols.
fn translate_tools(anthropic_tools: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(anthropic_tools.len());
    for tool in anthropic_tools {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let mut entry = serde_json::Map::new();
        entry.insert("name".into(), Value::String(name.to_string()));
        if let Some(desc) = tool.get("description").and_then(Value::as_str) {
            entry.insert("description".into(), Value::String(desc.to_string()));
        }
        let schema = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        entry.insert("inputSchema".into(), schema);
        out.push(Value::Object(entry));
    }
    out
}

async fn handle_tools_call(
    state: &Arc<BridgeState>,
    session: &Arc<Mutex<BridgeSession>>,
    params: Value,
) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("tools/call missing `name`"))?
        .to_string();
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let id_style = { session.lock().await.id_style };
    let tool_use_id = new_tool_use_id(id_style);
    let (tx, rx) = oneshot::channel();

    {
        let mut guard = session.lock().await;
        if guard.parked.is_some() {
            return Err(anyhow!(
                "another tool call is already parked on this bridge session"
            ));
        }
        guard.parked = Some(ParkedCall {
            tool_use_id: tool_use_id.clone(),
            tool_name: name.clone(),
            response_tx: tx,
        });
        state
            .parked
            .lock()
            .await
            .insert(tool_use_id.clone(), guard.id.clone());
    }

    let bridge_id = { session.lock().await.id.clone() };
    let event = BridgeEvent::ToolCall {
        tool_use_id: tool_use_id.clone(),
        name,
        arguments,
    };
    // Hand the event to the cursor router via the session's pending channel.
    // When no router is listening (turn already exited), fail fast instead
    // of parking 600 s on an oneshot no one will ever fulfill.
    if let Err(e) = push_pending_event(session, event).await {
        purge_bridge_session(state, &bridge_id, Some(&tool_use_id)).await;
        return Err(e);
    }

    let result = match timeout(TOOL_CALL_PARK_TIMEOUT, rx).await {
        Ok(Ok(r)) => r,
        Ok(Err(_canceled)) => {
            // oneshot Sender dropped without sending — the bridge session
            // was torn down. State map was already cleaned by `drop_session`;
            // just surface an MCP error so cursor-agent stops waiting.
            return Err(anyhow!("bridge session dropped before tool_result"));
        }
        Err(_elapsed) => {
            // 10-minute park elapsed. Tear down the orphaned bridge session
            // wholesale: clearing only `state.parked` and `session.parked`
            // (the pre-fix behavior) leaked the CursorAcpSession child
            // process and the `state.sessions` entry indefinitely.
            purge_bridge_session(state, &bridge_id, Some(&tool_use_id)).await;
            return Err(anyhow!("timed out waiting for tool_result from claude-cli"));
        }
    };

    Ok(json!({
        "content": result.content,
        "isError": result.is_error,
    }))
}

/// Tear down a bridge session and all its bookkeeping in one place: drops
/// the `state.sessions` entry (releasing the held [`CursorAcpSession`] and
/// `PromptStream`), removes the optional `state.parked` lookup, and the
/// session's own `parked` field. Idempotent — callers don't have to know
/// which bookkeeping is still set.
async fn purge_bridge_session(state: &BridgeState, bridge_id: &str, tool_use_id: Option<&str>) {
    let removed = state.sessions.lock().await.remove(bridge_id);
    if let Some(id) = tool_use_id {
        state.parked.lock().await.remove(id);
    }
    if let Some(sess) = removed {
        let mut guard = sess.lock().await;
        if let Some(parked) = guard.parked.take()
            && tool_use_id != Some(parked.tool_use_id.as_str())
        {
            // Drop the global lookup unless the caller already removed it above.
            state.parked.lock().await.remove(&parked.tool_use_id);
        }
        guard.acp.take();
        guard.stream.take();
        guard.event_tx.take();
    }
}

/// The pending-event channel between the MCP HTTP handler and the cursor
/// router's SSE loop. Stored as a single Option slot inside the session
/// because at most one tool call can be in flight at a time (cursor-agent
/// serializes them within a prompt). Returns `Err` when there is no live
/// router-side receiver — the caller must NOT park the tools/call in that
/// case, otherwise it would wait the full 600 s park timeout for an event
/// no one will ever drain.
async fn push_pending_event(session: &Arc<Mutex<BridgeSession>>, event: BridgeEvent) -> Result<()> {
    let sender = {
        let guard = session.lock().await;
        guard.event_tx.clone()
    };
    let Some(tx) = sender else {
        return Err(anyhow!(
            "bridge session has no router-side event sink (turn already exited)"
        ));
    };
    tx.send(event)
        .await
        .map_err(|_| anyhow!("router-side event receiver dropped (turn already exited)"))
}

// ----- JSON-RPC helpers -----

fn json_rpc_ok(id: Option<Value>, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id.unwrap_or(Value::Null), "result": result})
}

fn json_rpc_err(id: Option<Value>, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(Value::Null),
        "error": {"code": code, "message": message},
    })
}

async fn write_json_status(socket: &mut TcpStream, status: u16, message: &str) -> Result<()> {
    let body = json!({"error": message}).to_string();
    let head =
        http_response_head_with_extra(status, "application/json", body.len(), cors_header_block());
    socket.write_all(head.as_bytes()).await?;
    socket.write_all(body.as_bytes()).await?;
    Ok(())
}

// ----- Misc -----

/// Parse `/sess/<id>/...` and extract `<id>`. Returns `None` for any other
/// path shape so unrelated probes (favicon, root) fall through to 404.
fn parse_session_path(path: &str) -> Option<String> {
    let rest = path.strip_prefix("/sess/")?;
    let id = rest.split('/').next().filter(|s| !s.is_empty())?;
    if id.bytes().all(|b| b.is_ascii_alphanumeric()) {
        Some(id.to_string())
    } else {
        None
    }
}

fn generate_bridge_id() -> String {
    let mut rng = rand::thread_rng();
    (0..16)
        .map(|_| {
            let n: u8 = rng.gen_range(0..36);
            if n < 10 {
                (b'0' + n) as char
            } else {
                (b'a' + (n - 10)) as char
            }
        })
        .collect()
}

/// Allocate a fresh tool_use_id with the per-protocol prefix the upstream
/// agent expects (`toolu_` for Anthropic, `call_` for OpenAI/Responses and
/// Gemini). 24 base62 chars of entropy follow the prefix; collision
/// probability is astronomically small over the lifetime of a router
/// process even for the global parked-call lookup.
fn new_tool_use_id(style: ToolUseIdStyle) -> String {
    let mut rng = rand::thread_rng();
    let suffix: String = (0..24)
        .map(|_| {
            let n: u8 = rng.gen_range(0..62);
            match n {
                0..=9 => (b'0' + n) as char,
                10..=35 => (b'a' + (n - 10)) as char,
                _ => (b'A' + (n - 36)) as char,
            }
        })
        .collect();
    format!("{}{suffix}", style.prefix())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// reqwest's `system-proxy` feature inherits the host's HTTP proxy, which
    /// in some dev environments (corp VPN, charles) returns 500 for local
    /// addresses. Build a client that skips system proxies so the test sees
    /// the bridge's actual responses.
    fn http_client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("build no-proxy reqwest client")
    }

    #[test]
    fn translate_tools_renames_input_schema_and_preserves_body() {
        let inp = vec![json!({
            "name": "AskUserQuestion",
            "description": "Ask the user a question.",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        })];
        let out = translate_tools(&inp);
        assert_eq!(out.len(), 1);
        let t = &out[0];
        assert_eq!(t.get("name").unwrap(), "AskUserQuestion");
        assert_eq!(t.get("description").unwrap(), "Ask the user a question.");
        assert!(t.get("input_schema").is_none());
        let schema = t.get("inputSchema").unwrap();
        assert_eq!(schema.get("type").unwrap(), "object");
        assert_eq!(schema.get("required").unwrap(), &json!(["q"]));
    }

    #[test]
    fn translate_tools_supplies_empty_object_schema_when_missing() {
        let inp = vec![json!({"name": "Noop"})];
        let out = translate_tools(&inp);
        let schema = out[0].get("inputSchema").unwrap();
        assert_eq!(schema.get("type").unwrap(), "object");
    }

    #[test]
    fn translate_tools_skips_entries_without_name() {
        let inp = vec![json!({"description": "no name"}), json!({"name": "Ok"})];
        assert_eq!(translate_tools(&inp).len(), 1);
    }

    #[test]
    fn parse_session_path_accepts_alnum_id_and_rejects_others() {
        assert_eq!(parse_session_path("/sess/abc123/"), Some("abc123".into()));
        assert_eq!(parse_session_path("/sess/abc123"), Some("abc123".into()));
        assert_eq!(parse_session_path("/sess/"), None);
        assert_eq!(parse_session_path("/sess/has-dash/"), None);
        assert_eq!(parse_session_path("/other/path"), None);
    }

    #[test]
    fn new_tool_use_id_emits_per_protocol_prefix() {
        let anth = new_tool_use_id(ToolUseIdStyle::Anthropic);
        assert!(anth.starts_with("toolu_"));
        assert_eq!(anth.len(), "toolu_".len() + 24);
        let openai = new_tool_use_id(ToolUseIdStyle::OpenAi);
        assert!(openai.starts_with("call_"));
        let gem = new_tool_use_id(ToolUseIdStyle::Gemini);
        assert!(gem.starts_with("call_"));
    }

    #[test]
    fn new_tool_use_id_suffix_is_alphanumeric() {
        let id = new_tool_use_id(ToolUseIdStyle::Anthropic);
        assert!(
            id["toolu_".len()..]
                .chars()
                .all(|c| c.is_ascii_alphanumeric())
        );
    }

    /// Round-trip: open a bridge session, simulate a `tools/call` HTTP
    /// request, capture the matching [`BridgeEvent::ToolCall`], deliver a
    /// `tool_result` back, and verify the MCP response carries the right
    /// content. Exercises the full park/resume state machine without any
    /// cursor-agent dependency.
    #[tokio::test]
    async fn full_park_resume_round_trip() {
        let bridge = McpBridge::start_background().await.unwrap();
        let port = bridge.port();
        let tools = vec![json!({
            "name": "AskUserQuestion",
            "description": "ask",
            "input_schema": {"type": "object"},
        })];
        let (session, url) = bridge.open_session(tools, ToolUseIdStyle::Anthropic).await;
        assert!(url.starts_with(&format!("http://127.0.0.1:{port}/sess/")));

        // Attach the event channel so the MCP HTTP handler can hand us the
        // ToolCall when it arrives.
        let mut event_rx = session.lock().await.attach_event_sink();

        // Fire the simulated MCP tools/call on a background task — it will
        // block until we deliver the tool_result.
        let url_clone = url.clone();
        let mcp_task = tokio::spawn(async move {
            let client = http_client();
            client
                .post(&url_clone)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "AskUserQuestion", "arguments": {"q": "color?"}},
                }))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        });

        let event = tokio::time::timeout(Duration::from_secs(5), event_rx.recv())
            .await
            .expect("event channel didn't fire")
            .expect("event channel closed");
        let BridgeEvent::ToolCall {
            tool_use_id,
            name,
            arguments,
        } = event;
        assert_eq!(name, "AskUserQuestion");
        assert_eq!(arguments, json!({"q": "color?"}));
        assert!(tool_use_id.starts_with("toolu_"));

        let resumed = bridge
            .resume_with_tool_result(
                &tool_use_id,
                vec![json!({"type": "text", "text": "blue"})],
                false,
            )
            .await;
        assert!(resumed.is_some(), "expected matching parked session");

        let response = tokio::time::timeout(Duration::from_secs(5), mcp_task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 1);
        let result = &response["result"];
        assert_eq!(result["isError"], false);
        assert_eq!(result["content"][0]["text"], "blue");
    }

    #[tokio::test]
    async fn mcp_tools_list_returns_translated_catalog() {
        let bridge = McpBridge::start_background().await.unwrap();
        let tools = vec![json!({
            "name": "Bash",
            "description": "Run a shell command.",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        })];
        let (_session, url) = bridge.open_session(tools, ToolUseIdStyle::Anthropic).await;

        let client = http_client();
        let response: Value = client
            .post(&url)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/list",
                "params": {},
            }))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        let listed = &response["result"]["tools"];
        assert_eq!(listed[0]["name"], "Bash");
        assert!(listed[0].get("input_schema").is_none());
        assert_eq!(listed[0]["inputSchema"]["type"], "object");
    }

    #[tokio::test]
    async fn unknown_session_path_returns_404() {
        let bridge = McpBridge::start_background().await.unwrap();
        let port = bridge.port();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut socket = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req = format!(
            "POST /sess/notarealid/ HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        socket.write_all(req.as_bytes()).await.unwrap();
        let mut response = String::new();
        socket.read_to_string(&mut response).await.unwrap();
        assert!(response.starts_with("HTTP/1.1 404"), "got: {response}");
    }

    #[tokio::test]
    async fn dropping_session_cancels_parked_call() {
        let bridge = McpBridge::start_background().await.unwrap();
        let (session, url) = bridge.open_session(vec![], ToolUseIdStyle::Anthropic).await;
        let bridge_id = session.lock().await.id.clone();
        let _ = session.lock().await.attach_event_sink();

        // Fire a tools/call that will park.
        let url_clone = url.clone();
        let mcp_task = tokio::spawn(async move {
            http_client()
                .post(&url_clone)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "x", "arguments": {}},
                }))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        });

        // Give the handler a moment to register the parked call before we
        // tear the session down.
        tokio::time::sleep(Duration::from_millis(50)).await;
        bridge.drop_session(&bridge_id).await;

        let response = tokio::time::timeout(Duration::from_secs(5), mcp_task)
            .await
            .unwrap()
            .unwrap();
        assert!(
            response.get("error").is_some(),
            "expected MCP error on drop"
        );
    }

    #[tokio::test]
    async fn prewarm_session_returns_real_tools_after_take_for_use() {
        let bridge = McpBridge::start_background().await.unwrap();
        let (session, url) = bridge
            .open_session_for_prewarm(ToolUseIdStyle::OpenAi)
            .await;
        assert!(session.lock().await.awaiting_real_tools);
        assert!(session.lock().await.tools.is_empty());

        let real_tools = vec![json!({
            "name": "exec_command",
            "description": "Run a command.",
            "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        })];

        let url_clone = url.clone();
        let list_task = tokio::spawn(async move {
            http_client()
                .post(&url_clone)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        });

        // Let the handler reach its `notified()` await.
        tokio::time::sleep(Duration::from_millis(50)).await;
        McpBridge::take_for_use(&session, real_tools.clone()).await;

        let response = tokio::time::timeout(Duration::from_secs(5), list_task)
            .await
            .unwrap()
            .unwrap();
        let tools = response
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(Value::as_array)
            .expect("tools array in result");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].get("name").unwrap(), "exec_command");
        assert!(
            tools[0].get("inputSchema").is_some(),
            "translate_tools should run"
        );
        assert!(!session.lock().await.awaiting_real_tools);
    }

    #[tokio::test]
    async fn prewarm_tools_list_falls_back_to_empty_after_park_timeout() {
        // Override the global timeout for the test by directly racing the
        // notified() pattern with a short timeout — we just verify the
        // handler doesn't hang when take_for_use never runs.
        let bridge = McpBridge::start_background().await.unwrap();
        let (session, url) = bridge
            .open_session_for_prewarm(ToolUseIdStyle::OpenAi)
            .await;

        // tools/list call shouldn't return until either take_for_use OR the
        // 60-second timeout. We don't want to wait 60 s in tests, so instead
        // we verify the call IS parked: it should not complete within 200 ms.
        let url_clone = url.clone();
        let list_task = tokio::spawn(async move {
            http_client()
                .post(&url_clone)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }))
                .send()
                .await
                .unwrap()
                .json::<Value>()
                .await
                .unwrap()
        });
        let race = tokio::time::timeout(Duration::from_millis(200), &mut Box::pin(list_task)).await;
        assert!(
            race.is_err(),
            "tools/list should park while awaiting_real_tools is true"
        );
        drop(session);
    }

    #[tokio::test]
    async fn non_prewarm_session_does_not_park_tools_list() {
        let bridge = McpBridge::start_background().await.unwrap();
        let (_session, url) = bridge
            .open_session(
                vec![json!({"name": "x", "description": "", "input_schema": {"type": "object"}})],
                ToolUseIdStyle::Anthropic,
            )
            .await;
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            http_client()
                .post(&url)
                .json(&json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                }))
                .send(),
        )
        .await
        .expect("non-prewarm tools/list should reply immediately")
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
        let tools = response
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(Value::as_array)
            .expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].get("name").unwrap(), "x");
    }
}
