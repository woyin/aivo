//! Minimal JSON-RPC 2.0 client over stdio for ACP (Agent Client Protocol)
//! agents such as `cursor-agent acp`. Framing is NDJSON: one JSON object per
//! `\n`-terminated line on the child's stdin/stdout.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::services::http_debug::{self, DebugEntry, Phase};

#[derive(Debug, Clone)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl std::fmt::Display for JsonRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON-RPC error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for JsonRpcError {}

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, JsonRpcError>>>>>;
type SessionMap = Arc<Mutex<HashMap<String, mpsc::UnboundedSender<PromptEvent>>>>;
type Writer = Arc<Mutex<ChildStdin>>;
/// Bounded ring of the child's recent stderr lines: drained so the pipe never
/// fills (a full stderr pipe blocks cursor-agent), kept for hang/crash errors.
type StderrTail = Arc<std::sync::Mutex<VecDeque<String>>>;

/// Timeout for a one-shot ACP `request` (initialize / session/new / set_model)
/// so a wedged cursor-agent doesn't hang the caller forever. The streaming
/// `session/prompt` path ([`AcpClient::start_prompt`]) is exempt — a real turn
/// can run for minutes.
const ACP_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Cap on retained stderr lines and per-line length for [`StderrTail`].
const STDERR_TAIL_LINES: usize = 40;
const STDERR_TAIL_LINE_CAP: usize = 512;
/// Per-client map from outbound-request id → send instant. Read by the reader
/// task when an inbound response arrives to compute the round-trip
/// `duration_ms` for the debug log. Scoped per `AcpClient` because each child
/// has its own `next_id` counter starting at 1, so a global map would collide
/// across prewarmed sessions.
type RequestTimings = Arc<std::sync::Mutex<HashMap<u64, Instant>>>;

/// Caller-supplied policy for server-initiated `session/request_permission`
/// requests. Invoked from the reader task with the request `params` (owned, so
/// the returned future can be `'static`). It may `await` a decision — e.g. an
/// interactive permission card — because while it's pending the agent is itself
/// blocked on this permission request, so the reader pausing is harmless. `Send
/// + Sync` so it rides on the reader task.
pub type PermissionFn =
    Arc<dyn Fn(Value) -> futures::future::BoxFuture<'static, PermissionDecision> + Send + Sync>;

/// Handler for cursor's server-initiated `cursor/*` extension methods.
/// `Some(result)` replies with that JSON-RPC `result`; `None` falls through to
/// `-32601`. Shares [`PermissionFn`]'s reader contract for blocking handlers.
pub type ExtMethodFn =
    Arc<dyn Fn(String, Value) -> futures::future::BoxFuture<'static, Option<Value>> + Send + Sync>;

/// Idle `session/update` listener, installed at spawn so notifications that
/// arrive before any `session/prompt` (e.g. `available_commands_update`) are
/// not dropped. Invoked for every `session/update`, including during a turn.
pub type SessionUpdateFn = Arc<dyn Fn(Value) + Send + Sync>;

/// One of the two `outcome.outcome` shapes the ACP spec defines for
/// `session/request_permission` replies. The encoder picks a matching
/// `optionId` from the `options` array carried in the request — preferring
/// the first option whose `kind` matches the decision, falling back to a
/// literal `allow-once` / `reject-once` id, then to `cancelled`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionDecision {
    Allow,
    Reject,
}

/// Event delivered to a session prompt subscriber, ordered as it arrives.
#[derive(Debug)]
pub enum PromptEvent {
    Update(Value),
    Done(Result<Value, JsonRpcError>),
}

pub struct PromptStream {
    rx: mpsc::UnboundedReceiver<PromptEvent>,
}

impl PromptStream {
    pub async fn next(&mut self) -> Option<PromptEvent> {
        self.rx.recv().await
    }
}

pub struct AcpClient {
    writer: Writer,
    next_id: AtomicU64,
    pending: PendingMap,
    session_handlers: SessionMap,
    request_timings: RequestTimings,
    stderr_tail: StderrTail,
    child: Arc<Mutex<Option<Child>>>,
    _reader: JoinHandle<()>,
    _stderr_drain: JoinHandle<()>,
}

/// Spawned ACP children, reaped explicitly: aivo exits via `process::exit`,
/// which skips `kill_on_drop`, and cursor-agent ignores stdin EOF.
static ACP_CHILDREN: std::sync::Mutex<Vec<std::sync::Weak<Mutex<Option<Child>>>>> =
    std::sync::Mutex::new(Vec::new());

fn register_acp_child(slot: &Arc<Mutex<Option<Child>>>) {
    let mut reg = ACP_CHILDREN.lock().unwrap();
    reg.retain(|w| w.strong_count() > 0);
    reg.push(Arc::downgrade(slot));
}

/// Kill every live ACP child (see [`ACP_CHILDREN`]).
pub async fn kill_all_acp_children() {
    let live: Vec<_> = {
        let mut reg = ACP_CHILDREN.lock().unwrap();
        let live = reg.iter().filter_map(|w| w.upgrade()).collect();
        reg.clear();
        live
    };
    kill_children(live).await;
}

async fn kill_children(slots: Vec<Arc<Mutex<Option<Child>>>>) {
    for slot in slots {
        if let Some(mut child) = slot.lock().await.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

impl AcpClient {
    /// Spawn with the default permission policy (graceful reject). Most
    /// callers want [`spawn_with_permission_policy`] so they can opt into
    /// allowing tool execution.
    pub async fn spawn(cmd: Command) -> Result<Self> {
        Self::spawn_with_permission_policy(
            cmd,
            Arc::new(|_| Box::pin(async { PermissionDecision::Reject })),
        )
        .await
    }

    pub async fn spawn_with_permission_policy(
        cmd: Command,
        permission_fn: PermissionFn,
    ) -> Result<Self> {
        Self::spawn_with_handlers(cmd, permission_fn, None).await
    }

    /// Like [`spawn_with_permission_policy`] but also installs an
    /// [`ExtMethodFn`] for cursor's server-initiated `cursor/*` methods.
    pub async fn spawn_with_handlers(
        cmd: Command,
        permission_fn: PermissionFn,
        ext_method_fn: Option<ExtMethodFn>,
    ) -> Result<Self> {
        Self::spawn_with_session_hook(cmd, permission_fn, ext_method_fn, None).await
    }

    /// Like [`spawn_with_handlers`] plus an idle [`SessionUpdateFn`].
    pub(crate) async fn spawn_with_session_hook(
        mut cmd: Command,
        permission_fn: PermissionFn,
        ext_method_fn: Option<ExtMethodFn>,
        session_update_fn: Option<SessionUpdateFn>,
    ) -> Result<Self> {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut child = cmd.spawn().context("failed to spawn ACP child process")?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("ACP child has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("ACP child has no stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("ACP child has no stderr"))?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let sessions: SessionMap = Arc::new(Mutex::new(HashMap::new()));
        let writer: Writer = Arc::new(Mutex::new(stdin));
        let request_timings: RequestTimings = Arc::new(std::sync::Mutex::new(HashMap::new()));
        let stderr_tail: StderrTail = Arc::new(std::sync::Mutex::new(VecDeque::new()));

        let reader = spawn_reader(
            BufReader::new(stdout),
            pending.clone(),
            sessions.clone(),
            writer.clone(),
            request_timings.clone(),
            permission_fn,
            ext_method_fn,
            session_update_fn,
        );
        let stderr_drain = spawn_stderr_drain(BufReader::new(stderr), stderr_tail.clone());

        let child_slot = Arc::new(Mutex::new(Some(child)));
        register_acp_child(&child_slot);
        Ok(Self {
            writer,
            next_id: AtomicU64::new(1),
            pending,
            session_handlers: sessions,
            request_timings,
            stderr_tail,
            child: child_slot,
            _reader: reader,
            _stderr_drain: stderr_drain,
        })
    }

    /// The child's recent stderr lines, joined newest-last for a crash error.
    pub fn recent_stderr(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|q| q.iter().cloned().collect::<Vec<_>>().join("\n"))
            .unwrap_or_default()
    }

    /// Append the child's stderr tail to `msg` so a transport error carries
    /// the agent's own diagnostics.
    fn with_stderr_context(&self, msg: String) -> anyhow::Error {
        let tail = self.recent_stderr();
        if tail.is_empty() {
            anyhow!(msg)
        } else {
            anyhow!("{msg}\ncursor-agent stderr:\n{tail}")
        }
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        write_frame(&self.writer, &frame, Some(&self.request_timings))
            .await
            .inspect_err(|_| {
                // Reclaim the pending slot so it doesn't leak.
                let pending = self.pending.clone();
                tokio::spawn(async move {
                    pending.lock().await.remove(&id);
                });
            })?;

        match tokio::time::timeout(ACP_REQUEST_TIMEOUT, rx).await {
            Ok(Ok(Ok(v))) => Ok(v),
            Ok(Ok(Err(rpc_err))) => {
                Err(anyhow!(rpc_err).context(format!("ACP method `{method}` failed")))
            }
            Ok(Err(_)) => Err(self.with_stderr_context(format!(
                "ACP child closed connection before `{method}` returned"
            ))),
            Err(_elapsed) => {
                // Wedged child: reclaim the pending slot, surface a timeout.
                let pending = self.pending.clone();
                tokio::spawn(async move {
                    pending.lock().await.remove(&id);
                });
                Err(self.with_stderr_context(format!(
                    "ACP method `{method}` timed out after {ACP_REQUEST_TIMEOUT:?}"
                )))
            }
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
        write_frame(&self.writer, &frame, None).await
    }

    /// Subscribe to `session/update` notifications for `session_id` and send a
    /// `session/prompt` request. Updates are streamed in arrival order; the
    /// final `Done` event carries the prompt response (or the JSON-RPC error).
    pub async fn start_prompt(&self, session_id: &str, prompt: Vec<Value>) -> Result<PromptStream> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.session_handlers
            .lock()
            .await
            .insert(session_id.to_string(), tx.clone());

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (resp_tx, resp_rx) = oneshot::channel();
        self.pending.lock().await.insert(id, resp_tx);

        let frame = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": prompt},
        });
        if let Err(e) = write_frame(&self.writer, &frame, Some(&self.request_timings)).await {
            self.pending.lock().await.remove(&id);
            self.session_handlers.lock().await.remove(session_id);
            return Err(e);
        }

        // Forward the response into the same channel so callers see ordered
        // events: all session/update notifications, then exactly one Done.
        let sessions = self.session_handlers.clone();
        let session_id = session_id.to_string();
        tokio::spawn(async move {
            let result = match resp_rx.await {
                Ok(r) => r,
                Err(_) => Err(JsonRpcError {
                    code: -32603,
                    message: "ACP child closed connection before session/prompt returned".into(),
                    data: None,
                }),
            };
            let _ = tx.send(PromptEvent::Done(result));
            // Only clear the handler if it's still the one this prompt
            // registered: a cancelled-then-restarted turn reuses the session
            // id, and evicting the newer prompt's sender would drop its updates.
            let mut guard = sessions.lock().await;
            if guard
                .get(&session_id)
                .is_some_and(|current| current.same_channel(&tx))
            {
                guard.remove(&session_id);
            }
        });

        Ok(PromptStream { rx })
    }

    pub async fn shutdown(&self) {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

/// Drain the child's stderr into a bounded tail. An unread stderr pipe fills
/// (~64 KB) and blocks cursor-agent's next `write()`, freezing it mid-turn.
fn spawn_stderr_drain(
    mut stderr: BufReader<tokio::process::ChildStderr>,
    tail: StderrTail,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut line = String::new();
        loop {
            line.clear();
            match stderr.read_line(&mut line).await {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            let mut entry = trimmed.to_string();
            if entry.len() > STDERR_TAIL_LINE_CAP {
                entry.truncate(STDERR_TAIL_LINE_CAP);
                entry.push('…');
            }
            if let Ok(mut q) = tail.lock() {
                if q.len() == STDERR_TAIL_LINES {
                    q.pop_front();
                }
                q.push_back(entry);
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_reader(
    mut stdout: BufReader<tokio::process::ChildStdout>,
    pending: PendingMap,
    sessions: SessionMap,
    writer: Writer,
    request_timings: RequestTimings,
    permission_fn: PermissionFn,
    ext_method_fn: Option<ExtMethodFn>,
    session_update_fn: Option<SessionUpdateFn>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut line = String::new();
        loop {
            line.clear();
            match stdout.read_line(&mut line).await {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };
            // Pair inbound responses with the outbound `Instant` recorded
            // when the request was sent so the log entry carries an accurate
            // round-trip `duration_ms`. Notifications and server-initiated
            // requests have no paired start, so duration stays `None`.
            let duration_ms = inbound_duration_ms(&value, &request_timings);
            log_acp_frame(Direction::Inbound, &value, duration_ms).await;
            dispatch_inbound(
                value,
                &pending,
                &sessions,
                &writer,
                &permission_fn,
                &ext_method_fn,
                &session_update_fn,
            )
            .await;
        }

        // EOF: surface a clear error to anyone still waiting, and drop any
        // unmatched timing entries so the child's untracked requests don't
        // accumulate in memory until the process exits.
        let mut pending_guard = pending.lock().await;
        for (_, tx) in pending_guard.drain() {
            let _ = tx.send(Err(JsonRpcError {
                code: -32603,
                message: "ACP child stdout closed".into(),
                data: None,
            }));
        }
        if let Ok(mut timings) = request_timings.lock() {
            timings.clear();
        }
    })
}

/// Look up the recorded send-time for the inbound frame's JSON-RPC id and
/// compute the elapsed milliseconds. Returns `None` for frames that aren't
/// responses to one of our outbound requests (notifications, server-initiated
/// requests, unknown ids).
fn inbound_duration_ms(value: &Value, request_timings: &RequestTimings) -> Option<u64> {
    let id = value.get("id").and_then(value_to_u64)?;
    if value.get("method").is_some() {
        // Has both id and method ⇒ server-initiated request, not a response.
        return None;
    }
    let started = request_timings.lock().ok()?.remove(&id)?;
    Some(started.elapsed().as_millis() as u64)
}

async fn dispatch_inbound(
    value: Value,
    pending: &PendingMap,
    sessions: &SessionMap,
    writer: &Writer,
    permission_fn: &PermissionFn,
    ext_method_fn: &Option<ExtMethodFn>,
    session_update_fn: &Option<SessionUpdateFn>,
) {
    // Keep the id as a raw `Value`: server-initiated requests may carry a
    // string/negative id (JSON-RPC-legal) that must be echoed verbatim —
    // dropping it would leave `session/request_permission` unanswered.
    let raw_id = value.get("id").cloned();
    let method = value.get("method").and_then(Value::as_str);

    match (raw_id, method) {
        (Some(id), None) => {
            // Response to one of our requests (our ids are always u64).
            if let Some(id) = value_to_u64(&id)
                && let Some(tx) = pending.lock().await.remove(&id)
            {
                let _ = tx.send(parse_result_or_error(&value));
            }
        }
        (None, Some(method)) => {
            // Notification.
            handle_notification(method, value.get("params"), sessions, session_update_fn).await;
        }
        (Some(id), Some("session/request_permission")) => {
            let params = value.get("params").cloned().unwrap_or(Value::Null);
            let decision = permission_fn(params.clone()).await;
            let resp = build_permission_response(id, &params, decision);
            let _ = write_frame(writer, &resp, None).await;
        }
        (Some(id), Some(method)) => {
            // Extension handler gets first crack; declining falls through to -32601.
            let handled = match ext_method_fn {
                Some(ext) => {
                    let params = value.get("params").cloned().unwrap_or(Value::Null);
                    ext(method.to_string(), params).await
                }
                None => None,
            };
            let resp = match handled {
                Some(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                None => json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("aivo ACP client does not implement `{method}`"),
                    },
                }),
            };
            let _ = write_frame(writer, &resp, None).await;
        }
        (None, None) => {}
    }
}

/// Build a `session/request_permission` reply. ACP expects:
///
/// ```json
/// {"outcome": {"outcome": "selected", "optionId": "<id>"}}
/// ```
///
/// or `{"outcome": {"outcome": "cancelled"}}`. We prefer to echo an `optionId`
/// the agent already offered (matched by `kind`) so the agent can map the
/// reply to one of its own UI options; if no compatible option is offered we
/// fall back to `cancelled`.
fn build_permission_response(id: Value, params: &Value, decision: PermissionDecision) -> Value {
    let preferred_kind = match decision {
        PermissionDecision::Allow => "allow_once",
        PermissionDecision::Reject => "reject_once",
    };
    let option_id = params
        .get("options")
        .and_then(Value::as_array)
        .and_then(|opts| {
            opts.iter()
                .find(|o| o.get("kind").and_then(Value::as_str) == Some(preferred_kind))
                .and_then(|o| o.get("optionId").and_then(Value::as_str))
                .map(str::to_string)
        });
    let outcome = match option_id {
        Some(id) => json!({"outcome": "selected", "optionId": id}),
        None => json!({"outcome": "cancelled"}),
    };
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"outcome": outcome},
    })
}

async fn handle_notification(
    method: &str,
    params: Option<&Value>,
    sessions: &SessionMap,
    session_update_fn: &Option<SessionUpdateFn>,
) {
    if method != "session/update" {
        return;
    }
    let Some(params) = params else {
        return;
    };
    if let Some(cb) = session_update_fn {
        cb(params.clone());
    }
    let Some(session_id) = params.get("sessionId").and_then(Value::as_str) else {
        return;
    };
    let tx = sessions.lock().await.get(session_id).cloned();
    if let Some(tx) = tx {
        let _ = tx.send(PromptEvent::Update(params.clone()));
    }
}

fn parse_result_or_error(value: &Value) -> Result<Value, JsonRpcError> {
    if let Some(err) = value.get("error") {
        let code = err.get("code").and_then(Value::as_i64).unwrap_or(-1);
        let message = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("<no message>")
            .to_string();
        let data = err.get("data").cloned();
        Err(JsonRpcError {
            code,
            message,
            data,
        })
    } else {
        Ok(value.get("result").cloned().unwrap_or(Value::Null))
    }
}

fn value_to_u64(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_i64().map(|n| n as u64))
}

async fn write_frame<W>(
    writer: &Arc<Mutex<W>>,
    value: &Value,
    request_timings: Option<&RequestTimings>,
) -> Result<()>
where
    W: tokio::io::AsyncWrite + Unpin + Send,
{
    let mut buf = serde_json::to_vec(value).context("encode JSON-RPC frame")?;
    buf.push(b'\n');

    // Record the send instant *before* the bytes hit the pipe so a fast
    // response from the child can be paired with the timing entry. Inserting
    // after flush races the reader: the child can read+respond between
    // flush and insert, leaving the response with `duration_ms: None` and
    // orphaning the entry forever. Rollback on write failure keeps the map
    // clean.
    let timing_key = request_tracking_key(value, request_timings);
    with_timings(timing_key, |map, id| {
        map.insert(id, Instant::now());
    });

    let write_result: Result<()> = async {
        let mut w = writer.lock().await;
        w.write_all(&buf).await.context("write ACP frame")?;
        w.flush().await.context("flush ACP frame")?;
        Ok(())
    }
    .await;

    if let Err(e) = write_result {
        with_timings(timing_key, |map, id| {
            map.remove(&id);
        });
        return Err(e);
    }

    log_acp_frame(Direction::Outbound, value, None).await;
    Ok(())
}

/// Runs `f` against the timings map for `key`'s id, swallowing the lock-poison
/// case. No-op when `key` is `None` (untracked frame) or the mutex is poisoned.
fn with_timings<F>(key: Option<(&RequestTimings, u64)>, f: F)
where
    F: FnOnce(&mut HashMap<u64, Instant>, u64),
{
    if let Some((timings, id)) = key
        && let Ok(mut map) = timings.lock()
    {
        f(&mut map, id);
    }
}

/// Returns `Some((timings, id))` when this frame is an outbound JSON-RPC
/// request whose round-trip latency we should record. Notifications (no
/// `id`) and responses to server-initiated requests (no `method`) are
/// untracked and yield `None`.
fn request_tracking_key<'a>(
    value: &Value,
    request_timings: Option<&'a RequestTimings>,
) -> Option<(&'a RequestTimings, u64)> {
    let timings = request_timings?;
    let id = value.get("id").and_then(value_to_u64)?;
    value.get("method").and_then(Value::as_str)?;
    Some((timings, id))
}

/// Direction tag for a logged ACP frame.
#[derive(Copy, Clone)]
enum Direction {
    Inbound,
    Outbound,
}

const ACP_LOG_BODY_LIMIT: usize = 64 * 1024;

/// ACP `session/update` kinds whose body is dropped under `AIVO_DEBUG_QUIET`.
/// These two account for the bulk of bytes during a streaming turn while
/// carrying no actionable signal post-hoc (the agent's prose, not its tool
/// activity).
const QUIET_REDACTED_UPDATES: &[&str] = &["agent_message_chunk", "agent_thought_chunk"];

/// Emit one log entry per ACP frame, labeled by JSON-RPC frame shape
/// (request/response/notification) rather than transport direction so
/// `aivo logs` filters match how a reader thinks about the protocol.
/// No-op when `--debug` is not in effect.
async fn log_acp_frame(direction: Direction, value: &Value, duration_ms: Option<u64>) {
    let Some(logger) = http_debug::global() else {
        return;
    };
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let id = format!("acp-{n:x}");
    logger
        .log(build_acp_debug_entry(id, direction, value, duration_ms))
        .await;
}

/// JSON-RPC frame kind, derived from the presence/absence of `id` and `method`.
/// Maps onto `Phase` in the debug log: requests/notifications keep their
/// JSON-RPC method name as the entry's `method` field; responses use the
/// `result` / `error` label so the log reader can grep on it directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FrameKind {
    Request,
    Response,
    Notification,
}

fn classify_frame(value: &Value) -> FrameKind {
    let has_id = value.get("id").is_some();
    let has_method = value.get("method").is_some();
    match (has_id, has_method) {
        (true, true) => FrameKind::Request,
        (true, false) => FrameKind::Response,
        (false, _) => FrameKind::Notification,
    }
}

/// Extracts the `sessionUpdate` kind from an ACP `session/update` notification
/// envelope. Returns `None` for non-update frames.
fn session_update_kind(value: &Value) -> Option<&str> {
    value
        .get("params")
        .and_then(|p| p.get("update"))
        .and_then(|u| u.get("sessionUpdate"))
        .and_then(Value::as_str)
}

fn debug_quiet_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("AIVO_DEBUG_QUIET")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Pure builder for the JSONL schema; keeps `log_acp_frame` testable without a
/// real logger. `id` is injected by the caller so tests can pin it.
fn build_acp_debug_entry(
    id: String,
    direction: Direction,
    value: &Value,
    duration_ms: Option<u64>,
) -> DebugEntry {
    let kind = classify_frame(value);
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| {
            match value.get("error") {
                Some(_) => "error",
                None if value.get("result").is_some() => "result",
                _ => "frame",
            }
            .to_string()
        });
    let session_update = session_update_kind(value);
    let body_str = match session_update {
        // Preserve the envelope + sessionUpdate label so the entry stays
        // greppable; drop only the verbose text payload.
        Some(kind) if debug_quiet_enabled() && QUIET_REDACTED_UPDATES.contains(&kind) => {
            format!("[quiet: {kind}]")
        }
        _ => value.to_string(),
    };
    let body = if body_str.len() > ACP_LOG_BODY_LIMIT {
        let mut truncated = body_str[..ACP_LOG_BODY_LIMIT].to_string();
        truncated.push_str("…[truncated]");
        truncated
    } else {
        body_str
    };
    let phase = match kind {
        FrameKind::Request => Phase::Request,
        FrameKind::Response => Phase::Response,
        FrameKind::Notification => Phase::Notification,
    };
    let url = match direction {
        Direction::Outbound => "acp://outbound",
        Direction::Inbound => "acp://inbound",
    };
    // Body lives in `request_body` for anything we sent and `response_body`
    // for anything we received. The slot tracks transport direction; the
    // phase label tracks JSON-RPC frame kind.
    let (request_body, response_body) = match direction {
        Direction::Outbound => (Some(body), None),
        Direction::Inbound => (None, Some(body)),
    };
    // Surface JSON-RPC errors via the dedicated `error` field so post-hoc
    // log queries can `select * where error is not null` without parsing
    // the body envelope.
    let error = value
        .get("error")
        .and_then(|err| err.get("message").and_then(Value::as_str))
        .map(|msg| {
            let code = value
                .get("error")
                .and_then(|e| e.get("code"))
                .and_then(Value::as_i64)
                .unwrap_or(0);
            format!("JSON-RPC {code}: {msg}")
        });
    DebugEntry {
        ts: chrono::Utc::now().to_rfc3339(),
        id,
        phase,
        method,
        url: url.to_string(),
        status: None,
        duration_ms,
        request_headers: BTreeMap::new(),
        request_body,
        response_headers: BTreeMap::new(),
        response_body,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::process::Command as TokioCommand;

    fn fake_acp_script(script: &str) -> TokioCommand {
        // Build a sh -c invocation that echoes a scripted set of JSON-RPC
        // frames, optionally reading from stdin between them. Tests pass
        // pre-baked stdout sequences and rely on framing being NDJSON.
        let mut cmd = TokioCommand::new("sh");
        cmd.arg("-c").arg(script);
        cmd
    }

    #[tokio::test]
    async fn spawn_registers_child_in_reap_registry() {
        let client = AcpClient::spawn(fake_acp_script("sleep 30")).await.unwrap();
        // Pointer-match our entry — other tests spawn concurrently.
        let registered = ACP_CHILDREN
            .lock()
            .unwrap()
            .iter()
            .filter_map(|w| w.upgrade())
            .any(|slot| Arc::ptr_eq(&slot, &client.child));
        assert!(registered);
        client.shutdown().await;
    }

    #[tokio::test]
    async fn kill_children_reaps_and_empties_slot() {
        let mut cmd = fake_acp_script("sleep 30");
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let slot = Arc::new(Mutex::new(Some(cmd.spawn().unwrap())));
        kill_children(vec![slot.clone()]).await;
        assert!(slot.lock().await.is_none());
    }

    #[tokio::test]
    async fn responds_to_request_by_id() {
        let cmd = fake_acp_script(
            r#"read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let client = AcpClient::spawn(cmd).await.unwrap();
        let result = client.request("ping", json!({"x": 1})).await.unwrap();
        assert_eq!(result["ok"], json!(true));
    }

    #[tokio::test]
    async fn surfaces_json_rpc_error() {
        let cmd = fake_acp_script(
            r#"read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"nope"}}'"#,
        );
        let client = AcpClient::spawn(cmd).await.unwrap();
        let err = client.request("missing", json!({})).await.unwrap_err();
        assert!(err.to_string().contains("`missing`"));
        assert!(format!("{err:#}").contains("nope"));
    }

    #[tokio::test]
    async fn start_prompt_streams_updates_then_done() {
        // The script first reads one line (our session/prompt request), then
        // emits two session/update notifications followed by the response.
        let cmd = fake_acp_script(
            r#"read line
printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hello "}}}}'
printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"world"}}}}'
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"stopReason":"end_turn"}}'"#,
        );
        let client = AcpClient::spawn(cmd).await.unwrap();
        let mut stream = client
            .start_prompt("s1", vec![json!({"type":"text","text":"hi"})])
            .await
            .unwrap();

        let mut texts: Vec<String> = Vec::new();
        let mut stop: Option<String> = None;
        while let Some(ev) = tokio::time::timeout(Duration::from_secs(3), stream.next())
            .await
            .ok()
            .flatten()
        {
            match ev {
                PromptEvent::Update(v) => {
                    if let Some(t) = v
                        .get("update")
                        .and_then(|u| u.get("content"))
                        .and_then(|c| c.get("text"))
                        .and_then(Value::as_str)
                    {
                        texts.push(t.to_string());
                    }
                }
                PromptEvent::Done(r) => {
                    stop = r
                        .as_ref()
                        .ok()
                        .and_then(|res| res.get("stopReason"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    break;
                }
            }
        }
        assert_eq!(texts.join(""), "Hello world");
        assert_eq!(stop.as_deref(), Some("end_turn"));
    }

    #[tokio::test]
    async fn ignores_updates_for_unknown_session() {
        let cmd = fake_acp_script(
            r#"printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"ghost","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"drop"}}}}'
read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let client = AcpClient::spawn(cmd).await.unwrap();
        let r = client.request("ping", json!({})).await.unwrap();
        assert_eq!(r["ok"], json!(true));
    }

    #[tokio::test]
    async fn session_update_hook_sees_updates_without_a_prompt() {
        let cmd = fake_acp_script(
            r#"printf '%s\n' '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s1","update":{"sessionUpdate":"available_commands_update","availableCommands":[{"name":"copy-request-id","description":"Copy the last request ID to clipboard"}]}}}'
read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let seen: Arc<std::sync::Mutex<Option<Value>>> = Arc::new(std::sync::Mutex::new(None));
        let seen_in = seen.clone();
        let hook: SessionUpdateFn = Arc::new(move |params| {
            *seen_in.lock().unwrap() = Some(params);
        });
        let permission: PermissionFn = Arc::new(|_| Box::pin(async { PermissionDecision::Reject }));
        let client = AcpClient::spawn_with_session_hook(cmd, permission, None, Some(hook))
            .await
            .unwrap();
        let r = tokio::time::timeout(Duration::from_secs(3), client.request("ping", json!({})))
            .await
            .expect("request timed out")
            .expect("request failed");
        assert_eq!(r["ok"], json!(true));
        let v = seen
            .lock()
            .unwrap()
            .clone()
            .expect("session update hook was not called");
        assert_eq!(
            v["update"]["availableCommands"][0]["name"],
            "copy-request-id"
        );
    }

    #[test]
    fn outbound_debug_entry_carries_method_and_request_body() {
        let frame = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/prompt",
            "params": {"sessionId": "s", "prompt": [{"type":"text","text":"hi"}]},
        });
        let entry = build_acp_debug_entry("acp-7".into(), Direction::Outbound, &frame, None);
        assert_eq!(entry.id, "acp-7");
        assert!(matches!(entry.phase, Phase::Request));
        assert_eq!(entry.url, "acp://outbound");
        assert_eq!(entry.method, "session/prompt");
        assert!(entry.request_body.as_deref().unwrap().contains("\"hi\""));
        assert!(entry.response_body.is_none());
    }

    #[test]
    fn inbound_debug_entry_uses_result_method_label_when_no_method_present() {
        let frame = json!({"jsonrpc":"2.0","id":1,"result":{"ok":true}});
        let entry = build_acp_debug_entry("acp-1".into(), Direction::Inbound, &frame, Some(42));
        assert!(matches!(entry.phase, Phase::Response));
        assert_eq!(entry.url, "acp://inbound");
        assert_eq!(entry.method, "result");
        assert_eq!(entry.duration_ms, Some(42));
        assert!(
            entry
                .response_body
                .as_deref()
                .unwrap()
                .contains("\"ok\":true")
        );
        assert!(entry.request_body.is_none());
        assert!(entry.error.is_none());
    }

    #[test]
    fn inbound_debug_entry_marks_error_frames() {
        let frame = json!({"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"nope"}});
        let entry = build_acp_debug_entry("acp-1".into(), Direction::Inbound, &frame, None);
        assert_eq!(entry.method, "error");
        // JSON-RPC errors get surfaced in the dedicated `error` field so the
        // log reader doesn't have to parse the body envelope to spot failures.
        assert_eq!(
            entry.error.as_deref(),
            Some("JSON-RPC -32601: nope"),
            "error message should encode code + message"
        );
    }

    #[test]
    fn inbound_notification_is_labeled_notification_not_response() {
        // Regression: pre-fix, session/update notifications were labeled
        // Phase::Response, conflating them with replies to our requests.
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s1", "update": {"sessionUpdate": "tool_call"}},
        });
        let entry = build_acp_debug_entry("acp-n".into(), Direction::Inbound, &frame, None);
        assert!(matches!(entry.phase, Phase::Notification));
        assert_eq!(entry.method, "session/update");
        assert!(entry.duration_ms.is_none());
    }

    #[test]
    fn outbound_response_to_server_request_is_labeled_response() {
        // We reply to server-initiated requests (e.g. session/request_permission)
        // by writing a frame with id+result. The phase should reflect that
        // it's a response, even though we're the one writing it.
        let frame = json!({"jsonrpc":"2.0","id":42,"result":{"outcome":{"outcome":"cancelled"}}});
        let entry = build_acp_debug_entry("acp-r".into(), Direction::Outbound, &frame, None);
        assert!(matches!(entry.phase, Phase::Response));
        assert_eq!(entry.url, "acp://outbound");
        assert_eq!(entry.method, "result");
        // Outbound frames put the body in `request_body` regardless of phase.
        assert!(entry.request_body.is_some());
        assert!(entry.response_body.is_none());
    }

    #[test]
    fn classify_frame_distinguishes_all_three_kinds() {
        assert_eq!(
            classify_frame(&json!({"id":1,"method":"x"})),
            FrameKind::Request
        );
        assert_eq!(
            classify_frame(&json!({"id":1,"result":{}})),
            FrameKind::Response
        );
        assert_eq!(
            classify_frame(&json!({"method":"session/update"})),
            FrameKind::Notification
        );
    }

    #[test]
    fn permission_response_picks_reject_once_option_id_when_offered() {
        let params = json!({
            "sessionId": "s",
            "options": [
                {"kind": "allow_once", "name": "Allow once", "optionId": "allow-once"},
                {"kind": "allow_always", "name": "Allow always", "optionId": "allow-always"},
                {"kind": "reject_once", "name": "Reject", "optionId": "reject-once"},
            ],
            "toolCall": {"kind": "execute"},
        });
        let resp = build_permission_response(json!(42), &params, PermissionDecision::Reject);
        assert_eq!(resp["id"], 42);
        assert_eq!(resp["result"]["outcome"]["outcome"], "selected");
        assert_eq!(resp["result"]["outcome"]["optionId"], "reject-once");
    }

    #[test]
    fn permission_response_picks_allow_once_option_id_when_offered() {
        let params = json!({
            "options": [
                {"kind": "allow_once", "name": "Allow once", "optionId": "allow-once"},
                {"kind": "reject_once", "name": "Reject", "optionId": "reject-once"},
            ],
        });
        let resp = build_permission_response(json!(1), &params, PermissionDecision::Allow);
        assert_eq!(resp["result"]["outcome"]["optionId"], "allow-once");
    }

    #[test]
    fn permission_response_echoes_non_u64_id_verbatim() {
        // A string-id server request must be answered with that exact id.
        let params = json!({"options": [
            {"kind": "reject_once", "optionId": "reject-once"},
        ]});
        let resp = build_permission_response(json!("req-7"), &params, PermissionDecision::Reject);
        assert_eq!(resp["id"], json!("req-7"));
        assert_eq!(resp["result"]["outcome"]["optionId"], "reject-once");
    }

    #[test]
    fn permission_response_falls_back_to_cancelled_when_no_matching_option() {
        // Some agents may omit allow/reject altogether — `cancelled` is the
        // spec-defined fallback the protocol guarantees.
        let params = json!({"options": [
            {"kind": "weird_custom", "optionId": "x"},
        ]});
        let resp = build_permission_response(json!(1), &params, PermissionDecision::Reject);
        assert_eq!(resp["result"]["outcome"]["outcome"], "cancelled");
        assert!(resp["result"]["outcome"].get("optionId").is_none());
    }

    #[test]
    fn huge_acp_frame_bodies_are_truncated_with_marker() {
        let huge: String = "x".repeat(ACP_LOG_BODY_LIMIT + 5_000);
        let frame = json!({"method": "noisy", "data": huge});
        let entry = build_acp_debug_entry("acp-x".into(), Direction::Outbound, &frame, None);
        let body = entry.request_body.unwrap();
        assert!(body.ends_with("…[truncated]"));
        assert!(body.len() < frame.to_string().len());
    }

    #[test]
    fn quiet_mode_redaction_covers_only_chunk_sessionupdate_kinds() {
        // The QUIET_REDACTED_UPDATES list is what `build_acp_debug_entry`
        // checks against; tool_call updates carry signal worth keeping
        // verbatim and must NOT be in the list.
        let chunk = json!({
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "x"}}},
        });
        let thought = json!({
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "agent_thought_chunk", "content": {"text": "x"}}},
        });
        let tool = json!({
            "method": "session/update",
            "params": {"update": {"sessionUpdate": "tool_call_update", "status": "in_progress"}},
        });
        assert!(QUIET_REDACTED_UPDATES.contains(&session_update_kind(&chunk).unwrap()));
        assert!(QUIET_REDACTED_UPDATES.contains(&session_update_kind(&thought).unwrap()));
        assert!(!QUIET_REDACTED_UPDATES.contains(&session_update_kind(&tool).unwrap()));
    }

    #[test]
    fn request_tracking_key_skips_notifications_and_responses() {
        let timings: RequestTimings = Arc::new(std::sync::Mutex::new(HashMap::new()));
        // Outbound request (id + method): tracked.
        let req = json!({"jsonrpc":"2.0","id":7,"method":"ping"});
        assert!(request_tracking_key(&req, Some(&timings)).is_some());
        // Outbound notification (method only): not tracked.
        let notif = json!({"jsonrpc":"2.0","method":"session/update","params":{}});
        assert!(request_tracking_key(&notif, Some(&timings)).is_none());
        // Outbound response to server-initiated request (id only): not tracked.
        let resp = json!({"jsonrpc":"2.0","id":42,"result":{}});
        assert!(request_tracking_key(&resp, Some(&timings)).is_none());
        // No timings map at all: not tracked.
        assert!(request_tracking_key(&req, None).is_none());
    }

    #[tokio::test]
    async fn write_frame_inserts_timing_before_bytes_hit_pipe() {
        // Regression: inserting AFTER write+flush raced the reader task —
        // a fast child could respond before the insert landed, and the
        // matching inbound entry was logged with `duration_ms: None`. The
        // insert must happen before the bytes are flushed.
        use tokio::io::AsyncReadExt;

        let (mut reader_side, writer_side) = tokio::io::duplex(1024);
        let writer = Arc::new(Mutex::new(writer_side));
        let timings: RequestTimings = Arc::new(std::sync::Mutex::new(HashMap::new()));

        let frame = json!({
            "jsonrpc": "2.0",
            "id": 42,
            "method": "ping",
            "params": {},
        });

        let writer_clone = writer.clone();
        let timings_clone = timings.clone();
        let frame_clone = frame.clone();
        let write_handle = tokio::spawn(async move {
            write_frame(&writer_clone, &frame_clone, Some(&timings_clone))
                .await
                .unwrap();
        });

        // Pull bytes off the pipe. The moment we observe ANY byte, the
        // entry MUST already be in the map — otherwise the reader-side
        // race window we're guarding against is still open.
        let mut buf = vec![0u8; 1024];
        let n = reader_side.read(&mut buf).await.expect("pipe read");
        assert!(n > 0, "expected bytes on the pipe");
        assert!(
            timings.lock().unwrap().contains_key(&42),
            "timing entry must be present BEFORE the writer's flush is observable to the reader"
        );

        write_handle.await.unwrap();
    }

    #[tokio::test]
    async fn write_frame_rolls_back_timing_on_write_error() {
        // Drop the reader side of the duplex so the writer's flush fails;
        // the insert must be removed so the map doesn't leak entries that
        // outlive the request.
        let (reader_side, writer_side) = tokio::io::duplex(1);
        drop(reader_side);
        let writer = Arc::new(Mutex::new(writer_side));
        let timings: RequestTimings = Arc::new(std::sync::Mutex::new(HashMap::new()));

        // Pad the frame so the single-byte buffer can't absorb it before
        // hitting the closed reader side.
        let frame = json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "ping",
            "params": {"pad": "x".repeat(4096)},
        });
        let result = write_frame(&writer, &frame, Some(&timings)).await;
        assert!(
            result.is_err(),
            "expected the write to fail once the reader half is closed"
        );
        assert!(
            !timings.lock().unwrap().contains_key(&99),
            "timing entry must be removed when the write fails so the map stays clean"
        );
    }

    #[tokio::test]
    async fn answers_server_initiated_request_with_method_not_found() {
        // Server sends a request first; we must reply so it doesn't block.
        // The script writes a server-initiated request, then reads our reply.
        let cmd = fake_acp_script(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":42,"method":"fs/read_text_file","params":{"path":"/tmp/x"}}'
read reply
printf '%s\n' "$reply"
read line; printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let client = AcpClient::spawn(cmd).await.unwrap();
        // Give the reader time to process the server request and write its
        // -32601 reply. Then issue a request; the fake echoes our reply on
        // its stdout, but since the id (42) doesn't match anything pending,
        // it's dropped. The follow-up request must still complete.
        let r = tokio::time::timeout(Duration::from_secs(3), client.request("ping", json!({})))
            .await
            .expect("request timed out")
            .expect("request failed");
        assert_eq!(r["ok"], json!(true));
    }

    #[tokio::test]
    async fn invokes_ask_question_handler_and_replies_with_its_result() {
        use std::sync::Mutex as StdMutex;
        // Consume our reply + the ping (order-independent), then answer the ping.
        // Don't echo: the generic ext arm would re-dispatch an echoed ping.
        let cmd = fake_acp_script(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":"aq1","method":"cursor/ask_question","params":{"questions":[{"id":"q1","prompt":"pick","options":[{"id":"o1","label":"A"}],"allowMultiple":false}]}}'
read our_reply
read ping_req
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let seen: Arc<StdMutex<Option<(String, Value)>>> = Arc::new(StdMutex::new(None));
        let seen_in = seen.clone();
        let ext_fn: ExtMethodFn = Arc::new(move |method: String, params: Value| {
            let seen = seen_in.clone();
            Box::pin(async move {
                *seen.lock().unwrap() = Some((method, params));
                Some(
                    json!({"outcome": {"outcome": "answered", "answers": [{"questionId": "q1", "selectedOptionIds": ["o1"]}]}}),
                )
            })
        });
        let permission: PermissionFn = Arc::new(|_| Box::pin(async { PermissionDecision::Reject }));
        let client = AcpClient::spawn_with_handlers(cmd, permission, Some(ext_fn))
            .await
            .unwrap();
        let r = tokio::time::timeout(Duration::from_secs(3), client.request("ping", json!({})))
            .await
            .expect("request timed out")
            .expect("request failed");
        assert_eq!(r["ok"], json!(true));
        let (method, params) = seen
            .lock()
            .unwrap()
            .clone()
            .expect("ext-method handler must have been invoked");
        assert_eq!(method, "cursor/ask_question");
        assert_eq!(params["questions"][0]["id"], "q1");
    }

    #[tokio::test]
    async fn ext_method_handler_declining_falls_back_to_method_not_found() {
        // Handler returns None → aivo must still reply -32601.
        let cmd = fake_acp_script(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":"x1","method":"cursor/generate_image","params":{}}'
read our_reply
read ping_req
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'"#,
        );
        let ext_fn: ExtMethodFn =
            Arc::new(|_method: String, _params: Value| Box::pin(async { None }));
        let permission: PermissionFn = Arc::new(|_| Box::pin(async { PermissionDecision::Reject }));
        let client = AcpClient::spawn_with_handlers(cmd, permission, Some(ext_fn))
            .await
            .unwrap();
        let r = tokio::time::timeout(Duration::from_secs(3), client.request("ping", json!({})))
            .await
            .expect("request timed out")
            .expect("request failed");
        assert_eq!(r["ok"], json!(true));
    }
}
