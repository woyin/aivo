//! Minimal JSON-RPC 2.0 client over stdio for ACP (Agent Client Protocol)
//! agents such as `cursor-agent acp`. Framing is NDJSON: one JSON object per
//! `\n`-terminated line on the child's stdin/stdout.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Caller-supplied policy for server-initiated `session/request_permission`
/// requests. The closure is invoked from the reader task, so it must be
/// `Send + Sync` and cheap (no blocking I/O). Reading an env var is fine.
pub type PermissionFn = Arc<dyn Fn(&Value) -> PermissionDecision + Send + Sync>;

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
    child: Mutex<Option<Child>>,
    _reader: JoinHandle<()>,
}

impl AcpClient {
    /// Spawn with the default permission policy (graceful reject). Most
    /// callers want [`spawn_with_permission_policy`] so they can opt into
    /// allowing tool execution.
    pub async fn spawn(cmd: Command) -> Result<Self> {
        Self::spawn_with_permission_policy(cmd, Arc::new(|_| PermissionDecision::Reject)).await
    }

    pub async fn spawn_with_permission_policy(
        mut cmd: Command,
        permission_fn: PermissionFn,
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

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let sessions: SessionMap = Arc::new(Mutex::new(HashMap::new()));
        let writer: Writer = Arc::new(Mutex::new(stdin));

        let reader = spawn_reader(
            BufReader::new(stdout),
            pending.clone(),
            sessions.clone(),
            writer.clone(),
            permission_fn,
        );

        Ok(Self {
            writer,
            next_id: AtomicU64::new(1),
            pending,
            session_handlers: sessions,
            child: Mutex::new(Some(child)),
            _reader: reader,
        })
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        let frame = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        write_frame(&self.writer, &frame).await.inspect_err(|_| {
            // Reclaim the pending slot so it doesn't leak.
            let pending = self.pending.clone();
            tokio::spawn(async move {
                pending.lock().await.remove(&id);
            });
        })?;

        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(rpc_err)) => {
                Err(anyhow!(rpc_err).context(format!("ACP method `{method}` failed")))
            }
            Err(_) => Err(anyhow!(
                "ACP child closed connection before `{method}` returned"
            )),
        }
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let frame = json!({"jsonrpc": "2.0", "method": method, "params": params});
        write_frame(&self.writer, &frame).await
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
        if let Err(e) = write_frame(&self.writer, &frame).await {
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
            sessions.lock().await.remove(&session_id);
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

fn spawn_reader(
    mut stdout: BufReader<tokio::process::ChildStdout>,
    pending: PendingMap,
    sessions: SessionMap,
    writer: Writer,
    permission_fn: PermissionFn,
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
            log_acp_frame(Direction::Inbound, &value).await;
            dispatch_inbound(value, &pending, &sessions, &writer, &permission_fn).await;
        }

        // EOF: surface a clear error to anyone still waiting.
        let mut pending_guard = pending.lock().await;
        for (_, tx) in pending_guard.drain() {
            let _ = tx.send(Err(JsonRpcError {
                code: -32603,
                message: "ACP child stdout closed".into(),
                data: None,
            }));
        }
    })
}

async fn dispatch_inbound(
    value: Value,
    pending: &PendingMap,
    sessions: &SessionMap,
    writer: &Writer,
    permission_fn: &PermissionFn,
) {
    let id = value.get("id").and_then(value_to_u64);
    let method = value.get("method").and_then(Value::as_str);

    match (id, method) {
        (Some(id), None) => {
            // Response to one of our requests.
            if let Some(tx) = pending.lock().await.remove(&id) {
                let _ = tx.send(parse_result_or_error(&value));
            }
        }
        (None, Some(method)) => {
            // Notification.
            handle_notification(method, value.get("params"), sessions).await;
        }
        (Some(id), Some("session/request_permission")) => {
            let params = value.get("params").unwrap_or(&Value::Null);
            let decision = permission_fn(params);
            let resp = build_permission_response(id, params, decision);
            let _ = write_frame(writer, &resp).await;
        }
        (Some(id), Some(method)) => {
            // Other server-initiated requests (fs/*, terminal/*) aren't
            // implemented yet — reject with the JSON-RPC method-not-found
            // code so the agent has a defined error to handle.
            let resp = json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("aivo ACP client does not implement `{method}`"),
                },
            });
            let _ = write_frame(writer, &resp).await;
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
fn build_permission_response(id: u64, params: &Value, decision: PermissionDecision) -> Value {
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

async fn handle_notification(method: &str, params: Option<&Value>, sessions: &SessionMap) {
    if method != "session/update" {
        return;
    }
    let Some(params) = params else { return };
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

async fn write_frame(writer: &Writer, value: &Value) -> Result<()> {
    let mut buf = serde_json::to_vec(value).context("encode JSON-RPC frame")?;
    buf.push(b'\n');
    {
        let mut w = writer.lock().await;
        w.write_all(&buf).await.context("write ACP frame")?;
        w.flush().await.context("flush ACP frame")?;
    }
    log_acp_frame(Direction::Outbound, value).await;
    Ok(())
}

/// Direction tag for a logged ACP frame.
#[derive(Copy, Clone)]
enum Direction {
    Inbound,
    Outbound,
}

const ACP_LOG_BODY_LIMIT: usize = 64 * 1024;

/// Emit one paired Phase::Request / Phase::Response entry per ACP frame so the
/// `aivo logs` JSONL stays consistent with the HTTP routers. No-op when
/// `--debug` is not in effect.
async fn log_acp_frame(direction: Direction, value: &Value) {
    let Some(logger) = http_debug::global() else {
        return;
    };
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let id = format!("acp-{n:x}");
    logger
        .log(build_acp_debug_entry(id, direction, value))
        .await;
}

/// Pure builder for the JSONL schema; keeps `log_acp_frame` testable without a
/// real logger. `id` is injected by the caller so tests can pin it.
fn build_acp_debug_entry(id: String, direction: Direction, value: &Value) -> DebugEntry {
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_else(|| match value.get("error") {
            Some(_) => "error",
            None if value.get("result").is_some() => "result",
            _ => "frame",
        })
        .to_string();
    let body_str = value.to_string();
    let body = if body_str.len() > ACP_LOG_BODY_LIMIT {
        let mut truncated = body_str[..ACP_LOG_BODY_LIMIT].to_string();
        truncated.push_str("…[truncated]");
        truncated
    } else {
        body_str
    };
    let (phase, url) = match direction {
        Direction::Outbound => (Phase::Request, "acp://outbound"),
        Direction::Inbound => (Phase::Response, "acp://inbound"),
    };
    DebugEntry {
        ts: chrono::Utc::now().to_rfc3339(),
        id,
        phase,
        method,
        url: url.to_string(),
        status: None,
        duration_ms: None,
        request_headers: BTreeMap::new(),
        request_body: match direction {
            Direction::Outbound => Some(body.clone()),
            Direction::Inbound => None,
        },
        response_headers: BTreeMap::new(),
        response_body: match direction {
            Direction::Inbound => Some(body),
            Direction::Outbound => None,
        },
        error: None,
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
        // The notification for "ghost" should be silently dropped because no
        // handler is registered. Following request still succeeds.
        let r = client.request("ping", json!({})).await.unwrap();
        assert_eq!(r["ok"], json!(true));
    }

    #[test]
    fn outbound_debug_entry_carries_method_and_request_body() {
        let frame = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/prompt",
            "params": {"sessionId": "s", "prompt": [{"type":"text","text":"hi"}]},
        });
        let entry = build_acp_debug_entry("acp-7".into(), Direction::Outbound, &frame);
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
        let entry = build_acp_debug_entry("acp-1".into(), Direction::Inbound, &frame);
        assert!(matches!(entry.phase, Phase::Response));
        assert_eq!(entry.url, "acp://inbound");
        assert_eq!(entry.method, "result");
        assert!(
            entry
                .response_body
                .as_deref()
                .unwrap()
                .contains("\"ok\":true")
        );
        assert!(entry.request_body.is_none());
    }

    #[test]
    fn inbound_debug_entry_marks_error_frames() {
        let frame = json!({"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"nope"}});
        let entry = build_acp_debug_entry("acp-1".into(), Direction::Inbound, &frame);
        assert_eq!(entry.method, "error");
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
        let resp = build_permission_response(42, &params, PermissionDecision::Reject);
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
        let resp = build_permission_response(1, &params, PermissionDecision::Allow);
        assert_eq!(resp["result"]["outcome"]["optionId"], "allow-once");
    }

    #[test]
    fn permission_response_falls_back_to_cancelled_when_no_matching_option() {
        // Some agents may omit allow/reject altogether — `cancelled` is the
        // spec-defined fallback the protocol guarantees.
        let params = json!({"options": [
            {"kind": "weird_custom", "optionId": "x"},
        ]});
        let resp = build_permission_response(1, &params, PermissionDecision::Reject);
        assert_eq!(resp["result"]["outcome"]["outcome"], "cancelled");
        assert!(resp["result"]["outcome"].get("optionId").is_none());
    }

    #[test]
    fn huge_acp_frame_bodies_are_truncated_with_marker() {
        let huge: String = "x".repeat(ACP_LOG_BODY_LIMIT + 5_000);
        let frame = json!({"method": "noisy", "data": huge});
        let entry = build_acp_debug_entry("acp-x".into(), Direction::Outbound, &frame);
        let body = entry.request_body.unwrap();
        assert!(body.ends_with("…[truncated]"));
        assert!(body.len() < frame.to_string().len());
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
}
