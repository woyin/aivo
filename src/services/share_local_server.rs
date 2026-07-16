//! Tiny HTTP/1.1 server that serves a `SharePayload` over the v2 contract:
//! `GET /state?wait=<secs>&since=<cursor>` with long-poll and cursor deltas.
//! See `s.getaivo.dev/protocol.md`.
//!
//! Routes:
//! - `GET /state` — long-poll endpoint. Returns either `200` with
//!   `{meta, cursor, payload: {replace_from, messages, meta, schema_version}}`
//!   when the transcript advanced past `since`, or `304` when `wait` seconds
//!   elapse with no change.
//! - `HEAD /state` — head-only response (same status logic, empty body).
//!
//! ## Long-poll mechanics
//!
//! A request whose `since` matches the current cursor parks on the shared
//! `wake` notify with `tokio::select!`-d wait deadline. The notify fires
//! from two sources:
//!
//! - A background refresher re-resolves the transcript every
//!   `LIVE_REFRESH_INTERVAL` and notifies on any cursor advance.
//! - On shutdown, every parked handler wakes immediately and returns 304
//!   so the proxy can close out the request promptly. This is what makes
//!   `aivo logs share` Ctrl+C close in milliseconds instead of waiting on
//!   in-flight long-polls.
//!
//! ## Delta shape
//!
//! Cursor format: `"<message_count>:<last_message_json_byte_len>"`. From
//! the cursor + the current snapshot we derive `replace_from`:
//!
//! - `since.count == current.count` and `since.last_len != current.last_len`
//!   → last message is being streamed; `replace_from = count - 1`, the
//!   delta carries just the in-flight last message.
//! - `since.count <  current.count` → new turns appended;
//!   `replace_from = since.count`, delta carries the tail.
//! - `since.count >  current.count`, or `since` missing/invalid → reset;
//!   `replace_from = 0`, delta carries the full message list.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::{Notify, RwLock};

use crate::services::http_utils::{
    self, format_http_chunk, http_chunked_response_head_with_extra, http_error_response,
    http_response_head_with_extra, read_full_request,
};
use crate::services::share_payload::SharePayload;
use crate::services::share_redact::{RedactCtx, redact};
use crate::services::share_resolver::{ResolverContext, resolve_session};
use crate::services::shutdown_signal::ShutdownSignal;

const VIEWER_ORIGIN: &str = "https://s.getaivo.dev";
const LIVE_REFRESH_INTERVAL: Duration = Duration::from_millis(500);
const MAX_WAIT_SECS: u64 = 30;

/// What the local server has loaded. `messages_json` caches per-message
/// serialized bytes so a delta response is just a slice + concatenation,
/// not a re-serialize. `cursor` derives from `messages_json` exactly so
/// the streaming-last-message case (`last_len` advance) is detected.
pub struct LiveState {
    session_id: String,
    snapshot: SharePayload,
    messages_json: Vec<Vec<u8>>,
    cursor: String,
    redact_ctx: RedactCtx,
    resolver_ctx: Option<Arc<ResolverContext>>,
    /// Notified by the live refresher every time the cursor advances. Parked
    /// long-poll handlers select on this + shutdown + their wait timer.
    wake: Arc<Notify>,
}

impl LiveState {
    fn from_snapshot(
        session_id: String,
        mut snapshot: SharePayload,
        redact_ctx: RedactCtx,
        resolver_ctx: Option<Arc<ResolverContext>>,
    ) -> Self {
        // `live` in the wire meta means "the server follows the session" —
        // exactly when a resolver context is present.
        snapshot.meta.live = resolver_ctx.is_some();
        let (messages_json, cursor) = serialize_messages(&snapshot);
        Self {
            session_id,
            snapshot,
            messages_json,
            cursor,
            redact_ctx,
            resolver_ctx,
            wake: Arc::new(Notify::new()),
        }
    }

    /// Re-resolve + re-redact from disk. Returns `true` if the cursor changed
    /// (the caller fires `wake` on `true`). No-op without a resolver context.
    async fn refresh(&mut self) -> Result<bool> {
        let Some(resolver) = self.resolver_ctx.clone() else {
            return Ok(false);
        };
        let resolved = resolve_session(&self.session_id, &resolver).await?;
        let mut payload = resolved.payload;
        payload.meta.live = true;
        let (mut redacted, _) = redact(payload, &self.redact_ctx);
        redacted.meta.served_at = chrono::Utc::now();
        let (messages_json, cursor) = serialize_messages(&redacted);
        let changed = cursor != self.cursor;
        self.messages_json = messages_json;
        self.cursor = cursor;
        self.snapshot = redacted;
        Ok(changed)
    }
}

fn serialize_messages(snapshot: &SharePayload) -> (Vec<Vec<u8>>, String) {
    let messages_json: Vec<Vec<u8>> = snapshot
        .messages
        .iter()
        .map(|m| serde_json::to_vec(m).expect("serialize message"))
        .collect();
    let n = messages_json.len();
    let last_len = messages_json.last().map(|v| v.len()).unwrap_or(0);
    (messages_json, format!("{n}:{last_len}"))
}

fn parse_cursor(s: &str) -> Option<(usize, usize)> {
    let (a, b) = s.split_once(':')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

/// Bind a local listener and run the share server until `shutdown` fires.
/// In live mode also spawns a background refresher that notifies `wake` on
/// every cursor advance. Returns the bound port (when host is
/// `127.0.0.1:0`) alongside the join handle so the caller can print the
/// URL and await shutdown via Ctrl+C.
pub async fn start_local_server(
    bind_addr: &str,
    state: LiveState,
    shutdown: ShutdownSignal,
) -> Result<(u16, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(bind_addr).await?;
    let port = listener.local_addr()?.port();
    // Follow the session only when there's a resolver to re-read it from.
    let follow = state.resolver_ctx.is_some();
    let state = Arc::new(RwLock::new(state));

    if follow {
        let refresher_state = state.clone();
        let refresher_shutdown = shutdown.clone();
        tokio::spawn(live_refresher(refresher_state, refresher_shutdown));
    }

    let handle = tokio::spawn(run_loop(listener, state, shutdown));
    Ok((port, handle))
}

/// Public constructor. With a resolver context the server follows the session
/// live; `None` serves a static snapshot (used in tests).
pub fn build_state(
    session_id: String,
    snapshot: SharePayload,
    redact_ctx: RedactCtx,
    resolver_ctx: Option<Arc<ResolverContext>>,
) -> LiveState {
    LiveState::from_snapshot(session_id, snapshot, redact_ctx, resolver_ctx)
}

async fn live_refresher(state: Arc<RwLock<LiveState>>, shutdown: ShutdownSignal) {
    let mut ticker = tokio::time::interval(LIVE_REFRESH_INTERVAL);
    ticker.tick().await; // skip immediate first tick — initial state is already fresh
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            _ = shutdown.wait() => return,
        }
        let mut guard = state.write().await;
        match guard.refresh().await {
            Ok(true) => guard.wake.notify_waiters(),
            Ok(false) => {}
            // Not printed: stderr inside the TUI corrupts the alt screen; the
            // next tick retries anyway.
            Err(_) => {}
        }
    }
}

async fn run_loop(listener: TcpListener, state: Arc<RwLock<LiveState>>, shutdown: ShutdownSignal) {
    loop {
        let accept = tokio::select! {
            result = listener.accept() => result,
            _ = shutdown.wait() => return,
        };
        let Ok((mut socket, _peer)) = accept else {
            continue;
        };
        let state = state.clone();
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            let request_bytes =
                match tokio::time::timeout(Duration::from_secs(60), read_full_request(&mut socket))
                    .await
                {
                    Ok(Ok(b)) => b,
                    _ => {
                        let _ = socket
                            .write_all(http_error_response(400, "bad request").as_bytes())
                            .await;
                        return;
                    }
                };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let _ = handle_request(&request, &state, &shutdown, &mut socket).await;
        });
    }
}

async fn handle_request(
    request: &str,
    state: &Arc<RwLock<LiveState>>,
    shutdown: &ShutdownSignal,
    socket: &mut tokio::net::TcpStream,
) -> std::io::Result<()> {
    let path = http_utils::extract_request_path(request);
    let path_no_query = path.split('?').next().unwrap_or(&path);
    let query = path.split_once('?').map(|(_, q)| q).unwrap_or("");
    let method_is_head = request.starts_with("HEAD ");
    let method_is_get = request.starts_with("GET ") || method_is_head;
    let method_is_options = request.starts_with("OPTIONS ");

    if method_is_options {
        return socket.write_all(&cors_preflight()).await;
    }
    if !method_is_get {
        return socket
            .write_all(http_error_response(405, "method not allowed").as_bytes())
            .await;
    }

    match path_no_query {
        "/state" => serve_state(query, state, shutdown, method_is_head, socket).await,
        _ => {
            socket
                .write_all(http_error_response(404, "not found").as_bytes())
                .await
        }
    }
}

/// Serve `/state`. `wait == 0` is the immediate path — return one full
/// 200 (`Content-Length`) or 304, matching v2 legacy semantics. `wait > 0`
/// holds the connection open and streams every cursor advance during the
/// wait window as an ND-JSON line over `Transfer-Encoding: chunked`.
/// The public proxy splits on `\n` and emits one SSE event per line.
async fn serve_state(
    query: &str,
    state: &Arc<RwLock<LiveState>>,
    shutdown: &ShutdownSignal,
    head_only: bool,
    socket: &mut tokio::net::TcpStream,
) -> std::io::Result<()> {
    let wait_secs = query_u64(query, "wait").unwrap_or(0).min(MAX_WAIT_SECS);
    let mut since = query_string(query, "since").map(percent_decode);

    // Immediate-response path: caller didn't ask to long-poll. Preserves
    // the legacy Content-Length-bounded shape for snapshot fetches and
    // tests that read until EOF without decoding chunked framing.
    if wait_secs == 0 {
        let guard = state.read().await;
        let resp = if since.as_deref() == Some(guard.cursor.as_str()) {
            build_state_304(&guard.cursor, head_only)
        } else {
            build_state_200(&guard, since.as_deref(), head_only)
        };
        drop(guard);
        return socket.write_all(&resp).await;
    }

    // Long-poll path: stream cursor advances. Hold the response open until
    // the deadline so a burst of advances (token-by-token LLM stream)
    // collapses to one HTTP round-trip with many ND-JSON chunks.
    let deadline = Instant::now() + Duration::from_secs(wait_secs);
    let mut head_written = false;

    loop {
        // Arm wake-notify BEFORE reading the cursor so we don't miss a
        // refresh that lands between read and await.
        let wake = state.read().await.wake.clone();
        let notified = wake.notified();
        tokio::pin!(notified);

        let current_cursor = state.read().await.cursor.clone();
        if since.as_deref() != Some(current_cursor.as_str()) {
            let line = {
                let guard = state.read().await;
                build_state_advance_json(&guard, since.as_deref())
            };

            if !head_written {
                let head =
                    http_chunked_response_head_with_extra(200, "application/json", &cors_extra());
                socket.write_all(head.as_bytes()).await?;
                head_written = true;
                if head_only {
                    // HEAD: head already sent; close with empty body.
                    return socket.write_all(b"0\r\n\r\n").await;
                }
            }

            // ND-JSON record: one JSON object + trailing `\n`. Combine into
            // a single chunked write so the line is delivered as one HTTP
            // chunk; the public proxy splits its receive buffer on `\n`.
            let mut record = Vec::with_capacity(line.len() + 1);
            record.extend_from_slice(&line);
            record.push(b'\n');
            let chunk = format_http_chunk(&record);
            socket.write_all(&chunk).await?;
            socket.flush().await?;

            since = Some(current_cursor.clone());
            // Keep looping — more advances may arrive within this window.
        }

        if Instant::now() >= deadline {
            if head_written {
                return socket.write_all(b"0\r\n\r\n").await;
            }
            return socket
                .write_all(&build_state_304(&current_cursor, head_only))
                .await;
        }
        let remaining = deadline - Instant::now();

        tokio::select! {
            _ = notified.as_mut() => continue,
            _ = shutdown.wait() => {
                if head_written {
                    return socket.write_all(b"0\r\n\r\n").await;
                }
                return socket
                    .write_all(&build_state_304(&current_cursor, head_only))
                    .await;
            }
            _ = tokio::time::sleep(remaining) => continue,
        }
    }
}

/// Build one `{meta, cursor, payload}` state-advance JSON object. Shared
/// by the legacy single-response path and the chunked-streaming path —
/// the latter wraps each line in HTTP chunked framing plus a trailing
/// `\n` to form an ND-JSON record.
fn build_state_advance_json(state: &LiveState, since: Option<&str>) -> Vec<u8> {
    let count = state.messages_json.len();
    let (replace_from, slice_from) = match since.and_then(parse_cursor) {
        Some((s_count, s_last_len)) if s_count == count => {
            // Same count: either streaming last message (last_len differs)
            // or a no-op we would've caught above. If equal, we shouldn't
            // be here — but if we are (race), send back the last message.
            let last_len = state.messages_json.last().map(|v| v.len()).unwrap_or(0);
            if count == 0 || last_len == s_last_len {
                (0usize, 0usize)
            } else {
                (count - 1, count - 1)
            }
        }
        Some((s_count, _)) if s_count < count => (s_count, s_count),
        // Reset cases: count went backward (rare), or no/invalid since.
        _ => (0, 0),
    };

    let messages_array = concat_json_array(&state.messages_json[slice_from..]);
    let payload_meta = serde_json::to_vec(&state.snapshot.meta).expect("serialize ShareMeta");
    let envelope_meta = serde_json::to_vec(&meta_value(state)).expect("serialize meta");

    // Hand-assemble the JSON envelope to splice in the pre-serialized
    // message array without an intermediate Value tree. Shape:
    //   {"meta": <meta>, "cursor": "<c>", "payload": {
    //     "replace_from": N, "messages": <array>,
    //     "meta": <payload_meta>, "schema_version": "<v>"}}
    let mut body = Vec::with_capacity(envelope_meta.len() + messages_array.len() + 256);
    body.extend_from_slice(b"{\"meta\":");
    body.extend_from_slice(&envelope_meta);
    body.extend_from_slice(b",\"cursor\":");
    body.extend_from_slice(json_string(&state.cursor).as_bytes());
    body.extend_from_slice(b",\"payload\":{\"replace_from\":");
    body.extend_from_slice(replace_from.to_string().as_bytes());
    body.extend_from_slice(b",\"messages\":");
    body.extend_from_slice(&messages_array);
    body.extend_from_slice(b",\"meta\":");
    body.extend_from_slice(&payload_meta);
    body.extend_from_slice(b",\"schema_version\":");
    body.extend_from_slice(json_string(&state.snapshot.schema_version).as_bytes());
    body.extend_from_slice(b"}}");
    body
}

fn build_state_200(state: &LiveState, since: Option<&str>, head_only: bool) -> Vec<u8> {
    let body = build_state_advance_json(state, since);
    let head = http_response_head_with_extra(200, "application/json", body.len(), &cors_extra());
    let mut out = head.into_bytes();
    if !head_only {
        out.extend_from_slice(&body);
    }
    out
}

fn build_state_304(cursor: &str, _head_only: bool) -> Vec<u8> {
    // Include the cursor as an ETag-style hint so log lines / debugging can
    // see what cursor the parked poll was watching. The public proxy
    // doesn't use it — cursor-on-304 is implicitly "same as the request's
    // ?since=".
    let extra = format!("X-Aivo-Cursor: {cursor}\r\n{}", cors_extra());
    let head =
        http_response_head_with_extra(304, "application/json", 0, extra.trim_end_matches("\r\n"));
    head.into_bytes()
}

fn concat_json_array(items: &[Vec<u8>]) -> Vec<u8> {
    let total: usize = items.iter().map(|v| v.len()).sum();
    let mut out = Vec::with_capacity(total + items.len() + 2);
    out.push(b'[');
    for (i, bytes) in items.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(bytes);
    }
    out.push(b']');
    out
}

fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| String::from("\"\""))
}

fn meta_value(state: &LiveState) -> serde_json::Value {
    json!({
        "source_cli": state.snapshot.source_cli,
        "session_id": state.snapshot.session_id,
        "model": state.snapshot.model,
        "project": state.snapshot.project,
        "created_at": state.snapshot.created_at,
        "updated_at": state.snapshot.updated_at,
        "live": state.snapshot.meta.live,
        "message_count": state.snapshot.messages.len(),
        "schema_version": state.snapshot.schema_version,
    })
}

fn cors_extra() -> String {
    format!(
        "Access-Control-Allow-Origin: {}\r\nAccess-Control-Allow-Methods: GET, HEAD, OPTIONS\r\nAccess-Control-Allow-Headers: Accept",
        VIEWER_ORIGIN
    )
}

fn cors_preflight() -> Vec<u8> {
    let extra = format!("{}\r\nAccess-Control-Max-Age: 86400", cors_extra());
    let head = http_response_head_with_extra(204, "text/plain", 0, &extra);
    head.into_bytes()
}

fn query_string<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=')
            && k == key
        {
            return Some(v);
        }
    }
    None
}

fn query_u64(query: &str, key: &str) -> Option<u64> {
    query_string(query, key).and_then(|v| v.parse().ok())
}

/// Minimal percent-decoder for cursor values. Only handles `%XX` escapes;
/// any malformed sequence is passed through (the cursor's worst case is
/// "doesn't match current" → falls through to full snapshot, which is safe).
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex_value(bytes[i + 1]), hex_value(bytes[i + 2]))
        {
            out.push((h << 4) | l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::share_payload::{
        ContentBlock, ProjectInfo, SHARE_SCHEMA_VERSION, ShareMessage, SharePayload,
    };
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpStream;

    fn fake_payload() -> SharePayload {
        SharePayload {
            schema_version: SHARE_SCHEMA_VERSION.into(),
            source_cli: "claude".into(),
            session_id: "T-test".into(),
            project: ProjectInfo {
                root: Some("~/work/aivo".into()),
                name: Some("aivo".into()),
            },
            model: Some("claude-sonnet-4-5".into()),
            created_at: None,
            updated_at: None,
            messages: vec![ShareMessage {
                role: "user".into(),
                timestamp: None,
                model: None,
                reasoning: None,
                content: vec![ContentBlock::Text {
                    text: "hello".into(),
                }],
            }],
            meta: SharePayload::new_meta(false),
        }
    }

    async fn http_get(port: u16, path: &str) -> (u16, Vec<u8>) {
        http_request(port, "GET", path).await
    }

    async fn http_request(port: u16, method: &str, path: &str) -> (u16, Vec<u8>) {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        );
        tokio::io::AsyncWriteExt::write_all(&mut sock, req.as_bytes())
            .await
            .unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = std::str::from_utf8(&buf[..head_end]).unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = buf[head_end + 4..].to_vec();
        (status, body)
    }

    #[tokio::test]
    async fn first_request_returns_full_snapshot() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (status, body) = http_get(port, "/state").await;
        assert_eq!(status, 200);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["meta"]["source_cli"], "claude");
        assert_eq!(parsed["meta"]["message_count"], 1);
        assert_eq!(parsed["meta"]["live"], false);
        assert!(parsed["cursor"].as_str().unwrap().contains(':'));
        assert_eq!(parsed["payload"]["replace_from"], 0);
        assert_eq!(parsed["payload"]["messages"][0]["role"], "user");
        assert_eq!(parsed["payload"]["schema_version"], "1");

        shutdown.fire();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn cursor_matches_returns_304_after_wait() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        // First fetch to learn the cursor.
        let (_, body) = http_get(port, "/state").await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let cursor = v["cursor"].as_str().unwrap().to_string();

        // Re-poll with that cursor + wait=1; snapshot mode never changes,
        // so we should get 304 after ~1s.
        let started = Instant::now();
        let path = format!("/state?wait=1&since={cursor}");
        let (status, body) = http_get(port, &path).await;
        let elapsed = started.elapsed();
        assert_eq!(status, 304);
        assert!(body.is_empty());
        assert!(
            elapsed >= Duration::from_millis(900),
            "expected ~1s long-poll, got {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(2000),
            "long-poll overshot: {elapsed:?}"
        );

        shutdown.fire();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn shutdown_unblocks_parked_long_poll() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (_, body) = http_get(port, "/state").await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let cursor = v["cursor"].as_str().unwrap().to_string();

        // Start a wait=30 long-poll, then fire shutdown after 100ms. The
        // poll must return well under the wait window — this is the cancel
        // speed contract.
        let path = format!("/state?wait=30&since={cursor}");
        let started = Instant::now();
        let poll = tokio::spawn(async move { http_get(port, &path).await });
        tokio::time::sleep(Duration::from_millis(100)).await;
        shutdown.fire();

        let (status, _) = poll.await.unwrap();
        let elapsed = started.elapsed();
        assert_eq!(status, 304);
        assert!(
            elapsed < Duration::from_secs(2),
            "long-poll didn't unblock on shutdown: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn returns_404_for_unknown_routes_and_405_for_post() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (status, _) = http_get(port, "/nope").await;
        assert_eq!(status, 404);

        let (status, _) = http_request(port, "POST", "/state").await;
        assert_eq!(status, 405);

        shutdown.fire();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[tokio::test]
    async fn cors_preflight_returns_204() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();
        let (status, _) = http_request(port, "OPTIONS", "/state").await;
        assert_eq!(status, 204);
        shutdown.fire();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    #[test]
    fn cursor_changes_on_appended_message() {
        let mut payload = fake_payload();
        let (_, c1) = serialize_messages(&payload);
        payload.messages.push(ShareMessage {
            role: "assistant".into(),
            timestamp: None,
            model: None,
            reasoning: None,
            content: vec![ContentBlock::Text {
                text: "hi back".into(),
            }],
        });
        let (_, c2) = serialize_messages(&payload);
        assert_ne!(c1, c2);
        // Cursor format is "<count>:<last_len>".
        assert!(c2.starts_with("2:"));
    }

    #[test]
    fn cursor_changes_on_streaming_last_message() {
        let mut payload = fake_payload();
        let (_, c1) = serialize_messages(&payload);
        // Mutate the last message (simulating mid-stream growth).
        payload.messages.last_mut().unwrap().content = vec![ContentBlock::Text {
            text: "hello world".into(),
        }];
        let (_, c2) = serialize_messages(&payload);
        assert_ne!(c1, c2);
        // Same count, different last_len.
        assert!(c1.starts_with("1:") && c2.starts_with("1:"));
    }

    #[test]
    fn percent_decode_handles_colon() {
        assert_eq!(percent_decode("3%3A7"), "3:7");
        assert_eq!(percent_decode("3:7"), "3:7");
        assert_eq!(percent_decode("plain"), "plain");
    }

    /// Decode an HTTP/1.1 chunked-encoded body. Returns the concatenated
    /// chunk bodies — stops at the zero-length terminator chunk. Test-
    /// only; the public proxy uses reqwest which handles this natively.
    fn decode_chunked(body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < body.len() {
            let Some(crlf) = body[i..].windows(2).position(|w| w == b"\r\n") else {
                break;
            };
            let len_str = std::str::from_utf8(&body[i..i + crlf]).unwrap_or("0");
            // Strip any chunk-extension after `;` (we don't emit any).
            let len_str = len_str.split(';').next().unwrap_or("0").trim();
            let n = usize::from_str_radix(len_str, 16).unwrap_or(0);
            i += crlf + 2;
            if n == 0 {
                break;
            }
            out.extend_from_slice(&body[i..i + n]);
            i += n + 2; // skip trailing \r\n
        }
        out
    }

    /// Long-poll with `since != current` — the very first iteration sees
    /// an advance and writes the chunked HTTP head + one ND-JSON line.
    /// At the deadline the server writes the zero-length terminator and
    /// returns. Verifies wire shape: `Transfer-Encoding: chunked`, one
    /// JSON-line chunk, then terminator.
    #[tokio::test]
    async fn long_poll_with_advance_streams_chunked_ndjson() {
        let state = build_state("T-test".into(), fake_payload(), RedactCtx::default(), None);
        let shutdown = ShutdownSignal::new();
        let (port, _h) = start_local_server("127.0.0.1:0", state, shutdown.clone())
            .await
            .unwrap();

        let (status, head, body) = http_get_with_head(port, "/state?wait=1&since=bogus").await;
        assert_eq!(status, 200);
        assert!(
            head.to_lowercase().contains("transfer-encoding: chunked"),
            "expected chunked transfer encoding, got head:\n{head}"
        );
        let decoded = decode_chunked(&body);
        let text = std::str::from_utf8(&decoded).expect("decoded body is utf-8");
        // Exactly one ND-JSON record with a trailing `\n`.
        assert!(
            text.ends_with('\n'),
            "expected trailing newline on ND-JSON record, got: {text:?}"
        );
        let line = text.trim_end_matches('\n');
        assert!(
            !line.contains('\n'),
            "expected a single line for a single advance, got: {line:?}"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(line).expect("ND-JSON line parses as JSON");
        assert_eq!(parsed["payload"]["replace_from"], 0);
        assert_eq!(parsed["payload"]["messages"][0]["role"], "user");
        assert!(parsed["cursor"].as_str().is_some());

        shutdown.fire();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    /// Like `http_get` but also returns the head text — needed to
    /// inspect `Transfer-Encoding` on the chunked-streaming path.
    async fn http_get_with_head(port: u16, path: &str) -> (u16, String, Vec<u8>) {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let req =
            format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
        tokio::io::AsyncWriteExt::write_all(&mut sock, req.as_bytes())
            .await
            .unwrap();
        let mut buf = Vec::new();
        sock.read_to_end(&mut buf).await.unwrap();
        let head_end = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = std::str::from_utf8(&buf[..head_end]).unwrap().to_string();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        let body = buf[head_end + 4..].to_vec();
        (status, head, body)
    }
}
