//! Global HTTP debug logger for `aivo run` / `aivo code`.
//!
//! When initialized, captures every reqwest request/response pair as a JSONL
//! entry. When uninitialized, every helper is a fast no-op.

use bytes::Bytes;
use futures::stream::Stream;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::task::{Context, Poll};
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;

const REDACTED: &str = "[REDACTED]";

/// Per-stream capture cap. An Ollama model pull (multi-GB) under `--debug`
/// would otherwise balloon RSS until the stream ends. Beyond this we keep
/// passing chunks through to the consumer but stop appending to the buffer
/// and mark the entry as overflowed.
const MAX_BUFFERED_STREAM_BODY: usize = 8 * 1024 * 1024; // 8 MiB

const REDACTED_HEADERS: &[&str] = &[
    "authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "openai-organization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
];

const REDACTED_QUERY_PARAMS: &[&str] = &["key", "api_key", "token"];

/// Minimum token length we'll treat as a real secret. Short `sk-` / `Bearer`
/// fragments in unrelated text (e.g. docs, error messages) stay readable.
const MIN_TOKEN_LEN: usize = 20;

/// Credential parameter names whose values are masked in logged bodies. OAuth
/// token / dynamic-client-registration exchanges flow through `send_logged`, and
/// their raw token values lack the `Bearer `/`sk-` prefixes the byte scan below
/// keys on — so a token endpoint's request form (`code_verifier=…`,
/// `client_secret=…`) and response JSON (`"access_token":"…"`) would otherwise
/// land in the debug log verbatim.
const SECRET_PARAM_KEYS: &[&str] = &[
    "access_token",
    "refresh_token",
    "id_token",
    "code_verifier",
    "client_secret",
];

/// Scrub secrets from a body before logging: first mask the known credential
/// parameters in both JSON (`"key":"value"`) and form-urlencoded (`key=value`)
/// shapes, then the generic `Bearer <token>` / `sk-<token>` patterns.
pub fn redact_body(body: &str) -> String {
    redact_bearer_tokens(&redact_secret_params(body))
}

/// Mask the values of `SECRET_PARAM_KEYS` in JSON and form bodies. Fast-paths to
/// a plain copy when none of the key names appear at all.
fn redact_secret_params(body: &str) -> String {
    if !SECRET_PARAM_KEYS.iter().any(|k| body.contains(k)) {
        return body.to_string();
    }
    let mut out = body.to_string();
    for key in SECRET_PARAM_KEYS {
        out = redact_json_string_value(&out, key);
        out = redact_form_value(&out, key);
    }
    out
}

/// Mask the `"<value>"` after every `"key":` (a JSON string value).
fn redact_json_string_value(body: &str, key: &str) -> String {
    let needle = format!("\"{key}\"");
    let mut out = body.to_string();
    let mut cursor = 0;
    while let Some(rel) = out[cursor..].find(&needle) {
        let after = cursor + rel + needle.len();
        let rest = &out[after..];
        let Some(colon) = rest.find(':') else { break };
        let Some(open) = rest[colon..].find('"') else {
            cursor = after;
            continue;
        };
        let Some(close_rel) = rest[colon + open + 1..].find('"') else {
            cursor = after;
            continue;
        };
        let start = after + colon + open + 1;
        let end = start + close_rel;
        out.replace_range(start..end, REDACTED);
        cursor = start + REDACTED.len();
    }
    out
}

/// Mask the value after every `key=` (a form-urlencoded value, up to `&` or end),
/// but only where `key` starts at a parameter boundary (start, `&`, or `?`).
fn redact_form_value(body: &str, key: &str) -> String {
    let needle = format!("{key}=");
    let mut out = body.to_string();
    let mut cursor = 0;
    while let Some(rel) = out[cursor..].find(&needle) {
        let idx = cursor + rel;
        let at_boundary = idx == 0 || matches!(out.as_bytes()[idx - 1], b'&' | b'?');
        if !at_boundary {
            cursor = idx + needle.len();
            continue;
        }
        let val_start = idx + needle.len();
        let val_end = out[val_start..]
            .find('&')
            .map(|p| val_start + p)
            .unwrap_or(out.len());
        if val_end > val_start {
            out.replace_range(val_start..val_end, REDACTED);
            cursor = val_start + REDACTED.len();
        } else {
            cursor = val_start;
        }
    }
    out
}

/// Scrub `Bearer <token>` (any case) and bare `sk-<token>` from body text.
/// Operates on raw bytes so it catches keys buried in escaped or nested content
/// (codex's approved_command_prefixes etc.).
fn redact_bearer_tokens(body: &str) -> String {
    if !body
        .as_bytes()
        .windows(7)
        .any(|w| w.eq_ignore_ascii_case(b"Bearer "))
        && !body.contains("sk-")
    {
        return body.to_string();
    }
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + 7 <= bytes.len() && bytes[i..i + 7].eq_ignore_ascii_case(b"Bearer ") {
            out.push_str(&body[i..i + 7]);
            i += 7;
            let token_start = i;
            while i < bytes.len() && is_token_char(bytes[i]) {
                i += 1;
            }
            if i - token_start >= MIN_TOKEN_LEN {
                out.push_str(REDACTED);
            } else {
                out.push_str(&body[token_start..i]);
            }
            continue;
        }
        if bytes[i..].starts_with(b"sk-") && (i == 0 || !is_token_char(bytes[i - 1])) {
            let token_start = i;
            i += 3;
            while i < bytes.len() && is_token_char(bytes[i]) {
                i += 1;
            }
            if i - token_start >= MIN_TOKEN_LEN + 3 {
                out.push_str(REDACTED);
            } else {
                out.push_str(&body[token_start..i]);
            }
            continue;
        }
        let char_start = i;
        i += 1;
        while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
            i += 1;
        }
        out.push_str(&body[char_start..i]);
    }
    out
}

fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.'
}

/// Returns a copy of `headers` with sensitive values replaced by `[REDACTED]`.
/// Retained as a pub helper for the in-module unit tests; production code
/// goes through `collect_and_redact_headers` which operates on
/// `reqwest::header::HeaderMap` directly.
#[allow(dead_code)]
pub fn redact_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| {
            if REDACTED_HEADERS.iter().any(|r| r.eq_ignore_ascii_case(k)) {
                (k.clone(), REDACTED.to_string())
            } else {
                (k.clone(), v.clone())
            }
        })
        .collect()
}

/// Returns `url` with sensitive query parameter values replaced by `[REDACTED]`.
/// Returns the input unchanged if it cannot be parsed.
pub fn redact_url(url: &str) -> String {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return url.to_string();
    };
    if parsed.password().is_some() {
        let _ = parsed.set_password(Some(REDACTED));
    }
    if !parsed.username().is_empty() {
        let _ = parsed.set_username(REDACTED);
    }
    let pairs: Vec<(String, String)> = parsed
        .query_pairs()
        .map(|(k, v)| {
            let key = k.into_owned();
            let value = if REDACTED_QUERY_PARAMS
                .iter()
                .any(|r| r.eq_ignore_ascii_case(&key))
            {
                REDACTED.to_string()
            } else {
                v.into_owned()
            };
            (key, value)
        })
        .collect();
    if pairs.is_empty() {
        return parsed.to_string();
    }
    parsed.query_pairs_mut().clear();
    for (k, v) in pairs {
        parsed.query_pairs_mut().append_pair(&k, &v);
    }
    parsed.to_string()
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Request,
    Response,
    /// Final entry for a streaming response; carries the captured body bytes
    /// after the upstream stream completes (cleanly or mid-flight). Emitted
    /// in addition to the per-headers `Response` entry, sharing its `id`.
    ResponseBody,
    Error,
    /// The request future was dropped before `send()` resolved (interrupt /
    /// process exit while an upstream stalls). Closes the pair so a cancelled
    /// request isn't a dangling `phase=request` that looks like a truncated log.
    Cancelled,
    /// JSON-RPC notification: a frame with `method` but no `id`, used by ACP's
    /// `session/update` push stream. Distinct from `Request` / `Response`
    /// because notifications have no reply and shouldn't be paired up by id
    /// when post-processing the log.
    Notification,
}

#[derive(Debug, Serialize)]
pub(crate) struct DebugEntry {
    pub ts: String,
    pub id: String,
    pub phase: Phase,
    pub method: String,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// HTTP request headers, with sensitive values redacted. Header names are
    /// stored in lowercase because reqwest's `HeaderMap::iter()` yields names
    /// as their canonical lowercase form — downstream JSONL consumers should
    /// match keys case-insensitively (or just lowercase their queries).
    pub request_headers: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_body: Option<String>,
    /// HTTP response headers, with sensitive values redacted. Names are
    /// lowercase (see `request_headers` for rationale).
    pub response_headers: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct LogState {
    file: tokio::fs::File,
    warned: bool,
}

pub(crate) struct HttpDebugLogger {
    state: Mutex<LogState>,
    path: PathBuf,
}

impl HttpDebugLogger {
    pub(crate) async fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            state: Mutex::new(LogState {
                file,
                warned: false,
            }),
            path: path.to_path_buf(),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    /// Synchronous append of one entry, for `Drop` on a cancellation path where
    /// the process is often about to exit and a spawned async write would never
    /// be polled. O_APPEND keeps the single-line write atomic against the async
    /// writer. Bypasses `log`'s redaction, so the entry must carry no body.
    pub(crate) fn log_blocking(&self, entry: &DebugEntry) {
        use std::io::Write;
        let Ok(mut line) = serde_json::to_string(entry) else {
            return;
        };
        line.push('\n');
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            let _ = f.write_all(line.as_bytes());
            let _ = f.flush();
        }
    }

    pub(crate) async fn log(&self, mut entry: DebugEntry) {
        if let Some(body) = entry.request_body.take() {
            entry.request_body = Some(redact_body(&body));
        }
        if let Some(body) = entry.response_body.take() {
            entry.response_body = Some(redact_body(&body));
        }
        let mut line = match serde_json::to_string(&entry) {
            Ok(s) => s,
            Err(_) => return,
        };
        line.push('\n');

        let mut state = self.state.lock().await;
        if state.warned {
            return;
        }
        if let Err(e) = state.file.write_all(line.as_bytes()).await {
            eprintln!("[aivo] debug log write failed: {e}");
            state.warned = true;
            return;
        }
        if let Err(e) = state.file.flush().await {
            eprintln!("[aivo] debug log flush failed: {e}");
            state.warned = true;
        }
    }
}

static GLOBAL: OnceLock<HttpDebugLogger> = OnceLock::new();

/// Initialize the global logger. Subsequent calls are no-ops (first-init wins).
/// Returns the resolved log path on success, io error if the file cannot open.
pub async fn init(path: PathBuf) -> std::io::Result<PathBuf> {
    if let Some(existing) = GLOBAL.get() {
        return Ok(existing.path().to_path_buf());
    }
    let logger = HttpDebugLogger::open(&path).await?;
    let resolved = logger.path().to_path_buf();
    let _ = GLOBAL.set(logger);
    Ok(resolved)
}

pub(crate) fn global() -> Option<&'static HttpDebugLogger> {
    GLOBAL.get()
}

#[cfg(test)]
static FORCE_DEBUG_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test-only override for the debug-active predicate. Tests cannot reliably
/// initialize the file-backed `HttpDebugLogger` (it's a `OnceLock`, so only
/// the first init in a test binary wins), but they need a way to flip the
/// "is --debug on?" decision the `environment_injector` reads. Toggle this
/// flag instead, then reset it to `false` at the end of the test. Pair with
/// `DEBUG_TEST_MUTEX` to serialize the toggle across parallel tests.
#[cfg(test)]
pub(crate) fn set_test_debug_active(active: bool) {
    FORCE_DEBUG_ACTIVE.store(active, std::sync::atomic::Ordering::SeqCst);
}

/// Serialization mutex for tests that flip `FORCE_DEBUG_ACTIVE`. Without this,
/// parallel tests racing on the toggle would see each other's transient
/// `true` and assert against the wrong branch. Each test takes the lock for
/// its full body; `set_test_debug_active(false)` cleanup runs while holding
/// it.
#[cfg(test)]
pub(crate) static DEBUG_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Returns true when `--debug` is in effect — either because `init()` succeeded
/// (production path) or because a test flipped `FORCE_DEBUG_ACTIVE` on.
/// `environment_injector` consults this to force routing through the local
/// bridge for native-protocol upstreams (otherwise the child tool talks
/// straight to upstream and `--debug` captures nothing).
pub fn is_debug_active() -> bool {
    #[cfg(test)]
    if FORCE_DEBUG_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) {
        return true;
    }
    global().is_some()
}

/// Build the default per-invocation log path:
/// `~/.config/aivo/logs/debug-YYYYMMDD-HHMMSS-<pid>.jsonl`.
pub fn default_log_path() -> PathBuf {
    let now = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let pid = std::process::id();
    crate::services::paths::logs_dir(&crate::services::paths::config_dir())
        .join(format!("debug-{now}-{pid}.jsonl"))
}

// ---- LoggedSend extension trait ------------------------------------

/// Collects a reqwest `HeaderMap` into a `BTreeMap<String, String>` and applies
/// the standard sensitive-header redaction. Header names come out lowercase
/// because reqwest stores them in canonical form. Single-pass: builds the
/// redacted map directly without allocating an intermediate raw copy.
fn collect_and_redact_headers(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| {
            let key = k.as_str();
            let value = if REDACTED_HEADERS.iter().any(|r| r.eq_ignore_ascii_case(key)) {
                REDACTED.to_string()
            } else {
                v.to_str().unwrap_or("[binary]").to_string()
            };
            (key.to_string(), value)
        })
        .collect()
}

/// Returns true if `content_type` indicates a streaming body that should not be
/// buffered into memory by the logger. Currently recognizes server-sent events
/// (`text/event-stream`) and newline-delimited JSON (`application/x-ndjson`),
/// which cover every streaming code path aivo currently bridges.
fn is_streaming_content_type(content_type: &str) -> bool {
    content_type.contains("text/event-stream") || content_type.contains("application/x-ndjson")
}

/// Per-stream metadata captured at construction and consumed at drop time to
/// build the trailing `phase=response_body` log entry.
struct FinalizeData {
    id: String,
    method: String,
    url: String,
    /// Logger to write the entry to. We hold an explicit reference (rather
    /// than re-resolving `global()` at drop) so tests that use an injected
    /// per-test logger via `send_logged_with` get their captured-body entries
    /// in the right file.
    logger: &'static HttpDebugLogger,
}

/// Captured stream state. `bytes` holds the (possibly truncated) prefix of the
/// response body; `overflowed` flips to `true` once we hit
/// `MAX_BUFFERED_STREAM_BODY` and stop appending.
#[derive(Default)]
struct StreamBuffer {
    bytes: Vec<u8>,
    overflowed: bool,
}

/// Stream wrapper that tees every chunk into an internal buffer while passing
/// it through to the consumer unchanged. On drop — fired by both clean
/// completion and mid-stream errors — a tokio task writes a
/// `phase=response_body` entry containing the captured bytes.
///
/// Implementation notes:
/// - Uses `StdMutex<StreamBuffer>` (not `tokio::sync::Mutex`) because
///   `poll_next` is sync and cannot `.await`. The lock is held only briefly
///   while appending the chunk; never across an `.await`.
/// - The buffer is capped at `MAX_BUFFERED_STREAM_BODY`. Once exceeded,
///   subsequent chunks pass through to the consumer unmodified but are no
///   longer copied into the buffer; the captured-body log entry includes a
///   truncation marker so the user knows they didn't see everything.
/// - `Drop` cannot `.await`, so it spawns the file write via
///   `tokio::runtime::Handle::try_current()`. If we are not inside a runtime
///   (e.g., during program teardown), the entry is dropped on the floor.
///   Acceptable for a best-effort logger.
/// - When `poll_next` of the inner stream returns `Ready(None)` or
///   `Ready(Some(Err(_)))` the stream is over; subsequent polls or drop
///   safely see no concurrent mutation of the buffer.
struct StreamFinalizer<S> {
    inner: Pin<Box<S>>,
    buffer: Arc<StdMutex<StreamBuffer>>,
    finalize: Option<FinalizeData>,
}

impl<S> StreamFinalizer<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
{
    fn new(
        stream: S,
        id: String,
        method: String,
        url: String,
        logger: &'static HttpDebugLogger,
    ) -> Self {
        Self {
            inner: Box::pin(stream),
            buffer: Arc::new(StdMutex::new(StreamBuffer::default())),
            finalize: Some(FinalizeData {
                id,
                method,
                url,
                logger,
            }),
        }
    }
}

impl<S> Stream for StreamFinalizer<S>
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
{
    type Item = reqwest::Result<Bytes>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                if let Ok(mut buf) = self.buffer.lock() {
                    if buf.bytes.len() >= MAX_BUFFERED_STREAM_BODY {
                        // Already over the cap: pass through, mark overflow,
                        // do not grow the buffer further.
                        buf.overflowed = true;
                    } else {
                        buf.bytes.extend_from_slice(&chunk);
                        if buf.bytes.len() > MAX_BUFFERED_STREAM_BODY {
                            buf.overflowed = true;
                        }
                    }
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            other => other,
        }
    }
}

impl<S> Drop for StreamFinalizer<S> {
    fn drop(&mut self) {
        let Some(finalize) = self.finalize.take() else {
            return;
        };
        // Drain rather than clone — the buffer is never read again after Drop
        // fires, and a multi-MB clone would transiently double peak memory.
        let captured = match self.buffer.lock() {
            Ok(mut b) => std::mem::take(&mut *b),
            Err(_) => return,
        };
        let StreamBuffer { bytes, overflowed } = captured;
        let body_string = match std::str::from_utf8(&bytes) {
            Ok(s) if overflowed => format!(
                "{}\n\n[truncated; captured {} of >{} bytes]",
                s,
                bytes.len(),
                bytes.len()
            ),
            Ok(s) => s.to_string(),
            Err(_) if overflowed => format!(
                "[{} bytes binary; truncated at {} bytes]",
                bytes.len(),
                MAX_BUFFERED_STREAM_BODY
            ),
            Err(_) => format!("[{} bytes binary]", bytes.len()),
        };

        let entry = DebugEntry {
            ts: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            id: finalize.id,
            phase: Phase::ResponseBody,
            method: finalize.method,
            url: finalize.url,
            status: None,
            duration_ms: None,
            request_headers: BTreeMap::new(),
            request_body: None,
            response_headers: BTreeMap::new(),
            response_body: Some(body_string),
            error: None,
        };

        let logger = finalize.logger;
        // Spawn the write only if we have a runtime handle; otherwise drop the
        // entry on the floor. Outside-runtime drop is rare (only during
        // program teardown) and the user has bigger problems by that point.
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                logger.log(entry).await;
            });
        }
    }
}

struct CancelData {
    id: String,
    method: String,
    url: String,
    started: std::time::Instant,
    logger: &'static HttpDebugLogger,
}

/// Guards the `send()` await: if the future is dropped before `send()` resolves
/// (interrupt / process exit while an upstream stalls), `Drop` logs a
/// `phase=cancelled` entry. Disarmed the instant `send()` returns, so the normal
/// path never emits one.
struct CancelGuard {
    data: Option<CancelData>,
}

impl CancelGuard {
    fn disarm(&mut self) {
        self.data.take();
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };
        let duration_ms = data.started.elapsed().as_millis() as u64;
        let entry = DebugEntry {
            ts: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            id: data.id,
            phase: Phase::Cancelled,
            method: data.method,
            url: data.url,
            status: None,
            duration_ms: Some(duration_ms),
            request_headers: BTreeMap::new(),
            request_body: None,
            response_headers: BTreeMap::new(),
            response_body: None,
            error: Some("request cancelled before response (future dropped in-flight)".to_string()),
        };
        // Synchronous, not spawned: cancellation usually precedes process exit,
        // so a spawned write would never be polled in time.
        data.logger.log_blocking(&entry);
    }
}

/// Core logging logic for `send_logged`, parameterized on a logger reference so
/// tests can inject a per-test `HttpDebugLogger` without contending for the
/// process-global `OnceLock`. The trait impl below just resolves `global()` and
/// delegates here.
async fn send_logged_with(
    rb: reqwest::RequestBuilder,
    logger: &'static HttpDebugLogger,
) -> reqwest::Result<reqwest::Response> {
    // Try to clone the builder so we can build() it for inspection
    // without consuming the original send path. RequestBuilder::try_clone
    // returns None when the body is a non-cloneable stream.
    let inspect = rb.try_clone().and_then(|rb| rb.build().ok());

    let id = format!(
        "req_{}",
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
    );
    let started = std::time::Instant::now();
    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

    let (method, url, request_headers, request_body) = if let Some(req) = inspect.as_ref() {
        let body = req.body().and_then(|b| b.as_bytes()).map(|bytes| {
            String::from_utf8(bytes.to_vec())
                .unwrap_or_else(|_| format!("[{} bytes binary]", bytes.len()))
        });
        (
            req.method().as_str().to_string(),
            redact_url(req.url().as_str()),
            collect_and_redact_headers(req.headers()),
            body,
        )
    } else {
        ("?".to_string(), "?".to_string(), BTreeMap::new(), None)
    };

    // Emit a phase=request entry before the send completes so the outbound
    // payload is visible even when the upstream stalls or fails mid-flight.
    // Both entries share the same `id` for correlation. The clones here are
    // intentional — the alternative is contortions to avoid one BTreeMap
    // clone, which is not worth the readability cost.
    logger
        .log(DebugEntry {
            ts: ts.clone(),
            id: id.clone(),
            phase: Phase::Request,
            method: method.clone(),
            url: url.clone(),
            status: None,
            duration_ms: None,
            request_headers: request_headers.clone(),
            request_body: request_body.clone(),
            response_headers: BTreeMap::new(),
            response_body: None,
            error: None,
        })
        .await;

    // Logs `phase=cancelled` if this future is dropped before `send()` resolves;
    // disarmed below so the Ok/Err paths own the terminal entry.
    let mut cancel_guard = CancelGuard {
        data: Some(CancelData {
            id: id.clone(),
            method: method.clone(),
            url: url.clone(),
            started,
            logger,
        }),
    };
    let resp_result = rb.send().await;
    cancel_guard.disarm();
    let duration_ms = started.elapsed().as_millis() as u64;

    match resp_result {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let content_type = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            if is_streaming_content_type(&content_type) {
                // Streaming: log status + headers immediately (with response_body
                // omitted), then return a rebuilt Response whose body tees every
                // chunk into a buffer. When the consumer drops the stream, a
                // `phase=response_body` entry carrying the captured bytes is
                // appended to the log. This means partial bodies from streams
                // that fail mid-flight are preserved — the case the previous
                // "[streamed; ...]" placeholder hid most.
                let original_headers = resp.headers().clone();
                let response_headers = collect_and_redact_headers(&original_headers);
                logger
                    .log(DebugEntry {
                        ts,
                        id: id.clone(),
                        phase: Phase::Response,
                        method: method.clone(),
                        url: url.clone(),
                        status: Some(status),
                        duration_ms: Some(duration_ms),
                        request_headers,
                        request_body,
                        response_headers,
                        response_body: None,
                        error: None,
                    })
                    .await;

                let bytes_stream = resp.bytes_stream();
                let teed = StreamFinalizer::new(bytes_stream, id, method, url, logger);
                let body = reqwest::Body::wrap_stream(teed);
                let mut http_builder = http::Response::builder().status(status);
                // Preserve the original (non-redacted) response headers on the
                // rebuilt Response so downstream code that inspects e.g.
                // content-type keeps working.
                for (k, v) in original_headers.iter() {
                    http_builder = http_builder.header(k, v);
                }
                let http_resp = http_builder
                    .body(body)
                    .expect("response builder cannot fail with valid status+headers");
                return Ok(reqwest::Response::from(http_resp));
            }

            // Non-streaming: buffer body, log it, then reconstruct a Response
            // from the buffered bytes for the caller to consume. Capture the
            // original (un-redacted) header map BEFORE consuming the body so
            // the rebuilt Response carries true Set-Cookie / WWW-Authenticate
            // / etc. — only the LOG entry's response_headers is redacted.
            let original_headers = resp.headers().clone();
            let response_headers = collect_and_redact_headers(&original_headers);
            let bytes = resp.bytes().await?;
            let body_string = match std::str::from_utf8(&bytes) {
                Ok(s) => s.to_string(),
                Err(_) => format!("[{} bytes binary]", bytes.len()),
            };

            logger
                .log(DebugEntry {
                    ts,
                    id,
                    phase: Phase::Response,
                    method,
                    url,
                    status: Some(status),
                    duration_ms: Some(duration_ms),
                    request_headers,
                    request_body,
                    response_headers,
                    response_body: Some(body_string),
                    error: None,
                })
                .await;

            let mut http_builder = http::Response::builder().status(status);
            // Use the un-redacted headers so downstream consumers reading
            // e.g. content-type or set-cookie keep working.
            for (k, v) in original_headers.iter() {
                http_builder = http_builder.header(k, v);
            }
            let http_resp = http_builder
                .body(bytes)
                .expect("response builder cannot fail with valid status+headers");
            Ok(reqwest::Response::from(http_resp))
        }
        Err(e) => {
            logger
                .log(DebugEntry {
                    ts,
                    id,
                    phase: Phase::Error,
                    method,
                    url,
                    status: None,
                    duration_ms: Some(duration_ms),
                    request_headers,
                    request_body,
                    response_headers: BTreeMap::new(),
                    response_body: None,
                    error: Some(e.to_string()),
                })
                .await;
            Err(e)
        }
    }
}

/// Extension on `reqwest::RequestBuilder`. When the global debug logger is
/// initialized, captures request/response details to JSONL. When uninitialized,
/// `.send_logged()` is equivalent to `.send().await`.
///
/// Uses the explicit `impl Future + Send` form rather than `async fn` in trait
/// because `async fn` does not propagate a `Send` bound on the returned future,
/// and aivo passes some HTTP calls through `tokio::spawn` (see
/// `commands/keys.rs` parallel ping fan-out), which requires `Send + 'static`
/// regardless of runtime flavor. The impl-level `manual_async_fn` clippy lint
/// is suppressed for the same reason.
///
/// **Two-entry pattern:** every instrumented call writes a `phase=request`
/// entry before `send()` completes, then a matching `phase=response` (or
/// `phase=error`) entry afterward. Both share the same `id` field for
/// downstream correlation. This means the user sees the outbound payload even
/// when the upstream stream stalls or fails mid-flight. A future dropped before
/// `send()` resolves closes the pair with a `phase=cancelled` entry instead.
///
/// **Streaming-aware:** if the response's `Content-Type` is
/// `text/event-stream` or `application/x-ndjson`, the body is *not* buffered.
/// The log entry's `response_body` becomes a `[streamed; content-type=...]`
/// placeholder and the original `Response` is returned unmodified so callers
/// can consume chunks via `bytes_stream()` / `chunk()` for incremental
/// rendering. Non-streaming responses still buffer-and-reconstruct as before
/// (typical AI-API responses are <1 MB so this is fine).
pub trait LoggedSend {
    fn send_logged(
        self,
    ) -> impl std::future::Future<Output = reqwest::Result<reqwest::Response>> + Send;
}

impl LoggedSend for reqwest::RequestBuilder {
    #[allow(clippy::manual_async_fn)]
    fn send_logged(
        self,
    ) -> impl std::future::Future<Output = reqwest::Result<reqwest::Response>> + Send {
        async move {
            let Some(logger) = global() else {
                return self.send().await;
            };
            send_logged_with(self, logger).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(items: &[(&str, &str)]) -> BTreeMap<String, String> {
        items
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn redact_headers_replaces_authorization_case_insensitive() {
        let h = map(&[
            ("Authorization", "Bearer abc"),
            ("Content-Type", "application/json"),
        ]);
        let out = redact_headers(&h);
        assert_eq!(out["Authorization"], "[REDACTED]");
        assert_eq!(out["Content-Type"], "application/json");

        let h2 = map(&[("authorization", "Bearer xyz")]);
        assert_eq!(redact_headers(&h2)["authorization"], "[REDACTED]");
    }

    #[test]
    fn redact_headers_replaces_all_known_sensitive_headers() {
        let h = map(&[
            ("x-api-key", "k1"),
            ("api-key", "k2"),
            ("x-goog-api-key", "k3"),
            ("openai-organization", "org"),
            ("Cookie", "session=abc"),
            ("Set-Cookie", "session=abc"),
            ("Proxy-Authorization", "Basic xxx"),
        ]);
        let out = redact_headers(&h);
        for k in h.keys() {
            assert_eq!(out[k], "[REDACTED]", "header {k} not redacted");
        }
    }

    #[test]
    fn redact_url_replaces_known_query_params() {
        let out = redact_url("https://api.example.com/v1/m?key=abc&model=gpt-5");
        // url crate percent-encodes `[` and `]`, so accept either form
        assert!(
            out.contains("key=%5BREDACTED%5D") || out.contains("key=[REDACTED]"),
            "expected redacted key in {out}"
        );
        assert!(
            out.contains("model=gpt-5"),
            "expected model preserved in {out}"
        );
    }

    #[test]
    fn redact_url_passes_through_invalid_url() {
        assert_eq!(redact_url("not a url"), "not a url");
    }

    #[test]
    fn redact_url_no_query_unchanged() {
        let url = "https://api.example.com/v1/messages";
        assert_eq!(redact_url(url), url);
    }

    #[test]
    fn redact_url_redacts_userinfo() {
        let out = redact_url("https://user:supersecret@api.example.com/v1?model=x");
        assert!(!out.contains("supersecret"), "password leaked: {out}");
        assert!(
            !out.contains("user@") && !out.contains("user:"),
            "username leaked: {out}"
        );
        assert!(out.contains("model=x"), "non-sensitive query lost: {out}");
    }

    #[test]
    fn redact_body_scrubs_bearer_token_in_codex_approved_prefixes() {
        // Real shape pulled from a codex /responses payload: the bearer
        // sits inside an `approved_command_prefixes` curl literal, so the
        // scrubber has to find it without parsing JSON.
        let body = "{\"prefixes\":[\"curl\",\"-H\",\"Authorization: Bearer sk-8OHXJR2LGKlPkGNWa0dp1QVPdTJZO5IP\"]}";
        let out = redact_body(body);
        assert!(!out.contains("sk-8OHXJR2"), "token leaked: {out}");
        assert!(out.contains("Bearer [REDACTED]"), "shape lost: {out}");
    }

    #[test]
    fn redact_body_scrubs_bare_sk_key() {
        let body = "{\"key\":\"sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA\"}";
        let out = redact_body(body);
        assert!(!out.contains("sk-ant-api03"), "bare key leaked: {out}");
        assert!(out.contains("[REDACTED]"));
    }

    #[test]
    fn redact_body_preserves_short_non_secret_strings() {
        // "Bearer" mentioned in prose without a long token shouldn't be
        // scrubbed. Same for short hyphenated identifiers starting with sk-.
        let body = "see the Bearer auth flow; component id sk-btn";
        assert_eq!(redact_body(body), body);
    }

    #[test]
    fn redact_body_no_match_short_circuits() {
        let body = "{\"hello\":\"world\"}";
        assert_eq!(redact_body(body), body);
    }

    #[test]
    fn redact_body_preserves_utf8() {
        let body = "前缀 Bearer sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXX 后缀";
        let out = redact_body(body);
        assert!(out.starts_with("前缀 Bearer [REDACTED]"));
        assert!(out.ends_with(" 后缀"));
    }

    #[test]
    fn redact_body_scrubs_lowercase_bearer_in_shell_literal() {
        let body = "{\"cmd\":\"curl -H 'authorization: bearer xoxb-AAAAAAAAAAAAAAAAAAAAAAAA'\"}";
        let out = redact_body(body);
        assert!(
            !out.contains("xoxb-AAAAAAAAAAAA"),
            "lowercase bearer leaked: {out}"
        );
        assert!(
            out.contains("bearer [REDACTED]"),
            "casing should be preserved: {out}"
        );
    }

    #[test]
    fn redact_body_masks_oauth_token_response_json() {
        // A token endpoint's JSON response — the tokens lack a `Bearer `/`sk-`
        // prefix, so only the new key-based masking catches them.
        let body = r#"{"access_token":"eyJraWQ.AAAAAAAAAAAAAAAAAAAA","refresh_token":"rt-BBBBBBBBBBBBBBBBBBBB","expires_in":3600,"token_type":"Bearer"}"#;
        let out = redact_body(body);
        assert!(!out.contains("eyJraWQ"), "access_token leaked: {out}");
        assert!(!out.contains("rt-BBBB"), "refresh_token leaked: {out}");
        assert!(out.contains(r#""access_token":"[REDACTED]""#), "{out}");
        assert!(out.contains(r#""refresh_token":"[REDACTED]""#), "{out}");
        // Non-secret fields survive.
        assert!(out.contains("\"expires_in\":3600"), "{out}");
    }

    #[test]
    fn redact_body_masks_oauth_token_request_form() {
        // The token exchange request body is form-urlencoded — PKCE verifier and
        // client_secret must not leak (PkcePair docs it as never-logged).
        let body = "grant_type=authorization_code&code=auth123&code_verifier=VVVVVVVVVVVVVVVVVVVV&client_secret=SSSSSSSSSSSSSSSSSSSS";
        let out = redact_body(body);
        assert!(!out.contains("VVVVVVVV"), "code_verifier leaked: {out}");
        assert!(!out.contains("SSSSSSSS"), "client_secret leaked: {out}");
        assert!(out.contains("code_verifier=[REDACTED]"), "{out}");
        assert!(out.contains("client_secret=[REDACTED]"), "{out}");
        // The grant_type (non-secret) and the bare auth code survive.
        assert!(out.contains("grant_type=authorization_code"), "{out}");
    }

    #[tokio::test]
    async fn logger_writes_jsonl_to_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("debug.jsonl");
        let logger = HttpDebugLogger::open(&path).await.unwrap();

        logger
            .log(DebugEntry {
                ts: "2026-04-26T14:10:33.421Z".into(),
                id: "req_test".into(),
                phase: Phase::Response,
                method: "POST".into(),
                url: "https://api.example.com/v1/m".into(),
                status: Some(200),
                duration_ms: Some(123),
                request_headers: BTreeMap::new(),
                request_body: Some("{}".into()),
                response_headers: BTreeMap::new(),
                response_body: Some("ok".into()),
                error: None,
            })
            .await;

        drop(logger);
        let content = tokio::fs::read_to_string(&path).await.unwrap();
        let line = content.lines().next().unwrap();
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["status"], 200);
        assert_eq!(v["url"], "https://api.example.com/v1/m");
        assert_eq!(v["phase"], "response");
        assert_eq!(v["method"], "POST");
    }

    #[test]
    fn default_log_path_has_expected_shape() {
        let p = default_log_path();
        // `Path::ends_with` compares whole path components, so it tolerates
        // platform-native separators on Windows (`\`) and Unix (`/`).
        let parent = p.parent().expect("log path must have a parent");
        assert!(
            parent.ends_with(std::path::Path::new(".config/aivo/logs")),
            "unexpected parent: {}",
            parent.display()
        );
        let name = p.file_name().expect("log path must have a file name");
        let name = name.to_string_lossy();
        assert!(name.starts_with("debug-"), "missing prefix: {name}");
        assert!(name.ends_with(".jsonl"), "wrong extension: {name}");
    }

    #[test]
    fn send_logged_trait_is_implemented_for_request_builder() {
        // Type-level assertion: the trait method exists with the expected
        // signature on RequestBuilder. Compiles iff the impl is in scope.
        // Behavior is exercised by integration tests under tests/.
        fn _assert<T: LoggedSend>() {}
        _assert::<reqwest::RequestBuilder>();
    }

    #[test]
    fn is_streaming_content_type_recognizes_known_streaming_types() {
        assert!(is_streaming_content_type("text/event-stream"));
        assert!(is_streaming_content_type(
            "text/event-stream; charset=utf-8"
        ));
        assert!(is_streaming_content_type("application/x-ndjson"));
        assert!(!is_streaming_content_type("application/json"));
        assert!(!is_streaming_content_type(""));
        assert!(!is_streaming_content_type("text/plain"));
    }

    /// Spins up a one-shot HTTP listener that returns a fixed response. Returns
    /// the bound address and a join handle. The body of the response is
    /// supplied verbatim including headers/CRLFs.
    async fn one_shot_server(
        raw_response: &'static str,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            socket.write_all(raw_response.as_bytes()).await.unwrap();
            let _ = socket.flush().await;
        });
        (addr, handle)
    }

    #[tokio::test]
    async fn send_logged_emits_request_entry_before_response() {
        // Verify the two-entry pattern: a phase=request entry written before
        // send() completes, followed by a phase=response entry, both sharing
        // the same id field.
        let response = "HTTP/1.1 200 OK\r\n\
                        Content-Type: application/json\r\n\
                        Content-Length: 17\r\n\
                        Connection: close\r\n\
                        \r\n\
                        {\"text\":\"ok!!\"}\r\n";
        let (addr, server) = one_shot_server(response).await;

        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("debug.jsonl");
        // `send_logged_with` requires a `&'static HttpDebugLogger` because the
        // streaming branch hands the reference to a `StreamFinalizer` whose
        // Drop fires at an unknown future time. Leaking is safe in tests —
        // each test process exits shortly after.
        let logger: &'static HttpDebugLogger =
            Box::leak(Box::new(HttpDebugLogger::open(&log_path).await.unwrap()));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let rb = client
            .post(format!("http://{addr}/v1/messages"))
            .header("Content-Type", "application/json")
            .body(r#"{"q":"hi"}"#);
        let resp = send_logged_with(rb, logger)
            .await
            .expect("send_logged_with should succeed");
        assert_eq!(resp.status().as_u16(), 200);
        let _ = resp.text().await.unwrap();

        server.await.unwrap();
        // Give the file a chance to flush; logger is leaked, so we can read
        // through it safely.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "expected request + response entries, got {}: {content}",
            lines.len()
        );
        let req: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let resp_entry: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(req["phase"], "request", "first line: {req}");
        assert_eq!(resp_entry["phase"], "response", "second line: {resp_entry}");
        assert_eq!(
            req["id"], resp_entry["id"],
            "request and response should share id"
        );
        assert_eq!(req["method"], "POST");
        // Pre-send entry has no status/duration; response entry has both.
        assert!(req.get("status").map(|v| v.is_null()).unwrap_or(true));
        assert_eq!(resp_entry["status"], 200);
        assert!(resp_entry["duration_ms"].is_number());
    }

    #[tokio::test]
    async fn send_logged_tees_streaming_response_into_log() {
        // Verify the 3-entry streaming pattern:
        //   1. phase=request    — pre-send
        //   2. phase=response   — headers received, no body
        //   3. phase=response_body — captured bytes after stream drops
        // The caller must still see the SSE chunks at read time, AND the
        // captured body must contain those same chunks.
        let sse_body = "data: {\"hello\":1}\n\ndata: {\"world\":2}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            sse_body.len(),
            sse_body
        );
        let raw: &'static str = Box::leak(response.into_boxed_str());
        let (addr, server) = one_shot_server(raw).await;

        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("debug.jsonl");
        let logger: &'static HttpDebugLogger =
            Box::leak(Box::new(HttpDebugLogger::open(&log_path).await.unwrap()));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let rb = client
            .post(format!("http://{addr}/v1/stream"))
            .header("Content-Type", "application/json")
            .body(r#"{"stream":true}"#);
        let resp = send_logged_with(rb, logger)
            .await
            .expect("send_logged_with should succeed");
        assert_eq!(resp.status().as_u16(), 200);

        // Right after headers are received, the log file should already
        // contain the request + response entries (no body yet).
        // Allow a brief moment for the response-headers entry to flush.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let early = tokio::fs::read_to_string(&log_path).await.unwrap();
        let early_lines: Vec<&str> = early.lines().collect();
        assert_eq!(
            early_lines.len(),
            2,
            "before stream consumed, expected req+resp entries; got: {early}"
        );

        // Caller streams the body successfully.
        let body = resp.text().await.unwrap();
        assert!(
            body.contains("hello") && body.contains("world"),
            "caller-side body should contain SSE chunks; got: {body}"
        );

        server.await.unwrap();
        // After the response is dropped, the StreamFinalizer's Drop spawns the
        // response_body write. Give it a moment to land.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            3,
            "expected req+resp+response_body entries; got: {content}"
        );
        let req: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let resp_entry: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        let body_entry: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(req["phase"], "request");
        assert_eq!(resp_entry["phase"], "response");
        assert_eq!(body_entry["phase"], "response_body");
        // All three entries share the same id.
        assert_eq!(req["id"], resp_entry["id"]);
        assert_eq!(req["id"], body_entry["id"]);
        // The headers-only response entry omits response_body entirely.
        assert!(
            resp_entry.get("response_body").is_none(),
            "response entry should omit response_body; got: {resp_entry}"
        );
        // The body entry has the actual SSE bytes.
        let captured = body_entry["response_body"]
            .as_str()
            .expect("response_body should be a string on the body entry");
        assert!(
            captured.contains("hello") && captured.contains("world"),
            "captured body should contain SSE chunks; got: {captured}"
        );
    }

    #[tokio::test]
    async fn send_logged_streaming_captures_partial_body_on_mid_stream_close() {
        // Verify partial-body capture: the upstream sends a few SSE chunks
        // then closes the connection mid-frame. The log must contain
        // whatever did arrive before the abort.
        use tokio::io::AsyncWriteExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        // The body is *intentionally truncated* — fewer bytes than
        // Content-Length advertises — so the consumer's bytes_stream() will
        // either error on EOF or simply end short. Either way, our capture
        // should hold the bytes that did arrive.
        let chunk = "data: {\"partial\":\"yes\"}\n\ndata: {\"more\":\"al";
        let server = tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            // Promise more bytes than we'll send, then drop the socket.
            let head = "HTTP/1.1 200 OK\r\n\
                        Content-Type: text/event-stream\r\n\
                        Content-Length: 9999\r\n\
                        Connection: close\r\n\
                        \r\n";
            socket.write_all(head.as_bytes()).await.unwrap();
            socket.write_all(chunk.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
            // Drop without sending the promised remainder.
        });

        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("debug.jsonl");
        let logger: &'static HttpDebugLogger =
            Box::leak(Box::new(HttpDebugLogger::open(&log_path).await.unwrap()));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let rb = client
            .post(format!("http://{addr}/v1/stream"))
            .header("Content-Type", "application/json")
            .body(r#"{"stream":true}"#);
        let resp = send_logged_with(rb, logger)
            .await
            .expect("headers should arrive even though body aborts");
        // Drain the stream; depending on tokio/reqwest version the truncated
        // response may yield bytes then error, or yield bytes then EOF. Both
        // are acceptable for the partial-capture test.
        let _ = resp.bytes().await;

        let _ = server.await;
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(
            lines.len() >= 3,
            "expected at least req+resp+response_body entries; got {} lines: {content}",
            lines.len()
        );
        // The response_body entry is whatever has phase response_body; pick
        // it positionally — it's emitted last for any given request.
        let body_entry: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
        assert_eq!(body_entry["phase"], "response_body");
        let captured = body_entry["response_body"]
            .as_str()
            .expect("response_body should be a string");
        assert!(
            captured.contains("partial"),
            "partial bytes should be captured; got: {captured}"
        );
    }

    #[tokio::test]
    async fn send_logged_emits_cancelled_when_future_dropped_before_response() {
        // An upstream that accepts the request but never replies (the publicai
        // 32B hang), with the caller dropping the future before headers arrive:
        // the log must show request + cancelled, not a dangling request.
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(socket);
        });

        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("debug.jsonl");
        let logger: &'static HttpDebugLogger =
            Box::leak(Box::new(HttpDebugLogger::open(&log_path).await.unwrap()));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let rb = client
            .post(format!("http://{addr}/v1/chat/completions"))
            .header("Content-Type", "application/json")
            .body(r#"{"stream":true}"#);

        // Drop the send future before the server ever responds.
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(250),
            send_logged_with(rb, logger),
        )
        .await;
        assert!(res.is_err(), "expected the send to be cancelled by timeout");

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "expected request + cancelled entries; got: {content}"
        );
        let req: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let cancelled: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(req["phase"], "request");
        assert_eq!(cancelled["phase"], "cancelled", "second line: {cancelled}");
        assert_eq!(
            req["id"], cancelled["id"],
            "request and cancelled should share id"
        );
        // The cancelled entry carries a duration and no status/response.
        assert!(cancelled["duration_ms"].is_number());
        assert!(cancelled.get("status").map(|v| v.is_null()).unwrap_or(true));

        server.abort();
    }

    #[tokio::test]
    async fn send_logged_does_not_emit_cancelled_on_normal_response() {
        // A request that completes normally must disarm the guard.
        let response = "HTTP/1.1 200 OK\r\n\
                        Content-Type: application/json\r\n\
                        Content-Length: 17\r\n\
                        Connection: close\r\n\
                        \r\n\
                        {\"text\":\"ok!!\"}\r\n";
        let (addr, server) = one_shot_server(response).await;

        let dir = tempfile::TempDir::new().unwrap();
        let log_path = dir.path().join("debug.jsonl");
        let logger: &'static HttpDebugLogger =
            Box::leak(Box::new(HttpDebugLogger::open(&log_path).await.unwrap()));

        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let rb = client
            .post(format!("http://{addr}/v1/messages"))
            .body(r#"{"q":"hi"}"#);
        let resp = send_logged_with(rb, logger).await.unwrap();
        let _ = resp.text().await.unwrap();

        server.await.unwrap();

        let content = tokio::fs::read_to_string(&log_path).await.unwrap();
        assert!(
            !content.contains("\"phase\":\"cancelled\""),
            "normal response must not log a cancelled entry; got: {content}"
        );
    }
}
