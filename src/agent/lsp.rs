//! LSP diagnostics-after-edit (default on, `AIVO_AGENT_LSP=0` opts out). After the agent edits a
//! file, ask a language server for its *native* diagnostics — the syntax /
//! name-resolution / semantic errors the server computes itself — and fold the errors
//! back into the tool result so the model fixes them in the same turn.
//!
//! Deliberately "fast native only": a short, bounded wait for diagnostics to settle, so
//! it never blocks the loop for long. Slow whole-project type-check errors (e.g.
//! rust-analyzer's `cargo check` flycheck) are out of scope — those are the job of the
//! self-verify validator ([`crate::agent::verify`]). Graceful-degrade everywhere: not
//! enabled, no server binary, or a protocol hiccup → no diagnostics, never an error.

use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// One error-severity diagnostic, 1-based line for display.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub line: u32,
    pub message: String,
}

/// After sending a document update, keep draining diagnostics until this long passes
/// with no new message (the server has "settled")…
const SETTLE_QUIET: Duration = Duration::from_millis(350);
/// …but never wait longer than this overall. A warm server settles in ~SETTLE_QUIET;
/// this only backstops a still-indexing (cold) server, which pre-warming makes rare.
const OVERALL_CAP: Duration = Duration::from_millis(8000);

/// `(program, args, languageId)` for a file extension, or `None` if unsupported. Kept
/// tiny and additive — new languages are one row. Only servers whose binary is on PATH
/// actually start (see [`LspServer::start`]).
fn server_spec(path: &Path) -> Option<(&'static str, &'static [&'static str], &'static str)> {
    match path.extension()?.to_str()? {
        "rs" => Some(("rust-analyzer", &[], "rust")),
        "go" => Some(("gopls", &["serve"], "go")),
        // .ts and .tsx share one typescript-language-server but need distinct
        // languageIds (the server treats `typescriptreact` as JSX-aware).
        "ts" | "mts" | "cts" => Some(("typescript-language-server", &["--stdio"], "typescript")),
        "tsx" => Some((
            "typescript-language-server",
            &["--stdio"],
            "typescriptreact",
        )),
        "py" | "pyi" => Some(("pyright-langserver", &["--stdio"], "python")),
        _ => None,
    }
}

/// A running server shared between the manager and each blocking check.
type SharedServer = Arc<Mutex<LspServer>>;
/// Program name → its running server, shared across edits in a session.
type ServerMap = Arc<Mutex<HashMap<&'static str, SharedServer>>>;

/// Manages one long-lived language server per program, shared across edits in a session.
pub struct LspManager {
    root: PathBuf,
    servers: ServerMap,
}

impl LspManager {
    pub fn new(root: &Path) -> Self {
        Self {
            // Canonicalize once so the per-edit confinement check is a plain prefix compare.
            root: std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf()),
            servers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Pre-start the server for a language the workspace obviously uses (a `Cargo.toml`
    /// / `go.mod` at the root), in the background, so it indexes during the first model
    /// round-trips instead of stalling the first edit. Best-effort; a failed start just
    /// leaves the lazy path to try again.
    pub fn warm(&self) {
        // `server_spec` owns each server's (program, args); warm only maps a root marker
        // to a sample path, so adding a language stays a one-row edit in one place.
        const WARM: &[(&str, &str)] = &[
            ("Cargo.toml", "x.rs"),
            ("go.mod", "x.go"),
            ("tsconfig.json", "x.ts"),
            ("pyproject.toml", "x.py"),
        ];
        for (marker, sample) in WARM {
            if self.root.join(marker).is_file()
                && let Some((prog, args, _)) = server_spec(Path::new(sample))
            {
                self.warm_prog(prog, args);
            }
        }
    }

    /// Start `prog` in a background thread and publish it, unless the lazy path already
    /// did (checked before and after the slow handshake to keep it out of the lock).
    fn warm_prog(&self, prog: &'static str, args: &'static [&'static str]) {
        let servers = self.servers.clone();
        let root = self.root.clone();
        std::thread::spawn(move || {
            if servers.lock().unwrap().contains_key(prog) {
                return;
            }
            if let Some(s) = LspServer::start(prog, args, &root) {
                servers
                    .lock()
                    .unwrap()
                    .entry(prog)
                    .or_insert_with(|| Arc::new(Mutex::new(s)));
            }
        });
    }

    /// Error diagnostics the language server reports for `path` right after we (re)open
    /// it. Empty when unsupported, the server binary is absent, the path escapes the
    /// workspace root, or nothing settled in time. Runs its blocking I/O off-runtime.
    pub async fn diagnostics(&self, path: &Path) -> Vec<Diagnostic> {
        let Some((prog, args, lang)) = server_spec(path) else {
            return Vec::new();
        };
        // Path confinement: never open a file outside the workspace root.
        let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let root = self.root.clone();
        if !abs.starts_with(&root) {
            trace(&format!("path {abs:?} escapes root {root:?} — skip"));
            return Vec::new();
        }
        let servers = self.servers.clone();
        tokio::task::spawn_blocking(move || {
            let server = {
                let mut map = servers.lock().unwrap();
                match map.get(prog) {
                    Some(s) => s.clone(),
                    None => match LspServer::start(prog, args, &root) {
                        Some(s) => {
                            trace(&format!("{prog} started ok"));
                            let s = Arc::new(Mutex::new(s));
                            map.insert(prog, s.clone());
                            s
                        }
                        None => {
                            trace(&format!("{prog} failed to start — degrade"));
                            return Vec::new(); // binary missing / spawn failed → degrade
                        }
                    },
                }
            };
            let mut s = server.lock().unwrap();
            let d = s.diagnostics(&abs, lang);
            trace(&format!("diagnostics({abs:?}) -> {} error(s)", d.len()));
            d
        })
        .await
        .unwrap_or_default()
    }
}

/// A running language server: its stdin, a framed-message receiver fed by a reader
/// thread, and the doc-version bookkeeping the LSP text-sync protocol needs. One server
/// can host several languageIds (e.g. typescript-language-server for `.ts`/`.tsx`), so
/// the languageId travels per-open, not on the server.
struct LspServer {
    stdin: ChildStdin,
    rx: Receiver<Value>,
    opened: HashMap<PathBuf, i64>,
    next_id: i64,
    _child: Child,
}

impl LspServer {
    /// Spawn `prog args`, run the `initialize`/`initialized` handshake, and leave it
    /// ready. `None` if the binary isn't on PATH or the handshake doesn't complete.
    fn start(prog: &str, args: &[&str], root: &Path) -> Option<LspServer> {
        let stderr = if std::env::var("AIVO_LSP_TRACE").is_ok() {
            Stdio::inherit()
        } else {
            Stdio::null()
        };
        let mut child = match Command::new(prog)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                trace(&format!("spawn {prog} failed: {e}"));
                return None;
            }
        };
        trace(&format!("{prog} spawned (pid {})", child.id()));
        let stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || read_loop(BufReader::new(stdout), tx));

        let mut server = LspServer {
            stdin,
            rx,
            opened: HashMap::new(),
            next_id: 1,
            _child: child,
        };
        match server.handshake(root) {
            Some(()) => {
                trace("handshake ok");
                Some(server)
            }
            None => {
                trace("handshake failed");
                None
            }
        }
    }

    fn handshake(&mut self, root: &Path) -> Option<()> {
        let id = self.next_id;
        self.next_id += 1;
        let init = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "processId": std::process::id(),
                "rootUri": file_uri(root),
                "capabilities": { "textDocument": { "publishDiagnostics": {} } },
            }
        });
        self.write(&init)?;
        // Drain until the initialize response arrives (bounded), then confirm.
        let deadline = Instant::now() + OVERALL_CAP;
        loop {
            let msg = self.rx.recv_timeout(remaining(deadline)?).ok()?;
            if msg.get("id").and_then(Value::as_i64) == Some(id) {
                break;
            }
        }
        self.write(&json!({ "jsonrpc": "2.0", "method": "initialized", "params": {} }))?;
        Some(())
    }

    /// (Re)open `path` (as `lang`) with its current bytes and return the settled error
    /// diagnostics.
    fn diagnostics(&mut self, path: &Path, lang: &str) -> Vec<Diagnostic> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        let uri = file_uri(path);
        let sent = match self.opened.get(path).copied() {
            None => {
                let ok = self.write(&json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": { "textDocument": {
                        "uri": uri, "languageId": lang, "version": 1, "text": text
                    }}
                }));
                self.opened.insert(path.to_path_buf(), 1);
                ok
            }
            Some(v) => {
                let version = v + 1;
                let ok = self.write(&json!({
                    "jsonrpc": "2.0",
                    "method": "textDocument/didChange",
                    "params": {
                        "textDocument": { "uri": uri, "version": version },
                        "contentChanges": [ { "text": text } ]
                    }
                }));
                self.opened.insert(path.to_path_buf(), version);
                ok
            }
        };
        if sent.is_none() {
            return Vec::new();
        }
        self.collect_diagnostics(&uri)
    }

    /// Drain messages until `publishDiagnostics` for `uri` stops arriving (quiet gap) or
    /// the overall cap elapses; return the LAST-seen error set (the settled one).
    fn collect_diagnostics(&self, uri: &str) -> Vec<Diagnostic> {
        let overall_deadline = Instant::now() + OVERALL_CAP;
        let mut latest: Option<Vec<Diagnostic>> = None;
        loop {
            let wait = SETTLE_QUIET.min(remaining(overall_deadline).unwrap_or_default());
            if wait.is_zero() {
                break;
            }
            match self.rx.recv_timeout(wait) {
                Ok(msg) => {
                    if let Some(m) = msg.get("method").and_then(Value::as_str) {
                        trace(&format!(
                            "  <- {m} (uri={:?})",
                            msg.pointer("/params/uri").and_then(Value::as_str)
                        ));
                    }
                    if msg.get("method").and_then(Value::as_str)
                        == Some("textDocument/publishDiagnostics")
                        && msg.pointer("/params/uri").and_then(Value::as_str) == Some(uri)
                    {
                        latest = Some(parse_diagnostics(&msg));
                    }
                }
                // Quiet for SETTLE_QUIET: settled if we already have a set; else keep
                // waiting up to the overall cap (a cold server may not have replied yet).
                Err(RecvTimeoutError::Timeout) => {
                    if latest.is_some() || remaining(overall_deadline).is_none() {
                        break;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        latest.unwrap_or_default()
    }

    fn write(&mut self, msg: &Value) -> Option<()> {
        let body = msg.to_string();
        write!(self.stdin, "Content-Length: {}\r\n\r\n{}", body.len(), body).ok()?;
        self.stdin.flush().ok()
    }
}

/// Read Content-Length-framed JSON-RPC messages off the server's stdout, forwarding
/// each parsed value. Ends (dropping the sender) on EOF or a malformed frame.
fn read_loop<R: Read>(mut reader: BufReader<R>, tx: std::sync::mpsc::Sender<Value>) {
    use std::io::BufRead;
    loop {
        // Headers, terminated by a blank line; we only need Content-Length.
        let mut len = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return; // EOF
            }
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
                len = rest.trim().parse().unwrap_or(0);
            }
        }
        if len == 0 {
            continue;
        }
        let mut buf = vec![0u8; len];
        if reader.read_exact(&mut buf).is_err() {
            return;
        }
        match serde_json::from_slice::<Value>(&buf) {
            Ok(v) => {
                if tx.send(v).is_err() {
                    return; // receiver gone
                }
            }
            Err(_) => return,
        }
    }
}

/// Extract error-severity (LSP severity 1) diagnostics from a publishDiagnostics
/// message, converting 0-based LSP lines to 1-based for display.
fn parse_diagnostics(msg: &Value) -> Vec<Diagnostic> {
    let Some(items) = msg.pointer("/params/diagnostics").and_then(Value::as_array) else {
        return Vec::new();
    };
    items
        .iter()
        .filter(|d| d.get("severity").and_then(Value::as_i64) == Some(1))
        .filter_map(|d| {
            let line = d.pointer("/range/start/line").and_then(Value::as_u64)? as u32 + 1;
            let message = d.get("message").and_then(Value::as_str)?.trim().to_string();
            Some(Diagnostic { line, message })
        })
        .collect()
}

/// Render settled error diagnostics for one file as a compact block to append to the
/// tool result, or `None` if there are none. Capped so a flood can't blow up context.
pub fn format_block(path: &str, diags: &[Diagnostic]) -> Option<String> {
    if diags.is_empty() {
        return None;
    }
    let mut out = format!(
        "\n\n⚠ {} error(s) after your edit to {path} — fix these:",
        diags.len()
    );
    for d in diags.iter().take(20) {
        out.push_str(&format!(
            "\n  {}:{} {}",
            path,
            d.line,
            first_line(&d.message)
        ));
    }
    if diags.len() > 20 {
        out.push_str(&format!("\n  … and {} more", diags.len() - 20));
    }
    Some(out)
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

/// Opt-in stderr trace for diagnosing the LSP path (`AIVO_LSP_TRACE=1`).
fn trace(msg: &str) {
    if std::env::var("AIVO_LSP_TRACE").is_ok() {
        eprintln!("[lsp] {msg}");
    }
}

fn file_uri(path: &Path) -> String {
    // Minimal file URI; good enough for local paths the servers we drive accept.
    format!("file://{}", path.to_string_lossy())
}

/// Remaining time until `deadline`, or `None` if already elapsed.
fn remaining(deadline: Instant) -> Option<Duration> {
    deadline.checked_duration_since(Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_spec_maps_known_extensions_only() {
        assert_eq!(server_spec(Path::new("a.rs")).unwrap().0, "rust-analyzer");
        assert_eq!(server_spec(Path::new("a.go")).unwrap().0, "gopls");
        // TypeScript: .ts and .tsx share the server but carry distinct languageIds.
        let ts = server_spec(Path::new("a.ts")).unwrap();
        assert_eq!(ts.0, "typescript-language-server");
        assert_eq!(ts.2, "typescript");
        assert_eq!(
            server_spec(Path::new("a.tsx")).unwrap().2,
            "typescriptreact"
        );
        assert_eq!(
            server_spec(Path::new("a.mts")).unwrap().0,
            "typescript-language-server"
        );
        let py = server_spec(Path::new("a.py")).unwrap();
        assert_eq!(py.0, "pyright-langserver");
        assert_eq!(py.2, "python");
        assert!(server_spec(Path::new("a.txt")).is_none());
        assert!(server_spec(Path::new("noext")).is_none());
    }

    #[test]
    fn parse_diagnostics_keeps_errors_and_reindexes_lines() {
        let msg = json!({
            "method": "textDocument/publishDiagnostics",
            "params": { "uri": "file:///x.rs", "diagnostics": [
                { "severity": 1, "range": { "start": { "line": 4 } }, "message": "mismatched types" },
                { "severity": 2, "range": { "start": { "line": 9 } }, "message": "unused import" },
                { "severity": 1, "range": { "start": { "line": 0 } }, "message": "cannot find value" }
            ]}
        });
        let diags = parse_diagnostics(&msg);
        // Only the two errors (severity 1), warnings dropped; lines 0-based → 1-based.
        assert_eq!(diags.len(), 2);
        assert_eq!(
            diags[0],
            Diagnostic {
                line: 5,
                message: "mismatched types".into()
            }
        );
        assert_eq!(
            diags[1],
            Diagnostic {
                line: 1,
                message: "cannot find value".into()
            }
        );
    }

    #[test]
    fn format_block_is_none_when_clean_and_caps_the_flood() {
        assert!(format_block("x.rs", &[]).is_none());
        let many: Vec<Diagnostic> = (0..30)
            .map(|i| Diagnostic {
                line: i,
                message: "boom".into(),
            })
            .collect();
        let block = format_block("x.rs", &many).unwrap();
        assert!(block.contains("30 error(s)"));
        assert!(block.contains("… and 10 more"));
        assert!(block.contains("x.rs:"));
    }

    #[test]
    fn read_loop_frames_content_length_messages() {
        // Two framed messages back-to-back on a pipe → two parsed values in order.
        let a = json!({ "id": 1, "result": {} }).to_string();
        let b = json!({ "method": "x" }).to_string();
        let stream = format!(
            "Content-Length: {}\r\n\r\n{}Content-Length: {}\r\n\r\n{}",
            a.len(),
            a,
            b.len(),
            b
        );
        let (tx, rx) = std::sync::mpsc::channel();
        read_loop(
            BufReader::new(std::io::Cursor::new(stream.into_bytes())),
            tx,
        );
        assert_eq!(rx.recv().unwrap()["id"], json!(1));
        assert_eq!(rx.recv().unwrap()["method"], json!("x"));
        assert!(rx.recv().is_err()); // sender dropped at EOF
    }
}
