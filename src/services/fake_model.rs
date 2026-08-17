//! Deterministic scripted "model" for eval/CI. When `AIVO_AGENT_FAKE_SSE` points
//! at a script file, `aivo code -e` talks to this loopback server instead of a real
//! provider, so the *real* agent loop and *real* tool execution run against a fixed
//! sequence of model turns — no tokens, no flakiness, gate-able in CI.
//!
//! Script format: a JSON array of turns, each either a tool-call batch or a final
//! answer, optionally carrying an OpenAI-style `"usage"` object (scripts budget
//! breakers). Turns are replayed one per model call, in order; once exhausted a
//! terminal answer is repeated so an over-long loop converges instead of erroring.
//!
//! ```json
//! [
//!   {"tools": [{"name": "edit_file", "args": {"path": "calc.sh", "old_string": "-", "new_string": "+"}}]},
//!   {"tools": [{"name": "run_bash", "args": {"command": "sh run_tests.sh"}}]},
//!   {"text": "Fixed the subtraction bug and verified the tests print ok."}
//! ]
//! ```
//!
//! `AIVO_FAKE_CAPTURE=<path>` appends each request body as one JSON line, so
//! tests can assert what the engine sent per call.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

/// Convert one scripted turn to an OpenAI chat-completions SSE body the engine
/// consumes: `{"text": ...}` → a content delta, `{"tools": [...]}` → a tool-call
/// batch (each entry at its own `index`), optional `"usage"` riding the same chunk.
fn sse_body(turn: &Value) -> Result<String, String> {
    let mut chunk = if let Some(text) = turn.get("text").and_then(Value::as_str) {
        json!({"choices": [{"delta": {"content": text}}]})
    } else if let Some(tools) = turn.get("tools").and_then(Value::as_array) {
        if tools.is_empty() {
            return Err("fake_model: `tools` turn has no calls".to_string());
        }
        let entries: Vec<Value> = tools
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let name = t.get("name").and_then(Value::as_str).unwrap_or_default();
                let args = t.get("args").cloned().unwrap_or_else(|| json!({}));
                json!({
                    "index": i,
                    "id": format!("c{i}"),
                    "function": {"name": name, "arguments": args.to_string()},
                })
            })
            .collect();
        json!({"choices": [{"delta": {"tool_calls": entries}}]})
    } else {
        return Err(format!(
            "fake_model: each turn needs `text` or `tools`, got: {turn}"
        ));
    };
    if let Some(usage) = turn.get("usage") {
        if !usage.is_object() {
            return Err(format!(
                "fake_model: `usage` must be an object, got: {usage}"
            ));
        }
        chunk["usage"] = usage.clone();
    }
    Ok(format!("data: {chunk}\n\ndata: [DONE]\n\n"))
}

/// Served after the script is exhausted so an extra model call converges cleanly.
fn terminal_body() -> String {
    let delta = json!({"choices": [{"delta": {"content": "(scripted run complete)"}}]});
    format!("data: {delta}\n\ndata: [DONE]\n\n")
}

/// Parse a fake-model script (JSON array of turns) into ready-to-serve SSE bodies.
pub fn parse_script(json_str: &str) -> Result<Vec<String>, String> {
    let parsed: Value =
        serde_json::from_str(json_str).map_err(|e| format!("fake_model: invalid JSON: {e}"))?;
    let turns = parsed
        .as_array()
        .ok_or("fake_model: script must be a JSON array of turns")?;
    if turns.is_empty() {
        return Err("fake_model: script has no turns".to_string());
    }
    turns.iter().map(sse_body).collect()
}

/// Load and parse the script at `path`.
pub fn load_script(path: &str) -> Result<Vec<String>, String> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("fake_model: cannot read script {path}: {e}"))?;
    parse_script(&raw)
}

/// Read headers + `Content-Length` body so the client is never blocked writing;
/// timeout-bounded so a malformed request can't hang the server.
fn read_request(sock: &mut TcpStream) -> String {
    let _ = sock.set_read_timeout(Some(std::time::Duration::from_secs(2)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 16384];
    let mut header_end = None;
    loop {
        match sock.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if header_end.is_none() {
            header_end = buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4);
        }
        if let Some(end) = header_end
            && buf.len() >= end + content_length(&buf[..end])
        {
            break;
        }
    }
    match header_end {
        Some(end) => String::from_utf8_lossy(&buf[end..]).into_owned(),
        None => String::new(),
    }
}

/// `Content-Length` from raw headers; 0 when absent.
fn content_length(headers: &[u8]) -> usize {
    String::from_utf8_lossy(headers)
        .lines()
        .find_map(|l| {
            let (name, value) = l.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())?
        })
        .unwrap_or(0)
}

fn append_capture(path: &Path, body: &str) {
    let line = match serde_json::from_str::<Value>(body) {
        Ok(v) => v.to_string(),
        Err(_) => json!({ "unparsed": body }).to_string(),
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

/// Start a background scripted-model server; returns its loopback port. Replays
/// `bodies` in order (one per request), repeating a terminal answer afterward.
/// The thread is detached and dies with the process — fine for a one-shot `-e` run.
pub fn start(bodies: Vec<String>) -> std::io::Result<u16> {
    let capture = std::env::var("AIVO_FAKE_CAPTURE").ok().map(PathBuf::from);
    start_with_capture(bodies, capture)
}

/// [`start`] with an explicit capture path (env-race-free for in-process tests).
pub fn start_with_capture(bodies: Vec<String>, capture: Option<PathBuf>) -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    let trace = std::env::var("AIVO_FAKE_TRACE").is_ok();
    std::thread::spawn(move || {
        let terminal = terminal_body();
        let mut i = 0usize;
        loop {
            let Ok((mut sock, _)) = listener.accept() else {
                break;
            };
            if trace {
                eprintln!("[fake_model] accepted request {i}");
            }
            let request_body = read_request(&mut sock);
            if let Some(path) = &capture {
                append_capture(path, &request_body);
            }
            let body = bodies.get(i).unwrap_or(&terminal);
            if trace {
                eprintln!(
                    "[fake_model] replying to request {i} ({} bytes)",
                    body.len()
                );
            }
            i += 1;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_turn_becomes_a_content_delta() {
        let body = sse_body(&json!({"text": "all done"})).unwrap();
        assert!(body.contains(r#""content":"all done""#));
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }

    #[test]
    fn tools_turn_becomes_indexed_tool_calls() {
        let body = sse_body(&json!({"tools": [
            {"name": "run_bash", "args": {"command": "ls"}},
            {"name": "read_file", "args": {"path": "a.rs"}},
        ]}))
        .unwrap();
        assert!(body.contains(r#""name":"run_bash""#));
        assert!(body.contains(r#""index":0"#));
        assert!(body.contains(r#""name":"read_file""#));
        assert!(body.contains(r#""index":1"#));
        // args are serialized as a JSON string, as the OpenAI wire expects.
        assert!(body.contains(r#""arguments":"{\"command\":\"ls\"}""#));
    }

    #[test]
    fn usage_rides_the_turn_chunk() {
        let body = sse_body(&json!({
            "tools": [{"name": "run_bash", "args": {"command": "ls"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 42},
        }))
        .unwrap();
        assert!(body.contains(r#""completion_tokens":42"#));
        assert!(body.contains(r#""name":"run_bash""#));
        assert!(sse_body(&json!({"text": "x", "usage": 5})).is_err());
    }

    /// Drive the real server over a loopback socket and read back the raw response.
    fn raw_post(port: u16) -> String {
        raw_post_body(port, "{}")
    }

    fn raw_post_body(port: u16, body: &str) -> String {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.write_all(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .unwrap();
        s.flush().unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    }

    #[test]
    fn capture_records_each_request_body_in_order() {
        let bodies = parse_script(r#"[{"text": "one"}, {"text": "two"}]"#).unwrap();
        let path =
            std::env::temp_dir().join(format!("aivo-fake-capture-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let port = start_with_capture(bodies, Some(path.clone())).unwrap();

        raw_post_body(
            port,
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
        );
        raw_post_body(port, r#"{"model":"m","messages":[1,2]}"#);

        let raw = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<Value> = raw
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["messages"][0]["content"], "hi");
        assert_eq!(lines[1]["messages"], json!([1, 2]));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn server_replays_bodies_in_order_then_a_terminal_turn() {
        let bodies = parse_script(
            r#"[
                {"tools": [{"name": "run_bash", "args": {"command": "ls"}}]},
                {"text": "all done"}
            ]"#,
        )
        .unwrap();
        let port = start(bodies).unwrap();

        // Request 1 → the scripted tool call, framed as event-stream with a DONE.
        let r1 = raw_post(port);
        assert!(r1.starts_with("HTTP/1.1 200 OK"));
        assert!(r1.contains("text/event-stream"));
        assert!(r1.contains(r#""name":"run_bash""#));
        assert!(r1.contains("data: [DONE]"));

        // Request 2 → the scripted final answer.
        let r2 = raw_post(port);
        assert!(r2.contains(r#""content":"all done""#));

        // Request 3 → past the end, a terminal answer so an over-long loop converges.
        let r3 = raw_post(port);
        assert!(r3.contains("scripted run complete"));
    }

    #[test]
    fn parse_script_rejects_bad_shapes() {
        assert!(parse_script("not json").is_err());
        assert!(parse_script("{}").is_err()); // not an array
        assert!(parse_script("[]").is_err()); // empty
        assert!(parse_script(r#"[{"nope": 1}]"#).is_err()); // neither text nor tools
        assert!(parse_script(r#"[{"tools": []}]"#).is_err()); // empty batch
    }

    #[test]
    fn parse_script_accepts_a_valid_sequence() {
        let bodies = parse_script(
            r#"[
                {"tools": [{"name": "edit_file", "args": {"path": "x"}}]},
                {"text": "done"}
            ]"#,
        )
        .unwrap();
        assert_eq!(bodies.len(), 2);
        assert!(bodies[0].contains(r#""name":"edit_file""#));
        assert!(bodies[1].contains(r#""content":"done""#));
    }
}
