//! The capability-gated plugin handoff, end-to-end: a granted `coding-agent`
//! plugin gets the key/endpoint env and is accounted in stats; an ungranted one
//! gets neither. The `aivo` binary runs as a child; an in-process mock stands in
//! as the upstream provider the loopback endpoint forwards to.
#![cfg(unix)]

mod support;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use serde_json::Value;
use tempfile::TempDir;

fn aivo_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_aivo") {
        return PathBuf::from(path);
    }
    let mut path = std::env::current_exe().expect("current test exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("aivo");
    path
}

/// `aivo` child wired to a temp HOME and no proxy (so it — and the plugin's curl
/// — reach the loopback mock).
fn aivo(home: &TempDir) -> Command {
    let mut cmd = Command::new(aivo_bin());
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("AIVO_TEST_FAST_CRYPTO_OK", "1")
        .env("NO_COLOR", "1")
        .env("HTTP_PROXY", "")
        .env("HTTPS_PROXY", "")
        .env("NO_PROXY", "127.0.0.1,localhost");
    cmd
}

/// Serve a single fixed response on every connection from a background thread.
fn serve_upstream(listener: TcpListener, content_type: &str, body: Vec<u8>) {
    let content_type = content_type.to_string();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            // Drain headers + any request body framing up to the blank line.
            loop {
                let mut h = String::new();
                match reader.read_line(&mut h) {
                    Ok(0) => break,
                    Ok(_) if h == "\r\n" || h == "\n" => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        }
    });
}

fn bind() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

#[derive(Debug)]
struct CapturedRequest {
    line: String,
    body: String,
}

/// Serve one fixed API response and return the first non-models upstream
/// request the router made. `keys add` may kick off a background `/models`
/// sync before the plugin launches, so those probes are answered and ignored.
fn serve_upstream_capture_once(
    listener: TcpListener,
    content_type: &str,
    body: Vec<u8>,
) -> mpsc::Receiver<CapturedRequest> {
    let content_type = content_type.to_string();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            let mut content_len = 0usize;
            loop {
                let mut h = String::new();
                match reader.read_line(&mut h) {
                    Ok(0) => break,
                    Ok(_) if h == "\r\n" || h == "\n" => break,
                    Ok(_) => {
                        if let Some((name, value)) = h.split_once(':')
                            && name.eq_ignore_ascii_case("content-length")
                        {
                            content_len = value.trim().parse().unwrap_or(0);
                        }
                    }
                    Err(_) => break,
                }
            }
            let mut req_body = vec![0u8; content_len];
            let _ = reader.read_exact(&mut req_body);
            if line.contains("/models") {
                let models = br#"{"object":"list","data":[{"id":"gpt-x","object":"model","owned_by":"test"}]}"#;
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    models.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(models);
                let _ = stream.flush();
                continue;
            }
            let _ = tx.send(CapturedRequest {
                line,
                body: String::from_utf8_lossy(&req_body).to_string(),
            });
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
            let _ = stream.flush();
            break;
        }
    });
    rx
}

/// Serve different `(status, body)` by request path: `/responses` requests get
/// `responses_resp`, everything else (chat/completions) gets `chat_resp`. Loops
/// over connections — the router retries `/responses` on a fresh connection.
fn serve_upstream_by_path(
    listener: TcpListener,
    chat_resp: (u16, String),
    responses_resp: (u16, String),
) {
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            let _ = reader.read_line(&mut line);
            let is_responses = line.contains("/responses");
            loop {
                let mut h = String::new();
                match reader.read_line(&mut h) {
                    Ok(0) => break,
                    Ok(_) if h == "\r\n" || h == "\n" => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            let (status, body) = if is_responses {
                &responses_resp
            } else {
                &chat_resp
            };
            let reason = if *status == 200 { "OK" } else { "Bad Request" };
            let header = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
    });
}

/// Write an `aivo-<name>` plugin that self-describes as `coding-agent` requesting
/// `endpoint`, echoes its handoff env, and (if curl is present) drives one
/// non-streaming request through the loopback endpoint.
fn write_plugin(dir: &Path, name: &str) -> PathBuf {
    write_plugin_with(dir, name, false)
}

/// Like [`write_plugin`] but the driven request sets `stream` to `streaming` —
/// the real omp case is `stream:true`, which exercises the streaming code paths.
fn write_plugin_with(dir: &Path, name: &str, streaming: bool) -> PathBuf {
    let path = dir.join(format!("aivo-{name}"));
    let script = format!(
        r#"#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  printf '%s\n' '{{"name":"{name}","version":"1.0.0","protocol":"1","type":"coding-agent","capabilities":["endpoint"]}}'
  exit 0
fi
echo "KEY=$AIVO_KEY"
echo "BASE=$AIVO_KEY_BASE_URL"
echo "ENDPOINT=$AIVO_ENDPOINT_URL"
echo "TOKEN=$AIVO_ENDPOINT_TOKEN"
echo "DEBUG=$AIVO_DEBUG_LOG"
echo "ARGS=$*"
if [ -n "$AIVO_ENDPOINT_URL" ] && command -v curl >/dev/null 2>&1; then
  # AIVO_ENDPOINT_URL is an OpenAI-style base (ends in /v1); append /chat/completions.
  # Echo the response so the test can see which upstream (i.e. which key) answered.
  # Collapse newlines so multi-line SSE (stream:true) lands on the single RESP= line.
  RESP=$(curl -s -X POST "$AIVO_ENDPOINT_URL/chat/completions" \
    -H "Authorization: Bearer $AIVO_ENDPOINT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{{"model":"gpt-x","messages":[{{"role":"user","content":"hi"}}],"stream":{streaming}}}' 2>/dev/null | tr '\n' ' ')
  echo "RESP=$RESP"
fi
exit 0
"#
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// A plugin that drives the injected endpoint as a Responses API client, with a
/// built-in Responses tool that must survive first-request routing.
fn write_responses_plugin(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(format!("aivo-{name}"));
    let script = format!(
        r#"#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  printf '%s\n' '{{"name":"{name}","version":"1.0.0","protocol":"1","type":"coding-agent","capabilities":["endpoint"]}}'
  exit 0
fi
if [ -n "$AIVO_ENDPOINT_URL" ] && command -v curl >/dev/null 2>&1; then
  RESP=$(curl -s -X POST "$AIVO_ENDPOINT_URL/responses" \
    -H "Authorization: Bearer $AIVO_ENDPOINT_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{{"model":"gpt-x","input":[{{"role":"user","content":"hi"}}],"tools":[{{"type":"web_search_preview"}}],"stream":false}}' 2>/dev/null | tr '\n' ' ')
  echo "RESP=$RESP"
fi
exit 0
"#
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn stats(home: &TempDir) -> Value {
    let path = home.path().join(".config/aivo/state/stats.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("stats.json")).unwrap()
}

fn add_active_key(home: &TempDir, base_url: &str) {
    let out = aivo(home)
        .args([
            "keys",
            "add",
            "--name",
            "up",
            "--base-url",
            base_url,
            "--key",
            "sk-up",
        ])
        .output()
        .expect("spawn aivo keys add");
    assert!(
        out.status.success(),
        "keys add failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn have_curl() -> bool {
    Command::new("curl")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Add a named key with a distinct base URL (not activated by `use`).
fn add_key(home: &TempDir, name: &str, key: &str, base_url: &str) {
    let out = aivo(home)
        .args([
            "keys",
            "add",
            "--name",
            name,
            "--base-url",
            base_url,
            "--key",
            key,
        ])
        .output()
        .expect("spawn aivo keys add");
    assert!(
        out.status.success(),
        "keys add {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Activate `name` as the active key.
fn use_key(home: &TempDir, name: &str) {
    let out = aivo(home)
        .args(["keys", "use", name])
        .output()
        .expect("spawn aivo keys use");
    assert!(
        out.status.success(),
        "keys use {name} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A single-line chat completion whose content is `tag`, so a test can tell
/// which upstream answered. Carries a usage block for accounting.
fn completion_body(tag: &str) -> Vec<u8> {
    serde_json::json!({
        "id": tag,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": tag}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    })
    .to_string()
    .into_bytes()
}

/// Run the installed `omp` plugin and return the endpoint response it echoed
/// (`RESP=…`) — the upstream that answered reveals which key resolved.
fn run_omp_resp(home: &TempDir, args: &[&str]) -> Option<String> {
    let run = aivo(home).args(args).output().expect("spawn omp");
    assert!(
        run.status.success(),
        "omp failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("RESP=").map(str::to_string))
        .filter(|s| !s.is_empty())
}

#[test]
fn granted_coding_agent_gets_key_endpoint_and_is_accounted() {
    // Upstream that answers chat completions with a token-bearing usage block.
    let (listener, up_port) = bind();
    let completion = serde_json::json!({
        "id": "c1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    })
    .to_string();
    serve_upstream(listener, "application/json", completion.into_bytes());

    let home = TempDir::new().unwrap();
    add_active_key(&home, &format!("http://127.0.0.1:{up_port}/v1"));

    // Install the plugin locally with --trust so the grant is non-interactive.
    let work = TempDir::new().unwrap();
    let plugin = write_plugin(work.path(), "omp");
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .arg("--trust")
        .output()
        .expect("spawn install");
    assert!(
        install.status.success(),
        "install failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&install.stdout),
        String::from_utf8_lossy(&install.stderr),
    );
    // Disclosure shows the type and the granted endpoint cap.
    let install_err = String::from_utf8_lossy(&install.stderr);
    assert!(install_err.contains("coding-agent"), "{install_err}");
    assert!(install_err.contains("granted: endpoint"), "{install_err}");

    // Run it — the handoff env must be injected.
    let run = aivo(&home).args(["omp"]).output().expect("spawn omp");
    assert!(
        run.status.success(),
        "omp failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let out = String::from_utf8_lossy(&run.stdout);
    // No raw key/base is ever injected now — only the endpoint handoff.
    assert!(
        !out.contains("KEY=sk-up"),
        "raw key must not be injected:\n{out}"
    );
    // An OpenAI-style base URL ending in /v1 (so clients append /chat/completions).
    assert!(
        out.lines()
            .any(|l| l.starts_with("ENDPOINT=http://127.0.0.1:") && l.ends_with("/v1")),
        "endpoint url not injected as an OpenAI /v1 base:\n{out}"
    );
    assert!(
        out.lines()
            .any(|l| l.starts_with("TOKEN=") && l.len() > "TOKEN=".len()),
        "endpoint token not injected:\n{out}"
    );

    // coding-agent ⇒ the launch is counted in stats.
    let s = stats(&home);
    assert_eq!(
        s["toolCounts"]["omp"].as_u64(),
        Some(1),
        "plugin launch not recorded in toolCounts: {s}"
    );

    // When curl is available the plugin routed through the endpoint, so token
    // usage is accounted against the key (skip_if_zero ⇒ presence means non-zero).
    if have_curl() {
        let raw =
            std::fs::read_to_string(home.path().join(".config/aivo/state/stats.json")).unwrap();
        assert!(
            raw.contains("promptTokens"),
            "token usage not recorded at the endpoint:\n{raw}"
        );
    }
}

/// A coding-agent plugin launched with an explicit `-k` records that key as the
/// global last selection — native `aivo run -k` parity. A *thin* plugin drives
/// the handoff from the injected env, but a *fat* plugin (amp) re-resolves the
/// key from the store itself, so it never sees the `-k` aivo strips from its
/// argv unless it's persisted. Proven black-box through the endpoint: each key
/// proxies to a distinct upstream, so the echoed RESP reveals which key resolved.
/// After `omp -k up2`, even a *bare* `omp` resolves to up2, not the active up1.
#[test]
fn explicit_key_on_coding_agent_persists_last_selection() {
    if !have_curl() {
        return; // the round-trip proof drives the endpoint with curl
    }
    // Two reachable upstreams with distinct bodies; the endpoint proxies to
    // whichever key resolved. The picker is skipped (child stderr isn't a TTY),
    // so no model fetch is made.
    let (l1, p1) = bind();
    serve_upstream(l1, "application/json", completion_body("UP1"));
    let (l2, p2) = bind();
    serve_upstream(l2, "application/json", completion_body("UP2"));

    let home = TempDir::new().unwrap();
    add_key(&home, "up1", "sk-up1", &format!("http://127.0.0.1:{p1}/v1"));
    add_key(&home, "up2", "sk-up2", &format!("http://127.0.0.1:{p2}/v1"));
    use_key(&home, "up1");

    let work = TempDir::new().unwrap();
    let plugin = write_plugin(work.path(), "omp");
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .arg("--trust")
        .output()
        .expect("spawn install");
    assert!(
        install.status.success(),
        "install failed:\n{}",
        String::from_utf8_lossy(&install.stderr)
    );

    // A bare run resolves the active key up1 …
    assert!(
        run_omp_resp(&home, &["omp"])
            .unwrap_or_default()
            .contains("UP1"),
        "bare run should use the active key up1"
    );
    // … an explicit `-k up2` selects up2 …
    assert!(
        run_omp_resp(&home, &["omp", "-k", "up2"])
            .unwrap_or_default()
            .contains("UP2"),
        "explicit -k should select up2"
    );
    // … and the fix persists it: a later *bare* run now resolves to up2, not the
    // originally-active up1 — the fat-plugin path the host fix enables.
    assert!(
        run_omp_resp(&home, &["omp"])
            .unwrap_or_default()
            .contains("UP2"),
        "explicit -k was not persisted as last_selection"
    );
}

#[test]
fn ungranted_plugin_gets_no_secret_env() {
    let home = TempDir::new().unwrap();
    // An active key exists — proving it's the *grant* (not a missing key) that
    // withholds the handoff.
    add_active_key(&home, "https://api.openai.com/v1");

    let work = TempDir::new().unwrap();
    let plugin = write_plugin(work.path(), "omp");
    // No --trust and a non-interactive stdin ⇒ no caps granted.
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .output()
        .expect("spawn install");
    assert!(install.status.success());

    let run = aivo(&home).args(["omp"]).output().expect("spawn omp");
    assert!(run.status.success());
    let out = String::from_utf8_lossy(&run.stdout);
    // No key, no endpoint handed over.
    assert!(
        !out.contains("sk-up"),
        "raw key leaked without grant:\n{out}"
    );
    assert!(
        !out.contains("ENDPOINT=http"),
        "endpoint handed over without grant:\n{out}"
    );
    // Still a coding-agent ⇒ the run is logged in stats regardless of the grant.
    assert_eq!(stats(&home)["toolCounts"]["omp"].as_u64(), Some(1));
}

#[test]
fn coding_agent_plugin_consumes_debug_flag_like_native_tool() {
    let home = TempDir::new().unwrap();

    let work = TempDir::new().unwrap();
    let plugin = write_plugin(work.path(), "omp");
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .arg("--trust")
        .output()
        .expect("spawn install");
    assert!(install.status.success());

    let run = aivo(&home)
        .args(["omp", "--debug", "-p", "hi"])
        .output()
        .expect("spawn omp");
    assert!(run.status.success());
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        out.lines()
            .any(|l| l.starts_with("DEBUG=") && l.ends_with(".jsonl")),
        "debug log env not handed to plugin:\n{out}"
    );
    assert!(out.contains("ARGS=-p hi"), "argv not preserved:\n{out}");
    assert!(
        !out.contains("ARGS=--debug"),
        "--debug leaked to coding-agent plugin argv:\n{out}"
    );
}

/// An OpenAI-protocol key routes through the responses-capable internal proxy:
/// when the upstream rejects /chat/completions with a "use /v1/responses" 400
/// (gpt-5.x reasoning + tools), the endpoint escalates to /responses and returns
/// the (converted) result — instead of relaying the 400.
#[test]
fn responses_only_model_escalates_chat_to_responses() {
    if !have_curl() {
        return;
    }
    let (listener, port) = bind();
    let chat_400 = (
        400,
        serde_json::json!({"error": {
            "message": "Function tools with reasoning_effort are not supported for gpt-x in /v1/chat/completions. Please use /v1/responses instead.",
            "type": "invalid_request_body"
        }})
        .to_string(),
    );
    let responses_200 = (
        200,
        serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "model": "gpt-x",
            "output": [{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "VIA_RESPONSES"}]}],
            "usage": {"input_tokens": 11, "output_tokens": 7}
        })
        .to_string(),
    );
    serve_upstream_by_path(listener, chat_400, responses_200);

    let home = TempDir::new().unwrap();
    add_active_key(&home, &format!("http://127.0.0.1:{port}/v1"));

    let work = TempDir::new().unwrap();
    // `stream:true` — the real omp case (a streaming chat request whose buffered
    // /responses escalation must not be parsed as SSE).
    let plugin = write_plugin_with(work.path(), "omp", true);
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .arg("--trust")
        .output()
        .expect("spawn install");
    assert!(
        install.status.success(),
        "install failed:\n{}",
        String::from_utf8_lossy(&install.stderr)
    );

    // The chat request is rejected upstream; the endpoint escalates to /responses
    // and the converted body (content "VIA_RESPONSES") comes back to the plugin.
    let resp = run_omp_resp(&home, &["omp"]).unwrap_or_default();
    assert!(
        resp.contains("VIA_RESPONSES"),
        "endpoint should escalate streaming chat\u{2192}/responses; got: {resp}"
    );
}

#[test]
fn responses_request_uses_native_responses_first_and_preserves_tools() {
    if !have_curl() {
        return;
    }
    let (listener, port) = bind();
    let native_response = serde_json::json!({
        "id": "resp_native",
        "object": "response",
        "model": "gpt-x",
        "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "NATIVE_RESPONSES"}]}],
        "usage": {"input_tokens": 3, "output_tokens": 2}
    })
    .to_string()
    .into_bytes();
    let captured = serve_upstream_capture_once(listener, "application/json", native_response);

    let home = TempDir::new().unwrap();
    add_active_key(&home, &format!("http://127.0.0.1:{port}/v1"));

    let work = TempDir::new().unwrap();
    let plugin = write_responses_plugin(work.path(), "omp");
    let install = aivo(&home)
        .args(["plugins", "install"])
        .arg(&plugin)
        .arg("--trust")
        .output()
        .expect("spawn install");
    assert!(
        install.status.success(),
        "install failed:\n{}",
        String::from_utf8_lossy(&install.stderr)
    );

    let resp = run_omp_resp(&home, &["omp"]).unwrap_or_default();
    let req = captured
        .recv_timeout(Duration::from_secs(5))
        .expect("upstream request was not captured");
    assert!(
        req.line.contains("/v1/responses"),
        "first upstream request should be native Responses, got: {:?}",
        req.line
    );
    assert!(
        req.body.contains("web_search_preview"),
        "Responses built-in tool was stripped before upstream:\n{}",
        req.body
    );
    assert!(
        resp.contains("NATIVE_RESPONSES"),
        "plugin did not receive native Responses payload: {resp}"
    );
}
