//! WebSocket tunnel client for `aivo logs share`. Connects to
//! `s.getaivo.dev/_tunnel` (or whatever `AIVO_SHARE_BASE_URL` points at),
//! registers, prints the public URL, and proxies incoming framed HTTP
//! requests back to the local share server. See
//! `s.getaivo.dev/protocol.md` for the wire format.
//!
//! v1 has no auto-reconnect: a dropped tunnel kills the share — closing the
//! local process *is* the unshare mechanism. Re-running `aivo logs share`
//! allocates a fresh slot.
//!
//! The `--debug-local-only` flag in `aivo logs share` skips this whole module
//! and binds the local server on 127.0.0.1 directly.

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use futures::{SinkExt, StreamExt};
use http::HeaderValue;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;

const SUBPROTOCOL: &str = "aivo-tunnel/1";
const DEFAULT_BASE_URL: &str = "https://s.getaivo.dev";
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const OUTBOUND_BUFFER: usize = 64;

/// Connect, register, and run the tunnel until either the server drops it
/// or the user hits Ctrl+C. Returns Ok on clean shutdown; Err on connect /
/// register failure or on a server-side reject.
pub async fn run_tunnel(local_base: String, open_in_browser: bool) -> Result<()> {
    let api_base =
        std::env::var("AIVO_SHARE_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let ws_endpoint = format!("{}/_tunnel", http_to_ws_url(&api_base)?);

    let mut req = ws_endpoint
        .as_str()
        .into_client_request()
        .with_context(|| format!("invalid AIVO_SHARE_BASE_URL: {api_base}"))?;
    req.headers_mut().insert(
        "Sec-WebSocket-Protocol",
        HeaderValue::from_static(SUBPROTOCOL),
    );

    // Connect + handshake takes a network roundtrip; show a spinner so the
    // user sees feedback between Enter and the success banner. The spinner
    // is killed in every exit branch (early returns and the success path);
    // `print_share_started` does the final line-clear.
    let (spinner, spinner_handle) = crate::style::start_spinner(Some(" preparing share…"));

    let connect = tokio::time::timeout(HANDSHAKE_TIMEOUT, connect_async(req)).await;
    let (mut ws, _resp) = match connect {
        Ok(Ok(ok)) => ok,
        Ok(Err(e)) => {
            crate::style::stop_spinner(&spinner);
            let _ = spinner_handle.await;
            return Err(anyhow!(e)).with_context(|| format!("failed to connect to {ws_endpoint}"));
        }
        Err(_) => {
            crate::style::stop_spinner(&spinner);
            let _ = spinner_handle.await;
            return Err(anyhow!("timed out connecting to {ws_endpoint}"));
        }
    };

    // 1. register
    let register = json!({
        "type": "register",
        "client": {
            "platform": std::env::consts::OS,
            "aivo_version": env!("CARGO_PKG_VERSION"),
        }
    });
    if let Err(e) = ws.send(Message::Text(register.to_string().into())).await {
        crate::style::stop_spinner(&spinner);
        let _ = spinner_handle.await;
        return Err(e.into());
    }

    // 2. await registered (or reject)
    let first = match recv_text(&mut ws, HANDSHAKE_TIMEOUT).await {
        Ok(s) => s,
        Err(e) => {
            crate::style::stop_spinner(&spinner);
            let _ = spinner_handle.await;
            return Err(e);
        }
    };
    let first: Value = serde_json::from_str(&first).context("malformed first frame")?;
    match first["type"].as_str() {
        Some("registered") => {}
        Some("reject") => {
            crate::style::stop_spinner(&spinner);
            let _ = spinner_handle.await;
            let reason = first["reason"].as_str().unwrap_or("(no reason)");
            return Err(anyhow!("server rejected tunnel: {reason}"));
        }
        other => {
            crate::style::stop_spinner(&spinner);
            let _ = spinner_handle.await;
            return Err(anyhow!(
                "expected 'registered' frame, got {:?}",
                other.unwrap_or("?")
            ));
        }
    }
    crate::style::stop_spinner(&spinner);
    let _ = spinner_handle.await;
    let public_url = first["url"].as_str().unwrap_or(api_base.as_str());
    crate::commands::share::print_share_started(public_url);
    if open_in_browser {
        let _ = crate::services::browser_open::open_url(public_url);
    }

    // 3. split the socket; spawn writer task so the read loop and the
    // request-handling tasks can both submit outbound frames concurrently
    // without contending on a single sink.
    let (out_tx, mut out_rx) = mpsc::channel::<Value>(OUTBOUND_BUFFER);
    let (mut sink, mut stream_rx) = ws.split();

    let writer = tokio::spawn(async move {
        while let Some(v) = out_rx.recv().await {
            if sink
                .send(Message::Text(v.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
        let _ = sink.send(Message::Close(None)).await;
    });

    // One reqwest client reused across all proxied requests; reqwest pools
    // connections to the local server so this is cheap.
    let http = reqwest::Client::builder()
        .no_proxy()
        .build()
        .context("build local http client")?;

    let mut exit_reason: Option<String> = None;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                let _ = out_tx
                    .send(json!({"type": "close", "reason": "user_ctrl_c"}))
                    .await;
                println!();
                break;
            }
            msg = stream_rx.next() => {
                let Some(msg) = msg else { break };
                let msg = match msg {
                    Ok(m) => m,
                    Err(e) => {
                        exit_reason = Some(format!("read error: {e}"));
                        break;
                    }
                };
                match msg {
                    Message::Text(t) => {
                        let frame: Value = match serde_json::from_str(t.as_str()) {
                            Ok(v) => v,
                            Err(_) => continue,
                        };
                        match frame["type"].as_str() {
                            Some("ping") => {
                                let _ = out_tx.send(json!({"type": "pong"})).await;
                            }
                            Some("request") => {
                                let id = frame["id"].as_u64().unwrap_or(0);
                                let path = frame["path"].as_str().unwrap_or("/").to_string();
                                let method =
                                    frame["method"].as_str().unwrap_or("GET").to_string();
                                let req_headers = frame["headers"].clone();
                                let local = local_base.clone();
                                let http = http.clone();
                                let tx = out_tx.clone();
                                tokio::spawn(async move {
                                    proxy_one(&http, &local, id, &method, &path, &req_headers, tx)
                                        .await;
                                });
                            }
                            Some("reject") => {
                                let reason = frame["reason"]
                                    .as_str()
                                    .unwrap_or("(no reason)")
                                    .to_string();
                                exit_reason = Some(format!("server rejected: {reason}"));
                                break;
                            }
                            Some("close") => break,
                            _ => {}
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }

    // Drain the writer so any queued frames (including the close) are sent.
    drop(out_tx);
    let _ = writer.await;

    if let Some(reason) = exit_reason {
        return Err(anyhow!(reason));
    }
    Ok(())
}

/// Handle one framed `request` end to end: GET the local share server,
/// frame the response head + body back over the tunnel.
async fn proxy_one(
    http: &reqwest::Client,
    local_base: &str,
    id: u64,
    method: &str,
    path: &str,
    req_headers: &Value,
    out_tx: mpsc::Sender<Value>,
) {
    let url = format!("{local_base}{path}");
    let mut builder = match method.to_ascii_uppercase().as_str() {
        "HEAD" => http.head(&url),
        _ => http.get(&url),
    };
    if let Some(obj) = req_headers.as_object() {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                builder = builder.header(k, s);
            }
        }
    }

    let resp = match builder.send().await {
        Ok(r) => r,
        Err(e) => {
            send_error(&out_tx, id, 502, &format!("local server: {e}")).await;
            return;
        }
    };

    let status = resp.status().as_u16();
    let mut hmap = serde_json::Map::new();
    for (name, value) in resp.headers() {
        if let Ok(s) = value.to_str() {
            hmap.insert(name.as_str().to_string(), Value::String(s.to_string()));
        }
    }
    let _ = out_tx
        .send(json!({
            "type": "response_head",
            "id": id,
            "status": status,
            "headers": Value::Object(hmap),
        }))
        .await;

    let body = match resp.bytes().await {
        Ok(b) => b,
        Err(_) => bytes::Bytes::new(),
    };
    let _ = out_tx
        .send(json!({
            "type": "response_chunk",
            "id": id,
            "body_b64": B64.encode(&body),
            "last": true,
        }))
        .await;
}

async fn send_error(out_tx: &mpsc::Sender<Value>, id: u64, status: u16, msg: &str) {
    let _ = out_tx
        .send(json!({
            "type": "response_head",
            "id": id,
            "status": status,
            "headers": {"content-type": "application/json"},
        }))
        .await;
    let body = format!("{{\"error\":\"{}\"}}", msg.replace('"', "\\\""));
    let _ = out_tx
        .send(json!({
            "type": "response_chunk",
            "id": id,
            "body_b64": B64.encode(body.as_bytes()),
            "last": true,
        }))
        .await;
}

async fn recv_text<S>(
    ws: &mut tokio_tungstenite::WebSocketStream<S>,
    timeout: Duration,
) -> Result<String>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let msg = tokio::time::timeout(timeout, ws.next())
        .await
        .map_err(|_| anyhow!("timeout waiting for frame"))?
        .ok_or_else(|| anyhow!("server closed before sending a frame"))?
        .context("ws read error")?;
    match msg {
        Message::Text(t) => Ok(t.to_string()),
        other => Err(anyhow!("expected text frame, got {other:?}")),
    }
}

fn http_to_ws_url(base: &str) -> Result<String> {
    let trimmed = base.trim_end_matches('/');
    if let Some(rest) = trimmed.strip_prefix("https://") {
        Ok(format!("wss://{rest}"))
    } else if let Some(rest) = trimmed.strip_prefix("http://") {
        Ok(format!("ws://{rest}"))
    } else {
        Err(anyhow!(
            "AIVO_SHARE_BASE_URL must start with http:// or https://: {base}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::http_to_ws_url;

    #[test]
    fn url_scheme_swap() {
        assert_eq!(
            http_to_ws_url("https://s.getaivo.dev").unwrap(),
            "wss://s.getaivo.dev"
        );
        assert_eq!(
            http_to_ws_url("http://127.0.0.1:8080").unwrap(),
            "ws://127.0.0.1:8080"
        );
        assert_eq!(
            http_to_ws_url("https://example.com/").unwrap(),
            "wss://example.com"
        );
        assert!(http_to_ws_url("ftp://example.com").is_err());
    }
}
