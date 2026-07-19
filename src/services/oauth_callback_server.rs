//! Single-shot local HTTP server that captures an OAuth redirect on
//! loopback. Two bind modes cover both flows that use it: a fixed
//! registered port that is part of the provider's redirect URI (Codex
//! ChatGPT), and an ephemeral port the caller threads into the authorize
//! URL before printing it (MCP OAuth).
//!
//! Reads the first request line, pulls `code` + `state` from the query
//! string, returns a small success HTML to the browser. Resolves on the
//! first valid callback or on timeout; rejects mismatched `state`.

use anyhow::{Context, Result, anyhow};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Fixed-port bind failure. Distinct so callers can fall back to a
/// manual-paste flow instead of bailing hard.
#[derive(Debug)]
pub struct PortUnavailable {
    pub port: u16,
}

impl std::fmt::Display for PortUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "127.0.0.1:{} is in use; falling back to manual paste",
            self.port
        )
    }
}

impl std::error::Error for PortUnavailable {}

/// Provider-specific shape of the redirect: the path the provider redirects
/// to and the success-page strings shown in the browser.
pub struct CallbackSpec {
    pub path: &'static str,
    pub page_title: &'static str,
    pub page_heading: &'static str,
}

pub struct CallbackOutcome {
    pub code: String,
}

#[derive(Debug)]
pub struct CallbackServer {
    listener: TcpListener,
    port: u16,
}

impl CallbackServer {
    /// Binds an ephemeral loopback port; read it back via [`Self::port`] to
    /// embed in the authorize URL before waiting.
    pub async fn bind_ephemeral() -> Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .context("bind OAuth callback listener on 127.0.0.1:0")?;
        let port = listener.local_addr().context("resolve bound port")?.port();
        Ok(Self { listener, port })
    }

    /// Binds a fixed registered port. Port-in-use maps to [`PortUnavailable`]
    /// so callers can offer manual paste.
    pub async fn bind_fixed(port: u16) -> Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port)).await.map_err(|e| {
            if matches!(
                e.kind(),
                std::io::ErrorKind::AddrInUse | std::io::ErrorKind::PermissionDenied
            ) {
                anyhow!(PortUnavailable { port })
            } else {
                anyhow!(e).context("bind OAuth callback listener")
            }
        })?;
        Ok(Self { listener, port })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Waits for one valid hit on `spec.path`.
    ///
    /// On a state mismatch or error parameter the browser sees a 400 and the
    /// call returns `Err`. On timeout returns `Err`. Other paths
    /// (`/favicon.ico`, etc.) are 404'd and the loop continues.
    pub async fn wait_for_callback(
        self,
        spec: &CallbackSpec,
        expected_state: &str,
        timeout: Duration,
    ) -> Result<CallbackOutcome> {
        tokio::time::timeout(timeout, accept_one(self.listener, spec, expected_state))
            .await
            .map_err(|_| anyhow!("timed out waiting for OAuth callback"))?
    }
}

async fn accept_one(
    listener: TcpListener,
    spec: &CallbackSpec,
    expected_state: &str,
) -> Result<CallbackOutcome> {
    loop {
        let (mut stream, _) = listener.accept().await.context("accept OAuth callback")?;
        let request_line = match read_request_line(&mut stream).await {
            Ok(line) => line,
            Err(_) => {
                let _ = stream.shutdown().await;
                continue;
            }
        };

        let path_and_query = parse_request_target(&request_line);

        if !path_and_query.starts_with(spec.path) {
            respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found").await;
            continue;
        }

        let query = path_and_query.split_once('?').map(|(_, q)| q).unwrap_or("");
        let (code, state, error) = extract_callback_params(query);

        if let Some(err) = error {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                format!("OAuth error: {err}").as_bytes(),
            )
            .await;
            return Err(anyhow!("OAuth provider returned error: {err}"));
        }

        if state.as_deref() != Some(expected_state) {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                b"state mismatch",
            )
            .await;
            return Err(anyhow!("OAuth callback state mismatch"));
        }

        let code = code.ok_or_else(|| anyhow!("OAuth callback missing `code`"))?;

        respond(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            success_html(spec).as_bytes(),
        )
        .await;
        return Ok(CallbackOutcome { code });
    }
}

/// Reads up to the first CRLF (or plain LF fallback). Bounded to 8 KiB so a
/// misbehaving client can't eat memory.
async fn read_request_line(stream: &mut tokio::net::TcpStream) -> Result<String> {
    let mut buf = [0u8; 8192];
    let mut total = 0usize;
    loop {
        let n = stream.read(&mut buf[total..]).await?;
        if n == 0 {
            break;
        }
        total += n;
        if let Some(end) = find_line_end(&buf[..total]) {
            return Ok(String::from_utf8_lossy(&buf[..end]).into_owned());
        }
        if total == buf.len() {
            break;
        }
    }
    Err(anyhow!("request line missing or too long"))
}

fn find_line_end(bytes: &[u8]) -> Option<usize> {
    for i in 0..bytes.len() {
        if bytes[i] == b'\n' {
            let end = if i > 0 && bytes[i - 1] == b'\r' {
                i - 1
            } else {
                i
            };
            return Some(end);
        }
    }
    None
}

fn parse_request_target(request_line: &str) -> &str {
    // "GET /auth/callback?code=...&state=... HTTP/1.1"
    let mut parts = request_line.split_whitespace();
    let _method = parts.next();
    parts.next().unwrap_or("")
}

/// Returns `(code, state, error)` from a url-encoded query string.
pub(crate) fn extract_callback_params(
    query: &str,
) -> (Option<String>, Option<String>, Option<String>) {
    let mut code = None;
    let mut state = None;
    let mut error = None;
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some(kv) => kv,
            None => (pair, ""),
        };
        let decoded = crate::services::percent_codec::decode(v);
        match k {
            "code" => code = Some(decoded),
            "state" => state = Some(decoded),
            "error" => error = Some(decoded),
            "error_description" if error.is_none() => error = Some(decoded),
            _ => {}
        }
    }
    (code, state, error)
}

async fn respond(stream: &mut tokio::net::TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "",
    };
    let head = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         X-Frame-Options: DENY\r\n\
         X-Content-Type-Options: nosniff\r\n\
         Cache-Control: no-store\r\n\
         \r\n",
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).await;
    let _ = stream.write_all(body).await;
    let _ = stream.shutdown().await;
}

fn success_html(spec: &CallbackSpec) -> String {
    SUCCESS_HTML_TEMPLATE
        .replace("__TITLE__", spec.page_title)
        .replace("__HEADING__", spec.page_heading)
}

const SUCCESS_HTML_TEMPLATE: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #0b0b0e; color: #e5e7eb; display: flex; align-items: center;
           justify-content: center; height: 100vh; margin: 0; }
    .card { text-align: center; padding: 2rem 3rem; border: 1px solid #2a2a31;
            border-radius: 12px; background: #141418; }
    h1 { margin: 0 0 .5rem; font-size: 1.25rem; }
    p { margin: 0; color: #9ca3af; font-size: .95rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>__HEADING__</h1>
    <p>You can close this tab and return to your terminal.</p>
  </div>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: CallbackSpec = CallbackSpec {
        path: "/oauth2callback",
        page_title: "aivo — authorized",
        page_heading: "Authorized.",
    };

    #[tokio::test]
    async fn bind_ephemeral_assigns_nonzero_port() {
        let s = CallbackServer::bind_ephemeral().await.unwrap();
        assert!(s.port() > 0);
    }

    #[tokio::test]
    async fn bind_fixed_port_in_use_maps_to_port_unavailable() {
        let taken = CallbackServer::bind_ephemeral().await.unwrap();
        let err = CallbackServer::bind_fixed(taken.port()).await.unwrap_err();
        assert!(err.downcast_ref::<PortUnavailable>().is_some());
    }

    #[test]
    fn parses_request_target() {
        assert_eq!(
            parse_request_target("GET /oauth2callback?code=abc&state=xyz HTTP/1.1"),
            "/oauth2callback?code=abc&state=xyz"
        );
        assert_eq!(parse_request_target("GET / HTTP/1.1"), "/");
    }

    #[test]
    fn extracts_code_and_state() {
        let (code, state, err) = extract_callback_params("code=abc&state=xyz");
        assert_eq!(code.as_deref(), Some("abc"));
        assert_eq!(state.as_deref(), Some("xyz"));
        assert!(err.is_none());
    }

    #[test]
    fn decodes_percent_encoded_code() {
        let (code, _, _) = extract_callback_params("code=a%2Bb%3Dc&state=s");
        assert_eq!(code.as_deref(), Some("a+b=c"));
    }

    #[test]
    fn propagates_error_param() {
        let (code, _, err) = extract_callback_params("error=access_denied");
        assert!(code.is_none());
        assert_eq!(err.as_deref(), Some("access_denied"));
    }

    #[test]
    fn tolerates_empty_query() {
        let (code, state, err) = extract_callback_params("");
        assert!(code.is_none() && state.is_none() && err.is_none());
    }

    #[test]
    fn find_line_end_crlf_and_lf() {
        assert_eq!(find_line_end(b"GET /x\r\n"), Some(6));
        assert_eq!(find_line_end(b"GET /x\n"), Some(6));
        assert_eq!(find_line_end(b"no newline"), None);
    }

    #[test]
    fn success_html_substitutes_spec_strings() {
        let html = success_html(&SPEC);
        assert!(html.contains("<title>aivo — authorized</title>"));
        assert!(html.contains("<h1>Authorized.</h1>"));
        assert!(!html.contains("__TITLE__") && !html.contains("__HEADING__"));
    }
}
