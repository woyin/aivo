//! JoyCode authentication — browser-based login with automatic key retrieval.
//!
//! JoyCode (京东 JoyCode) uses a `ptKey` credential for API access.
//! The login flow opens the JoyCode login page in the browser; after the user
//! authenticates, JoyCode redirects back to a local callback server with the
//! credential, which is then validated and stored automatically.
//!
//! This is NOT OAuth — the ptKey is a bearer token used directly in
//! the `ptKey` HTTP header on every JoyCode API request.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::services::browser_open;
use crate::services::percent_codec;

/// JoyCode credential obtained from browser login or manual ptKey entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JoyCodeCredential {
    /// JD authentication token — used as the `ptKey` HTTP header.
    pub pt_key: String,
    pub pt_pin: String,
    pub user_id: String,
    pub real_name: String,
    pub color_base_url: String,
    pub master_base_url: String,
    pub tenant: String,
    pub login_type: String,
    pub org_full_name: String,
}

impl JoyCodeCredential {
    /// Serialize to JSON for storage in `ApiKey.key`.
    pub fn to_key_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize JoyCodeCredential")
    }
}

const JOYCODE_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    JoyCode/2.7.5 Chrome/133.0.0.0 Electron/35.2.0 Safari/537.36";

/// JoyCode SSO login page URL.
const JOYCODE_LOGIN_URL: &str = "https://joycode.jd.com/login";

/// Local callback path for the OAuth redirect.
const CALLBACK_PATH: &str = "/joycode/callback";

/// HTML shown in the browser after a successful login callback.
const SUCCESS_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>aivo — JoyCode login</title>
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
    <h1>Logged in to JoyCode.</h1>
    <p>You can close this tab and return to your terminal.</p>
  </div>
</body>
</html>
"#;

/// Browser-based login flow for JoyCode.
///
/// 1. Starts a local HTTP callback server on an ephemeral port.
/// 2. Opens the JoyCode login page in the browser with the callback URL.
/// 3. After the user authenticates, JoyCode redirects to the local server
///    with the ptKey (or an authorization code).
/// 4. Validates the credential with the JoyCode API.
/// 5. Returns the full credential for storage.
pub async fn browser_login(progress: &dyn Fn(&str)) -> Result<JoyCodeCredential> {
    // 1. Bind local callback server on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to start local callback server")?;
    let port = listener.local_addr()?.port();
    let callback_url = format!("http://127.0.0.1:{port}{CALLBACK_PATH}");

    // 2. Build the login URL with redirect_uri pointing to our callback.
    let login_url = format!(
        "{}?redirect_uri={}&from=aivo",
        JOYCODE_LOGIN_URL,
        percent_codec::encode(&callback_url)
    );

    progress("Opening browser for JoyCode login...");

    // 3. Try to open the browser; fall back to printing the URL.
    if browser_open::open_url(&login_url).is_err() {
        progress("Could not open browser automatically.");
    }
    eprintln!();
    eprintln!("  If the browser didn't open, visit:");
    eprintln!("  {}", crate::style::blue(&login_url));
    eprintln!();

    progress("Waiting for login to complete (5 min timeout)...");

    // 4. Wait for the callback redirect from JoyCode.
    let accept_future = accept_callback(listener, &callback_url);
    let pt_key = match tokio::time::timeout(Duration::from_secs(300), accept_future).await {
        Ok(Ok(key)) => key,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(anyhow!("JoyCode login timed out after 5 minutes")),
    };

    // 5. Validate the ptKey with JoyCode API.
    progress("Login successful! Validating credential...");
    validate_pt_key(&pt_key).await
}

/// Accept one valid callback hit on the local listener and extract the ptKey.
///
/// The JoyCode login page redirects to our callback URL with the token in
/// the query string. We accept several parameter names for robustness:
/// `ptKey`, `pt_key`, `token`, `code` — whatever JoyCode sends back.
async fn accept_callback(listener: TcpListener, _expected_callback: &str) -> Result<String> {
    loop {
        let (mut stream, _) = listener
            .accept()
            .await
            .context("accept callback connection")?;

        let request_line = match read_request_line(&mut stream).await {
            Ok(line) => line,
            Err(_) => {
                let _ = stream.shutdown().await;
                continue;
            }
        };

        let path_and_query = parse_request_target(&request_line);

        // Only respond to our callback path; 404 everything else (favicon, etc.)
        if !path_and_query.starts_with(CALLBACK_PATH) {
            respond(&mut stream, 404, "text/plain; charset=utf-8", b"not found").await;
            continue;
        }

        let query = path_and_query.split_once('?').map(|(_, q)| q).unwrap_or("");

        // Extract the token from the callback query parameters.
        let (pt_key, error) = extract_callback_token(query);

        if let Some(err) = error {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                format!("Login error: {err}").as_bytes(),
            )
            .await;
            return Err(anyhow!("JoyCode login returned error: {err}"));
        }

        let pt_key = pt_key.ok_or_else(|| anyhow!("JoyCode callback missing token"))?;

        // Send success page to the browser.
        respond(
            &mut stream,
            200,
            "text/html; charset=utf-8",
            SUCCESS_HTML.as_bytes(),
        )
        .await;

        return Ok(pt_key);
    }
}

/// Extract the token from the callback query string.
/// Tries multiple parameter names for robustness: ptKey, pt_key, token, code.
fn extract_callback_token(query: &str) -> (Option<String>, Option<String>) {
    let mut token = None;
    let mut error = None;

    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some(kv) => kv,
            None => (pair, ""),
        };
        let decoded = percent_codec::decode(v);
        match k {
            "ptKey" | "pt_key" | "token" | "code" => {
                if token.is_none() && !decoded.is_empty() {
                    token = Some(decoded);
                }
            }
            "error" | "error_description" if error.is_none() => {
                error = Some(decoded);
            }
            _ => {}
        }
    }

    (token, error)
}

/// Reads up to the first line of an HTTP request. Bounded to 8 KiB.
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
    let mut parts = request_line.split_whitespace();
    let _method = parts.next();
    parts.next().unwrap_or("")
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

/// Validate a ptKey and resolve full credential info.
/// Used when user provides ptKey manually via `aivo keys add --key`.
pub async fn validate_pt_key(pt_key: &str) -> Result<JoyCodeCredential> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://joycode-api.jd.com/api/saas/user/v1/userInfo")
        .header("Content-Type", "application/json; charset=UTF-8")
        .header("source-type", "joycoder-ide")
        .header("ptKey", pt_key)
        .header("loginType", "N_PIN_PC")
        .header("User-Agent", JOYCODE_USER_AGENT)
        .json(&serde_json::json!({
            "tenant": "JOYCODE", "userId": "", "client": "JoyCode",
            "clientVersion": "2.7.5", "language": "UNKNOWN"
        }))
        .send()
        .await
        .context("validate ptKey")?;

    let result: serde_json::Value = resp.json().await.context("parse response")?;
    if result["code"].as_i64().unwrap_or(-1) != 0 {
        let msg = result["msg"].as_str().unwrap_or("unknown");
        anyhow::bail!("ptKey validation failed: {msg}");
    }

    let d = &result["data"];
    Ok(JoyCodeCredential {
        pt_key: d["ptKey"].as_str().unwrap_or(pt_key).to_string(),
        pt_pin: String::new(),
        user_id: d["userId"].as_str().unwrap_or("").to_string(),
        real_name: d["realName"].as_str().unwrap_or("").to_string(),
        color_base_url: d["colorBaseUrl"].as_str().unwrap_or("").to_string(),
        master_base_url: d["masterBaseUrl"].as_str().unwrap_or("").to_string(),
        tenant: d["tenant"].as_str().unwrap_or("JOYCODE").to_string(),
        login_type: d["loginType"].as_str().unwrap_or("N_PIN_PC").to_string(),
        org_full_name: d["orgFullName"].as_str().unwrap_or("").to_string(),
    })
}

/// Check if a base_url is a JoyCode endpoint.
pub fn is_joycode_key(base_url: &str) -> bool {
    base_url.contains("joycode-api.jd.com")
        || base_url.contains("api-ai.jd.com")
        || base_url == "joycode"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_callback_token_pt_key() {
        let (token, err) = extract_callback_token("ptKey=abc123&state=xyz");
        assert_eq!(token.as_deref(), Some("abc123"));
        assert!(err.is_none());
    }

    #[test]
    fn extract_callback_token_code() {
        let (token, err) = extract_callback_token("code=mycode456");
        assert_eq!(token.as_deref(), Some("mycode456"));
        assert!(err.is_none());
    }

    #[test]
    fn extract_callback_token_error() {
        let (token, err) = extract_callback_token("error=access_denied");
        assert!(token.is_none());
        assert_eq!(err.as_deref(), Some("access_denied"));
    }

    #[test]
    fn extract_callback_token_empty() {
        let (token, err) = extract_callback_token("");
        assert!(token.is_none() && err.is_none());
    }

    #[test]
    fn is_joycode_key_detects_joycode_api() {
        assert!(is_joycode_key("https://joycode-api.jd.com"));
        assert!(is_joycode_key("https://api-ai.jd.com"));
        assert!(is_joycode_key("joycode"));
    }

    #[test]
    fn is_joycode_key_rejects_other_urls() {
        assert!(!is_joycode_key("https://api.openai.com"));
        assert!(!is_joycode_key("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn find_line_end_crlf_and_lf() {
        assert_eq!(find_line_end(b"GET /x\r\n"), Some(6));
        assert_eq!(find_line_end(b"GET /x\n"), Some(6));
        assert_eq!(find_line_end(b"no newline"), None);
    }
}
