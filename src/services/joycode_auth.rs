//! JoyCode authentication — browser-based, auto, and QR login.
//!
//! JoyCode (京东 JoyCode) uses a `ptKey` credential for API access.
//! Three login methods are supported:
//! 1. Browser Login — JoyCode redirects to a local callback with ptKey
//! 2. Auto Login — reads ptKey from local JoyCode IDE state database
//! 3. QR Login — scans JD APP QR code to obtain ptKey
//!
//! The ptKey is a bearer token used directly in the `ptKey` HTTP header
//! on every JoyCode API request.

use anyhow::{Context, Result, anyhow};
use base64::Engine as _;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::timeout;

use crate::services::browser_open;
use crate::services::percent_codec;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const JOYCODE_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    JoyCode/2.7.5 Chrome/133.0.0.0 Electron/35.2.0 Safari/537.36";

const JOYCODE_CLIENT_VERSION: &str = "2.7.5";

/// JoyCode SSO login page URL.
const JOYCODE_LOGIN_URL: &str = "https://joycode.jd.com/login";

/// Local callback path for the OAuth redirect.
const CALLBACK_PATH: &str = "/joycode/callback";

/// Maximum size for callback request line.
const MAX_REQUEST_LINE: usize = 8192;

/// Default callback timeout.
const CALLBACK_TIMEOUT_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// JoyCodeCredential
// ---------------------------------------------------------------------------

/// JoyCode credential obtained from browser login, auto login, or manual ptKey entry.
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

impl Default for JoyCodeCredential {
    fn default() -> Self {
        Self {
            pt_key: String::new(),
            pt_pin: String::new(),
            user_id: String::new(),
            real_name: String::new(),
            color_base_url: String::new(),
            master_base_url: String::new(),
            tenant: "JOYCODE".to_string(),
            login_type: "N_PIN_PC".to_string(),
            org_full_name: String::new(),
        }
    }
}

impl JoyCodeCredential {
    /// Serialize to JSON for storage in `ApiKey.key`.
    pub fn to_key_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize JoyCodeCredential")
    }
}

// ---------------------------------------------------------------------------
// Browser Login
// ---------------------------------------------------------------------------

/// Browser-based login flow for JoyCode.
///
/// 1. Starts a local HTTP callback server on an ephemeral port.
/// 2. Generates a random authKey and builds the JoyCode login URL with authPort + authKey.
/// 3. Opens the JoyCode login page in the browser.
/// 4. After user authenticates, JoyCode redirects to the local server with ptKey.
/// 5. Validates the credential with the JoyCode API.
/// 6. Returns the full credential for storage.
pub async fn browser_login(progress: &dyn Fn(&str)) -> Result<JoyCodeCredential> {
    // 1. Bind local callback server on an ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("failed to start local callback server")?;
    let port = listener.local_addr()?.port();

    // 2. Generate a random authKey (hex, 32 chars).
    let auth_key: String = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();

    // Build the login URL with authPort and authKey.
    let login_url = format!(
        "{}?ideAppName=JoyCode&fromIde=ide&redirect=0&authPort={}&authKey={}",
        JOYCODE_LOGIN_URL,
        percent_codec::encode(&port.to_string()),
        percent_codec::encode(&auth_key),
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
    let pt_key = match timeout(
        Duration::from_secs(CALLBACK_TIMEOUT_SECS),
        accept_browser_callback(listener, &auth_key),
    )
    .await
    {
        Ok(Ok(key)) => key,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err(anyhow!("JoyCode login timed out after 5 minutes")),
    };

    // 5. Validate the ptKey with JoyCode API.
    progress("Login successful! Validating credential...");
    validate_pt_key(&pt_key).await
}

/// Accept the callback from JoyCode browser login.
/// Validates authKey matches, then extracts ptKey from query params.
async fn accept_browser_callback(listener: TcpListener, expected_auth_key: &str) -> Result<String> {
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

        // Parse query parameters into a map.
        let params = parse_query_params(query);

        // Validate authKey.
        let auth_key = params.get("authKey").map(String::as_str).unwrap_or("");
        if auth_key != expected_auth_key {
            respond(
                &mut stream,
                403,
                "text/plain; charset=utf-8",
                b"Invalid authKey",
            )
            .await;
            continue;
        }

        // Check for errors.
        if let Some(err) = params.get("error") {
            respond(
                &mut stream,
                400,
                "text/plain; charset=utf-8",
                format!("Login error: {err}").as_bytes(),
            )
            .await;
            return Err(anyhow!("JoyCode login returned error: {err}"));
        }

        // Extract ptKey.
        let pt_key = params
            .get("pt_key")
            .or_else(|| params.get("ptKey"))
            .cloned()
            .ok_or_else(|| anyhow!("JoyCode callback missing pt_key"))?;

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

// ---------------------------------------------------------------------------
// Auto Login — read from local JoyCode IDE state database
// ---------------------------------------------------------------------------

/// Reads ptKey from the local JoyCode IDE state database (macOS).
/// Supports JOYCODE_STATE_DB env override and Docker container path.
pub fn auto_login_from_system() -> Result<JoyCodeCredential> {
    // Check environment override first.
    if let Ok(db_path) = std::env::var("JOYCODE_STATE_DB") {
        return load_from_state_db(&db_path);
    }

    // Check Docker/container path.
    let container_path = "/root/.joycode-ide/state.vscdb";
    if std::path::Path::new(container_path).exists() {
        return load_from_state_db(container_path);
    }

    // macOS default path.
    #[cfg(target_os = "macos")]
    {
        let home = crate::services::system_env::home_dir()
            .ok_or_else(|| anyhow!("cannot determine home directory"))?;
        let db_path =
            home.join("Library/Application Support/JoyCode/User/globalStorage/state.vscdb");
        if db_path.exists() {
            return load_from_state_db(&db_path);
        }
    }

    // Linux default path (if exists).
    #[cfg(not(target_os = "macos"))]
    {
        let home = crate::services::system_env::home_dir()
            .ok_or_else(|| anyhow!("cannot determine home directory"))?;
        let db_path = home.join(".joycode-ide/state.vscdb");
        if db_path.exists() {
            return load_from_state_db(&db_path);
        }
    }

    Err(anyhow!(
        "JoyCode IDE state database not found.\n  \
         Please install and log in to JoyCode IDE first, or use browser/QR login."
    ))
}

/// Load credentials from the JoyCode state SQLite database.
fn load_from_state_db(db_path: impl AsRef<std::path::Path>) -> Result<JoyCodeCredential> {
    use rusqlite::Connection;

    let db_path = db_path.as_ref();
    if !db_path.exists() {
        return Err(anyhow!(
            "JoyCode state database not found at {}",
            db_path.display()
        ));
    }

    let conn = Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("cannot open JoyCode database at {}", db_path.display()))?;

    let mut stmt = conn
        .prepare("SELECT value FROM ItemTable WHERE key = 'JoyCoder.IDE'")
        .context("prepare query")?;

    let value: String = stmt
        .query_row([], |row| row.get(0))
        .context("login info not found in database — please log in to JoyCode IDE first")?;

    parse_state_json(&value)
}

#[derive(Debug, serde::Deserialize)]
struct JoyCoderUserData {
    #[serde(rename = "ptKey")]
    pt_key: String,
    #[serde(rename = "userId")]
    user_id: String,
    #[serde(rename = "colorBaseUrl")]
    color_base_url: Option<String>,
    #[serde(rename = "masterBaseUrl")]
    master_base_url: Option<String>,
    tenant: Option<String>,
    #[serde(rename = "loginType")]
    login_type: Option<String>,
    #[serde(rename = "orgFullName")]
    org_full_name: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct StateData {
    #[serde(rename = "joyCoderUser")]
    joy_coder_user: JoyCoderUserData,
}

fn parse_state_json(value: &str) -> Result<JoyCodeCredential> {
    let data: StateData =
        serde_json::from_str(value).context("cannot parse login data from database")?;

    if data.joy_coder_user.pt_key.is_empty() {
        return Err(anyhow!(
            "ptKey is empty in stored credentials — please re-login to JoyCode IDE"
        ));
    }
    if data.joy_coder_user.user_id.is_empty() {
        return Err(anyhow!(
            "userId is empty in stored credentials — please re-login to JoyCode IDE"
        ));
    }

    Ok(JoyCodeCredential {
        pt_key: data.joy_coder_user.pt_key,
        pt_pin: String::new(),
        user_id: data.joy_coder_user.user_id,
        real_name: String::new(),
        color_base_url: data.joy_coder_user.color_base_url.unwrap_or_default(),
        master_base_url: data.joy_coder_user.master_base_url.unwrap_or_default(),
        tenant: data
            .joy_coder_user
            .tenant
            .unwrap_or_else(|| "JOYCODE".to_string()),
        login_type: data
            .joy_coder_user
            .login_type
            .unwrap_or_else(|| "N_PIN_PC".to_string()),
        org_full_name: data.joy_coder_user.org_full_name.unwrap_or_default(),
    })
}

// ---------------------------------------------------------------------------
// QR Login — JD APP scan
// ---------------------------------------------------------------------------

/// QR session state with manual cookie management.
struct QRSession {
    token: String,
    cookies: String,
    created_at: Instant,
}

/// Shared QR session storage (in-memory, auto-cleaned).
static QR_SESSIONS: Mutex<Option<HashMap<String, QRSession>>> = Mutex::new(None);

const QR_SESSION_TTL: Duration = Duration::from_secs(180);
const QR_CLEANUP_INTERVAL: Duration = Duration::from_secs(60);

/// Initialize QR session storage and start cleanup janitor.
fn init_qr_sessions() {
    let mut storage = QR_SESSIONS.lock().unwrap();
    if storage.is_none() {
        *storage = Some(HashMap::new());
        drop(storage);
        // Start cleanup janitor.
        tokio::spawn(async {
            let mut interval = tokio::time::interval(QR_CLEANUP_INTERVAL);
            loop {
                interval.tick().await;
                cleanup_expired_qr_sessions();
            }
        });
    }
}

fn cleanup_expired_qr_sessions() {
    let mut storage = QR_SESSIONS.lock().unwrap();
    if let Some(sessions) = storage.as_mut() {
        let now = Instant::now();
        sessions.retain(|_, session| now.duration_since(session.created_at) < QR_SESSION_TTL);
    }
}

/// Initialize a QR login session. Returns (session_id, qr_image_base64).
pub async fn qr_init() -> Result<(String, String)> {
    init_qr_sessions();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("build HTTP client")?;

    // Fetch QR image.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let qr_show_url = format!("https://qr.m.jd.com/show?appid=133&size=147&t={ts}");

    let resp = client
        .get(&qr_show_url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .send()
        .await
        .context("request QR code")?;

    // Extract Set-Cookie headers to find wlfstk_smdl.
    let mut cookies = Vec::new();
    let mut token = String::new();
    for (key, value) in resp.headers().iter() {
        if key == "set-cookie" {
            let cookie_str = value.to_str().unwrap_or("");
            cookies.push(cookie_str.to_string());
            if let Some(start) = cookie_str.find("wlfstk_smdl=") {
                let after = &cookie_str[start + "wlfstk_smdl=".len()..];
                let end = after.find(';').unwrap_or(after.len());
                token = after[..end].to_string();
            }
        }
    }

    let png_data = resp.bytes().await.context("read QR image")?;
    let qr_image = base64::engine::general_purpose::STANDARD.encode(&png_data);

    if token.is_empty() {
        return Err(anyhow!("wlfstk_smdl cookie not found in QR response"));
    }

    let cookie_header = cookies.join("; ");
    let session_id = format!("qr_{}", ts);

    let mut storage = QR_SESSIONS.lock().unwrap();
    if let Some(sessions) = storage.as_mut() {
        sessions.insert(
            session_id.clone(),
            QRSession {
                token,
                cookies: cookie_header,
                created_at: Instant::now(),
            },
        );
    }

    Ok((session_id, qr_image))
}

/// Poll QR login status.
/// Returns (status, result_on_success).
/// Status: "waiting" | "scanned" | "confirmed" | "expired" | "error"
pub async fn qr_poll_status(session_id: &str) -> Result<(String, Option<JoyCodeCredential>)> {
    let (token, cookies) = {
        let mut storage = QR_SESSIONS.lock().unwrap();
        let (token, cookies) = match storage.as_ref().and_then(|s| s.get(session_id)) {
            Some(s) => {
                if Instant::now().duration_since(s.created_at) > QR_SESSION_TTL {
                    if let Some(sessions) = storage.as_mut() {
                        sessions.remove(session_id);
                    }
                    return Ok(("expired".to_string(), None));
                }
                (s.token.clone(), s.cookies.clone())
            }
            None => return Ok(("expired".to_string(), None)),
        };
        // Drop lock before network I/O.
        drop(storage);
        (token, cookies)
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("build HTTP client")?;

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let check_url = format!(
        "https://qr.m.jd.com/check?appid=133&token={}&callback=jsonpCallback&_={}",
        percent_codec::encode(&token),
        ts
    );

    let resp = client
        .get(&check_url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .header("Cookie", &cookies)
        .send()
        .await
        .context("QR check request")?;

    let body = resp.text().await.context("read QR check response")?;

    // Parse JSONP response: jsonpCallback({"code":...})
    let json_start = body.find('(').map(|i| i + 1).unwrap_or(0);
    let json_end = body.rfind(')').unwrap_or(body.len());
    let json_str = &body[json_start..json_end];

    let check: serde_json::Value = serde_json::from_str(json_str).unwrap_or_default();
    let code = check["code"].as_i64().unwrap_or(-1);
    let ticket = check["ticket"].as_str().unwrap_or("");

    match code {
        200 => {
            if ticket.is_empty() {
                return Ok(("error".to_string(), None));
            }
            // Validate ticket and get pt_key.
            match validate_qr_ticket(&cookies, ticket).await {
                Ok(creds) => {
                    // Clean up session.
                    let mut storage = QR_SESSIONS.lock().unwrap();
                    if let Some(sessions) = storage.as_mut() {
                        sessions.remove(session_id);
                    }
                    Ok(("confirmed".to_string(), Some(creds)))
                }
                Err(e) => {
                    eprintln!("QR ticket validation failed: {}", e);
                    Ok(("error".to_string(), None))
                }
            }
        }
        201 => Ok(("waiting".to_string(), None)),
        202 => Ok(("scanned".to_string(), None)),
        203..=205 => {
            let mut storage = QR_SESSIONS.lock().unwrap();
            if let Some(sessions) = storage.as_mut() {
                sessions.remove(session_id);
            }
            Ok(("expired".to_string(), None))
        }
        257 => Ok(("error".to_string(), None)),
        _ => Ok(("error".to_string(), None)),
    }
}

/// Validate QR ticket and extract pt_key from cookies.
async fn validate_qr_ticket(cookies: &str, ticket: &str) -> Result<JoyCodeCredential> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .context("build HTTP client")?;

    let valid_url = format!(
        "https://passport.jd.com/uc/qrCodeTicketValidation?t={}",
        percent_codec::encode(ticket)
    );

    let resp = client
        .get(&valid_url)
        .header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36")
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .header("Cookie", cookies)
        .send()
        .await
        .context("validate ticket")?;

    // Collect pt_key from response cookies.
    let mut pt_key = String::new();
    for (key, value) in resp.headers().iter() {
        if key == "set-cookie" {
            let cookie_str = value.to_str().unwrap_or("");
            if let Some(start) = cookie_str.find("pt_key=") {
                let after = &cookie_str[start + "pt_key=".len()..];
                let end = after.find(';').unwrap_or(after.len());
                pt_key = after[..end].to_string();
                break;
            }
        }
    }

    if pt_key.is_empty() {
        // Also try the original request cookies as fallback.
        for cookie in cookies.split(';') {
            let cookie = cookie.trim();
            if let Some(val) = cookie.strip_prefix("pt_key=") {
                pt_key = val.to_string();
                break;
            }
        }
    }

    if pt_key.is_empty() {
        return Err(anyhow!("pt_key not found after QR ticket validation"));
    }

    // Validate and get user info.
    validate_pt_key(&pt_key).await
}

// ---------------------------------------------------------------------------
// Validate ptKey
// ---------------------------------------------------------------------------

/// Validate a ptKey and resolve full credential info.
/// Used by browser login, auto login, QR login, and manual ptKey entry.
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
            "tenant": "JOYCODE",
            "userId": "",
            "client": "JoyCode",
            "clientVersion": JOYCODE_CLIENT_VERSION,
            "language": "UNKNOWN"
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

// ---------------------------------------------------------------------------
// Check if a base_url is a JoyCode endpoint
// ---------------------------------------------------------------------------

/// Check if a base_url is a JoyCode endpoint.
pub fn is_joycode_key(base_url: &str) -> bool {
    base_url.contains("joycode-api.jd.com")
        || base_url.contains("api-ai.jd.com")
        || base_url == "joycode"
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Reads up to the first line of an HTTP request. Bounded to 8 KiB.
async fn read_request_line(stream: &mut tokio::net::TcpStream) -> Result<String> {
    let mut buf = [0u8; MAX_REQUEST_LINE];
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
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'\n' {
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

/// Parse query parameters into a HashMap.
fn parse_query_params(query: &str) -> HashMap<String, String> {
    let mut params = HashMap::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let (k, v) = match pair.split_once('=') {
            Some(kv) => kv,
            None => (pair, ""),
        };
        let decoded = percent_codec::decode(v);
        params.insert(k.to_string(), decoded);
    }
    params
}

async fn respond(stream: &mut tokio::net::TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_query_params_basic() {
        let params = parse_query_params("pt_key=abc&authKey=xyz");
        assert_eq!(params.get("pt_key"), Some(&"abc".to_string()));
        assert_eq!(params.get("authKey"), Some(&"xyz".to_string()));
    }

    #[test]
    fn parse_query_params_empty() {
        let params = parse_query_params("");
        assert!(params.is_empty());
    }

    #[test]
    fn find_line_end_crlf_and_lf() {
        assert_eq!(find_line_end(b"GET /x\r\n"), Some(6));
        assert_eq!(find_line_end(b"GET /x\n"), Some(6));
        assert_eq!(find_line_end(b"no newline"), None);
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
}
