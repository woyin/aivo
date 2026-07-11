//! JoyCode OAuth via JD QR code login.
//!
//! Implements the JD (京东) QR code scan login flow to obtain a `ptKey`
//! credential for JoyCode API access. Based on the protocol from
//! github.com/woyin/joycode2api.

use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// JoyCode credential pair obtained from JD QR login.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JoyCodeCredential {
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

/// Sentinel stored in `ApiKey.base_url` for JoyCode OAuth entries.
pub const JOYCODE_SENTINEL: &str = "joycode-oauth";

const QR_SHOW_URL: &str = "https://qr.m.jd.com/show?appid=133&size=147&t=";
const QR_CHECK_URL: &str = "https://qr.m.jd.com/check?appid=133&token=";
const QR_VALID_URL: &str = "https://passport.jd.com/uc/qrCodeTicketValidation?t=";
const JOYCODE_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    JoyCode/2.7.5 Chrome/133.0.0.0 Electron/35.2.0 Safari/537.36";
const JD_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36";
const QR_SESSION_TTL: Duration = Duration::from_secs(180);

/// Initiate a JD QR login session. Returns (session_id, qr_image_png_base64).
pub async fn qr_init() -> Result<(String, String)> {
    let client = reqwest::Client::builder()
        .cookie_store(true)
        .build()?;

    let url = format!("{}{}", QR_SHOW_URL, chrono::Utc::now().timestamp_millis());
    let resp = client
        .get(&url)
        .header("User-Agent", JD_USER_AGENT)
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .send()
        .await
        .context("request QR code image")?;

    // Extract token before consuming the body
    let token = resp
        .cookies()
        .find(|c| c.name() == "wlfstk_smdl")
        .map(|c| c.value().to_string())
        .context("wlfstk_smdl cookie not found")?;

    let png_data = resp.bytes().await.context("read QR image bytes")?;
    let qr_base64 = BASE64.encode(&png_data);

    let session_id = format!("qr_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0));
    let _ = token; // token managed by cookie store for interactive flow
    Ok((session_id, qr_base64))
}

/// Run the full interactive QR login flow.
/// Blocks until the user scans the QR code and confirms, or the QR expires.
pub async fn interactive_qr_login(progress: &dyn Fn(&str)) -> Result<JoyCodeCredential> {
    let client = reqwest::Client::builder()
        .cookie_store(true)
        .timeout(Duration::from_secs(30))
        .build()?;

    // Step 1: Get QR code
    progress("Fetching QR code...");
    let url = format!("{}{}", QR_SHOW_URL, chrono::Utc::now().timestamp_millis());
    let resp = client
        .get(&url)
        .header("User-Agent", JD_USER_AGENT)
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .send()
        .await
        .context("request QR code")?;

    // Extract token before consuming body
    let _token = resp
        .cookies()
        .find(|c| c.name() == "wlfstk_smdl")
        .map(|c| c.value().to_string())
        .context("wlfstk_smdl cookie not found")?;

    let _png_data = resp.bytes().await.context("read QR image")?;

    let start = Instant::now();
    progress("Scan the QR code with JD app (京东App)...");

    // Step 2: Poll for scan status
    loop {
        if start.elapsed() > QR_SESSION_TTL {
            anyhow::bail!("QR code expired (3 minute timeout). Please try again.");
        }

        tokio::time::sleep(Duration::from_secs(2)).await;

        let check_url = format!(
            "{}{}&callback=jsonpCallback&_={}",
            QR_CHECK_URL,
            urlencoding::encode(&_token),
            chrono::Utc::now().timestamp_millis()
        );

        let check_resp = match client
            .get(&check_url)
            .header("User-Agent", JD_USER_AGENT)
            .header("Referer", "https://passport.jd.com/new/login.aspx")
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                progress(&format!("Poll error: {e}, retrying..."));
                continue;
            }
        };

        let body = check_resp.text().await.unwrap_or_default();

        // Parse JSONP: jsonpCallback({...})
        let json_str = body
            .find('(')
            .and_then(|s| body.rfind(')').map(|e| &body[s + 1..e]))
            .unwrap_or("");

        let check: serde_json::Value = match serde_json::from_str(json_str) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let code = check["code"].as_i64().unwrap_or(0);
        let ticket = check["ticket"].as_str().unwrap_or("");

        match code {
            200 if !ticket.is_empty() => {
                progress("QR confirmed! Validating ticket...");
                return validate_ticket_and_fetch_info(&client, ticket, progress).await;
            }
            201 | 0 => {} // Waiting for scan
            202 => progress("QR scanned! Confirm on your phone..."),
            203 | 204 => anyhow::bail!("QR code expired. Please try again."),
            205 => anyhow::bail!("Login canceled by user."),
            257 => anyhow::bail!("JD server parameter error (code 257)"),
            _ => {}
        }
    }
}

/// Validate the JD login ticket and fetch JoyCode user info.
async fn validate_ticket_and_fetch_info(
    client: &reqwest::Client,
    ticket: &str,
    progress: &dyn Fn(&str),
) -> Result<JoyCodeCredential> {
    let valid_url = format!("{}{}", QR_VALID_URL, urlencoding::encode(ticket));

    let resp = client
        .get(&valid_url)
        .header("User-Agent", JD_USER_AGENT)
        .header("Referer", "https://passport.jd.com/new/login.aspx")
        .send()
        .await
        .context("validate ticket")?;

    let body = resp.text().await.context("read validation response")?;

    let v_result: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
    let return_code = v_result["returnCode"].as_i64().unwrap_or(-1);
    let risk_code = v_result["riskCode"].as_i64().unwrap_or(0);

    if return_code != 0 && risk_code != 0 {
        anyhow::bail!(
            "JD risk control triggered (riskCode={risk_code}). \
             Please complete security verification in your browser."
        );
    }

    // Follow redirect URL if provided
    if let Some(redirect_url) = v_result["url"].as_str() {
        let follow_url = if redirect_url.starts_with("http://") {
            format!("https://{}", &redirect_url[7..])
        } else {
            redirect_url.to_string()
        };
        let _ = client
            .get(&follow_url)
            .header("User-Agent", JD_USER_AGENT)
            .header("Referer", "https://passport.jd.com/new/login.aspx")
            .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
            .send()
            .await;
    }

    progress("Fetching user info from JoyCode...");

    // The cookie jar now has pt_key; userInfo endpoint will resolve userId.
    let user_info_resp = client
        .post("https://joycode-api.jd.com/api/saas/user/v1/userInfo")
        .header("Content-Type", "application/json; charset=UTF-8")
        .header("User-Agent", JOYCODE_USER_AGENT)
        .header("source-type", "joycoder-ide")
        .header("loginType", "N_PIN_PC")
        .json(&serde_json::json!({
            "tenant": "JOYCODE",
            "userId": "",
            "client": "JoyCode",
            "clientVersion": "2.7.5",
            "language": "UNKNOWN"
        }))
        .send()
        .await
        .context("fetch user info from JoyCode")?;

    let user_info: serde_json::Value =
        user_info_resp.json().await.context("parse user info response")?;

    let code = user_info["code"].as_i64().unwrap_or(-1);
    if code != 0 {
        let msg = user_info["msg"].as_str().unwrap_or("unknown error");
        anyhow::bail!("JoyCode user info failed (code={code}): {msg}");
    }

    let user_id = user_info["data"]["userId"].as_str().unwrap_or("").to_string();
    let real_name = user_info["data"]["realName"].as_str().unwrap_or("").to_string();
    let color_base_url = user_info["data"]["colorBaseUrl"].as_str().unwrap_or("").to_string();
    let master_base_url = user_info["data"]["masterBaseUrl"].as_str().unwrap_or("").to_string();
    let tenant = user_info["data"]["tenant"].as_str().unwrap_or("JOYCODE").to_string();
    let login_type = user_info["data"]["loginType"].as_str().unwrap_or("N_PIN_PC").to_string();
    let org_full_name = user_info["data"]["orgFullName"].as_str().unwrap_or("").to_string();
    let pt_key = user_info["data"]["ptKey"].as_str().unwrap_or("").to_string();

    if user_id.is_empty() {
        anyhow::bail!("Could not get userId from JoyCode. Your session may have expired.");
    }

    if pt_key.is_empty() {
        anyhow::bail!(
            "Could not extract ptKey from the login session. \
             Please try logging in again or provide your ptKey manually."
        );
    }

    Ok(JoyCodeCredential {
        pt_key,
        pt_pin: String::new(),
        user_id,
        real_name,
        color_base_url,
        master_base_url,
        tenant,
        login_type,
        org_full_name,
    })
}

/// Validate a ptKey by calling the JoyCode userInfo endpoint.
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
            "clientVersion": "2.7.5",
            "language": "UNKNOWN"
        }))
        .send()
        .await
        .context("validate ptKey against JoyCode")?;

    let result: serde_json::Value = resp.json().await.context("parse validation response")?;
    let code = result["code"].as_i64().unwrap_or(-1);
    if code != 0 {
        let msg = result["msg"].as_str().unwrap_or("unknown error");
        anyhow::bail!("ptKey validation failed (code={code}): {msg}");
    }

    let user_id = result["data"]["userId"].as_str().unwrap_or("").to_string();
    let real_name = result["data"]["realName"].as_str().unwrap_or("").to_string();
    let color_base_url = result["data"]["colorBaseUrl"].as_str().unwrap_or("").to_string();
    let master_base_url = result["data"]["masterBaseUrl"].as_str().unwrap_or("").to_string();
    let tenant = result["data"]["tenant"].as_str().unwrap_or("JOYCODE").to_string();
    let login_type = result["data"]["loginType"].as_str().unwrap_or("N_PIN_PC").to_string();
    let org_full_name = result["data"]["orgFullName"].as_str().unwrap_or("").to_string();
    let refreshed_pt_key = result["data"]["ptKey"].as_str().unwrap_or(pt_key).to_string();

    Ok(JoyCodeCredential {
        pt_key: refreshed_pt_key,
        pt_pin: String::new(),
        user_id,
        real_name,
        color_base_url,
        master_base_url,
        tenant,
        login_type,
        org_full_name,
    })
}

/// Check if a base_url indicates a JoyCode key.
pub fn is_joycode_key(base_url: &str) -> bool {
    base_url == JOYCODE_SENTINEL
        || base_url.contains("joycode-api.jd.com")
        || base_url.contains("api-ai.jd.com")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_joycode_key_detects_sentinel() {
        assert!(is_joycode_key(JOYCODE_SENTINEL));
    }

    #[test]
    fn is_joycode_key_detects_joycode_api() {
        assert!(is_joycode_key("https://joycode-api.jd.com"));
        assert!(is_joycode_key("https://api-ai.jd.com"));
    }

    #[test]
    fn is_joycode_key_rejects_other_urls() {
        assert!(!is_joycode_key("https://api.openai.com"));
        assert!(!is_joycode_key("https://openrouter.ai/api/v1"));
    }
}
