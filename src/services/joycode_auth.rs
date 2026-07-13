//! JoyCode authentication — JD QR login to obtain ptKey.
//!
//! JoyCode (京东 JoyCode) uses a `ptKey` credential for API access.
//! The ptKey is obtained via JD QR code scan login, then stored as a
//! standard API key in aivo's key store.
//!
//! This is NOT OAuth — the ptKey is a bearer token used directly in
//! the `ptKey` HTTP header on every JoyCode API request.

use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// JoyCode credential obtained from JD QR login.
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
        Ok(serde_json::to_string(self).context("serialize JoyCodeCredential")?)
    }
}

const QR_SHOW_URL: &str = "https://qr.m.jd.com/show?appid=133&size=147&t=";
const QR_CHECK_URL: &str = "https://qr.m.jd.com/check?appid=133&token=";
const QR_VALID_URL: &str = "https://passport.jd.com/uc/qrCodeTicketValidation?t=";
const JOYCODE_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) \
    JoyCode/2.7.5 Chrome/133.0.0.0 Electron/35.2.0 Safari/537.36";
const JD_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) \
    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36";
const QR_SESSION_TTL: Duration = Duration::from_secs(180);

/// Run the JD QR code login flow to obtain a JoyCode ptKey.
///
/// Displays QR code, polls for scan confirmation, validates ticket,
/// and resolves the full credential (ptKey + userId + tenant info).
pub async fn qr_login(progress: &dyn Fn(&str)) -> Result<JoyCodeCredential> {
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

    let token = resp
        .cookies()
        .find(|c| c.name() == "wlfstk_smdl")
        .map(|c| c.value().to_string())
        .context("wlfstk_smdl cookie not found")?;

    let _png_data = resp.bytes().await.context("read QR image")?;
    progress("Scan the QR code with JD app (京东App)...");

    // Step 2: Poll for scan status
    let start = Instant::now();
    loop {
        if start.elapsed() > QR_SESSION_TTL {
            anyhow::bail!("QR code expired (3 min). Try again.");
        }
        tokio::time::sleep(Duration::from_secs(2)).await;

        let check_url = format!(
            "{}{}&callback=jsonpCallback&_={}",
            QR_CHECK_URL,
            urlencoding::encode(&token),
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
                progress(&format!("Poll error: {e}"));
                continue;
            }
        };

        let body = check_resp.text().await.unwrap_or_default();
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
                progress("QR confirmed! Validating...");
                return validate_and_resolve(&client, ticket, progress).await;
            }
            201 | 0 => {}
            202 => progress("Scanned! Confirm on phone..."),
            203 | 204 => anyhow::bail!("QR expired."),
            205 => anyhow::bail!("Login canceled."),
            257 => anyhow::bail!("JD server error (code 257)"),
            _ => {}
        }
    }
}

/// Validate ticket and resolve full JoyCode credential.
async fn validate_and_resolve(
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
    let v: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();

    if v["riskCode"].as_i64().unwrap_or(0) != 0 {
        let rc = v["riskCode"].as_i64().unwrap_or(0);
        anyhow::bail!("JD risk control (riskCode={rc}). Complete security verification.");
    }

    // Follow redirect URL if provided
    if let Some(redirect_url) = v["url"].as_str() {
        let follow = if redirect_url.starts_with("http://") {
            format!("https://{}", &redirect_url[7..])
        } else {
            redirect_url.to_string()
        };
        let _ = client
            .get(&follow)
            .header("User-Agent", JD_USER_AGENT)
            .header("Referer", "https://passport.jd.com/new/login.aspx")
            .header(
                "Accept",
                "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            )
            .send()
            .await;
    }

    progress("Fetching user info...");
    let resp = client
        .post("https://joycode-api.jd.com/api/saas/user/v1/userInfo")
        .header("Content-Type", "application/json; charset=UTF-8")
        .header("User-Agent", JOYCODE_USER_AGENT)
        .header("source-type", "joycoder-ide")
        .header("loginType", "N_PIN_PC")
        .json(&serde_json::json!({
            "tenant": "JOYCODE", "userId": "", "client": "JoyCode",
            "clientVersion": "2.7.5", "language": "UNKNOWN"
        }))
        .send()
        .await
        .context("fetch user info")?;

    let info: serde_json::Value = resp.json().await.context("parse user info")?;
    if info["code"].as_i64().unwrap_or(-1) != 0 {
        let msg = info["msg"].as_str().unwrap_or("unknown");
        anyhow::bail!("JoyCode user info failed: {msg}");
    }

    let d = &info["data"];
    let user_id = d["userId"].as_str().unwrap_or("");
    let pt_key = d["ptKey"].as_str().unwrap_or("");

    if user_id.is_empty() {
        anyhow::bail!("userId empty — session may have expired.");
    }
    if pt_key.is_empty() {
        anyhow::bail!("ptKey empty — try again or provide manually.");
    }

    Ok(JoyCodeCredential {
        pt_key: pt_key.to_string(),
        pt_pin: String::new(),
        user_id: user_id.to_string(),
        real_name: d["realName"].as_str().unwrap_or("").to_string(),
        color_base_url: d["colorBaseUrl"].as_str().unwrap_or("").to_string(),
        master_base_url: d["masterBaseUrl"].as_str().unwrap_or("").to_string(),
        tenant: d["tenant"].as_str().unwrap_or("JOYCODE").to_string(),
        login_type: d["loginType"].as_str().unwrap_or("N_PIN_PC").to_string(),
        org_full_name: d["orgFullName"].as_str().unwrap_or("").to_string(),
    })
}

/// Validate a ptKey and resolve full credential info.
/// Used when user provides ptKey manually via `aivo keys add`.
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
