//! JoyCode authentication — JD ptKey validation.
//!
//! JoyCode (京东 JoyCode) uses a `ptKey` credential for API access.
//! The ptKey is stored as a standard API key in aivo's key store.
//!
//! This is NOT OAuth — the ptKey is a bearer token used directly in
//! the `ptKey` HTTP header on every JoyCode API request.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// JoyCode credential obtained from JD login.
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
