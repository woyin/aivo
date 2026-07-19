//! GitHub Copilot authentication via OAuth device flow and token management.
//!
//! Implements the same flow as OpenCode/VS Code:
//! 1. OAuth device flow with VS Code Copilot client ID
//! 2. Exchange GitHub token for short-lived Copilot token
//! 3. Auto-refresh expired Copilot tokens

use anyhow::{Context, Result};
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::constants::CONTENT_TYPE_JSON;
use crate::services::http_debug::LoggedSend;

/// VS Code Copilot OAuth client ID (same as OpenCode uses)
const COPILOT_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";

/// GitHub endpoints
const DEVICE_CODE_URL: &str = "https://github.com/login/device/code";
const ACCESS_TOKEN_URL: &str = "https://github.com/login/oauth/access_token";
const COPILOT_TOKEN_URL: &str = "https://api.github.com/copilot_internal/v2/token";

/// HTTP headers required by the Copilot API to identify the client as VS Code.
pub const COPILOT_EDITOR_VERSION: &str = "vscode/1.95.0";
pub const COPILOT_INTEGRATION_ID: &str = "vscode-chat";
pub const COPILOT_OPENAI_INTENT: &str = "conversation-panel";
pub const COPILOT_INITIATOR_HEADER: &str = "X-Initiator";

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    #[serde(default = "default_interval")]
    interval: u64,
}

fn default_interval() -> u64 {
    5
}

#[derive(Deserialize)]
struct AccessTokenResponse {
    access_token: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: i64,
    endpoints: CopilotEndpoints,
}

#[derive(Deserialize)]
struct CopilotEndpoints {
    api: String,
}

/// Cached Copilot token with its expiry and API endpoint.
#[derive(Clone)]
struct CachedToken {
    token: String,
    api_endpoint: String,
    expires_at: i64,
}

/// Manages Copilot token lifecycle: exchange GitHub token → Copilot token, auto-refresh.
#[derive(Clone)]
pub struct CopilotTokenManager {
    github_token: String,
    cached: Arc<RwLock<Option<CachedToken>>>,
}

impl CopilotTokenManager {
    pub fn new(github_token: String) -> Self {
        Self {
            github_token,
            cached: Arc::new(RwLock::new(None)),
        }
    }

    /// Drops the cached token so the next `get_token` re-exchanges. Called
    /// when Copilot rejects a token mid-validity-window (sign-out elsewhere,
    /// server-side rotation) — expiry-based refresh alone can't recover.
    pub async fn invalidate(&self) {
        let mut cached = self.cached.write().await;
        *cached = None;
    }

    /// Returns a valid (token, api_endpoint) pair, refreshing if expired.
    pub async fn get_token(&self) -> Result<(String, String)> {
        // Check cache first
        {
            let cached = self.cached.read().await;
            if let Some(ref c) = *cached {
                let now = chrono::Utc::now().timestamp();
                // Refresh 60s before expiry
                if now < c.expires_at - 60 {
                    return Ok((c.token.clone(), c.api_endpoint.clone()));
                }
            }
        }

        // Exchange GitHub token for Copilot token
        let client = crate::services::http_utils::aivo_http_client_builder()
            .connect_timeout(std::time::Duration::from_secs(30))
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let resp = client
            .get(COPILOT_TOKEN_URL)
            .header("Authorization", format!("token {}", self.github_token))
            .header("Accept", CONTENT_TYPE_JSON)
            .header("User-Agent", "aivo")
            .send_logged()
            .await
            .context("Failed to exchange GitHub token for Copilot token")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!(
                "Copilot token exchange failed ({}): {}",
                status.as_u16(),
                body
            );
        }

        let token_resp: CopilotTokenResponse = resp
            .json()
            .await
            .context("Failed to parse Copilot token response")?;

        let result = (token_resp.token.clone(), token_resp.endpoints.api.clone());

        // Cache the token
        let mut cached = self.cached.write().await;
        *cached = Some(CachedToken {
            token: token_resp.token,
            api_endpoint: token_resp.endpoints.api,
            expires_at: token_resp.expires_at,
        });

        Ok(result)
    }
}

/// Runs the GitHub OAuth device flow and returns the access token.
///
/// Mirrors the shared device-login UX (`aivo login`, SuperGrok, Kimi Code):
/// show the code, offer Enter-to-open-browser, poll until authorized.
pub async fn device_flow_login() -> Result<String> {
    use crate::services::device_login_ui;
    use crate::style;
    use std::io::IsTerminal;

    let client = crate::services::http_utils::aivo_http_client_builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let resp = client
        .post(DEVICE_CODE_URL)
        .header("Accept", CONTENT_TYPE_JSON)
        .form(&[("client_id", COPILOT_CLIENT_ID), ("scope", "copilot")])
        .send_logged()
        .await
        .context("Failed to request device code from GitHub")?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Device code request failed: {}", body);
    }

    let device: DeviceCodeResponse = resp
        .json()
        .await
        .context("Failed to parse device code response")?;

    // GitHub has no code-prefilled URL; Enter opens the verify page and the
    // user types the code there.
    let open_url = device.verification_uri.clone();
    let interactive = std::io::stdin().is_terminal();

    eprintln!();
    eprintln!("  {}", style::bold("Sign in to GitHub Copilot"));
    eprintln!(
        "  Enter this code when prompted:  {}",
        style::cyan(style::bold(&device.user_code))
    );
    if interactive {
        eprintln!(
            "  Press {} to open your browser, or visit {}",
            style::keycap(" Enter "),
            style::blue(&open_url)
        );
    } else {
        eprintln!("  Visit {} to sign in.", style::blue(&open_url));
    }
    eprintln!();

    let poll = poll_for_token(&client, &device.device_code, device.interval);
    match device_login_ui::wait_for_approval(open_url, poll).await {
        Some(result) => result,
        None => anyhow::bail!("sign-in cancelled"),
    }
}

/// Polls GitHub for the access token until the user authorizes.
async fn poll_for_token(
    client: &reqwest::Client,
    device_code: &str,
    initial_interval: u64,
) -> Result<String> {
    let mut interval = initial_interval;
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

        let resp = client
            .post(ACCESS_TOKEN_URL)
            .header("Accept", CONTENT_TYPE_JSON)
            .form(&[
                ("client_id", COPILOT_CLIENT_ID),
                ("device_code", device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send_logged()
            .await?;

        let token_resp: AccessTokenResponse = resp.json().await?;

        if let Some(token) = token_resp.access_token {
            return Ok(token);
        }

        match token_resp.error.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                // GitHub spec: increase interval by 5 seconds
                interval += 5;
                continue;
            }
            Some("expired_token") => anyhow::bail!("Device code expired. Please try again."),
            Some("access_denied") => anyhow::bail!("Authorization denied by user."),
            Some(err) => anyhow::bail!("OAuth error: {}", err),
            None => continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copilot_token_manager_creation() {
        let mgr = CopilotTokenManager::new("gho_test123".to_string());
        assert_eq!(mgr.github_token, "gho_test123");
    }

    #[test]
    fn test_default_interval() {
        assert_eq!(default_interval(), 5);
    }

    #[test]
    fn test_device_code_response_deserialize() {
        let json = r#"{"device_code":"abc123","user_code":"ABCD-1234","verification_uri":"https://github.com/login/device","interval":8}"#;
        let resp: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.device_code, "abc123");
        assert_eq!(resp.user_code, "ABCD-1234");
        assert_eq!(resp.verification_uri, "https://github.com/login/device");
        assert_eq!(resp.interval, 8);
    }

    #[test]
    fn test_device_code_response_default_interval() {
        let json =
            r#"{"device_code":"abc","user_code":"XY-12","verification_uri":"https://example.com"}"#;
        let resp: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.interval, 5);
    }

    #[test]
    fn test_access_token_response_with_token() {
        let json = r#"{"access_token":"gho_abc123"}"#;
        let resp: AccessTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.access_token.as_deref(), Some("gho_abc123"));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_access_token_response_with_error() {
        let json = r#"{"error":"authorization_pending"}"#;
        let resp: AccessTokenResponse = serde_json::from_str(json).unwrap();
        assert!(resp.access_token.is_none());
        assert_eq!(resp.error.as_deref(), Some("authorization_pending"));
    }

    #[test]
    fn test_copilot_token_response_deserialize() {
        let json = r#"{"token":"tok_abc","expires_at":1700000000,"endpoints":{"api":"https://api.githubcopilot.com"}}"#;
        let resp: CopilotTokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.token, "tok_abc");
        assert_eq!(resp.expires_at, 1700000000);
        assert_eq!(resp.endpoints.api, "https://api.githubcopilot.com");
    }

    #[tokio::test]
    async fn test_copilot_token_manager_cache_starts_empty() {
        let mgr = CopilotTokenManager::new("gho_test".to_string());
        let cached = mgr.cached.read().await;
        assert!(cached.is_none());
    }
}
