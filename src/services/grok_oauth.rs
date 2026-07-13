//! xAI SuperGrok OAuth device-code flow. Unlike Codex/Claude OAuth, the
//! credential is a *provider* bearer (OpenAI-compatible inference at
//! `cli-chat-proxy.grok.com`), usable by any coding agent. Endpoints/client-id/
//! scopes/headers match the real `grok` CLI's requests against `auth.x.ai`.

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";

/// `accounts.x.ai` is the sign-in SPA; the OAuth API lives on `auth.x.ai`.
pub const ISSUER: &str = "https://accounts.x.ai";
pub const TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
pub const DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";

pub const SCOPE: &str = "openid profile email offline_access grok-cli:access api:access conversations:read conversations:write";
pub const REFERRER: &str = "grok-build";
pub const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// OAuth inference host — NOT `api.x.ai`, which is only for console keys.
pub const INFERENCE_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";
pub const FALLBACK_API_BASE_URL: &str = "https://api.x.ai/v1";

pub const TOKEN_AUTH_HEADER: &str = "X-XAI-Token-Auth";
pub const TOKEN_AUTH_VALUE: &str = "xai-grok-cli";
pub const MODEL_OVERRIDE_HEADER: &str = "x-grok-model-override";
/// The inference proxy 426s without a current client version.
pub const CLIENT_VERSION_HEADER: &str = "x-grok-client-version";
pub const CLIENT_SURFACE_HEADER: &str = "x-grok-client-surface";
pub const INFERENCE_SURFACE: &str = "grok-build";

pub const GROK_OAUTH_SENTINEL: &str = "grok-oauth";

const CLIENT_SURFACE: &str = "headless";
pub const CLIENT_VERSION: &str = "0.2.93";
const GROK_USER_AGENT: &str = "grok-shell/0.2.93 (macos; aarch64)";

/// Client-identity headers the real CLI sends, so xAI's middleware accepts it.
fn with_client_headers(b: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    b.header("User-Agent", GROK_USER_AGENT)
        .header("Accept", "*/*")
        .header("x-grok-client-surface", CLIENT_SURFACE)
        .header("x-grok-client-version", CLIENT_VERSION)
}

pub const REFRESH_SKEW_SECS: i64 = 120;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GrokOAuthCredential {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_label: Option<String>,
    pub expires_at: DateTime<Utc>,
    pub last_refresh: DateTime<Utc>,
}

impl GrokOAuthCredential {
    pub fn is_expired(&self, skew_secs: i64) -> bool {
        Utc::now() + ChronoDuration::seconds(skew_secs) >= self.expires_at
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize GrokOAuthCredential")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse GrokOAuthCredential JSON")
    }
}

fn client() -> reqwest::Client {
    crate::services::http_utils::router_http_client_with_timeout(30)
}

#[derive(Deserialize)]
struct DeviceCodeResponse {
    device_code: String,
    user_code: String,
    #[serde(default)]
    verification_uri: Option<String>,
    #[serde(default)]
    verification_uri_complete: Option<String>,
    #[serde(default = "default_interval")]
    interval: u64,
    #[serde(default = "default_device_expiry")]
    expires_in: i64,
}

fn default_interval() -> u64 {
    5
}
fn default_device_expiry() -> i64 {
    900
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
}

#[derive(Deserialize)]
struct TokenErrorResponse {
    error: Option<String>,
}

/// Device-code request — no PKCE (matches the real CLI).
async fn request_device_code() -> Result<DeviceCodeResponse> {
    let resp = with_client_headers(client().post(DEVICE_CODE_URL))
        .form(&[
            ("client_id", CLIENT_ID),
            ("scope", SCOPE),
            ("referrer", REFERRER),
        ])
        .send()
        .await
        .context("POST /oauth2/device/code")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "device-code request failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }
    serde_json::from_str::<DeviceCodeResponse>(&body).with_context(|| {
        format!(
            "parse /oauth2/device/code response (status {}): {}",
            status.as_u16(),
            body.chars().take(300).collect::<String>()
        )
    })
}

/// Polls `/oauth2/token` until the user approves or the code expires.
async fn poll_device_token(
    device_code: &str,
    initial_interval: u64,
    expires_in: i64,
) -> Result<GrokOAuthCredential> {
    let mut interval = initial_interval.max(1);
    let deadline = Utc::now() + ChronoDuration::seconds(expires_in.max(1));

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
        if Utc::now() >= deadline {
            anyhow::bail!("device code expired before approval — run login again");
        }

        let resp = with_client_headers(client().post(TOKEN_URL))
            .form(&[
                ("grant_type", DEVICE_GRANT_TYPE),
                ("client_id", CLIENT_ID),
                ("device_code", device_code),
            ])
            .send()
            .await
            .context("POST /oauth2/token (device_code)")?;

        if resp.status().is_success() {
            let tokens: TokenResponse = resp.json().await.context("parse token response")?;
            return credential_from_tokens(tokens, None);
        }

        // Non-2xx: expect an OAuth error code steering the poll.
        let body = resp.text().await.unwrap_or_default();
        let err = serde_json::from_str::<TokenErrorResponse>(&body)
            .ok()
            .and_then(|e| e.error);
        match err.as_deref() {
            Some("authorization_pending") => continue,
            Some("slow_down") => {
                interval += 5;
                continue;
            }
            Some("expired_token") => anyhow::bail!("device code expired — run login again"),
            Some("access_denied") => anyhow::bail!("authorization denied"),
            Some(other) => anyhow::bail!("OAuth error: {other}"),
            None => anyhow::bail!("token poll failed: {}", redact_oauth_body(&body)),
        }
    }
}

/// Refreshes `access_token`, rotating `refresh_token` when reissued.
pub async fn refresh(creds: &mut GrokOAuthCredential) -> Result<()> {
    let resp = with_client_headers(client().post(TOKEN_URL))
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", creds.refresh_token.as_str()),
        ])
        .send()
        .await
        .context("POST /oauth2/token (refresh_token)")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!(
            "refresh failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }

    let tokens: TokenResponse = resp.json().await.context("parse refresh response")?;
    let now = Utc::now();
    creds.access_token = tokens.access_token;
    if let Some(new_refresh) = tokens.refresh_token {
        creds.refresh_token = new_refresh;
    }
    creds.expires_at = now + ChronoDuration::seconds(tokens.expires_in.unwrap_or(3600));
    creds.last_refresh = now;
    Ok(())
}

/// Refreshes only if near expiry; `true` if it did (caller persists).
pub async fn ensure_fresh(creds: &mut GrokOAuthCredential, skew_secs: i64) -> Result<bool> {
    if creds.is_expired(skew_secs) {
        refresh(creds).await?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Lists model ids via the CLI proxy's `/v1/models`.
pub async fn fetch_model_ids(creds: &mut GrokOAuthCredential) -> Result<Vec<String>> {
    ensure_fresh(creds, REFRESH_SKEW_SECS).await?;
    let url = format!("{}/models", INFERENCE_BASE_URL.trim_end_matches('/'));
    let resp = client()
        .get(&url)
        .header("Authorization", format!("Bearer {}", creds.access_token))
        .header(TOKEN_AUTH_HEADER, TOKEN_AUTH_VALUE)
        .header(CLIENT_VERSION_HEADER, CLIENT_VERSION)
        .header(CLIENT_SURFACE_HEADER, INFERENCE_SURFACE)
        .header("User-Agent", GROK_USER_AGENT)
        .header("Accept", "application/json")
        .send()
        .await
        .context("GET grok /v1/models")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "grok models request failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }

    #[derive(Deserialize)]
    struct ModelsResp {
        data: Vec<ModelEntry>,
    }
    #[derive(Deserialize)]
    struct ModelEntry {
        id: String,
    }
    let parsed: ModelsResp =
        serde_json::from_str(&body).context("parse grok /v1/models response")?;
    Ok(parsed.data.into_iter().map(|m| m.id).collect())
}

fn credential_from_tokens(
    tokens: TokenResponse,
    account_label: Option<String>,
) -> Result<GrokOAuthCredential> {
    let refresh_token = tokens
        .refresh_token
        .ok_or_else(|| anyhow!("token response missing refresh_token (offline_access scope?)"))?;
    let now = Utc::now();
    Ok(GrokOAuthCredential {
        access_token: tokens.access_token,
        refresh_token,
        account_label,
        expires_at: now + ChronoDuration::seconds(tokens.expires_in.unwrap_or(3600)),
        last_refresh: now,
    })
}

/// Device-code sign-in: show the code, offer Enter-to-open-browser, poll until
/// approved. Mirrors the `aivo login` UX — the browser open is a convenience
/// (the poll runs regardless), and Ctrl+C cancels cleanly.
pub async fn interactive_login() -> Result<GrokOAuthCredential> {
    use crate::services::device_login_ui;
    use crate::style;
    use std::io::IsTerminal;

    let device = request_device_code().await?;

    // The `_complete` URL pre-fills the code, so opening or scanning it needs no
    // typing; it's also what Enter opens.
    let open_url = device
        .verification_uri_complete
        .clone()
        .or_else(|| device.verification_uri.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| ISSUER.to_string());
    let interactive = std::io::stdin().is_terminal();

    eprintln!();
    eprintln!("  {}", style::bold("Sign in to SuperGrok"));
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

    let poll = poll_device_token(&device.device_code, device.interval, device.expires_in);
    match device_login_ui::wait_for_approval(open_url, poll).await {
        Some(result) => result,
        None => anyhow::bail!("sign-in cancelled"),
    }
}

/// Masks token values in a response body before logging.
pub fn redact_oauth_body(body: &str) -> String {
    let mut out = body.to_string();
    for key in ["access_token", "refresh_token", "code", "code_verifier"] {
        let needle = format!("\"{}\"", key);
        let mut cursor = 0usize;
        while let Some(rel_idx) = out[cursor..].find(&needle) {
            let idx = cursor + rel_idx;
            let after_key = idx + needle.len();
            let rest = &out[after_key..];
            let Some(colon) = rest.find(':') else { break };
            let Some(open) = rest[colon..].find('"') else {
                cursor = after_key;
                continue;
            };
            let Some(close_rel) = rest[colon + open + 1..].find('"') else {
                cursor = after_key;
                continue;
            };
            let start = after_key + colon + open + 1;
            let end = start + close_rel;
            out.replace_range(start..end, "<redacted>");
            cursor = start + "<redacted>".len();
        }
    }
    out
}

/// Per-request auth: which upstream to hit and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrokAuth {
    pub base_url: String,
    pub bearer: String,
    /// The console `XAI_API_KEY` fallback (skips the CLI-session headers).
    pub is_api_key: bool,
}

/// Request-time auth: refreshes the OAuth token on expiry, and after a 403
/// (tier-gating) switches to the `XAI_API_KEY` fallback for the session.
#[derive(Clone)]
pub struct GrokTokenManager {
    creds: Arc<RwLock<GrokOAuthCredential>>,
    fallback_api_key: Option<String>,
    gated: Arc<RwLock<bool>>,
}

impl GrokTokenManager {
    pub fn new(creds: GrokOAuthCredential, fallback_api_key: Option<String>) -> Self {
        Self {
            creds: Arc::new(RwLock::new(creds)),
            fallback_api_key,
            gated: Arc::new(RwLock::new(false)),
        }
    }

    /// Resolves auth, refreshing on expiry; the fallback path once gated.
    pub async fn authorize(&self) -> Result<GrokAuth> {
        if *self.gated.read().await
            && let Some(api_key) = &self.fallback_api_key
        {
            return Ok(GrokAuth {
                base_url: FALLBACK_API_BASE_URL.to_string(),
                bearer: api_key.clone(),
                is_api_key: true,
            });
        }

        let mut creds = self.creds.write().await;
        if creds.is_expired(REFRESH_SKEW_SECS) {
            refresh(&mut creds).await?;
        }
        Ok(GrokAuth {
            base_url: INFERENCE_BASE_URL.to_string(),
            bearer: creds.access_token.clone(),
            is_api_key: false,
        })
    }

    /// Latches the API-key fallback after a 403; `true` if one is configured.
    pub async fn mark_gated(&self) -> bool {
        if self.fallback_api_key.is_some() {
            *self.gated.write().await = true;
            true
        } else {
            false
        }
    }

    /// The current credential, for persisting rotated tokens after a run.
    pub async fn current_credential(&self) -> GrokOAuthCredential {
        self.creds.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn credential_json_roundtrip() {
        let c = GrokOAuthCredential {
            access_token: "at".into(),
            refresh_token: "rt".into(),
            account_label: Some("alice".into()),
            expires_at: Utc::now(),
            last_refresh: Utc::now(),
        };
        let back = GrokOAuthCredential::from_json(&c.to_json().unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn is_expired_respects_skew() {
        let mut c = GrokOAuthCredential {
            access_token: String::new(),
            refresh_token: String::new(),
            account_label: None,
            expires_at: Utc::now() + ChronoDuration::seconds(60),
            last_refresh: Utc::now(),
        };
        assert!(c.is_expired(120));
        assert!(!c.is_expired(30));
        c.expires_at = Utc::now() - ChronoDuration::seconds(1);
        assert!(c.is_expired(0));
    }

    #[test]
    fn redact_masks_token_values() {
        let body = r#"{"access_token":"real-at","refresh_token":"real-rt","expires_in":3600}"#;
        let red = redact_oauth_body(body);
        assert!(!red.contains("real-at"));
        assert!(!red.contains("real-rt"));
        assert!(red.contains("<redacted>"));
        assert!(red.contains("3600"));
    }

    #[tokio::test]
    async fn manager_authorize_uses_oauth_when_fresh() {
        let creds = GrokOAuthCredential {
            access_token: "session-tok".into(),
            refresh_token: "rt".into(),
            account_label: None,
            expires_at: Utc::now() + ChronoDuration::seconds(3600),
            last_refresh: Utc::now(),
        };
        let mgr = GrokTokenManager::new(creds, Some("xai-key".into()));
        let auth = mgr.authorize().await.unwrap();
        assert_eq!(auth.base_url, INFERENCE_BASE_URL);
        assert_eq!(auth.bearer, "session-tok");
        assert!(!auth.is_api_key);
    }

    #[tokio::test]
    async fn manager_falls_back_to_api_key_after_gating() {
        let creds = GrokOAuthCredential {
            access_token: "session-tok".into(),
            refresh_token: "rt".into(),
            account_label: None,
            expires_at: Utc::now() + ChronoDuration::seconds(3600),
            last_refresh: Utc::now(),
        };
        let mgr = GrokTokenManager::new(creds, Some("xai-key".into()));
        assert!(mgr.mark_gated().await);
        let auth = mgr.authorize().await.unwrap();
        assert_eq!(auth.base_url, FALLBACK_API_BASE_URL);
        assert_eq!(auth.bearer, "xai-key");
        assert!(auth.is_api_key);
    }

    #[tokio::test]
    async fn manager_gating_noops_without_fallback_key() {
        let creds = GrokOAuthCredential {
            access_token: "session-tok".into(),
            refresh_token: "rt".into(),
            account_label: None,
            expires_at: Utc::now() + ChronoDuration::seconds(3600),
            last_refresh: Utc::now(),
        };
        let mgr = GrokTokenManager::new(creds, None);
        assert!(!mgr.mark_gated().await);
        // Still resolves to the OAuth path (no fallback to switch to).
        let auth = mgr.authorize().await.unwrap();
        assert!(!auth.is_api_key);
    }
}
