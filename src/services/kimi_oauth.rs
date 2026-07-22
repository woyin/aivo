//! Kimi Code OAuth device-code flow (RFC 8628). The credential is a *provider*
//! bearer — OpenAI-compatible inference at `api.kimi.com/coding/v1` — usable by
//! any coding agent. Endpoints, client id, and `X-Msh-*` headers match the real
//! `kimi` CLI (MoonshotAI/kimi-code, `packages/oauth`). The server rotates the
//! refresh token on every refresh but does not revoke the prior one (verified
//! live), so persisting rotations is about freshness, not survival.

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::services::grok_oauth::redact_oauth_body;

pub const CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";

pub const OAUTH_HOST: &str = "https://auth.kimi.com";
pub const DEVICE_AUTH_URL: &str = "https://auth.kimi.com/api/oauth/device_authorization";
pub const TOKEN_URL: &str = "https://auth.kimi.com/api/oauth/token";
pub const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

pub const INFERENCE_BASE_URL: &str = "https://api.kimi.com/coding/v1";

pub const KIMI_OAUTH_SENTINEL: &str = "kimi-oauth";

/// Client identity the real CLI sends. The `X-Msh-*` set is optional (verified
/// live) but matching the CLI keeps us off any future enforcement path. Device
/// name/model/OS headers are omitted — no reason to ship the user's hostname.
pub const KIMI_USER_AGENT: &str = "kimi-code-cli/0.27.0";
pub const CLIENT_VERSION: &str = "0.27.0";
pub const PLATFORM_HEADER: &str = "X-Msh-Platform";
pub const PLATFORM_VALUE: &str = "kimi_code_cli";
pub const VERSION_HEADER: &str = "X-Msh-Version";
pub const DEVICE_ID_HEADER: &str = "X-Msh-Device-Id";

pub const REFRESH_SKEW_SECS: i64 = 120;

/// Discriminator: Grok JSON lacks `provider`, so Kimi's parse rejects it. The
/// reverse doesn't hold — Grok's looser parse accepts Kimi JSON — so provider
/// probes must try Kimi before Grok.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KimiProviderTag {
    #[serde(rename = "kimi-code")]
    KimiCode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KimiOAuthCredential {
    pub provider: KimiProviderTag,
    pub access_token: String,
    pub refresh_token: String,
    /// Stable per-login device id, echoed as `X-Msh-Device-Id` (the server
    /// bakes it into issued tokens).
    pub device_id: String,
    pub expires_at: DateTime<Utc>,
    pub last_refresh: DateTime<Utc>,
}

impl KimiOAuthCredential {
    pub fn is_expired(&self, skew_secs: i64) -> bool {
        Utc::now() + ChronoDuration::seconds(skew_secs) >= self.expires_at
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize KimiOAuthCredential")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse KimiOAuthCredential JSON")
    }
}

fn client() -> reqwest::Client {
    crate::services::http_utils::router_http_client_with_timeout(30)
}

fn with_client_headers(b: reqwest::RequestBuilder, device_id: &str) -> reqwest::RequestBuilder {
    b.header("User-Agent", KIMI_USER_AGENT)
        .header("Accept", "application/json")
        .header(PLATFORM_HEADER, PLATFORM_VALUE)
        .header(VERSION_HEADER, CLIENT_VERSION)
        .header(DEVICE_ID_HEADER, device_id)
}

#[derive(Deserialize)]
struct DeviceAuthResponse {
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

async fn request_device_authorization(device_id: &str) -> Result<DeviceAuthResponse> {
    let resp = with_client_headers(client().post(DEVICE_AUTH_URL), device_id)
        .form(&[("client_id", CLIENT_ID)])
        .send()
        .await
        .context("POST /api/oauth/device_authorization")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "device-authorization request failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }
    serde_json::from_str::<DeviceAuthResponse>(&body).with_context(|| {
        format!(
            "parse /api/oauth/device_authorization response (status {}): {}",
            status.as_u16(),
            body.chars().take(300).collect::<String>()
        )
    })
}

async fn poll_device_token(
    device_code: &str,
    device_id: &str,
    initial_interval: u64,
    expires_in: i64,
) -> Result<KimiOAuthCredential> {
    let mut interval = initial_interval.max(1);
    let deadline = Utc::now() + ChronoDuration::seconds(expires_in.max(1));

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
        if Utc::now() >= deadline {
            anyhow::bail!("device code expired before approval — run login again");
        }

        let resp = with_client_headers(client().post(TOKEN_URL), device_id)
            .form(&[
                ("grant_type", DEVICE_GRANT_TYPE),
                ("client_id", CLIENT_ID),
                ("device_code", device_code),
            ])
            .send()
            .await
            .context("POST /api/oauth/token (device_code)")?;

        if resp.status().is_success() {
            let tokens: TokenResponse = resp.json().await.context("parse token response")?;
            return credential_from_tokens(tokens, device_id.to_string());
        }

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
impl crate::services::oauth_credential::OAuthCredential for KimiOAuthCredential {
    fn is_expired(&self, skew_secs: i64) -> bool {
        KimiOAuthCredential::is_expired(self, skew_secs)
    }
    async fn refresh(&mut self) -> Result<()> {
        refresh(self).await
    }
}

impl crate::services::oauth_credential::StoredOAuthCredential for KimiOAuthCredential {
    fn key_matches(key: &crate::services::session_store::ApiKey) -> bool {
        key.is_kimi_oauth()
    }
    fn from_json(json: &str) -> Result<Self> {
        KimiOAuthCredential::from_json(json)
    }
    fn to_json(&self) -> Result<String> {
        KimiOAuthCredential::to_json(self)
    }
    fn refresh_token(&self) -> &str {
        &self.refresh_token
    }
}

pub async fn refresh(creds: &mut KimiOAuthCredential) -> Result<()> {
    let resp = with_client_headers(client().post(TOKEN_URL), &creds.device_id)
        .form(&[
            ("grant_type", "refresh_token"),
            ("client_id", CLIENT_ID),
            ("refresh_token", creds.refresh_token.as_str()),
        ])
        .send()
        .await
        .context("POST /api/oauth/token (refresh_token)")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        // 400..=404 mirrors `is_oauth_invalid_grant`.
        let relogin_hint = if matches!(status.as_u16(), 400..=404) {
            crate::services::oauth_credential::REAUTH_HINT
        } else {
            ""
        };
        anyhow::bail!(
            "kimi token refresh failed ({}){}: {}",
            status.as_u16(),
            relogin_hint,
            redact_oauth_body(&body)
        );
    }

    let tokens: TokenResponse = resp.json().await.context("parse refresh response")?;
    let now = Utc::now();
    creds.access_token = tokens.access_token;
    if let Some(new_refresh) = tokens.refresh_token {
        creds.refresh_token = new_refresh;
    }
    // Access tokens live ~900s; default matches the observed server value.
    creds.expires_at = now + ChronoDuration::seconds(tokens.expires_in.unwrap_or(900));
    creds.last_refresh = now;
    Ok(())
}

/// Refreshes only if near expiry; `true` if it did (caller persists).
pub async fn ensure_fresh(creds: &mut KimiOAuthCredential, skew_secs: i64) -> Result<bool> {
    crate::services::oauth_credential::ensure_fresh(creds, skew_secs).await
}

/// Writes a rotated credential back to the store so later processes start from
/// the freshest token. Best-effort; matches the kimi-oauth entry by base_url,
/// then by pre-rotation refresh token.
pub async fn persist_rotated_credential(
    store: &crate::services::session_store::SessionStore,
    prev_refresh_token: &str,
    creds: &KimiOAuthCredential,
) {
    crate::services::oauth_credential::persist_rotated_credential(store, prev_refresh_token, creds)
        .await
}

/// One `/models` entry; the coding endpoint reports per-model context length.
pub struct KimiModel {
    pub id: String,
    pub context_length: Option<u64>,
}

pub async fn fetch_models(
    creds: &mut KimiOAuthCredential,
    persist: Option<&crate::services::session_store::SessionStore>,
) -> Result<Vec<KimiModel>> {
    let prev_refresh = creds.refresh_token.clone();
    if ensure_fresh(creds, REFRESH_SKEW_SECS).await?
        && let Some(store) = persist
    {
        persist_rotated_credential(store, &prev_refresh, creds).await;
    }
    let url = format!("{}/models", INFERENCE_BASE_URL.trim_end_matches('/'));
    let resp = with_client_headers(client().get(&url), &creds.device_id)
        .header("Authorization", format!("Bearer {}", creds.access_token))
        .send()
        .await
        .context("GET kimi /models")?;

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    if status.as_u16() == 402 {
        anyhow::bail!(
            "Kimi Code requires an active membership plan for this account: {}",
            redact_oauth_body(&body)
        );
    }
    if !status.is_success() {
        // After ensure_fresh above, 401/403 means the sign-in itself is dead.
        let relogin_hint = if matches!(status.as_u16(), 401 | 403) {
            crate::services::oauth_credential::REAUTH_HINT
        } else {
            ""
        };
        anyhow::bail!(
            "kimi models request failed ({}){}: {}",
            status.as_u16(),
            relogin_hint,
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
        #[serde(default)]
        context_length: Option<u64>,
    }
    let parsed: ModelsResp = serde_json::from_str(&body).context("parse kimi /models response")?;
    Ok(parsed
        .data
        .into_iter()
        .map(|m| KimiModel {
            id: m.id,
            context_length: m.context_length,
        })
        .collect())
}

fn credential_from_tokens(tokens: TokenResponse, device_id: String) -> Result<KimiOAuthCredential> {
    let refresh_token = tokens
        .refresh_token
        .ok_or_else(|| anyhow!("token response missing refresh_token"))?;
    let now = Utc::now();
    Ok(KimiOAuthCredential {
        provider: KimiProviderTag::KimiCode,
        access_token: tokens.access_token,
        refresh_token,
        device_id,
        expires_at: now + ChronoDuration::seconds(tokens.expires_in.unwrap_or(900)),
        last_refresh: now,
    })
}

/// Device-code sign-in: shows the user code and polls until approved.
pub async fn interactive_login() -> Result<KimiOAuthCredential> {
    use crate::services::device_login_ui;
    use crate::style;
    use std::io::IsTerminal;

    let device_id = crate::services::codex_oauth::generate_session_id();
    let device = request_device_authorization(&device_id).await?;

    let open_url = device
        .verification_uri_complete
        .clone()
        .or_else(|| device.verification_uri.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| OAUTH_HOST.to_string());
    let interactive = std::io::stdin().is_terminal();

    eprintln!();
    eprintln!("  {}", style::bold("Sign in to Kimi Code"));
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

    let poll = poll_device_token(
        &device.device_code,
        &device_id,
        device.interval,
        device.expires_in,
    );
    match device_login_ui::wait_for_approval(open_url, poll).await {
        Some(result) => result,
        None => anyhow::bail!("sign-in cancelled"),
    }
}

/// Per-request auth resolved by the token manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KimiAuth {
    pub base_url: String,
    pub bearer: String,
    pub device_id: String,
}

/// Request-time auth: refreshes the OAuth token on expiry (they live ~15 min),
/// persisting rotations when a store is attached.
#[derive(Clone)]
pub struct KimiTokenManager {
    creds: Arc<RwLock<KimiOAuthCredential>>,
    persist_store: Option<crate::services::session_store::SessionStore>,
}

impl KimiTokenManager {
    pub fn new(creds: KimiOAuthCredential) -> Self {
        Self {
            creds: Arc::new(RwLock::new(creds)),
            persist_store: None,
        }
    }

    /// Persist rotations to `store` so they survive process exit.
    pub fn with_persist_store(
        mut self,
        store: crate::services::session_store::SessionStore,
    ) -> Self {
        self.persist_store = Some(store);
        self
    }

    pub async fn authorize(&self) -> Result<KimiAuth> {
        let mut creds = self.creds.write().await;
        if creds.is_expired(REFRESH_SKEW_SECS) {
            let prev_refresh = creds.refresh_token.clone();
            refresh(&mut creds).await?;
            if let Some(store) = &self.persist_store {
                persist_rotated_credential(store, &prev_refresh, &creds).await;
            }
        }
        Ok(KimiAuth {
            base_url: INFERENCE_BASE_URL.to_string(),
            bearer: creds.access_token.clone(),
            device_id: creds.device_id.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_credential() -> KimiOAuthCredential {
        KimiOAuthCredential {
            provider: KimiProviderTag::KimiCode,
            access_token: "at".into(),
            refresh_token: "rt".into(),
            device_id: "d-id".into(),
            expires_at: Utc::now() + ChronoDuration::seconds(900),
            last_refresh: Utc::now(),
        }
    }

    #[test]
    fn credential_json_roundtrip() {
        let c = test_credential();
        let back = KimiOAuthCredential::from_json(&c.to_json().unwrap()).unwrap();
        assert_eq!(back, c);
        assert!(c.to_json().unwrap().contains("\"kimi-code\""));
    }

    #[test]
    fn is_expired_respects_skew() {
        let mut c = test_credential();
        c.expires_at = Utc::now() + ChronoDuration::seconds(60);
        assert!(c.is_expired(120));
        assert!(!c.is_expired(30));
        c.expires_at = Utc::now() - ChronoDuration::seconds(1);
        assert!(c.is_expired(0));
    }

    #[test]
    fn provider_tag_disambiguates_from_grok() {
        // Grok JSON lacks the `provider` tag → kimi parse must reject it.
        let grok = crate::services::grok_oauth::GrokOAuthCredential {
            access_token: "at".into(),
            refresh_token: "rt".into(),
            account_label: None,
            expires_at: Utc::now(),
            last_refresh: Utc::now(),
        };
        assert!(KimiOAuthCredential::from_json(&grok.to_json().unwrap()).is_err());
        // The reverse is the known ambiguity: grok's looser shape accepts kimi
        // JSON, which is why provider probes must try kimi BEFORE grok.
        let kimi_json = test_credential().to_json().unwrap();
        assert!(crate::services::grok_oauth::GrokOAuthCredential::from_json(&kimi_json).is_ok());
    }

    #[tokio::test]
    async fn manager_authorize_uses_current_token_when_fresh() {
        let mgr = KimiTokenManager::new(test_credential());
        let auth = mgr.authorize().await.unwrap();
        assert_eq!(auth.base_url, INFERENCE_BASE_URL);
        assert_eq!(auth.bearer, "at");
        assert_eq!(auth.device_id, "d-id");
    }

    #[tokio::test]
    async fn persist_rotated_credential_writes_back_new_refresh_token() {
        use crate::services::session_store::SessionStore;

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let original = test_credential();
        store
            .add_key_with_protocol(
                "kimi",
                KIMI_OAUTH_SENTINEL,
                None,
                &original.to_json().unwrap(),
            )
            .await
            .unwrap();

        let rotated = KimiOAuthCredential {
            access_token: "at1".into(),
            refresh_token: "rt1".into(),
            ..original.clone()
        };
        persist_rotated_credential(&store, &original.refresh_token, &rotated).await;

        let keys = store.get_keys().await.unwrap();
        let mut stored = keys.into_iter().find(|k| k.is_kimi_oauth()).unwrap();
        SessionStore::decrypt_key_secret(&mut stored).unwrap();
        let reloaded = KimiOAuthCredential::from_json(&stored.key).unwrap();
        assert_eq!(reloaded.refresh_token, "rt1");
        assert_eq!(reloaded.access_token, "at1");
    }
}
