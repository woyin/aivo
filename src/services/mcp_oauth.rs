//! Interactive OAuth 2.1 for remote (Streamable HTTP) MCP servers.
//!
//! When an HTTP MCP server answers `401 Unauthorized`, this module runs the
//! full MCP authorization handshake and returns a token bundle the transport
//! can attach as `Authorization: Bearer …`:
//!
//! 1. **Protected-resource metadata** (RFC 9728): the `401`'s
//!    `WWW-Authenticate: Bearer resource_metadata="…"` (or a `.well-known`
//!    fallback) names the authorization server + the canonical resource id.
//! 2. **Authorization-server metadata** (RFC 8414, OIDC fallback): the
//!    authorize / token / registration endpoints.
//! 3. **Dynamic client registration** (RFC 7591): self-register a public
//!    client (`token_endpoint_auth_method = none`) with the loopback redirect.
//! 4. **Authorization code + PKCE** (S256), with the `resource` indicator
//!    (RFC 8707): browser flow via [`browser_open`] + the shared ephemeral
//!    loopback callback, then a token exchange.
//!
//! The PKCE pair and `state` are the same primitives the codex OAuth flow
//! uses; the loopback callback server lives in [`loopback_oauth_callback`].
//! The resulting [`McpOAuthCredential`] carries everything needed to silently
//! `refresh` later without re-running discovery; persistence + the `/mcp`
//! authorize UX live in a later phase.

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::services::codex_oauth::{PkcePair, generate_state, redact_oauth_body};
use crate::services::http_debug::LoggedSend;

/// Refresh `access_token` this long before its real expiry to avoid a
/// mid-flight expiration on the next tool call.
pub const MCP_OAUTH_REFRESH_SKEW_SECS: i64 = 120;

/// Client name advertised to the authorization server at registration.
const CLIENT_NAME: &str = "aivo";

/// Tokens + the metadata needed to refresh without re-discovery, persisted
/// (encrypted, in a later phase) per MCP server. `expiry_date` is milliseconds
/// since epoch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct McpOAuthCredential {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    pub token_type: String,
    pub expiry_date: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// The token endpoint, kept so `refresh` needs no re-discovery.
    pub token_endpoint: String,
    /// The dynamically-registered client id, reused on refresh.
    pub client_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// Canonical resource indicator (RFC 8707) sent on token requests.
    pub resource: String,
    /// The MCP endpoint URL this token was authorized against (the configured
    /// `url` at authorize time). The bearer is only re-attached to a request
    /// whose URL shares this origin, so re-pointing a server to a new host under
    /// the same name can't leak the old host's token. `Option` for forward-compat
    /// with credentials saved before this field existed (falls back to
    /// `resource`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authorized_url: Option<String>,
    pub last_refresh: DateTime<Utc>,
}

impl McpOAuthCredential {
    pub fn is_expired(&self, skew_secs: i64) -> bool {
        let now_ms = Utc::now().timestamp_millis();
        now_ms + skew_secs * 1000 >= self.expiry_date
    }

    /// The `Authorization` header value. Always `Bearer …` regardless of the
    /// server's `token_type` casing (RFC 6750 schemes are case-insensitive, but
    /// `Bearer` is the canonical spelling every server accepts).
    pub fn authorization_header(&self) -> String {
        format!("Bearer {}", self.access_token)
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).context("serialize McpOAuthCredential")
    }

    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse McpOAuthCredential JSON")
    }

    fn from_token_response(
        t: TokenResponse,
        token_endpoint: &str,
        reg: &ClientRegistration,
        resource: &str,
        authorized_url: &str,
    ) -> Self {
        McpOAuthCredential {
            access_token: t.access_token,
            refresh_token: t.refresh_token,
            token_type: t.token_type.unwrap_or_else(|| "Bearer".to_string()),
            expiry_date: compute_expiry(t.expires_in),
            scope: t.scope,
            token_endpoint: token_endpoint.to_string(),
            client_id: reg.client_id.clone(),
            client_secret: reg.client_secret.clone(),
            resource: resource.to_string(),
            authorized_url: Some(authorized_url.to_string()),
            last_refresh: Utc::now(),
        }
    }

    /// Whether this credential may be attached to a request to `url`: true only
    /// when `url` shares the origin the token was authorized against (the
    /// configured endpoint, falling back to the RFC 8707 `resource` for
    /// credentials saved before `authorized_url` existed).
    pub fn applies_to(&self, url: &str) -> bool {
        let authorized = self.authorized_url.as_deref().unwrap_or(&self.resource);
        same_origin(authorized, url)
    }
}

/// Token endpoints rarely omit `expires_in`; default to one hour when they do.
/// Saturating arithmetic so a hostile/buggy endpoint returning a huge (or
/// negative) `expires_in` can't overflow-panic the connect/refresh path.
fn compute_expiry(expires_in: Option<i64>) -> i64 {
    Utc::now()
        .timestamp_millis()
        .saturating_add(expires_in.unwrap_or(3600).saturating_mul(1000))
}

// ---- discovery / registration / token wire shapes ------------------------

#[derive(serde::Deserialize)]
struct ProtectedResourceMetadata {
    #[serde(default)]
    authorization_servers: Vec<String>,
    #[serde(default)]
    resource: Option<String>,
    #[serde(default)]
    scopes_supported: Option<Vec<String>>,
}

#[derive(serde::Deserialize)]
struct AuthServerMetadata {
    authorization_endpoint: String,
    token_endpoint: String,
    #[serde(default)]
    registration_endpoint: Option<String>,
    #[serde(default)]
    scopes_supported: Option<Vec<String>>,
}

#[derive(Serialize)]
struct ClientRegistrationRequest<'a> {
    client_name: &'a str,
    redirect_uris: Vec<String>,
    grant_types: Vec<&'a str>,
    response_types: Vec<&'a str>,
    token_endpoint_auth_method: &'a str,
}

#[derive(serde::Deserialize)]
struct ClientRegistration {
    client_id: String,
    #[serde(default)]
    client_secret: Option<String>,
}

#[derive(serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<i64>,
    #[serde(default)]
    token_type: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

// ---- pure helpers (unit-tested) -------------------------------------------

/// Extract the `resource_metadata="…"` URL from a `WWW-Authenticate` header
/// (RFC 9728 §5.1). Tolerates ordering, spacing, and an unquoted value.
fn parse_resource_metadata_url(www_authenticate: &str) -> Option<String> {
    const KEY: &str = "resource_metadata";
    let lower = www_authenticate.to_ascii_lowercase();
    let idx = lower.find(KEY)?;
    let after = www_authenticate[idx + KEY.len()..].trim_start();
    let after = after.strip_prefix('=')?.trim_start();
    if let Some(rest) = after.strip_prefix('"') {
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        let end = after.find([',', ' ', ';']).unwrap_or(after.len());
        let v = after[..end].trim();
        (!v.is_empty()).then(|| v.to_string())
    }
}

/// `.well-known` protected-resource URLs to try when the `401` carried no
/// `resource_metadata` (RFC 9728): origin-root first, then the path-scoped form.
fn protected_resource_candidates(server_url: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(u) = url::Url::parse(server_url) {
        let origin = u.origin().ascii_serialization();
        out.push(format!("{origin}/.well-known/oauth-protected-resource"));
        let path = u.path().trim_end_matches('/');
        if !path.is_empty() {
            out.push(format!(
                "{origin}/.well-known/oauth-protected-resource{path}"
            ));
        }
    }
    out
}

/// Authorization-server metadata URLs to try for an issuer: RFC 8414
/// (`oauth-authorization-server`) then OIDC (`openid-configuration`), each in
/// path-insertion and path-appended forms so both conventions are covered.
fn auth_server_metadata_urls(issuer: &str) -> Vec<String> {
    let trimmed = issuer.trim_end_matches('/');
    let mut out = Vec::new();
    if let Ok(u) = url::Url::parse(trimmed) {
        let origin = u.origin().ascii_serialization();
        let path = u.path().trim_end_matches('/');
        for kind in ["oauth-authorization-server", "openid-configuration"] {
            if path.is_empty() {
                out.push(format!("{origin}/.well-known/{kind}"));
            } else {
                // RFC 8414 inserts the well-known segment before the issuer path;
                // many servers also answer the path-appended form.
                out.push(format!("{origin}/.well-known/{kind}{path}"));
                out.push(format!("{trimmed}/.well-known/{kind}"));
            }
        }
    }
    out
}

/// The canonical resource indicator (RFC 8707) for an MCP server URL: the URL
/// without a fragment.
fn canonical_resource(server_url: &str) -> String {
    match url::Url::parse(server_url) {
        Ok(mut u) => {
            u.set_fragment(None);
            u.to_string()
        }
        Err(_) => server_url.to_string(),
    }
}

/// Whether two URLs share an origin (scheme + host + port). The guard that stops
/// a token issued for one host being attached to a request to a different host.
/// Conservative: an unparseable side is treated as a non-match.
fn same_origin(a: &str, b: &str) -> bool {
    match (url::Url::parse(a), url::Url::parse(b)) {
        (Ok(a), Ok(b)) => a.origin() == b.origin(),
        _ => false,
    }
}

/// Pick a space-joined scope string from the advertised metadata (protected
/// resource first, then the auth server), or `None` to omit `scope` entirely.
fn choose_scope(prm: &ProtectedResourceMetadata, asm: &AuthServerMetadata) -> Option<String> {
    prm.scopes_supported
        .as_ref()
        .or(asm.scopes_supported.as_ref())
        .filter(|s| !s.is_empty())
        .map(|s| s.join(" "))
}

/// The authorize URL the user opens (PKCE S256 + RFC 8707 `resource`).
fn build_authorize_url(
    authorization_endpoint: &str,
    client_id: &str,
    challenge: &str,
    state: &str,
    redirect_uri: &str,
    resource: &str,
    scope: Option<&str>,
) -> String {
    let enc = crate::services::percent_codec::encode;
    let sep = if authorization_endpoint.contains('?') {
        '&'
    } else {
        '?'
    };
    let mut url = format!(
        "{authorization_endpoint}{sep}response_type=code\
         &client_id={}\
         &redirect_uri={}\
         &code_challenge={challenge}\
         &code_challenge_method=S256\
         &state={state}\
         &resource={}",
        enc(client_id),
        enc(redirect_uri),
        enc(resource),
    );
    if let Some(s) = scope {
        url.push_str("&scope=");
        url.push_str(&enc(s));
    }
    url
}

// ---- HTTP steps -----------------------------------------------------------

async fn fetch_json<T: DeserializeOwned>(client: &reqwest::Client, url: &str) -> Result<T> {
    let resp = client
        .get(url)
        .header(reqwest::header::ACCEPT, "application/json")
        .send_logged()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        bail!("GET {url} → HTTP {}", resp.status().as_u16());
    }
    resp.json()
        .await
        .with_context(|| format!("parse JSON from {url}"))
}

async fn resolve_protected_resource(
    client: &reqwest::Client,
    server_url: &str,
    www_authenticate: Option<&str>,
) -> Result<ProtectedResourceMetadata> {
    let candidates = match www_authenticate.and_then(parse_resource_metadata_url) {
        Some(url) => vec![url],
        None => protected_resource_candidates(server_url),
    };
    let mut last_err = None;
    for url in &candidates {
        match fetch_json::<ProtectedResourceMetadata>(client, url).await {
            Ok(m) if !m.authorization_servers.is_empty() => return Ok(m),
            Ok(_) => last_err = Some(anyhow!("{url} listed no authorization_servers")),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow!("no protected-resource metadata for {server_url} (server didn't advertise OAuth)")
    }))
}

async fn fetch_auth_server_metadata(
    client: &reqwest::Client,
    issuer: &str,
) -> Result<AuthServerMetadata> {
    let mut last_err = None;
    for url in &auth_server_metadata_urls(issuer) {
        match fetch_json::<AuthServerMetadata>(client, url).await {
            Ok(m) => return Ok(m),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("no authorization-server metadata for {issuer}")))
}

async fn register_client(
    client: &reqwest::Client,
    registration_endpoint: &str,
    redirect_uri: &str,
) -> Result<ClientRegistration> {
    let body = ClientRegistrationRequest {
        client_name: CLIENT_NAME,
        redirect_uris: vec![redirect_uri.to_string()],
        grant_types: vec!["authorization_code", "refresh_token"],
        response_types: vec!["code"],
        token_endpoint_auth_method: "none",
    };
    let resp = client
        .post(registration_endpoint)
        .json(&body)
        .send_logged()
        .await
        .context("POST registration endpoint (dynamic client registration)")?;
    let status = resp.status();
    if !status.is_success() {
        let b = resp.text().await.unwrap_or_default();
        bail!(
            "dynamic client registration failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&b)
        );
    }
    resp.json()
        .await
        .context("parse client registration response")
}

async fn exchange_code(
    client: &reqwest::Client,
    token_endpoint: &str,
    reg: &ClientRegistration,
    code: &str,
    verifier: &str,
    redirect_uri: &str,
    resource: &str,
) -> Result<TokenResponse> {
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("client_id", reg.client_id.as_str()),
        ("code_verifier", verifier),
        ("resource", resource),
    ];
    if let Some(secret) = &reg.client_secret {
        form.push(("client_secret", secret.as_str()));
    }
    let resp = client
        .post(token_endpoint)
        .form(&form)
        .send_logged()
        .await
        .context("POST token endpoint (authorization_code)")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!(
            "MCP token exchange failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }
    resp.json().await.context("parse token response")
}

/// Refresh `access_token` using the stored `refresh_token`. Servers may rotate
/// the refresh token; the new one is kept when present.
pub async fn refresh(creds: &mut McpOAuthCredential) -> Result<()> {
    let Some(refresh_token) = creds.refresh_token.clone() else {
        bail!("no refresh_token stored for this MCP server; re-authorization required");
    };
    let client = crate::services::http_utils::router_http_client_with_timeout(30);
    let mut form: Vec<(&str, &str)> = vec![
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token.as_str()),
        ("client_id", creds.client_id.as_str()),
        ("resource", creds.resource.as_str()),
    ];
    if let Some(secret) = &creds.client_secret {
        form.push(("client_secret", secret.as_str()));
    }
    let resp = client
        .post(&creds.token_endpoint)
        .form(&form)
        .send_logged()
        .await
        .context("POST token endpoint (refresh_token)")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!(
            "MCP token refresh failed ({}): {}",
            status.as_u16(),
            redact_oauth_body(&body)
        );
    }
    let tokens: TokenResponse = resp.json().await.context("parse refresh response")?;
    creds.access_token = tokens.access_token;
    if let Some(rt) = tokens.refresh_token {
        creds.refresh_token = Some(rt);
    }
    if let Some(tt) = tokens.token_type {
        creds.token_type = tt;
    }
    if let Some(sc) = tokens.scope {
        creds.scope = Some(sc);
    }
    creds.expiry_date = compute_expiry(tokens.expires_in);
    creds.last_refresh = Utc::now();
    Ok(())
}

/// Refresh only if near expiry. Returns `true` when a refresh happened (so the
/// caller knows to persist the new tokens).
pub async fn ensure_fresh(creds: &mut McpOAuthCredential, skew_secs: i64) -> Result<bool> {
    crate::services::oauth_credential::ensure_fresh(creds, skew_secs).await
}

impl crate::services::oauth_credential::OAuthCredential for McpOAuthCredential {
    fn is_expired(&self, skew_secs: i64) -> bool {
        McpOAuthCredential::is_expired(self, skew_secs)
    }
    async fn refresh(&mut self) -> Result<()> {
        refresh(self).await
    }
}

/// Full interactive authorization for an HTTP MCP server. `www_authenticate` is
/// the `401`'s header (when present) — it points straight at the protected-
/// resource metadata; otherwise the `.well-known` fallback is used.
///
/// `on_authorize_url` is invoked once with the browser authorize URL before the
/// flow blocks on the loopback callback — the caller surfaces it (a TUI shows it;
/// a CLI prints it) so the user has it even if the auto-opened browser fails.
/// This function never reads stdin, so it's safe to run from inside a TUI that
/// already owns the terminal.
pub async fn authorize(
    server_url: &str,
    www_authenticate: Option<&str>,
    on_authorize_url: impl FnOnce(&str),
) -> Result<McpOAuthCredential> {
    use crate::services::browser_open;
    use crate::services::loopback_oauth_callback::{
        CALLBACK_PATH, bind_loopback, wait_for_callback,
    };
    use std::time::Duration;

    let client = crate::services::http_utils::router_http_client_with_timeout(30);

    // 1. Protected-resource metadata → authorization server + resource id.
    let prm = resolve_protected_resource(&client, server_url, www_authenticate).await?;
    let issuer = prm.authorization_servers.first().cloned().ok_or_else(|| {
        anyhow!("MCP server's protected-resource metadata lists no authorization_servers")
    })?;
    let resource = prm
        .resource
        .clone()
        .unwrap_or_else(|| canonical_resource(server_url));

    // 2. Authorization-server metadata.
    let asm = fetch_auth_server_metadata(&client, &issuer).await?;

    // 3. Bind the loopback redirect, then self-register a client for it.
    let binding = bind_loopback().await?;
    let redirect_uri = format!("http://127.0.0.1:{}{CALLBACK_PATH}", binding.port());
    let registration_endpoint = asm.registration_endpoint.clone().ok_or_else(|| {
        anyhow!(
            "authorization server doesn't support dynamic client registration; \
             aivo can't self-register a client"
        )
    })?;
    let reg = register_client(&client, &registration_endpoint, &redirect_uri).await?;

    // 4. PKCE authorize via the browser + the shared loopback callback.
    let pkce = PkcePair::generate();
    let state = generate_state();
    let scope = choose_scope(&prm, &asm);
    let authorize_url = build_authorize_url(
        &asm.authorization_endpoint,
        &reg.client_id,
        &pkce.challenge,
        &state,
        &redirect_uri,
        &resource,
        scope.as_deref(),
    );

    // Hand the URL to the caller, then best-effort auto-open the browser.
    on_authorize_url(&authorize_url);
    let _ = browser_open::open_url(&authorize_url);

    let outcome = wait_for_callback(binding, &state, Duration::from_secs(300)).await?;

    // 5. Exchange the code for tokens.
    let tokens = exchange_code(
        &client,
        &asm.token_endpoint,
        &reg,
        &outcome.code,
        &pkce.verifier,
        &redirect_uri,
        &resource,
    )
    .await?;

    Ok(McpOAuthCredential::from_token_response(
        tokens,
        &asm.token_endpoint,
        &reg,
        &resource,
        server_url,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_resource_metadata_from_www_authenticate() {
        let h = r#"Bearer resource_metadata="https://h/.well-known/oauth-protected-resource""#;
        assert_eq!(
            parse_resource_metadata_url(h).as_deref(),
            Some("https://h/.well-known/oauth-protected-resource")
        );
        // Other params present, different ordering / spacing.
        let h = r#"Bearer realm="x", error="invalid_token", resource_metadata = "https://h/prm""#;
        assert_eq!(
            parse_resource_metadata_url(h).as_deref(),
            Some("https://h/prm")
        );
        // Unquoted value, terminated by a comma.
        let h = "Bearer resource_metadata=https://h/prm, realm=x";
        assert_eq!(
            parse_resource_metadata_url(h).as_deref(),
            Some("https://h/prm")
        );
        // Absent → None.
        assert!(parse_resource_metadata_url(r#"Bearer realm="x""#).is_none());
    }

    #[test]
    fn protected_resource_candidates_cover_origin_and_path() {
        let c = protected_resource_candidates("https://mcp.example.com/mcp");
        assert_eq!(
            c[0],
            "https://mcp.example.com/.well-known/oauth-protected-resource"
        );
        assert!(c.contains(
            &"https://mcp.example.com/.well-known/oauth-protected-resource/mcp".to_string()
        ));
        // A bare origin → just the root form (no path variant).
        let c = protected_resource_candidates("https://mcp.example.com");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn auth_server_metadata_urls_cover_both_conventions() {
        // Issuer with a path → RFC 8414 path-insertion + path-appended, OAuth + OIDC.
        let c = auth_server_metadata_urls("https://auth.example.com/tenant1");
        assert!(c.contains(
            &"https://auth.example.com/.well-known/oauth-authorization-server/tenant1".to_string()
        ));
        assert!(c.contains(
            &"https://auth.example.com/tenant1/.well-known/oauth-authorization-server".to_string()
        ));
        assert!(c.contains(
            &"https://auth.example.com/.well-known/openid-configuration/tenant1".to_string()
        ));
        // Bare-origin issuer → host-root forms only.
        let c = auth_server_metadata_urls("https://auth.example.com");
        assert_eq!(
            c[0],
            "https://auth.example.com/.well-known/oauth-authorization-server"
        );
        assert!(
            c.contains(&"https://auth.example.com/.well-known/openid-configuration".to_string())
        );
    }

    #[test]
    fn canonical_resource_strips_fragment() {
        assert_eq!(canonical_resource("https://h/mcp#frag"), "https://h/mcp");
        assert_eq!(canonical_resource("https://h/mcp"), "https://h/mcp");
    }

    #[test]
    fn same_origin_compares_scheme_host_port() {
        // Same origin, different path → match (same trust domain).
        assert!(same_origin(
            "https://mcp.linear.app/mcp",
            "https://mcp.linear.app/other"
        ));
        // Different host / scheme / port → no match.
        assert!(!same_origin(
            "https://mcp.linear.app/mcp",
            "https://evil.example.com/mcp"
        ));
        assert!(!same_origin("https://h/mcp", "http://h/mcp"));
        assert!(!same_origin("https://h:443/mcp", "https://h:8443/mcp"));
        // Unparseable either side → conservative no-match.
        assert!(!same_origin("not a url", "https://h/mcp"));
    }

    fn cred_with(authorized_url: Option<&str>, resource: &str) -> McpOAuthCredential {
        McpOAuthCredential {
            access_token: "at".into(),
            refresh_token: None,
            token_type: "Bearer".into(),
            expiry_date: 0,
            scope: None,
            token_endpoint: "https://auth/token".into(),
            client_id: "cid".into(),
            client_secret: None,
            resource: resource.into(),
            authorized_url: authorized_url.map(str::to_string),
            last_refresh: Utc::now(),
        }
    }

    #[test]
    fn applies_to_binds_token_to_authorized_endpoint() {
        // Bound to the authorized endpoint origin (path may differ).
        let c = cred_with(
            Some("https://mcp.linear.app/mcp"),
            "https://api.linear.app/",
        );
        assert!(c.applies_to("https://mcp.linear.app/mcp"));
        assert!(c.applies_to("https://mcp.linear.app/v2")); // same origin
        // A different host is rejected even if the (RFC 8707) resource would match
        // it — this is the regression guard: a CDN-fronted server whose resource
        // origin differs from its endpoint is keyed on the *endpoint*.
        assert!(!c.applies_to("https://api.linear.app/x"));
        assert!(!c.applies_to("https://evil.example.com/mcp"));
        // Forward-compat: a credential saved before `authorized_url` existed falls
        // back to `resource`.
        let old = cred_with(None, "https://mcp.notion.com/mcp");
        assert!(old.applies_to("https://mcp.notion.com/sse"));
        assert!(!old.applies_to("https://elsewhere.com/mcp"));
    }

    #[test]
    fn authorize_url_has_pkce_and_resource() {
        let url = build_authorize_url(
            "https://auth.example.com/authorize",
            "client-123",
            "chal",
            "state-xyz",
            "http://127.0.0.1:5311/oauth2callback",
            "https://mcp.example.com/mcp",
            Some("read write"),
        );
        assert!(url.starts_with("https://auth.example.com/authorize?"));
        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=client-123"));
        assert!(url.contains("code_challenge=chal"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=state-xyz"));
        // Redirect + resource are percent-encoded.
        assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A5311%2Foauth2callback"));
        assert!(url.contains("resource=https%3A%2F%2Fmcp.example.com%2Fmcp"));
        assert!(url.contains("scope=read%20write"));
        // An endpoint already carrying a query gets `&`, not a second `?`.
        let url = build_authorize_url(
            "https://a/authorize?x=1",
            "c",
            "ch",
            "s",
            "http://127.0.0.1:1/oauth2callback",
            "https://r",
            None,
        );
        assert!(url.contains("authorize?x=1&response_type=code"));
        assert!(!url.contains("scope="));
    }

    #[test]
    fn credential_json_roundtrip_and_expiry() {
        let c = McpOAuthCredential {
            access_token: "at".into(),
            refresh_token: Some("rt".into()),
            token_type: "Bearer".into(),
            expiry_date: Utc::now().timestamp_millis() + 60_000,
            scope: Some("read".into()),
            token_endpoint: "https://auth/token".into(),
            client_id: "cid".into(),
            client_secret: None,
            resource: "https://mcp/x".into(),
            authorized_url: Some("https://mcp/x".into()),
            last_refresh: Utc::now(),
        };
        let back = McpOAuthCredential::from_json(&c.to_json().unwrap()).unwrap();
        assert_eq!(back, c);
        assert_eq!(back.authorization_header(), "Bearer at");
        // +60s remaining: 120s skew → expired; 30s skew → fresh.
        assert!(c.is_expired(120));
        assert!(!c.is_expired(30));
    }

    #[test]
    fn credential_accepts_minimal_shape() {
        let json = r#"{
            "access_token": "at",
            "token_type": "Bearer",
            "expiry_date": 0,
            "token_endpoint": "https://auth/token",
            "client_id": "cid",
            "resource": "https://mcp/x",
            "last_refresh": "2026-01-01T00:00:00Z"
        }"#;
        let c = McpOAuthCredential::from_json(json).unwrap();
        assert!(c.refresh_token.is_none());
        assert!(c.client_secret.is_none());
        assert!(c.scope.is_none());
    }
}
