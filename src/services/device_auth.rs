//! Device-authorization client for `aivo login` (the getaivo.dev web app).
//!
//! RFC 8628-style: `POST /api/device/code` starts a request and returns a short
//! `user_code` plus a verification URL; `POST /api/device/token` is polled until
//! the user approves the link in the browser. Both calls are Ed25519-signed with
//! this machine's device identity ([`with_starter_headers`]), so the binding is
//! tied to the exact device that started it. No bearer token is issued — the
//! gateway resolves entitlements from the server-side device→user binding.
//!
//! Gotcha: unlike GitHub's device flow (HTTP 200 with `error` in the body), this
//! endpoint returns HTTP 400 for the in-progress states, so the poll loop parses
//! the JSON body regardless of status.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::constants::{AIVO_WEBSITE_BASE_URL, CONTENT_TYPE_JSON};
use crate::errors::{CLIError, ErrorCategory};
use crate::services::device_fingerprint::with_starter_headers;
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils::aivo_http_client_builder;

/// Resolved web base URL: `AIVO_WEBSITE_BASE_URL` env override (for testing
/// against `wrangler pages dev`) else the compiled-in constant, trailing slash
/// trimmed.
pub fn website_base_url() -> String {
    std::env::var("AIVO_WEBSITE_BASE_URL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| AIVO_WEBSITE_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Response of `POST /api/device/code`.
#[derive(Debug, Clone, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    #[serde(default)]
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    #[serde(default = "default_expires_in")]
    pub expires_in: u64,
    #[serde(default = "default_interval")]
    pub interval: u64,
}

fn default_expires_in() -> u64 {
    600
}
fn default_interval() -> u64 {
    5
}

/// The account this device was linked to (from an approved `/api/device/token`).
#[derive(Debug, Clone, Deserialize)]
pub struct LinkedUser {
    pub id: String,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

/// Body of an approved `/api/device/token` (200) and of `/api/device/status`
/// (200) — both report `{ linked, user }`.
#[derive(Deserialize)]
struct LinkedBody {
    #[serde(default)]
    linked: bool,
    #[serde(default)]
    user: Option<LinkedUser>,
}

#[derive(Deserialize)]
struct ErrorBody {
    error: Option<String>,
}

/// Per-window rate caps for the linked account. A `None` field means "no cap".
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct UsageLimits {
    #[serde(default)]
    pub rpm: Option<u64>,
    #[serde(default)]
    pub rpd: Option<u64>,
    #[serde(default)]
    pub tpd: Option<u64>,
    /// Daily cost cap in USD; None/0 = uncapped.
    #[serde(default)]
    pub cpd: Option<f64>,
    /// Hosted web searches/day cap.
    #[serde(default)]
    pub spd: Option<u64>,
}

/// One row of the per-model usage breakdown (current 24h window).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsageModelRow {
    pub model: String,
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub tokens: u64,
}

/// Usage + entitlements for the linked account (gateway `/internal/usage`).
/// Every field is defaulted so gateway-shape drift degrades gracefully.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct UsageSummary {
    #[serde(default)]
    pub plan: Option<String>,
    #[serde(default)]
    pub billing_mode: Option<String>,
    #[serde(default)]
    pub is_pro: bool,
    #[serde(default)]
    pub subscription: Option<serde_json::Value>,
    #[serde(default)]
    pub limits: UsageLimits,
    #[serde(default)]
    pub rpm: u64,
    #[serde(default)]
    pub rpd: u64,
    #[serde(default)]
    pub tpd: u64,
    #[serde(default)]
    pub cpd: f64,
    #[serde(default)]
    pub requests_total: u64,
    #[serde(default)]
    pub tokens_total: u64,
    #[serde(default)]
    pub linked_devices: u64,
    #[serde(default)]
    pub by_model: Vec<UsageModelRow>,
    #[serde(default)]
    pub window_resets_at: Option<String>,
    /// Hosted web search: today's count + lifetime total.
    #[serde(default)]
    pub searches: u64,
    #[serde(default)]
    pub searches_total: u64,
}

/// Body of `/api/device/usage`: a `linked` discriminator over a `UsageSummary`.
#[derive(Deserialize)]
struct UsageBody {
    #[serde(default)]
    linked: bool,
    #[serde(flatten)]
    usage: UsageSummary,
}

/// Result of `/api/device/usage` — parallels [`DeviceStatus`].
pub enum AccountUsage {
    /// Linked: usage + caps for the account. Boxed to keep the enum small.
    Linked(Box<UsageSummary>),
    /// Server says this device isn't linked.
    Unlinked,
    /// Couldn't determine (network/transport/unexpected) — caller must NOT
    /// treat this as unlinked.
    Unknown,
}

/// Pulls a non-empty `error` field out of a JSON error body, if present.
fn parse_error_field(body: &str) -> Option<String> {
    serde_json::from_str::<ErrorBody>(body)
        .ok()
        .and_then(|e| e.error)
        .filter(|s| !s.is_empty())
}

/// Maps an HTTP status to a CLI error category: 5xx → network, 401 → auth,
/// everything else → user.
fn category_for_status(status: reqwest::StatusCode) -> ErrorCategory {
    if status.is_server_error() {
        ErrorCategory::Network
    } else if status.as_u16() == 401 {
        ErrorCategory::Auth
    } else {
        ErrorCategory::User
    }
}

fn http_client() -> reqwest::Client {
    aivo_http_client_builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Display name for this host's OS (e.g. "macOS").
fn os_label() -> &'static str {
    match std::env::consts::OS {
        "macos" => "macOS",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    }
}

/// Body for `POST /api/device/code`: label plus this machine's OS + arch.
fn device_code_body(label: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "label": label,
        "os": os_label(),
        "arch": std::env::consts::ARCH,
    })
}

/// Starts a device-authorization request. `label` is an optional human label
/// shown in the account's device list (e.g. "aivo CLI on host").
pub async fn start_device_auth(label: Option<&str>) -> Result<DeviceCodeResponse> {
    let url = format!("{}/api/device/code", website_base_url());
    let body = device_code_body(label);
    let req = http_client()
        .post(&url)
        .header("Accept", CONTENT_TYPE_JSON)
        .json(&body);
    let resp = with_starter_headers(req)
        .send_logged()
        .await
        .context("Failed to reach the aivo login service")?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(start_error(status, &text));
    }
    serde_json::from_str::<DeviceCodeResponse>(&text)
        .context("Unexpected response from the aivo login service")
}

/// Polls `/api/device/token` every `interval` seconds (honoring `slow_down`)
/// until the user approves, denies, or `expires_in` elapses.
pub async fn poll_device_token(
    device_code: &str,
    interval: u64,
    expires_in: u64,
) -> Result<LinkedUser> {
    let client = http_client();
    let url = format!("{}/api/device/token", website_base_url());
    let deadline = Instant::now() + Duration::from_secs(expires_in.max(1));
    let mut interval = interval.max(1);
    loop {
        if Instant::now() >= deadline {
            return Err(expired_error());
        }
        tokio::time::sleep(Duration::from_secs(interval)).await;
        match poll_once(&client, &url, device_code).await {
            PollOutcome::Linked(user) => return Ok(user),
            // A transient blip just means we try again before the deadline.
            PollOutcome::Pending | PollOutcome::Transient => continue,
            PollOutcome::SlowDown => {
                interval += 5;
                continue;
            }
            PollOutcome::Denied => return Err(denied_error()),
            PollOutcome::Expired => return Err(expired_error()),
            PollOutcome::Fatal(msg) => return Err(fatal_error(msg)),
        }
    }
}

/// Current server-side binding for this device (from `/api/device/status`).
pub enum DeviceStatus {
    /// Linked to this account.
    Linked(LinkedUser),
    /// Definitively not linked (server returned `linked: false`).
    Unlinked,
    /// Couldn't determine (network/transport/unexpected) — caller must NOT
    /// treat this as unlinked, or a blip would wrongly clear local state.
    Unknown,
}

/// Asks the web app whether this device is currently linked. Device-authed
/// (Ed25519); empty body. Never errors — ambiguity collapses to `Unknown`.
pub async fn fetch_device_status() -> DeviceStatus {
    let url = format!("{}/api/device/status", website_base_url());
    let req = http_client().post(&url).header("Accept", CONTENT_TYPE_JSON);
    let resp = match with_starter_headers(req).send_logged().await {
        Ok(r) => r,
        Err(_) => return DeviceStatus::Unknown,
    };
    if !resp.status().is_success() {
        return DeviceStatus::Unknown;
    }
    let text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<LinkedBody>(&text) {
        Ok(LinkedBody {
            linked: true,
            user: Some(user),
        }) => DeviceStatus::Linked(user),
        Ok(LinkedBody { linked: false, .. }) => DeviceStatus::Unlinked,
        _ => DeviceStatus::Unknown,
    }
}

/// Fetches usage + entitlements for this device's account (`/api/device/usage`).
/// Device-authed; empty body. Never errors — ambiguity collapses to `Unknown`.
pub async fn fetch_account_usage() -> AccountUsage {
    let url = format!("{}/api/device/usage", website_base_url());
    let req = http_client().post(&url).header("Accept", CONTENT_TYPE_JSON);
    let resp = match with_starter_headers(req).send_logged().await {
        Ok(r) => r,
        Err(_) => return AccountUsage::Unknown,
    };
    if !resp.status().is_success() {
        return AccountUsage::Unknown;
    }
    let text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<UsageBody>(&text) {
        Ok(b) if b.linked => AccountUsage::Linked(Box::new(b.usage)),
        Ok(_) => AccountUsage::Unlinked,
        Err(_) => AccountUsage::Unknown,
    }
}

/// Removes this device's server-side binding (device-authed self-unlink — the
/// `aivo logout` backend). Returns `Ok(())` once the server confirms; an `Err`
/// (network/transport or non-2xx) lets the caller keep local state consistent.
pub async fn unlink_device() -> Result<()> {
    let url = format!("{}/api/device/unlink", website_base_url());
    let req = http_client().post(&url).header("Accept", CONTENT_TYPE_JSON);
    let resp = with_starter_headers(req)
        .send_logged()
        .await
        .context("Failed to reach the aivo login service")?;
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    let text = resp.text().await.unwrap_or_default();
    Err(unlink_error(status, &text))
}

fn unlink_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    CLIError::new(
        format!("Could not unlink this device (HTTP {}).", status.as_u16()),
        category_for_status(status),
        parse_error_field(body),
        Some(format!(
            "Try again, or unlink it at {}/dashboard/devices.",
            website_base_url()
        )),
    )
    .into()
}

enum PollOutcome {
    Pending,
    SlowDown,
    Linked(LinkedUser),
    Denied,
    Expired,
    /// Network blip or 5xx — retry until the deadline.
    Transient,
    /// Unrecoverable protocol error (bad code, signature, etc.).
    Fatal(String),
}

async fn poll_once(client: &reqwest::Client, url: &str, device_code: &str) -> PollOutcome {
    let body = serde_json::json!({ "device_code": device_code });
    let req = client
        .post(url)
        .header("Accept", CONTENT_TYPE_JSON)
        .json(&body);
    let resp = match with_starter_headers(req).send_logged().await {
        Ok(r) => r,
        Err(_) => return PollOutcome::Transient,
    };
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        return match serde_json::from_str::<LinkedBody>(&text) {
            Ok(LinkedBody {
                user: Some(user), ..
            }) => PollOutcome::Linked(user),
            // 200 without a user shouldn't happen; treat as still-pending.
            _ => PollOutcome::Pending,
        };
    }
    let err = parse_error_field(&text).unwrap_or_default();
    match err.as_str() {
        "authorization_pending" => PollOutcome::Pending,
        "slow_down" => PollOutcome::SlowDown,
        "access_denied" => PollOutcome::Denied,
        "expired_token" => PollOutcome::Expired,
        _ if status.is_server_error() => PollOutcome::Transient,
        "" => PollOutcome::Fatal(format!("login failed (HTTP {})", status.as_u16())),
        other => PollOutcome::Fatal(format!("login failed: {other}")),
    }
}

fn start_error(status: reqwest::StatusCode, body: &str) -> anyhow::Error {
    let err = parse_error_field(body);
    if status.as_u16() == 401 {
        return CLIError::new(
            "The aivo login service rejected this device's signature.",
            ErrorCategory::Auth,
            err,
            Some("Update aivo (`aivo update`) and try again."),
        )
        .into();
    }
    CLIError::new(
        format!("Could not start login (HTTP {}).", status.as_u16()),
        category_for_status(status),
        err,
        None::<String>,
    )
    .into()
}

fn denied_error() -> anyhow::Error {
    CLIError::new(
        "Login was denied in the browser.",
        ErrorCategory::Auth,
        None::<String>,
        Some("Run `aivo login` again and choose Approve."),
    )
    .into()
}

fn expired_error() -> anyhow::Error {
    CLIError::new(
        "Login timed out before it was approved.",
        ErrorCategory::User,
        None::<String>,
        Some("Run `aivo login` again — the code is valid for 10 minutes."),
    )
    .into()
}

fn fatal_error(msg: String) -> anyhow::Error {
    CLIError::new(msg, ErrorCategory::User, None::<String>, None::<String>).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_code_response_deserializes() {
        let json = r#"{
            "device_code":"dc","user_code":"WDJB-MJHT","device_fingerprint":"fp",
            "verification_uri":"https://getaivo.dev/device",
            "verification_uri_complete":"https://getaivo.dev/device?c=WDJB-MJHT",
            "expires_in":600,"interval":5
        }"#;
        let r: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.device_code, "dc");
        assert_eq!(r.user_code, "WDJB-MJHT");
        assert_eq!(
            r.verification_uri_complete.as_deref(),
            Some("https://getaivo.dev/device?c=WDJB-MJHT")
        );
        assert_eq!(r.expires_in, 600);
        assert_eq!(r.interval, 5);
    }

    #[test]
    fn device_code_response_applies_defaults() {
        let json = r#"{"device_code":"dc","user_code":"AB-CD","verification_uri":"u"}"#;
        let r: DeviceCodeResponse = serde_json::from_str(json).unwrap();
        assert_eq!(r.expires_in, 600);
        assert_eq!(r.interval, 5);
        assert!(r.verification_uri_complete.is_none());
    }

    #[test]
    fn device_code_body_carries_os_and_arch() {
        let body = device_code_body(Some("work laptop"));
        assert_eq!(body["label"], "work laptop");
        assert_eq!(body["arch"], std::env::consts::ARCH);
        assert!(body["os"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn device_code_body_allows_absent_label() {
        let body = device_code_body(None);
        assert!(body["label"].is_null());
        assert!(body["os"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn linked_body_carries_the_linked_user() {
        let json = r#"{"linked":true,"user":{"id":"u1","email":"a@b.co","name":"Ann"}}"#;
        let body: LinkedBody = serde_json::from_str(json).unwrap();
        assert!(body.linked);
        let user = body.user.expect("user present");
        assert_eq!(user.id, "u1");
        assert_eq!(user.email.as_deref(), Some("a@b.co"));
        assert_eq!(user.name.as_deref(), Some("Ann"));
    }

    #[test]
    fn linked_body_unlinked_parses() {
        let body: LinkedBody = serde_json::from_str(r#"{"linked":false}"#).unwrap();
        assert!(!body.linked);
        assert!(body.user.is_none());
    }

    #[test]
    fn usage_body_full_payload_parses() {
        let json = r#"{
            "linked":true,"plan":"aivo-pro","billing_mode":"subscription","is_pro":true,
            "subscription":{"status":"active","current_period_end":"2026-07-26T00:00:00Z"},
            "limits":{"rpm":30,"rpd":1000,"tpd":100000,"cpd":5,"spd":10},
            "rpd":120,"tpd":45000,"rpm":3,"cpd":0.42,
            "requests_total":8490,"tokens_total":2100000,"linked_devices":2,
            "searches":4,"searches_total":37,
            "by_model":[{"model":"claude","tokens":1500000,"requests":4200}],
            "window_resets_at":"2026-06-27T14:32:00Z"
        }"#;
        let body: UsageBody = serde_json::from_str(json).unwrap();
        assert!(body.linked);
        let u = body.usage;
        assert_eq!(u.plan.as_deref(), Some("aivo-pro"));
        assert!(u.is_pro);
        assert_eq!(u.limits.rpd, Some(1000));
        assert_eq!(u.limits.cpd, Some(5.0));
        assert_eq!(u.limits.spd, Some(10));
        assert_eq!(u.rpd, 120);
        assert_eq!(u.cpd, 0.42);
        assert_eq!(u.tokens_total, 2_100_000);
        assert_eq!(u.linked_devices, 2);
        assert_eq!(u.searches, 4);
        assert_eq!(u.searches_total, 37);
        assert_eq!(u.by_model.len(), 1);
        assert_eq!(u.by_model[0].model, "claude");
        assert_eq!(u.window_resets_at.as_deref(), Some("2026-06-27T14:32:00Z"));
    }

    #[test]
    fn usage_body_minimal_applies_defaults() {
        let body: UsageBody = serde_json::from_str(r#"{"linked":true}"#).unwrap();
        assert!(body.linked);
        let u = body.usage;
        assert!(u.plan.is_none());
        assert!(!u.is_pro);
        assert_eq!(u.rpd, 0);
        assert_eq!(u.limits.rpm, None);
        assert!(u.by_model.is_empty());
        assert!(u.window_resets_at.is_none());
    }

    #[test]
    fn usage_body_unlinked_parses() {
        let body: UsageBody = serde_json::from_str(r#"{"linked":false}"#).unwrap();
        assert!(!body.linked);
    }

    #[test]
    fn pending_error_body_parses() {
        let json = r#"{"error":"authorization_pending"}"#;
        let e: ErrorBody = serde_json::from_str(json).unwrap();
        assert_eq!(e.error.as_deref(), Some("authorization_pending"));
    }

    #[test]
    fn denied_error_is_auth_category() {
        let e = denied_error();
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::AuthError
        );
    }

    #[test]
    fn expired_error_is_user_category() {
        let e = expired_error();
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::UserError
        );
    }

    #[test]
    fn start_error_401_is_auth() {
        let e = start_error(reqwest::StatusCode::UNAUTHORIZED, r#"{"error":"bad_sig"}"#);
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::AuthError
        );
    }

    #[test]
    fn start_error_5xx_is_network() {
        let e = start_error(reqwest::StatusCode::BAD_GATEWAY, "");
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::NetworkError
        );
    }

    #[test]
    fn unlink_error_401_is_auth() {
        let e = unlink_error(reqwest::StatusCode::UNAUTHORIZED, r#"{"error":"x"}"#);
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::AuthError
        );
    }

    #[test]
    fn unlink_error_5xx_is_network() {
        let e = unlink_error(reqwest::StatusCode::SERVICE_UNAVAILABLE, "");
        assert_eq!(
            crate::errors::exit_code_for_error(&e),
            crate::errors::ExitCode::NetworkError
        );
    }
}
