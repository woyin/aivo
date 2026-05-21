//! Cursor ACP provider glue: model discovery (Phase 1) and chat turn driver
//! (Phase 2). The driver speaks ACP/JSON-RPC over stdio against the hidden
//! `cursor-agent acp` subcommand via [`crate::services::acp_client::AcpClient`].

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use tokio::process::Command;

use crate::services::acp_client::{AcpClient, PermissionDecision, PromptEvent, PromptStream};
use crate::services::cursor_home_shadow::CursorShadow;
use crate::services::session_store::{ApiKey, AttachmentStorage, MessageAttachment};

pub const CURSOR_ACP_SENTINEL: &str = "cursor";
/// `key.key` prefix that marks a shadow-isolated cursor account. The slug
/// after the colon is the [`CursorShadow::account_id`].
pub const CURSOR_SHADOW_PREFIX: &str = "cursor-shadow:";
/// Pre-shadow sentinel that delegated to the user's real ~/.cursor login.
/// Retired — any key still carrying it is asked to re-add so cursor-agent
/// is never accidentally pointed at the user's global cursor home.
const LEGACY_CURSOR_LOGIN_SENTINEL: &str = "cursor-login";
const CURSOR_AGENT_BIN: &str = "cursor-agent";

/// Detects pre-shadow cursor keys that haven't been re-added since the
/// switch to per-account isolation. The launcher refuses to use them so
/// cursor-agent never reads/writes the user's real ~/.cursor through aivo.
pub fn is_legacy_cursor_login_secret(secret: &str) -> bool {
    secret == LEGACY_CURSOR_LOGIN_SENTINEL
}

/// Env var that flips Cursor's tool-execution policy from "graceful reject"
/// to "allow_once" for every `session/request_permission`. Off by default —
/// when aivo is the model provider, the launched tool (Claude/Codex) is
/// already the agent, and letting cursor-agent run shell/file ops in parallel
/// surprises users. Set to `1` to let cursor-agent execute its own tools.
pub const CURSOR_ALLOW_TOOLS_ENV: &str = "AIVO_CURSOR_ALLOW_TOOLS";

fn cursor_permission_decision(_params: &Value) -> PermissionDecision {
    match std::env::var(CURSOR_ALLOW_TOOLS_ENV) {
        Ok(v) if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") => {
            PermissionDecision::Allow
        }
        _ => PermissionDecision::Reject,
    }
}

pub fn is_cursor_acp_base(base_url: &str) -> bool {
    base_url == CURSOR_ACP_SENTINEL
}

/// Resolved cursor-agent path. PATH is walked at most once per process; a
/// missing binary caches `None` so the second call doesn't re-stat.
fn resolved_cursor_agent_path() -> Option<&'static PathBuf> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            let dirs = crate::services::path_search::collect_path_dirs();
            crate::services::path_search::find_in_dirs(CURSOR_AGENT_BIN, &dirs)
        })
        .as_ref()
}

pub fn ensure_cursor_agent_installed() -> Result<()> {
    if resolved_cursor_agent_path().is_some() {
        Ok(())
    } else {
        anyhow::bail!(
            "`cursor-agent` was not found on PATH. Install Cursor CLI support first, then run `aivo keys add cursor`."
        )
    }
}

/// Bare cursor-agent Command with no env applied. Prefer
/// [`cursor_agent_command_for_key`] so the shadow env is wired in.
fn cursor_agent_command_bare() -> Command {
    match resolved_cursor_agent_path() {
        Some(path) => Command::new(path),
        None => Command::new(CURSOR_AGENT_BIN),
    }
}

/// Build a Command for spawning cursor-agent against the given key. Wires
/// in the shadow's HOME/XDG/APPDATA + CURSOR_CONFIG_DIR/CURSOR_DATA_DIR
/// when the key is a shadow account; otherwise falls back to a bare spawn
/// (used by raw `--key sk-…` API-key keys).
pub fn cursor_agent_command_for_key(key: &ApiKey) -> Result<Command> {
    let mut cmd = cursor_agent_command_bare();
    apply_shadow_env_to_command(&mut cmd, key)?;
    apply_color_env_to_command(&mut cmd);
    if let Some((name, value)) = cursor_auth_env(key) {
        cmd.env(name, value);
    } else {
        cmd.env_remove("CURSOR_API_KEY");
    }
    Ok(cmd)
}

/// Same as [`cursor_agent_command_for_key`] but takes an explicit shadow.
/// Used by the `aivo keys add cursor` flow before any aivo key exists.
pub fn cursor_agent_command_for_shadow(shadow: &CursorShadow) -> Command {
    let mut cmd = cursor_agent_command_bare();
    for (name, value) in shadow.env_block() {
        cmd.env(name, value);
    }
    apply_color_env_to_command(&mut cmd);
    cmd.env_remove("CURSOR_API_KEY");
    cmd
}

fn apply_shadow_env_to_command(cmd: &mut Command, key: &ApiKey) -> Result<()> {
    if let Some(shadow) = cursor_shadow_for_key(key)? {
        ensure_shadow_once(&shadow)?;
        for (name, value) in shadow.env_block() {
            cmd.env(name, value);
        }
    }
    Ok(())
}

/// Runs `shadow.ensure()` at most once per `(process, account_id)`. The
/// first call repairs broken pre-fix shadows (`set-keychain-settings`,
/// missing `Library/Keychains/login.keychain-db`); subsequent spawns reuse
/// the result without re-shelling `/usr/bin/security`.
fn ensure_shadow_once(shadow: &CursorShadow) -> Result<()> {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen.lock().unwrap().contains(&shadow.account_id) {
        return Ok(());
    }
    shadow.ensure()?;
    seen.lock().unwrap().insert(shadow.account_id.clone());
    Ok(())
}

fn apply_color_env_to_command(cmd: &mut Command) {
    cmd.env("NO_COLOR", "1")
        .env("CLICOLOR", "0")
        .env("CLICOLOR_FORCE", "0");
}

/// Parsed shape of a cursor `key.key` secret. The wire format is
/// extensible — additional `:<kind>:<value>` segments can be added later
/// (e.g. `:auth_token:`, `:base_url:`) without breaking older callers.
///
/// - `cursor-shadow:<id>`              → OAuth login, no embedded creds
/// - `cursor-shadow:<id>:api:<key>`    → API-key auth, key embedded
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CursorSecret<'a> {
    pub account_id: &'a str,
    pub api_key: Option<&'a str>,
}

pub fn parse_cursor_shadow_secret(secret: &str) -> Option<CursorSecret<'_>> {
    let rest = secret.strip_prefix(CURSOR_SHADOW_PREFIX)?;
    let mut parts = rest.splitn(3, ':');
    let account_id = parts.next().filter(|s| !s.is_empty())?;
    match (parts.next(), parts.next()) {
        (None, _) => Some(CursorSecret {
            account_id,
            api_key: None,
        }),
        (Some("api"), Some(key)) if !key.is_empty() => Some(CursorSecret {
            account_id,
            api_key: Some(key),
        }),
        _ => None,
    }
}

/// Build the secret string for a freshly-allocated OAuth-login shadow.
pub fn build_cursor_oauth_secret(account_id: &str) -> String {
    format!("{CURSOR_SHADOW_PREFIX}{account_id}")
}

/// Build the secret string for an API-key shadow account. Caller must
/// have validated the api_key (no `:`, non-empty).
pub fn build_cursor_apikey_secret(account_id: &str, api_key: &str) -> String {
    format!("{CURSOR_SHADOW_PREFIX}{account_id}:api:{api_key}")
}

/// Extract the shadow account id encoded in `key.key`, if any.
pub fn cursor_account_id(key: &ApiKey) -> Option<&str> {
    parse_cursor_shadow_secret(key.key.as_str()).map(|s| s.account_id)
}

/// Resolve the on-disk shadow for a cursor key. Returns `Ok(None)` for
/// non-shadow keys (those don't exist after the multi-account refactor;
/// kept for forward compatibility).
pub fn cursor_shadow_for_key(key: &ApiKey) -> Result<Option<CursorShadow>> {
    match cursor_account_id(key) {
        Some(id) => Ok(Some(CursorShadow::for_account_id(id.to_string())?)),
        None => Ok(None),
    }
}

pub fn cursor_auth_env(key: &ApiKey) -> Option<(&'static str, OsString)> {
    let secret = key.key.as_str();
    if let Some(parsed) = parse_cursor_shadow_secret(secret) {
        return parsed
            .api_key
            .map(|k| ("CURSOR_API_KEY", OsString::from(k)));
    }
    if secret.is_empty() || is_legacy_cursor_login_secret(secret) {
        None
    } else {
        Some(("CURSOR_API_KEY", OsString::from(secret)))
    }
}

pub fn cursor_models_cache_identity(key: &ApiKey) -> String {
    let secret = key.key.as_str();
    if secret.is_empty() {
        return CURSOR_ACP_SENTINEL.to_string();
    }
    // Shadow keys: cache per account id. Two `aivo keys add cursor` runs
    // are independent accounts even if they happen to log in as the same
    // upstream user, so each id deserves its own cache slot. The cache
    // key only carries the account id — embedded API keys (if any) are
    // sensitive and don't belong in plain-text cache filenames.
    if let Some(parsed) = parse_cursor_shadow_secret(secret) {
        return format!("{CURSOR_ACP_SENTINEL}#shadow-{}", parsed.account_id);
    }
    let digest = Sha256::digest(secret.as_bytes());
    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        hex.push_str(&format!("{byte:02x}"));
    }
    format!("{CURSOR_ACP_SENTINEL}#{hex}")
}

pub async fn list_cursor_models(key: &ApiKey) -> Result<Vec<String>> {
    ensure_cursor_agent_installed()?;

    if is_legacy_cursor_login_secret(key.key.as_str()) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Run `aivo keys rm {0}` then `aivo keys add cursor` to recreate it as an isolated account.",
            key.id
        );
    }

    // OAuth-login shadow keys: `cursor-agent status` is authoritative.
    // (API-key shadows can't use status — cursor-agent only inspects
    // auth.json there, ignoring CURSOR_API_KEY, so a valid API-key key
    // would report "unauthenticated".)
    if let Some(parsed) = parse_cursor_shadow_secret(key.key.as_str())
        && parsed.api_key.is_none()
        && !cursor_status_authenticated_for_key(key)
            .await
            .unwrap_or(false)
    {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth {0}` to sign in again.",
            key.id
        );
    }

    let primary = run_cursor_agent(["models"], key).await;
    let output = match primary {
        Ok(output) => output,
        Err(primary_err) => run_cursor_agent(["--list-models"], key)
            .await
            .with_context(|| {
                format!(
                    "`cursor-agent models` failed ({primary_err}); fallback `cursor-agent --list-models` also failed"
                )
            })?,
    };

    if looks_like_no_models_message(&output) {
        let parsed = parse_cursor_shadow_secret(key.key.as_str());
        let hint = match parsed {
            Some(p) if p.api_key.is_some() => format!(
                "Run `aivo keys reauth {0}` and paste a valid Cursor API key.",
                key.id
            ),
            Some(_) => format!(
                "Run `aivo keys reauth {0}` to sign in again — the saved cursor account has been signed out.",
                key.id
            ),
            _ => "Re-run `aivo keys add cursor` to set up a fresh isolated cursor account."
                .to_string(),
        };
        anyhow::bail!("Cursor returned no models for this account. {hint}");
    }

    let models = parse_cursor_models(&output)?;
    if models.is_empty() {
        anyhow::bail!("Cursor returned no model IDs");
    }
    Ok(models)
}

/// Recognize cursor-agent's prose responses for the unauthenticated /
/// no-quota cases. The CLI exits 0 in these states, so we can't lean on
/// the exit code — only the body text.
fn looks_like_no_models_message(output: &str) -> bool {
    let stripped = crate::services::ansi::strip_ansi(output);
    let lower = stripped.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    lower.starts_with("no models")
        || lower.contains("not logged in")
        || lower.contains("not authenticated")
        || lower.contains("please log in")
        || lower.contains("please sign in")
}

async fn cursor_status_authenticated(mut cmd: Command) -> Result<bool> {
    let output = cmd
        .args(["status", "--format", "json"])
        .stdin(Stdio::null())
        .output()
        .await
        .context("failed to run `cursor-agent status --format json`")?;
    if !output.status.success() {
        return Ok(false);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_cursor_status_authenticated(&stdout).unwrap_or(false))
}

/// `cursor-agent status --format json` against the given key's shadow
/// (or the user's global cursor home if the key isn't shadow-backed).
pub async fn cursor_status_authenticated_for_key(key: &ApiKey) -> Result<bool> {
    ensure_cursor_agent_installed()?;
    cursor_status_authenticated(cursor_agent_command_for_key(key)?).await
}

/// Status check against an explicit shadow — used by `aivo keys add
/// cursor` after the login flow finishes and before an `ApiKey` exists.
pub async fn cursor_status_authenticated_for_shadow(shadow: &CursorShadow) -> Result<bool> {
    ensure_cursor_agent_installed()?;
    cursor_status_authenticated(cursor_agent_command_for_shadow(shadow)).await
}

/// Run `cursor-agent login` into the given shadow. Inherits stdio so the
/// device-code prompt and browser-open hint reach the user.
pub async fn run_cursor_login_for_shadow(shadow: &CursorShadow) -> Result<()> {
    ensure_cursor_agent_installed()?;
    let status = cursor_agent_command_for_shadow(shadow)
        .arg("login")
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .context("failed to run `cursor-agent login`")?;
    if !status.success() {
        anyhow::bail!("`cursor-agent login` exited with {status}");
    }
    Ok(())
}

async fn run_cursor_agent<const N: usize>(args: [&str; N], key: &ApiKey) -> Result<String> {
    let mut cmd = cursor_agent_command_for_key(key)?;
    cmd.args(args).stdin(Stdio::null());

    let output = cmd.output().await.context("failed to run cursor-agent")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            anyhow::bail!("cursor-agent exited with {}", output.status);
        }
        anyhow::bail!("cursor-agent exited with {}: {}", output.status, detail);
    }

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if text.trim().is_empty() {
        text = String::from_utf8_lossy(&output.stderr).to_string();
    }
    Ok(text)
}

pub fn parse_cursor_models(output: &str) -> Result<Vec<String>> {
    let stripped = crate::services::ansi::strip_ansi(output);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        let mut out = Vec::new();
        collect_model_ids_from_json(&value, &mut out);
        out.sort();
        out.dedup();
        return Ok(out);
    }

    let mut out = Vec::new();
    for line in stripped.lines() {
        if let Some(model) = parse_cursor_model_line(line) {
            out.push(model);
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_model_ids_from_json(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_model_ids_from_json(item, out);
            }
        }
        Value::Object(map) => {
            let mut recursed = false;
            for key in ["models", "data", "items", "result"] {
                if let Some(child) = map.get(key) {
                    collect_model_ids_from_json(child, out);
                    recursed = true;
                }
            }
            if recursed {
                return;
            }
            for key in ["id", "model", "name"] {
                if let Some(id) = map.get(key).and_then(Value::as_str)
                    && is_plausible_model_id(id)
                {
                    out.push(id.trim().to_string());
                    return;
                }
            }
        }
        Value::String(s) if is_plausible_model_id(s) => out.push(s.trim().to_string()),
        _ => {}
    }
}

fn parse_cursor_model_line(line: &str) -> Option<String> {
    let mut s = line.trim();
    if s.is_empty() {
        return None;
    }

    let lower = s.to_ascii_lowercase();
    let heading = lower.ends_with(':')
        || matches!(
            lower.as_str(),
            "models" | "available models" | "cursor models" | "model"
        );
    if heading {
        return None;
    }

    // Prose lines (multi-word sentences ending in punctuation) aren't model
    // ids even when their first word happens to be a plausible identifier —
    // e.g. cursor-agent emits "No models available for this account." when
    // signed out, and the unguarded parser used to serve "No" as a model.
    if s.split_whitespace().count() > 1 && matches!(s.chars().last(), Some('.' | '!' | '?')) {
        return None;
    }

    s = s.trim_start_matches(['*', '-', '+', '•', '·']).trim_start();
    s = s.trim_start_matches(['>', '✓', '✔', '●', '○']).trim_start();
    for marker in [
        "(current)",
        "[current]",
        "(default)",
        "[default]",
        "current:",
        "default:",
    ] {
        if let Some(rest) = s.strip_prefix(marker) {
            s = rest.trim_start();
        }
    }

    let first = s
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches([',', ';', ':']);
    if is_plausible_model_id(first) {
        Some(first.to_string())
    } else {
        None
    }
}

fn is_plausible_model_id(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() || s.len() > 160 {
        return false;
    }
    if s.contains("://") || s.contains(char::is_whitespace) {
        return false;
    }
    if !s.bytes().any(|b| b.is_ascii_alphanumeric()) {
        return false;
    }
    s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_' | b'/' | b':' | b'@'))
}

fn parse_cursor_status_authenticated(output: &str) -> Option<bool> {
    let value: Value = serde_json::from_str(output.trim()).ok()?;
    status_value_authenticated(&value)
}

fn status_value_authenticated(value: &Value) -> Option<bool> {
    match value {
        Value::Object(map) => {
            for key in [
                "authenticated",
                "isAuthenticated",
                "loggedIn",
                "logged_in",
                "signedIn",
                "signed_in",
            ] {
                if let Some(v) = map.get(key).and_then(Value::as_bool) {
                    return Some(v);
                }
            }
            for key in ["status", "auth", "login"] {
                if let Some(s) = map.get(key).and_then(Value::as_str) {
                    let lower = s.to_ascii_lowercase();
                    if [
                        "authenticated",
                        "logged_in",
                        "logged-in",
                        "logged in",
                        "signed_in",
                    ]
                    .contains(&lower.as_str())
                    {
                        return Some(true);
                    }
                    if ["unauthenticated", "logged_out", "logged-out", "logged out"]
                        .contains(&lower.as_str())
                    {
                        return Some(false);
                    }
                }
            }
            if map
                .get("user")
                .or_else(|| map.get("account"))
                .is_some_and(|v| !v.is_null())
            {
                return Some(true);
            }
            for child in map.values() {
                if let Some(v) = status_value_authenticated(child) {
                    return Some(v);
                }
            }
            None
        }
        _ => None,
    }
}

/// Reject image attachments before they reach `cursor-agent acp`.
///
/// Why: ACP defines an `image` content block, but as of writing
/// `cursor-agent`'s ACP server does not advertise the `image` prompt
/// capability — confirmed both by the spec
/// ([agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content),
/// which gates image blocks on `promptCapabilities.image`) and by the
/// `raphaelluethy/cursor-acp` adapter, which flattens images to text to
/// work around it. Bail with a clear message; once a session reports
/// `image: true` via [`PromptCapabilities`], callers can switch to
/// [`ensure_image_attachments_supported`] for the conditional path.
pub fn ensure_no_image_attachments(attachments: &[MessageAttachment]) -> Result<()> {
    if let Some(att) = first_image_attachment(attachments) {
        anyhow::bail!(
            "cursor: image attachments are not supported yet (attachment '{}', {})",
            att.name,
            att.mime_type
        );
    }
    Ok(())
}

/// Capability-conditional variant: only bails when the live session
/// advertises `promptCapabilities.image == false`. Use this once the caller
/// has an open [`CursorAcpSession`] and wants the behavior to follow what
/// cursor-agent actually reports.
pub fn ensure_image_attachments_supported(
    capabilities: &PromptCapabilities,
    attachments: &[MessageAttachment],
) -> Result<()> {
    if capabilities.image {
        return Ok(());
    }
    ensure_no_image_attachments(attachments)
}

fn first_image_attachment(attachments: &[MessageAttachment]) -> Option<&MessageAttachment> {
    attachments
        .iter()
        .find(|a| a.mime_type.starts_with("image/"))
}

/// Build the `session/prompt` content blocks for one user turn.
///
/// Image attachments become ACP `image` blocks (`{type, mimeType, data}`)
/// per [agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content);
/// the user text becomes a trailing `text` block. Non-image attachments are
/// dropped here — the chat layer only materializes image and text/document
/// attachments today, and document attachments aren't yet wired through
/// (they'd need either a `resource` block or a text-flatten path).
///
/// Caller is responsible for the capability check (see
/// [`ensure_image_attachments_supported`]) before sending.
pub fn build_prompt_blocks(text: &str, attachments: &[MessageAttachment]) -> Result<Vec<Value>> {
    let mut image_blocks = Vec::with_capacity(attachments.len());
    for att in attachments {
        if att.mime_type.starts_with("image/") {
            image_blocks.push(image_block_from_attachment(att)?);
        }
        // Non-image attachments are silently skipped for now; the chat layer
        // shouldn't be queuing types we don't handle yet, so this is mostly
        // defensive. Future work: text/document flattening + ACP `resource`
        // blocks once the agent advertises `embeddedContext: true`.
    }
    Ok(assemble_prompt_blocks(text, image_blocks))
}

/// Combine pre-shaped ACP image content blocks with the user text into one
/// `session/prompt` block list. Image blocks come first; the trailing text
/// block is omitted when the text is empty and at least one image block is
/// present, so an `/attach foo.png` + Enter with no draft text sends just
/// the image. A fully-empty submit still yields one empty text block so the
/// wire shape stays valid.
///
/// Router callers use this directly: they extract image blocks from the
/// inbound HTTP request (OpenAI/Anthropic/Responses/Gemini shapes) and
/// pair them with the reduced text prompt. Chat callers go through
/// [`build_prompt_blocks`] which does the [`MessageAttachment`] → image
/// block conversion first.
pub fn assemble_prompt_blocks(text: &str, mut image_blocks: Vec<Value>) -> Vec<Value> {
    if !text.is_empty() || image_blocks.is_empty() {
        image_blocks.push(text_block(text));
    }
    image_blocks
}

pub(crate) fn text_block(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Build an ACP `image` content block from an inline-encoded base64 string
/// plus a MIME type. Exposed `pub(crate)` for the router, which already
/// holds the decoded `(mime, data)` pair after parsing protocol-specific
/// image parts and doesn't have a `MessageAttachment` to hand.
pub(crate) fn image_block_from_inline(mime_type: &str, data_base64: &str) -> Value {
    json!({
        "type": "image",
        "mimeType": mime_type,
        "data": data_base64,
    })
}

fn image_block_from_attachment(att: &MessageAttachment) -> Result<Value> {
    let data = match &att.storage {
        AttachmentStorage::Inline { data } => data.as_str(),
        AttachmentStorage::FileRef { .. } => {
            anyhow::bail!(
                "cursor: image attachment '{}' was not materialized before send",
                att.name
            )
        }
    };
    Ok(image_block_from_inline(&att.mime_type, data))
}

/// Result of one ACP `session/prompt` round-trip.
#[derive(Debug, Default)]
pub struct CursorTurnResult {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub stop_reason: Option<String>,
    pub model: Option<String>,
}

/// Streaming chunk delivered to callers of [`run_cursor_acp_turn`]. Borrowed
/// to avoid allocating per delta on hot paths.
#[derive(Debug)]
pub enum CursorChunk<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

/// Live `cursor-agent acp` connection scoped to a single ACP session.
///
/// Reuses one spawned child for many prompts so the TUI doesn't pay Node.js
/// startup latency per turn. Dropping the session kills the child (via
/// `AcpClient`'s `kill_on_drop`), so callers can simply forget the value to
/// terminate cleanly.
pub struct CursorAcpSession {
    client: Arc<AcpClient>,
    session_id: String,
    model_id: Option<String>,
    models: Value,
    prompt_capabilities: PromptCapabilities,
}

/// Parsed `agentCapabilities.promptCapabilities` from the ACP `initialize`
/// response. Per spec ([agentclientprotocol.com/protocol/content](https://agentclientprotocol.com/protocol/content)),
/// these flags gate which content-block types the agent will accept in
/// `session/prompt`. cursor-agent's current ACP server appears to advertise
/// text-only — capturing the raw flags lets us flip image/audio guards from
/// blanket bails to capability-conditional once that changes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptCapabilities {
    pub image: bool,
    pub audio: bool,
    pub embedded_context: bool,
}

impl PromptCapabilities {
    pub fn from_init_response(init: &Value) -> Self {
        let prompt = init
            .get("agentCapabilities")
            .and_then(|c| c.get("promptCapabilities"));
        let flag = |name: &str| -> bool {
            prompt
                .and_then(|p| p.get(name))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };
        Self {
            image: flag("image"),
            audio: flag("audio"),
            embedded_context: flag("embeddedContext"),
        }
    }
}

impl CursorAcpSession {
    /// Spawn `cursor-agent acp`, run the initialize/authenticate/session/new
    /// handshake, and optionally lock in a starting model.
    pub async fn open(
        key: &ApiKey,
        requested_model: Option<&str>,
        workspace_cwd: &str,
    ) -> Result<Self> {
        ensure_cursor_agent_installed()?;
        let mut cmd = cursor_agent_command_for_key(key)?;
        cmd.arg("acp");

        let client = Arc::new(
            AcpClient::spawn_with_permission_policy(cmd, Arc::new(cursor_permission_decision))
                .await?,
        );

        let init = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": 1,
                    "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}},
                }),
            )
            .await
            .context("cursor-agent ACP initialize failed")?;

        // Skip the `authenticate` round-trip: cursor-agent is already
        // authed via CURSOR_API_KEY or on-disk shadow login. Calling it
        // would pop a macOS keychain-unlock dialog mid-chat; session/new
        // below surfaces a clean error if creds are actually missing.
        let _ = init.get("authMethods");

        let prompt_capabilities = PromptCapabilities::from_init_response(&init);
        if crate::services::http_debug::is_debug_active() {
            eprintln!(
                "aivo: cursor-agent acp promptCapabilities {{ image: {}, audio: {}, embeddedContext: {} }}",
                prompt_capabilities.image,
                prompt_capabilities.audio,
                prompt_capabilities.embedded_context
            );
        }

        let new_session = client
            .request(
                "session/new",
                json!({"cwd": workspace_cwd, "mcpServers": []}),
            )
            .await
            .with_context(|| {
                format!(
                    "cursor-agent ACP session/new failed — check the cursor key with `aivo keys reauth {0}`",
                    key.id
                )
            })?;

        let session_id = new_session
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("cursor-agent session/new omitted sessionId: {new_session}"))?
            .to_string();

        let models = new_session.get("models").cloned().unwrap_or(Value::Null);
        let active_model = pick_model_id_from_models(&models, requested_model);
        if let Some(model_id) = &active_model {
            let current = models.get("currentModelId").and_then(Value::as_str);
            if current != Some(model_id.as_str()) {
                let _ = client
                    .request(
                        "session/set_model",
                        json!({"sessionId": session_id, "modelId": model_id}),
                    )
                    .await;
            }
        }

        Ok(Self {
            client,
            session_id,
            model_id: active_model,
            models,
            prompt_capabilities,
        })
    }

    /// Capability flags from this session's ACP `initialize`. Useful for
    /// gating the image bail (see [`ensure_no_image_attachments`]) on what
    /// the agent actually advertises, instead of a blanket reject.
    pub fn prompt_capabilities(&self) -> &PromptCapabilities {
        &self.prompt_capabilities
    }

    pub fn supports_image_prompts(&self) -> bool {
        self.prompt_capabilities.image
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn model_id(&self) -> Option<&str> {
        self.model_id.as_deref()
    }

    pub fn client_handle(&self) -> Arc<AcpClient> {
        self.client.clone()
    }

    /// Switch the active model on the live session. `requested` accepts either
    /// the user-facing name (e.g. `composer-2.5`) or the encoded modelId
    /// (e.g. `composer-2.5[fast=true]`); falls back to passing `requested`
    /// through verbatim when the catalog has no matching entry.
    ///
    /// Short-circuits when the resolved id matches the current `model_id`.
    /// Without this, every router HTTP request would send an extra
    /// `session/set_model` round trip because the user-facing name from the
    /// request body (`composer-2.5`) never equals the stored encoded id
    /// (`composer-2.5[fast=true]`) under direct string comparison.
    pub async fn set_model(&mut self, requested: &str) -> Result<()> {
        let model_id = pick_model_id_from_models(&self.models, Some(requested))
            .unwrap_or_else(|| requested.to_string());
        if self.model_id.as_deref() == Some(model_id.as_str()) {
            return Ok(());
        }
        self.client
            .request(
                "session/set_model",
                json!({"sessionId": self.session_id, "modelId": model_id}),
            )
            .await
            .context("cursor-agent session/set_model failed")?;
        self.model_id = Some(model_id);
        Ok(())
    }

    /// Send a `session/prompt` and return a stream of session updates.
    pub async fn prompt(&self, text: &str) -> Result<PromptStream> {
        self.prompt_with_blocks(vec![text_block(text)]).await
    }

    /// Lower-level variant for callers that need to send multi-block prompts
    /// (image + text, etc.). The caller is responsible for honoring
    /// [`PromptCapabilities`] — see [`build_prompt_blocks`] for the
    /// canonical text+attachments → blocks conversion.
    pub async fn prompt_with_blocks(&self, blocks: Vec<Value>) -> Result<PromptStream> {
        self.client.start_prompt(&self.session_id, blocks).await
    }

    /// Best-effort `session/cancel` notification. The TUI fires this when the
    /// user aborts so the agent stops generating; the spawned task is also
    /// aborted independently, so a failure here is non-fatal.
    pub async fn cancel(&self) -> Result<()> {
        self.client
            .notify("session/cancel", json!({"sessionId": self.session_id}))
            .await
    }
}

/// Drive a single ACP turn against `cursor-agent`. Spawns the child, runs the
/// initialize/authenticate/session/new/session/prompt sequence, and streams
/// agent text/thought deltas through `on_chunk`. The child is killed when the
/// returned future resolves or is dropped.
pub async fn run_cursor_acp_turn<F>(
    key: &ApiKey,
    requested_model: Option<&str>,
    workspace_cwd: &str,
    prompt_text: &str,
    attachments: &[MessageAttachment],
    on_chunk: &mut F,
) -> Result<CursorTurnResult>
where
    F: FnMut(CursorChunk<'_>) -> Result<()>,
{
    let session = CursorAcpSession::open(key, requested_model, workspace_cwd).await?;
    ensure_image_attachments_supported(session.prompt_capabilities(), attachments)?;
    let blocks = build_prompt_blocks(prompt_text, attachments)?;
    let mut stream = session.prompt_with_blocks(blocks).await?;

    let mut out = CursorTurnResult {
        model: session.model_id().map(str::to_string),
        ..Default::default()
    };
    let mut reasoning_buf = String::new();
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                consume_session_update(&value, &mut out, &mut reasoning_buf, on_chunk)?;
            }
            PromptEvent::Done(result) => {
                let value = result
                    .map_err(|e| anyhow!(e))
                    .context("cursor-agent ACP session/prompt failed")?;
                out.stop_reason = value
                    .get("stopReason")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                break;
            }
        }
    }
    if !reasoning_buf.is_empty() {
        out.reasoning_content = Some(reasoning_buf);
    }
    Ok(out)
}

/// Folds one `session/update` value into the running turn result and emits a
/// chunk through `on_chunk` when the payload is plain text. Exposed
/// `pub(crate)` so the chat TUI can share the parsing rules.
pub(crate) fn consume_session_update<F>(
    value: &Value,
    out: &mut CursorTurnResult,
    reasoning_buf: &mut String,
    on_chunk: &mut F,
) -> Result<()>
where
    F: FnMut(CursorChunk<'_>) -> Result<()>,
{
    let Some(update) = value.get("update") else {
        return Ok(());
    };
    let kind = update.get("sessionUpdate").and_then(Value::as_str);
    let text = update
        .get("content")
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str);
    match (kind, text) {
        (Some("agent_message_chunk"), Some(t)) => {
            out.content.push_str(t);
            on_chunk(CursorChunk::Content(t))?;
        }
        (Some("agent_thought_chunk"), Some(t)) => {
            reasoning_buf.push_str(t);
            on_chunk(CursorChunk::Reasoning(t))?;
        }
        _ => {}
    }
    Ok(())
}

/// Look up the encoded modelId for `requested` against an ACP `models` object
/// (the `models` field of a `session/new` response). Match by user-facing
/// `name` (what `aivo models` shows). Falls back to `currentModelId`, then to
/// `None` if neither side surfaced a usable identifier.
fn pick_model_id_from_models(models: &Value, requested: Option<&str>) -> Option<String> {
    let current = models
        .get("currentModelId")
        .and_then(Value::as_str)
        .map(str::to_string);

    let Some(requested) = requested.filter(|s| !s.is_empty()) else {
        return current;
    };

    let available = models.get("availableModels").and_then(Value::as_array);
    if let Some(list) = available {
        for entry in list {
            let name = entry.get("name").and_then(Value::as_str);
            let id = entry.get("modelId").and_then(Value::as_str);
            if name == Some(requested) {
                return id.map(str::to_string);
            }
            if id == Some(requested) {
                return id.map(str::to_string);
            }
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeroize::Zeroizing;

    fn key(secret: &str) -> ApiKey {
        ApiKey {
            id: "abc".to_string(),
            name: "cursor".to_string(),
            base_url: CURSOR_ACP_SENTINEL.to_string(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            routing_schema_version: 0,
            key: Zeroizing::new(secret.to_string()),
            created_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    fn attachment(name: &str, mime: &str) -> MessageAttachment {
        MessageAttachment {
            name: name.to_string(),
            mime_type: mime.to_string(),
            storage: crate::services::session_store::AttachmentStorage::Inline {
                data: String::new(),
            },
        }
    }

    #[test]
    fn ensure_no_image_attachments_rejects_images_only() {
        assert!(ensure_no_image_attachments(&[]).is_ok());
        assert!(ensure_no_image_attachments(&[attachment("notes.txt", "text/plain")]).is_ok());
        assert!(ensure_no_image_attachments(&[attachment("doc.pdf", "application/pdf")]).is_ok());

        let err = ensure_no_image_attachments(&[attachment("shot.png", "image/png")])
            .expect_err("image rejection");
        let msg = format!("{err}");
        assert!(msg.contains("cursor"), "{msg}");
        assert!(msg.contains("shot.png"), "{msg}");
        assert!(msg.contains("image/png"), "{msg}");
    }

    #[test]
    fn prompt_capabilities_parses_flags_and_defaults_to_false() {
        let none = PromptCapabilities::from_init_response(&json!({}));
        assert_eq!(none, PromptCapabilities::default());

        let partial = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {"image": true}},
        }));
        assert!(partial.image);
        assert!(!partial.audio);
        assert!(!partial.embedded_context);

        let full = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {
                "image": true,
                "audio": true,
                "embeddedContext": true,
            }},
        }));
        assert!(full.image && full.audio && full.embedded_context);

        // Non-bool / missing fields fall back to false rather than panicking.
        let weird = PromptCapabilities::from_init_response(&json!({
            "agentCapabilities": {"promptCapabilities": {"image": "yes"}},
        }));
        assert!(!weird.image);
    }

    fn inline_attachment(name: &str, mime: &str, data: &str) -> MessageAttachment {
        MessageAttachment {
            name: name.to_string(),
            mime_type: mime.to_string(),
            storage: AttachmentStorage::Inline {
                data: data.to_string(),
            },
        }
    }

    #[test]
    fn build_prompt_blocks_emits_text_only_when_no_attachments() {
        let blocks = build_prompt_blocks("hi", &[]).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": "hi"})]);
    }

    #[test]
    fn build_prompt_blocks_image_only_send_omits_empty_text_block() {
        // /attach foo.png + Enter with no draft text: send the image alone,
        // no trailing empty text block. If cursor-agent turns out to require
        // a text peer for image blocks, flip this in `build_prompt_blocks`.
        let img = inline_attachment("a.png", "image/png", "AAA=");
        let blocks = build_prompt_blocks("", std::slice::from_ref(&img)).unwrap();
        assert_eq!(
            blocks,
            vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
        );
    }

    #[test]
    fn build_prompt_blocks_empty_input_with_no_attachments_yields_one_text_block() {
        // Pure empty submit should still produce a sendable block list.
        let blocks = build_prompt_blocks("", &[]).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": ""})]);
    }

    #[test]
    fn build_prompt_blocks_orders_images_before_text() {
        let img = inline_attachment("shot.png", "image/png", "QUJD");
        let blocks = build_prompt_blocks("describe", std::slice::from_ref(&img)).unwrap();
        assert_eq!(
            blocks,
            vec![
                json!({"type": "image", "mimeType": "image/png", "data": "QUJD"}),
                json!({"type": "text", "text": "describe"}),
            ]
        );
    }

    #[test]
    fn build_prompt_blocks_drops_unsupported_attachment_types() {
        let doc = inline_attachment("notes.pdf", "application/pdf", "JVBE");
        let blocks = build_prompt_blocks("read this", std::slice::from_ref(&doc)).unwrap();
        assert_eq!(blocks, vec![json!({"type": "text", "text": "read this"})]);
    }

    #[test]
    fn build_prompt_blocks_rejects_unmaterialized_images() {
        let img = MessageAttachment {
            name: "x.png".into(),
            mime_type: "image/png".into(),
            storage: AttachmentStorage::FileRef {
                path: "/tmp/x.png".into(),
            },
        };
        let err = build_prompt_blocks("hi", std::slice::from_ref(&img)).unwrap_err();
        assert!(format!("{err}").contains("not materialized"));
    }

    #[test]
    fn ensure_image_attachments_supported_follows_capability_flag() {
        let img = attachment("shot.png", "image/png");
        let caps_off = PromptCapabilities::default();
        let caps_on = PromptCapabilities {
            image: true,
            ..PromptCapabilities::default()
        };

        assert!(ensure_image_attachments_supported(&caps_off, std::slice::from_ref(&img)).is_err());
        assert!(ensure_image_attachments_supported(&caps_on, std::slice::from_ref(&img)).is_ok());
        // No attachments => always ok.
        assert!(ensure_image_attachments_supported(&caps_off, &[]).is_ok());
    }

    #[test]
    fn permission_decision_defaults_to_reject_and_flips_with_env() {
        // Serialize against other env-var tests in the binary by saving the
        // prior value (if any) and restoring it on exit.
        let prior = std::env::var(CURSOR_ALLOW_TOOLS_ENV).ok();
        // SAFETY: tests run single-threaded by default for this binary; the
        // restore in the guard below keeps state consistent for any parallel
        // peers that may read this env var afterwards.
        unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) };
        assert_eq!(
            cursor_permission_decision(&Value::Null),
            PermissionDecision::Reject
        );
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "1") };
        assert_eq!(
            cursor_permission_decision(&Value::Null),
            PermissionDecision::Allow
        );
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "true") };
        assert_eq!(
            cursor_permission_decision(&Value::Null),
            PermissionDecision::Allow
        );
        unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, "no") };
        assert_eq!(
            cursor_permission_decision(&Value::Null),
            PermissionDecision::Reject
        );
        match prior {
            Some(v) => unsafe { std::env::set_var(CURSOR_ALLOW_TOOLS_ENV, v) },
            None => unsafe { std::env::remove_var(CURSOR_ALLOW_TOOLS_ENV) },
        }
    }

    #[test]
    fn sentinel_detection_is_exact() {
        assert!(is_cursor_acp_base("cursor"));
        assert!(!is_cursor_acp_base(" cursor"));
        assert!(!is_cursor_acp_base("cursor/"));
        assert!(!is_cursor_acp_base("https://cursor.com"));
    }

    #[test]
    fn auth_env_skips_oauth_shadows_extracts_apikey_shadows_and_falls_back_for_raw_keys() {
        let oauth = key(&build_cursor_oauth_secret("abcd1234"));
        assert!(
            cursor_auth_env(&oauth).is_none(),
            "OAuth shadow keys must not leak CURSOR_API_KEY"
        );

        let api_shadow = key(&build_cursor_apikey_secret("abcd1234", "key_real_value"));
        let (name, value) = cursor_auth_env(&api_shadow).unwrap();
        assert_eq!(name, "CURSOR_API_KEY");
        assert_eq!(value, OsString::from("key_real_value"));

        let raw = key("sk-cursor");
        let (name, value) = cursor_auth_env(&raw).unwrap();
        assert_eq!(name, "CURSOR_API_KEY");
        assert_eq!(value, OsString::from("sk-cursor"));

        assert!(cursor_auth_env(&key(LEGACY_CURSOR_LOGIN_SENTINEL)).is_none());
    }

    #[test]
    fn parse_cursor_shadow_secret_round_trips() {
        let oauth = build_cursor_oauth_secret("abcd1234");
        let parsed = parse_cursor_shadow_secret(&oauth).unwrap();
        assert_eq!(parsed.account_id, "abcd1234");
        assert_eq!(parsed.api_key, None);

        let with_key = build_cursor_apikey_secret("abcd1234", "key_aabb");
        let parsed = parse_cursor_shadow_secret(&with_key).unwrap();
        assert_eq!(parsed.account_id, "abcd1234");
        assert_eq!(parsed.api_key, Some("key_aabb"));

        // Unknown extension keyword → reject rather than silently misroute
        // the secret. Defends against future format drift.
        assert!(parse_cursor_shadow_secret("cursor-shadow:abc:future:val").is_none());
        assert!(parse_cursor_shadow_secret("cursor-shadow:").is_none());
        assert!(parse_cursor_shadow_secret("sk-cursor").is_none());
    }

    #[test]
    fn cache_identity_separates_accounts_and_api_keys() {
        let a = cursor_models_cache_identity(&key(&format!("{CURSOR_SHADOW_PREFIX}aaaa1111")));
        let b = cursor_models_cache_identity(&key(&format!("{CURSOR_SHADOW_PREFIX}bbbb2222")));
        assert!(a.starts_with("cursor#shadow-"));
        assert_ne!(a, b);

        let x = cursor_models_cache_identity(&key("sk-x"));
        let y = cursor_models_cache_identity(&key("sk-y"));
        assert!(x.starts_with("cursor#"));
        assert!(!x.starts_with("cursor#shadow-"));
        assert_ne!(x, y);
        assert!(!x.contains("sk-x"));
    }

    #[test]
    fn cursor_account_id_round_trip() {
        let shadow = key(&format!("{CURSOR_SHADOW_PREFIX}abcd1234"));
        assert_eq!(cursor_account_id(&shadow), Some("abcd1234"));
        assert_eq!(cursor_account_id(&key("sk-cursor")), None);
    }

    #[test]
    fn parses_json_array_forms() {
        let ids = parse_cursor_models(r#"["composer-2.5","gpt-5"]"#).unwrap();
        assert_eq!(ids, vec!["composer-2.5", "gpt-5"]);
    }

    #[test]
    fn parses_json_object_forms() {
        let ids = parse_cursor_models(
            r#"{"models":[{"id":"composer-2.5"},{"model":"claude-sonnet-4.6"},{"name":"gpt-5"}]}"#,
        )
        .unwrap();
        assert_eq!(ids, vec!["claude-sonnet-4.6", "composer-2.5", "gpt-5"]);
    }

    #[test]
    fn parses_plain_line_output() {
        let ids = parse_cursor_models("composer-2.5\ngpt-5\n").unwrap();
        assert_eq!(ids, vec!["composer-2.5", "gpt-5"]);
    }

    #[test]
    fn rejects_no_models_prose_when_logged_out() {
        let ids = parse_cursor_models("No models available for this account.\n").unwrap();
        assert!(
            ids.is_empty(),
            "prose-style sentences must not parse as model ids, got {ids:?}"
        );

        assert!(looks_like_no_models_message(
            "No models available for this account."
        ));
        assert!(looks_like_no_models_message("Not logged in. Run login."));
        assert!(!looks_like_no_models_message("composer-2.5\ngpt-5"));
    }

    #[test]
    fn parses_headings_bullets_current_markers_and_blanks() {
        let ids = parse_cursor_models(
            "\nAvailable models:\n  * composer-2.5 (current)\n  - claude-sonnet-4.6\n  > gpt-5\n  default: o3\n",
        )
        .unwrap();
        assert_eq!(
            ids,
            vec!["claude-sonnet-4.6", "composer-2.5", "gpt-5", "o3"]
        );
    }

    #[test]
    fn status_parser_accepts_common_shapes() {
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"authenticated":true}"#),
            Some(true)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"auth":{"loggedIn":false}}"#),
            Some(false)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"status":"logged in"}"#),
            Some(true)
        );
        assert_eq!(
            parse_cursor_status_authenticated(r#"{"user":{"email":"a@example.com"}}"#),
            Some(true)
        );
    }

    #[test]
    fn pick_model_id_matches_user_facing_name() {
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
                {"modelId": "claude-sonnet-4-6[thinking=true]", "name": "claude-sonnet-4-6"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(&models, Some("claude-sonnet-4-6")).as_deref(),
            Some("claude-sonnet-4-6[thinking=true]")
        );
    }

    #[test]
    fn pick_model_id_accepts_encoded_id_directly() {
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
            ],
        });
        assert_eq!(
            pick_model_id_from_models(&models, Some("composer-2.5[fast=true]")).as_deref(),
            Some("composer-2.5[fast=true]")
        );
    }

    #[test]
    fn pick_model_id_falls_back_to_current_when_no_match() {
        let models = serde_json::json!({
            "currentModelId": "composer-2.5[fast=true]",
            "availableModels": [
                {"modelId": "composer-2.5[fast=true]", "name": "composer-2.5"},
            ],
        });
        // Unknown name → use current; null request → use current.
        assert_eq!(
            pick_model_id_from_models(&models, Some("totally-bogus")).as_deref(),
            Some("composer-2.5[fast=true]")
        );
        assert_eq!(
            pick_model_id_from_models(&models, None).as_deref(),
            Some("composer-2.5[fast=true]")
        );
    }

    #[test]
    fn consume_session_update_splits_content_and_thought() {
        let mut out = CursorTurnResult::default();
        let mut reasoning = String::new();
        let mut chunks: Vec<(String, String)> = Vec::new();
        let mut on_chunk = |chunk: CursorChunk<'_>| -> Result<()> {
            match chunk {
                CursorChunk::Content(t) => chunks.push(("content".into(), t.to_string())),
                CursorChunk::Reasoning(t) => chunks.push(("reasoning".into(), t.to_string())),
            }
            Ok(())
        };

        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "hmm "}},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();
        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hello"}},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();
        // Unknown update kinds (tool_call, plan, available_commands_update) are dropped.
        let msg = serde_json::json!({
            "sessionId": "s1",
            "update": {"sessionUpdate": "available_commands_update", "commands": []},
        });
        consume_session_update(&msg, &mut out, &mut reasoning, &mut on_chunk).unwrap();

        assert_eq!(out.content, "Hello");
        assert_eq!(reasoning, "hmm ");
        assert_eq!(
            chunks,
            vec![
                ("reasoning".to_string(), "hmm ".to_string()),
                ("content".to_string(), "Hello".to_string()),
            ]
        );
    }
}
