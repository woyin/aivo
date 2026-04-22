/**
 * KeysCommand handler for managing API keys.
 */
use anyhow::Result;
use serde_json::{Value, json};

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::cli::KeysArgs;
use crate::commands::keys_ui;
use crate::commands::truncate_url_for_display;
use crate::tui::FuzzySelect;

use crate::errors::ExitCode;
use crate::services::provider_profile::is_aivo_starter_base;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

#[allow(clippy::large_enum_variant)]
enum KeySelection {
    Key(ApiKey),
    Cancelled,
    Empty,
    NotFound,
}

// Force cooked (canonical + echo) mode on stdin. The `console` crate's FuzzySelect
// can leave termios flags off on macOS, and a previously-crashed `aivo` invocation
// may have left the terminal corrupted. `crossterm::disable_raw_mode` is a no-op
// when crossterm didn't enable raw mode itself, so we shell out to `stty sane`.
// Call this at entry of an interactive flow and after any FuzzySelect exits.
#[cfg(unix)]
fn restore_cooked_mode() {
    let _ = std::process::Command::new("stty")
        .arg("sane")
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn restore_cooked_mode() {
    let _ = crossterm::terminal::disable_raw_mode();
}

// Reads a line from stdin, flushing the prompt first so it appears before blocking.
fn term_read_line(prompt: &str) -> std::io::Result<String> {
    use std::io::{BufRead, Write};
    print!("{}", prompt);
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().lock().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

// Reads a line from stdin with masked echo (prints '*' per character) for secrets.
fn term_read_secret(prompt: &str) -> std::io::Result<String> {
    use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
    use crossterm::terminal;
    use std::io::Write;

    print!("{}", prompt);
    std::io::stdout().flush()?;

    terminal::enable_raw_mode()?;
    let mut input = String::new();
    let mut stdout = std::io::stdout();
    let result = loop {
        match event::read() {
            Ok(Event::Key(KeyEvent {
                code, modifiers, ..
            })) => match code {
                KeyCode::Enter => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Ok(input);
                }
                KeyCode::Backspace if !input.is_empty() => {
                    input.pop();
                    let _ = write!(stdout, "\x08 \x08");
                    let _ = stdout.flush();
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = write!(stdout, "\r\n");
                    let _ = stdout.flush();
                    break Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "interrupted",
                    ));
                }
                KeyCode::Char(c) => {
                    input.push(c);
                    let _ = write!(stdout, "*");
                    let _ = stdout.flush();
                }
                _ => {}
            },
            Ok(_) => {}
            Err(e) => break Err(e),
        }
    };
    let _ = terminal::disable_raw_mode();
    result
}

// Reads a confirmation from stdin (y/yes for true, anything else for false).
fn confirm(prompt: &str) -> std::io::Result<bool> {
    let input = term_read_line(&format!("{} [y/N]: ", prompt))?;
    Ok(matches!(input.to_ascii_lowercase().as_str(), "y" | "yes"))
}

// Creates a safe preview of an API key, handling short keys without panicking.
fn key_preview(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    if chars.len() <= 10 {
        let prefix: String = chars.iter().take(3).collect();
        format!("{prefix}...")
    } else {
        let prefix: String = chars.iter().take(6).collect();
        let suffix: String = chars.iter().skip(chars.len() - 4).collect();
        format!("{prefix}...{suffix}")
    }
}

fn display_secret(key: &ApiKey) -> String {
    key.credential_label()
        .map(str::to_string)
        .unwrap_or_else(|| key_preview(&key.key))
}

pub struct KeysCommand {
    session_store: SessionStore,
}

#[derive(Clone, Copy, Debug, Default)]
struct AddKeyOptions<'a> {
    name: Option<&'a str>,
    base_url: Option<&'a str>,
    key: Option<&'a str>,
}

/// Outcome of prompting the user about a conflicting existing key.
enum ReplaceDecision {
    NoExisting,
    Replace(String),
    Abort,
}

fn print_ping_result(result: &PingResult, max_name_len: usize) {
    let icon = match &result.status {
        PingStatus::Ok => style::green(result.status.icon()),
        _ => style::red(result.status.icon()),
    };
    let latency = result
        .latency
        .map(|d| format!("{}ms", d.as_millis()))
        .unwrap_or_default();
    let name_padded = format!("{:<width$}", result.name, width = max_name_len);
    let message = result.status.message();
    println!(
        " {} {}  {}  {:>6}  {}",
        icon,
        name_padded,
        style::dim(truncate_url_for_display(&result.url, 40)),
        style::dim(latency),
        match &result.status {
            PingStatus::Ok => style::green(&message),
            _ => style::red(&message),
        }
    );
}

fn detect_base_url(name: &str) -> Option<&str> {
    crate::services::known_providers::find_by_name_substring(name).map(|p| p.base_url.as_str())
}

/// Placeholder used in `providers.json` for Cloudflare Workers AI account ID.
const CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER: &str = "${CLOUDFLARE_ACCOUNT_ID}";

/// Validates a user-supplied base URL. Returns `Err` with a user-facing message
/// if the URL is malformed. Accepts `http(s)://host[/path]` with no whitespace.
fn validate_base_url(url: &str) -> Result<(), &'static str> {
    let rest = if let Some(r) = url.strip_prefix("https://") {
        r
    } else if let Some(r) = url.strip_prefix("http://") {
        r
    } else {
        return Err("URL must start with http:// or https://");
    };
    if url.chars().any(char::is_whitespace) {
        return Err("URL must not contain whitespace");
    }
    let host = rest.split('/').next().unwrap_or("");
    if host.is_empty() {
        return Err("URL must include a host");
    }
    Ok(())
}
const CLOUDFLARE_PROVIDER_ID: &str = "cloudflare-workers-ai";
const PICKER_LABEL_WIDTH: usize = 36;
const PICKER_URL_MAX_LEN: usize = 50;

/// Provider info lines shown after picker selection / during shortcut flows.
/// Defined once so the interactive and shortcut paths stay in sync.
const COPILOT_INFO: (&str, &str) = ("GitHub Copilot", "device login: github.com/login/device");
const CODEX_OAUTH_INFO: (&str, &str) = (
    "OpenAI Codex (ChatGPT)",
    "sign in to your ChatGPT account — multiple accounts supported",
);
const CLAUDE_OAUTH_INFO: (&str, &str) = (
    "Claude Code (Anthropic)",
    "sign in to your Anthropic account — multiple accounts supported",
);
const GEMINI_OAUTH_INFO: (&str, &str) = (
    "Gemini (Google)",
    "sign in with Google — multiple accounts supported",
);
const OLLAMA_INFO: (&str, &str) = ("Ollama", "install: ollama.com/download");
const STARTER_INFO: (&str, &str) = ("aivo starter", "free shared key, no signup needed");

fn format_picker_choice(label: &str, hint: &str) -> String {
    format!(
        "{:<width$} {}",
        label,
        style::dim(hint),
        width = PICKER_LABEL_WIDTH
    )
}

/// Prompts for a secret. If the user enters nothing, asks whether to save
/// without one; loops until a value is provided or confirmation is given.
fn prompt_secret(prompt: &str, secret_noun: &str) -> std::io::Result<String> {
    loop {
        let input = term_read_secret(&style::dim(prompt))?;
        if !input.is_empty() {
            return Ok(input);
        }
        let confirm_prompt = style::yellow(format!("Save without {}?", secret_noun));
        if confirm(&confirm_prompt)? {
            return Ok(String::new());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PingStatus {
    Ok,
    AuthError,
    Unreachable,
    Timeout,
    Error(String),
}

impl PingStatus {
    pub fn icon(&self) -> &'static str {
        match self {
            PingStatus::Ok => "✓",
            PingStatus::AuthError => "✗",
            PingStatus::Unreachable => "✗",
            PingStatus::Timeout => "✗",
            PingStatus::Error(_) => "✗",
        }
    }

    pub fn message(&self) -> String {
        match self {
            PingStatus::Ok => "ok".to_string(),
            PingStatus::AuthError => "auth failed".to_string(),
            PingStatus::Unreachable => "unreachable".to_string(),
            PingStatus::Timeout => "timeout".to_string(),
            PingStatus::Error(msg) => msg.clone(),
        }
    }

    /// Machine-readable identifier used in structured output.
    pub fn json_key(&self) -> &'static str {
        match self {
            PingStatus::Ok => "ok",
            PingStatus::AuthError => "auth_error",
            PingStatus::Unreachable => "unreachable",
            PingStatus::Timeout => "timeout",
            PingStatus::Error(_) => "error",
        }
    }

    pub fn from_http_status(status: u16) -> Self {
        match status {
            200..=299 => PingStatus::Ok,
            401 | 403 => PingStatus::AuthError,
            // 404/405 on probe endpoints means reachable but wrong path — still ok for ping
            404 | 405 => PingStatus::Ok,
            _ => PingStatus::Error(format!("HTTP {}", status)),
        }
    }
}

#[derive(Debug)]
pub struct PingResult {
    pub name: String,
    pub url: String,
    pub status: PingStatus,
    pub latency: Option<Duration>,
}

/// Builds a JSON metadata object for a key. Never includes the secret.
pub(crate) fn key_metadata_json(key: &ApiKey, selected_id: Option<&str>) -> Value {
    json!({
        "id": key.id,
        "name": key.name,
        "base_url": key.base_url,
        "active": selected_id == Some(key.id.as_str()),
        "created_at": key.created_at,
    })
}

/// Converts a PingResult into a JSON object for structured output.
pub(crate) fn ping_result_json(result: &PingResult) -> Value {
    json!({
        "ok": matches!(result.status, PingStatus::Ok),
        "status": result.status.json_key(),
        "message": result.status.message(),
        "latency_ms": result.latency.map(|d| d.as_millis() as u64),
    })
}

const PING_TIMEOUT: Duration = Duration::from_secs(5);

const PING_MAX_RETRIES: u32 = 3;

pub async fn ping_key(key: ApiKey) -> PingResult {
    let name = if key.name.is_empty() {
        key.short_id().to_string()
    } else {
        key.name.clone()
    };
    let url = key.base_url.clone();

    let start = Instant::now();
    let status = ping_with_retries(&key).await;
    let latency = Some(start.elapsed());

    PingResult {
        name,
        url,
        status,
        latency,
    }
}

/// Pings keys concurrently and calls `on_result(key_id, result)` for each as it resolves.
/// Decrypt failures are reported immediately; successful decrypts are spawned and awaited in order.
pub async fn ping_keys_streaming(keys: Vec<ApiKey>, mut on_result: impl FnMut(&str, &PingResult)) {
    let mut handles = Vec::new();
    for mut key in keys {
        let id = key.id.clone();
        if SessionStore::decrypt_key_secret(&mut key).is_err() {
            on_result(
                &id,
                &PingResult {
                    name: key.display_name().to_string(),
                    url: key.base_url.clone(),
                    status: PingStatus::Error("decrypt failed".to_string()),
                    latency: None,
                },
            );
            continue;
        }
        handles.push(tokio::spawn(async move {
            let result = ping_key(key).await;
            (id, result)
        }));
    }
    for handle in handles {
        if let Ok((id, result)) = handle.await {
            on_result(&id, &result);
        }
    }
}

async fn ping_with_retries(key: &ApiKey) -> PingStatus {
    let mut last_status = PingStatus::Unreachable;
    for _ in 0..PING_MAX_RETRIES {
        last_status = match tokio::time::timeout(PING_TIMEOUT, probe_key(key)).await {
            Ok(Ok(s)) => s,
            Ok(Err(_)) => PingStatus::Unreachable,
            Err(_) => PingStatus::Timeout,
        };
        if matches!(last_status, PingStatus::Ok) {
            return last_status;
        }
    }
    last_status
}

async fn probe_key(key: &ApiKey) -> Result<PingStatus> {
    use crate::services::provider_profile::{ModelListingStrategy, provider_profile_for_key};

    // OAuth keys have no reachable REST endpoint; tokens are consumed by the
    // native CLI against the provider's subscription backend. Report "Ok" so
    // the list doesn't look scary; the real health check is `aivo run <tool>`.
    if key.is_any_oauth() {
        return Ok(PingStatus::Ok);
    }

    let profile = provider_profile_for_key(key);
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(PING_TIMEOUT)
        .build()?;

    match profile.model_listing_strategy {
        ModelListingStrategy::Ollama => match client.get("http://localhost:11434/").send().await {
            Ok(_) => Ok(PingStatus::Ok),
            Err(_) => Ok(PingStatus::Unreachable),
        },
        ModelListingStrategy::Copilot => {
            use crate::services::copilot_auth::CopilotTokenManager;
            let tm = CopilotTokenManager::new(key.key.as_str().to_string());
            match tm.get_token().await {
                Ok(_) => Ok(PingStatus::Ok),
                Err(_) => Ok(PingStatus::AuthError),
            }
        }
        ModelListingStrategy::Google => {
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models?key={}",
                key.key.as_str()
            );
            match client.get(&url).send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::Anthropic | ModelListingStrategy::Static(_) => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            match client
                .get(&url)
                .header("x-api-key", key.key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send()
                .await
            {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::AivoStarter => {
            let url = format!(
                "{}/v1/models",
                crate::constants::AIVO_STARTER_REAL_URL.trim_end_matches('/')
            );
            let req = crate::services::device_fingerprint::with_starter_headers(
                client
                    .get(&url)
                    .header("Authorization", format!("Bearer {}", key.key.as_str())),
            );
            match req.send().await {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
        ModelListingStrategy::CloudflareSearch | ModelListingStrategy::OpenAiCompatible => {
            let base = key.base_url.trim_end_matches('/');
            let url = if base.ends_with("/v1") {
                format!("{}/models", base)
            } else {
                format!("{}/v1/models", base)
            };
            match client
                .get(&url)
                .header("Authorization", format!("Bearer {}", key.key.as_str()))
                .send()
                .await
            {
                Ok(r) => Ok(PingStatus::from_http_status(r.status().as_u16())),
                Err(_) => Ok(PingStatus::Unreachable),
            }
        }
    }
}

/// Warms the model cache for a newly added key so subsequent commands are instant.
fn sync_models_in_background(key_id: &str, base_url: &str) {
    if crate::services::provider_profile::is_ollama_base(base_url) {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return,
    };
    let _ = std::process::Command::new(exe)
        .args(["models", "--refresh", "--key", key_id])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

impl KeysCommand {
    /// Creates a new KeysCommand instance
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    /// Returns the active key ID from last_selection or active_key_id.
    async fn selected_key_id(&self) -> Option<String> {
        if let Some(id) = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|s| s.key_id)
        {
            return Some(id);
        }
        self.session_store
            .get_active_key_info()
            .await
            .ok()
            .flatten()
            .map(|k| k.id)
    }

    /// Executes the keys command with the specified action
    pub async fn execute(&self, keys_args: KeysArgs) -> ExitCode {
        let action = keys_args.action.as_deref();
        let args: Vec<_> = keys_args.args.iter().map(|s| s.as_str()).collect();
        let add_options = AddKeyOptions {
            name: keys_args.name.as_deref(),
            base_url: keys_args.base_url.as_deref(),
            key: keys_args.key.as_deref(),
        };
        let ping_all = keys_args.all;
        let list_ping = keys_args.ping;
        let json = keys_args.json;

        match self
            .execute_internal(action, Some(&args), add_options, ping_all, list_ping, json)
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(
        &self,
        action: Option<&str>,
        args: Option<&[&str]>,
        add_options: AddKeyOptions<'_>,
        ping_all: bool,
        list_ping: bool,
        json: bool,
    ) -> Result<ExitCode> {
        match action {
            None if list_ping => self.list_keys_with_ping(json).await,
            None => self.list_keys(json).await,
            Some("add") => {
                self.add_key(args.and_then(|a| a.first().copied()), add_options)
                    .await
            }
            Some("rm") => self.remove_key(args.and_then(|a| a.first().copied())).await,
            Some("use") => self.use_key(args.and_then(|a| a.first().copied())).await,
            Some("cat") => self.cat_key(args.and_then(|a| a.first().copied())).await,
            Some("edit") => self.edit_key(args.and_then(|a| a.first().copied())).await,
            Some("ping") => {
                self.ping_keys(args.and_then(|a| a.first().copied()), ping_all)
                    .await
            }
            Some(action) => {
                eprintln!("{} Unknown action '{}'", style::red("Error:"), action);
                Self::print_help();
                Ok(ExitCode::UserError)
            }
        }
    }

    /// Health-checks API keys.
    async fn ping_keys(&self, key_id_or_name: Option<&str>, ping_all: bool) -> Result<ExitCode> {
        let keys: Vec<ApiKey> = if ping_all {
            let all_keys = self.session_store.get_keys().await?;
            if all_keys.len() > 1 {
                let confirmed = confirm(&format!("Ping all {} keys?", all_keys.len()))?;
                if !confirmed {
                    println!("{}", style::dim("Cancelled."));
                    return Ok(ExitCode::Success);
                }
            }
            all_keys
        } else if let Some(filter) = key_id_or_name {
            let all_keys = self.session_store.get_keys().await?;
            let matched: Vec<ApiKey> = all_keys
                .into_iter()
                .filter(|k| {
                    k.id == filter || k.short_id() == filter || k.name.eq_ignore_ascii_case(filter)
                })
                .collect();
            if matched.is_empty() {
                eprintln!("{} API key \"{}\" not found", style::red("Error:"), filter);
                return Ok(ExitCode::UserError);
            }
            matched
        } else {
            match self.session_store.get_active_key().await? {
                Some(key) => vec![key],
                None => {
                    eprintln!(
                        "{} No active API key. Run 'aivo keys use' first.",
                        style::red("Error:")
                    );
                    return Ok(ExitCode::UserError);
                }
            }
        };

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys
            .iter()
            .map(|k| k.display_name().len())
            .max()
            .unwrap_or(0);

        ping_keys_streaming(keys, |_id, result| {
            print_ping_result(result, max_name_len);
        })
        .await;

        Ok(ExitCode::Success)
    }

    /// Lists all API keys
    async fn list_keys(&self, json: bool) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let selected_key_id = self.selected_key_id().await;

        if json {
            let payload: Vec<Value> = keys
                .iter()
                .map(|k| key_metadata_json(k, selected_key_id.as_deref()))
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys.iter().map(|k| k.name.len()).max().unwrap_or(0);

        for key in &keys {
            let is_selected = selected_key_id.as_deref() == Some(key.id.as_str());
            let active_indicator = if is_selected {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<3}", key.short_id());
            let name_padded = format!("{:<width$}", key.name, width = max_name_len);
            let is_starter = is_aivo_starter_base(&key.base_url);
            let name_col = if is_starter {
                style::magenta(&name_padded)
            } else {
                name_padded
            };
            println!(
                "{} {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                name_col,
                style::dim(truncate_url_for_display(&key.base_url, 50))
            );
        }

        Ok(ExitCode::Success)
    }

    /// Lists all API keys with live ping status, streaming results as they complete.
    async fn list_keys_with_ping(&self, json: bool) -> Result<ExitCode> {
        let keys = self.session_store.get_keys().await?;
        let selected_key_id = self.selected_key_id().await;

        if json {
            let mut ping_by_id: HashMap<String, Value> = HashMap::new();
            let spinner =
                (!keys.is_empty()).then(|| style::start_spinner(Some(" Pinging keys...")));
            ping_keys_streaming(keys.clone(), |id, result| {
                ping_by_id.insert(id.to_string(), ping_result_json(result));
            })
            .await;
            if let Some((spinning, handle)) = spinner {
                style::stop_spinner(&spinning);
                let _ = handle.await;
            }
            let payload: Vec<Value> = keys
                .iter()
                .map(|k| {
                    let mut obj = key_metadata_json(k, selected_key_id.as_deref());
                    if let Some(p) = ping_by_id.remove(&k.id) {
                        obj["ping"] = p;
                    }
                    obj
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if keys.is_empty() {
            println!("{}", style::dim("No API keys found."));
            return Ok(ExitCode::Success);
        }

        let max_name_len = keys.iter().map(|k| k.name.len()).max().unwrap_or(0);
        let url_display_width = 42;

        for mut key in keys {
            let is_selected = selected_key_id.as_deref() == Some(key.id.as_str());
            let active_indicator = if is_selected {
                style::bullet_symbol()
            } else {
                style::empty_bullet_symbol()
            };
            let id_padded = format!("{:<3}", key.short_id());
            let name_padded = format!("{:<width$}", key.name, width = max_name_len);
            let is_starter = is_aivo_starter_base(&key.base_url);
            let name_col = if is_starter {
                style::magenta(&name_padded)
            } else {
                name_padded
            };

            let ping_status = if SessionStore::decrypt_key_secret(&mut key).is_ok() {
                let start = Instant::now();
                let status = ping_with_retries(&key).await;
                let ms = start.elapsed().as_millis();
                let icon = match &status {
                    PingStatus::Ok => style::green(status.icon()),
                    _ => style::red(status.icon()),
                };
                let msg = match &status {
                    PingStatus::Ok => style::green(status.message()),
                    _ => style::red(status.message()),
                };
                format!("{} {:>5}ms {}", icon, ms, msg)
            } else {
                format!("{} {}", style::red("✗"), style::red("decrypt failed"))
            };

            let url_display = truncate_url_for_display(&key.base_url, url_display_width);
            let url_padded = format!("{:<width$}", url_display, width = url_display_width);
            println!(
                "{} {}  {}  {}  {}",
                active_indicator,
                style::cyan(&id_padded),
                name_col,
                style::dim(&url_padded),
                ping_status
            );
        }

        Ok(ExitCode::Success)
    }

    /// Activates a specific API key by ID or name
    async fn use_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to activate",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                self.activate_key(&key).await?;
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Activates a key and prints confirmation
    async fn activate_key(&self, key: &ApiKey) -> Result<()> {
        self.session_store.set_active_key(&key.id).await?;
        // Update global last-selection: keep existing tool, clear model since
        // the new key may have a different provider/model set.
        if let Some(existing_tool) = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|s| s.tool)
        {
            let _ = self
                .session_store
                .set_last_selection(key, &existing_tool, None)
                .await;
        }
        let preview = display_secret(key);
        println!(
            "{} Activated key: {} {}",
            style::success_symbol(),
            style::cyan(key.display_name()),
            style::dim(&preview)
        );
        Ok(())
    }

    /// Activates a newly added key and prints the post-add status.
    async fn finalize_add(
        &self,
        id: &str,
        name: &str,
        detail: &str,
        next_hint: Option<(&str, &str)>,
    ) -> Result<()> {
        self.session_store.set_active_key(id).await?;
        // Clear last_selection so key resolution picks the new active key
        // instead of a stale last_selection (e.g. aivo-starter).
        let _ = self.session_store.clear_last_selection().await;

        let display_name = style::cyan(if name.is_empty() { id } else { name });
        println!();
        println!(
            "{} Added and activated key: {}",
            style::success_symbol(),
            display_name
        );
        println!("  {}", style::dim(format!("ID: {}", id)));
        println!("  {}", style::dim(detail));
        println!();
        if let Some((cmd, desc)) = next_hint {
            if desc.is_empty() {
                println!("{} {}", style::yellow("Next:"), style::bold(cmd));
            } else {
                println!(
                    "{} {} {}",
                    style::yellow("Next:"),
                    style::bold(cmd),
                    style::dim(desc)
                );
            }
        }
        Ok(())
    }

    /// Displays details for a specific API key
    async fn cat_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to inspect",
                "No API keys found.",
            )
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                self.display_key_details(&key);
                Ok(ExitCode::Success)
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                Ok(ExitCode::Success)
            }
            KeySelection::Empty => Ok(ExitCode::Success),
            KeySelection::NotFound => Ok(ExitCode::UserError),
        }
    }

    /// Displays key details
    fn display_key_details(&self, key: &ApiKey) {
        println!("Name:     {}", style::cyan(key.display_name()));
        println!("Base URL: {}", style::blue(&key.base_url));
        if key.key.is_empty() {
            println!("API Key:  {}", style::dim("(none)"));
        } else if let Some(label) = key.credential_label() {
            // Dumping OAuth bundles / Copilot device tokens would leak live
            // access/refresh tokens that the user can't re-enter anywhere.
            println!("API Key:  {}", style::dim(label));
        } else {
            println!("API Key:  {}", style::yellow(&*key.key));
        }
    }

    /// Interactively edits an API key
    async fn edit_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        use std::io;

        let key = match self
            .resolve_key_selection(key_id_or_name, "Select a key to edit", "No API keys found.")
            .await?
        {
            KeySelection::Key(mut key) => {
                SessionStore::decrypt_key_secret(&mut key)?;
                key
            }
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        println!("{}", style::bold("Edit API Key"));
        println!();
        println!("Press Enter to keep the current value.");
        println!();

        fn read_line_with_default(prompt: &str) -> io::Result<String> {
            term_read_line(prompt)
        }

        // OAuth entries hold a serialized credential bundle in the encrypted
        // key slot — there is no meaningful base URL or user-editable "API
        // key" to change. Only the display name is safe to edit; everything
        // else is preserved verbatim.
        if key.is_any_oauth() {
            let current_name = if key.name.is_empty() {
                format!("unnamed; shown as {}", key.short_id())
            } else {
                key.name.clone()
            };
            let name = {
                let input = read_line_with_default(&format!("Name [{}]: ", current_name))?;
                if input.is_empty() {
                    key.name.clone()
                } else {
                    input
                }
            };

            if name == key.name {
                println!("{}", style::dim("No changes."));
                return Ok(ExitCode::Success);
            }

            let updated = self
                .session_store
                .update_key(
                    &key.id,
                    &name,
                    &key.base_url,
                    key.claude_protocol,
                    key.key.as_str(),
                )
                .await?;
            if updated {
                println!(
                    "  {} renamed to {}",
                    style::success_symbol(),
                    style::cyan(&name)
                );
            }
            return Ok(ExitCode::Success);
        }

        // Name
        let current_name = if key.name.is_empty() {
            format!("unnamed; shown as {}", key.short_id())
        } else {
            key.name.clone()
        };
        let name = {
            let input = read_line_with_default(&format!("Name [{}]: ", current_name))?;
            if input.is_empty() {
                key.name.clone()
            } else {
                input
            }
        };

        // Base URL
        let base_url = loop {
            let input = read_line_with_default(&format!("Base URL [{}]: ", key.base_url))?;
            let value = if input.is_empty() {
                key.base_url.clone()
            } else {
                input
            };
            if value == "copilot"
                || value == "ollama"
                || value == "aivo-starter"
                || value == "aivo starter"
                || value.starts_with("http://")
                || value.starts_with("https://")
            {
                break value;
            }
            eprintln!(
                "{} URL must start with http:// or https:// (or enter 'copilot' / 'ollama' for special providers)",
                style::red("Error:")
            );
        };

        // API Key
        let api_key = loop {
            let preview = display_secret(&key);
            let input = term_read_secret(&format!("API Key [{}]: ", preview))?;
            let value = if input.is_empty() {
                key.key.as_str().to_string()
            } else {
                input
            };
            if value.is_empty() {
                let prompt = style::yellow("Save without an API key?");
                if confirm(&prompt)? {
                    break String::new();
                }
            } else {
                break value;
            }
        };

        println!();

        let updated = self
            .session_store
            .update_key(
                &key.id,
                &name,
                &base_url,
                if base_url == key.base_url {
                    key.claude_protocol
                } else {
                    None
                },
                &api_key,
            )
            .await?;

        if updated && base_url != key.base_url {
            let _ = self
                .session_store
                .set_key_gemini_protocol(&key.id, None)
                .await?;
            let _ = self.session_store.set_key_codex_mode(&key.id, None).await?;
            let _ = self
                .session_store
                .set_key_opencode_mode(&key.id, None)
                .await?;
        }

        if !updated {
            eprintln!("{} Key no longer exists", style::red("Error:"));
            return Ok(ExitCode::UserError);
        }

        println!(
            "{} Updated key: {}",
            style::success_symbol(),
            style::cyan(if name.is_empty() {
                key.short_id()
            } else {
                &name
            })
        );

        Ok(ExitCode::Success)
    }

    /// Shows the provider picker, then routes to the appropriate add flow.
    /// `name` is the pre-collected key name (may be empty — each arm supplies
    /// its own default when empty).
    async fn interactive_add(&self, name: &str) -> Result<ExitCode> {
        #[derive(Clone, Copy, PartialEq)]
        enum ProviderChoice {
            Known(usize),
            Copilot,
            CodexOAuth,
            ClaudeOAuth,
            GeminiOAuth,
            Ollama,
            Starter,
            Custom,
        }

        let providers = crate::services::known_providers::all();
        let existing_keys = self.session_store.get_keys().await?;
        let has_starter = existing_keys
            .iter()
            .any(|k| is_aivo_starter_base(&k.base_url));

        let ollama_base_url = crate::services::ollama::ollama_openai_base_url();

        let label_for = |choice: ProviderChoice| -> (&str, String) {
            match choice {
                ProviderChoice::Known(i) => {
                    let p = &providers[i];
                    (
                        p.name.as_str(),
                        truncate_url_for_display(&p.base_url, PICKER_URL_MAX_LEN),
                    )
                }
                ProviderChoice::Ollama => ("Ollama", ollama_base_url.clone()),
                ProviderChoice::Copilot => ("GitHub Copilot", "device login".to_string()),
                ProviderChoice::CodexOAuth => (
                    "OpenAI Codex (ChatGPT)",
                    "browser login — multi-account".to_string(),
                ),
                ProviderChoice::ClaudeOAuth => (
                    "Claude Code (Anthropic)",
                    "browser login — multi-account".to_string(),
                ),
                ProviderChoice::GeminiOAuth => (
                    "Gemini (Google)",
                    "browser login — multi-account".to_string(),
                ),
                ProviderChoice::Starter => ("aivo starter", "free".to_string()),
                ProviderChoice::Custom => ("Custom URL", "enter manually".to_string()),
            }
        };

        let mut choices: Vec<ProviderChoice> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        let mut preselected: Option<usize> = None;
        let push = |choices: &mut Vec<ProviderChoice>,
                    labels: &mut Vec<String>,
                    choice: ProviderChoice| {
            let (label, hint) = label_for(choice);
            labels.push(format_picker_choice(label, &hint));
            choices.push(choice);
        };

        push(&mut choices, &mut labels, ProviderChoice::Custom);

        let detected_url = (!name.is_empty()).then(|| detect_base_url(name)).flatten();
        let detected_idx =
            detected_url.and_then(|url| providers.iter().position(|p| p.base_url == url));

        let hoisted_special: Option<ProviderChoice> = if detected_idx.is_none() && !name.is_empty()
        {
            match name.trim().to_ascii_lowercase().as_str() {
                "ollama" => Some(ProviderChoice::Ollama),
                "copilot" => Some(ProviderChoice::Copilot),
                "codex" => Some(ProviderChoice::CodexOAuth),
                "claude" => Some(ProviderChoice::ClaudeOAuth),
                "gemini" => Some(ProviderChoice::GeminiOAuth),
                _ => None,
            }
        } else {
            None
        };

        // Hoist the matched entry right after Custom URL so it's visible
        // without scrolling.
        if let Some(di) = detected_idx {
            push(&mut choices, &mut labels, ProviderChoice::Known(di));
            preselected = Some(labels.len() - 1);
        } else if let Some(special) = hoisted_special {
            push(&mut choices, &mut labels, special);
            preselected = Some(labels.len() - 1);
        }

        for (i, _) in providers.iter().enumerate() {
            if Some(i) == detected_idx {
                continue;
            }
            push(&mut choices, &mut labels, ProviderChoice::Known(i));
        }

        for choice in [
            ProviderChoice::Ollama,
            ProviderChoice::Copilot,
            ProviderChoice::CodexOAuth,
            ProviderChoice::ClaudeOAuth,
            ProviderChoice::GeminiOAuth,
        ] {
            if hoisted_special == Some(choice) {
                continue;
            }
            push(&mut choices, &mut labels, choice);
        }

        // Starter is a singleton — hide the picker entry once one is already set up.
        if !has_starter {
            push(&mut choices, &mut labels, ProviderChoice::Starter);
        }

        keys_ui::step_header(2, 3, "Provider", "preset or custom URL");

        let selection = FuzzySelect::new()
            .with_prompt("Provider")
            .items(&labels)
            .default(preselected.unwrap_or(0))
            .interact_opt()?;

        let Some(idx) = selection else {
            return Ok(ExitCode::Success);
        };

        // FuzzySelect's `console` crate can leave termios flags off.
        restore_cooked_mode();

        let picked = choices[idx];
        let picked_label = label_for(picked).0;
        let picked_url: Option<String> = match picked {
            ProviderChoice::Known(i) => Some(providers[i].base_url.clone()),
            ProviderChoice::Ollama => Some(ollama_base_url.clone()),
            _ => None,
        };
        match picked_url {
            Some(url) => println!(
                "{} {}  {}",
                style::dim("Provider:"),
                style::cyan(picked_label),
                style::dim(url),
            ),
            None => println!("{} {}", style::dim("Provider:"), style::cyan(picked_label)),
        }

        match picked {
            ProviderChoice::Known(i) => self.add_known_provider(name, &providers[i]).await,
            ProviderChoice::Copilot => self.add_copilot_interactive(name).await,
            ProviderChoice::CodexOAuth => self.add_codex_oauth_interactive(name).await,
            ProviderChoice::ClaudeOAuth => self.add_claude_oauth_interactive(name).await,
            ProviderChoice::GeminiOAuth => self.add_gemini_oauth_interactive(name).await,
            ProviderChoice::Ollama => self.add_ollama_interactive(name).await,
            ProviderChoice::Starter => self.add_starter_interactive(name).await,
            ProviderChoice::Custom => self.add_custom_interactive(name).await,
        }
    }

    async fn add_known_provider(
        &self,
        name: &str,
        provider: &crate::services::known_providers::KnownProvider,
    ) -> Result<ExitCode> {
        let name = if name.is_empty() {
            provider.id.clone()
        } else {
            name.to_string()
        };
        let is_cloudflare = provider.id == CLOUDFLARE_PROVIDER_ID;

        let base_url = if provider
            .base_url
            .contains(CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER)
        {
            keys_ui::step_header(
                3,
                3,
                "Credentials",
                "Cloudflare Account ID, then auth token",
            );
            let account_id = loop {
                let input = term_read_line(&style::dim("Cloudflare Account ID: "))?;
                if !input.is_empty() {
                    break input;
                }
                eprintln!("{} Account ID is required", style::red("Error:"));
            };
            provider
                .base_url
                .replace(CLOUDFLARE_ACCOUNT_ID_PLACEHOLDER, &account_id)
        } else {
            keys_ui::step_header(3, 3, "API Key", "input is hidden");
            provider.base_url.clone()
        };

        let (key_prompt, secret_noun) = if is_cloudflare {
            ("Auth Token: ", "an auth token")
        } else {
            ("API Key: ", "an API key")
        };
        let key = prompt_secret(key_prompt, secret_noun)?;

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            &name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;
        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    async fn add_copilot_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(COPILOT_INFO.0, COPILOT_INFO.1);

        let decision = self.confirm_replace_existing("copilot", "Copilot").await?;
        if matches!(decision, ReplaceDecision::Abort) {
            return Ok(ExitCode::Success);
        }

        keys_ui::step_header(3, 3, "Device login", "follow the code shown below");
        let token = crate::services::copilot_auth::device_flow_login().await?;
        if let ReplaceDecision::Replace(old_id) = decision {
            self.session_store.delete_key(&old_id).await?;
        }

        let name = if name.is_empty() { "copilot" } else { name };
        let id = self
            .session_store
            .add_key_with_protocol(name, "copilot", None, &token)
            .await?;
        self.finalize_add(
            &id,
            name,
            "Provider: GitHub Copilot",
            Some(("aivo run claude", "(uses Copilot subscription)")),
        )
        .await?;
        sync_models_in_background(&id, "copilot");
        Ok(ExitCode::Success)
    }

    /// Interactive Codex ChatGPT OAuth sign-in. Unlike Copilot we allow
    /// MULTIPLE accounts — each login produces a fresh key entry. The name
    /// defaults to the account's email claim on the id_token.
    async fn add_codex_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::codex_oauth::{CODEX_OAUTH_SENTINEL, interactive_login};

        keys_ui::provider_info(CODEX_OAUTH_INFO.0, CODEX_OAUTH_INFO.1);
        keys_ui::step_header(
            3,
            3,
            "Sign in",
            "follow the URL below — the browser opens automatically if possible",
        );

        let creds = interactive_login().await?;
        let derived_name = creds.email.clone().unwrap_or_else(|| "codex".to_string());
        let final_name = if name.is_empty() { &derived_name } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, CODEX_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        let email_line = match creds.email.as_deref() {
            Some(email) => format!("Signed in as {}", email),
            None => "Signed in to Codex".to_string(),
        };
        self.finalize_add(
            &id,
            final_name,
            &email_line,
            Some(("aivo run codex", "(launches codex with this account)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Interactive Claude Code OAuth sign-in. Unlike Copilot we allow MULTIPLE
    /// accounts — each login produces a fresh key entry. Shells out to
    /// `claude setup-token`, which drives the browser OAuth flow itself; we
    /// capture the opaque token it prints on stdout and store it encrypted.
    async fn add_claude_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::claude_oauth::{
            CLAUDE_OAUTH_SENTINEL, SetupTokenError, spawn_setup_token_and_capture,
        };

        keys_ui::provider_info(CLAUDE_OAUTH_INFO.0, CLAUDE_OAUTH_INFO.1);
        keys_ui::step_header(
            3,
            3,
            "Sign in",
            "running `claude setup-token` — follow the browser prompt",
        );

        let creds = match spawn_setup_token_and_capture().await {
            Ok(c) => c,
            Err(SetupTokenError::ClaudeNotFound) => {
                eprintln!(
                    "{} The `claude` CLI wasn't found on PATH.",
                    style::red("Error:")
                );
                eprintln!(
                    "  {} Install Claude Code first: {}",
                    style::dim("hint:"),
                    style::cyan("npm i -g @anthropic-ai/claude-code")
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::EmptyOutput) => {
                eprintln!(
                    "{} `claude setup-token` exited without printing a token.",
                    style::red("Error:")
                );
                eprintln!(
                    "  {} If the login was cancelled, try again. Otherwise report the output above.",
                    style::dim("hint:"),
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::NonZeroExit { status }) => {
                eprintln!(
                    "{} `claude setup-token` failed ({}).",
                    style::red("Error:"),
                    status
                );
                return Ok(ExitCode::UserError);
            }
            Err(SetupTokenError::Other(e)) => return Err(e),
        };

        let final_name = if name.is_empty() { "claude" } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, CLAUDE_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        self.finalize_add(
            &id,
            final_name,
            "Signed in to Claude Code",
            Some(("aivo run claude", "(launches claude with this account)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Interactive Google OAuth sign-in for the `gemini` CLI. Multiple
    /// accounts supported — each login produces a fresh key entry named
    /// after the account's email. Tokens are stored encrypted; at launch
    /// time aivo projects them into a shadow `GEMINI_CLI_HOME` so the
    /// native `gemini` CLI reads them without touching the user's real
    /// `~/.gemini/`.
    async fn add_gemini_oauth_interactive(&self, name: &str) -> Result<ExitCode> {
        use crate::services::gemini_oauth::{GEMINI_OAUTH_SENTINEL, interactive_login};

        keys_ui::provider_info(GEMINI_OAUTH_INFO.0, GEMINI_OAUTH_INFO.1);
        keys_ui::step_header(
            3,
            3,
            "Sign in",
            "follow the URL below — the browser opens automatically if possible",
        );

        let creds = interactive_login().await?;
        let derived_name = creds.email.clone().unwrap_or_else(|| "gemini".to_string());
        let final_name = if name.is_empty() { &derived_name } else { name };
        let creds_json = creds.to_json()?;

        let id = self
            .session_store
            .add_key_with_protocol(final_name, GEMINI_OAUTH_SENTINEL, None, &creds_json)
            .await?;

        let email_line = match creds.email.as_deref() {
            Some(email) => format!("Signed in as {}", email),
            None => "Signed in to Gemini".to_string(),
        };
        self.finalize_add(
            &id,
            final_name,
            &email_line,
            Some(("aivo run gemini", "(launches gemini with this account)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    async fn add_ollama_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(OLLAMA_INFO.0, OLLAMA_INFO.1);

        let decision = self.confirm_replace_existing("ollama", "Ollama").await?;
        if matches!(decision, ReplaceDecision::Abort) {
            return Ok(ExitCode::Success);
        }

        keys_ui::step_header(3, 3, "Verify local install", "checking ollama is reachable");
        crate::services::ollama::ensure_ready().await?;
        if let ReplaceDecision::Replace(old_id) = decision {
            self.session_store.delete_key(&old_id).await?;
        }

        let name = if name.is_empty() { "ollama" } else { name };
        let id = self
            .session_store
            .add_key_with_protocol(name, "ollama", None, "ollama-local")
            .await?;
        self.finalize_add(
            &id,
            name,
            "Provider: Ollama (local)",
            Some(("aivo models", "(list local models)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    async fn add_starter_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::provider_info(STARTER_INFO.0, STARTER_INFO.1);

        if let Some(code) = self.notify_if_starter_already_added().await? {
            return Ok(code);
        }

        let _ = self.session_store.set_starter_key_dismissed(false).await;

        let (starter, _) = self
            .session_store
            .ensure_starter_key()
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to create aivo starter key"))?;
        let display_name = if name.is_empty() {
            "aivo-starter"
        } else {
            name
        };
        self.finalize_add(
            &starter.id,
            display_name,
            "Provider: aivo starter (free)",
            Some(("aivo chat", "(start chatting)")),
        )
        .await?;
        Ok(ExitCode::Success)
    }

    /// Prints a notice and returns `Some(Success)` if an aivo-starter key is
    /// already configured; returns `None` otherwise so the caller can create one.
    async fn notify_if_starter_already_added(&self) -> Result<Option<ExitCode>> {
        let Some(existing) = self
            .session_store
            .get_keys()
            .await?
            .into_iter()
            .find(|k| k.base_url == crate::constants::AIVO_STARTER_SENTINEL)
        else {
            return Ok(None);
        };
        println!(
            "{} aivo starter is already added as {} (ID: {}). Run {} to use it.",
            style::yellow("Note:"),
            style::cyan(&existing.name),
            style::dim(&existing.id),
            style::cyan("aivo chat"),
        );
        Ok(Some(ExitCode::Success))
    }

    async fn add_custom_interactive(&self, name: &str) -> Result<ExitCode> {
        keys_ui::step_header(3, 3, "Credentials", "base URL, then API key");
        let base_url = loop {
            let input = term_read_line(&style::dim("Base URL: "))?;
            if input.is_empty() {
                continue;
            }
            match validate_base_url(&input) {
                Ok(()) => break input,
                Err(msg) => {
                    eprintln!("{} {}", style::red("Error:"), msg);
                }
            }
        };

        let key = prompt_secret("API Key: ", "an API key")?;

        let id = self
            .session_store
            .add_key_with_protocol(name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;
        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    /// Prompts to replace any existing key with the given `base_url`.
    /// - Returns `Ok(Some(id))` — caller must delete `id` before adding the new key.
    /// - Returns `Ok(None)` when there was no existing key (caller proceeds).
    /// - Returns `Err` on IO error; aborts on user decline by returning early via this type.
    async fn confirm_replace_existing(
        &self,
        base_url: &str,
        label: &str,
    ) -> Result<ReplaceDecision> {
        let existing_keys = self.session_store.get_keys().await?;
        let Some(existing) = existing_keys.iter().find(|k| k.base_url == base_url) else {
            return Ok(ReplaceDecision::NoExisting);
        };
        let answer = term_read_line(&style::dim(format!(
            "{} {} key '{}' (ID: {}) already exists. Replace it? [y/N] ",
            style::yellow("Warning:"),
            label,
            existing.name,
            existing.id
        )))?;
        if matches!(answer.to_lowercase().as_str(), "y" | "yes") {
            Ok(ReplaceDecision::Replace(existing.id.clone()))
        } else {
            println!("Aborted.");
            Ok(ReplaceDecision::Abort)
        }
    }

    /// Interactively adds an API key
    async fn add_key(
        &self,
        provided_name: Option<&str>,
        add_options: AddKeyOptions<'_>,
    ) -> Result<ExitCode> {
        use std::io;

        fn read_line(prompt: &str) -> io::Result<String> {
            term_read_line(&style::dim(prompt))
        }

        if provided_name.is_some() && add_options.name.is_some() {
            eprintln!(
                "{} Specify the key name either positionally or with --name",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }

        // Defensive: a previously-crashed invocation may have left the terminal
        // in raw mode, which breaks backspace in the first prompt.
        restore_cooked_mode();
        let mut name_was_prompted = false;
        let name = if let Some(n) = add_options.name.or(provided_name) {
            n.to_string()
        } else if add_options.base_url.is_some() && add_options.key.is_some() {
            String::new()
        } else {
            name_was_prompted = true;
            keys_ui::step_header(1, 3, "Name", "a short label for this key");
            read_line("Name (optional): ")?
        };

        let is_starter_name = name == "aivo-starter" || name == "aivo starter";
        let interactive =
            add_options.base_url.is_none() && add_options.key.is_none() && !is_starter_name;

        if interactive {
            // Echo the name when it came from arg/flag so the user sees what
            // was used before the provider picker takes over.
            if !name_was_prompted && !name.is_empty() {
                println!("{} {}", style::dim("Name:"), style::cyan(&name));
            }
            return self.interactive_add(&name).await;
        }

        let base_url = if is_starter_name {
            crate::constants::AIVO_STARTER_SENTINEL.to_string()
        } else {
            let detected_url = detect_base_url(&name);
            let mut provided_base_url = add_options.base_url.map(str::to_string);
            loop {
                let value = if let Some(value) = provided_base_url.take() {
                    value
                } else {
                    let prompt = match detected_url {
                        Some(default) => format!("Base URL [{}]: ", default),
                        None => "Base URL: ".to_string(),
                    };
                    let input = read_line(&prompt)?;
                    if input.is_empty() {
                        detected_url.unwrap_or("").to_string()
                    } else {
                        input
                    }
                };
                let picker_hint =
                    "run 'aivo keys add' (no flags) and pick it from the provider list";
                let picker_rejections: &[(&str, &str)] = &[
                    ("copilot", "GitHub Copilot login needs the device flow"),
                    ("ollama", "Ollama setup needs a local installation check"),
                    (
                        crate::services::codex_oauth::CODEX_OAUTH_SENTINEL,
                        "Codex ChatGPT login needs browser auth",
                    ),
                    (
                        crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL,
                        "Claude Code login needs browser auth",
                    ),
                    (
                        crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL,
                        "Gemini login needs browser auth",
                    ),
                ];
                let reject = |msg: String| -> Option<ExitCode> {
                    eprintln!("{} {msg}", style::red("Error:"));
                    add_options
                        .base_url
                        .is_some()
                        .then_some(ExitCode::UserError)
                };
                if let Some((_, reason)) = picker_rejections.iter().find(|(s, _)| value == *s) {
                    if let Some(code) = reject(format!("{reason} — {picker_hint}.")) {
                        return Ok(code);
                    }
                    continue;
                }
                if value == "aivo-starter" || value == "aivo starter" {
                    if let Some(code) =
                        reject("Use 'aivo keys add aivo-starter' instead.".to_string())
                    {
                        return Ok(code);
                    }
                    continue;
                }
                if value.starts_with("http://") || value.starts_with("https://") {
                    break value;
                }
                if let Some(code) = reject("URL must start with http:// or https://".to_string()) {
                    return Ok(code);
                }
            }
        };

        // GitHub Copilot: use device flow instead of manual key entry
        if base_url == "copilot" {
            if add_options.key.is_some() {
                eprintln!(
                    "{} GitHub Copilot uses device login; run 'aivo keys add' (no flags) and pick GitHub Copilot.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }

            keys_ui::provider_info(COPILOT_INFO.0, COPILOT_INFO.1);

            // Check for an existing Copilot key and prompt to replace
            let existing_keys = self.session_store.get_keys().await?;
            let existing_copilot_id =
                if let Some(existing) = existing_keys.iter().find(|k| k.base_url == "copilot") {
                    let answer = read_line(&format!(
                        "{} Copilot key '{}' (ID: {}) already exists. Replace it? [y/N] ",
                        style::yellow("Warning:"),
                        existing.name,
                        existing.id
                    ))?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        println!("Aborted.");
                        return Ok(ExitCode::Success);
                    }
                    Some(existing.id.clone())
                } else {
                    None
                };

            let token = crate::services::copilot_auth::device_flow_login().await?;

            // Device flow succeeded — now safe to remove the old key
            if let Some(old_id) = existing_copilot_id {
                self.session_store.delete_key(&old_id).await?;
            }

            let id = self
                .session_store
                .add_key_with_protocol(&name, "copilot", None, &token)
                .await?;
            self.finalize_add(
                &id,
                &name,
                "Provider: GitHub Copilot",
                Some(("aivo run claude", "(uses Copilot subscription)")),
            )
            .await?;

            sync_models_in_background(&id, &base_url);
            return Ok(ExitCode::Success);
        }

        // Ollama: verify installation, no API key needed
        if base_url == "ollama" {
            if add_options.key.is_some() {
                eprintln!(
                    "{} Ollama runs locally without authentication. Do not pass --key.",
                    style::red("Error:")
                );
                return Ok(ExitCode::UserError);
            }

            keys_ui::provider_info(OLLAMA_INFO.0, OLLAMA_INFO.1);

            crate::services::ollama::ensure_ready().await?;

            // Check for an existing Ollama key and prompt to replace
            let existing_keys = self.session_store.get_keys().await?;
            let existing_ollama_id =
                if let Some(existing) = existing_keys.iter().find(|k| k.base_url == "ollama") {
                    let answer = read_line(&format!(
                        "{} Ollama key '{}' (ID: {}) already exists. Replace it? [y/N] ",
                        style::yellow("Warning:"),
                        existing.name,
                        existing.id
                    ))?;
                    if !matches!(answer.to_lowercase().as_str(), "y" | "yes") {
                        println!("Aborted.");
                        return Ok(ExitCode::Success);
                    }
                    Some(existing.id.clone())
                } else {
                    None
                };

            if let Some(old_id) = existing_ollama_id {
                self.session_store.delete_key(&old_id).await?;
            }

            let id = self
                .session_store
                .add_key_with_protocol(&name, "ollama", None, "ollama-local")
                .await?;
            self.finalize_add(
                &id,
                &name,
                "Provider: Ollama (local)",
                Some(("aivo models", "(list local models)")),
            )
            .await?;

            return Ok(ExitCode::Success);
        }

        // Aivo starter: free provider, no API key needed
        if base_url == crate::constants::AIVO_STARTER_SENTINEL {
            keys_ui::provider_info(STARTER_INFO.0, STARTER_INFO.1);

            if let Some(code) = self.notify_if_starter_already_added().await? {
                return Ok(code);
            }

            // Clear the dismissed flag so ensure_starter_key works again
            let _ = self.session_store.set_starter_key_dismissed(false).await;

            let (starter, _) = self
                .session_store
                .ensure_starter_key()
                .await
                .ok_or_else(|| anyhow::anyhow!("Failed to create aivo starter key"))?;
            self.finalize_add(
                &starter.id,
                &name,
                "Provider: aivo starter (free)",
                Some(("aivo chat", "(start chatting)")),
            )
            .await?;

            return Ok(ExitCode::Success);
        }

        let key = if let Some(key) = add_options.key {
            key.to_string()
        } else {
            loop {
                let input = term_read_secret(&style::dim("API Key: "))?;
                if !input.is_empty() {
                    break input;
                }

                let prompt = style::yellow("Save without an API key?");
                if confirm(&prompt)? {
                    break String::new();
                }
            }
        };

        let id = self
            .session_store
            .add_key_with_protocol(&name, &base_url, None, &key)
            .await?;
        self.finalize_add(
            &id,
            &name,
            &format!("Base URL: {}", base_url),
            Some(("aivo claude", "")),
        )
        .await?;

        sync_models_in_background(&id, &base_url);
        Ok(ExitCode::Success)
    }

    /// Removes an API key by ID or name
    async fn remove_key(&self, key_id_or_name: Option<&str>) -> Result<ExitCode> {
        let key_to_remove = match self
            .resolve_key_selection(
                key_id_or_name,
                "Select a key to remove",
                "No keys to remove.",
            )
            .await?
        {
            KeySelection::Key(key) => key,
            KeySelection::Cancelled => {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
            KeySelection::Empty => return Ok(ExitCode::Success),
            KeySelection::NotFound => return Ok(ExitCode::UserError),
        };

        // Show confirmation — display full stored ID (not short form) before a destructive action
        println!("ID:  {}", style::cyan(&key_to_remove.id));
        println!("URL: {}", style::dim(&key_to_remove.base_url));
        println!();

        let confirmed = confirm(&format!("Remove \"{}\"?", key_to_remove.display_name()))?;

        if !confirmed {
            println!("{}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        }

        if self.session_store.delete_key(&key_to_remove.id).await? {
            let _ = self.session_store.remove_key_stats(&key_to_remove.id).await;
            // Remember if the user dismissed the aivo-starter key
            if key_to_remove.base_url == crate::constants::AIVO_STARTER_SENTINEL {
                let _ = self.session_store.set_starter_key_dismissed(true).await;
            }
            println!(
                "{} Removed key: {}",
                style::success_symbol(),
                style::cyan(key_to_remove.display_name())
            );
            Ok(ExitCode::Success)
        } else {
            eprintln!("{} Failed to remove key", style::red("Error:"));
            Ok(ExitCode::UserError)
        }
    }

    async fn resolve_key_selection(
        &self,
        key_id_or_name: Option<&str>,
        prompt: &str,
        empty_message: &str,
    ) -> Result<KeySelection> {
        // Load without decrypting — only metadata is needed for selection.
        let (all_keys, active_key_id) = self.session_store.get_keys_and_active_id_info().await?;

        if all_keys.is_empty() {
            println!("{}", style::dim(empty_message));
            return Ok(KeySelection::Empty);
        }

        let selected = if let Some(key_id_or_name) = key_id_or_name {
            if let Some(key) = all_keys
                .iter()
                .find(|k| k.id == key_id_or_name || k.short_id() == key_id_or_name)
            {
                Some(key.clone())
            } else {
                let name_matches: Vec<ApiKey> = all_keys
                    .iter()
                    .filter(|k| k.name == key_id_or_name)
                    .cloned()
                    .collect();

                match name_matches.len() {
                    0 => {
                        eprintln!(
                            "{} API key \"{}\" not found",
                            style::red("Error:"),
                            key_id_or_name
                        );
                        eprintln!();
                        eprintln!("{}", style::dim("Run 'aivo keys' to see available keys."));
                        return Ok(KeySelection::NotFound);
                    }
                    1 => Some(name_matches[0].clone()),
                    _ => {
                        println!(
                            "{} Multiple keys found with name \"{}\":",
                            style::yellow("Note:"),
                            key_id_or_name
                        );
                        prompt_pick_key(&name_matches, &[], prompt, 0)?
                    }
                }
            }
        } else {
            let default_idx = active_key_id
                .and_then(|id| all_keys.iter().position(|k| k.id == id))
                .unwrap_or(0);
            prompt_pick_key(&all_keys, &[], prompt, default_idx)?
        };

        match selected {
            Some(key) => Ok(KeySelection::Key(key)),
            None => Ok(KeySelection::Cancelled),
        }
    }

    // Shows usage information.
    pub fn print_help() {
        let print_row = |label: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<18}", label)),
                style::dim(desc)
            );
        };

        println!("{} aivo keys [action]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Manage API keys: add, remove, activate, and inspect.")
        );
        println!();
        println!("{}", style::bold("Actions:"));
        print_row("(no action)", "List all API keys");
        print_row("use [id|name]", "Activate a specific API key");
        print_row("cat [id|name]", "Display details for a key");
        print_row("rm [id|name]", "Remove an API key");
        print_row("add [name]", "Add an API key");
        print_row("edit [id|name]", "Edit an API key");
        print_row("ping [id|name]", "Health-check API keys (or: aivo ping)");
        println!();
        println!("{}", style::bold("Add Flags:"));
        print_row("--name <name>", "Set key name");
        print_row("--base-url <url>", "Set provider base URL");
        print_row("--key <api-key>", "Set provider API key");
        println!();
        println!("{}", style::bold("List Flags:"));
        print_row("--ping", "List keys with live ping status");
        print_row("--json", "Output list as JSON (secret is never included)");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo keys"));
        println!("  {}", style::dim("aivo keys use openrouter"));
        println!(
            "  {}",
            style::dim("aivo keys add --name abc --base-url https://example.io --key sk-...")
        );
        println!("  {}", style::dim("aivo keys --json"));
        println!(
            "  {}",
            style::dim("aivo keys --ping --json | jq '.[] | select(.ping.ok==false)'")
        );
    }
}

// Formats an API key as a choice string for interactive selectors.
pub(crate) fn format_key_choice(key: &ApiKey) -> String {
    format!(
        "{}  {}  {}",
        style::cyan(format!("{:<3}", key.short_id())),
        key.display_name(),
        style::dim(&key.base_url)
    )
}

// Prompts the user to select a key from the given list. `annotations`
// parallels `keys`: `Some(reason)` disables that row and shows the reason
// as a dim suffix. Pass `&[]` for no annotations.
fn prompt_pick_key(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    let choices: Vec<String> = keys.iter().map(format_key_choice).collect();
    let mut picker = FuzzySelect::new()
        .with_prompt(prompt)
        .items(&choices)
        .default(default);
    if !annotations.is_empty() {
        picker = picker.annotations(annotations.to_vec());
    }
    let selection = picker.interact_opt()?;
    Ok(selection.map(|idx| keys[idx].clone()))
}

pub(crate) fn prompt_pick_key_without_activation(
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            SessionStore::decrypt_key_secret(&mut key)?;
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

// Picks a key from `keys` and activates it. Returns `Ok(None)` if cancelled.
#[allow(dead_code)] // used by binary crate (key_resolution.rs)
pub(crate) async fn prompt_select_key(
    session_store: &SessionStore,
    keys: &[ApiKey],
    annotations: &[Option<String>],
    prompt: &str,
    default: usize,
) -> Result<Option<ApiKey>> {
    match prompt_pick_key(keys, annotations, prompt, default)? {
        Some(mut key) => {
            SessionStore::decrypt_key_secret(&mut key)?;
            session_store.set_active_key(&key.id).await?;
            let preview = display_secret(&key);
            eprintln!(
                "{} Activated key: {} {}",
                style::success_symbol(),
                style::cyan(key.display_name()),
                style::dim(&preview)
            );
            Ok(Some(key))
        }
        None => Ok(None),
    }
}

/// Offers a picker of compatible keys when `bad_key` is an OAuth credential
/// the current command can't use. OAuth keys stay visible but are disabled
/// with an inline reason. `context_phrase` is the user-visible command name
/// inserted into messages (e.g. `"aivo chat"` or `"aivo run codex"`).
///
/// Returns `Ok(Some(new_key))` when the user picks a replacement; `Ok(None)`
/// when there's no TTY, no eligible key, or the user cancelled — callers
/// should exit with `ExitCode::UserError`.
pub(crate) async fn swap_incompatible_key(
    session_store: &SessionStore,
    bad_key: &ApiKey,
    compat: crate::services::key_compat::KeyCompatContext,
    context_phrase: &str,
) -> Result<Option<ApiKey>> {
    use std::io::IsTerminal;

    let all_keys = session_store.get_keys().await?;
    let annotations = compat.annotations_for(&all_keys);
    let has_eligible = annotations.iter().any(Option::is_none);

    if !has_eligible || !std::io::stderr().is_terminal() {
        eprintln!(
            "{} Key '{}' is a {} OAuth account — `{}` can't use it.",
            style::red("Error:"),
            bad_key.display_name(),
            bad_key.oauth_kind_label(),
            context_phrase,
        );
        eprintln!(
            "  {} Use `{}` or select a regular API key.",
            style::dim("hint:"),
            bad_key.oauth_tool_hint(),
        );
        return Ok(None);
    }

    eprintln!(
        "{} Key '{}' is a {} OAuth account — pick a regular API key for `{}`.",
        style::yellow("Note:"),
        bad_key.display_name(),
        bad_key.oauth_kind_label(),
        context_phrase,
    );

    prompt_pick_key_without_activation(&all_keys, &annotations, "Select a key", 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::KeysArgs;

    fn keys_args(action: Option<&str>, args: &[&str]) -> KeysArgs {
        KeysArgs {
            action: action.map(str::to_string),
            args: args.iter().map(|s| s.to_string()).collect(),
            name: None,
            base_url: None,
            key: None,
            all: false,
            ping: false,
            json: false,
        }
    }

    #[test]
    fn test_keys_command_creation() {
        let session_store = SessionStore::new();
        let _command = KeysCommand::new(session_store);
    }

    #[tokio::test]
    async fn test_edit_key_missing_id() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_edit_key_not_found() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        store
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("edit"), &["nonexistent"])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_use_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        // No keys stored — should succeed (prints "No API keys found.")
        let code = cmd.execute(keys_args(Some("use"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_cat_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("cat"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_remove_key_no_arg_no_keys() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("rm"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::Success);
    }

    #[tokio::test]
    async fn test_keys_list_action_is_rejected() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);
        let code = cmd.execute(keys_args(Some("list"), &[])).await;
        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_with_flags() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: Some("minimax".to_string()),
                base_url: Some("https://api.minimax.io/anthropic".to_string()),
                key: Some("sk-minimax-test".to_string()),
                all: false,
                ping: false,
                json: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "minimax");
        assert_eq!(keys[0].base_url, "https://api.minimax.io/anthropic");
        assert_eq!(keys[0].claude_protocol, None);

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);
        assert_eq!(active.key.as_str(), "sk-minimax-test");
    }

    #[tokio::test]
    async fn test_add_key_rejects_conflicting_name_sources() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: vec!["positional-name".to_string()],
                name: Some("flag-name".to_string()),
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                key: Some("sk-or-v1-test".to_string()),
                all: false,
                ping: false,
                json: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_without_name_uses_empty_stored_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store.clone());

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("https://openrouter.ai/api/v1".to_string()),
                key: Some("sk-or-v1-test".to_string()),
                all: false,
                ping: false,
                json: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::Success);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "");
        assert_eq!(keys[0].display_name(), keys[0].short_id());

        let active = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(active.id, keys[0].id);
    }

    #[tokio::test]
    async fn test_add_key_rejects_ollama_base_url_without_ollama_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("ollama".to_string()),
                key: None,
                all: false,
                ping: false,
                json: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[tokio::test]
    async fn test_add_key_rejects_copilot_base_url_without_copilot_name() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = crate::services::session_store::SessionStore::with_path(config_path);
        let cmd = KeysCommand::new(store);

        let code = cmd
            .execute(KeysArgs {
                action: Some("add".to_string()),
                args: Vec::new(),
                name: None,
                base_url: Some("copilot".to_string()),
                key: None,
                all: false,
                ping: false,
                json: false,
            })
            .await;

        assert_eq!(code, crate::errors::ExitCode::UserError);
    }

    #[test]
    fn test_detect_base_url_exact_match() {
        assert_eq!(
            detect_base_url("openrouter"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("deepseek"),
            Some("https://api.deepseek.com/v1")
        );
        assert_eq!(
            detect_base_url("groq"),
            Some("https://api.groq.com/openai/v1")
        );
        assert_eq!(
            detect_base_url("mistral"),
            Some("https://api.mistral.ai/v1")
        );
        assert_eq!(detect_base_url("xai"), Some("https://api.x.ai/v1"));
        assert_eq!(
            detect_base_url("fireworks-ai"),
            Some("https://api.fireworks.ai/inference/v1/")
        );
        assert_eq!(
            detect_base_url("moonshotai"),
            Some("https://api.moonshot.ai/v1")
        );
        assert_eq!(
            detect_base_url("minimax"),
            Some("https://api.minimax.io/anthropic/v1")
        );
        assert_eq!(
            detect_base_url("vercel"),
            Some("https://ai-gateway.vercel.sh/v1")
        );
    }

    #[test]
    fn test_detect_base_url_case_insensitive() {
        assert_eq!(
            detect_base_url("OpenRouter"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("GROQ"),
            Some("https://api.groq.com/openai/v1")
        );
        assert_eq!(
            detect_base_url("DeepSeek"),
            Some("https://api.deepseek.com/v1")
        );
    }

    #[test]
    fn test_detect_base_url_substring() {
        assert_eq!(
            detect_base_url("my-openrouter-key"),
            Some("https://openrouter.ai/api/v1")
        );
        assert_eq!(
            detect_base_url("work_groq"),
            Some("https://api.groq.com/openai/v1")
        );
    }

    #[test]
    fn test_detect_base_url_no_match() {
        assert_eq!(detect_base_url("random"), None);
        assert_eq!(detect_base_url(""), None);
    }

    #[test]
    fn test_validate_base_url() {
        assert!(validate_base_url("https://api.example.com").is_ok());
        assert!(validate_base_url("http://localhost:8080").is_ok());
        assert!(validate_base_url("https://api.example.com/v1/path").is_ok());

        assert!(validate_base_url("ftp://example.com").is_err());
        assert!(validate_base_url("example.com").is_err());
        assert!(validate_base_url("https://").is_err());
        assert!(validate_base_url("https:// example.com").is_err());
        assert!(validate_base_url("https://example.com/path with space").is_err());
    }

    #[test]
    fn test_ping_status_from_http_status_ok() {
        assert_eq!(PingStatus::from_http_status(200), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(204), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_from_http_status_auth_error() {
        assert_eq!(PingStatus::from_http_status(401), PingStatus::AuthError);
        assert_eq!(PingStatus::from_http_status(403), PingStatus::AuthError);
    }

    #[test]
    fn test_ping_status_from_http_status_other() {
        assert_eq!(
            PingStatus::from_http_status(500),
            PingStatus::Error("HTTP 500".to_string())
        );
        assert_eq!(
            PingStatus::from_http_status(429),
            PingStatus::Error("HTTP 429".to_string())
        );
    }

    #[test]
    fn test_ping_status_from_http_status_reachable_wrong_path() {
        assert_eq!(PingStatus::from_http_status(404), PingStatus::Ok);
        assert_eq!(PingStatus::from_http_status(405), PingStatus::Ok);
    }

    #[test]
    fn test_ping_status_icons_and_messages() {
        assert_eq!(PingStatus::Ok.icon(), "✓");
        assert_eq!(PingStatus::Ok.message(), "ok");
        assert_eq!(PingStatus::AuthError.icon(), "✗");
        assert_eq!(PingStatus::AuthError.message(), "auth failed");
        assert_eq!(PingStatus::Unreachable.icon(), "✗");
        assert_eq!(PingStatus::Unreachable.message(), "unreachable");
        assert_eq!(PingStatus::Timeout.icon(), "✗");
        assert_eq!(PingStatus::Timeout.message(), "timeout");
    }

    #[test]
    fn test_ping_result_empty_name_uses_short_id() {
        let result = PingResult {
            name: "abc".to_string(),
            url: "https://api.openai.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        assert_eq!(result.name, "abc");
        assert_eq!(result.status, PingStatus::Ok);
    }

    #[test]
    fn test_format_key_choice_uses_id_for_unnamed_keys() {
        let key = ApiKey::new_with_protocol(
            "a2b".to_string(),
            String::new(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-test".to_string(),
        );

        let choice = format_key_choice(&key);

        assert!(choice.contains("a2b"));
        assert!(choice.contains("https://openrouter.ai/api/v1"));
    }

    #[test]
    fn key_preview_redacts_by_length() {
        for (input, expected) in [
            ("sk-abc", "sk-..."),
            ("1234567890", "123..."),
            ("sk-abcdefghijklmnop", "sk-abc...mnop"),
            ("", "..."),
        ] {
            assert_eq!(key_preview(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn key_preview_unicode_safe() {
        // Multi-byte chars: prove we slice on char boundaries, not bytes.
        let key = "🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑🔑";
        let out = key_preview(key);
        assert!(out.contains("..."));
        assert!(out.starts_with("🔑"));
    }

    #[test]
    fn display_secret_labels_oauth_and_copilot() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
        use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;

        let cases = [
            (CLAUDE_OAUTH_SENTINEL, "<Claude OAuth>"),
            (CODEX_OAUTH_SENTINEL, "<Codex OAuth>"),
            (GEMINI_OAUTH_SENTINEL, "<Gemini OAuth>"),
            ("copilot", "<Copilot>"),
        ];
        for (base_url, expected) in cases {
            let key = ApiKey::new_with_protocol(
                "id".to_string(),
                "name".to_string(),
                base_url.to_string(),
                None,
                "must-not-leak-this-credential-blob".to_string(),
            );
            let out = display_secret(&key);
            assert_eq!(out, expected, "base_url: {base_url}");
            assert!(!out.contains("must-not-leak"), "base_url: {base_url}");
        }
    }

    #[test]
    fn display_secret_falls_back_to_preview_for_api_keys() {
        let key = ApiKey::new_with_protocol(
            "id".to_string(),
            "name".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-abcdefghijklmnop".to_string(),
        );
        assert_eq!(display_secret(&key), "sk-abc...mnop");
    }

    #[test]
    fn key_metadata_json_excludes_secret() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk-must-not-leak".to_string(),
        );
        let payload = key_metadata_json(&key, Some("abc"));
        let s = serde_json::to_string(&payload).unwrap();
        assert!(!s.contains("sk-must-not-leak"));
        assert!(s.contains("\"active\":true"));
        assert_eq!(payload["id"], "abc");
        assert_eq!(payload["name"], "test");
        assert_eq!(payload["base_url"], "https://api.example.com");
    }

    #[test]
    fn key_metadata_json_inactive() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "https://api.example.com".to_string(),
            None,
            "sk".to_string(),
        );
        let payload = key_metadata_json(&key, Some("xyz"));
        assert_eq!(payload["active"], false);
        let payload = key_metadata_json(&key, None);
        assert_eq!(payload["active"], false);
    }

    #[test]
    fn ping_result_json_ok_includes_latency() {
        let result = PingResult {
            name: "test".to_string(),
            url: "https://api.example.com".to_string(),
            status: PingStatus::Ok,
            latency: Some(std::time::Duration::from_millis(42)),
        };
        let payload = ping_result_json(&result);
        assert_eq!(payload["ok"], true);
        assert_eq!(payload["status"], "ok");
        assert_eq!(payload["latency_ms"], 42);
    }

    #[test]
    fn ping_result_json_error_status_keys() {
        for (status, expected_key) in [
            (PingStatus::AuthError, "auth_error"),
            (PingStatus::Unreachable, "unreachable"),
            (PingStatus::Timeout, "timeout"),
            (PingStatus::Error("boom".into()), "error"),
        ] {
            let result = PingResult {
                name: "n".to_string(),
                url: "u".to_string(),
                status,
                latency: None,
            };
            let payload = ping_result_json(&result);
            assert_eq!(payload["ok"], false);
            assert_eq!(payload["status"], expected_key);
            assert!(payload["latency_ms"].is_null());
        }
    }
}
