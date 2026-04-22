use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

#[allow(unused_imports)]
pub use crate::services::session_crypto::{decrypt, encrypt, is_encrypted};
use crate::services::system_env;

use crate::services::api_key_store::ApiKeyStore;
use crate::services::atomic_write::atomic_write_secure;
use crate::services::chat_session_store::ChatSessionStore;
use crate::services::last_selection::LastSelectionStore;
use crate::services::log_store::LogStore;
use crate::services::usage_stats_store::UsageStatsStore;

/// Serde module for serializing/deserializing Zeroizing<String> as regular String
mod zeroizing_string {
    use serde::{Deserialize, Deserializer, Serializer};
    use zeroize::Zeroizing;

    pub fn serialize<S>(value: &Zeroizing<String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(value.as_str())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Zeroizing<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(Zeroizing::new(s))
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

/// API key stored on user's machine
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeProviderProtocol {
    Anthropic,
    Openai,
    Google,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GeminiProviderProtocol {
    Google,
    Openai,
    Anthropic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OpenAICompatibilityMode {
    Direct,
    Router,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(
        rename = "claudeProtocol",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub claude_protocol: Option<ClaudeProviderProtocol>,
    #[serde(
        rename = "geminiProtocol",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub gemini_protocol: Option<GeminiProviderProtocol>,
    #[serde(
        rename = "codexResponsesApi",
        default,
        alias = "responsesApiSupported",
        skip_serializing_if = "Option::is_none"
    )]
    pub responses_api_supported: Option<bool>,
    #[serde(rename = "codexMode", default, skip_serializing_if = "Option::is_none")]
    pub codex_mode: Option<OpenAICompatibilityMode>,
    #[serde(
        rename = "opencodeMode",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub opencode_mode: Option<OpenAICompatibilityMode>,
    #[serde(rename = "piMode", default, skip_serializing_if = "Option::is_none")]
    pub pi_mode: Option<OpenAICompatibilityMode>,
    #[serde(with = "zeroizing_string")]
    pub key: Zeroizing<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

impl ApiKey {
    pub fn new_with_protocol(
        id: String,
        name: String,
        base_url: String,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: String,
    ) -> Self {
        Self {
            id,
            name,
            base_url,
            claude_protocol,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            key: Zeroizing::new(key),
            created_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn short_id(&self) -> &str {
        &self.id[..self.id.len().min(3)]
    }

    pub fn display_name(&self) -> &str {
        if self.name.is_empty() {
            self.short_id()
        } else {
            &self.name
        }
    }

    /// True when this entry stores a Codex ChatGPT OAuth credential bundle
    /// (encrypted JSON in `key`) rather than a plain API key.
    pub fn is_codex_oauth(&self) -> bool {
        self.base_url == crate::services::codex_oauth::CODEX_OAUTH_SENTINEL
    }

    /// True when this entry stores a Claude Code OAuth token (captured via
    /// `claude setup-token`, stored as serialized `ClaudeOAuthCredential` JSON
    /// in `key`) rather than a plain API key.
    pub fn is_claude_oauth(&self) -> bool {
        self.base_url == crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL
    }

    /// True when this entry stores a Google OAuth credential bundle for the
    /// `gemini` CLI (encrypted `GeminiOAuthCredential` JSON in `key`) rather
    /// than a plain API key.
    pub fn is_gemini_oauth(&self) -> bool {
        self.base_url == crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL
    }

    /// True when this entry is any of the multi-account OAuth variants
    /// (Codex/Claude/Gemini) — used by callers that share the same
    /// "OAuth entries lack a REST endpoint / are tied to a specific CLI"
    /// semantics.
    pub fn is_any_oauth(&self) -> bool {
        self.is_codex_oauth() || self.is_claude_oauth() || self.is_gemini_oauth()
    }

    pub fn oauth_tool_hint(&self) -> &'static str {
        if self.is_claude_oauth() {
            "aivo run claude"
        } else if self.is_codex_oauth() {
            "aivo run codex"
        } else if self.is_gemini_oauth() {
            "aivo run gemini"
        } else {
            "aivo run <tool>"
        }
    }

    /// Short "why you can't use this key here" hint for pickers (e.g. ``needs
    /// `aivo run claude` ``). Returns `None` for non-OAuth keys.
    pub fn oauth_run_requirement(&self) -> Option<&'static str> {
        if self.is_claude_oauth() {
            Some("needs `aivo run claude`")
        } else if self.is_codex_oauth() {
            Some("needs `aivo run codex`")
        } else if self.is_gemini_oauth() {
            Some("needs `aivo run gemini`")
        } else {
            None
        }
    }

    /// "Claude Code" / "Codex ChatGPT" / "Gemini", or generic "OAuth" for
    /// non-OAuth keys so callers can unconditionally use it in messages
    /// guarded by `is_any_oauth`.
    pub fn oauth_kind_label(&self) -> &'static str {
        if self.is_claude_oauth() {
            "Claude Code"
        } else if self.is_codex_oauth() {
            "Codex ChatGPT"
        } else if self.is_gemini_oauth() {
            "Gemini"
        } else {
            "OAuth"
        }
    }

    /// True when this entry is a GitHub Copilot device-token login.
    pub fn is_copilot(&self) -> bool {
        crate::services::provider_profile::is_copilot_base(&self.base_url)
    }

    /// Returns a display label for credentials the user cannot retype (OAuth
    /// bundles, Copilot device tokens). Used by inspection UIs to avoid
    /// echoing live access/refresh tokens that have no copy-paste use.
    pub fn credential_label(&self) -> Option<&'static str> {
        if self.is_claude_oauth() {
            Some("<Claude OAuth>")
        } else if self.is_codex_oauth() {
            Some("<Codex OAuth>")
        } else if self.is_gemini_oauth() {
            Some("<Gemini OAuth>")
        } else if self.is_copilot() {
            Some("<Copilot>")
        } else {
            None
        }
    }
}

/// Per-directory, per-tool start records. Outer key = cwd, inner key = tool name.
pub type DirectoryStartsMap = HashMap<String, HashMap<String, DirectoryStartRecord>>;

/// Global last-used key/tool/model selection. Same shape as DirectoryStartRecord.
pub type LastSelection = DirectoryStartRecord;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectoryStartRecord {
    #[serde(rename = "keyId")]
    pub key_id: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct UsageCounter {
    #[serde(rename = "promptTokens", default, skip_serializing_if = "is_zero")]
    pub prompt_tokens: u64,
    #[serde(rename = "completionTokens", default, skip_serializing_if = "is_zero")]
    pub completion_tokens: u64,
    #[serde(
        rename = "cacheReadInputTokens",
        default,
        skip_serializing_if = "is_zero"
    )]
    pub cache_read_input_tokens: u64,
    #[serde(
        rename = "cacheCreationInputTokens",
        default,
        skip_serializing_if = "is_zero"
    )]
    pub cache_creation_input_tokens: u64,
    #[serde(rename = "totalTokens", default, skip_serializing_if = "is_zero")]
    pub total_tokens: u64,
    /// Per-tool selection counts (only populated in key_usage entries).
    #[serde(rename = "perTool", default, skip_serializing_if = "HashMap::is_empty")]
    pub per_tool: HashMap<String, u64>,
    /// Per-model total token counts (only populated in key_usage entries).
    #[serde(
        rename = "perModelTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_tokens: HashMap<String, u64>,
}

impl UsageCounter {
    fn add_tokens(
        &mut self,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt_tokens);
        self.completion_tokens = self.completion_tokens.saturating_add(completion_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .saturating_add(cache_read_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(cache_creation_input_tokens);
        self.total_tokens = self
            .total_tokens
            .saturating_add(prompt_tokens.saturating_add(completion_tokens));
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct UsageStats {
    #[serde(
        rename = "keyUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub key_usage: HashMap<String, UsageCounter>,
    #[serde(
        rename = "toolCounts",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub tool_counts: HashMap<String, u64>,
    #[serde(
        rename = "modelUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub model_usage: HashMap<String, UsageCounter>,
}

impl UsageStats {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }

    /// Removes stats linked to a key by subtracting its known contributions from globals.
    /// Uses subtraction instead of recomputing to preserve legacy global data that
    /// predates per-key model/tool tracking.
    pub(crate) fn remove_key(&mut self, key_id: &str) {
        let Some(removed) = self.key_usage.remove(key_id) else {
            return;
        };
        for (tool, count) in &removed.per_tool {
            if let Some(tc) = self.tool_counts.get_mut(tool) {
                *tc = tc.saturating_sub(*count);
                if *tc == 0 {
                    self.tool_counts.remove(tool);
                }
            }
        }
        for (model, tok) in &removed.per_model_tokens {
            if let Some(mu) = self.model_usage.get_mut(model) {
                mu.total_tokens = mu.total_tokens.saturating_sub(*tok);
                if mu.total_tokens == 0 {
                    self.model_usage.remove(model);
                }
            }
        }
    }

    pub(crate) fn record_selection(&mut self, key_id: &str, tool: &str, _model: Option<&str>) {
        let key_stats = self.key_usage.entry(key_id.to_string()).or_default();
        *key_stats.per_tool.entry(tool.to_string()).or_default() += 1;

        let tool_count = self.tool_counts.entry(tool.to_string()).or_default();
        *tool_count = tool_count.saturating_add(1);
        // Model is recorded in record_tokens only when tokens are produced,
        // to avoid counting invalid/alias model names.
    }

    pub(crate) fn record_tokens(
        &mut self,
        key_id: &str,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) {
        let key_stats = self.key_usage.entry(key_id.to_string()).or_default();
        key_stats.add_tokens(
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        );

        let total = prompt_tokens.saturating_add(completion_tokens);
        if let Some(model) = model.filter(|value| !value.trim().is_empty() && total > 0) {
            *key_stats
                .per_model_tokens
                .entry(model.to_string())
                .or_default() += total;
            let model_stats = self.model_usage.entry(model.to_string()).or_default();
            model_stats.total_tokens = model_stats.total_tokens.saturating_add(total);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageAttachment {
    pub name: String,
    pub mime_type: String,
    pub storage: AttachmentStorage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttachmentStorage {
    Inline { data: String },
    FileRef { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredChatMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<MessageAttachment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatSessionState {
    #[serde(rename = "sessionId", default = "default_chat_session_id")]
    pub session_id: String,
    #[serde(rename = "keyId")]
    pub key_id: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub cwd: String,
    pub model: String,
    /// Raw encrypted blob. Call `decrypt_messages()` to get the actual messages.
    #[serde(deserialize_with = "deserialize_messages_field")]
    pub messages: String,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    #[serde(rename = "createdAt", default)]
    pub created_at: String,
}

/// Deserializes the `messages` field, handling both the legacy array format and the current
/// encrypted string format. Legacy sessions stored messages as a JSON array; they are
/// re-encrypted on the fly so the field always holds an encrypted string after loading.
fn deserialize_messages_field<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;
    match value {
        // Current format: already an encrypted string
        Value::String(s) => Ok(s),
        // Legacy format: plain JSON array of {role, content} objects — re-encrypt it
        Value::Array(_) => {
            let json = serde_json::to_string(&value).map_err(D::Error::custom)?;
            encrypt(&json).map_err(D::Error::custom)
        }
        other => Err(D::Error::custom(format!(
            "expected string or array for messages, got {}",
            other
        ))),
    }
}

impl ChatSessionState {
    /// Returns the number of messages. Returns 0 if empty or on decryption error.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn message_count(&self) -> usize {
        self.decrypt_messages().map(|v| v.len()).unwrap_or(0)
    }

    /// Decrypts and returns the stored messages.
    pub fn decrypt_messages(&self) -> Result<Vec<StoredChatMessage>> {
        if self.messages.is_empty() {
            return Ok(vec![]);
        }
        let json = decrypt(&self.messages)?;
        serde_json::from_str(&json).context("Failed to parse stored messages")
    }
}

/// Lightweight session metadata used in the index (no message content).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionIndex {
    pub entries: Vec<SessionIndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionIndexEntry {
    pub session_id: String,
    pub key_id: String,
    pub base_url: String,
    pub cwd: String,
    pub model: String,
    pub updated_at: String,
    pub created_at: String,
    pub title: String,
    pub preview: String,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn default_chat_session_id() -> String {
    "legacy".to_string()
}

/// Stored configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredConfig {
    #[serde(rename = "api_keys", default)]
    pub api_keys: Vec<ApiKey>,
    #[serde(rename = "active_key_id")]
    pub active_key_id: Option<String>,
    #[serde(
        rename = "chat_models",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub chat_models: HashMap<String, String>,
    /// Legacy field — read from old configs but never written back.
    /// Replaced by `last_selection` (global single record).
    #[serde(
        rename = "directory_starts",
        default,
        skip_serializing,
        deserialize_with = "deserialize_directory_starts"
    )]
    pub directory_starts: DirectoryStartsMap,
    #[serde(
        rename = "stats",
        default,
        skip_serializing_if = "UsageStats::is_empty"
    )]
    pub stats: UsageStats,
    /// Model aliases (e.g. "fast" -> "claude-haiku-4-5")
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub aliases: HashMap<String, String>,
    /// Global last-used key/tool/model selection.
    #[serde(
        rename = "last_selection",
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_last_selection"
    )]
    pub last_selection: Option<LastSelection>,
    /// Legacy field — read from old configs but never written back.
    /// Sessions are now stored in individual files under sessions/.
    #[serde(rename = "chat_sessions", default, skip_serializing)]
    pub chat_sessions: HashMap<String, ChatSessionState>,
    /// Set to true when the user manually removes the aivo-starter key.
    /// Prevents auto-recreation until the user explicitly re-adds it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub starter_key_dismissed: bool,
}

/// Deserialize directory_starts supporting both legacy flat format and new nested format.
/// Legacy: `{ "/path": { "keyId": ..., "tool": "claude", ... } }`
/// New:    `{ "/path": { "claude": { "keyId": ..., ... }, "codex": { ... } } }`
fn deserialize_directory_starts<'de, D>(deserializer: D) -> Result<DirectoryStartsMap, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde::de::Error;
    use serde_json::Value;

    let raw: HashMap<String, Value> = HashMap::deserialize(deserializer)?;
    let mut result = DirectoryStartsMap::new();

    for (cwd, value) in raw {
        match value {
            Value::Object(map) => {
                // Check if this looks like a flat DirectoryStartRecord (has "keyId" field)
                if map.contains_key("keyId") {
                    // Legacy format: single record
                    let record: DirectoryStartRecord =
                        serde_json::from_value(Value::Object(map)).map_err(D::Error::custom)?;
                    let mut tools = HashMap::new();
                    tools.insert(record.tool.clone(), record);
                    result.insert(cwd, tools);
                } else {
                    // New format: tool name → record
                    let tools: HashMap<String, DirectoryStartRecord> =
                        serde_json::from_value(Value::Object(map)).map_err(D::Error::custom)?;
                    result.insert(cwd, tools);
                }
            }
            _ => continue, // skip malformed entries
        }
    }

    Ok(result)
}

/// Deserialize last_selection supporting both the new global format and legacy per-directory format.
/// New:    `{ "keyId": ..., "tool": "claude", ... }` (single record)
/// Legacy: `{ "/path": { "keyId": ..., ... }, "/other": { ... } }` (per-directory map → pick most recent)
fn deserialize_last_selection<'de, D>(deserializer: D) -> Result<Option<LastSelection>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    use serde_json::Value;

    let value = Option::<Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };

    match value {
        Value::Object(ref map) if map.contains_key("keyId") => {
            // New format: a single DirectoryStartRecord
            let record: DirectoryStartRecord =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            Ok(Some(record))
        }
        Value::Object(map) => {
            // Legacy format: HashMap<String, LastSelection> — pick most recently updated
            let mut best: Option<DirectoryStartRecord> = None;
            for (_cwd, val) in map {
                if let Ok(record) = serde_json::from_value::<DirectoryStartRecord>(val)
                    && best
                        .as_ref()
                        .is_none_or(|b| record.updated_at > b.updated_at)
                {
                    best = Some(record);
                }
            }
            Ok(best)
        }
        _ => Ok(None),
    }
}

impl Default for StoredConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StoredConfig {
    pub fn new() -> Self {
        Self {
            api_keys: Vec::new(),
            active_key_id: None,
            chat_models: HashMap::new(),
            directory_starts: HashMap::new(),
            stats: UsageStats::default(),
            aliases: HashMap::new(),
            last_selection: None,
            chat_sessions: HashMap::new(),
            starter_key_dismissed: false,
        }
    }
}

// ── Shared infrastructure ─────────────────────────────────────────────────────

#[cfg(any(unix, windows))]
pub(crate) struct ConfigLockGuard {
    _file: std::fs::File,
}

#[cfg(not(any(unix, windows)))]
pub(crate) struct ConfigLockGuard;

#[cfg(unix)]
impl Drop for ConfigLockGuard {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        // SAFETY: the file descriptor remains valid for the lifetime of the guard.
        unsafe {
            libc::flock(self._file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(windows)]
impl Drop for ConfigLockGuard {
    fn drop(&mut self) {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::UnlockFile;

        // SAFETY: the handle stays valid for the guard lifetime; UnlockFile is safe to call
        // on a handle previously locked with LockFileEx.
        unsafe {
            UnlockFile(self._file.as_raw_handle(), 0, 0, u32::MAX, u32::MAX);
        }
    }
}

impl ConfigLockGuard {
    pub(crate) fn acquire(lock_path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .with_context(|| format!("Failed to open lock file: {:?}", lock_path))?;

        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;

            loop {
                // SAFETY: the file descriptor stays open for the guard lifetime.
                let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
                if rc == 0 {
                    break;
                }

                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::Interrupted {
                    return Err(err)
                        .with_context(|| format!("Failed to acquire lock: {:?}", lock_path));
                }
            }

            Ok(ConfigLockGuard { _file: file })
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = file;
            Ok(ConfigLockGuard)
        }

        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Foundation::BOOL;
            use windows_sys::Win32::Storage::FileSystem::{LOCKFILE_EXCLUSIVE_LOCK, LockFileEx};
            use windows_sys::Win32::System::IO::OVERLAPPED;

            let handle = file.as_raw_handle();
            let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
            // SAFETY: handle is valid; we own `file` for the guard's lifetime.
            let rc: BOOL = unsafe {
                LockFileEx(
                    handle,
                    LOCKFILE_EXCLUSIVE_LOCK,
                    0,
                    u32::MAX,
                    u32::MAX,
                    &mut overlapped,
                )
            };
            if rc == 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("Failed to acquire lock: {:?}", lock_path));
            }
            Ok(ConfigLockGuard { _file: file })
        }
    }
}

/// Shared configuration I/O context used by all sub-stores.
#[derive(Debug, Clone)]
pub(crate) struct ConfigContext {
    pub(crate) config_path: PathBuf,
    pub(crate) config_dir: PathBuf,
}

impl ConfigContext {
    pub(crate) fn acquire_config_lock(&self) -> Result<ConfigLockGuard> {
        if !self.config_dir.as_os_str().is_empty() {
            std::fs::create_dir_all(&self.config_dir).with_context(|| {
                format!("Failed to create config directory: {:?}", self.config_dir)
            })?;
        }
        ConfigLockGuard::acquire(&self.config_dir.join("config.lock"))
    }

    /// Saves config to the config file.
    /// Keys must already be encrypted before calling this.
    /// Uses atomic write (write to temp file then rename) to prevent corruption.
    pub(crate) async fn save_raw(&self, config: &StoredConfig) -> Result<()> {
        tokio::fs::create_dir_all(&self.config_dir)
            .await
            .with_context(|| format!("Failed to create config directory: {:?}", self.config_dir))?;

        let data = serde_json::to_string_pretty(config).context("Failed to serialize config")?;
        atomic_write_secure(&self.config_path, data.into_bytes()).await
    }

    /// Loads config from the config file. Keys remain encrypted;
    /// use `decrypt_key_secret` on individual keys that need plaintext access.
    pub(crate) async fn load(&self) -> Result<StoredConfig> {
        let data = match tokio::fs::read_to_string(&self.config_path).await {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(StoredConfig::new());
            }
            Err(e) => return Err(e.into()),
        };

        match serde_json::from_str(&data) {
            Ok(p) => Ok(p),
            Err(e) => Err(anyhow::anyhow!(
                "config file is corrupted and cannot be read: {e}"
            )),
        }
    }
}

// ── SessionStore facade ───────────────────────────────────────────────────────

/// SessionStore manages API key persistence in ~/.config/aivo/config.json
#[derive(Debug, Clone)]
pub struct SessionStore {
    ctx: ConfigContext,
    api_keys: ApiKeyStore,
    sessions: ChatSessionStore,
    stats: UsageStatsStore,
    last_sel: LastSelectionStore,
    logs: LogStore,
}

impl SessionStore {
    pub fn new() -> Self {
        let config_dir = system_env::home_dir()
            .map(|p| p.join(".config").join("aivo"))
            .unwrap_or_else(|| PathBuf::from(".config/aivo"));
        let config_path = config_dir.join("config.json");
        Self::from_ctx(ConfigContext {
            config_path,
            config_dir,
        })
    }

    /// Creates a new SessionStore with a custom config path (for testing)
    #[allow(dead_code)]
    pub fn with_path(config_path: PathBuf) -> Self {
        let config_dir = config_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        Self::from_ctx(ConfigContext {
            config_path,
            config_dir,
        })
    }

    fn from_ctx(ctx: ConfigContext) -> Self {
        Self {
            api_keys: ApiKeyStore { ctx: ctx.clone() },
            sessions: ChatSessionStore { ctx: ctx.clone() },
            stats: UsageStatsStore::new(ctx.clone()),
            last_sel: LastSelectionStore { ctx: ctx.clone() },
            logs: LogStore::new(ctx.config_dir.clone()),
            ctx,
        }
    }

    // ── Config I/O ────────────────────────────────────────────────────────

    /// Loads config from the config file. Keys remain encrypted.
    pub async fn load(&self) -> Result<StoredConfig> {
        self.ctx.load().await
    }

    /// Gets the config path
    #[allow(dead_code)]
    pub fn get_config_path(&self) -> &PathBuf {
        &self.ctx.config_path
    }

    pub fn logs(&self) -> LogStore {
        self.logs.clone()
    }

    // ── API Key management (delegated to ApiKeyStore) ─────────────────────

    /// Adds a new API key with an optional explicit Claude protocol.
    pub async fn add_key_with_protocol(
        &self,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<String> {
        self.api_keys
            .add_key_with_protocol(name, base_url, claude_protocol, key)
            .await
    }

    /// Gets all API keys without decrypting secrets.
    pub async fn get_keys(&self) -> Result<Vec<ApiKey>> {
        self.api_keys.get_keys().await
    }

    /// Decrypts a single key's secret in place.
    pub fn decrypt_key_secret(key: &mut ApiKey) -> Result<()> {
        ApiKeyStore::decrypt_key_secret(key)
    }

    /// Gets a specific API key by ID with its secret decrypted.
    pub async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        self.api_keys.get_key_by_id(id).await
    }

    /// Deletes an API key by ID
    pub async fn delete_key(&self, id: &str) -> Result<bool> {
        let deleted = self.api_keys.delete_key(id).await?;
        if deleted {
            let _ = self.sessions.remove_sessions_for_key(id).await;
        }
        Ok(deleted)
    }

    /// Updates an existing API key's fields by ID. Returns false if not found.
    pub async fn update_key(
        &self,
        id: &str,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<bool> {
        let (found, base_url_changed) = self
            .api_keys
            .update_key(id, name, base_url, claude_protocol, key)
            .await?;
        if found && base_url_changed {
            let _ = self.sessions.remove_sessions_for_key(id).await;
        }
        Ok(found)
    }

    pub async fn set_key_claude_protocol(
        &self,
        id: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_claude_protocol(id, claude_protocol)
            .await
    }

    pub async fn set_key_gemini_protocol(
        &self,
        id: &str,
        gemini_protocol: Option<GeminiProviderProtocol>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_gemini_protocol(id, gemini_protocol)
            .await
    }

    pub async fn set_key_responses_api_supported(
        &self,
        id: &str,
        responses_api_supported: Option<bool>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_responses_api_supported(id, responses_api_supported)
            .await
    }

    pub async fn set_key_codex_mode(
        &self,
        id: &str,
        codex_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.api_keys.set_key_codex_mode(id, codex_mode).await
    }

    pub async fn set_key_opencode_mode(
        &self,
        id: &str,
        opencode_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.api_keys.set_key_opencode_mode(id, opencode_mode).await
    }

    /// Sets the currently active API key
    pub async fn set_active_key(&self, id: &str) -> Result<()> {
        self.api_keys.set_active_key(id).await
    }

    /// Resolves an API key by ID or name, decrypting only the matched key's secret.
    pub async fn resolve_key_by_id_or_name(&self, id_or_name: &str) -> Result<ApiKey> {
        self.api_keys.resolve_key_by_id_or_name(id_or_name).await
    }

    /// See `ApiKeyStore::find_keys_by_id_or_name`.
    pub async fn find_keys_by_id_or_name(&self, id_or_name: &str) -> Result<Vec<ApiKey>> {
        self.api_keys.find_keys_by_id_or_name(id_or_name).await
    }

    /// Gets the currently active API key with its secret decrypted.
    pub async fn get_active_key(&self) -> Result<Option<ApiKey>> {
        self.api_keys.get_active_key().await
    }

    /// Ensures the aivo starter key exists in the config.
    /// Creates it if missing, does NOT change the active key.
    /// Respects the dismissed flag — returns None if the user previously removed it.
    /// Returns `(key, is_new_user)` where `is_new_user` is true when no keys existed before.
    pub async fn ensure_starter_key(&self) -> Option<(ApiKey, bool)> {
        use crate::constants::{
            AIVO_STARTER_EMPTY_SECRET, AIVO_STARTER_KEY_NAME, AIVO_STARTER_MODEL,
            AIVO_STARTER_SENTINEL,
        };
        let config = self.api_keys.ctx.load().await.ok()?;
        if config.starter_key_dismissed {
            return None;
        }
        let is_new_user = config.api_keys.is_empty();
        // Check if starter key already exists
        if let Some(existing) = config
            .api_keys
            .iter()
            .find(|k| k.base_url == AIVO_STARTER_SENTINEL)
        {
            let key = self.get_key_by_id(&existing.id).await.ok().flatten()?;
            return Some((key, is_new_user));
        }
        let id = self
            .add_key_with_protocol(
                AIVO_STARTER_KEY_NAME,
                AIVO_STARTER_SENTINEL,
                None,
                AIVO_STARTER_EMPTY_SECRET,
            )
            .await
            .ok()?;
        let _ = self.set_chat_model(&id, AIVO_STARTER_MODEL).await;
        let key = self.get_key_by_id(&id).await.ok().flatten()?;
        Some((key, is_new_user))
    }

    /// Sets the starter_key_dismissed flag in the config.
    pub async fn set_starter_key_dismissed(&self, dismissed: bool) -> Result<()> {
        let _lock = self.api_keys.ctx.acquire_config_lock()?;
        let mut config = self.api_keys.ctx.load().await?;
        config.starter_key_dismissed = dismissed;
        self.api_keys.ctx.save_raw(&config).await
    }

    /// Gets all keys and the active key ID without decrypting secrets.
    pub async fn get_keys_and_active_id_info(&self) -> Result<(Vec<ApiKey>, Option<String>)> {
        self.api_keys.get_keys_and_active_id_info().await
    }

    /// Gets the active key's display metadata without decrypting secrets.
    pub async fn get_active_key_info(&self) -> Result<Option<ApiKey>> {
        self.api_keys.get_active_key_info().await
    }

    /// Gets the persisted chat model for a specific API key
    pub async fn get_chat_model(&self, key_id: &str) -> Result<Option<String>> {
        self.api_keys.get_chat_model(key_id).await
    }

    /// Saves the chat model for a specific API key
    pub async fn set_chat_model(&self, key_id: &str, model: &str) -> Result<()> {
        self.api_keys.set_chat_model(key_id, model).await
    }

    // ── Last selection (delegated to LastSelectionStore) ───────────────────

    pub async fn get_last_selection(&self) -> Result<Option<LastSelection>> {
        self.last_sel.get().await
    }

    pub async fn set_last_selection(
        &self,
        key: &ApiKey,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        self.last_sel.set(key, tool, model).await
    }

    #[allow(dead_code)]
    pub async fn clear_last_selection(&self) -> Result<()> {
        self.last_sel.clear().await
    }

    // ── Usage stats (delegated to UsageStatsStore) ────────────────────────

    pub async fn load_stats(&self) -> Result<UsageStats> {
        self.stats.load().await
    }

    #[allow(dead_code)]
    pub async fn clear_stats(&self) -> Result<()> {
        self.stats.clear().await
    }

    pub async fn remove_key_stats(&self, key_id: &str) -> Result<()> {
        self.stats.remove_key(key_id).await
    }

    pub async fn record_selection(
        &self,
        key_id: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        self.stats.record_selection(key_id, tool, model).await
    }

    pub async fn record_tokens(
        &self,
        key_id: &str,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) -> Result<()> {
        self.stats
            .record_tokens(
                key_id,
                model,
                prompt_tokens,
                completion_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
            )
            .await
    }

    // ── Chat sessions (delegated to ChatSessionStore) ─────────────────────

    #[allow(dead_code)]
    pub fn session_file_path(&self, session_id: &str) -> PathBuf {
        self.sessions.session_file_path(session_id)
    }

    pub async fn get_chat_session(&self, session_id: &str) -> Result<Option<ChatSessionState>> {
        self.sessions.get_chat_session(session_id).await
    }

    pub async fn list_chat_sessions(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
    ) -> Result<Vec<SessionIndexEntry>> {
        self.sessions
            .list_chat_sessions(key_id, base_url, cwd)
            .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn save_chat_session_with_id(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
        session_id: &str,
        model: &str,
        messages: &[StoredChatMessage],
        title: &str,
        preview: &str,
    ) -> Result<()> {
        self.sessions
            .save_chat_session_with_id(
                key_id, base_url, cwd, session_id, model, messages, title, preview,
            )
            .await
    }

    pub async fn delete_chat_session(&self, session_id: &str) -> Result<bool> {
        self.sessions.delete_chat_session(session_id).await
    }

    pub async fn count_chat_sessions(&self) -> u64 {
        self.sessions.count_chat_sessions().await
    }

    /// Removes session files for all sessions belonging to a key.
    #[allow(dead_code)]
    pub async fn remove_sessions_for_key(&self, key_id: &str) -> Result<()> {
        self.sessions.remove_sessions_for_key(key_id).await
    }

    // ── Model aliases ─────────────────────────────────────────────────────

    /// Returns all model aliases.
    pub async fn get_aliases(&self) -> Result<HashMap<String, String>> {
        let config = self.ctx.load().await?;
        Ok(config.aliases)
    }

    /// Sets a model alias. Returns the previous value if it existed.
    pub async fn set_alias(&self, name: String, model: String) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let prev = config.aliases.insert(name, model);
        self.ctx.save_raw(&config).await?;
        Ok(prev)
    }

    /// Removes a model alias. Returns the removed value if it existed.
    pub async fn remove_alias(&self, name: &str) -> Result<Option<String>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let removed = config.aliases.remove(name);
        if removed.is_some() {
            self.ctx.save_raw(&config).await?;
        }
        Ok(removed)
    }

    /// Resolves a model name through aliases, with cycle detection.
    /// Returns the final resolved model name.
    pub async fn resolve_alias(&self, model: &str) -> Result<String> {
        let aliases = self.get_aliases().await?;
        let mut current = model.to_string();
        let mut seen = std::collections::HashSet::new();
        while let Some(target) = aliases.get(&current) {
            if !seen.insert(current.clone()) {
                anyhow::bail!("circular alias detected: {}", model);
            }
            current = target.clone();
        }
        Ok(current)
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::api_key_store::{KEY_ID_ALPHABET, KEY_ID_LENGTH};
    use tempfile::TempDir;

    #[test]
    fn is_claude_oauth_tracks_sentinel() {
        let k = ApiKey {
            id: "x".into(),
            name: "".into(),
            base_url: "claude-oauth".into(),
            claude_protocol: None,
            gemini_protocol: None,
            responses_api_supported: None,
            codex_mode: None,
            opencode_mode: None,
            pi_mode: None,
            key: Zeroizing::new("{}".into()),
            created_at: Utc::now().to_rfc3339(),
        };
        assert!(k.is_claude_oauth());
        assert!(!k.is_codex_oauth());
    }

    #[tokio::test]
    async fn test_save_load_empty() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let config = store.load().await.unwrap();
        assert!(config.api_keys.is_empty());
        assert!(config.active_key_id.is_none());
    }

    #[tokio::test]
    async fn test_key_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // Add a key
        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test123")
            .await
            .unwrap();
        assert_eq!(id.len(), 3);

        // Verify it was saved
        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "my-key");
        assert_eq!(keys[0].base_url, "http://localhost:8080");
        assert_eq!(keys[0].claude_protocol, None);

        // Set as active
        store.set_active_key(&id).await.unwrap();
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().id, id);

        // Delete the key
        assert!(store.delete_key(&id).await.unwrap());
        let keys = store.get_keys().await.unwrap();
        assert!(keys.is_empty());
    }

    #[tokio::test]
    async fn test_key_encryption_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-secret-12345")
            .await
            .unwrap();

        // Verify the file contains encrypted key (v4 marker)
        let file_content = tokio::fs::read_to_string(&config_path).await.unwrap();
        assert!(file_content.contains("enc4:"));
        assert!(!file_content.contains("sk-secret-12345"));

        // Verify we can still read back the decrypted key
        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.key.as_str(), "sk-secret-12345");
    }

    #[tokio::test]
    async fn test_delete_active_key_clears_selection() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        // Delete the active key
        store.delete_key(&id).await.unwrap();

        // Active key should be cleared
        let active = store.get_active_key().await.unwrap();
        assert!(active.is_none());
    }

    #[tokio::test]
    async fn test_resolve_key_by_id() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name(&id).await.unwrap();
        assert_eq!(resolved.id, id);
        assert_eq!(resolved.name, "my-key");
    }

    #[tokio::test]
    async fn test_resolve_key_by_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("my-key", "http://localhost:8080", None, "sk-test")
            .await
            .unwrap();

        let resolved = store.resolve_key_by_id_or_name("my-key").await.unwrap();
        assert_eq!(resolved.id, id);
    }

    #[tokio::test]
    async fn test_resolve_key_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let result = store.resolve_key_by_id_or_name("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_find_keys_by_id_or_name_returns_all_matches() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id1 = store
            .add_key_with_protocol("dup", "http://localhost:8080", None, "sk-1")
            .await
            .unwrap();
        let id2 = store
            .add_key_with_protocol("dup", "http://localhost:9090", None, "sk-2")
            .await
            .unwrap();
        store
            .add_key_with_protocol("unique", "http://localhost:7070", None, "sk-3")
            .await
            .unwrap();

        // Name with multiple matches → all returned, decrypted.
        let dup_matches = store.find_keys_by_id_or_name("dup").await.unwrap();
        assert_eq!(dup_matches.len(), 2);
        let ids: Vec<_> = dup_matches.iter().map(|k| k.id.as_str()).collect();
        assert!(ids.contains(&id1.as_str()) && ids.contains(&id2.as_str()));
        assert!(dup_matches.iter().all(|k| !k.key.as_str().is_empty()));

        // Unique name → single match.
        let unique_matches = store.find_keys_by_id_or_name("unique").await.unwrap();
        assert_eq!(unique_matches.len(), 1);

        // Exact ID → single match regardless of name collisions.
        let by_id = store.find_keys_by_id_or_name(&id1).await.unwrap();
        assert_eq!(by_id.len(), 1);
        assert_eq!(by_id[0].id, id1);

        // Missing → empty Vec, not an error.
        let none = store.find_keys_by_id_or_name("nope").await.unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn test_resolve_key_ambiguous_name() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        store
            .add_key_with_protocol("same-name", "http://localhost:8080", None, "sk-test1")
            .await
            .unwrap();
        store
            .add_key_with_protocol("same-name", "http://localhost:9090", None, "sk-test2")
            .await
            .unwrap();

        let result = store.resolve_key_by_id_or_name("same-name").await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Multiple keys found")
        );
    }

    #[tokio::test]
    async fn test_load_corrupted_config_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(&config_path, b"not valid json {{{")
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        let result = store.load().await;
        assert!(result.is_err(), "expected Err on corrupted config, got Ok");
    }

    #[tokio::test]
    async fn test_decrypt_returns_error_on_invalid_encrypted_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        // "enc:" prefix triggers decryption; the payload is not valid ciphertext
        let bad_config = r#"{"api_keys":[{"id":"aaaa","name":"test","baseUrl":"http://example.com","key":"enc:notvalidbase64!!!","createdAt":"2024-01-01T00:00:00Z"}],"active_key_id":"aaaa"}"#;
        tokio::fs::write(&config_path, bad_config.as_bytes())
            .await
            .unwrap();
        let store = SessionStore::with_path(config_path);
        // load() succeeds — keys remain encrypted in memory
        let config = store.load().await.unwrap();
        assert_eq!(config.api_keys.len(), 1);
        // Decryption fails when we try to access the secret
        let mut key = config.api_keys[0].clone();
        let result = SessionStore::decrypt_key_secret(&mut key);
        assert!(
            result.is_err(),
            "expected Err on invalid encrypted key, got Ok"
        );
    }

    #[tokio::test]
    async fn test_update_key_fields() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("original", "http://localhost:8080", None, "sk-old")
            .await
            .unwrap();

        let updated = store
            .update_key(
                &id,
                "renamed",
                "https://new.example.com",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-new",
            )
            .await
            .unwrap();
        assert!(updated);

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.name, "renamed");
        assert_eq!(key.base_url, "https://new.example.com");
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
        assert_eq!(key.key.as_str(), "sk-new");
        assert_eq!(key.id, id);
    }

    #[test]
    fn test_api_key_display_name_falls_back_to_id() {
        let key = ApiKey::new_with_protocol(
            "a2b".to_string(),
            String::new(),
            "https://example.com".to_string(),
            None,
            "sk-test".to_string(),
        );

        assert_eq!(key.display_name(), "a2b");
    }

    #[tokio::test]
    async fn test_update_key_not_found_returns_false() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let updated = store
            .update_key("nonexistent", "name", "http://example.com", None, "sk-key")
            .await
            .unwrap();
        assert!(!updated);
    }

    #[tokio::test]
    async fn test_update_key_preserves_created_at() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        let before = store.get_key_by_id(&id).await.unwrap().unwrap();

        store
            .update_key(&id, "new-name", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        let after = store.get_key_by_id(&id).await.unwrap().unwrap();

        assert_eq!(before.created_at, after.created_at);
    }

    #[tokio::test]
    async fn test_record_stats_and_chat_session_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        let id = store
            .add_key_with_protocol("orig", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        store
            .record_selection(&id, "chat", Some("gpt-4o"))
            .await
            .unwrap();
        store
            .record_tokens(&id, Some("gpt-4o"), 10, 5, 90, 15)
            .await
            .unwrap();
        store
            .save_chat_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "legacy",
                "gpt-4o",
                &[StoredChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                "hello",
                "hello",
            )
            .await
            .unwrap();

        let stats = store.load_stats().await.unwrap();
        assert_eq!(stats.tool_counts.get("chat"), Some(&1));
        assert_eq!(
            stats
                .model_usage
                .get("gpt-4o")
                .map(|usage| usage.total_tokens),
            Some(15)
        );

        let session = store.get_chat_session("legacy").await.unwrap().unwrap();
        assert_eq!(session.message_count(), 1);
        assert_eq!(session.session_id, "legacy");

        store
            .save_chat_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "session-2",
                "gpt-4o-mini",
                &[StoredChatMessage {
                    role: "user".to_string(),
                    content: "second".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                "second",
                "second",
            )
            .await
            .unwrap();

        let sessions = store
            .list_chat_sessions(&id, "http://localhost", "/tmp/demo")
            .await
            .unwrap();
        assert_eq!(sessions.len(), 2);
        assert!(
            sessions
                .iter()
                .any(|session| session.session_id == "session-2")
        );

        // Session content should NOT appear in config.json (it lives in session files)
        let raw = tokio::fs::read_to_string(&config_path).await.unwrap();
        assert!(!raw.contains("\"hello\""));
        // Session file should exist and contain encrypted (not plaintext) content
        let session_path = store.session_file_path("legacy");
        let session_raw = tokio::fs::read_to_string(&session_path).await.unwrap();
        assert!(!session_raw.contains("\"hello\""));
    }

    #[tokio::test]
    async fn test_add_key_with_claude_protocol_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol(
                "minimax",
                "https://api.minimax.io/anthropic",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-test",
            )
            .await
            .unwrap();

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
    }

    #[tokio::test]
    async fn test_generated_key_id_excludes_ambiguous_characters() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert_eq!(id.len(), KEY_ID_LENGTH);
        assert!(!id.contains('0'));
        assert!(!id.contains('1'));
        assert!(!id.contains('l'));
        assert!(!id.contains('o'));
        assert!(id.chars().all(|c| KEY_ID_ALPHABET.contains(&(c as u8))));
    }

    #[tokio::test]
    async fn test_set_key_gemini_protocol_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_gemini_protocol(&id, Some(GeminiProviderProtocol::Google))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.gemini_protocol, Some(GeminiProviderProtocol::Google));
    }

    #[tokio::test]
    async fn test_set_key_codex_mode_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_codex_mode(&id, Some(OpenAICompatibilityMode::Router))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.codex_mode, Some(OpenAICompatibilityMode::Router));
    }

    #[tokio::test]
    async fn test_set_key_claude_protocol_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_claude_protocol(&id, Some(ClaudeProviderProtocol::Anthropic))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
    }

    #[tokio::test]
    async fn test_set_key_opencode_mode_updates_existing_key() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_opencode_mode(&id, Some(OpenAICompatibilityMode::Router))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.opencode_mode, Some(OpenAICompatibilityMode::Router));
    }

    #[tokio::test]
    async fn test_load_legacy_config_without_claude_protocol() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let plaintext = encrypt("sk-test").unwrap();
        let legacy_config = format!(
            r#"{{"api_keys":[{{"id":"aaaa","name":"legacy","baseUrl":"http://example.com","key":"{}","createdAt":"2024-01-01T00:00:00Z"}}],"active_key_id":"aaaa"}}"#,
            plaintext
        );
        tokio::fs::write(&config_path, legacy_config.as_bytes())
            .await
            .unwrap();

        let store = SessionStore::with_path(config_path);
        let key = store.get_active_key().await.unwrap().unwrap();
        assert_eq!(key.name, "legacy");
        assert_eq!(key.claude_protocol, None);
        assert_eq!(key.key.as_str(), "sk-test");
    }

    #[test]
    fn test_chat_session_messages_migration_from_legacy_array() {
        // Simulate a config.json written by the old code: messages is a JSON array
        let json = r#"{
            "sessionId": "sess1",
            "keyId": "key1",
            "baseUrl": "https://api.example.com",
            "cwd": "/tmp",
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"}
            ],
            "updatedAt": "2024-01-01T00:00:00Z"
        }"#;

        let session: ChatSessionState =
            serde_json::from_str(json).expect("should migrate legacy array");

        // After migration the field should be an encrypted string
        assert!(
            is_encrypted(&session.messages),
            "messages should be re-encrypted"
        );

        // And decryption should yield the original messages
        let messages = session
            .decrypt_messages()
            .expect("should decrypt migrated messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "hi there");
    }

    #[test]
    fn test_chat_session_messages_current_format_roundtrip() {
        let msgs = vec![
            StoredChatMessage {
                role: "user".into(),
                content: "ping".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                role: "assistant".into(),
                content: "pong".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
        ];
        let json = serde_json::to_string(&msgs).unwrap();
        let encrypted = encrypt(&json).unwrap();

        let session_json = format!(
            r#"{{"sessionId":"s","keyId":"k","baseUrl":"u","cwd":"/","model":"m","messages":{},"updatedAt":"2024-01-01T00:00:00Z"}}"#,
            serde_json::to_string(&encrypted).unwrap()
        );

        let session: ChatSessionState = serde_json::from_str(&session_json).unwrap();
        let decoded = session.decrypt_messages().unwrap();
        assert_eq!(decoded, msgs);
    }

    #[test]
    fn remove_key_subtracts_from_globals() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 50);
        stats.tool_counts.insert("codex".to_string(), 30);
        stats.model_usage.insert(
            "gpt-4o".to_string(),
            UsageCounter {
                total_tokens: 6000,
                ..Default::default()
            },
        );

        // Key to remove has partial contributions
        let mut entry = UsageCounter {
            prompt_tokens: 500,
            completion_tokens: 300,
            total_tokens: 800,
            ..Default::default()
        };
        entry.per_tool.insert("claude".to_string(), 5);
        entry.per_model_tokens.insert("gpt-4o".to_string(), 1000);
        stats.key_usage.insert("key1".to_string(), entry);

        stats.remove_key("key1");

        assert_eq!(stats.tool_counts.get("claude"), Some(&45));
        assert_eq!(stats.tool_counts.get("codex"), Some(&30));
        assert_eq!(stats.model_usage.get("gpt-4o").unwrap().total_tokens, 5000);
        assert!(!stats.key_usage.contains_key("key1"));
    }

    #[test]
    fn remove_key_noop_for_missing_key() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 10);
        stats.remove_key("nonexistent");
        assert_eq!(stats.tool_counts.get("claude"), Some(&10));
    }

    #[test]
    fn remove_key_cleans_up_zeroed_entries() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 5);
        stats.model_usage.insert(
            "gpt-4o".to_string(),
            UsageCounter {
                total_tokens: 1000,
                ..Default::default()
            },
        );

        let mut entry = UsageCounter::default();
        entry.per_tool.insert("claude".to_string(), 5);
        entry.per_model_tokens.insert("gpt-4o".to_string(), 1000);
        stats.key_usage.insert("key1".to_string(), entry);

        stats.remove_key("key1");

        // Zeroed tool count should be removed
        assert!(!stats.tool_counts.contains_key("claude"));
        // Zeroed model usage should be removed
        assert!(!stats.model_usage.contains_key("gpt-4o"));
    }

    /// Full config snapshot from v0.12 era: all legacy fields present, optional ApiKey
    /// fields absent, legacy flat directory_starts, legacy per-directory last_selection,
    /// legacy inline chat_sessions with plaintext messages. If this test ever breaks after
    /// a schema change, real users' configs will too.
    #[test]
    fn load_v012_full_config_snapshot() {
        let json = r#"{
            "api_keys": [
                {
                    "id": "a1b2c3",
                    "name": "work-key",
                    "baseUrl": "https://api.anthropic.com",
                    "key": "sk-ant-test-key",
                    "createdAt": "2025-06-01T10:00:00Z"
                },
                {
                    "id": "d4e5f6",
                    "name": "openrouter",
                    "baseUrl": "https://openrouter.ai/api/v1",
                    "key": "sk-or-test-key",
                    "createdAt": "2025-07-15T12:00:00Z"
                }
            ],
            "active_key_id": "a1b2c3",
            "chat_models": {
                "a1b2c3": "claude-sonnet-4-6",
                "d4e5f6": "gpt-4o"
            },
            "directory_starts": {
                "/home/user/project": {
                    "keyId": "a1b2c3",
                    "baseUrl": "https://api.anthropic.com",
                    "tool": "claude",
                    "model": "claude-sonnet-4-6",
                    "updatedAt": "2025-08-01T00:00:00Z"
                }
            },
            "last_selection": {
                "/home/user/project": {
                    "keyId": "a1b2c3",
                    "baseUrl": "https://api.anthropic.com",
                    "tool": "claude",
                    "model": "claude-sonnet-4-6",
                    "updatedAt": "2025-08-01T00:00:00Z"
                },
                "/home/user/other": {
                    "keyId": "d4e5f6",
                    "baseUrl": "https://openrouter.ai/api/v1",
                    "tool": "codex",
                    "model": "gpt-4o",
                    "updatedAt": "2025-09-01T00:00:00Z"
                }
            },
            "stats": {
                "toolCounts": { "claude": 42, "codex": 10 },
                "modelUsage": {
                    "claude-sonnet-4-6": { "total_tokens": 150000 }
                }
            },
            "chat_sessions": {
                "sess-legacy": {
                    "sessionId": "sess-legacy",
                    "keyId": "a1b2c3",
                    "baseUrl": "https://api.anthropic.com",
                    "cwd": "/home/user/project",
                    "model": "claude-sonnet-4-6",
                    "messages": [
                        { "role": "user", "content": "hello" },
                        { "role": "assistant", "content": "hi there" }
                    ],
                    "updatedAt": "2025-08-10T00:00:00Z"
                }
            }
        }"#;

        let config: StoredConfig = serde_json::from_str(json).unwrap();

        // API keys loaded with missing optional fields defaulting to None
        assert_eq!(config.api_keys.len(), 2);
        assert_eq!(config.api_keys[0].id, "a1b2c3");
        assert_eq!(config.api_keys[0].name, "work-key");
        assert!(config.api_keys[0].claude_protocol.is_none());
        assert!(config.api_keys[0].gemini_protocol.is_none());
        assert!(config.api_keys[0].responses_api_supported.is_none());
        assert!(config.api_keys[0].codex_mode.is_none());
        assert!(config.api_keys[0].opencode_mode.is_none());
        assert!(config.api_keys[0].pi_mode.is_none());

        // Active key preserved
        assert_eq!(config.active_key_id.as_deref(), Some("a1b2c3"));

        // Chat models preserved
        assert_eq!(
            config.chat_models.get("a1b2c3").unwrap(),
            "claude-sonnet-4-6"
        );
        assert_eq!(config.chat_models.get("d4e5f6").unwrap(), "gpt-4o");

        // Legacy flat directory_starts migrated to nested format
        let tools = config.directory_starts.get("/home/user/project").unwrap();
        assert_eq!(tools.get("claude").unwrap().key_id, "a1b2c3");

        // Legacy per-directory last_selection picked most recent entry
        let sel = config.last_selection.unwrap();
        assert_eq!(sel.key_id, "d4e5f6");
        assert_eq!(sel.tool, "codex");

        // Stats preserved
        assert_eq!(*config.stats.tool_counts.get("claude").unwrap(), 42);

        // Legacy inline chat_sessions loaded (messages auto-encrypted)
        let session = config.chat_sessions.get("sess-legacy").unwrap();
        assert_eq!(session.key_id, "a1b2c3");
        let msgs = session.decrypt_messages().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello");

        // New fields default correctly
        assert!(!config.starter_key_dismissed);
        assert!(config.aliases.is_empty());
    }

    /// Current config format: all new fields present, optional ApiKey fields populated.
    /// Guards against regressions in the latest schema.
    #[test]
    fn load_current_config_with_all_fields() {
        let json = r#"{
            "api_keys": [
                {
                    "id": "x1y2z3",
                    "name": "full-key",
                    "baseUrl": "https://api.anthropic.com",
                    "claudeProtocol": "anthropic",
                    "geminiProtocol": "openai",
                    "codexResponsesApi": true,
                    "codexMode": "router",
                    "opencodeMode": "direct",
                    "piMode": "direct",
                    "key": "sk-full-test",
                    "createdAt": "2026-01-01T00:00:00Z"
                }
            ],
            "active_key_id": "x1y2z3",
            "aliases": {
                "fast": "claude-haiku-4-5",
                "smart": "claude-sonnet-4-6"
            },
            "last_selection": {
                "keyId": "x1y2z3",
                "baseUrl": "https://api.anthropic.com",
                "tool": "claude",
                "model": "claude-sonnet-4-6",
                "updatedAt": "2026-01-15T00:00:00Z"
            },
            "starter_key_dismissed": true
        }"#;

        let config: StoredConfig = serde_json::from_str(json).unwrap();

        let key = &config.api_keys[0];
        assert_eq!(key.claude_protocol, Some(ClaudeProviderProtocol::Anthropic));
        assert_eq!(key.gemini_protocol, Some(GeminiProviderProtocol::Openai));
        assert_eq!(key.responses_api_supported, Some(true));
        assert_eq!(key.codex_mode, Some(OpenAICompatibilityMode::Router));
        assert_eq!(key.opencode_mode, Some(OpenAICompatibilityMode::Direct));
        assert_eq!(key.pi_mode, Some(OpenAICompatibilityMode::Direct));

        assert_eq!(config.aliases.get("fast").unwrap(), "claude-haiku-4-5");

        // Global last_selection (new format) loaded directly
        let sel = config.last_selection.unwrap();
        assert_eq!(sel.tool, "claude");

        assert!(config.starter_key_dismissed);
    }

    /// The legacy field name "responsesApiSupported" must still deserialize into
    /// responses_api_supported via the serde alias.
    #[test]
    fn load_legacy_responses_api_field_alias() {
        let json = r#"{
            "api_keys": [
                {
                    "id": "abc",
                    "name": "old-key",
                    "baseUrl": "https://api.openai.com/v1",
                    "responsesApiSupported": true,
                    "key": "sk-old",
                    "createdAt": "2025-05-01T00:00:00Z"
                }
            ]
        }"#;

        let config: StoredConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.api_keys[0].responses_api_supported, Some(true));
    }

    /// A minimal config with only api_keys (all other fields absent) must load
    /// without errors — this is what a first-run config looks like.
    #[test]
    fn load_minimal_config() {
        let json = r#"{ "api_keys": [] }"#;
        let config: StoredConfig = serde_json::from_str(json).unwrap();
        assert!(config.api_keys.is_empty());
        assert!(config.active_key_id.is_none());
        assert!(config.chat_models.is_empty());
        assert!(config.aliases.is_empty());
        assert!(config.last_selection.is_none());
        assert!(!config.starter_key_dismissed);
    }

    #[test]
    fn deserialize_legacy_flat_directory_starts() {
        let json = r#"{
            "api_keys": [],
            "directory_starts": {
                "/tmp/test": {
                    "keyId": "key1",
                    "baseUrl": "http://localhost",
                    "tool": "claude",
                    "model": "gpt-4o",
                    "updatedAt": "2026-01-01T00:00:00Z"
                }
            }
        }"#;
        let config: StoredConfig = serde_json::from_str(json).unwrap();
        let tools = config.directory_starts.get("/tmp/test").unwrap();
        let record = tools.get("claude").unwrap();
        assert_eq!(record.key_id, "key1");
        assert_eq!(record.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn deserialize_nested_directory_starts() {
        let json = r#"{
            "api_keys": [],
            "directory_starts": {
                "/tmp/test": {
                    "claude": {
                        "keyId": "key1",
                        "baseUrl": "http://localhost",
                        "tool": "claude",
                        "model": "gpt-4o",
                        "updatedAt": "2026-01-01T00:00:00Z"
                    },
                    "codex": {
                        "keyId": "key2",
                        "baseUrl": "http://other",
                        "tool": "codex",
                        "updatedAt": "2026-02-01T00:00:00Z"
                    }
                }
            }
        }"#;
        let config: StoredConfig = serde_json::from_str(json).unwrap();
        let tools = config.directory_starts.get("/tmp/test").unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools.get("claude").unwrap().key_id, "key1");
        assert_eq!(tools.get("codex").unwrap().key_id, "key2");
    }

    /// The custom `zeroizing_string` serde module bridges `Zeroizing<String>`
    /// to a plain JSON string. Guards against silent breakage if upstream
    /// `zeroize` derives change.
    #[test]
    fn zeroizing_string_serde_roundtrip() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct Wrap {
            #[serde(with = "super::zeroizing_string")]
            secret: Zeroizing<String>,
        }

        let original = Wrap {
            secret: Zeroizing::new("sk-secret-12345".to_string()),
        };
        let json = serde_json::to_string(&original).unwrap();
        assert_eq!(json, r#"{"secret":"sk-secret-12345"}"#);

        let decoded: Wrap = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.secret.as_str(), "sk-secret-12345");
        assert_eq!(decoded, original);
    }

    /// `ApiKey` uses `zeroizing_string` for its `key` field — verify the full
    /// struct roundtrips without exposing the secret in unexpected places.
    #[test]
    fn api_key_zeroizing_roundtrip() {
        let key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "test".to_string(),
            "http://localhost".to_string(),
            None,
            "sk-roundtrip-secret".to_string(),
        );
        let json = serde_json::to_string(&key).unwrap();
        assert!(json.contains("\"key\":\"sk-roundtrip-secret\""));

        let decoded: ApiKey = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.key.as_str(), "sk-roundtrip-secret");
        assert_eq!(decoded, key);
    }
}
