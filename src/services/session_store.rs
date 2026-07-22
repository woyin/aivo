use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use zeroize::Zeroizing;

pub use crate::services::session_crypto::{decrypt, encrypt, is_encrypted};

use crate::services::api_key_store::ApiKeyStore;
use crate::services::atomic_write::atomic_write_secure;
use crate::services::code_session_store::CodeSessionStore;
use crate::services::last_selection::LastSelectionStore;
use crate::services::log_store::LogStore;
use crate::services::route_cache::PersistedRoute;
use crate::services::usage_stats_store::UsageStatsStore;

/// Bump when one-shot migrations of routing fields on `ApiKey` are needed.
/// New keys are stamped with this version on creation; older keys are migrated
/// at launch by `launch_runtime::migrate_routing_schema_for_key`.
///
/// Version history:
///   1: clear `responses_api_supported = Some(false)` written by pre-fix
///      builds that latched on any non-200 (incl. transient 429/5xx).
///   2: replace the per-(tool, key) scalar route pins with the per-(tool, key,
///      model) `protocol_routes` map. The old scalars are folded into per-tool
///      `""` defaults (lossless under per-tool keying) and then dropped.
pub const CURRENT_ROUTING_SCHEMA_VERSION: u32 = 2;

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

impl ClaudeProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Google => "google",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GeminiProviderProtocol {
    Google,
    Openai,
    Anthropic,
}

impl GeminiProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Google => "google",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OpenAICompatibilityMode {
    Direct,
    Router,
}

impl OpenAICompatibilityMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Router => "router",
        }
    }
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
    /// Learned path variant for the Claude/Codex routers ("default" or
    /// "stripped"). Stripped wins (e.g., gateways serving `/messages`
    /// instead of `/v1/messages`) used to be relearned every launch because
    /// only the protocol bits were persisted. Stored as the same string the
    /// fallback module uses internally so future variants serialise without
    /// schema churn.
    #[serde(
        rename = "claudePathVariant",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub claude_path_variant: Option<String>,
    #[serde(
        rename = "geminiPathVariant",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub gemini_path_variant: Option<String>,
    /// Learned per-key override for the `requires_reasoning_content` quirk.
    /// `None` means "fall back to static `ProviderQuirks::for_base_url`".
    /// `Some(true)` is set when an upstream returns a parseable
    /// `reasoning_content` semantic rejection, so subsequent launches inject
    /// the strict-mode flag without needing the host to be in the static
    /// substring list. Avoids hardcoding new providers as they're added.
    #[serde(
        rename = "requiresReasoningContent",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub requires_reasoning_content: Option<bool>,
    /// Learned upstream protocol per `(tool, model)` (inner `""` = the tool's
    /// default). Replaces the five scalar pins above, which a multi-model key
    /// would thrash; those linger only so a v1 config can be migrated (schema v2).
    #[serde(
        rename = "protocolRoutes",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub protocol_routes: BTreeMap<String, BTreeMap<String, PersistedRoute>>,
    /// Schema version for one-shot migrations of routing-related fields. Bumped
    /// when older builds may have written values under buggy logic that should
    /// be cleared on first launch by a newer build. Missing/zero = legacy.
    #[serde(
        rename = "routingSchemaVersion",
        default,
        skip_serializing_if = "is_zero_u32"
    )]
    pub routing_schema_version: u32,
    #[serde(with = "zeroizing_string")]
    pub key: Zeroizing<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
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
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            protocol_routes: BTreeMap::new(),
            routing_schema_version: CURRENT_ROUTING_SCHEMA_VERSION,
            key: Zeroizing::new(key),
            created_at: Utc::now().to_rfc3339(),
        }
    }

    /// Persisted routes for one tool, for seeding that router's `RouteCache`.
    /// The built-in agent's routes were keyed under `"chat"` before the rename;
    /// when asked for `"code"` and no `"code"` entry exists yet, fall back to the
    /// legacy `"chat"` entry so a returning user keeps their learned routes.
    pub fn routes_for_tool(&self, tool: &str) -> BTreeMap<String, PersistedRoute> {
        if let Some(routes) = self.protocol_routes.get(tool) {
            return routes.clone();
        }
        if tool == "code"
            && let Some(legacy) = self.protocol_routes.get("chat")
        {
            return legacy.clone();
        }
        BTreeMap::new()
    }

    pub fn short_id(&self) -> &str {
        // Slice on a char boundary: a hand-edited config can carry a multi-byte id.
        match self.id.char_indices().nth(3) {
            Some((end, _)) => &self.id[..end],
            None => &self.id,
        }
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

    /// True when this entry is a legacy Gemini Google-OAuth credential bundle
    /// for the `gemini` CLI (encrypted token JSON in `key`). The OAuth sign-in
    /// flow has been removed; such keys are recognized only so they're redacted,
    /// excluded from exports, and rejected at launch rather than mistaken for a
    /// plain API key. See `services::gemini_oauth`.
    pub fn is_gemini_oauth(&self) -> bool {
        self.base_url == crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL
    }

    /// True when this entry stores a SuperGrok OAuth credential — a *provider*
    /// bearer usable by any coding agent, not a single-CLI credential.
    pub fn is_grok_oauth(&self) -> bool {
        self.base_url == crate::services::grok_oauth::GROK_OAUTH_SENTINEL
    }

    /// True when this entry stores a Kimi Code OAuth credential — a provider
    /// bearer like grok's.
    pub fn is_kimi_oauth(&self) -> bool {
        self.base_url == crate::services::kimi_oauth::KIMI_OAUTH_SENTINEL
    }

    /// OAuth credentials aivo can route to any coding agent via the ServeRouter
    /// (SuperGrok, Codex, Kimi), as opposed to single-CLI logins.
    pub fn is_provider_oauth(&self) -> bool {
        self.is_grok_oauth() || self.is_codex_oauth() || self.is_kimi_oauth()
    }

    /// True when this entry is any of the multi-account OAuth variants
    /// (Codex/Claude/Gemini/Grok/Kimi) — used by callers that share the same
    /// "OAuth entries lack a REST endpoint / hold a credential blob" semantics.
    pub fn is_any_oauth(&self) -> bool {
        self.is_codex_oauth()
            || self.is_claude_oauth()
            || self.is_gemini_oauth()
            || self.is_grok_oauth()
            || self.is_kimi_oauth()
    }

    pub fn oauth_tool_hint(&self) -> &'static str {
        if self.is_claude_oauth() {
            "aivo claude"
        } else if self.is_codex_oauth() {
            "aivo codex"
        } else if self.is_gemini_oauth() {
            "aivo gemini"
        } else if self.is_grok_oauth() || self.is_kimi_oauth() {
            "aivo code"
        } else {
            "aivo <tool>"
        }
    }

    /// Short "why you can't use this key here" hint for pickers (e.g. ``needs
    /// `aivo claude` ``). Returns `None` for non-OAuth keys.
    pub fn oauth_run_requirement(&self) -> Option<&'static str> {
        if self.is_claude_oauth() {
            Some("needs `aivo claude`")
        } else if self.is_gemini_oauth() {
            Some("Gemini sign-in removed — re-add with an API key")
        } else {
            // Grok/Codex are provider credentials usable by any agent; `aivo
            // codex` still prefers native launch via `is_codex_family`.
            None
        }
    }

    /// "Claude Code" / "Codex ChatGPT" / "Gemini" / "SuperGrok", or generic
    /// "OAuth" for non-OAuth keys so callers can unconditionally use it in
    /// messages guarded by `is_any_oauth`.
    pub fn oauth_kind_label(&self) -> &'static str {
        if self.is_claude_oauth() {
            "Claude Code"
        } else if self.is_codex_oauth() {
            "Codex ChatGPT"
        } else if self.is_gemini_oauth() {
            "Gemini"
        } else if self.is_grok_oauth() {
            "SuperGrok"
        } else if self.is_kimi_oauth() {
            "Kimi Code"
        } else {
            "OAuth"
        }
    }

    /// True when this entry is a GitHub Copilot device-token login.
    pub fn is_copilot(&self) -> bool {
        crate::services::provider_profile::is_copilot_base(&self.base_url)
    }

    /// True when this entry selects Cursor through the `cursor-agent` ACP
    /// provider sentinel.
    pub fn is_cursor_acp(&self) -> bool {
        crate::services::cursor_acp::is_cursor_acp_base(&self.base_url)
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
        } else if self.is_grok_oauth() {
            Some("<SuperGrok OAuth>")
        } else if self.is_kimi_oauth() {
            Some("<Kimi OAuth>")
        } else if self.is_copilot() {
            Some("<Copilot>")
        } else if self.is_cursor_acp() {
            match crate::services::cursor_acp::parse_cursor_shadow_secret(self.key.as_str()) {
                Some(s) if s.api_key.is_some() => Some("<Cursor API key>"),
                Some(_) => Some("<Cursor login>"),
                None => None,
            }
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

/// Per-model accumulator stored under `UsageCounter::per_model_usage`.
/// Mirrors the four token dimensions that providers report.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelCounter {
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
}

impl ModelCounter {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Accumulate one usage report's four token dimensions (saturating).
    fn add(&mut self, prompt: u64, completion: u64, cache_read: u64, cache_creation: u64) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(prompt);
        self.completion_tokens = self.completion_tokens.saturating_add(completion);
        self.cache_read_input_tokens = self.cache_read_input_tokens.saturating_add(cache_read);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .saturating_add(cache_creation);
    }
}

/// Per-named-subagent lifetime tally. Only delegations whose name matches a
/// discovered profile are attributed here (generic/labeled delegates are
/// excluded), so this never fills with ad-hoc task labels.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct AgentUsage {
    #[serde(default, skip_serializing_if = "is_zero")]
    pub runs: u64,
    /// Runs that finished successfully (`runs - ok_runs` = failures).
    #[serde(rename = "okRuns", default, skip_serializing_if = "is_zero")]
    pub ok_runs: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub steps: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tokens: u64,
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
    /// Per-model accumulator (only populated in key_usage entries).
    #[serde(
        rename = "perModelUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_usage: HashMap<String, ModelCounter>,
    /// Per-(tool, model) token usage. Only aivo-proxied launches that know the
    /// launching tool (plugins, chat) populate this; native CLIs read their own
    /// usage files, so they never appear here. Forward-only — old data is empty.
    #[serde(
        rename = "perToolModelUsage",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_tool_model_usage: HashMap<String, HashMap<String, ModelCounter>>,
    /// Per-named-subagent run tallies (only populated in key_usage entries).
    /// Forward-only — old stats files load with this empty.
    #[serde(
        rename = "perAgent",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_agent: HashMap<String, AgentUsage>,
    /// Legacy per-model total tokens. Read by the migration helper, never written
    /// after this version. Kept on the type for forward/backward compatibility
    /// with on-disk data recorded before the schema collapse.
    #[serde(
        rename = "perModelTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_tokens: HashMap<String, u64>,
    /// Legacy per-model prompt tokens. See `per_model_tokens` for context.
    #[serde(
        rename = "perModelPromptTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_prompt_tokens: HashMap<String, u64>,
    /// Legacy per-model completion tokens.
    #[serde(
        rename = "perModelCompletionTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_completion_tokens: HashMap<String, u64>,
    /// Legacy per-model cache-read tokens.
    #[serde(
        rename = "perModelCacheReadTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_cache_read_tokens: HashMap<String, u64>,
    /// Legacy per-model cache-creation (write) tokens.
    #[serde(
        rename = "perModelCacheCreationTokens",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub per_model_cache_creation_tokens: HashMap<String, u64>,
}

impl UsageCounter {
    /// Folds all four legacy `per_model_*` maps and the legacy `per_model_tokens`
    /// total into the canonical `per_model_usage` map, then clears the legacy maps.
    /// Idempotent: calling on already-migrated data is a no-op.
    fn migrate_legacy_per_model(&mut self) {
        if self.per_model_tokens.is_empty()
            && self.per_model_prompt_tokens.is_empty()
            && self.per_model_completion_tokens.is_empty()
            && self.per_model_cache_read_tokens.is_empty()
            && self.per_model_cache_creation_tokens.is_empty()
        {
            return;
        }
        let mut models: std::collections::HashSet<String> = std::collections::HashSet::new();
        models.extend(self.per_model_usage.keys().cloned());
        models.extend(self.per_model_tokens.keys().cloned());
        models.extend(self.per_model_prompt_tokens.keys().cloned());
        models.extend(self.per_model_completion_tokens.keys().cloned());
        models.extend(self.per_model_cache_read_tokens.keys().cloned());
        models.extend(self.per_model_cache_creation_tokens.keys().cloned());
        for model in models {
            let prompt = self
                .per_model_prompt_tokens
                .get(&model)
                .copied()
                .unwrap_or(0);
            let completion = self
                .per_model_completion_tokens
                .get(&model)
                .copied()
                .unwrap_or(0);
            let cache_read = self
                .per_model_cache_read_tokens
                .get(&model)
                .copied()
                .unwrap_or(0);
            let cache_create = self
                .per_model_cache_creation_tokens
                .get(&model)
                .copied()
                .unwrap_or(0);
            // Pre-split residue: portion of the legacy total not covered by
            // recorded prompt/completion. Fold into completion so the row
            // total stays accurate even when we don't know the input/output split.
            let split_total = prompt.saturating_add(completion);
            let legacy_total = self.per_model_tokens.get(&model).copied().unwrap_or(0);
            let residue = legacy_total.saturating_sub(split_total);
            let counter = self.per_model_usage.entry(model).or_default();
            counter.prompt_tokens = counter.prompt_tokens.saturating_add(prompt);
            counter.completion_tokens = counter
                .completion_tokens
                .saturating_add(completion)
                .saturating_add(residue);
            counter.cache_read_input_tokens =
                counter.cache_read_input_tokens.saturating_add(cache_read);
            counter.cache_creation_input_tokens = counter
                .cache_creation_input_tokens
                .saturating_add(cache_create);
        }
        self.per_model_tokens.clear();
        self.per_model_prompt_tokens.clear();
        self.per_model_completion_tokens.clear();
        self.per_model_cache_read_tokens.clear();
        self.per_model_cache_creation_tokens.clear();
    }

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

    /// Folds the four legacy `per_model_*` maps and the legacy `per_model_tokens`
    /// total on every key into the canonical `per_model_usage` field. Idempotent.
    pub(crate) fn migrate_legacy_per_model(&mut self) {
        for entry in self.key_usage.values_mut() {
            entry.migrate_legacy_per_model();
        }
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
        for (model, mc) in &removed.per_model_usage {
            let tok = mc.total_tokens();
            if let Some(mu) = self.model_usage.get_mut(model) {
                mu.total_tokens = mu.total_tokens.saturating_sub(tok);
                if mu.total_tokens == 0 {
                    self.model_usage.remove(model);
                }
            }
        }
        // Defensive: post-migration `per_model_tokens` is empty, so this is a
        // no-op on normal load paths. Kept for callers that mutate UsageStats
        // directly without going through `load_with_migration`.
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_tokens(
        &mut self,
        key_id: &str,
        tool: Option<&str>,
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
        let cache_total = cache_read_input_tokens.saturating_add(cache_creation_input_tokens);
        if let Some(model) =
            model.filter(|value| !value.trim().is_empty() && (total > 0 || cache_total > 0))
        {
            key_stats
                .per_model_usage
                .entry(model.to_string())
                .or_default()
                .add(
                    prompt_tokens,
                    completion_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                );
            // Per-tool attribution: only aivo-proxied launches that know the
            // tool (plugins, chat) supply one, so stats can attribute tokens.
            if let Some(tool) = tool.filter(|t| !t.trim().is_empty()) {
                key_stats
                    .per_tool_model_usage
                    .entry(tool.to_string())
                    .or_default()
                    .entry(model.to_string())
                    .or_default()
                    .add(
                        prompt_tokens,
                        completion_tokens,
                        cache_read_input_tokens,
                        cache_creation_input_tokens,
                    );
            }
            if total > 0 {
                let model_stats = self.model_usage.entry(model.to_string()).or_default();
                model_stats.total_tokens = model_stats.total_tokens.saturating_add(total);
            }
        }
    }

    /// Attribute one finished delegation to a named subagent profile. The caller
    /// only records rows whose delegate name matches a discovered profile.
    pub(crate) fn record_agent_run(
        &mut self,
        key_id: &str,
        agent: &str,
        ok: bool,
        steps: u64,
        tokens: u64,
    ) {
        let key_stats = self.key_usage.entry(key_id.to_string()).or_default();
        let entry = key_stats.per_agent.entry(agent.to_string()).or_default();
        entry.runs = entry.runs.saturating_add(1);
        if ok {
            entry.ok_runs = entry.ok_runs.saturating_add(1);
        }
        entry.steps = entry.steps.saturating_add(steps);
        entry.tokens = entry.tokens.saturating_add(tokens);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MessageAttachment {
    pub name: String,
    pub mime_type: String,
    pub storage: AttachmentStorage,
}

/// The persisted `aivo code` toggles, read together at startup (see
/// [`SessionStore::get_chat_toggles`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatToggles {
    pub auto_approve: bool,
    /// Whether an edit-bearing batch pauses for a diff-review card before writing
    /// (`/config`). Defaults off (opt-in).
    pub review_edits: bool,
    /// Whether the model is asked to think (and its reasoning shown, folded) — the
    /// single thinking on/off concept. Defaults on.
    pub thinking_enabled: bool,
    /// aivo's hosted web_search (`/config`). Defaults off (opt-in).
    pub web_search_enabled: bool,
    pub agent_tools_enabled: bool,
    /// Chat TUI color theme (`/config`). `None` = the user has never picked one,
    /// so startup auto-detects from the terminal background (falling back to dark);
    /// `Some` = an explicit choice that's always honored.
    pub theme: Option<ChatTheme>,
}

/// Persisted chat TUI color theme (`"theme"` in code-prefs.json).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTheme {
    #[default]
    Dark,
    Light,
}

impl ChatTheme {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Light => "light",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "dark" => Some(Self::Dark),
            "light" => Some(Self::Light),
            _ => None,
        }
    }
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
    /// Producing model (assistant turns only); absent on pre-feature sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeSessionState {
    #[serde(rename = "sessionId", default = "default_chat_session_id")]
    pub session_id: String,
    #[serde(rename = "keyId")]
    pub key_id: String,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub cwd: String,
    pub model: String,
    /// Stored as a plain JSON array. Legacy sessions hold an `enc*:` encrypted
    /// blob (machine/keyring-bound); the deserializer decrypts those on read.
    /// An undecryptable blob is a hard load error — the file stays on disk
    /// untouched rather than being clobbered by a later save.
    #[serde(deserialize_with = "deserialize_messages_field")]
    pub messages: Vec<StoredChatMessage>,
    /// Optional raw OpenAI-format conversation of the in-process agent engine
    /// (assistant `tool_calls` + `tool` results with ids), for exact resume.
    /// Absent for non-agent chats and pre-feature sessions. Legacy encrypted
    /// blobs decrypt on read; a decrypt/parse failure is treated as absent
    /// (resume falls back to the lossy text seed).
    #[serde(
        rename = "engineMessages",
        default,
        deserialize_with = "deserialize_engine_messages_field",
        skip_serializing_if = "Option::is_none"
    )]
    pub engine_messages: Option<Vec<serde_json::Value>>,
    /// Loss accounting from the foreign-transcript conversion this fork began
    /// as. Absent for native sessions and pre-feature forks.
    #[serde(
        rename = "importFidelity",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub import_fidelity: Option<crate::services::session_import::ImportFidelity>,
    /// Unfinished plan-mode snapshot, so a resume picks the plan back up.
    /// Absent once the plan is approved/discarded (and for legacy sessions).
    #[serde(rename = "planState", default, skip_serializing_if = "Option::is_none")]
    pub plan_state: Option<PlanState>,
    #[serde(rename = "updatedAt")]
    pub updated_at: String,
    #[serde(rename = "createdAt", default)]
    pub created_at: String,
}

/// Unfinished plan-mode snapshot saved with a code session: `mode` restores
/// read-only planning; `draft` re-arms the approval card / `/plan go`; `steps` is
/// the mid-execution `update_plan` checklist another session's `/plan resume` continues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanState {
    #[serde(default)]
    pub mode: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<serde_json::Value>,
}

/// Deserializes the `messages` field: a plain JSON array (current format), or a
/// legacy `enc*:` encrypted JSON string that is decrypted on read. Decrypt
/// failure is a hard error so an unreadable session is never rewritten empty.
fn deserialize_messages_field<'de, D>(deserializer: D) -> Result<Vec<StoredChatMessage>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error;
    use serde_json::Value;

    let value = Value::deserialize(deserializer)?;
    match value {
        Value::Array(_) => serde_json::from_value(value).map_err(D::Error::custom),
        Value::String(s) if s.is_empty() => Ok(vec![]),
        Value::String(s) => {
            let json = decrypt(&s).map_err(D::Error::custom)?;
            serde_json::from_str(&json).map_err(D::Error::custom)
        }
        other => Err(D::Error::custom(format!(
            "expected array or string for messages, got {}",
            other
        ))),
    }
}

/// Deserializes `engineMessages`: a plain JSON array (current format), or a
/// legacy encrypted string. Best-effort by contract — a corrupt or unreadable
/// blob degrades to `None` (lossy text resume), never an error.
fn deserialize_engine_messages_field<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<serde_json::Value>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde_json::Value;

    Ok(match Option::<Value>::deserialize(deserializer)? {
        Some(Value::Array(items)) => Some(items),
        Some(Value::String(blob)) => decrypt(&blob)
            .ok()
            .and_then(|json| serde_json::from_str(&json).ok()),
        _ => None,
    })
}

/// Legacy inline sessions, per-entry lenient: one undecryptable or malformed
/// entry must not brick the whole config (API key) load.
fn deserialize_legacy_chat_sessions<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, CodeSessionState>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let raw = HashMap::<String, serde_json::Value>::deserialize(deserializer)?;
    Ok(raw
        .into_iter()
        .filter_map(|(id, value)| serde_json::from_value(value).ok().map(|s| (id, s)))
        .collect())
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
    /// Upstream-resolved model from the provider response (e.g. `aivo/starter`
    /// resolves to `deepseek-v4-flash`). Stats prefers this over `model` so
    /// the chat per-model breakdown matches what claude-code records — and
    /// the user's typed alias is still used for display via `model`.
    /// `None` for legacy entries written before this was tracked, or when
    /// the provider didn't return a model name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub billed_model: Option<String>,
    pub updated_at: String,
    pub created_at: String,
    pub title: String,
    pub preview: String,
    /// Cumulative tokens for this session. Long TUI sessions over-attribute
    /// to a `--since` window (entire session counts if `updated_at` lands
    /// inside it); one-shots are exact.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub prompt_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub completion_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_read_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_write_tokens: u64,
    /// Cumulative estimated/reported spend, so resume doesn't lose the figure.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub cost_usd: f64,
}

/// Token usage for a single chat turn or accumulated across a session.
/// `prompt_tokens` is the total input side — cache reads/writes are a subset of
/// it (OpenAI-style); Anthropic's disjoint counts are normalized at ingestion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionTokens {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

impl SessionTokens {
    pub fn merge(self, other: SessionTokens) -> Self {
        Self {
            prompt_tokens: self.prompt_tokens.saturating_add(other.prompt_tokens),
            completion_tokens: self
                .completion_tokens
                .saturating_add(other.completion_tokens),
            cache_read_tokens: self
                .cache_read_tokens
                .saturating_add(other.cache_read_tokens),
            cache_write_tokens: self
                .cache_write_tokens
                .saturating_add(other.cache_write_tokens),
        }
    }

    pub fn total(&self) -> u64 {
        // Cache counts are ⊂ prompt_tokens — adding them would double count.
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// Chat activity inside a `--since` window: how many sessions were touched
/// and the per-model token totals across them. `total()` derives the top line
/// so it can't drift from `per_model`.
///
/// `per_model` keys are `billed_model` when known (the upstream-resolved
/// name), falling back to `model` (the user-typed alias) for legacy
/// entries — so aliases like `aivo/starter` collapse onto the upstream
/// they resolve to.
#[derive(Debug, Clone, Default)]
pub struct ChatTokenWindow {
    pub count: u64,
    pub per_model: std::collections::HashMap<String, SessionTokens>,
}

impl ChatTokenWindow {
    pub fn total(&self) -> SessionTokens {
        self.per_model
            .values()
            .fold(SessionTokens::default(), |acc, t| acc.merge(*t))
    }
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_zero_f64(value: &f64) -> bool {
    *value == 0.0
}

fn default_chat_session_id() -> String {
    "legacy".to_string()
}

/// One entry in the alias map. Model aliases stay as JSON strings so that
/// pre-Bundle configs deserialize unchanged; Bundle aliases serialize as a
/// tagged-by-shape object. `Bundle` is listed first so a JSON object hits it
/// before falling through to `Model` for any string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AliasValue {
    Bundle(BundleAlias),
    Model(String),
}

/// A preset launch — `tool` is one of the known AI tool names, and `args` is a
/// raw passthrough that gets spliced into the run argv (with conflict-filtering
/// against the user's own flags).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BundleAlias {
    pub tool: String,
    #[serde(default)]
    pub args: Vec<String>,
}

impl std::fmt::Display for AliasValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AliasValue::Model(m) => f.write_str(m),
            AliasValue::Bundle(b) => {
                f.write_str(&b.tool)?;
                for arg in &b.args {
                    write!(f, " {}", arg)?;
                }
                Ok(())
            }
        }
    }
}

/// Stored configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredConfig {
    #[serde(rename = "api_keys", default)]
    pub api_keys: Vec<ApiKey>,
    #[serde(rename = "active_key_id")]
    pub active_key_id: Option<String>,
    // `alias` reads the pre-rename `chat_models` key so a config written by an
    // older `aivo code` build still loads; new writes use `code_models`.
    #[serde(
        rename = "code_models",
        alias = "chat_models",
        default,
        skip_serializing_if = "HashMap::is_empty"
    )]
    pub code_models: HashMap<String, String>,
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
    /// Aliases. Two flavors share one namespace:
    /// - Model alias: short name → model name (e.g. "fast" → "claude-haiku-4-5"),
    ///   serialized as a JSON string for back-compat with pre-Bundle configs.
    /// - Bundle alias: short name → preset launch (tool + args), serialized as
    ///   `{"tool": "claude", "args": ["--key", "work", "--model", "fast"]}`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub aliases: HashMap<String, AliasValue>,
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
    #[serde(
        rename = "chat_sessions",
        default,
        skip_serializing,
        deserialize_with = "deserialize_legacy_chat_sessions"
    )]
    pub chat_sessions: HashMap<String, CodeSessionState>,
    /// Set to true when the user manually removes the aivo-starter key.
    /// Prevents auto-recreation until the user explicitly re-adds it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub starter_key_dismissed: bool,
    /// Legacy seed for the `/skills` opt-outs. Storage moved to code-prefs.json
    /// (`disabledSkills`) so the high-frequency config.json writers (`run`/`start`/
    /// `serve`, key edits, route learning) — and any older aivo binary that predates
    /// the field — can't drop it on a cross-version round trip. `migrate_disabled_
    /// toggles` copies this into chat-prefs at startup; it stays serialized here as a
    /// read fallback until then, but chat-prefs is authoritative once present.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_skills: Vec<String>,
    /// Legacy seed for the `/mcp` opt-outs. Moved to code-prefs.json
    /// (`disabledMcpServers`) for the same reason as `disabled_skills`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_mcp_servers: Vec<String>,
    /// Catch-all for config keys this binary doesn't recognize (e.g. a field a
    /// newer aivo added). Without it serde silently drops unknown keys on load, so
    /// the next save erases them — letting an older binary that shares config.json
    /// wipe a newer one's settings. Flattened so unknown keys round-trip verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
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
            code_models: HashMap::new(),
            directory_starts: HashMap::new(),
            stats: UsageStats::default(),
            aliases: HashMap::new(),
            last_selection: None,
            chat_sessions: HashMap::new(),
            starter_key_dismissed: false,
            disabled_skills: Vec::new(),
            disabled_mcp_servers: Vec::new(),
            extra: BTreeMap::new(),
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

/// Cap on lock waits — a wedged holder must not hang every aivo process;
/// legitimate holders take milliseconds.
const LOCK_WAIT_MAX: std::time::Duration = if cfg!(feature = "__internal_test_fast_crypto") {
    std::time::Duration::from_millis(200)
} else {
    std::time::Duration::from_secs(5)
};
const LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(50);

impl ConfigLockGuard {
    pub(crate) fn acquire(lock_path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path)
            .with_context(|| format!("Failed to open lock file: {:?}", lock_path))?;
        let deadline = std::time::Instant::now() + LOCK_WAIT_MAX;

        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;

            loop {
                // SAFETY: the file descriptor stays open for the guard lifetime.
                let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                if rc == 0 {
                    break;
                }
                let err = std::io::Error::last_os_error();
                let busy = err.raw_os_error() == Some(libc::EWOULDBLOCK)
                    || err.kind() == std::io::ErrorKind::Interrupted;
                if !busy {
                    return Err(err)
                        .with_context(|| format!("Failed to acquire lock: {:?}", lock_path));
                }
                if std::time::Instant::now() >= deadline {
                    anyhow::bail!(
                        "lock {:?} is held by another process (wedged aivo?)",
                        lock_path
                    );
                }
                std::thread::sleep(LOCK_POLL);
            }

            Ok(ConfigLockGuard { _file: file })
        }

        #[cfg(not(any(unix, windows)))]
        {
            let _ = (file, deadline);
            Ok(ConfigLockGuard)
        }

        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::Foundation::{BOOL, ERROR_LOCK_VIOLATION};
            use windows_sys::Win32::Storage::FileSystem::{
                LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY, LockFileEx,
            };
            use windows_sys::Win32::System::IO::OVERLAPPED;

            let handle = file.as_raw_handle();
            loop {
                let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
                // SAFETY: handle is valid; we own `file` for the guard's lifetime.
                let rc: BOOL = unsafe {
                    LockFileEx(
                        handle,
                        LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                        0,
                        u32::MAX,
                        u32::MAX,
                        &mut overlapped,
                    )
                };
                if rc != 0 {
                    break;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(ERROR_LOCK_VIOLATION as i32) {
                    return Err(err)
                        .with_context(|| format!("Failed to acquire lock: {:?}", lock_path));
                }
                if std::time::Instant::now() >= deadline {
                    anyhow::bail!(
                        "lock {:?} is held by another process (wedged aivo?)",
                        lock_path
                    );
                }
                std::thread::sleep(LOCK_POLL);
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
            crate::services::atomic_write::ensure_private_dir_blocking(&self.config_dir)?;
        }
        ConfigLockGuard::acquire(&self.config_dir.join("config.lock"))
    }

    /// Saves config to the config file.
    /// Keys must already be encrypted before calling this.
    /// Uses atomic write (write to temp file then rename) to prevent corruption.
    pub(crate) async fn save_raw(&self, config: &StoredConfig) -> Result<()> {
        crate::services::atomic_write::ensure_private_dir(&self.config_dir).await?;

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

/// Collect the string elements of a JSON array, skipping non-string entries.
fn json_str_array(arr: &[serde_json::Value]) -> Vec<String> {
    arr.iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect()
}

// ── SessionStore facade ───────────────────────────────────────────────────────

/// SessionStore manages API key persistence in ~/.config/aivo/config.json
#[derive(Debug, Clone)]
pub struct SessionStore {
    ctx: ConfigContext,
    api_keys: ApiKeyStore,
    sessions: CodeSessionStore,
    stats: UsageStatsStore,
    last_sel: LastSelectionStore,
    logs: LogStore,
}

impl SessionStore {
    pub fn new() -> Self {
        let config_dir = crate::services::paths::config_dir();
        let config_path = config_dir.join("config.json");
        Self::from_ctx(ConfigContext {
            config_path,
            config_dir,
        })
    }

    /// Creates a new SessionStore with a custom config path (for testing)
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
            sessions: CodeSessionStore { ctx: ctx.clone() },
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
    pub fn get_config_path(&self) -> &PathBuf {
        &self.ctx.config_path
    }

    /// Returns the directory holding `config.json` and other aivo state
    /// (`logs/`, `sessions/`, the audio cache, etc.).
    pub fn config_dir(&self) -> &std::path::Path {
        &self.ctx.config_dir
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

    /// See [`ApiKeyStore::export_keys`].
    pub async fn export_keys(
        &self,
        ids: Option<&[String]>,
        include_starter: bool,
        include_oauth: bool,
    ) -> Result<(
        Vec<ApiKey>,
        crate::services::api_key_store::ExportFilterReport,
    )> {
        self.api_keys
            .export_keys(ids, include_starter, include_oauth)
            .await
    }

    /// See [`ApiKeyStore::import_keys`].
    pub async fn import_keys(
        &self,
        records: Vec<ApiKey>,
        policy: crate::services::api_key_store::ImportPolicy,
    ) -> Result<crate::services::api_key_store::ImportReport> {
        self.api_keys.import_keys(records, policy).await
    }

    /// Decrypts a single key's secret in place. No-op if already decrypted.
    /// Pairs with the `_info` lookup variants for deferred decryption.
    pub fn decrypt_key_secret(key: &mut ApiKey) -> Result<()> {
        ApiKeyStore::decrypt_key_secret(key)
    }

    /// Gets a specific API key by ID with its secret decrypted.
    pub async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        self.api_keys.get_key_by_id(id).await
    }

    /// Like `get_key_by_id` but skips secret decryption.
    pub async fn get_key_by_id_info(&self, id: &str) -> Result<Option<ApiKey>> {
        self.api_keys.get_key_by_id_info(id).await
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

    pub async fn set_key_routing_schema_version(
        &self,
        id: &str,
        routing_schema_version: u32,
    ) -> Result<bool> {
        self.api_keys
            .set_key_routing_schema_version(id, routing_schema_version)
            .await
    }

    pub async fn set_key_claude_path_variant(
        &self,
        id: &str,
        claude_path_variant: Option<String>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_claude_path_variant(id, claude_path_variant)
            .await
    }

    pub async fn set_key_gemini_path_variant(
        &self,
        id: &str,
        gemini_path_variant: Option<String>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_gemini_path_variant(id, gemini_path_variant)
            .await
    }

    pub async fn set_key_requires_reasoning_content(
        &self,
        id: &str,
        requires_reasoning_content: Option<bool>,
    ) -> Result<bool> {
        self.api_keys
            .set_key_requires_reasoning_content(id, requires_reasoning_content)
            .await
    }

    /// See [`ApiKeyStore::clear_protocol_routes`].
    pub async fn clear_protocol_routes(&self, id: &str) -> Result<bool> {
        self.api_keys.clear_protocol_routes(id).await
    }

    /// See [`ApiKeyStore::merge_routes`].
    pub async fn merge_routes(
        &self,
        id: &str,
        tool: &str,
        routes: &[(String, PersistedRoute)],
    ) -> Result<bool> {
        self.api_keys.merge_routes(id, tool, routes).await
    }

    /// See [`ApiKeyStore::migrate_key_to_routes_v2`].
    pub async fn migrate_key_to_routes_v2(
        &self,
        id: &str,
        migrated: BTreeMap<String, BTreeMap<String, PersistedRoute>>,
        version: u32,
    ) -> Result<bool> {
        self.api_keys
            .migrate_key_to_routes_v2(id, migrated, version)
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

    /// See `ApiKeyStore::find_keys_by_id_or_name_info`.
    pub async fn find_keys_by_id_or_name_info(&self, id_or_name: &str) -> Result<Vec<ApiKey>> {
        self.api_keys.find_keys_by_id_or_name_info(id_or_name).await
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
        // Common path: starter already exists. Reuse the entry we just
        // loaded — no second `load()`, no PBKDF2 (callers only read `.id`).
        if let Some(existing) = config
            .api_keys
            .iter()
            .find(|k| k.base_url == AIVO_STARTER_SENTINEL)
        {
            return Some((existing.clone(), is_new_user));
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
        let _ = self.set_code_model(&id, AIVO_STARTER_MODEL).await;
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

    /// Skill names the user has turned off in `/skills`. Backed by code-prefs.json
    /// (`disabledSkills`) — see [`Self::get_disabled_list`].
    pub async fn get_disabled_skills(&self) -> Result<Vec<String>> {
        Ok(self.get_disabled_list("disabledSkills").await)
    }

    /// Enable or disable one skill by name (idempotent). Disabled skills are kept
    /// out of the agent's system prompt + `skill` tool.
    pub async fn set_skill_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        self.set_disabled_list("disabledSkills", name, enabled)
            .await
    }

    /// MCP server names the user has turned off in `/mcp`. Backed by code-prefs.json
    /// (`disabledMcpServers`) — see [`Self::get_disabled_list`].
    pub async fn get_disabled_mcp_servers(&self) -> Result<Vec<String>> {
        Ok(self.get_disabled_list("disabledMcpServers").await)
    }

    /// Enable or disable one MCP server by name (idempotent). Disabled servers are
    /// skipped at connect time so their tools aren't offered to the agent.
    pub async fn set_mcp_server_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        self.set_disabled_list("disabledMcpServers", name, enabled)
            .await
    }

    /// Individual MCP tools turned off in `/mcp` (Ctrl+T), as qualified
    /// `mcp__server__tool` names — the same form the engine advertises, so the
    /// filter needs no reverse parsing.
    pub async fn get_disabled_mcp_tools(&self) -> Result<Vec<String>> {
        Ok(self.get_disabled_list("disabledMcpTools").await)
    }

    /// Enable or disable one MCP tool by its qualified name (idempotent).
    pub async fn set_mcp_tool_enabled(&self, qualified: &str, enabled: bool) -> Result<()> {
        self.set_disabled_list("disabledMcpTools", qualified, enabled)
            .await
    }

    /// A "disabled names" list (skills or MCP servers) read from code-prefs.json by
    /// `key`. chat-prefs is an opaque JSON map that round-trips verbatim through any
    /// build (even an older chat that doesn't know the key), so — unlike the typed
    /// config.json — these can't be dropped on a cross-version write. Falls back to
    /// the legacy config.json field until the first toggle migrates the value over;
    /// once the chat-prefs key exists it is authoritative and the stale config.json
    /// field is ignored (and dropped on the next config write — it's `skip_serializing`).
    async fn get_disabled_list(&self, key: &str) -> Vec<String> {
        match self
            .read_code_prefs()
            .await
            .get(key)
            .and_then(serde_json::Value::as_array)
        {
            Some(arr) => json_str_array(arr),
            None => self.legacy_disabled(key).await,
        }
    }

    /// Toggle one `name` in the chat-prefs `key` list (idempotent), seeding from the
    /// legacy config.json field on the first write so no earlier opt-out is lost.
    /// Writes only code-prefs.json, leaving config.json (and the key store) untouched.
    async fn set_disabled_list(&self, key: &str, name: &str, enabled: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        let mut names = match prefs.get(key).and_then(serde_json::Value::as_array) {
            Some(arr) => json_str_array(arr),
            None => self.legacy_disabled(key).await,
        };
        names.retain(|n| n != name);
        if !enabled {
            names.push(name.to_string());
        }
        prefs.insert(
            key.to_string(),
            serde_json::Value::Array(names.into_iter().map(serde_json::Value::String).collect()),
        );
        self.write_code_prefs(&prefs).await
    }

    /// The pre-migration value of a `disabled*` list still living in config.json.
    /// Empty when config is absent/unreadable.
    async fn legacy_disabled(&self, key: &str) -> Vec<String> {
        let Ok(config) = self.load().await else {
            return Vec::new();
        };
        match key {
            "disabledSkills" => config.disabled_skills,
            "disabledMcpServers" => config.disabled_mcp_servers,
            _ => Vec::new(),
        }
    }

    /// One-time move of the `/skills` + `/mcp` opt-outs from the legacy config.json
    /// fields into code-prefs.json. Run once at chat startup so an existing opt-out
    /// reaches the clobber-proof store promptly — before an older aivo binary (which
    /// drops the unknown config.json field) gets a chance to wipe it. Idempotent:
    /// a chat-prefs key that already exists wins and is never overwritten, and a
    /// legacy field that's empty/absent is skipped, so there's nothing to do on the
    /// common path. Best-effort; a write failure just defers to the read fallback.
    pub async fn migrate_disabled_toggles(&self) {
        let prefs = self.read_code_prefs().await;
        let need_skills = !prefs.contains_key("disabledSkills");
        let need_mcp = !prefs.contains_key("disabledMcpServers");
        if !need_skills && !need_mcp {
            return;
        }
        let Ok(config) = self.load().await else {
            return;
        };
        let migrate_skills = need_skills && !config.disabled_skills.is_empty();
        let migrate_mcp = need_mcp && !config.disabled_mcp_servers.is_empty();
        if !migrate_skills && !migrate_mcp {
            return;
        }
        let mut prefs = prefs;
        let to_json = |names: Vec<String>| {
            serde_json::Value::Array(names.into_iter().map(serde_json::Value::String).collect())
        };
        if migrate_skills {
            prefs.insert("disabledSkills".into(), to_json(config.disabled_skills));
        }
        if migrate_mcp {
            prefs.insert(
                "disabledMcpServers".into(),
                to_json(config.disabled_mcp_servers),
            );
        }
        let _ = self.write_code_prefs(&prefs).await;
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
    pub async fn get_code_model(&self, key_id: &str) -> Result<Option<String>> {
        self.api_keys.get_code_model(key_id).await
    }

    /// Saves the chat model for a specific API key
    pub async fn set_code_model(&self, key_id: &str, model: &str) -> Result<()> {
        self.api_keys.set_code_model(key_id, model).await
    }

    /// Read a single boolean from code-prefs.json, falling back to `default` when
    /// the file or key is missing/unreadable. Shares the read+parse with
    /// `read_code_prefs` so each bool getter is one line and they can't drift.
    async fn get_chat_pref_bool(&self, key: &str, default: bool) -> bool {
        self.read_code_prefs()
            .await
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(default)
    }

    /// The persisted global chat auto-approve toggle (remembered across `aivo
    /// chat` sessions). Stored in its own small file rather than the encrypted
    /// keys config, so it never risks the key store. Missing/unreadable → off.
    pub async fn get_chat_auto_approve(&self) -> bool {
        self.get_chat_pref_bool("autoApprove", false).await
    }

    /// Persist the global chat auto-approve toggle, preserving any sibling prefs
    /// (e.g. the project-MCP allow-list). Best-effort; written atomically via a
    /// temp file + rename so a crash can't truncate it.
    pub async fn set_chat_auto_approve(&self, on: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert("autoApprove".into(), serde_json::Value::Bool(on));
        self.write_code_prefs(&prefs).await
    }

    /// The persisted edit-review toggle (code-prefs.json). Missing → off (opt-in).
    pub async fn get_chat_review_edits(&self) -> bool {
        self.get_chat_pref_bool("reviewEdits", false).await
    }

    /// Persist the global edit-review toggle, preserving sibling prefs. Best-effort.
    pub async fn set_chat_review_edits(&self, on: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert("reviewEdits".into(), serde_json::Value::Bool(on));
        self.write_code_prefs(&prefs).await
    }

    /// The persisted thinking on/off toggle (remembered across `aivo code`
    /// sessions, set in the `/config` overlay). Shares code-prefs.json with the
    /// auto-approve flag. Defaults to ON — reasoning is high-signal feedback
    /// during the silent gaps before tool calls. Falls back to the legacy
    /// `showThinking` key so a pre-rename preference still applies.
    pub async fn get_chat_thinking_enabled(&self) -> bool {
        let prefs = self.read_code_prefs().await;
        prefs
            .get("thinkingEnabled")
            .or_else(|| prefs.get("showThinking"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true)
    }

    /// Persist the thinking toggle, preserving any sibling prefs. Best-effort,
    /// written atomically (same path/permissions as the auto-approve flag).
    pub async fn set_chat_thinking_enabled(&self, on: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert("thinkingEnabled".into(), serde_json::Value::Bool(on));
        self.write_code_prefs(&prefs).await
    }

    /// Persist the web_search toggle, preserving sibling prefs (atomic write).
    pub async fn set_chat_web_search_enabled(&self, on: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert("useWebSearch".into(), serde_json::Value::Bool(on));
        self.write_code_prefs(&prefs).await
    }

    pub async fn set_chat_agent_tools_enabled(&self, on: bool) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert("agentTools".into(), serde_json::Value::Bool(on));
        self.write_code_prefs(&prefs).await
    }

    /// The persisted `/effort` reasoning level for `model` (remembered across
    /// `aivo code` sessions). Effort levels are model-specific, so they're stored
    /// per-model under `reasoningEffort: {<model>: <level>}` in code-prefs.json.
    /// `None` when unset — the engine then uses the model default. The caller
    /// still re-validates against the model's current level list.
    pub async fn get_chat_reasoning_effort(&self, model: &str) -> Option<String> {
        self.read_code_prefs()
            .await
            .get("reasoningEffort")
            .and_then(|v| v.get(model))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    }

    /// Persist `model`'s `/effort` level (or clear it with `None`), preserving
    /// other models' levels and sibling prefs. Best-effort, written atomically.
    pub async fn set_chat_reasoning_effort(&self, model: &str, level: Option<&str>) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        let entry = prefs
            .entry("reasoningEffort")
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        // Replace a legacy non-object value (the earlier global string form).
        if !entry.is_object() {
            *entry = serde_json::Value::Object(serde_json::Map::new());
        }
        let map = entry.as_object_mut().expect("just ensured object");
        match level {
            Some(l) => {
                map.insert(model.to_string(), serde_json::Value::String(l.into()));
            }
            None => {
                map.remove(model);
            }
        }
        self.write_code_prefs(&prefs).await
    }

    /// Both `aivo code` toggles in a single read of code-prefs.json, so startup
    /// doesn't open+parse the same file twice.
    pub async fn get_chat_toggles(&self) -> ChatToggles {
        let prefs = self.read_code_prefs().await;
        let bool_or = |key: &str, default: bool| {
            prefs
                .get(key)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(default)
        };
        // Current key, with legacy `showThinking` fallback (default on). See
        // `get_chat_thinking_enabled`.
        let thinking_enabled = prefs
            .get("thinkingEnabled")
            .or_else(|| prefs.get("showThinking"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        ChatToggles {
            auto_approve: bool_or("autoApprove", false),
            review_edits: bool_or("reviewEdits", false),
            thinking_enabled,
            web_search_enabled: bool_or("useWebSearch", false),
            agent_tools_enabled: bool_or("agentTools", true),
            // Absent or unparseable → None, so startup auto-detects.
            theme: prefs
                .get("theme")
                .and_then(|v| v.as_str())
                .and_then(ChatTheme::parse),
        }
    }

    /// Persist the chat TUI color theme (`dark` / `light`).
    pub async fn set_chat_theme(&self, theme: ChatTheme) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        prefs.insert(
            "theme".to_string(),
            serde_json::Value::String(theme.as_str().to_string()),
        );
        self.write_code_prefs(&prefs).await
    }

    /// code-prefs.json as a JSON object (empty when absent/unparseable), for a
    /// read-modify-write that preserves keys other than the one being changed.
    /// Falls back to the pre-rename `chat-prefs.json` so existing users keep
    /// their toggles/allow-lists on first launch after the `chat`→`code` rename.
    async fn read_code_prefs(&self) -> serde_json::Map<String, serde_json::Value> {
        let dir = self.config_dir();
        let read_obj = |path: std::path::PathBuf| async move {
            tokio::fs::read(&path)
                .await
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v.as_object().cloned())
        };
        if let Some(obj) = read_obj(crate::services::paths::code_prefs(dir)).await {
            return obj;
        }
        read_obj(crate::services::paths::chat_prefs_legacy(dir))
            .await
            .unwrap_or_default()
    }

    async fn write_code_prefs(
        &self,
        prefs: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<()> {
        let dir = self.config_dir();
        let path = crate::services::paths::code_prefs(dir);
        crate::services::atomic_write::ensure_private_dir(
            path.parent().unwrap_or(std::path::Path::new(".")),
        )
        .await?;
        let data = serde_json::to_vec_pretty(prefs)?;
        // code-prefs holds the project-MCP allow-list and the auto-approve flag —
        // security-relevant, so write it 0600 (and via a random-suffix temp that
        // leaves no orphan on a crash), like every other config write. A plain
        // `tokio::fs::write` would land at the process umask (typically 0644).
        atomic_write_secure(&path, data).await
    }

    /// Whether the user granted "always" approval to spawn the project
    /// `.mcp.json` stdio servers in `dir_key` (a canonical repo path) with the
    /// exact server set hashed into `digest`. These run arbitrary local commands,
    /// so they're gated until the user opts in once — and the approval is bound to
    /// the server content, so a later `.mcp.json` change (e.g. a `git pull` that
    /// swaps in a different command) no longer matches and re-prompts. Stored in
    /// code-prefs.json (non-secret, never the encrypted key store). Missing → false.
    /// Legacy bare-string entries (dir only, no digest) never match — they expire
    /// to one re-approval rather than silently honoring a changed config.
    pub async fn get_project_mcp_approved(&self, dir_key: &str, digest: &str) -> bool {
        self.read_code_prefs()
            .await
            .get("approvedProjectMcpDirs")
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter().any(|d| {
                    d.get("dir").and_then(|x| x.as_str()) == Some(dir_key)
                        && d.get("sha256").and_then(|x| x.as_str()) == Some(digest)
                })
            })
    }

    /// Remember an "always" approval for `dir_key`'s project MCP servers, bound to
    /// `digest`. Replaces any prior approval for the same dir so a re-approval
    /// after a config change supersedes (rather than accumulates) the old digest.
    pub async fn set_project_mcp_approved(&self, dir_key: &str, digest: &str) -> Result<()> {
        let mut prefs = self.read_code_prefs().await;
        let arr = prefs
            .entry("approvedProjectMcpDirs")
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        if let Some(list) = arr.as_array_mut() {
            list.retain(|d| d.get("dir").and_then(|x| x.as_str()) != Some(dir_key));
            list.push(serde_json::json!({ "dir": dir_key, "sha256": digest }));
        }
        self.write_code_prefs(&prefs).await
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

    pub async fn clear_last_selection(&self) -> Result<()> {
        self.last_sel.clear().await
    }

    // ── Usage stats (delegated to UsageStatsStore) ────────────────────────

    pub async fn load_stats(&self) -> Result<UsageStats> {
        self.stats.load().await
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

    #[allow(clippy::too_many_arguments)]
    pub async fn record_tokens(
        &self,
        key_id: &str,
        tool: Option<&str>,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) -> Result<()> {
        self.stats
            .record_tokens(
                key_id,
                tool,
                model,
                prompt_tokens,
                completion_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
            )
            .await
    }

    pub async fn record_agent_run(
        &self,
        key_id: &str,
        agent: &str,
        ok: bool,
        steps: u64,
        tokens: u64,
    ) -> Result<()> {
        self.stats
            .record_agent_run(key_id, agent, ok, steps, tokens)
            .await
    }

    // ── Chat sessions (delegated to CodeSessionStore) ─────────────────────

    pub fn session_file_path(&self, session_id: &str) -> PathBuf {
        self.sessions.session_file_path(session_id)
    }

    /// Per-session directory for durable agent artifacts (sub-agent reports, job logs).
    pub fn session_artifacts_dir(&self, session_id: &str) -> PathBuf {
        self.sessions.session_artifacts_dir(session_id)
    }

    pub async fn get_code_session(&self, session_id: &str) -> Result<Option<CodeSessionState>> {
        self.sessions.get_code_session(session_id).await
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

    pub async fn find_chat_session_near(
        &self,
        cwd: &str,
        key_id: Option<&str>,
        ts: chrono::DateTime<chrono::Utc>,
        max_skew_secs: i64,
    ) -> Result<Option<String>> {
        self.sessions
            .find_chat_session_near(cwd, key_id, ts, max_skew_secs)
            .await
    }

    pub async fn code_session_ids_on_disk(&self) -> std::collections::HashSet<String> {
        self.sessions.session_ids_on_disk().await
    }

    /// Cumulative tokens stored for a session's index entry (zero if unknown).
    /// Re-seeds the chat TUI's running total on resume so continued turns keep
    /// accumulating rather than overwriting the prior total.
    pub async fn chat_session_tokens(&self, session_id: &str) -> SessionTokens {
        self.sessions.chat_session_tokens(session_id).await
    }

    /// Index billing snapshot: cumulative tokens, upstream billed model, and spend.
    pub async fn chat_session_billing(
        &self,
        session_id: &str,
    ) -> (SessionTokens, Option<String>, f64) {
        self.sessions.chat_session_billing(session_id).await
    }

    pub async fn all_chat_sessions(&self) -> Result<Vec<SessionIndexEntry>> {
        self.sessions.all_chat_sessions().await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn save_code_session_with_id(
        &self,
        key_id: &str,
        base_url: &str,
        cwd: &str,
        session_id: &str,
        model: &str,
        billed_model: Option<&str>,
        messages: &[StoredChatMessage],
        title: &str,
        preview: &str,
        tokens: SessionTokens,
        cost_usd: f64,
    ) -> Result<()> {
        self.sessions
            .save_code_session_with_id(
                key_id,
                base_url,
                cwd,
                session_id,
                model,
                billed_model,
                messages,
                title,
                preview,
                tokens,
                cost_usd,
            )
            .await
    }

    /// Refresh only the durable agent-engine transcript of an existing chat
    /// session (for exact resume of the in-process agent). Best-effort.
    pub async fn save_agent_messages(
        &self,
        session_id: &str,
        engine_messages: &[serde_json::Value],
    ) -> Result<()> {
        self.sessions
            .save_agent_messages(session_id, engine_messages)
            .await
    }

    /// Refresh (or with `None` clear) the session's unfinished-plan snapshot.
    /// No-op when the session file doesn't exist yet — a plan alone doesn't
    /// create a session. Best-effort like `save_agent_messages`.
    pub async fn set_plan_state(&self, session_id: &str, plan: Option<&PlanState>) -> Result<()> {
        self.sessions.set_plan_state(session_id, plan).await
    }

    /// Write-once; no-op when the session is absent or already stamped.
    pub async fn set_import_fidelity(
        &self,
        session_id: &str,
        fidelity: &crate::services::session_import::ImportFidelity,
    ) -> Result<()> {
        self.sessions
            .set_import_fidelity(session_id, fidelity)
            .await
    }

    pub async fn delete_chat_session(&self, session_id: &str) -> Result<bool> {
        self.sessions.delete_chat_session(session_id).await
    }

    pub async fn count_chat_sessions(&self) -> u64 {
        self.sessions.count_chat_sessions().await
    }

    pub async fn aggregate_chat_window_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> ChatTokenWindow {
        self.sessions.aggregate_chat_window_since(cutoff).await
    }

    /// Removes session files for all sessions belonging to a key.
    pub async fn remove_sessions_for_key(&self, key_id: &str) -> Result<()> {
        self.sessions.remove_sessions_for_key(key_id).await
    }

    // ── Aliases ───────────────────────────────────────────────────────────

    /// Returns just the model aliases (Bundle entries filtered out). Model
    /// resolution paths use this directly so they never have to know Bundle
    /// exists.
    pub async fn get_aliases(&self) -> Result<HashMap<String, String>> {
        let config = self.ctx.load().await?;
        Ok(config
            .aliases
            .into_iter()
            .filter_map(|(k, v)| match v {
                AliasValue::Model(m) => Some((k, m)),
                AliasValue::Bundle(_) => None,
            })
            .collect())
    }

    /// Returns the full alias map — both Model and Bundle entries — for
    /// listing in the alias command.
    pub async fn list_alias_values(&self) -> Result<HashMap<String, AliasValue>> {
        let config = self.ctx.load().await?;
        Ok(config.aliases)
    }

    /// Sets a Model alias. Returns the previous value if it existed (which may
    /// have been a Bundle — the call replaces it either way).
    pub async fn set_alias(&self, name: String, model: String) -> Result<Option<AliasValue>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let prev = config.aliases.insert(name, AliasValue::Model(model));
        self.ctx.save_raw(&config).await?;
        Ok(prev)
    }

    /// Sets a Bundle alias. Returns the previous value if it existed.
    pub async fn set_bundle(
        &self,
        name: String,
        bundle: BundleAlias,
    ) -> Result<Option<AliasValue>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let prev = config.aliases.insert(name, AliasValue::Bundle(bundle));
        self.ctx.save_raw(&config).await?;
        Ok(prev)
    }

    /// Removes an alias of either kind. Returns the removed value if it existed.
    pub async fn remove_alias(&self, name: &str) -> Result<Option<AliasValue>> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let removed = config.aliases.remove(name);
        if removed.is_some() {
            self.ctx.save_raw(&config).await?;
        }
        Ok(removed)
    }

    /// Resolves a model name through Model aliases, with cycle detection.
    /// Bundle entries are ignored (they're not models). Returns the final
    /// resolved model name.
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

    /// A held lock must error within the bound, not wait forever.
    #[cfg(unix)]
    #[test]
    fn config_lock_acquire_is_bounded() {
        let dir = TempDir::new().unwrap();
        let lock_path = dir.path().join("config.lock");
        let held = ConfigLockGuard::acquire(&lock_path).unwrap();
        let start = std::time::Instant::now();
        let err = match ConfigLockGuard::acquire(&lock_path) {
            Ok(_) => panic!("second acquire should time out"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("held"), "got: {err}");
        assert!(start.elapsed() < std::time::Duration::from_secs(3));
        drop(held);
        ConfigLockGuard::acquire(&lock_path).unwrap();
    }

    #[test]
    fn migrate_legacy_per_model_is_idempotent() {
        // First run folds legacy maps into per_model_usage; a second run on the
        // already-migrated state must be a no-op (legacy maps stay empty,
        // per_model_usage stays unchanged).
        let mut counter = UsageCounter::default();
        counter.per_model_tokens.insert("kimi".to_string(), 450);
        counter
            .per_model_prompt_tokens
            .insert("kimi".to_string(), 200);
        counter
            .per_model_completion_tokens
            .insert("kimi".to_string(), 100);
        counter
            .per_model_cache_read_tokens
            .insert("kimi".to_string(), 25);

        let mut stats = UsageStats::default();
        stats.key_usage.insert("k1".to_string(), counter);

        stats.migrate_legacy_per_model();
        let after_first = stats.clone();
        stats.migrate_legacy_per_model();
        assert_eq!(stats, after_first);
    }

    #[test]
    fn model_counter_serde_round_trip() {
        // Skip-if-zero: only populated fields should serialize.
        let mc = ModelCounter {
            prompt_tokens: 700,
            completion_tokens: 0,
            cache_read_input_tokens: 25,
            cache_creation_input_tokens: 0,
        };
        let json = serde_json::to_string(&mc).unwrap();
        assert!(json.contains("promptTokens"));
        assert!(!json.contains("completionTokens"));
        assert!(json.contains("cacheReadInputTokens"));
        assert!(!json.contains("cacheCreationInputTokens"));
        let parsed: ModelCounter = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, mc);
        // Reading data with all zero fields produces a default-valued struct.
        let empty: ModelCounter = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, ModelCounter::default());
    }

    #[test]
    fn usage_counter_without_per_agent_loads_default() {
        // A stats file recorded before per-agent tracking has no `perAgent` key;
        // it must deserialize with an empty map (forward-compat).
        let json = r#"{"promptTokens":300,"totalTokens":300}"#;
        let counter: UsageCounter = serde_json::from_str(json).unwrap();
        assert!(counter.per_agent.is_empty());
        // Re-serializing an empty per_agent omits the field entirely.
        let out = serde_json::to_string(&counter).unwrap();
        assert!(!out.contains("perAgent"));
    }

    #[test]
    fn agent_usage_serde_round_trip() {
        let usage = AgentUsage {
            runs: 5,
            ok_runs: 4,
            steps: 42,
            tokens: 12_000,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("okRuns"));
        let parsed: AgentUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, usage);
        // All-zero fields collapse to `{}` and read back as the default.
        assert_eq!(serde_json::to_string(&AgentUsage::default()).unwrap(), "{}");
        let empty: AgentUsage = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, AgentUsage::default());
    }

    #[test]
    fn record_agent_run_accumulates_per_key() {
        let mut stats = UsageStats::default();
        stats.record_agent_run("k1", "code-reviewer", true, 6, 900);
        stats.record_agent_run("k1", "code-reviewer", false, 3, 400);
        stats.record_agent_run("k1", "explorer", true, 2, 100);
        let agents = &stats.key_usage.get("k1").unwrap().per_agent;
        let reviewer = agents.get("code-reviewer").unwrap();
        assert_eq!(reviewer.runs, 2);
        assert_eq!(reviewer.ok_runs, 1); // one run failed
        assert_eq!(reviewer.steps, 9);
        assert_eq!(reviewer.tokens, 1300);
        assert_eq!(agents.get("explorer").unwrap().runs, 1);
    }

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
            claude_path_variant: None,
            gemini_path_variant: None,
            requires_reasoning_content: None,
            protocol_routes: Default::default(),
            routing_schema_version: 0,
            key: Zeroizing::new("{}".into()),
            created_at: Utc::now().to_rfc3339(),
        };
        assert!(k.is_claude_oauth());
        assert!(!k.is_codex_oauth());
    }

    #[test]
    fn short_id_is_char_boundary_safe() {
        // A multi-byte id (e.g. hand-edited config) must not panic on a byte slice.
        let k =
            ApiKey::new_with_protocol("日本語id".into(), "".into(), "x".into(), None, "k".into());
        assert_eq!(k.short_id(), "日本語");
        assert_eq!(k.display_name(), "日本語");

        let short = ApiKey::new_with_protocol("ab".into(), "".into(), "x".into(), None, "k".into());
        assert_eq!(short.short_id(), "ab");

        let ascii =
            ApiKey::new_with_protocol("abcdef".into(), "".into(), "x".into(), None, "k".into());
        assert_eq!(ascii.short_id(), "abc");
    }

    #[test]
    fn cursor_credential_labels_distinguish_login_and_apikey() {
        let mut k = ApiKey::new_with_protocol(
            "x".into(),
            "cursor".into(),
            crate::services::cursor_acp::CURSOR_ACP_SENTINEL.into(),
            None,
            crate::services::cursor_acp::build_cursor_oauth_secret("testaccount1"),
        );
        assert!(k.is_cursor_acp());
        assert_eq!(k.credential_label(), Some("<Cursor login>"));

        k.key = Zeroizing::new(crate::services::cursor_acp::build_cursor_apikey_secret(
            "testaccount1",
            "key_xyz",
        ));
        assert_eq!(k.credential_label(), Some("<Cursor API key>"));

        // Non-shadow values (legacy raw API key) get no label, so the
        // suffix preview falls through and the key tail prints.
        k.key = Zeroizing::new("sk-cursor".to_string());
        assert_eq!(k.credential_label(), None);
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

    /// The per-repo project-MCP allow-list round-trips and shares code-prefs.json
    /// with the auto-approve toggle without either write clobbering the other.
    #[tokio::test]
    async fn project_mcp_approval_persists_and_coexists_with_auto_approve() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));

        store.set_chat_auto_approve(true).await.unwrap();
        assert!(!store.get_project_mcp_approved("/repo/a", "sig1").await);

        store
            .set_project_mcp_approved("/repo/a", "sig1")
            .await
            .unwrap();
        assert!(store.get_project_mcp_approved("/repo/a", "sig1").await);
        // A changed .mcp.json (different content digest) no longer matches.
        assert!(!store.get_project_mcp_approved("/repo/a", "sig2").await);
        assert!(!store.get_project_mcp_approved("/repo/b", "sig1").await);
        // Adding the allow-list entry preserved autoApprove.
        assert!(store.get_chat_auto_approve().await);

        // Flipping auto-approve preserves the allow-list.
        store.set_chat_auto_approve(false).await.unwrap();
        assert!(store.get_project_mcp_approved("/repo/a", "sig1").await);
        assert!(!store.get_chat_auto_approve().await);
        // Re-approving with a new digest supersedes the old one (one entry per dir).
        store
            .set_project_mcp_approved("/repo/a", "sig2")
            .await
            .unwrap();
        assert!(store.get_project_mcp_approved("/repo/a", "sig2").await);
        assert!(
            !store.get_project_mcp_approved("/repo/a", "sig1").await,
            "the superseded digest no longer matches"
        );
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

    /// The edit-review toggle defaults off, round-trips, and coexists with the
    /// auto-approve pref (each read-merge-writes without clobbering the other).
    #[tokio::test]
    async fn test_review_edits_pref_persists_beside_auto_approve() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));

        // Default off, and absent from the toggles snapshot's default.
        assert!(!store.get_chat_review_edits().await);
        assert!(!store.get_chat_toggles().await.review_edits);

        // Set a sibling pref first, then flip review-edits; neither clobbers the other.
        store.set_chat_auto_approve(true).await.unwrap();
        store.set_chat_review_edits(true).await.unwrap();

        assert!(store.get_chat_review_edits().await);
        let toggles = store.get_chat_toggles().await;
        assert!(toggles.review_edits);
        assert!(
            toggles.auto_approve,
            "sibling autoApprove survived the write"
        );

        // And it can be turned back off.
        store.set_chat_review_edits(false).await.unwrap();
        assert!(!store.get_chat_review_edits().await);
        assert!(store.get_chat_auto_approve().await);
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
    async fn chat_auto_approve_persists_and_defaults_off() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // No prefs file yet → defaults to off.
        assert!(!store.get_chat_auto_approve().await);

        // Round-trips both directions across (re)reads.
        store.set_chat_auto_approve(true).await.unwrap();
        assert!(store.get_chat_auto_approve().await);
        store.set_chat_auto_approve(false).await.unwrap();
        assert!(!store.get_chat_auto_approve().await);
    }

    #[tokio::test]
    async fn chat_thinking_enabled_persists_and_defaults_on() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // No prefs file yet → defaults to ON (thinking is high-signal).
        assert!(store.get_chat_thinking_enabled().await);

        // Round-trips both directions across (re)reads, and coexists with the
        // auto-approve flag in the same file.
        store.set_chat_auto_approve(true).await.unwrap();
        store.set_chat_thinking_enabled(false).await.unwrap();
        assert!(!store.get_chat_thinking_enabled().await);
        assert!(store.get_chat_auto_approve().await);
        store.set_chat_thinking_enabled(true).await.unwrap();
        assert!(store.get_chat_thinking_enabled().await);
    }

    #[tokio::test]
    async fn chat_theme_persists_and_defaults_unset() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // Never set → None, so startup auto-detects rather than forcing dark.
        assert_eq!(store.get_chat_toggles().await.theme, None);

        store.set_chat_theme(ChatTheme::Light).await.unwrap();
        assert_eq!(store.get_chat_toggles().await.theme, Some(ChatTheme::Light));
        let prefs: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(crate::services::paths::code_prefs(temp_dir.path()))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(prefs["theme"], serde_json::json!("light"));

        store.set_chat_theme(ChatTheme::Dark).await.unwrap();
        assert_eq!(store.get_chat_toggles().await.theme, Some(ChatTheme::Dark));
    }

    #[tokio::test]
    async fn chat_agent_tools_toggle_persists_and_defaults_on() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        assert!(store.get_chat_toggles().await.agent_tools_enabled);

        store.set_chat_agent_tools_enabled(false).await.unwrap();
        assert!(!store.get_chat_toggles().await.agent_tools_enabled);
        let prefs: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(crate::services::paths::code_prefs(temp_dir.path()))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(prefs["agentTools"], serde_json::json!(false));
        assert!(!tokio::fs::try_exists(&config_path).await.unwrap());

        store.set_chat_agent_tools_enabled(true).await.unwrap();
        assert!(store.get_chat_toggles().await.agent_tools_enabled);
    }

    #[tokio::test]
    async fn chat_thinking_enabled_falls_back_to_legacy_show_thinking() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        // A pre-rename prefs file only has `showThinking`; honor it until the
        // new key is written.
        let dir = store.config_dir();
        let prefs_path = crate::services::paths::code_prefs(dir);
        tokio::fs::create_dir_all(prefs_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&prefs_path, br#"{"showThinking": false}"#)
            .await
            .unwrap();
        assert!(!store.get_chat_thinking_enabled().await);
        assert!(!store.get_chat_toggles().await.thinking_enabled);

        // Writing the new key takes precedence on the next read.
        store.set_chat_thinking_enabled(true).await.unwrap();
        assert!(store.get_chat_thinking_enabled().await);
    }

    #[tokio::test]
    async fn code_prefs_fall_back_to_legacy_chat_prefs_file() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));

        // Only the pre-rename `chat-prefs.json` exists (no `code-prefs.json`);
        // the reader must still pick up the user's toggles.
        let dir = store.config_dir();
        let legacy_path = crate::services::paths::chat_prefs_legacy(dir);
        tokio::fs::create_dir_all(legacy_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(&legacy_path, br#"{"showThinking": false}"#)
            .await
            .unwrap();
        assert!(!store.get_chat_thinking_enabled().await);

        // The first write lands in the new `code-prefs.json`.
        store.set_chat_thinking_enabled(true).await.unwrap();
        assert!(crate::services::paths::code_prefs(dir).exists());
        assert!(store.get_chat_thinking_enabled().await);
    }

    #[tokio::test]
    async fn disabled_toggles_persist_in_chat_prefs_not_config() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path.clone());

        store.set_skill_enabled("repo-study", false).await.unwrap();
        store.set_mcp_server_enabled("fs", false).await.unwrap();

        // Reads round-trip.
        assert_eq!(
            store.get_disabled_skills().await.unwrap(),
            vec!["repo-study"]
        );
        assert_eq!(store.get_disabled_mcp_servers().await.unwrap(), vec!["fs"]);

        // They live in code-prefs.json, never the (key-bearing) config.json.
        let prefs: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(crate::services::paths::code_prefs(temp_dir.path()))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(prefs["disabledSkills"][0], "repo-study");
        assert_eq!(prefs["disabledMcpServers"][0], "fs");
        assert!(
            !tokio::fs::try_exists(&config_path).await.unwrap(),
            "toggling must not create/write config.json"
        );

        // Re-enabling removes the entry.
        store.set_skill_enabled("repo-study", true).await.unwrap();
        assert!(store.get_disabled_skills().await.unwrap().is_empty());
    }

    /// The Ctrl+T per-tool opt-outs share the disabled-list machinery, keyed by
    /// qualified `mcp__server__tool` names.
    #[tokio::test]
    async fn disabled_mcp_tools_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(temp_dir.path().join("config.json"));

        store
            .set_mcp_tool_enabled("mcp__github__create_issue", false)
            .await
            .unwrap();
        assert_eq!(
            store.get_disabled_mcp_tools().await.unwrap(),
            vec!["mcp__github__create_issue"]
        );
        // Idempotent re-disable doesn't duplicate.
        store
            .set_mcp_tool_enabled("mcp__github__create_issue", false)
            .await
            .unwrap();
        assert_eq!(store.get_disabled_mcp_tools().await.unwrap().len(), 1);
        // Re-enabling removes the entry.
        store
            .set_mcp_tool_enabled("mcp__github__create_issue", true)
            .await
            .unwrap();
        assert!(store.get_disabled_mcp_tools().await.unwrap().is_empty());
    }

    /// The original bug: toggles lived in config.json, so any other config writer
    /// (here `set_active_key`) that round-tripped the file would drop them. Now
    /// they live in code-prefs.json, so a config rewrite leaves them intact.
    #[tokio::test]
    async fn config_rewrite_does_not_clobber_disabled_toggles() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);

        let id = store
            .add_key_with_protocol("k", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store.set_skill_enabled("repo-study", false).await.unwrap();

        // A wholesale config.json rewrite from an unrelated setter.
        store.set_active_key(&id).await.unwrap();

        assert_eq!(
            store.get_disabled_skills().await.unwrap(),
            vec!["repo-study"]
        );
    }

    /// A config.json written by an older binary still carries `disabled_skills`
    /// inline; the first read honors it and the first toggle migrates the whole
    /// set into code-prefs.json without losing the pre-existing opt-out.
    #[tokio::test]
    async fn legacy_config_disabled_skills_migrate_on_first_toggle() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(
            &config_path,
            br#"{"api_keys": [], "disabled_skills": ["old-skill"]}"#,
        )
        .await
        .unwrap();
        let store = SessionStore::with_path(config_path);

        // Read falls back to the legacy field before any migration.
        assert_eq!(
            store.get_disabled_skills().await.unwrap(),
            vec!["old-skill"]
        );

        // The first toggle seeds from the legacy field, so the prior opt-out
        // survives alongside the new one.
        store.set_skill_enabled("new-skill", false).await.unwrap();
        let mut disabled = store.get_disabled_skills().await.unwrap();
        disabled.sort();
        assert_eq!(disabled, vec!["new-skill", "old-skill"]);
    }

    /// `migrate_disabled_toggles` (run at chat startup) copies the legacy config.json
    /// opt-outs into code-prefs.json so a later config rewrite can't drop them, even
    /// if the user never toggles. Idempotent and non-destructive to existing prefs.
    #[tokio::test]
    async fn eager_migration_moves_legacy_toggles_to_chat_prefs() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(
            &config_path,
            br#"{"api_keys": [], "disabled_skills": ["x"], "disabled_mcp_servers": ["fs"]}"#,
        )
        .await
        .unwrap();
        let store = SessionStore::with_path(config_path.clone());

        store.migrate_disabled_toggles().await;

        let prefs: serde_json::Value = serde_json::from_slice(
            &tokio::fs::read(crate::services::paths::code_prefs(temp_dir.path()))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(prefs["disabledSkills"][0], "x");
        assert_eq!(prefs["disabledMcpServers"][0], "fs");

        // Now a wholesale config rewrite that drops the legacy fields can't lose them.
        let id = store
            .add_key_with_protocol("k", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();
        assert_eq!(store.get_disabled_skills().await.unwrap(), vec!["x"]);
        assert_eq!(store.get_disabled_mcp_servers().await.unwrap(), vec!["fs"]);

        // Idempotent: an existing chat-prefs value is never overwritten by a later run.
        store.set_skill_enabled("x", true).await.unwrap(); // user re-enables x
        store.migrate_disabled_toggles().await; // must NOT resurrect "x"
        assert!(store.get_disabled_skills().await.unwrap().is_empty());
    }

    /// Forward-compat guard: a config key this binary doesn't recognize must
    /// survive a load→save round-trip instead of being silently dropped.
    #[tokio::test]
    async fn unknown_config_field_round_trips() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        tokio::fs::write(
            &config_path,
            br#"{"api_keys": [], "futureFeatureFlag": {"nested": true}}"#,
        )
        .await
        .unwrap();
        let store = SessionStore::with_path(config_path.clone());

        // A wholesale rewrite via a normal setter.
        store
            .add_key_with_protocol("k", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        let raw: serde_json::Value =
            serde_json::from_slice(&tokio::fs::read(&config_path).await.unwrap()).unwrap();
        assert_eq!(raw["futureFeatureFlag"]["nested"], true);
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
            .record_tokens(&id, Some("chat"), Some("gpt-4o"), 10, 5, 90, 15)
            .await
            .unwrap();
        store
            .save_code_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "legacy",
                "gpt-4o",
                None,
                &[StoredChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                "hello",
                "hello",
                SessionTokens::default(),
                0.0,
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

        let session = store.get_code_session("legacy").await.unwrap().unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.session_id, "legacy");

        store
            .save_code_session_with_id(
                &id,
                "http://localhost",
                "/tmp/demo",
                "session-2",
                "gpt-4o-mini",
                None,
                &[StoredChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "second".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                }],
                "second",
                "second",
                SessionTokens::default(),
                0.0,
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
        // Session file should exist and hold the messages as plain readable JSON
        let session_path = store.session_file_path("legacy");
        let session_raw = tokio::fs::read_to_string(&session_path).await.unwrap();
        assert!(session_raw.contains("\"hello\""));
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
    fn test_chat_session_messages_plain_array_roundtrips() {
        // Current format (also the pre-encryption legacy format): a JSON array.
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

        let session: CodeSessionState = serde_json::from_str(json).expect("should parse array");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "user");
        assert_eq!(session.messages[0].content, "hello");
        assert_eq!(session.messages[1].role, "assistant");
        assert_eq!(session.messages[1].content, "hi there");

        // Serializing writes the array back verbatim — never an encrypted blob.
        let written = serde_json::to_value(&session).unwrap();
        assert!(written["messages"].is_array());
        assert_eq!(written["messages"][0]["content"], "hello");
    }

    #[test]
    fn stored_chat_message_model_optional_and_roundtrips() {
        // Pre-feature JSON parses; None is omitted on write.
        let legacy: StoredChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":"hi"}"#).unwrap();
        assert_eq!(legacy.model, None);
        assert!(!serde_json::to_string(&legacy).unwrap().contains("model"));
        let stamped = StoredChatMessage {
            model: Some("m1".into()),
            ..legacy
        };
        let json = serde_json::to_string(&stamped).unwrap();
        let back: StoredChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model.as_deref(), Some("m1"));
    }

    /// Sessions written before the plaintext switch hold an `enc*:` string;
    /// they must keep decrypting on read.
    #[test]
    fn test_chat_session_legacy_encrypted_messages_decrypt_on_read() {
        let msgs = vec![
            StoredChatMessage {
                model: None,
                role: "user".into(),
                content: "ping".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
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

        let session: CodeSessionState = serde_json::from_str(&session_json).unwrap();
        assert_eq!(session.messages, msgs);

        // An empty legacy string is an empty conversation, not an error.
        let empty_json = r#"{"sessionId":"s","keyId":"k","baseUrl":"u","cwd":"/","model":"m","messages":"","updatedAt":"2024-01-01T00:00:00Z"}"#;
        let empty: CodeSessionState = serde_json::from_str(empty_json).unwrap();
        assert!(empty.messages.is_empty());
    }

    /// An undecryptable legacy blob is a HARD load error: the session file
    /// stays on disk untouched instead of being silently loaded as empty and
    /// clobbered by the next save.
    #[test]
    fn test_chat_session_undecryptable_messages_fail_load() {
        let json = r#"{"sessionId":"s","keyId":"k","baseUrl":"u","cwd":"/","model":"m","messages":"enc4:not-really-a-blob","updatedAt":"2024-01-01T00:00:00Z"}"#;
        assert!(serde_json::from_str::<CodeSessionState>(json).is_err());
    }

    /// Legacy inline chat_sessions in config.json are per-entry lenient: an
    /// undecryptable session must never brick the config (API key) load.
    #[test]
    fn test_legacy_inline_chat_sessions_skip_bad_entries() {
        let json = r#"{
            "api_keys": [],
            "chat_sessions": {
                "good": {
                    "sessionId": "good", "keyId": "k", "baseUrl": "u",
                    "cwd": "/", "model": "m",
                    "messages": [{"role": "user", "content": "hello"}],
                    "updatedAt": "2024-01-01T00:00:00Z"
                },
                "bad": {
                    "sessionId": "bad", "keyId": "k", "baseUrl": "u",
                    "cwd": "/", "model": "m",
                    "messages": "enc4:garbage",
                    "updatedAt": "2024-01-01T00:00:00Z"
                }
            }
        }"#;
        let config: StoredConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.chat_sessions.len(), 1);
        assert!(config.chat_sessions.contains_key("good"));
    }

    /// `engineMessages` is best-effort: a corrupt or unreadable legacy blob
    /// (lost keyring, schema mismatch) must degrade to `None` — a lossy text
    /// resume — never panic or brick the session. A valid legacy encrypted
    /// blob and the current plain-array format both round-trip.
    #[test]
    fn test_engine_messages_degrade_not_brick() {
        let make = |engine: Option<serde_json::Value>| -> CodeSessionState {
            let mut v = serde_json::json!({
                "sessionId": "s", "keyId": "k", "baseUrl": "u",
                "cwd": "/", "model": "m", "messages": [],
                "updatedAt": "2024-01-01T00:00:00Z"
            });
            if let Some(e) = engine {
                v["engineMessages"] = e;
            }
            serde_json::from_value(v).unwrap()
        };

        assert!(make(None).engine_messages.is_none());

        assert!(
            make(Some("not-a-valid-blob".into()))
                .engine_messages
                .is_none(),
            "a corrupt engine blob must degrade to None, never brick"
        );

        let payload = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "assistant", "tool_calls": [{"id": "t1"}]}),
        ];

        // Current format: plain array.
        assert_eq!(
            make(Some(serde_json::Value::Array(payload.clone())))
                .engine_messages
                .unwrap(),
            payload
        );

        // Legacy format: encrypted blob still decrypts.
        let blob = encrypt(&serde_json::to_string(&payload).unwrap()).unwrap();
        assert_eq!(make(Some(blob.into())).engine_messages.unwrap(), payload);
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
            config.code_models.get("a1b2c3").unwrap(),
            "claude-sonnet-4-6"
        );
        assert_eq!(config.code_models.get("d4e5f6").unwrap(), "gpt-4o");

        // Legacy flat directory_starts migrated to nested format
        let tools = config.directory_starts.get("/home/user/project").unwrap();
        assert_eq!(tools.get("claude").unwrap().key_id, "a1b2c3");

        // Legacy per-directory last_selection picked most recent entry
        let sel = config.last_selection.unwrap();
        assert_eq!(sel.key_id, "d4e5f6");
        assert_eq!(sel.tool, "codex");

        // Stats preserved
        assert_eq!(*config.stats.tool_counts.get("claude").unwrap(), 42);

        // Legacy inline chat_sessions loaded (messages stay plaintext)
        let session = config.chat_sessions.get("sess-legacy").unwrap();
        assert_eq!(session.key_id, "a1b2c3");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, "user");
        assert_eq!(session.messages[0].content, "hello");

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

        assert_eq!(
            config.aliases.get("fast").unwrap(),
            &AliasValue::Model("claude-haiku-4-5".to_string())
        );

        // Global last_selection (new format) loaded directly
        let sel = config.last_selection.unwrap();
        assert_eq!(sel.tool, "claude");

        assert!(config.starter_key_dismissed);
    }

    /// Aliases of both shapes (string for Model, object for Bundle) must
    /// deserialize side-by-side, and the round-trip must preserve both.
    #[test]
    fn aliases_mixed_model_and_bundle_round_trip() {
        let json = r#"{
            "api_keys": [],
            "aliases": {
                "fast": "claude-haiku-4-5",
                "quick": {
                    "tool": "claude",
                    "args": ["--key", "work", "--model", "fast"]
                }
            }
        }"#;

        let config: StoredConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            config.aliases.get("fast").unwrap(),
            &AliasValue::Model("claude-haiku-4-5".to_string())
        );
        let bundle = match config.aliases.get("quick").unwrap() {
            AliasValue::Bundle(b) => b,
            other => panic!("expected Bundle, got {other:?}"),
        };
        assert_eq!(bundle.tool, "claude");
        assert_eq!(bundle.args, vec!["--key", "work", "--model", "fast"]);

        // Round-trip: re-serialize and re-load, both shapes preserved.
        let reserialized = serde_json::to_string(&config).unwrap();
        let reloaded: StoredConfig = serde_json::from_str(&reserialized).unwrap();
        assert_eq!(reloaded.aliases, config.aliases);
        // Sanity: the JSON should keep `fast` as a bare string, not promote it
        // to an object — that's how legacy configs stay readable.
        assert!(reserialized.contains(r#""fast":"claude-haiku-4-5""#));
        assert!(reserialized.contains(r#""tool":"claude""#));
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
        assert!(config.code_models.is_empty());
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
