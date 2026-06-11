//! EnvironmentInjector service for preparing tool-specific environment variables.
//! Maps API keys to the correct environment variables per AI tool.
use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::constants::{
    AIVO_STARTER_MODEL, AIVO_STARTER_REAL_URL, AIVO_STARTER_SENTINEL, PLACEHOLDER_LOOPBACK_URL,
};
use crate::services::codex_model_map::map_model_for_codex_cli;
use crate::services::model_names::{anthropic_native_model_name, google_native_model_name};
use crate::services::ollama::ollama_openai_base_url;
use crate::services::provider_profile::{ProviderKind, ProviderProfile, provider_profile_for_key};
use crate::services::provider_protocol::{ProviderProtocol, is_official_anthropic_endpoint};
use crate::services::router_selection::{self, ConnectionMode, use_router_for_opencode};
use crate::services::session_store::ApiKey;

/// Describes which env vars a tool uses for base URL, auth, and router configuration.
struct ToolEnvConfig {
    base_url_env: &'static str,
    auth_env: &'static str,
    router_flag: &'static str,
    router_prefix: &'static str,
    copilot_flag: &'static str,
}

/// Tool a router (by env prefix) persists per-model routes under, for the
/// `inject_connection` path. `None` for cache-less routers (Ollama/Copilot/
/// OpenRouter). opencode/pi seed their own namespaces in `for_opencode`/`for_pi`.
fn router_tool_name(router_prefix: &str) -> Option<&'static str> {
    match router_prefix {
        "AIVO_ANTHROPIC_TO_OPENAI_ROUTER" => Some("claude"),
        "AIVO_RESPONSES_TO_CHAT_ROUTER" => Some("codex"),
        "AIVO_GEMINI_ROUTER" => Some("gemini"),
        _ => None,
    }
}

/// EnvironmentInjector prepares tool-specific environment variables for AI tools
#[derive(Debug, Clone, Default)]
pub struct EnvironmentInjector;

/// Raw CLI slot values for Claude per-task models, before alias/picker
/// resolution. `None` means the flag was absent; `Some("")` means a bare flag
/// (open the picker); `Some("name")` is an explicit model. The shape mirrors
/// `ClaudeModelOverrides` so the resolver can fill into the same field
/// positions.
#[derive(Debug, Clone, Default)]
pub struct ClaudeSlotFlags {
    pub reasoning: Option<String>,
    pub subagent: Option<String>,
    pub haiku: Option<String>,
    pub sonnet: Option<String>,
    pub opus: Option<String>,
}

impl ClaudeSlotFlags {
    pub fn any_set(&self) -> bool {
        self.reasoning.is_some()
            || self.subagent.is_some()
            || self.haiku.is_some()
            || self.sonnet.is_some()
            || self.opus.is_some()
    }
}

/// Per-slot model overrides for Claude. `None` means "no override — keep
/// whatever the main `model` argument fanned out". `Some(value)` replaces
/// that slot's env var. The deprecated `ANTHROPIC_SMALL_FAST_MODEL` slot is
/// intentionally absent; users override haiku-class routing via `haiku`.
///
/// `max_context` is the `--max-context` runtime flag, piggybacked here to
/// avoid a parallel parameter on every launch path. `Some("<N>m")` appends a
/// canonical `[<N>m]` suffix to the model name; Claude Code recognizes
/// specific tiers (1m, 2m). aivo doesn't validate the tier — it passes
/// whatever digits the user supplied. Only the fanned-out default slots get
/// the suffix; per-slot overrides stay verbatim.
#[derive(Debug, Clone, Default)]
pub struct ClaudeModelOverrides {
    pub reasoning: Option<String>,
    pub subagent: Option<String>,
    pub haiku: Option<String>,
    pub sonnet: Option<String>,
    pub opus: Option<String>,
    pub max_context: Option<String>,
}

/// Internal aivo carrier: comma-separated names of env vars that
/// `prepare_runtime_env` should `env_remove` from the child instead of
/// setting. Stripped from the env before spawn. Used so the OAuth path can
/// actually *unset* an inherited `ANTHROPIC_API_KEY` rather than masking it
/// with an empty string (which Claude Code's auth resolver still treats as
/// "set" → API-key mode).
pub(crate) const AIVO_INTERNAL_ENV_UNSET: &str = "_AIVO_INTERNAL_ENV_UNSET";

/// Per-launch loopback auth token carrier. The same value is injected as the
/// child's API key, read by `launch_runtime` into each router's
/// `expected_token`, then stripped before spawn — so the child holds only the
/// token and the real upstream key stays router-side.
pub(crate) const AIVO_ROUTER_AUTH_TOKEN: &str = "AIVO_ROUTER_AUTH_TOKEN";

/// Slots Claude Code reads to pick the model for each routing class. aivo
/// fans the user's `--model` value out to all of them so the chosen model
/// wins everywhere. `ANTHROPIC_SMALL_FAST_MODEL` is intentionally absent —
/// it's deprecated in favor of `ANTHROPIC_DEFAULT_HAIKU_MODEL`.
/// See https://code.claude.com/docs/en/env-vars.md.
const CLAUDE_DEFAULT_MODEL_SLOTS: [&str; 6] = [
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_REASONING_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
];

/// Strips the API version suffix (`/v1beta`, `/v1`) and trailing slashes from
/// a Google base URL.  Tools whose SDKs append their own `apiVersion` path
/// (Gemini CLI) would produce a double path like `/v1beta/v1beta/models/…`
/// if the suffix were left in place.
fn strip_google_version_suffix(base_url: &str) -> &str {
    let base = base_url.trim_end_matches('/');
    base.strip_suffix("/v1beta")
        .or_else(|| base.strip_suffix("/v1"))
        .unwrap_or(base)
}

/// Persistent aivo-managed path for the gemini CLI's folder-trust store.
/// Kept outside the shadow `GEMINI_CLI_HOME` so trust choices survive the
/// tempdir being recreated on every launch.
fn aivo_gemini_trusted_folders_path() -> Option<std::path::PathBuf> {
    crate::services::system_env::home_dir().map(|home| {
        home.join(".config")
            .join("aivo")
            .join("gemini-trusted-folders.json")
    })
}

/// Ensures the Google base URL ends with `/v1beta`.  Tools that set
/// `apiVersion = ""` when a custom base URL is provided (Pi) expect the
/// version to already be part of the URL.
fn ensure_google_version_suffix(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1beta") || base.ends_with("/v1") {
        base.to_string()
    } else {
        format!("{}/v1beta", base)
    }
}

impl EnvironmentInjector {
    fn should_disable_claude_nonessential_traffic(key: &ApiKey) -> bool {
        !key.is_claude_oauth() && !is_official_anthropic_endpoint(&key.base_url)
    }

    /// Creates a new EnvironmentInjector
    pub fn new() -> Self {
        Self
    }

    /// Injects connection env vars following the common Ollama/Copilot/OpenRouter/Direct/Routed pattern.
    fn inject_connection(
        cfg: &ToolEnvConfig,
        key: &ApiKey,
        mode: &ConnectionMode,
        profile: &ProviderProfile,
    ) -> HashMap<String, String> {
        // Resolve sentinel base URLs to real URLs for env injection.
        let resolved_base_url = if key.base_url == AIVO_STARTER_SENTINEL {
            AIVO_STARTER_REAL_URL.to_string()
        } else {
            key.base_url.to_string()
        };
        // Tools require a non-empty API key env var. Use a placeholder for
        // the aivo starter provider which needs no real authentication.
        let auth_value = if key.key.is_empty() {
            AIVO_STARTER_SENTINEL.to_string()
        } else {
            key.key.to_string()
        };

        // The child authenticates to aivo's loopback router with a per-launch
        // token; the real key never enters the child env (Direct mode aside).
        let loopback_token = crate::services::serve_router::random_auth_token();

        let mut env = HashMap::new();
        match mode {
            ConnectionMode::Ollama => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), loopback_token.clone());
                env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
                env.insert(cfg.router_flag.to_string(), "1".to_string());
                env.insert(
                    format!("{}_API_KEY", cfg.router_prefix),
                    "ollama".to_string(),
                );
                env.insert(
                    format!("{}_BASE_URL", cfg.router_prefix),
                    ollama_openai_base_url(),
                );
                env.insert(
                    format!("{}_UPSTREAM_PROTOCOL", cfg.router_prefix),
                    "openai".to_string(),
                );
            }
            ConnectionMode::Copilot => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), loopback_token.clone());
                env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
                env.insert(cfg.copilot_flag.to_string(), "1".to_string());
                env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
            }
            ConnectionMode::OpenRouter => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), loopback_token.clone());
                env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
                env.insert("AIVO_USE_ROUTER".to_string(), "1".to_string());
                env.insert("AIVO_ROUTER_API_KEY".to_string(), auth_value);
                env.insert("AIVO_ROUTER_BASE_URL".to_string(), resolved_base_url);
            }
            ConnectionMode::Direct { base_url } => {
                let url = if base_url == AIVO_STARTER_SENTINEL {
                    AIVO_STARTER_REAL_URL.to_string()
                } else {
                    base_url.clone()
                };
                env.insert(cfg.base_url_env.to_string(), url);
                env.insert(cfg.auth_env.to_string(), auth_value);
            }
            ConnectionMode::Routed { protocol } => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), loopback_token.clone());
                env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
                env.insert(cfg.router_flag.to_string(), "1".to_string());
                env.insert(format!("{}_API_KEY", cfg.router_prefix), auth_value);
                env.insert(format!("{}_BASE_URL", cfg.router_prefix), resolved_base_url);
                env.insert(
                    format!("{}_UPSTREAM_PROTOCOL", cfg.router_prefix),
                    protocol.as_str().to_string(),
                );
                // Seed the router's RouteCache with this tool's learned routes
                // (omitted when none → it falls back to the tool-native prior).
                if let Some(tool) = router_tool_name(cfg.router_prefix) {
                    let routes = key.routes_for_tool(tool);
                    if !routes.is_empty()
                        && let Ok(json) = serde_json::to_string(&routes)
                    {
                        env.insert(format!("{}_ROUTES_JSON", cfg.router_prefix), json);
                    }
                }
                // Per-key learned override merges into the static profile so
                // the env-var contract stays single-sourced in `inject`.
                let mut quirks = profile.quirks;
                quirks.requires_reasoning_content |= key.requires_reasoning_content == Some(true);
                quirks.inject(&mut env, cfg.router_prefix);
            }
        }
        if profile.serve_flags.is_starter {
            env.insert("AIVO_IS_STARTER".to_string(), "1".to_string());
        }
        env
    }

    /// `for_claude_with_overrides` with no per-slot overrides. Used by the
    /// in-tree and integration test suites; production callers go through
    /// the overrides-aware entry point directly. `#[cfg(test)]` doesn't fit
    /// here because integration tests live in a separate crate that sees
    /// only the non-test build of this lib.
    #[allow(dead_code)]
    /// Builds the env block for a tool whose key is a cursor ACP sentinel.
    /// Sets a placeholder loopback `base_url_env` plus the `AIVO_USE_CURSOR_ROUTER`
    /// scaffolding flags that `launch_runtime::start_cursor_router` reads to
    /// spawn `CursorModelRouter` and rewrite the placeholder with the real port.
    fn for_cursor_acp_tool(
        key: &ApiKey,
        base_url_env: &str,
        auth_env: Option<&str>,
    ) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string());
        env.insert(
            "AIVO_CURSOR_BASE_URL_ENV".to_string(),
            base_url_env.to_string(),
        );
        env.insert(
            "AIVO_CURSOR_KEY_SECRET".to_string(),
            key.key.as_str().to_string(),
        );
        // Placeholder; launch_runtime fills the bound port.
        env.insert(
            base_url_env.to_string(),
            PLACEHOLDER_LOOPBACK_URL.to_string(),
        );
        if let Some(auth_env) = auth_env {
            let loopback_token = crate::services::serve_router::random_auth_token();
            env.insert(auth_env.to_string(), loopback_token.clone());
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
        }
        env
    }

    /// Builds the OpenCode env block for a cursor ACP key. OpenCode reads its
    /// upstream from `OPENCODE_CONFIG_CONTENT` (JSON), so the placeholder URL
    /// is embedded there and `launch_runtime::patch_opencode_config_content`
    /// rewrites it once the cursor router has bound its port.
    fn for_opencode_cursor(
        key: &ApiKey,
        model: Option<&str>,
        discovered_models: Option<&[String]>,
    ) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string());
        env.insert(
            "AIVO_CURSOR_KEY_SECRET".to_string(),
            key.key.as_str().to_string(),
        );
        let loopback_token = crate::services::serve_router::random_auth_token();
        env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token.clone());

        let mut provider = Map::new();
        provider.insert("npm".to_string(), json!("@ai-sdk/openai-compatible"));
        provider.insert("name".to_string(), json!("aivo"));
        provider.insert(
            "options".to_string(),
            json!({
                "baseURL": PLACEHOLDER_LOOPBACK_URL,
                "apiKey": loopback_token,
            }),
        );

        let mut model_ids: Vec<String> = discovered_models
            .map(|models| {
                models
                    .iter()
                    .map(|m| strip_aivo_prefix(m).to_string())
                    .collect()
            })
            .unwrap_or_default();
        if let Some(model) = model {
            let model_name = strip_aivo_prefix(model).to_string();
            if !model_ids.contains(&model_name) {
                model_ids.push(model_name);
            }
        }
        model_ids.sort();
        model_ids.dedup();
        if !model_ids.is_empty() {
            let mut models = Map::new();
            for model_id in model_ids {
                models.insert(model_id.clone(), json!({ "name": model_id }));
            }
            provider.insert("models".to_string(), Value::Object(models));
        }

        let mut providers = Map::new();
        providers.insert("aivo".to_string(), Value::Object(provider));

        let mut config = Map::new();
        config.insert(
            "$schema".to_string(),
            json!("https://opencode.ai/config.json"),
        );
        config.insert("provider".to_string(), Value::Object(providers));
        if let Some(model) = model {
            config.insert(
                "model".to_string(),
                json!(format!("aivo/{}", strip_aivo_prefix(model))),
            );
        }
        env.insert(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            Value::Object(config).to_string(),
        );
        env
    }

    pub fn for_claude(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        self.for_claude_with_overrides(key, model, &ClaudeModelOverrides::default())
    }

    /// Prepares environment variables for Claude CLI with optional per-slot
    /// model overrides. The main `model` (if provided) still fans out into all
    /// seven `ANTHROPIC_*` slots; `overrides` then selectively replaces any of
    /// the fast / reasoning / subagent / haiku / sonnet / opus slots.
    pub fn for_claude_with_overrides(
        &self,
        key: &ApiKey,
        model: Option<&str>,
        overrides: &ClaudeModelOverrides,
    ) -> HashMap<String, String> {
        if key.is_cursor_acp() {
            let mut env =
                Self::for_cursor_acp_tool(key, "ANTHROPIC_BASE_URL", Some("ANTHROPIC_AUTH_TOKEN"));
            // Claude Code keys off both ANTHROPIC_AUTH_TOKEN *and* an empty
            // ANTHROPIC_API_KEY to pick the right auth path; mirror the
            // existing routed-mode setup.
            env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
            if let Some(model) = model {
                let anthropic_model = match overrides.max_context.as_deref() {
                    Some(tag) => format!("{model}[{tag}]"),
                    None => model.to_string(),
                };
                for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
                    env.insert(slot.to_string(), anthropic_model.clone());
                }
            }
            return env;
        }

        if key.is_claude_oauth() {
            let mut env = HashMap::new();
            match crate::services::claude_oauth::ClaudeOAuthCredential::from_json(key.key.as_str())
            {
                Ok(creds) => {
                    env.insert("CLAUDE_CODE_OAUTH_TOKEN".to_string(), creds.token);
                }
                Err(e) => {
                    eprintln!(
                        "aivo: failed to decode Claude OAuth credential for key '{}': {e}. \
                         Re-run `aivo keys add claude` to refresh.",
                        key.display_name()
                    );
                }
            }
            // Claude Code's auth resolver prefers ANTHROPIC_API_KEY /
            // ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL over the OAuth
            // token, so a shell-exported value would shadow or misroute it.
            // Setting them to empty string is *not* enough — Claude Code's
            // source-detection treats a set-but-empty var as "present" and
            // labels the session as API-key auth. The launcher must actually
            // unset (env_remove) them so the child inherits no value at all.
            env.insert(
                AIVO_INTERNAL_ENV_UNSET.to_string(),
                "ANTHROPIC_API_KEY,ANTHROPIC_AUTH_TOKEN,ANTHROPIC_BASE_URL".to_string(),
            );

            // `-m`, `--1m`/`--2m`, and per-slot overrides apply to OAuth the
            // same as to API keys: Claude Code reads ANTHROPIC_MODEL and the
            // `[1m]` suffix regardless of auth path. The upstream is Anthropic-
            // native (subscription backend), so normalize model ids the same
            // way as Direct mode (`claude-sonnet-4.6` → `claude-sonnet-4-6`).
            if let Some(model) = model {
                let normalized = anthropic_native_model_name(model);
                let anthropic_model = match overrides.max_context.as_deref() {
                    Some(tag) => format!("{normalized}[{tag}]"),
                    None => normalized,
                };
                for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
                    env.insert(slot.to_string(), anthropic_model.clone());
                }
            }
            for (env_var, value) in [
                ("ANTHROPIC_REASONING_MODEL", overrides.reasoning.as_deref()),
                ("CLAUDE_CODE_SUBAGENT_MODEL", overrides.subagent.as_deref()),
                ("ANTHROPIC_DEFAULT_HAIKU_MODEL", overrides.haiku.as_deref()),
                (
                    "ANTHROPIC_DEFAULT_SONNET_MODEL",
                    overrides.sonnet.as_deref(),
                ),
                ("ANTHROPIC_DEFAULT_OPUS_MODEL", overrides.opus.as_deref()),
            ] {
                if let Some(v) = value {
                    env.insert(env_var.to_string(), anthropic_native_model_name(v));
                }
            }
            return env;
        }

        let profile = provider_profile_for_key(key);
        let mode = router_selection::claude_connection_mode(key, &profile);

        let cfg = ToolEnvConfig {
            base_url_env: "ANTHROPIC_BASE_URL",
            auth_env: "ANTHROPIC_AUTH_TOKEN",
            router_flag: "AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER",
            router_prefix: "AIVO_ANTHROPIC_TO_OPENAI_ROUTER",
            copilot_flag: "AIVO_USE_COPILOT_ROUTER",
        };

        let mut env = Self::inject_connection(&cfg, key, &mode, &profile);
        // Forward the persisted path variant so the Anthropic-to-OpenAI router
        // can skip re-probing /v1/chat/completions vs /chat/completions on
        // every launch. Only meaningful in Routed mode; harmless otherwise.
        if let Some(variant) = key.claude_path_variant.as_deref() {
            env.insert(
                "AIVO_ANTHROPIC_TO_OPENAI_ROUTER_PATH_VARIANT".to_string(),
                variant.to_string(),
            );
        }
        env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
        if Self::should_disable_claude_nonessential_traffic(key) {
            env.insert(
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
                "1".to_string(),
            );
        }
        env.insert("BASH_DEFAULT_TIMEOUT_MS".to_string(), "2400000".to_string());
        env.insert("BASH_MAX_TIMEOUT_MS".to_string(), "2500000".to_string());
        env.insert(
            "CLAUDE_CODE_ATTRIBUTION_HEADER".to_string(),
            "0".to_string(),
        );
        env.insert("API_TIMEOUT_MS".to_string(), "30000000".to_string());
        // Beta-header policy. `ANTHROPIC_BASE_URL` (which we always set)
        // makes Claude Code treat the upstream as a gateway: fine-grained
        // tool-input streaming defaults off, and experimental beta fields
        // (`defer_loading`, `eager_input_streaming`) keep flowing. For real
        // Anthropic-shaped endpoints (Direct mode) we want streaming on; for
        // aivo's loopback Anthropic↔OpenAI router (every other mode) we want
        // the experimental fields stripped so the OpenAI-shaped upstream
        // doesn't 400 on unknown headers or extra schema keys.
        match &mode {
            ConnectionMode::Direct { .. } => {
                env.insert(
                    "CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING".to_string(),
                    "1".to_string(),
                );
            }
            ConnectionMode::Ollama
            | ConnectionMode::Copilot
            | ConnectionMode::OpenRouter
            | ConnectionMode::Routed { .. } => {
                env.insert(
                    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS".to_string(),
                    "1".to_string(),
                );
            }
        }
        if let Some(model) = model {
            let normalized = if matches!(mode, ConnectionMode::Direct { .. }) {
                anthropic_native_model_name(model)
            } else {
                model.to_string()
            };
            let anthropic_model = match overrides.max_context.as_deref() {
                Some(tag) => format!("{normalized}[{tag}]"),
                None => normalized,
            };
            for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
                env.insert(slot.to_string(), anthropic_model.clone());
            }
            // Surface the routed model in /model picker. Skip Direct mode —
            // slot values are already claude-* ids the picker labels natively.
            if !matches!(mode, ConnectionMode::Direct { .. }) {
                env.insert(
                    "ANTHROPIC_CUSTOM_MODEL_OPTION".to_string(),
                    anthropic_model.clone(),
                );
                env.insert(
                    "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION".to_string(),
                    format!("Routed via aivo ({})", key.display_name()),
                );
            }
        }

        // Per-slot overrides win over the fan-out from `model`. Each slot is
        // normalized through the same anthropic_native_model_name() pass when
        // talking to a native Anthropic endpoint so e.g. `claude-sonnet-4.6`
        // becomes `claude-sonnet-4-6`.
        let normalize = |v: &str| {
            if matches!(mode, ConnectionMode::Direct { .. }) {
                anthropic_native_model_name(v)
            } else {
                v.to_string()
            }
        };
        for (env_var, value) in [
            ("ANTHROPIC_REASONING_MODEL", overrides.reasoning.as_deref()),
            ("CLAUDE_CODE_SUBAGENT_MODEL", overrides.subagent.as_deref()),
            ("ANTHROPIC_DEFAULT_HAIKU_MODEL", overrides.haiku.as_deref()),
            (
                "ANTHROPIC_DEFAULT_SONNET_MODEL",
                overrides.sonnet.as_deref(),
            ),
            ("ANTHROPIC_DEFAULT_OPUS_MODEL", overrides.opus.as_deref()),
        ] {
            if let Some(v) = value {
                env.insert(env_var.to_string(), normalize(v));
            }
        }

        env
    }

    /// Prepares environment variables for Codex CLI
    pub fn for_codex(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        if key.is_cursor_acp() {
            let mut env = Self::for_cursor_acp_tool(key, "OPENAI_BASE_URL", Some("OPENAI_API_KEY"));
            if let Some(model) = model {
                env.insert("CODEX_MODEL".to_string(), model.to_string());
                env.insert("OPENAI_DEFAULT_MODEL".to_string(), model.to_string());
                env.insert("CODEX_MODEL_DEFAULT".to_string(), model.to_string());
            }
            return env;
        }

        // ChatGPT OAuth path: the credential lives encrypted in `key.key` as
        // serialized `CodexOAuthCredential` JSON. Pass it through to
        // launch_runtime via a private env var; launch_runtime writes a
        // shadow `CODEX_HOME` and spawns codex against that. We don't set
        // OPENAI_BASE_URL / OPENAI_API_KEY here — native codex will read
        // `auth.json` from the shadow dir.
        if key.is_codex_oauth() {
            let mut env = HashMap::new();
            env.insert(
                "AIVO_CODEX_OAUTH_CREDS".to_string(),
                key.key.as_str().to_string(),
            );
            env.insert("AIVO_CODEX_KEY_ID".to_string(), key.id.clone());
            if let Some(model) = model {
                let codex_model = map_model_for_codex_cli(model);
                env.insert("CODEX_MODEL".to_string(), codex_model.clone());
                env.insert("OPENAI_DEFAULT_MODEL".to_string(), codex_model.clone());
                env.insert("CODEX_MODEL_DEFAULT".to_string(), codex_model);
            }
            return env;
        }

        let profile = provider_profile_for_key(key);
        let mode = router_selection::codex_connection_mode(key, &profile);

        let cfg = ToolEnvConfig {
            base_url_env: "OPENAI_BASE_URL",
            auth_env: "OPENAI_API_KEY",
            router_flag: "AIVO_USE_RESPONSES_TO_CHAT_ROUTER",
            router_prefix: "AIVO_RESPONSES_TO_CHAT_ROUTER",
            copilot_flag: "AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER",
        };

        let mut env = Self::inject_connection(&cfg, key, &mode, &profile);

        // Codex-specific: responses_api_supported flag (routed mode only)
        if matches!(mode, ConnectionMode::Routed { .. })
            && let Some(supported) = key.responses_api_supported
        {
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_RESPONSES_API".to_string(),
                if supported { "1" } else { "0" }.to_string(),
            );
        }

        if let Some(model) = model {
            let using_router = env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER")
                || env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER");
            let codex_model = if using_router {
                model.to_string()
            } else {
                map_model_for_codex_cli(model)
            };
            env.insert("CODEX_MODEL".to_string(), codex_model.clone());
            env.insert("OPENAI_DEFAULT_MODEL".to_string(), codex_model.clone());
            env.insert("CODEX_MODEL_DEFAULT".to_string(), codex_model.clone());
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL".to_string(),
                model.to_string(),
            );
        }

        env
    }

    /// Prepares environment variables for Gemini CLI
    pub fn for_gemini(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        // Cursor ACP path: route gemini-cli through the local cursor router.
        // gemini-cli's API-key auth path reads `GOOGLE_GEMINI_BASE_URL` +
        // `GEMINI_API_KEY` and speaks Gemini's `/v1beta/models/<m>:generateContent`
        // protocol — the cursor router has a matching handler that translates
        // those requests into ACP prompts. The settings-override (forced via
        // AIVO_GEMINI_FORCE_API_KEY_AUTH) pins the CLI to api-key auth so it
        // never falls back to OAuth.
        if key.is_cursor_acp() {
            let mut env =
                Self::for_cursor_acp_tool(key, "GOOGLE_GEMINI_BASE_URL", Some("GEMINI_API_KEY"));
            // gemini-cli's OAuth picker would otherwise wait for stdin on
            // first launch — force api-key path via the system-scope settings
            // override prepared by `prepare_gemini_api_key_settings_override`.
            env.insert(
                "AIVO_GEMINI_FORCE_API_KEY_AUTH".to_string(),
                "1".to_string(),
            );
            if let Some(model) = model {
                env.insert("GEMINI_MODEL".to_string(), model.to_string());
                env.insert(
                    "AIVO_GEMINI_MODEL_CONFIG_MODEL".to_string(),
                    model.to_string(),
                );
            }
            return env;
        }

        // Google OAuth path: the credential lives encrypted in `key.key` as
        // serialized `GeminiOAuthCredential` JSON. Pass it through to
        // launch_runtime via a private env var; launch_runtime writes a
        // shadow `GEMINI_CLI_HOME` and spawns gemini against that. We don't
        // set GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL here — native gemini
        // will read `oauth_creds.json` from the shadow dir.
        if key.is_gemini_oauth() {
            let mut env = HashMap::new();
            env.insert(
                "AIVO_GEMINI_OAUTH_CREDS".to_string(),
                key.key.as_str().to_string(),
            );
            env.insert("AIVO_GEMINI_KEY_ID".to_string(), key.id.clone());
            if let Some(m) = model {
                env.insert(
                    "GEMINI_MODEL".to_string(),
                    google_native_model_name(m).to_string(),
                );
            }
            // `GOOGLE_GENAI_USE_GCA=true` is the gemini-cli's explicit signal
            // to use the personal Google OAuth auth path, bypassing the
            // first-run TUI auth-type picker. Without it the CLI would prompt
            // even though `oauth_creds.json` is already present.
            env.insert("GOOGLE_GENAI_USE_GCA".to_string(), "true".to_string());
            // Point folder-trust storage at a persistent aivo-managed file
            // (the shadow HOME is recreated each launch, so
            // `.gemini/trustedFolders.json` inside it would reset the user's
            // trust choices every run).
            if let Some(path) = aivo_gemini_trusted_folders_path() {
                env.insert(
                    "GEMINI_CLI_TRUSTED_FOLDERS_PATH".to_string(),
                    path.to_string_lossy().to_string(),
                );
            }
            // Clear direct-auth env vars so a caller export can't override
            // the shadow HOME's OAuth credentials.
            env.insert("GEMINI_API_KEY".to_string(), String::new());
            env.insert("GOOGLE_API_KEY".to_string(), String::new());
            env.insert("GOOGLE_GEMINI_BASE_URL".to_string(), String::new());
            return env;
        }

        let profile = provider_profile_for_key(key);
        let mode = router_selection::gemini_connection_mode(
            key,
            &profile,
            strip_google_version_suffix(&key.base_url),
        );

        let cfg = ToolEnvConfig {
            base_url_env: "GOOGLE_GEMINI_BASE_URL",
            auth_env: "GEMINI_API_KEY",
            router_flag: "AIVO_USE_GEMINI_ROUTER",
            router_prefix: "AIVO_GEMINI_ROUTER",
            copilot_flag: "AIVO_USE_GEMINI_COPILOT_ROUTER",
        };

        let mut env = Self::inject_connection(&cfg, key, &mode, &profile);

        // Gemini-specific: copilot forced model
        if matches!(mode, ConnectionMode::Copilot)
            && let Some(m) = model
        {
            env.insert(
                "AIVO_GEMINI_COPILOT_FORCED_MODEL".to_string(),
                m.to_string(),
            );
        }

        let gemini_model = model
            .map(|model| {
                if matches!(mode, ConnectionMode::Direct { .. }) {
                    google_native_model_name(model).to_string()
                } else {
                    model.to_string()
                }
            })
            .or_else(|| {
                profile
                    .serve_flags
                    .is_starter
                    .then(|| AIVO_STARTER_MODEL.to_string())
            });
        if let Some(gemini_model) = gemini_model {
            env.insert("GEMINI_MODEL".to_string(), gemini_model.clone());
            env.insert("AIVO_GEMINI_MODEL_CONFIG_MODEL".to_string(), gemini_model);
        }

        // Signal to launch_runtime::prepare_gemini_api_key_settings_override.
        env.insert(
            "AIVO_GEMINI_FORCE_API_KEY_AUTH".to_string(),
            "1".to_string(),
        );

        env
    }

    /// Prepares environment variables for OpenCode CLI.
    ///
    /// Uses OPENCODE_CONFIG_CONTENT to inject an inline OpenCode config
    /// so aivo can provide base URL and API key without writing config files.
    pub fn for_opencode(
        &self,
        key: &ApiKey,
        model: Option<&str>,
        discovered_models: Option<&[String]>,
        limits: &HashMap<String, crate::services::model_metadata::ResolvedLimits>,
    ) -> HashMap<String, String> {
        if key.is_cursor_acp() {
            return Self::for_opencode_cursor(key, model, discovered_models);
        }
        let mut env = HashMap::new();
        let profile = provider_profile_for_key(key);
        let resolved_url = if key.base_url == AIVO_STARTER_SENTINEL {
            AIVO_STARTER_REAL_URL.to_string()
        } else {
            key.base_url.clone()
        };
        let auth = if key.key.is_empty() {
            AIVO_STARTER_SENTINEL.to_string()
        } else {
            key.key.to_string()
        };

        // For Ollama, connect directly — OpenCode speaks OpenAI-compatible natively.
        // For GitHub Copilot, the base_url is the magic string "copilot" — not a real URL.
        // Use a placeholder that ai_launcher will overwrite with the actual CopilotRouter port.
        let loopback_token = crate::services::serve_router::random_auth_token();
        let (base_url, api_key) = if profile.kind == ProviderKind::Ollama {
            (ollama_openai_base_url(), "ollama".to_string())
        } else if profile.serve_flags.is_copilot {
            env.insert(
                "AIVO_USE_OPENCODE_COPILOT_ROUTER".to_string(),
                "1".to_string(),
            );
            env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token.clone());
            (PLACEHOLDER_LOOPBACK_URL.to_string(), loopback_token)
        } else if use_router_for_opencode(key) || profile.serve_flags.is_starter {
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token.clone());
            env.insert("AIVO_USE_OPENCODE_ROUTER".to_string(), "1".to_string());
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY".to_string(),
                auth.clone(),
            );
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL".to_string(),
                resolved_url.clone(),
            );
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL".to_string(),
                profile.default_protocol.as_str().to_string(),
            );
            // Seed the router's per-model RouteCache with what opencode learned
            // before (its own `(opencode, key, model)` namespace).
            let opencode_routes = key.routes_for_tool("opencode");
            if !opencode_routes.is_empty()
                && let Ok(json) = serde_json::to_string(&opencode_routes)
            {
                env.insert(
                    "AIVO_RESPONSES_TO_CHAT_ROUTER_ROUTES_JSON".to_string(),
                    json,
                );
            }
            if let Some(supported) = key.responses_api_supported {
                env.insert(
                    "AIVO_RESPONSES_TO_CHAT_ROUTER_RESPONSES_API".to_string(),
                    if supported { "1" } else { "0" }.to_string(),
                );
            }
            profile
                .quirks
                .inject(&mut env, "AIVO_RESPONSES_TO_CHAT_ROUTER");
            if profile.serve_flags.is_starter {
                env.insert("AIVO_IS_STARTER".to_string(), "1".to_string());
                // OpenCode's SDK strips the `aivo/` provider prefix from
                // outgoing body models, so the upstream sees `starter`
                // instead of `aivo/starter`. Pass the bare ids of catalog
                // entries that originally had the `aivo/` prefix so the
                // local router re-adds it per-request — this preserves
                // mid-session model switching (each request reflects the
                // user's current pick rather than a launch-time pin).
                let aivo_prefix_models: Vec<&str> = discovered_models
                    .map(|catalog| {
                        catalog
                            .iter()
                            .filter_map(|m| m.strip_prefix("aivo/"))
                            .collect()
                    })
                    .unwrap_or_default();
                if !aivo_prefix_models.is_empty() {
                    env.insert(
                        "AIVO_RESPONSES_TO_CHAT_ROUTER_AIVO_PREFIX_MODELS".to_string(),
                        aivo_prefix_models.join(","),
                    );
                }
            }
            (PLACEHOLDER_LOOPBACK_URL.to_string(), loopback_token)
        } else {
            (resolved_url, auth)
        };

        let mut provider = Map::new();
        provider.insert("npm".to_string(), json!("@ai-sdk/openai-compatible"));
        provider.insert("name".to_string(), json!("aivo"));
        provider.insert(
            "options".to_string(),
            json!({
                "baseURL": base_url,
                "apiKey": api_key,
            }),
        );

        let mut model_ids: Vec<String> = discovered_models
            .map(|models| {
                models
                    .iter()
                    .map(|m| strip_aivo_prefix(m).to_string())
                    .collect()
            })
            .unwrap_or_default();

        if let Some(model) = model {
            let model_name = strip_aivo_prefix(model).to_string();
            if !model_ids.contains(&model_name) {
                model_ids.push(model_name);
            }
        }

        model_ids.sort();
        model_ids.dedup();
        if !model_ids.is_empty() {
            let mut models = Map::new();
            for model_id in model_ids {
                let mut entry = json!({ "name": model_id });
                // opencode's schema requires context AND output together;
                // limits are keyed by the raw (possibly `aivo/`-prefixed) id.
                let resolved = limits
                    .get(&model_id)
                    .or_else(|| limits.get(&format!("aivo/{model_id}")));
                if let Some(l) = resolved
                    && let (Some(context), Some(output)) = (l.context, l.output)
                {
                    entry["limit"] = json!({ "context": context, "output": output });
                }
                models.insert(model_id.clone(), entry);
            }
            provider.insert("models".to_string(), Value::Object(models));
        }

        let mut providers = Map::new();
        providers.insert("aivo".to_string(), Value::Object(provider));

        let mut config = Map::new();
        config.insert(
            "$schema".to_string(),
            json!("https://opencode.ai/config.json"),
        );
        config.insert("provider".to_string(), Value::Object(providers));

        if let Some(model) = model {
            config.insert(
                "model".to_string(),
                json!(format!("aivo/{}", strip_aivo_prefix(model))),
            );
        }

        env.insert(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            Value::Object(config).to_string(),
        );
        env
    }

    /// Prepares environment variables for Pi CLI.
    ///
    /// Pi natively supports OpenAI, Anthropic, and Google protocols, so we point it
    /// directly at the upstream via a custom `aivo` provider in `models.json` with
    /// the appropriate `api` type. No conversion router is needed.
    ///
    /// - **Non-Copilot**: Write `models.json` with direct upstream URL + correct API type.
    /// - **Copilot**: Needs CopilotTokenManager for auth, so start a ResponsesToChatRouter
    ///   and point pi at it with `openai-completions`.
    pub fn for_pi(
        &self,
        key: &ApiKey,
        model: Option<&str>,
        catalog: &[String],
        limits: &HashMap<String, crate::services::model_metadata::ResolvedLimits>,
    ) -> HashMap<String, String> {
        let mut env = HashMap::new();
        let profile = provider_profile_for_key(key);

        if key.is_cursor_acp() {
            // Pi reads its upstream from a JSON config rather than an env var,
            // so the cursor router is wired in via PLACEHOLDER_LOOPBACK_URL:
            // launch_runtime starts the router, then patches the placeholder
            // with the real bound port before writing the temp agent dir.
            let loopback_token = crate::services::serve_router::random_auth_token();
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                &loopback_token,
                "openai-completions",
                model,
                catalog,
                limits,
            );
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string());
            env.insert(
                "AIVO_CURSOR_KEY_SECRET".to_string(),
                key.key.as_str().to_string(),
            );
            return env;
        }

        if profile.kind == ProviderKind::Ollama {
            // Ollama: direct connection via Pi's custom provider
            let models_json = build_pi_models_json(
                &ollama_openai_base_url(),
                "ollama",
                "openai-completions",
                model,
                catalog,
                limits,
            );
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_SETUP_PI_AGENT_DIR".to_string(), "1".to_string());
        } else if profile.serve_flags.is_copilot {
            // Copilot needs CopilotTokenManager — route through ResponsesToChatRouter
            let loopback_token = crate::services::serve_router::random_auth_token();
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                &loopback_token,
                "openai-completions",
                model,
                catalog,
                limits,
            );
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_USE_PI_COPILOT_ROUTER".to_string(), "1".to_string());
            env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
        } else if profile.serve_flags.is_starter {
            // Starter provider: route through a local router so device fingerprint
            // headers are attached (Pi's native HTTP client can't add them).
            let loopback_token = crate::services::serve_router::random_auth_token();
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                &loopback_token,
                "openai-completions",
                model,
                catalog,
                limits,
            );
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_USE_PI_STARTER_ROUTER".to_string(), "1".to_string());
            env.insert("AIVO_IS_STARTER".to_string(), "1".to_string());
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY".to_string(),
                AIVO_STARTER_SENTINEL.to_string(),
            );
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL".to_string(),
                AIVO_STARTER_REAL_URL.to_string(),
            );
        } else if crate::services::transform_mode::is_active()
            || crate::services::http_debug::is_debug_active()
        {
            // Force the local router. `--transform` does this explicitly to
            // normalize buggy SSE; `--debug` does it so the JSONL logger sees
            // traffic that pi's native HTTP client would otherwise skip.
            let loopback_token = crate::services::serve_router::random_auth_token();
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                &loopback_token,
                "openai-completions",
                model,
                catalog,
                limits,
            );
            env.insert(AIVO_ROUTER_AUTH_TOKEN.to_string(), loopback_token);
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_USE_PI_ROUTER".to_string(), "1".to_string());
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY".to_string(),
                key.key.to_string(),
            );
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL".to_string(),
                key.base_url.clone(),
            );
            env.insert(
                "AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL".to_string(),
                profile.default_protocol.as_str().to_string(),
            );
            // Seed the router with pi's own `(pi, key, model)` routes.
            let pi_routes = key.routes_for_tool("pi");
            if !pi_routes.is_empty()
                && let Ok(json) = serde_json::to_string(&pi_routes)
            {
                env.insert(
                    "AIVO_RESPONSES_TO_CHAT_ROUTER_ROUTES_JSON".to_string(),
                    json,
                );
            }
        } else {
            // Direct connection — pi talks to the upstream natively.
            // Map aivo's ProviderProtocol to pi's API type string.
            let pi_api = match profile.default_protocol {
                ProviderProtocol::Anthropic => "anthropic-messages",
                ProviderProtocol::Google => "google-generative-ai",
                ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => "openai-completions",
            };
            let resolved_url = if key.base_url == AIVO_STARTER_SENTINEL {
                AIVO_STARTER_REAL_URL.to_string()
            } else if pi_api == "google-generative-ai" {
                // Pi sets apiVersion="" when a custom baseUrl is provided,
                // expecting the version to be part of the URL already.
                ensure_google_version_suffix(&key.base_url)
            } else {
                key.base_url.clone()
            };
            let auth: &str = if key.key.is_empty() {
                AIVO_STARTER_SENTINEL
            } else {
                &key.key
            };
            let models_json =
                build_pi_models_json(&resolved_url, auth, pi_api, model, catalog, limits);
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_SETUP_PI_AGENT_DIR".to_string(), "1".to_string());
        }

        env
    }

    /// Merges tool-specific environment variables with the current process environment
    ///
    /// Tool environment variables take precedence over existing process.env values.
    /// Manual environment variables take precedence over tool variables.
    pub fn merge(
        &self,
        tool_env: &HashMap<String, String>,
        manual_env: Option<&HashMap<String, String>>,
    ) -> HashMap<String, String> {
        // Start with current environment
        let mut merged: HashMap<String, String> = std::env::vars().collect();

        // Add tool environment (overrides current env)
        for (key, value) in tool_env {
            merged.insert(key.clone(), value.clone());
        }

        // Add manual environment (overrides tool env)
        if let Some(manual) = manual_env {
            for (key, value) in manual {
                merged.insert(key.clone(), value.clone());
            }
        }

        merged
    }
}

fn strip_aivo_prefix(model: &str) -> &str {
    model.strip_prefix("aivo/").unwrap_or(model)
}

/// Builds a `models.json` string for Pi's custom "aivo" provider.
///
/// Pi reads `models.json` from `PI_CODING_AGENT_DIR` to discover custom providers.
/// The placeholder URL `http://127.0.0.1:0` is patched at runtime with the actual router port.
fn build_pi_models_json(
    base_url: &str,
    api_key: &str,
    api_type: &str,
    model: Option<&str>,
    catalog: &[String],
    limits: &HashMap<String, crate::services::model_metadata::ResolvedLimits>,
) -> String {
    // Pi doesn't auto-discover — its `/model` picker shows only what's listed
    // here. List the whole catalog so it's a real picker. The pinned id stays
    // first so pi defaults to it; the rest follow sorted. Without a pinned
    // model, default to the catalog's first entry — a literal "default"
    // placeholder is not a real id and nothing rewrites it, so upstreams 400
    // ("unknown model 'default'"). The placeholder survives only when
    // discovery failed AND no model is pinned (nothing better exists).
    let default_id = model
        .or_else(|| catalog.first().map(String::as_str))
        .unwrap_or("default");
    let mut rest: Vec<&str> = catalog
        .iter()
        .map(String::as_str)
        .filter(|id| *id != default_id)
        .collect();
    rest.sort_unstable();
    rest.dedup();
    // Pi assumes 128k context for models without an explicit contextWindow;
    // carry the resolved limits so its footer and compaction math match the
    // real model.
    let models: Vec<Value> = std::iter::once(default_id)
        .chain(rest)
        .map(|id| {
            let mut entry = json!({ "id": id, "name": id });
            if let Some(l) = limits.get(id) {
                if let Some(context) = l.context {
                    entry["contextWindow"] = json!(context);
                }
                if let Some(output) = l.output {
                    entry["maxTokens"] = json!(output);
                }
            }
            entry
        })
        .collect();
    let models_json = json!({
        "providers": {
            "aivo": {
                "baseUrl": base_url,
                "apiKey": api_key,
                "api": api_type,
                "models": models,
            }
        }
    });
    models_json.to_string()
}

pub(crate) fn redact_env_value(key: &str, value: &str) -> String {
    if key == "OPENCODE_CONFIG_CONTENT" || key == "AIVO_PI_MODELS_JSON" {
        return "<redacted>".to_string();
    }

    if key.contains("TOKEN") || key.contains("KEY") || key.contains("CREDS") {
        let char_count = value.chars().count();
        if char_count > 12 {
            // Safely slice at character boundaries
            let prefix: String = value.chars().take(8).collect();
            let suffix: String = value.chars().skip(char_count - 4).collect();
            format!("{}...{}", prefix, suffix)
        } else {
            "***".to_string()
        }
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::route_cache::PersistedRoute;
    use crate::services::session_store::OpenAICompatibilityMode;

    /// Pin a tool's default (`""`) route, the v2 equivalent of the old
    /// per-CLI `claude_protocol` / `gemini_protocol` scalar pins.
    fn pin_route(key: &mut ApiKey, tool: &str, protocol: &str) {
        key.protocol_routes
            .entry(tool.to_string())
            .or_default()
            .insert(
                String::new(),
                PersistedRoute {
                    protocol: protocol.to_string(),
                    path_variant: String::new(),
                },
            );
    }

    fn test_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "a1b2".to_string(),
            "test-key".to_string(),
            "http://localhost:8080".to_string(),
            None,
            "sk-test-key-12345".to_string(),
        )
    }

    fn test_api_key(base_url: &str) -> ApiKey {
        let mut k = test_key();
        k.base_url = base_url.to_string();
        k
    }

    /// The per-launch loopback token every router-backed launch must carry.
    fn loopback_token(env: &HashMap<String, String>) -> String {
        env.get("AIVO_ROUTER_AUTH_TOKEN")
            .expect("router launches must carry AIVO_ROUTER_AUTH_TOKEN")
            .clone()
    }

    /// Asserts the child's auth env var holds the loopback token, never the
    /// real key.
    fn assert_auth_is_loopback_token(env: &HashMap<String, String>, auth_env: &str) {
        assert_eq!(env.get(auth_env), Some(&loopback_token(env)));
    }

    /// All `use_direct_*` predicates consult `is_debug_active()`. The
    /// debug-toggling tests serialize via `DEBUG_TEST_MUTEX`; tests that
    /// assume the debug flag is off must take the same mutex (and explicitly
    /// reset the flag) to avoid racing with parallel toggles.
    fn debug_off_guard() -> std::sync::MutexGuard<'static, ()> {
        let guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        guard
    }

    #[test]
    fn for_claude_with_cursor_key_routes_through_cursor_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let env = injector.for_claude(&key, Some("composer-2.5"));

        assert_eq!(env.get("AIVO_USE_CURSOR_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CURSOR_BASE_URL_ENV"),
            Some(&"ANTHROPIC_BASE_URL".to_string())
        );
        assert_eq!(
            env.get("AIVO_CURSOR_KEY_SECRET"),
            Some(&format!(
                "{}testaccount1",
                crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
            ))
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "ANTHROPIC_AUTH_TOKEN");
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        // Model fans into the canonical slots so /model picks up cursor's id.
        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(env.get(slot), Some(&"composer-2.5".to_string()));
        }
        // Cursor routing bypasses the anthropic-to-openai router entirely.
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn for_codex_with_cursor_key_routes_through_cursor_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        let env = injector.for_codex(&key, Some("gpt-5"));

        assert_eq!(env.get("AIVO_USE_CURSOR_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CURSOR_BASE_URL_ENV"),
            Some(&"OPENAI_BASE_URL".to_string())
        );
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "OPENAI_API_KEY");
        assert_eq!(env.get("CODEX_MODEL"), Some(&"gpt-5".to_string()));
        assert_eq!(env.get("OPENAI_DEFAULT_MODEL"), Some(&"gpt-5".to_string()));
        assert!(!env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"));
    }

    #[test]
    fn for_opencode_with_cursor_key_routes_through_cursor_router_via_config_json() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let env = injector.for_opencode(&key, Some("composer-2.5"), None, &HashMap::new());

        assert_eq!(env.get("AIVO_USE_CURSOR_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CURSOR_KEY_SECRET"),
            Some(&format!(
                "{}testaccount1",
                crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
            ))
        );
        let config = env
            .get("OPENCODE_CONFIG_CONTENT")
            .expect("OPENCODE_CONFIG_CONTENT must be set for cursor keys");
        assert!(
            config.contains(PLACEHOLDER_LOOPBACK_URL),
            "OpenCode config must carry the loopback placeholder so launch_runtime can patch it: {config}"
        );
        assert!(config.contains("composer-2.5"));
        assert!(config.contains(&loopback_token(&env)));
        // Cursor wiring bypasses the generic OpenCode routers.
        assert!(!env.contains_key("AIVO_USE_OPENCODE_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER"));
    }

    #[test]
    fn for_pi_with_cursor_key_routes_through_cursor_router_via_models_json() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let env = injector.for_pi(&key, Some("composer-2.5"), &[], &HashMap::new());

        assert_eq!(env.get("AIVO_USE_CURSOR_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CURSOR_KEY_SECRET"),
            Some(&format!(
                "{}testaccount1",
                crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
            ))
        );
        let models_json = env
            .get("AIVO_PI_MODELS_JSON")
            .expect("AIVO_PI_MODELS_JSON must be set for cursor keys");
        assert!(
            models_json.contains(PLACEHOLDER_LOOPBACK_URL),
            "Pi config must carry the loopback placeholder so launch_runtime can patch it: {models_json}"
        );
        assert!(models_json.contains("openai-completions"));
        assert!(models_json.contains("composer-2.5"));
        // Pi reads its upstream from the JSON; don't also force a Pi-specific
        // router that the launcher would try to start as a second instance.
        assert!(!env.contains_key("AIVO_USE_PI_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_PI_COPILOT_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_PI_STARTER_ROUTER"));
        // Pi launches with a temp agent dir that's written *after* the router
        // port is known; the dir-writer is invoked from start_cursor_router's
        // Pi branch in launch_runtime, not from the AIVO_SETUP_PI_AGENT_DIR
        // direct path.
        assert!(!env.contains_key("AIVO_SETUP_PI_AGENT_DIR"));
    }

    #[test]
    fn for_gemini_with_cursor_key_routes_through_cursor_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
        key.key = zeroize::Zeroizing::new(format!(
            "{}testaccount1",
            crate::services::cursor_acp::CURSOR_SHADOW_PREFIX
        ));
        let env = injector.for_gemini(&key, Some("composer-2.5"));

        assert_eq!(env.get("AIVO_USE_CURSOR_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_CURSOR_BASE_URL_ENV"),
            Some(&"GOOGLE_GEMINI_BASE_URL".to_string())
        );
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "GEMINI_API_KEY");
        assert_eq!(env.get("GEMINI_MODEL"), Some(&"composer-2.5".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_FORCE_API_KEY_AUTH"),
            Some(&"1".to_string()),
            "cursor branch must force api-key auth so gemini-cli doesn't fall through to OAuth"
        );
        // Cursor routing bypasses the gemini compat router entirely.
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER"));
    }

    #[test]
    fn test_for_claude_starter_with_anthropic_pin_uses_router_not_direct() {
        // Regression: if the starter key's claude_protocol pin is Anthropic
        // (because upstream_protocol_for_cli prefers client-native for
        // Openai-default hosts), use_direct_anthropic_for_claude returns true
        // — but Direct mode bypasses device_fingerprint injection, so the
        // gateway 403s. Force Routed mode for starter regardless.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        pin_route(&mut key, "claude", "anthropic");
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string()),
            "starter must always route through the local router"
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string()),
        );
    }

    #[test]
    fn test_for_claude_anthropic_native_direct() {
        // Official Anthropic endpoints bypass all routers.
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.anthropic.com/v1".to_string();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.anthropic.com".to_string())
        );
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        assert_eq!(
            env.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert!(!env.contains_key("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"));
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn for_claude_injects_oauth_token_and_unsets_api_key_vars() {
        use crate::services::claude_oauth::{CLAUDE_OAUTH_SENTINEL, ClaudeOAuthCredential};
        let creds = ClaudeOAuthCredential {
            token: "sk-ant-oat01-TEST".into(),
            created_at: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let injector = EnvironmentInjector::new();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("CLAUDE_CODE_OAUTH_TOKEN"),
            Some(&"sk-ant-oat01-TEST".to_string())
        );
        // Conflicting auth vars must be requested for *removal* (not set to
        // empty string) so Claude Code's auth-source detector treats them as
        // truly absent — empty-string is still "set" to that detector, which
        // would flip the session to API-key mode and disable subscription
        // features.
        let unset = env
            .get(AIVO_INTERNAL_ENV_UNSET)
            .expect("OAuth path must request env_remove for conflicting auth vars");
        let names: Vec<&str> = unset.split(',').collect();
        assert!(names.contains(&"ANTHROPIC_API_KEY"));
        assert!(names.contains(&"ANTHROPIC_AUTH_TOKEN"));
        assert!(names.contains(&"ANTHROPIC_BASE_URL"));
        // And explicitly NOT set to empty string.
        assert!(!env.contains_key("ANTHROPIC_API_KEY"));
        assert!(!env.contains_key("ANTHROPIC_AUTH_TOKEN"));
        assert!(!env.contains_key("ANTHROPIC_BASE_URL"));
        assert!(!env.contains_key("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"));
        // No routed-mode indicators.
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn for_claude_oauth_fans_model_into_default_slots() {
        use crate::services::claude_oauth::{CLAUDE_OAUTH_SENTINEL, ClaudeOAuthCredential};
        let creds = ClaudeOAuthCredential {
            token: "sk-ant-oat01-TEST".into(),
            created_at: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        // Dotted form normalizes to the hyphenated native id; same as Direct mode.
        let env = EnvironmentInjector::new().for_claude(&key, Some("claude-opus-4.7"));
        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(
                env.get(slot),
                Some(&"claude-opus-4-7".to_string()),
                "slot {slot} must carry the user's model id for OAuth keys",
            );
        }
    }

    #[test]
    fn for_claude_oauth_appends_max_context_suffix() {
        use crate::services::claude_oauth::{CLAUDE_OAUTH_SENTINEL, ClaudeOAuthCredential};
        let creds = ClaudeOAuthCredential {
            token: "sk-ant-oat01-TEST".into(),
            created_at: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let overrides = ClaudeModelOverrides {
            max_context: Some("1m".to_string()),
            ..Default::default()
        };
        let env = EnvironmentInjector::new().for_claude_with_overrides(
            &key,
            Some("claude-opus-4-7"),
            &overrides,
        );
        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(
                env.get(slot),
                Some(&"claude-opus-4-7[1m]".to_string()),
                "slot {slot} must carry the [1m] suffix for OAuth keys",
            );
        }
    }

    #[test]
    fn for_claude_oauth_honors_per_slot_overrides() {
        use crate::services::claude_oauth::{CLAUDE_OAUTH_SENTINEL, ClaudeOAuthCredential};
        let creds = ClaudeOAuthCredential {
            token: "sk-ant-oat01-TEST".into(),
            created_at: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let overrides = ClaudeModelOverrides {
            opus: Some("claude-opus-4.7".into()),
            sonnet: Some("claude-sonnet-4.6".into()),
            haiku: Some("claude-haiku-4-5".into()),
            reasoning: Some("claude-opus-4.7".into()),
            subagent: Some("claude-haiku-4-5".into()),
            ..Default::default()
        };
        let env = EnvironmentInjector::new().for_claude_with_overrides(&key, None, &overrides);
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            Some(&"claude-opus-4-7".to_string()),
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"claude-sonnet-4-6".to_string()),
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"claude-haiku-4-5".to_string()),
        );
        assert_eq!(
            env.get("ANTHROPIC_REASONING_MODEL"),
            Some(&"claude-opus-4-7".to_string()),
        );
        assert_eq!(
            env.get("CLAUDE_CODE_SUBAGENT_MODEL"),
            Some(&"claude-haiku-4-5".to_string()),
        );
    }

    #[test]
    fn for_claude_oauth_with_corrupt_json_still_unsets_api_key_vars() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            "not valid json".into(),
        );
        let env = EnvironmentInjector::new().for_claude(&key, None);
        // Degraded but safe: no OAuth token → Claude Code fails auth at launch
        // loudly, rather than silently falling through to the user's Keychain
        // account. The unset request blocks that fall-through.
        let unset = env
            .get(AIVO_INTERNAL_ENV_UNSET)
            .expect("OAuth path must request env_remove even on corrupt creds");
        assert!(unset.contains("ANTHROPIC_API_KEY"));
        assert!(unset.contains("ANTHROPIC_AUTH_TOKEN"));
        assert!(!env.contains_key("CLAUDE_CODE_OAUTH_TOKEN"));
    }

    #[test]
    fn test_for_claude_anthropic_native_direct_normalizes_dotted_claude_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.anthropic.com/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4.6"));

        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
    }

    #[test]
    fn test_for_claude_minimax_anthropic_endpoint_direct() {
        // Defensive: ensure no other test left FORCE_DEBUG_ACTIVE on, which
        // would force the bridge and bust the direct-mode assertions below.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        let env = injector.for_claude(&key, Some("MiniMax-M1"));

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(env.get("ANTHROPIC_MODEL"), Some(&"MiniMax-M1".to_string()));
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_minimax_anthropic_v1_endpoint_direct() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic/v1".to_string();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_minimax_routes_through_bridge_under_debug() {
        // When `--debug` is active, native-Anthropic upstreams must route
        // through the local bridge so the bridge's `.send_logged()` sites
        // capture the outbound translation/forward call. Without this, claude
        // execs straight to upstream and `--debug` produces an empty log.
        // The override is `is_debug_active() => use_direct_*` returns false,
        // which falls into the routed branch.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(true);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        let env = injector.for_claude(&key, Some("MiniMax-M1"));

        // Routed mode: ANTHROPIC_BASE_URL points at the loopback placeholder
        // (launch_runtime patches it with the actual port at exec time), and
        // the AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER flag plus base URL env var
        // are set so the bridge knows where to forward.
        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string()),
            "expected router flag under --debug; got env keys: {:?}",
            env.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );

        crate::services::http_debug::set_test_debug_active(false);
    }

    #[test]
    fn test_for_claude_minimax_direct_when_debug_inactive() {
        // With `--debug` off, behavior is unchanged — minimax
        // anthropic-protocol endpoints stay in direct mode.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        let env = injector.for_claude(&key, Some("MiniMax-M1"));

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_protocol_override_anthropic_routes_on_non_native_host() {
        // Even when claude_protocol pins Anthropic, a non-native base URL
        // must go through the router so its protocol-fallback path can try
        // /v1/messages first and downgrade to /v1/chat/completions on 404.
        // Direct mode requires a genuinely Anthropic-native endpoint.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example.com/v1".to_string();
        pin_route(&mut key, "claude", "anthropic");
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string()),
            "non-native host must route through the Anthropic-to-OpenAI router"
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.example.com/v1".to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"anthropic".to_string()),
            "router should target the pinned Anthropic upstream"
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
    }

    #[test]
    fn test_for_claude_protocol_override_anthropic_to_openai_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        pin_route(&mut key, "claude", "openai");
        let env = injector.for_claude(&key, Some("MiniMax-M1"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
    }

    #[test]
    fn test_for_claude_router_uses_learned_protocol_override() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://example.com/custom".to_string();
        pin_route(&mut key, "claude", "google");

        let env = injector.for_claude(&key, None);
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"google".to_string())
        );
    }

    #[test]
    fn test_for_claude_unknown_endpoint_uses_anthropic_to_openai_router() {
        // Any non-Anthropic, non-OpenRouter, non-Copilot URL goes through the Anthropic-to-OpenAI router.
        let injector = EnvironmentInjector::new();
        let key = test_key(); // http://localhost:8080
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_for_claude_unknown_endpoint_targets_anthropic_upstream() {
        // Regression for the protocol-native default: an unknown host should
        // forward Anthropic upstream so a multi-protocol gateway sees the
        // client's native protocol. Protocol fallback handles OpenAI-only
        // hosts via 404 downgrade.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example-gateway.dev".to_string();
        // No claude_protocol pinned.
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"anthropic".to_string()),
            "unknown host + Claude should target Anthropic upstream",
        );
    }

    #[test]
    fn test_for_claude_openai_endpoint_targets_anthropic_upstream() {
        // Even for api.openai.com: Claude Code is already emitting /v1/messages,
        // so we forward that as-is. Protocol fallback handles the one-shot 404
        // and the key pin sticks for subsequent launches.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.openai.com/v1".to_string();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"anthropic".to_string()),
        );
    }

    #[test]
    fn test_for_claude_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, Some("claude-3-opus"));

        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        // The deprecated ANTHROPIC_SMALL_FAST_MODEL is intentionally NOT set;
        // ANTHROPIC_DEFAULT_HAIKU_MODEL covers the same routing.
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_REASONING_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
    }

    #[test]
    fn for_claude_per_slot_overrides_replace_only_targeted_slots() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let overrides = ClaudeModelOverrides {
            haiku: Some("custom-haiku".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, Some("claude-opus-4-7"), &overrides);

        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-opus-4-7".to_string()),
            "main slot keeps the -m value",
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"custom-haiku".to_string()),
            "haiku family slot is overridden",
        );
        // Deprecated ANTHROPIC_SMALL_FAST_MODEL must not be set anywhere.
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
        // The other slots fanned out from -m and stay on opus.
        for slot in [
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_REASONING_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ] {
            assert_eq!(
                env.get(slot),
                Some(&"claude-opus-4-7".to_string()),
                "unrelated slot {slot} keeps the fanned-out -m value",
            );
        }
    }

    #[test]
    fn for_claude_per_slot_overrides_without_main_only_set_named_slots() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let overrides = ClaudeModelOverrides {
            sonnet: Some("custom-sonnet".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, None, &overrides);

        // No -m → no fan-out. Only the explicitly overridden slots are set.
        assert!(!env.contains_key("ANTHROPIC_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_DEFAULT_HAIKU_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_DEFAULT_OPUS_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_REASONING_MODEL"));
        assert!(!env.contains_key("CLAUDE_CODE_SUBAGENT_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"custom-sonnet".to_string()),
        );
    }

    #[test]
    fn for_claude_max_context_1m_appends_suffix_to_default_slots() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.deepseek.com/anthropic".to_string();
        let overrides = ClaudeModelOverrides {
            max_context: Some("1m".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, Some("deepseek-v4-flash"), &overrides);

        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(
                env.get(slot),
                Some(&"deepseek-v4-flash[1m]".to_string()),
                "slot {slot} must carry the model with the [1m] suffix appended",
            );
        }
        // Direct mode for the Anthropic-shaped upstream is preserved —
        // max_context is a model-name annotation, not a routing knob.
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").map(|s| s.as_str()),
            Some("https://api.deepseek.com/anthropic"),
        );
    }

    #[test]
    fn for_claude_max_context_2m_appends_2m_suffix_to_default_slots() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.deepseek.com/anthropic".to_string();
        let overrides = ClaudeModelOverrides {
            max_context: Some("2m".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, Some("deepseek-v4-flash"), &overrides);

        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(
                env.get(slot),
                Some(&"deepseek-v4-flash[2m]".to_string()),
                "slot {slot} must carry the model with the [2m] suffix appended",
            );
        }
    }

    #[test]
    fn for_claude_max_context_1m_leaves_slot_overrides_verbatim() {
        // Per-slot overrides may name a model that doesn't support 1M
        // context, so the suffix must not be auto-appended there.
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.deepseek.com/anthropic".to_string();
        let overrides = ClaudeModelOverrides {
            max_context: Some("1m".to_string()),
            haiku: Some("small-fast-model".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, Some("deepseek-v4-flash"), &overrides);

        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"small-fast-model".to_string()),
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"deepseek-v4-flash[1m]".to_string()),
        );
    }

    #[test]
    fn for_claude_max_context_unset_omits_suffix() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.deepseek.com/anthropic".to_string();
        let env = injector.for_claude(&key, Some("deepseek-v4-pro"));
        for slot in CLAUDE_DEFAULT_MODEL_SLOTS {
            assert_eq!(
                env.get(slot),
                Some(&"deepseek-v4-pro".to_string()),
                "without --max-context, slot {slot} must keep the user's model unchanged",
            );
        }
    }

    #[test]
    fn for_claude_direct_mode_forces_fine_grained_tool_streaming_on() {
        // ANTHROPIC_BASE_URL being set makes Claude Code default fine-grained
        // tool-input streaming OFF (treats it as a gateway). For Direct mode
        // the upstream is a real Anthropic-shaped endpoint, so force it back on.
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.anthropic.com/v1".to_string();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING"),
            Some(&"1".to_string()),
        );
        assert!(!env.contains_key("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"));
    }

    #[test]
    fn for_claude_routed_mode_strips_experimental_betas() {
        // Routed/Ollama/Copilot/OpenRouter all bridge to OpenAI-shaped
        // upstreams via aivo's loopback router. Beta tool-schema fields
        // (defer_loading, eager_input_streaming) and anthropic-beta headers
        // are meaningless there and risk 400s on strict gateways.
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"),
            Some(&"1".to_string()),
        );
        assert!(!env.contains_key("CLAUDE_CODE_ENABLE_FINE_GRAINED_TOOL_STREAMING"));
    }

    #[test]
    fn for_claude_never_writes_deprecated_small_fast_model_slot() {
        // Regression guard: aivo must not propagate the deprecated
        // ANTHROPIC_SMALL_FAST_MODEL env var. Anthropic replaced it with
        // ANTHROPIC_DEFAULT_HAIKU_MODEL, which is what we set instead.
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, Some("claude-opus-4-7"));
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"claude-opus-4-7".to_string()),
        );
    }

    #[test]
    fn for_claude_family_default_overrides_target_distinct_env_vars() {
        // Each of haiku/sonnet/opus must land in its own ANTHROPIC_DEFAULT_*_MODEL
        // slot — these are the slots Claude Code's `/model` UI exposes, so a
        // typo would silently misroute one slot to another's env var.
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let overrides = ClaudeModelOverrides {
            haiku: Some("h-model".to_string()),
            sonnet: Some("s-model".to_string()),
            opus: Some("o-model".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, None, &overrides);

        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
            Some(&"h-model".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_SONNET_MODEL"),
            Some(&"s-model".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_DEFAULT_OPUS_MODEL"),
            Some(&"o-model".to_string())
        );
    }

    #[test]
    fn for_claude_per_slot_overrides_normalize_in_direct_mode() {
        // Direct Anthropic mode normalizes dotted versions like 4.6 → 4-6.
        // Per-slot overrides should pass through the same normalization so a
        // user passing `--reasoning-model claude-sonnet-4.6` doesn't get a
        // 404 from the native Anthropic endpoint.
        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.anthropic.com"); // → Direct mode.
        let overrides = ClaudeModelOverrides {
            reasoning: Some("claude-sonnet-4.6".to_string()),
            ..Default::default()
        };

        let env = injector.for_claude_with_overrides(&key, None, &overrides);

        assert_eq!(
            env.get("ANTHROPIC_REASONING_MODEL"),
            Some(&"claude-sonnet-4-6".to_string()),
            "dotted version should be normalized for Direct mode",
        );
    }

    #[test]
    fn for_claude_routed_mode_surfaces_model_in_picker() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.example.com/v1");
        let env = injector.for_claude(&key, Some("deepseek-chat"));

        assert_eq!(
            env.get("ANTHROPIC_CUSTOM_MODEL_OPTION"),
            Some(&"deepseek-chat".to_string()),
        );
        assert_eq!(
            env.get("ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"),
            Some(&"Routed via aivo (test-key)".to_string()),
        );
    }

    #[test]
    fn for_claude_routed_mode_includes_max_context_suffix_in_picker() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.example.com/v1");
        let overrides = ClaudeModelOverrides {
            max_context: Some("1m".to_string()),
            ..Default::default()
        };
        let env = injector.for_claude_with_overrides(&key, Some("deepseek-chat"), &overrides);

        assert_eq!(
            env.get("ANTHROPIC_CUSTOM_MODEL_OPTION"),
            Some(&"deepseek-chat[1m]".to_string()),
        );
    }

    #[test]
    fn for_claude_direct_anthropic_mode_skips_custom_model_option() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.anthropic.com");
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        assert!(!env.contains_key("ANTHROPIC_CUSTOM_MODEL_OPTION"));
        assert!(!env.contains_key("ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION"));
    }

    #[test]
    fn for_claude_oauth_skips_custom_model_option() {
        use crate::services::claude_oauth::{CLAUDE_OAUTH_SENTINEL, ClaudeOAuthCredential};
        let creds = ClaudeOAuthCredential {
            token: "sk-ant-oat01-TEST".into(),
            created_at: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "id".into(),
            "work".into(),
            CLAUDE_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let env = EnvironmentInjector::new().for_claude(&key, Some("claude-opus-4-7"));
        assert!(!env.contains_key("ANTHROPIC_CUSTOM_MODEL_OPTION"));
    }

    #[test]
    fn for_claude_routed_mode_without_model_skips_custom_model_option() {
        let _guard = debug_off_guard();
        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.example.com/v1");
        let env = injector.for_claude(&key, None);

        assert!(!env.contains_key("ANTHROPIC_CUSTOM_MODEL_OPTION"));
    }

    #[test]
    fn test_for_claude_openrouter_model_transformation() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-haiku-4-5"));

        // With built-in router: model names pass through unchanged
        // Router handles transformation
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-haiku-4-5".to_string())
        );
        // Router should be started
        assert_eq!(env.get("AIVO_USE_ROUTER"), Some(&"1".to_string()));
        // Base URL is a placeholder; AI launcher overwrites with actual port after binding
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
    }

    #[test]
    fn test_for_claude_openrouter_sonnet_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        // Model name passes through unchanged - router will transform it
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
        // Verify router configuration is set
        assert_eq!(env.get("AIVO_ROUTER_API_KEY"), Some(&key.key.to_string()));
    }

    #[test]
    fn test_for_claude_openrouter_opus_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-opus-4-6"));

        // Model passes through unchanged - router transforms
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-opus-4-6".to_string())
        );
    }

    #[test]
    fn test_for_claude_openrouter_future_models() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();

        // All models pass through unchanged - router handles transformation
        let env = injector.for_claude(&key, Some("claude-some-model-5-10"));
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-some-model-5-10".to_string())
        );
    }

    #[test]
    fn test_for_claude_non_claude_model_no_transformation() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("gpt-4o"));

        // Non-Claude models should not be transformed
        assert_eq!(env.get("ANTHROPIC_MODEL"), Some(&"gpt-4o".to_string()));
    }

    #[test]
    fn test_router_integration_example() {
        // The built-in router is always used for OpenRouter
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        // Placeholder; AI launcher overwrites with the actual random port after binding
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        // Model name passes through unchanged - router transforms it
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4-6".to_string())
        );
        // Router configuration is set
        assert_eq!(env.get("AIVO_USE_ROUTER"), Some(&"1".to_string()));
    }

    #[test]
    fn test_for_claude_cloudflare_uses_anthropic_to_openai_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1".to_string();
        let env = injector.for_claude(&key, Some("llama-3.1-8b"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.cloudflare.com/client/v4/accounts/abc/ai/v1".to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MODEL_PREFIX"),
            Some(&"@cf/".to_string())
        );
    }

    #[test]
    fn test_for_claude_openai_uses_anthropic_to_openai_router() {
        // api.openai.com is an OpenAI-compatible endpoint, so it goes through the Anthropic-to-OpenAI router.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.openai.com/v1".to_string();
        let env = injector.for_claude(&key, Some("gpt-4o"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.openai.com/v1".to_string())
        );
    }

    #[test]
    fn test_for_claude_moonshot_uses_anthropic_to_openai_router_with_reasoning() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.moonshot.cn/v1".to_string();
        let env = injector.for_claude(&key, Some("kimi-k2.5"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL"),
            Some(&"https://api.moonshot.cn/v1".to_string())
        );
        assert!(!env.contains_key("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MODEL_PREFIX"));
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_for_codex_non_openai_uses_router() {
        // test_key() uses http://localhost:8080 (non-OpenAI) → router enabled
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_codex(&key, None);

        // Placeholder; AI launcher overwrites with actual port after binding
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "OPENAI_API_KEY");
        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
    }

    #[test]
    fn test_for_codex_unknown_endpoint_targets_responses_upstream() {
        // Unknown host + Codex should target the Responses API upstream so a
        // multi-protocol gateway sees Codex's native protocol. Protocol
        // fallback downgrades on 404 for plain Chat-Completions-only hosts.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example-gateway.dev".to_string();
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"responses".to_string()),
        );
    }

    #[test]
    fn test_for_codex_routed_openai_host_still_seeds_responses_first() {
        // Pin: when codex spawns in routed mode against a host whose detected
        // protocol is Openai (api.openai.com), the cascade must still seed
        // `/v1/responses` first. Guards against a future "simplify
        // upstream_protocol_for_cli" change silently flipping the seed back
        // to the detected protocol.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.openai.com/v1".to_string();
        key.codex_mode = Some(OpenAICompatibilityMode::Router);
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string()),
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"responses".to_string()),
        );
    }

    #[test]
    fn test_for_codex_starter_with_direct_mode_uses_router_not_direct() {
        // Defense-in-depth: mirrors the Claude/Gemini starter guard. Even if
        // codex_mode is pinned to Direct, a starter key must route through
        // the local router so device fingerprint headers attach.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        key.codex_mode = Some(OpenAICompatibilityMode::Direct);
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string()),
            "starter must always route through the local router"
        );
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string()),
        );
    }

    #[test]
    fn test_for_codex_official_openai_direct() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.openai.com/v1".to_string();
        let env = injector.for_codex(&key, None);

        // Direct connection: no router, use actual base URL
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&"https://api.openai.com/v1".to_string())
        );
        assert_eq!(
            env.get("OPENAI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"));
    }

    #[test]
    fn test_for_codex_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_codex(&key, Some("o3"));

        assert_eq!(env.get("CODEX_MODEL"), Some(&"o3".to_string()));
        assert_eq!(env.get("OPENAI_DEFAULT_MODEL"), Some(&"o3".to_string()));
        assert_eq!(env.get("CODEX_MODEL_DEFAULT"), Some(&"o3".to_string()));
    }

    #[test]
    fn test_for_codex_vercel_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"https://ai-gateway.vercel.sh/v1".to_string())
        );
    }

    #[test]
    fn test_for_codex_openrouter_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_codex(&key, None);

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"https://openrouter.ai/api/v1".to_string())
        );
    }

    #[test]
    fn test_for_codex_cloudflare_uses_router_with_prefix() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1".to_string();
        let env = injector.for_codex(&key, Some("glm-4.7-flash"));

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"https://api.cloudflare.com/client/v4/accounts/abc/ai/v1".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_MODEL_PREFIX"),
            Some(&"@cf/".to_string())
        );
        // Model should still be set
        assert_eq!(env.get("CODEX_MODEL"), Some(&"glm-4.7-flash".to_string()));
    }

    #[test]
    fn test_for_gemini() {
        let injector = EnvironmentInjector::new();
        let key = test_key(); // base_url = http://localhost:8080 (non-Google → router)
        let env = injector.for_gemini(&key, None);

        // Non-Google URL: placeholder is used, router env vars are set
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "GEMINI_API_KEY");
        assert!(!env.contains_key("GEMINI_MODEL"));
    }

    #[test]
    fn test_for_gemini_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_gemini(&key, Some("google/gemini-2.0-flash"));
        assert_eq!(
            env.get("GEMINI_MODEL"),
            Some(&"google/gemini-2.0-flash".to_string())
        );
    }

    #[test]
    fn test_for_gemini_starter_with_google_pin_uses_router_not_direct() {
        // Same regression as Claude: starter must always route through the
        // Gemini router so device fingerprint headers attach.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        pin_route(&mut key, "gemini", "google");
        let env = injector.for_gemini(&key, None);

        assert_eq!(
            env.get("AIVO_USE_GEMINI_ROUTER"),
            Some(&"1".to_string()),
            "starter must always route through the local Gemini router"
        );
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string()),
        );
    }

    #[test]
    fn test_for_gemini_starter_sets_default_model() {
        // Gemini CLI's own default can be a Google-only model. Starter must
        // launch with aivo/starter even when the user did not pass -m.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        let env = injector.for_gemini(&key, None);

        assert_eq!(
            env.get("GEMINI_MODEL"),
            Some(&AIVO_STARTER_MODEL.to_string())
        );
        assert_eq!(
            env.get("AIVO_GEMINI_MODEL_CONFIG_MODEL"),
            Some(&AIVO_STARTER_MODEL.to_string())
        );
    }

    #[test]
    fn test_for_gemini_native_google_no_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/".to_string();
        let env = injector.for_gemini(&key, None);
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        // Trailing slash stripped; SDK adds /v1beta itself
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"https://generativelanguage.googleapis.com".to_string())
        );
    }

    #[test]
    fn test_for_gemini_native_google_strips_v1beta_suffix() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
        let env = injector.for_gemini(&key, None);
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        // /v1beta stripped; the Gemini CLI's @google/genai SDK adds it
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"https://generativelanguage.googleapis.com".to_string())
        );
    }

    #[test]
    fn test_for_gemini_protocol_override_google_routes_on_non_native_host() {
        // Same invariant as the Claude side: gemini_protocol pinning Google
        // on a non-Google host must still route through the Gemini router
        // so protocol fallback can kick in. Direct mode requires a genuinely
        // Google-native endpoint (generativelanguage.googleapis.com).
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example.com".to_string();
        pin_route(&mut key, "gemini", "google");
        let env = injector.for_gemini(&key, None);

        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL"),
            Some(&"https://api.example.com".to_string())
        );
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"google".to_string()),
            "router should target the pinned Google upstream"
        );
    }

    #[test]
    fn test_for_gemini_native_google_strips_provider_prefix_from_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
        let env = injector.for_gemini(&key, Some("google/gemini-2.0-flash"));

        assert_eq!(
            env.get("GEMINI_MODEL"),
            Some(&"gemini-2.0-flash".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
    }

    #[test]
    fn test_for_gemini_router_uses_learned_protocol_override() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://example.com/custom".to_string();
        pin_route(&mut key, "gemini", "anthropic");

        let env = injector.for_gemini(&key, None);
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"anthropic".to_string())
        );
    }

    #[test]
    fn test_for_gemini_non_google_uses_router() {
        let injector = EnvironmentInjector::new();
        let key = test_key(); // base_url = http://localhost:8080 (non-Google)
        let env = injector.for_gemini(&key, None);
        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );
        // Placeholder — launcher overwrites with actual port
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
    }

    #[test]
    fn test_for_gemini_unknown_endpoint_targets_google_upstream() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example-gateway.dev".to_string();
        let env = injector.for_gemini(&key, None);

        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"google".to_string()),
            "unknown host + Gemini should target Google upstream",
        );
    }

    #[test]
    fn test_for_gemini_vercel_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://ai-gateway.vercel.sh/v1".to_string();
        let env = injector.for_gemini(&key, None);
        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL"),
            Some(&"https://ai-gateway.vercel.sh/v1".to_string())
        );
    }

    #[test]
    fn test_for_gemini_copilot_uses_copilot_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_gemini(&key, None);
        assert_eq!(
            env.get("AIVO_USE_GEMINI_COPILOT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "GEMINI_API_KEY");
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        assert!(!env.contains_key("AIVO_GEMINI_COPILOT_FORCED_MODEL"));
    }

    #[test]
    fn test_for_gemini_copilot_with_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_gemini(&key, Some("gpt-4o"));
        assert_eq!(
            env.get("AIVO_USE_GEMINI_COPILOT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_GEMINI_COPILOT_FORCED_MODEL"),
            Some(&"gpt-4o".to_string())
        );
        assert_eq!(env.get("GEMINI_MODEL"), Some(&"gpt-4o".to_string()));
    }

    #[test]
    fn test_for_opencode() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, None, None, &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["$schema"], "https://opencode.ai/config.json");
        assert_eq!(
            config["provider"]["aivo"]["npm"],
            "@ai-sdk/openai-compatible"
        );
        assert_eq!(config["provider"]["aivo"]["name"], "aivo");
        assert_eq!(
            config["provider"]["aivo"]["options"]["baseURL"],
            "http://localhost:8080"
        );
        assert_eq!(
            config["provider"]["aivo"]["options"]["apiKey"],
            "sk-test-key-12345"
        );
        assert!(config.get("model").is_none());
    }

    #[test]
    fn for_opencode_emits_model_limits_when_cascade_knows_both() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let mut limits = HashMap::new();
        limits.insert(
            "deepseek-chat".to_string(),
            crate::services::model_metadata::ResolvedLimits {
                context: Some(128_000),
                output: Some(8_000),
                caps: None,
            },
        );
        // Raw catalog id carries the `aivo/` prefix the config id strips.
        limits.insert(
            "aivo/starter".to_string(),
            crate::services::model_metadata::ResolvedLimits {
                context: Some(1_000_000),
                output: Some(384_000),
                caps: None,
            },
        );
        // Context-only must not emit a half-filled limit.
        limits.insert(
            "mystery".to_string(),
            crate::services::model_metadata::ResolvedLimits {
                context: Some(64_000),
                output: None,
                caps: None,
            },
        );
        let discovered = vec![
            "deepseek-chat".to_string(),
            "aivo/starter".to_string(),
            "mystery".to_string(),
        ];
        let env = injector.for_opencode(&key, None, Some(&discovered), &limits);

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        let models = &config["provider"]["aivo"]["models"];
        assert_eq!(models["deepseek-chat"]["limit"]["context"], 128_000);
        assert_eq!(models["deepseek-chat"]["limit"]["output"], 8_000);
        assert_eq!(models["starter"]["limit"]["context"], 1_000_000);
        assert_eq!(models["starter"]["limit"]["output"], 384_000);
        assert!(models["mystery"].get("limit").is_none());
    }

    #[test]
    fn test_for_opencode_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, Some("gpt-5"), None, &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
    }

    #[test]
    fn test_for_opencode_with_prefixed_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, Some("aivo/gpt-5"), None, &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
    }

    #[test]
    fn test_for_opencode_with_discovered_models() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let discovered = vec!["gpt-4o".to_string(), "claude-sonnet-4".to_string()];
        let env = injector.for_opencode(&key, None, Some(&discovered), &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert!(config.get("model").is_none());
        assert_eq!(
            config["provider"]["aivo"]["models"]["claude-sonnet-4"]["name"],
            "claude-sonnet-4"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-4o"]["name"],
            "gpt-4o"
        );
    }

    #[test]
    fn test_for_opencode_with_model_and_discovered_models() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let discovered = vec!["gpt-4o".to_string(), "claude-sonnet-4".to_string()];
        let env = injector.for_opencode(&key, Some("gpt-5"), Some(&discovered), &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(config["model"], "aivo/gpt-5");
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-5"]["name"],
            "gpt-5"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["gpt-4o"]["name"],
            "gpt-4o"
        );
        assert_eq!(
            config["provider"]["aivo"]["models"]["claude-sonnet-4"]["name"],
            "claude-sonnet-4"
        );
    }

    #[test]
    fn test_for_opencode_starter_does_not_pin_actual_model() {
        // Regression: starter+opencode used to hardcode actual_model to
        // `aivo/starter`, which clobbered the body model on every request.
        // Now there is no static pin — the body's per-request model is
        // preserved (then dynamically re-prefixed by the router), which
        // restores mid-session model switching inside opencode.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        let env = injector.for_opencode(&key, Some("minimax/minimax-m2.7"), None, &HashMap::new());

        assert_eq!(env.get("AIVO_IS_STARTER"), Some(&"1".to_string()));
        assert!(
            !env.contains_key("AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL"),
            "starter+opencode must not pin actual_model — let the body's \
             per-request model flow through so UI model switches take effect",
        );
    }

    #[test]
    fn test_for_opencode_starter_publishes_aivo_prefix_models_for_router() {
        // The router needs the bare ids of catalog entries that originally
        // had the `aivo/` prefix so it can re-add the prefix on each
        // request. Without this, opencode's SDK strips `aivo/starter` to
        // `starter` and the upstream replies "model not found: starter".
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        let catalog = vec![
            "aivo/starter".to_string(),
            "minimax/minimax-m2.7".to_string(),
        ];

        let env = injector.for_opencode(&key, Some("starter"), Some(&catalog), &HashMap::new());
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_AIVO_PREFIX_MODELS"),
            Some(&"starter".to_string()),
        );
    }

    #[test]
    fn test_for_opencode_starter_omits_aivo_prefix_env_when_catalog_has_none() {
        // No env var if no catalog entry uses the `aivo/` prefix — keeps
        // the env clean for providers that don't need re-prefixing.
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = AIVO_STARTER_SENTINEL.to_string();
        let catalog = vec!["minimax/minimax-m2.7".to_string()];

        let env = injector.for_opencode(
            &key,
            Some("minimax/minimax-m2.7"),
            Some(&catalog),
            &HashMap::new(),
        );
        assert!(!env.contains_key("AIVO_RESPONSES_TO_CHAT_ROUTER_AIVO_PREFIX_MODELS"));
    }

    #[test]
    fn test_for_opencode_copilot_uses_placeholder_url() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_opencode(&key, None, None, &HashMap::new());

        // Must set the router trigger env vars
        assert_eq!(
            env.get("AIVO_USE_OPENCODE_COPILOT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("AIVO_COPILOT_GITHUB_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );

        // Config must use placeholder URL (not the magic string "copilot")
        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(
            config["provider"]["aivo"]["options"]["baseURL"],
            PLACEHOLDER_LOOPBACK_URL
        );
        assert_eq!(
            config["provider"]["aivo"]["options"]["apiKey"].as_str(),
            Some(loopback_token(&env).as_str())
        );
    }

    #[test]
    fn test_for_opencode_router_uses_placeholder_url() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.opencode_mode = Some(OpenAICompatibilityMode::Router);
        let env = injector.for_opencode(&key, Some("gpt-4o"), None, &HashMap::new());

        assert_eq!(env.get("AIVO_USE_OPENCODE_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"http://localhost:8080".to_string())
        );

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert_eq!(
            config["provider"]["aivo"]["options"]["baseURL"],
            PLACEHOLDER_LOOPBACK_URL
        );
        assert_eq!(
            config["provider"]["aivo"]["options"]["apiKey"].as_str(),
            Some(loopback_token(&env).as_str())
        );
    }

    #[test]
    fn test_merge() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let tool_env = injector.for_claude(&key, None);
        let merged = injector.merge(&tool_env, None);

        // Should contain all the tool env vars
        assert!(merged.contains_key("ANTHROPIC_BASE_URL"));
        assert!(merged.contains_key("ANTHROPIC_API_KEY"));
    }

    #[test]
    fn test_for_claude_copilot_uses_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_claude(&key, Some("claude-sonnet-4"));

        assert_eq!(env.get("AIVO_USE_COPILOT_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_COPILOT_GITHUB_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "ANTHROPIC_AUTH_TOKEN");
        // Should NOT set OpenRouter router
        assert!(!env.contains_key("AIVO_USE_ROUTER"));
        // Model should still be set
        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-sonnet-4".to_string())
        );
    }

    #[test]
    fn test_for_codex_copilot_uses_copilot_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_codex(&key, None);
        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "OPENAI_API_KEY");
        assert_eq!(
            env.get("AIVO_COPILOT_GITHUB_TOKEN"),
            Some(&"sk-test-key-12345".to_string())
        );
        // Should NOT set the regular codex router
        assert!(!env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"));
    }

    #[test]
    fn test_for_codex_copilot_with_model() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_codex(&key, Some("gpt-4o"));
        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER"),
            Some(&"1".to_string())
        );
        // model env vars should still be set
        assert_eq!(env.get("CODEX_MODEL"), Some(&"gpt-4o".to_string()));
        assert_eq!(env.get("OPENAI_DEFAULT_MODEL"), Some(&"gpt-4o".to_string()));
        assert_eq!(env.get("CODEX_MODEL_DEFAULT"), Some(&"gpt-4o".to_string()));
    }

    // --- Ollama tests ---

    fn ollama_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "oll1".to_string(),
            "ollama".to_string(),
            "ollama".to_string(),
            None,
            "ollama-local".to_string(),
        )
    }

    #[test]
    fn test_for_claude_ollama_uses_anthropic_to_openai_router() {
        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_claude(&key, Some("llama3.2"));

        assert_eq!(
            env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "ANTHROPIC_AUTH_TOKEN");
        assert_eq!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY"),
            Some(&"ollama".to_string())
        );
        assert!(
            env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL")
                .unwrap()
                .contains("11434")
        );
        assert_eq!(env.get("ANTHROPIC_MODEL"), Some(&"llama3.2".to_string()));
        // Should NOT use Copilot or OpenRouter routers
        assert!(!env.contains_key("AIVO_USE_COPILOT_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_ROUTER"));
    }

    #[test]
    fn test_for_codex_ollama_uses_responses_to_chat_router() {
        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_codex(&key, Some("llama3.2"));

        assert_eq!(
            env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"),
            Some(&"1".to_string())
        );
        assert_eq!(
            env.get("OPENAI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "OPENAI_API_KEY");
        assert!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL")
                .unwrap()
                .contains("11434")
        );
        assert_eq!(env.get("CODEX_MODEL"), Some(&"llama3.2".to_string()));
        assert!(!env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER"));
    }

    #[test]
    fn test_for_gemini_ollama_uses_gemini_router() {
        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_gemini(&key, Some("llama3.2"));

        assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&PLACEHOLDER_LOOPBACK_URL.to_string())
        );
        assert_auth_is_loopback_token(&env, "GEMINI_API_KEY");
        assert!(
            env.get("AIVO_GEMINI_ROUTER_BASE_URL")
                .unwrap()
                .contains("11434")
        );
        assert_eq!(env.get("GEMINI_MODEL"), Some(&"llama3.2".to_string()));
        assert!(!env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER"));
    }

    #[test]
    fn test_for_opencode_ollama_uses_direct_connection() {
        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_opencode(&key, Some("llama3.2"), None, &HashMap::new());

        let config: Value =
            serde_json::from_str(env.get("OPENCODE_CONFIG_CONTENT").unwrap()).unwrap();
        assert!(
            config["provider"]["aivo"]["options"]["baseURL"]
                .as_str()
                .unwrap()
                .contains("11434")
        );
        assert_eq!(config["provider"]["aivo"]["options"]["apiKey"], "ollama");
        assert_eq!(config["model"], "aivo/llama3.2");
        // No router needed for OpenCode
        assert!(!env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_OPENCODE_ROUTER"));
    }

    #[test]
    fn test_for_pi_google_preserves_v1beta_suffix() {
        // for_pi now consults is_debug_active(); take the same mutex the
        // debug-toggling tests use so a parallel toggle can't flip us into
        // the routed branch.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
        let env = injector.for_pi(&key, Some("gemini-2.5-flash"), &[], &HashMap::new());

        let models_json = env.get("AIVO_PI_MODELS_JSON").unwrap();
        let parsed: Value = serde_json::from_str(models_json).unwrap();
        // Pi sets apiVersion="" and expects version in the URL
        assert_eq!(
            parsed["providers"]["aivo"]["baseUrl"],
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(parsed["providers"]["aivo"]["api"], "google-generative-ai");
    }

    #[test]
    fn test_for_pi_google_adds_v1beta_when_missing() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com".to_string();
        let env = injector.for_pi(&key, Some("gemini-2.5-flash"), &[], &HashMap::new());

        let models_json = env.get("AIVO_PI_MODELS_JSON").unwrap();
        let parsed: Value = serde_json::from_str(models_json).unwrap();
        // Pi needs /v1beta in the URL since it sets apiVersion=""
        assert_eq!(
            parsed["providers"]["aivo"]["baseUrl"],
            "https://generativelanguage.googleapis.com/v1beta"
        );
        assert_eq!(parsed["providers"]["aivo"]["api"], "google-generative-ai");
    }

    #[test]
    fn test_for_pi_ollama_uses_direct_connection() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_pi(&key, Some("llama3.2"), &[], &HashMap::new());

        let models_json = env.get("AIVO_PI_MODELS_JSON").unwrap();
        let parsed: Value = serde_json::from_str(models_json).unwrap();
        assert!(
            parsed["providers"]["aivo"]["baseUrl"]
                .as_str()
                .unwrap()
                .contains("11434")
        );
        assert_eq!(parsed["providers"]["aivo"]["apiKey"], "ollama");
        assert_eq!(parsed["providers"]["aivo"]["api"], "openai-completions");
        assert_eq!(env.get("AIVO_SETUP_PI_AGENT_DIR"), Some(&"1".to_string()));
        assert!(!env.contains_key("AIVO_USE_PI_COPILOT_ROUTER"));
    }

    #[test]
    fn test_for_pi_routes_through_bridge_under_debug() {
        // Without this routing, pi talks straight to upstream and the JSONL
        // logger captures nothing — pi has no aivo `send_logged` instrumentation.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(true);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        let env = injector.for_pi(&key, Some("MiniMax-M1"), &[], &HashMap::new());

        assert_eq!(env.get("AIVO_USE_PI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL"),
            Some(&"anthropic".to_string()),
            "minimax /anthropic should map to anthropic upstream protocol"
        );
        // Pi's models.json points at the loopback placeholder; launch_runtime
        // substitutes the real port after the router binds.
        let models_json = env.get("AIVO_PI_MODELS_JSON").unwrap();
        assert!(models_json.contains(PLACEHOLDER_LOOPBACK_URL));
        assert!(!env.contains_key("AIVO_SETUP_PI_AGENT_DIR"));

        crate::services::http_debug::set_test_debug_active(false);
    }

    #[test]
    fn test_for_pi_direct_when_debug_inactive() {
        // With --debug off, behavior is unchanged: direct connection.
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        let env = injector.for_pi(&key, Some("MiniMax-M1"), &[], &HashMap::new());

        assert_eq!(env.get("AIVO_SETUP_PI_AGENT_DIR"), Some(&"1".to_string()));
        assert!(!env.contains_key("AIVO_USE_PI_ROUTER"));
    }

    #[test]
    fn test_for_pi_lists_full_catalog_pinned_first() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_active(false);

        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.openai.com/v1");
        let catalog = [
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "o3".to_string(),
            "gpt-4o".to_string(), // duplicate is dropped
        ];
        let env = injector.for_pi(&key, Some("o3"), &catalog, &HashMap::new());

        let parsed: Value = serde_json::from_str(env.get("AIVO_PI_MODELS_JSON").unwrap()).unwrap();
        let ids: Vec<&str> = parsed["providers"]["aivo"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        // Pinned model first; the rest sorted and de-duplicated against it.
        assert_eq!(ids, ["o3", "gpt-4o", "gpt-4o-mini"]);
    }

    #[test]
    fn test_for_pi_carries_resolved_limits_per_model() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_active(false);

        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.openai.com/v1");
        let catalog = ["aivo/starter".to_string(), "mystery-model-xyz".to_string()];
        let mut limits = HashMap::new();
        limits.insert(
            "aivo/starter".to_string(),
            crate::services::model_metadata::ResolvedLimits {
                context: Some(1_000_000),
                output: Some(384_000),
                caps: None,
            },
        );
        let env = injector.for_pi(&key, Some("aivo/starter"), &catalog, &limits);
        let parsed: Value = serde_json::from_str(env.get("AIVO_PI_MODELS_JSON").unwrap()).unwrap();
        let models = parsed["providers"]["aivo"]["models"].as_array().unwrap();
        // Known limits become pi's contextWindow/maxTokens (else pi assumes 128k).
        assert_eq!(models[0]["id"], "aivo/starter");
        assert_eq!(models[0]["contextWindow"], 1_000_000);
        assert_eq!(models[0]["maxTokens"], 384_000);
        // Unknown models omit the fields so pi's own defaults apply.
        assert_eq!(models[1]["id"], "mystery-model-xyz");
        assert!(models[1].get("contextWindow").is_none());
        assert!(models[1].get("maxTokens").is_none());
    }

    #[test]
    fn test_for_pi_without_model_defaults_to_first_catalog_entry() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_active(false);

        let injector = EnvironmentInjector::new();
        let key = test_api_key("https://api.openai.com/v1");
        // No pinned model: the catalog's first entry becomes pi's default.
        // A literal "default" placeholder would be sent verbatim upstream
        // and 400 ("unknown model 'default'").
        let catalog = ["MiniMax-M2.7".to_string(), "MiniMax-M2".to_string()];
        let env = injector.for_pi(&key, None, &catalog, &HashMap::new());
        let parsed: Value = serde_json::from_str(env.get("AIVO_PI_MODELS_JSON").unwrap()).unwrap();
        let ids: Vec<&str> = parsed["providers"]["aivo"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["MiniMax-M2.7", "MiniMax-M2"]);

        // Empty catalog AND no model: the placeholder soft-fail survives.
        let env = injector.for_pi(&key, None, &[], &HashMap::new());
        let parsed: Value = serde_json::from_str(env.get("AIVO_PI_MODELS_JSON").unwrap()).unwrap();
        assert_eq!(parsed["providers"]["aivo"]["models"][0]["id"], "default");
    }

    #[test]
    fn test_for_pi_transform_forces_router_without_debug() {
        let _guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_active(true);

        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://openrouter.ai/api/v1".to_string();
        let env = injector.for_pi(&key, Some("claude-sonnet-4-6"), &[], &HashMap::new());

        assert_eq!(env.get("AIVO_USE_PI_ROUTER"), Some(&"1".to_string()));
        assert_eq!(
            env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL"),
            Some(&"https://openrouter.ai/api/v1".to_string())
        );
        assert!(!env.contains_key("AIVO_SETUP_PI_AGENT_DIR"));

        crate::services::transform_mode::set_active(false);
    }

    #[test]
    fn for_gemini_oauth_sets_placeholder_vars_and_clears_direct_env() {
        use crate::services::gemini_oauth::{GEMINI_OAUTH_SENTINEL, GeminiOAuthCredential};
        let creds = GeminiOAuthCredential {
            access_token: "ya29.TEST".into(),
            refresh_token: "1//TEST".into(),
            id_token: None,
            scope: "https://www.googleapis.com/auth/cloud-platform".into(),
            token_type: "Bearer".into(),
            expiry_date: 1_700_000_000_000,
            email: Some("alice@example.com".into()),
            last_refresh: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "gid".into(),
            "alice".into(),
            GEMINI_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let injector = EnvironmentInjector::new();
        let env = injector.for_gemini(&key, Some("gemini-2.5-pro"));

        // Placeholder vars consumed by launch_runtime::prepare_gemini_oauth_shadow.
        assert_eq!(
            env.get("AIVO_GEMINI_OAUTH_CREDS"),
            Some(&key.key.as_str().to_string())
        );
        assert_eq!(env.get("AIVO_GEMINI_KEY_ID"), Some(&"gid".to_string()));

        // Bypass the TUI auth-type picker in the native CLI.
        assert_eq!(env.get("GOOGLE_GENAI_USE_GCA"), Some(&"true".to_string()));

        // Folder-trust store is pointed at a persistent aivo path so trust
        // choices survive the shadow HOME being recreated each launch.
        let trust_path = env
            .get("GEMINI_CLI_TRUSTED_FOLDERS_PATH")
            .expect("trust path env var");
        assert!(trust_path.ends_with("gemini-trusted-folders.json"));
        assert!(trust_path.replace('\\', "/").contains(".config/aivo"));

        // Model passes through (with google_native_model_name mapping).
        assert!(env.contains_key("GEMINI_MODEL"));

        // Direct-mode env vars must be empty so a caller export can't shadow
        // the OAuth creds inside the shadow HOME.
        assert_eq!(env.get("GEMINI_API_KEY"), Some(&String::new()));
        assert_eq!(env.get("GOOGLE_API_KEY"), Some(&String::new()));
        assert_eq!(env.get("GOOGLE_GEMINI_BASE_URL"), Some(&String::new()));

        // No router-mode indicators — OAuth is always native Google.
        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        assert!(!env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER"));
    }

    #[test]
    fn for_gemini_non_oauth_sets_force_api_key_auth_sentinel() {
        // Regression guard: without this sentinel, a user's stale
        // `oauth-personal` entry in ~/.gemini/settings.json wins over
        // GEMINI_API_KEY and every request lands at Google.
        let injector = EnvironmentInjector::new();

        for (label, base_url) in [
            ("routed non-google", "http://localhost:8080"),
            ("direct google", "https://generativelanguage.googleapis.com"),
            ("copilot", "copilot"),
        ] {
            let mut key = test_key();
            key.base_url = base_url.to_string();
            let env = injector.for_gemini(&key, None);
            assert_eq!(
                env.get("AIVO_GEMINI_FORCE_API_KEY_AUTH"),
                Some(&"1".to_string()),
                "{label}: sentinel must be set for non-OAuth keys"
            );
        }
    }

    #[test]
    fn for_gemini_oauth_does_not_set_force_api_key_auth_sentinel() {
        use crate::services::gemini_oauth::{GEMINI_OAUTH_SENTINEL, GeminiOAuthCredential};
        let creds = GeminiOAuthCredential {
            access_token: "at".into(),
            refresh_token: "rt".into(),
            id_token: None,
            scope: "s".into(),
            token_type: "Bearer".into(),
            expiry_date: 0,
            email: None,
            last_refresh: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "gid".into(),
            "alice".into(),
            GEMINI_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let injector = EnvironmentInjector::new();
        let env = injector.for_gemini(&key, None);
        assert!(
            !env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH"),
            "OAuth keys already use Google-OAuth auth; forcing api-key would break them"
        );
    }

    #[test]
    fn for_gemini_oauth_without_model_omits_gemini_model() {
        use crate::services::gemini_oauth::{GEMINI_OAUTH_SENTINEL, GeminiOAuthCredential};
        let creds = GeminiOAuthCredential {
            access_token: "ya29.TEST".into(),
            refresh_token: "1//TEST".into(),
            id_token: None,
            scope: "s".into(),
            token_type: "Bearer".into(),
            expiry_date: 0,
            email: None,
            last_refresh: chrono::Utc::now(),
        };
        let key = ApiKey::new_with_protocol(
            "gid".into(),
            "anon".into(),
            GEMINI_OAUTH_SENTINEL.into(),
            None,
            creds.to_json().unwrap(),
        );
        let injector = EnvironmentInjector::new();
        let env = injector.for_gemini(&key, None);
        assert!(!env.contains_key("GEMINI_MODEL"));
        assert_eq!(env.get("GOOGLE_GENAI_USE_GCA"), Some(&"true".to_string()));
    }
}
