/**
 * EnvironmentInjector service for preparing tool-specific environment variables.
 * Maps API keys to the correct environment variables per AI tool.
 */
use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::constants::{AIVO_STARTER_REAL_URL, AIVO_STARTER_SENTINEL, PLACEHOLDER_LOOPBACK_URL};
use crate::services::codex_model_map::map_model_for_codex_cli;
use crate::services::model_names::{anthropic_native_model_name, google_native_model_name};
use crate::services::ollama::ollama_openai_base_url;
use crate::services::provider_profile::{
    ProviderKind, ProviderProfile, is_direct_openai_base, provider_profile_for_key,
};
use crate::services::provider_protocol::{ProviderProtocol, is_anthropic_endpoint};
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, OpenAICompatibilityMode,
};

/// Describes which env vars a tool uses for base URL, auth, and router configuration.
struct ToolEnvConfig {
    base_url_env: &'static str,
    auth_env: &'static str,
    router_flag: &'static str,
    router_prefix: &'static str,
    copilot_flag: &'static str,
}

/// How the tool should connect to the upstream provider.
enum ConnectionMode {
    Ollama,
    Copilot,
    OpenRouter,
    Direct { base_url: String },
    Routed { protocol: ProviderProtocol },
}

/// EnvironmentInjector prepares tool-specific environment variables for AI tools
#[derive(Debug, Clone, Default)]
pub struct EnvironmentInjector;

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
    /// Returns true when the URL points to a native Anthropic endpoint that speaks
    /// the Anthropic Messages API directly (no format conversion needed).
    ///
    /// This includes Anthropic's official API plus provider-hosted Anthropic-compatible
    /// bases such as MiniMax's `/anthropic` endpoint.
    fn use_direct_anthropic_for_claude(key: &ApiKey) -> bool {
        match key.claude_protocol {
            Some(ClaudeProviderProtocol::Anthropic) => true,
            Some(ClaudeProviderProtocol::Openai | ClaudeProviderProtocol::Google) => false,
            None => is_anthropic_endpoint(&key.base_url),
        }
    }

    fn use_direct_openai_for_codex(key: &ApiKey) -> bool {
        match key.codex_mode {
            Some(OpenAICompatibilityMode::Direct) => true,
            Some(OpenAICompatibilityMode::Router) => false,
            None => is_direct_openai_base(&key.base_url),
        }
    }

    fn use_google_native_for_gemini(key: &ApiKey) -> bool {
        match key.gemini_protocol {
            Some(GeminiProviderProtocol::Google) => true,
            Some(GeminiProviderProtocol::Openai | GeminiProviderProtocol::Anthropic) => false,
            None => provider_profile_for_key(key).default_protocol == ProviderProtocol::Google,
        }
    }

    fn use_router_for_opencode(key: &ApiKey) -> bool {
        matches!(key.opencode_mode, Some(OpenAICompatibilityMode::Router))
    }

    fn routed_protocol_for_claude(key: &ApiKey) -> ProviderProtocol {
        match key.claude_protocol {
            Some(ClaudeProviderProtocol::Anthropic) => ProviderProtocol::Anthropic,
            Some(ClaudeProviderProtocol::Openai) => ProviderProtocol::Openai,
            Some(ClaudeProviderProtocol::Google) => ProviderProtocol::Google,
            None => provider_profile_for_key(key).default_protocol,
        }
    }

    fn routed_protocol_for_gemini(key: &ApiKey) -> ProviderProtocol {
        match key.gemini_protocol {
            Some(GeminiProviderProtocol::Google) => ProviderProtocol::Google,
            Some(GeminiProviderProtocol::Openai) => ProviderProtocol::Openai,
            Some(GeminiProviderProtocol::Anthropic) => ProviderProtocol::Anthropic,
            None => provider_profile_for_key(key).default_protocol,
        }
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

        let mut env = HashMap::new();
        match mode {
            ConnectionMode::Ollama => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), "ollama".to_string());
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
                env.insert(cfg.auth_env.to_string(), "copilot".to_string());
                env.insert(cfg.copilot_flag.to_string(), "1".to_string());
                env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
            }
            ConnectionMode::OpenRouter => {
                env.insert(
                    cfg.base_url_env.to_string(),
                    PLACEHOLDER_LOOPBACK_URL.to_string(),
                );
                env.insert(cfg.auth_env.to_string(), auth_value.clone());
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
                env.insert(cfg.auth_env.to_string(), auth_value.clone());
                env.insert(cfg.router_flag.to_string(), "1".to_string());
                env.insert(format!("{}_API_KEY", cfg.router_prefix), auth_value);
                env.insert(format!("{}_BASE_URL", cfg.router_prefix), resolved_base_url);
                env.insert(
                    format!("{}_UPSTREAM_PROTOCOL", cfg.router_prefix),
                    protocol.as_str().to_string(),
                );
                profile.quirks.inject(&mut env, cfg.router_prefix);
            }
        }
        if profile.serve_flags.is_starter {
            env.insert("AIVO_IS_STARTER".to_string(), "1".to_string());
        }
        env
    }

    /// Prepares environment variables for Claude CLI
    pub fn for_claude(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        if key.is_claude_oauth() {
            let _ = model;
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
            env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
            env.insert("ANTHROPIC_AUTH_TOKEN".to_string(), String::new());
            env.insert("ANTHROPIC_BASE_URL".to_string(), String::new());
            env.insert(
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
                "1".to_string(),
            );
            return env;
        }

        let profile = provider_profile_for_key(key);
        let mode = if profile.kind == ProviderKind::Ollama {
            ConnectionMode::Ollama
        } else if profile.serve_flags.is_copilot {
            ConnectionMode::Copilot
        } else if profile.serve_flags.is_openrouter {
            ConnectionMode::OpenRouter
        } else if Self::use_direct_anthropic_for_claude(key) {
            let base_url = key.base_url.trim_end_matches('/');
            let base_url = base_url.strip_suffix("/v1").unwrap_or(base_url);
            ConnectionMode::Direct {
                base_url: base_url.to_string(),
            }
        } else {
            ConnectionMode::Routed {
                protocol: Self::routed_protocol_for_claude(key),
            }
        };

        let cfg = ToolEnvConfig {
            base_url_env: "ANTHROPIC_BASE_URL",
            auth_env: "ANTHROPIC_AUTH_TOKEN",
            router_flag: "AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER",
            router_prefix: "AIVO_ANTHROPIC_TO_OPENAI_ROUTER",
            copilot_flag: "AIVO_USE_COPILOT_ROUTER",
        };

        let mut env = Self::inject_connection(&cfg, key, &mode, &profile);
        env.insert("ANTHROPIC_API_KEY".to_string(), String::new());
        env.insert(
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
            "1".to_string(),
        );
        env.insert("BASH_DEFAULT_TIMEOUT_MS".to_string(), "2400000".to_string());
        env.insert("BASH_MAX_TIMEOUT_MS".to_string(), "2500000".to_string());
        env.insert(
            "CLAUDE_CODE_ATTRIBUTION_HEADER".to_string(),
            "0".to_string(),
        );
        env.insert("API_TIMEOUT_MS".to_string(), "30000000".to_string());
        if let Some(model) = model {
            let anthropic_model = if matches!(mode, ConnectionMode::Direct { .. }) {
                anthropic_native_model_name(model)
            } else {
                model.to_string()
            };
            env.insert("ANTHROPIC_MODEL".to_string(), anthropic_model.clone());
            env.insert(
                "ANTHROPIC_SMALL_FAST_MODEL".to_string(),
                anthropic_model.clone(),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(),
                anthropic_model.clone(),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_SONNET_MODEL".to_string(),
                anthropic_model.clone(),
            );
            env.insert(
                "ANTHROPIC_DEFAULT_OPUS_MODEL".to_string(),
                anthropic_model.clone(),
            );
            env.insert(
                "ANTHROPIC_REASONING_MODEL".to_string(),
                anthropic_model.clone(),
            );
            env.insert(
                "CLAUDE_CODE_SUBAGENT_MODEL".to_string(),
                anthropic_model.clone(),
            );
        }

        env
    }

    /// Prepares environment variables for Codex CLI
    pub fn for_codex(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
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
        let mode = if profile.kind == ProviderKind::Ollama {
            ConnectionMode::Ollama
        } else if profile.serve_flags.is_copilot {
            ConnectionMode::Copilot
        } else if !Self::use_direct_openai_for_codex(key) {
            ConnectionMode::Routed {
                protocol: profile.default_protocol,
            }
        } else {
            ConnectionMode::Direct {
                base_url: key.base_url.clone(),
            }
        };

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
        let mode = if profile.kind == ProviderKind::Ollama {
            ConnectionMode::Ollama
        } else if profile.serve_flags.is_copilot {
            ConnectionMode::Copilot
        } else if Self::use_google_native_for_gemini(key) {
            ConnectionMode::Direct {
                base_url: strip_google_version_suffix(&key.base_url).to_string(),
            }
        } else {
            ConnectionMode::Routed {
                protocol: Self::routed_protocol_for_gemini(key),
            }
        };

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

        if let Some(model) = model {
            let gemini_model = if matches!(mode, ConnectionMode::Direct { .. }) {
                google_native_model_name(model).to_string()
            } else {
                model.to_string()
            };
            env.insert("GEMINI_MODEL".to_string(), gemini_model);
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
    ) -> HashMap<String, String> {
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
        let (base_url, api_key) = if profile.kind == ProviderKind::Ollama {
            (ollama_openai_base_url(), "ollama".to_string())
        } else if profile.serve_flags.is_copilot {
            env.insert(
                "AIVO_USE_OPENCODE_COPILOT_ROUTER".to_string(),
                "1".to_string(),
            );
            env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
            (PLACEHOLDER_LOOPBACK_URL.to_string(), "copilot".to_string())
        } else if Self::use_router_for_opencode(key) || profile.serve_flags.is_starter {
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
                // OpenCode's SDK strips the provider prefix (aivo/starter → starter),
                // but the API expects the full model name. Override via actual_model.
                env.insert(
                    "AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL".to_string(),
                    "aivo/starter".to_string(),
                );
            }
            (PLACEHOLDER_LOOPBACK_URL.to_string(), auth)
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

    /// Prepares environment variables for Pi CLI.
    ///
    /// Pi natively supports OpenAI, Anthropic, and Google protocols, so we point it
    /// directly at the upstream via a custom `aivo` provider in `models.json` with
    /// the appropriate `api` type. No conversion router is needed.
    ///
    /// - **Non-Copilot**: Write `models.json` with direct upstream URL + correct API type.
    /// - **Copilot**: Needs CopilotTokenManager for auth, so start a ResponsesToChatRouter
    ///   and point pi at it with `openai-completions`.
    pub fn for_pi(&self, key: &ApiKey, model: Option<&str>) -> HashMap<String, String> {
        let mut env = HashMap::new();
        let profile = provider_profile_for_key(key);

        if profile.kind == ProviderKind::Ollama {
            // Ollama: direct connection via Pi's custom provider
            let models_json = build_pi_models_json(
                &ollama_openai_base_url(),
                "ollama",
                "openai-completions",
                model,
            );
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_SETUP_PI_AGENT_DIR".to_string(), "1".to_string());
        } else if profile.serve_flags.is_copilot {
            // Copilot needs CopilotTokenManager — route through ResponsesToChatRouter
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                "copilot",
                "openai-completions",
                model,
            );
            env.insert("AIVO_PI_MODELS_JSON".to_string(), models_json);
            env.insert("AIVO_USE_PI_COPILOT_ROUTER".to_string(), "1".to_string());
            env.insert("AIVO_COPILOT_GITHUB_TOKEN".to_string(), key.key.to_string());
        } else if profile.serve_flags.is_starter {
            // Starter provider: route through a local router so device fingerprint
            // headers are attached (Pi's native HTTP client can't add them).
            let models_json = build_pi_models_json(
                PLACEHOLDER_LOOPBACK_URL,
                AIVO_STARTER_SENTINEL,
                "openai-completions",
                model,
            );
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
            let models_json = build_pi_models_json(&resolved_url, auth, pi_api, model);
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
        debug: bool,
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

        // Debug output if requested
        if debug {
            eprintln!("[aivo] Injecting environment variables:");
            let mut keys: Vec<_> = tool_env.keys().collect();
            keys.sort();
            for key in keys {
                let value = &tool_env[key];
                let display = redact_env_value(key, value);
                eprintln!("  {}={}", key, display);
            }

            if let Some(manual) = manual_env
                && !manual.is_empty()
            {
                eprintln!("[aivo] Manual environment overrides:");
                let mut keys: Vec<_> = manual.keys().collect();
                keys.sort();
                for key in keys {
                    let value = &manual[key];
                    let display = redact_env_value(key, value);
                    eprintln!("  {}={}", key, display);
                }
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
) -> String {
    let model_id = model.unwrap_or("default");
    let models_json = json!({
        "providers": {
            "aivo": {
                "baseUrl": base_url,
                "apiKey": api_key,
                "api": api_type,
                "models": [
                    { "id": model_id, "name": model_id }
                ]
            }
        }
    });
    models_json.to_string()
}

pub(crate) fn redact_env_value(key: &str, value: &str) -> String {
    if key == "OPENCODE_CONFIG_CONTENT" || key == "AIVO_PI_MODELS_JSON" {
        return "<redacted>".to_string();
    }

    if key.contains("TOKEN") || key.contains("KEY") {
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

    fn test_key() -> ApiKey {
        ApiKey::new_with_protocol(
            "a1b2".to_string(),
            "test-key".to_string(),
            "http://localhost:8080".to_string(),
            None,
            "sk-test-key-12345".to_string(),
        )
    }

    #[test]
    fn test_for_claude_anthropic_native_direct() {
        // Official Anthropic endpoints bypass all routers.
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
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn for_claude_injects_oauth_token_and_clears_api_key_vars() {
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
        // Must not let ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN shadow the OAuth
        // token — Claude Code's auth precedence prefers those env vars.
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        assert_eq!(env.get("ANTHROPIC_AUTH_TOKEN"), Some(&String::new()));
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
        // ANTHROPIC_BASE_URL is cleared so a caller-exported value can't
        // misroute OAuth traffic (Claude Code falls back to its default
        // subscription backend when the env var is empty).
        assert_eq!(env.get("ANTHROPIC_BASE_URL"), Some(&String::new()));
        // No routed-mode indicators.
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn for_claude_oauth_ignores_model_override() {
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
        assert!(!env.contains_key("ANTHROPIC_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_DEFAULT_OPUS_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_DEFAULT_SONNET_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_DEFAULT_HAIKU_MODEL"));
        assert!(!env.contains_key("ANTHROPIC_REASONING_MODEL"));
        assert!(!env.contains_key("CLAUDE_CODE_SUBAGENT_MODEL"));
    }

    #[test]
    fn for_claude_oauth_with_corrupt_json_still_clears_api_key_vars() {
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
        // account. The cleared API-key vars block that fall-through.
        assert_eq!(env.get("ANTHROPIC_API_KEY"), Some(&String::new()));
        assert_eq!(env.get("ANTHROPIC_AUTH_TOKEN"), Some(&String::new()));
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
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_minimax_anthropic_v1_endpoint_direct() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic/v1".to_string();
        let env = injector.for_claude(&key, None);

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.minimax.io/anthropic".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_protocol_override_anthropic_direct() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example.com/v1".to_string();
        key.claude_protocol = Some(ClaudeProviderProtocol::Anthropic);
        let env = injector.for_claude(&key, Some("claude-sonnet-4-6"));

        assert_eq!(
            env.get("ANTHROPIC_BASE_URL"),
            Some(&"https://api.example.com".to_string())
        );
        assert!(!env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"));
    }

    #[test]
    fn test_for_claude_protocol_override_anthropic_to_openai_router() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.minimax.io/anthropic".to_string();
        key.claude_protocol = Some(ClaudeProviderProtocol::Openai);
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
        key.claude_protocol = Some(ClaudeProviderProtocol::Google);

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
    fn test_for_claude_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_claude(&key, Some("claude-3-opus"));

        assert_eq!(
            env.get("ANTHROPIC_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
        assert_eq!(
            env.get("ANTHROPIC_SMALL_FAST_MODEL"),
            Some(&"claude-3-opus".to_string())
        );
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
        assert_eq!(
            env.get("OPENAI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
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
        assert_eq!(
            env.get("GEMINI_API_KEY"),
            Some(&"sk-test-key-12345".to_string())
        );
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
    fn test_for_gemini_protocol_override_google_direct() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://api.example.com".to_string();
        key.gemini_protocol = Some(GeminiProviderProtocol::Google);
        let env = injector.for_gemini(&key, None);

        assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
        assert_eq!(
            env.get("GOOGLE_GEMINI_BASE_URL"),
            Some(&"https://api.example.com".to_string())
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
        key.gemini_protocol = Some(GeminiProviderProtocol::Anthropic);

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
        assert_eq!(env.get("GEMINI_API_KEY"), Some(&"copilot".to_string()));
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
        let env = injector.for_opencode(&key, None, None);

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
    fn test_for_opencode_with_model() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let env = injector.for_opencode(&key, Some("gpt-5"), None);

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
        let env = injector.for_opencode(&key, Some("aivo/gpt-5"), None);

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
        let env = injector.for_opencode(&key, None, Some(&discovered));

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
        let env = injector.for_opencode(&key, Some("gpt-5"), Some(&discovered));

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
    fn test_for_opencode_copilot_uses_placeholder_url() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "copilot".to_string();
        let env = injector.for_opencode(&key, None, None);

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
        assert_eq!(config["provider"]["aivo"]["options"]["apiKey"], "copilot");
    }

    #[test]
    fn test_for_opencode_router_uses_placeholder_url() {
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.opencode_mode = Some(OpenAICompatibilityMode::Router);
        let env = injector.for_opencode(&key, Some("gpt-4o"), None);

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
            config["provider"]["aivo"]["options"]["apiKey"],
            "sk-test-key-12345"
        );
    }

    #[test]
    fn test_merge() {
        let injector = EnvironmentInjector::new();
        let key = test_key();
        let tool_env = injector.for_claude(&key, None);
        let merged = injector.merge(&tool_env, None, false);

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
        assert_eq!(
            env.get("ANTHROPIC_AUTH_TOKEN"),
            Some(&"copilot".to_string())
        );
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
        assert_eq!(env.get("OPENAI_API_KEY"), Some(&"copilot".to_string()));
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
        assert_eq!(env.get("ANTHROPIC_AUTH_TOKEN"), Some(&"ollama".to_string()));
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
        assert_eq!(env.get("OPENAI_API_KEY"), Some(&"ollama".to_string()));
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
        assert_eq!(env.get("GEMINI_API_KEY"), Some(&"ollama".to_string()));
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
        let env = injector.for_opencode(&key, Some("llama3.2"), None);

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
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com/v1beta".to_string();
        let env = injector.for_pi(&key, Some("gemini-2.5-flash"));

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
        let injector = EnvironmentInjector::new();
        let mut key = test_key();
        key.base_url = "https://generativelanguage.googleapis.com".to_string();
        let env = injector.for_pi(&key, Some("gemini-2.5-flash"));

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
        let injector = EnvironmentInjector::new();
        let key = ollama_key();
        let env = injector.for_pi(&key, Some("llama3.2"));

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
