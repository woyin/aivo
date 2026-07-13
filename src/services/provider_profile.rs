use std::collections::HashMap;

use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::ApiKey;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Copilot,
    CursorAcp,
    Ollama,
    OpenRouter,
    CloudflareAi,
    AnthropicCompatible,
    GoogleNative,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelListingStrategy {
    Copilot,
    CursorAcp,
    Ollama,
    Google,
    Anthropic,
    CloudflareSearch,
    OpenAiCompatible,
    AivoStarter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeFlags {
    pub is_copilot: bool,
    pub is_openrouter: bool,
    pub is_starter: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderQuirks {
    pub model_prefix: Option<&'static str>,
    pub requires_reasoning_content: bool,
    pub max_tokens_cap: Option<u64>,
    /// Whether to strip Anthropic-specific `cache_control` blocks from the
    /// converted request before forwarding upstream. Some OpenAI-compat shims
    /// (Bedrock proxies, custom gateways) reject unknown `cache_control` keys
    /// inside system / message content with a 400.
    pub strips_cache_control: bool,
}

impl ProviderQuirks {
    pub fn for_base_url(base_url: &str) -> Self {
        let model_prefix = if cloudflare_ai_base(base_url).is_some() {
            Some("@cf/")
        } else {
            None
        };
        let is_deepseek = base_url.contains("deepseek.com");
        let requires_reasoning_content =
            is_deepseek || base_url.contains("moonshot.cn") || base_url.contains("moonshot.ai");
        // Bedrock-style hosts and the AWS gateway shim reject Anthropic
        // cache_control fields when they appear on system/message content
        // converted into the OpenAI Chat shape. Strip them defensively for the
        // hosts we know reject; other providers accept the pass-through.
        let strips_cache_control = base_url.contains("bedrock-runtime.")
            || base_url.contains(".bedrock.")
            || base_url.contains("/bedrock/")
            || base_url.contains("aws.com");
        Self {
            model_prefix,
            requires_reasoning_content,
            max_tokens_cap: None,
            strips_cache_control,
        }
    }

    pub fn has_quirks(&self) -> bool {
        self.model_prefix.is_some()
            || self.requires_reasoning_content
            || self.max_tokens_cap.is_some()
            || self.strips_cache_control
    }

    pub fn inject(&self, env: &mut HashMap<String, String>, prefix: &str) {
        if let Some(pfx) = self.model_prefix {
            env.insert(format!("{prefix}_MODEL_PREFIX"), pfx.to_string());
        }
        if self.requires_reasoning_content {
            env.insert(format!("{prefix}_REQUIRE_REASONING"), "1".to_string());
        }
        if let Some(cap) = self.max_tokens_cap {
            env.insert(format!("{prefix}_MAX_TOKENS_CAP"), cap.to_string());
        }
        if self.strips_cache_control {
            env.insert(format!("{prefix}_STRIP_CACHE_CONTROL"), "1".to_string());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderProfile {
    pub kind: ProviderKind,
    pub default_protocol: ProviderProtocol,
    pub quirks: ProviderQuirks,
    pub model_listing_strategy: ModelListingStrategy,
    pub serve_flags: ServeFlags,
}

impl ProviderProfile {
    /// First protocol the router will try when no per-key user override is set.
    ///
    /// Always the CLI's native protocol. Any provider may speak multiple
    /// protocols (OpenAI-compat hosts that also serve `/v1/messages`, Anthropic
    /// hosts that also expose `/v1/chat/completions`, multi-protocol gateways),
    /// so we don't let the provider's perceived default protocol override the
    /// tool's choice. The router's fallback loop (`protocol_candidates` /
    /// `fallback_protocols`) handles protocol mismatches: a 404/401/403 on the
    /// first attempt triggers the next candidate, and the winning protocol is
    /// persisted to the key pin so subsequent launches skip the probe.
    ///
    /// Trade-off: cross-tool usage against a single-protocol host (e.g. `aivo
    /// gemini` against `api.anthropic.com`) pays extra round-trips on the very
    /// first launch — fine, because the pin persists and subsequent launches
    /// go straight to the learned protocol.
    pub fn upstream_protocol_for_cli(&self, cli_native: ProviderProtocol) -> ProviderProtocol {
        cli_native
    }
}

pub fn is_copilot_base(base_url: &str) -> bool {
    base_url == "copilot"
}

pub fn is_ollama_base(base_url: &str) -> bool {
    base_url == "ollama"
}

pub fn is_aivo_starter_base(base_url: &str) -> bool {
    base_url == crate::constants::AIVO_STARTER_SENTINEL
        || base_url == crate::constants::AIVO_STARTER_REAL_URL
}

/// Account-bound login sessions (browser/device-flow). Filtered from
/// exports by default — they grant subscription access, not pay-as-you-go
/// API quota, so they shouldn't travel silently with an API-key backup.
pub fn is_oauth_or_copilot_base(base_url: &str) -> bool {
    is_copilot_base(base_url)
        || base_url == crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL
        || base_url == crate::services::codex_oauth::CODEX_OAUTH_SENTINEL
        || base_url == crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL
}

/// Resolves the aivo-starter sentinel to the real API URL.
/// Returns the base_url unchanged for all other providers.
pub fn resolve_starter_base_url(base_url: &str) -> String {
    if base_url == crate::constants::AIVO_STARTER_SENTINEL {
        crate::constants::AIVO_STARTER_REAL_URL.to_string()
    } else if base_url == crate::services::grok_oauth::GROK_OAUTH_SENTINEL {
        // Resolve the grok sentinel to the CLI proxy so base_url-derived URLs
        // stay valid; the token manager still owns the final host per request.
        crate::services::grok_oauth::INFERENCE_BASE_URL.to_string()
    } else {
        base_url.to_string()
    }
}

pub fn is_openrouter_base(base_url: &str) -> bool {
    base_url.contains("openrouter")
}

pub fn is_direct_openai_base(base_url: &str) -> bool {
    base_url
        .trim_end_matches('/')
        .to_ascii_lowercase()
        .contains("api.openai.com")
}

/// Normalize a user-supplied AWS region input from the `keys add` Bedrock prompt.
///
/// Accepts any of:
///   - a bare region: `us-east-1`, `eu-central-1`, `us-gov-east-1`
///   - a Bedrock Mantle URL: `https://bedrock-mantle.us-east-1.api.aws/v1`
///   - a Bedrock Invoke URL: `https://bedrock-runtime.us-east-1.amazonaws.com`
///   - any of the above with or without scheme, trailing slash, or path
///
/// Returns the canonical region (e.g. `us-east-1`) or `None` if the input is
/// neither a plausible bare region nor a recognizable Bedrock URL.
pub fn parse_aws_region(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !trimmed.contains('.') && !trimmed.contains('/') {
        return is_plausible_region(trimmed).then(|| trimmed.to_string());
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("https://{trimmed}")
    };
    let parsed = reqwest::Url::parse(&with_scheme).ok()?;
    let host = parsed.host_str()?;
    let region = if let Some(rest) = host.strip_prefix("bedrock-mantle.") {
        rest.strip_suffix(".api.aws")?
    } else {
        let rest = host.strip_prefix("bedrock-runtime.")?;
        rest.strip_suffix(".amazonaws.com")?
    };
    is_plausible_region(region).then(|| region.to_string())
}

fn is_plausible_region(s: &str) -> bool {
    if !s.contains('-') {
        return false;
    }
    let bytes = s.as_bytes();
    if !bytes.first().is_some_and(|b| b.is_ascii_lowercase()) {
        return false;
    }
    if !bytes.last().is_some_and(|b| b.is_ascii_digit()) {
        return false;
    }
    s.bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

pub fn cloudflare_ai_base(base_url: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(base_url).ok()?;
    let host = parsed.host_str()?;
    if !host.contains("cloudflare.com") {
        return None;
    }

    let mut base = base_url.trim_end_matches('/').to_string();
    if base.ends_with("/v1/chat/completions") {
        base.truncate(base.len() - "/v1/chat/completions".len());
    } else if base.ends_with("/chat/completions") {
        base.truncate(base.len() - "/chat/completions".len());
    } else if base.ends_with("/v1") {
        base.truncate(base.len() - "/v1".len());
    }

    if !base.ends_with("/ai") {
        let idx = base.find("/ai/")?;
        base.truncate(idx + "/ai".len());
    }

    Some(base)
}

pub fn provider_profile_for_base_url(base_url: &str) -> ProviderProfile {
    let quirks = ProviderQuirks::for_base_url(base_url);
    if is_copilot_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::Copilot,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::Copilot,
            serve_flags: ServeFlags {
                is_copilot: true,
                is_openrouter: false,
                is_starter: false,
            },
        };
    }

    if crate::services::cursor_acp::is_cursor_acp_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::CursorAcp,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::CursorAcp,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        };
    }

    if is_ollama_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::Ollama,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::Ollama,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        };
    }

    if is_aivo_starter_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::OpenAiCompatible,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::AivoStarter,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: true,
            },
        };
    }

    if is_openrouter_base(base_url) {
        return ProviderProfile {
            kind: ProviderKind::OpenRouter,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::OpenAiCompatible,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: true,
                is_starter: false,
            },
        };
    }

    if cloudflare_ai_base(base_url).is_some() {
        return ProviderProfile {
            kind: ProviderKind::CloudflareAi,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::CloudflareSearch,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        };
    }

    match detect_provider_protocol(base_url) {
        ProviderProtocol::Anthropic => ProviderProfile {
            kind: ProviderKind::AnthropicCompatible,
            default_protocol: ProviderProtocol::Anthropic,
            quirks,
            model_listing_strategy: ModelListingStrategy::Anthropic,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        },
        ProviderProtocol::Google => ProviderProfile {
            kind: ProviderKind::GoogleNative,
            default_protocol: ProviderProtocol::Google,
            quirks,
            model_listing_strategy: ModelListingStrategy::Google,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        },
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => ProviderProfile {
            kind: ProviderKind::OpenAiCompatible,
            default_protocol: ProviderProtocol::Openai,
            quirks,
            model_listing_strategy: ModelListingStrategy::OpenAiCompatible,
            serve_flags: ServeFlags {
                is_copilot: false,
                is_openrouter: false,
                is_starter: false,
            },
        },
    }
}

pub fn provider_profile_for_key(key: &ApiKey) -> ProviderProfile {
    provider_profile_for_base_url(&key.base_url)
}

#[cfg(test)]
mod tests {
    use super::{
        ModelListingStrategy, ProviderKind, cloudflare_ai_base, provider_profile_for_base_url,
    };
    use crate::services::provider_protocol::ProviderProtocol;

    #[test]
    fn classifies_copilot() {
        let profile = provider_profile_for_base_url("copilot");
        assert_eq!(profile.kind, ProviderKind::Copilot);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::Copilot
        );
        assert!(profile.serve_flags.is_copilot);
        assert!(!profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_cursor_acp() {
        let profile = provider_profile_for_base_url("cursor");
        assert_eq!(profile.kind, ProviderKind::CursorAcp);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::CursorAcp
        );
        assert!(!profile.serve_flags.is_copilot);
        assert!(!profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_ollama() {
        let profile = provider_profile_for_base_url("ollama");
        assert_eq!(profile.kind, ProviderKind::Ollama);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(profile.model_listing_strategy, ModelListingStrategy::Ollama);
        assert!(!profile.serve_flags.is_copilot);
        assert!(!profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_aivo_starter() {
        let profile = provider_profile_for_base_url("aivo-starter");
        assert_eq!(profile.kind, ProviderKind::OpenAiCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::AivoStarter
        );
    }

    #[test]
    fn upstream_protocol_forwards_cli_native_for_starter() {
        let profile = provider_profile_for_base_url("aivo-starter");
        assert!(profile.serve_flags.is_starter);
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::Anthropic),
            ProviderProtocol::Anthropic,
        );
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::Google),
            ProviderProtocol::Google,
        );
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::ResponsesApi),
            ProviderProtocol::ResponsesApi,
        );
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::Openai),
            ProviderProtocol::Openai,
        );
    }

    #[test]
    fn upstream_protocol_prefers_cli_native_for_generic_openai_host() {
        let profile = provider_profile_for_base_url("https://api.example.com/v1");
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::Anthropic),
            ProviderProtocol::Anthropic,
        );
        assert_eq!(
            profile.upstream_protocol_for_cli(ProviderProtocol::Google),
            ProviderProtocol::Google,
        );
    }

    #[test]
    fn upstream_protocol_always_forwards_cli_native_even_on_known_hosts() {
        let anthropic = provider_profile_for_base_url("https://api.anthropic.com");
        assert_eq!(
            anthropic.upstream_protocol_for_cli(ProviderProtocol::Google),
            ProviderProtocol::Google,
        );
        assert_eq!(
            anthropic.upstream_protocol_for_cli(ProviderProtocol::Anthropic),
            ProviderProtocol::Anthropic,
        );

        let google =
            provider_profile_for_base_url("https://generativelanguage.googleapis.com/v1beta");
        assert_eq!(
            google.upstream_protocol_for_cli(ProviderProtocol::Anthropic),
            ProviderProtocol::Anthropic,
        );
        assert_eq!(
            google.upstream_protocol_for_cli(ProviderProtocol::ResponsesApi),
            ProviderProtocol::ResponsesApi,
        );
    }

    #[test]
    fn classifies_openrouter() {
        let profile = provider_profile_for_base_url("https://openrouter.ai/api/v1");
        assert_eq!(profile.kind, ProviderKind::OpenRouter);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::OpenAiCompatible
        );
        assert!(profile.serve_flags.is_openrouter);
    }

    #[test]
    fn classifies_minimax_like_deepseek() {
        // Bare host: falls through to OpenAI defaults (kind, protocol, listing)
        // — same as DeepSeek. Anthropic-shape requests get routed via the
        // `/anthropic` path-prefix quirk.
        for url in ["https://api.minimax.io", "https://api.minimax.com"] {
            let profile = provider_profile_for_base_url(url);
            assert_eq!(profile.kind, ProviderKind::OpenAiCompatible, "{url}");
            assert_eq!(profile.default_protocol, ProviderProtocol::Openai, "{url}");
            assert_eq!(
                profile.model_listing_strategy,
                ModelListingStrategy::OpenAiCompatible,
                "{url}"
            );
        }
        // `/anthropic` suffix: protocol detection flips to Anthropic.
        let profile = provider_profile_for_base_url("https://api.minimax.io/anthropic");
        assert_eq!(profile.kind, ProviderKind::AnthropicCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Anthropic);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::Anthropic
        );
    }

    #[test]
    fn classifies_anthropic_compatible_endpoints() {
        let profile = provider_profile_for_base_url("https://api.anthropic.com");
        assert_eq!(profile.kind, ProviderKind::AnthropicCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Anthropic);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::Anthropic
        );
    }

    #[test]
    fn classifies_google_native_endpoints() {
        let profile =
            provider_profile_for_base_url("https://generativelanguage.googleapis.com/v1beta");
        assert_eq!(profile.kind, ProviderKind::GoogleNative);
        assert_eq!(profile.default_protocol, ProviderProtocol::Google);
        assert_eq!(profile.model_listing_strategy, ModelListingStrategy::Google);
    }

    #[test]
    fn classifies_cloudflare_and_applies_prefix_quirk() {
        let profile = provider_profile_for_base_url(
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1",
        );
        assert_eq!(profile.kind, ProviderKind::CloudflareAi);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::CloudflareSearch
        );
        assert_eq!(profile.quirks.model_prefix, Some("@cf/"));
    }

    #[test]
    fn classifies_generic_openai_compatible_endpoints() {
        let profile = provider_profile_for_base_url("https://api.example.com/v1");
        assert_eq!(profile.kind, ProviderKind::OpenAiCompatible);
        assert_eq!(profile.default_protocol, ProviderProtocol::Openai);
        assert_eq!(
            profile.model_listing_strategy,
            ModelListingStrategy::OpenAiCompatible
        );
    }

    #[test]
    fn resolves_provider_quirks() {
        let moonshot = provider_profile_for_base_url("https://api.moonshot.cn/v1");
        assert!(moonshot.quirks.requires_reasoning_content);
        assert_eq!(moonshot.quirks.max_tokens_cap, None);

        let deepseek = provider_profile_for_base_url("https://api.deepseek.com/v1");
        assert!(deepseek.quirks.requires_reasoning_content);
        assert_eq!(deepseek.quirks.max_tokens_cap, None);

        let starter_url = provider_profile_for_base_url("https://api.getaivo.dev");
        assert_eq!(starter_url.quirks.max_tokens_cap, None);
        // Static quirks for aivo/starter intentionally leave
        // `requires_reasoning_content` unset — the actual value is discovered
        // per-key from the upstream's error body and persisted to ApiKey.
        assert!(!starter_url.quirks.requires_reasoning_content);

        let starter_sentinel = provider_profile_for_base_url("aivo-starter");
        assert_eq!(starter_sentinel.quirks.max_tokens_cap, None);
        assert!(!starter_sentinel.quirks.requires_reasoning_content);
    }

    #[test]
    fn normalizes_cloudflare_ai_root() {
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
        assert_eq!(
            cloudflare_ai_base("https://api.cloudflare.com/client/v4/accounts/abc/ai"),
            Some("https://api.cloudflare.com/client/v4/accounts/abc/ai".to_string())
        );
        assert_eq!(cloudflare_ai_base("https://api.openai.com/v1"), None);
    }

    #[test]
    fn is_direct_openai_base_matches_api_openai_com() {
        use super::is_direct_openai_base;
        assert!(is_direct_openai_base("https://api.openai.com/v1"));
        assert!(is_direct_openai_base("https://api.openai.com/v1/"));
        assert!(is_direct_openai_base("https://API.OPENAI.COM/v1"));
        assert!(!is_direct_openai_base("https://api.example.com/v1"));
        assert!(!is_direct_openai_base("copilot"));
    }

    #[test]
    fn provider_quirks_inject_populates_env() {
        use super::ProviderQuirks;
        use std::collections::HashMap;

        let quirks = ProviderQuirks {
            model_prefix: Some("@cf/"),
            requires_reasoning_content: true,
            max_tokens_cap: Some(8192),
            strips_cache_control: true,
        };
        let mut env = HashMap::new();
        quirks.inject(&mut env, "TEST");

        assert_eq!(env.get("TEST_MODEL_PREFIX").unwrap(), "@cf/");
        assert_eq!(env.get("TEST_REQUIRE_REASONING").unwrap(), "1");
        assert_eq!(env.get("TEST_MAX_TOKENS_CAP").unwrap(), "8192");
        assert_eq!(env.get("TEST_STRIP_CACHE_CONTROL").unwrap(), "1");
    }

    #[test]
    fn provider_quirks_inject_skips_none_fields() {
        use super::ProviderQuirks;
        use std::collections::HashMap;

        let quirks = ProviderQuirks {
            model_prefix: None,
            requires_reasoning_content: false,
            max_tokens_cap: None,
            strips_cache_control: false,
        };
        let mut env = HashMap::new();
        quirks.inject(&mut env, "TEST");

        assert!(env.is_empty());
    }

    #[test]
    fn parse_aws_region_accepts_bare_region() {
        use super::parse_aws_region;
        assert_eq!(parse_aws_region("us-east-1"), Some("us-east-1".into()));
        assert_eq!(
            parse_aws_region("eu-central-1"),
            Some("eu-central-1".into())
        );
        assert_eq!(
            parse_aws_region("ap-northeast-1"),
            Some("ap-northeast-1".into())
        );
        assert_eq!(
            parse_aws_region("us-gov-east-1"),
            Some("us-gov-east-1".into())
        );
        assert_eq!(parse_aws_region("  us-west-2  "), Some("us-west-2".into()));
    }

    #[test]
    fn parse_aws_region_accepts_mantle_url() {
        use super::parse_aws_region;
        assert_eq!(
            parse_aws_region("https://bedrock-mantle.us-east-1.api.aws/v1"),
            Some("us-east-1".into())
        );
        assert_eq!(
            parse_aws_region("https://bedrock-mantle.ap-northeast-1.api.aws/v1"),
            Some("ap-northeast-1".into())
        );
        assert_eq!(
            parse_aws_region("bedrock-mantle.eu-central-1.api.aws/v1"),
            Some("eu-central-1".into())
        );
    }

    #[test]
    fn parse_aws_region_accepts_invoke_url() {
        use super::parse_aws_region;
        assert_eq!(
            parse_aws_region("https://bedrock-runtime.us-east-1.amazonaws.com"),
            Some("us-east-1".into())
        );
        assert_eq!(
            parse_aws_region("http://bedrock-runtime.eu-central-1.amazonaws.com"),
            Some("eu-central-1".into())
        );
        assert_eq!(
            parse_aws_region("bedrock-runtime.ap-northeast-1.amazonaws.com"),
            Some("ap-northeast-1".into())
        );
    }

    #[test]
    fn parse_aws_region_tolerates_trailing_slash_and_path() {
        use super::parse_aws_region;
        assert_eq!(
            parse_aws_region("https://bedrock-mantle.us-east-1.api.aws/v1/"),
            Some("us-east-1".into())
        );
        assert_eq!(
            parse_aws_region("https://bedrock-runtime.us-east-1.amazonaws.com/"),
            Some("us-east-1".into())
        );
        assert_eq!(
            parse_aws_region("bedrock-runtime.us-west-2.amazonaws.com/model/foo/invoke"),
            Some("us-west-2".into())
        );
    }

    #[test]
    fn parse_aws_region_rejects_garbage() {
        use super::parse_aws_region;
        assert_eq!(parse_aws_region(""), None);
        assert_eq!(parse_aws_region("   "), None);
        assert_eq!(parse_aws_region("us east 1"), None);
        assert_eq!(parse_aws_region("US-EAST-1"), None);
        assert_eq!(parse_aws_region("garbage"), None);
        assert_eq!(parse_aws_region("https://api.openai.com/v1"), None);
        assert_eq!(
            parse_aws_region("https://bedrock-runtime.us-east-1.example.com"),
            None
        );
        assert_eq!(
            parse_aws_region("https://bedrock-mantle.us-east-1.example.com"),
            None
        );
        assert_eq!(parse_aws_region("us-east-1.amazonaws.com"), None);
    }
}
