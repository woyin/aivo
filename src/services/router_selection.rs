//! Tool→upstream routing decision: which `ConnectionMode` each CLI uses for a
//! key. Pure policy — env-var construction stays in `environment_injector`.

use crate::services::provider_profile::{
    ProviderKind, ProviderProfile, is_direct_openai_base, provider_profile_for_key,
};
use crate::services::provider_protocol::{
    ProviderProtocol, is_anthropic_endpoint, is_google_endpoint,
};
use crate::services::session_store::{ApiKey, OpenAICompatibilityMode};

/// How a tool should connect to the upstream provider.
pub(crate) enum ConnectionMode {
    Ollama,
    Copilot,
    OpenRouter,
    Direct { base_url: String },
    Routed { protocol: ProviderProtocol },
}

/// The tool's default upstream protocol (the route map's `""` entry) —
/// drives the Direct/Routed decision, replacing the old scalar pins.
pub(crate) fn tool_default_protocol(key: &ApiKey, tool: &str) -> Option<ProviderProtocol> {
    key.protocol_routes
        .get(tool)
        .and_then(|models| models.get(""))
        .and_then(|route| ProviderProtocol::parse(&route.protocol))
}

/// Returns true when the URL points to a native Anthropic endpoint that speaks
/// the Anthropic Messages API directly (no format conversion needed).
///
/// Invariant: Direct mode requires a native endpoint. The `claude_protocol`
/// pin expresses "send this protocol through the router first" and must not
/// on its own bypass the router, since the router is where protocol fallback
/// runs for generic OpenAI-compatible hosts.
pub(crate) fn use_direct_anthropic_for_claude(key: &ApiKey) -> bool {
    // When the HTTP debug logger is initialized, force the bridge so the
    // outbound translation/forward call is observable. The bridge's
    // existing forward sites are instrumented (`.send_logged()`); routing
    // through them is what makes `--debug` capture native-Anthropic
    // upstreams (e.g. minimax/deepseek configured with `/anthropic` base
    // URLs). The override returns `false` so the caller falls into the
    // routed branch.
    if crate::services::http_debug::is_debug_active() {
        return false;
    }
    if crate::services::transform_mode::is_transparent() {
        return true;
    }
    if !is_anthropic_endpoint(&key.base_url) {
        return false;
    }
    match tool_default_protocol(key, "claude") {
        Some(ProviderProtocol::Anthropic) | None => true,
        Some(_) => false,
    }
}

pub(crate) fn use_direct_openai_for_codex(key: &ApiKey) -> bool {
    // See `use_direct_anthropic_for_claude`: under `--debug`, force the
    // bridge so outbound traffic flows through `responses_to_chat_router`
    // (which is instrumented with `.send_logged()`).
    if crate::services::http_debug::is_debug_active() {
        return false;
    }
    if crate::services::transform_mode::is_transparent() {
        return true;
    }
    match key.codex_mode {
        Some(OpenAICompatibilityMode::Direct) => true,
        Some(OpenAICompatibilityMode::Router) => false,
        None => is_direct_openai_base(&key.base_url),
    }
}

pub(crate) fn use_google_native_for_gemini(key: &ApiKey) -> bool {
    // See `use_direct_anthropic_for_claude`: under `--debug`, force the
    // bridge so outbound traffic flows through `gemini_router` (which is
    // instrumented with `.send_logged()`).
    if crate::services::http_debug::is_debug_active() {
        return false;
    }
    if crate::services::transform_mode::is_transparent() {
        return true;
    }
    // Same invariant as use_direct_anthropic_for_claude: only a genuinely
    // Google-native endpoint may skip the router.
    if !is_google_endpoint(&key.base_url) {
        return false;
    }
    match tool_default_protocol(key, "gemini") {
        Some(ProviderProtocol::Google) | None => true,
        Some(_) => false,
    }
}

pub(crate) fn use_router_for_opencode(key: &ApiKey) -> bool {
    // OpenCode already routes through the local bridge whenever
    // `opencode_mode == Router`. Under `--debug`, force the bridge for
    // direct-mode keys too so outbound traffic is visible.
    if crate::services::http_debug::is_debug_active() {
        return true;
    }
    if crate::services::transform_mode::is_transparent() {
        return false;
    }
    matches!(key.opencode_mode, Some(OpenAICompatibilityMode::Router))
}

/// `--transparent` preflight: Some(reason) when the key is router-bound.
pub(crate) fn transparent_conflict(
    tool: crate::services::ai_launcher::AIToolType,
    key: &ApiKey,
    profile: &ProviderProfile,
) -> Option<String> {
    use crate::services::ai_launcher::AIToolType;
    if tool == AIToolType::Grok {
        return Some(
            "grok is always served through aivo's local router — token accounting and protocol \
             bridging live there"
                .to_string(),
        );
    }
    if key.is_cursor_acp() {
        return Some(
            "cursor keys speak ACP, not HTTP — there is no upstream to talk to directly"
                .to_string(),
        );
    }
    if profile.serve_flags.is_copilot {
        return Some(
            "GitHub Copilot auth (token exchange + headers) lives in the local router".to_string(),
        );
    }
    // Deliberately reasonless.
    if profile.serve_flags.is_starter {
        return Some(String::new());
    }
    if profile.kind == ProviderKind::Ollama && tool != AIToolType::Opencode {
        return Some(format!(
            "Ollama speaks OpenAI chat only — {} needs the conversion router",
            tool.as_str()
        ));
    }
    if profile.serve_flags.is_openrouter && tool == AIToolType::Claude {
        return Some(
            "OpenRouter speaks the OpenAI protocol — claude needs the conversion router"
                .to_string(),
        );
    }
    None
}

pub(crate) fn routed_protocol_for_claude(key: &ApiKey) -> ProviderProtocol {
    tool_default_protocol(key, "claude").unwrap_or_else(|| {
        provider_profile_for_key(key).upstream_protocol_for_cli(ProviderProtocol::Anthropic)
    })
}

pub(crate) fn routed_protocol_for_gemini(key: &ApiKey) -> ProviderProtocol {
    tool_default_protocol(key, "gemini").unwrap_or_else(|| {
        provider_profile_for_key(key).upstream_protocol_for_cli(ProviderProtocol::Google)
    })
}

pub(crate) fn claude_connection_mode(key: &ApiKey, profile: &ProviderProfile) -> ConnectionMode {
    if profile.kind == ProviderKind::Ollama {
        ConnectionMode::Ollama
    } else if profile.serve_flags.is_copilot {
        ConnectionMode::Copilot
    } else if profile.serve_flags.is_openrouter {
        ConnectionMode::OpenRouter
    } else if use_direct_anthropic_for_claude(key) && !profile.serve_flags.is_starter {
        // Starter must route through the local router — it's the only
        // place device_fingerprint::maybe_with_starter_headers runs.
        // Direct mode would skip the X-Aivo-* headers and 403 at the gateway.
        let base_url = key.base_url.trim_end_matches('/');
        let base_url = base_url.strip_suffix("/v1").unwrap_or(base_url);
        ConnectionMode::Direct {
            base_url: base_url.to_string(),
        }
    } else {
        ConnectionMode::Routed {
            protocol: routed_protocol_for_claude(key),
        }
    }
}

pub(crate) fn codex_connection_mode(key: &ApiKey, profile: &ProviderProfile) -> ConnectionMode {
    if profile.kind == ProviderKind::Ollama {
        ConnectionMode::Ollama
    } else if profile.serve_flags.is_copilot {
        ConnectionMode::Copilot
    } else if !use_direct_openai_for_codex(key) || profile.serve_flags.is_starter {
        // See claude_connection_mode: starter must route through the local
        // router so device_fingerprint headers attach.
        // Why ResponsesApi: seeds the router's cascade with `/v1/responses`
        // first so codex's native protocol is a pass-through; chat
        // completions remains in the fallback chain for legacy hosts.
        ConnectionMode::Routed {
            protocol: profile.upstream_protocol_for_cli(ProviderProtocol::ResponsesApi),
        }
    } else {
        ConnectionMode::Direct {
            base_url: key.base_url.clone(),
        }
    }
}

/// `direct_base_url` is the claude/gemini-style pre-munged URL (the Gemini CLI
/// needs the `/v1beta` suffix stripped); the caller owns that transform.
pub(crate) fn gemini_connection_mode(
    key: &ApiKey,
    profile: &ProviderProfile,
    direct_base_url: &str,
) -> ConnectionMode {
    if profile.kind == ProviderKind::Ollama {
        ConnectionMode::Ollama
    } else if profile.serve_flags.is_copilot {
        ConnectionMode::Copilot
    } else if use_google_native_for_gemini(key) && !profile.serve_flags.is_starter {
        // See claude_connection_mode: starter must route through the local
        // router so device_fingerprint headers attach.
        ConnectionMode::Direct {
            base_url: direct_base_url.to_string(),
        }
    } else {
        ConnectionMode::Routed {
            protocol: routed_protocol_for_gemini(key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::route_cache::PersistedRoute;

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

    fn test_api_key(base_url: &str) -> ApiKey {
        ApiKey::new_with_protocol(
            "a1b2".to_string(),
            "test-key".to_string(),
            base_url.to_string(),
            None,
            "sk-test-key-12345".to_string(),
        )
    }

    /// Tests toggling or assuming the debug/transparent globals serialize
    /// via `DEBUG_TEST_MUTEX`; both flags reset here.
    fn debug_off_guard() -> std::sync::MutexGuard<'static, ()> {
        let guard = crate::services::http_debug::DEBUG_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_transparent(false);
        guard
    }

    #[test]
    fn use_direct_anthropic_false_for_generic_openai_host_with_anthropic_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://api.example.com/v1");
        pin_route(&mut key, "claude", "anthropic");
        assert!(!use_direct_anthropic_for_claude(&key));
    }

    #[test]
    fn use_direct_anthropic_true_for_anthropic_host_with_no_pin() {
        let _guard = debug_off_guard();
        let key = test_api_key("https://api.anthropic.com");
        assert!(use_direct_anthropic_for_claude(&key));
    }

    #[test]
    fn use_direct_anthropic_true_for_anthropic_host_with_anthropic_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://api.anthropic.com");
        pin_route(&mut key, "claude", "anthropic");
        assert!(use_direct_anthropic_for_claude(&key));
    }

    #[test]
    fn use_direct_anthropic_false_for_anthropic_host_with_openai_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://api.anthropic.com");
        pin_route(&mut key, "claude", "openai");
        assert!(!use_direct_anthropic_for_claude(&key));
    }

    #[test]
    fn use_google_native_false_for_generic_openai_host_with_google_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://api.example.com/v1");
        pin_route(&mut key, "gemini", "google");
        assert!(!use_google_native_for_gemini(&key));
    }

    #[test]
    fn use_google_native_true_for_google_host_with_no_pin() {
        let _guard = debug_off_guard();
        let key = test_api_key("https://generativelanguage.googleapis.com/v1beta");
        assert!(use_google_native_for_gemini(&key));
    }

    #[test]
    fn use_google_native_true_for_google_host_with_google_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://generativelanguage.googleapis.com/v1beta");
        pin_route(&mut key, "gemini", "google");
        assert!(use_google_native_for_gemini(&key));
    }

    #[test]
    fn use_google_native_false_for_google_host_with_openai_pin() {
        let _guard = debug_off_guard();
        let mut key = test_api_key("https://generativelanguage.googleapis.com/v1beta");
        pin_route(&mut key, "gemini", "openai");
        assert!(!use_google_native_for_gemini(&key));
    }

    #[test]
    fn transparent_forces_direct_on_generic_hosts_and_over_pins() {
        let _guard = debug_off_guard();
        crate::services::transform_mode::set_transparent(true);

        let mut key = test_api_key("https://api.example.com/v1");
        assert!(use_direct_anthropic_for_claude(&key));
        assert!(use_direct_openai_for_codex(&key));
        assert!(use_google_native_for_gemini(&key));
        assert!(!use_router_for_opencode(&key));

        key.codex_mode = Some(OpenAICompatibilityMode::Router);
        key.opencode_mode = Some(OpenAICompatibilityMode::Router);
        pin_route(&mut key, "claude", "openai");
        assert!(use_direct_openai_for_codex(&key));
        assert!(!use_router_for_opencode(&key));
        assert!(use_direct_anthropic_for_claude(&key));

        crate::services::transform_mode::set_transparent(false);
    }

    #[test]
    fn debug_beats_transparent() {
        let _guard = debug_off_guard();
        crate::services::http_debug::set_test_debug_active(true);
        crate::services::transform_mode::set_transparent(true);

        let key = test_api_key("https://api.example.com/v1");
        assert!(!use_direct_anthropic_for_claude(&key));
        assert!(!use_direct_openai_for_codex(&key));
        assert!(!use_google_native_for_gemini(&key));
        assert!(use_router_for_opencode(&key));

        crate::services::http_debug::set_test_debug_active(false);
        crate::services::transform_mode::set_transparent(false);
    }

    #[test]
    fn transparent_conflict_rejects_router_bound_keys() {
        use crate::services::ai_launcher::AIToolType;
        let reject = |base_url: &str, tool: AIToolType| {
            let key = test_api_key(base_url);
            let profile = provider_profile_for_key(&key);
            transparent_conflict(tool, &key, &profile)
        };

        assert!(reject("copilot", AIToolType::Codex).is_some());
        assert!(reject("aivo-starter", AIToolType::Claude).is_some());
        assert!(reject("cursor", AIToolType::Claude).is_some());
        assert!(reject("ollama", AIToolType::Codex).is_some());
        assert!(reject("ollama", AIToolType::Claude).is_some());
        assert!(reject("ollama", AIToolType::Gemini).is_some());
        assert!(reject("ollama", AIToolType::Opencode).is_none());
        assert!(reject("https://openrouter.ai/api/v1", AIToolType::Claude).is_some());
        assert!(reject("https://openrouter.ai/api/v1", AIToolType::Codex).is_none());
        assert!(reject("https://openrouter.ai/api/v1", AIToolType::Opencode).is_none());
        assert!(reject("https://api.example.com/v1", AIToolType::Codex).is_none());
        assert!(reject("https://api.anthropic.com", AIToolType::Claude).is_none());
    }
}
