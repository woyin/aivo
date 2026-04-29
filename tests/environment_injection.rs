use aivo::services::environment_injector::EnvironmentInjector;
use aivo::services::session_store::ApiKey;
use zeroize::Zeroizing;

fn make_key(base_url: &str, key_secret: &str) -> ApiKey {
    ApiKey {
        id: "abc".to_string(),
        name: "test".to_string(),
        base_url: base_url.to_string(),
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
        key: Zeroizing::new(key_secret.to_string()),
        created_at: "2025-01-01T00:00:00Z".to_string(),
    }
}

// ── Claude ────────────────────────────────────────────────────────────

#[test]
fn claude_direct_anthropic() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.anthropic.com/v1", "sk-ant-test");
    let env = inj.for_claude(&key, Some("claude-sonnet-4-6"));

    assert_eq!(
        env.get("ANTHROPIC_BASE_URL").unwrap(),
        "https://api.anthropic.com"
    );
    assert_eq!(env.get("ANTHROPIC_AUTH_TOKEN").unwrap(), "sk-ant-test");
    assert_eq!(
        env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC").unwrap(),
        "1"
    );
    // Model vars set
    assert!(env.contains_key("ANTHROPIC_MODEL"));
    // Direct Anthropic uses native name (hyphens)
    assert_eq!(env.get("ANTHROPIC_MODEL").unwrap(), "claude-sonnet-4-6");
}

#[test]
fn claude_openrouter_uses_router() {
    let inj = EnvironmentInjector;
    let key = make_key("https://openrouter.ai/api/v1", "sk-or-test");
    let env = inj.for_claude(&key, None);

    assert_eq!(env.get("AIVO_USE_ROUTER").unwrap(), "1");
    assert_eq!(env.get("AIVO_ROUTER_API_KEY").unwrap(), "sk-or-test");
    assert_eq!(
        env.get("AIVO_ROUTER_BASE_URL").unwrap(),
        "https://openrouter.ai/api/v1"
    );
}

#[test]
fn claude_copilot_uses_copilot_router() {
    let inj = EnvironmentInjector;
    let key = make_key("copilot", "gho_test_token");
    let env = inj.for_claude(&key, None);

    assert_eq!(env.get("AIVO_USE_COPILOT_ROUTER").unwrap(), "1");
    assert_eq!(
        env.get("AIVO_COPILOT_GITHUB_TOKEN").unwrap(),
        "gho_test_token"
    );
    assert_eq!(env.get("ANTHROPIC_AUTH_TOKEN").unwrap(), "copilot");
}

#[test]
fn claude_openai_compat_uses_anthropic_to_openai_router() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.deepseek.com/v1", "sk-ds-test");
    let env = inj.for_claude(&key, None);

    assert_eq!(env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER").unwrap(), "1");
    assert_eq!(
        env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY").unwrap(),
        "sk-ds-test"
    );
    assert_eq!(
        env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL").unwrap(),
        "https://api.deepseek.com/v1"
    );
}

#[test]
fn claude_per_key_requires_reasoning_override_injects_strict_flag() {
    let inj = EnvironmentInjector;
    // A host that does NOT match the static substring list (no DeepSeek /
    // Moonshot in the URL) — without the per-key learned override, strict
    // mode would not be enabled.
    let mut key = make_key("https://api.example-thinking.dev/v1", "sk-test");
    key.requires_reasoning_content = Some(true);
    let env = inj.for_claude(&key, None);

    assert_eq!(
        env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING")
            .map(String::as_str),
        Some("1"),
        "per-key learned quirk must inject strict mode for hosts not in the static list"
    );
}

#[test]
fn claude_without_per_key_override_does_not_force_strict_flag() {
    let inj = EnvironmentInjector;
    // No DeepSeek/Moonshot in URL, no per-key override → no strict flag.
    let key = make_key("https://api.example-thinking.dev/v1", "sk-test");
    let env = inj.for_claude(&key, None);

    assert!(!env.contains_key("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING"));
}

#[test]
fn claude_ollama_uses_anthropic_to_openai_router() {
    let inj = EnvironmentInjector;
    let key = make_key("ollama", "ollama");
    let env = inj.for_claude(&key, None);

    assert_eq!(env.get("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER").unwrap(), "1");
    assert_eq!(
        env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY").unwrap(),
        "ollama"
    );
    assert_eq!(
        env.get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL")
            .unwrap(),
        "openai"
    );
}

#[test]
fn claude_no_model_omits_model_vars() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.anthropic.com/v1", "sk-ant-test");
    let env = inj.for_claude(&key, None);

    assert!(!env.contains_key("ANTHROPIC_MODEL"));
}

// ── Codex ─────────────────────────────────────────────────────────────

#[test]
fn codex_direct_openai() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.openai.com/v1", "sk-oai-test");
    let env = inj.for_codex(&key, None);

    // Direct OpenAI should NOT use any router
    assert!(!env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER"));
    assert_eq!(env.get("OPENAI_API_KEY").unwrap(), "sk-oai-test");
}

#[test]
fn codex_non_openai_uses_responses_router() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.deepseek.com/v1", "sk-ds-test");
    let env = inj.for_codex(&key, None);

    assert_eq!(env.get("AIVO_USE_RESPONSES_TO_CHAT_ROUTER").unwrap(), "1");
    assert_eq!(
        env.get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY").unwrap(),
        "sk-ds-test"
    );
}

#[test]
fn codex_copilot_uses_copilot_router() {
    let inj = EnvironmentInjector;
    let key = make_key("copilot", "gho_test_token");
    let env = inj.for_codex(&key, None);

    assert_eq!(
        env.get("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER")
            .unwrap(),
        "1"
    );
    assert_eq!(
        env.get("AIVO_COPILOT_GITHUB_TOKEN").unwrap(),
        "gho_test_token"
    );
}

// ── Gemini ────────────────────────────────────────────────────────────

#[test]
fn gemini_direct_google() {
    let inj = EnvironmentInjector;
    let key = make_key(
        "https://generativelanguage.googleapis.com",
        "ai-google-test",
    );
    let env = inj.for_gemini(&key, Some("gemini-2.5-pro"));

    // Direct Google should NOT route through Gemini router
    assert!(!env.contains_key("AIVO_USE_GEMINI_ROUTER"));
    assert!(env.contains_key("GEMINI_API_KEY"));
}

#[test]
fn gemini_non_google_uses_gemini_router() {
    let inj = EnvironmentInjector;
    let key = make_key("https://api.openai.com/v1", "sk-oai-test");
    let env = inj.for_gemini(&key, None);

    assert_eq!(env.get("AIVO_USE_GEMINI_ROUTER").unwrap(), "1");
}
