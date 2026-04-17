use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, SessionStore,
};

pub(crate) struct LaunchRuntimeState {
    pub(crate) env: HashMap<String, String>,
    pub(crate) router_protocol: Option<Arc<AtomicU8>>,
    pub(crate) responses_api_support: Option<Arc<AtomicU8>>,
    pub(crate) pi_agent_dir: Option<String>,
}

pub(crate) async fn prepare_runtime_env(
    tool: AIToolType,
    mut env: HashMap<String, String>,
) -> Result<LaunchRuntimeState> {
    let mut router_protocol = None;
    let mut responses_api_support = None;

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
        let port = start_anthropic_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER") {
        let (port, active) = start_anthropic_to_openai_router(&env).await?;
        router_protocol = Some(active);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_COPILOT_ROUTER") {
        let port = start_copilot_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER") {
        let (port, _active, responses_api) = start_responses_to_chat_router(&env).await?;
        responses_api_support = Some(responses_api);
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
        let (port, active) = start_gemini_router(&env).await?;
        router_protocol = Some(active);
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER") {
        let port = start_gemini_copilot_router(&env).await?;
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_ROUTER") {
        let (port, _active, _responses_api) = start_responses_to_chat_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_SETUP_PI_AGENT_DIR") {
        // Direct connection — no router needed, just write the temp agent dir.
        write_pi_agent_dir(&mut env, None).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_STARTER_ROUTER") {
        let (port, _active, _responses_api) = start_responses_to_chat_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    let pi_agent_dir = env.get("PI_CODING_AGENT_DIR").cloned();

    Ok(LaunchRuntimeState {
        env,
        router_protocol,
        responses_api_support,
        pi_agent_dir,
    })
}

pub(crate) async fn record_launch_state(
    session_store: &SessionStore,
    key: &ApiKey,
    tool: AIToolType,
    model: Option<&str>,
) {
    let _ = session_store
        .record_selection(&key.id, tool.as_str(), model)
        .await;
}

pub(crate) async fn persist_runtime_discoveries(
    session_store: &SessionStore,
    tool: AIToolType,
    key: &ApiKey,
    key_override_used: bool,
    router_protocol: Option<Arc<AtomicU8>>,
    responses_api_support: Option<Arc<AtomicU8>>,
) {
    if key_override_used {
        return;
    }

    if let Some(active) = router_protocol {
        let final_protocol = ProviderProtocol::from_u8(active.load(Ordering::Relaxed));
        match tool {
            AIToolType::Claude => {
                let current = key
                    .claude_protocol
                    .map(|p| match p {
                        ClaudeProviderProtocol::Openai => ProviderProtocol::Openai,
                        ClaudeProviderProtocol::Anthropic => ProviderProtocol::Anthropic,
                        ClaudeProviderProtocol::Google => ProviderProtocol::Google,
                    })
                    .unwrap_or(ProviderProtocol::Openai);
                if final_protocol != current {
                    let protocol = match final_protocol {
                        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                            ClaudeProviderProtocol::Openai
                        }
                        ProviderProtocol::Anthropic => ClaudeProviderProtocol::Anthropic,
                        ProviderProtocol::Google => ClaudeProviderProtocol::Google,
                    };
                    let _ = session_store
                        .set_key_claude_protocol(&key.id, Some(protocol))
                        .await;
                }
            }
            AIToolType::Gemini => {
                let current = key
                    .gemini_protocol
                    .map(|p| match p {
                        GeminiProviderProtocol::Google => ProviderProtocol::Google,
                        GeminiProviderProtocol::Openai => ProviderProtocol::Openai,
                        GeminiProviderProtocol::Anthropic => ProviderProtocol::Anthropic,
                    })
                    .unwrap_or(ProviderProtocol::Openai);
                if final_protocol != current {
                    let protocol = match final_protocol {
                        ProviderProtocol::Google => GeminiProviderProtocol::Google,
                        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => {
                            GeminiProviderProtocol::Openai
                        }
                        ProviderProtocol::Anthropic => GeminiProviderProtocol::Anthropic,
                    };
                    let _ = session_store
                        .set_key_gemini_protocol(&key.id, Some(protocol))
                        .await;
                }
            }
            _ => {}
        }
    }

    if let Some(active) = responses_api_support {
        let final_val = match active.load(Ordering::Relaxed) {
            1 => Some(true),
            2 => Some(false),
            _ => None,
        };
        if final_val.is_some() && final_val != key.responses_api_supported {
            let _ = session_store
                .set_key_responses_api_supported(&key.id, final_val)
                .await;
        }
    }
}

/// Walk Pi session JSONL files in the temp agent dir and copy them to
/// `~/.pi/agent/sessions/` for long-term storage.
pub(crate) async fn process_pi_sessions(pi_agent_dir: Option<&str>) {
    let temp_dir = match pi_agent_dir {
        Some(d) => d,
        None => return,
    };

    let temp_sessions = std::path::PathBuf::from(temp_dir).join("sessions");
    let real_sessions = crate::services::system_env::home_dir()
        .map(|h| h.join(".pi").join("agent").join("sessions"));

    let mut dirs = vec![temp_sessions.clone()];
    while let Some(dir) = dirs.pop() {
        let mut entries = match tokio::fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }

            if let Some(ref real) = real_sessions
                && let Ok(rel) = path.strip_prefix(&temp_sessions)
            {
                let dest = real.join(rel);
                if let Some(parent) = dest.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                let _ = tokio::fs::copy(&path, &dest).await;
            }
        }
    }
}

pub(crate) async fn cleanup_runtime_artifacts(
    codex_model_catalog_path: Option<&str>,
    pi_agent_dir: Option<&str>,
) {
    if let Some(path) = codex_model_catalog_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    if let Some(dir) = pi_agent_dir {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
}

/// Writes a temporary `PI_CODING_AGENT_DIR` with `models.json`, `auth.json`,
/// and `settings.json` so Pi discovers the aivo custom provider.
///
/// When `port` is `Some`, the placeholder `PLACEHOLDER_LOOPBACK_URL` in
/// `AIVO_PI_MODELS_JSON` is patched with the real router port.
/// When `port` is `None`, the JSON already contains the real upstream URL.
async fn write_pi_agent_dir(env: &mut HashMap<String, String>, port: Option<u16>) -> Result<()> {
    let raw = env
        .get("AIVO_PI_MODELS_JSON")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_PI_MODELS_JSON"))?
        .clone();

    let models_json = match port {
        Some(p) => {
            ensure_loopback_no_proxy(env);
            raw.replace(PLACEHOLDER_LOOPBACK_URL, &format!("http://127.0.0.1:{p}"))
        }
        None => raw,
    };

    let dir = tempfile::Builder::new()
        .prefix("aivo-pi-")
        .tempdir()?
        .keep();

    tokio::try_join!(
        tokio::fs::write(dir.join("models.json"), &models_json),
        tokio::fs::write(dir.join("auth.json"), "{}"),
        tokio::fs::write(dir.join("settings.json"), "{}"),
    )?;

    // Symlink the real pi agent's bin/ directory (contains managed fd, rg binaries)
    // so pi doesn't re-download them into the temp dir.
    #[cfg(unix)]
    if let Some(home) = crate::services::system_env::home_dir() {
        let real_bin = home.join(".pi").join("agent").join("bin");
        let _ = tokio::fs::symlink(&real_bin, dir.join("bin")).await;
    }

    env.insert(
        "PI_CODING_AGENT_DIR".to_string(),
        dir.to_string_lossy().to_string(),
    );
    Ok(())
}

fn set_local_base_url(env: &mut HashMap<String, String>, key: &str, port: u16) {
    env.insert(key.to_string(), format!("http://127.0.0.1:{port}"));
    ensure_loopback_no_proxy(env);
}

fn patch_opencode_config_content(env: &mut HashMap<String, String>, port: u16) {
    let real_url = format!("http://127.0.0.1:{port}");
    if let Some(content) = env.get("OPENCODE_CONFIG_CONTENT").cloned() {
        let patched = content.replace(PLACEHOLDER_LOOPBACK_URL, &real_url);
        env.insert("OPENCODE_CONFIG_CONTENT".to_string(), patched);
        ensure_loopback_no_proxy(env);
    }
}

/// Ensures the spawned subprocess will bypass any HTTP proxy when talking to
/// the local loopback router. Without this, a user's `HTTP_PROXY`/`HTTPS_PROXY`
/// would route the subprocess's `http://127.0.0.1:<port>` request through the
/// proxy, which cannot reach the user's localhost. We append loopback entries
/// to both the upper- and lower-case variants since different HTTP libraries
/// check different casings.
fn ensure_loopback_no_proxy(env: &mut HashMap<String, String>) {
    const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];
    for var in ["NO_PROXY", "no_proxy"] {
        let existing = env.get(var).cloned().unwrap_or_default();
        let mut entries: Vec<String> = existing
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        for host in LOOPBACK_HOSTS {
            // Case-insensitive check: a pre-existing `LOCALHOST` or
            // `127.0.0.1` entry should not get duplicated with `localhost`.
            if !entries.iter().any(|e| e.eq_ignore_ascii_case(host)) {
                entries.push((*host).to_string());
            }
        }
        env.insert(var.to_string(), entries.join(","));
    }
}

/// Starts the built-in AnthropicRouter and returns the port it bound to
async fn start_anthropic_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{AnthropicRouter, AnthropicRouterConfig};

    let api_key = env
        .get("AIVO_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_ROUTER_BASE_URL"))?
        .clone();

    let config = AnthropicRouterConfig {
        upstream_base_url: base_url,
        upstream_api_key: api_key,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    };

    let router = AnthropicRouter::new(config);
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_anthropic_to_openai_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig};

    let api_key = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing anthropic-to-openai router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let config = AnthropicToOpenAIRouterConfig {
        target_base_url: base_url,
        target_api_key: api_key,
        target_protocol,
        model_prefix,
        requires_reasoning_content,
        max_tokens_cap,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    };

    let router = AnthropicToOpenAIRouter::new(config);
    let (port, active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic-to-openai router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol))
}

async fn start_responses_to_chat_router(
    env: &HashMap<String, String>,
) -> Result<(u16, Arc<AtomicU8>, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let api_key = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router API key"))?
        .clone();

    let base_url = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing responses-to-chat router base URL"))?
        .clone();

    let model_prefix = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MODEL_PREFIX")
        .cloned();
    let requires_reasoning_content = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let actual_model = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_ACTUAL_MODEL")
        .cloned();
    let max_tokens_cap = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let target_protocol = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let responses_api_supported = match env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_RESPONSES_API")
        .map(|v| v.as_str())
    {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    };

    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        copilot_token_manager: None,
        model_prefix,
        requires_reasoning_content,
        actual_model,
        max_tokens_cap,
        responses_api_supported,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    });
    let (port, active_protocol, responses_api, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol, responses_api))
}

async fn start_gemini_router(env: &HashMap<String, String>) -> Result<(u16, Arc<AtomicU8>)> {
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let api_key = env
        .get("AIVO_GEMINI_ROUTER_API_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_API_KEY"))?
        .clone();

    let base_url = env
        .get("AIVO_GEMINI_ROUTER_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_GEMINI_ROUTER_BASE_URL"))?
        .clone();

    let requires_reasoning_content = env
        .get("AIVO_GEMINI_ROUTER_REQUIRE_REASONING")
        .map(|v| v == "1")
        .unwrap_or(false);
    let max_tokens_cap = env
        .get("AIVO_GEMINI_ROUTER_MAX_TOKENS_CAP")
        .and_then(|v| v.parse::<u64>().ok());
    let upstream_protocol = env
        .get("AIVO_GEMINI_ROUTER_UPSTREAM_PROTOCOL")
        .and_then(|value| ProviderProtocol::parse(value))
        .unwrap_or_else(|| detect_provider_protocol(&base_url));
    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: base_url,
        api_key,
        upstream_protocol,
        forced_model: None,
        copilot_token_manager: None,
        requires_reasoning_content,
        max_tokens_cap,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    });
    let (port, active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok((port, active_protocol))
}

async fn start_gemini_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{GeminiRouter, GeminiRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let forced_model = env.get("AIVO_GEMINI_COPILOT_FORCED_MODEL").cloned();

    if forced_model.is_none() {
        eprintln!(
            "  {} Gemini + Copilot: no model specified. Gemini models are not available on \
             Copilot. Pass --model <model> (e.g., --model gpt-4o).",
            crate::style::yellow("Warning:")
        );
    }

    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        upstream_protocol: ProviderProtocol::ResponsesApi,
        forced_model,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    });
    let (port, _active_protocol, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::{CopilotRouter, CopilotRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = CopilotRouter::new(CopilotRouterConfig { github_token });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

async fn start_responses_to_chat_copilot_router(env: &HashMap<String, String>) -> Result<u16> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let github_token = env
        .get("AIVO_COPILOT_GITHUB_TOKEN")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_COPILOT_GITHUB_TOKEN"))?
        .clone();

    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: String::new(),
        api_key: String::new(),
        target_protocol: ProviderProtocol::Openai,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: false,
    });
    let (port, _active_protocol, _responses_api, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::patch_opencode_config_content;
    use std::collections::HashMap;

    #[test]
    fn patch_opencode_config_content_rewrites_placeholder_url() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"http://127.0.0.1:0\"}".to_string(),
        )]);

        patch_opencode_config_content(&mut env, 24860);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"http://127.0.0.1:24860\"}"
        );
    }

    #[test]
    fn patch_opencode_config_content_ignores_missing_payload() {
        let mut env = HashMap::new();
        patch_opencode_config_content(&mut env, 24860);
        assert!(env.is_empty());
    }

    #[test]
    fn set_local_base_url_inserts_loopback_address() {
        use super::set_local_base_url;
        let mut env = HashMap::new();
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:9999"
        );
    }

    #[test]
    fn set_local_base_url_overwrites_existing() {
        use super::set_local_base_url;
        let mut env = HashMap::from([(
            "ANTHROPIC_BASE_URL".to_string(),
            "https://old-url.example.com".to_string(),
        )]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 12345);
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:12345"
        );
    }

    #[test]
    fn patch_opencode_config_content_preserves_non_placeholder() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"https://api.openai.com/v1\"}".to_string(),
        )]);

        patch_opencode_config_content(&mut env, 24860);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"https://api.openai.com/v1\"}"
        );
    }

    #[test]
    fn patch_opencode_config_content_replaces_multiple_occurrences() {
        use crate::constants::PLACEHOLDER_LOOPBACK_URL;

        let content = format!(
            "{{\"url1\":\"{}\",\"url2\":\"{}\"}}",
            PLACEHOLDER_LOOPBACK_URL, PLACEHOLDER_LOOPBACK_URL
        );
        let mut env = HashMap::from([("OPENCODE_CONFIG_CONTENT".to_string(), content)]);

        patch_opencode_config_content(&mut env, 55555);

        let result = env.get("OPENCODE_CONFIG_CONTENT").unwrap();
        assert!(!result.contains(PLACEHOLDER_LOOPBACK_URL));
        assert_eq!(result.matches("http://127.0.0.1:55555").count(), 2);
    }

    #[test]
    fn patch_opencode_config_content_uses_constant() {
        use crate::constants::PLACEHOLDER_LOOPBACK_URL;

        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            format!("{{\"baseUrl\":\"{}\"}}", PLACEHOLDER_LOOPBACK_URL),
        )]);

        patch_opencode_config_content(&mut env, 9876);

        assert_eq!(
            env.get("OPENCODE_CONFIG_CONTENT").unwrap(),
            "{\"baseUrl\":\"http://127.0.0.1:9876\"}"
        );
    }

    #[test]
    fn set_local_base_url_injects_loopback_no_proxy() {
        use super::set_local_base_url;
        let mut env = HashMap::new();
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").expect("NO_PROXY should be set");
        assert!(no_proxy.contains("127.0.0.1"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("localhost"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("::1"), "NO_PROXY={no_proxy}");

        let no_proxy_lower = env.get("no_proxy").expect("no_proxy should be set");
        assert!(no_proxy_lower.contains("127.0.0.1"));
    }

    #[test]
    fn set_local_base_url_appends_to_existing_no_proxy() {
        use super::set_local_base_url;
        let mut env = HashMap::from([("NO_PROXY".to_string(), "internal.corp".to_string())]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        assert!(no_proxy.contains("internal.corp"), "NO_PROXY={no_proxy}");
        assert!(no_proxy.contains("127.0.0.1"), "NO_PROXY={no_proxy}");
    }

    #[test]
    fn set_local_base_url_does_not_duplicate_existing_loopback_entry() {
        use super::set_local_base_url;
        let mut env = HashMap::from([(
            "NO_PROXY".to_string(),
            "127.0.0.1,localhost,::1".to_string(),
        )]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        assert_eq!(
            no_proxy.matches("127.0.0.1").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
        assert_eq!(
            no_proxy.matches("localhost").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
    }

    #[test]
    fn set_local_base_url_treats_existing_loopback_entries_case_insensitively() {
        use super::set_local_base_url;
        let mut env = HashMap::from([("NO_PROXY".to_string(), "LOCALHOST,127.0.0.1".to_string())]);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", 9999);

        let no_proxy = env.get("NO_PROXY").unwrap();
        // Existing uppercase LOCALHOST should be preserved without a
        // duplicate lowercase `localhost` appended.
        assert!(no_proxy.contains("LOCALHOST"), "NO_PROXY={no_proxy}");
        assert_eq!(
            no_proxy.to_ascii_lowercase().matches("localhost").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
        assert_eq!(
            no_proxy.matches("127.0.0.1").count(),
            1,
            "NO_PROXY={no_proxy}"
        );
    }

    #[test]
    fn patch_opencode_config_content_injects_loopback_no_proxy() {
        let mut env = HashMap::from([(
            "OPENCODE_CONFIG_CONTENT".to_string(),
            "{\"baseUrl\":\"http://127.0.0.1:0\"}".to_string(),
        )]);
        patch_opencode_config_content(&mut env, 24860);

        let no_proxy = env.get("NO_PROXY").expect("NO_PROXY should be set");
        assert!(no_proxy.contains("127.0.0.1"));
        assert!(no_proxy.contains("localhost"));
    }
}
