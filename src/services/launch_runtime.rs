use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::amp_trust::{find_workspace_amp_settings, read_amp_settings_file};
use crate::services::codex_home_shadow::{AuthDotJson, CodexHomeShadow, tokens_changed};
use crate::services::codex_oauth::{CodexOAuthCredential, REFRESH_SKEW_SECS, ensure_fresh};
use crate::services::gemini_home_shadow::GeminiHomeShadow;
use crate::services::gemini_oauth::{
    GeminiOAuthCredential, REFRESH_SKEW_SECS as GEMINI_REFRESH_SKEW_SECS,
    ensure_fresh as gemini_ensure_fresh,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, SessionStore,
};
use crate::services::symlink_util::{symlink_dir, symlink_file};

/// Holds the shadow `CODEX_HOME` dir + metadata needed to sync refreshed
/// tokens back into aivo's store after codex exits.
pub(crate) struct CodexOAuthSync {
    pub(crate) key_id: String,
    pub(crate) shadow: CodexHomeShadow,
    pub(crate) original: CodexOAuthCredential,
}

/// Holds the shadow `GEMINI_CLI_HOME` dir + metadata needed to sync
/// refreshed tokens back into aivo's store after gemini exits.
pub(crate) struct GeminiOAuthSync {
    pub(crate) key_id: String,
    pub(crate) shadow: GeminiHomeShadow,
    pub(crate) original: GeminiOAuthCredential,
}

pub(crate) struct LaunchRuntimeState {
    pub(crate) env: HashMap<String, String>,
    /// Env vars to `env_remove` from the child (vs setting via `env`).
    /// Populated by `prepare_runtime_env` from the
    /// `_AIVO_INTERNAL_ENV_UNSET` carrier emitted by the injector for the
    /// Claude OAuth path — see `environment_injector::AIVO_INTERNAL_ENV_UNSET`.
    pub(crate) env_unset: Vec<String>,
    pub(crate) router_protocol: Option<Arc<AtomicU8>>,
    pub(crate) responses_api_support: Option<Arc<AtomicU8>>,
    /// Set to `true` by a router after any non-error upstream response. Read
    /// by `persist_runtime_discoveries` to skip protocol pinning when no
    /// request actually succeeded — prevents bad keys / transient errors from
    /// silently rewriting `claude_protocol` to the wrong value.
    pub(crate) request_succeeded: Option<Arc<AtomicBool>>,
    /// Set to `true` by a router after observing an authoritative upstream
    /// response — a 2xx success or a 4xx with a parseable LLM-API error
    /// envelope. Read by `persist_runtime_discoveries` to persist the
    /// `claude_path_variant` / `gemini_path_variant` even when no 2xx was
    /// seen, so a session that fails semantically still teaches the next
    /// launch which path variant works. Excluded: terminal 401/403/429
    /// (cross-protocol auth-shape ambiguity) and endpoint-missing 404/405.
    pub(crate) saw_authoritative_response: Option<Arc<AtomicBool>>,
    /// Set to `true` by a router after observing a `reasoning_content` semantic
    /// rejection from the upstream. Persisted to the key so subsequent launches
    /// inject `_REQUIRE_REASONING=1` without needing the host in the static
    /// substring list in `ProviderQuirks::for_base_url`.
    pub(crate) learned_requires_reasoning: Option<Arc<AtomicBool>>,
    pub(crate) pi_agent_dir: Option<String>,
    /// Path of the temp amp settings.json the bridge wrote when any
    /// `internal.model` override (per-mode model flags) or `tools.disable`
    /// was active. Removed at launch exit by `cleanup_runtime_artifacts`
    /// so the cache dir doesn't grow.
    pub(crate) amp_settings_path: Option<String>,
    pub(crate) codex_oauth_sync: Option<CodexOAuthSync>,
    pub(crate) gemini_oauth_sync: Option<GeminiOAuthSync>,
    /// Holds the temp dir that backs `GEMINI_CLI_SYSTEM_SETTINGS_PATH`
    /// for non-OAuth gemini launches. Dropping it deletes the settings
    /// override file; must outlive the spawned gemini process.
    #[allow(dead_code)] // kept alive solely for its Drop impl
    pub(crate) gemini_system_settings: Option<tempfile::TempDir>,
}

pub(crate) async fn prepare_runtime_env(
    tool: AIToolType,
    mut env: HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<LaunchRuntimeState> {
    let mut router_protocol = None;
    let mut responses_api_support = None;
    let mut request_succeeded: Option<Arc<AtomicBool>> = None;
    let mut saw_authoritative_response: Option<Arc<AtomicBool>> = None;
    let mut learned_requires_reasoning: Option<Arc<AtomicBool>> = None;

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ROUTER") {
        let port = start_anthropic_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER") {
        let (port, active, success, authoritative, learned) =
            start_anthropic_to_openai_router(&env).await?;
        router_protocol = Some(active);
        request_succeeded = Some(success);
        saw_authoritative_response = Some(authoritative);
        learned_requires_reasoning = Some(learned);
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if tool == AIToolType::Claude && env.contains_key("AIVO_USE_COPILOT_ROUTER") {
        let port = start_copilot_router(&env).await?;
        set_local_base_url(&mut env, "ANTHROPIC_BASE_URL", port);
    }

    if env.contains_key("AIVO_USE_CURSOR_ROUTER") {
        let port = start_cursor_router(&mut env, tool).await?;
        if tool == AIToolType::Pi && env.contains_key("AIVO_PI_MODELS_JSON") {
            // Pi reads its upstream URL from a JSON file, not an env var, so
            // patch the placeholder in AIVO_PI_MODELS_JSON before writing the
            // temp agent dir.
            write_pi_agent_dir(&mut env, Some(port)).await?;
        } else if tool == AIToolType::Opencode && env.contains_key("OPENCODE_CONFIG_CONTENT") {
            // OpenCode reads its upstream from OPENCODE_CONFIG_CONTENT JSON;
            // swap the placeholder URL for the bound cursor-router port.
            patch_opencode_config_content(&mut env, port);
        } else if tool == AIToolType::Amp && env.contains_key("AIVO_AMP_UPSTREAM_BASE_URL") {
            // amp goes through the amp_bridge first; the cursor router sits
            // behind the bridge's translators as the OpenAI-chat upstream.
            env.insert(
                "AIVO_AMP_UPSTREAM_BASE_URL".to_string(),
                format!("http://127.0.0.1:{port}"),
            );
        } else {
            let base_url_env =
                env.remove("AIVO_CURSOR_BASE_URL_ENV")
                    .unwrap_or_else(|| match tool {
                        tool if tool.is_codex_family() => "OPENAI_BASE_URL".to_string(),
                        _ => "ANTHROPIC_BASE_URL".to_string(),
                    });
            set_local_base_url(&mut env, &base_url_env, port);
        }
        // Note: AIVO_USE_CURSOR_ROUTER is left in `env` here on purpose so
        // `build_runtime_args` (called next, after `prepare_runtime_env`
        // returns) can detect cursor mode and route codex through the
        // non-OpenAI catalog/model-passthrough path. The marker is stripped
        // later in `ai_launcher` right before spawn. `AIVO_CURSOR_KEY_SECRET`
        // is already consumed by `start_cursor_router` itself.
    }

    if tool.is_codex_family() && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER") {
        let (port, _active, responses_api, success, authoritative, learned) =
            start_responses_to_chat_router(&env).await?;
        responses_api_support = Some(responses_api);
        request_succeeded = Some(success);
        saw_authoritative_response = Some(authoritative);
        learned_requires_reasoning = Some(learned);
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool.is_codex_family() && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_ROUTER") {
        let (port, active, success, authoritative, learned) = start_gemini_router(&env).await?;
        router_protocol = Some(active);
        request_succeeded = Some(success);
        saw_authoritative_response = Some(authoritative);
        learned_requires_reasoning = Some(learned);
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
        clear_node_proxy_env(&mut env);
    }

    if tool == AIToolType::Gemini && env.contains_key("AIVO_USE_GEMINI_COPILOT_ROUTER") {
        let port = start_gemini_copilot_router(&env).await?;
        set_local_base_url(&mut env, "GOOGLE_GEMINI_BASE_URL", port);
        clear_node_proxy_env(&mut env);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_COPILOT_ROUTER") {
        let port = start_responses_to_chat_copilot_router(&env).await?;
        patch_opencode_config_content(&mut env, port);
    }

    if tool == AIToolType::Opencode && env.contains_key("AIVO_USE_OPENCODE_ROUTER") {
        let (port, _active, _responses_api, _success, _auth, _learned) =
            start_responses_to_chat_router(&env).await?;
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
        let (port, _active, _responses_api, _success, _auth, _learned) =
            start_responses_to_chat_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    if tool == AIToolType::Pi && env.contains_key("AIVO_USE_PI_ROUTER") {
        let (port, _active, _responses_api, _success, _auth, _learned) =
            start_responses_to_chat_router(&env).await?;
        write_pi_agent_dir(&mut env, Some(port)).await?;
    }

    if tool == AIToolType::Amp && env.contains_key("AIVO_USE_AMP_BRIDGE") {
        let port = start_amp_bridge(&mut env).await?;
        env.insert("AMP_URL".to_string(), format!("http://127.0.0.1:{port}"));
        // Real key never reaches Amp; the bridge holds it and forwards.
        env.insert("AMP_API_KEY".to_string(), "aivo-bridge".to_string());
        // Without this, a user's HTTP_PROXY routes amp's localhost call to the
        // bridge through their HTTP proxy (privoxy/Shadowsocks/etc.), which
        // can't reach 127.0.0.1:<port> and returns 500.
        ensure_loopback_no_proxy(&mut env);
    }

    let pi_agent_dir = env.get("PI_CODING_AGENT_DIR").cloned();
    let amp_settings_path = env.get("AIVO_AMP_SETTINGS_FILE").cloned();

    let codex_oauth_sync = if tool.is_codex_family() && env.contains_key("AIVO_CODEX_OAUTH_CREDS") {
        Some(prepare_codex_oauth_shadow(tool, &mut env, session_store).await?)
    } else {
        None
    };

    if tool == AIToolType::CodexApp && codex_oauth_sync.is_none() {
        prepare_codex_app_home_without_auth(&mut env, session_store).await?;
    }

    let gemini_oauth_sync =
        if tool == AIToolType::Gemini && env.contains_key("AIVO_GEMINI_OAUTH_CREDS") {
            Some(prepare_gemini_oauth_shadow(&mut env, session_store).await?)
        } else {
            None
        };

    let gemini_system_settings =
        if tool == AIToolType::Gemini && env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH") {
            Some(prepare_gemini_api_key_settings_override(&mut env).await?)
        } else {
            None
        };

    let env_unset = env
        .remove(crate::services::environment_injector::AIVO_INTERNAL_ENV_UNSET)
        .map(|v| {
            v.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    Ok(LaunchRuntimeState {
        env,
        env_unset,
        router_protocol,
        responses_api_support,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
        pi_agent_dir,
        amp_settings_path,
        codex_oauth_sync,
        gemini_oauth_sync,
        gemini_system_settings,
    })
}

/// Parses `AIVO_CODEX_OAUTH_CREDS` (set by `environment_injector::for_codex`
/// for ChatGPT OAuth keys), refreshes the access token if near expiry, and
/// writes a shadow `CODEX_HOME` temp dir containing a native `auth.json`.
///
/// The placeholder env vars are stripped before codex is spawned; all codex
/// sees is `CODEX_HOME=<shadow>` and the standard model overrides.
async fn prepare_codex_oauth_shadow(
    tool: AIToolType,
    env: &mut HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<CodexOAuthSync> {
    let raw = env
        .remove("AIVO_CODEX_OAUTH_CREDS")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_OAUTH_CREDS"))?;
    let key_id = env
        .remove("AIVO_CODEX_KEY_ID")
        .or_else(|| env.remove("AIVO_CODEX_APP_HOME_KEY"))
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_KEY_ID"))?;
    env.remove("AIVO_CODEX_APP_HOME_KEY");
    let mut creds = CodexOAuthCredential::from_json(&raw)?;

    if tool == AIToolType::CodexApp {
        adopt_persistent_codex_app_tokens(session_store, &key_id, &mut creds).await;
    }

    // Refresh pre-launch so codex starts with a valid access token. If the
    // refresh token has been invalidated — codex was killed before the
    // post-exit sync ran, or another process (native codex, a parallel aivo)
    // rotated it — fall back to an interactive re-login and persist the new
    // creds immediately so the next launch doesn't repeat the recovery.
    if let Err(e) = ensure_fresh(&mut creds, REFRESH_SKEW_SECS).await {
        if !is_oauth_invalid_grant(&e) {
            return Err(e);
        }
        eprintln!(
            "{} Codex refresh token is no longer valid — re-authenticating.",
            crate::style::yellow("aivo:")
        );
        let stale = creds.clone();
        creds = crate::services::codex_oauth::interactive_login()
            .await
            .map_err(|err| err.context("codex re-login after invalid refresh token"))?;
        persist_refreshed_if_needed(session_store, &key_id, &stale, &creds).await;
    }

    let shadow = if tool == AIToolType::CodexApp {
        CodexHomeShadow::create_persistent(&creds, session_store.config_dir(), &key_id).await?
    } else {
        CodexHomeShadow::create(&creds).await?
    };
    env.insert(
        "CODEX_HOME".to_string(),
        shadow.path().to_string_lossy().to_string(),
    );

    Ok(CodexOAuthSync {
        key_id,
        shadow,
        original: creds,
    })
}

async fn prepare_codex_app_home_without_auth(
    env: &mut HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<()> {
    let key_id = env
        .remove("AIVO_CODEX_APP_HOME_KEY")
        .unwrap_or_else(|| "default".to_string());
    let shadow =
        CodexHomeShadow::create_persistent_without_auth(session_store.config_dir(), &key_id)
            .await?;
    env.insert(
        "CODEX_HOME".to_string(),
        shadow.path().to_string_lossy().to_string(),
    );
    Ok(())
}

async fn adopt_persistent_codex_app_tokens(
    session_store: &SessionStore,
    key_id: &str,
    creds: &mut CodexOAuthCredential,
) {
    let auth_path =
        CodexHomeShadow::persistent_path(session_store.config_dir(), key_id).join("auth.json");
    let disk = match CodexHomeShadow::read_auth_path(auth_path).await {
        Ok(Some(disk)) => disk,
        _ => return,
    };
    if !tokens_changed(creds, &disk) {
        return;
    }
    let updated = auth_dot_json_into_credential(disk, creds);
    let original = creds.clone();
    persist_refreshed_if_needed(session_store, key_id, &original, &updated).await;
    *creds = updated;
}

/// Reads the shadow `auth.json` back after codex exits and, if any token
/// changed, persists the rotated credential into aivo's store. Errors are
/// logged but never propagated — the user's codex session has already
/// completed, and a failed sync just means the next launch will refresh
/// again.
pub(crate) async fn finalize_codex_oauth(
    session_store: &SessionStore,
    sync: Option<CodexOAuthSync>,
) {
    let Some(sync) = sync else {
        return;
    };

    let disk = match sync.shadow.read_back().await {
        Ok(Some(v)) => v,
        Ok(None) => {
            // File missing/truncated — codex probably crashed. Keep the
            // pre-launch credential intact. (It was refreshed before
            // launch, so the refresh_token is already up-to-date on disk.)
            persist_refreshed_if_needed(
                session_store,
                &sync.key_id,
                &sync.original,
                &sync.original,
            )
            .await;
            return;
        }
        Err(_) => return,
    };

    let updated = auth_dot_json_into_credential(disk, &sync.original);
    persist_refreshed_if_needed(session_store, &sync.key_id, &sync.original, &updated).await;
}

fn auth_dot_json_into_credential(
    disk: AuthDotJson,
    original: &CodexOAuthCredential,
) -> CodexOAuthCredential {
    disk.into_credential(original.email.clone(), original.expires_at)
}

async fn persist_refreshed_if_needed(
    session_store: &SessionStore,
    key_id: &str,
    original: &CodexOAuthCredential,
    updated: &CodexOAuthCredential,
) {
    if original == updated {
        return;
    }
    let json = match updated.to_json() {
        Ok(j) => j,
        Err(_) => return,
    };
    // base_url / name / protocols are preserved by passing the same values
    // as the existing entry. Pull the current entry first so we don't
    // clobber name changes made mid-session.
    if let Ok(Some(existing)) = session_store.get_key_by_id(key_id).await {
        let _ = session_store
            .update_key(
                key_id,
                &existing.name,
                &existing.base_url,
                existing.claude_protocol,
                &json,
            )
            .await;
    }
}

/// Convenience for the crash-path: delegates to `tokens_changed` via the
/// read-back value. Exposed so tests don't need to touch disk.
#[allow(dead_code)]
pub(crate) fn detect_token_rotation(
    original: &CodexOAuthCredential,
    disk: &crate::services::codex_home_shadow::AuthDotJson,
) -> bool {
    tokens_changed(original, disk)
}

/// Heuristic for "the refresh server told us our refresh token is bad" vs
/// "transient failure". 4xx from the token endpoint (`invalid_grant`,
/// `invalid_request_error`) is recoverable via an interactive re-login;
/// 5xx and network errors are not. Both `codex_oauth::refresh` and
/// `gemini_oauth::refresh` format their failures as
/// `"refresh failed (NNN): <body>"`, so a substring match is enough.
pub(crate) fn is_oauth_invalid_grant(e: &anyhow::Error) -> bool {
    let s = e.to_string();
    s.contains("refresh failed (400")
        || s.contains("refresh failed (401")
        || s.contains("refresh failed (403")
        || s.contains("refresh failed (404")
}

/// Parses `AIVO_GEMINI_OAUTH_CREDS` (set by `environment_injector::for_gemini`
/// for Google OAuth keys), refreshes the access token if near expiry, and
/// writes a shadow `GEMINI_CLI_HOME` temp dir containing `.gemini/
/// oauth_creds.json` + `google_accounts.json`.
///
/// The `AIVO_*` placeholder vars are stripped before gemini is spawned; all
/// gemini sees is `GEMINI_CLI_HOME=<shadow>`, `GOOGLE_GENAI_USE_GCA=true`,
/// and `GEMINI_MODEL=<model>`.
async fn prepare_gemini_oauth_shadow(
    env: &mut HashMap<String, String>,
    session_store: &SessionStore,
) -> Result<GeminiOAuthSync> {
    let raw = env
        .remove("AIVO_GEMINI_OAUTH_CREDS")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_GEMINI_OAUTH_CREDS"))?;
    let key_id = env
        .remove("AIVO_GEMINI_KEY_ID")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_GEMINI_KEY_ID"))?;
    let mut creds = GeminiOAuthCredential::from_json(&raw)?;

    // Refresh pre-launch so gemini starts with a valid access token. As with
    // codex (see `prepare_codex_oauth_shadow`), an invalid_grant from Google
    // means the stored refresh token is dead — drop into the OAuth flow
    // instead of bubbling a confusing 401 to the user.
    if let Err(e) = gemini_ensure_fresh(&mut creds, GEMINI_REFRESH_SKEW_SECS).await {
        if !is_oauth_invalid_grant(&e) {
            return Err(e);
        }
        eprintln!(
            "{} Gemini refresh token is no longer valid — re-authenticating.",
            crate::style::yellow("aivo:")
        );
        let stale = creds.clone();
        creds = crate::services::gemini_oauth::interactive_login()
            .await
            .map_err(|err| err.context("gemini re-login after invalid refresh token"))?;
        persist_refreshed_gemini_if_needed(session_store, &key_id, &stale, &creds).await;
    }

    let shadow = GeminiHomeShadow::create(&creds).await?;
    env.insert(
        "GEMINI_CLI_HOME".to_string(),
        shadow.path().to_string_lossy().to_string(),
    );

    Ok(GeminiOAuthSync {
        key_id,
        shadow,
        original: creds,
    })
}

/// Writes a gemini-cli *system-scope* settings file containing just
/// `security.auth.selectedType = "gemini-api-key"` and points
/// `GEMINI_CLI_SYSTEM_SETTINGS_PATH` at it. The CLI merges system over
/// user over defaults, so this forces the `USE_GEMINI` auth path (which
/// honors `GEMINI_API_KEY` + `GOOGLE_GEMINI_BASE_URL`) even when the
/// user's `~/.gemini/settings.json` has a stale `oauth-personal`
/// selection from a prior Google login. Without this,
/// `configuredAuthType || getAuthTypeFromEnv()` in gemini-cli returns
/// `LOGIN_WITH_GOOGLE` and every request bypasses aivo's router.
///
/// The user's real settings file is never read, copied, or written by
/// aivo; in-session edits (theme, vim mode, MCP server tweaks, model
/// preferences) persist to `~/.gemini/` as usual. Auto-fallbacks via
/// `activateFallbackMode` use `isTemporary=true` in gemini-cli and so
/// skip the user-scope `model.name` write, which is the only automatic
/// write path that could leak an aivo-injected model back into the
/// user's defaults.
async fn prepare_gemini_api_key_settings_override(
    env: &mut HashMap<String, String>,
) -> Result<tempfile::TempDir> {
    use anyhow::Context;
    env.remove("AIVO_GEMINI_FORCE_API_KEY_AUTH");
    let model_config_model = env.remove("AIVO_GEMINI_MODEL_CONFIG_MODEL");

    let dir = tempfile::Builder::new()
        .prefix("aivo-gemini-settings-")
        .tempdir()
        .context("create aivo gemini settings override temp dir")?;
    let path = dir.path().join("settings.json");
    let mut settings = serde_json::json!({
        "security": {
            "auth": {
                "selectedType": "gemini-api-key"
            }
        }
    });
    if let Some(model) = model_config_model.filter(|m| !m.trim().is_empty()) {
        settings["modelConfigs"] = gemini_internal_model_config_override(&model);
    }
    tokio::fs::write(&path, serde_json::to_vec(&settings)?)
        .await
        .context("write aivo gemini system settings override")?;

    env.insert(
        "GEMINI_CLI_SYSTEM_SETTINGS_PATH".to_string(),
        path.to_string_lossy().to_string(),
    );
    Ok(dir)
}

fn gemini_internal_model_config_override(model: &str) -> serde_json::Value {
    let aliases = [
        // Gemini CLI 0.40.x uses these helper aliases for routing,
        // completion, edit correction and summarization. Several default to
        // gemini-2.5-flash-lite, which may be unavailable or irrelevant when
        // aivo is routing the CLI to a non-Google provider.
        "gemini-2.5-flash-lite",
        "classifier",
        "prompt-completion",
        "edit-corrector",
        "summarizer-default",
        "summarizer-shell",
        "chat-compression-2.5-flash-lite",
    ];
    let custom_aliases = aliases
        .into_iter()
        .map(|alias| {
            (
                alias.to_string(),
                serde_json::json!({
                    "modelConfig": {
                        "model": model
                    }
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    serde_json::json!({
        "customAliases": custom_aliases
    })
}

/// Reads the shadow `oauth_creds.json` back after gemini exits and, if any
/// token changed, persists the rotated credential into aivo's store.
/// Errors are logged but never propagated — the user's gemini session has
/// already completed, and a failed sync just means the next launch refreshes
/// again.
pub(crate) async fn finalize_gemini_oauth(
    session_store: &SessionStore,
    sync: Option<GeminiOAuthSync>,
) {
    let Some(sync) = sync else {
        return;
    };

    let disk = match sync.shadow.read_back().await {
        Ok(Some(v)) => v,
        Ok(None) => {
            // File missing/truncated — gemini probably crashed before
            // writing. Persist the pre-launch (freshly refreshed) creds so
            // the refresh_token rotation isn't lost.
            persist_refreshed_gemini_if_needed(
                session_store,
                &sync.key_id,
                &sync.original,
                &sync.original,
            )
            .await;
            return;
        }
        Err(_) => return,
    };

    let updated = disk.into_credential(sync.original.email.clone(), sync.original.last_refresh);
    persist_refreshed_gemini_if_needed(session_store, &sync.key_id, &sync.original, &updated).await;
}

async fn persist_refreshed_gemini_if_needed(
    session_store: &SessionStore,
    key_id: &str,
    original: &GeminiOAuthCredential,
    updated: &GeminiOAuthCredential,
) {
    if original == updated {
        return;
    }
    let json = match updated.to_json() {
        Ok(j) => j,
        Err(_) => return,
    };
    if let Ok(Some(existing)) = session_store.get_key_by_id(key_id).await {
        let _ = session_store
            .update_key(
                key_id,
                &existing.name,
                &existing.base_url,
                existing.claude_protocol,
                &json,
            )
            .await;
    }
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

/// Apply one-shot migrations to routing-related fields on `key`.
///
/// Called early in the launch flow (before any router reads
/// `responses_api_supported` etc.) so that older fields written under buggy
/// logic don't keep poisoning new launches.
///
/// Mutates `key` in place when fields change, and persists the same change to
/// the session store. Failures to persist are logged-and-ignored: the
/// in-memory mutation still benefits this launch.
///
/// Migration history:
/// - **v1**: pre-fix builds latched `responses_api_supported = Some(false)` on
///   any non-200 (incl. transient 429/5xx). Clear it so the next request
///   re-probes under the new endpoint-missing-only rule.
pub(crate) async fn migrate_routing_schema_for_key(session_store: &SessionStore, key: &mut ApiKey) {
    use crate::services::session_store::CURRENT_ROUTING_SCHEMA_VERSION;
    if key.routing_schema_version >= CURRENT_ROUTING_SCHEMA_VERSION {
        return;
    }

    if key.routing_schema_version < 1 && key.responses_api_supported == Some(false) {
        key.responses_api_supported = None;
        let _ = session_store
            .set_key_responses_api_supported(&key.id, None)
            .await;
    }

    key.routing_schema_version = CURRENT_ROUTING_SCHEMA_VERSION;
    let _ = session_store
        .set_key_routing_schema_version(&key.id, CURRENT_ROUTING_SCHEMA_VERSION)
        .await;
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn persist_runtime_discoveries(
    session_store: &SessionStore,
    tool: AIToolType,
    key: &ApiKey,
    router_protocol: Option<Arc<AtomicU8>>,
    responses_api_support: Option<Arc<AtomicU8>>,
    request_succeeded: Option<Arc<AtomicBool>>,
    saw_authoritative_response: Option<Arc<AtomicBool>>,
    learned_requires_reasoning: Option<Arc<AtomicBool>>,
) {
    // Persist a learned `requires_reasoning_content` quirk regardless of the
    // success gate below: the upstream's *parseable* error envelope is itself
    // proof the protocol matches and the quirk is real, even if no 2xx
    // response was ever seen this session. Without this, a key with a strict
    // thinking-mode upstream that fails on first launch never learns and
    // re-cascades on every subsequent launch.
    if let Some(flag) = learned_requires_reasoning.as_ref()
        && flag.load(Ordering::Relaxed)
        && key.requires_reasoning_content != Some(true)
    {
        let _ = session_store
            .set_key_requires_reasoning_content(&key.id, Some(true))
            .await;
    }

    let saw_success = request_succeeded
        .as_ref()
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(true);
    let saw_authoritative = saw_authoritative_response
        .as_ref()
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false);

    // Persist path-variant alone when the cascade observed an authoritative
    // response (2xx OR a 4xx with a parseable LLM-API error envelope) but no
    // 2xx ever happened. The path responded — we can confidently remember
    // which variant won, even though the request body was rejected. Protocol
    // pinning still requires real success below: a 401/403 from a
    // cross-protocol gateway shouldn't be enough to rewrite `claude_protocol`.
    if !saw_success
        && saw_authoritative
        && let Some(active) = router_protocol.as_ref()
    {
        let (_, final_variant) =
            crate::services::provider_protocol::decode_route(active.load(Ordering::Relaxed));
        let final_variant_str = final_variant.as_str().to_string();
        match tool {
            AIToolType::Claude => {
                let current = key.claude_path_variant.as_deref().unwrap_or("default");
                if final_variant_str != current {
                    let _ = session_store
                        .set_key_claude_path_variant(&key.id, Some(final_variant_str))
                        .await;
                }
            }
            AIToolType::Gemini => {
                let current = key.gemini_path_variant.as_deref().unwrap_or("default");
                if final_variant_str != current {
                    let _ = session_store
                        .set_key_gemini_path_variant(&key.id, Some(final_variant_str))
                        .await;
                }
            }
            _ => {}
        }
    }

    // Gate protocol/responses-api persistence on at least one successful
    // upstream response. Without this, a session that only saw failures (bad
    // API key, transient 5xx, rate limits) could silently rewrite the
    // persisted protocol to whatever the runtime *guessed* — and lock the
    // user into a permanently broken configuration.
    if !saw_success {
        return;
    }

    if let Some(active) = router_protocol {
        let (final_protocol, final_variant) =
            crate::services::provider_protocol::decode_route(active.load(Ordering::Relaxed));
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
                // Persist the path variant separately so stripped-path wins
                // are remembered across launches.
                let final_variant_str = final_variant.as_str().to_string();
                let current_variant = key.claude_path_variant.as_deref().unwrap_or("default");
                if final_variant_str != current_variant {
                    let _ = session_store
                        .set_key_claude_path_variant(&key.id, Some(final_variant_str))
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
                let final_variant_str = final_variant.as_str().to_string();
                let current_variant = key.gemini_path_variant.as_deref().unwrap_or("default");
                if final_variant_str != current_variant {
                    let _ = session_store
                        .set_key_gemini_path_variant(&key.id, Some(final_variant_str))
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

    let Some(real_sessions) = real_sessions else {
        return;
    };

    if pi_sessions_share_storage(&temp_sessions, &real_sessions).await {
        return;
    }

    copy_pi_session_jsonl_tree(&temp_sessions, &real_sessions).await;
}

pub(crate) async fn cleanup_runtime_artifacts(
    codex_model_catalog_path: Option<&str>,
    pi_agent_dir: Option<&str>,
    amp_settings_path: Option<&str>,
) {
    if let Some(path) = codex_model_catalog_path {
        let _ = tokio::fs::remove_file(path).await;
    }
    if let Some(dir) = pi_agent_dir {
        let _ = tokio::fs::remove_dir_all(dir).await;
    }
    if let Some(path) = amp_settings_path {
        // Diagnostic escape hatch: when `AIVO_KEEP_AMP_SETTINGS=1`, leave
        // the merged settings file behind so users can inspect what aivo
        // actually handed amp. Useful when debugging tools.disable /
        // internal.model overrides that don't seem to take effect.
        if std::env::var("AIVO_KEEP_AMP_SETTINGS").as_deref() == Ok("1") {
            eprintln!("aivo: AIVO_KEEP_AMP_SETTINGS=1 — kept settings file at {path}");
        } else {
            let _ = tokio::fs::remove_file(path).await;
        }
    }
}

/// Writes a temporary `PI_CODING_AGENT_DIR` that surfaces the user's real
/// `~/.pi/agent/` customization (packages, MCP servers, rules, themes,
/// settings, auth) while pinning the provider to the aivo entry.
///
/// `models.json` is aivo-only so an explicit `--model anthropic/foo`
/// against an aivo launch errors with "unknown provider" instead of
/// silently routing through the user's real key. Everything else is
/// symlinked so mid-session writes (package installs, login flows)
/// persist back to the real home.
///
/// When `port` is `Some`, the placeholder `PLACEHOLDER_LOOPBACK_URL` in
/// `AIVO_PI_MODELS_JSON` is patched with the real router port.
/// When `port` is `None`, the JSON already contains the real upstream URL.
async fn write_pi_agent_dir(env: &mut HashMap<String, String>, port: Option<u16>) -> Result<()> {
    let real_agent = crate::services::system_env::home_dir().map(|h| h.join(".pi").join("agent"));
    write_pi_agent_dir_with_real(env, port, real_agent.as_deref()).await
}

async fn write_pi_agent_dir_with_real(
    env: &mut HashMap<String, String>,
    port: Option<u16>,
    real_agent: Option<&Path>,
) -> Result<()> {
    let raw = env
        .get("AIVO_PI_MODELS_JSON")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_PI_MODELS_JSON"))?
        .clone();

    let aivo_models_json = match port {
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
        tokio::fs::write(dir.join("models.json"), aivo_models_json.as_bytes()),
        link_or_default(real_agent, "settings.json", &dir),
        link_or_default(real_agent, "auth.json", &dir),
    )?;

    if let Some(real_agent) = real_agent {
        link_pi_agent_state(real_agent, &dir).await;
        populate_pi_bin_dir(&real_agent.join("bin"), &dir.join("bin")).await;
        seed_pi_sessions(Some(real_agent.join("sessions")), &dir).await;
    } else {
        seed_pi_sessions(None, &dir).await;
    }

    env.insert(
        "PI_CODING_AGENT_DIR".to_string(),
        dir.to_string_lossy().to_string(),
    );
    Ok(())
}

/// Tries symlink → hard-link → copy so writes propagate back to `real`
/// when possible. NTFS hard links work without Developer Mode and still
/// propagate writes; copy is the last-resort read-only path. Returns
/// `Err` only if all three fail (e.g., `real` became unreadable).
async fn link_existing_file(real: &Path, dest: &Path) -> std::io::Result<()> {
    if symlink_file(real, dest).await.is_ok() {
        return Ok(());
    }
    if tokio::fs::hard_link(real, dest).await.is_ok() {
        return Ok(());
    }
    let bytes = tokio::fs::read(real).await?;
    tokio::fs::write(dest, bytes).await
}

/// Touches the real file with `{}` if missing, then links it into
/// `dest` via `link_existing_file`. Used for files Pi expects to find
/// (`settings.json`, `auth.json`) — the default keeps Pi happy even on
/// a fresh `~/.pi/agent/`.
async fn link_or_default(
    real_agent: Option<&Path>,
    name: &str,
    dest: &Path,
) -> std::io::Result<()> {
    let dest_path = dest.join(name);
    let Some(real) = real_agent else {
        return tokio::fs::write(&dest_path, "{}").await;
    };
    let real_path = real.join(name);

    if !real_path.is_file() {
        if let Some(parent) = real_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        let _ = tokio::fs::write(&real_path, "{}").await;
    }

    if link_existing_file(&real_path, &dest_path).await.is_ok() {
        return Ok(());
    }
    // Last-resort: real became unreadable between is_file() check and copy.
    tokio::fs::write(&dest_path, b"{}").await
}

/// Symlinks the user's mutable `~/.pi/agent/` state (rules, tools,
/// prompts, themes, git-sourced packages, mcp.json) into the temp dir.
/// Dirs are pre-created so first-time `pi install <git-pkg>` lands in
/// the real home instead of vanishing with the temp dir. `mcp.json`
/// goes through the same symlink → hard-link → copy chain as the other
/// linked files so MCP servers stay reachable on Windows without
/// Developer Mode. On Windows without Developer Mode dir symlinks fail;
/// in-session writes to those dirs are then lost — same gap as
/// `codex_home_shadow`.
async fn link_pi_agent_state(real_agent: &Path, dest: &Path) {
    for d in ["rules", "tools", "prompts", "themes", "git"] {
        let real = real_agent.join(d);
        if tokio::fs::create_dir_all(&real).await.is_ok() {
            let _ = symlink_dir(&real, &dest.join(d)).await;
        }
    }

    let mcp = real_agent.join("mcp.json");
    if mcp.is_file() {
        let _ = link_existing_file(&mcp, &dest.join("mcp.json")).await;
    }
}

/// Populate `dest_bin` with pi's managed binaries from `real_bin`. Best
/// effort: any single failure is silently skipped so pi falls back to
/// re-downloading just that binary.
#[cfg(unix)]
async fn populate_pi_bin_dir(real_bin: &std::path::Path, dest_bin: &std::path::Path) {
    // A single symlink covers the whole directory — cheap and keeps pi's
    // post-launch writes (if any) pointing at the managed copy.
    let _ = tokio::fs::symlink(real_bin, dest_bin).await;
}

#[cfg(windows)]
async fn populate_pi_bin_dir(real_bin: &std::path::Path, dest_bin: &std::path::Path) {
    // Windows symlinks / junctions need elevation or developer mode; fall
    // back to per-file hard links, then copies. Works for the common case
    // where HOME and the temp dir share a filesystem.
    if tokio::fs::create_dir_all(dest_bin).await.is_err() {
        return;
    }
    let mut entries = match tokio::fs::read_dir(real_bin).await {
        Ok(rd) => rd,
        Err(_) => return,
    };
    while let Ok(Some(entry)) = entries.next_entry().await {
        let src = entry.path();
        let Some(name) = src.file_name() else {
            continue;
        };
        let dst = dest_bin.join(name);
        if tokio::fs::hard_link(&src, &dst).await.is_ok() {
            continue;
        }
        // hard_link fails across filesystems or on directories; try a
        // plain copy for regular files before giving up. entry.file_type()
        // reuses readdir metadata so we skip a redundant stat syscall.
        if let Ok(ft) = entry.file_type().await
            && ft.is_file()
        {
            let _ = tokio::fs::copy(&src, &dst).await;
        }
    }
}

#[cfg(not(any(unix, windows)))]
async fn populate_pi_bin_dir(_real_bin: &std::path::Path, _dest_bin: &std::path::Path) {}

async fn seed_pi_sessions(
    real_sessions: Option<std::path::PathBuf>,
    temp_agent_dir: &std::path::Path,
) {
    let temp_sessions = temp_agent_dir.join("sessions");

    let Some(real_sessions) = real_sessions else {
        let _ = tokio::fs::create_dir_all(&temp_sessions).await;
        return;
    };

    if tokio::fs::create_dir_all(&real_sessions).await.is_err() {
        let _ = tokio::fs::create_dir_all(&temp_sessions).await;
        return;
    }

    if link_pi_sessions_dir(&real_sessions, &temp_sessions).await {
        return;
    }

    copy_pi_session_jsonl_tree(&real_sessions, &temp_sessions).await;
}

#[cfg(unix)]
async fn link_pi_sessions_dir(
    real_sessions: &std::path::Path,
    temp_sessions: &std::path::Path,
) -> bool {
    tokio::fs::symlink(real_sessions, temp_sessions)
        .await
        .is_ok()
}

#[cfg(not(unix))]
async fn link_pi_sessions_dir(
    _real_sessions: &std::path::Path,
    _temp_sessions: &std::path::Path,
) -> bool {
    false
}

async fn pi_sessions_share_storage(
    temp_sessions: &std::path::Path,
    real_sessions: &std::path::Path,
) -> bool {
    let (Ok(temp), Ok(real)) = (
        tokio::fs::canonicalize(temp_sessions).await,
        tokio::fs::canonicalize(real_sessions).await,
    ) else {
        return false;
    };
    temp == real
}

async fn copy_pi_session_jsonl_tree(src_root: &std::path::Path, dst_root: &std::path::Path) {
    let mut dirs = vec![src_root.to_path_buf()];
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
            let Ok(rel) = path.strip_prefix(src_root) else {
                continue;
            };
            let dest = dst_root.join(rel);
            if let Some(parent) = dest.parent() {
                let _ = tokio::fs::create_dir_all(parent).await;
            }
            let _ = tokio::fs::copy(&path, &dest).await;
        }
    }
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

/// Clears every HTTP proxy env var the spawned gemini might honor, in
/// both casings. Needed because gemini reads them, installs a global
/// undici `ProxyAgent`, and from then on every `fetch` — including to
/// our `http://127.0.0.1:<port>` router — is sent through the proxy
/// regardless of `NO_PROXY`. Setting the vars to empty strings (rather
/// than removing them) is enough because the lookups use `||` chains
/// that treat `""` as falsy.
///
/// `ALL_PROXY` isn't on gemini-cli's primary lookup path, but the
/// bundled `proxy-from-env` library and various sub-deps (gaxios,
/// googleapis) do consult it, so we clear it too for defense in depth.
fn clear_node_proxy_env(env: &mut HashMap<String, String>) {
    for var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        env.insert(var.to_string(), String::new());
    }
}

/// Ensures the spawned subprocess will bypass any HTTP proxy when talking to
/// the local loopback router. Without this, a user's `HTTP_PROXY`/`HTTPS_PROXY`
/// would route the subprocess's `http://127.0.0.1:<port>` request through the
/// proxy, which cannot reach the user's localhost. We append loopback entries
/// to both the upper- and lower-case variants since different HTTP libraries
/// check different casings.
fn ensure_loopback_no_proxy(env: &mut HashMap<String, String>) {
    for var in NO_PROXY_VAR_NAMES {
        let existing = env.get(*var).cloned().unwrap_or_default();
        env.insert((*var).to_string(), merge_loopback_entries(&existing));
    }
}

/// Same as `ensure_loopback_no_proxy` but mutates the current process env.
///
/// SAFETY: aivo's tokio runtime is `current_thread` (see `main.rs`), so async
/// work shares one OS thread and concurrent env reads can't race. Must not
/// run inside or after a `spawn_blocking` on the env vars being modified —
/// the blocking pool's reads would race the write.
pub(crate) fn ensure_loopback_no_proxy_in_process_env() {
    // SAFETY: see fn-level comment.
    unsafe {
        for var in NO_PROXY_VAR_NAMES {
            let existing = std::env::var(var).unwrap_or_default();
            std::env::set_var(var, merge_loopback_entries(&existing));
        }
    }
}

const NO_PROXY_VAR_NAMES: &[&str] = &["NO_PROXY", "no_proxy"];
const LOOPBACK_HOSTS: &[&str] = &["localhost", "127.0.0.1", "::1"];

/// Merges loopback hosts into a comma-separated NO_PROXY value, deduping
/// case-insensitively so a pre-existing `LOCALHOST` or `127.0.0.1` entry
/// doesn't get duplicated.
fn merge_loopback_entries(existing: &str) -> String {
    let mut entries: Vec<String> = existing
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    for host in LOOPBACK_HOSTS {
        if !entries.iter().any(|e| e.eq_ignore_ascii_case(host)) {
            entries.push((*host).to_string());
        }
    }
    entries.join(",")
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
) -> Result<(
    u16,
    Arc<AtomicU8>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
)> {
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
    let target_path_variant = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_PATH_VARIANT")
        .and_then(|s| crate::services::provider_protocol::PathVariant::parse(s));
    let strip_cache_control = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_STRIP_CACHE_CONTROL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let config = AnthropicToOpenAIRouterConfig {
        target_base_url: base_url,
        target_api_key: api_key,
        target_protocol,
        target_path_variant,
        strip_cache_control,
        model_prefix,
        requires_reasoning_content,
        max_tokens_cap,
        is_starter: env
            .get("AIVO_IS_STARTER")
            .map(|v| v == "1")
            .unwrap_or(false),
    };

    let router = AnthropicToOpenAIRouter::new(config);
    let (
        port,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
        handle,
    ) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: anthropic-to-openai router exited unexpectedly: {e}");
        }
    });
    Ok((
        port,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
    ))
}

async fn start_responses_to_chat_router(
    env: &HashMap<String, String>,
) -> Result<(
    u16,
    Arc<AtomicU8>,
    Arc<AtomicU8>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
)> {
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

    let aivo_prefix_models = env
        .get("AIVO_RESPONSES_TO_CHAT_ROUTER_AIVO_PREFIX_MODELS")
        .map(|v| {
            v.split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        target_path_variant: None,
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
        aivo_prefix_models,
    });
    let (
        port,
        active_protocol,
        responses_api,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
        handle,
    ) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat router exited unexpectedly: {e}");
        }
    });
    Ok((
        port,
        active_protocol,
        responses_api,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
    ))
}

async fn start_gemini_router(
    env: &HashMap<String, String>,
) -> Result<(
    u16,
    Arc<AtomicU8>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
    Arc<AtomicBool>,
)> {
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
    let (
        port,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
        handle,
    ) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: gemini router exited unexpectedly: {e}");
        }
    });
    Ok((
        port,
        active_protocol,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
    ))
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
    let (
        port,
        _active_protocol,
        _request_succeeded,
        _saw_authoritative_response,
        _learned_requires_reasoning,
        handle,
    ) = router.start_background().await?;
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

async fn start_cursor_router(env: &mut HashMap<String, String>, tool: AIToolType) -> Result<u16> {
    use crate::services::cursor_acp::{self, CURSOR_ACP_SENTINEL};
    use crate::services::cursor_bridge::mcp::ToolUseIdStyle;
    use crate::services::cursor_bridge::{CursorModelRouter, CursorRouterConfig};
    use crate::services::session_store::ApiKey;
    use zeroize::Zeroizing;

    let key_secret = env.remove("AIVO_CURSOR_KEY_SECRET").ok_or_else(|| {
        anyhow::anyhow!(
            "Missing AIVO_CURSOR_KEY_SECRET; re-run `aivo keys add cursor` to set up an isolated cursor account."
        )
    })?;

    if cursor_acp::is_legacy_cursor_login_secret(&key_secret) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Remove it (`aivo keys rm cursor`) and re-add (`aivo keys add cursor`) so cursor-agent runs in its own isolated home."
        );
    }

    let key = ApiKey {
        id: "cursor-router".to_string(),
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
        key: Zeroizing::new(key_secret),
        created_at: String::new(),
    };

    // Bail before spawning the router when the saved OAuth shadow has
    // been signed out. `cursor-agent status` only inspects auth.json, so
    // API-key shadows always report "unauthenticated" — skip the check
    // for those and let the first /v1/models request surface a real
    // upstream error instead.
    if let Some(parsed) = cursor_acp::parse_cursor_shadow_secret(key.key.as_str())
        && parsed.api_key.is_none()
        && !cursor_acp::cursor_status_authenticated_for_key(&key)
            .await
            .unwrap_or(false)
    {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth <id>` (or pick `aivo keys reauth` interactively) to sign in again."
        );
    }
    let workspace_cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(String::from))
        .unwrap_or_else(|| ".".to_string());

    // Every launched tool that targets cursor sends `stream:true` + a tools
    // array on every real turn, so they all flow through
    // `run_*_bridged_fresh`, which opens its own session with the MCP bridge
    // attached at `session/new` time. MCP-less pool prewarms were dead
    // weight — observed in debug captures opening two cursor-agent processes
    // (Claude `prewarm_count=2`) that the bridged path never consumed.
    // Replace the pool prewarm with a single MCP-attached prewarm keyed to
    // the protocol's id-style; the first bridged turn picks it up via
    // `take_mcp_prewarmed`. Claude Code's main+subagent paired burst still
    // works: title-gen subagents short-circuit before hitting cursor, and
    // genuine subagent dispatch is internal to cursor-agent (we only see one
    // `/v1/messages` per main turn).
    let prewarm_count = 0;
    let mcp_prewarm_id_style = Some(match tool {
        AIToolType::Claude => ToolUseIdStyle::Anthropic,
        AIToolType::Gemini => ToolUseIdStyle::Gemini,
        AIToolType::Codex
        | AIToolType::CodexApp
        | AIToolType::Opencode
        | AIToolType::Pi
        | AIToolType::Amp => ToolUseIdStyle::OpenAi,
    });
    let router = CursorModelRouter::new(CursorRouterConfig {
        key,
        workspace_cwd,
        models_cache: Some(crate::services::models_cache::ModelsCache::new()),
        prewarm_count,
        mcp_prewarm_id_style,
    });
    let (port, handle) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: cursor router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Spawns the Amp bridge on a random local port. Strips the `AIVO_USE_AMP_BRIDGE`
/// scaffolding env vars so they don't leak into the spawned amp child.
async fn start_amp_bridge(env: &mut HashMap<String, String>) -> Result<u16> {
    use crate::services::amp_bridge::{AmpBridge, AmpBridgeConfig};
    use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
    use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;
    use crate::services::provider_protocol::detect_provider_protocol;
    use crate::services::{
        AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig, CopilotRouter, CopilotRouterConfig,
        ResponsesToChatRouter, ResponsesToChatRouterConfig,
    };
    use std::path::PathBuf;

    sweep_stale_amp_settings_files();

    let upstream_base_url = env
        .remove("AIVO_AMP_UPSTREAM_BASE_URL")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_AMP_UPSTREAM_BASE_URL"))?;
    let upstream_api_key = env
        .remove("AIVO_AMP_UPSTREAM_KEY")
        .ok_or_else(|| anyhow::anyhow!("Missing AIVO_AMP_UPSTREAM_KEY"))?;

    // OAuth sentinels need their own credential refreshers plumbed through
    // the bridge's translator slots, which isn't wired up yet. Bail loudly
    // before spawning anything — without this, the translators receive the
    // literal sentinel string as `target_base_url`, fail to parse it as a
    // URL, and amp sees a stream of opaque "builder error" 500s.
    if matches!(
        upstream_base_url.as_str(),
        CLAUDE_OAUTH_SENTINEL | CODEX_OAUTH_SENTINEL | GEMINI_OAUTH_SENTINEL
    ) {
        anyhow::bail!(
            "amp doesn't yet support `{upstream_base_url}` keys — pick a key with a real base URL, or `copilot`/`ollama`"
        );
    }

    // Resolve sentinel upstreams. `copilot` swaps in a local CopilotRouter
    // (which natively speaks Anthropic /v1/messages and translates to
    // Copilot's chat API internally) plus a Copilot-mode
    // ResponsesToChatRouter for amp's /v1/responses calls. `ollama` resolves
    // to its loopback OpenAI-compat URL — the regular translator setup
    // works unchanged from there.
    let copilot_github_token = (upstream_base_url == "copilot").then(|| upstream_api_key.clone());
    let upstream_base_url = if upstream_base_url == "ollama" {
        "http://localhost:11434/v1".to_string()
    } else {
        upstream_base_url
    };
    let native_amp_url = env.remove("AIVO_AMP_NATIVE_URL");
    let native_amp_key = env.remove("AIVO_AMP_NATIVE_KEY");
    let force_model = env.remove("AIVO_AMP_FORCE_MODEL");
    // Two ways the internal.model override arrives: `_JSON` (object form
    // from per-mode flags) wins over the bare string form (from `--1m`).
    let internal_model: Option<serde_json::Value> = env
        .remove("AIVO_AMP_INTERNAL_MODEL_JSON")
        .and_then(|s| serde_json::from_str(&s).ok())
        .or_else(|| {
            env.remove("AIVO_AMP_INTERNAL_MODEL")
                .map(serde_json::Value::String)
        });
    let tools_disable: Vec<String> = env
        .remove("AIVO_AMP_TOOLS_DISABLE")
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let is_starter = env.remove("AIVO_AMP_IS_STARTER").as_deref() == Some("1");
    env.remove("AIVO_USE_AMP_BRIDGE");

    // The bridge and its sub-routers (anthropic+responses translators) all
    // build reqwest clients in aivo's process. Those clients honor the
    // ambient `HTTP_PROXY`/`HTTPS_PROXY` at construction time — so a user
    // with e.g. privoxy on localhost has the bridge's localhost calls to
    // its own in-process sub-router round-tripped through their proxy,
    // which 500s.
    //
    // For claude/codex this asymmetry doesn't exist: the localhost call
    // (tool → router) lives in the spawned tool's process, and its env is
    // patched by `set_local_base_url → ensure_loopback_no_proxy`. The
    // amp bridge needs the same treatment but applied to *aivo's process*
    // env, since the localhost hop happens here, not in the amp child.
    //
    // Outbound to upstream (api.deepseek.com, api.getaivo.dev, …) still
    // honors HTTP_PROXY normally — only loopback bypasses.
    ensure_loopback_no_proxy_in_process_env();

    // For Copilot, both translator slots get filled with Copilot-aware
    // routers and the bridge's `upstream_base_url` is repointed at the
    // local CopilotRouter so any catch-all request lands somewhere safe.
    // The CopilotRouter natively accepts /v1/messages, so it slots in
    // where the AnthropicToOpenAIRouter would otherwise sit; the
    // ResponsesToChatRouter is configured with a CopilotTokenManager
    // (mirroring `start_responses_to_chat_copilot_router`).
    let (anthropic_translation_port, responses_translation_port, upstream_base_url) =
        if let Some(github_token) = copilot_github_token {
            let copilot_router = CopilotRouter::new(CopilotRouterConfig {
                github_token: github_token.clone(),
            });
            let (anthropic_port, anthropic_handle) = copilot_router.start_background().await?;
            tokio::spawn(async move {
                if let Ok(Err(e)) = anthropic_handle.await {
                    eprintln!("aivo: amp-bridge copilot router exited: {e}");
                }
            });

            let responses_router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
                target_base_url: String::new(),
                api_key: String::new(),
                target_protocol: ProviderProtocol::Openai,
                target_path_variant: None,
                copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
                model_prefix: None,
                requires_reasoning_content: false,
                actual_model: None,
                max_tokens_cap: None,
                responses_api_supported: None,
                is_starter: false,
                aivo_prefix_models: Vec::new(),
            });
            let (responses_port, _active, _resp_api, _success, _auth, _learned, responses_handle) =
                responses_router.start_background().await?;
            tokio::spawn(async move {
                if let Ok(Err(e)) = responses_handle.await {
                    eprintln!("aivo: amp-bridge copilot responses translator exited: {e}");
                }
            });

            (
                Some(anthropic_port),
                Some(responses_port),
                format!("http://127.0.0.1:{anthropic_port}"),
            )
        } else {
            // When the upstream isn't natively Anthropic, spawn aivo's existing
            // AnthropicToOpenAIRouter as a sub-component so the bridge can translate
            // Amp's Anthropic-protocol calls (`/api/provider/anthropic/v1/messages`)
            // into the upstream's native protocol.
            let upstream_protocol = detect_provider_protocol(&upstream_base_url);
            let anthropic_port = if upstream_protocol == ProviderProtocol::Anthropic {
                None
            } else {
                let translator = AnthropicToOpenAIRouter::new(AnthropicToOpenAIRouterConfig {
                    target_base_url: upstream_base_url.clone(),
                    target_api_key: upstream_api_key.clone(),
                    target_protocol: upstream_protocol,
                    target_path_variant: None,
                    strip_cache_control: false,
                    model_prefix: None,
                    requires_reasoning_content: false,
                    max_tokens_cap: None,
                    is_starter,
                });
                let (port, _active, _success, _auth, _learned, handle) =
                    translator.start_background().await?;
                tokio::spawn(async move {
                    if let Ok(Err(e)) = handle.await {
                        eprintln!("aivo: amp-bridge anthropic translator exited: {e}");
                    }
                });
                Some(port)
            };

            // Amp's interactive chat uses the OpenAI Responses API (`/v1/responses`).
            // Most non-OpenAI upstreams only have `/v1/chat/completions`, so spawn
            // aivo's ResponsesToChatRouter to translate. Skip for native upstreams
            // that already speak the Responses API.
            let responses_port = if upstream_protocol == ProviderProtocol::ResponsesApi {
                None
            } else {
                let translator = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
                    target_base_url: upstream_base_url.clone(),
                    api_key: upstream_api_key.clone(),
                    target_protocol: upstream_protocol,
                    target_path_variant: None,
                    copilot_token_manager: None,
                    model_prefix: None,
                    requires_reasoning_content: false,
                    actual_model: None,
                    max_tokens_cap: None,
                    responses_api_supported: Some(false),
                    is_starter,
                    aivo_prefix_models: Vec::new(),
                });
                let (port, _active, _resp_api, _success, _auth, _learned, handle) =
                    translator.start_background().await?;
                tokio::spawn(async move {
                    if let Ok(Err(e)) = handle.await {
                        eprintln!("aivo: amp-bridge responses translator exited: {e}");
                    }
                });
                Some(port)
            };

            (anthropic_port, responses_port, upstream_base_url)
        };

    // Only allocate a trace file when `--debug` is on. A normal `aivo amp`
    // run leaves `~/.config/aivo/logs/` untouched. Both legs of the bridge
    // are gated by the same flag: http_debug captures aivo→upstream,
    // amp-trace captures amp↔bridge.
    let trace_log_path = crate::services::http_debug::is_debug_active().then(|| {
        let home = crate::services::system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let now = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let pid = std::process::id();
        home.join(".config")
            .join("aivo")
            .join("logs")
            .join(format!("amp-trace-{now}-{pid}.jsonl"))
    });

    // When `--1m`, per-mode flags, or `--disable-tool` is requested,
    // write a merged settings file: user's existing
    // ~/.config/amp/settings.json + our overrides. Pass `--settings-file
    // <path>` to amp via the runtime args injector. We must merge rather
    // than overwrite — amp's `--settings-file` replaces the default; if
    // we wrote a bare override, the user would lose their MCP servers,
    // skills, permissions, etc.
    if internal_model.is_some() || !tools_disable.is_empty() {
        let path = write_amp_settings_override(internal_model.as_ref(), &tools_disable)?;
        env.insert(
            "AIVO_AMP_SETTINGS_FILE".to_string(),
            path.to_string_lossy().into_owned(),
        );
    }

    let threads_dir = crate::services::amp_threads::default_threads_dir();
    let bridge = AmpBridge::new(AmpBridgeConfig {
        upstream_base_url,
        upstream_api_key,
        trace_log_path,
        native_amp_url,
        native_amp_key,
        anthropic_translation_port,
        responses_translation_port,
        force_model,
        threads_dir,
    });
    let (port, handle) = bridge.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: amp bridge exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

/// Removes `amp-settings-*.json` files older than 24h from
/// `~/.config/aivo/cache/`. The on-exit cleanup in
/// `cleanup_runtime_artifacts` removes the file from the current launch,
/// but a crashed prior launch leaves its file behind — sweep them here
/// so the cache directory doesn't grow unbounded.
fn sweep_stale_amp_settings_files() {
    let Some(home) = crate::services::system_env::home_dir() else {
        return;
    };
    let dir = home.join(".config").join("aivo").join("cache");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let cutoff =
        std::time::SystemTime::now().checked_sub(std::time::Duration::from_secs(24 * 60 * 60));
    for entry in entries.flatten() {
        let path = entry.path();
        let is_amp_settings = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("amp-settings-") && n.ends_with(".json"));
        if !is_amp_settings {
            continue;
        }
        let too_old = match (
            cutoff,
            entry.metadata().ok().and_then(|m| m.modified().ok()),
        ) {
            (Some(cutoff_at), Some(mtime)) => mtime < cutoff_at,
            _ => false,
        };
        if too_old {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Generates a settings.json file at a known cache path that mirrors amp's
/// own discovery order — user (`~/.config/amp/settings.json`) merged with
/// workspace (nearest `.amp/settings.json[c]` walking up from CWD) plus
/// aivo's `amp.internal.model` overrides plus enterprise managed settings
/// last (so corporate enforcement always wins). amp will be launched with
/// `--settings-file <returned path>` so this becomes the active settings
/// instead of the default.
///
/// Why each layer matters:
/// - **User**: long-standing — preserves the user's own MCP servers,
///   skills, permissions.
/// - **Workspace**: `--settings-file` *replaces* (does not merge with)
///   amp's normal discovery, so without this layer a repo's
///   `.amp/settings.json` (project-specific skills, fuzzy paths)
///   silently disappears under the bridge while working under direct
///   `amp`.
/// - **Workspace MCP servers** are passed through `amp_trust` first:
///   only entries the user has approved via `aivo amp trust` survive.
///   Direct `amp` gates these via `amp mcp approve`; bypassing that
///   would let a hostile checkout's `.amp/settings.json` auto-launch
///   an MCP server with the user's credentials.
/// - **Aivo overrides**: `internal.model` for model rewriting plus the
///   bridge-aligned defaults (`amp.updates.mode: "disabled"`, etc.)
///   from `build_amp_settings_override`.
/// - **Managed settings**: `/Library/Application Support/ampcode/
///   managed-settings.json` etc. Layered LAST so corporate enforcement
///   (forbidden tools, MCP allowlist, IP allowlist, locked compatibility
///   date) is preserved end-to-end. Without this layer the bridge would
///   silently strip those policies — a compliance evasion.
///
/// `internal_model` is either a string (`"openai:gpt-5.5-pro"` from
/// `--1m`) or an object keyed by mode (`{"smart":"openai:...", "rush":...}`
/// from per-mode flags), or `None` when the caller is only setting
/// `tools_disable`. Both shapes are accepted by amp's settings reader.
/// `tools_disable` is the list of amp tool names to add to
/// `tools.disable` (union with the user's existing setting, deduped).
fn write_amp_settings_override(
    internal_model: Option<&serde_json::Value>,
    tools_disable: &[String],
) -> Result<std::path::PathBuf> {
    use std::path::PathBuf;
    let home = crate::services::system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let user_settings_path = home.join(".config").join("amp").join("settings.json");
    let user_value = read_amp_settings_file(&user_settings_path);

    let workspace_path = crate::services::system_env::current_dir()
        .and_then(|cwd| find_workspace_amp_settings(&cwd, Some(&home)));
    let workspace_value = match workspace_path.as_deref() {
        Some(p) => filter_workspace_settings(p, read_amp_settings_file(p)),
        None => None,
    };

    let merged_existing = merge_amp_settings_layers(user_value, workspace_value);
    let with_aivo_overrides =
        build_amp_settings_override(merged_existing, internal_model, tools_disable);

    // Managed settings layer: corporate policy wins over everything,
    // including aivo's bridge-needed overrides. If a managed
    // `internal.model` lock undoes the bridge's model rewrite, that's
    // the corp's call — better to fail loudly than silently bypass.
    let managed_value = find_managed_amp_settings()
        .as_deref()
        .and_then(read_amp_settings_file);
    let final_value = merge_amp_settings_layers(Some(with_aivo_overrides), managed_value)
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));

    let pid = std::process::id();
    let cache_dir = home.join(".config").join("aivo").join("cache");
    std::fs::create_dir_all(&cache_dir)?;
    let path = cache_dir.join(format!("amp-settings-{pid}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&final_value)?)?;
    Ok(path)
}

/// Applies the trust filter to workspace settings: any `amp.mcpServers`
/// entry the user hasn't explicitly approved via `aivo amp trust` is
/// dropped. Other workspace settings (skills paths, fuzzy paths, tool
/// disables) pass through unchanged — they're not in the same security
/// category as MCP servers, which can spawn arbitrary subprocesses.
///
/// Emits a one-line stderr warning when filtering happened so the user
/// knows why their workspace MCP servers aren't loading.
fn filter_workspace_settings(
    workspace_path: &std::path::Path,
    settings: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    let mut value = settings?;
    let trust = crate::services::amp_trust::AmpTrustStore::load();
    let dropped = crate::services::amp_trust::filter_workspace_mcp_servers(
        workspace_path,
        &mut value,
        &trust,
    );
    if !dropped.is_empty() {
        let count = dropped.len();
        let names = dropped.join(", ");
        eprintln!(
            "aivo: skipped {count} unapproved workspace MCP server(s) from {}: {names}",
            workspace_path.display()
        );
        eprintln!("       run `aivo amp trust` from this repo to approve");
    }
    Some(value)
}

/// Returns the nearest existing platform-specific managed-settings path,
/// or `None` if no managed settings file is present. Mirrors amp's own
/// search locations from the manual.
fn find_managed_amp_settings() -> Option<std::path::PathBuf> {
    managed_settings_paths().into_iter().find(|p| p.is_file())
}

fn managed_settings_paths() -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;
    let mut paths = Vec::new();
    #[cfg(target_os = "macos")]
    {
        paths.push(PathBuf::from(
            "/Library/Application Support/ampcode/managed-settings.json",
        ));
    }
    #[cfg(target_os = "linux")]
    {
        paths.push(PathBuf::from("/etc/ampcode/managed-settings.json"));
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(prog_data) = std::env::var("ProgramData") {
            paths.push(
                PathBuf::from(prog_data)
                    .join("ampcode")
                    .join("managed-settings.json"),
            );
        }
    }
    paths
}

// Workspace-settings discovery + JSONC parsing live in `amp_trust`; the
// bridge consumes them via `use crate::services::amp_trust::{...}`.

/// Layers workspace settings on top of user settings via shallow
/// top-level-key replacement: each key the workspace defines wins entirely
/// over the user's value (replacing maps, not deep-merging them). amp's
/// settings keys are flat dotted paths (`amp.mcpServers`, `amp.git.commit.
/// coauthor.enabled`), so per-key replacement is the natural granularity.
///
/// Tradeoff worth knowing: a user with `amp.mcpServers: {a, b}` and a
/// workspace with `amp.mcpServers: {c}` ends up with just `{c}`. That's
/// the simplest predictable rule; deep-merging maps quietly introduces
/// surprise overrides. Users who need both can declare both sides
/// explicitly.
fn merge_amp_settings_layers(
    user: Option<serde_json::Value>,
    workspace: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (user, workspace) {
        (None, None) => None,
        (Some(v), None) | (None, Some(v)) => Some(v),
        (Some(user_v), Some(ws_v)) => {
            let mut user_obj = match user_v {
                serde_json::Value::Object(m) => m,
                _ => serde_json::Map::new(),
            };
            if let serde_json::Value::Object(ws_obj) = ws_v {
                for (k, v) in ws_obj {
                    user_obj.insert(k, v);
                }
            }
            Some(serde_json::Value::Object(user_obj))
        }
    }
}

/// Builds the merged settings JSON: user's existing settings (if any) +
/// aivo's `amp.internal.model` override + a small set of "set-if-absent"
/// defaults that align amp's behavior with the bridge.
///
/// Defaults applied only when the user hasn't set them in their own
/// `~/.config/amp/settings.json`:
/// - `amp.showCosts: false` — amp's TUI cost is computed from a hardcoded
///   pricing table keyed on the model name amp *thinks* it's calling. With
///   the bridge rewriting models and rerouting traffic to deepseek/openrouter/
///   etc., that number is fiction. aivo tracks real cost in `aivo stats`.
/// - `amp.git.commit.coauthor.enabled: false` — amp adds itself as a commit
///   coauthor by default. Misattributes commits when the actual model is
///   not Claude on ampcode.com.
/// - `amp.git.commit.ampThread.enabled: false` — adds an `Amp-Thread:` trailer
///   pointing at `ampcode.com/threads/<id>`. The bridge stubs thread upload
///   locally, so the URL never resolves.
/// - `amp.updates.mode: "disabled"` — pins amp's binary so the bridge's
///   reverse-engineered protocol assumptions (RPC envelope, settings keys,
///   SSE event shapes) don't drift out from under aivo. Belt-and-suspenders
///   with `AMP_SKIP_UPDATE_CHECK=1` set in `for_amp`.
/// - `amp.notifications.enabled: false` — bridge launches are typically
///   scripted/background; the completion chime is noise rather than signal.
/// - `amp.network.timeout: 600` — amp's binary defaults this to 30s, which
///   is shorter than the time many reasoning models on remote upstreams
///   (deepseek-reasoner, gpt-5.5-pro at high effort, …) take to first
///   token. The result was amp aborting requests the bridge was still
///   patiently forwarding. aivo's own reqwest client caps upstream calls
///   at 300s, so 600s here just lets aivo be the authoritative timeout
///   instead of amp racing it.
///
/// `internal_model` is optional: when `None`, no model rewrite is
/// applied (used when the caller only wants `tools_disable`).
/// `tools_disable` entries are appended to the user's existing
/// `tools.disable` array (union, dedup-preserving order: user entries
/// first, aivo entries after).
fn build_amp_settings_override(
    existing: Option<serde_json::Value>,
    internal_model: Option<&serde_json::Value>,
    tools_disable: &[String],
) -> serde_json::Value {
    let mut value = match existing {
        Some(v) if v.is_object() => v,
        _ => serde_json::Value::Object(serde_json::Map::new()),
    };
    let Some(obj) = value.as_object_mut() else {
        return value;
    };
    // amp's binary reads `T["internal.model"]` directly. We don't know
    // for certain whether amp's settings loader strips the `amp.` prefix
    // on load, so write both forms — the prefixed form for user-facing
    // consistency, the bare form to match what the binary looks up.
    // Whichever amp honors, the override takes effect.
    if let Some(model) = internal_model {
        obj.insert("amp.internal.model".to_string(), model.clone());
        obj.insert("internal.model".to_string(), model.clone());
    }

    // Union with user's existing `tools.disable` (preserves user entries
    // first, then appends aivo's, dedup'd). Write BOTH `amp.tools.disable`
    // (the `amp.<key>` convention used by `amp.dangerouslyAllowAll`,
    // `amp.permissions`, `amp.tools.inactivityTimeout`, etc.) AND bare
    // `tools.disable` (the form amp's `dx` matcher reads directly via
    // `R.settings["tools.disable"]`). Same dual-write strategy as
    // `internal.model` / `amp.internal.model`. Bare-only didn't take
    // effect on a real launch — settings file was loaded yet all 40 tools
    // appeared in the request body — suggesting amp's loader keys some
    // code paths off the prefixed form.
    if !tools_disable.is_empty() {
        let read_existing = |key: &str| -> Vec<String> {
            obj.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default()
        };
        let mut merged = read_existing("amp.tools.disable");
        for entry in read_existing("tools.disable") {
            if !merged.iter().any(|e| e == &entry) {
                merged.push(entry);
            }
        }
        for tool in tools_disable {
            if !merged.iter().any(|existing| existing == tool) {
                merged.push(tool.clone());
            }
        }
        let arr =
            serde_json::Value::Array(merged.into_iter().map(serde_json::Value::String).collect());
        obj.insert("amp.tools.disable".to_string(), arr.clone());
        obj.insert("tools.disable".to_string(), arr);
    }

    for (key, default) in [
        ("amp.showCosts", serde_json::Value::Bool(false)),
        (
            "amp.git.commit.coauthor.enabled",
            serde_json::Value::Bool(false),
        ),
        (
            "amp.git.commit.ampThread.enabled",
            serde_json::Value::Bool(false),
        ),
        (
            "amp.updates.mode",
            serde_json::Value::String("disabled".to_string()),
        ),
        ("amp.notifications.enabled", serde_json::Value::Bool(false)),
        (
            "amp.network.timeout",
            serde_json::Value::Number(serde_json::Number::from(600)),
        ),
    ] {
        if !obj.contains_key(key) {
            obj.insert(key.to_string(), default);
        }
    }
    value
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
        target_path_variant: None,
        copilot_token_manager: Some(Arc::new(CopilotTokenManager::new(github_token))),
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: false,
        aivo_prefix_models: Vec::new(),
    });
    let (
        port,
        _active_protocol,
        _responses_api,
        _request_succeeded,
        _saw_authoritative_response,
        _learned_requires_reasoning,
        handle,
    ) = router.start_background().await?;
    tokio::spawn(async move {
        if let Ok(Err(e)) = handle.await {
            eprintln!("aivo: responses-to-chat copilot router exited unexpectedly: {e}");
        }
    });
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::{
        build_amp_settings_override, clear_node_proxy_env, is_oauth_invalid_grant,
        managed_settings_paths, merge_amp_settings_layers, patch_opencode_config_content,
        prepare_gemini_api_key_settings_override,
    };
    use std::collections::HashMap;

    #[test]
    fn amp_settings_override_applies_defaults_on_empty() {
        // No existing settings → bridge inserts internal.model + the
        // bridge-aligned defaults. Costs/coauthor/thread-trailer all off
        // because they're misleading or broken under the bridge; updates
        // pinned and notifications off to keep the bridge's protocol
        // assumptions stable and the launch quiet.
        let model = serde_json::json!("openai:gpt-5.5-pro");
        let v = build_amp_settings_override(None, Some(&model), &[]);
        assert_eq!(v["amp.internal.model"], model);
        assert_eq!(v["internal.model"], model);
        assert_eq!(v["amp.showCosts"], false);
        assert_eq!(v["amp.git.commit.coauthor.enabled"], false);
        assert_eq!(v["amp.git.commit.ampThread.enabled"], false);
        assert_eq!(v["amp.updates.mode"], "disabled");
        assert_eq!(v["amp.notifications.enabled"], false);
        // amp's binary defaults network.timeout to 30s, which is shorter
        // than reasoning models often need to first token. Bumped so aivo
        // (300s upstream cap) is the authoritative timeout, not amp.
        assert_eq!(v["amp.network.timeout"], 600);
    }

    #[test]
    fn merge_amp_settings_layers_workspace_overrides_user_per_key() {
        // Workspace wins by top-level key; user keys not touched by
        // workspace survive verbatim.
        let user = serde_json::json!({
            "amp.showCosts": true,
            "amp.mcpServers": {"user_only": {"command": "npx"}},
        });
        let workspace = serde_json::json!({
            "amp.mcpServers": {"ws_only": {"command": "uvx"}},
            "amp.fuzzy.alwaysIncludePaths": ["docs/**"],
        });
        let merged = merge_amp_settings_layers(Some(user), Some(workspace)).unwrap();
        // Workspace's mcpServers fully replaces user's (shallow per-key
        // semantics). User's unrelated key survives. Workspace-only key
        // appears.
        assert!(merged["amp.mcpServers"]["ws_only"].is_object());
        assert!(merged["amp.mcpServers"].get("user_only").is_none());
        assert_eq!(merged["amp.showCosts"], true);
        assert_eq!(merged["amp.fuzzy.alwaysIncludePaths"][0], "docs/**");
    }

    #[test]
    fn merge_amp_settings_layers_managed_overrides_aivo_defaults() {
        // Real layering scenario: aivo's bridge sets `amp.updates.mode:
        // "disabled"` to keep its protocol assumptions stable. A
        // corporate-managed file says `"auto"`. Managed wins, end of
        // story — aivo accepts the bridge instability rather than
        // bypassing corp policy.
        let aivo = serde_json::json!({
            "amp.updates.mode": "disabled",
            "amp.internal.model": "openai:gpt-5.5-pro",
        });
        let managed = serde_json::json!({"amp.updates.mode": "auto"});
        let merged = merge_amp_settings_layers(Some(aivo), Some(managed)).unwrap();
        assert_eq!(merged["amp.updates.mode"], "auto");
        // Keys the managed file doesn't touch survive from the layer
        // below.
        assert_eq!(merged["amp.internal.model"], "openai:gpt-5.5-pro");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn managed_settings_paths_macos_targets_library_dir() {
        let paths = managed_settings_paths();
        let display: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        assert!(
            display
                .iter()
                .any(|p| p == "/Library/Application Support/ampcode/managed-settings.json"),
            "expected macOS managed path, got {display:?}"
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn managed_settings_paths_linux_targets_etc() {
        let paths = managed_settings_paths();
        let display: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        assert!(
            display
                .iter()
                .any(|p| p == "/etc/ampcode/managed-settings.json"),
            "expected Linux managed path, got {display:?}"
        );
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn managed_settings_paths_windows_uses_program_data_when_set() {
        // Reference managed_settings_paths so the cfg-gated import
        // doesn't trigger a dead-code warning on Windows.
        let _ = managed_settings_paths;
        // SAFETY: single-threaded test setup.
        unsafe {
            std::env::set_var("ProgramData", "C:\\ProgramData");
        }
        let paths = super::managed_settings_paths();
        let display: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        assert!(
            display
                .iter()
                .any(|p| p.contains("ampcode") && p.ends_with("managed-settings.json")),
            "expected Windows managed path under %ProgramData%, got {display:?}"
        );
    }

    #[test]
    fn merge_amp_settings_layers_handles_one_sided_inputs() {
        // Either side missing → return the other side untouched. Both
        // missing → None (caller falls back to bare aivo defaults).
        let v = serde_json::json!({"k": 1});
        assert_eq!(
            merge_amp_settings_layers(Some(v.clone()), None),
            Some(v.clone())
        );
        assert_eq!(merge_amp_settings_layers(None, Some(v.clone())), Some(v));
        assert!(merge_amp_settings_layers(None, None).is_none());
    }

    #[test]
    fn amp_settings_override_preserves_user_updates_and_notifications() {
        // User explicitly opted into amp's auto-update / notifications,
        // and set their own network timeout — bridge must not silently
        // flip them back. Drift risk is on the user; aivo only sets
        // defaults when nothing's there.
        let existing = serde_json::json!({
            "amp.updates.mode": "auto",
            "amp.notifications.enabled": true,
            "amp.network.timeout": 30,
        });
        let model = serde_json::json!("openai:gpt-5.5-pro");
        let v = build_amp_settings_override(Some(existing), Some(&model), &[]);
        assert_eq!(v["amp.updates.mode"], "auto");
        assert_eq!(v["amp.notifications.enabled"], true);
        assert_eq!(v["amp.network.timeout"], 30);
    }

    #[test]
    fn amp_settings_override_preserves_existing_user_choices() {
        // User explicitly opted into costs / coauthor in their own
        // settings.json — bridge must not silently flip them back to false.
        let existing = serde_json::json!({
            "amp.showCosts": true,
            "amp.git.commit.coauthor.enabled": true,
            "amp.mcpServers": { "fs": { "command": "npx" } },
        });
        let model = serde_json::json!("openai:gpt-5.5-pro");
        let v = build_amp_settings_override(Some(existing), Some(&model), &[]);
        assert_eq!(v["amp.showCosts"], true);
        assert_eq!(v["amp.git.commit.coauthor.enabled"], true);
        // Defaults still kick in for the keys the user *didn't* set.
        assert_eq!(v["amp.git.commit.ampThread.enabled"], false);
        // Unrelated user settings (MCP servers, etc.) survive.
        assert!(v["amp.mcpServers"]["fs"].is_object());
        // Internal model still inserted.
        assert_eq!(v["amp.internal.model"], model);
    }

    #[test]
    fn amp_settings_override_writes_tools_disable_without_internal_model() {
        // `--disable-tool web_search` alone (no `--1m` / no per-mode flags)
        // should still produce a settings file with `tools.disable` set.
        // Internal-model keys must NOT appear when the caller passes None.
        // Both prefixed (`amp.tools.disable`) and bare (`tools.disable`)
        // forms are written — see comment in build_amp_settings_override.
        let v = build_amp_settings_override(None, None, &["web_search".to_string()]);
        assert!(v.get("amp.internal.model").is_none());
        assert!(v.get("internal.model").is_none());
        assert_eq!(v["tools.disable"][0], "web_search");
        assert_eq!(v["amp.tools.disable"][0], "web_search");
        // Bridge defaults still apply (the settings file is the active
        // config; missing the timeout bump etc. would defeat the purpose).
        assert_eq!(v["amp.network.timeout"], 600);
    }

    #[test]
    fn amp_settings_override_unions_tools_disable_with_user_existing() {
        // User has their own `tools.disable: ["foo"]`. aivo adds
        // `web_search`. Result: union, dedup'd, user entries first. Both
        // prefixed and bare keys carry the merged list.
        let existing = serde_json::json!({
            "tools.disable": ["foo", "web_search"],
        });
        let v = build_amp_settings_override(
            Some(existing),
            None,
            &["web_search".to_string(), "read_web_page".to_string()],
        );
        for key in ["tools.disable", "amp.tools.disable"] {
            let arr = v[key].as_array().unwrap_or_else(|| panic!("{key} array"));
            let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
            // user's "foo" stays first, "web_search" is dedup'd, "read_web_page" appended.
            assert_eq!(names, ["foo", "web_search", "read_web_page"], "key={key}");
        }
    }

    #[test]
    fn amp_settings_override_unions_tools_disable_across_both_keys() {
        // User wrote BOTH prefixed and bare in their existing settings,
        // each with different entries. Aivo unions all three sources
        // (prefixed-existing, bare-existing, aivo-supplied) into one list,
        // and writes that single merged list to both keys.
        let existing = serde_json::json!({
            "amp.tools.disable": ["foo"],
            "tools.disable": ["bar"],
        });
        let v = build_amp_settings_override(Some(existing), None, &["web_search".to_string()]);
        let prefixed: Vec<&str> = v["amp.tools.disable"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        let bare: Vec<&str> = v["tools.disable"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x.as_str())
            .collect();
        assert_eq!(prefixed, ["foo", "bar", "web_search"]);
        assert_eq!(bare, prefixed);
    }

    #[test]
    fn amp_settings_override_handles_non_object_existing() {
        // Garbage settings.json (array, scalar) — fall back to a fresh
        // object rather than crashing or returning the garbage.
        let model = serde_json::json!({"smart": "openai:m"});
        let v = build_amp_settings_override(Some(serde_json::json!([1, 2, 3])), Some(&model), &[]);
        assert!(v.is_object());
        assert_eq!(v["amp.internal.model"], model);
        assert_eq!(v["amp.showCosts"], false);
    }

    #[test]
    fn is_oauth_invalid_grant_matches_4xx_refresh_failures() {
        // Real-world wording from the OpenAI token endpoint. The whole point
        // of this branch is to trigger interactive re-login, so the bodies
        // matter less than the status prefix.
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (401): {{\"error\":{{\"message\":\"Your refresh token has already been used\"}}}}"
        )));
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (400): invalid_grant"
        )));
        assert!(is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (403): forbidden"
        )));
    }

    #[test]
    fn is_oauth_invalid_grant_skips_transient_failures() {
        // 5xx, network, and parse errors aren't recoverable via re-login —
        // we don't want to open a browser for a transient outage.
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "refresh failed (500): bad gateway"
        )));
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "POST /oauth/token (refresh_token)"
        )));
        assert!(!is_oauth_invalid_grant(&anyhow::anyhow!(
            "parse refresh response"
        )));
    }

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

    #[test]
    fn clear_node_proxy_env_empties_known_proxy_vars() {
        // Needed for gemini-cli: it builds a global undici ProxyAgent from
        // HTTP(S)_PROXY and then routes every fetch through it, including to
        // our loopback router. NO_PROXY is ignored by ProxyAgent, so the only
        // way to prevent this is to hide the proxy vars from the child. We
        // also clear ALL_PROXY because gemini's bundled `proxy-from-env`
        // lookup consults it as a fallback.
        let mut env = HashMap::from([
            ("HTTP_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("HTTPS_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("ALL_PROXY".to_string(), "http://proxy:8080".to_string()),
            ("http_proxy".to_string(), "http://proxy:8080".to_string()),
            ("https_proxy".to_string(), "http://proxy:8080".to_string()),
            ("all_proxy".to_string(), "http://proxy:8080".to_string()),
            ("PATH".to_string(), "/usr/bin".to_string()),
        ]);
        clear_node_proxy_env(&mut env);
        for var in [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ] {
            assert_eq!(env.get(var), Some(&String::new()), "{var} should be empty");
        }
        assert_eq!(
            env.get("PATH"),
            Some(&"/usr/bin".to_string()),
            "unrelated vars untouched"
        );
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_skips_protocol_write_when_no_request_succeeded() {
        // Reproduces the bug where a real-Anthropic endpoint with a bad API
        // key would flip `claude_protocol` to Openai because the failed native
        // probe poisoned the active_protocol atomic before exit. With the
        // success gate, the persisted protocol must remain Anthropic.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::ProviderProtocol;
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.anthropic.com",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-bad",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        // Simulate the post-probe state: active_protocol got optimistically
        // flipped to Openai (legacy behavior) but no request ever succeeded.
        let active = Arc::new(AtomicU8::new(ProviderProtocol::Openai.to_u8()));
        let succeeded = Arc::new(AtomicBool::new(false));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            None,
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.claude_protocol,
            Some(ClaudeProviderProtocol::Anthropic),
            "claude_protocol must NOT be rewritten to Openai when no request succeeded"
        );
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_writes_protocol_when_request_succeeded() {
        // Counterpart: when a request did succeed, the learned protocol
        // change must persist normally.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::ProviderProtocol;
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://gateway.example.com/v1",
                Some(ClaudeProviderProtocol::Anthropic),
                "sk-good",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let active = Arc::new(AtomicU8::new(ProviderProtocol::Openai.to_u8()));
        let succeeded = Arc::new(AtomicBool::new(true));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            None,
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.claude_protocol,
            Some(ClaudeProviderProtocol::Openai),
            "claude_protocol must be rewritten to the learned protocol after a successful request"
        );
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_writes_google_pin_for_google_host_key() {
        // Mirrors the live user flow: claude key against
        // generativelanguage.googleapis.com. resolve_claude_protocol pre-fills
        // the in-memory key.claude_protocol with Anthropic (the cli-native bet);
        // the router fallback then learns Google. Persistence must rewrite the
        // claude_protocol pin to Google so the next launch skips the probe.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::ProviderProtocol;
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "google",
                "https://generativelanguage.googleapis.com",
                None,
                "ya29-test",
            )
            .await
            .unwrap();
        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        // resolve_claude_protocol fills this in-memory before launch.
        key.claude_protocol = Some(ClaudeProviderProtocol::Anthropic);

        // Router observed Google success; pin advanced to Google.
        let active = Arc::new(AtomicU8::new(ProviderProtocol::Google.to_u8()));
        let succeeded = Arc::new(AtomicBool::new(true));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            None,
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.claude_protocol,
            Some(ClaudeProviderProtocol::Google),
            "google host pin must persist to disk so the next launch skips the probe"
        );
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_stores_path_variant_for_claude() {
        // A stripped-path win (e.g., upstream serves `/messages` instead of
        // `/v1/messages`) must be persisted as well, otherwise it has to be
        // re-probed every launch.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::{PathVariant, ProviderProtocol, encode_route};
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://gateway.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-good",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        // Router pinned (Openai, Stripped) after a successful probe.
        let active = Arc::new(AtomicU8::new(encode_route(
            ProviderProtocol::Openai,
            PathVariant::Stripped,
        )));
        let succeeded = Arc::new(AtomicBool::new(true));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            None,
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.claude_path_variant.as_deref(),
            Some("stripped"),
            "stripped path variant must persist so the next launch skips re-probing"
        );
    }

    #[tokio::test]
    async fn migrate_routing_schema_clears_stale_responses_api_false_and_bumps_version() {
        use crate::services::session_store::{
            CURRENT_ROUTING_SCHEMA_VERSION, ClaudeProviderProtocol, SessionStore,
        };

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-1",
            )
            .await
            .unwrap();

        // Force the legacy state: pre-fix builds wrote responses_api_supported =
        // Some(false) on transient errors. Simulate that by writing it directly,
        // then resetting routing_schema_version to 0 (legacy).
        store
            .set_key_responses_api_supported(&key_id, Some(false))
            .await
            .unwrap();
        store
            .set_key_routing_schema_version(&key_id, 0)
            .await
            .unwrap();

        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(key.responses_api_supported, Some(false));
        assert_eq!(key.routing_schema_version, 0);

        super::migrate_routing_schema_for_key(&store, &mut key).await;

        // In-memory key is updated.
        assert_eq!(key.responses_api_supported, None);
        assert_eq!(key.routing_schema_version, CURRENT_ROUTING_SCHEMA_VERSION);

        // Persisted key matches.
        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.responses_api_supported, None);
        assert_eq!(
            reloaded.routing_schema_version,
            CURRENT_ROUTING_SCHEMA_VERSION
        );
    }

    #[tokio::test]
    async fn migrate_routing_schema_preserves_some_true_and_some_none() {
        use crate::services::session_store::{
            CURRENT_ROUTING_SCHEMA_VERSION, ClaudeProviderProtocol, SessionStore,
        };

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-1",
            )
            .await
            .unwrap();

        // Some(true) means the upstream really does support responses — must
        // not be cleared.
        store
            .set_key_responses_api_supported(&key_id, Some(true))
            .await
            .unwrap();
        store
            .set_key_routing_schema_version(&key_id, 0)
            .await
            .unwrap();

        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        super::migrate_routing_schema_for_key(&store, &mut key).await;

        assert_eq!(key.responses_api_supported, Some(true));
        assert_eq!(key.routing_schema_version, CURRENT_ROUTING_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn migrate_routing_schema_is_idempotent_when_already_current() {
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-1",
            )
            .await
            .unwrap();

        // New keys are stamped at the current version, so migration is a no-op.
        let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        let before = key.clone();
        super::migrate_routing_schema_for_key(&store, &mut key).await;
        assert_eq!(key, before);
    }

    #[tokio::test]
    async fn prepare_gemini_api_key_settings_override_pins_selected_type() {
        let mut env = HashMap::from([
            (
                "AIVO_GEMINI_FORCE_API_KEY_AUTH".to_string(),
                "1".to_string(),
            ),
            (
                "AIVO_GEMINI_MODEL_CONFIG_MODEL".to_string(),
                "aivo/starter".to_string(),
            ),
        ]);
        let dir = prepare_gemini_api_key_settings_override(&mut env)
            .await
            .unwrap();

        // Sentinel consumed — must not leak to the spawned child.
        assert!(!env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH"));
        assert!(!env.contains_key("AIVO_GEMINI_MODEL_CONFIG_MODEL"));

        // Child sees a system-scope settings override path. Because
        // system-scope wins over user-scope in gemini-cli's merge, this
        // pins selectedType regardless of any stale `oauth-personal` in
        // the user's real ~/.gemini/settings.json.
        let path = env.get("GEMINI_CLI_SYSTEM_SETTINGS_PATH").unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            parsed["security"]["auth"]["selectedType"].as_str(),
            Some("gemini-api-key")
        );
        assert_eq!(
            parsed["modelConfigs"]["customAliases"]["prompt-completion"]["modelConfig"]["model"]
                .as_str(),
            Some("aivo/starter")
        );
        assert_eq!(
            parsed["modelConfigs"]["customAliases"]["classifier"]["modelConfig"]["model"].as_str(),
            Some("aivo/starter")
        );

        // We deliberately don't redirect GEMINI_CLI_HOME — user's real
        // ~/.gemini/ stays the gemini-cli user-scope root so in-session
        // edits (theme, vim mode, MCP tweaks) persist normally.
        assert!(!env.contains_key("GEMINI_CLI_HOME"));

        drop(dir);
        assert!(!std::path::Path::new(path).exists());
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_persists_learned_requires_reasoning_even_without_success()
    {
        use crate::services::ai_launcher::AIToolType;
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://example-strict-thinking.dev",
                Some(ClaudeProviderProtocol::Openai),
                "sk-test",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(key.requires_reasoning_content, None);

        // The cascade observed a parseable `reasoning_content` rejection but
        // never saw a 2xx — `request_succeeded` stays false. The learned
        // quirk must still persist so the next launch enables strict mode
        // for this key without growing the static substring list.
        let succeeded = Arc::new(AtomicBool::new(false));
        let learned = Arc::new(AtomicBool::new(true));
        learned.store(true, Ordering::Relaxed);

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            None,
            None,
            Some(succeeded),
            None,
            Some(learned),
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.requires_reasoning_content, Some(true));
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_skips_learning_when_flag_unset() {
        use crate::services::ai_launcher::AIToolType;
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-test",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let succeeded = Arc::new(AtomicBool::new(true));
        let learned = Arc::new(AtomicBool::new(false));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            None,
            None,
            Some(succeeded),
            None,
            Some(learned),
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.requires_reasoning_content, None);
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_persists_path_variant_after_authoritative_4xx() {
        // Scenario: cascade probed (Openai, Default) → 404 (path missing),
        // then (Openai, Stripped) → 400 with parseable error envelope.
        // No 2xx ever happened so request_succeeded is false, but the path
        // responded — the next launch should start at Stripped instead of
        // re-probing Default.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::{PathVariant, ProviderProtocol, encode_route};
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.stripped-path-gateway.example/v1",
                Some(ClaudeProviderProtocol::Openai),
                "sk-test",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(key.claude_path_variant, None);

        // commit_protocol_switch moved the in-memory pin to (Openai, Stripped)
        // after the second attempt observed a parseable 400.
        let active = Arc::new(AtomicU8::new(encode_route(
            ProviderProtocol::Openai,
            PathVariant::Stripped,
        )));
        let succeeded = Arc::new(AtomicBool::new(false));
        let authoritative = Arc::new(AtomicBool::new(true));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            Some(authoritative),
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.claude_path_variant.as_deref(), Some("stripped"));
        // Protocol must NOT be persisted — saw_success is false, the success
        // gate still protects against cross-protocol auth-shape rejections
        // silently rewriting the configured protocol.
        assert_eq!(
            reloaded.claude_protocol,
            Some(ClaudeProviderProtocol::Openai),
            "protocol must stay at the configured value when no 2xx was seen"
        );
    }

    #[tokio::test]
    async fn persist_runtime_discoveries_skips_path_variant_when_no_authoritative_response() {
        // Scenario: cascade exhausted with only terminal errors (e.g., 401
        // cross-protocol auth-shape). saw_authoritative stays false, no
        // path-variant should be written — we don't actually know if the
        // path responded or just rejected the auth header shape.
        use crate::services::ai_launcher::AIToolType;
        use crate::services::provider_protocol::{PathVariant, ProviderProtocol, encode_route};
        use crate::services::session_store::{ClaudeProviderProtocol, SessionStore};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU8};

        let temp = tempfile::tempdir().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let key_id = store
            .add_key_with_protocol(
                "test",
                "https://api.example.com",
                Some(ClaudeProviderProtocol::Openai),
                "sk-test",
            )
            .await
            .unwrap();
        let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

        let active = Arc::new(AtomicU8::new(encode_route(
            ProviderProtocol::Openai,
            PathVariant::Stripped,
        )));
        let succeeded = Arc::new(AtomicBool::new(false));
        let authoritative = Arc::new(AtomicBool::new(false));

        super::persist_runtime_discoveries(
            &store,
            AIToolType::Claude,
            &key,
            Some(active),
            None,
            Some(succeeded),
            Some(authoritative),
            None,
        )
        .await;

        let reloaded = store.get_key_by_id(&key_id).await.unwrap().unwrap();
        assert_eq!(reloaded.claude_path_variant, None);
    }

    #[tokio::test]
    async fn copy_pi_session_jsonl_tree_preserves_nested_history() {
        let temp = tempfile::tempdir().unwrap();
        let src = temp.path().join("src");
        let dst = temp.path().join("dst");
        let nested = src.join("--Users-alice-project-work-aivo--");
        tokio::fs::create_dir_all(&nested).await.unwrap();
        tokio::fs::write(nested.join("old.jsonl"), "{\"type\":\"session\"}\n")
            .await
            .unwrap();
        tokio::fs::write(nested.join("ignore.txt"), "nope")
            .await
            .unwrap();

        super::copy_pi_session_jsonl_tree(&src, &dst).await;

        let copied =
            tokio::fs::read_to_string(dst.join("--Users-alice-project-work-aivo--/old.jsonl"))
                .await
                .unwrap();
        assert_eq!(copied, "{\"type\":\"session\"}\n");
        assert!(
            !dst.join("--Users-alice-project-work-aivo--/ignore.txt")
                .exists()
        );
    }

    #[tokio::test]
    async fn seed_pi_sessions_exposes_prior_history_to_temp_agent_dir() {
        let temp = tempfile::tempdir().unwrap();
        let real_sessions = temp.path().join("real-sessions");
        let project_dir = real_sessions.join("--Users-alice-project-work-aivo--");
        tokio::fs::create_dir_all(&project_dir).await.unwrap();
        tokio::fs::write(project_dir.join("prior.jsonl"), "{\"type\":\"session\"}\n")
            .await
            .unwrap();

        let temp_agent = temp.path().join("temp-agent");
        tokio::fs::create_dir_all(&temp_agent).await.unwrap();

        super::seed_pi_sessions(Some(real_sessions), &temp_agent).await;

        let visible = tokio::fs::read_to_string(
            temp_agent.join("sessions/--Users-alice-project-work-aivo--/prior.jsonl"),
        )
        .await
        .unwrap();
        assert_eq!(visible, "{\"type\":\"session\"}\n");
    }

    fn aivo_models_only(api_key: &str) -> String {
        serde_json::json!({
            "providers": {
                "aivo": {
                    "baseUrl": "https://example.invalid",
                    "apiKey": api_key,
                    "api": "openai-completions",
                    "models": [{ "id": "m", "name": "m" }]
                }
            }
        })
        .to_string()
    }

    fn pi_env(models: &str) -> HashMap<String, String> {
        let mut env = HashMap::new();
        env.insert("AIVO_PI_MODELS_JSON".to_string(), models.to_string());
        env
    }

    fn temp_agent_dir_from(env: &HashMap<String, String>) -> std::path::PathBuf {
        std::path::PathBuf::from(
            env.get("PI_CODING_AGENT_DIR")
                .expect("PI_CODING_AGENT_DIR set"),
        )
    }

    /// Launches `write_pi_agent_dir_with_real` against a fresh `real` tempdir
    /// and returns the temp dir handle plus the resolved agent path.
    async fn launch_pi_agent(
        real: Option<&std::path::Path>,
    ) -> (HashMap<String, String>, std::path::PathBuf) {
        let mut env = pi_env(&aivo_models_only("k"));
        super::write_pi_agent_dir_with_real(&mut env, None, real)
            .await
            .unwrap();
        let agent = temp_agent_dir_from(&env);
        (env, agent)
    }

    #[tokio::test]
    async fn write_pi_agent_dir_symlinks_settings_and_auth() {
        // Mid-session writes must reach the real file, not vanish with the temp dir.
        let real = tempfile::tempdir().unwrap();
        let real_settings = "{\"packages\":[\"pi-subagents\"],\"defaultThinkingLevel\":\"high\"}";
        let real_auth = "{\"openai\":\"sk-real\"}";
        tokio::fs::write(real.path().join("settings.json"), real_settings)
            .await
            .unwrap();
        tokio::fs::write(real.path().join("auth.json"), real_auth)
            .await
            .unwrap();

        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("settings.json"))
                .await
                .unwrap(),
            real_settings
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("auth.json"))
                .await
                .unwrap(),
            real_auth
        );

        let updated = "{\"packages\":[\"pi-subagents\",\"pi-newpkg\"]}";
        tokio::fs::write(agent.join("settings.json"), updated)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            updated
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_persists_first_time_writes_back_to_real_home() {
        // First-time Pi use: real ~/.pi/agent/ exists, settings.json doesn't yet.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            "{}"
        );

        let installed = "{\"packages\":[\"pi-newpkg\"]}";
        tokio::fs::write(agent.join("settings.json"), installed)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("settings.json"))
                .await
                .unwrap(),
            installed
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_aivo_only_models() {
        // User's custom providers must NOT leak — otherwise --model anthropic/foo
        // silently bypasses the aivo key.
        let real = tempfile::tempdir().unwrap();
        let real_models = serde_json::json!({
            "providers": {
                "user-anthropic": { "baseUrl": "https://anthropic.example", "api": "anthropic-messages" }
            }
        })
        .to_string();
        tokio::fs::write(real.path().join("models.json"), &real_models)
            .await
            .unwrap();

        let mut env = pi_env(&aivo_models_only("fresh-key"));
        super::write_pi_agent_dir_with_real(&mut env, None, Some(real.path()))
            .await
            .unwrap();
        let agent = temp_agent_dir_from(&env);

        let written: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(agent.join("models.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(written["providers"]["aivo"].is_object());
        assert_eq!(written["providers"]["aivo"]["apiKey"], "fresh-key");
        assert!(written["providers"]["user-anthropic"].is_null());
    }

    #[tokio::test]
    async fn write_pi_agent_dir_persists_first_time_git_package_install() {
        // git-sourced packages land under <agentDir>/git/<host>/<path>/ — must
        // survive temp dir cleanup even on fresh ~/.pi/agent.
        let real = tempfile::tempdir().unwrap();
        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert!(real.path().join("git").is_dir());

        let installed = agent.join("git/github.com/example/pkg");
        tokio::fs::create_dir_all(&installed).await.unwrap();
        tokio::fs::write(installed.join("package.json"), "{\"name\":\"pkg\"}")
            .await
            .unwrap();

        assert_eq!(
            tokio::fs::read_to_string(real.path().join("git/github.com/example/pkg/package.json"))
                .await
                .unwrap(),
            "{\"name\":\"pkg\"}"
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_links_user_customization() {
        let real = tempfile::tempdir().unwrap();
        for d in ["rules", "tools", "prompts", "themes", "git"] {
            tokio::fs::create_dir_all(real.path().join(d))
                .await
                .unwrap();
        }
        tokio::fs::write(real.path().join("rules/lean-ctx.md"), "rule body")
            .await
            .unwrap();
        tokio::fs::write(real.path().join("mcp.json"), "{\"servers\":{}}")
            .await
            .unwrap();

        let (_env, agent) = launch_pi_agent(Some(real.path())).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("rules/lean-ctx.md"))
                .await
                .unwrap(),
            "rule body"
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("mcp.json"))
                .await
                .unwrap(),
            "{\"servers\":{}}"
        );
        for d in ["tools", "prompts", "themes", "git"] {
            assert!(agent.join(d).is_dir(), "{d} missing");
        }

        // mcp.json must persist mid-session writes back to the real home
        // — same persistence guarantee as settings.json. Regressions that
        // copy instead of link (e.g., dropping the symlink/hard-link
        // chain) would silently break `pi mcp add` during aivo sessions.
        let updated = "{\"servers\":{\"new\":{\"command\":\"x\"}}}";
        tokio::fs::write(agent.join("mcp.json"), updated)
            .await
            .unwrap();
        assert_eq!(
            tokio::fs::read_to_string(real.path().join("mcp.json"))
                .await
                .unwrap(),
            updated
        );
    }

    #[tokio::test]
    async fn write_pi_agent_dir_falls_back_when_no_real_home() {
        let (_env, agent) = launch_pi_agent(None).await;

        assert_eq!(
            tokio::fs::read_to_string(agent.join("settings.json"))
                .await
                .unwrap(),
            "{}"
        );
        assert_eq!(
            tokio::fs::read_to_string(agent.join("auth.json"))
                .await
                .unwrap(),
            "{}"
        );
        let models: serde_json::Value = serde_json::from_str(
            &tokio::fs::read_to_string(agent.join("models.json"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(models["providers"]["aivo"].is_object());
    }
}
