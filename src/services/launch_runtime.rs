use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::codex_home_shadow::{CodexHomeShadow, tokens_changed};
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

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER") {
        let (port, _active, responses_api, success, authoritative, learned) =
            start_responses_to_chat_router(&env).await?;
        responses_api_support = Some(responses_api);
        request_succeeded = Some(success);
        saw_authoritative_response = Some(authoritative);
        learned_requires_reasoning = Some(learned);
        set_local_base_url(&mut env, "OPENAI_BASE_URL", port);
    }

    if tool == AIToolType::Codex && env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER") {
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

    let pi_agent_dir = env.get("PI_CODING_AGENT_DIR").cloned();

    let codex_oauth_sync =
        if tool == AIToolType::Codex && env.contains_key("AIVO_CODEX_OAUTH_CREDS") {
            Some(prepare_codex_oauth_shadow(&mut env).await?)
        } else {
            None
        };

    let gemini_oauth_sync =
        if tool == AIToolType::Gemini && env.contains_key("AIVO_GEMINI_OAUTH_CREDS") {
            Some(prepare_gemini_oauth_shadow(&mut env).await?)
        } else {
            None
        };

    let gemini_system_settings =
        if tool == AIToolType::Gemini && env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH") {
            Some(prepare_gemini_api_key_settings_override(&mut env).await?)
        } else {
            None
        };

    Ok(LaunchRuntimeState {
        env,
        router_protocol,
        responses_api_support,
        request_succeeded,
        saw_authoritative_response,
        learned_requires_reasoning,
        pi_agent_dir,
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
async fn prepare_codex_oauth_shadow(env: &mut HashMap<String, String>) -> Result<CodexOAuthSync> {
    let raw = env
        .remove("AIVO_CODEX_OAUTH_CREDS")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_OAUTH_CREDS"))?;
    let key_id = env
        .remove("AIVO_CODEX_KEY_ID")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_CODEX_KEY_ID"))?;
    let mut creds = CodexOAuthCredential::from_json(&raw)?;

    // Refresh pre-launch so codex starts with a valid access token. If this
    // succeeds we DON'T persist here — the post-exit sync path will handle
    // it, picking up any further rotations codex may perform during the
    // session. If refresh fails the error surfaces to the user who must
    // re-run `aivo keys add codex`.
    let _refreshed = ensure_fresh(&mut creds, REFRESH_SKEW_SECS).await?;

    let shadow = CodexHomeShadow::create(&creds).await?;
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

    let updated = disk.into_credential(sync.original.email.clone(), sync.original.expires_at);
    persist_refreshed_if_needed(session_store, &sync.key_id, &sync.original, &updated).await;
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

/// Parses `AIVO_GEMINI_OAUTH_CREDS` (set by `environment_injector::for_gemini`
/// for Google OAuth keys), refreshes the access token if near expiry, and
/// writes a shadow `GEMINI_CLI_HOME` temp dir containing `.gemini/
/// oauth_creds.json` + `google_accounts.json`.
///
/// The `AIVO_*` placeholder vars are stripped before gemini is spawned; all
/// gemini sees is `GEMINI_CLI_HOME=<shadow>`, `GOOGLE_GENAI_USE_GCA=true`,
/// and `GEMINI_MODEL=<model>`.
async fn prepare_gemini_oauth_shadow(env: &mut HashMap<String, String>) -> Result<GeminiOAuthSync> {
    let raw = env
        .remove("AIVO_GEMINI_OAUTH_CREDS")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_GEMINI_OAUTH_CREDS"))?;
    let key_id = env
        .remove("AIVO_GEMINI_KEY_ID")
        .ok_or_else(|| anyhow::anyhow!("missing AIVO_GEMINI_KEY_ID"))?;
    let mut creds = GeminiOAuthCredential::from_json(&raw)?;

    // Refresh pre-launch so gemini starts with a valid access token. The
    // post-exit sync path persists both this refresh and any further
    // rotations gemini performs during the session, so we don't persist
    // here.
    let _refreshed = gemini_ensure_fresh(&mut creds, GEMINI_REFRESH_SKEW_SECS).await?;

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

    let dir = tempfile::Builder::new()
        .prefix("aivo-gemini-settings-")
        .tempdir()
        .context("create aivo gemini settings override temp dir")?;
    let path = dir.path().join("settings.json");
    tokio::fs::write(
        &path,
        br#"{"security":{"auth":{"selectedType":"gemini-api-key"}}}"#.as_slice(),
    )
    .await
    .context("write aivo gemini system settings override")?;

    env.insert(
        "GEMINI_CLI_SYSTEM_SETTINGS_PATH".to_string(),
        path.to_string_lossy().to_string(),
    );
    Ok(dir)
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

    // Reuse pi's managed bin/ (fd, rg) so pi doesn't re-download on each launch.
    if let Some(home) = crate::services::system_env::home_dir() {
        let real_bin = home.join(".pi").join("agent").join("bin");
        populate_pi_bin_dir(&real_bin, &dir.join("bin")).await;
    }

    env.insert(
        "PI_CODING_AGENT_DIR".to_string(),
        dir.to_string_lossy().to_string(),
    );
    Ok(())
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
    let anthropic_path_prefix = env
        .get("AIVO_ANTHROPIC_TO_OPENAI_ROUTER_ANTHROPIC_PATH_PREFIX")
        .cloned();
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
        anthropic_path_prefix,
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
        clear_node_proxy_env, patch_opencode_config_content,
        prepare_gemini_api_key_settings_override,
    };
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
        let mut env = HashMap::from([(
            "AIVO_GEMINI_FORCE_API_KEY_AUTH".to_string(),
            "1".to_string(),
        )]);
        let dir = prepare_gemini_api_key_settings_override(&mut env)
            .await
            .unwrap();

        // Sentinel consumed — must not leak to the spawned child.
        assert!(!env.contains_key("AIVO_GEMINI_FORCE_API_KEY_AUTH"));

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
}
