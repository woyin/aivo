//! Force-relogin helpers for the OAuth-backed key types
//! (codex / claude). Drives the same browser flow that
//! `aivo keys add` uses, persists the new credential into aivo's
//! store, and returns the refreshed `ApiKey` so the caller can hand
//! it straight to the launch path.
//!
//! Used by `aivo run <tool> --relogin` to recover from server-side
//! token revocation, single-use refresh-token races, or any other
//! state that leaves the stored credential dead.

use anyhow::{Result, anyhow};

use crate::services::session_store::{ApiKey, SessionStore};

/// Runs the OAuth re-login flow for `key`, persists the rotated
/// credential, and returns the updated `ApiKey`. Returns an error if
/// `key` isn't an OAuth-backed entry — REST API keys have nothing to
/// re-login.
pub async fn relogin_key(session_store: &SessionStore, key: &ApiKey) -> Result<ApiKey> {
    if key.is_codex_oauth() {
        let creds = crate::services::codex_oauth::interactive_login()
            .await
            .map_err(|e| e.context("codex re-login"))?;
        let json = creds.to_json()?;
        persist(session_store, key, &json).await
    } else if key.is_gemini_oauth() {
        Err(anyhow!(
            "Gemini OAuth sign-in has been removed — re-add '{}' with a Gemini API key (`aivo keys add`).",
            key.display_name()
        ))
    } else if key.is_claude_oauth() {
        let creds = crate::services::claude_oauth::spawn_setup_token_and_capture()
            .await
            .map_err(|e| anyhow!("claude re-login: {e}"))?;
        let json = creds.to_json()?;
        persist(session_store, key, &json).await
    } else if key.is_grok_oauth() {
        let creds = crate::services::grok_oauth::interactive_login()
            .await
            .map_err(|e| e.context("grok re-login"))?;
        let json = creds.to_json()?;
        persist(session_store, key, &json).await
    } else if key.is_kimi_oauth() {
        let creds = crate::services::kimi_oauth::interactive_login()
            .await
            .map_err(|e| e.context("kimi re-login"))?;
        let json = creds.to_json()?;
        persist(session_store, key, &json).await
    } else {
        Err(anyhow!(
            "--relogin only applies to OAuth keys (codex / claude / grok / kimi); '{}' is a plain API key",
            key.display_name()
        ))
    }
}

async fn persist(session_store: &SessionStore, key: &ApiKey, new_key_json: &str) -> Result<ApiKey> {
    let updated = session_store
        .update_key(
            &key.id,
            &key.name,
            &key.base_url,
            key.claude_protocol,
            new_key_json,
        )
        .await?;
    if !updated {
        return Err(anyhow!("key '{}' disappeared during re-login", key.id));
    }
    session_store
        .get_key_by_id(&key.id)
        .await?
        .ok_or_else(|| anyhow!("key '{}' disappeared after re-login persist", key.id))
}
