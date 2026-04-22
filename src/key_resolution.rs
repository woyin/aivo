use std::io::{self, IsTerminal};
use std::process;

use crate::commands;
use crate::errors::ExitCode;
use crate::services::key_compat::KeyCompatContext;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

#[allow(clippy::large_enum_variant)]
pub(crate) enum KeyResolution {
    Selected(ApiKey),
    Cancelled,
    MissingAuth,
}

pub(crate) enum KeyLookupMode {
    RequireActiveOrPrompt,
    PreferActiveAllowNone,
}

pub(crate) fn key_or_exit(result: anyhow::Result<KeyResolution>) -> Option<ApiKey> {
    match result {
        Ok(KeyResolution::Selected(key)) => Some(key),
        Ok(KeyResolution::Cancelled) => process::exit(ExitCode::Success.code()),
        Ok(KeyResolution::MissingAuth) => process::exit(ExitCode::AuthError.code()),
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            process::exit(ExitCode::UserError.code());
        }
    }
}

pub(crate) async fn resolve_key_override(
    session_store: &SessionStore,
    key_flag: Option<&str>,
    mode: KeyLookupMode,
    compat: KeyCompatContext,
) -> anyhow::Result<KeyResolution> {
    match key_flag {
        Some("") => prompt_temporary_key_override(session_store, compat).await,
        Some(key_id_or_name) => resolve_by_id_or_name_or_pick(session_store, key_id_or_name).await,
        None => match mode {
            KeyLookupMode::RequireActiveOrPrompt => {
                match resolve_active_key_or_prompt(session_store, compat).await {
                    Some(key) => Ok(KeyResolution::Selected(key)),
                    None => Ok(KeyResolution::MissingAuth),
                }
            }
            KeyLookupMode::PreferActiveAllowNone => {
                // Try last-used selection first
                if let Ok(Some(last_sel)) = session_store.get_last_selection().await
                    && let Ok(Some(key)) = session_store.get_key_by_id(&last_sel.key_id).await
                {
                    return Ok(KeyResolution::Selected(key));
                }
                match session_store.get_active_key().await? {
                    Some(key) => Ok(KeyResolution::Selected(key)),
                    None => Ok(KeyResolution::MissingAuth),
                }
            }
        },
    }
}

/// Resolves `key_id_or_name` to a single key. Shows a picker on ambiguous
/// name matches when a terminal is available; falls back to the low-level
/// error otherwise so scripts/CI still get a clear failure.
pub(crate) async fn resolve_by_id_or_name_or_pick(
    session_store: &SessionStore,
    key_id_or_name: &str,
) -> anyhow::Result<KeyResolution> {
    let matches = session_store
        .find_keys_by_id_or_name(key_id_or_name)
        .await?;
    match matches.len() {
        0 => {
            // Delegate to the existing error path for consistent messaging.
            Err(session_store
                .resolve_key_by_id_or_name(key_id_or_name)
                .await
                .expect_err("empty matches must produce a not-found error"))
        }
        1 => Ok(KeyResolution::Selected(matches.into_iter().next().unwrap())),
        _ => {
            if !io::stderr().is_terminal() {
                return Err(session_store
                    .resolve_key_by_id_or_name(key_id_or_name)
                    .await
                    .expect_err("ambiguous matches must produce an error"));
            }
            eprintln!(
                "{} Multiple keys match {}:",
                style::yellow("Note:"),
                style::cyan(key_id_or_name)
            );
            let prompt = format!("Select key '{}'", key_id_or_name);
            match commands::keys::prompt_pick_key_without_activation(&matches, &[], &prompt, 0)? {
                Some(key) => Ok(KeyResolution::Selected(key)),
                None => Ok(KeyResolution::Cancelled),
            }
        }
    }
}

async fn prompt_temporary_key_override(
    session_store: &SessionStore,
    compat: KeyCompatContext,
) -> anyhow::Result<KeyResolution> {
    let all_keys = session_store.get_keys().await?;
    if all_keys.is_empty() {
        eprintln!("{} No API keys configured.", style::yellow("Note:"));
        eprintln!();
        eprintln!("  Run {} to add one.", style::cyan("aivo keys add"));
        return Ok(KeyResolution::MissingAuth);
    }
    if !io::stderr().is_terminal() {
        anyhow::bail!(
            "Cannot open key picker without a terminal. Run in a terminal or pass --key <id|name>."
        );
    }

    let last_sel_key_id = session_store
        .get_last_selection()
        .await
        .ok()
        .flatten()
        .map(|s| s.key_id);
    let active_key_id = session_store
        .get_active_key_info()
        .await
        .ok()
        .flatten()
        .map(|k| k.id);
    let default_idx = last_sel_key_id
        .as_ref()
        .and_then(|id| all_keys.iter().position(|key| &key.id == id))
        .or_else(|| {
            active_key_id
                .as_ref()
                .and_then(|id| all_keys.iter().position(|key| &key.id == id))
        })
        .unwrap_or(0);

    let annotations = compat.annotations_for(&all_keys);
    match commands::keys::prompt_pick_key_without_activation(
        &all_keys,
        &annotations,
        "Select a key",
        default_idx,
    )? {
        Some(key) => Ok(KeyResolution::Selected(key)),
        None => Ok(KeyResolution::Cancelled),
    }
}

async fn resolve_active_key_or_prompt(
    session_store: &SessionStore,
    compat: KeyCompatContext,
) -> Option<ApiKey> {
    // Try last-used selection first
    if let Ok(Some(last_sel)) = session_store.get_last_selection().await
        && let Ok(Some(key)) = session_store.get_key_by_id(&last_sel.key_id).await
    {
        return Some(key);
    }
    // Then active key
    if let Ok(Some(key)) = session_store.get_active_key().await {
        return Some(key);
    }

    let all_keys = match session_store.get_keys().await {
        Ok(keys) => keys,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            return None;
        }
    };

    if all_keys.is_empty() {
        eprintln!("{} No API keys configured.", style::yellow("Note:"));
        eprintln!();
        eprintln!("  Run {} to add one.", style::cyan("aivo keys add"));
        return None;
    }

    eprintln!(
        "{} No active API key. Select one to continue:",
        style::yellow("Note:")
    );
    eprintln!();

    if !io::stderr().is_terminal() {
        eprintln!(
            "{} Cannot open key picker without a terminal. Run in a terminal or activate a key first.",
            style::red("Error:")
        );
        return None;
    }

    let annotations = compat.annotations_for(&all_keys);
    match commands::keys::prompt_select_key(
        session_store,
        &all_keys,
        &annotations,
        "Select a key",
        0,
    )
    .await
    {
        Ok(Some(key)) => {
            eprintln!();
            Some(key)
        }
        Ok(None) => {
            eprintln!("{}", style::dim("Cancelled."));
            None
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KeyCompatContext, KeyLookupMode, KeyResolution, resolve_key_override};
    use crate::services::session_store::SessionStore;
    use tempfile::TempDir;

    fn temp_store() -> (TempDir, SessionStore) {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        (temp_dir, SessionStore::with_path(config_path))
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_active_key() {
        let (_temp_dir, store) = temp_store();
        let id = store
            .add_key_with_protocol(
                "openrouter",
                "https://openrouter.ai/api/v1",
                None,
                "sk-test",
            )
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        let resolved = resolve_key_override(
            &store,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        match resolved.unwrap() {
            KeyResolution::Selected(key) => assert_eq!(key.id, id),
            _ => panic!("expected selected key"),
        }
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_missing_auth_without_keys() {
        let (_temp_dir, store) = temp_store();

        // resolve_key_override alone doesn't create the starter key;
        // that's handled by main.rs before dispatching commands.
        let resolved = resolve_key_override(
            &store,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        assert!(matches!(resolved.unwrap(), KeyResolution::MissingAuth));
    }

    #[tokio::test]
    async fn prefer_active_allow_none_returns_starter_after_ensure() {
        let (_temp_dir, store) = temp_store();

        // Simulate main.rs: ensure + activate for new users
        let (starter, _) = store.ensure_starter_key().await.unwrap();
        store.set_active_key(&starter.id).await.unwrap();

        let resolved = resolve_key_override(
            &store,
            None,
            KeyLookupMode::PreferActiveAllowNone,
            KeyCompatContext::None,
        )
        .await;

        match resolved.unwrap() {
            KeyResolution::Selected(key) => {
                assert_eq!(key.name, crate::constants::AIVO_STARTER_KEY_NAME);
                assert_eq!(key.base_url, crate::constants::AIVO_STARTER_SENTINEL);
            }
            _ => panic!("expected starter key"),
        }
    }

    #[tokio::test]
    async fn ensure_starter_key_creates_and_is_idempotent() {
        let (_temp_dir, store) = temp_store();

        let (key, is_new) = store
            .ensure_starter_key()
            .await
            .expect("should create starter key");
        assert_eq!(key.name, crate::constants::AIVO_STARTER_KEY_NAME);
        assert_eq!(key.base_url, crate::constants::AIVO_STARTER_SENTINEL);
        assert!(is_new);

        // Verify chat model was pre-set
        let model = store.get_chat_model(&key.id).await.unwrap();
        assert_eq!(
            model,
            Some(crate::constants::AIVO_STARTER_MODEL.to_string())
        );

        // Calling again returns the same key (idempotent)
        let (key2, is_new2) = store
            .ensure_starter_key()
            .await
            .expect("should return existing starter key");
        assert_eq!(key.id, key2.id);
        assert!(!is_new2);

        // Only one key exists
        let all_keys = store.get_keys().await.unwrap();
        assert_eq!(all_keys.len(), 1);
    }
}
