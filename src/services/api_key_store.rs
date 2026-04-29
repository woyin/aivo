use anyhow::{Context, Result};
use rand::Rng;
use std::collections::HashSet;
use zeroize::Zeroizing;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::session_crypto::{decrypt, encrypt, is_current_encryption, is_encrypted};
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, ConfigContext, GeminiProviderProtocol, OpenAICompatibilityMode,
    StoredConfig,
};

pub(crate) const KEY_ID_LENGTH: usize = 3;
pub(crate) const KEY_ID_ALPHABET: &[u8] = b"23456789abcdefghijkmnpqrstuvwxyz";

#[derive(Debug, Clone)]
pub(crate) struct ApiKeyStore {
    pub(crate) ctx: ConfigContext,
}

fn remove_runtime_state_for_key(config: &mut StoredConfig, key_id: &str) {
    config.chat_models.remove(key_id);
    for tools in config.directory_starts.values_mut() {
        tools.retain(|_, record| record.key_id != key_id);
    }
    config.directory_starts.retain(|_, tools| !tools.is_empty());
    if config
        .last_selection
        .as_ref()
        .is_some_and(|sel| sel.key_id == key_id)
    {
        config.last_selection = None;
    }
    // chat_sessions are now stored in individual files; file cleanup is handled
    // asynchronously by remove_sessions_for_key().
    config
        .chat_sessions
        .retain(|_, session| session.key_id != key_id);
}

pub(crate) fn generate_key_id(existing_ids: &HashSet<String>) -> Result<String> {
    let mut rng = rand::thread_rng();

    for _ in 0..1000 {
        let id: String = (0..KEY_ID_LENGTH)
            .map(|_| {
                let idx = rng.gen_range(0..KEY_ID_ALPHABET.len());
                KEY_ID_ALPHABET[idx] as char
            })
            .collect();

        if !existing_ids.contains(&id) {
            return Ok(id);
        }
    }

    anyhow::bail!(
        "Failed to generate unique key ID after 1000 attempts. Consider removing unused keys."
    );
}

impl ApiKeyStore {
    pub(crate) async fn add_key_with_protocol(
        &self,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<String> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;

        let existing_ids: HashSet<String> = config.api_keys.iter().map(|k| k.id.clone()).collect();
        let id = generate_key_id(&existing_ids)?;

        let mut new_key = ApiKey::new_with_protocol(
            id.clone(),
            name.to_string(),
            base_url.to_string(),
            claude_protocol,
            key.to_string(),
        );
        // Pre-encrypt the new key so save_raw can write it as-is
        new_key.key = Zeroizing::new(encrypt(&new_key.key)?);
        config.api_keys.push(new_key);

        // Save directly — existing keys are already encrypted in the raw config
        self.ctx.save_raw(&config).await?;
        Ok(id)
    }

    /// Gets all API keys without decrypting secrets.
    pub(crate) async fn get_keys(&self) -> Result<Vec<ApiKey>> {
        let config = self.ctx.load().await?;
        self.maybe_migrate_encryption(&config.api_keys).await;
        Ok(config.api_keys)
    }

    /// Re-encrypts any keys still using older encryption versions (v2/v3) to v4.
    async fn maybe_migrate_encryption(&self, keys: &[ApiKey]) {
        let needs_migration = keys
            .iter()
            .any(|k| is_encrypted(&k.key) && !is_current_encryption(&k.key));
        if !needs_migration {
            return;
        }

        let Ok(_lock) = self.ctx.acquire_config_lock() else {
            return;
        };
        let Ok(mut config) = self.ctx.load().await else {
            return;
        };

        let mut changed = false;
        for key in &mut config.api_keys {
            if is_encrypted(&key.key)
                && !is_current_encryption(&key.key)
                && let Ok(plaintext) = decrypt(&key.key)
                && let Ok(re_encrypted) = encrypt(&plaintext)
            {
                key.key = Zeroizing::new(re_encrypted);
                changed = true;
            }
        }

        if changed {
            let _ = self.ctx.save_raw(&config).await;
        }
    }

    /// Decrypts a single key's secret in place.
    pub(crate) fn decrypt_key_secret(key: &mut ApiKey) -> Result<()> {
        if is_encrypted(&key.key) {
            let plaintext = decrypt(&key.key)
                .with_context(|| format!("failed to decrypt key '{}'", key.display_name()))?;
            key.key = Zeroizing::new(plaintext);
        }
        Ok(())
    }

    /// Gets a specific API key by ID with its secret decrypted.
    pub(crate) async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        let keys = self.get_keys().await?;
        if let Some(mut key) = keys.into_iter().find(|k| k.id == id) {
            Self::decrypt_key_secret(&mut key)?;
            Ok(Some(key))
        } else {
            Ok(None)
        }
    }

    /// Deletes a key from config.json. Returns true if found and deleted.
    /// Caller is responsible for session file cleanup.
    pub(crate) async fn delete_key(&self, id: &str) -> Result<bool> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        let initial_len = config.api_keys.len();
        config.api_keys.retain(|k| k.id != id);

        if config.api_keys.len() < initial_len {
            if config.active_key_id.as_deref() == Some(id) {
                config.active_key_id = None;
            }
            remove_runtime_state_for_key(&mut config, id);
            self.ctx.save_raw(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Updates a key. Returns (found, base_url_changed).
    /// Caller is responsible for session file cleanup when base_url changes.
    pub(crate) async fn update_key(
        &self,
        id: &str,
        name: &str,
        base_url: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
        key: &str,
    ) -> Result<(bool, bool)> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        if let Some(entry) = config.api_keys.iter_mut().find(|k| k.id == id) {
            let base_url_changed = entry.base_url != base_url;
            entry.name = name.to_string();
            entry.base_url = base_url.to_string();
            entry.claude_protocol = claude_protocol;
            entry.key = Zeroizing::new(encrypt(key)?);
            if base_url_changed {
                remove_runtime_state_for_key(&mut config, id);
            }
            self.ctx.save_raw(&config).await?;
            Ok((true, base_url_changed))
        } else {
            Ok((false, false))
        }
    }

    async fn update_key_field(&self, id: &str, f: impl FnOnce(&mut ApiKey)) -> Result<bool> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        if let Some(entry) = config.api_keys.iter_mut().find(|k| k.id == id) {
            f(entry);
            self.ctx.save_raw(&config).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(crate) async fn set_key_claude_protocol(
        &self,
        id: &str,
        claude_protocol: Option<ClaudeProviderProtocol>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.claude_protocol = claude_protocol)
            .await
    }

    pub(crate) async fn set_key_gemini_protocol(
        &self,
        id: &str,
        gemini_protocol: Option<GeminiProviderProtocol>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.gemini_protocol = gemini_protocol)
            .await
    }

    pub(crate) async fn set_key_responses_api_supported(
        &self,
        id: &str,
        responses_api_supported: Option<bool>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| {
            entry.responses_api_supported = responses_api_supported
        })
        .await
    }

    pub(crate) async fn set_key_routing_schema_version(
        &self,
        id: &str,
        routing_schema_version: u32,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| {
            entry.routing_schema_version = routing_schema_version
        })
        .await
    }

    pub(crate) async fn set_key_claude_path_variant(
        &self,
        id: &str,
        claude_path_variant: Option<String>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.claude_path_variant = claude_path_variant)
            .await
    }

    pub(crate) async fn set_key_gemini_path_variant(
        &self,
        id: &str,
        gemini_path_variant: Option<String>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.gemini_path_variant = gemini_path_variant)
            .await
    }

    pub(crate) async fn set_key_requires_reasoning_content(
        &self,
        id: &str,
        requires_reasoning_content: Option<bool>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| {
            entry.requires_reasoning_content = requires_reasoning_content
        })
        .await
    }

    pub(crate) async fn set_key_codex_mode(
        &self,
        id: &str,
        codex_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.codex_mode = codex_mode)
            .await
    }

    pub(crate) async fn set_key_opencode_mode(
        &self,
        id: &str,
        opencode_mode: Option<OpenAICompatibilityMode>,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| entry.opencode_mode = opencode_mode)
            .await
    }

    pub(crate) async fn set_active_key(&self, id: &str) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;

        if !config.api_keys.iter().any(|k| k.id == id) {
            return Err(CLIError::new(
                format!("Key {} not found", id),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see available keys"),
            )
            .into());
        }

        config.active_key_id = Some(id.to_string());
        self.ctx.save_raw(&config).await
    }

    pub(crate) async fn resolve_key_by_id_or_name(&self, id_or_name: &str) -> Result<ApiKey> {
        let matches = self.find_keys_by_id_or_name(id_or_name).await?;
        match matches.len() {
            0 => Err(CLIError::new(
                format!("API key \"{}\" not found", id_or_name),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see available keys"),
            )
            .into()),
            1 => Ok(matches.into_iter().next().unwrap()),
            _ => Err(CLIError::new(
                format!(
                    "Multiple keys found with name \"{}\". Use the key ID instead.",
                    id_or_name
                ),
                ErrorCategory::User,
                None::<String>,
                Some("Run 'aivo keys' to see key IDs"),
            )
            .into()),
        }
    }

    /// Returns all keys matching `id_or_name` (decrypted). Exact/short ID
    /// always produces 0 or 1 matches; name matches may produce any number.
    /// Callers that want picker-on-ambiguity use this and branch on
    /// `matches.len()`.
    pub(crate) async fn find_keys_by_id_or_name(&self, id_or_name: &str) -> Result<Vec<ApiKey>> {
        let keys = self.get_keys().await?;

        if let Some(mut key) = keys
            .iter()
            .find(|k| k.id == id_or_name || k.short_id() == id_or_name)
            .cloned()
        {
            Self::decrypt_key_secret(&mut key)?;
            return Ok(vec![key]);
        }

        keys.into_iter()
            .filter(|k| k.name == id_or_name)
            .map(|mut k| {
                Self::decrypt_key_secret(&mut k)?;
                Ok(k)
            })
            .collect()
    }

    pub(crate) async fn get_active_key(&self) -> Result<Option<ApiKey>> {
        let config = self.ctx.load().await?;

        match config.active_key_id {
            Some(ref id) => {
                if let Some(mut key) = config.api_keys.into_iter().find(|k| k.id == *id) {
                    Self::decrypt_key_secret(&mut key)?;
                    Ok(Some(key))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    pub(crate) async fn get_keys_and_active_id_info(
        &self,
    ) -> Result<(Vec<ApiKey>, Option<String>)> {
        let config = self.ctx.load().await?;
        Ok((config.api_keys, config.active_key_id))
    }

    pub(crate) async fn get_active_key_info(&self) -> Result<Option<ApiKey>> {
        let config = self.ctx.load().await?;

        match config.active_key_id {
            Some(ref id) => Ok(config.api_keys.into_iter().find(|k| k.id == *id)),
            None => Ok(None),
        }
    }

    pub(crate) async fn get_chat_model(&self, key_id: &str) -> Result<Option<String>> {
        let config = self.ctx.load().await?;
        Ok(config.chat_models.get(key_id).cloned())
    }

    pub(crate) async fn set_chat_model(&self, key_id: &str, model: &str) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config
            .chat_models
            .insert(key_id.to_string(), model.to_string());
        self.ctx.save_raw(&config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::ConfigContext;
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> ApiKeyStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        ApiKeyStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    #[test]
    fn generate_key_id_produces_valid_ids() {
        let existing = HashSet::new();
        let id = generate_key_id(&existing).unwrap();
        assert_eq!(id.len(), KEY_ID_LENGTH);
        assert!(id.chars().all(|c| KEY_ID_ALPHABET.contains(&(c as u8))));
    }

    #[test]
    fn generate_key_id_avoids_collisions() {
        let mut existing = HashSet::new();
        // Generate several IDs and ensure no duplicates
        for _ in 0..50 {
            let id = generate_key_id(&existing).unwrap();
            assert!(!existing.contains(&id));
            existing.insert(id);
        }
    }

    #[tokio::test]
    async fn set_active_key_nonexistent_returns_error() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let result = store.set_active_key("nonexistent").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn chat_model_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        // No model set initially
        let model = store.get_chat_model("key1").await.unwrap();
        assert!(model.is_none());

        // Set and retrieve
        store.set_chat_model("key1", "gpt-4o").await.unwrap();
        let model = store.get_chat_model("key1").await.unwrap();
        assert_eq!(model.as_deref(), Some("gpt-4o"));

        // Overwrite
        store.set_chat_model("key1", "claude-sonnet").await.unwrap();
        let model = store.get_chat_model("key1").await.unwrap();
        assert_eq!(model.as_deref(), Some("claude-sonnet"));
    }

    #[tokio::test]
    async fn get_keys_and_active_id_info_returns_both() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        let (keys, active_id) = store.get_keys_and_active_id_info().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(active_id.as_deref(), Some(id.as_str()));
    }

    #[tokio::test]
    async fn get_active_key_info_returns_without_decryption() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        // No active key
        let info = store.get_active_key_info().await.unwrap();
        assert!(info.is_none());

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-secret")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();

        let info = store.get_active_key_info().await.unwrap().unwrap();
        assert_eq!(info.id, id);
        assert_eq!(info.name, "test");
        // Key should still be encrypted (not decrypted)
        assert!(is_encrypted(&info.key));
    }

    #[tokio::test]
    async fn delete_key_clears_chat_models() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store.set_chat_model(&id, "gpt-4o").await.unwrap();

        store.delete_key(&id).await.unwrap();

        let model = store.get_chat_model(&id).await.unwrap();
        assert!(model.is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_key_returns_false() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        assert!(!store.delete_key("nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn set_key_responses_api_supported_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        assert!(
            store
                .set_key_responses_api_supported(&id, Some(true))
                .await
                .unwrap()
        );

        let key = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(key.responses_api_supported, Some(true));
    }

    #[tokio::test]
    async fn update_key_returns_base_url_changed() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();

        // Same base_url — no change
        let (found, changed) = store
            .update_key(&id, "test", "http://localhost", None, "sk-new")
            .await
            .unwrap();
        assert!(found);
        assert!(!changed);

        // Different base_url — changed
        let (found, changed) = store
            .update_key(&id, "test", "http://new-host", None, "sk-new")
            .await
            .unwrap();
        assert!(found);
        assert!(changed);
    }
}
