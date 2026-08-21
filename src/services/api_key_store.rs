use anyhow::Result;

use std::collections::{BTreeMap, HashSet};
use zeroize::Zeroizing;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::os_keyring;
use crate::services::route_cache::PersistedRoute;
use crate::services::session_crypto::{
    V5_ENCRYPTION_MARKER, decrypt, encrypt, is_current_encryption, is_encrypted,
    would_rewrite_encryption,
};
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, ConfigContext, GeminiProviderProtocol, OpenAICompatibilityMode,
    StoredConfig,
};

/// Policy applied when an imported record conflicts with an existing one.
/// Conflict = matching `(name, base_url)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportPolicy {
    Overwrite,
    /// Insert the conflict under a fresh id and suffix the name with " (imported)".
    Rename,
    Skip,
}

#[derive(Debug, Default)]
pub struct ImportReport {
    pub imported: Vec<String>,
    pub overwritten: Vec<String>,
    pub renamed: Vec<(String, String)>,
    pub skipped: Vec<String>,
    /// Starter rows dropped from the incoming records (device-bound).
    pub skipped_starter: usize,
    /// Login-session rows dropped — old export files may still carry them.
    pub skipped_oauth: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ExportFilterReport {
    pub skipped_starter: usize,
    pub skipped_oauth: usize,
}

pub(crate) const KEY_ID_LENGTH: usize = 3;
pub(crate) const KEY_ID_ALPHABET: &[u8] = b"23456789abcdefghijkmnpqrstuvwxyz";

#[derive(Debug, Clone)]
pub(crate) struct ApiKeyStore {
    pub(crate) ctx: ConfigContext,
}

fn remove_runtime_state_for_key(config: &mut StoredConfig, key_id: &str) {
    config.code_models.remove(key_id);
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

/// Write-path guard: when stored v5 values exist but the keyring item is
/// gone, encrypting anything would mint a FRESH master secret — splitting
/// the store across two secrets and masking the lockout. Fail loudly
/// instead. A present-but-mismatched secret is deliberately NOT blocked:
/// re-entering values via add/edit is the recovery path.
fn refuse_masked_lockout(keys: &[ApiKey]) -> Result<()> {
    if !os_keyring::keyring_enabled() || !os_keyring::master_secret_absent() {
        return Ok(());
    }
    let locked = keys
        .iter()
        .filter(|k| k.key.starts_with(V5_ENCRYPTION_MARKER))
        .count();
    if locked == 0 {
        return Ok(());
    }
    Err(anyhow::Error::new(CLIError::new(
        format!(
            "refusing to write: {locked} stored key(s) are encrypted with a keyring \
             master secret that no longer exists"
        ),
        ErrorCategory::Auth,
        Some(
            "writing now would create a fresh \"aivo / master-secret\" keyring item \
             and silently split the store across two secrets",
        ),
        Some(
            "restore the keyring item from a backup, or remove the locked keys \
             (`aivo keys rm <name>`) and re-add them",
        ),
    )))
}

/// Re-encrypts keys on older encryption versions in place; returns whether
/// anything changed. Caller must hold the config lock: re-encrypt may CREATE
/// the keyring master secret.
fn migrate_keys_in_place(keys: &mut [ApiKey]) -> bool {
    let mut changed = false;
    for key in keys.iter_mut() {
        if is_encrypted(&key.key)
            && !is_current_encryption(&key.key)
            && let Ok(plaintext) = decrypt(&key.key)
            && let Ok(re_encrypted) = encrypt(&plaintext)
        {
            key.key = Zeroizing::new(re_encrypted);
            changed = true;
        }
    }
    changed
}

pub(crate) fn generate_key_id(existing_ids: &HashSet<String>) -> Result<String> {
    for _ in 0..1000 {
        let id = super::rng::string_from(KEY_ID_ALPHABET, KEY_ID_LENGTH);

        if !existing_ids.contains(&id) {
            return Ok(id);
        }
    }

    anyhow::bail!(
        "Failed to generate unique key ID after 1000 attempts. Consider removing unused keys."
    );
}

/// Exact id / short-id is a single match, else every exact-name match.
/// The one source of the id-vs-name precedence rule.
pub(crate) fn match_keys_by_id_or_name<'a>(
    keys: &'a [ApiKey],
    id_or_name: &str,
) -> Vec<&'a ApiKey> {
    if let Some(key) = keys
        .iter()
        .find(|k| k.id == id_or_name || k.short_id() == id_or_name)
    {
        return vec![key];
    }
    keys.iter().filter(|k| k.name == id_or_name).collect()
}

/// Account-bound login sessions: OAuth sentinels, Copilot, Cursor browser
/// login. Cursor needs a decrypt probe — a plain Cursor API key shares the
/// base sentinel.
pub(crate) fn is_login_session_record(key: &ApiKey) -> bool {
    use crate::services::provider_profile::is_oauth_or_copilot_base;
    if is_oauth_or_copilot_base(&key.base_url) {
        return true;
    }
    if crate::services::cursor_acp::is_cursor_acp_base(&key.base_url) {
        let mut probe = key.clone();
        return ApiKeyStore::decrypt_key_secret(&mut probe).is_ok()
            && is_login_session_plaintext(&probe);
    }
    false
}

/// Same test for records whose secret is already plaintext (import payloads).
pub(crate) fn is_login_session_plaintext(key: &ApiKey) -> bool {
    crate::services::provider_profile::is_oauth_or_copilot_base(&key.base_url)
        || (crate::services::cursor_acp::is_cursor_acp_base(&key.base_url)
            && crate::services::cursor_acp::cursor_account_id(key).is_some())
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
        refuse_masked_lockout(&config.api_keys)?;

        // Migrate existing keys too so a version bump doesn't leave a mixed store.
        if migrate_keys_in_place(&mut config.api_keys) {
            self.backup_before_encryption_rewrite().await;
        }

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

    /// The starter key never exports — the real credential is the
    /// per-install device key (`secrets/device-key`), so the record is dead
    /// weight elsewhere. Login sessions never export either (account-bound).
    /// `ids` match by full id, short id, or exact name; ambiguous names
    /// error.
    pub(crate) async fn export_keys(
        &self,
        ids: Option<&[String]>,
    ) -> Result<(Vec<ApiKey>, ExportFilterReport)> {
        use crate::services::provider_profile::is_aivo_starter_base;

        let keys = self.get_keys().await?;

        let mut selected: Vec<ApiKey> = if let Some(filter) = ids {
            let mut missing = Vec::new();
            let mut found: Vec<ApiKey> = Vec::new();
            for needle in filter {
                let matches = match_keys_by_id_or_name(&keys, needle);
                if matches.len() > 1 {
                    return Err(anyhow::anyhow!(
                        "Key name \"{}\" is ambiguous — matches ids {}. Use an id instead.",
                        needle,
                        matches
                            .iter()
                            .map(|k| k.short_id())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
                match matches.first() {
                    // The same key named twice (e.g. by id and by name) exports once.
                    Some(k) if !found.iter().any(|f| f.id == k.id) => found.push((*k).clone()),
                    Some(_) => {}
                    None => missing.push(needle.clone()),
                }
            }
            if !missing.is_empty() {
                return Err(anyhow::anyhow!(
                    "Unknown key id or name: {}. Run `aivo keys list` to see keys.",
                    missing.join(", ")
                ));
            }
            found
        } else {
            keys
        };

        let mut report = ExportFilterReport::default();
        let before = selected.len();
        selected.retain(|k| !is_aivo_starter_base(&k.base_url));
        report.skipped_starter = before - selected.len();
        let before = selected.len();
        selected.retain(|key| !is_login_session_record(key));
        report.skipped_oauth = before - selected.len();

        for key in &mut selected {
            Self::decrypt_key_secret(key)?;
        }
        Ok((selected, report))
    }

    /// Records' file IDs are always discarded and replaced with fresh
    /// local-alphabet IDs. Prevents 3-char id collisions from silently
    /// overwriting unrelated local keys, and keeps adversarial non-ASCII
    /// IDs out of storage where `short_id()`'s byte-slice would panic.
    pub(crate) async fn import_keys(
        &self,
        records: Vec<ApiKey>,
        policy: ImportPolicy,
    ) -> Result<ImportReport> {
        use crate::services::provider_profile::is_aivo_starter_base;

        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        refuse_masked_lockout(&config.api_keys)?;
        // Same as add: migrate existing keys while we hold the lock anyway.
        if migrate_keys_in_place(&mut config.api_keys) {
            self.backup_before_encryption_rewrite().await;
        }
        let mut report = ImportReport::default();

        for mut incoming in records {
            // Starter rows are dead off their origin machine; old export
            // files may still contain them.
            if is_aivo_starter_base(&incoming.base_url) {
                report.skipped_starter += 1;
                continue;
            }
            // Old export files may still carry login rows.
            if is_login_session_plaintext(&incoming) {
                report.skipped_oauth += 1;
                continue;
            }
            let source_id = incoming.id.clone();
            let conflict_idx = config
                .api_keys
                .iter()
                .position(|k| k.name == incoming.name && k.base_url == incoming.base_url);

            incoming.key = Zeroizing::new(encrypt(&incoming.key)?);

            match conflict_idx {
                None => {
                    let existing_ids: HashSet<String> =
                        config.api_keys.iter().map(|k| k.id.clone()).collect();
                    incoming.id = generate_key_id(&existing_ids)?;
                    report.imported.push(incoming.id.clone());
                    config.api_keys.push(incoming);
                }
                Some(idx) => match policy {
                    ImportPolicy::Overwrite => {
                        let existing_id = config.api_keys[idx].id.clone();
                        remove_runtime_state_for_key(&mut config, &existing_id);
                        incoming.id = existing_id.clone();
                        config.api_keys[idx] = incoming;
                        report.overwritten.push(existing_id);
                    }
                    ImportPolicy::Rename => {
                        let existing_ids: HashSet<String> =
                            config.api_keys.iter().map(|k| k.id.clone()).collect();
                        incoming.id = generate_key_id(&existing_ids)?;
                        if !incoming.name.is_empty() {
                            incoming.name = format!("{} (imported)", incoming.name);
                        }
                        report.renamed.push((source_id, incoming.id.clone()));
                        config.api_keys.push(incoming);
                    }
                    ImportPolicy::Skip => {
                        report.skipped.push(source_id);
                    }
                },
            }
        }

        self.ctx.save_raw(&config).await?;
        Ok(report)
    }

    /// Gets all API keys without decrypting secrets.
    pub(crate) async fn get_keys(&self) -> Result<Vec<ApiKey>> {
        let config = self.ctx.load().await?;
        self.maybe_migrate_encryption(&config.api_keys).await;
        Ok(config.api_keys)
    }

    /// Re-encrypts any keys still using an older encryption version to the
    /// current one (v5 when the OS keyring is active, otherwise v4).
    async fn maybe_migrate_encryption(&self, keys: &[ApiKey]) {
        // Lock-free pre-flight only — `is_current_encryption` can CREATE the
        // keyring master secret, which must stay serialized by the config
        // lock (racing first-runs would brick keys on Linux/Windows, where
        // the keyring store primitives overwrite).
        let needs_migration = keys.iter().any(|k| would_rewrite_encryption(&k.key));
        if !needs_migration {
            return;
        }

        let Ok(_lock) = self.ctx.acquire_config_lock() else {
            return;
        };
        let Ok(mut config) = self.ctx.load().await else {
            return;
        };
        // Read path: skip silently rather than error the whole listing; the
        // lockout itself surfaces loudly when any v5 value is decrypted.
        if refuse_masked_lockout(&config.api_keys).is_err() {
            return;
        }

        if migrate_keys_in_place(&mut config.api_keys) {
            self.backup_before_encryption_rewrite().await;
            let _ = self.ctx.save_raw(&config).await;
        }
    }

    /// Snapshots config.json beside itself before a whole-store re-encryption
    /// is saved over it. Callers hold the config lock with the rewrite still
    /// unsaved, so the on-disk file is the pre-migration store. Non-fatal:
    /// the migration matters more than its safety net.
    async fn backup_before_encryption_rewrite(&self) {
        let src = &self.ctx.config_path;
        let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
        let dst = src.with_file_name(format!("config.json.pre-migration-{stamp}.bak"));
        if let Err(e) = tokio::fs::copy(src, &dst).await
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("Warning: config backup before encryption migration failed: {e}");
        }
    }

    /// Decrypts a single key's secret in place.
    pub(crate) fn decrypt_key_secret(key: &mut ApiKey) -> Result<()> {
        if is_encrypted(&key.key) {
            let plaintext = decrypt(&key.key).map_err(|e| {
                // A CLIError carries its own actionable message + suggestion;
                // wrapping would hide it (top-level display is outermost only)
                // — rebuild it with the key name folded into the message.
                match e.downcast_ref::<CLIError>() {
                    Some(cli) => anyhow::Error::new(
                        cli.with_message_prefix(&format!("key '{}': ", key.display_name())),
                    ),
                    None => e.context(format!("failed to decrypt key '{}'", key.display_name())),
                }
            })?;
            key.key = Zeroizing::new(plaintext);
        }
        Ok(())
    }

    /// Gets a specific API key by ID with its secret decrypted.
    pub(crate) async fn get_key_by_id(&self, id: &str) -> Result<Option<ApiKey>> {
        let mut key = match self.get_key_by_id_info(id).await? {
            Some(k) => k,
            None => return Ok(None),
        };
        Self::decrypt_key_secret(&mut key)?;
        Ok(Some(key))
    }

    /// Gets a specific API key by ID without decrypting its secret.
    pub(crate) async fn get_key_by_id_info(&self, id: &str) -> Result<Option<ApiKey>> {
        let keys = self.get_keys().await?;
        Ok(keys.into_iter().find(|k| k.id == id))
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
        refuse_masked_lockout(&config.api_keys)?;
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

    /// Merge learned routes for one tool under the config lock — each
    /// `(model, route)` overwrites only its own entry, so a concurrent `serve`
    /// and `run` can't clobber each other via the wholesale `save_raw`.
    pub(crate) async fn merge_routes(
        &self,
        id: &str,
        tool: &str,
        routes: &[(String, PersistedRoute)],
    ) -> Result<bool> {
        if routes.is_empty() {
            return Ok(true);
        }
        self.update_key_field(id, |entry| {
            let tool_map = entry.protocol_routes.entry(tool.to_string()).or_default();
            for (model, route) in routes {
                tool_map.insert(model.clone(), route.clone());
            }
        })
        .await
    }

    /// Drop all learned per-model routes for a key (the `reset-route` flow), so
    /// the next launch re-learns from the tool-native protocol.
    pub(crate) async fn clear_protocol_routes(&self, id: &str) -> Result<bool> {
        self.update_key_field(id, |entry| entry.protocol_routes.clear())
            .await
    }

    /// One-shot schema-v2 write under one lock: install the caller's migrated
    /// per-tool routes (existing entries win), drop the scalar pins, stamp the
    /// version.
    pub(crate) async fn migrate_key_to_routes_v2(
        &self,
        id: &str,
        migrated: BTreeMap<String, BTreeMap<String, PersistedRoute>>,
        version: u32,
    ) -> Result<bool> {
        self.update_key_field(id, |entry| {
            for (tool, models) in migrated {
                let dst = entry.protocol_routes.entry(tool).or_default();
                for (model, route) in models {
                    dst.entry(model).or_insert(route);
                }
            }
            entry.claude_protocol = None;
            entry.gemini_protocol = None;
            entry.responses_api_supported = None;
            entry.claude_path_variant = None;
            entry.gemini_path_variant = None;
            entry.routing_schema_version = version;
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
        let mut matches = self.find_keys_by_id_or_name_info(id_or_name).await?;
        for key in &mut matches {
            Self::decrypt_key_secret(key)?;
        }
        Ok(matches)
    }

    /// Like `find_keys_by_id_or_name` but skips PBKDF2 decryption — the
    /// returned `ApiKey.key` may still hold the encrypted ciphertext.
    /// Use when only metadata (id, name, base_url) is needed; callers that
    /// later need the secret can decrypt on demand.
    pub(crate) async fn find_keys_by_id_or_name_info(
        &self,
        id_or_name: &str,
    ) -> Result<Vec<ApiKey>> {
        let keys = self.get_keys().await?;
        Ok(match_keys_by_id_or_name(&keys, id_or_name)
            .into_iter()
            .cloned()
            .collect())
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

    pub(crate) async fn get_code_model(&self, key_id: &str) -> Result<Option<String>> {
        let config = self.ctx.load().await?;
        Ok(config.code_models.get(key_id).cloned())
    }

    pub(crate) async fn set_code_model(&self, key_id: &str, model: &str) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config
            .code_models
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

    #[tokio::test]
    async fn add_refuses_fresh_secret_when_v5_values_locked_out() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        os_keyring::test_state::set(true, Some([7u8; os_keyring::SECRET_LEN]));
        store
            .add_key_with_protocol("first", "https://api.example.com/v1", None, "sk-first")
            .await
            .unwrap();

        // The keyring item vanished (deleted/re-created elsewhere): writes
        // must refuse to mint a fresh secret over the undecryptable v5 store.
        os_keyring::test_state::set(true, None);
        let err = store
            .add_key_with_protocol("second", "https://api.example.com/v1", None, "sk-second")
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("master secret"), "unexpected error: {err}");

        // Explicit keyring opt-out still writes (v4 fallback), no fresh mint.
        os_keyring::test_state::set(false, None);
        store
            .add_key_with_protocol("third", "https://api.example.com/v1", None, "sk-third")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn whole_store_migration_backs_up_config() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        os_keyring::test_state::set(false, None);
        store
            .add_key_with_protocol("legacy", "https://api.example.com/v1", None, "sk-legacy")
            .await
            .unwrap();

        // Keyring appears: the next write migrates the whole store to v5 and
        // must snapshot the pre-migration file first.
        os_keyring::test_state::set(true, Some([9u8; os_keyring::SECRET_LEN]));
        store
            .add_key_with_protocol("fresh", "https://api.example.com/v1", None, "sk-fresh")
            .await
            .unwrap();

        let backups: Vec<_> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("config.json.pre-migration-")
            })
            .collect();
        assert_eq!(
            backups.len(),
            1,
            "expected exactly one pre-migration backup"
        );
        let backup_text = std::fs::read_to_string(backups[0].path()).unwrap();
        assert!(backup_text.contains("enc4:"));
        assert!(!backup_text.contains("enc5:"));
    }

    #[test]
    fn decrypt_error_keeps_key_name_with_cli_error() {
        os_keyring::test_state::set(true, Some([1u8; os_keyring::SECRET_LEN]));
        let enc = encrypt("sk-x").unwrap();
        let mut key = ApiKey::new_with_protocol(
            "abc".to_string(),
            "mykey".to_string(),
            "https://api.example.com/v1".to_string(),
            None,
            enc,
        );
        os_keyring::test_state::set(true, Some([2u8; os_keyring::SECRET_LEN]));
        let err = ApiKeyStore::decrypt_key_secret(&mut key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mykey"), "unexpected error: {err}");
        assert!(err.contains("does not match"), "unexpected error: {err}");
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
        let model = store.get_code_model("key1").await.unwrap();
        assert!(model.is_none());

        // Set and retrieve
        store.set_code_model("key1", "gpt-4o").await.unwrap();
        let model = store.get_code_model("key1").await.unwrap();
        assert_eq!(model.as_deref(), Some("gpt-4o"));

        // Overwrite
        store.set_code_model("key1", "claude-sonnet").await.unwrap();
        let model = store.get_code_model("key1").await.unwrap();
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
    async fn delete_key_clears_code_models() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id = store
            .add_key_with_protocol("test", "http://localhost", None, "sk-test")
            .await
            .unwrap();
        store.set_code_model(&id, "gpt-4o").await.unwrap();

        store.delete_key(&id).await.unwrap();

        let model = store.get_code_model(&id).await.unwrap();
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

    #[tokio::test]
    async fn export_keys_returns_plaintext_secrets() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();
        store
            .add_key_with_protocol("beta", "http://b", None, "sk-beta")
            .await
            .unwrap();

        let (exported, _) = store.export_keys(None).await.unwrap();
        assert_eq!(exported.len(), 2);
        for key in &exported {
            assert!(!is_encrypted(&key.key), "exported secret must be plaintext");
        }
        let secrets: Vec<&str> = exported.iter().map(|k| k.key.as_str()).collect();
        assert!(secrets.contains(&"sk-alpha"));
        assert!(secrets.contains(&"sk-beta"));
    }

    #[tokio::test]
    async fn export_always_skips_aivo_starter() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        store
            .add_key_with_protocol("aivo", "aivo-starter", None, "starter-token")
            .await
            .unwrap();
        store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();

        let (exported, report) = store.export_keys(None).await.unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].name, "alpha");
        assert_eq!(report.skipped_starter, 1);

        // Even an explicit --ids selection drops the starter row.
        let (by_id, report) = store
            .export_keys(Some(&["aivo".to_string(), "alpha".to_string()]))
            .await
            .unwrap();
        assert_eq!(by_id.len(), 1);
        assert_eq!(by_id[0].name, "alpha");
        assert_eq!(report.skipped_starter, 1);
    }

    #[tokio::test]
    async fn import_drops_starter_rows() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let records = vec![
            ApiKey::new_with_protocol(
                "aa1".into(),
                "aivo".into(),
                "aivo-starter".into(),
                None,
                "starter-token".into(),
            ),
            ApiKey::new_with_protocol(
                "bb2".into(),
                "alpha".into(),
                "http://a".into(),
                None,
                "sk-alpha".into(),
            ),
        ];

        let report = store
            .import_keys(records, ImportPolicy::Skip)
            .await
            .unwrap();
        assert_eq!(report.skipped_starter, 1);
        assert_eq!(report.imported.len(), 1);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "alpha");
    }

    #[tokio::test]
    async fn export_always_skips_login_sessions() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
        use crate::services::cursor_acp::{CURSOR_ACP_SENTINEL, CURSOR_SHADOW_PREFIX};
        use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;
        use crate::services::grok_oauth::GROK_OAUTH_SENTINEL;
        use crate::services::kimi_oauth::KIMI_OAUTH_SENTINEL;

        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        store
            .add_key_with_protocol("claude", CLAUDE_OAUTH_SENTINEL, None, "{\"token\":\"t\"}")
            .await
            .unwrap();
        store
            .add_key_with_protocol("codex", CODEX_OAUTH_SENTINEL, None, "{\"token\":\"t\"}")
            .await
            .unwrap();
        store
            .add_key_with_protocol("gemini", GEMINI_OAUTH_SENTINEL, None, "{\"token\":\"t\"}")
            .await
            .unwrap();
        store
            .add_key_with_protocol("copilot", "copilot", None, "ghu_test")
            .await
            .unwrap();
        store
            .add_key_with_protocol("grok", GROK_OAUTH_SENTINEL, None, "{\"token\":\"t\"}")
            .await
            .unwrap();
        store
            .add_key_with_protocol("kimi", KIMI_OAUTH_SENTINEL, None, "{\"token\":\"t\"}")
            .await
            .unwrap();
        store
            .add_key_with_protocol(
                "cursor",
                CURSOR_ACP_SENTINEL,
                None,
                &format!("{CURSOR_SHADOW_PREFIX}testaccount1"),
            )
            .await
            .unwrap();
        store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();

        let (exported, report) = store.export_keys(None).await.unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].name, "alpha");
        assert_eq!(report.skipped_oauth, 7);

        let (by_id, report) = store
            .export_keys(Some(&["claude".to_string()]))
            .await
            .unwrap();
        assert!(by_id.is_empty());
        assert_eq!(report.skipped_oauth, 1);
    }

    #[tokio::test]
    async fn import_drops_login_session_rows() {
        use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
        use crate::services::cursor_acp::{CURSOR_ACP_SENTINEL, CURSOR_SHADOW_PREFIX};
        use crate::services::grok_oauth::GROK_OAUTH_SENTINEL;

        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let records = vec![
            ApiKey::new_with_protocol(
                "aa1".into(),
                "claude".into(),
                CLAUDE_OAUTH_SENTINEL.into(),
                None,
                "{\"token\":\"t\"}".into(),
            ),
            ApiKey::new_with_protocol(
                "aa2".into(),
                "grok".into(),
                GROK_OAUTH_SENTINEL.into(),
                None,
                "{\"token\":\"t\"}".into(),
            ),
            ApiKey::new_with_protocol(
                "aa3".into(),
                "cursor".into(),
                CURSOR_ACP_SENTINEL.into(),
                None,
                format!("{CURSOR_SHADOW_PREFIX}testaccount1"),
            ),
            ApiKey::new_with_protocol(
                "aa4".into(),
                "alpha".into(),
                "http://a".into(),
                None,
                "sk-alpha".into(),
            ),
        ];

        let report = store
            .import_keys(records, ImportPolicy::Skip)
            .await
            .unwrap();
        assert_eq!(report.skipped_oauth, 3);
        assert_eq!(report.imported.len(), 1);

        let keys = store.get_keys().await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].name, "alpha");
    }

    #[tokio::test]
    async fn export_keeps_cursor_api_key_by_default() {
        use crate::services::cursor_acp::CURSOR_ACP_SENTINEL;

        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        store
            .add_key_with_protocol("cursor", CURSOR_ACP_SENTINEL, None, "sk-cursor")
            .await
            .unwrap();

        let (without, report) = store.export_keys(None).await.unwrap();
        assert_eq!(without.len(), 1);
        assert_eq!(without[0].name, "cursor");
        assert_eq!(without[0].key.as_str(), "sk-cursor");
        assert_eq!(report.skipped_oauth, 0);
    }

    #[tokio::test]
    async fn export_filters_by_id_and_rejects_unknown() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id_a = store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();
        store
            .add_key_with_protocol("beta", "http://b", None, "sk-beta")
            .await
            .unwrap();

        let (only_a, _) = store
            .export_keys(Some(std::slice::from_ref(&id_a)))
            .await
            .unwrap();
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].name, "alpha");

        let err = store
            .export_keys(Some(&["does-not-exist".to_string()]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Unknown key id or name"));
    }

    #[tokio::test]
    async fn export_filters_by_name_and_dedupes() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let id_a = store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();
        store
            .add_key_with_protocol("beta", "http://b", None, "sk-beta")
            .await
            .unwrap();

        let (by_name, _) = store
            .export_keys(Some(&["alpha".to_string()]))
            .await
            .unwrap();
        assert_eq!(by_name.len(), 1);
        assert_eq!(by_name[0].name, "alpha");

        // Same key selected by name and id exports once.
        let (deduped, _) = store
            .export_keys(Some(&["alpha".to_string(), id_a.clone()]))
            .await
            .unwrap();
        assert_eq!(deduped.len(), 1);
    }

    #[tokio::test]
    async fn export_rejects_ambiguous_name() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        store
            .add_key_with_protocol("dup", "http://a", None, "sk-a")
            .await
            .unwrap();
        store
            .add_key_with_protocol("dup", "http://b", None, "sk-b")
            .await
            .unwrap();

        let err = store
            .export_keys(Some(&["dup".to_string()]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("ambiguous"));
    }

    #[tokio::test]
    async fn import_into_empty_store_inserts_and_encrypts() {
        let temp_dir = TempDir::new().unwrap();
        let src = make_store(&temp_dir);
        src.add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();
        let (exported, _) = src.export_keys(None).await.unwrap();

        let dst_dir = TempDir::new().unwrap();
        let dst = make_store(&dst_dir);
        let report = dst.import_keys(exported, ImportPolicy::Skip).await.unwrap();
        assert_eq!(report.imported.len(), 1);

        let new_id = &report.imported[0];
        assert_eq!(new_id.len(), KEY_ID_LENGTH);

        let roundtripped = dst.get_key_by_id(new_id).await.unwrap().unwrap();
        assert_eq!(roundtripped.key.as_str(), "sk-alpha");
        assert_eq!(roundtripped.name, "alpha");
        let info = dst.get_key_by_id_info(new_id).await.unwrap().unwrap();
        assert!(is_encrypted(&info.key));
    }

    #[tokio::test]
    async fn import_id_collision_with_unrelated_key_does_not_overwrite() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let local_id = store
            .add_key_with_protocol("anthropic", "https://api.anthropic.com", None, "sk-local")
            .await
            .unwrap();

        let evil = ApiKey::new_with_protocol(
            local_id.clone(),
            "openrouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
            None,
            "sk-evil-from-other-machine".to_string(),
        );

        let report = store
            .import_keys(vec![evil], ImportPolicy::Overwrite)
            .await
            .unwrap();
        assert!(report.overwritten.is_empty(), "must not overwrite by id");
        assert_eq!(report.imported.len(), 1);

        let local = store.get_key_by_id(&local_id).await.unwrap().unwrap();
        assert_eq!(local.name, "anthropic");
        assert_eq!(local.key.as_str(), "sk-local");

        let new_id = &report.imported[0];
        assert_ne!(new_id, &local_id);
        let imported = store.get_key_by_id(new_id).await.unwrap().unwrap();
        assert_eq!(imported.name, "openrouter");
        assert_eq!(imported.key.as_str(), "sk-evil-from-other-machine");
    }

    #[tokio::test]
    async fn import_normalises_non_ascii_id() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);

        let evil = ApiKey::new_with_protocol(
            "🔑🔑".to_string(),
            "alpha".to_string(),
            "http://a".to_string(),
            None,
            "sk-alpha".to_string(),
        );

        let report = store
            .import_keys(vec![evil], ImportPolicy::Skip)
            .await
            .unwrap();
        assert_eq!(report.imported.len(), 1);
        let new_id = &report.imported[0];

        assert_eq!(new_id.len(), KEY_ID_LENGTH);
        assert!(new_id.chars().all(|c| KEY_ID_ALPHABET.contains(&(c as u8))));

        let stored = store.get_key_by_id(new_id).await.unwrap().unwrap();
        assert_eq!(stored.short_id(), new_id.as_str());
    }

    #[tokio::test]
    async fn import_same_machine_is_idempotent_on_skip() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let id = store
            .add_key_with_protocol("alpha", "http://a", None, "sk-alpha")
            .await
            .unwrap();
        let (exported, _) = store.export_keys(None).await.unwrap();

        let report = store
            .import_keys(exported, ImportPolicy::Skip)
            .await
            .unwrap();
        assert_eq!(report.skipped, vec![id]);
        assert!(report.imported.is_empty());
    }

    #[tokio::test]
    async fn import_overwrite_replaces_existing_secret() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let id = store
            .add_key_with_protocol("alpha", "http://a", None, "sk-original")
            .await
            .unwrap();

        let (mut imported, _) = store.export_keys(None).await.unwrap();
        imported[0].key = Zeroizing::new("sk-rotated".to_string());

        let report = store
            .import_keys(imported, ImportPolicy::Overwrite)
            .await
            .unwrap();
        assert_eq!(report.overwritten, vec![id.clone()]);

        let after = store.get_key_by_id(&id).await.unwrap().unwrap();
        assert_eq!(after.key.as_str(), "sk-rotated");
    }

    #[tokio::test]
    async fn import_rename_keeps_existing_and_adds_new() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let original_id = store
            .add_key_with_protocol("alpha", "http://a", None, "sk-original")
            .await
            .unwrap();

        let (mut imported, _) = store.export_keys(None).await.unwrap();
        imported[0].key = Zeroizing::new("sk-incoming".to_string());

        let report = store
            .import_keys(imported, ImportPolicy::Rename)
            .await
            .unwrap();
        assert_eq!(report.renamed.len(), 1);
        let (orig, new_id) = &report.renamed[0];
        assert_eq!(orig, &original_id);
        assert_ne!(new_id, &original_id);

        let original = store.get_key_by_id(&original_id).await.unwrap().unwrap();
        assert_eq!(original.key.as_str(), "sk-original");
        let imported_back = store.get_key_by_id(new_id).await.unwrap().unwrap();
        assert_eq!(imported_back.key.as_str(), "sk-incoming");
        assert!(imported_back.name.ends_with("(imported)"));
    }
}
