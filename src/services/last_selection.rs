use anyhow::Result;
use chrono::Utc;

use crate::services::session_store::{ApiKey, ConfigContext, DirectoryStartRecord, LastSelection};

fn has_valid_key(record: &LastSelection, keys: &[ApiKey]) -> bool {
    keys.iter()
        .any(|key| key.id == record.key_id && key.base_url == record.base_url)
}

#[derive(Debug, Clone)]
pub(crate) struct LastSelectionStore {
    pub(crate) ctx: ConfigContext,
}

impl LastSelectionStore {
    pub(crate) async fn get(&self) -> Result<Option<LastSelection>> {
        let config = self.ctx.load().await?;
        let Some(record) = config.last_selection.clone() else {
            return Ok(None);
        };

        if has_valid_key(&record, &config.api_keys) {
            return Ok(Some(record));
        }

        // Stale record — referenced key was deleted. Re-check under lock
        // to avoid racing with a concurrent write that may have fixed it.
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        if config
            .last_selection
            .as_ref()
            .is_some_and(|r| !has_valid_key(r, &config.api_keys))
        {
            config.last_selection = None;
            self.ctx.save_raw(&config).await?;
        }
        Ok(None)
    }

    pub(crate) async fn set(&self, key: &ApiKey, tool: &str, model: Option<&str>) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        config.last_selection = Some(DirectoryStartRecord {
            key_id: key.id.clone(),
            base_url: key.base_url.clone(),
            tool: tool.to_string(),
            model: model.map(ToString::to_string),
            updated_at: Utc::now().to_rfc3339(),
        });
        self.ctx.save_raw(&config).await
    }

    pub(crate) async fn clear(&self) -> Result<()> {
        let _lock = self.ctx.acquire_config_lock()?;
        let mut config = self.ctx.load().await?;
        if config.last_selection.is_some() {
            config.last_selection = None;
            self.ctx.save_raw(&config).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::{ApiKey, ConfigContext, StoredConfig};
    use tempfile::TempDir;

    fn make_store(temp_dir: &TempDir) -> LastSelectionStore {
        let config_path = temp_dir.path().join("config.json");
        let config_dir = temp_dir.path().to_path_buf();
        LastSelectionStore {
            ctx: ConfigContext {
                config_path,
                config_dir,
            },
        }
    }

    fn make_key(key_id: &str, base_url: &str) -> ApiKey {
        ApiKey::new_with_protocol(
            key_id.to_string(),
            "test".to_string(),
            base_url.to_string(),
            None,
            "sk-test".to_string(),
        )
    }

    async fn write_config_with_key(store: &LastSelectionStore, key: &ApiKey) {
        let config = StoredConfig {
            api_keys: vec![key.clone()],
            ..StoredConfig::new()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::create_dir_all(&store.ctx.config_dir)
            .await
            .unwrap();
        tokio::fs::write(&store.ctx.config_path, &data)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn get_returns_none_when_empty() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        let result = store.get().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_and_get_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store.set(&key, "claude", Some("gpt-4o")).await.unwrap();

        let record = store.get().await.unwrap().unwrap();
        assert_eq!(record.key_id, "key1");
        assert_eq!(record.base_url, "http://localhost");
        assert_eq!(record.tool, "claude");
        assert_eq!(record.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn set_with_none_model() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store.set(&key, "codex", None).await.unwrap();

        let record = store.get().await.unwrap().unwrap();
        assert_eq!(record.tool, "codex");
        assert!(record.model.is_none());
    }

    #[tokio::test]
    async fn set_with_default_placeholder() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store
            .set(
                &key,
                "claude",
                Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER),
            )
            .await
            .unwrap();

        let record = store.get().await.unwrap().unwrap();
        assert_eq!(
            record.model.as_deref(),
            Some(crate::constants::MODEL_DEFAULT_PLACEHOLDER)
        );
    }

    #[tokio::test]
    async fn get_prunes_stale_record() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store.set(&key, "claude", None).await.unwrap();

        // Remove the key from config
        let config = StoredConfig::new();
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::write(&store.ctx.config_path, &data)
            .await
            .unwrap();

        let result = store.get().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_overwrites_existing() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store.set(&key, "claude", Some("sonnet")).await.unwrap();

        store.set(&key, "codex", Some("gpt-4o")).await.unwrap();

        let record = store.get().await.unwrap().unwrap();
        assert_eq!(record.tool, "codex");
        assert_eq!(record.model.as_deref(), Some("gpt-4o"));
    }

    #[tokio::test]
    async fn clear_removes_record() {
        let temp_dir = TempDir::new().unwrap();
        let store = make_store(&temp_dir);
        let key = make_key("key1", "http://localhost");
        write_config_with_key(&store, &key).await;

        store.set(&key, "claude", None).await.unwrap();

        store.clear().await.unwrap();

        let result = store.get().await.unwrap();
        assert!(result.is_none());
    }
}
