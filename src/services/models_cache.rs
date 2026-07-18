use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::{OnceCell, RwLock};

const CACHE_TTL_SECS: u64 = 4 * 3600; // 4 hours

/// Per-model metadata harvested from `aivo models`. Stores every column the
/// table renders so `aivo models` can serve from cache without a network
/// roundtrip. `context_window` is also consumed by `aivo run claude
/// --max-context` and is treated as long-lived (returned regardless of the
/// entry's `fetched_at` TTL).
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ModelMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output: Option<String>,
    /// Numeric twin of `max_output` (which is display-formatted, e.g. "64K").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_price: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    /// Reasoning-effort levels advertised by `/v1/models` (e.g. for `aivo/starter`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasoning_efforts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CacheEntry {
    models: Vec<String>,
    fetched_at: u64,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    metadata: HashMap<String, ModelMetadata>,
}

/// Disk cache for model lists keyed by base_url.
/// Stored at ~/.config/aivo/models-cache.json as plaintext JSON.
///
/// The disk file is read at most once per process lifetime via `OnceCell`;
/// concurrent callers wait on the same initialisation rather than each
/// reading the file independently.
#[derive(Debug, Clone)]
pub struct ModelsCache {
    cache_path: PathBuf,
    /// Initialised exactly once (first call to `get` or `set`).
    entries: Arc<OnceCell<RwLock<HashMap<String, CacheEntry>>>>,
}

impl ModelsCache {
    pub fn new() -> Self {
        let cache_path =
            crate::services::paths::models_cache(&crate::services::paths::config_dir());
        Self {
            cache_path,
            entries: Arc::new(OnceCell::new()),
        }
    }

    #[cfg(test)]
    pub fn with_path(cache_path: PathBuf) -> Self {
        Self {
            cache_path,
            entries: Arc::new(OnceCell::new()),
        }
    }

    /// Process-wide shared instance: reads disk at most once and reuses the
    /// in-memory map. Hot paths (routers resolving model names) should prefer
    /// this over `new()`, which re-reads the file per instance.
    pub fn shared() -> &'static ModelsCache {
        static SHARED: std::sync::OnceLock<ModelsCache> = std::sync::OnceLock::new();
        SHARED.get_or_init(ModelsCache::new)
    }

    /// Returns the initialised entries map, loading from disk exactly once.
    async fn entries(&self) -> &RwLock<HashMap<String, CacheEntry>> {
        self.entries
            .get_or_init(|| async {
                let entries = Self::read_disk_cache(&self.cache_path).await;
                RwLock::new(entries)
            })
            .await
    }

    async fn read_disk_cache(cache_path: &PathBuf) -> HashMap<String, CacheEntry> {
        tokio::fs::read_to_string(cache_path)
            .await
            .ok()
            .and_then(|data| serde_json::from_str(&data).ok())
            .unwrap_or_default()
    }

    fn fresh_models(entry: &CacheEntry) -> Option<Vec<String>> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
        if now.saturating_sub(entry.fetched_at) < CACHE_TTL_SECS {
            Some(entry.models.clone())
        } else {
            None
        }
    }

    /// Returns cached models for `base_url` if present and not expired.
    pub async fn get(&self, base_url: &str) -> Option<Vec<String>> {
        let entries = self.entries().await;
        let state = entries.read().await;
        state.get(base_url).and_then(Self::fresh_models)
    }

    /// Model ids advertised for `base_url`, ignoring the TTL (ids are stable
    /// once published) and merging both namespaces — the picker key (`base_url`)
    /// and the `aivo models` key (`{base_url}#all`). `None` if never fetched.
    pub async fn model_ids(&self, base_url: &str) -> Option<Vec<String>> {
        let entries = self.entries().await;
        let state = entries.read().await;
        let mut ids: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for key in [base_url.to_string(), full_catalog_key(base_url)] {
            if let Some(entry) = state.get(&key) {
                for m in &entry.models {
                    if seen.insert(m.as_str()) {
                        ids.push(m.clone());
                    }
                }
            }
        }
        (!ids.is_empty()).then_some(ids)
    }

    /// Returns the cached id list and per-model metadata for `base_url` when
    /// the entry is fresh. Used by `aivo models` to reconstruct its rich
    /// table from cache without re-fetching.
    pub async fn get_with_metadata(
        &self,
        base_url: &str,
    ) -> Option<(Vec<String>, HashMap<String, ModelMetadata>)> {
        let entries = self.entries().await;
        let state = entries.read().await;
        let entry = state.get(base_url)?;
        let models = Self::fresh_models(entry)?;
        Some((models, entry.metadata.clone()))
    }

    /// Writes models for `base_url`. Plain `set` preserves any existing
    /// metadata so a chat-picker refresh doesn't wipe what `aivo models`
    /// harvested.
    pub async fn set(&self, base_url: &str, models: Vec<String>) {
        self.write_entry(base_url, models, None).await;
    }

    /// Writes models and per-model metadata in one pass. Used by `aivo models`.
    pub async fn set_with_metadata(
        &self,
        base_url: &str,
        models: Vec<String>,
        metadata: HashMap<String, ModelMetadata>,
    ) {
        self.write_entry(base_url, models, Some(metadata)).await;
    }

    async fn write_entry(
        &self,
        base_url: &str,
        models: Vec<String>,
        metadata: Option<HashMap<String, ModelMetadata>>,
    ) {
        let entries = self.entries().await;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let json = {
            let mut state = entries.write().await;
            match state.entry(base_url.to_string()) {
                Entry::Occupied(mut o) => {
                    let e = o.get_mut();
                    e.models = models;
                    e.fetched_at = now;
                    if let Some(m) = metadata {
                        e.metadata = m;
                    }
                }
                Entry::Vacant(v) => {
                    v.insert(CacheEntry {
                        models,
                        fetched_at: now,
                        metadata: metadata.unwrap_or_default(),
                    });
                }
            }
            serde_json::to_string_pretty(&*state).ok()
        };

        if let Some(json) = json {
            if let Some(parent) = self.cache_path.parent() {
                let _ = crate::services::atomic_write::ensure_private_dir(parent).await;
            }
            // Atomic so a crash mid-write can't leave a truncated cache file.
            let _ = crate::services::atomic_write::atomic_write_secure(
                &self.cache_path,
                json.into_bytes(),
            )
            .await;
        }
    }

    /// Drops the entries stored under `cache_key` and its `#all` catalog twin
    /// (models and metadata), persisting the change. Used by `keys
    /// reset-route` so the next launch re-fetches instead of serving a stale
    /// catalog.
    pub async fn remove(&self, cache_key: &str) {
        let entries = self.entries().await;
        let json = {
            let mut state = entries.write().await;
            let removed = state.remove(cache_key).is_some()
                | state.remove(&full_catalog_key(cache_key)).is_some();
            removed
                .then(|| serde_json::to_string_pretty(&*state).ok())
                .flatten()
        };
        if let Some(json) = json {
            if let Some(parent) = self.cache_path.parent() {
                let _ = crate::services::atomic_write::ensure_private_dir(parent).await;
            }
            // Atomic so a crash mid-write can't leave a truncated cache file.
            let _ = crate::services::atomic_write::atomic_write_secure(
                &self.cache_path,
                json.into_bytes(),
            )
            .await;
        }
    }

    /// Drops both starter spellings (sentinel + real gateway URL) and their
    /// `#all` twins. The starter catalog is per-account and its metadata is
    /// looked up ignoring the TTL, so a profile change must clear it explicitly.
    pub async fn clear_starter(&self) {
        self.remove(crate::constants::AIVO_STARTER_SENTINEL).await;
        self.remove(crate::constants::AIVO_STARTER_REAL_URL).await;
    }

    /// Lower-level: returns metadata stored under `cache_key`, ignoring TTL.
    pub async fn get_metadata(&self, cache_key: &str, model_id: &str) -> Option<ModelMetadata> {
        let entries = self.entries().await;
        let state = entries.read().await;
        state.get(cache_key)?.metadata.get(model_id).cloned()
    }

    /// Returns the cached context window (in tokens) for `model_id` served by
    /// `base_url`. Ignores TTL — context windows are stable once published.
    pub async fn get_context_window(&self, base_url: &str, model_id: &str) -> Option<u64> {
        self.get_metadata(&full_catalog_key(base_url), model_id)
            .await?
            .context_window
    }
}

/// Cache key for the unfiltered catalog stored by `aivo models`. A separate
/// namespace from the chat picker's key (`base_url`) so a broad fetch doesn't
/// pollute chat pickers with image / embedding entries.
pub fn full_catalog_key(base_url: &str) -> String {
    format!("{base_url}#all")
}

impl Default for ModelsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_cache(dir: &TempDir) -> ModelsCache {
        ModelsCache::with_path(dir.path().join("models-cache.json"))
    }

    #[tokio::test]
    async fn cache_miss_on_empty() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        assert!(cache.get("https://api.example.com").await.is_none());
    }

    #[tokio::test]
    async fn roundtrip_set_and_get() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        let models = vec!["gpt-4o".to_string(), "gpt-4o-mini".to_string()];
        cache.set("https://api.example.com", models.clone()).await;
        let got = cache.get("https://api.example.com").await.unwrap();
        assert_eq!(got, models);
    }

    #[tokio::test]
    async fn corrupt_file_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        tokio::fs::write(&path, b"not json {{{").await.unwrap();
        let cache = ModelsCache::with_path(path);
        assert!(cache.get("https://api.example.com").await.is_none());
    }

    #[tokio::test]
    async fn expired_entry_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        // Write a cache entry with fetched_at = 0 (epoch, definitely expired)
        let entry = serde_json::json!({
            "https://api.example.com": {
                "models": ["gpt-4o"],
                "fetched_at": 0u64
            }
        });
        tokio::fs::write(
            dir.path().join("models-cache.json"),
            serde_json::to_string(&entry).unwrap(),
        )
        .await
        .unwrap();
        assert!(cache.get("https://api.example.com").await.is_none());
    }

    #[tokio::test]
    async fn metadata_roundtrip() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        let mut metadata = HashMap::new();
        metadata.insert(
            "gpt-4.1".to_string(),
            ModelMetadata {
                context_window: Some(1_000_000),
                ..Default::default()
            },
        );
        cache
            .set_with_metadata(
                &full_catalog_key("https://api.example.com"),
                vec!["gpt-4.1".to_string()],
                metadata,
            )
            .await;
        // High-level accessor mirrors `aivo run claude`'s lookup path.
        assert_eq!(
            cache
                .get_context_window("https://api.example.com", "gpt-4.1")
                .await,
            Some(1_000_000)
        );
    }

    #[tokio::test]
    async fn metadata_ignores_ttl() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        let entry = serde_json::json!({
            "https://api.example.com": {
                "models": ["gpt-4.1"],
                "fetched_at": 0u64,
                "metadata": {
                    "gpt-4.1": { "context_window": 1_000_000u64 }
                }
            }
        });
        tokio::fs::write(&path, serde_json::to_string(&entry).unwrap())
            .await
            .unwrap();
        let cache = ModelsCache::with_path(path);
        // Models list is expired and returns None…
        assert!(cache.get("https://api.example.com").await.is_none());
        // …but metadata is still served.
        let meta = cache
            .get_metadata("https://api.example.com", "gpt-4.1")
            .await
            .unwrap();
        assert_eq!(meta.context_window, Some(1_000_000));
    }

    #[tokio::test]
    async fn set_preserves_existing_metadata() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        let mut metadata = HashMap::new();
        metadata.insert(
            "gpt-4.1".to_string(),
            ModelMetadata {
                context_window: Some(1_000_000),
                ..Default::default()
            },
        );
        cache
            .set_with_metadata(
                "https://api.example.com",
                vec!["gpt-4.1".to_string()],
                metadata,
            )
            .await;
        // Plain set() — e.g. by the chat picker — must not wipe metadata.
        cache
            .set("https://api.example.com", vec!["gpt-4.1".to_string()])
            .await;
        let got = cache
            .get_metadata("https://api.example.com", "gpt-4.1")
            .await
            .unwrap();
        assert_eq!(got.context_window, Some(1_000_000));
    }

    #[tokio::test]
    async fn metadata_missing_for_unknown_model() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        cache
            .set("https://api.example.com", vec!["gpt-4o".to_string()])
            .await;
        assert!(
            cache
                .get_metadata("https://api.example.com", "gpt-4o")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn remove_drops_picker_and_catalog_entries_and_persists() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        let cache = ModelsCache::with_path(path.clone());
        let base = "https://api.example.com";
        cache.set(base, vec!["gpt-4o".to_string()]).await;
        let mut metadata = HashMap::new();
        metadata.insert(
            "gpt-4o".to_string(),
            ModelMetadata {
                context_window: Some(128_000),
                ..Default::default()
            },
        );
        cache
            .set_with_metadata(
                &full_catalog_key(base),
                vec!["gpt-4o".to_string()],
                metadata,
            )
            .await;

        cache.remove(base).await;

        assert!(cache.get(base).await.is_none());
        assert!(cache.get(&full_catalog_key(base)).await.is_none());
        assert!(
            cache
                .get_metadata(&full_catalog_key(base), "gpt-4o")
                .await
                .is_none()
        );
        // Removal must survive a fresh load from disk.
        let reloaded = ModelsCache::with_path(path);
        assert!(reloaded.get(base).await.is_none());
        assert!(reloaded.get(&full_catalog_key(base)).await.is_none());
    }

    #[tokio::test]
    async fn remove_missing_key_is_noop() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        cache
            .set("https://api.kept.com", vec!["m".to_string()])
            .await;
        cache.remove("https://api.example.com").await;
        assert!(cache.get("https://api.kept.com").await.is_some());
    }

    #[tokio::test]
    async fn clear_starter_drops_all_four_starter_keys_only() {
        let dir = TempDir::new().unwrap();
        let cache = make_cache(&dir);
        let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
        let real = crate::constants::AIVO_STARTER_REAL_URL;
        for key in [
            sentinel.to_string(),
            full_catalog_key(sentinel),
            real.to_string(),
            full_catalog_key(real),
        ] {
            cache.set(&key, vec!["aivo/starter".to_string()]).await;
        }
        cache
            .set("https://api.other.com", vec!["gpt-4o".to_string()])
            .await;

        cache.clear_starter().await;

        assert!(cache.model_ids(sentinel).await.is_none());
        assert!(cache.model_ids(real).await.is_none());
        assert!(cache.get("https://api.other.com").await.is_some());
    }

    #[tokio::test]
    async fn warm_cache_serves_from_memory_after_disk_changes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("models-cache.json");
        let entry = serde_json::json!({
            "https://api.example.com": {
                "models": ["gpt-4o"],
                "fetched_at": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            }
        });
        tokio::fs::write(&path, serde_json::to_string(&entry).unwrap())
            .await
            .unwrap();

        let cache = ModelsCache::with_path(path.clone());
        assert_eq!(
            cache.get("https://api.example.com").await,
            Some(vec!["gpt-4o".to_string()])
        );

        tokio::fs::write(&path, b"broken now").await.unwrap();

        assert_eq!(
            cache.get("https://api.example.com").await,
            Some(vec!["gpt-4o".to_string()])
        );
    }
}
