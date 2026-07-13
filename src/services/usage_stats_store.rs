use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::services::atomic_write::atomic_write_secure;
use crate::services::session_store::{ConfigContext, ConfigLockGuard, UsageStats};

/// Per-run token accumulator the plugin endpoint taps so a coding-agent run's
/// `finished` log row carries *timestamped* token totals.
///
/// The lifetime [`UsageStats`] is keyed by `(key, tool, model)` with no
/// timestamp, so it can't be windowed by `aivo stats --since` — but logs.db
/// rows can. Both endpoint engines (`ServeRouter` / `ResponsesToChatRouter`) add
/// to this at the same point they record lifetime usage, and `finish_accounting`
/// reads the snapshot onto the run's finished `tool_launch` row. Shared across
/// the router's async tasks via `Arc`, hence atomics.
#[derive(Debug, Default)]
pub struct RunTokenTally {
    prompt: AtomicU64,
    completion: AtomicU64,
    cache_read: AtomicU64,
    cache_creation: AtomicU64,
}

impl RunTokenTally {
    /// Fold one usage report into the running totals.
    pub(crate) fn add(&self, prompt: u64, completion: u64, cache_read: u64, cache_creation: u64) {
        self.prompt.fetch_add(prompt, Ordering::Relaxed);
        self.completion.fetch_add(completion, Ordering::Relaxed);
        self.cache_read.fetch_add(cache_read, Ordering::Relaxed);
        self.cache_creation
            .fetch_add(cache_creation, Ordering::Relaxed);
    }

    /// `(prompt, completion, cache_read, cache_creation)` accumulated so far.
    pub(crate) fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.prompt.load(Ordering::Relaxed),
            self.completion.load(Ordering::Relaxed),
            self.cache_read.load(Ordering::Relaxed),
            self.cache_creation.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug, Clone)]
struct StatsFileContext {
    stats_path: PathBuf,
    lock_path: PathBuf,
}

impl StatsFileContext {
    fn acquire_lock(&self) -> Result<ConfigLockGuard> {
        if let Some(parent) = self.stats_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }
        ConfigLockGuard::acquire(&self.lock_path)
    }

    async fn load(&self) -> Result<UsageStats> {
        let data = match tokio::fs::read_to_string(&self.stats_path).await {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(UsageStats::default());
            }
            Err(e) => return Err(e.into()),
        };
        serde_json::from_str(&data).context("Failed to parse stats.json")
    }

    async fn save(&self, stats: &UsageStats) -> Result<()> {
        if let Some(parent) = self.stats_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let data = serde_json::to_string_pretty(stats).context("Failed to serialize stats")?;
        atomic_write_secure(&self.stats_path, data.into_bytes()).await
    }
}

#[derive(Debug, Clone)]
pub(crate) struct UsageStatsStore {
    stats_ctx: StatsFileContext,
    config_ctx: ConfigContext,
}

impl UsageStatsStore {
    pub(crate) fn new(config_ctx: ConfigContext) -> Self {
        let stats_path = crate::services::paths::stats_json(&config_ctx.config_dir);
        let lock_path = crate::services::paths::stats_lock(&config_ctx.config_dir);
        Self {
            stats_ctx: StatsFileContext {
                stats_path,
                lock_path,
            },
            config_ctx,
        }
    }

    /// Loads stats, migrating from config.json on first access if needed.
    /// Also folds any legacy per-model maps into the canonical `per_model_usage`
    /// field so all downstream code sees a single shape.
    async fn load_with_migration(&self) -> Result<UsageStats> {
        let mut stats = self.load_raw_with_legacy_config_migration().await?;
        stats.migrate_legacy_per_model();
        Ok(stats)
    }

    async fn load_raw_with_legacy_config_migration(&self) -> Result<UsageStats> {
        if tokio::fs::try_exists(&self.stats_ctx.stats_path).await? {
            return self.stats_ctx.load().await;
        }

        // Check config.json for inline stats to migrate
        let _config_lock = self.config_ctx.acquire_config_lock()?;

        // Double-check after acquiring lock
        if tokio::fs::try_exists(&self.stats_ctx.stats_path).await? {
            return self.stats_ctx.load().await;
        }

        let config = self.config_ctx.load().await?;
        if !config.stats.is_empty() {
            let migrated = config.stats.clone();
            self.stats_ctx.save(&migrated).await?;
            let mut config = config;
            config.stats = UsageStats::default();
            self.config_ctx.save_raw(&config).await?;
            return Ok(migrated);
        }

        Ok(UsageStats::default())
    }

    pub(crate) async fn load(&self) -> Result<UsageStats> {
        let _lock = self.stats_ctx.acquire_lock()?;
        self.load_with_migration().await
    }

    pub(crate) async fn record_selection(
        &self,
        key_id: &str,
        tool: &str,
        model: Option<&str>,
    ) -> Result<()> {
        // Best-effort: an unavailable lock skips the write, never fails the launch.
        let Ok(_lock) = self.stats_ctx.acquire_lock() else {
            return Ok(());
        };
        let mut stats = self.load_with_migration().await?;
        stats.record_selection(key_id, tool, model);
        self.stats_ctx.save(&stats).await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn record_tokens(
        &self,
        key_id: &str,
        tool: Option<&str>,
        model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
        cache_read_input_tokens: u64,
        cache_creation_input_tokens: u64,
    ) -> Result<()> {
        let Ok(_lock) = self.stats_ctx.acquire_lock() else {
            return Ok(());
        };
        let mut stats = self.load_with_migration().await?;
        stats.record_tokens(
            key_id,
            tool,
            model,
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        );
        self.stats_ctx.save(&stats).await
    }

    pub(crate) async fn record_agent_run(
        &self,
        key_id: &str,
        agent: &str,
        ok: bool,
        steps: u64,
        tokens: u64,
    ) -> Result<()> {
        let Ok(_lock) = self.stats_ctx.acquire_lock() else {
            return Ok(());
        };
        let mut stats = self.load_with_migration().await?;
        stats.record_agent_run(key_id, agent, ok, steps, tokens);
        self.stats_ctx.save(&stats).await
    }

    pub(crate) async fn remove_key(&self, key_id: &str) -> Result<()> {
        let _lock = self.stats_ctx.acquire_lock()?;
        let mut stats = self.load_with_migration().await?;
        stats.remove_key(key_id);
        self.stats_ctx.save(&stats).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_store(dir: &TempDir) -> UsageStatsStore {
        let config_dir = dir.path().to_path_buf();
        let config_path = config_dir.join("config.json");
        let ctx = ConfigContext {
            config_path,
            config_dir,
        };
        UsageStatsStore::new(ctx)
    }

    #[tokio::test]
    async fn load_returns_default_when_no_file() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        let stats = store.load().await.unwrap();
        assert!(stats.is_empty());
    }

    /// A held lock (wedged process) must skip the write, not fail or stall.
    #[cfg(unix)]
    #[tokio::test]
    async fn record_skips_when_lock_held() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        let held = store.stats_ctx.acquire_lock().unwrap();
        store
            .record_selection("key1", "claude", Some("opus"))
            .await
            .unwrap();
        drop(held);
        assert!(store.load().await.unwrap().is_empty());
        store
            .record_selection("key1", "claude", Some("opus"))
            .await
            .unwrap();
        assert!(!store.load().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn record_selection_creates_stats_file() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        store
            .record_selection("key1", "claude", Some("opus"))
            .await
            .unwrap();
        assert!(crate::services::paths::stats_json(dir.path()).exists());
        let stats = store.load().await.unwrap();
        assert_eq!(*stats.tool_counts.get("claude").unwrap(), 1);
    }

    #[tokio::test]
    async fn record_tokens_accumulates() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        store
            .record_tokens("key1", Some("amp"), Some("gpt-4o"), 100, 50, 80, 10)
            .await
            .unwrap();
        store
            .record_tokens("key1", Some("amp"), Some("gpt-4o"), 200, 100, 0, 0)
            .await
            .unwrap();
        let stats = store.load().await.unwrap();
        let key_stats = stats.key_usage.get("key1").unwrap();
        assert_eq!(key_stats.prompt_tokens, 300);
        assert_eq!(key_stats.completion_tokens, 150);
        assert_eq!(key_stats.total_tokens, 450);
        assert_eq!(key_stats.cache_read_input_tokens, 80);
        // Per-tool attribution accumulates per (tool, model).
        let amp = key_stats.per_tool_model_usage.get("amp").unwrap();
        let model = amp.get("gpt-4o").unwrap();
        assert_eq!(model.prompt_tokens, 300);
        assert_eq!(model.completion_tokens, 150);
        assert_eq!(model.cache_read_input_tokens, 80);
    }

    #[tokio::test]
    async fn record_agent_run_persists_and_accumulates() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        store
            .record_agent_run("key1", "code-reviewer", true, 6, 900)
            .await
            .unwrap();
        store
            .record_agent_run("key1", "code-reviewer", false, 3, 400)
            .await
            .unwrap();
        let stats = store.load().await.unwrap();
        let agent = stats
            .key_usage
            .get("key1")
            .unwrap()
            .per_agent
            .get("code-reviewer")
            .unwrap();
        assert_eq!(agent.runs, 2);
        assert_eq!(agent.ok_runs, 1);
        assert_eq!(agent.steps, 9);
        assert_eq!(agent.tokens, 1300);
    }

    #[tokio::test]
    async fn migrates_stats_from_config_json() {
        let dir = TempDir::new().unwrap();
        let config_dir = dir.path().to_path_buf();
        let config_path = config_dir.join("config.json");
        let ctx = ConfigContext {
            config_path: config_path.clone(),
            config_dir,
        };

        // Write a config.json with inline stats
        let mut stats = UsageStats::default();
        stats.record_selection("key1", "claude", Some("opus"));
        let config = crate::services::session_store::StoredConfig {
            stats,
            ..Default::default()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::write(&config_path, &data).await.unwrap();

        let store = UsageStatsStore::new(ctx);
        let loaded = store.load().await.unwrap();
        assert_eq!(*loaded.tool_counts.get("claude").unwrap(), 1);

        // stats.json should now exist
        assert!(crate::services::paths::stats_json(dir.path()).exists());

        // config.json stats should be cleared
        let config_data = tokio::fs::read_to_string(&config_path).await.unwrap();
        let config: crate::services::session_store::StoredConfig =
            serde_json::from_str(&config_data).unwrap();
        assert!(config.stats.is_empty());
    }

    #[tokio::test]
    async fn does_not_touch_config_when_stats_file_exists() {
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);

        // Create stats.json directly
        store
            .record_selection("key1", "claude", None)
            .await
            .unwrap();

        // Write config.json with different stats (should be ignored)
        let mut stats = UsageStats::default();
        stats.record_selection("key2", "codex", None);
        stats.record_selection("key2", "codex", None);
        let config = crate::services::session_store::StoredConfig {
            stats,
            ..Default::default()
        };
        let data = serde_json::to_string_pretty(&config).unwrap();
        tokio::fs::write(dir.path().join("config.json"), &data)
            .await
            .unwrap();

        let loaded = store.load().await.unwrap();
        assert_eq!(*loaded.tool_counts.get("claude").unwrap(), 1); // from stats.json, not config.json
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stats_file_has_restricted_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let store = test_store(&dir);
        store
            .record_selection("key1", "claude", None)
            .await
            .unwrap();
        let metadata = std::fs::metadata(crate::services::paths::stats_json(dir.path())).unwrap();
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
    }
}
