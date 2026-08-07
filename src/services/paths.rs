//! Single source of truth for the on-disk layout under the aivo config dir.
//! Top level holds only hand-editable config and user content; everything
//! machine-managed lives in a chartered subdir: `state/` (prefs, history,
//! stats), `secrets/` (key material, 0700), `cache/` (regenerable — safe to
//! delete), `logs/`, `run/` (pid files). `migrate_layout` renames pre-split
//! files into place once; shadow HOMEs and plugin-owned entries are never
//! touched.

use std::path::{Path, PathBuf};

use crate::services::system_env;

pub const STATE_DIR: &str = "state";
pub const SECRETS_DIR: &str = "secrets";
pub const CACHE_DIR: &str = "cache";
pub const LOGS_DIR: &str = "logs";
pub const RUN_DIR: &str = "run";

/// Base dir: `$AIVO_CONFIG_DIR` (already exported to plugin children, now
/// honored by aivo itself), else `~/.config/aivo`.
pub fn config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AIVO_CONFIG_DIR")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    system_env::home_dir()
        .map(|p| p.join(".config").join("aivo"))
        .unwrap_or_else(|| PathBuf::from(".config/aivo"))
}

// ── state/ ────────────────────────────────────────────────────────────────

pub fn stats_json(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("stats.json")
}

pub fn stats_lock(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("stats.lock")
}

pub fn code_prefs(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("code-prefs.json")
}

/// Pre-rename prefs file, kept as a read fallback (see `read_code_prefs`).
pub fn chat_prefs_legacy(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("chat-prefs.json")
}

pub fn chat_history(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("chat_history")
}

pub fn grants_json(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("grants.json")
}

pub fn gemini_thought_signatures(base: &Path) -> PathBuf {
    base.join(STATE_DIR).join("gemini_thought_signatures.json")
}

// ── plans/ (user content: `/plan save` markdown) ──────────────────────────

pub fn plans_dir(base: &Path) -> PathBuf {
    base.join("plans")
}

// ── images/ (user content: tool-generated images, content-addressed) ──────

pub fn images_dir(base: &Path) -> PathBuf {
    base.join("images")
}

// ── secrets/ ──────────────────────────────────────────────────────────────

pub fn device_key(base: &Path) -> PathBuf {
    base.join(SECRETS_DIR).join("device-key")
}

pub fn account_json(base: &Path) -> PathBuf {
    base.join(SECRETS_DIR).join("account.json")
}

pub fn mcp_tokens(base: &Path) -> PathBuf {
    base.join(SECRETS_DIR).join("mcp_tokens.json")
}

// ── cache/ ────────────────────────────────────────────────────────────────

pub fn models_cache(base: &Path) -> PathBuf {
    base.join(CACHE_DIR).join("models-cache.json")
}

pub fn stats_cache(base: &Path, tool: &str) -> PathBuf {
    base.join(CACHE_DIR)
        .join(format!("stats-cache-{tool}.json"))
}

pub fn update_check(base: &Path) -> PathBuf {
    base.join(CACHE_DIR).join("update_check.json")
}

pub fn model_limits(base: &Path) -> PathBuf {
    base.join(CACHE_DIR).join("model_limits.json")
}

// ── logs/ ─────────────────────────────────────────────────────────────────

pub fn logs_db(base: &Path) -> PathBuf {
    base.join(LOGS_DIR).join("logs.db")
}

pub fn logs_dir(base: &Path) -> PathBuf {
    base.join(LOGS_DIR)
}

// ── run/ ──────────────────────────────────────────────────────────────────

pub fn ollama_pids(base: &Path) -> PathBuf {
    base.join(RUN_DIR).join("ollama-pids")
}

// ── migration ─────────────────────────────────────────────────────────────

type PathBuilder = fn(&Path) -> PathBuf;

/// Flat-layout files that move into a chartered subdir, as
/// (old top-level name, new path builder).
fn moved_files() -> Vec<(&'static str, PathBuilder)> {
    vec![
        ("stats.json", stats_json),
        ("stats.lock", stats_lock),
        ("code-prefs.json", code_prefs),
        ("chat-prefs.json", chat_prefs_legacy),
        ("chat_history", chat_history),
        ("grants.json", grants_json),
        ("gemini_thought_signatures.json", gemini_thought_signatures),
        ("device-key", device_key),
        ("account.json", account_json),
        ("mcp_tokens.json", mcp_tokens),
        ("models-cache.json", models_cache),
        ("update_check.json", update_check),
        ("model_limits.json", model_limits),
    ]
}

/// One-shot migration from the flat pre-split layout. Rename-if-absent keeps
/// it idempotent and safe against a concurrent instance racing the same move;
/// a file recreated at the old path by an older binary after the move is left
/// alone and reported back so the caller can warn. Best-effort throughout —
/// a failed rename just means that file keeps working from its old path via
/// nothing (new path wins on next write), never a startup failure.
pub fn migrate_layout(base: &Path) -> Vec<String> {
    use crate::services::atomic_write::ensure_private_dir_blocking;

    if !base.exists() {
        return Vec::new();
    }
    for dir in [STATE_DIR, SECRETS_DIR, RUN_DIR] {
        let _ = ensure_private_dir_blocking(&base.join(dir));
    }
    for dir in [CACHE_DIR, LOGS_DIR] {
        let _ = std::fs::create_dir_all(base.join(dir));
    }

    let mut stragglers = Vec::new();
    let mut rename = |old: PathBuf, new: PathBuf| {
        if !old.exists() {
            return;
        }
        if new.exists() {
            stragglers.push(
                old.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
            );
            return;
        }
        let _ = std::fs::rename(&old, &new);
    };

    for (name, new_path) in moved_files() {
        rename(base.join(name), new_path(base));
    }
    // SQLite WAL sidecars must travel with the db or recent commits are lost.
    for suffix in ["-wal", "-shm"] {
        rename(
            base.join(format!("logs.db{suffix}")),
            base.join(LOGS_DIR).join(format!("logs.db{suffix}")),
        );
    }
    rename(base.join("logs.db"), logs_db(base));
    rename(base.join("ollama-pids"), ollama_pids(base));

    // Per-tool stats caches have dynamic names; sweep by pattern. Same pass
    // drops leaked atomic-write temps (>1 day old) and stale manual .baks.
    if let Ok(entries) = std::fs::read_dir(base) {
        let day_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(24 * 60 * 60);
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("stats-cache-") && name.ends_with(".json") {
                rename(entry.path(), base.join(CACHE_DIR).join(&name));
            } else if name.starts_with("code-prefs.json.bak")
                || (name.starts_with(".aivo-tmp-")
                    && entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .is_ok_and(|t| t < day_ago))
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }

    stragglers
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn config_dir_honors_env_override() {
        // Env mutation is process-global; keep this the only test touching it.
        let dir = TempDir::new().unwrap();
        unsafe { std::env::set_var("AIVO_CONFIG_DIR", dir.path()) };
        assert_eq!(config_dir(), dir.path());
        unsafe { std::env::remove_var("AIVO_CONFIG_DIR") };
        assert!(
            config_dir().ends_with(".config/aivo") || config_dir() == Path::new(".config/aivo")
        );
    }

    #[test]
    fn migrates_flat_files_into_subdirs() {
        let dir = TempDir::new().unwrap();
        let base = dir.path();
        for name in [
            "stats.json",
            "device-key",
            "models-cache.json",
            "logs.db",
            "logs.db-wal",
        ] {
            std::fs::write(base.join(name), b"x").unwrap();
        }
        std::fs::create_dir(base.join("ollama-pids")).unwrap();
        std::fs::write(base.join("stats-cache-claude.json"), b"x").unwrap();

        let stragglers = migrate_layout(base);

        assert!(stragglers.is_empty());
        assert!(stats_json(base).exists());
        assert!(device_key(base).exists());
        assert!(models_cache(base).exists());
        assert!(logs_db(base).exists());
        assert!(base.join(LOGS_DIR).join("logs.db-wal").exists());
        assert!(ollama_pids(base).exists());
        assert!(stats_cache(base, "claude").exists());
        assert!(!base.join("stats.json").exists());
    }

    #[test]
    fn migration_is_idempotent_and_never_clobbers() {
        let dir = TempDir::new().unwrap();
        let base = dir.path();
        std::fs::write(base.join("stats.json"), b"old").unwrap();
        migrate_layout(base);
        assert_eq!(std::fs::read(stats_json(base)).unwrap(), b"old");

        // Straggler recreated by an older binary: reported, not merged.
        std::fs::write(base.join("stats.json"), b"downgrade").unwrap();
        let stragglers = migrate_layout(base);
        assert_eq!(stragglers, vec!["stats.json".to_string()]);
        assert_eq!(std::fs::read(stats_json(base)).unwrap(), b"old");
        assert!(base.join("stats.json").exists());
    }

    #[test]
    fn sweeps_stale_tmp_and_bak_files() {
        let dir = TempDir::new().unwrap();
        let base = dir.path();
        let stale = base.join(".aivo-tmp-abc123");
        std::fs::write(&stale, b"x").unwrap();
        let two_days_ago =
            std::time::SystemTime::now() - std::time::Duration::from_secs(2 * 24 * 60 * 60);
        // Windows: set_modified needs write access; close before the sweep deletes.
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(&stale)
            .unwrap();
        f.set_modified(two_days_ago).unwrap();
        drop(f);
        let fresh = base.join(".aivo-tmp-fresh");
        std::fs::write(&fresh, b"x").unwrap();
        std::fs::write(base.join("code-prefs.json.bak-before-disable-all"), b"x").unwrap();

        migrate_layout(base);

        assert!(!stale.exists(), "stale tmp should be swept");
        assert!(fresh.exists(), "in-flight tmp must survive");
        assert!(!base.join("code-prefs.json.bak-before-disable-all").exists());
    }

    #[cfg(unix)]
    #[test]
    fn secrets_dir_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        migrate_layout(dir.path());
        let mode = std::fs::metadata(dir.path().join(SECRETS_DIR))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            mode & 0o077,
            0,
            "secrets/ must not be group/other readable: {mode:o}"
        );
    }
}
