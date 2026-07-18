//! Shared load/save orchestration for aivo's small JSON state files.
//!
//! Every store used to hand-roll the same read→parse→default-on-missing→
//! atomic-save sequence, and two of them independently reinvented the subtle
//! part: a *lenient* read for the load path (missing/corrupt → default, the
//! store just looks empty) versus a *strict* read for the write path (a
//! present-but-unparseable file is an error, so a save never rewrites the
//! store from scratch and destroys entries it couldn't see). These helpers are
//! that sequence, once.
//!
//! Not for `config.json` (owned by `session_store::ConfigContext`, which adds
//! locking and field-level crypto) or SQLite stores.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::services::atomic_write::{
    atomic_write_secure, atomic_write_secure_blocking, ensure_private_dir,
    ensure_private_dir_blocking,
};

/// Lenient read: `None` when the file is missing, unreadable, or unparseable.
/// For load paths where a corrupt store should read as "nothing stored".
pub(crate) fn load_optional<T: DeserializeOwned>(path: &Path) -> Option<T> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Lenient read with a default: missing/corrupt → `T::default()`.
pub(crate) fn load_or_default<T: DeserializeOwned + Default>(path: &Path) -> T {
    load_optional(path).unwrap_or_default()
}

/// Strict read for the write path: a present-but-unparseable file is an
/// error — the caller must not save over data it failed to read. A genuinely
/// absent file starts fresh.
pub(crate) fn load_for_write<T: DeserializeOwned + Default>(path: &Path) -> Result<T> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).with_context(|| {
            format!(
                "{} is present but unparseable; refusing to overwrite it",
                path.display()
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(anyhow::Error::new(e).context(format!("read {}", path.display()))),
    }
}

/// Async variant of [`load_for_write`] for stores on the tokio path.
pub(crate) async fn load_for_write_async<T: DeserializeOwned + Default>(path: &Path) -> Result<T> {
    match tokio::fs::read_to_string(path).await {
        Ok(text) => serde_json::from_str(&text).with_context(|| {
            format!(
                "{} is present but unparseable; refusing to overwrite it",
                path.display()
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(anyhow::Error::new(e).context(format!("read {}", path.display()))),
    }
}

/// Pretty-serialize and atomically persist (parent dir created 0700, file 0600).
pub(crate) async fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        ensure_private_dir(parent).await?;
    }
    let data = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", path.display()))?;
    atomic_write_secure(path, data)
        .await
        .with_context(|| format!("write {}", path.display()))
}

/// Blocking variant of [`save`] for sync contexts.
pub(crate) fn save_blocking<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        ensure_private_dir_blocking(parent)?;
    }
    let data = serde_json::to_vec_pretty(value)
        .with_context(|| format!("serialize {}", path.display()))?;
    atomic_write_secure_blocking(path, &data).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("aivo-json-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[tokio::test]
    async fn save_then_load_round_trips() {
        let path = tmp("round.json");
        let mut map = BTreeMap::new();
        map.insert("k".to_string(), 1u32);
        save(&path, &map).await.unwrap();
        assert_eq!(load_optional::<BTreeMap<String, u32>>(&path), Some(map));
    }

    #[test]
    fn load_optional_missing_and_corrupt_are_none() {
        assert_eq!(load_optional::<u32>(&tmp("missing.json")), None);
        let path = tmp("corrupt-lenient.json");
        std::fs::write(&path, b"{not json").unwrap();
        assert_eq!(load_optional::<u32>(&path), None);
        assert_eq!(load_or_default::<u32>(&path), 0);
    }

    #[test]
    fn load_for_write_guards_against_clobbering() {
        // Missing file starts fresh…
        let missing = tmp("fresh.json");
        assert_eq!(
            load_for_write::<BTreeMap<String, u32>>(&missing).unwrap(),
            BTreeMap::new()
        );
        // …but a present-yet-unparseable one refuses, so the caller can't
        // rewrite the store from an empty map.
        let path = tmp("corrupt-strict.json");
        std::fs::write(&path, b"{not json").unwrap();
        let err = load_for_write::<BTreeMap<String, u32>>(&path).unwrap_err();
        assert!(err.to_string().contains("refusing to overwrite"));
    }

    #[tokio::test]
    async fn load_for_write_async_matches_sync_semantics() {
        let path = tmp("corrupt-async.json");
        std::fs::write(&path, b"{not json").unwrap();
        assert!(
            load_for_write_async::<BTreeMap<String, u32>>(&path)
                .await
                .is_err()
        );
        assert_eq!(
            load_for_write_async::<BTreeMap<String, u32>>(&tmp("fresh-async.json"))
                .await
                .unwrap(),
            BTreeMap::new()
        );
    }

    #[test]
    fn save_blocking_creates_parent_dir() {
        let path = tmp("nested").join("deep").join("file.json");
        save_blocking(&path, &42u32).unwrap();
        assert_eq!(load_optional::<u32>(&path), Some(42));
    }
}
