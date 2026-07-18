//! The plugin registry (`~/.config/aivo/plugins/.registry.json`): per-plugin
//! provenance the bare `aivo-<name>` file can't carry — install source (so
//! `update` can re-fetch), an integrity pin (`sha256:…`), and the cached
//! `--aivo-manifest`. Migrated transparently from the older source-only
//! `.sources.json`. A dotfile, so plugin discovery skips it.
//!
//! Reads (`load`) never touch disk. The on-disk migration and all writes happen
//! only on `install`/`update`/`remove` (via `record`/`forget`), atomically, and
//! a present-but-corrupt registry is moved aside rather than silently wiped.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::manifest::PluginManifest;
use super::plugins_dir;
use crate::style;

const REGISTRY_VERSION: u32 = 1;
const REGISTRY_FILE: &str = ".registry.json";
const CORRUPT_FILE: &str = ".registry.json.corrupt";
const LEGACY_SOURCES_FILE: &str = ".sources.json";

/// What aivo knows about one installed plugin. Everything but `source` is
/// optional so migrated and manifest-less plugins round-trip cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PluginRecord {
    /// Where it was installed from (absolute path or URL), for `update`.
    pub source: String,
    /// `sha256:<hex>` of the installed bytes at install time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PluginManifest>,
    /// RFC3339 timestamp of the last install/update (matches `ApiKey.created_at`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub installed_at: Option<String>,
    /// Grantable capabilities the user approved at install (or on first
    /// dispatch). Distinct from `manifest.capabilities` (what was *requested*):
    /// only these are acted on at launch. Empty until consent is given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub granted_caps: Vec<String>,
    /// The user confirmed running this binary (first-dispatch gate for remote
    /// installs whose manifest probe would otherwise be its first execution).
    #[serde(default, skip_serializing_if = "is_false")]
    pub run_approved: bool,
    /// The `checksum` the user's consent (run approval + caps) was given for.
    /// Consent is bound to these bytes: a record whose `checksum` drifts from
    /// this pin is re-gated at dispatch instead of inheriting approval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approved_checksum: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Registry {
    pub version: u32,
    pub plugins: BTreeMap<String, PluginRecord>,
}

impl Default for Registry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            plugins: BTreeMap::new(),
        }
    }
}

/// The full registry as a read-only view — **never writes**. A legacy
/// `.sources.json` is surfaced in memory so `list`/`--help-json` still show
/// pre-migration plugins; the on-disk migration happens on the next write.
pub(crate) fn load() -> Registry {
    match plugins_dir() {
        Some(dir) => read_view(&dir),
        None => Registry::default(),
    }
}

/// Insert or replace a plugin's record (best-effort atomic persist).
pub(crate) fn record(name: &str, rec: PluginRecord) {
    apply_default(|reg| {
        reg.plugins.insert(name.to_string(), rec);
    });
}

/// Drop a plugin's record (best-effort atomic persist).
pub(crate) fn forget(name: &str) {
    apply_default(|reg| {
        reg.plugins.remove(name);
    });
}

/// Mark a plugin's first run as user-approved without touching anything else —
/// used when install-on-demand consent already covered execution, so the
/// first-dispatch gate must not ask again.
pub(crate) fn approve_run(name: &str) {
    apply_default(|reg| approve_run_in(reg, name));
}

fn approve_run_in(reg: &mut Registry, name: &str) {
    if let Some(rec) = reg.plugins.get_mut(name) {
        rec.run_approved = true;
        // The consent just given covers exactly the bytes on record.
        rec.approved_checksum = rec.checksum.clone();
    }
}

/// Resolve the managed dir and `apply`; a persist failure only costs a future
/// `update`, so it warns rather than aborts.
fn apply_default(f: impl FnOnce(&mut Registry)) {
    let Some(dir) = plugins_dir() else {
        eprintln!(
            "  {} could not resolve ~/.config/aivo/plugins to update the registry",
            style::yellow("!")
        );
        return;
    };
    if let Err(e) = apply(&dir, f) {
        eprintln!(
            "  {} could not write the plugin registry: {e:#}",
            style::yellow("!")
        );
    }
}

// ── dir-parameterized core (unit-testable without $HOME) ───────────────────

/// Pure read. A missing or corrupt registry falls back to a legacy
/// `.sources.json` view (also pure); corruption is *repaired* only on the write
/// path (`load_for_write`), never here.
fn read_view(dir: &Path) -> Registry {
    if let Ok(text) = std::fs::read_to_string(dir.join(REGISTRY_FILE))
        && let Ok(reg) = serde_json::from_str::<Registry>(&text)
    {
        return reg;
    }
    sources_view(dir).unwrap_or_default()
}

/// Read a legacy `.sources.json` (`{name: source}`) into a `Registry` in memory.
/// No writes — the persisted migration happens in `apply`.
fn sources_view(dir: &Path) -> Option<Registry> {
    let text = std::fs::read_to_string(dir.join(LEGACY_SOURCES_FILE)).ok()?;
    let sources: BTreeMap<String, String> = serde_json::from_str(&text).ok()?;
    if sources.is_empty() {
        return None;
    }
    let plugins = sources
        .into_iter()
        .map(|(name, source)| {
            (
                name,
                PluginRecord {
                    source,
                    checksum: None,
                    manifest: None,
                    installed_at: None,
                    granted_caps: Vec::new(),
                    run_approved: false,
                    approved_checksum: None,
                },
            )
        })
        .collect();
    Some(Registry {
        version: REGISTRY_VERSION,
        plugins,
    })
}

/// Load (preserving a corrupt file by backing it up, not overwriting), apply
/// `f`, persist atomically, then finish any legacy migration by removing
/// `.sources.json` once the registry is authoritative on disk.
fn apply(dir: &Path, f: impl FnOnce(&mut Registry)) -> Result<()> {
    let mut reg = load_for_write(dir);
    f(&mut reg);
    save_to(dir, &reg)?;
    let _ = std::fs::remove_file(dir.join(LEGACY_SOURCES_FILE));
    Ok(())
}

/// Like `read_view`, but a *present-but-unparseable* registry is moved aside to
/// `.registry.json.corrupt` (with a warning) so the impending write can't wipe
/// it. Missing → migrate the legacy view.
fn load_for_write(dir: &Path) -> Registry {
    let path = dir.join(REGISTRY_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => match serde_json::from_str::<Registry>(&text) {
            Ok(reg) => reg,
            Err(e) => {
                let backup = dir.join(CORRUPT_FILE);
                let _ = std::fs::rename(&path, &backup);
                eprintln!(
                    "  {} plugin registry was unreadable ({e}); backed it up to {} and started fresh",
                    style::yellow("!"),
                    backup.display(),
                );
                Registry::default()
            }
        },
        Err(_) => sources_view(dir).unwrap_or_default(),
    }
}

fn save_to(dir: &Path, reg: &Registry) -> Result<()> {
    crate::services::json_store::save_blocking(&dir.join(REGISTRY_FILE), reg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn rec(source: &str) -> PluginRecord {
        PluginRecord {
            source: source.to_string(),
            checksum: Some("sha256:abc".to_string()),
            manifest: None,
            installed_at: Some("2026-06-04T00:00:00+00:00".to_string()),
            granted_caps: Vec::new(),
            run_approved: false,
            approved_checksum: None,
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let mut reg = Registry::default();
        reg.plugins.insert("amp".to_string(), rec("/abs/aivo-amp"));
        save_to(dir.path(), &reg).unwrap();

        let back = read_view(dir.path());
        assert_eq!(back.plugins["amp"].source, "/abs/aivo-amp");
        assert_eq!(back.plugins["amp"].checksum.as_deref(), Some("sha256:abc"));
    }

    #[test]
    fn read_view_is_pure_and_surfaces_legacy() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(LEGACY_SOURCES_FILE),
            r#"{"amp":"/abs/aivo-amp"}"#,
        )
        .unwrap();

        let view = read_view(dir.path());
        assert_eq!(view.plugins["amp"].source, "/abs/aivo-amp");
        assert!(view.plugins["amp"].manifest.is_none());
        // A read must neither migrate nor delete.
        assert!(dir.path().join(LEGACY_SOURCES_FILE).exists());
        assert!(!dir.path().join(REGISTRY_FILE).exists());
    }

    #[test]
    fn write_migrates_legacy_then_removes_it() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join(LEGACY_SOURCES_FILE),
            r#"{"amp":"/abs/aivo-amp"}"#,
        )
        .unwrap();

        apply(dir.path(), |reg| {
            reg.plugins.insert("new".to_string(), rec("/x"));
        })
        .unwrap();

        let reg = read_view(dir.path());
        assert!(reg.plugins.contains_key("amp"), "legacy entry carried over");
        assert!(reg.plugins.contains_key("new"));
        assert!(dir.path().join(REGISTRY_FILE).exists());
        assert!(!dir.path().join(LEGACY_SOURCES_FILE).exists());
    }

    #[test]
    fn corrupt_registry_is_backed_up_not_wiped() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(REGISTRY_FILE), "{ not json").unwrap();

        apply(dir.path(), |reg| {
            reg.plugins.insert("new".to_string(), rec("/x"));
        })
        .unwrap();

        assert!(
            dir.path().join(CORRUPT_FILE).exists(),
            "the unparseable file must be preserved, not overwritten"
        );
        assert!(read_view(dir.path()).plugins.contains_key("new"));
    }

    #[test]
    fn missing_files_yield_empty() {
        let dir = TempDir::new().unwrap();
        assert!(read_view(dir.path()).plugins.is_empty());
    }

    #[test]
    fn approve_run_flips_only_the_flag() {
        let dir = TempDir::new().unwrap();
        let mut reg = Registry::default();
        reg.plugins.insert("amp".to_string(), rec("/abs/aivo-amp"));
        save_to(dir.path(), &reg).unwrap();

        apply(dir.path(), |reg| approve_run_in(reg, "amp")).unwrap();
        let back = read_view(dir.path());
        assert!(back.plugins["amp"].run_approved);
        assert_eq!(back.plugins["amp"].source, "/abs/aivo-amp");
        // Approval is pinned to the bytes on record.
        assert_eq!(
            back.plugins["amp"].approved_checksum.as_deref(),
            Some("sha256:abc")
        );

        // An unknown name is a no-op, not an insert.
        apply(dir.path(), |reg| approve_run_in(reg, "ghost")).unwrap();
        assert!(!read_view(dir.path()).plugins.contains_key("ghost"));
    }

    #[test]
    fn granted_caps_round_trip_and_default_empty() {
        let dir = TempDir::new().unwrap();
        let mut r = rec("/x");
        r.granted_caps = vec!["endpoint".to_string()];
        let mut reg = Registry::default();
        reg.plugins.insert("omp".to_string(), r);
        save_to(dir.path(), &reg).unwrap();

        let back = read_view(dir.path());
        assert_eq!(back.plugins["omp"].granted_caps, ["endpoint"]);

        // A record persisted without the field deserializes to an empty vec.
        std::fs::write(
            dir.path().join(REGISTRY_FILE),
            r#"{"version":1,"plugins":{"old":{"source":"/o"}}}"#,
        )
        .unwrap();
        assert!(read_view(dir.path()).plugins["old"].granted_caps.is_empty());
    }
}
