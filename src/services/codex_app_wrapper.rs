//! Per-key shell wrapper that hooks the codex app-server subprocess Codex.app
//! spawns. The Electron main process reads `CODEX_CLI_PATH` from its env and
//! spawns `<that> app-server --analytics-default-enabled` (see Codex.app
//! v26.519.81530 `Pd()` in workspace-root-drop-handler-*.js). The wrapper
//! `exec`s the bundled codex with our `-c` flags prepended, so the GUI's
//! own app-server picks up aivo's provider/profile/catalog overrides — no
//! write to `~/.codex/config.toml` required.
//!
//! aivo's parent `codex app` invocation's `--config` flags are NOT propagated
//! to the GUI's spawned app-server (verified empirically). The wrapper is
//! what actually makes overrides visible to the codex subprocess that does
//! the real inference.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

const APP_NAME: &str = "Codex.app";
const BUNDLED_CODEX: &str = "Contents/Resources/codex";

/// Locates `Codex.app` on disk. Honors `AIVO_CODEX_APP_PATH` first (testing),
/// then `/Applications`, then `~/Applications`. Returns `None` on non-macOS
/// or when the bundle is missing.
pub fn locate_codex_app() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AIVO_CODEX_APP_PATH") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let system = PathBuf::from("/Applications").join(APP_NAME);
        if system.is_dir() {
            return Some(system);
        }
        if let Some(home) = crate::services::system_env::home_dir() {
            let user = home.join("Applications").join(APP_NAME);
            if user.is_dir() {
                return Some(user);
            }
        }
    }
    None
}

/// Returns the codex binary bundled inside Codex.app, the one whose protocol
/// version matches the GUI app-server expectations.
pub fn locate_bundled_codex() -> Option<PathBuf> {
    let app = locate_codex_app()?;
    let codex = app.join(BUNDLED_CODEX);
    codex.is_file().then_some(codex)
}

/// Writes a per-launch wrapper script under `parent_dir` and returns its path.
/// The script `exec`s `codex_bin` with each entry of `extra_args` passed as a
/// separate, single-quoted argument, then forwards `"$@"`.
///
/// The filename includes `<key>-<pid>-<nanos>.sh` so concurrent
/// `aivo codex-app -k <same-key>` runs don't clobber each other — Codex.app
/// captures `CODEX_CLI_PATH` at GUI launch and re-reads the file on app-server
/// restarts within a session, so a shared filename would let a second launch's
/// wrapper point the first's Codex.app at a now-dead router port.
pub async fn write_wrapper(
    parent_dir: &Path,
    key_id: &str,
    codex_bin: &Path,
    extra_args: &[String],
) -> Result<PathBuf> {
    tokio::fs::create_dir_all(parent_dir)
        .await
        .with_context(|| format!("create codex-app wrapper dir {}", parent_dir.display()))?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = parent_dir.join(format!(
        "{}-{}-{}.sh",
        sanitize_filename(key_id),
        std::process::id(),
        nonce
    ));
    let script = build_script(codex_bin, extra_args);
    tokio::fs::write(&path, script.as_bytes())
        .await
        .with_context(|| format!("write codex-app wrapper {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = tokio::fs::metadata(&path).await?.permissions();
        perms.set_mode(0o755);
        tokio::fs::set_permissions(&path, perms).await?;
    }
    Ok(path)
}

/// Conventional parent dir for wrappers under aivo's config dir.
pub fn wrapper_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("codex-app-wrappers")
}

/// Best-effort cleanup of stale wrapper files (older than ~1 day) under
/// `parent_dir`. Called at launch start so per-launch wrapper accretion from
/// crashed / SIGKILLed aivo runs doesn't grow unbounded. Silently ignores I/O
/// errors — cleanup is opportunistic.
pub async fn cleanup_stale_wrappers(parent_dir: &Path) {
    use std::time::{Duration, SystemTime};
    const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
    let Ok(mut entries) = tokio::fs::read_dir(parent_dir).await else {
        return;
    };
    let now = SystemTime::now();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("sh") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        if now.duration_since(modified).is_ok_and(|age| age > MAX_AGE) {
            let _ = tokio::fs::remove_file(&path).await;
        }
    }
}

fn build_script(codex_bin: &Path, extra_args: &[String]) -> String {
    let mut out = String::from("#!/bin/sh\n");
    out.push_str("# aivo-managed wrapper — do not edit by hand. Overwritten on each `aivo codex-app` launch.\n");
    out.push_str("exec ");
    out.push_str(&shell_quote(&codex_bin.to_string_lossy()));
    for arg in extra_args {
        out.push_str(" \\\n  ");
        out.push_str(&shell_quote(arg));
    }
    out.push_str(" \"$@\"\n");
    out
}

fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn sanitize_filename(key_id: &str) -> String {
    let mut out = String::with_capacity(key_id.len());
    for ch in key_id.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "default".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_wraps_simple_values() {
        assert_eq!(shell_quote("foo"), "'foo'");
        assert_eq!(shell_quote("with space"), "'with space'");
    }

    #[test]
    fn shell_quote_escapes_embedded_single_quote() {
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn shell_quote_preserves_inline_toml() {
        // The exact kind of value the wrapper receives.
        let toml = r#"model_providers.aivo = {name="aivo", base_url="http://127.0.0.1:8080/v1/"}"#;
        let quoted = shell_quote(toml);
        assert!(quoted.starts_with('\''));
        assert!(quoted.ends_with('\''));
        assert!(quoted.contains("name=\"aivo\""));
    }

    #[test]
    fn sanitize_filename_strips_path_separators() {
        assert_eq!(
            sanitize_filename("ahq/key with space"),
            "ahq_key_with_space"
        );
        assert_eq!(sanitize_filename(""), "default");
        assert_eq!(sanitize_filename("abc-123_xyz"), "abc-123_xyz");
    }

    #[test]
    fn build_script_emits_valid_shell() {
        let bin = PathBuf::from("/Applications/Codex.app/Contents/Resources/codex");
        let flags = vec![
            "--config".to_string(),
            r#"model_provider="aivo""#.to_string(),
            "--config".to_string(),
            r#"model_providers.aivo.base_url="http://127.0.0.1:1234/v1/""#.to_string(),
            "--disable".to_string(),
            "apps".to_string(),
        ];
        let script = build_script(&bin, &flags);
        assert!(script.starts_with("#!/bin/sh\n"));
        assert!(script.contains("exec '/Applications/Codex.app/Contents/Resources/codex'"));
        assert!(script.contains(r#"'model_provider="aivo"'"#));
        assert!(script.contains(r#"'model_providers.aivo.base_url="http://127.0.0.1:1234/v1/"'"#));
        assert!(script.contains("'--disable'"));
        assert!(script.contains("'apps'"));
        assert!(script.trim_end().ends_with("\"$@\""));
    }

    #[tokio::test]
    async fn write_wrapper_produces_executable_file() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let tmp = tempfile::tempdir().unwrap();
            let bin = PathBuf::from("/bin/echo");
            let path = write_wrapper(
                tmp.path(),
                "my/key id",
                &bin,
                &["--config".into(), "x=1".into()],
            )
            .await
            .unwrap();
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            assert!(
                name.starts_with("my_key_id-") && name.ends_with(".sh"),
                "expected my_key_id-<pid>-<nanos>.sh, got {name}"
            );
            let perms = tokio::fs::metadata(&path).await.unwrap().permissions();
            assert_eq!(perms.mode() & 0o777, 0o755);
            let body = tokio::fs::read_to_string(&path).await.unwrap();
            assert!(body.contains("exec '/bin/echo'"));
            assert!(body.contains("'--config'"));
            assert!(body.contains("'x=1'"));
        }
    }

    #[tokio::test]
    async fn write_wrapper_uses_distinct_paths_per_launch() {
        // Per-launch unique filenames prevent concurrent `aivo codex-app -k k`
        // runs from clobbering each other — see the failure_scenario in the
        // code-review finding for concurrent wrapper races.
        let tmp = tempfile::tempdir().unwrap();
        let bin = PathBuf::from("/bin/true");
        let p = write_wrapper(tmp.path(), "k", &bin, &["--config".into(), "old=1".into()])
            .await
            .unwrap();
        // Sleep at least 1ns to guarantee a distinct nanosecond timestamp.
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        let p2 = write_wrapper(tmp.path(), "k", &bin, &["--config".into(), "new=2".into()])
            .await
            .unwrap();
        assert_ne!(p, p2, "concurrent launches must get distinct wrapper paths");
        let body1 = tokio::fs::read_to_string(&p).await.unwrap();
        assert!(body1.contains("old=1"));
        let body2 = tokio::fs::read_to_string(&p2).await.unwrap();
        assert!(body2.contains("new=2"));
    }

    #[tokio::test]
    async fn cleanup_stale_wrappers_preserves_fresh_files() {
        // Backdating an mtime portably from a test would pull in an extra
        // crate; assert at least that the reaper doesn't touch recently-
        // written files, and that it tolerates a missing directory.
        let tmp = tempfile::tempdir().unwrap();
        let bin = PathBuf::from("/bin/true");
        let fresh = write_wrapper(tmp.path(), "k", &bin, &["--config".into(), "x=1".into()])
            .await
            .unwrap();
        cleanup_stale_wrappers(tmp.path()).await;
        assert!(fresh.exists(), "fresh wrapper should be kept");
        cleanup_stale_wrappers(&tmp.path().join("does-not-exist")).await; // no panic
    }

    #[test]
    fn wrapper_dir_under_config_dir() {
        let cfg = PathBuf::from("/cfg");
        assert_eq!(wrapper_dir(&cfg), PathBuf::from("/cfg/codex-app-wrappers"));
    }
}
