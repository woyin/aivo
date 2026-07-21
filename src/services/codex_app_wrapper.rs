//! Per-key wrapper that hooks the codex app-server subprocess the desktop app
//! spawns: the Electron main reads `CODEX_CLI_PATH` from its env and spawns
//! `<that> app-server --analytics-default-enabled`. The wrapper re-execs the
//! bundled codex with aivo's `-c` flags prepended — the only channel that
//! reaches the GUI's app-server without writing `~/.codex/config.toml`.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Bundle names the Codex desktop app has shipped under: `ChatGPT.app` from
/// v26.707 on, `Codex.app` through v26.6xx. Newer name first so a dual
/// install prefers the bundle LaunchServices will actually launch.
#[cfg(target_os = "macos")]
const APP_NAMES: [&str; 2] = ["ChatGPT.app", "Codex.app"];
#[cfg(not(windows))]
const BUNDLED_CODEX: &str = "Contents/Resources/codex";

const WRAPPER_DIR_NAME: &str = "codex-app-wrappers";

/// Locates the Codex desktop app bundle: `AIVO_CODEX_APP_PATH` first
/// (testing), then `/Applications` and `~/Applications` per `APP_NAMES`.
/// Named candidates must contain the bundled codex — that distinguishes the
/// Codex app from the legacy chat-only `ChatGPT.app` (`com.openai.chat`).
pub fn locate_codex_app() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AIVO_CODEX_APP_PATH") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let user_apps = crate::services::system_env::home_dir().map(|h| h.join("Applications"));
        for name in APP_NAMES {
            for dir in std::iter::once(PathBuf::from("/Applications")).chain(user_apps.clone()) {
                let app = dir.join(name);
                if app.join(BUNDLED_CODEX).is_file() {
                    return Some(app);
                }
            }
        }
    }
    None
}

/// Returns the codex binary bundled inside Codex.app, the one whose protocol
/// version matches the GUI app-server expectations.
#[cfg(not(windows))]
pub fn locate_bundled_codex() -> Option<PathBuf> {
    bundled_codex_in(&locate_codex_app()?)
}

/// The bundled codex inside an already-located app bundle.
#[cfg(not(windows))]
pub fn bundled_codex_in(app: &Path) -> Option<PathBuf> {
    let codex = app.join(BUNDLED_CODEX);
    codex.is_file().then_some(codex)
}

/// Windows: the MSIX package dir is ACL-locked, so the GUI materializes its
/// bundled codex into `<CODEX_HOME>\bin` on first run (CODEX_HOME = env
/// override, else `~\.codex`; the GUI force-sets the same on its app-server).
/// Older builds used `%LOCALAPPDATA%\OpenAI\Codex\bin`. Package resources
/// are a last-ditch probe (listing usually denied). `AIVO_CODEX_APP_PATH`
/// may be the codex.exe itself or a package root.
#[cfg(windows)]
pub fn locate_bundled_codex() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("AIVO_CODEX_APP_PATH") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
        let nested = p.join("app").join("resources").join("codex.exe");
        if nested.is_file() {
            return Some(nested);
        }
    }
    let codex_home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| crate::services::system_env::home_dir().map(|h| h.join(".codex")));
    if let Some(home) = codex_home
        && let Some(hit) = newest_codex_exe(&home.join("bin"))
    {
        return Some(hit);
    }
    if let Some(lad) = std::env::var_os("LOCALAPPDATA")
        && let Some(hit) =
            newest_codex_exe(&PathBuf::from(lad).join("OpenAI").join("Codex").join("bin"))
    {
        return Some(hit);
    }
    for var in ["ProgramFiles", "ProgramW6432"] {
        let Some(root) = std::env::var_os(var) else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(PathBuf::from(root).join("WindowsApps")) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with("OpenAI.Codex_")
            {
                let codex = entry.path().join("app").join("resources").join("codex.exe");
                if codex.is_file() {
                    return Some(codex);
                }
            }
        }
    }
    None
}

/// `bin\codex.exe`, else the newest-mtime `bin\<version>\codex.exe` — the GUI
/// has shipped both flat and versioned layouts.
#[cfg(windows)]
fn newest_codex_exe(bin: &Path) -> Option<PathBuf> {
    let flat = bin.join("codex.exe");
    if flat.is_file() {
        return Some(flat);
    }
    let entries = std::fs::read_dir(bin).ok()?;
    entries
        .flatten()
        .filter_map(|e| {
            let codex = e.path().join("codex.exe");
            let modified = codex.metadata().ok()?.modified().ok()?;
            Some((modified, codex))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, codex)| codex)
}

/// Sidecar consumed by the Windows shim. Node spawns `CODEX_CLI_PATH` without
/// a shell and rejects `.cmd`/`.bat`, so the wrapper must be a real PE — a
/// copy of aivo.exe — and `<wrapper>.json` carries what the shell script
/// embeds on Unix.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ShimSidecar {
    pub codex_bin: PathBuf,
    pub prefix_args: Vec<String>,
}

fn sidecar_path_for(wrapper: &Path) -> PathBuf {
    let mut os = wrapper.as_os_str().to_owned();
    os.push(".json");
    PathBuf::from(os)
}

/// Best-effort removal of a wrapper and its Windows sidecar.
pub async fn remove_wrapper(wrapper: &Path) {
    let _ = tokio::fs::remove_file(wrapper).await;
    let _ = tokio::fs::remove_file(sidecar_path_for(wrapper)).await;
}

/// Pre-CLI hook: when this process is a wrapper copy (in `codex-app-wrappers`
/// with a sidecar), re-exec the bundled codex with the sidecar prefix plus
/// the GUI's argv. Any sidecar failure exits — falling through would pump
/// aivo output into the GUI's app-server stdio.
#[cfg(windows)]
pub fn maybe_run_windows_shim() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let in_wrapper_dir = exe
        .parent()
        .and_then(|p| p.file_name())
        .is_some_and(|name| name == WRAPPER_DIR_NAME);
    if !in_wrapper_dir {
        return;
    }
    let sidecar_path = sidecar_path_for(&exe);
    if !sidecar_path.exists() {
        return;
    }
    let sidecar: ShimSidecar = match std::fs::read(&sidecar_path)
        .map_err(anyhow::Error::from)
        .and_then(|bytes| serde_json::from_slice(&bytes).map_err(anyhow::Error::from))
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "aivo codex-app shim: bad sidecar {}: {e}",
                sidecar_path.display()
            );
            std::process::exit(1);
        }
    };
    let status = std::process::Command::new(&sidecar.codex_bin)
        .args(&sidecar.prefix_args)
        .args(std::env::args_os().skip(1))
        .status();
    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!(
                "aivo codex-app shim: spawn {} failed: {e}",
                sidecar.codex_bin.display()
            );
            std::process::exit(1);
        }
    }
}

/// Writes a per-launch wrapper under `parent_dir`: a shell script `exec`ing
/// `codex_bin` with `extra_args` prepended (Unix), or an aivo.exe copy plus
/// [`ShimSidecar`] (Windows; see `maybe_run_windows_shim`).
///
/// The filename includes `<key>-<pid>-<nanos>` so concurrent
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
    let stem = format!(
        "{}-{}-{}",
        sanitize_filename(key_id),
        std::process::id(),
        nonce
    );
    #[cfg(not(windows))]
    {
        let path = parent_dir.join(format!("{stem}.sh"));
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
    #[cfg(windows)]
    {
        let path = parent_dir.join(format!("{stem}.exe"));
        let me = std::env::current_exe().context("resolve current aivo executable for shim")?;
        // Hardlink instead of copying ~10 MB; `current_exe` on the spawned
        // link resolves to the link path, so the sidecar lookup still works.
        if std::fs::hard_link(&me, &path).is_err() {
            tokio::fs::copy(&me, &path)
                .await
                .with_context(|| format!("copy aivo shim to {}", path.display()))?;
        }
        let sidecar = ShimSidecar {
            codex_bin: codex_bin.to_path_buf(),
            prefix_args: extra_args.to_vec(),
        };
        crate::services::json_store::save(&sidecar_path_for(&path), &sidecar)
            .await
            .with_context(|| format!("write shim sidecar for {}", path.display()))?;
        Ok(path)
    }
}

/// Conventional parent dir for wrappers under aivo's config dir.
pub fn wrapper_dir(config_dir: &Path) -> PathBuf {
    config_dir.join(WRAPPER_DIR_NAME)
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
        // `.sh` on Unix; `.exe` shim copies + `.json` sidecars on Windows.
        if !matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("sh" | "exe" | "json")
        ) {
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

#[cfg(not(windows))]
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

#[cfg(not(windows))]
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

    #[cfg(not(windows))]
    #[test]
    fn shell_quote_wraps_simple_values() {
        assert_eq!(shell_quote("foo"), "'foo'");
        assert_eq!(shell_quote("with space"), "'with space'");
    }

    #[cfg(not(windows))]
    #[test]
    fn shell_quote_escapes_embedded_single_quote() {
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[cfg(not(windows))]
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

    #[cfg(not(windows))]
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
        // Args live in the script body on Unix, in the sidecar on Windows.
        let args_carrier = |p: &Path| {
            if cfg!(windows) {
                sidecar_path_for(p)
            } else {
                p.to_path_buf()
            }
        };
        let body1 = tokio::fs::read_to_string(args_carrier(&p)).await.unwrap();
        assert!(body1.contains("old=1"));
        let body2 = tokio::fs::read_to_string(args_carrier(&p2)).await.unwrap();
        assert!(body2.contains("new=2"));
    }

    #[test]
    fn sidecar_path_appends_json_to_full_name() {
        assert_eq!(
            sidecar_path_for(Path::new("/cfg/codex-app-wrappers/k-1-2.exe")),
            PathBuf::from("/cfg/codex-app-wrappers/k-1-2.exe.json")
        );
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn windows_wrapper_is_exe_copy_with_sidecar() {
        let tmp = tempfile::tempdir().unwrap();
        let bin = PathBuf::from(r"C:\codex\codex.exe");
        let p = write_wrapper(tmp.path(), "k", &bin, &["--config".into(), "x=1".into()])
            .await
            .unwrap();
        assert_eq!(p.extension().and_then(|e| e.to_str()), Some("exe"));
        let me_len = std::fs::metadata(std::env::current_exe().unwrap())
            .unwrap()
            .len();
        assert_eq!(std::fs::metadata(&p).unwrap().len(), me_len);
        let sidecar: ShimSidecar =
            serde_json::from_slice(&std::fs::read(sidecar_path_for(&p)).unwrap()).unwrap();
        assert_eq!(sidecar.codex_bin, bin);
        assert_eq!(
            sidecar.prefix_args,
            vec!["--config".to_string(), "x=1".to_string()]
        );
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
