//! Background "update available" check. A detached `aivo __update-check` child
//! refreshes a weekly cache; the next command reads it and nags (stderr, TTY
//! only) if a newer release is out. Non-blocking and silent on failure; only
//! contacts the `getaivo.dev/dl/latest` endpoint `aivo update` uses. Opt out
//! with AIVO_NO_UPDATE_NOTICE / NO_UPDATE_NOTIFIER / CI.

use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const LATEST_URL: &str = "https://getaivo.dev/dl/latest";
const CHECK_INTERVAL_SECS: i64 = 7 * 24 * 60 * 60;

/// Last network check result, persisted between runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateCheck {
    /// RFC3339 stamp of the last check; None/absent → never checked (always due).
    #[serde(default)]
    pub last_check: Option<String>,
    /// Latest version from the server (no leading `v`); empty if unknown.
    #[serde(default)]
    pub latest_version: String,
}

fn cache_path() -> Option<PathBuf> {
    Some(crate::services::paths::update_check(
        &crate::services::paths::config_dir(),
    ))
}

pub fn load() -> Option<UpdateCheck> {
    load_from(&cache_path()?)
}

fn load_from(path: &Path) -> Option<UpdateCheck> {
    crate::services::json_store::load_optional(path)
}

/// Best-effort persist (0600); a failed write just means we re-check sooner.
fn save_blocking(check: &UpdateCheck) {
    let Some(path) = cache_path() else {
        return;
    };
    let _ = crate::services::json_store::save_blocking(&path, check);
}

/// Semver compare (a pre-release sorts below its release); shared with `UpdateCommand`.
pub fn is_newer_version(latest: &str, current: &str) -> bool {
    let parse = |version: &str| -> (Vec<u32>, bool) {
        let cleaned = version.trim_start_matches('v');
        let (num, has_pre) = match cleaned.split_once('-') {
            Some((v, _)) => (v, true),
            None => (cleaned, false),
        };
        let parts = num
            .split('.')
            .filter_map(|p| p.parse::<u32>().ok())
            .collect();
        (parts, has_pre)
    };

    let (latest_parts, latest_pre) = parse(latest);
    let (current_parts, current_pre) = parse(current);

    let max_len = latest_parts.len().max(current_parts.len());
    for i in 0..max_len {
        let l = latest_parts.get(i).copied().unwrap_or(0);
        let c = current_parts.get(i).copied().unwrap_or(0);
        if l > c {
            return true;
        }
        if l < c {
            return false;
        }
    }
    // Equal numbers: a release outranks a pre-release.
    current_pre && !latest_pre
}

pub fn pending_upgrade(cache: Option<&UpdateCheck>, current: &str) -> Option<String> {
    let latest = cache?.latest_version.trim();
    (!latest.is_empty() && is_newer_version(latest, current)).then(|| latest.to_string())
}

/// No cache, no/unparseable `last_check`, or older than the interval.
fn due_for_check(cache: Option<&UpdateCheck>, now: DateTime<Utc>) -> bool {
    let Some(last) = cache.and_then(|c| c.last_check.as_deref()) else {
        return true;
    };
    match DateTime::parse_from_rfc3339(last) {
        Ok(ts) => {
            now.signed_duration_since(ts.with_timezone(&Utc))
                .num_seconds()
                >= CHECK_INTERVAL_SECS
        }
        Err(_) => true,
    }
}

/// Opt-out check (testable seam); CI present — even empty — counts.
fn disabled_by(get: impl Fn(&str) -> Option<String>) -> bool {
    let truthy = |k: &str| get(k).is_some_and(|v| !v.is_empty() && v != "0");
    truthy("AIVO_NO_UPDATE_NOTICE") || truthy("NO_UPDATE_NOTIFIER") || get("CI").is_some()
}

pub fn disabled() -> bool {
    disabled_by(|k| std::env::var(k).ok())
}

/// One-line notice to stderr when a newer version is cached. No-op off a TTY or opted out.
pub fn maybe_print_notice(current: &str) {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() || disabled() {
        return;
    }
    let Some(latest) = pending_upgrade(load().as_ref(), current) else {
        return;
    };
    let cmd = crate::commands::update::upgrade_command_for_current_install();
    eprintln!(
        "✨ Update available: {} {} {} {} run {}",
        crate::style::dim(current),
        crate::style::dim("→"),
        crate::style::green(&latest),
        crate::style::dim("·"),
        crate::style::green(format!("`{cmd}`")),
    );
}

/// If due, debounce-stamp and fire a detached `aivo __update-check`; the next run reads its result.
pub fn maybe_spawn_background_check() {
    if disabled() {
        return;
    }
    let cache = load();
    let now = Utc::now();
    if !due_for_check(cache.as_ref(), now) {
        return;
    }
    // Debounce so an offline/failed check doesn't respawn on every command.
    save_blocking(&UpdateCheck {
        last_check: Some(now.to_rfc3339()),
        latest_version: cache.map(|c| c.latest_version).unwrap_or_default(),
    });
    spawn_detached_check();
}

fn spawn_detached_check() {
    use std::process::Stdio;
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("__update-check")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(unix)]
    {
        // setsid: detach from our process group/terminal (avoids SIGHUP on close).
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    #[cfg(windows)]
    {
        // DETACHED_PROCESS | CREATE_NO_WINDOW: no console window/flash.
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0000_0008 | 0x0800_0000);
    }
    let _ = cmd.spawn();
}

/// The hidden `aivo __update-check` subcommand: fetch, write cache, exit. Silent.
pub async fn run_check_and_exit() -> ! {
    if let Some(latest) = fetch_latest().await {
        save_blocking(&UpdateCheck {
            last_check: Some(Utc::now().to_rfc3339()),
            latest_version: latest,
        });
    }
    std::process::exit(0);
}

/// Like `get_latest_version`, but silent (Option) with a short timeout.
async fn fetch_latest() -> Option<String> {
    let client = crate::services::http_utils::aivo_http_client_builder()
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client
        .get(LATEST_URL)
        .header("User-Agent", "aivo-cli")
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let version = resp
        .text()
        .await
        .ok()?
        .trim()
        .trim_start_matches('v')
        .to_string();
    let valid = !version.is_empty()
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
    valid.then_some(version)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(last_check: &str, latest: &str) -> UpdateCheck {
        UpdateCheck {
            last_check: Some(last_check.into()),
            latest_version: latest.into(),
        }
    }

    #[test]
    fn round_trips_through_json() {
        let c = check("2026-06-30T00:00:00Z", "0.34.0");
        let back: UpdateCheck = serde_json::from_slice(&serde_json::to_vec(&c).unwrap()).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn load_from_missing_or_corrupt_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_from(&dir.path().join("nope.json")).is_none());
        let bad = dir.path().join("update_check.json");
        std::fs::write(&bad, b"not json").unwrap();
        assert!(load_from(&bad).is_none());
    }

    #[test]
    fn latest_version_defaults_when_absent() {
        let c: UpdateCheck = serde_json::from_slice(br#"{"last_check":"t"}"#).unwrap();
        assert_eq!(c.latest_version, "");
    }

    #[test]
    fn absent_or_null_last_check_loads_and_keeps_version() {
        for json in [
            &br#"{"latest_version":"0.40.0"}"#[..],
            &br#"{"last_check":null,"latest_version":"0.40.0"}"#[..],
        ] {
            let c: UpdateCheck = serde_json::from_slice(json).unwrap();
            assert_eq!(c.last_check, None);
            assert_eq!(c.latest_version, "0.40.0");
        }
    }

    #[test]
    fn is_newer_version_basics() {
        assert!(is_newer_version("1.1.0", "1.0.0"));
        assert!(is_newer_version("1.0.1", "1.0.0"));
        assert!(is_newer_version("v1.1.0", "v1.0.0"));
        assert!(!is_newer_version("1.0.0", "1.0.0"));
        assert!(!is_newer_version("0.9.0", "1.0.0"));
        assert!(is_newer_version("2.0.0", "2.0.0-rc1"));
        assert!(!is_newer_version("2.0.0-rc1", "2.0.0"));
        assert!(is_newer_version("2.1.0-rc1", "2.0.0"));
    }

    #[test]
    fn pending_upgrade_only_when_newer() {
        assert_eq!(
            pending_upgrade(Some(&check("t", "0.34.0")), "0.33.2"),
            Some("0.34.0".to_string())
        );
        assert_eq!(pending_upgrade(Some(&check("t", "0.33.2")), "0.33.2"), None);
        assert_eq!(pending_upgrade(Some(&check("t", "0.10.0")), "0.33.2"), None);
        assert_eq!(pending_upgrade(None, "0.33.2"), None);
        assert_eq!(pending_upgrade(Some(&check("t", "")), "0.33.2"), None);
    }

    #[test]
    fn due_for_check_window() {
        let now = DateTime::parse_from_rfc3339("2026-06-30T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(due_for_check(None, now), "missing cache is due");
        let never = UpdateCheck {
            last_check: None,
            latest_version: "0.1.0".into(),
        };
        assert!(due_for_check(Some(&never), now), "null last_check is due");
        assert!(
            due_for_check(Some(&check("garbage", "0.1.0")), now),
            "unparseable stamp is due"
        );
        assert!(
            due_for_check(Some(&check("2026-06-22T11:00:00Z", "0.1.0")), now),
            "older than 7 days is due"
        );
        assert!(
            !due_for_check(Some(&check("2026-06-25T13:00:00Z", "0.1.0")), now),
            "within 7 days is not due"
        );
    }

    #[test]
    fn disabled_by_respects_opt_out_vars() {
        let with = |k: &str, v: &str| {
            let (key, val) = (k.to_string(), v.to_string());
            disabled_by(move |q| (q == key).then(|| val.clone()))
        };
        assert!(with("AIVO_NO_UPDATE_NOTICE", "1"));
        assert!(with("NO_UPDATE_NOTIFIER", "true"));
        assert!(with("CI", ""));
        assert!(!with("AIVO_NO_UPDATE_NOTICE", "0"));
        assert!(!with("AIVO_NO_UPDATE_NOTICE", ""));
        assert!(!disabled_by(|_| None));
    }

    #[test]
    fn save_then_load_round_trips_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("update_check.json");
        let c = check("2026-06-30T00:00:00Z", "0.40.0");
        std::fs::write(&path, serde_json::to_vec_pretty(&c).unwrap()).unwrap();
        assert_eq!(load_from(&path), Some(c));
    }
}
