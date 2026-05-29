//! Per-aivo-key isolated home dir for cursor-agent, so multiple logins
//! coexist without touching `~/.cursor`. See the "cursor-agent env-var matrix"
//! section in `ARCHITECTURE.md` for the platform env-var matrix and the
//! `CURSOR_CONFIG_DIR`/`CURSOR_DATA_DIR` pinning rationale.

use anyhow::{Context, Result};
use rand::RngCore;
use std::ffi::OsString;
use std::path::PathBuf;

const ACCOUNT_ID_LEN: usize = 12;
const ACCOUNTS_DIR_NAME: &str = "cursor-accounts";

/// On-disk layout for one cursor account.
#[derive(Debug, Clone)]
pub struct CursorShadow {
    pub account_id: String,
    pub root: PathBuf,
}

impl CursorShadow {
    /// Compute a shadow path for a given account id. Does not touch disk.
    pub fn for_account_id(account_id: impl Into<String>) -> Result<Self> {
        let account_id = account_id.into();
        ensure_valid_account_id(&account_id)?;
        let root = accounts_dir()?.join(&account_id);
        Ok(Self { account_id, root })
    }

    /// Generate a fresh account id and create the shadow on disk. Used by
    /// the `aivo keys add cursor` flow before `cursor-agent login` runs.
    pub fn create_new() -> Result<Self> {
        let shadow = Self::for_account_id(generate_account_id())?;
        shadow.ensure()?;
        Ok(shadow)
    }

    pub fn ensure(&self) -> Result<()> {
        std::fs::create_dir_all(self.cursor_dir())
            .with_context(|| format!("creating cursor shadow at {}", self.root.display()))?;
        #[cfg(target_os = "macos")]
        self.ensure_macos_keychain()?;
        Ok(())
    }

    /// Pre-creates an empty-password `login.keychain-db` under the shadow's
    /// `Library/Keychains/` and runs `set-keychain-settings` (no flags) so
    /// cursor-agent's `security add-generic-password` doesn't pop a GUI
    /// prompt on sleep/idle. Reapplied on every call so pre-fix shadows
    /// get repaired without forcing the user to re-add their key.
    #[cfg(target_os = "macos")]
    fn ensure_macos_keychain(&self) -> Result<()> {
        let keychain_dir = self.root.join("Library").join("Keychains");
        std::fs::create_dir_all(&keychain_dir)
            .with_context(|| format!("creating shadow keychain dir {}", keychain_dir.display()))?;
        let keychain = keychain_dir.join("login.keychain-db");
        let keychain_str = keychain
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 cursor shadow keychain path"))?;
        if !keychain.exists() {
            let output = std::process::Command::new("/usr/bin/security")
                .env("HOME", &self.root)
                .args(["create-keychain", "-p", "", keychain_str])
                .output()
                .context("invoking `/usr/bin/security create-keychain` for cursor shadow")?;
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!(
                    "failed to pre-create cursor shadow keychain ({}): {}",
                    output.status,
                    stderr.trim()
                );
            }
        }
        // No flags → no lock-on-sleep, no idle timeout. Best-effort:
        // a failure here just means the user might see the password
        // prompt later, which is the pre-fix behavior — no reason to
        // hard-fail the spawn.
        let _ = std::process::Command::new("/usr/bin/security")
            .env("HOME", &self.root)
            .args(["set-keychain-settings", keychain_str])
            .output();
        Ok(())
    }

    pub fn delete(&self) -> Result<()> {
        if self.root.exists() {
            std::fs::remove_dir_all(&self.root)
                .with_context(|| format!("removing cursor shadow at {}", self.root.display()))?;
        }
        Ok(())
    }

    /// `<.cursor|cursor|Cursor>` inside the shadow — where cursor-agent
    /// stores `auth.json`, `cli-config.json`, and project state.
    pub fn cursor_dir(&self) -> PathBuf {
        self.root.join(cursor_subdir_name())
    }

    pub fn auth_file_path(&self) -> PathBuf {
        self.cursor_dir().join("auth.json")
    }

    /// True when `auth.json` exists and is non-empty. Authoritative
    /// authentication state lives in `cursor-agent status`; this is just a
    /// cheap "has the user ever logged in here?" gate for UI hints.
    pub fn has_credential_file(&self) -> bool {
        std::fs::metadata(self.auth_file_path())
            .map(|m| m.len() > 0)
            .unwrap_or(false)
    }

    /// Env vars to inject when spawning cursor-agent for this account.
    /// Returned as a `Vec` so callers can `cmd.env(name, value)` over it
    /// regardless of whether they hold a `Command` or a `HashMap`.
    pub fn env_block(&self) -> Vec<(&'static str, OsString)> {
        let mut out = Vec::with_capacity(3);
        let root_os = OsString::from(&self.root);
        let cursor_dir_os = OsString::from(self.cursor_dir());

        #[cfg(target_os = "macos")]
        out.push(("HOME", root_os));
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        out.push(("XDG_CONFIG_HOME", root_os));
        #[cfg(target_os = "windows")]
        out.push(("APPDATA", root_os));
        #[cfg(not(any(
            target_os = "macos",
            target_os = "linux",
            target_os = "freebsd",
            target_os = "windows"
        )))]
        let _ = root_os;

        out.push(("CURSOR_CONFIG_DIR", cursor_dir_os.clone()));
        out.push(("CURSOR_DATA_DIR", cursor_dir_os));
        out
    }
}

fn ensure_valid_account_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 64 {
        anyhow::bail!("invalid cursor account id length");
    }
    if !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        anyhow::bail!("invalid cursor account id (alnum and '-' only)");
    }
    Ok(())
}

/// 12-char base36 id. ~62 bits of entropy — collision-free for the
/// realistic ceiling of cursor accounts a user will manage on one machine.
fn generate_account_id() -> String {
    let mut bytes = [0u8; ACCOUNT_ID_LEN];
    rand::thread_rng().fill_bytes(&mut bytes);
    bytes
        .iter()
        .map(|b| {
            let n = b % 36;
            if n < 10 {
                (b'0' + n) as char
            } else {
                (b'a' + (n - 10)) as char
            }
        })
        .collect()
}

fn accounts_dir() -> Result<PathBuf> {
    let home = crate::services::system_env::home_dir()
        .context("cannot resolve $HOME for cursor shadow dir")?;
    Ok(home.join(".config").join("aivo").join(ACCOUNTS_DIR_NAME))
}

fn cursor_subdir_name() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        ".cursor"
    }
    #[cfg(target_os = "windows")]
    {
        "Cursor"
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "cursor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_ids_are_alphanumeric_and_unique() {
        let a = generate_account_id();
        let b = generate_account_id();
        assert_eq!(a.len(), ACCOUNT_ID_LEN);
        assert_ne!(a, b);
        assert!(a.bytes().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn account_id_validator_rejects_traversal() {
        assert!(ensure_valid_account_id("good-id-123").is_ok());
        assert!(ensure_valid_account_id("").is_err());
        assert!(ensure_valid_account_id("../etc").is_err());
        assert!(ensure_valid_account_id("with space").is_err());
        assert!(ensure_valid_account_id(&"a".repeat(65)).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_ensure_disables_lock_on_sleep_and_idle_timeout() {
        // Regression: cursor-agent stores OAuth tokens in macOS Keychain.
        // The default `lock-on-sleep timeout=300s` triggered a GUI password
        // prompt on the next read after system sleep / 5 min idle. ensure()
        // must reapply `set-keychain-settings` (no flags = no-timeout) on
        // every call so existing shadows from before this fix get repaired.
        let dir = tempfile::tempdir().unwrap();
        let shadow = CursorShadow {
            account_id: "test-keychain-fixture".to_string(),
            root: dir.path().to_path_buf(),
        };
        shadow.ensure().expect("first ensure should succeed");
        let keychain = shadow
            .root
            .join("Library")
            .join("Keychains")
            .join("login.keychain-db");
        assert!(keychain.exists(), "keychain should be created");

        // Simulate the broken pre-fix state, then call ensure() again and
        // confirm the settings are reapplied. Skip the assertion if the
        // sandbox blocks /usr/bin/security so CI doesn't false-fail.
        let set_bad = std::process::Command::new("/usr/bin/security")
            .env("HOME", &shadow.root)
            .args([
                "set-keychain-settings",
                "-t",
                "60",
                keychain.to_str().unwrap(),
            ])
            .output();
        let Ok(out) = set_bad else { return };
        if !out.status.success() {
            return;
        }
        shadow.ensure().expect("repair ensure should succeed");
        let info = std::process::Command::new("/usr/bin/security")
            .env("HOME", &shadow.root)
            .args(["show-keychain-info", keychain.to_str().unwrap()])
            .output()
            .expect("show-keychain-info should run");
        let stdout = String::from_utf8_lossy(&info.stdout);
        let stderr = String::from_utf8_lossy(&info.stderr);
        let combined = format!("{stdout}{stderr}");
        assert!(
            combined.contains("no-timeout"),
            "ensure() must clear timeout/lock-on-sleep: {combined:?}"
        );
    }

    #[test]
    fn env_block_pins_cursor_dirs() {
        let shadow = CursorShadow::for_account_id("abc123").unwrap();
        let env = shadow.env_block();
        let names: Vec<&str> = env.iter().map(|(k, _)| *k).collect();
        assert!(names.contains(&"CURSOR_CONFIG_DIR"));
        assert!(names.contains(&"CURSOR_DATA_DIR"));
        #[cfg(target_os = "macos")]
        assert!(names.contains(&"HOME"));
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        assert!(names.contains(&"XDG_CONFIG_HOME"));
    }
}
