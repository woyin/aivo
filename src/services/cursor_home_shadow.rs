//! Per-aivo-key isolated home dir for cursor-agent, so multiple logins
//! coexist without touching `~/.cursor`. The isolation env applies to the
//! cursor-agent process only; shell hooks written into the shadow restore
//! the real user environment for everything it spawns through a shell.

use anyhow::{Context, Result};
use rand::RngCore;
use std::ffi::OsString;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

const ACCOUNT_ID_LEN: usize = 12;
const ACCOUNTS_DIR_NAME: &str = "cursor-accounts";
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
const BASH_ENV_FILE_NAME: &str = "aivo-bash-env.sh";

/// `0` turns the shell-bootstrap hooks into no-ops: child shells keep the
/// shadow env instead of getting the real user env back.
pub const SHELL_BOOTSTRAP_ENV: &str = "AIVO_CURSOR_SHELL_BOOTSTRAP";

static SHELLS_ISOLATED: AtomicBool = AtomicBool::new(false);

/// Serve mode: shells here are remote-driven and must never see the
/// operator's real env.
pub fn isolate_shells_for_process() {
    SHELLS_ISOLATED.store(true, Ordering::Relaxed);
}

fn shells_isolated() -> bool {
    SHELLS_ISOLATED.load(Ordering::Relaxed)
        || std::env::var(SHELL_BOOTSTRAP_ENV).is_ok_and(|v| v.trim() == "0")
}

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
        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
        self.write_shell_bootstrap(&ShellBootstrap::capture())?;
        #[cfg(target_os = "macos")]
        self.ensure_macos_keychain()?;
        Ok(())
    }

    /// Hooks that undo the shadow for cursor-agent's child shells (zsh via the
    /// injected `ZDOTDIR`, bash via `BASH_ENV` / the profile files): restore
    /// the real environment, then chain to the user's own startup file.
    /// Direct non-shell children (e.g. MCP servers) still see the shadow.
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    fn write_shell_bootstrap(&self, boot: &ShellBootstrap) -> Result<()> {
        for (name, content) in boot.files() {
            write_atomic(&self.root.join(name), &content)?;
        }
        Ok(())
    }

    /// Pre-creates an empty-password `login.keychain-db` under the shadow's
    /// `Library/Keychains/`, **unlocks** it, and runs `set-keychain-settings`
    /// (no flags) so cursor-agent's `security add-generic-password` probe never
    /// pops a GUI password prompt. The unlock is the load-bearing part: a
    /// keychain that has locked (reboot / login-session restart) does NOT
    /// auto-unlock on access even with an empty password — `add`/`find-generic-
    /// password` show the dialog instead. `unlock-keychain -p ""` is silent
    /// (empty password) and the unlocked state persists for the whole securityd
    /// session, so every later cursor-agent spawn is prompt-free. Reapplied on
    /// every call so pre-fix shadows get repaired without re-adding the key.
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
            // A concurrent ensure() may have won the race; only absence is fatal.
            if !output.status.success() && !keychain.exists() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                anyhow::bail!(
                    "failed to pre-create cursor shadow keychain ({}): {}",
                    output.status,
                    stderr.trim()
                );
            }
        }
        // Unlock first: an already-locked keychain (post-reboot) makes
        // cursor-agent's probe pop the GUI prompt; the empty-password unlock is
        // silent and sticks for the session. Then clear lock-on-sleep / idle
        // timeout so it can't re-lock under us. Both best-effort — a failure
        // just restores the pre-fix prompt, no reason to hard-fail the spawn.
        let security = |args: &[&str]| {
            let _ = std::process::Command::new("/usr/bin/security")
                .env("HOME", &self.root)
                .args(args)
                .output();
        };
        security(&["unlock-keychain", "-p", "", keychain_str]);
        security(&["set-keychain-settings", keychain_str]);
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

    /// Env vars to inject when spawning cursor-agent for this account.
    /// Returned as a `Vec` so callers can `cmd.env(name, value)` over it
    /// regardless of whether they hold a `Command` or a `HashMap`.
    pub fn env_block(&self) -> Vec<(&'static str, OsString)> {
        self.env_block_with(shells_isolated())
    }

    /// `isolated` trips the hooks' gate so child shells stay in the shadow
    /// and source nothing of the user's; git config isn't handed in either.
    fn env_block_with(&self, isolated: bool) -> Vec<(&'static str, OsString)> {
        let mut out = Vec::with_capacity(8);
        let root_os = OsString::from(&self.root);
        let cursor_dir_os = OsString::from(self.cursor_dir());

        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
        {
            if isolated {
                out.push((SHELL_BOOTSTRAP_ENV, OsString::from("0")));
            } else if std::env::var_os("GIT_CONFIG_GLOBAL").is_none()
                && let Some(config) = discover_global_git_config_from_env()
            {
                // Git identity for git spawned directly (no shell hook between).
                out.push(("GIT_CONFIG_GLOBAL", config.into_os_string()));
            }
            // Route zsh and non-interactive bash through the (gated) hooks.
            out.push(("ZDOTDIR", root_os.clone()));
            out.push((
                "BASH_ENV",
                self.root.join(BASH_ENV_FILE_NAME).into_os_string(),
            ));
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "freebsd")))]
        let _ = isolated;
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

/// Pre-shadow env snapshot baked into the bootstrap hooks.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
struct ShellBootstrap {
    /// `export`/`unset` lines undoing the shadow override.
    restore_lines: String,
    orig_zdotdir: Option<String>,
    orig_bash_env: Option<String>,
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
impl ShellBootstrap {
    fn capture() -> Self {
        #[cfg(target_os = "macos")]
        let restore_lines = match real_home_dir().and_then(|h| h.to_str().map(str::to_owned)) {
            Some(home) => format!("export HOME={}\n", sh_quote(&home)),
            None => String::new(),
        };
        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
        let restore_lines = match clean_env_var("XDG_CONFIG_HOME") {
            Some(v) => format!("export XDG_CONFIG_HOME={}\n", sh_quote(&v)),
            None => "unset XDG_CONFIG_HOME\n".to_string(),
        };
        Self {
            restore_lines,
            orig_zdotdir: clean_env_var("ZDOTDIR"),
            orig_bash_env: clean_env_var("BASH_ENV"),
        }
    }

    fn files(&self) -> Vec<(&'static str, String)> {
        // Gate at runtime, not at write time: hooks are shared per account
        // across concurrent processes.
        let header = format!(
            "# Generated by aivo — restores the real user env in shells cursor-agent\n\
             # spawns under its shadow home; rewritten on every launch, do not edit.\n\
             if [ \"${{{SHELL_BOOTSTRAP_ENV}:-}}\" = 0 ]; then return 0; fi\n"
        );
        let restore = &self.restore_lines;
        let zdotdir = match &self.orig_zdotdir {
            Some(v) => format!("export ZDOTDIR={}\n", sh_quote(v)),
            None => "unset ZDOTDIR\n".to_string(),
        };
        let bash_env = match &self.orig_bash_env {
            Some(v) => {
                let q = sh_quote(v);
                format!("export BASH_ENV={q}\nif [ -f {q} ]; then . {q}; fi\n")
            }
            None => "unset BASH_ENV\n".to_string(),
        };
        vec![
            (
                ".zshenv",
                format!(
                    "{header}{restore}{zdotdir}if [ -f \"${{ZDOTDIR:-$HOME}}/.zshenv\" ]; then . \"${{ZDOTDIR:-$HOME}}/.zshenv\"; fi\n"
                ),
            ),
            (
                ".bashrc",
                format!(
                    "{header}{restore}if [ -f \"$HOME/.bashrc\" ]; then . \"$HOME/.bashrc\"; fi\n"
                ),
            ),
            (
                ".bash_profile",
                format!(
                    "{header}{restore}if [ -f \"$HOME/.bash_profile\" ]; then . \"$HOME/.bash_profile\"\n\
                     elif [ -f \"$HOME/.bash_login\" ]; then . \"$HOME/.bash_login\"\n\
                     elif [ -f \"$HOME/.profile\" ]; then . \"$HOME/.profile\"\nfi\n"
                ),
            ),
            (BASH_ENV_FILE_NAME, format!("{header}{restore}{bash_env}")),
        ]
    }
}

/// `$HOME`, unless it points inside a shadow (nested aivo) — then getpwuid.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn real_home_dir() -> Option<PathBuf> {
    crate::services::system_env::home_dir()
        .filter(|h| !is_shadow_path(h))
        .or_else(crate::services::system_env::passwd_home_dir)
}

/// Env var as UTF-8; values pointing inside a shadow count as unset.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn clean_env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|v| !v.is_empty() && !is_shadow_path(Path::new(v)))
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn is_shadow_path(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str() == ACCOUNTS_DIR_NAME)
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Write-then-rename so a shell never sources a half-written hook. Per-writer
/// temp name: `keys add cursor` detaches a `models --refresh` that installs the
/// same hooks concurrently, and a shared name left the loser renaming a path
/// the winner had already taken.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn write_atomic(path: &Path, content: &str) -> Result<()> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(format!(".{}.{seq}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    std::fs::write(&tmp, content)
        .with_context(|| format!("writing shadow shell hook {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("installing shadow shell hook {}", path.display()))?;
    Ok(())
}

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn discover_global_git_config_from_env() -> Option<PathBuf> {
    let home = real_home_dir()?;
    let xdg = clean_env_var("XDG_CONFIG_HOME").map(PathBuf::from);
    discover_global_git_config(&home, xdg.as_deref())
}

/// git's global-scope lookup: `~/.gitconfig`, then `$XDG_CONFIG_HOME/git/config`.
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
fn discover_global_git_config(home: &Path, xdg_config: Option<&Path>) -> Option<PathBuf> {
    let dotfile = home.join(".gitconfig");
    if dotfile.is_file() {
        return Some(dotfile);
    }
    let base = xdg_config
        .map(Path::to_path_buf)
        .unwrap_or_else(|| home.join(".config"));
    let config = base.join("git").join("config");
    config.is_file().then_some(config)
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
    Ok(crate::services::paths::config_dir().join(ACCOUNTS_DIR_NAME))
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

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_ensure_unlocks_a_locked_keychain() {
        // Regression: a locked keychain does NOT auto-unlock on write even with
        // an empty password — cursor-agent's probe would pop the GUI prompt.
        // ensure() must leave it unlocked so a write succeeds silently.
        let dir = tempfile::tempdir().unwrap();
        let shadow = CursorShadow {
            account_id: "test-unlock-fixture".to_string(),
            root: dir.path().to_path_buf(),
        };
        shadow.ensure().expect("first ensure should succeed");
        let keychain = shadow
            .root
            .join("Library")
            .join("Keychains")
            .join("login.keychain-db");
        let kc = keychain.to_str().unwrap();

        // Lock it, simulating a post-reboot state. Skip if the sandbox blocks
        // /usr/bin/security so CI doesn't false-fail.
        let lock = std::process::Command::new("/usr/bin/security")
            .env("HOME", &shadow.root)
            .args(["lock-keychain", kc])
            .output();
        let Ok(out) = lock else { return };
        if !out.status.success() {
            return;
        }

        shadow.ensure().expect("repair ensure should succeed");

        // A write now targets the shadow keychain; it succeeds only if ensure()
        // left it unlocked (otherwise: errSecUserCanceled in headless CI).
        let add = std::process::Command::new("/usr/bin/security")
            .env("HOME", &shadow.root)
            .args([
                "add-generic-password",
                "-a",
                "probe",
                "-s",
                "probe",
                "-w",
                "x",
                "-U",
                kc,
            ])
            .output()
            .expect("add-generic-password should run");
        assert!(
            add.status.success(),
            "ensure() must leave the keychain unlocked: {}",
            String::from_utf8_lossy(&add.stderr)
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
        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
        {
            assert!(names.contains(&"ZDOTDIR"));
            let bash_env = env
                .iter()
                .find(|(k, _)| *k == "BASH_ENV")
                .expect("BASH_ENV injected");
            assert_eq!(
                PathBuf::from(&bash_env.1),
                shadow.root.join(BASH_ENV_FILE_NAME)
            );
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn global_git_config_prefers_dotfile_and_falls_back_to_xdg() {
        let home = tempfile::tempdir().unwrap();
        let xdg = home.path().join(".config/git/config");
        std::fs::create_dir_all(xdg.parent().unwrap()).unwrap();
        std::fs::write(&xdg, "[user]\nname = xdg\n").unwrap();
        assert_eq!(
            discover_global_git_config(home.path(), None),
            Some(xdg.clone())
        );

        let custom = tempfile::tempdir().unwrap();
        let custom_cfg = custom.path().join("git/config");
        std::fs::create_dir_all(custom_cfg.parent().unwrap()).unwrap();
        std::fs::write(&custom_cfg, "[user]\nname = custom\n").unwrap();
        assert_eq!(
            discover_global_git_config(home.path(), Some(custom.path())),
            Some(custom_cfg)
        );

        let dotfile = home.path().join(".gitconfig");
        std::fs::write(&dotfile, "[user]\nname = dotfile\n").unwrap();
        assert_eq!(discover_global_git_config(home.path(), None), Some(dotfile));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn shadow_paths_and_quoting_are_detected() {
        assert!(is_shadow_path(Path::new(
            "/Users/u/.config/aivo/cursor-accounts/abc123"
        )));
        assert!(is_shadow_path(Path::new(
            "/x/cursor-accounts/abc123/aivo-bash-env.sh"
        )));
        assert!(!is_shadow_path(Path::new("/Users/u/.config/zsh")));
        assert_eq!(sh_quote("/plain/path"), "'/plain/path'");
        assert_eq!(sh_quote("it's"), r"'it'\''s'");
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn bootstrap_files_restore_and_chain_the_real_environment() {
        use std::collections::HashMap;
        let boot = ShellBootstrap {
            restore_lines: "export HOME='/real/home'\n".to_string(),
            orig_zdotdir: None,
            orig_bash_env: None,
        };
        let files: HashMap<_, _> = boot.files().into_iter().collect();
        let zshenv = &files[".zshenv"];
        assert!(zshenv.contains("export HOME='/real/home'"), "{zshenv}");
        assert!(zshenv.contains("unset ZDOTDIR"), "{zshenv}");
        assert!(
            zshenv.contains(r#". "${ZDOTDIR:-$HOME}/.zshenv""#),
            "{zshenv}"
        );
        assert!(files[BASH_ENV_FILE_NAME].contains("unset BASH_ENV"));
        assert!(files[".bashrc"].contains(r#". "$HOME/.bashrc""#));
        assert!(files[".bash_profile"].contains(r#". "$HOME/.profile""#));

        let boot = ShellBootstrap {
            restore_lines: String::new(),
            orig_zdotdir: Some("/dot/zsh".to_string()),
            orig_bash_env: Some("/user/env.sh".to_string()),
        };
        let files: HashMap<_, _> = boot.files().into_iter().collect();
        assert!(files[".zshenv"].contains("export ZDOTDIR='/dot/zsh'"));
        let bash_env = &files[BASH_ENV_FILE_NAME];
        assert!(bash_env.contains("export BASH_ENV='/user/env.sh'"));
        assert!(bash_env.contains(". '/user/env.sh'"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn bootstrap_files_gate_on_the_isolation_env_var() {
        let boot = ShellBootstrap {
            restore_lines: "export HOME='/real/home'\n".to_string(),
            orig_zdotdir: None,
            orig_bash_env: None,
        };
        let gate = format!("if [ \"${{{SHELL_BOOTSTRAP_ENV}:-}}\" = 0 ]; then return 0; fi");
        for (name, content) in boot.files() {
            let gate_at = content
                .find(&gate)
                .unwrap_or_else(|| panic!("{name} lacks gate"));
            if let Some(restore_at) = content.find("export HOME=") {
                assert!(gate_at < restore_at, "{name}: gate must precede restore");
            }
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn isolated_env_block_trips_gate_and_skips_git_config() {
        let shadow = CursorShadow::for_account_id("abc123").unwrap();
        let env = shadow.env_block_with(true);
        let gate = env
            .iter()
            .find(|(k, _)| *k == SHELL_BOOTSTRAP_ENV)
            .expect("gate var injected");
        assert_eq!(gate.1, OsString::from("0"));
        let names: Vec<&str> = env.iter().map(|(k, _)| *k).collect();
        assert!(!names.contains(&"GIT_CONFIG_GLOBAL"));
        // Still routed through the (gated) hooks so nothing of the user's runs.
        assert!(names.contains(&"ZDOTDIR"));
        assert!(names.contains(&"BASH_ENV"));
    }

    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "freebsd"))]
    #[test]
    fn ensure_writes_shell_bootstrap_hooks() {
        let dir = tempfile::tempdir().unwrap();
        let shadow = CursorShadow {
            account_id: "test-bootstrap".to_string(),
            root: dir.path().to_path_buf(),
        };
        shadow.ensure().expect("ensure should succeed");
        for name in [".zshenv", ".bashrc", ".bash_profile", BASH_ENV_FILE_NAME] {
            assert!(shadow.root.join(name).is_file(), "{name} missing");
        }
        let zshenv = std::fs::read_to_string(shadow.root.join(".zshenv")).unwrap();
        #[cfg(target_os = "macos")]
        assert!(zshenv.contains("export HOME="), "{zshenv}");
        #[cfg(not(target_os = "macos"))]
        assert!(zshenv.contains("XDG_CONFIG_HOME"), "{zshenv}");
    }

    /// A shell spawned with the shadow env must see the real environment again.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn shells_spawned_under_shadow_env_restore_real_environment() {
        let dir = tempfile::tempdir().unwrap();
        let shadow = CursorShadow {
            account_id: "test-shell-restore".to_string(),
            root: dir.path().to_path_buf(),
        };
        shadow.ensure().expect("ensure should succeed");

        #[cfg(target_os = "macos")]
        let (probe, expected) = ("printf %s \"$HOME\"", std::env::var("HOME").unwrap());
        #[cfg(target_os = "linux")]
        let (probe, expected) = (
            "printf %s \"$XDG_CONFIG_HOME\"",
            std::env::var("XDG_CONFIG_HOME").unwrap_or_default(),
        );

        let mut shells = vec![PathBuf::from("bash")];
        // zsh is guaranteed on macOS; probe it on Linux only if installed.
        #[cfg(target_os = "macos")]
        shells.push(PathBuf::from("zsh"));
        #[cfg(target_os = "linux")]
        if which_zsh().is_some() {
            shells.push(PathBuf::from("zsh"));
        }
        for shell in shells {
            let mut cmd = std::process::Command::new(&shell);
            for (name, value) in shadow.env_block_with(false) {
                cmd.env(name, value);
            }
            let out = cmd
                .args(["-c", probe])
                .output()
                .expect("shell should spawn");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert_eq!(
                stdout,
                expected,
                "{} must restore the real env (stderr: {})",
                shell.display(),
                String::from_utf8_lossy(&out.stderr)
            );
            assert_ne!(PathBuf::from(stdout.as_ref()), shadow.root);
        }
    }

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn shells_spawned_under_isolated_env_stay_in_the_shadow() {
        let dir = tempfile::tempdir().unwrap();
        let shadow = CursorShadow {
            account_id: "test-shell-isolated".to_string(),
            root: dir.path().to_path_buf(),
        };
        shadow.ensure().expect("ensure should succeed");

        #[cfg(target_os = "macos")]
        let probe = "printf %s \"$HOME\"";
        #[cfg(target_os = "linux")]
        let probe = "printf %s \"$XDG_CONFIG_HOME\"";

        let mut shells = vec![PathBuf::from("bash")];
        #[cfg(target_os = "macos")]
        shells.push(PathBuf::from("zsh"));
        #[cfg(target_os = "linux")]
        if which_zsh().is_some() {
            shells.push(PathBuf::from("zsh"));
        }
        for shell in shells {
            let mut cmd = std::process::Command::new(&shell);
            for (name, value) in shadow.env_block_with(true) {
                cmd.env(name, value);
            }
            let out = cmd
                .args(["-c", probe])
                .output()
                .expect("shell should spawn");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert_eq!(
                PathBuf::from(stdout.as_ref()),
                shadow.root,
                "{} must keep the shadow env (stderr: {})",
                shell.display(),
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    #[cfg(target_os = "linux")]
    fn which_zsh() -> Option<std::path::PathBuf> {
        let out = std::process::Command::new("sh")
            .args(["-c", "command -v zsh"])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string()))
    }
}
