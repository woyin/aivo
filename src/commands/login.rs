//! `aivo login` / `aivo logout` — link this device to a getaivo.dev account.
//!
//! Login drives the web app's RFC 8628 device-authorization flow: it prints a
//! short code + URL and polls until the device is linked (the user signs in
//! with OAuth at that URL; Enter opens it in a browser but isn't required).
//! Approval binds this machine's Ed25519 device id to the user server-side, so
//! its already-signed requests resolve to the account's entitlements. No bearer
//! token is stored — the device key stays the credential; we only record the
//! account for display.

use anyhow::Result;

use crate::cli::{LoginArgs, LogoutArgs};
use crate::errors::ExitCode;
use crate::services::session_store::SessionStore;
use crate::services::{account_store, browser_open, device_auth};
use crate::style;
use crate::version::VERSION;

pub struct LoginCommand {
    session_store: SessionStore,
}

impl LoginCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: LoginArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(&self, args: LoginArgs) -> Result<ExitCode> {
        // Re-verify against the server before deciding we're already logged in:
        // the device may have been unlinked from the dashboard since last time.
        let relink = match sync_account_status().await {
            AccountSync::Linked(account) => {
                println!(
                    "  {} Already logged in as {}.",
                    style::success_symbol(),
                    style::bold(account.display())
                );
                return Ok(ExitCode::Success);
            }
            AccountSync::Unverified(Some(account)) => {
                // Couldn't reach the server; trust the local cache rather than
                // forcing a re-login on a flaky network.
                println!(
                    "  {} Already logged in as {} {}.",
                    style::success_symbol(),
                    style::bold(account.display()),
                    style::dim("(couldn't verify with the server)")
                );
                return Ok(ExitCode::Success);
            }
            // Previous link removed: note it in the header, then re-link.
            AccountSync::Unlinked { had_local: true } => true,
            AccountSync::Unlinked { had_local: false } | AccountSync::Unverified(None) => false,
        };

        // Headless (agent shell, CI, both ends piped): nobody can see the code or
        // approve it, so the poll below would just block to expiry. Refuse fast.
        {
            use std::io::IsTerminal as _;
            if is_headless(
                std::io::stdin().is_terminal(),
                std::io::stdout().is_terminal(),
            ) {
                eprintln!(
                    "{} `aivo account login` needs an interactive terminal to sign in.",
                    style::red("Error:")
                );
                eprintln!(
                    "  Run it directly in your terminal:  {}",
                    style::cyan("aivo account login")
                );
                return Ok(ExitCode::UserError);
            }
        }

        let label = args.label.unwrap_or_else(default_label);

        let device = device_auth::start_device_auth(Some(&label)).await?;

        // Show the code-prefilled URL so visiting or scanning it needs no
        // typing; it's also what Enter opens. Polling below starts regardless,
        // so Enter is a convenience (e.g. approve on a phone), never a gate.
        let bare_url = device.verification_uri.clone();
        let open_url = device
            .verification_uri_complete
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| bare_url.clone());
        let interactive = std::io::IsTerminal::is_terminal(&std::io::stdin());

        println!();
        println!("  {}", style::bold("Sign in to aivo"));
        if relink {
            println!(
                "  {}",
                style::dim("This device's previous link was removed — signing in again.")
            );
        }
        println!(
            "  Confirm this code in your browser:  {}",
            style::cyan(style::bold(&device.user_code))
        );
        if interactive {
            println!(
                "  Press {} to open your browser, or visit {}",
                style::keycap(" Enter "),
                style::blue(&open_url)
            );
        } else {
            println!("  Visit {} to confirm.", style::blue(&open_url));
        }
        println!();

        // Scoped so echo is restored — and the Enter→browser opener cancelled —
        // the moment polling ends, before any later output.
        let outcome = {
            // Without this, the newline echoed on Enter scrolls the in-place
            // spinner, duplicating the "Waiting…" line.
            let _echo_guard = EchoGuard::disable();
            // Opens the browser on Enter, concurrently with (never blocking) the poll.
            let opener = spawn_browser_opener_on_enter(open_url);
            let (spinning, spinner_handle) = style::start_spinner(Some(" Waiting for approval…"));
            // Catch Ctrl+C here rather than let the default SIGINT kill the
            // process mid-poll: that skips `EchoGuard`'s drop, leaving the
            // terminal with echo off and a half-drawn spinner line.
            let outcome = tokio::select! {
                r = device_auth::poll_device_token(
                    &device.device_code,
                    device.interval,
                    device.expires_in,
                ) => Some(r),
                _ = tokio::signal::ctrl_c() => None,
            };
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            opener.abort();
            outcome
        };
        let user = match outcome {
            Some(result) => result?,
            None => {
                println!("  {}", style::dim("Cancelled."));
                return Ok(ExitCode::ToolExit(130));
            }
        };

        let account = finalize_login(user, &self.session_store).await?;

        println!(
            "  {} Logged in as {}",
            style::success_symbol(),
            style::bold(account.display())
        );
        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo login [--label <LABEL>]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Sign in to your aivo account and link this device (shows a code + URL to approve in the browser)."
            )
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<20}", "--label <LABEL>")),
            style::dim("Name for this device in your account's device list")
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo login"));
        println!("  {}", style::dim("aivo login --label \"work laptop\""));
    }
}

/// Result of reconciling the local `account.json` cache against the server's
/// device→user binding.
pub(crate) enum AccountSync {
    /// Server confirms this device is linked; local cache refreshed/created.
    Linked(account_store::Account),
    /// Server says this device is not linked; any local record was cleared.
    /// `had_local` records whether a cached record existed before clearing.
    Unlinked { had_local: bool },
    /// Couldn't reach the server; local state (if any) left untouched.
    Unverified(Option<account_store::Account>),
}

impl AccountSync {
    /// The account to display after reconciliation: the linked account, the
    /// untouched cache when unverified, or none when unlinked.
    pub(crate) fn into_account(self) -> Option<account_store::Account> {
        match self {
            AccountSync::Linked(a) => Some(a),
            AccountSync::Unverified(local) => local,
            AccountSync::Unlinked { .. } => None,
        }
    }
}

/// Checks the server's binding for this device and reconciles the local
/// `account.json` cache: refresh email/name when linked, clear it when the
/// server reports the device is no longer linked. A network failure leaves the
/// cache untouched (returns `Unverified`).
pub(crate) async fn sync_account_status() -> AccountSync {
    let local = account_store::load();
    match device_auth::fetch_device_status().await {
        device_auth::DeviceStatus::Linked(user) => {
            // Preserve the original link time when we already had it.
            let linked_at = local
                .as_ref()
                .map(|a| a.linked_at.clone())
                .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());
            let account = account_store::Account {
                user_id: user.id,
                email: user.email,
                name: user.name,
                linked_at,
                // Preserve the cached plan + label — this status sync doesn't fetch them.
                plan: local.as_ref().and_then(|a| a.plan.clone()),
                plan_label: local.as_ref().and_then(|a| a.plan_label.clone()),
            };
            if local.as_ref() != Some(&account) {
                // Plan is preserved from `local` here, so only identity can differ.
                let identity_changed = local
                    .as_ref()
                    .map(|a| a.user_id != account.user_id)
                    .unwrap_or(true);
                let _ = account_store::save(&account).await;
                if identity_changed {
                    clear_starter_model_cache().await;
                }
            }
            AccountSync::Linked(account)
        }
        device_auth::DeviceStatus::Unlinked => {
            let had_local = local.is_some();
            if had_local {
                account_store::clear();
                // Entitlements revert to the anonymous starter tier.
                clear_starter_model_cache().await;
            }
            AccountSync::Unlinked { had_local }
        }
        device_auth::DeviceStatus::Unknown => AccountSync::Unverified(local),
    }
}

/// Post-approval bookkeeping shared by `aivo login` and the TUI `/login`:
/// cache the account + plan for display (no secret stored), drop the starter
/// catalog on a profile change, and ensure a device-signed key exists.
pub(crate) async fn finalize_login(
    user: device_auth::LinkedUser,
    session_store: &SessionStore,
) -> Result<account_store::Account> {
    let (plan, plan_label) = match device_auth::fetch_account_usage().await {
        device_auth::AccountUsage::Linked(s) => account_store::canonical_plan_with_label(
            s.plan.as_deref(),
            s.is_pro,
            s.plan_label.as_deref(),
        ),
        _ => (None, None),
    };
    let account = account_store::Account {
        user_id: user.id,
        email: user.email,
        name: user.name,
        linked_at: chrono::Utc::now().to_rfc3339(),
        plan,
        plan_label,
    };
    // Different account/plan → the cached starter catalog is the old profile's.
    let profile_changed = account_store::load()
        .map(|prev| prev.user_id != account.user_id || prev.plan != account.plan)
        .unwrap_or(true);
    account_store::save(&account).await?;
    if profile_changed {
        clear_starter_model_cache().await;
    }

    if let Some((starter, is_new_user)) = session_store.ensure_starter_key().await
        && is_new_user
    {
        let _ = session_store.set_active_key(&starter.id).await;
    }
    Ok(account)
}

/// Drops the cached `aivo/starter` catalog after a login-profile change.
async fn clear_starter_model_cache() {
    crate::services::models_cache::ModelsCache::shared()
        .clear_starter()
        .await;
}

/// Prints `prompt`, then on an interactive terminal reads one line and returns
/// it trimmed. Returns `None` on a non-TTY session — there's no one to ask.
async fn read_line_if_tty(prompt: &str) -> Option<String> {
    use std::io::{IsTerminal, Write};
    if !std::io::stdin().is_terminal() {
        return None;
    }
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let line = tokio::task::spawn_blocking(|| {
        let mut s = String::new();
        let _ = std::io::stdin().read_line(&mut s);
        s
    })
    .await
    .unwrap_or_default();
    Some(line.trim().to_string())
}

/// Spawns a detached task that opens `verify_url` when the user presses Enter
/// on a TTY (no-op on a non-TTY). The caller aborts it once polling ends, so a
/// never-pressed Enter never gates login — it's purely a convenience.
fn spawn_browser_opener_on_enter(verify_url: String) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        if read_line_if_tty("").await.is_some() && browser_open::open_url(&verify_url).is_err() {
            println!(
                "  {}",
                style::dim("(couldn't open a browser — visit the URL above)")
            );
        }
    })
}

/// Suppresses terminal echo for its lifetime (restored on drop), leaving
/// canonical mode and signals on so line reads and Ctrl+C still work. `disable`
/// returns `None` (a no-op) off a TTY or on non-Unix.
#[cfg(unix)]
struct EchoGuard {
    fd: std::os::fd::RawFd,
    original: libc::termios,
}

#[cfg(unix)]
impl EchoGuard {
    fn disable() -> Option<Self> {
        use std::os::fd::AsRawFd;
        let fd = std::io::stdin().as_raw_fd();
        let mut original = std::mem::MaybeUninit::uninit();
        if unsafe { libc::tcgetattr(fd, original.as_mut_ptr()) } != 0 {
            return None;
        }
        let original = unsafe { original.assume_init() };
        let mut quiet = original;
        quiet.c_lflag &= !libc::ECHO;
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &quiet) } != 0 {
            return None;
        }
        Some(Self { fd, original })
    }
}

#[cfg(unix)]
impl Drop for EchoGuard {
    fn drop(&mut self) {
        let _ = unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.original) };
    }
}

#[cfg(not(unix))]
#[allow(dead_code)] // Never constructed here; `disable` is always a no-op.
struct EchoGuard;

#[cfg(not(unix))]
impl EchoGuard {
    // The echo trail is Unix-only; nothing to suppress here.
    fn disable() -> Option<Self> {
        None
    }
}

/// Default device label: `aivo <version> on <hostname>` when a hostname is
/// discoverable, else `aivo <version> CLI`. Best-effort and cross-platform.
fn default_label() -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    match host {
        Some(h) => format!("aivo {VERSION} on {h}"),
        None => format!("aivo {VERSION} CLI"),
    }
}

/// No terminal to complete the device flow: neither stdin nor stdout a TTY.
fn is_headless(stdin_tty: bool, stdout_tty: bool) -> bool {
    !(stdin_tty || stdout_tty)
}

#[derive(Default)]
pub struct LogoutCommand;

impl LogoutCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: LogoutArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(&self, args: LogoutArgs) -> Result<ExitCode> {
        let Some(account) = account_store::load() else {
            println!("  {}", style::dim("Not logged in."));
            return Ok(ExitCode::Success);
        };

        // Logout now revokes this device's access, so confirm first (a local
        // clear alone is a no-op — the next status check would heal it).
        if !args.yes && !confirm_unlink(&account).await? {
            println!("  {}", style::dim("Cancelled."));
            return Ok(ExitCode::Success);
        }

        // Unlink server-side (device-authed). On failure we keep the local
        // record so local and server stay consistent.
        device_auth::unlink_device().await?;
        account_store::clear();
        clear_starter_model_cache().await;

        println!(
            "  {} Logged out — this device is no longer linked.",
            style::success_symbol()
        );
        Ok(ExitCode::Success)
    }

    pub fn print_help() {
        println!("{} aivo logout [-y]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Unlink this device from your aivo account (clears the server link and local record; `aivo login` to re-link)."
            )
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<12}", "-y, --yes")),
            style::dim("Skip the confirmation prompt")
        );
    }
}

/// Confirms before unlinking. On a non-TTY session there's no way to ask, so it
/// refuses (fail-closed) — scripts must pass `--yes`.
async fn confirm_unlink(account: &account_store::Account) -> Result<bool> {
    let prompt = format!(
        "  Unlink this device from {}? [y/N]: ",
        style::bold(account.display())
    );
    match read_line_if_tty(&prompt).await {
        None => anyhow::bail!(
            "Refusing to unlink without confirmation on a non-interactive session. Pass --yes to proceed."
        ),
        Some(answer) => Ok(answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headless_only_when_neither_stream_is_a_tty() {
        assert!(is_headless(false, false));
        assert!(!is_headless(true, false));
        assert!(!is_headless(false, true));
        assert!(!is_headless(true, true));
    }
}
