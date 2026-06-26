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
        match sync_account_status().await {
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
            AccountSync::Unlinked { had_local: true } => {
                println!(
                    "  {}",
                    style::dim("This device's previous link was removed; signing in again.")
                );
                // fall through to a fresh login
            }
            AccountSync::Unlinked { had_local: false } | AccountSync::Unverified(None) => {
                // Nothing linked locally or server-side → just log in.
            }
        }

        let label = args.label.unwrap_or_else(default_label);

        // Start the device-authorization request (Ed25519-signed).
        let device = device_auth::start_device_auth(Some(&label)).await?;

        // Show the code + URL. Polling starts immediately, so approval is
        // detected even if the user never presses Enter (e.g. opens the URL on
        // their phone); Enter just opens a browser here — optional, not a gate.
        let verify_url = device
            .verification_uri_complete
            .filter(|s| !s.is_empty())
            .unwrap_or(device.verification_uri);
        println!();
        println!(
            "  {} To link this device, sign in and confirm this code:",
            style::arrow_symbol()
        );
        println!("    {}", style::bold(&device.user_code));
        println!("    {} {}", style::dim("at"), style::blue(&verify_url));
        println!();
        println!(
            "  {}",
            style::dim("Press Enter to open your browser (optional)…")
        );

        // Open the browser if/when the user presses Enter, concurrently with —
        // never blocking — the poll below. Aborted once polling finishes.
        let opener = spawn_browser_opener_on_enter(verify_url.clone());

        // Poll until the user approves in the browser.
        let (spinning, spinner_handle) =
            style::start_spinner(Some(" Waiting for approval in the browser..."));
        let result =
            device_auth::poll_device_token(&device.device_code, device.interval, device.expires_in)
                .await;
        style::stop_spinner(&spinning);
        let _ = spinner_handle.await;
        opener.abort();
        let user = result?;

        // Record the account locally for display (no secret stored).
        let account = account_store::Account {
            user_id: user.id,
            email: user.email,
            name: user.name,
            linked_at: chrono::Utc::now().to_rfc3339(),
        };
        account_store::save(&account).await?;

        // Ensure a device-signed key exists so the binding is usable now.
        if let Some((starter, is_new_user)) = self.session_store.ensure_starter_key().await
            && is_new_user
        {
            let _ = self.session_store.set_active_key(&starter.id).await;
        }

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
            style::dim("Sign in to your aivo account and link this device.")
        );
        println!(
            "{}",
            style::dim("Shows a short code + URL and waits for you to approve it in the")
        );
        println!(
            "{}",
            style::dim("browser (press Enter to open it — optional), then binds this")
        );
        println!("{}", style::dim("device to your account (plan + credits)."));
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
            };
            if local.as_ref() != Some(&account) {
                let _ = account_store::save(&account).await;
            }
            AccountSync::Linked(account)
        }
        device_auth::DeviceStatus::Unlinked => {
            let had_local = local.is_some();
            if had_local {
                account_store::clear();
            }
            AccountSync::Unlinked { had_local }
        }
        device_auth::DeviceStatus::Unknown => AccountSync::Unverified(local),
    }
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
            style::dim("Unlink this device from your aivo account.")
        );
        println!(
            "{}",
            style::dim("Removes the server-side device link and clears the local record;")
        );
        println!(
            "{}",
            style::dim("you'll need `aivo login` again to use your plan on this device.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<10}", "-y, --yes")),
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
