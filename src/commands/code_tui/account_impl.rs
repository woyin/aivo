//! `/login` `/logout` `/usage` — the aivo account flows, gated to the bundled
//! aivo provider (hidden + refused on BYOK keys). `/login` parks a status card
//! above the composer while the device flow polls, `/logout` confirms on a y/n
//! card, `/usage` runs the CLI through the `!` machinery. Each flow runs on
//! `account_task` and reports via `Account*` events stamped with `account_gen`,
//! so a cancelled or superseded flow's late result is dropped.

use super::*;

use crate::commands::login::{AccountSync, sync_account_status};
use crate::services::account_store;
use crate::services::device_auth;

const ACCOUNT_ONLY_HINT: &str = "Account commands only work on the aivo provider — /key to switch.";

impl CodeTuiApp {
    pub(super) fn is_aivo_account_key(&self) -> bool {
        crate::services::provider_profile::is_aivo_starter_base(&self.key.base_url)
    }

    /// Menu/help visibility for the active key: account commands are aivo-only.
    pub(super) fn slash_command_visible(&self, name: &str) -> bool {
        match name {
            "login" | "logout" | "usage" => self.is_aivo_account_key(),
            _ => true,
        }
    }

    /// Bump the generation and abort the prior task; returns the fresh gen.
    fn begin_account_flow(&mut self) -> u64 {
        self.account_gen = self.account_gen.wrapping_add(1);
        if let Some(task) = self.account_task.take() {
            task.abort();
        }
        self.account_gen
    }

    pub(super) fn cancel_account_login(&mut self) {
        let _ = self.begin_account_flow();
        self.account_login = None;
        self.notice = Some((MUTED(), "Sign-in cancelled.".to_string()));
    }

    /// `/login`: starts behind a notice; the card appears once the code arrives.
    pub(super) async fn run_login_command(&mut self) {
        if !self.is_aivo_account_key() {
            self.notice = Some((MUTED(), ACCOUNT_ONLY_HINT.to_string()));
            return;
        }
        let account_gen = self.begin_account_flow();
        self.pending_logout = None;
        self.account_login = None;
        self.notice = Some((MUTED(), "Starting sign-in…".to_string()));
        let tx = self.tx.clone();
        let session_store = self.session_store.clone();
        let label = device_label();
        self.account_task = Some(tokio::spawn(async move {
            run_login_flow(tx, session_store, account_gen, label).await;
        }));
    }

    /// `/logout`: raise the y/n confirm card ("who" comes from the local cache).
    pub(super) async fn run_logout_command(&mut self) {
        if !self.is_aivo_account_key() {
            self.notice = Some((MUTED(), ACCOUNT_ONLY_HINT.to_string()));
            return;
        }
        let Some(account) = account_store::load() else {
            self.notice = Some((MUTED(), "Not logged in.".to_string()));
            return;
        };
        if self.account_login.is_some() {
            self.cancel_account_login();
            self.notice = None;
        }
        self.pending_logout = Some(account.display().to_string());
    }

    /// `/logout` confirm card: y/Enter unlinks, n/Esc dismisses, else card stays.
    pub(super) fn handle_logout_confirm_key(&mut self, key: KeyEvent) {
        let allow = matches!(key.code, KeyCode::Char('y' | 'Y') | KeyCode::Enter);
        let deny = matches!(key.code, KeyCode::Char('n' | 'N') | KeyCode::Esc);
        if !allow && !deny {
            return;
        }
        if self.pending_logout.take().is_none() {
            return;
        }
        if deny {
            self.show_toast("Sign-out cancelled");
            return;
        }
        self.notice = Some((MUTED(), "Signing out…".to_string()));
        let account_gen = self.begin_account_flow();
        let tx = self.tx.clone();
        self.account_task = Some(tokio::spawn(async move {
            let result = match device_auth::unlink_device().await {
                Ok(()) => {
                    account_store::clear();
                    // Entitlements revert to the anonymous starter tier.
                    crate::services::models_cache::ModelsCache::shared()
                        .clear_starter()
                        .await;
                    Ok(())
                }
                Err(e) => Err(format!("{e:#}")),
            };
            let _ = tx.send(RuntimeEvent::AccountLogoutDone {
                account_gen,
                result,
            });
        }));
    }

    /// Login-card keys (Enter opens browser, Esc cancels) — only on an empty
    /// composer with no overlay; Esc also yields to a turn/`!cmd` interrupt.
    pub(super) fn handle_login_card_key(&mut self, key: KeyEvent) -> bool {
        if !self.draft.is_empty() || !matches!(self.overlay, Overlay::None) {
            return false;
        }
        match key.code {
            KeyCode::Enter => {
                let Some(url) = self.account_login.as_ref().map(|c| c.open_url.clone()) else {
                    return false;
                };
                if crate::services::browser_open::open_url(&url).is_err() {
                    self.notice = Some((
                        MUTED(),
                        "Couldn't open a browser — visit the URL shown.".to_string(),
                    ));
                }
                true
            }
            KeyCode::Esc if !self.sending && self.local_command.is_none() => {
                self.cancel_account_login();
                true
            }
            _ => false,
        }
    }

    /// `/usage`: run the CLI itself as a `!cmd`, so the block IS the CLI output
    /// (the PTY reader strips its colors/spinner to plain text).
    pub(super) async fn run_usage_command(&mut self) {
        if !self.is_aivo_account_key() {
            self.notice = Some((MUTED(), ACCOUNT_ONLY_HINT.to_string()));
            return;
        }
        self.start_local_command("aivo account usage".to_string());
    }

    /// Device code arrived → raise the card; failed to start → error notice.
    pub(super) fn apply_account_login_prompt(
        &mut self,
        account_gen: u64,
        result: std::result::Result<(String, String), String>,
    ) {
        if account_gen != self.account_gen {
            return;
        }
        match result {
            Ok((user_code, open_url)) => {
                self.notice = None;
                self.account_login = Some(AccountLoginCard {
                    user_code,
                    open_url,
                });
            }
            Err(msg) => {
                self.account_login = None;
                self.notice = Some((ERROR(), msg));
            }
        }
    }

    /// Login resolved: drop the card and show the outcome.
    pub(super) async fn apply_account_login_done(
        &mut self,
        account_gen: u64,
        result: std::result::Result<String, String>,
    ) {
        if account_gen != self.account_gen {
            return;
        }
        self.account_task = None;
        self.account_login = None;
        match result {
            Ok(msg) => {
                // The TUI reads its own `ModelsCache`, not the shared instance
                // the flow cleared — drop the old plan's catalog here too.
                self.cache.clear_starter().await;
                self.refresh_context_window().await;
                self.notice = Some((ACCENT(), msg));
            }
            Err(msg) => self.notice = Some((ERROR(), msg)),
        }
    }

    /// Unlink resolved: show the outcome.
    pub(super) async fn apply_account_logout_done(
        &mut self,
        account_gen: u64,
        result: std::result::Result<(), String>,
    ) {
        if account_gen != self.account_gen {
            return;
        }
        self.account_task = None;
        match result {
            Ok(()) => {
                // See `apply_account_login_done` on why this instance too.
                self.cache.clear_starter().await;
                self.refresh_context_window().await;
                self.notice = Some((
                    MUTED(),
                    "Logged out — this device is no longer linked.".to_string(),
                ));
            }
            Err(msg) => self.notice = Some((ERROR(), format!("Couldn't sign out: {msg}"))),
        }
    }
}

/// Label for this device in the account's device list.
fn device_label() -> String {
    let host = std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    match host {
        Some(h) => format!("aivo code on {h}"),
        None => format!("aivo code {}", crate::version::VERSION),
    }
}

/// The device flow off the event loop: emits `AccountLoginPrompt` (code + URL)
/// then `AccountLoginDone`. Mirrors CLI `aivo login`.
async fn run_login_flow(
    tx: UnboundedSender<RuntimeEvent>,
    session_store: SessionStore,
    account_gen: u64,
    label: String,
) {
    // Re-verify first — the device may have been unlinked from the dashboard.
    if let AccountSync::Linked(a) | AccountSync::Unverified(Some(a)) = sync_account_status().await {
        let _ = tx.send(RuntimeEvent::AccountLoginDone {
            account_gen,
            result: Ok(format!("Already logged in as {}.", a.display())),
        });
        return;
    }

    let device = match device_auth::start_device_auth(Some(&label)).await {
        Ok(d) => d,
        Err(e) => {
            let _ = tx.send(RuntimeEvent::AccountLoginPrompt {
                account_gen,
                result: Err(format!("{e:#}")),
            });
            return;
        }
    };
    let open_url = device
        .verification_uri_complete
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| device.verification_uri.clone());
    let _ = tx.send(RuntimeEvent::AccountLoginPrompt {
        account_gen,
        result: Ok((device.user_code.clone(), open_url)),
    });

    let user = match device_auth::poll_device_token(
        &device.device_code,
        device.interval,
        device.expires_in,
    )
    .await
    {
        Ok(u) => u,
        Err(e) => {
            let _ = tx.send(RuntimeEvent::AccountLoginDone {
                account_gen,
                result: Err(format!("{e:#}")),
            });
            return;
        }
    };

    let result = match crate::commands::login::finalize_login(user, &session_store).await {
        Ok(account) => Ok(format!("Logged in as {}", account.display())),
        Err(e) => Err(format!("Signed in, but couldn't save the account: {e:#}")),
    };
    let _ = tx.send(RuntimeEvent::AccountLoginDone {
        account_gen,
        result,
    });
}
