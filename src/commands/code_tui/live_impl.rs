//! `--share` / `/share` live-share lifecycle for the chat TUI. Starts off the
//! event loop (so the handshake never blocks rendering) and reports back via
//! [`RuntimeEvent::LiveShareReady`]; torn down on `/new`, resume, and exit.

use super::*;

impl CodeTuiApp {
    /// `/share [stop]`: bare/`start` opens a share (re-showing the URL if already
    /// live); `stop` ends it.
    pub(super) async fn run_share_command(&mut self, arg: Option<String>) {
        match arg.as_deref().map(str::trim) {
            Some("stop") | Some("off") | Some("end") => {
                if self.stop_live_share() {
                    self.notice = Some((MUTED(), "Sharing stopped.".to_string()));
                } else {
                    self.notice = Some((MUTED(), "Not currently sharing.".to_string()));
                }
            }
            Some(other) if !other.is_empty() && other != "start" && other != "on" => {
                self.notice = Some((ERROR(), format!("Usage: /share [stop]  (got '{other}')")));
            }
            _ => self.begin_live_share().await,
        }
    }

    /// Kick off the launch-time `--share` once the session settles — after any
    /// `--resume` load and with no startup picker open — so it pins the final
    /// session id, not the transient launch one. Fires at most once.
    pub(super) async fn maybe_start_live_share(&mut self) -> bool {
        if !self.live_requested
            || self.live_share.is_some()
            || self.live_share_starting
            || self.loading_resume.is_some()
            || self.overlay.blocks_input()
        {
            return false;
        }
        self.live_requested = false;
        self.begin_live_share().await;
        true
    }

    /// Persist, then kick off the share on a background task. No-op (re-shows the
    /// URL) when already sharing; ignored while a start is mid-flight.
    pub(super) async fn begin_live_share(&mut self) {
        if let Some(handle) = &self.live_share {
            let url = handle.url().to_string();
            self.announce_live_url(&url);
            return;
        }
        if self.live_share_starting {
            self.notice = Some((MUTED(), "Share is already starting…".to_string()));
            return;
        }
        // Persist first so the resolver can read this chat — even an empty one.
        if let Err(e) = self.persist_history().await {
            self.notice = Some((ERROR(), format!("Couldn't start share: {e:#}")));
            return;
        }

        self.live_share_starting = true;
        self.notice = Some((MUTED(), "Starting share…".to_string()));

        let tx = self.tx.clone();
        let session_store = self.session_store.clone();
        let session_id = self.session_id.clone();
        let cwd = self.real_cwd.clone();
        // Stamped so a stop//new//resume during the handshake makes this start stale.
        let share_gen = self.live_share_gen;
        tokio::spawn(async move {
            // `/share` without a login gets a clear notice; `--share` gated pre-TUI.
            if !crate::commands::share::device_linked().await {
                let _ = tx.send(RuntimeEvent::LiveShareReady {
                    share_gen,
                    result: Err(
                        "Sharing needs a linked account — run `aivo login`, then `/share` again."
                            .to_string(),
                    ),
                });
                return;
            }
            let project_root = if cwd.is_empty() {
                std::path::PathBuf::from(".")
            } else {
                std::path::PathBuf::from(&cwd)
            };
            let ctx = std::sync::Arc::new(
                crate::services::share_resolver::ResolverContext::from_system(
                    project_root,
                    session_store,
                ),
            );
            // Redact by default — a transcript can hold pasted secrets.
            let result = crate::services::share_live::start_live_share(session_id, ctx, true)
                .await
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(RuntimeEvent::LiveShareReady { share_gen, result });
        });
    }

    /// Show the URL notice and copy it to the clipboard — mouse capture makes
    /// drag-selecting one line unreliable, so paste-ready is the dependable path.
    /// Copy is best-effort and compiled out under test (never touches the real
    /// clipboard).
    fn announce_live_url(&mut self, url: &str) {
        // `notice_spans` paints the `LIVE_NOTICE_PREFIX` red and the URL a link color.
        self.notice = Some((LIVE(), format!("{LIVE_NOTICE_PREFIX}{url}")));
        #[cfg(not(test))]
        {
            if write_system_clipboard(url).is_ok() {
                self.show_toast("Share URL copied to clipboard");
            }
        }
    }

    /// Tear down the active share and invalidate any start still mid-handshake.
    /// Returns whether anything was stopped.
    pub(super) fn stop_live_share(&mut self) -> bool {
        self.live_share_gen = self.live_share_gen.wrapping_add(1);
        let was_starting = std::mem::take(&mut self.live_share_starting);
        if let Some(handle) = self.live_share.take() {
            handle.stop();
            true
        } else {
            was_starting
        }
    }

    pub(super) fn apply_live_share_ready(
        &mut self,
        share_gen: u64,
        result: std::result::Result<crate::services::share_live::LiveShareHandle, String>,
    ) {
        // Stale start (stopped//new//resumed mid-handshake): tear down, don't
        // install; `live_share_starting` belongs to a newer start, if any.
        if share_gen != self.live_share_gen {
            if let Ok(handle) = result {
                handle.stop();
            }
            return;
        }
        self.live_share_starting = false;
        match result {
            Ok(handle) => {
                let url = handle.url().to_string();
                self.live_share = Some(handle);
                self.announce_live_url(&url);
            }
            Err(msg) => self.notice = Some((ERROR(), msg)),
        }
    }

    /// Frame-tick check: a dead tunnel (no auto-reconnect) clears the badge and
    /// its server/refresher instead of lying until the URL stops resolving.
    pub(super) fn check_live_share_health(&mut self) {
        if self.live_share.as_ref().is_some_and(|h| h.is_dead()) {
            self.stop_live_share();
            self.notice = Some((
                ERROR(),
                "Share disconnected — /share to start a new one".to_string(),
            ));
        }
    }
}
