//! `aivo logs share <session-id>` command — resolves the session, redacts it,
//! confirms with the user, then binds a local HTTP server. The tunnel client
//! to `s.getaivo.dev` lives in `share_tunnel.rs`; `--debug-local-only` skips
//! the tunnel so the local server can be exercised end to end via
//! `curl 127.0.0.1:<port>/state` without the public host.

use std::sync::Arc;

use anyhow::Result;

use crate::cli::ShareArgs;
use crate::errors::ExitCode;
use crate::services::session_store::SessionStore;
use crate::services::share_local_server::{build_state, start_local_server};
use crate::services::share_payload::{RedactionHit, SharePayload};
use crate::services::share_picker;
use crate::services::share_redact::{RedactCtx, redact};
use crate::services::share_resolver::{ResolverContext, resolve_session};
use crate::services::share_tunnel;
use crate::style;

pub struct ShareCommand {
    session_store: SessionStore,
}

impl ShareCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(self, args: ShareArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(self, args: ShareArgs) -> Result<ExitCode> {
        // Always resolve from the user's current working directory — the
        // native CLI extractors all match by canonical cwd.
        let cwd = std::env::current_dir()?;

        // No id passed → open the picker. The picker bails on non-TTY with
        // an actionable message; a clean cancel returns Ok(None) so we exit
        // quietly without printing help (the user explicitly chose to back
        // out, they don't need a wall of flags).
        let session_id = match args.session_id.as_deref().filter(|s| !s.is_empty()) {
            Some(id) => id.to_string(),
            None => match share_picker::pick_session_id(&self.session_store, &cwd, args.all).await?
            {
                Some(id) => id,
                None => return Ok(ExitCode::Success),
            },
        };

        let resolver_ctx = Arc::new(ResolverContext::from_system(cwd, self.session_store));

        let resolved = resolve_session(&session_id, &resolver_ctx).await?;
        let mut payload = resolved.payload;

        let redact_ctx = RedactCtx::from_system();
        let report: Vec<RedactionHit> = if args.no_redact {
            payload.meta.redacted = false;
            Vec::new()
        } else {
            let (red, hits) = redact(payload, &redact_ctx);
            payload = red;
            hits
        };

        print_preview(&payload, &report, args.live);

        let live = args.live;
        let session_id_owned = session_id.to_string();
        let resolver_ctx_for_state = if live { Some(resolver_ctx) } else { None };
        let state = build_state(
            session_id_owned,
            payload,
            live,
            redact_ctx,
            resolver_ctx_for_state,
        );

        let shutdown = Arc::new(tokio::sync::Notify::new());
        let (port, handle) = start_local_server("127.0.0.1:0", state, shutdown.clone()).await?;
        let local_base = format!("http://127.0.0.1:{port}");

        let exit_code = if args.debug_local_only {
            let state_url = format!("{local_base}/state");
            print_share_started(&state_url);
            if args.open {
                let _ = crate::services::browser_open::open_url(&state_url);
            }
            let _ = tokio::signal::ctrl_c().await;
            println!();
            ExitCode::Success
        } else {
            match share_tunnel::run_tunnel(local_base, args.open).await {
                Ok(()) => ExitCode::Success,
                Err(e) => {
                    eprintln!("{} {e}", style::red("Tunnel error:"));
                    ExitCode::NetworkError
                }
            }
        };

        shutdown.notify_waiters();
        // Best-effort drain — the loop is non-blocking and exits on the next
        // accept-or-shutdown poll.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = handle.await;
        println!("{} {}", style::green("✓"), style::dim("Stopped sharing"));
        Ok(exit_code)
    }

    pub fn print_help() {
        println!(
            "{} aivo logs share [SESSION_ID] [OPTIONS]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim(
                "Share an AI session (claude, codex, gemini, pi, opencode, chat, amp) via a"
            )
        );
        println!(
            "{}",
            style::dim("tunneled, ephemeral viewer URL. Closing the process invalidates the link.")
        );
        println!();
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<22}", flag)),
                style::dim(desc)
            );
        };
        println!("{}", style::bold("Arguments:"));
        print_opt(
            "[SESSION_ID]",
            "Any unique id prefix from `aivo logs`; omit to open the picker",
        );
        println!();
        println!("{}", style::bold("Options:"));
        print_opt(
            "--live",
            "Follow ongoing changes (default: snapshot at share time)",
        );
        print_opt(
            "--no-redact",
            "Skip redaction (default: scrub API keys, OAuth tokens, $HOME, secret env)",
        );
        print_opt(
            "--all",
            "Picker shows sessions from every project (default: current cwd)",
        );
        print_opt(
            "--open",
            "Open the share URL in the default browser once the link is ready",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!(
            "  {}",
            style::dim("aivo logs share                   # pick from current project")
        );
        println!(
            "  {}",
            style::dim("aivo logs share --all             # pick from every project")
        );
        println!(
            "  {}",
            style::dim("aivo logs share 1335c631          # share by id prefix")
        );
        println!(
            "  {}",
            style::dim("aivo logs share T-019cafea --live # amp thread, follow changes")
        );
    }
}

fn print_preview(payload: &SharePayload, report: &[RedactionHit], live: bool) {
    let chars = payload.approximate_chars();
    let kb = chars / 1024;
    println!(
        "{} {} messages, ~{} KB after redaction",
        style::bold("Share preview:"),
        payload.messages.len(),
        kb,
    );
    if !report.is_empty() {
        let summary = report
            .iter()
            .map(|h| format!("{} {}", h.count, h.category))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {} {}", style::dim("redacted:"), style::dim(summary));
    }
    let model = payload.model.as_deref().unwrap_or("(none)");
    println!(
        "  {} {} · {} {} · {} {}",
        style::dim("source:"),
        style::cyan(&payload.source_cli),
        style::dim("model:"),
        style::cyan(model),
        style::dim("mode:"),
        style::cyan(if live { "live" } else { "snapshot" }),
    );
}

/// Render the success banner shown once a share is live. Used by both the
/// `--debug-local-only` path here and the tunnel path in `share_tunnel.rs`
/// so they print the same shape regardless of where the URL came from.
///
/// Begins with an ANSI clear-line so any spinner residue from the
/// connection phase is wiped out before the banner draws.
pub(crate) fn print_share_started(url: &str) {
    use std::io::Write;
    // \r\x1b[2K = carriage return + erase entire line. `start_spinner`'s
    // companion `stop_spinner` only clears one char, which leaves long
    // labels showing through. Doing it here once means callers don't have
    // to remember.
    eprint!("\r\x1b[2K");
    let _ = std::io::stderr().flush();

    println!();
    println!(
        "{} {}  {}",
        style::green("✓"),
        style::bold("Sharing"),
        style::dim("Ctrl+C stop sharing")
    );
    println!("  {}", style::cyan(url));
    println!();
}
