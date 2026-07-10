//! `aivo logs share <session-id>` command — resolves the session, redacts it,
//! confirms with the user, then binds a local HTTP server. The tunnel client
//! to `s.getaivo.dev` lives in `share_tunnel.rs`; `--debug-local-only` skips
//! the tunnel so the local server can be exercised end to end via
//! `curl 127.0.0.1:<port>/state` without the public host.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;

use crate::cli::ShareArgs;
use crate::errors::{CLIError, ErrorCategory, ExitCode};
use crate::services::session_store::SessionStore;
use crate::services::share_local_server::{build_state, start_local_server};
use crate::services::share_payload::SharePayload;
use crate::services::share_picker;
use crate::services::share_redact::{RedactCtx, redact};
use crate::services::share_resolver::{PluginTranscript, ResolverContext, resolve_session};
use crate::services::share_tunnel;
use crate::services::shutdown_signal::ShutdownSignal;
use crate::services::system_env::expand_tilde;
use crate::style;

/// Plugin tools that declared a transcript source aivo can read (`aivo share`
/// reuses the matching built-in reader). Resolved here, where the plugin
/// registry is reachable, and handed to the resolver as plain data.
fn plugin_transcript_sources() -> HashMap<String, PluginTranscript> {
    crate::plugin::installed_transcript_sources()
        .into_iter()
        .map(|(name, format, dir)| {
            // `native` format emits its own transcript via the binary — resolve
            // it here, where discovery is reachable.
            let bin = crate::plugin::discover(&name);
            (
                name,
                PluginTranscript {
                    format,
                    dir: expand_tilde(&dir),
                    bin,
                },
            )
        })
        .collect()
}

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
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(self, args: ShareArgs) -> Result<ExitCode> {
        // Publishing needs a linked device; `--debug-local-only` (127.0.0.1) is exempt.
        if !args.debug_local_only {
            ensure_device_linked().await?;
        }

        // Always resolve from the user's current working directory — the
        // native CLI extractors all match by canonical cwd.
        let cwd = std::env::current_dir()?;

        // No id passed → open the picker. The picker bails on non-TTY with
        // an actionable message; a clean cancel returns Ok(None) so we exit
        // quietly without printing help (the user explicitly chose to back
        // out, they don't need a wall of flags).
        let session_id = match args.session_id.as_deref().filter(|s| !s.is_empty()) {
            Some(id) => id.to_string(),
            None => match share_picker::pick_session_id(
                &self.session_store,
                &cwd,
                args.all,
                "Share which session?",
                "aivo logs share <id>",
            )
            .await?
            {
                Some(id) => id,
                None => return Ok(ExitCode::Success),
            },
        };

        let resolver_ctx = Arc::new(
            ResolverContext::from_system(cwd, self.session_store)
                .with_plugin_transcripts(plugin_transcript_sources()),
        );

        let resolved = resolve_session(&session_id, &resolver_ctx).await?;
        let mut payload = resolved.payload;

        let redact_ctx = RedactCtx::from_system();
        if args.no_redact {
            payload.meta.redacted = false;
        } else {
            let (red, _hits) = redact(payload, &redact_ctx);
            payload = red;
        }

        print_preview(&payload);

        let session_id_owned = session_id.to_string();
        let state = build_state(session_id_owned, payload, redact_ctx, Some(resolver_ctx));

        let shutdown = ShutdownSignal::new();
        let (port, handle) = start_local_server("127.0.0.1:0", state, shutdown.clone()).await?;
        let local_base = format!("http://127.0.0.1:{port}");

        // Single Ctrl+C watcher fires `shutdown` for everyone — tunnel +
        // local share. The tunnel's main loop selects on it; the local
        // share's parked long-polls do too. One signal, two consumers,
        // no per-component signal handling drift.
        {
            let shutdown = shutdown.clone();
            tokio::spawn(async move {
                let _ = tokio::signal::ctrl_c().await;
                shutdown.fire();
            });
        }

        let exit_code = if args.debug_local_only {
            let state_url = format!("{local_base}/state");
            print_share_started(&state_url);
            if args.open {
                let _ = crate::services::browser_open::open_url(&state_url);
            }
            shutdown.wait().await;
            println!();
            ExitCode::Success
        } else {
            match share_tunnel::run_tunnel(
                local_base,
                share_tunnel::TunnelUi::Cli {
                    open_in_browser: args.open,
                },
                shutdown.clone(),
            )
            .await
            {
                Ok(()) => ExitCode::Success,
                Err(e) => {
                    eprintln!("  {} {e}", style::red("✗"));
                    ExitCode::NetworkError
                }
            }
        };

        shutdown.fire();
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
                "Share an AI session (claude, codex, gemini, pi, opencode, code) via a tunneled, ephemeral viewer URL. Closing the process invalidates the link."
            )
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
            "Unique id prefix from `aivo logs` (omit → picker)",
        );
        println!();
        println!("{}", style::bold("Options:"));
        print_opt(
            "--no-redact",
            "Skip redaction of keys, tokens, $HOME, secrets",
        );
        print_opt("--all", "Pick from every project (default: current cwd)");
        print_opt("--open", "Open the share URL in the browser when ready");
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
    }
}

/// Refuse to share from an unlinked device. Fails open when the server is
/// unreachable so a network blip doesn't block a linked user.
async fn ensure_device_linked() -> Result<()> {
    use crate::commands::login::{AccountSync, sync_account_status};
    match sync_account_status().await {
        AccountSync::Linked(account) => {
            println!(
                "  {} Sharing as {}",
                style::success_symbol(),
                style::dim(account.display())
            );
            Ok(())
        }
        AccountSync::Unverified(Some(_)) => Ok(()),
        AccountSync::Unlinked { .. } | AccountSync::Unverified(None) => Err(not_linked_error()),
    }
}

/// Non-printing device-link check for the chat TUI's `--share` / `/share`. Fails
/// open on a server hiccup, like `ensure_device_linked`.
pub(crate) async fn device_linked() -> bool {
    use crate::commands::login::{AccountSync, sync_account_status};
    matches!(
        sync_account_status().await,
        AccountSync::Linked(_) | AccountSync::Unverified(Some(_))
    )
}

pub(crate) fn not_linked_error() -> anyhow::Error {
    CLIError::new(
        "Sharing requires a linked aivo account.",
        ErrorCategory::Auth,
        None::<String>,
        Some("Run `aivo login` to link this device, then try sharing again."),
    )
    .into()
}

fn print_preview(payload: &SharePayload) {
    let kb = payload.approximate_chars() / 1024;
    let model = payload.model.as_deref().unwrap_or("(none)");
    println!(
        "    {} messages · ~{} KB · {} · {}",
        payload.messages.len(),
        kb,
        style::cyan(&payload.source_cli),
        style::cyan(model),
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
        "  {} {}   {}",
        style::green("✓"),
        style::bold("Sharing"),
        style::dim("Ctrl+C to stop")
    );
    println!("    {}", style::cyan(url));
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::{ExitCode, exit_code_for_error};

    #[test]
    fn not_linked_error_is_auth_category() {
        let e = not_linked_error();
        assert_eq!(exit_code_for_error(&e), ExitCode::AuthError);
    }

    #[test]
    fn not_linked_error_points_at_login() {
        let e = not_linked_error();
        assert!(e.to_string().contains("aivo login"));
    }
}
