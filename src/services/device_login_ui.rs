//! Shared terminal UX for device-code sign-in flows (`aivo login`, SuperGrok,
//! Kimi Code, GitHub Copilot).
//!
//! Centralizes the pieces every device flow wants: an Enter-to-open-browser
//! convenience that never gates the poll, echo suppression so the Enter newline
//! doesn't scroll the in-place spinner, and a Ctrl+C-catching wait that restores
//! the terminal instead of leaving echo off on a half-drawn line.

use crate::style;

/// Prints `prompt`, then on an interactive terminal reads one line and returns
/// it trimmed. Returns `None` on a non-TTY session — there's no one to ask.
pub async fn read_line_if_tty(prompt: &str) -> Option<String> {
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
        if read_line_if_tty("").await.is_some()
            && crate::services::browser_open::open_url(&verify_url).is_err()
        {
            println!(
                "  {}",
                style::dim("(couldn't open a browser — visit the URL above)")
            );
        }
    })
}

/// Runs `poll` to completion while offering Enter-to-open-browser and honoring
/// Ctrl+C. Returns `Some(poll_output)` on completion, or `None` if the user
/// pressed Ctrl+C mid-poll. Echo is suppressed for the duration (Unix) so the
/// Enter newline doesn't duplicate the spinner line, and the browser opener is
/// aborted before returning.
pub async fn wait_for_approval<F, T>(open_url: String, poll: F) -> Option<T>
where
    F: std::future::Future<Output = T>,
{
    // Scoped so echo is restored — and the Enter→browser opener cancelled —
    // the moment polling ends, before any later output.
    let _echo_guard = EchoGuard::disable();
    let opener = spawn_browser_opener_on_enter(open_url);
    let (spinning, spinner_handle) = style::start_spinner(Some(" Waiting for approval…"));
    // Catch Ctrl+C here rather than let the default SIGINT kill the process
    // mid-poll: that skips `EchoGuard`'s drop, leaving the terminal with echo
    // off and a half-drawn spinner line.
    let outcome = tokio::select! {
        r = poll => Some(r),
        _ = tokio::signal::ctrl_c() => None,
    };
    style::stop_spinner(&spinning);
    let _ = spinner_handle.await;
    opener.abort();
    outcome
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
