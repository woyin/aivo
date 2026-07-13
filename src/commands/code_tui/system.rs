use super::*;

use portable_pty::ChildKiller;
#[cfg(unix)]
use portable_pty::{Child, CommandBuilder, MasterPty, PtySize, native_pty_system};
use std::io::Read;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
#[cfg(windows)]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{fmt, io::Write};

pub(super) fn read_system_clipboard() -> Result<ClipboardPayload> {
    // Image paste is macOS-only (NSPasteboard via `swift`); text paste works on
    // every platform via the same CLI tools the copy path uses (see
    // `clipboard_read_candidates`).
    #[cfg(target_os = "macos")]
    if let Some(attachment) = read_macos_clipboard_image()? {
        return Ok(ClipboardPayload::Attachment(attachment));
    }

    let text = read_clipboard_text();
    Ok(if text.is_empty() {
        ClipboardPayload::Empty
    } else {
        ClipboardPayload::Text(text)
    })
}

/// Best-effort read of the clipboard's text via the platform's CLI tools, trying
/// each until one succeeds. Returns "" when none is installed or the clipboard is
/// empty (mirrors macOS, where `pbpaste` always exists) — the caller renders that
/// as "Clipboard is empty" rather than an error, and ordinary terminal paste
/// (bracketed paste → `Event::Paste`) is unaffected either way.
fn read_clipboard_text() -> String {
    for candidate in clipboard_read_candidates(current_clipboard_os()) {
        if let Ok(text) = read_command_stdout(candidate.program, candidate.args) {
            return text;
        }
    }
    String::new()
}

pub(super) fn read_command_stdout(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|err| anyhow::anyhow!("Failed to run '{}': {err}", program))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            anyhow::bail!("'{}' exited with {}", program, output.status);
        }
        anyhow::bail!("'{}' failed: {}", program, stderr);
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(target_os = "macos")]
pub(super) fn read_macos_clipboard_image() -> Result<Option<MessageAttachment>> {
    let script = r#"import AppKit
import Foundation

let pasteboard = NSPasteboard.general
if let data = pasteboard.data(forType: .png) {
    print(data.base64EncodedString())
} else if
    let tiff = pasteboard.data(forType: .tiff),
    let image = NSImage(data: tiff),
    let tiffData = image.tiffRepresentation,
    let bitmap = NSBitmapImageRep(data: tiffData),
    let png = bitmap.representation(using: .png, properties: [:])
{
    print(png.base64EncodedString())
}
"#;

    let mut command = Command::new("swift");
    command.env("CLANG_MODULE_CACHE_PATH", "/tmp/clang-module-cache");
    command.arg("-e").arg(script);

    let output = command
        .output()
        .map_err(|err| anyhow::anyhow!("Failed to access clipboard image: {err}"))?;
    if !output.status.success() {
        return Ok(None);
    }

    let data = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if data.is_empty() {
        return Ok(None);
    }

    Ok(Some(MessageAttachment {
        name: format!("clipboard-{}.png", Utc::now().format("%Y%m%d-%H%M%S")),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline { data },
    }))
}

/// Hard safety ceiling on what a `!cmd` local run captures, so a runaway flood
/// like `yes` can't grow memory without bound. The transcript still shows only
/// the first `MAX_OUTPUT_LINES` (in `render`); everything captured beyond that is
/// kept in memory (see `local_outputs`) so expanding the block in place shows a
/// big-but-finite command like `find .` whole.
/// Only output past this ceiling is dropped — the child is killed and the run
/// marked `truncated`. Generous enough that ordinary floody commands complete;
/// the 120s timeout bounds the time dimension.
const MAX_CAPTURED_LINES: usize = 50_000;
const MAX_CAPTURED_BYTES: usize = 8 * 1024 * 1024;
/// A single output line longer than this is truncated for storage/display, so one
/// newline-less stream (e.g. `cat` of a binary) can't become one giant line.
const MAX_LINE_CHARS: usize = 2000;
/// Wall-clock ceiling on a `!cmd` run; on hit the child is killed.
const SHELL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// A spawned `!cmd`: the open PTY, a line reader over its (merged stdout+stderr)
/// output, and the child handle (for `wait`/exit status). The command runs under a
/// pseudo-terminal so child programs line-buffer and stream live — the way grok's
/// CLI does it; plain pipes let many programs (`find`, `git`, builds) block-buffer,
/// so their output wouldn't appear until they exit.
///
/// Unix only. Windows uses plain pipes (see [`PipeShell`]) because `portable-pty`'s
/// ConPTY backend never returns EOF on the output pipe when the child exits — the
/// blocking read can hang the run indefinitely (`!cmd` never completed on Windows).
#[cfg(unix)]
pub(super) struct PtyShell {
    master: Box<dyn MasterPty + Send>,
    reader: Box<dyn Read + Send>,
    child: Box<dyn Child + Send + Sync>,
}

#[cfg(unix)]
impl PtyShell {
    /// A killer for the child, so the app can stop a running command on `esc` or
    /// exit (aborting the blocking read task alone won't kill the child).
    pub(super) fn killer_handle(&self) -> Box<dyn ChildKiller + Send + Sync> {
        self.child.clone_killer()
    }
}

/// Spawn `command` through the platform shell in `cwd` under a PTY. Returns the
/// pieces the caller streams from; `Err` only on PTY/spawn failure.
#[cfg(unix)]
pub(super) fn spawn_pty_shell(command: &str, cwd: &std::path::Path) -> std::io::Result<PtyShell> {
    // One shell-selection source of truth with the agent's `run_bash`: POSIX `sh`
    // on Unix, PowerShell on Windows (see `agent::sandbox::bare_shell`). `!cmd` is
    // the user's own command, so it runs unconfined (bare, no sandbox wrapper).
    let invocation = crate::agent::sandbox::bare_shell(command);
    let to_io = |e: anyhow::Error| std::io::Error::other(e.to_string());
    let pair = native_pty_system()
        .openpty(PtySize {
            rows: 40,
            cols: 200,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(to_io)?;
    let mut cmd = CommandBuilder::new(invocation.program.as_str());
    for arg in &invocation.args {
        cmd.arg(arg);
    }
    cmd.cwd(cwd);
    cmd.env("TERM", "xterm-256color");
    // `!cmd` runs under a PTY (so children line-buffer and stream) but we never
    // forward keystrokes to its stdin. A pager-spawning command (`git diff`,
    // `git log`, `systemctl`, `man`) would see a tty, launch `less`, render the
    // first screen, and block forever waiting for a keypress that can't arrive.
    // Neutralize pagers so output streams straight through; the ctrl+o overlay is
    // our scrollback. `cat`/empty both disable git's pager. Also fail fast instead
    // of blocking on an interactive credential prompt (`git push` over HTTPS).
    cmd.env("PAGER", "cat");
    cmd.env("GIT_PAGER", "cat");
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    let child = pair.slave.spawn_command(cmd).map_err(to_io)?;
    // Drop the slave so the master reader sees EOF once the child exits.
    drop(pair.slave);
    let reader = pair.master.try_clone_reader().map_err(to_io)?;
    Ok(PtyShell {
        master: pair.master,
        reader,
        child,
    })
}

/// Drive a [`PtyShell`] to completion: a reader thread streams output live (one
/// [`RuntimeEvent::LocalCommandLine`] per line) while this thread waits for the
/// child and enforces the wall-clock ceiling, then closes the PTY and emits a
/// terminal [`RuntimeEvent::LocalCommandDone`]. Reading stops — and the child is
/// killed — once output crosses [`MAX_CAPTURED_LINES`]/[`MAX_CAPTURED_BYTES`]
/// (`truncated`) or [`SHELL_TIMEOUT`] elapses. Runs under `spawn_blocking`.
///
/// The reader lives on its own thread so the controller can drop `master` to give
/// any still-writing grandchild (e.g. `yes &`) an EIO the instant the foreground
/// child exits, instead of the read holding the pty open. Unix only — Windows runs
/// `!cmd` through plain pipes ([`run_pipe_to_completion`]).
#[cfg(unix)]
pub(super) fn run_pty_to_completion(shell: PtyShell, tx: UnboundedSender<RuntimeEvent>) {
    let PtyShell {
        master,
        reader,
        mut child,
    } = shell;

    let truncated = Arc::new(AtomicBool::new(false));

    // Reader thread: blocking PTY reads, one emitted line per newline. It owns the
    // reader and its own killer so it can stop the child the instant output crosses
    // the capture cap, without waiting for the controller below to notice.
    let reader_thread = {
        let tx = tx.clone();
        let truncated = truncated.clone();
        let mut cap_killer = child.clone_killer();
        let mut reader = reader;
        std::thread::spawn(move || {
            let mut lines = 0usize;
            let mut bytes = 0usize;
            let mut acc: Vec<u8> = Vec::new();
            let mut buf = [0u8; 8192];
            'read: loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break, // EOF (child exited / PTY closed) or read error
                    Ok(n) => {
                        for &byte in &buf[..n] {
                            // The PTY translates `\n` → `\r\n`; the `\r` is dropped as a
                            // control char when the line is cleaned below. Flush a line on
                            // newline, or when a newline-less run gets over-long (e.g. `cat`
                            // of a binary) so `acc` can't grow without bound.
                            let flush = byte == b'\n' || acc.len() >= MAX_LINE_CHARS * 4;
                            if byte != b'\n' {
                                acc.push(byte);
                            }
                            if flush {
                                let capped = emit_pty_line(&tx, &acc, &mut lines, &mut bytes);
                                acc.clear();
                                if capped {
                                    truncated.store(true, Ordering::Release);
                                    let _ = cap_killer.kill();
                                    break 'read;
                                }
                            }
                        }
                    }
                }
            }
            // Flush a trailing partial line (output without a final newline).
            if !acc.is_empty() {
                let _ = emit_pty_line(&tx, &acc, &mut lines, &mut bytes);
            }
        })
    };

    // Controller: wait for the child, enforcing the wall-clock ceiling. A blocking
    // `wait()` was observed to hang on a macOS PTY after a kill, so poll `try_wait`,
    // ramping the interval so a quick command returns fast without busy-waiting a
    // slow one. On timeout, signal via a cloned killer (one SIGHUP / TerminateProcess);
    // the owned `child.kill()` would block while it escalates to SIGKILL.
    let mut timeout_killer = child.clone_killer();
    let start = std::time::Instant::now();
    let mut timed_out = false;
    let mut exit_status = None;
    let mut poll = std::time::Duration::from_millis(10);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_status = Some(status);
                break;
            }
            Ok(None) => {}
            Err(_) => break,
        }
        if truncated.load(Ordering::Acquire) {
            // The reader hit the capture cap and stopped draining the PTY. Stop
            // waiting and close it (drop(master) below): a child still flooding a
            // now-undrained PTY gets EIO and dies, where the cap's kill signal alone
            // may not stop it (e.g. `yes` blocked writing). Without this we'd block
            // here for the full timeout waiting on a child that can't make progress.
            break;
        }
        if start.elapsed() >= SHELL_TIMEOUT {
            timed_out = true;
            let _ = timeout_killer.kill();
            break;
        }
        std::thread::sleep(poll);
        poll = (poll * 2).min(std::time::Duration::from_millis(50));
    }

    // Close the PTY. Dropping our last `master` handle gives EIO to any still-writing
    // grandchild (e.g. `yes &`) so it dies instead of holding the pty open, and lets
    // the reader's `read()` EOF. The slave was dropped in `spawn_pty_shell` and
    // `master` is never cloned, so this is the last live PTY handle.
    drop(master);

    // Reap a child we stopped (cap/timeout) so it doesn't linger as a zombie; its exit
    // code is unused on that path. A natural exit already recorded `exit_status`.
    if exit_status.is_none() {
        for _ in 0..40 {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exit_status = Some(status);
                    break;
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
                Err(_) => break,
            }
        }
    }

    let _ = reader_thread.join();

    let truncated = truncated.load(Ordering::Acquire);
    if timed_out {
        let _ = tx.send(RuntimeEvent::LocalCommandLine {
            is_err: true,
            line: "command timed out after 120s".to_string(),
        });
    }

    // A natural finish reports the command's real exit code; a run WE stopped
    // (cap/timeout) reports -1, and folds timeout into `truncated` so the render shows
    // the "truncated" note, not a bogus `[exited -1]`.
    let stopped_early = truncated || timed_out;
    let exit_code = match exit_status {
        Some(status) if !stopped_early => i64::from(status.exit_code()),
        _ => -1,
    };
    let _ = tx.send(RuntimeEvent::LocalCommandDone {
        exit_code,
        truncated: stopped_early,
    });
}

/// Clean one raw output line for display: collapse cursor overwrites and strip
/// escapes (`render_output_line`), then cap an over-long (newline-less) line so a
/// binary stream can't become one giant line. Shared by the PTY and pipe readers.
fn clean_output_line(raw: &[u8]) -> String {
    let mut line = render_output_line(&String::from_utf8_lossy(raw));
    if line.chars().count() > MAX_LINE_CHARS {
        line = line.chars().take(MAX_LINE_CHARS).collect();
        line.push('…');
    }
    line
}

/// Emit one cleaned output line and roll the line/byte counters. Returns `true`
/// once a capture cap is crossed (caller should stop and kill).
#[cfg(unix)]
fn emit_pty_line(
    tx: &UnboundedSender<RuntimeEvent>,
    raw: &[u8],
    lines: &mut usize,
    bytes: &mut usize,
) -> bool {
    let line = clean_output_line(raw);
    *lines += 1;
    *bytes += line.len() + 1;
    // PTY merges stdout+stderr into one stream, so everything renders as stdout.
    let _ = tx.send(RuntimeEvent::LocalCommandLine {
        is_err: false,
        line,
    });
    *lines >= MAX_CAPTURED_LINES || *bytes >= MAX_CAPTURED_BYTES
}

// ---------------------------------------------------------------------------
// Windows `!cmd`: plain pipes, not ConPTY
// ---------------------------------------------------------------------------
//
// `portable-pty`'s ConPTY backend never delivers EOF on the output pipe when the
// child exits, so the blocking PTY read hangs forever and `!cmd` never completes
// (the run sits at "running…" with no output). Plain pipes — the same mechanism
// the agent's `run_bash` uses successfully on Windows — close deterministically
// when the child exits, so the reader threads EOF and the run finishes. The cost
// is that a program which block-buffers when stdout isn't a tty won't stream line
// by line (its output appears when it exits); correctness over live streaming.

/// A spawned `!cmd` on Windows: the child (shared so a killer can stop it on `esc`
/// or app exit) and its piped stdout/stderr, read on dedicated threads.
#[cfg(windows)]
pub(super) struct PipeShell {
    child: Arc<Mutex<std::process::Child>>,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
}

/// Kills the shared child on `esc`/app exit. Mirrors `portable-pty`'s `ChildKiller`
/// so `LocalCommandRun` holds one killer type across platforms. `Debug` is required
/// because `ChildKiller: Debug`.
#[cfg(windows)]
#[derive(Clone, Debug)]
struct PipeChildKiller {
    child: Arc<Mutex<std::process::Child>>,
}

#[cfg(windows)]
impl ChildKiller for PipeChildKiller {
    fn kill(&mut self) -> std::io::Result<()> {
        if let Ok(mut child) = self.child.lock() {
            // An already-exited child returns `InvalidInput` here; ignore it.
            let _ = child.kill();
        }
        Ok(())
    }

    fn clone_killer(&self) -> Box<dyn ChildKiller + Send + Sync> {
        Box::new(self.clone())
    }
}

#[cfg(windows)]
impl PipeShell {
    pub(super) fn killer_handle(&self) -> Box<dyn ChildKiller + Send + Sync> {
        Box::new(PipeChildKiller {
            child: self.child.clone(),
        })
    }
}

/// Spawn `command` through the platform shell in `cwd` with piped stdout/stderr and
/// no stdin (so an interactive prompt fails fast instead of blocking). `Err` only on
/// spawn failure.
#[cfg(windows)]
pub(super) fn spawn_pipe_shell(command: &str, cwd: &std::path::Path) -> std::io::Result<PipeShell> {
    // Same shell-selection source of truth as the agent's `run_bash` (PowerShell on
    // Windows); `!cmd` is the user's own command, so it runs bare (no sandbox wrap).
    let invocation = crate::agent::sandbox::bare_shell(command);
    let mut cmd = Command::new(invocation.program.as_str());
    cmd.args(&invocation.args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Neutralize pagers/credential prompts so output streams straight through and
        // nothing blocks waiting for a keypress (matches the Unix PTY path's env).
        .env("PAGER", "cat")
        .env("GIT_PAGER", "cat")
        .env("GIT_TERMINAL_PROMPT", "0");
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    Ok(PipeShell {
        child: Arc::new(Mutex::new(child)),
        stdout,
        stderr,
    })
}

/// Read one pipe (stdout or stderr) to EOF, emitting a [`RuntimeEvent::LocalCommandLine`]
/// per newline, until the capture caps are crossed (then it flags `truncated`, kills
/// the child, and stops). EOF arrives when the child exits and closes its write end.
#[cfg(windows)]
fn pipe_reader<R: Read>(
    mut reader: R,
    is_err: bool,
    tx: UnboundedSender<RuntimeEvent>,
    lines: Arc<std::sync::atomic::AtomicUsize>,
    bytes: Arc<std::sync::atomic::AtomicUsize>,
    truncated: Arc<AtomicBool>,
    child: Arc<Mutex<std::process::Child>>,
) {
    let mut acc: Vec<u8> = Vec::new();
    let mut buf = [0u8; 8192];
    'read: loop {
        match reader.read(&mut buf) {
            Ok(0) | Err(_) => break, // EOF (child exited / pipe closed) or read error
            Ok(n) => {
                for &byte in &buf[..n] {
                    // Flush a line on newline, or when a newline-less run gets over-long
                    // so `acc` can't grow without bound. The trailing `\r` of a Windows
                    // `\r\n` is dropped by `strip_ansi_and_controls` in `emit_pipe_line`.
                    let flush = byte == b'\n' || acc.len() >= MAX_LINE_CHARS * 4;
                    if byte != b'\n' {
                        acc.push(byte);
                    }
                    if flush {
                        let capped = emit_pipe_line(&tx, &acc, is_err, &lines, &bytes);
                        acc.clear();
                        if capped {
                            truncated.store(true, Ordering::Release);
                            if let Ok(mut c) = child.lock() {
                                let _ = c.kill();
                            }
                            break 'read;
                        }
                    }
                }
            }
        }
    }
    // Flush a trailing partial line (output without a final newline).
    if !acc.is_empty() {
        let _ = emit_pipe_line(&tx, &acc, is_err, &lines, &bytes);
    }
}

/// Like [`emit_pty_line`] but for the Windows pipe path: shared atomic counters
/// (two reader threads share the caps) and an explicit stdout/stderr flag.
#[cfg(windows)]
fn emit_pipe_line(
    tx: &UnboundedSender<RuntimeEvent>,
    raw: &[u8],
    is_err: bool,
    lines: &std::sync::atomic::AtomicUsize,
    bytes: &std::sync::atomic::AtomicUsize,
) -> bool {
    let line = clean_output_line(raw);
    let total_lines = lines.fetch_add(1, Ordering::Relaxed) + 1;
    let added = line.len() + 1;
    let total_bytes = bytes.fetch_add(added, Ordering::Relaxed) + added;
    let _ = tx.send(RuntimeEvent::LocalCommandLine { is_err, line });
    total_lines >= MAX_CAPTURED_LINES || total_bytes >= MAX_CAPTURED_BYTES
}

/// Drive a [`PipeShell`] to completion (Windows counterpart of
/// [`run_pty_to_completion`]): two reader threads stream stdout/stderr live while
/// this thread waits for the child and enforces [`SHELL_TIMEOUT`], then joins the
/// readers and emits a terminal [`RuntimeEvent::LocalCommandDone`]. Runs under
/// `spawn_blocking`.
#[cfg(windows)]
pub(super) fn run_pipe_to_completion(shell: PipeShell, tx: UnboundedSender<RuntimeEvent>) {
    let PipeShell {
        child,
        stdout,
        stderr,
    } = shell;
    let truncated = Arc::new(AtomicBool::new(false));
    let lines = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let bytes = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let spawn_reader = |reader: Option<Box<dyn Read + Send>>, is_err: bool| {
        reader.map(|reader| {
            let tx = tx.clone();
            let lines = lines.clone();
            let bytes = bytes.clone();
            let truncated = truncated.clone();
            let child = child.clone();
            std::thread::spawn(move || {
                pipe_reader(reader, is_err, tx, lines, bytes, truncated, child)
            })
        })
    };
    let out_thread = spawn_reader(stdout.map(|r| Box::new(r) as Box<dyn Read + Send>), false);
    let err_thread = spawn_reader(stderr.map(|r| Box::new(r) as Box<dyn Read + Send>), true);

    // Controller: poll for child exit, enforcing the wall-clock ceiling. `try_wait`
    // is non-blocking, so the lock is held only briefly and never contends with a
    // reader's kill. On timeout, kill the child (closing the pipes so the readers
    // EOF); a reader hitting the cap sets `truncated` and kills the child itself.
    let start = std::time::Instant::now();
    let mut timed_out = false;
    let mut exit_status = None;
    let mut poll = std::time::Duration::from_millis(10);
    loop {
        let waited = match child.lock() {
            Ok(mut c) => c.try_wait(),
            Err(_) => break, // poisoned (a reader panicked) — stop waiting
        };
        match waited {
            Ok(Some(status)) => {
                exit_status = Some(status);
                break;
            }
            Ok(None) => {}
            Err(_) => break,
        }
        if truncated.load(Ordering::Acquire) {
            break;
        }
        if start.elapsed() >= SHELL_TIMEOUT {
            timed_out = true;
            if let Ok(mut c) = child.lock() {
                let _ = c.kill();
            }
            break;
        }
        std::thread::sleep(poll);
        poll = (poll * 2).min(std::time::Duration::from_millis(50));
    }

    // The child has exited or been killed, so its stdout/stderr write ends close and
    // the reader threads EOF; joining them drains the final output. (A backgrounded
    // grandchild still holding the pipe is the one case a join can wait on — the same
    // behavior as the agent's `run_bash`; the user can still `esc` to abandon it.)
    if let Some(handle) = out_thread {
        let _ = handle.join();
    }
    if let Some(handle) = err_thread {
        let _ = handle.join();
    }

    // Reap a child we stopped (cap/timeout) for its status; unused on that path.
    if exit_status.is_none()
        && let Ok(mut c) = child.lock()
    {
        exit_status = c.try_wait().ok().flatten();
    }

    let truncated = truncated.load(Ordering::Acquire);
    if timed_out {
        let _ = tx.send(RuntimeEvent::LocalCommandLine {
            is_err: true,
            line: "command timed out after 120s".to_string(),
        });
    }

    // A natural finish reports the real exit code; a run WE stopped (cap/timeout)
    // reports -1 and folds timeout into `truncated` for the render.
    let stopped_early = truncated || timed_out;
    let exit_code = match exit_status {
        Some(status) if !stopped_early => status.code().map(i64::from).unwrap_or(-1),
        _ => -1,
    };
    let _ = tx.send(RuntimeEvent::LocalCommandDone {
        exit_code,
        truncated: stopped_early,
    });
}

// ---------------------------------------------------------------------------
// Cross-platform `!cmd` dispatch
// ---------------------------------------------------------------------------

/// A spawned `!cmd`, backed by a PTY on Unix and plain pipes on Windows. The caller
/// (`start_local_command`) is platform-agnostic: it spawns one of these, grabs a
/// killer, and hands it to [`run_local_to_completion`].
pub(super) enum LocalShell {
    #[cfg(unix)]
    Pty(PtyShell),
    #[cfg(windows)]
    Pipe(PipeShell),
}

impl LocalShell {
    pub(super) fn killer_handle(&self) -> Box<dyn ChildKiller + Send + Sync> {
        match self {
            #[cfg(unix)]
            LocalShell::Pty(shell) => shell.killer_handle(),
            #[cfg(windows)]
            LocalShell::Pipe(shell) => shell.killer_handle(),
        }
    }
}

/// Spawn `command` for `!cmd` in `cwd`, choosing the platform backend.
pub(super) fn spawn_local_shell(
    command: &str,
    cwd: &std::path::Path,
) -> std::io::Result<LocalShell> {
    #[cfg(unix)]
    {
        Ok(LocalShell::Pty(spawn_pty_shell(command, cwd)?))
    }
    #[cfg(windows)]
    {
        Ok(LocalShell::Pipe(spawn_pipe_shell(command, cwd)?))
    }
}

/// Drive a spawned `!cmd` to completion (the platform backend's runner). Runs under
/// `spawn_blocking`.
pub(super) fn run_local_to_completion(shell: LocalShell, tx: UnboundedSender<RuntimeEvent>) {
    match shell {
        #[cfg(unix)]
        LocalShell::Pty(shell) => run_pty_to_completion(shell, tx),
        #[cfg(windows)]
        LocalShell::Pipe(shell) => run_pipe_to_completion(shell, tx),
    }
}

pub(super) fn parse_slash_command(input: &str) -> Result<SlashCommand> {
    let trimmed = input.trim();
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or_default();
    let argument = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    match command {
        "new" => Ok(SlashCommand::New),
        "exit" | "quit" => Ok(SlashCommand::Exit),
        "resume" => Ok(SlashCommand::Resume(argument)),
        "model" => Ok(SlashCommand::Model(argument)),
        "key" => Ok(SlashCommand::Key(argument)),
        "attach" => Ok(SlashCommand::Attach(
            argument.ok_or_else(|| anyhow::anyhow!("Usage: /attach <path>"))?,
        )),
        "detach" => Ok(SlashCommand::Detach(
            argument
                .ok_or_else(|| anyhow::anyhow!("Usage: /detach <n>"))?
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("Usage: /detach <n>"))?,
        )),
        "copy" => Ok(SlashCommand::Copy(match argument {
            None => None,
            Some(value) => Some(
                value
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("Usage: /copy [n]"))?,
            ),
        })),
        "skills" => Ok(SlashCommand::Skills(argument)),
        "agents" => Ok(SlashCommand::Agents(argument)),
        "mcp" => Ok(SlashCommand::Mcp(argument)),
        "goal" => Ok(SlashCommand::Goal(argument)),
        "plan" => Ok(SlashCommand::Plan(argument)),
        "review" => Ok(SlashCommand::Review(argument)),
        "memory" => Ok(SlashCommand::Memory),
        "effort" => Ok(SlashCommand::Effort(argument)),
        "create-skill" => Ok(SlashCommand::CreateSkill(argument)),
        "rewind" | "undo" | "unwind" => Ok(SlashCommand::Rewind),
        "config" => Ok(SlashCommand::Config),
        "compact" => Ok(SlashCommand::Compact {
            fast: argument.as_deref() == Some("fast"),
        }),
        "context" => Ok(SlashCommand::Context),
        "share" => Ok(SlashCommand::Share(argument)),
        "login" => Ok(SlashCommand::Login),
        "logout" => Ok(SlashCommand::Logout),
        "usage" => Ok(SlashCommand::Usage),
        "help" => Ok(SlashCommand::Help),
        "" => anyhow::bail!("Type a command after '/'"),
        other => anyhow::bail!("Unknown command '/{other}'"),
    }
}

pub(super) fn reduce_motion_requested() -> bool {
    env::var("AIVO_REDUCE_MOTION")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ClipboardOs {
    Macos,
    Linux,
    Windows,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ClipboardCommand {
    pub(super) program: &'static str,
    pub(super) args: &'static [&'static str],
}

impl fmt::Display for ClipboardCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            write!(f, "{}", self.program)
        } else {
            write!(f, "{} {}", self.program, self.args.join(" "))
        }
    }
}

pub(super) fn current_clipboard_os() -> ClipboardOs {
    if cfg!(target_os = "macos") {
        ClipboardOs::Macos
    } else if cfg!(target_os = "linux") {
        ClipboardOs::Linux
    } else if cfg!(target_os = "windows") {
        ClipboardOs::Windows
    } else {
        ClipboardOs::Other
    }
}

pub(super) fn clipboard_command_candidates(os: ClipboardOs) -> Vec<ClipboardCommand> {
    match os {
        ClipboardOs::Macos => vec![ClipboardCommand {
            program: "pbcopy",
            args: &[],
        }],
        ClipboardOs::Linux => vec![
            ClipboardCommand {
                program: "wl-copy",
                args: &[],
            },
            ClipboardCommand {
                program: "xclip",
                args: &["-selection", "clipboard"],
            },
            ClipboardCommand {
                program: "xsel",
                args: &["--clipboard", "--input"],
            },
        ],
        ClipboardOs::Windows => vec![ClipboardCommand {
            program: "powershell.exe",
            args: &["-NoProfile", "-Command", "Set-Clipboard"],
        }],
        ClipboardOs::Other => Vec::new(),
    }
}

/// The CLI tools that print the clipboard's text to stdout, tried in order — the
/// read counterpart of [`clipboard_command_candidates`]. (Image paste is handled
/// separately and is macOS-only.)
pub(super) fn clipboard_read_candidates(os: ClipboardOs) -> Vec<ClipboardCommand> {
    match os {
        ClipboardOs::Macos => vec![ClipboardCommand {
            program: "pbpaste",
            args: &[],
        }],
        ClipboardOs::Linux => vec![
            ClipboardCommand {
                program: "wl-paste",
                args: &["--no-newline"],
            },
            ClipboardCommand {
                program: "xclip",
                args: &["-selection", "clipboard", "-o"],
            },
            ClipboardCommand {
                program: "xsel",
                args: &["--clipboard", "--output"],
            },
        ],
        ClipboardOs::Windows => vec![ClipboardCommand {
            program: "powershell.exe",
            args: &["-NoProfile", "-Command", "Get-Clipboard"],
        }],
        ClipboardOs::Other => Vec::new(),
    }
}

pub(super) fn write_system_clipboard(text: &str) -> Result<()> {
    let mut errors = Vec::new();
    for candidate in clipboard_command_candidates(current_clipboard_os()) {
        match write_clipboard_command(&candidate, text) {
            Ok(()) => return Ok(()),
            Err(err) => errors.push(format!("{candidate}: {err}")),
        }
    }

    write_osc52_clipboard(text).map_err(|osc_err| {
        let detail = if errors.is_empty() {
            osc_err.to_string()
        } else {
            format!("{}; OSC52: {osc_err}", errors.join("; "))
        };
        anyhow::anyhow!(detail)
    })
}

fn write_clipboard_command(candidate: &ClipboardCommand, text: &str) -> Result<()> {
    let mut child = Command::new(candidate.program)
        .args(candidate.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    if let Some(stdin) = &mut child.stdin {
        stdin.write_all(text.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if stderr.is_empty() {
        anyhow::bail!("exited with {}", output.status);
    }
    anyhow::bail!("{stderr}");
}

fn write_osc52_clipboard(text: &str) -> Result<()> {
    let encoded =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, text.as_bytes());
    let mut stderr = std::io::stderr();
    write!(stderr, "\x1b]52;c;{encoded}\x07")?;
    stderr.flush()?;
    Ok(())
}

pub(super) fn is_help_shortcut(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::F(1))
}

/// The auto-approve toggle chord: Shift+Tab (arrives as `BackTab`, or `Tab` with
/// SHIFT on some terminals) — aligned with Claude Code's permission-mode switch.
pub(super) fn is_auto_approve_toggle(key: KeyEvent) -> bool {
    matches!(key.code, KeyCode::BackTab)
        || (matches!(key.code, KeyCode::Tab) && key.modifiers.contains(KeyModifiers::SHIFT))
}

pub(super) fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

pub(super) fn copilot_token_manager_for_key(key: &ApiKey) -> Option<Arc<CopilotTokenManager>> {
    if key.base_url == "copilot" {
        Some(Arc::new(CopilotTokenManager::new(
            key.key.as_str().to_string(),
        )))
    } else {
        None
    }
}
