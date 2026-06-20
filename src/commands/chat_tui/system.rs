use super::*;

use portable_pty::{Child, ChildKiller, CommandBuilder, MasterPty, PtySize, native_pty_system};
use std::io::Read;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
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
/// kept in memory for the `ctrl+o` full-output pager (see `last_local_output`),
/// so a big-but-finite command like `find .` is captured whole and scrollable.
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
/// Windows-only grace after the child exits, letting the reader drain trailing
/// ConPTY output before we close the pseudoconsole (`ClosePseudoConsole` can drop
/// not-yet-read bytes). Best-effort: output mostly streams live as it's produced,
/// so this only covers the final buffered chunk.
#[cfg(windows)]
const PTY_DRAIN_GRACE: std::time::Duration = std::time::Duration::from_millis(150);

/// A spawned `!cmd`: the open PTY, a line reader over its (merged stdout+stderr)
/// output, and the child handle (for `wait`/exit status). The command runs under a
/// pseudo-terminal so child programs line-buffer and stream live — the way grok's
/// CLI does it; plain pipes let many programs (`find`, `git`, builds) block-buffer,
/// so their output wouldn't appear until they exit.
pub(super) struct PtyShell {
    master: Box<dyn MasterPty + Send>,
    reader: Box<dyn Read + Send>,
    child: Box<dyn Child + Send + Sync>,
}

impl PtyShell {
    /// A killer for the child, so the app can stop a running command on `esc` or
    /// exit (aborting the blocking read task alone won't kill the child).
    pub(super) fn killer_handle(&self) -> Box<dyn ChildKiller + Send + Sync> {
        self.child.clone_killer()
    }
}

/// Spawn `command` through the platform shell in `cwd` under a PTY. Returns the
/// pieces the caller streams from; `Err` only on PTY/spawn failure.
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
/// The reader lives on its own thread because closing the PTY is what unblocks a
/// blocked read on Windows: `portable-pty`'s ConPTY output pipe never returns EOF
/// when the child exits — only `ClosePseudoConsole` (dropping `master`) does — so a
/// single-threaded read-then-close would hang forever. Waiting on the child here
/// lets us drop `master` the moment it exits, unblocking the reader on every
/// platform (broken pipe on Windows; on Unix the read already EOF'd).
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

    // Windows: the reader is still blocked (ConPTY won't EOF on child exit), and
    // `ClosePseudoConsole` can discard output it hasn't read yet — a short grace lets
    // it drain first. Skipped on Unix (the read already EOF'd when the slave closed)
    // and on timeout (we're tearing down regardless).
    #[cfg(windows)]
    if !timed_out {
        std::thread::sleep(PTY_DRAIN_GRACE);
    }

    // Close the PTY. Dropping our last `master` handle calls `ClosePseudoConsole` on
    // Windows — which unblocks the reader's blocked `read()` (broken pipe) — and on
    // Unix gives EIO to any still-writing grandchild (e.g. `yes &`) so it dies instead
    // of holding the pty open. LOAD-BEARING: `master` must be the last live PTY handle
    // (the slave is dropped in `spawn_pty_shell` and `master` is never cloned) or the
    // reader never unblocks and the `join` below hangs.
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

/// Clean one raw output line (strip the PTY's ANSI/color escapes and control
/// bytes, cap an over-long line), emit it, and roll the line/byte counters.
/// Returns `true` once a capture cap is crossed (caller should stop and kill).
fn emit_pty_line(
    tx: &UnboundedSender<RuntimeEvent>,
    raw: &[u8],
    lines: &mut usize,
    bytes: &mut usize,
) -> bool {
    let mut line = strip_ansi_and_controls(&String::from_utf8_lossy(raw));
    if line.chars().count() > MAX_LINE_CHARS {
        line = line.chars().take(MAX_LINE_CHARS).collect();
        line.push('…');
    }
    *lines += 1;
    *bytes += line.len() + 1;
    // PTY merges stdout+stderr into one stream, so everything renders as stdout.
    let _ = tx.send(RuntimeEvent::LocalCommandLine {
        is_err: false,
        line,
    });
    *lines >= MAX_CAPTURED_LINES || *bytes >= MAX_CAPTURED_BYTES
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
        "exit" => Ok(SlashCommand::Exit),
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
        "mcp" => Ok(SlashCommand::Mcp(argument)),
        "agent" => Ok(SlashCommand::Agent(argument)),
        "goal" => Ok(SlashCommand::Goal(argument)),
        "create-skill" => Ok(SlashCommand::CreateSkill(argument)),
        // `undo` kept as a hidden alias for muscle memory; only `/rewind` is advertised.
        "rewind" | "undo" => Ok(SlashCommand::Rewind),
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
