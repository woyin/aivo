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

/// Best-effort clipboard text via the platform's CLI tools; "" when none is
/// installed or the clipboard is empty (the caller renders "Clipboard is empty"
/// rather than an error).
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

// JXA: `4` = NSBitmapImageRep.FileType.png; the named constant is undefined there (coerces to TIFF).
#[cfg(target_os = "macos")]
pub(super) fn read_macos_clipboard_image() -> Result<Option<MessageAttachment>> {
    let script = r#"ObjC.import("AppKit");
const pb = $.NSPasteboard.generalPasteboard;
let out = "";
const png = pb.dataForType($.NSPasteboardTypePNG);
if (!png.isNil()) {
    out = ObjC.unwrap(png.base64EncodedStringWithOptions(0));
} else {
    const tiff = pb.dataForType($.NSPasteboardTypeTIFF);
    if (!tiff.isNil()) {
        const rep = $.NSBitmapImageRep.imageRepWithData(tiff);
        if (!rep.isNil()) {
            const data = rep.representationUsingTypeProperties(4, $.NSDictionary.dictionary);
            if (!data.isNil()) out = ObjC.unwrap(data.base64EncodedStringWithOptions(0));
        }
    }
}
out"#;

    let output = Command::new("osascript")
        .args(["-l", "JavaScript", "-e", script])
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

/// Capture caps for a `!cmd` run: past either ceiling the child is killed and the
/// run marked `truncated`, so a flood like `yes` can't grow memory unbounded. The
/// transcript renders only `MAX_OUTPUT_LINES`; the rest stays in `local_outputs`
/// so expanding the block shows a big-but-finite command whole.
const MAX_CAPTURED_LINES: usize = 50_000;
const MAX_CAPTURED_BYTES: usize = 8 * 1024 * 1024;
/// Per-line cap so a newline-less stream (e.g. `cat` of a binary) can't become
/// one giant line.
const MAX_LINE_CHARS: usize = 2000;
/// Wall-clock ceiling on a `!cmd` run; on hit the child is killed.
const SHELL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// A spawned `!cmd` under a pseudo-terminal so child programs line-buffer and
/// stream live (plain pipes make `find`/`git`/builds block-buffer until exit).
/// Unix only: Windows uses [`PipeShell`] because `portable-pty`'s ConPTY backend
/// never returns EOF on the output pipe when the child exits, hanging the
/// blocking read indefinitely.
#[cfg(unix)]
pub(super) struct PtyShell {
    master: Box<dyn MasterPty + Send>,
    reader: Box<dyn Read + Send>,
    child: Box<dyn Child + Send + Sync>,
}

#[cfg(unix)]
impl PtyShell {
    /// Killer for the child so `esc`/exit can stop it (aborting the blocking
    /// read task alone won't).
    pub(super) fn killer_handle(&self) -> Box<dyn ChildKiller + Send + Sync> {
        self.child.clone_killer()
    }
}

/// Spawn `command` through the platform shell in `cwd` under a PTY; `Err` only on
/// PTY/spawn failure.
#[cfg(unix)]
pub(super) fn spawn_pty_shell(command: &str, cwd: &std::path::Path) -> std::io::Result<PtyShell> {
    // Same shell selection as the agent's `run_bash`; `!cmd` is the user's own
    // command, so it runs bare (no sandbox wrapper).
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
    // We never forward stdin, so a pager-spawning command (`git diff`, `man`)
    // would see a tty, launch `less`, and block forever. Neutralize pagers and
    // interactive git credential prompts so output streams straight through.
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

/// Drive a [`PtyShell`] to completion: a reader thread streams lines while this
/// thread waits on the child and enforces [`SHELL_TIMEOUT`]; crossing the capture
/// caps kills the child and marks the run `truncated`. Runs under `spawn_blocking`.
///
/// The reader lives on its own thread so the controller can drop `master`, giving
/// a still-writing grandchild (e.g. `yes &`) EIO the instant the foreground child
/// exits instead of the read holding the pty open.
#[cfg(unix)]
pub(super) fn run_pty_to_completion(shell: PtyShell, tx: UnboundedSender<RuntimeEvent>) {
    let PtyShell {
        master,
        reader,
        mut child,
    } = shell;

    let truncated = Arc::new(AtomicBool::new(false));

    // The reader owns its own killer so it can stop the child the instant output
    // crosses the capture cap, without waiting for the controller below to notice.
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
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        for &byte in &buf[..n] {
                            // The PTY translates `\n` → `\r\n`; the `\r` is stripped as a
                            // control char downstream. Flush on newline, or when a
                            // newline-less run gets over-long so `acc` can't grow unbounded.
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
            if !acc.is_empty() {
                let _ = emit_pty_line(&tx, &acc, &mut lines, &mut bytes);
            }
        })
    };

    // Poll `try_wait` with a ramping interval — a blocking `wait()` was observed to
    // hang on a macOS PTY after a kill. On timeout, signal via a cloned killer; the
    // owned `child.kill()` would block while it escalates to SIGKILL.
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
            // The reader stopped draining at the cap. Close the PTY (drop(master)
            // below) so a child blocked writing (e.g. `yes`) gets EIO and dies,
            // instead of waiting out the full timeout on a stuck child.
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

    // Last live PTY handle (slave dropped at spawn, `master` never cloned): dropping
    // it EIOs any still-writing grandchild and lets the reader's `read()` EOF.
    drop(master);

    // Reap a child we stopped (cap/timeout) so it doesn't linger as a zombie.
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

    // A run we stopped (cap/timeout) reports -1 and folds timeout into `truncated`
    // so the render shows the truncation note, not a bogus `[exited -1]`.
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

/// Clean one raw output line for display: collapse cursor overwrites, strip
/// escapes, and cap over-long lines. Shared by the PTY and pipe readers.
fn clean_output_line(raw: &[u8]) -> String {
    let mut line = render_output_line(&String::from_utf8_lossy(raw));
    if line.chars().count() > MAX_LINE_CHARS {
        line = line.chars().take(MAX_LINE_CHARS).collect();
        line.push('…');
    }
    line
}

/// Emit one cleaned line and roll the counters; `true` once a capture cap is
/// crossed (caller should stop and kill).
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

// Windows `!cmd` uses plain pipes, not ConPTY: `portable-pty`'s ConPTY backend
// never delivers EOF on the output pipe when the child exits, so the blocking
// read hangs forever. Pipes close deterministically; the cost is block-buffered
// programs don't stream live (output appears when they exit).

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
    // Same shell selection as the agent's `run_bash`; `!cmd` is the user's own
    // command, so it runs bare (no sandbox wrapper).
    let invocation = crate::agent::sandbox::bare_shell(command);
    let mut cmd = Command::new(invocation.program.as_str());
    cmd.args(&invocation.args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // Neutralize pagers/credential prompts (matches the Unix PTY path's env).
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

/// Read one pipe to EOF, emitting a line event per newline; crossing the capture
/// caps flags `truncated`, kills the child, and stops.
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
            Ok(0) | Err(_) => break,
            Ok(n) => {
                for &byte in &buf[..n] {
                    // Flush on newline, or when a newline-less run gets over-long so
                    // `acc` can't grow unbounded; the trailing `\r` of a Windows `\r\n`
                    // is stripped downstream.
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

/// Windows counterpart of [`run_pty_to_completion`]: two reader threads stream
/// stdout/stderr while this thread waits on the child and enforces
/// [`SHELL_TIMEOUT`]. Runs under `spawn_blocking`.
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

    // `try_wait` is non-blocking, so the lock never contends with a reader's kill.
    // On timeout, killing the child closes the pipes so the readers EOF; a reader
    // hitting the cap kills the child itself.
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

    // Child exit closes the pipe write ends, so the readers EOF; joining drains the
    // final output. A backgrounded grandchild still holding the pipe can make the
    // join wait — same as the agent's `run_bash`; `esc` still abandons it.
    if let Some(handle) = out_thread {
        let _ = handle.join();
    }
    if let Some(handle) = err_thread {
        let _ = handle.join();
    }

    // Reap a child we stopped (cap/timeout).
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

    // A run we stopped (cap/timeout) reports -1 and folds timeout into `truncated`
    // for the render (see the PTY path).
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

/// A spawned `!cmd`, backed by a PTY on Unix and plain pipes on Windows.
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

/// Drive a spawned `!cmd` to completion; runs under `spawn_blocking`.
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
        "memory" => Ok(SlashCommand::Memory {
            dream: argument.as_deref() == Some("dream"),
        }),
        "effort" => Ok(SlashCommand::Effort(argument)),
        "create-skill" => Ok(SlashCommand::CreateSkill(argument)),
        "rewind" | "undo" | "unwind" => Ok(SlashCommand::Rewind),
        "config" => Ok(SlashCommand::Config),
        "compact" => Ok(SlashCommand::Compact {
            fast: argument.as_deref() == Some("fast"),
        }),
        "context" => Ok(SlashCommand::Context),
        "session" => Ok(SlashCommand::Session),
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

/// Read counterpart of [`clipboard_command_candidates`], tried in order.
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
