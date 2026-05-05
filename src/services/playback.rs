//! Cross-platform audio playback for `aivo speak`.
//!
//! Two paths:
//!
//! 1. `play_interactive` — primary. Decodes the file in-process with
//!    [`rodio`], opens a sink on the system default output device, and on a
//!    TTY drives a small key loop (SPACE pause / ←→ seek / q quit) while
//!    streaming a status line. Returns the position at exit so callers can
//!    persist a resume point.
//! 2. `play_external` — fallback. Hands the file to a per-OS CLI player
//!    (`afplay` / `paplay` / PowerShell `SoundPlayer`). Used when rodio
//!    can't decode the format (AAC, Opus on some builds) or no audio device
//!    is available. No pause / seek / position tracking — the function
//!    simply blocks until the child exits.
//!
use std::fs::File;
use std::io::{BufReader, IsTerminal, Write};
use std::path::Path;
#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use crossterm::cursor::MoveUp;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{Clear, ClearType, disable_raw_mode, enable_raw_mode};
use rodio::buffer::SamplesBuffer;
use rodio::{Decoder, OutputStream, Sink, Source};

use crate::services::audio_gen::{PCM_STREAM_CHANNELS, PCM_STREAM_RATE_HZ};

/// What happened during interactive playback. Callers use this to decide
/// whether to clear or persist the resume position.
#[derive(Debug, Clone)]
pub struct PlaybackOutcome {
    /// True if the audio reached its natural end (sink emptied). False if
    /// the user quit or playback was interrupted.
    pub completed: bool,
    /// Last known playback position. Zero on natural completion (so the
    /// next run starts from the beginning).
    pub last_pos: Duration,
}

/// Linux CLI players in priority order. Each entry is
/// `(binary, args_before_path)`. Used by the external fallback path.
#[cfg(target_os = "linux")]
const LINUX_PLAYERS: &[(&str, &[&str])] = &[
    ("paplay", &[]),
    ("pw-play", &[]),
    ("aplay", &["-q"]),
    ("ffplay", &["-autoexit", "-nodisp", "-loglevel", "quiet"]),
    ("mpv", &["--no-video", "--really-quiet"]),
    ("mpg123", &["-q"]),
];

/// Best-effort total-duration probe. Opens `path` with rodio's decoder
/// (no audio device touched) and reads `Source::total_duration`. Returns
/// `None` for formats whose decoder can't report a length (some MP3s
/// without Xing/VBRI headers, or formats rodio can't decode at all).
pub fn probe_duration(path: &Path) -> Option<Duration> {
    let file = File::open(path).ok()?;
    let decoder = Decoder::new(BufReader::new(file)).ok()?;
    decoder.total_duration()
}

/// Plays `path` with controllable in-process playback. Falls through to the
/// external player when rodio can't decode the file. `start_at` is the
/// initial seek offset (0 = beginning).
pub fn play_interactive(path: &Path, start_at: Duration) -> Result<PlaybackOutcome> {
    if !path.exists() {
        bail!("audio file '{}' does not exist", path.display());
    }
    match play_with_rodio(path, start_at) {
        Ok(outcome) => Ok(outcome),
        Err(e) => {
            eprintln!(
                "{} interactive playback unavailable ({e}); falling back to system player",
                crate::style::dim("note:")
            );
            play_external(path)?;
            Ok(PlaybackOutcome {
                completed: true,
                last_pos: Duration::ZERO,
            })
        }
    }
}

fn play_with_rodio(path: &Path, start_at: Duration) -> Result<PlaybackOutcome> {
    let file = File::open(path)
        .map_err(|e| anyhow::anyhow!("opening '{}' for playback: {e}", path.display()))?;
    let decoder = Decoder::new(BufReader::new(file))
        .map_err(|e| anyhow::anyhow!("rodio cannot decode '{}': {e}", path.display()))?;
    // Total duration is best-effort: many MP3s report None. We only use it
    // for the status line and seek clamping.
    let total = decoder.total_duration();

    let (_stream, handle) = OutputStream::try_default()
        .map_err(|e| anyhow::anyhow!("opening default audio output device: {e}"))?;
    let sink = Sink::try_new(&handle).map_err(|e| anyhow::anyhow!("creating audio sink: {e}"))?;
    sink.append(decoder);

    if !start_at.is_zero()
        && let Err(e) = sink.try_seek(start_at)
    {
        eprintln!(
            "{} seek to {:?} failed ({e}); starting from the beginning",
            crate::style::dim("note:"),
            start_at
        );
    }
    sink.play();

    if std::io::stderr().is_terminal() {
        run_interactive_loop(&sink, total, start_at)
    } else {
        sink.sleep_until_end();
        Ok(PlaybackOutcome {
            completed: true,
            last_pos: Duration::ZERO,
        })
    }
}

/// Drives the keyboard / status-line loop until the sink finishes or the
/// user quits. Owns the raw-mode lifecycle through `RawModeGuard`.
fn run_interactive_loop(
    sink: &Sink,
    total: Option<Duration>,
    start_at: Duration,
) -> Result<PlaybackOutcome> {
    let _raw = RawModeGuard::enter()?;
    print_help_line();

    let started = Instant::now();
    let mut last_render = Instant::now() - Duration::from_millis(500);
    let mut user_quit = false;

    loop {
        if sink.empty() {
            break;
        }
        // Poll with a short timeout so we re-render at ~4 Hz even when the
        // user isn't typing.
        if event::poll(Duration::from_millis(150))? {
            match event::read()? {
                Event::Key(KeyEvent {
                    code, modifiers, ..
                }) => match code {
                    KeyCode::Char(' ') => {
                        if sink.is_paused() {
                            sink.play();
                        } else {
                            sink.pause();
                        }
                    }
                    KeyCode::Left => {
                        let cur = current_pos(sink, started, start_at);
                        let target = cur.saturating_sub(Duration::from_secs(5));
                        let _ = sink.try_seek(target);
                    }
                    KeyCode::Right => {
                        let cur = current_pos(sink, started, start_at);
                        let target = cur + Duration::from_secs(5);
                        let target = match total {
                            Some(t) if target >= t => t.saturating_sub(Duration::from_millis(100)),
                            _ => target,
                        };
                        let _ = sink.try_seek(target);
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        user_quit = true;
                        break;
                    }
                    KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                        user_quit = true;
                        break;
                    }
                    _ => {}
                },
                Event::Resize(_, _) => {
                    last_render = Instant::now() - Duration::from_secs(1);
                }
                _ => {}
            }
        }
        if last_render.elapsed() >= Duration::from_millis(250) {
            render_status(sink, total);
            last_render = Instant::now();
        }
    }

    let last_pos = current_pos(sink, started, start_at);
    sink.stop();
    clear_playback_lines(2);

    Ok(PlaybackOutcome {
        completed: !user_quit,
        last_pos: if user_quit { last_pos } else { Duration::ZERO },
    })
}

/// Best-available current position. `Sink::get_pos` is accurate after seek;
/// `started + start_at` is the wall-clock fallback for the very first frame
/// before rodio publishes a position.
fn current_pos(sink: &Sink, started: Instant, start_at: Duration) -> Duration {
    let from_sink = sink.get_pos();
    if from_sink.is_zero() {
        start_at + started.elapsed()
    } else {
        from_sink
    }
}

fn print_help_line() {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(
        err,
        "{}",
        crate::style::dim("controls: SPACE pause · ←/→ seek 5s · q quit")
    );
}

fn render_status(sink: &Sink, total: Option<Duration>) {
    let pos = sink.get_pos();
    let state = if sink.is_paused() { "❙❙" } else { "▶" };
    let line = match total {
        Some(t) => format!(
            "  {state} {} / {}            ",
            fmt_clock(pos),
            fmt_clock(t),
        ),
        None => format!("  {state} {}            ", fmt_clock(pos)),
    };
    let mut err = std::io::stderr().lock();
    // \r returns to column 0; trailing spaces clear stale chars when the
    // line shrinks (e.g. paused → playing label change).
    let _ = write!(err, "\r{line}");
    let _ = err.flush();
}

fn clear_playback_lines(lines: u16) {
    if lines == 0 {
        return;
    }
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r");
    let _ = crossterm::execute!(err, Clear(ClearType::CurrentLine));
    for _ in 1..lines {
        let _ = crossterm::execute!(err, MoveUp(1), Clear(ClearType::CurrentLine));
    }
    let _ = write!(err, "\r");
    let _ = err.flush();
}

fn fmt_clock(d: Duration) -> String {
    let total = d.as_secs();
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}:{s:02}")
}

/// RAII guard that enables crossterm raw mode on entry and disables it on
/// drop. We *must* restore cooked mode on every exit path or the user's
/// terminal stays mute.
struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode().map_err(|e| anyhow::anyhow!("entering raw terminal mode: {e}"))?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

// ── Streaming playback (push raw PCM chunks as they arrive) ──────────────

/// Runs streaming playback on the current thread until the producer
/// drops `chunk_tx` (signalling end-of-stream) and the queue has played
/// out, or the user quits. Reads raw little-endian 16-bit PCM (mono,
/// 24 kHz) byte chunks from `chunk_rx`.
///
/// This entry point is called from a dedicated `std::thread` because
/// rodio's `OutputStream` is `!Send` on macOS — the audio device handle,
/// sink, and queue all stay on this thread for their entire lifetime.
///
/// On TTY: SPACE pauses, q/Esc/Ctrl-C quits, ←/→ are no-ops with a
/// one-line hint. On non-TTY: silent drain to completion.
pub fn run_streaming_playback(
    chunk_rx: std::sync::mpsc::Receiver<Vec<u8>>,
) -> Result<PlaybackOutcome> {
    let (_stream, handle) = OutputStream::try_default()
        .map_err(|e| anyhow::anyhow!("opening default audio output device: {e}"))?;
    let sink = Sink::try_new(&handle).map_err(|e| anyhow::anyhow!("creating audio sink: {e}"))?;
    // `keep_alive_if_empty = true` so a brief network under-run between
    // chunks doesn't end the sink — the queue plays silence instead.
    let (queue, queue_out) = rodio::queue::queue::<i16>(true);
    sink.append(queue_out);
    sink.play();

    let samples_pushed: u64 = 0;
    let samples_pushed = std::cell::Cell::new(samples_pushed);

    // Drain everything pending in chunk_rx and push to the queue. Returns
    // true if the channel disconnected (sender dropped → no more chunks).
    let drain_pending = || -> bool {
        loop {
            match chunk_rx.try_recv() {
                Ok(chunk) => {
                    let pushed = push_pcm_into_queue(&queue, &chunk);
                    samples_pushed.set(samples_pushed.get() + pushed);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return false,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return true,
            }
        }
    };

    let on_tty = std::io::stderr().is_terminal();
    if !on_tty {
        // Silent drain: pull chunks, wait for sender disconnect + caught up.
        let mut channel_disconnected = false;
        loop {
            if !channel_disconnected {
                channel_disconnected = drain_pending();
            }
            if channel_disconnected {
                let pushed_secs = samples_to_seconds(samples_pushed.get());
                if sink.get_pos().as_secs_f32() >= pushed_secs - 0.05 {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        sink.stop();
        return Ok(PlaybackOutcome {
            completed: true,
            last_pos: Duration::ZERO,
        });
    }

    let _raw = RawModeGuard::enter()?;
    print_streaming_help_line();

    let mut last_render = Instant::now() - Duration::from_millis(500);
    let mut user_quit = false;
    let mut channel_disconnected = false;

    loop {
        if !channel_disconnected {
            channel_disconnected = drain_pending();
        }
        if channel_disconnected {
            let pushed_secs = samples_to_seconds(samples_pushed.get());
            if sink.get_pos().as_secs_f32() >= pushed_secs - 0.05 {
                break;
            }
        }

        if event::poll(Duration::from_millis(100))?
            && let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = event::read()?
        {
            match code {
                KeyCode::Char(' ') => {
                    if sink.is_paused() {
                        sink.play();
                    } else {
                        sink.pause();
                    }
                }
                KeyCode::Left | KeyCode::Right => {
                    // Queue has discarded played samples; seek can't go
                    // backward and forward-seek would just skip silence.
                    print_streaming_seek_unavailable();
                    last_render = Instant::now() - Duration::from_secs(1);
                }
                KeyCode::Char('q') | KeyCode::Esc => {
                    user_quit = true;
                    break;
                }
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    user_quit = true;
                    break;
                }
                _ => {}
            }
        }
        if last_render.elapsed() >= Duration::from_millis(250) {
            render_streaming_status(&sink, samples_pushed.get(), channel_disconnected);
            last_render = Instant::now();
        }
    }

    let last_pos = sink.get_pos();
    sink.stop();
    clear_playback_lines(2);

    Ok(PlaybackOutcome {
        completed: !user_quit,
        last_pos: if user_quit { last_pos } else { Duration::ZERO },
    })
}

/// Returns the number of i16 samples pushed.
fn push_pcm_into_queue(queue: &Arc<rodio::queue::SourcesQueueInput<i16>>, chunk: &[u8]) -> u64 {
    let pair_count = chunk.len() / 2;
    if pair_count == 0 {
        return 0;
    }
    let mut samples = Vec::with_capacity(pair_count);
    for i in 0..pair_count {
        samples.push(i16::from_le_bytes([chunk[i * 2], chunk[i * 2 + 1]]));
    }
    let n = samples.len() as u64;
    queue.append(SamplesBuffer::new(
        PCM_STREAM_CHANNELS,
        PCM_STREAM_RATE_HZ,
        samples,
    ));
    n
}

fn samples_to_seconds(samples: u64) -> f32 {
    let per_sec = u64::from(PCM_STREAM_RATE_HZ) * u64::from(PCM_STREAM_CHANNELS);
    if per_sec == 0 {
        0.0
    } else {
        samples as f32 / per_sec as f32
    }
}

fn print_streaming_help_line() {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(
        err,
        "{}",
        crate::style::dim("controls: SPACE pause · q quit (seek disabled while streaming)")
    );
}

fn print_streaming_seek_unavailable() {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(
        err,
        "\r{}",
        crate::style::dim("seek unavailable while streaming — wait for cache then seek on replay")
    );
}

fn render_streaming_status(sink: &Sink, samples_pushed: u64, producer_done: bool) {
    let pos = sink.get_pos();
    let pushed_secs = samples_to_seconds(samples_pushed);
    let state = if sink.is_paused() { "❙❙" } else { "▶" };
    // Suffix shows whether the streamer is still feeding chunks. While
    // downloading the duration column ticks up; once done it's fixed.
    let stream_marker = if producer_done { "" } else { " · streaming" };
    let line = format!(
        "  {state} {} / {}{}            ",
        fmt_clock(pos),
        fmt_clock(Duration::from_secs_f32(pushed_secs)),
        stream_marker,
    );
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{line}");
    let _ = err.flush();
}

// ── External fallback path (per-OS CLI player) ────────────────────────────

fn play_external(path: &Path) -> Result<()> {
    play_external_impl(path)
}

#[cfg(target_os = "macos")]
fn play_external_impl(path: &Path) -> Result<()> {
    let status = Command::new("afplay").arg(path).status().map_err(|e| {
        anyhow::anyhow!(
            "failed to invoke `afplay`: {e} \
             (afplay is preinstalled on macOS — is your $PATH unusual?)"
        )
    })?;
    if !status.success() {
        bail!("afplay exited with status {}", status);
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn play_external_impl(path: &Path) -> Result<()> {
    // PowerShell single-quoted strings need internal `'` doubled.
    let path_str = path.to_string_lossy().replace('\'', "''");
    let script = format!("(New-Object System.Media.SoundPlayer '{path_str}').PlaySync()");
    let status = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .status()
        .map_err(|e| anyhow::anyhow!("failed to invoke `powershell`: {e}"))?;
    if !status.success() {
        bail!(
            "powershell SoundPlayer exited with status {} (is the WAV file valid?)",
            status
        );
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn play_external_impl(path: &Path) -> Result<()> {
    let mut not_found = Vec::new();
    for (binary, args) in LINUX_PLAYERS {
        let mut cmd = Command::new(binary);
        cmd.args(*args).arg(path);
        match cmd.status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => bail!("{binary} exited with status {status}"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                not_found.push(*binary);
                continue;
            }
            Err(e) => bail!("failed to invoke `{binary}`: {e}"),
        }
    }
    bail!(
        "no audio player found on PATH (tried: {}). \
         Install one (`apt install alsa-utils`, `brew install mpg123`, …) \
         or use `-o <path>` to save the audio to a file instead.",
        not_found.join(", ")
    );
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn play_external_impl(_path: &Path) -> Result<()> {
    bail!(
        "audio playback isn't supported on this platform; use `-o <path>` to save to a file instead"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_file() {
        let err = play_interactive(
            Path::new("/definitely/does/not/exist/aivo-test.wav"),
            Duration::ZERO,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not exist"), "got: {err}");
    }

    #[test]
    fn fmt_clock_renders_mm_ss() {
        assert_eq!(fmt_clock(Duration::from_secs(0)), "00:00");
        assert_eq!(fmt_clock(Duration::from_secs(5)), "00:05");
        assert_eq!(fmt_clock(Duration::from_secs(75)), "01:15");
        assert_eq!(fmt_clock(Duration::from_secs(3600)), "60:00");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_player_list_is_non_empty_and_distinct() {
        assert!(!LINUX_PLAYERS.is_empty());
        let names: Vec<_> = LINUX_PLAYERS.iter().map(|(n, _)| *n).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            names.len(),
            "duplicate binary in LINUX_PLAYERS"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_player_list_prefers_pulse_over_alsa() {
        let pulse_idx = LINUX_PLAYERS
            .iter()
            .position(|(n, _)| *n == "paplay")
            .expect("paplay should be in LINUX_PLAYERS");
        let alsa_idx = LINUX_PLAYERS
            .iter()
            .position(|(n, _)| *n == "aplay")
            .expect("aplay should be in LINUX_PLAYERS");
        assert!(pulse_idx < alsa_idx);
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_path_with_apostrophe_is_doubled_in_script() {
        let raw = r"C:\Users\bob's stuff\hi.wav";
        let escaped = raw.replace('\'', "''");
        assert_eq!(escaped, r"C:\Users\bob''s stuff\hi.wav");
    }
}
