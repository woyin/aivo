//! `aivo speak` — generate speech (TTS) from a text prompt and play it.
//!
//! Resolves a key, takes the prompt from a positional arg / `--file` /
//! piped stdin, calls the provider, saves the result, and plays it. Every
//! invocation lands in the on-disk cache at `~/.config/aivo/audio/`,
//! keyed by `(prompt, voice, model, format, speed)`. Repeat calls with
//! identical inputs hit the cache and skip the provider entirely.

use std::fs;
use std::io::{IsTerminal, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde_json::json;
use tokio::task::JoinHandle;

use crate::cli::AudioArgs;
use crate::errors::ExitCode;
use crate::services::audio_cache::{self, CacheEntry, CacheKey, Sidecar};
use crate::services::audio_gen::{
    self, AudioArtifact, AudioRequest, PCM_STREAM_BITS, PCM_STREAM_CHANNELS, PCM_STREAM_RATE_HZ,
};
use crate::services::http_utils::router_http_client;
use crate::services::media_io::{self, OutputTarget, OverwritePolicy, human_bytes};
use crate::services::models_cache::ModelsCache;
use crate::services::playback::{self, PlaybackOutcome};
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;
use crate::tui::FuzzySelect;

pub struct AudioCommand {
    session_store: SessionStore,
    cache: ModelsCache,
}

impl AudioCommand {
    pub fn new(session_store: SessionStore, cache: ModelsCache) -> Self {
        Self {
            session_store,
            cache,
        }
    }

    pub fn print_help() {
        let name = "aivo speak";
        println!(
            "{} {}",
            style::cyan(name),
            style::dim("— speak a prompt aloud (TTS, cached, plays by default)")
        );
        println!();
        println!("{} {} [OPTIONS] [<PROMPT>]", style::bold("Usage:"), name);
        println!();
        println!("{}", style::bold("Arguments:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<24}", "PROMPT")),
            style::dim("Text to read aloud (or use -f <path>, or pipe via stdin)")
        );
        let opt = |f: &str, d: &str| {
            println!("  {}{}", style::cyan(format!("{:<24}", f)), style::dim(d));
        };
        let group = |label: &str| {
            println!();
            println!("{}", style::bold(label));
        };

        group("Input:");
        opt(
            "-f, --file [PATH]",
            "Read prompt from a file; omit PATH or pass - for stdin",
        );

        group("Voice & Model:");
        opt("-k, --key <ID|NAME>", "API key to use");
        opt("-m, --model <MODEL>", "TTS model (e.g. tts-1, tts-1-hd)");
        opt("    --voice <VOICE>", "alloy | nova | onyx | echo | …");
        opt("    --speed <SPEED>", "Synthesis speed, typically 0.25–4.0");

        group("Output:");
        opt(
            "-o, --output <PATH>",
            "File, directory, or template ({ts}/{model}); default: cache dir",
        );
        opt(
            "    --format <FORMAT>",
            "mp3 | wav | opus | aac | flac | pcm  (default: wav streaming, mp3 buffered)",
        );
        opt(
            "    --overwrite",
            "Bypass cache and overwrite -o without prompting",
        );

        group("Playback:");
        opt("    --no-play", "Save without playing");

        group("List:");
        opt("    --list", "Browse cached entries (replay or delete)");

        group("Other:");
        opt("-r, --refresh", "Bypass model-list cache");
        opt("    --json", "Emit JSON result (for scripting)");

        group("Playback controls (TTY):");
        let ctl = |s: &str| println!("  {}", style::dim(s));
        ctl("SPACE  pause / resume");
        ctl("← / →  seek 5s back / forward");
        ctl("q      quit");
        println!();
        println!("{}", style::bold("Examples:"));
        let ex = |s: &str| println!("  {}", style::dim(s));
        ex("aivo speak \"hello world\"");
        ex("aivo speak \"narration line\" -m tts-1-hd --voice nova");
        ex("aivo speak -f script.txt");
        ex("aivo speak -f -");
        ex("echo \"hi from pipe\" | aivo speak");
        ex("aivo speak \"...\" --no-play -o out.mp3   # save only");
        ex("aivo speak \"...\" --overwrite           # force regenerate");
        ex("aivo speak --list                       # browse cached entries");
    }

    /// Prints the audio-scope active key and model under the help output.
    /// Reads the audio-only `last_audio_selection` slot so it doesn't surface
    /// a chat key the user picked for `aivo chat`.
    pub async fn print_active_selection(session_store: &SessionStore) {
        let sel = session_store
            .get_last_audio_selection()
            .await
            .ok()
            .flatten();
        crate::commands::print_active_selection_for(session_store, sel).await;
    }

    pub async fn execute(self, args: AudioArgs, key: ApiKey, prompt: String) -> ExitCode {
        let provider_protocol = detect_provider_protocol(&key.base_url);
        if let Err(e) = validate_prompt_len(&prompt, provider_protocol) {
            eprintln!("{} {}", style::red("Error:"), e);
            return ExitCode::UserError;
        }

        let model = match resolve_audio_model(&self.session_store, &self.cache, &args, &key).await {
            Ok(Some(m)) => m,
            Ok(None) => return ExitCode::Success, // picker cancelled
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                return ExitCode::UserError;
            }
        };

        // Streaming path: when the user just runs `aivo speak "..."` with
        // no -o, no --format, and no --no-play, and the provider speaks the
        // OpenAI protocol — request raw PCM, push chunks to rodio as they
        // arrive, and wrap the accumulated PCM into WAV for the cache. The
        // cache slot uses format=wav so subsequent identical runs hit it.
        // Anything else (explicit format, output path, save-only, Gemini)
        // takes the existing buffered path.
        let streaming_eligible = !args.no_play
            && args.output.is_none()
            && args.format.is_none()
            && matches!(
                provider_protocol,
                ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
            );
        let effective_format: Option<String> = if streaming_eligible {
            Some("wav".to_string())
        } else {
            args.format.clone()
        };

        let cache_key = CacheKey::from_inputs(
            &prompt,
            args.voice.as_deref(),
            &model,
            effective_format.as_deref(),
            args.speed,
        );
        let ext = default_extension(effective_format.as_deref());
        let cache_dir = audio_cache::audio_cache_dir(self.session_store.config_dir());
        let cache_file = audio_cache::cache_path(&cache_dir, &cache_key, &ext);
        let cache_hash = audio_cache::hash_key(&cache_key);

        // Resolve the user-visible save path. None → save *is* the cache file.
        // Some(path) → save into the user's chosen path; the cache file is
        // still populated for future hits.
        let user_output_path: Option<PathBuf> = match args.output.as_deref() {
            None => None,
            Some(_) => {
                let target = OutputTarget::parse(args.output.as_deref());
                let initial = match media_io::resolve_output_path(&target, &model, &ext) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        return ExitCode::UserError;
                    }
                };
                let policy = OverwritePolicy::from_flags(args.overwrite, args.json);
                match crate::commands::resolve_final_path(&initial, policy, "--overwrite") {
                    Some(p) => Some(p),
                    None => return ExitCode::UserError,
                }
            }
        };

        // Try the cache first. ENOENT == cache miss; any other error is
        // surfaced (corrupt permissions, etc.). `--overwrite` skips the
        // lookup so the next branch always regenerates.
        let cache_metadata = if args.overwrite {
            None
        } else {
            match fs::metadata(&cache_file) {
                Ok(m) => Some(m),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
                Err(e) => {
                    eprintln!(
                        "{} cached file unreadable ({}): {}",
                        style::red("Error:"),
                        cache_file.display(),
                        e
                    );
                    return ExitCode::UserError;
                }
            }
        };
        let cache_hit = cache_metadata.is_some();

        let mut elapsed = std::time::Duration::ZERO;
        // Set when the streaming path already played the audio in-process.
        // Tells the post-cache playback section to skip its own play call.
        let mut streamed_outcome: Option<PlaybackOutcome> = None;
        let bytes = if let Some(meta) = cache_metadata {
            meta.len()
        } else {
            if let Some(parent) = cache_file.parent()
                && let Err(e) = fs::create_dir_all(parent)
            {
                eprintln!(
                    "{} cannot create cache dir '{}': {}",
                    style::red("Error:"),
                    parent.display(),
                    e
                );
                return ExitCode::UserError;
            }
            let request = AudioRequest {
                prompt: prompt.clone(),
                model: model.clone(),
                voice: args.voice.clone(),
                format: effective_format.clone(),
                speed: args.speed,
            };

            if streaming_eligible {
                let start = std::time::Instant::now();
                let result = stream_and_save(&key, &request, &cache_file).await;
                elapsed = start.elapsed();
                match result {
                    Ok((bytes, outcome)) => {
                        // Write sidecar with metadata + duration computed
                        // directly from PCM byte count (cheap and exact;
                        // probing the file would also work but is slower).
                        let pcm_bytes = bytes.saturating_sub(44); // strip WAV header
                        let bytes_per_sec = u64::from(PCM_STREAM_RATE_HZ)
                            * u64::from(PCM_STREAM_CHANNELS)
                            * u64::from(PCM_STREAM_BITS / 8);
                        let duration = if bytes_per_sec > 0 {
                            Some(pcm_bytes as f32 / bytes_per_sec as f32)
                        } else {
                            None
                        };
                        let mut sidecar = Sidecar::new(
                            &prompt,
                            args.voice.as_deref(),
                            &model,
                            effective_format.as_deref(),
                            &ext,
                            args.speed,
                            bytes,
                        );
                        sidecar.duration_seconds = duration;
                        if let Err(e) =
                            audio_cache::write_sidecar(&cache_dir, &cache_hash, &sidecar)
                        {
                            eprintln!(
                                "{} could not write audio metadata: {e}",
                                style::dim("note:")
                            );
                        }
                        streamed_outcome = Some(outcome);
                        bytes
                    }
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        return ExitCode::NetworkError;
                    }
                }
            } else {
                let spinner = start_spinner_if_tty(&model);
                let start = std::time::Instant::now();
                let result = audio_gen::generate(&key, &request, Some(&cache_file), true).await;
                elapsed = start.elapsed();
                stop_spinner(spinner);
                let bytes = match result {
                    Ok(a) => a.bytes,
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        return ExitCode::NetworkError;
                    }
                };
                // Write the metadata sidecar alongside the freshly-generated
                // audio. Failure here is non-fatal — the audio is the source
                // of truth; the sidecar just powers `--list`.
                let mut sidecar = Sidecar::new(
                    &prompt,
                    args.voice.as_deref(),
                    &model,
                    args.format.as_deref(),
                    &ext,
                    args.speed,
                    bytes,
                );
                sidecar.duration_seconds =
                    playback::probe_duration(&cache_file).map(|d| d.as_secs_f32());
                if let Err(e) = audio_cache::write_sidecar(&cache_dir, &cache_hash, &sidecar) {
                    eprintln!(
                        "{} could not write audio metadata: {e}",
                        style::dim("note:")
                    );
                }
                bytes
            }
        };

        let final_path = match user_output_path {
            Some(dest) if dest != cache_file => {
                if let Err(e) = fs::copy(&cache_file, &dest) {
                    eprintln!(
                        "{} cannot write '{}': {}",
                        style::red("Error:"),
                        dest.display(),
                        e
                    );
                    return ExitCode::UserError;
                }
                dest
            }
            Some(dest) => dest,
            None => cache_file.clone(),
        };
        let artifact = AudioArtifact {
            path: Some(final_path.clone()),
            bytes,
        };

        let mut played = false;
        let mut playback_error: Option<String> = None;
        if let Some(outcome) = streamed_outcome {
            // Streaming path already drove playback in-process. Just
            // record the outcome.
            played = true;
            persist_playback_outcome(&cache_dir, &cache_hash, &outcome);
        } else if !args.no_play {
            let start_at = Duration::ZERO;
            match playback::play_interactive(&final_path, start_at) {
                Ok(outcome) => {
                    played = true;
                    persist_playback_outcome(&cache_dir, &cache_hash, &outcome);
                }
                Err(e) => playback_error = Some(e.to_string()),
            }
        }

        let _ = self
            .session_store
            .set_last_audio_selection(&key, Some(&model))
            .await;

        if args.json {
            print_json(
                &artifact,
                &key,
                &model,
                args.voice.as_deref(),
                args.format.as_deref(),
                effective_format.as_deref(),
                &ext,
                args.speed,
                elapsed,
                played,
                cache_hit,
            );
        } else {
            print_human(
                &artifact,
                &key,
                &model,
                args.voice.as_deref(),
                played,
                cache_hit,
                playback_error.as_deref(),
            );
        }
        ExitCode::Success
    }

    /// `aivo speak --list` — fuzzy-pick a cached entry, then play or
    /// delete it. Pure local I/O; no key, no provider call.
    pub async fn run_list(self) -> ExitCode {
        let cache_dir = audio_cache::audio_cache_dir(self.session_store.config_dir());

        if !std::io::stderr().is_terminal() {
            // Non-TTY: print a one-shot listing for scripts.
            let entries = match audio_cache::list_entries(&cache_dir) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    return ExitCode::UserError;
                }
            };
            if entries.is_empty() {
                println!(
                    "{} no cached entries yet. Run `aivo speak \"…\"` first.",
                    style::dim("list:")
                );
                return ExitCode::Success;
            }
            for entry in &entries {
                println!("{}", format_entry_oneliner(entry));
            }
            return ExitCode::Success;
        }

        // Interactive: loop list → action → list. Exits only when the user
        // cancels (Esc/Ctrl-C) from the entry list itself, or when the
        // cache is empty (e.g. after deleting the last entry).
        let mut selected_hash: Option<String> = None;
        let mut selected_index = 0usize;
        loop {
            let entries = match audio_cache::list_entries(&cache_dir) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    return ExitCode::UserError;
                }
            };
            if entries.is_empty() {
                println!(
                    "{} no cached entries yet. Run `aivo speak \"…\"` first.",
                    style::dim("list:")
                );
                return ExitCode::Success;
            }

            let labels: Vec<String> = entries.iter().map(format_entry_oneliner).collect();
            let default = selected_hash
                .as_ref()
                .and_then(|hash| entries.iter().position(|entry| &entry.hash == hash))
                .unwrap_or_else(|| selected_index.min(entries.len().saturating_sub(1)));
            let picked = match FuzzySelect::new()
                .with_prompt("Cached TTS")
                .items(&labels)
                .default(default)
                .interact_opt()
            {
                Ok(Some(i)) => i,
                Ok(None) => return ExitCode::Success, // cancel from list = exit
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    return ExitCode::UserError;
                }
            };
            selected_hash = Some(entries[picked].hash.clone());
            selected_index = picked;
            let entry = &entries[picked];

            let actions = vec![
                "Play".to_string(),
                "Delete".to_string(),
                "Cancel".to_string(),
            ];
            let action = match FuzzySelect::new()
                .with_prompt("Action")
                .items(&actions)
                .interact_opt()
            {
                Ok(Some(i)) => i,
                // Cancel from action menu returns to the list, not exits.
                Ok(None) => continue,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    return ExitCode::UserError;
                }
            };
            match action {
                // Play / Delete errors are printed inline; we keep looping
                // so the user can pick another entry.
                0 => {
                    let _ = self.list_play(entry, &cache_dir);
                }
                1 if list_delete(entry, &cache_dir) == ExitCode::Success => {
                    selected_hash = None;
                }
                _ => {} // Cancel: fall through to next iteration
            }
        }
    }

    fn list_play(&self, entry: &CacheEntry, cache_dir: &Path) -> ExitCode {
        let start_at = Duration::ZERO;
        match playback::play_interactive(&entry.audio_path, start_at) {
            Ok(outcome) => {
                persist_playback_outcome(cache_dir, &entry.hash, &outcome);
                ExitCode::Success
            }
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }
}

/// Streams TTS audio from the provider while playing it through rodio's
/// queue, then wraps the accumulated PCM into WAV and saves to
/// `cache_file`. Returns `(file_size_bytes, outcome)`.
///
/// On user-quit mid-stream, the download task is aborted and no cache
/// file is written — the partial audio they heard is discarded. They can
/// re-run to get a complete cached version.
async fn stream_and_save(
    key: &ApiKey,
    request: &AudioRequest,
    cache_file: &Path,
) -> anyhow::Result<(u64, PlaybackOutcome)> {
    eprintln!("{} streaming…", style::dim(format!("{} ", request.model)));

    // Channel: download task → playback thread. Std mpsc, not tokio's,
    // because the consumer is a plain `std::thread` (rodio's OutputStream
    // is `!Send` on macOS, so the audio device + sink must stay on a
    // dedicated thread for their entire lifetime).
    let (chunk_tx, chunk_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let player_thread = std::thread::spawn(move || -> anyhow::Result<PlaybackOutcome> {
        playback::run_streaming_playback(chunk_rx)
    });

    // Drive the download on this (tokio) thread. The closure runs
    // synchronously between `await` points. `send` returns Err when the
    // player thread has dropped its receiver (e.g. the user pressed q),
    // and the closure's `false` return tells `stream_openai_pcm` to stop
    // requesting more bytes.
    let dl_result =
        audio_gen::stream_openai_pcm(key, request, |chunk| chunk_tx.send(chunk.to_vec()).is_ok())
            .await;
    drop(chunk_tx); // close channel so player drains and exits

    // Joining a `std::thread` blocks the OS thread; on tokio's
    // single-threaded runtime that would freeze the reactor. Move the
    // join onto the blocking pool. JoinHandle<T: Send> is itself Send.
    let outcome = tokio::task::spawn_blocking(move || {
        player_thread
            .join()
            .map_err(|_| anyhow::anyhow!("playback thread panicked"))?
    })
    .await??;

    let pcm = dl_result?;

    if !outcome.completed {
        // User quit mid-stream — discard the partial cache. The audio
        // they heard is lost; a re-run will restart streaming from scratch.
        return Ok((0, outcome));
    }
    let wav = audio_gen::wrap_pcm_as_wav(
        &pcm,
        PCM_STREAM_RATE_HZ,
        PCM_STREAM_CHANNELS,
        PCM_STREAM_BITS,
    );
    let bytes = wav.len() as u64;
    fs::write(cache_file, &wav)
        .map_err(|e| anyhow::anyhow!("writing cache file '{}': {e}", cache_file.display()))?;
    Ok((bytes, outcome))
}

fn list_delete(entry: &CacheEntry, cache_dir: &Path) -> ExitCode {
    match audio_cache::delete_entry(cache_dir, &entry.hash) {
        Ok(removed) => {
            for p in removed {
                println!("{} {}", style::dim("removed"), p.display());
            }
            ExitCode::Success
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            ExitCode::UserError
        }
    }
}

/// Hard ceiling on prompt length for OpenAI-protocol providers, in
/// Unicode scalar values. Matches the documented `input` cap on
/// `/v1/audio/speech`; sending more lands an opaque 400 or a truncated
/// read, which is exactly what we're trying to spare the user.
const MAX_PROMPT_CHARS_OPENAI: usize = 4096;

/// Hard ceiling on prompt length for Google's Gemini TTS, in Unicode
/// scalar values. Gemini's docs cite a 32k *token* context window for a
/// TTS session; English text averages ~4 chars/token, so 100_000 chars
/// stays well clear of that boundary while leaving room for non-English
/// scripts that tokenize denser. Past this point split the text.
const MAX_PROMPT_CHARS_GEMINI: usize = 100_000;

/// Selects a per-protocol prompt-length cap. Anthropic doesn't speak TTS
/// (gets rejected later with a clear message), so its cap is moot — we
/// borrow the OpenAI value for safety.
fn max_prompt_chars(protocol: ProviderProtocol) -> usize {
    match protocol {
        ProviderProtocol::Google => MAX_PROMPT_CHARS_GEMINI,
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi | ProviderProtocol::Anthropic => {
            MAX_PROMPT_CHARS_OPENAI
        }
    }
}

/// Returns an Err message when `text` exceeds the protocol-specific cap.
fn validate_prompt_len(text: &str, protocol: ProviderProtocol) -> anyhow::Result<()> {
    let max = max_prompt_chars(protocol);
    let len = text.chars().count();
    if len > max {
        anyhow::bail!(
            "prompt is {len} chars (max {max} for this provider); split the text into shorter chunks"
        );
    }
    Ok(())
}

/// Mirrors a `PlaybackOutcome` back into the sidecar. Completed playback
/// clears the legacy position field; a quit / interrupt stores the current
/// offset for metadata compatibility with older sidecars.
fn persist_playback_outcome(cache_dir: &Path, hash: &str, outcome: &PlaybackOutcome) {
    let new_pos = if outcome.completed {
        0.0
    } else {
        outcome.last_pos.as_secs_f32()
    };
    if let Err(e) = audio_cache::update_sidecar(cache_dir, hash, |s| {
        s.last_pos_seconds = new_pos;
    }) {
        eprintln!(
            "{} could not save playback position: {e}",
            style::dim("note:")
        );
    }
}

/// One-line summary for the list picker. Format:
/// `MM-DD HH:MM M:SS [voice[·1.5x]] "preview…"` — date and duration sit
/// adjacent (single space) so the eye lands on the preview quickly. The
/// optional discriminator (voice and non-default speed) appears only
/// when set, because most invocations use the default voice and speed.
/// Model is intentionally omitted — the cache already disambiguates by
/// hash, and showing model on every row adds noise for the common case.
fn format_entry_oneliner(entry: &CacheEntry) -> String {
    let when = entry
        .sidecar
        .as_ref()
        .map(|s| format_list_time(s.created_at))
        .unwrap_or_else(|| {
            entry
                .mtime
                .map(format_system_time_list)
                .unwrap_or_else(|| "??-?? ??:??".to_string())
        });
    let duration = entry
        .sidecar
        .as_ref()
        .and_then(|s| s.duration_seconds)
        .map(|secs| fmt_duration_short(Duration::from_secs_f32(secs)))
        .unwrap_or_else(|| "?:??".to_string());

    let discriminator = entry
        .sidecar
        .as_ref()
        .map(format_discriminator)
        .filter(|s| !s.is_empty())
        .map(|d| format!(" {d}"))
        .unwrap_or_default();

    let preview: String = entry
        .sidecar
        .as_ref()
        .map(|s| {
            // Collapse newlines so each entry stays on one line.
            let one_line = s.text_preview.replace(['\n', '\r'], " ");
            one_line.chars().take(80).collect::<String>()
        })
        .unwrap_or_else(|| {
            format!(
                "(no metadata · {})",
                entry.hash.chars().take(10).collect::<String>()
            )
        });
    format!("{when} {duration}{discriminator}  \"{preview}\"")
}

fn format_list_time(ts: chrono::DateTime<chrono::Utc>) -> String {
    ts.with_timezone(&chrono::Local)
        .format("%m-%d %H:%M")
        .to_string()
}

fn format_system_time_list(ts: std::time::SystemTime) -> String {
    let local: chrono::DateTime<chrono::Local> = ts.into();
    local.format("%m-%d %H:%M").to_string()
}

/// `voice[·SPEEDx]` — optional middle column. Empty when the entry was
/// generated with the default voice and default speed (the common case),
/// so most rows skip it entirely.
fn format_discriminator(s: &Sidecar) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !s.voice.is_empty() {
        parts.push(s.voice.clone());
    }
    if let Some(sp) = s.speed
        && (sp - 1.0).abs() > f32::EPSILON
    {
        parts.push(format!("{sp}x"));
    }
    parts.join(" · ")
}

/// `M:SS` for clips under an hour, `H:MM:SS` for the rare longer one. Used
/// for compact display in the list picker, distinct from the playback
/// status line's `MM:SS` (which is fixed-width and zero-padded).
fn fmt_duration_short(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

async fn resolve_audio_model(
    session_store: &SessionStore,
    cache: &ModelsCache,
    args: &AudioArgs,
    key: &ApiKey,
) -> anyhow::Result<Option<String>> {
    match &args.model {
        Some(m) if !m.is_empty() => {
            let resolved = session_store.resolve_alias(m).await.unwrap_or(m.clone());
            Ok(Some(resolved))
        }
        Some(_) => pick_audio_model_interactively(cache, key, args.refresh).await,
        None => {
            if let Ok(Some(sel)) = session_store.get_last_audio_selection().await
                && sel.key_id == key.id
                && let Some(model) = sel.model
                && !model.is_empty()
            {
                return Ok(Some(model));
            }
            pick_audio_model_interactively(cache, key, args.refresh).await
        }
    }
}

/// Opens a model picker over the provider's full model list. Same approach
/// as the image picker: don't filter heuristically, let the provider error
/// be the signal — TTS model naming varies wildly across providers.
async fn pick_audio_model_interactively(
    cache: &ModelsCache,
    key: &ApiKey,
    refresh: bool,
) -> anyhow::Result<Option<String>> {
    if !std::io::stderr().is_terminal() {
        anyhow::bail!(
            "no audio model specified and no terminal available; pass -m <name> (e.g. tts-1)"
        );
    }

    let client = router_http_client();
    let all_models = crate::commands::models::fetch_all_models_cached(&client, key, cache, refresh)
        .await
        .unwrap_or_default();

    if all_models.is_empty() {
        anyhow::bail!(
            "could not fetch a model list for this key; pass -m <name> explicitly (e.g. tts-1, tts-1-hd)"
        );
    }

    Ok(crate::commands::models::prompt_model_picker(
        all_models,
        None,
        Vec::new(),
        "Select model",
    ))
}

/// Maps the user-requested format flag to a default file extension. When
/// `--format` is missing, MP3 is used (matches OpenAI's server-side default).
fn default_extension(format: Option<&str>) -> String {
    match format.map(str::to_ascii_lowercase).as_deref() {
        Some("wav") => "wav".into(),
        Some("opus") => "opus".into(),
        Some("aac") => "aac".into(),
        Some("flac") => "flac".into(),
        Some("pcm") => "pcm".into(),
        // mp3, anything unknown, or missing → .mp3 default
        _ => "mp3".into(),
    }
}

fn start_spinner_if_tty(model: &str) -> Option<(Arc<AtomicBool>, JoinHandle<()>)> {
    if std::io::stderr().is_terminal() {
        let label = format!(" Speaking with {}…", model);
        Some(style::start_spinner(Some(&label)))
    } else {
        None
    }
}

fn stop_spinner(spinner: Option<(Arc<AtomicBool>, JoinHandle<()>)>) {
    if let Some((flag, _handle)) = spinner {
        style::stop_spinner(&flag);
    }
}

fn print_human(
    artifact: &AudioArtifact,
    key: &ApiKey,
    model: &str,
    voice: Option<&str>,
    played: bool,
    cached: bool,
    playback_error: Option<&str>,
) {
    let Some(path) = &artifact.path else {
        return;
    };
    let mut tags: Vec<String> = Vec::new();
    if cached {
        tags.push(style::dim("cached").to_string());
    }
    if played {
        tags.push(style::dim("played").to_string());
    } else if let Some(err) = playback_error {
        tags.push(format!("{}: {}", style::yellow("playback skipped"), err));
    }
    let suffix = if tags.is_empty() {
        String::new()
    } else {
        format!(" ({})", tags.join(", "))
    };
    println!(
        "{} saved {} ({}, {}) via {}/{}{}",
        style::success_symbol(),
        style::cyan(path.display().to_string()),
        voice.unwrap_or("default voice"),
        human_bytes(artifact.bytes),
        style::dim(key.display_name()),
        style::dim(model),
        suffix,
    );
}

#[allow(clippy::too_many_arguments)]
fn print_json(
    artifact: &AudioArtifact,
    key: &ApiKey,
    model: &str,
    voice: Option<&str>,
    format_requested: Option<&str>,
    format_effective: Option<&str>,
    ext: &str,
    speed: Option<f32>,
    elapsed: std::time::Duration,
    played: bool,
    cached: bool,
) {
    let path = artifact.path.as_ref().map(|p| p.display().to_string());
    // `format_requested` is the user's --format flag (or null);
    // `format_effective` is what we actually sent to the provider — they
    // diverge on the streaming path (request becomes pcm, file becomes
    // wav). `ext` is the on-disk extension and is the most useful field
    // for downstream tooling that needs to open the file.
    let out = json!({
        "model": model,
        "key": key.display_name(),
        "voice": voice,
        "format_requested": format_requested,
        "format_effective": format_effective,
        "ext": ext,
        "speed": speed,
        "duration_ms": elapsed.as_millis() as u64,
        "path": path,
        "bytes": artifact.bytes,
        "played": played,
        "cached": cached,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
}

/// Reads prompt text from a file path. Trims trailing whitespace; rejects
/// empty/whitespace-only files. Errors include the path for triage.
#[allow(dead_code)] // used by the binary's main.rs; lib build doesn't see it
pub fn read_prompt_file(path: &Path) -> anyhow::Result<String> {
    let raw = fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read --file '{}': {}", path.display(), e))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("--file '{}' is empty", path.display());
    }
    Ok(trimmed.to_string())
}

#[allow(dead_code)] // used by the binary's main.rs; lib build doesn't see it
pub fn is_stdin_file_arg(path: &str) -> bool {
    path == "-"
}

#[allow(dead_code)] // used by the binary's main.rs; lib build doesn't see it
pub fn read_prompt_stdin_explicit() -> anyhow::Result<String> {
    if std::io::stdin().is_terminal() {
        eprintln!("{}", style::dim("Enter prompt, then press Ctrl-D to send."));
    }

    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .map_err(|e| anyhow::anyhow!("cannot read stdin for --file: {e}"))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        anyhow::bail!("stdin for --file is empty");
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_extension_picks_mp3_for_default_or_unknown() {
        assert_eq!(default_extension(None), "mp3");
        assert_eq!(default_extension(Some("mp3")), "mp3");
        assert_eq!(default_extension(Some("garbage")), "mp3");
    }

    #[test]
    fn default_extension_picks_explicit_formats() {
        assert_eq!(default_extension(Some("wav")), "wav");
        assert_eq!(default_extension(Some("WAV")), "wav");
        assert_eq!(default_extension(Some("opus")), "opus");
        assert_eq!(default_extension(Some("aac")), "aac");
        assert_eq!(default_extension(Some("flac")), "flac");
        assert_eq!(default_extension(Some("pcm")), "pcm");
    }

    #[test]
    fn read_prompt_file_returns_trimmed_contents() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompt.txt");
        fs::write(&path, "  hello world  \n").unwrap();
        let out = read_prompt_file(&path).unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn read_prompt_file_rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        fs::write(&path, "   \n\t\n").unwrap();
        assert!(read_prompt_file(&path).is_err());
    }

    #[test]
    fn read_prompt_file_reports_missing_path() {
        let err = read_prompt_file(Path::new("/nonexistent/aivo-test.txt")).unwrap_err();
        assert!(err.to_string().contains("--file"));
    }

    #[test]
    fn stdin_file_arg_is_dash_only() {
        assert!(is_stdin_file_arg("-"));
        assert!(!is_stdin_file_arg("prompt.txt"));
        assert!(!is_stdin_file_arg(""));
    }

    #[test]
    fn validate_prompt_len_accepts_at_openai_limit() {
        assert!(validate_prompt_len("hello", ProviderProtocol::Openai).is_ok());
        assert!(
            validate_prompt_len(
                &"x".repeat(MAX_PROMPT_CHARS_OPENAI),
                ProviderProtocol::Openai,
            )
            .is_ok()
        );
    }

    #[test]
    fn validate_prompt_len_rejects_over_openai_limit() {
        let over = "x".repeat(MAX_PROMPT_CHARS_OPENAI + 1);
        let err = validate_prompt_len(&over, ProviderProtocol::Openai).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(&(MAX_PROMPT_CHARS_OPENAI + 1).to_string()),
            "got: {msg}"
        );
        assert!(
            msg.contains(&MAX_PROMPT_CHARS_OPENAI.to_string()),
            "got: {msg}"
        );
        assert!(msg.contains("split"), "got: {msg}");
    }

    #[test]
    fn validate_prompt_len_gemini_accepts_past_openai_cap() {
        // 10k chars rejects on OpenAI, accepts on Gemini's wider window.
        let s = "x".repeat(10_000);
        assert!(validate_prompt_len(&s, ProviderProtocol::Openai).is_err());
        assert!(validate_prompt_len(&s, ProviderProtocol::Google).is_ok());
    }

    #[test]
    fn validate_prompt_len_gemini_rejects_past_its_own_cap() {
        let over = "x".repeat(MAX_PROMPT_CHARS_GEMINI + 1);
        assert!(validate_prompt_len(&over, ProviderProtocol::Google).is_err());
    }

    #[test]
    fn validate_prompt_len_counts_unicode_scalars() {
        // Multi-byte emoji must be counted as a single scalar so a Japanese
        // or emoji-heavy prompt isn't rejected based on byte length.
        let scalars = MAX_PROMPT_CHARS_OPENAI;
        let s = "🎵".repeat(scalars); // each emoji = 1 scalar, 4 bytes
        assert!(validate_prompt_len(&s, ProviderProtocol::Openai).is_ok());
        let s_over = "🎵".repeat(scalars + 1);
        assert!(validate_prompt_len(&s_over, ProviderProtocol::Openai).is_err());
    }

    #[test]
    fn fmt_duration_short_drops_leading_zeros_under_an_hour() {
        assert_eq!(fmt_duration_short(Duration::from_secs(0)), "0:00");
        assert_eq!(fmt_duration_short(Duration::from_secs(9)), "0:09");
        assert_eq!(fmt_duration_short(Duration::from_secs(75)), "1:15");
        assert_eq!(fmt_duration_short(Duration::from_secs(599)), "9:59");
    }

    #[test]
    fn fmt_duration_short_uses_hours_when_needed() {
        assert_eq!(fmt_duration_short(Duration::from_secs(3600)), "1:00:00");
        assert_eq!(fmt_duration_short(Duration::from_secs(3661)), "1:01:01");
    }

    #[test]
    fn format_entry_oneliner_shows_date_duration_and_preview() {
        use crate::services::audio_cache::Sidecar;
        let mut s = Sidecar::new(
            "the quick brown fox jumps over the lazy dog",
            None,
            "openai/gpt-4o-mini-tts",
            Some("wav"),
            "wav",
            None,
            1234,
        );
        s.created_at = "2026-05-04T15:22:00Z".parse().unwrap();
        s.duration_seconds = Some(11.0);
        let entry = CacheEntry {
            hash: "a".repeat(64),
            audio_path: PathBuf::from("/tmp/fake.wav"),
            sidecar: Some(s),
            mtime: None,
        };
        let line = format_entry_oneliner(&entry);
        let expected_when = format_list_time("2026-05-04T15:22:00Z".parse().unwrap());
        assert!(line.contains(&expected_when), "got: {line}");
        assert!(line.contains("0:11"), "got: {line}");
        assert!(line.contains("the quick brown fox"), "got: {line}");
        assert!(
            !line.contains("gpt-4o-mini"),
            "model should be hidden: {line}"
        );
        // Single space between date and duration (the "too big" gap fix).
        assert!(
            line.contains(&format!("{expected_when} 0:11")),
            "got: {line}"
        );
    }

    #[test]
    fn format_entry_oneliner_includes_voice_when_set() {
        use crate::services::audio_cache::Sidecar;
        let mut s = Sidecar::new("hi", Some("alloy"), "tts-1", None, "wav", None, 1);
        s.created_at = "2026-05-04T15:22:00Z".parse().unwrap();
        s.duration_seconds = Some(2.0);
        let entry = CacheEntry {
            hash: "a".repeat(64),
            audio_path: PathBuf::from("/tmp/fake.wav"),
            sidecar: Some(s),
            mtime: None,
        };
        let line = format_entry_oneliner(&entry);
        assert!(line.contains("alloy"), "voice should appear: {line}");
    }

    #[test]
    fn format_discriminator_empty_when_default_voice_and_speed() {
        use crate::services::audio_cache::Sidecar;
        let s = Sidecar::new("hi", None, "tts-1", None, "mp3", Some(1.0), 1);
        assert_eq!(format_discriminator(&s), "");
    }

    #[test]
    fn format_discriminator_voice_only() {
        use crate::services::audio_cache::Sidecar;
        let s = Sidecar::new("hi", Some("nova"), "tts-1", None, "mp3", None, 1);
        assert_eq!(format_discriminator(&s), "nova");
    }

    #[test]
    fn format_discriminator_voice_and_non_default_speed() {
        use crate::services::audio_cache::Sidecar;
        let s = Sidecar::new("hi", Some("nova"), "tts-1", None, "mp3", Some(1.5), 1);
        assert_eq!(format_discriminator(&s), "nova · 1.5x");
    }

    #[test]
    fn format_discriminator_speed_only() {
        use crate::services::audio_cache::Sidecar;
        // No voice but non-default speed: only the speed shows.
        let s = Sidecar::new("hi", None, "tts-1", None, "mp3", Some(0.75), 1);
        assert_eq!(format_discriminator(&s), "0.75x");
    }

    #[test]
    fn format_entry_oneliner_falls_back_for_missing_duration() {
        use crate::services::audio_cache::Sidecar;
        let mut s = Sidecar::new("hello", None, "tts-1", None, "mp3", None, 1);
        s.created_at = "2026-05-04T15:22:00Z".parse().unwrap();
        // duration_seconds left as None
        let entry = CacheEntry {
            hash: "a".repeat(64),
            audio_path: PathBuf::from("/tmp/fake.mp3"),
            sidecar: Some(s),
            mtime: None,
        };
        let line = format_entry_oneliner(&entry);
        assert!(line.contains("?:??"), "got: {line}");
    }
}
