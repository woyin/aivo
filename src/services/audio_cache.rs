//! Persistent on-disk cache for `aivo audio` TTS output.
//!
//! Cache files live under `<config_dir>/audio/<hash>.<ext>`. The hash is
//! derived from every input field that materially affects the generated
//! bytes — text, voice, model, format, speed — so a change in any one
//! produces a different cache entry. The leading `v1\n` lets us bump the
//! cache schema later without renaming existing files.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// File extension used for the JSON metadata sidecar that lives next to each
/// `<hash>.<ext>` audio file.
pub const SIDECAR_EXT: &str = "json";

/// Audio extensions we recognize when scanning the cache directory. Mirrors
/// the formats `audio.rs::default_extension` can emit.
const AUDIO_EXTS: &[&str] = &["mp3", "wav", "opus", "aac", "flac", "pcm"];

/// Persisted metadata for a cached TTS entry. Lives at
/// `<cache_dir>/<hash>.json` next to its audio file. Fields use defaults so
/// loading a sidecar written by an older aivo doesn't fail when we add new
/// keys (e.g., `last_pos_seconds`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    /// Up to 200 characters of the original prompt (NFC, trimmed). Never the
    /// full text — the prompt could be tens of KB.
    #[serde(default)]
    pub text_preview: String,
    /// Original prompt length in characters, before truncation for the preview.
    #[serde(default)]
    pub text_len: usize,
    /// Voice as requested by the user (empty when unset).
    #[serde(default)]
    pub voice: String,
    /// TTS model (e.g. "tts-1").
    #[serde(default)]
    pub model: String,
    /// Audio container format ("mp3", "wav", …).
    #[serde(default)]
    pub format: String,
    /// File extension actually written for the audio. Stored separately from
    /// `format` because the server's Content-Type can override the request.
    #[serde(default)]
    pub ext: String,
    /// Optional speed multiplier, encoded for human readability.
    #[serde(default)]
    pub speed: Option<f32>,
    /// When this entry was generated.
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    /// Size of the audio file in bytes.
    #[serde(default)]
    pub bytes: u64,
    /// Last interrupted playback position in seconds. Cleared back to 0 on
    /// natural completion. Floats survive serde round-trips losslessly enough
    /// for second-resolution resume.
    #[serde(default)]
    pub last_pos_seconds: f32,
    /// Total audio duration in seconds, probed once at generation time.
    /// `None` when the decoder couldn't report a duration (some MP3s
    /// without a Xing/VBRI header) or for legacy entries written before
    /// duration was recorded.
    #[serde(default)]
    pub duration_seconds: Option<f32>,
}

impl Sidecar {
    /// Builds a fresh sidecar from a TTS request and its result. `text` is
    /// truncated to the first 200 characters for the preview.
    pub fn new(
        text: &str,
        voice: Option<&str>,
        model: &str,
        format: Option<&str>,
        ext: &str,
        speed: Option<f32>,
        bytes: u64,
    ) -> Self {
        let trimmed = text.trim();
        let preview: String = trimmed.chars().take(200).collect();
        Self {
            text_preview: preview,
            text_len: trimmed.chars().count(),
            voice: voice.unwrap_or("").to_string(),
            model: model.to_string(),
            format: format.unwrap_or("").to_ascii_lowercase(),
            ext: ext.to_string(),
            speed,
            created_at: Utc::now(),
            bytes,
            last_pos_seconds: 0.0,
            duration_seconds: None,
        }
    }
}

/// Subdirectory of the aivo config dir that holds cached TTS files.
const AUDIO_CACHE_SUBDIR: &str = "audio";
/// Schema version baked into every hash. Bump to invalidate the cache.
const HASH_SCHEMA: &str = "v1";
/// Length of the hex-encoded cache hash. 16 hex chars = 64 bits, which is
/// far more than enough collision resistance for a single-user on-disk TTS
/// cache and keeps filenames readable.
const HASH_LEN: usize = 16;

/// Inputs that determine TTS output bytes. Two requests with equal
/// `CacheKey`s should produce byte-identical audio from the same provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    pub text: String,
    pub voice: String,
    pub model: String,
    pub format: String,
    pub speed: String,
}

impl CacheKey {
    /// Builds a cache key from the same fields used by `AudioRequest`.
    /// `Option`s collapse to empty strings so "voice unset" is one cache
    /// slot rather than scattered across providers' default-voice names.
    pub fn from_inputs(
        text: &str,
        voice: Option<&str>,
        model: &str,
        format: Option<&str>,
        speed: Option<f32>,
    ) -> Self {
        Self {
            text: text.trim().to_string(),
            voice: voice.unwrap_or("").to_string(),
            model: model.to_string(),
            format: format.unwrap_or("").to_ascii_lowercase(),
            speed: speed.map(format_speed).unwrap_or_default(),
        }
    }
}

/// Stable f32 → string conversion. `{:?}` produces a round-trippable form
/// (e.g. `1.0` not `1`), so two equal floats always hash to the same key.
fn format_speed(s: f32) -> String {
    format!("{s:?}")
}

/// `<config_dir>/audio/`.
pub fn audio_cache_dir(config_dir: &Path) -> PathBuf {
    config_dir.join(AUDIO_CACHE_SUBDIR)
}

/// `<config_dir>/audio/<hash>.<ext>`.
pub fn cache_path(cache_dir: &Path, key: &CacheKey, ext: &str) -> PathBuf {
    cache_dir.join(format!("{}.{}", hash_key(key), ext))
}

/// `<config_dir>/audio/<hash>.json` — the metadata sidecar path.
pub fn sidecar_path(cache_dir: &Path, hash: &str) -> PathBuf {
    cache_dir.join(format!("{hash}.{SIDECAR_EXT}"))
}

/// Writes a sidecar atomically: serialize → temp → rename. We use
/// `atomic_write::*` elsewhere in the codebase, but the dependency footprint
/// here is small enough that a local temp+rename keeps this module
/// dependency-light.
pub fn write_sidecar(cache_dir: &Path, hash: &str, sidecar: &Sidecar) -> Result<()> {
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;
    let final_path = sidecar_path(cache_dir, hash);
    let tmp_path = cache_dir.join(format!("{hash}.{SIDECAR_EXT}.tmp"));
    let json = serde_json::to_vec_pretty(sidecar).context("serializing sidecar")?;
    {
        let mut f = fs::File::create(&tmp_path)
            .with_context(|| format!("creating sidecar tmp {}", tmp_path.display()))?;
        f.write_all(&json)
            .with_context(|| format!("writing sidecar tmp {}", tmp_path.display()))?;
        f.sync_all().ok();
    }
    fs::rename(&tmp_path, &final_path).with_context(|| {
        format!(
            "renaming sidecar {} -> {}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}

/// Returns the sidecar for `hash`, or `None` if it's missing or unreadable.
/// We swallow read/parse errors — the audio file is the source of truth, the
/// sidecar is best-effort metadata.
pub fn read_sidecar(cache_dir: &Path, hash: &str) -> Option<Sidecar> {
    let path = sidecar_path(cache_dir, hash);
    let mut buf = String::new();
    fs::File::open(&path).ok()?.read_to_string(&mut buf).ok()?;
    serde_json::from_str(&buf).ok()
}

/// Read-modify-write for a sidecar. If no sidecar exists, `f` is called on a
/// freshly defaulted one. Use this for narrow updates like `last_pos_seconds`.
pub fn update_sidecar<F>(cache_dir: &Path, hash: &str, f: F) -> Result<Sidecar>
where
    F: FnOnce(&mut Sidecar),
{
    let mut current = read_sidecar(cache_dir, hash).unwrap_or_else(|| Sidecar {
        text_preview: String::new(),
        text_len: 0,
        voice: String::new(),
        model: String::new(),
        format: String::new(),
        ext: String::new(),
        speed: None,
        created_at: Utc::now(),
        bytes: 0,
        last_pos_seconds: 0.0,
        duration_seconds: None,
    });
    f(&mut current);
    write_sidecar(cache_dir, hash, &current)?;
    Ok(current)
}

/// One row of the audio cache listing.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// SHA-256 hash (the cache key, also the filename stem).
    pub hash: String,
    /// Path to the audio file on disk.
    pub audio_path: PathBuf,
    /// Sidecar metadata if available; `None` for legacy entries written
    /// before sidecars existed.
    pub sidecar: Option<Sidecar>,
    /// File mtime of the audio file. Used as the sort key for legacy entries
    /// without a sidecar.
    pub mtime: Option<SystemTime>,
}

/// Lists all cached audio entries, newest first. Entries with sidecars sort
/// by `created_at`; entries without sidecars sort by file mtime; ties broken
/// by hash for determinism. Missing cache dir → empty list (not an error).
pub fn list_entries(cache_dir: &Path) -> Result<Vec<CacheEntry>> {
    let mut entries: Vec<CacheEntry> = Vec::new();
    let read_dir = match fs::read_dir(cache_dir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(entries),
        Err(e) => {
            return Err(anyhow::anyhow!(
                "reading cache dir {}: {e}",
                cache_dir.display()
            ));
        }
    };
    for dirent in read_dir.flatten() {
        let path = dirent.path();
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
            continue;
        };
        let ext_lower = ext.to_ascii_lowercase();
        if !AUDIO_EXTS.contains(&ext_lower.as_str()) {
            continue;
        }
        // Filename stems are truncated SHA-256 hex (`HASH_LEN` chars). Skip
        // anything that doesn't look like one to avoid surfacing user files
        // dropped here.
        if stem.len() != HASH_LEN || !stem.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let mtime = dirent.metadata().and_then(|m| m.modified()).ok();
        let sidecar = read_sidecar(cache_dir, stem);
        entries.push(CacheEntry {
            hash: stem.to_string(),
            audio_path: path,
            sidecar,
            mtime,
        });
    }
    entries.sort_by(|a, b| {
        let key_a = a
            .sidecar
            .as_ref()
            .map(|s| s.created_at.into())
            .or(a.mtime)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let key_b = b
            .sidecar
            .as_ref()
            .map(|s| s.created_at.into())
            .or(b.mtime)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        key_b.cmp(&key_a).then(a.hash.cmp(&b.hash))
    });
    Ok(entries)
}

/// Removes the audio file and its sidecar. Both are removed best-effort —
/// missing files don't error; permission errors do. Returns the paths
/// actually removed for caller-side reporting.
pub fn delete_entry(cache_dir: &Path, hash: &str) -> Result<Vec<PathBuf>> {
    let mut removed = Vec::new();
    for ext in AUDIO_EXTS {
        let p = cache_dir.join(format!("{hash}.{ext}"));
        match fs::remove_file(&p) {
            Ok(()) => removed.push(p),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::anyhow!("removing {}: {e}", p.display()));
            }
        }
    }
    let sc = sidecar_path(cache_dir, hash);
    match fs::remove_file(&sc) {
        Ok(()) => removed.push(sc),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(anyhow::anyhow!("removing {}: {e}", sc.display())),
    }
    Ok(removed)
}

/// Hex-encoded SHA-256 over a versioned, newline-joined serialization of
/// the cache fields, truncated to `HASH_LEN` chars. Newlines inside `text`
/// are fine — the field order is fixed and every other field is a short
/// identifier without newlines, so the serialization is unambiguous.
pub fn hash_key(key: &CacheKey) -> String {
    let mut hasher = Sha256::new();
    hasher.update(HASH_SCHEMA.as_bytes());
    hasher.update(b"\n");
    hasher.update(key.text.as_bytes());
    hasher.update(b"\n");
    hasher.update(key.voice.as_bytes());
    hasher.update(b"\n");
    hasher.update(key.model.as_bytes());
    hasher.update(b"\n");
    hasher.update(key.format.as_bytes());
    hasher.update(b"\n");
    hasher.update(key.speed.as_bytes());
    let mut full = format!("{:x}", hasher.finalize());
    full.truncate(HASH_LEN);
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(text: &str) -> CacheKey {
        CacheKey::from_inputs(text, Some("nova"), "tts-1", Some("mp3"), Some(1.0))
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash_key(&key("hello")), hash_key(&key("hello")));
    }

    #[test]
    fn hash_trims_text() {
        assert_eq!(hash_key(&key("  hello  ")), hash_key(&key("hello")));
    }

    #[test]
    fn hash_changes_with_text() {
        assert_ne!(hash_key(&key("hello")), hash_key(&key("hello world")));
    }

    #[test]
    fn hash_changes_with_voice() {
        let a = CacheKey::from_inputs("hi", Some("nova"), "tts-1", None, None);
        let b = CacheKey::from_inputs("hi", Some("alloy"), "tts-1", None, None);
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn hash_changes_with_model() {
        let a = CacheKey::from_inputs("hi", None, "tts-1", None, None);
        let b = CacheKey::from_inputs("hi", None, "tts-1-hd", None, None);
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn hash_changes_with_format() {
        let a = CacheKey::from_inputs("hi", None, "tts-1", Some("mp3"), None);
        let b = CacheKey::from_inputs("hi", None, "tts-1", Some("wav"), None);
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn hash_changes_with_speed() {
        let a = CacheKey::from_inputs("hi", None, "tts-1", None, Some(1.0));
        let b = CacheKey::from_inputs("hi", None, "tts-1", None, Some(1.5));
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn format_is_case_insensitive() {
        let a = CacheKey::from_inputs("hi", None, "tts-1", Some("MP3"), None);
        let b = CacheKey::from_inputs("hi", None, "tts-1", Some("mp3"), None);
        assert_eq!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn unset_voice_and_default_voice_differ() {
        // An empty voice (None) is its own slot, distinct from any
        // explicitly-named voice.
        let a = CacheKey::from_inputs("hi", None, "tts-1", None, None);
        let b = CacheKey::from_inputs("hi", Some("alloy"), "tts-1", None, None);
        assert_ne!(hash_key(&a), hash_key(&b));
    }

    #[test]
    fn cache_path_joins_dir_and_filename() {
        let dir = Path::new("/tmp/aivo-test/audio");
        let key = key("hello");
        let path = cache_path(dir, &key, "mp3");
        assert_eq!(path.parent(), Some(dir));
        let file_name = path.file_name().and_then(|s| s.to_str()).unwrap();
        let hash = hash_key(&key);
        assert_eq!(file_name, format!("{hash}.mp3"));
    }

    #[test]
    fn audio_cache_dir_appends_subdir() {
        let dir = audio_cache_dir(Path::new("/tmp/aivo-test"));
        assert_eq!(dir, PathBuf::from("/tmp/aivo-test/audio"));
    }

    fn touch(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn sidecar_round_trip_preserves_fields() {
        let dir = tempfile::tempdir().unwrap();
        let s = Sidecar::new(
            "  hello world ",
            Some("alloy"),
            "tts-1",
            Some("MP3"),
            "mp3",
            Some(1.25),
            42,
        );
        write_sidecar(dir.path(), "abc", &s).unwrap();
        let got = read_sidecar(dir.path(), "abc").unwrap();
        assert_eq!(got.text_preview, "hello world");
        assert_eq!(got.text_len, 11);
        assert_eq!(got.voice, "alloy");
        assert_eq!(got.model, "tts-1");
        assert_eq!(got.format, "mp3");
        assert_eq!(got.ext, "mp3");
        assert_eq!(got.speed, Some(1.25));
        assert_eq!(got.bytes, 42);
        assert_eq!(got.last_pos_seconds, 0.0);
        assert_eq!(got.duration_seconds, None);
    }

    #[test]
    fn sidecar_persists_duration() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Sidecar::new("hi", None, "tts-1", None, "mp3", None, 1);
        s.duration_seconds = Some(11.5);
        write_sidecar(dir.path(), "abc", &s).unwrap();
        let got = read_sidecar(dir.path(), "abc").unwrap();
        assert_eq!(got.duration_seconds, Some(11.5));
    }

    #[test]
    fn sidecar_preview_is_capped_at_200_chars() {
        let long = "x".repeat(500);
        let s = Sidecar::new(&long, None, "tts-1", None, "mp3", None, 0);
        assert_eq!(s.text_preview.chars().count(), 200);
        assert_eq!(s.text_len, 500);
    }

    #[test]
    fn update_sidecar_modifies_position() {
        let dir = tempfile::tempdir().unwrap();
        let s = Sidecar::new("hi", None, "tts-1", None, "mp3", None, 1);
        write_sidecar(dir.path(), "abc", &s).unwrap();
        update_sidecar(dir.path(), "abc", |s| s.last_pos_seconds = 12.5).unwrap();
        assert_eq!(
            read_sidecar(dir.path(), "abc").unwrap().last_pos_seconds,
            12.5
        );
    }

    #[test]
    fn update_sidecar_creates_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        update_sidecar(dir.path(), "abc", |s| s.last_pos_seconds = 7.0).unwrap();
        assert_eq!(
            read_sidecar(dir.path(), "abc").unwrap().last_pos_seconds,
            7.0
        );
    }

    #[test]
    fn read_sidecar_returns_none_for_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_sidecar(dir.path(), "abc").is_none());
    }

    #[test]
    fn list_entries_skips_non_hex_filenames() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "a".repeat(HASH_LEN);
        touch(&dir.path().join(format!("{hash}.mp3")), b"audio");
        touch(&dir.path().join("not-a-hash.mp3"), b"x");
        touch(&dir.path().join("README.txt"), b"x");
        let entries = list_entries(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].hash, hash);
    }

    #[test]
    fn list_entries_returns_empty_for_missing_dir() {
        let dir = tempfile::tempdir().unwrap();
        let absent = dir.path().join("does-not-exist");
        let entries = list_entries(&absent).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn list_entries_sorts_newest_first_via_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let h1 = "a".repeat(HASH_LEN);
        let h2 = "b".repeat(HASH_LEN);
        touch(&dir.path().join(format!("{h1}.mp3")), b"x");
        touch(&dir.path().join(format!("{h2}.mp3")), b"x");
        let mut older = Sidecar::new("older", None, "tts-1", None, "mp3", None, 1);
        older.created_at = "2025-01-01T00:00:00Z".parse().unwrap();
        let mut newer = Sidecar::new("newer", None, "tts-1", None, "mp3", None, 1);
        newer.created_at = "2026-01-01T00:00:00Z".parse().unwrap();
        write_sidecar(dir.path(), &h1, &older).unwrap();
        write_sidecar(dir.path(), &h2, &newer).unwrap();
        let entries = list_entries(dir.path()).unwrap();
        assert_eq!(entries[0].hash, h2);
        assert_eq!(entries[1].hash, h1);
    }

    #[test]
    fn delete_entry_removes_audio_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "a".repeat(HASH_LEN);
        touch(&dir.path().join(format!("{hash}.mp3")), b"audio");
        let s = Sidecar::new("hi", None, "tts-1", None, "mp3", None, 5);
        write_sidecar(dir.path(), &hash, &s).unwrap();
        let removed = delete_entry(dir.path(), &hash).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(!dir.path().join(format!("{hash}.mp3")).exists());
        assert!(!dir.path().join(format!("{hash}.json")).exists());
    }

    #[test]
    fn delete_entry_is_idempotent_when_files_missing() {
        let dir = tempfile::tempdir().unwrap();
        let hash = "a".repeat(HASH_LEN);
        let removed = delete_entry(dir.path(), &hash).unwrap();
        assert!(removed.is_empty());
    }
}
