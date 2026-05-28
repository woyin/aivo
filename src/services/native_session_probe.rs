//! Snapshot+diff to link an `aivo run` invocation to the native session
//! file the launched CLI produces. Each `[run]` event recorded by
//! `ai_launcher` carries the aivo-side metadata (key, base_url, exit code,
//! duration); this module fills in the missing piece — the session id of
//! the conversation file the underlying CLI just wrote — so the unified
//! `aivo logs` view can join the two.
//!
//! Performance discipline: the snapshot runs **before every spawn**, so
//! the launcher's hot path. Two rules keep it cheap on a heavy user's
//! machine (which has thousands of historical session files):
//!   1. Scope by cwd when the layout encodes it — claude/gemini/pi keep one
//!      subdir per project, so we walk that subdir only. Codex partitions
//!      by `YYYY/MM/DD`, so we walk today + yesterday.
//!   2. Filter by mtime: we only care about files touched within a small
//!      pre-launch window (`PRE_LAUNCH_SLACK`). Anything older can never be
//!      "the new session" so it doesn't need to enter the snapshot set.
//!
//! The detect phase reuses the same scoping and only opens file content
//! for tools whose id isn't in the filename (gemini → one open total).
//!
//! Layout per CLI is mirrored from `context_ingest`:
//!   claude   ~/.claude/projects/<encoded-cwd>/<uuid>.jsonl
//!   codex    ~/.codex/sessions/YYYY/MM/DD/rollout-...-<uuid>.jsonl
//!   gemini   ~/.gemini/tmp/<sha256-cwd>/chats/session-*.json{,l}
//!   pi       ~/.pi/agent/sessions/<encoded-cwd>/<ts>_<uuid>.jsonl
//!   opencode ~/.local/share/opencode/opencode.db (sqlite `session` table)
//!   amp      ~/.config/aivo/amp-threads/T-<id>.json (aivo's bridge cache)

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use chrono::{DateTime, Utc};
use tokio::fs;

use crate::services::ai_launcher::AIToolType;
use crate::services::context_ingest::{
    encode_claude_dir, gemini_matching_session_files_in, normalize_gemini_session, pi_session_dir,
};
use crate::services::system_env;

/// Window before `launch_start` that we still consider "concurrent" for
/// snapshot purposes. Files touched in this window can't be attributed to
/// the new launch, so they go into the exclusion set. Generous enough to
/// cover a clock jitter or a session another aivo run wrote a moment ago.
const PRE_LAUNCH_SLACK: Duration = Duration::from_secs(5);

/// Snapshot of the native session ids that already existed (within the
/// recent window) at the moment a launch begins. `detect_new` finds what
/// appeared after the launch ran.
pub struct SessionProbe {
    tool: AIToolType,
    cwd: Option<String>,
    launch_start: SystemTime,
    /// Ids of files touched within `PRE_LAUNCH_SLACK` of `launch_start`.
    /// Almost always empty; populated only when the user happens to have
    /// recent activity in the same CLI's session dir.
    recent_before: HashSet<String>,
}

impl SessionProbe {
    /// Capture the pre-launch state. Cheap: only files inside the recent
    /// window are stat'd; for layouts that key by cwd, only that one
    /// subdir is read.
    pub async fn snapshot(tool: AIToolType, cwd: Option<&str>) -> Self {
        let launch_start = SystemTime::now();
        let canonical_cwd = cwd.map(canonicalize_cwd);
        let recent_before = recent_session_ids(
            tool,
            canonical_cwd.as_deref(),
            launch_start - PRE_LAUNCH_SLACK,
        )
        .await;
        Self {
            tool,
            cwd: canonical_cwd,
            launch_start,
            recent_before,
        }
    }

    /// Identify the session the launched CLI just produced. Returns the
    /// id of the most-recently-modified session file with `mtime >
    /// launch_start - slack` that wasn't in the pre-launch snapshot.
    pub async fn detect_new(&self) -> Option<String> {
        let cutoff = self.launch_start - PRE_LAUNCH_SLACK;
        let entries = recent_session_entries(self.tool, self.cwd.as_deref(), cutoff).await;
        entries
            .into_iter()
            .filter(|(_, id)| !self.recent_before.contains(id))
            .max_by_key(|(mtime, _)| *mtime)
            .map(|(_, id)| id)
    }
}

fn canonicalize_cwd(cwd: &str) -> String {
    std::fs::canonicalize(cwd)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| cwd.to_string())
}

/// Snapshot variant: ids only, no mtime, no file-content reads. Used for
/// the exclusion set built before spawn.
async fn recent_session_ids(
    tool: AIToolType,
    cwd: Option<&str>,
    cutoff: SystemTime,
) -> HashSet<String> {
    recent_session_entries(tool, cwd, cutoff)
        .await
        .into_iter()
        .map(|(_, id)| id)
        .collect()
}

/// Per-tool listing returning `(mtime, id)` for files with `mtime >=
/// cutoff`. Scoped by cwd where the layout supports it.
async fn recent_session_entries(
    tool: AIToolType,
    cwd: Option<&str>,
    cutoff: SystemTime,
) -> Vec<(SystemTime, String)> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    match tool {
        AIToolType::Claude => claude_entries(&home, cwd, cutoff).await,
        AIToolType::Codex | AIToolType::CodexApp => codex_entries(&home, cutoff).await,
        AIToolType::Gemini => gemini_entries(&home, cwd, cutoff).await,
        AIToolType::Pi => pi_entries(&home, cwd, cutoff).await,
        AIToolType::Opencode => {
            opencode_entries(
                &home
                    .join(".local")
                    .join("share")
                    .join("opencode")
                    .join("opencode.db"),
                cutoff,
            )
            .await
        }
        AIToolType::Amp => {
            amp_entries(&crate::services::amp_threads::default_threads_dir(), cutoff).await
        }
    }
}

// ---------------------------------------------------------------------------
// Claude — id is the .jsonl filename stem; scope by cwd subdir
// ---------------------------------------------------------------------------

async fn claude_entries(
    home: &Path,
    cwd: Option<&str>,
    cutoff: SystemTime,
) -> Vec<(SystemTime, String)> {
    let projects_root = home.join(".claude").join("projects");
    let dirs: Vec<PathBuf> = match cwd {
        Some(c) => vec![projects_root.join(encode_claude_dir(c))],
        None => collect_subdirs(&projects_root).await,
    };
    let mut out = Vec::new();
    for dir in dirs {
        scan_jsonl_dir(&dir, cutoff, &mut out, |p| {
            p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        })
        .await;
    }
    out
}

// ---------------------------------------------------------------------------
// Codex — id is the trailing UUID in `rollout-<ts>-<uuid>.jsonl`; scope by
// today's / yesterday's date partition (covers cross-midnight launches)
// ---------------------------------------------------------------------------

async fn codex_entries(home: &Path, cutoff: SystemTime) -> Vec<(SystemTime, String)> {
    let sessions_root = home.join(".codex").join("sessions");
    let mut out = Vec::new();
    for date_dir in recent_date_partitions(&sessions_root, cutoff) {
        scan_jsonl_dir(&date_dir, cutoff, &mut out, |p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(parse_codex_filename_id)
        })
        .await;
    }
    out
}

/// `~/.codex/sessions/YYYY/MM/DD/`. Returns today's dir and yesterday's
/// (so a launch that straddles midnight still finds its session). Older
/// dates can't possibly hold a session newer than `cutoff`.
fn recent_date_partitions(root: &Path, cutoff: SystemTime) -> Vec<PathBuf> {
    let today = DateTime::<Utc>::from(SystemTime::now()).date_naive();
    let yesterday = DateTime::<Utc>::from(
        cutoff
            .checked_sub(Duration::from_secs(86_400))
            .unwrap_or(SystemTime::UNIX_EPOCH),
    )
    .date_naive();
    let mut dirs = vec![partition_path(root, today)];
    if yesterday != today {
        dirs.push(partition_path(root, yesterday));
    }
    dirs
}

fn partition_path(root: &Path, date: chrono::NaiveDate) -> PathBuf {
    root.join(format!("{:04}", date.format("%Y")))
        .join(format!("{:02}", date.format("%m")))
        .join(format!("{:02}", date.format("%d")))
}

/// `rollout-2025-10-12T11-16-34-0199d66b-832e-7a83-858a-0c09e3751b31` →
/// `0199d66b-832e-7a83-858a-0c09e3751b31`. The UUID is the last 5
/// hyphen-separated components.
fn parse_codex_filename_id(stem: &str) -> Option<String> {
    let parts: Vec<&str> = stem.split('-').collect();
    if parts.len() < 5 {
        return None;
    }
    let tail = &parts[parts.len() - 5..];
    let id = tail.join("-");
    if looks_like_uuid(&id) { Some(id) } else { None }
}

fn looks_like_uuid(s: &str) -> bool {
    s.len() == 36
        && s.as_bytes().iter().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => *b == b'-',
            _ => b.is_ascii_hexdigit(),
        })
}

// ---------------------------------------------------------------------------
// Gemini — id lives inside the JSON; defer the read until we actually need
// the canonical id (i.e., for the new file post-launch). Snapshot uses the
// path string as a stand-in id, which is enough for the exclusion set.
// ---------------------------------------------------------------------------

async fn gemini_entries(
    home: &Path,
    cwd: Option<&str>,
    cutoff: SystemTime,
) -> Vec<(SystemTime, String)> {
    let tmp_root = home.join(".gemini").join("tmp");
    let paths: Vec<PathBuf> = match cwd {
        Some(c) => gemini_matching_session_files_in(&tmp_root, c).await,
        None => gemini_all_session_files(&tmp_root).await,
    };
    let mut out = Vec::new();
    for p in paths {
        let mtime = match file_mtime_within(&p, cutoff).await {
            Some(m) => m,
            None => continue,
        };
        if let Some(id) = read_gemini_session_id(&p).await {
            out.push((mtime, id));
        } else {
            // Fall back to path-as-id so the snapshot's exclusion set still
            // covers this file. The detect side won't surface it because no
            // sessionId means it can't be linked anyway.
            out.push((mtime, p.to_string_lossy().to_string()));
        }
    }
    out
}

async fn gemini_all_session_files(tmp_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(tmp_root).await else {
        return out;
    };
    while let Ok(Some(dir_entry)) = rd.next_entry().await {
        let chats = dir_entry.path().join("chats");
        let Ok(mut sub) = fs::read_dir(&chats).await else {
            continue;
        };
        while let Ok(Some(f)) = sub.next_entry().await {
            let p = f.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("session-") && (name.ends_with(".json") || name.ends_with(".jsonl"))
            {
                out.push(p);
            }
        }
    }
    out
}

async fn read_gemini_session_id(path: &Path) -> Option<String> {
    let content = fs::read_to_string(path).await.ok()?;
    let v = normalize_gemini_session(&content)?;
    v.get("sessionId")
        .and_then(|s| s.as_str())
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Pi — `<ts>_<uuid>.jsonl`; id is the part after `_`; scope by cwd dir
// ---------------------------------------------------------------------------

async fn pi_entries(
    home: &Path,
    cwd: Option<&str>,
    cutoff: SystemTime,
) -> Vec<(SystemTime, String)> {
    let dirs: Vec<PathBuf> = match cwd {
        Some(c) => pi_session_dir(c).into_iter().collect(),
        None => collect_subdirs(&home.join(".pi").join("agent").join("sessions")).await,
    };
    let mut out = Vec::new();
    for dir in dirs {
        scan_jsonl_dir(&dir, cutoff, &mut out, |p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|stem| stem.split_once('_').map(|(_, id)| id.to_string()))
        })
        .await;
    }
    out
}

// ---------------------------------------------------------------------------
// Opencode — sqlite; bound by `time_updated` so the snapshot doesn't scan
// the entire session table on a 70 MB DB
// ---------------------------------------------------------------------------

async fn opencode_entries(db_path: &Path, cutoff: SystemTime) -> Vec<(SystemTime, String)> {
    if !db_path.exists() {
        return Vec::new();
    }
    let cutoff_ms = cutoff
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    let path = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || opencode_query(&path, cutoff_ms))
        .await
        .unwrap_or_default()
}

fn opencode_query(db_path: &Path, cutoff_ms: i64) -> Vec<(SystemTime, String)> {
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut stmt =
        match conn.prepare("SELECT id, time_updated FROM session WHERE time_updated >= ?") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
    let rows = stmt
        .query_map([cutoff_ms], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })
        .ok();
    let mut out = Vec::new();
    if let Some(rows) = rows {
        for row in rows.flatten() {
            let (id, ms) = row;
            let mtime = SystemTime::UNIX_EPOCH + Duration::from_millis(ms.max(0) as u64);
            out.push((mtime, id));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Amp — id is the filename stem (`T-<ulid>.json`); flat dir, no scoping
// ---------------------------------------------------------------------------

async fn amp_entries(threads_dir: &Path, cutoff: SystemTime) -> Vec<(SystemTime, String)> {
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(threads_dir).await else {
        return out;
    };
    while let Ok(Some(f)) = rd.next_entry().await {
        let p = f.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Some(id) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if !id.starts_with("T-") {
            continue;
        }
        let Some(mtime) = file_mtime_within(&p, cutoff).await else {
            continue;
        };
        out.push((mtime, id.to_string()));
    }
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk a single directory of `*.jsonl` files, append `(mtime, id)` for
/// each whose `mtime >= cutoff`. The id-extraction closure decides what
/// counts as a session id; returning `None` skips the file.
async fn scan_jsonl_dir<F>(
    dir: &Path,
    cutoff: SystemTime,
    out: &mut Vec<(SystemTime, String)>,
    id_from: F,
) where
    F: Fn(&Path) -> Option<String>,
{
    let Ok(mut rd) = fs::read_dir(dir).await else {
        return;
    };
    while let Ok(Some(f)) = rd.next_entry().await {
        let p = f.path();
        if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(id) = id_from(&p) else { continue };
        let Some(mtime) = file_mtime_within(&p, cutoff).await else {
            continue;
        };
        out.push((mtime, id));
    }
}

/// `Some(mtime)` iff the file exists and `mtime >= cutoff`. Lets callers
/// short-circuit on stale entries without a separate metadata lookup.
async fn file_mtime_within(path: &Path, cutoff: SystemTime) -> Option<SystemTime> {
    let mtime = fs::metadata(path).await.ok()?.modified().ok()?;
    if mtime >= cutoff { Some(mtime) } else { None }
}

async fn collect_subdirs(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(root).await else {
        return out;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let p = entry.path();
        if p.is_dir() {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn parses_codex_filename() {
        let id = parse_codex_filename_id(
            "rollout-2025-10-12T11-16-34-0199d66b-832e-7a83-858a-0c09e3751b31",
        );
        assert_eq!(id.as_deref(), Some("0199d66b-832e-7a83-858a-0c09e3751b31"));
    }

    #[test]
    fn rejects_non_uuid_codex_tail() {
        assert!(parse_codex_filename_id("rollout-not-a-real-uuid-at-all").is_none());
    }

    #[tokio::test]
    async fn time_window_skips_stale_files() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let stale = dir.join("stale-1.jsonl");
        std::fs::write(&stale, b"{}\n").unwrap();
        // Backdate via std::fs::File::set_modified (stable since Rust 1.75).
        let hour_ago = SystemTime::now() - Duration::from_secs(3_600);
        std::fs::File::options()
            .write(true)
            .open(&stale)
            .unwrap()
            .set_modified(hour_ago)
            .unwrap();

        let mut out = Vec::new();
        let cutoff = SystemTime::now() - Duration::from_secs(5);
        scan_jsonl_dir(&dir, cutoff, &mut out, |p| {
            p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        })
        .await;
        assert!(out.is_empty(), "stale file should have been filtered");
    }

    #[tokio::test]
    async fn detect_picks_newest_post_launch_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cwd_dir = tmp.path().join("-private-tmp-x");
        std::fs::create_dir_all(&cwd_dir).unwrap();
        // Pre-existing file should land in the snapshot's exclusion set.
        std::fs::write(cwd_dir.join("aaaa-1111.jsonl"), b"{}\n").unwrap();
        let cutoff = SystemTime::now() - Duration::from_secs(5);
        let before: HashSet<String> = scan_collected_ids(&cwd_dir, cutoff).await;
        assert!(before.contains("aaaa-1111"));

        // Simulate the launch producing a new file.
        tokio::time::sleep(Duration::from_millis(20)).await;
        std::fs::write(cwd_dir.join("bbbb-2222.jsonl"), b"{}\n").unwrap();
        let mut after: Vec<(SystemTime, String)> = Vec::new();
        scan_jsonl_dir(&cwd_dir, cutoff, &mut after, |p| {
            p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        })
        .await;

        let new_id = after
            .into_iter()
            .filter(|(_, id)| !before.contains(id))
            .max_by_key(|(mtime, _)| *mtime)
            .map(|(_, id)| id);
        assert_eq!(new_id.as_deref(), Some("bbbb-2222"));
    }

    async fn scan_collected_ids(dir: &Path, cutoff: SystemTime) -> HashSet<String> {
        let mut out = Vec::new();
        scan_jsonl_dir(dir, cutoff, &mut out, |p| {
            p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
        })
        .await;
        out.into_iter().map(|(_, id)| id).collect()
    }
}
