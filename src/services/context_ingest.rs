//! On-demand ingestion of AI CLI session content into normalized context threads.
//! Sources: Claude Code (`~/.claude/projects/`), Codex (`~/.codex/sessions/`),
//! Gemini (`~/.gemini/tmp/`), Pi (`~/.pi/agent/sessions/`), OpenCode (`~/.local/share/opencode/opencode.db`).

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::services::ansi;
use crate::services::device_fingerprint::hex_sha256;
use crate::services::project_id::{DEFAULT_THREAD_MAX_AGE_DAYS, Thread};
use crate::services::session_store::{SessionIndexEntry, SessionStore};
use crate::services::system_env;

/// Minimum character count for a turn to count as substantive.
const MIN_TURN_CHARS: usize = 40;

/// Maximum character count retained per turn; longer turns get `…` truncated.
/// Bounds both context-budget consumption and recursion-echo damage.
const MAX_TURN_CHARS: usize = 400;

/// Default per-source cap for early-exit during most-recent-first walks.
/// Picker's working set — the sessions a user might plausibly want to
/// revisit. Users can bypass with `--all`.
const DEFAULT_MAX_THREADS_PER_SOURCE: usize = 20;

/// Lowercase substrings that mark turns to skip entirely (CLI harness
/// metadata wrappers rather than the user's actual intent).
const BOILERPLATE_MARKERS: &[&str] = &[
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<command-name>",
    "<command-message>",
    "<system-reminder>",
    "<environment_context>",
    "<user_instructions>",
    "<developer_instructions>",
    "<turn_aborted>",
    "<user_turn_aborted>",
];

/// Markers for aivo's own previously-injected context. Everything from the
/// first marker is stripped so a CLI that echoes the injection back doesn't
/// create context-in-context recursion on re-ingest.
const AIVO_CONTEXT_MARKERS: &[&str] = &[
    "<aivo_context>",
    "<aivo_memory>",
    "# aivo context",
    "# aivo memory",
    "The block below is aivo context",
    "The block below is aivo memory",
    "aivo context — auto-extracted",
    "aivo memory — context auto-extracted",
];

/// Options controlling ingestion scope. Defaults apply both caps; `--all`
/// clears them, and `--last-days` overrides just the age cap.
#[derive(Debug, Clone, Copy)]
pub struct IngestOptions {
    /// `Some(days)` filters threads older than this; `None` = unlimited.
    pub max_age_days: Option<i64>,
    /// Absolute lower bound on `updated_at`. Combined with `max_age_days`:
    /// when both are set, the more restrictive (later) cutoff wins. Set by
    /// `aivo logs --since` so the global ingester can skip files by mtime
    /// before parsing them.
    pub min_updated_at: Option<DateTime<Utc>>,
    /// `Some(n)` early-exits each source after `n` extractions; `None` = all.
    pub max_per_source: Option<usize>,
    /// When true, read each session only until id/title/time are in hand
    /// (multi-MB transcripts stay unread; `last_response` is empty), and skip
    /// sources with no stop-early read (gemini, opencode). For listing
    /// surfaces; injection callers need the full extraction.
    pub headline: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            max_age_days: Some(DEFAULT_THREAD_MAX_AGE_DAYS),
            min_updated_at: None,
            max_per_source: Some(DEFAULT_MAX_THREADS_PER_SOURCE),
            headline: false,
        }
    }
}

impl IngestOptions {
    /// Bypass both caps — used when `--resume <id>` names a session that may
    /// be older than the capped walk reaches.
    pub fn unlimited() -> Self {
        Self {
            max_age_days: None,
            min_updated_at: None,
            max_per_source: None,
            headline: false,
        }
    }
}

/// Resolve the effective lower-bound timestamp from both filter knobs. When
/// both are set, the more restrictive (later) cutoff wins so callers can't
/// accidentally widen a date filter by also setting an age cap.
fn effective_cutoff(opts: &IngestOptions) -> Option<DateTime<Utc>> {
    let from_days = opts
        .max_age_days
        .map(|d| Utc::now() - chrono::Duration::days(d));
    match (from_days, opts.min_updated_at) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Read the native-CLI sources for the given project root, merge + age-filter,
/// and return threads newest-first. Nothing is persisted.
pub async fn ingest_project(project_root: &Path, opts: IngestOptions) -> Result<Vec<Thread>> {
    ingest_project_inner(None, project_root, opts).await
}

/// `ingest_project` plus aivo's own code sessions (from `store`'s session
/// index) as a sixth source. This is the `--resume` digest path: `aivo
/// logs` lists `[code]` sessions, so resolution must be able to find them too.
pub async fn ingest_project_with_code(
    store: &SessionStore,
    project_root: &Path,
    opts: IngestOptions,
) -> Result<Vec<Thread>> {
    ingest_project_inner(Some(store), project_root, opts).await
}

/// `ingest_project` with `opts.headline` forced on (see the field doc).
/// Powers the `/resume` importable listing so it, `aivo logs`, and the
/// resume resolver share one discovery layer.
pub async fn ingest_project_headlines(
    project_root: &Path,
    mut opts: IngestOptions,
) -> Result<Vec<Thread>> {
    opts.headline = true;
    ingest_project_inner(None, project_root, opts).await
}

async fn ingest_project_inner(
    store: Option<&SessionStore>,
    project_root: &Path,
    opts: IngestOptions,
) -> Result<Vec<Thread>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let cap = opts.max_per_source;

    // Walk-level mtime pruning (mtime ≥ last-message-ts for append-only jsonl;
    // the post-extraction age filter below still applies either way).
    let walk_cutoff = effective_cutoff(&opts).map(SystemTime::from);

    let (claude, codex, gemini, pi, opencode, code) = tokio::join!(
        ingest_claude(&canonical_root, opts, walk_cutoff),
        ingest_codex(&canonical_str, opts, walk_cutoff),
        async {
            match opts.headline {
                // No stop-early read for gemini's single-blob JSON.
                true => Vec::new(),
                false => ingest_gemini(&canonical_str, cap).await,
            }
        },
        ingest_pi(&canonical_str, opts, walk_cutoff),
        async {
            match opts.headline {
                // Nor for opencode's sqlite rows.
                true => Vec::new(),
                false => ingest_opencode(canonical_str.clone(), cap).await,
            }
        },
        async {
            match store {
                Some(s) => ingest_code(s, &canonical_str, cap).await,
                None => Vec::new(),
            }
        },
    );

    let mut threads: Vec<Thread> = Vec::with_capacity(
        claude.len() + codex.len() + gemini.len() + pi.len() + opencode.len() + code.len(),
    );
    threads.extend(claude);
    threads.extend(codex);
    threads.extend(gemini);
    threads.extend(pi);
    threads.extend(opencode);
    threads.extend(code);

    if let Some(cutoff) = effective_cutoff(&opts) {
        threads.retain(|t| t.updated_at >= cutoff);
    }
    threads.sort_by_key(|t| std::cmp::Reverse(t.updated_at));
    Ok(threads)
}

/// Global counterpart to `ingest_project`: walks every native CLI session
/// aivo can see, regardless of cwd.
pub async fn ingest_native_sessions_global(
    opts: IngestOptions,
    need_last_response: bool,
) -> Result<Vec<Thread>> {
    let cap = opts.max_per_source;
    let headline = !need_last_response;
    let cutoff = effective_cutoff(&opts);
    let cutoff_st = cutoff.map(SystemTime::from);
    let (claude, codex, gemini, pi, opencode) = tokio::join!(
        ingest_claude_global(cap, cutoff_st, headline),
        ingest_codex_global(cap, cutoff_st, headline),
        ingest_gemini_global(cap, cutoff_st),
        ingest_pi_global(cap, cutoff_st, headline),
        ingest_opencode_global(cap, cutoff),
    );

    let mut threads: Vec<Thread> =
        Vec::with_capacity(claude.len() + codex.len() + gemini.len() + pi.len() + opencode.len());
    threads.extend(claude);
    threads.extend(codex);
    threads.extend(gemini);
    threads.extend(pi);
    threads.extend(opencode);

    // Re-check thread updated_at (mtime may lag behind last message ts).
    if let Some(cutoff) = cutoff {
        threads.retain(|t| t.updated_at >= cutoff);
    }
    threads.sort_by_key(|t| std::cmp::Reverse(t.updated_at));
    Ok(threads)
}

async fn ingest_claude_global(
    cap: Option<usize>,
    after: Option<SystemTime>,
    headline: bool,
) -> Vec<Thread> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    let projects_root = home.join(".claude").join("projects");
    let Ok(mut rd) = fs::read_dir(&projects_root).await else {
        return Vec::new();
    };

    // Collect all jsonl files under each cwd-encoded subdir, sorted desc by
    // mtime. Early-exit at the cap.
    let mut all_paths: Vec<(PathBuf, SystemTime)> = Vec::new();
    while let Ok(Some(entry)) = rd.next_entry().await {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let mut sub = match fs::read_dir(&dir).await {
            Ok(s) => s,
            Err(_) => continue,
        };
        while let Ok(Some(f)) = sub.next_entry().await {
            let p = f.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let mtime = f
                .metadata()
                .await
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if let Some(c) = after
                && mtime < c
            {
                continue;
            }
            all_paths.push((p, mtime));
        }
    }
    all_paths.sort_by_key(|e| std::cmp::Reverse(e.1));

    let mut out = Vec::new();
    for (path, mtime) in all_paths {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        let thread = if headline {
            extract_claude_thread_headline(&path, mtime).await
        } else {
            extract_claude_thread(&path).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

async fn ingest_codex_global(
    cap: Option<usize>,
    after: Option<SystemTime>,
    headline: bool,
) -> Vec<Thread> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    let codex_root = home.join(".codex").join("sessions");
    if !codex_root.exists() {
        return Vec::new();
    }
    let files = walk_jsonl_newest_first(&codex_root, after).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        let thread = if headline {
            extract_codex_thread_headline(&path).await
        } else {
            extract_codex_thread_any(&path).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

/// Codex extractor without the cwd filter — accepts any session_meta.cwd
/// and surfaces it on the resulting `Thread`. Used by the global ingester.
async fn extract_codex_thread_any(path: &Path) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;
    let mut session_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str()) {
                session_cwd = Some(cwd.to_string());
            }
        }
        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }
        if kind == "response_item"
            && let Some(payload) = v.get("payload")
            && payload.get("type").and_then(|t| t.as_str()) == Some("message")
        {
            let role = payload.get("role").and_then(|s| s.as_str()).unwrap_or("");
            let raw = extract_codex_message_text(payload).unwrap_or_default();
            match role {
                "user" if first_user.is_none() => {
                    first_user = pick_first_user_turn(&raw);
                }
                "assistant" => {
                    if let Some(t) = sanitize_turn(&raw) {
                        last_assistant = Some(t);
                    }
                }
                _ => {}
            }
        }
    }

    Some(Thread {
        cli: "codex".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at: last_timestamp.unwrap_or_else(Utc::now),
        cwd: session_cwd,
    })
}

/// Head-only codex extractor; see `extract_claude_thread_headline`.
async fn extract_codex_thread_headline(path: &Path) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut session_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str()) {
                session_cwd = Some(cwd.to_string());
            }
        }
        if first_user.is_none()
            && kind == "response_item"
            && let Some(payload) = v.get("payload")
            && payload.get("type").and_then(|t| t.as_str()) == Some("message")
            && payload.get("role").and_then(|s| s.as_str()) == Some("user")
        {
            let raw = extract_codex_message_text(payload).unwrap_or_default();
            first_user = pick_first_user_turn(&raw);
        }
        if session_id.is_some() && first_user.is_some() {
            break;
        }
    }

    let updated_at = fs::metadata(path)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .map(DateTime::<Utc>::from)
        .unwrap_or_else(Utc::now);
    Some(Thread {
        cli: "codex".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: String::new(),
        updated_at,
        cwd: session_cwd,
    })
}

async fn ingest_gemini_global(cap: Option<usize>, after: Option<SystemTime>) -> Vec<Thread> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    let tmp_root = home.join(".gemini").join("tmp");
    let Ok(mut rd) = fs::read_dir(&tmp_root).await else {
        return Vec::new();
    };
    let mut all_paths: Vec<(PathBuf, SystemTime)> = Vec::new();
    while let Ok(Some(dir_entry)) = rd.next_entry().await {
        let chats = dir_entry.path().join("chats");
        if !chats.is_dir() {
            continue;
        }
        let mut sub = match fs::read_dir(&chats).await {
            Ok(s) => s,
            Err(_) => continue,
        };
        while let Ok(Some(f)) = sub.next_entry().await {
            let p = f.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !is_gemini_session_file(name) {
                continue;
            }
            let mtime = f
                .metadata()
                .await
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if let Some(c) = after
                && mtime < c
            {
                continue;
            }
            all_paths.push((p, mtime));
        }
    }
    all_paths.sort_by_key(|e| std::cmp::Reverse(e.1));

    let mut out = Vec::new();
    for (path, _) in all_paths {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_gemini_thread(&path).await {
            out.push(thread);
        }
    }
    out
}

async fn ingest_pi_global(
    cap: Option<usize>,
    after: Option<SystemTime>,
    headline: bool,
) -> Vec<Thread> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    let sessions_root = home.join(".pi").join("agent").join("sessions");
    let Ok(mut rd) = fs::read_dir(&sessions_root).await else {
        return Vec::new();
    };
    let mut all_paths: Vec<(PathBuf, SystemTime)> = Vec::new();
    while let Ok(Some(dir_entry)) = rd.next_entry().await {
        let dir = dir_entry.path();
        if !dir.is_dir() {
            continue;
        }
        let mut sub = match fs::read_dir(&dir).await {
            Ok(s) => s,
            Err(_) => continue,
        };
        while let Ok(Some(f)) = sub.next_entry().await {
            let p = f.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let mtime = f
                .metadata()
                .await
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            if let Some(c) = after
                && mtime < c
            {
                continue;
            }
            all_paths.push((p, mtime));
        }
    }
    all_paths.sort_by_key(|e| std::cmp::Reverse(e.1));

    let mut out = Vec::new();
    for (path, mtime) in all_paths {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        let thread = if headline {
            extract_pi_thread_headline(&path, mtime).await
        } else {
            extract_pi_thread(&path).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

async fn ingest_opencode_global(cap: Option<usize>, after: Option<DateTime<Utc>>) -> Vec<Thread> {
    let Some(home) = system_env::home_dir() else {
        return Vec::new();
    };
    let db_path = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.exists() {
        return Vec::new();
    }
    let cap_i = cap.unwrap_or(1_000) as i64;
    // opencode stores `time_updated` as unix-millis; bind 0 when there's no
    // cutoff so the SQL doesn't need two prepared statements.
    let after_ms = after.map(|dt| dt.timestamp_millis()).unwrap_or(0);

    tokio::task::spawn_blocking(move || opencode_query_global(&db_path, cap_i, after_ms))
        .await
        .unwrap_or_default()
}

fn opencode_query_global(db_path: &Path, cap: i64, after_ms: i64) -> Vec<Thread> {
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(err) => {
            warn_unreadable_session(db_path, &err.to_string());
            return Vec::new();
        }
    };

    // Pull every project's recent sessions ordered globally by recency.
    let mut stmt = match conn.prepare(
        "SELECT s.id, s.time_updated, p.worktree
           FROM session s
           JOIN project p ON p.id = s.project_id
           WHERE s.time_updated >= ?2
           ORDER BY s.time_updated DESC
           LIMIT ?1",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let rows: Vec<(String, i64, String)> = stmt
        .query_map([cap, after_ms], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })
        .and_then(|it| it.collect::<rusqlite::Result<Vec<_>>>())
        .unwrap_or_default();

    let mut out = Vec::new();
    for (session_id, time_updated_ms, worktree) in rows {
        if let Some(thread) =
            opencode_extract_session(db_path, &conn, &session_id, time_updated_ms, Some(worktree))
        {
            out.push(thread);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Claude: ~/.claude/projects/<encoded-cwd>/*.jsonl
// ---------------------------------------------------------------------------

async fn ingest_claude(
    canonical_root: &Path,
    opts: IngestOptions,
    walk_cutoff: Option<SystemTime>,
) -> Vec<Thread> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let session_dir = home
        .join(".claude")
        .join("projects")
        .join(encode_claude_dir(&canonical_root.to_string_lossy()));
    if !session_dir.exists() {
        return Vec::new();
    }

    let files = list_jsonl_newest_first(&session_dir, walk_cutoff).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = opts.max_per_source
            && out.len() >= n
        {
            break;
        }
        let thread = if opts.headline {
            let mtime = file_mtime(&path).await;
            extract_claude_thread_headline(&path, mtime).await
        } else {
            extract_claude_thread(&path).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

/// Stub enumerator for the share resolver: one `Thread` per claude session file
/// under the project's encoded cwd dir, with only `session_id`, `source_path`,
/// `updated_at` set. Unlike `extract_claude_thread` it keeps files with no
/// extractable user turn, which would otherwise be un-shareable.
pub async fn list_claude_sessions_for_cwd(project_root: &Path) -> Vec<Thread> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let session_dir = home
        .join(".claude")
        .join("projects")
        .join(encode_claude_dir(&canonical_root.to_string_lossy()));
    if !session_dir.exists() {
        return Vec::new();
    }

    let files = list_jsonl_newest_first(&session_dir, None).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(stub) = extract_claude_session_stub(&path).await {
            out.push(stub);
        }
    }
    out
}

/// Read the headline fields of a claude JSONL: first `sessionId` and latest
/// `timestamp`. Falls back to file mtime when the file carries no timestamps.
async fn extract_claude_session_stub(path: &Path) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut latest_ts: Option<DateTime<Utc>> = None;
    let mut event_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if session_id.is_none()
            && let Some(sid) = v.get("sessionId").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }
        if event_cwd.is_none()
            && let Some(c) = v.get("cwd").and_then(|s| s.as_str())
            && !c.is_empty()
        {
            event_cwd = Some(c.to_string());
        }
        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            let parsed = parsed.with_timezone(&Utc);
            latest_ts = Some(latest_ts.map_or(parsed, |cur| cur.max(parsed)));
        }
    }

    let session_id = session_id?;
    let updated_at = match latest_ts {
        Some(ts) => ts,
        None => fs::metadata(path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now),
    };
    Some(Thread {
        cli: "claude".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        topic: String::new(),
        last_response: String::new(),
        updated_at,
        cwd: event_cwd.or_else(|| decode_claude_cwd(path)),
    })
}

// ---------------------------------------------------------------------------
// Codex: ~/.codex/sessions/YYYY/MM/DD/*.jsonl, per-file cwd match
// ---------------------------------------------------------------------------

/// Codex sessions live in one flat tree for every repo; in headline mode
/// (listing surfaces) bound how many files we peek at (one header read each)
/// while filtering to this cwd. The full mode stays uncapped — the digest
/// path may legitimately name an old session by id.
const CODEX_PROJECT_PROBE_LIMIT: usize = 400;

async fn ingest_codex(
    canonical_root: &str,
    opts: IngestOptions,
    walk_cutoff: Option<SystemTime>,
) -> Vec<Thread> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let codex_root = home.join(".codex").join("sessions");
    if !codex_root.exists() {
        return Vec::new();
    }

    let files = walk_jsonl_newest_first(&codex_root, walk_cutoff).await;
    let mut out = Vec::new();
    for (probed, path) in files.into_iter().enumerate() {
        if let Some(n) = opts.max_per_source
            && out.len() >= n
        {
            break;
        }
        if opts.headline && probed >= CODEX_PROJECT_PROBE_LIMIT {
            break;
        }
        let thread = if opts.headline {
            // The headline extractor has no cwd filter (it serves the global
            // walk too) — match on the returned session cwd here.
            extract_codex_thread_headline(&path).await.filter(|t| {
                t.cwd
                    .as_deref()
                    .is_some_and(|cwd| paths_match(cwd, canonical_root))
            })
        } else {
            extract_codex_thread(&path, canonical_root).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

/// Stub enumerator for the share resolver's run-event path: one `Thread` per
/// codex session file whose `session_meta.cwd` matches `project_root`, with
/// `topic` / `last_response` left empty. Unlike `extract_codex_thread` it keeps
/// files with no extractable user turn, else the resolver would drop them and
/// pick a stale older session in the same cwd. `codex_root` is parameterized so
/// tests can inject a temp dir.
pub async fn list_codex_sessions_for_cwd(codex_root: &Path, project_root: &Path) -> Vec<Thread> {
    if !codex_root.exists() {
        return Vec::new();
    }
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();

    let files = walk_jsonl_newest_first(codex_root, None).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(stub) = extract_codex_session_stub(&path, &canonical_str).await {
            out.push(stub);
        }
    }
    out
}

/// Read codex session headlines without scanning every response_item. We
/// need session_meta (id + cwd), the file's last timestamp for closest-
/// mtime sorting, and that's it. Falls back to filesystem mtime when the
/// jsonl carries no timestamps.
async fn extract_codex_session_stub(path: &Path, project_root: &str) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut session_cwd: Option<String> = None;
    let mut project_matches = false;
    let mut latest_ts: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            latest_ts = Some(parsed.with_timezone(&Utc));
        }
        if v.get("type").and_then(|t| t.as_str()) == Some("session_meta")
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str()) {
                session_cwd = Some(cwd.to_string());
                if paths_match(cwd, project_root) {
                    project_matches = true;
                }
            }
        }
    }

    if !project_matches {
        return None;
    }
    let updated_at = match latest_ts {
        Some(ts) => ts,
        None => fs::metadata(path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now),
    };
    Some(Thread {
        cli: "codex".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: String::new(),
        last_response: String::new(),
        updated_at,
        cwd: session_cwd,
    })
}

// ---------------------------------------------------------------------------
// Gemini: ~/.gemini/tmp/<sha256(abs_cwd)>/chats/session-*.json
// ---------------------------------------------------------------------------

async fn ingest_gemini(canonical_root: &str, cap: Option<usize>) -> Vec<Thread> {
    let paths = gemini_matching_session_files(canonical_root).await;
    let mut out = Vec::new();
    for path in paths {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_gemini_thread(&path).await {
            out.push(thread);
        }
    }
    out
}

/// Return every Gemini `session-*.json` path whose parent `chats/` dir
/// belongs to this canonical project, sorted newest-first by mtime.
///
/// Two layouts are in the wild under `~/.gemini/tmp/`: `<hash>/chats/`
/// (default) and `<friendly-name>/chats/` (older or configured). We can't
/// recognize the latter from the directory name, so we scan every dir,
/// peek at one session per dir to learn its `projectHash`, and only keep
/// dirs whose hash matches `sha256(canonical_root)`.
pub(crate) async fn gemini_matching_session_files(canonical_root: &str) -> Vec<PathBuf> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    gemini_matching_session_files_in(&home.join(".gemini").join("tmp"), canonical_root).await
}

/// `gemini_matching_session_files` parameterized on the `~/.gemini/tmp/` root
/// so tests (and the share resolver's run-event path, which carries an
/// explicit `gemini_tmp_root` in `ResolverContext`) can inject a non-HOME
/// directory without mutating env vars.
pub(crate) async fn gemini_matching_session_files_in(
    tmp_root: &Path,
    canonical_root: &str,
) -> Vec<PathBuf> {
    let project_hash = hex_sha256(canonical_root.as_bytes());
    if !tmp_root.exists() {
        return Vec::new();
    }
    let mut entries: Vec<(PathBuf, SystemTime)> = Vec::new();
    let mut tmp_rd = match fs::read_dir(tmp_root).await {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    while let Ok(Some(dir_entry)) = tmp_rd.next_entry().await {
        let chats_dir = dir_entry.path().join("chats");
        if !chats_dir.is_dir() {
            continue;
        }
        let mut chats_rd = match fs::read_dir(&chats_dir).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        let mut chat_files: Vec<(PathBuf, SystemTime)> = Vec::new();
        while let Ok(Some(f)) = chats_rd.next_entry().await {
            let path = f.path();
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !is_gemini_session_file(name) {
                continue;
            }
            let mtime = f
                .metadata()
                .await
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            chat_files.push((path, mtime));
        }
        if chat_files.is_empty() {
            continue;
        }
        if !gemini_dir_matches(&chat_files[0].0, &project_hash).await {
            continue;
        }
        entries.extend(chat_files);
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

/// Accept both the legacy `.json` and the newer `.jsonl` gemini session
/// files. gemini-cli switched the per-message storage format around early
/// 2026; ignoring `.jsonl` silently hides every recent session.
pub(crate) fn is_gemini_session_file(name: &str) -> bool {
    name.starts_with("session-") && (name.ends_with(".json") || name.ends_with(".jsonl"))
}

/// Normalize a gemini session file into the legacy `{sessionId, lastUpdated,
/// messages: [...]}` Value, regardless of on-disk format. Returns `None`
/// when the content doesn't carry a `sessionId`.
///
/// The `.jsonl` format is: header line (`{sessionId, projectHash, …}`),
/// then message/$set lines. `$set` is a mongo-style header patch — we fold
/// it into the header so downstream code sees the latest `lastUpdated`.
pub(crate) fn normalize_gemini_session(content: &str) -> Option<Value> {
    if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(content)
        && obj.contains_key("sessionId")
        && obj.contains_key("messages")
    {
        return Some(Value::Object(obj));
    }
    let mut header: Option<serde_json::Map<String, Value>> = None;
    let mut messages: Vec<Value> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if header.is_none() && obj.contains_key("sessionId") {
            header = Some(obj);
            continue;
        }
        if let Some(set) = obj.get("$set").and_then(|s| s.as_object())
            && let Some(h) = header.as_mut()
        {
            for (k, v) in set {
                h.insert(k.clone(), v.clone());
            }
            continue;
        }
        if obj.contains_key("type") {
            messages.push(Value::Object(obj));
        }
    }
    let mut header = header?;
    header.insert("messages".to_string(), Value::Array(messages));
    Some(Value::Object(header))
}

/// Read the project root recorded by gemini-cli next to the `chats/`
/// directory. Newer gemini-cli layouts write a `.project_root` plaintext
/// file holding the absolute cwd; older sha256-hashed layouts omit it,
/// in which case we leave cwd unknown.
async fn gemini_project_root_for_session(session_path: &Path) -> Option<String> {
    let chats_dir = session_path.parent()?;
    let project_dir = chats_dir.parent()?;
    let content = fs::read_to_string(project_dir.join(".project_root"))
        .await
        .ok()?;
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

async fn extract_gemini_thread(path: &Path) -> Option<Thread> {
    let content = match fs::read_to_string(path).await {
        Ok(c) => c,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let v = match normalize_gemini_session(&content) {
        Some(v) => v,
        None => {
            warn_unreadable_session(path, "not a recognized gemini session format");
            return None;
        }
    };

    let session_id = v.get("sessionId").and_then(|s| s.as_str())?.to_string();
    let messages = v.get("messages")?.as_array()?;

    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;

    for msg in messages {
        let kind = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != "user" && kind != "gemini" {
            continue;
        }
        let raw = extract_gemini_content(msg.get("content")).unwrap_or_default();
        if let Some(ts) = msg.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }
        match kind {
            "user" if first_user.is_none() => {
                first_user = pick_first_user_turn(&raw);
            }
            "gemini" => {
                if let Some(t) = sanitize_turn(&raw) {
                    last_assistant = Some(t);
                }
            }
            _ => {}
        }
    }

    // Fall back to top-level lastUpdated when per-message timestamps missing.
    if last_timestamp.is_none()
        && let Some(ts) = v.get("lastUpdated").and_then(|s| s.as_str())
        && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
    {
        last_timestamp = Some(parsed.with_timezone(&Utc));
    }

    Some(Thread {
        cli: "gemini".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at: last_timestamp.unwrap_or_else(Utc::now),
        cwd: gemini_project_root_for_session(path).await,
    })
}

/// Stub enumerator for the share resolver's run-event path: one `Thread` per
/// gemini session file, keeping files with no extractable user turn that
/// `extract_gemini_thread` would drop (same bug class as claude/codex/pi).
/// `gemini_tmp_root` is parameterized so tests can inject a temp dir.
pub async fn list_gemini_sessions_for_cwd(
    gemini_tmp_root: &Path,
    project_root: &Path,
) -> Vec<Thread> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let paths = gemini_matching_session_files_in(gemini_tmp_root, &canonical_str).await;
    let mut out = Vec::new();
    for path in paths {
        if let Some(stub) = extract_gemini_session_stub(&path).await {
            out.push(stub);
        }
    }
    out
}

/// Headline-only read: `sessionId` + the latest per-message timestamp (or
/// `lastUpdated` as a fallback). Bounded — gemini session files are JSON
/// blobs but we still only need a handful of fields.
async fn extract_gemini_session_stub(path: &Path) -> Option<Thread> {
    let content = fs::read_to_string(path).await.ok()?;
    let v = normalize_gemini_session(&content)?;
    let session_id = v.get("sessionId").and_then(|s| s.as_str())?.to_string();

    let mut latest_ts: Option<DateTime<Utc>> = None;
    if let Some(messages) = v.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if let Some(ts) = msg.get("timestamp").and_then(|s| s.as_str())
                && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
            {
                let parsed = parsed.with_timezone(&Utc);
                latest_ts = Some(latest_ts.map_or(parsed, |cur| cur.max(parsed)));
            }
        }
    }
    if latest_ts.is_none()
        && let Some(ts) = v.get("lastUpdated").and_then(|s| s.as_str())
        && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
    {
        latest_ts = Some(parsed.with_timezone(&Utc));
    }
    let updated_at = match latest_ts {
        Some(ts) => ts,
        None => fs::metadata(path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now),
    };
    Some(Thread {
        cli: "gemini".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        topic: String::new(),
        last_response: String::new(),
        updated_at,
        cwd: gemini_project_root_for_session(path).await,
    })
}

/// Returns true iff the session file's `projectHash` matches `target_hash`.
/// `projectHash` is at the top of every session file, so a bounded read is
/// enough — avoids loading multi-MB conversation JSON just to check 64 bytes.
async fn gemini_dir_matches(sample_path: &Path, target_hash: &str) -> bool {
    let mut file = match fs::File::open(sample_path).await {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut head = vec![0u8; 4096];
    let n = match file.read(&mut head).await {
        Ok(n) => n,
        Err(_) => return false,
    };
    let head_str = match std::str::from_utf8(&head[..n]) {
        Ok(s) => s,
        Err(_) => return false,
    };
    // The field is a top-level string; no JSON parser needed for a 64-char
    // hex match. Tolerate the optional whitespace pretty-printed JSON inserts
    // between the key and value (e.g. `"projectHash": "<hash>"`).
    let Some(after_key) = head_str.find("\"projectHash\"") else {
        return false;
    };
    let tail = &head_str[after_key..];
    let Some(value_start) = tail.find('"').and_then(|first| {
        // skip the opening quote of the key, find the colon, then the value's opening quote.
        let after_colon = tail[first + 1..].find(':')?;
        let value_quote_rel = tail[first + 1 + after_colon + 1..].find('"')?;
        Some(first + 1 + after_colon + 1 + value_quote_rel + 1)
    }) else {
        return false;
    };
    tail[value_start..]
        .get(..target_hash.len())
        .is_some_and(|h| h == target_hash)
}

/// Gemini message `content` shape varies by role:
/// - assistant (`type:"gemini"`) is a plain string
/// - user (`type:"user"`) is an array of `{text}` blocks
///
/// We accept both forms.
pub(crate) fn extract_gemini_content(content: Option<&Value>) -> Option<String> {
    let v = content?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = v.as_array() {
        let mut buf = String::new();
        for block in arr {
            if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(t);
            }
        }
        if !buf.is_empty() {
            return Some(buf);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Pi: ~/.pi/agent/sessions/--<cwd-slashes-as-dashes>--/*.jsonl
// ---------------------------------------------------------------------------

/// Pi's per-cwd session directory. Returns `None` if HOME is unavailable.
pub(crate) fn pi_session_dir(canonical_root: &str) -> Option<PathBuf> {
    let home = system_env::home_dir()?;
    let encoded = format!("--{}--", canonical_root.trim_matches('/').replace('/', "-"));
    Some(
        home.join(".pi")
            .join("agent")
            .join("sessions")
            .join(encoded),
    )
}

async fn ingest_pi(
    canonical_root: &str,
    opts: IngestOptions,
    walk_cutoff: Option<SystemTime>,
) -> Vec<Thread> {
    let session_dir = match pi_session_dir(canonical_root) {
        Some(d) if d.exists() => d,
        _ => return Vec::new(),
    };

    let files = list_jsonl_newest_first(&session_dir, walk_cutoff).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = opts.max_per_source
            && out.len() >= n
        {
            break;
        }
        let thread = if opts.headline {
            let mtime = file_mtime(&path).await;
            extract_pi_thread_headline(&path, mtime).await
        } else {
            extract_pi_thread(&path).await
        };
        if let Some(thread) = thread {
            out.push(thread);
        }
    }
    out
}

async fn extract_pi_thread(path: &Path) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        if kind == "session"
            && let Some(id) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(id.to_string());
        }

        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }

        if kind != "message" {
            continue;
        }
        let message = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        let role = message.get("role").and_then(|s| s.as_str()).unwrap_or("");
        let raw = extract_pi_text(message).unwrap_or_default();
        match role {
            "user" if first_user.is_none() => {
                first_user = pick_first_user_turn(&raw);
            }
            "assistant" => {
                if let Some(t) = sanitize_turn(&raw) {
                    last_assistant = Some(t);
                }
            }
            _ => {}
        }
    }

    Some(Thread {
        cli: "pi".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at: last_timestamp.unwrap_or_else(Utc::now),
        cwd: decode_pi_cwd(path),
    })
}

/// Head-only pi extractor; see `extract_claude_thread_headline`.
async fn extract_pi_thread_headline(path: &Path, mtime: SystemTime) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if session_id.is_none()
            && kind == "session"
            && let Some(id) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(id.to_string());
        }
        if first_user.is_none()
            && kind == "message"
            && let Some(message) = v.get("message")
            && message.get("role").and_then(|s| s.as_str()) == Some("user")
        {
            let raw = extract_pi_text(message).unwrap_or_default();
            first_user = pick_first_user_turn(&raw);
        }
        if session_id.is_some() && first_user.is_some() {
            break;
        }
    }

    Some(Thread {
        cli: "pi".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: String::new(),
        updated_at: DateTime::<Utc>::from(mtime),
        cwd: decode_pi_cwd(path),
    })
}

/// Stub enumerator for the share resolver's run-event path: one `Thread` per pi
/// session file under the project's encoded cwd dir, keeping files with no
/// extractable user turn that `extract_pi_thread` would drop — else a brand-new
/// run resolves to a stale older session in the same cwd. `pi_root` is
/// parameterized so tests can inject a temp dir.
pub async fn list_pi_sessions_for_cwd(pi_root: &Path, project_root: &Path) -> Vec<Thread> {
    if !pi_root.exists() {
        return Vec::new();
    }
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let encoded = format!("--{}--", canonical_str.trim_matches('/').replace('/', "-"));
    let session_dir = pi_root.join(encoded);
    if !session_dir.exists() {
        return Vec::new();
    }

    let files = list_jsonl_newest_first(&session_dir, None).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(stub) = extract_pi_session_stub(&path).await {
            out.push(stub);
        }
    }
    out
}

/// Headline-only read for resolver use: session id + latest timestamp.
/// Falls back to file mtime when the jsonl has no timestamps yet (e.g. a
/// session that just started and hasn't recorded its first turn).
async fn extract_pi_session_stub(path: &Path) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut latest_ts: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if session_id.is_none()
            && v.get("type").and_then(|t| t.as_str()) == Some("session")
            && let Some(id) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(id.to_string());
        }
        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            let parsed = parsed.with_timezone(&Utc);
            latest_ts = Some(latest_ts.map_or(parsed, |cur| cur.max(parsed)));
        }
    }

    let session_id = session_id?;
    let updated_at = match latest_ts {
        Some(ts) => ts,
        None => fs::metadata(path)
            .await
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now),
    };
    Some(Thread {
        cli: "pi".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        topic: String::new(),
        last_response: String::new(),
        updated_at,
        cwd: decode_pi_cwd(path),
    })
}

/// Reverse the `--<dashes>--` encoding that Pi uses for its per-cwd session
/// dirs. `--Users-alice-foo--` → `/Users/alice/foo`. Lossy when the original cwd
/// itself contained literal `-` characters; acceptable for display and
/// substring filtering.
fn decode_pi_cwd(path: &Path) -> Option<String> {
    let parent = path.parent()?.file_name()?.to_str()?;
    let inner = parent.strip_prefix("--")?.strip_suffix("--")?;
    Some(format!("/{}", inner.replace('-', "/")))
}

/// Pi message content is an array of blocks; text blocks carry `text`.
pub(crate) fn extract_pi_text(message: &Value) -> Option<String> {
    let arr = message.get("content")?.as_array()?;
    let mut buf = String::new();
    for block in arr {
        if block.get("type").and_then(|t| t.as_str()) == Some("text")
            && let Some(t) = block.get("text").and_then(|t| t.as_str())
        {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(t);
        }
    }
    if buf.is_empty() { None } else { Some(buf) }
}

// ---------------------------------------------------------------------------
// OpenCode: SQLite at ~/.local/share/opencode/opencode.db
// ---------------------------------------------------------------------------

async fn ingest_opencode(canonical_root: String, cap: Option<usize>) -> Vec<Thread> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let db_path = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.exists() {
        return Vec::new();
    }
    let cap_i = cap.unwrap_or(1_000) as i64;

    // rusqlite is sync — run the whole query block in spawn_blocking.
    tokio::task::spawn_blocking(move || opencode_query(&db_path, &canonical_root, cap_i))
        .await
        .unwrap_or_default()
}

fn opencode_query(db_path: &Path, project_root: &str, cap: i64) -> Vec<Thread> {
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(err) => {
            warn_unreadable_session(db_path, &err.to_string());
            return Vec::new();
        }
    };

    // Find the project id for this worktree. Most common case is an exact
    // match on the canonical path.
    let project_id: Option<String> = conn
        .query_row(
            "SELECT id FROM project WHERE worktree = ?1 LIMIT 1",
            [project_root],
            |row| row.get::<_, String>(0),
        )
        .ok();
    let project_id = match project_id {
        Some(p) => p,
        None => return Vec::new(),
    };

    // Pull the N most-recent sessions.
    let mut stmt = match conn.prepare(
        "SELECT id, time_updated FROM session
           WHERE project_id = ?1
           ORDER BY time_updated DESC
           LIMIT ?2",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let sessions: Vec<(String, i64)> = stmt
        .query_map(rusqlite::params![project_id, cap], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .unwrap_or_default();

    let mut out = Vec::new();
    for (session_id, time_updated_ms) in sessions {
        if let Some(thread) = opencode_extract_session(
            db_path,
            &conn,
            &session_id,
            time_updated_ms,
            Some(project_root.to_string()),
        ) {
            out.push(thread);
        }
    }
    out
}

fn opencode_extract_session(
    db_path: &Path,
    conn: &rusqlite::Connection,
    session_id: &str,
    time_updated_ms: i64,
    worktree: Option<String>,
) -> Option<Thread> {
    // Join messages to their text parts, ordered chronologically. First
    // `role=user` → topic; last `role=assistant` → last_response.
    let mut stmt = conn
        .prepare(
            "SELECT json_extract(m.data, '$.role') AS role, p.data
               FROM message m
               JOIN part p ON p.message_id = m.id
              WHERE m.session_id = ?1
                AND json_extract(p.data, '$.type') = 'text'
              ORDER BY m.time_created ASC, p.time_created ASC",
        )
        .ok()?;
    let rows = stmt
        .query_map([session_id], |row| {
            Ok((
                row.get::<_, Option<String>>(0)?.unwrap_or_default(),
                row.get::<_, String>(1)?,
            ))
        })
        .ok()?;

    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    for row in rows.flatten() {
        let (role, part_json) = row;
        let v: Value = match serde_json::from_str(&part_json) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let raw = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
        match role.as_str() {
            "user" if first_user.is_none() => {
                first_user = pick_first_user_turn(raw);
            }
            "assistant" => {
                if let Some(t) = sanitize_turn(raw) {
                    last_assistant = Some(t);
                }
            }
            _ => {}
        }
    }

    let updated_at =
        chrono::DateTime::<Utc>::from_timestamp_millis(time_updated_ms).unwrap_or_else(Utc::now);
    Some(Thread {
        cli: "opencode".into(),
        session_id: session_id.to_string(),
        source_path: db_path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at,
        cwd: worktree,
    })
}

/// Stub enumerator for the share resolver: one `Thread` per `session` row whose
/// `project_id` matches, skipping the `message`/`part` join that
/// `opencode_extract_session` uses for `topic`/`last_response`. The resolver
/// only needs `session_id` + `updated_at` for its closest-mtime pick.
pub async fn list_opencode_sessions_for_cwd(db_path: &Path, project_root: &Path) -> Vec<Thread> {
    if !db_path.exists() {
        return Vec::new();
    }
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let db_path = db_path.to_path_buf();
    tokio::task::spawn_blocking(move || opencode_list_stubs(&db_path, &canonical_str))
        .await
        .unwrap_or_default()
}

fn opencode_list_stubs(db_path: &Path, project_root: &str) -> Vec<Thread> {
    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let project_id: Option<String> = conn
        .query_row(
            "SELECT id FROM project WHERE worktree = ?1 LIMIT 1",
            [project_root],
            |row| row.get::<_, String>(0),
        )
        .ok();
    let Some(project_id) = project_id else {
        return Vec::new();
    };
    let mut stmt = match conn.prepare(
        "SELECT id, time_updated FROM session
           WHERE project_id = ?1
           ORDER BY time_updated DESC",
    ) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let rows: Vec<(String, i64)> = stmt
        .query_map([&project_id], |row| Ok((row.get(0)?, row.get(1)?)))
        .and_then(|it| it.collect::<rusqlite::Result<Vec<_>>>())
        .unwrap_or_default();
    rows.into_iter()
        .map(|(session_id, time_ms)| Thread {
            cli: "opencode".into(),
            session_id: session_id.clone(),
            source_path: db_path.to_string_lossy().to_string(),
            topic: String::new(),
            last_response: String::new(),
            updated_at: chrono::DateTime::<Utc>::from_timestamp_millis(time_ms)
                .unwrap_or_else(Utc::now),
            cwd: Some(project_root.to_string()),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// aivo code: session-store index (`~/.config/aivo/sessions`)
// ---------------------------------------------------------------------------

/// Aivo's own code sessions as a source. The TUI persists the real launch
/// dir as `cwd`, so per-project matching works; plain `-p` one-shots save
/// under the chat sandbox dir and never match a project root.
async fn ingest_code(
    store: &SessionStore,
    canonical_root: &str,
    cap: Option<usize>,
) -> Vec<Thread> {
    let mut entries = match store.all_chat_sessions().await {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    entries.retain(|e| paths_match(&e.cwd, canonical_root));
    // Index order isn't guaranteed; `updated_at` is RFC3339 (UTC), so the
    // lexicographic sort is chronological.
    entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    let mut out = Vec::new();
    for entry in entries {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_code_thread(store, &entry).await {
            out.push(thread);
        }
    }
    out
}

/// Same bracket extraction as the native sources, over the stored display
/// messages. `strip_aivo_context` matters here too: a `-p` one-shot seeded
/// with a context digest stores the injected block at the head of its first user
/// message.
async fn extract_code_thread(store: &SessionStore, entry: &SessionIndexEntry) -> Option<Thread> {
    let state = store.get_code_session(&entry.session_id).await.ok()??;

    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    for msg in &state.messages {
        match msg.role.as_str() {
            "user" if first_user.is_none() => {
                first_user = pick_first_user_turn(&msg.content);
            }
            "assistant" => {
                if let Some(t) = sanitize_turn(&msg.content) {
                    last_assistant = Some(t);
                }
            }
            _ => {}
        }
    }

    let updated_at = DateTime::parse_from_rfc3339(&entry.updated_at)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Some(Thread {
        cli: "code".into(),
        session_id: entry.session_id.clone(),
        source_path: store
            .session_file_path(&entry.session_id)
            .to_string_lossy()
            .to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at,
        cwd: Some(entry.cwd.clone()),
    })
}

/// Code sessions matching a session-id prefix, cwd-agnostic — the global
/// fallback for an explicit `--resume <id>` (parity with `/resume <id>`).
pub async fn code_threads_by_id_global(store: &SessionStore, id_prefix: &str) -> Vec<Thread> {
    let mut entries = match store.all_chat_sessions().await {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    entries.retain(|e| e.session_id.starts_with(id_prefix));
    entries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    let mut out = Vec::new();
    for entry in entries {
        if let Some(thread) = extract_code_thread(store, &entry).await {
            out.push(thread);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Filesystem walking — newest-first
// ---------------------------------------------------------------------------

/// File mtime, `UNIX_EPOCH` when unreadable (sorts oldest; age-filtered out).
async fn file_mtime(path: &Path) -> SystemTime {
    fs::metadata(path)
        .await
        .ok()
        .and_then(|m| m.modified().ok())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

pub(crate) async fn list_jsonl_newest_first(dir: &Path, after: Option<SystemTime>) -> Vec<PathBuf> {
    let mut entries: Vec<(PathBuf, SystemTime)> = Vec::new();
    let mut read_dir = match fs::read_dir(dir).await {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        let mtime = entry
            .metadata()
            .await
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if let Some(c) = after
            && mtime < c
        {
            continue;
        }
        entries.push((path, mtime));
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

pub(crate) async fn walk_jsonl_newest_first(
    root: &Path,
    after: Option<SystemTime>,
) -> Vec<PathBuf> {
    let mut entries: Vec<(PathBuf, SystemTime)> = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let mut rd = match fs::read_dir(&dir).await {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                let mtime = entry
                    .metadata()
                    .await
                    .and_then(|m| m.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                if let Some(c) = after
                    && mtime < c
                {
                    continue;
                }
                entries.push((path, mtime));
            }
        }
    }
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

/// Encodes an absolute directory path using Claude Code's convention:
/// path separators and the Windows drive-letter colon collapse to `-`.
/// `/Users/alice/foo` → `-Users-alice-foo`, `C:\Users\alice\foo` → `C--Users-alice-foo`.
/// Caller must pass an already-canonicalized path; we don't canonicalize
/// here so this is a pure string transform suitable for sharing the
/// canonical computation.
pub fn encode_claude_dir(canonical_path: &str) -> String {
    // `std::fs::canonicalize` on Windows returns the extended-length form
    // (`\\?\C:\…` or `\\?\UNC\server\share\…`), but Claude Code encodes the
    // user-visible cwd. Strip the prefix so both sides agree.
    let stripped = canonical_path
        .strip_prefix(r"\\?\UNC\")
        .or_else(|| canonical_path.strip_prefix(r"\\?\"))
        .unwrap_or(canonical_path);

    stripped
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '-',
            other => other,
        })
        .collect()
}

/// Log when a session file/DB that was discoverable cannot be read or
/// parsed. Release builds stay silent (per-line JSONL churn and mid-write
/// files are routine); debug builds surface it so developers chasing a
/// "--resume doesn't see my session" report have a breadcrumb.
fn warn_unreadable_session(path: &Path, reason: &str) {
    #[cfg(debug_assertions)]
    eprintln!(
        "aivo: skipping unreadable session {}: {}",
        path.display(),
        reason
    );
    #[cfg(not(debug_assertions))]
    let _ = (path, reason);
}

// ---------------------------------------------------------------------------
// Per-file JSONL extractors
// ---------------------------------------------------------------------------

async fn extract_claude_thread(path: &Path) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut session_id: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;
    let mut event_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if session_id.is_none()
            && let Some(sid) = v.get("sessionId").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }

        // Claude events carry the original cwd verbatim; prefer it over the
        // lossy dir-name decoder, which collapses `.` and `/` to the same `-`.
        if event_cwd.is_none()
            && let Some(c) = v.get("cwd").and_then(|s| s.as_str())
            && !c.is_empty()
        {
            event_cwd = Some(c.to_string());
        }

        if v.get("isSidechain")
            .and_then(|b| b.as_bool())
            .unwrap_or(false)
        {
            continue;
        }

        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }

        let raw = extract_claude_text(v.get("message")).unwrap_or_default();

        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }

        match kind {
            "user" if first_user.is_none() => {
                first_user = pick_first_user_turn(&raw);
            }
            "assistant" => {
                if let Some(t) = sanitize_turn(&raw) {
                    last_assistant = Some(t);
                }
            }
            _ => {}
        }
    }

    Some(Thread {
        cli: "claude".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at: last_timestamp.unwrap_or_else(Utc::now),
        cwd: event_cwd.or_else(|| decode_claude_cwd(path)),
    })
}

/// Listing-optimized: stops once session id + cwd + first user turn are read,
/// since the `aivo logs` view shows only the topic + time — parsing the whole
/// (often multi-MB) transcript for `last_response` + the trailing timestamp is
/// waste. `updated_at` uses the walk's file mtime (≥ last-message-ts for an
/// append-only jsonl); search/`--json` callers use `extract_claude_thread`.
async fn extract_claude_thread_headline(path: &Path, mtime: SystemTime) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut first_user: Option<String> = None;
    let mut session_id: Option<String> = None;
    let mut event_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if session_id.is_none()
            && let Some(sid) = v.get("sessionId").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }
        if event_cwd.is_none()
            && let Some(c) = v.get("cwd").and_then(|s| s.as_str())
            && !c.is_empty()
        {
            event_cwd = Some(c.to_string());
        }
        if first_user.is_none()
            && !v
                .get("isSidechain")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
            && v.get("type").and_then(|t| t.as_str()) == Some("user")
        {
            let raw = extract_claude_text(v.get("message")).unwrap_or_default();
            first_user = pick_first_user_turn(&raw);
        }
        if first_user.is_some() && session_id.is_some() && event_cwd.is_some() {
            break;
        }
    }

    Some(Thread {
        cli: "claude".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: String::new(),
        updated_at: DateTime::<Utc>::from(mtime),
        cwd: event_cwd.or_else(|| decode_claude_cwd(path)),
    })
}

/// Reverse the encoded-dir convention used by Claude Code under
/// `~/.claude/projects/`. `-Users-alice-foo` → `/Users/alice/foo`. Lossy when the
/// original cwd contained literal hyphens or dots (both encode to `-`);
/// callers should prefer the event-level `cwd` field when present, and only
/// fall back to this for empty/legacy files. Windows paths (`C--Users-...`)
/// round-trip cosmetically only — kept for at-a-glance display.
fn decode_claude_cwd(path: &Path) -> Option<String> {
    let parent = path.parent()?.file_name()?.to_str()?;
    Some(parent.replace('-', "/"))
}

/// Returns Some iff the session's session_meta.cwd matches the project root.
async fn extract_codex_thread(path: &Path, project_root: &str) -> Option<Thread> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(err) => {
            warn_unreadable_session(path, &err.to_string());
            return None;
        }
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;
    let mut session_cwd: Option<String> = None;
    let mut project_matches = false;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str()) {
                session_cwd = Some(cwd.to_string());
                if paths_match(cwd, project_root) {
                    project_matches = true;
                }
            }
        }

        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }

        if kind == "response_item"
            && let Some(payload) = v.get("payload")
            && payload.get("type").and_then(|t| t.as_str()) == Some("message")
        {
            let role = payload.get("role").and_then(|s| s.as_str()).unwrap_or("");
            let raw = extract_codex_message_text(payload).unwrap_or_default();
            match role {
                "user" if first_user.is_none() => {
                    first_user = pick_first_user_turn(&raw);
                }
                "assistant" => {
                    if let Some(t) = sanitize_turn(&raw) {
                        last_assistant = Some(t);
                    }
                }
                _ => {}
            }
        }
    }

    if !project_matches {
        return None;
    }
    Some(Thread {
        cli: "codex".into(),
        session_id: session_id?,
        source_path: path.to_string_lossy().to_string(),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at: last_timestamp.unwrap_or_else(Utc::now),
        cwd: session_cwd,
    })
}

/// Pulls natural-language text from a Claude `message` field.
pub(crate) fn extract_claude_text(message: Option<&Value>) -> Option<String> {
    let content = message?.get("content")?;
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = content.as_array() {
        let mut buf = String::new();
        for block in arr {
            if block.get("type").and_then(|t| t.as_str()) == Some("text")
                && let Some(t) = block.get("text").and_then(|t| t.as_str())
            {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(t);
            }
        }
        if !buf.is_empty() {
            return Some(buf);
        }
    }
    None
}

/// Pulls natural-language text from a Codex `response_item.payload`.
pub(crate) fn extract_codex_message_text(payload: &Value) -> Option<String> {
    let arr = payload.get("content")?.as_array()?;
    let mut buf = String::new();
    for block in arr {
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if !(kind == "input_text" || kind == "output_text" || kind == "text") {
            continue;
        }
        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(t);
        }
    }
    if buf.is_empty() { None } else { Some(buf) }
}

/// Scrub escapes and controls (newlines survive) so transcript content —
/// e.g. an echoed `/model` stdout with color codes — can't paint live
/// escapes into list rows or digest text.
fn scrub_ansi_and_controls(s: &str) -> Cow<'_, str> {
    ansi::scrub(s, ansi::ControlPolicy::DropExceptNewlines)
}

/// Pick the first user turn for use as a session title: strip aivo-context
/// echo and skip turns dominated by CLI-harness boilerplate
/// (`<environment_context>`, `<local-command-caveat>`, …) — those aren't
/// what the user said. Short prompts like "hi" are kept as the row title;
/// a length floor here would hide CJK sessions (a full sentence rarely
/// reaches `MIN_TURN_CHARS`) and mistitle others with a later long turn.
pub(crate) fn pick_first_user_turn(raw: &str) -> Option<String> {
    let cleaned = strip_aivo_context(raw);
    let cleaned = scrub_ansi_and_controls(cleaned);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_lowercase();
    if BOILERPLATE_MARKERS.iter().any(|m| lower.contains(m)) {
        return None;
    }
    Some(truncate_chars(trimmed, MAX_TURN_CHARS))
}

fn is_substantive(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.chars().count() < MIN_TURN_CHARS {
        return false;
    }
    let lower = trimmed.to_lowercase();
    if BOILERPLATE_MARKERS.iter().any(|m| lower.contains(m)) {
        return false;
    }
    true
}

/// Strip any echoed aivo-context payload, then cap at `MAX_TURN_CHARS`.
/// Returns None if what's left after stripping isn't substantive.
pub(crate) fn sanitize_turn(text: &str) -> Option<String> {
    let cleaned = strip_aivo_context(text);
    let cleaned = scrub_ansi_and_controls(cleaned);
    let trimmed = cleaned.trim();
    if !is_substantive(trimmed) {
        return None;
    }
    Some(truncate_chars(trimmed, MAX_TURN_CHARS))
}

fn strip_aivo_context(text: &str) -> &str {
    let mut earliest = text.len();
    for marker in AIVO_CONTEXT_MARKERS {
        if let Some(pos) = text.find(marker) {
            earliest = earliest.min(pos);
        }
    }
    &text[..earliest]
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let prefix: String = text.chars().take(max_chars.saturating_sub(1)).collect();
    format!("{}…", prefix)
}

pub(crate) fn paths_match(a: &str, b: &str) -> bool {
    let norm = |s: &str| {
        let trimmed = s.trim_end_matches('/');
        std::fs::canonicalize(trimmed)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| trimmed.to_string())
    };
    norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn stored_msg(role: &str, content: &str) -> crate::services::session_store::StoredChatMessage {
        crate::services::session_store::StoredChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
            model: None,
        }
    }

    async fn save_code_session(
        store: &SessionStore,
        cwd: &str,
        session_id: &str,
        messages: &[crate::services::session_store::StoredChatMessage],
    ) {
        store
            .save_code_session_with_id(
                "key1",
                "https://api.example.com",
                cwd,
                session_id,
                "test-model",
                None,
                messages,
                "title",
                "preview",
                crate::services::session_store::SessionTokens::default(),
                0.0,
            )
            .await
            .expect("save code session");
    }

    #[tokio::test]
    async fn ingest_code_returns_project_sessions_only() {
        let config_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(config_dir.path().join("config.json"));
        let project = TempDir::new().unwrap();
        let cwd = std::fs::canonicalize(project.path())
            .unwrap()
            .to_string_lossy()
            .to_string();

        save_code_session(
            &store,
            &cwd,
            "sess-in-project",
            &[
                stored_msg(
                    "user",
                    "please wire the payments webhook retry queue with backoff",
                ),
                stored_msg(
                    "assistant",
                    "added exponential backoff with a dead-letter queue after five attempts",
                ),
            ],
        )
        .await;
        save_code_session(
            &store,
            "/somewhere/else",
            "sess-elsewhere",
            &[
                stored_msg(
                    "user",
                    "a completely unrelated task in a different project dir",
                ),
                stored_msg(
                    "assistant",
                    "done — that other project's task is finished now",
                ),
            ],
        )
        .await;

        let threads = ingest_code(&store, &cwd, None).await;
        assert_eq!(threads.len(), 1);
        let t = &threads[0];
        assert_eq!(t.cli, "code");
        assert_eq!(t.session_id, "sess-in-project");
        assert!(t.topic.contains("webhook retry queue"));
        assert!(t.last_response.contains("exponential backoff"));
        assert_eq!(t.cwd.as_deref(), Some(cwd.as_str()));
    }

    #[tokio::test]
    async fn code_threads_by_id_global_ignores_cwd() {
        let config_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(config_dir.path().join("config.json"));

        save_code_session(
            &store,
            "/project/one",
            "aaaa1111-one",
            &[
                stored_msg("user", "first"),
                stored_msg("assistant", "ok one"),
            ],
        )
        .await;
        save_code_session(
            &store,
            "/project/two",
            "bbbb2222-two",
            &[
                stored_msg("user", "second"),
                stored_msg("assistant", "ok two"),
            ],
        )
        .await;

        // A prefix from a foreign dir resolves — the `--resume <id>` fallback.
        let hits = code_threads_by_id_global(&store, "bbbb2222").await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id, "bbbb2222-two");

        assert!(
            code_threads_by_id_global(&store, "cccc3333")
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn ingest_code_drops_context_seeded_one_shots() {
        // A `-p` one-shot launched with a context digest stores the injected block
        // at the head of its first user message; re-ingesting it must not
        // produce context-in-context recursion.
        let config_dir = TempDir::new().unwrap();
        let store = SessionStore::with_path(config_dir.path().join("config.json"));
        let project = TempDir::new().unwrap();
        let cwd = std::fs::canonicalize(project.path())
            .unwrap()
            .to_string_lossy()
            .to_string();

        save_code_session(
            &store,
            &cwd,
            "sess-seeded",
            &[
                stored_msg(
                    "user",
                    "# aivo context\n\nCross-tool context from one past session.\n\nfix the login bug",
                ),
                stored_msg("assistant", "the login bug is fixed — the token refresh now retries"),
            ],
        )
        .await;

        let threads = ingest_code(&store, &cwd, None).await;
        assert!(threads.is_empty());
    }

    #[test]
    fn encode_claude_dir_uses_hyphens() {
        assert_eq!(
            encode_claude_dir("/Users/alice/project/aivo"),
            "-Users-alice-project-aivo"
        );
    }

    #[test]
    fn encode_claude_dir_handles_windows_paths() {
        assert_eq!(
            encode_claude_dir(r"C:\Users\alice\repo"),
            "C--Users-alice-repo"
        );
        // Rust's canonicalize emits the extended-length form on Windows;
        // strip it so we match what Claude Code actually writes.
        assert_eq!(
            encode_claude_dir(r"\\?\C:\Users\alice\repo"),
            "C--Users-alice-repo"
        );
        assert_eq!(
            encode_claude_dir(r"\\?\UNC\server\share\repo"),
            "server-share-repo"
        );
    }

    #[test]
    fn strip_aivo_context_cuts_at_earliest_marker() {
        let text = "Here's the plan for today.\n\n# aivo memory\n## [claude] ...rest";
        assert_eq!(
            strip_aivo_context(text).trim(),
            "Here's the plan for today."
        );

        let xml = "Acknowledged. <aivo_memory>blah</aivo_memory> done.";
        assert_eq!(strip_aivo_context(xml).trim(), "Acknowledged.");

        let plain = "Hello, world.";
        assert_eq!(strip_aivo_context(plain), plain);
    }

    #[test]
    fn sanitize_turn_strips_recursion_and_truncates() {
        let echoed = format!(
            "{} {}",
            "I'll acknowledge and wait for your next message.",
            "# aivo memory\n## [claude] old stuff\n".repeat(50)
        );
        let cleaned = sanitize_turn(&echoed).expect("substantive after strip");
        assert!(!cleaned.contains("# aivo memory"));
        assert!(cleaned.starts_with("I'll acknowledge"));

        let long = "a".repeat(1000);
        let capped = sanitize_turn(&long).unwrap();
        assert!(capped.chars().count() <= MAX_TURN_CHARS);
        assert!(capped.ends_with('…'));
    }

    #[test]
    fn sanitize_turn_returns_none_when_echo_is_entire_content() {
        let only_echo = "# aivo memory\n## [claude] old\n**Topic:** x\n**Last response:** y";
        assert!(sanitize_turn(only_echo).is_none());
    }

    #[test]
    fn is_substantive_skips_short_and_boilerplate() {
        assert!(!is_substantive("ok"));
        assert!(!is_substantive("<local-command-caveat>Some short caveat"));
        assert!(is_substantive(
            "Please review the pagination approach in handlers/users.go"
        ));
    }

    #[test]
    fn pick_first_user_turn_skips_local_command_stdout() {
        // A `/model` echo is harness output, not the user's intent — it must
        // never become a session title.
        let echo = "<local-command-stdout>Set model to \x1b[1mFable 5\x1b[22m and saved as your default</local-command-stdout>";
        assert!(pick_first_user_turn(echo).is_none());
    }

    #[test]
    fn pick_first_user_turn_scrubs_ansi_escapes() {
        let colored =
            "please look at the \x1b[1mbold\x1b[22m rendering issue in the footer status line";
        let picked = pick_first_user_turn(colored).unwrap();
        assert_eq!(
            picked,
            "please look at the bold rendering issue in the footer status line"
        );
    }

    #[test]
    fn pick_first_user_turn_keeps_short_prompts() {
        // No length floor — short prompts stay visible on listing surfaces.
        let short = "fix ci error";
        assert_eq!(pick_first_user_turn(short).as_deref(), Some(short));
    }

    #[test]
    fn extract_claude_text_handles_string_and_array() {
        let as_string: Value = serde_json::from_str(r#"{"content":"hello world"}"#).unwrap();
        assert_eq!(
            extract_claude_text(Some(&as_string)).unwrap(),
            "hello world"
        );

        let as_array: Value = serde_json::from_str(
            r#"{"content":[{"type":"text","text":"a"},{"type":"tool_use"},{"type":"text","text":"b"}]}"#,
        )
        .unwrap();
        assert_eq!(extract_claude_text(Some(&as_array)).unwrap(), "a\nb");

        let nothing: Value = serde_json::from_str(r#"{"content":[{"type":"tool_use"}]}"#).unwrap();
        assert!(extract_claude_text(Some(&nothing)).is_none());
    }

    #[tokio::test]
    async fn extract_claude_thread_picks_first_user_and_last_assistant() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            // Boilerplate first turn is skipped; the real prompt wins.
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","message":{"content":"<command-name>/clear</command-name>"}}"#,
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:01:00Z","message":{"content":"Please explain how the cursor pagination helper in handlers/users.go handles empty cursors."}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":true,"timestamp":"2026-04-01T10:02:00Z","message":{"content":[{"type":"text","text":"SHOULD NOT APPEAR"}]}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:03:00Z","message":{"content":[{"type":"text","text":"Empty cursors default to the first page — see the helper at line 42."}]}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:05:00Z","message":{"content":[{"type":"text","text":"Final recommendation: treat null cursor as an explicit 'start from page 1' request."}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let thread = extract_claude_thread(&path).await.expect("should extract");
        assert_eq!(thread.cli, "claude");
        assert_eq!(thread.session_id, "sess1");
        assert!(thread.topic.starts_with("Please explain how the cursor"));
        assert!(thread.last_response.starts_with("Final recommendation"));
        assert!(!thread.last_response.contains("SHOULD NOT APPEAR"));
    }

    #[tokio::test]
    async fn extract_claude_thread_returns_none_without_user_turns() {
        // Only harness boilerplate — no title to extract, so no thread.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"message":{"content":"<command-name>/clear</command-name>"}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"message":{"content":"ok"}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();
        assert!(extract_claude_thread(&path).await.is_none());
    }

    #[tokio::test]
    async fn extract_claude_thread_keeps_short_turns() {
        // `hi`/`ok` sessions show up in listings instead of vanishing under
        // a length floor.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","cwd":"/tmp/p","message":{"content":"hi"}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:05Z","message":{"content":[{"type":"text","text":"Hi!"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let thread = extract_claude_thread(&path)
            .await
            .expect("extractor keeps short-turn sessions");
        assert_eq!(thread.session_id, "sess1");
        assert_eq!(thread.topic, "hi");
    }

    #[tokio::test]
    async fn extract_claude_session_stub_works_on_short_turns() {
        // The stub variant finds the sessionId without reading turns, so the
        // share resolver can surface even sessions with no extractable title.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","cwd":"/tmp/p","message":{"content":"say hi"}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:05Z","message":{"content":[{"type":"text","text":"Hi!"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let stub = extract_claude_session_stub(&path)
            .await
            .expect("stub extractor keeps short sessions");
        assert_eq!(stub.cli, "claude");
        assert_eq!(stub.session_id, "sess1");
        assert_eq!(stub.cwd.as_deref(), Some("/tmp/p"));
        assert!(stub.topic.is_empty());
        assert!(stub.last_response.is_empty());
        assert_eq!(stub.updated_at.to_rfc3339(), "2026-04-01T10:00:05+00:00");
    }

    #[tokio::test]
    async fn extract_claude_session_stub_returns_none_without_session_id() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        fs::write(&path, r#"{"type":"queue-operation","content":"hi"}"#)
            .await
            .unwrap();
        assert!(extract_claude_session_stub(&path).await.is_none());
    }

    #[tokio::test]
    async fn extract_codex_thread_matches_by_cwd_and_extracts_bracket() {
        let dir = TempDir::new().unwrap();
        let project_root = dir.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let proj_str = project_root.to_string_lossy().to_string();
        let proj_json = proj_str.replace('\\', "\\\\");

        let path = dir.path().join("rollout.jsonl");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-04-01T10:00:00Z","payload":{{"id":"codex-1","cwd":"{}","timestamp":"2026-04-01T10:00:00Z"}}}}"#,
                proj_json
            ),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:01:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Review the pagination patch I made in handlers/users.go please."}]}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:02:00Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Looks good overall. Consider handling empty cursor by returning the first page explicitly."}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let thread = extract_codex_thread(&path, &proj_str)
            .await
            .expect("matched cwd");
        assert_eq!(thread.cli, "codex");
        assert_eq!(thread.session_id, "codex-1");
        assert!(thread.topic.contains("pagination"));
        assert!(thread.last_response.contains("first page"));
    }

    #[tokio::test]
    async fn extract_codex_thread_skips_non_matching_cwd() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rollout.jsonl");
        let lines = [
            r#"{"type":"session_meta","payload":{"id":"codex-1","cwd":"/nope/elsewhere"}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Lorem ipsum dolor sit amet consectetur"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();
        assert!(extract_codex_thread(&path, "/not/matching").await.is_none());
    }

    #[tokio::test]
    async fn list_jsonl_newest_first_orders_by_mtime_desc() {
        let dir = TempDir::new().unwrap();
        let old = dir.path().join("old.jsonl");
        let new = dir.path().join("new.jsonl");
        fs::write(&old, "x").await.unwrap();
        // Simulate the newer file by touching it after a small sleep.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        fs::write(&new, "y").await.unwrap();

        let ordered = list_jsonl_newest_first(dir.path(), None).await;
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].file_name().unwrap(), "new.jsonl");
        assert_eq!(ordered[1].file_name().unwrap(), "old.jsonl");
    }

    #[tokio::test]
    async fn list_jsonl_newest_first_skips_files_older_than_cutoff() {
        let dir = TempDir::new().unwrap();
        let old = dir.path().join("old.jsonl");
        let new = dir.path().join("new.jsonl");
        fs::write(&old, "x").await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let cutoff = SystemTime::now();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        fs::write(&new, "y").await.unwrap();

        let filtered = list_jsonl_newest_first(dir.path(), Some(cutoff)).await;
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].file_name().unwrap(), "new.jsonl");
    }

    #[test]
    fn effective_cutoff_picks_more_restrictive_bound() {
        // max_age_days = 1 ⇒ cutoff ≈ now - 1d
        // min_updated_at = now - 30d
        // The 1-day bound is later (more restrictive) and wins.
        let recent = IngestOptions {
            max_age_days: Some(1),
            min_updated_at: Some(Utc::now() - chrono::Duration::days(30)),
            ..IngestOptions::unlimited()
        };
        let cutoff = effective_cutoff(&recent).unwrap();
        assert!(cutoff > Utc::now() - chrono::Duration::days(2));

        // Reverse: explicit min_updated_at is more restrictive than days bound.
        let explicit = IngestOptions {
            max_age_days: Some(30),
            min_updated_at: Some(Utc::now() - chrono::Duration::days(1)),
            ..IngestOptions::unlimited()
        };
        let cutoff = effective_cutoff(&explicit).unwrap();
        assert!(cutoff > Utc::now() - chrono::Duration::days(2));
    }

    #[test]
    fn effective_cutoff_none_when_both_unset() {
        assert!(effective_cutoff(&IngestOptions::unlimited()).is_none());
    }

    /// `list_codex_sessions_for_cwd` must include a session whose only user
    /// turn is a short prompt, alongside an older longer session in the same
    /// cwd — else the resolver's closest-mtime pick resolves to the wrong
    /// session.
    #[tokio::test]
    async fn list_codex_sessions_for_cwd_includes_short_prompt_session() {
        let codex_root = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        let project_path = std::fs::canonicalize(project.path()).unwrap();
        let project_str = project_path.to_string_lossy().to_string();

        // Older, longer session in the same cwd — must appear alongside the new one.
        let day_old = codex_root.path().join("2026").join("05").join("01");
        std::fs::create_dir_all(&day_old).unwrap();
        let old_file = day_old.join("rollout-old.jsonl");
        let long_text = "x".repeat(120);
        let old_jsonl = format!(
            "{}\n{}\n",
            serde_json::json!({
                "timestamp": "2026-05-01T00:00:00Z",
                "type": "session_meta",
                "payload": {"id": "OLD-ID", "cwd": project_str}
            }),
            serde_json::json!({
                "timestamp": "2026-05-01T00:00:10Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": long_text}]}
            })
        );
        fs::write(&old_file, old_jsonl).await.unwrap();

        // Today's session in the same cwd with a short prompt — the case
        // the user-reported bug exercised.
        let day_today = codex_root.path().join("2026").join("05").join("12");
        std::fs::create_dir_all(&day_today).unwrap();
        let new_file = day_today.join("rollout-new.jsonl");
        let new_jsonl = format!(
            "{}\n{}\n",
            serde_json::json!({
                "timestamp": "2026-05-12T03:26:40Z",
                "type": "session_meta",
                "payload": {"id": "NEW-ID", "cwd": project_str}
            }),
            serde_json::json!({
                "timestamp": "2026-05-12T03:26:41Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user",
                            "content": [{"type": "input_text", "text": "say hi"}]}
            })
        );
        fs::write(&new_file, new_jsonl).await.unwrap();

        let threads = list_codex_sessions_for_cwd(codex_root.path(), project.path()).await;
        let ids: Vec<&str> = threads.iter().map(|t| t.session_id.as_str()).collect();
        assert!(
            ids.contains(&"NEW-ID"),
            "short-prompt session must be included; got {ids:?}"
        );
        assert!(
            ids.contains(&"OLD-ID"),
            "long-prompt session must still be included; got {ids:?}"
        );
    }

    /// `paths_match` excludes sessions that recorded a different cwd —
    /// the prefix-collision bug only mattered because the resolver was
    /// scoped to a cwd; that scoping must keep working with the new
    /// non-filtering enumerator.
    #[tokio::test]
    async fn list_codex_sessions_for_cwd_skips_other_cwds() {
        let codex_root = TempDir::new().unwrap();
        let mine = TempDir::new().unwrap();
        let theirs = TempDir::new().unwrap();
        let day = codex_root.path().join("2026").join("05").join("12");
        std::fs::create_dir_all(&day).unwrap();
        let theirs_str = std::fs::canonicalize(theirs.path())
            .unwrap()
            .to_string_lossy()
            .to_string();
        let jsonl = format!(
            "{}\n",
            serde_json::json!({
                "timestamp": "2026-05-12T03:26:40Z",
                "type": "session_meta",
                "payload": {"id": "OTHER-CWD", "cwd": theirs_str}
            })
        );
        fs::write(day.join("rollout.jsonl"), jsonl).await.unwrap();

        let threads = list_codex_sessions_for_cwd(codex_root.path(), mine.path()).await;
        assert!(
            threads.is_empty(),
            "session in a different cwd must not appear; got {:?}",
            threads.iter().map(|t| &t.session_id).collect::<Vec<_>>()
        );
    }

    /// `list_pi_sessions_for_cwd` must include a live pi session whose only
    /// turns so far are short prompts (`hi`), alongside an older longer session
    /// in the same cwd — else the resolver's closest-mtime pick resolves both
    /// run events to the same stale session.
    ///
    /// Unix-only: pi's per-cwd dir encoding (`list_pi_sessions_for_cwd`)
    /// uses a Unix-path-shaped scheme (`trim '/'`, `replace '/' → '-'`).
    /// Windows-canonicalized paths (`\\?\C:\…`) yield directory names with
    /// `\`, `:`, `?` — invalid on NTFS — so the test can't even set up its
    /// fixture there. Pi itself runs Unix-only so production never hits this.
    #[cfg(unix)]
    #[tokio::test]
    async fn list_pi_sessions_for_cwd_includes_short_prompt_session() {
        let pi_root = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        let project_path = std::fs::canonicalize(project.path()).unwrap();
        let project_str = project_path.to_string_lossy().to_string();
        let encoded = format!("--{}--", project_str.trim_matches('/').replace('/', "-"));
        let session_dir = pi_root.path().join(&encoded);
        std::fs::create_dir_all(&session_dir).unwrap();

        // Older, longer session in the same cwd — must appear alongside the new one.
        let long_text = "x".repeat(120);
        let old_jsonl = format!(
            "{}\n{}\n",
            serde_json::json!({"type": "session", "id": "OLD-PI"}),
            serde_json::json!({
                "type": "message",
                "timestamp": "2026-05-06T00:00:00Z",
                "message": {"role": "user",
                            "content": [{"type": "text", "text": long_text}]}
            })
        );
        let old_path = session_dir.join("old.jsonl");
        fs::write(&old_path, old_jsonl).await.unwrap();

        // Today's short-prompt session — the bug case.
        let new_jsonl = format!(
            "{}\n{}\n",
            serde_json::json!({"type": "session", "id": "NEW-PI"}),
            serde_json::json!({
                "type": "message",
                "timestamp": "2026-05-12T03:26:40Z",
                "message": {"role": "user",
                            "content": [{"type": "text", "text": "hi"}]}
            })
        );
        let new_path = session_dir.join("new.jsonl");
        fs::write(&new_path, new_jsonl).await.unwrap();

        let threads = list_pi_sessions_for_cwd(pi_root.path(), project.path()).await;
        let ids: Vec<&str> = threads.iter().map(|t| t.session_id.as_str()).collect();
        assert!(
            ids.contains(&"NEW-PI"),
            "short-prompt session must be included; got {ids:?}"
        );
        assert!(
            ids.contains(&"OLD-PI"),
            "long-prompt session must still be included; got {ids:?}"
        );
    }

    /// `list_gemini_sessions_for_cwd` must include a short-prompt session
    /// alongside an older longer one in the same cwd — same closest-mtime
    /// fall-back symptom as claude/codex/pi.
    #[tokio::test]
    async fn list_gemini_sessions_for_cwd_includes_short_prompt_session() {
        let gemini_tmp = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        let project_path = std::fs::canonicalize(project.path()).unwrap();
        let project_str = project_path.to_string_lossy().to_string();
        let project_hash = hex_sha256(project_str.as_bytes());

        let chats_dir = gemini_tmp.path().join(&project_hash).join("chats");
        std::fs::create_dir_all(&chats_dir).unwrap();

        // Older, longer session in the same cwd — must appear alongside the new one.
        let long_text = "x".repeat(120);
        let old_file = chats_dir.join("session-old.json");
        let old_json = serde_json::json!({
            "projectHash": project_hash,
            "sessionId": "OLD-G",
            "lastUpdated": "2026-05-06T00:00:00Z",
            "messages": [
                {"type": "user", "timestamp": "2026-05-06T00:00:00Z",
                 "content": [{"text": long_text}]}
            ]
        });
        fs::write(&old_file, old_json.to_string()).await.unwrap();

        // Today's short-prompt session — the bug case.
        let new_file = chats_dir.join("session-new.json");
        let new_json = serde_json::json!({
            "projectHash": project_hash,
            "sessionId": "NEW-G",
            "lastUpdated": "2026-05-12T03:26:40Z",
            "messages": [
                {"type": "user", "timestamp": "2026-05-12T03:26:40Z",
                 "content": [{"text": "hi"}]}
            ]
        });
        fs::write(&new_file, new_json.to_string()).await.unwrap();

        let threads = list_gemini_sessions_for_cwd(gemini_tmp.path(), project.path()).await;
        let ids: Vec<&str> = threads.iter().map(|t| t.session_id.as_str()).collect();
        assert!(ids.contains(&"NEW-G"), "short session missing; got {ids:?}");
        assert!(ids.contains(&"OLD-G"), "long session missing; got {ids:?}");
    }

    /// Regression: gemini-cli switched its session storage from one-JSON-per-file
    /// (`session-*.json`) to JSONL (`session-*.jsonl`, header line + message
    /// lines + `$set` patches) in early 2026. The ingester used to filter on
    /// `.json` only and parse with `serde_json::from_str`, so every recent
    /// session vanished from `aivo logs`. Both formats must now be ingested.
    #[tokio::test]
    async fn list_gemini_sessions_for_cwd_includes_jsonl_format() {
        let gemini_tmp = TempDir::new().unwrap();
        let project = TempDir::new().unwrap();
        let project_path = std::fs::canonicalize(project.path()).unwrap();
        let project_str = project_path.to_string_lossy().to_string();
        let project_hash = hex_sha256(project_str.as_bytes());

        let chats_dir = gemini_tmp.path().join(&project_hash).join("chats");
        std::fs::create_dir_all(&chats_dir).unwrap();

        let jsonl_path = chats_dir.join("session-new.jsonl");
        let header = serde_json::json!({
            "sessionId": "JSONL-G",
            "projectHash": project_hash,
            "startTime": "2026-05-22T02:37:02.231Z",
            "lastUpdated": "2026-05-22T02:37:02.231Z",
            "kind": "main",
        });
        let user_msg = serde_json::json!({
            "id": "u1",
            "timestamp": "2026-05-22T02:37:58.725Z",
            "type": "user",
            "content": [{"text": "say hi in 5 words"}],
        });
        let set_patch = serde_json::json!({"$set": {"lastUpdated": "2026-05-22T02:37:58.726Z"}});
        let jsonl = format!("{}\n{}\n{}\n", header, user_msg, set_patch);
        fs::write(&jsonl_path, jsonl).await.unwrap();

        let threads = list_gemini_sessions_for_cwd(gemini_tmp.path(), project.path()).await;
        let ids: Vec<&str> = threads.iter().map(|t| t.session_id.as_str()).collect();
        assert!(
            ids.contains(&"JSONL-G"),
            "jsonl-format session must be ingested; got {ids:?}"
        );
    }

    /// Gemini dirs with a different `projectHash` must not leak in.
    #[tokio::test]
    async fn list_gemini_sessions_for_cwd_skips_other_cwds() {
        let gemini_tmp = TempDir::new().unwrap();
        let mine = TempDir::new().unwrap();
        let theirs = TempDir::new().unwrap();

        let theirs_str = std::fs::canonicalize(theirs.path())
            .unwrap()
            .to_string_lossy()
            .to_string();
        let theirs_hash = hex_sha256(theirs_str.as_bytes());
        let chats_dir = gemini_tmp.path().join(&theirs_hash).join("chats");
        std::fs::create_dir_all(&chats_dir).unwrap();
        let body = serde_json::json!({
            "projectHash": theirs_hash,
            "sessionId": "OTHER-CWD",
            "messages": []
        });
        fs::write(chats_dir.join("session-x.json"), body.to_string())
            .await
            .unwrap();

        let threads = list_gemini_sessions_for_cwd(gemini_tmp.path(), mine.path()).await;
        assert!(
            threads.is_empty(),
            "session in a different cwd must not appear; got {:?}",
            threads.iter().map(|t| &t.session_id).collect::<Vec<_>>()
        );
    }

    /// Regression: opencode sessions whose `message`/`part` rows are all
    /// short text used to be filtered out by `opencode_extract_session`'s
    /// sanitize_turn check. The enumerator must include them so two
    /// distinct runs don't both collapse onto the same older session.
    #[tokio::test]
    async fn list_opencode_sessions_for_cwd_includes_messageless_session() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("opencode.db");
        let project = TempDir::new().unwrap();
        let project_path = std::fs::canonicalize(project.path()).unwrap();
        let project_str = project_path.to_string_lossy().to_string();

        // Minimal opencode schema. Only `project` and `session` are touched
        // by the stub enumerator — we deliberately omit `message`/`part` to
        // prove the stub doesn't depend on them.
        tokio::task::spawn_blocking({
            let db_path = db_path.clone();
            let project_str = project_str.clone();
            move || {
                let conn = rusqlite::Connection::open(&db_path).unwrap();
                conn.execute(
                    "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT NOT NULL)",
                    [],
                )
                .unwrap();
                conn.execute(
                    "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, time_updated INTEGER NOT NULL)",
                    [],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO project (id, worktree) VALUES ('p1', ?1)",
                    [&project_str],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO session (id, project_id, time_updated) VALUES ('OLD-OC', 'p1', 1714867200000)",
                    [],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO session (id, project_id, time_updated) VALUES ('NEW-OC', 'p1', 1715472400000)",
                    [],
                )
                .unwrap();
            }
        })
        .await
        .unwrap();

        let threads = list_opencode_sessions_for_cwd(&db_path, project.path()).await;
        let ids: Vec<&str> = threads.iter().map(|t| t.session_id.as_str()).collect();
        assert!(
            ids.contains(&"NEW-OC"),
            "newer session missing; got {ids:?}"
        );
        assert!(
            ids.contains(&"OLD-OC"),
            "older session missing; got {ids:?}"
        );
    }

    /// Sessions whose `project_id` belongs to a different worktree must
    /// not leak in.
    #[tokio::test]
    async fn list_opencode_sessions_for_cwd_skips_other_cwds() {
        let temp = TempDir::new().unwrap();
        let db_path = temp.path().join("opencode.db");
        let mine = TempDir::new().unwrap();
        let theirs = TempDir::new().unwrap();
        let mine_str = std::fs::canonicalize(mine.path())
            .unwrap()
            .to_string_lossy()
            .to_string();
        let theirs_str = std::fs::canonicalize(theirs.path())
            .unwrap()
            .to_string_lossy()
            .to_string();

        tokio::task::spawn_blocking({
            let db_path = db_path.clone();
            move || {
                let conn = rusqlite::Connection::open(&db_path).unwrap();
                conn.execute(
                    "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT NOT NULL)",
                    [],
                )
                .unwrap();
                conn.execute(
                    "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, time_updated INTEGER NOT NULL)",
                    [],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO project (id, worktree) VALUES ('p1', ?1)",
                    [&mine_str],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO project (id, worktree) VALUES ('p2', ?1)",
                    [&theirs_str],
                )
                .unwrap();
                conn.execute(
                    "INSERT INTO session (id, project_id, time_updated) VALUES ('NOT-MINE', 'p2', 1715472400000)",
                    [],
                )
                .unwrap();
            }
        })
        .await
        .unwrap();

        let threads = list_opencode_sessions_for_cwd(&db_path, mine.path()).await;
        assert!(
            threads.is_empty(),
            "session in a different cwd must not appear; got {:?}",
            threads.iter().map(|t| &t.session_id).collect::<Vec<_>>()
        );
    }

    /// Different-cwd dirs must not leak in — pi's per-cwd encoded dir
    /// scoping must keep working with the new enumerator.
    ///
    /// Unix-only for the same reason as the sibling test above:
    /// the encoded-cwd scheme is Unix-path-shaped.
    #[cfg(unix)]
    #[tokio::test]
    async fn list_pi_sessions_for_cwd_skips_other_cwds() {
        let pi_root = TempDir::new().unwrap();
        let mine = TempDir::new().unwrap();
        let theirs = TempDir::new().unwrap();

        let theirs_str = std::fs::canonicalize(theirs.path())
            .unwrap()
            .to_string_lossy()
            .to_string();
        let theirs_encoded = format!("--{}--", theirs_str.trim_matches('/').replace('/', "-"));
        let theirs_dir = pi_root.path().join(&theirs_encoded);
        std::fs::create_dir_all(&theirs_dir).unwrap();
        let jsonl = format!(
            "{}\n",
            serde_json::json!({"type": "session", "id": "OTHER-CWD"})
        );
        fs::write(theirs_dir.join("a.jsonl"), jsonl).await.unwrap();

        let threads = list_pi_sessions_for_cwd(pi_root.path(), mine.path()).await;
        assert!(
            threads.is_empty(),
            "session in a different cwd must not appear; got {:?}",
            threads.iter().map(|t| &t.session_id).collect::<Vec<_>>()
        );
    }
}
