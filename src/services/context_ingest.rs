//! On-demand ingestion of AI CLI session content into normalized context
//! threads. **No persistent storage** — each call reads the authoritative
//! sources fresh and returns an in-memory thread list.
//!
//! Scope: only tools launchable via `aivo run` are sources, because
//! `--context` injects into those tools. Aivo's own chat is excluded
//! deliberately — its sessions belong to a different workflow.
//!
//! Sources:
//! - Claude Code: `~/.claude/projects/<encoded-cwd>/*.jsonl` (matched by
//!   encoded directory name, which is `/a/b/c` → `-a-b-c`).
//! - Codex: `~/.codex/sessions/YYYY/MM/DD/*.jsonl`. Per-file `session_meta`
//!   payload's `cwd` must match the project root.
//!
//! Extraction uses the bracket (first substantive user + last substantive
//! assistant) + substance filter (min chars, skip CLI boilerplate and echoed
//! aivo context). Each source is walked most-recent first and stops once
//! enough candidate threads are collected, so heavy projects don't parse
//! every historical session.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};

use crate::services::device_fingerprint::hex_sha256;
use crate::services::project_id::{DEFAULT_THREAD_MAX_AGE_DAYS, Thread};
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
    /// `Some(n)` early-exits each source after `n` extractions; `None` = all.
    pub max_per_source: Option<usize>,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self {
            max_age_days: Some(DEFAULT_THREAD_MAX_AGE_DAYS),
            max_per_source: Some(DEFAULT_MAX_THREADS_PER_SOURCE),
        }
    }
}

impl IngestOptions {
    /// Bypass both caps — used by `aivo context --all` / `-a`.
    pub fn unlimited() -> Self {
        Self {
            max_age_days: None,
            max_per_source: None,
        }
    }
}

/// Read all supported sources for the given project root, merge + dedup +
/// age-filter, and return threads newest-first. Nothing is persisted.
pub async fn ingest_project(project_root: &Path, opts: IngestOptions) -> Result<Vec<Thread>> {
    // Compute the canonical project root once and pass it down — five
    // separate `canonicalize` syscalls would otherwise hit the same path.
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let cap = opts.max_per_source;

    // Independent I/O — fan out across sources concurrently.
    let (claude, codex, gemini, pi, opencode) = tokio::join!(
        ingest_claude(&canonical_root, cap),
        ingest_codex(&canonical_str, cap),
        ingest_gemini(&canonical_str, cap),
        ingest_pi(&canonical_str, cap),
        ingest_opencode(canonical_str.clone(), cap),
    );

    let mut threads: Vec<Thread> =
        Vec::with_capacity(claude.len() + codex.len() + gemini.len() + pi.len() + opencode.len());
    threads.extend(claude);
    threads.extend(codex);
    threads.extend(gemini);
    threads.extend(pi);
    threads.extend(opencode);

    // Optional age filter (replaces the old `gc` command — evaluated lazily).
    if let Some(days) = opts.max_age_days {
        let cutoff = Utc::now() - chrono::Duration::days(days);
        threads.retain(|t| t.updated_at >= cutoff);
    }
    threads.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(threads)
}

// ---------------------------------------------------------------------------
// Claude: ~/.claude/projects/<encoded-cwd>/*.jsonl
// ---------------------------------------------------------------------------

async fn ingest_claude(canonical_root: &Path, cap: Option<usize>) -> Vec<Thread> {
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

    let files = list_jsonl_newest_first(&session_dir).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_claude_thread(&path).await {
            out.push(thread);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Codex: ~/.codex/sessions/YYYY/MM/DD/*.jsonl, per-file cwd match
// ---------------------------------------------------------------------------

async fn ingest_codex(canonical_root: &str, cap: Option<usize>) -> Vec<Thread> {
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let codex_root = home.join(".codex").join("sessions");
    if !codex_root.exists() {
        return Vec::new();
    }

    let files = walk_jsonl_newest_first(&codex_root).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_codex_thread(&path, canonical_root).await {
            out.push(thread);
        }
    }
    out
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
    let project_hash = hex_sha256(canonical_root.as_bytes());
    let tmp_root = home.join(".gemini").join("tmp");
    if !tmp_root.exists() {
        return Vec::new();
    }
    let mut entries: Vec<(PathBuf, SystemTime)> = Vec::new();
    let mut tmp_rd = match fs::read_dir(&tmp_root).await {
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
            if !name.starts_with("session-") || !name.ends_with(".json") {
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
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

async fn extract_gemini_thread(path: &Path) -> Option<Thread> {
    let content = fs::read_to_string(path).await.ok()?;
    let v: Value = serde_json::from_str(&content).ok()?;

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
        let text = match sanitize_turn(&raw) {
            Some(t) => t,
            None => continue,
        };
        if let Some(ts) = msg.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }
        match kind {
            "user" => {
                if first_user.is_none() {
                    first_user = Some(text);
                }
            }
            "gemini" => {
                last_assistant = Some(text);
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

async fn ingest_pi(canonical_root: &str, cap: Option<usize>) -> Vec<Thread> {
    let session_dir = match pi_session_dir(canonical_root) {
        Some(d) if d.exists() => d,
        _ => return Vec::new(),
    };

    let files = list_jsonl_newest_first(&session_dir).await;
    let mut out = Vec::new();
    for path in files {
        if let Some(n) = cap
            && out.len() >= n
        {
            break;
        }
        if let Some(thread) = extract_pi_thread(&path).await {
            out.push(thread);
        }
    }
    out
}

async fn extract_pi_thread(path: &Path) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
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
        let text = match sanitize_turn(&raw) {
            Some(t) => t,
            None => continue,
        };
        match role {
            "user" => {
                if first_user.is_none() {
                    first_user = Some(text);
                }
            }
            "assistant" => {
                last_assistant = Some(text);
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
    })
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
        Err(_) => return Vec::new(),
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
        if let Some(thread) = opencode_extract_session(&conn, &session_id, time_updated_ms) {
            out.push(thread);
        }
    }
    out
}

fn opencode_extract_session(
    conn: &rusqlite::Connection,
    session_id: &str,
    time_updated_ms: i64,
) -> Option<Thread> {
    // Join messages to their text parts, ordered chronologically. The first
    // `role=user` with substantive text → topic; the last `role=assistant` → last_response.
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
        let text = match sanitize_turn(raw) {
            Some(t) => t,
            None => continue,
        };
        match role.as_str() {
            "user" => {
                if first_user.is_none() {
                    first_user = Some(text);
                }
            }
            "assistant" => {
                last_assistant = Some(text);
            }
            _ => {}
        }
    }

    let updated_at =
        chrono::DateTime::<Utc>::from_timestamp_millis(time_updated_ms).unwrap_or_else(Utc::now);
    Some(Thread {
        cli: "opencode".into(),
        session_id: session_id.to_string(),
        source_path: format!("db://opencode/{session_id}"),
        topic: first_user?,
        last_response: last_assistant.unwrap_or_default(),
        updated_at,
    })
}

// ---------------------------------------------------------------------------
// Filesystem walking — newest-first
// ---------------------------------------------------------------------------

pub(crate) async fn list_jsonl_newest_first(dir: &Path) -> Vec<PathBuf> {
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
        entries.push((path, mtime));
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

pub(crate) async fn walk_jsonl_newest_first(root: &Path) -> Vec<PathBuf> {
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
                entries.push((path, mtime));
            }
        }
    }
    entries.sort_by(|a, b| b.1.cmp(&a.1));
    entries.into_iter().map(|(p, _)| p).collect()
}

/// Encodes an absolute directory path using Claude Code's convention:
/// `/` → `-`. `/Users/alice/foo` → `-Users-alice-foo`. Caller must pass an
/// already-canonicalized path; we don't canonicalize here so this is a pure
/// string transform suitable for sharing the canonical computation.
pub fn encode_claude_dir(canonical_path: &str) -> String {
    canonical_path.replace('/', "-")
}

// ---------------------------------------------------------------------------
// Per-file JSONL extractors
// ---------------------------------------------------------------------------

async fn extract_claude_thread(path: &Path) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();

    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut session_id: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;

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
        let text = match sanitize_turn(&raw) {
            Some(t) => t,
            None => continue,
        };

        if let Some(ts) = v.get("timestamp").and_then(|s| s.as_str())
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
        {
            last_timestamp = Some(parsed.with_timezone(&Utc));
        }

        match kind {
            "user" => {
                if first_user.is_none() {
                    first_user = Some(text);
                }
            }
            "assistant" => {
                last_assistant = Some(text);
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
    })
}

/// Returns Some iff the session's session_meta.cwd matches the project root.
async fn extract_codex_thread(path: &Path, project_root: &str) -> Option<Thread> {
    let file = fs::File::open(path).await.ok()?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut first_user: Option<String> = None;
    let mut last_assistant: Option<String> = None;
    let mut last_timestamp: Option<DateTime<Utc>> = None;
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
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str())
                && paths_match(cwd, project_root)
            {
                project_matches = true;
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
            let text = match sanitize_turn(&raw) {
                Some(t) => t,
                None => continue,
            };
            match role {
                "user" => {
                    if first_user.is_none() {
                        first_user = Some(text);
                    }
                }
                "assistant" => {
                    last_assistant = Some(text);
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
fn sanitize_turn(text: &str) -> Option<String> {
    let cleaned = strip_aivo_context(text);
    let trimmed = cleaned.trim();
    if !is_substantive(trimmed) {
        return None;
    }
    Some(truncate_chars(trimmed, MAX_TURN_CHARS))
}

fn strip_aivo_context(text: &str) -> String {
    let mut earliest = text.len();
    for marker in AIVO_CONTEXT_MARKERS {
        if let Some(pos) = text.find(marker) {
            earliest = earliest.min(pos);
        }
    }
    text[..earliest].to_string()
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

    #[test]
    fn encode_claude_dir_uses_hyphens() {
        assert_eq!(
            encode_claude_dir("/Users/alice/project/aivo"),
            "-Users-alice-project-aivo"
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
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","message":{"content":"hi"}}"#,
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
    async fn extract_claude_thread_returns_none_without_substantive_turns() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sess1","isSidechain":false,"message":{"content":"hi"}}"#,
            r#"{"type":"assistant","sessionId":"sess1","isSidechain":false,"message":{"content":"ok"}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();
        assert!(extract_claude_thread(&path).await.is_none());
    }

    #[tokio::test]
    async fn extract_codex_thread_matches_by_cwd_and_extracts_bracket() {
        let dir = TempDir::new().unwrap();
        let project_root = dir.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let proj_str = project_root.to_string_lossy().to_string();

        let path = dir.path().join("rollout.jsonl");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-04-01T10:00:00Z","payload":{{"id":"codex-1","cwd":"{}","timestamp":"2026-04-01T10:00:00Z"}}}}"#,
                proj_str
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

        let ordered = list_jsonl_newest_first(dir.path()).await;
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].file_name().unwrap(), "new.jsonl");
        assert_eq!(ordered[1].file_name().unwrap(), "old.jsonl");
    }
}
