//! Verbatim recent-turns extraction for the cross-tool MCP bridge.
//!
//! Unlike `context_ingest`, which returns a compressed `(topic, last_response)`
//! pair per session for the `--context` flow, this module returns a
//! chronological list of natural-language turns with no text truncation beyond
//! a per-turn safety cap. Consumers are MCP tools (`list_sessions`,
//! `get_session`) exposed by `aivo mcp-serve`, which Claude/Codex call to
//! inspect each other's in-flight conversations.
//!
//! Design notes:
//! - Skip Claude `isSidechain=true` turns (agent-within-agent noise).
//! - Skip tool-use / tool-result blocks; keep only natural-language text.
//! - Per-turn cap of 8 KB guards against multi-MB code pastes blowing up the
//!   MCP response envelope. Top-level `max_turns` cap is applied by the server.
//! - Silently skip unparseable JSONL lines — a partial/streaming last line
//!   from a peer tool that's still writing is normal, not an error.
//!
//! Pi, Gemini, and OpenCode are supported as *queryable peers only*:
//! Claude/Codex can read their transcripts via this module, but those tools
//! cannot inject the aivo MCP server themselves — Pi has no MCP client hook
//! upstream, Gemini's MCP config is persistent (`~/.gemini/settings.json`)
//! which conflicts with the no-persistent-state design here, and OpenCode
//! client wiring is deferred until an explicit need arises.

use std::path::Path;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::services::context_ingest::{
    encode_claude_dir, extract_claude_text, extract_codex_message_text, extract_gemini_content,
    extract_pi_text, gemini_matching_session_files, list_jsonl_newest_first, paths_match,
    pi_session_dir, walk_jsonl_newest_first,
};
use crate::services::system_env;

/// Hard cap on a single turn's text payload. Longer turns are truncated with
/// `…` to protect the MCP response from multi-MB pastes.
const MAX_TURN_BYTES: usize = 8 * 1024;

/// A single conversational turn, in chronological order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Turn {
    /// "user" or "assistant".
    pub role: String,
    /// Verbatim text, capped at `MAX_TURN_BYTES`.
    pub text: String,
    /// RFC 3339 timestamp when available.
    pub timestamp: Option<DateTime<Utc>>,
}

/// A full transcript of one session: last N turns, chronological.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Transcript {
    /// "claude", "codex", "pi", "gemini", or "opencode".
    pub cli: String,
    /// Native session id.
    pub session_id: String,
    /// Source JSONL path for provenance.
    pub source_path: String,
    /// Turns in chronological order (oldest first), bounded by `max_turns`.
    pub turns: Vec<Turn>,
    /// Most recent turn's timestamp, if any (for sorting by freshness).
    pub updated_at: Option<DateTime<Utc>>,
}

/// Resolve a session for the given CLI, optionally by id prefix, and load its
/// recent turns. Returns `None` if no matching session exists for this cwd.
///
/// - `cli`: "claude", "codex", "pi", "gemini", or "opencode". Other values
///   return `Ok(None)`.
/// - `session_id`: `None` → most-recent for this project; `Some(prefix)` →
///   prefix-match on the native session id.
/// - `exclude_session_ids`: any transcript whose `session_id` starts with
///   one of these strings is skipped. Used to skip the caller's own
///   session in same-CLI peer queries (the calling tool is actively
///   writing its own file, so without this it would be the newest match).
/// - `max_turns`: cap on the number of turns returned (chronologically the
///   last N).
pub async fn resolve_session(
    project_root: &Path,
    cli: &str,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    match cli {
        "claude" => {
            resolve_claude(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        "codex" => {
            resolve_codex(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        "pi" => {
            resolve_pi(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        "gemini" => {
            resolve_gemini(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        "opencode" => {
            resolve_opencode(
                project_root,
                session_id,
                exclude_session_ids,
                started_after,
                max_turns,
            )
            .await
        }
        _ => Ok(None),
    }
}

/// Returns true if `session_id` starts with any exclude prefix.
fn is_excluded(session_id: &str, exclude_prefixes: &[String]) -> bool {
    exclude_prefixes.iter().any(|p| session_id.starts_with(p))
}

/// Shared post-load filter for file-based resolvers: reject by session-id
/// prefix mismatch, exclude list, or a `started_after` cutoff.
fn matches_filters(
    transcript: &Transcript,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
) -> bool {
    if let Some(prefix) = session_id
        && !transcript.session_id.starts_with(prefix)
    {
        return false;
    }
    if is_excluded(&transcript.session_id, exclude_session_ids) {
        return false;
    }
    if let Some(cutoff) = started_after
        && let Some(updated) = transcript.updated_at
        && updated < cutoff
    {
        return false;
    }
    true
}

/// Find matching claude JSONL file(s) and extract the first/best transcript.
async fn resolve_claude(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };
    let session_dir = home
        .join(".claude")
        .join("projects")
        .join(encode_claude_dir(&canonical_root.to_string_lossy()));
    if !session_dir.exists() {
        return Ok(None);
    }

    let files = list_jsonl_newest_first(&session_dir).await;
    for path in files {
        if let Some(prefix) = session_id
            && !claude_file_matches_prefix(&path, prefix)
        {
            continue;
        }
        // Fast-path exclusion: claude filenames are `<session_id>.jsonl`.
        if let Some(name) = path.file_stem().and_then(|s| s.to_str())
            && is_excluded(name, exclude_session_ids)
        {
            continue;
        }
        if let Some(transcript) = load_claude_transcript(&path, max_turns).await? {
            if is_excluded(&transcript.session_id, exclude_session_ids) {
                continue;
            }
            if let Some(cutoff) = started_after
                && let Some(updated) = transcript.updated_at
                && updated < cutoff
            {
                continue; // session predates this nickname's registration
            }
            return Ok(Some(transcript));
        }
    }
    Ok(None)
}

/// Claude stores session id inside each JSONL line (`sessionId` field).
/// Filenames are typically `<uuid>.jsonl`, but we scan the first valid line to
/// be safe across layout changes.
fn claude_file_matches_prefix(path: &Path, prefix: &str) -> bool {
    // Fast path: filename is usually the UUID.
    if let Some(name) = path.file_stem().and_then(|s| s.to_str())
        && name.starts_with(prefix)
    {
        return true;
    }
    false
}

/// Load the last `max_turns` natural-language turns from a Claude session.
pub async fn load_claude_transcript(path: &Path, max_turns: usize) -> Result<Option<Transcript>> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            // Skip partial/malformed lines (e.g. streaming tail).
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

        let raw = match extract_claude_text(v.get("message")) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }

        turns.push(Turn {
            role: kind.to_string(),
            text,
            timestamp,
        });
    }

    let session_id = match session_id {
        Some(s) => s,
        None => return Ok(None),
    };
    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "claude".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Find matching codex rollout JSONL file(s) and extract the first valid one.
async fn resolve_codex(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };
    let codex_root = home.join(".codex").join("sessions");
    if !codex_root.exists() {
        return Ok(None);
    }

    let files = walk_jsonl_newest_first(&codex_root).await;
    for path in files {
        match load_codex_transcript(&path, &canonical_str, max_turns).await? {
            Some(transcript) => {
                if let Some(prefix) = session_id
                    && !transcript.session_id.starts_with(prefix)
                {
                    continue;
                }
                if is_excluded(&transcript.session_id, exclude_session_ids) {
                    continue;
                }
                if let Some(cutoff) = started_after
                    && let Some(updated) = transcript.updated_at
                    && updated < cutoff
                {
                    continue; // session predates this nickname's registration
                }
                return Ok(Some(transcript));
            }
            None => continue,
        }
    }
    Ok(None)
}

/// Load the last `max_turns` natural-language turns from a Codex rollout file,
/// but only if its `session_meta.cwd` matches `project_root`.
pub async fn load_codex_transcript(
    path: &Path,
    project_root: &str,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut project_matches = false;
    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

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

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }

        if kind != "response_item" {
            continue;
        }
        let payload = match v.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        let role = payload
            .get("role")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        if role != "user" && role != "assistant" {
            continue;
        }
        let raw = match extract_codex_message_text(payload) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }

        turns.push(Turn {
            role,
            text,
            timestamp,
        });
    }

    if !project_matches {
        return Ok(None);
    }
    let session_id = match session_id {
        Some(s) => s,
        None => return Ok(None),
    };
    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "codex".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Find matching pi JSONL file(s) and extract the first/best transcript.
/// Pi stores one JSONL per session in a per-cwd directory.
async fn resolve_pi(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let session_dir = match pi_session_dir(&canonical_root.to_string_lossy()) {
        Some(d) if d.exists() => d,
        _ => return Ok(None),
    };

    for path in list_jsonl_newest_first(&session_dir).await {
        if let Some(t) = load_pi_transcript(&path, max_turns).await?
            && matches_filters(&t, session_id, exclude_session_ids, started_after)
        {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

/// Load the last `max_turns` natural-language turns from a Pi session.
///
/// Pi JSONL records: `type:"session"` (carries `id`) and `type:"message"`
/// (carries `message.role` and `message.content[]` text blocks). Top-level
/// `timestamp` is RFC 3339.
pub async fn load_pi_transcript(path: &Path, max_turns: usize) -> Result<Option<Transcript>> {
    let file = match fs::File::open(path).await {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

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
            && session_id.is_none()
            && let Some(id) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(id.to_string());
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }

        if kind != "message" {
            continue;
        }
        let message = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        let role = message.get("role").and_then(|s| s.as_str()).unwrap_or("");
        if role != "user" && role != "assistant" {
            continue;
        }
        let raw = match extract_pi_text(message) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }

        turns.push(Turn {
            role: role.to_string(),
            text,
            timestamp,
        });
    }

    let session_id = match session_id {
        Some(s) => s,
        None => return Ok(None),
    };
    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "pi".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Find matching gemini session JSON file(s) and extract the first/best
/// transcript. Gemini stores one JSON file per session under
/// `~/.gemini/tmp/<dir>/chats/`, matched to this cwd by `projectHash`.
async fn resolve_gemini(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();

    for path in gemini_matching_session_files(&canonical_str).await {
        if let Some(t) = load_gemini_transcript(&path, max_turns).await?
            && matches_filters(&t, session_id, exclude_session_ids, started_after)
        {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

/// Load the last `max_turns` natural-language turns from a Gemini session.
///
/// Each Gemini session file is a single JSON document with `sessionId`,
/// `messages[]`, and `lastUpdated`. Message types are `"user"` (content =
/// array of `{text}` blocks) and `"gemini"` (content = plain string). We
/// normalize the `gemini` role to `assistant` so MCP consumers see a uniform
/// `user`/`assistant` shape.
pub async fn load_gemini_transcript(path: &Path, max_turns: usize) -> Result<Option<Transcript>> {
    let content = match fs::read_to_string(path).await {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let v: Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let session_id = match v.get("sessionId").and_then(|s| s.as_str()) {
        Some(s) => s.to_string(),
        None => return Ok(None),
    };
    let messages = match v.get("messages").and_then(|m| m.as_array()) {
        Some(arr) => arr,
        None => return Ok(None),
    };

    let mut turns: Vec<Turn> = Vec::new();
    let mut updated_at: Option<DateTime<Utc>> = None;

    for msg in messages {
        let kind = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let role = match kind {
            "user" => "user",
            "gemini" => "assistant",
            _ => continue,
        };
        let raw = match extract_gemini_content(msg.get("content")) {
            Some(t) => t,
            None => continue,
        };
        let text = cap_turn(&raw);
        if text.trim().is_empty() {
            continue;
        }
        let timestamp = msg
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if timestamp.is_some() {
            updated_at = timestamp;
        }
        turns.push(Turn {
            role: role.to_string(),
            text,
            timestamp,
        });
    }

    // Fall back to top-level lastUpdated when per-message timestamps missing.
    if updated_at.is_none()
        && let Some(ts) = v.get("lastUpdated").and_then(|s| s.as_str())
        && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
    {
        updated_at = Some(parsed.with_timezone(&Utc));
    }

    if turns.is_empty() {
        return Ok(None);
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    Ok(Some(Transcript {
        cli: "gemini".into(),
        session_id,
        source_path: path.to_string_lossy().to_string(),
        turns,
        updated_at,
    }))
}

/// Find matching opencode session(s) in SQLite and extract the first/best
/// transcript. rusqlite is sync; the DB block runs in `spawn_blocking`.
async fn resolve_opencode(
    project_root: &Path,
    session_id: Option<&str>,
    exclude_session_ids: &[String],
    started_after: Option<DateTime<Utc>>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    let canonical_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let canonical_str = canonical_root.to_string_lossy().to_string();
    let home = match system_env::home_dir() {
        Some(h) => h,
        None => return Ok(None),
    };
    let db_path = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.exists() {
        return Ok(None);
    }

    let session_prefix = session_id.map(|s| s.to_string());
    let exclude = exclude_session_ids.to_vec();
    let started_after_ms = started_after.map(|t| t.timestamp_millis());

    tokio::task::spawn_blocking(move || {
        opencode_resolve_blocking(
            &db_path,
            &canonical_str,
            session_prefix.as_deref(),
            &exclude,
            started_after_ms,
            max_turns,
        )
    })
    .await
    .unwrap_or(Ok(None))
}

fn opencode_resolve_blocking(
    db_path: &Path,
    project_root: &str,
    session_prefix: Option<&str>,
    exclude_session_ids: &[String],
    started_after_ms: Option<i64>,
    max_turns: usize,
) -> Result<Option<Transcript>> {
    // Fetch a small bounded window of recent sessions so that prefix /
    // exclude filters have something to choose from. 50 is generous for
    // interactive use.
    const SESSION_FETCH_CAP: i64 = 50;

    let conn = match rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    ) {
        Ok(c) => c,
        Err(_) => return Ok(None),
    };

    let project_id: Option<String> = conn
        .query_row(
            "SELECT id FROM project WHERE worktree = ?1 LIMIT 1",
            [project_root],
            |row| row.get::<_, String>(0),
        )
        .ok();
    let project_id = match project_id {
        Some(p) => p,
        None => return Ok(None),
    };

    // `COALESCE(?3, time_updated)` lets the caller skip the cutoff by passing NULL.
    let mut stmt = match conn.prepare(
        "SELECT id, time_updated FROM session
           WHERE project_id = ?1
             AND time_updated >= COALESCE(?3, time_updated)
           ORDER BY time_updated DESC
           LIMIT ?2",
    ) {
        Ok(s) => s,
        Err(_) => return Ok(None),
    };
    let sessions: Vec<(String, i64)> = stmt
        .query_map(
            rusqlite::params![project_id, SESSION_FETCH_CAP, started_after_ms],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
        .unwrap_or_default();

    for (sid, time_updated_ms) in sessions {
        if let Some(prefix) = session_prefix
            && !sid.starts_with(prefix)
        {
            continue;
        }
        if is_excluded(&sid, exclude_session_ids) {
            continue;
        }
        if let Some(t) = opencode_load_session(&conn, &sid, time_updated_ms, max_turns) {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

/// Pull a full transcript for one opencode session, ordered chronologically.
fn opencode_load_session(
    conn: &rusqlite::Connection,
    session_id: &str,
    time_updated_ms: i64,
    max_turns: usize,
) -> Option<Transcript> {
    let mut stmt = conn
        .prepare(
            "SELECT json_extract(m.data, '$.role') AS role,
                    p.data,
                    m.time_created
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
                row.get::<_, Option<i64>>(2)?,
            ))
        })
        .ok()?;

    // Coalesce parts of the same message (same role + m.time_created) into
    // one Turn. The JOIN produces one row per text part.
    let mut turns: Vec<Turn> = Vec::new();
    for row in rows.flatten() {
        let (role, part_json, time_created_ms) = row;
        if role != "user" && role != "assistant" {
            continue;
        }
        let v: Value = match serde_json::from_str(&part_json) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let raw = v.get("text").and_then(|t| t.as_str()).unwrap_or("");
        if raw.trim().is_empty() {
            continue;
        }
        let timestamp = time_created_ms.and_then(chrono::DateTime::<Utc>::from_timestamp_millis);
        let text = cap_turn(raw);

        let coalesced = turns
            .last_mut()
            .filter(|prev| prev.role == role && prev.timestamp == timestamp);
        if let Some(prev) = coalesced {
            // Merge into previous turn, re-capping in case the combined size
            // would exceed the per-turn limit.
            let mut merged = prev.text.clone();
            merged.push('\n');
            merged.push_str(&text);
            prev.text = cap_turn(&merged);
        } else {
            turns.push(Turn {
                role,
                text,
                timestamp,
            });
        }
    }

    if turns.is_empty() {
        return None;
    }
    if turns.len() > max_turns {
        let start = turns.len() - max_turns;
        turns = turns.split_off(start);
    }
    let updated_at = chrono::DateTime::<Utc>::from_timestamp_millis(time_updated_ms);
    Some(Transcript {
        cli: "opencode".into(),
        session_id: session_id.to_string(),
        source_path: format!("db://opencode/{session_id}"),
        turns,
        updated_at,
    })
}

/// Cap a turn's text at `MAX_TURN_BYTES`, respecting UTF-8 char boundaries.
/// Suffixes with `…` when truncated.
fn cap_turn(text: &str) -> String {
    if text.len() <= MAX_TURN_BYTES {
        return text.to_string();
    }
    // Walk char boundaries to find the largest prefix <= MAX_TURN_BYTES - 3
    // (reserving three bytes for the `…` we append).
    let budget = MAX_TURN_BYTES.saturating_sub(3);
    let mut end = 0;
    for (idx, _) in text.char_indices() {
        if idx > budget {
            break;
        }
        end = idx;
    }
    let mut out = String::with_capacity(end + 3);
    out.push_str(&text[..end]);
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn is_excluded_matches_prefix() {
        let ex = vec!["abc".to_string(), "xyz-1".to_string()];
        assert!(is_excluded("abc123", &ex));
        assert!(is_excluded("xyz-1234", &ex));
        assert!(!is_excluded("def", &ex));
        assert!(!is_excluded("abc", &[])); // empty exclude list never matches
    }

    #[tokio::test]
    async fn load_claude_transcript_returns_verbatim_turns_chronologically() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","message":{"content":"Please review the pagination helper in handlers/users.go."}}"#,
            r#"{"type":"assistant","sessionId":"sid-abc","isSidechain":true,"timestamp":"2026-04-01T10:01:00Z","message":{"content":[{"type":"text","text":"SIDECHAIN - SHOULD NOT APPEAR"}]}}"#,
            r#"{"type":"assistant","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:02:00Z","message":{"content":[{"type":"text","text":"Found two issues: (1) empty cursor returns 500, (2) limit > 1000 is not clamped."}]}}"#,
            r#"{"type":"user","sessionId":"sid-abc","isSidechain":false,"timestamp":"2026-04-01T10:03:00Z","message":{"content":"fix them"}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_claude_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "claude");
        assert_eq!(t.session_id, "sid-abc");
        assert_eq!(t.turns.len(), 3); // sidechain skipped
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[0].text.starts_with("Please review"));
        assert_eq!(t.turns[1].role, "assistant");
        assert!(t.turns[1].text.starts_with("Found two issues"));
        assert!(!t.turns[1].text.contains("SIDECHAIN"));
        assert_eq!(t.turns[2].role, "user");
        assert_eq!(t.turns[2].text, "fix them");
    }

    #[tokio::test]
    async fn load_claude_transcript_respects_max_turns() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let mut lines = Vec::new();
        for i in 0..10 {
            lines.push(format!(
                r#"{{"type":"user","sessionId":"sid-x","isSidechain":false,"message":{{"content":"turn {i} content long enough to count"}}}}"#
            ));
        }
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_claude_transcript(&path, 3)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.turns.len(), 3);
        // Chronological: last 3 → 7, 8, 9
        assert!(t.turns[0].text.contains("turn 7"));
        assert!(t.turns[2].text.contains("turn 9"));
    }

    #[tokio::test]
    async fn load_claude_transcript_silently_skips_malformed_lines() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let mut content = String::new();
        content.push_str(r#"{"type":"user","sessionId":"sid-y","isSidechain":false,"message":{"content":"hello world"}}"#);
        content.push('\n');
        content.push_str("{not json at all");
        content.push('\n');
        content.push_str(r#"{"type":"assistant","sessionId":"sid-y","isSidechain":false,"message":{"content":[{"type":"text","text":"hi"}]}}"#);
        fs::write(&path, &content).await.unwrap();

        let t = load_claude_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract despite one garbage line");
        assert_eq!(t.turns.len(), 2);
    }

    #[tokio::test]
    async fn load_codex_transcript_matches_cwd_and_returns_turns() {
        let dir = TempDir::new().unwrap();
        let project_root = dir.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let proj_str = project_root.to_string_lossy().to_string();

        let path = dir.path().join("rollout.jsonl");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-04-01T10:00:00Z","payload":{{"id":"codex-abc","cwd":"{}"}}}}"#,
                proj_str
            ),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:01:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"review this pagination patch"}]}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:02:00Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Looks mostly fine. One issue: empty cursor returns 500."}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_codex_transcript(&path, &proj_str, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "codex");
        assert_eq!(t.session_id, "codex-abc");
        assert_eq!(t.turns.len(), 2);
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[1].text.contains("empty cursor"));
    }

    #[tokio::test]
    async fn load_pi_transcript_extracts_text_turns_chronologically() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"session","id":"pi-xyz","timestamp":"2026-04-01T10:00:00Z"}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:01:00Z","message":{"role":"user","content":[{"type":"text","text":"refactor the cache layer"}]}}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:02:00Z","message":{"role":"assistant","content":[{"type":"text","text":"Done. Switched to LRU and added a 5 minute TTL."}]}}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:03:00Z","message":{"role":"tool","content":[{"type":"text","text":"SHOULD NOT APPEAR"}]}}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:04:00Z","message":{"role":"user","content":[{"type":"text","text":"add tests"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_pi_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "pi");
        assert_eq!(t.session_id, "pi-xyz");
        assert_eq!(t.turns.len(), 3); // tool role skipped
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[0].text.contains("refactor the cache layer"));
        assert_eq!(t.turns[1].role, "assistant");
        assert!(t.turns[1].text.contains("LRU"));
        assert!(!t.turns.iter().any(|x| x.text.contains("SHOULD NOT APPEAR")));
        assert_eq!(t.turns[2].role, "user");
        assert_eq!(t.turns[2].text, "add tests");
        assert!(t.updated_at.is_some());
    }

    #[tokio::test]
    async fn load_pi_transcript_respects_max_turns() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let mut lines = vec![r#"{"type":"session","id":"pi-cap"}"#.to_string()];
        for i in 0..10 {
            lines.push(format!(
                r#"{{"type":"message","message":{{"role":"user","content":[{{"type":"text","text":"turn {i}"}}]}}}}"#
            ));
        }
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_pi_transcript(&path, 3)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.turns.len(), 3);
        // Chronological: last 3 → 7, 8, 9
        assert!(t.turns[0].text.contains("turn 7"));
        assert!(t.turns[2].text.contains("turn 9"));
    }

    #[tokio::test]
    async fn load_pi_transcript_returns_none_without_session_record() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        // Messages but no `type:"session"` record → should return None.
        let lines = [
            r#"{"type":"message","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_pi_transcript(&path, 10).await.unwrap();
        assert!(t.is_none());
    }

    #[tokio::test]
    async fn load_codex_transcript_rejects_non_matching_cwd() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rollout.jsonl");
        let lines = [
            r#"{"type":"session_meta","payload":{"id":"codex-1","cwd":"/nope"}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let t = load_codex_transcript(&path, "/other", 10).await.unwrap();
        assert!(t.is_none());
    }

    #[tokio::test]
    async fn load_gemini_transcript_normalizes_role_and_skips_non_chat() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session-abc.json");
        let body = serde_json::json!({
            "sessionId": "gem-abc",
            "projectHash": "deadbeef",
            "lastUpdated": "2026-04-01T10:10:00Z",
            "messages": [
                {
                    "type": "user",
                    "timestamp": "2026-04-01T10:00:00Z",
                    "content": [{"text": "Why is the sky blue?"}]
                },
                {
                    "type": "tool_call",
                    "timestamp": "2026-04-01T10:01:00Z",
                    "content": "SHOULD NOT APPEAR"
                },
                {
                    "type": "gemini",
                    "timestamp": "2026-04-01T10:02:00Z",
                    "content": "Rayleigh scattering."
                },
                {
                    "type": "user",
                    "timestamp": "2026-04-01T10:03:00Z",
                    "content": [{"text": "explain more"}]
                }
            ]
        });
        fs::write(&path, body.to_string()).await.unwrap();

        let t = load_gemini_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert_eq!(t.cli, "gemini");
        assert_eq!(t.session_id, "gem-abc");
        assert_eq!(t.turns.len(), 3); // tool_call skipped
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[0].text.contains("sky blue"));
        // gemini role normalized to assistant for uniform peer reads.
        assert_eq!(t.turns[1].role, "assistant");
        assert_eq!(t.turns[1].text, "Rayleigh scattering.");
        assert_eq!(t.turns[2].role, "user");
        assert_eq!(t.turns[2].text, "explain more");
        assert!(t.updated_at.is_some());
    }

    #[tokio::test]
    async fn load_gemini_transcript_falls_back_to_last_updated_when_no_msg_timestamps() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session-x.json");
        let body = serde_json::json!({
            "sessionId": "gem-x",
            "lastUpdated": "2026-04-01T12:00:00Z",
            "messages": [
                {"type": "user", "content": [{"text": "hi"}]},
                {"type": "gemini", "content": "hello"}
            ]
        });
        fs::write(&path, body.to_string()).await.unwrap();

        let t = load_gemini_transcript(&path, 10)
            .await
            .unwrap()
            .expect("should extract");
        assert!(t.updated_at.is_some());
        assert_eq!(
            t.updated_at.unwrap().to_rfc3339(),
            "2026-04-01T12:00:00+00:00"
        );
    }

    #[test]
    fn opencode_resolve_blocking_returns_chronological_turns_for_matching_project() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("opencode.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
            CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT);
            INSERT INTO project (id, worktree) VALUES ('p1', '/work/proj');
            INSERT INTO session (id, project_id, time_updated) VALUES
                ('sess-old', 'p1', 1000),
                ('sess-new', 'p1', 5000);
            INSERT INTO message (id, session_id, time_created, data) VALUES
                ('m1', 'sess-new', 2000, '{"role":"user"}'),
                ('m2', 'sess-new', 2500, '{"role":"assistant"}'),
                ('m3', 'sess-new', 3000, '{"role":"user"}');
            INSERT INTO part (id, message_id, time_created, data) VALUES
                ('pa1', 'm1', 2000, '{"type":"text","text":"refactor cache"}'),
                ('pa2', 'm2', 2500, '{"type":"text","text":"done — switched to LRU"}'),
                ('pa3', 'm2', 2510, '{"type":"text","text":"with 5min TTL"}'),
                ('pa4', 'm3', 3000, '{"type":"text","text":"add tests"}');
            "#,
        )
        .unwrap();
        drop(conn);

        let t = opencode_resolve_blocking(&db_path, "/work/proj", None, &[], None, 10)
            .unwrap()
            .expect("should resolve newest matching session");
        assert_eq!(t.cli, "opencode");
        assert_eq!(t.session_id, "sess-new"); // newer time_updated
        assert_eq!(t.source_path, "db://opencode/sess-new");
        // Three turns: user, assistant (two parts merged into one Turn), user.
        assert_eq!(t.turns.len(), 3);
        assert_eq!(t.turns[0].role, "user");
        assert!(t.turns[0].text.contains("refactor cache"));
        assert_eq!(t.turns[1].role, "assistant");
        assert!(t.turns[1].text.contains("LRU"));
        assert!(t.turns[1].text.contains("5min TTL")); // coalesced parts
        assert_eq!(t.turns[2].role, "user");
        assert!(t.turns[2].text.contains("add tests"));
    }

    #[test]
    fn opencode_resolve_blocking_honors_started_after_filter() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("opencode.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
            CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT);
            INSERT INTO project (id, worktree) VALUES ('p1', '/work/proj');
            INSERT INTO session (id, project_id, time_updated) VALUES ('sess-stale', 'p1', 1000);
            INSERT INTO message (id, session_id, time_created, data) VALUES
                ('m1', 'sess-stale', 500, '{"role":"user"}');
            INSERT INTO part (id, message_id, time_created, data) VALUES
                ('pa1', 'm1', 500, '{"type":"text","text":"old chat"}');
            "#,
        )
        .unwrap();
        drop(conn);

        // started_after = 9999ms is later than time_updated (1000ms) → reject.
        let t =
            opencode_resolve_blocking(&db_path, "/work/proj", None, &[], Some(9999), 10).unwrap();
        assert!(t.is_none());
    }

    #[test]
    fn opencode_resolve_blocking_returns_none_for_unknown_project() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("opencode.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            r#"
            CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT);
            CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, time_created INTEGER, data TEXT);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, time_created INTEGER, data TEXT);
            INSERT INTO project (id, worktree) VALUES ('p1', '/other/path');
            "#,
        )
        .unwrap();
        drop(conn);

        let t = opencode_resolve_blocking(&db_path, "/work/proj", None, &[], None, 10).unwrap();
        assert!(t.is_none());
    }

    #[test]
    fn cap_turn_truncates_oversize_text() {
        let big = "a".repeat(MAX_TURN_BYTES + 100);
        let capped = cap_turn(&big);
        assert!(capped.ends_with('…'));
        assert!(capped.len() <= MAX_TURN_BYTES);
    }

    #[test]
    fn cap_turn_respects_utf8_boundaries() {
        // Construct a string whose MAX_TURN_BYTES-th byte falls inside a multi-byte char.
        let mut s = "a".repeat(MAX_TURN_BYTES - 2);
        s.push('🚀'); // 4 bytes, crosses the cap boundary
        s.push_str("xyz");
        let capped = cap_turn(&s);
        // Must still be valid UTF-8 — the test itself would panic if not.
        assert!(capped.ends_with('…'));
        assert!(capped.is_char_boundary(capped.len()));
    }
}
