use anyhow::{Context, Result};
use chrono::Utc;
use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params_from_iter};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LogStore {
    path: PathBuf,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct LogEntry {
    pub id: String,
    pub ts_utc: String,
    pub source: String,
    pub kind: String,
    pub event_group_id: Option<String>,
    pub phase: Option<String>,
    pub key_id: Option<String>,
    pub key_name: Option<String>,
    pub base_url: Option<String>,
    pub tool: Option<String>,
    pub model: Option<String>,
    pub cwd: Option<String>,
    pub session_id: Option<String>,
    pub status_code: Option<i64>,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<i64>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    pub title: Option<String>,
    pub body_text: Option<String>,
    pub payload_json: Option<JsonValue>,
}

#[derive(Debug, Clone, Default)]
pub struct LogEvent {
    pub source: String,
    pub kind: String,
    pub event_group_id: Option<String>,
    pub phase: Option<String>,
    pub key_id: Option<String>,
    pub key_name: Option<String>,
    pub base_url: Option<String>,
    pub tool: Option<String>,
    pub model: Option<String>,
    pub cwd: Option<String>,
    pub session_id: Option<String>,
    pub status_code: Option<i64>,
    pub exit_code: Option<i64>,
    pub duration_ms: Option<i64>,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    pub title: Option<String>,
    pub body_text: Option<String>,
    pub payload_json: Option<JsonValue>,
}

#[derive(Debug, Clone, Default)]
pub struct LogQuery {
    pub limit: usize,
    pub search: Option<String>,
    /// Matches against the `source` column (exact) or `tool` column (substring).
    pub by: Option<String>,
    pub model: Option<String>,
    pub key_query: Option<String>,
    pub cwd: Option<String>,
    pub since: Option<String>,
    pub until: Option<String>,
    pub errors_only: bool,
}

/// Aivo-side metadata captured at launch and joined onto a native session
/// row. Sourced from the `[run]` finished event whose `session_id` matches
/// the native session file the launched CLI just produced.
#[derive(Debug, Clone, Default)]
pub struct RunMeta {
    pub key_name: Option<String>,
    pub exit_code: Option<i64>,
}

impl LogStore {
    pub fn new(config_dir: PathBuf) -> Self {
        Self {
            path: crate::services::paths::logs_db(&config_dir),
        }
    }

    pub async fn append(&self, event: LogEvent) -> Result<String> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            let conn = open_connection(&path)?;
            let id = new_log_id();
            let ts_utc = Utc::now().to_rfc3339();
            let payload_json = event.payload_json.map(|value| value.to_string());
            let params = vec![
                SqlValue::Text(id.clone()),
                SqlValue::Text(ts_utc),
                SqlValue::Text(event.source),
                SqlValue::Text(event.kind),
                option_text(event.event_group_id),
                option_text(event.phase),
                option_text(event.key_id),
                option_text(event.key_name),
                option_text(event.base_url),
                option_text(event.tool),
                option_text(event.model),
                option_text(event.cwd),
                option_text(event.session_id),
                option_integer(event.status_code),
                option_integer(event.exit_code),
                option_integer(event.duration_ms),
                option_integer(event.input_tokens),
                option_integer(event.output_tokens),
                option_integer(event.cache_read_input_tokens),
                option_integer(event.cache_creation_input_tokens),
                option_text(event.title),
                option_text(event.body_text),
                option_text(payload_json),
            ];
            conn.execute(
                "insert into events (
                    id, ts_utc, source, kind, event_group_id, phase, key_id, key_name, base_url, tool, model, cwd,
                    session_id, status_code, exit_code, duration_ms, input_tokens, output_tokens,
                    cache_read_input_tokens, cache_creation_input_tokens, title, body_text,
                    payload_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                params_from_iter(params),
            )
            .context("Failed to insert log entry")?;
            Ok(id)
        })
        .await
        .context("Failed to join log insert task")?
    }

    pub async fn list(&self, query: LogQuery) -> Result<Vec<LogEntry>> {
        self.read_with_fallback("logs", move |conn| list_with_connection(conn, &query))
            .await
    }

    #[allow(dead_code)]
    pub async fn get(&self, id: &str) -> Result<Option<LogEntry>> {
        let id = id.to_string();
        self.read_with_fallback("log entry", move |conn| get_with_connection(conn, &id))
            .await
    }

    /// Prefix-matches up to `limit` rows. Matches `id` / `event_group_id`
    /// first (what `aivo logs` displays) and only falls back to
    /// `session_id` when that pass is empty — UUIDv7 session ids share
    /// 10+ leading hex chars for same-minute creates, which used to
    /// create false ambiguity for users copy-pasting a displayed id.
    pub async fn find_by_id_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<LogEntry>> {
        let path = self.path.clone();
        let prefix = prefix.trim().to_string();
        let limit = limit.max(1);
        tokio::task::spawn_blocking(move || -> Result<Vec<LogEntry>> {
            if !path.exists() {
                return Ok(Vec::new());
            }
            let conn = open_read_connection(&path)?;
            let primary_sql = format!(
                "select {} from events
                  where id like ?1 || '%'
                     or event_group_id like ?1 || '%'
                  order by ts_utc desc
                  limit ?2",
                event_select_columns(true)
            );
            let mut stmt = conn.prepare(&primary_sql)?;
            let rows = stmt.query_map(rusqlite::params![prefix, limit as i64], map_log_row)?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row?);
            }
            if out.is_empty() {
                let fallback_sql = format!(
                    "select {} from events
                      where session_id like ?1 || '%'
                      order by ts_utc desc
                      limit ?2",
                    event_select_columns(true)
                );
                let mut stmt = conn.prepare(&fallback_sql)?;
                let rows = stmt.query_map(rusqlite::params![prefix, limit as i64], map_log_row)?;
                for row in rows {
                    out.push(row?);
                }
            }
            // Collapse start+finish pairs and per-chat events; rows are
            // ts_utc-desc so the most recent in each group wins.
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            out.retain(|e| {
                let key = e
                    .event_group_id
                    .clone()
                    .or_else(|| e.session_id.clone())
                    .unwrap_or_else(|| e.id.clone());
                seen.insert(key)
            });
            Ok(out)
        })
        .await
        .context("Failed to join log prefix-find task")?
    }

    /// Every distinct `session_id` referenced by a chat event in logs.db.
    /// Used by `aivo logs prune` to spot chat sessions whose underlying
    /// file has been deleted (orphan logs.db entries).
    pub async fn distinct_code_session_ids(&self) -> Result<std::collections::HashSet<String>> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || -> Result<std::collections::HashSet<String>> {
            if !path.exists() {
                return Ok(std::collections::HashSet::new());
            }
            let conn = open_read_connection(&path)?;
            let mut stmt = conn.prepare(
                "select distinct session_id from events \
                 where source in ('chat','code') and session_id is not null",
            )?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
            let mut out = std::collections::HashSet::new();
            for row in rows {
                out.insert(row?);
            }
            Ok(out)
        })
        .await
        .context("Failed to join distinct-chat-session-ids task")?
    }

    /// Look up `[run]` finished-event metadata for a batch of native
    /// session ids. Returns `(key_name, exit_code)` per session id that
    /// has a matching `tool_launch` finished row. Used by `aivo logs` to
    /// enrich native rows with the aivo-side context that the native
    /// session file alone doesn't carry (which key was used, exit status).
    pub async fn run_meta_for_sessions(
        &self,
        session_ids: &[String],
    ) -> Result<std::collections::HashMap<String, RunMeta>> {
        if session_ids.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        let path = self.path.clone();
        let ids: Vec<String> = session_ids.to_vec();
        tokio::task::spawn_blocking(
            move || -> Result<std::collections::HashMap<String, RunMeta>> {
                if !path.exists() {
                    return Ok(std::collections::HashMap::new());
                }
                let conn = open_read_connection(&path)?;
                let placeholders = std::iter::repeat_n("?", ids.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "select session_id, key_name, exit_code from events \
                 where source = 'run' and phase = 'finished' \
                   and session_id in ({placeholders})"
                );
                let mut stmt = conn.prepare(&sql)?;
                let params: Vec<&dyn rusqlite::ToSql> =
                    ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
                let rows = stmt.query_map(params.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                })?;
                let mut out = std::collections::HashMap::new();
                for row in rows {
                    let (sid, key_name, exit_code) = row?;
                    out.insert(
                        sid,
                        RunMeta {
                            key_name,
                            exit_code,
                        },
                    );
                }
                Ok(out)
            },
        )
        .await
        .context("Failed to join run-meta-for-sessions task")?
    }

    /// Delete every chat event whose `session_id` is in `ids`. Used by
    /// `aivo logs prune` to clean up orphan rows. Returns the number of
    /// events removed.
    pub async fn delete_code_events_by_session_ids(&self, ids: &[String]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }
        let path = self.path.clone();
        let ids: Vec<String> = ids.to_vec();
        tokio::task::spawn_blocking(move || -> Result<u64> {
            if !path.exists() {
                return Ok(0);
            }
            let mut conn = open_connection(&path)?;
            let tx = conn.transaction()?;
            let placeholders = std::iter::repeat_n("?", ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "delete from events where source in ('chat','code') and session_id in ({placeholders})"
            );
            let params: Vec<&dyn rusqlite::ToSql> =
                ids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let affected = tx.execute(&sql, params.as_slice())?;
            tx.commit()?;
            Ok(affected as u64)
        })
        .await
        .context("Failed to join delete-chat-events task")?
    }

    pub async fn get_by_reference(&self, reference: &str) -> Result<Option<LogEntry>> {
        let reference = reference.trim().to_string();
        self.read_with_fallback("log entry by reference", move |conn| {
            get_by_reference_with_connection(conn, &reference)
        })
        .await
    }

    /// Counts `tool_launch` events (phase = `started`) since `cutoff`,
    /// grouped by model. Surfaces models the user actually launched in the
    /// window even when no upstream usage was recorded — the table-of-truth
    /// for "what did I run", independent of provider-side `usage` fields.
    /// `tool_filter` scopes to a single tool (e.g. `claude`) when set.
    pub async fn aggregate_run_models_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        tool_filter: Option<&str>,
    ) -> Result<HashMap<String, u64>> {
        let cutoff_str = cutoff.to_rfc3339();
        let tool_filter = tool_filter.map(|t| t.to_string());
        self.read_with_fallback("run models", move |conn| {
            aggregate_run_models_with_connection(conn, &cutoff_str, tool_filter.as_deref())
        })
        .await
    }

    /// Counts `tool_launch` *started* rows since `cutoff` — the windowed launch
    /// count. Unlike `aggregate_run_models_since`, model-less launches count too
    /// (a run can proxy traffic without ever resolving a model name).
    pub async fn count_runs_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        tool_filter: Option<&str>,
    ) -> Result<u64> {
        let cutoff_str = cutoff.to_rfc3339();
        let tool_filter = tool_filter.map(|t| t.to_string());
        self.read_with_fallback("run counts", move |conn| {
            count_runs_with_connection(conn, &cutoff_str, tool_filter.as_deref())
        })
        .await
    }

    /// Counts distinct chat sessions with any logged turn since `cutoff`.
    /// Turn events outlive session-index eviction, making this the windowed floor.
    pub async fn count_distinct_code_sessions_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
    ) -> Result<u64> {
        let cutoff_str = cutoff.to_rfc3339();
        self.read_with_fallback("chat session counts", move |conn| {
            count_distinct_code_sessions_with_connection(conn, &cutoff_str)
        })
        .await
    }

    /// Sums per-model token usage from `tool_launch` *finished* rows since
    /// `cutoff`. These tokens are stamped by the plugin endpoint at run end (see
    /// `finish_accounting`), so this is the timestamped, windowable view of a
    /// coding-agent plugin's usage — the counterpart of `aggregate_run_models_since`
    /// (which counts launches). Native tools carry no tokens on these rows, so a
    /// `tool_filter` is unnecessary for correctness but scopes the query when set.
    /// Returns `model → (input, output, cache_read, cache_creation)`, dropping
    /// zero-token rows (those still surface as launches).
    pub async fn aggregate_run_tokens_since(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        tool_filter: Option<&str>,
    ) -> Result<HashMap<String, RunModelTokens>> {
        let cutoff_str = cutoff.to_rfc3339();
        let tool_filter = tool_filter.map(|t| t.to_string());
        self.read_with_fallback("run tokens", move |conn| {
            aggregate_run_tokens_with_connection(conn, &cutoff_str, tool_filter.as_deref())
        })
        .await
    }

    /// Run a read-only query off the runtime: direct read connection first,
    /// snapshot-copy fallback when the live DB can't be read. A missing DB
    /// file yields `T::default()` (no rows yet, not an error).
    async fn read_with_fallback<T, F>(&self, what: &'static str, op: F) -> Result<T>
    where
        T: Default + Send + 'static,
        F: Fn(&Connection) -> Result<T> + Send + 'static,
    {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || {
            if !path.exists() {
                return Ok(T::default());
            }
            match open_read_connection(&path).and_then(|conn| op(&conn)) {
                Ok(v) => Ok(v),
                Err(direct_err) => {
                    with_snapshot_connection(&path, |conn| op(conn)).with_context(|| {
                        format!("Failed to read {what} from {path:?}: {direct_err:#}")
                    })
                }
            }
        })
        .await
        .with_context(|| format!("Failed to join {what} read task"))?
    }
}

/// Per-model token totals summed from `tool_launch` finished rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunModelTokens {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_creation: u64,
}

fn normalize_query_value(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn normalize_text_filter(value: Option<String>) -> Option<String> {
    normalize_query_value(value).map(|value| value.to_lowercase())
}

fn option_text(value: Option<String>) -> SqlValue {
    value.map(SqlValue::Text).unwrap_or(SqlValue::Null)
}

fn option_integer(value: Option<i64>) -> SqlValue {
    value.map(SqlValue::Integer).unwrap_or(SqlValue::Null)
}

pub fn new_log_id() -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"23456789abcdefghjkmnpqrstvwxyz";
    let mut rng = rand::thread_rng();
    (0..12)
        .map(|_| {
            let index = rng.gen_range(0..ALPHABET.len());
            ALPHABET[index] as char
        })
        .collect()
}

fn open_connection(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create log directory: {:?}", parent))?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("Failed to open SQLite log database: {:?}", path))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .context("Failed to configure SQLite busy timeout")?;
    conn.execute_batch(
        "
        pragma journal_mode = wal;
        pragma synchronous = normal;
        create table if not exists events (
            id text primary key,
            ts_utc text not null,
            source text not null,
            kind text not null,
            event_group_id text,
            phase text,
            key_id text,
            key_name text,
            base_url text,
            tool text,
            model text,
            cwd text,
            session_id text,
            status_code integer,
            exit_code integer,
            duration_ms integer,
            input_tokens integer,
            output_tokens integer,
            cache_read_input_tokens integer,
            cache_creation_input_tokens integer,
            title text,
            body_text text,
            payload_json text
        );
        ",
    )
    .context("Failed to initialize SQLite log schema")?;
    ensure_column_exists(&conn, "events", "event_group_id", "text")?;
    ensure_column_exists(&conn, "events", "phase", "text")?;
    conn.execute_batch(
        "
        create index if not exists idx_events_ts on events(ts_utc desc);
        create index if not exists idx_events_source_ts on events(source, ts_utc desc);
        create index if not exists idx_events_tool_ts on events(tool, ts_utc desc);
        create index if not exists idx_events_model_ts on events(model, ts_utc desc);
        create index if not exists idx_events_key_ts on events(key_id, ts_utc desc);
        create index if not exists idx_events_cwd_ts on events(cwd, ts_utc desc);
        create index if not exists idx_events_session_ts on events(session_id, ts_utc desc);
        create index if not exists idx_events_group_ts on events(event_group_id, ts_utc desc);
        ",
    )
    .context("Failed to initialize SQLite log indexes")?;
    Ok(conn)
}

fn open_read_connection(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("Failed to open SQLite log database for reading: {:?}", path))
}

fn with_snapshot_connection<T, F>(path: &Path, op: F) -> Result<T>
where
    F: FnOnce(&Connection) -> Result<T>,
{
    let temp_dir = tempfile::tempdir().context("Failed to create temporary SQLite snapshot dir")?;
    let snapshot_path = temp_dir.path().join("logs.db");
    copy_sqlite_snapshot(path, &snapshot_path)?;
    let conn = Connection::open(&snapshot_path)
        .with_context(|| format!("Failed to open SQLite log snapshot: {:?}", snapshot_path))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .context("Failed to configure SQLite snapshot busy timeout")?;
    op(&conn)
}

fn copy_sqlite_snapshot(path: &Path, snapshot_path: &Path) -> Result<()> {
    std::fs::copy(path, snapshot_path).with_context(|| {
        format!(
            "Failed to copy SQLite log database from {:?} to {:?}",
            path, snapshot_path
        )
    })?;
    for suffix in ["-wal", "-shm"] {
        let sidecar = sqlite_sidecar_path(path, suffix);
        if sidecar.exists() {
            let snapshot_sidecar = sqlite_sidecar_path(snapshot_path, suffix);
            std::fs::copy(&sidecar, &snapshot_sidecar).with_context(|| {
                format!(
                    "Failed to copy SQLite sidecar from {:?} to {:?}",
                    sidecar, snapshot_sidecar
                )
            })?;
        }
    }
    Ok(())
}

fn sqlite_sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", path.display(), suffix))
}

fn event_select_columns(include_run_phase_fields: bool) -> String {
    let phase_cols: [&str; 2] = if include_run_phase_fields {
        ["event_group_id", "phase"]
    } else {
        ["null as event_group_id", "null as phase"]
    };
    let columns: &[&str] = &[
        "id",
        "ts_utc",
        "source",
        "kind",
        phase_cols[0],
        phase_cols[1],
        "key_id",
        "key_name",
        "base_url",
        "tool",
        "model",
        "cwd",
        "session_id",
        "status_code",
        "exit_code",
        "duration_ms",
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "title",
        "body_text",
        "payload_json",
    ];
    columns.join(", ")
}

fn build_list_query(query: &LogQuery, include_run_phase_fields: bool) -> (String, Vec<SqlValue>) {
    let mut sql = format!(
        "select {} from events where 1 = 1",
        event_select_columns(include_run_phase_fields)
    );
    let mut params: Vec<SqlValue> = Vec::new();

    if let Some(by) = normalize_text_filter(query.by.clone()) {
        if by == "code" {
            // The built-in agent's rows: post-rename `code` plus pre-rename
            // `chat`. Exact source match (no `tool like`) so it can't grab the
            // `codex`/`opencode` tools, whose names contain "code".
            sql.push_str(" and source in ('chat','code')");
        } else {
            sql.push_str(" and (source = ? or lower(coalesce(tool, '')) like ?)");
            params.push(SqlValue::Text(by.clone()));
            params.push(SqlValue::Text(format!("%{by}%")));
        }
    }
    if let Some(model) = normalize_text_filter(query.model.clone()) {
        sql.push_str(" and lower(coalesce(model, '')) like ?");
        params.push(SqlValue::Text(format!("%{model}%")));
    }
    if let Some(key_query) = normalize_text_filter(query.key_query.clone()) {
        sql.push_str(
            " and (
                lower(coalesce(key_id, '')) like ?
                or lower(coalesce(key_name, '')) like ?
            )",
        );
        let term = format!("%{key_query}%");
        params.push(SqlValue::Text(term.clone()));
        params.push(SqlValue::Text(term));
    }
    if let Some(cwd) = normalize_text_filter(query.cwd.clone()) {
        // Match the dir or its descendants via `= ?` / `like ?/%`; a substring
        // match would catch sibling prefixes (`/foo/bar` vs `/foo/bar-other`).
        // Normalize separators to `/` and lower-case both sides so Windows
        // backslash/case variants compare equal.
        let cwd: String = cwd
            .trim_end_matches(['/', '\\'])
            .chars()
            .map(|ch| if ch == '\\' { '/' } else { ch })
            .collect();
        sql.push_str(
            " and (replace(lower(coalesce(cwd, '')), '\\', '/') = ? \
                or replace(lower(coalesce(cwd, '')), '\\', '/') like ? || '/%')",
        );
        params.push(SqlValue::Text(cwd.clone()));
        params.push(SqlValue::Text(cwd));
    }
    if let Some(since) = normalize_query_value(query.since.clone()) {
        sql.push_str(" and ts_utc >= ?");
        params.push(SqlValue::Text(since));
    }
    if let Some(until) = normalize_query_value(query.until.clone()) {
        sql.push_str(" and ts_utc <= ?");
        params.push(SqlValue::Text(until));
    }
    if query.errors_only {
        sql.push_str(
            " and (
                (status_code is not null and status_code >= 400)
                or (exit_code is not null and exit_code != 0)
            )",
        );
    }
    if let Some(search) = normalize_text_filter(query.search.clone()) {
        sql.push_str(
            " and (
                lower(coalesce(title, '')) like ?
                or lower(coalesce(body_text, '')) like ?
                or lower(coalesce(model, '')) like ?
                or lower(coalesce(tool, '')) like ?
                or lower(coalesce(key_name, '')) like ?
                or lower(coalesce(key_id, '')) like ?
                or lower(coalesce(base_url, '')) like ?
                or lower(coalesce(cwd, '')) like ?
            )",
        );
        let term = format!("%{search}%");
        for _ in 0..8 {
            params.push(SqlValue::Text(term.clone()));
        }
    }

    sql.push_str(" order by ts_utc desc limit ?");
    params.push(SqlValue::Integer(query.limit.max(1) as i64));
    (sql, params)
}

fn is_legacy_log_schema_error(err: &rusqlite::Error) -> bool {
    let message = err.to_string();
    message.contains("no such column: event_group_id") || message.contains("no such column: phase")
}

fn list_with_connection(conn: &Connection, query: &LogQuery) -> Result<Vec<LogEntry>> {
    let (sql, params) = build_list_query(query, true);
    let mut statement = match conn.prepare(&sql) {
        Ok(statement) => statement,
        Err(err) if is_legacy_log_schema_error(&err) => {
            let (legacy_sql, legacy_params) = build_list_query(query, false);
            let mut statement = conn
                .prepare(&legacy_sql)
                .with_context(|| format!("Failed to prepare legacy log query: {legacy_sql}"))?;
            let rows = statement
                .query_map(params_from_iter(legacy_params), map_log_row)
                .context("Failed to read legacy log rows")?;
            let mut entries = Vec::new();
            for row in rows {
                entries.push(row?);
            }
            return Ok(entries);
        }
        Err(err) => {
            let err_text = err.to_string();
            return Err(err).with_context(|| {
                format!("Failed to prepare log query: {sql}; sqlite error: {err_text}")
            });
        }
    };
    let rows = statement
        .query_map(params_from_iter(params), map_log_row)
        .context("Failed to read log rows")?;
    let mut entries = Vec::new();
    for row in rows {
        entries.push(row?);
    }
    Ok(entries)
}

fn get_with_connection(conn: &Connection, id: &str) -> Result<Option<LogEntry>> {
    let modern_sql = format!(
        "select {} from events where id = ?",
        event_select_columns(true)
    );
    match conn.query_row(&modern_sql, [id], map_log_row).optional() {
        Ok(entry) => Ok(entry),
        Err(err) if is_legacy_log_schema_error(&err) => conn
            .query_row(
                &format!(
                    "select {} from events where id = ?",
                    event_select_columns(false)
                ),
                [id],
                map_log_row,
            )
            .optional()
            .context("Failed to load legacy log entry"),
        Err(err) => {
            let err_text = err.to_string();
            Err(err).with_context(|| {
                format!(
                    "Failed to load log entry with query: {modern_sql}; sqlite error: {err_text}"
                )
            })
        }
    }
}

fn get_by_reference_with_connection(
    conn: &Connection,
    reference: &str,
) -> Result<Option<LogEntry>> {
    if let Some(entry) = get_with_connection(conn, reference)? {
        return Ok(Some(entry));
    }

    let modern_sql = format!(
        "select {} from events where event_group_id = ? order by ts_utc desc limit 1",
        event_select_columns(true)
    );
    match conn
        .query_row(&modern_sql, [reference], map_log_row)
        .optional()
    {
        Ok(entry) => Ok(entry),
        Err(err) if is_legacy_log_schema_error(&err) => Ok(None),
        Err(err) => {
            let err_text = err.to_string();
            Err(err).with_context(|| {
                format!(
                    "Failed to load log entry by group reference with query: {modern_sql}; sqlite error: {err_text}"
                )
            })
        }
    }
}

fn aggregate_run_models_with_connection(
    conn: &Connection,
    cutoff: &str,
    tool_filter: Option<&str>,
) -> Result<HashMap<String, u64>> {
    let mut sql = String::from(
        "select model, count(*) from events \
         where kind = 'tool_launch' and phase = 'started' \
         and ts_utc >= ? and model is not null and trim(model) != ''",
    );
    let mut params: Vec<SqlValue> = vec![SqlValue::Text(cutoff.to_string())];
    if let Some(t) = tool_filter {
        sql.push_str(" and tool = ?");
        params.push(SqlValue::Text(t.to_string()));
    }
    sql.push_str(" group by model");

    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare run-model aggregation query")?;
    let rows = stmt
        .query_map(params_from_iter(params), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as u64))
        })
        .context("Failed to execute run-model aggregation query")?;
    let mut out = HashMap::new();
    for row in rows {
        let (model, count) = row.context("Failed to read run-model aggregation row")?;
        out.insert(model, count);
    }
    Ok(out)
}

fn count_runs_with_connection(
    conn: &Connection,
    cutoff: &str,
    tool_filter: Option<&str>,
) -> Result<u64> {
    let mut sql = String::from(
        "select count(*) from events \
         where kind = 'tool_launch' and phase = 'started' and ts_utc >= ?",
    );
    let mut params: Vec<SqlValue> = vec![SqlValue::Text(cutoff.to_string())];
    if let Some(t) = tool_filter {
        sql.push_str(" and tool = ?");
        params.push(SqlValue::Text(t.to_string()));
    }
    let count: i64 = conn
        .prepare(&sql)
        .context("Failed to prepare run-count query")?
        .query_row(params_from_iter(params), |row| row.get(0))
        .context("Failed to execute run-count query")?;
    Ok(count.max(0) as u64)
}

fn count_distinct_code_sessions_with_connection(conn: &Connection, cutoff: &str) -> Result<u64> {
    let count: i64 = conn
        .prepare(
            "select count(distinct session_id) from events \
             where source in ('chat','code') and session_id is not null and ts_utc >= ?",
        )
        .context("Failed to prepare chat-session-count query")?
        .query_row([cutoff], |row| row.get(0))
        .context("Failed to execute chat-session-count query")?;
    Ok(count.max(0) as u64)
}

fn aggregate_run_tokens_with_connection(
    conn: &Connection,
    cutoff: &str,
    tool_filter: Option<&str>,
) -> Result<HashMap<String, RunModelTokens>> {
    let mut sql = String::from(
        "select model, \
            coalesce(sum(input_tokens), 0), coalesce(sum(output_tokens), 0), \
            coalesce(sum(cache_read_input_tokens), 0), coalesce(sum(cache_creation_input_tokens), 0) \
         from events \
         where kind = 'tool_launch' and phase = 'finished' \
         and ts_utc >= ? and model is not null and trim(model) != ''",
    );
    let mut params: Vec<SqlValue> = vec![SqlValue::Text(cutoff.to_string())];
    if let Some(t) = tool_filter {
        sql.push_str(" and tool = ?");
        params.push(SqlValue::Text(t.to_string()));
    }
    // Coalesce each term: `finish_accounting` stamps zero fields as NULL, and a
    // single all-NULL column would otherwise NULL-poison the sum and drop the group.
    sql.push_str(
        " group by model having \
         coalesce(sum(input_tokens), 0) + coalesce(sum(output_tokens), 0) \
         + coalesce(sum(cache_read_input_tokens), 0) + coalesce(sum(cache_creation_input_tokens), 0) > 0",
    );

    let mut stmt = conn
        .prepare(&sql)
        .context("Failed to prepare run-token aggregation query")?;
    let rows = stmt
        .query_map(params_from_iter(params), |row| {
            Ok((
                row.get::<_, String>(0)?,
                RunModelTokens {
                    input: row.get::<_, i64>(1)?.max(0) as u64,
                    output: row.get::<_, i64>(2)?.max(0) as u64,
                    cache_read: row.get::<_, i64>(3)?.max(0) as u64,
                    cache_creation: row.get::<_, i64>(4)?.max(0) as u64,
                },
            ))
        })
        .context("Failed to execute run-token aggregation query")?;
    let mut out = HashMap::new();
    for row in rows {
        let (model, tokens) = row.context("Failed to read run-token aggregation row")?;
        out.insert(model, tokens);
    }
    Ok(out)
}

fn map_log_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<LogEntry> {
    let payload_json: Option<String> = row.get(22)?;
    let payload_json = payload_json.and_then(|raw| serde_json::from_str(&raw).ok());
    Ok(LogEntry {
        id: row.get(0)?,
        ts_utc: row.get(1)?,
        source: row.get(2)?,
        kind: row.get(3)?,
        event_group_id: row.get(4)?,
        phase: row.get(5)?,
        key_id: row.get(6)?,
        key_name: row.get(7)?,
        base_url: row.get(8)?,
        tool: row.get(9)?,
        model: row.get(10)?,
        cwd: row.get(11)?,
        session_id: row.get(12)?,
        status_code: row.get(13)?,
        exit_code: row.get(14)?,
        duration_ms: row.get(15)?,
        input_tokens: row.get(16)?,
        output_tokens: row.get(17)?,
        cache_read_input_tokens: row.get(18)?,
        cache_creation_input_tokens: row.get(19)?,
        title: row.get(20)?,
        body_text: row.get(21)?,
        payload_json,
    })
}

fn ensure_column_exists(
    conn: &Connection,
    table: &str,
    column: &str,
    column_type: &str,
) -> Result<()> {
    let pragma = format!("pragma table_info({table})");
    let mut stmt = conn
        .prepare(&pragma)
        .with_context(|| format!("Failed to inspect SQLite schema for {table}"))?;
    let found = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .context("Failed to read SQLite schema rows")?
        .filter_map(|row| row.ok())
        .any(|name| name == column);
    if !found {
        conn.execute(
            &format!("alter table {table} add column {column} {column_type}"),
            [],
        )
        .with_context(|| format!("Failed to add SQLite column {column} to {table}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn store(dir: &TempDir) -> LogStore {
        LogStore::new(dir.path().to_path_buf())
    }

    #[tokio::test]
    async fn append_and_get_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        let id = store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("run-1".to_string()),
                phase: Some("finished".to_string()),
                key_id: Some("key1".to_string()),
                key_name: Some("primary".to_string()),
                base_url: Some("https://api.openai.com".to_string()),
                tool: Some("claude".to_string()),
                model: Some("claude-sonnet-4-6".to_string()),
                cwd: Some("/repo".to_string()),
                exit_code: Some(0),
                duration_ms: Some(1234),
                title: Some("claude".to_string()),
                body_text: Some("--resume 123".to_string()),
                payload_json: Some(serde_json::json!({"args":["--resume","123"]})),
                ..Default::default()
            })
            .await
            .unwrap();

        let entry = store.get(&id).await.unwrap().unwrap();
        assert_eq!(entry.source, "run");
        assert_eq!(entry.tool.as_deref(), Some("claude"));
        assert_eq!(entry.exit_code, Some(0));
    }

    #[tokio::test]
    async fn list_supports_filters() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);

        store
            .append(LogEvent {
                source: "chat".to_string(),
                kind: "chat_turn".to_string(),
                key_id: Some("key1".to_string()),
                key_name: Some("alpha".to_string()),
                tool: Some("chat".to_string()),
                model: Some("gpt-4o".to_string()),
                cwd: Some("/repo".to_string()),
                session_id: Some("session-1".to_string()),
                duration_ms: Some(10),
                input_tokens: Some(10),
                output_tokens: Some(20),
                cache_read_input_tokens: Some(0),
                cache_creation_input_tokens: Some(0),
                title: Some("Summarize".to_string()),
                body_text: Some("User: summarize\nAssistant: ok".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        store
            .append(LogEvent {
                source: "serve".to_string(),
                kind: "serve_request".to_string(),
                key_id: Some("key2".to_string()),
                key_name: Some("beta".to_string()),
                tool: Some("serve".to_string()),
                model: Some("text-embedding-3-small".to_string()),
                status_code: Some(500),
                duration_ms: Some(42),
                title: Some("POST /v1/embeddings".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let filtered = store
            .list(LogQuery {
                limit: 10,
                by: Some("chat".to_string()),
                search: Some("summarize".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].source, "chat");

        let errors = store
            .list(LogQuery {
                limit: 10,
                errors_only: true,
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].source, "serve");
    }

    #[test]
    fn new_log_id_is_short_and_alphanumeric() {
        let id = new_log_id();
        assert_eq!(id.len(), 12);
        assert!(
            id.chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        );
    }

    #[tokio::test]
    async fn get_by_reference_returns_latest_group_event() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);

        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("runabc123xyz".to_string()),
                phase: Some("started".to_string()),
                tool: Some("claude".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let finished_id = store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("runabc123xyz".to_string()),
                phase: Some("finished".to_string()),
                tool: Some("claude".to_string()),
                exit_code: Some(0),
                duration_ms: Some(10),
                ..Default::default()
            })
            .await
            .unwrap();

        let entry = store
            .get_by_reference("runabc123xyz")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.id, finished_id);
        assert_eq!(entry.phase.as_deref(), Some("finished"));
    }

    #[tokio::test]
    async fn find_by_id_prefix_prefers_event_group_over_session_id() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("abcd1234zzzz".to_string()),
                phase: Some("finished".to_string()),
                tool: Some("claude".to_string()),
                session_id: Some("ffff-no-collide".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        // Different event_group_id, but session_id shares the prefix —
        // must not pollute the result.
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("eeee5555yyyy".to_string()),
                phase: Some("finished".to_string()),
                tool: Some("claude".to_string()),
                session_id: Some("abcd1234-uuid-suffix".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let hits = store.find_by_id_prefix("abcd1234", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].event_group_id.as_deref(), Some("abcd1234zzzz"));
    }

    #[tokio::test]
    async fn find_by_id_prefix_falls_back_to_session_id_when_no_aivo_match() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                event_group_id: Some("eeee5555yyyy".to_string()),
                phase: Some("finished".to_string()),
                tool: Some("claude".to_string()),
                session_id: Some("019e47b1-uuid-suffix".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let hits = store.find_by_id_prefix("019e47b1", 5).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].session_id.as_deref(), Some("019e47b1-uuid-suffix"));
    }

    #[tokio::test]
    async fn aggregate_run_models_since_groups_started_launches() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);

        // Two `started` launches for grok-4.3 under claude — should count 2.
        for _ in 0..2 {
            store
                .append(LogEvent {
                    source: "run".to_string(),
                    kind: "tool_launch".to_string(),
                    phase: Some("started".to_string()),
                    tool: Some("claude".to_string()),
                    model: Some("grok-4.3".to_string()),
                    ..Default::default()
                })
                .await
                .unwrap();
        }
        // A `finished` row for the same model — must NOT be double-counted.
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                phase: Some("finished".to_string()),
                tool: Some("claude".to_string()),
                model: Some("grok-4.3".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        // A `started` launch under codex for a different model.
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                phase: Some("started".to_string()),
                tool: Some("codex".to_string()),
                model: Some("kimi-k2.6".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        // A chat_turn (kind != tool_launch) — must be ignored.
        store
            .append(LogEvent {
                source: "chat".to_string(),
                kind: "chat_turn".to_string(),
                model: Some("deepseek-v4-flash".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();
        // A row with empty model — must be skipped.
        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                phase: Some("started".to_string()),
                tool: Some("claude".to_string()),
                model: Some("".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let all = store
            .aggregate_run_models_since(cutoff, None)
            .await
            .unwrap();
        assert_eq!(all.get("grok-4.3").copied(), Some(2));
        assert_eq!(all.get("kimi-k2.6").copied(), Some(1));
        assert!(!all.contains_key("deepseek-v4-flash"));
        assert!(!all.contains_key(""));

        let only_claude = store
            .aggregate_run_models_since(cutoff, Some("claude"))
            .await
            .unwrap();
        assert_eq!(only_claude.get("grok-4.3").copied(), Some(2));
        assert!(!only_claude.contains_key("kimi-k2.6"));
    }

    #[tokio::test]
    async fn aggregate_run_models_since_respects_cutoff() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);

        store
            .append(LogEvent {
                source: "run".to_string(),
                kind: "tool_launch".to_string(),
                phase: Some("started".to_string()),
                tool: Some("claude".to_string()),
                model: Some("grok-4.3".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        // Cutoff one hour in the future — the just-appended row is older than
        // the cutoff and must be excluded.
        let future = chrono::Utc::now() + chrono::Duration::hours(1);
        let map = store
            .aggregate_run_models_since(future, None)
            .await
            .unwrap();
        assert!(map.is_empty());
    }

    #[tokio::test]
    async fn count_distinct_code_sessions_since_dedupes_and_windows() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);

        let turn = |source: &str, session: Option<&str>| LogEvent {
            source: source.to_string(),
            kind: "code_turn".to_string(),
            session_id: session.map(str::to_string),
            ..Default::default()
        };

        // Two turns in one session count once; legacy 'chat' source counts too.
        store.append(turn("code", Some("s1"))).await.unwrap();
        store.append(turn("code", Some("s1"))).await.unwrap();
        store.append(turn("chat", Some("s2"))).await.unwrap();
        // Session-less one-shot and non-chat sources are excluded.
        store.append(turn("code", None)).await.unwrap();
        store.append(turn("run", Some("s3"))).await.unwrap();

        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        assert_eq!(
            store
                .count_distinct_code_sessions_since(past)
                .await
                .unwrap(),
            2
        );

        let future = chrono::Utc::now() + chrono::Duration::hours(1);
        assert_eq!(
            store
                .count_distinct_code_sessions_since(future)
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn aggregate_run_tokens_since_sums_finished_rows_by_model_and_tool() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);

        let finished = |tool: &str, model: &str, i: i64, o: i64, cr: i64, cc: i64| LogEvent {
            source: "run".to_string(),
            kind: "tool_launch".to_string(),
            phase: Some("finished".to_string()),
            tool: Some(tool.to_string()),
            model: Some(model.to_string()),
            // Mirror `finish_accounting`: zero fields are stamped as NULL, not 0.
            input_tokens: (i > 0).then_some(i),
            output_tokens: (o > 0).then_some(o),
            cache_read_input_tokens: (cr > 0).then_some(cr),
            cache_creation_input_tokens: (cc > 0).then_some(cc),
            ..Default::default()
        };

        // Two finished copilot runs on the same model sum together.
        store
            .append(finished("copilot", "gpt-5.2", 100, 50, 10, 5))
            .await
            .unwrap();
        store
            .append(finished("copilot", "gpt-5.2", 20, 10, 0, 0))
            .await
            .unwrap();
        // Zero-token finished row is dropped (it still shows as a launch elsewhere).
        store
            .append(finished("copilot", "idle", 0, 0, 0, 0))
            .await
            .unwrap();
        // A different tool is excluded by the tool filter.
        store
            .append(finished("claude", "claude-x", 7, 0, 0, 0))
            .await
            .unwrap();
        // A `started` row carrying tokens is excluded by the phase filter.
        store
            .append(LogEvent {
                phase: Some("started".to_string()),
                ..finished("copilot", "ghost", 999, 999, 0, 0)
            })
            .await
            .unwrap();
        // Output-only row (input + cache columns NULL): sqlite sum() over an
        // all-NULL group is NULL, which must not poison the HAVING filter.
        store
            .append(finished("copilot", "text-only", 0, 40, 0, 0))
            .await
            .unwrap();

        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        let map = store
            .aggregate_run_tokens_since(past, Some("copilot"))
            .await
            .unwrap();
        assert_eq!(
            map.len(),
            2,
            "the summed gpt-5.2 row and the output-only row survive: {map:?}"
        );
        let t = map.get("gpt-5.2").copied().unwrap();
        assert_eq!(
            (t.input, t.output, t.cache_read, t.cache_creation),
            (120, 60, 10, 5)
        );
        let t = map.get("text-only").copied().unwrap();
        assert_eq!(
            (t.input, t.output, t.cache_read, t.cache_creation),
            (0, 40, 0, 0)
        );
        assert!(!map.contains_key("idle"));
        assert!(!map.contains_key("ghost"));

        // No tool filter still excludes the zero-token + started rows, but now
        // includes the claude row.
        let all = store.aggregate_run_tokens_since(past, None).await.unwrap();
        assert_eq!(all.get("claude-x").map(|t| t.input), Some(7));
        assert!(all.contains_key("gpt-5.2"));

        // A future cutoff windows everything out.
        let future = chrono::Utc::now() + chrono::Duration::hours(1);
        assert!(
            store
                .aggregate_run_tokens_since(future, Some("copilot"))
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn count_runs_since_counts_started_rows_including_model_less() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        let started = |tool: &str, model: Option<&str>| LogEvent {
            source: "run".to_string(),
            kind: "tool_launch".to_string(),
            phase: Some("started".to_string()),
            tool: Some(tool.to_string()),
            model: model.map(|m| m.to_string()),
            ..Default::default()
        };

        store
            .append(started("copilot", Some("gpt-5.2")))
            .await
            .unwrap();
        // A launch that never resolved a model still counts.
        store.append(started("copilot", None)).await.unwrap();
        store
            .append(started("claude", Some("claude-x")))
            .await
            .unwrap();
        // Finished rows are not launches.
        store
            .append(LogEvent {
                phase: Some("finished".to_string()),
                ..started("copilot", Some("gpt-5.2"))
            })
            .await
            .unwrap();

        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        assert_eq!(
            store.count_runs_since(past, Some("copilot")).await.unwrap(),
            2
        );
        assert_eq!(store.count_runs_since(past, None).await.unwrap(), 3);

        let future = chrono::Utc::now() + chrono::Duration::hours(1);
        assert_eq!(
            store
                .count_runs_since(future, Some("copilot"))
                .await
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn aggregate_run_models_since_returns_empty_when_db_missing() {
        let dir = TempDir::new().unwrap();
        let store = store(&dir);
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        let map = store
            .aggregate_run_models_since(cutoff, None)
            .await
            .unwrap();
        assert!(map.is_empty());
    }
}
