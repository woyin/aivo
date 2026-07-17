use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::Result;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::services::context_ingest::{is_gemini_session_file, normalize_gemini_session};
use crate::services::model_names::normalize_claude_version;
use crate::services::system_env;

/// Aggregated stats from a tool's native data files.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct GlobalToolStats {
    pub sessions: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub models: HashMap<String, ModelTokens>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ModelTokens {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_read_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub cache_write_tokens: u64,
}

impl ModelTokens {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

impl GlobalToolStats {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

// ---------------------------------------------------------------------------
// Per-file cache: stores stats per file keyed by path, with file size for
// change detection. Only files whose size changed get re-parsed.
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Default)]
struct FileEntry {
    size: u64,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    models: HashMap<String, ModelTokens>,
    has_session: bool,
}

#[derive(Serialize, Deserialize, Default)]
struct StatsCache {
    files: HashMap<String, FileEntry>,
}

/// Delta parses keyed to the `lastComputedDate` they were cut against;
/// a recomputed date invalidates them all.
#[derive(Serialize, Deserialize, Default)]
struct DeltaCache {
    cutoff_date: String,
    files: HashMap<String, FileEntry>,
}

/// Collect global stats for all known tools sequentially.
/// Sequential avoids progress line flickering (all tools share one stderr line).
/// Returns a map of tool name → stats (only tools with data).
///
/// `extra_steps` (the caller's following plugin-probe steps) folds into the
/// `(x/N)` denominator so it spans native tools *and* plugins, not a constant 5.
pub async fn collect_all(
    refresh: bool,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
    extra_steps: usize,
) -> HashMap<String, GlobalToolStats> {
    let Some(home) = system_env::home_dir() else {
        return HashMap::new();
    };
    collect_all_in(&home, refresh, cutoff, extra_steps).await
}

/// [`collect_all`] against an explicit home dir, so tests can drive it with
/// fixture files instead of the process environment.
async fn collect_all_in(
    home: &Path,
    refresh: bool,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
    extra_steps: usize,
) -> HashMap<String, GlobalToolStats> {
    let tools = native_present_tools(home);
    let total_steps = tools.len() + extra_steps;
    let mut result = HashMap::new();
    for (i, &tool) in tools.iter().enumerate() {
        let step = Some((i + 1, total_steps));
        if let Ok(Some(stats)) = collect_with_step(home, tool, refresh, step, cutoff).await
            && (stats.total_tokens() > 0 || stats.sessions > 0)
        {
            result.insert(tool.to_string(), stats);
        }
    }
    result
}

pub async fn collect(
    tool: &str,
    refresh: bool,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Result<Option<GlobalToolStats>> {
    let Some(home) = system_env::home_dir() else {
        return Ok(None);
    };
    collect_with_step(&home, tool, refresh, None, cutoff).await
}

async fn collect_with_step(
    home: &Path,
    tool: &str,
    refresh: bool,
    step: Option<(usize, usize)>,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Result<Option<GlobalToolStats>> {
    // Prefer Claude Code's own ~/.claude/stats-cache.json — it's the same
    // data source its `/stats` UI uses, so totals match exactly. The cache
    // persists across JSONL pruning, which a raw walk of ~/.claude/projects
    // cannot reproduce. Skip this short-circuit when `cutoff` is set: the
    // stats-cache holds lifetime-only totals with no per-period breakdown.
    if tool == "claude"
        && cutoff.is_none()
        && let Some(stats) = collect_claude_from_cache(home, refresh, step).await
    {
        return Ok(Some(stats));
    }

    if !matches!(tool, "claude" | "codex" | "gemini") {
        return match tool {
            "opencode" => collect_opencode(home, cutoff).await,
            "pi" => collect_pi(home, cutoff).await,
            _ => Ok(None),
        };
    }

    let data_dir = match tool_data_dir(home, tool) {
        Some(d) if d.exists() => d,
        _ => return Ok(None),
    };

    let filter = tool_file_filter(tool);
    let cache_path = cache_path(tool);
    // When `cutoff` is set, parsed `FileEntry`s contain only post-cutoff
    // token totals. Persisting them would corrupt later non-cutoff runs
    // (size matches → no re-parse → wrong lifetime totals). Treat cutoff
    // as "rebuild fresh, in-memory only, then drop" — same semantic as
    // `refresh: true`, but with no write-back.
    let mut cache = if cutoff.is_some() || refresh {
        StatsCache::default()
    } else {
        read_cache(&cache_path).await.unwrap_or_default()
    };

    // Walk files and collect paths + sizes
    let all_files = walk_files_with_size(&data_dir, filter).await;
    if all_files.is_empty() {
        return Ok(None);
    }

    // Find stale files (new or size changed)
    let current_paths: HashSet<&str> = all_files
        .iter()
        .map(|(p, _, _)| p.to_str().unwrap_or(""))
        .collect();

    let mut stale: Vec<(&Path, u64)> = Vec::new();
    for (path, size, _) in &all_files {
        let key = path.to_string_lossy();
        match cache.files.get(key.as_ref()) {
            Some(cached) if cached.size == *size => {} // unchanged
            _ => stale.push((path, *size)),
        }
    }

    // Remove deleted files from cache
    cache
        .files
        .retain(|k, _| current_paths.contains(k.as_str()));

    // Re-parse stale files
    if !stale.is_empty() {
        let total = stale.len();

        let show_progress = total > 5 && std::io::IsTerminal::is_terminal(&std::io::stderr());
        let update_interval = (total / 50).max(1);
        if show_progress {
            print_progress(0, total, step);
        }

        for (i, (path, size)) in stale.iter().enumerate() {
            let entry_opt = match tool {
                "claude" => parse_claude_file_with_cutoff(path, cutoff).await,
                "codex" => parse_codex_file(path, cutoff).await,
                "gemini" => parse_gemini_file(path, cutoff).await,
                _ => None,
            };
            // Cache no-usage parses too — uncached files re-parse every run.
            let entry = entry_opt.unwrap_or_default();
            cache.files.insert(
                path.to_string_lossy().to_string(),
                FileEntry {
                    size: *size,
                    ..entry
                },
            );
            if show_progress && ((i + 1) % update_interval == 0 || i + 1 == total) {
                print_progress(i + 1, total, step);
            }
        }

        if show_progress {
            eprint!("\r{:<30}\r", "");
        }
        if cutoff.is_none() {
            let _ = write_cache(&cache_path, &cache).await;
        }
    }

    // When `cutoff` is set the per-file parsers already filter by event or
    // message timestamp, so aggregating by file mtime would skew results.
    let stats = aggregate_cache(&cache);
    if stats.sessions == 0 && stats.total_tokens() == 0 {
        return Ok(None);
    }
    Ok(Some(stats))
}

fn aggregate_cache(cache: &StatsCache) -> GlobalToolStats {
    let mut stats = GlobalToolStats::default();
    for entry in cache.files.values() {
        stats.input_tokens += entry.input_tokens;
        stats.output_tokens += entry.output_tokens;
        stats.cache_read_tokens += entry.cache_read_tokens;
        stats.cache_write_tokens += entry.cache_write_tokens;
        if entry.has_session {
            stats.sessions += 1;
        }
        for (model, mt) in &entry.models {
            let m = stats.models.entry(model.clone()).or_default();
            m.input_tokens += mt.input_tokens;
            m.output_tokens += mt.output_tokens;
            m.cache_read_tokens += mt.cache_read_tokens;
            m.cache_write_tokens += mt.cache_write_tokens;
        }
    }
    stats
}

#[cfg(test)]
fn aggregate_cache_filtered(
    cache: &StatsCache,
    mtimes: &HashMap<String, SystemTime>,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> GlobalToolStats {
    let cutoff_st = cutoff.and_then(cutoff_to_systemtime);

    let mut stats = GlobalToolStats::default();
    for (path, entry) in &cache.files {
        if let (Some(c), Some(m)) = (cutoff_st, mtimes.get(path))
            && *m < c
        {
            continue;
        }
        stats.input_tokens += entry.input_tokens;
        stats.output_tokens += entry.output_tokens;
        stats.cache_read_tokens += entry.cache_read_tokens;
        stats.cache_write_tokens += entry.cache_write_tokens;
        if entry.has_session {
            stats.sessions += 1;
        }
        for (model, mt) in &entry.models {
            let m = stats.models.entry(model.clone()).or_default();
            m.input_tokens += mt.input_tokens;
            m.output_tokens += mt.output_tokens;
            m.cache_read_tokens += mt.cache_read_tokens;
            m.cache_write_tokens += mt.cache_write_tokens;
        }
    }
    stats
}

// ---------------------------------------------------------------------------
// Infrastructure helpers
// ---------------------------------------------------------------------------

/// Built-in tools aivo scans directly, as opposed to coding-agent plugins
/// (accounted via launch counts and `--aivo-stats` probes).
pub fn is_native_tool(tool: &str) -> bool {
    matches!(tool, "claude" | "codex" | "gemini" | "opencode" | "pi")
}

/// True when `tool`'s native data store exists on disk.
fn native_data_present(home: &Path, tool: &str) -> bool {
    match tool {
        // claude's stats-cache.json survives projects-dir pruning, so either counts.
        "claude" => {
            home.join(".claude").join("projects").exists()
                || home.join(".claude").join("stats-cache.json").exists()
        }
        "codex" => home.join(".codex").join("sessions").exists(),
        "gemini" => home.join(".gemini").join("tmp").exists(),
        "opencode" => home
            .join(".local")
            .join("share")
            .join("opencode")
            .join("opencode.db")
            .exists(),
        "pi" => home.join(".pi").join("agent").join("sessions").exists(),
        _ => false,
    }
}

/// Native tools (in scan order) whose data is present on disk.
fn native_present_tools(home: &Path) -> Vec<&'static str> {
    ["claude", "codex", "gemini", "opencode", "pi"]
        .into_iter()
        .filter(|t| native_data_present(home, t))
        .collect()
}

/// Native-tool count for the next `collect_all`; the caller adds its probe
/// count to size the `(x/N)` counter.
pub fn native_present_count() -> usize {
    system_env::home_dir()
        .map(|home| native_present_tools(&home).len())
        .unwrap_or(0)
}

fn tool_data_dir(home: &Path, tool: &str) -> Option<PathBuf> {
    match tool {
        "claude" => Some(home.join(".claude").join("projects")),
        "codex" => Some(home.join(".codex").join("sessions")),
        "gemini" => Some(home.join(".gemini").join("tmp")),
        _ => None,
    }
}

fn cache_path(tool: &str) -> PathBuf {
    crate::services::paths::stats_cache(&crate::services::paths::config_dir(), tool)
}

fn tool_file_filter(tool: &str) -> fn(&str) -> bool {
    match tool {
        "claude" | "codex" => |name: &str| name.ends_with(".jsonl"),
        "gemini" => is_gemini_session_file,
        _ => |_: &str| true,
    }
}

/// Walk directory recursively, returning matching files with their size and
/// last-modified time (both may be `0`/`None` when metadata is unreadable).
async fn walk_files_with_size(
    root: &Path,
    filter: fn(&str) -> bool,
) -> Vec<(PathBuf, u64, Option<SystemTime>)> {
    let mut result = Vec::new();
    let mut dirs = vec![root.to_path_buf()];

    while let Some(dir) = dirs.pop() {
        let mut entries = match fs::read_dir(&dir).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && filter(name)
            {
                let meta = entry.metadata().await.ok();
                let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                let mtime = meta.and_then(|m| m.modified().ok());
                result.push((path, size, mtime));
            }
        }
    }

    result
}

fn print_progress(current: usize, total: usize, step: Option<(usize, usize)>) {
    let pct = (current * 100).checked_div(total).unwrap_or(0);
    let step_prefix = match step {
        Some((i, n)) => format!("({i}/{n}) "),
        None => String::new(),
    };
    eprint!(
        "\r{}{} {pct:>3}%",
        step_prefix,
        crate::style::dim("reading")
    );
}

/// Render one probe-phase step on the shared progress line: a step counter and
/// the plugin name (fixed-width, so a shorter name can't leave a tail behind).
pub fn render_step(current: usize, total: usize, detail: &str) {
    eprint!(
        "\r({current}/{total}) {} {:<12}",
        crate::style::dim("reading"),
        detail
    );
}

/// Clear the shared progress line once the probe phase finishes.
pub fn clear_progress_line() {
    eprint!("\r{:<40}\r", "");
}

async fn read_cache<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let data = fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&data).ok()
}

async fn write_cache<T: Serialize>(path: &Path, cache: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).await?;
    }
    let data = serde_json::to_string(cache)?;
    // Atomic rename: concurrent aivo processes refresh this cache; a plain
    // write could interleave and leave torn JSON.
    crate::services::atomic_write::atomic_write_secure(path, data.into_bytes()).await
}

// ---------------------------------------------------------------------------
// Per-file parsers — return FileEntry for a single file
// ---------------------------------------------------------------------------

/// Read Claude Code's own persistent stats cache at
/// `~/.claude/stats-cache.json`. Returns `None` when the file is missing
/// or malformed; callers fall back to walking the JSONL files directly.
///
/// Claude Code merges historical session totals into this cache even after
/// the underlying JSONL files are pruned, so reading it directly is the
/// only way to reproduce the totals shown in Claude Code's `/stats` UI.
async fn collect_claude_from_cache(
    home: &Path,
    refresh: bool,
    step: Option<(usize, usize)>,
) -> Option<GlobalToolStats> {
    let cc_cache_path = home.join(".claude").join("stats-cache.json");
    let data = fs::read_to_string(&cc_cache_path).await.ok()?;
    let v: Value = serde_json::from_str(&data).ok()?;
    let mut stats = parse_claude_stats_cache(&v)?;

    // Claude Code stamps the cache with `lastComputedDate` (YYYY-MM-DD) and
    // processes any JSONL activity beyond that live when rendering `/stats`.
    // Replay the same merge so aivo shows the same live total.
    if let Some(cutoff) = v.get("lastComputedDate").and_then(|s| s.as_str()) {
        let projects_dir = home.join(".claude").join("projects");
        merge_claude_jsonl_deltas(
            &projects_dir,
            cutoff,
            &mut stats,
            &cache_path("claude-delta"),
            refresh,
            step,
        )
        .await;
    }

    Some(stats)
}

/// Walk `dir`/**/*.jsonl and fold any assistant activity dated after
/// `cutoff_date` (YYYY-MM-DD, UTC) into `stats`. Files whose mtime is
/// strictly before the day after `cutoff_date` are skipped without being
/// opened; parses persist at `delta_cache_path` — the window grows until
/// Claude Code recomputes its stats-cache.
async fn merge_claude_jsonl_deltas(
    dir: &Path,
    cutoff_date: &str,
    stats: &mut GlobalToolStats,
    delta_cache_path: &Path,
    refresh: bool,
    step: Option<(usize, usize)>,
) {
    let mtime_threshold = day_after_start_utc(cutoff_date);
    let Some(cutoff_dt) = NaiveDate::parse_from_str(cutoff_date, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.succ_opt())
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|ndt| ndt.and_utc())
    else {
        return;
    };
    let files = walk_files_with_size(dir, |name| name.ends_with(".jsonl")).await;

    let mut cache: DeltaCache = if refresh {
        DeltaCache::default()
    } else {
        read_cache(delta_cache_path).await.unwrap_or_default()
    };
    let date_reset = cache.cutoff_date != cutoff_date;
    if date_reset {
        cache = DeltaCache {
            cutoff_date: cutoff_date.to_string(),
            files: HashMap::new(),
        };
    }

    let in_window: Vec<(&Path, u64)> = files
        .iter()
        .filter(|(_, _, mtime)| {
            !matches!((mtime_threshold, mtime.as_ref()), (Some(t), Some(m)) if *m < t)
        })
        .map(|(path, size, _)| (path.as_path(), *size))
        .collect();

    let current: HashSet<String> = in_window
        .iter()
        .map(|(p, _)| p.to_string_lossy().to_string())
        .collect();
    let before = cache.files.len();
    cache.files.retain(|k, _| current.contains(k));
    let pruned = cache.files.len() != before;

    let stale: Vec<(&Path, u64)> = in_window
        .iter()
        .filter(|(p, size)| {
            cache
                .files
                .get(p.to_string_lossy().as_ref())
                .is_none_or(|e| e.size != *size)
        })
        .copied()
        .collect();

    let total = stale.len();
    let show_progress = total > 5 && std::io::IsTerminal::is_terminal(&std::io::stderr());
    let update_interval = (total / 50).max(1);
    if show_progress {
        print_progress(0, total, step);
    }
    for (i, (path, size)) in stale.iter().enumerate() {
        let entry = parse_claude_file_with_cutoff(path, Some(cutoff_dt))
            .await
            .unwrap_or_default();
        cache.files.insert(
            path.to_string_lossy().to_string(),
            FileEntry {
                size: *size,
                ..entry
            },
        );
        if show_progress && ((i + 1) % update_interval == 0 || i + 1 == total) {
            print_progress(i + 1, total, step);
        }
    }
    if show_progress {
        eprint!("\r{:<30}\r", "");
    }
    if date_reset || pruned || !stale.is_empty() {
        let _ = write_cache(delta_cache_path, &cache).await;
    }

    for (path, _) in &in_window {
        let Some(entry) = cache.files.get(path.to_string_lossy().as_ref()) else {
            continue;
        };
        stats.input_tokens += entry.input_tokens;
        stats.output_tokens += entry.output_tokens;
        stats.cache_read_tokens += entry.cache_read_tokens;
        stats.cache_write_tokens += entry.cache_write_tokens;
        if entry.has_session {
            stats.sessions += 1;
        }
        for (model, mt) in &entry.models {
            let m = stats.models.entry(model.clone()).or_default();
            m.input_tokens += mt.input_tokens;
            m.output_tokens += mt.output_tokens;
            m.cache_read_tokens += mt.cache_read_tokens;
            m.cache_write_tokens += mt.cache_write_tokens;
        }
    }
}

/// Start-of-day UTC for the day *after* `date` (YYYY-MM-DD). Used as an
/// mtime threshold: any file modified strictly before this can't contain
/// activity newer than `date`.
fn day_after_start_utc(date: &str) -> Option<SystemTime> {
    let next = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .ok()?
        .succ_opt()?;
    let ts = next.and_hms_opt(0, 0, 0)?.and_utc().timestamp();
    Some(SystemTime::UNIX_EPOCH + Duration::from_secs(u64::try_from(ts).ok()?))
}

#[cfg(test)]
fn cutoff_to_systemtime(cutoff: chrono::DateTime<chrono::Utc>) -> Option<SystemTime> {
    let secs = u64::try_from(cutoff.timestamp()).ok()?;
    Some(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
}

/// Pure parser for Claude Code's `stats-cache.json` schema. Split out for
/// unit testing without filesystem IO.
fn parse_claude_stats_cache(v: &Value) -> Option<GlobalToolStats> {
    let model_usage = v.get("modelUsage").and_then(|m| m.as_object())?;

    let mut stats = GlobalToolStats {
        sessions: v.get("totalSessions").and_then(|s| s.as_u64()).unwrap_or(0),
        ..Default::default()
    };

    for (model_name, mu) in model_usage {
        let input = mu.get("inputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = mu.get("outputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let cache_read = mu
            .get("cacheReadInputTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_create = mu
            .get("cacheCreationInputTokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        stats.input_tokens += input;
        stats.output_tokens += output;
        stats.cache_read_tokens += cache_read;
        stats.cache_write_tokens += cache_create;

        let key = normalize_model_for_display(model_name);
        let m = stats.models.entry(key).or_default();
        m.input_tokens += input;
        m.output_tokens += output;
        m.cache_read_tokens += cache_read;
        m.cache_write_tokens += cache_create;
    }

    if stats.sessions == 0 && stats.total_tokens() == 0 {
        return None;
    }
    Some(stats)
}

/// Parse a single Claude Code JSONL file.
///
/// When `cutoff` is `Some(dt)`, assistant lines whose RFC3339 `timestamp`
/// is strictly before `dt` (or is missing/unparseable) are skipped. Used
/// by both the stats-cache delta merge and the `--since` filter on the
/// direct JSONL walk.
async fn parse_claude_file_with_cutoff(
    path: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Option<FileEntry> {
    let file = fs::File::open(path).await.ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut entry = FileEntry::default();
    let mut seen_session = false;

    while let Ok(Some(line)) = lines.next_line().await {
        // Fast pre-filter: skip full JSON parse for non-assistant lines
        if !line.contains("\"type\":\"assistant\"") {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(c) = cutoff {
            let ts_str = v.get("timestamp").and_then(|t| t.as_str()).unwrap_or("");
            let ts = match chrono::DateTime::parse_from_rfc3339(ts_str) {
                Ok(t) => t.with_timezone(&chrono::Utc),
                Err(_) => continue,
            };
            if ts < c {
                continue;
            }
        }
        let usage = match v.get("message").and_then(|m| m.get("usage")) {
            Some(u) => u,
            None => continue,
        };
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cache_write = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cache_read;
        entry.cache_write_tokens += cache_write;

        // Skip sidechain (subagent) assistant lines when deciding whether
        // this file represents a real user-facing session. Claude Code
        // stores each Task subagent conversation in its own `agent-*.jsonl`
        // file under `<session>/subagents/`, tagged `isSidechain: true`.
        // Counting those inflates the session count by the number of
        // subagent invocations. Tokens above are still accumulated because
        // subagent calls are real API spend.
        let is_sidechain = v
            .get("isSidechain")
            .and_then(|s| s.as_bool())
            .unwrap_or(false);
        if !seen_session && !is_sidechain && v.get("sessionId").and_then(|s| s.as_str()).is_some() {
            seen_session = true;
            entry.has_session = true;
        }
        if let Some(model) = v
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|m| m.as_str())
        {
            let key = normalize_model_for_display(model);
            let e = entry.models.entry(key).or_default();
            e.input_tokens += input;
            e.output_tokens += output;
            e.cache_read_tokens += cache_read;
            e.cache_write_tokens += cache_write;
        }
    }

    Some(entry)
}

fn parse_rfc3339_utc(value: &Value, key: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    chrono::DateTime::parse_from_rfc3339(value.get(key)?.as_str()?)
        .ok()
        .map(|ts| ts.with_timezone(&chrono::Utc))
}

/// Parse a single Codex JSONL file.
async fn parse_codex_file(
    path: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Option<FileEntry> {
    let file = fs::File::open(path).await.ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut prev_input = 0u64;
    let mut prev_output = 0u64;
    let mut prev_cached = 0u64;
    let mut saw_usage = false;
    let mut model: Option<String> = None;
    let mut entry = FileEntry::default();

    while let Ok(Some(line)) = lines.next_line().await {
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if v.get("type").and_then(|t| t.as_str()) == Some("turn_context")
            && let Some(m) = v
                .get("payload")
                .and_then(|p| p.get("model"))
                .and_then(|m| m.as_str())
        {
            model = Some(m.to_string());
        }

        if v.get("type").and_then(|t| t.as_str()) != Some("event_msg") {
            continue;
        }
        let payload = match v.get("payload") {
            Some(p) => p,
            None => continue,
        };
        if payload.get("type").and_then(|t| t.as_str()) != Some("token_count") {
            continue;
        }
        let info = match payload.get("info") {
            Some(info @ Value::Object(_)) => info,
            _ => continue,
        };
        let usage = match info.get("total_token_usage") {
            Some(u) => u,
            None => continue,
        };

        let total_input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let total_output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let total_cached = usage
            .get("cached_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let delta_input = total_input.saturating_sub(prev_input);
        let delta_output = total_output.saturating_sub(prev_output);
        let delta_cached = total_cached.saturating_sub(prev_cached);
        prev_input = total_input;
        prev_output = total_output;
        prev_cached = total_cached;
        saw_usage = true;

        if let Some(c) = cutoff {
            let Some(ts) = parse_rfc3339_utc(&v, "timestamp") else {
                continue;
            };
            if ts < c {
                continue;
            }
        }

        let fresh_input = delta_input.saturating_sub(delta_cached);
        entry.has_session = true;
        entry.input_tokens += fresh_input;
        entry.output_tokens += delta_output;
        entry.cache_read_tokens += delta_cached;

        if let Some(ref m) = model {
            let key = normalize_model_for_display(m);
            let e = entry.models.entry(key).or_default();
            e.input_tokens += fresh_input;
            e.output_tokens += delta_output;
            e.cache_read_tokens += delta_cached;
        }
    }

    if !saw_usage {
        return None;
    }

    Some(entry)
}

/// Parse a single Gemini session file. Token usage lives on `type:"gemini"`
/// records under `tokens.{input,output,cached}`.
///
/// `normalize_gemini_session` absorbs the format difference (legacy
/// single-object `session-*.json` vs. current per-line `chats/session-*.jsonl`),
/// handing back a uniform `{messages:[…]}` either way.
async fn parse_gemini_file(
    path: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Option<FileEntry> {
    let content = fs::read_to_string(path).await.ok()?;
    let session = normalize_gemini_session(&content)?;
    let messages = session.get("messages").and_then(|m| m.as_array())?;

    let mut entry = FileEntry::default();

    for msg in messages {
        if msg.get("type").and_then(|t| t.as_str()) != Some("gemini") {
            continue;
        }
        if let Some(c) = cutoff {
            let Some(ts) = parse_rfc3339_utc(msg, "timestamp") else {
                continue;
            };
            if ts < c {
                continue;
            }
        }
        let tokens = match msg.get("tokens") {
            Some(t) => t,
            None => continue,
        };

        let input = tokens.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = tokens.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
        let cached = tokens.get("cached").and_then(|v| v.as_u64()).unwrap_or(0);

        // Gemini's `tokens.input` is the total turn input INCLUDING the
        // `tokens.cached` portion. Normalize to fresh-only so the
        // aggregated "tokens" column doesn't overlap with "cached".
        let fresh_input = input.saturating_sub(cached);
        entry.has_session = true;
        entry.input_tokens += fresh_input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cached;

        if let Some(model) = msg.get("model").and_then(|m| m.as_str()) {
            let key = normalize_model_for_display(model);
            let e = entry.models.entry(key).or_default();
            e.input_tokens += fresh_input;
            e.output_tokens += output;
            e.cache_read_tokens += cached;
        }
    }

    Some(entry)
}

// ---------------------------------------------------------------------------
// Non-cached tool collectors (OpenCode via SQLite, Pi)
// ---------------------------------------------------------------------------

/// OpenCode: ~/.local/share/opencode/opencode.db (SQLite via rusqlite)
async fn collect_opencode(
    home: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Result<Option<GlobalToolStats>> {
    let db_path = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("opencode.db");
    if !db_path.exists() {
        return Ok(None);
    }

    tokio::task::spawn_blocking(move || {
        let conn = match rusqlite::Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) {
            Ok(c) => c,
            Err(_) => return Ok(None),
        };
        match aggregate_opencode_messages(&conn, cutoff) {
            Ok(stats) if stats.sessions > 0 => Ok(Some(stats)),
            _ => Ok(None),
        }
    })
    .await?
}

/// Sum tokens from opencode's `message` table. `time_created` is epoch
/// milliseconds; `COALESCE(?1, time_created)` lets the caller skip the
/// cutoff by binding NULL.
fn aggregate_opencode_messages(
    conn: &rusqlite::Connection,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> rusqlite::Result<GlobalToolStats> {
    let mut stmt = conn.prepare(
        "SELECT session_id,
                json_extract(data, '$.modelID'),
                json_extract(data, '$.tokens.input'),
                json_extract(data, '$.tokens.output'),
                json_extract(data, '$.tokens.cache.read'),
                json_extract(data, '$.tokens.cache.write')
         FROM message
         WHERE json_extract(data, '$.role') = 'assistant'
           AND json_extract(data, '$.tokens') IS NOT NULL
           AND time_created >= COALESCE(?1, time_created)",
    )?;
    let cutoff_ms: Option<i64> = cutoff.map(|c| c.timestamp_millis());
    let rows = stmt.query_map([cutoff_ms], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1).unwrap_or_default(),
            row.get::<_, u64>(2).unwrap_or(0),
            row.get::<_, u64>(3).unwrap_or(0),
            row.get::<_, u64>(4).unwrap_or(0),
            row.get::<_, u64>(5).unwrap_or(0),
        ))
    })?;

    let mut stats = GlobalToolStats::default();
    let mut session_ids = HashSet::new();
    for row in rows.filter_map(Result::ok) {
        let (session_id, model, input, output, cache_read, cache_write) = row;
        session_ids.insert(session_id);
        stats.input_tokens += input;
        stats.output_tokens += output;
        stats.cache_read_tokens += cache_read;
        stats.cache_write_tokens += cache_write;
        if !model.is_empty() {
            let key = normalize_model_for_display(&model);
            let entry = stats.models.entry(key).or_default();
            entry.input_tokens += input;
            entry.output_tokens += output;
            entry.cache_read_tokens += cache_read;
            entry.cache_write_tokens += cache_write;
        }
    }
    stats.sessions = session_ids.len() as u64;
    Ok(stats)
}

/// Pi: ~/.pi/agent/sessions/**/*.jsonl
async fn collect_pi(
    home: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Result<Option<GlobalToolStats>> {
    let data_dir = home.join(".pi").join("agent").join("sessions");
    if !data_dir.exists() {
        return Ok(None);
    }

    let files = walk_files_with_size(&data_dir, |name| name.ends_with(".jsonl")).await;
    if files.is_empty() {
        return Ok(None);
    }

    let mut stats = GlobalToolStats::default();
    let mut session_ids: HashSet<String> = HashSet::new();

    for (path, _, _) in &files {
        if let Some((entry, ids)) = parse_pi_file(path, cutoff).await {
            stats.input_tokens += entry.input_tokens;
            stats.output_tokens += entry.output_tokens;
            stats.cache_read_tokens += entry.cache_read_tokens;
            stats.cache_write_tokens += entry.cache_write_tokens;
            for (model, mt) in entry.models {
                let m = stats.models.entry(model).or_default();
                m.input_tokens += mt.input_tokens;
                m.output_tokens += mt.output_tokens;
                m.cache_read_tokens += mt.cache_read_tokens;
                m.cache_write_tokens += mt.cache_write_tokens;
            }
            for id in ids {
                session_ids.insert(id);
            }
        }
    }

    stats.sessions = session_ids.len() as u64;
    Ok(Some(stats))
}

/// Parse a single Pi JSONL file.
///
/// Returns the per-file token totals plus any session ids found in
/// `type:session` records (Pi's session ids are file-scoped but tracked
/// globally to dedupe across files).
async fn parse_pi_file(
    path: &Path,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Option<(FileEntry, Vec<String>)> {
    let file = fs::File::open(path).await.ok()?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut entry = FileEntry::default();
    let mut session_id: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if v.get("type").and_then(|t| t.as_str()) == Some("session")
            && let Some(sid) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }

        if v.get("type").and_then(|t| t.as_str()) != Some("message") {
            continue;
        }
        if let Some(c) = cutoff {
            let Some(ts) = parse_rfc3339_utc(&v, "timestamp") else {
                continue;
            };
            if ts < c {
                continue;
            }
        }

        let usage = match v.get("message").and_then(|m| m.get("usage")) {
            Some(u) => u,
            None => continue,
        };

        let input = usage.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
        let output = usage.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
        let cache_read = usage.get("cacheRead").and_then(|v| v.as_u64()).unwrap_or(0);
        let cache_write = usage
            .get("cacheWrite")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        // Pi's `usage.input` already excludes `cacheRead`, so it matches
        // Claude's fresh-only convention as-is. Don't add cache to it.
        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.cache_read_tokens += cache_read;
        entry.cache_write_tokens += cache_write;

        if let Some(model) = v
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|m| m.as_str())
        {
            let key = normalize_model_for_display(model);
            let e = entry.models.entry(key).or_default();
            e.input_tokens += input;
            e.output_tokens += output;
            e.cache_read_tokens += cache_read;
            e.cache_write_tokens += cache_write;
        }
    }

    entry.has_session = entry.input_tokens > 0
        || entry.output_tokens > 0
        || entry.cache_read_tokens > 0
        || entry.cache_write_tokens > 0;
    let session_ids = if entry.has_session {
        session_id.into_iter().collect()
    } else {
        Vec::new()
    };
    Some((entry, session_ids))
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/// Normalize a model name for display and merging.
/// Strips provider prefixes, normalizes version separators, lowercases.
pub fn normalize_model_for_display(model: &str) -> String {
    let base = if let Some(pos) = model.rfind('/') {
        &model[pos + 1..]
    } else {
        model
    };
    let normalized = normalize_claude_version(base);
    normalized.to_lowercase()
}

/// Display name for each tool.
pub fn tool_display_name(tool: &str) -> &str {
    match tool {
        "claude" => "Claude Code",
        "codex" => "Codex",
        "gemini" => "Gemini",
        "opencode" => "OpenCode",
        "pi" => "Pi",
        // `chat` is the pre-rename bucket; both render as the built-in agent.
        "chat" | "code" => "Code",
        _ => tool,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claude_assistant_line(ts: &str, input: u64, output: u64) -> String {
        format!(
            r#"{{"type":"assistant","timestamp":"{ts}","sessionId":"s1","message":{{"model":"claude-opus-4-8","usage":{{"input_tokens":{input},"output_tokens":{output}}}}}}}"#
        )
    }

    // All fixture-driven collects pass `refresh: true`: aivo's own stats cache
    // lives under the shared (sandboxed) config dir, and skipping reads keeps
    // parallel tests from seeing each other's write-backs.

    #[tokio::test]
    async fn collect_all_walks_claude_jsonl_and_applies_cutoff() {
        let tmp = tempfile::TempDir::new().unwrap();
        let home = tmp.path();
        let proj = home.join(".claude").join("projects").join("p1");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(
            proj.join("session1.jsonl"),
            format!(
                "{}\n{}\n",
                claude_assistant_line("2026-01-10T10:00:00Z", 100, 10),
                claude_assistant_line("2026-03-10T10:00:00Z", 200, 20),
            ),
        )
        .unwrap();

        let all = collect_all_in(home, true, None, 0).await;
        assert_eq!(all.len(), 1, "only claude has data: {:?}", all.keys());
        let claude = all.get("claude").expect("claude present");
        assert_eq!(claude.input_tokens, 300);
        assert_eq!(claude.output_tokens, 30);
        assert_eq!(claude.sessions, 1);
        assert_eq!(claude.models.len(), 1);

        // A cutoff between the two lines keeps only the later one.
        let cutoff = chrono::DateTime::parse_from_rfc3339("2026-02-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let all = collect_all_in(home, true, Some(cutoff), 2).await;
        let claude = all.get("claude").expect("claude present after cutoff");
        assert_eq!(claude.input_tokens, 200);
        assert_eq!(claude.output_tokens, 20);
    }

    #[tokio::test]
    async fn collect_all_returns_empty_for_a_home_with_no_tool_data() {
        let tmp = tempfile::TempDir::new().unwrap();
        let all = collect_all_in(tmp.path(), true, None, 0).await;
        assert!(all.is_empty(), "unexpected tools: {:?}", all.keys());
    }

    #[tokio::test]
    async fn claude_stats_cache_short_circuits_and_merges_jsonl_deltas() {
        let tmp = tempfile::TempDir::new().unwrap();
        let home = tmp.path();
        std::fs::create_dir_all(home.join(".claude")).unwrap();
        std::fs::write(
            home.join(".claude").join("stats-cache.json"),
            r#"{"totalSessions":2,"lastComputedDate":"2026-01-31","modelUsage":{"claude-opus-4-8":{"inputTokens":1000,"outputTokens":100}}}"#,
        )
        .unwrap();
        let proj = home.join(".claude").join("projects").join("p1");
        std::fs::create_dir_all(&proj).unwrap();
        // One line already covered by the cache's computed window, one after
        // it — only the latter may fold into the delta merge.
        std::fs::write(
            proj.join("s.jsonl"),
            format!(
                "{}\n{}\n",
                claude_assistant_line("2026-01-15T10:00:00Z", 400, 40),
                claude_assistant_line("2026-02-02T10:00:00Z", 50, 5),
            ),
        )
        .unwrap();

        let stats = collect_with_step(home, "claude", true, None, None)
            .await
            .expect("collect claude")
            .expect("stats present");
        assert_eq!(stats.input_tokens, 1050, "cache total + post-window delta");
        assert_eq!(stats.output_tokens, 105);
        // The delta file carries a session on top of the cache's two.
        assert_eq!(stats.sessions, 3);
    }

    #[test]
    fn is_native_tool_matches_the_scanned_set() {
        for t in ["claude", "codex", "gemini", "opencode", "pi"] {
            assert!(is_native_tool(t), "{t} should be native");
        }
        for t in ["amp", "grok", "omp", "copilot", "cursor", "chat", ""] {
            assert!(!is_native_tool(t), "{t} should not be native");
        }
    }

    fn parse_claude_line(line: &str) -> (u64, u64, u64, u64, Option<String>) {
        let v: Value = serde_json::from_str(line).unwrap();
        if v.get("type").and_then(|t| t.as_str()) != Some("assistant") {
            return (0, 0, 0, 0, None);
        }
        let usage = match v.get("message").and_then(|m| m.get("usage")) {
            Some(u) => u,
            None => return (0, 0, 0, 0, None),
        };
        let input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cr = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cw = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let model = v
            .get("message")
            .and_then(|m| m.get("model"))
            .and_then(|m| m.as_str())
            .map(String::from);
        (input, output, cr, cw, model)
    }

    #[test]
    fn claude_line_with_usage() {
        let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-20250514","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":10}},"sessionId":"abc"}"#;
        let (i, o, cr, cw, model) = parse_claude_line(line);
        assert_eq!(i, 100);
        assert_eq!(o, 50);
        assert_eq!(cr, 20);
        assert_eq!(cw, 10);
        assert_eq!(model.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn claude_line_without_usage() {
        let line = r#"{"type":"assistant","message":{"model":"claude-sonnet-4-20250514"},"sessionId":"abc"}"#;
        let (i, o, cr, cw, _) = parse_claude_line(line);
        assert_eq!((i, o, cr, cw), (0, 0, 0, 0));
    }

    #[test]
    fn claude_line_non_assistant() {
        let line = r#"{"type":"progress","data":{"type":"hook_progress"}}"#;
        let (i, o, cr, cw, model) = parse_claude_line(line);
        assert_eq!((i, o, cr, cw), (0, 0, 0, 0));
        assert!(model.is_none());
    }

    async fn write_jsonl(dir: &tempfile::TempDir, name: &str, lines: &[&str]) -> PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, lines.join("\n")).await.unwrap();
        path
    }

    #[tokio::test]
    async fn merge_claude_jsonl_deltas_persists_parses_and_invalidates() {
        let dir = tempfile::tempdir().unwrap();
        let line = r#"{"type":"assistant","isSidechain":false,"timestamp":"2026-01-05T00:00:00Z","message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":100,"output_tokens":40}},"sessionId":"abc"}"#;
        write_jsonl(&dir, "delta.jsonl", &[line]).await;
        let cache_dir = tempfile::tempdir().unwrap();
        let cache_path = cache_dir.path().join("delta-cache.json");

        let merge = |cutoff: &'static str, refresh: bool| {
            let dir = dir.path().to_path_buf();
            let cache_path = cache_path.clone();
            async move {
                let mut stats = GlobalToolStats::default();
                merge_claude_jsonl_deltas(&dir, cutoff, &mut stats, &cache_path, refresh, None)
                    .await;
                stats
            }
        };

        let first = merge("2026-01-01", false).await;
        assert_eq!(first.input_tokens, 100);
        assert_eq!(first.sessions, 1);
        assert!(cache_path.exists(), "first merge must persist the cache");

        // Tamper with the cached entry; the sentinel showing through proves a cache hit.
        let mut v: Value =
            serde_json::from_str(&std::fs::read_to_string(&cache_path).unwrap()).unwrap();
        let files = v.get_mut("files").unwrap().as_object_mut().unwrap();
        files.values_mut().next().unwrap()["input_tokens"] = 999.into();
        std::fs::write(&cache_path, v.to_string()).unwrap();

        let second = merge("2026-01-01", false).await;
        assert_eq!(
            second.input_tokens, 999,
            "unchanged file must be read from the delta cache"
        );

        let third = merge("2026-01-02", false).await;
        assert_eq!(third.input_tokens, 100, "date change must force a re-parse");

        let mut v: Value =
            serde_json::from_str(&std::fs::read_to_string(&cache_path).unwrap()).unwrap();
        let files = v.get_mut("files").unwrap().as_object_mut().unwrap();
        files.values_mut().next().unwrap()["input_tokens"] = 555.into();
        std::fs::write(&cache_path, v.to_string()).unwrap();
        let fourth = merge("2026-01-02", true).await;
        assert_eq!(fourth.input_tokens, 100, "refresh must bypass the cache");
    }

    #[tokio::test]
    async fn parse_claude_file_counts_main_session() {
        let dir = tempfile::tempdir().unwrap();
        let line = r#"{"type":"assistant","isSidechain":false,"message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":10,"output_tokens":5}},"sessionId":"abc"}"#;
        let path = write_jsonl(&dir, "main.jsonl", &[line]).await;
        let entry = parse_claude_file_with_cutoff(&path, None).await.unwrap();
        assert!(
            entry.has_session,
            "main assistant line should register a session"
        );
        assert_eq!(entry.input_tokens, 10);
        assert_eq!(entry.output_tokens, 5);
    }

    #[tokio::test]
    async fn parse_claude_file_ignores_sidechain_only_file() {
        let dir = tempfile::tempdir().unwrap();
        let line = r#"{"type":"assistant","isSidechain":true,"message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":10,"output_tokens":5}},"sessionId":"abc"}"#;
        let path = write_jsonl(&dir, "agent-xxx.jsonl", &[line]).await;
        let entry = parse_claude_file_with_cutoff(&path, None).await.unwrap();
        assert!(
            !entry.has_session,
            "a file containing only sidechain assistant lines must not count as a session"
        );
        // Tokens are still real API spend and should be preserved.
        assert_eq!(entry.input_tokens, 10);
        assert_eq!(entry.output_tokens, 5);
    }

    #[tokio::test]
    async fn parse_gemini_file_subtracts_cached_from_input() {
        // Gemini's `tokens.input` is the total turn input INCLUDING the
        // `tokens.cached` portion. Normalize to fresh-only.
        let dir = tempfile::tempdir().unwrap();
        let lines = [
            r#"{"sessionId":"s1","kind":"main"}"#,
            r#"{"type":"user","content":[{"text":"hi"}]}"#,
            r#"{"$set":{"lastUpdated":"2026-06-19T13:37:21.014Z"}}"#,
            r#"{"type":"gemini","model":"gemini-2.5-flash","tokens":{"input":7613,"output":11,"cached":7036,"thoughts":29,"tool":0,"total":7653}}"#,
        ];
        let path = write_jsonl(&dir, "session-x.jsonl", &lines).await;
        let entry = parse_gemini_file(&path, None).await.unwrap();
        assert_eq!(
            entry.input_tokens,
            7613 - 7036,
            "input should exclude cached portion"
        );
        assert_eq!(entry.output_tokens, 11);
        assert_eq!(entry.cache_read_tokens, 7036);
        let m = entry.models.get("gemini-2.5-flash").unwrap();
        assert_eq!(m.input_tokens, 7613 - 7036);
        assert_eq!(m.output_tokens, 11);
        assert_eq!(m.cache_read_tokens, 7036);
    }

    #[tokio::test]
    async fn parse_gemini_file_accepts_legacy_json_object() {
        // Pre-migration layout: a single `{messages:[…]}` object. Must still
        // parse so a user's old `session-*.json` history isn't dropped.
        let dir = tempfile::tempdir().unwrap();
        let body = r#"{"sessionId":"s1","messages":[
            {"type":"user","content":"hi"},
            {"type":"gemini","model":"gemini-2.5-flash","tokens":{"input":7613,"output":11,"cached":7036}}
        ]}"#;
        let path = dir.path().join("session-legacy.json");
        fs::write(&path, body).await.unwrap();
        let entry = parse_gemini_file(&path, None).await.unwrap();
        assert_eq!(entry.input_tokens, 7613 - 7036);
        assert_eq!(entry.output_tokens, 11);
        assert_eq!(entry.cache_read_tokens, 7036);
    }

    #[tokio::test]
    async fn parse_gemini_file_sums_multiple_messages_with_cache() {
        let dir = tempfile::tempdir().unwrap();
        let lines = [
            r#"{"sessionId":"s1","kind":"main"}"#,
            r#"{"type":"gemini","model":"gemini-2.5-flash","tokens":{"input":1000,"output":50,"cached":800}}"#,
            r#"{"type":"gemini","model":"gemini-2.5-flash","tokens":{"input":2000,"output":100,"cached":1500}}"#,
        ];
        let path = write_jsonl(&dir, "session-y.jsonl", &lines).await;
        let entry = parse_gemini_file(&path, None).await.unwrap();
        assert_eq!(entry.input_tokens, (1000 - 800) + (2000 - 1500));
        assert_eq!(entry.output_tokens, 150);
        assert_eq!(entry.cache_read_tokens, 800 + 1500);
    }

    #[tokio::test]
    async fn parse_gemini_file_filters_messages_by_timestamp_cutoff() {
        let dir = tempfile::tempdir().unwrap();
        let lines = [
            r#"{"sessionId":"s1","kind":"main"}"#,
            r#"{"type":"gemini","timestamp":"2025-05-31T23:59:59Z","model":"gemini-2.5-flash","tokens":{"input":1000,"output":50,"cached":800}}"#,
            r#"{"type":"gemini","timestamp":"2025-06-01T00:00:00Z","model":"gemini-2.5-flash","tokens":{"input":2000,"output":100,"cached":1500}}"#,
        ];
        let path = write_jsonl(&dir, "session-z.jsonl", &lines).await;
        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let entry = parse_gemini_file(&path, Some(cutoff)).await.unwrap();
        assert_eq!(entry.input_tokens, 2000 - 1500);
        assert_eq!(entry.output_tokens, 100);
        assert_eq!(entry.cache_read_tokens, 1500);
    }

    #[tokio::test]
    async fn parse_pi_file_keeps_input_fresh_only() {
        // Pi's `usage.input` already excludes `cacheRead`. The previous
        // implementation added `cacheRead` to it, double-counting cache.
        let dir = tempfile::tempdir().unwrap();
        let session_record = r#"{"type":"session","id":"sess-abc"}"#;
        let message_record = r#"{"type":"message","message":{"model":"pi-coder","usage":{"input":38,"output":23,"cacheRead":5376,"cacheWrite":0,"totalTokens":5437}}}"#;
        let path = write_jsonl(&dir, "sess.jsonl", &[session_record, message_record]).await;
        let (entry, ids) = parse_pi_file(&path, None).await.unwrap();
        assert_eq!(entry.input_tokens, 38, "input must be the fresh-only value");
        assert_eq!(entry.output_tokens, 23);
        assert_eq!(entry.cache_read_tokens, 5376);
        assert_eq!(entry.cache_write_tokens, 0);
        let m = entry.models.get("pi-coder").unwrap();
        assert_eq!(m.input_tokens, 38);
        assert_eq!(m.output_tokens, 23);
        assert_eq!(m.cache_read_tokens, 5376);
        assert_eq!(ids, vec!["sess-abc".to_string()]);
    }

    #[tokio::test]
    async fn parse_pi_file_filters_messages_by_timestamp_cutoff() {
        let dir = tempfile::tempdir().unwrap();
        let session_record = r#"{"type":"session","id":"sess-abc"}"#;
        let old_message = r#"{"type":"message","timestamp":"2025-05-31T23:59:59Z","message":{"model":"pi-coder","usage":{"input":38,"output":23,"cacheRead":5376,"cacheWrite":0}}}"#;
        let new_message = r#"{"type":"message","timestamp":"2025-06-01T00:00:00Z","message":{"model":"pi-coder","usage":{"input":7,"output":11,"cacheRead":13,"cacheWrite":0}}}"#;
        let path = write_jsonl(
            &dir,
            "sess-cutoff.jsonl",
            &[session_record, old_message, new_message],
        )
        .await;
        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let (entry, ids) = parse_pi_file(&path, Some(cutoff)).await.unwrap();
        assert_eq!(entry.input_tokens, 7);
        assert_eq!(entry.output_tokens, 11);
        assert_eq!(entry.cache_read_tokens, 13);
        assert_eq!(ids, vec!["sess-abc".to_string()]);
    }

    #[tokio::test]
    async fn parse_codex_file_subtracts_cached_from_input() {
        // Codex's `input_tokens` in `total_token_usage` is the total input
        // for the session INCLUDING cached tokens. aivo's display treats
        // `input_tokens` as "fresh only" (Claude's convention), so the
        // parser must subtract `cached_input_tokens` before storing.
        let dir = tempfile::tempdir().unwrap();
        let turn_context = r#"{"type":"turn_context","payload":{"model":"gpt-5.4"}}"#;
        let token_event = r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":13482,"cached_input_tokens":3968,"output_tokens":202,"total_tokens":13684},"model_context_window":258400}}}"#;
        let path = write_jsonl(&dir, "rollout.jsonl", &[turn_context, token_event]).await;
        let entry = parse_codex_file(&path, None).await.unwrap();
        assert!(entry.has_session);
        assert_eq!(
            entry.input_tokens,
            13482 - 3968,
            "input should exclude cached portion"
        );
        assert_eq!(entry.output_tokens, 202);
        assert_eq!(entry.cache_read_tokens, 3968);
        // Per-model entry should store fresh-only input + cached separately.
        let m = entry.models.get("gpt-5.4").unwrap();
        assert_eq!(m.input_tokens, 13482 - 3968);
        assert_eq!(m.output_tokens, 202);
        assert_eq!(m.cache_read_tokens, 3968);
    }

    #[tokio::test]
    async fn parse_codex_file_handles_cached_equal_to_input() {
        // Degenerate case: cached >= input. Use saturating_sub to avoid underflow.
        let dir = tempfile::tempdir().unwrap();
        let token_event = r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":500,"cached_input_tokens":500,"output_tokens":10,"total_tokens":510}}}}"#;
        let path = write_jsonl(&dir, "rollout.jsonl", &[token_event]).await;
        let entry = parse_codex_file(&path, None).await.unwrap();
        assert_eq!(entry.input_tokens, 0);
        assert_eq!(entry.output_tokens, 10);
        assert_eq!(entry.cache_read_tokens, 500);
    }

    #[tokio::test]
    async fn parse_codex_file_filters_by_timestamp_and_uses_deltas() {
        let dir = tempfile::tempdir().unwrap();
        let turn_context = r#"{"type":"turn_context","payload":{"model":"gpt-5.4"}}"#;
        let before = r#"{"type":"event_msg","timestamp":"2025-05-31T23:59:59Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":30}}}}"#;
        let boundary = r#"{"type":"event_msg","timestamp":"2025-06-01T00:00:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":150,"cached_input_tokens":30,"output_tokens":45}}}}"#;
        let after = r#"{"type":"event_msg","timestamp":"2025-06-01T00:05:00Z","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":210,"cached_input_tokens":50,"output_tokens":80}}}}"#;
        let path = write_jsonl(
            &dir,
            "rollout-cutoff.jsonl",
            &[turn_context, before, boundary, after],
        )
        .await;
        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let entry = parse_codex_file(&path, Some(cutoff)).await.unwrap();
        assert_eq!(
            entry.input_tokens,
            (150 - 100) - (30 - 20) + (210 - 150) - (50 - 30)
        );
        assert_eq!(entry.output_tokens, (45 - 30) + (80 - 45));
        assert_eq!(entry.cache_read_tokens, (30 - 20) + (50 - 30));
    }

    #[tokio::test]
    async fn parse_claude_file_mixed_main_and_sidechain_counts_once() {
        let dir = tempfile::tempdir().unwrap();
        let sidechain = r#"{"type":"assistant","isSidechain":true,"message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":1,"output_tokens":2}},"sessionId":"abc"}"#;
        let main = r#"{"type":"assistant","isSidechain":false,"message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":3,"output_tokens":4}},"sessionId":"abc"}"#;
        let path = write_jsonl(&dir, "mixed.jsonl", &[sidechain, main]).await;
        let entry = parse_claude_file_with_cutoff(&path, None).await.unwrap();
        assert!(entry.has_session);
        assert_eq!(entry.input_tokens, 4);
        assert_eq!(entry.output_tokens, 6);
    }

    #[test]
    fn gemini_message_parsing() {
        let json = r#"{"sessionId":"s1","messages":[
            {"type":"user","content":"hi"},
            {"type":"gemini","content":"hello","tokens":{"input":100,"output":50,"cached":20,"thoughts":10,"tool":0}},
            {"type":"gemini","content":"bye","tokens":{"input":200,"output":100,"cached":0,"thoughts":5,"tool":0}}
        ]}"#;
        let v: Value = serde_json::from_str(json).unwrap();
        let messages = v.get("messages").unwrap().as_array().unwrap();
        let mut total_input = 0u64;
        let mut total_output = 0u64;
        let mut total_cached = 0u64;
        for msg in messages {
            if msg.get("type").and_then(|t| t.as_str()) != Some("gemini") {
                continue;
            }
            if let Some(tokens) = msg.get("tokens") {
                total_input += tokens.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
                total_output += tokens.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
                total_cached += tokens.get("cached").and_then(|v| v.as_u64()).unwrap_or(0);
            }
        }
        assert_eq!(total_input, 300);
        assert_eq!(total_output, 150);
        assert_eq!(total_cached, 20);
    }

    #[test]
    fn pi_message_parsing() {
        let line = r#"{"type":"message","id":"x","message":{"role":"assistant","model":"gpt-5.2","usage":{"input":500,"output":200,"cacheRead":100,"cacheWrite":50,"totalTokens":700}}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let usage = v.get("message").unwrap().get("usage").unwrap();
        assert_eq!(usage.get("input").unwrap().as_u64(), Some(500));
        assert_eq!(usage.get("output").unwrap().as_u64(), Some(200));
        assert_eq!(usage.get("cacheRead").unwrap().as_u64(), Some(100));
        assert_eq!(usage.get("cacheWrite").unwrap().as_u64(), Some(50));
    }

    #[test]
    fn codex_token_count_parsing() {
        let line = r#"{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"input_tokens":1000,"cached_input_tokens":500,"output_tokens":300,"reasoning_output_tokens":100,"total_tokens":1300},"model_context_window":258400},"rate_limits":null}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let usage = v
            .get("payload")
            .unwrap()
            .get("info")
            .unwrap()
            .get("total_token_usage")
            .unwrap();
        assert_eq!(usage.get("input_tokens").unwrap().as_u64(), Some(1000));
        assert_eq!(usage.get("output_tokens").unwrap().as_u64(), Some(300));
        assert_eq!(
            usage.get("cached_input_tokens").unwrap().as_u64(),
            Some(500)
        );
    }

    #[test]
    fn codex_null_info_skipped() {
        let line = r#"{"type":"event_msg","payload":{"type":"token_count","info":null,"rate_limits":null}}"#;
        let v: Value = serde_json::from_str(line).unwrap();
        let info = v.get("payload").unwrap().get("info").unwrap();
        assert!(info.is_null());
    }

    #[test]
    fn normalize_model_strips_prefix_and_version() {
        assert_eq!(
            normalize_model_for_display("anthropic/claude-sonnet-4.6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_display("claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_model_for_display("anthropic/claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(normalize_model_for_display("openai/gpt-5.4"), "gpt-5.4");
        assert_eq!(
            normalize_model_for_display("accounts/fireworks/models/kimi-k2-instruct-0905"),
            "kimi-k2-instruct-0905"
        );
        assert_eq!(normalize_model_for_display("MiniMax-M2.5"), "minimax-m2.5");
        assert_eq!(
            normalize_model_for_display("minimax/minimax-m2.5"),
            "minimax-m2.5"
        );
        assert_eq!(
            normalize_model_for_display("deepseek-chat"),
            "deepseek-chat"
        );
        assert_eq!(
            normalize_model_for_display("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn tool_display_names() {
        assert_eq!(tool_display_name("claude"), "Claude Code");
        assert_eq!(tool_display_name("codex"), "Codex");
        assert_eq!(tool_display_name("gemini"), "Gemini");
        assert_eq!(tool_display_name("pi"), "Pi");
        assert_eq!(tool_display_name("code"), "Code");
        // Pre-rename bucket still renders as the built-in agent.
        assert_eq!(tool_display_name("chat"), "Code");
        assert_eq!(tool_display_name("unknown"), "unknown");
    }

    #[test]
    fn parse_claude_stats_cache_aggregates_per_model() {
        let v: Value = serde_json::from_str(
            r#"{
                "version": 3,
                "totalSessions": 3812,
                "modelUsage": {
                    "claude-opus-4-6": {
                        "inputTokens": 11157776,
                        "outputTokens": 17954156,
                        "cacheReadInputTokens": 5692730263,
                        "cacheCreationInputTokens": 312040660
                    },
                    "claude-sonnet-4-6": {
                        "inputTokens": 39314566,
                        "outputTokens": 2838025,
                        "cacheReadInputTokens": 639032928,
                        "cacheCreationInputTokens": 40982361
                    }
                }
            }"#,
        )
        .unwrap();

        let stats = parse_claude_stats_cache(&v).expect("cache should parse");
        assert_eq!(stats.sessions, 3812);
        assert_eq!(stats.input_tokens, 11157776 + 39314566);
        assert_eq!(stats.output_tokens, 17954156 + 2838025);
        assert_eq!(stats.cache_read_tokens, 5692730263 + 639032928);
        assert_eq!(stats.cache_write_tokens, 312040660 + 40982361);
        // total_tokens matches Claude's /stats UI convention (input + output).
        assert_eq!(
            stats.total_tokens(),
            (11157776 + 39314566) + (17954156 + 2838025)
        );
        // Per-model entries go through normalize_model_for_display — version
        // separators collapse (4-6 → 4.6).
        let opus = stats.models.get("claude-opus-4.6").expect("opus present");
        assert_eq!(opus.input_tokens, 11157776);
        assert_eq!(opus.output_tokens, 17954156);
    }

    #[test]
    fn parse_claude_stats_cache_rejects_missing_model_usage() {
        let v: Value = serde_json::from_str(r#"{"version": 3}"#).unwrap();
        assert!(parse_claude_stats_cache(&v).is_none());
    }

    #[test]
    fn parse_claude_stats_cache_rejects_empty_payload() {
        // modelUsage present but empty, no sessions, no tokens → None so the
        // caller can fall back to walking JSONL files.
        let v: Value = serde_json::from_str(r#"{"totalSessions": 0, "modelUsage": {}}"#).unwrap();
        assert!(parse_claude_stats_cache(&v).is_none());
    }

    #[test]
    fn day_after_start_utc_advances_by_one_day() {
        let t0 = day_after_start_utc("2026-04-17").expect("valid date");
        let t1 = day_after_start_utc("2026-04-18").expect("valid date");
        // Consecutive dates should differ by exactly 86,400 seconds.
        assert_eq!(t1.duration_since(t0).unwrap().as_secs(), 86_400);
    }

    #[test]
    fn day_after_start_utc_rejects_bad_input() {
        assert!(day_after_start_utc("not-a-date").is_none());
        assert!(day_after_start_utc("").is_none());
    }

    #[test]
    fn aggregate_cache_cutoff_includes_boundary_excludes_older() {
        use std::time::{Duration, SystemTime};
        let now = SystemTime::now();
        let old = now - Duration::from_secs(30 * 24 * 3600);
        let edge_mtime = now - Duration::from_secs(7 * 24 * 3600);

        let mut cache = StatsCache::default();
        cache.files.insert(
            "/old.jsonl".into(),
            FileEntry {
                size: 1,
                input_tokens: 100,
                has_session: true,
                ..Default::default()
            },
        );
        cache.files.insert(
            "/new.jsonl".into(),
            FileEntry {
                size: 1,
                input_tokens: 50,
                has_session: true,
                ..Default::default()
            },
        );
        cache.files.insert(
            "/edge.jsonl".into(),
            FileEntry {
                size: 1,
                input_tokens: 7,
                has_session: true,
                ..Default::default()
            },
        );

        let mut mtimes = std::collections::HashMap::new();
        mtimes.insert("/old.jsonl".to_string(), old);
        mtimes.insert("/new.jsonl".to_string(), now);
        mtimes.insert("/edge.jsonl".to_string(), edge_mtime);

        // Cutoff is exactly the edge file's mtime — boundary must be included
        // (spec: skip files whose mtime is `< cutoff`).
        let cutoff = chrono::DateTime::<chrono::Utc>::from(edge_mtime);
        let stats = aggregate_cache_filtered(&cache, &mtimes, Some(cutoff));
        assert_eq!(
            stats.input_tokens,
            50 + 7,
            "boundary file must be included; older file excluded"
        );
        assert_eq!(stats.sessions, 2);
    }

    #[tokio::test]
    async fn parse_claude_file_filters_by_datetime_cutoff() {
        use tokio::io::AsyncWriteExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut f = tokio::fs::File::create(&path).await.unwrap();
        let stale = r#"{"type":"assistant","timestamp":"2024-01-01T00:00:00Z","message":{"usage":{"input_tokens":1000,"output_tokens":1000}}}"#;
        let fresh = r#"{"type":"assistant","timestamp":"2099-01-01T00:00:00Z","message":{"usage":{"input_tokens":7,"output_tokens":11}}}"#;
        f.write_all(format!("{stale}\n{fresh}\n").as_bytes())
            .await
            .unwrap();
        f.flush().await.unwrap();

        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let entry = parse_claude_file_with_cutoff(&path, Some(cutoff))
            .await
            .unwrap();
        assert_eq!(entry.input_tokens, 7);
        assert_eq!(entry.output_tokens, 11);
    }

    #[tokio::test]
    async fn parse_claude_file_with_cutoff_handles_boundary_cases() {
        use tokio::io::AsyncWriteExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let mut f = tokio::fs::File::create(&path).await.unwrap();

        // Line 1: missing timestamp field -- skipped when cutoff is set.
        let no_ts =
            r#"{"type":"assistant","message":{"usage":{"input_tokens":100,"output_tokens":100}}}"#;
        // Line 2: malformed timestamp -- skipped when cutoff is set.
        let bad_ts = r#"{"type":"assistant","timestamp":"not-a-date","message":{"usage":{"input_tokens":200,"output_tokens":200}}}"#;
        // Line 3: timestamp == cutoff exactly -- INCLUDED (half-open [cutoff, ∞)).
        let on_boundary = r#"{"type":"assistant","timestamp":"2025-06-01T00:00:00Z","message":{"usage":{"input_tokens":3,"output_tokens":5}}}"#;
        // Line 4: pre-cutoff -- skipped.
        let before = r#"{"type":"assistant","timestamp":"2025-05-31T23:59:59Z","message":{"usage":{"input_tokens":7,"output_tokens":11}}}"#;
        // Line 5: post-cutoff -- INCLUDED.
        let after = r#"{"type":"assistant","timestamp":"2025-06-02T00:00:00Z","message":{"usage":{"input_tokens":13,"output_tokens":17}}}"#;

        f.write_all(format!("{no_ts}\n{bad_ts}\n{on_boundary}\n{before}\n{after}\n").as_bytes())
            .await
            .unwrap();
        f.flush().await.unwrap();

        let cutoff = chrono::DateTime::parse_from_rfc3339("2025-06-01T00:00:00Z")
            .unwrap()
            .with_timezone(&chrono::Utc);
        let entry = parse_claude_file_with_cutoff(&path, Some(cutoff))
            .await
            .expect("file with at least one valid usage should produce an entry");

        // Only on_boundary (3+5) and after (13+17) should contribute.
        assert_eq!(entry.input_tokens, 3 + 13);
        assert_eq!(entry.output_tokens, 5 + 17);
    }

    #[test]
    fn aggregate_cache_with_no_cutoff_includes_all() {
        let mut cache = StatsCache::default();
        cache.files.insert(
            "/a.jsonl".into(),
            FileEntry {
                size: 1,
                input_tokens: 10,
                has_session: true,
                ..Default::default()
            },
        );
        let mtimes = std::collections::HashMap::new();
        let stats = aggregate_cache_filtered(&cache, &mtimes, None);
        assert_eq!(stats.input_tokens, 10);
    }

    #[tokio::test]
    async fn collect_opencode_filters_by_cutoff_when_set() {
        use rusqlite::Connection;

        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                time_updated INTEGER NOT NULL,
                data TEXT NOT NULL
            );",
        )
        .unwrap();

        // Old row (epoch ms = 1000000000000 -> 2001-09-09)
        conn.execute(
            "INSERT INTO message VALUES (?1, ?2, ?3, ?3, ?4)",
            rusqlite::params![
                "msg-old",
                "sess-1",
                1_000_000_000_000_i64,
                r#"{"role":"assistant","modelID":"gpt-4","tokens":{"input":100,"output":100,"cache":{"read":0,"write":0}}}"#,
            ],
        )
        .unwrap();
        // New row (epoch ms ~ now)
        let now_ms = chrono::Utc::now().timestamp_millis();
        conn.execute(
            "INSERT INTO message VALUES (?1, ?2, ?3, ?3, ?4)",
            rusqlite::params![
                "msg-new",
                "sess-2",
                now_ms,
                r#"{"role":"assistant","modelID":"gpt-4","tokens":{"input":7,"output":11,"cache":{"read":0,"write":0}}}"#,
            ],
        )
        .unwrap();

        let cutoff = chrono::Utc::now() - chrono::Duration::days(7);
        let stats = aggregate_opencode_messages(&conn, Some(cutoff)).expect("query OK");
        assert_eq!(stats.input_tokens, 7);
        assert_eq!(stats.output_tokens, 11);
        assert_eq!(stats.sessions, 1);

        let stats_no_cutoff = aggregate_opencode_messages(&conn, None).expect("query OK");
        assert_eq!(stats_no_cutoff.input_tokens, 107);
        assert_eq!(stats_no_cutoff.sessions, 2);
    }
}
