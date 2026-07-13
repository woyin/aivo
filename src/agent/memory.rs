//! Cross-session project memory: `remember` appends durable facts to a
//! per-project file under the config dir (never in the repo), injected into
//! future sessions via the guide inlining in `system_prompt`.

use crate::agent::protocol::ToolSpec;
use serde_json::json;
use std::path::{Path, PathBuf};

const MAX_ENTRIES: usize = 100;
/// Below the 24 KiB guide inline cap, so memory always arrives verbatim.
const MAX_FILE_BYTES: usize = 16 * 1024;
const MAX_ENTRY_CHARS: usize = 600;

const HEADER: &str = "# aivo memory\n\
Durable facts and decisions the agent saved with the `remember` tool.\n\
One `- ` bullet per memory; safe to edit or delete lines by hand.\n";

/// Per-project key from the repo root (main checkout for linked worktrees),
/// so all worktrees of one repo share the same memory files.
fn project_key(cwd: &Path) -> String {
    let root = project_root(cwd);
    let sanitized: Vec<char> = root
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    // Truncate from the front: the tail carries the project name.
    sanitized[sanitized.len().saturating_sub(120)..]
        .iter()
        .collect()
}

/// Workspace tier: this project's `remember` facts, injected into every session here.
pub fn project_memory_path(cwd: &Path) -> PathBuf {
    memory_dir().join(format!("{}.md", project_key(cwd)))
}

/// Global tier: facts injected into every project.
pub fn global_memory_path() -> PathBuf {
    memory_dir().join("GLOBAL.md")
}

/// Auto-saved dated session topics; searchable via `memory_search`, never injected.
pub fn session_log_path(cwd: &Path) -> PathBuf {
    memory_dir().join(format!("{}.sessions.md", project_key(cwd)))
}

fn memory_dir() -> PathBuf {
    crate::services::paths::config_dir().join("memory")
}

/// Which tier a `remember` fact lands in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum MemoryScope {
    #[default]
    Workspace,
    Global,
}

impl MemoryScope {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "workspace" | "project" | "local" => Some(Self::Workspace),
            "global" | "user" => Some(Self::Global),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Workspace => "workspace",
            Self::Global => "global",
        }
    }
}

pub fn path_for_scope(cwd: &Path, scope: MemoryScope) -> PathBuf {
    match scope {
        MemoryScope::Workspace => project_memory_path(cwd),
        MemoryScope::Global => global_memory_path(),
    }
}

/// Walk up to the repo root; a `.git` pointer file (linked worktree) resolves
/// to the main checkout's root.
fn project_root(cwd: &Path) -> PathBuf {
    for dir in cwd.ancestors() {
        let dot_git = dir.join(".git");
        if dot_git.is_dir() {
            return dir.to_path_buf();
        }
        if dot_git.is_file() {
            if let Some(main_root) = main_root_from_gitfile(&dot_git) {
                return main_root;
            }
            return dir.to_path_buf();
        }
    }
    cwd.to_path_buf()
}

fn main_root_from_gitfile(dot_git: &Path) -> Option<PathBuf> {
    let content = std::fs::read_to_string(dot_git).ok()?;
    let gitdir = content
        .lines()
        .find_map(|l| l.strip_prefix("gitdir:"))?
        .trim();
    let gitdir = if Path::new(gitdir).is_absolute() {
        PathBuf::from(gitdir)
    } else {
        dot_git.parent()?.join(gitdir)
    };
    // `<main>/.git/worktrees/<name>` → `<main>`.
    let mut cur = gitdir.as_path();
    while let Some(parent) = cur.parent() {
        if cur.file_name().is_some_and(|n| n == "worktrees")
            && parent.file_name().is_some_and(|n| n == ".git")
        {
            return parent.parent().map(Path::to_path_buf);
        }
        cur = parent;
    }
    None
}

/// Stored entries (the `- ` bullets), oldest first.
pub fn load_entries(path: &Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    content
        .lines()
        .filter_map(|l| l.strip_prefix("- "))
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
        .collect()
}

pub enum RememberOutcome {
    Added(usize),
    Refreshed,
}

/// Append one fact; an exact duplicate refreshes recency instead of stacking.
pub fn remember(path: &Path, text: &str) -> Result<RememberOutcome, String> {
    let fact = normalize(text)?;
    append_fact(path, HEADER, fact, whole_entry)
}

fn whole_entry(e: &str) -> &str {
    e
}

/// The topic part of a `<date>: <topic>` session entry, so the same topic
/// refreshes across dates instead of stacking one line per day.
fn session_topic(e: &str) -> &str {
    e.split_once(": ").map_or(e, |(_, t)| t)
}

/// Append an entry to a capped, atomically-written bullet file; a duplicate
/// under `key` refreshes recency, oldest entries drop past the caps.
fn append_fact(
    path: &Path,
    header: &str,
    fact: String,
    key: fn(&str) -> &str,
) -> Result<RememberOutcome, String> {
    let mut entries = load_entries(path);
    let refreshed = if let Some(pos) = entries.iter().position(|e| key(e) == key(&fact)) {
        entries.remove(pos);
        true
    } else {
        false
    };
    entries.push(fact);
    while entries.len() > MAX_ENTRIES
        || entries.iter().map(|e| e.len() + 3).sum::<usize>() + header.len() > MAX_FILE_BYTES
    {
        entries.remove(0);
    }
    if let Some(dir) = path.parent() {
        crate::services::atomic_write::ensure_private_dir_blocking(dir)
            .map_err(|e| format!("create memory dir: {e}"))?;
    }
    let mut out = String::with_capacity(header.len() + 64 * entries.len());
    out.push_str(header);
    out.push('\n');
    for e in &entries {
        out.push_str("- ");
        out.push_str(e);
        out.push('\n');
    }
    crate::services::atomic_write::atomic_write_secure_blocking(path, out.as_bytes())
        .map_err(|e| format!("write memory file: {e}"))?;
    Ok(if refreshed {
        RememberOutcome::Refreshed
    } else {
        RememberOutcome::Added(entries.len())
    })
}

const SESSION_HEADER: &str = "# aivo session log\n\
Auto-saved one-line summaries of past sessions in this project (topic = the \
opening request). Searchable via `memory_search`; not injected verbatim.\n";

/// Best-effort auto-save of a dated session topic. `topic` is the opening
/// request — user text only, so shell commands (which can embed secrets)
/// are never persisted here.
pub fn record_session_summary(cwd: &Path, topic: &str, date: &str) {
    let topic = topic.split_whitespace().collect::<Vec<_>>().join(" ");
    if topic.is_empty() {
        return;
    }
    let topic: String = topic
        .chars()
        .take(MAX_ENTRY_CHARS.saturating_sub(date.len() + 2))
        .collect();
    let _ = append_fact(
        &session_log_path(cwd),
        SESSION_HEADER,
        format!("{date}: {topic}"),
        session_topic,
    );
}

pub struct MemoryHit {
    pub source: &'static str,
    pub text: String,
    score: usize,
}

/// Rank all tiers by term overlap with `query`, best-first, up to `limit` hits.
pub fn search(cwd: &Path, query: &str, limit: usize) -> Vec<MemoryHit> {
    search_in(
        &[
            ("workspace", project_memory_path(cwd)),
            ("global", global_memory_path()),
            ("session", session_log_path(cwd)),
        ],
        query,
        limit,
    )
}

/// Path-injectable core of [`search`], so tests don't need the real config dir.
fn search_in(sources: &[(&'static str, PathBuf)], query: &str, limit: usize) -> Vec<MemoryHit> {
    let terms = tokenize(query);
    if terms.is_empty() || limit == 0 {
        return Vec::new();
    }
    let needle = query.trim().to_lowercase();
    let mut hits: Vec<MemoryHit> = Vec::new();
    for (source, path) in sources {
        for text in load_entries(path) {
            let hay = text.to_lowercase();
            let mut score = terms.iter().filter(|t| hay.contains(*t)).count();
            if score == 0 {
                continue;
            }
            if needle.len() > 2 && hay.contains(&needle) {
                score += 2; // phrase match beats scattered terms
            }
            hits.push(MemoryHit {
                source,
                text,
                score,
            });
        }
    }
    // Stable sort keeps tier order (workspace > global > session) on ties.
    hits.sort_by_key(|h| std::cmp::Reverse(h.score));
    hits.truncate(limit);
    hits
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(str::to_string)
        .collect()
}

/// One line, bounded, non-empty.
fn normalize(text: &str) -> Result<String, String> {
    let fact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if fact.is_empty() {
        return Err("remember: `fact` is empty.".to_string());
    }
    if fact.chars().count() > MAX_ENTRY_CHARS {
        return Err(format!(
            "remember: keep one fact under {MAX_ENTRY_CHARS} chars (got {}). Split it, or drop detail.",
            fact.chars().count()
        ));
    }
    Ok(fact)
}

/// The `remember` function schema; engine-handled like `take_note`.
pub fn memory_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "remember".to_string(),
        description: "Save one durable fact to persistent memory — it is injected into EVERY \
future session, unlike `take_note` (which lasts only for the current session). Use it sparingly, \
for things worth knowing weeks from now: a decision and its why, a user preference or correction, \
a non-obvious constraint or gotcha. Don't save session progress, anything derivable from the \
code, or secrets. One concise fact per call. Default scope is this project; use scope `global` \
for a fact that applies across all projects (e.g. a personal preference)."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to persist, as one self-contained sentence (who/what/why included)."
                },
                "scope": {
                    "type": "string",
                    "enum": ["workspace", "global"],
                    "description": "`workspace` (default) = this project only; `global` = all projects."
                }
            },
            "required": ["fact"]
        }),
    }
}

/// The `memory_search` function schema; engine-handled.
pub fn memory_search_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "memory_search".to_string(),
        description: "Search your cross-session memory (this project's saved facts, global facts, \
and past-session summaries) for anything relevant to a topic. Use it when a task might relate to \
earlier decisions, conventions, or debugging you may have recorded before — the first-turn \
injection only surfaces the most recent facts, so search when you need more."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for (keywords or a short phrase)."
                }
            },
            "required": ["query"]
        }),
    }
}

pub fn parse_remember(args: &serde_json::Value) -> Result<(String, MemoryScope), String> {
    let fact = args
        .get("fact")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| "remember: missing `fact`.".to_string())?;
    let scope = match args.get("scope").and_then(|v| v.as_str()) {
        Some(s) => MemoryScope::parse(s)
            .ok_or_else(|| format!("remember: unknown scope '{s}' (use workspace or global)."))?,
        None => MemoryScope::default(),
    };
    Ok((fact, scope))
}

pub fn parse_query(args: &serde_json::Value) -> Result<String, String> {
    args.get("query")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| "memory_search: missing `query`.".to_string())
}

/// Render a memory search as the tool result the model reads.
pub fn search_result_text(cwd: &Path, query: &str) -> String {
    const LIMIT: usize = 6;
    let hits = search(cwd, query, LIMIT);
    if hits.is_empty() {
        return format!("No memory matched \"{query}\".");
    }
    let mut out = format!("{} memory match(es) for \"{query}\":\n", hits.len());
    for h in &hits {
        out.push_str(&format!("- [{}] {}\n", h.source, h.text));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "aivo_memory_test_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn remember_appends_and_loads_round_trip() {
        let dir = tmp();
        let path = dir.join("mem.md");
        assert!(matches!(
            remember(&path, "use ripgrep for search").unwrap(),
            RememberOutcome::Added(1)
        ));
        assert!(matches!(
            remember(&path, "tests need fast crypto feature").unwrap(),
            RememberOutcome::Added(2)
        ));
        assert_eq!(
            load_entries(&path),
            vec![
                "use ripgrep for search".to_string(),
                "tests need fast crypto feature".to_string()
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn duplicate_refreshes_recency_instead_of_stacking() {
        let dir = tmp();
        let path = dir.join("mem.md");
        remember(&path, "a").unwrap();
        remember(&path, "b").unwrap();
        assert!(matches!(
            remember(&path, "a").unwrap(),
            RememberOutcome::Refreshed
        ));
        assert_eq!(load_entries(&path), vec!["b".to_string(), "a".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn caps_drop_oldest_and_reject_essays() {
        let dir = tmp();
        let path = dir.join("mem.md");
        for i in 0..(MAX_ENTRIES + 5) {
            remember(&path, &format!("fact {i}")).unwrap();
        }
        let entries = load_entries(&path);
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert_eq!(entries[0], "fact 5");
        assert!(remember(&path, &"x".repeat(MAX_ENTRY_CHARS + 1)).is_err());
        assert!(remember(&path, "  \n ").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn newlines_collapse_to_one_line() {
        let dir = tmp();
        let path = dir.join("mem.md");
        remember(&path, "line one\nline  two").unwrap();
        assert_eq!(load_entries(&path), vec!["line one line two".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_remember_reads_fact_and_scope() {
        let (fact, scope) = parse_remember(&json!({"fact": "use ripgrep"})).unwrap();
        assert_eq!(fact, "use ripgrep");
        assert_eq!(scope, MemoryScope::Workspace); // default

        let (_, scope) =
            parse_remember(&json!({"fact": "2-space indent", "scope": "global"})).unwrap();
        assert_eq!(scope, MemoryScope::Global);

        assert!(parse_remember(&json!({"scope": "global"})).is_err()); // missing fact
        assert!(parse_remember(&json!({"fact": "x", "scope": "bogus"})).is_err());
    }

    #[test]
    fn scope_parse_accepts_aliases() {
        assert_eq!(
            MemoryScope::parse("workspace"),
            Some(MemoryScope::Workspace)
        );
        assert_eq!(MemoryScope::parse("project"), Some(MemoryScope::Workspace));
        assert_eq!(MemoryScope::parse("Global"), Some(MemoryScope::Global));
        assert_eq!(MemoryScope::parse("user"), Some(MemoryScope::Global));
        assert_eq!(MemoryScope::parse("nope"), None);
    }

    #[test]
    fn parse_query_trims_and_rejects_empty() {
        assert_eq!(parse_query(&json!({"query": "  auth  "})).unwrap(), "auth");
        assert!(parse_query(&json!({"query": "  "})).is_err());
        assert!(parse_query(&json!({})).is_err());
    }

    #[test]
    fn search_ranks_by_term_overlap_and_phrase_bonus() {
        let dir = tmp();
        let ws = dir.join("ws.md");
        let global = dir.join("global.md");
        let sessions = dir.join("ws.sessions.md");
        remember(&ws, "auth uses JWT tokens with rotation").unwrap();
        remember(&ws, "database is postgres").unwrap();
        remember(&global, "prefer 2-space indentation").unwrap();
        record_session_summary_at(&sessions, "2026-07-11: debugging the auth flow");

        let sources = [
            ("workspace", ws.clone()),
            ("global", global.clone()),
            ("session", sessions.clone()),
        ];
        let hits = search_in(&sources, "auth JWT", 6);
        // The JWT fact (2 term hits) outranks the session log (1 term hit); the
        // unrelated postgres/indentation entries don't appear at all.
        assert!(hits.len() >= 2);
        assert!(hits[0].text.contains("JWT"));
        assert_eq!(hits[0].source, "workspace");
        assert!(hits.iter().all(|h| !h.text.contains("postgres")));
        assert!(hits.iter().all(|h| !h.text.contains("indentation")));

        // Limit is honored; empty query returns nothing.
        assert_eq!(search_in(&sources, "auth", 1).len(), 1);
        assert!(search_in(&sources, "   ", 6).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_summary_refreshes_same_topic_across_dates() {
        let dir = tmp();
        let log = dir.join("s.sessions.md");
        record_session_summary_at(&log, "2026-07-10: fix the parser");
        record_session_summary_at(&log, "2026-07-11: fix the parser");
        record_session_summary_at(&log, "2026-07-11: fix the parser");
        // Same topic dedups (latest date wins); a new topic stacks.
        assert_eq!(load_entries(&log), vec!["2026-07-11: fix the parser"]);
        record_session_summary_at(&log, "2026-07-11: add dark mode");
        assert_eq!(load_entries(&log).len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Path-explicit variant of [`record_session_summary`] (which resolves via config dir).
    fn record_session_summary_at(path: &Path, entry: &str) {
        append_fact(path, SESSION_HEADER, entry.to_string(), session_topic).unwrap();
    }

    #[test]
    fn worktree_gitfile_maps_to_main_root() {
        let dir = tmp();
        let main = dir.join("repo");
        let wt = dir.join("repo").join(".claude").join("wt");
        std::fs::create_dir_all(main.join(".git").join("worktrees").join("wt")).unwrap();
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(
            wt.join(".git"),
            format!("gitdir: {}\n", main.join(".git/worktrees/wt").display()),
        )
        .unwrap();
        assert_eq!(project_root(&wt), main);
        // Same key for main checkout and worktree → one shared memory file.
        assert_eq!(project_memory_path(&wt), project_memory_path(&main));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
