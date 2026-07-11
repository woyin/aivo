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

const HEADER: &str = "# aivo project memory\n\
Durable facts and decisions this project's agent saved with the `remember` tool.\n\
One `- ` bullet per memory; safe to edit or delete lines by hand.\n";

/// `<config>/memory/<sanitized-root>.md`, keyed by repo root (main checkout
/// for linked worktrees) so all worktrees share one memory.
pub fn project_memory_path(cwd: &Path) -> PathBuf {
    let root = project_root(cwd);
    let sanitized: Vec<char> = root
        .to_string_lossy()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    // Truncate from the front: the tail carries the project name.
    let key: String = sanitized[sanitized.len().saturating_sub(120)..]
        .iter()
        .collect();
    crate::services::paths::config_dir()
        .join("memory")
        .join(format!("{key}.md"))
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
    let mut entries = load_entries(path);
    let refreshed = if let Some(pos) = entries.iter().position(|e| *e == fact) {
        entries.remove(pos);
        true
    } else {
        false
    };
    entries.push(fact);
    while entries.len() > MAX_ENTRIES
        || entries.iter().map(|e| e.len() + 3).sum::<usize>() + HEADER.len() > MAX_FILE_BYTES
    {
        entries.remove(0);
    }
    if let Some(dir) = path.parent() {
        crate::services::atomic_write::ensure_private_dir_blocking(dir)
            .map_err(|e| format!("create memory dir: {e}"))?;
    }
    let mut out = String::with_capacity(HEADER.len() + 64 * entries.len());
    out.push_str(HEADER);
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
        description: "Save one durable fact about this project to persistent memory — it is \
injected into EVERY future session here, unlike `take_note` (which lasts only for the current \
session). Use it sparingly, for things worth knowing weeks from now: a decision and its why, a \
user preference or correction, a non-obvious constraint or gotcha. Don't save session progress, \
anything derivable from the code, or secrets. One concise fact per call."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to persist, as one self-contained sentence (who/what/why included)."
                }
            },
            "required": ["fact"]
        }),
    }
}

pub fn parse_fact(args: &serde_json::Value) -> Result<String, String> {
    args.get("fact")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| "remember: missing `fact`.".to_string())
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
