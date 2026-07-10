//! Named specialist sub-agents, defined as Markdown files (frontmatter + body),
//! the way the `subagents/` slot works in Vercel's Eve and `.claude/agents/` does
//! in Claude Code. A sub-agent is a single `<name>.md` with YAML frontmatter
//! (`description` [the model-visible hint used to decide delegation], optional
//! `model`, optional `tools` to scope its toolset) and a body that becomes its
//! system-prompt instructions. The parent's generic `subagent` tool gains an
//! `agent` parameter naming which specialist to run; an absent profile falls back
//! to the generic sub-agent. Discovery is progressive-disclosure, mirroring
//! `skills`: only names + one-line descriptions ride in the system prompt.
//!
//! Discovery: project dirs (`.aivo/agents`, `.claude/agents`) before user-global
//! `~/.config/aivo/agents`, first name wins — same trust posture as project skills.
//! `tools:` may use Claude Code vocabulary (mapped; unknown names ignored, never
//! stripping to zero). `isolation: worktree` = disposable git worktree per run.

use crate::agent::skills::advert_description;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Subagent {
    pub name: String,
    pub description: String,
    /// Optional model id the sub-agent runs on (else it inherits the parent's).
    pub model: Option<String>,
    /// Optional allow-list of tool names (raw, as authored). Resolve through
    /// [`Subagent::resolved_tools`] to map onto aivo's built-ins before scoping.
    pub tools: Option<Vec<String>>,
    /// The body after the frontmatter — the sub-agent's extra instructions.
    pub body: String,
    /// `isolation: worktree` — run in a disposable git worktree (snapshot of HEAD).
    pub isolation_worktree: bool,
    pub source: PathBuf,
}

impl Subagent {
    /// The authored `tools` list mapped onto aivo's built-in tool names (Claude
    /// Code's `Read`/`Bash`/… included), unknown names dropped, deduped. `None`
    /// when no scope was authored OR nothing resolved — so an unrecognized list
    /// never strips the sub-agent down to zero tools (it just runs unscoped).
    pub fn resolved_tools(&self) -> Option<Vec<&'static str>> {
        let raw = self.tools.as_ref()?;
        let mut out: Vec<&'static str> = Vec::new();
        for name in raw {
            if let Some(mapped) = normalize_tool_name(name)
                && !out.contains(&mapped)
            {
                out.push(mapped);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

/// Project dirs first (a repo can ship/shadow profiles), then user-global.
pub fn discover_subagents(cwd: &Path, config_dir: &Path) -> Vec<Subagent> {
    discover_from_roots(&[
        cwd.join(".aivo").join("agents"),
        cwd.join(".claude").join("agents"),
        config_dir.join("agents"),
    ])
}

/// Collect sub-agents from `roots` in order, first name winning on collision.
fn discover_from_roots(roots: &[PathBuf]) -> Vec<Subagent> {
    let mut found: Vec<Subagent> = Vec::new();
    for root in roots {
        for sa in read_root(root) {
            if !found.iter().any(|e| e.name == sa.name) {
                found.push(sa);
            }
        }
    }
    found
}

/// Parse every `<root>/<name>.md` under one root (alphabetical, deterministic).
/// Missing/unreadable roots yield nothing.
fn read_root(root: &Path) -> Vec<Subagent> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "md"))
        .collect();
    files.sort();
    files.iter().filter_map(|p| load_subagent(p)).collect()
}

/// Load one `<name>.md`; `None` if it's unreadable or its resolved name is
/// invalid (so a stray file can't inject a bogus tool-enum value).
fn load_subagent(path: &Path) -> Option<Subagent> {
    let text = std::fs::read_to_string(path).ok()?;
    let stem = path.file_stem()?.to_string_lossy().into_owned();
    let (front, body) = split_frontmatter(&text);
    let name = front
        .as_ref()
        .and_then(|f| field(f, "name"))
        .unwrap_or(stem);
    if !is_valid_name(&name) {
        return None;
    }
    let description = front
        .as_ref()
        .and_then(|f| field(f, "description"))
        .unwrap_or_else(|| first_non_empty_line(body));
    let model = front.as_ref().and_then(|f| field(f, "model"));
    let tools = front.as_ref().and_then(|f| field_list(f, "tools"));
    let isolation_worktree = front
        .as_ref()
        .and_then(|f| field(f, "isolation"))
        .is_some_and(|v| v.eq_ignore_ascii_case("worktree"));
    Some(Subagent {
        name,
        description,
        model,
        tools,
        body: body.trim().to_string(),
        isolation_worktree,
        source: path.to_path_buf(),
    })
}

/// A usable sub-agent name (also the `agent` enum value): non-empty, `[A-Za-z0-9_-]`.
pub fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Map an authored tool name (aivo's own, or Claude Code's vocabulary) onto one of
/// aivo's built-in tool names. `None` for anything unrecognized. The match is on a
/// canonical form (lowercased, separators stripped) so `read_file`, `Read`,
/// `read-file`, and `ReadFile` all land on `read_file`.
pub fn normalize_tool_name(name: &str) -> Option<&'static str> {
    let canon: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    Some(match canon.as_str() {
        "read" | "readfile" | "view" | "cat" | "openfile" => "read_file",
        "write" | "writefile" | "create" | "createfile" | "newfile" => "write_file",
        "edit" | "editfile" | "strreplace" | "strreplaceeditor" => "edit_file",
        // apply_patch is its own built-in (V4A patch via `input`), NOT edit_file
        // (which needs path/old/new) — mapping it to edit_file breaks GPT-5/Codex.
        "apply" | "applypatch" | "patch" => "apply_patch",
        "multiedit" | "multiedits" => "multi_edit",
        "bash" | "shell" | "runbash" | "terminal" | "exec" | "command" | "run" => "run_bash",
        "grep" | "search" | "ripgrep" | "rg" | "searchtext" => "grep",
        "glob" | "find" | "findfiles" | "filesearch" => "glob",
        "ls" | "list" | "listdir" | "listfiles" | "dir" => "list_dir",
        "webfetch" | "fetch" | "fetchurl" | "urlfetch" | "http" | "httpget" => "web_fetch",
        "skill" | "skills" | "loadskill" => "skill",
        _ => return None,
    })
}

/// The system-prompt block advertising available sub-agents (names + one-line
/// descriptions). Empty when there are none. Mirrors `skills_prompt_section`.
pub fn subagents_prompt_section(subagents: &[Subagent]) -> String {
    if subagents.is_empty() {
        return String::new();
    }
    let mut list = String::new();
    for sa in subagents {
        list.push_str(&format!(
            "\n- {}: {}",
            sa.name,
            advert_description(&sa.description)
        ));
    }
    format!(
        "\n\nYou have specialist sub-agents — pre-configured roles you can delegate to. To use one, \
call the `subagent` tool with its name in the `agent` field (plus a complete, standalone `task`). \
Each runs its own loop with its own instructions and only the `task` you pass — it never sees this \
conversation — and hands back a result. Omit `agent` for a generic sub-agent. Available \
sub-agents:{list}"
    )
}

// ── worktree isolation ────────────────────────────────────────────────────────

/// Disposable detached worktree of `parent`'s HEAD for one sub-agent run. Err =
/// why isolation is unavailable — callers fall back to the shared workspace.
pub fn create_worktree(parent: &Path) -> Result<PathBuf, String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "aivo-worktree-{}-{}",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let out = std::process::Command::new("git")
        .args([
            "-C",
            &parent.display().to_string(),
            "worktree",
            "add",
            "--detach",
        ])
        .arg(&dir)
        .output()
        .map_err(|e| format!("git not runnable: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(dir)
}

/// Unchanged → remove the worktree; changed → keep it and tell the parent where
/// it is and how to apply. Appended to the sub-agent's result.
pub fn finalize_worktree(parent: &Path, wt: &Path) -> String {
    let porcelain = std::process::Command::new("git")
        .args(["-C", &wt.display().to_string(), "status", "--porcelain"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    if porcelain.is_empty() {
        let _ = std::process::Command::new("git")
            .args([
                "-C",
                &parent.display().to_string(),
                "worktree",
                "remove",
                "--force",
            ])
            .arg(wt)
            .output();
        return "\n\n[worktree isolation] The sub-agent ran in an isolated worktree; it made no file changes, so the worktree was removed.".to_string();
    }
    let changed = porcelain.lines().count();
    format!(
        "\n\n[worktree isolation] The sub-agent's file changes are in an isolated worktree at {wt_disp} ({changed} path(s) changed) — NOT in your workspace. Review with `git -C {wt_disp} status`/`diff`; apply with `git -C {wt_disp} add -A && git -C {wt_disp} diff --cached | git -C {parent_disp} apply`; then clean up with `git -C {parent_disp} worktree remove --force {wt_disp}`.",
        wt_disp = wt.display(),
        parent_disp = parent.display(),
    )
}

// ── minimal frontmatter parsing (same shape as skills::SKILL.md) ─────────────

/// Split a leading `---`…`---` YAML block from the body. Returns `(frontmatter,
/// body)`; the frontmatter is `None` when the file doesn't open with `---`.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let rest = match text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    {
        Some(rest) => rest,
        None => return (None, text),
    };
    for marker in ["\n---\n", "\n---\r\n", "\r\n---\r\n"] {
        if let Some(idx) = rest.find(marker) {
            return (Some(&rest[..idx]), &rest[idx + marker.len()..]);
        }
    }
    if let Some(stripped) = rest.strip_suffix("\n---") {
        return (Some(stripped), "");
    }
    (None, text)
}

/// Pull a single-line `key: value` from a minimal frontmatter block, trimming
/// matched surrounding quotes.
fn field(frontmatter: &str, key: &str) -> Option<String> {
    for line in frontmatter.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(key)
            && let Some(value) = rest.trim_start().strip_prefix(':')
        {
            let value = value.trim();
            let unquoted = value
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
                .unwrap_or(value);
            if !unquoted.is_empty() {
                return Some(unquoted.to_string());
            }
        }
    }
    None
}

/// Parse a `key:` value as a list: an inline `[a, b]` or a bare `a, b, c`,
/// comma-separated, trimmed, empties dropped. Block-style YAML lists (one item
/// per line) are not parsed — an authored `tools:` with no inline value yields
/// `None`, leaving the sub-agent unscoped (safe default).
fn field_list(frontmatter: &str, key: &str) -> Option<Vec<String>> {
    let raw = field(frontmatter, key)?;
    let inner = raw
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .unwrap_or(&raw);
    let items: Vec<String> = inner
        .split(',')
        .map(|s| s.trim().trim_matches(['"', '\'']).trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if items.is_empty() { None } else { Some(items) }
}

fn first_non_empty_line(body: &str) -> String {
    body.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "aivo-subagents-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(root: &Path, file: &str, contents: &str) {
        std::fs::write(root.join(file), contents).unwrap();
    }

    #[test]
    fn parses_frontmatter_model_and_tools() {
        let root = tmp();
        write(
            &root,
            "reviewer.md",
            "---\nname: reviewer\ndescription: \"Reviews a diff for bugs.\"\nmodel: anthropic/claude-opus-4-8\ntools: read_file, grep, Bash\n---\nYou are a careful code reviewer. Be terse.\n",
        );
        let subs = discover_from_roots(&[root]);
        assert_eq!(subs.len(), 1);
        let r = &subs[0];
        assert_eq!(r.name, "reviewer");
        assert_eq!(r.description, "Reviews a diff for bugs.");
        assert_eq!(r.model.as_deref(), Some("anthropic/claude-opus-4-8"));
        assert_eq!(r.body, "You are a careful code reviewer. Be terse.");
        // Authored vocabulary maps onto aivo's built-ins (incl. Claude's `Bash`).
        assert_eq!(
            r.resolved_tools(),
            Some(vec!["read_file", "grep", "run_bash"])
        );
    }

    #[test]
    fn inline_bracket_tools_list_parses() {
        let root = tmp();
        write(
            &root,
            "t.md",
            "---\nname: t\ndescription: d\ntools: [Read, Edit, Write]\n---\nbody\n",
        );
        let subs = discover_from_roots(&[root]);
        assert_eq!(
            subs[0].resolved_tools(),
            Some(vec!["read_file", "edit_file", "write_file"])
        );
    }

    #[test]
    fn isolation_worktree_parses_from_frontmatter() {
        let root = tmp();
        write(
            &root,
            "fixer.md",
            "---\nname: fixer\ndescription: d\nisolation: worktree\n---\nb\n",
        );
        write(
            &root,
            "plain.md",
            "---\nname: plain\ndescription: d\n---\nb\n",
        );
        let subs = discover_from_roots(&[root]);
        assert!(
            subs.iter()
                .find(|s| s.name == "fixer")
                .unwrap()
                .isolation_worktree
        );
        assert!(
            !subs
                .iter()
                .find(|s| s.name == "plain")
                .unwrap()
                .isolation_worktree
        );
    }

    #[test]
    fn discovers_project_dirs_before_user_global() {
        let cwd = tmp();
        let config = tmp();
        let aivo = cwd.join(".aivo/agents");
        let claude = cwd.join(".claude/agents");
        std::fs::create_dir_all(&aivo).unwrap();
        std::fs::create_dir_all(&claude).unwrap();
        std::fs::create_dir_all(config.join("agents")).unwrap();
        write(
            &aivo,
            "dup.md",
            "---\nname: dup\ndescription: from aivo\n---\nA\n",
        );
        write(
            &claude,
            "dup.md",
            "---\nname: dup\ndescription: from claude\n---\nB\n",
        );
        write(
            &claude,
            "cc.md",
            "---\nname: cc\ndescription: claude only\n---\nC\n",
        );
        write(
            &config.join("agents"),
            "global.md",
            "---\nname: global\ndescription: user global\n---\nG\n",
        );
        let subs = discover_subagents(&cwd, &config);
        let names: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["dup", "cc", "global"]);
        assert_eq!(subs[0].description, "from aivo", ".aivo shadows .claude");
    }

    #[test]
    fn worktree_roundtrip_isolates_changes() {
        let repo = tmp();
        let git = |args: &[&str]| {
            let out = std::process::Command::new("git")
                .arg("-C")
                .arg(&repo)
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        git(&["init", "-q"]);
        std::fs::write(repo.join("a.txt"), "one").unwrap();
        git(&["add", "-A"]);
        git(&["commit", "-qm", "init"]);

        // Unchanged worktree → removed.
        let wt = create_worktree(&repo).unwrap();
        assert!(wt.join("a.txt").is_file(), "worktree snapshots HEAD");
        let note = finalize_worktree(&repo, &wt);
        assert!(note.contains("no file changes"), "{note}");
        assert!(!wt.exists(), "clean worktree removed");

        // Changed worktree → kept, reported, parent untouched.
        let wt = create_worktree(&repo).unwrap();
        std::fs::write(wt.join("b.txt"), "two").unwrap();
        let note = finalize_worktree(&repo, &wt);
        assert!(note.contains("1 path(s) changed"), "{note}");
        assert!(wt.join("b.txt").is_file(), "changed worktree kept");
        assert!(!repo.join("b.txt").exists(), "parent tree untouched");
        // Not a repo → Err (callers fall back to the shared workspace).
        assert!(create_worktree(&tmp()).is_err());
    }

    #[test]
    fn falls_back_to_filename_and_first_line() {
        let root = tmp();
        write(&root, "helper.md", "Just an instruction line.\nmore\n");
        let subs = discover_from_roots(&[root]);
        assert_eq!(subs[0].name, "helper");
        assert_eq!(subs[0].description, "Just an instruction line.");
        assert!(subs[0].model.is_none());
        assert!(subs[0].tools.is_none());
        assert!(subs[0].resolved_tools().is_none());
    }

    #[test]
    fn unknown_tools_resolve_to_none_not_empty_scope() {
        // A list of names that don't map to any aivo built-in must NOT scope the
        // sub-agent down to zero tools — it falls back to unscoped.
        let root = tmp();
        write(
            &root,
            "x.md",
            "---\nname: x\ndescription: d\ntools: TodoWrite, mcp__foo__bar\n---\nb\n",
        );
        let subs = discover_from_roots(&[root]);
        assert!(subs[0].tools.is_some(), "raw list is preserved");
        assert!(
            subs[0].resolved_tools().is_none(),
            "no known names → unscoped, not empty"
        );
    }

    #[test]
    fn earlier_root_shadows_same_name() {
        let project = tmp();
        let user = tmp();
        write(
            &project,
            "dup.md",
            "---\nname: dup\ndescription: from project\n---\nA\n",
        );
        write(
            &user,
            "dup.md",
            "---\nname: dup\ndescription: from user\n---\nB\n",
        );
        let subs = discover_from_roots(&[project, user]);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].description, "from project");
    }

    #[test]
    fn invalid_name_is_skipped() {
        let root = tmp();
        write(
            &root,
            "bad.md",
            "---\nname: has spaces\ndescription: d\n---\nb\n",
        );
        write(
            &root,
            "good.md",
            "---\nname: good\ndescription: d\n---\nb\n",
        );
        let subs = discover_from_roots(&[root]);
        let names: Vec<&str> = subs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["good"]);
    }

    #[test]
    fn non_md_files_ignored() {
        let root = tmp();
        write(&root, "notes.txt", "not a subagent");
        write(&root, "r.md", "---\nname: r\ndescription: d\n---\nb\n");
        let subs = discover_from_roots(&[root]);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].name, "r");
    }

    #[test]
    fn prompt_section_lists_names_and_truncates() {
        let subs = vec![Subagent {
            name: "reviewer".into(),
            description: format!("Reviews diffs. {}", "blah ".repeat(60)),
            model: None,
            tools: None,
            body: String::new(),
            isolation_worktree: false,
            source: PathBuf::new(),
        }];
        let section = subagents_prompt_section(&subs);
        assert!(section.contains("`agent`"));
        assert!(section.contains("- reviewer: Reviews diffs."));
        assert!(!section.contains("blah blah blah blah"));
        assert!(subagents_prompt_section(&[]).is_empty());
    }

    #[test]
    fn tool_name_mapping_covers_both_vocabularies() {
        assert_eq!(normalize_tool_name("Read"), Some("read_file"));
        assert_eq!(normalize_tool_name("read_file"), Some("read_file"));
        assert_eq!(normalize_tool_name("MultiEdit"), Some("multi_edit"));
        assert_eq!(normalize_tool_name("bash"), Some("run_bash"));
        assert_eq!(normalize_tool_name("Glob"), Some("glob"));
        assert_eq!(normalize_tool_name("WebFetch"), Some("web_fetch"));
        assert_eq!(normalize_tool_name("Frobnicate"), None);
        // apply_patch is idempotent and distinct from edit_file — mapping it to
        // edit_file (different arg schema) breaks GPT-5/Codex editing entirely.
        assert_eq!(normalize_tool_name("apply_patch"), Some("apply_patch"));
        assert_eq!(normalize_tool_name("applypatch"), Some("apply_patch"));
        assert_eq!(normalize_tool_name("apply"), Some("apply_patch"));
        assert_eq!(normalize_tool_name("str_replace_editor"), Some("edit_file"));
        assert_eq!(normalize_tool_name("edit_file"), Some("edit_file"));
    }
}
