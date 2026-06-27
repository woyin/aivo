//! OpenAI V4A `apply_patch` grammar: the context-anchored, line-number-free patch
//! format GPT-5/Codex models emit. Parsed into per-file changes, then applied with
//! 3-level fuzzy context matching (exact → trailing-ws → all-ws).

use crate::agent::tools::{atomic_write, resolve};
use std::path::{Path, PathBuf};

const BEGIN: &str = "*** Begin Patch";
const END: &str = "*** End Patch";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Ctx,
    Del,
    Ins,
}

/// One `@@`-delimited hunk: ordered (op, line) pairs.
type Section = Vec<(Op, String)>;

#[derive(Debug)]
enum ChangeKind {
    Add {
        content: String,
    },
    Delete,
    Update {
        move_to: Option<String>,
        sections: Vec<Section>,
    },
}

#[derive(Debug)]
struct FileChange {
    path: String,
    kind: ChangeKind,
}

/// A before/after block for the chat diff card; `path` carries any rename arrow.
pub struct DiffBlock {
    pub path: String,
    pub old: String,
    pub new: String,
}

/// Parse the V4A body between `*** Begin Patch`/`*** End Patch` (anything outside
/// the markers — fences, a heredoc wrapper — is ignored).
fn parse(patch: &str) -> Result<Vec<FileChange>, String> {
    let lines: Vec<String> = patch
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
        .collect();
    let start = lines
        .iter()
        .position(|l| l.trim() == BEGIN)
        .ok_or("patch must contain '*** Begin Patch'")?;
    let end = lines
        .iter()
        .rposition(|l| l.trim() == END)
        .ok_or("patch must contain '*** End Patch'")?;
    if end <= start {
        return Err("'*** End Patch' appears before '*** Begin Patch'".into());
    }

    let mut changes = Vec::new();
    let mut i = start + 1;
    while i < end {
        let line = &lines[i];
        if line.trim().is_empty() {
            i += 1;
            continue;
        }
        if let Some(p) = line.strip_prefix("*** Add File: ") {
            let path = p.trim().to_string();
            i += 1;
            let mut content = Vec::new();
            while i < end && !lines[i].starts_with("*** ") {
                content.push(lines[i].strip_prefix('+').unwrap_or(&lines[i]).to_string());
                i += 1;
            }
            let content = if content.is_empty() {
                String::new()
            } else {
                format!("{}\n", content.join("\n"))
            };
            changes.push(FileChange {
                path,
                kind: ChangeKind::Add { content },
            });
        } else if let Some(p) = line.strip_prefix("*** Delete File: ") {
            changes.push(FileChange {
                path: p.trim().to_string(),
                kind: ChangeKind::Delete,
            });
            i += 1;
        } else if let Some(p) = line.strip_prefix("*** Update File: ") {
            let path = p.trim().to_string();
            i += 1;
            let move_to = lines
                .get(i)
                .and_then(|l| l.strip_prefix("*** Move to: "))
                .map(|m| m.trim().to_string());
            if move_to.is_some() {
                i += 1;
            }
            let mut sections: Vec<Section> = Vec::new();
            let mut cur: Section = Vec::new();
            while i < end && !lines[i].starts_with("*** ") {
                let l = &lines[i];
                if l.starts_with("@@") {
                    if !cur.is_empty() {
                        sections.push(std::mem::take(&mut cur));
                    }
                } else {
                    cur.push(classify(l)?);
                }
                i += 1;
            }
            if !cur.is_empty() {
                sections.push(cur);
            }
            if sections.is_empty() {
                return Err(format!("'*** Update File: {path}' has no hunks"));
            }
            changes.push(FileChange {
                path,
                kind: ChangeKind::Update { move_to, sections },
            });
        } else {
            return Err(format!("unexpected line in patch: {line:?}"));
        }
    }
    if changes.is_empty() {
        return Err("patch contained no file actions".into());
    }
    Ok(changes)
}

/// ` ` context, `-` deletion, `+` insertion; a bare empty line is blank context.
fn classify(l: &str) -> Result<(Op, String), String> {
    match l.as_bytes().first() {
        None => Ok((Op::Ctx, String::new())),
        Some(b'+') => Ok((Op::Ins, l[1..].to_string())),
        Some(b'-') => Ok((Op::Del, l[1..].to_string())),
        Some(b' ') => Ok((Op::Ctx, l[1..].to_string())),
        _ => Err(format!("hunk line must start with ' ', '+', or '-': {l:?}")),
    }
}

/// Locate `block` at/after `start`, trying exact → trailing-ws → trimmed matches.
fn find_context(lines: &[&str], block: &[&str], start: usize) -> Option<usize> {
    if block.is_empty() {
        return Some(start.min(lines.len()));
    }
    if block.len() > lines.len() {
        return None;
    }
    let passes: [fn(&str, &str) -> bool; 3] = [
        |a, b| a == b,
        |a, b| a.trim_end() == b.trim_end(),
        |a, b| a.trim() == b.trim(),
    ];
    for eq in passes {
        let mut i = start;
        while i + block.len() <= lines.len() {
            if (0..block.len()).all(|k| eq(lines[i + k], block[k])) {
                return Some(i);
            }
            i += 1;
        }
    }
    None
}

/// Apply an Update's sections to `orig`: each section's context+deletions are
/// located forward from the running position, so repeated context resolves in order.
fn apply_update(orig: &str, sections: &[Section]) -> Result<String, String> {
    let lines: Vec<&str> = orig.lines().collect();
    let mut out: Vec<String> = Vec::new();
    let mut pos = 0usize;
    for sec in sections {
        let before: Vec<&str> = sec
            .iter()
            .filter(|(o, _)| *o != Op::Ins)
            .map(|(_, t)| t.as_str())
            .collect();
        let at = find_context(&lines, &before, pos)
            .ok_or_else(|| format!("could not locate patch context:\n{}", before.join("\n")))?;
        out.extend(lines[pos..at].iter().map(|s| s.to_string()));
        // Walk file offsets so fuzzy-matched context keeps the file's own text.
        let mut f = at;
        for (op, text) in sec {
            match op {
                Op::Ctx => {
                    out.push(lines[f].to_string());
                    f += 1;
                }
                Op::Del => f += 1,
                Op::Ins => out.push(text.clone()),
            }
        }
        pos = f;
    }
    out.extend(lines[pos..].iter().map(|s| s.to_string()));
    let sep = if orig.contains("\r\n") { "\r\n" } else { "\n" };
    let mut result = out.join(sep);
    if orig.ends_with('\n') && !result.is_empty() {
        result.push_str(sep);
    }
    Ok(result)
}

/// Every path a patch touches (rename destinations included); empty if unparseable.
pub fn target_paths(patch: &str) -> Vec<String> {
    let Ok(changes) = parse(patch) else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for ch in &changes {
        paths.push(ch.path.clone());
        if let ChangeKind::Update {
            move_to: Some(m), ..
        } = &ch.kind
        {
            paths.push(m.clone());
        }
    }
    paths
}

/// Before/after blocks for the chat diff card. Empty if the patch doesn't parse.
pub fn diff_blocks(patch: &str) -> Vec<DiffBlock> {
    let Ok(changes) = parse(patch) else {
        return Vec::new();
    };
    let mut blocks = Vec::new();
    for ch in &changes {
        match &ch.kind {
            ChangeKind::Add { content } => blocks.push(DiffBlock {
                path: format!("{} (new)", ch.path),
                old: String::new(),
                new: content.clone(),
            }),
            ChangeKind::Delete => blocks.push(DiffBlock {
                path: format!("{} (deleted)", ch.path),
                old: String::new(),
                new: String::new(),
            }),
            ChangeKind::Update { move_to, sections } => {
                let label = match move_to {
                    Some(m) => format!("{} → {}", ch.path, m),
                    None => ch.path.clone(),
                };
                for sec in sections {
                    let pick = |skip: Op| {
                        sec.iter()
                            .filter(|(o, _)| *o != skip)
                            .map(|(_, t)| t.clone())
                            .collect::<Vec<_>>()
                            .join("\n")
                    };
                    blocks.push(DiffBlock {
                        path: label.clone(),
                        old: pick(Op::Ins),
                        new: pick(Op::Del),
                    });
                }
            }
        }
    }
    blocks
}

/// Parse and apply a V4A patch under `cwd`, computing every change in memory
/// before writing so a validation error leaves the workspace untouched.
pub fn apply(patch: &str, cwd: &Path) -> Result<String, String> {
    let changes = parse(patch)?;
    enum Step {
        Write(PathBuf, String),
        Remove(PathBuf),
    }
    let mut steps: Vec<Step> = Vec::new();
    let mut summary: Vec<String> = Vec::new();
    for ch in &changes {
        match &ch.kind {
            ChangeKind::Add { content } => {
                let full = resolve(cwd, &ch.path);
                if full.exists() {
                    return Err(format!("Add File: {} already exists", ch.path));
                }
                steps.push(Step::Write(full, content.clone()));
                summary.push(format!("+{}", ch.path));
            }
            ChangeKind::Delete => {
                let full = resolve(cwd, &ch.path);
                if !full.exists() {
                    return Err(format!("Delete File: {} not found", ch.path));
                }
                steps.push(Step::Remove(full));
                summary.push(format!("-{}", ch.path));
            }
            ChangeKind::Update { move_to, sections } => {
                let full = resolve(cwd, &ch.path);
                let content =
                    std::fs::read_to_string(&full).map_err(|e| format!("read {}: {e}", ch.path))?;
                let updated =
                    apply_update(&content, sections).map_err(|e| format!("{}: {e}", ch.path))?;
                match move_to {
                    Some(m) => {
                        let dest = resolve(cwd, m);
                        steps.push(Step::Write(dest.clone(), updated));
                        if dest != full {
                            steps.push(Step::Remove(full));
                        }
                        summary.push(format!("{} → {}", ch.path, m));
                    }
                    None => {
                        steps.push(Step::Write(full, updated));
                        summary.push(format!("~{}", ch.path));
                    }
                }
            }
        }
    }
    for step in steps {
        match step {
            Step::Write(p, c) => {
                if let Some(parent) = p.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("create dir for {}: {e}", p.display()))?;
                }
                atomic_write(&p, &c).map_err(|e| format!("write {}: {e}", p.display()))?;
            }
            Step::Remove(p) => {
                std::fs::remove_file(&p).map_err(|e| format!("remove {}: {e}", p.display()))?;
            }
        }
    }
    Ok(format!(
        "applied patch ({} file{}): {}",
        changes.len(),
        if changes.len() == 1 { "" } else { "s" },
        summary.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cwd() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn update_replaces_anchored_block() {
        let dir = cwd();
        std::fs::write(
            dir.path().join("a.rs"),
            "fn main() {\n    let x = 1;\n    foo();\n}\n",
        )
        .unwrap();
        let patch = "*** Begin Patch\n*** Update File: a.rs\n@@ fn main()\n     let x = 1;\n-    foo();\n+    bar();\n }\n*** End Patch";
        let out = apply(patch, dir.path()).unwrap();
        assert!(out.contains("~a.rs"));
        let after = std::fs::read_to_string(dir.path().join("a.rs")).unwrap();
        assert_eq!(after, "fn main() {\n    let x = 1;\n    bar();\n}\n");
    }

    #[test]
    fn fuzzy_matches_trailing_whitespace() {
        let dir = cwd();
        // File has a trailing space the model's context omits.
        std::fs::write(dir.path().join("b.txt"), "alpha \nbeta\n").unwrap();
        let patch = "*** Begin Patch\n*** Update File: b.txt\n alpha\n-beta\n+gamma\n*** End Patch";
        apply(patch, dir.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("b.txt")).unwrap(),
            "alpha \ngamma\n"
        );
    }

    #[test]
    fn add_and_delete_files() {
        let dir = cwd();
        std::fs::write(dir.path().join("old.txt"), "gone\n").unwrap();
        let patch = "*** Begin Patch\n*** Add File: new/x.txt\n+hello\n+world\n*** Delete File: old.txt\n*** End Patch";
        apply(patch, dir.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new/x.txt")).unwrap(),
            "hello\nworld\n"
        );
        assert!(!dir.path().join("old.txt").exists());
    }

    #[test]
    fn move_to_renames_and_edits() {
        let dir = cwd();
        std::fs::write(dir.path().join("src.txt"), "one\ntwo\n").unwrap();
        let patch = "*** Begin Patch\n*** Update File: src.txt\n*** Move to: dst.txt\n one\n-two\n+TWO\n*** End Patch";
        apply(patch, dir.path()).unwrap();
        assert!(!dir.path().join("src.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("dst.txt")).unwrap(),
            "one\nTWO\n"
        );
    }

    #[test]
    fn missing_context_errors_without_writing() {
        let dir = cwd();
        std::fs::write(dir.path().join("c.txt"), "real\n").unwrap();
        let patch =
            "*** Begin Patch\n*** Update File: c.txt\n nope\n-real\n+changed\n*** End Patch";
        assert!(apply(patch, dir.path()).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("c.txt")).unwrap(),
            "real\n"
        );
    }

    #[test]
    fn add_existing_file_is_rejected() {
        let dir = cwd();
        std::fs::write(dir.path().join("e.txt"), "x\n").unwrap();
        let patch = "*** Begin Patch\n*** Add File: e.txt\n+y\n*** End Patch";
        assert!(apply(patch, dir.path()).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("e.txt")).unwrap(),
            "x\n"
        );
    }

    #[test]
    fn missing_envelope_errors() {
        assert!(parse("*** Update File: a\n x\n").is_err());
    }

    #[test]
    fn target_paths_lists_sources_and_dests() {
        let patch = "*** Begin Patch\n*** Update File: a.txt\n*** Move to: b.txt\n x\n-x\n+y\n*** Add File: c.txt\n+z\n*** End Patch";
        let paths = target_paths(patch);
        assert_eq!(paths, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn ordered_sections_resolve_repeated_context() {
        let dir = cwd();
        std::fs::write(dir.path().join("d.txt"), "x\ny\nx\n").unwrap();
        // Two sections each anchored on "x"; the second must match the later one.
        let patch =
            "*** Begin Patch\n*** Update File: d.txt\n-x\n+a\n@@\n y\n-x\n+b\n*** End Patch";
        apply(patch, dir.path()).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("d.txt")).unwrap(),
            "a\ny\nb\n"
        );
    }
}
