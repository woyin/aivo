//! Search tools: glob and grep with ignored-dir walking and context windows.

use super::*;

pub(super) fn glob(args: &Value, cwd: &Path) -> Result<String, String> {
    let pattern = arg_str(args, "pattern")?;
    let base = resolve(cwd, arg_str_opt(args, "path").unwrap_or("."));
    let mut out = Vec::new();
    walk_glob(&base, &base, pattern, &mut out);
    out.sort();
    if out.is_empty() {
        return Ok("(no matches)".to_string());
    }
    Ok(cap_head(out.join("\n")))
}

pub(super) fn walk_glob(root: &Path, dir: &Path, pattern: &str, out: &mut Vec<String>) {
    if out.len() >= GLOB_CAP {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = rd.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let path = e.path();
        // `file_type()` reads the entry's own type WITHOUT following the link, so
        // a symlinked directory is treated as a leaf and never descended into.
        // That bounds the walk: a symlink cycle (`loop -> .`) would otherwise
        // recurse until the stack overflows. A symlink whose name matches the
        // pattern is still listed below; we just never traverse through it.
        let is_symlink = e.file_type().map(|t| t.is_symlink()).unwrap_or(false);
        let is_dir = !is_symlink && path.is_dir();
        let name = e.file_name();
        if is_dir && IGNORED_DIRS.contains(&name.to_string_lossy().as_ref()) {
            continue;
        }
        if let Ok(rel) = path.strip_prefix(root) {
            let rel_s = rel.to_string_lossy().replace('\\', "/");
            if glob_match(pattern, &rel_s) {
                out.push(rel_s);
                if out.len() >= GLOB_CAP {
                    return;
                }
            }
        }
        if is_dir {
            walk_glob(root, &path, pattern, out);
        }
    }
}

/// Path-segment glob: `**` matches zero or more segments; within a segment `*`
/// matches any run of non-`/` chars and `?` matches one char.
pub(super) fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<&str> = pattern.split('/').collect();
    let txt: Vec<&str> = text.split('/').collect();
    seg_match(&pat, &txt)
}

pub(super) fn seg_match(pat: &[&str], txt: &[&str]) -> bool {
    match pat.first() {
        None => txt.is_empty(),
        Some(&"**") => (0..=txt.len()).any(|i| seg_match(&pat[1..], &txt[i..])),
        Some(seg) => !txt.is_empty() && wildcard(seg, txt[0]) && seg_match(&pat[1..], &txt[1..]),
    }
}

pub(super) fn wildcard(pat: &str, text: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = text.chars().collect();
    wm(&p, &t)
}

pub(super) fn wm(p: &[char], t: &[char]) -> bool {
    match p.first() {
        None => t.is_empty(),
        Some('*') => (0..=t.len()).any(|i| wm(&p[1..], &t[i..])),
        Some('?') => !t.is_empty() && wm(&p[1..], &t[1..]),
        Some(&c) => !t.is_empty() && t[0] == c && wm(&p[1..], &t[1..]),
    }
}

pub(super) async fn grep(args: &Value, cwd: &Path) -> Result<String, String> {
    let pattern = arg_str(args, "pattern")?;
    let path = arg_str_opt(args, "path").unwrap_or(".");
    // Context lines per match (grep -C), clamped so it can't dump whole files.
    let context = grep_context(args) as usize;
    // All three tiers (rg → grep -rn → pure-Rust walk) must search the SAME file
    // set, or results would vary by which tool is installed. The pure-Rust walk
    // defines the contract: skip only `IGNORED_DIRS`, do NOT honor `.gitignore`.
    // So rg runs with `--no-ignore --hidden` + per-dir excludes (rg otherwise
    // honors .gitignore and hides dotfiles), and grep gets matching --exclude-dir.

    // Prefer ripgrep, then grep; both exit 1 (empty stdout) on no matches.
    let mut rg_args: Vec<String> = vec![
        "--line-number".into(),
        "--no-heading".into(),
        "--color=never".into(),
        "--no-ignore".into(),
        "--hidden".into(),
    ];
    if context > 0 {
        rg_args.push("-C".into());
        rg_args.push(context.to_string());
    }
    for dir in IGNORED_DIRS {
        rg_args.push("-g".into());
        rg_args.push(format!("!{dir}"));
    }
    rg_args.push(pattern.to_string());
    rg_args.push(path.to_string());
    let rg_refs: Vec<&str> = rg_args.iter().map(String::as_str).collect();
    if let Some(out) = run_capture("rg", &rg_refs, cwd).await {
        return Ok(grep_result(out));
    }

    let mut grep_args: Vec<String> = vec!["-rn".into()];
    if context > 0 {
        grep_args.push("-C".into());
        grep_args.push(context.to_string());
    }
    for dir in IGNORED_DIRS {
        grep_args.push(format!("--exclude-dir={dir}"));
    }
    grep_args.push(pattern.to_string());
    grep_args.push(path.to_string());
    let grep_refs: Vec<&str> = grep_args.iter().map(String::as_str).collect();
    if let Some(out) = run_capture("grep", &grep_refs, cwd).await {
        return Ok(grep_result(out));
    }

    // No external tool: literal-substring walk (also skips IGNORED_DIRS only).
    let base = resolve(cwd, path);
    let mut out = Vec::new();
    grep_fallback(&base, &base, pattern, context, &mut out);
    if out.is_empty() {
        return Ok("(no matches)".to_string());
    }
    Ok(cap_head(out.join("\n")))
}

pub(super) fn grep_result(out: String) -> String {
    if out.trim().is_empty() {
        "(no matches)".to_string()
    } else {
        cap_head(out)
    }
}

pub(super) fn grep_fallback(
    root: &Path,
    dir: &Path,
    needle: &str,
    context: usize,
    out: &mut Vec<String>,
) {
    if out.len() >= GLOB_CAP {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.filter_map(|e| e.ok()) {
        // Skip symlinks during traversal, matching ripgrep's default (the grep
        // fast path) — so the pure-Rust fallback searches the SAME file set as
        // rg, and a symlinked directory cycle (`loop -> .`) can't recurse until
        // the stack overflows.
        let Ok(ft) = e.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        let path = e.path();
        let name = e.file_name();
        if ft.is_dir() {
            if !IGNORED_DIRS.contains(&name.to_string_lossy().as_ref()) {
                grep_fallback(root, &path, needle, context, out);
            }
            continue;
        }
        // rg skips non-regular files too; a FIFO would block read_to_string.
        if !ft.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = content.lines().collect();
        let matched: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(needle))
            .map(|(i, _)| i)
            .collect();
        if matched.is_empty() {
            continue;
        }
        // grep -C divides separated groups (including across files) with `--`.
        if context > 0 && !out.is_empty() {
            out.push("--".into());
        }
        emit_context(&rel, &lines, &matched, context, out);
        if out.len() >= GLOB_CAP {
            return;
        }
    }
}

/// Emit one file's matches like `grep -C`: `:` on match lines, `-` on context
/// lines, `--` between non-adjacent windows. context 0 = one path:line:text/match.
pub(super) fn emit_context(
    rel: &str,
    lines: &[&str],
    matched: &[usize],
    context: usize,
    out: &mut Vec<String>,
) {
    let mut last: Option<usize> = None; // last emitted line index
    for &m in matched {
        let start = m.saturating_sub(context);
        let end = (m + context).min(lines.len().saturating_sub(1));
        let from = match last {
            Some(l) if start <= l + 1 => l + 1, // merge with previous window
            Some(_) => {
                out.push("--".into());
                start
            }
            None => start,
        };
        for (offset, line) in lines[from..=end].iter().enumerate() {
            let i = from + offset;
            let sep = if matched.binary_search(&i).is_ok() {
                ':'
            } else {
                '-'
            };
            out.push(format!("{rel}{sep}{}{sep}{}", i + 1, line));
            if out.len() >= GLOB_CAP {
                return;
            }
        }
        last = Some(end);
    }
}
