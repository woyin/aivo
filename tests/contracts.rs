//! Executable architecture contracts: settled decisions asserted over the
//! source tree, so violating one fails CI instead of relying on memory.
//! Exceptions are conscious edits to the lists here, or an inline
//! `contract-ok: <reason>` waiver on the hit line.

mod support;

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Cut at the first `#[cfg(test)]` so fixture strings don't false-positive.
fn code_only(content: &str) -> &str {
    match content.find("#[cfg(test)]") {
        Some(i) => &content[..i],
        None => content,
    }
}

fn find_token(file: &Path, content: &str, token: &str) -> Vec<String> {
    let mut hits = Vec::new();
    for (n, line) in content.lines().enumerate() {
        if line.contains(token) && !line.contains("contract-ok:") {
            hits.push(format!("{}:{}", file.display(), n + 1));
        }
    }
    hits
}

// Corruption self-heals via full repaint; clearing caused visible corruption.
// Any crossterm clear needs the `ClearType` import, so the token ban covers
// the class. keys.rs's line-level clear is outside this scope.
#[test]
fn tui_never_clears_the_terminal() {
    let root = repo();
    let mut files = rust_sources(&root.join("src/commands/code_tui"));
    files.push(root.join("src/tui.rs"));
    let mut violations = Vec::new();
    for f in &files {
        let content = std::fs::read_to_string(f).unwrap();
        for token in ["ClearType", "terminal.clear("] {
            violations.extend(find_token(f, &content, token));
        }
    }
    assert!(
        violations.is_empty(),
        "terminal-clear reintroduced in the code TUI — renders must self-heal \
via full repaint, never clear (corruption history). Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn removed_features_stay_removed() {
    let dead: &[(&str, &str)] = &[
        (
            "\"/vision\"",
            "/vision command removed; the vision fallback shim stays",
        ),
        (
            "\"/review\"",
            "/review command removed; review MODE + evaluate.md stay",
        ),
        (
            "\"/detach\"",
            "/detach removed; [image #n] draft tags are the only attachment surface",
        ),
        (
            "mod packs",
            "extension packs dropped in 4274b419; skills/agents/hooks/MCP install separately",
        ),
    ];
    let mut violations = Vec::new();
    for f in rust_sources(&repo().join("src")) {
        let content = std::fs::read_to_string(&f).unwrap();
        for (token, reason) in dead {
            for hit in find_token(&f, &content, token) {
                violations.push(format!("{hit}: {token} — {reason}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "removed feature symbol reintroduced:\n{}",
        violations.join("\n")
    );
}

// One terminal owner: modules render into buffers or a passed-in writer
// (inline_images' kitty sequences); acquiring stdout elsewhere interleaves
// output and corrupts frames.
#[test]
fn tui_stdout_acquisition_confined_to_event_loop() {
    let allowed = ["event_loop_impl.rs"];
    let mut violations = Vec::new();
    for f in rust_sources(&repo().join("src/commands/code_tui")) {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        if allowed.contains(&name.as_str()) {
            continue;
        }
        let content = std::fs::read_to_string(&f).unwrap();
        for token in ["io::stdout(", "execute!(", "queue!("] {
            violations.extend(find_token(&f, &content, token));
        }
    }
    assert!(
        violations.is_empty(),
        "stdout acquired outside the event loop — route output through the \
render path or a passed writer. Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn no_direct_prints_in_code_tui() {
    let mut violations = Vec::new();
    for f in rust_sources(&repo().join("src/commands/code_tui")) {
        if f.components().any(|c| c.as_os_str() == "tests") {
            continue;
        }
        let content = std::fs::read_to_string(&f).unwrap();
        let code = code_only(&content);
        for token in ["println!(", "print!(", "eprintln!(", "eprint!("] {
            violations.extend(find_token(&f, code, token));
        }
    }
    assert!(
        violations.is_empty(),
        "direct print in the code TUI — writes must go through the render \
path. Violations:\n{}",
        violations.join("\n")
    );
}

// Public repo. Fixture paths use the generic user names below; bare
// "yuanchuan" is the public plugin GitHub handle, so only the email stem is
// banned. share_redact.rs is exempt from credential shapes: they're its
// subject matter.
#[test]
fn no_personal_paths_or_credential_shapes() {
    let fixture_users = ["alice", "x", "u", "me", "dev", "user", "example", "tester"];
    let root = repo();
    let mut files = rust_sources(&root.join("src"));
    files.extend(rust_sources(&root.join("tests")));
    let this_file = Path::new(file!()).file_name().unwrap().to_owned();
    let mut violations = Vec::new();
    for f in &files {
        if f.file_name() == Some(this_file.as_os_str()) {
            continue; // names the banned shapes itself
        }
        let is_redactor = f.file_name().is_some_and(|n| n == "share_redact.rs");
        let content = std::fs::read_to_string(f).unwrap();
        for (n, line) in content.lines().enumerate() {
            let mut from = 0;
            while let Some(i) = line[from..].find("/Users/") {
                let rest = &line[from + i + "/Users/".len()..];
                let user: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if !user.is_empty() && !fixture_users.contains(&user.as_str()) {
                    violations.push(format!(
                        "{}:{}: personal path /Users/{user}",
                        f.display(),
                        n + 1
                    ));
                }
                from += i + "/Users/".len();
            }
            if line.contains("yuanchuan23") {
                violations.push(format!("{}:{}: personal identifier", f.display(), n + 1));
            }
            if !is_redactor && line.contains("AKIA") && !line.contains("EXAMPLE") {
                violations.push(format!(
                    "{}:{}: AWS-key shape without EXAMPLE marker",
                    f.display(),
                    n + 1
                ));
            }
            if !is_redactor && line.contains("ghp_") {
                violations.push(format!("{}:{}: GitHub-token shape", f.display(), n + 1));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "public-surface hygiene (this repo is public):\n{}",
        violations.join("\n")
    );
}
