use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn glob_recursive_and_flat() {
    let dir = tmp();
    write_file(&json!({"path":"src/main.rs","content":"x"}), &dir).unwrap();
    write_file(&json!({"path":"src/lib/util.rs","content":"x"}), &dir).unwrap();
    write_file(&json!({"path":"top.rs","content":"x"}), &dir).unwrap();
    let all = glob(&json!({"pattern":"**/*.rs"}), &dir).unwrap();
    assert!(all.contains("src/main.rs"));
    assert!(all.contains("src/lib/util.rs"));
    assert!(all.contains("top.rs"));
    let flat = glob(&json!({"pattern":"*.rs"}), &dir).unwrap();
    assert!(flat.contains("top.rs"));
    assert!(!flat.contains("src/main.rs"));
}

#[test]
fn glob_skips_ignored_dirs() {
    let dir = tmp();
    write_file(&json!({"path":"node_modules/dep/x.rs","content":"x"}), &dir).unwrap();
    write_file(&json!({"path":"keep.rs","content":"x"}), &dir).unwrap();
    let out = glob(&json!({"pattern":"**/*.rs"}), &dir).unwrap();
    assert!(out.contains("keep.rs"));
    assert!(!out.contains("node_modules"));
}

/// A self-referential symlink (`loop -> .`) must not make the glob walk
/// recurse forever (stack overflow): symlinked directories are never
/// descended into. The walk terminating at all is the real assertion.
#[cfg(unix)]
#[test]
fn glob_does_not_follow_symlink_cycle() {
    let dir = tmp();
    write_file(&json!({"path":"real.rs","content":"x"}), &dir).unwrap();
    std::os::unix::fs::symlink(&dir, dir.join("loop")).unwrap();
    let out = glob(&json!({"pattern":"**/*.rs"}), &dir).unwrap();
    assert!(out.contains("real.rs"));
    assert!(!out.contains("loop/"), "descended through a symlink: {out}");
}

/// The pure-Rust grep fallback skips symlinks during traversal — both so it
/// matches ripgrep's default file set and so a symlink cycle can't overflow
/// the stack. Drives `grep_fallback` directly (the public `grep` would prefer
/// rg/grep when installed, never reaching the fallback).
#[cfg(unix)]
#[test]
fn grep_fallback_skips_symlinks() {
    let dir = tmp();
    write_file(&json!({"path":"f.txt","content":"needle"}), &dir).unwrap();
    std::os::unix::fs::symlink(&dir, dir.join("loop")).unwrap();
    let mut out = Vec::new();
    grep_fallback(&dir, &dir, "needle", 0, &mut out);
    assert!(
        out.iter().any(|l| l.contains("f.txt")),
        "missing match: {out:?}"
    );
    assert!(
        !out.iter().any(|l| l.contains("loop")),
        "followed a symlink during traversal: {out:?}"
    );
}

#[cfg(unix)]
#[test]
fn grep_fallback_skips_fifo() {
    let dir = tmp();
    write_file(&json!({"path":"f.txt","content":"needle"}), &dir).unwrap();
    mkfifo(&dir.join("pipe"));
    let mut out = Vec::new();
    grep_fallback(&dir, &dir, "needle", 0, &mut out);
    assert!(
        out.iter().any(|l| l.contains("f.txt")),
        "missing match: {out:?}"
    );
}

#[test]
fn glob_match_semantics() {
    assert!(glob_match("*.rs", "main.rs"));
    assert!(!glob_match("*.rs", "src/main.rs"));
    assert!(glob_match("**/*.rs", "src/a/b.rs"));
    assert!(glob_match("src/*.rs", "src/main.rs"));
    assert!(glob_match("src/**", "src/a/b/c.rs"));
    assert!(glob_match("?.txt", "a.txt"));
    assert!(!glob_match("?.txt", "ab.txt"));
}

#[tokio::test]
async fn grep_finds_match() {
    let dir = tmp();
    write_file(
        &json!({"path":"f.txt","content":"alpha\nbeta\ngamma"}),
        &dir,
    )
    .unwrap();
    let out = grep(&json!({"pattern":"beta"}), &dir).await.unwrap();
    assert!(out.contains("beta"));
}

/// Consistency: grep skips IGNORED_DIRS (so the heavy build dirs never show)
/// the same way whether it runs via rg, grep, or the pure-Rust fallback.
#[tokio::test]
async fn grep_skips_ignored_dirs() {
    let dir = tmp();
    write_file(
        &json!({"path":"node_modules/dep/x.txt","content":"needle"}),
        &dir,
    )
    .unwrap();
    write_file(&json!({"path":"keep.txt","content":"needle"}), &dir).unwrap();
    let out = grep(&json!({"pattern":"needle"}), &dir).await.unwrap();
    assert!(out.contains("keep.txt"), "missing kept file: {out}");
    assert!(!out.contains("node_modules"), "ignored dir leaked: {out}");
}

/// Consistency: grep does NOT honor .gitignore, so a gitignored file is still
/// found — and crucially, found the same way regardless of whether `rg` (which
/// would otherwise hide it) is installed.
#[tokio::test]
async fn grep_ignores_gitignore() {
    let dir = tmp();
    std::fs::write(dir.join(".gitignore"), "secret.txt\n").unwrap();
    write_file(&json!({"path":"secret.txt","content":"needle here"}), &dir).unwrap();
    let out = grep(&json!({"pattern":"needle"}), &dir).await.unwrap();
    assert!(
        out.contains("secret.txt"),
        "gitignored file should still be searched (consistency): {out}"
    );
}

/// Tier-agnostic: rg, grep, and the pure-Rust fallback all honor `context`.
#[tokio::test]
async fn grep_context_shows_neighbors() {
    let dir = tmp();
    write_file(
        &json!({"path":"f.txt","content":"before_line\nHITHERE\nafter_line"}),
        &dir,
    )
    .unwrap();
    let plain = grep(&json!({"pattern":"HITHERE"}), &dir).await.unwrap();
    assert!(plain.contains("HITHERE") && !plain.contains("before_line"));
    let ctx = grep(&json!({"pattern":"HITHERE","context":1}), &dir)
        .await
        .unwrap();
    assert!(
        ctx.contains("before_line") && ctx.contains("HITHERE") && ctx.contains("after_line"),
        "context=1 should include both neighbors: {ctx}"
    );
}

#[test]
fn emit_context_zero_is_one_line_per_match() {
    let lines = ["a", "b", "c"];
    let mut out = Vec::new();
    emit_context("f", &lines, &[1], 0, &mut out);
    assert_eq!(out, vec!["f:2:b"]);
}

#[test]
fn emit_context_marks_match_and_context_separators() {
    let lines = ["a", "b", "c", "d", "e"];
    let mut out = Vec::new();
    emit_context("f", &lines, &[2], 1, &mut out);
    assert_eq!(out, vec!["f-2-b", "f:3:c", "f-4-d"]);
}

#[test]
fn emit_context_merges_adjacent_and_splits_disjoint() {
    let lines = ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"];
    let mut merged = Vec::new();
    emit_context("f", &lines, &[1, 2], 1, &mut merged); // context 1 → one window
    assert!(!merged.iter().any(|l| l == "--"), "adjacent: {merged:?}");
    assert_eq!(merged.len(), 4); // lines 0..=3

    let mut split = Vec::new();
    emit_context("f", &lines, &[1, 5], 1, &mut split); // gap → two windows
    assert!(split.iter().any(|l| l == "--"), "disjoint: {split:?}");
}
