use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn edit_requires_unique_match() {
    let dir = tmp();
    write_file(&json!({"path":"c.txt","content":"x\nx\n"}), &dir).unwrap();
    let err = edit_file(
        &json!({"path":"c.txt","old_string":"x","new_string":"y"}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("2 times"));
    write_file(&json!({"path":"d.txt","content":"foo bar"}), &dir).unwrap();
    edit_file(
        &json!({"path":"d.txt","old_string":"bar","new_string":"baz"}),
        &dir,
    )
    .unwrap();
    let out = read_file(&json!({"path":"d.txt"}), &dir).unwrap();
    assert!(out.contains("foo baz"));
}

#[test]
fn edit_missing_string_errors() {
    let dir = tmp();
    write_file(&json!({"path":"e.txt","content":"abc"}), &dir).unwrap();
    let err = edit_file(
        &json!({"path":"e.txt","old_string":"zzz","new_string":"q"}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("not found"));
}

#[cfg(unix)]
#[test]
fn edit_tools_refuse_fifo() {
    let dir = tmp();
    mkfifo(&dir.join("pipe"));
    let err = edit_file(
        &json!({"path":"pipe","old_string":"a","new_string":"b"}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("not a regular file"), "got: {err}");
    let err = multi_edit(
        &json!({"path":"pipe","edits":[{"old_string":"a","new_string":"b"}]}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("not a regular file"), "got: {err}");
}

#[test]
fn edit_replace_all_replaces_every_occurrence() {
    let dir = tmp();
    write_file(&json!({"path":"r.txt","content":"a a a"}), &dir).unwrap();
    // Without replace_all, an ambiguous match is refused (safe default).
    let err = edit_file(
        &json!({"path":"r.txt","old_string":"a","new_string":"b"}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("set replace_all"), "got: {err}");
    // With replace_all, all occurrences change and the count is reported.
    let ok = edit_file(
        &json!({"path":"r.txt","old_string":"a","new_string":"b","replace_all":true}),
        &dir,
    )
    .unwrap();
    assert!(ok.contains("3 replacements"), "got: {ok}");
    let out = read_file(&json!({"path":"r.txt"}), &dir).unwrap();
    assert!(out.contains("b b b"));
}

#[test]
fn edit_rejects_empty_and_noop() {
    let dir = tmp();
    write_file(&json!({"path":"n.txt","content":"abc"}), &dir).unwrap();
    let empty = edit_file(
        &json!({"path":"n.txt","old_string":"","new_string":"x"}),
        &dir,
    )
    .unwrap_err();
    assert!(empty.contains("must not be empty"));
    let noop = edit_file(
        &json!({"path":"n.txt","old_string":"abc","new_string":"abc"}),
        &dir,
    )
    .unwrap_err();
    assert!(noop.contains("identical"));
}

#[test]
fn multi_edit_is_atomic_and_sequential() {
    let dir = tmp();
    write_file(&json!({"path":"m.txt","content":"one two three"}), &dir).unwrap();
    // Two good edits apply in order.
    let ok = multi_edit(
        &json!({"path":"m.txt","edits":[
            {"old_string":"one","new_string":"1"},
            {"old_string":"two","new_string":"2"}
        ]}),
        &dir,
    )
    .unwrap();
    assert!(ok.contains("2 edits"), "got: {ok}");
    let out = read_file(&json!({"path":"m.txt"}), &dir).unwrap();
    assert!(out.contains("1 2 three"));

    // A failing later edit leaves the file untouched (atomic).
    let err = multi_edit(
        &json!({"path":"m.txt","edits":[
            {"old_string":"1","new_string":"X"},
            {"old_string":"absent","new_string":"Y"}
        ]}),
        &dir,
    )
    .unwrap_err();
    assert!(err.contains("edit #2"), "got: {err}");
    let after = read_file(&json!({"path":"m.txt"}), &dir).unwrap();
    assert!(after.contains("1 2 three"), "file was half-edited: {after}");
}

/// An edit whose args use LF still lands on a CRLF file, and the file keeps
/// its CRLF endings (inserted text included) instead of being corrupted.
#[test]
fn edit_matches_crlf_file_with_lf_args_and_preserves_endings() {
    let dir = tmp();
    // Written directly: write_file would normalize to the arg's LF.
    std::fs::write(dir.join("c.txt"), "alpha\r\nbeta\r\ngamma\r\n").unwrap();
    let ok = edit_file(
        &json!({"path":"c.txt","old_string":"beta\ngamma","new_string":"beta\nGAMMA"}),
        &dir,
    )
    .unwrap();
    assert!(ok.contains("edited c.txt"), "got: {ok}");
    let raw = std::fs::read_to_string(dir.join("c.txt")).unwrap();
    assert!(raw.contains("GAMMA"), "edit did not land: {raw:?}");
    assert!(
        raw.contains("beta\r\nGAMMA\r\n"),
        "CRLF endings not preserved: {raw:?}"
    );
    assert!(
        !raw.contains("beta\nGAMMA"),
        "introduced a lone LF: {raw:?}"
    );
}

#[test]
fn edit_no_match_flags_leaked_line_number_prefixes() {
    let content = "fn main() {\n    let x = 1;\n}\n";
    let old = "    12\tfn main() {\n    13\t    let x = 1;";
    let err = apply_one_edit(content, old, "X", false, "f.rs").unwrap_err();
    assert!(err.contains("line-number prefixes"), "got: {err}");
}

#[test]
fn edit_no_match_points_at_whitespace_mismatch_with_exact_text() {
    let content = "impl Foo {\n    fn bar(&self) -> u32 {\n        self.n\n    }\n}\n";
    let old = "impl Foo {\n\tfn bar(&self) -> u32 {\n\t\tself.n\n\t}\n}";
    let err = apply_one_edit(content, old, "X", false, "f.rs").unwrap_err();
    assert!(err.contains("whitespace"), "got: {err}");
    assert!(
        err.contains("    fn bar(&self) -> u32 {"),
        "snippet must carry the file's exact text: {err}"
    );
    assert!(err.contains("Lines 1\u{2013}5"), "got: {err}");
}

#[test]
fn edit_no_match_anchors_on_the_closest_distinctive_line() {
    let content = "a\nb\nfn compute_total_price(cart: &Cart) -> f64 {\n    cart.sum()\n}\n";
    let old = "fn compute_total_price(cart: &Cart) -> f64 {\n    cart.total\n}";
    let err = apply_one_edit(content, old, "X", false, "f.rs").unwrap_err();
    assert!(err.contains("near line 3"), "got: {err}");
    assert!(
        err.contains("cart.sum()"),
        "snippet must show the actual body: {err}"
    );
}

/// Different wrong old_strings must share one failure signature (streak guard).
#[test]
fn edit_no_match_error_head_is_stable_across_attempts() {
    let content = "impl Foo {\n    fn bar(&self) -> u32 {\n        self.n\n    }\n}\n";
    let e1 = apply_one_edit(
        content,
        "impl Foo {\n\tfn bar(&self) -> u32 {",
        "X",
        false,
        "f.rs",
    )
    .unwrap_err();
    let e2 = apply_one_edit(
        content,
        "totally unrelated missing text",
        "X",
        false,
        "f.rs",
    )
    .unwrap_err();
    assert_ne!(e1, e2, "hints differ per attempt");
    assert_eq!(
        crate::agent::guards::failure_signature("edit_file", &e1),
        crate::agent::guards::failure_signature("edit_file", &e2),
        "the signature head must stay constant per path"
    );
}

/// Numeric-key literals must not read as leaked line numbers (tab-only rule).
#[test]
fn edit_no_match_does_not_flag_numeric_key_literals_as_prefixes() {
    let content = "codes = {\n    \"OK\",\n    \"Not Found\",\n}\n";
    let old = "200: \"OK\",\n404: \"Not Found\",";
    let err = apply_one_edit(content, old, "X", false, "f.rs").unwrap_err();
    assert!(
        !err.contains("line-number prefixes"),
        "colon-form numeric keys misclassified: {err}"
    );
}

/// Blank boundary lines must not anchor the window scan.
#[test]
fn edit_no_match_ignores_blank_boundary_lines_in_window_scan() {
    let content = "alpha\nbeta\ngamma\n";
    let old = "\n\tbeta\n";
    let err = apply_one_edit(content, old, "X", false, "f.rs").unwrap_err();
    assert!(
        err.contains("Lines 2\u{2013}2") || err.contains("beta"),
        "hint should anchor on the real line: {err}"
    );
    assert!(
        !err.contains("Lines 1\u{2013}"),
        "must not anchor on a blank-line pseudo-match at the top: {err}"
    );
}

/// A deep anchor must still appear inside the capped snippet.
#[test]
fn edit_no_match_anchor_stays_inside_the_capped_snippet() {
    let mut content: String = (1..=19).map(|i| format!("x{i}\n")).collect();
    content.push_str("let the_special_marker = compute_value();\n");
    let old: String = (1..=9).map(|i| format!("aaa{i}\n")).collect::<String>()
        + "let the_special_marker = compute_value();";
    let err = apply_one_edit(&content, &old, "X", false, "f.rs").unwrap_err();
    assert!(err.contains("near line 20"), "got: {err}");
    assert!(
        err.contains("the_special_marker"),
        "the named line must be inside the snippet: {err}"
    );
}

/// Brace-only old_strings must not produce a (wrong-block) whitespace hint.
#[test]
fn edit_no_match_skips_window_scan_for_brace_only_old_strings() {
    let err = apply_one_edit(
        "}\n{\nlet unique_a = 1;\n}\n{\n",
        "\t}\n\t{",
        "X\nY",
        false,
        "f.rs",
    )
    .unwrap_err();
    assert!(
        !err.contains("differ only in whitespace"),
        "brace-only window must not be trusted: {err}"
    );
    assert!(err.contains("re-read"), "got: {err}");
}

#[test]
fn edit_no_match_suggests_reread_when_nothing_is_similar() {
    let err = apply_one_edit(
        "alpha beta\n",
        "zzz qqq totally different line here",
        "X",
        false,
        "f.rs",
    )
    .unwrap_err();
    assert!(err.contains("re-read"), "got: {err}");
}
