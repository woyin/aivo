use super::super::*;
use super::helpers::*;

#[test]
fn test_in_flight_tool_card_hidden_until_result() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "lsof -ti:3000"}),
        vec![],
        None,
    );
    // In flight: only the status names it — the `→ run_bash(…)` card is held back
    // so the same action isn't shown twice (the dup the user reported).
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("run_bash("),
        "card hidden while running: {plain:?}"
    );
    assert!(
        plain.contains("running lsof"),
        "status names the step: {plain:?}"
    );
    // Result lands → the card (with the command) renders.
    app.apply_agent_tool_result("ok".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("run_bash("),
        "card shown after result: {plain:?}"
    );
}

#[test]
fn test_parallel_bridged_batch_counts_and_lists_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for (id, task) in [("1", "Audit auth"), ("2", "Map engine"), ("3", "Scan CLI")] {
        app.apply_agent_tool_call(
            Some(id.to_string()),
            "subagent".to_string(),
            serde_json::json!({"label": task}),
            vec![],
            None,
        );
    }
    assert_eq!(app.desired_status(), "running 3 sub-agents (0 done)");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("↳ Audit auth"), "per-call rows: {plain:?}");
    assert!(plain.contains("↳ Scan CLI"), "per-call rows: {plain:?}");

    // Newest call resolving first: no "Thinking" flip, card held with the batch.
    app.apply_agent_tool_update("3".to_string(), None, Some("12 files".to_string()), false);
    assert_eq!(app.desired_status(), "running 3 sub-agents (1 done)");
    assert!(app.current_action_label().is_some());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("✓ Scan CLI — 12 files"),
        "done row: {plain:?}"
    );
    assert!(
        !plain.contains("Scan CLI ⎿") && !plain.contains("subagent("),
        "cards held until the batch settles: {plain:?}"
    );

    // All resolved → the batch is over: rows gone, cards render.
    app.apply_agent_tool_update("1".to_string(), None, Some("ok".to_string()), false);
    app.apply_agent_tool_update("2".to_string(), None, Some("ok".to_string()), true);
    assert_ne!(app.desired_status(), "running 3 sub-agents (3 done)");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("↳ "), "live rows cleared: {plain:?}");
    assert!(plain.contains("Audit auth"), "cards render: {plain:?}");
}

#[test]
fn test_parallel_bridged_batch_mixed_tools_noun() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for (id, name, args) in [
        ("1", "grep", serde_json::json!({"pattern": "hover"})),
        ("2", "subagent", serde_json::json!({"label": "Audit"})),
    ] {
        app.apply_agent_tool_call(Some(id.to_string()), name.to_string(), args, vec![], None);
    }
    assert_eq!(app.desired_status(), "running 2 parallel steps (0 done)");
}

/// Clicking a folded `!cmd` block's `▸ +N more lines` expander reveals the full
/// output inline (the in-process successor to the ctrl+o pager); clicking the
/// `▾ collapse` toggle folds it back to the preview. Drives the real render +
/// hitbox + click-mapping path end to end.
#[test]
fn output_block_expands_inline_on_click() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 250 lines: past the 40-line fold preview, so the block renders an expander.
    let full: String = (1..=250).map(|i| format!("L{i:04}\n")).collect();
    app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    let idx = app.history.len() - 1;

    // The transcript rows the click handler resolves against — built exactly as the
    // real render does (wrap the body, then seed the hitbox).
    let refresh = |app: &mut CodeTuiApp| -> Vec<String> {
        let body = app.build_transcript_history_body(80);
        let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
        app.transcript_hitbox = Some(TranscriptHitbox {
            area: Rect::new(0, 0, 80, 40),
            first_row: 0,
            rows: wrapped.rows.clone(),
        });
        wrapped.rows
    };

    // Folded: the preview stops at the cap, the tail (L0250) is hidden behind a
    // clickable expander.
    let rows = refresh(&mut app);
    assert!(rows.iter().any(|r| r.contains("+210 more lines")));
    assert!(rows.iter().all(|r| !r.contains("L0250")));
    let marker_row = rows
        .iter()
        .position(|r| is_output_expander(r))
        .expect("folded block renders an expander row");

    // Clicking the expander row toggles this block into `expanded_output`.
    assert!(app.toggle_output_at_row(marker_row));
    assert!(app.expanded_output.contains(&idx));

    // Expanded: the full tail is shown in place, over a `▾ collapse` toggle.
    let rows = refresh(&mut app);
    assert!(
        rows.iter().any(|r| r.contains("L0250")),
        "expanded block reveals the elided tail:\n{}",
        rows.join("\n")
    );
    let collapse_row = rows
        .iter()
        .position(|r| r.trim_start().starts_with(OUTPUT_EXPANDED_PREFIX))
        .expect("expanded block renders a collapse toggle");

    // Clicking the collapse toggle folds it back.
    assert!(app.toggle_output_at_row(collapse_row));
    assert!(!app.expanded_output.contains(&idx));
    let rows = refresh(&mut app);
    assert!(rows.iter().all(|r| !r.contains("L0250")));
}

#[test]
fn test_edit_diff_rows_carry_add_remove_tints() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let lines = app.build_transcript().lines;
    let bg_of = |needle: &str| -> Option<Color> {
        lines
            .iter()
            .find(|l| l.plain.contains(needle))
            .and_then(|l| l.line.spans.iter().find_map(|s| s.style.bg))
    };
    assert_eq!(
        bg_of("- let x = 1;"),
        Some(DIFF_DEL_BG()),
        "removed line tint"
    );
    assert_eq!(
        bg_of("+ let x = 2;"),
        Some(DIFF_ADD_BG()),
        "added line tint"
    );
}

#[test]
fn test_edit_diff_word_highlight_brightens_only_changed_tokens() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let lines = app.build_transcript().lines;
    // Background of the span carrying exactly `text`, on the line containing `needle`.
    let span_bg = |needle: &str, text: &str| -> Option<Color> {
        lines
            .iter()
            .find(|l| l.plain.contains(needle))
            .and_then(|l| l.line.spans.iter().find(|s| s.content.as_ref() == text))
            .and_then(|s| s.style.bg)
    };
    // Only the differing token jumps to the brighter emphasis tint; the shared
    // run of the line keeps the base tint (the intra-line word diff).
    assert_eq!(
        span_bg("- let x = 1;", "1"),
        Some(DIFF_DEL_HL_BG()),
        "changed token emphasised on the removed line"
    );
    assert_eq!(
        span_bg("- let x = 1;", "let x = "),
        Some(DIFF_DEL_BG()),
        "common run stays at the base tint"
    );
    assert_eq!(
        span_bg("+ let x = 2;", "2"),
        Some(DIFF_ADD_HL_BG()),
        "changed token emphasised on the added line"
    );
    assert_eq!(
        span_bg("+ let x = 2;", "let x = "),
        Some(DIFF_ADD_BG()),
        "common run stays at the base tint"
    );
}

#[test]
fn test_edit_diff_numbers_rows_from_line_starts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // `old_string` begins at file line 10 (a=10, b=11, c=12); only the middle
    // line changes. `line_starts` is the pre-edit probe the runtime stamps.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"a.rs","old_string":"a\nb\nc","new_string":"a\nB\nc"},"line_starts":[10]}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("11 - b"),
        "removed line numbered by its old-file offset:\n{plain}"
    );
    assert!(
        plain.contains("11 + B"),
        "added line numbered by its new-file offset:\n{plain}"
    );
    assert!(
        plain.lines().any(|l| l.contains("10") && l.contains('a')),
        "leading context carries its file number:\n{plain}"
    );
    assert!(
        plain.lines().any(|l| l.contains("12") && l.contains('c')),
        "trailing context carries its file number:\n{plain}"
    );
}

#[test]
fn test_edit_diff_shows_context_only_marks_changed_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Only the middle line changes; the first and last are shared context.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"fn a() {\n    let y = 2;\n}","new_string":"fn a() {\n    let y = 20;\n}"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let lines = app.build_transcript().lines;
    let find = |needle: &str| lines.iter().find(|l| l.plain.contains(needle));
    let bg_of =
        |needle: &str| find(needle).and_then(|l| l.line.spans.iter().find_map(|s| s.style.bg));

    // The shared lines render as context: present, but with no +/- and no tint.
    let ctx = find("fn a() {").expect("context line missing");
    assert!(
        !ctx.plain.contains("- ") && !ctx.plain.contains("+ "),
        "context line should carry no diff sign: {:?}",
        ctx.plain
    );
    assert_eq!(bg_of("fn a() {"), None, "context line must not be tinted");
    assert!(find("}").is_some(), "trailing context line missing");

    // Only the genuinely changed line is flagged on each side.
    assert_eq!(
        bg_of("- "),
        Some(DIFF_DEL_BG()),
        "old line removed + tinted"
    );
    assert_eq!(bg_of("+ "), Some(DIFF_ADD_BG()), "new line added + tinted");
    assert!(find("- ").unwrap().plain.contains("let y = 2;"));
    assert!(find("+ ").unwrap().plain.contains("let y = 20;"));
    // The unchanged lines are NOT duplicated as removed/added.
    assert!(
        !lines
            .iter()
            .any(|l| l.plain.contains("- fn a()") || l.plain.contains("+ fn a()")),
        "shared line should not appear as a change"
    );
}

#[test]
fn test_edit_diff_trims_context_and_collapses_gap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Two far-apart changes (first and last line) with a long unchanged middle.
    // Context is limited to a few lines, so the middle collapses to a `⋯`.
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"x.rs","old_string":"A\nc1\nc2\nc3\nMID\nc5\nc6\nc7\nB","new_string":"A2\nc1\nc2\nc3\nMID\nc5\nc6\nc7\nB2"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    // Both edited lines are flagged; the deep-interior context collapses.
    assert!(
        plain.contains("- A") && plain.contains("+ A2"),
        "first change:\n{plain}"
    );
    assert!(
        plain.contains("- B") && plain.contains("+ B2"),
        "last change:\n{plain}"
    );
    assert!(
        plain.contains('⋯'),
        "collapsed gap marker missing:\n{plain}"
    );
    assert!(
        !plain.contains("MID"),
        "context beyond the window should be dropped:\n{plain}"
    );
    // Context adjacent to a change is still shown.
    assert!(
        plain.contains("c1") && plain.contains("c7"),
        "near context kept:\n{plain}"
    );
}

#[test]
fn test_agent_events_build_display_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Streamed prose arrives, then the engine calls a tool: the prose is
    // committed as an assistant entry *before* the tool-call entry.
    app.pending_response = "I'll read it.".to_string();
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    assert!(app.pending_response.is_empty());
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[0].role, "assistant");
    assert_eq!(app.history[0].content, "I'll read it.");
    assert_eq!(app.history[1].role, "tool_call");
    assert!(app.history[1].content.contains("read_file"));
    assert!(app.history[1].content.contains("a.rs"));

    app.apply_agent_tool_result("128 lines".to_string());
    assert_eq!(app.history[2].role, "tool_result");
    assert_eq!(app.history[2].content, "128 lines");

    // A second tool call with no intervening prose doesn't inject a blank entry.
    app.apply_agent_tool_call(
        None,
        "edit_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    assert_eq!(app.history.len(), 4);
    assert_eq!(app.history[3].role, "tool_call");
}

#[test]
fn test_folded_run_bash_result_keeps_streaming_tail_height() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Short output stays whole under the summary — nothing vanishes.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "make"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("line one\nline two\nline three".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("line one"), "{plain}");
    assert!(plain.contains("line three"), "{plain}");

    // Long output keeps only the last `STREAM_TAIL_LINES` — no collapse, no flood.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "curl"}),
        vec![],
        None,
    );
    let long = (1..=40)
        .map(|n| format!("row {n}"))
        .collect::<Vec<_>>()
        .join("\n");
    app.apply_agent_tool_result(long);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("row 40"), "last line kept: {plain}");
    assert!(plain.contains("row 37"), "tail window kept: {plain}");
    assert!(
        !plain.contains("row 36"),
        "earlier lines fold away: {plain}"
    );
    assert!(
        !plain.contains("row 20"),
        "earlier lines fold away: {plain}"
    );

    // One huge single line (a JSON blob) is clamped like the live tail rows —
    // not wrapped into a screenful under a "+3 lines" fold.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "gh api"}),
        vec![],
        None,
    );
    let blob = format!("head-marker {}", "x".repeat(30_000));
    app.apply_agent_tool_result(format!("banner line\n{blob}\n[exit 0]"));
    let plain_lines = app.build_transcript().plain_lines;
    let row = plain_lines
        .iter()
        .find(|l| l.contains("head-marker"))
        .expect("tail row present");
    assert!(
        row.len() < 120,
        "tail row must be clamped, got {} chars",
        row.len()
    );

    // A non-run_bash tool has no tail, so it stays a summary line only.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("secret-alpha\nsecret-beta".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("secret-alpha"),
        "a non-run_bash result stays folded: {plain}"
    );
}

#[test]
fn test_parallel_mcp_batch_interleaves_call_and_result() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let names = ["query_qoe_devices", "query_qoe_quality", "query_qoe_users"];
    let markers = ["ALPHA", "BETA", "GAMMA"];
    for name in names {
        app.apply_agent_tool_call(
            None,
            format!("mcp__localhost__{name}"),
            serde_json::json!({"mode": "count"}),
            vec![],
            None,
        );
    }
    for marker in markers {
        app.apply_agent_tool_result(format!(
            "<untrusted source=\"mcp:localhost\">\n{{\"count\":1,\"m\":\"{marker}\"}}\n</untrusted>"
        ));
    }

    let refresh = |app: &mut CodeTuiApp| -> Vec<String> {
        let body = app.build_transcript_history_body(80);
        let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
        app.transcript_hitbox = Some(TranscriptHitbox {
            area: Rect::new(0, 0, 80, 40),
            first_row: 0,
            rows: wrapped.rows.clone(),
        });
        wrapped.rows
    };
    let rows = refresh(&mut app);

    let shape: Vec<&str> = rows
        .iter()
        .filter_map(|r| {
            if r.contains("localhost/") {
                Some("call")
            } else if is_output_expander(r) {
                Some("fold")
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        shape,
        vec!["call", "fold", "call", "fold", "call", "fold"],
        "each fold draws under its call:\n{}",
        rows.join("\n")
    );
    let devices_row = rows
        .iter()
        .position(|r| r.contains("query_qoe_devices"))
        .unwrap();
    assert!(is_output_expander(&rows[devices_row + 1]));
    assert!(markers.iter().all(|m| rows.iter().all(|r| !r.contains(m))));

    let fold_rows: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| is_output_expander(r))
        .map(|(i, _)| i)
        .collect();
    assert!(app.toggle_output_at_row(fold_rows[1]));
    let rows = refresh(&mut app);
    assert!(
        rows.iter().any(|r| r.contains("BETA")),
        "middle result expands:\n{}",
        rows.join("\n")
    );
    assert!(
        rows.iter()
            .all(|r| !r.contains("ALPHA") && !r.contains("GAMMA")),
        "only the clicked result expands:\n{}",
        rows.join("\n")
    );
}

#[test]
fn test_native_tool_paths_render_relative_to_cwd() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/Users/alice/proj".to_string();

    // Paths under cwd render relative (the absolute prefix is footer noise).
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/alice/proj/src/ui/views/panel.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("128 lines".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("read_file(src/ui/views/panel.rs)"),
        "{plain}"
    );
    assert!(
        !plain.contains("/Users/alice/proj"),
        "absolute path leaked: {plain}"
    );

    // Over-long paths left-truncate on a segment boundary to keep the basename.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/alice/proj/src/module/feature/component/section/detail/view/inner/widget.rs"}),
    vec![], None);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("…/"), "expected left-truncation: {plain}");
    assert!(plain.contains("widget.rs"), "basename lost: {plain}");
}

#[test]
fn test_tool_result_count_units_by_tool() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Each tool reports its multi-line count in the right unit, not bare "lines".
    let cases = [
        ("glob", "a.rs\nb.rs\nc.rs", "3 files"),
        ("list_dir", "src/\ntests/", "2 entries"),
        ("grep", "1: x\n2: y\n3: z\n4: w", "4 matches"),
        ("read_file", "line one\nline two", "2 lines"),
    ];
    for (tool, output, expected) in cases {
        app.apply_agent_tool_call(
            None,
            tool.to_string(),
            serde_json::json!({"path": "x"}),
            vec![],
            None,
        );
        app.apply_agent_tool_result(output.to_string());
        let plain = app.build_transcript().plain_lines.join("\n");
        assert!(
            plain.contains(expected),
            "{tool}: expected {expected:?} in {plain}"
        );
    }
}

#[test]
fn test_batched_tool_results_resolve_unit_and_target_by_position() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Batch of [grep×3] then [r0, r1, r2]: each result must resolve its own call
    // by position, not idx-1 (which left the 2nd/3rd mislabelled "lines").
    for pat in ["alpha", "beta", "gamma"] {
        app.apply_agent_tool_call(
            None,
            "grep".to_string(),
            serde_json::json!({"pattern": pat}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_result("1: x\n2: y".to_string()); // 2 matches
    app.apply_agent_tool_result("1: x\n2: y\n3: z".to_string()); // 3 matches
    app.apply_agent_tool_result("1: x\n2: y\n3: z\n4: w".to_string()); // 4 matches

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("2 matches"), "{plain}");
    assert!(plain.contains("3 matches"), "{plain}");
    assert!(plain.contains("4 matches"), "{plain}");
    // Leading `· ` (the search preview that used to trail the label is gone).
    assert!(plain.contains("· alpha"), "{plain}");
    assert!(plain.contains("· beta"), "{plain}");
    assert!(plain.contains("· gamma"), "{plain}");
    assert!(plain.contains("searched ×3"), "{plain}");
    assert!(!plain.contains("searched ×3: alpha"), "{plain}");
}

#[test]
fn test_adjacent_search_tools_merge_into_one_group() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 2 globs + 2 greps both read as "searched" → one `searched ×4` header, not
    // two indistinguishable `searched ×2`, with results keeping their own units.
    for (tool, pat) in [
        ("glob", "**/*canary*"),
        ("glob", "**/*gemini*"),
        ("grep", "gemini"),
        ("grep", "canary"),
    ] {
        app.apply_agent_tool_call(
            None,
            tool.to_string(),
            serde_json::json!({"pattern": pat}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_result("a.rs\nb.rs\nc.rs".to_string()); // glob -> 3 files
    app.apply_agent_tool_result("x.rs".to_string());
    app.apply_agent_tool_result("1:a\n2:b".to_string()); // grep -> 2 matches
    app.apply_agent_tool_result("1:a\n2:b\n3:c\n4:d".to_string()); // grep -> 4 matches

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("searched ×4"), "{plain}");
    assert_eq!(plain.matches("searched ×").count(), 1, "{plain}");
    assert!(plain.contains("3 files"), "{plain}");
    assert!(plain.contains("2 matches"), "{plain}");
    assert!(plain.contains("4 matches"), "{plain}");
    assert!(plain.contains("· **/*canary*"), "{plain}");
    assert!(plain.contains("· gemini"), "{plain}");
}

#[test]
fn test_run_bash_label_strips_redundant_cd_prefix() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/Users/alice/project/work/aivo".to_string();
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cd /Users/alice/project/work/aivo && git show f391ffd"}),
        vec![],
        None,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("run_bash(git show f391ffd)"), "{plain}");
    assert!(
        !plain.contains("cd /Users/alice/project/work/aivo"),
        "{plain}"
    );
}

#[test]
fn test_detached_results_in_batch_interleave_under_their_call() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "src/gemini_router.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "400|sanitize"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("a\nb\nc".to_string()); // read -> 3 lines
    app.apply_agent_tool_result("1:x\n2:y".to_string()); // grep -> 2 matches

    let lines = app.build_transcript().plain_lines;
    let read_call = lines
        .iter()
        .position(|l| l.contains("gemini_router.rs"))
        .unwrap();
    let grep_call = lines
        .iter()
        .position(|l| l.contains("400|sanitize"))
        .unwrap();
    assert!(
        lines[read_call + 1].contains("+3 lines"),
        "read fold under read call:\n{}",
        lines.join("\n")
    );
    assert!(
        read_call + 1 < grep_call,
        "read pair precedes grep pair:\n{}",
        lines.join("\n")
    );
    assert!(
        lines[grep_call + 1].contains("+2 matches"),
        "grep fold under grep call:\n{}",
        lines.join("\n")
    );
    assert!(
        lines[read_call].trim_start().starts_with("→"),
        "{}",
        lines.join("\n")
    );
    assert!(
        lines[grep_call].trim_start().starts_with("→"),
        "{}",
        lines.join("\n")
    );
}

#[test]
fn test_mixed_batch_with_adjacent_pair_keeps_results_under_their_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // A mixed batch with an adjacent same-tool pair (the two list_dirs) must not
    // fold it into `list_dir ×2` and strand the results in a clump — each stays
    // glued under its own call.
    for (tool, key, arg) in [
        ("glob", "pattern", "**/*"),
        ("read_file", "path", "pkg.json"),
        ("list_dir", "path", "src"),
        ("list_dir", "path", "public"),
    ] {
        app.apply_agent_tool_call(
            None,
            tool.to_string(),
            serde_json::json!({ key: arg }),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_result("a\nb\nc".to_string()); // glob -> 3 files
    app.apply_agent_tool_result("1\n2".to_string()); // read_file -> 2 lines
    app.apply_agent_tool_result("x/\ny/\nz/".to_string()); // list_dir src -> 3 entries
    app.apply_agent_tool_result("m/\nn/".to_string()); // list_dir public -> 2 entries

    let lines = app.build_transcript().plain_lines;
    let plain = lines.join("\n");
    // The stray adjacent pair never collapses in a mixed batch.
    assert!(!plain.contains("list_dir ×"), "must not coalesce:\n{plain}");
    // Every result hugs the call that produced it, in the right unit.
    let under = |call: &str, result: &str| {
        let i = lines
            .iter()
            .position(|l| l.contains(call))
            .unwrap_or_else(|| panic!("no call {call:?} in:\n{plain}"));
        assert!(
            lines[i + 1].contains(result),
            "expected {result:?} under {call:?}:\n{plain}"
        );
    };
    under("→ glob(**/*)", "3 files");
    under("→ read_file(pkg.json)", "2 lines");
    under("→ list_dir(src)", "3 entries");
    under("→ list_dir(public)", "2 entries");
}

#[test]
fn test_failed_tool_result_renders_in_error_hue() {
    // An in-process failure arrives as a single-line `error: …`; the result reads
    // in the error hue, not dim, so a timeout/denial is legible as a failure.
    let mut lines = Vec::new();
    render_tool_result(
        &mut lines,
        "error: command timed out after 120s",
        "",
        Some("run_bash"),
        None,
        false,
    );
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(ERROR()))
    );
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .all(|s| s.style.fg != Some(FAINT()))
    );

    // A normal multi-line result stays neutral even when a line says "error:".
    let mut ok = Vec::new();
    render_tool_result(
        &mut ok,
        "error: x\nmore\nlines",
        "",
        Some("grep"),
        None,
        false,
    );
    assert!(ok[0].line.spans.iter().any(|s| s.style.fg == Some(FAINT())));
    assert!(ok[0].line.spans.iter().all(|s| s.style.fg != Some(ERROR())));
}

#[test]
fn test_failed_bash_result_shows_exit_code_in_error_hue() {
    // Nonzero exit arrives as Ok(output + "[exit N]") — the summary must still read red.
    let mut lines = Vec::new();
    render_tool_result(
        &mut lines,
        "compiling…\nerror[E0308]: mismatched types\n[exit 101]",
        "",
        Some("run_bash"),
        None,
        false,
    );
    assert!(lines[0].plain.contains("exited 101"), "{}", lines[0].plain);
    assert!(
        lines[0]
            .line
            .spans
            .iter()
            .any(|s| s.style.fg == Some(ERROR())),
        "failed bash summary should carry the error hue"
    );

    // The `[exit N]` tail is found past the spill pointer, and a clean run
    // stays neutral.
    assert_eq!(
        bash_exit_code("x\ny\n[exit 2]\n[full output: /tmp/f]"),
        Some(2)
    );
    assert_eq!(bash_exit_code("all good\n[done]"), None);
    let mut ok = Vec::new();
    render_tool_result(&mut ok, "a\nb\nc", "", Some("run_bash"), None, false);
    assert!(
        ok[0].line.spans.iter().all(|s| s.style.fg != Some(ERROR())),
        "clean bash output must not read as a failure"
    );
}

#[test]
fn test_tool_result_expands_inline_via_keyboard_toggle() {
    // Folded run_bash keeps only the streamed tail; Ctrl+O (toggle_latest_output)
    // reveals the full output in place and folds back.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cargo test"}),
        vec![],
        None,
    );
    // Eight lines: folded keeps the tail, so the first lines only appear expanded.
    app.apply_agent_tool_result(
        "test 1 ... ok\ntest 2 ... ok\ntest 3 ... ok\ntest a ... ok\n\
         test b ... FAILED\ntest c ... ok\ntest d ... ok\n[exit 1]"
            .to_string(),
    );

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("▸ +8 lines"), "{plain}");
    assert!(plain.contains("exited 1"), "{plain}");
    // The streamed tail stays put, but earlier lines fold away.
    assert!(plain.contains("[exit 1]"), "tail line kept: {plain}");
    assert!(
        !plain.contains("test 1 ... ok"),
        "folded result hides its early lines: {plain}"
    );

    assert!(app.toggle_latest_output());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("test 1 ... ok"),
        "expanded result must show all its lines: {plain}"
    );
    // The summary stays the (sole) toggle for a short block — no trailing collapse.
    assert!(plain.contains("▾ 8 lines"), "{plain}");
    assert!(!plain.contains(OUTPUT_EXPANDED_PREFIX), "{plain}");

    assert!(app.toggle_latest_output());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("test 1 ... ok"), "{plain}");
}

#[test]
fn expanded_tool_result_refolds_from_its_summary_row() {
    // Regression: expanding flipped the summary to a dead "▾ N lines" row, so a
    // long read_file could only fold from the far end of the block.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "big.rs"}),
        vec![],
        None,
    );
    let content: String = (1..=100).map(|i| format!("{i}: line\n")).collect();
    app.apply_agent_tool_result(content.trim_end().to_string());
    let idx = app.history.len() - 1;

    let refresh = |app: &mut CodeTuiApp| -> Vec<String> {
        let body = app.build_transcript_history_body(80);
        let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
        app.transcript_hitbox = Some(TranscriptHitbox {
            area: Rect::new(0, 0, 80, 40),
            first_row: 0,
            rows: wrapped.rows.clone(),
        });
        wrapped.rows
    };

    let rows = refresh(&mut app);
    let marker = rows.iter().position(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(marker));
    assert!(app.expanded_output.contains(&idx));

    // A long expanded block renders two toggles, both mapping to this entry.
    let rows = refresh(&mut app);
    let markers: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| is_output_expander(r))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(markers.len(), 2, "rows:\n{}", rows.join("\n"));
    assert!(
        rows[markers[0]].contains("▾ 100 lines"),
        "{}",
        rows[markers[0]]
    );
    assert!(rows[markers[1]].contains(OUTPUT_EXPANDED_PREFIX));

    assert!(app.toggle_output_at_row(markers[0]));
    assert!(!app.expanded_output.contains(&idx));

    refresh(&mut app);
    let rows = refresh(&mut app);
    let marker = rows.iter().position(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(marker));
    let rows = refresh(&mut app);
    let bottom = rows.iter().rposition(|r| is_output_expander(r)).unwrap();
    assert!(app.toggle_output_at_row(bottom));
    assert!(!app.expanded_output.contains(&idx));
}

#[test]
fn tool_result_expander_click_maps_across_mixed_blocks() {
    // A `!cmd` fold and a tool-result fold interleave; clicking the second
    // marker row must toggle the tool result, not the shell block.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let full: String = (1..=250).map(|i| format!("L{i:04}\n")).collect();
    app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "x.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("alpha\nbeta".to_string());
    let result_idx = app.history.len() - 1;

    let body = app.build_transcript_history_body(80);
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, 80);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 80, 40),
        first_row: 0,
        rows: wrapped.rows.clone(),
    });
    let marker_rows: Vec<usize> = wrapped
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| is_output_expander(r))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(marker_rows.len(), 2, "rows:\n{}", wrapped.rows.join("\n"));

    assert!(app.toggle_output_at_row(marker_rows[1]));
    assert!(app.expanded_output.contains(&result_idx));
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("beta"), "{plain}");
}

#[test]
fn write_file_snapshot_rides_tool_call_entry() {
    // The tool_start snapshot turns the write_file card into a real diff.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "write_file".to_string(),
        serde_json::json!({"path": "a.rs", "content": "fn keep() {}\nfn renamed() {}\n"}),
        vec![Some(1)],
        Some("fn keep() {}\nfn original() {}\n".to_string()),
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("original"), "old side missing: {plain}");
    assert!(plain.contains("renamed"), "new side missing: {plain}");
    assert!(
        plain.contains(" - ") && plain.contains(" + "),
        "expected real del/ins rows, not all-additions: {plain}"
    );
}

#[test]
fn test_run_bash_label_drops_redirection_noise() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "which aivo 2>/dev/null && aivo --help 2>&1"}),
        vec![],
        None,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("which aivo && aivo --help"), "{plain}");
    assert!(!plain.contains("2>/dev/null"), "{plain}");
    assert!(!plain.contains("2>&1"), "{plain}");
}

#[test]
fn test_cursor_tool_update_enriches_call_in_place() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Cursor's start event lacks the real target → the label falls back to the
    // generic title.
    app.apply_agent_tool_call(
        Some("c1".to_string()),
        "read_file".to_string(),
        serde_json::json!({"path": "Read File"}),
        vec![],
        None,
    );
    let rev_before = app.transcript_revision;
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("read_file(Read File)"), "{plain}");

    // The follow-up update resolves the path and carries a compact result.
    app.apply_agent_tool_update(
        "c1".to_string(),
        Some(serde_json::json!({"path": "src/chat.rs"})),
        Some("42 lines".to_string()),
        false,
    );
    // The cache fingerprint must move so the edited (middle) entry re-renders.
    assert!(app.transcript_revision > rev_before);

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("read_file(src/chat.rs)"), "{plain}");
    assert!(plain.contains("⎿ 42 lines"), "{plain}");
    assert!(
        !plain.contains("Read File"),
        "stale title remained: {plain}"
    );

    // A failed update surfaces the error in place.
    app.apply_agent_tool_update(
        "c1".to_string(),
        None,
        Some("permission denied".into()),
        true,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("⎿ permission denied"), "{plain}");
}

#[tokio::test]
async fn test_cursor_turn_ending_on_tool_does_not_duplicate_prose() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    // Cursor streams prose, runs a tool, and ends with no trailing text — the
    // prose is flushed as an assistant entry before the tool steps.
    app.pending_response = "Reading and fixing.".to_string();
    app.apply_agent_tool_call(
        None,
        "edit_file".to_string(),
        serde_json::json!({"path": "a.rs"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("done".to_string());
    assert!(app.pending_response.is_empty());

    // The turn finishes. `turn.content` carries the full accumulated prose
    // (cursor reports no usage) — re-pushing it would duplicate the reply.
    app.tx
        .send(RuntimeEvent::Finished {
            result: Ok(ChatTurnResult {
                content: "Reading and fixing.".to_string(),
                usage: None,
                model: None,
                raw_body: None,
            }),
            format: ChatFormat::OpenAI,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    let roles: Vec<&str> = app.history.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "tool_call", "tool_result"]);
    assert_eq!(app.history[1].content, "Reading and fixing.");
}

#[test]
fn test_build_transcript_renders_tool_steps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"read_file","args":{"path":"src/parser.rs"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_result".to_string(),
        // Realistic read_file output: right-aligned line numbers + tab + content.
        content: "     1\t/**\n     2\t * sum\n     3\t */".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");
    // Tree-style call line with the salient arg, and a clean line-count result —
    // NOT the noisy first numbered line.
    assert!(
        plain.contains("→ read_file(src/parser.rs)"),
        "missing tool-call line in:\n{plain}"
    );
    assert!(
        plain.contains("⎿ ▸ +3 lines"),
        "missing tool-result line in:\n{plain}"
    );
    assert!(
        !plain.contains("/**"),
        "tool-result summary should not leak the noisy first line:\n{plain}"
    );

    // The call and its result hug (no blank line between them) and both carry
    // the cyan TOOL accent bar.
    let call_idx = transcript
        .plain_lines
        .iter()
        .position(|l| l.contains("→ read_file"))
        .unwrap();
    let result_idx = transcript
        .plain_lines
        .iter()
        .position(|l| l.contains("⎿ ▸ +3 lines"))
        .unwrap();
    assert_eq!(result_idx, call_idx + 1, "result should hug its call");
    assert_eq!(transcript.bar_colors[call_idx], Some(TOOL()));
    assert_eq!(transcript.bar_colors[result_idx], Some(TOOL()));
}

#[test]
fn test_agent_seed_turns_folds_tool_steps() {
    let msg = |role: &str, content: &str| ChatMessage {
        model: None,
        role: role.to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };
    let history = vec![
        msg("user", "what's in a.rs?"),
        msg("assistant", "let me look"),
        msg(
            "tool_call",
            r#"{"name":"read_file","args":{"path":"src/a.rs"}}"#,
        ),
        msg("tool_result", "     1\tfn main() {}"),
        msg("assistant", "it's an empty main"),
    ];
    let seed = super::super::runtime_impl::agent_seed_turns(&history);
    // user, assistant, [folded tool note], assistant — the tool step is preserved
    // as an assistant note instead of dropped (which caused resume amnesia).
    assert_eq!(seed.len(), 4);
    assert_eq!(seed[0], ("user".to_string(), "what's in a.rs?".to_string()));
    assert_eq!(seed[2].0, "assistant");
    assert!(
        seed[2].1.contains("read_file") && seed[2].1.contains("a.rs"),
        "tool note missing detail: {}",
        seed[2].1
    );
    assert!(
        seed[2].1.contains('→'),
        "tool note missing outcome: {}",
        seed[2].1
    );
    assert_eq!(seed[3].1, "it's an empty main");
}

#[test]
fn test_build_transcript_renders_edit_diff() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_result".to_string(),
        content: "edited src/a.rs".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines.join("\n");
    // The call line, a removed line, an added line, then the confirmation.
    assert!(
        plain.contains("→ edit_file(src/a.rs)"),
        "missing call in:\n{plain}"
    );
    assert!(
        plain.contains("- let x = 1;"),
        "missing removed line in:\n{plain}"
    );
    assert!(
        plain.contains("+ let x = 2;"),
        "missing added line in:\n{plain}"
    );
    assert!(
        plain.contains("edited src/a.rs"),
        "missing result in:\n{plain}"
    );
}

#[test]
fn test_build_transcript_prettifies_mcp_tool_name() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"mcp__filesystem__read_file","args":{}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("→ filesystem/read_file"),
        "mcp tool name not prettified in:\n{plain}"
    );
    assert!(!plain.contains("mcp__"), "raw mcp__ name leaked:\n{plain}");
}

#[test]
fn test_build_transcript_coalesces_consecutive_tool_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "study the sidebar".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Cursor-style: a run of read_file calls with no interleaved results.
    for path in ["src/sidebar.rs", "src/session.rs", "src/time.rs"] {
        app.history.push(ChatMessage {
            model: None,
            role: "tool_call".to_string(),
            content: format!(r#"{{"name":"read_file","args":{{"path":"{path}"}}}}"#),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    // A different kind right after starts a new run (not merged with the reads).
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: r#"{"name":"grep","args":{"pattern":"hover"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines.join("\n");
    // The three reads collapse to one line naming their basenames; the lone grep
    // renders on its own.
    assert!(
        plain.contains("→ read 3 files: sidebar.rs, session.rs, time.rs"),
        "missing coalesced read line in:\n{plain}"
    );
    assert!(
        plain.contains("→ grep(hover)"),
        "lone grep should render normally in:\n{plain}"
    );
    // No per-file card survived the coalescing.
    assert!(
        !plain.contains("read_file(src/sidebar.rs)"),
        "individual read cards should be coalesced away:\n{plain}"
    );
}
