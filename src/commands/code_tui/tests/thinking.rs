use super::super::*;
use super::helpers::*;

#[test]
fn test_streaming_reasoning_windows_when_answering() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.sending = true;
    // Once the answer starts, the finished thought shows the same bounded window
    // (most recent rows) above the reply — not the whole thing.
    app.pending_reasoning =
        "Inspecting the request\nSecond detail\nThird detail\nExtra detail\nFourth detail"
            .to_string();
    app.pending_response = "Working on it".to_string();

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(
        plain.contains("▸"),
        "windowed-with-more shows the ▸ expand chevron: {plain}"
    );
    assert!(
        plain.contains("Fourth detail"),
        "most recent line in the window"
    );
    assert!(
        !plain.contains("Inspecting the request"),
        "earliest line scrolls off the window"
    );
    assert!(plain.contains("Working on it"));
}

#[test]
fn test_streams_thinking_window_during_thinking_only_phase() {
    // Thinking-only gap: the reasoning streams live as a rolling window.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.sending = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_reasoning = "Working out the approach".to_string();
    assert!(app.pending_response.is_empty(), "no answer text yet");

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✻"), "streaming window marker: {plain}");
    assert!(
        plain.contains("Working out the approach"),
        "live thought is shown: {plain}"
    );
    assert!(
        !plain.contains("▸ thought"),
        "not the committed fold: {plain}"
    );
}

#[test]
fn test_thinking_window_shows_only_recent_lines() {
    // The live window keeps only the most recent THINKING_WINDOW_LINES lines;
    // older ones scroll off so it never grows unbounded.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.sending = true;
    app.pending_reasoning = "aaa\nbbb\nccc\nddd\neee".to_string();
    assert!(app.pending_response.is_empty());

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("bbb")
            && plain.contains("ccc")
            && plain.contains("ddd")
            && plain.contains("eee"),
        "recent lines visible: {plain}"
    );
    assert!(!plain.contains("aaa"), "older lines scroll off: {plain}");
}

#[test]
fn test_thinking_wrapped_lines_hang_indented_under_marker() {
    // A long thought that soft-wraps keeps its continuation rows indented under
    // the `✻` marker (pre-wrapped to the content width so the transcript wrapper
    // leaves the already-fitted rows alone).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 46;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "Here are the changes.".to_string(),
        reasoning_content: Some(
            "The user is asking which files changed — likely in the git working \
directory. I should check git status and recent git log."
                .to_string(),
        ),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let full = app.build_transcript();
    let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
    let think: Vec<&String> = wrapped
        .rows
        .iter()
        .filter(|r| r.contains("git") || r.starts_with("✻"))
        .collect();
    assert!(
        think.len() >= 2,
        "the thought wrapped to multiple rows: {think:?}"
    );
    assert!(think[0].starts_with("✻ "), "first row carries the marker");
    // Every continuation row is indented (two spaces), never flush left.
    for row in &think[1..] {
        assert!(
            row.starts_with("  ") && !row.starts_with("✻"),
            "wrapped row must hang-indent under the marker: {row:?}"
        );
    }
}

#[test]
fn test_build_transcript_hides_streaming_reasoning_when_disabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = false;
    app.sending = true;
    app.pending_reasoning = "Inspecting the request".to_string();
    app.pending_response = "Working on it".to_string();

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(!plain.contains("▸ thought"));
    assert!(!plain.contains("✻"));
    assert!(!plain.contains("Inspecting the request"));
    assert!(plain.contains("Working on it"));
}

#[test]
fn test_flush_pending_assistant_keeps_reasoning() {
    // Regression: the native agent path (flush_pending_assistant, used before each
    // tool step and at turn end) must carry pending_reasoning onto the committed
    // message, not drop it.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_response = "the answer".to_string();
    app.pending_reasoning = "the chain of thought".to_string();

    app.flush_pending_assistant();

    let last = app.history.last().expect("assistant message committed");
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "the answer");
    assert_eq!(
        last.reasoning_content.as_deref(),
        Some("the chain of thought")
    );
    assert!(app.pending_reasoning.is_empty());
}

#[test]
fn test_flush_pending_assistant_commits_reasoning_only_segment() {
    // A reasoning-only segment (model thought, then a tool call with no prose)
    // still commits, so the thinking isn't lost at the tool boundary.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_reasoning = "thinking before a tool".to_string();

    app.flush_pending_assistant();

    let last = app
        .history
        .last()
        .expect("reasoning-only message committed");
    assert_eq!(last.role, "assistant");
    assert!(last.content.is_empty());
    assert_eq!(
        last.reasoning_content.as_deref(),
        Some("thinking before a tool")
    );
}

#[test]
fn test_volatile_tail_fp_tracks_reasoning_and_toggle() {
    // Regression: the live "Thinking" block lives in the volatile tail, so its
    // fingerprint must change as reasoning streams and when thinking_enabled flips —
    // otherwise the cached tail never repaints during the reasoning-only gap.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.thinking_enabled = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let base = app.volatile_tail_fp();
    app.pending_reasoning.push_str("more thinking");
    let after_reasoning = app.volatile_tail_fp();
    assert_ne!(
        base, after_reasoning,
        "fp must change as reasoning streams (pending_response stays empty)"
    );

    app.thinking_enabled = false;
    let after_toggle = app.volatile_tail_fp();
    assert_ne!(
        after_reasoning, after_toggle,
        "fp must change when thinking_enabled flips so a /config toggle repaints"
    );
}

#[test]
fn test_thinking_block_has_distinct_bar_color() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some("the reasoning".to_string()),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);

    let t = app.build_transcript();
    let bar_for = |needle: &str| {
        t.plain_lines
            .iter()
            .position(|l| l.contains(needle))
            .and_then(|i| t.bar_colors[i])
    };
    // A committed turn shows its thinking in full, marked with `✻`, barless.
    let thinking_bar = bar_for("the reasoning");
    let answer_bar = bar_for("the answer");
    assert_eq!(
        thinking_bar, None,
        "thinking block is barless so it recedes as ephemeral meta"
    );
    assert_eq!(answer_bar, Some(ACCENT()), "answer uses the accent bar");
    assert_ne!(thinking_bar, answer_bar);
}

#[test]
fn test_history_reasoning_windows_when_thinking_enabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_width = 80;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some(
            "the gist line\nsecond thought\nthird thought\nfourth thought\nthe private chain of thought"
                .to_string(),
        ),
        attachments: vec![],
    });

    // On: a bounded window shows the most recent lines behind the `▸` expand
    // chevron; earlier lines scroll off (expand to see them).
    app.thinking_enabled = true;
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let shown = app.build_transcript().plain_lines.join("\n");
    assert!(shown.contains("▸"), "windowed-with-more shows ▸: {shown}");
    assert!(
        shown.contains("the private chain of thought"),
        "most recent line is in the window"
    );
    assert!(
        !shown.contains("the gist line"),
        "the earliest line scrolls off the window"
    );
    assert!(shown.contains("the answer"));

    // Expanded (user clicked): the chevron flips to `▾` and every line shows.
    app.expanded_thinking.insert(0);
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let expanded = app.build_transcript().plain_lines.join("\n");
    assert!(expanded.contains("▾"), "expanded shows ▾: {expanded}");
    assert!(
        expanded.contains("the gist line") && expanded.contains("the private chain of thought"),
        "expanding reveals every line: {expanded}"
    );

    app.thinking_enabled = false;
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let hidden = app.build_transcript().plain_lines.join("\n");
    assert!(!hidden.contains("✻"));
    assert!(!hidden.contains("the private chain of thought"));
    assert!(hidden.contains("the answer"));
}

#[test]
fn test_click_thinking_header_toggles_inline_expansion() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    for (content, reasoning) in [
        ("first answer", "FIRST cot"),
        ("second answer", "SECOND cot"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: content.to_string(),
            reasoning_content: Some(reasoning.to_string()),
            attachments: vec![],
        });
    }
    // Simulate the rendered rows: each assistant turn is an answer line preceded
    // by its full-thought `✻` header, in display (top→bottom) order.
    let hitbox = |rows: Vec<&str>| {
        Some(TranscriptHitbox {
            area: Rect::new(0, 0, 40, 10),
            first_row: 0,
            rows: rows.into_iter().map(str::to_string).collect(),
        })
    };
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "✻ SECOND cot",
        "second answer",
    ]);

    // Click the second thought's `✻` header → expand block 1.
    assert!(app.toggle_thinking_at_row(2));
    assert!(app.expanded_thinking.contains(&1));
    assert!(!app.expanded_thinking.contains(&0));

    // Expanded now leads with `▾`; clicking it collapses back to the window.
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "▾ SECOND cot",
        "second answer",
    ]);
    assert!(app.toggle_thinking_at_row(2));
    assert!(!app.expanded_thinking.contains(&1));

    // First header → first block (history index 0).
    app.transcript_hitbox = hitbox(vec![
        "✻ FIRST cot",
        "first answer",
        "✻ SECOND cot",
        "second answer",
    ]);
    assert!(app.toggle_thinking_at_row(0));
    assert!(app.expanded_thinking.contains(&0));

    // Clicking a non-header (answer / indented content) row does nothing.
    assert!(!app.toggle_thinking_at_row(1));

    // With thinking off, no headers render, so clicks never resolve a block.
    app.thinking_enabled = false;
    assert!(!app.toggle_thinking_at_row(0));
}

#[test]
fn test_committed_thinking_windows_and_expands_on_click() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.transcript_width = 80;
    // Short (≤ window): everything already fits, marked `✻`.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer".to_string(),
        reasoning_content: Some("line one\nline two".to_string()),
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✻"), "thinking marker present: {plain}");
    assert!(
        plain.contains("line one") && plain.contains("line two"),
        "short thought fits entirely in the window: {plain}"
    );

    // Long (> window): the window shows only the most recent rows; earliest scroll
    // off — until the user expands it.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer2".to_string(),
        reasoning_content: Some("alpha\nbeta\ngamma\ndelta\nepsilon".to_string()),
        attachments: vec![],
    });
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    // Short thought keeps `✻` (nothing hidden); the long one shows `▸` (more to
    // reveal) and is not yet expanded.
    assert!(plain.contains("✻"), "short thought keeps ✻: {plain}");
    assert!(
        plain.contains("▸"),
        "long windowed thought shows ▸: {plain}"
    );
    assert!(!plain.contains("▾"), "nothing expanded yet: {plain}");
    assert!(plain.contains("epsilon"), "most recent line shown: {plain}");
    assert!(
        !plain.contains("alpha"),
        "earliest line scrolls off: {plain}"
    );

    // Expand the long one (index 1): the marker flips to `▾` and every line shows.
    app.expanded_thinking.insert(1);
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("▾"),
        "expanded thought carries the ▾ marker: {plain}"
    );
    assert!(
        plain.contains("alpha") && plain.contains("delta"),
        "expanding reveals every line: {plain}"
    );
}

#[test]
fn test_commit_records_thinking_duration_and_resets_clock() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Simulate a segment whose thinking was frozen at 2s when the answer began.
    app.pending_reasoning = "the chain of thought".to_string();
    app.pending_response = "the answer".to_string();
    app.reasoning_elapsed_ms = Some(2000);

    app.flush_pending_assistant();

    let idx = app.history.len() - 1;
    assert_eq!(app.history[idx].role, "assistant");
    assert_eq!(app.reasoning_durations.get(&idx), Some(&2000));
    // The clock resets so the next segment times from its own first reasoning.
    assert!(app.reasoning_started_at.is_none());
    assert!(app.reasoning_elapsed_ms.is_none());
}

#[test]
fn test_composer_placeholder_stays_plain_when_history_has_reasoning() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "answer".to_string(),
        reasoning_content: Some("private reasoning".to_string()),
        attachments: vec![],
    });

    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);

    assert_eq!(plain, ">  Ask, plan, or build · / for commands");
}

#[test]
fn test_normalized_reasoning_lines_trims_and_removes_blank_runs() {
    assert_eq!(
        normalized_reasoning_lines("\nalpha\n\n\nbeta\n\n"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn test_punctuation_only_reasoning_renders_no_thought_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the answer".to_string(),
        reasoning_content: Some("...".to_string()),
        attachments: vec![],
    });
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("▸ thought"), "{plain}");
    assert!(plain.contains("the answer"));
}
