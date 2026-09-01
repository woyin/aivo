//! Differential tests for the settled/live streamed-reply split: at every
//! stream step the incremental tail must be byte-identical to the single-pass
//! reference (`build_transcript` → `wrap_transcript`).

use super::super::*;
use super::helpers::make_test_app;
use ratatui::Terminal;
use ratatui::backend::TestBackend;

/// Replies covering every block the settle boundary must split safely or
/// refuse to split.
fn corpus() -> Vec<&'static str> {
    vec![
        // Plain paragraphs — the common settle point.
        "First paragraph of the answer, long enough to wrap at narrow widths for sure.\n\nSecond paragraph with more detail.\n\nThird paragraph closing the thought.",
        // Fenced code with blank lines and a shorter fence-lookalike inside.
        "Intro paragraph.\n\n```rust\nfn main() {\n\n    println!(\"``` not a fence\");\n\n}\n```\n\nAfter the code block.",
        // Tilde fence plus longer-backtick fence content.
        "Look:\n\n~~~\n```\ninner backticks\n```\n~~~\n\nDone.",
        // Loose list across blank lines — must NOT settle between items.
        "Options:\n\n- first choice\n\n- second choice\n\n- third choice\n\nAfter the list.",
        // Ordered list continuing across a blank line.
        "Steps:\n\n1. do this\n\n2. then this\n\n3. finally this\n\nEnd.",
        // Nested + indented continuation — indented lines never open a boundary.
        "Plan:\n\n- outer item\n  - nested item\n\n  continuation paragraph of outer\n\nNext block.",
        // Headings, quote, thematic break, table.
        "# Title\n\nSome prose.\n\n> a quoted line\n\n> second quote\n\n---\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nTail prose.",
        // Task list and inline styles.
        "Checklist:\n\n- [x] done thing\n- [ ] pending thing\n\nThen **bold** and `code` and *italic* text to finish.",
        // CJK prose (hard-break wrapping) around a code block.
        "这是一个很长的中文段落，用来验证宽字符换行在流式渲染下的一致性表现。\n\n```\n中文代码注释\n```\n\n结束段落。",
        // Indented code block (4-space) — boundary must not land inside it.
        "Before.\n\n    indented code line one\n\n    indented code line two\n\nAfter.",
    ]
}

/// The full composed row model from the frame path (hitbox segments).
fn composed_rows(app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>) -> Vec<String> {
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    app.transcript_hitbox
        .as_ref()
        .unwrap()
        .rows()
        .map(str::to_string)
        .collect()
}

/// The single-pass reference at the same width.
fn reference_rows(app: &CodeTuiApp) -> Vec<String> {
    let full = app.build_transcript();
    wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width)
        .rows
        .to_vec()
}

fn assert_stream_matches(reply: &str, width: u16, step: usize) {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "go".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    // No turn clock on purpose: the spinner re-reads `elapsed()` per render, so
    // a second ticking over between the composed frame and the reference build
    // would diverge on "(0s)" vs "(1s)" with every row identical.
    let mut terminal = Terminal::new(TestBackend::new(width, 30)).unwrap();

    let boundaries: Vec<usize> = reply
        .char_indices()
        .map(|(i, _)| i)
        .step_by(step.max(1))
        .chain(std::iter::once(reply.len()))
        .collect();
    for &end in &boundaries {
        app.pending_response = reply[..end].to_string();
        let composed = composed_rows(&mut app, &mut terminal);
        let reference = reference_rows(&app);
        assert_eq!(
            composed,
            reference,
            "incremental tail diverged at prefix {end}/{} of reply {:?}…",
            reply.len(),
            &reply[..reply.len().min(40)]
        );
    }
}

/// Every corpus reply, streamed at several step sizes and widths, matches the
/// single-pass render at every prefix.
#[test]
fn streamed_reply_matches_single_pass_at_every_prefix() {
    for reply in corpus() {
        for &(width, step) in &[(60u16, 7usize), (60, 33), (36, 11), (100, 64)] {
            assert_stream_matches(reply, width, step);
        }
    }
}

/// Non-reply inputs (growing reasoning, notice set/cleared) reset the sections
/// without desyncing them from the reply.
#[test]
fn streamed_reply_with_reasoning_and_notice_matches() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "go".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    // No turn clock — see `assert_stream_matches`.
    app.thinking_enabled = true;
    let mut terminal = Terminal::new(TestBackend::new(60, 30)).unwrap();

    let reply = corpus()[0];
    let steps: Vec<usize> = reply
        .char_indices()
        .map(|(i, _)| i)
        .step_by(17)
        .chain(std::iter::once(reply.len()))
        .collect();
    for (n, &end) in steps.iter().enumerate() {
        app.pending_reasoning
            .push_str(&format!("thinking about part {n} of the request\n"));
        if n == 2 {
            app.notice = Some((MUTED(), "compacting context…".to_string()));
        }
        if n == 4 {
            app.notice = None;
        }
        app.pending_response = reply[..end].to_string();
        let composed = composed_rows(&mut app, &mut terminal);
        let reference = reference_rows(&app);
        assert_eq!(composed, reference, "diverged at step {n} (prefix {end})");
    }
}

/// A reply that SHRINKS (interrupt → new turn) resets the settled sections
/// instead of composing stale chunks.
#[test]
fn shrunken_reply_resets_settled_sections() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "go".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    // No turn clock — see `assert_stream_matches`.
    let mut terminal = Terminal::new(TestBackend::new(60, 30)).unwrap();

    let long = corpus()[0];
    app.pending_response = long.to_string();
    let _ = composed_rows(&mut app, &mut terminal);

    app.pending_response = "A different, shorter answer.\n\nWith two blocks.".to_string();
    let composed = composed_rows(&mut app, &mut terminal);
    assert_eq!(composed, reference_rows(&app));
}

/// The live `run_bash` tail must never shrink the transcript mid-stream —
/// the bottom-pinned view would bob as lines complete or long rows rotate out.
#[test]
fn streaming_tool_tail_height_never_decreases() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "go".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.transcript_width = 60;

    // Rows longer than the width mixed with short ones, streamed in chunks
    // that leave partials at every boundary.
    let mut script = String::new();
    for i in 0..12 {
        if i % 3 == 0 {
            script.push_str(&"x".repeat(120));
        } else {
            script.push_str(&format!("short line {i}"));
        }
        script.push('\n');
    }
    let mut prev = 0usize;
    let mut at = 0usize;
    while at < script.len() {
        let end = (at + 7).min(script.len());
        app.push_tool_output(&script[at..end]);
        at = end;
        let full = app.build_transcript();
        let rows = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width)
            .rows
            .len();
        assert!(
            rows >= prev,
            "tail block shrank ({prev} → {rows} rows) at byte {end}"
        );
        prev = rows;
    }
}
