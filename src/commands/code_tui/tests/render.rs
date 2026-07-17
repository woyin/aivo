use super::super::*;
use super::helpers::*;

#[test]
fn test_markdown_renderer_formats_code_and_lists() {
    let lines = render_markdown_lines("## Title\n\n- one\n- two\n\n```rust\nlet x = 1;\n```", 80);
    let plain = lines
        .into_iter()
        .map(|line| line.plain)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(plain.contains("Title"));
    assert!(plain.contains("• one"));
    assert!(plain.contains("rust"));
    assert!(plain.contains("let x = 1;"));
}

#[test]
fn test_markdown_loose_list_renders_tight() {
    // A "loose" list (blank lines between items in source) used to get a blank
    // line between every rendered item — double-spaced. Items should be adjacent.
    let md = "1. first item\n\n2. second item\n\n3. third item";
    let plain: Vec<String> = render_markdown_lines(md, 80)
        .into_iter()
        .map(|l| l.plain)
        .collect();
    let items: Vec<usize> = plain
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains("item"))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(items.len(), 3, "{plain:?}");
    // Consecutive item rows with no blank between them.
    assert_eq!(items[1] - items[0], 1, "blank between items 1-2: {plain:?}");
    assert_eq!(items[2] - items[1], 1, "blank between items 2-3: {plain:?}");
    // And no double-blank tail after the list.
    assert!(
        !plain.windows(2).any(|w| w[0].is_empty() && w[1].is_empty()),
        "consecutive blank lines: {plain:?}"
    );
}

#[test]
fn test_render_assistant_streaming_does_not_append_cursor_glyph() {
    let mut lines = Vec::new();
    render_assistant_message(&mut lines, None, "- item", 80);

    let plain = lines
        .into_iter()
        .map(|line| line.plain)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!plain.contains('▋'));
    assert!(plain.contains("• item"));
}

#[test]
fn test_build_transcript_shows_pending_status_without_visible_stream() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(plain.contains("Thinking"));
}

#[test]
fn test_rect_contains() {
    let area = Rect::new(10, 4, 8, 3);
    assert!(rect_contains(area, (10, 4)));
    assert!(rect_contains(area, (17, 6)));
    assert!(!rect_contains(area, (18, 6)));
    assert!(!rect_contains(area, (17, 7)));
}

#[test]
fn test_render_main_keeps_composer_near_short_empty_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut composer_area = Rect::default();

    terminal
        .draw(|frame| {
            composer_area = app.render_main(frame, frame.area());
        })
        .unwrap();

    assert!(composer_area.y + composer_area.height < 11);
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.y, 0);
    // 80 cols minus the 2-col accent gutter (no scrollbar column reserved).
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.width, 78);
    assert_eq!(app.transcript_width, 78);
}

#[test]
fn test_markdown_table_renders_aligned_columns() {
    let md = "| Name | ID |\n|------|----|\n| aivo | uz9 |\n| openrouter | 598 |\n";
    let plain: Vec<String> = render_markdown_lines(md, 80)
        .iter()
        .map(|l| l.plain.clone())
        .collect();
    let joined = plain.join("\n");

    // The table is a closed box: top/header-rule/bottom borders all present.
    assert!(
        joined.contains('┌') && joined.contains('┐'),
        "no top border:\n{joined}"
    );
    assert!(joined.contains('┼'), "no header rule:\n{joined}");
    assert!(
        joined.contains('└') && joined.contains('┘'),
        "no bottom border:\n{joined}"
    );

    // Cells are separated by a column divider — NOT concatenated into a blob.
    assert!(joined.contains('│'), "no column divider:\n{joined}");
    assert!(
        !joined.contains("aivouz9") && !joined.contains("NameID"),
        "cells got concatenated:\n{joined}"
    );
    // It fits the width, so nothing is truncated.
    assert!(joined.contains("openrouter") && joined.contains("598"));

    // Every box line is exactly the same display width (a rectangular box) and
    // its `│`/junction columns line up across rows.
    let box_lines: Vec<&String> = plain
        .iter()
        .filter(|l| l.chars().any(|c| "│┌┐└┘├┤┬┴┼".contains(c)))
        .collect();
    assert!(
        box_lines.len() >= 5,
        "expected 4 borders + ≥1 body row:\n{joined}"
    );
    let width = |s: &str| {
        s.chars()
            .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
            .sum::<usize>()
    };
    let w0 = width(box_lines[0]);
    assert!(
        box_lines.iter().all(|l| width(l) == w0),
        "box lines are not the same width: {:?}",
        box_lines.iter().map(|l| width(l)).collect::<Vec<_>>()
    );
    assert!(w0 <= 80, "box overflows the 80-col width: {w0}");
    let bar_cols = |s: &str| -> Vec<usize> {
        let mut col = 0usize;
        let mut cols = Vec::new();
        for c in s.chars() {
            if "│┌┐└┘├┤┬┴┼".contains(c) {
                cols.push(col);
            }
            col += UnicodeWidthChar::width(c).unwrap_or(0);
        }
        cols
    };
    let cols0 = bar_cols(box_lines[0]);
    assert!(
        box_lines.iter().all(|l| bar_cols(l) == cols0),
        "column separators not aligned across rows:\n{joined}"
    );
}

#[test]
fn test_markdown_table_is_responsive_to_width() {
    // A table whose natural width exceeds the pane must shrink to fit (wrapping
    // long cells) rather than overflow — so the transcript wrapper never shears it.
    let md = "| Tool | Description |\n|------|-------------|\n\
        | aivo | A unified CLI for several AI coding assistants with secure key storage |\n";
    for width in [30u16, 50, 72] {
        let lines = render_markdown_lines(md, width);
        let widths: Vec<usize> = lines
            .iter()
            .map(|l| {
                l.plain
                    .chars()
                    .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
                    .sum::<usize>()
            })
            .collect();
        assert!(
            widths.iter().all(|&w| w <= usize::from(width)),
            "table overflows width {width}: {widths:?}"
        );
        // The long description survives — just wrapped across multiple lines.
        let joined: String = lines
            .iter()
            .map(|l| l.plain.clone())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            joined.contains("unified"),
            "content dropped at width {width}"
        );
    }
}

#[test]
fn test_markdown_table_stacks_when_grid_would_break_words() {
    // Four columns with a giant token can't grid at 44 cols without breaking
    // words mid-character — the table must stack into `Header: value` blocks.
    let md = "| Dimension | Their approach | Our approach | Worth adopting? |\n\
        |---|---|---|---|\n\
        | Shape | Bun workspace monorepo: packages/{shared,core,ui,session-core} | Single Rust crate | No |\n\
        | Agent ownership | Thin agent-service | Native engine.rs | Partial |\n";
    let lines = render_markdown_lines(md, 44);
    let plain: Vec<&str> = lines.iter().map(|l| l.plain.as_str()).collect();
    let joined = plain.join("\n");

    assert!(
        !joined.chars().any(|c| "│┌┐└┘├┤┬┴┼".contains(c)),
        "expected stacked layout, got a grid:\n{joined}"
    );
    assert!(joined.contains("Dimension: Shape"), "{joined}");
    assert!(joined.contains("Their approach: Bun workspace"), "{joined}");
    let rules = plain.iter().filter(|l| l.contains("────")).count();
    assert_eq!(rules, 1, "expected one row separator:\n{joined}");
    let width = |s: &str| {
        s.chars()
            .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
            .sum::<usize>()
    };
    assert!(plain.iter().all(|l| width(l) <= 44), "overflow:\n{joined}");
    assert!(
        plain.iter().any(|l| l.contains("monorepo:")),
        "word hard-broken in stacked mode:\n{joined}"
    );
}

#[test]
fn test_markdown_table_grid_never_hard_breaks_words() {
    // The long-token column must get its longest word, not an even per-column
    // share that shears the token mid-character.
    let md = "| Dimension | Their approach | Our approach | Worth adopting? |\n\
        |---|---|---|---|\n\
        | Shape | Bun workspace monorepo: packages/{shared,core,ui,session-core} | Single Rust crate | No |\n";
    let lines = render_markdown_lines(md, 100);
    let plain: Vec<&str> = lines.iter().map(|l| l.plain.as_str()).collect();
    let joined = plain.join("\n");

    assert!(joined.contains('┌'), "expected a grid:\n{joined}");
    assert!(
        plain
            .iter()
            .any(|l| l.contains("packages/{shared,core,ui,session-core}")),
        "long token hard-broken inside grid cell:\n{joined}"
    );
}

#[test]
fn test_wrap_styled_line_word_wraps_and_hard_breaks() {
    use ratatui::text::Line;

    // Word wrap: every row fits the width and no word is split across rows.
    let line = Line::from("the quick brown fox jumps");
    let rows = wrap_styled_line(&line.spans, 10);
    for r in &rows {
        assert!(
            r.plain.chars().count() <= 10,
            "row exceeds width: {:?}",
            r.plain
        );
    }
    let words: Vec<String> = rows
        .iter()
        .flat_map(|r| r.plain.split_whitespace().map(String::from))
        .collect();
    assert_eq!(words, ["the", "quick", "brown", "fox", "jumps"]);
    assert!(rows.len() >= 3);

    // A word longer than the width hard-breaks, losing no characters.
    let long = Line::from("abcdefghij");
    let rows = wrap_styled_line(&long.spans, 4);
    let joined: String = rows.iter().map(|r| r.plain.as_str()).collect();
    assert_eq!(joined, "abcdefghij");
    for r in &rows {
        assert!(r.plain.chars().count() <= 4);
    }

    // Leading whitespace is a hanging indent: continuation rows re-apply it so
    // wrapped text (e.g. expanded reasoning) stays aligned under the first line.
    let indented = Line::from("  alpha beta gamma");
    let rows = wrap_styled_line(&indented.spans, 9);
    let plains: Vec<&str> = rows.iter().map(|r| r.plain.as_str()).collect();
    assert!(rows.len() >= 2, "should wrap: {plains:?}");
    for r in &rows {
        assert!(r.plain.starts_with("  "), "row lost indent: {:?}", r.plain);
        assert!(
            r.plain.chars().count() <= 9,
            "row exceeds width: {:?}",
            r.plain
        );
    }
}

#[test]
fn test_long_reply_scrolls_to_show_last_line() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Many words → word-wraps to far more rows than the view; ends in a sentinel.
    let mut body = (0..80)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    body.push_str(" TAILWORD");
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: body,
        reasoning_content: None,
        attachments: vec![],
    });
    assert!(app.follow_output);

    let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..10u16 {
        for x in 0..40u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // Following output must scroll to the bottom; the last word is visible. (A
    // char-wrap row count under-counts and would clip it — the original bug.)
    assert!(
        screen.contains("TAILWORD"),
        "last line clipped — did not scroll to bottom:\n{screen}"
    );
}

#[test]
fn test_transcript_cache_reuses_across_frames_until_content_changes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let draw = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
    };

    draw(&mut app, &mut terminal);
    let fp_after_first = app.transcript_cache.as_ref().unwrap().fp;
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();

    // A second frame with identical content must NOT rebuild the body: same
    // fingerprint and the same backing allocation (no re-parse / re-wrap).
    draw(&mut app, &mut terminal);
    assert_eq!(app.transcript_cache.as_ref().unwrap().fp, fp_after_first);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "identical content must reuse the cached body without rebuilding"
    );

    // Appending a message changes the fingerprint → the cache rebuilds.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "another turn".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    draw(&mut app, &mut terminal);
    assert_ne!(
        app.transcript_cache.as_ref().unwrap().fp,
        fp_after_first,
        "new content must invalidate the cache fingerprint"
    );
}

#[test]
fn test_spinner_animation_does_not_invalidate_transcript_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..60u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    };

    let screen = render_screen(&mut app, &mut terminal);
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();
    assert!(screen.contains("Thinking"), "spinner missing:\n{screen}");

    // Advancing the spinner glyph must not rebuild the body — only the appended
    // status line is volatile, so a long transcript never reparses to animate.
    app.frame_tick = app.frame_tick.wrapping_add(7);
    let screen = render_screen(&mut app, &mut terminal);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "spinner animation must reuse the cached body"
    );
    assert!(screen.contains("Thinking"), "spinner missing:\n{screen}");
}

#[test]
fn test_streaming_tokens_do_not_invalidate_history_body_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Stream".to_string();

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut CodeTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..60u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    };

    let screen = render_screen(&mut app, &mut terminal);
    let body_ptr = app.transcript_cache.as_ref().unwrap().body.lines.as_ptr();
    assert!(
        screen.contains("Stream"),
        "streamed text missing:\n{screen}"
    );

    // Appending streamed tokens must NOT rebuild the cached history body: the live
    // reply lives in the volatile tail (composed fresh each frame), so a long
    // conversation never re-parses/re-wraps its whole history per token.
    app.pending_response.push_str(" more text");
    let screen = render_screen(&mut app, &mut terminal);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "streamed tokens must reuse the cached history body (no full re-render)"
    );
    assert!(
        screen.contains("more text"),
        "new streamed text missing:\n{screen}"
    );
}

/// The split render (cached history body + volatile tail + spinner) must
/// reproduce the single-pass `build_transcript` row-for-row at the same width —
/// scroll math goes through `max_scroll` → `build_transcript`, so any divergence
/// would desync what's painted from where it scrolls. Locks the invariant the
/// streaming-perf split (keeping the live reply out of the cached body) relies on.
#[test]
fn test_streaming_composed_render_matches_full_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "explain the plan in detail please".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content:
            "# Heading\n\nA committed reply with **markdown** and a fairly long line that wraps."
                .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Mid-stream: a partial reply (the volatile tail) + a notice, plus the spinner.
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.pending_response =
        "Streaming this answer now, with another long line that should wrap across the pane."
            .to_string();
    app.notice = Some((MUTED(), "compacting context…".to_string()));

    // Render through the split path; the hitbox carries the full composed row set.
    let mut terminal = Terminal::new(TestBackend::new(40, 24)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let composed: Vec<String> = app.transcript_hitbox.as_ref().unwrap().rows.clone();

    // Reference: the single-pass transcript wrapped to the same width.
    let full = app.build_transcript();
    let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
    assert_eq!(
        composed, wrapped.rows,
        "split render diverged from the single-pass build_transcript"
    );
}

/// The volatile-tail cache must invalidate when the streamed reply grows, a
/// notice changes, or the reply clears at turn end. Render across each state and
/// confirm the composed render still matches the single-pass `build_transcript`,
/// so no stale render survives a content change at the same width.
#[test]
fn streaming_reply_cache_invalidates_on_change() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

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

    let long = "Short reply that has now grown a good deal longer and wraps across the pane.";
    let states: [(&str, Option<&str>); 4] = [
        ("Short", None),
        (long, None),                // grew: cache must re-render
        (long, Some("compacting…")), // notice added at same reply len
        ("", None),                  // reply cleared at turn end: tail collapses
    ];

    let mut terminal = Terminal::new(TestBackend::new(40, 24)).unwrap();
    for (reply, notice) in states {
        app.pending_response = reply.to_string();
        app.notice = notice.map(|t| (MUTED(), t.to_string()));
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let composed: Vec<String> = app.transcript_hitbox.as_ref().unwrap().rows.clone();
        let full = app.build_transcript();
        let wrapped = wrap_transcript(&full.lines, &full.bar_colors, app.transcript_width);
        assert_eq!(
            composed, wrapped.rows,
            "cached render diverged from build_transcript at reply {reply:?} notice {notice:?}"
        );
    }
}

#[test]
fn test_wrap_transcript_carries_bar_color_per_row() {
    use ratatui::text::Line;
    let lines = vec![StyledLine {
        line: Line::from("alpha beta gamma delta"),
        plain: "alpha beta gamma delta".to_string(),
    }];
    let wrapped = wrap_transcript(&lines, &[Some(TOOL())], 8);
    assert!(wrapped.rows.len() >= 3);
    // Every wrapped row inherits the source line's bar color.
    assert!(wrapped.bars.iter().all(|b| *b == Some(TOOL())));
    assert_eq!(wrapped.rows.len(), wrapped.bars.len());
}

#[test]
fn test_wrap_transcript_fills_background_to_full_width() {
    use ratatui::text::Line;
    // A line that ends in a background-colored span (an inline-diff row) is padded
    // so the tint fills the whole row width and a wrap reads as one block.
    let diff = vec![StyledLine {
        line: Line::from(vec![
            Span::styled("  ".to_string(), Style::default()),
            Span::styled(" + ".to_string(), Style::default().bg(DIFF_ADD_BG())),
            Span::styled(
                "let very long added line".to_string(),
                Style::default().bg(DIFF_ADD_BG()),
            ),
        ]),
        plain: "   + let very long added line".to_string(),
    }];
    let wrapped = wrap_transcript(&diff, &[None], 12);
    assert!(wrapped.rows.len() >= 2, "long diff line should wrap");
    for row in &wrapped.rows {
        assert_eq!(
            row_display_width(row),
            12,
            "every wrapped diff row should be padded to full width: {row:?}"
        );
    }
    // Each visual row still ends in the add tint, so the block stays contiguous.
    for line in &wrapped.text.lines {
        assert_eq!(
            line.spans.last().and_then(|s| s.style.bg),
            Some(DIFF_ADD_BG())
        );
    }

    // A plain line (no trailing background) is never padded — only opted-in tinted
    // rows gain a background.
    let plain = vec![StyledLine {
        line: Line::from(Span::styled("hi".to_string(), Style::default().fg(TEXT()))),
        plain: "hi".to_string(),
    }];
    let wrapped = wrap_transcript(&plain, &[None], 12);
    assert_eq!(wrapped.rows[0], "hi");
}

#[test]
fn test_typewriter_reveals_buffer_progressively() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 34 chars: the floor (30) dominates the 1/2 catch-up (17), revealing 30 on
    // the first frame.
    let full = "字".repeat(34);
    app.incoming_buffer = full.clone();
    assert!(app.tick_typewriter());
    assert_eq!(app.pending_response, "字".repeat(30)); // exactly 30 chars, no split glyph
    assert_eq!(app.incoming_buffer.chars().count(), 4);

    // Drains fully over the next frames, never losing or splitting a char.
    let mut guard = 0;
    while app.tick_typewriter() {
        guard += 1;
        assert!(guard < 100, "typewriter should converge");
    }
    assert_eq!(app.pending_response, full);
    assert!(app.incoming_buffer.is_empty());
}

#[tokio::test]
async fn test_finish_deferred_until_typewriter_drains() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // A chunk arrived but hasn't been revealed yet.
    app.incoming_buffer = "the full reply".to_string();

    // Finished lands while text is still buffered → it must be held back.
    app.tx
        .send(RuntimeEvent::Finished {
            result: Ok(ChatTurnResult {
                content: "the full reply".to_string(),
                usage: None,
                model: None,
                raw_body: None,
            }),
            format: ChatFormat::OpenAI,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.pending_finish.is_some(), "finish should be deferred");
    assert!(!app.history.iter().any(|m| m.role == "assistant"));

    // Drive the reveal + deferred finish to completion, as the run loop does.
    let mut guard = 0;
    loop {
        app.tick_typewriter();
        if app.run_deferred_finish_if_ready().await.unwrap() {
            break;
        }
        guard += 1;
        assert!(guard < 100, "deferred finish should fire once drained");
    }
    assert!(app.pending_finish.is_none());
    assert!(app.incoming_buffer.is_empty());
    assert_eq!(
        app.history
            .last()
            .map(|m| (m.role.as_str(), m.content.as_str())),
        Some(("assistant", "the full reply")),
    );
}

#[test]
fn test_render_main_paints_per_role_accent_gutter() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "ping".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Long enough to wrap across several rows at width 24.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "alpha beta gamma delta epsilon zeta eta theta iota kappa".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let backend = TestBackend::new(24, 16);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();

    // Collect the gutter column (x = 0): which rows carry a "▌" and in what color.
    let mut user_bar_rows = 0;
    let mut assistant_bar_rows = 0;
    for y in 0..16u16 {
        let cell = &buf[(0, y)];
        if cell.symbol() != "▌" {
            continue;
        }
        if cell.fg == USER() {
            user_bar_rows += 1;
        } else if cell.fg == ACCENT() {
            assistant_bar_rows += 1;
        }
    }

    // User block is one row; assistant block wraps onto multiple rows and the
    // accent bar must repeat on every wrapped continuation row, not just the
    // first. Content is also inset past the gutter (no "▌" at the text origin).
    assert_eq!(
        user_bar_rows, 1,
        "user block should show one blue accent bar"
    );
    assert!(
        assistant_bar_rows >= 2,
        "assistant bar must repeat on wrapped rows, got {assistant_bar_rows}"
    );
    assert_ne!(
        buf[(ACCENT_GUTTER_WIDTH, 0)].symbol(),
        "▌",
        "transcript text must render past the gutter"
    );
}

#[test]
fn test_render_main_uses_full_height_for_long_transcript() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let backend = TestBackend::new(80, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    let mut composer_area = Rect::default();

    terminal
        .draw(|frame| {
            composer_area = app.render_main(frame, frame.area());
        })
        .unwrap();

    // Composer bottom = height (12) minus the footer's single row.
    assert_eq!(composer_area.y + composer_area.height, 11);
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.y, 0);
    // 80 cols minus the 2-col accent gutter; the overflow transcript keeps full
    // width now that no scrollbar column is reserved.
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.width, 78);
    assert_eq!(app.transcript_width, 78);
}
