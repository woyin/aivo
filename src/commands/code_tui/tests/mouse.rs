use super::super::*;
use super::helpers::*;

#[test]
fn test_chat_mouse_enabled_policy() {
    // No override: on everywhere except Termux, where taps must keep toggling
    // the soft keyboard instead of being eaten as mouse events.
    assert!(chat_mouse_enabled_for(None, false));
    assert!(!chat_mouse_enabled_for(None, true));
    // Explicit override wins in both directions, even under Termux.
    assert!(!chat_mouse_enabled_for(Some("1"), false));
    assert!(!chat_mouse_enabled_for(Some("yes"), true));
    assert!(chat_mouse_enabled_for(Some("0"), true));
    assert!(chat_mouse_enabled_for(Some("false"), true));
}

#[test]
fn test_chat_swipe_scroll_policy() {
    // No override: on under Termux (swipes arrive as arrows there), off elsewhere.
    assert!(!chat_swipe_scroll_enabled_for(None, false));
    assert!(chat_swipe_scroll_enabled_for(None, true));
    // Explicit override wins either way.
    assert!(chat_swipe_scroll_enabled_for(Some("1"), false));
    assert!(chat_swipe_scroll_enabled_for(Some("yes"), false));
    assert!(!chat_swipe_scroll_enabled_for(Some("0"), true));
    assert!(!chat_swipe_scroll_enabled_for(Some("false"), true));
}

#[tokio::test]
async fn test_streamed_steps_keep_scroll_when_user_scrolled_up() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // The user scrolled up to read earlier output (follow disengaged).
    app.follow_output = false;

    app.apply_agent_tool_call(
        Some("1".to_string()),
        "read_file".to_string(),
        serde_json::json!({"path": "x"}),
        vec![],
        None,
    );
    assert!(
        !app.follow_output,
        "a tool call must not snap the view to the bottom"
    );
    app.apply_agent_tool_result("ok".to_string());
    assert!(
        !app.follow_output,
        "a tool result must not snap the view to the bottom"
    );
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "pending"}]));
    assert!(
        !app.follow_output,
        "a plan update must not snap the view to the bottom"
    );

    // While following (at the bottom), streamed steps keep following.
    app.follow_output = true;
    app.apply_agent_tool_result("more".to_string());
    assert!(app.follow_output);
}

#[tokio::test]
async fn test_swipe_scroll_arrows_scroll_transcript() {
    // Mobile: a swipe arrives as Up/Down; with an empty composer they scroll.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = true;
    app.draft_history = vec!["older".to_string()];
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.transcript_scroll < 50);
    assert!(!app.follow_output);
    assert!(app.draft.is_empty());
    assert!(app.draft_history_index.is_none());

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);
    assert!(app.draft.is_empty());
}

#[tokio::test]
async fn test_swipe_scroll_yields_to_multiline_cursor() {
    // Swipe-scroll still yields to caret movement in a multi-line draft; it only
    // scrolls at the top edge.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = true;
    app.draft = "line one\nline two".to_string();
    app.cursor = app.draft.len();
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.cursor < app.draft.len());
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.transcript_scroll < 50);
}

#[tokio::test]
async fn test_mouse_wheel_scrolls_only_inside_transcript_hitbox() {
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
    app.transcript_width = 80;
    app.transcript_view_height = 6;
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 80, 6),
        first_row: 0,
        rows: wrap_plain_lines(&app.build_transcript().plain_lines, 80),
    });
    app.follow_output = false;
    app.scroll_speed = 4;

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 10,
        row: 8,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert_eq!(app.transcript_scroll, 0);

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::ScrollDown,
        column: 10,
        row: 2,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert_eq!(app.transcript_scroll, 4);
}

#[test]
fn test_chat_scroll_speed_clamps_to_safe_range() {
    assert_eq!(DEFAULT_CHAT_SCROLL_SPEED.clamp(1, MAX_CHAT_SCROLL_SPEED), 3);
    assert_eq!(0usize.clamp(1, MAX_CHAT_SCROLL_SPEED), 1);
    assert_eq!(
        999usize.clamp(1, MAX_CHAT_SCROLL_SPEED),
        MAX_CHAT_SCROLL_SPEED
    );
}

#[test]
fn test_selected_text_normalizes_drag_direction_and_preserves_lines() {
    let rows = vec![
        "alpha beta".to_string(),
        "second line".to_string(),
        "third".to_string(),
    ];
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 2, column: 2 },
        focus: TranscriptPoint { row: 0, column: 6 },
    };

    assert_eq!(
        selected_text_from_rows(&rows, selection).unwrap(),
        "beta\nsecond line\nth"
    );
}

#[test]
fn test_zero_length_selection_is_not_rendered_as_selection() {
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 1, column: 4 },
        focus: TranscriptPoint { row: 1, column: 4 },
    };

    assert!(selection.is_empty());
    assert!(selected_text_from_rows(&["alpha".to_string()], selection).is_none());
}

#[test]
fn test_selection_highlight_preserves_rendered_text_and_foreground() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn rendered_cells(app: &mut CodeTuiApp) -> Vec<(String, Color)> {
        let backend = TestBackend::new(48, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let area = buffer.area;
        let mut cells = Vec::new();
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width {
                let cell = buffer.cell((x, y)).unwrap();
                cells.push((cell.symbol().to_string(), cell.fg));
            }
        }
        cells
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut normal = make_test_app(tx, rx);
    normal.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "A **styled** answer with enough words to wrap across multiple visual lines."
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut selected = make_test_app(tx, rx);
    selected.history = normal.history.clone();
    selected.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 3, column: 2 },
        focus: TranscriptPoint { row: 4, column: 10 },
    });

    assert_eq!(rendered_cells(&mut normal), rendered_cells(&mut selected));
}

#[test]
fn test_copy_toast_expires_without_touching_transcript_notice() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.notice = None;
    app.toast = Some(Toast {
        text: "Copied selection".to_string(),
        created_at: Instant::now() - TOAST_DURATION,
        expires_at: Instant::now() - Duration::from_millis(1),
    });
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|frame| app.render_toast(frame, frame.area()))
        .unwrap();

    assert!(app.toast.is_none());
    assert!(app.notice.is_none());
}

#[test]
fn test_copy_toast_anchors_bottom_right() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.notice = None;
    // Composer text top sits near the bottom of a 12-row frame; the toast should
    // float on the last transcript row (composer top - 2), not the top edge.
    app.composer_text_area = Some(Rect::new(0, 10, 40, 2));
    app.toast = Some(Toast {
        text: "Copied 5 chars".to_string(),
        created_at: Instant::now(),
        expires_at: Instant::now() + TOAST_DURATION,
    });
    let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
    terminal
        .draw(|frame| app.render_toast(frame, frame.area()))
        .unwrap();
    let buf = terminal.backend().buffer();

    let row_text = |y: u16| -> String {
        (0..40)
            .map(|x| buf.cell((x, y)).unwrap().symbol().to_owned())
            .collect()
    };
    // Anchored on the last transcript row (composer top 10 - 2 = row 8).
    assert!(
        row_text(8).contains("Copied 5 chars"),
        "toast should anchor bottom-right: {:?}",
        row_text(8)
    );
    // The top row stays clear — the old top-right placement is gone.
    assert!(
        !row_text(0).contains("Copied"),
        "toast must not render at the top: {:?}",
        row_text(0)
    );
}

#[tokio::test]
async fn test_mouse_drag_coordinates_map_to_transcript_rows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A populated transcript hitbox only exists with real content; an empty
    // transcript routes selection to the screen surface (see `selection_target`).
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "x".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(4, 2, 20, 4),
        first_row: 10,
        rows: vec!["a".to_string(); 20],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 7,
        row: 3,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 12,
        row: 5,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 11, column: 3 },
            focus: TranscriptPoint { row: 13, column: 8 },
        })
    );
}

#[tokio::test]
async fn test_screen_drag_selects_off_transcript_from_screen_capture() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A press below the transcript (composer/footer band) selects the screen capture.
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 1),
        first_row: 0,
        rows: vec!["transcript".to_string()],
    });
    app.screen_surface = Some(ScreenSurface {
        area: Rect::new(0, 0, 20, 4),
        rows: vec![
            "transcript".to_string(),
            "alpha beta gamma".to_string(),
            "second line here".to_string(),
            String::new(),
        ],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 6,
        row: 1,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 6,
        row: 2,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert!(app.transcript_selection.is_none());
    assert_eq!(
        app.screen_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 1, column: 6 },
            focus: TranscriptPoint { row: 2, column: 6 },
        })
    );
    assert_eq!(
        app.selected_screen_text().as_deref(),
        Some("beta gamma\nsecond")
    );
}

#[tokio::test]
async fn test_open_overlay_left_drag_selects_overlay_text() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // An open overlay covers the transcript: a left drag selects the overlay text.
    app.overlay = Overlay::Help { scroll: 0 };
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 40, 10),
        first_row: 0,
        rows: vec!["hidden transcript".to_string(); 10],
    });
    app.screen_surface = Some(ScreenSurface {
        area: Rect::new(0, 0, 40, 4),
        rows: vec![
            "  command output line one".to_string(),
            "  command output line two".to_string(),
            String::new(),
            String::new(),
        ],
    });

    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 2,
        row: 0,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 25,
        row: 0,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();

    assert!(app.transcript_selection.is_none());
    assert_eq!(
        app.selected_screen_text().as_deref(),
        Some("command output line one")
    );
}

/// The jump-to-bottom pill appears only while scrolled up (content below the
/// viewport), and clicking it pins back to the latest output.
#[tokio::test]
async fn jump_to_bottom_pill_shows_when_scrolled_up_and_clicks_to_latest() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Enough turns to overflow an 80x24 viewport, so the jump-to-bottom pill applies.
    for i in 0..40 {
        app.history.push(ChatMessage {
            model: None,
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("message line {i}"),
            reasoning_content: None,
            attachments: vec![],
        });
    }

    let render = |app: &mut CodeTuiApp| {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
    };

    // Pinned to the bottom → no pill.
    app.follow_output = true;
    render(&mut app);
    assert!(
        app.jump_to_bottom_hit.is_none(),
        "pill hidden while following the bottom"
    );

    // Scroll to the top → content below the viewport, pill appears bottom-right.
    app.follow_output = false;
    app.transcript_scroll = 0;
    render(&mut app);
    let hit = app
        .jump_to_bottom_hit
        .expect("pill shown while scrolled up");

    // Click it → pins back to the latest output.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: hit.x,
        row: hit.y,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert!(
        app.follow_output,
        "clicking the pill jumps to the latest output"
    );

    // Following again → the pill is gone.
    render(&mut app);
    assert!(
        app.jump_to_bottom_hit.is_none(),
        "pill hidden after jumping to the bottom"
    );
}

#[test]
fn test_open_modal_confines_screen_selection_to_its_content() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Help { scroll: 0 };

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();

    // A modal is open → the selection surface is its inner content rect, not the
    // whole screen, so a drag/line-select can't grab the full terminal line.
    let region = app
        .screen_region
        .expect("an open modal sets a selection region");
    assert!(
        region.width < 100,
        "region must be narrower than the screen"
    );
    assert!(region.x > 0, "region starts inside the modal border");
    let surface = app.screen_surface.as_ref().unwrap();
    assert_eq!(surface.area, region);
    assert!(
        surface
            .rows
            .iter()
            .all(|r| row_display_width(r) <= region.width),
        "captured rows must not exceed the modal width"
    );
    assert!(
        surface.rows.iter().any(|r| r.contains("Slash commands")),
        "modal content should be captured: {:?}",
        surface.rows
    );
}

#[test]
fn test_word_bounds_at_grabs_contiguous_word() {
    // "alpha beta gamma" → clicking inside "beta" (cols 6..10) selects 6..10.
    let row = "alpha beta gamma";
    assert_eq!(word_bounds_at(row, 7), Some((6, 10)));
    // Click at the first char of "gamma" (col 11) → 11..16.
    assert_eq!(word_bounds_at(row, 11), Some((11, 16)));
    // Click on the space between words → no word.
    assert_eq!(word_bounds_at(row, 5), None);
    // Click past the end of the text → no word.
    assert_eq!(word_bounds_at(row, 40), None);
}

#[test]
fn test_word_bounds_at_handles_wide_chars() {
    // Each CJK char is 2 display columns: "你好 abc" → 你=0..2, 好=2..4, space=4..5.
    let row = "你好 abc";
    assert_eq!(word_bounds_at(row, 0), Some((0, 4)));
    assert_eq!(word_bounds_at(row, 3), Some((0, 4)));
    assert_eq!(word_bounds_at(row, 4), None); // the space
    assert_eq!(word_bounds_at(row, 5), Some((5, 8))); // "abc"
}

#[test]
fn test_row_display_width_counts_wide_chars() {
    assert_eq!(row_display_width("abc"), 3);
    assert_eq!(row_display_width("你好"), 4);
    assert_eq!(row_display_width(""), 0);
}

#[test]
fn test_selected_text_trims_trailing_wrap_padding() {
    let rows = vec![
        "alpha   ".to_string(),
        "beta".to_string(),
        "gamma  ".to_string(),
    ];
    let selection = TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 2, column: 7 },
    };
    assert_eq!(
        selected_text_from_rows(&rows, selection).unwrap(),
        "alpha\nbeta\ngamma"
    );
}

#[test]
fn test_register_click_counts_double_and_triple() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let point = TranscriptPoint { row: 2, column: 4 };
    assert_eq!(app.register_click(point), 1);
    assert_eq!(app.register_click(point), 2);
    assert_eq!(app.register_click(point), 3);
    // Caps at 3.
    assert_eq!(app.register_click(point), 3);
    // A click on a different row resets the run.
    assert_eq!(app.register_click(TranscriptPoint { row: 9, column: 4 }), 1);
}

#[test]
fn test_register_click_resets_after_interval() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let point = TranscriptPoint { row: 1, column: 1 };
    app.last_click = Some(ClickTracker {
        at: Instant::now() - MULTI_CLICK_INTERVAL - Duration::from_millis(10),
        point,
        count: 1,
    });
    // Too slow to chain → counts as a fresh first click.
    assert_eq!(app.register_click(point), 1);
}

#[test]
fn test_select_word_and_line_set_expected_bounds() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 3),
        first_row: 0,
        rows: vec!["alpha beta".to_string(), "x".to_string()],
    });

    assert!(app.select_word_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 7 }
    ));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 6 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    assert!(app.select_line_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 2 }
    ));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 0 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    // A click on whitespace produces no word selection.
    app.transcript_selection = None;
    assert!(!app.select_word_on(
        SelectionSurface::Transcript,
        TranscriptPoint { row: 0, column: 5 }
    ));
    assert!(app.transcript_selection.is_none());
}

#[tokio::test]
async fn test_drag_to_bottom_edge_arms_and_advances_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Real content so max_scroll() (which rebuilds from history) leaves room to
    // scroll past the 4-row viewport.
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_width = 20;
    app.transcript_view_height = 4;
    app.transcript_scroll = 0;
    app.follow_output = false;
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 0,
        rows: vec!["row".to_string(); 60],
    });
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 0 },
    });
    app.transcript_drag_active = true;

    // Drag below the bottom edge (row 4 == area.y + height) arms downward scroll.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 5,
        row: 4,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(
        app.drag_autoscroll,
        Some(DragAutoscroll { dir: 1, column: 5 })
    );

    // First tick scrolls one line and re-anchors the focus to the exposed row.
    assert!(app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 1);
    let focus = app.transcript_selection.unwrap().focus;
    assert_eq!(focus.column, 5);
    assert_eq!(focus.row, app.transcript_scroll + 3); // bottom visible row

    // An immediate second tick is throttled (no time has passed).
    assert!(!app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 1);
}

#[tokio::test]
async fn test_drag_to_top_edge_arms_and_advances_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..40)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    app.transcript_width = 20;
    app.transcript_view_height = 4;
    app.transcript_scroll = 5;
    app.follow_output = false;
    // The transcript is flush with the top of the screen (area.y == 0), so the
    // pointer can never sit *above* it — scroll-up must arm on the top row.
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 5,
        rows: vec!["row".to_string(); 60],
    });
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 8, column: 0 },
        focus: TranscriptPoint { row: 8, column: 0 },
    });
    app.transcript_drag_active = true;

    // Drag onto the top edge row (row 0 == area.y) arms upward scroll.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 3,
        row: 0,
        modifiers: KeyModifiers::NONE,
    });
    assert_eq!(
        app.drag_autoscroll,
        Some(DragAutoscroll { dir: -1, column: 3 })
    );

    // First tick scrolls up one line and re-anchors the focus to the top row.
    assert!(app.tick_drag_autoscroll());
    assert_eq!(app.transcript_scroll, 4);
    let focus = app.transcript_selection.unwrap().focus;
    assert_eq!(focus.column, 3);
    assert_eq!(focus.row, app.transcript_scroll); // top visible row
}

#[tokio::test]
async fn test_drag_inside_viewport_disarms_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_hitbox = Some(TranscriptHitbox {
        area: Rect::new(0, 0, 20, 4),
        first_row: 0,
        rows: vec!["row".to_string(); 40],
    });
    app.drag_autoscroll = Some(DragAutoscroll { dir: 1, column: 2 });

    // A drag back inside the viewport clears the arming.
    app.update_drag_autoscroll(MouseEvent {
        kind: MouseEventKind::Drag(MouseButton::Left),
        column: 5,
        row: 2,
        modifiers: KeyModifiers::NONE,
    });
    assert!(app.drag_autoscroll.is_none());
}

#[test]
fn test_selection_flash_auto_clears_after_window() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 4 },
    });

    // Flash still active → selection retained.
    app.selection_flash_until = Some(Instant::now() + Duration::from_secs(5));
    app.tick_selection_flash();
    assert!(app.transcript_selection.is_some());
    assert!(app.selection_flash_until.is_some());

    // Flash elapsed → selection auto-clears.
    app.selection_flash_until = Some(Instant::now() - Duration::from_millis(1));
    app.tick_selection_flash();
    assert!(app.transcript_selection.is_none());
    assert!(app.selection_flash_until.is_none());
}

#[test]
fn test_highlight_does_not_wash_blank_cells_past_text() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Select well past the two-character "hi" line.
    app.transcript_selection = Some(TranscriptSelection {
        anchor: TranscriptPoint { row: 0, column: 0 },
        focus: TranscriptPoint { row: 0, column: 30 },
    });

    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();

    let buffer = terminal.backend().buffer();
    // The wash must stop at each row's text: a blank cell *past* the last washed
    // character is the bug. Interior spaces (e.g. the gap in "AIVO Chat") are
    // legitimately washed and must not trip this.
    let area = buffer.area;
    let mut trailing_wash = false;
    for y in area.y..area.y + area.height {
        let mut last_text_col: Option<u16> = None;
        let mut washed_cols = Vec::new();
        for x in area.x..area.x + area.width {
            let cell = buffer.cell((x, y)).unwrap();
            if cell.bg == SELECT_WASH() {
                washed_cols.push(x);
                if !cell.symbol().trim().is_empty() {
                    last_text_col = Some(x);
                }
            }
        }
        let past_text = match last_text_col {
            Some(last) => washed_cols.iter().any(|&x| x > last),
            None => !washed_cols.is_empty(), // washed cells, none textual → all trailing
        };
        trailing_wash |= past_text;
    }
    assert!(
        !trailing_wash,
        "selection wash should not cover blank cells past the line's text"
    );
}

#[test]
fn test_clipboard_command_candidates_are_platform_specific() {
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Macos)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["pbcopy"]
    );
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Linux)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["wl-copy", "xclip", "xsel"]
    );
    assert_eq!(
        clipboard_command_candidates(ClipboardOs::Windows)[0].program,
        "powershell.exe"
    );
    assert!(clipboard_command_candidates(ClipboardOs::Other).is_empty());
}

#[test]
fn test_clipboard_read_candidates_are_platform_specific() {
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Macos)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["pbpaste"]
    );
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Linux)
            .into_iter()
            .map(|command| command.program)
            .collect::<Vec<_>>(),
        vec!["wl-paste", "xclip", "xsel"]
    );
    assert_eq!(
        clipboard_read_candidates(ClipboardOs::Windows)[0].program,
        "powershell.exe"
    );
    assert!(clipboard_read_candidates(ClipboardOs::Other).is_empty());
}

#[test]
fn test_empty_state_notice_selects_via_screen_surface() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // `--share` launch state: empty transcript, share-URL notice (the notice draws
    // the URL line; the handle just drives the badge).
    assert!(app.is_transcript_empty());
    app.share.handle = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/s/uniqueurlzz",
    ));
    app.notice = Some((
        LIVE(),
        format!("{LIVE_NOTICE_PREFIX}https://s.getaivo.dev/s/uniqueurlzz"),
    ));
    let (_, rows) = render_full_screen(&mut app, 80, 16);
    let url_row = rows
        .iter()
        .position(|r| r.contains("uniqueurlzz"))
        .expect("live URL rendered in the empty state") as u16;

    // The press must target the screen surface, not the empty transcript hitbox.
    let mouse = MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 12,
        row: url_row,
        modifiers: KeyModifiers::NONE,
    };
    assert!(
        matches!(
            app.selection_target(mouse, false),
            Some((SelectionSurface::Screen, _))
        ),
        "empty-state notice should select via the screen surface"
    );
}
