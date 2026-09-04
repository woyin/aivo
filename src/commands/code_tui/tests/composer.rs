use super::super::*;
use super::helpers::*;

#[test]
fn test_cursor_position_multiline() {
    assert_eq!(cursor_position("hello", 5, 10, 2), (7, 0));
    assert_eq!(cursor_position("hello\nworld", 11, 10, 2), (7, 1));
    assert_eq!(cursor_position("hello\nworld", 6, 10, 2), (2, 1));
    assert_eq!(cursor_position("hello\nworld", 0, 10, 2), (2, 0));
}

#[test]
fn test_truncate_path_left_uses_display_width_for_cjk() {
    // 4 CJK glyphs = 8 columns but only 4 chars; the whole path is 12 chars yet
    // 16 columns. A char-count check (≤12) would keep it unshortened and overflow
    // the 12-column budget; display width must truncate on a segment boundary.
    let out = truncate_path_left("aaa/bbb/项目目录", 12);
    assert_eq!(out, "…/项目目录");
    assert!(
        display_width(&out) <= 12,
        "overflows {} cols",
        display_width(&out)
    );
}

#[test]
fn test_cursor_position_uses_display_width_for_cjk() {
    assert_eq!(
        cursor_position("最新的软件开发工具", "最新的软件开发工具".len(), 30, 2),
        (20, 0)
    );
}

#[test]
fn test_cursor_position_wraps_after_prefix_width() {
    // width 8, prompt indent 2 → 6 text cols per row. "abcdef" fills row 0, "gh"
    // wraps to row 1. Wrapped rows carry the same 2-col hanging indent, so the
    // cursor after "gh" sits at column 2 + 2 = 4 (not column 2).
    assert_eq!(cursor_position("abcdefgh", 8, 8, 2), (4, 1));
}

#[test]
fn test_composer_visual_rows_wraps_with_hanging_indent() {
    // One word, no break opportunity → hard-wraps mid-word: "ij" spills to row 1.
    let rows = composer_visual_rows("abcdefghij", 8);
    assert_eq!(rows, vec![(0, 8), (8, 10)]);
    // A trailing newline yields a final empty row so the caret can rest there.
    assert_eq!(composer_visual_rows("ab\n", 8), vec![(0, 2), (3, 3)]);
    // An empty draft is a single empty row.
    assert_eq!(composer_visual_rows("", 8), vec![(0, 0)]);
}

#[test]
fn test_composer_visual_rows_wraps_at_word_boundary() {
    // "world" overflows → moves whole to row 1; the space stays on row 0.
    assert_eq!(
        composer_visual_rows("hello world", 8),
        vec![(0, 6), (6, 11)]
    );
    // Word longer than the row breaks at the boundary, then hard-wraps mid-word.
    assert_eq!(
        composer_visual_rows("a bcdefghij", 5),
        vec![(0, 2), (2, 7), (7, 11)]
    );
}

#[test]
fn test_composer_cursor_visual_up_down_on_wrapped_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Inner width 10 → 8 text cols. "abcdefghij" → row0 (0..8), row1 (8..10).
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();
    app.cursor = 10; // end of row 1 (display col 4)

    // Up moves to the same display column on row 0 (after 'b').
    assert!(app.composer_cursor_up());
    assert_eq!(app.cursor, 2);
    // Already on the top row → false, so the caller recalls history instead.
    assert!(!app.composer_cursor_up());
    // Down returns to row 1 at the same column.
    assert!(app.composer_cursor_down());
    assert_eq!(app.cursor, 10);
    // Bottom row → false.
    assert!(!app.composer_cursor_down());
}

#[test]
fn test_composer_wrapped_row_renders_hanging_indent() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();

    let text = app.render_composer_text();
    assert_eq!(plain_text_from_spans(&text.lines[0].spans), "❯ abcdefgh");
    // Continuation rows carry a hanging indent matching the prompt's width.
    assert_eq!(plain_text_from_spans(&text.lines[1].spans), "  ij");
}

#[test]
fn test_composer_highlights_shell_command_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 40, 5));

    // A plain message draft renders in the normal text color.
    app.draft = "hello".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(TEXT()));

    // Shell state stays on the prompt and border; long command text remains calm.
    app.draft = "!ls -la".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[0].style.fg, Some(SHELL()));
    assert_eq!(line.spans[1].style.fg, Some(TEXT()));

    // `!!` is the literal-`!` escape (sent to the model), not shell mode.
    app.draft = "!!not a command".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(TEXT()));
}

#[test]
fn test_composer_mouse_click_maps_to_cursor_offset() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();

    let click = |column: u16, row: u16| MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    };
    // Column 4 on row 0 is the 'c' cell ("❯ " = 0..1, 'a'=2, 'b'=3, 'c'=4) →
    // caret before 'c'; a click inside the prompt prefix snaps to the row start.
    assert_eq!(app.composer_offset_for_mouse(click(4, 0)), Some(2));
    assert_eq!(app.composer_offset_for_mouse(click(1, 0)), Some(0));
    // Row 1 holds "  ij"; clicking the 'j' cell (col 3) lands before it.
    assert_eq!(app.composer_offset_for_mouse(click(3, 1)), Some(9));
    // A click outside the composer misses.
    assert_eq!(app.composer_offset_for_mouse(click(40, 40)), None);
}

#[test]
fn test_insert_pasted_text_updates_draft_and_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "abc".to_string();
    app.cursor = 1;

    app.insert_pasted_text("XYZ");

    assert_eq!(app.draft, "aXYZbc");
    assert_eq!(app.cursor, 4);
}

#[test]
fn test_insert_pasted_text_normalizes_cr_to_newline() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.insert_pasted_text("one\rtwo\r\nthree\nfour");

    assert_eq!(app.draft, "one\ntwo\nthree\nfour");
    assert_eq!(app.cursor, app.draft.len());
}

#[tokio::test]
async fn paste_composes_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    app.handle_terminal_event(Event::Paste("queued prompt".into()))
        .await
        .unwrap();

    assert_eq!(app.draft, "queued prompt");
}

#[tokio::test]
async fn paste_blocked_while_resume_loading() {
    // The one state that blocks typing blocks paste too (see handle_key).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.loading_resume = Some(LoadingResume {
        request_id: 1,
        continue_plan: false,
        preview: SessionPreview {
            key_id: "k".into(),
            key_name: "k".into(),
            base_url: "u".into(),
            session_id: "s".into(),
            raw_model: "m".into(),
            updated_at: "t".into(),
            title: "t".into(),
            preview_text: "p".into(),
            origin: None,
        },
    });

    app.handle_terminal_event(Event::Paste("late".into()))
        .await
        .unwrap();

    assert_eq!(app.draft, "");
}

#[tokio::test]
async fn paste_burst_enter_inserts_newline_instead_of_submitting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft = "line1".to_string();
    app.cursor = app.draft.len();

    app.paste_burst_newline = true;
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
        .await
        .unwrap();

    assert_eq!(app.draft, "line1\nx");
    assert!(app.queued_messages.is_empty(), "must not submit per line");
    assert!(!app.paste_burst_newline, "flag is one-shot");
}

#[tokio::test]
async fn paste_burst_enter_blocked_while_resume_loading() {
    // Same gate as Event::Paste: a resume-load blocks composer input.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.loading_resume = Some(LoadingResume {
        request_id: 1,
        continue_plan: false,
        preview: SessionPreview {
            key_id: "k".into(),
            key_name: "k".into(),
            base_url: "u".into(),
            session_id: "s".into(),
            raw_model: "m".into(),
            updated_at: "t".into(),
            title: "t".into(),
            preview_text: "p".into(),
            origin: None,
        },
    });

    app.paste_burst_newline = true;
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert_eq!(app.draft, "");
}

#[test]
fn test_cursor_movement_basic() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();

    app.cursor_left();
    assert_eq!(app.cursor, 10); // before 'd'

    app.cursor_right();
    assert_eq!(app.cursor, 11); // end

    app.cursor_home();
    assert_eq!(app.cursor, 0);

    app.cursor_end();
    assert_eq!(app.cursor, 11);
}

#[test]
fn test_cursor_insert_delete() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hllo".to_string();
    app.cursor = 1; // after 'h'

    app.insert_char_at_cursor('e');
    assert_eq!(app.draft, "hello");
    assert_eq!(app.cursor, 2);

    app.cursor = app.draft.len();
    app.delete_char_before_cursor();
    assert_eq!(app.draft, "hell");
    assert_eq!(app.cursor, 4);

    app.cursor = 0;
    app.delete_char_at_cursor();
    assert_eq!(app.draft, "ell");
    assert_eq!(app.cursor, 0);
}

#[tokio::test]
async fn ctrl_d_deletes_forward_in_prompt() {
    // Ctrl+D in the composer is emacs `delete-char` (forward), not transcript
    // scroll — it must reach the editor, deleting the char under the cursor.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello".to_string();
    app.cursor = 0;
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.draft, "ello");
    assert_eq!(app.cursor, 0);
}

#[test]
fn test_cursor_word_movement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world foo".to_string();
    app.cursor = app.draft.len();

    app.cursor_word_left();
    assert_eq!(app.cursor, 12); // start of 'foo'

    app.cursor_word_left();
    assert_eq!(app.cursor, 6); // start of 'world'

    app.cursor_word_right();
    assert_eq!(app.cursor, 11); // end of 'world'
}

#[tokio::test]
async fn test_arrows_recall_history_without_swipe_scroll() {
    // Desktop: bare Up still walks draft history, transcript stays pinned.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.swipe_scroll = false;
    app.draft_history = vec!["older".to_string()];
    app.last_max_scroll = Some(50);
    app.transcript_scroll = 50;
    app.follow_output = true;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "older");
    assert_eq!(app.draft_history_index, Some(0));
    assert_eq!(app.transcript_scroll, 50);
    assert!(app.follow_output);
}

#[test]
fn test_delete_word_backward() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();
    app.delete_word_backward();
    assert_eq!(app.draft, "hello ");
    assert_eq!(app.cursor, 6);
}

#[test]
fn test_kill_to_end_of_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello\nworld".to_string();
    app.cursor = 2;
    app.kill_to_end_of_line();
    assert_eq!(app.draft, "he\nworld");
    assert_eq!(app.cursor, 2);
}

#[tokio::test]
async fn test_ctrl_l_clears_prompt_without_exiting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();
    app.draft_attachments.push(MessageAttachment {
        name: "notes.md".to_string(),
        mime_type: "text/markdown".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./notes.md".to_string(),
        },
    });

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft.is_empty());
    assert!(app.draft_attachments.is_empty());
    assert_eq!(app.cursor, 0);
}

#[tokio::test]
async fn test_ctrl_l_clears_attachment_only_prompt() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "image.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./image.png".to_string(),
        },
    });

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft_attachments.is_empty());
}

#[tokio::test]
async fn test_ctrl_l_clears_history_navigation_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["older".to_string(), "newer".to_string()];
    app.history_prev();
    assert!(app.draft_history_index.is_some());
    assert!(!app.draft.is_empty());

    app.handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert!(app.draft_history_index.is_none());
    assert!(app.draft_history_stash.is_none());
}

#[tokio::test]
async fn test_submit_slash_command_records_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/help".to_string();
    app.cursor = app.draft.len();

    let should_exit = app.submit_draft().await.unwrap();

    // The command ran (help overlay opened) and the draft cleared...
    assert!(!should_exit);
    assert!(app.draft.is_empty());
    assert!(matches!(app.overlay, Overlay::Help { .. }));
    // ...but the typed `/help` is now recallable from input history, just like
    // a normal message or `!cmd`.
    assert_eq!(app.draft_history, vec!["/help".to_string()]);
    app.history_prev();
    assert_eq!(app.draft, "/help");
}

#[tokio::test]
async fn test_record_draft_history_dedupes_consecutive() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.record_draft_history("/help");
    app.record_draft_history("/help"); // consecutive dup → ignored
    app.record_draft_history("/model");
    app.record_draft_history("/help"); // non-adjacent repeat → kept

    assert_eq!(
        app.draft_history,
        vec![
            "/help".to_string(),
            "/model".to_string(),
            "/help".to_string()
        ]
    );
}

#[tokio::test]
async fn test_ctrl_c_requires_confirmation_to_exit() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!should_exit);
    assert!(app.exit_confirm_pending);
    assert!(app.notice.is_some());

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(should_exit);
}

#[tokio::test]
async fn test_ctrl_c_pending_resets_on_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.exit_confirm_pending);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.exit_confirm_pending);
    assert!(app.notice.is_none());
}

#[tokio::test]
async fn test_ctrl_x_ctrl_e_chord_requests_external_edit() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.pending_ctrl_x);
    assert!(!app.pending_external_edit);

    app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!app.pending_ctrl_x);
    assert!(app.pending_external_edit);
}

#[tokio::test]
async fn test_ctrl_x_chord_cancelled_by_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.handle_key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(app.pending_ctrl_x);

    app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.pending_ctrl_x);
    assert!(!app.pending_external_edit);
    assert_eq!(app.draft, "h");

    app.handle_key(KeyEvent::new(KeyCode::Char('e'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(!app.pending_external_edit);
}

#[test]
fn test_composer_empty_lines_align_with_cursor_position() {
    use ratatui::buffer::Buffer;
    use ratatui::widgets::Widget;

    // Empty lines must use Line::from("") (no whitespace prefix) to avoid
    // ratatui WordWrapper producing extra visual rows for whitespace-only Lines.
    let lines = vec![
        Line::from(vec![
            Span::styled(" ", Style::default().fg(USER())),
            Span::styled("hello", Style::default().fg(TEXT())),
        ]),
        Line::from(vec![
            Span::raw(" "),
            Span::styled("hel", Style::default().fg(TEXT())),
        ]),
        Line::from(vec![
            Span::raw(" "),
            Span::styled("sadf", Style::default().fg(TEXT())),
        ]),
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::raw(" "),
            Span::styled("dsf", Style::default().fg(TEXT())),
        ]),
    ];
    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    let area = Rect::new(0, 0, 20, 10);
    let mut buf = Buffer::empty(area);
    paragraph.render(area, &mut buf);

    let cell_row = |r: u16| -> String {
        (0..20)
            .map(|x| {
                buf.cell((x, r))
                    .unwrap()
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ')
            })
            .collect()
    };

    assert!(cell_row(0).starts_with(" hello"), "row 0");
    assert!(cell_row(1).starts_with(" hel"), "row 1");
    assert!(cell_row(2).starts_with(" sadf"), "row 2");
    assert!(
        cell_row(5).starts_with(" dsf"),
        "row 5: dsf must align with cursor_position y=5"
    );

    // cursor_position must agree
    let (cx, cy) = cursor_position("hello\nhel\nsadf\n\n\ndsf", 17, 20, 1);
    assert_eq!((cx, cy), (1, 5));
}

/// The composer rule shows a live `/goal` step indicator (and the auto-approve
/// badge) while a goal loop runs, nothing goal-related when off, within width.
#[test]
fn test_composer_rule_shows_goal_step_indicator() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let off = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!off.contains("goal"), "no goal badge when off: {off:?}");
    assert!(off.contains("default"), "mode badge shows: {off:?}");

    app.goal_mode = Some(GoalState {
        objective: "ship it".to_string(),
        iteration: 2,
        max: 20,
        msg_floor: 0,
    });
    let on = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(on.contains("goal 2/20"), "goal step indicator: {on:?}");
    assert!(on.contains("default"), "mode badge stays: {on:?}");
    assert!(
        display_width(&on) <= 80,
        "rule fits width: {}",
        display_width(&on)
    );
}

/// While recalling input history with up-arrow, the composer rule titles itself
/// `History pos/total` (newest is total/total, counting down going back) and
/// drops the title once navigation ends.
#[tokio::test]
async fn test_composer_rule_shows_history_position_while_navigating() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = (0..100).map(|i| format!("/cmd {i}")).collect();

    // Idle: no history title, just the auto-approve badge.
    let idle = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!idle.contains("History"), "no title when idle: {idle:?}");

    // Up-arrow into the newest entry → `History 100/100`.
    app.history_prev();
    let newest = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(newest.contains("History 100/100"), "newest: {newest:?}");
    assert!(
        display_width(&newest) <= 80,
        "rule fits width: {}",
        display_width(&newest)
    );

    // One step further back → `History 99/100`.
    app.history_prev();
    let back = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(back.contains("History 99/100"), "back one: {back:?}");

    // Leaving navigation drops the title again.
    app.leave_history_navigation();
    let done = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!done.contains("History"), "title gone after exit: {done:?}");
}

/// The input/draft history is bounded to the most recent `MAX_DRAFT_HISTORY`
/// entries, dropping the oldest first so the on-disk recall file can't grow
/// without bound.
#[tokio::test]
async fn test_draft_history_capped_at_max() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Record 50 over the cap; each entry is distinct so none dedupe away.
    for i in 0..(MAX_DRAFT_HISTORY + 50) {
        app.record_draft_history(&format!("/cmd {i}"));
    }
    assert_eq!(app.draft_history.len(), MAX_DRAFT_HISTORY);
    // The 50 oldest were dropped; the newest is retained.
    assert_eq!(app.draft_history.first().unwrap(), "/cmd 50");
    assert_eq!(
        app.draft_history.last().unwrap(),
        &format!("/cmd {}", MAX_DRAFT_HISTORY + 49)
    );
}

#[tokio::test]
async fn test_submit_draft_keeps_oversized_mention_and_shows_notice() {
    let dir = crate::test_sandbox::tmp("aivo-mention-big");
    std::fs::write(dir.join("big.log"), vec![b'x'; 300 * 1024]).unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = dir.to_string_lossy().into_owned();
    app.draft = "why does @big.log grow?".to_string();
    app.cursor = app.draft.len();

    let should_exit = app.submit_draft().await.unwrap();

    assert!(!should_exit);
    assert_eq!(app.draft, "why does @big.log grow?");
    assert!(app.history.is_empty(), "nothing sent");
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(color, text)| *color == ERROR() && text.contains("too large to attach"))
    );
}

#[test]
fn test_mention_attachments_resolve_from_draft_text() {
    let dir = crate::test_sandbox::tmp("aivo-mention-resolve");
    std::fs::write(dir.join("notes.md"), "hi").unwrap();
    std::fs::write(dir.join("explorer"), "a file named like a profile").unwrap();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = dir.to_string_lossy().into_owned();
    app.last_subagents = crate::agent::subagents::builtin_subagents();

    let attachments = app
        .mention_attachments("read @notes.md and @notes.md, ask @agent-explorer, @explorer, @nope")
        .unwrap();
    let names: Vec<&str> = attachments.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, ["notes.md", "explorer"]);
    assert_eq!(attachments[0].mime_type, "text/markdown");
}

#[test]
fn test_empty_composer_placeholder_reserves_cursor_cell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);
    assert_eq!(plain, "❯ Ask, plan, or build · / for commands");
}

fn image_attachment(name: &str) -> MessageAttachment {
    MessageAttachment {
        name: name.to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::FileRef {
            path: format!("./{name}"),
        },
    }
}

#[test]
fn test_insert_attachment_tag_pads_with_spaces() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "look at this".to_string();
    app.cursor = app.draft.len();
    app.draft_attachments.push(image_attachment("shot.png"));
    app.insert_attachment_tag("[image #1]");
    assert_eq!(app.draft, "look at this [image #1] ");
    assert_eq!(app.cursor, app.draft.len());
}

#[test]
fn test_backspace_into_tag_removes_whole_tag_and_detaches() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "before [image #1] after".to_string();
    // Cursor right after the closing bracket.
    app.cursor = "before [image #1]".len();
    app.delete_char_before_cursor();
    assert_eq!(app.draft, "before  after");
    assert_eq!(app.cursor, "before ".len());
    assert!(app.draft_attachments.is_empty());
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Removed image: shot.png")
    );
}

#[test]
fn test_deleting_first_tag_renumbers_survivors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("one.png"));
    app.draft_attachments.push(image_attachment("two.png"));
    app.draft = "[image #1] [image #2] hi".to_string();
    app.cursor = "[image #1]".len();
    app.delete_char_before_cursor();
    assert_eq!(app.draft, " [image #1] hi");
    assert_eq!(app.draft_attachments.len(), 1);
    assert_eq!(app.draft_attachments[0].name, "two.png");
}

#[test]
fn test_typing_inside_tag_snaps_to_tag_end() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "[image #1] hi".to_string();
    app.cursor = 4; // inside the tag
    app.insert_char_at_cursor('x');
    assert_eq!(app.draft, "[image #1]x hi");
    assert_eq!(app.draft_attachments.len(), 1);
}

#[test]
fn test_word_delete_swallows_tag() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "see [image #1] now".to_string();
    // Cursor mid-tag (e.g. after word-nav); Ctrl+W must not leave a half-tag.
    app.cursor = "see [image".len();
    app.delete_word_backward();
    assert!(!app.draft.contains('['), "no tag remnant: {:?}", app.draft);
    assert!(app.draft_attachments.is_empty());
}

#[test]
fn test_kill_line_swallows_tag_and_detaches() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "keep [image #1] tail".to_string();
    app.cursor = "keep ".len();
    app.kill_to_end_of_line();
    assert_eq!(app.draft, "keep ");
    assert!(app.draft_attachments.is_empty());
}

#[test]
fn test_tagless_attachment_survives_draft_edits() {
    // A recalled-history draft has no tags; editing it must not drop staged files.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "an old prompt".to_string();
    app.cursor = app.draft.len();
    app.delete_char_before_cursor();
    app.delete_word_backward();
    assert_eq!(app.draft_attachments.len(), 1);
}

#[test]
fn test_cursor_left_hops_over_tag() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(image_attachment("shot.png"));
    app.draft = "a [image #1] b".to_string();
    app.cursor = "a [image #1]".len();
    app.cursor_left();
    assert_eq!(app.cursor, "a ".len());
}
