use super::*;
use chrono::Duration as ChronoDuration;
use tempfile::TempDir;

#[test]
fn test_matches_fuzzy() {
    assert!(matches_fuzzy("g4", "gpt-4o"));
    assert!(matches_fuzzy("", "anything"));
    assert!(!matches_fuzzy("xyz", "gpt-4o"));
}

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
fn test_composer_cursor_position_offsets_attachment_rows() {
    let (x, y) = cursor_position("hello", 5, 20, 2);
    assert_eq!((x, y.saturating_add(1)), (7, 1));
    let (x, y) = cursor_position("", 0, 20, 2);
    assert_eq!((x, y.saturating_add(2)), (2, 2));
}

#[test]
fn test_composer_visual_rows_wraps_with_hanging_indent() {
    // 8 text cols per row; "abcdefghij" fills row 0 and wraps "ij" to row 1.
    let rows = composer_visual_rows("abcdefghij", 8);
    assert_eq!(rows, vec![(0, 8), (8, 10)]);
    // A trailing newline yields a final empty row so the caret can rest there.
    assert_eq!(composer_visual_rows("ab\n", 8), vec![(0, 2), (3, 3)]);
    // An empty draft is a single empty row.
    assert_eq!(composer_visual_rows("", 8), vec![(0, 0)]);
}

#[test]
fn test_composer_cursor_visual_up_down_on_wrapped_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // width 10 → 8 text cols. "abcdefghij" → row0 (0..8), row1 (8..10).
    app.composer_text_area = Some(Rect::new(0, 0, 10, 5));
    app.draft = "abcdefghij".to_string();
    app.cursor = 9; // row 1, one char in (display col 3)

    // Up moves to the same display column on row 0 (after 'a').
    assert!(app.composer_cursor_up());
    assert_eq!(app.cursor, 1);
    // Already on the top row → false, so the caller recalls history instead.
    assert!(!app.composer_cursor_up());
    // Down returns to row 1 at the same column.
    assert!(app.composer_cursor_down());
    assert_eq!(app.cursor, 9);
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
    assert_eq!(plain_text_from_spans(&text.lines[0].spans), "> abcdefgh");
    // Wrapped continuation aligns under the text with a 2-col indent.
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
    assert_eq!(line.spans[1].style.fg, Some(TEXT));

    // A `!cmd` draft is tinted in the accent hue to signal shell mode.
    app.draft = "!ls -la".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(ACCENT));

    // `!!` is the literal-`!` escape (sent to the model), not shell mode.
    app.draft = "!!not a command".to_string();
    app.cursor = app.draft.len();
    let line = app.render_composer_text().lines[0].clone();
    assert_eq!(line.spans[1].style.fg, Some(TEXT));
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
    // Column 4 on row 0 is the 'c' cell ('>'=0, ' '=1, 'a'=2, 'b'=3, 'c'=4) → caret before 'c'.
    assert_eq!(app.composer_offset_for_mouse(click(4, 0)), Some(2));
    // Row 1 holds "  ij"; clicking the 'j' cell (col 3) lands the caret before 'j'.
    assert_eq!(app.composer_offset_for_mouse(click(3, 1)), Some(9));
    // A click outside the composer misses.
    assert_eq!(app.composer_offset_for_mouse(click(40, 40)), None);
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
fn test_question_mark_is_not_help_shortcut() {
    let question = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE);
    let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
    assert!(!is_help_shortcut(question));
    assert!(is_help_shortcut(f1));
}

#[tokio::test]
async fn test_help_overlay_groups_lists_every_command_and_scrolls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `/help` opens the overlay at the top of its body.
    app.open_help_overlay();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));

    // A tall render shows the top: the section header, every purpose group, and
    // every command label (commands sit before the fold, so they all fit).
    let (top, _) = render_full_screen(&mut app, 90, 70);
    assert!(top.contains("Slash commands"), "missing header:\n{top}");
    for group in [
        "Chat",
        "Model & key",
        "Context",
        "Skills & tools",
        "Autonomous",
    ] {
        assert!(top.contains(group), "missing command group {group}:\n{top}");
    }
    for command in SLASH_COMMANDS {
        assert!(
            top.contains(command.help_label),
            "command {} missing from help:\n{top}",
            command.help_label
        );
    }
    // Every command is grouped, so the completeness-guard "More" bucket is empty.
    assert!(
        !top.contains("More"),
        "unexpected ungrouped commands:\n{top}"
    );

    // End scrolls to the bottom; the keybindings + text-entry tips are reachable.
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let (bottom, _) = render_full_screen(&mut app, 90, 24);
    let scrolled = match app.overlay {
        Overlay::Help { scroll } => scroll,
        _ => panic!("help overlay closed unexpectedly"),
    };
    assert!(scrolled > 0, "End did not scroll the help body");
    assert!(
        bottom.contains("Keybindings") || bottom.contains("Text entry"),
        "bottom sections not reachable by scrolling:\n{bottom}"
    );
    assert!(
        bottom.contains("shell command"),
        "text-entry tip not reachable:\n{bottom}"
    );

    // Home snaps back to the top; Esc closes.
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_sgr_mouse_frag_step_classifies_fragment() {
    use super::event_loop_impl::{FragStep, sgr_mouse_frag_step};
    // Valid growing prefixes of `[<{params}`.
    assert!(matches!(sgr_mouse_frag_step("["), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<"), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<6"), FragStep::Continue));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23"),
        FragStep::Continue
    ));
    // Complete reports (press `M` / release `m`).
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23M"),
        FragStep::Final
    ));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23m"),
        FragStep::Final
    ));
    // Not SGR mouse: `[` not followed by `<`, empty params, stray char.
    assert!(matches!(sgr_mouse_frag_step("[h"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<M"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<6x"), FragStep::Invalid));
}

#[test]
fn test_parse_sgr_scroll_only_wheel_buttons() {
    use super::event_loop_impl::parse_sgr_scroll;
    let up = parse_sgr_scroll("[<64;56;23M").unwrap();
    assert!(matches!(up.kind, MouseEventKind::ScrollUp));
    assert_eq!((up.column, up.row), (55, 22)); // SGR is 1-based, MouseEvent 0-based
    let down = parse_sgr_scroll("[<65;1;1m").unwrap();
    assert!(matches!(down.kind, MouseEventKind::ScrollDown));
    assert_eq!((down.column, down.row), (0, 0));
    assert!(parse_sgr_scroll("[<0;5;5M").is_none()); // left-button press, not a wheel
    assert!(parse_sgr_scroll("[<64;5M").is_none()); // missing the row param
    assert!(parse_sgr_scroll("[<64;5;5;5M").is_none()); // an extra param
}

// A fast mouse-wheel report (`\x1b[<65;…M`) that crossterm splits at its ESC must
// not leak its tail into the composer nor spuriously close the open overlay — the
// bare Esc is withheld, the tail swallowed, and the scroll re-synthesized.
#[tokio::test]
async fn test_split_mouse_report_is_reassembled_not_leaked() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    // The leading ESC arrives alone; it is held, not acted on.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "Esc closed overlay early"
    );

    // The tail `[<65;56;23M` (wheel-down) follows as literal chars in the burst.
    for c in "[<65;56;23M".chars() {
        let step = app
            .step_esc_reassembly(
                &mut esc,
                Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
            )
            .await
            .unwrap();
        assert!(
            matches!(step, EscStep::Consumed),
            "char {c:?} leaked through"
        );
    }

    // Nothing typed, overlay still open, and the wheel-down scrolled the body.
    assert_eq!(app.draft, "", "mouse tail leaked into composer");
    match app.overlay {
        Overlay::Help { scroll } => assert_eq!(scroll, 3, "re-synthesized scroll missing"),
        _ => panic!("help overlay was spuriously closed by the split ESC"),
    }
}

#[tokio::test]
async fn test_lone_esc_still_closes_overlay_at_burst_end() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(matches!(app.overlay, Overlay::Help { .. }));

    // Burst ends with the Esc still held: it was real, so flushing closes help.
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_esc_then_non_mouse_text_is_replayed_losslessly() {
    use super::event_loop_impl::{EscReassembly, EscStep};
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // No overlay: characters reach the composer.

    let mut esc = EscReassembly::Idle;
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    // `h` breaks the SGR-mouse shape, so the held `[` and this `h` are real text.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert_eq!(app.draft, "[h", "non-mouse run after Esc was not replayed");
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

#[test]
fn bare_url_in_mcp_add_becomes_url_config() {
    use super::session_impl::bare_url_to_config;
    // A bare http(s) URL → a {url} server config (no JSON typing needed).
    let json = bare_url_to_config("https://mcp.linear.app/mcp").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://mcp.linear.app/mcp");
    assert!(bare_url_to_config("http://127.0.0.1:8080/mcp").is_some());
    // Only the first token is taken (a URL has no spaces).
    let json = bare_url_to_config("https://h/mcp  oops").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://h/mcp");
    // A command line or a JSON block is NOT a bare URL (handled elsewhere).
    assert!(bare_url_to_config("npx -y @scope/server").is_none());
    assert!(bare_url_to_config(r#"{"url":"https://h"}"#).is_none());
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

#[test]
fn test_visible_command_menu_filters_and_hides_escaped_slash() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = 3;
    app.sync_command_menu_state();
    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.entries.len(), 1);
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
    assert_eq!(menu.selected, Some(0));

    app.draft = "//literal".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());

    app.draft = "/model claude".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_filter_slash_commands_prefers_prefix_matches() {
    let matches = filter_slash_commands("m");
    assert_eq!(matches.first().map(|command| command.name), Some("model"));
}

#[test]
fn test_command_menu_area_prefers_below_when_above_space_is_tight() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) = command_menu_area(composer, frame, 6, None);
    assert_eq!(placement, CommandMenuPlacement::Below);
    assert!(area.y >= composer.y + composer.height);
}

#[test]
fn test_command_menu_area_respects_sticky_placement() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) =
        command_menu_area(composer, frame, 6, Some(CommandMenuPlacement::Above));
    assert_eq!(placement, CommandMenuPlacement::Above);
    assert_eq!(area.y, frame.y);
}

#[test]
fn test_command_menu_query_change_keeps_existing_placement_until_reopen() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.command_menu.placement = Some(CommandMenuPlacement::Above);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.command_menu.query = "m".to_string();
    app.command_menu.dismissed = false;

    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Above)
    );

    app.dismiss_command_menu();
    app.draft = "/mod".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(app.command_menu.placement, None);
}

#[test]
fn test_bare_slash_filtering_keeps_detected_placement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    app.command_menu.placement = Some(CommandMenuPlacement::Below);

    app.draft = "/new".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Below)
    );
}

#[test]
fn test_insert_selected_command_uses_argument_space_when_needed() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/model ");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_insert_selected_command_omits_space_for_zero_arg_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/ex".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/exit");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[tokio::test]
async fn test_ctrl_p_navigate_command_menu_before_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["previous prompt".to_string()];
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.selected, Some(menu.entries.len() - 1));
    assert_eq!(app.draft, "/");
    assert!(app.draft_history_index.is_none());
}

#[tokio::test]
async fn test_escape_dismisses_command_menu_until_query_changes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_some());

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    let menu = app.visible_command_menu().unwrap();
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
}

#[tokio::test]
async fn test_enter_executes_selected_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert_eq!(app.cursor, 0);
    assert!(should_exit);
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_render_command_menu_rows_shows_empty_state() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: Vec::new(),
        selected: None,
    };
    let lines = render_command_menu_rows(&menu, 32);
    let plain = plain_text_from_spans(&lines[0].spans);
    assert_eq!(plain, "No matching command");
}

#[test]
fn test_render_command_menu_rows_aligns_description_column() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: vec![
            ComposerMenuEntry::Command(&SLASH_COMMANDS[0]),
            ComposerMenuEntry::Command(&SLASH_COMMANDS[2]),
        ],
        selected: Some(0),
    };
    let lines = render_command_menu_rows(&menu, 48);
    let first = plain_text_from_spans(&lines[0].spans);
    let second = plain_text_from_spans(&lines[1].spans);
    let first_desc = first.find("start a fresh chat").unwrap();
    let second_desc = second.find("resume a saved chat").unwrap();
    assert_eq!(first_desc, second_desc);
}

#[test]
fn test_collect_attach_path_suggestions_lists_matching_entries() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("alpha.txt"), "hi").unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let entries = collect_attach_path_suggestions(temp_dir.path().to_str().unwrap(), "a");

    assert!(entries.iter().any(|entry| entry.label == "assets/"));
    assert!(entries.iter().any(|entry| entry.label == "alpha.txt"));
}

#[test]
fn test_attach_query_uses_path_menu_and_tab_inserts_selected_path() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = temp_dir.path().to_string_lossy().into_owned();
    app.draft = "/attach a".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.kind, MenuKind::AttachPath);
    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/attach assets/");
    // Menu stays open after tab on a directory so the user can continue navigating.
    assert!(app.visible_command_menu().is_some());
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

fn make_test_app(
    tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) -> ChatTuiApp {
    ChatTuiApp {
        // A unique throwaway store — NEVER the real `~/.config/aivo` (which
        // `SessionStore::new()` points at). Tests that drive a save (persist /
        // flush / turn-finish) would otherwise pollute the user's real config.
        session_store: {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("aivo-test-{}-{n}", std::process::id()));
            SessionStore::with_path(dir.join("config.json"))
        },
        cache: ModelsCache::new(),
        client: reqwest::Client::new(),
        key: ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            "https://api.anthropic.com".to_string(),
            None,
            String::new(),
        ),
        copilot_tm: None,
        cwd: String::new(),
        real_cwd: String::new(),
        git_branch: None,
        git_branch_checked_at: None,
        raw_model: String::new(),
        model: String::new(),
        billed_model: None,
        format: ChatFormat::OpenAI,
        history: Vec::new(),
        draft: String::new(),
        draft_attachments: Vec::new(),
        cursor: 0,
        command_menu: CommandMenuState::default(),
        skill_commands: Vec::new(),
        draft_history: Vec::new(),
        draft_history_index: None,
        draft_history_stash: None,
        session_id: String::new(),
        overlay: Overlay::None,
        notice: None,
        pending_response: String::new(),
        incoming_buffer: String::new(),
        pending_finish: None,
        pending_reasoning: String::new(),
        pending_submit: None,
        sending: false,
        request_started_at: None,
        last_usage: None,
        live_usage: None,
        context_tokens: 0,
        session_tokens: crate::services::session_store::SessionTokens::default(),
        context_window: 0,
        context_is_estimate: true,
        follow_output: true,
        transcript_revision: 0,
        transcript_scroll: 0,
        transcript_width: 0,
        transcript_view_height: 0,
        transcript_hitbox: None,
        composer_text_area: None,
        composer_scroll: 0,
        transcript_cache: None,
        volatile_tail_cache: None,
        transcript_selection: None,
        transcript_drag_active: false,
        drag_autoscroll: None,
        last_autoscroll: None,
        last_click: None,
        selection_flash_until: None,
        scroll_speed: DEFAULT_CHAT_SCROLL_SPEED,
        toast: None,
        tx,
        rx,
        response_task: None,
        resume_task: None,
        resume_request_id: 0,
        loading_resume: None,
        resume_restore_state: None,
        reduce_motion: false,
        frame_tick: 0,
        picker_hitbox: None,
        exit_confirm_pending: false,
        cursor_acp_session: None,
        active_agent: None,
        pending_agent_messages: None,
        goal_mode: None,
        agent_engine: None,
        mcp_client: None,
        mcp_connecting: false,
        mcp_connect_progress: std::collections::HashMap::new(),
        mcp_connect_gen: 0,
        mcp_rebuild_pending: false,
        pending_mcp_auth: std::collections::HashMap::new(),
        agent_serve: None,
        agent_permission: None,
        agent_auto_approve: false,
        auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        queued_messages: Vec::new(),
        project_mcp_consent: ProjectMcpConsent::default(),
        pending_mcp_consent: None,
        local_command: None,
        last_local_output: None,
    }
}

#[test]
fn test_resumable_session_id_skips_empty_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "abc-123".to_string();

    // An untouched chat has nothing saved → no resume hint.
    assert!(app.history.is_empty());
    assert_eq!(app.resumable_session_id(), None);

    // Once something has been said, the exit hint points back at this session.
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    assert_eq!(app.resumable_session_id(), Some("abc-123"));
}

fn seed_two_exchanges(app: &mut ChatTuiApp) {
    for (role, content) in [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ] {
        app.history.push(ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

#[tokio::test]
async fn test_rewind_truncates_history_and_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-1".to_string();
    seed_two_exchanges(&mut app);

    // Rewind to the second user turn (history index 2). No live engine in the
    // test app → conversation-only path (ordinal None).
    app.rewind_to_turn(2, None, true).await.unwrap();

    // That turn and everything after it are gone; the prior exchange stays.
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[0].content, "first question");
    assert_eq!(app.history[1].content, "first answer");
    // The rewound message is restored to the composer with the cursor at the end.
    assert_eq!(app.draft, "second question");
    assert_eq!(app.cursor, app.draft.len());
}

#[tokio::test]
async fn test_rewind_to_first_turn_clears_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "rewind-2".to_string();
    seed_two_exchanges(&mut app);

    app.rewind_to_turn(0, None, true).await.unwrap();

    assert!(app.history.is_empty());
    assert_eq!(app.draft, "first question");
}

#[tokio::test]
async fn test_open_rewind_picker_lists_user_turns_newest_first() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);

    app.open_rewind_picker().await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected a rewind picker overlay");
    };
    assert!(matches!(picker.kind, PickerKind::Rewind));
    // One row per user turn, newest first.
    assert_eq!(picker.items.len(), 2);
    let PickerValue::RewindTurn {
        history_index,
        conversation_only,
        ordinal,
    } = &picker.items[0].value
    else {
        panic!("expected a RewindTurn value");
    };
    assert_eq!(*history_index, 2);
    assert!(*conversation_only);
    assert!(ordinal.is_none());
}

#[tokio::test]
async fn test_open_rewind_picker_with_no_turns_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_rewind_picker().await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    let (_, msg) = app.notice.as_ref().expect("a notice");
    assert!(msg.contains("Nothing to rewind to"), "got: {msg}");
}

#[test]
fn test_composer_empty_lines_align_with_cursor_position() {
    use ratatui::buffer::Buffer;
    use ratatui::widgets::Widget;

    // Empty lines must use Line::from("") (no whitespace prefix) to avoid
    // ratatui WordWrapper producing extra visual rows for whitespace-only Lines.
    let lines = vec![
        Line::from(vec![
            Span::styled("> ", Style::default().fg(USER)),
            Span::styled("hello", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("hel", Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("sadf", Style::default().fg(TEXT)),
        ]),
        Line::from(""),
        Line::from(""),
        Line::from(vec![
            Span::raw("  "),
            Span::styled("dsf", Style::default().fg(TEXT)),
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

    assert!(cell_row(0).starts_with("> hello"), "row 0");
    assert!(cell_row(1).starts_with("  hel"), "row 1");
    assert!(cell_row(2).starts_with("  sadf"), "row 2");
    assert!(
        cell_row(5).starts_with("  dsf"),
        "row 5: dsf must align with cursor_position y=5"
    );

    // cursor_position must agree
    let (cx, cy) = cursor_position("hello\nhel\nsadf\n\n\ndsf", 17, 20, 2);
    assert_eq!((cx, cy), (2, 5));
}

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

    assert!(plain.contains("esc to interrupt"));
}

#[test]
fn test_build_transcript_ignores_streaming_reasoning_in_chat() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.pending_reasoning = "Inspecting the request".to_string();
    app.pending_response = "Working on it".to_string();

    let transcript = app.build_transcript();
    let plain = transcript.plain_lines.join("\n");

    assert!(!plain.contains("Thinking"));
    assert!(!plain.contains("Inspecting the request"));
    assert!(plain.contains("Working on it"));
}

#[test]
fn test_composer_placeholder_stays_plain_when_history_has_reasoning() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "assistant".to_string(),
        content: "answer".to_string(),
        reasoning_content: Some("private reasoning".to_string()),
        attachments: vec![],
    });

    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);

    assert_eq!(plain, ">  Ask anything · / for commands");
}

#[test]
fn test_normalized_reasoning_lines_trims_and_removes_blank_runs() {
    assert_eq!(
        normalized_reasoning_lines("\nalpha\n\n\nbeta\n\n"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn test_footer_status_label_is_token_count_without_window() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_tokens = 5_120; // window unknown (0) → plain count

    // Idle and while sending: the footer is the token count — processing status
    // lives in the transcript, not the footer corner.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "~5.1k tokens");
    assert_eq!(color, MUTED);

    app.sending = true;
    assert_eq!(app.footer_status_label().0, "~5.1k tokens");
}

#[test]
fn test_footer_status_label_shows_context_utilization() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;
    app.context_is_estimate = false; // a provider-measured fill

    // used / window · pct%, quiet until it nears the limit.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "10k / 200k · 5%");
    assert_eq!(color, MUTED);

    // Warms toward the window limit (compaction territory).
    app.context_tokens = 170_000; // 85%
    assert_eq!(app.footer_status_label().1, WARNING);
    app.context_tokens = 195_000; // 97%
    assert_eq!(app.footer_status_label().1, ERROR);

    // A measured last-turn total wins over the chars/4 estimate.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k / 200k · 20%");
}

#[test]
fn test_footer_status_label_marks_estimates() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;

    // cursor ACP / agents without reported usage: the chars/4 transcript figure
    // understates the model's real context, so flag it with `~`.
    app.context_is_estimate = true;
    app.last_usage = None;
    assert_eq!(app.footer_status_label().0, "~10k / 200k · ~5%");

    // A provider-measured last-turn total is exact even if the estimate flag
    // lingers from a prior turn — no tilde.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k / 200k · 20%");
}

#[test]
fn test_footer_status_label_updates_live_during_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 40_000; // prior turn's measured fill
    app.context_is_estimate = false;

    // Turn in flight, no measured usage yet: the footer holds the prior fill as a
    // baseline (no drop at turn start) and grows it as text streams in, flagged `~`.
    app.sending = true;
    app.pending_response = "x".repeat(4_000); // ~1k tokens streamed so far
    assert_eq!(app.footer_status_label().0, "~41k / 200k · ~20%");

    // Provider-measured usage arrives mid-stream (Anthropic message_start/_delta):
    // the live figure replaces the estimate immediately, no `~`.
    app.live_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_000,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52k / 200k · 26%");

    // Turn ends: the fold into last_usage keeps the measured total on the footer.
    app.sending = false;
    app.live_usage = None;
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_500,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52.5k / 200k · 26%");
}

#[test]
fn test_agent_context_drives_footer_live() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.sending = true;

    // Pre-usage request estimate (system prompt + tool schemas + conversation):
    // counts the real request the engine sends, not just the visible transcript.
    // Flagged `~` since it is a chars/4 estimate.
    app.apply_agent_context(50_000, false);
    assert_eq!(app.footer_status_label().0, "~50k / 200k · ~25%");

    // The step's measured total arrives → exact figure, no `~`, and streamed text
    // is not re-added on top (it is already in the measured completion).
    app.pending_response = "x".repeat(8_000); // would add ~2k if double-counted
    app.apply_agent_context(60_000, true);
    assert_eq!(app.footer_status_label().0, "60k / 200k · 30%");
}

#[test]
fn test_processing_activity_reflects_phase() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // No streamed text, no tool in flight → waiting on the model.
    assert_eq!(app.processing_activity(), "thinking");
    // A tool call in flight → name the running tool.
    app.history.push(ChatMessage {
        role: "tool_call".to_string(),
        content: r#"{"name":"run_bash","args":{"cmd":"ls"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    assert_eq!(app.processing_activity(), "running run_bash");
    // Streamed tokens arriving → working (takes priority over the tool tail).
    app.pending_response = "partial".to_string();
    assert_eq!(app.processing_activity(), "working");
}

#[test]
fn test_intro_column_stable_from_empty_to_message() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn aivo_col(app: &mut ChatTuiApp) -> u16 {
        let mut terminal = Terminal::new(TestBackend::new(48, 16)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        let area = buf.area;
        // The half-block wordmark's top-left glyphs are "▄▀█" — find that run.
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width.saturating_sub(2) {
                if buf.cell((x, y)).unwrap().symbol() == "▄"
                    && buf.cell((x + 1, y)).unwrap().symbol() == "▀"
                    && buf.cell((x + 2, y)).unwrap().symbol() == "█"
                {
                    return x;
                }
            }
        }
        panic!("AIVO banner not found");
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let empty_col = aivo_col(&mut app);

    app.history.push(ChatMessage {
        role: "assistant".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let message_col = aivo_col(&mut app);

    assert_eq!(
        empty_col, message_col,
        "AIVO Chat banner shifts columns between the empty and message states"
    );
}

#[test]
fn test_transcript_intro_is_brand_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "claude-sonnet-4".to_string();
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/tmp/project".to_string();

    // The intro is the brand banner (wordmark + tagline) — model / base_url / cwd
    // are not repeated here (the footer status bar shows them).
    assert_eq!(
        app.transcript_intro_lines(),
        vec![
            "▄▀█ █ █░█ █▀█".to_string(),
            "█▀█ █ ▀▄▀ █▄█".to_string(),
            "chat · ask anything".to_string(),
        ]
    );
}

#[test]
fn test_session_picker_item_line_fits_mixed_width_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "deepseek".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hi".to_string(),
        preview_text: "hi · Hi there! ✨ 想聊点什么？还是需要我帮忙呢？ 我随时待命～ 😊🌟"
            .to_string(),
    };

    let line = session_picker_item_lines(&preview, true, false, 64)
        .into_iter()
        .next()
        .unwrap();
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 64);
}

#[test]
fn test_key_picker_item_line_fits_modal_width() {
    let key = ApiKey::new_with_protocol(
        "deepseek".to_string(),
        "deepseek".to_string(),
        "https://api.cloudflare.com/client/v4/accounts/long/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let line = key_picker_item_line(&key, true, 36);
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 36);
    assert!(plain.contains("deepseek"));
}

#[test]
fn test_key_search_text_uses_host_not_full_path() {
    let key = ApiKey::new_with_protocol(
        "testgw".to_string(),
        "testgw".to_string(),
        "https://api.ai.example-gateway.net/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let search = key_search_text(&key);
    assert!(search.contains("testgw"));
    assert!(search.contains("api.ai.example-gateway.net"));
    assert!(!search.contains("/endpoint"));
}

#[test]
fn test_key_filter_does_not_match_across_full_url_path() {
    let unrelated = "groq groq api.groq.com";
    let target = "testgw testgw api.ai.example-gateway.net";

    assert!(matches_fuzzy("gapn", target));
    assert!(!matches_fuzzy("gapn", unrelated));
}

#[test]
fn test_notice_display_prefixes_errors_only() {
    let error = (ERROR, "boom".to_string());
    let info = (MUTED, "ok".to_string());

    let displayed = notice_display(Some(&error)).unwrap();
    assert_eq!(displayed.0, ERROR);
    assert_eq!(displayed.1.as_ref(), "Error: boom");

    let displayed = notice_display(Some(&info)).unwrap();
    assert_eq!(displayed.0, MUTED);
    assert_eq!(displayed.1.as_ref(), "ok");

    assert!(notice_display(None).is_none());
}

#[test]
fn test_picker_visible_items_track_selection_for_single_line_rows() {
    let picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..6)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 4,
        kind: PickerKind::Session,
        pending_delete: None,
    };

    let visible = picker.visible_items(3);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 2);
    assert_eq!(visible[2].0, 4);
}

#[test]
fn test_picker_navigation_wraps() {
    let mut picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..3)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 0,
        kind: PickerKind::Session,
        pending_delete: None,
    };

    picker.select_prev();
    assert_eq!(picker.selected, 2);

    picker.select_next();
    assert_eq!(picker.selected, 0);
}

#[test]
fn test_picker_visible_items_respect_single_line_session_rows() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };
    let picker = PickerState {
        title: "Resume",
        query: String::new(),
        items: vec![
            PickerEntry {
                label: "one".to_string(),
                search_text: "one".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "two".to_string(),
                search_text: "two".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "three".to_string(),
                search_text: "three".to_string(),
                value: PickerValue::Session(preview),
            },
        ],
        loading: false,
        selected: 2,
        kind: PickerKind::Session,
        pending_delete: None,
    };

    let visible = picker.visible_items(4);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 0);
    assert_eq!(visible[2].0, 2);
}

#[test]
fn test_session_picker_header_targets_newest_session() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 8, 48);
    let first = plain_text_from_spans(&lines[0].spans);

    assert_eq!(row_map.first().copied(), Some(Some(0)));
    assert!(!first.contains("Newest chat"));
    assert_eq!(row_map.get(1).copied(), Some(Some(0)));
}

#[test]
fn test_grouped_session_picker_short_view_shows_selected_session_row() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 1, 48);
    let only = plain_text_from_spans(&lines[0].spans);

    assert!(only.contains("Newest chat"));
    assert_eq!(row_map, vec![Some(0)]);
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
    // 80 cols minus the 2-col accent gutter, no scrollbar on a short transcript.
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
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let draw = |app: &mut ChatTuiApp, terminal: &mut Terminal<TestBackend>| {
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

// Render the whole screen (transcript + composer + any card/overlay) to a plain
// string plus the per-row strings, for layout assertions.
#[cfg(test)]
fn render_full_screen(app: &mut ChatTuiApp, w: u16, h: u16) -> (String, Vec<String>) {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut rows = Vec::new();
    for y in 0..h {
        let mut row = String::new();
        for x in 0..w {
            row.push_str(buf[(x, y)].symbol());
        }
        rows.push(row);
    }
    (rows.join("\n"), rows)
}

#[test]
fn test_permission_card_anchored_above_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Some history so the composer is pushed toward the bottom of the screen.
    for _ in 0..4 {
        app.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: "working on it".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build/".to_string()),
        reply,
    });

    let (screen, rows) = render_full_screen(&mut app, 60, 20);

    // The friendly heading + command + color-coded keys all render.
    assert!(
        screen.contains("Run a command?"),
        "heading missing:\n{screen}"
    );
    assert!(
        screen.contains("rm -rf build/"),
        "command missing:\n{screen}"
    );
    assert!(
        screen.contains("allow once") && screen.contains("always") && screen.contains("deny"),
        "keys missing:\n{screen}"
    );
    // A destructive command is flagged.
    assert!(
        screen.contains("⚠ looks destructive"),
        "destructive flag missing:\n{screen}"
    );

    // The card hugs the composer: its bottom border sits just above the input's
    // full-width divider rule, which sits directly above the prompt (not floating
    // mid-screen like a centered modal).
    let bottom_border_row = rows
        .iter()
        .position(|r| r.contains('└'))
        .expect("card bottom border");
    let composer_row = rows
        .iter()
        .position(|r| r.trim_start().starts_with('>'))
        .expect("composer prompt row");
    assert_eq!(
        bottom_border_row + 2,
        composer_row,
        "card bottom must sit one row (the divider) above the composer:\n{screen}"
    );
    // The divider directly under the card is the full-width composer rule — now
    // always carrying the auto-approve badge (here "off", since a card only
    // shows when auto-approve is off) — so the narrower card never leaves it
    // poking out past the card's right edge.
    let divider = &rows[bottom_border_row + 1];
    assert!(
        divider.contains('─') && divider.contains("auto-approve"),
        "full-width composer rule (with the auto-approve badge) must sit under the card:\n{screen}"
    );

    // The card is sized to its content, not stretched across the full 60-col
    // input row — its border must be clearly narrower than the screen.
    let border_w = rows[bottom_border_row].trim_end().chars().count();
    assert!(
        (10..50).contains(&border_w),
        "card should be a compact width, got {border_w}:\n{screen}"
    );
}

#[test]
fn test_permission_card_keys_always_visible_in_short_terminal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    // A long multi-line preview that cannot fully fit above the composer.
    let preview = (0..30)
        .map(|i| format!("line {i} of a very long command"))
        .collect::<Vec<_>>()
        .join("\n");
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some(preview),
        reply,
    });

    // Short screen: the preview must be trimmed so the keys row still shows.
    let (screen, _rows) = render_full_screen(&mut app, 50, 10);
    assert!(
        screen.contains("allow once") && screen.contains("deny"),
        "keys must survive a cramped card:\n{screen}"
    );
}

#[test]
fn test_spinner_animation_does_not_invalidate_transcript_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut ChatTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
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
        screen.contains("esc to interrupt"),
        "spinner missing:\n{screen}"
    );

    // Advancing the spinner glyph must not rebuild the body — only the appended
    // status line is volatile, so a long transcript never reparses to animate.
    app.frame_tick = app.frame_tick.wrapping_add(7);
    let screen = render_screen(&mut app, &mut terminal);
    assert_eq!(
        app.transcript_cache.as_ref().unwrap().body.lines.as_ptr(),
        body_ptr,
        "spinner animation must reuse the cached body"
    );
    assert!(
        screen.contains("esc to interrupt"),
        "spinner missing:\n{screen}"
    );
}

#[test]
fn test_streaming_tokens_do_not_invalidate_history_body_cache() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "assistant".to_string(),
        content: "# Heading\n\nSome **markdown** reply.".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Stream".to_string();

    let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
    let render_screen = |app: &mut ChatTuiApp, terminal: &mut Terminal<TestBackend>| -> String {
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
        role: "user".to_string(),
        content: "explain the plan in detail please".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
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
    app.notice = Some((MUTED, "compacting context…".to_string()));

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
        app.notice = notice.map(|t| (MUTED, t.to_string()));
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

/// The `!cmd` output pager windows a large capture: it draws only the visible
/// slice (with a correct range/total footer) and clamps over-scroll to the last
/// page, instead of materializing the whole buffer into `Line`s every keystroke.
#[test]
fn output_pager_windows_large_capture() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let stdout: String = (0..1000).map(|i| format!("L{i:04}\n")).collect();
    app.last_local_output = Some(LocalCommandOutput {
        command: "seq 1000".to_string(),
        stdout,
        stderr: String::new(),
        exit_code: 0,
        truncated: false,
        interrupted: false,
    });

    let render = |app: &ChatTuiApp, scroll: u16| -> (String, u16) {
        let mut terminal = Terminal::new(TestBackend::new(60, 20)).unwrap();
        let mut clamped = 0u16;
        terminal
            .draw(|frame| {
                clamped = app.render_output_overlay(frame, Rect::new(0, 0, 60, 20), scroll);
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let screen: String = (0..20)
            .map(|y| (0..60).map(|x| buf[(x, y)].symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        (screen, clamped)
    };

    // 60x20 minus margin(v1,h2) ⇒ inner height 18 ⇒ body_h 17.
    let (screen, clamped) = render(&app, 50);
    assert_eq!(clamped, 50);
    assert!(
        screen.contains("L0050"),
        "window starts at the scroll offset:\n{screen}"
    );
    assert!(
        screen.contains("L0066"),
        "window ends at scroll+body_h-1:\n{screen}"
    );
    assert!(
        !screen.contains("L0049"),
        "the line above the window isn't drawn"
    );
    assert!(
        !screen.contains("L0067"),
        "the line below the window isn't drawn"
    );
    assert!(
        screen.contains("51–67/1000"),
        "footer shows the true range/total:\n{screen}"
    );

    // Over-scroll clamps to the last full page (total - body_h = 983).
    let (screen, clamped) = render(&app, 60_000);
    assert_eq!(clamped, 983, "over-scroll clamps to the last page");
    assert!(
        screen.contains("L0999"),
        "the final line is visible:\n{screen}"
    );
    assert!(screen.contains("984–1000/1000"), "footer at end:\n{screen}");
}

#[test]
fn test_wrap_transcript_carries_bar_color_per_row() {
    use ratatui::text::Line;
    let lines = vec![StyledLine {
        line: Line::from("alpha beta gamma delta"),
        plain: "alpha beta gamma delta".to_string(),
    }];
    let wrapped = wrap_transcript(&lines, &[Some(TOOL)], 8);
    assert!(wrapped.rows.len() >= 3);
    // Every wrapped row inherits the source line's bar color.
    assert!(wrapped.bars.iter().all(|b| *b == Some(TOOL)));
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
            Span::styled(" + ".to_string(), Style::default().bg(DIFF_ADD_BG)),
            Span::styled(
                "let very long added line".to_string(),
                Style::default().bg(DIFF_ADD_BG),
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
            Some(DIFF_ADD_BG)
        );
    }

    // A plain line (no trailing background) is never padded — only opted-in tinted
    // rows gain a background.
    let plain = vec![StyledLine {
        line: Line::from(Span::styled("hi".to_string(), Style::default().fg(TEXT))),
        plain: "hi".to_string(),
    }];
    let wrapped = wrap_transcript(&plain, &[None], 12);
    assert_eq!(wrapped.rows[0], "hi");
}

#[test]
fn test_edit_diff_rows_carry_add_remove_tints() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
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
        Some(DIFF_DEL_BG),
        "removed line tint"
    );
    assert_eq!(bg_of("+ let x = 2;"), Some(DIFF_ADD_BG), "added line tint");
}

#[test]
fn test_edit_diff_shows_context_only_marks_changed_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Only the middle line changes; the first and last are shared context.
    app.history.push(ChatMessage {
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
    assert_eq!(bg_of("- "), Some(DIFF_DEL_BG), "old line removed + tinted");
    assert_eq!(bg_of("+ "), Some(DIFF_ADD_BG), "new line added + tinted");
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
fn test_hint_bar_reflects_state() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // Render the app, returning just the bottom row (the hint bar).
    fn bottom_row(configure: impl Fn(&mut ChatTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        // A long transcript so the footer fills to the terminal's bottom row.
        app.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: (0..40)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            reasoning_content: None,
            attachments: vec![],
        });
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut row = String::new();
        for x in 0..80u16 {
            row.push_str(buf[(x, 11)].symbol());
        }
        row
    }

    // The whole rendered screen as one string, for state shown outside the bottom
    // hint-bar row (e.g. the composer-rule mode badge).
    fn full_screen(configure: impl Fn(&mut ChatTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..80u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    }

    assert!(bottom_row(|_| {}).contains("commands"), "idle hint bar");
    assert!(
        bottom_row(|a| a.sending = true).contains("interrupt"),
        "sending hint bar"
    );
    // Auto-approve rides the composer rule, not the bottom hint bar, and BOTH
    // states are shown so the mode + its toggle key are always discoverable.
    assert!(
        full_screen(|a| a.agent_auto_approve = true).contains("auto-approve: on"),
        "auto-approve ON badge on composer rule"
    );
    assert!(
        full_screen(|a| a.agent_auto_approve = false).contains("auto-approve: off"),
        "auto-approve OFF state shown on composer rule (discoverable)"
    );
    assert!(
        !bottom_row(|a| a.agent_auto_approve = true).contains("auto-approve"),
        "auto-approve no longer in the bottom hint bar"
    );
    assert!(
        bottom_row(|a| a.queued_messages = vec!["next".to_string()]).contains("queued"),
        "queued indicator"
    );
}

#[test]
fn test_inline_status_stays_in_transcript_across_phases() {
    // The processing status renders in the transcript in every phase, so its
    // position never jumps to the footer when the reply starts streaming.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    // Thinking: status line, no streamed text yet.
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("thinking"), "thinking phase: {plain:?}");
    assert!(plain.contains("esc to interrupt"));

    // Working: the status line follows the streamed reply, still in-stream.
    app.pending_response = "streaming the answer".to_string();
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("streaming the answer"));
    assert!(plain.contains("working"), "working phase: {plain:?}");
    assert!(plain.contains("esc to interrupt"));
}

#[test]
fn test_display_cwd_shows_real_dir_for_agent_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // make_test_app uses a plain API key → agent-capable → show the real dir.
    assert!(app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
}

#[test]
fn test_display_cwd_shows_real_dir_for_cursor_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key = ApiKey::new_with_protocol(
        "cursor".to_string(),
        String::new(),
        "cursor".to_string(),
        None,
        String::new(),
    );
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // cursor-agent now runs as a real agent in the launch dir (not the in-process
    // engine, so still not `agent_capable`), so the footer shows the real dir.
    assert!(app.key.is_cursor_acp());
    assert!(!app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
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
    );
    assert_eq!(app.history.len(), 4);
    assert_eq!(app.history[3].role, "tool_call");
}

#[test]
fn test_native_tool_paths_render_relative_to_cwd() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/Users/yc/project/work/llm-server".to_string();

    // A path under the cwd renders relative — the absolute prefix is noise the
    // footer already shows, and it pushed the distinguishing basename off-screen.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/yc/project/work/llm-server/apps/web/src/routes/chat/+page.svelte"}),
    );
    app.apply_agent_tool_result("128 lines".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("read_file(apps/web/src/routes/chat/+page.svelte)"),
        "{plain}"
    );
    assert!(
        !plain.contains("/Users/yc/project"),
        "absolute path leaked: {plain}"
    );

    // A relative path too long to fit is left-truncated on a segment boundary so
    // the basename survives (vs. the old tail-truncation that cut it off).
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "/Users/yc/project/work/llm-server/apps/web/src/routes/dashboard/settings/billing/invoices/detail/+page.svelte"}),
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("…/"), "expected left-truncation: {plain}");
    assert!(plain.contains("+page.svelte"), "basename lost: {plain}");
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
        app.apply_agent_tool_call(None, tool.to_string(), serde_json::json!({"path": "x"}));
        app.apply_agent_tool_result(output.to_string());
        let plain = app.build_transcript().plain_lines.join("\n");
        assert!(
            plain.contains(expected),
            "{tool}: expected {expected:?} in {plain}"
        );
    }
}

#[test]
fn test_subagents_render_individually_not_coalesced() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Two subagents dispatched in parallel arrive as adjacent tool_calls. Unlike
    // the many tiny cursor steps coalescing is meant to fold, each subagent is a
    // distinct unit of work and must keep its task visible — never an opaque
    // `subagent ×2`.
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Review the chat API endpoint for gaps"}),
    );
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"agent": "reviewer", "task": "Audit the auth flow"}),
    );
    app.apply_agent_tool_result("## Findings\nfirst\nsecond".to_string());

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("subagent ×"),
        "subagents must not coalesce: {plain}"
    );
    // The word "subagent" is jargon and must not appear — each delegated task is
    // shown directly after the arrow, with no `subagent(...)` wrapper.
    assert!(
        !plain.contains("subagent"),
        "the word 'subagent' must not be shown: {plain}"
    );
    assert!(
        plain.contains("→ Review the chat API endpoint for gaps"),
        "first delegated task missing: {plain}"
    );
    assert!(
        plain.contains("→ Audit the auth flow"),
        "second delegated task missing: {plain}"
    );
    // The result previews the report's first line + a `+N more` tail — not a bare
    // "3 lines" count that says nothing about what the subagent found.
    assert!(
        plain.contains("## Findings (+2 more)"),
        "subagent result preview missing: {plain}"
    );
    assert!(
        !plain.contains("3 lines"),
        "bare line count leaked for subagent: {plain}"
    );
}

#[test]
fn test_delegating_spinner_label_drops_subagent_word() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Audit the auth flow"}),
    );
    let activity = app.processing_activity();
    assert_eq!(activity, "delegating", "spinner should read 'delegating'");
    assert!(!activity.contains("subagent"), "spinner leaked 'subagent'");
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
fn test_typewriter_reveals_buffer_progressively() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 14 chars: the floor (10) dominates the 1/2 catch-up (7), revealing 10 on
    // the first frame.
    app.incoming_buffer = "你好世界一二三四五六七八九十".to_string();
    assert!(app.tick_typewriter());
    assert_eq!(app.pending_response, "你好世界一二三四五六"); // exactly 10 chars, no split glyph
    assert_eq!(app.incoming_buffer.chars().count(), 4);

    // Drains fully over the next frames, never losing or splitting a char.
    let mut guard = 0;
    while app.tick_typewriter() {
        guard += 1;
        assert!(guard < 100, "typewriter should converge");
    }
    assert_eq!(app.pending_response, "你好世界一二三四五六七八九十");
    assert!(app.incoming_buffer.is_empty());
}

#[tokio::test]
async fn test_finish_deferred_until_typewriter_drains() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.history.push(ChatMessage {
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
fn test_build_transcript_renders_tool_steps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        role: "tool_call".to_string(),
        content: r#"{"name":"read_file","args":{"path":"src/parser.rs"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
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
        plain.contains("⎿ 3 lines"),
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
        .position(|l| l.contains("⎿ 3 lines"))
        .unwrap();
    assert_eq!(result_idx, call_idx + 1, "result should hug its call");
    assert_eq!(transcript.bar_colors[call_idx], Some(TOOL));
    assert_eq!(transcript.bar_colors[result_idx], Some(TOOL));
}

#[test]
fn test_agent_seed_turns_folds_tool_steps() {
    let msg = |role: &str, content: &str| ChatMessage {
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
    let seed = super::runtime_impl::agent_seed_turns(&history);
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
        role: "tool_call".to_string(),
        content: r#"{"name":"edit_file","args":{"path":"src/a.rs","old_string":"let x = 1;","new_string":"let x = 2;"}}"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
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
fn test_plan_renders_in_pinned_panel_not_inline() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "plan".to_string(),
        content: r#"[{"step":"scan code","status":"completed"},{"step":"write fix","status":"in_progress"},{"step":"run tests","status":"pending"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    // The plan is pinned in its own panel above the composer — it must NOT render
    // inline in the transcript (where it would scroll away under later content).
    let inline = app.build_transcript().plain_lines.join("\n");
    assert!(
        !inline.contains("Plan") && !inline.contains("scan code"),
        "plan leaked into the inline transcript:\n{inline}"
    );

    // Render the full UI: the pinned panel carries the header, every step, and the
    // per-status glyphs.
    let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let screen: String = (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(screen.contains("Plan"), "panel header missing:\n{screen}");
    assert!(
        screen.contains("1/3 done"),
        "panel progress missing:\n{screen}"
    );
    for step in ["scan code", "write fix", "run tests"] {
        assert!(
            screen.contains(step),
            "panel step {step} missing:\n{screen}"
        );
    }
    assert!(screen.contains('✔') && screen.contains('▸') && screen.contains('○'));
}

#[test]
fn test_completed_plan_clears_on_next_user_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plan_count = |a: &ChatTuiApp| a.history.iter().filter(|m| m.role == "plan").count();

    // A finished plan is recorded and stays pinned (nothing clears it on its own).
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(plan_count(&app), 1);

    // The next user message clears it — `send_user_message` runs this before
    // pushing the turn, so a done plan doesn't linger into a new task.
    app.clear_completed_plan();
    assert_eq!(plan_count(&app), 0, "done plan cleared on next message");

    // An UNFINISHED plan is never auto-cleared — work is still in progress.
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "in_progress"}]));
    app.clear_completed_plan();
    assert_eq!(plan_count(&app), 1, "an active plan must not be cleared");
}

fn dummy_agent_session() -> AgentSession {
    AgentSession {
        key_id: "k".to_string(),
        model: "m".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::engine::AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0),
        )),
    }
}

#[tokio::test]
async fn test_apply_mcp_connected_empty_keeps_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.mcp_connecting = true;
    app.agent_engine = Some(dummy_agent_session());
    // An empty client (no mcp.json in this temp dir) brings no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-empty-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    app.apply_mcp_connected(client);
    assert!(!app.mcp_connecting, "connecting flag should clear");
    assert!(app.mcp_client.is_some(), "client should be cached");
    assert!(
        app.agent_engine.is_some(),
        "an empty MCP result must not drop the engine"
    );
    assert!(!app.mcp_rebuild_pending);
}

#[tokio::test]
async fn test_apply_mcp_connected_surfaces_connect_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A config pointing at a non-spawnable command → a connect error, no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-notice-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    assert!(!client.errors().is_empty(), "expected a connect error");

    app.apply_mcp_connected(client);
    let notice = app
        .notice
        .as_ref()
        .expect("a failed MCP connect should notify");
    assert!(notice.1.contains("MCP"), "notice: {}", notice.1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// When a background connect resolves while the `/mcp` overlay is open, its rows
/// must refresh from the new client in place (no close-and-reopen) — here the
/// "connecting…" row flips to the failure once the broken server's error lands.
#[tokio::test]
async fn test_apply_mcp_connected_refreshes_open_overlay() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "broken".to_string(),
            status: "connecting…".to_string(),
            health: McpHealth::Idle,
            enabled: true,
            scope: ServerScope::Project,
            command: "aivo_no_such_binary_zzz".to_string(),
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-refresh-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );

    app.apply_mcp_connected(client);
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(
            state.items[0].health,
            McpHealth::Failed,
            "open overlay row not refreshed to failed: {}",
            state.items[0].status
        );
        assert!(
            state.items[0].status.contains("failed"),
            "status: {}",
            state.items[0].status
        );
    } else {
        panic!("mcp overlay vanished");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A connect launched before a `/mcp` toggle (older generation) must be dropped,
/// so it can't resurrect a just-disabled server; the current generation applies.
#[tokio::test]
async fn test_stale_mcp_connect_is_dropped() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx2 = tx.clone();
    let mut app = make_test_app(tx, rx);
    // A toggle advanced the generation while a previous connect is in flight.
    app.mcp_connect_gen = 1;
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-stale-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let stale = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: stale,
        generation: 0,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_none(), "stale connect must not be cached");
    assert!(
        app.mcp_connecting,
        "stale connect must not clear the in-flight flag"
    );

    let fresh = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: fresh,
        generation: 1,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_some(), "current-gen connect should cache");
    assert!(!app.mcp_connecting, "current-gen connect clears the flag");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_maybe_apply_mcp_rebuild_drops_engine_when_pending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.agent_engine = Some(dummy_agent_session());
    app.mcp_rebuild_pending = true;
    app.maybe_apply_mcp_rebuild();
    assert!(
        app.agent_engine.is_none(),
        "pending rebuild should drop engine"
    );
    assert!(!app.mcp_rebuild_pending, "flag should clear");
    // Not pending → engine left alone.
    app.agent_engine = Some(dummy_agent_session());
    app.maybe_apply_mcp_rebuild();
    assert!(app.agent_engine.is_some());
}

#[test]
fn test_apply_agent_plan_keeps_single_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let count_plans = |app: &ChatTuiApp| app.history.iter().filter(|m| m.role == "plan").count();

    // Two updates with nothing between → one card, updated in place.
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "pending"}]));
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(count_plans(&app), 1, "consecutive updates should collapse");
    assert!(app.history.last().unwrap().content.contains("completed"));

    // A plan after real work still keeps ONE card, relocated to the latest point
    // (so the transcript never stacks a near-identical copy after each batch).
    app.history.push(ChatMessage {
        role: "tool_call".to_string(),
        content: "{}".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(count_plans(&app), 1, "plan after work stays a single card");
    assert_eq!(
        app.history.last().unwrap().role,
        "plan",
        "the card relocates to the latest position"
    );
}

#[test]
fn test_build_transcript_coalesces_consecutive_tool_calls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "study the sidebar".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Cursor-style: a run of read_file calls with no interleaved results.
    for path in ["src/sidebar.rs", "src/session.rs", "src/time.rs"] {
        app.history.push(ChatMessage {
            role: "tool_call".to_string(),
            content: format!(r#"{{"name":"read_file","args":{{"path":"{path}"}}}}"#),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    // A different kind right after starts a new run (not merged with the reads).
    app.history.push(ChatMessage {
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

#[test]
fn test_render_main_paints_per_role_accent_gutter() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "ping".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Long enough to wrap across several rows at width 24.
    app.history.push(ChatMessage {
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
        if cell.fg == USER {
            user_bar_rows += 1;
        } else if cell.fg == ACCENT {
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

    // Bottom of the composer = terminal height (12) minus the 2-row footer.
    assert_eq!(composer_area.y + composer_area.height, 10);
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.y, 0);
    // 80 cols minus 1 for the scrollbar, minus the 2-col accent gutter.
    assert_eq!(app.transcript_hitbox.as_ref().unwrap().area.width, 77);
    assert_eq!(app.transcript_width, 77);
}

#[tokio::test]
async fn test_mouse_wheel_scrolls_only_inside_transcript_hitbox() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
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

    fn rendered_cells(app: &mut ChatTuiApp) -> Vec<(String, Color)> {
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
async fn test_auto_approve_toggle_shows_toast_not_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // Shift+Tab (BackTab) flips auto-approve. The confirmation must be a
    // self-expiring toast, NOT a persistent transcript notice pinned above the
    // input for the rest of the session.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve);
    assert!(
        app.notice.is_none(),
        "toggle must not pin a lingering notice"
    );
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Auto-approve ON")),
        "toggle should flash a toast"
    );

    // Toggling back off behaves the same way.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.agent_auto_approve);
    assert!(app.notice.is_none());
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Auto-approve off"))
    );
}

#[tokio::test]
async fn test_mouse_drag_coordinates_map_to_transcript_rows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
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

    assert!(app.select_word_at(TranscriptPoint { row: 0, column: 7 }));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 6 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    assert!(app.select_line_at(TranscriptPoint { row: 0, column: 2 }));
    assert_eq!(
        app.transcript_selection,
        Some(TranscriptSelection {
            anchor: TranscriptPoint { row: 0, column: 0 },
            focus: TranscriptPoint { row: 0, column: 10 },
        })
    );

    // A click on whitespace produces no word selection.
    app.transcript_selection = None;
    assert!(!app.select_word_at(TranscriptPoint { row: 0, column: 5 }));
    assert!(app.transcript_selection.is_none());
}

#[tokio::test]
async fn test_drag_to_bottom_edge_arms_and_advances_autoscroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Real content so max_scroll() (which rebuilds from history) leaves room to
    // scroll past the 4-row viewport.
    app.history.push(ChatMessage {
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
            if cell.bg == SELECT_WARM {
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
fn test_parse_slash_command_with_argument() {
    assert_eq!(
        parse_slash_command("model claude-sonnet-4").unwrap(),
        SlashCommand::Model(Some("claude-sonnet-4".to_string()))
    );
    assert_eq!(
        parse_slash_command("attach ./README.md").unwrap(),
        SlashCommand::Attach("./README.md".to_string())
    );
    assert_eq!(
        parse_slash_command("resume").unwrap(),
        SlashCommand::Resume(None)
    );
    assert_eq!(
        parse_slash_command("detach 2").unwrap(),
        SlashCommand::Detach(2)
    );
    assert_eq!(
        parse_slash_command("mcp add fs npx -y srv").unwrap(),
        SlashCommand::Mcp(Some("add fs npx -y srv".to_string()))
    );
    assert_eq!(parse_slash_command("mcp").unwrap(), SlashCommand::Mcp(None));
}

#[tokio::test]
async fn test_mcp_add_json_routes_and_reports_parse_error() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A `{`-leading add input routes to the JSON path; malformed JSON surfaces a
    // parse error (and writes nothing — verified by never reaching a real write).
    app.submit_mcp_add("{ not valid json".to_string())
        .await
        .unwrap();
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("Couldn't parse MCP config"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

#[tokio::test]
async fn test_mcp_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_mcp_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Mcp(_)),
        "bare /mcp opens overlay"
    );

    // Unknown subcommand → usage notice, overlay not opened.
    app.overlay = Overlay::None;
    app.run_mcp_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name → usage notice.
    app.run_mcp_command(Some("rm".to_string())).await.unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent server → "No MCP server" notice, no config write.
    app.run_mcp_command(Some("rm __aivo_no_such_server__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No MCP server"),
        "notice: {:?}",
        app.notice
    );
}

#[test]
fn test_parse_slash_command_unknown() {
    let err = parse_slash_command("wat").unwrap_err().to_string();
    assert!(err.contains("Unknown command"));
}

#[test]
fn test_parse_slash_skills() {
    assert_eq!(
        parse_slash_command("skills").unwrap(),
        SlashCommand::Skills(None)
    );
    assert_eq!(
        parse_slash_command("skills add fs Helper").unwrap(),
        SlashCommand::Skills(Some("add fs Helper".to_string()))
    );
    // `/skills` is advertised in the command menu + help listing.
    assert!(SLASH_COMMANDS.iter().any(|c| c.name == "skills"));
}

fn skill_command(name: &str, description: &str) -> SkillCommand {
    SkillCommand {
        name: name.to_string(),
        description: description.to_string(),
    }
}

#[test]
fn test_filter_skill_commands_ranks_prefix_before_fuzzy() {
    let commands = vec![
        skill_command("repo-study", "Study a repo"),
        skill_command("review", "Review a PR"),
        skill_command("deep-research", "Research a topic"),
    ];
    // Empty query returns everything, order preserved.
    assert_eq!(filter_skill_commands(&commands, "").len(), 3);
    // Prefix match wins; the fuzzy `re…h` (deep-research) still shows, but after.
    let names: Vec<String> = filter_skill_commands(&commands, "re")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names.first().map(String::as_str), Some("repo-study"));
    assert!(names.contains(&"review".to_string()));
    // A query matching nothing yields nothing.
    assert!(filter_skill_commands(&commands, "zzz").is_empty());
}

#[test]
fn test_resolve_slash_command_skill_vs_builtin_vs_typo() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![skill_command("repo-study", "Study a repo")];

    // A discovered skill resolves to the Skill variant, with trailing args.
    assert_eq!(
        app.resolve_slash_command("repo-study https://x/y").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: Some("https://x/y".to_string()),
        }
    );
    // No args → None.
    assert_eq!(
        app.resolve_slash_command("repo-study").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: None,
        }
    );
    // A built-in always wins, even if a same-named skill exists.
    app.skill_commands.push(skill_command("model", "shadow"));
    assert_eq!(
        app.resolve_slash_command("model gpt").unwrap(),
        SlashCommand::Model(Some("gpt".to_string()))
    );
    // An unknown name (not a built-in, not a skill) still errors.
    let err = app.resolve_slash_command("nope").unwrap_err().to_string();
    assert!(err.contains("Unknown command"), "{err}");
}

#[test]
fn test_matching_command_entries_includes_skills_after_builtins() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![
        skill_command("repo-study", "Study a repo"),
        // A skill colliding with the built-in `/model` must be dropped.
        skill_command("model", "shadow"),
    ];
    // No built-in starts with "repo", so the skill is the sole match.
    let entries = app.matching_command_entries("repo");
    let labels: Vec<String> = entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(labels, vec!["/repo-study".to_string()]);

    // `/model` resolves to the built-in only — the colliding skill never appears.
    let model_entries = app.matching_command_entries("model");
    let model_labels: Vec<String> = model_entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(
        model_labels.iter().filter(|l| *l == "/model").count(),
        1,
        "a skill must not duplicate or shadow a built-in command"
    );
}

#[tokio::test]
async fn test_refresh_skill_commands_discovers_and_respects_disabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A project skill under the app's working dir, with a unique name so real
    // home-dir skills can't collide with the assertions.
    let proj = std::env::temp_dir().join(format!(
        "aivo-skillcmd-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let name = "zz-unique-skill-cmd";
    let dir = proj.join(".aivo").join("skills").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: A unique test skill.\n---\nBody.\n"),
    )
    .unwrap();
    app.real_cwd = proj.to_string_lossy().into_owned();

    app.refresh_skill_commands().await;
    let found = app.skill_commands.iter().find(|c| c.name == name);
    assert!(found.is_some(), "discovered skill should become a command");
    assert_eq!(found.unwrap().description, "A unique test skill.");

    // Disabling it in `/skills` drops it from the command set.
    app.session_store
        .set_skill_enabled(name, false)
        .await
        .unwrap();
    app.refresh_skill_commands().await;
    assert!(
        !app.skill_commands.iter().any(|c| c.name == name),
        "a disabled skill must not be offered as a command"
    );

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn test_expand_skill_invocation_arguments_and_fallback() {
    use crate::agent::skills::Skill;
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // `$ARGUMENTS` is substituted in place.
    assert_eq!(
        super::runtime_impl::expand_skill_invocation(&placeholder, Some("the repo")),
        "Study the repo now."
    );

    let plain = Skill {
        name: "repo-study".to_string(),
        description: "d".to_string(),
        body: "Follow these steps.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // No placeholder + args → directive + body + appended input.
    let out = super::runtime_impl::expand_skill_invocation(&plain, Some("https://x/y"));
    assert!(out.contains("Use the \"repo-study\" skill"), "{out}");
    assert!(out.contains("Follow these steps."), "{out}");
    assert!(out.ends_with("Input: https://x/y"), "{out}");
    // No placeholder, no args → no trailing input line.
    let bare = super::runtime_impl::expand_skill_invocation(&plain, None);
    assert!(bare.contains("Follow these steps."));
    assert!(!bare.contains("Input:"));
}

/// The display/log recognizer recovers the compact `/name args` the user typed
/// from an expanded invocation, round-tripping with the producer, and declines
/// ordinary messages and `$ARGUMENTS`-style skills (which leave no wrapper).
#[test]
fn test_skill_invocation_label_recovers_typed_command() {
    use super::runtime_impl::{expand_skill_invocation, skill_invocation_label};
    use crate::agent::skills::Skill;
    let skill = Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "Search Baidu.\n\nUse the bundled script.".to_string(),
        dir: std::path::PathBuf::new(),
    };

    // With args (incl. CJK) → `/name args`.
    let expanded = expand_skill_invocation(&skill, Some("歌曲"));
    assert_eq!(
        skill_invocation_label(&expanded).as_deref(),
        Some("/baidu-search 歌曲")
    );
    // No args → bare `/name`.
    let bare = expand_skill_invocation(&skill, None);
    assert_eq!(
        skill_invocation_label(&bare).as_deref(),
        Some("/baidu-search")
    );

    // A body that itself contains an `Input:`-style line must not be mistaken for
    // args when none were passed (multi-line tail ⇒ no args).
    let trappy = Skill {
        name: "trap".to_string(),
        description: "d".to_string(),
        body: "Step.\n\nInput: foo\nmore body".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&trappy, None)).as_deref(),
        Some("/trap")
    );

    // Ordinary user text and a `$ARGUMENTS` skill (no wrapper) → None.
    assert_eq!(skill_invocation_label("just a normal question"), None);
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&placeholder, Some("x"))),
        None
    );
}

/// The transcript shows the compact `/name args` for a skill turn, not the whole
/// inlined SKILL.md body (the verbosity reported in the field).
#[tokio::test]
async fn test_skill_turn_renders_compact_not_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let skill = crate::agent::skills::Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "SEARCH_BAIDU_BODY_MARKER do the thing.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    let expanded = super::runtime_impl::expand_skill_invocation(&skill, Some("歌曲"));
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: expanded,
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The compact command shows (the test backend pads wide CJK glyphs with an
    // inter-cell space, so assert on the ASCII command + each arg glyph rather
    // than the exact contiguous string).
    assert!(
        screen.contains("/baidu-search"),
        "compact label missing:\n{screen}"
    );
    assert!(
        screen.contains('歌') && screen.contains('曲'),
        "skill arg missing from the turn:\n{screen}"
    );
    assert!(
        !screen.contains("SEARCH_BAIDU_BODY_MARKER"),
        "the inlined body leaked into the transcript:\n{screen}"
    );
}

fn skills_overlay_fixture() -> SkillsOverlay {
    use crate::agent::skills::SkillScope;
    SkillsOverlay {
        items: vec![
            SkillToggle {
                name: "brandkit".to_string(),
                description: "Premium brand-kit image generation.".to_string(),
                enabled: true,
                dir: std::path::PathBuf::from("/home/me/.config/aivo/skills/brandkit"),
                scope: SkillScope::User,
                body: "Step 1. Render the boards.".to_string(),
            },
            SkillToggle {
                name: "critique".to_string(),
                description: "Evaluate design effectiveness from a UX perspective.".to_string(),
                enabled: false,
                dir: std::path::PathBuf::from("/repo/.agents/skills/critique"),
                scope: SkillScope::Project,
                body: "Step 1. Score the design.".to_string(),
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

#[test]
fn test_skills_overlay_renders_toggle_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("Skills"), "missing title:\n{screen}");
    assert!(screen.contains("brandkit"), "missing skill name:\n{screen}");
    assert!(
        screen.contains("[✓]"),
        "missing enabled checkbox:\n{screen}"
    );
    assert!(
        screen.contains("[ ]"),
        "missing disabled checkbox:\n{screen}"
    );
    // The name and its description render on separate lines.
    assert!(
        screen.contains("Premium brand-kit"),
        "missing description line:\n{screen}"
    );
    // The on-count badge sits in the top border (1 of 2 on).
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
    // Search placeholder up top, controls along the footer.
    assert!(
        screen.contains("filter skills") && screen.contains("toggle"),
        "missing controls:\n{screen}"
    );
}

/// The detail line shows the selected skill's location, and tags a project
/// skill (which `d` can't delete). Selecting the project-scoped row surfaces
/// the `project` marker that appears nowhere else on screen.
#[test]
fn test_skills_overlay_detail_line_shows_scope_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.selected = 1; // "critique", a project skill
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project"),
        "detail line should tag a project-scoped skill:\n{screen}"
    );
    assert!(
        screen.contains("skills/critique"),
        "detail line should show the skill's path:\n{screen}"
    );
}

#[tokio::test]
async fn test_toggle_skill_persists_and_resets_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Toggle "brandkit" (index 0) from enabled → disabled.
    app.toggle_skill(0).await.unwrap();

    if let Overlay::Skills(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
    } else {
        panic!("skills overlay vanished");
    }
    // Persisted to the store, and the engine is dropped so the next turn rebuilds.
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert_eq!(disabled, vec!["brandkit".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_skill(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_skills()
            .await
            .unwrap()
            .is_empty()
    );
}

#[test]
fn test_parse_skill_add_input() {
    use super::session_impl::parse_skill_add_input;
    // First token is the name; the rest is a free-text description.
    let (name, desc) = parse_skill_add_input("changelog Summarize the git log").unwrap();
    assert_eq!(name, "changelog");
    assert_eq!(desc, "Summarize the git log");
    // A bare name (no description) is fine — a placeholder is templated in.
    assert_eq!(
        parse_skill_add_input("solo").unwrap(),
        ("solo".to_string(), String::new())
    );
    // The first token is the name (so a name can't contain a space) and the
    // remainder is the description.
    let (name, desc) = parse_skill_add_input("multi word description here").unwrap();
    assert_eq!(name, "multi");
    assert_eq!(desc, "word description here");
    // A folder-unsafe single-token name is rejected.
    assert!(parse_skill_add_input("a.b desc").is_err(), "dot in name");
    assert!(parse_skill_add_input("").is_err(), "empty");
}

#[tokio::test]
async fn test_skills_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_skills_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Skills(_)),
        "bare /skills opens overlay"
    );

    // An unknown verb is a usage error.
    app.run_skills_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name is a usage error.
    app.run_skills_command(Some("rm".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent skill → "No skill" notice, no deletion.
    app.run_skills_command(Some("rm __aivo_no_such_skill__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No skill"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

#[test]
fn test_parse_agent_command() {
    assert_eq!(
        parse_slash_command("agent").unwrap(),
        SlashCommand::Agent(None)
    );
    assert_eq!(
        parse_slash_command("agent reviewer").unwrap(),
        SlashCommand::Agent(Some("reviewer".to_string()))
    );
}

/// `/agent` lists, selects (at chat start), clears, rejects unknowns, and is
/// blocked once a conversation has started. Discovers from a tempdir, never the
/// real `.aivo/agents` / `.claude/agents`.
#[tokio::test]
async fn test_agent_command_dispatch() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "aivo-agent-cmd-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let agents = dir.join(".aivo").join("agents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("reviewer.md"),
        "---\nname: reviewer\ndescription: Reviews a diff.\n---\nBe terse.\n",
    )
    .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = dir.to_string_lossy().into_owned();

    // Bare /agent lists what's available + the current (default).
    app.run_agent_command(None).await;
    let n = app.notice.as_ref().unwrap().1.clone();
    assert!(n.contains("reviewer"), "notice: {n}");
    assert!(n.contains("agent: default"), "notice: {n}");

    // Select a known agent at chat start.
    app.run_agent_command(Some("reviewer".to_string())).await;
    assert_eq!(app.active_agent.as_deref(), Some("reviewer"));
    assert!(app.notice.as_ref().unwrap().1.contains("reviewer"));

    // Reset to the built-in default agent via `default`.
    app.run_agent_command(Some("default".to_string())).await;
    assert!(app.active_agent.is_none());
    assert!(
        app.notice.as_ref().unwrap().1.contains("default"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );

    // Unknown name → error notice, stays cleared.
    app.run_agent_command(Some("ghost".to_string())).await;
    assert!(app.active_agent.is_none());
    assert!(
        app.notice.as_ref().unwrap().1.contains("no agent named"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );

    // Re-select, then a started conversation blocks any further switch.
    app.run_agent_command(Some("reviewer".to_string())).await;
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: Vec::new(),
    });
    app.run_agent_command(Some("default".to_string())).await;
    assert_eq!(
        app.active_agent.as_deref(),
        Some("reviewer"),
        "agent switch is blocked once a conversation has started"
    );
    assert!(app.notice.as_ref().unwrap().1.contains("/new"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_parse_goal_command() {
    assert_eq!(
        parse_slash_command("goal").unwrap(),
        SlashCommand::Goal(None)
    );
    assert_eq!(
        parse_slash_command("goal ship the feature").unwrap(),
        SlashCommand::Goal(Some("ship the feature".to_string()))
    );
    assert_eq!(
        parse_slash_command("goal stop").unwrap(),
        SlashCommand::Goal(Some("stop".to_string()))
    );
}

/// `/create-skill` is a first-class built-in command (in `SLASH_COMMANDS` and
/// `/help`), parses with an optional intent argument, and dispatches the embedded
/// create-skill instructions as a turn — shown compactly in the transcript.
#[tokio::test]
async fn test_create_skill_is_a_builtin_command() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // It's registered as a built-in command, so `/help` and the `/` menu list it.
    assert!(
        SLASH_COMMANDS.iter().any(|c| c.name == "create-skill"),
        "create-skill must be a built-in command"
    );
    // Parses bare and with an intent argument.
    assert_eq!(
        parse_slash_command("create-skill").unwrap(),
        SlashCommand::CreateSkill(None)
    );
    assert_eq!(
        parse_slash_command("create-skill a git-diff summarizer").unwrap(),
        SlashCommand::CreateSkill(Some("a git-diff summarizer".to_string()))
    );

    // Running it queues/sends the embedded instructions and shows the compact
    // command (not the inlined body) in the transcript.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_create_skill_command(Some("a git-diff summarizer".to_string()))
        .await
        .unwrap();
    let user = app
        .history
        .iter()
        .find(|m| m.role == "user")
        .expect("a user turn was dispatched");
    assert!(
        user.content.contains("create-skill") && user.content.contains("git-diff summarizer"),
        "the model receives the expanded instructions + intent"
    );
    // Regression: the body documents the literal `$ARGUMENTS` token, which must
    // survive verbatim — the command must NOT run it through `$ARGUMENTS`
    // substitution and splice the user's intent into the documentation.
    assert!(
        user.content.contains("$ARGUMENTS"),
        "the `$ARGUMENTS` doc token must not be substituted away"
    );

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("/create-skill"),
        "transcript should show the compact command:\n{screen}"
    );
}

/// `/goal` status/stop and the start guards — none of which send a turn.
#[tokio::test]
async fn test_goal_command_status_stop_and_guards() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Bare /goal with nothing active → usage notice, starts nothing.
    app.run_goal_command(None).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // stop when inactive → notice, still inactive.
    app.run_goal_command(Some("stop".to_string())).await;
    assert!(app.goal_mode.is_none());

    // Starting mid-turn is refused (no send).
    app.sending = true;
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("Wait"));
    app.sending = false;

    // A non-agent key (copilot) is refused (no send).
    app.key.base_url = "copilot".to_string();
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("native agent"));
}

/// The goal loop ends on the completion marker and at the iteration cap (both
/// terminal — neither sends another turn). Exercises `signals_goal_complete`.
#[tokio::test]
async fn test_goal_loop_stops_on_marker_and_cap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let assistant = |content: &str| ChatMessage {
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    // Marker on its own line → loop ends, complete notice.
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
    });
    app.history.push(assistant("all set.\nGOAL COMPLETE"));
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none(), "marker ends the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("complete"));

    // Prose merely mentioning the marker does NOT count (strict whole-line match).
    app.history.clear();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 20,
        max: 20,
    });
    app.history.push(assistant(
        "I will reply GOAL COMPLETE once everything is finished.",
    ));
    // At the cap with no real marker → loop ends with the cap notice (no send).
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none(), "cap ends the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("cap"));

    // Not in goal mode → no-op.
    app.maybe_continue_goal().await.unwrap();
    assert!(app.goal_mode.is_none());
}

/// The composer rule shows a live `/goal` step indicator (and the auto-approve
/// badge) while a goal loop runs, nothing goal-related when off, within width.
#[test]
fn test_composer_rule_shows_goal_step_indicator() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let off = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(!off.contains("goal"), "no goal badge when off: {off:?}");
    assert!(off.contains("auto-approve"));

    app.goal_mode = Some(GoalState {
        objective: "ship it".to_string(),
        iteration: 2,
        max: 20,
    });
    let on = plain_text_from_spans(&app.composer_rule_line(80).spans);
    assert!(on.contains("goal 2/20"), "goal step indicator: {on:?}");
    assert!(
        on.contains("auto-approve"),
        "auto-approve badge stays: {on:?}"
    );
    assert!(
        display_width(&on) <= 80,
        "rule fits width: {}",
        display_width(&on)
    );
}

#[tokio::test]
async fn test_skills_add_routes_source_not_scaffold() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A token that isn't a bare skill name (has `/` or `.`) routes to install-
    // from-source, not scaffold. A bad local path surfaces an install error
    // (and never scaffolds a literal `./…`-named skill).
    app.submit_skill_add("./aivo_no_such_skill_dir_zzz".to_string())
        .await
        .unwrap();
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(
        notice.contains("Failed to install") || notice.contains("not a directory"),
        "expected an install error, got: {notice}"
    );
}

#[tokio::test]
async fn test_skills_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without scaffolding.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("skills overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[tokio::test]
async fn test_skills_delete_arms_then_cancels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.pending_delete, Some(0), "first Ctrl+D should arm");
    } else {
        panic!("skills overlay vanished on arm");
    }
    // Esc cancels the arm rather than closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.pending_delete.is_none(), "Esc should disarm");
    } else {
        panic!("Esc on an armed delete must not close the overlay");
    }
}

#[tokio::test]
async fn test_shift_tab_toggles_auto_approve() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);
    // Shift+Tab arrives as BackTab — toggles auto-approve on.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "Shift+Tab should enable auto-approve"
    );
    // The shared LIVE flag the running agent turn reads tracks the toggle.
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve ON"
    );
    // The Tab+SHIFT form some terminals send toggles it back off.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert!(!app.agent_auto_approve, "Tab+SHIFT should disable it");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve OFF"
    );
    // Ctrl+O is no longer an auto-approve alias (Shift+Tab only).
    app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "Ctrl+O no longer toggles auto-approve"
    );
}

#[tokio::test]
async fn test_shift_tab_on_permission_card_approves_and_enables() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // A tool-permission card is up, holding the keyboard.
    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // Shift+Tab on the card must enable auto-approve AND approve THIS request —
    // not get swallowed (the old behavior, where only y/a/n worked).
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve, "toggle enables auto-approve");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the live flag the running turn reads is set"
    );
    assert!(app.agent_permission.is_none(), "the card is dismissed");
    assert_eq!(
        decision_rx.await.unwrap(),
        Decision::Allow,
        "the pending request is approved"
    );
}

/// "Always" on a Cursor permission card flips session-wide auto-approve, so the
/// composer badge and exit-persistence (which read `agent_auto_approve`) must
/// agree with the shared live flag — the badge can't say "off" while Cursor
/// silently allows the rest of the session.
#[tokio::test]
async fn test_always_on_cursor_card_enables_auto_approve() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("Cursor wants to run:\nedit main.rs".to_string()),
        reply,
    });

    // 'a' = always: cursor "always" is session-wide auto-approve.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "cursor 'always' flips the bool the badge/persistence read"
    );
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "and the shared live flag the cursor session reads"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// In-process-engine "always" stays scoped to that (tool,args): it must NOT flip
/// the global auto-approve badge on — only Cursor cards do that.
#[tokio::test]
async fn test_always_on_native_card_leaves_auto_approve_off() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "native 'always' is per-tool, not global auto-approve"
    );
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the global live flag stays off for native 'always'"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// A project `.mcp.json` server is not in the *base* opt-out set (the consent
/// gate that holds stdio servers back lives in `connect_mcp_with_consent`, not
/// here), and toggling a project row off in `/mcp` adds it to the global opt-out
/// list, exactly like a user server.
#[tokio::test]
async fn project_mcp_server_connects_by_default_and_toggles_like_user() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-proj-mcp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"fs":{"command":"echo"}}}"#,
    )
    .unwrap();
    app.real_cwd = repo.to_str().unwrap().to_string();

    // The base opt-out set is empty (the consent gate is applied separately).
    assert!(
        !app.effective_disabled_mcp_servers().await.contains("fs"),
        "a project .mcp.json server is not in the base opt-out set"
    );

    // Toggling a project row off goes to the global opt-out, like a user server.
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "fs".to_string(),
            status: "1 tool".to_string(),
            health: McpHealth::Connected,
            enabled: true,
            scope: ServerScope::Project,
            command: "echo".to_string(),
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(
        app.session_store.get_disabled_mcp_servers().await.unwrap(),
        vec!["fs".to_string()],
        "toggling a project row off adds it to the user opt-out list"
    );
    assert!(app.effective_disabled_mcp_servers().await.contains("fs"));

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn test_humanize_count() {
    use super::overlay_render_impl::humanize_count;
    assert_eq!(humanize_count(0), "0");
    assert_eq!(humanize_count(999), "999");
    assert_eq!(humanize_count(1234), "1.2k");
    assert_eq!(humanize_count(12345), "12k");
}

#[test]
fn test_sort_mcp_rows_problems_first() {
    use super::session_impl::sort_mcp_rows;
    use crate::agent::mcp::ServerScope;
    let row = |name: &str, health| McpServerRow {
        name: name.to_string(),
        status: String::new(),
        health,
        enabled: !matches!(health, McpHealth::Disabled),
        scope: ServerScope::User,
        command: "x".to_string(),
    };
    let mut rows = vec![
        row("zeta", McpHealth::Connected),
        row("off-one", McpHealth::Disabled),
        row("beta", McpHealth::Failed),
        row("needs", McpHealth::NeedsAuth),
        row("alpha", McpHealth::Connected),
        row("idle", McpHealth::Idle),
    ];
    sort_mcp_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Failed first, then needs-auth (actionable), then connected (alphabetical),
    // then idle, then disabled last.
    assert_eq!(
        order,
        vec!["beta", "needs", "alpha", "zeta", "idle", "off-one"]
    );
}

#[test]
fn test_sort_skill_rows_enabled_first() {
    use super::session_impl::sort_skill_rows;
    use crate::agent::skills::SkillScope;
    let row = |name: &str, enabled| SkillToggle {
        name: name.to_string(),
        description: String::new(),
        enabled,
        dir: std::path::PathBuf::from(name),
        scope: SkillScope::User,
        body: String::new(),
    };
    let mut rows = vec![
        row("zoff", false),
        row("able", true),
        row("aoff", false),
        row("baker", true),
    ];
    sort_skill_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Enabled first (alphabetical), disabled at the bottom (alphabetical).
    assert_eq!(order, vec!["able", "baker", "aoff", "zoff"]);
}

#[tokio::test]
async fn test_skills_filter_narrows_and_enter_toggles() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // brandkit(on), critique(off)

    // Typing 'c' filters to "critique" (index 1); the selection re-anchors to it.
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.query, "c");
        assert_eq!(
            state.filtered_indices(),
            vec![1],
            "only critique matches 'c'"
        );
        assert_eq!(state.selected, 1, "selection re-anchored to the match");
    } else {
        panic!("overlay vanished");
    }
    // Enter toggles the matched skill (critique off → on).
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert!(
        !disabled.contains(&"critique".to_string()),
        "Enter should have toggled critique on"
    );
    // Backspace clears the one-char filter.
    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.query.is_empty(), "Backspace should clear the filter");
    }
}

#[tokio::test]
async fn test_mcp_filter_narrows_selection() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // filesystem, github

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.filtered_indices(), vec![1], "only github matches 'g'");
        assert_eq!(state.selected, 1);
        assert!(state.has_selection());
    } else {
        panic!("overlay vanished");
    }
    // A query matching nothing leaves no visible selection (so Enter/Tab no-op).
    app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(
            state.filtered_indices().is_empty(),
            "no server matches 'gz'"
        );
        assert!(!state.has_selection());
    }
}

#[test]
fn test_skills_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some("changelog".to_string());
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The `+` prompt marks the add field; the typed buffer and the install hint
    // both show, and the footer offers save/cancel.
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(
        screen.contains("changelog"),
        "missing typed input:\n{screen}"
    );
    assert!(
        screen.contains("github:owner/repo"),
        "missing add hint:\n{screen}"
    );
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

fn mcp_overlay_fixture() -> McpOverlay {
    use crate::agent::mcp::ServerScope;
    McpOverlay {
        items: vec![
            McpServerRow {
                name: "filesystem".to_string(),
                status: "5 tools".to_string(),
                health: McpHealth::Connected,
                enabled: true,
                scope: ServerScope::User,
                command: "npx".to_string(),
            },
            McpServerRow {
                name: "github".to_string(),
                status: "off".to_string(),
                health: McpHealth::Disabled,
                enabled: false,
                scope: ServerScope::User,
                command: "docker".to_string(),
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

fn wheel(kind: MouseEventKind) -> MouseEvent {
    MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }
}

#[tokio::test]
async fn test_skills_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection (the list follows it).
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // 2 items, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the body, leaving the selection put.
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 3 && s.selected == 0));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 0));

    // Add-input mode: the wheel is ignored (no selection move, no scroll).
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some(String::new());
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0 && s.detail_scroll == 0));
}

#[tokio::test]
async fn test_mcp_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection.
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // 2 servers, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the tool list.
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Mcp(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.detail_scroll == 3 && s.selected == 0));
}

#[test]
fn test_composer_command_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let set = |app: &mut ChatTuiApp, draft: &str| {
        app.draft = draft.to_string();
        app.cursor = app.draft.len();
    };

    // Bare command (and a trailing space) → ghost hint with the arg syntax.
    set(&mut app, "/mcp");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add <command>")),
        "typing /mcp should ghost the add/rm syntax"
    );
    set(&mut app, "/mcp ");
    assert!(
        app.composer_command_hint().is_some(),
        "trailing space still bare"
    );
    // Once a real argument is typed, the hint clears.
    set(&mut app, "/mcp add");
    assert!(
        app.composer_command_hint().is_none(),
        "args typed → no ghost"
    );

    // Other argument-taking commands ghost their arg syntax too.
    for (draft, expect) in [
        ("/model", "[name]"),
        ("/key", "[id|name]"),
        ("/resume", "[query]"),
        ("/copy", "[n]"),
        ("/detach", "<n>"),
    ] {
        set(&mut app, draft);
        assert_eq!(
            app.composer_command_hint(),
            Some(expect),
            "{draft} should ghost {expect}"
        );
    }
    // `/skills` ghosts its add/rm subcommand syntax, like `/mcp`.
    set(&mut app, "/skills");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add <name>")),
        "/skills should ghost the add/rm syntax"
    );

    // `/attach` is deliberately hint-less: typing `/attach ` opens path
    // completion, and a ghost would suppress that menu.
    set(&mut app, "/attach");
    assert!(
        app.composer_command_hint().is_none(),
        "attach must not ghost (path menu owns it)"
    );
    // A no-argument command has no hint either.
    set(&mut app, "/new");
    assert!(app.composer_command_hint().is_none());

    // None for a literal slash or plain text.
    set(&mut app, "//mcp");
    assert!(app.composer_command_hint().is_none());
    set(&mut app, "hello");
    assert!(app.composer_command_hint().is_none());
    // Cursor not at the end → no ghost (e.g. editing mid-line).
    app.draft = "/mcp".to_string();
    app.cursor = 2;
    assert!(app.composer_command_hint().is_none());
}

#[test]
fn test_mcp_ghost_hint_renders_in_composer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mcp".to_string();
    app.cursor = app.draft.len();

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The command and its ghost arg-hint share the composer line.
    assert!(
        screen.contains("/mcp [add <command>"),
        "composer should show the inline ghost hint:\n{screen}"
    );
}

#[test]
fn test_mcp_overlay_renders_server_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("MCP servers"), "missing title:\n{screen}");
    assert!(
        screen.contains("filesystem"),
        "missing server name:\n{screen}"
    );
    // The status renders on its own line under the server name.
    assert!(screen.contains("5 tools"), "missing status:\n{screen}");
    assert!(
        screen.contains("[✓]") && screen.contains("[ ]"),
        "missing checkboxes:\n{screen}"
    );
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
}

#[test]
fn test_mcp_overlay_renders_detail_line_for_selected() {
    use crate::agent::mcp::ServerScope;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Select a disabled, project-scoped server: with no live client its detail
    // line should still spell out the scope and the (actionable) disabled state.
    let mut overlay = mcp_overlay_fixture();
    overlay.items[1].scope = ServerScope::Project;
    overlay.selected = 1; // "github", off
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project (.mcp.json)"),
        "detail line should tag a project-scoped server:\n{screen}"
    );
    assert!(
        screen.contains("disabled"),
        "detail line should show the disabled state:\n{screen}"
    );
}

/// A per-server progress event mid-connect flips just that server's row to its
/// resolved status; other still-connecting servers keep reading "connecting…".
/// A stale-generation event is ignored.
#[tokio::test]
async fn test_mcp_progress_flips_only_resolved_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture(); // filesystem, github
    // Both enabled and mid-connect.
    for item in &mut overlay.items {
        item.enabled = true;
        item.status = "connecting…".to_string();
        item.health = McpHealth::Idle;
    }
    app.overlay = Overlay::Mcp(overlay);
    app.mcp_connecting = true;
    app.mcp_client = None;

    // "filesystem" resolves; "github" hasn't yet.
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "filesystem".to_string(),
            status: "5 tools".to_string(),
            health: McpHealth::Connected,
            generation: app.mcp_connect_gen,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    let row = |app: &ChatTuiApp, name: &str| -> (String, McpHealth) {
        match &app.overlay {
            Overlay::Mcp(s) => s
                .items
                .iter()
                .find(|i| i.name == name)
                .map(|i| (i.status.clone(), i.health))
                .unwrap(),
            _ => panic!("overlay vanished"),
        }
    };
    assert_eq!(
        row(&app, "filesystem"),
        ("5 tools".to_string(), McpHealth::Connected),
        "resolved server should flip to its tool count"
    );
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "unresolved server should still read connecting"
    );

    // A stale-generation event (a connect superseded by a toggle) is dropped.
    let stale = app.mcp_connect_gen.wrapping_add(7);
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "github".to_string(),
            status: "9 tools".to_string(),
            health: McpHealth::Connected,
            generation: stale,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "a stale-generation progress event must be ignored"
    );
}

#[tokio::test]
async fn test_toggle_mcp_server_persists_and_resets() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Disable "filesystem" (index 0, currently enabled).
    app.toggle_mcp_server(0).await.unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
        assert_eq!(state.items[0].status, "off");
    } else {
        panic!("mcp overlay vanished");
    }
    let disabled = app.session_store.get_disabled_mcp_servers().await.unwrap();
    assert_eq!(disabled, vec!["filesystem".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_mcp_server(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap()
            .is_empty()
    );
}

/// Toggling a server keeps the live client (rather than nulling it) so the
/// servers that *aren't* being toggled keep serving their status during the
/// reconnect — and bumps the generation so the reconnect supersedes any in-flight
/// one. (The connection-level reuse itself is covered by mcp's
/// `reconnect_reuses_live_servers`.)
#[tokio::test]
async fn test_toggle_preserves_live_client() {
    let empty_dir = tempfile::tempdir().unwrap();
    let live = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(
            empty_dir.path(),
            &std::collections::HashSet::new(),
        )
        .await,
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());
    app.mcp_client = Some(live);
    let gen0 = app.mcp_connect_gen;

    app.toggle_mcp_server(0).await.unwrap();

    assert!(
        app.mcp_client.is_some(),
        "toggle must keep the live client (for the other servers' status), not null it"
    );
    assert_ne!(
        app.mcp_connect_gen, gen0,
        "generation should advance so the reconnect supersedes any stale one"
    );
    assert!(app.mcp_connecting, "a reconnect should be in flight");
}

#[test]
fn test_parse_mcp_add_input() {
    use super::session_impl::parse_mcp_add_input;
    // No name — the first token is the command; a shell-quoted path survives.
    let (command, args) = parse_mcp_add_input("npx -y srv \"/a b/c\"").unwrap();
    assert_eq!(command, "npx");
    assert_eq!(args, vec!["-y", "srv", "/a b/c"]);
    // A bare command with no args is fine.
    assert_eq!(
        parse_mcp_add_input("my-mcp-binary").unwrap(),
        ("my-mcp-binary".to_string(), Vec::<String>::new())
    );
    // Empty input is a usage error.
    assert!(parse_mcp_add_input("   ").is_err());
}

#[tokio::test]
async fn test_mcp_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without writing config.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("mcp overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[test]
fn test_mcp_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.adding = Some("fs npx".to_string());
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(screen.contains("fs npx"), "missing typed input:\n{screen}");
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

#[tokio::test]
async fn test_mcp_drill_in_tab_then_esc() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Tab drills into the selected server's details.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.viewing, Some(0), "Tab should open the detail view");
    } else {
        panic!("overlay closed on Tab instead of drilling in");
    }
    // Esc backs out to the list, NOT closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.viewing.is_none(), "Esc should back out of detail");
    } else {
        panic!("Esc in detail closed the overlay instead of returning to the list");
    }
}

#[test]
fn test_mcp_drill_in_renders_command_and_footer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0); // "filesystem", command "npx"
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("command:"),
        "detail missing command:\n{screen}"
    );
    assert!(
        screen.contains("npx"),
        "detail missing the command value:\n{screen}"
    );
    assert!(
        screen.contains("Esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// The drill-in tool list stacks each name over its wrapped description (no
/// far-right column): a `•` name line, then `    `-indented description lines,
/// with a blank line separating tools and long descriptions wrapping.
#[test]
fn test_mcp_tool_lines_stacks_name_over_wrapped_desc() {
    use super::overlay_render_impl::mcp_tool_lines;

    let line_text =
        |line: &Line| -> String { line.spans.iter().map(|s| s.content.as_ref()).collect() };
    let tools = [
        ("short_tool", "A brief description."),
        (
            "browserslist_compatibility_check",
            "Check web feature compatibility against your browserslist configuration across many supported browsers.",
        ),
    ];
    let lines: Vec<String> = mcp_tool_lines(&tools, 40).iter().map(line_text).collect();

    // Each name is on its own bulleted line.
    assert!(
        lines.iter().any(|l| l == "  • short_tool"),
        "first tool name not on its own line:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "  • browserslist_compatibility_check"),
        "second tool name not on its own line:\n{lines:#?}"
    );
    // The description sits indented beneath the name (not in a right-hand column).
    assert!(
        lines.iter().any(|l| l == "    A brief description."),
        "description not indented under the name:\n{lines:#?}"
    );
    // A blank line separates the two tools.
    assert!(
        lines.iter().any(|l| l.is_empty()),
        "expected a blank separator between tools:\n{lines:#?}"
    );
    // The long description wraps onto multiple indented lines, none over width.
    let desc_lines = lines
        .iter()
        .filter(|l| l.starts_with("    ") && l.contains("browserslist") || l.contains("supported"))
        .count();
    assert!(
        desc_lines >= 2,
        "long description should wrap to multiple lines:\n{lines:#?}"
    );
    assert!(
        lines.iter().all(|l| display_width(l) <= 40),
        "no rendered line should exceed the width:\n{lines:#?}"
    );
}

#[test]
fn test_skill_drill_in_renders_body_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0); // "brandkit", body "Step 1. Render the boards."
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Instructions:"),
        "detail missing body header:\n{screen}"
    );
    assert!(
        screen.contains("Render the boards"),
        "detail missing the SKILL.md body:\n{screen}"
    );
    assert!(
        screen.contains("Esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// A long SKILL.md body is scrollable in the drill-in: the top hides later lines,
/// End reveals the last line (clamped by the renderer), and Esc resets the scroll.
#[tokio::test]
async fn test_skill_drill_in_scrolls_long_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.items[0].body = (1..=60)
        .map(|i| format!("Line number {i} of the instructions"))
        .collect::<Vec<_>>()
        .join("\n");
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);

    let render_screen = |app: &mut ChatTuiApp| -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut s = String::new();
        for y in 0..20u16 {
            for x in 0..80u16 {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    };

    let top = render_screen(&mut app);
    assert!(
        top.contains("Line number 1 of"),
        "top hides first line:\n{top}"
    );
    assert!(
        !top.contains("Line number 60 of"),
        "last line should be off-screen at the top:\n{top}"
    );
    assert!(
        top.contains("scroll"),
        "a scrollable body shows the scroll hint:\n{top}"
    );

    // End jumps to the bottom (the renderer clamps the offset).
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let bottom = render_screen(&mut app);
    assert!(
        bottom.contains("Line number 60 of"),
        "End reveals the last line:\n{bottom}"
    );
    match &app.overlay {
        Overlay::Skills(s) => assert!(s.detail_scroll > 0, "scroll offset advanced"),
        _ => panic!("overlay vanished"),
    }

    // Esc backs out of the drill-in and resets the scroll.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Skills(s) => {
            assert!(s.viewing.is_none(), "Esc leaves the drill-in");
            assert_eq!(s.detail_scroll, 0, "scroll resets on back-out");
        }
        _ => panic!("overlay vanished"),
    }
}

#[test]
fn test_restore_cancelled_submission_puts_prompt_back() {
    let mut history = vec![ChatMessage {
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    }];
    let mut draft = String::new();
    let mut draft_attachments = Vec::new();
    let mut pending_submit = Some(PendingSubmission {
        content: "draft".to_string(),
        attachments: Vec::new(),
    });

    restore_cancelled_submission(
        &mut history,
        &mut draft,
        &mut draft_attachments,
        &mut pending_submit,
    );

    assert!(history.is_empty());
    assert_eq!(draft, "draft");
    assert!(draft_attachments.is_empty());
    assert!(pending_submit.is_none());
}

#[test]
fn test_prepare_submit_action_bang_runs_shell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!ls -la".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Shell(cmd)) if cmd == "ls -la"
    ));
}

#[test]
fn test_prepare_submit_action_double_bang_escapes_to_literal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!!important".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(text)) if text == "!important"
    ));
}

#[test]
fn test_prepare_submit_action_bare_bang_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!   ".to_string();

    assert!(app.prepare_submit_action().is_err());
}

// Unix-only: drives the `!cmd` PTY with POSIX commands (`printf`) that the
// Windows shell (PowerShell) doesn't provide, and a PTY read that blocks until
// the child exits would stall the tokio runtime drop on Windows. The `!cmd`
// feature itself is cross-platform; only this Unix-command assertion is gated.
#[cfg(unix)]
#[tokio::test]
async fn test_run_local_command_is_display_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf hi".to_string());
    run_local_command_to_completion(&mut app).await;

    // A transcript step is recorded for display once the run finishes.
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"command\":\"printf hi\""));
    assert!(step.content.contains("hi"));

    // It is purely local: the `local_command` role is excluded from the model
    // context (only user/assistant turns are sent), so nothing reaches the server.
    let sent_to_model = app
        .history
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .count();
    assert_eq!(sent_to_model, 0);
}

// Unix-only: see `test_run_local_command_is_display_only` — POSIX `printf`
// through the PTY isn't portable to the Windows shell.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf 'a\\nb\\nc\\n'".to_string());
    // The run is live (not yet in history) right after starting.
    assert!(app.local_command.is_some());
    assert!(app.history.is_empty());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once, with the streamed output and a zero exit code.
    assert!(app.local_command.is_none());
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[cfg(unix)]
#[tokio::test]
async fn test_local_command_neutralizes_pager() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // A pager-spawning command (`git diff`) would launch `less` under the PTY and
    // hang waiting for a keypress we never send. The spawn env disables pagers, so
    // the child sees PAGER/GIT_PAGER=cat — verify that reaches it through the PTY.
    app.start_local_command("echo \"p=[$PAGER] g=[$GIT_PAGER]\"".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert!(
        step.content.contains("p=[cat] g=[cat]"),
        "pager env not injected into the PTY child: {}",
        step.content
    );
}

// Unix-only: `yes` (the infinite-flood command this caps) has no PowerShell
// equivalent, and the PTY drive is Unix-oriented like the sibling tests above.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_caps_huge_output() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `yes` floods forever — the reader must cap, kill it, and still finish.
    app.start_local_command("yes aivo".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"truncated\":true"));
    let stdout = serde_json::from_str::<serde_json::Value>(&step.content)
        .unwrap()
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .lines()
        .count();
    // Bounded by the capture cap, not unbounded.
    assert!(stdout <= 1000, "captured {stdout} lines, expected ≤ 1000");

    // A run we killed at the cap must NOT render a scary `[exited -1]` — the
    // "truncated" note explains the stop; the SIGKILL status is ours, not `yes`'s.
    let mut block = Vec::new();
    render_local_command(&mut block, &step.content);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !rendered.contains("[exited"),
        "truncated run should not show an exit code:\n{rendered}"
    );
    assert!(
        rendered.contains("truncated"),
        "truncated run should show the truncated note:\n{rendered}"
    );
}

#[tokio::test]
async fn test_local_command_full_output_kept_for_pager() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 250 output lines: past the 40-line display cap AND the persisted preview.
    let full: String = (1..=250).map(|i| format!("{i}\n")).collect();
    let total =
        app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    assert_eq!(total, 250);

    // The committed transcript entry keeps only a bounded preview…
    let step = app.history.last().expect("history entry");
    let decoded: serde_json::Value = serde_json::from_str(&step.content).unwrap();
    let persisted = decoded["stdout"].as_str().unwrap().lines().count();
    assert!(
        persisted <= MAX_PERSISTED_OUTPUT_LINES,
        "persisted {persisted} lines, expected ≤ {MAX_PERSISTED_OUTPUT_LINES}"
    );
    // …but records the TRUE total, so the transcript's "+N more" stays honest.
    assert_eq!(decoded["total_lines"].as_u64(), Some(250));

    // The full output is retained in memory for the ctrl+o pager (all 250 lines),
    // never persisted into history.
    let kept = app
        .last_local_output
        .as_ref()
        .expect("full output retained");
    assert_eq!(kept.stdout.lines().count(), 250);
}

#[test]
fn test_render_local_command_marker_counts_total_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let _app = make_test_app(tx, rx);
    // A committed entry stores a small preview but the true total in `total_lines`.
    let content = serde_json::json!({
        "command": "find .",
        "stdout": "a\nb\nc\n",
        "stderr": "",
        "exit_code": 0,
        "total_lines": 41243,
    })
    .to_string();
    let mut block = Vec::new();
    render_local_command(&mut block, &content);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    // Marker reflects the true total (41243 − 3 shown), not the 3 persisted lines.
    assert!(
        rendered.contains("+41240 more lines"),
        "marker should count the true total:\n{rendered}"
    );
}

#[tokio::test]
async fn test_ctrl_o_pager_reveals_elided_tail() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let full: String = (1..=250).map(|i| format!("{i}\n")).collect();
    app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);

    // ctrl+o opens the scrollable pager at the top.
    app.open_output_overlay();
    assert!(matches!(app.overlay, Overlay::Output { scroll: 0 }));

    // Scroll to the bottom and render: the pager surfaces line 250 — far past the
    // 40-line transcript cap — proving the elided tail is now viewable.
    if let Overlay::Output { scroll } = &mut app.overlay {
        *scroll = u16::MAX;
    }
    let mut terminal = Terminal::new(TestBackend::new(80, 40)).unwrap();
    terminal
        .draw(|frame| {
            app.render(frame);
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..40 {
        for x in 0..80 {
            screen.push_str(buf[(x, y)].symbol());
        }
    }
    assert!(
        screen.contains("250"),
        "pager should reveal the elided tail (line 250):\n{screen}"
    );
}

#[test]
fn test_open_output_overlay_without_output_is_noop() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_output_overlay();
    // No command has run, so the overlay stays closed and a hint is shown instead.
    assert!(matches!(app.overlay, Overlay::None));
    assert!(app.notice.is_some());
}

#[test]
fn test_local_command_long_line_wraps_in_full() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long = "./target/release/.fingerprint/typenum-2abcdef0123456789-very-long/lib-typenum.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let width: u16 = 58;
    let body = app.build_transcript_history_body(width);
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, width);

    // No row overflows the width (so ratatui's wrap-OFF render can't clip)…
    for row in &wrapped.rows {
        assert!(
            row_display_width(row) <= width,
            "row exceeds width: {row:?}"
        );
    }
    // …the long path is shown IN FULL — wrapped onto an extra row, not truncated.
    // Stitch every row (the path has no spaces, so dropping spaces rejoins it).
    let stitched: String = wrapped.rows.iter().map(|r| r.replace(' ', "")).collect();
    assert!(
        stitched.contains(long),
        "long path not shown in full across wrapped rows:\n{stitched}"
    );
    assert!(
        !stitched.contains('…'),
        "output should not be per-line truncated with an ellipsis"
    );
    let path_rows = wrapped
        .rows
        .iter()
        .filter(|r| r.contains("target") || r.contains("typenum"))
        .count();
    assert!(
        path_rows >= 2,
        "the long line should wrap onto multiple rows"
    );
}

#[test]
fn test_render_main_local_command_no_clip() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long =
        "./target/release/.fingerprint/typenum-2ebc5dae76d28bAAAAAA/lib-typenum-bbbbbbbbbbbb.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let (w, h) = (60u16, 20u16);
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();

    // Through the FULL render pipeline (cache → wrap → slice → paint), the path is
    // shown in full — wrapped across rows, never truncated (`…`) nor edge-clipped
    // (which would drop the tail). The path has no spaces, so stitching the whole
    // screen with spaces and the accent-bar glyph removed rejoins it intact.
    let mut screen = String::new();
    for y in 0..h {
        for x in 0..w {
            screen.push_str(buf[(x, y)].symbol());
        }
    }
    let compact: String = screen
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '▌')
        .collect();
    assert!(
        compact.contains(long),
        "path not shown in full (clipped/truncated):\n{screen}"
    );
    assert!(
        !screen.contains('…'),
        "output should wrap in full, not truncate with an ellipsis:\n{screen}"
    );
}

/// Drive a started `!cmd` to completion: drain runtime events until the run
/// commits to history (clearing `local_command`). Bounded so a `!cmd` that never
/// finishes (the Windows ConPTY hang this guards against) fails the test instead of
/// hanging the runner forever.
#[cfg(any(unix, windows))]
async fn run_local_command_to_completion(app: &mut ChatTuiApp) {
    for _ in 0..5000 {
        app.handle_runtime_events().await.unwrap();
        if app.local_command.is_none() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    panic!("local command did not finish in time");
}

// Windows counterpart of `test_local_command_streams_then_commits`: `!cmd` runs
// through plain pipes (not ConPTY), whose output EOFs when the child exits — so the
// run must stream output AND commit (clear `local_command`). The bug this guards
// against left every `!cmd` stuck at "running…" forever. Uses a PowerShell command
// since `bare_shell` spawns PowerShell on Windows.
#[cfg(windows)]
#[tokio::test]
async fn test_local_command_pipe_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("Write-Output a; Write-Output b; Write-Output c".to_string());
    assert!(app.local_command.is_some());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once (run finished), with the streamed output and exit 0.
    assert!(app.local_command.is_none(), "the run must finish, not hang");
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[test]
fn test_prepare_submit_action_allows_attachment_only_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "notes.md".to_string(),
        mime_type: "text/markdown".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./notes.md".to_string(),
        },
    });

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(input)) if input.is_empty()
    ));
}

#[test]
fn test_detach_attachment_removes_selected_item() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments = vec![
        MessageAttachment {
            name: "one.txt".to_string(),
            mime_type: "text/plain".to_string(),
            storage: AttachmentStorage::FileRef {
                path: "./one.txt".to_string(),
            },
        },
        MessageAttachment {
            name: "two.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::FileRef {
                path: "./two.png".to_string(),
            },
        },
    ];

    app.detach_attachment(2).unwrap();

    assert_eq!(app.draft_attachments.len(), 1);
    assert_eq!(app.draft_attachments[0].name, "one.txt");
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Removed image: two.png")
    );
}

#[tokio::test]
async fn test_submit_draft_keeps_failed_attach_command_and_shows_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/attach ./definitely-missing-file.txt".to_string();
    app.cursor = app.draft.len();

    let should_exit = app.submit_draft().await.unwrap();

    assert!(!should_exit);
    assert_eq!(app.draft, "/attach ./definitely-missing-file.txt");
    assert!(app.draft_attachments.is_empty());
    assert!(
        app.notice.as_ref().is_some_and(
            |(color, text)| *color == ERROR && text.contains("Failed to read attachment")
        )
    );
}

#[test]
fn test_composer_attachment_lines_show_indices() {
    let lines = composer_attachment_lines(&[MessageAttachment {
        name: "hi.css".to_string(),
        mime_type: "text/css".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./hi.css".to_string(),
        },
    }]);
    let plain = plain_text_from_spans(&lines[0].spans);
    assert_eq!(plain, "· 1. [file] hi.css");
}

#[test]
fn test_prepare_for_model_picker_cancels_inflight_request() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "draft".to_string(),
        attachments: Vec::new(),
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.prepare_for_model_picker();

    assert!(!app.sending);
    assert!(app.pending_response.is_empty());
    assert_eq!(app.draft, "draft");
    assert!(app.history.is_empty());
    assert!(app.request_started_at.is_none());
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Request cancelled")
    );
}

#[tokio::test]
async fn test_cancel_keeps_user_turn_for_in_process_agent_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "edit the config".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "edit the config".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    // Mark an in-process agent turn as in flight (its per-turn serve is up).
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.cancel_inflight_request();

    // The engine already consumed this turn (and may have edited files), so the
    // request stays in the transcript instead of being silently un-sent — unlike
    // the plain-chat path (see test_prepare_for_model_picker_cancels_inflight_request).
    assert_eq!(app.history.len(), 1, "agent user turn must be kept");
    assert_eq!(app.history[0].content, "edit the config");
    assert!(
        app.draft.is_empty(),
        "an agent turn must not be restored to the composer"
    );
    assert!(app.pending_submit.is_none());
    assert!(!app.sending);
}

#[tokio::test]
async fn test_interrupt_inflight_request_keeps_partial_response() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "session-123".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "draft".to_string(),
        attachments: Vec::new(),
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert!(app.pending_response.is_empty());
    assert!(app.pending_submit.is_none());
    assert!(app.draft.is_empty());
    assert_eq!(app.history.len(), 2);
    assert_eq!(app.history[1].role, "assistant");
    assert_eq!(app.history[1].content, "partial");
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Response interrupted")
    );
}

/// Watchdog: a task that finished WITHOUT a terminal event (a `run_turn` panic
/// before `ui.footer`) must not leave the turn stuck "sending"; it salvages
/// partial text, resets, and stops the `/goal` loop. A running turn is untouched.
#[tokio::test]
async fn test_recover_dead_response_task_resets_stuck_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "do it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    app.goal_mode = Some(GoalState {
        objective: "do it".to_string(),
        iteration: 1,
        max: 20,
    });
    // A finished task that sent NO terminal event (stands in for a panic).
    let dead = tokio::spawn(async {});
    for _ in 0..100 {
        if dead.is_finished() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(dead.is_finished(), "spawned task should have completed");
    app.response_task = Some(dead);

    let recovered = app.recover_dead_response_task().await.unwrap();
    assert!(recovered, "a dead, still-sending turn must be recovered");
    assert!(!app.sending, "sending must be reset");
    assert!(app.response_task.is_none());
    assert!(
        app.goal_mode.is_none(),
        "goal loop must stop, not auto-continue into a likely repeat"
    );
    let last = app.history.last().unwrap();
    assert_eq!(last.role, "assistant");
    assert_eq!(last.content, "partial");
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("goal mode stopped"), "{notice}");

    // A healthy in-flight turn (task still running) is left strictly alone.
    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
    let mut app2 = make_test_app(tx2, rx2);
    app2.sending = true;
    app2.response_task = Some(tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
    }));
    let recovered2 = app2.recover_dead_response_task().await.unwrap();
    assert!(!recovered2, "a running turn must not be touched");
    assert!(app2.sending, "a running turn stays sending");
    if let Some(task) = app2.response_task.take() {
        task.abort();
    }
}

/// Stopping a turn (cancel / empty interrupt / partial interrupt) must exit any
/// `/goal` loop, so `maybe_continue_goal` can't auto-continue after the user stops.
#[tokio::test]
async fn test_stopping_a_turn_clears_goal_mode() {
    let armed = || {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        app.sending = true;
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 1,
            max: 20,
        });
        app
    };

    let mut app = armed();
    app.cancel_inflight_request();
    assert!(app.goal_mode.is_none(), "cancel must clear goal mode");

    // Interrupt with NO partial response routes through cancel.
    let mut app = armed();
    app.interrupt_inflight_request().await.unwrap();
    assert!(
        app.goal_mode.is_none(),
        "empty interrupt must clear goal mode"
    );
    assert_eq!(app.notice.as_ref().unwrap().1, "Goal mode stopped");

    // Interrupt WITH a partial response takes the salvage path.
    let mut app = armed();
    app.pending_response = "half a reply".to_string();
    app.interrupt_inflight_request().await.unwrap();
    assert!(
        app.goal_mode.is_none(),
        "partial interrupt must clear goal mode"
    );
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("goal mode stopped"), "{notice}");
}

#[test]
fn test_empty_composer_placeholder_reserves_cursor_cell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let line = app.render_composer_text().lines[0].clone();
    let plain = plain_text_from_spans(&line.spans);
    assert_eq!(plain, ">  Ask anything · / for commands");
}

#[test]
fn test_overlay_hides_input_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.should_show_input_cursor());

    app.overlay = Overlay::Picker(Box::new(PickerState::loading(
        "Select model",
        String::new(),
        PickerKind::Model {
            target: ModelSelectionTarget::CurrentChat,
            auto_accept_exact: false,
        },
    )));

    assert!(!app.should_show_input_cursor());
}

#[test]
fn test_persisted_draft_history_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("chat_history");
    let history = vec!["first".to_string(), "second".to_string()];

    save_persisted_draft_history_to_path(&path, &history).unwrap();

    assert_eq!(load_persisted_draft_history_from_path(&path), history);
}

#[test]
fn test_session_preview_uses_last_user_message() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: session_title_from_messages(
            &[
                ChatMessage {
                    role: "assistant".to_string(),
                    content: "Hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "What is the deployment status for api gateway?".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
            "claude",
        ),
        preview_text: "What is the deployment status for api gateway?".to_string(),
    };

    assert_eq!(
        preview.title,
        "What is the deployment status for api gateway?".to_string()
    );
}

#[test]
fn test_session_preview_text_uses_two_latest_turns() {
    let preview = session_preview_text_from_messages(
        &[
            ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "hi there".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
        ],
        "claude",
    );

    assert_eq!(preview, "hello · hi there");
}

#[test]
fn test_resume_metadata_spans_drop_labels_and_id() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };

    let plain = plain_text_from_spans(&resume_metadata_spans(&preview, 40));
    assert!(plain.contains("2h"));
    assert!(plain.contains("prod"));
    assert!(plain.contains("claude"));
    assert!(!plain.contains("time"));
    assert!(!plain.contains("key"));
    assert!(!plain.contains("model"));
    assert!(!plain.contains("session-1"));
}

#[test]
fn test_session_picker_item_line_shows_two_turn_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text:
            "What is the deployment status for api gateway after the canary rollout finished?"
                .to_string(),
    };

    let lines = session_picker_item_lines(&preview, false, false, 32);
    let first = plain_text_from_spans(&lines[0].spans);

    assert!(first.contains("What is"));
    assert!(first.chars().any(|ch| ch.is_ascii_digit()));
    assert!(!first.contains("key"));
}

#[tokio::test]
async fn test_begin_resume_load_clears_transcript_before_result() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "old".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "pending".to_string();
    app.draft = "draft".to_string();
    let preview = SessionPreview {
        key_id: app.key.id.clone(),
        key_name: app.key.display_name().to_string(),
        base_url: app.key.base_url.clone(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };

    app.begin_resume_load(preview.clone());

    assert!(app.history.is_empty());
    assert!(app.pending_response.is_empty());
    assert!(app.draft.is_empty());
    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.title.clone()),
        Some(preview.title)
    );
}

fn one_user_message(content: &str) -> Vec<crate::services::session_store::StoredChatMessage> {
    vec![crate::services::session_store::StoredChatMessage {
        role: "user".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        id: None,
        timestamp: None,
        attachments: None,
    }]
}

#[tokio::test]
async fn test_empty_chat_persists_no_session_on_exit() {
    // Opening `aivo chat` and leaving without saying anything must NOT create a
    // session — `flush_for_exit` only persists a non-empty history, so an
    // untouched chat leaves the resume list untouched.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "untouched-sess".to_string();
    app.raw_model = "claude".to_string();
    assert!(app.history.is_empty());

    app.flush_for_exit().await;

    assert_eq!(store.count_chat_sessions().await, 0);
    assert!(
        store
            .get_chat_session("untouched-sess")
            .await
            .unwrap()
            .is_none()
    );
    // Nothing to resume, so no exit hint either.
    assert_eq!(app.resumable_session_id(), None);
}

#[tokio::test]
async fn test_resume_last_jumps_to_newest_from_fresh_launch() {
    // `aivo chat --resume last` from a fresh process (empty history) reopens the
    // most recent saved chat directly — the exit hint's round-trip.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    store
        .save_chat_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "older-sess",
            "claude",
            None,
            &one_user_message("older"),
            "older",
            "older",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();
    // Guarantee a strictly-later updated_at for the second save.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    store
        .save_chat_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "newer-sess",
            "claude",
            None,
            &one_user_message("newer"),
            "newer",
            "newer",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.open_resume_picker(Some("last".to_string()))
        .await
        .unwrap();

    assert!(
        matches!(app.overlay, Overlay::None),
        "`last` should resume directly, not open the picker"
    );
    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.session_id.clone()),
        Some("newer-sess".to_string()),
    );
}

#[tokio::test]
async fn test_resume_last_in_session_skips_current_chat() {
    // `/resume last` mid-conversation lands on the PREVIOUS chat, not a reload of
    // the one you're already in (which sorts newest after being persisted).
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    store
        .save_chat_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/demo",
            "prev-sess",
            "claude",
            None,
            &one_user_message("previous"),
            "previous",
            "previous",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "current-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "live conversation".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.open_resume_picker(Some("last".to_string()))
        .await
        .unwrap();

    assert_eq!(
        app.loading_resume
            .as_ref()
            .map(|loading| loading.preview.session_id.clone()),
        Some("prev-sess".to_string()),
        "in-session `last` should skip the current chat"
    );
}

#[tokio::test]
async fn test_open_resume_picker_saves_current_unsaved_session() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "fresh-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "hello from a new chat".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.open_resume_picker(None).await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected session picker");
    };
    assert!(
        picker.items.iter().any(|item| {
            matches!(
                &item.value,
                PickerValue::Session(session) if session.session_id == "fresh-session"
            )
        }),
        "current unsaved session should be listed"
    );

    let saved = store
        .get_chat_session("fresh-session")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(saved.session_id, "fresh-session");
}

#[tokio::test]
async fn test_delete_picker_selection_removes_saved_chat() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    store
        .save_chat_session_with_id(
            &key_id,
            "https://api.example.com",
            "/tmp/demo",
            "session-1234",
            "claude",
            None,
            &[
                crate::services::session_store::StoredChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                },
                crate::services::session_store::StoredChatMessage {
                    role: "assistant".to_string(),
                    content: "hi there".to_string(),
                    reasoning_content: None,
                    id: None,
                    timestamp: None,
                    attachments: None,
                },
            ],
            "hello",
            "hello · hi there",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();

    let preview = SessionPreview {
        key_id: key_id.clone(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hello".to_string(),
        preview_text: "hello · hi there".to_string(),
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.cwd = "/tmp/demo".to_string();
    app.overlay = Overlay::Picker(Box::new(PickerState::ready(
        "Sessions",
        String::new(),
        vec![PickerEntry {
            label: preview.title.clone(),
            search_text: preview.search_text(),
            value: PickerValue::Session(preview),
        }],
        PickerKind::Session,
    )));

    app.delete_picker_selection(0).await.unwrap();

    assert!(matches!(app.overlay, Overlay::None));
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("Saved chat deleted")
    );
    let saved = app
        .session_store
        .get_chat_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_none());
}

#[tokio::test]
async fn test_ctrl_d_requires_confirmation_before_delete() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    store
        .save_chat_session_with_id(
            &key_id,
            "https://api.example.com",
            "/tmp/demo",
            "session-1234",
            "claude",
            None,
            &[crate::services::session_store::StoredChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            }],
            "hello",
            "hello",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();

    let preview = SessionPreview {
        key_id: key_id.clone(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hello".to_string(),
        preview_text: "hello".to_string(),
    };

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.cwd = "/tmp/demo".to_string();
    app.overlay = Overlay::Picker(Box::new(PickerState::ready(
        "Sessions",
        String::new(),
        vec![PickerEntry {
            label: preview.title.clone(),
            search_text: preview.search_text(),
            value: PickerValue::Session(preview),
        }],
        PickerKind::Session,
    )));

    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let saved = app
        .session_store
        .get_chat_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_some());
    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected picker overlay");
    };
    assert!(picker.pending_delete.is_some());

    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let saved = app
        .session_store
        .get_chat_session("session-1234")
        .await
        .unwrap();
    assert!(saved.is_none());
}

#[tokio::test]
async fn test_resume_loaded_failure_restores_previous_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx.clone(), rx);
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "old".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let preview = SessionPreview {
        key_id: app.key.id.clone(),
        key_name: app.key.display_name().to_string(),
        base_url: app.key.base_url.clone(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
    };

    app.begin_resume_load(preview);
    let request_id = app.loading_resume.as_ref().unwrap().request_id;
    tx.send(RuntimeEvent::ResumeLoaded {
        request_id,
        result: Err("boom".to_string()),
    })
    .unwrap();

    app.handle_runtime_events().await.unwrap();

    assert_eq!(app.history.len(), 1);
    assert_eq!(app.history[0].content, "old");
    assert!(app.loading_resume.is_none());
    assert_eq!(
        app.notice.as_ref().map(|(_, text)| text.as_str()),
        Some("boom")
    );
}

#[tokio::test]
async fn test_resume_resets_agent_engine() {
    // A resumed conversation must drop any live engine so the next turn re-seeds
    // from the loaded history; reusing it would continue the prior thread.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store;
    app.key = key.clone();
    app.model = "claude".to_string();

    let engine =
        crate::agent::engine::AgentEngine::new("/tmp", "claude", "2026-06-14", &[], &[], 0, 0);
    app.agent_engine = Some(AgentSession {
        key_id: key.id.clone(),
        model: "claude".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });

    let session = LoadedSession {
        key_id: key.id.clone(),
        session_id: "resumed".to_string(),
        raw_model: "claude".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "earlier turn".to_string(),
            reasoning_content: None,
            attachments: vec![],
        }],
        // A durable transcript on the resumed session is stashed for the next
        // engine build to restore verbatim (exact tool history).
        engine_messages: Some(vec![
            serde_json::json!({"role": "user", "content": "earlier turn"}),
            serde_json::json!({"role": "assistant", "content": "earlier reply"}),
        ]),
    };
    app.apply_loaded_session(session).await.unwrap();

    assert!(
        app.agent_engine.is_none(),
        "resume must drop the prior engine so the next turn re-seeds"
    );
    assert_eq!(app.session_id, "resumed");
    assert_eq!(app.history.len(), 1);
    assert_eq!(
        app.pending_agent_messages.as_ref().map(|m| m.len()),
        Some(2),
        "the durable transcript is stashed for exact restore on the next build"
    );
}

#[tokio::test]
async fn test_resume_lists_sessions_regardless_of_cwd() {
    // Sessions persist under the stable launch dir (for logs), but /resume is
    // global — it must surface sessions saved under ANY cwd, including the dead
    // per-pid sandbox dirs older sessions live in.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    // One session under an old ephemeral sandbox path (saved directly).
    store
        .save_chat_session_with_id(
            &key_id,
            &key.base_url,
            "/tmp/aivo-chat-old",
            "sandbox-sess",
            "claude",
            None,
            &[crate::services::session_store::StoredChatMessage {
                role: "user".into(),
                content: "older".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            }],
            "older",
            "older",
            crate::services::session_store::SessionTokens::default(),
        )
        .await
        .unwrap();

    // One persisted through the app under the real launch dir.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/aivo-chat-99999".to_string();
    app.real_cwd = "/home/me/project".to_string();
    app.session_id = "real-cwd-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "remember me".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    assert_eq!(app.persist_cwd(), "/home/me/project"); // logs key on real dir
    app.persist_history().await.unwrap();

    // /resume shows BOTH, newest first — no cwd filter.
    let sessions = load_resume_snapshots(&store).await.unwrap();
    let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    assert!(ids.contains(&"real-cwd-sess"), "got {ids:?}");
    assert!(ids.contains(&"sandbox-sess"), "got {ids:?}");
}

/// `session_tokens` (the running per-session total folded from each turn) is
/// written into the chat index entry, so `aivo stats --since` can attribute
/// windowed chat usage; and resuming re-seeds the running total from it.
#[tokio::test]
async fn test_persist_history_writes_session_tokens_to_index() {
    use crate::services::session_store::SessionTokens;

    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.session_id = "tok-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // A turn folded this much real usage into the session total.
    app.session_tokens = SessionTokens {
        prompt_tokens: 100,
        completion_tokens: 20,
        cache_read_tokens: 40,
        cache_write_tokens: 0,
    };
    app.persist_history().await.unwrap();

    // The windowed aggregation (what `aivo stats --since` reads) now sees them.
    let far_past = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let window = store.aggregate_chat_window_since(far_past).await;
    let total = window.total();
    assert_eq!(total.prompt_tokens, 100);
    assert_eq!(total.completion_tokens, 20);
    assert_eq!(total.cache_read_tokens, 40);

    // The getter that re-seeds the running total on resume returns the same.
    let seeded = store.chat_session_tokens("tok-sess").await;
    assert_eq!(seeded, app.session_tokens);
    assert_eq!(
        store.chat_session_tokens("nope").await,
        SessionTokens::default()
    );
}

#[tokio::test]
async fn test_log_agent_turn_records_under_real_cwd() {
    use crate::services::log_store::LogQuery;

    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/aivo-chat-1".to_string();
    app.real_cwd = "/home/me/proj".to_string();
    app.session_id = "agent-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "do the thing".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.log_agent_turn(1234).await;

    // The turn shows in `aivo logs` filtered to the real project dir.
    let rows = store
        .logs()
        .list(LogQuery {
            limit: 100,
            cwd: Some("/home/me/proj".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, "chat_turn");
    assert_eq!(rows[0].session_id.as_deref(), Some("agent-sess"));
    assert_eq!(rows[0].output_tokens, Some(1234));
}

#[tokio::test]
async fn test_flush_for_exit_persists_partial_response_when_streaming() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "exit-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Once upon a time".to_string();

    app.flush_for_exit().await;

    let saved = store
        .get_chat_session("exit-session")
        .await
        .unwrap()
        .expect("session should be persisted on exit");
    let messages = saved.decrypt_messages().unwrap();
    assert_eq!(messages.len(), 2, "user prompt + partial reply should save");
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, "Once upon a time");
}

#[tokio::test]
async fn test_flush_for_exit_persists_user_only_history() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "user-only-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.flush_for_exit().await;

    let saved = store
        .get_chat_session("user-only-session")
        .await
        .unwrap()
        .expect("session with only a user message should still persist on exit");
    let messages = saved.decrypt_messages().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
}

#[tokio::test]
async fn test_flush_for_exit_skips_persist_for_empty_history() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "empty-session".to_string();
    app.raw_model = "claude".to_string();

    app.flush_for_exit().await;

    let saved = store.get_chat_session("empty-session").await.unwrap();
    assert!(
        saved.is_none(),
        "empty history should not produce a session"
    );
}

#[tokio::test]
async fn test_apply_model_updates_last_selection_preserving_tool() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Seed a prior launchable selection so we can assert the tool is preserved
    // (a `/model` switch must not overwrite it with "chat").
    store
        .set_last_selection(&key, "claude", Some("old-model"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("new-model".to_string()).await.unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_id);
    assert_eq!(sel.model.as_deref(), Some("new-model"));
    assert_eq!(sel.tool, "claude", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_complete_key_switch_updates_last_selection() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();
    store
        .set_last_selection(&key_a_full, "codex", Some("model-a"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_b_id, "switched-to key must be selected");
    assert_eq!(sel.model.as_deref(), Some("model-b"));
    assert_eq!(sel.tool, "codex", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_apply_model_survives_resolved_sentinel_base_url() {
    // The live key may carry a base_url resolved away from a sentinel (ollama,
    // aivo-starter). The persisted selection must use the *stored* key's
    // base_url, or `get_last_selection` prunes it as stale.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("ollama", "ollama", None, "")
        .await
        .unwrap();
    let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Simulate the launch-time sentinel resolution that mutates the live key.
    key.base_url = "http://localhost:11434/v1".to_string();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("llama3".to_string()).await.unwrap();

    let sel = store
        .get_last_selection()
        .await
        .unwrap()
        .expect("selection must survive the sentinel/resolved base_url mismatch");
    assert_eq!(sel.key_id, key_id);
    assert_eq!(
        sel.base_url, "ollama",
        "stored sentinel base_url is persisted"
    );
    assert_eq!(sel.model.as_deref(), Some("llama3"));
}

#[tokio::test]
async fn test_apply_model_skips_last_selection_for_hf_synthetic_key() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = ApiKey::new_with_protocol(
        crate::services::huggingface::HF_LOCAL_KEY_ID.to_string(),
        "hf:demo".to_string(),
        "http://localhost:8080/v1".to_string(),
        None,
        "huggingface".to_string(),
    );

    app.apply_model("hf-model".to_string()).await.unwrap();

    assert!(
        store.get_last_selection().await.unwrap().is_none(),
        "ephemeral HF synthetic key must not be remembered as the selection"
    );
}

/// A permission card must not steal in-flight composer typing. With a queued
/// draft in progress the single-letter decision keys (y/a/n) belong to that
/// message — they type into the draft and leave the card up, so a stray
/// keystroke can't approve a tool by accident. Esc still resolves the card.
#[tokio::test]
async fn test_permission_card_keys_do_not_decide_while_composing() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The user is mid-composing a queued message when a card appears.
    app.draft = "deplo".to_string();
    app.cursor = app.draft.len();
    let (reply, mut decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // 'y' is part of the word "deploy", not an approval.
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "deploy", "the keystroke typed into the draft");
    assert!(
        app.agent_permission.is_some(),
        "the card stays up — the key did not decide"
    );
    assert!(
        matches!(
            decision_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ),
        "no decision was sent to the waiting engine"
    );

    // Esc always denies — it can't be message content.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "Esc resolves the card");
    assert_eq!(decision_rx.await.unwrap(), Decision::Deny);
}

/// With an empty composer (the idle case) the quick single-key decisions still
/// work — 'y' approves immediately.
#[tokio::test]
async fn test_permission_card_y_approves_with_empty_composer() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.draft.is_empty());

    let (reply, decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "the card is resolved");
    assert_eq!(decision_rx.await.unwrap(), Decision::Allow);
}

/// The card swaps its keycap row for a hint while a draft is in progress, so the
/// user knows y/a/n won't decide and how to respond without losing the message.
#[test]
fn test_permission_card_shows_composing_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "next message".to_string();
    app.cursor = app.draft.len();
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("type into your message"),
        "composing hint missing:\n{screen}"
    );
}

/// A Cursor card's "always" is session-wide auto-approve, unlike the native
/// engine's per-action "always"; the keycap label spells that out so the broader
/// scope isn't a surprise.
#[test]
fn test_permission_card_cursor_always_label_says_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("edit main.rs".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always (this session)"),
        "cursor 'always' must disclose its session-wide scope:\n{screen}"
    );
}

/// The native engine's "always" stays scoped to this action, so its label is the
/// plain "always" with no session-wide qualifier.
#[test]
fn test_permission_card_native_always_label_is_plain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always"),
        "native card still offers 'always':\n{screen}"
    );
    assert!(
        !screen.contains("this session"),
        "native 'always' is scoped, not session-wide:\n{screen}"
    );
}

/// Messages submitted while a turn is in flight queue in order — a second one
/// must not silently clobber the first (the old single-slot behavior).
#[tokio::test]
async fn test_queued_messages_fifo_no_clobber() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    app.draft = "first".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    app.draft = "second".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    assert_eq!(
        app.queued_messages,
        vec!["first".to_string(), "second".to_string()],
        "both messages are queued in submit order"
    );
    let (_lvl, notice) = app.notice.clone().expect("a queued notice");
    assert!(
        notice.contains("2 waiting"),
        "the notice reflects the queue count: {notice}"
    );
}

/// The queued indicator shows the real count, not a hardcoded "1".
#[test]
fn test_queued_indicator_shows_count() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let (screen, _rows) = render_full_screen(&mut app, 80, 16);
    assert!(
        screen.contains("3 queued"),
        "the indicator must show the queue length:\n{screen}"
    );
}

/// `/mcp` Ctrl+D arms a two-press delete (removal edits the user mcp.json), the
/// same confirm as /skills and the resume picker — the first press only arms and
/// surfaces a confirm prompt; Esc disarms without closing the overlay.
#[tokio::test]
async fn test_mcp_delete_arms_then_esc_disarms() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, Some(0), "first Ctrl+D arms"),
        _ => panic!("the overlay must stay open after the first Ctrl+D"),
    }
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("confirm"),
        "an armed delete shows a confirm prompt:\n{screen}"
    );

    // Esc cancels the arm but must NOT close the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, None, "Esc disarms"),
        _ => panic!("Esc on an armed delete must not close the overlay"),
    }
}

/// A repo whose .mcp.json defines stdio servers must NOT spawn them silently:
/// `connect_mcp_with_consent` raises a consent card listing the exact commands
/// and leaves the decision Unknown until the user answers.
#[tokio::test]
async fn project_mcp_stdio_raises_consent_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"sh","args":["-c","echo hi"]}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;

    let prompt = app.pending_mcp_consent.as_ref().expect("a consent card");
    assert_eq!(
        prompt.servers,
        vec![("x".to_string(), "sh -c echo hi".to_string())],
        "the card lists the exact command to be run"
    );
    assert_eq!(
        app.project_mcp_consent,
        ProjectMcpConsent::Unknown,
        "no decision until the user answers"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// Denying the consent card holds the servers back for the session and clears
/// the card; nothing is persisted.
#[tokio::test]
async fn project_mcp_consent_deny_holds_back() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "sh -c echo hi".to_string())],
        cwd: ".".to_string(),
        base_disabled: Default::default(),
    });
    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await;
    assert!(app.pending_mcp_consent.is_none(), "the card is cleared");
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Denied);
}

/// "always" approves for this repo: it persists to the per-repo allow-list (so a
/// future session in the same dir skips the card) and clears the card.
#[tokio::test]
async fn project_mcp_consent_always_persists() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-always-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    app.pending_mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "echo".to_string())],
        cwd: cwd.clone(),
        base_disabled: Default::default(),
    });

    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await;
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    assert!(app.pending_mcp_consent.is_none());

    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let digest = project_mcp_digest(&[("x".to_string(), "echo".to_string())]);
    assert!(
        app.session_store
            .get_project_mcp_approved(&dir_key, &digest)
            .await,
        "'always' is persisted to the per-repo allow-list, bound to the server digest"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo already on the per-repo allow-list connects its project stdio servers
/// without re-prompting — the consent is seeded from the persistent store.
#[tokio::test]
async fn project_mcp_preapproved_repo_skips_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-pre-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        // Bogus command: the background connect's spawn fails fast, no real process.
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let servers = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&servers))
        .await
        .unwrap();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_none(),
        "a pre-approved repo doesn't prompt"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo previously approved "always" but whose `.mcp.json` then CHANGES (a
/// different command) re-prompts: the stored approval is bound to the server
/// content digest, so a swapped-in command can't ride the old consent.
#[tokio::test]
async fn project_mcp_changed_config_reprompts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-changed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();

    // Approve the ORIGINAL server set.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let orig = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&orig))
        .await
        .unwrap();

    // The author swaps in a DIFFERENT command — the prior approval must not apply.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_evil_binary_zzz"}}}"#,
    )
    .unwrap();
    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_some(),
        "a changed .mcp.json re-prompts instead of reusing the old approval"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Unknown);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo with no project `.mcp.json` (or only HTTP servers) never raises the
/// consent card — there's no local command to gate.
#[tokio::test]
async fn project_mcp_no_stdio_no_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-http-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"remote":{"url":"https://h/mcp"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.pending_mcp_consent.is_none(),
        "HTTP-only project servers aren't gated (no local exec)"
    );
    let _ = std::fs::remove_dir_all(&repo);
}
