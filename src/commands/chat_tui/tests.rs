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
    // cursor at end of text
    assert_eq!(cursor_position("hello", 5, 10, 2), (7, 0));
    assert_eq!(cursor_position("hello\nworld", 11, 10, 2), (7, 1));
    // cursor in middle
    assert_eq!(cursor_position("hello\nworld", 6, 10, 2), (2, 1));
    assert_eq!(cursor_position("hello\nworld", 0, 10, 2), (2, 0));
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
    assert_eq!(cursor_position("abcdefgh", 8, 8, 2), (2, 1));
}

#[test]
fn test_composer_cursor_position_offsets_attachment_rows() {
    let (x, y) = cursor_position("hello", 5, 20, 2);
    assert_eq!((x, y.saturating_add(1)), (7, 1));
    let (x, y) = cursor_position("", 0, 20, 2);
    assert_eq!((x, y.saturating_add(2)), (2, 2));
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

#[test]
fn test_cursor_movement_basic() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();

    // cursor_left moves back one char
    app.cursor_left();
    assert_eq!(app.cursor, 10); // before 'd'

    // cursor_right moves forward
    app.cursor_right();
    assert_eq!(app.cursor, 11); // end

    // cursor_home goes to start
    app.cursor_home();
    assert_eq!(app.cursor, 0);

    // cursor_end goes to end
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
async fn test_ctrl_c_clears_prompt_without_exiting() {
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
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft.is_empty());
    assert!(app.draft_attachments.is_empty());
    assert_eq!(app.cursor, 0);
    assert!(!app.pending_clear_screen);
}

#[tokio::test]
async fn test_ctrl_c_exits_when_prompt_empty() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(should_exit);
}

#[tokio::test]
async fn test_ctrl_c_clears_attachment_only_prompt() {
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
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.draft_attachments.is_empty());
}

#[tokio::test]
async fn test_ctrl_c_clears_history_navigation_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["older".to_string(), "newer".to_string()];
    app.history_prev();
    assert!(app.draft_history_index.is_some());
    assert!(!app.draft.is_empty());

    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert!(app.draft_history_index.is_none());
    assert!(app.draft_history_stash.is_none());
}

#[tokio::test]
async fn test_ctrl_c_exits_when_overlay_open() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hidden behind overlay".to_string();
    app.cursor = app.draft.len();
    app.overlay = Overlay::Help;

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(should_exit);
    assert_eq!(app.draft, "hidden behind overlay");
}

#[tokio::test]
async fn test_ctrl_l_requests_clear_screen_without_touching_prompt() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "hello world".to_string();
    app.cursor = app.draft.len();

    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    assert!(!should_exit);
    assert!(app.pending_clear_screen);
    assert_eq!(app.draft, "hello world");
    assert_eq!(app.cursor, "hello world".len());
}

fn make_test_app(
    tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) -> ChatTuiApp {
    ChatTuiApp {
        session_store: SessionStore::new(),
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
        raw_model: String::new(),
        model: String::new(),
        format: ChatFormat::OpenAI,
        history: Vec::new(),
        draft: String::new(),
        draft_attachments: Vec::new(),
        cursor: 0,
        command_menu: CommandMenuState::default(),
        draft_history: Vec::new(),
        draft_history_index: None,
        draft_history_stash: None,
        session_id: String::new(),
        overlay: Overlay::None,
        notice: None,
        show_reasoning: true,
        pending_response: String::new(),
        pending_reasoning: String::new(),
        pending_submit: None,
        sending: false,
        request_started_at: None,
        last_usage: None,
        context_tokens: 0,
        follow_output: true,
        transcript_scroll: 0,
        transcript_width: 0,
        transcript_view_height: 0,
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
        pending_clear_screen: false,
    }
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
    let lines = render_markdown_lines("## Title\n\n- one\n- two\n\n```rust\nlet x = 1;\n```");
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
fn test_render_assistant_streaming_does_not_append_cursor_glyph() {
    let mut lines = Vec::new();
    render_assistant_message(&mut lines, true, None, "- item");

    let plain = lines
        .into_iter()
        .map(|line| line.plain)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!plain.contains('▋'));
    assert!(plain.contains("• item"));
}

#[test]
fn test_build_transcript_shows_streaming_reasoning_before_content() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.pending_reasoning = "Inspecting the request".to_string();

    let transcript = app.build_transcript();
    let plain = transcript
        .text
        .lines
        .iter()
        .map(|line| plain_text_from_spans(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(plain.contains("Thinking"));
    assert!(plain.contains("Inspecting the request"));
    assert!(!plain.contains("esc to interrupt"));
}

#[test]
fn test_build_transcript_hides_reasoning_when_hidden() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.show_reasoning = false;
    app.pending_reasoning = "Inspecting the request".to_string();

    let transcript = app.build_transcript();
    let plain = transcript
        .text
        .lines
        .iter()
        .map(|line| plain_text_from_spans(&line.spans))
        .collect::<Vec<_>>()
        .join("\n");

    assert!(!plain.contains("Inspecting the request"));
    assert!(!plain.contains("Thinking hidden"));
}

#[test]
fn test_hidden_reasoning_hint_moves_to_composer_placeholder() {
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

    assert_eq!(
        plain,
        ">  Ask anything · / for commands · Ctrl+T toggle think"
    );
}

#[test]
fn test_normalized_reasoning_lines_trims_and_removes_blank_runs() {
    assert_eq!(
        normalized_reasoning_lines("\nalpha\n\n\nbeta\n\n"),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn test_footer_status_label_stays_token_count_while_streaming() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.request_started_at = Some(Instant::now() - Duration::from_secs(12));
    app.context_tokens = 5_120;

    let (label, color) = app.footer_status_label();
    assert_eq!(label, "~5.1k tokens");
    assert_eq!(color, MUTED);
}

#[test]
fn test_transcript_intro_lines_use_model_and_base_url() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "claude-sonnet-4".to_string();
    app.key = ApiKey::new_with_protocol(
        "prod".to_string(),
        "test".to_string(),
        "https://openrouter.ai/api/v1".to_string(),
        None,
        String::new(),
    );
    app.cwd = "/tmp/project".to_string();

    assert_eq!(
        app.transcript_intro_lines(),
        vec![
            "AIVO Chat".to_string(),
            "claude-sonnet-4 · https://openrouter.ai/api/v1".to_string(),
            "/tmp/project".to_string(),
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
fn test_error_notice_only_returns_errors() {
    let error = (ERROR, "boom".to_string());
    let info = (MUTED, "ok".to_string());

    assert_eq!(error_notice(Some(&error)), Some("boom"));
    assert_eq!(error_notice(Some(&info)), None);
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
fn test_rect_contains() {
    let area = Rect::new(10, 4, 8, 3);
    assert!(rect_contains(area, (10, 4)));
    assert!(rect_contains(area, (17, 6)));
    assert!(!rect_contains(area, (18, 6)));
    assert!(!rect_contains(area, (17, 7)));
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
    assert_eq!(parse_slash_command("clear").unwrap(), SlashCommand::Clear);
}

#[test]
fn test_parse_slash_command_unknown() {
    let err = parse_slash_command("wat").unwrap_err().to_string();
    assert!(err.contains("Unknown command"));
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
