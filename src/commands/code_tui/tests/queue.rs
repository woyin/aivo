use super::super::*;
use super::helpers::*;

#[test]
fn test_discard_queued_input_counts_and_clears() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages.push("a".to_string());
    app.queued_messages.push("b".to_string());
    app.steering_queue.lock().unwrap().push("steer".to_string());
    assert_eq!(app.discard_queued_input(), 3);
    assert!(app.queued_messages.is_empty());
    assert!(app.steering_queue.lock().unwrap().is_empty());
    assert_eq!(app.discard_queued_input(), 0);
}

/// Draining a queued message records nothing — every queue site already recorded
/// the recallable form, and a queued skill's expanded body must not enter ↑/↓.
#[tokio::test]
async fn test_drained_queued_message_not_recorded_in_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.queued_messages
        .push("Use the \"x\" skill. Follow these instructions:\n\n…pages…".to_string());
    app.drain_queued_message().await.unwrap();

    assert!(
        app.draft_history.is_empty(),
        "expanded machine text must not enter recall"
    );
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content.starts_with("Use the")),
        "the queued message still went out"
    );
}

#[tokio::test]
async fn test_mid_turn_message_steers_then_reclaims_or_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.agent_serve = Some((
        tokio::spawn(async { Ok(()) }),
        std::sync::Arc::new(tokio::sync::Notify::new()),
    ));

    app.draft = "actually use tabs".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();
    assert!(
        app.queued_messages.is_empty(),
        "engine turns steer, not queue"
    );
    {
        let steering = app.steering_queue.lock().unwrap();
        assert_eq!(steering.as_slice(), ["actually use tabs".to_string()]);
    }

    app.reclaim_unsent_steering();
    assert_eq!(app.queued_messages, vec!["actually use tabs".to_string()]);
    assert!(app.steering_queue.lock().unwrap().is_empty());

    app.apply_agent_steered("also add a test".to_string());
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "also add a test")
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

/// The `/` menu (dropdown + inline hint) stays available while a turn is in
/// flight — it used to be blanket-hidden by `is_busy()`, which made slash
/// commands look dead mid-turn.
#[test]
fn test_command_menu_visible_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    let menu = app.visible_command_menu().expect("menu shows mid-turn");
    assert!(!menu.entries.is_empty(), "matching commands are listed");
}

/// While the `/` menu is open mid-turn, ↑/↓ navigate the menu instead of
/// scrolling the transcript (which owns bare arrows during a turn otherwise).
#[tokio::test]
async fn test_arrows_navigate_menu_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_some());
    assert_eq!(app.command_menu.selected, 0);

    app.handle_key(KeyEvent::from(KeyCode::Down)).await.unwrap();
    assert_eq!(
        app.command_menu.selected, 1,
        "Down moves the menu selection, not the transcript"
    );
    app.handle_key(KeyEvent::from(KeyCode::Up)).await.unwrap();
    assert_eq!(app.command_menu.selected, 0);
}

/// Commands that need the engine idle queue mid-turn (instead of refusing) and
/// run when the turn finishes.
#[tokio::test]
async fn test_engine_idle_commands_queue_and_drain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    app.open_rewind_picker().await.unwrap();
    assert_eq!(app.queued_commands, vec![SlashCommand::Rewind]);
    let (_lvl, notice) = app.notice.clone().expect("a queued notice");
    assert!(
        notice.contains("/rewind queued"),
        "the notice names the queued command: {notice}"
    );

    app.sending = false;
    app.drain_queued_commands().await;
    assert!(app.queued_commands.is_empty(), "the queue drained");
    // With no history there is nothing to rewind to — the command still ran.
    let (_lvl, notice) = app.notice.clone().expect("the drained command's notice");
    assert!(notice.contains("Nothing to rewind"), "{notice}");
}

/// A queued command is dropped by an interrupt/cancel, like queued messages.
#[tokio::test]
async fn test_queued_commands_cleared_on_cancel() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.run_compact_command(true).await;
    assert_eq!(
        app.queued_commands,
        vec![SlashCommand::Compact { fast: true }]
    );

    app.cancel_inflight_request(CancelKind::Discard);
    assert!(app.queued_commands.is_empty());
}

#[tokio::test]
async fn test_queue_focus_entered_by_up_on_empty_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["first".to_string(), "second".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1), "newest row selected");

    // A non-empty draft blocks entry.
    app.queue_focus = None;
    app.draft = "typing".to_string();
    app.cursor = app.draft.len();
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert_eq!(app.draft, "typing");
}

#[tokio::test]
async fn test_queue_focus_selection_and_down_past_end_exits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(2));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(2));
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None, "↓ past the last row exits");
}

#[tokio::test]
async fn test_queue_focus_enter_recalls_message_into_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["fix login".to_string(), "run tests".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "run tests");
    assert_eq!(app.cursor, app.draft.len());
    assert_eq!(app.queued_messages, vec!["fix login".to_string()]);
    assert_eq!(app.queue_focus, None);
}

#[tokio::test]
async fn test_queue_focus_recalls_steering_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .push("steer it".to_string());

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "steer it");
    assert!(app.steering_queue.lock().unwrap().is_empty());
}

/// Ops on a steering row the engine drained mid-event fail gracefully.
#[test]
fn test_queue_row_ops_validate_after_steering_drain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.steering_queue.lock().unwrap().push("steer".to_string());
    let rows = app.queued_rows();
    assert_eq!(rows.len(), 1);

    // Simulate the engine draining the batch between snapshot and op.
    app.steering_queue.lock().unwrap().clear();
    assert!(!app.queue_row_remove(&rows[0]));
    assert!(app.queue_row_recall(&rows[0]).is_none());
    assert!(!app.queue_row_move(&rows[0], 1));
}

#[tokio::test]
async fn test_queue_focus_delete_clamps_and_exits_when_empty() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string(), "b".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(1));
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
    assert_eq!(app.queue_focus, Some(0), "selection clamps to the survivor");
    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.queued_messages.is_empty());
    assert_eq!(app.queue_focus, None, "focus exits with the last row");
}

/// Reorder stays within a segment — delivery semantics differ across them.
#[tokio::test]
async fn test_queue_focus_reorder_within_segment_not_across() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .extend(["s1".to_string(), "s2".to_string()]);
    app.queued_commands.push(SlashCommand::Rewind);
    app.queued_messages = vec!["m1".to_string(), "m2".to_string()];
    // Unified rows: [s1, s2, /rewind, m1, m2].

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(4)); // m2
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m2".to_string(), "m1".to_string()]
    );
    assert_eq!(app.queue_focus, Some(3), "selection follows the moved row");

    // No crossing into the command segment.
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m2".to_string(), "m1".to_string()]
    );
    assert_eq!(app.queued_commands, vec![SlashCommand::Rewind]);
    assert_eq!(app.queue_focus, Some(3));

    // Shift is the alias for terminals that don't deliver Alt+arrows.
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m1".to_string(), "m2".to_string()]
    );
    assert_eq!(app.queue_focus, Some(4));
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        app.queued_messages,
        vec!["m1".to_string(), "m2".to_string()]
    );
    assert_eq!(app.queue_focus, Some(4));

    app.queue_focus = Some(1); // s2
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        *app.steering_queue.lock().unwrap(),
        vec!["s2".to_string(), "s1".to_string()]
    );
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT))
        .await
        .unwrap();
    assert_eq!(
        *app.steering_queue.lock().unwrap(),
        vec!["s2".to_string(), "s1".to_string()]
    );
}

#[tokio::test]
async fn test_queue_focus_esc_exits_without_interrupting() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, Some(0));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert!(app.sending, "Esc in focus mode must not interrupt the turn");
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
}

#[tokio::test]
async fn test_queue_focus_typing_char_exits_and_inserts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec!["a".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.queue_focus, None);
    assert_eq!(app.draft, "h");
    assert_eq!(app.queued_messages, vec!["a".to_string()]);
}

#[test]
fn test_command_recall_text_round_trips() {
    for cmd in [
        SlashCommand::Compact { fast: false },
        SlashCommand::Compact { fast: true },
        SlashCommand::Rewind,
        SlashCommand::Review(None),
        SlashCommand::Review(Some("main".to_string())),
        SlashCommand::Goal(Some("ship the fix".to_string())),
        SlashCommand::Plan(Some("go".to_string())),
    ] {
        let text = queue_impl::command_recall_text(&cmd);
        assert!(text.starts_with('/'), "{text}");
        assert_eq!(parse_slash_command(&text[1..]).unwrap(), cmd, "{text}");
    }
    // Skills aren't in the static parser; they resolve by name at submit time.
    assert_eq!(
        queue_impl::command_recall_text(&SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: Some("this repo".to_string()),
        }),
        "/repo-study this repo"
    );
}

#[test]
fn test_queued_panel_rows_render_with_cap_and_more() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .push("steer msg".to_string());
    app.queued_commands.push(SlashCommand::Rewind);
    app.queued_messages.push("plain msg".to_string());
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("» steer msg"), "steering row:\n{screen}");
    assert!(screen.contains("/rewind"), "command row:\n{screen}");
    assert!(screen.contains("· plain msg"), "message row:\n{screen}");

    // An expanded skill body renders as its compact /name form.
    app.queued_messages.push(
        "Use the \"my-skill\" skill. Follow these instructions:\n\nLong body.\n\nInput: hello"
            .to_string(),
    );
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("/my-skill hello"), "{screen}");
    assert!(!screen.contains("Follow these instructions"), "{screen}");

    // Overflow: 7 messages cap at QUEUE_PANEL_MAX_ROWS + an indicator.
    app.steering_queue.lock().unwrap().clear();
    app.queued_commands.clear();
    app.queued_messages = (1..=7).map(|i| format!("q{i}")).collect();
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("· q1") && screen.contains("· q5"),
        "{screen}"
    );
    assert!(!screen.contains("· q6"), "{screen}");
    assert!(screen.contains("… +2 more"), "{screen}");
    assert!(
        !screen.contains("enter edit"),
        "no hint unfocused:\n{screen}"
    );

    // Focused on the newest row: the window follows the selection.
    app.queue_focus = Some(6);
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("▸ · q7"), "{screen}");
    assert!(screen.contains("… +2 earlier"), "{screen}");
    assert!(screen.contains("enter edit · ctrl+d remove"), "{screen}");
}

#[test]
fn test_queue_focus_cleared_by_discard_queued_input() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages = vec!["a".to_string()];
    app.queue_focus = Some(0);
    app.discard_queued_input();
    assert_eq!(app.queue_focus, None);
}
