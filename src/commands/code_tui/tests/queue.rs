use super::super::*;
use super::helpers::*;

#[test]
fn test_discard_queued_input_counts_and_clears() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages.push(queued("a"));
    app.queued_messages.push(queued("b"));
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

    app.queued_messages.push(queued(
        "Use the \"x\" skill. Follow these instructions:\n\n…pages…",
    ));
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
    assert_eq!(queued_texts(&app), vec!["actually use tabs"]);
    assert!(app.steering_queue.lock().unwrap().is_empty());

    app.apply_agent_steered("also add a test".to_string());
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "also add a test")
    );
    assert!(app.notice.is_none(), "notice: {:?}", app.notice);
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
        queued_texts(&app),
        vec!["first", "second"],
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

#[tokio::test]
async fn test_new_chat_clears_queued_commands() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.run_compact_command(true).await;
    assert_eq!(
        app.queued_commands,
        vec![SlashCommand::Compact { fast: true }]
    );

    app.start_new_chat().await;
    assert!(app.queued_commands.is_empty());
}

#[tokio::test]
async fn test_interrupt_drains_queued_messages() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first".to_string(),
        reasoning_content: None,
        attachments: vec![],
        id: None,
        timestamp: None,
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.queued_messages = vec![queued("fix login"), queued("run tests")];

    app.interrupt_inflight_request().await.unwrap();

    assert!(
        app.history
            .iter()
            .any(|m| m.role == "assistant" && m.content == "partial")
    );
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "fix login")
    );
    assert_eq!(queued_texts(&app), vec!["run tests"]);
    assert!(app.sending);
}

#[tokio::test]
async fn test_interrupt_reclaims_steering_then_drains() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first".to_string(),
        reasoning_content: None,
        attachments: vec![],
        id: None,
        timestamp: None,
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .push("actually use tabs".to_string());

    app.interrupt_inflight_request().await.unwrap();

    assert!(app.steering_queue.lock().unwrap().is_empty());
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "actually use tabs")
    );
    assert!(app.sending);
}

#[tokio::test]
async fn test_interrupt_empty_unsends_then_drains_queue() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first message".to_string(),
        reasoning_content: None,
        attachments: vec![],
        id: None,
        timestamp: None,
    });
    app.pending_submit = Some(PendingSubmission {
        content: "first message".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.queued_messages = vec![queued("do this instead")];

    app.interrupt_inflight_request().await.unwrap();

    assert_eq!(app.draft, "first message");
    assert!(
        app.history
            .last()
            .is_some_and(|m| m.role == "user" && m.content == "do this instead")
    );
    assert!(app.queued_messages.is_empty());
    assert!(app.sending);
}

#[tokio::test]
async fn test_up_recalls_whole_queue_into_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![queued("fix login"), queued("run tests")];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "fix login\nrun tests");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.queued_messages.is_empty(), "the queue is drained");
}

#[tokio::test]
async fn test_ctrl_p_recalls_the_queue_like_up() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![queued("fix login"), queued("run tests")];

    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.draft, "fix login\nrun tests");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.queued_messages.is_empty());

    app.draft.clear();
    app.cursor = 0;
    app.draft_history = vec!["an older draft".to_string()];
    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert_eq!(app.draft, "an older draft");
}

#[tokio::test]
async fn test_up_recalls_all_three_queues_in_delivery_order() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.steering_queue
        .lock()
        .unwrap()
        .extend(["steer it".to_string()]);
    app.queued_commands.push(SlashCommand::Rewind);
    app.queued_messages = vec![queued("m1"), queued("m2")];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "steer it\n/rewind\nm1\nm2");
    assert!(app.steering_queue.lock().unwrap().is_empty());
    assert!(app.queued_commands.is_empty());
    assert!(app.queued_messages.is_empty());
}

#[tokio::test]
async fn test_up_keeps_the_typed_draft_below_the_recall() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![queued("fix login")];
    app.draft = "typing".to_string();
    app.cursor = 3;

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "fix login\ntyping");
    assert_eq!(app.cursor, "fix login\ntyp".len());
}

#[tokio::test]
async fn test_up_below_the_top_row_leaves_the_queue_alone() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.queued_messages = vec![queued("queued")];
    app.draft = "one\ntwo".to_string();
    app.cursor = app.draft.len();

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "one\ntwo", "the draft is untouched");
    assert_eq!(queued_texts(&app), vec!["queued"]);
    assert_eq!(app.cursor, 3, "the cursor moved to the first line");

    // Same guard mid-turn, where bare ↑ belongs to the transcript scroller.
    app.sending = true;
    app.cursor = app.draft.len();
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "one\ntwo");
    assert_eq!(queued_texts(&app), vec!["queued"]);
}

#[tokio::test]
async fn test_up_falls_through_to_history_once_the_queue_is_empty() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["an older draft".to_string()];
    app.queued_messages = vec![queued("queued")];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "queued");
    app.draft.clear();
    app.cursor = 0;
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "an older draft");
}

#[tokio::test]
async fn test_recall_clears_the_queue_tip_but_not_other_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![queued("fix login")];
    app.notice = Some((
        MUTED(),
        "Queued — sends when the current turn finishes".to_string(),
    ));

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.notice, None, "queue tip clears with the recall");

    app.queued_messages = vec![queued("fix login")];
    app.draft.clear();
    app.cursor = 0;
    app.notice = Some((ERROR(), "Copy failed: nope".to_string()));
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg == "Copy failed: nope"),
        "an unrelated notice survives"
    );
}

#[tokio::test]
async fn test_up_navigates_the_command_menu_over_the_queue() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![queued("queued")];
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    for prev in [
        KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
    ] {
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .await
            .unwrap();
        app.handle_key(prev).await.unwrap();
        assert_eq!(app.command_menu.selected, 0);
        assert_eq!(app.draft, "/", "the queue stayed out of it");
        assert_eq!(queued_texts(&app), vec!["queued"]);
    }
}

#[test]
fn test_command_recall_text_round_trips() {
    for cmd in [
        SlashCommand::Compact { fast: false },
        SlashCommand::Compact { fast: true },
        SlashCommand::Rewind,
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
    app.queued_messages.push(queued("plain msg"));
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("» steer msg"), "steering row:\n{screen}");
    assert!(screen.contains("/rewind"), "command row:\n{screen}");
    assert!(screen.contains("· plain msg"), "message row:\n{screen}");

    // An expanded skill body renders as its compact /name form.
    app.queued_messages.push(queued(
        "Use the \"my-skill\" skill. Follow these instructions:\n\nLong body.\n\nInput: hello",
    ));
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("/my-skill hello"), "{screen}");
    assert!(!screen.contains("Follow these instructions"), "{screen}");

    // Overflow: 7 messages cap at QUEUE_PANEL_MAX_ROWS + an indicator.
    app.steering_queue.lock().unwrap().clear();
    app.queued_commands.clear();
    app.queued_messages = (1..=7).map(|i| queued(&format!("q{i}"))).collect();
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("· q1") && screen.contains("· q5"),
        "{screen}"
    );
    assert!(!screen.contains("· q6"), "{screen}");
    assert!(screen.contains("… +2 more"), "{screen}");
    assert!(screen.contains(QUEUE_RECALL_HINT.trim()), "{screen}");
    assert!(!screen.contains("▸ ·"), "no selection marker:\n{screen}");
}

fn staged_image() -> MessageAttachment {
    named_image("shot.png")
}

fn named_image(name: &str) -> MessageAttachment {
    MessageAttachment {
        name: name.to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    }
}

#[tokio::test]
async fn test_queued_message_carries_its_staged_image() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.model_image_input = Some(true);
    app.sending = true;

    app.draft_attachments.push(staged_image());
    app.draft = "[image #1] what is this".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    assert_eq!(queued_texts(&app), vec!["[image #1] what is this"]);
    assert_eq!(
        app.queued_messages[0].attachments.len(),
        1,
        "the image is queued with its own message"
    );
    assert!(
        app.draft_attachments.is_empty(),
        "the image moved into the queue"
    );

    app.sending = false;
    app.drain_queued_message().await.unwrap();

    let last = app.history.last().expect("a user turn");
    assert_eq!(last.content, "[image #1] what is this");
    assert_eq!(
        last.attachments
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        ["shot.png"],
        "the queued turn carries the screenshot"
    );
}

#[tokio::test]
async fn test_drain_leaves_the_composing_draft_attachments_alone() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.model_image_input = Some(true);
    app.sending = true;

    app.draft = "first".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();
    assert_eq!(queued_texts(&app), vec!["first"]);

    app.draft_attachments.push(staged_image());
    app.sending = false;
    app.drain_queued_message().await.unwrap();

    let last = app.history.last().expect("a user turn");
    assert_eq!(last.content, "first");
    assert!(
        last.attachments.is_empty(),
        "the queued text-only message must not pick up a later staged image"
    );
    assert_eq!(
        app.draft_attachments.len(),
        1,
        "the composer's own staged image is untouched"
    );
}

#[tokio::test]
async fn test_steered_message_keeps_its_image_and_says_so() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.agent_serve = Some((
        tokio::spawn(async { Ok(()) }),
        std::sync::Arc::new(tokio::sync::Notify::new()),
    ));

    app.draft_attachments.push(staged_image());
    app.draft = "[image #1] use tabs".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();

    {
        let steering = app.steering_queue.lock().unwrap();
        assert_eq!(steering.as_slice(), ["[image #1] use tabs".to_string()]);
    }
    assert!(
        app.queued_messages.is_empty(),
        "an engine turn steers, it doesn't queue"
    );
    assert_eq!(app.draft_attachments.len(), 1, "the image stays staged");
    assert!(
        notice_text(&app).contains("1 staged attachment not sent"),
        "notice: {}",
        notice_text(&app)
    );
}

#[tokio::test]
async fn test_attachment_only_follow_up_queues_mid_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft_attachments.push(staged_image());

    app.submit_draft().await.unwrap();

    assert_eq!(queued_texts(&app), vec![""]);
    assert_eq!(app.queued_messages[0].attachments.len(), 1);
    assert!(app.draft_attachments.is_empty(), "the image is queued");

    let (tx2, rx2) = tokio::sync::mpsc::unbounded_channel();
    let mut steered = make_test_app(tx2, rx2);
    steered.sending = true;
    steered.agent_serve = Some((
        tokio::spawn(async { Ok(()) }),
        std::sync::Arc::new(tokio::sync::Notify::new()),
    ));
    steered.draft_attachments.push(staged_image());
    steered.submit_draft().await.unwrap();
    assert!(steered.steering_queue.lock().unwrap().is_empty());
    assert_eq!(steered.queued_messages[0].attachments.len(), 1);

    let (screen, _rows) = render_full_screen(&mut steered, 70, 20);
    assert!(screen.contains("[image] shot.png"), "{screen}");

    steered
        .handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(steered.draft.is_empty());
    assert_eq!(steered.draft_attachments.len(), 1, "the image is recalled");
}

#[tokio::test]
async fn test_recall_restores_queued_images() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.draft_attachments.push(staged_image());
    app.draft = "[image #1] what is this".to_string();
    app.cursor = app.draft.len();
    app.submit_draft().await.unwrap();
    assert_eq!(app.draft_attachments.len(), 0);

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();

    assert_eq!(app.draft, "[image #1] what is this");
    assert_eq!(
        app.draft_attachments
            .iter()
            .map(|a| a.name.as_str())
            .collect::<Vec<_>>(),
        ["shot.png"],
        "the recalled image is staged again"
    );
    assert_eq!(
        app.attachment_tag_spans(),
        vec![Some((0, "[image #1]".len()))],
        "the tag binds to the restored image"
    );
}

fn staged_names(app: &CodeTuiApp) -> Vec<&str> {
    app.draft_attachments
        .iter()
        .map(|a| a.name.as_str())
        .collect()
}

#[tokio::test]
async fn test_recall_renumbers_tags_across_queued_messages() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![
        QueuedMessage {
            text: "[image #1] a".to_string(),
            attachments: vec![named_image("a.png")],
        },
        QueuedMessage {
            text: "[image #1] b [image #2]".to_string(),
            attachments: vec![named_image("b.png"), named_image("c.png")],
        },
    ];

    assert!(app.recall_queued_into_draft());

    assert_eq!(app.draft, "[image #1] a\n[image #2] b [image #3]");
    assert_eq!(staged_names(&app), ["a.png", "b.png", "c.png"]);
    assert!(app.attachment_tag_spans().iter().all(Option::is_some));
}

#[tokio::test]
async fn test_recall_shifts_the_drafts_own_tags() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages = vec![QueuedMessage {
        text: "[image #1] queued".to_string(),
        attachments: vec![named_image("queued.png")],
    }];
    app.draft_attachments.push(named_image("mine.png"));
    app.draft = "[image #1] mine".to_string();
    app.cursor = app.draft.len();

    assert!(app.recall_queued_into_draft());

    assert_eq!(app.draft, "[image #1] queued\n[image #2] mine");
    assert_eq!(
        app.cursor,
        app.draft.len(),
        "the cursor stays at the draft's end"
    );
    assert_eq!(staged_names(&app), ["queued.png", "mine.png"]);
    assert_eq!(
        app.attachment_tag_spans(),
        vec![Some((0, 10)), Some((18, 28))]
    );
}

#[tokio::test]
async fn test_refused_queued_message_returns_to_the_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_dispatch_refusal(&mut app);
    let image = app.draft_attachments.remove(0);
    app.queued_messages = vec![
        QueuedMessage {
            text: "[image #1] what is this".to_string(),
            attachments: vec![image],
        },
        queued("then this"),
    ];
    app.draft = "typing".to_string();
    app.cursor = app.draft.len();

    app.drain_queued_message().await.unwrap();

    assert!(!app.sending, "nothing went out");
    assert!(app.queued_messages.is_empty());
    assert_eq!(app.draft, "[image #1] what is this\nthen this\ntyping");
    assert_eq!(staged_names(&app), ["shot.png"]);
    assert!(
        notice_text(&app).contains("can't read images"),
        "{}",
        notice_text(&app)
    );
}
