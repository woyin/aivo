use super::super::*;
use super::helpers::*;

#[test]
fn test_btw_parse_and_recall_round_trip() {
    assert_eq!(
        parse_slash_command("btw what does PBKDF2 do").unwrap(),
        SlashCommand::Btw(Some("what does PBKDF2 do".to_string()))
    );
    assert_eq!(parse_slash_command("btw").unwrap(), SlashCommand::Btw(None));
    for cmd in [
        SlashCommand::Btw(None),
        SlashCommand::Btw(Some("why is the sky blue".to_string())),
    ] {
        let text = queue_impl::command_recall_text(&cmd);
        assert_eq!(parse_slash_command(&text[1..]).unwrap(), cmd, "{text}");
    }
}

#[test]
fn test_btw_request_is_system_plus_single_user_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_two_exchanges(&mut app);
    let messages = runtime_impl::btw_request_messages(
        &app.history,
        &app.vision_descriptions,
        "what was the first answer?",
    );
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[1]["role"], "user");
    let user = messages[1]["content"].as_str().unwrap();
    assert!(
        user.contains("first answer"),
        "transcript folded in:\n{user}"
    );
    assert!(
        user.contains("Side question: what was the first answer?"),
        "{user}"
    );
    assert!(!user.contains("omitted"), "nothing trimmed here:\n{user}");

    // Empty history: just the question, no empty transcript wrapper.
    let bare = runtime_impl::btw_request_messages(&[], &app.vision_descriptions, "hi?");
    assert_eq!(bare.len(), 2);
    assert!(!bare[1]["content"].as_str().unwrap().contains("transcript"));
}

#[test]
fn test_btw_request_trims_oldest_turns_over_budget() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for i in 0..5 {
        app.history.push(ChatMessage {
            model: None,
            role: "user".to_string(),
            content: format!("turn-{i} {}", "x".repeat(30_000)),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let messages = runtime_impl::btw_request_messages(&app.history, &app.vision_descriptions, "q?");
    let user = messages[1]["content"].as_str().unwrap();
    assert!(user.contains("(earlier turns omitted)"), "trim is labeled");
    assert!(user.contains("turn-4"), "newest turn survives");
    assert!(!user.contains("turn-0"), "oldest turn is trimmed");
}

#[tokio::test]
async fn test_bare_btw_reopens_last_exchange_or_hints() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // No exchange yet: usage hint, no overlay.
    app.run_btw_command(None).await;
    assert!(matches!(app.overlay, Overlay::None));
    assert!(notice_text(&app).contains("/btw <question>"));

    // With a stored exchange, bare /btw reopens the panel.
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer: "a".to_string(),
        error: None,
    });
    app.run_btw_command(None).await;
    assert!(matches!(app.overlay, Overlay::Btw { scroll: 0 }));
}

#[tokio::test]
async fn test_btw_stream_events_respect_seq_and_finish() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx2 = tx.clone();
    let mut app = make_test_app(tx, rx);
    app.btw_seq = 3;
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer: String::new(),
        error: None,
    });

    // A superseded call's delta (stale seq) is dropped; the live one appends.
    tx2.send(RuntimeEvent::BtwDelta {
        seq: 2,
        text: "stale ".to_string(),
    })
    .unwrap();
    tx2.send(RuntimeEvent::BtwDelta {
        seq: 3,
        text: "Hel".to_string(),
    })
    .unwrap();
    tx2.send(RuntimeEvent::BtwDelta {
        seq: 3,
        text: "lo".to_string(),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.btw.as_ref().unwrap().answer, "Hello");

    // Finish: the assembled reply wins.
    tx2.send(RuntimeEvent::BtwFinished {
        seq: 3,
        result: Ok("Hello there".to_string()),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    let exchange = app.btw.as_ref().unwrap();
    assert_eq!(exchange.answer, "Hello there");
    assert!(exchange.error.is_none());

    // A failed call with nothing streamed reads as an error, not a blank panel.
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer: String::new(),
        error: None,
    });
    tx2.send(RuntimeEvent::BtwFinished {
        seq: 3,
        result: Err("upstream 500".to_string()),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.btw.as_ref().unwrap().error.as_deref(),
        Some("upstream 500")
    );
}

#[test]
fn test_btw_overlay_renders_question_answer_and_streaming_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.btw = Some(BtwExchange {
        question: "what is a monad?".to_string(),
        answer: String::new(),
        error: None,
    });
    app.overlay = Overlay::Btw { scroll: 0 };
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Btw"), "title:\n{screen}");
    assert!(screen.contains("what is a monad?"), "{screen}");
    assert!(screen.contains("thinking…"), "pre-answer state:\n{screen}");

    app.btw.as_mut().unwrap().answer = "a monoid in the category of endofunctors".to_string();
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("monoid in the category"), "{screen}");
    assert!(!screen.contains("thinking…"), "{screen}");
    assert!(
        screen.contains("answered outside the conversation"),
        "off-transcript note:\n{screen}"
    );
}
