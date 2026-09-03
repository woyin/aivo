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
        streaming: false,
        error: None,
    });
    app.run_btw_command(None).await;
    assert!(matches!(
        app.overlay,
        Overlay::Btw {
            scroll: 0,
            follow: true
        }
    ));
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
        streaming: true,
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
    assert!(!exchange.streaming, "finish clears the streaming state");

    // A failed call with nothing streamed reads as an error, not a blank panel.
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer: String::new(),
        streaming: true,
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
        streaming: true,
        error: None,
    });
    app.overlay = Overlay::Btw {
        scroll: 0,
        follow: true,
    };
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Btw"), "title:\n{screen}");
    assert!(screen.contains("what is a monad?"), "{screen}");
    assert!(screen.contains("thinking…"), "pre-answer state:\n{screen}");
    assert!(screen.contains("answering"), "stream badge:\n{screen}");

    app.btw.as_mut().unwrap().answer = "a monoid in the category of endofunctors".to_string();
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("monoid in the category"), "{screen}");
    assert!(!screen.contains("thinking…"), "{screen}");
    assert!(
        screen.contains("answered outside the conversation"),
        "off-transcript note:\n{screen}"
    );
    assert!(screen.contains("answering"), "still streaming:\n{screen}");

    app.btw.as_mut().unwrap().streaming = false;
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(!screen.contains("answering"), "{screen}");
    assert!(screen.contains("c copy"), "copy hint:\n{screen}");
}

#[test]
fn test_btw_answer_renders_as_markdown() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.btw = Some(BtwExchange {
        question: "how?".to_string(),
        answer: "## Pointers\n\n- first item\n- second item".to_string(),
        streaming: false,
        error: None,
    });
    app.overlay = Overlay::Btw {
        scroll: 0,
        follow: true,
    };
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Pointers"), "{screen}");
    assert!(screen.contains("first item"), "{screen}");
    assert!(!screen.contains("##"), "heading markup consumed:\n{screen}");
}

#[tokio::test]
async fn test_btw_follows_the_tail_until_a_manual_scroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let answer: String = (0..60)
        .map(|i| format!("line-{i:02}\n"))
        .collect::<Vec<_>>()
        .join("");
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer,
        streaming: true,
        error: None,
    });
    app.overlay = Overlay::Btw {
        scroll: 0,
        follow: true,
    };

    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("line-59"), "tail visible:\n{screen}");
    assert!(!screen.contains("line-00"), "head scrolled off:\n{screen}");

    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Btw { follow: false, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("line-00"), "parked at the top:\n{screen}");

    // End re-joins the tail.

    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Btw { follow: true, .. }));
}

#[tokio::test]
async fn test_btw_c_copies_the_answer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.btw = Some(BtwExchange {
        question: "q".to_string(),
        answer: "the answer".to_string(),
        streaming: false,
        error: None,
    });
    app.overlay = Overlay::Btw {
        scroll: 0,
        follow: true,
    };
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("copied")),
        "copy toast: {:?}",
        app.toast.as_ref().map(|t| &t.text)
    );
    assert!(matches!(app.overlay, Overlay::Btw { .. }));

    app.toast = None;
    app.btw.as_mut().unwrap().answer.clear();
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.toast.is_none());
}

#[test]
fn test_btw_box_hugs_its_content_and_caps_its_width() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.btw = Some(BtwExchange {
        question: "does PBKDF2 slow down brute force?".to_string(),
        answer: "Yes — it iterates a PRF.".to_string(),
        streaming: false,
        error: None,
    });

    let body = Rect::new(0, 0, 160, 48);
    let (small, lines) = app.btw_overlay_layout(body);
    assert_eq!(small.width, 96, "width capped");
    assert_eq!(usize::from(small.height), lines.len() + 4, "chrome only");
    assert!(small.height < 12, "hugs the answer: {small:?}");

    let tall_answer = (0..200).map(|i| format!("line {i}\n")).collect::<String>();
    app.btw.as_mut().unwrap().answer = tall_answer;
    let (tall, _) = app.btw_overlay_layout(body);
    assert_eq!(tall.height, centered_rect(72, 88, body).height);
    assert_eq!(
        (tall.x, tall.y),
        (small.x, small.y),
        "anchored, not recentered"
    );
}
