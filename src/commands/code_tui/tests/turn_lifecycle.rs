use super::super::*;
use super::helpers::*;
use tempfile::TempDir;

/// Mid-turn tool-set changes (skill/MCP toggles, async skill installs) defer the
/// engine drop to turn end — an immediate drop loses the turn's usage + transcript.
#[test]
fn test_request_engine_rebuild_defers_while_sending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.sending = true;
    app.request_engine_rebuild();
    assert!(
        app.engine_rebuild_pending,
        "deferred while a turn is in flight"
    );

    app.sending = false;
    app.maybe_apply_engine_rebuild();
    assert!(!app.engine_rebuild_pending, "applied at turn end");

    // Idle: applies immediately, no pending flag.
    app.request_engine_rebuild();
    assert!(!app.engine_rebuild_pending);
}

/// A non-UTF8 file under a text mime (unknown extension) is refused with a clear
/// error instead of being sent as a base64 blob labeled text/plain.
#[tokio::test]
async fn test_dispatch_refuses_binary_text_attachment() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("blob.bin");
    std::fs::write(&path, [0xffu8, 0xfe, 0x01, 0x00]).unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "blob.bin".to_string(),
        mime_type: "text/plain".to_string(),
        storage: AttachmentStorage::FileRef {
            path: path.to_string_lossy().into_owned(),
        },
    });

    let err = app
        .dispatch_user_message("look at this".to_string(), None)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("binary"), "{err}");
}

#[test]
fn test_restore_cancelled_submission_puts_prompt_back() {
    let mut history = vec![ChatMessage {
        model: None,
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

#[tokio::test]
async fn test_cancel_keeps_user_turn_for_in_process_agent_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
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

    app.cancel_inflight_request(CancelKind::Discard);

    // The engine already consumed this turn (and may have edited files), so the
    // request stays in the transcript instead of being silently un-sent — unlike
    // the plain-chat Discard path, which drops the dangling user message.
    assert_eq!(app.history.len(), 1, "agent user turn must be kept");
    assert_eq!(app.history[0].content, "edit the config");
    assert!(
        app.draft.is_empty(),
        "an agent turn must not be restored to the composer"
    );
    assert!(app.pending_submit.is_none());
    assert!(!app.sending);
}

/// Esc before anything streamed un-sends the turn from the in-process engine
/// too — leaving the bare user message there made the next submit merge with
/// it, so the model saw text the transcript no longer showed.
#[tokio::test]
async fn test_esc_unsend_removes_agent_engine_user_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // The dispatched turn reached the engine: its opening user message is recorded.
    let mut engine = crate::agent::engine::AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.begin_user_turn(
        serde_json::Value::String("hello".into()),
        "hello".to_string(),
    );
    app.agent_engine = Some(AgentSession {
        key_id: "k".to_string(),
        model: "m".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
    });
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hello".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "hello".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    // Esc with nothing streamed → the Unsend path.
    app.interrupt_inflight_request().await.unwrap();

    assert_eq!(app.draft, "hello", "text returned to the composer");
    assert!(app.history.is_empty(), "transcript row un-sent");
    assert!(
        app.agent_unsend_pending,
        "backstop armed for the next dispatch"
    );

    // Apply the dispatch backstop; idempotent with the async apply, so the final
    // state is deterministic whichever ran first.
    let engine = app.agent_engine.as_ref().unwrap().engine.clone();
    let mut engine = engine.lock().await;
    engine.unsend_last_user_turn();
    assert_eq!(
        engine.export_conversation().len(),
        0,
        "the engine's stale user turn is gone"
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
        model: None,
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

/// A leaked tool-call streamed as text must not persist in the scrollback — the
/// engine emits `AgentDiscardSegment` so only the retry's clean answer commits.
#[tokio::test]
async fn test_discard_segment_drops_leaked_markup_from_scrollback() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "<tool_calls>{\"name\":\"read_file\"}</tool_calls>".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    // Engine strips + retries → tells the UI to drop the leaked segment.
    app.tx.send(RuntimeEvent::AgentDiscardSegment).unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.pending_response.is_empty(), "typed reply cleared");
    assert!(app.incoming_buffer.is_empty(), "buffered reply cleared");
    // The retry's real answer streams in fresh.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "done".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.flush_pending_assistant();
    let last = app.history.last().expect("a committed assistant segment");
    assert_eq!(last.content, "done");
    assert!(
        !last.content.contains("<tool_calls>"),
        "leaked markup must never reach the scrollback: {:?}",
        last.content
    );
}

/// Esc on a still-pending request returns the message to the composer, un-sent.
#[tokio::test]
async fn test_interrupt_empty_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first message".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "first message".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert_eq!(
        app.draft, "first message",
        "the pending message returns to the composer"
    );
    assert_eq!(app.cursor, "first message".len());
    assert!(app.pending_submit.is_none());
    assert!(
        app.history.is_empty(),
        "the unanswered user turn is un-sent so resent history stays alternating"
    );

    app.insert_char_at_cursor('!');
    assert_eq!(app.draft, "first message!");
}

/// A draft typed while pending is not clobbered by the un-sent message.
#[tokio::test]
async fn test_interrupt_empty_keeps_typed_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first message".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "first message".to_string(),
        attachments: Vec::new(),
    });
    app.draft = "typed while waiting".to_string();
    app.cursor = app.draft.len();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.interrupt_inflight_request().await.unwrap();

    assert_eq!(
        app.draft, "typed while waiting",
        "a freshly typed draft is not overwritten by the cancelled message"
    );
    assert!(
        app.history.is_empty(),
        "the unanswered user turn is un-sent"
    );
}

/// An agent turn that produced nothing is un-sent too (engine merges on resend).
#[tokio::test]
async fn test_interrupt_empty_agent_turn_restores_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
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
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert_eq!(app.draft, "edit the config");
    assert!(app.pending_submit.is_none());
    assert!(
        app.history.is_empty(),
        "the untouched agent turn is un-sent"
    );
}

/// An agent turn that already ran a tool is kept, not un-sent.
#[tokio::test]
async fn test_interrupt_empty_agent_turn_with_tool_keeps_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "edit the config".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content: "{\"name\":\"edit_file\"}".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_submit = Some(PendingSubmission {
        content: "edit the config".to_string(),
        attachments: Vec::new(),
    });
    app.sending = true;
    app.request_started_at = Some(Instant::now());
    let handle = tokio::spawn(async { anyhow::Ok(()) });
    let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
    app.agent_serve = Some((handle, shutdown));

    app.interrupt_inflight_request().await.unwrap();

    assert!(!app.sending);
    assert!(
        app.draft.is_empty(),
        "a turn that ran a tool is not restored"
    );
    assert_eq!(app.history.len(), 2, "the user + tool rows are kept");
    assert!(app.pending_submit.is_none());
}

/// Watchdog: a task that finished WITHOUT a terminal event (a `run_turn` panic
/// before `ui.footer`) must not leave the turn stuck "sending"; it salvages
/// partial text, resets, and stops the `/goal` loop. A running turn is untouched.
#[tokio::test]
async fn test_recover_dead_response_task_resets_stuck_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
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
        msg_floor: 0,
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

#[test]
fn test_reframe_image_input_error_leads_with_action() {
    use super::super::event_loop_impl::reframe_image_input_error;
    // The provider's stable wording is reframed with an actionable first line,
    // keeping the raw envelope below for debuggability.
    let raw = r#"API returned 400 Bad Request — {"error":{"message":"Error from provider: This model does not support image inputs"}}"#;
    let out = reframe_image_input_error(raw.to_string(), "glm-5.2");
    assert!(out.starts_with("glm-5.2 can't read images"), "got: {out}");
    assert!(out.contains("/model"));
    assert!(out.contains(raw), "raw envelope retained");

    // Unrelated errors pass through untouched.
    let other = "API returned 500 Bad Gateway".to_string();
    assert_eq!(reframe_image_input_error(other.clone(), "glm-5.2"), other);
}

#[tokio::test]
async fn test_history_has_image_detects_image_attachment() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.history_has_image(), "empty history has no image");

    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "just text".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "notes.txt".to_string(),
            mime_type: "text/plain".to_string(),
            storage: AttachmentStorage::Inline {
                data: "abc".to_string(),
            },
        }],
    });
    assert!(
        !app.history_has_image(),
        "a text attachment is not an image"
    );

    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "look".to_string(),
        reasoning_content: None,
        attachments: vec![MessageAttachment {
            name: "shot.png".to_string(),
            mime_type: "image/png".to_string(),
            storage: AttachmentStorage::Inline {
                data: "iVBOR".to_string(),
            },
        }],
    });
    assert!(app.history_has_image(), "image attachment detected");
}

#[tokio::test]
async fn test_preflight_refuses_image_on_known_text_only_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "glm-5.1".to_string();
    app.model_image_input = Some(false); // snapshot says text-only
    app.draft_attachments.push(MessageAttachment {
        name: "shot.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    });

    app.dispatch_user_message("what's in this".to_string(), None)
        .await
        .unwrap();

    let (style, msg) = app.notice.clone().expect("a refusal notice is shown");
    assert_eq!(style, ERROR());
    assert!(msg.contains("can't read images"), "got: {msg}");
    // The draft + attachment survive so the user can switch models and resend;
    // nothing was sent.
    assert_eq!(app.draft_attachments.len(), 1, "attachment retained");
    assert!(app.history.is_empty(), "no user turn was pushed");
    assert!(!app.sending, "no turn started");
}
