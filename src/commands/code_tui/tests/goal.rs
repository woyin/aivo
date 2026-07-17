use super::super::*;
use super::helpers::*;

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

    // Starting mid-turn queues the command for turn end (no send yet).
    app.sending = true;
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert_eq!(
        app.queued_commands,
        vec![SlashCommand::Goal(Some("do it".to_string()))]
    );
    assert!(app.notice.as_ref().unwrap().1.contains("queued"));
    app.sending = false;
    app.queued_commands.clear();

    // A non-agent key (OAuth) is refused (no send).
    app.key.base_url = "claude-oauth".to_string();
    app.run_goal_command(Some("do it".to_string())).await;
    assert!(app.goal_mode.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("native agent"));
}

/// Goal mode sends machine text to the model but must not leak it into ↑/↓
/// recall — `record: None` records nothing; only the typed `/goal <obj>` does.
#[tokio::test]
async fn test_goal_machine_text_never_enters_draft_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.dispatch_user_message("[Goal mode] preamble".to_string(), None)
        .await
        .unwrap();
    assert!(app.draft_history.is_empty());

    app.dispatch_user_message(
        "[Goal mode] preamble".to_string(),
        Some("/goal x".to_string()),
    )
    .await
    .unwrap();
    assert_eq!(app.draft_history, vec!["/goal x".to_string()]);
}

/// The goal loop ends on the completion marker and at the turn cap (both
/// terminal — neither sends another turn). Exercises `signals_goal_complete`.
#[tokio::test]
async fn test_goal_loop_stops_on_marker_and_cap() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let assistant = |content: &str| ChatMessage {
        model: None,
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
        msg_floor: 0,
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
        msg_floor: 0,
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

/// The marker counts when markdown-wrapped (models echo the prompts' backticks);
/// prose around the words still doesn't.
#[tokio::test]
async fn test_goal_marker_tolerates_markdown_wrapping() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let assistant = |content: &str| ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    for reply in [
        "`GOAL COMPLETE`",
        "**GOAL COMPLETE**",
        "All tests pass.\n\ngoal complete.",
        "done\n`GOAL COMPLETE.`",
        "\"GOAL COMPLETE\"",
        "> GOAL COMPLETE",
    ] {
        app.history.clear();
        app.history.push(assistant(reply));
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 2,
            max: 20,
            msg_floor: 0,
        });
        app.maybe_continue_goal().await.unwrap();
        assert!(app.goal_mode.is_none(), "marker ends the loop: {reply:?}");
    }

    // `sending` blocks the continuation dispatch so only the check runs.
    for reply in [
        "I will reply GOAL COMPLETE once everything is finished.",
        "- [ ] GOAL COMPLETE",
    ] {
        app.history.clear();
        app.history.push(assistant(reply));
        app.goal_mode = Some(GoalState {
            objective: "x".to_string(),
            iteration: 2,
            max: 20,
            msg_floor: 0,
        });
        app.sending = true;
        app.maybe_continue_goal().await.unwrap();
        assert!(
            app.goal_mode.is_some(),
            "prose must not end the loop: {reply:?}"
        );
        app.sending = false;
    }
}

/// An errored turn — signalled by the durable `error` transcript row — stops the
/// loop instead of replaying to the cap, and the error notice stays visible.
#[tokio::test]
async fn test_goal_stops_on_errored_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
        msg_floor: 0,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "partial work".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // What `apply_agent_error` records: the ERROR notice + the durable error row.
    app.history.push(ChatMessage {
        model: None,
        role: "error".to_string(),
        content: "LLM error: insufficient credits".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.notice = Some((ERROR(), "LLM error: insufficient credits".to_string()));

    app.maybe_continue_goal().await.unwrap();

    assert!(app.goal_mode.is_none(), "an errored turn ends the loop");
    assert!(!app.sending, "no continuation was sent");
    let (style, msg) = app.notice.clone().unwrap();
    assert_eq!(style, ERROR());
    assert!(msg.contains("insufficient credits"), "error kept: {msg}");
    assert!(msg.contains("goal mode stopped"), "stop noted: {msg}");
}

/// An incidental ERROR notice with no `error` transcript row (e.g. a failed
/// /copy pressed mid-turn) must NOT stop an unattended loop — the continuation
/// still goes out.
#[tokio::test]
async fn test_goal_survives_incidental_error_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = "claude-oauth".to_string();

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 2,
        max: 20,
        msg_floor: 0,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.notice = Some((ERROR(), "Copy failed: no clipboard".to_string()));

    app.maybe_continue_goal().await.unwrap();

    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.contains("Continue toward the goal"),
        "the continuation was dispatched: {}",
        sent.content
    );
    assert_eq!(app.history.last().unwrap().content, "/goal — continue");
}

/// Rows from before the goal armed can't end it: a queued `/goal` restart runs
/// right after a turn whose reply said the marker (or errored) — the fresh loop
/// must survive both.
#[tokio::test]
async fn test_goal_ignores_marker_and_error_below_floor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    for role in ["assistant", "error"] {
        app.history.clear();
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: "GOAL COMPLETE".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
        // Fresh goal armed after that row; its first turn is in flight.
        app.goal_mode = Some(GoalState {
            objective: "new objective".to_string(),
            iteration: 1,
            max: 20,
            msg_floor: app.history.len(),
        });
        app.sending = true;
        app.notice = None;

        app.maybe_continue_goal().await.unwrap();

        assert!(
            app.goal_mode.is_some(),
            "a stale {role} row must not end the fresh goal"
        );
        app.sending = false;
        app.goal_mode = None;
    }
}

/// In `/goal` mode a single Esc only arms a confirm — the loop keeps running;
/// a second consecutive Esc interrupts and stops it.
#[tokio::test]
async fn test_goal_esc_requires_confirmation_to_stop() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
        msg_floor: 0,
    });
    app.sending = true;

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.goal_stop_confirm_pending, "first Esc arms the confirm");
    assert!(app.goal_mode.is_some(), "loop still armed after one Esc");
    assert!(app.sending, "turn not interrupted by one Esc");
    assert!(app.notice.is_some());

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.goal_stop_confirm_pending);
    assert!(app.goal_mode.is_none(), "second Esc stops the loop");
    assert!(!app.sending, "second Esc interrupts the turn");
}

/// A non-Esc key between the two Esc presses disarms the confirm, so the loop
/// keeps running and the next Esc re-arms rather than stopping.
#[tokio::test]
async fn test_goal_esc_confirm_resets_on_other_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 3,
        max: 20,
        msg_floor: 0,
    });
    app.sending = true;

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.goal_stop_confirm_pending);

    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.goal_stop_confirm_pending,
        "other key disarms the confirm"
    );
    assert!(app.notice.is_none(), "confirm notice cleared");
    assert!(app.goal_mode.is_some(), "loop untouched");
    assert!(app.sending);
}

/// The goal turn's marker ends the loop even when a queued user message already
/// started the next turn — otherwise the completed goal burns an extra round.
#[tokio::test]
async fn test_goal_completion_detected_while_queued_message_runs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 2,
        max: 20,
        msg_floor: 0,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done\nGOAL COMPLETE".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // The queued message's user turn is already in flight.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "also rename the module".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;

    app.maybe_continue_goal().await.unwrap();

    assert!(app.goal_mode.is_none(), "marker ends the loop mid-queue");
    assert!(app.notice.as_ref().unwrap().1.contains("Goal complete"));
}

/// The synthetic continuation must not consume a draft typed mid-turn; a
/// plain-chat route also disarms the loop (its finish never auto-continues).
#[tokio::test]
async fn test_goal_continuation_preserves_composer_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();

    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
        msg_floor: 0,
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.draft = "half-typed reply".to_string();
    app.cursor = 4;

    app.maybe_continue_goal().await.unwrap();

    assert_eq!(app.draft, "half-typed reply", "draft survives the dispatch");
    assert_eq!(app.cursor, 4, "cursor survives the dispatch");
    let last = app.history.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(last.content, "/goal — continue");
    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.starts_with("Continue toward the goal"),
        "the continuation still went out: {}",
        sent.content
    );
    assert!(app.goal_mode.is_none(), "plain-chat route disarms the loop");
    assert!(app.notice.as_ref().unwrap().1.contains("plain chat"));
}

/// A guard-stopped turn steers the next `/goal` continuation: the message tells the
/// model not to retry the dead end, and the stored guard-stop is consumed.
#[tokio::test]
async fn test_goal_guard_stop_enriches_continuation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = "claude-oauth".to_string();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
        msg_floor: 0,
    });
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::NoProgress);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.maybe_continue_goal().await.unwrap();

    let last = app.history.last().unwrap();
    assert_eq!(last.role, "user");
    assert_eq!(
        last.content, "/goal — continue",
        "compact transcript marker"
    );
    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.starts_with("[Previous turn stopped early:"),
        "guard-stop should enrich the continuation: {}",
        sent.content
    );
    assert!(
        sent.content.contains("Continue toward the goal: x"),
        "the base continuation restates the objective: {}",
        sent.content
    );
    assert!(
        app.goal_guard_stop.is_none(),
        "the guard-stop is consumed, not resent next turn"
    );
}

/// A step-limit stop — which the old text-match silently ignored — also steers the
/// continuation, telling the model to break the work into smaller pieces.
#[tokio::test]
async fn test_goal_step_limit_steers_continuation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = "claude-oauth".to_string();
    app.goal_mode = Some(GoalState {
        objective: "x".to_string(),
        iteration: 1,
        max: 20,
        msg_floor: 0,
    });
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::StepLimit);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "still working".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.maybe_continue_goal().await.unwrap();

    let sent = app.pending_submit.as_ref().unwrap();
    assert!(
        sent.content.contains("ran out of steps"),
        "step-limit gets its own steering: {}",
        sent.content
    );
}

/// Starting a fresh `/goal` clears any stale guard-stop from a prior loop.
#[tokio::test]
async fn test_goal_guard_stop_cleared_on_new_goal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Force the plain-chat route (image in history, vision unknown) to keep the
    // dispatch lightweight; the guard-stop clear happens before dispatch either way.
    app.model_image_input = None;
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
    app.goal_guard_stop = Some(crate::agent::engine::TurnStop::ToolFailureLoop);

    app.run_goal_command(Some("do the thing".to_string())).await;

    assert!(
        app.goal_guard_stop.is_none(),
        "a fresh goal must not carry a stale guard-stop"
    );
}

/// `/goal` where dispatch falls back to plain chat (image in history, vision
/// unconfirmed) must not arm a loop `finish_response` will never drive.
#[tokio::test]
async fn test_goal_start_disarms_on_plain_chat_route() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = None; // vision support unknown → plain-chat fallback
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

    app.run_goal_command(Some("fix it".to_string())).await;

    assert!(
        app.goal_mode.is_none(),
        "plain-chat route must not arm a loop"
    );
    assert!(
        app.sending,
        "the objective still went out as a plain message"
    );
    assert!(app.notice.as_ref().unwrap().1.contains("plain chat"));
}

/// `/goal` whose first dispatch is refused outright (staged image on a known
/// text-only model) must not leave an armed loop with nothing in flight.
#[tokio::test]
async fn test_goal_start_cleared_when_dispatch_refuses() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_image_input = Some(false); // snapshot says text-only
    app.draft_attachments.push(MessageAttachment {
        name: "shot.png".to_string(),
        mime_type: "image/png".to_string(),
        storage: AttachmentStorage::Inline {
            data: "iVBOR".to_string(),
        },
    });

    app.run_goal_command(Some("describe the screenshot".to_string()))
        .await;

    assert!(
        app.goal_mode.is_none(),
        "refused send must not arm the loop"
    );
    assert!(!app.sending, "nothing went out");
    assert!(app.notice.as_ref().unwrap().1.contains("can't read images"));
    assert_eq!(app.draft_attachments.len(), 1, "attachment kept for resend");
}

/// Entering plan mode stops an active goal loop (mirrors `/goal`'s refusal to
/// start while planning).
#[tokio::test]
async fn test_plan_entry_stops_goal_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.goal_mode = Some(GoalState {
        objective: "ship".to_string(),
        iteration: 3,
        max: 20,
        msg_floor: 0,
    });
    // Image in history + unknown vision pins the kick-off to plain chat — an
    // agent engine build would touch real config/git.
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
    app.model_image_input = None;
    app.run_plan_command(None).await;

    assert!(app.plan_mode, "plan mode is on");
    assert!(app.goal_mode.is_none(), "plan entry ends the goal loop");
    assert!(app.sending, "the kick-off turn went out");
    assert!(app.notice.as_ref().unwrap().1.contains("Goal mode stopped"));
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
            msg_floor: 0,
        });
        app
    };

    let mut app = armed();
    app.cancel_inflight_request(CancelKind::Discard);
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
