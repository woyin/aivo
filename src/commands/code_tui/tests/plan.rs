use super::super::*;
use super::helpers::*;

#[test]
fn test_done_marker_stays_above_new_input_after_plan_clear() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A finished turn: a reply, then a completed plan pinned in its panel. The
    // Done marker is stamped on the last VISIBLE entry (the reply, idx 1).
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first task".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "the reply".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_durations.insert(1, 78_000);

    // The next user message clears the completed plan, then appends.
    app.clear_stale_plan();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "second task".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let plain = app.build_transcript().plain_lines;
    let done = plain.iter().position(|l| l.contains("Done in"));
    let next = plain.iter().position(|l| l.contains("second task"));
    assert!(done.is_some(), "Done marker still shown: {plain:?}");
    assert!(
        done < next,
        "Done marker must stay above the new input: {plain:?}"
    );
}

#[test]
fn test_clear_completed_plan_shifts_index_maps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "a".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "b".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Markers keyed to the entry AFTER the plan (idx 2) must slide to idx 1.
    app.turn_durations.insert(2, 5_000);
    app.turn_pauses.insert(2, 3);
    app.expanded_thinking.insert(2);

    app.clear_stale_plan();

    assert_eq!(app.turn_durations.get(&2), None, "stale key dropped");
    assert_eq!(app.turn_durations.get(&1), Some(&5_000), "shifted down one");
    assert_eq!(app.turn_pauses.get(&1), Some(&3), "pause key shifted too");
    assert!(app.expanded_thinking.contains(&1), "set key shifted too");
}

#[test]
fn test_plan_renders_in_pinned_panel_not_inline() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"scan code","status":"completed"},{"step":"write fix","status":"in_progress"},{"step":"run tests","status":"pending"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    // The plan is pinned in its own panel above the composer — it must NOT render
    // inline in the transcript (where it would scroll away under later content).
    let inline = app.build_transcript().plain_lines.join("\n");
    assert!(
        !inline.contains("Tasks") && !inline.contains("scan code"),
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
    assert!(screen.contains("Tasks"), "panel header missing:\n{screen}");
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
fn test_completed_plan_hidden_from_panel() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"scan code","status":"completed"},{"step":"write fix","status":"completed"}]"#.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let screen = render_screen(&mut app, 80, 20);
    assert!(
        !screen.contains("Tasks") && !screen.contains("scan code"),
        "a fully-done plan must not stay pinned:\n{screen}"
    );
}

#[test]
fn test_long_plan_windows_to_five_with_more_marker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 10 steps: 3 done, step 3 in progress, rest pending.
    let mut plan = Vec::new();
    for i in 0..10 {
        let status = match i {
            0..=2 => "completed",
            3 => "in_progress",
            _ => "pending",
        };
        plan.push(serde_json::json!({"step": format!("step {i}"), "status": status}));
    }
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: serde_json::Value::Array(plan).to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let screen = render_screen(&mut app, 80, 24);
    assert!(
        screen.contains("3/10 done"),
        "full count in header:\n{screen}"
    );
    assert!(
        screen.contains("step 3"),
        "active step must show:\n{screen}"
    );
    assert!(
        screen.contains("more"),
        "hidden steps need a marker:\n{screen}"
    );
    let step_rows = screen
        .lines()
        .filter(|l| l.contains('✔') || l.contains('▸') || l.contains('○'))
        .count();
    assert!(
        step_rows <= 5,
        "at most 5 steps shown, got {step_rows}:\n{screen}"
    );
    assert!(
        !screen.contains("step 0"),
        "collapsed done step leaked:\n{screen}"
    );
}

#[test]
fn test_completed_plan_clears_on_next_user_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plan_count = |a: &CodeTuiApp| a.history.iter().filter(|m| m.role == "plan").count();

    // A finished plan is recorded and stays pinned (nothing clears it on its own).
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(plan_count(&app), 1);

    // The next user message clears it — `send_user_message` runs this before
    // pushing the turn, so a done plan doesn't linger into a new task.
    app.clear_stale_plan();
    assert_eq!(plan_count(&app), 0, "done plan cleared on next message");

    // A mid-execution plan (some pending, some done) is never auto-cleared.
    app.apply_agent_plan(serde_json::json!([
        {"step": "a", "status": "completed"},
        {"step": "b", "status": "pending"},
    ]));
    app.clear_stale_plan();
    assert_eq!(plan_count(&app), 1, "an active plan must not be cleared");

    // An all-pending proposal is stale once the user moves on — cleared on pivot.
    app.apply_agent_plan(serde_json::json!([
        {"step": "a", "status": "pending"},
        {"step": "b", "status": "pending"},
    ]));
    app.clear_stale_plan();
    assert_eq!(
        plan_count(&app),
        0,
        "unstarted proposal cleared on next message"
    );
}

#[test]
fn test_apply_agent_plan_keeps_single_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let count_plans = |app: &CodeTuiApp| app.history.iter().filter(|m| m.role == "plan").count();

    // Two updates with nothing between → one card, updated in place.
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "pending"}]));
    app.apply_agent_plan(serde_json::json!([{"step": "a", "status": "completed"}]));
    assert_eq!(count_plans(&app), 1, "consecutive updates should collapse");
    assert!(app.history.last().unwrap().content.contains("completed"));

    // A plan after real work still keeps ONE card, relocated to the latest point
    // (so the transcript never stacks a near-identical copy after each batch).
    app.history.push(ChatMessage {
        model: None,
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
fn test_parse_plan_command() {
    assert_eq!(
        parse_slash_command("plan").unwrap(),
        SlashCommand::Plan(None)
    );
    assert_eq!(
        parse_slash_command("plan add a cache layer").unwrap(),
        SlashCommand::Plan(Some("add a cache layer".to_string()))
    );
    assert_eq!(
        parse_slash_command("plan go").unwrap(),
        SlashCommand::Plan(Some("go".to_string()))
    );
}

#[test]
fn test_plan_go_message_appends_guidance() {
    use super::super::runtime_impl::plan_go_message;
    let bare = plan_go_message("");
    assert!(bare.contains("approved"));
    assert!(!bare.contains("Additional guidance"));
    let steered = plan_go_message("use the existing retry helper");
    assert!(steered.starts_with(&bare));
    assert!(steered.contains("Additional guidance: use the existing retry helper"));
}

/// The plan-card anchor slides down when an earlier history entry is removed
/// (e.g. an `update_plan` checklist card dropped by `drop_plan_entries`).
#[tokio::test]
async fn test_plan_card_idx_shifts_on_removal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let msg = |role: &str, c: &str| ChatMessage {
        model: None,
        role: role.to_string(),
        content: c.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };
    app.history.clear();
    app.history.push(msg("user", "hi")); // 0
    app.history.push(msg("plan", "[]")); // 1 (checklist card — dropped below)
    app.history.push(msg("assistant", "plan body")); // 2
    app.plan_card_idx = Some(2);
    app.drop_plan_entries();
    assert_eq!(
        app.plan_card_idx,
        Some(1),
        "anchor follows the assistant down"
    );
    assert_eq!(app.history[1].role, "assistant");
}

/// Plan-mode state machine without the dispatch paths (which need a serve):
/// a finished plan-mode turn drafts its reply as the pending plan while the
/// MODE PERSISTS; `stop` leaves the mode; bare while on reports status (with
/// vs without a draft); `go` with nothing pending just guides.
#[tokio::test]
async fn test_plan_capture_discard_and_status() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let assistant = |content: &str| ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    };

    // Bare `/plan` in the mode with nothing drafted points at the composer.
    app.plan_mode = true;
    app.run_plan_command(None).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("describe what to plan")
    );

    // A finished plan-mode turn stashes the reply as the draft — and stays in
    // plan mode (persistent until approved or stopped).
    app.history.push(assistant("1. do X\n2. do Y"));
    app.capture_plan_draft();
    assert!(app.plan_mode, "plan mode persists after a draft");
    assert_eq!(app.pending_plan.as_deref(), Some("1. do X\n2. do Y"));
    assert!(app.notice.as_ref().unwrap().1.contains("/plan go"));
    // The captured reply is anchored as the plan card.
    assert_eq!(
        app.plan_card_idx,
        app.history.iter().rposition(|m| m.role == "assistant")
    );

    // Bare `/plan` with a drafted plan points at the approval card instead.
    app.run_plan_command(None).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("approve the plan card")
    );

    // `/plan stop` leaves plan mode, discarding the draft and the card frame.
    app.run_plan_command(Some("stop".to_string())).await;
    assert!(!app.plan_mode);
    assert!(app.pending_plan.is_none());
    assert!(app.plan_card_idx.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("discarded"));

    // `/plan go` with nothing pending guides instead of dispatching.
    app.run_plan_command(Some("go".to_string())).await;
    assert!(app.notice.as_ref().unwrap().1.contains("No plan yet"));

    // `/plan go <guidance>` routes to execute (first word), not a new objective.
    app.run_plan_command(Some("go also add tests".to_string()))
        .await;
    assert!(app.notice.as_ref().unwrap().1.contains("No plan yet"));

    // An empty reply leaves the draft untouched (all-tool-call turns).
    app.plan_mode = true;
    app.history.push(assistant("   "));
    app.capture_plan_draft();
    assert!(app.pending_plan.is_none(), "blank reply isn't a plan");

    // An interrupt keeps plan mode on (regression: the old one-way read-only
    // restriction leaked past the mode when the engine survived a cancel).
    app.cancel_inflight_request(super::super::CancelKind::Discard);
    assert!(app.plan_mode, "plan mode persists across an interrupt");
    app.run_plan_command(Some("stop".to_string())).await;
    assert!(!app.plan_mode);
}

/// `/plan exit` returns to the mode plan was entered from.
#[tokio::test]
async fn test_plan_exit_restores_prior_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Entered from auto (the ring path): exit lands back on auto.
    app.set_auto_quiet(true);
    assert!(app.enter_plan_mode().await);
    assert!(!app.agent_auto_approve, "plan quiets auto while on");
    app.run_plan_command(Some("exit".to_string())).await;
    assert!(!app.plan_mode);
    assert!(app.agent_auto_approve, "back to auto-approve");
    assert!(notice_text(&app).contains("back to auto-approve"));

    // Entered from default: exit lands on default.
    app.set_auto_quiet(false);
    assert!(app.enter_plan_mode().await);
    app.run_plan_command(Some("exit".to_string())).await;
    assert!(!app.plan_mode && !app.agent_auto_approve);
}

/// Approval-card verdicts, Claude Code-style: 1 = approve + auto-approve,
/// 2 = approve + default mode, 3 = keep planning. (Bare `/plan`'s silent
/// entry is covered in `test_plan_bare_enters_silently`.)
#[tokio::test]
async fn test_plan_mode_enter_and_approval_verdicts() {
    use crate::agent::protocol::PlanDecision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Enter the mode (no engine yet — build-time entry).
    assert!(app.enter_plan_mode().await);
    assert!(app.plan_mode);

    // Approve & auto-approve: mode off, execution continues unattended.
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.cards
        .set_plan_approval(super::super::PendingPlanApproval {
            body: vec![],
            scroll: 0,
            selected: 0,
            reply,
        });
    app.pick_plan_approval_option(0);
    assert!(!app.plan_mode, "approval exits plan mode");
    assert!(!app.plan_exit_pending);
    assert!(app.agent_auto_approve, "option 1 lands in auto mode");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows so the running turn sees it"
    );
    assert_eq!(rx1.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Approve into default mode: mode off, auto-approve stays off.
    app.plan_mode = true;
    let (reply, mut rx2) = tokio::sync::oneshot::channel();
    app.cards
        .set_plan_approval(super::super::PendingPlanApproval {
            body: vec![],
            scroll: 0,
            selected: 0,
            reply,
        });
    app.pick_plan_approval_option(1);
    assert!(!app.plan_mode);
    assert!(!app.agent_auto_approve, "option 2 lands in default mode");
    assert_eq!(rx2.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Keep planning: mode stays on.
    app.plan_mode = true;
    let (reply, mut rx3) = tokio::sync::oneshot::channel();
    app.cards
        .set_plan_approval(super::super::PendingPlanApproval {
            body: vec![],
            scroll: 0,
            selected: 0,
            reply,
        });
    app.pick_plan_approval_option(2);
    assert!(app.plan_mode, "keep-planning stays in plan mode");
    assert_eq!(
        rx3.try_recv().unwrap(),
        Ok(PlanDecision::KeepPlanning { feedback: None })
    );
}

/// A cursor key's approved plan arms the turn-end auto-continue; a native key
/// must not (its engine resumes in-turn).
#[tokio::test]
async fn test_cursor_plan_approval_arms_auto_continue() {
    use crate::agent::protocol::PlanDecision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let approve = |app: &mut super::super::CodeTuiApp| {
        let (reply, rx) = tokio::sync::oneshot::channel();
        app.cards
            .set_plan_approval(super::super::PendingPlanApproval {
                body: vec![],
                scroll: 0,
                selected: 0,
                reply,
            });
        app.pick_plan_approval_option(0);
        rx
    };

    let mut rx1 = approve(&mut app);
    assert_eq!(rx1.try_recv().unwrap(), Ok(PlanDecision::Approve));
    assert!(!app.cursor_plan_go_pending, "native key must not arm");

    app.key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();
    let mut rx2 = approve(&mut app);
    assert_eq!(rx2.try_recv().unwrap(), Ok(PlanDecision::Approve));
    assert!(app.cursor_plan_go_pending, "cursor approval arms");

    // Keep-planning revises within the same turn — must not arm.
    app.cursor_plan_go_pending = false;
    let (reply, _rx3) = tokio::sync::oneshot::channel();
    app.cards
        .set_plan_approval(super::super::PendingPlanApproval {
            body: vec![],
            scroll: 0,
            selected: 0,
            reply,
        });
    app.pick_plan_approval_option(2);
    assert!(!app.cursor_plan_go_pending);
}

/// The armed auto-continue is one-shot: unarmed → no-op; armed while `sending`
/// or after an errored turn → disarms without dispatching.
#[tokio::test]
async fn test_cursor_plan_auto_continue_guards() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = crate::services::cursor_acp::CURSOR_ACP_SENTINEL.to_string();

    // Unarmed → no-op.
    app.maybe_continue_cursor_plan().await.unwrap();
    assert!(app.history.is_empty());
    assert!(!app.sending);

    // Armed but sending → disarm, no send.
    app.cursor_plan_go_pending = true;
    app.sending = true;
    app.maybe_continue_cursor_plan().await.unwrap();
    assert!(!app.cursor_plan_go_pending);
    assert!(app.history.is_empty());
    app.sending = false;

    // Armed after an errored turn → disarm, no send.
    app.history.push(ChatMessage {
        model: None,
        role: "error".to_string(),
        content: "boom".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.cursor_plan_go_pending = true;
    app.maybe_continue_cursor_plan().await.unwrap();
    assert!(!app.cursor_plan_go_pending);
    assert_eq!(app.history.len(), 1, "no continuation row was dispatched");
    assert!(!app.sending);
}

/// Shift+Tab: default → auto → plan → ask → default (Claude Code's ring plus
/// ask). Mid-turn entry arms the live flag instead of skipping.
#[tokio::test]
async fn test_shift_tab_cycles_agent_modes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mode =
        |app: &super::super::CodeTuiApp| (app.agent_auto_approve, app.plan_mode, app.ask_mode);
    assert_eq!(mode(&app), (false, false, false), "starts in default");

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (true, false, false));

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, true, false), "auto cycles into plan");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    );

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "plan cycles into ask");

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, false), "full circle");

    // Mid-turn, plan entry arms the LIVE flag: the running turn restricts at
    // its next tool-call boundary instead of the entry silently degrading.
    app.cycle_agent_mode().await;
    app.sending = true;
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, true, false), "plan enters mid-turn too");
    assert!(
        app.plan_enter_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the running turn sees the entry"
    );

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "ask enters mid-turn too");
    assert!(
        app.ask_enter_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the running turn sees the ask entry"
    );

    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, false), "ask exits mid-turn");
    assert!(app.ask_exit_pending, "engine restore deferred to turn end");
    assert!(
        !app.ask_enter_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the exit clears a queued live entry"
    );
    app.sending = false;

    app.plan_mode = true;
    app.sending = true;
    app.cycle_agent_mode().await;
    assert!(!app.plan_mode);
    assert!(app.ask_mode, "leaving plan mid-turn lands on ask");
    assert!(app.plan_exit_pending, "engine restore deferred to turn end");
    assert!(
        !app.plan_enter_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the exit clears a queued live entry"
    );
}

/// Shift+Tab on a permission card during plan mode exits plan mode (live), enables
/// auto-approve, and allows this call — the only reachable exit while back-to-back
/// plan cards keep coming.
#[tokio::test]
async fn test_permission_card_shift_tab_in_plan_mode_exits_plan_into_auto() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.plan_mode = true;
    app.sending = true; // a card implies a turn in flight
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.cards.set_permission(super::super::PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("cargo build".to_string()),
        once_only: true,
        reply,
    });
    let consumed = app.handle_permission_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
    assert!(consumed);
    assert_eq!(rx1.try_recv().unwrap(), Decision::Allow);
    assert!(!app.plan_mode, "plan mode exited");
    assert!(app.agent_auto_approve, "auto-approve enabled");
    assert!(
        app.plan_exit_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag signals the running turn's engine"
    );
    assert!(app.plan_exit_pending, "turn-end fallback armed");
}

/// A floor prompt (`once_only`) never remembers its decision: a typed `a`
/// resolves as allow-once, not AlwaysAllow.
#[tokio::test]
async fn test_once_only_permission_card_maps_always_to_allow() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.cards.set_permission(super::super::PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf /".to_string()),
        once_only: true,
        reply,
    });
    let consumed = app.handle_permission_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
    assert!(consumed);
    assert_eq!(rx1.try_recv().unwrap(), Decision::Allow);
}

/// An `exit_plan_mode` tool call renders as the plan card (the plan is the
/// payload), not as an opaque `→ exit_plan_mode(…)` row.
#[test]
fn test_exit_plan_mode_call_renders_plan_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "tool_call".to_string(),
        content:
            r#"{"name":"exit_plan_mode","args":{"plan":"1. refactor the gate\n2. add tests"}}"#
                .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let lines = app.build_transcript().lines;
    assert!(
        lines.iter().any(|l| l.plain == "Implementation plan"),
        "plan card header shown"
    );
    assert!(
        lines.iter().any(|l| l.plain.contains("refactor the gate")),
        "plan body shown"
    );
    assert!(
        !lines.iter().any(|l| l.plain.contains("exit_plan_mode")),
        "no raw tool-call row"
    );
}

/// The composer rule shows the persistent `◇ plan` indicator while plan mode is
/// on (and not while it's off), carries the cycle hint, and tints the rule ACCENT.
#[tokio::test]
async fn test_plan_badge_on_composer_rule() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plain = |line: &ratatui::text::Line<'_>| -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    };

    let off = app.composer_rule_line(80);
    assert!(!plain(&off).contains("plan"));
    assert!(plain(&off).contains("default"), "default mode shown");
    // Every mode carries the cycle hint.
    assert!(plain(&off).contains("(Shift+Tab)"));

    app.plan_mode = true;
    let on = app.composer_rule_line(80);
    assert!(plain(&on).contains("◇ plan"));
    assert!(plain(&on).contains("(Shift+Tab)"));
    // The rule dashes tint ACCENT in plan mode (FAINT otherwise).
    let dash_color = |line: &ratatui::text::Line<'_>| {
        line.spans
            .iter()
            .find(|s| s.content.contains('─'))
            .and_then(|s| s.style.fg)
    };
    assert_eq!(
        dash_color(&on),
        Some(ACCENT()),
        "plan rule is accent-tinted"
    );
    assert_eq!(dash_color(&off), Some(FAINT()), "default rule is faint");
}

/// `/plan go` sends machine text — it must not swallow a draft or staged
/// attachment the user prepared mid-planning (same treatment as `/goal`).
#[tokio::test]
async fn test_plan_go_preserves_composer_draft() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key (OAuth) keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();
    app.pending_plan = Some("the plan".to_string());
    app.draft = "note to self".to_string();
    app.cursor = 4;

    app.run_plan_command(Some("go".to_string())).await;

    assert!(app.sending, "the go message went out");
    assert_eq!(app.draft, "note to self", "draft survives the dispatch");
    assert_eq!(app.cursor, 4);
}

/// Bare `/plan` enters the mode silently: no kick-off turn, no card — just the
/// composer hint. The draft is untouched.
#[tokio::test]
async fn test_plan_bare_enters_silently() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.draft = "note to self".to_string();
    app.cursor = 4;

    app.run_plan_command(None).await;

    assert!(app.plan_mode, "bare /plan enters the mode");
    assert!(!app.sending, "no turn is dispatched");
    assert!(
        notice_text(&app).contains("describe what to plan"),
        "the hint points at the composer"
    );
    assert_eq!(app.draft, "note to self", "draft is untouched");
    assert_eq!(app.cursor, 4);
}

/// Bare `/plan` over this session's interrupted checklist enters the mode and
/// points at `/plan resume` — it never auto-continues execution.
#[tokio::test]
async fn test_plan_bare_with_interrupted_checklist_points_at_resume() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"in_progress"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.run_plan_command(None).await;

    assert!(app.plan_mode, "bare /plan enters the mode");
    assert!(!app.sending, "the checklist does not auto-continue");
    assert!(
        notice_text(&app).contains("/plan resume continues it"),
        "the notice points at the explicit path"
    );
}

/// The plan tip is advisory: it upgrades the entry notice in place, but once
/// anything newer shows it drops instead of clobbering.
#[tokio::test]
async fn test_plan_tip_never_clobbers_newer_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx2 = tx.clone();
    let mut app = make_test_app(tx, rx);
    app.notice = Some((MUTED(), "something newer".to_string()));

    tx2.send(RuntimeEvent::PlanHintReady {
        expect: "Plan mode — describe what to plan (read-only until you approve)".to_string(),
        tip: "Plan mode — describe what to plan · an unfinished plan in this directory: /plan list"
            .to_string(),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        notice_text(&app),
        "something newer",
        "a stale tip never replaces a newer notice"
    );

    app.notice = Some((
        MUTED(),
        "Plan mode — describe what to plan (read-only until you approve)".to_string(),
    ));
    tx2.send(RuntimeEvent::PlanHintReady {
        expect: "Plan mode — describe what to plan (read-only until you approve)".to_string(),
        tip: "Plan mode — describe what to plan · an unfinished plan in this directory: /plan list"
            .to_string(),
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(
        notice_text(&app).contains("/plan list"),
        "the live entry notice upgrades to the tip"
    );
}

#[tokio::test]
async fn test_shift_tab_cycles_modes_through_handle_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve && !app.plan_mode);
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "Shift+Tab should enable auto-approve"
    );
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve ON"
    );
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert!(app.plan_mode, "second press cycles into plan");
    assert!(!app.agent_auto_approve, "modes are mutually exclusive");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve OFF"
    );
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.ask_mode && !app.plan_mode && !app.agent_auto_approve,
        "third press cycles into ask"
    );
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.ask_mode && !app.plan_mode && !app.agent_auto_approve,
        "fourth press lands back on default"
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

/// `/plan resume` lists unfinished plans from other sessions in this directory (a
/// draft and a mid-execution checklist), excluding no-plan/mode-only/other-dir/
/// current sessions.
#[tokio::test]
async fn test_plan_resume_picker_lists_unfinished_plans() {
    use crate::services::session_store::PlanState;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/proj".to_string();

    let save = |id: &str, cwd: &str, title: &str| {
        let store = app.session_store.clone();
        let (id, cwd, title) = (id.to_string(), cwd.to_string(), title.to_string());
        async move {
            store
                .save_code_session_with_id(
                    "test",
                    "https://api.anthropic.com",
                    &cwd,
                    &id,
                    "claude",
                    None,
                    &[],
                    &title,
                    "",
                    Default::default(),
                    0.0,
                )
                .await
                .unwrap();
        }
    };
    save("s-plan", "/proj", "fix the gate").await;
    app.session_store
        .set_plan_state(
            "s-plan",
            Some(&PlanState {
                mode: true,
                ask: false,
                draft: Some("1. fix gate\n2. add tests".to_string()),
                steps: None,
            }),
        )
        .await
        .unwrap();
    save("s-executing", "/proj", "nginx cleanup").await;
    app.session_store
        .set_plan_state(
            "s-executing",
            Some(&PlanState {
                mode: false,
                ask: false,
                draft: None,
                steps: Some(serde_json::json!([
                    {"step": "dedupe server blocks", "status": "completed"},
                    {"step": "reload nginx", "status": "pending"}
                ])),
            }),
        )
        .await
        .unwrap();
    save("s-none", "/proj", "no plan here").await;
    save("s-elsewhere", "/other", "different dir").await;
    app.session_store
        .set_plan_state(
            "s-elsewhere",
            Some(&PlanState {
                mode: true,
                ask: false,
                draft: Some("out of scope".to_string()),
                steps: None,
            }),
        )
        .await
        .unwrap();
    save("s-modeonly", "/proj", "mode only").await;
    app.session_store
        .set_plan_state(
            "s-modeonly",
            Some(&PlanState {
                mode: true,
                ask: false,
                draft: None,
                steps: None,
            }),
        )
        .await
        .unwrap();

    app.run_plan_command(Some("resume".to_string())).await;

    let Overlay::Picker(picker) = &app.overlay else {
        panic!(
            "expected the unfinished-plans picker, got notice {:?}",
            app.notice
        );
    };
    assert!(matches!(picker.kind, PickerKind::PlanResume));
    assert_eq!(picker.items.len(), 2, "the drafted + executing plans list");
    let drafted = picker
        .items
        .iter()
        .find(|item| item.label.contains("fix the gate"))
        .expect("drafted plan row");
    assert!(drafted.label.contains("1. fix gate"));
    let PickerValue::PlanResume(PlanCarry::Draft(draft)) = &drafted.value else {
        panic!("expected a draft PlanResume value");
    };
    assert_eq!(draft, "1. fix gate\n2. add tests");
    let executing = picker
        .items
        .iter()
        .find(|item| item.label.contains("nginx cleanup"))
        .expect("executing plan row");
    assert!(
        executing.label.contains("1/2 steps done"),
        "progress in the label: {:?}",
        executing.label
    );
    assert!(executing.label.contains("next: reload nginx"));
    let PickerValue::PlanResume(PlanCarry::Continue(steps)) = &executing.value else {
        panic!("expected a continue PlanResume value");
    };
    assert!(steps.to_string().contains("reload nginx"));
}

/// `/plan resume` with nothing to pick up notices instead of an empty picker.
#[tokio::test]
async fn test_plan_resume_without_plans_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/proj".to_string();

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(matches!(app.overlay, Overlay::None));
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("No unfinished plans")
    );
}

/// Carrying over an unapproved draft enters plan mode, arms `/plan go`, and
/// dispatches a kick-off embedding the plan while the transcript shows `/plan resume`.
#[tokio::test]
async fn test_plan_resume_activation_arms_and_dispatches() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);

    app.resume_plan_from_session(PlanCarry::Draft("1. carried plan".to_string()))
        .await;

    assert!(app.plan_mode, "carry-over enters plan mode");
    assert_eq!(
        app.pending_plan.as_deref(),
        Some("1. carried plan"),
        "draft armed for /plan go before the model re-frames it"
    );
    assert!(app.sending, "the kick-off went out");
    let content = &app.pending_submit.as_ref().unwrap().content;
    assert!(
        content.contains("<carried-over-plan>\n1. carried plan"),
        "machine text embeds the plan: {content:?}"
    );
    assert_eq!(
        app.history.last().unwrap().content,
        "/plan resume",
        "the transcript shows the compact command"
    );
}

/// Carrying over a mid-execution checklist continues directly — no plan mode, no
/// re-approval; the machine text embeds the checklist with its completed marks.
#[tokio::test]
async fn test_plan_resume_continues_executing_checklist() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    let steps: serde_json::Value =
        serde_json::json!([{"step":"a","status":"completed"},{"step":"b","status":"pending"}]);

    app.resume_plan_from_session(PlanCarry::Continue(steps))
        .await;

    assert!(
        !app.plan_mode,
        "an approved plan continues without plan mode"
    );
    assert!(app.pending_plan.is_none(), "nothing to re-approve");
    assert!(app.sending, "the continuation went out");
    let content = &app.pending_submit.as_ref().unwrap().content;
    assert!(
        content.contains("<carried-over-checklist>") && content.contains("\"step\":\"b\""),
        "machine text embeds the checklist: {content:?}"
    );
    assert_eq!(app.history.last().unwrap().content, "/plan resume");
}

/// `/plan resume` with exactly one candidate and no filter skips the picker and
/// carries the plan over directly — the `/new` handoff stays one command.
#[tokio::test]
async fn test_plan_resume_single_candidate_carries_directly() {
    use crate::services::session_store::PlanState;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/proj".to_string();
    app.session_store
        .save_code_session_with_id(
            "test",
            "https://api.anthropic.com",
            "/proj",
            "s-only",
            "claude",
            None,
            &[],
            "the one plan",
            "",
            Default::default(),
            0.0,
        )
        .await
        .unwrap();
    app.session_store
        .set_plan_state(
            "s-only",
            Some(&PlanState {
                mode: true,
                ask: false,
                draft: Some("1. only plan".to_string()),
                steps: None,
            }),
        )
        .await
        .unwrap();
    // Mid-conversation — an empty session would restore the plan's own session.
    seed_two_exchanges(&mut app);
    pin_to_plain_chat(&mut app);

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(
        matches!(app.overlay, Overlay::None),
        "no picker for a single candidate"
    );
    assert!(app.plan_mode, "carried straight into plan mode");
    assert_eq!(app.pending_plan.as_deref(), Some("1. only plan"));
    assert!(app.sending, "the kick-off went out");
}

/// The turn-end persist snapshots an unfinished execution checklist into
/// planState; an all-done checklist clears it (a finished plan isn't resumable).
#[tokio::test]
async fn test_persist_plan_state_tracks_execution_checklist() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "s-exec".to_string();
    app.session_store
        .save_code_session_with_id(
            "test",
            "https://api.anthropic.com",
            "/proj",
            "s-exec",
            "claude",
            None,
            &[],
            "t",
            "",
            Default::default(),
            0.0,
        )
        .await
        .unwrap();
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"in_progress"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.persist_plan_state().await;
    let saved = app
        .session_store
        .get_code_session("s-exec")
        .await
        .unwrap()
        .unwrap();
    let steps = saved
        .plan_state
        .expect("checklist persisted")
        .steps
        .unwrap();
    assert_eq!(steps.as_array().unwrap().len(), 2);

    // All steps done → the snapshot clears on the next persist.
    app.history.last_mut().unwrap().content =
        r#"[{"step":"a","status":"completed"},{"step":"b","status":"completed"}]"#.to_string();
    app.persist_plan_state().await;
    let saved = app
        .session_store
        .get_code_session("s-exec")
        .await
        .unwrap()
        .unwrap();
    assert!(
        saved.plan_state.is_none(),
        "a finished plan isn't resumable"
    );
}

/// `/new` after a mid-execution plan leaves the kept-plan hint; a fresh session
/// with no plan leaves none.
#[tokio::test]
async fn test_new_chat_hands_off_leftover_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"pending"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.start_new_chat().await;

    assert!(app.history.is_empty(), "the new session starts fresh");
    assert!(notice_text(&app).contains("/plan resume continues it"));

    // No plan left → no hint.
    app.start_new_chat().await;
    assert!(app.notice.is_none());
}

/// `/new` after an unapproved DRAFT does not auto-continue (re-approval is a
/// decision) — it only hints at `/plan resume`.
#[tokio::test]
async fn test_new_chat_draft_hints_instead_of_auto() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "plan something".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.plan_mode = true;
    app.pending_plan = Some("1. draft".to_string());

    app.start_new_chat().await;

    let notice = notice_text(&app);
    assert!(
        notice.contains("unapproved draft") && notice.contains("/plan"),
        "draft hint: {notice:?}"
    );
}

/// `/new` after a mid-execution plan keeps it on disk and hints — no card,
/// nothing dispatched; `/plan resume` picks it up.
#[tokio::test]
async fn test_new_keeps_plan_and_hints() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "s-outgoing".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"pending"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.execute_slash_command(SlashCommand::New).await.unwrap();

    assert!(!app.sending, "nothing dispatched");
    let notice = notice_text(&app);
    assert!(
        notice.contains("/plan resume continues it"),
        "hint points at /plan resume: {notice:?}"
    );
}

/// Seed a persisted prior session ("s-old", cwd `/proj`) holding a
/// mid-execution checklist, then leave the app in a fresh "s-new" session.
async fn seed_prior_plan_session(app: &mut super::super::CodeTuiApp) {
    app.real_cwd = "/proj".to_string();
    app.session_id = "s-old".to_string();
    app.session_store
        .save_code_session_with_id(
            "test",
            "https://api.anthropic.com",
            "/proj",
            "s-old",
            "claude",
            None,
            &[],
            "t",
            "",
            Default::default(),
            0.0,
        )
        .await
        .unwrap();
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"pending"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.persist_plan_state().await;
    app.session_id = "s-new".to_string();
    app.history.clear();
    app.plan_state_written.set(false);
}

/// Bare `/plan` never surfaces old plans: with an unfinished plan nearby it
/// still just enters plan mode, and the background tip upgrades the entry
/// notice to point at `/plan list`.
#[tokio::test]
async fn test_bare_plan_with_old_plans_enters_mode_and_tips() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_prior_plan_session(&mut app).await;
    pin_to_plain_chat(&mut app);

    app.run_plan_command(None).await;

    assert!(app.plan_mode, "bare /plan only enters the mode");
    assert!(app.loading_resume.is_none(), "no session restore");
    assert!(!app.sending, "nothing dispatched");
    assert!(notice_text(&app).contains("describe what to plan"));
    // Pump until the spawned tip lands.
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        app.handle_runtime_events().await.unwrap();
        if notice_text(&app).contains("/plan list") {
            break;
        }
    }
    let notice = notice_text(&app);
    assert!(
        notice.contains("an unfinished plan in this directory: /plan list"),
        "tip counts old plans and points at /plan list: {notice:?}"
    );
}

/// `/plan resume` in a FRESH session with one unfinished plan nearby restores
/// that session (full context beats carrying the bare checklist) and arms the
/// auto-continue for when the load lands.
#[tokio::test]
async fn test_plan_resume_fresh_session_restores_plan_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_prior_plan_session(&mut app).await;

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(
        app.loading_resume.is_some(),
        "a fresh session restores the plan's own session"
    );
    assert!(
        app.loading_resume.as_ref().unwrap().continue_plan,
        "the continue turn fires once the load lands"
    );
}

/// The same single candidate mid-conversation only carries the checklist —
/// restoring a session would clobber the current conversation.
#[tokio::test]
async fn test_plan_resume_mid_conversation_carries_checklist() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    seed_prior_plan_session(&mut app).await;
    seed_two_exchanges(&mut app);
    pin_to_plain_chat(&mut app);

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(app.loading_resume.is_none(), "no session swap mid-chat");
    assert!(app.sending, "the carry went out");
    let content = &app.pending_submit.as_ref().unwrap().content;
    assert!(
        content.contains("<carried-over-checklist>") && content.contains("\"step\":\"b\""),
        "machine text embeds the checklist: {content:?}"
    );
}

/// `/plan <objective>` over a drafted plan supersedes it: the draft clears and
/// a managed save file moves to plans/archive/ instead of being destroyed.
#[tokio::test]
async fn test_new_objective_archives_superseded_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "s-old".to_string();
    app.pending_plan = Some("# Old plan\n1. step".to_string());
    let plans_dir = crate::services::paths::plans_dir(app.session_store.config_dir());
    std::fs::create_dir_all(&plans_dir).unwrap();
    let managed = plans_dir.join(format!("old-plan{}", plan_file_suffix("s-old")));
    std::fs::write(&managed, "# Old plan\n").unwrap();
    pin_to_plain_chat(&mut app);

    app.run_plan_command(Some("build something new".to_string()))
        .await;

    assert!(app.pending_plan.is_none(), "the stale draft cleared");
    assert!(!managed.exists(), "the managed file left the plans dir");
    assert!(
        plans_dir
            .join("archive")
            .join(managed.file_name().unwrap())
            .exists(),
        "…and landed in plans/archive/"
    );
}

/// `/plan save` writes the draft plus the execution checklist (as markdown
/// checkboxes) under `<config>/plans/`, slugged from the draft title.
#[tokio::test]
async fn test_plan_save_writes_markdown() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_plan = Some("# Ship the cache\n1. add store".to_string());
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"pending"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.run_plan_command(Some("save".to_string())).await;

    // Managed save: no path to manage — the notice points at /plan resume.
    let notice = notice_text(&app);
    assert!(
        notice.contains("Plan saved") && notice.contains("/plan resume"),
        "{notice:?}"
    );
    let plans_dir = crate::services::paths::plans_dir(app.session_store.config_dir());
    let names = |dir: &std::path::Path| -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    };
    let listed = names(&plans_dir);
    let [name] = listed.as_slice() else {
        panic!("one saved plan file, got {listed:?}");
    };
    assert!(
        name.starts_with("ship-the-cache-") && name.ends_with(".md"),
        "slug + session suffix: {name}"
    );
    let body = std::fs::read_to_string(plans_dir.join(name)).unwrap();
    assert!(
        body.starts_with("# Ship the cache\n1. add store\n"),
        "{body:?}"
    );
    assert!(body.contains("## Progress"), "{body:?}");
    assert!(
        body.contains("- [x] a") && body.contains("- [ ] b"),
        "{body:?}"
    );

    // Re-saving with a new title replaces the session's file (update in place,
    // no accumulation).
    app.pending_plan = Some("# Renamed effort\n1. add store".to_string());
    app.run_plan_command(Some("save".to_string())).await;
    let listed = names(&plans_dir);
    let [name] = listed.as_slice() else {
        panic!("still one saved plan file, got {listed:?}");
    };
    assert!(name.starts_with("renamed-effort-"), "{name}");
}

/// `/plan save <path>` resolves relative to the working dir (creating parents);
/// with no plan at all it only notices.
#[tokio::test]
async fn test_plan_save_explicit_path_and_no_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_plan_command(Some("save".to_string())).await;
    let notice = notice_text(&app);
    assert!(notice.contains("No plan to save"), "{notice:?}");

    let dir = tempfile::tempdir().unwrap();
    app.real_cwd = dir.path().to_string_lossy().into_owned();
    app.pending_plan = Some("1. lone step".to_string());
    app.run_plan_command(Some("save nested/my-plan.md".to_string()))
        .await;

    let body = std::fs::read_to_string(dir.path().join("nested/my-plan.md")).unwrap();
    assert_eq!(body, "1. lone step\n");
    // An explicit path was asked for, so the notice reports it.
    let notice = notice_text(&app);
    assert!(notice.starts_with("Plan saved to "), "{notice:?}");
}

/// `/plan resume <file>` carries the file's contents as a draft into plan mode
/// (the `/plan save` round trip).
#[tokio::test]
async fn test_plan_resume_from_file() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("saved-plan.md"), "# Saved\n1. carry me\n").unwrap();
    app.real_cwd = dir.path().to_string_lossy().into_owned();
    pin_to_plain_chat(&mut app);

    app.run_plan_command(Some("resume saved-plan.md".to_string()))
        .await;

    assert!(app.plan_mode, "file resume enters plan mode");
    assert_eq!(app.pending_plan.as_deref(), Some("# Saved\n1. carry me\n"));
    assert!(app.sending, "the kick-off went out");
    let content = &app.pending_submit.as_ref().unwrap().content;
    assert!(
        content.contains("<carried-over-plan>"),
        "machine text embeds the plan: {content:?}"
    );
}

/// The real-terminal repro: a plan drafted, approved, and executing all inside
/// ONE turn (so nothing ever hit disk) → `/new` → Esc (later) → bare
/// `/plan resume`. `/new` must persist the outgoing session first, or the
/// checklist exists only in the dropped memory and resume finds nothing.
#[tokio::test]
async fn test_new_esc_then_plan_resume_recovers_unpersisted_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "build the blog".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"in_progress"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.execute_slash_command(SlashCommand::New).await.unwrap();
    assert!(
        notice_text(&app).contains("/plan resume continues it"),
        "the hint replaced the old prompt card"
    );

    pin_to_plain_chat(&mut app);
    app.run_plan_command(Some("resume".to_string())).await;

    // Restorable only because /new persisted the abandoned session.
    assert!(
        app.loading_resume.is_some(),
        "the kept-aside session restores"
    );
    assert!(
        app.loading_resume.as_ref().unwrap().continue_plan,
        "the continue turn fires once the load lands"
    );
}

/// Bare `/plan resume` after Esc interrupted execution continues THIS session's
/// plan — the cross-session picker skips the current session, so without the
/// fast path it would claim there are no unfinished plans.
#[tokio::test]
async fn test_plan_resume_continues_interrupted_plan_here() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    pin_to_plain_chat(&mut app);
    app.history.push(ChatMessage {
        model: None,
        role: "plan".to_string(),
        content: r#"[{"step":"a","status":"completed"},{"step":"b","status":"in_progress"}]"#
            .to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(
        !app.plan_mode,
        "an approved plan continues without plan mode"
    );
    assert!(app.sending, "the continuation went out");
    let content = &app.pending_submit.as_ref().unwrap().content;
    assert!(
        content.contains("<interrupted-checklist>") && content.contains("\"step\":\"b\""),
        "same-session framing with the checklist: {content:?}"
    );
    let notice = notice_text(&app);
    assert!(notice.contains("this session's plan"), "{notice:?}");
}

/// Bare `/plan resume` with a drafted (unapproved) plan in this session points
/// at the approval path instead of the picker.
#[tokio::test]
async fn test_plan_resume_with_local_draft_hints_go() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.pending_plan = Some("1. drafted here".to_string());

    app.run_plan_command(Some("resume".to_string())).await;

    assert!(!app.sending, "nothing dispatched for a draft");
    let notice = notice_text(&app);
    assert!(notice.contains("/plan go"), "{notice:?}");
}

/// A finished plan retires the session's managed save file along with its
/// planState (nothing left for the picker to resurrect).
#[tokio::test]
async fn test_finished_plan_removes_managed_file() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "s-done".to_string();
    app.pending_plan = Some("# Done soon\n1. step".to_string());
    app.run_plan_command(Some("save".to_string())).await;
    let plans_dir = crate::services::paths::plans_dir(app.session_store.config_dir());
    assert_eq!(std::fs::read_dir(&plans_dir).unwrap().count(), 1);

    // Plan approved and finished: no draft, no open steps → persist clears.
    app.pending_plan = None;
    app.plan_mode = false;
    app.persist_plan_state().await;

    assert_eq!(
        std::fs::read_dir(&plans_dir).unwrap().count(),
        0,
        "the managed file retires with the plan"
    );
}

/// `/plan resume` lists managed save files whose session has no row of its own
/// (deleted / other directory), while a listed session's file is deduped away.
#[tokio::test]
async fn test_plan_resume_picker_lists_orphan_saved_files() {
    use crate::services::session_store::PlanState;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.real_cwd = "/proj".to_string();
    // One session in this dir with a draft → a session row.
    app.session_store
        .save_code_session_with_id(
            "test",
            "https://api.anthropic.com",
            "/proj",
            "s-live",
            "claude",
            None,
            &[],
            "the live one",
            "",
            Default::default(),
            0.0,
        )
        .await
        .unwrap();
    app.session_store
        .set_plan_state(
            "s-live",
            Some(&PlanState {
                mode: true,
                ask: false,
                draft: Some("1. live plan".to_string()),
                steps: None,
            }),
        )
        .await
        .unwrap();
    let plans_dir = crate::services::paths::plans_dir(app.session_store.config_dir());
    std::fs::create_dir_all(&plans_dir).unwrap();
    // Its managed file must NOT appear as a second row…
    std::fs::write(
        plans_dir.join(format!("live-plan{}", plan_file_suffix("s-live"))),
        "1. live plan\n",
    )
    .unwrap();
    // …but an orphan file (session gone) must.
    std::fs::write(
        plans_dir.join(format!("orphan-plan{}", plan_file_suffix("s-gone"))),
        "# Orphan plan\n1. carry me\n",
    )
    .unwrap();

    app.open_plan_resume_picker(String::new()).await;

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected the unfinished-plans picker");
    };
    let labels: Vec<&str> = picker.items.iter().map(|i| i.label.as_str()).collect();
    assert_eq!(labels.len(), 2, "session row + orphan file row: {labels:?}");
    assert!(labels.iter().any(|l| l.contains("the live one")));
    assert!(
        labels
            .iter()
            .any(|l| l.contains("Orphan plan") && l.contains("saved plan")),
        "{labels:?}"
    );
}

#[test]
fn test_interrupted_plan_clears_on_next_user_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plan_count = |a: &CodeTuiApp| a.history.iter().filter(|m| m.role == "plan").count();
    let mid_execution = serde_json::json!([
        {"step": "a", "status": "completed"},
        {"step": "b", "status": "in_progress"},
    ]);

    app.apply_agent_plan(mid_execution.clone());
    app.last_turn_interrupted = true;
    app.clear_stale_plan();
    assert_eq!(
        plan_count(&app),
        0,
        "interrupted plan cleared on next message"
    );
    assert!(!app.last_turn_interrupted, "consumed by the dispatch");

    // The same card, this time not interrupted.
    app.apply_agent_plan(mid_execution);
    app.clear_stale_plan();
    assert_eq!(plan_count(&app), 1, "a paused plan must not be cleared");
}

#[test]
fn test_apply_agent_plan_empty_drops_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plan_count = |a: &CodeTuiApp| a.history.iter().filter(|m| m.role == "plan").count();
    app.apply_agent_plan(serde_json::json!([
        {"step": "a", "status": "completed"},
        {"step": "b", "status": "pending"},
    ]));
    assert_eq!(plan_count(&app), 1);
    let revision = app.transcript_revision;

    app.apply_agent_plan(serde_json::json!([]));
    assert_eq!(plan_count(&app), 0, "empty plan clears the card");
    assert!(app.unfinished_plan_steps().is_none());
    assert_ne!(app.transcript_revision, revision, "panel re-renders");
}
