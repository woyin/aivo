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
    app.expanded_thinking.insert(2);

    app.clear_stale_plan();

    assert_eq!(app.turn_durations.get(&2), None, "stale key dropped");
    assert_eq!(app.turn_durations.get(&1), Some(&5_000), "shifted down one");
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
        !screen.contains("Plan") && !screen.contains("scan code"),
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

/// Approval-card verdicts, Claude Code-style: 1 = approve + auto-approve,
/// 2 = approve + manual approval, 3 = keep planning. (Bare `/plan`'s kick-off
/// dispatch is covered in `test_plan_bare_dispatches_kickoff`.)
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
    app.cards.plan_approval = Some(super::super::PendingPlanApproval {
        body: vec![],
        scroll: 0,
        selected: 0,
        reply,
    });
    app.pick_plan_approval_option(0);
    assert!(!app.plan_mode, "approval exits plan mode");
    assert!(!app.plan_exit_pending);
    assert!(app.agent_auto_approve, "option 1 lands in auto mode");
    assert!(!app.agent_review_edits, "modes are exclusive");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows so the running turn sees it"
    );
    assert_eq!(rx1.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Approve with per-edit review: mode off, review mode on.
    app.plan_mode = true;
    let (reply, mut rx2) = tokio::sync::oneshot::channel();
    app.cards.plan_approval = Some(super::super::PendingPlanApproval {
        body: vec![],
        scroll: 0,
        selected: 0,
        reply,
    });
    app.pick_plan_approval_option(1);
    assert!(!app.plan_mode);
    assert!(!app.agent_auto_approve, "option 2 lands in review mode");
    assert!(app.agent_review_edits, "each edit will show a diff");
    assert!(
        app.review_edits_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live review flag follows mid-turn"
    );
    assert_eq!(rx2.try_recv().unwrap(), Ok(PlanDecision::Approve));

    // Keep planning: mode stays on.
    app.plan_mode = true;
    let (reply, mut rx3) = tokio::sync::oneshot::channel();
    app.cards.plan_approval = Some(super::super::PendingPlanApproval {
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

/// Shift+Tab cycles the agent mode Claude Code-style: default → auto → plan →
/// review → default, with the modes mutually exclusive. Mid-turn the plan step
/// is skipped (the engine can't be restricted while a turn holds it).
#[tokio::test]
async fn test_shift_tab_cycles_agent_modes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mode = |app: &super::super::CodeTuiApp| {
        (
            app.agent_auto_approve,
            app.plan_mode,
            app.agent_review_edits,
        )
    };
    assert_eq!(mode(&app), (false, false, false), "starts in default");

    // default → auto.
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (true, false, false));

    // auto → plan (auto forced off: exclusive).
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, true, false), "auto cycles into plan");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    );

    // plan → review (the drafted plan would survive; here there is none).
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "plan cycles into review");
    assert!(
        app.review_edits_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live review flag follows"
    );

    // review → default.
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, false), "full circle");

    // Mid-turn: auto → review directly — plan entry is skipped while sending.
    app.cycle_agent_mode().await; // default → auto
    app.sending = true;
    app.cycle_agent_mode().await;
    assert_eq!(mode(&app), (false, false, true), "plan is skipped mid-turn");
    app.cycle_agent_mode().await; // review → default
    app.sending = false;

    // Leaving plan mid-turn defers the engine restore (badge flips now).
    app.plan_mode = true;
    app.sending = true;
    app.cycle_agent_mode().await;
    assert!(!app.plan_mode);
    assert!(app.agent_review_edits, "plan still cycles into review");
    assert!(app.plan_exit_pending, "engine restore deferred to turn end");
}

/// Shift+Tab on a permission card during plan mode allows that one call only —
/// it must NOT silently enable auto-approve (plan mode has no auto-approve).
#[tokio::test]
async fn test_permission_card_shift_tab_in_plan_mode_allows_once() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.plan_mode = true;
    let (reply, mut rx1) = tokio::sync::oneshot::channel();
    app.cards.permission = Some(super::super::PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("cargo build".to_string()),
        reply,
    });
    let consumed = app.handle_permission_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT));
    assert!(consumed);
    assert_eq!(rx1.try_recv().unwrap(), Decision::Allow);
    assert!(app.plan_mode, "still planning");
    assert!(!app.agent_auto_approve, "auto-approve NOT enabled");
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
    assert!(plain(&off).contains("normal"), "normal mode shown");
    // Every mode carries the cycle hint.
    assert!(plain(&off).contains("(shift+tab)"));

    app.plan_mode = true;
    let on = app.composer_rule_line(80);
    assert!(plain(&on).contains("◇ plan"));
    assert!(plain(&on).contains("(shift+tab)"));
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

/// Bare `/plan` enters the mode AND dispatches the kick-off turn: the model
/// gets the machine text, the transcript shows the compact `/plan`, and
/// neither a composer draft nor ↑ recall picks it up.
#[tokio::test]
async fn test_plan_bare_dispatches_kickoff() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
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
    app.draft = "note to self".to_string();
    app.cursor = 4;

    app.run_plan_command(None).await;

    assert!(app.plan_mode, "bare /plan enters the mode");
    assert!(app.sending, "the kick-off went out");
    assert_eq!(
        app.pending_submit.as_ref().unwrap().content,
        super::super::runtime_impl::PLAN_KICKOFF_MESSAGE,
        "the model receives the interview instructions"
    );
    assert_eq!(
        app.history.last().unwrap().content,
        "/plan",
        "the transcript shows the compact command, not the machine text"
    );
    assert_eq!(app.draft, "note to self", "draft survives the dispatch");
    assert_eq!(app.cursor, 4);
    assert!(
        app.draft_history.is_empty(),
        "machine text never enters ↑ recall"
    );
}

#[tokio::test]
async fn test_shift_tab_cycles_modes_through_handle_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve && !app.plan_mode);
    // Shift+Tab arrives as BackTab — normal → auto-approve.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "Shift+Tab should enable auto-approve"
    );
    // The shared LIVE flag the running agent turn reads tracks the mode.
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve ON"
    );
    // The Tab+SHIFT form some terminals send — auto-approve → plan mode.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT))
        .await
        .unwrap();
    assert!(app.plan_mode, "second press cycles into plan mode");
    assert!(!app.agent_auto_approve, "modes are mutually exclusive");
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "live flag follows auto-approve OFF"
    );
    // Third press — plan → review.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.plan_mode && app.agent_review_edits);
    // Fourth press — review → default.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.plan_mode && !app.agent_auto_approve && !app.agent_review_edits);
    // Ctrl+O is no longer an auto-approve alias (Shift+Tab only).
    app.handle_key(KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "Ctrl+O no longer toggles auto-approve"
    );
}
