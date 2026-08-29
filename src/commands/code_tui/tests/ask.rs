use super::super::*;
use super::helpers::*;

#[tokio::test]
async fn test_ask_command_enters_and_exits_with_prior_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Entered from auto-approve: exit returns there.
    app.set_auto_quiet(true);
    app.run_ask_command(None).await;
    assert!(app.ask_mode && !app.agent_auto_approve, "ask quiets auto");
    assert!(!app.plan_mode, "modes are exclusive");
    app.run_ask_command(Some("exit".to_string())).await;
    assert!(!app.ask_mode);
    assert!(app.agent_auto_approve, "back to auto-approve");
    let notice = app.notice.as_ref().map(|(_, m)| m.clone()).unwrap();
    assert!(notice.contains("back to auto-approve"), "{notice}");

    // Entered from default: exit lands on default.
    app.set_auto_quiet(false);
    app.run_ask_command(None).await;
    assert!(app.ask_mode);
    app.run_ask_command(Some("exit".to_string())).await;
    assert!(!app.ask_mode && !app.agent_auto_approve);

    // Exit while off: friendly notice, nothing flips.
    app.run_ask_command(Some("exit".to_string())).await;
    let notice = app.notice.as_ref().map(|(_, m)| m.clone()).unwrap();
    assert!(notice.contains("isn't on"), "{notice}");
}

#[tokio::test]
async fn test_ask_command_leaves_plan_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.enter_plan_mode().await);
    app.pending_plan = Some("1. do X".to_string());
    app.run_ask_command(None).await;
    assert!(app.ask_mode && !app.plan_mode, "ask replaces plan");
    assert!(
        app.pending_plan.is_some(),
        "the drafted plan survives the mode swap"
    );
}

#[test]
fn test_ask_badge_on_composer_rule() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let plain = |line: &ratatui::text::Line<'_>| -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    };
    app.ask_mode = true;
    let on = app.composer_rule_line(80);
    assert!(plain(&on).contains("◆ ask"));
    assert!(plain(&on).contains("(Shift+Tab)"));
    let dash_color = on
        .spans
        .iter()
        .find(|s| s.content.contains('─'))
        .and_then(|s| s.style.fg);
    assert_eq!(dash_color, Some(INFO()), "rule tints INFO in ask mode");
    assert_ne!(INFO(), ACCENT(), "ask must not share plan's hue");
}

#[test]
fn test_ask_mode_captured_in_plan_state() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.current_plan_state().is_none(), "no mode, no snapshot");
    app.ask_mode = true;
    let state = app.current_plan_state().expect("ask mode snapshots");
    assert!(state.ask && !state.mode);
    assert!(state.draft.is_none() && state.steps.is_none());
}

/// Read-only modes change what typing does, so the generic placeholder misleads.
#[test]
fn test_placeholder_follows_the_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let placeholder = |app: &CodeTuiApp| -> String {
        app.render_composer_text().lines[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect()
    };

    assert!(placeholder(&app).contains("Ask, plan, or build"));

    app.ask_mode = true;
    let p = placeholder(&app);
    assert!(p.contains("concepts, docs, code"), "{p}");

    app.ask_mode = false;
    app.plan_mode = true;
    let p = placeholder(&app);
    assert!(p.contains("what to plan"), "{p}");

    // Mid-turn queueing wins over any mode placeholder.
    app.sending = true;
    let p = placeholder(&app);
    assert!(p.contains("queue"), "{p}");
}

#[tokio::test]
async fn test_ask_entry_hints_when_web_search_is_off() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.web_search_enabled = false;
    app.run_ask_command(None).await;
    let notice = app.notice.as_ref().map(|(_, m)| m.clone()).unwrap();
    assert!(notice.contains("web search is off"), "{notice}");

    app.run_ask_command(Some("exit".to_string())).await;
    app.web_search_enabled = true;
    app.run_ask_command(None).await;
    let notice = app.notice.as_ref().map(|(_, m)| m.clone()).unwrap();
    assert!(!notice.contains("web search is off"), "{notice}");
}

/// auto-approve → /ask → /new must land back in auto-approve, not default.
#[tokio::test]
async fn test_new_chat_restores_standing_mode_from_ask_and_plan() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.set_auto_quiet(true);
    app.run_ask_command(None).await;
    assert!(app.ask_mode && !app.agent_auto_approve);
    app.start_new_chat().await;
    assert!(!app.ask_mode, "ask mode dies with the conversation");
    assert!(app.agent_auto_approve, "auto-approve comes back after /new");

    assert!(app.enter_plan_mode().await);
    assert!(app.plan_mode && !app.agent_auto_approve);
    app.start_new_chat().await;
    assert!(!app.plan_mode);
    assert!(app.agent_auto_approve, "plan analog restores too");

    // From default, /new stays default.
    app.set_auto_quiet(false);
    app.run_ask_command(None).await;
    app.start_new_chat().await;
    assert!(!app.ask_mode && !app.agent_auto_approve);
}

/// A swap must not overwrite the original standing mode with the mode it left.
#[tokio::test]
async fn test_mode_swap_carries_prior_standing_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.set_auto_quiet(true);
    assert!(app.enter_plan_mode().await);
    app.run_ask_command(None).await;
    assert!(app.ask_mode && !app.plan_mode);
    app.run_ask_command(Some("exit".to_string())).await;
    assert!(
        app.agent_auto_approve,
        "the auto prior survives the plan→ask swap"
    );

    // And the reverse: ask → plan keeps it too.
    app.run_ask_command(None).await;
    assert!(app.enter_plan_mode().await);
    app.run_plan_command(Some("exit".to_string())).await;
    assert!(
        app.agent_auto_approve,
        "the auto prior survives the ask→plan swap"
    );
}

/// An autonomous execution loop contradicts the read-only floor (as in plan mode).
#[tokio::test]
async fn test_goal_refused_in_ask_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_ask_command(None).await;
    app.run_goal_command(Some("get france time".to_string()))
        .await;
    assert!(app.goal_mode.is_none(), "no goal loop under ask mode");
    assert!(app.ask_mode, "ask mode stays on");
    let notice = app.notice.as_ref().map(|(_, m)| m.clone()).unwrap();
    assert!(notice.contains("/ask exit before /goal"), "{notice}");
}

#[tokio::test]
async fn test_permission_card_shift_tab_in_ask_mode_exits_ask_into_auto() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.ask_mode = true;
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
    assert!(!app.ask_mode, "ask mode exited");
    assert!(app.agent_auto_approve, "auto-approve enabled");
    assert!(
        app.ask_exit_flag.load(std::sync::atomic::Ordering::Relaxed),
        "live flag signals the running turn's engine"
    );
    assert!(app.ask_exit_pending, "turn-end fallback armed");
}

#[tokio::test]
async fn test_set_approval_mode_leaves_ask_mode() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_ask_command(None).await;
    assert!(app.ask_mode);
    app.set_approval_mode("auto-approve").await;
    assert!(!app.ask_mode && app.agent_auto_approve);
}
