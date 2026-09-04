use super::super::*;
use super::helpers::*;
use crate::agent::plan::PlanStatus;
use crate::agent::request::content_str;
use serde_json::json;

/// An `update_plan` call is intercepted by the engine: it drives the plan
/// card (`plan_updated`), is NOT rendered as a generic tool step, and feeds a
/// confirmation back so the conversation converges on the next turn.
#[tokio::test]
async fn engine_handles_update_plan() {
    let dir = tmp();
    let plan = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "read", "status": "completed"},
            {"step": "edit", "status": "in_progress"}
        ]}),
    );
    let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("do the thing".into()),
        &mut ui,
    )
    .await;

    // Only the model's update fires — convergence never rewrites step status.
    assert_eq!(ui.plans, vec![2], "plan_updated should fire once");
    assert_eq!(
        ui.last_plan,
        vec![PlanStatus::Completed, PlanStatus::InProgress],
        "statuses stay as the model set them"
    );
    assert!(
        !ui.tools.contains(&"update_plan".to_string()),
        "update_plan must not render as a generic tool step"
    );
    assert_eq!(ui.text, "done");
    // The tool result was fed back into history (call ↔ result invariant).
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m).contains("Plan updated")),
        "missing update_plan confirmation in history"
    );
}

/// A started plan the model never finished stays honestly unfinished on
/// convergence — `update_plan` is the only source of step status.
#[tokio::test]
async fn engine_leaves_unfinished_plan_honest_on_convergence() {
    let dir = tmp();
    let plan = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "investigate", "status": "in_progress"},
            {"step": "fix", "status": "pending"},
            {"step": "verify", "status": "pending"}
        ]}),
    );
    let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("do the thing".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.plans, vec![3]);
    assert_eq!(
        ui.last_plan,
        vec![
            PlanStatus::InProgress,
            PlanStatus::Pending,
            PlanStatus::Pending
        ],
        "no step is fabricated as completed on convergence"
    );
}

/// An all-pending plan at convergence gets one nudge; if the model still
/// stops, the `started` gate must not fabricate completion.
#[tokio::test]
async fn engine_nudges_unstarted_plan_once_then_leaves_it_alone() {
    let dir = tmp();
    let plan = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "a", "status": "pending"},
            {"step": "b", "status": "pending"}
        ]}),
    );
    let port = spawn_sse_sequence(vec![
        plan,
        FINAL_TEXT_SSE.to_string(),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan only".into()),
        &mut ui,
    )
    .await;

    let nudges = engine
        .messages
        .iter()
        .filter(|m| {
            role(m) == "user" && content_str(m).contains("haven't started any of its steps")
        })
        .count();
    assert_eq!(nudges, 1, "unstarted plan gets exactly one nudge");
    assert_no_consecutive_user(&engine.messages);
    assert_eq!(engine.messages.last().unwrap()["content"], "done");
    // Only the model's event fired — no engine finalization.
    assert_eq!(ui.plans, vec![2]);
    assert_eq!(ui.last_plan, vec![PlanStatus::Pending, PlanStatus::Pending]);
}

/// Plan mode proposes without executing — no unstarted-plan nudge there.
#[tokio::test]
async fn plan_mode_skips_unstarted_plan_nudge() {
    let dir = tmp();
    let plan = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "a", "status": "pending"},
            {"step": "b", "status": "pending"}
        ]}),
    );
    let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan only".into()),
        &mut ui,
    )
    .await;

    assert!(
        !engine
            .messages
            .iter()
            .any(|m| content_str(m).contains("haven't started any of its steps")),
        "plan mode must not nudge an unstarted plan"
    );
    assert_eq!(engine.messages.last().unwrap()["content"], "done");
}

/// A stale unstarted plan from an earlier turn must not nudge a later turn.
#[tokio::test]
async fn stale_plan_from_prior_turn_does_not_nudge() {
    let dir = tmp();
    let plan = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "a", "status": "pending"},
            {"step": "b", "status": "pending"}
        ]}),
    );
    // Turn 1: plan, nudged converge, stop. Turn 2: plain answer.
    let port = spawn_sse_sequence(vec![
        plan,
        FINAL_TEXT_SSE.to_string(),
        FINAL_TEXT_SSE.to_string(),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    let ctx = turn_ctx(&client, &base, &dir);
    run_session(&mut engine, &ctx, Some("plan only".into()), &mut ui).await;
    let nudges_after_turn1 = engine
        .messages
        .iter()
        .filter(|m| content_str(m).contains("haven't started any of its steps"))
        .count();
    run_session(
        &mut engine,
        &ctx,
        Some("unrelated question".into()),
        &mut ui,
    )
    .await;

    let nudges = engine
        .messages
        .iter()
        .filter(|m| content_str(m).contains("haven't started any of its steps"))
        .count();
    assert_eq!(nudges_after_turn1, 1);
    assert_eq!(nudges, 1, "the stale plan must not nudge a later turn");
    assert_eq!(engine.messages.last().unwrap()["content"], "done");
}

/// The transcript as an Esc left it: a tool batch cut off before its results.
fn engine_with_interrupted_plan(dir: &std::path::Path) -> AgentEngine {
    use crate::agent::plan::PlanItem;
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine
        .messages
        .push(json!({"role":"user","content":"migrate the meter view"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[{"id":"p1","type":"function","function":{"name":"update_plan","arguments":"{}"}}]
    }));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"p1","content":"Plan updated (1/2 done)"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[{"id":"b1","type":"function","function":{"name":"run_bash","arguments":"{}"}}]
    }));
    engine.plan = vec![
        PlanItem {
            step: "edit".into(),
            status: PlanStatus::Completed,
        },
        PlanItem {
            step: "verify".into(),
            status: PlanStatus::InProgress,
        },
    ];
    engine
}

#[test]
fn interrupted_turn_drops_unfinished_plan_and_reminds() {
    use super::super::conversation::PLAN_INTERRUPTED_REMINDER;
    let dir = tmp();
    let mut engine = engine_with_interrupted_plan(&dir);

    engine.begin_user_turn(json!("unrelated question"), "unrelated question".into());

    assert!(engine.plan.is_empty(), "interrupted plan must be dropped");
    assert!(engine.plan_interrupted);
    let out = engine.outgoing_messages();
    assert!(
        content_str(out.last().unwrap()).contains(PLAN_INTERRUPTED_REMINDER),
        "reminder must ride the request tail: {:?}",
        out.last()
    );
    assert!(
        !engine
            .messages
            .iter()
            .any(|m| content_str(m).contains("<system-reminder>")),
        "reminder must not persist in history"
    );
}

#[test]
fn normally_ended_turn_keeps_unfinished_plan() {
    let dir = tmp();
    let mut engine = engine_with_interrupted_plan(&dir);
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"b1","content":"ok"}));
    engine.messages.push(json!({
        "role":"assistant",
        "content":"Which channel should the migration use?"
    }));

    engine.begin_user_turn(json!("use dev"), "use dev".into());

    assert_eq!(engine.plan.len(), 2, "a paused plan is kept");
    assert!(!engine.plan_interrupted);
    assert!(!content_str(engine.outgoing_messages().last().unwrap()).contains("<system-reminder>"));
}

#[tokio::test]
async fn interrupted_plan_card_clears_then_returns_when_resent() {
    let dir = tmp();
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let mut ui = CapturingUi::default();

    // Turn A: unrelated request, the model never touches the plan.
    let port = spawn_sse_sequence(vec![FINAL_TEXT_SSE.to_string()]);
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = engine_with_interrupted_plan(&dir);
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("unrelated question".into()),
        &mut ui,
    )
    .await;
    assert_eq!(ui.plans, vec![0], "card cleared once at turn entry");
    assert!(engine.plan.is_empty());
    assert!(!engine.plan_interrupted, "reminder is one-request only");

    // Turn B: fresh interrupted state, the model continues and re-sends.
    let resend = tool_call_sse(
        "update_plan",
        json!({"plan": [
            {"step": "edit", "status": "completed"},
            {"step": "verify", "status": "in_progress"}
        ]}),
    );
    let port = spawn_sse_sequence(vec![resend, FINAL_TEXT_SSE.to_string()]);
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = engine_with_interrupted_plan(&dir);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("continue".into()),
        &mut ui,
    )
    .await;
    assert_eq!(
        ui.plans,
        vec![0, 2],
        "cleared, then restored by the re-send"
    );
    assert_eq!(
        ui.last_plan,
        vec![PlanStatus::Completed, PlanStatus::InProgress]
    );
    assert_eq!(engine.plan.len(), 2);
}
