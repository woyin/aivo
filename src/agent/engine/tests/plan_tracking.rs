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

    // Model's update (2 steps), then the engine's finalization on convergence.
    assert_eq!(ui.plans, vec![2, 2], "plan_updated should fire twice");
    assert_eq!(
        ui.last_plan,
        vec![PlanStatus::Completed, PlanStatus::Completed],
        "a converged turn finalizes a started plan to all-completed"
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

/// A started plan the model never finished is finalized by the engine on convergence (the "0/N done" stuck-card bug).
#[tokio::test]
async fn engine_finalizes_started_plan_on_convergence() {
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

    assert_eq!(ui.plans, vec![3, 3]);
    assert_eq!(
        ui.last_plan,
        vec![
            PlanStatus::Completed,
            PlanStatus::Completed,
            PlanStatus::Completed
        ],
        "every step is completed once the turn converges"
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
