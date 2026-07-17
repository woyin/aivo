use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn plan_mode_hides_mutating_tools_but_keeps_bash() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    assert!(engine.read_only);
    let names = tool_names(&engine);
    // File-mutating built-ins + subagent are stripped; run_bash stays
    // (confirmation-gated per call), as do read-only tools + plan/notes.
    for gone in ["write_file", "edit_file", "multi_edit", "subagent"] {
        assert!(
            !names.iter().any(|n| n == gone),
            "{gone} should be hidden in plan mode"
        );
    }
    for kept in [
        "read_file",
        "grep",
        "glob",
        "list_dir",
        "run_bash",
        "update_plan",
        "exit_plan_mode",
    ] {
        assert!(
            names.iter().any(|n| n == kept),
            "{kept} should be offered in plan mode"
        );
    }
    let system = engine.messages[0]["content"].as_str().unwrap();
    assert!(system.contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
}

#[test]
fn set_plan_mode_round_trips_tools_and_directive() {
    // Both editor families: edit_file/multi_edit models and apply_patch models.
    for model in ["m", "gpt-5"] {
        let mut engine = AgentEngine::new("/tmp", model, "", &[], &[], 0, 0);
        let before_tools = tool_names(&engine);
        let before_system = engine.messages[0]["content"].as_str().unwrap().to_string();

        engine.set_plan_mode(true);
        engine.set_plan_mode(true); // idempotent on
        engine.set_plan_mode(false);
        engine.set_plan_mode(false); // idempotent off

        assert!(!engine.read_only, "model={model}");
        let mut after = tool_names(&engine);
        let mut before = before_tools.clone();
        after.sort();
        before.sort();
        assert_eq!(after, before, "model={model}: tool set restored exactly");
        assert!(!after.iter().any(|n| n == "exit_plan_mode"));
        assert_eq!(
            engine.messages[0]["content"].as_str().unwrap(),
            before_system,
            "model={model}: directive stripped exactly"
        );
    }
}

/// In plan mode a non-allowlisted `run_bash` call prompts EVERY time —
/// neither `-y` (ctx.yes) nor an `AlwaysAllow` answer suppresses the next one.
#[tokio::test]
async fn plan_mode_bash_always_prompts() {
    let dir = tmp();
    // `touch` is not in the read-only allowlist — it mutates the workspace.
    let bash = tool_call_sse("run_bash", json!({ "command": "touch probe.txt" }));
    let port = spawn_sse_sequence(vec![bash.clone(), bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir), // yes: true — must not bypass either
        Some("inspect".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.ask_tools, vec!["run_bash", "run_bash"]);
    assert!(
        dir.join("probe.txt").exists(),
        "the confirmed command actually ran"
    );
    assert!(engine.read_only, "plan mode persists through the turn");
}

/// A recognized read-only inspection command runs in plan mode with NO
/// prompt, even with auto-approve off — it can't break the read-only promise.
#[tokio::test]
async fn plan_mode_readonly_bash_skips_prompt() {
    let dir = tmp();
    let bash = tool_call_sse(
        "run_bash",
        json!({ "command": format!("cd {} && git diff --cached --stat", dir.display()) }),
    );
    let echo = tool_call_sse("run_bash", json!({ "command": "echo readonly-probe" }));
    let port = spawn_sse_sequence(vec![bash, echo, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi::default();
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: None,
        cwd: &dir,
        yes: false, // no auto-approve anywhere — the exemption alone applies
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
    };
    run_session(&mut engine, &ctx, Some("inspect".into()), &mut ui).await;

    assert_eq!(ui.asks, 0, "read-only inspection never prompts");
    assert!(
        tool_result_texts(&engine)
            .iter()
            .any(|c| c.contains("readonly-probe")),
        "the commands ran"
    );
    assert!(engine.read_only, "plan mode persists");
}

/// A plan-mode batch: `write_file` is refused with the steering error while a
/// read-only `run_bash` in the same batch runs (promptless — allowlisted).
#[tokio::test]
async fn plan_mode_refuses_write_but_allows_readonly_bash() {
    let dir = tmp();
    let batch = batch_tool_call_sse(&[
        (
            "c1",
            "write_file",
            json!({"path": "out.txt", "content": "hi"}),
        ),
        ("c2", "run_bash", json!({"command": "echo probe"})),
    ]);
    let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan it".into()),
        &mut ui,
    )
    .await;

    let results = tool_result_texts(&engine);
    assert!(
        results.iter().any(|c| c.contains("Plan mode is read-only")),
        "write_file refused: {results:?}"
    );
    assert!(
        results.iter().any(|c| c.contains("probe")),
        "confirmed bash ran: {results:?}"
    );
    assert!(
        !dir.join("out.txt").exists(),
        "no file written in plan mode"
    );
}

/// Approval mid-turn restores full tools so the SAME turn continues into
/// execution: exit_plan_mode → approve → write_file succeeds → converges.
#[tokio::test]
async fn exit_plan_mode_approve_restores_and_continues() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. write out.txt"}));
    let port = spawn_sse_sequence(vec![
        exit,
        WRITE_TOOL_SSE.to_string(),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Approve),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan then build".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.approved_plans, vec!["1. write out.txt"]);
    assert!(!engine.read_only, "approval exits plan mode");
    let results = tool_result_texts(&engine);
    assert!(
        results
            .iter()
            .any(|c| c.contains(plan_mode::PLAN_APPROVED_RESULT)),
        "approval result fed back: {results:?}"
    );
    assert!(
        dir.join("out.txt").exists(),
        "the same turn continued into a successful write"
    );
}

#[tokio::test]
async fn exit_plan_mode_keep_planning_stays_read_only() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. do X"}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::KeepPlanning {
            feedback: Some("cover the error path too".into()),
        }),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan it".into()),
        &mut ui,
    )
    .await;

    assert!(engine.read_only, "keep-planning stays in plan mode");
    let results = tool_result_texts(&engine);
    assert!(
        results
            .iter()
            .any(|c| c.contains("cover the error path too")),
        "feedback fed back: {results:?}"
    );
}

/// Discard tells the model to stop but the engine stays read-only for the rest
/// of the turn (the TUI exits the mode after the turn ends).
#[tokio::test]
async fn exit_plan_mode_discard_stays_read_only() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. do X"}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Discard),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan it".into()),
        &mut ui,
    )
    .await;

    assert!(engine.read_only);
    assert!(
        tool_result_texts(&engine)
            .iter()
            .any(|c| c.contains(plan_mode::PLAN_DISCARDED_RESULT))
    );
}

/// An empty `plan` argument is a steering error, not a card.
#[tokio::test]
async fn exit_plan_mode_empty_plan_errors() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Approve),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("plan it".into()),
        &mut ui,
    )
    .await;

    assert!(ui.approved_plans.is_empty(), "no card for an empty plan");
    assert!(engine.read_only);
    assert!(
        tool_result_texts(&engine)
            .iter()
            .any(|c| c.contains("missing `plan`"))
    );
}

/// A hallucinated call outside plan mode gets a steering error (never a card).
#[tokio::test]
async fn exit_plan_mode_outside_plan_mode_errors() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. do X"}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Approve),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("hi".into()),
        &mut ui,
    )
    .await;

    assert!(ui.approved_plans.is_empty());
    assert!(
        tool_result_texts(&engine)
            .iter()
            .any(|c| c.contains("not in plan mode"))
    );
}
