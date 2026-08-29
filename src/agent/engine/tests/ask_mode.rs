use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn ask_mode_hides_mutating_tools_and_offers_no_exit_tool() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    assert!(engine.read_only);
    let names = tool_names(&engine);
    for gone in [
        "write_file",
        "edit_file",
        "multi_edit",
        "subagent",
        "exit_plan_mode",
    ] {
        assert!(
            !names.iter().any(|n| n == gone),
            "{gone} should not be offered in ask mode"
        );
    }
    for kept in ["read_file", "grep", "glob", "list_dir", "run_bash"] {
        assert!(
            names.iter().any(|n| n == kept),
            "{kept} should be offered in ask mode"
        );
    }
    let system = engine.messages[0]["content"].as_str().unwrap();
    assert!(system.contains(crate::agent::ask_mode::ASK_MODE_DIRECTIVE));
    assert!(!system.contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
}

#[test]
fn set_ask_mode_round_trips_tools_and_directive() {
    // Both editor families: edit_file/multi_edit models and apply_patch models.
    for model in ["m", "gpt-5"] {
        let mut engine = AgentEngine::new("/tmp", model, "", &[], &[], 0, 0);
        let before_tools = tool_names(&engine);
        let before_system = engine.messages[0]["content"].as_str().unwrap().to_string();

        engine.set_ask_mode(true);
        engine.set_ask_mode(true); // idempotent on
        engine.set_ask_mode(false);
        engine.set_ask_mode(false); // idempotent off

        assert!(!engine.read_only, "model={model}");
        let mut after = tool_names(&engine);
        let mut before = before_tools.clone();
        after.sort();
        before.sort();
        assert_eq!(after, before, "model={model}: tool set restored exactly");
        assert_eq!(
            engine.messages[0]["content"].as_str().unwrap(),
            before_system,
            "model={model}: directive stripped exactly"
        );
    }
}

#[test]
fn plan_and_ask_mode_are_exclusive_and_swap_cleanly() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let before_tools = tool_names(&engine);
    let before_system = engine.messages[0]["content"].as_str().unwrap().to_string();

    engine.set_plan_mode(true);
    engine.set_ask_mode(true); // leaves plan mode first
    let system = engine.messages[0]["content"].as_str().unwrap();
    assert!(system.contains(crate::agent::ask_mode::ASK_MODE_DIRECTIVE));
    assert!(!system.contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
    assert!(!tool_names(&engine).iter().any(|n| n == "exit_plan_mode"));

    engine.set_plan_mode(true); // leaves ask mode first
    let system = engine.messages[0]["content"].as_str().unwrap();
    assert!(system.contains(crate::agent::plan_mode::PLAN_MODE_DIRECTIVE));
    assert!(!system.contains(crate::agent::ask_mode::ASK_MODE_DIRECTIVE));
    assert!(tool_names(&engine).iter().any(|n| n == "exit_plan_mode"));

    engine.set_plan_mode(false);
    assert!(!engine.read_only);
    let mut after = tool_names(&engine);
    let mut before = before_tools.clone();
    after.sort();
    before.sort();
    assert_eq!(after, before, "tool set restored exactly");
    assert_eq!(
        engine.messages[0]["content"].as_str().unwrap(),
        before_system,
        "no directive residue after cycling both modes"
    );
}

/// The TUI re-runs the per-turn setters AFTER the mode setters, so a tool
/// configured mid-ask must land in the stash, not in the live spec list.
#[test]
fn image_model_set_during_ask_mode_stays_stashed() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    engine.set_image_source(Some(test_generator()));
    assert!(
        !tool_names(&engine).iter().any(|n| n == "generate_image"),
        "configured mid-ask, the tool must stay hidden"
    );
    engine.set_ask_mode(false);
    assert_eq!(
        tool_names(&engine)
            .iter()
            .filter(|n| *n == "generate_image")
            .count(),
        1,
        "exactly one spec after the stash is restored"
    );
}

#[tokio::test]
async fn ask_mode_refuses_write_but_allows_readonly_bash() {
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
    engine.set_ask_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("explain it".into()),
        &mut ui,
    )
    .await;

    let results = tool_result_texts(&engine);
    assert!(
        results.iter().any(|c| c.contains("Ask mode is read-only")),
        "write_file refused with the ask-mode error: {results:?}"
    );
    assert!(
        results.iter().any(|c| c.contains("probe")),
        "read-only bash ran: {results:?}"
    );
    assert!(!dir.join("out.txt").exists(), "no file written in ask mode");
    assert!(engine.read_only, "ask mode persists through the turn");
}

/// Ask-mode bash rides the confirm tier, not plan's Once floor — so an
/// "always allow" answer sticks for the session.
#[tokio::test]
async fn ask_mode_bash_always_allow_remembers_family() {
    let dir = tmp();
    let one = tool_call_sse("run_bash", json!({ "command": "touch probe1.txt" }));
    let two = tool_call_sse("run_bash", json!({ "command": "touch probe2.txt" }));
    let port = spawn_sse_sequence(vec![one, two, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    // yes: false — the interactive shape; `-y` would waive the confirm tier.
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: None,
        cwd: &dir,
        yes: false,
        auto_approve_all: false,
        auto_approve: None,
        plan_exit: None,
        plan_enter: None,
        ask_exit: None,
        ask_enter: None,
    };
    run_session(&mut engine, &ctx, Some("check".into()), &mut ui).await;

    assert_eq!(
        ui.ask_tools,
        vec!["run_bash"],
        "one prompt; the family grant covers the second call"
    );
    assert!(dir.join("probe1.txt").exists() && dir.join("probe2.txt").exists());
    assert!(engine.read_only, "ask mode persists through the turn");
}

/// A hallucinated `exit_plan_mode` must not pop the approval card (the UI would
/// approve) or drop the read-only floor.
#[tokio::test]
async fn ask_mode_refuses_exit_plan_mode() {
    let dir = tmp();
    let exit = tool_call_sse("exit_plan_mode", json!({"plan": "1. do it"}));
    let port = spawn_sse_sequence(vec![exit, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_ask_mode(true);
    let mut ui = CapturingUi {
        plan_decision: Some(PlanDecision::Approve),
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("explain".into()),
        &mut ui,
    )
    .await;

    let results = tool_result_texts(&engine);
    assert!(
        results.iter().any(|c| c.contains("not in plan mode")),
        "exit_plan_mode refused: {results:?}"
    );
    assert!(engine.read_only, "ask mode's read-only floor holds");
}

#[test]
fn outgoing_reminder_matches_the_active_mode() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.push_text_turn("user", "hello".to_string());

    engine.set_ask_mode(true);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(tail.contains("Ask mode is still active"));
    assert!(!tail.contains("Plan mode is still active"));

    engine.set_plan_mode(true);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(tail.contains("Plan mode is still active"));
    assert!(!tail.contains("Ask mode is still active"));

    engine.set_plan_mode(false);
    let out = engine.outgoing_messages();
    let tail = out.last().unwrap()["content"].as_str().unwrap();
    assert!(!tail.contains("still active"), "no reminder outside a mode");
    assert!(
        !engine.messages.iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|c| c.contains("still active"))),
        "reminders never persisted"
    );
}
