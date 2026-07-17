use super::super::*;
use super::helpers::*;
use serde_json::json;

#[cfg(unix)] // every caller is a #[cfg(unix)] hook test; ungated it's dead code on Windows
fn hookset(dir: &std::path::Path, json_str: &str) -> std::sync::Arc<crate::agent::hooks::HookSet> {
    let path = dir.join("hooks.json");
    std::fs::write(&path, json_str).unwrap();
    std::sync::Arc::new(crate::agent::hooks::HookSet::load_from(&path))
}

/// A PreToolUse veto (exit 2) blocks the call before it runs; the stderr reason
/// reaches the model as the tool error.
#[cfg(unix)]
#[tokio::test]
async fn pre_tool_use_hook_veto_blocks_the_call() {
    let dir = tmp();
    let call = tool_call_sse("write_file", json!({"path": "f", "content": "x"}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_hooks(hookset(
        &dir,
        r#"{"hooks":{"PreToolUse":[{"matcher":"write_file","hooks":[
            {"command":"echo writes are frozen >&2; exit 2"}]}]}}"#,
    ));
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("write it".into()),
        &mut ui,
    )
    .await;
    assert!(!dir.join("f").exists(), "vetoed call must not run");
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("blocked by PreToolUse hook: writes are frozen"))
        }),
        "the veto reason must reach the model"
    );
}

/// PostToolUse stdout is folded into the tool result (same pattern as LSP diagnostics).
#[cfg(unix)]
#[tokio::test]
async fn post_tool_use_hook_feedback_lands_in_the_tool_result() {
    let dir = tmp();
    let call = tool_call_sse("write_file", json!({"path": "f", "content": "x"}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_hooks(hookset(
        &dir,
        r#"{"hooks":{"PostToolUse":[{"matcher":"write_file","hooks":[
            {"command":"echo formatting applied"}]}]}}"#,
    ));
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("write it".into()),
        &mut ui,
    )
    .await;
    assert!(dir.join("f").exists(), "allowed call runs");
    let folded = engine.messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("tool")
            && m.get("content").and_then(Value::as_str).is_some_and(|s| {
                s.contains("[PostToolUse hook]") && s.contains("formatting applied")
            })
    });
    assert!(folded, "hook stdout must land in the tool result");
}

/// A Stop-hook refusal (exit 2) feeds guidance back and the turn continues;
/// once the hook allows, the turn converges normally.
#[cfg(unix)]
#[tokio::test]
async fn stop_hook_refusal_continues_the_turn_with_guidance() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![
        FINAL_TEXT_SSE.to_string(), // 1st done attempt → hook refuses
        FINAL_TEXT_SSE.to_string(), // 2nd done attempt → hook allows
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // Refuse once (flag file marks the first refusal), then allow.
    engine.set_hooks(hookset(
        &dir,
        r#"{"hooks":{"Stop":[{"hooks":[
            {"command":"[ -f stop-flag ] || { touch stop-flag; echo also run the linter >&2; exit 2; }"}]}]}}"#,
    ));
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("do the thing".into()),
        &mut ui,
    )
    .await;
    let guided = engine.messages.iter().any(|m| {
        m.get("content")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("also run the linter"))
    });
    assert!(guided, "the refusal guidance must be fed back");
    assert!(
        ui.notices
            .iter()
            .any(|n| n.contains("Stop hook asked the agent to continue")),
        "expected a stop-hook notice: {:?}",
        ui.notices
    );
}
