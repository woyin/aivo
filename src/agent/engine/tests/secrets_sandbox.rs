use super::super::*;
use super::helpers::*;
use serde_json::json;

/// Reading `.env` is gated: with auto-approve off the card fires, and denying it
/// blocks the read so the key never enters the transcript.
#[tokio::test]
async fn reading_dotenv_prompts_and_deny_blocks_it() {
    let dir = tmp();
    std::fs::write(
        dir.join(".env"),
        "OPENAI_API_KEY=sk-AAAAAAAAAAAAAAAAAAAAAAAA\n",
    )
    .unwrap();
    let read = tool_call_sse("read_file", json!({ "path": ".env" }));
    let port = spawn_sse_sequence(vec![read, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // Auto-approve OFF so the consent gate engages; deny the read.
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: None,
        cwd: dir.as_path(),
        yes: false,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
        plan_exit: None,
    };
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    run_session(&mut engine, &ctx, Some("read env".into()), &mut ui).await;
    assert_eq!(ui.ask_tools, vec!["read_file"]);
    let tool_content: String = engine
        .messages
        .iter()
        .filter(|m| m["role"] == "tool")
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert!(tool_content.contains("denied by user"));
    assert!(!tool_content.contains("sk-AAAAAAAAAAAAAAAAAAAAAAAA"));
}

/// A key-shaped string in a tool result is masked before it reaches the model.
#[tokio::test]
async fn secret_values_are_redacted_from_the_transcript() {
    let dir = tmp();
    std::fs::write(dir.join("notes.txt"), "deploy key AKIAIOSFODNN7EXAMPLE\n").unwrap();
    let read = tool_call_sse("read_file", json!({ "path": "notes.txt" }));
    let port = spawn_sse_sequence(vec![read, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read notes".into()),
        &mut ui,
    )
    .await;
    let tool_content: String = engine
        .messages
        .iter()
        .filter(|m| m["role"] == "tool")
        .filter_map(|m| m["content"].as_str())
        .collect();
    assert!(
        tool_content.contains("<redacted:aws_access_key>"),
        "got: {tool_content}"
    );
    assert!(!tool_content.contains("AKIAIOSFODNN7EXAMPLE"));
    assert!(ui.notices.iter().any(|n| n.contains("redacted")));
}

/// A sandbox-blocked `run_bash` (write outside the workspace) prompts to re-run
/// outside, scoped to `run_bash_unsandboxed`; declining keeps the blocked result. macOS-only.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn sandbox_block_prompts_to_run_unsandboxed_and_respects_deny() {
    if !crate::agent::sandbox::active() {
        return;
    }
    let dir = tmp();
    let home = crate::services::system_env::home_dir().unwrap();
    let outside = home.join(format!("aivo_esc_deny_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let cmd = format!("echo escalated > '{}'", outside.display());
    let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: None,
        cwd: &dir,
        yes: false,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
        plan_exit: None,
    };
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    run_session(&mut engine, &ctx, Some("commit".into()), &mut ui).await;

    let existed = outside.exists();
    let _ = std::fs::remove_file(&outside);
    assert_eq!(ui.ask_tools, vec!["run_bash_unsandboxed"]);
    // Declined → never ran unconfined, so the file was never written…
    assert!(
        !existed,
        "denied escalation still wrote outside the workspace"
    );
    // …and no re-run notice was emitted.
    assert!(
        !ui.notices
            .iter()
            .any(|n| n.contains("outside the workspace sandbox")),
        "unexpected re-run notice on deny: {:?}",
        ui.notices
    );
}

/// Approving the escalation re-runs outside the sandbox, so the blocked out-of-workspace write now succeeds.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn sandbox_block_reruns_outside_when_approved() {
    if !crate::agent::sandbox::active() {
        return;
    }
    let dir = tmp();
    let home = crate::services::system_env::home_dir().unwrap();
    let outside = home.join(format!("aivo_esc_allow_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let cmd = format!("echo escalated > '{}'", outside.display());
    let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: None,
        cwd: &dir,
        yes: false,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
        plan_exit: None,
    };
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    run_session(&mut engine, &ctx, Some("commit".into()), &mut ui).await;

    let existed = outside.exists();
    let contents = std::fs::read_to_string(&outside).unwrap_or_default();
    let _ = std::fs::remove_file(&outside);
    assert_eq!(ui.ask_tools, vec!["run_bash_unsandboxed"]);
    assert!(
        existed,
        "approved escalation did not write outside the workspace"
    );
    assert_eq!(contents.trim(), "escalated");
    assert!(
        ui.notices
            .iter()
            .any(|n| n.contains("outside the workspace sandbox")),
        "missing re-run notice: {:?}",
        ui.notices
    );
}
