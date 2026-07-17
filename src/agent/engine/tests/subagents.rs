use super::super::*;
use super::helpers::*;
use crate::agent::request::content_str;
use serde_json::json;
use std::path::PathBuf;

#[test]
fn subagent_forwards_live_tokens_to_parent_with_base() {
    let mut parent = CapturingUi::default();
    let mut sub = SubagentUi {
        base: 100,
        parent: Some(&mut parent),
        ..Default::default()
    };
    sub.turn_tokens(20);
    sub.turn_tokens(55);
    drop(sub);
    assert_eq!(parent.turn_token_reports, vec![120, 155]);
}

/// `turn_start` bumps the step + reports thinking (empty tool); `tool_start`
/// reports the tool — both tagged with the specialist name and 1-based step.
#[test]
fn subagent_forwards_step_activity_to_parent() {
    let mut parent = CapturingUi::default();
    let mut sub = SubagentUi {
        agent_name: "code-reviewer".to_string(),
        parent: Some(&mut parent),
        ..Default::default()
    };
    sub.turn_start();
    sub.tool_start("grep", &json!({"pattern": "fn"}));
    sub.turn_start();
    sub.tool_start("read_file", &json!({"path": "src/lib.rs"}));
    drop(sub);
    assert_eq!(
        parent.sub_activity,
        vec![
            ("code-reviewer".to_string(), String::new(), 1),
            ("code-reviewer".to_string(), "grep".to_string(), 1),
            ("code-reviewer".to_string(), String::new(), 2),
            ("code-reviewer".to_string(), "read_file".to_string(), 2),
        ]
    );
}

/// A sub-agent forwards permission asks to the parent (else the catastrophic floor is
/// skipped for delegated work); a denying/headless parent blocks it, detached denies.
#[tokio::test]
async fn subagent_forwards_permission_to_parent_and_fails_closed() {
    let mut parent = CapturingUi {
        deny: true,
        ..Default::default()
    };
    let mut sub = SubagentUi {
        parent: Some(&mut parent),
        ..Default::default()
    };
    let decision = sub.ask_permission("run_bash", Some("rm -rf /")).await;
    assert!(matches!(decision, Decision::Deny));
    drop(sub);
    assert_eq!(parent.ask_tools, vec!["run_bash"]);

    // Detached (no parent) fails closed.
    let mut orphan = SubagentUi::default();
    assert!(matches!(
        orphan.ask_permission("run_bash", Some("rm -rf /")).await,
        Decision::Deny
    ));
}

/// A `subagent` call spawns a fresh sub-engine; its text result feeds back as the parent's tool result and the parent converges.
#[tokio::test]
async fn engine_runs_subagent_and_returns_result() {
    let dir = tmp();
    let call = tool_call_sse("subagent", json!({"task": "investigate the bug"}));
    let sub_text =
        "data: {\"choices\":[{\"delta\":{\"content\":\"subresult\"}}]}\n\ndata: [DONE]\n\n"
            .to_string();
    let port = spawn_sse_sequence(vec![call, sub_text, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate it".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.tools, vec!["subagent"]);
    assert_eq!(ui.text, "done");
    // The sub-agent's answer came back as the parent's tool result.
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m).contains("subresult")),
        "sub-agent result missing from parent history"
    );
}

/// The sub-agent UI recovers an answer emitted in the same step as the final tool call, instead of losing it.
#[test]
fn subagent_ui_recovers_answer_before_final_tool() {
    let mut ui = SubagentUi::default();
    // Step 1: the model gives its answer AND calls a tool in the same step.
    ui.turn_start();
    ui.assistant_text("The answer is 42.");
    ui.tool_start("run_bash", &json!({}));
    ui.tool_result("run_bash", &Ok("ok".to_string()));
    // Step 2: converges with no text of its own.
    ui.turn_start();
    ui.footer(None, 2, 0, 0, 0);
    assert_eq!(ui.answer(), "The answer is 42.");

    // Normal case: the converging step carries the answer.
    let mut ui2 = SubagentUi::default();
    ui2.turn_start();
    ui2.assistant_text("plain answer");
    ui2.footer(None, 1, 0, 0, 0);
    assert_eq!(ui2.answer(), "plain answer");
}

/// `isolation: "worktree"`: the sub-agent's writes land in a disposable worktree,
/// the parent tree stays untouched, and the result says where the changes are.
#[tokio::test]
async fn subagent_worktree_isolation_keeps_parent_tree_clean() {
    let dir = tmp();
    let git = |args: &[&str]| {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(&dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?} failed");
    };
    git(&["init", "-q"]);
    std::fs::write(dir.join("a.txt"), "one").unwrap();
    git(&["add", "-A"]);
    git(&["commit", "-qm", "init"]);

    let parent_call = tool_call_sse(
        "subagent",
        json!({"task": "add b.txt", "isolation": "worktree"}),
    );
    let sub_write = tool_call_sse("write_file", json!({"path": "b.txt", "content": "two"}));
    let port = spawn_sse_sequence(vec![
        parent_call,
        sub_write,
        FINAL_TEXT_SSE.to_string(), // sub declares done
        FINAL_TEXT_SSE.to_string(), // parent declares done
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate the edit".into()),
        &mut ui,
    )
    .await;

    assert!(
        !dir.join("b.txt").exists(),
        "parent tree must stay untouched"
    );
    let report = engine
        .messages
        .iter()
        .filter_map(|m| m.get("content").and_then(Value::as_str))
        .find(|s| s.contains("[worktree isolation]"))
        .expect("result must carry the worktree note")
        .to_string();
    assert!(report.contains("1 path(s) changed"), "{report}");
    // The reported worktree really holds the edit; clean it up.
    let wt = report
        .split("worktree at ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .map(PathBuf::from)
        .expect("note names the worktree path");
    assert!(wt.join("b.txt").is_file(), "edit landed in the worktree");
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(&dir)
        .args(["worktree", "remove", "--force"])
        .arg(&wt)
        .output();
}

/// Isolation requested outside a git repo falls back to the shared workspace
/// with a note, rather than failing the delegation.
#[tokio::test]
async fn subagent_worktree_isolation_falls_back_outside_git() {
    let dir = tmp();
    let parent_call = tool_call_sse(
        "subagent",
        json!({"task": "look around", "isolation": "worktree"}),
    );
    let port = spawn_sse_sequence(vec![
        parent_call,
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
        Some("delegate".into()),
        &mut ui,
    )
    .await;
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("[worktree isolation] unavailable"))
        }),
        "fallback must be reported"
    );
}

/// A sub-agent's token usage folds into the parent turn's total (the sub's LLM calls aren't parent steps).
#[tokio::test]
async fn subagent_tokens_fold_into_parent_total() {
    let dir = tmp();
    let call = tool_call_sse("subagent", json!({"task": "investigate"}));
    // The sub-agent's turn reports 100 tokens of usage.
    let sub_text = "data: {\"choices\":[{\"delta\":{\"content\":\"subresult\"}}],\"usage\":{\"total_tokens\":100}}\n\ndata: [DONE]\n\n".to_string();
    let port = spawn_sse_sequence(vec![call, sub_text, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate it".into()),
        &mut ui,
    )
    .await;

    // The parent's own steps report no usage, so the 100 came from the sub-agent.
    assert!(
        ui.footer_tokens >= 100,
        "sub-agent tokens not folded into the parent total: {}",
        ui.footer_tokens
    );
}

/// A long sub-agent report is written to the artifacts dir and the parent's tool
/// result gains a pointer line, so the work survives compaction.
#[tokio::test]
async fn subagent_report_saved_and_pointered() {
    let dir = tmp();
    let artifacts = tmp();
    let call = tool_call_sse("subagent", json!({"task": "summarize the engine module"}));
    let long = "SUBRESULT ".repeat(200); // > TOOL_RESULT_CLEAR_MIN
    let sub_final = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content": long}}]})
    );
    let port = spawn_sse_sequence(vec![call, sub_final, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-07-08",
        &[],
        &[],
        0,
        0,
    );
    engine.set_artifacts_dir(artifacts.clone());
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate it".into()),
        &mut ui,
    )
    .await;

    // Exactly one report file, named sub-001-*.md, carrying the sub answer.
    let files: Vec<_> = std::fs::read_dir(&artifacts)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert_eq!(files.len(), 1, "one report expected");
    let name = files[0].file_name().to_string_lossy().into_owned();
    assert!(name.starts_with("sub-001-"), "unexpected name: {name}");
    let body = std::fs::read_to_string(files[0].path()).unwrap();
    assert!(body.contains("SUBRESULT"), "report missing the sub answer");
    assert!(body.contains("# Sub-agent report"), "report missing header");
    // The parent's tool result carries the pointer.
    let has_pointer = engine.messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains(ARTIFACT_POINTER_PREFIX))
    });
    assert!(has_pointer, "pointer missing from the tool result");
}

#[test]
fn parse_artifact_seq_reads_the_number() {
    assert_eq!(parse_artifact_seq("sub-001-foo.md"), Some(1));
    assert_eq!(parse_artifact_seq("sub-042-a-b-c.md"), Some(42));
    assert_eq!(parse_artifact_seq("notes.md"), None);
    assert_eq!(parse_artifact_seq("sub-x.md"), None);
}

/// A rebuilt engine on the SAME artifacts dir resumes numbering instead of overwriting.
#[tokio::test]
async fn subagent_report_numbering_survives_rebuild() {
    let dir = tmp();
    let artifacts = tmp();
    let long = "SUBRESULT ".repeat(200);
    let sub_final = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content": long}}]})
    );
    let run = |port: u16, artifacts: PathBuf, dir: PathBuf| async move {
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-07-08",
            &[],
            &[],
            0,
            0,
        );
        engine.set_artifacts_dir(artifacts);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("delegate it".into()),
            &mut ui,
        )
        .await;
    };
    let call = tool_call_sse("subagent", json!({"task": "summarize the engine module"}));
    // Engine A.
    let port = spawn_sse_sequence(vec![
        call.clone(),
        sub_final.clone(),
        FINAL_TEXT_SSE.to_string(),
    ]);
    run(port, artifacts.clone(), dir.clone()).await;
    // Engine B — fresh engine, SAME artifacts dir, same task slug.
    let port = spawn_sse_sequence(vec![call, sub_final, FINAL_TEXT_SSE.to_string()]);
    run(port, artifacts.clone(), dir.clone()).await;

    let files: Vec<String> = std::fs::read_dir(&artifacts)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        files.len(),
        2,
        "two distinct reports, no overwrite: {files:?}"
    );
    assert!(files.iter().any(|f| f.starts_with("sub-001-")));
    assert!(files.iter().any(|f| f.starts_with("sub-002-")));
}

/// A short sub-agent answer isn't worth a file — nothing is written and no pointer added.
#[tokio::test]
async fn subagent_short_answer_writes_nothing() {
    let dir = tmp();
    let artifacts = tmp();
    let call = tool_call_sse("subagent", json!({"task": "quick check"}));
    let sub_final = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content": "short answer"}}]})
    );
    let port = spawn_sse_sequence(vec![call, sub_final, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-07-08",
        &[],
        &[],
        0,
        0,
    );
    engine.set_artifacts_dir(artifacts.clone());
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate it".into()),
        &mut ui,
    )
    .await;

    assert!(
        std::fs::read_dir(&artifacts).unwrap().next().is_none(),
        "no report should be written for a short answer"
    );
    let has_pointer = engine.messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains(ARTIFACT_POINTER_PREFIX))
    });
    assert!(!has_pointer, "no pointer expected for a short answer");
}

/// With no artifacts dir (headless/tests), the sub-agent result keeps today's exact
/// format: the `[sub-agent: N step(s)]` tail, no pointer.
#[tokio::test]
async fn subagent_no_artifacts_dir_unchanged() {
    let dir = tmp();
    let call = tool_call_sse("subagent", json!({"task": "investigate"}));
    let long = "SUBRESULT ".repeat(200);
    let sub_final = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content": long}}]})
    );
    let port = spawn_sse_sequence(vec![call, sub_final, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate it".into()),
        &mut ui,
    )
    .await;

    let has_pointer = engine.messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains(ARTIFACT_POINTER_PREFIX))
    });
    assert!(!has_pointer, "no artifacts dir → no pointer");
    let has_tail = engine.messages.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_str())
            .is_some_and(|s| s.contains("[sub-agent:"))
    });
    assert!(has_tail, "the classic sub-agent step tail should remain");
}

/// When a sub-agent produces no answer, the failure reason is surfaced instead of a vague "no answer".
#[test]
fn subagent_ui_surfaces_failure_notice_when_no_answer() {
    let mut ui = SubagentUi::default();
    ui.turn_start();
    ui.notify("reached the step limit (20)");
    // No assistant text emitted → no answer.
    let msg = ui.result_message();
    assert!(msg.contains("no answer"), "got: {msg}");
    assert!(msg.contains("step limit"), "failure reason missing: {msg}");

    // With an answer, the notice is ignored.
    let mut ui2 = SubagentUi::default();
    ui2.turn_start();
    ui2.assistant_text("the result is 42");
    ui2.notify("compacting context…");
    let msg2 = ui2.result_message();
    assert!(msg2.contains("the result is 42"));
    assert!(
        !msg2.contains("compacting"),
        "notice leaked into a good answer"
    );
}

#[test]
fn subagent_tool_offered_and_droppable_for_recursion_guard() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let has_subagent = |e: &AgentEngine| {
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"].as_str() == Some("subagent"))
    };
    assert!(has_subagent(&engine), "top-level engine offers subagent");
    engine.drop_subagent_tool();
    assert!(
        !has_subagent(&engine),
        "sub-engine must not offer subagent (no recursion)"
    );
}

#[tokio::test]
async fn subagent_ui_sink_forwards_activity_and_denies_visibly() {
    #[derive(Default)]
    struct Recording(std::sync::Mutex<Vec<String>>);
    impl SubagentSink for Recording {
        fn begin(&self, labels: &[String]) {
            self.0
                .lock()
                .unwrap()
                .push(format!("begin:{}", labels.len()));
        }
        fn activity(&self, slot: usize, agent: &str, tool: &str, _args: &Value, step: usize) {
            self.0
                .lock()
                .unwrap()
                .push(format!("activity:{slot}:{agent}:{tool}:{step}"));
        }
        fn denied(&self, slot: usize, tool: &str) {
            self.0.lock().unwrap().push(format!("denied:{slot}:{tool}"));
        }
        fn done(&self, slot: usize, ok: bool, steps: usize, _tokens: u64) {
            self.0
                .lock()
                .unwrap()
                .push(format!("done:{slot}:{ok}:{steps}"));
        }
        fn finish(&self) {
            self.0.lock().unwrap().push("finish".to_string());
        }
    }
    let sink = std::sync::Arc::new(Recording::default());
    let mut ui = SubagentUi {
        sink: Some((sink.clone(), 3)),
        agent_name: "audit auth".to_string(),
        ..Default::default()
    };
    ui.turn_start();
    ui.tool_start("read_file", &json!({"path": "a.rs"}));
    let decision = ui.ask_permission("run_bash", None).await;
    assert!(matches!(decision, Decision::Deny));
    let events = sink.0.lock().unwrap().clone();
    assert_eq!(
        events,
        vec![
            "activity:3:audit auth::1",
            "activity:3:audit auth:read_file:1",
            "denied:3:run_bash",
        ]
    );
}
