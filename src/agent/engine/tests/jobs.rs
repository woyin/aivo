use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn set_jobs_advertises_check_job() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let has = |e: &AgentEngine| {
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"] == "check_job")
    };
    assert!(!has(&e), "no check_job before a table is wired");
    e.set_jobs(jobs::JobTable::new(None));
    assert!(has(&e), "check_job advertised after set_jobs");
    // Idempotent: a second call must not double-advertise.
    e.set_jobs(jobs::JobTable::new(None));
    let count = e
        .tools_openai
        .iter()
        .filter(|t| t["function"]["name"] == "check_job")
        .count();
    assert_eq!(count, 1, "check_job advertised exactly once");
}

/// A `run_bash {background:true}` routes to the job table and returns a job id.
#[cfg(unix)]
#[tokio::test]
async fn background_run_bash_returns_job_id() {
    let dir = tmp();
    let call = tool_call_sse(
        "run_bash",
        json!({"command": "echo hi", "background": true}),
    );
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(jobs::JobTable::new(Some(dir.join("jobs"))));
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("start it".into()),
        &mut ui,
    )
    .await;
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("started background job j1"))
        }),
        "expected a job-started tool result"
    );
}

/// A background job that finished is surfaced at the next step boundary, folded
/// into the last tool result (bridge-safe), so the model needn't busy-poll.
#[cfg(unix)]
#[tokio::test]
async fn finished_background_job_notice_is_injected_at_step_boundary() {
    let dir = tmp();
    let jobs = jobs::JobTable::new(Some(dir.join("jobs")));
    jobs.spawn("echo bye", &dir).unwrap();
    // Already reaped-finished before the turn: the first boundary drains it.
    for _ in 0..100 {
        if jobs.running_count() == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let call = tool_call_sse("list_dir", json!({"path": "."}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(jobs);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("look around".into()),
        &mut ui,
    )
    .await;
    let folded = engine.messages.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("tool")
            && m.get("content")
                .and_then(Value::as_str)
                .is_some_and(|s| s.contains("<background_jobs>") && s.contains("job j1"))
    });
    assert!(
        folded,
        "expected the finish notice folded into a tool result: {:?}",
        engine.messages
    );
}

// --- user lifecycle hooks ---

/// Without a job table, a background request errs with actionable guidance (never panics).
#[tokio::test]
async fn background_without_job_table_errs_with_guidance() {
    let dir = tmp();
    let call = tool_call_sse(
        "run_bash",
        json!({"command": "sleep 1", "background": true}),
    );
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // No set_jobs.
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("start it".into()),
        &mut ui,
    )
    .await;
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("background jobs aren't available"))
        }),
        "expected a guidance error"
    );
}

/// `check_job {kill:true}` terminates a running job via the engine dispatch.
#[cfg(unix)]
#[tokio::test]
async fn check_job_kill_terminates() {
    let dir = tmp();
    let table = jobs::JobTable::new(Some(dir.join("jobs")));
    let bg = tool_call_sse(
        "run_bash",
        json!({"command": "sleep 30", "background": true}),
    );
    let kill = tool_call_sse("check_job", json!({"id": "j1", "kill": true}));
    let port = spawn_sse_sequence(vec![bg, kill, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(table.clone());
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("start then stop it".into()),
        &mut ui,
    )
    .await;
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("killed job j1"))
        }),
        "expected a kill confirmation"
    );
    assert_eq!(table.running_count(), 0, "job should be gone");
}

/// `check_job` needs no permission and runs even in plan mode (job control on a
/// process the agent started, never a file write).
#[cfg(unix)]
#[tokio::test]
async fn check_job_survives_plan_mode_and_runs_unprompted() {
    let dir = tmp();
    let table = jobs::JobTable::new(Some(dir.join("jobs")));
    table.spawn("sleep 30", &dir).unwrap();
    let call = tool_call_sse("check_job", json!({"id": "j1"}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(table.clone());
    engine.set_plan_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("check it".into()),
        &mut ui,
    )
    .await;
    assert_eq!(
        ui.asks, 0,
        "check_job needs no permission, even in plan mode"
    );
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("job j1: running"))
        }),
        "check_job should have run and reported the job"
    );
    let _ = table.kill_all().await;
}

/// Deliberate: it only signals a process the agent itself started and touches
/// no files (see `is_read_only`), so it stays prompt-free even in plan mode.
#[cfg(unix)]
#[tokio::test]
async fn check_job_kill_unprompted_in_plan_mode() {
    let dir = tmp();
    let table = jobs::JobTable::new(Some(dir.join("jobs")));
    table.spawn("sleep 30", &dir).unwrap();
    let call = tool_call_sse("check_job", json!({"id": "j1", "kill": true}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(table.clone());
    engine.set_plan_mode(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("stop it".into()),
        &mut ui,
    )
    .await;
    assert_eq!(ui.asks, 0, "kill of an agent-started job is prompt-free");
    assert!(
        engine.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains("killed job j1"))
        }),
        "the kill should have run and reported"
    );
    assert_eq!(table.running_count(), 0, "the job is really gone");
}

/// A background `run_bash` in plan mode still hits the plan-bash confirmation.
#[cfg(unix)]
#[tokio::test]
async fn background_bash_still_gated_in_plan_mode() {
    let dir = tmp();
    let table = jobs::JobTable::new(Some(dir.join("jobs")));
    let call = tool_call_sse(
        "run_bash",
        json!({"command": "sleep 30", "background": true}),
    );
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(table.clone());
    engine.set_plan_mode(true);
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("start a server".into()),
        &mut ui,
    )
    .await;
    assert!(ui.asks >= 1, "plan-mode background bash must prompt");
    assert_eq!(table.running_count(), 0, "denied → nothing spawned");
}

/// A sub-agent inherits the parent's job table (it can poll/kill the parent's jobs).
#[cfg(unix)]
#[tokio::test]
async fn subagent_engine_inherits_job_table() {
    let dir = tmp();
    let table = jobs::JobTable::new(Some(dir.join("jobs")));
    table.spawn("sleep 30", &dir).unwrap();
    let parent_call = tool_call_sse("subagent", json!({"task": "stop job j1"}));
    let sub_kill = tool_call_sse("check_job", json!({"id": "j1", "kill": true}));
    let port = spawn_sse_sequence(vec![
        parent_call,
        sub_kill,
        FINAL_TEXT_SSE.to_string(), // sub converges
        FINAL_TEXT_SSE.to_string(), // parent converges
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_jobs(table.clone());
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("delegate the stop".into()),
        &mut ui,
    )
    .await;
    // The sub killed j1 → it must have had the shared table.
    assert_eq!(
        table.running_count(),
        0,
        "sub-agent should have killed the parent's job"
    );
}
