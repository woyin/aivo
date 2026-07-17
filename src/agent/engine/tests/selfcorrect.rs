use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn set_self_correct_toggles_the_flag() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    assert!(!engine.self_correct, "off by default");
    engine.set_self_correct(true);
    assert!(engine.self_correct, "enabled");
    engine.set_self_correct(false);
    assert!(
        !engine.self_correct,
        "disabled again (goal mode toggles off)"
    );
}

/// With self-correct on, a declared-done turn can't converge over a red suite: the
/// failure is fed back (VERIFY_FAILED_PREFIX) until the model makes it pass.
/// Unix-only: relies on `sh run_tests.sh` (absent on the Windows runner → inconclusive).
#[cfg(unix)]
#[tokio::test]
async fn selfcorrect_blocks_done_until_green() {
    let dir = tmp();
    // run_tests.sh fails until the marker file `passing` exists.
    std::fs::write(
        dir.join("run_tests.sh"),
        "[ -f passing ] && exit 0 || exit 1\n",
    )
    .unwrap();

    // 1) text "done" → validator fails → nudge; 2) write the marker; 3) "done" → passes.
    let write = tool_call_sse("write_file", json!({"path": "passing", "content": "ok"}));
    let port = spawn_sse_sequence(vec![
        FINAL_TEXT_SSE.to_string(),
        write,
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_self_correct(true);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("make the tests pass".into()),
        &mut ui,
    )
    .await;

    // The failure was fed back exactly once, then the suite went green.
    let vf = engine
        .messages
        .iter()
        .filter(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .is_some_and(|s| s.contains(VERIFY_FAILED_PREFIX))
        })
        .count();
    assert_eq!(vf, 1, "verify failure fed back exactly once");
    assert!(dir.join("passing").exists(), "the marker write ran");
    assert!(
        ui.notices.iter().any(|n| n.contains("run_tests.sh failed")),
        "expected a failing-suite notice: {:?}",
        ui.notices
    );
    assert!(
        ui.notices
            .iter()
            .any(|n| n.contains("verified: run_tests.sh passed")),
        "expected a passing-suite notice: {:?}",
        ui.notices
    );
}

/// The first done-turn always verifies (green baseline); a later clean turn skips it.
#[cfg(unix)]
#[tokio::test]
async fn selfcorrect_skips_verify_when_clean_since_green() {
    let dir = tmp();
    // Passing suite that logs each invocation.
    std::fs::write(dir.join("run_tests.sh"), "echo run >> runs.log; exit 0\n").unwrap();

    let write = tool_call_sse("write_file", json!({"path": "f", "content": "x"}));
    // Turn 1: edit + done → verify runs (green baseline).
    let port = spawn_sse_sequence(vec![
        write,
        FINAL_TEXT_SSE.to_string(),
        // Turn 2: read + done → clean since green → verify skipped.
        tool_call_sse("read_file", json!({"path": "f"})),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_self_correct(true);
    let mut ui = CapturingUi::default();
    let ctx = turn_ctx(&client, &base, &dir);
    run_session(&mut engine, &ctx, Some("edit then check".into()), &mut ui).await;
    run_session(&mut engine, &ctx, Some("just look around".into()), &mut ui).await;

    let runs = std::fs::read_to_string(dir.join("runs.log")).unwrap_or_default();
    assert_eq!(runs.lines().count(), 1, "suite ran once, not per turn");
}

/// Under a verified baseline (the default-on headless arrangement), an
/// investigate-only run converges without paying for a suite run; a mutating run
/// still verifies at declared-done.
#[cfg(unix)]
#[tokio::test]
async fn selfcorrect_verified_baseline_skips_investigate_only_runs() {
    let dir = tmp();
    std::fs::write(dir.join("run_tests.sh"), "echo run >> runs.log; exit 0\n").unwrap();

    let port = spawn_sse_sequence(vec![
        // Turn 1: read + done → clean baseline → no suite run.
        tool_call_sse("read_file", json!({"path": "run_tests.sh"})),
        FINAL_TEXT_SSE.to_string(),
        // Turn 2: write + done → dirty → suite runs.
        tool_call_sse("write_file", json!({"path": "f", "content": "x"})),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_self_correct(true);
    engine.set_verified_baseline();
    let mut ui = CapturingUi::default();
    let ctx = turn_ctx(&client, &base, &dir);
    run_session(&mut engine, &ctx, Some("just look around".into()), &mut ui).await;
    assert!(
        !dir.join("runs.log").exists(),
        "investigate-only run must not trigger the suite"
    );
    run_session(&mut engine, &ctx, Some("now edit".into()), &mut ui).await;
    let runs = std::fs::read_to_string(dir.join("runs.log")).unwrap_or_default();
    assert_eq!(runs.lines().count(), 1, "mutation re-arms verification");
}

// --- background jobs (Phase 4) ---
