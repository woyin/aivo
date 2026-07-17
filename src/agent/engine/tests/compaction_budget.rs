use super::super::*;
use super::helpers::*;
use crate::agent::compaction::{COMPACT_RESERVE, TOOL_RESULT_CLEARED};
use crate::agent::request::content_str;
use crate::agent::tokens::MAX_CALIBRATION;
use serde_json::json;

/// `set_context_window` fills an unknown (0) window (late catalog warm) but never overrides a known one.
#[test]
fn set_context_window_fills_only_a_missing_window() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 0; // force unknown, ignoring any env override
    engine.set_context_window(200_000);
    assert_eq!(engine.context_window, 200_000, "missing window should fill");
    engine.set_context_window(100_000);
    assert_eq!(
        engine.context_window, 200_000,
        "a known window must not change"
    );
    engine.set_context_window(0);
    assert_eq!(engine.context_window, 200_000, "a 0 update is a no-op");
}

/// An unknown window (0) compacts at `DEFAULT_CONTEXT_WINDOW`; a known window
/// takes precedence.
#[test]
fn compaction_window_falls_back_to_default_when_unknown() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 0; // unknown, ignoring any env override
    assert_eq!(
        engine.compaction_window(),
        DEFAULT_CONTEXT_WINDOW,
        "unknown window should compact at the default backstop, not be skipped"
    );
    engine.set_context_window(500_000);
    assert_eq!(
        engine.compaction_window(),
        500_000,
        "a known window takes precedence over the default"
    );
}

#[test]
fn token_calibration_deflates_compaction_budget() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 262_144;
    let raw = engine.compaction_window() - COMPACT_RESERVE;
    assert_eq!(
        engine.compaction_budget_estimate(),
        raw,
        "calibration 1.0 leaves the old window − reserve budget unchanged"
    );
    engine.token_calibration = 1.2;
    let deflated = engine.compaction_budget_estimate();
    assert_eq!(deflated, ((raw as f64) / 1.2).floor() as usize);
    assert!(
        deflated < raw,
        "calibration > 1 shrinks the estimate-space budget so denser-than-chars/4 \
         content still fits the real window"
    );
}

#[test]
fn update_calibration_rises_on_undershoot_then_eases_and_clamps() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.update_calibration(100, 400);
    assert_eq!(engine.token_calibration, 1.0, "tiny request is ignored");
    engine.update_calibration(100_000, 110_000);
    assert!(
        (engine.token_calibration - 1.1).abs() < 1e-9,
        "undershoot raises calibration to the measured ratio, got {}",
        engine.token_calibration
    );
    engine.update_calibration(100_000, 100_000);
    assert!(
        engine.token_calibration > 1.0 && engine.token_calibration < 1.1,
        "calibration eases down slowly, got {}",
        engine.token_calibration
    );
    engine.update_calibration(100_000, 100_000_000);
    assert_eq!(
        engine.token_calibration, MAX_CALIBRATION,
        "ratio clamped to the ceiling"
    );
}

/// A tool call whose bulk is in `arguments` (empty `content`) in the irreducible recent turn must be shrunk — content-only truncation would leave it over.
#[test]
fn enforce_budget_shrinks_oversized_tool_call_arguments() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let huge_args = format!("{{\"content\":\"{}\"}}", "x".repeat(40_000));
    engine.messages = vec![
        json!({"role":"system","content":"sys"}),
        json!({"role":"user","content":"write the file"}),
        json!({"role":"assistant","content":"","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"write_file","arguments": huge_args}}]}),
    ];
    assert!(estimate_tokens(&engine.messages) > 300);
    engine.enforce_budget(300);
    assert!(
        estimate_tokens(&engine.messages) <= 300,
        "must fit budget by shrinking tool-call arguments, got {}",
        estimate_tokens(&engine.messages)
    );
    let tc = &engine.messages[2]["tool_calls"][0];
    assert_eq!(tc["id"], "c1", "tool_call_id preserved");
    assert_eq!(tc["function"]["name"], "write_file", "call name preserved");
    assert!(
        tc["function"]["arguments"].as_str().unwrap().len() < huge_args.len(),
        "arguments were truncated"
    );
    assert_eq!(role(&engine.messages[0]), "system", "system prompt kept");
}

/// A transcript whose estimate clears the raw budget but the provider rejects: calibrating from the rejection + force-fitting brings the real size under the window.
#[test]
fn overflow_recovery_calibrates_from_rejection_and_fits() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 262_144;
    let raw_budget = engine.compaction_window() - COMPACT_RESERVE;
    let pad = "x".repeat(4 * (raw_budget - 20_000));
    engine.messages = vec![
        json!({"role":"system","content":"sys"}),
        json!({"role":"user","content":"q1"}),
        json!({"role":"assistant","content":"","tool_calls":[
            {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"a","content": pad}),
        json!({"role":"user","content":"now"}),
    ];
    let est = estimate_tokens(&engine.messages);
    assert!(
        est <= raw_budget,
        "pre-fix budget check would pass (no compaction): est={est} raw={raw_budget}"
    );

    let err = "token count of 290000 exceeds the maximum allowed input length of 262112 tokens";
    assert!(is_context_overflow_error(err));
    engine.recalibrate_from_overflow(err);
    assert!(
        engine.token_calibration > 1.0,
        "the rejection raised the calibration, got {}",
        engine.token_calibration
    );
    engine.force_fit_budget();

    let budget = engine.compaction_budget_estimate();
    assert!(
        estimate_tokens(&engine.messages) <= budget,
        "recovery brought the transcript under the calibrated budget"
    );
    let projected = (estimate_tokens(&engine.messages) as f64 * engine.token_calibration) as usize;
    assert!(
        projected <= engine.compaction_window(),
        "the calibrated real size now fits the window: projected={projected}"
    );
    assert_eq!(
        role(&engine.messages[0]),
        "system",
        "the system prompt is never dropped"
    );
}

/// Overflow recovery on a resumed single long turn keeps a marker for dropped work instead of vanishing to `[system, latest-user]`.
#[test]
fn force_fit_recovery_keeps_prior_context_on_single_long_turn() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
    let big = "reasoning ".repeat(4_000);
    let mut messages = vec![
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "original task: fix the shell tool"}),
    ];
    for i in 0..4 {
        messages.push(json!({"role": "assistant", "content": format!("step {i}: {big}")}));
    }
    messages.push(json!({"role": "user", "content": "continue"}));
    engine.messages = messages;

    engine.force_fit_budget();

    let budget = engine.compaction_budget_estimate();
    assert!(
        estimate_tokens(&engine.messages) <= budget,
        "recovery must fit the budget"
    );
    let last = engine.messages.last().unwrap();
    assert_eq!(role(last), "user");
    assert!(content_str(last).contains("continue"), "latest turn kept");
    assert!(
        content_str(last).contains("[Summary of earlier conversation]"),
        "dropped prior turn must leave a marker, not vanish: {}",
        content_str(last)
    );
}

/// A huge RECENT tool result fills the keep window; an OLDER one before the cut
/// gets stubbed.
#[test]
fn compact_now_local_clears_stale_tool_output() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let recent = "r".repeat(200_000); // ~25k tokens — fills the 20k keep window
    let stale = "s".repeat(8_000); // > TOOL_RESULT_CLEAR_MIN, older than the cut
    engine.messages = vec![
        json!({"role":"system","content":"sys"}),
        json!({"role":"user","content":"q1"}),
        json!({"role":"assistant","content":"","tool_calls":[
            {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"a","content": stale}),
        json!({"role":"user","content":"q2"}),
        json!({"role":"assistant","content":"","tool_calls":[
            {"id":"b","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
        json!({"role":"tool","tool_call_id":"b","content": recent.clone()}),
    ];
    assert!(
        engine.has_compactable_history(),
        "an old bulky tool result is foldable"
    );
    let (before, after) = engine.compact_now_local();
    assert!(
        after < before,
        "clearing stale output frees context: {before} → {after}"
    );
    assert_eq!(
        engine.messages[3]["content"].as_str(),
        Some(TOOL_RESULT_CLEARED),
        "old stale tool output cleared"
    );
    assert_eq!(
        engine.messages[6]["content"].as_str(),
        Some(recent.as_str()),
        "recent tool output kept intact"
    );
}

#[test]
fn has_compactable_history_false_for_short_conversation() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.messages = vec![
        json!({"role":"system","content":"sys"}),
        json!({"role":"user","content":"hi"}),
        json!({"role":"assistant","content":"hello"}),
    ];
    assert!(
        !engine.has_compactable_history(),
        "a tiny recent-only conversation has nothing to fold"
    );
}

/// Tiny window + zero keep-recent: with only stale OLD tool output overflowing,
/// `maybe_compact` takes the no-model cheap path (clears them, returns 0).
#[tokio::test]
async fn forced_tiny_window_compacts_at_boundary_without_a_model_call() {
    // SAFETY: scoped mutation of an env var no other test reads.
    unsafe { std::env::set_var("AIVO_AGENT_KEEP_RECENT", "0") };

    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
    let huge = "x".repeat(200_000);
    engine.messages = vec![
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "q1"}),
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}),
        json!({"role": "tool", "tool_call_id": "a", "content": huge.clone()}),
        json!({"role": "user", "content": "q2"}),
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}),
        json!({"role": "tool", "tool_call_id": "b", "content": huge}),
        json!({"role": "user", "content": "now"}),
    ];
    let budget = engine.compaction_window() - COMPACT_RESERVE;
    assert!(
        estimate_tokens(&engine.messages) > budget,
        "transcript must start over budget so the boundary is actually crossed"
    );

    let client = reqwest::Client::new();
    let cwd = std::path::Path::new(".");
    let ctx = turn_ctx(&client, "", cwd);
    let mut ui = CapturingUi::default();
    let tokens = engine.maybe_compact(&ctx, &mut ui).await;

    unsafe { std::env::remove_var("AIVO_AGENT_KEEP_RECENT") };

    assert_eq!(tokens, 0, "cheap path must not call the model");
    assert!(
        estimate_tokens(&engine.messages) <= budget,
        "compaction must bring the transcript under budget"
    );
    let cleared = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .filter(|m| m.get("content").and_then(|c| c.as_str()) == Some(TOOL_RESULT_CLEARED))
        .count();
    assert_eq!(
        cleared, 2,
        "stale OLD tool output cleared without a model call"
    );
    assert!(
        ui.notices
            .iter()
            .any(|n| n.contains("cleared older tool output")),
        "the user is told the cheap path ran"
    );
}

/// A resumed single long turn over budget keeps prior context as a folded summary instead of dropping to `[system, latest-user]`.
#[tokio::test]
async fn resume_single_long_turn_keeps_prior_context_on_compaction() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
    // Assistant-only run: no tool results for the cheap clear path to reclaim.
    let big = "reasoning ".repeat(4_000);
    let mut messages = vec![
        json!({"role": "system", "content": "sys"}),
        json!({"role": "user", "content": "original task: fix the shell tool"}),
    ];
    for i in 0..4 {
        messages.push(json!({"role": "assistant", "content": format!("step {i}: {big}")}));
    }
    messages.push(json!({"role": "user", "content": "continue"}));
    engine.messages = messages;
    let budget = engine.compaction_window() - COMPACT_RESERVE;
    assert!(
        estimate_tokens(&engine.messages) > budget,
        "transcript must start over budget"
    );

    let client = reqwest::Client::new();
    let cwd = std::path::Path::new(".");
    let ctx = turn_ctx(&client, "", cwd); // empty base → summary fails → mechanical fold
    let mut ui = CapturingUi::default();
    engine.maybe_compact(&ctx, &mut ui).await;

    assert!(
        estimate_tokens(&engine.messages) <= budget,
        "compaction must fit the budget"
    );
    let last = engine.messages.last().unwrap();
    assert_eq!(role(last), "user");
    assert!(content_str(last).contains("continue"), "latest turn kept");
    assert!(
        content_str(last).contains("[Summary of earlier conversation]"),
        "the dropped prior turn must be summarized in, not silently discarded: {}",
        content_str(last)
    );
}

/// The budget backstop drops oldest turns at user boundaries (never the system prompt) until it fits — guards against a non-retryable post-compaction 413.
#[test]
fn enforce_budget_drops_oldest_turns_at_user_boundaries() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let pad = "x".repeat(400);
    engine.messages = vec![
        json!({"role":"system","content":"SYS"}),
        json!({"role":"user","content":format!("u1 {pad}")}),
        json!({"role":"assistant","content":format!("a1 {pad}")}),
        json!({"role":"user","content":format!("u2 {pad}")}),
        json!({"role":"assistant","content":format!("a2 {pad}")}),
        json!({"role":"user","content":"u3 keep me"}),
        json!({"role":"assistant","content":"a3 keep me"}),
    ];
    engine.enforce_budget(200);

    assert!(
        estimate_tokens(&engine.messages) <= 200,
        "must fit budget, got {}",
        estimate_tokens(&engine.messages)
    );
    assert_eq!(role(&engine.messages[0]), "system"); // never dropped
    assert!(
        engine
            .messages
            .iter()
            .any(|m| content_str(m).contains("u3 keep me")),
        "latest turn must survive"
    );
    assert!(
        !engine
            .messages
            .iter()
            .any(|m| content_str(m).contains("u1")),
        "an old turn must be dropped"
    );
}

/// When even [system, last user turn] overflows (one huge pasted turn), the backstop shortens the content instead of looping forever.
#[test]
fn enforce_budget_truncates_a_single_oversized_turn() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.messages = vec![
        json!({"role":"system","content":"SYS"}),
        json!({"role":"user","content":"y".repeat(8000)}),
    ];
    engine.enforce_budget(300);

    assert!(
        estimate_tokens(&engine.messages) <= 300,
        "must fit budget, got {}",
        estimate_tokens(&engine.messages)
    );
    assert_eq!(role(&engine.messages[0]), "system");
    assert_eq!(
        engine.messages.len(),
        2,
        "the turn is shortened, not dropped"
    );
    assert!(
        content_str(&engine.messages[1]).contains("chars)"),
        "oversized content carries the truncation marker"
    );
}
