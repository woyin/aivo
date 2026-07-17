use super::super::*;
use super::helpers::*;
use crate::agent::tokens::estimate_str_tokens;

#[test]
fn context_report_splits_system_tools_and_transcript() {
    let mut e = AgentEngine::new(
        "/tmp/proj",
        "deepseek-v4",
        "2026-01-01",
        &[],
        &[],
        200_000,
        0,
    );
    let base = e.context_report();
    assert_eq!(base.context_window, 200_000);
    assert!(base.system_prompt > 0, "system prompt counted");
    assert!(base.tools > 0, "built-in tools counted");
    assert!(base.tool_count > 0, "built-in tools enumerated");
    assert_eq!(base.injected_context, 0, "no -c block yet");
    assert_eq!(base.mcp_tools, 0);
    assert_eq!(base.mcp_tool_count, 0);
    assert_eq!(base.messages, 0, "no transcript yet");
    assert_eq!(base.message_count, 0);
    assert_eq!(base.used(), base.system_prompt + base.tools);
    assert!(base.free() > 0, "window leaves headroom");

    // A `-c` block splits out without inflating the base (only rounding drift).
    let injected = format!("PRIOR SESSION CONTEXT: {}", "x".repeat(4_000));
    e.append_system_context(&injected);
    let with_ctx = e.context_report();
    let injected_est = estimate_str_tokens(&injected) as u64;
    assert!(
        with_ctx.injected_context.abs_diff(injected_est) <= 5,
        "injected segment ≈ the block's estimate: {} vs {injected_est}",
        with_ctx.injected_context
    );
    assert!(
        with_ctx.system_prompt.abs_diff(base.system_prompt) <= 5,
        "base system prompt unchanged by the append: {} vs {}",
        with_ctx.system_prompt,
        base.system_prompt
    );

    // A transcript turn lands in `messages`, not system/tools.
    e.seed_history([("user".to_string(), "hello there".to_string())]);
    let with_msg = e.context_report();
    assert!(with_msg.messages > 0, "transcript counted");
    assert_eq!(with_msg.message_count, 1);
    assert!(with_msg.system_prompt.abs_diff(base.system_prompt) <= 5);
}

#[test]
fn context_report_rescale_anchors_total_and_keeps_proportions() {
    let mut r = ContextReport {
        context_window: 100_000,
        system_prompt: 6_000,
        tools: 4_000,
        messages: 2_000,
        ..Default::default()
    };
    assert_eq!(r.used(), 12_000);
    r.rescale(6_000);
    assert!(
        r.used().abs_diff(6_000) <= 2,
        "total ≈ target: {}",
        r.used()
    );
    assert!(r.system_prompt.abs_diff(3_000) <= 1, "shares preserved");
    assert!(r.tools.abs_diff(2_000) <= 1);
    assert!(r.messages.abs_diff(1_000) <= 1);
    r.rescale(0); // no-op, never blank the breakdown
    assert!(r.used() > 0);
}

#[test]
fn take_turn_cost_usd_drains_reported_spend() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    assert_eq!(e.take_turn_cost_usd(), None);
    e.turn_cost_usd = 0.0421;
    assert_eq!(e.take_turn_cost_usd(), Some(0.0421));
    assert_eq!(e.take_turn_cost_usd(), None, "drained with the turn");
}

/// The engine sums each step's provider-measured token split across a turn and surfaces it via `take_turn_usage`.
#[tokio::test]
async fn turn_usage_accumulates_split_across_steps() {
    let dir = tmp();
    // Step 1: a tool call that ALSO reports usage with a cache split.
    let step1 = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_bash\",\"arguments\":\"{\\\"command\\\":\\\"echo hi\\\"}\"}}]}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"prompt_tokens_details\":{\"cached_tokens\":40}}}\n\ndata: [DONE]\n\n".to_string();
    // Step 2: the converging text reply, reporting more usage.
    let step2 = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}],\"usage\":{\"prompt_tokens\":200,\"completion_tokens\":30}}\n\ndata: [DONE]\n\n".to_string();
    let port = spawn_sse_sequence(vec![step1, step2]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("go".into()),
        &mut ui,
    )
    .await;

    let usage = engine.take_turn_usage();
    assert_eq!(usage.prompt_tokens, 300, "prompt summed across steps");
    assert_eq!(
        usage.completion_tokens, 50,
        "completion summed across steps"
    );
    assert_eq!(usage.cache_read_tokens, 40, "cached tokens captured");
    // Draining leaves the accumulator zeroed for the next turn.
    assert_eq!(engine.take_turn_usage(), SessionTokens::default());
}
