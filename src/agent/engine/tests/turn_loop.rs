use super::super::*;
use super::helpers::*;
use crate::agent::request::content_str;
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;

/// Full loop: first turn emits a write_file call, second turn answers with text → converges.
#[tokio::test]
async fn engine_runs_tool_then_converges() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![WRITE_TOOL_SSE.to_string(), FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("write out.txt".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.tools, vec!["write_file"]);
    assert_eq!(ui.text, "done");
    assert!(dir.join("out.txt").exists());
    assert_eq!(std::fs::read_to_string(dir.join("out.txt")).unwrap(), "hi");
}

#[tokio::test]
async fn leaked_tool_call_markup_is_stripped_and_nudged() {
    let dir = tmp();
    let leaked = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
    );
    let port = spawn_sse_sequence(vec![leaked, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read the file".into()),
        &mut ui,
    )
    .await;

    // Nudge is its own `user` message, preceded by an `assistant` turn.
    let nudge_idx = engine
        .messages
        .iter()
        .position(|m| {
            m["content"]
                .as_str()
                .is_some_and(|c| c.contains("wrote tool calls as plain text"))
        })
        .expect("expected a leaked-tool-call nudge in history");
    assert_eq!(engine.messages[nudge_idx]["role"], "user");
    assert_eq!(engine.messages[nudge_idx - 1]["role"], "assistant");
    assert_no_consecutive_user(&engine.messages);
    assert!(
        !engine.messages.iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|c| c.contains("<tool_calls>"))),
        "leaked markup should be stripped from history"
    );
    let last = engine.messages.last().unwrap();
    assert_eq!(last["role"], "assistant");
    assert_eq!(last["content"], "done");
    assert_eq!(
        ui.discards, 1,
        "engine must drop the leaked streamed segment"
    );
    assert_eq!(ui.text, "done");
}

/// Regression: a leak after a tool step must not produce a user-after-tool 400.
#[tokio::test]
async fn leaked_tool_call_after_tool_step_keeps_roles_alternating() {
    let dir = tmp();
    let bash = tool_call_sse("run_bash", json!({ "command": "echo hi" }));
    let leaked = format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
    );
    let port = spawn_sse_sequence(vec![bash, leaked, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("run it".into()),
        &mut ui,
    )
    .await;

    assert_no_consecutive_user(&engine.messages);
    for i in 1..engine.messages.len() {
        if role(&engine.messages[i]) == "user" {
            assert_ne!(
                role(&engine.messages[i - 1]),
                "tool",
                "a user message directly after tool results bricks the Anthropic bridge"
            );
        }
    }
    assert_eq!(engine.messages.last().unwrap()["content"], "done");
}

#[tokio::test]
async fn leaked_tool_call_nudges_are_capped() {
    let dir = tmp();
    let leaked = || {
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
        )
    };
    let port = spawn_sse_sequence(vec![leaked(), leaked(), leaked(), leaked()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read it".into()),
        &mut ui,
    )
    .await;

    let nudges = engine
        .messages
        .iter()
        .filter(|m| {
            m["content"]
                .as_str()
                .is_some_and(|c| c.contains("wrote tool calls as plain text"))
        })
        .count();
    assert_eq!(nudges, MAX_LEAKED_NUDGES, "nudges must be capped");
    assert_no_consecutive_user(&engine.messages);
}

/// A paging loop varying an ignored arg (`limit`) makes a distinct `batch_sig` each
/// step, so only the page-read guard can stop it — the read_file runaway shape.
#[tokio::test]
async fn paging_loop_with_varying_junk_args_is_stopped() {
    let dir = tmp();
    std::fs::write(dir.join("big.txt"), "x\n".repeat(200)).unwrap();
    let mut seq: Vec<String> = (0..8)
        .map(|i| {
            tool_call_sse(
                "read_file",
                json!({ "path": "big.txt", "offset": 1, "limit": 10 + i }),
            )
        })
        .collect();
    seq.push(FINAL_TEXT_SSE.to_string());
    let port = spawn_sse_sequence(seq);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read the file".into()),
        &mut ui,
    )
    .await;
    let reads = ui
        .tools
        .iter()
        .filter(|t| t.as_str() == "read_file")
        .count();
    assert!(
        reads <= REPEAT_LIMIT,
        "page guard should stop the loop; ran {reads} reads"
    );
    assert!(
        ui.notices
            .iter()
            .any(|n| n.contains("repeated the same action"))
    );
}

/// A vision turn keeps the tool loop: the image rides in the opening message while
/// tools still run.
#[tokio::test]
async fn run_turn_with_content_keeps_image_and_runs_tools() {
    let dir = tmp();
    let ls = tool_call_sse("list_dir", json!({ "path": "." }));
    let port = spawn_sse_sequence(vec![ls, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let content = json!([
        {"type": "text", "text": "what's in this screenshot?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBBCCCC"}},
    ]);
    let mut ui = CapturingUi::default();
    engine
        .run_turn_with_content(
            &turn_ctx(&client, &base, &dir),
            &mut ui,
            content,
            "what's in this screenshot?".into(),
        )
        .await;
    assert!(ui.tools.contains(&"list_dir".to_string()));
    let user = engine
        .messages
        .iter()
        .find(|m| m["role"] == "user")
        .expect("a user message");
    let parts = user["content"].as_array().expect("array content");
    assert!(parts.iter().any(|p| p["type"] == "image_url"));
}

#[tokio::test]
async fn steering_folds_into_the_last_tool_result_at_the_batch_boundary() {
    let dir = tmp();
    let ls = tool_call_sse("list_dir", json!({ "path": "." }));
    let port = spawn_sse_sequence(vec![ls, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi {
        steering: vec!["actually, focus on the README".to_string()],
        ..Default::default()
    };
    engine
        .run_turn(
            &turn_ctx(&client, &base, &dir),
            &mut ui,
            "look around".into(),
        )
        .await;
    assert!(ui.steering.is_empty(), "the queue must be drained");
    let tool_msg = engine
        .messages
        .iter()
        .rfind(|m| m["role"] == "tool")
        .expect("a tool result");
    let content = tool_msg["content"].as_str().unwrap();
    assert!(
        content.contains("<user_interjection>")
            && content.contains("actually, focus on the README"),
        "interjection must ride the tool result: {content}"
    );
    let roles: Vec<&str> = engine
        .messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or(""))
        .collect();
    assert!(
        !roles.windows(2).any(|w| w == ["tool", "user"]),
        "no bare user turn after tool results: {roles:?}"
    );
}

/// A batch of read-only calls runs concurrently but results stay in call order, each paired to its `tool_call_id`.
#[tokio::test]
async fn parallel_read_batch_preserves_order_and_pairing() {
    let dir = tmp();
    std::fs::write(dir.join("a.txt"), "ALPHA").unwrap();
    std::fs::write(dir.join("b.txt"), "BETA").unwrap();
    std::fs::write(dir.join("c.txt"), "GAMMA").unwrap();
    let batch = batch_tool_call_sse(&[
        ("c0", "read_file", json!({"path": "a.txt"})),
        ("c1", "read_file", json!({"path": "b.txt"})),
        ("c2", "read_file", json!({"path": "c.txt"})),
    ]);
    let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read all three".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.tools, vec!["read_file", "read_file", "read_file"]);
    // Results in call order, each keyed to the right id and content.
    let tool_msgs: Vec<(&str, &str)> = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .map(|m| {
            (
                m["tool_call_id"].as_str().unwrap(),
                m["content"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(tool_msgs.len(), 3);
    assert_eq!(tool_msgs[0].0, "c0");
    assert!(tool_msgs[0].1.contains("ALPHA"));
    assert_eq!(tool_msgs[1].0, "c1");
    assert!(tool_msgs[1].1.contains("BETA"));
    assert_eq!(tool_msgs[2].0, "c2");
    assert!(tool_msgs[2].1.contains("GAMMA"));
    assert_eq!(ui.text, "done");
}

/// A mixed batch (parallel read + ordered write) records results in call order and the write lands.
#[tokio::test]
async fn mixed_batch_orders_results_and_runs_write() {
    let dir = tmp();
    std::fs::write(dir.join("a.txt"), "ALPHA").unwrap();
    let batch = batch_tool_call_sse(&[
        ("c0", "read_file", json!({"path": "a.txt"})),
        (
            "c1",
            "write_file",
            json!({"path": "out.txt", "content": "WROTE"}),
        ),
    ]);
    let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("read then write".into()),
        &mut ui,
    )
    .await;

    let tool_msgs: Vec<(&str, &str)> = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .map(|m| {
            (
                m["tool_call_id"].as_str().unwrap(),
                m["content"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(tool_msgs.len(), 2);
    assert_eq!(tool_msgs[0].0, "c0");
    assert!(tool_msgs[0].1.contains("ALPHA"));
    assert_eq!(tool_msgs[1].0, "c1");
    assert!(dir.join("out.txt").exists());
    assert_eq!(
        std::fs::read_to_string(dir.join("out.txt")).unwrap(),
        "WROTE"
    );
}

/// An empty completion converges the turn but isn't recorded as an assistant message (empty → invalid Anthropic content array).
#[tokio::test]
async fn empty_completion_is_not_recorded_as_assistant_turn() {
    let dir = tmp();
    let empty = "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n".to_string();
    let port = spawn_sse_sequence(vec![empty]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("hi".into()),
        &mut ui,
    )
    .await;

    assert!(
        !engine.messages.iter().any(|m| role(m) == "assistant"),
        "empty completion must not record an assistant turn: {:?}",
        engine.messages
    );
    // The turn still ran (user message recorded).
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "user" && content_str(m) == "hi")
    );
    // No answer must ride the ERROR channel (persisted entry, no `✶ Done`,
    // headless fails closed).
    assert!(
        ui.errors.iter().any(|e| e.contains("empty response")),
        "empty response must use notify_error: {:?} / {:?}",
        ui.errors,
        ui.notices
    );
}

// Unix-only: the mock's raw sequential-`accept()` sequencing is fragile on Windows; the retry-past-503 logic is platform-agnostic.
#[cfg(unix)]
#[tokio::test]
async fn engine_retries_then_succeeds() {
    // First connection returns 503 (retryable, before any stream); the retry hits a 200.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = sock.read(&mut buf);
            let body = "overloaded";
            let resp = format!(
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes());
        }
        if let Ok((mut sock, _)) = listener.accept() {
            let mut buf = [0u8; 8192];
            let _ = sock.read(&mut buf);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                FINAL_TEXT_SSE.len(),
                FINAL_TEXT_SSE
            );
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    // Make the backoff instant for the test.
    unsafe { std::env::set_var("AIVO_AGENT_RETRY_BASE_MS", "1") };
    let dir = tmp();
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("hi".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.text, "done"); // retried past the 503 and got the content
    assert!(
        ui.notices.iter().any(|n| n.contains("retrying")),
        "expected a retry notice, got {:?}",
        ui.notices
    );
}
