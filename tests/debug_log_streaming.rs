//! Integration test for the streaming-body capture pipeline.
//!
//! Spins up an in-process HTTP listener that returns a `text/event-stream`
//! response, sends one request through `send_logged()`, drains the streaming
//! body, and verifies the JSONL log file records all three entries
//! (`request`, `response`, `response_body`) — with the actual SSE bytes
//! in the trailing `response_body` entry.
//!
//! Lives in its own integration-test binary because each `tests/*.rs` is
//! compiled separately, so the http_debug `OnceLock` global init in
//! `debug_log_chat.rs` does not collide with this one.

mod support;

use aivo::services::http_debug::{self, LoggedSend};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test(flavor = "current_thread")]
async fn send_logged_streaming_writes_three_jsonl_entries() {
    // 1. Mock HTTP server: respond with a text/event-stream body containing
    //    two SSE frames, then close.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;

        let body = "data: {\"type\":\"message_start\"}\n\n\
                    data: {\"type\":\"message_delta\",\"text\":\"hi\"}\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/event-stream\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            body.len(),
            body
        );
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
    });

    // 2. Initialize the global logger at a tempfile path. Each test binary
    //    has its own GLOBAL OnceLock, so this is fresh.
    let dir = TempDir::new().unwrap();
    let log_path: PathBuf = dir.path().join("debug.jsonl");
    let resolved = http_debug::init(log_path.clone()).await.unwrap();
    let read_path = if resolved == log_path {
        log_path.clone()
    } else {
        resolved
    };

    // 3. Send one POST through send_logged and consume the streaming body.
    let url = format!("http://{addr}/v1/messages");
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("reqwest client should build");
    let resp = client
        .post(&url)
        .header("Authorization", "Bearer secret-test-key")
        .header("Content-Type", "application/json")
        .body(r#"{"prompt":"hi","stream":true}"#)
        .send_logged()
        .await
        .expect("send_logged should succeed against the mock server");
    assert_eq!(resp.status().as_u16(), 200);

    let body_text = resp.text().await.unwrap();
    assert!(
        body_text.contains("message_start") && body_text.contains("message_delta"),
        "caller-side body should contain SSE chunks; got: {body_text}"
    );

    server_handle.await.unwrap();

    // 4. The StreamFinalizer's Drop spawns the response_body write
    //    asynchronously; give it a moment to flush before reading.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let content = tokio::fs::read_to_string(&read_path)
        .await
        .expect("log file should exist after send_logged");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(
        lines.len(),
        3,
        "expected request + response + response_body entries; got {}: {content}",
        lines.len()
    );

    let request_entry: serde_json::Value =
        serde_json::from_str(lines[0]).expect("request line should be valid JSON");
    let response_entry: serde_json::Value =
        serde_json::from_str(lines[1]).expect("response line should be valid JSON");
    let body_entry: serde_json::Value =
        serde_json::from_str(lines[2]).expect("response_body line should be valid JSON");

    assert_eq!(request_entry["phase"], "request");
    assert_eq!(response_entry["phase"], "response");
    assert_eq!(body_entry["phase"], "response_body");
    // All three entries share the same id.
    assert_eq!(request_entry["id"], response_entry["id"]);
    assert_eq!(request_entry["id"], body_entry["id"]);
    // Sensitive header still redacted on the request entry.
    assert_eq!(
        request_entry["request_headers"]["authorization"], "[REDACTED]",
        "Authorization should be redacted; entry: {request_entry}"
    );
    // Headers-only response entry omits response_body entirely.
    assert!(
        response_entry.get("response_body").is_none(),
        "response entry should omit response_body; got: {response_entry}"
    );
    // Body entry has the actual SSE bytes.
    let captured = body_entry["response_body"]
        .as_str()
        .expect("response_body should be a string on the body entry");
    assert!(
        captured.contains("message_start") && captured.contains("message_delta"),
        "captured response_body should contain the SSE chunks; got: {captured}"
    );
}
