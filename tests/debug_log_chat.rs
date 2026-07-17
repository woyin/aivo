//! Integration test for the --debug HTTP capture pipeline.
//!
//! Spins up an in-process HTTP listener, sends one request through
//! `reqwest::RequestBuilder::send_logged`, and verifies the JSONL log file
//! records the round-trip with redacted sensitive headers.
//!
//! Single-test file: the http_debug global OnceLock means only one init()
//! is meaningful per binary. Each tests/*.rs is its own binary, so this
//! file is dedicated to this single test.

mod support;

use aivo::services::http_debug::{self, LoggedSend};
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[tokio::test(flavor = "current_thread")]
async fn send_logged_writes_redacted_jsonl_entry() {
    // 1. Mock HTTP server: accept exactly one connection, drain the request,
    // and reply with a canned 200 + JSON body.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 4096];
        let _ = socket.read(&mut buf).await;

        let body = r#"{"text":"hello-from-mock"}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: application/json\r\n\
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

    // 2. Initialize logger at a tempfile path.
    let dir = TempDir::new().unwrap();
    let log_path: PathBuf = dir.path().join("debug.jsonl");
    let resolved = http_debug::init(log_path.clone()).await.unwrap();
    // Defensive: if some other init won the race, our log file is whatever
    // resolved points to. Assert against that path either way.
    let read_path = if resolved == log_path {
        log_path.clone()
    } else {
        resolved
    };

    // 3. Send one POST through send_logged.
    let url = format!("http://{addr}/v1/messages");
    // Bypass any HTTP_PROXY in the developer's environment — we're talking to
    // a localhost listener that no proxy should intercept.
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("reqwest client should build");
    let resp = client
        .post(&url)
        .header("Authorization", "Bearer secret-test-key")
        .header("Content-Type", "application/json")
        .body(r#"{"prompt":"hi"}"#)
        .send_logged()
        .await
        .expect("send_logged should succeed against the mock server");

    assert_eq!(resp.status().as_u16(), 200);
    let response_text = resp.text().await.unwrap();
    assert!(
        response_text.contains("hello-from-mock"),
        "response body: {response_text}"
    );

    server_handle.await.unwrap();

    // 4. Inspect the JSONL log. The two-entry pattern: a phase=request entry
    // written before send() completes, then a matching phase=response entry
    // afterward. Both share the same `id` field.
    let content = tokio::fs::read_to_string(&read_path)
        .await
        .expect("log file should exist after send_logged");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "expected request + response JSONL lines, got {}: {content}",
        lines.len()
    );

    let request_entry: serde_json::Value =
        serde_json::from_str(lines[0]).expect("request log line should be valid JSON");
    let response_entry: serde_json::Value =
        serde_json::from_str(lines[1]).expect("response log line should be valid JSON");

    assert_eq!(
        request_entry["phase"], "request",
        "first entry: {request_entry}"
    );
    assert_eq!(
        response_entry["phase"], "response",
        "second entry: {response_entry}"
    );
    assert_eq!(
        request_entry["id"], response_entry["id"],
        "request and response entries should share an id"
    );
    assert_eq!(response_entry["status"], 200, "entry: {response_entry}");
    assert_eq!(response_entry["method"], "POST", "entry: {response_entry}");
    // reqwest's HeaderMap lowercases names when iterated, so the captured
    // request_headers map keys are lowercase. The redaction applies to both
    // request entries — verify on the response entry which is the historical
    // baseline this test was written against.
    assert_eq!(
        response_entry["request_headers"]["authorization"], "[REDACTED]",
        "Authorization should be redacted; entry: {response_entry}"
    );
    assert_eq!(
        request_entry["request_headers"]["authorization"], "[REDACTED]",
        "Authorization should be redacted on the pre-send entry too; entry: {request_entry}"
    );
    let response_body = response_entry["response_body"]
        .as_str()
        .expect("response_body should be a string");
    assert!(
        response_body.contains("hello-from-mock"),
        "response_body should contain the canned reply; got: {response_body}"
    );
    let entry_url = response_entry["url"].as_str().unwrap();
    assert!(
        entry_url.contains(&addr.to_string()),
        "url should contain the listener address; got: {entry_url}"
    );
}
