//! End-to-end cascade tests: each router against a fake provider that speaks
//! only one protocol, asserting convergence, learned routes, and that failed
//! cascades surface the real upstream error instead of thrashing.

mod support;

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use aivo::services::anthropic_to_openai_router::{
    AnthropicToOpenAIRouter, AnthropicToOpenAIRouterConfig,
};
use aivo::services::gemini_router::{GeminiRouter, GeminiRouterConfig};
use aivo::services::log_store::LogStore;
use aivo::services::provider_protocol::ProviderProtocol;
use aivo::services::responses_to_chat_router::{
    ResponsesToChatRouter, ResponsesToChatRouterConfig,
};
use aivo::services::serve_router::{ServeRouter, ServeRouterConfig};
use aivo::services::session_store::ApiKey;
use serde_json::Value;
use zeroize::Zeroizing;

// ── Fake provider ────────────────────────────────────────────────────────

/// Which wire protocol a request hit, classified from its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Endpoint {
    Chat,
    Messages,
    Responses,
    Gemini,
    Other,
}

fn classify(path: &str) -> Endpoint {
    if path.contains("chat/completions") {
        Endpoint::Chat
    } else if path.contains(":generateContent") || path.contains(":streamGenerateContent") {
        Endpoint::Gemini
    } else if path.contains("/responses") {
        Endpoint::Responses
    } else if path.contains("/messages") {
        Endpoint::Messages
    } else {
        Endpoint::Other
    }
}

/// Per-endpoint behavior of the fake provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// 200 with a canned success body in the endpoint's native shape.
    Ok,
    OkSse,
    /// 400 with a structured error envelope — a semantic rejection that must
    /// bail. Unlisted endpoints 404 with `{"error":"Not found"}`, a
    /// path-missing mismatch the cascade may walk past.
    SemanticReject,
}

#[derive(Clone)]
struct FakeProvider {
    port: u16,
    hits: Arc<Mutex<Vec<Endpoint>>>,
    heads: Arc<Mutex<Vec<String>>>,
}

impl FakeProvider {
    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn hits(&self) -> Vec<Endpoint> {
        self.hits.lock().unwrap().clone()
    }

    fn hit_count(&self, endpoint: Endpoint) -> usize {
        self.hits().iter().filter(|e| **e == endpoint).count()
    }

    /// Raw request heads (request line + headers) in arrival order.
    fn heads(&self) -> Vec<String> {
        self.heads.lock().unwrap().clone()
    }
}

fn success_body(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Chat => r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"test-model","choices":[{"index":0,"message":{"role":"assistant","content":"hello from openai"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#.to_string(),
        Endpoint::Messages => r#"{"id":"msg_1","type":"message","role":"assistant","model":"test-model","content":[{"type":"text","text":"hello from anthropic"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#.to_string(),
        Endpoint::Responses => r#"{"id":"resp_1","object":"response","status":"completed","model":"test-model","output":[{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello from responses"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}"#.to_string(),
        Endpoint::Gemini => r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"hello from gemini"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#.to_string(),
        Endpoint::Other => r#"{}"#.to_string(),
    }
}

fn sse_body(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Messages => concat!(
            "event: message_start\n",
            r#"data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"test-model","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}"#,
            "\n\n",
            "event: content_block_start\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            "\n\n",
            "event: content_block_delta\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello from anthropic"}}"#,
            "\n\n",
            "event: content_block_stop\n",
            r#"data: {"type":"content_block_stop","index":0}"#,
            "\n\n",
            "event: message_delta\n",
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#,
            "\n\n",
            "event: message_stop\n",
            r#"data: {"type":"message_stop"}"#,
            "\n\n",
        )
        .to_string(),
        // Gemini streams bare JSON candidates as data lines.
        _ => format!("data: {}\n\n", success_body(endpoint)),
    }
}

/// Drains the body so the client never sees a closed pipe mid-write.
fn read_request_head(stream: &mut std::net::TcpStream) -> String {
    let mut request = Vec::new();
    let mut buf = [0u8; 4096];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => request.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    let head = String::from_utf8_lossy(&request).into_owned();
    if let Some(len) = head
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        let header_end = request
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .map(|p| p + 4)
            .unwrap_or(request.len());
        let mut have = request.len() - header_end;
        while have < len {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => have += n,
                Err(_) => break,
            }
        }
    }
    head
}

fn request_endpoint(head: &str) -> Endpoint {
    let path = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("/");
    classify(path)
}

/// Spawns a blocking fake provider; `modes` maps endpoints to behaviors,
/// anything unlisted 404s.
fn spawn_fake(modes: &[(Endpoint, Mode)]) -> FakeProvider {
    let modes: HashMap<Endpoint, Mode> = modes.iter().copied().collect();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits: Arc<Mutex<Vec<Endpoint>>> = Arc::new(Mutex::new(Vec::new()));
    let hits_writer = hits.clone();
    let heads: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let heads_writer = heads.clone();

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let head = read_request_head(&mut stream);
            let endpoint = request_endpoint(&head);
            hits_writer.lock().unwrap().push(endpoint);
            heads_writer.lock().unwrap().push(head);

            let (status, reason, content_type, body) = match modes.get(&endpoint) {
                Some(Mode::Ok) => (200, "OK", "application/json", success_body(endpoint)),
                Some(Mode::OkSse) => (200, "OK", "text/event-stream", sse_body(endpoint)),
                Some(Mode::SemanticReject) => (
                    400,
                    "Bad Request",
                    "application/json",
                    r#"{"error":{"type":"invalid_request_error","message":"bad request body"}}"#
                        .to_string(),
                ),
                None => (
                    404,
                    "Not Found",
                    "application/json",
                    r#"{"error":"Not found"}"#.to_string(),
                ),
            };
            let head = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
    });

    FakeProvider { port, hits, heads }
}

/// Withholds the closing SSE frames until the returned sender fires — distinguishes incremental forwarding from buffer-to-EOF.
fn spawn_fake_sse_drip() -> (FakeProvider, std::sync::mpsc::Sender<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits: Arc<Mutex<Vec<Endpoint>>> = Arc::new(Mutex::new(Vec::new()));
    let hits_writer = hits.clone();
    let heads: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let head = read_request_head(&mut stream);
            let endpoint = request_endpoint(&head);
            hits_writer.lock().unwrap().push(endpoint);

            if endpoint != Endpoint::Messages {
                let body = r#"{"error":"Not found"}"#;
                let head = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body.as_bytes());
                continue;
            }

            let body = sse_body(Endpoint::Messages);
            let split = body
                .find("event: message_delta")
                .expect("drip split marker");
            let (first, tail) = body.split_at(split);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(first.as_bytes());
            let _ = stream.flush();
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(15));
            let _ = stream.write_all(tail.as_bytes());
        }
    });

    (FakeProvider { port, hits, heads }, release_tx)
}

// ── Client helpers ───────────────────────────────────────────────────────

/// Sends a raw request with arbitrary extra header lines; `None` body → GET-style.
async fn raw_request(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &[&str],
    body: Option<&str>,
) -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let mut headers = String::new();
    for h in extra_headers {
        headers.push_str(h);
        headers.push_str("\r\n");
    }
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{headers}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let _ = stream.shutdown().await;
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
}

async fn raw_post(port: u16, path: &str, body: &str) -> String {
    raw_request(
        port,
        "POST",
        path,
        &["Authorization: Bearer tok"],
        Some(body),
    )
    .await
}

/// Reads until `marker` appears (10s cap — a buffering proxy fails here), then to EOF after firing `release`.
/// Returns (bytes seen at the marker, full response).
async fn raw_post_until_marker_then_release(
    port: u16,
    path: &str,
    body: &str,
    marker: &str,
    release: std::sync::mpsc::Sender<()>,
) -> (String, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer tok\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let _ = stream.shutdown().await;

    let mut buf = Vec::new();
    let read_until_marker = async {
        let mut chunk = [0u8; 4096];
        while !String::from_utf8_lossy(&buf).contains(marker) {
            let n = stream.read(&mut chunk).await.unwrap();
            assert!(
                n > 0,
                "stream closed before {marker:?}: {}",
                String::from_utf8_lossy(&buf)
            );
            buf.extend_from_slice(&chunk[..n]);
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(10), read_until_marker)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "proxy did not forward {marker:?} while upstream was still open (buffered?): {}",
                String::from_utf8_lossy(&buf)
            )
        });
    let at_marker = String::from_utf8_lossy(&buf).into_owned();

    release.send(()).unwrap();
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest).await;
    buf.extend_from_slice(&rest);
    (at_marker, String::from_utf8_lossy(&buf).into_owned())
}

fn response_status(response: &str) -> u16 {
    response
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

fn response_json(response: &str) -> Value {
    let body = response.split("\r\n\r\n").nth(1).unwrap_or("");
    // Strip chunked transfer-encoding framing if present.
    if response.contains("Transfer-Encoding: chunked") {
        let mut out = String::new();
        let mut rest = body;
        while let Some(nl) = rest.find("\r\n") {
            let (size_line, tail) = rest.split_at(nl);
            let size = usize::from_str_radix(size_line.trim(), 16).unwrap_or(0);
            if size == 0 {
                break;
            }
            let tail = &tail[2..];
            out.push_str(&tail[..size.min(tail.len())]);
            rest = tail.get(size + 2..).unwrap_or("");
        }
        serde_json::from_str(&out).unwrap_or(Value::Null)
    } else {
        serde_json::from_str(body).unwrap_or(Value::Null)
    }
}

fn no_proxy() {
    aivo::services::launch_runtime::ensure_loopback_no_proxy_in_process_env();
}

fn test_key(base_url: &str) -> ApiKey {
    ApiKey {
        id: "e2e".to_string(),
        name: "e2e".to_string(),
        base_url: base_url.to_string(),
        claude_protocol: None,
        gemini_protocol: None,
        responses_api_supported: None,
        codex_mode: None,
        opencode_mode: None,
        pi_mode: None,
        claude_path_variant: None,
        gemini_path_variant: None,
        requires_reasoning_content: None,
        protocol_routes: Default::default(),
        routing_schema_version: 0,
        key: Zeroizing::new("sk-test".to_string()),
        created_at: "2025-01-01T00:00:00Z".to_string(),
    }
}

fn responses_router_config(base_url: String) -> ResponsesToChatRouterConfig {
    ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key: "sk-test".to_string(),
        target_protocol: ProviderProtocol::Openai,
        target_path_variant: None,
        copilot_token_manager: None,
        model_prefix: None,
        requires_reasoning_content: false,
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: None,
        is_starter: false,
        aivo_prefix_models: Vec::new(),
    }
}

fn serve_config(base_url: String, protocol: ProviderProtocol) -> ServeRouterConfig {
    ServeRouterConfig {
        upstream_base_url: base_url,
        upstream_api_key: "sk-test".to_string(),
        upstream_protocol: protocol,
        is_copilot: false,
        is_grok: false,
        grok_fallback_api_key: None,
        is_kimi: false,
        is_codex: false,
        is_claude_native_oauth: false,
        is_openrouter: false,
        is_starter: false,
        requires_reasoning_content: false,
        cors: false,
        timeout: 30,
        auth_token: None,
        aliases: HashMap::new(),
    }
}

// ── aivo claude (anthropic_to_openai_router) ─────────────────────────────

const ANTHROPIC_REQ: &str =
    r#"{"model":"test-model","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}"#;

#[tokio::test]
async fn claude_router_converges_on_chat_only_upstream_and_learns_route() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = AnthropicToOpenAIRouter::new(AnthropicToOpenAIRouterConfig {
        target_base_url: fake.base_url(),
        target_api_key: "sk-test".to_string(),
        seed_routes: BTreeMap::new(),
        strip_cache_control: false,
        model_prefix: None,
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    });
    let (port, routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(json["content"][0]["text"], "hello from openai", "{resp}");

    // The model's route converged on OpenAI chat and is marked for persistence.
    let dirty = routes.dirty_routes();
    assert_eq!(dirty.len(), 1, "dirty routes: {dirty:?}");
    assert_eq!(dirty[0].0, "test-model");
    assert_eq!(dirty[0].1.protocol, "openai");

    // Second request goes straight to chat/completions — no /messages re-probe.
    let messages_probes = fake.hit_count(Endpoint::Messages);
    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert_eq!(
        fake.hit_count(Endpoint::Messages),
        messages_probes,
        "pinned route must skip the native /v1/messages probe: {:?}",
        fake.hits()
    );
    handle.abort();
}

#[tokio::test]
async fn claude_router_surfaces_semantic_rejection_without_thrash() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::SemanticReject)]);
    let router = AnthropicToOpenAIRouter::new(AnthropicToOpenAIRouterConfig {
        target_base_url: fake.base_url(),
        target_api_key: "sk-test".to_string(),
        seed_routes: BTreeMap::new(),
        strip_cache_control: false,
        model_prefix: None,
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    });
    let (port, _routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 400, "{resp}");
    // The structured 400 is authoritative — the cascade must not probe
    // Google/other protocols after it.
    assert_eq!(fake.hit_count(Endpoint::Chat), 1, "{:?}", fake.hits());
    assert_eq!(fake.hit_count(Endpoint::Gemini), 0, "{:?}", fake.hits());
    handle.abort();
}

// ── aivo codex (responses_to_chat_router) ────────────────────────────────

const RESPONSES_REQ: &str = r#"{"model":"test-model","stream":false,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#;

#[tokio::test]
async fn codex_router_bridges_responses_client_to_chat_upstream() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = ResponsesToChatRouter::new(responses_router_config(fake.base_url()))
        .with_auth_token("tok".to_string());
    let (port, _routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post(port, "/v1/responses", RESPONSES_REQ).await;
    assert_eq!(
        response_status(&resp),
        200,
        "hits={:?} resp={resp}",
        fake.hits()
    );
    assert!(resp.contains("hello from openai"), "{resp}");
    handle.abort();
}

#[tokio::test]
async fn codex_router_cascades_to_anthropic_messages_and_learns_route() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let router = ResponsesToChatRouter::new(responses_router_config(fake.base_url()))
        .with_auth_token("tok".to_string());
    let (port, routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post(port, "/v1/responses", RESPONSES_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("hello from anthropic"), "{resp}");

    let dirty = routes.dirty_routes();
    assert_eq!(dirty.len(), 1, "dirty routes: {dirty:?}");
    assert_eq!(dirty[0].0, "test-model");
    assert_eq!(dirty[0].1.protocol, "anthropic");
    handle.abort();
}

// ── aivo gemini (gemini_router) ──────────────────────────────────────────

const GEMINI_REQ: &str = r#"{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}"#;

#[tokio::test]
async fn gemini_router_cascades_to_chat_upstream_and_learns_route() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: fake.base_url(),
        api_key: "sk-test".to_string(),
        upstream_protocol: ProviderProtocol::Google,
        forced_model: None,
        copilot_token_manager: None,
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    });
    let (port, routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post(
        port,
        "/v1beta/models/test-model:generateContent",
        GEMINI_REQ,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(
        json["candidates"][0]["content"]["parts"][0]["text"], "hello from openai",
        "{resp}"
    );
    assert!(fake.hit_count(Endpoint::Gemini) >= 1, "{:?}", fake.hits());

    let dirty = routes.dirty_routes();
    assert_eq!(dirty.len(), 1, "dirty routes: {dirty:?}");
    assert_eq!(dirty[0].0, "test-model");
    assert_eq!(dirty[0].1.protocol, "openai");
    handle.abort();
}

// ── aivo serve (serve_router) ────────────────────────────────────────────

const CHAT_REQ: &str = r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#;
const CHAT_REQ_STREAM: &str =
    r#"{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

async fn start_serve(fake: &FakeProvider, protocol: ProviderProtocol) -> u16 {
    start_serve_opt(fake, protocol, None).await
}

/// `start_serve`, optionally injecting a caller-owned route cache (the seam
/// `aivo code` uses to remember the negotiated protocol across turns/launches).
async fn start_serve_opt(
    fake: &FakeProvider,
    protocol: ProviderProtocol,
    cache: Option<Arc<aivo::services::route_cache::RouteCache>>,
) -> u16 {
    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    // tempdir leaks for the test's lifetime; the router holds only the path.
    std::mem::forget(tmp);
    let mut router = ServeRouter::new(
        serve_config(fake.base_url(), protocol),
        test_key(&fake.base_url()),
        log_store,
    );
    if let Some(cache) = cache {
        router = router.with_route_cache(cache);
    }
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();
    port
}

async fn start_serve_with_token(fake: &FakeProvider, protocol: ProviderProtocol) -> u16 {
    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    std::mem::forget(tmp);
    let mut config = serve_config(fake.base_url(), protocol);
    config.auth_token = Some("tok".to_string());
    let router = ServeRouter::new(config, test_key(&fake.base_url()), log_store);
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();
    port
}

/// An injected cache must capture the protocol the serve auto-switched to (and
/// mark it confirmed), so `aivo code` can persist it after the turn.
#[tokio::test]
async fn serve_router_learns_into_injected_route_cache() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let cache = Arc::new(aivo::services::route_cache::RouteCache::new(
        "code",
        ProviderProtocol::Openai,
        BTreeMap::new(),
    ));
    let port = start_serve_opt(&fake, ProviderProtocol::Openai, Some(cache.clone())).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");

    // The shared cache now holds the negotiated route: Anthropic, confirmed.
    let slot = cache.resolve("test-model");
    assert_eq!(slot.current().0, ProviderProtocol::Anthropic);
    assert!(slot.is_confirmed(), "proven route must be confirmed");
    let dirty = cache.dirty_routes();
    assert_eq!(dirty.len(), 1, "dirty routes: {dirty:?}");
    assert_eq!(dirty[0].0, "test-model");
    assert_eq!(dirty[0].1.protocol, "anthropic");
}

/// A cache seeded with a model's confirmed route makes the serve skip the
/// protocol probe — the cross-turn/cross-launch memory that stops `aivo code`'s
/// agent path re-negotiating every turn.
#[tokio::test]
async fn serve_router_seeded_route_skips_probe() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let mut seed = BTreeMap::new();
    seed.insert(
        "test-model".to_string(),
        aivo::services::route_cache::PersistedRoute {
            protocol: "anthropic".to_string(),
            path_variant: String::new(),
        },
    );
    // Default guess is OpenAI, but the seed pins Anthropic as already-proven.
    let cache = Arc::new(aivo::services::route_cache::RouteCache::new(
        "code",
        ProviderProtocol::Openai,
        seed,
    ));
    let port = start_serve_opt(&fake, ProviderProtocol::Openai, Some(cache)).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert_eq!(
        fake.hit_count(Endpoint::Chat),
        0,
        "seeded route must skip the chat probe: {:?}",
        fake.hits()
    );
}

#[tokio::test]
async fn serve_streams_anthropic_sse_as_chat_sse() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::OkSse)]);
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ_STREAM).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("chat.completion.chunk"), "{resp}");
    assert!(resp.contains("hello from anthropic"), "{resp}");
    assert!(resp.contains("data: [DONE]"), "{resp}");
}

#[tokio::test]
async fn serve_streams_gemini_sse_as_chat_sse() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Gemini, Mode::OkSse)]);
    let port = start_serve(&fake, ProviderProtocol::Google).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ_STREAM).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("chat.completion.chunk"), "{resp}");
    assert!(resp.contains("hello from gemini"), "{resp}");
    assert!(resp.contains("data: [DONE]"), "{resp}");
}

/// First chunk must reach the client while the upstream SSE stream is still open — pins incremental forwarding against a buffer-to-EOF regression.
#[tokio::test]
async fn serve_streams_chat_first_chunk_before_upstream_completes() {
    no_proxy();
    let (fake, release) = spawn_fake_sse_drip();
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let (at_marker, full) = raw_post_until_marker_then_release(
        port,
        "/v1/chat/completions",
        CHAT_REQ_STREAM,
        "hello from anthropic",
        release,
    )
    .await;
    assert!(at_marker.contains("chat.completion.chunk"), "{at_marker}");
    assert!(
        !at_marker.contains("data: [DONE]"),
        "stream ended before upstream tail: {at_marker}"
    );
    assert!(full.contains("data: [DONE]"), "{full}");
}

/// Same incremental guarantee through the two-hop Responses route (chained adapters).
#[tokio::test]
async fn serve_responses_route_streams_first_chunk_before_upstream_completes() {
    no_proxy();
    let (fake, release) = spawn_fake_sse_drip();
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let stream_req = r#"{"model":"test-model","stream":true,"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]}"#;
    let (at_marker, full) = raw_post_until_marker_then_release(
        port,
        "/v1/responses",
        stream_req,
        "hello from anthropic",
        release,
    )
    .await;
    assert!(
        at_marker.contains("response.output_text.delta"),
        "{at_marker}"
    );
    assert!(full.contains("response.completed"), "{full}");
}

// ── serve native-protocol inbound ────────────────────────────────────────

const MESSAGES_REQ_STREAM: &str = r#"{"model":"test-model","max_tokens":128,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;

#[tokio::test]
async fn serve_messages_route_bridges_anthropic_client_to_chat_upstream() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(json["type"], "message", "{resp}");
    assert_eq!(json["role"], "assistant", "{resp}");
    assert_eq!(json["content"][0]["text"], "hello from openai", "{resp}");
    assert!(json["usage"]["input_tokens"].is_u64(), "{resp}");
}

/// Double hop (Anthropic SSE → Chat → Anthropic SSE) must still forward incrementally.
#[tokio::test]
async fn serve_messages_route_streams_first_chunk_before_upstream_completes() {
    no_proxy();
    let (fake, release) = spawn_fake_sse_drip();
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let (at_marker, full) = raw_post_until_marker_then_release(
        port,
        "/v1/messages",
        MESSAGES_REQ_STREAM,
        "hello from anthropic",
        release,
    )
    .await;
    assert!(at_marker.contains("content_block_delta"), "{at_marker}");
    assert!(full.contains("message_stop"), "{full}");
}

#[tokio::test]
async fn serve_gemini_route_bridges_gemini_client_to_chat_upstream() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(
        port,
        "/v1beta/models/test-model:generateContent",
        GEMINI_REQ,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(
        json["candidates"][0]["content"]["parts"][0]["text"], "hello from openai",
        "{resp}"
    );
}

/// streamGenerateContent is emulated here (caps.stream: false): the buffered response ships as one Gemini SSE event.
#[tokio::test]
async fn serve_gemini_route_streams_as_single_sse_event() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(
        port,
        "/v1beta/models/test-model:streamGenerateContent",
        GEMINI_REQ,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("hello from openai"), "{resp}");
    assert!(resp.contains("data: "), "{resp}");
}

/// generateContent takes the direct reverse edge — hits /v1/messages, never /chat/completions.
#[tokio::test]
async fn serve_gemini_route_uses_direct_anthropic_edge() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let resp = raw_post(
        port,
        "/v1beta/models/test-model:generateContent",
        GEMINI_REQ,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(
        json["candidates"][0]["content"]["parts"][0]["text"], "hello from anthropic",
        "{resp}"
    );
    assert!(fake.hit_count(Endpoint::Messages) >= 1, "{:?}", fake.hits());
    assert_eq!(fake.hit_count(Endpoint::Chat), 0, "{:?}", fake.hits());
}

/// Reverse edge streaming is emulated (caps.stream: false): the buffered reply ships as one Gemini SSE event.
#[tokio::test]
async fn serve_gemini_route_direct_anthropic_edge_streams_single_event() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Anthropic).await;

    let resp = raw_post(
        port,
        "/v1beta/models/test-model:streamGenerateContent",
        GEMINI_REQ,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("hello from anthropic"), "{resp}");
    assert_eq!(fake.hit_count(Endpoint::Chat), 0, "{:?}", fake.hits());
}

#[tokio::test]
async fn serve_native_routes_accept_native_auth_forms() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let port = start_serve_with_token(&fake, ProviderProtocol::Openai).await;

    let resp =
        raw_post_with_auth(port, "/v1/messages", ANTHROPIC_REQ, Some("x-api-key: tok")).await;
    assert_eq!(response_status(&resp), 200, "{resp}");

    let resp = raw_post_with_auth(
        port,
        "/v1beta/models/test-model:generateContent",
        GEMINI_REQ,
        Some("x-goog-api-key: tok"),
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let resp = raw_post_with_auth(
        port,
        "/v1beta/models/test-model:generateContent?key=tok",
        GEMINI_REQ,
        None,
    )
    .await;
    assert_eq!(response_status(&resp), 200, "{resp}");

    let resp = raw_post_with_auth(
        port,
        "/v1beta/models/test-model:generateContent",
        GEMINI_REQ,
        Some("x-goog-api-key: wrong"),
    )
    .await;
    assert_eq!(response_status(&resp), 401, "{resp}");
}

/// /v1/messages takes the direct Anthropic → Gemini edge — hits the Gemini endpoint, never /chat/completions.
#[tokio::test]
async fn serve_messages_route_uses_direct_gemini_edge() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Gemini, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Google).await;

    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(json["type"], "message", "{resp}");
    assert_eq!(json["content"][0]["text"], "hello from gemini", "{resp}");
    assert!(fake.hit_count(Endpoint::Gemini) >= 1, "{:?}", fake.hits());
    assert_eq!(fake.hit_count(Endpoint::Chat), 0, "{:?}", fake.hits());
}

#[tokio::test]
async fn serve_messages_route_direct_gemini_edge_streams() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Gemini, Mode::OkSse)]);
    let port = start_serve(&fake, ProviderProtocol::Google).await;

    let resp = raw_post(port, "/v1/messages", MESSAGES_REQ_STREAM).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert!(resp.contains("text/event-stream"), "{resp}");
    assert!(resp.contains("event: message_start"), "{resp}");
    assert!(resp.contains("content_block_delta"), "{resp}");
    assert!(resp.contains("hello from gemini"), "{resp}");
    assert!(resp.contains("event: message_stop"), "{resp}");
    assert_eq!(fake.hit_count(Endpoint::Chat), 0, "{:?}", fake.hits());
}

#[tokio::test]
async fn serve_router_auto_switches_to_anthropic_upstream_and_pins() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    // Wrong initial guess: serve starts on OpenAI; upstream only speaks Anthropic.
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let json = response_json(&resp);
    assert_eq!(
        json["choices"][0]["message"]["content"], "hello from anthropic",
        "{resp}"
    );

    // Second request must hit /v1/messages directly — the pin holds.
    let chat_probes = fake.hit_count(Endpoint::Chat);
    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert_eq!(
        fake.hit_count(Endpoint::Chat),
        chat_probes,
        "pinned protocol must skip chat probes: {:?}",
        fake.hits()
    );
}

#[tokio::test]
async fn serve_router_surfaces_semantic_rejection_without_thrash() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::SemanticReject)]);
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 400, "{resp}");
    assert!(resp.contains("invalid_request_error"), "{resp}");
    // The structured 400 proves the protocol matches — exactly one upstream
    // probe, no fan-out across Anthropic/Google.
    assert_eq!(fake.hits(), vec![Endpoint::Chat], "{:?}", fake.hits());
}

#[tokio::test]
async fn serve_router_learns_routes_per_model() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let port = start_serve(&fake, ProviderProtocol::Openai).await;

    let resp = raw_post(port, "/v1/chat/completions", CHAT_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    let chat_probes_after_first = fake.hit_count(Endpoint::Chat);

    // A different model gets its own cascade (chat is probed again) instead of
    // inheriting the first model's pin — per-model slots, not a process pin.
    let other = r#"{"model":"other-model","messages":[{"role":"user","content":"hi"}]}"#;
    let resp = raw_post(port, "/v1/chat/completions", other).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    assert_eq!(
        fake.hit_count(Endpoint::Chat),
        chat_probes_after_first + 1,
        "second model must run its own probe: {:?}",
        fake.hits()
    );
}

// ── Loopback token enforcement ───────────────────────────────────────────

/// Like `raw_post` but with a caller-controlled auth header line
/// (`None` = no auth header at all).
async fn raw_post_with_auth(port: u16, path: &str, body: &str, auth_line: Option<&str>) -> String {
    let headers: Vec<&str> = auth_line.into_iter().collect();
    raw_request(port, "POST", path, &headers, Some(body)).await
}

#[tokio::test]
async fn claude_router_enforces_loopback_token() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = AnthropicToOpenAIRouter::new(AnthropicToOpenAIRouterConfig {
        target_base_url: fake.base_url(),
        target_api_key: "sk-test".to_string(),
        seed_routes: BTreeMap::new(),
        strip_cache_control: false,
        model_prefix: None,
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    })
    .with_auth_token("tok".to_string());
    let (port, _routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post_with_auth(port, "/v1/messages", ANTHROPIC_REQ, None).await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    let resp = raw_post_with_auth(
        port,
        "/v1/messages",
        ANTHROPIC_REQ,
        Some("Authorization: Bearer wrong"),
    )
    .await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    assert!(
        fake.hits().is_empty(),
        "unauthorized requests must never reach upstream: {:?}",
        fake.hits()
    );

    // `raw_post` sends `Authorization: Bearer tok` — the launch token.
    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    handle.abort();
}

#[tokio::test]
async fn codex_router_enforces_loopback_token() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = ResponsesToChatRouter::new(responses_router_config(fake.base_url()))
        .with_auth_token("tok".to_string());
    let (port, _routes, _learned, handle) = router.start_background().await.unwrap();

    let resp = raw_post_with_auth(port, "/v1/responses", RESPONSES_REQ, None).await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    assert!(
        fake.hits().is_empty(),
        "unauthorized requests must never reach upstream: {:?}",
        fake.hits()
    );

    let resp = raw_post(port, "/v1/responses", RESPONSES_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    handle.abort();
}

#[tokio::test]
async fn gemini_router_enforces_loopback_token_in_google_auth_forms() {
    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let router = GeminiRouter::new(GeminiRouterConfig {
        target_base_url: fake.base_url(),
        api_key: "sk-test".to_string(),
        upstream_protocol: ProviderProtocol::Google,
        forced_model: None,
        copilot_token_manager: None,
        requires_reasoning_content: false,
        max_tokens_cap: None,
        is_starter: false,
    })
    .with_auth_token("tok".to_string());
    let (port, _routes, _learned, handle) = router.start_background().await.unwrap();

    let path = "/v1beta/models/test-model:generateContent";
    let resp = raw_post_with_auth(port, path, GEMINI_REQ, None).await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    assert!(
        fake.hits().is_empty(),
        "unauthorized requests must never reach upstream: {:?}",
        fake.hits()
    );

    // Gemini CLI sends its key as `x-goog-api-key`.
    let resp = raw_post_with_auth(port, path, GEMINI_REQ, Some("x-goog-api-key: tok")).await;
    assert_eq!(response_status(&resp), 200, "{resp}");

    // Legacy Google clients send `?key=` instead of any header.
    let with_query = format!("{path}?key=tok");
    let resp = raw_post_with_auth(port, &with_query, GEMINI_REQ, None).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    handle.abort();
}

#[tokio::test]
async fn openrouter_anthropic_router_enforces_loopback_token() {
    use aivo::services::anthropic_router::{AnthropicRouter, AnthropicRouterConfig};

    no_proxy();
    let fake = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);
    let router = AnthropicRouter::new(AnthropicRouterConfig {
        upstream_base_url: fake.base_url(),
        upstream_api_key: "sk-test".to_string(),
        is_starter: false,
    })
    .with_auth_token("tok".to_string());
    let (port, handle) = router.start_background().await.unwrap();

    let resp = raw_post_with_auth(port, "/v1/messages", ANTHROPIC_REQ, None).await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    assert!(
        fake.hits().is_empty(),
        "unauthorized requests must never reach upstream: {:?}",
        fake.hits()
    );

    let resp = raw_post(port, "/v1/messages", ANTHROPIC_REQ).await;
    assert_eq!(response_status(&resp), 200, "{resp}");
    handle.abort();
}

#[tokio::test]
async fn copilot_router_enforces_loopback_token() {
    use aivo::services::copilot_router::{CopilotRouter, CopilotRouterConfig};

    no_proxy();
    // No 200-path here: a valid request would call the real Copilot API.
    let router = CopilotRouter::new(CopilotRouterConfig {
        github_token: "gho_test".to_string(),
    })
    .with_auth_token("tok".to_string());
    let (port, handle) = router.start_background().await.unwrap();

    let resp = raw_post_with_auth(port, "/v1/messages", ANTHROPIC_REQ, None).await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    let resp = raw_post_with_auth(
        port,
        "/v1/messages",
        ANTHROPIC_REQ,
        Some("x-api-key: wrong"),
    )
    .await;
    assert_eq!(response_status(&resp), 401, "{resp}");
    handle.abort();
}

// ── per-tier provider/key routing (aivo claude tiers) ────────────────────

/// A `/v1/messages` tier model reaches its own provider/key; other models stay
/// on the base upstream.
#[tokio::test]
async fn serve_router_dispatches_per_model_to_tier_upstream() {
    no_proxy();
    let base = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let tier = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);

    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    std::mem::forget(tmp);

    let router = ServeRouter::new(
        serve_config(base.base_url(), ProviderProtocol::Openai),
        test_key(&base.base_url()),
        log_store,
    )
    .with_model_upstreams(vec![("model-tier".to_string(), test_key(&tier.base_url()))]);
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();

    let body = |model: &str| {
        format!(
            r#"{{"model":"{model}","max_tokens":128,"messages":[{{"role":"user","content":"hi"}}]}}"#
        )
    };

    let r = raw_post(port, "/v1/messages", &body("model-main")).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(base.hit_count(Endpoint::Chat), 1, "base model hits base");
    assert_eq!(
        tier.hit_count(Endpoint::Chat),
        0,
        "base model must not hit tier"
    );

    let r = raw_post(port, "/v1/messages", &body("model-tier")).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(tier.hit_count(Endpoint::Chat), 1, "tier model hits tier");
    assert_eq!(
        base.hit_count(Endpoint::Chat),
        1,
        "tier model must not add a base hit"
    );

    let r = raw_post(port, "/v1/messages", &body("model-tier[1m]")).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(
        tier.hit_count(Endpoint::Chat),
        2,
        "suffixed tier model hits tier"
    );
    assert_eq!(
        base.hit_count(Endpoint::Chat),
        1,
        "suffixed tier model must not hit base"
    );
}

// ── Claude-subscription OAuth passthrough ────────────────────────────────

/// Must equal the `token` inside [`CLAUDE_OAUTH_CREDS`].
const CLAUDE_OAUTH_BEARER: &str = "sk-ant-oat01-TEST";
const CLAUDE_OAUTH_CREDS: &str =
    r#"{"token":"sk-ant-oat01-TEST","created_at":"2026-01-01T00:00:00Z"}"#;

fn claude_oauth_key() -> ApiKey {
    let mut key = test_key("claude-oauth");
    key.key = Zeroizing::new(CLAUDE_OAUTH_CREDS.to_string());
    key
}

/// Config for a subscription main upstream on a fake at `upstream_port`;
/// the loopback gate expects the client's own OAuth bearer.
fn claude_oauth_serve_config(upstream_port: u16) -> ServeRouterConfig {
    let mut config = serve_config(
        format!("http://127.0.0.1:{upstream_port}"),
        ProviderProtocol::Anthropic,
    );
    config.is_claude_native_oauth = true;
    config.upstream_api_key = CLAUDE_OAUTH_CREDS.to_string();
    config.auth_token = Some(CLAUDE_OAUTH_BEARER.to_string());
    config
}

fn anthropic_body(model: &str) -> String {
    format!(
        r#"{{"model":"{model}","max_tokens":128,"messages":[{{"role":"user","content":"hi"}}]}}"#
    )
}

#[tokio::test]
async fn serve_router_claude_oauth_main_passthrough() {
    no_proxy();
    let anthropic = spawn_fake(&[(Endpoint::Messages, Mode::Ok), (Endpoint::Other, Mode::Ok)]);
    let tier = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);

    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    std::mem::forget(tmp);

    let router = ServeRouter::new(
        claude_oauth_serve_config(anthropic.port),
        claude_oauth_key(),
        log_store,
    )
    .with_model_upstreams(vec![("model-tier".to_string(), test_key(&tier.base_url()))]);
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();

    let oauth_auth = format!("Authorization: Bearer {CLAUDE_OAUTH_BEARER}");
    let oauth_auth = oauth_auth.as_str();

    // Wrong bearer → 401 before any upstream traffic.
    let r = raw_request(
        port,
        "POST",
        "/v1/messages",
        &["Authorization: Bearer wrong"],
        Some(&anthropic_body("model-main")),
    )
    .await;
    assert_eq!(response_status(&r), 401, "{r}");
    assert_eq!(anthropic.hit_count(Endpoint::Messages), 0);

    // Subscription main: forwarded verbatim with merged oauth beta.
    let r = raw_request(
        port,
        "POST",
        "/v1/messages?beta=true",
        &[
            oauth_auth,
            "anthropic-beta: claude-code-20250219",
            "User-Agent: claude-cli/2.1.205 (external, cli)",
        ],
        Some(&anthropic_body("model-main")),
    )
    .await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert!(r.contains("hello from anthropic"), "{r}");
    let head = anthropic.heads().pop().unwrap();
    let head_lower = head.to_lowercase();
    assert!(
        head.lines()
            .next()
            .unwrap()
            .contains("/v1/messages?beta=true"),
        "query preserved: {head}"
    );
    assert!(
        head.contains(CLAUDE_OAUTH_BEARER),
        "bearer forwarded: {head}"
    );
    assert!(
        !head_lower.contains("x-api-key"),
        "no api key header: {head}"
    );
    assert!(
        head_lower.contains("oauth-2025-04-20"),
        "oauth beta merged: {head}"
    );
    assert!(
        head_lower.contains("claude-code-20250219"),
        "inbound betas kept: {head}"
    );
    assert!(
        head_lower.contains("claude-cli/2.1.205"),
        "client UA forwarded: {head}"
    );

    // Tier model dispatches to its own provider with its own auth.
    let r = raw_request(
        port,
        "POST",
        "/v1/messages",
        &[oauth_auth],
        Some(&anthropic_body("model-tier")),
    )
    .await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(tier.hit_count(Endpoint::Chat), 1, "tier model hits tier");
    let tier_head = tier.heads().pop().unwrap();
    assert!(
        !tier_head.contains(CLAUDE_OAUTH_BEARER),
        "subscription bearer must not leak to tier: {tier_head}"
    );
    assert!(
        tier_head.contains("sk-test"),
        "tier uses its own key: {tier_head}"
    );

    // /v1/models and count_tokens forward to the native backend.
    let r = raw_request(port, "GET", "/v1/models", &[oauth_auth], None).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(anthropic.hit_count(Endpoint::Other), 1, "models forwarded");
    let r = raw_request(
        port,
        "POST",
        "/v1/messages/count_tokens",
        &[oauth_auth],
        Some(&anthropic_body("model-main")),
    )
    .await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(
        anthropic.hit_count(Endpoint::Messages),
        2,
        "count_tokens forwarded"
    );

    // count_tokens for a non-subscription tier keeps the historical 404.
    let r = raw_request(
        port,
        "POST",
        "/v1/messages/count_tokens",
        &[oauth_auth],
        Some(&anthropic_body("model-tier")),
    )
    .await;
    assert_eq!(response_status(&r), 404, "{r}");
}

#[tokio::test]
async fn serve_router_claude_oauth_main_streams_sse_passthrough() {
    no_proxy();
    let anthropic = spawn_fake(&[(Endpoint::Messages, Mode::OkSse)]);

    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    std::mem::forget(tmp);

    let router = ServeRouter::new(
        claude_oauth_serve_config(anthropic.port),
        claude_oauth_key(),
        log_store,
    );
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();

    let oauth_auth = format!("Authorization: Bearer {CLAUDE_OAUTH_BEARER}");
    let r = raw_request(
        port,
        "POST",
        "/v1/messages?beta=true",
        &[oauth_auth.as_str()],
        Some(
            r#"{"model":"claude-fable-5","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}"#,
        ),
    )
    .await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert!(r.contains("text/event-stream"), "{r}");
    assert!(r.contains("event: message_start"), "{r}");
    assert!(r.contains("hello from anthropic"), "{r}");
}

#[tokio::test]
async fn serve_router_claude_oauth_tier_passthrough() {
    no_proxy();
    let base = spawn_fake(&[(Endpoint::Chat, Mode::Ok)]);
    let sub = spawn_fake(&[(Endpoint::Messages, Mode::Ok)]);

    // Point the subscription upstream at the fake; only this test touches the var.
    unsafe {
        std::env::set_var(
            "AIVO_CLAUDE_OAUTH_UPSTREAM",
            format!("http://127.0.0.1:{}", sub.port),
        )
    };

    let tmp = tempfile::tempdir().unwrap();
    let log_store = LogStore::new(tmp.path().to_path_buf());
    std::mem::forget(tmp);

    let router = ServeRouter::new(
        serve_config(base.base_url(), ProviderProtocol::Openai),
        test_key(&base.base_url()),
        log_store,
    )
    .with_model_upstreams(vec![("claude-fable-5".to_string(), claude_oauth_key())]);
    let (_handle, _shutdown, port) = router
        .start_background_with_addr("127.0.0.1", 0)
        .await
        .unwrap();

    // raw_post sends no anthropic-beta — the router must add the oauth beta itself.
    let r = raw_post(port, "/v1/messages", &anthropic_body("claude-fable-5")).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert!(r.contains("hello from anthropic"), "{r}");
    assert_eq!(
        sub.hit_count(Endpoint::Messages),
        1,
        "tier hits subscription"
    );
    let head = sub.heads().pop().unwrap();
    let head_lower = head.to_lowercase();
    assert!(
        head.contains(CLAUDE_OAUTH_BEARER),
        "stored bearer injected: {head}"
    );
    assert!(
        head_lower.contains("oauth-2025-04-20"),
        "oauth beta added: {head}"
    );
    assert!(!head_lower.contains("x-api-key"), "{head}");

    // Base model unaffected.
    let r = raw_post(port, "/v1/messages", &anthropic_body("model-main")).await;
    assert_eq!(response_status(&r), 200, "{r}");
    assert_eq!(base.hit_count(Endpoint::Chat), 1);

    unsafe { std::env::remove_var("AIVO_CLAUDE_OAUTH_UPSTREAM") };
}
