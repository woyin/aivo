//! End-to-end cascade tests: each router against a fake provider that speaks
//! only one protocol, asserting convergence, learned routes, and that failed
//! cascades surface the real upstream error instead of thrashing.

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
    /// 400 with a structured error envelope — a semantic rejection that must
    /// bail. Unlisted endpoints 404 with `{"error":"Not found"}`, a
    /// path-missing mismatch the cascade may walk past.
    SemanticReject,
}

#[derive(Clone)]
struct FakeProvider {
    port: u16,
    hits: Arc<Mutex<Vec<Endpoint>>>,
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

/// Spawns a blocking fake provider; `modes` maps endpoints to behaviors,
/// anything unlisted 404s.
fn spawn_fake(modes: &[(Endpoint, Mode)]) -> FakeProvider {
    let modes: HashMap<Endpoint, Mode> = modes.iter().copied().collect();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let hits: Arc<Mutex<Vec<Endpoint>>> = Arc::new(Mutex::new(Vec::new()));
    let hits_writer = hits.clone();

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            // Read headers.
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => request.extend_from_slice(&buf[..n]),
                    Err(_) => break,
                }
            }
            let head = String::from_utf8_lossy(&request).into_owned();
            // Drain the body so the client never sees a closed pipe mid-write.
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

            let path = head
                .lines()
                .next()
                .and_then(|l| l.split_whitespace().nth(1))
                .unwrap_or("/")
                .to_string();
            let endpoint = classify(&path);
            hits_writer.lock().unwrap().push(endpoint);

            let (status, reason, body) = match modes.get(&endpoint) {
                Some(Mode::Ok) => (200, "OK", success_body(endpoint)),
                Some(Mode::SemanticReject) => (
                    400,
                    "Bad Request",
                    r#"{"error":{"type":"invalid_request_error","message":"bad request body"}}"#
                        .to_string(),
                ),
                None => (404, "Not Found", r#"{"error":"Not found"}"#.to_string()),
            };
            let head = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            let _ = stream.flush();
        }
    });

    FakeProvider { port, hits }
}

// ── Client helpers ───────────────────────────────────────────────────────

async fn raw_post(port: u16, path: &str, body: &str) -> String {
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
    let _ = stream.read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
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
        is_openrouter: false,
        is_starter: false,
        is_joycode: false,
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let auth = auth_line.map(|l| format!("{l}\r\n")).unwrap_or_default();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).await.unwrap();
    let _ = stream.shutdown().await;
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf).await;
    String::from_utf8_lossy(&buf).into_owned()
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
