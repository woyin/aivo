//! Test module for `cursor_bridge`. Pulled out of the original
//! `cursor_model_router.rs` test block when the module was split into
//! per-protocol submodules; the test helpers and per-protocol assertions
//! continue to share a single `use super::*` surface.

#![cfg(test)]

use super::anthropic::*;
use super::gemini::*;
use super::openai_chat::*;
use super::responses::*;
use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use zeroize::Zeroizing;

fn fake_key() -> ApiKey {
    ApiKey {
        id: "cursor-test".to_string(),
        name: "cursor".to_string(),
        base_url: CURSOR_ACP_SENTINEL.to_string(),
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
        key: Zeroizing::new("cursor-login".to_string()),
        created_at: "2026-01-01T00:00:00Z".to_string(),
    }
}

fn state_with_models(models: Vec<&str>) -> Arc<RouterState> {
    Arc::new(RouterState {
        config: CursorRouterConfig {
            key: fake_key(),
            workspace_cwd: "/tmp".to_string(),
            // Tests pin the in-memory cache directly via `cached_models`;
            // skip disk plumbing so they don't touch `~/.config/aivo/`.
            models_cache: None,
            prewarm_count: 0,
            mcp_prewarm_id_style: None,
            expected_token: None,
        },
        cached_models: Mutex::new(Some(models.into_iter().map(String::from).collect())),
        mcp_bridge: McpBridge::for_tests(),
        pool: Mutex::new(Vec::new()),
        prewarming: AtomicUsize::new(0),
        mcp_prewarmed: Arc::new(Mutex::new(McpPrewarmSlot::new())),
    })
}

async fn round_trip(state: Arc<RouterState>, request: &str) -> String {
    let (listener, port) = bind_local_listener().await.unwrap();
    let handle = tokio::spawn(run_router(listener, state));

    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    // Half-close so the server stops trying to read more from this
    // direction; on Windows that avoids the abortive close (WSAECONNRESET)
    // that surfaces when the server drops the socket with unread data
    // still in its receive buffer.
    let _ = stream.shutdown().await;
    let mut buf = Vec::new();
    // Tolerate ConnectionReset on Windows: any bytes we did receive
    // before the RST are still the server's response.
    let _ = stream.read_to_end(&mut buf).await;
    handle.abort();
    String::from_utf8(buf).unwrap()
}

#[tokio::test]
async fn get_v1_models_returns_openai_shaped_list() {
    let state = state_with_models(vec!["composer-2.5", "claude-sonnet-4-6"]);
    let response = round_trip(
        state,
        "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(response.contains("200 OK"));
    let body = response.split("\r\n\r\n").nth(1).unwrap();
    let parsed: Value = serde_json::from_str(body).unwrap();
    assert_eq!(parsed["object"], "list");
    let data = parsed["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["id"], "composer-2.5");
    assert_eq!(data[0]["owned_by"], "cursor");
}

#[tokio::test]
async fn options_preflight_returns_cors() {
    let state = state_with_models(vec![]);
    let response = round_trip(
        state,
        "OPTIONS /v1/models HTTP/1.1\r\nOrigin: x\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(response.contains("204 No Content"));
    assert!(response.contains("Access-Control-Allow-Origin: *"));
}

#[test]
fn models_response_body_satisfies_both_openai_and_codex_consumers() {
    // Regression: codex 0.132+ parses /models as `ModelsResponse { models }`
    // with each entry strictly-typed (codex-rs/protocol/src/openai_models.rs::
    // ModelInfo). The OpenAI-style `{"object":"list","data":[...]}` shape
    // alone makes codex spam `failed to decode models response: missing
    // field "models"`/`"slug"`/etc. on every interaction. We emit both
    // shapes as twin arrays.
    let body = build_models_response_body(&["composer-2.5".to_string(), "auto".to_string()]);

    // OpenAI side: object="list", data=[{id, object, owned_by}, ...]
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().expect("data must be an array");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["id"], "composer-2.5");
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["owned_by"], "cursor");

    // Codex side: every field codex considers required (no `#[serde(default)]`,
    // no `Option` with default attribute) must be present. Verify the most
    // load-bearing ones; missing any of these is what produced the
    // production error.
    let models = body["models"].as_array().expect("models must be an array");
    assert_eq!(models.len(), 2);
    let first = &models[0];
    for required in [
        "slug",
        "display_name",
        "description",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "availability_nux",
        "upgrade",
        "base_instructions",
        "supports_reasoning_summaries",
        "support_verbosity",
        "default_verbosity",
        "apply_patch_tool_type",
        "truncation_policy",
        "supports_parallel_tool_calls",
        "experimental_supported_tools",
    ] {
        assert!(
            first.get(required).is_some(),
            "codex requires field `{required}`; codex models-manager will refuse the response otherwise: {first:?}"
        );
    }
    assert_eq!(first["slug"], "composer-2.5");
    assert_eq!(first["truncation_policy"]["mode"], "tokens");
}

#[tokio::test]
async fn cached_models_serves_from_disk_cache_without_spawning_cursor_agent() {
    // A warm entry under `cursor_models_cache_identity(&key)` (the same key
    // `aivo models cursor` and the picker write) must be returned without
    // shelling out to `cursor-agent`. The test would otherwise hang or
    // error trying to spawn cursor-agent in the sandbox.
    let dir = tempfile::tempdir().unwrap();
    let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
    let key = fake_key();
    let cache_key = cursor_acp::cursor_models_cache_identity(&key);
    cache
        .set(
            &cache_key,
            vec!["composer-2.5".to_string(), "claude-sonnet-4-6".to_string()],
        )
        .await;

    let state = Arc::new(RouterState {
        config: CursorRouterConfig {
            key,
            workspace_cwd: "/tmp".to_string(),
            models_cache: Some(cache),
            prewarm_count: 0,
            mcp_prewarm_id_style: None,
            expected_token: None,
        },
        // In-memory cache deliberately empty so the lookup falls through
        // to the disk-backed branch.
        cached_models: Mutex::new(None),
        mcp_bridge: McpBridge::for_tests(),
        pool: Mutex::new(Vec::new()),
        prewarming: AtomicUsize::new(0),
        mcp_prewarmed: Arc::new(Mutex::new(McpPrewarmSlot::new())),
    });

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        round_trip(
            state,
            "GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        ),
    )
    .await
    .expect("disk-cache path must not spawn cursor-agent");
    assert!(response.contains("200 OK"));
    let body = response.split("\r\n\r\n").nth(1).unwrap();
    let parsed: Value = serde_json::from_str(body).unwrap();
    let data = parsed["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["id"], "composer-2.5");
}

#[tokio::test]
async fn acquire_session_slot_grows_pool_when_existing_slot_is_busy() {
    let state = state_with_models(vec![]);
    // Hold slot 0 by acquiring its guard and not dropping it. The next
    // acquire should append a new slot instead of blocking.
    let busy_guard = acquire_session_slot(&state).await;
    assert_eq!(state.pool.lock().await.len(), 1);

    let second = acquire_session_slot(&state).await;
    assert_eq!(state.pool.lock().await.len(), 2);
    drop(second);

    // Re-acquiring without releasing slot 0 should reuse slot 1, not grow.
    let reused = acquire_session_slot(&state).await;
    assert_eq!(state.pool.lock().await.len(), 2);
    drop(reused);
    drop(busy_guard);
}

#[tokio::test]
async fn acquire_session_slot_caps_pool_and_then_waits_on_first() {
    let state = state_with_models(vec![]);
    let mut guards = Vec::new();
    for _ in 0..MAX_POOL_SESSIONS {
        guards.push(acquire_session_slot(&state).await);
    }
    assert_eq!(state.pool.lock().await.len(), MAX_POOL_SESSIONS);

    // Cap reached. A further acquire should block. Race a release of slot 0
    // against the queued acquire and confirm it unblocks (rather than
    // growing past the cap).
    let state_clone = state.clone();
    let pending = tokio::spawn(async move { acquire_session_slot(&state_clone).await });
    // Yield a couple of times to let the spawned task park on the mutex.
    for _ in 0..4 {
        tokio::task::yield_now().await;
    }
    assert!(!pending.is_finished(), "pending acquire must block at cap");
    drop(guards.remove(0));
    let _granted = tokio::time::timeout(std::time::Duration::from_secs(1), pending)
        .await
        .expect("pending acquire should unblock once slot 0 frees")
        .expect("spawn join");
    assert_eq!(
        state.pool.lock().await.len(),
        MAX_POOL_SESSIONS,
        "pool must not grow past cap when waiting on existing slot"
    );
}

#[tokio::test]
async fn head_and_get_root_return_200_for_probes() {
    let state = state_with_models(vec![]);
    for verb in ["HEAD", "GET"] {
        let response = round_trip(
            state.clone(),
            &format!("{verb} / HTTP/1.1\r\nConnection: close\r\n\r\n"),
        )
        .await;
        assert!(
            response.contains("200 OK"),
            "{verb} / must succeed; got: {response}"
        );
    }
}

#[tokio::test]
async fn unknown_path_returns_404_with_hint() {
    let state = state_with_models(vec![]);
    let response = round_trip(state, "GET /nope HTTP/1.1\r\nConnection: close\r\n\r\n").await;
    assert!(response.contains("404"));
    assert!(response.contains("/nope"));
    assert!(response.contains("/v1/responses"));
}

#[test]
fn is_client_disconnect_detects_broken_pipe_through_anyhow_context() {
    let raw = std::io::Error::from(std::io::ErrorKind::BrokenPipe);
    let wrapped: anyhow::Error = anyhow::Error::new(raw).context("cursor-agent session/prompt");
    assert!(is_client_disconnect(&wrapped));
    assert_eq!(status_for_handler_error(&wrapped), 499);
}

#[test]
fn is_client_disconnect_detects_connection_reset_and_aborted() {
    for kind in [
        std::io::ErrorKind::ConnectionReset,
        std::io::ErrorKind::ConnectionAborted,
    ] {
        let err: anyhow::Error = anyhow::Error::new(std::io::Error::from(kind));
        assert!(
            is_client_disconnect(&err),
            "kind {kind:?} should map to 499"
        );
    }
}

#[test]
fn is_client_disconnect_returns_false_for_unrelated_io_errors() {
    let err: anyhow::Error =
        anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied));
    assert!(!is_client_disconnect(&err));
    assert_eq!(status_for_handler_error(&err), 502);
}

#[test]
fn is_client_disconnect_returns_false_for_non_io_errors() {
    let err = anyhow::anyhow!("upstream rejected request");
    assert!(!is_client_disconnect(&err));
    assert_eq!(status_for_handler_error(&err), 502);
}

#[test]
fn anthropic_thinking_delta_has_thinking_delta_payload() {
    let frame = anthropic_thinking_delta(2, "reasoning text");
    assert!(frame.starts_with("event: content_block_delta\n"));
    let data = frame
        .split_once("data: ")
        .and_then(|(_, s)| s.split("\n\n").next())
        .unwrap();
    let parsed: Value = serde_json::from_str(data).unwrap();
    assert_eq!(parsed["index"], 2);
    assert_eq!(parsed["delta"]["type"], "thinking_delta");
    assert_eq!(parsed["delta"]["thinking"], "reasoning text");
}

#[test]
fn anthropic_text_delta_has_text_delta_payload() {
    let frame = anthropic_text_delta(0, "hello");
    let data = frame
        .split_once("data: ")
        .and_then(|(_, s)| s.split("\n\n").next())
        .unwrap();
    let parsed: Value = serde_json::from_str(data).unwrap();
    assert_eq!(parsed["index"], 0);
    assert_eq!(parsed["delta"]["type"], "text_delta");
    assert_eq!(parsed["delta"]["text"], "hello");
}

#[test]
fn anthropic_content_block_start_emits_correct_inner_shape() {
    let text = anthropic_content_block_start(0, AnthropicBlockKind::Text);
    assert!(text.contains("\"type\":\"text\""));
    assert!(text.contains("\"text\":\"\""));
    let thinking = anthropic_content_block_start(1, AnthropicBlockKind::Thinking);
    assert!(thinking.contains("\"type\":\"thinking\""));
    assert!(thinking.contains("\"thinking\":\"\""));
}

#[test]
fn extract_tool_call_marker_formats_known_event_shapes() {
    let pending = json!({
        "update": {"sessionUpdate": "tool_call", "kind": "search",
                   "status": "pending", "title": "Find sum.js"},
    });
    let s = extract_tool_call_marker(&pending).unwrap();
    assert!(s.contains("pending"));
    assert!(s.contains("Find sum.js"));

    let no_title = json!({
        "update": {"sessionUpdate": "tool_call_update",
                   "kind": "read", "status": "in_progress"},
    });
    let s = extract_tool_call_marker(&no_title).unwrap();
    assert!(s.contains("read"));
    assert!(s.contains("in_progress"));

    // Non-tool events are ignored.
    let irrelevant = json!({
        "update": {"sessionUpdate": "session_info_update"},
    });
    assert!(extract_tool_call_marker(&irrelevant).is_none());
}

#[test]
fn extract_tool_call_marker_appends_summary_from_raw_output() {
    // Bare completion (no title, no kind) — must still surface.
    let search_done = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "toolCallId": "tool_5c71212",
            "rawOutput": {"totalMatches": 19, "truncated": false},
        },
    });
    let s = extract_tool_call_marker(&search_done).unwrap();
    assert!(s.contains("completed"), "got: {s:?}");
    assert!(s.contains("19 matches"), "got: {s:?}");

    let one_match = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "rawOutput": {"totalMatches": 1, "truncated": false},
        },
    });
    let s = extract_tool_call_marker(&one_match).unwrap();
    assert!(s.contains("1 match\n"), "expected singular: {s:?}");
    assert!(!s.contains("1 matches"), "expected singular: {s:?}");

    let truncated = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "rawOutput": {"totalMatches": 200, "truncated": true},
        },
    });
    let s = extract_tool_call_marker(&truncated).unwrap();
    assert!(s.contains("200 matches"));
    assert!(s.contains("truncated"));

    let exec_done = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "rawOutput": {"exitCode": 1, "stderr": "...", "stdout": "..."},
        },
    });
    assert!(
        extract_tool_call_marker(&exec_done)
            .unwrap()
            .contains("exit 1")
    );

    let read_done = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "rawOutput": {"content": "a\nb\nc\n"},
        },
    });
    assert!(
        extract_tool_call_marker(&read_done)
            .unwrap()
            .contains("3 lines")
    );

    // Title + rawOutput together: suffix appends without dropping title.
    let start_with_summary = json!({
        "update": {
            "sessionUpdate": "tool_call",
            "kind": "search",
            "status": "pending",
            "title": "grep",
            "rawOutput": {"totalMatches": 5, "truncated": false},
        },
    });
    let s = extract_tool_call_marker(&start_with_summary).unwrap();
    assert!(s.contains("[pending] grep"));
    assert!(s.contains("5 matches"));
}

#[test]
fn extract_tool_call_marker_surfaces_edit_diff_completions() {
    // Edit/write tools emit `update.content = [{type:"diff",...}]` instead
    // of populating `rawOutput`. Before the fix, completed edit tools fell
    // through the match arms and returned None — users saw `[pending] Edit
    // File` then silence. Verified against cursor-agent 2026.05.20-2b5dd59
    // ACP capture in /tmp/aivo-cursor-acp-probe/events.jsonl.
    let edit_done = json!({
        "update": {
            "sessionUpdate": "tool_call_update",
            "status": "completed",
            "toolCallId": "tool_x",
            "content": [{
                "type": "diff",
                "path": "/tmp/scratch.txt",
                "oldText": "-- /dev/null\n",
                "newText": "++ b//tmp/scratch.txt\nhello\nworld",
            }],
        },
    });
    let s = extract_tool_call_marker(&edit_done).expect("edit completion should produce a marker");
    assert!(s.contains("completed"), "got: {s:?}");
    assert!(
        s.contains("scratch.txt"),
        "expected basename in marker: {s:?}"
    );
    assert!(s.contains("+3"), "expected newText line count: {s:?}");
    assert!(s.contains("-1"), "expected oldText line count: {s:?}");
}

#[tokio::test]
async fn acquire_session_slot_waits_for_prewarm_instead_of_expanding() {
    // Regression: when a prewarm task is mid-`CursorAcpSession::open` it
    // holds the slot's lock. The previous `try_lock_owned` path
    // immediately gave up and expanded the pool, spawning a second
    // cursor-agent that the prewarmed one would duplicate. With the fix,
    // an arriving request must wait on the prewarmed slot when
    // `prewarming > 0`, not append a new slot.
    let state = state_with_models(vec![]);
    let slot: SessionSlot = Arc::new(Mutex::new(None));
    state.pool.lock().await.push(slot.clone());
    // Simulate the prewarm task holding the slot's lock and the
    // prewarming counter being non-zero.
    let held = slot.clone().lock_owned().await;
    state.prewarming.fetch_add(1, Ordering::SeqCst);

    let acquire_state = state.clone();
    let mut acquire = Box::pin(async move { acquire_session_slot(&acquire_state).await });

    // While the prewarm slot is held, acquire must NOT complete (it
    // should be parked on the slot's lock) AND must NOT expand the
    // pool.
    let parked = tokio::time::timeout(std::time::Duration::from_millis(50), &mut acquire)
        .await
        .is_err();
    assert!(
        parked,
        "acquire should block, not expand, while prewarm holds the slot"
    );
    assert_eq!(
        state.pool.lock().await.len(),
        1,
        "pool must stay at 1 slot — expanding would orphan a cursor-agent"
    );

    // Release the prewarm's hold; acquire should now resolve with the
    // existing slot.
    state.prewarming.fetch_sub(1, Ordering::SeqCst);
    drop(held);
    let guard = tokio::time::timeout(std::time::Duration::from_secs(1), acquire)
        .await
        .expect("acquire should resolve once prewarm releases");
    assert!(guard.is_none(), "slot's inner session is still empty");
    assert_eq!(state.pool.lock().await.len(), 1);
}

#[tokio::test]
async fn acquire_session_slot_picks_whichever_prewarm_slot_frees_first() {
    // With two prewarmed slots both locked, two concurrent acquirers
    // must each land on a separate slot. Pinning to existing[0] would
    // deadlock the second acquirer if slot[1] freed first.
    let state = state_with_models(vec![]);
    let slot_a: SessionSlot = Arc::new(Mutex::new(None));
    let slot_b: SessionSlot = Arc::new(Mutex::new(None));
    {
        let mut pool = state.pool.lock().await;
        pool.push(slot_a.clone());
        pool.push(slot_b.clone());
    }
    let held_a = slot_a.clone().lock_owned().await;
    let held_b = slot_b.clone().lock_owned().await;
    state.prewarming.fetch_add(2, Ordering::SeqCst);

    let s1 = state.clone();
    let s2 = state.clone();
    let acquire_1 = tokio::spawn(async move { acquire_session_slot(&s1).await });
    let acquire_2 = tokio::spawn(async move { acquire_session_slot(&s2).await });

    // Free slot B first: one acquirer wakes.
    drop(held_b);
    // Then slot A: the other acquirer wakes.
    drop(held_a);
    state.prewarming.store(0, Ordering::SeqCst);

    let r1 = tokio::time::timeout(std::time::Duration::from_secs(1), acquire_1).await;
    let r2 = tokio::time::timeout(std::time::Duration::from_secs(1), acquire_2).await;
    assert!(r1.is_ok() && r2.is_ok(), "both acquirers must resolve");
    assert_eq!(
        state.pool.lock().await.len(),
        2,
        "pool size unchanged — no expansion happened"
    );
}

#[tokio::test]
async fn spawn_prewarm_increments_counter_synchronously() {
    // The counter must reflect pending prewarms BEFORE the spawned
    // tasks run, so an early request that arrives in the gap between
    // `start_background` returning and the prewarm task starting
    // observes `prewarming > 0` and picks the wait-on-slot branch.
    let state = state_with_models(vec![]);
    spawn_prewarm(state.clone(), 2);
    // No `.await` since the increment is synchronous.
    assert_eq!(state.prewarming.load(Ordering::SeqCst), 2);
    // Drain any background work and let the PrewarmCounterGuard drop
    // so we don't leave state behind for sibling tests.
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn spawn_prewarm_reserves_requested_number_of_slots() {
    // Open `count` slot Arcs in the pool before any HTTP request lands,
    // so two paired requests both find a slot already in the pool (even
    // if the underlying CursorAcpSession::open is still in flight). We
    // can't run the real open() in unit tests — it would spawn a real
    // cursor-agent binary — so the assertion is about pool *shape*:
    // requesting 2 prewarms must yield at least 2 slot Arcs.
    let state = state_with_models(vec![]);
    spawn_prewarm(state.clone(), 2);
    // Yield enough that the prewarm tasks have time to register their
    // slots in the outer pool Mutex (the slot reservation is the very
    // first await they hit).
    for _ in 0..16 {
        tokio::task::yield_now().await;
    }
    let n = state.pool.lock().await.len();
    assert!(
        n >= 2,
        "prewarm should reserve >= 2 slot Arcs in the pool, saw {n}"
    );
}

#[test]
fn anthropic_block_state_defaults_to_no_block_and_index_zero() {
    let s = AnthropicBlockState::default();
    assert!(s.current.is_none());
    assert_eq!(s.next_index, 0);
    // Without a current block the helper exposes index 0 as a safe
    // default so deltas have something to address before the first
    // ensure_kind. The dispatcher always calls ensure_kind first.
    assert_eq!(s.index(), 0);
}

#[test]
fn sse_keepalive_uses_comment_prefix_per_spec() {
    // The SSE spec says lines starting with `:` are comments — clients
    // ignore them but their idle-timer logic still counts them as
    // "the connection is alive". Two trailing newlines terminate the
    // message frame.
    assert!(SSE_KEEPALIVE.starts_with(':'));
    assert!(SSE_KEEPALIVE.ends_with("\n\n"));
    assert!(!SSE_KEEPALIVE.contains("data:"));
}

#[test]
fn strip_v1_prefix_collapses_only_real_v1_segments() {
    assert_eq!(strip_v1_prefix("/v1/chat/completions"), "/chat/completions");
    assert_eq!(strip_v1_prefix("/v1/models"), "/models");
    assert_eq!(strip_v1_prefix("/v1/"), "/");
    // Bare `/v1` collapses to empty so the root probe handler catches it.
    assert_eq!(strip_v1_prefix("/v1"), "");
    // Don't accidentally re-interpret `/v1bogus` as `/bogus`.
    assert_eq!(strip_v1_prefix("/v1bogus"), "/v1bogus");
    // Anything without the prefix passes through unchanged.
    assert_eq!(strip_v1_prefix("/chat/completions"), "/chat/completions");
    assert_eq!(strip_v1_prefix("/"), "/");
    assert_eq!(strip_v1_prefix(""), "");
}

#[tokio::test]
async fn unversioned_post_paths_route_to_the_same_handlers() {
    // The handlers themselves need an ACP session, which we can't spawn
    // in unit tests. Assert that the *router* dispatches the unversioned
    // form to the right handler by checking the body parsing path: a body
    // missing `messages` should yield "messages array is required" (the
    // OpenAI handler's error) for both /v1/chat/completions and
    // /chat/completions, *not* the router 404 message.
    let state = state_with_models(vec![]);
    let body = r#"{"model": "composer-2.5"}"#;
    for path in [
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/messages",
        "/messages",
        "/v1/responses",
        "/responses",
    ] {
        let request = format!(
            "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
            len = body.len()
        );
        let response = round_trip(state.clone(), &request).await;
        assert!(
            !response.contains("cursor_model_router_not_found"),
            "{path} unexpectedly routed to the 404 handler: {response}"
        );
    }
}

#[test]
fn reduce_openai_request_joins_role_labels_and_renders_tool_results() {
    let body = json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "read_file", "content": "file body"},
            {"role": "user", "content": "Explain X."},
        ],
    });
    let prompt = reduce_openai_request_to_prompt(&body);
    assert_eq!(
        prompt,
        "System: You are helpful.\n\nUser: Hi\n\n[Tool result for read_file]\nfile body\n\nUser: Explain X."
    );
}

#[test]
fn reduce_openai_request_emits_available_tools_header_and_tool_calls() {
    let body = json!({
        "tools": [
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file from disk.",
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "  Write\n  content  to disk.",
            }},
        ],
        "messages": [
            {"role": "user", "content": "Update the readme."},
            {"role": "assistant", "content": "Reading first.", "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}",
                }},
            ]},
            {"role": "tool", "name": "read_file", "content": "# Title"},
        ],
    });
    let prompt = reduce_openai_request_to_prompt(&body);
    assert!(prompt.starts_with(
        "Available tools:\n- read_file: Read a file from disk.\n- write_file: Write content to disk."
    ));
    assert!(prompt.contains("Assistant: Reading first."));
    assert!(prompt.contains("[Tool call] read_file({\"path\":\"README.md\"})"));
    assert!(prompt.contains("[Tool result for read_file]\n# Title"));
}

#[test]
fn reduce_openai_request_extracts_text_from_content_arrays() {
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this:"},
                {"type": "image_url", "image_url": {"url": "..."}},
                {"type": "text", "text": "Now explain."},
            ]},
        ],
    });
    let prompt = reduce_openai_request_to_prompt(&body);
    assert_eq!(
        prompt,
        "User: Look at this:\n[image attachment omitted]\nNow explain."
    );
}

#[test]
fn extract_image_blocks_inline_base64_each_protocol() {
    let openai = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}},
            ]},
        ],
    });
    assert_eq!(
        extract_openai_image_blocks(&openai).unwrap(),
        vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
    );

    let responses = json!({
        "input": [
            {"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/jpeg;base64,QUJD"},
            ]},
        ],
    });
    assert_eq!(
        extract_responses_image_blocks(&responses).unwrap(),
        vec![json!({"type": "image", "mimeType": "image/jpeg", "data": "QUJD"})]
    );

    let anthropic = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "see"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA="}},
            ]},
        ],
    });
    assert_eq!(
        extract_anthropic_image_blocks(&anthropic).unwrap(),
        vec![json!({"type": "image", "mimeType": "image/png", "data": "AAA="})]
    );

    let gemini = json!({
        "contents": [
            {"role": "user", "parts": [
                {"text": "look"},
                {"inlineData": {"mimeType": "image/jpeg", "data": "ZZZ="}},
            ]},
        ],
    });
    assert_eq!(
        extract_gemini_image_blocks(&gemini).unwrap(),
        vec![json!({"type": "image", "mimeType": "image/jpeg", "data": "ZZZ="})]
    );
}

#[test]
fn extract_image_blocks_rejects_remote_urls() {
    // Remote URLs would need fetching — refuse rather than silently drop
    // or call out to the network on behalf of the launched tool.
    let openai = json!({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
        ]}],
    });
    assert!(extract_openai_image_blocks(&openai).is_err());

    let anthropic = json!({
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
        ]}],
    });
    assert!(extract_anthropic_image_blocks(&anthropic).is_err());

    let gemini = json!({
        "contents": [{"role": "user", "parts": [
            {"fileData": {"mimeType": "image/png", "fileUri": "gs://x/y"}},
        ]}],
    });
    assert!(extract_gemini_image_blocks(&gemini).is_err());
}

#[test]
fn extract_image_blocks_empty_for_text_only_bodies() {
    assert!(
        extract_openai_image_blocks(&json!({
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap()
        .is_empty()
    );
    assert!(
        extract_anthropic_image_blocks(&json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }))
        .unwrap()
        .is_empty()
    );
    // Non-image inline data (PDF) must not show up as an image block —
    // images are gated on mime starting with `image/`.
    assert!(
        extract_gemini_image_blocks(&json!({
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "application/pdf", "data": "AAA="}},
            ]}],
        }))
        .unwrap()
        .is_empty()
    );
}

#[test]
fn parse_data_url_recognizes_base64_and_skips_other_forms() {
    assert_eq!(
        parse_data_url("data:image/png;base64,AAA="),
        Some(("image/png".to_string(), "AAA=".to_string()))
    );
    // Case-insensitive token.
    assert_eq!(
        parse_data_url("data:image/jpeg;BASE64,ZZZ"),
        Some(("image/jpeg".to_string(), "ZZZ".to_string()))
    );
    // Charset param before base64.
    assert_eq!(
        parse_data_url("data:image/svg+xml;charset=utf-8;base64,QUJD"),
        Some(("image/svg+xml".to_string(), "QUJD".to_string()))
    );
    // URL-encoded text (no base64 token) — refused.
    assert_eq!(parse_data_url("data:text/plain,hello"), None);
    // Not a data URL at all.
    assert_eq!(parse_data_url("https://example.com/x.png"), None);
}

#[test]
fn openai_chunk_frame_shapes_choices_and_delta() {
    let frame = openai_chunk_frame(
        "chatcmpl-test",
        1234567890,
        "composer-2.5",
        json!({"content": "hello"}),
        None,
    );
    assert!(frame.starts_with("data: "));
    assert!(frame.ends_with("\n\n"));
    let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
    let parsed: Value = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed["object"], "chat.completion.chunk");
    assert_eq!(parsed["model"], "composer-2.5");
    assert_eq!(parsed["choices"][0]["delta"]["content"], "hello");
    assert!(parsed["choices"][0]["finish_reason"].is_null());
}

#[test]
fn openai_chunk_frame_emits_finish_reason_when_provided() {
    let frame = openai_chunk_frame(
        "chatcmpl-test",
        1234567890,
        "composer-2.5",
        json!({}),
        Some("stop"),
    );
    let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
    let parsed: Value = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
}

#[test]
fn openai_completion_body_surfaces_content_and_reasoning() {
    let turn = AggregatedTurn {
        content: "answer".to_string(),
        reasoning: "thinking".to_string(),
    };
    let body = openai_completion_body(&turn, "composer-2.5", 42);
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "answer");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "thinking"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 42);
    // "answer" is 6 chars → (6+3)/4 = 2 tokens.
    assert_eq!(body["usage"]["completion_tokens"], 2);
    assert_eq!(body["usage"]["total_tokens"], 44);
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[test]
fn extract_agent_text_picks_only_agent_message_chunks() {
    let msg = json!({
        "sessionId": "s",
        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hi"}},
    });
    assert_eq!(extract_agent_text(&msg), Some("Hi"));

    let thought = json!({
        "sessionId": "s",
        "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "..."}},
    });
    assert_eq!(extract_agent_text(&thought), None);
    assert_eq!(extract_agent_thought(&thought), Some("..."));
}

#[test]
fn reduce_anthropic_request_preserves_tool_use_and_tool_result_blocks() {
    let body = json!({
        "model": "claude-sonnet-4-6",
        "system": "Be terse.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Hello!"},
                {"type": "tool_use", "id": "t", "name": "lookup", "input": {"q": "x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": [
                    {"type": "text", "text": "result body"},
                ]},
                {"type": "text", "text": "What's 2+2?"},
            ]},
        ],
    });
    let prompt = reduce_anthropic_request_to_prompt(&body);
    assert_eq!(
        prompt,
        "System: Be terse.\n\nUser: Hi\n\nAssistant: Hello!\n\n[Tool call] lookup({\"q\":\"x\"})\n\n[Tool result for t]\nresult body\n\nUser: What's 2+2?"
    );
}

#[test]
fn reduce_anthropic_request_emits_available_tools_header_and_system_blocks() {
    let body = json!({
        "system": [
            {"type": "text", "text": "Always be terse."},
            {"type": "text", "text": "Cite sources."},
        ],
        "tools": [
            {"name": "search", "description": "Search the web.", "input_schema": {}},
            {"name": "edit", "description": "Edit a file."},
        ],
        "messages": [
            {"role": "user", "content": "Find something."},
        ],
    });
    let prompt = reduce_anthropic_request_to_prompt(&body);
    assert!(prompt.starts_with(
        "Available tools:\n- search: Search the web.\n- edit: Edit a file.\n\nSystem: Always be terse.\nCite sources."
    ));
    assert!(prompt.ends_with("User: Find something."));
}

#[test]
fn reduce_anthropic_request_marks_image_attachments() {
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "See this:"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
                {"type": "text", "text": "Thoughts?"},
            ]},
        ],
    });
    let prompt = reduce_anthropic_request_to_prompt(&body);
    assert_eq!(
        prompt,
        "User: See this:\n\n[image attachment omitted]\n\nUser: Thoughts?"
    );
}

#[test]
fn reduce_anthropic_request_handles_string_content() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "Just a string"},
        ],
    });
    assert_eq!(
        reduce_anthropic_request_to_prompt(&body),
        "User: Just a string"
    );
}

#[test]
fn anthropic_event_uses_sse_event_data_pair() {
    let frame = sse_named_event(
        "content_block_delta",
        &json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
    );
    assert!(frame.starts_with("event: content_block_delta\n"));
    assert!(frame.contains("data: "));
    assert!(frame.ends_with("\n\n"));
}

#[test]
fn anthropic_message_body_emits_text_content_block_and_usage() {
    let turn = AggregatedTurn {
        content: "hello world".to_string(),
        reasoning: String::new(),
    };
    let body = anthropic_message_body(&turn, "claude-sonnet-4-6", 100);
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "claude-sonnet-4-6");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "hello world");
    assert_eq!(body["usage"]["input_tokens"], 100);
    // "hello world" is 11 chars → (11+3)/4 = 3 tokens.
    assert_eq!(body["usage"]["output_tokens"], 3);
    assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
    assert_eq!(body["usage"]["cache_read_input_tokens"], 0);
}

#[test]
fn extract_openai_chat_tools_normalized_unwraps_function_wrapper() {
    let body = json!({
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "ask",
                    "description": "Ask.",
                    "parameters": {"type": "object"},
                },
            },
            // Flat shape (some clients emit this)
            {"type": "function", "name": "noop"},
        ],
    });
    let out = extract_openai_chat_tools_normalized(&body);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["name"], "ask");
    assert_eq!(out[0]["input_schema"]["type"], "object");
    assert_eq!(out[1]["name"], "noop");
}

#[test]
fn extract_last_openai_tool_message_picks_latest_tool_role() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_old", "function": {"name": "x", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_old", "content": "ignored"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_new", "function": {"name": "y", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_new", "content": "blue"},
        ],
    });
    let (id, content) = extract_last_openai_tool_message(&body).unwrap();
    assert_eq!(id, "call_new");
    assert_eq!(content, "blue");
}

#[test]
fn extract_last_tool_results_returns_all_blocks_in_order() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "run both"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "one"},
                {"type": "text", "text": "aside"},
                {"type": "tool_result", "tool_use_id": "t2", "is_error": true, "content": [
                    {"type": "text", "text": "two"},
                ]},
            ]},
        ],
    });
    let results = extract_last_tool_results(&body);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "t1");
    assert_eq!(results[0].1, vec![json!({"type": "text", "text": "one"})]);
    assert!(!results[0].2);
    assert_eq!(results[1].0, "t2");
    assert_eq!(results[1].1, vec![json!({"type": "text", "text": "two"})]);
    assert!(results[1].2);
}

#[test]
fn extract_last_tool_results_skips_malformed_blocks_and_non_user_last() {
    let body = json!({
        "messages": [
            {"role": "user", "content": [
                {"note": "typeless"},
                {"type": "tool_result", "content": "no id"},
                {"type": "tool_result", "tool_use_id": "kept"},
            ]},
        ],
    });
    let results = extract_last_tool_results(&body);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "kept");
    assert!(results[0].1.is_empty());

    let assistant_last = json!({
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "x"},
            ]},
        ],
    });
    assert!(extract_last_tool_results(&assistant_last).is_empty());
}

#[test]
fn extract_gemini_tools_normalized_flattens_function_declarations() {
    let body = json!({
        "tools": [
            {"functionDeclarations": [
                {"name": "ask", "description": "Ask.", "parameters": {"type": "object"}},
                {"name": "noop"},
            ]},
            {"functionDeclarations": [
                {"name": "second", "parameters": {"type": "object"}},
            ]},
        ],
    });
    let out = extract_gemini_tools_normalized(&body);
    let names: Vec<_> = out.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["ask", "noop", "second"]);
}

#[test]
fn extract_last_gemini_function_response_finds_id_and_name() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [
                {"functionCall": {"name": "ask", "args": {"q": "x"}}},
            ]},
            {"role": "user", "parts": [
                {"functionResponse": {
                    "id": "call_42",
                    "name": "ask",
                    "response": "blue",
                }},
            ]},
        ],
    });
    let (name, id, text, is_error) = extract_last_gemini_function_response(&body).unwrap();
    assert_eq!(name, "ask");
    assert_eq!(id.as_deref(), Some("call_42"));
    assert_eq!(text, "blue");
    assert!(!is_error);
}

#[test]
fn extract_last_gemini_function_response_handles_missing_id() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [
                {"functionResponse": {"name": "ask", "response": "blue"}},
            ]},
        ],
    });
    let (name, id, text, is_error) = extract_last_gemini_function_response(&body).unwrap();
    assert_eq!(name, "ask");
    assert!(id.is_none());
    assert_eq!(text, "blue");
    assert!(!is_error);
}

#[test]
fn extract_last_gemini_function_response_detects_error_in_structured_response() {
    let body = json!({
        "contents": [
            {"role": "user", "parts": [
                {"functionResponse": {
                    "name": "ask",
                    "response": {"error": "tool exploded"},
                }},
            ]},
        ],
    });
    let (_name, _id, _text, is_error) = extract_last_gemini_function_response(&body).unwrap();
    assert!(is_error, "top-level `error` key should mark is_error");
}

#[test]
fn extract_responses_tools_normalized_renames_parameters_to_input_schema() {
    let body = json!({
        "tools": [
            {
                "type": "function",
                "name": "request_user_input",
                "description": "Ask the user.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
            {"type": "function", "name": "Noop"},
        ],
    });
    let out = extract_responses_tools_normalized(&body);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["name"], "request_user_input");
    assert_eq!(out[0]["description"], "Ask the user.");
    assert_eq!(out[0]["input_schema"]["type"], "object");
    assert_eq!(out[1]["input_schema"]["type"], "object");
}

#[test]
fn extract_last_function_call_output_finds_last_match() {
    let body = json!({
        "input": [
            {"role": "user", "content": []},
            {"type": "function_call", "call_id": "call_a", "name": "x", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "ignored"},
            {"type": "function_call", "call_id": "call_b", "name": "y", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_b", "output": "blue"},
        ],
    });
    let (id, output) = extract_last_function_call_output(&body).unwrap();
    assert_eq!(id, "call_b");
    assert_eq!(output, "blue");
}

#[test]
fn extract_last_function_call_output_returns_none_without_output_item() {
    let body = json!({"input": [{"role": "user", "content": []}]});
    assert!(extract_last_function_call_output(&body).is_none());
}

#[test]
fn reduce_responses_request_accepts_string_input() {
    let body = json!({
        "model": "gpt-5",
        "instructions": "Be terse.",
        "input": "Hi there",
    });
    assert_eq!(
        reduce_responses_request_to_prompt(&body),
        "System: Be terse.\n\nUser: Hi there"
    );
}

#[test]
fn reduce_responses_request_handles_array_input_with_typed_blocks() {
    let body = json!({
        "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "Be helpful."}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Question A."},
                {"type": "input_image", "image_url": "..."},
            ]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Sure."}]},
        ],
    });
    assert_eq!(
        reduce_responses_request_to_prompt(&body),
        "System: Be helpful.\n\nUser: Question A.\n\nAssistant: Sure."
    );
}

#[test]
fn reduce_responses_request_renders_tool_schemas_and_function_calls() {
    let body = json!({
        "tools": [
            {"type": "function", "name": "shell", "description": "Run a shell command."},
            {"type": "function", "function": {
                "name": "patch",
                "description": "Apply a unified diff.",
            }},
        ],
        "instructions": "Be terse.",
        "input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "Edit README."},
            ]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"type": "function_call", "name": "shell", "call_id": "c1",
             "arguments": "{\"command\":\"ls\"}"},
            {"type": "function_call_output", "call_id": "c1", "output": "README.md\n"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "Done."},
            ]},
        ],
    });
    let prompt = reduce_responses_request_to_prompt(&body);
    assert!(prompt.starts_with(
        "Available tools:\n- shell: Run a shell command.\n- patch: Apply a unified diff."
    ));
    assert!(prompt.contains("System: Be terse."));
    assert!(prompt.contains("User: Edit README."));
    assert!(!prompt.contains("thinking"));
    assert!(prompt.contains("[Tool call] shell({\"command\":\"ls\"})"));
    assert!(prompt.contains("[Tool result for c1]\nREADME.md"));
    assert!(prompt.ends_with("Assistant: Done."));
}

#[test]
fn tool_schema_line_truncates_long_descriptions() {
    let desc = "x".repeat(400);
    let line = format_tool_schema_line("tool", &desc);
    assert!(line.starts_with("- tool: "));
    assert!(line.ends_with("…[truncated]"));
    assert!(line.chars().count() < desc.len());
}

#[test]
fn tool_result_block_truncates_huge_outputs() {
    let huge = "x".repeat(10_000);
    let block = format_tool_result_block("read_file", &huge);
    assert!(block.starts_with("[Tool result for read_file]\n"));
    assert!(block.ends_with("…[truncated]"));
    let body_chars = block.chars().count();
    assert!(body_chars < huge.len());
}

#[test]
fn parse_request_headers_lowercases_names_and_skips_request_line() {
    let req = "POST /v1/chat/completions HTTP/1.1\r\n\
               Host: localhost\r\n\
               Content-Type: application/json\r\n\
               X-Custom-Header: value\r\n\
               \r\n\
               {}";
    let headers = parse_request_headers(req);
    assert_eq!(headers.get("host"), Some(&"localhost".to_string()));
    assert_eq!(
        headers.get("content-type"),
        Some(&"application/json".to_string())
    );
    assert_eq!(headers.get("x-custom-header"), Some(&"value".to_string()));
    // The request line should never show up as a header.
    assert!(!headers.contains_key("post /v1/chat/completions http/1.1"));
}

#[test]
fn truncate_for_log_marks_overflow() {
    let s = "x".repeat(100);
    let out = truncate_for_log(&s, 50);
    assert!(out.starts_with(&"x".repeat(50)));
    assert!(out.ends_with("…[truncated]"));
    // Under-cap inputs are returned verbatim.
    assert_eq!(truncate_for_log("short", 50), "short");
}

#[test]
fn truncate_for_log_does_not_panic_on_multibyte_char_boundary() {
    // byte 49 falls inside '前' (3-byte UTF-8); naive s[..max] would panic.
    let body = format!("{}前缀", "a".repeat(48));
    let out = truncate_for_log(&body, 49);
    assert!(out.ends_with("…[truncated]"));
    assert!(out.starts_with(&"a".repeat(48)));
}

#[test]
fn responses_completion_body_wraps_output_text_block_and_usage() {
    let turn = AggregatedTurn {
        content: "answer text".to_string(),
        reasoning: String::new(),
    };
    let body = responses_completion_body(&turn, "gpt-5", 80);
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "answer text");
    assert_eq!(body["usage"]["input_tokens"], 80);
    // "answer text" is 11 chars → (11+3)/4 = 3 tokens.
    assert_eq!(body["usage"]["output_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 83);
}

#[test]
fn title_gen_detector_matches_claude_code_signature() {
    // Real prompt observed in production logs.
    let body = json!({
        "model": "claude-haiku-4-5",
        "system": "You are Claude Code, Anthropic's official CLI for Claude.\n\
            Generate a concise, sentence-case title (3-7 words) that \
            captures the main topic or goal of this coding session. \
            Return JSON with a single \"title\" field.",
        "messages": [{"role": "user", "content": "fix the login bug"}],
    });
    assert!(is_title_generation_request(&body));
}

#[test]
fn title_gen_detector_rejects_regular_coding_prompts() {
    let body = json!({
        "system": "You are Claude Code, a coding assistant. Help the user fix bugs.",
        "messages": [{"role": "user", "content": "What's broken?"}],
    });
    assert!(!is_title_generation_request(&body));
}

#[test]
fn title_gen_detector_handles_array_system_blocks() {
    let body = json!({
        "system": [
            {"type": "text", "text": "You are Claude Code."},
            {"type": "text", "text": "Generate a concise, sentence-case title for the session. Return JSON with a single 'title' field."},
        ],
        "messages": [],
    });
    assert!(is_title_generation_request(&body));
}

#[test]
fn build_title_uses_first_user_message_and_truncates() {
    let body = json!({
        "messages": [
            {"role": "user", "content": "Help me fix the login button on the mobile homepage"},
            {"role": "assistant", "content": "Sure, let's look."},
        ],
    });
    let title = build_title_from_anthropic_body(&body);
    // Capped at 7 words.
    assert_eq!(title.split_whitespace().count(), 7);
    assert!(title.starts_with("Help me fix the login button"));
}

#[test]
fn build_title_falls_back_to_coding_session_when_empty() {
    assert_eq!(
        build_title_from_anthropic_body(&json!({"messages": []})),
        "Coding session"
    );
    assert_eq!(
        build_title_from_anthropic_body(&json!({})),
        "Coding session"
    );
}

#[test]
fn compose_short_title_breaks_on_word_boundary_when_long() {
    let raw = "thisisaverylongwordthatshouldgetbrokenat the boundary somewhere along the way";
    let title = compose_short_title(raw);
    assert!(title.chars().count() <= 60);
    assert!(!title.ends_with(' '));
}

#[tokio::test]
async fn title_gen_short_circuit_returns_json_title_without_cursor() {
    // Bind a router with no cursor session at all — if the short-circuit
    // works, the response will arrive without acquire_session_slot ever
    // hitting CursorAcpSession::open (which would hang here since there's
    // no real cursor-agent binary in tests).
    let state = state_with_models(vec![]);
    let body = json!({
        "model": "claude-haiku-4-5",
        "stream": false,
        "system": "You are Claude Code, Anthropic's official CLI for Claude. \
            Generate a concise, sentence-case title (3-7 words) that captures \
            the main topic. Return JSON with a single \"title\" field.",
        "messages": [{"role": "user", "content": "investigate the auth race condition"}],
    });
    let body_bytes = body.to_string();
    let request = format!(
        "POST /v1/messages HTTP/1.1\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n{body_bytes}",
        len = body_bytes.len()
    );
    // round_trip times out at the OS level if no response arrives. Wrap in
    // a tokio::time::timeout so a hang surfaces as a test failure instead
    // of a 60 s wait.
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        round_trip(state, &request),
    )
    .await
    .expect("title-gen must short-circuit without opening a cursor session");
    assert!(response.contains("200 OK"));
    let body = response.split("\r\n\r\n").nth(1).unwrap();
    let parsed: Value = serde_json::from_str(body).unwrap();
    // Content is itself JSON: {"title":"..."}.
    let text = parsed["content"][0]["text"].as_str().unwrap();
    let inner: Value = serde_json::from_str(text).unwrap();
    assert!(
        inner["title"]
            .as_str()
            .is_some_and(|t| t.starts_with("investigate the auth"))
    );
}

#[test]
fn estimate_tokens_rounds_up_and_handles_empty() {
    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("a"), 1);
    assert_eq!(estimate_tokens("abcd"), 1);
    assert_eq!(estimate_tokens("abcde"), 2);
    assert_eq!(estimate_tokens(&"x".repeat(100)), 25);
}

#[test]
fn openai_usage_chunk_includes_prompt_and_completion_tokens() {
    let frame = openai_usage_chunk("chatcmpl-test", 42, "composer-2.5", 120, 30);
    let payload = frame.trim_start_matches("data: ").trim_end_matches("\n\n");
    let parsed: Value = serde_json::from_str(payload).unwrap();
    assert_eq!(parsed["object"], "chat.completion.chunk");
    assert!(parsed["choices"].as_array().unwrap().is_empty());
    assert_eq!(parsed["usage"]["prompt_tokens"], 120);
    assert_eq!(parsed["usage"]["completion_tokens"], 30);
    assert_eq!(parsed["usage"]["total_tokens"], 150);
}

#[test]
fn parse_gemini_generate_path_accepts_known_prefixes_and_actions() {
    assert_eq!(
        parse_gemini_generate_path("/v1beta/models/gemini-2.5-pro:generateContent"),
        Some(GeminiGenerate {
            model: "gemini-2.5-pro".to_string(),
            stream: false,
        })
    );
    assert_eq!(
        parse_gemini_generate_path("/v1beta/models/gemini-2.5-pro:streamGenerateContent"),
        Some(GeminiGenerate {
            model: "gemini-2.5-pro".to_string(),
            stream: true,
        })
    );
    assert_eq!(
        parse_gemini_generate_path("/v1/models/composer-2.5:generateContent"),
        Some(GeminiGenerate {
            model: "composer-2.5".to_string(),
            stream: false,
        })
    );
    assert_eq!(
        parse_gemini_generate_path("/models/composer-2.5:generateContent"),
        Some(GeminiGenerate {
            model: "composer-2.5".to_string(),
            stream: false,
        })
    );
    // Wrong action verb falls through so the unknown-path 404 fires.
    assert_eq!(
        parse_gemini_generate_path("/v1beta/models/x:countTokens"),
        None
    );
    // Non-Gemini paths are not falsely matched.
    assert_eq!(parse_gemini_generate_path("/v1/messages"), None);
    assert_eq!(parse_gemini_generate_path("/v1/chat/completions"), None);
}

#[test]
fn reduce_gemini_request_renders_system_user_and_tool_history() {
    let body = json!({
        "systemInstruction": {"parts": [{"text": "Be terse."}]},
        "tools": [{"functionDeclarations": [
            {"name": "read_file", "description": "Read a file."},
        ]}],
        "contents": [
            {"role": "user", "parts": [{"text": "Look at config.toml."}]},
            {"role": "model", "parts": [
                {"text": "Reading."},
                {"functionCall": {"name": "read_file", "args": {"path": "config.toml"}}},
            ]},
            {"role": "user", "parts": [{"functionResponse": {
                "name": "read_file",
                "response": {"output": "# title"},
            }}]},
        ],
    });
    let prompt = reduce_gemini_request_to_prompt(&body);
    assert!(prompt.starts_with("Available tools:\n- read_file: Read a file."));
    assert!(prompt.contains("System: Be terse."));
    assert!(prompt.contains("User: Look at config.toml."));
    assert!(prompt.contains("Assistant: Reading."));
    assert!(prompt.contains("[Tool call] read_file"));
    assert!(prompt.contains("[Tool result for read_file]"));
}

#[test]
fn gemini_response_body_shapes_candidates_and_usage() {
    let turn = AggregatedTurn {
        content: "hello".to_string(),
        reasoning: String::new(),
    };
    let body = gemini_response_body(&turn, "composer-2.5", 12);
    let candidate = &body["candidates"][0];
    assert_eq!(candidate["content"]["role"], "model");
    assert_eq!(candidate["content"]["parts"][0]["text"], "hello");
    assert_eq!(candidate["finishReason"], "STOP");
    assert_eq!(body["usageMetadata"]["promptTokenCount"], 12);
    assert_eq!(body["modelVersion"], "composer-2.5");
}

#[test]
fn gemini_stream_frames_have_no_done_marker_and_carry_usage_on_final() {
    let text_frame = gemini_stream_text_frame("composer-2.5", "delta");
    assert!(text_frame.starts_with("data: "));
    let parsed: Value = serde_json::from_str(
        text_frame
            .trim_start_matches("data: ")
            .trim_end_matches("\n\n"),
    )
    .unwrap();
    assert_eq!(
        parsed["candidates"][0]["content"]["parts"][0]["text"],
        "delta"
    );
    // Intermediate frames carry no finishReason — clients buffer text until
    // the final frame announces STOP.
    assert!(parsed["candidates"][0]["finishReason"].is_null());

    let final_frame = gemini_stream_final_frame("composer-2.5", "STOP", 100, 50, 150);
    let parsed: Value = serde_json::from_str(
        final_frame
            .trim_start_matches("data: ")
            .trim_end_matches("\n\n"),
    )
    .unwrap();
    assert_eq!(parsed["candidates"][0]["finishReason"], "STOP");
    assert_eq!(parsed["usageMetadata"]["totalTokenCount"], 150);
}

#[tokio::test]
async fn gemini_generate_path_routes_to_gemini_handler() {
    // Mirrors `unversioned_post_paths_route_to_the_same_handlers`: a body
    // missing `contents` triggers the handler's "reduced prompt is empty"
    // error, which is the proof the dispatcher reached the Gemini handler
    // (and not the catch-all 404).
    let state = state_with_models(vec![]);
    for path in [
        "/v1beta/models/gemini-2.5-pro:generateContent",
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent",
        "/v1/models/composer-2.5:generateContent",
    ] {
        let body = r#"{"contents": []}"#;
        let request = format!(
            "POST {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
            len = body.len()
        );
        let response = round_trip(state.clone(), &request).await;
        assert!(
            !response.contains("cursor_model_router_not_found"),
            "{path} unexpectedly routed to the 404 handler: {response}"
        );
        assert!(
            response.contains("reduced prompt is empty"),
            "{path} should have reached the Gemini handler's empty-prompt error: {response}"
        );
    }
}

/// Cold path: when no prewarm has been configured (`in_flight=false`,
/// `ready=None`), `take_mcp_prewarmed` returns `None` immediately
/// without blocking. The 25 s timeout must never kick in here.
#[tokio::test]
async fn take_mcp_prewarmed_cold_returns_none_immediately() {
    let state = state_with_models(vec!["composer-2.5"]);
    let start = std::time::Instant::now();
    let result = take_mcp_prewarmed(&state).await;
    assert!(result.is_none());
    assert!(
        start.elapsed() < Duration::from_millis(100),
        "should not block when in_flight=false; took {:?}",
        start.elapsed()
    );
}

/// The message-item JSON matches the existing assistant message shape
/// codex consumes (output_text with annotations array).
#[test]
fn message_item_done_has_codex_compatible_shape() {
    let item = message_item_done("msg_test", "Hi");
    assert_eq!(item.get("type").unwrap(), "message");
    assert_eq!(item.get("status").unwrap(), "completed");
    assert_eq!(item.get("role").unwrap(), "assistant");
    let content = item.get("content").and_then(Value::as_array).unwrap();
    assert_eq!(content[0].get("type").unwrap(), "output_text");
    assert_eq!(content[0].get("text").unwrap(), "Hi");
    assert!(content[0].get("annotations").is_some());
}

/// The reasoning-item JSON matches what codex 0.132+ expects so the
/// reasoning panel renders. Mirrors responses_chat_conversion.rs shape.
#[test]
fn reasoning_item_done_has_codex_compatible_shape() {
    let item = reasoning_item_done("rs_test", "Looking up sum.js");
    assert_eq!(item.get("id").unwrap(), "rs_test");
    assert_eq!(item.get("type").unwrap(), "reasoning");
    let summary = item.get("summary").and_then(Value::as_array).unwrap();
    assert_eq!(summary.len(), 1);
    assert_eq!(summary[0].get("type").unwrap(), "summary_text");
    assert_eq!(summary[0].get("text").unwrap(), "Looking up sum.js");
}

/// Both thoughts and tool_call session updates must be extractable as
/// reasoning text — without this, codex shows nothing during cursor's
/// long native-tool runs and the user sees a blank terminal for ~17 s.
#[test]
fn thoughts_and_tool_calls_both_produce_reasoning_text() {
    let thought = json!({
        "update": {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "Thinking about sum.js"},
        }
    });
    let tool_call = json!({
        "update": {
            "sessionUpdate": "tool_call",
            "kind": "execute",
            "title": "`node sum.js`",
            "status": "in_progress",
        }
    });
    assert_eq!(
        extract_agent_thought(&thought),
        Some("Thinking about sum.js")
    );
    let marker = extract_tool_call_marker(&tool_call);
    assert!(marker.is_some());
    assert!(marker.unwrap().contains("node sum.js"));
}

/// Wait path: when a prewarm is in flight, `take_mcp_prewarmed` waits
/// on the `completed` notify. If the prewarm finishes without filling
/// `ready` (i.e. the prewarm task hit an error), the take loop sees
/// `in_flight=false` on its next iteration and returns `None` —
/// callers then fall through to the cold path.
#[tokio::test]
async fn take_mcp_prewarmed_wakes_on_failed_prewarm_and_falls_back_to_cold() {
    let state = state_with_models(vec!["composer-2.5"]);
    // Simulate a scheduled-but-not-yet-completed prewarm.
    {
        let mut guard = state.mcp_prewarmed.lock().await;
        guard.in_flight = true;
    }
    let state_clone = state.clone();
    // Simulate the prewarm task failing: clear in_flight + notify, but
    // leave ready=None.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let notify = {
            let mut guard = state_clone.mcp_prewarmed.lock().await;
            guard.in_flight = false;
            guard.completed.clone()
        };
        notify.notify_waiters();
    });

    let start = std::time::Instant::now();
    let result = take_mcp_prewarmed(&state).await;
    assert!(result.is_none());
    assert!(
        start.elapsed() < Duration::from_secs(1),
        "should wake on notify, not on the 25 s timeout; took {:?}",
        start.elapsed()
    );
}

#[test]
fn json_constraint_openai_json_object_appended() {
    let body = json!({"response_format": {"type": "json_object"}});
    let out = append_json_output_constraint("User: extract fields".to_string(), &body, false);
    assert_eq!(
        out,
        "User: extract fields\n\nOUTPUT CONSTRAINT: Respond with a single valid JSON object and nothing else — no prose, no markdown, no code fences."
    );
}

#[test]
fn json_constraint_openai_json_schema_includes_schema() {
    let body = json!({"response_format": {
        "type": "json_schema",
        "json_schema": {"name": "r", "schema": {"type": "object", "properties": {"x": {"type": "number"}}}}
    }});
    let out = append_json_output_constraint("User: hi".to_string(), &body, false);
    assert!(out.contains("must conform to this schema:"));
    assert!(out.contains("\"properties\":{\"x\":{\"type\":\"number\"}}"));
}

#[test]
fn json_constraint_responses_text_format_schema_at_top() {
    let body = json!({"text": {"format": {"type": "json_schema", "schema": {"type": "array"}}}});
    let out = append_json_output_constraint("User: hi".to_string(), &body, false);
    assert!(out.contains("conform to this schema: {\"type\":\"array\"}"));
}

#[test]
fn json_constraint_gemini_response_mime_type() {
    let body = json!({"generationConfig": {
        "responseMimeType": "application/json",
        "responseSchema": {"type": "object"}
    }});
    let out = append_json_output_constraint("User: hi".to_string(), &body, false);
    assert!(out.contains("conform to this schema: {\"type\":\"object\"}"));
}

#[test]
fn json_constraint_absent_without_format() {
    let body = json!({"messages": [{"role": "user", "content": "hi"}]});
    assert_eq!(json_output_constraint(&body), None);
    assert_eq!(
        append_json_output_constraint("User: hi".to_string(), &body, false),
        "User: hi"
    );
}

#[test]
fn json_constraint_absent_for_plain_text_format() {
    let body = json!({"response_format": {"type": "text"}});
    assert_eq!(json_output_constraint(&body), None);
}

#[test]
fn json_constraint_noop_on_empty_prompt() {
    let body = json!({"response_format": {"type": "json_object"}});
    assert_eq!(
        append_json_output_constraint(String::new(), &body, false),
        ""
    );
}

#[test]
fn json_constraint_image_only_emits_lone_constraint() {
    let body = json!({"response_format": {"type": "json_object"}});
    let out = append_json_output_constraint(String::new(), &body, true);
    assert_eq!(
        out,
        "OUTPUT CONSTRAINT: Respond with a single valid JSON object and nothing else — no prose, no markdown, no code fences."
    );
}

#[test]
fn json_constraint_reaches_image_only_responses_turn() {
    // Responses drops input_image from reduced text → empty prompt; constraint must survive.
    let body = json!({
        "input": [{"type": "message", "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}],
        "text": {"format": {"type": "json_object"}}
    });
    let reduced = reduce_responses_request_to_prompt(&body);
    assert!(reduced.trim().is_empty());
    assert!(append_json_output_constraint(reduced, &body, true).contains("OUTPUT CONSTRAINT"));
}

#[test]
fn request_authorized_accepts_bearer_or_x_api_key() {
    use crate::services::http_utils::request_bearer_authorized as auth;
    let req = |h: &str| format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{h}\r\n\r\n{{}}");
    // Either form of the expected token is accepted.
    assert!(auth(&req("Authorization: Bearer tok123"), "tok123"));
    assert!(auth(&req("x-api-key: tok123"), "tok123"));
    // Wrong / missing token is rejected.
    assert!(!auth(&req("Authorization: Bearer nope"), "tok123"));
    assert!(!auth(&req("Host: x"), "tok123"));
}

// === Regression: mid-stream error signaling + stopReason mapping ===

#[test]
fn acp_stop_from_result_normalizes_stop_reasons() {
    assert_eq!(
        acp_stop_from_result(&json!({"stopReason": "max_tokens"})),
        AcpStop::MaxTokens
    );
    assert_eq!(
        acp_stop_from_result(&json!({"stopReason": "max_turn_requests"})),
        AcpStop::MaxTokens
    );
    assert_eq!(
        acp_stop_from_result(&json!({"stopReason": "refusal"})),
        AcpStop::Refusal
    );
    // end_turn, cancelled, unknown, and absent all fold to EndTurn.
    for v in [
        json!({"stopReason": "end_turn"}),
        json!({"stopReason": "cancelled"}),
        json!({"stopReason": "something_new"}),
        json!({}),
    ] {
        assert_eq!(acp_stop_from_result(&v), AcpStop::EndTurn);
    }
}

#[test]
fn protocol_finish_reasons_map_each_stop_kind() {
    // Anthropic uses its closed enum; OpenAI + Gemini use their own vocab.
    assert_eq!(anthropic_stop_reason(AcpStop::MaxTokens), "max_tokens");
    assert_eq!(anthropic_stop_reason(AcpStop::Refusal), "refusal");
    assert_eq!(anthropic_stop_reason(AcpStop::EndTurn), "end_turn");
    assert_eq!(openai_finish_reason(AcpStop::MaxTokens), "length");
    assert_eq!(openai_finish_reason(AcpStop::Refusal), "content_filter");
    assert_eq!(openai_finish_reason(AcpStop::EndTurn), "stop");
    assert_eq!(gemini_finish_reason(AcpStop::MaxTokens), "MAX_TOKENS");
    assert_eq!(gemini_finish_reason(AcpStop::Refusal), "SAFETY");
    assert_eq!(gemini_finish_reason(AcpStop::EndTurn), "STOP");
}

#[test]
fn error_frames_carry_a_client_visible_error_signal() {
    // Anthropic: a spec `error` event, not a bogus stop_reason.
    let anth = anthropic_error_event("boom");
    assert!(anth.contains("event: error"));
    assert!(anth.contains("\"type\":\"error\""));
    assert!(anth.contains("boom"));

    // OpenAI chat: a terminal `error` object (not finish_reason: "error:..").
    let oai = openai_error_chunk("boom");
    assert!(oai.starts_with("data: "));
    assert!(oai.contains("\"error\""));
    assert!(oai.contains("boom"));

    // Gemini: an `error` object with a numeric code.
    let gem = gemini_error_frame("boom");
    assert!(gem.contains("\"error\""));
    assert!(gem.contains("\"code\":500"));
    assert!(gem.contains("boom"));
}

#[test]
fn malformed_request_body_maps_to_400_not_502() {
    // A JSON parse failure is the client's fault — 400 so SDKs fail fast
    // instead of retry-looping on a 5xx.
    let parse_err = serde_json::from_str::<serde_json::Value>("{not json").unwrap_err();
    let err = anyhow::Error::new(parse_err).context("parse request body");
    assert_eq!(status_for_handler_error(&err), 400);

    // An upstream failure with no serde error in the chain stays 502.
    let upstream = anyhow::anyhow!("cursor-agent session/prompt failed");
    assert_eq!(status_for_handler_error(&upstream), 502);
}
