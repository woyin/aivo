use super::super::*;
use super::helpers::*;
use std::io::Write;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

#[test]
fn default_reasoning_effort_gates_on_model_capability() {
    // Reasoning-capable models (snapshot `r` flag) get an effort to send…
    for model in ["o3", "gpt-5", "claude-sonnet-4-5", "gemini-2.5-pro"] {
        assert_eq!(
            default_reasoning_effort(model).as_deref(),
            Some("medium"),
            "model={model} should request reasoning"
        );
    }
    // …non-reasoning models and unknown ids never send it (would 400 strict providers).
    // claude-3-haiku, not 3.5-sonnet: the latter's folded key now unions to a
    // reasoning-capable upstream listing after the dash alias was dropped.
    for model in [
        "gpt-4o",
        "claude-3-haiku",
        "definitely-not-a-real-model-xyz",
    ] {
        assert_eq!(
            default_reasoning_effort(model),
            None,
            "model={model} must not request reasoning"
        );
    }
}

#[test]
fn thinking_request_tracks_capability_when_enabled() {
    // Reasoning-capable model: the level is always requested; `/effort` changes it.
    let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    assert_eq!(engine.thinking_request(), (Some("medium"), false));
    engine.set_reasoning_effort("high".into());
    assert_eq!(engine.thinking_request(), (Some("high"), false));

    // Non-reasoning model: never requested.
    let plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
    assert_eq!(plain.thinking_request(), (None, false));
}

#[test]
fn thinking_request_clamps_effort_to_catalog() {
    // A level carried across a model switch that this model's catalog doesn't
    // advertise is omitted (sending it would 400), not forwarded verbatim.
    let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    engine.set_reasoning_effort("xhigh".into());
    engine.set_reasoning_efforts(vec!["low".into(), "medium".into(), "high".into()]);
    assert_eq!(engine.thinking_request(), (None, false));

    engine.set_reasoning_effort("high".into());
    assert_eq!(engine.thinking_request(), (Some("high"), false));

    // No catalog → nothing to clamp against; the level passes through.
    engine.set_reasoning_efforts(Vec::new());
    engine.set_reasoning_effort("custom".into());
    assert_eq!(engine.thinking_request(), (Some("custom"), false));
}

#[test]
fn thinking_request_disables_per_provider_disable_form() {
    // gpt-5 / o-series reject `"none"` alongside tools and reject `thinking` → family effort floor.
    let mut g5 = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
    g5.set_thinking_enabled(false);
    assert_eq!(g5.thinking_request(), (Some("minimal"), false));
    let mut o = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    o.set_thinking_enabled(false);
    assert_eq!(o.thinking_request(), (Some("low"), false));

    // A catalog that lists `none` → send it (a real effort-level off).
    let mut has_none = AgentEngine::new("/tmp", "deepseek-reasoner", "", &[], &[], 0, 0);
    has_none.set_reasoning_efforts(vec!["none".into(), "low".into(), "high".into()]);
    has_none.set_thinking_enabled(false);
    assert_eq!(has_none.thinking_request(), (Some("none"), false));

    // gpt-5.4+ refuses tools + `none` (`tools_require_reasoning_effort`), so
    // with agent tools on (the default) the off request is proactively floored
    // to the lowest advertised thinking level — no 400 round-trip.
    let mut g54 = AgentEngine::new("/tmp", "gpt-5.4", "", &[], &[], 0, 0);
    g54.set_reasoning_efforts(
        ["none", "low", "medium", "high", "xhigh"]
            .map(String::from)
            .to_vec(),
    );
    g54.set_thinking_enabled(false);
    assert_eq!(g54.thinking_request(), (Some("low"), false));
    // Plain chat (no tools): `none` is valid and ships (c5d6b17 regression).
    g54.set_agent_tools_enabled(false);
    assert_eq!(g54.thinking_request(), (Some("none"), false));

    // No catalog: with tools the proactive clause guesses `low` (same family
    // guess as o-series); without tools the 5.1+ heuristic guesses `none`,
    // never the 400ing `minimal`.
    let mut g54_bare = AgentEngine::new("/tmp", "gpt-5.4", "", &[], &[], 0, 0);
    g54_bare.set_thinking_enabled(false);
    assert_eq!(g54_bare.thinking_request(), (Some("low"), false));
    g54_bare.set_agent_tools_enabled(false);
    assert_eq!(g54_bare.thinking_request(), (Some("none"), false));

    // A stale catalog advertising `minimal` for 5.1+ must not resurrect it.
    let mut g51 = AgentEngine::new("/tmp", "gpt-5.1", "", &[], &[], 0, 0);
    g51.set_reasoning_efforts(
        ["minimal", "low", "medium", "high"]
            .map(String::from)
            .to_vec(),
    );
    g51.set_thinking_enabled(false);
    assert_eq!(g51.thinking_request(), (Some("low"), false));

    // codex advertises only low/medium/high → its `low` floor, not `minimal`.
    let mut codex = AgentEngine::new("/tmp", "gpt-5-codex", "", &[], &[], 0, 0);
    codex.set_reasoning_efforts(["low", "medium", "high"].map(String::from).to_vec());
    codex.set_thinking_enabled(false);
    assert_eq!(codex.thinking_request(), (Some("low"), false));

    // Effort scale with no off (aivo/starter, snapshot-absent): emit the `thinking` disable field, not an invalid `"none"` effort.
    let mut alias = AgentEngine::new("/tmp", "aivo/starter", "", &[], &[], 0, 0);
    assert!(
        !alias.reasoning_capable,
        "alias is absent from the snapshot"
    );
    alias.set_reasoning_efforts(vec![
        "low".into(),
        "medium".into(),
        "high".into(),
        "xhigh".into(),
        "max".into(),
    ]);
    alias.set_thinking_enabled(false);
    assert_eq!(alias.thinking_request(), (None, true));

    // Snapshot-known Anthropic model (no none/minimal): the `thinking` field, carried by the bridge.
    let mut claude = AgentEngine::new("/tmp", "claude-sonnet-4-5", "", &[], &[], 0, 0);
    claude.set_thinking_enabled(false);
    assert_eq!(claude.thinking_request(), (None, true));

    // Genuinely non-reasoning model with no catalog level: stay silent.
    let mut plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
    plain.set_thinking_enabled(false);
    assert_eq!(plain.thinking_request(), (None, false));
}

#[test]
fn gpt54_tools_constraint_is_proactive() {
    // Explicit "none" with tools on ships the lowest advertised level — the
    // known constraint never costs a 400 round-trip.
    let mut g56 = AgentEngine::new("/tmp", "gpt-5.6-sol", "", &[], &[], 0, 0);
    g56.set_reasoning_efforts(["none", "low", "medium", "high"].map(String::from).to_vec());
    g56.set_reasoning_effort("none".into());
    assert_eq!(g56.thinking_request(), (Some("low"), false));

    // Real levels pass through untouched.
    g56.set_reasoning_effort("medium".into());
    assert_eq!(g56.thinking_request(), (Some("medium"), false));

    // Plain chat (no tools): none is valid again.
    g56.set_reasoning_effort("none".into());
    g56.set_agent_tools_enabled(false);
    assert_eq!(g56.thinking_request(), (Some("none"), false));
}

#[test]
fn effort_floor_overrides_off_grade_requests() {
    // The `tools_require_reasoning_effort` 400 raises the floor to the lowest
    // advertised thinking level; off-grade outcomes then send it instead.
    // A non-gpt id: the reactive floor must work on its own, without the
    // proactive gpt-5.4+ clause.
    let mut engine = AgentEngine::new("/tmp", "laguna-nova", "", &[], &[], 0, 0);
    engine.set_reasoning_efforts(
        ["none", "low", "medium", "high", "xhigh", "max"]
            .map(String::from)
            .to_vec(),
    );
    engine.set_reasoning_effort("none".into());
    assert_eq!(engine.thinking_request(), (Some("none"), false));
    assert_eq!(engine.raise_effort_floor("").as_deref(), Some("low"));
    assert_eq!(engine.thinking_request(), (Some("low"), false));

    // The thinking-off path (which resolves to "none" here) floors too.
    engine.set_thinking_enabled(false);
    assert_eq!(engine.thinking_request(), (Some("low"), false));

    // A real level is untouched by the floor.
    engine.set_thinking_enabled(true);
    engine.set_reasoning_effort("high".into());
    assert_eq!(engine.thinking_request(), (Some("high"), false));

    // A catalog with nothing above off can't heal — no guessed level.
    let mut only_off = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    only_off.set_reasoning_efforts(vec!["none".into()]);
    assert_eq!(only_off.raise_effort_floor(""), None);

    // The rejection's own allowed list outranks the catalog.
    let mut stale = AgentEngine::new("/tmp", "laguna-nova", "", &[], &[], 0, 0);
    stale.set_reasoning_efforts(["none", "low", "medium"].map(String::from).to_vec());
    stale.set_reasoning_effort("none".into());
    assert_eq!(
        stale
            .raise_effort_floor(
                r#"{"error":{"message":"Invalid option: expected one of \"medium\"|\"high\"","param":"reasoning_effort"}}"#
            )
            .as_deref(),
        Some("medium")
    );
    assert_eq!(stale.thinking_request(), (Some("medium"), false));
}

/// First request → the given effort 400; retry → a normal completion.
/// Every request body lands in `captured`.
fn spawn_effort_400_then_ok(captured: Arc<Mutex<Vec<String>>>, body: &'static str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            captured.lock().unwrap().push(drain_request(&mut sock));
            let resp = format!(
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(resp.as_bytes());
        }
        if let Ok((mut sock, _)) = listener.accept() {
            captured.lock().unwrap().push(drain_request(&mut sock));
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{FINAL_TEXT_SSE}",
                FINAL_TEXT_SSE.len()
            );
            let _ = sock.write_all(resp.as_bytes());
        }
    });
    port
}

/// End-to-end: the 400 is healed in-flight — floored retry, no terminal error,
/// and the floor sticks for the session's later requests. A non-gpt id: this
/// is the last-resort path for providers the proactive clause can't predict.
#[tokio::test]
async fn tools_effort_400_floors_and_retries() {
    let dir = tmp();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_effort_400_then_ok(
        captured.clone(),
        r#"{"error":{"code":"tools_require_reasoning_effort","message":"GPT-5.4+ Chat Completions does not support tools with reasoning_effort: \"none\". Switch to a higher effort or use the Responses API.","type":"invalid_request_error"}}"#,
    );
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "laguna-nova",
        "",
        &[],
        &[],
        0,
        0,
    );
    engine.set_reasoning_efforts(["none", "low", "medium", "high"].map(String::from).to_vec());
    engine.set_reasoning_effort("none".into());

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("hi".into()),
        &mut ui,
    )
    .await;

    let reqs = captured.lock().unwrap();
    assert_eq!(reqs.len(), 2, "one rejection, one healed retry");
    assert!(reqs[0].contains("\"reasoning_effort\":\"none\""));
    assert!(
        reqs[1].contains("\"reasoning_effort\":\"low\""),
        "retry floors to low"
    );
    assert_eq!(ui.text, "done");
    assert!(
        ui.errors.is_empty(),
        "healed, not terminal: {:?}",
        ui.errors
    );
    // The host was told the level that actually ships (badge/persistence follow),
    // and the explanation lands after it so it wins the notice slot.
    assert_eq!(ui.efforts, vec!["low".to_string()]);
    assert!(
        ui.notices
            .last()
            .is_some_and(|n| n.contains("can't turn thinking off")),
        "notices: {:?}",
        ui.notices
    );
    // Later turns reuse the engine — the floor keeps them from re-400ing.
    assert_eq!(engine.thinking_request(), (Some("low"), false));
}

/// The catalog advertises "none" but the route's schema doesn't (gpt-5.6 via
/// gateway): the 400 floors to the error's own allowed list and retries.
#[tokio::test]
async fn invalid_effort_option_400_floors_and_retries() {
    let dir = tmp();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let port = spawn_effort_400_then_ok(
        captured.clone(),
        r#"{"error":{"message":"Invalid option: expected one of \"low\"|\"medium\"|\"high\"|\"xhigh\"|\"max\"","type":"invalid_request_error","param":"reasoning_effort"}}"#,
    );
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "laguna-nova",
        "",
        &[],
        &[],
        0,
        0,
    );
    engine.set_reasoning_efforts(["none", "low", "medium", "high"].map(String::from).to_vec());
    engine.set_reasoning_effort("none".into());

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("hi".into()),
        &mut ui,
    )
    .await;

    let reqs = captured.lock().unwrap();
    assert_eq!(reqs.len(), 2, "one rejection, one healed retry");
    assert!(reqs[0].contains("\"reasoning_effort\":\"none\""));
    assert!(
        reqs[1].contains("\"reasoning_effort\":\"low\""),
        "retry uses the error's lowest allowed level"
    );
    assert_eq!(ui.text, "done");
    assert!(
        ui.errors.is_empty(),
        "healed, not terminal: {:?}",
        ui.errors
    );
    assert!(
        ui.notices
            .last()
            .is_some_and(|n| n.contains("rejected the requested level")),
        "notices: {:?}",
        ui.notices
    );
}
