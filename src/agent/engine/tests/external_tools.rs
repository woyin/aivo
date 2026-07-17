use super::super::*;
use super::helpers::*;
use crate::agent::request::content_str;
use serde_json::json;

/// An external source's schemas are offered and a call routes to the source, not the built-in executor (mock, no real MCP subprocess).
#[tokio::test]
async fn external_tools_are_offered_and_routed() {
    struct MockExt;
    impl crate::agent::engine::ExternalTools for MockExt {
        fn specs(&self) -> Vec<Value> {
            vec![json!({
                "type": "function",
                "function": {"name": "mcp__demo__ping", "description": "d", "parameters": {"type": "object"}}
            })]
        }
        fn handles(&self, name: &str) -> bool {
            name == "mcp__demo__ping"
        }
        fn call<'a>(
            &'a self,
            _name: &'a str,
            _args: &'a Value,
        ) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(async { Ok("pong".to_string()) })
        }
    }

    let dir = tmp();
    // Turn 1: the model calls the external tool; turn 2: it converges.
    let call = tool_call_sse("mcp__demo__ping", json!({}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_external_tools(std::sync::Arc::new(MockExt));
    // The external schema is advertised alongside the built-ins.
    assert!(
        engine
            .tools_openai
            .iter()
            .any(|t| t["function"]["name"] == "mcp__demo__ping")
    );

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("use the tool".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.tools, vec!["mcp__demo__ping"]);
    assert_eq!(ui.text, "done");
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m) == "pong"),
        "external tool result not routed back"
    );
}

/// 80 filler tools + one distinctive `alpha_sync` (~16k est tokens — defers).
struct BigExt;

impl crate::agent::engine::ExternalTools for BigExt {
    fn specs(&self) -> Vec<Value> {
        let mut specs: Vec<Value> = (0..80)
            .map(|i| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("mcp__demo__filler_{i}"),
                        "description": "filler tool with a long schema description ".repeat(20),
                        "parameters": {"type": "object"}
                    }
                })
            })
            .collect();
        specs.push(json!({
            "type": "function",
            "function": {
                "name": "mcp__demo__alpha_sync",
                "description": "Synchronize alpha records upstream",
                "parameters": {"type": "object"}
            }
        }));
        specs
    }
    fn handles(&self, name: &str) -> bool {
        name.starts_with("mcp__demo__")
    }
    fn call<'a>(
        &'a self,
        name: &'a str,
        _args: &'a Value,
    ) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move { Ok(format!("ran {name}")) })
    }
}

/// Defer → search loads the match → the loaded tool routes to its source.
#[tokio::test]
async fn bulky_external_tools_defer_and_load_via_search() {
    let dir = tmp();
    let search = tool_call_sse("search_tools", json!({"query": "alpha sync"}));
    let call = tool_call_sse("mcp__demo__alpha_sync", json!({}));
    let port = spawn_sse_sequence(vec![search, call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // Pin the threshold so an exported AIVO_AGENT_MCP_DEFER_TOKENS can't flip the test.
    engine.mcp_defer_tokens = Some(8_000);
    engine.set_external_tools(std::sync::Arc::new(BigExt));

    let names = tool_names(&engine);
    assert!(
        names.iter().any(|n| n == "search_tools"),
        "meta-tool advertised"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("mcp__")),
        "no external schema inlined while deferred"
    );
    assert_eq!(engine.deferred_tools.len(), 81);
    let report = engine.context_report();
    assert_eq!(report.mcp_tool_count, 0, "deferred specs cost no context");
    assert_eq!(report.mcp_tools, 0);
    assert_eq!(report.mcp_deferred_count, 81);

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("sync the alpha records".into()),
        &mut ui,
    )
    .await;

    assert!(
        tool_names(&engine)
            .iter()
            .any(|n| n == "mcp__demo__alpha_sync"),
        "search must load the matching schema"
    );
    assert_eq!(engine.deferred_tools.len(), 80);
    assert!(
        engine.messages.iter().any(|m| role(m) == "tool"
            && content_str(m).contains("Loaded 1 tool(s)")
            && content_str(m).contains("mcp__demo__alpha_sync")),
        "search result names what loaded"
    );
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m) == "ran mcp__demo__alpha_sync"),
        "loaded tool routed to the external source"
    );
    let report = engine.context_report();
    assert_eq!(report.mcp_tool_count, 1, "loaded spec now counted");
    assert!(report.mcp_tools > 0);
    assert_eq!(report.mcp_deferred_count, 80);
}

/// A direct call to a still-deferred tool executes and promotes its schema.
#[tokio::test]
async fn direct_call_to_deferred_tool_routes_and_promotes() {
    let dir = tmp();
    let call = tool_call_sse("mcp__demo__filler_3", json!({}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.mcp_defer_tokens = Some(8_000);
    engine.set_external_tools(std::sync::Arc::new(BigExt));
    assert!(
        !tool_names(&engine)
            .iter()
            .any(|n| n == "mcp__demo__filler_3")
    );

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("run filler 3".into()),
        &mut ui,
    )
    .await;

    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m) == "ran mcp__demo__filler_3"),
        "deferred tool must still execute when called directly"
    );
    assert!(
        tool_names(&engine)
            .iter()
            .any(|n| n == "mcp__demo__filler_3"),
        "direct call promotes the schema"
    );
    assert_eq!(engine.deferred_tools.len(), 80);
}

/// The model calls `take_note`; the engine stores it (no prompt, no `tools::execute`), echoes a confirmation, retains it for pinning.
#[tokio::test]
async fn take_note_is_dispatched_and_stored() {
    let dir = tmp();
    let call = tool_call_sse("take_note", json!({"note": "the parser is in lexer.rs"}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // `take_note` is advertised alongside the built-ins.
    assert!(
        engine
            .tools_openai
            .iter()
            .any(|t| t["function"]["name"] == "take_note")
    );

    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("remember where the parser is".into()),
        &mut ui,
    )
    .await;

    assert_eq!(engine.notes.len(), 1);
    assert_eq!(engine.notes[0].text, "the parser is in lexer.rs");
    assert!(engine.notes[0].id.is_none());
    // No permission prompt was raised for the note.
    assert_eq!(ui.asks, 0);
    // The confirmation came back as the tool result.
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m).starts_with("Noted (1 saved)")),
        "note confirmation not echoed back"
    );
}

/// An external source that `requires_approval` is gated: a denying UI refuses the call (not executed).
#[tokio::test]
async fn external_tool_requiring_approval_is_gated() {
    struct GatedExt;
    impl crate::agent::engine::ExternalTools for GatedExt {
        fn specs(&self) -> Vec<Value> {
            vec![json!({
                "type": "function",
                "function": {"name": "mcp__risky__wipe", "description": "d", "parameters": {"type": "object"}}
            })]
        }
        fn handles(&self, name: &str) -> bool {
            name == "mcp__risky__wipe"
        }
        fn requires_approval(&self, _name: &str) -> bool {
            true
        }
        fn call<'a>(
            &'a self,
            _name: &'a str,
            _args: &'a Value,
        ) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(async { Ok("WIPED".to_string()) })
        }
    }

    let dir = tmp();
    let call = tool_call_sse("mcp__risky__wipe", json!({}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_external_tools(std::sync::Arc::new(GatedExt));
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("wipe it".into()), &mut ui).await;

    assert_eq!(ui.tools, vec!["mcp__risky__wipe"]);
    assert!(
        engine
            .messages
            .iter()
            .any(|m| role(m) == "tool" && content_str(m).contains("denied")),
        "gated external call should have been denied"
    );
    assert!(
        !engine.messages.iter().any(|m| content_str(m) == "WIPED"),
        "denied tool must not have executed"
    );
}
