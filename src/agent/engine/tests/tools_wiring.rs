use super::super::*;
use super::helpers::test_generator;
use std::path::PathBuf;

#[test]
fn web_search_toggle_adds_and_removes_local_tool() {
    let mut e = AgentEngine::new("/tmp", "deepseek-v4", "", &[], &[], 0, 0);
    let has = |e: &AgentEngine| {
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"].as_str() == Some("web_search"))
    };
    assert!(has(&e), "non-native model starts with web_search");
    e.set_web_search_enabled(false);
    assert!(!has(&e), "toggle off removes it");
    e.set_web_search_enabled(false);
    assert!(!has(&e));
    e.set_web_search_enabled(true);
    assert!(has(&e), "toggle on re-adds it");
}

#[test]
fn image_model_gates_generate_image_tool() {
    let mut e = AgentEngine::new("/tmp", "deepseek-v4", "", &[], &[], 0, 0);
    let has = |e: &AgentEngine| {
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"].as_str() == Some("generate_image"))
    };
    assert!(!has(&e), "no image model → no tool");
    e.set_image_source(Some(test_generator()));
    assert!(has(&e), "configured model advertises the tool");
    e.set_image_source(Some(test_generator()));
    assert_eq!(
        e.tools_openai
            .iter()
            .filter(|t| t["function"]["name"].as_str() == Some("generate_image"))
            .count(),
        1,
        "idempotent — never a duplicate spec"
    );
    e.set_image_source(None);
    assert!(!has(&e), "clearing the model removes the tool");
}

#[test]
fn preview_support_gates_preview_tool() {
    let mut e = AgentEngine::new("/tmp", "deepseek-v4", "", &[], &[], 0, 0);
    let has = |e: &AgentEngine| {
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"].as_str() == Some("preview"))
    };
    assert!(!has(&e), "headless default → no tool");
    e.set_preview_supported(true);
    assert!(has(&e), "capable client advertises the tool");
    e.set_preview_supported(true);
    assert_eq!(
        e.tools_openai
            .iter()
            .filter(|t| t["function"]["name"].as_str() == Some("preview"))
            .count(),
        1,
        "idempotent — never a duplicate spec"
    );
    e.set_preview_supported(false);
    assert!(!has(&e), "losing the capability removes the tool");
}

#[test]
fn gemini_keeps_local_web_search_not_native_server_tool() {
    // Gemini 400s on google_search + function tools, and the agent always has function tools.
    let e = AgentEngine::new("/tmp", "gemini-2.5-flash", "", &[], &[], 0, 0);
    assert!(
        e.tools_openai
            .iter()
            .any(|t| t["function"]["name"].as_str() == Some("web_search")),
        "gemini keeps the local web_search function tool"
    );
    assert!(
        !e.tools_openai
            .iter()
            .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("web_search")),
        "gemini must not carry the native web_search server tool"
    );
}

#[test]
fn skills_wire_into_tools_and_system_prompt() {
    let skill = Skill {
        name: "demo".to_string(),
        description: "does a demo".to_string(),
        body: "BODY".to_string(),
        dir: PathBuf::from("/tmp/demo"),
    };
    let engine = AgentEngine::new("/tmp", "m", "", &[], std::slice::from_ref(&skill), 0, 0);

    // The `skill` tool is offered alongside the built-ins.
    let tool_names: Vec<&str> = engine
        .tools_openai
        .iter()
        .filter_map(|t| t["function"]["name"].as_str())
        .collect();
    assert!(tool_names.contains(&"skill"));

    // The system prompt advertises the skill (name + description).
    let system = engine.messages[0]["content"].as_str().unwrap();
    assert!(system.contains("demo"));
    assert!(system.contains("does a demo"));
}

#[test]
fn no_skill_tool_without_skills() {
    let engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let tool_names: Vec<&str> = engine
        .tools_openai
        .iter()
        .filter_map(|t| t["function"]["name"].as_str())
        .collect();
    assert!(!tool_names.contains(&"skill"));
}

#[test]
fn append_system_context_lands_in_system_prompt_only() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.append_system_context("# aivo context\n\nprior session facts");

    let sys = &engine.outgoing_messages()[0];
    assert_eq!(role(sys), "system");
    assert!(
        sys["content"]
            .as_str()
            .unwrap()
            .ends_with("# aivo context\n\nprior session facts")
    );
    assert!(engine.export_conversation().is_empty());

    engine.append_system_context("");
    let unchanged = engine.outgoing_messages()[0]["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(unchanged.ends_with("prior session facts"));
}

#[test]
fn agent_tools_off_strips_system_prompt() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.push_text_turn("user", "hi".into());

    assert!(engine.agent_tools_enabled);
    assert_eq!(role(&engine.outgoing_messages()[0]), "system");

    engine.set_agent_tools_enabled(false);
    let out = engine.outgoing_messages();
    assert!(out.iter().all(|m| role(m) != "system"));
    assert_eq!(role(&out[0]), "user");

    engine.set_agent_tools_enabled(true);
    assert_eq!(role(&engine.outgoing_messages()[0]), "system");
}

/// The client half of `/v1/generate-image`: tool call → gateway POST → saved
/// bytes → a path the TUI can preview.
#[tokio::test]
async fn gateway_generate_image_saves_the_returned_bytes() {
    use super::helpers::*;
    let _guard = crate::services::image_generate::TEST_GENERATE_LOCK
        .lock()
        .await;
    let dir = tmp();

    // Fake gateway, `{images:[{media_type,data}]}`.
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let seen_body = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let seen = seen_body.clone();
    std::thread::spawn(move || {
        use std::io::{Read, Write};
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let n = sock.read(&mut buf).unwrap_or(0);
        *seen.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).to_string();
        let body = format!(
            r#"{{"images":[{{"media_type":"image/png","data":"{png}"}}],"model":"banana"}}"#
        );
        let _ = sock.write_all(
            format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        );
    });
    unsafe {
        std::env::set_var(
            "AIVO_GENERATE_IMAGE_ENDPOINT",
            format!("http://{addr}/v1/generate-image"),
        );
    }

    let call = tool_call_sse("generate_image", serde_json::json!({"prompt": "a red dot"}));
    let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_image_source(Some(
        crate::services::image_generate::GeneratorSource::Gateway,
    ));
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("draw a red dot".into()),
        &mut ui,
    )
    .await;
    unsafe {
        std::env::remove_var("AIVO_GENERATE_IMAGE_ENDPOINT");
    }

    assert_eq!(ui.tools, vec!["generate_image"]);
    assert!(
        ui.tool_errors.is_empty(),
        "gateway call failed: {:?}",
        ui.tool_errors
    );
    assert!(
        seen_body
            .lock()
            .unwrap()
            .contains("\"prompt\":\"a red dot\""),
        "the prompt must reach the gateway body"
    );
    let saved = tool_result_texts(&engine)
        .into_iter()
        .find(|t| t.contains("[image saved:"))
        .expect("the tool result carries the saved path");
    let path = saved
        .split("[image saved: ")
        .nth(1)
        .and_then(|s| s.split(" (").next())
        .expect("path in the note");
    assert!(
        std::path::Path::new(path).exists(),
        "decoded bytes landed on disk: {path}"
    );
}
