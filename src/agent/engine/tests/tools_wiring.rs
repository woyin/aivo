use super::super::*;
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
