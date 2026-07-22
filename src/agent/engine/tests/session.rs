use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn parse_slash_classifies() {
    assert!(parse_slash("hello").is_none());
    assert!(matches!(parse_slash("/help"), Some(SlashCmd::Help)));
    assert!(matches!(parse_slash("  /clear "), Some(SlashCmd::Clear)));
    assert!(matches!(parse_slash("/?"), Some(SlashCmd::Help)));
    assert!(matches!(parse_slash("/bogus"), Some(SlashCmd::Unknown(_))));
}

#[test]
fn chat_session_context_injects_facts_and_tools() {
    let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
    assert!(!system_content(&e).contains("interactive `aivo code` session"));
    assert!(!tool_names(&e).contains(&"switch_model".to_string()));

    e.set_chat_session_context(ChatSessionContext {
        model_label: "gpt-5".to_string(),
        provider_label: "openrouter".to_string(),
        effort: Some("high".to_string()),
        effort_levels: vec!["low".into(), "medium".into(), "high".into()],
    });
    let p = system_content(&e);
    assert!(p.contains("interactive `aivo code` session"));
    assert!(p.contains("gpt-5") && p.contains("openrouter") && p.contains("high"));
    assert!(p.contains("/model") && p.contains("switch_model") && p.contains("/key"));
    let names = tool_names(&e);
    assert!(names.contains(&"switch_model".to_string()));
    assert!(names.contains(&"set_effort".to_string()));
    assert!(names.contains(&"ask_user".to_string()));
    // single-system-message invariant + idempotent
    assert_eq!(e.messages.len(), 1);
    e.set_chat_session_context(ChatSessionContext {
        model_label: "x".into(),
        provider_label: "y".into(),
        effort: None,
        effort_levels: vec![],
    });
    assert_eq!(
        tool_names(&e)
            .iter()
            .filter(|n| *n == "switch_model")
            .count(),
        1
    );
}

#[test]
fn chat_session_context_reports_no_effort_levels() {
    let mut e = AgentEngine::new("/tmp", "some-model", "", &[], &[], 0, 0);
    e.set_chat_session_context(ChatSessionContext {
        model_label: "some-model".into(),
        provider_label: "prov".into(),
        effort: None,
        effort_levels: vec![],
    });
    assert!(system_content(&e).contains("no reasoning-effort levels"));
}

#[tokio::test]
async fn session_control_tools_route_to_ui_callbacks() {
    #[derive(Default)]
    struct SwitchUi {
        switched: Vec<String>,
        efforts: Vec<String>,
        asked: Vec<(String, Vec<String>, bool, bool)>,
    }
    impl AgentUi for SwitchUi {
        fn assistant_text(&mut self, _: &str) {}
        fn tool_start(&mut self, _: &str, _: &Value) {}
        fn tool_result(&mut self, _: &str, _: &Result<String, String>) {}
        fn notify(&mut self, _: &str) {}
        fn footer(&mut self, _: Option<&str>, _: usize, _: u64, _: u64, _: u64) {}
        fn ask_permission<'a>(
            &'a mut self,
            _: &'a str,
            _: Option<&'a str>,
            _: bool,
        ) -> BoxFuture<'a, Decision> {
            Box::pin(async { Decision::Allow })
        }
        fn switch_chat_model<'a>(
            &'a mut self,
            model: &'a str,
        ) -> BoxFuture<'a, Result<String, String>> {
            self.switched.push(model.to_string());
            Box::pin(async { Ok("switched".to_string()) })
        }
        fn set_chat_effort<'a>(
            &'a mut self,
            level: &'a str,
        ) -> BoxFuture<'a, Result<String, String>> {
            self.efforts.push(level.to_string());
            Box::pin(async { Ok("ok".to_string()) })
        }
        fn ask_user<'a>(
            &'a mut self,
            question: &'a str,
            options: &'a [crate::agent::ask::AskOption],
            allow_free_text: bool,
            multi_select: bool,
        ) -> BoxFuture<'a, Result<String, String>> {
            self.asked.push((
                question.to_string(),
                options.iter().map(|o| o.label.clone()).collect(),
                allow_free_text,
                multi_select,
            ));
            let answer = options.first().map(|o| o.label.clone()).unwrap_or_default();
            Box::pin(async move { Ok(answer) })
        }
    }

    let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
    let client = reqwest::Client::new();
    let cwd = std::path::Path::new(".");
    let ctx = turn_ctx(&client, "", cwd);
    let mut ui = SwitchUi::default();
    let calls = vec![
        ToolCall {
            id: "1".into(),
            name: "switch_model".into(),
            arguments: json!({"model": "opus"}),
        },
        ToolCall {
            id: "2".into(),
            name: "set_effort".into(),
            arguments: json!({"level": "high"}),
        },
        ToolCall {
            id: "3".into(),
            name: "ask_user".into(),
            arguments: json!({
                "question": "Add release notes now?",
                "options": [{"label": "You add them"}, {"label": "Auto-generate"}],
                "allow_free_text": false
            }),
        },
    ];
    e.execute_tool_batch(&ctx, &mut ui, &calls).await;
    assert_eq!(ui.switched, vec!["opus".to_string()]);
    assert_eq!(ui.efforts, vec!["high".to_string()]);
    assert_eq!(
        ui.asked,
        vec![(
            "Add release notes now?".to_string(),
            vec!["You add them".to_string(), "Auto-generate".to_string()],
            false, // allow_free_text
            false, // multi_select
        )]
    );

    // The default trait impl (non-chat host) declines.
    let mut plain = CapturingUi::default();
    let declined = plain.switch_chat_model("opus").await.unwrap_err();
    assert!(declined.contains("interactive"));
}
