use super::super::*;
use super::helpers::*;
use tempfile::TempDir;

#[tokio::test]
async fn prewarm_cursor_session_noops_for_non_cursor_key() {
    // Non-cursor key => prewarm must not spawn cursor-agent or arm the handle.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.key.is_cursor_acp());
    app.prewarm_cursor_session();
    assert!(app.cursor_prewarm.is_none());
}

#[tokio::test]
async fn test_session_pricing_falls_back_to_billed_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "aivo/starter".to_string();
    assert!(app.session_pricing().is_none(), "alias alone is unpriced");
    app.billed_model = Some("claude-opus-4-8".to_string());
    assert!(app.session_pricing().is_some(), "billed model resolves");
}

#[tokio::test]
async fn effort_command_sets_level_enables_thinking_and_validates() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];
    app.thinking_enabled = false;
    app.reasoning_effort = None;

    // `/effort high` also turns thinking on — picking an effort implies you want the model to reason.
    app.run_effort_command(Some("HIGH".into())).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(app.thinking_enabled, "setting effort must turn thinking on");

    // An unknown level errors and leaves the choice unchanged.
    app.run_effort_command(Some("bogus".into())).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(app.notice.as_ref().is_some_and(|(c, _)| *c == ERROR()));

    // Bare `/effort` opens the picker of the model's levels.
    app.run_effort_command(None).await;
    assert!(
        matches!(&app.overlay, Overlay::Picker(p) if matches!(p.kind, PickerKind::Effort)),
        "bare /effort opens the effort picker"
    );
}

#[tokio::test]
async fn effort_command_noop_when_model_has_no_levels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model_reasoning_efforts.clear();
    app.run_effort_command(None).await;
    assert!(
        matches!(app.overlay, Overlay::None),
        "no picker without levels"
    );
    assert!(app.notice.as_ref().is_some_and(|(c, _)| *c == MUTED()));
}

#[test]
fn test_footer_effort_label_reports_thinking_off_on_capable_models() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.model_supports_thinking = false;
    app.thinking_enabled = false;
    assert_eq!(app.footer_effort_label(), None);

    app.model_supports_thinking = true;
    assert_eq!(app.footer_effort_label().as_deref(), Some("thinking off"));

    app.thinking_enabled = true;
    assert_ne!(app.footer_effort_label().as_deref(), Some("thinking off"));

    // A cursor-derived label wins over the local toggles.
    app.cursor_effort_label = Some("max".to_string());
    assert_eq!(app.footer_effort_label().as_deref(), Some("max"));
}

#[tokio::test]
async fn test_cursor_model_refresh_sets_window_and_effort_badge() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key = ApiKey::new_with_protocol(
        "cursor".to_string(),
        String::new(),
        "cursor".to_string(),
        None,
        String::new(),
    );

    // Claude tier → underlying-model window + tier badge.
    app.model = "claude-opus-4-8-max".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 1_000_000);
    assert_eq!(app.cursor_effort_label.as_deref(), Some("max"));

    // Cursor-native windows (not in models.dev): composer 200k, auto 2M.
    app.model = "composer-2.5".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 200_000);
    assert_eq!(app.cursor_effort_label, None);

    app.model = "auto".to_string();
    app.refresh_context_window().await;
    assert_eq!(app.context_window, 2_000_000);
    assert_eq!(app.cursor_effort_label, None);
}

/// A level not offered by the current model is refused at apply time — a stale
/// effort picker across an agent-driven model switch must not 400 later turns.
#[tokio::test]
async fn test_apply_reasoning_effort_rejects_foreign_level() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "m".to_string();
    app.model_reasoning_efforts = vec!["low".to_string(), "high".to_string()];

    app.apply_reasoning_effort("xhigh".to_string()).await;
    assert!(app.reasoning_effort.is_none(), "foreign level refused");
    assert!(app.notice.as_ref().unwrap().1.contains("isn't a level"));

    app.apply_reasoning_effort("high".to_string()).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
}

/// Opening the model picker mid-turn must NOT cancel the in-flight turn (it
/// used to): the running turn keeps its model and the pick applies next turn,
/// same as the agent's `switch_model` tool.
#[tokio::test]
async fn test_open_model_picker_keeps_inflight_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "draft".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.pending_response = "partial".to_string();
    app.sending = true;
    app.request_started_at = Some(Instant::now());

    app.open_model_picker(None, ModelSelectionTarget::CurrentChat, false);

    assert!(app.sending, "the in-flight turn must keep running");
    assert_eq!(app.pending_response, "partial");
    assert_eq!(
        app.history.len(),
        1,
        "the user turn stays in the transcript"
    );
    assert!(matches!(app.overlay, Overlay::Picker(_)));
}

/// `/model <name>` applies the name directly, opening no picker.
#[tokio::test]
async fn test_model_command_applies_name_directly() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.set_model_direct("my-model".to_string()).await.unwrap();

    assert_eq!(app.raw_model, "my-model");
    assert!(matches!(app.overlay, Overlay::None));
    let (color, msg) = app.notice.as_ref().expect("a confirmation notice");
    assert_eq!(*color, MUTED());
    assert!(msg.contains("my-model"), "notice names the model: {msg}");
}

#[tokio::test]
async fn test_apply_model_updates_last_selection_preserving_tool() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Seed a prior launchable selection so we can assert the tool is preserved
    // (a `/model` switch must not overwrite it with "code").
    store
        .set_last_selection(&key, "claude", Some("old-model"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("new-model".to_string()).await.unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_id);
    assert_eq!(sel.model.as_deref(), Some("new-model"));
    assert_eq!(sel.tool, "claude", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_complete_key_switch_updates_last_selection() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();
    store
        .set_last_selection(&key_a_full, "codex", Some("model-a"))
        .await
        .unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    let sel = store.get_last_selection().await.unwrap().unwrap();
    assert_eq!(sel.key_id, key_b_id, "switched-to key must be selected");
    assert_eq!(sel.model.as_deref(), Some("model-b"));
    assert_eq!(sel.tool, "codex", "launchable tool must be preserved");
}

#[tokio::test]
async fn test_complete_key_switch_same_provider_preserves_chat() {
    // Same base_url = credential swap → chat survives.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("personal", "https://same.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("work", "https://same.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "keep-me".to_string();
    seed_two_exchanges(&mut app);

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    assert_eq!(app.key.id, key_b_id, "switched to the new key");
    assert_eq!(
        app.session_id, "keep-me",
        "same-provider switch keeps the session"
    );
    assert_eq!(app.history.len(), 4, "conversation is preserved");
}

#[tokio::test]
async fn test_complete_key_switch_different_provider_keeps_chat() {
    // A different provider keeps the conversation — it replays on the new
    // provider (OpenAI-wire transcript bridged by aivo serve), same session.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "old-session".to_string();
    seed_two_exchanges(&mut app);

    app.complete_key_switch(key_b_full, "model-b".to_string())
        .await
        .unwrap();

    assert_eq!(app.key.id, key_b_id, "switched to the new key");
    assert_eq!(app.history.len(), 4, "conversation preserved");
    assert_eq!(app.session_id, "old-session", "same session — no reset");
}

#[tokio::test]
async fn test_cross_provider_switch_keeps_conversation() {
    // Switching to a different-provider key applies directly and keeps the
    // conversation — no reset, no confirm. It replays on the new provider.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    store.set_code_model(&key_b_id, "model-b").await.unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "keep-me".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();

    assert_eq!(app.key.id, key_b_id, "switch applied directly, no confirm");
    assert_eq!(
        app.history.len(),
        4,
        "conversation preserved across providers"
    );
    assert_eq!(app.session_id, "keep-me", "same session — no reset");
}

#[tokio::test]
async fn test_begin_key_switch_same_provider_skips_confirm() {
    // Same provider = credential swap: apply straight through, no card.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_a = store
        .add_key_with_protocol("personal", "https://same.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("work", "https://same.example.com", None, "sk-b")
        .await
        .unwrap();
    store.set_code_model(&key_b_id, "model-b").await.unwrap();
    let key_a_full = store.get_key_by_id(&key_a).await.unwrap().unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_a_full;
    app.session_id = "keep-me".to_string();
    seed_two_exchanges(&mut app);

    app.begin_key_switch(key_b_full).await.unwrap();

    assert_eq!(app.key.id, key_b_id, "applied directly");
    assert_eq!(app.session_id, "keep-me", "chat preserved");
    assert_eq!(app.history.len(), 4);
}

#[tokio::test]
async fn test_apply_model_survives_resolved_sentinel_base_url() {
    // The live key may carry a base_url resolved away from a sentinel (ollama,
    // aivo-starter). The persisted selection must use the *stored* key's
    // base_url, or `get_last_selection` prunes it as stale.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("ollama", "ollama", None, "")
        .await
        .unwrap();
    let mut key = store.get_key_by_id(&key_id).await.unwrap().unwrap();
    // Simulate the launch-time sentinel resolution that mutates the live key.
    key.base_url = "http://localhost:11434/v1".to_string();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;

    app.apply_model("llama3".to_string()).await.unwrap();

    let sel = store
        .get_last_selection()
        .await
        .unwrap()
        .expect("selection must survive the sentinel/resolved base_url mismatch");
    assert_eq!(sel.key_id, key_id);
    assert_eq!(
        sel.base_url, "ollama",
        "stored sentinel base_url is persisted"
    );
    assert_eq!(sel.model.as_deref(), Some("llama3"));
}

#[tokio::test]
async fn test_apply_model_skips_last_selection_for_hf_synthetic_key() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = ApiKey::new_with_protocol(
        crate::services::huggingface::HF_LOCAL_KEY_ID.to_string(),
        "hf:demo".to_string(),
        "http://localhost:8080/v1".to_string(),
        None,
        "huggingface".to_string(),
    );

    app.apply_model("hf-model".to_string()).await.unwrap();

    assert!(
        store.get_last_selection().await.unwrap().is_none(),
        "ephemeral HF synthetic key must not be remembered as the selection"
    );
}

// ---- agent session-control tools (switch_model / set_effort) ----

fn model_choice(id: &str) -> ModelChoice {
    ModelChoice {
        id: id.to_string(),
        label: id.to_string(),
    }
}

#[test]
fn resolve_model_request_exact_and_unique_substring() {
    let choices = [
        model_choice("anthropic/claude-opus-4-8"),
        model_choice("openai/gpt-5"),
        model_choice("openai/gpt-5-mini"),
    ];
    // exact id wins even though it's also a substring of another
    assert_eq!(
        super::super::session_impl::resolve_model_request("OPENAI/GPT-5", &choices).unwrap(),
        "openai/gpt-5"
    );
    assert_eq!(
        super::super::session_impl::resolve_model_request("opus", &choices).unwrap(),
        "anthropic/claude-opus-4-8"
    );
}

#[test]
fn resolve_model_request_ambiguous_and_missing() {
    let choices = [
        model_choice("openai/gpt-5"),
        model_choice("openai/gpt-5-mini"),
    ];
    // substring of both, no exact "gpt-5" id → ambiguous
    let err = super::super::session_impl::resolve_model_request("gpt-5", &choices).unwrap_err();
    assert!(err.contains("ambiguous"));
    assert!(err.contains("openai/gpt-5") && err.contains("openai/gpt-5-mini"));
    let miss = super::super::session_impl::resolve_model_request("llama", &choices).unwrap_err();
    assert!(miss.contains("no model matches") && miss.contains("/model"));
    // empty catalog accepts the raw string
    assert_eq!(
        super::super::session_impl::resolve_model_request("whatever", &[]).unwrap(),
        "whatever"
    );
}

#[tokio::test]
async fn agent_set_effort_validates_against_levels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "gpt-5".to_string();
    app.model = "gpt-5".to_string();
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];

    let ok = app.agent_set_effort("High".to_string()).await.unwrap();
    assert!(ok.contains("high"));
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    // invalid level rejected, effort unchanged
    let err = app.agent_set_effort("turbo".to_string()).await.unwrap_err();
    assert!(err.contains("low, medium, high"));
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    app.model_reasoning_efforts.clear();
    let none = app.agent_set_effort("high".to_string()).await.unwrap_err();
    assert!(none.contains("no reasoning-effort levels"));
}

#[tokio::test]
async fn agent_switch_model_noops_when_already_on_it() {
    // The already-on-it short-circuit returns before any catalog fetch (no network).
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "gpt-5".to_string();
    let msg = app.agent_switch_model("GPT-5".to_string()).await.unwrap();
    assert!(msg.contains("Already using gpt-5"));
}

/// Assistant turns are stamped with their dispatch-time model, and the
/// transcript draws a `model →` divider where the stamp changes.
#[test]
fn model_switch_stamps_turns_and_renders_divider() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Turn 1 dispatched on model-a.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "first question".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_model = Some("model-a".to_string());
    // Mid-turn switch: the running turn must keep its dispatch-time stamp.
    app.raw_model = "model-b".to_string();
    app.pending_response = "answer one".to_string();
    app.flush_pending_assistant();
    assert_eq!(
        app.history.last().unwrap().model.as_deref(),
        Some("model-a")
    );

    // Turn 2 dispatched on model-b.
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "second question".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_model = Some("model-b".to_string());
    app.pending_response = "answer two".to_string();
    app.flush_pending_assistant();
    assert_eq!(
        app.history.last().unwrap().model.as_deref(),
        Some("model-b")
    );

    let body = app.build_transcript_history_body(80);
    let rows = wrap_transcript(&body.lines, &body.bar_colors, 80).rows;
    // One divider at the boundary; none above the first stamped turn.
    assert_eq!(
        rows.iter()
            .filter(|r| r.contains("model → model-b"))
            .count(),
        1
    );
    assert!(rows.iter().all(|r| !r.contains("model → model-a")));
    let first = rows.iter().position(|r| r.contains("answer one")).unwrap();
    let divider = rows
        .iter()
        .position(|r| r.contains("model → model-b"))
        .unwrap();
    let second = rows.iter().position(|r| r.contains("answer two")).unwrap();
    assert!(first < divider && divider < second);
}

/// Unstamped (pre-feature) history renders no divider.
#[test]
fn unstamped_history_renders_no_model_divider() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for (role, content) in [
        ("user", "q1"),
        ("assistant", "a1"),
        ("user", "q2"),
        ("assistant", "a2"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let body = app.build_transcript_history_body(80);
    let rows = wrap_transcript(&body.lines, &body.bar_colors, 80).rows;
    assert!(rows.iter().all(|r| !r.contains("model →")));
}

/// Dispatch freezes the selected model into `turn_model`.
#[tokio::test]
async fn test_dispatch_captures_turn_model() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Non-agent key keeps the send on the lightweight plain-chat path.
    app.key.base_url = "claude-oauth".to_string();
    app.raw_model = "model-a".to_string();

    app.dispatch_user_message("hello".to_string(), None)
        .await
        .unwrap();
    assert_eq!(app.turn_model.as_deref(), Some("model-a"));
}
