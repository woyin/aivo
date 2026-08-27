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

/// The `/config` Thinking row is one scale — off, then the catalog levels.
/// Picking a level turns thinking on; picking off remembers the level.
#[tokio::test]
async fn thinking_config_row_is_one_scale_with_off() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "m".to_string();
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];
    app.thinking_enabled = true;
    app.reasoning_effort = Some("medium".into());

    let segs = app.config_segments(ConfigSetting::Thinking);
    assert_eq!(segs.options, &["off", "low", "medium", "high"]);
    assert_eq!(segs.active, 2, "medium = level index 1, after off");

    app.open_config_overlay_at(ConfigSetting::Thinking);
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let row = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::Thinking)
        .expect("Thinking row present");

    app.step_config_setting(row, 1).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(app.thinking_enabled);

    // All the way left is off — the level stays remembered for re-enable.
    app.step_config_setting(row, -3).await;
    assert!(!app.thinking_enabled);
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 0);

    // Picking a level from off turns thinking back on.
    app.step_config_setting(row, 2).await;
    assert!(app.thinking_enabled);
    assert_eq!(app.reasoning_effort.as_deref(), Some("medium"));
}

#[test]
fn test_footer_effort_label_reports_thinking_off_on_capable_models() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.model_supports_thinking = false;
    app.thinking_enabled = false;
    assert_eq!(app.footer_effort_label(), None);

    app.model_supports_thinking = true;
    assert_eq!(app.footer_effort_label().as_deref(), Some("off"));

    app.thinking_enabled = true;
    assert_ne!(app.footer_effort_label().as_deref(), Some("off"));

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

/// A stale menu after a model switch must not apply a foreign level.
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

/// A binary catalog ([none, high], poolside laguna-style) must default to the
/// thinking side: the provider ships thinking ON by default, so the off level
/// can't win the middle-pick fallback.
#[tokio::test]
async fn effective_effort_default_skips_native_off() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "laguna-s-2.1".to_string();
    app.model_reasoning_efforts = vec!["none".into(), "high".into()];
    app.thinking_enabled = true;
    app.reasoning_effort = None;

    assert_eq!(app.effective_reasoning_effort().as_deref(), Some("high"));
    assert_eq!(
        app.config_segments(ConfigSetting::Thinking).active,
        1,
        "the row highlights high, not none"
    );

    // A wider catalog missing the default level skips off-equivalents too:
    // the middle of [low, high, max] is high, not the list's byte-middle.
    app.model_reasoning_efforts = vec!["minimal".into(), "low".into(), "high".into(), "max".into()];
    assert_eq!(app.effective_reasoning_effort().as_deref(), Some("high"));
}

/// gpt-5.4+ with agent tools on has NO off state — the provider refuses
/// tools + none, so the scale is levels-only and stale off choices clamp to
/// the lowest level that actually ships.
#[tokio::test]
async fn thinking_scale_drops_off_when_tools_forbid_it() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "gpt-5.6-sol".to_string();
    app.model_reasoning_efforts = ["none", "low", "medium", "high"].map(String::from).to_vec();
    app.agent_tools_enabled = true;
    app.thinking_enabled = true;

    let segs = app.config_segments(ConfigSetting::Thinking);
    assert_eq!(segs.options, &["low", "medium", "high"], "no off, no none");
    assert_eq!(segs.active, 1, "default medium");

    // An off request collapses into the toggle; with tools it can't ship, so
    // the footer shows the substitute the engine sends, not "thinking off".
    app.apply_reasoning_effort("none".to_string()).await;
    assert!(!app.thinking_enabled);
    assert!(
        app.reasoning_effort.is_none(),
        "off is never a stored level"
    );
    app.model_supports_thinking = true;
    assert_eq!(app.footer_effort_label().as_deref(), Some("low"));
    app.thinking_enabled = true;

    // Ctrl+T rings the levels alone: high wraps to low, thinking stays on.
    app.reasoning_effort = Some("high".into());
    app.cycle_reasoning_effort().await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("low"));
    assert!(app.thinking_enabled);

    // Plain chat (tools off) brings the off pill back — as aivo's word.
    app.agent_tools_enabled = false;
    assert_eq!(
        app.config_segments(ConfigSetting::Thinking).options,
        &["off", "low", "medium", "high"]
    );
}

/// A catalog spelling its off as `none`/`minimal` surfaces the unified `off`
/// pill — the provider's off word never appears as a level; the engine's
/// translation (`thinking_off_wire`) sends the right spelling.
#[tokio::test]
async fn thinking_scale_unifies_native_off_spellings() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "m".to_string();
    app.model_reasoning_efforts = vec!["none".into(), "low".into(), "high".into()];
    app.thinking_enabled = true;
    app.reasoning_effort = Some("low".into());

    let segs = app.config_segments(ConfigSetting::Thinking);
    assert_eq!(segs.options, &["off", "low", "high"], "one off vocabulary");
    assert_eq!(segs.active, 1);

    // Thinking off highlights the off pill.
    app.thinking_enabled = false;
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 0);
    app.thinking_enabled = true;

    // The ring passes through off: high wraps to off, off resumes at low.
    app.reasoning_effort = Some("high".into());
    app.cycle_reasoning_effort().await;
    assert!(!app.thinking_enabled, "top of the scale wraps to off");
    app.cycle_reasoning_effort().await;
    assert!(app.thinking_enabled);
    assert_eq!(app.reasoning_effort.as_deref(), Some("low"));

    // Segment picks skip the off pill when mapping to levels.
    app.open_config_overlay_at(ConfigSetting::Thinking);
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let row = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::Thinking)
        .expect("Thinking row present");
    app.step_config_setting(row, 1).await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    // o-series style: off would just alias `low` on the wire — no off pill,
    // and the footer reports the `low` that actually ships when the global
    // toggle is off.
    app.model = "o3".to_string();
    app.model_reasoning_efforts = vec!["low".into(), "medium".into(), "high".into()];
    app.reasoning_effort = None;
    assert_eq!(
        app.config_segments(ConfigSetting::Thinking).options,
        &["low", "medium", "high"],
        "off→low alias would duplicate the low pill"
    );
    app.model_supports_thinking = true;
    app.thinking_enabled = false;
    assert_eq!(app.footer_effort_label().as_deref(), Some("low"));
    app.thinking_enabled = true;
}

/// Ctrl+T steps one ring — off → low → … → high → off. A thinking-capable
/// model without levels just toggles; a model with no thinking gets a notice.
#[tokio::test]
async fn test_cycle_reasoning_effort_rings_through_off() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.model = "m".to_string();
    app.model_reasoning_efforts = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
    app.thinking_enabled = true;

    // No explicit choice yet: the effective default is "medium", so the ring
    // continues at "high", not the list head.
    app.cycle_reasoning_effort().await;
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));
    assert!(app.thinking_enabled);

    // Top of the scale wraps to off; the level stays remembered.
    app.cycle_reasoning_effort().await;
    assert!(!app.thinking_enabled);
    assert_eq!(app.reasoning_effort.as_deref(), Some("high"));

    // Off resumes at the lowest level.
    app.cycle_reasoning_effort().await;
    assert!(app.thinking_enabled);
    assert_eq!(app.reasoning_effort.as_deref(), Some("low"));

    // Thinking-capable but no catalog levels: a plain toggle.
    app.model_reasoning_efforts.clear();
    app.model_supports_thinking = true;
    app.cycle_reasoning_effort().await;
    assert!(!app.thinking_enabled);
    app.cycle_reasoning_effort().await;
    assert!(app.thinking_enabled);

    // No thinking at all: notice only, nothing flips.
    app.model_supports_thinking = false;
    app.cycle_reasoning_effort().await;
    assert!(app.thinking_enabled, "unchanged without thinking support");
    assert!(app.notice.as_ref().unwrap().1.contains("no thinking"));
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

fn choice(id: &str) -> ModelChoice {
    ModelChoice {
        id: id.to_string(),
        label: id.to_string(),
        image_input: None,
    }
}

#[tokio::test]
async fn test_cross_provider_switch_keeps_conversation() {
    // A saved model still routes through the picker; picking applies the
    // switch and keeps the conversation.
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

    assert!(
        matches!(
            &app.overlay,
            Overlay::Picker(p) if matches!(
                p.kind,
                PickerKind::Model {
                    target: ModelSelectionTarget::KeySwitch { .. },
                    ..
                }
            )
        ),
        "a saved model still goes through the picker"
    );
    assert_eq!(app.key.id, key_a, "switch waits for the model pick");

    app.populate_model_picker(vec![choice("model-a"), choice("model-b")]);
    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected model picker");
    };
    assert_eq!(picker.selected, 1, "saved model is focused");

    app.activate_picker_selection(1).await.unwrap();

    assert_eq!(app.key.id, key_b_id, "switch applied on pick, no confirm");
    assert_eq!(
        app.history.len(),
        4,
        "conversation preserved across providers"
    );
    assert_eq!(app.session_id, "keep-me", "same session — no reset");
}

#[tokio::test]
async fn test_begin_key_switch_same_provider_skips_confirm() {
    // Same provider = credential swap: the pick applies straight through, no card.
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
    app.populate_model_picker(vec![choice("model-b")]);
    app.activate_picker_selection(0).await.unwrap();

    assert_eq!(app.key.id, key_b_id, "applied on pick");
    assert_eq!(app.session_id, "keep-me", "chat preserved");
    assert_eq!(app.history.len(), 4);
}

#[tokio::test]
async fn test_key_switch_without_listing_falls_back_to_saved_model() {
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

    app.begin_key_switch(key_b_full).await.unwrap();
    app.tx
        .send(RuntimeEvent::ModelsLoaded(Err("no listing".to_string())))
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    assert_eq!(app.key.id, key_b_id, "switch fell back to the saved model");
    assert_eq!(app.raw_model, "model-b");
}

#[tokio::test]
async fn test_key_picker_focuses_active_key() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    store
        .add_key_with_protocol("a", "https://a.example.com", None, "sk-a")
        .await
        .unwrap();
    let key_b_id = store
        .add_key_with_protocol("b", "https://b.example.com", None, "sk-b")
        .await
        .unwrap();
    let key_b_full = store.get_key_by_id(&key_b_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key_b_full;

    app.open_key_picker(None).await.unwrap();

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected key picker");
    };
    assert_eq!(picker.selected, 1, "active key is focused");
}

#[tokio::test]
async fn test_model_picker_focuses_model_in_use() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "model-two".to_string();

    app.open_model_picker(None, ModelSelectionTarget::CurrentChat, false);
    app.populate_model_picker(vec![
        choice("model-one"),
        choice("model-two"),
        choice("model-three"),
    ]);

    let Overlay::Picker(picker) = &app.overlay else {
        panic!("expected model picker");
    };
    assert_eq!(picker.selected, 1, "model in use is focused");
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
        image_input: None,
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
