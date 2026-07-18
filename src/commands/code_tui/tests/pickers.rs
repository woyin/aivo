use super::super::*;
use super::helpers::*;
use chrono::Duration as ChronoDuration;

#[test]
fn test_session_picker_item_line_fits_mixed_width_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "deepseek".to_string(),
        updated_at: (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339(),
        title: "hi".to_string(),
        preview_text: "hi · Hi there! ✨ 想聊点什么？还是需要我帮忙呢？ 我随时待命～ 😊🌟"
            .to_string(),
        origin: None,
    };

    let line = session_picker_item_lines(&preview, true, false, 64)
        .into_iter()
        .next()
        .unwrap();
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 64);
}

#[test]
fn test_key_picker_item_line_fits_modal_width() {
    let key = ApiKey::new_with_protocol(
        "deepseek".to_string(),
        "deepseek".to_string(),
        "https://api.cloudflare.com/client/v4/accounts/long/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let line = key_picker_item_line(&key, true, 36);
    let plain = plain_text_from_spans(&line.spans);
    assert!(display_width(&plain) <= 36);
    assert!(plain.contains("deepseek"));
}

#[test]
fn test_key_search_text_uses_host_not_full_path() {
    let key = ApiKey::new_with_protocol(
        "testgw".to_string(),
        "testgw".to_string(),
        "https://api.ai.example-gateway.net/endpoint".to_string(),
        None,
        "sk-test".to_string(),
    );

    let search = key_search_text(&key);
    assert!(search.contains("testgw"));
    assert!(search.contains("api.ai.example-gateway.net"));
    assert!(!search.contains("/endpoint"));
}

#[test]
fn test_key_filter_does_not_match_across_full_url_path() {
    let unrelated = "groq groq api.groq.com";
    let target = "testgw testgw api.ai.example-gateway.net";

    assert!(matches_fuzzy("gapn", target));
    assert!(!matches_fuzzy("gapn", unrelated));
}

#[test]
fn test_picker_visible_items_track_selection_for_single_line_rows() {
    let picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..6)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 4,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    let visible = picker.visible_items(3);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 2);
    assert_eq!(visible[2].0, 4);
}

#[test]
fn test_picker_navigation_wraps() {
    let mut picker = PickerState {
        title: "Select model",
        query: String::new(),
        items: (0..3)
            .map(|index| PickerEntry {
                label: format!("item-{index}"),
                search_text: format!("item-{index}"),
                value: PickerValue::Model(format!("item-{index}")),
            })
            .collect(),
        loading: false,
        selected: 0,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    picker.select_prev();
    assert_eq!(picker.selected, 2);

    picker.select_next();
    assert_eq!(picker.selected, 0);
}

#[test]
fn foreign_import_preview_renders_source_badge_no_model() {
    use crate::services::session_import::{ImportableSession, SessionOrigin};
    let imp = ImportableSession {
        origin: SessionOrigin {
            cli: "claude".to_string(),
            foreign_id: "abc-123".to_string(),
            source_path: "/x/abc-123.jsonl".to_string(),
        },
        title: "fix the login bug".to_string(),
        updated_at: Utc::now(),
        aivo_id: "import-claude-abc-123".to_string(),
    };
    let preview = SessionPreview::from_importable(imp);
    assert_eq!(preview.session_id, "import-claude-abc-123");
    assert!(preview.origin.is_some());

    // The row is prefixed with the source tag; the topic follows.
    let render = |p: &SessionPreview| -> String {
        session_picker_item_lines(p, false, false, 80)
            .iter()
            .flat_map(|l| l.spans.iter())
            .map(|s| s.content.as_ref().to_string())
            .collect()
    };
    // A not-yet-opened foreign row is [Claude], not a fork.
    let text = render(&preview);
    assert!(!preview.is_fork());
    assert!(text.contains("[Claude]"), "row: {text:?}");
    assert!(!text.contains('↳'), "row: {text:?}");
    assert!(text.contains("fix the login bug"), "row: {text:?}");

    // A keyless foreign row omits the model metadata segment.
    let (_, key_value, model) = super::super::storage::resume_metadata_values(&preview, 80);
    assert_eq!(key_value, "Claude");
    assert!(model.is_none());

    // A native session is tagged too — [aivo], not [Claude]/[Codex].
    let mut native = preview.clone();
    native.origin = None;
    native.session_id = "9f8e-native".to_string();
    native.preview_text = "deploy the gateway".to_string();
    let native_text = render(&native);
    assert!(!native.is_fork());
    assert!(native_text.contains("[aivo]"), "row: {native_text:?}");
    assert!(!native_text.contains("[Claude]"), "row: {native_text:?}");

    // A persisted import (native row, origin dropped) is always a fork under
    // lazy import — it only got persisted because a real turn was taken.
    let mut imported = preview.clone();
    imported.origin = None;
    assert_eq!(imported.session_id, "import-claude-abc-123");
    assert!(imported.is_fork());
    assert!(render(&imported).contains("[Claude ↳]"));
}

#[tokio::test]
async fn foreign_row_previews_from_transcript_without_error() {
    use crate::services::session_import::{ImportableSession, SessionOrigin};
    let dir = tempfile::tempdir().unwrap();
    let jsonl = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello there friend\"}}\n{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hi back\"}]}}\n";
    let src = dir.path().join("f.jsonl");
    std::fs::write(&src, jsonl).unwrap();
    let preview = SessionPreview::from_importable(ImportableSession {
        origin: SessionOrigin {
            cli: "claude".to_string(),
            foreign_id: "zzz".to_string(),
            source_path: src.to_string_lossy().to_string(),
        },
        title: "hello there friend".to_string(),
        updated_at: Utc::now(),
        aivo_id: "import-claude-zzz".to_string(),
    });
    // The aivo session doesn't exist yet — highlighting must preview from the
    // source transcript, not raise "Saved session is no longer available".
    let store =
        crate::services::session_store::SessionStore::with_path(dir.path().join("config.json"));
    let (messages, _truncated) = super::super::storage::load_preview_for(&store, &preview, 10)
        .await
        .expect("foreign preview should not error");
    assert!(messages.iter().any(|m| m.role == "user"));
    assert!(messages.iter().any(|m| m.role == "assistant"));
}

#[test]
fn test_picker_visible_items_respect_single_line_session_rows() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
        origin: None,
    };
    let picker = PickerState {
        title: "Resume",
        query: String::new(),
        items: vec![
            PickerEntry {
                label: "one".to_string(),
                search_text: "one".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "two".to_string(),
                search_text: "two".to_string(),
                value: PickerValue::Session(preview.clone()),
            },
            PickerEntry {
                label: "three".to_string(),
                search_text: "three".to_string(),
                value: PickerValue::Session(preview),
            },
        ],
        loading: false,
        selected: 2,
        kind: PickerKind::Session,
        pending_delete: None,
        preview_scroll: 0,
        preview_scroll_for: None,
    };

    let visible = picker.visible_items(4);
    assert_eq!(visible.len(), 3);
    assert_eq!(visible[0].0, 0);
    assert_eq!(visible[2].0, 2);
}

#[test]
fn test_session_picker_header_targets_newest_session() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
        origin: None,
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
        origin: None,
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 8, 48);
    let first = plain_text_from_spans(&lines[0].spans);

    assert_eq!(row_map.first().copied(), Some(Some(0)));
    assert!(!first.contains("Newest chat"));
    assert_eq!(row_map.get(1).copied(), Some(Some(0)));
}

#[test]
fn test_grouped_session_picker_short_view_shows_selected_session_row() {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "newest".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
        origin: None,
    };
    let older = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "older".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
        origin: None,
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );

    let (lines, row_map) = render_session_picker_rows(&picker, 1, 48);
    let only = plain_text_from_spans(&lines[0].spans);

    assert!(only.contains("Newest chat"));
    assert_eq!(row_map, vec![Some(0)]);
}

fn session_picker_fixture() -> (PickerState, SessionPreview) {
    let newest = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "sess-new".to_string(),
        raw_model: "claude".to_string(),
        updated_at: Utc::now().to_rfc3339(),
        title: "Newest".to_string(),
        preview_text: "Newest chat".to_string(),
        origin: None,
    };
    let older = SessionPreview {
        session_id: "sess-old".to_string(),
        updated_at: (Utc::now() - ChronoDuration::days(2)).to_rfc3339(),
        title: "Older".to_string(),
        preview_text: "Older chat".to_string(),
        ..newest.clone()
    };
    let picker = PickerState::ready(
        "Sessions",
        String::new(),
        vec![
            PickerEntry {
                label: newest.title.clone(),
                search_text: newest.search_text(),
                value: PickerValue::Session(newest.clone()),
            },
            PickerEntry {
                label: older.title.clone(),
                search_text: older.search_text(),
                value: PickerValue::Session(older),
            },
        ],
        PickerKind::Session,
    );
    (picker, newest)
}

fn preview_chat_message(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        model: None,
        role: role.to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    }
}

#[test]
fn test_session_preview_lines_collapses_tool_runs() {
    let messages = vec![
        preview_chat_message("user", "hello there"),
        preview_chat_message("assistant", "**hi** back"),
        preview_chat_message("tool_call", "{\"name\":\"run_bash\"}"),
        preview_chat_message("tool_result", "output"),
        preview_chat_message("tool_call", "{\"name\":\"read_file\"}"),
        preview_chat_message("assistant", "done"),
    ];
    let (lines, bars) = session_preview_lines(&messages, 60, true);
    let plain: Vec<&str> = lines.iter().map(|l| l.plain.as_str()).collect();

    assert!(
        plain[0].contains("earlier messages not shown"),
        "truncation banner missing: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("⚙ 3 tool steps")),
        "tool run should collapse to one line: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("hi back")),
        "assistant markdown missing: {plain:?}"
    );
    assert!(
        plain.iter().any(|l| l.contains("hello there")),
        "user message missing: {plain:?}"
    );
    let tool_row = plain
        .iter()
        .position(|l| l.contains("⚙ 3 tool steps"))
        .unwrap();
    assert_eq!(bars[tool_row], Some(TOOL()));
}

#[tokio::test]
async fn test_session_picker_preview_bottom_anchor_and_clamp() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, newest) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));
    let messages: Vec<ChatMessage> = (0..80)
        .map(|i| {
            preview_chat_message(
                if i % 2 == 0 { "user" } else { "assistant" },
                &format!("message number {i}"),
            )
        })
        .collect();
    app.session_preview.cache.insert(
        newest.session_id.clone(),
        PreviewEntry {
            updated_at: newest.updated_at.clone(),
            messages,
            truncated: false,
            error: None,
        },
    );

    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("message number 79"),
        "preview should anchor to the latest message:\n{screen}"
    );
    assert!(
        !screen.contains("message number 0 "),
        "oldest message should be scrolled out:\n{screen}"
    );

    // Home over-scrolls to u16::MAX; the renderer clamps to the oldest line.
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("message number 0 "),
        "Home should land on the oldest loaded message:\n{screen}"
    );
    assert!(!screen.contains("message number 79"));
    if let Overlay::Picker(p) = &app.overlay {
        assert!(p.preview_scroll < u16::MAX);
        assert_eq!(p.preview_scroll_for.as_deref(), Some("sess-new"));
    } else {
        panic!("picker vanished");
    }
}

#[tokio::test]
async fn test_session_preview_loaded_caches_even_when_not_selected() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, _) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    app.tx
        .send(RuntimeEvent::SessionPreviewLoaded {
            session_id: "sess-old".to_string(),
            entry: PreviewEntry {
                updated_at: "t1".to_string(),
                messages: vec![preview_chat_message("user", "old hello")],
                truncated: false,
                error: None,
            },
        })
        .unwrap();
    assert!(app.handle_runtime_events().await.unwrap());
    assert!(app.session_preview.cache.contains_key("sess-old"));
}

#[tokio::test]
async fn test_tick_session_preview_debounce_and_invalidation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, newest) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    // No split pane rendered yet → nothing scheduled.
    assert!(!app.tick_session_preview());
    assert!(app.session_preview.pending.is_none());

    // Split active: the first tick arms the debounce, the next is not yet due.
    app.overlay_detail_area = Some(Rect::new(0, 0, 40, 20));
    assert!(!app.tick_session_preview());
    assert!(app.session_preview.pending.is_some());
    assert!(!app.tick_session_preview());
    assert!(app.session_preview.task.is_none());

    // Once due, exactly one load task spawns and the pending slot clears.
    if let Some((_, due)) = &mut app.session_preview.pending {
        *due = Instant::now() - Duration::from_millis(1);
    }
    assert!(app.tick_session_preview());
    assert!(app.session_preview.task.is_some());
    assert!(app.session_preview.pending.is_none());

    // A valid cache entry (matching updated_at) suppresses any reload…
    app.session_preview.task = None;
    app.session_preview.cache.insert(
        newest.session_id.clone(),
        PreviewEntry {
            updated_at: newest.updated_at.clone(),
            messages: vec![],
            truncated: false,
            error: None,
        },
    );
    assert!(!app.tick_session_preview());
    assert!(app.session_preview.pending.is_none());

    // …while a stale one (index row updated since) re-arms the debounce.
    app.session_preview
        .cache
        .get_mut(&newest.session_id)
        .unwrap()
        .updated_at = "stale".to_string();
    assert!(!app.tick_session_preview());
    assert!(app.session_preview.pending.is_some());
}

#[tokio::test]
async fn test_session_picker_split_click_routes_left_rows_only() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (picker, _) = session_picker_fixture();
    app.overlay = Overlay::Picker(Box::new(picker));

    let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let detail = app.overlay_detail_area.expect("split active");
    let hitbox = app.picker_hitbox.clone().expect("hitbox recorded");

    // A click inside the preview pane neither activates nor closes.
    let mut click = wheel(MouseEventKind::Down(MouseButton::Left));
    click.column = detail.x + 2;
    click.row = detail.y + 2;
    app.handle_mouse(click).await.unwrap();
    assert!(matches!(&app.overlay, Overlay::Picker(_)));

    // A click on a mapped list row resumes that session (picker closes).
    let row = hitbox
        .row_to_filtered_index
        .iter()
        .position(|idx| idx.is_some())
        .expect("a clickable row") as u16;
    let mut click = wheel(MouseEventKind::Down(MouseButton::Left));
    click.column = hitbox.list_area.x + 1;
    click.row = hitbox.list_area.y + row;
    app.handle_mouse(click).await.unwrap();
    assert!(
        !matches!(&app.overlay, Overlay::Picker(_)),
        "row click should activate the session"
    );
    assert!(app.loading_resume.is_some());
}

#[test]
fn test_session_preview_uses_last_user_message() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session".to_string(),
        raw_model: "claude".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: session_title_from_messages(
            &[
                ChatMessage {
                    model: None,
                    role: "assistant".to_string(),
                    content: "Hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "What is the deployment status for api gateway?".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
            "claude",
        ),
        preview_text: "What is the deployment status for api gateway?".to_string(),
        origin: None,
    };

    assert_eq!(
        preview.title,
        "What is the deployment status for api gateway?".to_string()
    );
}

#[test]
fn test_session_preview_text_uses_two_latest_turns() {
    let preview = session_preview_text_from_messages(
        &[
            ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
            ChatMessage {
                model: None,
                role: "assistant".to_string(),
                content: "hi there".to_string(),
                reasoning_content: None,
                attachments: vec![],
            },
        ],
        "claude",
    );

    assert_eq!(preview, "hello · hi there");
}

#[test]
fn test_resume_metadata_spans_drop_labels_and_id() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text: "Deploy status for api gateway after rollout".to_string(),
        origin: None,
    };

    let plain = plain_text_from_spans(&resume_metadata_spans(&preview, 40));
    assert!(plain.contains("2h"));
    assert!(plain.contains("prod"));
    assert!(plain.contains("claude"));
    assert!(!plain.contains("time"));
    assert!(!plain.contains("key"));
    assert!(!plain.contains("model"));
    assert!(!plain.contains("session-1"));
}

#[test]
fn test_session_picker_item_line_shows_two_turn_preview() {
    let preview = SessionPreview {
        key_id: "key-1".to_string(),
        key_name: "prod".to_string(),
        base_url: "https://api.example.com".to_string(),
        session_id: "session-1234".to_string(),
        raw_model: "claude-sonnet-4-extended".to_string(),
        updated_at: (Utc::now() - ChronoDuration::hours(2)).to_rfc3339(),
        title: "Deploy status".to_string(),
        preview_text:
            "What is the deployment status for api gateway after the canary rollout finished?"
                .to_string(),
        origin: None,
    };

    let lines = session_picker_item_lines(&preview, false, false, 32);
    let first = plain_text_from_spans(&lines[0].spans);

    assert!(first.contains("What is"));
    assert!(first.chars().any(|ch| ch.is_ascii_digit()));
    assert!(!first.contains("key"));
}
