use super::super::*;
use super::helpers::*;
use tempfile::TempDir;

/// A cancelled or interrupted `/compact` must not mark the NEXT turn as a
/// compact (bogus "freed" notice, skipped logs, corrupted context stats).
#[tokio::test]
async fn test_teardown_clears_compact_before() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // ESC / /new / resume / key-switch route (cancel).
    app.sending = true;
    app.compact_before = Some(50_000);
    app.cancel_inflight_request(CancelKind::Discard);
    assert_eq!(app.compact_before, None, "cancel clears the compact flag");

    // Interrupt-with-partial-text route (skips cancel_inflight_request).
    app.sending = true;
    app.compact_before = Some(50_000);
    app.pending_response = "partial".to_string();
    app.interrupt_inflight_request().await.unwrap();
    assert_eq!(
        app.compact_before, None,
        "interrupt clears the compact flag"
    );
}

#[test]
fn test_persisted_draft_history_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("chat_history");
    let history = vec![
        DraftHistoryEntry {
            cwd: "/work/a".to_string(),
            text: "first".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/b".to_string(),
            text: "second".to_string(),
        },
    ];

    save_persisted_draft_history_to_path(&path, &history).unwrap();

    let loaded = load_persisted_draft_history_from_path(&path);
    let pairs: Vec<(String, String)> = loaded
        .into_iter()
        .map(|entry| (entry.cwd, entry.text))
        .collect();
    assert_eq!(
        pairs,
        vec![
            ("/work/a".to_string(), "first".to_string()),
            ("/work/b".to_string(), "second".to_string()),
        ]
    );
}

#[test]
fn test_draft_history_view_filters_by_cwd() {
    let all = vec![
        DraftHistoryEntry {
            cwd: String::new(),
            text: "legacy".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/a".to_string(),
            text: "in-a".to_string(),
        },
        DraftHistoryEntry {
            cwd: "/work/b".to_string(),
            text: "in-b".to_string(),
        },
    ];

    // Current dir's entries plus the legacy fallback; other dirs filtered out.
    assert_eq!(
        draft_history_view(&all, "/work/a"),
        vec!["legacy".to_string(), "in-a".to_string()]
    );
    assert_eq!(
        draft_history_view(&all, "/work/b"),
        vec!["legacy".to_string(), "in-b".to_string()]
    );
    // A fresh dir sees only the legacy fallback.
    assert_eq!(
        draft_history_view(&all, "/work/new"),
        vec!["legacy".to_string()]
    );
}

#[test]
fn test_legacy_plaintext_history_loads_untagged() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("chat_history");
    // The old writer encrypted raw newline-joined prompt lines (no JSON).
    let blob = crate::services::session_store::encrypt("old one\nold two").unwrap();
    std::fs::write(&path, blob).unwrap();

    let loaded = load_persisted_draft_history_from_path(&path);
    assert_eq!(loaded.len(), 2);
    assert!(loaded.iter().all(|entry| entry.cwd.is_empty()));
    assert_eq!(loaded[0].text, "old one");
    assert_eq!(loaded[1].text, "old two");
    // Untagged entries surface in every dir's view.
    assert_eq!(
        draft_history_view(&loaded, "/anywhere"),
        vec!["old one".to_string(), "old two".to_string()]
    );
}

#[tokio::test]
async fn test_empty_chat_persists_no_session_on_exit() {
    // Opening `aivo code` and leaving without saying anything must NOT create a
    // session — `flush_for_exit` only persists a non-empty history, so an
    // untouched chat leaves the resume list untouched.
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "untouched-sess".to_string();
    app.raw_model = "claude".to_string();
    assert!(app.history.is_empty());

    app.flush_for_exit().await;

    assert_eq!(store.count_chat_sessions().await, 0);
    assert!(
        store
            .get_code_session("untouched-sess")
            .await
            .unwrap()
            .is_none()
    );
    // Nothing to resume, so no exit hint either.
    assert_eq!(app.resumable_session_id(), None);
}

/// `session_tokens` (the running per-session total folded from each turn) is
/// written into the chat index entry, so `aivo stats --since` can attribute
/// windowed chat usage; and resuming re-seeds the running total from it.
#[tokio::test]
async fn test_persist_history_writes_session_tokens_to_index() {
    use crate::services::session_store::SessionTokens;

    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.session_id = "tok-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // A turn folded this much real usage into the session total.
    app.session_tokens = SessionTokens {
        prompt_tokens: 100,
        completion_tokens: 20,
        cache_read_tokens: 40,
        cache_write_tokens: 0,
    };
    app.persist_history().await.unwrap();

    // The windowed aggregation (what `aivo stats --since` reads) now sees them.
    let far_past = chrono::DateTime::parse_from_rfc3339("2000-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    let window = store.aggregate_chat_window_since(far_past).await;
    let total = window.total();
    assert_eq!(total.prompt_tokens, 100);
    assert_eq!(total.completion_tokens, 20);
    assert_eq!(total.cache_read_tokens, 40);

    // The getter that re-seeds the running total on resume returns the same.
    let seeded = store.chat_session_tokens("tok-sess").await;
    assert_eq!(seeded, app.session_tokens);
    assert_eq!(
        store.chat_session_tokens("nope").await,
        SessionTokens::default()
    );
}

#[tokio::test]
async fn test_log_agent_turn_records_under_real_cwd() {
    use crate::services::log_store::LogQuery;

    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/aivo-chat-1".to_string();
    app.real_cwd = "/home/me/proj".to_string();
    app.session_id = "agent-sess".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "do the thing".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.log_agent_turn(1234).await;

    // The turn shows in `aivo logs` filtered to the real project dir.
    let rows = store
        .logs()
        .list(LogQuery {
            limit: 100,
            cwd: Some("/home/me/proj".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kind, "code_turn");
    assert_eq!(rows[0].session_id.as_deref(), Some("agent-sess"));
    assert_eq!(rows[0].output_tokens, Some(1234));
}

#[tokio::test]
async fn test_flush_for_exit_persists_partial_response_when_streaming() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "exit-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.pending_response = "Once upon a time".to_string();

    app.flush_for_exit().await;

    let saved = store
        .get_code_session("exit-session")
        .await
        .unwrap()
        .expect("session should be persisted on exit");
    let messages = saved.messages;
    assert_eq!(messages.len(), 2, "user prompt + partial reply should save");
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, "Once upon a time");
}

#[tokio::test]
async fn test_flush_for_exit_persists_user_only_history() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "user-only-session".to_string();
    app.raw_model = "claude".to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "tell me a story".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    app.flush_for_exit().await;

    let saved = store
        .get_code_session("user-only-session")
        .await
        .unwrap()
        .expect("session with only a user message should still persist on exit");
    let messages = saved.messages;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    assert_eq!(messages[0].content, "tell me a story");
}

#[tokio::test]
async fn test_flush_for_exit_skips_persist_for_empty_history() {
    let temp_dir = TempDir::new().unwrap();
    let store = SessionStore::with_path(temp_dir.path().join("config.json"));
    let key_id = store
        .add_key_with_protocol("prod", "https://api.example.com", None, "sk-test")
        .await
        .unwrap();
    let key = store.get_key_by_id(&key_id).await.unwrap().unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_store = store.clone();
    app.key = key;
    app.cwd = "/tmp/demo".to_string();
    app.session_id = "empty-session".to_string();
    app.raw_model = "claude".to_string();

    app.flush_for_exit().await;

    let saved = store.get_code_session("empty-session").await.unwrap();
    assert!(
        saved.is_none(),
        "empty history should not produce a session"
    );
}
