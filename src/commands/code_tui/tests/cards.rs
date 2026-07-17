use super::super::*;
use super::helpers::*;

#[test]
fn test_permission_card_anchored_above_composer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Some history so the composer is pushed toward the bottom of the screen.
    for _ in 0..4 {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: "working on it".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build/".to_string()),
        reply,
    });

    let (screen, rows) = render_full_screen(&mut app, 60, 20);

    // The friendly heading + command + color-coded keys all render.
    assert!(
        screen.contains("Run a command?"),
        "heading missing:\n{screen}"
    );
    assert!(
        screen.contains("rm -rf build/"),
        "command missing:\n{screen}"
    );
    assert!(
        screen.contains("allow once") && screen.contains("always") && screen.contains("deny"),
        "keys missing:\n{screen}"
    );
    // A destructive command is flagged.
    assert!(
        screen.contains("⚠ looks destructive"),
        "destructive flag missing:\n{screen}"
    );

    // The card hugs the composer: its bottom border sits just above the input's
    // full-width divider rule, which sits directly above the prompt (not floating
    // mid-screen like a centered modal).
    let bottom_border_row = rows
        .iter()
        .position(|r| r.contains('└'))
        .expect("card bottom border");
    let composer_row = rows
        .iter()
        .position(|r| r.trim_start().starts_with('>'))
        .expect("composer prompt row");
    assert_eq!(
        bottom_border_row + 2,
        composer_row,
        "card bottom must sit one row (the divider) above the composer:\n{screen}"
    );
    // The divider directly under the card is the full-width composer rule — now
    // always carrying the mode badge (here "normal", since a card only shows
    // when auto-approve is off) — so the narrower card never leaves it poking
    // out past the card's right edge.
    let divider = &rows[bottom_border_row + 1];
    assert!(
        divider.contains('─') && divider.contains("normal"),
        "full-width composer rule (with the mode badge) must sit under the card:\n{screen}"
    );

    // The card is sized to its content, not stretched across the full 60-col
    // input row — its border must be clearly narrower than the screen.
    let border_w = rows[bottom_border_row].trim_end().chars().count();
    assert!(
        (10..50).contains(&border_w),
        "card should be a compact width, got {border_w}:\n{screen}"
    );
}

#[test]
fn test_permission_card_keys_always_visible_in_short_terminal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    // A long multi-line preview that cannot fully fit above the composer.
    let preview = (0..30)
        .map(|i| format!("line {i} of a very long command"))
        .collect::<Vec<_>>()
        .join("\n");
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some(preview),
        reply,
    });

    // Short screen: the preview must be trimmed so the keys row still shows.
    let (screen, _rows) = render_full_screen(&mut app, 50, 10);
    assert!(
        screen.contains("allow once") && screen.contains("deny"),
        "keys must survive a cramped card:\n{screen}"
    );
}

#[tokio::test]
async fn test_auto_approve_toggle_shows_toast_not_notice() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // Shift+Tab (BackTab) flips auto-approve. The confirmation must be a
    // self-expiring toast, NOT a persistent transcript notice pinned above the
    // input for the rest of the session.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve);
    assert!(
        app.notice.is_none(),
        "toggle must not pin a lingering notice"
    );
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Auto-approve mode")),
        "toggle should flash a toast"
    );

    // The next press cycles into plan mode — same toast-not-notice contract.
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(!app.agent_auto_approve);
    assert!(app.plan_mode, "auto cycles into plan mode");
    assert!(app.notice.is_none());
    assert!(
        app.toast
            .as_ref()
            .is_some_and(|t| t.text.contains("Plan mode"))
    );
}

#[tokio::test]
async fn test_shift_tab_on_permission_card_approves_and_enables() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    // A tool-permission card is up, holding the keyboard.
    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // Shift+Tab on the card must enable auto-approve AND approve THIS request —
    // not get swallowed (the old behavior, where only y/a/n worked).
    app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_auto_approve, "toggle enables auto-approve");
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the live flag the running turn reads is set"
    );
    assert!(app.agent_permission.is_none(), "the card is dismissed");
    assert_eq!(
        decision_rx.await.unwrap(),
        Decision::Allow,
        "the pending request is approved"
    );
}

/// "Always" on a Cursor permission card flips session-wide auto-approve, so the
/// composer badge and exit-persistence (which read `agent_auto_approve`) must
/// agree with the shared live flag — the badge can't say "off" while Cursor
/// silently allows the rest of the session.
#[tokio::test]
async fn test_always_on_cursor_card_enables_auto_approve() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("Cursor wants to run:\nedit main.rs".to_string()),
        reply,
    });

    // 'a' = always: cursor "always" is session-wide auto-approve.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_auto_approve,
        "cursor 'always' flips the bool the badge/persistence read"
    );
    assert!(
        app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "and the shared live flag the cursor session reads"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// In-process-engine "always" stays scoped to that (tool,args): it must NOT flip
/// the global auto-approve badge on — only Cursor cards do that.
#[tokio::test]
async fn test_always_on_native_card_leaves_auto_approve_off() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(!app.agent_auto_approve);

    let (reply, decision_rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        !app.agent_auto_approve,
        "native 'always' is per-tool, not global auto-approve"
    );
    assert!(
        !app.auto_approve_flag
            .load(std::sync::atomic::Ordering::Relaxed),
        "the global live flag stays off for native 'always'"
    );
    assert_eq!(decision_rx.await.unwrap(), Decision::AlwaysAllow);
}

/// A permission card must not steal in-flight composer typing. With a queued
/// draft in progress the single-letter decision keys (y/a/n) belong to that
/// message — they type into the draft and leave the card up, so a stray
/// keystroke can't approve a tool by accident. Esc still resolves the card.
#[tokio::test]
async fn test_permission_card_keys_do_not_decide_while_composing() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The user is mid-composing a queued message when a card appears.
    app.draft = "deplo".to_string();
    app.cursor = app.draft.len();
    let (reply, mut decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });

    // 'y' is part of the word "deploy", not an approval.
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "deploy", "the keystroke typed into the draft");
    assert!(
        app.agent_permission.is_some(),
        "the card stays up — the key did not decide"
    );
    assert!(
        matches!(
            decision_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ),
        "no decision was sent to the waiting engine"
    );

    // Esc always denies — it can't be message content.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "Esc resolves the card");
    assert_eq!(decision_rx.await.unwrap(), Decision::Deny);
}

/// With an empty composer (the idle case) the quick single-key decisions still
/// work — 'y' approves immediately.
#[tokio::test]
async fn test_permission_card_y_approves_with_empty_composer() {
    use crate::agent::protocol::Decision;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.draft.is_empty());

    let (reply, decision_rx) = tokio::sync::oneshot::channel::<Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    app.handle_key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_permission.is_none(), "the card is resolved");
    assert_eq!(decision_rx.await.unwrap(), Decision::Allow);
}

/// The card swaps its keycap row for a hint while a draft is in progress, so the
/// user knows y/a/n won't decide and how to respond without losing the message.
#[test]
fn test_permission_card_shows_composing_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "next message".to_string();
    app.cursor = app.draft.len();
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("type into your message"),
        "composing hint missing:\n{screen}"
    );
}

/// A Cursor card's "always" is session-wide auto-approve, unlike the native
/// engine's per-action "always"; the keycap label spells that out so the broader
/// scope isn't a surprise.
#[test]
fn test_permission_card_cursor_always_label_says_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "cursor".to_string(),
        preview: Some("edit main.rs".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always (this session)"),
        "cursor 'always' must disclose its session-wide scope:\n{screen}"
    );
}

/// The native engine's "always" stays scoped to this action, so its label is the
/// plain "always" with no session-wide qualifier.
#[test]
fn test_permission_card_native_always_label_is_plain() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::protocol::Decision>();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("ls".to_string()),
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("always"),
        "native card still offers 'always':\n{screen}"
    );
    assert!(
        !screen.contains("this session"),
        "native 'always' is scoped, not session-wide:\n{screen}"
    );
}

fn ask_options(labels: &[&str]) -> Vec<crate::agent::ask::AskOption> {
    labels
        .iter()
        .map(|l| crate::agent::ask::AskOption {
            label: (*l).to_string(),
            description: None,
        })
        .collect()
}

/// The `ask_user` card: ↓ moves the highlight, Enter picks it, and the chosen
/// label is sent back to the waiting engine task as the answer.
#[tokio::test]
async fn test_ask_card_arrow_then_enter_selects_option() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Add release notes now?".to_string(),
        options: ask_options(&["Yes, I'll write them", "You add them", "No, auto-generate"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none(), "answering resolves the card");
    assert_eq!(answer_rx.await.unwrap(), Ok("You add them".to_string()));
}

/// A digit key jumps straight to that option and picks it (numbered-menu style).
#[tokio::test]
async fn test_ask_card_digit_picks_option() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Pick one".to_string(),
        options: ask_options(&["alpha", "beta", "gamma"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Char('3'), KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none());
    assert_eq!(answer_rx.await.unwrap(), Ok("gamma".to_string()));
}

/// With free text allowed, typing an answer and pressing Enter submits the draft
/// as the answer (rather than queueing it as a chat message).
#[tokio::test]
async fn test_ask_card_free_text_answer() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which version?".to_string(),
        options: ask_options(&["patch", "minor"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.draft = "0.36.0".to_string();
    app.cursor = app.draft.len();
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none());
    assert!(app.draft.is_empty(), "the draft is consumed as the answer");
    assert_eq!(answer_rx.await.unwrap(), Ok("0.36.0".to_string()));
}

/// Esc dismisses the card; the engine's `ask_user` future then resolves to an
/// error it surfaces as the tool result.
#[tokio::test]
async fn test_ask_card_esc_dismisses() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Proceed?".to_string(),
        options: ask_options(&["Yes", "No"]),
        allow_free_text: false,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.agent_ask.is_none(), "Esc resolves the card");
    assert!(answer_rx.await.unwrap().is_err());
}

/// The card renders the question, the numbered options, and the nav hint.
#[test]
fn test_ask_card_renders_question_and_options() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Add release notes now?".to_string(),
        options: ask_options(&["You add them", "Auto-generate"]),
        allow_free_text: true,
        multi_select: false,
        checked: Vec::new(),
        selected: 0,
        reply,
    });
    // A transcript pushes the composer to the bottom so the floating card has room
    // above it to render its own key-hint line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("Add release notes now?"),
        "question missing:\n{screen}"
    );
    assert!(screen.contains("You add them"), "option missing:\n{screen}");
    assert!(screen.contains("select"), "nav hint missing:\n{screen}");
}

/// Multi-select: `space` toggles the highlighted box, arrows move, and Enter
/// returns the checked labels joined by ", " (not just the highlighted one).
#[tokio::test]
async fn test_ask_card_multi_select_space_toggles_and_enter_joins() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which checks?".to_string(),
        options: ask_options(&["fmt", "clippy", "test"]),
        allow_free_text: false,
        multi_select: true,
        checked: vec![false; 3],
        selected: 0,
        reply,
    });

    // Check fmt via space, move down twice, check test.
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.agent_ask.is_some(), "space toggles without confirming");
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(
        app.agent_ask.is_none(),
        "Enter resolves the multi-select card"
    );
    assert_eq!(answer_rx.await.unwrap(), Ok("fmt, test".to_string()));
}

/// Multi-select renders checkboxes and the "toggle" hint, and a digit toggles the
/// matching box rather than immediately submitting.
#[tokio::test]
async fn test_ask_card_multi_select_digit_toggles_box() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, answer_rx) = tokio::sync::oneshot::channel::<std::result::Result<String, String>>();
    app.agent_ask = Some(PendingAskUser {
        question: "Which checks?".to_string(),
        options: ask_options(&["fmt", "clippy", "test"]),
        allow_free_text: false,
        multi_select: true,
        checked: vec![false; 3],
        selected: 0,
        reply,
    });

    // A transcript pushes the composer to the bottom so the card has room to show
    // its own key-hint line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(screen.contains("[ ]"), "unchecked boxes render:\n{screen}");
    assert!(
        screen.contains("toggle"),
        "multi-select hint missing:\n{screen}"
    );

    // Digit 2 toggles clippy (index 1) without submitting. Assert on state: a
    // slim card can scroll the second option out of view.
    app.handle_key(KeyEvent::new(KeyCode::Char('2'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.agent_ask.is_some(),
        "a digit toggles, it does not submit"
    );
    assert_eq!(
        app.agent_ask.as_ref().unwrap().checked,
        vec![false, true, false],
        "digit 2 checks the second box"
    );

    // Toggle the always-visible first option so a [✓] is on screen regardless
    // of card height.
    app.handle_key(KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE))
        .await
        .unwrap();
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("[✓]"),
        "a toggled box shows checked:\n{screen}"
    );

    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(answer_rx.await.unwrap(), Ok("fmt, clippy".to_string()));
}

/// The edit-review card renders the heading, the per-file diff, and the y/n keys.
#[test]
fn test_review_card_renders_diff_and_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let items = vec![crate::agent::review::review_item(
        0,
        "edit_file",
        &serde_json::json!({
            "path": "src/lib.rs",
            "old_string": "let x = 1;",
            "new_string": "let x = 2;",
        }),
    )];
    let body = super::super::render::review_body_lines(&items, std::path::Path::new("."));
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    // A transcript pushes the composer to the bottom so the card has room to show
    // its own y/n key line (there's no hint-bar fallback now).
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: (0..30)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n"),
        reasoning_content: None,
        attachments: vec![],
    });
    let (screen, _rows) = render_full_screen(&mut app, 80, 24);
    assert!(
        screen.contains("review 1 edit"),
        "heading missing:\n{screen}"
    );
    assert!(
        screen.contains("src/lib.rs"),
        "file header missing:\n{screen}"
    );
    assert!(
        screen.contains("approve") && screen.contains("reject"),
        "y/n hints missing:\n{screen}"
    );
}

/// `y`/Enter approve the batch, `n`/Esc reject it — each resolves the card and
/// sends the verdict to the waiting engine task.
#[tokio::test]
async fn test_review_card_keys_resolve_decision() {
    use crate::agent::review::ReviewDecision;
    let cases: [(KeyCode, ReviewDecision); 4] = [
        (KeyCode::Char('y'), ReviewDecision::ApproveAll),
        (KeyCode::Enter, ReviewDecision::ApproveAll),
        (KeyCode::Char('n'), ReviewDecision::Reject),
        (KeyCode::Esc, ReviewDecision::Reject),
    ];
    for (code, expected) in cases {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        let (reply, decision_rx) = tokio::sync::oneshot::channel::<ReviewDecision>();
        app.agent_review = Some(PendingReview {
            count: 1,
            body: vec![ratatui::text::Line::from("diff")],
            scroll: 0,
            reply,
        });
        app.handle_key(KeyEvent::new(code, KeyModifiers::NONE))
            .await
            .unwrap();
        assert!(app.agent_review.is_none(), "{code:?} resolves the card");
        assert_eq!(decision_rx.await.unwrap(), expected, "{code:?}");
    }
}

/// Arrow keys scroll the review body without resolving the card.
#[tokio::test]
async fn test_review_card_arrows_scroll() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..10)
        .map(|i| ratatui::text::Line::from(format!("line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.agent_review.as_ref().unwrap().scroll, 2, "down scrolls");
    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(
        app.agent_review.as_ref().unwrap().scroll,
        1,
        "up scrolls back"
    );
    assert!(app.agent_review.is_some(), "scrolling does not resolve");
}

/// History pushes the composer to the bottom, giving cards a real viewport.
fn push_review_test_history(app: &mut CodeTuiApp) {
    for _ in 0..4 {
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: "working on it".to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

/// A diff taller than the screen must not push the y/n keys off the card.
#[test]
fn test_review_card_long_diff_keeps_keys_visible() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    push_review_test_history(&mut app);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..100)
        .map(|i| ratatui::text::Line::from(format!("diff line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    let (screen, _rows) = render_full_screen(&mut app, 80, 24);
    assert!(
        screen.contains("approve") && screen.contains("reject"),
        "y/n hints clipped off the overflowing card:\n{screen}"
    );
    assert!(
        screen.contains("more (↑↓ scroll)"),
        "overflow marker missing:\n{screen}"
    );
}

/// Overscrolling past the bottom is a no-op — the card height never changes.
#[tokio::test]
async fn test_review_card_overscroll_clamps_stable_height() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    push_review_test_history(&mut app);
    let (reply, _rx) = tokio::sync::oneshot::channel::<crate::agent::review::ReviewDecision>();
    let body: Vec<ratatui::text::Line> = (0..100)
        .map(|i| ratatui::text::Line::from(format!("diff line {i}")))
        .collect();
    app.agent_review = Some(PendingReview {
        count: 1,
        body,
        scroll: 0,
        reply,
    });
    for _ in 0..200 {
        app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
            .await
            .unwrap();
    }
    let (bottom, _rows) = render_full_screen(&mut app, 80, 24);
    let clamped = app.agent_review.as_ref().unwrap().scroll;
    assert!(
        usize::from(clamped) < 99,
        "scroll clamps to the last page, not the last line (got {clamped})"
    );
    assert!(
        bottom.contains("end of diff") && bottom.contains("approve"),
        "bottom keeps the marker row and keys:\n{bottom}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    let (after, _rows) = render_full_screen(&mut app, 80, 24);
    assert_eq!(bottom, after, "scrolling past the bottom changes nothing");
}
