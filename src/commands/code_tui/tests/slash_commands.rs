use super::super::*;
use super::helpers::*;

#[test]
fn test_parse_slash_command_with_argument() {
    assert_eq!(
        parse_slash_command("model claude-sonnet-4").unwrap(),
        SlashCommand::Model(Some("claude-sonnet-4".to_string()))
    );
    assert_eq!(
        parse_slash_command("attach ./README.md").unwrap(),
        SlashCommand::Attach("./README.md".to_string())
    );
    assert_eq!(
        parse_slash_command("resume").unwrap(),
        SlashCommand::Resume(None)
    );
    assert_eq!(
        parse_slash_command("detach 2").unwrap(),
        SlashCommand::Detach(2)
    );
    assert_eq!(
        parse_slash_command("mcp add fs npx -y srv").unwrap(),
        SlashCommand::Mcp(Some("add fs npx -y srv".to_string()))
    );
    assert_eq!(parse_slash_command("mcp").unwrap(), SlashCommand::Mcp(None));
}

#[test]
fn test_parse_slash_share() {
    assert_eq!(
        parse_slash_command("share").unwrap(),
        SlashCommand::Share(None)
    );
    assert_eq!(
        parse_slash_command("share stop").unwrap(),
        SlashCommand::Share(Some("stop".to_string()))
    );
}

#[test]
fn test_parse_slash_compact() {
    assert_eq!(
        parse_slash_command("compact").unwrap(),
        SlashCommand::Compact { fast: false }
    );
    assert_eq!(
        parse_slash_command("compact fast").unwrap(),
        SlashCommand::Compact { fast: true }
    );
    assert_eq!(
        parse_slash_command("compact now").unwrap(),
        SlashCommand::Compact { fast: false }
    );
}

#[test]
fn test_parse_slash_context() {
    assert_eq!(
        parse_slash_command("context").unwrap(),
        SlashCommand::Context
    );
}

#[test]
fn test_parse_slash_session() {
    assert_eq!(
        parse_slash_command("session").unwrap(),
        SlashCommand::Session
    );
}

/// `/context` always opens the breakdown, folding in the injected digest section when present.
#[tokio::test]
async fn test_context_overlay_shows_breakdown() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Nothing injected: the breakdown still opens.
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 30);
    assert!(screen.contains("Context"), "title:\n{screen}");
    assert!(screen.contains("System prompt"), "segments:\n{screen}");
    assert!(screen.contains("Tools"), "segments:\n{screen}");
    assert!(
        !screen.contains("Injected context"),
        "no injection expected:\n{screen}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));

    // With an injected block, the breakdown also carries the injected section + body.
    app.injected_context = Some("# aivo context\n\n**Topic:** prior work".to_string());
    app.injected_context_summary = Some("injected ~9 tokens from claude session abc (2m)".into());
    app.open_context_overlay().await;
    assert!(matches!(app.overlay, Overlay::Context { scroll: 0, .. }));
    let (screen, _) = render_full_screen(&mut app, 90, 44);
    assert!(screen.contains("Injected context"), "section:\n{screen}");
    assert!(
        screen.contains("injected ~9 tokens from claude session abc"),
        "summary header:\n{screen}"
    );
    assert!(screen.contains("Topic:"), "body:\n{screen}");
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_compact_command_no_engine_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_compact_command(true).await;
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("nothing to compact"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.agent_serve.is_none() && app.response_task.is_none());
}

#[tokio::test]
async fn test_share_command_stop_and_usage_notices() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `/share stop` with no active share → informative notice, nothing started.
    app.run_share_command(Some("stop".to_string())).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("Not currently"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.share.handle.is_none());

    // Unknown argument → usage notice (no background start).
    app.run_share_command(Some("frobnicate".to_string())).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("Usage"),
        "notice: {:?}",
        app.notice
    );
    assert!(!app.share.starting);
}

#[tokio::test]
async fn test_share_command_reshows_url_then_stops() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.share.handle = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/v.html?t=zz",
    ));

    // Bare `/share` while already sharing just re-shows the URL — no new start.
    app.run_share_command(None).await;
    assert!(
        app.notice.as_ref().unwrap().1.contains("t=zz"),
        "notice: {:?}",
        app.notice
    );
    assert!(app.share.handle.is_some());
    assert!(!app.share.starting);

    // `/share stop` tears it down.
    app.run_share_command(Some("stop".to_string())).await;
    assert!(app.share.handle.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("stopped"));
}

#[test]
fn test_apply_live_share_ready_ok_and_err() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.share.starting = true;
    app.apply_live_share_ready(
        app.share.generation,
        Ok(LiveShareHandle::for_test(
            "https://s.getaivo.dev/v.html?t=ok",
        )),
    );
    assert!(!app.share.starting);
    assert!(app.share.handle.is_some());
    assert!(app.notice.as_ref().unwrap().1.contains("t=ok"));

    // Failure: clears the starting flag, surfaces the reason, stores nothing.
    app.share.handle = None;
    app.share.starting = true;
    app.apply_live_share_ready(app.share.generation, Err("no link".to_string()));
    assert!(!app.share.starting);
    assert!(app.share.handle.is_none());
    assert_eq!(app.notice.as_ref().unwrap().1, "no link");
}

/// A share start outlived by a stop//new//resume must not install its tunnel;
/// a start under the fresh generation still lands.
#[test]
fn test_stale_live_share_ready_is_dropped() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.share.starting = true;
    let stale_gen = app.share.generation;
    assert!(
        app.stop_live_share(),
        "cancelling a mid-handshake start counts as a stop"
    );
    assert!(!app.share.starting);

    app.apply_live_share_ready(stale_gen, Ok(LiveShareHandle::for_test("https://s/old")));
    assert!(app.share.handle.is_none(), "stale handle must not install");

    // A new start under the bumped generation works.
    app.share.starting = true;
    app.apply_live_share_ready(
        app.share.generation,
        Ok(LiveShareHandle::for_test("https://s/new")),
    );
    assert!(app.share.handle.is_some());
}

/// A dead tunnel (network drop — no auto-reconnect) clears the badge and its
/// server on the next frame instead of showing a live share that no longer serves.
#[test]
fn test_dead_share_tunnel_clears_badge() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let handle = LiveShareHandle::for_test("https://s/z");
    handle.mark_dead_for_test();
    app.share.handle = Some(handle);

    app.check_live_share_health();

    assert!(app.share.handle.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("disconnected"));

    // No share → no-op (must not overwrite an unrelated notice).
    app.notice = None;
    app.check_live_share_health();
    assert!(app.notice.is_none());
}

#[tokio::test]
async fn test_maybe_start_live_share_defers_until_session_settles() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // No `--share` request → never starts.
    assert!(!app.maybe_start_live_share().await);

    app.share.requested = true;

    // A pending `--resume` load defers the start (it must pin the resumed session).
    app.loading_resume = Some(LoadingResume {
        request_id: 1,
        preview: SessionPreview {
            key_id: "k".into(),
            key_name: "k".into(),
            base_url: "u".into(),
            session_id: "resumed".into(),
            raw_model: "m".into(),
            updated_at: "t".into(),
            title: "t".into(),
            preview_text: "p".into(),
            origin: None,
        },
    });
    assert!(!app.maybe_start_live_share().await);
    assert!(
        app.share.requested,
        "request stays pending while resume loads"
    );
    app.loading_resume = None;

    // An already-running share or an in-flight start are both no-ops.
    app.share.handle = Some(LiveShareHandle::for_test("https://x"));
    assert!(!app.maybe_start_live_share().await);
    app.share.handle = None;
    app.share.starting = true;
    assert!(!app.maybe_start_live_share().await);
    assert!(app.share.requested);
}

#[test]
fn test_notice_spans_splits_live_url_from_indicator() {
    // The share notice paints `● Sharing:` red but the URL a calm link color, so
    // the long line doesn't read as an error. Other notices stay a single span.
    let share = (
        LIVE(),
        format!("{LIVE_NOTICE_PREFIX}https://s.getaivo.dev/s/abc"),
    );
    let spans = notice_spans(Some(&share)).unwrap();
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].content.as_ref(), LIVE_NOTICE_PREFIX);
    assert_eq!(spans[0].style.fg, Some(LIVE()));
    assert_eq!(spans[1].content.as_ref(), "https://s.getaivo.dev/s/abc");
    assert_eq!(spans[1].style.fg, Some(LINK()));

    let plain = (MUTED(), "just a status".to_string());
    let spans = notice_spans(Some(&plain)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].style.fg, Some(MUTED()));

    // ERROR keeps its `Error:` prefix and single span.
    let err = (ERROR(), "boom".to_string());
    let spans = notice_spans(Some(&err)).unwrap();
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].content.as_ref(), "Error: boom");
}

#[test]
fn test_parse_slash_command_unknown() {
    let err = parse_slash_command("wat").unwrap_err().to_string();
    assert!(err.contains("Unknown command"));
}

/// `/copy 0` is a usage error and out-of-range names the real count, instead of
/// claiming no reply exists.
#[tokio::test]
async fn test_copy_rejects_zero_and_reports_range() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "only reply".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });

    let err = app.copy_reply_to_clipboard(Some(0)).unwrap_err();
    assert!(err.to_string().contains("Usage"), "{err}");

    let err = app.copy_reply_to_clipboard(Some(5)).unwrap_err();
    assert!(err.to_string().contains("Only 1 reply"), "{err}");
}
