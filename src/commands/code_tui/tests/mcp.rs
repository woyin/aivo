use super::super::*;
use super::helpers::*;

#[test]
fn bare_url_in_mcp_add_becomes_url_config() {
    use super::super::session_impl::bare_url_to_config;
    // A bare http(s) URL → a {url} server config (no JSON typing needed).
    let json = bare_url_to_config("https://mcp.linear.app/mcp").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://mcp.linear.app/mcp");
    assert!(bare_url_to_config("http://127.0.0.1:8080/mcp").is_some());
    // Only the first token is taken (a URL has no spaces).
    let json = bare_url_to_config("https://h/mcp  oops").unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["url"], "https://h/mcp");
    // A command line or a JSON block is NOT a bare URL (handled elsewhere).
    assert!(bare_url_to_config("npx -y @scope/server").is_none());
    assert!(bare_url_to_config(r#"{"url":"https://h"}"#).is_none());
}

fn dummy_agent_session() -> AgentSession {
    AgentSession {
        key_id: "k".to_string(),
        model: "m".to_string(),
        engine: std::sync::Arc::new(tokio::sync::Mutex::new(
            crate::agent::engine::AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0),
        )),
    }
}

#[tokio::test]
async fn test_apply_mcp_connected_empty_keeps_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.mcp_connecting = true;
    app.agent_engine = Some(dummy_agent_session());
    // An empty client (no mcp.json in this temp dir) brings no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-empty-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    app.apply_mcp_connected(client);
    assert!(!app.mcp_connecting, "connecting flag should clear");
    assert!(app.mcp_client.is_some(), "client should be cached");
    assert!(
        app.agent_engine.is_some(),
        "an empty MCP result must not drop the engine"
    );
    assert!(!app.engine_rebuild_pending);
}

#[tokio::test]
async fn test_apply_mcp_connected_surfaces_connect_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A config pointing at a non-spawnable command → a connect error, no tools.
    let dir = std::env::temp_dir().join(format!("aivo-mcp-notice-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    assert!(!client.errors().is_empty(), "expected a connect error");

    app.apply_mcp_connected(client);
    let notice = app
        .notice
        .as_ref()
        .expect("a failed MCP connect should notify");
    assert!(notice.1.contains("MCP"), "notice: {}", notice.1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// When a background connect resolves while the `/mcp` overlay is open, its rows
/// must refresh from the new client in place (no close-and-reopen) — here the
/// "connecting…" row flips to the failure once the broken server's error lands.
#[tokio::test]
async fn test_apply_mcp_connected_refreshes_open_overlay() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "broken".to_string(),
            status: "connecting…".to_string(),
            health: McpHealth::Idle,
            enabled: true,
            scope: ServerScope::Project,
            command: "aivo_no_such_binary_zzz".to_string(),
            remote: false,
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-refresh-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let client = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );

    app.apply_mcp_connected(client);
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(
            state.items[0].health,
            McpHealth::Failed,
            "open overlay row not refreshed to failed: {}",
            state.items[0].status
        );
        assert!(
            state.items[0].status.contains("failed"),
            "status: {}",
            state.items[0].status
        );
    } else {
        panic!("mcp overlay vanished");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A connect launched before a `/mcp` toggle (older generation) must be dropped,
/// so it can't resurrect a just-disabled server; the current generation applies.
#[tokio::test]
async fn test_stale_mcp_connect_is_dropped() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let tx2 = tx.clone();
    let mut app = make_test_app(tx, rx);
    // A toggle advanced the generation while a previous connect is in flight.
    app.mcp_connect_gen = 1;
    app.mcp_connecting = true;

    let dir = std::env::temp_dir().join(format!("aivo-mcp-stale-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let stale = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: stale,
        generation: 0,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_none(), "stale connect must not be cached");
    assert!(
        app.mcp_connecting,
        "stale connect must not clear the in-flight flag"
    );

    let fresh = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(&dir, &std::collections::HashSet::new())
            .await,
    );
    tx2.send(RuntimeEvent::McpConnected {
        client: fresh,
        generation: 1,
    })
    .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.mcp_client.is_some(), "current-gen connect should cache");
    assert!(!app.mcp_connecting, "current-gen connect clears the flag");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_maybe_apply_engine_rebuild_drops_engine_when_pending() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.agent_engine = Some(dummy_agent_session());
    app.engine_rebuild_pending = true;
    app.maybe_apply_engine_rebuild();
    assert!(
        app.agent_engine.is_none(),
        "pending rebuild should drop engine"
    );
    assert!(!app.engine_rebuild_pending, "flag should clear");
    // Not pending → engine left alone.
    app.agent_engine = Some(dummy_agent_session());
    app.maybe_apply_engine_rebuild();
    assert!(app.agent_engine.is_some());
}

#[tokio::test]
async fn test_mcp_add_project_flag_writes_repo_config_and_grants_consent() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let dir = std::env::temp_dir().join(format!("aivo-mcp-project-add-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();

    app.submit_mcp_add("-p echo hi".to_string()).await.unwrap();

    // Written to the repo `.mcp.json`, not the user config.
    let root: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(".mcp.json")).unwrap()).unwrap();
    let servers = root["mcpServers"].as_object().unwrap();
    assert!(
        servers.values().any(|v| v["command"] == "echo"),
        "project .mcp.json holds the added server: {root}"
    );
    // Typing the command IS the consent — run-once session approval, like `y`.
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    assert!(app.cards.mcp_consent.is_none());
    let notice = app.notice.as_ref().unwrap().1.clone();
    assert!(notice.contains("./.mcp.json"), "notice: {notice}");
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn test_mcp_multi_paste_opens_picker_and_replaces_on_apply() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let dir = std::env::temp_dir().join(format!("aivo-mcp-paste-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();
    // A same-named server is already configured (project scope).
    std::fs::write(
        dir.join(".mcp.json"),
        r#"{"mcpServers":{"github":{"command":"old-cmd"}}}"#,
    )
    .unwrap();

    app.submit_mcp_add(
        r#"-p {"mcpServers":{
            "github":{"command":"echo","args":["new"]},
            "linear":{"url":"http://127.0.0.1:1/mcp"}
        }}"#
        .to_string(),
    )
    .await
    .unwrap();

    // ≥2 servers → picker, not a blind add: the new name is prechecked, the
    // existing one needs an explicit replace mark.
    let github = {
        let Overlay::McpPaste(state) = &app.overlay else {
            panic!("expected the paste picker to open");
        };
        assert!(state.project);
        assert!(
            state.parent.is_none(),
            "composer paste has no /mcp to restore"
        );
        let github = state.items.iter().position(|i| i.name == "github").unwrap();
        let linear = state.items.iter().position(|i| i.name == "linear").unwrap();
        assert!(state.items[github].exists && !state.items[github].checked);
        assert!(!state.items[linear].exists && state.items[linear].checked);
        github
    };
    // Mark the existing row too — that means replace-in-place.
    if let Overlay::McpPaste(state) = &mut app.overlay {
        state.items[github].checked = true;
    }
    app.apply_mcp_paste().await.unwrap();

    let root: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(".mcp.json")).unwrap()).unwrap();
    let servers = root["mcpServers"].as_object().unwrap();
    assert_eq!(servers.len(), 2, "no `github-2` duplicate: {root}");
    assert_eq!(servers["github"]["command"], "echo", "replaced in place");
    assert_eq!(servers["linear"]["url"], "http://127.0.0.1:1/mcp");
    let notice = app.notice.as_ref().unwrap().1.clone();
    assert!(
        notice.contains("Added linear") && notice.contains("replaced github"),
        "notice: {notice}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn test_mcp_add_json_routes_and_reports_parse_error() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A `{`-leading add input routes to the JSON path; malformed JSON surfaces a
    // parse error (and writes nothing — verified by never reaching a real write).
    app.submit_mcp_add("{ not valid json".to_string())
        .await
        .unwrap();
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("Couldn't parse MCP config"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

#[tokio::test]
async fn test_mcp_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_mcp_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Mcp(_)),
        "bare /mcp opens overlay"
    );

    // Unknown subcommand → usage notice, overlay not opened.
    app.overlay = Overlay::None;
    app.run_mcp_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name → usage notice.
    app.run_mcp_command(Some("rm".to_string())).await.unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent server → "No MCP server" notice, no config write.
    app.run_mcp_command(Some("rm __aivo_no_such_server__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No MCP server"),
        "notice: {:?}",
        app.notice
    );
}

/// A project `.mcp.json` server is not in the *base* opt-out set (the consent
/// gate that holds stdio servers back lives in `connect_mcp_with_consent`, not
/// here), and toggling a project row off in `/mcp` adds it to the global opt-out
/// list, exactly like a user server.
#[tokio::test]
async fn project_mcp_server_connects_by_default_and_toggles_like_user() {
    use crate::agent::mcp::ServerScope;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-proj-mcp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"fs":{"command":"echo"}}}"#,
    )
    .unwrap();
    app.real_cwd = repo.to_str().unwrap().to_string();

    // The base opt-out set is empty (the consent gate is applied separately).
    assert!(
        !app.effective_disabled_mcp_servers().await.contains("fs"),
        "a project .mcp.json server is not in the base opt-out set"
    );

    // Toggling a project row off goes to the global opt-out, like a user server.
    app.overlay = Overlay::Mcp(McpOverlay {
        items: vec![McpServerRow {
            name: "fs".to_string(),
            status: "1 tool".to_string(),
            health: McpHealth::Connected,
            enabled: true,
            scope: ServerScope::Project,
            command: "echo".to_string(),
            remote: false,
        }],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(
        app.session_store.get_disabled_mcp_servers().await.unwrap(),
        vec!["fs".to_string()],
        "toggling a project row off adds it to the user opt-out list"
    );
    assert!(app.effective_disabled_mcp_servers().await.contains("fs"));

    let _ = std::fs::remove_dir_all(&repo);
}

#[test]
fn test_sort_mcp_rows_problems_first() {
    use super::super::session_impl::sort_mcp_rows;
    use crate::agent::mcp::ServerScope;
    let row = |name: &str, health| McpServerRow {
        name: name.to_string(),
        status: String::new(),
        health,
        enabled: !matches!(health, McpHealth::Disabled),
        scope: ServerScope::User,
        command: "x".to_string(),
        remote: false,
    };
    let mut rows = vec![
        row("zeta", McpHealth::Connected),
        row("off-one", McpHealth::Disabled),
        row("beta", McpHealth::Failed),
        row("needs", McpHealth::NeedsAuth),
        row("alpha", McpHealth::Connected),
        row("idle", McpHealth::Idle),
    ];
    sort_mcp_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Failed first, then needs-auth (actionable), then connected (alphabetical),
    // then idle, then disabled last.
    assert_eq!(
        order,
        vec!["beta", "needs", "alpha", "zeta", "idle", "off-one"]
    );
}

#[tokio::test]
async fn test_mcp_filter_narrows_selection() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // filesystem, github

    app.handle_key(KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.filtered_indices(), vec![1], "only github matches 'g'");
        assert_eq!(state.selected, 1);
        assert!(state.has_selection());
    } else {
        panic!("overlay vanished");
    }
    // A query matching nothing leaves no visible selection (so Enter/Tab no-op).
    app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(
            state.filtered_indices().is_empty(),
            "no server matches 'gz'"
        );
        assert!(!state.has_selection());
    }
}

#[tokio::test]
async fn test_mcp_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection.
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // 2 servers, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the tool list.
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Mcp(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.detail_scroll == 3 && s.selected == 0));
}

#[tokio::test]
async fn test_mcp_split_wheel_routes_by_pane() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let detail = app.overlay_detail_area.expect("split active");

    let mut over_detail = wheel(MouseEventKind::ScrollDown);
    over_detail.column = detail.x + 1;
    over_detail.row = detail.y + 1;
    app.handle_mouse(over_detail).await.unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.detail_scroll == 3 && s.selected == 0));

    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Mcp(s) if s.selected == 1 && s.detail_scroll == 0));
}

#[test]
fn test_mcp_overlay_renders_server_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("MCP servers"), "missing title:\n{screen}");
    assert!(
        screen.contains("filesystem"),
        "missing server name:\n{screen}"
    );
    // The status renders on its own line under the server name.
    assert!(screen.contains("5 tools"), "missing status:\n{screen}");
    assert!(
        screen.contains("[✓]") && screen.contains("[ ]"),
        "missing checkboxes:\n{screen}"
    );
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
}

#[test]
fn test_mcp_overlay_renders_detail_line_for_selected() {
    use crate::agent::mcp::ServerScope;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Select a disabled, project-scoped server: with no live client its detail
    // line should still spell out the scope and the (actionable) disabled state.
    let mut overlay = mcp_overlay_fixture();
    overlay.items[1].scope = ServerScope::Project;
    overlay.selected = 1; // "github", off
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project (.mcp.json)"),
        "detail line should tag a project-scoped server:\n{screen}"
    );
    assert!(
        screen.contains("disabled"),
        "detail line should show the disabled state:\n{screen}"
    );
}

/// A per-server progress event mid-connect flips just that server's row to its
/// resolved status; other still-connecting servers keep reading "connecting…".
/// A stale-generation event is ignored.
#[tokio::test]
async fn test_mcp_progress_flips_only_resolved_row() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture(); // filesystem, github
    // Both enabled and mid-connect.
    for item in &mut overlay.items {
        item.enabled = true;
        item.status = "connecting…".to_string();
        item.health = McpHealth::Idle;
    }
    app.overlay = Overlay::Mcp(overlay);
    app.mcp_connecting = true;
    app.mcp_client = None;

    // "filesystem" resolves; "github" hasn't yet.
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "filesystem".to_string(),
            status: "5 tools".to_string(),
            health: McpHealth::Connected,
            generation: app.mcp_connect_gen,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();

    let row = |app: &CodeTuiApp, name: &str| -> (String, McpHealth) {
        match &app.overlay {
            Overlay::Mcp(s) => s
                .items
                .iter()
                .find(|i| i.name == name)
                .map(|i| (i.status.clone(), i.health))
                .unwrap(),
            _ => panic!("overlay vanished"),
        }
    };
    assert_eq!(
        row(&app, "filesystem"),
        ("5 tools".to_string(), McpHealth::Connected),
        "resolved server should flip to its tool count"
    );
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "unresolved server should still read connecting"
    );

    // A stale-generation event (a connect superseded by a toggle) is dropped.
    let stale = app.mcp_connect_gen.wrapping_add(7);
    app.tx
        .send(RuntimeEvent::McpServerProgress {
            name: "github".to_string(),
            status: "9 tools".to_string(),
            health: McpHealth::Connected,
            generation: stale,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        row(&app, "github").0,
        "connecting…",
        "a stale-generation progress event must be ignored"
    );
}

#[tokio::test]
async fn test_toggle_mcp_server_persists_and_resets() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Disable "filesystem" (index 0, currently enabled).
    app.toggle_mcp_server(0).await.unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
        assert_eq!(state.items[0].status, "off");
    } else {
        panic!("mcp overlay vanished");
    }
    let disabled = app.session_store.get_disabled_mcp_servers().await.unwrap();
    assert_eq!(disabled, vec!["filesystem".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_mcp_server(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap()
            .is_empty()
    );
}

/// Toggling a server refreshes the welcome chip's MCP count instead of freezing
/// it at the startup value.
#[tokio::test]
async fn test_toggle_mcp_server_updates_welcome_count() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture()); // filesystem on, github off
    app.mcp_configured_count = 42; // stale value the fix must overwrite

    // Disable the one enabled server → 0 enabled.
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(
        app.mcp_configured_count, 0,
        "count not refreshed on disable"
    );

    // Re-enable it → back to 1.
    app.toggle_mcp_server(0).await.unwrap();
    assert_eq!(app.mcp_configured_count, 1, "count not refreshed on enable");
}

/// Toggling a server keeps the live client (rather than nulling it) so the
/// servers that *aren't* being toggled keep serving their status during the
/// reconnect — and bumps the generation so the reconnect supersedes any in-flight
/// one. (The connection-level reuse itself is covered by mcp's
/// `reconnect_reuses_live_servers`.)
#[tokio::test]
async fn test_toggle_preserves_live_client() {
    let empty_dir = tempfile::tempdir().unwrap();
    let live = std::sync::Arc::new(
        crate::agent::mcp::McpClient::connect_isolated(
            empty_dir.path(),
            &std::collections::HashSet::new(),
        )
        .await,
    );

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());
    app.mcp_client = Some(live);
    let gen0 = app.mcp_connect_gen;

    app.toggle_mcp_server(0).await.unwrap();

    assert!(
        app.mcp_client.is_some(),
        "toggle must keep the live client (for the other servers' status), not null it"
    );
    assert_ne!(
        app.mcp_connect_gen, gen0,
        "generation should advance so the reconnect supersedes any stale one"
    );
    assert!(app.mcp_connecting, "a reconnect should be in flight");
}

#[test]
fn test_parse_mcp_add_input() {
    use super::super::session_impl::parse_mcp_add_input;
    // No name — the first token is the command; a shell-quoted path survives.
    let (command, args) = parse_mcp_add_input("npx -y srv \"/a b/c\"").unwrap();
    assert_eq!(command, "npx");
    assert_eq!(args, vec!["-y", "srv", "/a b/c"]);
    // A bare command with no args is fine.
    assert_eq!(
        parse_mcp_add_input("my-mcp-binary").unwrap(),
        ("my-mcp-binary".to_string(), Vec::<String>::new())
    );
    // Empty input is a usage error.
    assert!(parse_mcp_add_input("   ").is_err());
}

#[tokio::test]
async fn test_mcp_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without writing config.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("mcp overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[test]
fn test_mcp_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.adding = Some("fs npx".to_string());
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(screen.contains("fs npx"), "missing typed input:\n{screen}");
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

#[tokio::test]
async fn test_mcp_drill_in_tab_then_esc() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // Tab drills into the selected server's details.
    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert_eq!(state.viewing, Some(0), "Tab should open the detail view");
    } else {
        panic!("overlay closed on Tab instead of drilling in");
    }
    // Esc backs out to the list, NOT closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(state) = &app.overlay {
        assert!(state.viewing.is_none(), "Esc should back out of detail");
    } else {
        panic!("Esc in detail closed the overlay instead of returning to the list");
    }
}

#[test]
fn test_mcp_drill_in_renders_command_and_footer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = mcp_overlay_fixture();
    overlay.viewing = Some(0); // "filesystem", command "npx"
    app.overlay = Overlay::Mcp(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("command:"),
        "detail missing command:\n{screen}"
    );
    assert!(
        screen.contains("npx"),
        "detail missing the command value:\n{screen}"
    );
    assert!(
        screen.contains("esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// The drill-in tool list stacks each name over its wrapped description (no
/// far-right column): a `•` name line, then `    `-indented description lines,
/// with a blank line separating tools and long descriptions wrapping.
#[test]
fn test_mcp_tool_lines_stacks_name_over_wrapped_desc() {
    use super::super::overlay_render_impl::mcp_tool_lines;

    let line_text =
        |line: &Line| -> String { line.spans.iter().map(|s| s.content.as_ref()).collect() };
    let tools = [
        ("short_tool", "A brief description.", true),
        (
            "browserslist_compatibility_check",
            "Check web feature compatibility against your browserslist configuration across many supported browsers.",
            true,
        ),
    ];
    let lines: Vec<String> = mcp_tool_lines(&tools, 40).iter().map(line_text).collect();

    // Each name is on its own bulleted line.
    assert!(
        lines.iter().any(|l| l == "  • short_tool"),
        "first tool name not on its own line:\n{lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "  • browserslist_compatibility_check"),
        "second tool name not on its own line:\n{lines:#?}"
    );
    // The description sits indented beneath the name (not in a right-hand column).
    assert!(
        lines.iter().any(|l| l == "    A brief description."),
        "description not indented under the name:\n{lines:#?}"
    );
    // A blank line separates the two tools.
    assert!(
        lines.iter().any(|l| l.is_empty()),
        "expected a blank separator between tools:\n{lines:#?}"
    );
    // The long description wraps onto multiple indented lines, none over width.
    let desc_lines = lines
        .iter()
        .filter(|l| l.starts_with("    ") && l.contains("browserslist") || l.contains("supported"))
        .count();
    assert!(
        desc_lines >= 2,
        "long description should wrap to multiple lines:\n{lines:#?}"
    );
    assert!(
        lines.iter().all(|l| display_width(l) <= 40),
        "no rendered line should exceed the width:\n{lines:#?}"
    );
}

/// `/mcp` Ctrl+D arms a two-press delete (removal edits the user mcp.json), the
/// same confirm as /skills and the resume picker — the first press only arms and
/// surfaces a confirm prompt; Esc disarms without closing the overlay.
#[tokio::test]
async fn test_mcp_delete_arms_then_esc_disarms() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Mcp(mcp_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, Some(0), "first Ctrl+D arms"),
        _ => panic!("the overlay must stay open after the first Ctrl+D"),
    }
    let (screen, _rows) = render_full_screen(&mut app, 70, 20);
    assert!(
        screen.contains("confirm"),
        "an armed delete shows a confirm prompt:\n{screen}"
    );

    // Esc cancels the arm but must NOT close the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Mcp(state) => assert_eq!(state.pending_delete, None, "Esc disarms"),
        _ => panic!("Esc on an armed delete must not close the overlay"),
    }
}

/// A repo whose .mcp.json defines stdio servers must NOT spawn them silently:
/// `connect_mcp_with_consent` raises a consent card listing the exact commands
/// and leaves the decision Unknown until the user answers.
#[tokio::test]
async fn project_mcp_stdio_raises_consent_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"sh","args":["-c","echo hi"]}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;

    let prompt = app.cards.mcp_consent.as_ref().expect("a consent card");
    assert_eq!(
        prompt.servers,
        vec![("x".to_string(), "sh -c echo hi".to_string())],
        "the card lists the exact command to be run"
    );
    assert_eq!(
        app.project_mcp_consent,
        ProjectMcpConsent::Unknown,
        "no decision until the user answers"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// Denying the consent card holds the servers back for the session and clears
/// the card; nothing is persisted.
#[tokio::test]
async fn project_mcp_consent_deny_holds_back() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cards.mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "sh -c echo hi".to_string())],
        cwd: ".".to_string(),
        base_disabled: Default::default(),
    });
    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await;
    assert!(app.cards.mcp_consent.is_none(), "the card is cleared");
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Denied);
}

/// "always" approves for this repo: it persists to the per-repo allow-list (so a
/// future session in the same dir skips the card) and clears the card.
#[tokio::test]
async fn project_mcp_consent_always_persists() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-always-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    app.cards.mcp_consent = Some(McpConsentPrompt {
        servers: vec![("x".to_string(), "echo".to_string())],
        cwd: cwd.clone(),
        base_disabled: Default::default(),
    });

    app.handle_mcp_consent_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE))
        .await;
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    assert!(app.cards.mcp_consent.is_none());

    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let digest = project_mcp_digest(&[("x".to_string(), "echo".to_string())]);
    assert!(
        app.session_store
            .get_project_mcp_approved(&dir_key, &digest)
            .await,
        "'always' is persisted to the per-repo allow-list, bound to the server digest"
    );
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo already on the per-repo allow-list connects its project stdio servers
/// without re-prompting — the consent is seeded from the persistent store.
#[tokio::test]
async fn project_mcp_preapproved_repo_skips_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-pre-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        // Bogus command: the background connect's spawn fails fast, no real process.
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let servers = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&servers))
        .await
        .unwrap();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.cards.mcp_consent.is_none(),
        "a pre-approved repo doesn't prompt"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Allowed);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo previously approved "always" but whose `.mcp.json` then CHANGES (a
/// different command) re-prompts: the stored approval is bound to the server
/// content digest, so a swapped-in command can't ride the old consent.
#[tokio::test]
async fn project_mcp_changed_config_reprompts() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-changed-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    let cwd = repo.to_str().unwrap().to_string();
    let dir_key = std::fs::canonicalize(&repo)
        .unwrap()
        .to_string_lossy()
        .into_owned();

    // Approve the ORIGINAL server set.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_no_such_binary_zzz"}}}"#,
    )
    .unwrap();
    let orig = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
    app.session_store
        .set_project_mcp_approved(&dir_key, &project_mcp_digest(&orig))
        .await
        .unwrap();

    // The author swaps in a DIFFERENT command — the prior approval must not apply.
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"x":{"command":"aivo_evil_binary_zzz"}}}"#,
    )
    .unwrap();
    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.cards.mcp_consent.is_some(),
        "a changed .mcp.json re-prompts instead of reusing the old approval"
    );
    assert_eq!(app.project_mcp_consent, ProjectMcpConsent::Unknown);
    let _ = std::fs::remove_dir_all(&repo);
}

/// A repo with no project `.mcp.json` (or only HTTP servers) never raises the
/// consent card — there's no local command to gate.
#[tokio::test]
async fn project_mcp_no_stdio_no_card() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let repo = std::env::temp_dir().join(format!("aivo-consent-http-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&repo);
    std::fs::create_dir_all(&repo).unwrap();
    std::fs::write(
        repo.join(".mcp.json"),
        r#"{"mcpServers":{"remote":{"url":"https://h/mcp"}}}"#,
    )
    .unwrap();
    let cwd = repo.to_str().unwrap().to_string();

    app.connect_mcp_with_consent(cwd, Default::default()).await;
    assert!(
        app.cards.mcp_consent.is_none(),
        "HTTP-only project servers aren't gated (no local exec)"
    );
    let _ = std::fs::remove_dir_all(&repo);
}
