use super::super::*;
use super::helpers::*;
use tempfile::TempDir;

#[test]
fn test_matches_fuzzy() {
    assert!(matches_fuzzy("g4", "gpt-4o"));
    assert!(matches_fuzzy("", "anything"));
    assert!(!matches_fuzzy("xyz", "gpt-4o"));
}

#[test]
fn test_visible_command_menu_filters_and_hides_escaped_slash() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = 3;
    app.sync_command_menu_state();
    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.entries.len(), 2); // "mo" → model + memory
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
    assert!(matches!(
        menu.entries[1],
        ComposerMenuEntry::Command(command) if command.name == "memory"
    ));
    assert_eq!(menu.selected, Some(0));

    app.draft = "//literal".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());

    app.draft = "/model claude".to_string();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_filter_slash_commands_prefers_prefix_matches() {
    let matches = filter_slash_commands("m");
    assert_eq!(matches.first().map(|command| command.name), Some("model"));
}

#[test]
fn test_command_menu_area_prefers_below_when_above_space_is_tight() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) = command_menu_area(composer, frame, 6, None);
    assert_eq!(placement, CommandMenuPlacement::Below);
    assert!(area.y >= composer.y + composer.height);
}

#[test]
fn test_command_menu_area_respects_sticky_placement() {
    let composer = Rect::new(2, 1, 40, 2);
    let frame = Rect::new(0, 0, 80, 20);
    let (area, placement) =
        command_menu_area(composer, frame, 6, Some(CommandMenuPlacement::Above));
    assert_eq!(placement, CommandMenuPlacement::Above);
    assert_eq!(area.y, frame.y);
}

#[test]
fn test_command_menu_query_change_keeps_existing_placement_until_reopen() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.command_menu.placement = Some(CommandMenuPlacement::Above);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.command_menu.query = "m".to_string();
    app.command_menu.dismissed = false;

    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Above)
    );

    app.dismiss_command_menu();
    app.draft = "/mod".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(app.command_menu.placement, None);
}

#[test]
fn test_bare_slash_filtering_keeps_detected_placement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    app.command_menu.placement = Some(CommandMenuPlacement::Below);

    app.draft = "/new".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert_eq!(
        app.command_menu.placement,
        Some(CommandMenuPlacement::Below)
    );
}

#[test]
fn test_insert_selected_command_uses_argument_space_when_needed() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/m".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/model ");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[test]
fn test_insert_selected_command_omits_space_for_zero_arg_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/ex".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/exit");
    assert_eq!(app.cursor, app.draft.len());
    assert!(app.visible_command_menu().is_none());
}

#[tokio::test]
async fn test_ctrl_p_navigate_command_menu_before_history() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["previous prompt".to_string()];
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL))
        .await
        .unwrap();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.selected, Some(menu.entries.len() - 1));
    assert_eq!(app.draft, "/");
    assert!(app.draft_history_index.is_none());
}

#[tokio::test]
async fn test_history_nav_past_recalled_slash_command() {
    // Recalling a `/command` from history must not pop the command menu and
    // steal the arrow keys — ↑ keeps walking history past it.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_history = vec!["older".to_string(), "/help".to_string()];

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    // First ↑ recalls the newest entry (a slash command); the menu stays hidden.
    assert_eq!(app.draft, "/help");
    assert_eq!(app.draft_history_index, Some(1));
    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE))
        .await
        .unwrap();
    // Second ↑ continues up through history rather than navigating the menu.
    assert_eq!(app.draft, "older");
    assert_eq!(app.draft_history_index, Some(0));

    // ↓ steps back down past the slash command, too.
    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert_eq!(app.draft, "/help");
    assert_eq!(app.draft_history_index, Some(1));
}

#[tokio::test]
async fn test_escape_dismisses_command_menu_until_query_changes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mo".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();
    assert!(app.visible_command_menu().is_some());

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Left, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.visible_command_menu().is_none());

    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    let menu = app.visible_command_menu().unwrap();
    assert!(matches!(
        menu.entries[0],
        ComposerMenuEntry::Command(command) if command.name == "model"
    ));
}

#[tokio::test]
async fn test_enter_executes_selected_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    let should_exit = app
        .handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();

    assert!(app.draft.is_empty());
    assert_eq!(app.cursor, 0);
    assert!(should_exit);
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_render_command_menu_rows_shows_empty_state() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: Vec::new(),
        selected: None,
    };
    let lines = render_command_menu_rows(&menu, 32);
    let plain = plain_text_from_spans(&lines[0].spans);
    assert_eq!(plain, "No matching command");
}

#[test]
fn test_render_command_menu_rows_aligns_description_column() {
    let menu = VisibleCommandMenu {
        kind: MenuKind::Commands,
        entries: vec![
            ComposerMenuEntry::Command(&SLASH_COMMANDS[0]),
            ComposerMenuEntry::Command(&SLASH_COMMANDS[2]),
        ],
        selected: Some(0),
    };
    let lines = render_command_menu_rows(&menu, 48);
    let first = plain_text_from_spans(&lines[0].spans);
    let second = plain_text_from_spans(&lines[1].spans);
    let first_desc = first.find("start a fresh session").unwrap();
    let second_desc = second.find("resume a saved session").unwrap();
    assert_eq!(first_desc, second_desc);
}

#[test]
fn test_collect_attach_path_suggestions_lists_matching_entries() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::write(temp_dir.path().join("alpha.txt"), "hi").unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let entries = collect_attach_path_suggestions(temp_dir.path().to_str().unwrap(), "a");

    assert!(entries.iter().any(|entry| entry.label == "assets/"));
    assert!(entries.iter().any(|entry| entry.label == "alpha.txt"));
}

#[test]
fn test_attach_query_uses_path_menu_and_tab_inserts_selected_path() {
    let temp_dir = TempDir::new().unwrap();
    std::fs::create_dir(temp_dir.path().join("assets")).unwrap();

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = temp_dir.path().to_string_lossy().into_owned();
    app.draft = "/attach a".to_string();
    app.cursor = app.draft.len();
    app.sync_command_menu_state();

    let menu = app.visible_command_menu().unwrap();
    assert_eq!(menu.kind, MenuKind::AttachPath);
    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "/attach assets/");
    // Menu stays open after tab on a directory so the user can continue navigating.
    assert!(app.visible_command_menu().is_some());
}

/// The `@` mention menu: word-boundary trigger (mid-message ok, emails no),
/// prefix-first filtering over discovered profiles, and completion that inserts
/// `@name ` at the token without submitting the draft.
#[test]
fn test_at_mention_menu_completes_subagent_names() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let profile = |name: &str| crate::agent::subagents::Subagent {
        name: name.to_string(),
        description: format!("{name} does things. Extra sentence."),
        model: None,
        tools: None,
        body: String::new(),
        isolation_worktree: false,
        repo_local: false,
        source: std::path::PathBuf::new(),
    };

    // No discovered profiles → no menu, even on a bare `@`.
    app.draft = "@".to_string();
    app.cursor = 1;
    assert!(app.active_mention_query().is_none());

    app.last_subagents = vec![profile("code-reviewer"), profile("architect")];

    // Mid-message mention at a word boundary; query is the partial after `@`.
    app.draft = "use @co on the diff".to_string();
    app.cursor = 7; // after "use @co"
    let (at, query) = app.active_mention_query().expect("mention active");
    assert_eq!((at, query.as_str()), (4, "co"));
    let menu = app.visible_command_menu().expect("menu visible");
    assert!(matches!(menu.kind, MenuKind::Mention));
    assert_eq!(menu.entries.len(), 1);
    assert_eq!(menu.entries[0].label(), "@code-reviewer");

    // Tab completion replaces just the token and keeps composing (no submit).
    assert!(app.insert_selected_command());
    assert_eq!(app.draft, "use @code-reviewer  on the diff");
    assert_eq!(app.cursor, "use @code-reviewer ".len());

    // A bare `@` lists every profile.
    app.draft = "@".to_string();
    app.cursor = 1;
    app.command_menu.reset();
    let menu = app.visible_command_menu().expect("menu visible");
    assert_eq!(menu.entries.len(), 2);

    // No word boundary (email-style) → no menu; ditto once the token has a space.
    app.draft = "mail me a@b".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());
    app.draft = "@code-reviewer go".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());

    // `/attach` path mode wins over mention parsing…
    app.draft = "/attach @x".to_string();
    app.cursor = app.draft.len();
    assert!(app.active_mention_query().is_none());
    // …but a mention inside an ordinary command ARGUMENT works (`/goal`, `/plan`
    // steering a named agent is a legit composition).
    app.draft = "/goal ship it with @arch".to_string();
    app.cursor = app.draft.len();
    let (_, query) = app.active_mention_query().expect("mention in command arg");
    assert_eq!(query, "arch");
}

#[test]
fn test_composer_command_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let set = |app: &mut CodeTuiApp, draft: &str| {
        app.draft = draft.to_string();
        app.cursor = app.draft.len();
    };

    // Bare command (and a trailing space) → ghost hint with the arg syntax.
    set(&mut app, "/mcp");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add [-p] <command>")),
        "typing /mcp should ghost the add/rm syntax"
    );
    set(&mut app, "/mcp ");
    assert!(
        app.composer_command_hint().is_some(),
        "trailing space still bare"
    );
    // Once a real argument is typed, the hint clears.
    set(&mut app, "/mcp add");
    assert!(
        app.composer_command_hint().is_none(),
        "args typed → no ghost"
    );

    // Other argument-taking commands ghost their arg syntax too.
    for (draft, expect) in [
        ("/model", "[name]"),
        ("/key", "[id|name]"),
        ("/resume", "[query]"),
        ("/copy", "[n]"),
        ("/detach", "<n>"),
    ] {
        set(&mut app, draft);
        assert_eq!(
            app.composer_command_hint(),
            Some(expect),
            "{draft} should ghost {expect}"
        );
    }
    // `/skills` ghosts its add/rm subcommand syntax, like `/mcp`.
    set(&mut app, "/skills");
    assert!(
        app.composer_command_hint()
            .is_some_and(|h| h.contains("add [-p] <name>")),
        "/skills should ghost the add/rm syntax"
    );

    // `/attach` is deliberately hint-less: typing `/attach ` opens path
    // completion, and a ghost would suppress that menu.
    set(&mut app, "/attach");
    assert!(
        app.composer_command_hint().is_none(),
        "attach must not ghost (path menu owns it)"
    );
    // A no-argument command has no hint either.
    set(&mut app, "/new");
    assert!(app.composer_command_hint().is_none());

    // None for a literal slash or plain text.
    set(&mut app, "//mcp");
    assert!(app.composer_command_hint().is_none());
    set(&mut app, "hello");
    assert!(app.composer_command_hint().is_none());
    // Cursor not at the end → no ghost (e.g. editing mid-line).
    app.draft = "/mcp".to_string();
    app.cursor = 2;
    assert!(app.composer_command_hint().is_none());
}

#[test]
fn test_mcp_ghost_hint_renders_in_composer() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "/mcp".to_string();
    app.cursor = app.draft.len();

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
    // The command and its ghost arg-hint share the composer line.
    assert!(
        screen.contains("/mcp [add [-p] <command>"),
        "composer should show the inline ghost hint:\n{screen}"
    );
}
