use super::super::*;
use super::helpers::*;

#[test]
fn test_question_mark_is_not_help_shortcut() {
    let question = KeyEvent::new(KeyCode::Char('?'), KeyModifiers::NONE);
    let f1 = KeyEvent::new(KeyCode::F(1), KeyModifiers::NONE);
    assert!(!is_help_shortcut(question));
    assert!(is_help_shortcut(f1));
}

#[tokio::test]
async fn test_help_overlay_groups_lists_every_command_and_scrolls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `/help` opens the overlay at the top of its body.
    app.open_help_overlay();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));

    // A tall render shows the top: the section header, every purpose group, and
    // every command label (commands sit before the fold, so they all fit).
    let (top, _) = render_full_screen(&mut app, 90, 70);
    assert!(top.contains("Slash commands"), "missing header:\n{top}");
    for group in [
        "Session",
        "Model & key",
        "Context",
        "Skills & tools",
        "Autonomous",
    ] {
        assert!(top.contains(group), "missing command group {group}:\n{top}");
    }
    for command in SLASH_COMMANDS {
        // Account commands are hidden on this (non-aivo) test key.
        if !app.slash_command_visible(command.name) {
            assert!(
                !top.contains(command.help_label),
                "hidden command {} leaked into help:\n{top}",
                command.help_label
            );
            continue;
        }
        assert!(
            top.contains(command.help_label),
            "command {} missing from help:\n{top}",
            command.help_label
        );
    }
    // The aivo-only account group is absent on a BYOK key.
    assert!(
        !top.contains("aivo account"),
        "account group shown on a non-aivo key:\n{top}"
    );
    // Every visible command is grouped, so the completeness-guard "More" bucket is empty.
    assert!(
        !top.contains("More"),
        "unexpected ungrouped commands:\n{top}"
    );

    // End scrolls to the bottom; the keybindings + text-entry tips are reachable.
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let (bottom, _) = render_full_screen(&mut app, 90, 24);
    let scrolled = match app.overlay {
        Overlay::Help { scroll } => scroll,
        _ => panic!("help overlay closed unexpectedly"),
    };
    assert!(scrolled > 0, "End did not scroll the help body");
    assert!(
        bottom.contains("Keybindings") || bottom.contains("Text entry"),
        "bottom sections not reachable by scrolling:\n{bottom}"
    );
    assert!(
        bottom.contains("shell command"),
        "text-entry tip not reachable:\n{bottom}"
    );

    // Home snaps back to the top; Esc closes.
    app.handle_key(KeyEvent::new(KeyCode::Home, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Help { scroll: 0 }));
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_account_commands_gated_to_aivo_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The default test key is BYOK → account commands are hidden and refused.
    assert!(!app.is_aivo_account_key());
    for name in ["login", "logout", "usage"] {
        assert!(
            !app.slash_command_visible(name),
            "/{name} should be hidden on a BYOK key"
        );
    }
    // `/usage` on a BYOK key is a no-op with a hint — no task spawned.
    app.run_usage_command().await;
    assert!(app.account.task.is_none());
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("aivo provider")),
        "expected the aivo-only hint, got {:?}",
        app.notice
    );

    // On the bundled aivo starter key the three commands surface.
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();
    assert!(app.is_aivo_account_key());
    for name in ["login", "logout", "usage"] {
        assert!(
            app.slash_command_visible(name),
            "/{name} should show on the aivo key"
        );
    }
    // The `/` menu now offers them.
    let entries = app.matching_command_entries("login");
    assert!(
        entries.iter().any(|e| e.label() == "/login"),
        "/login missing from the menu on the aivo key"
    );
}

#[tokio::test]
async fn test_account_login_card_flow_and_stale_generation() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();

    // Stand in for `run_login_command` (no network poll): notice, no card yet.
    app.account.generation = 7;
    app.notice = Some((MUTED(), "Starting sign-in…".to_string()));
    assert!(app.account.login.is_none());

    // The device code + URL arrive → the card appears, notice cleared.
    app.apply_account_login_prompt(
        7,
        Ok((
            "WXYZ-1234".to_string(),
            "https://getaivo.dev/device?code=WXYZ-1234".to_string(),
        )),
    );
    assert!(app.notice.is_none(), "starting notice not cleared");
    let (frame, _) = render_full_screen(&mut app, 80, 24);
    assert!(frame.contains("sign in to aivo"), "title missing:\n{frame}");
    assert!(frame.contains("WXYZ-1234"), "code missing:\n{frame}");
    assert!(
        frame.contains("Waiting for approval…"),
        "status missing:\n{frame}"
    );
    assert!(
        frame.contains("Enter open browser"),
        "key hints missing:\n{frame}"
    );
    // Empty session parks the composer at top → the card takes the space below.
    assert!(
        frame.find("Ask, plan, or build").unwrap() < frame.find("sign in to aivo").unwrap(),
        "card should sit below the parked composer:\n{frame}"
    );

    // A prompt stamped with a stale generation is ignored (card stays).
    app.apply_account_login_prompt(3, Err("boom".to_string()));
    assert!(app.account.login.is_some(), "stale error dropped the card");

    // Esc with a non-empty composer belongs to the draft — the card stays.
    app.draft = "half a thought".to_string();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account.login.is_some(), "Esc stole the draft's key");

    // Esc on an empty composer cancels: card gone, generation bumped.
    app.draft.clear();
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account.login.is_none());
    assert_ne!(app.account.generation, 7, "cancel must invalidate the flow");

    // A late success for the cancelled flow is dropped (no login notice).
    app.apply_account_login_done(7, Ok("Logged in as x".to_string()))
        .await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("cancelled")),
        "late result overwrote the cancel notice: {:?}",
        app.notice
    );

    // A current-generation success drops the TUI's starter catalog.
    let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
    app.cache
        .set(sentinel, vec!["aivo/starter".to_string()])
        .await;
    let account_gen = app.account.generation;
    app.apply_account_login_done(account_gen, Ok("Logged in as x".to_string()))
        .await;
    assert!(
        app.cache.model_ids(sentinel).await.is_none(),
        "login left the TUI's starter catalog stale"
    );
}

#[tokio::test]
async fn test_account_usage_runs_the_cli_as_a_local_command() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key.base_url = crate::constants::AIVO_STARTER_SENTINEL.to_string();

    // `/usage` runs the CLI itself through the `!` machinery.
    app.run_usage_command().await;
    let run = app
        .local_command
        .as_ref()
        .expect("no local command spawned");
    assert_eq!(run.command, "aivo account usage");
    // Kill it before it does anything — this test is wiring-only.
    app.interrupt_local_command().await.unwrap();
    assert!(app.local_command.is_none());

    // A second `/usage` while one is still streaming is refused like any `!cmd`.
    app.run_usage_command().await;
    assert!(app.local_command.is_some());
    app.run_usage_command().await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("already running")),
        "expected the busy notice, got {:?}",
        app.notice
    );
    app.interrupt_local_command().await.unwrap();
}

#[tokio::test]
async fn test_logout_confirm_card_and_stale_done() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // The confirm card owns the keyboard: n dismisses without unlinking.
    app.account.pending_logout = Some("me@example.com".to_string());
    let (frame, _) = render_full_screen(&mut app, 80, 24);
    assert!(
        frame.contains("sign out of aivo"),
        "title missing:\n{frame}"
    );
    assert!(
        frame.contains("me@example.com"),
        "account missing:\n{frame}"
    );
    app.handle_key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(app.account.pending_logout.is_none());
    assert!(app.account.task.is_none(), "deny must not spawn an unlink");

    // A stale unlink result is ignored; the current one lands as a notice.
    let sentinel = crate::constants::AIVO_STARTER_SENTINEL;
    app.cache
        .set(sentinel, vec!["aivo/starter".to_string()])
        .await;
    app.account.generation = 4;
    app.apply_account_logout_done(1, Ok(())).await;
    assert!(
        app.notice
            .as_ref()
            .is_none_or(|(_, m)| !m.contains("Logged out")),
        "stale result produced a notice: {:?}",
        app.notice
    );
    assert!(
        app.cache.model_ids(sentinel).await.is_some(),
        "stale result cleared the catalog"
    );
    app.apply_account_logout_done(4, Ok(())).await;
    assert!(
        app.notice
            .as_ref()
            .is_some_and(|(_, m)| m.contains("Logged out")),
        "expected the logout confirmation, got {:?}",
        app.notice
    );
    // The TUI's own catalog (distinct from the shared instance) dropped too.
    assert!(
        app.cache.model_ids(sentinel).await.is_none(),
        "logout left the TUI's starter catalog stale"
    );
}

#[tokio::test]
async fn test_config_overlay_toggles_thinking() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.thinking_enabled = true;

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::Thinking)
        .expect("Thinking row present");
    // `on` is segment 0 — the renderer derives the highlighted pill from this.
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 0);

    // Advancing the switch flips the live flag (off is segment 1).
    app.cycle_config_setting(idx).await;
    assert!(!app.thinking_enabled);
    assert_eq!(app.config_segments(ConfigSetting::Thinking).active, 1);
}

#[tokio::test]
async fn test_config_overlay_cycles_theme() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert_eq!(app.theme, UiTheme::Dark);
    assert_eq!(ui_theme(), UiTheme::Dark);
    assert_eq!(TEXT(), Palette::DARK.text);

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    assert_eq!(state.items[0].setting, ConfigSetting::Theme);

    app.cycle_config_setting(0).await;
    assert_eq!(app.theme, UiTheme::Light);
    assert_eq!(ui_theme(), UiTheme::Light);
    assert_eq!(TEXT(), Palette::LIGHT.text);

    // Light mode paints the warm-paper canvas across the whole screen so dark ink
    // stays readable even on a dark terminal; dark mode keeps the terminal's own bg.
    let canvas = Palette::LIGHT.canvas.expect("light theme fills the canvas");
    assert!(
        Palette::DARK.canvas.is_none(),
        "dark theme keeps the terminal bg"
    );
    {
        use ratatui::backend::TestBackend;
        let mut terminal = Terminal::new(TestBackend::new(60, 12)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        // The floating transcript/overlay regions are `Clear`ed and must be
        // repainted with the canvas, not left on the terminal's native bg — so the
        // paper reaches the interior, not just the uncleared margins. A strong
        // majority of cells should carry the canvas fill.
        let cells = terminal.backend().buffer().content();
        let on_canvas = cells.iter().filter(|c| c.bg == canvas).count();
        assert!(
            on_canvas * 2 > cells.len(),
            "light canvas must fill cleared regions ({on_canvas}/{} cells)",
            cells.len()
        );
    }

    app.cycle_config_setting(0).await;
    assert_eq!(app.theme, UiTheme::Dark);
    assert_eq!(ui_theme(), UiTheme::Dark);
}

#[test]
fn resolve_startup_theme_stored_choice_else_dark() {
    use crate::services::session_store::ChatTheme;
    assert_eq!(
        resolve_startup_theme(Some(ChatTheme::Light)),
        UiTheme::Light
    );
    assert_eq!(resolve_startup_theme(Some(ChatTheme::Dark)), UiTheme::Dark);
    // Unset (first launch) defaults to dark; /config persists a choice.
    assert_eq!(resolve_startup_theme(None), UiTheme::Dark);
}

#[tokio::test]
async fn test_config_overlay_toggles_agent_tools() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.agent_tools_enabled = true;

    app.open_config_overlay();
    let Overlay::Config(state) = &app.overlay else {
        panic!("expected config overlay");
    };
    let idx = state
        .items
        .iter()
        .position(|i| i.setting == ConfigSetting::AgentTools)
        .expect("Agent tools row present");
    assert_eq!(app.config_segments(ConfigSetting::AgentTools).active, 0);

    app.cycle_config_setting(idx).await;
    assert!(!app.agent_tools_enabled);
    assert_eq!(app.config_segments(ConfigSetting::AgentTools).active, 1);
}

#[tokio::test]
async fn test_config_approval_radio_is_mutually_exclusive() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Three standing modes; plan is a transient state, not a segment here.
    assert_eq!(
        app.config_segments(ConfigSetting::Approval).options,
        &["normal", "auto-approve", "review"]
    );

    // Fresh session: normal mode — segment 0 is live.
    assert!(!app.agent_auto_approve && !app.agent_review_edits && !app.plan_mode);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 0);

    // Auto-approve sets exactly one flag.
    app.set_approval_mode("auto-approve").await;
    assert!(app.agent_auto_approve && !app.agent_review_edits);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 1);

    // Switching to Review clears auto-approve — the fold's whole point.
    app.set_approval_mode("review").await;
    assert!(app.agent_review_edits && !app.agent_auto_approve);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 2);

    // Back to Normal leaves every mode off.
    app.set_approval_mode("normal").await;
    assert!(!app.agent_auto_approve && !app.agent_review_edits && !app.plan_mode);
    assert_eq!(app.config_segments(ConfigSetting::Approval).active, 0);
}

#[tokio::test]
async fn config_overlay_renders_segmented_switches() {
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_config_overlay();

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buffer = terminal.backend().buffer();
    let text: String = (0..buffer.area.height)
        .map(|y| {
            (0..buffer.area.width)
                .map(|x| buffer.cell((x, y)).map_or(" ", |c| c.symbol()))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Every segment of every switch is visible inline — not just the active value.
    for label in [
        "dark",
        "light",
        "on",
        "off",
        "normal",
        "auto-approve",
        "review",
    ] {
        assert!(text.contains(label), "segment {label:?} missing:\n{text}");
    }
    // The footer advertises ←→ as the way to change a value.
    assert!(
        text.contains("change"),
        "footer missing change hint:\n{text}"
    );
}

#[tokio::test]
async fn test_clicking_footer_session_id_opens_overlay() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "import-claude-a1b2c3d4".to_string();

    // Render the footer so the id's click rect is recorded.
    let mut terminal = Terminal::new(TestBackend::new(100, 1)).unwrap();
    terminal
        .draw(|frame| app.render_footer(frame, frame.area()))
        .unwrap();
    let hit = app.session_id_hit.expect("session id click rect recorded");

    // A click inside that rect opens the detail overlay.
    app.handle_mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: hit.x,
        row: hit.y,
        modifiers: KeyModifiers::NONE,
    })
    .await
    .unwrap();
    assert!(matches!(app.overlay, Overlay::Session { .. }));
}

#[tokio::test]
async fn test_overlay_backdrop_click_dismisses_help() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Help { scroll: 0 };

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let hit = app.overlay_hitbox.expect("overlay box recorded");

    // Inside the box the press falls through to text selection — overlay stays.
    app.handle_mouse(left_click(hit.x + hit.width / 2, hit.y + hit.height / 2))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Help { .. }));

    // On the backdrop it dismisses, like Esc.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_overlay_backdrop_click_steps_back_like_esc() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(SkillsOverlay {
        items: Vec::new(),
        selected: 0,
        query: "abc".to_string(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    });

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    assert!(app.overlay_hitbox.is_some());

    // First backdrop press clears the filter (Esc's first stage)…
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    match &app.overlay {
        Overlay::Skills(state) => assert!(state.query.is_empty()),
        _ => panic!("overlay closed too early"),
    }

    // …the next one closes the overlay.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_loading_picker_backdrop_click_closes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Picker(Box::new(PickerState::loading(
        "Select model",
        String::new(),
        PickerKind::Key,
    )));

    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let list_area = app
        .picker_hitbox
        .as_ref()
        .expect("loading picker box recorded")
        .list_area;

    // A click on the (empty) list while loading is inert.
    app.handle_mouse(left_click(list_area.x, list_area.y))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::Picker(_)));

    // A backdrop click closes it, same as a loaded picker.
    app.handle_mouse(left_click(0, 0)).await.unwrap();
    assert!(matches!(app.overlay, Overlay::None));
}

#[test]
fn test_session_overlay_shows_full_id_and_provenance() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "import-claude-a1b2c3d4".to_string();
    app.open_session_overlay();
    assert!(matches!(app.overlay, Overlay::Session { .. }));

    // Tall + wide enough that the whole (short) detail box shows without scrolling
    // and the resume command doesn't wrap.
    let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut text = String::new();
    for y in 0..buf.area.height {
        for x in 0..buf.area.width {
            text.push_str(buf[(x, y)].symbol());
        }
        text.push('\n');
    }
    // Full id (the footer only had room for a handle), plus the fork provenance
    // and the resume command.
    assert!(text.contains("import-claude-a1b2c3d4"), "overlay:\n{text}");
    assert!(text.contains("Claude (forked)"), "overlay:\n{text}");
    assert!(
        text.contains("aivo code --resume import-claude-a1b2c3d4"),
        "overlay:\n{text}"
    );
}

/// Creating a subagent is natural-language only, by design: there is NO
/// `/create-agent` slash command (it would be redundant with the advertised
/// skill and clutter the menu). The workflow is instead exposed to the model as
/// a folderless built-in skill it reaches for on a request like "make a
/// code-reviewer subagent".
#[test]
fn test_create_agent_has_no_slash_command() {
    // Not registered as a typeable command — absent from the menu/help and unknown
    // to the parser.
    assert!(
        !SLASH_COMMANDS.iter().any(|c| c.name == "create-agent"),
        "create-agent must NOT be a slash command — it's natural-language only"
    );
    assert!(
        parse_slash_command("create-agent").is_err(),
        "typing /create-agent is an unknown command, not a builtin"
    );

    // The workflow still exists as a model-facing builtin skill (this is what the
    // send path injects into the engine's skill list to advertise it).
    let sc = crate::agent::skills::create_agent_builtin();
    assert_eq!(sc.name, "create-agent");
    assert!(!sc.body.is_empty());
}

/// `/agents`: registered as a typeable command; bare opens the overlay, `rm`
/// on an unknown name reports instead of erroring, and anything else prints
/// usage (there is no `add` — creation is conversational by design).
#[tokio::test]
async fn test_agents_command_opens_overlay_and_validates_args() {
    assert!(SLASH_COMMANDS.iter().any(|c| c.name == "agents"));
    assert!(matches!(
        parse_slash_command("agents"),
        Ok(SlashCommand::Agents(None))
    ));

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_agents_command(None).await.unwrap();
    assert!(matches!(app.overlay, Overlay::Agents(_)));

    app.overlay = Overlay::None;
    app.run_agents_command(Some("rm no-such-agent".to_string()))
        .await
        .unwrap();
    assert!(matches!(app.overlay, Overlay::None));
    let notice = app.notice.as_ref().expect("notice set").1.clone();
    assert!(notice.contains("no-such-agent"), "{notice}");

    app.run_agents_command(Some("add reviewer".to_string()))
        .await
        .unwrap();
    let notice = app.notice.as_ref().expect("usage notice").1.clone();
    assert!(notice.contains("Usage: /agents"), "{notice}");

    // Built-ins can't be removed — the notice points at shadowing instead.
    app.run_agents_command(Some("rm explorer".to_string()))
        .await
        .unwrap();
    let notice = app.notice.as_ref().expect("builtin notice").1.clone();
    assert!(notice.contains("built into aivo"), "{notice}");
}

/// The `/agents` empty state reads intact on a narrow terminal: the body clips
/// rather than wraps, so every line must be pre-wrapped short enough (~40 cols)
/// — the quoted example must survive whole, not as "make me a cod".
#[test]
fn test_agents_overlay_empty_state_fits_narrow_terminals() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Agents(AgentsOverlay::default());
    let (top, _) = render_full_screen(&mut app, 46, 18);
    // Keep only the modal interior (between the │ borders), then collapse
    // whitespace: wrapping may split lines, but every WORD must survive whole.
    let interior: String = top
        .lines()
        .filter_map(|row| {
            let first = row.find('\u{2502}')?;
            let last = row.rfind('\u{2502}')?;
            (last > first).then(|| row[first + '\u{2502}'.len_utf8()..last].to_string())
        })
        .collect::<Vec<_>>()
        .join(" ");
    let flat = interior.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains("No sub-agents yet"), "{top}");
    assert!(
        flat.contains("\u{201c}make me a code-reviewer subagent\u{201d}"),
        "quoted example clipped:\n{top}"
    );
    assert!(flat.contains("or drop a <name>.md profile in:"), "{top}");
    assert!(flat.contains("~/.config/aivo/agents"), "{top}");
}

/// A bracketed paste lands in the open overlay's text input (add field first,
/// else filter) instead of being dropped.
#[tokio::test]
async fn test_paste_routes_into_overlay_inputs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some("".to_string());
    app.overlay = Overlay::Skills(overlay);
    assert!(app.overlay_paste("github:anthropics/skills\n"));
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("github:anthropics/skills"));
    } else {
        panic!("skills overlay vanished");
    }

    let mut overlay = skills_overlay_fixture();
    overlay.adding = None;
    app.overlay = Overlay::Skills(overlay);
    assert!(app.overlay_paste("brand"));
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.query, "brand");
    } else {
        panic!("skills overlay vanished");
    }

    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:o/r".to_string(),
        ..Default::default()
    });
    assert!(app.overlay_paste("pdf"));
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert_eq!(state.query, "pdf");
    } else {
        panic!("install picker vanished");
    }

    app.overlay = Overlay::None;
    assert!(!app.overlay_paste("plain text"));
}

#[test]
fn test_overlay_hides_input_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    assert!(app.should_show_input_cursor());

    app.overlay = Overlay::Picker(Box::new(PickerState::loading(
        "Select model",
        String::new(),
        PickerKind::Model {
            target: ModelSelectionTarget::CurrentChat,
            auto_accept_exact: false,
        },
    )));

    assert!(!app.should_show_input_cursor());
}
