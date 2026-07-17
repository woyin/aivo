use super::super::*;

pub(super) fn make_test_app(
    tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) -> CodeTuiApp {
    // Unique throwaway stores — NEVER the real `~/.config/aivo` / models
    // cache. Tests that drive a save (persist / flush / turn-finish) would
    // otherwise write through them (the process HOME sandbox is the backstop).
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("aivo-test-{}-{n}", std::process::id()));
    let mut app = CodeTuiApp::bare(
        tx,
        rx,
        SessionStore::with_path(dir.join("config.json")),
        ModelsCache::with_path(dir.join("models-cache.json")),
        reqwest::Client::new(),
        ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            "https://api.anthropic.com".to_string(),
            None,
            String::new(),
        ),
    );
    // Tests assume a thinking-capable model (production resolves this per
    // model via `refresh_context_window`; bare's neutral default is false).
    app.model_supports_thinking = true;
    app
}

pub(super) fn seed_two_exchanges(app: &mut CodeTuiApp) {
    for (role, content) in [
        ("user", "first question"),
        ("assistant", "first answer"),
        ("user", "second question"),
        ("assistant", "second answer"),
    ] {
        app.history.push(ChatMessage {
            model: None,
            role: role.to_string(),
            content: content.to_string(),
            reasoning_content: None,
            attachments: vec![],
        });
    }
}

pub(super) fn left_click(column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

// Render the whole screen (transcript + composer + any card/overlay) to a plain
// string plus the per-row strings, for layout assertions.
pub(super) fn render_full_screen(app: &mut CodeTuiApp, w: u16, h: u16) -> (String, Vec<String>) {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut rows = Vec::new();
    for y in 0..h {
        let mut row = String::new();
        for x in 0..w {
            row.push_str(buf[(x, y)].symbol());
        }
        rows.push(row);
    }
    (rows.join("\n"), rows)
}

pub(super) fn render_screen(app: &mut CodeTuiApp, w: u16, h: u16) -> String {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(super) fn skill_command(name: &str, description: &str) -> SkillCommand {
    SkillCommand {
        name: name.to_string(),
        description: description.to_string(),
    }
}

pub(super) fn skills_overlay_fixture() -> SkillsOverlay {
    use crate::agent::skills::SkillScope;
    SkillsOverlay {
        items: vec![
            SkillToggle {
                name: "brandkit".to_string(),
                description: "Premium brand-kit image generation.".to_string(),
                enabled: true,
                dir: std::path::PathBuf::from("/home/me/.config/aivo/skills/brandkit"),
                scope: SkillScope::User,
                body: "Step 1. Render the boards.".to_string(),
            },
            SkillToggle {
                name: "critique".to_string(),
                description: "Evaluate design effectiveness from a UX perspective.".to_string(),
                enabled: false,
                dir: std::path::PathBuf::from("/repo/.agents/skills/critique"),
                scope: SkillScope::Project,
                body: "Step 1. Score the design.".to_string(),
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

pub(super) fn mcp_overlay_fixture() -> McpOverlay {
    use crate::agent::mcp::ServerScope;
    McpOverlay {
        items: vec![
            McpServerRow {
                name: "filesystem".to_string(),
                status: "5 tools".to_string(),
                health: McpHealth::Connected,
                enabled: true,
                scope: ServerScope::User,
                command: "npx".to_string(),
                remote: false,
            },
            McpServerRow {
                name: "github".to_string(),
                status: "off".to_string(),
                health: McpHealth::Disabled,
                enabled: false,
                scope: ServerScope::User,
                command: "docker".to_string(),
                remote: false,
            },
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    }
}

pub(super) fn wheel(kind: MouseEventKind) -> MouseEvent {
    MouseEvent {
        kind,
        column: 0,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }
}

pub(super) fn test_screen(terminal: &ratatui::Terminal<ratatui::backend::TestBackend>) -> String {
    let buf = terminal.backend().buffer().clone();
    let area = *buf.area();
    let mut screen = String::new();
    for y in area.y..area.y + area.height {
        for x in area.x..area.x + area.width {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    screen
}

pub(super) fn one_user_message(
    content: &str,
) -> Vec<crate::services::session_store::StoredChatMessage> {
    vec![crate::services::session_store::StoredChatMessage {
        model: None,
        role: "user".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        id: None,
        timestamp: None,
        attachments: None,
    }]
}
