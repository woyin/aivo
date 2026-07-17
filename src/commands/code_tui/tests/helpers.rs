use super::super::*;

pub(super) fn make_test_app(
    tx: tokio::sync::mpsc::UnboundedSender<RuntimeEvent>,
    rx: tokio::sync::mpsc::UnboundedReceiver<RuntimeEvent>,
) -> CodeTuiApp {
    CodeTuiApp {
        // A unique throwaway store — NEVER the real `~/.config/aivo` (which
        // `SessionStore::new()` points at). Tests that drive a save (persist /
        // flush / turn-finish) would otherwise pollute the user's real config.
        session_store: {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!("aivo-test-{}-{n}", std::process::id()));
            SessionStore::with_path(dir.join("config.json"))
        },
        // Same isolation: `new()` points at the real models-cache.json and the
        // account flows write through this instance.
        cache: {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("aivo-test-cache-{}-{n}", std::process::id()));
            ModelsCache::with_path(dir.join("models-cache.json"))
        },
        client: reqwest::Client::new(),
        key: ApiKey::new_with_protocol(
            "test".to_string(),
            "test".to_string(),
            "https://api.anthropic.com".to_string(),
            None,
            String::new(),
        ),
        copilot_tm: None,
        cwd: String::new(),
        real_cwd: String::new(),
        git_branch: None,
        git_branch_checked_at: None,
        raw_model: String::new(),
        model: String::new(),
        billed_model: None,
        turn_model: None,
        format: ChatFormat::OpenAI,
        history: Vec::new(),
        draft: String::new(),
        draft_attachments: Vec::new(),
        cursor: 0,
        command_menu: CommandMenuState::default(),
        skill_commands: Vec::new(),
        last_subagents: Vec::new(),
        mcp_configured_count: 0,
        welcome_tip_index: 0,
        welcome_tip_rotated_at: None,
        draft_history: Vec::new(),
        draft_history_all: Vec::new(),
        draft_history_index: None,
        draft_history_stash: None,
        session_id: String::new(),
        overlay: Overlay::None,
        notice: None,
        pending_response: String::new(),
        incoming_buffer: String::new(),
        pending_finish: None,
        pending_reasoning: String::new(),
        pending_submit: None,
        sending: false,
        request_started_at: None,
        compact_before: None,
        last_tool_action: None,
        wait_tick: None,
        last_stream_activity: None,
        subagent_rows: Vec::new(),
        tool_output_tail: std::collections::VecDeque::new(),
        tool_output_partial: String::new(),
        status_display: None,
        turn_output_tokens: 0,
        retrying: false,
        last_usage: None,
        live_usage: None,
        context_tokens: 0,
        session_tokens: crate::services::session_store::SessionTokens::default(),
        session_cost_usd: 0.0,
        context_window: 0,
        context_window_override: None,
        injected_context: None,
        injected_context_summary: None,
        context_is_estimate: true,
        follow_output: true,
        transcript_revision: 0,
        transcript_scroll: 0,
        transcript_width: 0,
        transcript_view_height: 0,
        last_max_scroll: None,
        transcript_hitbox: None,
        jump_to_bottom_hit: None,
        session_id_hit: None,
        composer_text_area: None,
        composer_scroll: 0,
        transcript_cache: None,
        volatile_tail_cache: None,
        transcript_selection: None,
        transcript_drag_active: false,
        screen_selection: None,
        screen_drag_active: false,
        screen_surface: None,
        screen_region: None,
        drag_autoscroll: None,
        last_autoscroll: None,
        last_click: None,
        selection_flash_until: None,
        scroll_speed: DEFAULT_CHAT_SCROLL_SPEED,
        swipe_scroll: false,
        toast: None,
        tx,
        rx,
        response_task: None,
        resume_task: None,
        resume_request_id: 0,
        loading_resume: None,
        resume_restore_state: None,
        session_preview_cache: std::collections::HashMap::new(),
        session_preview_pending: None,
        session_preview_task: None,
        reduce_motion: false,
        frame_tick: 0,
        picker_hitbox: None,
        overlay_detail_area: None,
        overlay_hitbox: None,
        exit_confirm_pending: false,
        goal_stop_confirm_pending: false,
        pending_ctrl_x: false,
        pending_external_edit: false,
        cursor_acp_session: None,
        cursor_prewarm: None,
        cursor_plan_mode: false,
        pending_agent_messages: None,
        pristine_import_len: None,
        goal_mode: None,
        goal_guard_stop: None,
        plan_mode: false,
        plan_exit_pending: false,
        pending_plan: None,
        plan_card_idx: None,
        agent_engine: None,
        agent_route_cache: None,
        mcp_client: None,
        mcp_connecting: false,
        mcp_connect_progress: std::collections::HashMap::new(),
        disabled_mcp_tools: std::collections::HashSet::new(),
        mcp_connect_gen: 0,
        engine_rebuild_pending: false,
        live_share_gen: 0,
        pending_mcp_auth: std::collections::HashMap::new(),
        agent_serve: None,
        agent_permission: None,
        agent_ask: None,
        agent_review: None,
        agent_plan_approval: None,
        agent_auto_approve: false,
        auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        agent_review_edits: false,
        review_edits_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        thinking_enabled: true,
        web_search_enabled: true,
        agent_tools_enabled: true,
        theme: UiTheme::Dark,
        model_supports_thinking: true,
        model_image_input: None,
        cursor_effort_label: None,
        reasoning_effort: None,
        model_reasoning_efforts: Vec::new(),
        queued_messages: Vec::new(),
        steering_queue: SteeringQueue::default(),
        queued_commands: Vec::new(),
        queue_focus: None,
        project_mcp_consent: ProjectMcpConsent::default(),
        pending_mcp_consent: None,
        local_command: None,
        jobs: crate::agent::jobs::JobTable::new(None),
        last_jobs_poll: std::time::Instant::now(),
        jobs_running: 0,
        local_outputs: std::collections::HashMap::new(),
        expanded_output: std::collections::HashSet::new(),
        expanded_thinking: std::collections::HashSet::new(),
        agent_turn_indices: std::collections::HashSet::new(),
        reasoning_durations: std::collections::HashMap::new(),
        turn_durations: std::collections::HashMap::new(),
        turn_notes: std::collections::HashMap::new(),
        reasoning_started_at: None,
        reasoning_elapsed_ms: None,
        installing_skill: None,
        staged_skill_install: None,
        live_share: None,
        live_share_starting: false,
        live_requested: false,
        account_gen: 0,
        account_task: None,
        account_login: None,
        pending_logout: None,
        pending_full_repaint: false,
    }
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
