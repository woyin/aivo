use std::env;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use pulldown_cmark::{
    CodeBlockKind, Event as MdEvent, HeadingLevel, Options as MdOptions, Parser, Tag, TagEnd,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Wrap,
};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use unicode_width::UnicodeWidthChar;

use crate::style::spinner_frame;
use crate::tui::matches_fuzzy;

use super::chat_tui_format::{
    build_footer_text, display_width, estimate_context_tokens, footer_host_label,
    format_picker_match_count, format_request_elapsed, format_session_group_label,
    format_session_match_count, format_session_time, format_time_ago_short, format_token_count,
    format_token_count_value, git_branch_for, truncate_for_display_width, truncate_for_width,
};
use super::*;

#[path = "chat_tui/menu.rs"]
mod menu;
#[path = "chat_tui/overlay_render_impl.rs"]
mod overlay_render_impl;
#[path = "chat_tui/render.rs"]
mod render;
#[path = "chat_tui/render_impl.rs"]
mod render_impl;
#[path = "chat_tui/storage.rs"]
mod storage;
#[path = "chat_tui/system.rs"]
mod system;

#[path = "chat_tui/shared.rs"]
mod shared;

#[path = "chat_tui/app_state_impl.rs"]
mod app_state_impl;
#[path = "chat_tui/event_loop_impl.rs"]
mod event_loop_impl;
#[path = "chat_tui/input_impl.rs"]
mod input_impl;
#[path = "chat_tui/key_handler_impl.rs"]
mod key_handler_impl;
#[path = "chat_tui/runtime_impl.rs"]
mod runtime_impl;
#[path = "chat_tui/session_impl.rs"]
mod session_impl;

use self::menu::*;
use self::render::*;
pub(crate) use self::runtime_impl::skill_invocation_label;
pub(crate) use self::shared::ChatTuiParams;
use self::shared::*;
use self::storage::*;
pub(crate) use self::storage::{session_preview_text_from_messages, session_title_from_messages};
use self::system::*;

impl ChatTuiApp {
    async fn new(params: ChatTuiParams) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let startup_notice = params
            .startup_notice
            .map(|message| (MUTED, message))
            .or(Some((MUTED, "Ready".to_string())));

        let initial_format = seeded_chat_format(&params.key, &params.raw_model);
        // Remembered across sessions (the user picked "remember last choice");
        // both toggles come from one read of chat-prefs.json. auto_approve
        // defaults off (safe); thinking_enabled defaults on (high-signal).
        let crate::services::session_store::ChatToggles {
            auto_approve,
            thinking_enabled,
        } = params.session_store.get_chat_toggles().await;
        // The launch dir keys the recall view; the persisted file stays global.
        let real_cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let draft_history_all = load_persisted_draft_history();
        let draft_history = draft_history_view(&draft_history_all, &real_cwd);
        Ok(Self {
            session_store: params.session_store,
            cache: params.cache,
            client: params.client,
            key: params.key,
            copilot_tm: params.copilot_tm,
            cwd: params.cwd,
            real_cwd,
            git_branch: None,
            git_branch_checked_at: None,
            raw_model: params.raw_model,
            model: params.model,
            billed_model: None,
            format: initial_format,
            history: params.initial_history,
            draft: String::new(),
            draft_attachments: params.initial_draft_attachments,
            cursor: 0,
            command_menu: CommandMenuState::default(),
            skill_commands: Vec::new(),
            draft_history,
            draft_history_all,
            draft_history_index: None,
            draft_history_stash: None,
            session_id: params.initial_session,
            overlay: Overlay::None,
            notice: startup_notice,
            pending_response: String::new(),
            incoming_buffer: String::new(),
            pending_finish: None,
            pending_reasoning: String::new(),
            pending_submit: None,
            sending: false,
            request_started_at: None,
            last_usage: None,
            live_usage: None,
            context_tokens: 0,
            session_tokens: crate::services::session_store::SessionTokens::default(),
            context_window: 0,
            context_window_override: params.max_context,
            context_is_estimate: true,
            follow_output: true,
            transcript_revision: 0,
            transcript_scroll: 0,
            transcript_width: 0,
            transcript_view_height: 0,
            transcript_hitbox: None,
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
            scroll_speed: chat_scroll_speed(),
            toast: None,
            tx,
            rx,
            response_task: None,
            resume_task: None,
            resume_request_id: 0,
            loading_resume: None,
            resume_restore_state: None,
            reduce_motion: reduce_motion_requested(),
            frame_tick: 0,
            picker_hitbox: None,
            exit_confirm_pending: false,
            cursor_acp_session: None,
            active_agent: params.initial_agent,
            pending_agent_messages: None,
            goal_mode: None,
            agent_engine: None,
            agent_route_cache: None,
            mcp_client: None,
            mcp_connecting: false,
            mcp_connect_progress: std::collections::HashMap::new(),
            mcp_connect_gen: 0,
            mcp_rebuild_pending: false,
            pending_mcp_auth: std::collections::HashMap::new(),
            agent_serve: None,
            agent_permission: None,
            agent_auto_approve: auto_approve,
            auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                auto_approve,
            )),
            thinking_enabled,
            // Set by `refresh_context_window` (called right after construction and
            // on every model switch); false until the first resolve.
            model_supports_thinking: false,
            model_image_input: None,
            // Loaded per-model by `refresh_context_window` (called right after).
            reasoning_effort: None,
            model_reasoning_efforts: Vec::new(),
            queued_messages: Vec::new(),
            project_mcp_consent: ProjectMcpConsent::default(),
            pending_mcp_consent: None,
            local_command: None,
            last_local_output: None,
            expanded_thinking: std::collections::HashSet::new(),
            reasoning_durations: std::collections::HashMap::new(),
            reasoning_started_at: None,
            reasoning_elapsed_ms: None,
        })
    }
}

pub(super) async fn run_chat_tui(params: ChatTuiParams) -> Result<()> {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(
            stdout,
            LeaveAlternateScreen,
            DisableBracketedPaste,
            DisableMouseCapture
        );
        original_hook(info);
    }));
    let initial_resume = params.initial_resume.clone();
    let mut app = ChatTuiApp::new(params).await?;
    app.refresh_context_window().await;
    // Surface discovered skills as `/`-typeable slash commands (e.g. `/repo-study`)
    // before the first keystroke, so the command menu suggests them right away.
    app.refresh_skill_commands().await;
    // Warm the catalog in the background when the window is unknown or the
    // cache is stale, so server-side edits (e.g. reasoning-effort levels)
    // refresh on the next launch. Best-effort.
    let catalog_stale =
        !crate::commands::models::full_catalog_metadata_fresh(&app.key, &app.cache).await;
    if app.context_window == 0 || catalog_stale {
        let cache = app.cache.clone();
        let key = app.key.clone();
        let client = app.client.clone();
        let tx = app.tx.clone();
        tokio::spawn(async move {
            crate::commands::models::warm_full_catalog_metadata(&client, &key, &cache).await;
            // Re-resolve the full limits (window + efforts), not just the window.
            let _ = tx.send(RuntimeEvent::CatalogWarmed);
        });
    }
    // `--resume`: open the session picker (empty arg) or jump straight to a
    // session by id. Mirrors the in-chat `/resume [query]`; failure is
    // non-fatal — surface it as a notice and fall back to a fresh chat.
    if let Some(query) = initial_resume {
        let query = (!query.is_empty()).then_some(query);
        if let Err(err) = app.open_resume_picker(query).await {
            app.notice = Some((ERROR, format!("Resume failed: {err:#}")));
        }
    }
    let result = app.run().await;
    app.persist_draft_history();
    // Remember the auto-approve toggle for next time (best-effort).
    app.session_store
        .set_chat_auto_approve(app.agent_auto_approve)
        .await
        .ok();
    // After a clean exit, point the user back to this exact conversation by id
    // (the terminal is already restored inside `run`, so this lands in normal
    // scrollback). Skipped for an untouched chat — nothing was saved.
    if result.is_ok()
        && let Some(id) = app.resumable_session_id()
    {
        println!(
            "\n{}  {}",
            crate::style::dim("Resume this chat:"),
            crate::style::cyan(format!("aivo chat --resume {id}")),
        );
    }
    result
}

fn setup_terminal(mouse_enabled: bool) -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let result: Result<_> = (|| {
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
        if mouse_enabled {
            execute!(stdout, EnableMouseCapture)?;
        }
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(terminal)
    })();
    if result.is_err() {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, DisableMouseCapture);
    }
    result
}

fn restore_terminal(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture,
    )?;
    terminal.show_cursor()?;
    Ok(())
}

fn chat_scroll_speed() -> usize {
    env::var("AIVO_CHAT_SCROLL_SPEED")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CHAT_SCROLL_SPEED)
        .clamp(1, MAX_CHAT_SCROLL_SPEED)
}

fn chat_mouse_enabled() -> bool {
    chat_mouse_enabled_for(
        env::var("AIVO_CHAT_DISABLE_MOUSE").ok().as_deref(),
        crate::services::termux_exec::is_termux(),
    )
}

/// Pure mouse-capture policy, split out for testing. Off by default under
/// Termux, where capturing the mouse makes screen taps stop toggling the soft
/// keyboard; an explicit `AIVO_CHAT_DISABLE_MOUSE` override wins either way.
fn chat_mouse_enabled_for(disable_override: Option<&str>, is_termux: bool) -> bool {
    if let Some(value) = disable_override {
        return !matches!(value, "1" | "true" | "TRUE" | "yes" | "YES");
    }
    !is_termux
}

#[cfg(test)]
#[path = "chat_tui/tests.rs"]
mod tests;
