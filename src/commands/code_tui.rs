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
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use unicode_width::UnicodeWidthChar;

use crate::style::spinner_frame;
use crate::tui::matches_fuzzy;

use super::code_tui_format::{
    build_footer_text, display_width, estimate_context_tokens, footer_host_label,
    footer_session_label, format_picker_match_count, format_request_elapsed,
    format_session_group_label, format_session_match_count, format_session_time,
    format_time_ago_short, format_token_count, format_token_count_value, format_usd,
    git_branch_for, truncate_for_display_width, truncate_for_width,
};
use super::*;

#[path = "code_tui/menu.rs"]
mod menu;
#[path = "code_tui/overlay_render_impl.rs"]
mod overlay_render_impl;
#[path = "code_tui/render.rs"]
mod render;
#[path = "code_tui/render_impl.rs"]
mod render_impl;
#[path = "code_tui/storage.rs"]
mod storage;
#[path = "code_tui/system.rs"]
mod system;

#[path = "code_tui/shared.rs"]
mod shared;

#[path = "code_tui/account_impl.rs"]
mod account_impl;
#[path = "code_tui/app_state_impl.rs"]
mod app_state_impl;
#[path = "code_tui/event_loop_impl.rs"]
mod event_loop_impl;
#[path = "code_tui/input_impl.rs"]
mod input_impl;
#[path = "code_tui/key_handler_impl.rs"]
mod key_handler_impl;
#[path = "code_tui/live_impl.rs"]
mod live_impl;
#[path = "code_tui/queue_impl.rs"]
mod queue_impl;
#[path = "code_tui/runtime_impl.rs"]
mod runtime_impl;
#[path = "code_tui/session_impl.rs"]
mod session_impl;

use self::menu::*;
use self::render::*;
pub(crate) use self::runtime_impl::skill_invocation_label;
pub(crate) use self::shared::CodeTuiParams;
use self::shared::*;
use self::storage::*;
pub(crate) use self::storage::{session_preview_text_from_messages, session_title_from_messages};
use self::system::*;

impl CodeTuiApp {
    async fn new(params: CodeTuiParams) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        // No "Ready" filler — the welcome chip + tip cover the empty state.
        // The `-c` summary rides in as the startup notice (the pre-TUI stderr
        // line is wiped by the alt-screen); combine with any attachment notice.
        let startup_message = match (
            params.injected_context_summary.clone(),
            params.startup_notice,
        ) {
            (Some(ctx), Some(attach)) => Some(format!("{ctx} · {attach}")),
            (Some(ctx), None) => Some(ctx),
            (None, Some(attach)) => Some(attach),
            (None, None) => None,
        };
        // Platforms without write confinement (Windows) say so up front.
        let startup_message = match (startup_message, crate::agent::sandbox::confinement_notice()) {
            (Some(m), Some(warn)) => Some(format!("{m} · {warn}")),
            (None, Some(warn)) => Some(warn.to_string()),
            (m, None) => m,
        };
        let startup_notice = startup_message.map(|message| (MUTED(), message));

        let initial_format = seeded_chat_format(&params.key, &params.raw_model);
        // Remembered across sessions (the user picked "remember last choice");
        // both toggles come from one read of code-prefs.json. auto_approve
        // defaults off (safe); thinking_enabled defaults on (high-signal).
        let crate::services::session_store::ChatToggles {
            auto_approve,
            review_edits,
            thinking_enabled,
            web_search_enabled,
            agent_tools_enabled,
            theme: chat_theme,
        } = params.session_store.get_chat_toggles().await;
        // First launch (no stored choice) auto-detects the terminal background;
        // once the user picks in /config it's persisted and always honored, so the
        // probe never runs again. Detection is bounded and off-thread — see
        // `detect_terminal_theme`.
        let detected = if chat_theme.is_none() {
            tokio::task::spawn_blocking(detect_terminal_theme)
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        let theme = resolve_startup_theme(chat_theme, detected);
        set_ui_theme(theme);
        // Move any pre-existing `/skills` + `/mcp` opt-outs out of config.json (where
        // a routine key/route/selection write — or an older aivo binary — can drop
        // them) into code-prefs.json, before the chat flow writes config.json.
        params.session_store.migrate_disabled_toggles().await;
        // The launch dir keys the recall view; the persisted file stays global.
        let real_cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        let draft_history_all = load_persisted_draft_history();
        let draft_history = draft_history_view(&draft_history_all, &real_cwd);
        // Enabled MCP servers for the welcome chip (skills counted live elsewhere).
        let mcp_cwd = if real_cwd.is_empty() { "." } else { &real_cwd };
        let disabled_mcp: std::collections::HashSet<String> = params
            .session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mcp_configured_count = crate::agent::mcp::configured_servers(Path::new(mcp_cwd))
            .into_iter()
            .filter(|server| !disabled_mcp.contains(&server.name))
            .count();
        // Seed the rotating tip from the wall clock so it varies between launches.
        let welcome_tip_index = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs() as usize % WELCOME_TIPS.len())
            .unwrap_or(0);
        // Job logs under the session's artifacts dir; re-rooted on `/new`/resume.
        let jobs = crate::agent::jobs::JobTable::new(Some(
            params
                .session_store
                .session_artifacts_dir(&params.initial_session)
                .join("jobs"),
        ));
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
            turn_model: None,
            format: initial_format,
            history: params.initial_history,
            draft: String::new(),
            draft_attachments: params.initial_draft_attachments,
            cursor: 0,
            command_menu: CommandMenuState::default(),
            skill_commands: Vec::new(),
            last_subagents: Vec::new(),
            mcp_configured_count,
            welcome_tip_index,
            welcome_tip_rotated_at: None,
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
            context_window_override: params.max_context,
            injected_context: params.injected_context,
            injected_context_summary: params.injected_context_summary,
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
            scroll_speed: chat_scroll_speed(),
            swipe_scroll: chat_swipe_scroll_enabled(),
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
            reduce_motion: reduce_motion_requested(),
            frame_tick: 0,
            picker_hitbox: None,
            overlay_detail_area: None,
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
            pending_mcp_auth: std::collections::HashMap::new(),
            agent_serve: None,
            agent_permission: None,
            agent_ask: None,
            agent_review: None,
            agent_plan_approval: None,
            // Modes are exclusive; stale prefs with both on → review wins.
            // `--auto-approve` pre-sets the toggle (session-only, outranks review).
            agent_auto_approve: params.auto_approve || (auto_approve && !review_edits),
            auto_approve_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                params.auto_approve || (auto_approve && !review_edits),
            )),
            agent_review_edits: review_edits && !params.auto_approve,
            review_edits_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
                review_edits && !params.auto_approve,
            )),
            thinking_enabled,
            web_search_enabled,
            agent_tools_enabled,
            theme,
            // Set by `refresh_context_window` (called right after construction and
            // on every model switch); false until the first resolve.
            model_supports_thinking: false,
            model_image_input: None,
            cursor_effort_label: None,
            // Loaded per-model by `refresh_context_window` (called right after).
            reasoning_effort: None,
            model_reasoning_efforts: Vec::new(),
            queued_messages: Vec::new(),
            steering_queue: SteeringQueue::default(),
            queued_commands: Vec::new(),
            queue_focus: None,
            project_mcp_consent: ProjectMcpConsent::default(),
            pending_mcp_consent: None,
            local_command: None,
            jobs,
            jobs_running: 0,
            last_jobs_poll: std::time::Instant::now(),
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
            live_share_gen: 0,
            live_requested: false,
            account_gen: 0,
            account_task: None,
            account_login: None,
            pending_logout: None,
        })
    }
}

pub(super) async fn run_chat_tui(params: CodeTuiParams) -> Result<()> {
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
    let initial_prompt = params.initial_prompt.clone();
    let share = params.share;
    let mut app = CodeTuiApp::new(params).await?;
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
            app.notice = Some((ERROR(), format!("Resume failed: {err:#}")));
        }
    }
    // Positional `aivo code "<text>"`: first turn starts now, streams in once
    // the event loop renders.
    if let Some(prompt) = initial_prompt
        && let Err(err) = app.send_user_message(prompt).await
    {
        app.notice = Some((ERROR(), format!("Failed to send: {err:#}")));
    }
    // The event loop starts the share once the session settles (an async
    // `--resume` could still be loading a different session id here).
    app.live_requested = share;
    let result = app.run().await;
    // The public link dies with the chat.
    app.stop_live_share();
    app.persist_draft_history();
    // Remember the auto-approve toggle for next time (best-effort).
    app.session_store
        .set_chat_auto_approve(app.agent_auto_approve)
        .await
        .ok();
    app.session_store
        .set_chat_review_edits(app.agent_review_edits)
        .await
        .ok();
    // After a clean exit, point the user back to this exact conversation by id
    // (the terminal is already restored inside `run`, so this lands in normal
    // scrollback). Skipped for an untouched chat — nothing was saved.
    if result.is_ok()
        && let Some(id) = app.resumable_session_id()
    {
        println!(
            "{}  {}",
            crate::style::dim("Resume:"),
            crate::style::cyan(format!("aivo code --resume {id}")),
        );
    }
    result
}

/// First-launch theme resolution: an explicit stored choice always wins; with
/// none, use the detected terminal background, else dark.
fn resolve_startup_theme(
    stored: Option<crate::services::session_store::ChatTheme>,
    detected: Option<UiTheme>,
) -> UiTheme {
    use crate::services::session_store::ChatTheme;
    match stored {
        Some(ChatTheme::Light) => UiTheme::Light,
        Some(ChatTheme::Dark) => UiTheme::Dark,
        None => detected.unwrap_or(UiTheme::Dark),
    }
}

/// Best-effort terminal-background probe (OSC 11) run once on first launch, before
/// the user has chosen a theme. Bounded and self-restoring: colorsaurus queries
/// `/dev/tty` directly, feature-detects terminals that don't support the query,
/// and times out otherwise — every failure path yields `None` so the caller
/// defaults to dark. It never blocks or hangs startup. Called before we enter raw
/// mode / the alternate screen, and off the reactor via `spawn_blocking`.
fn detect_terminal_theme() -> Option<UiTheme> {
    use terminal_colorsaurus::{QueryOptions, ThemeMode, theme_mode};
    // DA1 feature-detection means real terminals answer within a round-trip, so a
    // short ceiling covers local + typical SSH latency while bounding the rare
    // no-response case; a miss just falls back to dark (recoverable via /config).
    let mut opts = QueryOptions::default();
    opts.timeout = std::time::Duration::from_millis(200);
    match theme_mode(opts) {
        Ok(ThemeMode::Light) => Some(UiTheme::Light),
        Ok(ThemeMode::Dark) => Some(UiTheme::Dark),
        Err(_) => None,
    }
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

/// Read an `AIVO_CODE_<suffix>` var, falling back to the pre-rename
/// `AIVO_CHAT_<suffix>` so existing users' shell configs keep working.
fn code_env(suffix: &str) -> Option<String> {
    env::var(format!("AIVO_CODE_{suffix}"))
        .or_else(|_| env::var(format!("AIVO_CHAT_{suffix}")))
        .ok()
}

fn chat_scroll_speed() -> usize {
    code_env("SCROLL_SPEED")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_CHAT_SCROLL_SPEED)
        .clamp(1, MAX_CHAT_SCROLL_SPEED)
}

fn chat_mouse_enabled() -> bool {
    chat_mouse_enabled_for(
        code_env("DISABLE_MOUSE").as_deref(),
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

fn chat_swipe_scroll_enabled() -> bool {
    chat_swipe_scroll_enabled_for(
        code_env("SWIPE_SCROLL").as_deref(),
        crate::services::termux_exec::is_termux(),
    )
}

/// Pure swipe-scroll policy (see the `swipe_scroll` field), split out for testing.
/// On under Termux; `AIVO_CHAT_SWIPE_SCROLL` forces it on/off.
fn chat_swipe_scroll_enabled_for(override_val: Option<&str>, is_termux: bool) -> bool {
    if let Some(value) = override_val {
        return matches!(value, "1" | "true" | "TRUE" | "yes" | "YES");
    }
    is_termux
}

#[cfg(test)]
#[path = "code_tui/tests.rs"]
mod tests;
