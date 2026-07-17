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
        let theme = resolve_startup_theme(chat_theme);
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
        // Everything below overrides a `bare()` default; fields not named here
        // keep the neutral value from the one exhaustive literal in shared.rs.
        let mut app = Self::bare(
            tx,
            rx,
            params.session_store,
            params.cache,
            params.client,
            params.key,
        );
        app.copilot_tm = params.copilot_tm;
        app.cwd = params.cwd;
        app.real_cwd = real_cwd;
        app.raw_model = params.raw_model;
        app.model = params.model;
        app.format = initial_format;
        app.history = params.initial_history;
        app.draft_attachments = params.initial_draft_attachments;
        app.mcp_configured_count = mcp_configured_count;
        app.welcome_tip_index = welcome_tip_index;
        app.draft_history = draft_history;
        app.draft_history_all = draft_history_all;
        app.session_id = params.initial_session;
        app.notice = startup_notice;
        app.context_window_override = params.max_context;
        app.injected_context = params.injected_context;
        app.injected_context_summary = params.injected_context_summary;
        app.scroll_speed = chat_scroll_speed();
        app.swipe_scroll = chat_swipe_scroll_enabled();
        app.reduce_motion = reduce_motion_requested();
        // Modes are exclusive; stale prefs with both on → review wins.
        // `--auto-approve` pre-sets the toggle (session-only, outranks review).
        let auto = params.auto_approve || (auto_approve && !review_edits);
        let review = review_edits && !params.auto_approve;
        app.agent_auto_approve = auto;
        app.auto_approve_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(auto));
        app.agent_review_edits = review;
        app.review_edits_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(review));
        app.thinking_enabled = thinking_enabled;
        app.web_search_enabled = web_search_enabled;
        app.agent_tools_enabled = agent_tools_enabled;
        app.theme = theme;
        app.jobs = jobs;
        Ok(app)
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

/// Stored `/config` choice, else dark. Deliberately no terminal-background
/// auto-detection: an OSC 10/11 probe leaks late replies as typed input on
/// slow (SSH) links.
fn resolve_startup_theme(stored: Option<crate::services::session_store::ChatTheme>) -> UiTheme {
    use crate::services::session_store::ChatTheme;
    match stored {
        Some(ChatTheme::Light) => UiTheme::Light,
        Some(ChatTheme::Dark) | None => UiTheme::Dark,
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
#[path = "code_tui/tests/mod.rs"]
mod tests;
