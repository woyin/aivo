use std::env;
use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
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
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
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
    truncate_for_display_width, truncate_for_width, wrapped_text_line_count,
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
pub(crate) use self::shared::ChatTuiParams;
use self::shared::*;
use self::storage::*;
use self::system::*;

impl ChatTuiApp {
    async fn new(params: ChatTuiParams) -> Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let startup_notice = params
            .startup_notice
            .map(|message| (MUTED, message))
            .or(Some((MUTED, "Ready".to_string())));

        let initial_format = detect_initial_chat_format(&params.key.base_url);
        Ok(Self {
            session_store: params.session_store,
            cache: params.cache,
            client: params.client,
            key: params.key,
            copilot_tm: params.copilot_tm,
            cwd: params.cwd,
            raw_model: params.raw_model,
            model: params.model,
            format: initial_format,
            history: params.initial_history,
            draft: String::new(),
            draft_attachments: params.initial_draft_attachments,
            cursor: 0,
            command_menu: CommandMenuState::default(),
            draft_history: load_persisted_draft_history(),
            draft_history_index: None,
            draft_history_stash: None,
            session_id: params.initial_session,
            overlay: Overlay::None,
            notice: startup_notice,
            show_reasoning: true,
            pending_response: String::new(),
            pending_reasoning: String::new(),
            pending_submit: None,
            sending: false,
            request_started_at: None,
            last_usage: None,
            context_tokens: 0,
            follow_output: true,
            transcript_scroll: 0,
            transcript_width: 0,
            transcript_view_height: 0,
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
            pending_clear_screen: false,
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
    let mut app = ChatTuiApp::new(params).await?;
    let result = app.run().await;
    app.persist_draft_history();
    result
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let result: Result<_> = (|| {
        let mut stdout = io::stdout();
        execute!(
            stdout,
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableMouseCapture
        )?;
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

#[cfg(test)]
#[path = "chat_tui/tests.rs"]
mod tests;
