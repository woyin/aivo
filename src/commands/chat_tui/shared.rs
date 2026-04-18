use super::*;

pub(super) const TEXT: Color = Color::Rgb(224, 225, 221);
pub(super) const MUTED: Color = Color::Rgb(136, 142, 139);
pub(super) const FAINT: Color = Color::Rgb(92, 99, 102);
pub(super) const ACCENT: Color = Color::Rgb(208, 180, 132);
pub(super) const ASSISTANT: Color = Color::Rgb(174, 202, 161);
pub(super) const USER: Color = Color::Rgb(166, 193, 226);
pub(super) const LINK: Color = Color::Rgb(142, 181, 219);
pub(super) const QUOTE: Color = Color::Rgb(143, 164, 146);
pub(super) const ERROR: Color = Color::Rgb(230, 134, 128);
pub(super) const EMPTY_STATE_BOTTOM_GAP: u16 = 1;
pub(super) const TRANSCRIPT_BOTTOM_PADDING: u16 = 1;
pub(super) const COMPOSER_PREFIX_WIDTH: u16 = 2;

pub(super) const COMMAND_MENU_MAX_ROWS: usize = 7;
pub(super) const PICKER_ROW_PREFIX_WIDTH: usize = 2;
pub(super) const SELECT_WARM: Color = Color::Rgb(255, 228, 194);

#[derive(Clone, Copy)]
pub(super) struct SlashCommandSpec {
    pub(super) name: &'static str,
    pub(super) help_label: &'static str,
    pub(super) description: &'static str,
    pub(super) takes_argument: bool,
}

impl SlashCommandSpec {
    pub(super) fn command_label(self) -> String {
        format!("/{}", self.name)
    }

    pub(super) fn insertion_text(self) -> String {
        let suffix = if self.takes_argument { " " } else { "" };
        format!("/{}{}", self.name, suffix)
    }
}

pub(super) const SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        name: "new",
        help_label: "/new",
        description: "start a fresh chat",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "exit",
        help_label: "/exit",
        description: "leave chat",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "resume",
        help_label: "/resume [query]",
        description: "resume a saved chat",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "model",
        help_label: "/model [name]",
        description: "switch model",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "key",
        help_label: "/key [id|name]",
        description: "switch saved key",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "attach",
        help_label: "/attach <path>",
        description: "attach a file or image",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "detach",
        help_label: "/detach <n>",
        description: "remove one queued attachment",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "clear",
        help_label: "/clear",
        description: "clear queued attachments",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "help",
        help_label: "/help",
        description: "open help",
        takes_argument: false,
    },
];

pub(crate) struct ChatTuiParams {
    pub session_store: SessionStore,
    pub cache: ModelsCache,
    pub client: Client,
    pub key: ApiKey,
    pub copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub cwd: String,
    pub raw_model: String,
    pub model: String,
    pub initial_session: String,
    pub initial_history: Vec<ChatMessage>,
    pub initial_draft_attachments: Vec<MessageAttachment>,
    pub startup_notice: Option<String>,
}

#[derive(Clone)]
pub(super) struct SessionPreview {
    pub(super) key_id: String,
    pub(super) key_name: String,
    pub(super) base_url: String,
    pub(super) session_id: String,
    pub(super) raw_model: String,
    pub(super) updated_at: String,
    pub(super) title: String,
    pub(super) preview_text: String,
}

pub(super) fn decrypt_to_chat_messages(
    state: &crate::services::session_store::ChatSessionState,
) -> Result<Vec<ChatMessage>> {
    let messages = state
        .decrypt_messages()?
        .into_iter()
        .map(|m| ChatMessage {
            role: m.role,
            content: m.content,
            reasoning_content: m.reasoning_content,
            attachments: m.attachments.unwrap_or_default(),
        })
        .collect();
    Ok(messages)
}

impl SessionPreview {
    pub(super) fn from_index_entry(
        entry: crate::services::session_store::SessionIndexEntry,
        key: &ApiKey,
    ) -> Self {
        Self {
            key_id: key.id.clone(),
            key_name: key.display_name().to_string(),
            base_url: key.base_url.clone(),
            session_id: entry.session_id,
            raw_model: entry.model,
            updated_at: entry.updated_at,
            title: entry.title,
            preview_text: entry.preview,
        }
    }

    pub(super) fn search_text(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            self.session_id,
            self.title,
            self.preview_text,
            self.key_name,
            self.raw_model,
            self.base_url
        )
    }
}

#[derive(Clone)]
pub(super) struct LoadedSession {
    pub(super) key_id: String,
    pub(super) session_id: String,
    pub(super) raw_model: String,
    pub(super) messages: Vec<ChatMessage>,
}

impl LoadedSession {
    pub(super) fn from_state(
        state: crate::services::session_store::ChatSessionState,
    ) -> Result<Self> {
        let messages = decrypt_to_chat_messages(&state)?;

        Ok(Self {
            key_id: state.key_id,
            session_id: state.session_id,
            raw_model: state.model,
            messages,
        })
    }
}

#[derive(Clone)]
pub(super) enum Overlay {
    None,
    Help,
    Picker(Box<PickerState>),
}

impl Overlay {
    pub(super) fn blocks_input(&self) -> bool {
        !matches!(self, Self::None)
    }
}

#[derive(Clone)]
pub(super) enum PickerValue {
    Model(String),
    Key(ApiKey),
    Session(SessionPreview),
}

/// A model option ready for the picker: stable id and display label.
#[derive(Clone, Debug)]
pub(super) struct ModelChoice {
    pub(super) id: String,
    pub(super) label: String,
}

#[derive(Clone)]
pub(super) struct PickerEntry {
    pub(super) label: String,
    pub(super) search_text: String,
    pub(super) value: PickerValue,
}

impl PickerEntry {
    pub(super) fn row_height(&self) -> usize {
        1
    }
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum ModelSelectionTarget {
    CurrentChat,
    KeySwitch(ApiKey),
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(super) enum PickerKind {
    Model {
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    },
    Key,
    Session,
}

#[derive(Clone)]
pub(super) struct PickerState {
    pub(super) title: &'static str,
    pub(super) query: String,
    pub(super) items: Vec<PickerEntry>,
    pub(super) loading: bool,
    pub(super) selected: usize,
    pub(super) kind: PickerKind,
    pub(super) pending_delete: Option<DeleteConfirmTarget>,
}

#[derive(Clone, Default)]
pub(super) struct PickerHitbox {
    pub(super) overlay_area: Rect,
    pub(super) list_area: Rect,
    pub(super) row_to_filtered_index: Vec<Option<usize>>,
}

#[derive(Clone)]
pub(super) struct LoadingResume {
    pub(super) request_id: u64,
    pub(super) preview: SessionPreview,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct DeleteConfirmTarget {
    pub(super) key_id: String,
    pub(super) session_id: String,
}

#[derive(Clone)]
pub(super) struct ResumeRestoreState {
    pub(super) key: ApiKey,
    pub(super) copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub(super) raw_model: String,
    pub(super) model: String,
    pub(super) format: ChatFormat,
    pub(super) history: Vec<ChatMessage>,
    pub(super) draft: String,
    pub(super) draft_attachments: Vec<MessageAttachment>,
    pub(super) cursor: usize,
    pub(super) command_menu: CommandMenuState,
    pub(super) draft_history_index: Option<usize>,
    pub(super) draft_history_stash: Option<String>,
    pub(super) session_id: String,
    pub(super) notice: Option<(Color, String)>,
    pub(super) show_reasoning: bool,
    pub(super) pending_response: String,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) last_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    pub(super) follow_output: bool,
    pub(super) transcript_scroll: usize,
}

#[derive(Clone)]
pub(super) struct PendingSubmission {
    pub(super) content: String,
    pub(super) attachments: Vec<MessageAttachment>,
}

#[derive(Clone, Default)]
pub(super) struct CommandMenuState {
    pub(super) query: String,
    pub(super) selected: usize,
    pub(super) dismissed: bool,
    pub(super) placement: Option<CommandMenuPlacement>,
}

impl CommandMenuState {
    pub(super) fn reset(&mut self) {
        self.query.clear();
        self.selected = 0;
        self.dismissed = false;
        self.placement = None;
    }
}

#[derive(Clone)]
pub(super) struct PathMenuEntry {
    pub(super) label: String,
    pub(super) is_dir: bool,
    pub(super) description: String,
    pub(super) insertion_text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MenuKind {
    Commands,
    AttachPath,
}

#[derive(Clone)]
pub(super) enum ComposerMenuEntry {
    Command(&'static SlashCommandSpec),
    Path(PathMenuEntry),
}

impl ComposerMenuEntry {
    pub(super) fn label(&self) -> String {
        match self {
            Self::Command(command) => command.command_label(),
            Self::Path(path) => path.label.clone(),
        }
    }

    pub(super) fn description(&self) -> &str {
        match self {
            Self::Command(command) => command.description,
            Self::Path(path) => &path.description,
        }
    }
}

pub(super) struct VisibleCommandMenu {
    pub(super) kind: MenuKind,
    pub(super) entries: Vec<ComposerMenuEntry>,
    pub(super) selected: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CommandMenuPlacement {
    Above,
    Below,
}

impl ResumeRestoreState {
    pub(super) fn capture(app: &ChatTuiApp) -> Self {
        Self {
            key: app.key.clone(),
            copilot_tm: app.copilot_tm.clone(),
            raw_model: app.raw_model.clone(),
            model: app.model.clone(),
            format: app.format.clone(),
            history: app.history.clone(),
            draft: app.draft.clone(),
            draft_attachments: app.draft_attachments.clone(),
            cursor: app.cursor,
            command_menu: app.command_menu.clone(),
            draft_history_index: app.draft_history_index,
            draft_history_stash: app.draft_history_stash.clone(),
            session_id: app.session_id.clone(),
            notice: app.notice.clone(),
            show_reasoning: app.show_reasoning,
            pending_response: app.pending_response.clone(),
            pending_reasoning: app.pending_reasoning.clone(),
            pending_submit: app.pending_submit.clone(),
            last_usage: app.last_usage,
            context_tokens: app.context_tokens,
            follow_output: app.follow_output,
            transcript_scroll: app.transcript_scroll,
        }
    }
}

impl PickerState {
    pub(super) fn loading(title: &'static str, query: String, kind: PickerKind) -> Self {
        Self {
            title,
            query,
            items: Vec::new(),
            loading: true,
            selected: 0,
            kind,
            pending_delete: None,
        }
    }

    pub(super) fn ready(
        title: &'static str,
        query: String,
        items: Vec<PickerEntry>,
        kind: PickerKind,
    ) -> Self {
        Self {
            title,
            query,
            items,
            loading: false,
            selected: 0,
            kind,
            pending_delete: None,
        }
    }

    pub(super) fn filtered_items(&self) -> Vec<(usize, &PickerEntry)> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, item)| matches_fuzzy(&self.query, &item.search_text))
            .collect()
    }

    pub(super) fn exact_match_index(&self) -> Option<usize> {
        let PickerKind::Model {
            auto_accept_exact, ..
        } = &self.kind
        else {
            return None;
        };
        if !*auto_accept_exact || self.query.is_empty() {
            return None;
        }
        self.filtered_items().iter().position(
            |(_, item)| matches!(&item.value, PickerValue::Model(model) if model == &self.query),
        )
    }

    pub(super) fn visible_items(&self, max_rows: usize) -> Vec<(usize, &PickerEntry)> {
        let filtered = self.filtered_items();
        if filtered.is_empty() || max_rows == 0 {
            return Vec::new();
        }

        let selected = self.selected.min(filtered.len().saturating_sub(1));
        let mut start = selected;
        let mut used_rows = filtered[selected].1.row_height();

        while start > 0 {
            let next_height = filtered[start - 1].1.row_height();
            if used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            start -= 1;
        }

        let mut end = selected + 1;
        while end < filtered.len() {
            let next_height = filtered[end].1.row_height();
            if used_rows + next_height > max_rows {
                break;
            }
            used_rows += next_height;
            end += 1;
        }

        filtered[start..end]
            .iter()
            .enumerate()
            .map(|(offset, (_, item))| (start + offset, *item))
            .collect()
    }

    pub(super) fn select_prev(&mut self) {
        let len = self.filtered_items().len();
        if len == 0 {
            self.selected = 0;
        } else if self.selected == 0 {
            self.selected = len - 1;
        } else {
            self.selected -= 1;
        }
    }

    pub(super) fn select_next(&mut self) {
        let len = self.filtered_items().len();
        if len > 0 {
            self.selected = if self.selected + 1 >= len {
                0
            } else {
                self.selected + 1
            };
        }
    }

    pub(super) fn clear_pending_delete(&mut self) {
        self.pending_delete = None;
    }

    pub(super) fn selected_delete_target(&self) -> Option<DeleteConfirmTarget> {
        let (_, item) = self.filtered_items().get(self.selected).copied()?;
        match &item.value {
            PickerValue::Session(session) => Some(DeleteConfirmTarget {
                key_id: session.key_id.clone(),
                session_id: session.session_id.clone(),
            }),
            _ => None,
        }
    }

    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        let Some(target) = self.selected_delete_target() else {
            return false;
        };
        if self.pending_delete.as_ref() == Some(&target) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(target);
            false
        }
    }

    pub(super) fn delete_is_armed_for_selected(&self) -> bool {
        self.selected_delete_target()
            .is_some_and(|target| self.pending_delete.as_ref() == Some(&target))
    }

    pub(super) fn delete_is_armed_for_session(&self, preview: &SessionPreview) -> bool {
        self.pending_delete.as_ref().is_some_and(|target| {
            target.key_id == preview.key_id && target.session_id == preview.session_id
        })
    }
}

pub(super) enum SubmitAction {
    Send(String),
    Command(SlashCommand),
}

#[allow(dead_code)]
pub(super) enum ClipboardPayload {
    Text(String),
    Attachment(MessageAttachment),
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SlashCommand {
    New,
    Exit,
    Resume(Option<String>),
    Model(Option<String>),
    Key(Option<String>),
    Attach(String),
    Detach(usize),
    Clear,
    Help,
}

pub(super) enum RuntimeEvent {
    Delta(ChatResponseChunk),
    Finished {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    ModelsLoaded(std::result::Result<Vec<ModelChoice>, String>),
    ResumeLoaded {
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    },
}

pub(super) struct ChatTuiApp {
    pub(super) session_store: SessionStore,
    pub(super) cache: ModelsCache,
    pub(super) client: Client,
    pub(super) key: ApiKey,
    pub(super) copilot_tm: Option<Arc<CopilotTokenManager>>,
    pub(super) cwd: String,
    pub(super) raw_model: String,
    pub(super) model: String,
    pub(super) format: ChatFormat,
    pub(super) history: Vec<ChatMessage>,
    pub(super) draft: String,
    pub(super) draft_attachments: Vec<MessageAttachment>,
    pub(super) cursor: usize,
    pub(super) command_menu: CommandMenuState,
    pub(super) draft_history: Vec<String>,
    pub(super) draft_history_index: Option<usize>,
    pub(super) draft_history_stash: Option<String>,
    pub(super) session_id: String,
    pub(super) overlay: Overlay,
    pub(super) notice: Option<(Color, String)>,
    pub(super) show_reasoning: bool,
    pub(super) pending_response: String,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) sending: bool,
    pub(super) request_started_at: Option<Instant>,
    pub(super) last_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    pub(super) follow_output: bool,
    pub(super) transcript_scroll: usize,
    pub(super) transcript_width: u16,
    pub(super) transcript_view_height: u16,
    pub(super) tx: UnboundedSender<RuntimeEvent>,
    pub(super) rx: UnboundedReceiver<RuntimeEvent>,
    pub(super) response_task: Option<JoinHandle<()>>,
    pub(super) resume_task: Option<JoinHandle<()>>,
    pub(super) resume_request_id: u64,
    pub(super) loading_resume: Option<LoadingResume>,
    pub(super) resume_restore_state: Option<ResumeRestoreState>,
    pub(super) reduce_motion: bool,
    pub(super) frame_tick: usize,
    pub(super) picker_hitbox: Option<PickerHitbox>,
    pub(super) pending_clear_screen: bool,
}
