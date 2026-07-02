use super::*;

// Warm-night palette mirroring getaivo.dev's dark theme (aivo-website
// global.css): warm off-white ink on warm near-black, brand yellow accent, role
// hues borrowed from the site's syntax tokens so the CLI and the web read as one
// product. Everything stays muted — accents mark, they don't shout.
pub(super) const TEXT: Color = Color::Rgb(237, 233, 226); // --text-primary
pub(super) const MUTED: Color = Color::Rgb(158, 152, 140); // warm secondary ink
pub(super) const FAINT: Color = Color::Rgb(109, 103, 93); // --text-muted
// getaivo.dev brand yellow (#DEFC09), toned down for the muted dark palette so
// it reads as the brand accent without the full neon punch of the raw web color.
pub(super) const ACCENT: Color = Color::Rgb(206, 226, 70);
// Assistant = warm jade (site --code-op), pulled clearly off the lime accent.
pub(super) const ASSISTANT: Color = Color::Rgb(140, 190, 176);
// User = lavender (site --code-var), the brand's complement to the yellow accent.
pub(super) const USER: Color = Color::Rgb(181, 164, 235);
// Agent tool steps (call/result) — a steel cyan distinct from user lavender,
// assistant jade, and link blue.
pub(super) const TOOL: Color = Color::Rgb(110, 170, 188);
// `!cmd` local shell runs — a muted magenta so a user-run shell command reads
// apart from agent tool steps (cyan) and the brand accent (lime).
pub(super) const SHELL: Color = Color::Rgb(204, 112, 176);
pub(super) const LINK: Color = Color::Rgb(143, 178, 222);
pub(super) const QUOTE: Color = Color::Rgb(150, 150, 128); // warm olive aside
pub(super) const ERROR: Color = Color::Rgb(228, 128, 114); // warm coral
pub(super) const WARNING: Color = Color::Rgb(224, 180, 104); // brand gold (--code-string)
pub(super) const LIVE: Color = Color::Rgb(232, 96, 92); // share "recording" red
/// Share URL notice prefix; `notice_spans` matches it to color the line.
pub(super) const LIVE_NOTICE_PREFIX: &str = "● Sharing: ";
/// Footer badge shown while sharing.
pub(super) const LIVE_BADGE: &str = "● sharing";
/// Footer badge shown when `/config` "Agent tools" is off (plain-chat mode).
pub(super) const PLAIN_CHAT_BADGE: &str = "plain chat";
// Inline-diff palette for the compact edit preview under a tool call. The
// changed line gets a subtle dark tint (not a saturated terminal-diff fill) that
// fills the full row width (see `fill_trailing_background`) so a wrapped line
// still reads as one contiguous block; the gutter `+`/`-` is brighter than the
// code text so the eye lands on the sign. Greens echo ASSISTANT jade, reds ERROR
// coral, so the diff sits inside the warm palette.
pub(super) const DIFF_ADD_BG: Color = Color::Rgb(26, 42, 32);
pub(super) const DIFF_DEL_BG: Color = Color::Rgb(48, 30, 30);
pub(super) const DIFF_ADD_FG: Color = Color::Rgb(168, 204, 182);
pub(super) const DIFF_DEL_FG: Color = Color::Rgb(224, 162, 154);
pub(super) const DIFF_ADD_SIGN: Color = Color::Rgb(120, 190, 150);
pub(super) const DIFF_DEL_SIGN: Color = Color::Rgb(230, 120, 112);
// Emphasis tints for the changed tokens in a word-diff line (see `word_segments`).
pub(super) const DIFF_ADD_HL_BG: Color = Color::Rgb(33, 84, 50);
pub(super) const DIFF_DEL_HL_BG: Color = Color::Rgb(92, 38, 38);
pub(super) const EMPTY_STATE_TOP_GAP: u16 = 1;
// No bottom padding: the composer already reserves its own blank spacing row
// above the divider, so the welcome screen's "Ready" line keeps the same single
// blank gap above the prompt as a live conversation does (not a doubled gap).
pub(super) const EMPTY_STATE_BOTTOM_GAP: u16 = 0;
pub(super) const COMPOSER_PREFIX_WIDTH: u16 = 2;
pub(super) const DEFAULT_CHAT_SCROLL_SPEED: usize = 3;
pub(super) const MAX_CHAT_SCROLL_SPEED: usize = 50;
pub(super) const TOAST_DURATION: Duration = Duration::from_secs(3);
pub(super) const TOAST_FADE_AFTER: Duration = Duration::from_secs(2);
/// How long the selection stays lit in the bright "just copied" color before
/// it auto-clears (amp-style flash). Kept short so it reads as a confirmation.
pub(super) const SELECTION_FLASH_DURATION: Duration = Duration::from_millis(550);
/// Max gap between clicks to count as a double/triple click.
pub(super) const MULTI_CLICK_INTERVAL: Duration = Duration::from_millis(400);
/// Minimum delay between auto-scroll steps while dragging at an edge.
pub(super) const DRAG_AUTOSCROLL_INTERVAL: Duration = Duration::from_millis(40);
// Tight repaint cadence while animating; slower when idle to cut wakeups.
pub(super) const ANIMATING_FRAME_INTERVAL: Duration = Duration::from_millis(16);
pub(super) const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// Nap after a pass that handled input: short enough that a scroll/keystroke
/// repaints near-instantly and in fine increments (not trailing the idle
/// cadence), but still a real yield so the streaming task keeps progressing on
/// the current-thread runtime.
pub(super) const INPUT_REPAINT_INTERVAL: Duration = Duration::from_millis(1);
/// Minimum time the live status label holds before it may change, so fast steps
/// don't flash by unreadably.
pub(super) const STATUS_MIN_DURATION: Duration = Duration::from_millis(1500);
/// Typewriter reveal rate. Each animation frame reveals at least
/// `TYPEWRITER_MIN_CHARS` of the buffered stream text (a steady floor so a slow
/// trickle still types out) plus `1/TYPEWRITER_CATCHUP_DIVISOR` of whatever
/// backlog remains, so a fast burst catches up in a few frames. At the ~60fps
/// animating cadence this empties even a long reply in well under a second while
/// still reading as fast typing.
pub(super) const TYPEWRITER_MIN_CHARS: usize = 10;
pub(super) const TYPEWRITER_CATCHUP_DIVISOR: usize = 2;

/// Upper bound on input events drained in a single event-loop pass. A fast
/// mouse drag emits one report per cell crossed, so the whole burst must be
/// consumed before the next repaint — otherwise the selection lags the cursor.
/// The cap keeps a pathological flood (e.g. a giant paste) from starving the
/// repaint; leftover events are drained on the next tick.
pub(super) const MAX_INPUT_EVENTS_PER_TICK: usize = 512;

// Left transcript gutter: 1 col for the per-role accent bar + 1 col of padding.
pub(super) const ACCENT_GUTTER_WIDTH: u16 = 2;

/// The pinned plan panel never crushes the transcript below this many rows, and
/// its body is also capped at a fraction of the screen (see `plan_panel_height`).
pub(super) const PLAN_PANEL_MIN_TRANSCRIPT: u16 = 4;

pub(super) const COMMAND_MENU_MAX_ROWS: usize = 7;
pub(super) const PICKER_ROW_PREFIX_WIDTH: usize = 2;
/// Lines a PageUp/PageDn moves the `/skills` `/mcp` detail drill-in.
pub(super) const DETAIL_PAGE_LINES: u16 = 10;
// Drag-copy wash: a dim warm olive (the brand yellow knocked back into the dark)
// so a selection reads as brand-tinted rather than a cool steel block.
pub(super) const SELECT_WARM: Color = Color::Rgb(58, 58, 40);
/// Brighter selection wash shown for the brief post-copy flash.
pub(super) const SELECT_FLASH: Color = Color::Rgb(96, 98, 58);

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
        name: "copy",
        help_label: "/copy [n]",
        description: "copy the latest reply (or Nth) to the clipboard",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "skills",
        help_label: "/skills [add|rm …]",
        description: "list, add, or remove agent skills",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "create-skill",
        help_label: "/create-skill [intent]",
        description: "create or improve an agent skill, guided",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "mcp",
        help_label: "/mcp [add|rm …]",
        description: "list, add, or remove MCP servers",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "goal",
        help_label: "/goal <objective>",
        description: "work autonomously toward a goal until done",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "plan",
        help_label: "/plan <objective>",
        description: "investigate read-only, draft a plan, then /plan go [guidance] to execute it",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "rewind",
        help_label: "/rewind",
        description: "rewind to an earlier turn (reverts file edits)",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "config",
        help_label: "/config",
        description: "toggle chat settings (thinking, auto-approve)",
        takes_argument: false,
    },
    SlashCommandSpec {
        name: "compact",
        help_label: "/compact [fast]",
        description: "compact context now (fast = clear stale output, no model call)",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "effort",
        help_label: "/effort [level]",
        description: "set reasoning effort (bare opens a picker)",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "share",
        help_label: "/share [stop]",
        description: "share this chat to a viewer URL (stop to end)",
        takes_argument: true,
    },
    SlashCommandSpec {
        name: "help",
        help_label: "/help",
        description: "open help",
        takes_argument: false,
    },
];

/// Ghost-text argument hint trailing a bare slash command in the composer
/// (matched by command name), so `/model` shows `[name]` inline. The menu row
/// already gives the one-line description; this teaches the argument syntax,
/// and the dropdown is suppressed while it shows (see `visible_command_menu`).
/// `[…]` marks an optional argument, `<…>` a required one.
pub(super) fn command_usage_hint(name: &str) -> Option<&'static str> {
    match name {
        // Richer than a placeholder — teaches the subcommand grammar.
        "mcp" => Some("[add <command> [args…] | rm <name>]"),
        "skills" => Some("[add <name>|<github:owner/repo> | rm <name>]"),
        "create-skill" => Some("[what the skill should do]"),
        "goal" => Some("<objective> | stop"),
        "plan" => Some("<objective> | go [guidance] | stop"),
        "share" => Some("[stop]"),
        "compact" => Some("[fast]"),
        "model" => Some("[name]"),
        "key" => Some("[id|name]"),
        "resume" => Some("[query]"),
        "copy" => Some("[n]"),
        "detach" => Some("<n>"),
        // `attach` is deliberately omitted: typing `/attach ` opens path
        // completion, which is more useful than a static `<path>` ghost — and a
        // ghost would suppress that menu (see `visible_command_menu`).
        _ => None,
    }
}

/// Compact a count: `1234` → `1.2k`, `12345` → `12k`, `<1000` verbatim.
pub(super) fn humanize_count(n: usize) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{}k", n / 1000)
    }
}

/// `/compact` result notice; `kind` names what happened ("cleared stale output" /
/// "summarized older turns").
pub(super) fn freed_notice(freed: usize, kind: &str) -> (Color, String) {
    if freed == 0 {
        (MUTED, "already compact — nothing to free".to_string())
    } else {
        (
            MUTED,
            format!("freed ~{} tokens — {kind}", humanize_count(freed)),
        )
    }
}

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
    /// `--resume` request: `Some("")` opens the session picker at startup,
    /// `Some(id)` jumps straight to that session, `None` starts fresh.
    pub initial_resume: Option<String>,
    /// `--max-context <SIZE>` manual context-window override (tokens). Session-only.
    pub max_context: Option<u64>,
    /// `--share`: start live sharing at launch (device-link verified beforehand).
    pub share: bool,
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
    /// The durably-persisted agent-engine transcript (raw OpenAI messages with
    /// tool_calls + results), restored verbatim into the engine on resume for
    /// exact tool history. `None` for non-agent or pre-feature sessions.
    pub(super) engine_messages: Option<Vec<serde_json::Value>>,
}

impl LoadedSession {
    pub(super) fn from_state(
        state: crate::services::session_store::ChatSessionState,
    ) -> Result<Self> {
        let messages = decrypt_to_chat_messages(&state)?;
        let engine_messages = state.decrypt_engine_messages();

        Ok(Self {
            key_id: state.key_id,
            session_id: state.session_id,
            raw_model: state.model,
            messages,
            engine_messages,
        })
    }
}

/// One discovered skill in the `/skills` overlay: name + a one-line advert,
/// whether it's enabled (the full body still loads on demand when called), and
/// where it lives (so the overlay can show the path and protect repo skills from
/// deletion).
#[derive(Clone, Debug)]
pub(super) struct SkillToggle {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) enabled: bool,
    pub(super) dir: std::path::PathBuf,
    pub(super) scope: crate::agent::skills::SkillScope,
    /// The SKILL.md body, for the Enter drill-in preview (loaded at open time).
    pub(super) body: String,
}

/// The interactive `/skills` overlay: a filterable toggle list. `query` is the
/// type-to-filter text; `selected` is the highlighted item's index INTO `items`
/// (not the filtered view). `adding` holds the in-progress add text; `pending_delete`
/// is the index armed for a two-press Ctrl+D delete; `viewing` is the index whose
/// detail drill-in is open; `detail_scroll` is that drill-in's vertical scroll
/// offset in (already-wrapped) lines.
#[derive(Clone, Debug, Default)]
pub(super) struct SkillsOverlay {
    pub(super) items: Vec<SkillToggle>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) adding: Option<String>,
    pub(super) pending_delete: Option<usize>,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
}

impl SkillsOverlay {
    /// Indices of `items` matching `query` (fuzzy over name + description), in
    /// order. An empty query matches everything.
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| {
                matches_fuzzy(&self.query, &format!("{} {}", it.name, it.description))
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub(super) fn select_prev(&mut self) {
        self.pending_delete = None;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.pending_delete = None;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    /// Re-anchor the selection to the first match after the query changed.
    pub(super) fn refilter(&mut self) {
        self.pending_delete = None;
        let filtered = self.filtered_indices();
        if !filtered.contains(&self.selected) {
            self.selected = filtered.first().copied().unwrap_or(0);
        }
    }

    /// Whether the highlighted item is currently visible (matches the filter) —
    /// gates Enter/Tab/Ctrl+D so they never act on a filtered-out row.
    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// First Ctrl+D on a row arms the delete (returns `false`); a second on the
    /// same row confirms it (returns `true`). Selecting away clears the arm.
    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        if self.pending_delete == Some(self.selected) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(self.selected);
            false
        }
    }
}

/// Coarse health of one MCP server, for coloring its `/mcp` status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum McpHealth {
    Connected,
    Failed,
    /// An HTTP server that answered 401 — it needs OAuth (authorize with Ctrl+O).
    NeedsAuth,
    Idle,
    Disabled,
}

/// One row in the `/mcp` overlay: a configured server, its live status (a snapshot
/// taken when the overlay opened), whether it's enabled, and which file defines
/// it (only `User` servers can be removed in-overlay).
#[derive(Clone, Debug)]
pub(super) struct McpServerRow {
    pub(super) name: String,
    pub(super) status: String,
    pub(super) health: McpHealth,
    pub(super) enabled: bool,
    pub(super) scope: crate::agent::mcp::ServerScope,
    /// The configured launch command, for the Enter drill-in detail view.
    pub(super) command: String,
}

/// The interactive `/mcp` overlay: a filterable toggle list of configured MCP
/// servers. `query` is the type-to-filter text; `selected` is the highlighted
/// item's index INTO `items`; `adding` holds the in-progress add text;
/// `pending_delete` is the index armed for a two-press Ctrl+D delete (removal
/// edits the user `mcp.json`, so it confirms); `viewing` is the index whose
/// detail drill-in is open; `detail_scroll` is that drill-in's vertical scroll
/// offset in (already-wrapped) lines.
#[derive(Clone, Debug, Default)]
pub(super) struct McpOverlay {
    pub(super) items: Vec<McpServerRow>,
    pub(super) selected: usize,
    pub(super) query: String,
    pub(super) adding: Option<String>,
    pub(super) pending_delete: Option<usize>,
    pub(super) viewing: Option<usize>,
    pub(super) detail_scroll: u16,
}

impl McpOverlay {
    /// Indices of `items` matching `query` (fuzzy over name + status), in order.
    pub(super) fn filtered_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter(|(_, it)| matches_fuzzy(&self.query, &format!("{} {}", it.name, it.status)))
            .map(|(i, _)| i)
            .collect()
    }

    pub(super) fn select_prev(&mut self) {
        self.pending_delete = None;
        move_within(&self.filtered_indices(), &mut self.selected, -1);
    }

    pub(super) fn select_next(&mut self) {
        self.pending_delete = None;
        move_within(&self.filtered_indices(), &mut self.selected, 1);
    }

    /// Re-anchor the selection to the first match after the query changed.
    pub(super) fn refilter(&mut self) {
        self.pending_delete = None;
        let filtered = self.filtered_indices();
        if !filtered.contains(&self.selected) {
            self.selected = filtered.first().copied().unwrap_or(0);
        }
    }

    /// Whether the highlighted item is currently visible (matches the filter).
    pub(super) fn has_selection(&self) -> bool {
        self.filtered_indices().contains(&self.selected)
    }

    /// First Ctrl+D on a row arms the delete (returns `false`); a second on the
    /// same row confirms it (returns `true`). Selecting away clears the arm.
    pub(super) fn arm_or_confirm_delete(&mut self) -> bool {
        if self.pending_delete == Some(self.selected) {
            self.pending_delete = None;
            true
        } else {
            self.pending_delete = Some(self.selected);
            false
        }
    }
}

/// One toggleable chat preference, identified so the handler knows which flag to
/// flip (and where to persist it) without matching on the row's label text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ConfigSetting {
    /// Whether the model reasons before answering (off stops it entirely).
    Thinking,
    /// Run the agent's tools without asking (mirrors the Shift+Tab toggle).
    AutoApprove,
    /// Whether the agent may use aivo's hosted web_search (`/v1/search`).
    UseWebSearch,
    AgentTools,
}

/// One row in the `/config` overlay: a boolean preference with a label and a
/// one-line description. The current value is read live from the app (see
/// `config_setting_enabled`), not cached here, so the row can't drift.
#[derive(Clone, Debug)]
pub(super) struct ConfigToggle {
    pub(super) setting: ConfigSetting,
    pub(super) label: &'static str,
    pub(super) description: &'static str,
}

/// The interactive `/config` overlay: a small *fixed* toggle list. Unlike
/// `/skills` and `/mcp` there's nothing to filter, add, or remove — so it's just
/// a navigable list whose rows flip on Enter/Space. `selected` indexes `items`.
#[derive(Clone, Debug, Default)]
pub(super) struct ConfigOverlay {
    pub(super) items: Vec<ConfigToggle>,
    pub(super) selected: usize,
}

impl ConfigOverlay {
    pub(super) fn select_prev(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    pub(super) fn select_next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1).min(self.items.len() - 1);
        }
    }
}

/// Move `selected` (an index into the full list) one step (`dir` = -1/+1) within
/// `filtered` (the in-order matching indices), clamped to the filtered ends.
fn move_within(filtered: &[usize], selected: &mut usize, dir: i32) {
    if filtered.is_empty() {
        return;
    }
    let pos = filtered.iter().position(|&i| i == *selected).unwrap_or(0);
    let next = if dir < 0 {
        pos.saturating_sub(1)
    } else {
        (pos + 1).min(filtered.len() - 1)
    };
    *selected = filtered[next];
}

/// Session decision on whether to spawn a repo's project `.mcp.json` STDIO
/// servers — the one piece of project content aivo *executes* (a local child
/// process) rather than reads, so it's gated like aivo's own actions. Seeded
/// from the persistent per-repo allow-list on the first connect; the consent
/// card sets it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum ProjectMcpConsent {
    #[default]
    Unknown,
    Allowed,
    Denied,
}

/// A pending consent card: a repo's `.mcp.json` wants to spawn these stdio
/// servers. Carries what's needed to (re)connect once the user decides.
#[derive(Clone, Debug)]
pub(super) struct McpConsentPrompt {
    /// `(name, "command args…")` for each project stdio server, for display.
    pub(super) servers: Vec<(String, String)>,
    pub(super) cwd: String,
    /// The base opt-out set (the user's `/mcp` toggles) to reconnect with.
    pub(super) base_disabled: std::collections::HashSet<String>,
}

/// Content digest of a repo's project `.mcp.json` stdio servers — the exact
/// `(name, "command args…")` set the user is shown on the consent card, already
/// sorted by name. An "always" approval is bound to this digest, so a later
/// edit that swaps in a different command changes the hash and re-prompts rather
/// than silently reusing the old consent. (Covers the spawn command + args; env
/// is not yet folded in.)
pub(super) fn project_mcp_digest(servers: &[(String, String)]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (name, display) in servers {
        hasher.update(name.as_bytes());
        hasher.update([0u8]);
        hasher.update(display.as_bytes());
        hasher.update([0u8]);
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Clone)]
pub(super) enum Overlay {
    None,
    /// `/help` — slash commands, skills, and keybindings. `scroll` is the body's
    /// vertical scroll offset (the content is taller than the box on most
    /// terminals), clamped by the renderer and written back each frame.
    Help {
        scroll: u16,
    },
    /// `/skills` — the agent skills discovered for the working dir, toggleable.
    Skills(SkillsOverlay),
    /// `/mcp` — the configured MCP servers with status, toggleable.
    Mcp(McpOverlay),
    /// `/config` — a small fixed list of chat preferences, toggleable.
    Config(ConfigOverlay),
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
    /// A `/rewind` target. `history_index` is the truncation point in the chat
    /// history; `ordinal` is the matched engine checkpoint (reverts files through
    /// it), or `None` for turns with no live checkpoint (rewind conversation-only).
    RewindTurn {
        history_index: usize,
        ordinal: Option<usize>,
    },
    /// A `/effort` reasoning level (e.g. `low`/`high`).
    Effort(String),
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
    Rewind,
    Effort,
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
    pub(super) pending_response: String,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) last_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    pub(super) context_window: u64,
    pub(super) context_is_estimate: bool,
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

/// A discovered skill surfaced as a user-typeable slash command (`/repo-study`),
/// so the `/` menu suggests it and submitting it invokes the skill. `name` is the
/// skill name; `description` is its one-line advert (already truncated for the
/// menu). The full body is loaded on invocation, not cached here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SkillCommand {
    pub(super) name: String,
    pub(super) description: String,
}

impl SkillCommand {
    pub(super) fn command_label(&self) -> String {
        format!("/{}", self.name)
    }

    /// What lands in the composer when the row is Tab-completed: `/name ` with a
    /// trailing space so the user can type the skill's input right away.
    pub(super) fn insertion_text(&self) -> String {
        format!("/{} ", self.name)
    }
}

#[derive(Clone)]
pub(super) enum ComposerMenuEntry {
    Command(&'static SlashCommandSpec),
    Skill(SkillCommand),
    Path(PathMenuEntry),
}

impl ComposerMenuEntry {
    pub(super) fn label(&self) -> String {
        match self {
            Self::Command(command) => command.command_label(),
            Self::Skill(skill) => skill.command_label(),
            Self::Path(path) => path.label.clone(),
        }
    }

    pub(super) fn description(&self) -> &str {
        match self {
            Self::Command(command) => command.description,
            Self::Skill(skill) => &skill.description,
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
            pending_response: app.pending_response.clone(),
            pending_reasoning: app.pending_reasoning.clone(),
            pending_submit: app.pending_submit.clone(),
            last_usage: app.last_usage,
            context_tokens: app.context_tokens,
            context_window: app.context_window,
            context_is_estimate: app.context_is_estimate,
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
    /// `!cmd`: run a shell command locally (Claude Code's bash prefix).
    Shell(String),
}

/// An in-flight `!cmd` local shell run. Independent of model turns (`sending`):
/// the background reader task streams output here line by line, rendered live in
/// the transcript's volatile tail and committed to a `local_command` history
/// entry when it finishes. `stdout`/`stderr` are capped by the reader (see
/// `run_shell_streaming`).
pub(super) struct LocalCommandRun {
    pub(super) task: JoinHandle<()>,
    /// Kills the PTY child so `esc` (and app exit) can stop a running command —
    /// the blocking PTY read can't be cancelled by aborting `task` alone.
    pub(super) killer: Box<dyn portable_pty::ChildKiller + Send + Sync>,
    pub(super) started_at: Instant,
    pub(super) command: String,
    pub(super) stdout: String,
    pub(super) stderr: String,
}

/// The full captured stdout/stderr of a finished/interrupted `!cmd` run, kept in
/// memory only (never persisted) so an inline-expanded block can show everything
/// even though the transcript and the on-disk session keep just a bounded preview.
/// Held in `local_outputs` keyed by history index; cleared on `/new` and resume.
/// The command line and exit status aren't stored here — an expanded block reads
/// those from its persisted `local_command` entry, which survives a resume.
#[derive(Clone)]
pub(super) struct LocalCommandOutput {
    pub(super) stdout: String,
    pub(super) stderr: String,
}

#[allow(dead_code)]
pub(super) enum ClipboardPayload {
    Text(String),
    Attachment(MessageAttachment),
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TranscriptPoint {
    pub(super) row: usize,
    pub(super) column: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TranscriptSelection {
    pub(super) anchor: TranscriptPoint,
    pub(super) focus: TranscriptPoint,
}

impl TranscriptSelection {
    pub(super) fn is_empty(self) -> bool {
        self.anchor == self.focus
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct TranscriptHitbox {
    pub(super) area: Rect,
    pub(super) first_row: usize,
    pub(super) rows: Vec<String>,
}

/// Snapshot of the rendered screen so a drag can copy from anywhere on it, not
/// just the transcript. `rows[i]` is screen row `area.y + i`, one symbol per
/// display column (CJK wide cells included).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ScreenSurface {
    pub(super) area: Rect,
    pub(super) rows: Vec<String>,
}

/// Which surface a drag-selection reads from: the scroll-aware transcript, or the
/// flat [`ScreenSurface`] (overlays, composer, footer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectionSurface {
    Transcript,
    Screen,
}

/// Memoizes the heavy transcript build + word-wrap across frames. The transcript
/// body (intro + history + the streamed reply + notice) is re-rendered and
/// re-wrapped only when its content or the render width actually changes — not on
/// every animation tick. The live spinner status line is deliberately excluded
/// (it changes every frame) and appended fresh at draw time, so spinner
/// animation and idle keystrokes hit the cache instead of reparsing all history.
pub(super) struct TranscriptCache {
    /// Fingerprint of the content that produced `body` (see `transcript_body_fp`).
    pub(super) fp: u64,
    /// Terminal width the `plain_prepass` height estimate was computed for.
    pub(super) area_width: u16,
    /// The logical body lines (no spinner), already compacted.
    pub(super) body: RenderedTranscript,
    /// Char-wrapped row count of `body` at `area_width - ACCENT_GUTTER_WIDTH`,
    /// used to size the transcript pane; with no scrollbar column reserved this
    /// is also the exact text width.
    pub(super) plain_prepass: usize,
    /// Text width `wrapped` is valid for (0 = not wrapped yet).
    pub(super) styled_width: u16,
    /// `body` word-wrapped to `styled_width`.
    pub(super) wrapped: Option<WrappedTranscript>,
}

/// Cross-frame memo of the VOLATILE tail (the streamed reply, a running `!cmd`'s
/// preview, and any notice) — the part `TranscriptCache` deliberately excludes.
/// The reply and command stream append-only, so a cheap fingerprint of their
/// lengths + the notice identifies the rendered output; caching the markdown
/// render AND the word-wrap turns the per-frame O(reply) re-parse and re-wrap
/// (which the 60fps spinner redraw makes O(reply²) over a long answer) into an
/// O(1) lookup whenever no new token arrived. The spinner itself is excluded (it
/// animates every frame) and wrapped fresh at compose time.
pub(super) struct VolatileTailCache {
    /// Fingerprint of the tail inputs (see `volatile_tail_fp`).
    pub(super) fp: u64,
    /// Render width the `lines` markdown/table layout was produced at.
    pub(super) render_width: u16,
    /// The rendered tail blocks (each with its leading spacing blank), styled.
    pub(super) lines: Vec<StyledLine>,
    pub(super) bars: Vec<Option<Color>>,
    /// Width the `prepass` char-wrap height was computed for (0 = none yet).
    pub(super) plain_width: u16,
    /// Char-wrapped row count of `lines`, to size the pane before wrapping.
    pub(super) prepass: usize,
    /// Text width `wrapped` is valid for (0 = not wrapped yet).
    pub(super) styled_width: u16,
    /// `lines` word-wrapped to `styled_width` (None when `lines` is empty).
    pub(super) wrapped: Option<WrappedTranscript>,
}

/// Live drag that has reached the top/bottom edge of the transcript: the event
/// loop keeps scrolling in `dir` (−1 up, +1 down) and re-anchoring the focus to
/// `column` so a selection can run past a single screenful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DragAutoscroll {
    pub(super) dir: i8,
    pub(super) column: u16,
}

/// Tracks consecutive left-clicks so double/triple clicks select word/line.
#[derive(Clone, Copy, Debug)]
pub(super) struct ClickTracker {
    pub(super) at: Instant,
    pub(super) point: TranscriptPoint,
    pub(super) count: u8,
}

/// A brief floating confirmation (top-right) that auto-expires and fades — used
/// for copy confirmations and transient mode toggles (e.g. auto-approve), so
/// they flash and vanish instead of lingering in the transcript.
#[derive(Clone, Debug)]
pub(super) struct Toast {
    pub(super) text: String,
    pub(super) created_at: Instant,
    pub(super) expires_at: Instant,
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
    /// Copy the Nth-latest assistant reply (default 1 = most recent) to the clipboard.
    Copy(Option<usize>),
    /// Agent skills: bare opens the overlay; `add …` / `rm <name>` manage them.
    Skills(Option<String>),
    /// MCP servers: bare opens the overlay; `add …` / `rm <name>` manage them.
    Mcp(Option<String>),
    /// Goal mode: `<objective>` works autonomously until done; bare shows status,
    /// `stop` ends it.
    Goal(Option<String>),
    /// Plan mode: `<objective>` investigates read-only and drafts an
    /// implementation plan; `go` executes it in a fresh context; bare shows
    /// status, `stop` discards the pending plan.
    Plan(Option<String>),
    /// Reasoning effort: bare opens a picker of the model's levels, `<level>`
    /// sets it directly.
    Effort(Option<String>),
    /// Built-in `create-skill` command: starts the guided create/improve-a-skill
    /// workflow. The optional argument is the initial intent (what the skill
    /// should do); bare just opens the workflow.
    CreateSkill(Option<String>),
    /// Invoke a discovered skill by name as a slash command (`/repo-study`), the
    /// way Claude Code / grok CLI surface skills. `argument` is the optional text
    /// after the name, passed to the skill (substituted into `$ARGUMENTS` if the
    /// body uses it, else appended as input).
    Skill {
        name: String,
        argument: Option<String>,
    },
    /// Open the rewind picker: jump back to an earlier turn, reverting the file
    /// edits made since and restoring that turn's prompt to the composer.
    Rewind,
    /// Open the `/config` overlay: a toggle list of chat preferences.
    Config,
    /// `/compact` folds older turns via the LLM; `fast` clears stale tool output only.
    Compact {
        fast: bool,
    },
    /// Share this chat: bare/`start` opens a viewer URL (re-shown if already
    /// live); `stop` ends it.
    Share(Option<String>),
    Help,
}

/// A turn-finish event captured while the typewriter still has buffered text to
/// reveal. The event loop replays it once the buffer drains so the final reply
/// types out fully before the turn commits. Mirrors the finish-bearing
/// [`RuntimeEvent`] variants.
pub(super) enum DeferredFinish {
    Chat {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    Agent {
        steps: usize,
        tokens: u64,
        context_tokens: u64,
    },
}

pub(super) enum RuntimeEvent {
    Delta(ChatResponseChunk),
    Finished {
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    },
    ModelsLoaded(std::result::Result<Vec<ModelChoice>, String>),
    /// The background catalog warm finished — re-resolve the active model's
    /// limits (window + `/effort` levels) so catalog-advertised efforts surface.
    CatalogWarmed,
    ResumeLoaded {
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    },
    /// A cursor-agent ACP session finished opening on a background task. The
    /// event loop stores it on the app so subsequent turns reuse it without
    /// paying the Node.js startup cost again.
    CursorSessionOpened(crate::services::cursor_acp::CursorAcpSession),
    /// The agent engine invoked a tool — render a `→ verb(args)` transcript step.
    /// `id` (cursor's `toolCallId`) correlates a later `AgentToolUpdate`; `None`
    /// for the in-process agent, which reports results separately.
    AgentToolCall {
        id: Option<String>,
        name: String,
        args: serde_json::Value,
        /// Pre-edit start line of each diff pair (aligned with `edit_diffs`), so
        /// the card can number rows; empty for non-edit and cursor calls.
        line_starts: Vec<Option<usize>>,
    },
    /// Enriches an earlier `AgentToolCall` (matched by `id`) in place: the
    /// resolved target (real path/pattern) and/or a compact result. Cursor only —
    /// its start event omits the target, which arrives in a `tool_call_update`.
    AgentToolUpdate {
        id: String,
        args: Option<serde_json::Value>,
        result: Option<String>,
        failed: bool,
    },
    /// The agent engine's tool returned — render the `⎿ result` step.
    AgentToolResult {
        content: String,
    },
    /// The agent's just-streamed output was a tool call written as text: drop the
    /// uncommitted segment so the markup never reaches the scrollback.
    AgentDiscardSegment,
    /// A background MCP connect finished. Carries the client (possibly empty) and
    /// the generation it started under; the event loop caches it and rebuilds the
    /// engine if it brought tools, but drops a result from a stale generation.
    McpConnected {
        client: std::sync::Arc<crate::agent::mcp::McpClient>,
        generation: u64,
    },
    /// One MCP server's handshake resolved mid-connect (servers connect
    /// concurrently). Lets the open `/mcp` overlay flip that single row to its real
    /// status while the rest are still connecting. `generation` guards against a
    /// stale connect; `health`/`status` are the already-mapped display values.
    McpServerProgress {
        name: String,
        status: String,
        health: McpHealth,
        generation: u64,
    },
    /// A background OAuth authorize for an HTTP MCP server produced its browser
    /// URL — shown as a notice so the user has it if the auto-opened browser
    /// didn't appear.
    McpAuthorizeUrl {
        url: String,
    },
    /// A background OAuth authorize finished. On `Ok`, the event loop persists the
    /// credential and reconnects so the now-authorized server's tools appear.
    McpAuthorized {
        name: String,
        result: std::result::Result<crate::services::mcp_oauth::McpOAuthCredential, String>,
    },
    /// The agent set/updated its task plan (update_plan tool). Carries the JSON
    /// array of `{step, status}`; rendered as a checklist card in the transcript.
    AgentPlan(serde_json::Value),
    /// Engine status line (compaction, step limit, …) — shown as a notice.
    AgentNotice(String),
    /// A mutating tool needs approval. The event loop shows a permission card
    /// and replies with the decision; the engine task awaits `reply`.
    AgentPermission {
        tool: String,
        preview: Option<String>,
        reply: tokio::sync::oneshot::Sender<crate::agent::protocol::Decision>,
    },
    /// The agent's `switch_model`/`set_effort` tools: apply the change and reply to the
    /// waiting engine task (Ok = confirmation, Err = why not). Same oneshot pattern as
    /// [`AgentPermission`](Self::AgentPermission).
    AgentSwitchModel {
        model: String,
        reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
    },
    AgentSetEffort {
        level: String,
        reply: tokio::sync::oneshot::Sender<std::result::Result<String, String>>,
    },
    /// Live context-window fill from the agent engine mid-turn: `measured` true =
    /// a provider step total (exact), false = a chars/4 request estimate. Moves
    /// the footer's context stat during an agent turn instead of only at the end.
    AgentContext {
        tokens: u64,
        measured: bool,
    },
    /// The turn's cumulative generated (output) tokens so far — drives the live
    /// per-turn counter in the status line.
    AgentTurnTokens(u64),
    /// A delegated sub-agent began a step — updates the parent's status-line label
    /// (`↳ <agent>: <action> · step N`) only, never a transcript card.
    AgentSubActivity {
        agent: String,
        tool: String,
        args: serde_json::Value,
        step: usize,
    },
    /// An agent error (e.g. an LLM/API failure) — shown as an error-hued notice.
    AgentError(String),
    /// The agent turn finished (engine called `footer`) — commit the turn.
    AgentFinished {
        steps: usize,
        tokens: u64,
        /// Last step's real prompt+completion (context-window fill), 0 if the
        /// provider reported no usage.
        context_tokens: u64,
    },
    /// A `!cmd` local shell run produced one output line — appended to the live
    /// in-progress run and shown immediately in the transcript tail.
    LocalCommandLine {
        is_err: bool,
        line: String,
    },
    /// A `!cmd` local shell run finished (drained, capped, or timed out). Commits
    /// the captured output as a `local_command` history entry. `truncated` marks
    /// output cut short by the capture caps.
    LocalCommandDone {
        exit_code: i64,
        truncated: bool,
    },
    /// A background `/skills add <source>` install finished, off the event loop.
    SkillInstalled {
        source: String,
        result: std::result::Result<crate::agent::skills::InstallOutcome, String>,
    },
    /// A `/share` (or `--share`) start finished: `Ok` the handle, `Err` the reason.
    LiveShareReady(std::result::Result<crate::services::share_live::LiveShareHandle, String>),
}

/// A live in-process agent for the current chat, keyed by the (key, model) it
/// was built for so a key/model switch rebuilds it. The engine owns the real
/// LLM conversation; chat history is a display log.
pub(super) struct AgentSession {
    pub(super) key_id: String,
    pub(super) model: String,
    pub(super) engine: std::sync::Arc<tokio::sync::Mutex<crate::agent::engine::AgentEngine>>,
}

/// A pending tool-permission prompt: shown as a card until the user answers,
/// then `reply` delivers the decision to the waiting engine task.
pub(super) struct PendingPermission {
    pub(super) tool: String,
    pub(super) preview: Option<String>,
    pub(super) reply: tokio::sync::oneshot::Sender<crate::agent::protocol::Decision>,
}

/// Active `/goal` autonomous loop: after each agent turn the app auto-continues
/// toward `objective` (self-checking for completion) until the agent signals
/// done, `iteration` hits `max`, or the user interrupts. `None` = not in goal mode.
#[derive(Clone)]
pub(super) struct GoalState {
    pub(super) objective: String,
    pub(super) iteration: usize,
    pub(super) max: usize,
}

/// One persisted input-history entry, tagged with the launch dir it was typed
/// in. Legacy plain-text lines load with an empty `cwd` (shown everywhere).
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct DraftHistoryEntry {
    pub(super) cwd: String,
    pub(super) text: String,
}

pub(super) struct ChatTuiApp {
    pub(super) session_store: SessionStore,
    pub(super) cache: ModelsCache,
    pub(super) client: Client,
    pub(super) key: ApiKey,
    pub(super) copilot_tm: Option<Arc<CopilotTokenManager>>,
    /// Chat's isolated sandbox dir (cursor cwd / display for non-agent chats).
    pub(super) cwd: String,
    /// The real launch dir — where the agent reads/edits files, and what the
    /// header/footer show for agent-capable chats (a safety signal).
    pub(super) real_cwd: String,
    /// Current git branch of `display_cwd` (when it's a repo), shown after the cwd
    /// in the footer. Refreshed on a throttle (see `refresh_git_branch`) so a
    /// checkout — by the user or the agent — shows up without re-reading
    /// `.git/HEAD` every frame. `None` when the dir isn't a git work tree.
    pub(super) git_branch: Option<String>,
    /// When `git_branch` was last refreshed, to throttle the `.git/HEAD` read.
    pub(super) git_branch_checked_at: Option<Instant>,
    pub(super) raw_model: String,
    pub(super) model: String,
    /// Upstream model captured from the most recent turn so heartbeat
    /// saves (no fresh turn in scope) can still write `billed_model` to
    /// the session index. Cleared on key/model switch and resume.
    pub(super) billed_model: Option<String>,
    pub(super) format: ChatFormat,
    pub(super) history: Vec<ChatMessage>,
    pub(super) draft: String,
    pub(super) draft_attachments: Vec<MessageAttachment>,
    pub(super) cursor: usize,
    pub(super) command_menu: CommandMenuState,
    /// Discovered skills offered as user-typeable slash commands (`/repo-study`).
    /// Refreshed from `discover_skills` (minus the `/skills` disabled set) at
    /// startup and after any skill mutation; read by the `/` menu and command
    /// resolver. Empty when no skills are available.
    pub(super) skill_commands: Vec<SkillCommand>,
    /// Up-arrow recall list for the current launch dir — the cwd-filtered view
    /// of `draft_history_all` that `history_prev`/`history_next` walk.
    pub(super) draft_history: Vec<String>,
    /// Full global history across every dir; persisted source of truth that the
    /// `draft_history` view is derived from.
    pub(super) draft_history_all: Vec<DraftHistoryEntry>,
    pub(super) draft_history_index: Option<usize>,
    pub(super) draft_history_stash: Option<String>,
    pub(super) session_id: String,
    pub(super) overlay: Overlay,
    pub(super) notice: Option<(Color, String)>,
    pub(super) pending_response: String,
    /// Stream text received but not yet revealed by the typewriter. Deltas land
    /// here; [`tick_typewriter`](ChatTuiApp::tick_typewriter) drips it into
    /// `pending_response` (the displayed reply) over successive frames.
    pub(super) incoming_buffer: String,
    /// A turn-finish event held back until the typewriter has revealed the whole
    /// buffer, so the tail types out instead of snapping in at the end.
    pub(super) pending_finish: Option<DeferredFinish>,
    pub(super) pending_reasoning: String,
    pub(super) pending_submit: Option<PendingSubmission>,
    pub(super) sending: bool,
    pub(super) request_started_at: Option<Instant>,
    /// Context fill before a manual `/compact` (LLM) turn, so the finish path reports
    /// the freed delta and skips the duration marker. `None` outside a compact.
    pub(super) compact_before: Option<u64>,
    /// Current tool step, present-tense (`running grep`), + when it started.
    /// Feeds the inline status label.
    pub(super) last_tool_action: Option<(String, Instant)>,
    /// The status label on screen + when first shown; throttled by
    /// `tick_status_throttle` so it switches at most once per `STATUS_MIN_DURATION`.
    pub(super) status_display: Option<(String, Instant)>,
    /// This turn's cumulative generated tokens (status-line tail). Reset at turn
    /// start; fed by `AgentTurnTokens` (agent) or `Usage` (plain chat).
    pub(super) turn_output_tokens: u64,
    /// A connection retry is in progress → status reads "Working", not
    /// "Thinking". Set on the retry notice, cleared on progress.
    pub(super) retrying: bool,
    pub(super) last_usage: Option<TokenUsage>,
    /// Provider-measured usage streamed mid-turn (Anthropic reports it from
    /// `message_start`; OpenAI/Responses/Google only at the end). Drives the
    /// footer's context-fill while `sending`, then folds into `last_usage` at
    /// turn end. `None` outside an in-flight turn.
    pub(super) live_usage: Option<TokenUsage>,
    pub(super) context_tokens: u64,
    /// Cumulative provider-measured tokens for the CURRENT chat session (across
    /// all its turns), persisted into the chat index entry on every save so
    /// `aivo stats --since` can attribute windowed chat usage. Reset on `/new`,
    /// re-seeded from the stored entry on resume.
    pub(super) session_tokens: crate::services::session_store::SessionTokens,
    /// Active model's context window (tokens), 0 = unknown. Cached on model/key
    /// change for the footer utilization stat; see `refresh_context_window`.
    pub(super) context_window: u64,
    /// `--max-context` manual override (tokens); wins over the resolved window in
    /// `refresh_context_window` and the engine build. Session-only.
    pub(super) context_window_override: Option<u64>,
    /// `context_tokens` is a chars/4 estimate of the visible transcript, not a
    /// provider-measured count (cursor ACP and agents-without-usage). The footer
    /// marks these with `~` since the model's real context is larger.
    pub(super) context_is_estimate: bool,
    pub(super) follow_output: bool,
    /// Bumped on any in-place edit of a history entry (cursor tool-call
    /// enrichment) so the transcript cache fingerprint invalidates — the
    /// fingerprint otherwise assumes entries are only appended/cleared.
    pub(super) transcript_revision: u64,
    pub(super) transcript_scroll: usize,
    pub(super) transcript_width: u16,
    pub(super) transcript_view_height: u16,
    /// Max scroll offset the last render computed from the composed rows. The
    /// hot scroll handlers reuse it (per wheel event) so a fast scroll skips the
    /// full `build_transcript` rebuild `max_scroll` would do; `None` before the
    /// first render falls back to recomputing. At most one frame stale, and the
    /// render re-clamps `transcript_scroll` each pass, so staleness is benign.
    pub(super) last_max_scroll: Option<usize>,
    pub(super) transcript_hitbox: Option<TranscriptHitbox>,
    /// Click region of the "jump to bottom" pill from the last render; `None` when
    /// it isn't shown (transcript pinned to the bottom, or no overflow).
    pub(super) jump_to_bottom_hit: Option<Rect>,
    /// The composer text region from the last render, for mouse cursor-placement
    /// and the key-handler's wrap math (it needs the width before the next frame).
    /// `None` until the first render.
    pub(super) composer_text_area: Option<Rect>,
    /// Vertical scroll (in visual rows) of the draft within the composer, so the
    /// cursor stays visible when a multi-line draft outgrows the composer's rows.
    /// Recomputed each render from the cursor position; never persisted.
    pub(super) composer_scroll: usize,
    /// Cross-frame memo of the built + wrapped transcript body; see
    /// [`TranscriptCache`]. Rebuilt only on content/width change.
    pub(super) transcript_cache: Option<TranscriptCache>,
    /// Cross-frame memo of the volatile tail (streamed reply + running `!cmd` +
    /// notice); see [`VolatileTailCache`]. Keeps a 60fps redraw off the O(reply²)
    /// re-parse/re-wrap path while the answer streams.
    pub(super) volatile_tail_cache: Option<VolatileTailCache>,
    pub(super) transcript_selection: Option<TranscriptSelection>,
    pub(super) transcript_drag_active: bool,
    /// Full-screen drag selection (overlays, composer, footer), mutually exclusive
    /// with `transcript_selection` — starting either drag clears the other.
    pub(super) screen_selection: Option<TranscriptSelection>,
    pub(super) screen_drag_active: bool,
    pub(super) screen_surface: Option<ScreenSurface>,
    /// Region the screen selection is confined to — a modal's inner content rect
    /// while one is open, so a drag selects inside the modal, not the whole line.
    /// `None` = the full screen.
    pub(super) screen_region: Option<Rect>,
    /// Live edge auto-scroll state while dragging a selection.
    pub(super) drag_autoscroll: Option<DragAutoscroll>,
    /// When the last auto-scroll step fired, to throttle the scroll rate.
    pub(super) last_autoscroll: Option<Instant>,
    /// Last left-click, for double/triple-click word/line selection.
    pub(super) last_click: Option<ClickTracker>,
    /// While set (and not expired) the selection is painted in the bright flash
    /// color; on expiry the selection auto-clears.
    pub(super) selection_flash_until: Option<Instant>,
    pub(super) scroll_speed: usize,
    /// Bare Up/Down at the composer edge scroll the transcript instead of walking
    /// draft history — for mobile terminals (Termux) where swipes arrive as arrows.
    pub(super) swipe_scroll: bool,
    pub(super) toast: Option<Toast>,
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
    pub(super) exit_confirm_pending: bool,
    /// Live `cursor-agent acp` connection scoped to the current chat session.
    /// `None` outside of cursor keys and before the first turn.
    pub(super) cursor_acp_session: Option<crate::services::cursor_acp::CursorAcpSession>,
    /// A resumed session's durable agent transcript (raw OpenAI messages with
    /// tool_calls + results), awaiting the next engine build to be restored
    /// verbatim (exact tool history). Consumed (`take`) on build; `None` otherwise.
    pub(super) pending_agent_messages: Option<Vec<serde_json::Value>>,
    /// Active `/goal` autonomous loop, or `None`. Drives auto-continuation between
    /// agent turns; cleared on completion, the iteration cap, `/goal stop`, an
    /// interrupt, `/new`, resume, or a key/model switch.
    pub(super) goal_mode: Option<GoalState>,
    /// The in-flight turn is a `/plan` investigation: capture its reply as the
    /// pending plan when it finishes. Reset on interrupt/cancel/`/new`.
    pub(super) capturing_plan: bool,
    /// A drafted plan from `/plan`, awaiting `/plan go` to execute it in a fresh
    /// context. Cleared on execute, `/plan stop`, or `/new`.
    pub(super) pending_plan: Option<String>,
    /// History index of the plan reply, framed as the plan card; shifted on
    /// history removal (like `turn_durations`), cleared on execute/discard/`/new`.
    pub(super) plan_card_idx: Option<usize>,
    /// In-process agent for API-key chats (the agent path); `None` until the
    /// first agent turn, rebuilt on key/model switch, cleared on `/new`.
    pub(super) agent_engine: Option<AgentSession>,
    /// `(key_id, cache)` shared across the agent path's per-turn serves so the
    /// learned wire protocol is remembered across turns/launches instead of
    /// re-probed. Seeded from the key's `chat` routes; rebuilt on key switch.
    pub(super) agent_route_cache: Option<(
        String,
        std::sync::Arc<crate::services::route_cache::RouteCache>,
    )>,
    /// Connected MCP servers, shared across engine rebuilds so the servers spawn
    /// once per session. `None` until the first background connect resolves.
    pub(super) mcp_client: Option<std::sync::Arc<crate::agent::mcp::McpClient>>,
    /// A background MCP connect is in flight (don't start a second one).
    pub(super) mcp_connecting: bool,
    /// Per-server interim status while a connect is in flight (server name →
    /// already-mapped `(status, health)`), populated as each server's handshake
    /// resolves so the `/mcp` overlay shows a connected server's real tool count
    /// while slower ones still read "connecting…". Cleared when a new connect
    /// starts; superseded by `mcp_client` once the whole connect lands.
    pub(super) mcp_connect_progress: std::collections::HashMap<String, (String, McpHealth)>,
    /// Bumped whenever the configured server set changes (a `/mcp` toggle). A
    /// background connect carries the generation it started under; a result from
    /// an older generation is dropped, so a connect launched before a toggle can't
    /// resurrect a just-disabled server.
    pub(super) mcp_connect_gen: u64,
    /// MCP tools arrived mid-turn — rebuild the engine to advertise them once the
    /// current turn finishes (rebuild re-seeds from history, so context survives).
    pub(super) mcp_rebuild_pending: bool,
    /// HTTP MCP servers added this session (name → url) to auto-authorize once
    /// their connect reports a 401 — so adding an OAuth server is one step, not a
    /// separate Ctrl+O. Drained when that connect resolves.
    pub(super) pending_mcp_auth: std::collections::HashMap<String, String>,
    /// Per-turn loopback serve backing the agent engine: (accept-loop handle,
    /// shutdown notify). Torn down when the turn finishes or is cancelled.
    pub(super) agent_serve: Option<(
        JoinHandle<anyhow::Result<()>>,
        std::sync::Arc<tokio::sync::Notify>,
    )>,
    /// Pending tool-permission card, while the agent waits for the user's y/n/a.
    pub(super) agent_permission: Option<PendingPermission>,
    /// Session decision on spawning a repo's project `.mcp.json` stdio servers.
    pub(super) project_mcp_consent: ProjectMcpConsent,
    /// Pending consent card for project MCP servers (held back until decided).
    pub(super) pending_mcp_consent: Option<McpConsentPrompt>,
    /// Session-wide auto-approve (Shift+Tab): when on, the agent runs mutating
    /// tools without a permission card. Off by default (safe).
    pub(super) agent_auto_approve: bool,
    /// The same auto-approve state as a shared atomic, so a running agent turn
    /// consults the LIVE toggle rather than a per-turn snapshot: the native
    /// in-process engine reads it on each tool call, and a long-lived
    /// cursor-agent ACP session reads it on each out-of-process
    /// `request_permission`. Kept in lockstep with `agent_auto_approve`.
    pub(super) auto_approve_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether the model reasons before answering. On (default): engine requests
    /// reasoning at the effective effort; off: engine sends the family's "off"
    /// floor, which the loopback bridge maps to `thinking:{type:"disabled"}` for
    /// Anthropic upstreams (so off truly stops reasoning, not just hides it).
    /// Toggled in `/config`, remembered across sessions.
    pub(super) thinking_enabled: bool,
    /// aivo's hosted web_search; `/config` toggle, applied to the engine each turn.
    pub(super) web_search_enabled: bool,
    pub(super) agent_tools_enabled: bool,
    /// Whether the current model is known to support reasoning/thinking (from the
    /// model-limits snapshot). Cached on each model resolve (see
    /// `refresh_context_window`); gates the footer effort badge so it only shows
    /// where thinking can actually appear. Unknown models → false.
    pub(super) model_supports_thinking: bool,
    /// Snapshot vision support, cached on each model resolve. `Some(false)` =
    /// text-only (image sends refused pre-flight); `None` = unknown (let through).
    pub(super) model_image_input: Option<bool>,
    /// Parsed Cursor effort tier for the footer badge (`None` for non-cursor or
    /// bare ids); set in `refresh_context_window`.
    pub(super) cursor_effort_label: Option<String>,
    /// `/effort` reasoning level chosen by the user (None = model default);
    /// applied to the engine on build/change and persisted in chat-prefs.
    pub(super) reasoning_effort: Option<String>,
    /// Valid effort levels for the active model (from `caps.reasoning_efforts`),
    /// refreshed on model/key change. Empty = the model exposes none.
    pub(super) model_reasoning_efforts: Vec<String>,
    /// Messages typed while a turn was in flight, in submit order; each is
    /// auto-sent (one per turn) as the preceding turn finishes — a real FIFO so
    /// a second queued message doesn't silently clobber the first.
    pub(super) queued_messages: Vec<String>,
    /// An in-flight `!cmd` local shell run streaming output into the transcript,
    /// or `None`. Separate from `sending` (model turns) so the two don't entangle.
    pub(super) local_command: Option<LocalCommandRun>,
    /// FULL output of each finished `!cmd`, keyed by the history index of its
    /// `local_command` entry — the source an expanded block renders from (the
    /// transcript and on-disk session keep only a bounded preview). In-memory only,
    /// so after a resume an old block expands to just its persisted preview. Cleared
    /// alongside `expanded_thinking` when history is replaced.
    pub(super) local_outputs: std::collections::HashMap<usize, LocalCommandOutput>,
    /// History indices of `local_command` entries the user expanded inline (full
    /// output shown in place of the folded preview). In-memory only; cleared with
    /// `expanded_thinking`. A toggle bumps `transcript_revision` so the flip repaints.
    pub(super) expanded_output: std::collections::HashSet<usize>,
    /// History indices of assistant turns the user expanded inline (full reasoning
    /// shown in place of the folded summary). In-memory only; cleared when history
    /// is replaced (new chat, resume, rewind). A toggle bumps `transcript_revision`,
    /// the body-cache key, so a flip repaints.
    pub(super) expanded_thinking: std::collections::HashSet<usize>,
    /// Thinking duration (ms) per committed assistant turn, by history index;
    /// drives the folded `▸ thought for Ns` summary. In-memory only, cleared
    /// alongside `expanded_thinking`.
    pub(super) reasoning_durations: std::collections::HashMap<usize, u64>,
    /// Wall time (ms) a finished turn took, by the history index of its last entry;
    /// drives the `✶ Done in …` marker. In-memory only, cleared with `expanded_thinking`.
    pub(super) turn_durations: std::collections::HashMap<usize, u64>,
    /// When the current segment's reasoning started streaming (first reasoning
    /// chunk), for the live `▸ thought for Ns` timer. `None` between segments.
    pub(super) reasoning_started_at: Option<Instant>,
    /// The current segment's thinking duration (ms), frozen when the answer began
    /// streaming so the displayed time excludes answer-streaming. `None` until the
    /// answer starts (the live timer runs from `reasoning_started_at` until then).
    pub(super) reasoning_elapsed_ms: Option<u64>,
    /// In-flight `/skills add` install `(source, started)`; drives the spinner.
    pub(super) installing_skill: Option<(String, Instant)>,
    /// Active share, `None` when not sharing; its presence drives the footer
    /// `● sharing` badge. Stopped on `/share stop`, `/new`, resume, and exit.
    pub(super) live_share: Option<crate::services::share_live::LiveShareHandle>,
    /// True between a start and its `LiveShareReady` event; blocks a second start.
    pub(super) live_share_starting: bool,
    /// `--share` requested but not yet started — `maybe_start_live_share` defers it
    /// until the session settles so it pins the final session id.
    pub(super) live_requested: bool,
}
