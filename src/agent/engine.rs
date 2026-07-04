//! aivo's native in-process agent engine. Holds the conversation, composes
//! OpenAI chat requests, calls the model through the loopback serve (sole network
//! egress), executes tools (permission-gated), compacts on overflow, converges.
//! Rendering/permission go through `AgentUi` (terminal, `--json`, chat TUI).

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};

use crate::agent::ask;
use crate::agent::guards::{self, batch_sig, page_read_key};
use crate::agent::notes;
use crate::agent::plan::{self, PlanItem};
use crate::agent::protocol::{AssistantMessage, ChatRequest, Decision, ToolCall, ToolSpec};
use crate::agent::request::{assistant_to_openai, role, tool_to_openai};
use crate::agent::retry::{
    error_is_retryable, is_context_overflow_error, resolve_max_steps, retry_delay,
};
use crate::agent::secrets_guard;
use crate::agent::skills::{self, Skill};
use crate::agent::subagents::{self, Subagent};
use crate::agent::system_prompt::system_prompt;
use crate::agent::tokens::{content_to_parts, estimate_tokens, usage_tokens};
use crate::agent::{serve_client, tool_repair, tools, verify};
use crate::services::serve_router::extract_usage_from_value;
use crate::services::session_store::SessionTokens;

/// Stop a turn after this many identical consecutive tool-call batches (weak-model loop).
const REPEAT_LIMIT: usize = 3;
/// Per-turn cap on plain-text-markup nudges; after this the turn converges.
const MAX_LEAKED_NUDGES: usize = 2;
const LEAKED_TOOL_CALL_NUDGE: &str = "Your last reply wrote tool calls as plain text, so nothing ran. To call a tool, emit it through the structured tool-call API — not as message text.";
/// Stands in for an all-markup assistant turn (non-empty, keeps alternation).
const LEAKED_TOOL_CALL_PLACEHOLDER: &str =
    "(I wrote a tool call as text by mistake; reissuing it properly.)";
/// Auto-retry budget for transient LLM/network failures.
const MAX_RETRIES: usize = 3;
/// Step cap for a `subagent` run — below the top-level budget so a delegated subtask can't run away.
const SUBAGENT_MAX_STEPS: u32 = 20;
/// Max sub-agents run concurrently when the model fans out several in one batch — a
/// ceiling on parallel sub-engines (each is a full model loop) so a wide fan-out
/// doesn't stampede the provider.
const SUBAGENT_PARALLEL_CAP: usize = 4;
/// Cap on completion-gate re-nudges (unattended `-e`), so a stubborn model can't loop.
const MAX_COMPLETION_NUDGES: usize = 2;
/// Cap on self-correct verify→fix rounds, so a stubborn failure can't loop the run.
const MAX_SELFCORRECT_ATTEMPTS: usize = 3;
const VERIFY_FAILED_PREFIX: &str = "The project's checks are failing, so the task isn't done. \
Fix the cause and continue — don't stop until they pass:";
const COMPLETION_NUDGE: &str = "That may not be finished. If the task is genuinely complete, \
briefly confirm what you did and verified, then stop. Otherwise keep going — don't stop until \
it's done or you're truly blocked (then say exactly what's blocking you).";

/// Compaction window assumed when the model's real one is unknown (0); without it
/// such models never compact and resend the whole transcript. A real window wins.
pub(crate) const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

/// Cap on force-compact-and-retry attempts per step after an input-overflow rejection.
const MAX_FORCED_COMPACTIONS: usize = 3;

/// Ack when a sandbox-blocked `run_bash` is approved to re-run unconfined; cleared
/// on the next agent output so it isn't pinned all turn.
pub const SANDBOX_ESCALATION_NOTICE: &str = "re-running outside the workspace sandbox (approved)";

/// One-line diagnostic to stderr, gated by `AIVO_DEBUG=1`.
fn agent_debug(msg: &str) {
    if matches!(std::env::var("AIVO_DEBUG").as_deref(), Ok("1")) {
        eprintln!("aivo[agent]: {msg}");
    }
}

/// Cap on the tracked touched-files list (most-recent kept).
const MAX_TOUCHED_FILES: usize = 200;
/// Cap on the agent's durable scratchpad (most-recent kept).
const MAX_NOTES: usize = 50;

/// Side-effects the engine delegates: rendering and the permission prompt.
/// `ask_permission` fires only for mutating tools that aren't pre-approved; a
/// non-TTY impl must fail closed (Deny). `Send` so the chat TUI can drive it on a task.
pub trait AgentUi: Send {
    /// Before each LLM turn (before any text) — for a "thinking…" indicator. Default no-op.
    fn turn_start(&mut self) {}
    /// Live context-window fill for the in-flight turn. `measured` true = a
    /// provider-reported step total (exact); false = a chars/4 estimate of the
    /// about-to-send request, emitted before usage is known. Default no-op.
    fn context_usage(&mut self, _tokens: u64, _measured: bool) {}
    /// Turn's cumulative output tokens so far (live per-turn counter). Default no-op.
    fn turn_tokens(&mut self, _output: u64) {}
    /// A delegated sub-agent began a step — surfaced on the parent's status line
    /// (label-only) so a long delegation isn't a frozen label. `tool` empty =
    /// thinking between calls; `step` = child's 1-based turn. Default no-op.
    fn subagent_activity(&mut self, _agent: &str, _tool: &str, _args: &Value, _step: usize) {}
    /// Prompt for the next REPL turn. `None` ends the session (EOF / `/exit`);
    /// default `None` → one-shot only.
    fn read_user_input(&mut self) -> Option<String> {
        None
    }
    fn assistant_text(&mut self, delta: &str);
    /// A streamed reasoning/thinking delta (separate from the visible reply). Default no-op.
    fn assistant_reasoning(&mut self, _delta: &str) {}
    /// Drop the just-streamed segment — it was a tool call written as text (stripped + retried). Default no-op.
    fn discard_streamed_segment(&mut self) {}
    /// The agent set/updated its plan via `update_plan`; rendered as a checklist card. Default no-op.
    fn plan_updated(&mut self, _items: &[PlanItem]) {}
    fn tool_start(&mut self, name: &str, args: &Value);
    fn tool_result(&mut self, name: &str, result: &Result<String, String>);
    fn notify(&mut self, text: &str);
    /// Like [`notify`](Self::notify) but for a genuine error (error hue). Default delegates to `notify`.
    fn notify_error(&mut self, text: &str) {
        self.notify(text);
    }
    /// End-of-turn line: optional summary + stats. `tokens` = cumulative work (prompt
    /// re-counted each step); `context_tokens` = last step's prompt+completion (real fill, 0 if no usage).
    fn footer(
        &mut self,
        summary: Option<&str>,
        steps: usize,
        tokens: u64,
        context_tokens: u64,
        elapsed_secs: u64,
    );
    /// Decide whether a mutating tool may run. Async so a TUI can await a permission
    /// card; the terminal impl resolves synchronously. Must fail closed off a TTY.
    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        preview: Option<&'a str>,
    ) -> BoxFuture<'a, Decision>;
    /// The `switch_model` tool. Default declines — only the chat TUI drives it.
    fn switch_chat_model<'a>(
        &'a mut self,
        _model: &'a str,
    ) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async {
            Err("Switching model is only available in interactive `aivo code`.".to_string())
        })
    }
    /// The `set_effort` tool. Default declines — only the chat TUI drives it.
    fn set_chat_effort<'a>(&'a mut self, _level: &'a str) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async {
            Err(
                "Changing reasoning effort is only available in interactive `aivo code`."
                    .to_string(),
            )
        })
    }
    /// The `ask_user` tool: pose a question with selectable options and return the
    /// user's answer as the tool result. Async so the chat TUI can await the card.
    /// Default declines — headless/sub-agents have no interactive user.
    fn ask_user<'a>(
        &'a mut self,
        _question: &'a str,
        _options: &'a [crate::agent::ask::AskOption],
        _allow_free_text: bool,
    ) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async {
            Err(
                "Asking the user a question is only available in interactive `aivo code`."
                    .to_string(),
            )
        })
    }
}

/// Extra tools beyond the built-ins — currently MCP servers. The engine advertises
/// `specs()` and routes any call it `handles()` to `call()`. Abstract to keep the
/// engine free of process/transport knowledge; `Send + Sync` so it can be shared.
pub trait ExternalTools: Send + Sync {
    /// OpenAI tool schemas to advertise (already `mcp__server__tool`-named).
    fn specs(&self) -> Vec<Value>;
    /// Whether this source owns `name` (routed here, not to the built-in executor).
    fn handles(&self, name: &str) -> bool;
    /// Whether a call to `name` is permission-gated (e.g. an untrusted MCP server).
    /// Default `false` — configured sources are trusted.
    fn requires_approval(&self, _name: &str) -> bool {
        false
    }
    /// Execute one tool call; the result string is fed back as the tool result (Err continues the loop).
    fn call<'a>(&'a self, name: &'a str, args: &'a Value) -> BoxFuture<'a, Result<String, String>>;
}

/// Per-turn I/O: the loopback serve to reach the provider and the working dir tools
/// run against. (Model, history, limits are owned by the engine.)
pub struct TurnCtx<'a> {
    pub client: &'a reqwest::Client,
    pub serve_base: &'a str,
    pub auth: Option<&'a str>,
    pub cwd: &'a Path,
    /// Auto-approve every mutating tool: the static CLI `-y` flag.
    pub yes: bool,
    /// Live auto-approve toggle (the chat TUI's Shift+Tab/Ctrl+O). Read fresh per
    /// tool call so flipping it mid-turn takes effect on the *running* turn,
    /// unlike the `yes` snapshot. `None` outside the chat TUI.
    pub auto_approve: Option<&'a std::sync::atomic::AtomicBool>,
}

impl TurnCtx<'_> {
    /// True when mutating tools run without a prompt — the `-y` flag or the live chat toggle.
    pub fn auto_approve_enabled(&self) -> bool {
        self.yes
            || self
                .auto_approve
                .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// A `/rewind` turn boundary. `msg_index` = the turn's opening user message
/// (truncation point), kept valid across compaction via `rebase_checkpoints`.
/// `tree` = working-tree snapshot at turn start (`None` = conversation-only).
/// `prompt` = opening user text stored verbatim (the picker matches on it, since
/// `messages[msg_index]` gets mutated in place). `changed` = paths the turn modified
/// (a rewind reverts only their union); `None` until recorded / for interrupted turns.
#[derive(Clone)]
pub(crate) struct Checkpoint {
    pub(crate) msg_index: usize,
    prompt: String,
    tree: Option<String>,
    changed: Option<Vec<std::path::PathBuf>>,
}

/// Result of a [`AgentEngine::rewind_to`] — counts for the notice.
#[derive(Default)]
pub struct RewindOutcome {
    /// Files rewritten/recreated to match the snapshot.
    pub restored: usize,
    /// Files removed (created since / the new side of a rename).
    pub deleted: usize,
    /// A git failure during restore (the conversation still rewound).
    pub error: Option<String>,
}

/// The agent's brain: system prompt + conversation + decision/convergence logic.
/// No rendering or direct provider knowledge — those flow through `TurnCtx`/`AgentUi`.
pub struct AgentEngine {
    pub(crate) model: String,
    tools_openai: Vec<Value>,
    pub(crate) messages: Vec<Value>,
    pub(crate) context_window: u32,
    /// Multiplier (>= 1.0) correcting the chars/4 [`estimate_tokens`] undershoot toward
    /// the real tokenizer, learned from measured usage; starts at 1.0.
    pub(crate) token_calibration: f64,
    max_steps: usize,
    /// Per-turn completion-token cap (0 = none) — backstop for unattended `-e` runs.
    max_output_tokens: u64,
    /// "Always"-approved actions. Scoped grants (exact / command-prefix / dir / tool);
    /// session-only unless a grant is safe to persist (see [`crate::agent::grant_store`]).
    /// A durable "always allow `rm …`" is a footgun, so dangerous acts stay exact+session.
    grants: crate::agent::grant_store::GrantStore,
    /// Discovered SKILL.md skills, loaded on demand via the `skill` tool.
    skills: Vec<Skill>,
    /// Named specialist sub-agents (top-level engine only). The `subagent` tool's
    /// `agent` field selects one; `run_subagent` applies its model/instructions/scope.
    subagents: Vec<Subagent>,
    /// Kept so `subagent` can build a sub-engine with the same identity (date + guides).
    date: String,
    guides: Vec<String>,
    /// Extra tools beyond the built-ins (MCP servers), if any are configured.
    external: Option<std::sync::Arc<dyn ExternalTools>>,
    /// Body of the last compaction summary (no prefix). Fed back to the summarizer
    /// next compaction so facts carry forward instead of being re-compressed lossily.
    pub(crate) last_summary: Option<String>,
    /// Latest `update_plan` plan. Pinned into every compaction fold, verbatim.
    pub(crate) plan: Vec<PlanItem>,
    /// Files touched this session (insertion order, deduped, capped). Maintained
    /// incrementally so it survives summarization; pinned into every compaction.
    pub(crate) touched_files: Vec<String>,
    /// Durable scratchpad: `take_note` entries. Pinned verbatim into compaction and
    /// rebuilt from the log on resume, so they outlive turns/summaries. Capped at [`MAX_NOTES`].
    pub(crate) notes: Vec<String>,
    /// Provider-measured token split (prompt/completion/cache) for the LAST turn,
    /// summed across steps. The chat TUI drains it (`take_turn_usage`) for `aivo stats`. Reset per turn.
    turn_usage: SessionTokens,
    /// `/rewind`: one checkpoint per `run_turn`, in order. The chat TUI maps display
    /// turns by matching prompt text newest-backward (robust to trim/compaction/rebuild,
    /// which a positional index isn't). In-memory; tree objects live in `checkpoint_store`.
    pub(crate) checkpoints: Vec<Checkpoint>,
    /// Tree-level snapshot/restore via a shadow git store. `None` until `/rewind` is
    /// enabled (top-level chat only). See [`crate::agent::checkpoint`].
    checkpoint_store: Option<crate::agent::checkpoint::CheckpointStore>,
    /// `reasoning_effort` for a reasoning-capable model, else `None` (the field 400s
    /// some providers). Defaults from the snapshot; changed live by `/effort`.
    reasoning_effort: Option<String>,
    /// Catalog-advertised effort levels (set per turn); used to pick a valid "off". See `thinking_request`.
    reasoning_efforts: Vec<String>,
    /// Whether the model is asked to think this turn. Off makes [`Self::thinking_request`]
    /// emit a disable signal. Set per turn from the `/config` toggle.
    thinking_enabled: bool,
    /// `/config` toggle for aivo's hosted web_search (the local tool); native search untouched.
    use_web_search_enabled: bool,
    /// `/config` master switch; off → plain chat (no tools, no system prompt).
    agent_tools_enabled: bool,
    /// Whether this model can reason at all (snapshot). Cached at construction so the
    /// disable path doesn't send an effort field that would 400.
    reasoning_capable: bool,
    /// Plan mode: mutating tools refused so a `/plan` investigation can't modify the
    /// workspace. See `restrict_read_only`.
    read_only: bool,
    /// Unattended `-e` only: reject a text turn that admits it isn't done (or trails
    /// off mid-step) and nudge to continue, rather than accept it as the final answer.
    require_completion: bool,
    /// Opt-in (`AIVO_AGENT_SELF_CORRECT`): when the agent declares done, run the
    /// project's validator and, on failure, feed it back so it fixes the cause.
    self_correct: bool,
    /// Interactive chat only (off for headless/sub-agents): see [`CONFIRM_BEFORE_BUILD`].
    confirm_before_build: bool,
    /// First-party branding (aivo-starter): present as aivo, not the upstream model.
    first_party: bool,
    /// Interactive chat only: the `switch_model`/`set_effort` tools are advertised.
    session_controls: bool,
    /// `(system, tools)` prefix fingerprint from the last turn; checked under `AIVO_DEBUG`.
    prefix_fp: Option<(u64, u64)>,
    /// File-staleness guard: baselines of files read this session, so a mutating tool
    /// can be refused when its target changed on disk since the model last read it.
    file_tracker: crate::agent::file_tracker::FileTracker,
    /// Opt-in LSP diagnostics-after-edit (`AIVO_AGENT_LSP=1`); `None` = disabled.
    lsp: Option<crate::agent::lsp::LspManager>,
}

/// Live `aivo code` session facts injected into the system prompt so the agent can answer
/// "what model am I on?" and knows how to change model/effort/key. `model_label` is the
/// user-facing `raw_model` (safe under first-party branding); `provider_label` is the key name.
pub struct ChatSessionContext {
    pub model_label: String,
    pub provider_label: String,
    pub effort: Option<String>,
    pub effort_levels: Vec<String>,
}

/// Default reasoning-effort level: `AIVO_AGENT_REASONING_EFFORT` or `"medium"`.
/// Whether it's requested depends on model capability (see [`default_reasoning_effort`]).
pub fn default_reasoning_effort_level() -> String {
    std::env::var("AIVO_AGENT_REASONING_EFFORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "medium".to_string())
}

/// `reasoning_effort` for `model`, or `None` for non-reasoning models (the field
/// would 400 strict providers). Capability from the snapshot; level from [`default_reasoning_effort_level`].
fn default_reasoning_effort(model: &str) -> Option<String> {
    crate::services::model_metadata::snapshot_limits(model)
        .is_some_and(|c| c.reasoning)
        .then(default_reasoning_effort_level)
}

impl AgentEngine {
    /// Seed an engine with the identity system prompt. `guides` = names of project
    /// convention files in cwd (read on demand, not injected). `context_window`
    /// (0 = unknown → [`DEFAULT_CONTEXT_WINDOW`]) honors an env override; `max_steps`
    /// is the per-turn step budget (0 = no cap).
    pub fn new(
        cwd: &str,
        model: &str,
        date: &str,
        guides: &[String],
        skills: &[Skill],
        context_window: u32,
        max_steps: u32,
    ) -> Self {
        // Env override so compaction can be exercised without a small-context model.
        let context_window = std::env::var("AIVO_AGENT_CONTEXT_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(context_window);
        let max_steps = resolve_max_steps(max_steps);
        let mut specs = tools::tool_specs_for(model);
        if !skills.is_empty() {
            specs.push(skills::skill_tool_spec(skills));
        }
        specs.push(plan::plan_tool_spec());
        specs.push(notes::note_tool_spec());
        specs.push(subagent_tool_spec(&[]));
        let mut tools_openai: Vec<Value> = specs.into_iter().map(tool_to_openai).collect();
        // Native-search providers get the server tool instead of the local one (mutually exclusive).
        if tools::native_web_search_enabled(model) {
            tools_openai.retain(|t| t["function"]["name"].as_str() != Some("web_search"));
            tools_openai.push(json!({ "type": "web_search" }));
        }
        let messages = vec![json!({
            "role": "system",
            "content": system_prompt(cwd, date, guides, skills),
        })];
        Self {
            model: model.to_string(),
            tools_openai,
            messages,
            context_window,
            token_calibration: 1.0,
            max_steps,
            max_output_tokens: 0,
            grants: crate::agent::grant_store::GrantStore::default(),
            skills: skills.to_vec(),
            subagents: Vec::new(),
            date: date.to_string(),
            guides: guides.to_vec(),
            external: None,
            last_summary: None,
            plan: Vec::new(),
            touched_files: Vec::new(),
            notes: Vec::new(),
            turn_usage: SessionTokens::default(),
            checkpoints: Vec::new(),
            checkpoint_store: None,
            reasoning_effort: default_reasoning_effort(model),
            reasoning_efforts: Vec::new(),
            thinking_enabled: true,
            use_web_search_enabled: true,
            agent_tools_enabled: true,
            reasoning_capable: default_reasoning_effort(model).is_some(),
            read_only: false,
            require_completion: false,
            self_correct: false,
            confirm_before_build: false,
            first_party: false,
            session_controls: false,
            prefix_fp: None,
            file_tracker: crate::agent::file_tracker::FileTracker::default(),
            lsp: None,
        }
    }

    /// Cap per-turn completion tokens (0 = no cap).
    pub fn set_output_budget(&mut self, tokens: u64) {
        self.max_output_tokens = tokens;
    }

    /// Back the "always allow" grants with a persistent store at `<config>/grants.json`,
    /// loading any grants already saved there. Without this the store is session-only.
    pub fn set_grants_path(&mut self, config_dir: &Path) {
        self.grants = crate::agent::grant_store::GrantStore::load(config_dir.join("grants.json"));
    }

    /// Enable LSP diagnostics-after-edit rooted at `cwd` when `AIVO_AGENT_LSP` is set —
    /// after a successful edit, the language server's native errors are fed back.
    pub fn maybe_enable_lsp(&mut self, cwd: &Path) {
        if std::env::var("AIVO_AGENT_LSP").is_ok_and(|v| v != "0" && !v.is_empty()) {
            let mgr = crate::agent::lsp::LspManager::new(cwd);
            mgr.warm(); // start indexing now so the first edit's check isn't cold
            self.lsp = Some(mgr);
        }
    }

    /// Append [`FIRST_PARTY_IDENTITY`] to the system prompt in place — keeps the
    /// single-system-message invariant `restore_conversation` relies on. Idempotent.
    pub fn set_first_party(&mut self) {
        if self.first_party {
            return;
        }
        self.first_party = true;
        let Some(content) = self.messages.first_mut().and_then(|m| m.get_mut("content")) else {
            return;
        };
        if let Some(s) = content.as_str() {
            *content = Value::String(format!("{s}\n\n{FIRST_PARTY_IDENTITY}"));
        }
    }

    /// Append [`CONFIRM_BEFORE_BUILD`] to the system prompt in place, like
    /// [`Self::set_first_party`] (single-system-message invariant). Idempotent.
    pub fn set_confirm_before_build(&mut self) {
        if self.confirm_before_build {
            return;
        }
        self.confirm_before_build = true;
        let Some(content) = self.messages.first_mut().and_then(|m| m.get_mut("content")) else {
            return;
        };
        if let Some(s) = content.as_str() {
            *content = Value::String(format!("{s}\n\n{CONFIRM_BEFORE_BUILD}"));
        }
    }

    /// Interactive chat only: append the live session facts + switch guidance to the system
    /// prompt (in place, like [`Self::set_first_party`]) and advertise `switch_model`/`set_effort`.
    pub fn set_chat_session_context(&mut self, ctx: ChatSessionContext) {
        if self.session_controls {
            return;
        }
        self.session_controls = true;
        self.tools_openai
            .push(tool_to_openai(switch_model_tool_spec()));
        self.tools_openai
            .push(tool_to_openai(set_effort_tool_spec()));
        self.tools_openai
            .push(tool_to_openai(crate::agent::ask::ask_user_tool_spec()));
        let effort_clause = match &ctx.effort {
            Some(e) => format!(", reasoning effort: `{e}`"),
            None => String::new(),
        };
        let levels_clause = if ctx.effort_levels.is_empty() {
            "This model exposes no reasoning-effort levels.".to_string()
        } else {
            format!(
                "This model's effort levels: {}.",
                ctx.effort_levels.join(", ")
            )
        };
        let block = format!(
            "This is an interactive `aivo code` session (not a plain shell). Live setup — model: \
`{model}`, provider: `{provider}`{effort_clause}. When the user asks what model, provider, or \
effort they're on, answer from these facts directly. The user can change them live with the \
slash commands `/model [name]`, `/key [name]` (switches provider/key — starts a fresh chat), \
and `/effort [level]`. You can change the model or reasoning effort YOURSELF when the user asks: \
call `switch_model` (pass the model id or a distinctive part) or `set_effort` — don't tell the \
user you're unable to switch. For a key/provider change, tell them to run `/key` (it starts a new \
chat, so you shouldn't do it for them). {levels_clause} When you need a decision from the user and \
the answer is one of a few options — a yes/no, a this-or-that, approving a plan — call `ask_user` \
with those options so they can pick, instead of ending your turn with a plain-text question; the \
answer returns as the tool result and you continue. Ask in prose only for genuinely open-ended \
questions.",
            model = ctx.model_label,
            provider = ctx.provider_label,
        );
        let Some(content) = self.messages.first_mut().and_then(|m| m.get_mut("content")) else {
            return;
        };
        if let Some(s) = content.as_str() {
            *content = Value::String(format!("{s}\n\n{block}"));
        }
    }

    /// Read-only mode for `/plan`: hide mutating tools + `subagent`. The execution
    /// guard also refuses them (in case one is hallucinated). One-way.
    pub fn restrict_read_only(&mut self) {
        self.read_only = true;
        self.tools_openai.retain(|t| {
            let name = t["function"]["name"].as_str().unwrap_or("");
            !tools::is_mutating(name) && name != "subagent"
        });
    }

    /// Enable the headless completion gate (unattended `-e`): a text-only turn that
    /// admits it isn't done, or trails off mid-step, is nudged to continue (bounded)
    /// instead of being accepted as the final answer. Off for interactive/sub-agents.
    pub fn set_require_completion(&mut self) {
        self.require_completion = true;
    }

    /// Enable post-edit self-verification (opt-in): on a declared-done turn, run the
    /// project's validator and feed failures back so the model fixes them. See [`verify`].
    pub fn set_self_correct(&mut self) {
        self.self_correct = true;
    }

    /// Set the `reasoning_effort` level (`/effort`). Only meaningful for reasoning models.
    pub fn set_reasoning_effort(&mut self, effort: String) {
        self.reasoning_effort = Some(effort);
    }

    /// Turn thinking on/off for upcoming turns (`/config`). Off makes [`Self::thinking_request`] emit a disable signal.
    pub fn set_thinking_enabled(&mut self, on: bool) {
        self.thinking_enabled = on;
    }

    /// `/config` toggle: add/remove the local hosted `web_search` tool. Idempotent;
    /// a native-search model (which carries the server tool instead) is untouched.
    pub fn set_web_search_enabled(&mut self, on: bool) {
        self.use_web_search_enabled = on;
        if tools::native_web_search_enabled(&self.model) {
            return; // native models don't carry the local tool
        }
        let is_web_search = |t: &Value| t["function"]["name"].as_str() == Some("web_search");
        let has = self.tools_openai.iter().any(is_web_search);
        if on && !has {
            if let Some(s) = tools::tool_specs()
                .into_iter()
                .find(|s| s.name == "web_search")
            {
                self.tools_openai.push(tool_to_openai(s));
            }
        } else if !on && has {
            self.tools_openai.retain(|t| !is_web_search(t));
        }
    }

    pub fn set_agent_tools_enabled(&mut self, on: bool) {
        self.agent_tools_enabled = on;
    }

    /// Set the catalog-advertised effort levels for this turn. See `reasoning_efforts`.
    pub fn set_reasoning_efforts(&mut self, efforts: Vec<String>) {
        self.reasoning_efforts = efforts;
    }

    /// Whether `level` is one the model's catalog advertises (so it won't 400).
    fn effort_is_valid(&self, level: &str) -> bool {
        self.reasoning_efforts.iter().any(|e| e == level)
    }

    /// Thinking control for this step: `(reasoning_effort, emit_thinking_disabled)`.
    /// Enabled → resolved level. Disabled → the lowest "off" the catalog advertises
    /// (gpt-5 diverged: 5.0 `minimal`, 5.1+/5.4 `none`, codex `low` — a guess 400s);
    /// a depth-only scale with no off (aivo/starter, Anthropic) → `thinking:{type:"disabled"}`.
    fn thinking_request(&self) -> (Option<&str>, bool) {
        if self.thinking_enabled {
            return (self.reasoning_effort.as_deref(), false);
        }
        let capable = self.reasoning_capable
            || self.reasoning_effort.is_some()
            || !self.reasoning_efforts.is_empty();
        if !capable {
            return (None, false);
        }
        let lower = self.model.to_ascii_lowercase();
        let name = lower.rsplit('/').next().unwrap_or(&lower);
        if self.effort_is_valid("none") {
            (Some("none"), false)
        } else if self.effort_is_valid("minimal") {
            (Some("minimal"), false)
        } else if name.starts_with("o1") || name.starts_with("o3") || name.starts_with("o4") {
            (Some("low"), false)
        } else if name.starts_with("gpt-5") || name.contains("codex") {
            // codex floor is low (no off); snapshot-absent gpt-5.0 → minimal.
            if self.effort_is_valid("low") {
                (Some("low"), false)
            } else {
                (Some("minimal"), false)
            }
        } else {
            (None, true)
        }
    }

    /// Enable `/rewind` tree-checkpointing (top-level chat only, to avoid the git cost). Idempotent.
    pub fn enable_rewind_checkpoints(&mut self, cwd: &str) {
        if self.checkpoint_store.is_none() {
            self.checkpoint_store = Some(crate::agent::checkpoint::CheckpointStore::new(
                std::path::Path::new(cwd),
            ));
        }
    }

    /// Drain the last turn's provider-measured token split (zeroing the accumulator);
    /// the chat TUI folds it into the chat session index for `aivo stats`.
    pub fn take_turn_usage(&mut self) -> SessionTokens {
        std::mem::take(&mut self.turn_usage)
    }

    /// Attach an external tool source (MCP): advertise its schemas alongside the
    /// built-ins and route its calls to it. Call once, after construction.
    pub fn set_external_tools(&mut self, ext: std::sync::Arc<dyn ExternalTools>) {
        self.tools_openai.extend(ext.specs());
        self.external = Some(ext);
    }

    /// Fill in the compaction context window if unknown (0) at construction (a
    /// catalog-warmed model resolves it after the engine is built). Only fills a
    /// missing window — never overrides a known one.
    pub fn set_context_window(&mut self, window: u32) {
        if self.context_window == 0 && window > 0 {
            agent_debug(&format!(
                "context window resolved at model lookup: {window} (was unknown)"
            ));
            self.context_window = window;
        } else if window > 0 && window != self.context_window {
            // Keep the known window, but surface drift so a wrong one can't mis-size compaction.
            agent_debug(&format!(
                "context window drift: budgeting {} (assumed) but model lookup reports {window} (served)",
                self.context_window
            ));
        }
    }

    /// Register named specialist sub-agents (top-level engine only): swap the
    /// generic `subagent` tool for one enumerating them in `agent`, and advertise
    /// each in the system prompt (progressive disclosure). No-op when empty.
    pub fn set_subagents(&mut self, subagents: &[Subagent]) {
        if subagents.is_empty() {
            return;
        }
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("subagent"));
        self.tools_openai
            .push(tool_to_openai(subagent_tool_spec(subagents)));
        let section = subagents::subagents_prompt_section(subagents);
        if !section.is_empty()
            && let Some(sys) = self.messages.first_mut()
        {
            let cur = sys
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            sys["content"] = json!(format!("{cur}{section}"));
        }
        self.subagents = subagents.to_vec();
    }

    /// Apply a named agent profile: fold its instructions into the system prompt
    /// and, if it authored a `tools` scope, restrict the offered tools to that
    /// allow-list (any unlisted tool, incl. MCP, is dropped; an empty resolution
    /// doesn't scope). Applied to a delegated sub-agent's fresh sub-engine.
    pub fn apply_profile(&mut self, sa: &Subagent) {
        if !sa.body.is_empty()
            && let Some(sys) = self.messages.first_mut()
        {
            let cur = sys
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            sys["content"] = json!(format!("{cur}\n\n## Your role: {}\n{}", sa.name, sa.body));
        }
        if let Some(allowed) = sa.resolved_tools() {
            // Edit tools are one equivalence class: authoring any grants whichever
            // the model advertises (apply_patch on GPT-5/Codex, else edit_file/multi_edit).
            let editor_allowed = allowed.contains(&"edit_file")
                || allowed.contains(&"multi_edit")
                || allowed.contains(&"apply_patch");
            self.tools_openai.retain(|t| {
                let name = t["function"]["name"].as_str().unwrap_or("");
                let is_editor = matches!(name, "edit_file" | "multi_edit" | "apply_patch");
                // update_plan/take_note have no side effects, so a scoped specialist always keeps them.
                name == "update_plan"
                    || name == "take_note"
                    || allowed.contains(&name)
                    || (is_editor && editor_allowed)
            });
        }
    }

    /// Remove the `subagent` tool — used on a sub-engine so it can't spawn sub-agents (depth-1 only).
    fn drop_subagent_tool(&mut self) {
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("subagent"));
    }

    /// `/clear`: drop the conversation, keep the system prompt. Also clears the
    /// compaction working set, else a cleared session would re-inject stale facts.
    pub fn reset(&mut self) {
        self.messages.truncate(1);
        self.last_summary = None;
        self.plan.clear();
        self.touched_files.clear();
        self.notes.clear();
        // `/rewind` checkpoints' `msg_index` pointed into the cleared transcript.
        self.checkpoints.clear();
    }

    /// Seed prior conversation into a fresh engine (resume / mid-chat switch) so it
    /// isn't amnesiac. Only user/assistant text turns carry (tool steps lack call IDs).
    /// No-op once a turn has run.
    pub fn seed_history(&mut self, turns: impl IntoIterator<Item = (String, String)>) {
        let mut seen_user = false;
        for (role, content) in turns {
            if !matches!(role.as_str(), "user" | "assistant") {
                continue;
            }
            // Must open with a user turn — Anthropic rejects assistant-first; drop leading assistants.
            if !seen_user {
                if role != "user" {
                    continue;
                }
                seen_user = true;
            }
            self.push_text_turn(&role, content);
        }
    }

    /// Export the conversation after the system prompt as raw OpenAI messages
    /// (tool_calls/results with ids intact) for persistence. The system prompt is
    /// omitted — rebuilt fresh on restore. Empty before any turn has run.
    pub fn export_conversation(&self) -> Vec<Value> {
        self.messages.iter().skip(1).cloned().collect()
    }

    /// Restore an [`export_conversation`]ed transcript into a fresh engine (resume),
    /// appended after the system prompt verbatim. No-op unless fresh — never after a
    /// turn or `seed_history`. `run_turn`'s `repair_interrupted_tail` heals a mid-tool tail.
    pub fn restore_conversation(&mut self, conversation: Vec<Value>) {
        if self.messages.len() != 1 {
            return;
        }
        // These turns predate this engine: no `checkpoints` entry, so the back-match marks them conversation-only.
        self.messages.extend(conversation);
        self.rebuild_working_set_from_log();
    }

    /// Re-derive the working set (plan, notes, touched files) from the restored log
    /// so a resumed session isn't amnesiac — the stateless-reducer property (log is
    /// the source of truth). Calls folded into a summary live on as text, so nothing
    /// visible is lost. Only meaningful right after restore.
    fn rebuild_working_set_from_log(&mut self) {
        // Collect first (immutable borrow), then apply — `record_touched_file` borrows mut.
        let calls: Vec<(String, Value)> = self
            .messages
            .iter()
            .filter(|m| role(m) == "assistant")
            .filter_map(|m| m.get("tool_calls").and_then(|c| c.as_array()))
            .flatten()
            .filter_map(|call| {
                let name = call.pointer("/function/name").and_then(|v| v.as_str())?;
                let args = call
                    .pointer("/function/arguments")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or(Value::Null);
                Some((name.to_string(), args))
            })
            .collect();
        for (name, args) in calls {
            match name.as_str() {
                "read_file" | "write_file" | "edit_file" | "multi_edit" => {
                    self.record_touched_file(&name, &args);
                }
                "update_plan" => {
                    if let Ok(mut items) = plan::parse_plan(&args) {
                        plan::normalize_progress(&mut items);
                        self.plan = items;
                    }
                }
                "take_note" => {
                    if let Ok(note) = notes::parse_note(&args) {
                        if self.notes.len() >= MAX_NOTES {
                            self.notes.remove(0);
                        }
                        self.notes.push(note);
                    }
                }
                _ => {}
            }
        }
    }

    /// Append a user/assistant text turn, MERGING into the previous message when it
    /// has the same role and is plain text. The engine must never hold two
    /// consecutive same-role messages — Anthropic (via the bridge) 400s on
    /// non-alternating roles (non-retryable brick).
    /// Append the opening user turn (plain string or multimodal array), folding into a
    /// trailing user turn so two consecutive `user` messages never occur.
    fn push_user_content(&mut self, content: Value) {
        if let Value::String(s) = content {
            self.push_text_turn("user", s);
            return;
        }
        if let Some(last) = self.messages.last_mut()
            && last.get("role").and_then(|r| r.as_str()) == Some("user")
            && last.get("tool_calls").is_none()
        {
            let mut parts = content_to_parts(last["content"].take());
            parts.extend(content_to_parts(content));
            last["content"] = Value::Array(parts);
            return;
        }
        self.messages
            .push(json!({"role": "user", "content": content}));
    }

    fn push_text_turn(&mut self, role: &str, content: String) {
        if let Some(last) = self.messages.last_mut()
            && last.get("role").and_then(|r| r.as_str()) == Some(role)
            && last.get("content").and_then(|c| c.as_str()).is_some()
            && last.get("tool_calls").is_none()
        {
            let prev = last["content"].as_str().unwrap_or("");
            last["content"] = if prev.is_empty() {
                json!(content)
            } else {
                json!(format!("{prev}\n\n{content}"))
            };
            return;
        }
        self.messages
            .push(json!({"role": role, "content": content}));
    }

    /// Restore the assistant↔tool invariant before a new turn. A turn torn down
    /// mid-tool (Esc/interrupt) can leave an `assistant` with `tool_calls` whose
    /// results were never pushed; appending a `user` then 400s every provider
    /// (non-retryable → the corrupted prefix re-sends every turn, bricking the
    /// session). Synthesize an `[interrupted]` result per unanswered call id.
    fn repair_interrupted_tail(&mut self) {
        let Some(idx) = self.messages.iter().rposition(|m| {
            role(m) == "assistant"
                && m.get("tool_calls")
                    .and_then(|t| t.as_array())
                    .is_some_and(|a| !a.is_empty())
        }) else {
            return;
        };
        let call_ids: Vec<String> = self.messages[idx]["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        // Tool results sit immediately after the call — answers live in that contiguous run.
        let answered: HashSet<&str> = self.messages[idx + 1..]
            .iter()
            .take_while(|m| role(m) == "tool")
            .filter_map(|m| m.get("tool_call_id").and_then(|v| v.as_str()))
            .collect();
        let missing: Vec<Value> = call_ids
            .iter()
            .filter(|id| !answered.contains(id.as_str()))
            .map(|id| json!({"role": "tool", "tool_call_id": id, "content": "[interrupted]"}))
            .collect();
        if missing.is_empty() {
            return;
        }
        let missing_count = missing.len();
        let insert_at = idx
            + 1
            + self.messages[idx + 1..]
                .iter()
                .take_while(|m| role(m) == "tool")
                .count();
        for (offset, msg) in missing.into_iter().enumerate() {
            self.messages.insert(insert_at + offset, msg);
        }
        // The bridge maps each tool result to a `user` message, so [tool_result, next user]
        // becomes two consecutive users (Anthropic 400 / brick). Insert an assistant turn
        // after the results to keep alternation — unless one already follows.
        let after_results = insert_at + missing_count;
        if self.messages.get(after_results).map(role) != Some("assistant") {
            self.messages.insert(
                after_results,
                json!({"role": "assistant", "content": "[interrupted]"}),
            );
        }
    }

    /// Run one user turn to convergence: call the model, execute tool calls
    /// (permission-gated), repeat until it stops or a stop condition trips; footer.
    /// chars/4 estimate of the next request's prompt (system + tools + conversation).
    /// Seeds the live context-fill before real usage — the visible transcript omits
    /// the system prompt and tool defs, which dominate an agent prompt.
    pub(crate) fn estimated_prompt_tokens(&self) -> u64 {
        let msg_chars: usize = self.messages.iter().map(|m| m.to_string().len()).sum();
        let tool_chars: usize = self.tools_openai.iter().map(|t| t.to_string().len()).sum();
        ((msg_chars + tool_chars) / 4) as u64
    }

    /// Under `AIVO_DEBUG`, warn when the cached prefix (system prompt + tools) drifts.
    fn check_prefix_drift(&mut self) {
        if std::env::var("AIVO_DEBUG").as_deref() != Ok("1") {
            return;
        }
        let fp = tool_repair::prefix_fingerprint(&self.messages[0], &self.tools_openai);
        if let Some(prev) = self.prefix_fp
            && prev != fp
        {
            let what = match (prev.0 != fp.0, prev.1 != fp.1) {
                (true, true) => "system prompt and tool schema",
                (true, false) => "system prompt",
                _ => "tool schema",
            };
            agent_debug(&format!(
                "prefix drift: {what} changed — prompt cache will miss"
            ));
        }
        self.prefix_fp = Some(fp);
    }

    /// Cloned per step; strips the leading system prompt in plain-chat mode, so the
    /// single-system-message invariant `restore_conversation` relies on stays intact.
    fn outgoing_messages(&self) -> Vec<Value> {
        if self.agent_tools_enabled {
            return self.messages.clone();
        }
        self.messages
            .iter()
            .filter(|m| role(m) != "system")
            .cloned()
            .collect()
    }

    pub async fn run_turn(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi, user_text: String) {
        self.begin_user_turn(Value::String(user_text.clone()), user_text);
        self.run_loop(ctx, ui).await;
    }

    /// Like [`run_turn`], but the opening message carries multimodal content (text +
    /// image parts) so a vision model keeps the tool loop. `checkpoint_prompt` is the
    /// plain-text `/rewind` label.
    pub async fn run_turn_with_content(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        content: Value,
        checkpoint_prompt: String,
    ) {
        self.begin_user_turn(content, checkpoint_prompt);
        self.run_loop(ctx, ui).await;
    }

    /// Record the opening user turn: repair the tail, checkpoint, append (merging into a
    /// trailing user turn to keep the no-consecutive-user invariant).
    fn begin_user_turn(&mut self, user_content: Value, checkpoint_prompt: String) {
        self.repair_interrupted_tail();
        self.check_prefix_drift();
        // `/rewind` checkpoint at this turn's opening user message. The push below
        // merges into a trailing `user`, so the turn starts there if the tail is `user`.
        let turn_start = if self.messages.last().map(role) == Some("user") {
            self.messages.len().saturating_sub(1)
        } else {
            self.messages.len()
        };
        // Reuse an existing checkpoint at this index (merging into an interrupted turn):
        // a second would alias `msg_index` and snapshot the partial edits; the existing pre-edit tree is right.
        let already_checkpointed = self.checkpoints.last().map(|c| c.msg_index) == Some(turn_start);
        if !already_checkpointed {
            // Tree snapshot is lazy (taken in `execute_tool_batch` once about to mutate),
            // so a read-only turn pays no git cost; stays `None` for a turn that never mutates.
            self.checkpoints.push(Checkpoint {
                msg_index: turn_start,
                prompt: checkpoint_prompt,
                tree: None,
                changed: None,
            });
        }
        // Merge into a preceding user turn (e.g. a turn cancelled before its first
        // reply) rather than appending a second one (two consecutive users → Anthropic 400 / brick).
        self.push_user_content(user_content);
    }

    async fn run_loop(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi) {
        let mut steps = 0usize;
        let mut leaked_nudges = 0usize;
        let mut completion_nudges = 0usize;
        // Post-edit self-verification (opt-in): the project's validator, detected once.
        let validator = self.self_correct.then(|| verify::detect(ctx.cwd)).flatten();
        let mut selfcorrect_attempts = 0usize;
        let mut tokens = 0u64;
        // Real provider-measured split, summed across steps (drained by the TUI for stats). Reset per turn.
        self.turn_usage = SessionTokens::default();
        // Last step's prompt+completion — the real context fill (`tokens` re-counts the prompt each step).
        let mut context_tokens = 0u64;
        let started = Instant::now();
        let mut last_batch = String::new();
        let mut repeats = 0usize;
        // Track the effective file region separately — a paging loop varies junk args,
        // defeating `batch_sig`.
        let mut last_page: Option<(String, u64)> = None;
        let mut page_repeats = 0usize;
        // Same-signature tool-failure streaks: hint the schema, then hard-stop a loop.
        let mut failure_guard = guards::FailureGuard::default();
        let mut converged = false;

        for _ in 0..self.max_steps {
            // Compact before composing the request if we'd otherwise overflow.
            tokens += self.maybe_compact(ctx, ui).await;

            let mut extra = Map::new();
            // Omit tool_choice when no tools are offered — a bridge can 400 on it.
            if self.agent_tools_enabled {
                extra.insert("tool_choice".into(), json!("auto"));
            }
            // Thinking control (see `thinking_request`); the serve translates
            // `reasoning_effort` per upstream. `thinking:{type:"disabled"}` = the off-switch where the scale has no "off".
            let (effort, disable_thinking) = self.thinking_request();
            if let Some(effort) = effort {
                extra.insert("reasoning_effort".into(), json!(effort));
            }
            if disable_thinking {
                extra.insert("thinking".into(), json!({ "type": "disabled" }));
            }
            let tools = if self.agent_tools_enabled {
                self.tools_openai.clone()
            } else {
                Vec::new()
            };
            let mut request = ChatRequest {
                model: self.model.clone(),
                messages: self.outgoing_messages(),
                tools,
                extra,
            };
            // Paired with measured usage below to calibrate; re-measured if overflow recovery shrinks the request.
            let mut sent_estimate = estimate_tokens(&request.messages);

            ui.turn_start();
            // Seed the live context-fill; the measured total replaces it once the step returns.
            ui.context_usage(self.estimated_context_tokens(), false);
            // Auto-retry transient failures with backoff — only when nothing streamed yet (re-streaming double-renders).
            let mut retries = 0usize;
            let mut forced_compactions = 0usize;
            let message = loop {
                let mut streamed = false;
                let result = serve_client::complete(
                    ctx.client,
                    ctx.serve_base,
                    ctx.auth,
                    &request,
                    &mut |delta| {
                        // Any streamed output means a retry would double-render.
                        streamed = true;
                        match delta {
                            serve_client::StreamDelta::Text(t) => ui.assistant_text(t),
                            serve_client::StreamDelta::Reasoning(r) => ui.assistant_reasoning(r),
                        }
                    },
                )
                .await;
                match result {
                    Ok(m) => break m,
                    Err(e) if retries < MAX_RETRIES && !streamed && error_is_retryable(&e) => {
                        retries += 1;
                        ui.notify(&format!(
                            "connection issue — retrying ({retries}/{MAX_RETRIES})…"
                        ));
                        tokio::time::sleep(retry_delay(retries, e.retry_after)).await;
                    }
                    // Over the input limit despite our budget check: calibrate from the
                    // rejection, force-fit, retry — else the 400 is terminal and re-sends every turn.
                    Err(e)
                        if forced_compactions < MAX_FORCED_COMPACTIONS
                            && !streamed
                            && is_context_overflow_error(&e.message) =>
                    {
                        forced_compactions += 1;
                        self.recalibrate_from_overflow(&e.message);
                        self.force_fit_budget();
                        request.messages = self.outgoing_messages();
                        sent_estimate = estimate_tokens(&request.messages);
                        ui.notify("context over the model's limit — compacting and retrying…");
                    }
                    Err(e) => {
                        ui.notify_error(&format!("LLM error: {e}"));
                        break AssistantMessage {
                            content: Some(format!("[error: {e}]")),
                            tool_calls: vec![],
                            usage: None,
                        };
                    }
                }
            };
            steps += 1;
            let step_tokens = usage_tokens(&message.usage);
            tokens += step_tokens;
            if message.usage.is_some() {
                context_tokens = step_tokens;
                ui.context_usage(step_tokens, true);
                self.update_calibration(sent_estimate, step_tokens);
            }
            // Sum the real prompt/completion/cache split across steps (same parser as the serve, for a consistent index).
            if let Some(u) = &message.usage
                && let Some(split) = extract_usage_from_value(&json!({ "usage": u }))
            {
                self.turn_usage = self.turn_usage.merge(SessionTokens {
                    prompt_tokens: split.prompt,
                    completion_tokens: split.completion,
                    cache_read_tokens: split.cache_read,
                    cache_write_tokens: split.cache_creation,
                });
                ui.turn_tokens(self.turn_usage.completion_tokens);
            }

            // Per-turn cost breaker for unattended runs (0 = no cap; TUI relies on esc).
            if self.max_output_tokens > 0
                && self.turn_usage.completion_tokens >= self.max_output_tokens
            {
                ui.notify(&format!(
                    "stopping: reached the per-turn output-token budget ({})",
                    self.max_output_tokens
                ));
                converged = true;
                break;
            }

            // Empty completion converges the turn; don't record it — an empty assistant 400s the Anthropic bridge (non-retryable → bricks the next turn).
            let no_output = message.tool_calls.is_empty()
                && message.content.as_deref().is_none_or(str::is_empty);
            if no_output {
                // Silent convergence reads as success ("Done" with no answer); say so.
                ui.notify("the model returned an empty response — no answer produced");
                converged = true;
                break;
            }
            // Tool calls emitted as text ran nothing: strip, nudge, and retry.
            if message.tool_calls.is_empty()
                && leaked_nudges < MAX_LEAKED_NUDGES
                && let Some(cleaned) = message
                    .content
                    .as_deref()
                    .and_then(tool_repair::strip_if_leaked)
            {
                leaked_nudges += 1;
                // Drop the markup that already streamed so it never persists.
                ui.discard_streamed_segment();
                // Assistant turn before the nudge keeps alternation: a user nudge right after `tool` results 400s the bridge.
                let recorded = if cleaned.trim().is_empty() {
                    LEAKED_TOOL_CALL_PLACEHOLDER.to_string()
                } else {
                    cleaned
                };
                let recorded_msg = AssistantMessage {
                    content: Some(recorded),
                    tool_calls: Vec::new(),
                    usage: message.usage.clone(),
                };
                self.messages.push(assistant_to_openai(&recorded_msg));
                self.push_text_turn("user", LEAKED_TOOL_CALL_NUDGE.to_string());
                continue;
            }
            self.messages.push(assistant_to_openai(&message));

            if message.tool_calls.is_empty() {
                // A text-only turn that isn't actually done shouldn't be accepted as the
                // final answer — nudge once (bounded). The assistant turn is already
                // recorded above, so the user nudge keeps role alternation.
                if self.require_completion
                    && completion_nudges < MAX_COMPLETION_NUDGES
                    && message.content.as_deref().is_some_and(|c| {
                        guards::is_incomplete_answer(c) || guards::ends_with_continuation_cue(c)
                    })
                {
                    completion_nudges += 1;
                    ui.notify("the answer looks unfinished — asking the model to continue");
                    self.push_text_turn("user", COMPLETION_NUDGE.to_string());
                    continue;
                }
                // A declared-done turn isn't accepted while the validator fails — feed
                // the failure back (bounded) so the model fixes the cause.
                if let Some(v) = &validator
                    && selfcorrect_attempts < MAX_SELFCORRECT_ATTEMPTS
                {
                    match verify::run(v.clone(), ctx.cwd).await {
                        Err(summary) => {
                            selfcorrect_attempts += 1;
                            ui.notify(&format!("{} failed — asking the model to fix", v.label));
                            self.push_text_turn(
                                "user",
                                format!("{VERIFY_FAILED_PREFIX}\n{summary}"),
                            );
                            continue;
                        }
                        Ok(()) => ui.notify(&format!("verified: {} passed", v.label)),
                    }
                }
                converged = true; // answered without calling tools
                // Finalize a started plan on real convergence so it can't linger as
                // "0/N done". Gated on `started` — an all-pending plan (planned then converged) is left alone.
                if plan::started(&self.plan) && plan::complete_all(&mut self.plan) {
                    ui.plan_updated(&self.plan);
                }
                break;
            }

            // No-progress guard: identical consecutive batches, plus a paging loop that
            // re-reads one region while varying junk args (which `batch_sig` misses).
            let batch = batch_sig(&message.tool_calls);
            if batch == last_batch {
                repeats += 1;
            } else {
                repeats = 0;
                last_batch = batch;
            }
            let page = page_read_key(&message.tool_calls);
            if page.is_some() && page == last_page {
                page_repeats += 1;
            } else {
                page_repeats = 0;
                last_page = page;
            }

            // Execute this batch (permission-gated); returns extra tokens accrued inside
            // it (sub-agent calls) plus each failed call's (tool, error) for the guard.
            let (batch_tokens, failures) =
                self.execute_tool_batch(ctx, ui, &message.tool_calls).await;
            tokens += batch_tokens;

            if repeats + 1 >= REPEAT_LIMIT || page_repeats + 1 >= REPEAT_LIMIT {
                ui.notify("stopping: the model repeated the same action with no progress");
                converged = true;
                break;
            }

            // Same-signature tool-failure guard: hint the schema, then hard-stop a loop.
            match failure_guard.observe(&failures) {
                guards::FailureAction::Stop => {
                    ui.notify("stopping: a tool call kept failing the same way");
                    converged = true;
                    break;
                }
                guards::FailureAction::Hint { tool, error } => {
                    // Append to the last tool result — a fresh user turn after tool
                    // results would 400 the Anthropic bridge (two consecutive user turns).
                    if let Some(hint) = self.tool_failure_hint(&tool, &error)
                        && let Some(last) = self.messages.last_mut()
                        && let Some(c) = last.get("content").and_then(Value::as_str)
                    {
                        last["content"] = json!(format!("{c}\n\n{hint}"));
                        ui.notify(&format!("re-sent {tool}'s schema after repeated failures"));
                    }
                }
                guards::FailureAction::None => {}
            }
        }

        if !converged {
            ui.notify(&format!("reached the step limit ({})", self.max_steps));
        }
        ui.footer(
            None,
            steps,
            tokens,
            context_tokens,
            started.elapsed().as_secs(),
        );

        // Record paths this turn changed so `/rewind` reverts only the agent's edits.
        // Interrupted turns skip this — finalized lazily by `rewind_to`.
        let changed = match self.checkpoints.last().and_then(|c| c.tree.clone()) {
            Some(tree) => match self.checkpoint_store.as_mut() {
                Some(store) => Some(store.changed_since(&tree).await),
                None => Some(Vec::new()),
            },
            None => Some(Vec::new()),
        };
        if let Some(cp) = self.checkpoints.last_mut() {
            cp.changed = changed;
        }
    }

    /// Execute one turn's batch of tool calls, appending a `tool` message for each
    /// in call order: classify + permission-gate up front, run side-effect-free
    /// built-ins concurrently and the rest sequentially, then report in call order.
    /// Returns extra tokens accrued by any sub-agent runs.
    async fn execute_tool_batch(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        tool_calls: &[ToolCall],
    ) -> (u64, Vec<(String, String)>) {
        // Lazy `/rewind` checkpoint: snapshot the pre-edit (turn-start) tree the first
        // time a batch isn't entirely read-only. Conservative — anything off the `is_read_only` allowlist triggers it.
        if self.checkpoints.last().is_some_and(|c| c.tree.is_none())
            && !tool_calls.iter().all(|c| tools::is_read_only(&c.name))
        {
            let tree = match self.checkpoint_store.as_mut() {
                Some(store) => store.snapshot().await,
                None => None,
            };
            if let Some(cp) = self.checkpoints.last_mut() {
                cp.tree = tree;
            }
        }

        let mut extra_tokens = 0u64;
        // (tool, error) per failed call, for the same-signature failure guard.
        let mut failures: Vec<(String, String)> = Vec::new();
        let mut outcomes: Vec<Option<Result<String, String>>> = vec![None; tool_calls.len()];
        let mut parallel_idx: Vec<usize> = Vec::new();
        let mut sequential_idx: Vec<usize> = Vec::new();

        for (i, call) in tool_calls.iter().enumerate() {
            let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
            // The plan tool renders as a checklist card and never needs permission —
            // resolve it up front; its result still joins history (call↔result invariant).
            if n == "update_plan" {
                let content = match plan::parse_plan(&call.arguments) {
                    Ok(mut items) => {
                        // Fill in steps the model advanced past but forgot to mark done, so the checklist stays monotone.
                        plan::normalize_progress(&mut items);
                        self.plan = items.clone();
                        ui.plan_updated(&items);
                        plan::confirmation(&items)
                    }
                    Err(e) => e,
                };
                outcomes[i] = Some(Ok(content));
                continue;
            }
            ui.tool_start(n, &call.arguments);
            // Plan mode backstop (the tool is also hidden); the error steers the model.
            if self.read_only && tools::is_mutating(n) {
                outcomes[i] = Some(Err(
                    "Plan mode is read-only — do not modify files or run commands. \
Investigate with read-only tools and write the implementation plan instead."
                        .to_string(),
                ));
                continue;
            }
            // Confirm only genuinely risky actions: destructive command, out-of-cwd
            // write, blind overwrite of an unread file, or an untrusted external tool.
            let needs_confirm = tools::is_dangerous(n, &call.arguments, ctx.cwd)
                || self.write_clobbers_unread(n, &call.arguments, ctx.cwd)
                || secrets_guard::read_targets_secret(n, &call.arguments, ctx.cwd)
                || self
                    .external
                    .as_ref()
                    .is_some_and(|e| e.requires_approval(&call.name));
            // Hard floor: an unrecoverable command is confirmed even under auto-approve, never remembered; off a TTY fails closed.
            let catastrophic = tools::is_catastrophic(n, &call.arguments);
            // Remote mutation: also confirmed under auto-approve, but AlwaysAllow may
            // remember it so a deploy loop isn't re-prompted each identical call.
            let remote_side_effect =
                !catastrophic && tools::is_remote_side_effect(n, &call.arguments);
            let allowed = if catastrophic {
                let preview = tools::preview(n, &call.arguments);
                // Allow and AlwaysAllow both run it once only — never remembered.
                !matches!(
                    ui.ask_permission(n, preview.as_deref()).await,
                    Decision::Deny
                )
            } else if remote_side_effect && !self.grants.covers(n, &call.arguments, ctx.cwd) {
                let preview = tools::preview(n, &call.arguments);
                match ui.ask_permission(n, preview.as_deref()).await {
                    Decision::Allow => true,
                    Decision::AlwaysAllow => {
                        self.grants.remember(n, &call.arguments, ctx.cwd);
                        true
                    }
                    Decision::Deny => false,
                }
            } else if !needs_confirm
                || ctx.auto_approve_enabled()
                || self.grants.covers(n, &call.arguments, ctx.cwd)
            {
                true
            } else {
                let preview = tools::preview(n, &call.arguments);
                match ui.ask_permission(n, preview.as_deref()).await {
                    Decision::Allow => true,
                    Decision::AlwaysAllow => {
                        self.grants.remember(n, &call.arguments, ctx.cwd);
                        true
                    }
                    Decision::Deny => false,
                }
            };
            if !allowed {
                outcomes[i] = Some(Err("denied by user".to_string()));
                continue;
            }
            // A side-effect-free built-in runs concurrently — unless an external tool
            // shadows the same name, which must route to its source sequentially.
            let shadowed = self
                .external
                .as_ref()
                .is_some_and(|e| e.handles(&call.name));
            if tools::is_parallel_safe(n) && !shadowed {
                parallel_idx.push(i);
            } else {
                sequential_idx.push(i);
            }
        }

        // Fan out the side-effect-free calls: they share no mutable state, so poll them together (no spawn, no Send bound).
        if !parallel_idx.is_empty() {
            let cwd = ctx.cwd;
            let runs = parallel_idx.iter().map(|&i| {
                let call = &tool_calls[i];
                async move { (i, tools::execute(&call.name, &call.arguments, cwd).await) }
            });
            for (i, result) in futures::future::join_all(runs).await {
                // Anchor a read baseline as soon as the read succeeds, before the
                // sequential pass runs — so a same-batch edit is checked against what
                // was just read, not a stale prior-turn snapshot.
                if result.is_ok() {
                    let call = &tool_calls[i];
                    let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
                    self.file_tracker.record(n, &call.arguments, cwd);
                }
                outcomes[i] = Some(result);
            }
        }

        // Concurrent sub-agents: if the model fanned out several `subagent` calls in
        // one batch (and we're not in read-only plan mode), run them together — each a
        // buffered sub-engine sharing no UI — instead of one at a time. A lone
        // sub-agent stays in the sequential pass so its progress still streams live.
        let subagent_idx: Vec<usize> = if self.read_only {
            Vec::new()
        } else {
            sequential_idx
                .iter()
                .copied()
                .filter(|&i| {
                    let c = &tool_calls[i];
                    subagents::normalize_tool_name(&c.name).unwrap_or(&c.name) == "subagent"
                })
                .collect()
        };
        if subagent_idx.len() >= 2 {
            ui.notify(&format!(
                "running {} sub-agents in parallel",
                subagent_idx.len()
            ));
            let base = self.turn_usage.completion_tokens;
            let this: &Self = self;
            let mut sub_tokens_total = 0u64;
            // Chunk by the cap so a wide fan-out doesn't stampede the provider: each
            // chunk runs concurrently (join_all — same primitive as the read batch, and
            // unlike buffer_unordered it doesn't impose a higher-ranked Send bound on
            // the heavy sub-engine future), chunks run one after another.
            for chunk in subagent_idx.chunks(SUBAGENT_PARALLEL_CAP) {
                let runs = chunk.iter().map(|&i| {
                    let args = &tool_calls[i].arguments;
                    async move { (i, this.run_subagent(ctx, None, base, args).await) }
                });
                for (i, res) in futures::future::join_all(runs).await {
                    outcomes[i] = Some(match res {
                        Ok((msg, toks)) => {
                            sub_tokens_total = sub_tokens_total.saturating_add(toks);
                            Ok(msg)
                        }
                        Err(e) => Err(e),
                    });
                }
            }
            extra_tokens = extra_tokens.saturating_add(sub_tokens_total);
            self.turn_usage.completion_tokens = self
                .turn_usage
                .completion_tokens
                .saturating_add(sub_tokens_total);
            sequential_idx.retain(|i| !subagent_idx.contains(i));
        }

        // Run the ordered calls one at a time — they mutate the engine or workspace, so concurrency is unsafe.
        for &i in &sequential_idx {
            let call = &tool_calls[i];
            let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
            // Fail closed if a mutating tool targets a file changed on disk since the
            // model read it — clobbering an external edit is worse than a re-read.
            if let Some(msg) = self.file_tracker.stale_block(n, &call.arguments, ctx.cwd) {
                outcomes[i] = Some(Err(msg));
                continue;
            }
            let result = if n == "skill" {
                // Resolved from the engine's discovered skills, not tools::execute.
                let name = call
                    .arguments
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                skills::load_skill_result(&self.skills, name)
            } else if n == "subagent" && self.read_only {
                // A sub-engine isn't read-only; refuse delegation in plan mode.
                Err(
                    "Plan mode is read-only — cannot delegate to a subagent while planning."
                        .to_string(),
                )
            } else if n == "subagent" {
                // Fresh sub-engine on the same serve/cwd; fold its total in. Pass the UI + base so it forwards live token growth.
                let base = self.turn_usage.completion_tokens;
                match self
                    .run_subagent(ctx, Some(&mut *ui), base, &call.arguments)
                    .await
                {
                    Ok((msg, sub_tokens)) => {
                        extra_tokens += sub_tokens;
                        self.turn_usage.completion_tokens =
                            self.turn_usage.completion_tokens.saturating_add(sub_tokens);
                        Ok(msg)
                    }
                    Err(e) => Err(e),
                }
            } else if n == "take_note" {
                // Durable scratchpad (capped, oldest dropped). Held in the engine, so it runs in the ordered pass.
                match notes::parse_note(&call.arguments) {
                    Ok(note) => {
                        if self.notes.len() >= MAX_NOTES {
                            self.notes.remove(0);
                        }
                        self.notes.push(note);
                        Ok(format!("Noted ({} saved).", self.notes.len()))
                    }
                    Err(e) => Err(e),
                }
            } else if n == "switch_model" {
                match call.arguments.get("model").and_then(|v| v.as_str()) {
                    Some(m) if !m.trim().is_empty() => ui.switch_chat_model(m.trim()).await,
                    _ => Err("switch_model: missing `model`.".to_string()),
                }
            } else if n == "set_effort" {
                match call.arguments.get("level").and_then(|v| v.as_str()) {
                    Some(l) if !l.trim().is_empty() => ui.set_chat_effort(l.trim()).await,
                    _ => Err("set_effort: missing `level`.".to_string()),
                }
            } else if n == "ask_user" {
                match ask::parse_ask(&call.arguments) {
                    Ok((question, options, allow_free_text)) => ui
                        .ask_user(&question, &options, allow_free_text)
                        .await
                        .map(|answer| ask::confirmation(&answer)),
                    Err(e) => Err(e),
                }
            } else if let Some(ext) = self.external.clone().filter(|e| e.handles(&call.name)) {
                // External tool — keyed on its raw advertised name (`mcp__*`), never normalized (matches the shadow check).
                ext.call(&call.name, &call.arguments).await
            } else if n == "run_bash" {
                // Run confined; a sandbox write-block offers an in-session escape hatch instead of a dead-end error.
                self.run_bash_with_escalation(ctx, ui, &call.arguments)
                    .await
            } else {
                tools::execute(n, &call.arguments, ctx.cwd).await
            };
            // Refresh the baseline right after our own write so a later edit to the same
            // file in this batch compares against what we just wrote, not the pre-edit state.
            if result.is_ok() {
                self.file_tracker.record(n, &call.arguments, ctx.cwd);
            }
            outcomes[i] = Some(result);
        }

        // LSP diagnostics-after-edit (opt-in): for each file an edit tool just wrote,
        // fold the language server's native error diagnostics into that tool's result
        // so the model fixes them this turn. Bounded + graceful-degrade.
        if let Some(lsp) = &self.lsp {
            // Write tools only; dedup so a path edited twice in the batch settles once.
            let mut targets: Vec<(usize, String)> = Vec::new();
            for (i, call) in tool_calls.iter().enumerate() {
                if !matches!(outcomes[i], Some(Ok(_))) {
                    continue;
                }
                let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
                if !crate::agent::file_tracker::is_write_tool(n) {
                    continue;
                }
                for p in crate::agent::file_tracker::tracked_paths(n, &call.arguments) {
                    if !targets.iter().any(|(_, t)| t == &p) {
                        targets.push((i, p));
                    }
                }
            }
            for (i, disp) in targets {
                let diags = lsp.diagnostics(&tools::resolve(ctx.cwd, &disp)).await;
                if let Some(block) = crate::agent::lsp::format_block(&disp, &diags)
                    && let Some(Ok(msg)) = &mut outcomes[i]
                {
                    msg.push_str(&block);
                }
            }
        }

        // Emit results and append tool messages in call order (call↔result pairing intact).
        for (i, call) in tool_calls.iter().enumerate() {
            let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
            let result = outcomes[i]
                .take()
                .unwrap_or_else(|| Err("tool produced no result".to_string()));
            // update_plan already surfaced via plan_updated. Normalized name so the label matches and aliased reads/writes track.
            if n != "update_plan" {
                ui.tool_result(n, &result);
            }
            if result.is_ok() {
                self.record_touched_file(n, &call.arguments);
            }
            let raw = match result {
                Ok(c) => c,
                Err(e) => {
                    failures.push((n.to_string(), e.clone()));
                    e
                }
            };
            // Redact secrets before going upstream; the local `tool_result` already showed the real output.
            let (content, redacted) = secrets_guard::redact_for_model(&raw);
            if redacted > 0 {
                ui.notify(&format!(
                    "redacted {redacted} secret-shaped value(s) from `{n}` output before sending upstream"
                ));
            }
            self.messages.push(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            }));
        }

        (extra_tokens, failures)
    }

    /// Corrective hint for a repeatedly-failing tool: the exact error plus the tool's
    /// JSON schema, so the model can fix its arguments. `None` if the tool isn't in the
    /// current tool set (e.g. a hallucinated name) — nothing useful to echo.
    fn tool_failure_hint(&self, tool: &str, error: &str) -> Option<String> {
        let schema = self.tools_openai.iter().find_map(|t| {
            let f = t.get("function")?;
            (f.get("name").and_then(Value::as_str) == Some(tool))
                .then(|| f.get("parameters").cloned())
                .flatten()
        })?;
        let schema = serde_json::to_string_pretty(&schema).ok()?;
        Some(format!(
            "[aivo] `{tool}` has now failed repeatedly with: {error}\n\
Before calling `{tool}` again, make its arguments match this schema exactly:\n{schema}"
        ))
    }

    /// Run a `run_bash` call confined to the workspace. If the OS sandbox blocks a
    /// write, offer to re-run outside the sandbox (same approval flow) instead of a
    /// dead-end error. Auto-approve / a prior "always" skip the prompt; off a TTY it
    /// fails closed, so the blocked result (with its hint) flows back.
    async fn run_bash_with_escalation(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        args: &Value,
    ) -> Result<String, String> {
        let outcome = tools::run_bash_confined(args, ctx.cwd).await;
        if !outcome.sandbox_blocked {
            return outcome.result;
        }
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        // Scoped to the exact command so "always" doesn't blanket-escalate every bash call.
        let ekey = format!("run_bash_unsandboxed\u{0}{command}");
        let approved = ctx.auto_approve_enabled() || self.grants.covers_key(&ekey) || {
            let preview = format!(
                "{command}\n\nThe workspace sandbox blocked this — it writes outside {}. \
Re-run the full command without write confinement?",
                ctx.cwd.display()
            );
            match ui
                .ask_permission("run_bash_unsandboxed", Some(&preview))
                .await
            {
                Decision::Allow => true,
                Decision::AlwaysAllow => {
                    self.grants.remember_key(ekey);
                    true
                }
                Decision::Deny => false,
            }
        };
        if !approved {
            // Keep the blocked output + hint so the model sees the escalation was declined.
            return outcome.result;
        }
        ui.notify(SANDBOX_ESCALATION_NOTICE);
        tools::run_bash_unconfined(args, ctx.cwd).await
    }

    /// True when a `write_file` would overwrite an existing file the model hasn't
    /// read/written this session — a blind clobber worth confirming. New or
    /// already-touched files pass through; edit_file/multi_edit must read first, so never blind.
    fn write_clobbers_unread(&self, name: &str, args: &Value, cwd: &Path) -> bool {
        if name != "write_file" {
            return false;
        }
        let Some(path) = args.get("path").and_then(|p| p.as_str()).map(str::trim) else {
            return false;
        };
        if path.is_empty() || self.touched_files.iter().any(|p| p == path) {
            return false;
        }
        let full = if Path::new(path).is_absolute() {
            std::path::PathBuf::from(path)
        } else {
            cwd.join(path)
        };
        full.exists()
    }

    fn record_touched_file(&mut self, name: &str, args: &Value) {
        // One definition of "which paths does this tool touch", shared with the staleness
        // tracker and grant store (`apply_patch` carries many in its V4A body; the rest one).
        for path in crate::agent::file_tracker::tracked_paths(name, args) {
            let path = path.trim();
            if path.is_empty() || self.touched_files.iter().any(|p| p == path) {
                continue;
            }
            if self.touched_files.len() >= MAX_TOUCHED_FILES {
                self.touched_files.remove(0);
            }
            self.touched_files.push(path.to_string());
        }
    }

    // --- /rewind: tree checkpoints ---

    /// Per-checkpoint `/rewind` targets in order for the picker: `(prompt, file_revertible)`.
    /// The TUI matches by prompt text newest-backward. Cheap and in-memory (no git).
    pub fn rewind_targets(&self) -> Vec<(String, bool)> {
        self.checkpoints
            .iter()
            .map(|c| (c.prompt.clone(), c.tree.is_some()))
            .collect()
    }

    /// Rewind to checkpoint `ordinal`: revert the union of files the rewound turns
    /// changed (leaving the user's independent edits), truncate to the turn's user
    /// message, drop the rewound checkpoints, re-derive the working set. A `None`-tree
    /// checkpoint rewinds the conversation only.
    pub async fn rewind_to(&mut self, ordinal: usize) -> RewindOutcome {
        let mut outcome = RewindOutcome::default();
        let tree = self.checkpoints.get(ordinal).and_then(|c| c.tree.clone());
        // Union of paths every rewound turn changed; finalize interrupted turns (`changed == None`) lazily.
        let mut paths: std::collections::BTreeSet<std::path::PathBuf> =
            std::collections::BTreeSet::new();
        for i in ordinal..self.checkpoints.len() {
            let recorded = self.checkpoints[i].changed.clone();
            let changed = match recorded {
                Some(c) => c,
                None => match self.checkpoints[i].tree.clone() {
                    Some(t) => match self.checkpoint_store.as_mut() {
                        Some(store) => store.changed_since(&t).await,
                        None => Vec::new(),
                    },
                    None => Vec::new(),
                },
            };
            paths.extend(changed);
        }
        let paths: Vec<std::path::PathBuf> = paths.into_iter().collect();
        if let (Some(tree), Some(store)) = (tree, self.checkpoint_store.as_mut()) {
            let report = store.restore_paths(&tree, &paths).await;
            outcome.restored = report.restored;
            outcome.deleted = report.deleted;
            outcome.error = report.error;
        }
        if let Some(cp) = self.checkpoints.get(ordinal) {
            let at = cp.msg_index.min(self.messages.len());
            self.messages.truncate(at);
        }
        self.checkpoints.truncate(ordinal);
        self.rebuild_working_set_from_log();
        outcome
    }

    /// Execute a `subagent` tool call: build a fresh sub-engine (same tools minus
    /// `subagent`, same cwd + serve, optionally a stronger model), run to convergence,
    /// return its answer. Capturing UI (only the result surfaces). Dangerous ops inherit
    /// the parent's auto-approve, else fail closed (no nested prompt).
    /// Run one sub-agent to completion and hand back `(result, tokens)`. `parent_ui`
    /// `Some` streams its activity to the parent (the lone-sub-agent path); `None`
    /// buffers silently, so several can run concurrently without sharing the UI.
    async fn run_subagent(
        &self,
        ctx: &TurnCtx<'_>,
        parent_ui: Option<&mut dyn AgentUi>,
        base: u64,
        args: &Value,
    ) -> Result<(String, u64), String> {
        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| "subagent: missing `task`".to_string())?;
        // Named specialist if `agent` matches; unknown names fall back to generic (lenient, don't fail the turn).
        let profile = args
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .and_then(|n| self.subagents.iter().find(|s| s.name == n));
        // Model precedence: explicit `model` arg > profile's pinned model > parent's model.
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|m| !m.is_empty())
            .or_else(|| profile.and_then(|p| p.model.as_deref()))
            .unwrap_or(&self.model);

        let mut sub = AgentEngine::new(
            &ctx.cwd.display().to_string(),
            model,
            &self.date,
            &self.guides,
            &self.skills,
            self.context_window,
            SUBAGENT_MAX_STEPS,
        );
        sub.drop_subagent_tool();
        // First-party parent keeps delegates first-party so their output won't disclose the provider.
        if self.first_party {
            sub.set_first_party();
        }
        // Honor the parent's hosted-web-search opt-in/out in delegated work.
        sub.set_web_search_enabled(self.use_web_search_enabled);
        // Carry the parent's reasoning effort — but only if it's valid for the sub's model (may differ), else keep the sub's default.
        if let Some(effort) = &self.reasoning_effort
            && crate::services::model_metadata::snapshot_limits(model)
                .is_some_and(|c| c.reasoning_efforts.iter().any(|l| l == effort))
        {
            sub.set_reasoning_effort(effort.clone());
        }
        // Share the parent's external tools (MCP), reusing the already-connected servers.
        if let Some(ext) = &self.external {
            sub.set_external_tools(ext.clone());
        }
        // Fold in the specialist's role + scope. After MCP wiring so a `tools` allow-list applies to the full offered set.
        if let Some(p) = profile {
            sub.apply_profile(p);
        }

        // Matched specialist, else the requested name, else generic.
        let agent_name = profile
            .map(|p| p.name.clone())
            .or_else(|| {
                args.get("agent")
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|n| !n.is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let mut ui = SubagentUi {
            parent: parent_ui,
            base,
            agent_name,
            ..Default::default()
        };
        // Box the recursive future (run_turn → subagent → run_turn) so it isn't infinitely-sized.
        Box::pin(sub.run_turn(ctx, &mut ui, task.to_string())).await;
        Ok((ui.result_message(), ui.tokens))
    }
}

/// The `subagent` tool — engine-handled (needs the serve + a fresh engine), top-level
/// engine only. When named specialists exist, an `agent` field enumerates them.
fn subagent_tool_spec(subagents: &[Subagent]) -> ToolSpec {
    let mut properties = json!({
        "task": {"type": "string", "description": "A complete, standalone instruction for the sub-agent."},
        "model": {"type": "string", "description": "Optional model id to run the sub-agent on (default: the agent's configured model, else same as you)."}
    });
    let mut description = "Delegate a self-contained subtask to a fresh sub-agent that has the same \
file/shell tools and runs its own loop, then hands back its result. Use it to keep your own context \
focused (offload a big investigation), or pass `model` to delegate hard work to a stronger model. The \
sub-agent does not see this conversation, so make `task` complete and standalone; it cannot spawn \
further sub-agents. Call `subagent` several times in one turn to run independent investigations in \
parallel — they execute concurrently and each result comes back separately."
        .to_string();
    if !subagents.is_empty() {
        let names: Vec<&str> = subagents.iter().map(|s| s.name.as_str()).collect();
        if let Some(props) = properties.as_object_mut() {
            props.insert(
                "agent".to_string(),
                json!({
                    "type": "string",
                    "enum": names,
                    "description": "Optional named specialist to run (listed in your instructions). It brings its own role and may pin its own model. Omit for a generic sub-agent."
                }),
            );
        }
        description.push_str(
            " You also have named specialist sub-agents (see your instructions); pass one in `agent` to \
use its role instead of a generic sub-agent.",
        );
    }
    ToolSpec {
        name: "subagent".to_string(),
        description,
        parameters: json!({
            "type": "object",
            "properties": properties,
            "required": ["task"]
        }),
    }
}

/// Engine-handled — routed to `AgentUi::switch_chat_model`, not `tools::execute`.
fn switch_model_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "switch_model".to_string(),
        description: "Switch the model powering THIS aivo code session when the user asks for a \
different one. Pass the model id or a distinctive part of it (e.g. \"opus\", \"gpt-5\"). The switch \
takes effect on the user's next message and the conversation is preserved. If the name is \
ambiguous or unavailable you'll get candidates back — relay them or suggest `/model`."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model id, or a distinctive part of it, to switch to."}
            },
            "required": ["model"]
        }),
    }
}

/// Engine-handled — routed to `AgentUi::set_chat_effort`.
fn set_effort_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "set_effort".to_string(),
        description:
            "Set the reasoning-effort level for THIS aivo code session (e.g. low, medium, \
high) when the user asks. Only valid for models that expose effort levels — you'll get the valid \
options back if the level or the current model doesn't support it."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "level": {"type": "string", "description": "Effort level to set (e.g. low, medium, high)."}
            },
            "required": ["level"]
        }),
    }
}

/// Capturing UI for a sub-agent run. `cur_text` holds the in-flight step's text,
/// rolling into `last_nonempty` at each new step. The answer is the converging step's
/// text, falling back to the last non-empty step (so an answer emitted alongside the
/// final tool call isn't lost). Permission prompts forward to the parent UI, so the
/// catastrophic-command floor holds for sub-agents too; denies if detached.
#[derive(Default)]
struct SubagentUi<'a> {
    cur_text: String,
    last_nonempty: String,
    /// Last engine notice — surfaced when the sub-agent produces no answer, so the failure reason isn't swallowed.
    last_notice: String,
    steps: usize,
    /// The sub-agent's cumulative token usage, folded into the parent turn's total.
    tokens: u64,
    /// Forward live token growth (base + sub so-far) to the parent UI.
    parent: Option<&'a mut dyn AgentUi>,
    base: u64,
    /// Specialist name + turn counter, forwarded to the parent's status feed.
    agent_name: String,
    turn_no: usize,
}

impl SubagentUi<'_> {
    /// The sub-agent's answer: the converging step's text, else the last non-empty step's.
    fn answer(&self) -> &str {
        if self.cur_text.trim().is_empty() {
            self.last_nonempty.trim()
        } else {
            self.cur_text.trim()
        }
    }

    /// The tool result the parent receives: the answer (+ step count), else the
    /// failure notice (so an LLM error / step-limit isn't masked as "no answer").
    fn result_message(&self) -> String {
        let answer = self.answer();
        if !answer.is_empty() {
            format!("{answer}\n\n[sub-agent: {} step(s)]", self.steps)
        } else if !self.last_notice.trim().is_empty() {
            format!(
                "(sub-agent produced no answer — {})",
                self.last_notice.trim()
            )
        } else {
            format!(
                "(sub-agent finished in {} step(s) without a textual answer)",
                self.steps
            )
        }
    }

    fn forward_activity(&mut self, tool: &str, args: &Value) {
        let Self {
            parent,
            agent_name,
            turn_no,
            ..
        } = self;
        if let Some(p) = parent.as_deref_mut() {
            p.subagent_activity(agent_name, tool, args, *turn_no);
        }
    }
}

impl AgentUi for SubagentUi<'_> {
    fn turn_start(&mut self) {
        // New step: the previous step's text becomes the fallback, current buffer resets.
        if !self.cur_text.trim().is_empty() {
            self.last_nonempty = std::mem::take(&mut self.cur_text);
        }
        self.turn_no += 1;
        self.forward_activity("", &Value::Null);
    }
    fn assistant_text(&mut self, delta: &str) {
        self.cur_text.push_str(delta);
    }
    fn discard_streamed_segment(&mut self) {
        self.cur_text.clear();
    }
    fn tool_start(&mut self, name: &str, args: &Value) {
        self.forward_activity(name, args);
    }
    fn tool_result(&mut self, _name: &str, _result: &Result<String, String>) {}
    fn notify(&mut self, text: &str) {
        self.last_notice = text.to_string();
    }
    fn footer(&mut self, _summary: Option<&str>, steps: usize, tokens: u64, _c: u64, _e: u64) {
        self.steps = steps;
        self.tokens = tokens;
    }
    fn turn_tokens(&mut self, output: u64) {
        let total = self.base.saturating_add(output);
        if let Some(p) = self.parent.as_deref_mut() {
            p.turn_tokens(total);
        }
    }
    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        preview: Option<&'a str>,
    ) -> BoxFuture<'a, Decision> {
        // Forward to the parent (card in the TUI, fail-closed when headless) rather than
        // auto-allowing, so the catastrophic-command floor holds for sub-agents too.
        match self.parent.as_deref_mut() {
            Some(p) => p.ask_permission(tool, preview),
            None => Box::pin(async { Decision::Deny }),
        }
    }
}

/// Drive one agent session: a one-shot task, or an interactive REPL when none is
/// given. Slash commands (`/help`, `/clear`, `/exit`) are handled between turns.
pub async fn run_session(
    engine: &mut AgentEngine,
    ctx: &TurnCtx<'_>,
    task: Option<String>,
    ui: &mut dyn AgentUi,
) {
    let interactive = task.is_none();
    let first = match task {
        Some(t) => Some(t),
        None => next_turn(engine, ui),
    };
    let Some(mut turn) = first else {
        return; // user quit at the first prompt
    };
    loop {
        engine.run_turn(ctx, ui, turn).await;
        if !interactive {
            break;
        }
        match next_turn(engine, ui) {
            Some(t) => turn = t,
            None => break,
        }
    }
}

/// REPL slash commands (besides `/exit`, which `read_user_input` maps to None).
enum SlashCmd {
    Help,
    Clear,
    Unknown(String),
}

fn parse_slash(input: &str) -> Option<SlashCmd> {
    let cmd = input.trim().strip_prefix('/')?;
    Some(match cmd {
        "help" | "?" => SlashCmd::Help,
        "clear" => SlashCmd::Clear,
        other => SlashCmd::Unknown(other.to_string()),
    })
}

/// Read the next REPL turn, servicing slash commands in-loop. Returns the turn
/// text, or `None` to end the session (EOF / `/exit`).
fn next_turn(engine: &mut AgentEngine, ui: &mut dyn AgentUi) -> Option<String> {
    loop {
        let input = ui.read_user_input()?;
        if input.trim().is_empty() {
            continue;
        }
        match parse_slash(&input) {
            None => return Some(input),
            Some(SlashCmd::Help) => ui.notify("commands: /clear  reset context · /exit  quit"),
            Some(SlashCmd::Clear) => {
                engine.reset();
                ui.notify("context cleared");
            }
            Some(SlashCmd::Unknown(c)) => ui.notify(&format!("unknown command: /{c}")),
        }
    }
}

/// Appended (interactive chat only) via [`AgentEngine::set_confirm_before_build`]: the
/// agent proposes a plan and waits for the user before a sizeable build.
const CONFIRM_BEFORE_BUILD: &str = "One refinement to the action-bias above, for this \
interactive session: before you BUILD something substantial — scaffolding a new project, \
adding a whole feature, or making a large multi-file change — don't dive straight into \
creating or editing files. You may first investigate read-only (`read_file`, `grep`, `glob`, \
`list_dir`) to ground the plan, then reply with a short plan — the approach and a numbered \
list of the steps you'd take — and ask the user to confirm or adjust it. Do NOT create or \
modify files, or run build/scaffold/state-changing commands, until they approve; end your \
turn after asking. Wait for a clear go-ahead before you build. A reply that only asks for a \
change or states a preference — e.g. \"use a light theme\", \"add auth\", \"drop the export \
step\", \"make it Postgres\" — is a plan REVISION, not approval: fold it in, show the updated \
plan, and ask again. Start building only when the user clearly signals to proceed with no \
further changes (\"go\", \"yes, build it\", \"proceed\", \"lgtm\"). If one message both \
requests a change and says go (\"use a light theme, then go\"), apply the change and build. \
When it's unclear whether a reply means \"proceed\" or \"also change this\", ask rather than \
assuming. This is ONLY for \
sizeable, multi-step build work — for quick fixes, small single-file edits, answering \
questions, and read-only exploration, keep acting directly rather than making the user \
approve trivial work. And if the user has already told you to proceed, handed you a plan to \
implement, or asked you to work autonomously, skip the confirmation and just build.";

const FIRST_PARTY_IDENTITY: &str = "You are aivo's own assistant. If the user asks what model you \
are, who built you, or which provider or company is behind you, present yourself as aivo's \
assistant and don't name, confirm, or guess at the underlying model or provider — that's an \
implementation detail. Point them to `aivo models` for their configured setup instead.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::compaction::{
        COMPACT_RESERVE, PINNED_MAX_TOKENS, SUMMARY_SYSTEM_PROMPT, SUMMARY_UPDATE_SYSTEM_PROMPT,
        TOOL_RESULT_CLEARED, find_cut,
    };
    use crate::agent::plan::PlanStatus;
    use crate::agent::request::content_str;
    use crate::agent::tokens::{MAX_CALIBRATION, estimate_str_tokens};
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;

    #[test]
    fn web_search_toggle_adds_and_removes_local_tool() {
        let mut e = AgentEngine::new("/tmp", "deepseek-v4", "", &[], &[], 0, 0);
        let has = |e: &AgentEngine| {
            e.tools_openai
                .iter()
                .any(|t| t["function"]["name"].as_str() == Some("web_search"))
        };
        assert!(has(&e), "non-native model starts with web_search");
        e.set_web_search_enabled(false);
        assert!(!has(&e), "toggle off removes it");
        e.set_web_search_enabled(false);
        assert!(!has(&e));
        e.set_web_search_enabled(true);
        assert!(has(&e), "toggle on re-adds it");
    }

    #[test]
    fn gemini_keeps_local_web_search_not_native_server_tool() {
        // Gemini 400s on google_search + function tools, and the agent always has function tools.
        let e = AgentEngine::new("/tmp", "gemini-2.5-flash", "", &[], &[], 0, 0);
        assert!(
            e.tools_openai
                .iter()
                .any(|t| t["function"]["name"].as_str() == Some("web_search")),
            "gemini keeps the local web_search function tool"
        );
        assert!(
            !e.tools_openai
                .iter()
                .any(|t| t.get("type").and_then(|v| v.as_str()) == Some("web_search")),
            "gemini must not carry the native web_search server tool"
        );
    }

    #[derive(Default)]
    struct CapturingUi {
        tools: Vec<String>,
        text: String,
        notices: Vec<String>,
        plans: Vec<usize>,
        /// Statuses from the most recent `plan_updated` (to assert finalization).
        last_plan: Vec<PlanStatus>,
        footer_tokens: u64,
        deny: bool,
        /// Reply `AlwaysAllow` instead of `Allow`/`Deny` (takes precedence).
        always_allow: bool,
        asks: usize,
        /// The `tool` argument of each `ask_permission` call, in order.
        ask_tools: Vec<String>,
        turn_token_reports: Vec<u64>,
        discards: usize,
        /// Each forwarded sub-agent step: `(agent, tool, step)`.
        sub_activity: Vec<(String, String, usize)>,
    }
    impl AgentUi for CapturingUi {
        fn assistant_text(&mut self, t: &str) {
            self.text.push_str(t);
        }
        fn discard_streamed_segment(&mut self) {
            self.discards += 1;
            self.text.clear();
        }
        fn plan_updated(&mut self, items: &[PlanItem]) {
            self.plans.push(items.len());
            self.last_plan = items.iter().map(|i| i.status).collect();
        }
        fn tool_start(&mut self, name: &str, _: &Value) {
            self.tools.push(name.to_string());
        }
        fn tool_result(&mut self, _: &str, _: &Result<String, String>) {}
        fn notify(&mut self, t: &str) {
            self.notices.push(t.to_string());
        }
        fn footer(&mut self, _: Option<&str>, _: usize, tokens: u64, _: u64, _: u64) {
            self.footer_tokens = tokens;
        }
        fn turn_tokens(&mut self, output: u64) {
            self.turn_token_reports.push(output);
        }
        fn subagent_activity(&mut self, agent: &str, tool: &str, _: &Value, step: usize) {
            self.sub_activity
                .push((agent.to_string(), tool.to_string(), step));
        }
        fn ask_permission<'a>(
            &'a mut self,
            tool: &'a str,
            _: Option<&'a str>,
        ) -> BoxFuture<'a, Decision> {
            self.asks += 1;
            self.ask_tools.push(tool.to_string());
            let (always_allow, deny) = (self.always_allow, self.deny);
            Box::pin(async move {
                if always_allow {
                    Decision::AlwaysAllow
                } else if deny {
                    Decision::Deny
                } else {
                    Decision::Allow
                }
            })
        }
    }

    #[test]
    fn subagent_forwards_live_tokens_to_parent_with_base() {
        let mut parent = CapturingUi::default();
        let mut sub = SubagentUi {
            base: 100,
            parent: Some(&mut parent),
            ..Default::default()
        };
        sub.turn_tokens(20);
        sub.turn_tokens(55);
        drop(sub);
        assert_eq!(parent.turn_token_reports, vec![120, 155]);
    }

    /// `turn_start` bumps the step + reports thinking (empty tool); `tool_start`
    /// reports the tool — both tagged with the specialist name and 1-based step.
    #[test]
    fn subagent_forwards_step_activity_to_parent() {
        let mut parent = CapturingUi::default();
        let mut sub = SubagentUi {
            agent_name: "code-reviewer".to_string(),
            parent: Some(&mut parent),
            ..Default::default()
        };
        sub.turn_start();
        sub.tool_start("grep", &json!({"pattern": "fn"}));
        sub.turn_start();
        sub.tool_start("read_file", &json!({"path": "src/lib.rs"}));
        drop(sub);
        assert_eq!(
            parent.sub_activity,
            vec![
                ("code-reviewer".to_string(), String::new(), 1),
                ("code-reviewer".to_string(), "grep".to_string(), 1),
                ("code-reviewer".to_string(), String::new(), 2),
                ("code-reviewer".to_string(), "read_file".to_string(), 2),
            ]
        );
    }

    /// A sub-agent forwards permission asks to the parent (else the catastrophic floor is
    /// skipped for delegated work); a denying/headless parent blocks it, detached denies.
    #[tokio::test]
    async fn subagent_forwards_permission_to_parent_and_fails_closed() {
        let mut parent = CapturingUi {
            deny: true,
            ..Default::default()
        };
        let mut sub = SubagentUi {
            parent: Some(&mut parent),
            ..Default::default()
        };
        let decision = sub.ask_permission("run_bash", Some("rm -rf /")).await;
        assert!(matches!(decision, Decision::Deny));
        drop(sub);
        assert_eq!(parent.ask_tools, vec!["run_bash"]);

        // Detached (no parent) fails closed.
        let mut orphan = SubagentUi::default();
        assert!(matches!(
            orphan.ask_permission("run_bash", Some("rm -rf /")).await,
            Decision::Deny
        ));
    }

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-engine-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// No two adjacent `user` messages (the sequence the Anthropic bridge 400s on).
    fn assert_no_consecutive_user(messages: &[Value]) {
        for w in messages.windows(2) {
            assert!(
                !(role(&w[0]) == "user" && role(&w[1]) == "user"),
                "two consecutive user messages: {w:?}"
            );
        }
    }

    /// Build a one-tool-call SSE body for the fake serve.
    fn tool_call_sse(name: &str, args: Value) -> String {
        let delta = json!({"choices":[{"delta":{"tool_calls":[{
            "index": 0, "id": "c1",
            "function": {"name": name, "arguments": args.to_string()}
        }]}}]});
        format!("data: {delta}\n\ndata: [DONE]\n\n")
    }

    /// Build a single SSE body carrying a whole batch of tool calls (one assistant
    /// turn), each `(id, name, args)` placed at its own `index`.
    fn batch_tool_call_sse(calls: &[(&str, &str, Value)]) -> String {
        let entries: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, (id, name, args))| {
                json!({
                    "index": i, "id": id,
                    "function": {"name": name, "arguments": args.to_string()}
                })
            })
            .collect();
        let delta = json!({"choices":[{"delta":{"tool_calls": entries}}]});
        format!("data: {delta}\n\ndata: [DONE]\n\n")
    }

    fn spawn_sse_sequence(bodies: Vec<String>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for body in bodies {
                let Ok((mut sock, _)) = listener.accept() else {
                    break;
                };
                let mut buf = [0u8; 16384];
                let _ = sock.read(&mut buf); // drain the request
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });
        port
    }

    fn turn_ctx<'a>(client: &'a reqwest::Client, base: &'a str, cwd: &'a Path) -> TurnCtx<'a> {
        TurnCtx {
            client,
            serve_base: base,
            auth: None,
            cwd,
            yes: true,
            auto_approve: None,
        }
    }

    const WRITE_TOOL_SSE: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"out.txt\\\",\\\"content\\\":\\\"hi\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n";
    const FINAL_TEXT_SSE: &str =
        "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n";

    #[test]
    fn auto_approve_enabled_tracks_static_flag_and_live_toggle() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let client = reqwest::Client::new();
        let cwd = std::path::Path::new(".");
        let ctx = |yes, flag| TurnCtx {
            client: &client,
            serve_base: "",
            auth: None,
            cwd,
            yes,
            auto_approve: flag,
        };
        assert!(ctx(true, None).auto_approve_enabled());
        assert!(!ctx(false, None).auto_approve_enabled());
        // The live flag flips the SAME ctx: a mid-turn Shift+Tab is seen by the running turn.
        let flag = AtomicBool::new(false);
        let live = ctx(false, Some(&flag));
        assert!(!live.auto_approve_enabled());
        flag.store(true, Ordering::Relaxed);
        assert!(
            live.auto_approve_enabled(),
            "a mid-turn toggle is seen live"
        );
    }

    #[test]
    fn skills_wire_into_tools_and_system_prompt() {
        let skill = Skill {
            name: "demo".to_string(),
            description: "does a demo".to_string(),
            body: "BODY".to_string(),
            dir: PathBuf::from("/tmp/demo"),
        };
        let engine = AgentEngine::new("/tmp", "m", "", &[], std::slice::from_ref(&skill), 0, 0);

        // The `skill` tool is offered alongside the built-ins.
        let tool_names: Vec<&str> = engine
            .tools_openai
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(tool_names.contains(&"skill"));

        // The system prompt advertises the skill (name + description).
        let system = engine.messages[0]["content"].as_str().unwrap();
        assert!(system.contains("demo"));
        assert!(system.contains("does a demo"));
    }

    #[test]
    fn no_skill_tool_without_skills() {
        let engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let tool_names: Vec<&str> = engine
            .tools_openai
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        assert!(!tool_names.contains(&"skill"));
    }

    #[test]
    fn agent_tools_off_strips_system_prompt() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.push_text_turn("user", "hi".into());

        assert!(engine.agent_tools_enabled);
        assert_eq!(role(&engine.outgoing_messages()[0]), "system");

        engine.set_agent_tools_enabled(false);
        let out = engine.outgoing_messages();
        assert!(out.iter().all(|m| role(m) != "system"));
        assert_eq!(role(&out[0]), "user");

        engine.set_agent_tools_enabled(true);
        assert_eq!(role(&engine.outgoing_messages()[0]), "system");
    }

    #[test]
    fn restrict_read_only_hides_mutating_tools() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.restrict_read_only();
        assert!(engine.read_only);
        let names: Vec<&str> = engine
            .tools_openai
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        // Mutating built-ins + subagent are stripped; read-only ones + plan/notes remain.
        for gone in [
            "write_file",
            "edit_file",
            "multi_edit",
            "run_bash",
            "subagent",
        ] {
            assert!(
                !names.contains(&gone),
                "{gone} should be hidden in plan mode"
            );
        }
        for kept in ["read_file", "grep", "glob", "list_dir", "update_plan"] {
            assert!(names.contains(&kept), "{kept} should remain in plan mode");
        }
    }

    #[test]
    fn default_reasoning_effort_gates_on_model_capability() {
        // Reasoning-capable models (snapshot `r` flag) get an effort to send…
        for model in ["o3", "gpt-5", "claude-sonnet-4-5", "gemini-2.5-pro"] {
            assert_eq!(
                default_reasoning_effort(model).as_deref(),
                Some("medium"),
                "model={model} should request reasoning"
            );
        }
        // …non-reasoning models and unknown ids never send it (would 400 strict providers).
        for model in [
            "gpt-4o",
            "claude-3-5-sonnet",
            "definitely-not-a-real-model-xyz",
        ] {
            assert_eq!(
                default_reasoning_effort(model),
                None,
                "model={model} must not request reasoning"
            );
        }
    }

    #[test]
    fn thinking_request_tracks_capability_when_enabled() {
        // Reasoning-capable model: the level is always requested; `/effort` changes it.
        let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
        assert_eq!(engine.thinking_request(), (Some("medium"), false));
        engine.set_reasoning_effort("high".into());
        assert_eq!(engine.thinking_request(), (Some("high"), false));

        // Non-reasoning model: never requested.
        let plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
        assert_eq!(plain.thinking_request(), (None, false));
    }

    #[test]
    fn thinking_request_disables_per_provider_disable_form() {
        // gpt-5 / o-series reject `"none"` alongside tools and reject `thinking` → family effort floor.
        let mut g5 = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
        g5.set_thinking_enabled(false);
        assert_eq!(g5.thinking_request(), (Some("minimal"), false));
        let mut o = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
        o.set_thinking_enabled(false);
        assert_eq!(o.thinking_request(), (Some("low"), false));

        // A catalog that lists `none` → send it (a real effort-level off).
        let mut has_none = AgentEngine::new("/tmp", "deepseek-reasoner", "", &[], &[], 0, 0);
        has_none.set_reasoning_efforts(vec!["none".into(), "low".into(), "high".into()]);
        has_none.set_thinking_enabled(false);
        assert_eq!(has_none.thinking_request(), (Some("none"), false));

        // gpt-5.4 lists `none` but not `minimal` → catalog wins (c5d6b17 regression).
        let mut g54 = AgentEngine::new("/tmp", "gpt-5.4", "", &[], &[], 0, 0);
        g54.set_reasoning_efforts(
            ["none", "low", "medium", "high", "xhigh"]
                .map(String::from)
                .to_vec(),
        );
        g54.set_thinking_enabled(false);
        assert_eq!(g54.thinking_request(), (Some("none"), false));

        // codex advertises only low/medium/high → its `low` floor, not `minimal`.
        let mut codex = AgentEngine::new("/tmp", "gpt-5-codex", "", &[], &[], 0, 0);
        codex.set_reasoning_efforts(["low", "medium", "high"].map(String::from).to_vec());
        codex.set_thinking_enabled(false);
        assert_eq!(codex.thinking_request(), (Some("low"), false));

        // Effort scale with no off (aivo/starter, snapshot-absent): emit the `thinking` disable field, not an invalid `"none"` effort.
        let mut alias = AgentEngine::new("/tmp", "aivo/starter", "", &[], &[], 0, 0);
        assert!(
            !alias.reasoning_capable,
            "alias is absent from the snapshot"
        );
        alias.set_reasoning_efforts(vec![
            "low".into(),
            "medium".into(),
            "high".into(),
            "xhigh".into(),
            "max".into(),
        ]);
        alias.set_thinking_enabled(false);
        assert_eq!(alias.thinking_request(), (None, true));

        // Snapshot-known Anthropic model (no none/minimal): the `thinking` field, carried by the bridge.
        let mut claude = AgentEngine::new("/tmp", "claude-sonnet-4-5", "", &[], &[], 0, 0);
        claude.set_thinking_enabled(false);
        assert_eq!(claude.thinking_request(), (None, true));

        // Genuinely non-reasoning model with no catalog level: stay silent.
        let mut plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
        plain.set_thinking_enabled(false);
        assert_eq!(plain.thinking_request(), (None, false));
    }

    /// Full loop: first turn emits a write_file call, second turn answers with text → converges.
    #[tokio::test]
    async fn engine_runs_tool_then_converges() {
        let dir = tmp();
        let port = spawn_sse_sequence(vec![WRITE_TOOL_SSE.to_string(), FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("write out.txt".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.tools, vec!["write_file"]);
        assert_eq!(ui.text, "done");
        assert!(dir.join("out.txt").exists());
        assert_eq!(std::fs::read_to_string(dir.join("out.txt")).unwrap(), "hi");
    }

    #[tokio::test]
    async fn leaked_tool_call_markup_is_stripped_and_nudged() {
        let dir = tmp();
        let leaked = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
        );
        let port = spawn_sse_sequence(vec![leaked, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read the file".into()),
            &mut ui,
        )
        .await;

        // Nudge is its own `user` message, preceded by an `assistant` turn.
        let nudge_idx = engine
            .messages
            .iter()
            .position(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("wrote tool calls as plain text"))
            })
            .expect("expected a leaked-tool-call nudge in history");
        assert_eq!(engine.messages[nudge_idx]["role"], "user");
        assert_eq!(engine.messages[nudge_idx - 1]["role"], "assistant");
        assert_no_consecutive_user(&engine.messages);
        assert!(
            !engine.messages.iter().any(|m| m["content"]
                .as_str()
                .is_some_and(|c| c.contains("<tool_calls>"))),
            "leaked markup should be stripped from history"
        );
        let last = engine.messages.last().unwrap();
        assert_eq!(last["role"], "assistant");
        assert_eq!(last["content"], "done");
        assert_eq!(
            ui.discards, 1,
            "engine must drop the leaked streamed segment"
        );
        assert_eq!(ui.text, "done");
    }

    /// Regression: a leak after a tool step must not produce a user-after-tool 400.
    #[tokio::test]
    async fn leaked_tool_call_after_tool_step_keeps_roles_alternating() {
        let dir = tmp();
        let bash = tool_call_sse("run_bash", json!({ "command": "echo hi" }));
        let leaked = format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
        );
        let port = spawn_sse_sequence(vec![bash, leaked, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("run it".into()),
            &mut ui,
        )
        .await;

        assert_no_consecutive_user(&engine.messages);
        for i in 1..engine.messages.len() {
            if role(&engine.messages[i]) == "user" {
                assert_ne!(
                    role(&engine.messages[i - 1]),
                    "tool",
                    "a user message directly after tool results bricks the Anthropic bridge"
                );
            }
        }
        assert_eq!(engine.messages.last().unwrap()["content"], "done");
    }

    #[tokio::test]
    async fn leaked_tool_call_nudges_are_capped() {
        let dir = tmp();
        let leaked = || {
            format!(
                "data: {}\n\ndata: [DONE]\n\n",
                json!({"choices":[{"delta":{"content":"<tool_calls>{\"name\":\"read_file\"}</tool_calls>"}}]})
            )
        };
        let port = spawn_sse_sequence(vec![leaked(), leaked(), leaked(), leaked()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read it".into()),
            &mut ui,
        )
        .await;

        let nudges = engine
            .messages
            .iter()
            .filter(|m| {
                m["content"]
                    .as_str()
                    .is_some_and(|c| c.contains("wrote tool calls as plain text"))
            })
            .count();
        assert_eq!(nudges, MAX_LEAKED_NUDGES, "nudges must be capped");
        assert_no_consecutive_user(&engine.messages);
    }

    /// `rm -rf /` prompts even with auto-approve on; the mock denies so it never runs.
    #[tokio::test]
    async fn catastrophic_command_prompts_even_under_auto_approve() {
        let dir = tmp();
        let bash = tool_call_sse("run_bash", json!({ "command": "rm -rf /" }));
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi {
            deny: true, // never let a real `rm -rf /` execute
            ..Default::default()
        };
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("clean up".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.ask_tools, vec!["run_bash"]);
    }

    /// A remote mutation (`gh repo delete`) prompts even with auto-approve on; the
    /// mock denies, so the outward-facing action never runs.
    #[tokio::test]
    async fn remote_side_effect_prompts_even_under_auto_approve() {
        let dir = tmp();
        let bash = tool_call_sse(
            "run_bash",
            json!({ "command": "gh repo delete acme/prod --yes" }),
        );
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        // turn_ctx sets yes:true (auto-approve); deny so the delete never fires.
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("remove the prod repo".into()),
            &mut ui,
        )
        .await;

        // Prompted despite auto-approve (deny keeps the delete from running).
        assert_eq!(ui.ask_tools, vec!["run_bash"]);
    }

    /// A paging loop varying an ignored arg (`limit`) makes a distinct `batch_sig` each
    /// step, so only the page-read guard can stop it — the read_file runaway shape.
    #[tokio::test]
    async fn paging_loop_with_varying_junk_args_is_stopped() {
        let dir = tmp();
        std::fs::write(dir.join("big.txt"), "x\n".repeat(200)).unwrap();
        let mut seq: Vec<String> = (0..8)
            .map(|i| {
                tool_call_sse(
                    "read_file",
                    json!({ "path": "big.txt", "offset": 1, "limit": 10 + i }),
                )
            })
            .collect();
        seq.push(FINAL_TEXT_SSE.to_string());
        let port = spawn_sse_sequence(seq);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read the file".into()),
            &mut ui,
        )
        .await;
        let reads = ui
            .tools
            .iter()
            .filter(|t| t.as_str() == "read_file")
            .count();
        assert!(
            reads <= REPEAT_LIMIT,
            "page guard should stop the loop; ran {reads} reads"
        );
        assert!(
            ui.notices
                .iter()
                .any(|n| n.contains("repeated the same action"))
        );
    }

    /// Reading `.env` is gated: with auto-approve off the card fires, and denying it
    /// blocks the read so the key never enters the transcript.
    #[tokio::test]
    async fn reading_dotenv_prompts_and_deny_blocks_it() {
        let dir = tmp();
        std::fs::write(
            dir.join(".env"),
            "OPENAI_API_KEY=sk-AAAAAAAAAAAAAAAAAAAAAAAA\n",
        )
        .unwrap();
        let read = tool_call_sse("read_file", json!({ "path": ".env" }));
        let port = spawn_sse_sequence(vec![read, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        // Auto-approve OFF so the consent gate engages; deny the read.
        let ctx = TurnCtx {
            client: &client,
            serve_base: &base,
            auth: None,
            cwd: dir.as_path(),
            yes: false,
            auto_approve: None,
        };
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        run_session(&mut engine, &ctx, Some("read env".into()), &mut ui).await;
        assert_eq!(ui.ask_tools, vec!["read_file"]);
        let tool_content: String = engine
            .messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(tool_content.contains("denied by user"));
        assert!(!tool_content.contains("sk-AAAAAAAAAAAAAAAAAAAAAAAA"));
    }

    /// A key-shaped string in a tool result is masked before it reaches the model.
    #[tokio::test]
    async fn secret_values_are_redacted_from_the_transcript() {
        let dir = tmp();
        std::fs::write(dir.join("notes.txt"), "deploy key AKIAIOSFODNN7EXAMPLE\n").unwrap();
        let read = tool_call_sse("read_file", json!({ "path": "notes.txt" }));
        let port = spawn_sse_sequence(vec![read, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read notes".into()),
            &mut ui,
        )
        .await;
        let tool_content: String = engine
            .messages
            .iter()
            .filter(|m| m["role"] == "tool")
            .filter_map(|m| m["content"].as_str())
            .collect();
        assert!(
            tool_content.contains("<redacted:aws_access_key>"),
            "got: {tool_content}"
        );
        assert!(!tool_content.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(ui.notices.iter().any(|n| n.contains("redacted")));
    }

    /// A vision turn keeps the tool loop: the image rides in the opening message while
    /// tools still run.
    #[tokio::test]
    async fn run_turn_with_content_keeps_image_and_runs_tools() {
        let dir = tmp();
        let ls = tool_call_sse("list_dir", json!({ "path": "." }));
        let port = spawn_sse_sequence(vec![ls, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let content = json!([
            {"type": "text", "text": "what's in this screenshot?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAABBBBCCCC"}},
        ]);
        let mut ui = CapturingUi::default();
        engine
            .run_turn_with_content(
                &turn_ctx(&client, &base, &dir),
                &mut ui,
                content,
                "what's in this screenshot?".into(),
            )
            .await;
        assert!(ui.tools.contains(&"list_dir".to_string()));
        let user = engine
            .messages
            .iter()
            .find(|m| m["role"] == "user")
            .expect("a user message");
        let parts = user["content"].as_array().expect("array content");
        assert!(parts.iter().any(|p| p["type"] == "image_url"));
    }

    #[test]
    fn push_user_content_never_makes_consecutive_user_turns() {
        let dir = tmp();
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        engine.push_user_content(Value::String("first".into()));
        engine.push_user_content(json!([
            {"type": "text", "text": "second"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]));
        let users: Vec<_> = engine
            .messages
            .iter()
            .filter(|m| m["role"] == "user")
            .collect();
        assert_eq!(
            users.len(),
            1,
            "the image turn must fold into the trailing user turn"
        );
        let parts = users[0]["content"].as_array().unwrap();
        assert!(parts.iter().any(|p| p["type"] == "image_url"));
        assert!(
            parts
                .iter()
                .any(|p| p.get("text").and_then(|t| t.as_str()) == Some("first"))
        );
    }

    /// Contrast: a workspace-local `rm -rf ./build` isn't in the floor, so auto-approve waives it (path absent → no-op).
    #[tokio::test]
    async fn auto_approve_waives_workspace_local_destructive() {
        let dir = tmp();
        let bash = tool_call_sse(
            "run_bash",
            json!({ "command": "rm -rf ./build_does_not_exist" }),
        );
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("clean build dir".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.asks, 0);
        assert!(ui.tools.contains(&"run_bash".to_string()));
    }

    /// A sandbox-blocked `run_bash` (write outside the workspace) prompts to re-run
    /// outside, scoped to `run_bash_unsandboxed`; declining keeps the blocked result. macOS-only.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn sandbox_block_prompts_to_run_unsandboxed_and_respects_deny() {
        if !crate::agent::sandbox::active() {
            return;
        }
        let dir = tmp();
        let home = crate::services::system_env::home_dir().unwrap();
        let outside = home.join(format!("aivo_esc_deny_{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&outside);
        let cmd = format!("echo escalated > '{}'", outside.display());
        let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let ctx = TurnCtx {
            client: &client,
            serve_base: &base,
            auth: None,
            cwd: &dir,
            yes: false,
            auto_approve: None,
        };
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        run_session(&mut engine, &ctx, Some("commit".into()), &mut ui).await;

        let existed = outside.exists();
        let _ = std::fs::remove_file(&outside);
        assert_eq!(ui.ask_tools, vec!["run_bash_unsandboxed"]);
        // Declined → never ran unconfined, so the file was never written…
        assert!(
            !existed,
            "denied escalation still wrote outside the workspace"
        );
        // …and no re-run notice was emitted.
        assert!(
            !ui.notices
                .iter()
                .any(|n| n.contains("outside the workspace sandbox")),
            "unexpected re-run notice on deny: {:?}",
            ui.notices
        );
    }

    /// Approving the escalation re-runs outside the sandbox, so the blocked out-of-workspace write now succeeds.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn sandbox_block_reruns_outside_when_approved() {
        if !crate::agent::sandbox::active() {
            return;
        }
        let dir = tmp();
        let home = crate::services::system_env::home_dir().unwrap();
        let outside = home.join(format!("aivo_esc_allow_{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&outside);
        let cmd = format!("echo escalated > '{}'", outside.display());
        let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let ctx = TurnCtx {
            client: &client,
            serve_base: &base,
            auth: None,
            cwd: &dir,
            yes: false,
            auto_approve: None,
        };
        let mut ui = CapturingUi {
            always_allow: true,
            ..Default::default()
        };
        run_session(&mut engine, &ctx, Some("commit".into()), &mut ui).await;

        let existed = outside.exists();
        let contents = std::fs::read_to_string(&outside).unwrap_or_default();
        let _ = std::fs::remove_file(&outside);
        assert_eq!(ui.ask_tools, vec!["run_bash_unsandboxed"]);
        assert!(
            existed,
            "approved escalation did not write outside the workspace"
        );
        assert_eq!(contents.trim(), "escalated");
        assert!(
            ui.notices
                .iter()
                .any(|n| n.contains("outside the workspace sandbox")),
            "missing re-run notice: {:?}",
            ui.notices
        );
    }

    /// A batch of read-only calls runs concurrently but results stay in call order, each paired to its `tool_call_id`.
    #[tokio::test]
    async fn parallel_read_batch_preserves_order_and_pairing() {
        let dir = tmp();
        std::fs::write(dir.join("a.txt"), "ALPHA").unwrap();
        std::fs::write(dir.join("b.txt"), "BETA").unwrap();
        std::fs::write(dir.join("c.txt"), "GAMMA").unwrap();
        let batch = batch_tool_call_sse(&[
            ("c0", "read_file", json!({"path": "a.txt"})),
            ("c1", "read_file", json!({"path": "b.txt"})),
            ("c2", "read_file", json!({"path": "c.txt"})),
        ]);
        let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read all three".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.tools, vec!["read_file", "read_file", "read_file"]);
        // Results in call order, each keyed to the right id and content.
        let tool_msgs: Vec<(&str, &str)> = engine
            .messages
            .iter()
            .filter(|m| role(m) == "tool")
            .map(|m| {
                (
                    m["tool_call_id"].as_str().unwrap(),
                    m["content"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(tool_msgs.len(), 3);
        assert_eq!(tool_msgs[0].0, "c0");
        assert!(tool_msgs[0].1.contains("ALPHA"));
        assert_eq!(tool_msgs[1].0, "c1");
        assert!(tool_msgs[1].1.contains("BETA"));
        assert_eq!(tool_msgs[2].0, "c2");
        assert!(tool_msgs[2].1.contains("GAMMA"));
        assert_eq!(ui.text, "done");
    }

    /// A mixed batch (parallel read + ordered write) records results in call order and the write lands.
    #[tokio::test]
    async fn mixed_batch_orders_results_and_runs_write() {
        let dir = tmp();
        std::fs::write(dir.join("a.txt"), "ALPHA").unwrap();
        let batch = batch_tool_call_sse(&[
            ("c0", "read_file", json!({"path": "a.txt"})),
            (
                "c1",
                "write_file",
                json!({"path": "out.txt", "content": "WROTE"}),
            ),
        ]);
        let port = spawn_sse_sequence(vec![batch, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("read then write".into()),
            &mut ui,
        )
        .await;

        let tool_msgs: Vec<(&str, &str)> = engine
            .messages
            .iter()
            .filter(|m| role(m) == "tool")
            .map(|m| {
                (
                    m["tool_call_id"].as_str().unwrap(),
                    m["content"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(tool_msgs.len(), 2);
        assert_eq!(tool_msgs[0].0, "c0");
        assert!(tool_msgs[0].1.contains("ALPHA"));
        assert_eq!(tool_msgs[1].0, "c1");
        assert!(dir.join("out.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("out.txt")).unwrap(),
            "WROTE"
        );
    }

    /// An empty completion converges the turn but isn't recorded as an assistant message (empty → invalid Anthropic content array).
    #[tokio::test]
    async fn empty_completion_is_not_recorded_as_assistant_turn() {
        let dir = tmp();
        let empty = "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n".to_string();
        let port = spawn_sse_sequence(vec![empty]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("hi".into()),
            &mut ui,
        )
        .await;

        assert!(
            !engine.messages.iter().any(|m| role(m) == "assistant"),
            "empty completion must not record an assistant turn: {:?}",
            engine.messages
        );
        // The turn still ran (user message recorded).
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "user" && content_str(m) == "hi")
        );
    }

    /// A denied dangerous tool (destructive bash) doesn't run; the refusal feeds back and the next turn converges.
    #[tokio::test]
    async fn denied_dangerous_tool_does_not_run() {
        let dir = tmp();
        let sentinel = dir.join("RAN");
        // `rm -rf` makes this dangerous → gated; if it ran it would touch RAN.
        let cmd = format!("rm -rf zzz_absent && touch {}", sentinel.display());
        let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
        let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

        assert_eq!(ui.tools, vec!["run_bash"]);
        assert!(!sentinel.exists(), "denied command still ran");
    }

    /// "Always allow" remembers the exact command, not the whole tool — a different
    /// destructive command prompts again. Unix-only (uses `rm -rf … && touch …`); the logic is platform-agnostic.
    #[cfg(unix)]
    #[tokio::test]
    async fn always_allow_is_scoped_to_the_exact_command() {
        let dir = tmp();
        let (sa, sb) = (dir.join("RAN_A"), dir.join("RAN_B"));
        let cmd_a = format!("rm -rf zzz_a && touch {}", sa.display());
        let cmd_b = format!("rm -rf zzz_b && touch {}", sb.display());
        // Steps in one turn: A, A again, a different B, then text.
        let port = spawn_sse_sequence(vec![
            tool_call_sse("run_bash", json!({ "command": cmd_a })),
            tool_call_sse("run_bash", json!({ "command": cmd_a })),
            tool_call_sse("run_bash", json!({ "command": cmd_b })),
            FINAL_TEXT_SSE.to_string(),
        ]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        let mut ui = CapturingUi {
            always_allow: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

        // A prompts once (repeat reuses the scope); B is different → two asks total.
        assert_eq!(ui.asks, 2, "expected A once + B once");
        assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
        assert!(sa.exists(), "command A did not run");
        assert!(sb.exists(), "command B did not run");
    }

    /// "Always allow" on a subcommand-style command (`git reset`) covers the whole
    /// family for the session, so a varied re-run isn't re-prompted — while a
    /// different subcommand still asks. (Commands are no-ops in a non-repo tmp dir.)
    #[tokio::test]
    async fn always_allow_broadens_a_subcommand_family_but_not_siblings() {
        let dir = tmp();
        // All destructive (so they prompt); `git` is a subcommand tool (so it broadens).
        let port = spawn_sse_sequence(vec![
            tool_call_sse("run_bash", json!({ "command": "git reset --hard HEAD~1" })),
            tool_call_sse("run_bash", json!({ "command": "git reset --hard HEAD~2" })),
            tool_call_sse("run_bash", json!({ "command": "git clean -fd" })),
            FINAL_TEXT_SSE.to_string(),
        ]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        let mut ui = CapturingUi {
            always_allow: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

        // `git reset` approved once covers the second reset; `git clean` is a different
        // subcommand → one more ask. Two asks total, all three commands attempted.
        assert_eq!(ui.asks, 2, "reset once (family reused) + clean once");
        assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
    }

    #[test]
    fn write_clobbers_unread_only_flags_blind_overwrites() {
        let dir = tmp();
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        // A new file is not a clobber.
        assert!(!engine.write_clobbers_unread("write_file", &json!({"path":"new.txt"}), &dir));
        // An existing file the model never touched IS a blind clobber.
        std::fs::write(dir.join("exists.txt"), "old").unwrap();
        assert!(engine.write_clobbers_unread("write_file", &json!({"path":"exists.txt"}), &dir));
        // Once read, overwriting it is fine.
        engine.record_touched_file("read_file", &json!({"path":"exists.txt"}));
        assert!(!engine.write_clobbers_unread("write_file", &json!({"path":"exists.txt"}), &dir));
        // edit_file / multi_edit are never blind (they read to match).
        assert!(!engine.write_clobbers_unread("edit_file", &json!({"path":"exists.txt"}), &dir));
    }

    /// A `write_file` overwriting a pre-existing unread file is gated; denying leaves it intact.
    #[tokio::test]
    async fn blind_overwrite_of_existing_file_is_gated() {
        let dir = tmp();
        std::fs::write(dir.join("precious.txt"), "USER DATA").unwrap();
        let sse = tool_call_sse(
            "write_file",
            json!({"path":"precious.txt","content":"CLOBBERED"}),
        );
        let port = spawn_sse_sequence(vec![sse, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("overwrite it".into()), &mut ui).await;

        assert_eq!(ui.asks, 1, "a blind overwrite should prompt");
        assert_eq!(
            std::fs::read_to_string(dir.join("precious.txt")).unwrap(),
            "USER DATA",
            "denied overwrite must leave the file untouched"
        );
    }

    /// A safe mutating tool (in-project write) runs WITHOUT a prompt even when the UI would deny — only dangerous actions are gated.
    #[tokio::test]
    async fn safe_tool_runs_without_prompt() {
        let dir = tmp();
        let port = spawn_sse_sequence(vec![WRITE_TOOL_SSE.to_string(), FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(
            &dir.display().to_string(),
            "m",
            "2026-01-01",
            &[],
            &[],
            0,
            0,
        );
        // deny=true would block anything that asked — but a safe write never asks.
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("write out.txt".into()), &mut ui).await;

        assert_eq!(ui.tools, vec!["write_file"]);
        assert!(dir.join("out.txt").exists(), "safe write was blocked");
    }

    /// An `update_plan` call is intercepted by the engine: it drives the plan
    /// card (`plan_updated`), is NOT rendered as a generic tool step, and feeds a
    /// confirmation back so the conversation converges on the next turn.
    #[tokio::test]
    async fn engine_handles_update_plan() {
        let dir = tmp();
        let plan = tool_call_sse(
            "update_plan",
            json!({"plan": [
                {"step": "read", "status": "completed"},
                {"step": "edit", "status": "in_progress"}
            ]}),
        );
        let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("do the thing".into()),
            &mut ui,
        )
        .await;

        // Model's update (2 steps), then the engine's finalization on convergence.
        assert_eq!(ui.plans, vec![2, 2], "plan_updated should fire twice");
        assert_eq!(
            ui.last_plan,
            vec![PlanStatus::Completed, PlanStatus::Completed],
            "a converged turn finalizes a started plan to all-completed"
        );
        assert!(
            !ui.tools.contains(&"update_plan".to_string()),
            "update_plan must not render as a generic tool step"
        );
        assert_eq!(ui.text, "done");
        // The tool result was fed back into history (call ↔ result invariant).
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "tool" && content_str(m).contains("Plan updated")),
            "missing update_plan confirmation in history"
        );
    }

    /// A started plan the model never finished is finalized by the engine on convergence (the "0/N done" stuck-card bug).
    #[tokio::test]
    async fn engine_finalizes_started_plan_on_convergence() {
        let dir = tmp();
        let plan = tool_call_sse(
            "update_plan",
            json!({"plan": [
                {"step": "investigate", "status": "in_progress"},
                {"step": "fix", "status": "pending"},
                {"step": "verify", "status": "pending"}
            ]}),
        );
        let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("do the thing".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.plans, vec![3, 3]);
        assert_eq!(
            ui.last_plan,
            vec![
                PlanStatus::Completed,
                PlanStatus::Completed,
                PlanStatus::Completed
            ],
            "every step is completed once the turn converges"
        );
    }

    /// An all-pending plan means the model planned but converged WITHOUT executing;
    /// the `started` gate must not fabricate completion.
    #[tokio::test]
    async fn engine_leaves_unstarted_plan_alone_on_convergence() {
        let dir = tmp();
        let plan = tool_call_sse(
            "update_plan",
            json!({"plan": [
                {"step": "a", "status": "pending"},
                {"step": "b", "status": "pending"}
            ]}),
        );
        let port = spawn_sse_sequence(vec![plan, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("plan only".into()),
            &mut ui,
        )
        .await;

        // Only the model's event fired — no engine finalization.
        assert_eq!(ui.plans, vec![2]);
        assert_eq!(ui.last_plan, vec![PlanStatus::Pending, PlanStatus::Pending]);
    }

    /// A `subagent` call spawns a fresh sub-engine; its text result feeds back as the parent's tool result and the parent converges.
    #[tokio::test]
    async fn engine_runs_subagent_and_returns_result() {
        let dir = tmp();
        let call = tool_call_sse("subagent", json!({"task": "investigate the bug"}));
        let sub_text =
            "data: {\"choices\":[{\"delta\":{\"content\":\"subresult\"}}]}\n\ndata: [DONE]\n\n"
                .to_string();
        let port = spawn_sse_sequence(vec![call, sub_text, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("delegate it".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.tools, vec!["subagent"]);
        assert_eq!(ui.text, "done");
        // The sub-agent's answer came back as the parent's tool result.
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "tool" && content_str(m).contains("subresult")),
            "sub-agent result missing from parent history"
        );
    }

    /// An external source's schemas are offered and a call routes to the source, not the built-in executor (mock, no real MCP subprocess).
    #[tokio::test]
    async fn external_tools_are_offered_and_routed() {
        struct MockExt;
        impl crate::agent::engine::ExternalTools for MockExt {
            fn specs(&self) -> Vec<Value> {
                vec![json!({
                    "type": "function",
                    "function": {"name": "mcp__demo__ping", "description": "d", "parameters": {"type": "object"}}
                })]
            }
            fn handles(&self, name: &str) -> bool {
                name == "mcp__demo__ping"
            }
            fn call<'a>(
                &'a self,
                _name: &'a str,
                _args: &'a Value,
            ) -> BoxFuture<'a, Result<String, String>> {
                Box::pin(async { Ok("pong".to_string()) })
            }
        }

        let dir = tmp();
        // Turn 1: the model calls the external tool; turn 2: it converges.
        let call = tool_call_sse("mcp__demo__ping", json!({}));
        let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        engine.set_external_tools(std::sync::Arc::new(MockExt));
        // The external schema is advertised alongside the built-ins.
        assert!(
            engine
                .tools_openai
                .iter()
                .any(|t| t["function"]["name"] == "mcp__demo__ping")
        );

        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("use the tool".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.tools, vec!["mcp__demo__ping"]);
        assert_eq!(ui.text, "done");
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "tool" && content_str(m) == "pong"),
            "external tool result not routed back"
        );
    }

    /// The model calls `take_note`; the engine stores it (no prompt, no `tools::execute`), echoes a confirmation, retains it for pinning.
    #[tokio::test]
    async fn take_note_is_dispatched_and_stored() {
        let dir = tmp();
        let call = tool_call_sse("take_note", json!({"note": "the parser is in lexer.rs"}));
        let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        // `take_note` is advertised alongside the built-ins.
        assert!(
            engine
                .tools_openai
                .iter()
                .any(|t| t["function"]["name"] == "take_note")
        );

        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("remember where the parser is".into()),
            &mut ui,
        )
        .await;

        assert_eq!(engine.notes, vec!["the parser is in lexer.rs".to_string()]);
        // No permission prompt was raised for the note.
        assert_eq!(ui.asks, 0);
        // The confirmation came back as the tool result.
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "tool" && content_str(m).starts_with("Noted (1 saved)")),
            "note confirmation not echoed back"
        );
    }

    /// An external source that `requires_approval` is gated: a denying UI refuses the call (not executed).
    #[tokio::test]
    async fn external_tool_requiring_approval_is_gated() {
        struct GatedExt;
        impl crate::agent::engine::ExternalTools for GatedExt {
            fn specs(&self) -> Vec<Value> {
                vec![json!({
                    "type": "function",
                    "function": {"name": "mcp__risky__wipe", "description": "d", "parameters": {"type": "object"}}
                })]
            }
            fn handles(&self, name: &str) -> bool {
                name == "mcp__risky__wipe"
            }
            fn requires_approval(&self, _name: &str) -> bool {
                true
            }
            fn call<'a>(
                &'a self,
                _name: &'a str,
                _args: &'a Value,
            ) -> BoxFuture<'a, Result<String, String>> {
                Box::pin(async { Ok("WIPED".to_string()) })
            }
        }

        let dir = tmp();
        let call = tool_call_sse("mcp__risky__wipe", json!({}));
        let port = spawn_sse_sequence(vec![call, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        engine.set_external_tools(std::sync::Arc::new(GatedExt));
        let mut ui = CapturingUi {
            deny: true,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: false,
            ..turn_ctx(&client, &base, &dir)
        };
        run_session(&mut engine, &ctx, Some("wipe it".into()), &mut ui).await;

        assert_eq!(ui.tools, vec!["mcp__risky__wipe"]);
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "tool" && content_str(m).contains("denied")),
            "gated external call should have been denied"
        );
        assert!(
            !engine.messages.iter().any(|m| content_str(m) == "WIPED"),
            "denied tool must not have executed"
        );
    }

    /// The sub-agent UI recovers an answer emitted in the same step as the final tool call, instead of losing it.
    #[test]
    fn subagent_ui_recovers_answer_before_final_tool() {
        let mut ui = SubagentUi::default();
        // Step 1: the model gives its answer AND calls a tool in the same step.
        ui.turn_start();
        ui.assistant_text("The answer is 42.");
        ui.tool_start("run_bash", &json!({}));
        ui.tool_result("run_bash", &Ok("ok".to_string()));
        // Step 2: converges with no text of its own.
        ui.turn_start();
        ui.footer(None, 2, 0, 0, 0);
        assert_eq!(ui.answer(), "The answer is 42.");

        // Normal case: the converging step carries the answer.
        let mut ui2 = SubagentUi::default();
        ui2.turn_start();
        ui2.assistant_text("plain answer");
        ui2.footer(None, 1, 0, 0, 0);
        assert_eq!(ui2.answer(), "plain answer");
    }

    /// A sub-agent's token usage folds into the parent turn's total (the sub's LLM calls aren't parent steps).
    #[tokio::test]
    async fn subagent_tokens_fold_into_parent_total() {
        let dir = tmp();
        let call = tool_call_sse("subagent", json!({"task": "investigate"}));
        // The sub-agent's turn reports 100 tokens of usage.
        let sub_text = "data: {\"choices\":[{\"delta\":{\"content\":\"subresult\"}}],\"usage\":{\"total_tokens\":100}}\n\ndata: [DONE]\n\n".to_string();
        let port = spawn_sse_sequence(vec![call, sub_text, FINAL_TEXT_SSE.to_string()]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("delegate it".into()),
            &mut ui,
        )
        .await;

        // The parent's own steps report no usage, so the 100 came from the sub-agent.
        assert!(
            ui.footer_tokens >= 100,
            "sub-agent tokens not folded into the parent total: {}",
            ui.footer_tokens
        );
    }

    /// The engine sums each step's provider-measured token split across a turn and surfaces it via `take_turn_usage`.
    #[tokio::test]
    async fn turn_usage_accumulates_split_across_steps() {
        let dir = tmp();
        // Step 1: a tool call that ALSO reports usage with a cache split.
        let step1 = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_bash\",\"arguments\":\"{\\\"command\\\":\\\"echo hi\\\"}\"}}]}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"prompt_tokens_details\":{\"cached_tokens\":40}}}\n\ndata: [DONE]\n\n".to_string();
        // Step 2: the converging text reply, reporting more usage.
        let step2 = "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}],\"usage\":{\"prompt_tokens\":200,\"completion_tokens\":30}}\n\ndata: [DONE]\n\n".to_string();
        let port = spawn_sse_sequence(vec![step1, step2]);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("go".into()),
            &mut ui,
        )
        .await;

        let usage = engine.take_turn_usage();
        assert_eq!(usage.prompt_tokens, 300, "prompt summed across steps");
        assert_eq!(
            usage.completion_tokens, 50,
            "completion summed across steps"
        );
        assert_eq!(usage.cache_read_tokens, 40, "cached tokens captured");
        // Draining leaves the accumulator zeroed for the next turn.
        assert_eq!(engine.take_turn_usage(), SessionTokens::default());
    }

    /// When a sub-agent produces no answer, the failure reason is surfaced instead of a vague "no answer".
    #[test]
    fn subagent_ui_surfaces_failure_notice_when_no_answer() {
        let mut ui = SubagentUi::default();
        ui.turn_start();
        ui.notify("reached the step limit (20)");
        // No assistant text emitted → no answer.
        let msg = ui.result_message();
        assert!(msg.contains("no answer"), "got: {msg}");
        assert!(msg.contains("step limit"), "failure reason missing: {msg}");

        // With an answer, the notice is ignored.
        let mut ui2 = SubagentUi::default();
        ui2.turn_start();
        ui2.assistant_text("the result is 42");
        ui2.notify("compacting context…");
        let msg2 = ui2.result_message();
        assert!(msg2.contains("the result is 42"));
        assert!(
            !msg2.contains("compacting"),
            "notice leaked into a good answer"
        );
    }

    #[test]
    fn subagent_tool_offered_and_droppable_for_recursion_guard() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let has_subagent = |e: &AgentEngine| {
            e.tools_openai
                .iter()
                .any(|t| t["function"]["name"].as_str() == Some("subagent"))
        };
        assert!(has_subagent(&engine), "top-level engine offers subagent");
        engine.drop_subagent_tool();
        assert!(
            !has_subagent(&engine),
            "sub-engine must not offer subagent (no recursion)"
        );
    }

    // Unix-only: the mock's raw sequential-`accept()` sequencing is fragile on Windows; the retry-past-503 logic is platform-agnostic.
    #[cfg(unix)]
    #[tokio::test]
    async fn engine_retries_then_succeeds() {
        // First connection returns 503 (retryable, before any stream); the retry hits a 200.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf);
                let body = "overloaded";
                let resp = format!(
                    "HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
            }
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf);
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    FINAL_TEXT_SSE.len(),
                    FINAL_TEXT_SSE
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        // Make the backoff instant for the test.
        unsafe { std::env::set_var("AIVO_AGENT_RETRY_BASE_MS", "1") };
        let dir = tmp();
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let base = format!("http://127.0.0.1:{port}");
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        let mut ui = CapturingUi::default();
        run_session(
            &mut engine,
            &turn_ctx(&client, &base, &dir),
            Some("hi".into()),
            &mut ui,
        )
        .await;

        assert_eq!(ui.text, "done"); // retried past the 503 and got the content
        assert!(
            ui.notices.iter().any(|n| n.contains("retrying")),
            "expected a retry notice, got {:?}",
            ui.notices
        );
    }

    #[test]
    fn parse_slash_classifies() {
        assert!(parse_slash("hello").is_none());
        assert!(matches!(parse_slash("/help"), Some(SlashCmd::Help)));
        assert!(matches!(parse_slash("  /clear "), Some(SlashCmd::Clear)));
        assert!(matches!(parse_slash("/?"), Some(SlashCmd::Help)));
        assert!(matches!(parse_slash("/bogus"), Some(SlashCmd::Unknown(_))));
    }

    #[test]
    fn reset_keeps_only_the_system_prompt() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.messages.push(json!({"role":"user","content":"hi"}));
        engine
            .messages
            .push(json!({"role":"assistant","content":"yo"}));
        engine.reset();
        assert_eq!(engine.messages.len(), 1);
        assert_eq!(role(&engine.messages[0]), "system");
    }

    /// `set_context_window` fills an unknown (0) window (late catalog warm) but never overrides a known one.
    #[test]
    fn set_context_window_fills_only_a_missing_window() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 0; // force unknown, ignoring any env override
        engine.set_context_window(200_000);
        assert_eq!(engine.context_window, 200_000, "missing window should fill");
        engine.set_context_window(100_000);
        assert_eq!(
            engine.context_window, 200_000,
            "a known window must not change"
        );
        engine.set_context_window(0);
        assert_eq!(engine.context_window, 200_000, "a 0 update is a no-op");
    }

    /// An unknown window (0) compacts at `DEFAULT_CONTEXT_WINDOW`; a known window
    /// takes precedence.
    #[test]
    fn compaction_window_falls_back_to_default_when_unknown() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 0; // unknown, ignoring any env override
        assert_eq!(
            engine.compaction_window(),
            DEFAULT_CONTEXT_WINDOW,
            "unknown window should compact at the default backstop, not be skipped"
        );
        engine.set_context_window(500_000);
        assert_eq!(
            engine.compaction_window(),
            500_000,
            "a known window takes precedence over the default"
        );
    }

    #[test]
    fn token_calibration_deflates_compaction_budget() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 262_144;
        let raw = engine.compaction_window() - COMPACT_RESERVE;
        assert_eq!(
            engine.compaction_budget_estimate(),
            raw,
            "calibration 1.0 leaves the old window − reserve budget unchanged"
        );
        engine.token_calibration = 1.2;
        let deflated = engine.compaction_budget_estimate();
        assert_eq!(deflated, ((raw as f64) / 1.2).floor() as usize);
        assert!(
            deflated < raw,
            "calibration > 1 shrinks the estimate-space budget so denser-than-chars/4 \
             content still fits the real window"
        );
    }

    #[test]
    fn update_calibration_rises_on_undershoot_then_eases_and_clamps() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.update_calibration(100, 400);
        assert_eq!(engine.token_calibration, 1.0, "tiny request is ignored");
        engine.update_calibration(100_000, 110_000);
        assert!(
            (engine.token_calibration - 1.1).abs() < 1e-9,
            "undershoot raises calibration to the measured ratio, got {}",
            engine.token_calibration
        );
        engine.update_calibration(100_000, 100_000);
        assert!(
            engine.token_calibration > 1.0 && engine.token_calibration < 1.1,
            "calibration eases down slowly, got {}",
            engine.token_calibration
        );
        engine.update_calibration(100_000, 100_000_000);
        assert_eq!(
            engine.token_calibration, MAX_CALIBRATION,
            "ratio clamped to the ceiling"
        );
    }

    /// A tool call whose bulk is in `arguments` (empty `content`) in the irreducible recent turn must be shrunk — content-only truncation would leave it over.
    #[test]
    fn enforce_budget_shrinks_oversized_tool_call_arguments() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let huge_args = format!("{{\"content\":\"{}\"}}", "x".repeat(40_000));
        engine.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"write the file"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"write_file","arguments": huge_args}}]}),
        ];
        assert!(estimate_tokens(&engine.messages) > 300);
        engine.enforce_budget(300);
        assert!(
            estimate_tokens(&engine.messages) <= 300,
            "must fit budget by shrinking tool-call arguments, got {}",
            estimate_tokens(&engine.messages)
        );
        let tc = &engine.messages[2]["tool_calls"][0];
        assert_eq!(tc["id"], "c1", "tool_call_id preserved");
        assert_eq!(tc["function"]["name"], "write_file", "call name preserved");
        assert!(
            tc["function"]["arguments"].as_str().unwrap().len() < huge_args.len(),
            "arguments were truncated"
        );
        assert_eq!(role(&engine.messages[0]), "system", "system prompt kept");
    }

    /// A transcript whose estimate clears the raw budget but the provider rejects: calibrating from the rejection + force-fitting brings the real size under the window.
    #[test]
    fn overflow_recovery_calibrates_from_rejection_and_fits() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 262_144;
        let raw_budget = engine.compaction_window() - COMPACT_RESERVE;
        let pad = "x".repeat(4 * (raw_budget - 20_000));
        engine.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q1"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": pad}),
            json!({"role":"user","content":"now"}),
        ];
        let est = estimate_tokens(&engine.messages);
        assert!(
            est <= raw_budget,
            "pre-fix budget check would pass (no compaction): est={est} raw={raw_budget}"
        );

        let err = "token count of 290000 exceeds the maximum allowed input length of 262112 tokens";
        assert!(is_context_overflow_error(err));
        engine.recalibrate_from_overflow(err);
        assert!(
            engine.token_calibration > 1.0,
            "the rejection raised the calibration, got {}",
            engine.token_calibration
        );
        engine.force_fit_budget();

        let budget = engine.compaction_budget_estimate();
        assert!(
            estimate_tokens(&engine.messages) <= budget,
            "recovery brought the transcript under the calibrated budget"
        );
        let projected =
            (estimate_tokens(&engine.messages) as f64 * engine.token_calibration) as usize;
        assert!(
            projected <= engine.compaction_window(),
            "the calibrated real size now fits the window: projected={projected}"
        );
        assert_eq!(
            role(&engine.messages[0]),
            "system",
            "the system prompt is never dropped"
        );
    }

    /// Overflow recovery on a resumed single long turn keeps a marker for dropped work instead of vanishing to `[system, latest-user]`.
    #[test]
    fn force_fit_recovery_keeps_prior_context_on_single_long_turn() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
        let big = "reasoning ".repeat(4_000);
        let mut messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "original task: fix the shell tool"}),
        ];
        for i in 0..4 {
            messages.push(json!({"role": "assistant", "content": format!("step {i}: {big}")}));
        }
        messages.push(json!({"role": "user", "content": "continue"}));
        engine.messages = messages;

        engine.force_fit_budget();

        let budget = engine.compaction_budget_estimate();
        assert!(
            estimate_tokens(&engine.messages) <= budget,
            "recovery must fit the budget"
        );
        let last = engine.messages.last().unwrap();
        assert_eq!(role(last), "user");
        assert!(content_str(last).contains("continue"), "latest turn kept");
        assert!(
            content_str(last).contains("[Summary of earlier conversation]"),
            "dropped prior turn must leave a marker, not vanish: {}",
            content_str(last)
        );
    }

    /// A huge RECENT tool result fills the keep window; an OLDER one before the cut
    /// gets stubbed.
    #[test]
    fn compact_now_local_clears_stale_tool_output() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let recent = "r".repeat(120_000); // ~30k tokens — fills the 20k keep window
        let stale = "s".repeat(8_000); // > TOOL_RESULT_CLEAR_MIN, older than the cut
        engine.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q1"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": stale}),
            json!({"role":"user","content":"q2"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"b","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"b","content": recent.clone()}),
        ];
        assert!(
            engine.has_compactable_history(),
            "an old bulky tool result is foldable"
        );
        let (before, after) = engine.compact_now_local();
        assert!(
            after < before,
            "clearing stale output frees context: {before} → {after}"
        );
        assert_eq!(
            engine.messages[3]["content"].as_str(),
            Some(TOOL_RESULT_CLEARED),
            "old stale tool output cleared"
        );
        assert_eq!(
            engine.messages[6]["content"].as_str(),
            Some(recent.as_str()),
            "recent tool output kept intact"
        );
    }

    #[test]
    fn has_compactable_history_false_for_short_conversation() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"hi"}),
            json!({"role":"assistant","content":"hello"}),
        ];
        assert!(
            !engine.has_compactable_history(),
            "a tiny recent-only conversation has nothing to fold"
        );
    }

    /// Tiny window + zero keep-recent: with only stale OLD tool output overflowing,
    /// `maybe_compact` takes the no-model cheap path (clears them, returns 0).
    #[tokio::test]
    async fn forced_tiny_window_compacts_at_boundary_without_a_model_call() {
        // SAFETY: scoped mutation of an env var no other test reads.
        unsafe { std::env::set_var("AIVO_AGENT_KEEP_RECENT", "0") };

        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
        let huge = "x".repeat(200_000);
        engine.messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "a", "content": huge.clone()}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "assistant", "content": "", "tool_calls": [
                {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]}),
            json!({"role": "tool", "tool_call_id": "b", "content": huge}),
            json!({"role": "user", "content": "now"}),
        ];
        let budget = engine.compaction_window() - COMPACT_RESERVE;
        assert!(
            estimate_tokens(&engine.messages) > budget,
            "transcript must start over budget so the boundary is actually crossed"
        );

        let client = reqwest::Client::new();
        let cwd = std::path::Path::new(".");
        let ctx = turn_ctx(&client, "", cwd);
        let mut ui = CapturingUi::default();
        let tokens = engine.maybe_compact(&ctx, &mut ui).await;

        unsafe { std::env::remove_var("AIVO_AGENT_KEEP_RECENT") };

        assert_eq!(tokens, 0, "cheap path must not call the model");
        assert!(
            estimate_tokens(&engine.messages) <= budget,
            "compaction must bring the transcript under budget"
        );
        let cleared = engine
            .messages
            .iter()
            .filter(|m| role(m) == "tool")
            .filter(|m| m.get("content").and_then(|c| c.as_str()) == Some(TOOL_RESULT_CLEARED))
            .count();
        assert_eq!(
            cleared, 2,
            "stale OLD tool output cleared without a model call"
        );
        assert!(
            ui.notices
                .iter()
                .any(|n| n.contains("cleared older tool output")),
            "the user is told the cheap path ran"
        );
    }

    /// A resumed single long turn over budget keeps prior context as a folded summary instead of dropping to `[system, latest-user]`.
    #[tokio::test]
    async fn resume_single_long_turn_keeps_prior_context_on_compaction() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
        // Assistant-only run: no tool results for the cheap clear path to reclaim.
        let big = "reasoning ".repeat(4_000);
        let mut messages = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "original task: fix the shell tool"}),
        ];
        for i in 0..4 {
            messages.push(json!({"role": "assistant", "content": format!("step {i}: {big}")}));
        }
        messages.push(json!({"role": "user", "content": "continue"}));
        engine.messages = messages;
        let budget = engine.compaction_window() - COMPACT_RESERVE;
        assert!(
            estimate_tokens(&engine.messages) > budget,
            "transcript must start over budget"
        );

        let client = reqwest::Client::new();
        let cwd = std::path::Path::new(".");
        let ctx = turn_ctx(&client, "", cwd); // empty base → summary fails → mechanical fold
        let mut ui = CapturingUi::default();
        engine.maybe_compact(&ctx, &mut ui).await;

        assert!(
            estimate_tokens(&engine.messages) <= budget,
            "compaction must fit the budget"
        );
        let last = engine.messages.last().unwrap();
        assert_eq!(role(last), "user");
        assert!(content_str(last).contains("continue"), "latest turn kept");
        assert!(
            content_str(last).contains("[Summary of earlier conversation]"),
            "the dropped prior turn must be summarized in, not silently discarded: {}",
            content_str(last)
        );
    }

    #[test]
    fn seed_history_carries_user_and_assistant_only() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.seed_history(vec![
            ("user".to_string(), "hi".to_string()),
            ("assistant".to_string(), "hello".to_string()),
            ("tool_call".to_string(), "{}".to_string()), // dropped
            ("tool_result".to_string(), "x".to_string()), // dropped
            ("user".to_string(), "next".to_string()),
        ]);
        // system + user + assistant + user (tool entries skipped)
        assert_eq!(engine.messages.len(), 4);
        assert_eq!(role(&engine.messages[0]), "system");
        assert_eq!(role(&engine.messages[1]), "user");
        assert_eq!(content_str(&engine.messages[1]), "hi");
        assert_eq!(role(&engine.messages[2]), "assistant");
        assert_eq!(role(&engine.messages[3]), "user");
        assert_eq!(content_str(&engine.messages[3]), "next");
    }

    /// `push_text_turn` merges into a preceding same-role plain-text message (never
    /// two consecutive same-role turns — Anthropic 400s). Different roles / tool_call assistants aren't merged.
    #[test]
    fn push_text_turn_merges_consecutive_same_role() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.push_text_turn("user", "first".to_string());
        engine.push_text_turn("user", "second".to_string()); // merges into "first"
        engine.push_text_turn("assistant", "reply".to_string());
        engine.push_text_turn("assistant", "more".to_string()); // merges into "reply"
        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert_eq!(roles, vec!["system", "user", "assistant"]);
        assert_eq!(content_str(&engine.messages[1]), "first\n\nsecond");
        assert_eq!(content_str(&engine.messages[2]), "reply\n\nmore");

        // A tool_calls-bearing assistant is not a plain-text turn → never merged.
        let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e2.messages.push(json!({
            "role":"assistant",
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]
        }));
        e2.push_text_turn("assistant", "text".to_string());
        assert_eq!(
            e2.messages.len(),
            3,
            "must not merge into a tool_calls assistant"
        );
    }

    /// Seeding a history with two adjacent user turns (cancelled + next) must not reproduce them as consecutive user messages.
    #[test]
    fn seed_history_merges_adjacent_user_turns() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.seed_history(vec![
            ("user".to_string(), "cancelled task".to_string()),
            ("user".to_string(), "real task".to_string()),
        ]);
        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert!(
            !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
            "consecutive user after seeding: {roles:?}"
        );
        assert_eq!(
            content_str(&engine.messages[1]),
            "cancelled task\n\nreal task"
        );
    }

    /// Seeding drops leading assistant turns so the conversation opens with a user message (Anthropic rejects assistant-first).
    #[test]
    fn seed_history_drops_leading_assistant_for_user_first() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.seed_history(vec![
            ("assistant".to_string(), "(mid-exchange reply)".to_string()),
            ("user".to_string(), "real question".to_string()),
            ("assistant".to_string(), "answer".to_string()),
        ]);
        // system + user + assistant — the leading assistant turn was dropped.
        assert_eq!(engine.messages.len(), 3);
        assert_eq!(role(&engine.messages[1]), "user");
        assert_eq!(content_str(&engine.messages[1]), "real question");
        assert_eq!(role(&engine.messages[2]), "assistant");

        // All-assistant history seeds nothing (a following user turn opens it).
        let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e2.seed_history(vec![("assistant".to_string(), "orphan".to_string())]);
        assert_eq!(e2.messages.len(), 1); // system only
    }

    #[test]
    fn first_party_branding_is_opt_in_idempotent_and_durable() {
        let mut e = AgentEngine::new("/tmp", "aivo/starter", "", &[], &[], 0, 0);
        // Off by default: the base prompt never names the model/provider (BYOK stays honest).
        assert!(!system_content(&e).contains("aivo's own assistant"));

        e.set_first_party();
        let branded = system_content(&e);
        assert!(branded.contains("aivo's own assistant"));
        assert!(branded.contains("aivo models"));
        // Must mutate in place, not push — `restore_conversation` no-ops unless `messages.len() == 1`.
        assert_eq!(e.messages.len(), 1);

        // Idempotent: a rebuild/resume re-runs it; a double call doesn't duplicate.
        e.set_first_party();
        assert_eq!(
            system_content(&e).matches("aivo's own assistant").count(),
            1
        );

        // Survives `reset()` (which keeps only the system message).
        e.reset();
        assert!(system_content(&e).contains("aivo's own assistant"));
    }

    #[test]
    fn confirm_before_build_is_opt_in_idempotent_and_durable() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        // Off by default (headless/-e and sub-agents must build without asking).
        assert!(!system_content(&e).contains("before you BUILD something substantial"));

        e.set_confirm_before_build();
        let gated = system_content(&e);
        assert!(gated.contains("before you BUILD something substantial"));
        // Carve-outs: small edits pass through, go-aheads skip, refinements re-ask.
        assert!(gated.contains("small single-file edits"));
        assert!(gated.contains("work autonomously"));
        assert!(gated.contains("plan REVISION, not approval"));
        // Mutate in place (single-system-message invariant).
        assert_eq!(e.messages.len(), 1);

        // Idempotent: a rebuild/resume re-runs it; a double call doesn't duplicate.
        e.set_confirm_before_build();
        assert_eq!(
            system_content(&e)
                .matches("before you BUILD something substantial")
                .count(),
            1
        );

        // Survives `reset()` (keeps only the system message).
        e.reset();
        assert!(system_content(&e).contains("before you BUILD something substantial"));
    }

    #[test]
    fn find_cut_lands_on_user_boundary() {
        let m = |role: &str, content: &str| json!({"role": role, "content": content});
        let messages = vec![
            m("system", "sys"),
            m("user", "turn1"),
            m("assistant", "a1"),
            m("tool", "t1"),
            m("user", "turn2"),
            m("assistant", "a2"),
        ];
        let cut = find_cut(&messages, 1);
        assert_eq!(cut, 4);
        assert_eq!(role(&messages[cut]), "user");
    }

    /// Compaction folds the summary INTO the first kept user turn (not before it) so
    /// roles keep alternating — Anthropic 400s otherwise, bricking the agent post-compaction.
    #[test]
    fn apply_compaction_folds_summary_and_keeps_roles_alternating() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        // [system, user, assistant, tool, user(=cut), assistant]
        engine
            .messages
            .push(json!({"role":"user","content":"first task"}));
        engine
            .messages
            .push(json!({"role":"assistant","content":"working"}));
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"c1","content":"result"}));
        engine
            .messages
            .push(json!({"role":"user","content":"second task"}));
        engine
            .messages
            .push(json!({"role":"assistant","content":"done"}));

        engine.apply_compaction(4, "did the early work");

        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert!(
            !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
            "compaction left consecutive user messages: {roles:?}"
        );
        // Summary folded into the (former) messages[4] user turn, now at index 1.
        assert_eq!(role(&engine.messages[1]), "user");
        let folded = content_str(&engine.messages[1]);
        assert!(
            folded.contains("did the early work") && folded.contains("second task"),
            "summary not folded into the kept user turn: {folded}"
        );
        // …and its assistant reply still follows it (alternation intact).
        assert_eq!(role(&engine.messages[2]), "assistant");
    }

    /// The pinned working set survives a compaction verbatim, folded into the SAME kept user turn so alternation holds even with a non-empty block.
    #[test]
    fn pinned_plan_and_files_survive_compaction() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.plan = plan::parse_plan(&json!({"plan":[
            {"step":"scan code","status":"completed"},
            {"step":"write fix","status":"in_progress"}
        ]}))
        .unwrap();
        engine.touched_files = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
        // [system, user, assistant, tool, user(=cut), assistant]
        engine
            .messages
            .push(json!({"role":"user","content":"first task"}));
        engine
            .messages
            .push(json!({"role":"assistant","content":"working"}));
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"c1","content":"result"}));
        engine
            .messages
            .push(json!({"role":"user","content":"second task"}));
        engine
            .messages
            .push(json!({"role":"assistant","content":"done"}));

        engine.apply_compaction(4, "summary body");

        let folded = content_str(&engine.messages[1]);
        assert!(folded.contains("summary body"), "{folded}");
        assert!(folded.contains("## Pinned Plan"), "{folded}");
        assert!(folded.contains("scan code") && folded.contains("write fix"));
        assert!(folded.contains("## Files touched"));
        assert!(folded.contains("src/a.rs") && folded.contains("src/b.rs"));
        // Alternation intact with a non-empty pinned block.
        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert!(
            !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
            "consecutive user after pinned compaction: {roles:?}"
        );
        assert_eq!(role(&engine.messages[2]), "assistant");
    }

    /// An empty working set folds byte-identically to the pre-pinning behavior
    /// (no `## Pinned …` sections leak in) — guards the existing-test invariant.
    #[test]
    fn apply_compaction_without_working_set_adds_no_pinned_sections() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine
            .messages
            .push(json!({"role":"user","content":"only task"}));
        engine.apply_compaction(1, "sum");
        let folded = content_str(&engine.messages[1]);
        assert!(!folded.contains("## Pinned Plan"));
        assert!(!folded.contains("## Files touched"));
    }

    /// Compaction preserves tool_use↔tool_result pairing in the KEPT region: every
    /// surviving `tool` follows an assistant `tool_calls` naming its id, and no orphan
    /// tool heads the kept history (a leading tool result also 400s strict providers).
    #[test]
    fn apply_compaction_preserves_tool_pairing_across_cut() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        // [system, user, assistant(call x1), tool(x1), user(=cut), assistant(call y1), tool(y1)]
        engine
            .messages
            .push(json!({"role":"user","content":"first task"}));
        engine.messages.push(json!({
            "role":"assistant",
            "tool_calls":[{"id":"x1","type":"function","function":{"name":"read_file","arguments":"{}"}}]
        }));
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"x1","content":"early result"}));
        engine
            .messages
            .push(json!({"role":"user","content":"second task"}));
        engine.messages.push(json!({
            "role":"assistant",
            "tool_calls":[{"id":"y1","type":"function","function":{"name":"grep","arguments":"{}"}}]
        }));
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"y1","content":"late result"}));

        // Cut at the second user turn (index 4): everything before is summarized away.
        let cut = find_cut(&engine.messages, 1);
        assert_eq!(cut, 4, "cut should land on the second user boundary");
        engine.apply_compaction(cut, "summary of the early work");

        // The early pair (x1) is gone; the kept pair (y1) survives intact.
        let ids: Vec<&str> = engine
            .messages
            .iter()
            .filter(|m| role(m) == "tool")
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        assert_eq!(ids, vec!["y1"], "only the kept tool result should remain");

        // No orphan tool result: every surviving `tool` follows an assistant whose
        // `tool_calls` names its id.
        for (i, m) in engine.messages.iter().enumerate() {
            if role(m) != "tool" {
                continue;
            }
            let id = m["tool_call_id"].as_str().unwrap();
            let prev = &engine.messages[i - 1];
            assert_eq!(
                role(prev),
                "assistant",
                "tool result not preceded by assistant"
            );
            let names: Vec<&str> = prev["tool_calls"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|c| c["id"].as_str())
                .collect();
            assert!(names.contains(&id), "tool result {id} has no matching call");
        }
        // First kept message after the system prompt is the folded user turn, never
        // an orphan tool/assistant — alternation holds from the very top.
        assert_eq!(role(&engine.messages[0]), "system");
        assert_eq!(role(&engine.messages[1]), "user");
        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert!(
            !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
            "compaction left consecutive user messages: {roles:?}"
        );
    }

    /// The budget backstop drops oldest turns at user boundaries (never the system prompt) until it fits — guards against a non-retryable post-compaction 413.
    #[test]
    fn enforce_budget_drops_oldest_turns_at_user_boundaries() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let pad = "x".repeat(400);
        engine.messages = vec![
            json!({"role":"system","content":"SYS"}),
            json!({"role":"user","content":format!("u1 {pad}")}),
            json!({"role":"assistant","content":format!("a1 {pad}")}),
            json!({"role":"user","content":format!("u2 {pad}")}),
            json!({"role":"assistant","content":format!("a2 {pad}")}),
            json!({"role":"user","content":"u3 keep me"}),
            json!({"role":"assistant","content":"a3 keep me"}),
        ];
        engine.enforce_budget(200);

        assert!(
            estimate_tokens(&engine.messages) <= 200,
            "must fit budget, got {}",
            estimate_tokens(&engine.messages)
        );
        assert_eq!(role(&engine.messages[0]), "system"); // never dropped
        assert!(
            engine
                .messages
                .iter()
                .any(|m| content_str(m).contains("u3 keep me")),
            "latest turn must survive"
        );
        assert!(
            !engine
                .messages
                .iter()
                .any(|m| content_str(m).contains("u1")),
            "an old turn must be dropped"
        );
    }

    /// When even [system, last user turn] overflows (one huge pasted turn), the backstop shortens the content instead of looping forever.
    #[test]
    fn enforce_budget_truncates_a_single_oversized_turn() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.messages = vec![
            json!({"role":"system","content":"SYS"}),
            json!({"role":"user","content":"y".repeat(8000)}),
        ];
        engine.enforce_budget(300);

        assert!(
            estimate_tokens(&engine.messages) <= 300,
            "must fit budget, got {}",
            estimate_tokens(&engine.messages)
        );
        assert_eq!(role(&engine.messages[0]), "system");
        assert_eq!(
            engine.messages.len(),
            2,
            "the turn is shortened, not dropped"
        );
        assert!(
            content_str(&engine.messages[1]).contains("chars)"),
            "oversized content carries the truncation marker"
        );
    }

    #[test]
    fn record_touched_file_dedups_orders_and_filters_tools() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.record_touched_file("read_file", &json!({"path":"a.rs"}));
        engine.record_touched_file("read_file", &json!({"path":"a.rs"})); // dup
        engine.record_touched_file("write_file", &json!({"path":"b.rs"}));
        engine.record_touched_file("run_bash", &json!({"command":"ls"})); // not a file tool
        engine.record_touched_file("grep", &json!({"path":"c.rs"})); // tracked? no
        assert_eq!(
            engine.touched_files,
            vec!["a.rs".to_string(), "b.rs".to_string()]
        );
    }

    #[test]
    fn record_touched_file_caps_and_evicts_oldest() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        for i in 0..(MAX_TOUCHED_FILES + 5) {
            engine.record_touched_file("read_file", &json!({ "path": format!("f{i}.rs") }));
        }
        assert_eq!(engine.touched_files.len(), MAX_TOUCHED_FILES);
        assert!(!engine.touched_files.contains(&"f0.rs".to_string())); // oldest evicted
        assert!(
            engine
                .touched_files
                .contains(&format!("f{}.rs", MAX_TOUCHED_FILES + 4))
        );
    }

    #[test]
    fn build_summary_request_carries_prior_summary() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        // No prior summary → fresh prompt, transcript verbatim.
        let r1 = engine.build_summary_request("TRANSCRIPT");
        assert_eq!(r1.messages[0]["content"], json!(SUMMARY_SYSTEM_PROMPT));
        assert_eq!(r1.messages[1]["content"], json!("TRANSCRIPT"));
        // Carry-forward: prior summary set → update prompt + prior summary in user.
        engine.last_summary = Some("PRIOR".to_string());
        let r2 = engine.build_summary_request("NEWEVENTS");
        assert_eq!(
            r2.messages[0]["content"],
            json!(SUMMARY_UPDATE_SYSTEM_PROMPT)
        );
        let user = r2.messages[1]["content"].as_str().unwrap();
        assert!(
            user.contains("PRIOR") && user.contains("NEWEVENTS"),
            "{user}"
        );
    }

    #[test]
    fn pinned_block_token_cap_trims_files_keeps_plan() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.plan =
            plan::parse_plan(&json!({"plan":[{"step":"keep me","status":"pending"}]})).unwrap();
        // Far more files than fit under PINNED_MAX_TOKENS (set directly to bypass MAX_TOUCHED_FILES).
        engine.touched_files = (0..600)
            .map(|i| format!("src/very/long/path/segment/file_{i}.rs"))
            .collect();
        let block = engine.render_pinned_block();
        assert!(
            estimate_str_tokens(&block) <= PINNED_MAX_TOKENS,
            "pinned block over cap: {} tokens",
            estimate_str_tokens(&block)
        );
        assert!(block.contains("keep me"), "plan must be kept whole");
        // Most-recent file kept, oldest trimmed.
        assert!(block.contains("file_599.rs"));
        assert!(!block.contains("file_0.rs"));
    }

    #[test]
    fn reset_clears_compaction_working_set() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine.last_summary = Some("stale".to_string());
        engine.plan = plan::parse_plan(&json!({"plan":[{"step":"x","status":"pending"}]})).unwrap();
        engine.touched_files = vec!["a.rs".to_string()];
        engine.notes = vec!["a finding".to_string()];
        engine.reset();
        assert!(engine.last_summary.is_none());
        assert!(engine.plan.is_empty());
        assert!(engine.touched_files.is_empty());
        assert!(engine.notes.is_empty());
    }

    /// The cheap pass stubs bulky OLD tool outputs (before the keep window), leaving recent ones + their ids intact; idempotent.
    #[test]
    fn clear_stale_tool_results_clears_only_old_bulky_outputs() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        let big = "x".repeat(5000);
        // [0 system, 1 user, 2 assistant(call), 3 tool BIG(old), 4 user, 5 tool BIG(recent), 6 asst]
        e.messages.push(json!({"role":"user","content":"go"}));
        e.messages.push(json!({"role":"assistant","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}));
        e.messages
            .push(json!({"role":"tool","tool_call_id":"c1","content": big.clone()}));
        e.messages.push(json!({"role":"user","content":"more"}));
        e.messages
            .push(json!({"role":"tool","tool_call_id":"c2","content": big.clone()}));
        e.messages.push(json!({"role":"assistant","content":"ok"}));

        let cut = 4; // messages[4] is the second user turn → [1..4] is "old"
        assert!(e.stale_tool_result_savings(cut) > 1000);
        e.clear_stale_tool_results(cut);
        assert_eq!(e.messages[3]["content"], TOOL_RESULT_CLEARED);
        assert_eq!(e.messages[3]["tool_call_id"], "c1"); // pairing intact
        assert_eq!(
            e.messages[5]["content"].as_str().unwrap().len(),
            5000,
            "recent tool output untouched"
        );
        assert_eq!(e.stale_tool_result_savings(cut), 0, "idempotent");
    }

    /// `take_note` content rides into a compaction via the pinned block; the cap trims files before notes (notes kept, plan whole).
    #[test]
    fn notes_pin_into_compaction_block() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.plan =
            plan::parse_plan(&json!({"plan":[{"step":"keep me","status":"pending"}]})).unwrap();
        e.notes = vec!["decided on X".to_string(), "Y 500s — avoid".to_string()];
        e.touched_files = (0..600).map(|i| format!("src/seg/file_{i}.rs")).collect();
        let block = e.render_pinned_block();
        assert!(estimate_str_tokens(&block) <= PINNED_MAX_TOKENS);
        assert!(block.contains("## Notes"));
        assert!(block.contains("decided on X"));
        assert!(block.contains("keep me"), "plan kept whole");
        assert!(!block.contains("file_0.rs"), "files trimmed before notes");
    }

    /// Restore re-derives the working set (plan, notes, touched files) from the message log — the stateless-reducer property.
    #[test]
    fn restore_rebuilds_working_set_from_log() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.messages.push(json!({"role":"user","content":"do it"}));
        e.messages.push(json!({"role":"assistant","tool_calls":[
            {"id":"c1","type":"function","function":{"name":"update_plan",
             "arguments":"{\"plan\":[{\"step\":\"a\",\"status\":\"completed\"},{\"step\":\"b\",\"status\":\"in_progress\"}]}"}},
            {"id":"c2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}},
            {"id":"c3","type":"function","function":{"name":"take_note","arguments":"{\"note\":\"x uses async\"}"}}
        ]}));
        e.messages
            .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));
        e.messages
            .push(json!({"role":"tool","tool_call_id":"c2","content":"FILE"}));
        e.messages
            .push(json!({"role":"tool","tool_call_id":"c3","content":"Noted (1 saved)."}));
        e.messages
            .push(json!({"role":"assistant","content":"done"}));
        let convo = e.export_conversation();

        let mut restored = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        restored.restore_conversation(convo);
        assert_eq!(restored.plan.len(), 2);
        assert_eq!(restored.plan[1].status, plan::PlanStatus::InProgress);
        assert_eq!(restored.touched_files, vec!["src/x.rs".to_string()]);
        assert_eq!(restored.notes, vec!["x uses async".to_string()]);
    }

    /// A dangling-tool_calls assistant (interrupted mid-tool) is repaired before the next turn: each unanswered call id gets a synthetic result.
    #[test]
    fn repair_interrupted_tail_answers_dangling_calls() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine
            .messages
            .push(json!({"role":"user","content":"do it"}));
        engine.messages.push(json!({
            "role":"assistant",
            "tool_calls":[
                {"id":"c1","type":"function","function":{"name":"run_bash","arguments":"{}"}},
                {"id":"c2","type":"function","function":{"name":"read_file","arguments":"{}"}}
            ]
        }));
        // Only the first call's result made it in before the interrupt.
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));

        engine.repair_interrupted_tail();

        // c2 now has a result, sitting in the contiguous tool run after the call.
        let tool_ids: Vec<&str> = engine
            .messages
            .iter()
            .filter(|m| role(m) == "tool")
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        assert_eq!(tool_ids, vec!["c1", "c2"]);
        // A short assistant turn caps the synthesized results so the next user turn alternates (bare user after them → 2nd consecutive user, Anthropic 400).
        let last = engine.messages.last().unwrap();
        assert_eq!(role(last), "assistant");
        assert_eq!(last["content"], "[interrupted]");

        // Idempotent: a fully-answered + capped tail is left untouched.
        let len = engine.messages.len();
        engine.repair_interrupted_tail();
        assert_eq!(engine.messages.len(), len);

        // With a real next turn appended, the synthetic assistant sits between the results and the user, so roles alternate.
        engine
            .messages
            .push(json!({"role":"user","content":"next"}));
        let roles: Vec<&str> = engine.messages.iter().map(role).collect();
        assert_eq!(roles.last(), Some(&"user"));
        assert_eq!(
            roles[roles.len() - 2],
            "assistant",
            "tool results must be capped by an assistant before the next user: {roles:?}"
        );
    }

    /// Repaired-tail invariant: no assistant `tool_calls` is left without a matching `tool` result for every call id in the following run.
    #[test]
    fn repair_interrupted_tail_leaves_no_unanswered_tool_use() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine
            .messages
            .push(json!({"role":"user","content":"do it"}));
        engine.messages.push(json!({
            "role":"assistant",
            "tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"glob","arguments":"{}"}},
                {"id":"c","type":"function","function":{"name":"grep","arguments":"{}"}}
            ]
        }));
        // None of the three results landed before the interrupt.
        engine.repair_interrupted_tail();

        // Every assistant-with-tool_calls is fully answered: each call id appears in
        // the contiguous `tool` run immediately following the assistant.
        for (idx, m) in engine.messages.iter().enumerate() {
            let Some(calls) = m
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .filter(|a| !a.is_empty())
            else {
                continue;
            };
            let answered: HashSet<&str> = engine.messages[idx + 1..]
                .iter()
                .take_while(|m| role(m) == "tool")
                .filter_map(|m| m["tool_call_id"].as_str())
                .collect();
            for call in calls {
                let id = call["id"].as_str().unwrap();
                assert!(
                    answered.contains(id),
                    "dangling tool_use {id} left unanswered: {:?}",
                    engine.messages
                );
            }
        }
    }

    /// A clean transcript (every tool_use answered AND capped by a following assistant) must be left byte-for-byte unchanged.
    #[test]
    fn repair_interrupted_tail_leaves_clean_transcript_unchanged() {
        let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        engine
            .messages
            .push(json!({"role":"user","content":"do it"}));
        engine.messages.push(json!({
            "role":"assistant",
            "tool_calls":[
                {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}
            ]
        }));
        engine
            .messages
            .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));
        // An assistant already follows the result → already capped, so the alternation guard must NOT add a second cap.
        engine
            .messages
            .push(json!({"role":"assistant","content":"all done"}));

        let before = engine.messages.clone();
        engine.repair_interrupted_tail();
        assert_eq!(
            engine.messages, before,
            "clean (answered + capped) transcript was modified"
        );
    }

    // ── named specialist sub-agents ─────────────────────────────────────────

    fn subagent(name: &str, model: Option<&str>, tools: Option<Vec<&str>>) -> Subagent {
        Subagent {
            name: name.to_string(),
            description: format!("the {name} specialist"),
            model: model.map(str::to_string),
            tools: tools.map(|t| t.into_iter().map(str::to_string).collect()),
            body: format!("You are {name}. Follow the {name} playbook."),
            source: PathBuf::new(),
        }
    }

    fn tool_names(engine: &AgentEngine) -> Vec<String> {
        engine
            .tools_openai
            .iter()
            .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
            .collect()
    }

    fn system_content(engine: &AgentEngine) -> String {
        engine.messages[0]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    #[test]
    fn chat_session_context_injects_facts_and_tools() {
        let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
        assert!(!system_content(&e).contains("interactive `aivo code` session"));
        assert!(!tool_names(&e).contains(&"switch_model".to_string()));

        e.set_chat_session_context(ChatSessionContext {
            model_label: "gpt-5".to_string(),
            provider_label: "openrouter".to_string(),
            effort: Some("high".to_string()),
            effort_levels: vec!["low".into(), "medium".into(), "high".into()],
        });
        let p = system_content(&e);
        assert!(p.contains("interactive `aivo code` session"));
        assert!(p.contains("gpt-5") && p.contains("openrouter") && p.contains("high"));
        assert!(p.contains("/model") && p.contains("switch_model") && p.contains("/key"));
        let names = tool_names(&e);
        assert!(names.contains(&"switch_model".to_string()));
        assert!(names.contains(&"set_effort".to_string()));
        assert!(names.contains(&"ask_user".to_string()));
        // single-system-message invariant + idempotent
        assert_eq!(e.messages.len(), 1);
        e.set_chat_session_context(ChatSessionContext {
            model_label: "x".into(),
            provider_label: "y".into(),
            effort: None,
            effort_levels: vec![],
        });
        assert_eq!(
            tool_names(&e)
                .iter()
                .filter(|n| *n == "switch_model")
                .count(),
            1
        );
    }

    #[test]
    fn chat_session_context_reports_no_effort_levels() {
        let mut e = AgentEngine::new("/tmp", "some-model", "", &[], &[], 0, 0);
        e.set_chat_session_context(ChatSessionContext {
            model_label: "some-model".into(),
            provider_label: "prov".into(),
            effort: None,
            effort_levels: vec![],
        });
        assert!(system_content(&e).contains("no reasoning-effort levels"));
    }

    #[tokio::test]
    async fn session_control_tools_route_to_ui_callbacks() {
        #[derive(Default)]
        struct SwitchUi {
            switched: Vec<String>,
            efforts: Vec<String>,
            asked: Vec<(String, Vec<String>, bool)>,
        }
        impl AgentUi for SwitchUi {
            fn assistant_text(&mut self, _: &str) {}
            fn tool_start(&mut self, _: &str, _: &Value) {}
            fn tool_result(&mut self, _: &str, _: &Result<String, String>) {}
            fn notify(&mut self, _: &str) {}
            fn footer(&mut self, _: Option<&str>, _: usize, _: u64, _: u64, _: u64) {}
            fn ask_permission<'a>(
                &'a mut self,
                _: &'a str,
                _: Option<&'a str>,
            ) -> BoxFuture<'a, Decision> {
                Box::pin(async { Decision::Allow })
            }
            fn switch_chat_model<'a>(
                &'a mut self,
                model: &'a str,
            ) -> BoxFuture<'a, Result<String, String>> {
                self.switched.push(model.to_string());
                Box::pin(async { Ok("switched".to_string()) })
            }
            fn set_chat_effort<'a>(
                &'a mut self,
                level: &'a str,
            ) -> BoxFuture<'a, Result<String, String>> {
                self.efforts.push(level.to_string());
                Box::pin(async { Ok("ok".to_string()) })
            }
            fn ask_user<'a>(
                &'a mut self,
                question: &'a str,
                options: &'a [crate::agent::ask::AskOption],
                allow_free_text: bool,
            ) -> BoxFuture<'a, Result<String, String>> {
                self.asked.push((
                    question.to_string(),
                    options.iter().map(|o| o.label.clone()).collect(),
                    allow_free_text,
                ));
                let answer = options.first().map(|o| o.label.clone()).unwrap_or_default();
                Box::pin(async move { Ok(answer) })
            }
        }

        let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
        let client = reqwest::Client::new();
        let cwd = std::path::Path::new(".");
        let ctx = turn_ctx(&client, "", cwd);
        let mut ui = SwitchUi::default();
        let calls = vec![
            ToolCall {
                id: "1".into(),
                name: "switch_model".into(),
                arguments: json!({"model": "opus"}),
            },
            ToolCall {
                id: "2".into(),
                name: "set_effort".into(),
                arguments: json!({"level": "high"}),
            },
            ToolCall {
                id: "3".into(),
                name: "ask_user".into(),
                arguments: json!({
                    "question": "Add release notes now?",
                    "options": [{"label": "You add them"}, {"label": "Auto-generate"}],
                    "allow_free_text": false
                }),
            },
        ];
        e.execute_tool_batch(&ctx, &mut ui, &calls).await;
        assert_eq!(ui.switched, vec!["opus".to_string()]);
        assert_eq!(ui.efforts, vec!["high".to_string()]);
        assert_eq!(
            ui.asked,
            vec![(
                "Add release notes now?".to_string(),
                vec!["You add them".to_string(), "Auto-generate".to_string()],
                false
            )]
        );

        // The default trait impl (non-chat host) declines.
        let mut plain = CapturingUi::default();
        let declined = plain.switch_chat_model("opus").await.unwrap_err();
        assert!(declined.contains("interactive"));
    }

    /// With no subagents the tool stays generic — no `agent` field, no listing.
    #[test]
    fn subagent_tool_spec_is_generic_without_profiles() {
        let spec = subagent_tool_spec(&[]);
        assert_eq!(spec.name, "subagent");
        assert!(spec.parameters["properties"].get("agent").is_none());
        assert_eq!(spec.parameters["required"], json!(["task"]));
    }

    /// With profiles, the tool advertises them via an `agent` enum.
    #[test]
    fn subagent_tool_spec_enumerates_named_profiles() {
        let subs = vec![
            subagent("reviewer", None, None),
            subagent("researcher", None, None),
        ];
        let spec = subagent_tool_spec(&subs);
        let enumv = &spec.parameters["properties"]["agent"]["enum"];
        assert_eq!(enumv, &json!(["reviewer", "researcher"]));
        assert!(spec.description.contains("named specialist"));
    }

    /// `set_subagents` swaps in the enum-bearing tool and advertises the names in
    /// the system prompt (progressive disclosure — body is NOT inlined).
    #[test]
    fn set_subagents_wires_tool_and_prompt() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.set_subagents(&[subagent("reviewer", None, None)]);
        // The subagent tool now carries the enum.
        let sub_tool = e
            .tools_openai
            .iter()
            .find(|t| t["function"]["name"] == "subagent")
            .unwrap();
        assert_eq!(
            sub_tool["function"]["parameters"]["properties"]["agent"]["enum"],
            json!(["reviewer"])
        );
        // exactly one `subagent` tool (the generic one was replaced, not duplicated).
        assert_eq!(
            tool_names(&e).iter().filter(|n| *n == "subagent").count(),
            1
        );
        // System prompt lists the name + one-liner, not the full body.
        let sys = system_content(&e);
        assert!(sys.contains("- reviewer: the reviewer specialist"));
        assert!(!sys.contains("Follow the reviewer playbook"));
        // Empty set is a no-op (no agent field).
        let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e2.set_subagents(&[]);
        let sub_tool2 = e2
            .tools_openai
            .iter()
            .find(|t| t["function"]["name"] == "subagent")
            .unwrap();
        assert!(
            sub_tool2["function"]["parameters"]["properties"]
                .get("agent")
                .is_none()
        );
    }

    /// A profile's body folds into the system prompt; a `tools` allow-list restricts the built-ins (keeping `update_plan`).
    #[test]
    fn apply_profile_folds_role_and_scopes_tools() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.drop_subagent_tool(); // mirror a real sub-engine
        let before = tool_names(&e);
        assert!(before.contains(&"write_file".to_string()));
        assert!(before.contains(&"run_bash".to_string()));

        e.apply_profile(&subagent("reviewer", None, Some(vec!["read_file", "grep"])));

        // Role instructions are appended verbatim.
        let sys = system_content(&e);
        assert!(sys.contains("## Your role: reviewer"));
        assert!(sys.contains("Follow the reviewer playbook"));

        // Scoped to the allow-list (+ update_plan); writes/bash are gone.
        let after = tool_names(&e);
        assert!(after.contains(&"read_file".to_string()));
        assert!(after.contains(&"grep".to_string()));
        assert!(after.contains(&"update_plan".to_string()));
        assert!(!after.contains(&"write_file".to_string()));
        assert!(!after.contains(&"run_bash".to_string()));
    }

    /// On a gpt-5 engine, an authored `Edit` scope grants `apply_patch`.
    #[test]
    fn apply_profile_edit_scope_grants_apply_patch_on_gpt5() {
        let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
        e.drop_subagent_tool();
        assert!(tool_names(&e).contains(&"apply_patch".to_string()));
        e.apply_profile(&subagent(
            "editor",
            None,
            Some(vec!["read_file", "edit_file"]),
        ));
        let after = tool_names(&e);
        assert!(
            after.contains(&"apply_patch".to_string()),
            "lost editor on gpt-5"
        );
        assert!(after.contains(&"read_file".to_string()));
        assert!(!after.contains(&"run_bash".to_string()));
    }

    /// Reverse of the gpt-5 case: an `apply_patch` scope grants `edit_file` — the edit family is one class, so scoping is symmetric.
    #[test]
    fn apply_profile_apply_patch_scope_grants_edit_file_off_codex() {
        let mut e = AgentEngine::new("/tmp", "claude-sonnet-4-6", "", &[], &[], 0, 0);
        e.drop_subagent_tool();
        assert!(tool_names(&e).contains(&"edit_file".to_string()));
        e.apply_profile(&subagent(
            "patcher",
            None,
            Some(vec!["read_file", "apply_patch"]),
        ));
        let after = tool_names(&e);
        assert!(
            after.contains(&"edit_file".to_string()),
            "lost editor off codex"
        );
        assert!(after.contains(&"read_file".to_string()));
        assert!(!after.contains(&"run_bash".to_string()));
    }

    /// A profile with no `tools` scope leaves the toolset untouched.
    #[test]
    fn apply_profile_without_scope_keeps_all_tools() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.drop_subagent_tool();
        let before = tool_names(&e);
        e.apply_profile(&subagent("helper", None, None));
        assert_eq!(tool_names(&e), before);
    }

    /// Durable resume round trip: `export_conversation` drops the system prompt but keeps
    /// exact tool-call/result pairing, and `restore_conversation` rebuilds it after a fresh
    /// system prompt. Restore is a no-op once non-fresh.
    #[test]
    fn export_then_restore_round_trips_tool_history() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.messages
            .push(json!({"role": "user", "content": "read it"}));
        e.messages.push(json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}
            }]
        }));
        e.messages
            .push(json!({"role": "tool", "tool_call_id": "call_1", "content": "FILE BODY"}));
        e.messages
            .push(json!({"role": "assistant", "content": "done"}));

        let convo = e.export_conversation();
        assert_eq!(convo.len(), 4, "system prompt is excluded");
        assert_eq!(convo[1]["tool_calls"][0]["id"], "call_1");

        let mut restored = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        restored.restore_conversation(convo.clone());
        assert_eq!(restored.messages.len(), 5, "fresh system prompt + 4 turns");
        // Tool-call id and its matching result survive exactly (the lost-on-resume bug).
        assert_eq!(restored.messages[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(restored.messages[3]["tool_call_id"], "call_1");
        assert_eq!(restored.messages[3]["content"], "FILE BODY");

        // Restoring into a non-fresh engine is a no-op (guards double-restore).
        restored.restore_conversation(convo);
        assert_eq!(restored.messages.len(), 5);
    }

    // --- /rewind: tree checkpoints ---
    // (Git file-revert is covered exhaustively in `agent::checkpoint`; these
    // exercise the engine's truncation + mapping + store wiring.)

    fn rewind_engine(dir: &Path) -> AgentEngine {
        AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0)
    }

    #[tokio::test]
    async fn rewind_to_truncates_messages_and_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = rewind_engine(dir.path());
        // [system]; then two turns each adding a user + assistant message.
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "a".into(),
            tree: None,
            changed: None,
        });
        engine
            .messages
            .push(json!({"role": "user", "content": "a"}));
        engine
            .messages
            .push(json!({"role": "assistant", "content": "b"}));
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "c".into(),
            tree: None,
            changed: None,
        });
        engine
            .messages
            .push(json!({"role": "user", "content": "c"}));
        engine
            .messages
            .push(json!({"role": "assistant", "content": "d"}));

        let outcome = engine.rewind_to(1).await;
        // Truncated back to the start of turn 1 (system + turn 0's two messages).
        assert_eq!(engine.messages.len(), 3);
        let targets = engine.rewind_targets();
        assert_eq!(targets.len(), 1);
        // No tree on these checkpoints → conversation-only, nothing reverted.
        assert!(!targets[0].1);
        assert_eq!((outcome.restored, outcome.deleted), (0, 0));
        assert!(outcome.error.is_none());
    }

    #[tokio::test]
    async fn rewind_targets_report_prompts_and_revertibility() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = rewind_engine(dir.path());
        // Two turns restored on resume carry no checkpoints (conversation-only).
        engine.restore_conversation(vec![
            json!({"role": "user", "content": "a"}),
            json!({"role": "assistant", "content": "b"}),
        ]);
        // A live turn (user "c") with a tree snapshot, then one (user "e") without.
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "c".into(),
            tree: Some("abc".into()),
            changed: None,
        });
        engine
            .messages
            .push(json!({"role": "user", "content": "c"}));
        engine
            .messages
            .push(json!({"role": "assistant", "content": "d"}));
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "e".into(),
            tree: None,
            changed: None,
        });
        engine
            .messages
            .push(json!({"role": "user", "content": "e"}));

        // Targets carry the opening prompt + revertibility, in order.
        let targets = engine.rewind_targets();
        assert_eq!(
            targets,
            vec![("c".to_string(), true), ("e".to_string(), false)]
        );
    }

    #[tokio::test]
    async fn compaction_rebases_checkpoint_indices() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = rewind_engine(dir.path());
        // Three turns: [system, u0, a0, u1, a1, u2, a2] with a checkpoint opening
        // each user turn (indices 1, 3, 5).
        for (u, a) in [("u0", "a0"), ("u1", "a1"), ("u2", "a2")] {
            engine.checkpoints.push(Checkpoint {
                msg_index: engine.messages.len(),
                prompt: u.into(),
                tree: None,
                changed: None,
            });
            engine.messages.push(json!({"role": "user", "content": u}));
            engine
                .messages
                .push(json!({"role": "assistant", "content": a}));
        }
        assert_eq!(
            engine
                .checkpoints
                .iter()
                .map(|c| c.msg_index)
                .collect::<Vec<_>>(),
            vec![1, 3, 5]
        );

        // Fold the first turn (cut lands on u1 at index 3).
        engine.apply_compaction(3, "S");

        // u0's checkpoint dropped; survivors shifted down to stay valid (not stale).
        // messages is now [system, "S\n\nu1", a1, u2, a2].
        assert_eq!(
            engine
                .checkpoints
                .iter()
                .map(|c| c.msg_index)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        // Verbatim "u1", not the folded "S\n\nu1" at messages[1].
        assert_eq!(engine.rewind_targets()[0].0, "u1");
        assert_eq!(engine.rewind_targets()[1].0, "u2");

        // Rewinding to the last turn truncates at the correct (rebased) index.
        engine.rewind_to(1).await;
        assert_eq!(engine.messages.len(), 3);
        assert_eq!(role(engine.messages.last().unwrap()), "assistant");
        assert_eq!(engine.messages[2]["content"], "a1");
    }

    #[tokio::test]
    async fn rewind_target_survives_interrupted_turn_merge() {
        // A resend after an interrupt merges into the trailing `user` ("first\n\nsecond");
        // the stored prompt must stay "first" so the turn keeps file revert, not conversation-only.
        let dir = tempfile::tempdir().unwrap();
        let mut engine = rewind_engine(dir.path());
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "first".into(),
            tree: Some("abc".into()),
            changed: None,
        });
        engine.push_text_turn("user", "first".into());
        engine.push_text_turn("user", "second".into());
        assert_eq!(
            engine.messages.last().unwrap()["content"],
            "first\n\nsecond"
        );

        let targets = engine.rewind_targets();
        assert_eq!(targets, vec![("first".to_string(), true)]);
    }

    #[tokio::test]
    async fn rewind_reverts_files_through_the_engine() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("a.txt"), "v0").unwrap();
        let mut engine = rewind_engine(p);
        engine.enable_rewind_checkpoints(&p.display().to_string());
        let tree = {
            let store = engine.checkpoint_store.as_mut().unwrap();
            if !store.git_available().await {
                return; // git missing → skip
            }
            store.snapshot().await
        };
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: "go".into(),
            tree,
            changed: None,
        });
        // Simulate the turn: rename + edit (the case byte-snapshots couldn't revert).
        std::fs::rename(p.join("a.txt"), p.join("b.txt")).unwrap();
        std::fs::write(p.join("b.txt"), "v1").unwrap();

        let outcome = engine.rewind_to(0).await;
        assert_eq!(std::fs::read_to_string(p.join("a.txt")).unwrap(), "v0");
        assert!(!p.join("b.txt").exists(), "renamed file removed");
        assert!(outcome.restored >= 1);
        assert!(outcome.error.is_none());
    }

    #[tokio::test]
    async fn lazy_checkpoint_snapshots_only_before_a_mutating_tool() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(p.join("a.txt"), "v0").unwrap();
        let mut engine = rewind_engine(p);
        engine.enable_rewind_checkpoints(&p.display().to_string());
        if !engine
            .checkpoint_store
            .as_mut()
            .unwrap()
            .git_available()
            .await
        {
            return; // git missing → skip
        }
        let client = reqwest::Client::new();
        let ctx = turn_ctx(&client, "http://127.0.0.1", p);
        let mut ui = CapturingUi::default();
        // Stands in for the turn-start checkpoint (tree filled lazily, if at all).
        engine.checkpoints.push(Checkpoint {
            msg_index: 0,
            prompt: "go".into(),
            tree: None,
            changed: None,
        });

        // A read-only batch must NOT snapshot.
        let read = vec![ToolCall {
            id: "1".into(),
            name: "read_file".into(),
            arguments: json!({ "path": "a.txt" }),
        }];
        engine.execute_tool_batch(&ctx, &mut ui, &read).await;
        assert!(
            engine.checkpoints.last().unwrap().tree.is_none(),
            "read-only turn pays no snapshot"
        );

        // A mutating batch snapshots the pre-edit tree first.
        let write = vec![ToolCall {
            id: "2".into(),
            name: "write_file".into(),
            arguments: json!({ "path": "a.txt", "content": "v1" }),
        }];
        engine.execute_tool_batch(&ctx, &mut ui, &write).await;
        assert!(
            engine.checkpoints.last().unwrap().tree.is_some(),
            "snapshot taken before a mutating tool"
        );
    }
}
