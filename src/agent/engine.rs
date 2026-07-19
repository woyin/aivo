//! aivo's native in-process agent engine. Holds the conversation, composes
//! OpenAI chat requests, calls the model through the loopback serve (sole network
//! egress), executes tools (permission-gated), compacts on overflow, converges.
//! Rendering/permission go through `AgentUi` (terminal, `--json`, chat TUI).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};

use crate::agent::ask;
use crate::agent::guards::{self, batch_sig, page_read_key};
use crate::agent::jobs;
use crate::agent::notes;
use crate::agent::plan::{self, PlanItem};
use crate::agent::plan_mode;
use crate::agent::protocol::{
    AssistantMessage, ChatRequest, Decision, PlanDecision, ToolCall, ToolSpec,
};
use crate::agent::request::{assistant_to_openai, role, tool_to_openai};
use crate::agent::retry::{
    error_is_retryable, is_context_overflow_error, resolve_max_steps, retry_delay,
    retryable_error_label, terminal_error_notice,
};
use crate::agent::secrets_guard;
use crate::agent::skills::{self, Skill};
use crate::agent::subagents::{self, Subagent};
use crate::agent::system_prompt::system_prompt;
use crate::agent::tokens::{content_to_parts, estimate_str_tokens, estimate_tokens, usage_tokens};
use crate::agent::{serve_client, tool_repair, tool_search, tools, verify};
use crate::services::session_store::SessionTokens;
use crate::services::token_usage::extract_usage_from_value;

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
/// Cap on Stop-hook refusals per turn, so a hook that always exits 2 can't loop the run.
const MAX_STOP_HOOK_CONTINUES: usize = 3;
const STOP_HOOK_PREFIX: &str = "A user-configured Stop hook reviewed your answer and asked you \
to continue. Address the following, then finish:";
/// Prefix of the artifact-pointer line; compaction preserves it so the parent can
/// `read_file` a cleared sub-agent report back.
pub(crate) const ARTIFACT_POINTER_PREFIX: &str = "[full report saved: ";
/// Guard-stop notice text (display only — drivers get the typed [`TurnStop`]).
pub(crate) const STOP_NO_PROGRESS: &str =
    "stopping: the model repeated the same action with no progress";
pub(crate) const STOP_TOOL_FAILURE: &str = "stopping: a tool call kept failing the same way";
const COMPLETION_NUDGE: &str = "That may not be finished. If the task is genuinely complete, \
briefly confirm what you did and verified, then stop. Otherwise keep going — don't stop until \
it's done or you're truly blocked (then say exactly what's blocking you).";
/// Cap on unstarted-plan nudges per turn.
const MAX_PLAN_NUDGES: usize = 1;
const PLAN_NUDGE: &str = "You set a plan this turn but haven't started any of its steps. \
Execute the plan now, updating each step's status as you go. If the work is already done or \
you can't proceed, call `update_plan` to reflect that (or say exactly what's blocking you), \
then finish.";

/// Compaction window assumed when the model's real one is unknown (0); without it
/// such models never compact and resend the whole transcript. A real window wins.
pub(crate) const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

/// Why a turn ended early, surfaced via [`AgentUi::turn_stopped`]. Typed so a
/// driver like the `/goal` loop must handle each variant in a `match`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnStop {
    NoProgress,
    ToolFailureLoop,
    StepLimit,
}

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

/// Live progress for a parallel sub-agent batch, slot-tagged in call order.
/// `Arc`-shared (`&self` + `Send + Sync`) so concurrent delegates report into one UI.
pub trait SubagentSink: Send + Sync {
    fn begin(&self, labels: &[String]);
    /// Empty `tool` = thinking between calls.
    fn activity(&self, slot: usize, agent: &str, tool: &str, args: &Value, step: usize);
    /// A gated call was auto-denied (no interactive approval mid-batch).
    fn denied(&self, slot: usize, tool: &str);
    /// `ok` = the delegate produced an answer.
    fn done(&self, slot: usize, ok: bool, steps: usize, tokens: u64);
    fn finish(&self);
}

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
    /// Live feed for a parallel sub-agent batch; `None` (default) keeps it quiet.
    fn subagent_sink(&mut self) -> Option<std::sync::Arc<dyn SubagentSink>> {
        None
    }
    /// Prompt for the next REPL turn. `None` ends the session (EOF / `/exit`);
    /// default `None` → one-shot only.
    fn read_user_input(&mut self) -> Option<String> {
        None
    }
    /// Mid-turn messages, drained after each tool batch. Default none so a
    /// sub-agent can't consume the parent's queue.
    fn drain_steering(&mut self) -> Vec<String> {
        Vec::new()
    }
    fn assistant_text(&mut self, delta: &str);
    /// A streamed reasoning/thinking delta (separate from the visible reply). Default no-op.
    fn assistant_reasoning(&mut self, _delta: &str) {}
    /// Drop the just-streamed segment — it was a tool call written as text (stripped + retried). Default no-op.
    fn discard_streamed_segment(&mut self) {}
    /// The agent set/updated its plan via `update_plan`; rendered as a checklist card. Default no-op.
    fn plan_updated(&mut self, _items: &[PlanItem]) {}
    fn tool_start(&mut self, name: &str, args: &Value);
    /// Live output chunk from an in-flight `run_bash` (local display only,
    /// pre-redaction like `tool_result`). Default no-op.
    fn tool_output(&mut self, _name: &str, _chunk: &str) {}
    fn tool_result(&mut self, name: &str, result: &Result<String, String>);
    fn notify(&mut self, text: &str);
    /// The turn ended early for `stop` (also announced via `notify` for display).
    /// Default no-op.
    fn turn_stopped(&mut self, _stop: TurnStop) {}
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
        _multi_select: bool,
    ) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async {
            Err(
                "Asking the user a question is only available in interactive `aivo code`."
                    .to_string(),
            )
        })
    }
    /// The `exit_plan_mode` tool: show the plan and await the user's verdict on the
    /// approval card. Default declines — headless/sub-agents have no interactive user.
    fn approve_plan<'a>(
        &'a mut self,
        _plan: &'a str,
    ) -> BoxFuture<'a, Result<PlanDecision, String>> {
        Box::pin(async {
            Err("Plan approval is only available in interactive `aivo code`.".to_string())
        })
    }
    /// Show the pending edits and return the user's verdict. Only reached with the
    /// live toggle on; the default fails closed to `Reject` like `ask_permission`.
    fn review_edits<'a>(
        &'a mut self,
        _items: &'a [crate::agent::review::ReviewItem],
    ) -> BoxFuture<'a, crate::agent::review::ReviewDecision> {
        Box::pin(async { crate::agent::review::ReviewDecision::Reject })
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
    /// Waive the confirm tier only (headless `-e` baseline); the remote-mutation
    /// gate still stands — unattended runs deploy only via `--auto-approve`.
    pub yes: bool,
    /// Static `--auto-approve`: auto-approve mode for runs with no live toggle.
    pub auto_approve_all: bool,
    /// Live auto-approve-mode toggle (the chat TUI's Shift+Tab). Read fresh per
    /// tool call so flipping it mid-turn takes effect on the *running* turn,
    /// unlike the `yes` snapshot. `None` outside the chat TUI.
    pub auto_approve: Option<&'a std::sync::atomic::AtomicBool>,
    /// Live edit-review toggle (chat `/config`), read fresh per batch. `None`
    /// outside the chat TUI — headless / `-y` / sub-agents never gate.
    pub review_edits: Option<&'a std::sync::atomic::AtomicBool>,
}

impl TurnCtx<'_> {
    /// Auto-approve mode (flag or live toggle): everything short of the
    /// catastrophic/plan floors runs unprompted, remote mutations included.
    pub fn auto_approve_mode(&self) -> bool {
        self.auto_approve_all
            || self
                .auto_approve
                .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// True when the confirm tier runs without a prompt (`yes` or the mode).
    pub fn auto_approve_enabled(&self) -> bool {
        self.yes || self.auto_approve_mode()
    }

    /// True when edits should pause for review — the live toggle only (no `-y`).
    pub fn review_edits_enabled(&self) -> bool {
        self.review_edits
            .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// A `/rewind` turn boundary. `msg_index` = the turn's opening user message
/// (truncation point), kept valid across compaction via `rebase_checkpoints`.
/// `tree` = working-tree snapshot at turn start (`None` = conversation-only).
/// `prompt` = opening user text stored verbatim (the picker matches on it, since
/// `messages[msg_index]` gets mutated in place). `changed` = paths the turn modified
/// (a rewind reverts only their union); `None` until recorded. `seg_tree` = diff
/// base for a segment resumed after an interrupt recorded `changed`, so the
/// user's hand-edits made while the turn sat interrupted stay out of the diff.
#[derive(Clone)]
pub(crate) struct Checkpoint {
    pub(crate) msg_index: usize,
    pub(crate) prompt: String,
    pub(crate) tree: Option<String>,
    pub(crate) changed: Option<Vec<std::path::PathBuf>>,
    pub(crate) seg_tree: Option<String>,
}

/// Undo record for [`AgentEngine::begin_user_turn`], consumed by
/// [`AgentEngine::unsend_last_user_turn`]. `merged_prior` = the trailing user
/// message the turn merged into (restored verbatim); `None` = fresh append (popped).
pub(crate) struct TurnUnsend {
    msg_index: usize,
    merged_prior: Option<Value>,
    checkpoint_pushed: bool,
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
    /// Config dir for re-discovering profiles at delegation time. When set,
    /// `run_subagent` resolves `agent` fresh from disk — a profile authored or
    /// edited this very turn delegates correctly; unset falls back to the
    /// build-time `subagents` snapshot (tests, callers without a store).
    agents_dir: Option<PathBuf>,
    /// Kept so `subagent` can build a sub-engine with the same identity (date + guides).
    date: String,
    guides: Vec<String>,
    /// Extra tools beyond the built-ins (MCP servers), if any are configured.
    external: Option<std::sync::Arc<dyn ExternalTools>>,
    /// External specs deferred behind `search_tools`; loading moves a spec
    /// from here into `tools_openai`. See [`tool_search`].
    pub(crate) deferred_tools: Vec<Value>,
    /// Deferral threshold (estimate tokens); `None` = always inline. Read from
    /// the env once at construction so tests can override the field.
    pub(crate) mcp_defer_tokens: Option<usize>,
    /// Body of the last compaction summary (no prefix). Fed back to the summarizer
    /// next compaction so facts carry forward instead of being re-compressed lossily.
    pub(crate) last_summary: Option<String>,
    /// Latest `update_plan` plan. Pinned into every compaction fold, verbatim.
    pub(crate) plan: Vec<PlanItem>,
    /// Files touched this session (insertion order, deduped, capped). Maintained
    /// incrementally so it survives summarization; pinned into every compaction.
    pub(crate) touched_files: Vec<String>,
    /// Durable scratchpad: `take_note` entries, merged deterministically (reuse an id
    /// to revise, dedup exact text). Pinned verbatim into compaction and rebuilt from
    /// the log on resume, so they outlive turns/summaries. Capped at [`MAX_NOTES`].
    pub(crate) notes: Vec<notes::Note>,
    /// Provider-measured token split (prompt/completion/cache) for the LAST turn,
    /// summed across steps. The chat TUI drains it (`take_turn_usage`) for `aivo stats`. Reset per turn.
    turn_usage: SessionTokens,
    /// `/rewind`: one checkpoint per `run_turn`, in order. The chat TUI maps display
    /// turns by matching prompt text newest-backward (robust to trim/compaction/rebuild,
    /// which a positional index isn't). In-memory; tree objects live in `checkpoint_store`.
    pub(crate) checkpoints: Vec<Checkpoint>,
    /// Lets an Esc before anything streamed un-send the turn's opening user
    /// message. Overwritten each `begin_user_turn`.
    turn_unsend: Option<TurnUnsend>,
    /// Tree-level snapshot/restore via a shadow git store. `None` until `/rewind` is
    /// enabled (top-level chat only). See [`crate::agent::checkpoint`].
    checkpoint_store: Option<crate::agent::checkpoint::CheckpointStore>,
    /// Per-session dir for durable sub-agent reports; `None` = feature off.
    artifacts_dir: Option<PathBuf>,
    /// Artifact filename counter; atomic — concurrent fan-out runs under `&self`.
    artifact_seq: AtomicUsize,
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
    /// Plan mode: mutating tools refused so planning can't modify the workspace.
    /// Reversible — see `set_plan_mode`.
    read_only: bool,
    /// Tool specs stripped by `set_plan_mode(true)`, restored verbatim on exit.
    plan_mode_stash: Vec<Value>,
    /// Unattended `-e` only: reject a text turn that admits it isn't done (or trails
    /// off mid-step) and nudge to continue, rather than accept it as the final answer.
    require_completion: bool,
    /// Run the project's validator at declared-done and feed failures back. Default
    /// on for headless `-e` (`AIVO_AGENT_SELF_CORRECT=0` opts out), opt-in (`=1`) interactive.
    self_correct: bool,
    /// Gates self-correct so investigate-only turns don't re-run the whole suite.
    /// Starts true (tree state unknown); [`Self::set_verified_baseline`] clears it.
    dirty_since_verify: bool,
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
    /// LSP diagnostics-after-edit (default on, `AIVO_AGENT_LSP=0` opts out); `None` = disabled.
    lsp: Option<crate::agent::lsp::LspManager>,
    /// App-owned background-job table; `None` → `check_job` unadvertised.
    jobs: Option<crate::agent::jobs::SharedJobs>,
    /// User lifecycle hooks; `None` = none configured. Shared with sub-engines.
    hooks: Option<std::sync::Arc<crate::agent::hooks::HookSet>>,
    /// `--max-cost`: stop the turn at this estimated spend (USD; 0 = off).
    max_cost_usd: f64,
    cost_pricing: Option<crate::services::model_metadata::Pricing>,
    /// chars/4 tokens of the `-c` block folded into the system prompt, so
    /// `context_report` can split it back out.
    injected_context_tokens: usize,
    /// Provider-reported USD spend summed across the turn's steps (`usage.cost`).
    turn_cost_usd: f64,
    /// Upstream model echoed by responses — resolves aliases for pricing/stats.
    billed_model: Option<String>,
}

/// Calibrated chars/4 breakdown of what fills the context window, for `/context`.
#[derive(Clone, Debug, Default)]
pub struct ContextReport {
    /// Context window in tokens (0 = unknown).
    pub context_window: u32,
    /// System message minus the injected `-c` block.
    pub system_prompt: u64,
    /// The injected `-c` block (0 when none).
    pub injected_context: u64,
    pub tools: u64,
    pub tool_count: usize,
    pub mcp_tools: u64,
    pub mcp_tool_count: usize,
    /// External tools deferred behind `search_tools` (schemas not in context).
    pub mcp_deferred_count: usize,
    /// Transcript: every message after the system prompt.
    pub messages: u64,
    pub message_count: usize,
    pub calibration: f64,
}

impl ContextReport {
    pub fn used(&self) -> u64 {
        self.system_prompt + self.injected_context + self.tools + self.mcp_tools + self.messages
    }

    pub fn free(&self) -> u64 {
        u64::from(self.context_window).saturating_sub(self.used())
    }

    /// Scale the chars/4 split so `used()` matches `target` (the measured
    /// prompt), keeping each share — the safety calibration is clamped `>= 1.0`
    /// and can't deflate chars/4's ~2x over-count of JSON tool defs.
    pub fn rescale(&mut self, target: u64) {
        let used = self.used();
        if used == 0 || target == 0 {
            return;
        }
        let scale = |v: u64| ((u128::from(v) * u128::from(target)) / u128::from(used)) as u64;
        self.system_prompt = scale(self.system_prompt);
        self.injected_context = scale(self.injected_context);
        self.tools = scale(self.tools);
        self.mcp_tools = scale(self.mcp_tools);
        self.messages = scale(self.messages);
    }
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

mod config;
mod conversation;
mod subagent;
mod tool_batch;
mod turn;

use subagent::*;

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
list of the steps you'd take. Do NOT create or modify files, or run build/scaffold/\
state-changing commands, until the user approves; end your turn after presenting the plan. To \
get the go-ahead, call `ask_user` with the plan's decision as the options — e.g. `Approve` and \
`Cancel` — and leave free text on, so the user can pick or type a tweak. Picking `Approve` (or \
a plain \"go\"/\"proceed\"/\"lgtm\" typed in the composer) is your signal to build. A free-text \
answer that only asks for a change or states a preference — e.g. \"use a light theme\", \"add \
auth\", \"drop the export step\", \"make it Postgres\" — is a plan REVISION, not approval: fold \
it in, show the updated plan, and ask again with `ask_user`. If one message both requests a \
change and says go (\"use a light theme, then go\"), apply the change and build. Picking \
`Cancel` or dismissing the card ends the turn — stop and let the user say how to proceed. This \
is ONLY for sizeable, multi-step build work — for quick fixes, small single-file edits, \
answering questions, and read-only exploration, keep acting directly rather than making the \
user approve trivial work. And if the user has already told you to proceed, handed you a plan \
to implement, or asked you to work autonomously, skip the confirmation and just build.";

const FIRST_PARTY_IDENTITY: &str = "You are aivo's own assistant. If the user asks what model you \
are, who built you, or which provider or company is behind you, present yourself as aivo's \
assistant and don't name, confirm, or guess at the underlying model or provider — that's an \
implementation detail. Point them to `aivo models` for their configured setup instead.";

#[cfg(test)]
mod tests;
