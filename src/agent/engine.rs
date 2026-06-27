//! aivo's native agent engine — in-process. Holds the conversation, composes
//! OpenAI chat requests, calls the model through the loopback serve (the sole
//! network egress), executes tools (permission-gated), compacts on overflow,
//! and converges. This replaces the former closed `aivo-agent-core` brain + the
//! stdio protocol: the loop and all I/O now live here, behind an `AgentUi` for
//! rendering/permission so the same engine drives the terminal, `--json`, and
//! (later) the chat TUI.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use futures::future::BoxFuture;
use serde_json::{Map, Value, json};

use crate::agent::notes;
use crate::agent::plan::{self, PlanItem};
use crate::agent::protocol::{AssistantMessage, ChatRequest, Decision, ToolCall, ToolSpec};
use crate::agent::skills::{self, Skill};
use crate::agent::subagents::{self, Subagent};
use crate::agent::{serve_client, tools};
use crate::services::serve_router::extract_usage_from_value;
use crate::services::session_store::SessionTokens;

/// Sanity ceiling for a *finite* step budget, so a caller can't accidentally
/// ask for billions of steps.
const MAX_STEPS_CEILING: usize = 10_000;

/// Resolves the per-turn step budget from the [`AgentEngine::new`] argument:
/// `0` means **no cap** (the default for an interactive turn — the repeat-limit
/// and esc-interrupt are the real safeties), any positive value is taken as-is,
/// sanity-capped at [`MAX_STEPS_CEILING`].
fn resolve_max_steps(max_steps: u32) -> usize {
    if max_steps == 0 {
        usize::MAX
    } else {
        (max_steps as usize).min(MAX_STEPS_CEILING)
    }
}
/// Stop a turn when the model emits the identical tool-call batch this many
/// times in a row — a weak-model loop that would otherwise burn the step budget.
const REPEAT_LIMIT: usize = 3;
/// Auto-retry budget for transient LLM/network failures (matches pi's default).
const MAX_RETRIES: usize = 3;
/// Step cap for a `subagent` run — bounded below the top-level budget so a
/// delegated subtask can't run away.
const SUBAGENT_MAX_STEPS: u32 = 20;

/// Exponential backoff before retry attempt `n` (1-based). Base is overridable
/// for tests via `AIVO_AGENT_RETRY_BASE_MS`.
fn retry_delay(attempt: usize) -> std::time::Duration {
    let base = std::env::var("AIVO_AGENT_RETRY_BASE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600u64);
    std::time::Duration::from_millis(base * (1u64 << attempt.saturating_sub(1)))
}

/// Whether an LLM/serve error is worth retrying: transient rate-limit / overload
/// / 5xx / network blips. Context-overflow is NOT retryable (compaction handles
/// it); auth (401/403) and bad-request (400) aren't either.
fn is_retryable_error(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    if e.contains("context length")
        || e.contains("context_length")
        || e.contains("maximum context")
        || e.contains("too many tokens")
    {
        return false;
    }
    // Auth / bad-request are terminal — surface them now, don't burn retries (and
    // delay the error) just because the provider's message happens to contain a
    // retryable word like "connection" or "timeout". Checked before the retryable
    // patterns. (Phrases, not bare status codes — "400" would match "5400ms".)
    const TERMINAL: &[&str] = &[
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "bad request",
        "bad_request",
    ];
    if TERMINAL.iter().any(|p| e.contains(p)) {
        return false;
    }
    const PATTERNS: &[&str] = &[
        "429",
        "500",
        "502",
        "503",
        "504",
        "overload",
        "rate limit",
        "rate_limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily",
        "service unavailable",
        "connection",
        "network",
        "fetch failed",
        "stream error",
        "request failed",
        "reset",
        "socket",
        "try again",
    ];
    PATTERNS.iter().any(|p| e.contains(p))
}
const KEEP_RECENT_TOKENS: usize = 20_000;
/// Tokens held back from the window for the response + tool schemas.
const COMPACT_RESERVE: usize = 16_000;
/// Window assumed for compaction when the model's real context window is unknown
/// (0); without it such models never compact and resend the whole transcript
/// every turn. A real window, once known, takes precedence.
const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

/// A `tool` result longer than this (in chars) is eligible for clearing once it
/// ages out of the recent window; smaller results aren't worth the churn.
const TOOL_RESULT_CLEAR_MIN: usize = 1_000;
/// Stub left in place of a cleared tool result. Short enough to fall below
/// [`TOOL_RESULT_CLEAR_MIN`] so clearing is idempotent, and the message (with its
/// `tool_call_id`) stays so the assistant↔tool pairing every provider requires
/// is intact — only the now-stale bytes go.
const TOOL_RESULT_CLEARED: &str = "[earlier tool output cleared to save context]";

const SUMMARY_SYSTEM_PROMPT: &str = "You are compressing a coding-agent conversation to free up \
context. Write a concise but complete summary under these exact headings:\n\
## Goal\n## Constraints & Preferences\n## Progress (Done / In Progress / Blocked)\n\
## Key Decisions\n## Next Steps\n## Critical Context\n\n\
Preserve specifics: file paths, function/identifier names, exact values, commands run. Drop \
chit-chat. Output only the summary.";

/// Carry-forward variant: instead of re-summarizing a blob that already contains
/// a prior summary (lossy drift over repeated compactions), the model is given
/// the CURRENT running summary plus only the NEW events, and asked to update the
/// summary in place — preserving still-relevant facts verbatim.
const SUMMARY_UPDATE_SYSTEM_PROMPT: &str = "You are MAINTAINING a running summary of an ongoing \
coding-agent session. Below is the CURRENT summary, then the NEW events since it was written. \
Produce the UPDATED summary under these exact headings:\n\
## Goal\n## Constraints & Preferences\n## Progress (Done / In Progress / Blocked)\n\
## Key Decisions\n## Next Steps\n## Critical Context\n\n\
Preserve every still-relevant fact from the current summary verbatim (file paths, \
function/identifier names, exact values, commands run); merge in the new events; drop a fact \
only when the new events explicitly supersede it. Output only the updated summary.";

/// Hard ceiling (chars/4 tokens) on the pinned working-set block folded into a
/// compaction. The plan is kept whole; the touched-files list is trimmed
/// (oldest first) until the block fits, so pinning can't reintroduce overflow.
const PINNED_MAX_TOKENS: usize = 2_000;
/// Cap on the tracked touched-files list (most-recent kept).
const MAX_TOUCHED_FILES: usize = 40;
/// Cap on the agent's durable scratchpad (most-recent kept). Bounds memory; the
/// pinned-block budget bounds how much rides into a compaction.
const MAX_NOTES: usize = 50;

/// Side-effects the engine delegates: rendering and the permission prompt.
/// `ask_permission` is only called for mutating tools that aren't pre-approved;
/// a non-TTY impl must fail closed (Deny). `Send` so the chat TUI can drive the
/// engine on a spawned task.
pub trait AgentUi: Send {
    /// Called once before each LLM turn (before any text) — for a "thinking…"
    /// indicator. Default no-op so non-rendering impls can ignore it.
    fn turn_start(&mut self) {}
    /// Live context-window fill for the in-flight turn, so a UI can move its
    /// usage stat mid-turn instead of waiting for `footer`. `measured` is true
    /// when `tokens` is a provider-reported step total (exact); false when it is
    /// a chars/4 estimate of the request the engine is about to send (system
    /// prompt + tool schemas + conversation), emitted before usage is known.
    /// Default no-op.
    fn context_usage(&mut self, _tokens: u64, _measured: bool) {}
    /// The turn's cumulative generated (output) tokens so far, summed across
    /// steps — for a live per-turn counter. Default no-op.
    fn turn_tokens(&mut self, _output: u64) {}
    /// Prompt the user for the next turn (REPL). `None` ends the session (EOF /
    /// `/exit`). Default `None` → one-shot only (used by tests/non-interactive).
    fn read_user_input(&mut self) -> Option<String> {
        None
    }
    fn assistant_text(&mut self, delta: &str);
    /// A streamed reasoning/thinking delta (separate from the visible reply). The
    /// UI may render it in a muted "Thinking" block; default no-op so non-rendering
    /// impls (and impls that don't surface thinking) ignore it.
    fn assistant_reasoning(&mut self, _delta: &str) {}
    /// The agent set or updated its task plan via the `update_plan` tool. The UI
    /// renders this as a checklist card instead of a generic tool step. Default
    /// no-op so non-rendering impls ignore it.
    fn plan_updated(&mut self, _items: &[PlanItem]) {}
    fn tool_start(&mut self, name: &str, args: &Value);
    fn tool_result(&mut self, name: &str, result: &Result<String, String>);
    fn notify(&mut self, text: &str);
    /// Like [`notify`](Self::notify) but for a genuine error, so a UI can use an
    /// error hue. Default delegates to `notify`.
    fn notify_error(&mut self, text: &str) {
        self.notify(text);
    }
    /// End-of-turn line: an optional summary plus this turn's stats.
    /// `tokens` is the cumulative work across all steps (prompt re-counted each
    /// step); `context_tokens` is the *last* step's prompt+completion — the real
    /// context-window fill after the turn (0 when no usage was reported).
    fn footer(
        &mut self,
        summary: Option<&str>,
        steps: usize,
        tokens: u64,
        context_tokens: u64,
        elapsed_secs: u64,
    );
    /// Decide whether a mutating tool may run. Async so a TUI can await a
    /// permission card via the event loop; the terminal impl resolves it
    /// synchronously inside the returned future. Must fail closed off a TTY.
    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        preview: Option<&'a str>,
    ) -> BoxFuture<'a, Decision>;
}

/// A source of extra tools beyond the built-ins — currently MCP servers (see
/// `agent::mcp`). The engine offers `specs()` alongside its own tool schemas and
/// routes any call it `handles()` to `call()`. Kept abstract so the engine stays
/// free of process/transport knowledge. `Send + Sync` so it rides on the spawned
/// engine task and can be shared.
pub trait ExternalTools: Send + Sync {
    /// OpenAI tool schemas to advertise (already `mcp__server__tool`-named).
    fn specs(&self) -> Vec<Value>;
    /// Whether this source owns `name` (so the engine routes it here, not to the
    /// built-in executor).
    fn handles(&self, name: &str) -> bool;
    /// Whether a call to `name` should be permission-gated (e.g. an MCP server the
    /// user marked untrusted). Default `false` — configured sources are trusted.
    fn requires_approval(&self, _name: &str) -> bool {
        false
    }
    /// Execute one tool call. The result string is fed back to the model as the
    /// tool result (Err is rendered as an error but still continues the loop).
    fn call<'a>(&'a self, name: &'a str, args: &'a Value) -> BoxFuture<'a, Result<String, String>>;
}

/// Per-turn I/O the engine needs: the loopback serve to reach the provider and
/// the working directory tools execute against. (Model, history, and limits are
/// owned by the engine itself.)
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
    /// True when mutating tools should run without a prompt — the static `-y`
    /// flag, or the live chat toggle checked fresh on each call.
    pub fn auto_approve_enabled(&self) -> bool {
        self.yes
            || self
                .auto_approve
                .is_some_and(|f| f.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// A `/rewind` turn boundary: the `messages` index of the turn's opening user
/// message (truncation point) plus the working-tree snapshot taken at turn start.
/// `tree` is `None` when git is unavailable or the guard skipped it (conversation
/// only). `msg_index` stays valid across compaction via `rebase_checkpoints`.
///
/// `prompt` is the opening user text, stored verbatim so the picker can match it
/// to a display row — `messages[msg_index]` is later mutated in place (merge on
/// resend, summary fold on compaction) and would stop matching.
///
/// `changed` is the paths the turn modified (recorded at turn end); a rewind
/// reverts only the union of these across rewound turns, leaving the user's
/// independent edits alone. `None` until recorded — an interrupted turn is
/// finalized lazily by [`AgentEngine::rewind_to`].
#[derive(Clone)]
struct Checkpoint {
    msg_index: usize,
    prompt: String,
    tree: Option<String>,
    changed: Option<Vec<std::path::PathBuf>>,
}

/// The result of applying a [`AgentEngine::rewind_to`] — counts for the notice.
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
/// Zero rendering and zero direct provider knowledge — those flow through
/// `TurnCtx`/`AgentUi`.
pub struct AgentEngine {
    model: String,
    tools_openai: Vec<Value>,
    messages: Vec<Value>,
    context_window: u32,
    max_steps: usize,
    /// Actions the user approved for the rest of the session ("always"), keyed by
    /// [`permission_key`] — scoped to the specific command/path, not the whole
    /// tool, so approving one risky action doesn't whitelist every future one.
    always: HashSet<String>,
    /// Discovered SKILL.md skills, loaded on demand via the `skill` tool.
    skills: Vec<Skill>,
    /// Named specialist sub-agents discovered from `~/.config/aivo/agents`
    /// (top-level engine only — sub-engines drop the `subagent` tool). The
    /// `subagent` tool's `agent` field selects one; `run_subagent` applies its
    /// model + instructions + tool scope. Empty when none are authored.
    subagents: Vec<Subagent>,
    /// Kept so the `subagent` tool can build a fresh sub-engine with the same
    /// identity (date + project guides).
    date: String,
    guides: Vec<String>,
    /// Extra tools beyond the built-ins (MCP servers), if any are configured.
    external: Option<std::sync::Arc<dyn ExternalTools>>,
    /// Body of the last applied compaction summary (without the
    /// "[Summary of earlier conversation]" prefix). Fed back to the summarizer on
    /// the next compaction so earlier facts carry forward faithfully instead of
    /// being re-lossily re-compressed.
    last_summary: Option<String>,
    /// Latest plan from the most recent `update_plan` call. Pinned into every
    /// compaction fold so the task plan survives verbatim.
    plan: Vec<PlanItem>,
    /// Files the agent has read/written/edited this session (insertion order,
    /// deduped, capped). Maintained incrementally so the list isn't lost when the
    /// turns that touched them are summarized away; pinned into every compaction.
    touched_files: Vec<String>,
    /// The agent's durable scratchpad: notes it appends via `take_note` during a
    /// long task. Pinned verbatim into every compaction and rebuilt from the log
    /// on resume, so they outlive the turns and summaries — the agentic-memory
    /// pattern for long-horizon work. Capped at [`MAX_NOTES`] (oldest dropped).
    notes: Vec<String>,
    /// Provider-measured token split for the LAST run turn (prompt / completion /
    /// cache), summed across all of the turn's steps. The chat TUI drains this
    /// after each turn (`take_turn_usage`) to record real per-session tokens in
    /// the chat index, so `aivo stats --since` can attribute chat usage. Reset at
    /// the start of every turn.
    turn_usage: SessionTokens,
    /// `/rewind` support: one checkpoint per live `run_turn`, in order. The chat
    /// TUI maps its display turns onto these by matching prompt text from the
    /// newest backward (see [`AgentEngine::rewind_targets`]) — robust to history
    /// trimming, compaction, and rebuilds, which a positional index isn't.
    /// In-memory (the shadow tree objects live in `checkpoint_store`).
    checkpoints: Vec<Checkpoint>,
    /// Tree-level file snapshot/restore via a shadow git store. `None` until
    /// `/rewind` checkpointing is enabled for this engine (top-level chat only —
    /// sub-engines don't checkpoint). See [`crate::agent::checkpoint`].
    checkpoint_store: Option<crate::agent::checkpoint::CheckpointStore>,
    /// `reasoning_effort` to request for a reasoning-capable model, or `None`
    /// otherwise (sending the field then 400s some providers). Defaults from the
    /// model's snapshot capability at construction; changed live by `/effort`.
    reasoning_effort: Option<String>,
    /// Catalog-advertised effort levels, set per turn by the chat layer; used to
    /// pick a provider-valid "off" effort. Empty when unknown. See `thinking_request`.
    reasoning_efforts: Vec<String>,
    /// Whether the model is asked to think this turn. Off makes
    /// [`Self::thinking_request`] emit a disable signal instead of the level.
    /// Set per turn from the chat `/config` toggle.
    thinking_enabled: bool,
    /// Whether this model can reason at all (from the model-limits snapshot).
    /// Cached at construction (the engine is rebuilt on model switch) so the
    /// disable path doesn't send an effort field to a model that would 400 on it.
    reasoning_capable: bool,
    /// Plan mode: mutating tools (writes, edits, `run_bash`, subagent) are refused
    /// so a `/plan` investigation can't modify the workspace. See `restrict_read_only`.
    read_only: bool,
}

/// The reasoning-effort level to default to: `AIVO_AGENT_REASONING_EFFORT` if
/// set, else `"medium"`. Whether it's actually requested depends on model
/// capability — see [`default_reasoning_effort`]. Exposed so the chat footer can
/// show the same level the engine will send when the user hasn't picked one.
pub fn default_reasoning_effort_level() -> String {
    std::env::var("AIVO_AGENT_REASONING_EFFORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| "medium".to_string())
}

/// The `reasoning_effort` to request for `model`, or `None` for non-reasoning
/// models (where the field would 400 strict providers). Capability comes from the
/// model-limits snapshot; the level is [`default_reasoning_effort_level`].
fn default_reasoning_effort(model: &str) -> Option<String> {
    crate::services::model_metadata::snapshot_limits(model)
        .is_some_and(|c| c.reasoning)
        .then(default_reasoning_effort_level)
}

impl AgentEngine {
    /// Seed an engine with the identity system prompt. `guides` are the names of
    /// project convention files present in cwd — the agent reads them on demand
    /// (we don't inject their contents). `context_window` (0 = unknown →
    /// compaction falls back to [`DEFAULT_CONTEXT_WINDOW`]) honors an env
    /// override; `max_steps` is the per-turn tool-step budget (0 = no cap — the
    /// default for an interactive turn).
    pub fn new(
        cwd: &str,
        model: &str,
        date: &str,
        guides: &[String],
        skills: &[Skill],
        context_window: u32,
        max_steps: u32,
    ) -> Self {
        // Env override exists so compaction can be exercised without a
        // small-context model (e.g. AIVO_AGENT_CONTEXT_WINDOW=2000).
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
        let tools_openai = specs.into_iter().map(tool_to_openai).collect();
        let messages = vec![json!({
            "role": "system",
            "content": system_prompt(cwd, date, guides, skills),
        })];
        Self {
            model: model.to_string(),
            tools_openai,
            messages,
            context_window,
            max_steps,
            always: HashSet::new(),
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
            reasoning_capable: default_reasoning_effort(model).is_some(),
            read_only: false,
        }
    }

    /// Make this engine read-only for `/plan`: hide the mutating tools (and
    /// `subagent`, which could spawn an editing sub-engine) from the model. The
    /// execution guard refuses them too, in case one is hallucinated. One-way.
    pub fn restrict_read_only(&mut self) {
        self.read_only = true;
        self.tools_openai.retain(|t| {
            let name = t["function"]["name"].as_str().unwrap_or("");
            !tools::is_mutating(name) && name != "subagent"
        });
    }

    /// Set the `reasoning_effort` level (the `/effort` command). Only meaningful
    /// for reasoning-capable models.
    pub fn set_reasoning_effort(&mut self, effort: String) {
        self.reasoning_effort = Some(effort);
    }

    /// Turn the model's thinking on/off for upcoming turns (the `/config`
    /// toggle). Off makes [`Self::thinking_request`] emit a disable signal
    /// instead of the chosen level.
    pub fn set_thinking_enabled(&mut self, on: bool) {
        self.thinking_enabled = on;
    }

    /// Set the catalog-advertised effort levels for this turn. See `reasoning_efforts`.
    pub fn set_reasoning_efforts(&mut self, efforts: Vec<String>) {
        self.reasoning_efforts = efforts;
    }

    /// Whether `level` is one the model's catalog advertises (so it won't 400).
    fn effort_is_valid(&self, level: &str) -> bool {
        self.reasoning_efforts.iter().any(|e| e == level)
    }

    /// How to express thinking control on this step's request: `(reasoning_effort,
    /// emit_thinking_disabled)`.
    ///
    /// Enabled → the resolved level (or `None` for a non-reasoning model), no
    /// disable field. Disabled, for a reasoning-capable model:
    /// - **gpt-5 / codex**: `"minimal"` — they reject `"none"` alongside tools and
    ///   reject the `thinking` field.
    /// - **o-series**: `"low"` — the floor they accept (no none/minimal).
    /// - **catalog lists `none`/`minimal`**: send it — a real effort-level off.
    /// - **otherwise** (effort is depth-only, e.g. `aivo/starter`, Anthropic): emit
    ///   `thinking:{type:"disabled"}` and NO effort — `"none"` isn't in their effort
    ///   scale (400s), so the separate `thinking` field is the toggle; the
    ///   OpenAI→Anthropic bridge carries it through.
    ///
    /// Capability also accepts a chat-set level/catalog, since alias models (e.g.
    /// `aivo/starter`) are absent from the snapshot.
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
        if name.starts_with("gpt-5") || name.contains("codex") {
            (Some("minimal"), false)
        } else if name.starts_with("o1") || name.starts_with("o3") || name.starts_with("o4") {
            (Some("low"), false)
        } else if self.effort_is_valid("none") {
            (Some("none"), false)
        } else if self.effort_is_valid("minimal") {
            (Some("minimal"), false)
        } else {
            (None, true)
        }
    }

    /// Enable `/rewind` tree-checkpointing for this engine (top-level chat only;
    /// sub-engines never call this, so they don't pay the git cost). Idempotent.
    pub fn enable_rewind_checkpoints(&mut self, cwd: &str) {
        if self.checkpoint_store.is_none() {
            self.checkpoint_store = Some(crate::agent::checkpoint::CheckpointStore::new(
                std::path::Path::new(cwd),
            ));
        }
    }

    /// Drain the last turn's provider-measured token split (prompt / completion /
    /// cache, summed across the turn's steps). Returns the accumulated usage and
    /// leaves the accumulator zeroed. The chat TUI calls this after a turn to fold
    /// the real tokens into the chat session index for `aivo stats`.
    pub fn take_turn_usage(&mut self) -> SessionTokens {
        std::mem::take(&mut self.turn_usage)
    }

    /// Attach an external tool source (MCP): advertise its schemas alongside the
    /// built-ins and route its calls to it. Call once, after construction.
    pub fn set_external_tools(&mut self, ext: std::sync::Arc<dyn ExternalTools>) {
        self.tools_openai.extend(ext.specs());
        self.external = Some(ext);
    }

    /// Fill in the compaction context window if it was unknown (0) at
    /// construction. A model only known via a background catalog warm resolves
    /// its window AFTER the engine is built. Only fills a missing window — never
    /// overrides a known one (incl. the test env override).
    pub fn set_context_window(&mut self, window: u32) {
        if self.context_window == 0 && window > 0 {
            self.context_window = window;
        }
    }

    /// Register named specialist sub-agents (top-level engine only). Replaces the
    /// `subagent` tool with one whose `agent` field enumerates them, and appends a
    /// one-line advert of each to the system prompt (progressive disclosure, like
    /// skills). No-op when empty so a project without authored sub-agents keeps the
    /// plain generic `subagent` tool. Call once, after construction.
    pub fn set_subagents(&mut self, subagents: &[Subagent]) {
        if subagents.is_empty() {
            return;
        }
        // Swap the generic subagent tool for one that advertises the named agents.
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("subagent"));
        self.tools_openai
            .push(tool_to_openai(subagent_tool_spec(subagents)));
        // Advertise the names in the system prompt (messages[0]).
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

    /// Apply a named agent profile by folding its instructions into the system
    /// prompt and, when it authored a `tools` scope, restricting the offered tools
    /// to that allow-list (plus `update_plan`, always harmless). The list is an
    /// allow-list over the full offered set, so under a scope any unlisted tool —
    /// including MCP tools — is dropped. A list that resolves to nothing doesn't
    /// scope (see `Subagent::resolved_tools`). Used both for a delegated sub-agent
    /// (a freshly built sub-engine) and for a top-level `--agent`/`/agent` profile.
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
            // An authored `Edit`/`MultiEdit` scope grants `apply_patch` (its stand-in).
            let editor_allowed = allowed.contains(&"edit_file") || allowed.contains(&"multi_edit");
            self.tools_openai.retain(|t| {
                let name = t["function"]["name"].as_str().unwrap_or("");
                // `update_plan`/`take_note` have no side effects outside the engine,
                // so a scoped specialist keeps planning + note-taking regardless.
                name == "update_plan"
                    || name == "take_note"
                    || allowed.contains(&name)
                    || (name == "apply_patch" && editor_allowed)
            });
        }
    }

    /// Remove the `subagent` tool from this engine's offered tools — used on a
    /// sub-engine so it can't itself spawn sub-agents (depth-1 only).
    fn drop_subagent_tool(&mut self) {
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("subagent"));
    }

    /// REPL `/clear`: drop the conversation, keep the system prompt (index 0).
    /// Also clears the compaction working set (running summary, pinned plan,
    /// touched files) — otherwise a cleared session would re-inject stale facts.
    pub fn reset(&mut self) {
        self.messages.truncate(1);
        self.last_summary = None;
        self.plan.clear();
        self.touched_files.clear();
        self.notes.clear();
        // Drop `/rewind` checkpoints: their `msg_index` pointed into the cleared
        // transcript.
        self.checkpoints.clear();
    }

    /// Seed prior conversation into a freshly built engine (resume, or a mid-chat
    /// model/key switch) so it isn't amnesiac. Only user/assistant text turns are
    /// carried — tool steps are display-only and lack the call IDs needed to
    /// rebuild valid tool messages. No-op once a turn has run (keeps the seed to
    /// engine construction time).
    pub fn seed_history(&mut self, turns: impl IntoIterator<Item = (String, String)>) {
        let mut seen_user = false;
        for (role, content) in turns {
            if !matches!(role.as_str(), "user" | "assistant") {
                continue;
            }
            // The conversation must open with a user turn — Anthropic (via the
            // serve bridge) rejects an assistant-first sequence, so drop leading
            // assistant turns until the first user.
            if !seen_user {
                if role != "user" {
                    continue;
                }
                seen_user = true;
            }
            self.push_text_turn(&role, content);
        }
    }

    /// Export the conversation (everything after the system prompt) as raw
    /// OpenAI-format messages — assistant `tool_calls` and `tool` results with
    /// their ids intact — for durable persistence. The system prompt (index 0) is
    /// omitted; it's rebuilt fresh on restore (cwd/date/guides/skills/agent may
    /// have changed). Empty before any turn has run.
    pub fn export_conversation(&self) -> Vec<Value> {
        self.messages.iter().skip(1).cloned().collect()
    }

    /// Restore a previously [`export_conversation`]ed transcript into a freshly
    /// built engine (resume), appending it after the system prompt verbatim so
    /// tool-call history (and any folded compaction summary) survives exactly. A
    /// no-op unless the engine is fresh (only the system prompt present) — never
    /// call after a turn has run or `seed_history` was used. The exported tail came
    /// from a converged engine, so it's already a valid alternating sequence;
    /// `run_turn`'s `repair_interrupted_tail` heals it if it was captured mid-tool.
    pub fn restore_conversation(&mut self, conversation: Vec<Value>) {
        if self.messages.len() != 1 {
            return;
        }
        // These turns predate this engine: no `checkpoints` entry, so the TUI's
        // prompt back-match marks them conversation-only (no file revert).
        self.messages.extend(conversation);
        self.rebuild_working_set_from_log();
    }

    /// Re-derive the live working set — plan, notes, touched files — from the
    /// restored message log, so a resumed session continues with the state it had
    /// rather than an amnesiac one. This is the stateless-reducer property
    /// (12-factor: unify execution state with the log): the message log is the
    /// single source of truth, and these fields are a deterministic reduction over
    /// it. Reads the surviving `update_plan` / `take_note` / file-tool calls in
    /// order; calls that were folded into a compaction summary live on as text in
    /// the restored messages, so nothing the model can see is lost. Only meaningful
    /// right after restore (the fields start empty on a fresh engine).
    fn rebuild_working_set_from_log(&mut self) {
        // Collect first (immutable borrow), then apply (mutable) — `record_touched_file`
        // borrows `self` mutably.
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

    /// Append a user/assistant text turn, MERGING into the previous message when
    /// it has the same role (and is plain text — not a `tool_calls`-bearing
    /// assistant). The engine must never hold two consecutive same-role
    /// messages: the OpenAI→Anthropic bridge forwards them verbatim and Anthropic
    /// 400s on non-alternating roles (a non-retryable brick). Adjacent same-role
    /// turns arise from seeding a history that already has them (a cancelled user
    /// turn followed by the next) or appending a user turn right after one.
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

    /// Restore the assistant↔tool invariant before composing a new turn. If a
    /// prior turn was torn down mid-tool (the chat TUI aborts the engine task on
    /// Esc/interrupt while a tool runs), `messages` can end with an `assistant`
    /// bearing `tool_calls` whose `tool` results were never pushed. Appending a
    /// `user` turn after that yields a sequence every provider rejects with a
    /// 400 — and since 400 isn't retryable, the corrupted prefix re-sends every
    /// turn and bricks the session. Synthesize an `[interrupted]` result for each
    /// unanswered call id so the conversation can continue (and self-heal a
    /// history already corrupted by an earlier interrupt).
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
        // Tool results must sit immediately after the call, before any non-tool
        // message — so answers (and the gap) live in that contiguous run.
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
        // The interrupted assistant never produced a textual reply to its tools,
        // so the next message is a `user` turn sitting directly after the tool
        // results. The OpenAI→Anthropic bridge maps each tool result to a `user`
        // message, so [tool_result(user), next(user)] becomes two consecutive
        // user messages — which Anthropic 400s on (non-retryable → bricks the
        // session, same family as the compaction fix). Insert a short assistant
        // turn after the results to keep roles alternating on every upstream.
        // Only when the results sit at the tail / before a non-assistant turn —
        // an assistant already following would itself be the alternating reply.
        let after_results = insert_at + missing_count;
        if self.messages.get(after_results).map(role) != Some("assistant") {
            self.messages.insert(
                after_results,
                json!({"role": "assistant", "content": "[interrupted]"}),
            );
        }
    }

    /// Run one user turn to convergence: call the model, execute any tool calls
    /// it makes (permission-gated), repeat until it stops calling tools or a
    /// stop condition trips (step cap / no-progress). Renders the end-of-turn
    /// footer with this turn's stats.
    /// chars/4 estimate of the next request's prompt: the system prompt + tool
    /// schemas + the conversation so far. Seeds the live context-fill before the
    /// model reports real usage — the visible chat transcript omits the system
    /// prompt and tool definitions, which dominate an agent prompt.
    fn estimated_prompt_tokens(&self) -> u64 {
        let msg_chars: usize = self.messages.iter().map(|m| m.to_string().len()).sum();
        let tool_chars: usize = self.tools_openai.iter().map(|t| t.to_string().len()).sum();
        ((msg_chars + tool_chars) / 4) as u64
    }

    pub async fn run_turn(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi, user_text: String) {
        self.repair_interrupted_tail();
        // Record a `/rewind` checkpoint at the user message that opens this turn.
        // `push_text_turn` merges into a trailing `user` message rather than
        // appending, so when the tail is already `user` the turn starts there;
        // otherwise it starts at the message about to be appended.
        let turn_start = if self.messages.last().map(role) == Some("user") {
            self.messages.len().saturating_sub(1)
        } else {
            self.messages.len()
        };
        // When merging into an interrupted turn (a checkpoint already sits at this
        // index), reuse it: a second checkpoint would alias `msg_index` (breaking
        // the back-match) and snapshot a tree with the interrupted turn's partial
        // edits. The existing pre-edit tree is the right rewind target.
        let already_checkpointed = self.checkpoints.last().map(|c| c.msg_index) == Some(turn_start);
        if !already_checkpointed {
            // The tree snapshot is taken lazily — only once this turn is about to
            // touch the workspace (see `execute_tool_batch`) — so a read-only or
            // pure-Q&A turn pays no git cost. `tree` is filled in then; `None` here
            // and `None` forever for a turn that never mutates.
            self.checkpoints.push(Checkpoint {
                msg_index: turn_start,
                // Before `push_text_turn` may merge a resend in (see `Checkpoint`).
                prompt: user_text.clone(),
                tree: None,
                changed: None,
            });
        }
        // Merge into a preceding user turn rather than appending a second one —
        // e.g. after a turn cancelled before its first reply, the tail is still a
        // `user` message; a bare append would be two consecutive user messages
        // (Anthropic 400 / brick).
        self.push_text_turn("user", user_text);

        let mut steps = 0usize;
        let mut tokens = 0u64;
        // Real provider-measured split, summed across this turn's steps (drained
        // by the TUI into the chat index for stats). Reset per turn.
        self.turn_usage = SessionTokens::default();
        // Last step's prompt+completion — the real context-window fill after the
        // turn (the cumulative `tokens` above re-counts the prompt every step).
        let mut context_tokens = 0u64;
        let started = Instant::now();
        let mut last_batch = String::new();
        let mut repeats = 0usize;
        let mut converged = false;

        for _ in 0..self.max_steps {
            // Compact before composing the request if we'd otherwise overflow.
            tokens += self.maybe_compact(ctx, ui).await;

            let mut extra = Map::new();
            extra.insert("tool_choice".into(), json!("auto"));
            // Thinking control for this step (see `thinking_request`). The loopback
            // serve translates `reasoning_effort` to each upstream's surface
            // (Anthropic `thinking`, Gemini `thinkingConfig`, OpenAI passthrough);
            // `thinking:{type:"disabled"}` is the separate off-switch where the
            // effort scale has no "off".
            let (effort, disable_thinking) = self.thinking_request();
            if let Some(effort) = effort {
                extra.insert("reasoning_effort".into(), json!(effort));
            }
            if disable_thinking {
                extra.insert("thinking".into(), json!({ "type": "disabled" }));
            }
            let request = ChatRequest {
                model: self.model.clone(),
                messages: self.messages.clone(),
                tools: self.tools_openai.clone(),
                extra,
            };

            ui.turn_start();
            // Seed the live context-fill with a request estimate (system prompt +
            // tools + conversation) so the UI's stat is realistic before the model
            // reports usage; the measured total replaces it once the step returns.
            ui.context_usage(self.estimated_prompt_tokens(), false);
            // Auto-retry transient failures (rate limit / overload / 5xx / network)
            // with exponential backoff — but only when nothing streamed yet, since
            // re-streaming would duplicate the rendered text.
            let mut retries = 0usize;
            let message = loop {
                let mut streamed = false;
                let result = serve_client::complete(
                    ctx.client,
                    ctx.serve_base,
                    ctx.auth,
                    &request,
                    &mut |delta| {
                        // Any streamed output (text OR reasoning) means a retry
                        // would double-render, so guard the retry on `streamed`.
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
                    Err(e) if retries < MAX_RETRIES && !streamed && is_retryable_error(&e) => {
                        retries += 1;
                        ui.notify(&format!(
                            "connection issue — retrying ({retries}/{MAX_RETRIES})…"
                        ));
                        tokio::time::sleep(retry_delay(retries)).await;
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
            }
            // Sum the real prompt/completion/cache split across steps (same parser
            // the loopback serve accounts with, so the index stays consistent with
            // the lifetime per-tool counters).
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

            // An empty completion (no text, no tool calls — content filters,
            // empty refusals, a provider hiccup) converges the turn. Don't record
            // it as an assistant turn: an empty assistant becomes an empty
            // content array for Anthropic via the serve bridge, which 400s on it
            // (non-retryable → bricks the next turn). Nothing was produced, so
            // there's nothing to keep.
            let no_output = message.tool_calls.is_empty()
                && message.content.as_deref().is_none_or(str::is_empty);
            if no_output {
                converged = true;
                break;
            }
            self.messages.push(assistant_to_openai(&message));

            if message.tool_calls.is_empty() {
                converged = true; // the model answered without calling tools
                // Engine owns plan state: on a real convergence (the model gave its
                // final answer), finalize a started plan so it can't linger as
                // "0/N done" when the model forgot to flip the last steps. Gated on
                // `started` — an all-pending plan means the model planned but
                // converged without executing (e.g. it asked the user something),
                // so we leave that untouched.
                if plan::started(&self.plan) && plan::complete_all(&mut self.plan) {
                    ui.plan_updated(&self.plan);
                }
                break;
            }

            // No-progress guard: count identical consecutive tool-call batches.
            let batch = batch_sig(&message.tool_calls);
            if batch == last_batch {
                repeats += 1;
            } else {
                repeats = 0;
                last_batch = batch;
            }

            // Execute this batch of tool calls (permission-gated, parallel-safe
            // built-ins fanned out, the rest ordered), appending each result to
            // `messages` in call order. Returns any extra tokens accrued inside the
            // batch (sub-agent LLM calls), which the cumulative turn total absorbs.
            tokens += self.execute_tool_batch(ctx, ui, &message.tool_calls).await;

            if repeats + 1 >= REPEAT_LIMIT {
                ui.notify("stopping: the model repeated the same action with no progress");
                converged = true;
                break;
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

        // Record which paths this turn changed, so a later `/rewind` reverts only
        // the agent's edits (not the user's). Interrupted turns skip this and are
        // finalized lazily by `rewind_to`.
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

    /// Execute one assistant turn's batch of tool calls, appending a `tool`
    /// message for each in call order. Must stay behavior-identical to the inline
    /// loop it replaced: classify + permission-gate up front (in call order), run
    /// the side-effect-free built-ins concurrently and the rest sequentially, then
    /// report results in call order. Returns the extra tokens accrued by any
    /// sub-agent runs (the caller folds them into the turn total).
    async fn execute_tool_batch(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        tool_calls: &[ToolCall],
    ) -> u64 {
        // Lazy `/rewind` checkpoint: take the pre-edit tree snapshot now, the first
        // time a turn runs a batch that isn't entirely read-only. Conservative —
        // anything not on the `is_read_only` allowlist (writes, run_bash, subagent,
        // MCP) triggers it, so we never miss a mutation; a fully read-only batch
        // doesn't. The snapshot is the turn-start tree (reads didn't mutate it).
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
        let mut outcomes: Vec<Option<Result<String, String>>> = vec![None; tool_calls.len()];
        let mut parallel_idx: Vec<usize> = Vec::new();
        let mut sequential_idx: Vec<usize> = Vec::new();

        for (i, call) in tool_calls.iter().enumerate() {
            // The plan tool renders as its own checklist card, not a generic
            // tool step, and never needs permission — resolve it up front. Its
            // result still joins history below so the call↔result invariant holds.
            if call.name == "update_plan" {
                let content = match plan::parse_plan(&call.arguments) {
                    Ok(mut items) => {
                        // The engine owns plan progression: fill in steps the
                        // model advanced past but forgot to mark done, so the
                        // checklist stays monotone (and the confirmation echoes
                        // the corrected view back to the model).
                        plan::normalize_progress(&mut items);
                        // Retain the latest plan so it survives compaction.
                        self.plan = items.clone();
                        ui.plan_updated(&items);
                        plan::confirmation(&items)
                    }
                    Err(e) => e,
                };
                outcomes[i] = Some(Ok(content));
                continue;
            }
            ui.tool_start(&call.name, &call.arguments);
            // Plan mode backstop (the tool is also hidden); the error steers the model.
            if self.read_only && tools::is_mutating(&call.name) {
                outcomes[i] = Some(Err(
                    "Plan mode is read-only — do not modify files or run commands. \
Investigate with read-only tools and write the implementation plan instead."
                        .to_string(),
                ));
                continue;
            }
            // Confirm only genuinely risky actions: a destructive command, an
            // out-of-cwd write, a blind overwrite of an existing file the model
            // never read, or an external (MCP) tool whose server the user
            // marked untrusted. Everything else runs uninterrupted.
            let needs_confirm = tools::is_dangerous(&call.name, &call.arguments, ctx.cwd)
                || self.write_clobbers_unread(&call.name, &call.arguments, ctx.cwd)
                || self
                    .external
                    .as_ref()
                    .is_some_and(|e| e.requires_approval(&call.name));
            let pkey = permission_key(&call.name, &call.arguments);
            let allowed =
                if !needs_confirm || ctx.auto_approve_enabled() || self.always.contains(&pkey) {
                    true
                } else {
                    let preview = tools::preview(&call.name, &call.arguments);
                    match ui.ask_permission(&call.name, preview.as_deref()).await {
                        Decision::Allow => true,
                        Decision::AlwaysAllow => {
                            self.always.insert(pkey);
                            true
                        }
                        Decision::Deny => false,
                    }
                };
            if !allowed {
                outcomes[i] = Some(Err("denied by user".to_string()));
                continue;
            }
            // A side-effect-free built-in can run concurrently — unless an
            // external (MCP) tool shadows the same name, in which case it must
            // route to its source sequentially.
            let shadowed = self
                .external
                .as_ref()
                .is_some_and(|e| e.handles(&call.name));
            if tools::is_parallel_safe(&call.name) && !shadowed {
                parallel_idx.push(i);
            } else {
                sequential_idx.push(i);
            }
        }

        // Fan out the side-effect-free calls. They share no mutable state, so
        // we just poll them together on this task — no spawn, no Send bound.
        if !parallel_idx.is_empty() {
            let cwd = ctx.cwd;
            let runs = parallel_idx.iter().map(|&i| {
                let call = &tool_calls[i];
                async move { (i, tools::execute(&call.name, &call.arguments, cwd).await) }
            });
            for (i, result) in futures::future::join_all(runs).await {
                outcomes[i] = Some(result);
            }
        }

        // Run the ordered calls one at a time — these mutate the engine
        // (subagent token folding) or the workspace, so concurrency is unsafe.
        for &i in &sequential_idx {
            let call = &tool_calls[i];
            let result = if call.name == "skill" {
                // Resolved from the engine's discovered skills, not tools::execute.
                let name = call
                    .arguments
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                skills::load_skill_result(&self.skills, name)
            } else if call.name == "subagent" && self.read_only {
                // A sub-engine isn't read-only; refuse delegation in plan mode.
                Err(
                    "Plan mode is read-only — cannot delegate to a subagent while planning."
                        .to_string(),
                )
            } else if call.name == "subagent" {
                // Fresh sub-engine on the same serve/cwd; fold its total into the
                // footer + turn usage below. Pass the UI + output base so it
                // forwards live token growth.
                let base = self.turn_usage.completion_tokens;
                match self.run_subagent(ctx, ui, base, &call.arguments).await {
                    Ok((msg, sub_tokens)) => {
                        extra_tokens += sub_tokens;
                        self.turn_usage.completion_tokens =
                            self.turn_usage.completion_tokens.saturating_add(sub_tokens);
                        Ok(msg)
                    }
                    Err(e) => Err(e),
                }
            } else if call.name == "take_note" {
                // Durable scratchpad: append to notes (capped, oldest dropped).
                // Pinned into compaction + rebuilt on resume so it outlives the
                // turns. Held in the engine, so it runs in the ordered pass.
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
            } else if let Some(ext) = self.external.clone().filter(|e| e.handles(&call.name)) {
                // MCP (or other) external tool — routed to its source.
                ext.call(&call.name, &call.arguments).await
            } else if call.name == "run_bash" {
                // Run confined; if the sandbox blocks a write, offer an
                // in-session escape hatch instead of a dead-end error.
                self.run_bash_with_escalation(ctx, ui, &call.arguments)
                    .await
            } else {
                tools::execute(&call.name, &call.arguments, ctx.cwd).await
            };
            outcomes[i] = Some(result);
        }

        // Emit results and append tool messages in the original call order so
        // the call↔result pairing the providers require stays intact.
        for (i, call) in tool_calls.iter().enumerate() {
            let result = outcomes[i]
                .take()
                .unwrap_or_else(|| Err("tool produced no result".to_string()));
            // update_plan already surfaced via plan_updated; the rest report here.
            if call.name != "update_plan" {
                ui.tool_result(&call.name, &result);
            }
            if result.is_ok() {
                self.record_touched_file(&call.name, &call.arguments);
            }
            let content = match result {
                Ok(c) => c,
                Err(e) => e,
            };
            self.messages.push(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            }));
        }

        extra_tokens
    }

    /// Run a `run_bash` call confined to the workspace. If the OS sandbox blocks
    /// a write (the command tried to write outside the workspace), offer to
    /// re-run it *outside* the sandbox — gated by the same approval flow as other
    /// risky actions — instead of returning a dead-end error that makes the model
    /// give up and tell the user to run the command by hand. Auto-approve (`-y` /
    /// the live toggle) and a prior "always" both skip the prompt; off a TTY the
    /// UI fails closed (Deny), so the blocked result (with its hint) flows back.
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
        // Scoped to the exact command so "always" doesn't blanket-escalate every
        // future bash call — mirrors `permission_key`'s per-command scoping.
        let ekey = format!("run_bash_unsandboxed\u{0}{command}");
        let approved = ctx.auto_approve_enabled() || self.always.contains(&ekey) || {
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
                    self.always.insert(ekey);
                    true
                }
                Decision::Deny => false,
            }
        };
        if !approved {
            // Keep the blocked output + hint so the model sees the escalation was
            // declined (rather than a silent success).
            return outcome.result;
        }
        ui.notify("re-running outside the workspace sandbox (approved)");
        tools::run_bash_unconfined(args, ctx.cwd).await
    }

    /// The window `maybe_compact` budgets against: the real context window, or
    /// [`DEFAULT_CONTEXT_WINDOW`] when it's unknown (0).
    fn compaction_window(&self) -> usize {
        if self.context_window == 0 {
            DEFAULT_CONTEXT_WINDOW
        } else {
            self.context_window as usize
        }
    }

    /// If the history would overflow the model's context window, summarize the
    /// older messages (via a quiet `complete`) and replace them with the
    /// summary. Cuts only at user-turn boundaries so tool-call/result pairs stay
    /// intact; falls back to [`DEFAULT_CONTEXT_WINDOW`] when the window is unknown
    /// (0). Returns tokens the summarization call consumed (counted toward the
    /// turn, but not as a step).
    async fn maybe_compact(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi) -> u64 {
        let budget = self.compaction_window().saturating_sub(COMPACT_RESERVE);
        let total = estimate_tokens(&self.messages);
        if total <= budget {
            return 0;
        }
        let keep_recent = std::env::var("AIVO_AGENT_KEEP_RECENT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(KEEP_RECENT_TOKENS);
        let cut = find_cut(&self.messages, keep_recent);

        // Cheap pass first: if clearing the bulky raw output of OLD tool messages
        // (file dumps, command output the model has already acted on, sitting
        // before the recent keep window) is enough on its own, do that and skip
        // the LLM summary entirely — no model round-trip. Only taken when it alone
        // brings us under budget, so the summary path below still sees full
        // content whenever a real summary is actually needed.
        let savings = self.stale_tool_result_savings(cut);
        if savings > 0 && total.saturating_sub(savings) <= budget {
            ui.notify("freed context — cleared older tool output");
            self.clear_stale_tool_results(cut);
            return 0;
        }

        let mut tokens = 0u64;
        // Prefer an LLM summary of the old turns. Only attempt it when there's a
        // real user boundary past the system prompt to fold; otherwise fall
        // straight through to the mechanical backstop below.
        if cut > 1 {
            let transcript = serialize_transcript(&self.messages[1..cut]);
            let request = self.build_summary_request(&transcript);
            ui.notify("compacting context…");
            match serve_client::complete(
                ctx.client,
                ctx.serve_base,
                ctx.auth,
                &request,
                &mut |_| {},
            )
            .await
            {
                Ok(m) => {
                    tokens = usage_tokens(&m.usage);
                    let summary = m.content.unwrap_or_default();
                    if summary.trim().is_empty() {
                        // Empty summary — fold a mechanical note rather than leave
                        // the history overflowed.
                        let note = self.mechanical_summary();
                        self.apply_compaction(cut, &note);
                    } else {
                        self.apply_compaction(cut, &summary);
                        // Carry this summary forward so the next compaction updates
                        // it in place instead of re-summarizing a blob that already
                        // contains it (anti-drift).
                        self.last_summary = Some(summary);
                    }
                }
                Err(_) => {
                    // Summarization failed. Do NOT give up and re-send an
                    // overflowed request — context-overflow errors aren't
                    // retryable, so that would brick the turn (and, since the bad
                    // prefix re-sends every turn, the whole session). Drop the old
                    // transcript mechanically instead.
                    ui.notify("compaction summary unavailable — trimming older context");
                    let note = self.mechanical_summary();
                    self.apply_compaction(cut, &note);
                }
            }
        }
        // Backstop: guarantee the next request actually fits the window. A single
        // summary pass can fall short when the recent tail is itself huge, and
        // `cut <= 1` means nothing was old enough to fold even though we're over
        // budget. Trim deterministically (no model call) so a turn is always
        // sendable.
        self.enforce_budget(budget);
        tokens
    }

    /// Tokens reclaimable by [`clear_stale_tool_results`]: for each OLD
    /// (`messages[1..cut]`) `tool` message whose body exceeds the clear threshold,
    /// the bytes dropped when its content is replaced by the stub, as a chars/4
    /// estimate. The message overhead (role, id) stays in both the before and
    /// after, so this is the accurate saving — `maybe_compact` takes the cheap
    /// path only when it suffices.
    fn stale_tool_result_savings(&self, cut: usize) -> usize {
        self.messages
            .get(1..cut)
            .unwrap_or(&[])
            .iter()
            .filter(|m| role(m) == "tool")
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .filter(|s| s.len() > TOOL_RESULT_CLEAR_MIN)
            .map(|s| s.len().saturating_sub(TOOL_RESULT_CLEARED.len()) / 4)
            .sum()
    }

    /// Replace the bulky raw output of OLD (`messages[1..cut]`) `tool` messages
    /// with [`TOOL_RESULT_CLEARED`], reclaiming context without a model call. The
    /// message and its `tool_call_id` stay (pairing intact); only stale bytes go.
    /// Idempotent — an already-stubbed result is below the threshold and skipped.
    fn clear_stale_tool_results(&mut self, cut: usize) {
        let Some(old) = self.messages.get_mut(1..cut) else {
            return;
        };
        for m in old {
            if role(m) != "tool" {
                continue;
            }
            let len = m
                .get("content")
                .and_then(|c| c.as_str())
                .map_or(0, str::len);
            if len > TOOL_RESULT_CLEAR_MIN {
                m["content"] = json!(TOOL_RESULT_CLEARED);
            }
        }
    }

    /// A model-free stand-in for an LLM summary, used when summarization fails or
    /// returns empty. Preserves any running summary so the thread isn't lost, and
    /// notes that older turns were dropped without a fresh one.
    fn mechanical_summary(&self) -> String {
        match &self.last_summary {
            Some(prev) => {
                format!("{prev}\n\n[Additional earlier turns omitted — summarization unavailable.]")
            }
            None => "[Earlier conversation omitted — summarization unavailable.]".to_string(),
        }
    }

    /// Last-resort, model-free trim guaranteeing `messages` fits `budget`. First
    /// drops whole oldest turns at user boundaries (keeping the call↔result
    /// pairing intact); then, if a single turn is still irreducibly large (a giant
    /// pasted message or tool result), shortens string contents largest first.
    /// Always terminates and never touches the system prompt or any `tool_calls`.
    fn enforce_budget(&mut self, budget: usize) {
        while estimate_tokens(&self.messages) > budget {
            let cut = find_cut(&self.messages, 0);
            if cut <= 1 {
                break; // only [system, last user turn] left — no boundary to drop
            }
            self.messages.drain(1..cut);
            self.rebase_checkpoints(cut, cut - 1); // keep checkpoint indices valid
        }
        // Threshold well above truncate_str's marker so each pass strictly shrinks
        // the history; bail when nothing sizeable remains.
        while estimate_tokens(&self.messages) > budget {
            let pick = self
                .messages
                .iter()
                .enumerate()
                .skip(1)
                .filter_map(|(i, m)| {
                    m.get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| (i, s.chars().count()))
                })
                .filter(|&(_, n)| n > 256)
                .max_by_key(|&(_, n)| n);
            let Some((idx, n)) = pick else { break };
            let cur = self.messages[idx]["content"]
                .as_str()
                .unwrap_or("")
                .to_string();
            let shortened = truncate_str(&cur, n / 2);
            if shortened.len() >= cur.len() {
                break;
            }
            self.messages[idx]["content"] = json!(shortened);
        }
    }

    /// Build the throwaway 2-message summarization request. The first compaction
    /// (`last_summary` None) summarizes the cut transcript fresh; later ones feed
    /// the prior summary back and ask for an in-place update. Either way it's
    /// exactly system + one user message — never folded into `self.messages`, so
    /// it can't affect role alternation.
    fn build_summary_request(&self, transcript: &str) -> ChatRequest {
        let (system, user) = match &self.last_summary {
            Some(prev) => (
                SUMMARY_UPDATE_SYSTEM_PROMPT,
                format!(
                    "## Current running summary\n{prev}\n\n## New events since then\n{transcript}"
                ),
            ),
            None => (SUMMARY_SYSTEM_PROMPT, transcript.to_string()),
        };
        ChatRequest {
            model: self.model.clone(),
            messages: vec![
                json!({"role": "system", "content": system}),
                json!({"role": "user", "content": user}),
            ],
            tools: vec![],
            extra: Map::new(),
        }
    }

    /// Record a file the agent touched via a built-in file tool, for the pinned
    /// working set. Deduped, insertion-ordered, capped (oldest evicted).
    /// True when a `write_file` would overwrite an existing file the model hasn't
    /// read or written this session — a blind clobber worth confirming. New files
    /// pass through, as do files already in the working set (`read_file` and the
    /// write tools all record one). `edit_file`/`multi_edit` are excluded: they
    /// must read the file to match, so they're never blind.
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
        // `apply_patch` carries many paths in its V4A body; the rest carry one.
        let paths: Vec<String> = match name {
            "read_file" | "write_file" | "edit_file" | "multi_edit" => args
                .get("path")
                .and_then(|v| v.as_str())
                .map(|p| vec![p.to_string()])
                .unwrap_or_default(),
            "apply_patch" => args
                .get("input")
                .and_then(|v| v.as_str())
                .map(crate::agent::apply_patch::target_paths)
                .unwrap_or_default(),
            _ => return,
        };
        for path in paths {
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

    /// Per-checkpoint `/rewind` targets in order, for the picker: `(prompt,
    /// file_revertible)`. The TUI matches these to its display history by prompt
    /// text from the newest backward, so trims/compaction/rebuilds can't misalign
    /// the mapping. Cheap and in-memory (no git).
    pub fn rewind_targets(&self) -> Vec<(String, bool)> {
        self.checkpoints
            .iter()
            .map(|c| (c.prompt.clone(), c.tree.is_some()))
            .collect()
    }

    /// Rewind to checkpoint `ordinal`: revert the files the rewound turns changed
    /// (scoped to their union, so the user's independent edits are left alone),
    /// truncate the conversation to the turn's user message, drop the rewound
    /// checkpoints, and re-derive the working set. A `None`-tree checkpoint rewinds
    /// the conversation only.
    pub async fn rewind_to(&mut self, ordinal: usize) -> RewindOutcome {
        let mut outcome = RewindOutcome::default();
        let tree = self.checkpoints.get(ordinal).and_then(|c| c.tree.clone());
        // Union of paths every rewound turn changed; finalize any interrupted turn
        // (`changed == None`) lazily against the current tree.
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

    /// The pinned working set (plan + touched files) rendered for a compaction
    /// fold, trimmed to `PINNED_MAX_TOKENS`. The plan is kept whole; the files
    /// list is trimmed (oldest first) until the block fits — so pinning can't
    /// reintroduce overflow. Empty when there's nothing to pin.
    fn render_pinned_block(&self) -> String {
        let plan_block = plan::pinned_block(&self.plan);
        let mut notes: &[String] = &self.notes;
        let mut files: &[String] = &self.touched_files;
        loop {
            let block = compose_pinned(&plan_block, notes, files);
            if block.is_empty() || estimate_str_tokens(&block) <= PINNED_MAX_TOKENS {
                return block;
            }
            // Keep the plan whole; trim the files list (just paths) first, then
            // the notes (agent-curated, more valuable) — both oldest-first. Bail
            // when only the plan remains so we always make progress.
            if !files.is_empty() {
                files = &files[1..];
            } else if !notes.is_empty() {
                notes = &notes[1..];
            } else {
                return block;
            }
        }
    }

    /// Replace `messages[1..cut]` with the compaction summary, folding it INTO
    /// the first kept turn (a user message — `find_cut` lands on a user boundary)
    /// rather than inserting a standalone summary message before it. A standalone
    /// summary would sit immediately before that user turn as two consecutive
    /// `user` messages; the OpenAI→Anthropic bridge forwards consecutive user
    /// messages verbatim and Anthropic 400s on non-alternating roles — and since
    /// 400 isn't retryable and the corrupted prefix re-sends every turn, that
    /// would brick the agent right after a compaction. Folding keeps roles
    /// alternating on every upstream.
    fn apply_compaction(&mut self, cut: usize, summary: &str) {
        let mut folded = format!("[Summary of earlier conversation]\n{summary}");
        // Pin the plan + touched-files verbatim into the SAME fold, so they ride
        // inside the one user message and never become a standalone same-role
        // message (which would break role alternation — see the doc above).
        let pinned = self.render_pinned_block();
        if !pinned.is_empty() {
            folded.push_str("\n\n");
            folded.push_str(&pinned);
        }
        let summary = folded;
        if self.messages.get(cut).map(role) == Some("user") {
            let original = self.messages[cut]
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            self.messages[cut]["content"] = if original.is_empty() {
                json!(summary)
            } else {
                json!(format!("{summary}\n\n{original}"))
            };
            self.messages.drain(1..cut);
            self.rebase_checkpoints(cut, cut - 1); // drain removes cut-1 messages
        } else {
            // Defensive: no user turn at `cut` (shouldn't happen with find_cut) —
            // keep a standalone summary rather than drop it.
            self.messages.splice(
                1..cut,
                std::iter::once(json!({"role": "user", "content": summary})),
            );
            self.rebase_checkpoints(cut, cut.saturating_sub(2)); // splice: -cut+1, +1
        }
    }

    /// Keep `/rewind` checkpoints valid after a front-trim/compaction that removed
    /// `removed` messages over `messages[1..cut]`: drop checkpoints whose turn was
    /// folded away (`msg_index < cut`) and shift the survivors down. Without this,
    /// `rewind_to` would truncate at a stale index after a compaction.
    fn rebase_checkpoints(&mut self, cut: usize, removed: usize) {
        self.checkpoints.retain_mut(|cp| {
            if cp.msg_index >= cut {
                cp.msg_index -= removed;
                true
            } else {
                false
            }
        });
    }

    /// Execute a `subagent` tool call: build a fresh sub-engine (same tools minus
    /// `subagent`, same cwd + serve, optionally a stronger model) and run the
    /// subtask to convergence, returning its final answer. The sub-engine runs
    /// with a capturing UI — its individual tool steps don't surface in the parent
    /// transcript, only the result comes back. Dangerous ops inside it inherit the
    /// parent's auto-approve and otherwise fail closed (no interactive prompt can
    /// nest inside a tool call).
    async fn run_subagent(
        &self,
        ctx: &TurnCtx<'_>,
        parent_ui: &mut dyn AgentUi,
        base: u64,
        args: &Value,
    ) -> Result<(String, u64), String> {
        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .ok_or_else(|| "subagent: missing `task`".to_string())?;
        // A named specialist, if `agent` matches a discovered profile. Unknown
        // names fall back to a generic sub-agent (the tool's enum should prevent
        // them, but be lenient rather than fail the turn).
        let profile = args
            .get("agent")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .and_then(|n| self.subagents.iter().find(|s| s.name == n));
        // Model precedence: an explicit `model` arg overrides the profile's pinned
        // model, which overrides the parent's model.
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
        // Carry the parent's reasoning effort into delegated work, but only when
        // it's a valid level for the sub's model (they may differ) — otherwise
        // keep the sub model's own default rather than risk sending a level the
        // model rejects.
        if let Some(effort) = &self.reasoning_effort
            && crate::services::model_metadata::snapshot_limits(model)
                .is_some_and(|c| c.reasoning_efforts.iter().any(|l| l == effort))
        {
            sub.set_reasoning_effort(effort.clone());
        }
        // Share the parent's external tools (MCP) so a sub-agent can use them too,
        // reusing the same already-connected servers.
        if let Some(ext) = &self.external {
            sub.set_external_tools(ext.clone());
        }
        // Fold in the specialist's role + tool scope. Done after MCP wiring so a
        // `tools` allow-list applies to the full offered set (a scoped specialist
        // keeps only its listed built-ins — MCP tools it didn't list are dropped).
        if let Some(p) = profile {
            sub.apply_profile(p);
        }

        let mut ui = SubagentUi {
            yes: ctx.auto_approve_enabled(),
            parent: Some(parent_ui),
            base,
            ..Default::default()
        };
        // Box the recursive future (run_turn → subagent → run_turn) so the async
        // fn isn't an infinitely-sized type.
        Box::pin(sub.run_turn(ctx, &mut ui, task.to_string())).await;
        Ok((ui.result_message(), ui.tokens))
    }
}

/// The `subagent` tool — engine-handled (it needs the serve + a fresh engine, so
/// it can't live in `tools::execute`). Offered on the top-level engine only. When
/// named specialist sub-agents are discovered, an `agent` field enumerates them.
fn subagent_tool_spec(subagents: &[Subagent]) -> ToolSpec {
    let mut properties = json!({
        "task": {"type": "string", "description": "A complete, standalone instruction for the sub-agent."},
        "model": {"type": "string", "description": "Optional model id to run the sub-agent on (default: the agent's configured model, else same as you)."}
    });
    let mut description = "Delegate a self-contained subtask to a fresh sub-agent that has the same \
file/shell tools and runs its own loop, then hands back its result. Use it to keep your own context \
focused (offload a big investigation), or pass `model` to delegate hard work to a stronger model. The \
sub-agent does not see this conversation, so make `task` complete and standalone; it cannot spawn \
further sub-agents."
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

/// Capturing UI for a sub-agent run. Text is captured per step: `cur_text` holds
/// the in-flight step's text, and at each new step it rolls into `last_nonempty`.
/// The answer is the converging (last) step's text, falling back to the most
/// recent non-empty step — so a sub-agent that emits its answer in the same step
/// as its final tool call (then converges silently) doesn't lose it. Permission
/// inherits the parent's auto-approve, else denies (a nested tool call has no
/// interactive prompt).
#[derive(Default)]
struct SubagentUi<'a> {
    cur_text: String,
    last_nonempty: String,
    /// Last engine notice (LLM error / step-limit / no-progress) — surfaced when
    /// the sub-agent produces no answer, so the failure reason isn't swallowed.
    last_notice: String,
    steps: usize,
    /// The sub-agent's cumulative token usage, folded into the parent turn's total.
    tokens: u64,
    yes: bool,
    /// Forward live token growth (base + sub so-far) to the parent UI.
    parent: Option<&'a mut dyn AgentUi>,
    base: u64,
}

impl SubagentUi<'_> {
    /// The sub-agent's answer: the converging step's text, or the last non-empty
    /// step's text if it converged without emitting any of its own.
    fn answer(&self) -> &str {
        if self.cur_text.trim().is_empty() {
            self.last_nonempty.trim()
        } else {
            self.cur_text.trim()
        }
    }

    /// The tool result the parent receives: the answer (+ step count), or — when
    /// there's no answer — the failure notice (so an LLM error / step-limit isn't
    /// masked as a vague "no answer").
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
}

impl AgentUi for SubagentUi<'_> {
    fn turn_start(&mut self) {
        // A new step begins: the previous step's text (if any) becomes the
        // fallback, and the current buffer resets for this step.
        if !self.cur_text.trim().is_empty() {
            self.last_nonempty = std::mem::take(&mut self.cur_text);
        }
    }
    fn assistant_text(&mut self, delta: &str) {
        self.cur_text.push_str(delta);
    }
    fn tool_start(&mut self, _name: &str, _args: &Value) {}
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
        _tool: &'a str,
        _preview: Option<&'a str>,
    ) -> BoxFuture<'a, Decision> {
        let yes = self.yes;
        Box::pin(async move { if yes { Decision::Allow } else { Decision::Deny } })
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

fn tool_to_openai(t: ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters}
    })
}

/// Convert an assistant reply back into an OpenAI chat message for the history.
/// OpenAI requires `arguments` as a string and `content` present when there are
/// no tool calls.
fn assistant_to_openai(m: &AssistantMessage) -> Value {
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    if let Some(c) = &m.content
        && !c.is_empty()
    {
        msg.insert("content".into(), json!(c));
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|t| {
                json!({
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "arguments": serde_json::to_string(&t.arguments).unwrap_or_else(|_| "{}".into()),
                    }
                })
            })
            .collect();
        msg.insert("tool_calls".into(), json!(calls));
    } else if !msg.contains_key("content") {
        msg.insert("content".into(), json!(""));
    }
    Value::Object(msg)
}

fn batch_sig(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|c| format!("{}:{}", c.name, c.arguments))
        .collect::<Vec<_>>()
        .join("|")
}

/// Names of project-convention / AI-guide files present in `cwd`. We tell the
/// agent which exist and let it read them on demand rather than injecting their
/// contents into every turn (which would weigh down even a "hi").
pub fn discover_project_guides(cwd: &Path) -> Vec<String> {
    const NAMES: &[&str] = &[
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".cursorrules",
        ".github/copilot-instructions.md",
    ];
    NAMES
        .iter()
        .filter(|name| cwd.join(name).is_file())
        .map(|name| name.to_string())
        .collect()
}

fn system_prompt(cwd: &str, date: &str, guides: &[String], skills: &[Skill]) -> String {
    let mut p = format!(
        "You are the coding agent built into the aivo CLI. You work in `{cwd}` and have file \
and shell tools.\n\n\
Match your effort to the request: answer simple questions or greetings directly, and only \
reach for tools and project context when the task actually needs them — don't investigate or \
read guide files just to say hello.\n\n\
Bias toward doing. Your `run_bash` is a real shell with network access — fetch live data \
(e.g. `curl wttr.in/<city>` for weather, web/HTTP APIs for other lookups), inspect the system, \
run any command. If a command answers the request, run it instead of claiming you can't access \
the internet or external services, explaining how the user could do it themselves, telling them it \
\"can't be run from here,\" or asking whether to proceed. (Only destructive commands prompt for \
approval — everything else runs immediately — so never ask permission in prose.) A non-zero exit \
is normal feedback, not a wall: read the actual error and act on it — e.g. `git commit` reporting \
\"nothing added to commit\" means stage with `git add` first, and a missing tool means install it. \
If the same approach keeps failing the same way, change tactics rather than repeating it. The only \
genuinely unrunnable case is a sandbox write-block (a tool result noting writes are confined to the \
workspace), and even then the user is prompted to re-run it outside the sandbox — so keep going \
rather than handing the command back.\n\n\
That action bias is for read-only and easily-reversible local work. The approval prompt only \
catches local file and history damage — it does NOT catch outward-facing or hard-to-undo \
actions. Before you send a mutating request to a remote API (POST/PUT/DELETE), publish or \
deploy, send mail, or delete remote, cloud, or database data, say plainly what you're about to \
do and wait for the user to confirm, even though no approval card appears. And never print, log, \
hard-code, or commit secrets or credentials, and decline to write code whose evident purpose is \
malicious.\n\n\
Be resourceful: when a request is unclear or names something that isn't in the working \
directory, investigate with your tools before asking the user to clarify. `glob`, `grep`, and \
`list_dir` default to the working directory — to look elsewhere, pass an absolute path or `~`, \
or use `run_bash` (e.g. `find`, `ls`, `rg`). Only ask the user once you're genuinely stuck \
after looking.\n\n\
You are part of aivo, so you can inspect aivo itself: for questions about its API keys, models, \
providers, configuration, or usage, run the `aivo` command (e.g. `aivo keys list`, `aivo \
models`, `aivo stats`) or read the usage from `aivo --help-json`.\n\n\
Read files before editing, and make focused changes. After changing code, verify it before you \
call the task done: run the project's build, tests, and linter (find the commands in the \
convention files, README, Makefile, or build config — don't guess or invent a framework) and \
read the output. Never report a fix as working or a task as done unless you've observed it pass — \
if it comes back red, say so and fix it rather than papering over it. Report only what your tools actually returned — never invent file contents, \
command output, test results, or paths; if you don't know, say so. Don't commit, push, create \
branches, or open a PR unless the user asks; just make the changes and stop. Be concise; act \
rather than narrate. When the task is genuinely done, reply with a short summary and stop \
calling tools.\n\n\
For a task that takes several steps, call `update_plan` with a short ordered checklist up front, \
then keep it current as you go — mark each step `completed` the moment you finish it (and the next \
one `in_progress`), and send a final update marking every step `completed` once you're done so it \
never lingers as unfinished. It shows the user your progress. Don't bother for trivial one-step \
requests.\n\n\
For a long, multi-step task, use `take_note` to jot down decisions, findings, and dead-ends as \
you go — notes persist verbatim even after older conversation is compacted away, so they keep you \
oriented across many steps. Skip it for quick work.\n\n\
For a large, self-contained chunk of work — a deep investigation that would clutter your context, or \
something a stronger model should handle — you can hand it to a fresh sub-agent with `subagent` (pass \
`model` to use a stronger model) and build on its result. For ordinary steps, just use your own tools."
    );
    let os = match std::env::consts::OS {
        "macos" => "macOS",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    };
    p.push_str(&format!(
        "\n\nEnvironment: this host runs {os}; your `run_bash` runs each command through {shell}, \
so write every command in {shell} syntax — don't assume a different OS's shell.",
        shell = crate::agent::sandbox::shell_label()
    ));
    if cfg!(windows) {
        p.push_str(
            " On Windows that means PowerShell, not bash: use cmdlets/aliases (`Select-String` not \
`grep`, `Get-Content` not `cat`, `Get-ChildItem` not `find`, `curl.exe` or `Invoke-RestMethod` not \
the `curl` alias) and chain with `;` (not `&&`). Paths use `\\`.",
        );
    }
    if !guides.is_empty() {
        p.push_str(&format!(
            "\n\nThis project has convention file(s): {}. Before you create or edit ANY file here, \
read the relevant one(s) first — they may dictate file headers, style, or workflow, and you must \
follow them. (Skip them for questions, chat, or read-only exploration.)",
            guides.join(", ")
        ));
    }
    p.push_str(&skills::skills_prompt_section(skills));
    if !date.is_empty() {
        p.push_str(&format!("\n\nCurrent date: {date}."));
    }
    p
}

/// Total tokens from an OpenAI/Anthropic-style `usage` object (0 if absent).
fn usage_tokens(usage: &Option<Value>) -> u64 {
    let Some(u) = usage else {
        return 0;
    };
    if let Some(t) = u.get("total_tokens").and_then(|x| x.as_u64()) {
        return t;
    }
    let pick = |keys: &[&str]| {
        keys.iter()
            .find_map(|k| u.get(*k).and_then(|x| x.as_u64()))
            .unwrap_or(0)
    };
    pick(&["input_tokens", "prompt_tokens"]) + pick(&["output_tokens", "completion_tokens"])
}

/// Conservative token estimate: serialized JSON length / 4 (pi's heuristic).
fn estimate_tokens(messages: &[Value]) -> usize {
    messages
        .iter()
        .map(|m| serde_json::to_string(m).map(|s| s.len()).unwrap_or(0) / 4)
        .sum()
}

/// chars/4 token estimate for a plain string (same heuristic as `estimate_tokens`).
fn estimate_str_tokens(s: &str) -> usize {
    s.len() / 4
}

/// Render the pinned working set folded into a compaction: a `## Pinned Plan`
/// section (the rendered checklist), a `## Notes` scratchpad, and a `## Files
/// touched` list. Each section is omitted when empty; returns "" when all are.
fn compose_pinned(plan_block: &str, notes: &[String], files: &[String]) -> String {
    let mut out = String::new();
    if !plan_block.is_empty() {
        out.push_str("## Pinned Plan\n");
        out.push_str(plan_block);
    }
    if !notes.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("## Notes\n");
        for n in notes {
            out.push_str("- ");
            out.push_str(n);
            out.push('\n');
        }
        out = out.trim_end().to_string();
    }
    if !files.is_empty() {
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("## Files touched\n");
        for f in files {
            out.push_str("- ");
            out.push_str(f);
            out.push('\n');
        }
        out = out.trim_end().to_string();
    }
    out
}

fn role(m: &Value) -> &str {
    m.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

/// The scope an "always allow" decision is remembered under. Deliberately
/// narrow: approving one destructive command or out-of-cwd write must not
/// silently whitelist *every* future call of that tool. `run_bash` keys on the
/// exact command, file writes on the path; anything else (e.g. an untrusted MCP
/// tool, already a specific server-tool) keys on the tool name. The NUL
/// separator keeps a tool name from colliding with an argument value.
fn permission_key(name: &str, args: &Value) -> String {
    let arg = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim();
    match name {
        "run_bash" => format!("run_bash\u{0}{}", arg("command")),
        "write_file" | "edit_file" | "multi_edit" => format!("{name}\u{0}{}", arg("path")),
        // Scope to the exact set of files the patch touches, not all patches.
        "apply_patch" => format!(
            "apply_patch\u{0}{}",
            crate::agent::apply_patch::target_paths(arg("input")).join("\u{1}")
        ),
        _ => name.to_string(),
    }
}

/// Index `cut` such that `messages[cut..]` is kept — chosen at a user-turn
/// boundary nearest to `keep_recent_tokens` of recent history.
fn find_cut(messages: &[Value], keep_recent_tokens: usize) -> usize {
    let mut acc = 0usize;
    let mut cut = messages.len();
    for i in (1..messages.len()).rev() {
        acc += estimate_tokens(&messages[i..=i]);
        if role(&messages[i]) == "user" {
            cut = i;
            if acc >= keep_recent_tokens {
                break;
            }
        }
    }
    cut
}

fn content_str(m: &Value) -> String {
    m.get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max).collect();
    format!("{kept}… (+{} chars)", s.chars().count() - max)
}

/// Render messages to a plain transcript for the summarizer (tool results
/// capped at 2000 chars so the summarization request stays tractable).
fn serialize_transcript(messages: &[Value]) -> String {
    let mut out = String::new();
    for m in messages {
        match role(m) {
            "user" => out.push_str(&format!("[User]: {}\n", content_str(m))),
            "assistant" => {
                let c = content_str(m);
                if !c.is_empty() {
                    out.push_str(&format!("[Assistant]: {c}\n"));
                }
                if let Some(calls) = m.get("tool_calls").and_then(|t| t.as_array()) {
                    let rendered: Vec<String> = calls
                        .iter()
                        .filter_map(|tc| {
                            let f = tc.get("function")?;
                            let name = f.get("name")?.as_str()?;
                            let args = f.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                            Some(format!("{name}({})", truncate_str(args, 200)))
                        })
                        .collect();
                    if !rendered.is_empty() {
                        out.push_str(&format!("[Tool calls]: {}\n", rendered.join("; ")));
                    }
                }
            }
            "tool" => out.push_str(&format!(
                "[Tool result]: {}\n",
                truncate_str(&content_str(m), 2000)
            )),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::plan::PlanStatus;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;

    #[test]
    fn test_resolve_max_steps() {
        // 0 → no cap (the interactive default).
        assert_eq!(resolve_max_steps(0), usize::MAX);
        // A positive budget (e.g. the subagent's) is taken as-is.
        assert_eq!(resolve_max_steps(20), 20);
        // Absurd values are sanity-capped.
        assert_eq!(resolve_max_steps(1_000_000), MAX_STEPS_CEILING);
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
        /// How many times the engine asked for permission.
        asks: usize,
        /// The `tool` argument of each `ask_permission` call, in order.
        ask_tools: Vec<String>,
        /// Each `turn_tokens` report, in order.
        turn_token_reports: Vec<u64>,
    }
    impl AgentUi for CapturingUi {
        fn assistant_text(&mut self, t: &str) {
            self.text.push_str(t);
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

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-engine-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Raw-HTTP server that answers each connection with the next SSE body in
    /// `bodies` (one connection per `complete()` call), then closes.
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
        // The static `-y` flag.
        assert!(ctx(true, None).auto_approve_enabled());
        // Neither source on → prompt.
        assert!(!ctx(false, None).auto_approve_enabled());
        // The live flag flips the SAME ctx without rebuilding it: a mid-turn
        // Shift+Tab is seen by the running turn (the snapshot bug is gone).
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
        // Reasoning-capable model: the level is always requested (independent of
        // whether the chat shows thinking); `/effort` changes it.
        let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
        assert_eq!(engine.thinking_request(), (Some("medium"), false));
        engine.set_reasoning_effort("high".into());
        assert_eq!(engine.thinking_request(), (Some("high"), false));

        // Non-reasoning model: never requested (would 400 strict providers).
        let plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
        assert_eq!(plain.thinking_request(), (None, false));
    }

    #[test]
    fn thinking_request_disables_per_provider_disable_form() {
        // gpt-5 / o-series reject `"none"` alongside tools and reject the
        // `thinking` field → family effort floor instead.
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

        // Effort scale with no off (e.g. `aivo/starter` → deepseek: low..max, and
        // absent from the snapshot): emit the `thinking` disable field, NOT an
        // invalid `"none"` effort (which 400s the provider).
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

        // Snapshot-known Anthropic model (no none/minimal in its scale): the
        // `thinking` field too — the OpenAI→Anthropic bridge carries it through.
        let mut claude = AgentEngine::new("/tmp", "claude-sonnet-4-5", "", &[], &[], 0, 0);
        claude.set_thinking_enabled(false);
        assert_eq!(claude.thinking_request(), (None, true));

        // Genuinely non-reasoning model with no catalog level: stay silent.
        let mut plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
        plain.set_thinking_enabled(false);
        assert_eq!(plain.thinking_request(), (None, false));
    }

    /// Full loop: first model turn emits a write_file tool call (executed
    /// locally), the second turn answers with text → converges. Mirrors canary's
    /// fake_openai idea but pure-Rust and provider-free.
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

    /// A `run_bash` call the sandbox blocks (a write outside the workspace)
    /// prompts to re-run *outside* the sandbox — scoped to the synthetic
    /// `run_bash_unsandboxed` tool. Declining keeps the blocked result and never
    /// runs the command unconfined. macOS-only + skipped when the sandbox is off.
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
        // The escalation prompt fired, scoped to the synthetic unsandboxed tool.
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

    /// Approving the escalation re-runs the command outside the sandbox, so the
    /// out-of-workspace write that was blocked now succeeds.
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

    /// A batch of independent read-only calls runs concurrently (the fan-out
    /// path), but its results are still recorded in call order and each paired to
    /// its own `tool_call_id` — the invariant the providers require.
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

        // All three tool starts fire, in call order.
        assert_eq!(ui.tools, vec!["read_file", "read_file", "read_file"]);
        // Tool results land in call order, each keyed to the right id and content.
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

    /// A mixed batch — a parallel-safe read and an ordered write — still records
    /// every result in call order with the right pairing, and the write lands.
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

    /// An empty completion (no content, no tool calls) converges the turn but is
    /// NOT recorded as an assistant message — an empty assistant becomes an empty
    /// (invalid) Anthropic content array via the serve bridge.
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
        // The turn still ran (the user message is recorded).
        assert!(
            engine
                .messages
                .iter()
                .any(|m| role(m) == "user" && content_str(m) == "hi")
        );
    }

    /// A denied DANGEROUS tool (destructive bash) doesn't run; the engine feeds
    /// the refusal back and the second turn converges.
    #[tokio::test]
    async fn denied_dangerous_tool_does_not_run() {
        let dir = tmp();
        let sentinel = dir.join("RAN");
        // `rm -rf` makes this dangerous → gated. If it ran it would touch RAN.
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

    #[test]
    fn permission_key_scopes_to_command_and_path() {
        // Different bash commands get different scopes — approving one risky
        // command must not whitelist another.
        assert_ne!(
            permission_key("run_bash", &json!({"command":"rm -rf build"})),
            permission_key("run_bash", &json!({"command":"rm -rf /"})),
        );
        // Whitespace-only differences collapse to the same scope.
        assert_eq!(
            permission_key("run_bash", &json!({"command":"  cargo test "})),
            permission_key("run_bash", &json!({"command":"cargo test"})),
        );
        // File writes scope per path.
        assert_ne!(
            permission_key("write_file", &json!({"path":"/etc/a"})),
            permission_key("write_file", &json!({"path":"/etc/b"})),
        );
        // Anything else (e.g. an MCP tool) keys on the tool name.
        assert_eq!(
            permission_key("mcp__srv__tool", &json!({"x":1})),
            "mcp__srv__tool"
        );
    }

    /// "Always allow" remembers the exact command, not the whole tool: approving
    /// one destructive `rm` doesn't silently auto-run a *different* destructive
    /// command — that one prompts again.
    ///
    /// Unix-only: the test executes `rm -rf … && touch …`, which PowerShell 5.1
    /// (the Windows shell) can't run (no `touch`, no `&&`). The permission-scoping
    /// logic under test is platform-agnostic.
    #[cfg(unix)]
    #[tokio::test]
    async fn always_allow_is_scoped_to_the_exact_command() {
        let dir = tmp();
        let (sa, sb) = (dir.join("RAN_A"), dir.join("RAN_B"));
        let cmd_a = format!("rm -rf zzz_a && touch {}", sa.display());
        let cmd_b = format!("rm -rf zzz_b && touch {}", sb.display());
        // Steps within one turn: A, then A again, then a different B, then text.
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

        // A prompted once (then its repeat reused the remembered scope); B, a
        // different command, prompted separately → two asks total.
        assert_eq!(ui.asks, 2, "expected A once + B once");
        assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
        assert!(sa.exists(), "command A did not run");
        assert!(sb.exists(), "command B did not run");
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

    /// A `write_file` that would overwrite a pre-existing, unread file is gated;
    /// denying it leaves the original contents intact.
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

    /// A SAFE mutating tool (in-project write) runs WITHOUT a permission prompt,
    /// even when the UI would deny — i.e. only dangerous actions are gated.
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

        // Two plan events: the model's update (2 steps), then the engine's
        // finalization when the turn converged.
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

    /// A started plan whose steps the model never finished is finalized by the
    /// engine when the turn converges — the screenshot bug: the model set step 1
    /// `in_progress`, did the work, then answered without ever flipping the steps,
    /// leaving the card stuck at "0/N done".
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

        // The model's update, then the engine's finalization.
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

    /// An all-pending plan means the model planned but converged WITHOUT executing
    /// (e.g. it asked the user a question). The engine must not fabricate
    /// completion — the `started` gate leaves it alone.
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

    /// A `subagent` call spawns a fresh sub-engine on the same fake serve: the
    /// sub-engine answers with text, its result is fed back to the parent as the
    /// tool result, and the parent converges. Three connections: parent's call,
    /// the sub-agent's turn, the parent's final answer.
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

        // The parent saw `subagent` as a tool step and converged on the final text.
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

    /// An external tool source's schemas are offered to the model, and a call to
    /// one is routed to the source (not the built-in executor). Uses a mock so the
    /// engine wiring is tested without a real MCP subprocess.
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

    /// #3 end-to-end: the model calls `take_note`; the engine stores it (no
    /// permission prompt, no `tools::execute`), echoes a confirmation, and the
    /// note is retained for pinning into later compactions.
    #[tokio::test]
    async fn take_note_is_dispatched_and_stored() {
        let dir = tmp();
        // Turn 1: take a note; turn 2: converge.
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

    /// An external source that `requires_approval` gets permission-gated: with a
    /// denying UI and no auto-approve, the call is refused (not executed).
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

        // The untrusted tool was offered + asked-about, but the deny blocked it.
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

    /// The sub-agent UI recovers an answer emitted in the same step as the final
    /// tool call (then a silent convergence), instead of losing it.
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

    /// A sub-agent's token usage is folded into the parent turn's total, so the
    /// footer reflects the real cost (the sub's LLM calls aren't parent steps).
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

        // The parent's own steps report no usage here, so the 100 came from the
        // sub-agent — proving it was folded into the turn total.
        assert!(
            ui.footer_tokens >= 100,
            "sub-agent tokens not folded into the parent total: {}",
            ui.footer_tokens
        );
    }

    /// The engine sums each step's provider-measured token split (prompt /
    /// completion / cache) across a turn and surfaces it via `take_turn_usage`,
    /// so the chat TUI can fold real tokens into the session index for stats.
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

    /// When a sub-agent produces no answer, the failure reason (a step-limit /
    /// LLM-error notice) is surfaced instead of a vague "no answer".
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

    #[test]
    fn is_retryable_error_classifies() {
        assert!(is_retryable_error(
            "upstream 503 Service Unavailable: overloaded"
        ));
        assert!(is_retryable_error("request failed: connection refused"));
        assert!(is_retryable_error("upstream 429: rate limit exceeded"));
        // Not retryable: auth, bad request, context overflow.
        assert!(!is_retryable_error("upstream 401: invalid api key"));
        assert!(!is_retryable_error(
            "upstream 400: maximum context length exceeded"
        ));
        // Auth/bad-request stay terminal even when the message mentions a
        // retryable word — don't burn retries (and delay the error).
        assert!(!is_retryable_error(
            "401 unauthorized: connection token expired"
        ));
        assert!(!is_retryable_error("403 forbidden: network policy blocked"));
        assert!(!is_retryable_error("bad request: malformed timeout field"));
    }

    // Unix-only: the mock is a hand-rolled raw `TcpListener` serving sequential
    // blocking `accept()`s (503 then 200), whose connection sequencing is fragile
    // on Windows. The retry-past-503 logic it exercises is platform-agnostic and
    // stays covered on Linux/macOS.
    #[cfg(unix)]
    #[tokio::test]
    async fn engine_retries_then_succeeds() {
        // First connection returns 503 (retryable, before any stream); the retry
        // hits a 200 with content.
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
    fn usage_tokens_handles_both_shapes() {
        assert_eq!(usage_tokens(&Some(json!({"total_tokens": 42}))), 42);
        assert_eq!(
            usage_tokens(&Some(json!({"input_tokens": 10, "output_tokens": 5}))),
            15
        );
        assert_eq!(usage_tokens(&None), 0);
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

    /// `set_context_window` fills a window that was unknown (0) at construction —
    /// the late-catalog-warm case — but never overrides a known one.
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

    /// `push_text_turn` merges into a preceding same-role plain-text message so
    /// the engine never holds two consecutive same-role turns (Anthropic 400s on
    /// those via the bridge). Different roles and tool_call-bearing assistants
    /// are never merged.
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

    /// Seeding a history that already has two adjacent user turns (a cancelled
    /// turn + the next) must not reproduce them as consecutive user messages.
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

    /// A trimmed history can start mid-exchange with assistant turns; seeding
    /// must drop those leading non-user turns so the conversation opens with a
    /// user message (Anthropic rejects an assistant-first sequence).
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
    fn discover_project_guides_lists_only_present_guide_files() {
        let dir = tmp();
        std::fs::write(dir.join("AGENTS.md"), "rules").unwrap();
        std::fs::write(dir.join("README.md"), "not a guide").unwrap();
        assert_eq!(discover_project_guides(&dir), vec!["AGENTS.md".to_string()]);
    }

    #[test]
    fn system_prompt_points_to_guides_lazily() {
        // With guides present: name is referenced, content is NOT inlined, and the
        // agent is told to skip them for trivial messages.
        let p = system_prompt("/tmp/proj", "2026-01-01", &["AGENTS.md".to_string()], &[]);
        assert!(p.contains("AGENTS.md"));
        assert!(p.contains("Skip them for questions"));
        assert!(p.contains("just to say hello"));
        // No guides → no convention-file section at all. (Match the section's
        // unique opener, not the bare phrase "convention file" — the base prompt
        // now points at convention files as a place to find build/test commands.)
        let none = system_prompt("/tmp/proj", "", &[], &[]);
        assert!(!none.contains("This project has convention file"));
    }

    #[test]
    fn system_prompt_names_the_host_shell() {
        // The model is told which shell `run_bash` uses so it writes commands in
        // the right syntax (not bash on a Windows host). The label must match what
        // `sandbox::bare_shell` actually spawns on this platform.
        let p = system_prompt("/tmp/proj", "", &[], &[]);
        assert!(p.contains("Environment:"));
        assert!(p.contains(crate::agent::sandbox::shell_label()));
    }

    #[test]
    fn system_prompt_includes_restraint_guardrails() {
        // The action-biased prompt carries its counterweights: verify-before-done,
        // don't-claim-unverified, don't-commit-unprompted, and confirm-before an
        // irreversible / outward-facing action (the gap the destructive-command
        // heuristic can't see).
        let p = system_prompt("/tmp/proj", "", &[], &[]);
        assert!(p.contains("verify it before you call the task done"));
        assert!(p.contains(
            "Never report a fix as working or a task as done unless you've observed it pass"
        ));
        assert!(p.contains("Don't commit, push, create"));
        assert!(p.contains("does NOT catch outward-facing or hard-to-undo"));
        assert!(p.contains("wait for the user to confirm"));
        // Added after a cross-model review converged on the same gaps:
        assert!(p.contains("never invent file contents")); // don't fabricate
        assert!(p.contains("never print, log, hard-code, or commit secrets")); // secrets hygiene
        assert!(p.contains("change tactics rather than repeating it")); // loop-breaking
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

    /// Compaction must not leave two consecutive `user` messages: the summary is
    /// folded INTO the first kept user turn (not inserted before it), so the
    /// roles keep alternating — Anthropic (via the serve bridge) 400s otherwise,
    /// which would brick the agent right after a compaction.
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

    /// The pinned working set (plan + touched files) survives a compaction
    /// verbatim — folded into the SAME kept user turn, so alternation still holds
    /// even with a non-empty pinned block (the critical regression).
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

    /// Compaction must preserve the tool_use↔tool_result pairing in the KEPT
    /// region: every `tool` message that survives the drain still follows an
    /// assistant `tool_calls` that names its `tool_call_id`, and no orphan `tool`
    /// message is left at the head of the kept history (a leading tool result with
    /// no preceding call also 400s strict providers). `find_cut` lands on a user
    /// boundary, so a whole assistant→tool pair is never split by the cut.
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

    /// The budget backstop drops whole oldest turns at user boundaries (never the
    /// system prompt) until the history fits — the guard that keeps a
    /// post-compaction overflow from bricking a turn with a non-retryable 413.
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

    /// When even [system, last user turn] overflows — a single pasted turn larger
    /// than the window — the backstop shortens the oversized content instead of
    /// looping forever, so the turn still fits.
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
        // First pass: no prior summary → fresh prompt, transcript verbatim.
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
        // Far more files than fit under PINNED_MAX_TOKENS (set the field directly
        // to bypass MAX_TOUCHED_FILES, simulating a giant historical list).
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

    /// #1: the cheap compaction pass clears bulky OLD tool outputs (before the
    /// keep window) to a stub, leaving recent ones — and their `tool_call_id`s —
    /// intact, and is idempotent.
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

    /// #3: `take_note` content rides into a compaction via the pinned block, and
    /// the per-block cap trims files before notes (notes kept, plan whole).
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

    /// #4: restore re-derives the working set (plan, notes, touched files) from the
    /// message log — the stateless-reducer property, so a resumed session is not
    /// amnesiac about its state.
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

    #[test]
    fn serialize_transcript_renders_roles() {
        let messages = vec![
            json!({"role":"user","content":"do X"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}
            ]}),
            json!({"role":"tool","content":"file contents"}),
        ];
        let t = serialize_transcript(&messages);
        assert!(t.contains("[User]: do X"));
        assert!(t.contains("[Tool calls]: read_file("));
        assert!(t.contains("[Tool result]: file contents"));
    }

    #[test]
    fn truncate_str_marks_overflow() {
        assert_eq!(truncate_str("abc", 5), "abc");
        let out = truncate_str("abcdefgh", 3);
        assert!(out.starts_with("abc…") && out.contains("+5 chars"));
    }

    /// An assistant turn left with dangling tool_calls (interrupted mid-tool) is
    /// repaired into a valid sequence before the next user turn: every unanswered
    /// call id gets a synthetic tool result, inserted right after the call.
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
        // A short assistant turn caps the synthesized tool results so the next
        // user turn alternates (tool results map to a user message upstream; a
        // bare user turn after them would be a 2nd consecutive user → Anthropic
        // 400). The tail is now that assistant turn, not the last tool result.
        let last = engine.messages.last().unwrap();
        assert_eq!(role(last), "assistant");
        assert_eq!(last["content"], "[interrupted]");

        // Idempotent: a fully-answered + capped tail is left untouched.
        let len = engine.messages.len();
        engine.repair_interrupted_tail();
        assert_eq!(engine.messages.len(), len);

        // With a real next turn appended, the synthetic assistant sits between
        // the tool results and the user — so roles alternate, no consecutive user.
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

    /// The repaired tail's core invariant: no assistant turn bearing `tool_calls`
    /// is left without a matching `tool` result for EVERY one of its call ids in
    /// the contiguous tool run that follows. Asserted generically over a transcript
    /// with several unanswered ids (the dangling-tool_use → Anthropic-400 brick).
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

    /// A clean transcript — every tool_use already answered AND already capped by a
    /// following assistant — must be left byte-for-byte unchanged (no spurious
    /// `[interrupted]` results, no duplicate synthetic assistant cap on rerun).
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
        // An assistant already follows the result → the tool run is already capped,
        // so the alternation-guard branch (line ~648) must NOT add a second cap.
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

    #[test]
    fn batch_sig_ignores_id_but_not_args() {
        let call = |id: &str, path: &str| ToolCall {
            id: id.into(),
            name: "read_file".into(),
            arguments: json!({ "path": path }),
        };
        assert_eq!(batch_sig(&[call("1", "a")]), batch_sig(&[call("2", "a")]));
        assert_ne!(batch_sig(&[call("1", "a")]), batch_sig(&[call("1", "b")]));
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

    /// A profile's body folds into the system prompt, and a `tools` allow-list
    /// restricts the offered built-ins (keeping `update_plan`), dropping the rest.
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

    /// A profile with no `tools` scope leaves the toolset untouched.
    #[test]
    fn apply_profile_without_scope_keeps_all_tools() {
        let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
        e.drop_subagent_tool();
        let before = tool_names(&e);
        e.apply_profile(&subagent("helper", None, None));
        assert_eq!(tool_names(&e), before);
    }

    /// Durable resume: `export_conversation` drops the system prompt but keeps the
    /// exact tool-call / tool-result pairing (with ids), and `restore_conversation`
    /// rebuilds it verbatim after a fresh system prompt — the round trip a resume
    /// performs. Restore is a no-op once the engine is non-fresh.
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
        // A resend after an interrupt merges into the trailing `user` message
        // ("first\n\nsecond"); the stored prompt must stay "first" so the turn
        // keeps file revert instead of dropping to conversation-only.
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
