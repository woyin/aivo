//! Construction and configuration: builder-style setters wiring budgets,
//! hooks, jobs, external tools, effort/thinking, and session context.

use super::*;

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
        let context_window = crate::services::system_env::env_parse("AIVO_AGENT_CONTEXT_WINDOW")
            .unwrap_or(context_window);
        let max_steps = resolve_max_steps(max_steps);
        let mut specs = tools::tool_specs_for(model);
        if !skills.is_empty() {
            specs.push(skills::skill_tool_spec(skills));
        }
        specs.push(plan::plan_tool_spec());
        specs.push(crate::agent::finish::finish_tool_spec());
        specs.push(notes::note_tool_spec());
        specs.push(crate::agent::memory::memory_tool_spec());
        specs.push(crate::agent::memory::memory_search_tool_spec());
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
            denied_sigs: Default::default(),
            turn_dismissals: 0,
            plan_card_dismissed: false,
            skills: skills.to_vec(),
            subagents: Vec::new(),
            agents_dir: None,
            date: date.to_string(),
            guides: guides.to_vec(),
            external: None,
            deferred_tools: Vec::new(),
            mcp_defer_tokens: tool_search::defer_threshold(),
            last_summary: None,
            plan: Vec::new(),
            touched_files: Vec::new(),
            notes: Vec::new(),
            evidence: Vec::new(),
            turn_usage: SessionTokens::default(),
            checkpoints: Vec::new(),
            turn_unsend: None,
            checkpoint_store: None,
            artifacts_dir: None,
            artifact_seq: AtomicUsize::new(1),
            reasoning_effort: default_reasoning_effort(model),
            reasoning_efforts: Vec::new(),
            effort_floor: None,
            thinking_enabled: true,
            use_web_search_enabled: true,
            image_source: None,
            preview_supported: false,
            agent_tools_enabled: true,
            reasoning_capable: default_reasoning_effort(model).is_some(),
            read_only: false,
            plan_mode_stash: Vec::new(),
            require_completion: false,
            self_correct: false,
            verify_state: verify::VerifyState::Dirty,
            finish_report: None,
            finish_rejections: 0,
            confirm_before_build: false,
            first_party: false,
            session_controls: false,
            prefix_fp: None,
            file_tracker: crate::agent::file_tracker::FileTracker::default(),
            lsp: None,
            jobs: None,
            session_mail: None,
            reply_obligation: None,
            turn_last_text: None,
            hooks: None,
            max_cost_usd: 0.0,
            cost_pricing: None,
            injected_context_tokens: 0,
            turn_cost_usd: 0.0,
            billed_model: None,
            prefix_cache_seen: false,
            snipped_originals: std::collections::HashMap::new(),
            image_descriptions: std::collections::HashMap::new(),
            substitute_images: false,
            model_reads_images: false,
        }
    }

    /// Cap the turn's estimated spend (USD; 0 = no cap). Headless `--max-cost`.
    pub fn set_cost_budget(&mut self, usd: f64, pricing: crate::services::model_metadata::Pricing) {
        self.max_cost_usd = usd;
        self.cost_pricing = Some(pricing);
    }

    /// Wire the user's lifecycle hooks (see [`crate::agent::hooks`]).
    pub fn set_hooks(&mut self, hooks: std::sync::Arc<crate::agent::hooks::HookSet>) {
        if !hooks.is_empty() {
            self.hooks = Some(hooks);
        }
    }

    /// Cap per-turn completion tokens (0 = no cap).
    pub fn set_output_budget(&mut self, tokens: u64) {
        self.max_output_tokens = tokens;
    }

    /// Back the "always allow" grants with a persistent store at `<config>/state/grants.json`,
    /// loading any grants already saved there. Without this the store is session-only.
    pub fn set_grants_path(&mut self, config_dir: &Path) {
        self.grants = crate::agent::grant_store::GrantStore::load(
            crate::services::paths::grants_json(config_dir),
        );
    }

    /// Persist sub-agent reports under `dir` so a long delegation survives compaction.
    /// Resumes numbering past existing reports so a rebuilt engine can't overwrite one.
    pub fn set_artifacts_dir(&mut self, dir: PathBuf) {
        let next = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| parse_artifact_seq(&e.file_name().to_string_lossy()))
            .max()
            .map_or(1, |n| n + 1);
        self.artifact_seq = AtomicUsize::new(next);
        self.artifacts_dir = Some(dir);
    }

    /// Wire the background-job table and advertise `check_job`. Idempotent.
    pub fn set_jobs(&mut self, jobs: crate::agent::jobs::SharedJobs) {
        let first = self.jobs.is_none();
        self.jobs = Some(jobs);
        if first {
            self.tools_openai
                .push(tool_to_openai(crate::agent::jobs::check_job_tool_spec()));
        }
    }

    /// Wire the open-session mailbox and advertise `list_sessions` /
    /// `send_session` (top-level engine only). Idempotent.
    pub fn set_session_mail(&mut self, mail: crate::services::session_mail::SessionMail) {
        let first = self.session_mail.is_none();
        self.session_mail = Some(mail);
        if first {
            self.tools_openai
                .push(tool_to_openai(list_sessions_tool_spec()));
            self.tools_openai
                .push(tool_to_openai(send_session_tool_spec()));
        }
    }

    /// Arm (or clear) the turn's reply obligation; set every turn so a stale
    /// one can't leak forward.
    pub fn set_reply_obligation(&mut self, obligation: Option<ReplyObligation>) {
        self.reply_obligation = obligation;
    }

    /// Enable LSP diagnostics-after-edit rooted at `cwd` (default on; `AIVO_AGENT_LSP=0`
    /// opts out) — after a successful edit, the language server's native errors are fed back.
    pub fn maybe_enable_lsp(&mut self, cwd: &Path) {
        if crate::services::system_env::env_flag("AIVO_AGENT_LSP").unwrap_or(true) {
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
        // Without this the model thinks the terminal is text-only: it reads an
        // image to narrate it, then runs `open` on an unasked-for desktop viewer.
        let images_clause = if ctx.inline_images {
            " This terminal DOES render images: name an image file's path in your \
reply and it appears inline in the transcript. So for \"show me x.png\" just say the \
path — don't read the file to describe it, and never run `open`/`xdg-open` to launch \
an external viewer unless the user explicitly asks you to open it."
        } else {
            ""
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
questions. A decision that is genuinely the user's — a requirement the request left open, a \
trade-off between reasonable designs — deserves an `ask_user` BEFORE you build on one answer; a \
choice with an obvious conventional default doesn't (take the default and say so). For bigger \
work the user can also run `/plan`, which puts the session in read-only planning mode with a \
plan-approval card — suggest it when a task deserves real design discussion.{images_clause}",
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

    /// Reversible read-only mode. On: stash file-mutating tools + `subagent`,
    /// advertise `exit_plan_mode`, append the directive. Off: restore the stashed
    /// specs verbatim (no rebuild — history survives into same-turn execution).
    /// `run_bash` stays offered (confirmation-gated). Idempotent both ways.
    pub fn set_plan_mode(&mut self, on: bool) {
        if on == self.read_only {
            return;
        }
        self.read_only = on;
        if on {
            let (stashed, kept) = std::mem::take(&mut self.tools_openai)
                .into_iter()
                .partition(|t| {
                    let name = t["function"]["name"].as_str().unwrap_or("");
                    tools::hidden_in_plan_mode(name)
                });
            self.plan_mode_stash = stashed;
            self.tools_openai = kept;
            self.tools_openai
                .push(tool_to_openai(plan_mode::exit_plan_mode_tool_spec()));
        } else {
            self.tools_openai
                .retain(|t| t["function"]["name"].as_str() != Some("exit_plan_mode"));
            self.tools_openai
                .extend(std::mem::take(&mut self.plan_mode_stash));
            // A plan-mode deny often means "not yet" — exiting reverses it.
            self.denied_sigs.clear();
        }
        let directive = format!("\n\n{}", plan_mode::PLAN_MODE_DIRECTIVE);
        if let Some(content) = self.messages.first_mut().and_then(|m| m.get_mut("content"))
            && let Some(s) = content.as_str()
        {
            *content = Value::String(if on {
                format!("{s}{directive}")
            } else {
                s.replacen(&directive, "", 1)
            });
        }
    }

    /// Enable the headless completion gate (unattended `-e`): a text-only turn that
    /// admits it isn't done, or trails off mid-step, is nudged to continue (bounded)
    /// instead of being accepted as the final answer. Off for interactive/sub-agents.
    pub fn set_require_completion(&mut self) {
        self.require_completion = true;
    }

    /// Enable/disable post-edit self-verification: on a declared-done turn, run the
    /// project's validator and feed failures back so the model fixes them. See [`verify`].
    /// Takes a bool so goal mode can toggle it per turn on a reused engine.
    pub fn set_self_correct(&mut self, on: bool) {
        self.self_correct = on;
    }

    /// Treat the current tree as verified — only a mutation re-arms self-verify, so
    /// the default-on headless path doesn't pay a suite run for investigate-only work.
    pub fn set_verified_baseline(&mut self) {
        self.verify_state = verify::VerifyState::Clean;
    }

    /// Set the `reasoning_effort` level (`/effort`). Only meaningful for reasoning models.
    pub fn set_reasoning_effort(&mut self, effort: String) {
        self.reasoning_effort = Some(effort);
    }

    /// Turn thinking on/off for upcoming turns (`/config`). Off makes [`Self::thinking_request`] emit a disable signal.
    pub fn set_thinking_enabled(&mut self, on: bool) {
        self.thinking_enabled = on;
    }

    pub fn set_image_substitution(&mut self, on: bool) {
        self.substitute_images = on;
    }

    pub fn set_model_reads_images(&mut self, on: bool) {
        self.model_reads_images = on;
    }

    pub fn insert_image_description(&mut self, hash: String, text: String) {
        self.image_descriptions.insert(hash, text);
    }

    /// The app's cache is the copy that survives engine rebuilds.
    pub fn merge_image_descriptions(&mut self, map: &std::collections::HashMap<String, String>) {
        self.image_descriptions
            .extend(map.iter().map(|(k, v)| (k.clone(), v.clone())));
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

    /// `/config` Image generation: `Some(source)` advertises `generate_image`,
    /// `None` removes it.
    pub fn set_image_source(
        &mut self,
        source: Option<crate::services::image_generate::GeneratorSource>,
    ) {
        let is_gen = |t: &Value| t["function"]["name"].as_str() == Some("generate_image");
        let has = self.tools_openai.iter().any(is_gen) || self.plan_mode_stash.iter().any(is_gen);
        if source.is_some() && !has {
            self.advertise_tool(tool_to_openai(tools::generate_image_tool_spec()));
        } else if source.is_none() && has {
            self.tools_openai.retain(|t| !is_gen(t));
            self.plan_mode_stash.retain(|t| !is_gen(t));
        }
        self.image_source = source;
    }

    /// Idempotent, like `set_image_source`.
    pub fn set_preview_supported(&mut self, on: bool) {
        let is_preview = |t: &Value| t["function"]["name"].as_str() == Some("preview");
        let has =
            self.tools_openai.iter().any(is_preview) || self.plan_mode_stash.iter().any(is_preview);
        if on && !has {
            self.advertise_tool(tool_to_openai(tools::preview_tool_spec()));
        } else if !on && has {
            self.tools_openai.retain(|t| !is_preview(t));
            self.plan_mode_stash.retain(|t| !is_preview(t));
        }
        self.preview_supported = on;
    }

    /// Add a tool spec — into the plan-mode stash when plan mode hides it. The TUI
    /// re-runs these setters every turn AFTER `set_plan_mode`, so a live push
    /// would leak the tool into plan mode and duplicate it on exit.
    pub(super) fn advertise_tool(&mut self, spec: Value) {
        let hidden = spec["function"]["name"]
            .as_str()
            .is_some_and(tools::hidden_in_plan_mode);
        if self.read_only && hidden {
            self.plan_mode_stash.push(spec);
        } else {
            self.tools_openai.push(spec);
        }
    }

    /// Set the catalog-advertised effort levels for this turn. See `reasoning_efforts`.
    pub fn set_reasoning_efforts(&mut self, efforts: Vec<String>) {
        self.reasoning_efforts = efforts;
    }

    /// Whether `level` is one the model's catalog advertises (so it won't 400).
    pub(super) fn effort_is_valid(&self, level: &str) -> bool {
        self.reasoning_efforts.iter().any(|e| e == level)
    }

    /// Lowest advertised non-off level — what off-grade requests are floored to.
    fn lowest_real_effort(&self) -> Option<&str> {
        self.reasoning_efforts
            .iter()
            .map(String::as_str)
            .find(|l| !crate::services::model_metadata::effort_level_is_off(l))
    }

    /// Raise the session's effort floor to the lowest advertised thinking level
    /// and return it — the reaction to an upstream `tools_require_reasoning_effort`
    /// 400. `None` when the catalog has nothing above off (heal impossible;
    /// guessing a level 400s other providers).
    pub(crate) fn raise_effort_floor(&mut self) -> Option<String> {
        let level = self.lowest_real_effort()?.to_string();
        self.effort_floor = Some(level.clone());
        Some(level)
    }

    /// Thinking control for this step: `(reasoning_effort, emit_thinking_disabled)`.
    /// Enabled → resolved level. Disabled → `thinking_off_wire`'s translation.
    /// Off-grade outcomes are floored (reactively via `effort_floor`,
    /// proactively for gpt-5.4+ with tools) so the request never 400s.
    pub(super) fn thinking_request(&self) -> (Option<&str>, bool) {
        let (effort, disable) = self.thinking_request_unfloored();
        let offish =
            disable || effort.is_some_and(crate::services::model_metadata::effort_level_is_off);
        if !offish {
            return (effort, disable);
        }
        if let Some(floor) = self.effort_floor.as_deref() {
            return (Some(floor), false);
        }
        if self.agent_tools_enabled
            && crate::services::model_metadata::gpt_rejects_none_with_tools(&self.model)
        {
            // "low" when the catalog is silent — the same family guess the
            // o-series off path makes.
            return (Some(self.lowest_real_effort().unwrap_or("low")), false);
        }
        (effort, disable)
    }

    /// Write the thinking controls onto a request's `extra` map. The serve
    /// translates `reasoning_effort` per upstream; `thinking:{type:"disabled"}`
    /// is the off-switch where the scale has no "off".
    pub(super) fn write_thinking_extra(&self, extra: &mut Map<String, Value>) {
        let (effort, disable) = self.thinking_request();
        if let Some(effort) = effort {
            extra.insert("reasoning_effort".into(), json!(effort));
        } else {
            extra.remove("reasoning_effort");
        }
        if disable {
            extra.insert("thinking".into(), json!({ "type": "disabled" }));
        } else {
            extra.remove("thinking");
        }
    }

    fn thinking_request_unfloored(&self) -> (Option<&str>, bool) {
        if self.thinking_enabled {
            // A level carried across a model switch may not exist here (→ 400);
            // omit rather than guess.
            let effort = self
                .reasoning_effort
                .as_deref()
                .filter(|e| self.reasoning_efforts.is_empty() || self.effort_is_valid(e));
            return (effort, false);
        }
        let capable = self.reasoning_capable
            || self.reasoning_effort.is_some()
            || !self.reasoning_efforts.is_empty();
        if !capable {
            return (None, false);
        }
        crate::services::model_metadata::thinking_off_wire(&self.model, &self.reasoning_efforts)
    }

    /// Enable `/rewind` tree-checkpointing (top-level chat only, to avoid the git cost). Idempotent.
    pub fn enable_rewind_checkpoints(&mut self, cwd: &str) {
        if self.checkpoint_store.is_none() {
            self.checkpoint_store = Some(crate::agent::checkpoint::CheckpointStore::new(
                std::path::Path::new(cwd),
            ));
        }
    }

    /// Permanent reason `/rewind` file revert is off (home / fs-root launch dir), if any.
    pub fn rewind_file_revert_block(&self) -> Option<&'static str> {
        self.checkpoint_store
            .as_ref()
            .and_then(|s| s.permanent_disabled_reason())
    }

    /// Hand the `/rewind` state to a successor engine on a model/key switch.
    /// Only valid when the successor restores this exact conversation —
    /// `restore_conversation` preserves message indices, so every `msg_index` stays right.
    pub(crate) fn take_rewind_state(
        &mut self,
    ) -> (
        Option<crate::agent::checkpoint::CheckpointStore>,
        Vec<Checkpoint>,
    ) {
        (
            self.checkpoint_store.take(),
            std::mem::take(&mut self.checkpoints),
        )
    }

    /// Install state from [`Self::take_rewind_state`] (same index contract);
    /// keeps the fresh store when the donor had none.
    pub(crate) fn adopt_rewind_state(
        &mut self,
        store: Option<crate::agent::checkpoint::CheckpointStore>,
        checkpoints: Vec<Checkpoint>,
    ) {
        if store.is_some() {
            self.checkpoint_store = store;
        }
        self.checkpoints = checkpoints;
    }

    /// Drain the last turn's provider-measured token split (zeroing the accumulator);
    /// the chat TUI folds it into the chat session index for `aivo stats`.
    pub fn take_turn_usage(&mut self) -> SessionTokens {
        std::mem::take(&mut self.turn_usage)
    }

    /// Upstream model echoed by this session's responses, when any step carried one.
    pub fn billed_model(&self) -> Option<&str> {
        self.billed_model.as_deref()
    }

    /// Drain the turn's provider-reported USD spend, when any step carried one.
    pub fn take_turn_cost_usd(&mut self) -> Option<f64> {
        let cost = std::mem::take(&mut self.turn_cost_usd);
        (cost > 0.0).then_some(cost)
    }

    /// Attach an external tool source (MCP). Call once, after construction. Past
    /// the deferral threshold the schemas defer behind `search_tools` instead of
    /// permanently occupying the window; calls still route by name.
    pub fn set_external_tools(&mut self, ext: std::sync::Arc<dyn ExternalTools>) {
        let specs = ext.specs();
        self.deferred_tools.clear();
        if self
            .mcp_defer_tokens
            .is_some_and(|t| tool_search::should_defer_at(&specs, t))
        {
            self.deferred_tools = specs;
        } else {
            // Via `advertise_tool`: plan mode must stash these, not leave them live.
            for spec in specs {
                self.advertise_tool(spec);
            }
        }
        self.refresh_search_tools();
        self.external = Some(ext);
    }

    /// Rebuild the `search_tools` advertisement: present iff anything is deferred.
    pub(super) fn refresh_search_tools(&mut self) {
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("search_tools"));
        if !self.deferred_tools.is_empty() {
            self.tools_openai
                .push(tool_to_openai(tool_search::search_tools_spec(
                    &self.deferred_tools,
                )));
        }
    }

    /// Move the deferred specs at `idxs` into the live tool list and refresh
    /// `search_tools`; returns the loaded specs in deferred order.
    pub(super) fn load_deferred_tools(&mut self, idxs: &[usize]) -> Vec<Value> {
        let want: std::collections::HashSet<usize> = idxs.iter().copied().collect();
        if want.is_empty() {
            return Vec::new();
        }
        let mut loaded = Vec::with_capacity(want.len());
        let mut kept = Vec::with_capacity(self.deferred_tools.len());
        for (i, t) in std::mem::take(&mut self.deferred_tools)
            .into_iter()
            .enumerate()
        {
            if want.contains(&i) {
                loaded.push(t);
            } else {
                kept.push(t);
            }
        }
        self.deferred_tools = kept;
        if !loaded.is_empty() {
            // `advertise_tool` so a load during plan mode stashes instead of going live.
            for spec in loaded.iter().cloned() {
                self.advertise_tool(spec);
            }
            self.refresh_search_tools();
        }
        loaded
    }

    /// A direct call to a still-deferred tool works (routing is name-based) —
    /// promote its schema so later steps see it as a first-class tool.
    pub(super) fn promote_deferred_tool(&mut self, name: &str) {
        if let Some(i) = self
            .deferred_tools
            .iter()
            .position(|t| t["function"]["name"].as_str() == Some(name))
        {
            self.load_deferred_tools(&[i]);
        }
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

    /// Append a `--context` block to the system prompt. Re-applied per build
    /// since `export_conversation` omits the system message.
    pub fn append_system_context(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.injected_context_tokens += estimate_str_tokens(text);
        if let Some(sys) = self.messages.first_mut() {
            let cur = sys
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();
            sys["content"] = json!(format!("{cur}\n\n{text}"));
        }
    }
}

/// Engine-handled — routed to `AgentUi::switch_chat_model`, not `tools::execute`.
pub(super) fn switch_model_tool_spec() -> ToolSpec {
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
pub(super) fn set_effort_tool_spec() -> ToolSpec {
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
