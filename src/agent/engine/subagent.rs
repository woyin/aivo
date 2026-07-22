//! Sub-agents: profile wiring, the subagent tool, spawning, and reports.

use super::*;

impl AgentEngine {
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

    /// Enable delegation-time profile re-discovery (see the `agents_dir` field).
    pub fn set_agents_dir(&mut self, config_dir: &Path) {
        self.agents_dir = Some(config_dir.to_path_buf());
    }

    /// The profile a delegation should run: fresh from disk when an agents dir is
    /// configured (the snapshot goes stale the moment the model authors or edits
    /// a profile mid-turn), else from the build-time snapshot.
    pub(super) fn resolve_profile(&self, cwd: &Path, name: &str) -> Option<Subagent> {
        match &self.agents_dir {
            Some(cfg) => subagents::discover_subagents(cwd, cfg)
                .into_iter()
                .find(|s| s.name == name),
            None => self.subagents.iter().find(|s| s.name == name).cloned(),
        }
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
            // `resolved_tools` normalizes to built-ins, so the retain just dropped
            // `search_tools`; clear the now-unreachable deferred set too.
            self.deferred_tools.clear();
        }
    }

    /// Remove the `subagent` tool — used on a sub-engine so it can't spawn sub-agents (depth-1 only).
    pub(super) fn drop_subagent_tool(&mut self) {
        self.tools_openai
            .retain(|t| t["function"]["name"].as_str() != Some("subagent"));
    }

    /// Execute a `subagent` tool call: build a fresh sub-engine (same tools minus
    /// `subagent`, same cwd + serve, optionally a stronger model), run to convergence,
    /// return its answer. Capturing UI (only the result surfaces). Dangerous ops inherit
    /// the parent's auto-approve, else fail closed (no nested prompt).
    /// Run one sub-agent to completion and hand back `(result, tokens)`. `parent_ui`
    /// `Some` streams its activity to the parent (the lone-sub-agent path); `None`
    /// buffers silently, so several can run concurrently without sharing the UI.
    pub(super) async fn run_subagent(
        &self,
        ctx: &TurnCtx<'_>,
        parent_ui: Option<&mut dyn AgentUi>,
        sink: Option<(std::sync::Arc<dyn SubagentSink>, usize)>,
        base: u64,
        args: &Value,
    ) -> Result<(String, u64), String> {
        // Fallback keys are Claude Code's names, so a Task-vocabulary call still delegates.
        let str_arg = |keys: &[&str]| {
            keys.iter().find_map(|k| {
                args.get(k)
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
            })
        };
        let task =
            str_arg(&["task", "prompt"]).ok_or_else(|| "subagent: missing `task`".to_string())?;
        // Named specialist if `agent` matches — resolved fresh from disk (see
        // `resolve_profile`), so a profile authored this turn delegates correctly.
        // Unknown names fall back to generic (lenient, don't fail the turn) but
        // the result says so; a silent fallback would fake the specialist's test.
        let requested_agent = str_arg(&["agent", "subagent_type"]);
        let profile = requested_agent.and_then(|n| self.resolve_profile(ctx.cwd, n));
        // Model precedence: explicit `model` arg > profile's pinned model > parent's model.
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .filter(|m| !m.is_empty())
            .or_else(|| profile.as_ref().and_then(|p| p.model.as_deref()))
            .unwrap_or(&self.model);

        // Isolation: explicit arg wins, else the profile's flag; unavailable falls
        // back to the shared workspace with a note.
        let isolate = args.get("isolation").and_then(|v| v.as_str()) == Some("worktree")
            || profile.as_ref().is_some_and(|p| p.isolation_worktree);
        let mut guard: Option<subagents::WorktreeGuard> = None;
        let mut sub_cwd: PathBuf = ctx.cwd.to_path_buf();
        let mut isolation_note: Option<String> = None;
        if isolate {
            match subagents::create_worktree(ctx.cwd) {
                Ok(wt) => {
                    // Keep the delegate at the parent's subdir vantage point; guard
                    // prunes the worktree if this future is dropped before finalize.
                    sub_cwd = subagents::worktree_cwd(ctx.cwd, &wt);
                    guard = Some(subagents::WorktreeGuard::new(ctx.cwd, &wt));
                }
                Err(why) => {
                    isolation_note = Some(format!(
                        "\n\n[worktree isolation] unavailable ({why}); ran in the shared workspace."
                    ));
                }
            }
        }

        // Delegates keep the parent's skills except the create-agent builtin: a
        // sub-engine can't delegate (tool dropped below), so it could author a
        // profile but never test it — the workflow is top-level only. A profile
        // whose tool scope excludes `skill` gets NO skills: `apply_profile` would
        // strip the tool, and an advert the tool can't load is a prompt/toolset
        // contradiction.
        let scope_has_skill = profile
            .as_ref()
            .and_then(|p| p.resolved_tools())
            .is_none_or(|allowed| allowed.contains(&"skill"));
        let sub_skills: Vec<Skill> = if scope_has_skill {
            self.skills
                .iter()
                .filter(|s| {
                    !(s.name == skills::CREATE_AGENT_SKILL_NAME && s.dir.as_os_str().is_empty())
                })
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let mut sub = AgentEngine::new(
            &sub_cwd.display().to_string(),
            model,
            &self.date,
            &self.guides,
            &sub_skills,
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
        // Pre/PostToolUse guards cover delegated work; Stop hooks don't (they gate
        // the user-facing answer, not each delegate).
        if let Some(hooks) = &self.hooks {
            sub.set_hooks(std::sync::Arc::new(hooks.without_stop()));
        }
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
        // Share the job table so a sub-agent-started server stays pollable by the parent.
        if let Some(jobs) = &self.jobs {
            sub.set_jobs(jobs.clone());
        }
        // Fold in the specialist's role + scope. After MCP wiring so a `tools` allow-list applies to the full offered set.
        if let Some(p) = &profile {
            sub.apply_profile(p);
        }

        let agent_name = subagent_display_name(args);
        let mut ui = SubagentUi {
            parent: parent_ui,
            sink,
            base,
            agent_name,
            ..Default::default()
        };
        // The sub's ctx roots tool execution + sandbox confinement at its own cwd.
        let sub_ctx = TurnCtx {
            client: ctx.client,
            serve_base: ctx.serve_base,
            auth: ctx.auth,
            cwd: &sub_cwd,
            yes: ctx.yes,
            auto_approve_all: ctx.auto_approve_all,
            auto_approve: ctx.auto_approve,
            review_edits: ctx.review_edits,
            // Sub-agents never run in plan mode (plan strips the subagent tool).
            plan_exit: None,
        };
        // Box the recursive future (run_turn → subagent → run_turn) so it isn't infinitely-sized.
        Box::pin(sub.run_turn(&sub_ctx, &mut ui, task.to_string())).await;
        if let Some((s, slot)) = &ui.sink {
            s.done(*slot, !ui.answer().is_empty(), ui.steps, ui.tokens);
        }
        let mut msg = ui.result_message();
        if let Some(g) = guard {
            msg.push_str(&g.finalize());
        } else if let Some(note) = isolation_note {
            msg.push_str(&note);
        }
        if profile.is_none()
            && let Some(name) = requested_agent
        {
            msg.push_str(&format!(
                "\n\n[subagent] no profile named `{name}` — ran a generic sub-agent \
instead. No such file in the agents dirs (project `.aivo/agents`/`.claude/agents`, \
user config, packs); check the filename / `name:` frontmatter."
            ));
        }
        // A failed run on a model aivo's catalog doesn't know is most often a bad
        // profile `model:` — name the likely cause instead of a bare empty result.
        if ui.answer().is_empty()
            && model != self.model
            && crate::services::model_metadata::snapshot_limits(model).is_none()
        {
            msg.push_str(&format!(
                "\n\n[subagent] note: model `{model}` isn't in aivo's catalog — if the \
run failed at the provider, fix the profile's `model:` (use a full model id) or omit \
it to inherit yours."
            ));
        }
        // Gate on the STORED length (not the bare answer) so the tail can't push it over
        // the clear threshold and get cleared without a pointer.
        if let Some(dir) = &self.artifacts_dir
            && msg.len() > crate::agent::compaction::TOOL_RESULT_CLEAR_MIN
            && let Some(path) = self
                .save_subagent_report(dir, task, &ui.agent_name, model, ui.steps, ui.answer())
                .await
        {
            msg.push_str(&format!(
                "\n\n{ARTIFACT_POINTER_PREFIX}{} — re-read it with read_file if this result is cleared]",
                path.display()
            ));
        }
        Ok((msg, ui.tokens))
    }

    /// Write a sub-agent report (`sub-<seq>-<slug>.md`); `None` on IO failure (never fails the turn).
    pub(super) async fn save_subagent_report(
        &self,
        dir: &Path,
        task: &str,
        agent: &str,
        model: &str,
        steps: usize,
        answer: &str,
    ) -> Option<PathBuf> {
        tokio::fs::create_dir_all(dir).await.ok()?;
        let seq = self.artifact_seq.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("sub-{seq:03}-{}.md", slug_for_artifact(task)));
        let agent = if agent.trim().is_empty() {
            "generic"
        } else {
            agent.trim()
        };
        let task_line: String = task
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(200)
            .collect();
        let content = format!(
            "# Sub-agent report\n- task: {task_line}\n- agent: {agent} · model: {model} · steps: {steps} · date: {}\n---\n{answer}\n",
            self.date
        );
        tokio::fs::write(&path, content).await.ok()?;
        Some(path)
    }
}

/// The numeric sequence from a `sub-<NNN>-<slug>.md` artifact filename, if it is one.
pub(super) fn parse_artifact_seq(name: &str) -> Option<usize> {
    name.strip_prefix("sub-")?.split('-').next()?.parse().ok()
}

/// Filesystem-safe slug from a task's first 40 chars; empty → `report`.
fn slug_for_artifact(task: &str) -> String {
    let mut slug = String::new();
    let mut prev_dash = false;
    for ch in task.chars().take(40) {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            slug.push(c);
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "report".to_string()
    } else {
        slug
    }
}

/// Delegate display name from call args; empty → the UI numbers it.
pub(super) fn subagent_display_name(args: &Value) -> String {
    for key in ["label", "description", "agent", "subagent_type"] {
        if let Some(v) = args.get(key).and_then(|v| v.as_str()) {
            let v = v.trim();
            if !v.is_empty() {
                return v.to_string();
            }
        }
    }
    String::new()
}

/// The `subagent` tool — engine-handled (needs the serve + a fresh engine), top-level
/// engine only. When named specialists exist, an `agent` field enumerates them.
pub(super) fn subagent_tool_spec(subagents: &[Subagent]) -> ToolSpec {
    let mut properties = json!({
        "task": {"type": "string", "description": "A complete, standalone instruction for the sub-agent."},
        "label": {"type": "string", "description": "Short name for this delegate (3–6 words), shown in the live progress UI — e.g. \"audit auth flow\"."},
        "model": {"type": "string", "description": "Optional model id to run the sub-agent on (default: the agent's configured model, else same as you)."},
        "isolation": {"type": "string", "enum": ["worktree"], "description": "Optional: run the sub-agent in a disposable git worktree — an isolated snapshot of the last commit (uncommitted changes not included). Its edits stay there and the result tells you how to review/apply them. Use when a delegate will edit files, especially several delegates in parallel."}
    });
    let mut description = "Delegate a self-contained subtask to a fresh sub-agent that has the same \
file/shell tools and runs its own loop, then hands back its result. Use it to keep your own context \
focused (offload a big investigation), or pass `model` to delegate hard work to a stronger model. The \
sub-agent does not see this conversation, so make `task` complete and standalone; it cannot spawn \
further sub-agents. Call `subagent` several times in one turn to run independent investigations in \
parallel — they execute concurrently and each result comes back separately; give parallel delegates \
that edit files `isolation: \"worktree\"` so they can't clobber each other. Always pass a short \
`label` so the user can follow each delegate's progress."
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

/// Capturing UI for a sub-agent run. `cur_text` holds the in-flight step's text,
/// rolling into `last_nonempty` at each new step. The answer is the converging step's
/// text, falling back to the last non-empty step (so an answer emitted alongside the
/// final tool call isn't lost). Permission prompts forward to the parent UI, so the
/// catastrophic-command floor holds for sub-agents too; denies if detached.
#[derive(Default)]
pub(super) struct SubagentUi<'a> {
    pub(super) cur_text: String,
    pub(super) last_nonempty: String,
    /// Last engine notice — surfaced when the sub-agent produces no answer, so the failure reason isn't swallowed.
    pub(super) last_notice: String,
    pub(super) steps: usize,
    /// The sub-agent's cumulative token usage, folded into the parent turn's total.
    pub(super) tokens: u64,
    /// Forward live token growth (base + sub so-far) to the parent UI.
    pub(super) parent: Option<&'a mut dyn AgentUi>,
    /// Slot-tagged live feed for the detached (parallel) path, where `parent` is `None`.
    pub(super) sink: Option<(std::sync::Arc<dyn SubagentSink>, usize)>,
    pub(super) base: u64,
    /// Specialist name + turn counter, forwarded to the parent's status feed.
    pub(super) agent_name: String,
    pub(super) turn_no: usize,
}

impl SubagentUi<'_> {
    /// The sub-agent's answer: the converging step's text, else the last non-empty step's.
    pub(super) fn answer(&self) -> &str {
        if self.cur_text.trim().is_empty() {
            self.last_nonempty.trim()
        } else {
            self.cur_text.trim()
        }
    }

    /// The tool result the parent receives: the answer (+ step count), else the
    /// failure notice (so an LLM error / step-limit isn't masked as "no answer").
    pub(super) fn result_message(&self) -> String {
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

    pub(super) fn forward_activity(&mut self, tool: &str, args: &Value) {
        let Self {
            parent,
            sink,
            agent_name,
            turn_no,
            ..
        } = self;
        if let Some(p) = parent.as_deref_mut() {
            p.subagent_activity(agent_name, tool, args, *turn_no);
        } else if let Some((s, slot)) = sink {
            s.activity(*slot, agent_name, tool, args, *turn_no);
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
        once_only: bool,
    ) -> BoxFuture<'a, Decision> {
        // Forward to the parent (card in the TUI, fail-closed when headless) rather than
        // auto-allowing, so the catastrophic-command floor holds for sub-agents too.
        // Detached (parallel batch): deny, but visibly — a silent deny reads as
        // the delegate just doing a bad job.
        match self.parent.as_deref_mut() {
            Some(p) => p.ask_permission(tool, preview, once_only),
            None => {
                if let Some((s, slot)) = &self.sink {
                    s.denied(*slot, tool);
                }
                Box::pin(async { Decision::Deny })
            }
        }
    }
}
