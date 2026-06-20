use super::*;

use crate::agent::engine::RewindOutcome;
use crate::agent::protocol::Decision;
use crate::agent::subagents::is_default_agent_name;
use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CursorAcpSession, CursorChunk, CursorTurnResult};
use anyhow::Context;

/// Default cap on autonomous `/goal` continuations (override: `AIVO_GOAL_MAX_ITERS`).
const GOAL_DEFAULT_MAX_ITERS: usize = 20;
/// Framing prepended to the first `/goal` turn so the agent knows the contract.
const GOAL_PREAMBLE: &str = "[Goal mode] Work autonomously toward this objective, doing as many \
steps as it takes. When the objective is FULLY achieved, reply with exactly `GOAL COMPLETE` on its \
own line. If anything remains, keep going.";
/// Self-checking continuation sent between goal turns.
const GOAL_CONTINUE: &str = "Continue toward the goal. If the objective is now fully met, reply \
with exactly `GOAL COMPLETE` and nothing else; otherwise do the next step.";

fn goal_max_iterations() -> usize {
    std::env::var("AIVO_GOAL_MAX_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(GOAL_DEFAULT_MAX_ITERS)
}

/// Whether an assistant reply carries the goal-completion marker on its own line.
/// Deliberately strict (a whole-line match) so prose *mentioning* the marker —
/// e.g. "I'll say GOAL COMPLETE when done" — doesn't end the loop prematurely.
fn signals_goal_complete(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim().eq_ignore_ascii_case("GOAL COMPLETE"))
}

/// Build the message a `/<skill> [args]` invocation sends to the model. If the
/// body uses the `$ARGUMENTS` placeholder, the args are substituted in place;
/// otherwise the body is sent under a short directive with the args (if any)
/// appended as input. The full instructions are inlined so the skill runs
/// deterministically regardless of whether the model would call the `skill` tool.
pub(super) fn expand_skill_invocation(
    skill: &crate::agent::skills::Skill,
    args: Option<&str>,
) -> String {
    let args = args.unwrap_or("").trim();
    if skill.body.contains("$ARGUMENTS") {
        return skill.body.replace("$ARGUMENTS", args);
    }
    let mut out = format!(
        "Use the \"{}\" skill. Follow these instructions:\n\n{}",
        skill.name, skill.body
    );
    if !args.is_empty() {
        out.push_str(&format!("\n\nInput: {args}"));
    }
    out
}

/// Inverse of [`expand_skill_invocation`], for DISPLAY and LOGGING only: if
/// `content` is an expanded skill invocation, recover the compact `/name args`
/// the user typed — so the transcript and `aivo logs` show `/baidu-search 歌曲`
/// instead of the whole inlined `SKILL.md` body (the model still receives the
/// full body via `content`). Returns `None` for ordinary messages and for
/// `$ARGUMENTS`-style skills, which substitute in place and leave no recoverable
/// wrapper. The first-line marker is a fixed string we emit and skill names are
/// `[A-Za-z0-9_-]` (no quotes), so the match is unambiguous.
pub(crate) fn skill_invocation_label(content: &str) -> Option<String> {
    let name = content
        .lines()
        .next()?
        .strip_prefix("Use the \"")?
        .strip_suffix("\" skill. Follow these instructions:")?;
    if !crate::agent::skills::is_valid_skill_name(name) {
        return None;
    }
    // Args, when present, were appended as a single trailing `\n\nInput: <args>`
    // line; the body may itself contain that marker, so take the LAST one and
    // require a single line (a multi-line tail means there were no args).
    let args = content
        .rsplit_once("\n\nInput: ")
        .map(|(_, rest)| rest.trim())
        .filter(|rest| !rest.is_empty() && !rest.contains('\n'));
    Some(match args {
        Some(args) => format!("/{name} {args}"),
        None => format!("/{name}"),
    })
}

impl ChatTuiApp {
    pub(super) async fn submit_draft(&mut self) -> Result<bool> {
        let action = match self.prepare_submit_action() {
            Ok(action) => action,
            Err(err) => {
                self.notice = Some((ERROR, err.to_string()));
                return Ok(false);
            }
        };
        let Some(action) = action else {
            return Ok(false);
        };

        // A running `!cmd` owns the transcript tail and the spinner; serialize
        // submissions behind it (esc stops it) rather than overlapping a second.
        if self.local_command.is_some() {
            self.notice = Some((
                MUTED,
                "A command is running — press esc to stop it".to_string(),
            ));
            return Ok(false);
        }

        match action {
            SubmitAction::Send(input) => {
                if self.sending {
                    // A turn is in flight — queue this message instead of sending
                    // now; it goes out when the current turn finishes.
                    self.queue_message(input);
                } else if let Err(err) = self.send_user_message(input).await {
                    self.notice = Some((ERROR, err.to_string()));
                }
                Ok(false)
            }
            SubmitAction::Command(command) => {
                // Record the typed `/command` so up-arrow recalls it, the same as
                // a normal message or `!cmd`. Skills and `/create-skill` record
                // their own normalized `/name args` form inside their handlers
                // (see `run_skill_command`), so skip them here to avoid a
                // duplicate entry. Capture the draft before it's cleared below.
                let recordable = !matches!(
                    command,
                    SlashCommand::Skill { .. } | SlashCommand::CreateSkill(_)
                );
                let typed = self.draft.trim().to_string();
                match self.execute_slash_command(command).await {
                    Ok(should_exit) => {
                        if recordable {
                            self.record_draft_history(&typed);
                        }
                        self.draft.clear();
                        self.cursor = 0;
                        self.command_menu.reset();
                        self.draft_history_index = None;
                        self.draft_history_stash = None;
                        Ok(should_exit)
                    }
                    Err(err) => {
                        self.notice = Some((ERROR, err.to_string()));
                        Ok(false)
                    }
                }
            }
            SubmitAction::Shell(command) => {
                // Don't run a local command on top of a model turn; interrupt it
                // (esc) first. Keeps the draft so the command can be retried.
                if self.sending {
                    self.notice = Some((
                        MUTED,
                        "Interrupt the current turn (esc) before running a command".to_string(),
                    ));
                    return Ok(false);
                }
                self.record_draft_history(&format!("!{command}"));
                self.start_local_command(command);
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
                self.draft_history_index = None;
                self.draft_history_stash = None;
                Ok(false)
            }
        }
    }

    pub(super) fn prepare_submit_action(&self) -> Result<Option<SubmitAction>> {
        let trimmed = self.draft.trim();
        if trimmed.is_empty() {
            return if self.draft_attachments.is_empty() {
                Ok(None)
            } else {
                Ok(Some(SubmitAction::Send(String::new())))
            };
        }
        if self.draft.contains('\n') {
            return Ok(Some(SubmitAction::Send(trimmed.to_string())));
        }
        if let Some(escaped) = trimmed.strip_prefix("//") {
            return Ok(Some(SubmitAction::Send(format!("/{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('/') {
            return Ok(Some(SubmitAction::Command(
                self.resolve_slash_command(command)?,
            )));
        }
        if let Some(escaped) = trimmed.strip_prefix("!!") {
            return Ok(Some(SubmitAction::Send(format!("!{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('!') {
            let command = command.trim();
            if command.is_empty() {
                anyhow::bail!("Type a command after '!'");
            }
            return Ok(Some(SubmitAction::Shell(command.to_string())));
        }
        Ok(Some(SubmitAction::Send(trimmed.to_string())))
    }

    /// Parse `input` (the text after the leading `/`) into a [`SlashCommand`]. A
    /// name that isn't a built-in but matches a discovered skill resolves to
    /// [`SlashCommand::Skill`] (the `/repo-study` case); everything else falls back
    /// to [`parse_slash_command`], so a built-in always wins over a same-named skill
    /// and a true typo still errors with "Unknown command".
    pub(super) fn resolve_slash_command(&self, input: &str) -> Result<SlashCommand> {
        let name = input.split_whitespace().next().unwrap_or("");
        let is_builtin = SLASH_COMMANDS.iter().any(|c| c.name == name);
        if !is_builtin && self.skill_commands.iter().any(|s| s.name == name) {
            let argument = input
                .split_once(char::is_whitespace)
                .map(|(_, rest)| rest.trim())
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned);
            return Ok(SlashCommand::Skill {
                name: name.to_string(),
                argument,
            });
        }
        parse_slash_command(input)
    }

    pub(super) async fn send_user_message(&mut self, input: String) -> Result<()> {
        let record = input.clone();
        self.dispatch_user_message(input, Some(record)).await
    }

    /// Send a skill invocation: the model receives `content` (the expanded skill
    /// body), but the draft history records `typed` (`/name args`) so up-arrow
    /// recalls the re-runnable command, not the expanded body.
    pub(super) async fn send_skill_message(
        &mut self,
        content: String,
        typed: String,
    ) -> Result<()> {
        self.dispatch_user_message(content, Some(typed)).await
    }

    async fn dispatch_user_message(&mut self, input: String, record: Option<String>) -> Result<()> {
        let attachments = materialize_attachments(&self.draft_attachments).await?;
        if self.key.is_cursor_acp()
            && let Some(session) = self.cursor_acp_session.as_ref()
        {
            // Existing session: capabilities are already known, fail fast
            // without paying a session-open round trip. Cold-open path runs
            // the same check post-open inside `spawn_cursor_turn`.
            cursor_acp::ensure_image_attachments_supported(
                session.prompt_capabilities(),
                &attachments,
            )?;
        }
        if let Some(record) = record {
            self.record_draft_history(&record);
        }
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.overlay = Overlay::None;
        self.notice = None;
        self.last_usage = None;
        self.live_usage = None;
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        self.pending_submit = Some(PendingSubmission {
            content: input.clone(),
            attachments: attachments.clone(),
        });
        self.request_started_at = Some(Instant::now());
        // A finished plan stays pinned as a done marker until now; a new user
        // message starts (possibly) new work, so clear a completed checklist so it
        // doesn't linger above the composer into an unrelated task.
        self.clear_completed_plan();
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: input.clone(),
            reasoning_content: None,
            attachments: attachments.clone(),
        });
        trim_history(&mut self.history, MAX_HISTORY_MESSAGES);
        // A new turn rebuilds the transcript rows and snaps to the bottom, so any
        // prior selection would point at the wrong content — drop it.
        self.clear_transcript_selection();
        self.sending = true;
        self.follow_output = true;

        if self.key.is_cursor_acp() {
            self.spawn_cursor_turn(input, attachments);
        } else if self.agent_capable() && attachments.is_empty() {
            // The API-key path is the native agent: tools + cwd + permission gate.
            // (Attachments/OAuth/copilot fall back to plain chat below.)
            self.spawn_agent_turn(input).await;
        } else {
            // Surface the silent downgrade: an attachment turns the agent into a
            // plain vision chat, so its file/shell tools are off for this message.
            if self.agent_capable() && !attachments.is_empty() {
                self.notice = Some((
                    MUTED,
                    "Attachment sent as plain chat — agent tools are off for this message"
                        .to_string(),
                ));
            }
            self.spawn_http_turn();
        }
        Ok(())
    }

    /// Stash a message typed during an in-flight turn; sent when the turn ends.
    /// Appends to a FIFO so several can queue without clobbering each other.
    fn queue_message(&mut self, input: String) {
        if input.trim().is_empty() {
            return;
        }
        self.record_draft_history(&input);
        self.queued_messages.push(input);
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.notice = Some((MUTED, self.queued_notice()));
    }

    /// Notice text for the queue, reflecting how many are waiting.
    fn queued_notice(&self) -> String {
        match self.queued_messages.len() {
            0 | 1 => "Queued — sends when the current turn finishes".to_string(),
            n => format!("Queued ({n} waiting) — sent one per turn, in order"),
        }
    }

    /// After a turn ends, send the oldest message queued mid-turn (if any). One
    /// per turn-end, so each queued message becomes its own user turn in order.
    pub(super) async fn drain_queued_message(&mut self) -> Result<()> {
        if !self.sending && !self.queued_messages.is_empty() {
            let queued = self.queued_messages.remove(0);
            self.send_user_message(queued).await?;
        }
        Ok(())
    }

    /// True when the current key can drive the in-process agent: a plain API key
    /// reachable through serve (not OAuth, cursor, or copilot).
    pub(super) fn agent_capable(&self) -> bool {
        !self.key.is_any_oauth() && !self.key.is_cursor_acp() && !self.key.is_copilot()
    }

    /// Refresh the cached git branch for `display_cwd`, throttled so the footer's
    /// `.git/HEAD` read happens at most every couple of seconds rather than on
    /// every (≈60fps) frame. A checkout — by the user in another terminal or by
    /// the agent via run_bash — is reflected on the next refresh. Cheap file read,
    /// no subprocess; `None` when the dir isn't a git work tree.
    pub(super) fn refresh_git_branch(&mut self) {
        const REFRESH_AFTER: std::time::Duration = std::time::Duration::from_secs(2);
        let due = self
            .git_branch_checked_at
            .is_none_or(|at| at.elapsed() >= REFRESH_AFTER);
        if !due {
            return;
        }
        let cwd = self.display_cwd().to_string();
        self.git_branch = git_branch_for(&cwd);
        self.git_branch_checked_at = Some(std::time::Instant::now());
    }

    /// Directory shown in the header/footer: the real launch dir for any chat
    /// that actually runs there (the in-process agent *and* the cursor ACP
    /// backend, where files are edited — a safety signal), else chat's sandbox
    /// (the plain OAuth/copilot relay).
    pub(super) fn display_cwd(&self) -> &str {
        if (self.agent_capable() || self.key.is_cursor_acp()) && !self.real_cwd.is_empty() {
            &self.real_cwd
        } else {
            &self.cwd
        }
    }

    /// Directory key for persistence / logs / resume: the real launch dir, which
    /// is stable across runs. `self.cwd` is the per-pid ephemeral sandbox, so
    /// keying on it hides every session from `/resume` and `aivo logs` on the
    /// next launch. Falls back to the sandbox only when the real dir is unknown.
    pub(super) fn persist_cwd(&self) -> &str {
        if self.real_cwd.is_empty() {
            &self.cwd
        } else {
            &self.real_cwd
        }
    }

    fn spawn_http_turn(&mut self) {
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = self.key.clone();
        let model = self.model.clone();
        // Drop agent tool / local-command steps — they're display-only and aren't
        // valid provider turns on the plain-chat path.
        let history: Vec<ChatMessage> = self
            .history
            .iter()
            .filter(|m| m.role == "user" || m.role == "assistant")
            .cloned()
            .collect();
        let copilot_tm = self.copilot_tm.clone();
        let mut format = self.format.clone();

        self.response_task = Some(tokio::spawn(async move {
            let spinning = Arc::new(AtomicBool::new(false));
            let result = send_message_turn(
                &client,
                &key,
                copilot_tm.as_deref(),
                &model,
                &history,
                &mut format,
                &spinning,
                false, // TUI always streams for live rendering
                &mut |chunk| {
                    tx.send(RuntimeEvent::Delta(chunk)).ok();
                    Ok(())
                },
            )
            .await;
            let result = result.map_err(|err| err.to_string());

            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
    }

    /// Run one agent turn: (re)build the in-process engine, start a per-turn
    /// loopback serve, then drive `engine.run_turn` on a background task that
    /// streams text/tool-steps and permission requests back as `RuntimeEvent`s.
    async fn spawn_agent_turn(&mut self, input: String) {
        use crate::agent::engine::{AgentEngine, TurnCtx};
        use crate::services::serve_router::{ServeRouter, ServeRouterConfig, random_auth_token};

        // The agent works in the real launch directory — NOT chat's sandbox
        // (`self.cwd`). It reads/edits the user's actual project.
        let real_cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };

        // Rebuild the engine when absent or when the key/model changed; otherwise
        // reuse it so multi-turn context carries over.
        let need_new = self
            .agent_engine
            .as_ref()
            .is_none_or(|s| s.key_id != self.key.id || s.model != self.model);
        if need_new {
            let date = chrono::Local::now().format("%Y-%m-%d").to_string();
            let guides =
                crate::agent::engine::discover_project_guides(std::path::Path::new(&real_cwd));
            let mut skills = crate::agent::skills::discover_skills(std::path::Path::new(&real_cwd));
            // Drop skills the user turned off in `/skills`.
            if let Ok(disabled) = self.session_store.get_disabled_skills().await {
                let disabled: std::collections::HashSet<String> = disabled.into_iter().collect();
                skills.retain(|s| !disabled.contains(&s.name));
            }
            let context_window = crate::services::model_metadata::resolve_limits(
                &self.cache,
                Some(&self.key.base_url),
                &self.model,
            )
            .await
            .context
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32;
            let mut engine = AgentEngine::new(
                &real_cwd,
                &self.model,
                &date,
                &guides,
                &skills,
                context_window,
                0,
            );
            // Enable `/rewind` tree-checkpointing (top-level chat only — sub-engines
            // never call this, so they don't pay the git cost).
            engine.enable_rewind_checkpoints(&real_cwd);
            // Offer any named specialist sub-agents authored under
            // `.aivo/agents` / `.claude/agents` (project + user). The model
            // delegates to them via the `subagent` tool's `agent` field.
            let subagents =
                crate::agent::subagents::discover_subagents(std::path::Path::new(&real_cwd));
            engine.set_subagents(&subagents);
            // If a top-level agent is active (`--agent` / `/agent`), fold its role +
            // tool scope into THIS engine. `default` (the built-in agent) is not a
            // profile — clear it silently. An unknown name warns once and clears, so
            // a stale `--agent foo` can't keep applying nothing every rebuild.
            if let Some(name) = self.active_agent.clone() {
                if is_default_agent_name(&name) {
                    self.active_agent = None;
                } else if let Some(profile) = subagents.iter().find(|s| s.name == name) {
                    engine.apply_profile(profile);
                } else {
                    let available: Vec<&str> = subagents.iter().map(|s| s.name.as_str()).collect();
                    self.notice = Some((
                        ERROR,
                        if available.is_empty() {
                            format!("no agent named `{name}` (none are defined)")
                        } else {
                            format!(
                                "no agent named `{name}` (available: {})",
                                available.join(", ")
                            )
                        },
                    ));
                    self.active_agent = None;
                }
            }
            // Carry prior conversation into the new engine. A resumed session
            // restores its DURABLE transcript verbatim (exact tool_calls + results
            // with ids); otherwise (model/key switch, old/non-agent sessions) fall
            // back to the lossy text seed of the display history (tool steps folded
            // into compact notes, since their call ids are gone). Either way the
            // just-pushed current user turn is excluded — run_turn re-adds it.
            if let Some(conversation) = self.pending_agent_messages.take() {
                engine.restore_conversation(conversation);
            } else {
                let prior = self.history.len().saturating_sub(1);
                let seed = agent_seed_turns(&self.history[..prior]);
                engine.seed_history(seed);
            }
            // Offer any configured MCP servers' tools. Connect in the BACKGROUND
            // (not inline) so a slow server — an `npx` download on first use can
            // take seconds — can't freeze the UI on the first turn. Tools arrive via
            // `McpConnected`, which rebuilds the engine (re-seeding from history) to
            // advertise them. Cached after the first connect, so this turn just uses
            // whatever's ready; the common no-mcp.json case is empty + instant.
            let connected = match &self.mcp_client {
                Some(client) if client.has_tools() => Some(client.clone()),
                _ => None,
            };
            if let Some(client) = connected {
                engine.set_external_tools(client);
            } else if self.mcp_client.is_none() {
                // Connect the configured servers the user hasn't disabled. A repo's
                // project `.mcp.json` STDIO servers run local code, so they're held
                // back behind a one-time consent card (user + HTTP servers connect
                // freely); see `connect_mcp_with_consent`.
                let disabled = self.effective_disabled_mcp_servers().await;
                self.connect_mcp_with_consent(real_cwd.clone(), disabled)
                    .await;
            }
            self.agent_engine = Some(AgentSession {
                key_id: self.key.id.clone(),
                model: self.model.clone(),
                engine: std::sync::Arc::new(tokio::sync::Mutex::new(engine)),
            });
        }
        let engine = self.agent_engine.as_ref().unwrap().engine.clone();

        // Per-turn loopback serve from the current key — the sole network egress,
        // with usage counted under the "chat" tool (`aivo chat` IS the in-process
        // agent; the standalone `agent` command was retired, so its tokens belong
        // to chat's per-tool stats, not a phantom "agent" bucket).
        let auth = random_auth_token();
        let config = ServeRouterConfig::from_key(
            &self.key,
            false,
            300,
            Some(auth.clone()),
            std::collections::HashMap::new(),
        );
        let router = ServeRouter::new(config, self.key.clone(), self.session_store.logs())
            .with_usage_accounting(self.session_store.clone(), "chat".to_string())
            // The chat TUI owns the terminal in raw mode; router progress lines on
            // stderr (protocol auto-switch, failover) would corrupt the prompt box.
            .quiet(true);
        let (handle, shutdown, port) = match router.start_background_with_addr("127.0.0.1", 0).await
        {
            Ok(t) => t,
            Err(e) => {
                self.notice = Some((ERROR, format!("agent serve failed to start: {e}")));
                self.sending = false;
                self.request_started_at = None;
                self.pending_submit = None;
                return;
            }
        };
        self.agent_serve = Some((handle, shutdown));
        let base = format!("http://127.0.0.1:{port}");

        let tx = self.tx.clone();
        let cwd = real_cwd;
        // Clone the shared LIVE flag (not a snapshot) so a mid-turn Shift+Tab
        // toggle takes effect on this running turn's permission gate.
        let auto_approve = self.auto_approve_flag.clone();
        // The context window may have resolved AFTER the engine was built (a model
        // only known via the background catalog warm). Carry the latest value in
        // so the engine can fill a still-missing window and start compacting,
        // instead of letting a long conversation overflow uncompacted.
        let context_window = self.context_window.min(u32::MAX as u64) as u32;
        self.response_task = Some(tokio::spawn(async move {
            let client = crate::services::http_utils::router_http_client();
            let ctx = TurnCtx {
                client: &client,
                serve_base: &base,
                auth: Some(&auth),
                cwd: std::path::Path::new(&cwd),
                yes: false,
                auto_approve: Some(&auto_approve),
            };
            let mut ui = ChatAgentUi { tx };
            let mut engine = engine.lock().await;
            engine.set_context_window(context_window);
            // run_turn ends by calling ui.footer → AgentFinished commits the turn.
            engine.run_turn(&ctx, &mut ui, input).await;
        }));
    }

    /// Tear down the per-turn agent serve (shutdown notify + abort accept loop).
    pub(super) fn stop_agent_serve(&mut self) {
        if let Some((handle, shutdown)) = self.agent_serve.take() {
            shutdown.notify_one();
            handle.abort();
        }
    }

    fn spawn_cursor_turn(&mut self, input: String, attachments: Vec<MessageAttachment>) {
        // Existing session: clone handles cheaply and skip the open step.
        let existing = self.cursor_acp_session.as_ref().map(|session| {
            (
                session.client_handle(),
                session.session_id().to_string(),
                session.model_id().map(str::to_string),
                session.prompt_capabilities().clone(),
            )
        });
        let key = self.key.clone();
        let requested_model = (!self.raw_model.is_empty()).then(|| self.raw_model.clone());
        // cursor-agent runs as a real coding agent in the actual launch dir (like
        // the in-process agent path), so it can read/edit the user's project.
        // Falls back to the sandbox only when the real dir is unknown (tests).
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.clone()
        } else {
            self.real_cwd.clone()
        };
        let tx = self.tx.clone();
        let format = self.format.clone();
        let cursor_auto_approve = self.auto_approve_flag.clone();
        // When auto-approve is off, surface cursor's out-of-process tool
        // requests on aivo's own permission card (allow / always / deny) instead
        // of a blanket reject. Reuses the same AgentPermission channel as the
        // in-process agent; "always" flips auto-approve on for the session.
        let permission_prompt: cursor_acp::CursorPermissionPrompt = {
            let tx = self.tx.clone();
            std::sync::Arc::new(move |req: cursor_acp::CursorPermissionRequest| {
                let tx = tx.clone();
                Box::pin(async move {
                    let (reply, rx) = tokio::sync::oneshot::channel();
                    if tx
                        .send(RuntimeEvent::AgentPermission {
                            tool: req.tool,
                            preview: Some(req.preview),
                            reply,
                        })
                        .is_err()
                    {
                        return Decision::Deny;
                    }
                    rx.await.unwrap_or(Decision::Deny)
                })
            })
        };

        // Open + prompt happen inside the spawned task so the TUI event loop
        // keeps polling input. The Node.js startup + 3 RPC roundtrips on a
        // first-message cold open used to block keyboard handling.
        self.response_task = Some(tokio::spawn(async move {
            let (client, session_id, model_id, capabilities) = match existing {
                Some(handles) => handles,
                None => {
                    // cursor-agent acts as a full agent here, running its
                    // Read/Edit/execute tools against the real workspace — but
                    // gated by the live Shift+Tab auto-approve toggle (OFF =>
                    // conversation-only) so the displayed safety state is honest.
                    // AIVO_CURSOR_ALLOW_TOOLS still hard-overrides either way.
                    match CursorAcpSession::open_with_options(
                        &key,
                        requested_model.as_deref(),
                        &cwd,
                        None,
                        cursor_acp::ModelPickPreference::PreferNoThinking,
                        Some(cursor_auto_approve),
                        Some(permission_prompt),
                    )
                    .await
                    {
                        Ok(session) => {
                            let handles = (
                                session.client_handle(),
                                session.session_id().to_string(),
                                session.model_id().map(str::to_string),
                                session.prompt_capabilities().clone(),
                            );
                            // Hand the live session to the event loop so future
                            // turns reuse it. The clone above keeps the Arc alive
                            // for this task even if the event loop drops it.
                            tx.send(RuntimeEvent::CursorSessionOpened(session)).ok();
                            handles
                        }
                        Err(e) => {
                            tx.send(RuntimeEvent::Finished {
                                result: Err(e.to_string()),
                                format,
                            })
                            .ok();
                            return;
                        }
                    }
                }
            };

            if let Err(e) =
                cursor_acp::ensure_image_attachments_supported(&capabilities, &attachments)
            {
                tx.send(RuntimeEvent::Finished {
                    result: Err(e.to_string()),
                    format,
                })
                .ok();
                return;
            }

            let result = drive_cursor_turn(client, session_id, model_id, input, attachments, &tx)
                .await
                .map_err(|err| err.to_string());
            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
    }

    pub(super) fn queue_attachment(&mut self, path: String) -> Result<()> {
        let attachment = build_pending_attachment(&path)?;
        let name = attachment.name.clone();
        let kind = attachment_kind_label(&attachment);
        self.draft_attachments.push(attachment);
        self.notice = Some((MUTED, format!("Queued {kind}: {name}")));
        Ok(())
    }

    pub(super) fn detach_attachment(&mut self, index: usize) -> Result<()> {
        if index == 0 {
            anyhow::bail!("Usage: /detach <n> where n starts at 1");
        }
        let remove_at = index - 1;
        if remove_at >= self.draft_attachments.len() {
            anyhow::bail!(
                "No queued attachment #{index}. There {} {} queued.",
                if self.draft_attachments.len() == 1 {
                    "is"
                } else {
                    "are"
                },
                self.draft_attachments.len()
            );
        }
        let attachment = self.draft_attachments.remove(remove_at);
        let kind = attachment_kind_label(&attachment);
        self.notice = Some((MUTED, format!("Removed {kind}: {}", attachment.name)));
        Ok(())
    }

    pub(super) async fn execute_slash_command(&mut self, command: SlashCommand) -> Result<bool> {
        match command {
            SlashCommand::New => {
                self.start_new_chat();
                Ok(false)
            }
            SlashCommand::Exit => Ok(true),
            SlashCommand::Resume(query) => {
                self.open_resume_picker(query).await?;
                Ok(false)
            }
            SlashCommand::Model(query) => {
                let auto_accept_exact = query.is_some();
                self.open_model_picker(query, ModelSelectionTarget::CurrentChat, auto_accept_exact);
                Ok(false)
            }
            SlashCommand::Key(query) => {
                self.open_or_switch_key(query).await?;
                Ok(false)
            }
            SlashCommand::Attach(path) => {
                self.queue_attachment(path)?;
                Ok(false)
            }
            SlashCommand::Detach(index) => {
                self.detach_attachment(index)?;
                Ok(false)
            }
            SlashCommand::Copy(n) => {
                self.copy_reply_to_clipboard(n)?;
                Ok(false)
            }
            SlashCommand::Skills(arg) => {
                self.run_skills_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Mcp(arg) => {
                self.run_mcp_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Agent(arg) => {
                self.run_agent_command(arg).await;
                Ok(false)
            }
            SlashCommand::Goal(arg) => {
                self.run_goal_command(arg).await;
                Ok(false)
            }
            SlashCommand::CreateSkill(arg) => {
                self.run_create_skill_command(arg).await?;
                Ok(false)
            }
            SlashCommand::Skill { name, argument } => {
                self.run_skill_command(name, argument).await?;
                Ok(false)
            }
            SlashCommand::Rewind => {
                self.open_rewind_picker().await?;
                Ok(false)
            }
            SlashCommand::Help => {
                self.open_help_overlay();
                Ok(false)
            }
        }
    }

    /// Re-discover the skills available for the working dir, drop any disabled in
    /// `/skills`, and cache them as slash commands so the `/` menu suggests them
    /// (`/repo-study`). Cheap dir reads; called at startup and after skill changes.
    pub(super) async fn refresh_skill_commands(&mut self) {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let mut skills = crate::agent::skills::discover_skills(std::path::Path::new(&cwd));
        if let Ok(disabled) = self.session_store.get_disabled_skills().await {
            let disabled: std::collections::HashSet<String> = disabled.into_iter().collect();
            skills.retain(|s| !disabled.contains(&s.name));
        }
        let next: Vec<SkillCommand> = skills
            .into_iter()
            .map(|s| SkillCommand {
                description: crate::agent::skills::advert_description(&s.description),
                name: s.name,
            })
            .collect();
        // Auto-reload: when the available skill set changes mid-session — the agent
        // just wrote a new `SKILL.md`, or one was created/renamed/removed/disabled —
        // drop the cached engine so the next turn re-advertises the new set to the
        // model (and the `skill` tool's enum updates). The exact conversation is
        // preserved across the rebuild. Body-only edits don't change this list; the
        // `/name` path re-reads the body fresh regardless.
        if next != self.skill_commands {
            self.reset_engine_preserving_conversation();
        }
        self.skill_commands = next;
    }

    /// Drop the cached agent engine while keeping its exact transcript, so the next
    /// turn rebuilds (re-reading skills/tools/guides) without losing tool history.
    /// If a turn is mid-flight the lock is held and we fall back to the lossy
    /// history seed on rebuild — same as the MCP-tools rebuild path. Caller must be
    /// off the sending path for the lossless case to take effect.
    pub(super) fn reset_engine_preserving_conversation(&mut self) {
        let Some(session) = self.agent_engine.take() else {
            return;
        };
        if let Ok(engine) = session.engine.try_lock() {
            let conversation = engine.export_conversation();
            if !conversation.is_empty() {
                self.pending_agent_messages = Some(conversation);
            }
        }
    }

    /// `/repo-study [args]`: invoke a discovered skill as a slash command. Loads the
    /// skill's instructions fresh (so an edited `SKILL.md` is honored), expands them
    /// with the user's args, and sends them as a turn — a deterministic manual
    /// trigger rather than hoping the model elects to call the `skill` tool.
    pub(super) async fn run_skill_command(
        &mut self,
        name: String,
        argument: Option<String>,
    ) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let Some(skill) = crate::agent::skills::discover_skills(std::path::Path::new(&cwd))
            .into_iter()
            .find(|s| s.name == name)
        else {
            anyhow::bail!("no skill named `{name}`");
        };
        let content = expand_skill_invocation(&skill, argument.as_deref());
        let typed = match argument.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
            Some(args) => format!("/{name} {args}"),
            None => format!("/{name}"),
        };
        if self.sending {
            // A turn is in flight — queue the expanded prompt; record the typed
            // form so up-arrow recalls the command.
            self.record_draft_history(&typed);
            self.queued_messages.push(content);
            self.notice = Some((MUTED, self.queued_notice()));
        } else {
            self.send_skill_message(content, typed).await?;
        }
        Ok(())
    }

    /// `/create-skill [intent]`: the built-in create-skill command. Unlike a
    /// discovered skill it ships in the binary (no folder, never in `/skills`); it
    /// just dispatches its embedded instructions as a turn so the agent walks the
    /// user through creating or improving a skill. The optional argument is the
    /// initial intent. Reuses the skill-invocation plumbing, so the transcript and
    /// logs show the compact `/create-skill …` (see `skill_invocation_label`).
    pub(super) async fn run_create_skill_command(
        &mut self,
        argument: Option<String>,
    ) -> Result<()> {
        let skill = crate::agent::skills::create_skill_builtin();
        // Build the prompt directly rather than via `expand_skill_invocation`:
        // create-skill's body documents the literal `$ARGUMENTS` token, which that
        // helper would otherwise substitute away. Keep the same wrapper so the
        // transcript recognizer renders the compact `/create-skill …`.
        let args = argument.as_deref().map(str::trim).filter(|a| !a.is_empty());
        let mut content = format!(
            "Use the \"{}\" skill. Follow these instructions:\n\n{}",
            skill.name, skill.body
        );
        if let Some(args) = args {
            content.push_str(&format!("\n\nInput: {args}"));
        }
        let typed = match args {
            Some(args) => format!("/create-skill {args}"),
            None => "/create-skill".to_string(),
        };
        if self.sending {
            self.record_draft_history(&typed);
            self.queued_messages.push(content);
            self.notice = Some((MUTED, self.queued_notice()));
        } else {
            self.send_skill_message(content, typed).await?;
        }
        Ok(())
    }

    /// `/rewind`: open the picker listing every past user turn (newest first), so
    /// the user can jump back to one. Selecting a turn (see [`rewind_to_turn`])
    /// truncates the conversation there, reverts the agent's file edits made since,
    /// and restores that turn's prompt to the composer for edit/resend. Each row is
    /// labeled with the file impact, or marked "conversation only" for turns that
    /// predate the live engine (restored on resume — no file snapshots).
    pub(super) async fn open_rewind_picker(&mut self) -> Result<()> {
        if self.sending {
            self.notice = Some((
                MUTED,
                "Can't rewind while a turn is in progress".to_string(),
            ));
            return Ok(());
        }
        let user_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "user")
            .map(|(i, _)| i)
            .collect();
        if user_indices.is_empty() {
            self.notice = Some((MUTED, "Nothing to rewind to".to_string()));
            return Ok(());
        }
        let turn_count = user_indices.len();
        // Under a single engine lock: `base` (turns restored on resume — no
        // checkpoints) and per-checkpoint revertibility (whether the turn has a
        // tree snapshot). A live turn at history index `u` maps to engine
        // checkpoint `u - base`; turns outside `base..base+len` (restored / beyond
        // the live checkpoints) or without a tree snapshot are conversation-only.
        let mut base: Option<usize> = None;
        let mut revertible: Vec<bool> = Vec::new();
        if let Some(session) = self.agent_engine.as_ref()
            && let Ok(engine) = session.engine.try_lock()
        {
            let (b, rev) = engine.rewind_checkpoints();
            base = Some(b);
            revertible = rev;
        }
        let live_end = base
            .map(|b| (b + revertible.len()).min(turn_count))
            .unwrap_or(0);
        let mut items = Vec::with_capacity(turn_count);
        for (turn_idx, &history_index) in user_indices.iter().enumerate() {
            let conversation_only = match base {
                Some(b) => {
                    turn_idx < b
                        || turn_idx >= live_end
                        || !revertible.get(turn_idx - b).copied().unwrap_or(false)
                }
                None => true,
            };
            let ordinal = if conversation_only {
                None
            } else {
                base.map(|b| turn_idx - b)
            };
            let prompt = rewind_excerpt(&self.history[history_index].content);
            let suffix = rewind_label_suffix(conversation_only);
            items.push(PickerEntry {
                label: format!("{}. {prompt}{suffix}", turn_idx + 1),
                search_text: self.history[history_index].content.clone(),
                value: PickerValue::RewindTurn {
                    history_index,
                    ordinal,
                    conversation_only,
                },
            });
        }
        // Newest turn at the top — the common rewind target.
        items.reverse();
        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Rewind to turn",
            String::new(),
            items,
            PickerKind::Rewind,
        )));
        Ok(())
    }

    /// Apply a `/rewind` to the turn the user picked: truncate history at that turn,
    /// restore its prompt + attachments to the composer, and — for a live turn —
    /// revert the agent's file edits via the engine while keeping the engine so a
    /// further rewind still works. Conversation-only turns fall back to the lossy
    /// engine reset (file edits are not reverted) and say so.
    pub(super) async fn rewind_to_turn(
        &mut self,
        history_index: usize,
        ordinal: Option<usize>,
        conversation_only: bool,
    ) -> Result<()> {
        if self.sending {
            self.notice = Some((
                MUTED,
                "Can't rewind while a turn is in progress".to_string(),
            ));
            return Ok(());
        }
        if history_index >= self.history.len() {
            // Stale selection (history changed under the overlay) — ignore.
            return Ok(());
        }
        let removed = self.history[history_index].clone();
        self.history.truncate(history_index);
        self.draft = removed.content;
        self.cursor = self.draft.len();
        self.draft_attachments = removed.attachments;
        self.clear_transcript_selection();
        self.cursor_acp_session = None;
        self.follow_output = true;

        let notice = match ordinal {
            Some(ord) if !conversation_only => {
                // Restore the working tree via the engine's shadow git store. The
                // engine is kept (not nulled) so its surviving checkpoints let a
                // later rewind to an even earlier point still revert. `self.sending`
                // is guarded above, so the lock is uncontended.
                let outcome = if let Some(session) = self.agent_engine.as_ref() {
                    let mut engine = session.engine.lock().await;
                    Some(engine.rewind_to(ord).await)
                } else {
                    None
                };
                match outcome {
                    Some(outcome) => rewind_notice(&outcome),
                    None => {
                        self.agent_engine = None;
                        "Rewound (conversation only — file edits not reverted)".to_string()
                    }
                }
            }
            _ => {
                // No tree snapshot for this turn: rebuild from the trimmed history.
                self.agent_engine = None;
                "Rewound (conversation only — file edits not reverted)".to_string()
            }
        };
        self.notice = Some((MUTED, notice));
        self.persist_history().await?;
        Ok(())
    }

    /// `!cmd`: run a shell command locally in the agent's working dir, streaming
    /// its output live into the transcript. Purely local — the command and its
    /// output are never sent to the model (a display-only escape hatch; the agent
    /// path already has a model-driven bash tool for that). Returns immediately
    /// after spawning the reader task; output arrives via `LocalCommandLine`
    /// events and the run is committed to history on `LocalCommandDone`.
    pub(super) fn start_local_command(&mut self, command: String) {
        if self.local_command.is_some() {
            self.notice = Some((MUTED, "A command is already running".to_string()));
            return;
        }
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        // Spawn the command (PTY on Unix for live line-buffered streaming, plain pipes
        // on Windows where ConPTY never EOFs); hold its killer so esc can stop it. The
        // blocking read runs on a worker.
        let shell = match spawn_local_shell(&command, std::path::Path::new(&cwd)) {
            Ok(shell) => shell,
            Err(err) => {
                self.notice = Some((ERROR, format!("Failed to run command: {err}")));
                return;
            }
        };
        let killer = shell.killer_handle();
        let tx = self.tx.clone();
        let task = tokio::spawn(async move {
            let _ = tokio::task::spawn_blocking(move || run_local_to_completion(shell, tx)).await;
        });
        self.local_command = Some(LocalCommandRun {
            task,
            killer,
            started_at: Instant::now(),
            command,
            stdout: String::new(),
            stderr: String::new(),
        });
        self.follow_output = true;
        self.notice = None;
    }

    /// Esc while a `!cmd` is running: kill the child (via the reader task's
    /// `kill_on_drop`) and commit whatever it produced so far as an interrupted
    /// `local_command` entry.
    pub(super) async fn interrupt_local_command(&mut self) -> Result<()> {
        let Some(mut run) = self.local_command.take() else {
            return Ok(());
        };
        // Kill the PTY child (the blocking read can't be cancelled by aborting the
        // task alone); the reader then hits EOF and the worker winds down.
        let _ = run.killer.kill();
        run.task.abort();
        let total = self.record_local_output(run.command, run.stdout, run.stderr, -1, false, true);
        self.notice = Some((
            MUTED,
            if total > MAX_OUTPUT_LINES {
                format!("Command interrupted — ctrl+o to view all {total} lines")
            } else {
                "Command interrupted".to_string()
            },
        ));
        self.persist_history().await?;
        Ok(())
    }

    /// `/copy [n]`: copy the Nth-latest assistant reply (default most recent) to
    /// the system clipboard.
    fn copy_reply_to_clipboard(&mut self, n: Option<usize>) -> Result<()> {
        let nth = n.unwrap_or(1).max(1);
        let reply = self
            .history
            .iter()
            .rev()
            .filter(|m| m.role == "assistant" && !m.content.trim().is_empty())
            .nth(nth - 1)
            .map(|m| m.content.clone());
        let Some(reply) = reply else {
            anyhow::bail!("No assistant reply to copy yet");
        };
        write_system_clipboard(&reply)?;
        let label = if nth == 1 {
            "Copied the latest reply".to_string()
        } else {
            format!("Copied reply #{nth}")
        };
        self.notice = Some((MUTED, label));
        Ok(())
    }

    pub(super) fn push_newline(&mut self) {
        if !self.draft.is_empty() {
            self.leave_history_navigation();
            self.insert_char_at_cursor('\n');
        }
    }

    pub(super) fn reset_composer(&mut self) {
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    /// `/agent`: switch the top-level agent profile (from `.aivo/agents` /
    /// `.claude/agents`). Bare `/agent` lists what's available and what's active;
    /// `/agent none` (or `off`) clears it; `/agent <name>` selects it — but ONLY at
    /// the start of a conversation. Switching the role/tools mid-thread would desync
    /// the system prompt the model has already been steered by, so any change is
    /// gated to an empty, idle history (run `/new` first). The cached engine is
    /// dropped so the next turn rebuilds with the new profile folded in. If the
    /// chosen profile pins a `model:`, it's adopted (a later `/model` overrides
    /// it) — mirroring how `--agent` supplies a default model at launch.
    pub(super) async fn run_agent_command(&mut self, arg: Option<String>) {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let available = crate::agent::subagents::discover_subagents(std::path::Path::new(&cwd));

        // Bare `/agent`: read-only listing — allowed any time.
        let Some(name) = arg else {
            if available.is_empty() {
                self.notice = Some((
                    MUTED,
                    "No agents defined. Add one at .aivo/agents/<name>.md (or .claude/agents/<name>.md)".to_string(),
                ));
                return;
            }
            let names: Vec<&str> = available.iter().map(|s| s.name.as_str()).collect();
            let current = self.active_agent.as_deref().unwrap_or("default");
            self.notice = Some((
                MUTED,
                format!(
                    "agent: {current} · available: {} · /agent <name> to switch (at chat start), /agent default to reset",
                    names.join(", ")
                ),
            ));
            return;
        };

        // Any change to the active agent is gated to the start of a conversation.
        if self.sending {
            self.notice = Some((
                ERROR,
                "Can't switch agent while a turn is in progress".to_string(),
            ));
            return;
        }
        if !self.history.is_empty() {
            self.notice = Some((
                ERROR,
                "Switch agents at the start of a chat — run /new first".to_string(),
            ));
            return;
        }

        if is_default_agent_name(&name) {
            if self.active_agent.is_none() {
                self.notice = Some((MUTED, "Already using the default agent".to_string()));
            } else {
                self.active_agent = None;
                self.agent_engine = None; // rebuild as the built-in default agent
                self.notice = Some((MUTED, "Switched to the default agent".to_string()));
            }
            return;
        }

        match available.iter().find(|s| s.name == name) {
            Some(profile) => {
                let agent_name = profile.name.clone();
                let desc = crate::agent::skills::advert_description(&profile.description);
                let model = profile.model.clone();
                self.active_agent = Some(agent_name.clone());
                self.agent_engine = None; // rebuild with the new profile folded in
                // Adopt the profile's pinned model, if any (composes like `--agent`
                // at launch; a later `/model` overrides it). `apply_model` clears
                // the notice, so set ours after.
                if let Some(model) = model
                    && let Err(e) = self.apply_model(model).await
                {
                    self.notice = Some((
                        ERROR,
                        format!("agent `{agent_name}` set, model switch failed: {e}"),
                    ));
                    return;
                }
                self.notice = Some((MUTED, format!("agent: {agent_name} — {desc}")));
            }
            None => {
                let names: Vec<&str> = available.iter().map(|s| s.name.as_str()).collect();
                self.notice = Some((
                    ERROR,
                    format!("no agent named `{name}` (available: {})", names.join(", ")),
                ));
            }
        }
    }

    /// `/goal`: autonomous goal mode. `<objective>` starts it (submits the first
    /// turn with goal framing); bare shows status/usage; `stop`/`off`/`cancel`
    /// ends it. Only for the native agent path. The loop itself is driven by
    /// `maybe_continue_goal`, called after each agent turn finishes.
    pub(super) async fn run_goal_command(&mut self, arg: Option<String>) {
        match arg.as_deref().map(str::trim) {
            None | Some("") => {
                let msg = match &self.goal_mode {
                    Some(g) => {
                        let mut obj: String = g.objective.chars().take(48).collect();
                        if g.objective.chars().count() > 48 {
                            obj.push('…');
                        }
                        format!(
                            "Goal: \"{}\" (step {}/{}) — /goal stop to end",
                            obj, g.iteration, g.max
                        )
                    }
                    None => {
                        "Usage: /goal <objective> — work autonomously until done; /goal stop to end"
                            .to_string()
                    }
                };
                self.notice = Some((MUTED, msg));
            }
            Some("stop") | Some("off") | Some("cancel") => {
                let msg = if self.goal_mode.take().is_some() {
                    "Goal mode stopped"
                } else {
                    "Goal mode wasn't active"
                };
                self.notice = Some((MUTED, msg.to_string()));
            }
            Some(objective) => {
                if self.sending {
                    self.notice = Some((
                        ERROR,
                        "Wait for the current turn to finish before starting a goal".to_string(),
                    ));
                    return;
                }
                if !self.agent_capable() {
                    self.notice = Some((
                        ERROR,
                        "Goal mode needs the native agent (a plain API key — not OAuth, cursor, or copilot)"
                            .to_string(),
                    ));
                    return;
                }
                self.goal_mode = Some(GoalState {
                    objective: objective.to_string(),
                    iteration: 0,
                    max: goal_max_iterations(),
                });
                let first = format!("{GOAL_PREAMBLE}\n\nObjective: {objective}");
                if let Err(e) = self.send_user_message(first).await {
                    self.goal_mode = None;
                    self.notice = Some((ERROR, e.to_string()));
                    return;
                }
                // `send_user_message` clears the notice; hint about unattended runs after.
                if !self.agent_auto_approve {
                    self.notice = Some((
                        MUTED,
                        "Goal mode on — press Shift+Tab to auto-approve tools so it runs unattended"
                            .to_string(),
                    ));
                }
            }
        }
    }

    /// After an agent turn finishes, drive the active `/goal` loop: stop on the
    /// completion marker, otherwise auto-continue with a self-checking prompt until
    /// the iteration cap. No-op when not in goal mode or a turn is already in flight
    /// (e.g. a queued user message took over — the goal resumes after it finishes).
    pub(super) async fn maybe_continue_goal(&mut self) -> Result<()> {
        if self.goal_mode.is_none() || self.sending {
            return Ok(());
        }
        let last_reply = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "assistant")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        if signals_goal_complete(&last_reply) {
            let steps = self.goal_mode.take().map(|g| g.iteration).unwrap_or(0);
            self.notice = Some((
                MUTED,
                format!("Goal complete (after {steps} continuation(s))"),
            ));
            return Ok(());
        }
        let Some(goal) = self.goal_mode.as_mut() else {
            return Ok(());
        };
        goal.iteration += 1;
        if goal.iteration > goal.max {
            let max = goal.max;
            self.goal_mode = None;
            self.notice = Some((
                MUTED,
                format!(
                    "Goal mode stopped at the {max}-step cap (/goal <objective> to keep going)"
                ),
            ));
            return Ok(());
        }
        self.send_user_message(GOAL_CONTINUE.to_string()).await
    }

    pub(super) fn start_new_chat(&mut self) {
        self.discard_resume_state();
        self.cancel_inflight_request();
        self.overlay = Overlay::None;
        self.history.clear();
        self.last_local_output = None;
        self.clear_transcript_selection();
        self.reset_composer();
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.sending = false;
        self.request_started_at = None;
        self.session_id = new_chat_session_id();
        self.format = seeded_chat_format(&self.key, &self.raw_model);
        self.last_usage = None;
        self.context_tokens = 0;
        // Fresh session → fresh token tally (the index entry starts at zero).
        self.session_tokens = crate::services::session_store::SessionTokens::default();
        self.context_is_estimate = true;
        self.follow_output = true;
        self.notice = None;
        // Drop the cursor-agent session so the next turn opens a fresh ACP
        // session — cursor's server-side chat context shouldn't bleed across
        // /new.
        self.cursor_acp_session = None;
        // Drop the agent engine + serve so a fresh chat starts with no context.
        self.agent_engine = None;
        // A fresh chat must not inherit a resumed session's pending transcript.
        self.pending_agent_messages = None;
        self.agent_permission = None;
        self.stop_agent_serve();
    }

    pub(super) fn cancel_inflight_request(&mut self) {
        let was_sending = self.sending;
        // Cancelling (interrupt path 1, /new, resume, key switch) also exits any
        // autonomous /goal loop, so it can't auto-continue after the dropped turn.
        // The interrupt-with-partial path clears it separately, before this runs.
        self.goal_mode = None;
        // An in-process agent turn is in flight when its per-turn serve is up. The
        // engine has ALREADY consumed this turn (and may have run side-effecting
        // tools — file writes, shell commands), and it keeps its own conversation
        // record. So un-sending the turn from the transcript would hide that work
        // and diverge the display from the engine; keep the user turn instead.
        let was_agent_turn = self.agent_serve.is_some();
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        // Tear down the agent turn's serve and drop any pending permission card
        // (the dropped reply makes the engine's awaiting tool fail closed).
        self.stop_agent_serve();
        self.agent_permission = None;
        self.queued_messages.clear();
        if was_sending && let Some(session) = self.cursor_acp_session.as_ref() {
            // Fire-and-forget session/cancel so the agent stops generating
            // even though our task already dropped the prompt stream.
            let client = session.client_handle();
            let sid = session.session_id().to_string();
            tokio::spawn(async move {
                let _ = client
                    .notify("session/cancel", serde_json::json!({"sessionId": sid}))
                    .await;
            });
        }
        if was_agent_turn {
            // Keep the user turn in the transcript; just drop the restore buffer so
            // it can't be resurrected by a later non-agent cancel.
            self.pending_submit = None;
        } else {
            restore_cancelled_submission(
                &mut self.history,
                &mut self.draft,
                &mut self.draft_attachments,
                &mut self.pending_submit,
            );
        }
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        self.follow_output = true;
        self.notice = Some((MUTED, "Request cancelled".to_string()));
    }

    pub(super) async fn interrupt_inflight_request(&mut self) -> Result<()> {
        // Interrupting ends any autonomous /goal loop (both interrupt paths route
        // through here; the partial-text path below doesn't call
        // `cancel_inflight_request`, so clear it up front for both).
        let goal_was_active = self.goal_mode.take().is_some();
        // Reveal any buffered text so the full received reply is kept, and drop
        // a deferred finish — we're committing the partial turn ourselves.
        self.drain_incoming_buffer();
        self.pending_finish = None;
        if self.pending_response.is_empty() {
            self.cancel_inflight_request();
            if goal_was_active {
                self.notice = Some((MUTED, "Goal mode stopped".to_string()));
            }
            return Ok(());
        }

        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        // Tear down an agent turn's serve / permission card if this was one.
        self.stop_agent_serve();
        self.agent_permission = None;
        self.queued_messages.clear();

        let partial = std::mem::take(&mut self.pending_response);
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.follow_output = true;
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: partial,
            reasoning_content: None,
            attachments: vec![],
        });
        self.context_tokens = estimate_context_tokens(&self.history);
        self.context_is_estimate = true;
        self.last_usage = None;
        self.persist_history().await?;
        self.notice = Some((
            MUTED,
            if goal_was_active {
                "Response interrupted — goal mode stopped"
            } else {
                "Response interrupted"
            }
            .to_string(),
        ));
        Ok(())
    }

    pub(super) fn record_draft_history(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }
        // Drop consecutive duplicates (shell `ignoredups` behavior): re-running
        // the same command or recalling an entry with up-arrow and resending it
        // shouldn't stack identical rows. Non-adjacent repeats are kept.
        if self.draft_history.last().map(String::as_str) != Some(input) {
            self.draft_history.push(input.to_string());
        }
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    pub(super) fn history_prev(&mut self) {
        if self.draft_history.is_empty() {
            return;
        }

        let next_index = match self.draft_history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.draft_history_stash = Some(self.draft.clone());
                self.draft_history.len().saturating_sub(1)
            }
        };

        self.draft_history_index = Some(next_index);
        self.draft = self.draft_history[next_index].clone();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn history_next(&mut self) {
        let Some(index) = self.draft_history_index else {
            return;
        };

        if index + 1 < self.draft_history.len() {
            let next_index = index + 1;
            self.draft_history_index = Some(next_index);
            self.draft = self.draft_history[next_index].clone();
            self.cursor = self.draft.len();
            self.sync_command_menu_state();
            return;
        }

        self.draft_history_index = None;
        self.draft = self.draft_history_stash.take().unwrap_or_default();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn leave_history_navigation(&mut self) {
        if self.draft_history_index.is_some() && self.draft_history_stash.is_none() {
            self.draft_history_stash = Some(self.draft.clone());
        }
        self.draft_history_index = None;
    }
}

/// Build the (role, content) turns seeded into a freshly (re)built agent engine
/// on resume or a key/model switch. User/assistant text carries over verbatim;
/// tool steps — which `seed_history` can't replay as real tool messages (no call
/// IDs) — are folded into compact assistant notes (`[used read_file: a.rs → 3
/// lines]`) so the engine remembers what it already did instead of going amnesiac
/// about its own prior work.
pub(super) fn agent_seed_turns(history: &[ChatMessage]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < history.len() {
        let m = &history[i];
        match m.role.as_str() {
            "user" | "assistant" if !m.content.trim().is_empty() => {
                out.push((m.role.clone(), m.content.clone()));
            }
            "tool_call" => {
                let (name, args) = decode_tool_call(&m.content);
                let target = tool_call_target(&name, &args);
                let mut note = if target.is_empty() {
                    format!("[used {name}]")
                } else {
                    format!("[used {name}: {target}]")
                };
                // Fold the immediately-following result's first line in as the outcome.
                if let Some(next) = history.get(i + 1).filter(|n| n.role == "tool_result") {
                    let summary: String = next
                        .content
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(120)
                        .collect();
                    if !summary.trim().is_empty() {
                        note.push_str(&format!(" → {}", summary.trim()));
                    }
                    i += 1; // consume the result
                }
                out.push(("assistant".to_string(), note));
            }
            _ => {} // plan / other display-only roles aren't seeded
        }
        i += 1;
    }
    out
}

/// One-line, length-capped excerpt of a turn's prompt for the `/rewind` picker.
fn rewind_excerpt(text: &str) -> String {
    const MAX: usize = 56;
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    if line.is_empty() {
        "(no text)".to_string()
    } else if line.chars().count() > MAX {
        let mut s: String = line.chars().take(MAX - 1).collect();
        s.push('…');
        s
    } else {
        line.to_string()
    }
}

/// Trailing annotation on a `/rewind` picker row: a "conversation only" marker
/// when the turn has no tree snapshot (file revert unavailable); otherwise empty
/// (the exact file impact is reported in the notice after applying).
fn rewind_label_suffix(conversation_only: bool) -> String {
    if conversation_only {
        "  · conversation only".to_string()
    } else {
        String::new()
    }
}

/// Human-readable notice summarizing what a completed `/rewind` did.
fn rewind_notice(outcome: &RewindOutcome) -> String {
    if let Some(err) = &outcome.error {
        return format!("Rewound the conversation; file revert failed: {err}");
    }
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let mut parts = Vec::new();
    if outcome.restored > 0 {
        parts.push(format!(
            "restored {} file{}",
            outcome.restored,
            plural(outcome.restored)
        ));
    }
    if outcome.deleted > 0 {
        parts.push(format!(
            "removed {} file{}",
            outcome.deleted,
            plural(outcome.deleted)
        ));
    }
    if parts.is_empty() {
        "Rewound".to_string()
    } else {
        format!("Rewound — {}", parts.join(", "))
    }
}

/// Bridges the in-process `AgentEngine` to the chat TUI: engine callbacks become
/// `RuntimeEvent`s the event loop renders, and a permission request round-trips
/// through the loop's permission card via a oneshot.
struct ChatAgentUi {
    tx: UnboundedSender<RuntimeEvent>,
}

impl crate::agent::engine::AgentUi for ChatAgentUi {
    fn assistant_text(&mut self, delta: &str) {
        self.tx
            .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
                delta.to_string(),
            )))
            .ok();
    }

    fn context_usage(&mut self, tokens: u64, measured: bool) {
        self.tx
            .send(RuntimeEvent::AgentContext { tokens, measured })
            .ok();
    }

    fn plan_updated(&mut self, items: &[crate::agent::plan::PlanItem]) {
        let value = serde_json::to_value(items).unwrap_or(serde_json::Value::Null);
        self.tx.send(RuntimeEvent::AgentPlan(value)).ok();
    }

    fn tool_start(&mut self, name: &str, args: &serde_json::Value) {
        self.tx
            .send(RuntimeEvent::AgentToolCall {
                id: None,
                name: name.to_string(),
                args: args.clone(),
            })
            .ok();
    }

    fn tool_result(&mut self, _name: &str, result: &Result<String, String>) {
        let content = match result {
            Ok(s) => s.clone(),
            Err(e) => format!("error: {e}"),
        };
        self.tx.send(RuntimeEvent::AgentToolResult { content }).ok();
    }

    fn notify(&mut self, text: &str) {
        self.tx
            .send(RuntimeEvent::AgentNotice(text.to_string()))
            .ok();
    }

    fn footer(
        &mut self,
        _summary: Option<&str>,
        steps: usize,
        tokens: u64,
        context_tokens: u64,
        _elapsed: u64,
    ) {
        self.tx
            .send(RuntimeEvent::AgentFinished {
                steps,
                tokens,
                context_tokens,
            })
            .ok();
    }

    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        preview: Option<&'a str>,
    ) -> futures::future::BoxFuture<'a, Decision> {
        let tx = self.tx.clone();
        let tool = tool.to_string();
        let preview = preview.map(str::to_string);
        Box::pin(async move {
            let (reply, rx) = tokio::sync::oneshot::channel();
            if tx
                .send(RuntimeEvent::AgentPermission {
                    tool,
                    preview,
                    reply,
                })
                .is_err()
            {
                return Decision::Deny;
            }
            rx.await.unwrap_or(Decision::Deny)
        })
    }
}

async fn drive_cursor_turn(
    client: std::sync::Arc<crate::services::acp_client::AcpClient>,
    session_id: String,
    model_id: Option<String>,
    user_input: String,
    attachments: Vec<MessageAttachment>,
    tx: &UnboundedSender<RuntimeEvent>,
) -> Result<ChatTurnResult> {
    let blocks = cursor_acp::build_prompt_blocks(&user_input, &attachments)?;
    let mut stream = client.start_prompt(&session_id, blocks).await?;

    let mut turn_result = CursorTurnResult::default();
    let mut reasoning_buf = String::new();
    let mut forward = |chunk: CursorChunk<'_>| -> Result<()> {
        // Tool calls reuse the in-process agent's tool-call card (the renderer
        // coalesces runs); text/reasoning stream as deltas.
        let event = match chunk {
            CursorChunk::Content(t) => {
                RuntimeEvent::Delta(ChatResponseChunk::Content(t.to_string()))
            }
            CursorChunk::Reasoning(t) => {
                RuntimeEvent::Delta(ChatResponseChunk::Reasoning(t.to_string()))
            }
            CursorChunk::ToolCall { id, name, args } => {
                RuntimeEvent::AgentToolCall { id, name, args }
            }
            CursorChunk::ToolUpdate {
                id,
                args,
                result,
                failed,
            } => RuntimeEvent::AgentToolUpdate {
                id,
                args,
                result,
                failed,
            },
        };
        tx.send(event).ok();
        Ok(())
    };

    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                cursor_acp::consume_session_update(
                    &value,
                    &mut turn_result,
                    &mut reasoning_buf,
                    &mut forward,
                )?;
            }
            PromptEvent::Done(result) => {
                result
                    .map_err(|e| anyhow::anyhow!(e))
                    .context("cursor-agent ACP session/prompt failed")?;
                break;
            }
        }
    }
    if !reasoning_buf.is_empty() {
        turn_result.reasoning_content = Some(reasoning_buf);
    }

    Ok(ChatTurnResult {
        content: turn_result.content,
        usage: None,
        model: model_id,
        raw_body: None,
    })
}
