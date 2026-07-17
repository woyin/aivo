//! The turn loop: user-turn entry, the step loop, steering, and job notices.

use super::*;

impl AgentEngine {
    /// Run one user turn to convergence: call the model, execute tool calls
    /// (permission-gated), repeat until it stops or a stop condition trips; footer.
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
    pub(super) fn begin_user_turn(&mut self, user_content: Value, checkpoint_prompt: String) {
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
                seg_tree: None,
            });
        }
        // Merge into a preceding user turn (e.g. a turn cancelled before its first
        // reply) rather than appending a second one (two consecutive users → Anthropic 400 / brick).
        self.push_user_content(user_content);
    }

    pub(super) async fn run_loop(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi) {
        let mut steps = 0usize;
        let mut leaked_nudges = 0usize;
        let mut completion_nudges = 0usize;
        let mut plan_nudges = 0usize;
        // Keeps a stale plan from an earlier turn from triggering the nudge.
        let mut plan_set_this_turn = false;
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
        let mut stop_hook_continues = 0usize;
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
            let mut terminal_error = false;
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
                        // Show the wait so a Retry-After pause doesn't read as a frozen UI.
                        let delay = retry_delay(retries, e.retry_after);
                        let wait = if delay.as_secs() >= 2 {
                            format!(" in {}s", delay.as_secs())
                        } else {
                            String::new()
                        };
                        ui.notify(&format!(
                            "{} — retrying{wait} ({retries}/{MAX_RETRIES})…",
                            retryable_error_label(&e)
                        ));
                        tokio::time::sleep(delay).await;
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
                        ui.notify_error(&terminal_error_notice(&e));
                        terminal_error = true;
                        break AssistantMessage {
                            content: None,
                            tool_calls: vec![],
                            usage: None,
                            truncated: false,
                            model: None,
                        };
                    }
                }
            };
            // End the turn without recording — an "[error: …]" assistant turn
            // would replay the failure to the model every later step and on resume.
            if terminal_error {
                converged = true;
                break;
            }
            if message.truncated {
                // The kept partial must not pass for a complete answer.
                ui.notify_error(
                    "the connection dropped mid-reply — the answer above may be incomplete",
                );
            }
            steps += 1;
            if let Some(m) = &message.model {
                self.billed_model = Some(m.clone());
            }
            let step_tokens = usage_tokens(&message.usage);
            tokens += step_tokens;
            if message.usage.is_some() {
                context_tokens = step_tokens;
                ui.context_usage(step_tokens, true);
                self.update_calibration(sent_estimate, step_tokens);
            }
            // Sum the real prompt/completion/cache split across steps (same parser as the serve, for a consistent index).
            if let Some(u) = &message.usage {
                if let Some(split) = extract_usage_from_value(&json!({ "usage": u })) {
                    self.turn_usage = self.turn_usage.merge(SessionTokens {
                        prompt_tokens: split.prompt,
                        completion_tokens: split.completion,
                        cache_read_tokens: split.cache_read,
                        cache_write_tokens: split.cache_creation,
                    });
                    ui.turn_tokens(self.turn_usage.completion_tokens);
                }
                if let Some(cost) = u.get("cost").and_then(|x| x.as_f64())
                    && cost > 0.0
                {
                    self.turn_cost_usd += cost;
                }
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
            // Same for the USD budget (`--max-cost`): estimated spend from measured usage.
            if self.max_cost_usd > 0.0
                && let Some(cost) = self
                    .cost_pricing
                    .as_ref()
                    .and_then(|p| p.cost_usd(&self.turn_usage))
                && cost >= self.max_cost_usd
            {
                ui.notify(&format!(
                    "stopping: reached the cost budget (~${cost:.2} of ${:.2})",
                    self.max_cost_usd
                ));
                converged = true;
                break;
            }

            // Empty completion converges the turn; don't record it — an empty assistant 400s the Anthropic bridge (non-retryable → bricks the next turn).
            let no_output = message.tool_calls.is_empty()
                && message.content.as_deref().is_none_or(str::is_empty);
            if no_output {
                // No answer = a failed turn: the error channel persists it, skips
                // the `✶ Done` marker, and fails a headless run closed.
                ui.notify_error("the model returned an empty response — no answer produced");
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
                    truncated: false,
                    model: None,
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
                // A plan set this turn but never started isn't done — nudge once.
                // Plan mode is exempt: proposing without executing is the point.
                if !self.read_only
                    && plan_set_this_turn
                    && plan_nudges < MAX_PLAN_NUDGES
                    && !self.plan.is_empty()
                    && !plan::started(&self.plan)
                {
                    plan_nudges += 1;
                    ui.notify("the plan hasn't been started — asking the model to continue");
                    self.push_text_turn("user", PLAN_NUDGE.to_string());
                    continue;
                }
                // A declared-done turn isn't accepted while the validator fails — feed
                // the failure back (bounded) so the model fixes the cause.
                if let Some(v) = &validator
                    && self.dirty_since_verify
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
                        Ok(()) => {
                            self.dirty_since_verify = false;
                            ui.notify(&format!("verified: {} passed", v.label));
                        }
                    }
                }
                // A Stop hook may refuse the stop; the turn continues with its guidance.
                if stop_hook_continues < MAX_STOP_HOOK_CONTINUES
                    && let Some(hooks) = self.hooks.clone()
                    && let Some(guidance) = hooks
                        .stop_guidance(message.content.as_deref().unwrap_or(""), ctx.cwd)
                        .await
                {
                    stop_hook_continues += 1;
                    ui.notify("a Stop hook asked the agent to continue");
                    self.push_text_turn("user", format!("{STOP_HOOK_PREFIX}\n{guidance}"));
                    continue;
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

            plan_set_this_turn |= message.tool_calls.iter().any(|c| {
                subagents::normalize_tool_name(&c.name).unwrap_or(&c.name) == "update_plan"
            });

            // Execute this batch (permission-gated); returns extra tokens accrued inside
            // it (sub-agent calls) plus each failed call's (tool, error) for the guard.
            let (batch_tokens, failures) =
                self.execute_tool_batch(ctx, ui, &message.tool_calls).await;
            tokens += batch_tokens;

            if repeats + 1 >= REPEAT_LIMIT || page_repeats + 1 >= REPEAT_LIMIT {
                ui.notify(STOP_NO_PROGRESS);
                ui.turn_stopped(TurnStop::NoProgress);
                converged = true;
                break;
            }

            // Same-signature tool-failure guard: hint the schema, then hard-stop a loop.
            match failure_guard.observe(&failures) {
                guards::FailureAction::Stop => {
                    ui.notify(STOP_TOOL_FAILURE);
                    ui.turn_stopped(TurnStop::ToolFailureLoop);
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

            self.inject_job_notices();
            self.inject_steering(ui);
        }

        if !converged {
            ui.notify(&format!("reached the step limit ({})", self.max_steps));
            ui.turn_stopped(TurnStop::StepLimit);
        }
        ui.footer(
            None,
            steps,
            tokens,
            context_tokens,
            started.elapsed().as_secs(),
        );

        // Record paths this turn changed so `/rewind` reverts only the agent's edits.
        self.record_turn_changes().await;
    }

    /// Fold interjections into the last tool result — a fresh user turn after
    /// tool results 400s the Anthropic bridge.
    pub(super) fn inject_steering(&mut self, ui: &mut dyn AgentUi) {
        let steering = ui.drain_steering();
        if steering.is_empty() {
            return;
        }
        let block = format!(
            "<user_interjection>\n{}\n</user_interjection>\nThe user sent this while you were \
working. Factor it in before continuing — it may change what to do next.",
            steering.join("\n\n")
        );
        self.fold_into_last_tool_result(block);
    }

    /// Surface jobs that finished since the last step, so the model needn't busy-poll.
    pub(super) fn inject_job_notices(&mut self) {
        let Some(jobs) = &self.jobs else { return };
        let notices = jobs.drain_finished_notices();
        if notices.is_empty() {
            return;
        }
        let block = format!(
            "<background_jobs>\n{}\n</background_jobs>\nThese background job(s) finished while \
you were working. If the outcome matters to the task, inspect the log; otherwise continue.",
            notices.join("\n")
        );
        self.fold_into_last_tool_result(block);
    }

    /// Append `block` to the last tool result, or push a user turn when there is
    /// none — a fresh user turn straight after tool results 400s the Anthropic bridge.
    pub(super) fn fold_into_last_tool_result(&mut self, block: String) {
        if let Some(last) = self.messages.last_mut()
            && last.get("role").and_then(Value::as_str) == Some("tool")
            && let Some(c) = last.get("content").and_then(Value::as_str)
        {
            last["content"] = json!(format!("{c}\n\n{block}"));
        } else {
            self.push_text_turn("user", block);
        }
    }
}
