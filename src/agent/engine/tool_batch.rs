//! Tool-batch execution: permission gates, parallel reads, bash escalation.

use super::*;

impl AgentEngine {
    /// Execute one turn's batch of tool calls, appending a `tool` message for each
    /// in call order: classify + permission-gate up front, run side-effect-free
    /// built-ins concurrently and the rest sequentially, then report in call order.
    /// Returns extra tokens accrued by any sub-agent runs.
    pub(super) async fn execute_tool_batch(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        tool_calls: &[ToolCall],
    ) -> (u64, Vec<(String, String)>) {
        // Lazy `/rewind` checkpoint: snapshot the pre-edit (turn-start) tree the first
        // time a batch isn't entirely read-only. Conservative — anything off the
        // `is_read_only` allowlist triggers it. A turn resumed after an interrupt
        // (`changed` already recorded) also snapshots a `seg_tree` diff base, so the
        // resumed segment's diff excludes the user's edits made in between.
        if !tool_calls.iter().all(|c| tools::is_read_only(&c.name)) {
            let need_tree = self.checkpoints.last().is_some_and(|c| c.tree.is_none());
            let need_seg = !need_tree
                && self
                    .checkpoints
                    .last()
                    .is_some_and(|c| c.changed.is_some() && c.seg_tree.is_none());
            if need_tree || need_seg {
                let tree = match self.checkpoint_store.as_mut() {
                    Some(store) => store.snapshot().await,
                    None => None,
                };
                if let Some(cp) = self.checkpoints.last_mut() {
                    if need_tree {
                        cp.tree = tree.clone();
                        // A prior record means an interrupt closed the first
                        // segment pre-mutation — this snapshot is also the seg base.
                        if cp.changed.is_some() {
                            cp.seg_tree = tree;
                        }
                    } else {
                        match tree {
                            Some(t) => cp.seg_tree = Some(t),
                            // Can't isolate the segment's diff → non-revertible.
                            None => cp.tree = None,
                        }
                    }
                }
            }
        }

        let mut extra_tokens = 0u64;
        // (tool, error) per failed call, for the same-signature failure guard.
        let mut failures: Vec<(String, String)> = Vec::new();
        let mut outcomes: Vec<Option<Result<String, String>>> = vec![None; tool_calls.len()];
        let mut parallel_idx: Vec<usize> = Vec::new();
        let mut sequential_idx: Vec<usize> = Vec::new();

        for (i, call) in tool_calls.iter().enumerate() {
            // A live mid-turn plan exit (Shift+Tab), picked up at this call
            // boundary so the rest of the turn runs unrestricted.
            if self.read_only && ctx.plan_exit_requested() {
                self.set_plan_mode(false);
            }
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
            // Backstop for a hallucinated mutating tool (also hidden from the schema).
            if self.read_only && tools::is_mutating(n) && n != "run_bash" {
                outcomes[i] = Some(Err(
                    "Plan mode is read-only — do not modify files. Investigate, or call \
`exit_plan_mode` with your plan."
                        .to_string(),
                ));
                continue;
            }
            // PreToolUse veto runs before the permission tiers — a veto never
            // prompts; an allow still goes through them.
            if let Some(hooks) = self.hooks.clone()
                && let Some(reason) = hooks.pre_tool_use_deny(n, &call.arguments, ctx.cwd).await
            {
                outcomes[i] = Some(Err(format!("blocked by PreToolUse hook: {reason}")));
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
            // Plan-mode bash confirms per call (allow-once, bypasses -y/auto/grants
            // like `catastrophic`); provably read-only inspection is exempt.
            let plan_bash =
                self.read_only && n == "run_bash" && !tools::is_readonly_command(&call.arguments);
            // Remote mutation: only auto-approve mode waives it; AlwaysAllow
            // remembers the command family so a deploy loop isn't re-prompted.
            let remote_side_effect = !catastrophic
                && !ctx.auto_approve_mode()
                && tools::is_remote_side_effect(n, &call.arguments);
            let remote_families = if remote_side_effect {
                call.arguments
                    .get("command")
                    .and_then(|c| c.as_str())
                    .map(tools::remote_mutation_prefixes)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let allowed = if catastrophic || plan_bash {
                let preview = tools::preview(n, &call.arguments);
                // Allow and AlwaysAllow both run it once only — never remembered.
                !matches!(
                    ui.ask_permission(n, preview.as_deref(), true).await,
                    Decision::Deny
                )
            } else if remote_side_effect
                && !self.grants.covers(n, &call.arguments, ctx.cwd)
                && !self.grants.covers_remote(&remote_families)
            {
                let preview = tools::preview(n, &call.arguments);
                match ui.ask_permission(n, preview.as_deref(), false).await {
                    Decision::Allow => true,
                    Decision::AlwaysAllow => {
                        if remote_families.is_empty() {
                            self.grants.remember(n, &call.arguments, ctx.cwd);
                        } else {
                            self.grants.remember_remote(&remote_families);
                        }
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
                match ui.ask_permission(n, preview.as_deref(), false).await {
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
            let sink = ui.subagent_sink();
            // A sink's live rows already show the fan-out; notify headless only.
            if sink.is_none() {
                ui.notify(&format!(
                    "running {} sub-agents in parallel",
                    subagent_idx.len()
                ));
            }
            if let Some(s) = &sink {
                let labels: Vec<String> = subagent_idx
                    .iter()
                    .map(|&i| subagent_display_name(&tool_calls[i].arguments))
                    .collect();
                s.begin(&labels);
            }
            let base = self.turn_usage.completion_tokens;
            let this: &Self = self;
            let mut sub_tokens_total = 0u64;
            // Chunk by the cap so a wide fan-out doesn't stampede the provider: each
            // chunk runs concurrently (join_all — same primitive as the read batch, and
            // unlike buffer_unordered it doesn't impose a higher-ranked Send bound on
            // the heavy sub-engine future), chunks run one after another.
            for (chunk_no, chunk) in subagent_idx.chunks(SUBAGENT_PARALLEL_CAP).enumerate() {
                let runs = chunk.iter().enumerate().map(|(j, &i)| {
                    let args = &tool_calls[i].arguments;
                    let slot = chunk_no * SUBAGENT_PARALLEL_CAP + j;
                    let sink = sink.clone().map(|s| (s, slot));
                    async move { (i, this.run_subagent(ctx, None, sink, base, args).await) }
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
            if let Some(s) = &sink {
                s.finish();
            }
            extra_tokens = extra_tokens.saturating_add(sub_tokens_total);
            self.turn_usage.completion_tokens = self
                .turn_usage
                .completion_tokens
                .saturating_add(sub_tokens_total);
            sequential_idx.retain(|i| !subagent_idx.contains(i));
        }

        // Opt-in edit-review gate: pause an edit-bearing batch for approval before
        // any write. Reject drops the reviewed calls (a sibling `run_bash` still runs).
        if ctx.review_edits_enabled() {
            let reviewed: Vec<usize> = sequential_idx
                .iter()
                .copied()
                .filter(|&i| {
                    let c = &tool_calls[i];
                    let n = subagents::normalize_tool_name(&c.name).unwrap_or(&c.name);
                    crate::agent::review::is_edit_tool(n)
                })
                .collect();
            if !reviewed.is_empty() {
                let items: Vec<crate::agent::review::ReviewItem> = reviewed
                    .iter()
                    .map(|&i| {
                        let c = &tool_calls[i];
                        let n = subagents::normalize_tool_name(&c.name).unwrap_or(&c.name);
                        crate::agent::review::review_item(i, n, &c.arguments)
                    })
                    .collect();
                if ui.review_edits(&items).await == crate::agent::review::ReviewDecision::Reject {
                    for &i in &reviewed {
                        outcomes[i] = Some(Err(
                            crate::agent::review::REVIEW_REJECTED_DIRECTIVE.to_string()
                        ));
                    }
                    sequential_idx.retain(|i| !reviewed.contains(i));
                }
            }
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
                    .run_subagent(ctx, Some(&mut *ui), None, base, &call.arguments)
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
                // Durable scratchpad (deterministic merge, capped oldest-first). Held in the engine, so it runs in the ordered pass.
                match notes::parse_note(&call.arguments) {
                    Ok(note) => Ok(match notes::merge_note(&mut self.notes, note, MAX_NOTES) {
                        notes::MergeOutcome::Added(n) => format!("Noted ({n} saved)."),
                        notes::MergeOutcome::Updated(id) => format!("Updated note '{id}'."),
                        notes::MergeOutcome::Refreshed => "Already noted (refreshed).".to_string(),
                    }),
                    Err(e) => Err(e),
                }
            } else if n == "remember" {
                // Notify so a saved memory never lands silently (poison audit).
                match crate::agent::memory::parse_remember(&call.arguments) {
                    Ok((fact, scope)) => {
                        let path = crate::agent::memory::path_for_scope(ctx.cwd, scope);
                        let label = scope.label();
                        match crate::agent::memory::remember(&path, &fact) {
                            Ok(crate::agent::memory::RememberOutcome::Added(count)) => {
                                // Global facts ride into every project — call that out.
                                if scope == crate::agent::memory::MemoryScope::Global {
                                    ui.notify(&format!(
                                        "remembered (GLOBAL — injected into ALL projects): {fact}"
                                    ));
                                } else {
                                    ui.notify(&format!("remembered ({label}): {fact}"));
                                }
                                Ok(format!(
                                    "Remembered ({count} saved, {label} scope) — this is injected \
into every future session. The user can audit or edit it via /memory."
                                ))
                            }
                            Ok(crate::agent::memory::RememberOutcome::Refreshed) => {
                                Ok("Already remembered (recency refreshed).".to_string())
                            }
                            Err(e) => Err(e),
                        }
                    }
                    Err(e) => Err(e),
                }
            } else if n == "memory_search" {
                match crate::agent::memory::parse_query(&call.arguments) {
                    Ok(query) => Ok(crate::agent::memory::search_result_text(ctx.cwd, &query)),
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
                    Ok((question, options, allow_free_text, multi_select)) => ui
                        .ask_user(&question, &options, allow_free_text, multi_select)
                        .await
                        .map(|answer| ask::confirmation(&answer)),
                    Err(e) => Err(e),
                }
            } else if n == "exit_plan_mode" {
                if !self.read_only {
                    Err(
                        "exit_plan_mode: not in plan mode (the plan was already approved or \
planning is off) — continue with the task."
                            .to_string(),
                    )
                } else {
                    match plan_mode::parse_exit_plan(&call.arguments) {
                        Ok(plan) => match ui.approve_plan(&plan).await {
                            Ok(PlanDecision::Approve) => {
                                // Restore tools now so this turn continues into execution.
                                self.set_plan_mode(false);
                                Ok(plan_mode::PLAN_APPROVED_RESULT.to_string())
                            }
                            Ok(PlanDecision::KeepPlanning { feedback }) => {
                                Ok(plan_mode::keep_planning_result(feedback.as_deref()))
                            }
                            Ok(PlanDecision::Discard) => {
                                Ok(plan_mode::PLAN_DISCARDED_RESULT.to_string())
                            }
                            Err(e) => Err(e),
                        },
                        Err(e) => Err(e),
                    }
                }
            } else if n == "search_tools" {
                // Deferred-MCP discovery: load matching schemas (engine state → ordered pass).
                match call.arguments.get("query").and_then(|v| v.as_str()) {
                    Some(q) if !q.trim().is_empty() => {
                        let max = call
                            .arguments
                            .get("max_results")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or(tool_search::SEARCH_DEFAULT_RESULTS)
                            .clamp(1, tool_search::SEARCH_MAX_RESULTS);
                        let hits = tool_search::rank(&self.deferred_tools, q.trim(), max);
                        let loaded = self.load_deferred_tools(&hits);
                        Ok(tool_search::format_loaded(
                            &loaded,
                            self.deferred_tools.len(),
                        ))
                    }
                    _ => Err("missing required string argument `query`".to_string()),
                }
            } else if let Some(ext) = self.external.clone().filter(|e| e.handles(&call.name)) {
                // External tool — keyed on its raw advertised name (`mcp__*`), never normalized (matches the shadow check).
                self.promote_deferred_tool(&call.name);
                ext.call(&call.name, &call.arguments).await
            } else if n == "run_bash" && jobs::wants_background(&call.arguments) {
                // Detached job — no escalation flow (a spawn returns before a sandbox block shows).
                match (
                    &self.jobs,
                    call.arguments.get("command").and_then(|v| v.as_str()),
                ) {
                    (Some(t), Some(cmd)) => t.spawn(cmd, ctx.cwd),
                    (None, _) => Err(
                        "background jobs aren't available in this run mode — run the \
command in the foreground (drop `background`)."
                            .into(),
                    ),
                    (_, None) => Err("missing required string argument `command`".into()),
                }
            } else if n == "run_bash" {
                // Run confined; a sandbox write-block offers an in-session escape hatch instead of a dead-end error.
                self.run_bash_with_escalation(ctx, ui, &call.arguments)
                    .await
            } else if n == "check_job" {
                match &self.jobs {
                    Some(t) => {
                        let id = call
                            .arguments
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim();
                        if call
                            .arguments
                            .get("kill")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            t.kill(id).await
                        } else {
                            t.check(id)
                        }
                    }
                    None => Err("no background jobs in this run mode.".into()),
                }
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

        // PostToolUse feedback folds into each call's result (like the LSP fold above).
        if let Some(hooks) = self.hooks.clone().filter(|h| h.has_post()) {
            for (i, call) in tool_calls.iter().enumerate() {
                let Some(result) = outcomes[i].as_ref() else {
                    continue;
                };
                let n = subagents::normalize_tool_name(&call.name).unwrap_or(&call.name);
                let Some(extra) = hooks
                    .post_tool_use(n, &call.arguments, result, ctx.cwd)
                    .await
                else {
                    continue;
                };
                let block = format!("\n\n[PostToolUse hook]\n{extra}");
                match outcomes[i].as_mut() {
                    Some(Ok(msg)) => msg.push_str(&block),
                    Some(Err(msg)) => msg.push_str(&block),
                    None => {}
                }
            }
        }

        // Emit results and append tool messages in call order (call↔result pairing intact).
        let mut repeated_reads: Vec<(String, String)> = Vec::new();
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
                // A successful mutation (or delegated work) invalidates the last green verify.
                if tools::is_mutating(n) || n == "subagent" {
                    self.dirty_since_verify = true;
                }
                if let Some(k) = tools::read_dedupe_key(n, &call.arguments, ctx.cwd) {
                    repeated_reads.push((k, call.id.clone()));
                }
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
        // Older copies of any read this batch repeated verbatim are now dead weight.
        self.supersede_duplicate_reads(ctx.cwd, &repeated_reads);

        (extra_tokens, failures)
    }

    /// Corrective hint for a repeatedly-failing tool: the exact error plus the tool's
    /// JSON schema, so the model can fix its arguments. `None` if the tool isn't in the
    /// current tool set (e.g. a hallucinated name) — nothing useful to echo.
    pub(super) fn tool_failure_hint(&self, tool: &str, error: &str) -> Option<String> {
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
    pub(super) async fn run_bash_with_escalation(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        args: &Value,
    ) -> Result<String, String> {
        let outcome =
            Self::pump_bash_progress(ui, |tx| tools::run_bash_confined(args, ctx.cwd, Some(tx)))
                .await;
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
                .ask_permission("run_bash_unsandboxed", Some(&preview), false)
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
        Self::pump_bash_progress(ui, |tx| tools::run_bash_unconfined(args, ctx.cwd, Some(tx))).await
    }

    /// Run a `run_bash` future, forwarding its live output chunks to the UI.
    pub(super) async fn pump_bash_progress<T, F, Fut>(ui: &mut dyn AgentUi, run: F) -> T
    where
        F: FnOnce(tools::BashProgress) -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let fut = run(tx);
        let mut fut = std::pin::pin!(fut);
        let out = loop {
            tokio::select! {
                out = &mut fut => break out,
                Some(chunk) = rx.recv() => ui.tool_output("run_bash", &chunk),
            }
        };
        // Completion can race chunks still queued in the channel.
        while let Ok(chunk) = rx.try_recv() {
            ui.tool_output("run_bash", &chunk);
        }
        out
    }

    /// True when a `write_file` would overwrite an existing file the model hasn't
    /// read/written this session — a blind clobber worth confirming. New or
    /// already-touched files pass through; edit_file/multi_edit must read first, so never blind.
    pub(super) fn write_clobbers_unread(&self, name: &str, args: &Value, cwd: &Path) -> bool {
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

    pub(super) fn record_touched_file(&mut self, name: &str, args: &Value) {
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
}
