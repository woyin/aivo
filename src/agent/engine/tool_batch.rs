//! Tool-batch execution: permission gates, parallel reads, bash escalation.

use super::*;

use crate::agent::permission::{self, PermissionAction, Resolution};

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
        // Normalize aliased tool names once for the whole batch. External
        // (`mcp__*`) names never normalize; routing that needs the raw advertised
        // name still reads `call.name`.
        let names: Vec<&str> = tool_calls
            .iter()
            .map(|c| subagents::normalize_tool_name(&c.name).unwrap_or(&c.name))
            .collect();
        // Lazy `/rewind` checkpoint: snapshot the pre-edit (turn-start) tree the first
        // time a batch isn't entirely read-only. Conservative — anything off the
        // `is_read_only` allowlist triggers it. A turn resumed after an interrupt
        // (`changed` already recorded) also snapshots a `seg_tree` diff base, so the
        // resumed segment's diff excludes the user's edits made in between.
        if !names.iter().all(|n| tools::is_read_only(n)) {
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
        let mut batch_images: Vec<(String, Vec<crate::agent::engine::ToolImage>)> = Vec::new();
        let mut parallel_idx: Vec<usize> = Vec::new();
        let mut sequential_idx: Vec<usize> = Vec::new();

        // A read placed after a mutation must see its effects: parallel-safe calls
        // past the first workspace-touching call run in the ordered pass instead.
        // Inline-resolved calls never reach the executor, so they don't count.
        let barrier = tool_calls.iter().enumerate().position(|(i, c)| {
            let n = names[i];
            !matches!(n, "update_plan" | "finish_turn")
                && (!tools::is_read_only(n)
                    || self.external.as_ref().is_some_and(|e| e.handles(&c.name)))
        });

        for (i, call) in tool_calls.iter().enumerate() {
            // Live mid-turn plan transitions (Shift+Tab), picked up at this call
            // boundary: exit unrestricts the rest of the turn, entry restricts it.
            if self.read_only && ctx.plan_exit_requested() {
                self.set_plan_mode(false);
            } else if !self.read_only && ctx.plan_enter_requested() {
                self.set_plan_mode(true);
            }
            let n = names[i];
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
            // Engine-handled convergence report; a rejection is a normal tool error.
            if n == "finish_turn" {
                ui.tool_start(n, &call.arguments);
                outcomes[i] = Some(
                    self.handle_finish_request(ctx, ui, tool_calls.len(), &call.arguments)
                        .await,
                );
                continue;
            }
            ui.tool_start(n, &call.arguments);
            // Already declined this turn — auto-deny instead of re-prompting.
            let deny_sig = deny_sig(n, &call.arguments);
            if self.denied_sigs.contains(&deny_sig) {
                outcomes[i] = Some(Err(format!(
                    "{DENIED_BY_USER_RESULT} (auto-denied: the user already declined this \
exact action this turn.)"
                )));
                continue;
            }
            // Backstop for a hallucinated state-changing tool (also hidden from the
            // schema); `subagent` has its own refusal below.
            if self.read_only && tools::hidden_in_plan_mode(n) && n != "subagent" {
                outcomes[i] = Some(Err(
                    "Plan mode is read-only — do not modify files or write artifacts. \
Investigate, or call `exit_plan_mode` with your plan."
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
            // Confirm only genuinely risky actions: destructive command, blind
            // overwrite of an unread file, or an untrusted external tool.
            let needs_confirm = tools::is_dangerous(n, &call.arguments)
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
            // At most one prompt per call: a remote gate that had to ask covers the
            // confirm gate too; passed silently (grant), the confirm gate still applies.
            let allowed = if catastrophic || plan_bash {
                self.resolve_permission(
                    ctx,
                    ui,
                    PermissionAction::Once {
                        ask_name: n,
                        preview: tools::preview(n, &call.arguments),
                    },
                )
                .await
                .allowed()
            } else {
                let remote = if remote_side_effect {
                    self.resolve_permission(
                        ctx,
                        ui,
                        PermissionAction::Remote {
                            name: n,
                            args: &call.arguments,
                            families: &remote_families,
                        },
                    )
                    .await
                } else {
                    Resolution::Covered
                };
                match remote {
                    Resolution::Denied => false,
                    Resolution::Approved => true,
                    Resolution::Covered if !needs_confirm => true,
                    Resolution::Covered => self
                        .resolve_permission(
                            ctx,
                            ui,
                            PermissionAction::Confirm {
                                name: n,
                                args: &call.arguments,
                            },
                        )
                        .await
                        .allowed(),
                }
            };
            if !allowed {
                self.denied_sigs.insert(deny_sig);
                outcomes[i] = Some(Err(DENIED_BY_USER_RESULT.to_string()));
                continue;
            }
            // A side-effect-free built-in runs concurrently — unless an external tool
            // shadows the same name, which must route to its source sequentially.
            let shadowed = self
                .external
                .as_ref()
                .is_some_and(|e| e.handles(&call.name));
            if tools::is_parallel_safe(n) && !shadowed && barrier.is_none_or(|b| i < b) {
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
                    let n = names[i];
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
            // Only the leading run of delegates pools — the pool runs before the
            // ordered pass, so a delegate behind a write must run in order.
            sequential_idx
                .iter()
                .copied()
                .take_while(|&i| names[i] == "subagent")
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
            // Worker pool, not chunked barriers: a shared cursor hands the next
            // delegate to whichever worker frees up first. join_all over the fixed
            // worker set avoids buffer_unordered's Send bound on the sub-engine future.
            let cursor = std::sync::atomic::AtomicUsize::new(0);
            // Mutex (not RefCell): the turn future must stay Send; locked only between awaits.
            type DelegateOutcome = (usize, Result<String, String>, u64);
            let done: std::sync::Mutex<Vec<DelegateOutcome>> =
                std::sync::Mutex::new(Vec::with_capacity(subagent_idx.len()));
            let workers = (0..SUBAGENT_PARALLEL_CAP.min(subagent_idx.len())).map(|_| {
                let (cursor, done, sink, subagent_idx) = (&cursor, &done, &sink, &subagent_idx);
                async move {
                    loop {
                        // Cursor position doubles as the sink slot (row order = call order).
                        let slot = cursor.fetch_add(1, Ordering::Relaxed);
                        let Some(&i) = subagent_idx.get(slot) else {
                            break;
                        };
                        let s = sink.clone().map(|s| (s, slot));
                        let (res, toks) = this
                            .run_subagent(
                                ctx,
                                None,
                                s,
                                base,
                                &tool_calls[i].arguments,
                                true,
                                subagent_idx.len(),
                            )
                            .await;
                        done.lock().unwrap().push((i, res, toks));
                    }
                }
            });
            futures::future::join_all(workers).await;
            let mut sub_tokens_total = 0u64;
            for (i, res, toks) in done.into_inner().unwrap() {
                sub_tokens_total = sub_tokens_total.saturating_add(toks);
                outcomes[i] = Some(res);
            }
            if let Some(s) = &sink {
                s.finish();
            }
            extra_tokens = extra_tokens.saturating_add(sub_tokens_total);
            self.turn_usage.completion_tokens = self
                .turn_usage
                .completion_tokens
                .saturating_add(sub_tokens_total);
            // Keep the status counter at the folded total after sink rows clear.
            ui.turn_tokens(self.turn_usage.completion_tokens);
            sequential_idx.retain(|i| !subagent_idx.contains(i));
        }

        // Opt-in edit-review gate: pause an edit-bearing batch for approval before
        // any write. Reject drops the reviewed calls (a sibling `run_bash` still runs).
        if ctx.review_edits_enabled() {
            let reviewed: Vec<usize> = sequential_idx
                .iter()
                .copied()
                .filter(|&i| crate::agent::review::is_edit_tool(names[i]))
                .collect();
            if !reviewed.is_empty() {
                let items: Vec<crate::agent::review::ReviewItem> = reviewed
                    .iter()
                    .map(|&i| {
                        crate::agent::review::review_item(i, names[i], &tool_calls[i].arguments)
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
            let n = names[i];
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
                // Fresh sub-engine on the same serve/cwd; tokens fold in even on failure.
                let base = self.turn_usage.completion_tokens;
                let (res, sub_tokens) = self
                    .run_subagent(ctx, Some(&mut *ui), None, base, &call.arguments, false, 1)
                    .await;
                extra_tokens += sub_tokens;
                self.turn_usage.completion_tokens =
                    self.turn_usage.completion_tokens.saturating_add(sub_tokens);
                res
            } else if n == "list_sessions" {
                self.list_sessions_result()
            } else if n == "send_session" {
                self.send_session(&call.arguments, &mut *ui).await
            } else if let Some(res) = self.run_tool_intrinsic(ui, n, &call.arguments).await {
                // Engine-state intrinsics (notes/session controls/schema search).
                res
            } else if n == "exit_plan_mode" {
                if !self.read_only {
                    Err(
                        "exit_plan_mode: not in plan mode (the plan was already approved or \
planning is off) — continue with the task."
                            .to_string(),
                    )
                } else if self.plan_card_dismissed {
                    // Esc'd plan card — don't pop a revised one.
                    self.turn_dismissals += 1;
                    ui.notify(
                        "suppressed a repeat plan card — the user dismissed the previous one",
                    );
                    Ok(plan_mode::PLAN_APPROVAL_DISMISSED.to_string())
                } else {
                    match plan_mode::parse_exit_plan(&call.arguments) {
                        Ok(plan) => match ui.approve_plan(&plan).await {
                            Ok(PlanDecision::Approve) => {
                                self.turn_dismissals = 0;
                                self.plan_card_dismissed = false;
                                // Restore tools now so this turn continues into execution.
                                self.set_plan_mode(false);
                                Ok(plan_mode::PLAN_APPROVED_RESULT.to_string())
                            }
                            Ok(PlanDecision::KeepPlanning { feedback }) => {
                                self.turn_dismissals = 0;
                                self.plan_card_dismissed = false;
                                Ok(plan_mode::keep_planning_result(feedback.as_deref()))
                            }
                            Ok(PlanDecision::Discard) => {
                                self.turn_dismissals = 0;
                                self.plan_card_dismissed = false;
                                Ok(plan_mode::PLAN_DISCARDED_RESULT.to_string())
                            }
                            // User action, not a tool failure (see the ask_user intrinsic).
                            Err(e) if e == plan_mode::PLAN_APPROVAL_DISMISSED => {
                                self.turn_dismissals += 1;
                                self.plan_card_dismissed = true;
                                Ok(e)
                            }
                            Err(e) => Err(e),
                        },
                        Err(e) => Err(e),
                    }
                }
            } else if let Some(ext) = self.external.clone().filter(|e| e.handles(&call.name)) {
                // External tool — keyed on its raw advertised name (`mcp__*`), never normalized (matches the shadow check).
                self.promote_deferred_tool(&call.name);
                ext.call(&call.name, &call.arguments).await.map(|out| {
                    if !out.images.is_empty() {
                        batch_images.push((call.name.clone(), out.images));
                    }
                    out.text
                })
            } else if n == "generate_image" {
                self.generate_image_call(ctx, &call.arguments)
                    .await
                    .map(|out| {
                        if !out.images.is_empty() {
                            batch_images.push((call.name.clone(), out.images));
                        }
                        out.text
                    })
            } else if n == "preview" {
                self.preview_call(ctx, &call.arguments)
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
            } else if crate::agent::file_tracker::is_write_tool(n) {
                // Same escape hatch as bash for an out-of-workspace target.
                self.run_write_with_escalation(ctx, ui, n, &call.arguments)
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
                        let wait = call
                            .arguments
                            .get("wait")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        if call
                            .arguments
                            .get("kill")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            t.kill(id).await
                        } else if wait > 0 {
                            t.check_wait(id, wait).await
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
                let n = names[i];
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
                let n = names[i];
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
            let n = names[i];
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
                    self.verify_state = verify::VerifyState::Dirty;
                }
                if let Some(k) = tools::read_dedupe_key(n, &call.arguments, ctx.cwd) {
                    repeated_reads.push((k, call.id.clone()));
                }
            } else if n == "subagent" {
                // A failed delegate may still have edited files (step limit mid-work).
                self.verify_state = verify::VerifyState::Dirty;
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
        // KNOWN-vision only — shim descriptions are fetched pre-turn, so a mid-turn
        // image would go out raw and 400 with no reroute. A user message because
        // tool messages are text-only on every wire.
        if !batch_images.is_empty() && self.model_reads_images {
            let mut parts: Vec<Value> = Vec::new();
            for (tool, images) in batch_images {
                for mut img in images {
                    parts.push(json!({
                        "type": "text",
                        "text": format!("Image generated by `{tool}` (saved to {}):", img.path.display()),
                    }));
                    // Prefix in place — no second multi-MB copy.
                    img.data_b64
                        .insert_str(0, &format!("data:{};base64,", img.mime));
                    parts.push(json!({
                        "type": "image_url",
                        "image_url": {"url": img.data_b64},
                    }));
                }
            }
            // Stripped before any request.
            let mut msg = json!({"role": "user", "content": parts});
            msg[crate::agent::engine::SYNTHETIC_MARKER_KEY] =
                json!(crate::agent::engine::SYNTHETIC_TOOL_IMAGES);
            self.messages.push(msg);
        }
        // Older copies of any read this batch repeated verbatim are now dead weight.
        self.supersede_duplicate_reads(ctx.cwd, &repeated_reads);

        (extra_tokens, failures)
    }

    /// Corrective hint for a repeatedly-failing tool: the exact error plus the tool's
    /// JSON schema so the model can fix its arguments. `None` if the tool isn't in the
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

    /// Saved like MCP image results, so the TUI preview path is reused.
    async fn generate_image_call(
        &self,
        ctx: &TurnCtx<'_>,
        args: &Value,
    ) -> Result<crate::agent::engine::ToolOutput, String> {
        let Some(source) = self.image_source.as_ref() else {
            return Err(
                "image generation isn't configured — turn it on in /config → Image generation"
                    .to_string(),
            );
        };
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or("missing required string argument `prompt`")?;
        let urls = crate::services::image_generate::generate(
            source,
            ctx.client,
            ctx.serve_base,
            ctx.auth.unwrap_or_default(),
            prompt,
        )
        .await?;
        let (saved_notes, images) = tokio::task::spawn_blocking(move || {
            crate::agent::mcp::save_data_url_images(&urls, "gen")
        })
        .await
        .map_err(|e| format!("image save task failed: {e}"))??;
        Ok(crate::agent::engine::ToolOutput {
            text: saved_notes.trim_start().to_string(),
            images,
        })
    }

    /// The pane lives in the client — the TUI pins from these arguments — so
    /// the engine only vets the target.
    fn preview_call(&self, ctx: &TurnCtx<'_>, args: &Value) -> Result<String, String> {
        if !self.preview_supported {
            return Err("the preview pane isn't available in this run mode.".to_string());
        }
        if args.get("close").and_then(Value::as_bool).unwrap_or(false) {
            return Ok("Closed the preview pane.".to_string());
        }
        let reload = args.get("reload").and_then(Value::as_bool).unwrap_or(false);
        let Some(target) = args
            .get("target")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            if reload {
                return Ok("Re-rendering the pinned preview.".to_string());
            }
            return Err(
                "missing required string argument `target` (or pass reload=true / close=true)"
                    .to_string(),
            );
        };
        if target.starts_with("http://") || target.starts_with("https://") {
            return Ok(format!("Previewing {target} in the side pane."));
        }
        let path = tools::resolve(ctx.cwd, target);
        let meta =
            std::fs::metadata(&path).map_err(|e| format!("preview target `{target}`: {e}"))?;
        if !meta.is_file() {
            return Err(format!("preview target `{target}` is not a file"));
        }
        if crate::services::svg_raster::classify_preview_target(&path).is_none() {
            return Err(format!(
                "preview target `{target}` isn't previewable — supported: PNG/JPEG images, SVG, HTML"
            ));
        }
        Ok(format!(
            "Previewing {target} in the side pane; it re-renders as the file changes on disk."
        ))
    }

    /// Run a `run_bash` call confined to the workspace. If the OS sandbox blocks a
    /// write, offer to re-run outside the sandbox (same approval flow) instead of a
    /// dead-end error. Auto-approve / a prior "always" skip that prompt; off a TTY it
    /// fails closed, so the blocked result (with its hint) flows back.
    ///
    /// The approved re-run still denies the protected roots where enforceable;
    /// a command that hits (or names) one is confirmed per call even under
    /// auto-approve, like the catastrophic floor.
    pub(super) async fn run_bash_with_escalation(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        args: &Value,
    ) -> Result<String, String> {
        let mut outcome =
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
        // Only seatbelt enforces the protected floor on the escalated re-run;
        // elsewhere escalation is a bare shell and the textual heuristic is all
        // that's left, so confirm per call instead of waiving silently.
        let floor_enforced = crate::agent::sandbox::escalated_sandbox_active();
        if tools::command_mentions_protected_path(command) || !floor_enforced {
            let why = if tools::command_mentions_protected_path(command) {
                "and it names a protected path (aivo's config dir or ~/.ssh)"
            } else {
                "and on this platform re-running it lifts write confinement entirely — \
protected paths (aivo's config dir, ~/.ssh) included"
            };
            let preview = format!(
                "{command}\n\nThe workspace sandbox blocked this, {why}. Re-run the full \
command with no write confinement?"
            );
            let action = PermissionAction::Once {
                ask_name: "run_bash_unsandboxed",
                preview: Some(preview),
            };
            if !self.resolve_permission(ctx, ui, action).await.allowed() {
                return outcome.result;
            }
            ui.notify(SANDBOX_ESCALATION_NOTICE);
            return Self::pump_bash_progress(ui, |tx| {
                tools::run_bash_unconfined(args, ctx.cwd, Some(tx))
            })
            .await;
        }
        // Narrow remedy first: open just the blocked directory instead of lifting
        // confinement. Read-only ignores extra roots, so the offer would be a no-op there.
        if crate::agent::sandbox::current_profile().writes_workspace()
            && let Some(root) = tools::addable_root_for_command(command, ctx.cwd)
        {
            let preview = format!(
                "{command}\n\nThe workspace sandbox blocked this — it touches {root}, \
outside {cwd}. Add {root} to this session's writable roots and re-run?",
                root = root.display(),
                cwd = ctx.cwd.display()
            );
            if !self.offer_add_write_root(ctx, ui, &root, preview).await {
                return outcome.result;
            }
            let rerun = Self::pump_bash_progress(ui, |tx| {
                tools::run_bash_confined(args, ctx.cwd, Some(tx))
            })
            .await;
            if !rerun.sandbox_blocked {
                return rerun.result;
            }
            outcome = rerun; // the derived root wasn't (all of) it
        }
        // Scoped to the exact command so "always" doesn't blanket-escalate every bash call.
        let action = PermissionAction::Escalated {
            ask_name: "run_bash_unsandboxed",
            key: permission::escalation_key("run_bash_unsandboxed", command),
            preview: format!(
                "{command}\n\nThe workspace sandbox blocked this — it writes outside {}. \
Re-run the full command without write confinement?",
                ctx.cwd.display()
            ),
        };
        if !self.resolve_permission(ctx, ui, action).await.allowed() {
            // Keep the blocked output + hint so the model sees the escalation was declined.
            return outcome.result;
        }
        ui.notify(SANDBOX_ESCALATION_NOTICE);
        let escalated =
            Self::pump_bash_progress(ui, |tx| tools::run_bash_escalated(args, ctx.cwd, Some(tx)))
                .await;
        if !escalated.sandbox_blocked {
            return escalated.result;
        }
        // Blocked again = a protected root — confirm per call, even under auto-approve.
        let action = PermissionAction::Once {
            ask_name: "run_bash_unsandboxed",
            preview: Some(format!(
                "{command}\n\nStill blocked after escalation: it writes to a protected path \
(aivo's config dir or ~/.ssh). Re-run with no write confinement at all?"
            )),
        };
        if !self.resolve_permission(ctx, ui, action).await.allowed() {
            return escalated.result;
        }
        ui.notify(SANDBOX_ESCALATION_NOTICE);
        Self::pump_bash_progress(ui, |tx| tools::run_bash_unconfined(args, ctx.cwd, Some(tx))).await
    }

    /// Run a file-write tool with the same escalation as a sandbox-blocked
    /// `run_bash`: an out-of-workspace target prompts (auto-approve / "always"
    /// skip it), then runs with confinement waived. A protected-root target is
    /// confirmed per call even under auto-approve.
    pub(super) async fn run_write_with_escalation(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        name: &str,
        args: &Value,
    ) -> Result<String, String> {
        let outside = tools::escaping_write_paths(name, args, ctx.cwd);
        // Independent of `outside`: with cwd = $HOME a `~/.ssh/…` write never escapes.
        let protected = tools::protected_write_paths(name, args, ctx.cwd);
        // Escalation must never bypass the read-only profile's refusal.
        if (outside.is_empty() && protected.is_empty())
            || crate::agent::sandbox::current_profile()
                == crate::agent::sandbox::SandboxProfile::ReadOnly
        {
            return tools::execute(name, args, ctx.cwd).await;
        }
        // Narrow remedy first: one shared root covers every target — add it and
        // run the normal confined path.
        if protected.is_empty()
            && let Some(root) = tools::addable_root_for_paths(&outside, ctx.cwd)
        {
            let joined = outside.join(", ");
            let preview = format!(
                "{name}: {joined}\n\nThis writes under {root}, outside the workspace {cwd}. \
Add {root} to this session's writable roots?",
                root = root.display(),
                cwd = ctx.cwd.display()
            );
            if !self.offer_add_write_root(ctx, ui, &root, preview).await {
                return Err(refuse_outside(&joined, ctx.cwd));
            }
            return tools::execute(name, args, ctx.cwd).await;
        }
        // One approval covers the whole call, so preview and refusal name every target.
        let mut targets = protected.clone();
        targets.extend(outside.iter().filter(|p| !protected.contains(p)).cloned());
        let action = if !protected.is_empty() {
            let joined = targets.join(", ");
            PermissionAction::Once {
                ask_name: "write_outside_workspace",
                preview: Some(format!(
                    "{name}: {joined}\n\nThis writes to a protected path (aivo's config dir or \
~/.ssh). Allow this write?"
                )),
            }
        } else {
            let joined = outside.join(", ");
            // Scoped to the exact targets so "always" doesn't blanket-open the filesystem.
            PermissionAction::Escalated {
                ask_name: "write_outside_workspace",
                key: permission::escalation_key("write_outside_workspace", &joined),
                preview: format!(
                    "{name}: {joined}\n\nThis writes outside the workspace {}. Allow the write?",
                    ctx.cwd.display()
                ),
            }
        };
        let approved = self.resolve_permission(ctx, ui, action).await.allowed();
        if !approved {
            return Err(refuse_outside(&targets.join(", "), ctx.cwd));
        }
        ui.notify(WRITE_ESCALATION_NOTICE);
        tools::execute_write_unconfined(name, args, ctx.cwd).await
    }

    /// Offer to open `root` for this session after a workspace block; on
    /// approval applies it immediately and posts the notice. "Always" persists
    /// the (project, root) grant, so the root is silently re-added next session.
    async fn offer_add_write_root(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        root: &std::path::Path,
        preview: String,
    ) -> bool {
        let root_str = root.display().to_string();
        let action = PermissionAction::Escalated {
            ask_name: "add_write_root",
            key: permission::escalation_key(
                "add_write_root",
                &format!("{}\u{0}{root_str}", ctx.cwd.display()),
            ),
            preview,
        };
        if !self.resolve_permission(ctx, ui, action).await.allowed() {
            return false;
        }
        crate::agent::sandbox::add_extra_write_root(root.to_path_buf());
        ui.notify(&format!("{ADD_WRITE_ROOT_NOTICE}{root_str}"));
        true
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

    /// Validate a `finish_turn` call: store an accepted report (the turn loop
    /// converges on it); reject a premature `done` (unfinished plan steps,
    /// unverified changes) back to the model, bounded then fail-open.
    async fn handle_finish_request(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        batch_len: usize,
        args: &Value,
    ) -> Result<String, String> {
        if batch_len > 1 {
            return Err(
                "finish_turn must be the only call in its step — finish the other \
work first, then call it alone."
                    .to_string(),
            );
        }
        let report = crate::agent::finish::parse_finish(args)?;
        if report.status == crate::agent::finish::FinishStatus::Done
            && self.finish_rejections < MAX_FINISH_REJECTIONS
        {
            if plan::started(&self.plan)
                && self
                    .plan
                    .iter()
                    .any(|i| i.status != plan::PlanStatus::Completed)
            {
                self.finish_rejections += 1;
                return Err(format!(
                    "finish_turn(done) rejected — the plan still has unfinished steps:\n{}\n\
Complete them, or update the plan to reflect reality (mark blocked / remove with a reason), \
or finish with status \"blocked\".",
                    plan::pinned_block(&self.plan)
                ));
            }
            if self.verify_state == verify::VerifyState::Dirty && self.self_correct {
                let vplan = verify::detect_plan(ctx.cwd);
                if !vplan.is_empty() {
                    match self.run_verify_plan(ctx.cwd, ui, &vplan).await {
                        VerifyRun::Fail {
                            label,
                            summary,
                            lines,
                        } => {
                            self.finish_rejections += 1;
                            ui.notify(&format!("{label} failed — rejecting finish_turn(done)"));
                            return Err(format!(
                                "finish_turn(done) rejected — {label} is failing:\n{summary}\n{}\n\
Fix the cause, or finish with status \"blocked\".",
                                lines.join("\n")
                            ));
                        }
                        VerifyRun::Clean { .. } | VerifyRun::Unverified { .. } => {}
                    }
                }
            }
        }
        let confirmation = format!("Finish recorded (status: {}).", report.status.wire_name());
        self.finish_report = Some(report);
        Ok(confirmation)
    }

    pub(super) fn record_touched_file(&mut self, name: &str, args: &Value) {
        // One definition of "which paths does this tool touch", shared with the staleness
        // tracker and grant store (`apply_patch` carries many in its V4A body; the rest one).
        for path in crate::agent::file_tracker::tracked_paths(name, args) {
            self.record_touched_path(&path);
        }
    }

    /// Dedup-append one touched path, dropping the oldest past the cap.
    pub(super) fn record_touched_path(&mut self, path: &str) {
        let path = path.trim();
        if path.is_empty() || self.touched_files.iter().any(|p| p == path) {
            return;
        }
        if self.touched_files.len() >= MAX_TOUCHED_FILES {
            self.touched_files.remove(0);
        }
        self.touched_files.push(path.to_string());
    }

    // --- /rewind: tree checkpoints ---
}

/// Turn-scoped denial signature — ignores the call id so an identical re-issue matches.
fn deny_sig(name: &str, args: &Value) -> String {
    format!("{name}\u{0}{args}")
}

/// The one refusal wording for a declined out-of-workspace write.
fn refuse_outside(targets: &str, cwd: &std::path::Path) -> String {
    format!(
        "refused: `{targets}` is outside the workspace (or under a protected root) and the user \
declined the write. Use a path inside {} instead, or ask the user to relaunch with `--add-dir`.",
        cwd.display()
    )
}
