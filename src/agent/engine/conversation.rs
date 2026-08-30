//! Conversation state: history seeding/export/restore, transcript repair,
//! context reporting, and /rewind truncation.

use super::*;

use crate::agent::compaction;
use crate::agent::tokens;
use crate::services::vision_describe;

impl AgentEngine {
    /// `/clear`: drop the conversation, keep the system prompt. Also clears the
    /// compaction working set, else a cleared session would re-inject stale facts.
    pub fn reset(&mut self) {
        self.messages.truncate(1);
        self.last_summary = None;
        self.plan.clear();
        self.touched_files.clear();
        self.notes.clear();
        self.evidence.clear();
        // `/rewind` checkpoints' `msg_index` pointed into the cleared transcript.
        self.checkpoints.clear();
        self.turn_unsend = None;
        self.snipped_originals.clear();
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
            // Raw prior chat text — same forgery surface as a prompt.
            let content = if role == "user" {
                verify::neutralize_marker_lines(&content)
            } else {
                content
            };
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
    pub(super) fn rebuild_working_set_from_log(&mut self) {
        enum Derived {
            Call(String, Value),
            // A fold's pinned block — sole carrier of state whose tool calls were drained.
            Pinned(compaction::PinnedWorkingSet),
        }
        // Collect first (immutable borrow), then apply in message order —
        // `record_touched_file` borrows mut, and calls after a fold refine its state.
        let derived: Vec<Derived> = self
            .messages
            .iter()
            .flat_map(|m| match role(m) {
                "assistant" => m
                    .get("tool_calls")
                    .and_then(|c| c.as_array())
                    .into_iter()
                    .flatten()
                    .filter_map(|call| {
                        let name = call.pointer("/function/name").and_then(|v| v.as_str())?;
                        let args = call
                            .pointer("/function/arguments")
                            .and_then(|v| v.as_str())
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or(Value::Null);
                        Some(Derived::Call(name.to_string(), args))
                    })
                    .collect::<Vec<_>>(),
                // Line-exact = engine fold (typed copies defanged); group folds sit at the tail.
                "user" => user_text(m)
                    .filter(|t| t.lines().any(|l| l == compaction::SUMMARY_FOLD_PREFIX))
                    .and_then(|t| compaction::parse_pinned_block(&t))
                    .map(Derived::Pinned)
                    .into_iter()
                    .collect(),
                _ => Vec::new(),
            })
            .collect();
        for item in derived {
            match item {
                Derived::Call(name, args) => match name.as_str() {
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
                        // Same deterministic merge as the live path, so resume can't drift from it.
                        if let Ok(note) = notes::parse_note(&args) {
                            notes::merge_note(&mut self.notes, note, MAX_NOTES);
                        }
                    }
                    _ => {}
                },
                Derived::Pinned(pinned) => {
                    if !pinned.plan.is_empty() {
                        self.plan = pinned.plan;
                    }
                    for note in pinned.notes {
                        notes::merge_note(&mut self.notes, note, MAX_NOTES);
                    }
                    for path in pinned.files {
                        self.record_touched_path(&path);
                    }
                }
            }
        }
        // Evidence lives in user-role `[self-verify]` lines, not tool calls. Cleared
        // then re-derived: a rewind reverted the files, so its evidence must go too.
        self.evidence.clear();
        let records: Vec<verify::EvidenceRecord> = self
            .messages
            .iter()
            .filter(|m| role(m) == "user")
            .filter_map(user_text)
            .flat_map(|text| {
                text.lines()
                    .filter_map(verify::parse_evidence_line)
                    .collect::<Vec<_>>()
            })
            .collect();
        for record in records {
            verify::merge_evidence(&mut self.evidence, record, MAX_EVIDENCE);
        }
    }

    /// Append a user/assistant text turn, MERGING into the previous message when it
    /// has the same role and is plain text. The engine must never hold two
    /// consecutive same-role messages — Anthropic (via the bridge) 400s on
    /// non-alternating roles (non-retryable brick).
    /// Append the opening user turn (plain string or multimodal array), folding into a
    /// trailing user turn so two consecutive `user` messages never occur.
    pub(super) fn push_user_content(&mut self, content: Value) {
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

    pub(super) fn push_text_turn(&mut self, role: &str, content: String) {
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

    /// Un-send the current turn's opening user message: Esc with nothing streamed
    /// returned the text to the composer, so the engine's copy must go too or the
    /// next submit merges with it. Restores a merged-into prior tail verbatim,
    /// drops the checkpoint this turn pushed; no-op once anything followed.
    pub fn unsend_last_user_turn(&mut self) {
        let Some(undo) = self.turn_unsend.take() else {
            return;
        };
        if self.messages.len() != undo.msg_index + 1
            || self.messages.last().map(role) != Some("user")
        {
            return;
        }
        match undo.merged_prior {
            Some(prior) => self.messages[undo.msg_index] = prior,
            None => {
                self.messages.pop();
            }
        }
        if undo.checkpoint_pushed
            && self
                .checkpoints
                .last()
                .is_some_and(|c| c.msg_index == undo.msg_index)
        {
            self.checkpoints.pop();
        }
    }

    /// Restore the assistant↔tool invariant before a new turn. A turn torn down
    /// mid-tool (Esc/interrupt) can leave an `assistant` with `tool_calls` whose
    /// results were never pushed; appending a `user` then 400s every provider
    /// (non-retryable → the corrupted prefix re-sends every turn, bricking the
    /// session). Synthesize an `[interrupted]` result per unanswered call id.
    pub(super) fn repair_interrupted_tail(&mut self) {
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

    /// Estimate of the next request's prompt (system + tools + conversation), on the
    /// same [`estimate_tokens`] ruler as `context_report` so footer and `/context`
    /// agree. Seeds the live context-fill before real usage.
    pub(crate) fn estimated_prompt_tokens(&self) -> u64 {
        (estimate_tokens(&self.messages) + estimate_tokens(&self.tools_openai)) as u64
    }

    /// Calibrated composition snapshot for the `/context` viewer.
    pub fn context_report(&self) -> ContextReport {
        let cal = self.token_calibration;
        let calib = |tokens: usize| (tokens as f64 * cal) as u64;

        // `messages[0]` is the system message, with the `-c` block folded in.
        let sys_full = estimate_tokens(&self.messages[..self.messages.len().min(1)]);
        let injected = self.injected_context_tokens.min(sys_full);

        // Count the external specs actually advertised (deferred ones cost nothing).
        let is_external = |t: &Value| {
            self.external
                .as_ref()
                .is_some_and(|e| t["function"]["name"].as_str().is_some_and(|n| e.handles(n)))
        };
        let (mcp_tok, mcp_tool_count) = self.tools_openai.iter().filter(|t| is_external(t)).fold(
            (0usize, 0usize),
            |(tok, n), t| {
                (
                    tok + crate::agent::tokens::estimate_message_tokens(t),
                    n + 1,
                )
            },
        );
        let tools_full = estimate_tokens(&self.tools_openai);
        let transcript = &self.messages[self.messages.len().min(1)..];

        ContextReport {
            context_window: self.context_window,
            system_prompt: calib(sys_full.saturating_sub(injected)),
            injected_context: calib(injected),
            tools: calib(tools_full.saturating_sub(mcp_tok)),
            tool_count: self.tools_openai.len().saturating_sub(mcp_tool_count),
            mcp_tools: calib(mcp_tok),
            mcp_tool_count,
            mcp_deferred_count: self.deferred_tools.len(),
            messages: calib(estimate_tokens(transcript)),
            message_count: transcript.len(),
            calibration: cal,
        }
    }

    /// Under `AIVO_DEBUG`, warn when the cached prefix (system prompt + tools) drifts.
    pub(super) fn check_prefix_drift(&mut self) {
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
    /// In plan/ask mode the read-only reminder rides the copy's latest user message —
    /// ephemeral (never in `self.messages`), so it vanishes on mode exit.
    pub(super) fn outgoing_messages(&self) -> Vec<Value> {
        let mut out: Vec<Value> = if self.agent_tools_enabled {
            self.messages.clone()
        } else {
            self.messages
                .iter()
                .filter(|m| role(m) != "system")
                .cloned()
                .collect()
        };
        // Synthesized-message markers stay local; strict providers reject unknown fields.
        for m in &mut out {
            if let Some(obj) = m.as_object_mut() {
                obj.remove(super::SYNTHETIC_MARKER_KEY);
            }
        }
        // Reminders ride the request only, never `self.messages`, and must sit at
        // the tail: folded in behind this turn's own output, the next request's
        // prefix diverges there and re-bills every tool result uncached.
        let mut reminders: Vec<String> = Vec::new();
        if self.read_only {
            reminders.push(
                if self.ask_mode_on() {
                    crate::agent::ask_mode::ASK_TURN_REMINDER
                } else {
                    plan_mode::PLAN_TURN_REMINDER
                }
                .to_string(),
            );
        }
        if let Some(jobs) = &self.jobs {
            let running = jobs.running_snapshot();
            if !running.is_empty() {
                reminders.push(running_jobs_reminder(&running));
            }
        }
        if !reminders.is_empty()
            && let Some(last) = out.last_mut()
        {
            let block = reminders.join("\n\n");
            if role(last) == "user" {
                if let Some(content) = last.get_mut("content") {
                    let owned = content.take();
                    *content = append_text_block(owned, &block);
                }
            } else {
                out.push(json!({"role": "user", "content": block}));
            }
        }
        if self.substitute_images {
            substitute_image_parts(&mut out, &self.image_descriptions);
        }
        out
    }

    /// Any image part in the transcript — tool images live only here, never as TUI attachments.
    pub fn history_has_images(&self) -> bool {
        self.messages.iter().any(|m| {
            m.get("content")
                .and_then(|c| c.as_array())
                .is_some_and(|parts| parts.iter().any(tokens::is_image_part))
        })
    }

    /// Uncached image parts in the history plus `pending`, deduped by hash.
    pub(crate) fn undescribed_images(&self, pending: Option<&Value>) -> Vec<(String, String)> {
        let mut seen = std::collections::HashSet::new();
        let mut todo = Vec::new();
        let history = self.messages.iter().filter_map(|m| m.get("content"));
        for content in history.chain(pending) {
            let Value::Array(parts) = content else {
                continue;
            };
            for part in parts.iter().filter(|p| tokens::is_image_part(p)) {
                let Some(url) = part["image_url"]["url"].as_str() else {
                    continue;
                };
                // Hash the raw base64 payload — attachments (no data: prefix) key identically.
                let Some((_, data)) = vision_describe::parse_data_url(url) else {
                    continue;
                };
                let hash = vision_describe::image_hash(data);
                if self.image_descriptions.contains_key(&hash) || !seen.insert(hash.clone()) {
                    continue;
                }
                todo.push((hash, url.to_string()));
            }
        }
        todo
    }

    /// Record the paths the turn's yet-unrecorded segment changed into its
    /// checkpoint: the first segment diffs from the turn tree, a resumed segment
    /// from its own `seg_tree` (keeping idle-gap hand-edits out of the revert set).
    /// Idempotent, so the TUI's cancel path can call it without racing turn end.
    pub(crate) async fn record_turn_changes(&mut self) {
        let Some(cp) = self.checkpoints.last() else {
            return;
        };
        let prior = cp.changed.clone();
        let base = match (&prior, cp.seg_tree.clone()) {
            (None, _) => cp.tree.clone(),
            (Some(_), seg @ Some(_)) => seg,
            (Some(_), None) => return, // fully recorded — no open segment
        };
        let diff = match &base {
            Some(b) => match self.checkpoint_store.as_mut() {
                Some(store) => store.changed_since(b).await,
                None => Some(Vec::new()),
            },
            // No tree → nothing revertible; Some([]) still marks the turn
            // recorded so `rewind_to` won't lazy-diff it.
            None => Some(Vec::new()),
        };
        let Some(cp) = self.checkpoints.last_mut() else {
            return;
        };
        match diff {
            Some(paths) => {
                let mut set: std::collections::BTreeSet<std::path::PathBuf> =
                    prior.unwrap_or_default().into_iter().collect();
                set.extend(paths);
                cp.changed = Some(set.into_iter().collect());
            }
            None => {
                // Diff unavailable (git error / size cap) → the segment's changes
                // are unknown; drop the tree so the turn reads non-revertible.
                cp.tree = None;
                cp.changed = Some(prior.unwrap_or_default());
            }
        }
        cp.seg_tree = None;
    }

    /// Per-checkpoint `/rewind` targets in order for the picker: `(prompt, file_revertible)`.
    /// The TUI matches by prompt text newest-backward. Cheap and in-memory (no git).
    /// Revertible = any checkpoint from the turn onward has a tree (a read-only
    /// turn restores from the first later one — see [`Self::rewind_to`]).
    pub fn rewind_targets(&self) -> Vec<(String, bool)> {
        let mut out = Vec::with_capacity(self.checkpoints.len());
        let mut tree_from_here = false;
        for c in self.checkpoints.iter().rev() {
            tree_from_here |= c.tree.is_some();
            out.push((c.prompt.clone(), tree_from_here));
        }
        out.reverse();
        out
    }

    /// Rewind to checkpoint `ordinal`: revert the union of files the rewound turns
    /// changed (leaving the user's independent edits), truncate to the turn's user
    /// message, drop the rewound checkpoints, re-derive the working set. When no
    /// checkpoint from `ordinal` onward has a tree, the conversation alone rewinds.
    pub async fn rewind_to(&mut self, ordinal: usize) -> RewindOutcome {
        let mut outcome = RewindOutcome::default();
        // Close the last turn's open segment (an abort whose cancel record raced us).
        self.record_turn_changes().await;
        // Restore target: the turn's own pre-edit tree, or — for a read-only turn
        // that never snapshotted — the first later checkpoint's tree (turns in
        // between have no tree because they changed nothing, so it equals this
        // turn's end state).
        let tree = self.checkpoints[ordinal.min(self.checkpoints.len())..]
            .iter()
            .find_map(|c| c.tree.clone());
        // Union of paths every rewound turn changed; finalize never-recorded
        // (crash-orphaned) checkpoints lazily.
        let mut paths: std::collections::BTreeSet<std::path::PathBuf> =
            std::collections::BTreeSet::new();
        for i in ordinal..self.checkpoints.len() {
            let recorded = self.checkpoints[i].changed.clone();
            let changed = match recorded {
                Some(c) => c,
                None => match self.checkpoints[i].tree.clone() {
                    Some(t) => match self.checkpoint_store.as_mut() {
                        Some(store) => store.changed_since(&t).await.unwrap_or_default(),
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
        // The un-send record pointed into the truncated region.
        self.turn_unsend = None;
        self.rebuild_working_set_from_log();
        outcome
    }
}

/// Defang marker lines in user-supplied content (string or multimodal text parts).
pub(super) fn neutralize_evidence_markers(content: Value) -> Value {
    match content {
        Value::String(s) => Value::String(verify::neutralize_marker_lines(&s)),
        Value::Array(mut parts) => {
            for p in &mut parts {
                if p.get("type").and_then(|t| t.as_str()) == Some("text")
                    && let Some(t) = p.get("text").and_then(|t| t.as_str())
                {
                    p["text"] = json!(verify::neutralize_marker_lines(t));
                }
            }
            Value::Array(parts)
        }
        other => other,
    }
}

/// A message's text: the plain string, or joined multimodal text parts (a merge
/// can move a marker line into a part).
fn user_text(m: &Value) -> Option<String> {
    match m.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => Some(
            parts
                .iter()
                .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

fn running_jobs_reminder(running: &[String]) -> String {
    format!(
        "<system-reminder>Background jobs still running right now (you started these via run_bash \
with background: true):\n{}\nBefore telling the user a job finished or was stopped, confirm with \
check_job {{\"id\": \"…\"}} — don't assume. Stop one with check_job {{\"id\": \"…\", \"kill\": \
true}}.</system-reminder>",
        running.join("\n")
    )
}

fn append_text_block(content: Value, block: &str) -> Value {
    match content {
        Value::String(s) if s.trim().is_empty() => Value::String(block.to_string()),
        Value::String(s) => Value::String(format!("{s}\n\n{block}")),
        Value::Array(mut parts) => {
            parts.push(json!({"type": "text", "text": block}));
            Value::Array(parts)
        }
        other => other,
    }
}

/// A miss leaves the part untouched, so the provider error surfaces normally
/// instead of an image silently vanishing from the request.
pub(crate) fn substitute_image_parts(
    messages: &mut [Value],
    cache: &std::collections::HashMap<String, String>,
) {
    if cache.is_empty() {
        return;
    }
    for m in messages {
        let Some(Value::Array(parts)) = m.get_mut("content") else {
            continue;
        };
        for part in parts.iter_mut().filter(|p| tokens::is_image_part(p)) {
            let Some(text) = part["image_url"]["url"]
                .as_str()
                .and_then(vision_describe::parse_data_url)
                .and_then(|(_, data)| cache.get(&vision_describe::image_hash(data)))
            else {
                continue;
            };
            *part = json!({"type": "text", "text": text});
        }
    }
}
