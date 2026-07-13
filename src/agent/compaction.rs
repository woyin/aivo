//! Context compaction for the agent engine: token-budget measurement, the
//! prune-then-summarize reclamation, calibration from measured/overflow usage,
//! the pinned working-set fold, and the deterministic hard-fit backstop.

use serde_json::{Map, Value, json};

use crate::agent::engine::{AgentEngine, AgentUi, DEFAULT_CONTEXT_WINDOW, TurnCtx};
use crate::agent::notes::Note;
use crate::agent::plan;
use crate::agent::protocol::ChatRequest;
use crate::agent::request::{role, serialize_transcript, truncate_str};
use crate::agent::retry::parse_overflow_actual;
use crate::agent::serve_client;
use crate::agent::tokens::{
    CALIBRATION_MIN_SAMPLE, MAX_CALIBRATION, calibration_ratio, estimate_str_tokens,
    estimate_tokens, keep_recent_tokens, usage_tokens,
};

/// Tokens held back from the window for the response + tool schemas.
pub(crate) const COMPACT_RESERVE: usize = 16_000;
/// A `tool` result longer than this (chars) is eligible for clearing once it ages
/// out of the recent window. Also the engine's "worth saving to an artifact" threshold.
pub(crate) const TOOL_RESULT_CLEAR_MIN: usize = 1_000;
/// Stub for a cleared tool result. Below [`TOOL_RESULT_CLEAR_MIN`] so clearing is
/// idempotent; the message + `tool_call_id` stay so assistant↔tool pairing holds.
pub(crate) const TOOL_RESULT_CLEARED: &str = "[earlier tool output cleared to save context]";
/// Stub for a read result superseded by an identical later call. Below
/// [`TOOL_RESULT_CLEAR_MIN`] so the stale-clear pass leaves it alone; idempotent.
pub(crate) const TOOL_RESULT_SUPERSEDED: &str =
    "[superseded — this exact call was repeated later; see the newest result]";
pub(crate) const SUMMARY_SYSTEM_PROMPT: &str = "You are compressing a coding-agent conversation to free up \
context. Write a concise but complete summary under these exact headings:\n\
## Goal\n## Constraints & Preferences\n## Progress (Done / In Progress / Blocked)\n\
## Key Decisions\n## Next Steps\n## Critical Context\n\n\
Preserve specifics: file paths, function/identifier names, exact values, commands run. Drop \
chit-chat. Output only the summary.";
/// Carry-forward variant: feeds the current running summary + only the NEW events
/// and asks for an in-place update, avoiding lossy drift from re-summarizing a blob.
pub(crate) const SUMMARY_UPDATE_SYSTEM_PROMPT: &str = "You are MAINTAINING a running summary of an ongoing \
coding-agent session. Below is the CURRENT summary, then the NEW events since it was written. \
Produce the UPDATED summary under these exact headings:\n\
## Goal\n## Constraints & Preferences\n## Progress (Done / In Progress / Blocked)\n\
## Key Decisions\n## Next Steps\n## Critical Context\n\n\
Preserve every still-relevant fact from the current summary verbatim (file paths, \
function/identifier names, exact values, commands run); merge in the new events; drop a fact \
only when the new events explicitly supersede it. Output only the updated summary.";
/// Ceiling (chars/4 tokens) on the pinned working-set block folded into a compaction;
/// plan kept whole, touched-files trimmed oldest-first so pinning can't re-overflow.
pub(crate) const PINNED_MAX_TOKENS: usize = 2_000;

impl AgentEngine {
    /// The window `maybe_compact` budgets against: the real one, or [`DEFAULT_CONTEXT_WINDOW`] if unknown (0).
    pub(crate) fn compaction_window(&self) -> usize {
        if self.context_window == 0 {
            DEFAULT_CONTEXT_WINDOW
        } else {
            self.context_window as usize
        }
    }

    /// Compaction budget in chars/4-estimate space: `(window - reserve) / calibration`,
    /// so `estimate <= budget` implies the calibrated real size fits.
    pub(crate) fn compaction_budget_estimate(&self) -> usize {
        let real = self.compaction_window().saturating_sub(COMPACT_RESERVE);
        ((real as f64) / self.token_calibration).floor() as usize
    }

    /// Fold a `(sent estimate, measured total)` sample into the calibration (measured
    /// total dodges cache-accounting quirks). Rises at once on undershoot, eases down slowly.
    pub(crate) fn update_calibration(&mut self, sent_estimate: usize, measured_total: u64) {
        if sent_estimate < CALIBRATION_MIN_SAMPLE || measured_total == 0 {
            return;
        }
        let ratio = calibration_ratio(measured_total, sent_estimate);
        // both operands >= 1.0, so the blend needs no floor
        self.token_calibration = if ratio > self.token_calibration {
            ratio
        } else {
            0.8 * self.token_calibration + 0.2 * ratio
        };
    }

    /// Raise the calibration from an overflow rejection: use the cited token count if present, else nudge up.
    pub(crate) fn recalibrate_from_overflow(&mut self, err: &str) {
        let estimate = estimate_tokens(&self.messages);
        match parse_overflow_actual(err) {
            Some(actual) if estimate >= CALIBRATION_MIN_SAMPLE => {
                // rise-only on overflow, unlike update_calibration's EMA
                self.token_calibration = self
                    .token_calibration
                    .max(calibration_ratio(actual, estimate));
            }
            _ => self.token_calibration = (self.token_calibration * 1.2).min(MAX_CALIBRATION),
        }
    }

    /// Deterministic recovery: fit the calibrated budget without a model call (a summary
    /// round-trip could itself overflow mid-recovery). Clears stale tool output, then hard-trims.
    pub(crate) fn force_fit_budget(&mut self) {
        let budget = self.compaction_budget_estimate();
        let mut cut = find_cut(&self.messages, keep_recent_tokens());
        // Single long turn (resume) has no interior user boundary → fall back so `enforce_budget` doesn't drop it to `[system, user]`.
        if cut <= 1 {
            cut = find_cut(&self.messages, 0);
        }
        self.clear_stale_tool_results(cut);
        // No summary round-trip is safe mid-overflow; fold a model-free marker.
        if cut > 1 && self.messages.get(cut).map(role) == Some("user") {
            let note = self.mechanical_summary();
            self.apply_compaction(cut, &note);
        }
        self.enforce_budget(budget);
    }

    /// If the history would overflow, summarize the older messages (quiet `complete`)
    /// and replace them. Cuts only at user boundaries so tool-call/result pairs stay
    /// intact. Returns tokens the summarization consumed (counted toward the turn, not a step).
    pub(crate) async fn maybe_compact(&mut self, ctx: &TurnCtx<'_>, ui: &mut dyn AgentUi) -> u64 {
        let budget = self.compaction_budget_estimate();
        let total = estimate_tokens(&self.messages);
        if total <= budget {
            return 0;
        }
        let mut cut = find_cut(&self.messages, keep_recent_tokens());
        // Single long turn (resume) has no interior user boundary → summarize into the latest user turn.
        if cut <= 1 {
            cut = find_cut(&self.messages, 0);
        }

        // Cheap pass first: if clearing OLD tool output alone brings us under budget,
        // do that and skip the LLM summary. Only when it alone suffices, so the summary path still sees full content.
        let savings = self.stale_tool_result_savings(cut);
        if savings > 0 && total.saturating_sub(savings) <= budget {
            ui.notify("freed context — cleared older tool output");
            self.clear_stale_tool_results(cut);
            return 0;
        }

        let tokens = self.summarize_range(ctx, ui, cut).await;
        // Backstop: guarantee the next request fits. A single summary pass can fall
        // short (huge recent tail, or `cut <= 1`). Trim deterministically so a turn is always sendable.
        self.enforce_budget(budget);
        tokens
    }

    /// Summarize `messages[1..cut]` and fold it in (no-op when `cut <= 1`); on empty
    /// output or failure folds a mechanical note. Returns tokens the call consumed.
    pub(crate) async fn summarize_range(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        cut: usize,
    ) -> u64 {
        if cut <= 1 {
            return 0;
        }
        let transcript = serialize_transcript(&self.messages[1..cut]);
        let request = self.build_summary_request(&transcript);
        ui.notify("compacting context…");
        match serve_client::complete(ctx.client, ctx.serve_base, ctx.auth, &request, &mut |_| {})
            .await
        {
            Ok(m) => {
                let summary = m.content.unwrap_or_default();
                if summary.trim().is_empty() {
                    let note = self.mechanical_summary();
                    self.apply_compaction(cut, &note);
                } else {
                    self.apply_compaction(cut, &summary);
                    // Carry forward so the next compaction updates it in place (anti-drift).
                    self.last_summary = Some(summary);
                }
                usage_tokens(&m.usage)
            }
            Err(_) => {
                // Don't re-send an overflowed request (not retryable → bricks the turn); drop mechanically.
                ui.notify("compaction summary unavailable — trimming older context");
                let note = self.mechanical_summary();
                self.apply_compaction(cut, &note);
                0
            }
        }
    }

    /// Calibrated estimate of the current context fill (the footer's pre-measurement value).
    pub fn estimated_context_tokens(&self) -> u64 {
        (self.estimated_prompt_tokens() as f64 * self.token_calibration) as u64
    }

    /// Whether a compaction could fold/clear anything — lets `/compact` skip a pointless round-trip.
    pub fn has_compactable_history(&self) -> bool {
        let cut = find_cut(&self.messages, keep_recent_tokens());
        cut > 1 || self.stale_tool_result_savings(cut) > 0
    }

    /// Manual `/compact`: summarize older turns regardless of budget (or clear stale output), then `footer`.
    pub async fn compact_now(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        elapsed_secs: u64,
    ) {
        let cut = find_cut(&self.messages, keep_recent_tokens());
        let tokens = if cut > 1 {
            self.summarize_range(ctx, ui, cut).await
        } else {
            self.clear_stale_tool_results(cut);
            0
        };
        // Footer carries the reduced fill; the chat layer reports the freed delta.
        ui.footer(
            None,
            0,
            tokens,
            self.estimated_context_tokens(),
            elapsed_secs,
        );
    }

    /// `/compact fast`: clear stale tool output, no model call. Returns `(before, after)` calibrated estimate.
    pub fn compact_now_local(&mut self) -> (u64, u64) {
        let before = self.estimated_context_tokens();
        let cut = find_cut(&self.messages, keep_recent_tokens());
        self.clear_stale_tool_results(cut);
        (before, self.estimated_context_tokens())
    }

    /// Tokens reclaimable by [`clear_stale_tool_results`], on the [`estimate_tokens`]
    /// ruler — the cheap compaction path trusts this figure without re-checking the budget.
    pub(crate) fn stale_tool_result_savings(&self, cut: usize) -> usize {
        self.messages
            .get(1..cut)
            .unwrap_or(&[])
            .iter()
            .filter(|m| role(m) == "tool")
            .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
            .filter(|s| s.len() > TOOL_RESULT_CLEAR_MIN)
            .map(|s| {
                let retained = estimate_str_tokens(TOOL_RESULT_CLEARED)
                    + artifact_pointer_line(s).map_or(0, |p| estimate_str_tokens(p) + 1);
                estimate_str_tokens(s).saturating_sub(retained)
            })
            .sum()
    }

    /// Stub older results of calls the latest batch repeated verbatim. `batch` =
    /// `(dedupe key, tool_call_id)` per SUCCESSFUL eligible call; that result is
    /// authoritative and only results BEFORE it are stubbed (a same-batch failed
    /// duplicate keeps its error text). Pairing stays intact.
    pub(crate) fn supersede_duplicate_reads(
        &mut self,
        cwd: &std::path::Path,
        batch: &[(String, String)],
    ) {
        use std::collections::{HashMap, HashSet};
        if batch.is_empty() {
            return;
        }
        let keys: HashSet<&str> = batch.iter().map(|(k, _)| k.as_str()).collect();
        // Newest successful call wins when one batch repeats a key.
        let mut authoritative: HashMap<&str, &str> = HashMap::new();
        for (k, id) in batch {
            authoritative.insert(k.as_str(), id.as_str());
        }
        // tool_call_id → key for calls matching a batch key; the name gate runs
        // before the arguments parse so bulky ineligible calls cost nothing.
        let mut call_keys: HashMap<String, String> = HashMap::new();
        for m in &self.messages {
            let Some(tcs) = m.get("tool_calls").and_then(Value::as_array) else {
                continue;
            };
            for tc in tcs {
                let (Some(id), Some(f)) =
                    (tc.get("id").and_then(Value::as_str), tc.get("function"))
                else {
                    continue;
                };
                let Some(name) = f.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let name = crate::agent::subagents::normalize_tool_name(name).unwrap_or(name);
                if !crate::agent::tools::is_dedupe_eligible(name) {
                    continue;
                }
                let Some(args) = f.get("arguments").and_then(Value::as_str) else {
                    continue;
                };
                let Ok(args) = serde_json::from_str::<Value>(args) else {
                    continue;
                };
                if let Some(k) = crate::agent::tools::read_dedupe_key(name, &args, cwd)
                    && keys.contains(k.as_str())
                {
                    call_keys.insert(id.to_string(), k);
                }
            }
        }
        let mut stub: Vec<usize> = Vec::new();
        for (key, auth_id) in &authoritative {
            // rposition: providers may reuse call ids across turns — bind to the newest.
            let Some(auth_idx) = self.messages.iter().rposition(|m| {
                role(m) == "tool" && m.get("tool_call_id").and_then(Value::as_str) == Some(*auth_id)
            }) else {
                continue;
            };
            stub.extend(
                self.messages[..auth_idx]
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| {
                        role(m) == "tool"
                            && m.get("tool_call_id")
                                .and_then(Value::as_str)
                                .is_some_and(|id| {
                                    call_keys.get(id).map(String::as_str) == Some(*key)
                                })
                    })
                    .map(|(i, _)| i),
            );
        }
        for i in stub {
            // Skip results already smaller than the stub.
            let len = self.messages[i]
                .get("content")
                .and_then(Value::as_str)
                .map_or(0, str::len);
            if len > TOOL_RESULT_SUPERSEDED.len() {
                self.messages[i]["content"] = json!(TOOL_RESULT_SUPERSEDED);
            }
        }
    }

    /// Replace bulky OLD `tool` output with [`TOOL_RESULT_CLEARED`], reclaiming
    /// context without a model call; message + `tool_call_id` stay (pairing intact). Idempotent.
    pub(crate) fn clear_stale_tool_results(&mut self, cut: usize) {
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
                // Keep any artifact-pointer line so the parent can re-read the saved report.
                let pointer = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .and_then(artifact_pointer_line)
                    .map(str::to_string);
                m["content"] = match pointer {
                    Some(p) => json!(format!("{TOOL_RESULT_CLEARED}\n{p}")),
                    None => json!(TOOL_RESULT_CLEARED),
                };
            }
        }
    }

    /// Model-free stand-in for a failed/empty summary; preserves any running summary so the thread isn't lost.
    pub(crate) fn mechanical_summary(&self) -> String {
        match &self.last_summary {
            Some(prev) => {
                format!("{prev}\n\n[Additional earlier turns omitted — summarization unavailable.]")
            }
            None => "[Earlier conversation omitted — summarization unavailable.]".to_string(),
        }
    }

    /// Last-resort, model-free trim to fit `budget`: drop whole oldest turns at user
    /// boundaries, then shorten the biggest string left (a `content` or a tool-call
    /// `arguments` blob). Always terminates; keeps the system prompt and call↔result pairing.
    pub(crate) fn enforce_budget(&mut self, budget: usize) {
        while estimate_tokens(&self.messages) > budget {
            let cut = find_cut(&self.messages, 0);
            if cut <= 1 {
                break; // only [system, last user turn] left — no boundary to drop
            }
            self.messages.drain(1..cut);
            self.rebase_checkpoints(cut, cut - 1);
        }
        // Shrink the largest string left, incl. tool-call `arguments`: a big call with
        // empty `content` in the irreducible recent turn is otherwise unreducible; truncated args stay paired with their id.
        while estimate_tokens(&self.messages) > budget {
            // loc: None = content; Some(j) = tool_calls[j] arguments.
            let pick = self
                .messages
                .iter()
                .enumerate()
                .skip(1)
                .flat_map(|(i, m)| {
                    let content = m
                        .get("content")
                        .and_then(|c| c.as_str())
                        .map(|s| (i, None, s.chars().count()));
                    let args = m
                        .get("tool_calls")
                        .and_then(|c| c.as_array())
                        .into_iter()
                        .flatten()
                        .enumerate()
                        .filter_map(move |(j, tc)| {
                            tc.get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .map(|s| (i, Some(j), s.chars().count()))
                        });
                    content.into_iter().chain(args)
                })
                .filter(|&(_, _, n)| n > 256)
                .max_by_key(|&(_, _, n)| n);
            let Some((idx, loc, n)) = pick else { break };
            let slot: &mut Value = match loc {
                None => &mut self.messages[idx]["content"],
                Some(j) => &mut self.messages[idx]["tool_calls"][j]["function"]["arguments"],
            };
            let cur = slot.as_str().unwrap_or("").to_string();
            let shortened = truncate_str(&cur, n / 2);
            if shortened.len() >= cur.len() {
                break;
            }
            *slot = json!(shortened);
        }
    }

    /// Build the throwaway system + user summarization request. First compaction
    /// summarizes fresh; later ones feed the prior summary back for an in-place update.
    /// Never folded into `self.messages`, so it can't affect role alternation.
    pub(crate) fn build_summary_request(&self, transcript: &str) -> ChatRequest {
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

    /// The pinned working set (plan + touched files) rendered for a compaction fold,
    /// trimmed to `PINNED_MAX_TOKENS` (plan kept whole, files trimmed oldest-first). Empty when nothing to pin.
    pub(crate) fn render_pinned_block(&self) -> String {
        let plan_block = plan::pinned_block(&self.plan);
        let mut notes: &[Note] = &self.notes;
        let mut files: &[String] = &self.touched_files;
        loop {
            let block = compose_pinned(&plan_block, notes, files);
            if block.is_empty() || estimate_str_tokens(&block) <= PINNED_MAX_TOKENS {
                return block;
            }
            // Keep the plan whole; trim files first, then notes (more valuable) — oldest-first. Bail at plan-only for progress.
            if !files.is_empty() {
                files = &files[1..];
            } else if !notes.is_empty() {
                notes = &notes[1..];
            } else {
                return block;
            }
        }
    }

    /// Replace `messages[1..cut]` with the summary, folding it INTO the first kept
    /// turn (a user message) rather than a standalone message before it — a standalone
    /// summary would be two consecutive users, which Anthropic 400s on (non-retryable → bricks after compaction).
    pub(crate) fn apply_compaction(&mut self, cut: usize, summary: &str) {
        let mut folded = format!("[Summary of earlier conversation]\n{summary}");
        // Pin plan + touched-files into the SAME fold so they never become a standalone same-role message.
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
            // Defensive (find_cut should land on a user turn): keep a standalone summary rather than drop it.
            self.messages.splice(
                1..cut,
                std::iter::once(json!({"role": "user", "content": summary})),
            );
            self.rebase_checkpoints(cut, cut.saturating_sub(2)); // splice: -cut+1, +1
        }
    }

    /// Keep `/rewind` checkpoints valid after a trim/compaction removed `removed`
    /// messages over `[1..cut]`: drop folded-away checkpoints (`msg_index < cut`),
    /// shift survivors down. Else `rewind_to` truncates at a stale index.
    pub(crate) fn rebase_checkpoints(&mut self, cut: usize, removed: usize) {
        self.checkpoints.retain_mut(|cp| {
            if cp.msg_index >= cut {
                cp.msg_index -= removed;
                true
            } else {
                false
            }
        });
    }
}

/// The artifact-pointer line, if any — the LAST match, since the real pointer is
/// appended at the end (a body line starting with the prefix must not shadow it).
fn artifact_pointer_line(content: &str) -> Option<&str> {
    content
        .lines()
        .rev()
        .find(|l| l.starts_with(crate::agent::engine::ARTIFACT_POINTER_PREFIX))
}

/// Render the pinned working set for a compaction: `## Pinned Plan`, `## Notes`,
/// `## Files touched`. Each section omitted when empty; "" when all are.
pub(crate) fn compose_pinned(plan_block: &str, notes: &[Note], files: &[String]) -> String {
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
            // id shown so the model can target an in-place update.
            match &n.id {
                Some(id) => out.push_str(&format!("- ({id}) {}\n", n.text)),
                None => out.push_str(&format!("- {}\n", n.text)),
            }
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

/// Index `cut` such that `messages[cut..]` is kept — chosen at a user-turn
/// boundary nearest to `keep_recent_tokens` of recent history.
pub(crate) fn find_cut(messages: &[Value], keep_recent_tokens: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::engine::{ARTIFACT_POINTER_PREFIX, Checkpoint};
    use crate::agent::protocol::Decision;
    use futures::future::BoxFuture;
    use serde_json::Value;

    fn engine() -> AgentEngine {
        AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0)
    }

    struct NoopUi;
    impl AgentUi for NoopUi {
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
            Box::pin(async { Decision::Deny })
        }
    }

    fn turn_ctx<'a>(client: &'a reqwest::Client, cwd: &'a std::path::Path) -> TurnCtx<'a> {
        TurnCtx {
            client,
            serve_base: "",
            auth: None,
            cwd,
            yes: true,
            auto_approve_all: false,
            auto_approve: None,
            review_edits: None,
        }
    }

    #[test]
    fn recalibrate_from_overflow_falls_back_to_a_nudge() {
        let mut e = engine();
        e.recalibrate_from_overflow("context length exceeded");
        assert!(
            (e.token_calibration - 1.2).abs() < 1e-9,
            "unparseable rejection nudges ×1.2, got {}",
            e.token_calibration
        );
        for _ in 0..20 {
            e.recalibrate_from_overflow("context length exceeded");
        }
        assert_eq!(
            e.token_calibration, MAX_CALIBRATION,
            "repeated nudges clamp at the ceiling"
        );
    }

    #[test]
    fn recalibrate_from_overflow_ignores_ratio_below_min_sample() {
        let mut e = engine();
        e.messages = vec![json!({"role":"system","content":"sys"})];
        assert!(estimate_tokens(&e.messages) < CALIBRATION_MIN_SAMPLE);
        e.recalibrate_from_overflow(
            "token count of 290000 exceeds the maximum allowed input length of 262112 tokens",
        );
        assert!(
            (e.token_calibration - 1.2).abs() < 1e-9,
            "tiny estimate must nudge, not calibrate from the cited count, got {}",
            e.token_calibration
        );
    }

    #[test]
    fn update_calibration_ignores_zero_measured_total() {
        let mut e = engine();
        e.update_calibration(100_000, 0);
        assert_eq!(e.token_calibration, 1.0, "a zero measurement is no signal");
    }

    /// The cheap path in `maybe_compact` trusts savings and skips `enforce_budget`;
    /// an overestimate would leave the next request over the window.
    #[test]
    fn stale_savings_do_not_overestimate_actual_reclaim() {
        let mut e = engine();
        let pointer = format!("{ARTIFACT_POINTER_PREFIX}/tmp/r.md — re-read it]");
        let escaped = format!("line \"quoted\"\n{}\n{pointer}", "y".repeat(6_000));
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q1"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": "s".repeat(8_000)}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"b","type":"function","function":{"name":"grep","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"b","content": escaped}),
            json!({"role":"user","content":"q2"}),
            json!({"role":"assistant","content":"done"}),
        ];
        let cut = find_cut(&e.messages, 0);
        assert_eq!(cut, 6, "both tool results are old");
        let savings = e.stale_tool_result_savings(cut);
        assert!(savings > 0);
        let before = estimate_tokens(&e.messages);
        e.clear_stale_tool_results(cut);
        let reclaimed = before - estimate_tokens(&e.messages);
        assert!(
            savings <= reclaimed + 2,
            "savings {savings} overestimates actual reclaim {reclaimed}"
        );
    }

    #[test]
    fn supersede_duplicate_reads_stubs_all_but_the_newest() {
        let mut e = engine();
        let cwd = std::path::Path::new("/w");
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": "old read ".repeat(50)}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"b","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"./src/x.rs\",\"offset\":1}"}}]}),
            json!({"role":"tool","tool_call_id":"b","content":"new read"}),
        ];
        let key =
            crate::agent::tools::read_dedupe_key("read_file", &json!({"path":"src/x.rs"}), cwd)
                .unwrap();
        e.supersede_duplicate_reads(cwd, &[(key, "b".to_string())]);
        assert_eq!(
            e.messages[3]["content"],
            json!(TOOL_RESULT_SUPERSEDED),
            "older duplicate stubbed (path + default-offset normalization)"
        );
        assert_eq!(
            e.messages[5]["content"],
            json!("new read"),
            "the newest result stays authoritative"
        );
    }

    /// Reused call ids across turns must bind to the newest message, not the first.
    #[test]
    fn supersede_duplicate_reads_handles_reused_call_ids_across_turns() {
        let mut e = engine();
        let cwd = std::path::Path::new("/w");
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"c0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"c0","content": "first read ".repeat(50)}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"c0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"c0","content": "second read ".repeat(50)}),
        ];
        let key =
            crate::agent::tools::read_dedupe_key("read_file", &json!({"path":"src/x.rs"}), cwd)
                .unwrap();
        e.supersede_duplicate_reads(cwd, &[(key, "c0".to_string())]);
        assert_eq!(
            e.messages[3]["content"],
            json!(TOOL_RESULT_SUPERSEDED),
            "the older same-id duplicate is stubbed"
        );
        assert!(
            e.messages[5]["content"]
                .as_str()
                .unwrap()
                .starts_with("second read"),
            "the newest result survives"
        );
    }

    /// A same-batch failed duplicate after the success keeps its error; the success survives.
    #[test]
    fn supersede_duplicate_reads_keeps_success_over_later_same_batch_failure() {
        let mut e = engine();
        let cwd = std::path::Path::new("/w");
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"old","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"old","content": "stale read ".repeat(50)}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"ok","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}},
                {"id":"boom","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"ok","content": "fresh read ".repeat(50)}),
            json!({"role":"tool","tool_call_id":"boom","content":"read src/x.rs: transient error while the file was being replaced, plus enough text to exceed the stub length"}),
        ];
        let key =
            crate::agent::tools::read_dedupe_key("read_file", &json!({"path":"src/x.rs"}), cwd)
                .unwrap();
        // Only the successful call reaches `batch` (keys are pushed on Ok only).
        e.supersede_duplicate_reads(cwd, &[(key, "ok".to_string())]);
        assert_eq!(
            e.messages[3]["content"],
            json!(TOOL_RESULT_SUPERSEDED),
            "the older duplicate is stubbed"
        );
        assert!(
            e.messages[5]["content"]
                .as_str()
                .unwrap()
                .starts_with("fresh read"),
            "the successful (authoritative) result must survive"
        );
        assert!(
            e.messages[6]["content"]
                .as_str()
                .unwrap()
                .contains("transient error"),
            "a later same-batch failure keeps its error text"
        );
    }

    #[test]
    fn supersede_duplicate_reads_skips_other_pages_and_tiny_results() {
        let mut e = engine();
        let cwd = std::path::Path::new("/w");
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q"}),
            // A different page of the same file: not a duplicate of the full read.
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\",\"offset\":100}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": "page two ".repeat(50)}),
            // An identical earlier grep, but its result is already tiny.
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"g1","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"foo\"}"}}]}),
            json!({"role":"tool","tool_call_id":"g1","content":"(no matches)"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"g2","type":"function","function":{"name":"grep","arguments":"{\"pattern\":\"foo\"}"}},
                {"id":"b","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}}]}),
            json!({"role":"tool","tool_call_id":"g2","content":"(no matches)"}),
            json!({"role":"tool","tool_call_id":"b","content": "full read ".repeat(50)}),
        ];
        let read_key =
            crate::agent::tools::read_dedupe_key("read_file", &json!({"path":"src/x.rs"}), cwd)
                .unwrap();
        let grep_key =
            crate::agent::tools::read_dedupe_key("grep", &json!({"pattern":"foo"}), cwd).unwrap();
        e.supersede_duplicate_reads(
            cwd,
            &[(read_key, "b".to_string()), (grep_key, "g2".to_string())],
        );
        assert!(
            e.messages[3]["content"]
                .as_str()
                .unwrap()
                .starts_with("page two"),
            "a different page must not be stubbed"
        );
        assert_eq!(
            e.messages[5]["content"],
            json!("(no matches)"),
            "a result already smaller than the stub is left alone"
        );
    }

    #[test]
    fn clear_stale_tool_results_is_idempotent() {
        let mut e = engine();
        let pointer = format!("{ARTIFACT_POINTER_PREFIX}/tmp/r.md — re-read it]");
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"q1"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content": format!("{}\n{pointer}", "x".repeat(5_000))}),
            json!({"role":"user","content":"q2"}),
        ];
        e.clear_stale_tool_results(4);
        let once = e.messages.clone();
        assert!(
            once[3]["content"].as_str().unwrap().contains(&pointer),
            "pointer survives the first clear"
        );
        e.clear_stale_tool_results(4);
        assert_eq!(e.messages, once, "second clear must be a no-op");
    }

    #[test]
    fn mechanical_summary_preserves_running_summary() {
        let mut e = engine();
        assert!(e.mechanical_summary().contains("omitted"));
        e.last_summary = Some("KEYFACT: db is postgres".to_string());
        let s = e.mechanical_summary();
        assert!(s.contains("KEYFACT: db is postgres"), "{s}");
        assert!(s.contains("summarization unavailable"), "{s}");
    }

    #[test]
    fn force_fit_fold_carries_running_summary_forward() {
        let mut e = engine();
        e.context_window = 20_000; // budget = 20_000 − COMPACT_RESERVE = 4_000
        e.last_summary = Some("KEYFACT: db is postgres".to_string());
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"original task"}),
            json!({"role":"assistant","content": "reasoning ".repeat(4_000)}),
            json!({"role":"user","content":"continue"}),
        ];
        e.force_fit_budget();
        assert!(estimate_tokens(&e.messages) <= e.compaction_budget_estimate());
        let last = e.messages.last().unwrap();
        assert_eq!(role(last), "user");
        let content = last["content"].as_str().unwrap();
        assert!(
            content.contains("KEYFACT: db is postgres"),
            "running summary lost in recovery: {content}"
        );
        assert!(content.contains("continue"), "latest turn kept: {content}");
    }

    #[test]
    fn apply_compaction_nonuser_cut_splices_standalone_summary() {
        let mut e = engine();
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"u1"}),
            json!({"role":"assistant","content":"a1"}),
            json!({"role":"assistant","content":"a2"}),
        ];
        for i in [1usize, 3] {
            e.checkpoints.push(Checkpoint {
                msg_index: i,
                prompt: format!("cp{i}"),
                tree: None,
                changed: None,
                seg_tree: None,
            });
        }
        e.apply_compaction(3, "early work");

        let roles: Vec<&str> = e.messages.iter().map(role).collect();
        assert_eq!(roles, vec!["system", "user", "assistant"]);
        let summary = e.messages[1]["content"].as_str().unwrap();
        assert!(
            summary.starts_with("[Summary of earlier conversation]")
                && summary.contains("early work"),
            "{summary}"
        );
        assert_eq!(e.messages[2]["content"], "a2", "kept turn intact");
        // splice nets −(cut−2): survivor 3 → 2; folded cp dropped
        assert_eq!(
            e.checkpoints
                .iter()
                .map(|c| c.msg_index)
                .collect::<Vec<_>>(),
            vec![2]
        );
    }

    #[test]
    fn find_cut_honors_keep_recent_and_no_user_boundary() {
        let m = |role: &str, content: String| json!({"role": role, "content": content});
        let messages = vec![
            m("system", "sys".into()),
            m("user", "u1".into()),
            m("assistant", "a".repeat(4_000)),
            m("user", "u2".into()),
            m("assistant", "a2".into()),
        ];
        assert_eq!(find_cut(&messages, 0), 3, "keep=0 cuts at the last user");
        assert_eq!(
            find_cut(&messages, 10_000),
            1,
            "a large keep window walks back to an earlier user boundary"
        );
        let no_user = vec![
            m("system", "sys".into()),
            m("assistant", "a1".into()),
            m("tool", "t1".into()),
        ];
        assert_eq!(find_cut(&no_user, 0), no_user.len());
    }

    #[test]
    fn render_pinned_block_trims_notes_then_bails_at_plan_only() {
        let mut e = engine();
        e.plan =
            plan::parse_plan(&json!({"plan":[{"step":"keep me","status":"pending"}]})).unwrap();
        e.notes = (0..400)
            .map(|i| Note {
                id: None,
                text: format!("note-{i} {}", "n".repeat(30)),
            })
            .collect();
        let block = e.render_pinned_block();
        assert!(estimate_str_tokens(&block) <= PINNED_MAX_TOKENS);
        assert!(block.contains("keep me"), "plan kept whole");
        assert!(!block.contains("note-0 "), "oldest note trimmed");
        assert!(block.contains("note-399 "), "newest note kept");

        let giant_plan: Vec<Value> = (0..300)
            .map(|i| json!({"step": format!("step-{i} {}", "p".repeat(40)), "status": "pending"}))
            .collect();
        e.plan = plan::parse_plan(&json!({ "plan": giant_plan })).unwrap();
        e.notes = Vec::new();
        let block = e.render_pinned_block();
        assert!(
            estimate_str_tokens(&block) > PINNED_MAX_TOKENS,
            "a plan alone can exceed the cap"
        );
        assert!(
            block.contains("step-0 ") && block.contains("step-299 "),
            "over-cap plan still returned whole, not looped on"
        );
    }

    #[test]
    fn compose_pinned_omits_empty_sections() {
        assert_eq!(compose_pinned("", &[], &[]), "");
        let files_only = compose_pinned("", &[], &["src/a.rs".to_string()]);
        assert!(files_only.starts_with("## Files touched"), "{files_only}");
        assert!(!files_only.contains("## Pinned Plan") && !files_only.contains("## Notes"));
    }

    #[tokio::test]
    async fn maybe_compact_under_budget_is_a_noop() {
        let mut e = engine();
        e.context_window = 100_000;
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"hi"}),
            json!({"role":"assistant","content":"yo"}),
        ];
        let snapshot = e.messages.clone();
        let client = reqwest::Client::new();
        let ctx = turn_ctx(&client, std::path::Path::new("."));
        let tokens = e.maybe_compact(&ctx, &mut NoopUi).await;
        assert_eq!(tokens, 0);
        assert_eq!(
            e.messages, snapshot,
            "under-budget compaction must not touch history"
        );
    }

    #[tokio::test]
    async fn summarize_range_noop_when_cut_at_most_one() {
        let mut e = engine();
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":"hi"}),
        ];
        let snapshot = e.messages.clone();
        let client = reqwest::Client::new();
        let ctx = turn_ctx(&client, std::path::Path::new("."));
        for cut in [0, 1] {
            let tokens = e.summarize_range(&ctx, &mut NoopUi, cut).await;
            assert_eq!(tokens, 0);
            assert_eq!(e.messages, snapshot);
        }
    }

    #[test]
    fn enforce_budget_drop_rebases_checkpoints() {
        let mut e = engine();
        let pad = "x".repeat(400);
        e.messages = vec![
            json!({"role":"system","content":"sys"}),
            json!({"role":"user","content":format!("u1 {pad}")}),
            json!({"role":"assistant","content":format!("a1 {pad}")}),
            json!({"role":"user","content":"u2 keep"}),
            json!({"role":"assistant","content":"a2 keep"}),
        ];
        for i in [1usize, 3] {
            e.checkpoints.push(Checkpoint {
                msg_index: i,
                prompt: format!("cp{i}"),
                tree: None,
                changed: None,
                seg_tree: None,
            });
        }
        e.enforce_budget(100);
        assert!(estimate_tokens(&e.messages) <= 100);
        assert!(
            e.messages[1]["content"]
                .as_str()
                .unwrap()
                .contains("u2 keep")
        );
        let cps: Vec<(usize, &str)> = e
            .checkpoints
            .iter()
            .map(|c| (c.msg_index, c.prompt.as_str()))
            .collect();
        assert_eq!(
            cps,
            vec![(1, "cp3")],
            "dropped-turn cp gone; survivor rebased"
        );
    }
}
