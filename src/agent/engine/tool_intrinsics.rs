//! Engine-state intrinsics: tools that read or mutate the ENGINE (notes,
//! session controls, deferred-schema search), not the workspace. Pulled out of
//! `execute_tool_batch` so its dispatch chain stays routing-only.

use super::*;

impl AgentEngine {
    /// Run `name` if it's an engine intrinsic; `None` sends the call down the
    /// rest of the dispatch chain. All intrinsics run in the ordered pass —
    /// they borrow engine state or await the user.
    pub(super) async fn run_tool_intrinsic(
        &mut self,
        ui: &mut dyn AgentUi,
        name: &str,
        args: &Value,
    ) -> Option<Result<String, String>> {
        Some(match name {
            // Durable scratchpad (deterministic merge, capped oldest-first).
            "take_note" => match notes::parse_note(args) {
                Ok(note) => Ok(match notes::merge_note(&mut self.notes, note, MAX_NOTES) {
                    notes::MergeOutcome::Added(n) => format!("Noted ({n} saved)."),
                    notes::MergeOutcome::Updated(id) => format!("Updated note '{id}'."),
                    notes::MergeOutcome::Refreshed => "Already noted (refreshed).".to_string(),
                }),
                Err(e) => Err(e),
            },
            "switch_model" => match args.get("model").and_then(|v| v.as_str()) {
                Some(m) if !m.trim().is_empty() => ui.switch_chat_model(m.trim()).await,
                _ => Err("switch_model: missing `model`.".to_string()),
            },
            "switch_key" => match args.get("key").and_then(|v| v.as_str()) {
                Some(k) if !k.trim().is_empty() => {
                    let model = args
                        .get("model")
                        .and_then(|v| v.as_str())
                        .map(str::trim)
                        .filter(|m| !m.is_empty());
                    ui.switch_chat_key(k.trim(), model).await
                }
                _ => Err("switch_key: missing `key`.".to_string()),
            },
            "set_effort" => match args.get("level").and_then(|v| v.as_str()) {
                Some(l) if !l.trim().is_empty() => ui.set_chat_effort(l.trim()).await,
                _ => Err("set_effort: missing `level`.".to_string()),
            },
            "ask_user" => match ask::parse_ask(args) {
                // A card was already Esc'd this turn — don't pop another; the
                // auto-dismissal also advances the ladder, ending the turn.
                Ok(_) if self.turn_dismissals > 0 => {
                    self.turn_dismissals += 1;
                    ui.notify("suppressed a repeat question — the user dismissed the previous one");
                    Ok(ask::DISMISSED_DIRECTIVE.to_string())
                }
                Ok((question, options, allow_free_text, multi_select)) => {
                    match ui
                        .ask_user(&question, &options, allow_free_text, multi_select)
                        .await
                    {
                        Ok(answer) => {
                            self.turn_dismissals = 0;
                            self.plan_card_dismissed = false;
                            Ok(ask::confirmation(&answer))
                        }
                        // User action, not a tool failure — an Err would hit the
                        // failure guard, whose schema hint says to re-ask.
                        Err(e) if e == ask::DISMISSED_DIRECTIVE => {
                            self.turn_dismissals += 1;
                            Ok(e)
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            },
            // Deferred-MCP discovery: load matching schemas (engine state).
            "search_tools" => match args.get("query").and_then(|v| v.as_str()) {
                Some(q) if !q.trim().is_empty() => {
                    let max = args
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
            },
            _ => return None,
        })
    }
}
