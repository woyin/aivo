//! Engine-state intrinsics: tools that read or mutate the ENGINE (notes, memory,
//! session controls, deferred-schema search), not the workspace. Pulled out of
//! `execute_tool_batch` so its dispatch chain stays routing-only.

use super::*;

impl AgentEngine {
    /// Run `name` if it's an engine intrinsic; `None` sends the call down the
    /// rest of the dispatch chain. All intrinsics run in the ordered pass —
    /// they borrow engine state or await the user.
    pub(super) async fn run_tool_intrinsic(
        &mut self,
        ctx: &TurnCtx<'_>,
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
            // Notify so a saved memory never lands silently (poison audit).
            "remember" => match crate::agent::memory::parse_remember(args) {
                Ok((fact, scope, replaces)) => {
                    let path = crate::agent::memory::path_for_scope(ctx.cwd, scope);
                    let label = scope.label();
                    match crate::agent::memory::remember(
                        &path,
                        &fact,
                        &self.date,
                        replaces.as_deref(),
                    ) {
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
                        Ok(crate::agent::memory::RememberOutcome::Replaced(count)) => {
                            ui.notify(&format!("memory corrected ({label}): {fact}"));
                            Ok(format!(
                                "Replaced the outdated entry ({count} saved, {label} scope)."
                            ))
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            },
            "memory_search" => match crate::agent::memory::parse_query(args) {
                Ok(query) => Ok(crate::agent::memory::search_result_text(ctx.cwd, &query)),
                Err(e) => Err(e),
            },
            "switch_model" => match args.get("model").and_then(|v| v.as_str()) {
                Some(m) if !m.trim().is_empty() => ui.switch_chat_model(m.trim()).await,
                _ => Err("switch_model: missing `model`.".to_string()),
            },
            "set_effort" => match args.get("level").and_then(|v| v.as_str()) {
                Some(l) if !l.trim().is_empty() => ui.set_chat_effort(l.trim()).await,
                _ => Err("set_effort: missing `level`.".to_string()),
            },
            "ask_user" => match ask::parse_ask(args) {
                Ok((question, options, allow_free_text, multi_select)) => ui
                    .ask_user(&question, &options, allow_free_text, multi_select)
                    .await
                    .map(|answer| ask::confirmation(&answer)),
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
