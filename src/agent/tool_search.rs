//! Deferred loading for external (MCP) tool schemas: past a threshold the
//! engine advertises one `search_tools` meta-tool instead of every schema,
//! and schemas load on demand. Calls route by name regardless — only the
//! advertisement is lazy.

use serde_json::{Value, json};

use crate::agent::protocol::ToolSpec;
use crate::agent::tokens::estimate_tokens;

/// Estimated-token cost of external specs above which they defer behind `search_tools`.
pub(crate) const DEFAULT_DEFER_TOKENS: usize = 8_000;
/// Cap on the tool-name list embedded in the `search_tools` description.
const NAME_LIST_MAX_CHARS: usize = 1_500;
/// Per-tool description excerpt in a search result.
const RESULT_DESC_MAX_CHARS: usize = 160;
/// Default / ceiling on tools loaded per search.
pub(crate) const SEARCH_DEFAULT_RESULTS: usize = 5;
pub(crate) const SEARCH_MAX_RESULTS: usize = 10;

/// Deferral threshold in estimate-token space. `AIVO_AGENT_MCP_DEFER_TOKENS`
/// overrides the default; `0` disables deferral (schemas always inline).
pub(crate) fn defer_threshold() -> Option<usize> {
    match crate::services::system_env::env_parse::<usize>("AIVO_AGENT_MCP_DEFER_TOKENS") {
        Some(0) => None,
        Some(n) => Some(n),
        None => Some(DEFAULT_DEFER_TOKENS),
    }
}

/// Whether `specs` cost enough to defer (a single tool never defers).
pub(crate) fn should_defer_at(specs: &[Value], threshold: usize) -> bool {
    specs.len() > 1 && estimate_tokens(specs) > threshold
}

fn spec_name(t: &Value) -> &str {
    t["function"]["name"].as_str().unwrap_or("")
}

fn spec_description(t: &Value) -> &str {
    t["function"]["description"].as_str().unwrap_or("")
}

/// The `search_tools` meta-tool spec. Embeds a capped name list so the model
/// can call a known tool directly without a search round-trip.
pub(crate) fn search_tools_spec(deferred: &[Value]) -> ToolSpec {
    let mut names = String::new();
    let mut listed = 0usize;
    for t in deferred {
        let n = spec_name(t);
        if n.is_empty() {
            continue;
        }
        if !names.is_empty() && names.len() + n.len() + 2 > NAME_LIST_MAX_CHARS {
            break;
        }
        if !names.is_empty() {
            names.push_str(", ");
        }
        names.push_str(n);
        listed += 1;
    }
    let more = deferred.len().saturating_sub(listed);
    let suffix = if more > 0 {
        format!(" (+{more} more — search to discover them)")
    } else {
        String::new()
    };
    ToolSpec {
        name: "search_tools".to_string(),
        description: format!(
            "{} external (MCP) tools are connected but their schemas aren't loaded into context. \
Search by keywords (tool purpose, entity, action) to find and load the ones you need — loaded \
tools become directly callable from your next step. You can also call any tool below directly \
by name if you already know its arguments. Available: {names}{suffix}.",
            deferred.len()
        ),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to match against tool names and descriptions"},
                "max_results": {"type": "integer", "description": "Max tools to load (default 5, max 10)"}
            },
            "required": ["query"]
        }),
    }
}

/// Rank deferred specs against `query` (name hits outweigh description hits);
/// returns matching indices, best first, capped at `max`.
pub(crate) fn rank(deferred: &[Value], query: &str, max: usize) -> Vec<usize> {
    let q = query.to_lowercase();
    let whole = q.trim();
    let terms: Vec<&str> = q
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let mut scored: Vec<(i64, usize)> = deferred
        .iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let name = spec_name(t).to_lowercase();
            let desc = spec_description(t).to_lowercase();
            let mut score = 0i64;
            if !whole.is_empty() && name.contains(whole) {
                score += 4;
            }
            for term in &terms {
                if name.contains(term) {
                    score += 3;
                } else if desc.contains(term) {
                    score += 1;
                }
            }
            (score > 0).then_some((score, i))
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(max).map(|(_, i)| i).collect()
}

/// Tool-result text for a search: what loaded, and how much remains unloaded.
pub(crate) fn format_loaded(loaded: &[Value], remaining: usize) -> String {
    if loaded.is_empty() {
        return if remaining == 0 {
            "(no unloaded tools remain — every external tool is already callable)".to_string()
        } else {
            format!(
                "(no matching tools; {remaining} unloaded tool(s) remain — try broader keywords)"
            )
        };
    }
    let mut out = format!("Loaded {} tool(s) — callable from now on:\n", loaded.len());
    for t in loaded {
        let desc: String = spec_description(t)
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(RESULT_DESC_MAX_CHARS)
            .collect();
        out.push_str(&format!("- {}: {desc}\n", spec_name(t)));
    }
    if remaining > 0 {
        out.push_str(&format!(
            "({remaining} more unloaded — search again if needed)"
        ));
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str, desc: &str) -> Value {
        json!({
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": {"type": "object"}}
        })
    }

    #[test]
    fn should_defer_needs_bulk_and_more_than_one_tool() {
        let one_big = vec![spec("mcp__s__a", &"x".repeat(60_000))];
        assert!(
            !should_defer_at(&one_big, 100),
            "a single tool never defers"
        );
        let small: Vec<Value> = (0..3)
            .map(|i| spec(&format!("mcp__s__t{i}"), "d"))
            .collect();
        assert!(!should_defer_at(&small, 8_000));
        let big: Vec<Value> = (0..60)
            .map(|i| spec(&format!("mcp__s__t{i}"), &"words and schemas ".repeat(80)))
            .collect();
        assert!(should_defer_at(&big, 8_000));
    }

    #[test]
    fn rank_prefers_name_hits_and_caps_results() {
        let specs = vec![
            spec("mcp__jira__create_issue", "Create a new Jira issue"),
            spec("mcp__jira__search_issues", "Search Jira issues by JQL"),
            spec(
                "mcp__gh__list_prs",
                "List pull requests; mentions issue links",
            ),
        ];
        let hits = rank(&specs, "issue", 10);
        assert_eq!(hits[..2], [0, 1], "name matches outrank description ones");
        assert_eq!(hits[2], 2, "description match still included");
        assert_eq!(rank(&specs, "issue", 1).len(), 1, "cap respected");
        assert!(rank(&specs, "zzz", 10).is_empty(), "no match, no result");
    }

    #[test]
    fn search_spec_lists_names_and_truncates_the_tail() {
        let specs: Vec<Value> = (0..200)
            .map(|i| spec(&format!("mcp__server__tool_number_{i}"), "d"))
            .collect();
        let s = search_tools_spec(&specs);
        assert_eq!(s.name, "search_tools");
        assert!(s.description.contains("200 external"));
        assert!(s.description.contains("mcp__server__tool_number_0"));
        assert!(
            s.description.contains("more — search to discover"),
            "over-cap names collapse to a count"
        );
        assert!(s.description.len() < 2_500, "name list is capped");
    }

    #[test]
    fn format_loaded_reports_names_and_remainder() {
        let loaded = vec![spec("mcp__s__a", "Does A things.\nSecond line ignored.")];
        let out = format_loaded(&loaded, 7);
        assert!(out.contains("mcp__s__a: Does A things."));
        assert!(!out.contains("Second line"));
        assert!(out.contains("7 more unloaded"));
        assert!(format_loaded(&[], 0).contains("already callable"));
        assert!(format_loaded(&[], 3).contains("broader keywords"));
    }
}
