//! Index-keyed accumulator for streamed OpenAI chat `tool_calls` deltas.
//! Shared by the agent's serve client and the buffered SSE fallback so the
//! delta-merge quirks (qwen full-name re-sends, bogus huge indexes) are handled
//! once. Finalization (id fallback, argument repair) stays with each caller.

use serde_json::Value;

/// Upper bound on parallel tool calls in one streamed response — guards the
/// index-keyed accumulator against a bogus huge `index` from upstream (a 2^31
/// index would OOM). No real response has anywhere near this many.
pub const MAX_TOOL_CALLS: usize = 256;

/// One tool call being assembled from streamed deltas.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct StreamedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Merge a streamed `tool_calls` delta array into the per-index accumulators.
/// OpenAI sends `id`/`function.name` once and `function.arguments` as fragments.
/// Some providers (e.g. qwen) re-send the *full* `function.name` on every delta,
/// so the name is assigned (replaced), not appended — otherwise `run_bash` would
/// accumulate into `run_bashrun_bashrun_bash…` and fail tool lookup. Arguments are
/// still genuine fragments and are appended.
pub fn accumulate_tool_call_deltas(tcs: &[Value], calls: &mut Vec<StreamedToolCall>) {
    for tc in tcs {
        let idx = tc.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        if idx >= MAX_TOOL_CALLS {
            continue;
        }
        while calls.len() <= idx {
            calls.push(StreamedToolCall::default());
        }
        let acc = &mut calls[idx];
        if let Some(id) = tc.get("id").and_then(|x| x.as_str())
            && !id.is_empty()
        {
            acc.id = id.to_string();
        }
        if let Some(f) = tc.get("function") {
            if let Some(n) = f.get("name").and_then(|x| x.as_str())
                && !n.is_empty()
            {
                acc.name = n.to_string();
            }
            if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                acc.arguments.push_str(a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bounds_a_huge_index() {
        let mut calls = Vec::new();
        // A bogus huge index must be ignored, not allocated up to.
        accumulate_tool_call_deltas(
            &[json!({"index": 1_000_000_000_u64, "function": {"name": "x"}})],
            &mut calls,
        );
        assert!(
            calls.is_empty(),
            "huge index should be dropped, not allocated"
        );
        // Normal small indices still accumulate.
        accumulate_tool_call_deltas(
            &[
                json!({"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}),
                json!({"index": 1, "id": "b", "function": {"name": "g", "arguments": "[]"}}),
            ],
            &mut calls,
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[1].name, "g");
    }

    /// qwen (and some other OpenAI-compatible providers) re-send the *full*
    /// `function.name` on every delta instead of only the first. The name must be
    /// assigned, not appended — otherwise `run_bash` corrupts into
    /// `run_bashrun_bashrun_bash…` and fails tool lookup with "unknown tool".
    #[test]
    fn handles_repeated_full_name() {
        let mut calls = Vec::new();
        accumulate_tool_call_deltas(
            &[
                json!({"index": 0, "id": "c1", "function": {"name": "run_bash", "arguments": "{\"cmd\":"}}),
            ],
            &mut calls,
        );
        // Subsequent deltas repeat the whole name and carry argument fragments.
        accumulate_tool_call_deltas(
            &[json!({"index": 0, "function": {"name": "run_bash", "arguments": "\"ls\"}"}})],
            &mut calls,
        );
        accumulate_tool_call_deltas(
            &[json!({"index": 0, "function": {"name": "run_bash"}})],
            &mut calls,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run_bash", "name must not duplicate");
        // Genuine argument fragments are still appended into one valid JSON string.
        assert_eq!(calls[0].arguments, "{\"cmd\":\"ls\"}");
    }
}
