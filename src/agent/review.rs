//! The opt-in edit-review gate: with the "review edits" toggle on, an edit-bearing
//! batch pauses to show its diffs before writing. Engine-side stays thin — collect
//! the edit calls into [`ReviewItem`]s and await [`AgentUi::review_edits`]; the TUI
//! computes and renders the diffs. Interactive `aivo code` only.

use serde_json::Value;

/// True when `name` is one of the file-writing tools this gate intercepts;
/// `run_bash` and other side effects are out of scope.
pub fn is_edit_tool(name: &str) -> bool {
    crate::agent::file_tracker::is_write_tool(name)
}

/// One pending edit awaiting review: the raw call (`tool` + `args`) the TUI diffs,
/// `call_index` to map the verdict back onto the batch, and `paths` for the heading.
#[derive(Clone, Debug)]
pub struct ReviewItem {
    pub call_index: usize,
    pub tool: String,
    pub paths: Vec<String>,
    pub args: Value,
}

/// The user's verdict on a reviewed batch: run the edits, or drop them all (the
/// model gets [`REVIEW_REJECTED_DIRECTIVE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReviewDecision {
    ApproveAll,
    Reject,
}

/// The tool result a rejected edit reports back to the model — a directive to stop
/// and ask, not to silently re-apply the same change.
pub const REVIEW_REJECTED_DIRECTIVE: &str = "The user reviewed this edit and chose NOT to apply it \
— nothing was written. Do not silently retry the same change. Stop and ask the user what they'd \
like different before editing this file again.";

/// Best-effort extraction of the file paths a tool call targets, for the review
/// heading. Never touches disk; non-write tools yield nothing.
pub fn edited_paths(name: &str, args: &Value) -> Vec<String> {
    if !is_edit_tool(name) {
        return Vec::new();
    }
    crate::agent::file_tracker::tracked_paths(name, args)
}

/// Build a [`ReviewItem`] for tool call `call_index` (assumed an edit tool).
pub fn review_item(call_index: usize, name: &str, args: &Value) -> ReviewItem {
    ReviewItem {
        call_index,
        tool: name.to_string(),
        paths: edited_paths(name, args),
        args: args.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn edit_tool_membership() {
        assert!(is_edit_tool("write_file"));
        assert!(is_edit_tool("apply_patch"));
        assert!(!is_edit_tool("run_bash"));
        assert!(!is_edit_tool("read_file"));
    }

    #[test]
    fn paths_from_simple_edit_tools() {
        assert_eq!(
            edited_paths("write_file", &json!({"path": "a.txt", "content": "x"})),
            vec!["a.txt".to_string()]
        );
        assert_eq!(
            edited_paths("edit_file", &json!({"path": "src/lib.rs"})),
            vec!["src/lib.rs".to_string()]
        );
        assert!(edited_paths("write_file", &json!({"content": "x"})).is_empty());
    }

    #[test]
    fn review_item_captures_call() {
        let item = review_item(
            3,
            "write_file",
            &json!({"path": "out.txt", "content": "hi"}),
        );
        assert_eq!(item.call_index, 3);
        assert_eq!(item.tool, "write_file");
        assert_eq!(item.paths, vec!["out.txt".to_string()]);
        assert_eq!(item.args["content"], "hi");
    }
}
