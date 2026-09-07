//! The `update_plan` tool: a lightweight, model-maintained task list (codex's
//! `update_plan` / Claude Code's `TodoWrite`). The model sends the FULL plan on
//! every call, so the engine holds no durable state — it just parses the list,
//! forwards it to the UI for a plan card, and echoes a confirmation back so the
//! model sees the plan it just set. Visible in the chat transcript as a checklist.

use crate::agent::protocol::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Pending,
    InProgress,
    Completed,
}

impl PlanStatus {
    /// Lenient parse: accepts the canonical names plus common synonyms and
    /// space/hyphen spellings (`in progress`, `in-progress`, `done`, `todo`).
    fn parse(s: &str) -> Option<Self> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '-'], "_")
            .as_str()
        {
            "pending" | "todo" | "not_started" => Some(Self::Pending),
            "in_progress" | "active" | "doing" | "current" => Some(Self::InProgress),
            "completed" | "complete" | "done" | "finished" => Some(Self::Completed),
            _ => None,
        }
    }

    /// ASCII checkbox for the model-facing confirmation text.
    fn checkbox(self) -> &'static str {
        match self {
            Self::Pending => "[ ]",
            Self::InProgress => "[~]",
            Self::Completed => "[x]",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanItem {
    pub step: String,
    pub status: PlanStatus,
}

/// The `update_plan` function schema, offered on every turn (the system prompt
/// tells the model when to use it).
pub fn plan_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "update_plan".to_string(),
        description: "Record or update a step-by-step plan for a multi-step task. Always send the \
COMPLETE ordered list (not a delta). Mark the step you're working on `in_progress` and flip each \
step to `completed` the moment its work is actually done (not merely intended). Call this when you \
begin a multi-step task and again \
every time a step's status changes — and ALWAYS send a final call marking every step `completed` \
once the task is done, so the plan doesn't linger as unfinished. Skip it for trivial one-step work. \
Steps are work you do this turn: a decision only the user can make is an `ask_user` call inside a \
step, never a step that waits for approval."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "description": "The complete ordered list of steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string", "description": "Short imperative description of the step."},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}
                        },
                        "required": ["step", "status"]
                    }
                }
            },
            "required": ["plan"]
        }),
    }
}

/// Parse the `plan` argument into items. Lenient: blank steps are dropped and an
/// unrecognized status falls back to `pending`, so a near-miss doesn't fail the
/// whole call (the model would just burn a step recovering).
pub fn parse_plan(args: &Value) -> Result<Vec<PlanItem>, String> {
    let arr = args
        .get("plan")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "update_plan: missing `plan` array".to_string())?;
    let mut items = Vec::with_capacity(arr.len());
    for entry in arr {
        // Accept `step` (our schema) or `content` (TodoWrite spelling).
        let step = entry
            .get("step")
            .or_else(|| entry.get("content"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if step.is_empty() {
            continue;
        }
        let status = entry
            .get("status")
            .and_then(|v| v.as_str())
            .and_then(PlanStatus::parse)
            .unwrap_or(PlanStatus::Pending);
        items.push(PlanItem { step, status });
    }
    if items.is_empty() {
        return Err("update_plan: `plan` must contain at least one step".to_string());
    }
    Ok(items)
}

/// Parse one [`pinned_block`] checklist line back into an item; `None` otherwise.
pub fn parse_checkbox_line(line: &str) -> Option<PlanItem> {
    let (status, step) = match line.split_once(' ') {
        Some(("[ ]", rest)) => (PlanStatus::Pending, rest),
        Some(("[~]", rest)) => (PlanStatus::InProgress, rest),
        Some(("[x]", rest)) => (PlanStatus::Completed, rest),
        _ => return None,
    };
    let step = step.trim();
    (!step.is_empty()).then(|| PlanItem {
        step: step.to_string(),
        status,
    })
}

/// Whether the plan has begun executing — any step past `pending`.
pub fn started(items: &[PlanItem]) -> bool {
    items.iter().any(|i| i.status != PlanStatus::Pending)
}

/// Normalize a model-sent plan so progress reads monotonically: every step before
/// the furthest-reached one (the `in_progress` step, or the last `completed`) is
/// forced to `completed`, leaving at most one `in_progress`. Models routinely
/// advance the active step without flipping the ones they passed, so the engine —
/// which owns plan state — fills those in deterministically instead of trusting
/// the model to keep every status honest. Returns whether anything changed.
pub fn normalize_progress(items: &mut [PlanItem]) -> bool {
    // The furthest step reached so far: the last one that isn't still pending.
    let Some(frontier) = items.iter().rposition(|i| i.status != PlanStatus::Pending) else {
        return false; // a fresh, all-pending plan — nothing started yet
    };
    let mut changed = false;
    for item in items.iter_mut().take(frontier) {
        if item.status != PlanStatus::Completed {
            item.status = PlanStatus::Completed;
            changed = true;
        }
    }
    changed
}

/// Render the plan as a bare checklist for the compaction "pinned working set"
/// (no count header — that's `confirmation`'s job). Empty slice → empty string
/// so the caller can omit the section entirely.
pub fn pinned_block(items: &[PlanItem]) -> String {
    let mut out = String::new();
    for item in items {
        out.push_str(&format!("{} {}\n", item.status.checkbox(), item.step));
    }
    out.trim_end().to_string()
}

/// Every session whose plan had an open "approved"/"for approval" step ended in a
/// proposal hand-back unless the model asked via `ask_user`.
pub const APPROVAL_GATE_NOTE: &str = "A step that waits for the user's approval must not end the \
turn: get the decision with ask_user (e.g. Approve / Adjust) and carry on in this turn — don't hand \
back a written proposal.";

fn awaits_approval(items: &[PlanItem]) -> bool {
    const CUES: [&str; 6] = [
        "approv",
        "sign-off",
        "sign off",
        "go-ahead",
        "go ahead",
        "confirm with",
    ];
    items
        .iter()
        .filter(|i| i.status != PlanStatus::Completed)
        .any(|i| {
            let step = i.step.to_ascii_lowercase();
            CUES.iter().any(|c| step.contains(c))
        })
}

/// The confirmation echoed back to the model as the tool result, so it knows the
/// plan it just set (and how many steps remain).
pub fn confirmation(items: &[PlanItem]) -> String {
    let done = items
        .iter()
        .filter(|i| i.status == PlanStatus::Completed)
        .count();
    let mut out = format!("Plan updated ({done}/{} done):\n", items.len());
    for item in items {
        out.push_str(&format!("{} {}\n", item.status.checkbox(), item.step));
    }
    if awaits_approval(items) {
        out.push_str(APPROVAL_GATE_NOTE);
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_steps_and_lenient_status() {
        let items = parse_plan(&json!({"plan": [
            {"step": "scan code", "status": "completed"},
            {"step": "write fix", "status": "in progress"},
            {"content": "run tests", "status": "todo"},
            {"step": "", "status": "pending"},
            {"step": "ship", "status": "???"}
        ]}))
        .unwrap();
        assert_eq!(items.len(), 4); // blank step dropped
        assert_eq!(items[0].status, PlanStatus::Completed);
        assert_eq!(items[1].status, PlanStatus::InProgress); // "in progress" → in_progress
        assert_eq!(items[2].step, "run tests"); // `content` alias
        assert_eq!(items[3].status, PlanStatus::Pending); // unknown → pending
    }

    #[test]
    fn empty_or_missing_plan_errors() {
        assert!(parse_plan(&json!({})).is_err());
        assert!(parse_plan(&json!({"plan": []})).is_err());
        assert!(parse_plan(&json!({"plan": [{"step": "  ", "status": "pending"}]})).is_err());
    }

    #[test]
    fn pinned_block_renders_checkboxes_and_is_empty_when_no_items() {
        assert_eq!(pinned_block(&[]), "");
        let items = parse_plan(&json!({"plan": [
            {"step": "a", "status": "completed"},
            {"step": "b", "status": "in_progress"}
        ]}))
        .unwrap();
        let block = pinned_block(&items);
        assert_eq!(block, "[x] a\n[~] b");
        assert!(!block.contains("done")); // no count header (unlike confirmation)
    }

    fn plan(statuses: &[&str]) -> Vec<PlanItem> {
        statuses
            .iter()
            .enumerate()
            .map(|(i, s)| PlanItem {
                step: format!("step {i}"),
                status: PlanStatus::parse(s).unwrap(),
            })
            .collect()
    }

    fn statuses(items: &[PlanItem]) -> Vec<PlanStatus> {
        items.iter().map(|i| i.status).collect()
    }

    #[test]
    fn normalize_fills_forward_past_steps_and_keeps_one_in_progress() {
        use PlanStatus::*;
        // The model jumped to step 3 without flipping the ones it passed.
        let mut p = plan(&["pending", "pending", "in_progress", "pending"]);
        assert!(normalize_progress(&mut p));
        assert_eq!(statuses(&p), [Completed, Completed, InProgress, Pending]);

        // A later `completed` pulls an earlier stray `in_progress` to completed too.
        let mut p = plan(&["in_progress", "completed"]);
        assert!(normalize_progress(&mut p));
        assert_eq!(statuses(&p), [Completed, Completed]);

        // A fresh, all-pending plan is left untouched (nothing started).
        let mut p = plan(&["pending", "pending"]);
        assert!(!normalize_progress(&mut p));
        assert_eq!(statuses(&p), [Pending, Pending]);

        // Just-started: step 1 in_progress, rest pending — already monotone.
        let mut p = plan(&["in_progress", "pending"]);
        assert!(!normalize_progress(&mut p));
        assert_eq!(statuses(&p), [InProgress, Pending]);
    }

    #[test]
    fn started_detects_any_non_pending_step() {
        assert!(!started(&plan(&["pending", "pending"])));
        assert!(started(&plan(&["pending", "in_progress"])));
        assert!(started(&plan(&["completed"])));
    }

    #[test]
    fn open_approval_step_carries_the_note_until_done() {
        let gated = parse_plan(&json!({"plan": [
            {"step": "Propose a redesign for approval", "status": "in_progress"},
            {"step": "Implement the approved changes", "status": "pending"}
        ]}))
        .unwrap();
        assert!(confirmation(&gated).ends_with(APPROVAL_GATE_NOTE));

        let done = parse_plan(&json!({"plan": [
            {"step": "Implement the approved changes", "status": "completed"}
        ]}))
        .unwrap();
        assert!(!confirmation(&done).contains(APPROVAL_GATE_NOTE));

        let plain =
            parse_plan(&json!({"plan": [{"step": "Run tests", "status": "pending"}]})).unwrap();
        assert!(!confirmation(&plain).contains(APPROVAL_GATE_NOTE));
    }

    #[test]
    fn confirmation_counts_and_checkboxes() {
        let items = parse_plan(&json!({"plan": [
            {"step": "a", "status": "completed"},
            {"step": "b", "status": "in_progress"}
        ]}))
        .unwrap();
        let c = confirmation(&items);
        assert!(c.contains("1/2 done"));
        assert!(c.contains("[x] a"));
        assert!(c.contains("[~] b"));
    }
}
