//! Persistent plan mode (the analog of Claude Code's plan mode): the engine goes
//! read-only, the model investigates and calls `exit_plan_mode` with its plan, and
//! the chat TUI shows an approval card. Approve restores full tools mid-turn so the
//! same turn continues into execution. Interactive `aivo code` only.

use crate::agent::protocol::ToolSpec;
use serde_json::{Value, json};

pub fn exit_plan_mode_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "exit_plan_mode".to_string(),
        description:
            "Call this when your implementation plan is ready for the user's review. Pass \
the COMPLETE plan in `plan` as markdown: the approach, the specific files and functions to change, \
and a numbered list of steps. The user will approve it, ask you to keep planning (their feedback \
comes back as the tool result), or discard it. Do not start implementing before this returns \
approval, and do not also paste the plan as message text — the card shows it."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "The full implementation plan, in markdown."
                }
            },
            "required": ["plan"]
        }),
    }
}

pub fn parse_exit_plan(args: &Value) -> Result<String, String> {
    let plan = args
        .get("plan")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if plan.is_empty() {
        return Err(
            "exit_plan_mode: missing `plan` — write the full implementation plan in \
markdown and call it again."
                .to_string(),
        );
    }
    Ok(plan)
}

/// A fixed constant so `set_plan_mode(false)` can strip it by exact substring.
pub const PLAN_MODE_DIRECTIVE: &str = "PLAN MODE is on. It persists across user messages until the \
user approves a plan or turns it off — treat their follow-ups as plan revisions, not permission to \
build. Investigate the codebase with read-only tools; file-mutating tools are unavailable and will \
be refused. You MAY call run_bash: recognized read-only inspection commands (git diff/log/status, \
ls, rg, grep, cat, find, …) run without confirmation, while anything that could change state — \
builds, installs, scripts — needs the user's explicit approval each time, so prefer plain \
inspection. When your plan is ready, call `exit_plan_mode` with the complete plan in markdown — \
use it instead of `ask_user` for plan approval. Do not implement anything until it returns \
approval.";

/// Rides every plan-mode request's latest user message (ephemeral, never
/// persisted): the system-prompt directive decays over a long conversation, and
/// a user "go ahead" otherwise tempts the model into executing while read-only.
pub const PLAN_TURN_REMINDER: &str = "<system-reminder>Plan mode is still active — the session is \
read-only. Investigate and refine the plan only; do not execute the task or mutate state (files, \
configs, deployments), not even via run_bash. If the user asks you to start or execute, call \
`exit_plan_mode` with the complete plan first and wait for approval.</system-reminder>";

/// Append [`PLAN_TURN_REMINDER`] to outgoing user content (text or multimodal).
pub fn append_turn_reminder(content: Value) -> Value {
    match content {
        Value::String(s) => Value::String(format!("{s}\n\n{PLAN_TURN_REMINDER}")),
        Value::Array(mut parts) => {
            parts.push(json!({"type": "text", "text": PLAN_TURN_REMINDER}));
            Value::Array(parts)
        }
        other => other,
    }
}

pub const PLAN_APPROVED_RESULT: &str = "Plan approved — implement it now. Plan mode is off and \
your full tools are restored; make the edits, run the commands, and verify as you go.";

pub fn keep_planning_result(feedback: Option<&str>) -> String {
    match feedback.map(str::trim).filter(|f| !f.is_empty()) {
        Some(f) => format!(
            "The user wants to keep planning — plan mode stays on. Their feedback:\n{f}\n\nRevise \
the plan accordingly and call `exit_plan_mode` again when it's ready."
        ),
        None => "The user wants to keep planning — plan mode stays on. Revise the plan from the \
conversation so far and call `exit_plan_mode` again when it's ready."
            .to_string(),
    }
}

pub const PLAN_DISCARDED_RESULT: &str = "The user discarded the plan and cancelled planning. \
STOP: do not implement anything. End your turn; the user will say how to proceed.";

pub const PLAN_APPROVAL_DISMISSED: &str = "The user dismissed the plan without deciding — they \
did NOT approve it. Plan mode stays on. End your turn; the user will approve, give feedback, or \
stop planning.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_requires_nonempty_plan() {
        assert!(parse_exit_plan(&json!({})).is_err());
        assert!(parse_exit_plan(&json!({"plan": "   "})).is_err());
        assert_eq!(
            parse_exit_plan(&json!({"plan": " 1. do X "})).unwrap(),
            "1. do X"
        );
    }

    #[test]
    fn keep_planning_result_embeds_feedback() {
        assert!(
            keep_planning_result(Some("use the retry helper")).contains("use the retry helper")
        );
        assert!(!keep_planning_result(Some("  ")).contains("Their feedback"));
        assert!(keep_planning_result(None).contains("keep planning"));
    }
}
