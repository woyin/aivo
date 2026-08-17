//! The `finish_turn` tool: a structured end-of-turn outcome report. Convergence
//! stays hybrid — clean text-only turns still converge — but a premature `done`
//! (unfinished plan, unverified changes) is rejected instead of accepted on faith.

use crate::agent::protocol::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishStatus {
    Done,
    Blocked,
    NeedsUser,
}

impl FinishStatus {
    fn parse(s: &str) -> Option<Self> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace([' ', '-'], "_")
            .as_str()
        {
            "done" | "completed" | "complete" | "finished" => Some(Self::Done),
            "blocked" | "stuck" => Some(Self::Blocked),
            "needs_user" | "need_user" | "ask_user" | "question" => Some(Self::NeedsUser),
            _ => None,
        }
    }

    /// Stable name for machine consumers (`finishStatus` field, `finished` event).
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Blocked => "blocked",
            Self::NeedsUser => "needsUser",
        }
    }
}

/// A verification claim the model attached: a command it ran and what happened.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationClaim {
    pub command: String,
    pub result: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinishReport {
    pub status: FinishStatus,
    pub summary: String,
    pub verification: Vec<VerificationClaim>,
    pub remaining: Vec<String>,
    pub blocker: Option<String>,
}

pub fn finish_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "finish_turn".to_string(),
        description: "End your turn with an explicit outcome report. Call this ALONE (no other \
tool calls in the same step) as your final action on non-trivial tasks. Use status `done` only \
when the task is genuinely complete — a premature done (unfinished plan steps, failing checks) \
will be rejected. Use `blocked` when you cannot proceed and `needs_user` when only the user can \
decide; both require a concrete `blocker`. List commands you ran as `verification` evidence and \
any work left undone in `remaining`. Trivial pure-answer turns may simply reply in text instead."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["done", "blocked", "needs_user"]},
                "summary": {"type": "string", "description": "What was accomplished or found."},
                "verification": {
                    "type": "array",
                    "description": "Commands run as evidence, with their outcomes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "result": {"type": "string"}
                        },
                        "required": ["command", "result"]
                    }
                },
                "remaining": {
                    "type": "array",
                    "description": "Work intentionally left undone.",
                    "items": {"type": "string"}
                },
                "blocker": {
                    "type": "string",
                    "description": "Required for blocked/needs_user: the concrete blocker or question."
                }
            },
            "required": ["status", "summary"]
        }),
    }
}

/// Parse `finish_turn` arguments. Strict where honesty depends on it (status,
/// non-empty summary, blocker for blocked/needs_user), lenient elsewhere.
pub fn parse_finish(args: &Value) -> Result<FinishReport, String> {
    let status = args
        .get("status")
        .and_then(Value::as_str)
        .and_then(FinishStatus::parse)
        .ok_or("finish_turn: `status` must be one of done, blocked, needs_user")?;
    let summary = args
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    if summary.is_empty() {
        return Err("finish_turn: `summary` is required".to_string());
    }
    let blocker = args
        .get("blocker")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    if status != FinishStatus::Done && blocker.is_none() {
        return Err(format!(
            "finish_turn: status `{}` requires a concrete `blocker`",
            args["status"].as_str().unwrap_or("")
        ));
    }
    let verification = args
        .get("verification")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let command = v.get("command").and_then(Value::as_str)?.trim();
                    let result = v.get("result").and_then(Value::as_str).unwrap_or("").trim();
                    (!command.is_empty()).then(|| VerificationClaim {
                        command: command.to_string(),
                        result: result.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let remaining = args
        .get("remaining")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    Ok(FinishReport {
        status,
        summary,
        verification,
        remaining,
        blocker,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_done_with_evidence() {
        let r = parse_finish(&json!({
            "status": "done",
            "summary": "fixed the bug",
            "verification": [{"command": "cargo test", "result": "passed"}],
            "remaining": ["docs update"]
        }))
        .unwrap();
        assert_eq!(r.status, FinishStatus::Done);
        assert_eq!(r.verification.len(), 1);
        assert_eq!(r.remaining, vec!["docs update"]);
        assert!(r.blocker.is_none());
    }

    #[test]
    fn lenient_status_spellings() {
        for (s, want) in [
            ("completed", FinishStatus::Done),
            ("stuck", FinishStatus::Blocked),
            ("need-user", FinishStatus::NeedsUser),
        ] {
            let r = parse_finish(&json!({"status": s, "summary": "x", "blocker": "y"})).unwrap();
            assert_eq!(r.status, want);
        }
    }

    #[test]
    fn rejects_missing_pieces() {
        assert!(parse_finish(&json!({"summary": "x"})).is_err()); // no status
        assert!(parse_finish(&json!({"status": "done"})).is_err()); // no summary
        assert!(parse_finish(&json!({"status": "???", "summary": "x"})).is_err());
        // blocked/needs_user without a blocker
        assert!(parse_finish(&json!({"status": "blocked", "summary": "x"})).is_err());
        assert!(
            parse_finish(&json!({"status": "needs_user", "summary": "x", "blocker": " "})).is_err()
        );
    }

    #[test]
    fn malformed_lists_degrade_instead_of_failing() {
        let r = parse_finish(&json!({
            "status": "done",
            "summary": "x",
            "verification": [{"command": "  "}, {"result": "orphan"}, "junk"],
            "remaining": [1, "", "real"]
        }))
        .unwrap();
        assert!(r.verification.is_empty());
        assert_eq!(r.remaining, vec!["real"]);
    }
}
