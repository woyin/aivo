//! The `ask_user` tool: a structured question with selectable options (the analog
//! of Claude Code's AskUserQuestion). Engine-handled via `AgentUi::ask_user`; the
//! chat TUI renders a card and feeds the chosen answer back as the tool result, so
//! the turn continues without a prose round-trip. Interactive `aivo code` only.

use crate::agent::protocol::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// One selectable answer: a short `label` plus an optional one-line `description`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskOption {
    pub label: String,
    pub description: Option<String>,
}

/// Options past this are dropped — the list stops reading as a quick pick.
const MAX_OPTIONS: usize = 8;

/// The `ask_user` function schema (advertised only in interactive chat).
pub fn ask_user_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "ask_user".to_string(),
        description: "Ask the user a question and let them pick from a short list of options rather \
than answering in prose. Reach for this whenever you would otherwise stop and ask a question with a \
small set of likely answers — a yes/no, a this-or-that, a plan approval, or a choice among a few \
paths. Provide 2–8 concrete `options`; the user picks one (or types their own, unless you set \
`allow_free_text` to false). The chosen answer comes back as the tool result, so do NOT also pose \
the question in your text — just call the tool and wait for the result. Skip it when the answer is \
open-ended (ask in prose instead) or when you could find the answer yourself with your other tools."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask, phrased for a quick decision."
                },
                "options": {
                    "type": "array",
                    "description": "2–8 concrete answers the user can choose from.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "description": "Short answer text the user selects."},
                            "description": {"type": "string", "description": "Optional one-line note about this choice."}
                        },
                        "required": ["label"]
                    }
                },
                "allow_free_text": {
                    "type": "boolean",
                    "description": "Whether the user may type their own answer instead of picking one (default true). Set false only when a listed option is truly required."
                }
            },
            "required": ["question", "options"]
        }),
    }
}

/// Parse into `(question, options, allow_free_text)`. Lenient: an option may be a
/// bare string or a `{label, description}` object; blanks dropped; capped at
/// [`MAX_OPTIONS`]; `allow_free_text` defaults to `true`.
pub fn parse_ask(args: &Value) -> Result<(String, Vec<AskOption>, bool), String> {
    let question = args
        .get("question")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if question.is_empty() {
        return Err("ask_user: missing `question`".to_string());
    }
    let arr = args
        .get("options")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "ask_user: missing `options` array".to_string())?;
    let mut options = Vec::new();
    for entry in arr {
        let (label, description) = match entry {
            Value::String(s) => (s.trim().to_string(), None),
            _ => {
                let label = entry
                    .get("label")
                    .or_else(|| entry.get("value"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let description = entry
                    .get("description")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty());
                (label, description)
            }
        };
        if label.is_empty() {
            continue;
        }
        options.push(AskOption { label, description });
        if options.len() >= MAX_OPTIONS {
            break;
        }
    }
    if options.is_empty() {
        return Err("ask_user: `options` must contain at least one non-empty choice".to_string());
    }
    let allow_free_text = args
        .get("allow_free_text")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    Ok((question, options, allow_free_text))
}

/// The tool result echoing the user's answer back to the model.
pub fn confirmation(answer: &str) -> String {
    format!("The user answered: {answer}")
}

/// The `ask_user` result when the user dismisses the card (Esc) — a directive to
/// stop, not decide for them, so the model doesn't read dismissal as consent.
pub const DISMISSED_DIRECTIVE: &str = "The user dismissed the question without answering — they did NOT \
choose any option. Treat this as a signal to STOP, not as permission to decide for them. Do not pick \
an option, guess, or start building on an assumption. End your turn and let the user tell you how they \
want to proceed.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_question_options_and_default_free_text() {
        let (q, opts, free) = parse_ask(&json!({
            "question": "  Add release notes now?  ",
            "options": [
                {"label": "Yes, I'll write them"},
                {"label": "You add them", "description": "Draft from the diff"},
                "No, auto-generate"
            ]
        }))
        .unwrap();
        assert_eq!(q, "Add release notes now?"); // trimmed
        assert_eq!(opts.len(), 3);
        assert_eq!(opts[1].description.as_deref(), Some("Draft from the diff"));
        assert_eq!(opts[2].label, "No, auto-generate"); // bare string → label
        assert!(free); // defaults on
    }

    #[test]
    fn bare_string_option_becomes_label() {
        let (_q, opts, _) = parse_ask(&json!({
            "question": "Pick one",
            "options": ["a", "b"]
        }))
        .unwrap();
        assert_eq!(opts[0].label, "a");
        assert!(opts[0].description.is_none());
    }

    #[test]
    fn drops_blank_labels_and_caps_at_max() {
        let many: Vec<Value> = (0..20)
            .map(|i| json!({"label": format!("opt {i}")}))
            .collect();
        let (_q, opts, _) = parse_ask(&json!({
            "question": "q",
            "options": many
        }))
        .unwrap();
        assert_eq!(opts.len(), MAX_OPTIONS);

        let (_q, opts, _) = parse_ask(&json!({
            "question": "q",
            "options": [{"label": "  "}, {"label": "real"}]
        }))
        .unwrap();
        assert_eq!(opts.len(), 1);
        assert_eq!(opts[0].label, "real");
    }

    #[test]
    fn allow_free_text_can_be_disabled() {
        let (_q, _opts, free) = parse_ask(&json!({
            "question": "q",
            "options": ["a"],
            "allow_free_text": false
        }))
        .unwrap();
        assert!(!free);
    }

    #[test]
    fn missing_question_or_options_errors() {
        assert!(parse_ask(&json!({"options": ["a"]})).is_err());
        assert!(parse_ask(&json!({"question": "q"})).is_err());
        assert!(parse_ask(&json!({"question": "q", "options": []})).is_err());
        assert!(parse_ask(&json!({"question": "q", "options": [{"label": ""}]})).is_err());
    }
}
