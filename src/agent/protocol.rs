//! Shared data types for the in-process agent: the OpenAI-chat request/response
//! shapes the engine composes, the tool schema/result types, and the permission
//! decision. (Formerly the brain↔client wire protocol; the loop is now in-process
//! — see `engine.rs` — so only the data types remain.)

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// OpenAI chat-completions body the engine composes; the client sends it verbatim
/// through aivo serve (which translates to the upstream protocol).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// Sampling params etc. (temperature, stream, …) passed straight through.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default)]
    pub usage: Option<Value>,
    /// `content` is a kept partial from a mid-stream drop (see
    /// `serve_client::complete`). Never on the wire — a sent message can't be partial.
    #[serde(skip)]
    pub truncated: bool,
    /// Upstream model echoed by the response chunks; never on the wire.
    #[serde(skip)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed JSON object of arguments.
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments (OpenAI function `parameters`).
    pub parameters: Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allow,
    Deny,
    /// Allow this tool kind for the rest of the session.
    AlwaysAllow,
}

/// The user's verdict on an `exit_plan_mode` plan (plan-mode approval card).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PlanDecision {
    Approve,
    KeepPlanning { feedback: Option<String> },
    Discard,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_call_roundtrips() {
        let call = ToolCall {
            id: "call_1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({ "path": "src/main.rs" }),
        };
        let line = serde_json::to_string(&call).unwrap();
        let back: ToolCall = serde_json::from_str(&line).unwrap();
        assert_eq!(back.name, "read_file");
        assert_eq!(back.arguments["path"], "src/main.rs");
    }

    #[test]
    fn decision_is_snake_case() {
        assert_eq!(
            serde_json::to_string(&Decision::AlwaysAllow).unwrap(),
            "\"always_allow\""
        );
    }
}
