//! Message and request shaping for the agent engine: converting tool specs and
//! assistant replies to OpenAI chat-wire JSON, and rendering the transcript to
//! plain text for the summarizer. Pure functions over serde_json values.

use crate::agent::protocol::{AssistantMessage, ToolSpec};
use crate::agent::tokens::estimate_str_tokens;
use serde_json::{Map, Value, json};

pub(crate) fn tool_to_openai(t: ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {"name": t.name, "description": t.description, "parameters": t.parameters}
    })
}

/// Convert an assistant reply to an OpenAI chat message for the history (`arguments`
/// as a string, `content` present when there are no tool calls).
pub(crate) fn assistant_to_openai(m: &AssistantMessage) -> Value {
    let mut msg = Map::new();
    msg.insert("role".into(), json!("assistant"));
    if let Some(c) = &m.content
        && !c.is_empty()
    {
        msg.insert("content".into(), json!(c));
    }
    if !m.tool_calls.is_empty() {
        let calls: Vec<Value> = m
            .tool_calls
            .iter()
            .map(|t| {
                json!({
                    "id": t.id,
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "arguments": serde_json::to_string(&t.arguments).unwrap_or_else(|_| "{}".into()),
                    }
                })
            })
            .collect();
        msg.insert("tool_calls".into(), json!(calls));
    } else if !msg.contains_key("content") {
        msg.insert("content".into(), json!(""));
    }
    Value::Object(msg)
}

pub(crate) fn role(m: &Value) -> &str {
    m.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

pub(crate) fn content_str(m: &Value) -> String {
    m.get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string()
}

/// Multimodal-aware [`content_str`]: image-bearing turns would otherwise
/// serialize empty.
pub(crate) fn content_text(m: &Value) -> String {
    match m.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(|t| t.as_str()) {
                Some("text") => p.get("text").and_then(|t| t.as_str()).unwrap_or(""),
                Some("image_url") => "[image]",
                _ => "",
            })
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub(crate) fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max).collect();
    format!("{kept}… (+{} chars)", s.chars().count() - max)
}

/// One entry per message; assistant text + tool calls stay one entry so elision
/// can't split them.
fn transcript_parts(messages: &[Value]) -> Vec<String> {
    let mut out = Vec::new();
    for m in messages {
        match role(m) {
            "user" => out.push(format!("[User]: {}\n", content_text(m))),
            "assistant" => {
                let mut entry = String::new();
                let c = content_str(m);
                if !c.is_empty() {
                    entry.push_str(&format!("[Assistant]: {c}\n"));
                }
                if let Some(calls) = m.get("tool_calls").and_then(|t| t.as_array()) {
                    let rendered: Vec<String> = calls
                        .iter()
                        .filter_map(|tc| {
                            let f = tc.get("function")?;
                            let name = f.get("name")?.as_str()?;
                            let args = f.get("arguments").and_then(|a| a.as_str()).unwrap_or("");
                            Some(format!("{name}({})", truncate_str(args, 200)))
                        })
                        .collect();
                    if !rendered.is_empty() {
                        entry.push_str(&format!("[Tool calls]: {}\n", rendered.join("; ")));
                    }
                }
                if !entry.is_empty() {
                    out.push(entry);
                }
            }
            "tool" => out.push(format!(
                "[Tool result]: {}\n",
                truncate_str(&content_str(m), 2000)
            )),
            _ => {}
        }
    }
    out
}

/// Test-only on purpose: production goes through [`serialize_transcript_bounded`].
#[cfg(test)]
pub(crate) fn serialize_transcript(messages: &[Value]) -> String {
    transcript_parts(messages).concat()
}

pub(crate) const TRANSCRIPT_OMITTED_MARKER: &str =
    "[... middle of the conversation omitted — too long to summarize in full ...]";

/// Transcript capped at `max_tokens` (estimate ruler): whole oldest entries fill a
/// third, newest the rest, middle elided — the summary request must never overflow.
pub(crate) fn serialize_transcript_bounded(messages: &[Value], max_tokens: usize) -> String {
    let parts = transcript_parts(messages);
    let costs: Vec<usize> = parts.iter().map(|p| estimate_str_tokens(p)).collect();
    if costs.iter().sum::<usize>() <= max_tokens {
        return parts.concat();
    }
    let content_budget =
        max_tokens.saturating_sub(estimate_str_tokens(TRANSCRIPT_OMITTED_MARKER) + 1);
    let head_budget = content_budget / 3;
    let mut used = 0;
    let mut head_end = 0;
    while head_end < parts.len() && used + costs[head_end] <= head_budget {
        used += costs[head_end];
        head_end += 1;
    }
    let mut tail_start = parts.len();
    while tail_start > head_end && used + costs[tail_start - 1] <= content_budget {
        used += costs[tail_start - 1];
        tail_start -= 1;
    }
    if head_end == 0 && tail_start == parts.len() {
        // One entry dwarfs the budget: cut the newest by chars (≤ 1 token/char).
        let newest = parts.last().map(String::as_str).unwrap_or("");
        let mut kept = truncate_str(newest, content_budget.saturating_mul(3));
        if estimate_str_tokens(&kept) > content_budget {
            kept = truncate_str(newest, content_budget);
        }
        return format!("{TRANSCRIPT_OMITTED_MARKER}\n{kept}");
    }
    format!(
        "{}{TRANSCRIPT_OMITTED_MARKER}\n{}",
        parts[..head_end].concat(),
        parts[tail_start..].concat()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_transcript_renders_roles() {
        let messages = vec![
            json!({"role":"user","content":"do X"}),
            json!({"role":"assistant","content":"","tool_calls":[
                {"function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}
            ]}),
            json!({"role":"tool","content":"file contents"}),
        ];
        let t = serialize_transcript(&messages);
        assert!(t.contains("[User]: do X"));
        assert!(t.contains("[Tool calls]: read_file("));
        assert!(t.contains("[Tool result]: file contents"));
    }

    #[test]
    fn truncate_str_marks_overflow() {
        assert_eq!(truncate_str("abc", 5), "abc");
        let out = truncate_str("abcdefgh", 3);
        assert!(out.starts_with("abc…") && out.contains("+5 chars"));
    }

    #[test]
    fn content_text_flattens_multimodal() {
        let m = json!({"role":"user","content":[
            {"type":"text","text":"look at this"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,x"}},
        ]});
        assert_eq!(content_text(&m), "look at this\n[image]");
        assert_eq!(
            content_text(&json!({"role":"user","content":"plain"})),
            "plain"
        );
        assert_eq!(content_text(&json!({"role":"user"})), "");
    }

    #[test]
    fn serialize_transcript_keeps_multimodal_user_turns() {
        let messages = vec![json!({"role":"user","content":[
            {"type":"text","text":"what is in the screenshot?"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,x"}},
        ]})];
        let t = serialize_transcript(&messages);
        assert!(t.contains("[User]: what is in the screenshot?"), "{t}");
        assert!(t.contains("[image]"), "{t}");
    }

    #[test]
    fn bounded_transcript_is_verbatim_under_budget() {
        let messages = vec![
            json!({"role":"user","content":"do X"}),
            json!({"role":"assistant","content":"done"}),
        ];
        assert_eq!(
            serialize_transcript_bounded(&messages, usize::MAX),
            serialize_transcript(&messages)
        );
    }

    #[test]
    fn bounded_transcript_elides_middle_keeps_both_ends() {
        let mut messages = vec![json!({"role":"user","content":"the original goal statement"})];
        for i in 0..200 {
            messages.push(json!({"role":"assistant","content":format!("middle step number {i} with some filler words to carry weight")}));
        }
        messages.push(json!({"role":"user","content":"the newest request"}));
        let budget = 300;
        let t = serialize_transcript_bounded(&messages, budget);
        assert!(t.contains("the original goal statement"), "{t}");
        assert!(t.contains("the newest request"), "{t}");
        assert!(t.contains(TRANSCRIPT_OMITTED_MARKER), "{t}");
        assert!(!t.contains("middle step number 100"), "{t}");
        assert!(
            estimate_str_tokens(&t) <= budget,
            "over budget: {} > {budget}",
            estimate_str_tokens(&t)
        );
    }

    // CJK exercises the tight re-cut (≈1 token/char beats the ~4 guess)
    #[test]
    fn bounded_transcript_cuts_a_single_oversized_entry() {
        let messages = vec![json!({"role":"user","content":"码".repeat(50_000)})];
        let budget = 1_000;
        let t = serialize_transcript_bounded(&messages, budget);
        assert!(t.contains(TRANSCRIPT_OMITTED_MARKER), "{t}");
        assert!(t.contains('码'), "{t}");
        let est = estimate_str_tokens(&t);
        assert!(est <= budget + 10, "over budget: {est} > {budget}");
    }
}
