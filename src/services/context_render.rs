//! Renders context from a single past session into a per-CLI system-prompt
//! string. With the session-id selector model, injection is always one
//! session at a time — no budget, no multi-thread packing.

use crate::services::ai_launcher::AIToolType;
use crate::services::project_id::Thread;

/// Same ruler as the agent engine's injected block, so the "injecting N tokens"
/// summary matches `/context`.
pub fn estimate_tokens(text: &str) -> usize {
    crate::agent::tokens::estimate_str_tokens(text)
}

/// Output of a render: the rendered string plus a token estimate for the
/// status line shown to users.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedContext {
    pub text: String,
    pub tokens: usize,
}

/// Render exactly one past session for injection into `tool`. Claude gets
/// XML-tagged structure; OpenAI-style tools get compact markdown.
pub fn render_single_session(tool: AIToolType, thread: &Thread) -> RenderedContext {
    let text = match tool {
        AIToolType::Claude => format_claude_single(thread),
        AIToolType::Codex
        | AIToolType::CodexApp
        | AIToolType::Gemini
        | AIToolType::Opencode
        | AIToolType::Pi => format_markdown_single(thread),
    };
    let tokens = estimate_tokens(&text);
    RenderedContext { text, tokens }
}

/// Render one past session for `aivo code -c` as compact markdown, appended
/// to the system prompt rather than shown as a user message.
pub fn render_for_aivo_code(thread: &Thread) -> RenderedContext {
    let text = format_markdown_single(thread);
    let tokens = estimate_tokens(&text);
    RenderedContext { text, tokens }
}

fn format_claude_single(t: &Thread) -> String {
    let mut s = String::from(
        "<aivo_context>\nCross-tool context from one past session. Use as background awareness; the user is continuing prior work, not expecting you to reference this explicitly.\n",
    );
    s.push_str("<session cli=\"");
    s.push_str(&t.cli);
    s.push_str("\" updated_at=\"");
    s.push_str(&t.updated_at.to_rfc3339());
    s.push_str("\">\n  <topic>");
    s.push_str(&escape_xml(&t.topic));
    s.push_str("</topic>\n");
    if !t.last_response.trim().is_empty() {
        s.push_str("  <last_response>");
        s.push_str(&escape_xml(&t.last_response));
        s.push_str("</last_response>\n");
    }
    s.push_str("</session>\n</aivo_context>\n");
    s
}

fn format_markdown_single(t: &Thread) -> String {
    let mut s = String::from(
        "# aivo context\n\nCross-tool context from one past session. Background awareness only.\n\n",
    );
    s.push_str(&format!(
        "**Session:** {} — {}\n",
        t.cli,
        t.updated_at.to_rfc3339()
    ));
    s.push_str(&format!("**Topic:** {}\n", t.topic.trim()));
    if !t.last_response.trim().is_empty() {
        s.push_str(&format!("**Last response:** {}\n", t.last_response.trim()));
    }
    s
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn thread(cli: &str, topic: &str, last: &str) -> Thread {
        Thread {
            cli: cli.into(),
            session_id: format!("sess-{cli}"),
            source_path: "/tmp/x.jsonl".into(),
            topic: topic.into(),
            last_response: last.into(),
            updated_at: Utc::now(),
            cwd: None,
        }
    }

    #[test]
    fn claude_uses_xml_tags() {
        let t = thread(
            "claude",
            "discussing pagination for /users endpoint",
            "suggested cursor-based with id > cursor",
        );
        let r = render_single_session(AIToolType::Claude, &t);
        assert!(r.text.contains("<aivo_context>"));
        assert!(r.text.contains("<session cli=\"claude\""));
        assert!(r.text.contains("<topic>"));
        assert!(r.text.contains("<last_response>"));
        assert!(r.text.trim_end().ends_with("</aivo_context>"));
    }

    #[test]
    fn codex_uses_markdown_headers() {
        let t = thread(
            "codex",
            "refactoring the auth middleware to drop legacy token storage",
            "removed session table writes",
        );
        let r = render_single_session(AIToolType::Codex, &t);
        assert!(r.text.contains("# aivo context"));
        assert!(r.text.contains("**Session:**"));
        assert!(r.text.contains("codex"));
        assert!(r.text.contains("**Topic:**"));
    }

    #[test]
    fn xml_escape_prevents_broken_tags() {
        let t = thread(
            "claude",
            "look at <script>alert(1)</script> in user input handling flow",
            "it's safely escaped downstream",
        );
        let r = render_single_session(AIToolType::Claude, &t);
        assert!(!r.text.contains("<script>"));
        assert!(r.text.contains("&lt;script&gt;"));
    }

    #[test]
    fn aivo_code_render_uses_markdown() {
        let t = thread(
            "codex",
            "wiring the payments webhook retry queue",
            "added exponential backoff",
        );
        let r = render_for_aivo_code(&t);
        assert!(r.text.contains("# aivo context"));
        assert!(r.text.contains("**Topic:**"));
        assert!(r.tokens > 0);
    }

    #[test]
    fn estimate_tokens_matches_rough_ratio() {
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("the quick brown fox jumps"), 5);
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn renders_nonempty_for_real_session() {
        let t = thread("claude", "some substantive topic", "some last response");
        let r = render_single_session(AIToolType::Claude, &t);
        assert!(!r.text.trim().is_empty());
        assert!(r.tokens > 0);
    }
}
