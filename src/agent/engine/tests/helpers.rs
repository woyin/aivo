//! Shared fixtures: mock UI, SSE mock server, request builders.

use super::super::*;
use crate::agent::plan::PlanStatus;
use serde_json::json;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

#[derive(Default)]
pub(super) struct CapturingUi {
    pub(super) tools: Vec<String>,
    /// Tool names whose result came back `Err` (failure rendering), in order.
    pub(super) tool_errors: Vec<String>,
    pub(super) text: String,
    pub(super) notices: Vec<String>,
    /// `notify_error` notices, separate so tests can assert the channel.
    pub(super) errors: Vec<String>,
    pub(super) plans: Vec<usize>,
    /// Statuses from the most recent `plan_updated` (to assert finalization).
    pub(super) last_plan: Vec<PlanStatus>,
    pub(super) footer_tokens: u64,
    pub(super) deny: bool,
    /// Reply `AlwaysAllow` instead of `Allow`/`Deny` (takes precedence).
    pub(super) always_allow: bool,
    pub(super) asks: usize,
    /// The `tool` argument of each `ask_permission` call, in order.
    pub(super) ask_tools: Vec<String>,
    pub(super) turn_token_reports: Vec<u64>,
    pub(super) discards: usize,
    /// Each forwarded sub-agent step: `(agent, tool, step)`.
    pub(super) sub_activity: Vec<(String, String, usize)>,
    /// `ask_user` replies with the dismissal directive (Esc) instead of an answer.
    pub(super) dismiss_asks: bool,
    /// The `question` of each `ask_user` call, in order.
    pub(super) ask_user_questions: Vec<String>,
    /// Verdict `approve_plan` replies with (`None` → dismissed).
    pub(super) plan_decision: Option<crate::agent::protocol::PlanDecision>,
    /// The plan text of each `approve_plan` call, in order.
    pub(super) approved_plans: Vec<String>,
    pub(super) steering: Vec<String>,
    /// Every `verify_evidence` record, in arrival order (not merged).
    pub(super) verify_records: Vec<crate::agent::verify::EvidenceRecord>,
    /// Every typed guard stop, in order.
    pub(super) stops: Vec<TurnStop>,
    /// `session_message` transcript forms (✉ rows), in order.
    pub(super) mail_rows: Vec<String>,
}

impl AgentUi for CapturingUi {
    fn drain_steering(&mut self) -> Vec<String> {
        std::mem::take(&mut self.steering)
    }
    fn assistant_text(&mut self, t: &str) {
        self.text.push_str(t);
    }
    fn discard_streamed_segment(&mut self) {
        self.discards += 1;
        self.text.clear();
    }
    fn plan_updated(&mut self, items: &[PlanItem]) {
        self.plans.push(items.len());
        self.last_plan = items.iter().map(|i| i.status).collect();
    }
    fn tool_start(&mut self, name: &str, _: &Value) {
        self.tools.push(name.to_string());
    }
    fn tool_result(&mut self, name: &str, r: &Result<String, String>) {
        if r.is_err() {
            self.tool_errors.push(name.to_string());
        }
    }
    fn notify(&mut self, t: &str) {
        self.notices.push(t.to_string());
    }
    fn verify_evidence(&mut self, record: &crate::agent::verify::EvidenceRecord) {
        self.verify_records.push(record.clone());
    }
    fn session_message(&mut self, msg: &crate::services::session_mail::Message) {
        self.mail_rows.push(msg.transcript_display());
    }
    fn notify_error(&mut self, t: &str) {
        self.errors.push(t.to_string());
    }
    fn turn_stopped(&mut self, stop: TurnStop) {
        self.stops.push(stop);
    }
    fn footer(&mut self, _: Option<&str>, _: usize, tokens: u64, _: u64, _: u64) {
        self.footer_tokens = tokens;
    }
    fn turn_tokens(&mut self, output: u64) {
        self.turn_token_reports.push(output);
    }
    fn subagent_activity(&mut self, agent: &str, tool: &str, _: &Value, step: usize) {
        self.sub_activity
            .push((agent.to_string(), tool.to_string(), step));
    }
    fn ask_permission<'a>(
        &'a mut self,
        tool: &'a str,
        _: Option<&'a str>,
        _: bool,
    ) -> BoxFuture<'a, Decision> {
        self.asks += 1;
        self.ask_tools.push(tool.to_string());
        let (always_allow, deny) = (self.always_allow, self.deny);
        Box::pin(async move {
            if always_allow {
                Decision::AlwaysAllow
            } else if deny {
                Decision::Deny
            } else {
                Decision::Allow
            }
        })
    }
    fn ask_user<'a>(
        &'a mut self,
        question: &'a str,
        options: &'a [crate::agent::ask::AskOption],
        _: bool,
        _: bool,
    ) -> BoxFuture<'a, Result<String, String>> {
        self.ask_user_questions.push(question.to_string());
        let dismiss = self.dismiss_asks;
        let answer = options.first().map(|o| o.label.clone()).unwrap_or_default();
        Box::pin(async move {
            if dismiss {
                Err(crate::agent::ask::DISMISSED_DIRECTIVE.to_string())
            } else {
                Ok(answer)
            }
        })
    }
    fn approve_plan<'a>(
        &'a mut self,
        plan: &'a str,
    ) -> BoxFuture<'a, Result<PlanDecision, String>> {
        self.approved_plans.push(plan.to_string());
        let decision = self.plan_decision.clone();
        Box::pin(
            async move { decision.ok_or_else(|| plan_mode::PLAN_APPROVAL_DISMISSED.to_string()) },
        )
    }
}

pub(super) fn tmp() -> PathBuf {
    crate::test_sandbox::tmp("aivo-engine")
}

/// No two adjacent `user` messages (the sequence the Anthropic bridge 400s on).
pub(super) fn assert_no_consecutive_user(messages: &[Value]) {
    for w in messages.windows(2) {
        assert!(
            !(role(&w[0]) == "user" && role(&w[1]) == "user"),
            "two consecutive user messages: {w:?}"
        );
    }
}

/// Build a one-tool-call SSE body for the fake serve.
pub(super) fn tool_call_sse(name: &str, args: Value) -> String {
    let delta = json!({"choices":[{"delta":{"tool_calls":[{
        "index": 0, "id": "c1",
        "function": {"name": name, "arguments": args.to_string()}
    }]}}]});
    format!("data: {delta}\n\ndata: [DONE]\n\n")
}

/// Build a single SSE body carrying a whole batch of tool calls (one assistant
/// turn), each `(id, name, args)` placed at its own `index`.
pub(super) fn batch_tool_call_sse(calls: &[(&str, &str, Value)]) -> String {
    let entries: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(i, (id, name, args))| {
            json!({
                "index": i, "id": id,
                "function": {"name": name, "arguments": args.to_string()}
            })
        })
        .collect();
    let delta = json!({"choices":[{"delta":{"tool_calls": entries}}]});
    format!("data: {delta}\n\ndata: [DONE]\n\n")
}

/// Drain the full request before replying: closing with unread bytes RSTs the
/// response (Windows), and the engine's retry then eats the next scripted body.
pub(super) fn drain_request(sock: &mut std::net::TcpStream) {
    let mut data = Vec::new();
    let mut buf = [0u8; 16384];
    loop {
        let n = match sock.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        data.extend_from_slice(&buf[..n]);
        let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let body_len: usize = String::from_utf8_lossy(&data[..pos])
            .lines()
            .find_map(|l| {
                l.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .and_then(|v| v.trim().parse().ok())
            })
            .unwrap_or(0);
        if data.len() >= pos + 4 + body_len {
            return;
        }
    }
}

pub(super) fn spawn_sse_sequence(bodies: Vec<String>) -> u16 {
    spawn_sse_sequence_cut(bodies.into_iter().map(|b| (b, false)).collect())
}

/// Like [`spawn_sse_sequence`], but a body flagged `true` over-claims its
/// Content-Length then closes — a mid-stream drop (cf. `serve_client`'s
/// `keeps_partial_text_on_midstream_error` mock).
pub(super) fn spawn_sse_sequence_cut(bodies: Vec<(String, bool)>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for (body, cut) in bodies {
            let Ok((mut sock, _)) = listener.accept() else {
                break;
            };
            drain_request(&mut sock);
            let claimed = body.len() + if cut { 200 } else { 0 };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {claimed}\r\nConnection: close\r\n\r\n{body}"
            );
            let _ = sock.write_all(resp.as_bytes());
            let _ = sock.flush();
            if cut {
                // Let reqwest deliver the partial chunk before the short close.
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    });
    port
}

pub(super) fn turn_ctx<'a>(
    client: &'a reqwest::Client,
    base: &'a str,
    cwd: &'a Path,
) -> TurnCtx<'a> {
    TurnCtx {
        client,
        serve_base: base,
        auth: None,
        cwd,
        yes: true,
        auto_approve_all: false,
        auto_approve: None,
        review_edits: None,
        plan_exit: None,
        plan_enter: None,
    }
}

pub(super) const WRITE_TOOL_SSE: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"write_file\",\"arguments\":\"{\\\"path\\\":\\\"out.txt\\\",\\\"content\\\":\\\"hi\\\"}\"}}]}}]}\n\ndata: [DONE]\n\n";

pub(super) const FINAL_TEXT_SSE: &str =
    "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\ndata: [DONE]\n\n";

/// A completion with no content and no tool calls.
pub(super) const EMPTY_SSE: &str = "data: {\"choices\":[{\"delta\":{}}]}\n\ndata: [DONE]\n\n";

/// Tool-result message contents from history, in order.
pub(super) fn tool_result_texts(engine: &AgentEngine) -> Vec<String> {
    engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .filter_map(|m| m["content"].as_str().map(str::to_string))
        .collect()
}

pub(super) fn subagent(name: &str, model: Option<&str>, tools: Option<Vec<&str>>) -> Subagent {
    Subagent {
        name: name.to_string(),
        description: format!("the {name} specialist"),
        model: model.map(str::to_string),
        effort: None,
        tools: tools.map(|t| t.into_iter().map(str::to_string).collect()),
        body: format!("You are {name}. Follow the {name} playbook."),
        isolation_worktree: false,
        repo_local: false,
        source: PathBuf::new(),
    }
}

pub(super) fn tool_names(engine: &AgentEngine) -> Vec<String> {
    engine
        .tools_openai
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
        .collect()
}

pub(super) fn system_content(engine: &AgentEngine) -> String {
    engine.messages[0]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

/// Own-key generator for tool-wiring tests; never called.
pub(super) fn test_generator() -> crate::services::image_generate::GeneratorSource {
    crate::services::image_generate::GeneratorSource::OwnKey {
        model: "gemini-2.5-flash-image".to_string(),
        key: Box::new(crate::services::session_store::ApiKey::new_with_protocol(
            "k1".into(),
            "k".into(),
            "https://example.invalid/v1".into(),
            None,
            "sk-test".into(),
        )),
    }
}
