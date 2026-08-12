//! Open-session communication tools: `list_sessions` and `send_session`,
//! backed by the file mailboxes in `services::session_mail`. Top-level engine
//! only (sub-engines never get `set_session_mail`).

use super::*;

pub(super) fn list_sessions_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "list_sessions".to_string(),
        description: "List the user's other open aivo code sessions (id, model, directory, age). \
Use it to find a session to message with send_session."
            .to_string(),
        parameters: json!({"type": "object", "properties": {}}),
    }
}

pub(super) fn send_session_tool_spec() -> ToolSpec {
    ToolSpec {
        name: "send_session".to_string(),
        description: "Send a message to another open aivo code session (the user's own, on this \
machine — find targets with list_sessions). The receiving session sees it as an incoming message \
and its agent can reply. Default is fire-and-forget: your turn continues and any reply arrives \
later as a new incoming message. Set wait=true to block for the reply (up to timeout_ms) when you \
need the answer to proceed. When answering an incoming message, set reply_to to that message's id \
so the sender's wait is satisfied."
            .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target session id (or unique prefix) from list_sessions, or the `from` id of a message you're answering"},
                "text": {"type": "string", "description": "The message"},
                "reply_to": {"type": "string", "description": "Id of the incoming message this answers (from its frame), if any"},
                "wait": {"type": "boolean", "description": "Block until the target replies (default false)"},
                "timeout_ms": {"type": "integer", "description": "Max wait in ms (default 120000, max 600000); only with wait"}
            },
            "required": ["target", "text"]
        }),
    }
}

use crate::services::session_mail::short_sid;

const REPLY_POLL_MS: u64 = 250;

impl AgentEngine {
    pub(super) fn list_sessions_result(&self) -> Result<String, String> {
        let Some(mail) = &self.session_mail else {
            return Err("list_sessions: session mailbox not available here.".to_string());
        };
        let peers = mail.live_peers();
        if peers.is_empty() {
            return Ok(format!(
                "No other open sessions. (This session's id: {}.)",
                short_sid(mail.own_id())
            ));
        }
        let now = crate::services::session_mail::now_millis();
        let mut out = format!("Open sessions (yours: {}):\n", short_sid(mail.own_id()));
        for p in peers {
            out.push_str(&format!(
                "- {}  model: {}  dir: {}  open: {}\n",
                short_sid(&p.session_id),
                p.model.as_deref().unwrap_or("?"),
                p.cwd.as_deref().unwrap_or("?"),
                age_label(now.saturating_sub(p.started_at)),
            ));
        }
        out.push_str("Message one with send_session(target, text).");
        Ok(out)
    }

    pub(super) async fn send_session(&self, args: &Value) -> Result<String, String> {
        let Some(mail) = &self.session_mail else {
            return Err("send_session: session mailbox not available here.".to_string());
        };
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or("send_session: missing `target`.")?;
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or("send_session: missing `text`.")?;
        let reply_to = args.get("reply_to").and_then(|v| v.as_str());
        let wait = args.get("wait").and_then(|v| v.as_bool()).unwrap_or(false);

        let peer = mail
            .resolve_peer(target)
            .map_err(|e| format!("send_session: {e}"))?;
        let own_cwd = crate::services::system_env::current_dir_string();
        let msg_id = mail
            .send(&peer.session_id, text, reply_to, own_cwd)
            .map_err(|e| format!("send_session: delivery failed: {e}"))?;
        let to = short_sid(&peer.session_id);
        if !wait {
            return Ok(format!(
                "Delivered to session {to}. Not waiting — if it replies, the reply arrives \
as a new incoming message."
            ));
        }

        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(120_000)
            .clamp(5_000, 600_000);
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
        loop {
            if let Some(reply) = mail.take_reply(&msg_id) {
                return Ok(format!(
                    "Reply from session {}:\n{}",
                    short_sid(&reply.from),
                    reply.text
                ));
            }
            if std::time::Instant::now() >= deadline {
                return Err(format!(
                    "send_session: no reply from {to} within {}s. The message WAS delivered; \
the target may still answer later — its reply would arrive as a new incoming message. Don't \
resend unless you have something new to say.",
                    timeout_ms / 1000
                ));
            }
            if !mail.peer_alive(&peer.session_id) {
                return Err(format!(
                    "send_session: session {to} closed before replying (message id {msg_id})."
                ));
            }
            tokio::time::sleep(std::time::Duration::from_millis(REPLY_POLL_MS)).await;
        }
    }
}

fn age_label(millis: u64) -> String {
    let secs = millis / 1000;
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m", secs / 60),
        3600..=86_399 => format!("{}h", secs / 3600),
        _ => format!("{}d", secs / 86_400),
    }
}
