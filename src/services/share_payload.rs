//! Normalized share payload schema and per-source full-transcript extractors.
//!
//! `SharePayload` is the lossless, JSON-serializable representation of one
//! conversation that aivo serves over the share tunnel. Each native source
//! (amp / claude / codex / gemini / pi / opencode / aivo chat) has its own
//! `extract_*_full` that maps the source's on-disk shape into this schema.
//!
//! Companion to `context_ingest.rs`: that module collapses a session into a
//! one-line topic + one-line last-response summary for the picker; this one
//! preserves every message, including tool calls/results.

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

use crate::services::amp_threads;
use crate::services::context_ingest::paths_match;
use crate::services::device_fingerprint::hex_sha256;
use crate::services::session_store::{
    AttachmentStorage, ChatSessionState, MessageAttachment, StoredChatMessage,
};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Wire schema version. Bump on breaking shape changes; the public viewer
/// keys backward-compat behavior off this.
pub const SHARE_SCHEMA_VERSION: &str = "1";

/// Top-level payload: one shared conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharePayload {
    pub schema_version: String,
    pub source_cli: String,
    pub session_id: String,
    pub project: ProjectInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<DateTime<Utc>>,
    pub messages: Vec<ShareMessage>,
    pub meta: ShareMeta,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Code {
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
        text: String,
    },
    ToolCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    ToolResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        ok: bool,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    Attachment {
        kind: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        sha256: String,
        size_bytes: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareMeta {
    pub aivo_version: String,
    pub redacted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redaction_summary: Option<Vec<RedactionHit>>,
    pub live: bool,
    pub served_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RedactionHit {
    pub category: String,
    pub count: usize,
}

impl SharePayload {
    /// Build a fresh `meta` block with `served_at = now`. Callers may flip
    /// `live` and fill `redaction_summary` after redaction runs.
    pub fn new_meta(live: bool) -> ShareMeta {
        ShareMeta {
            aivo_version: crate::version::VERSION.to_string(),
            redacted: false,
            redaction_summary: None,
            live,
            served_at: Utc::now(),
        }
    }

    /// Per-message char count, summed across all text-bearing blocks. Useful
    /// for size warnings and the preview header.
    pub fn approximate_chars(&self) -> usize {
        self.messages
            .iter()
            .map(|m| {
                let body: usize = m
                    .content
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } | ContentBlock::Code { text, .. } => {
                            text.chars().count()
                        }
                        ContentBlock::ToolCall { arguments, .. } => arguments.to_string().len(),
                        ContentBlock::ToolResult { output, error, .. } => {
                            output.chars().count() + error.as_deref().map(str::len).unwrap_or(0)
                        }
                        ContentBlock::Attachment { .. } => 0,
                    })
                    .sum();
                body + m.reasoning.as_deref().map(str::len).unwrap_or(0)
            })
            .sum()
    }
}

/// Fold messages whose content is *only* tool_result blocks into the
/// previous message. Anthropic-style wires (amp, claude) emit each tool
/// result as a fresh `role: "user"` message; codex emits a separate
/// `role: "tool"` message. The share viewer always renders those results
/// inline with the preceding tool_use, so a standalone count of
/// `messages.len()` overcounts vs. what the viewer (and `aivo logs show`)
/// actually displays. Collapsing here keeps those counts in sync.
fn merge_tool_result_turns(messages: Vec<ShareMessage>) -> Vec<ShareMessage> {
    let mut out: Vec<ShareMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        let only_tool_results = !msg.content.is_empty()
            && msg
                .content
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }));
        if only_tool_results && let Some(prev) = out.last_mut() {
            prev.content.extend(msg.content);
            continue;
        }
        out.push(msg);
    }
    out
}

// ---------------------------------------------------------------------------
// Amp extractor
// ---------------------------------------------------------------------------

/// Read an amp thread JSON file (`~/.config/aivo/amp-threads/T-*.json`) and
/// map it onto the share schema. Amp's payload is the raw `params.thread`
/// the bridge captured from `uploadThread`, so we read it tolerantly: missing
/// fields fall back to defaults rather than failing the share.
pub async fn extract_amp_full(
    threads_dir: &Path,
    thread_id: &str,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let raw = amp_threads::load_thread(threads_dir, thread_id)
        .await
        .ok_or_else(|| {
            anyhow!(
                "amp thread '{thread_id}' not found in {}",
                threads_dir.display()
            )
        })?;
    extract_amp_value(&raw, project_root)
}

/// Extracts a SharePayload from an already-parsed amp thread JSON value.
/// Pulled out so tests don't have to round-trip through the filesystem.
pub fn extract_amp_value(raw: &Value, project_root: Option<&str>) -> Result<SharePayload> {
    let id = raw
        .get("id")
        .and_then(|v| v.as_str())
        .context("amp thread missing string `id`")?
        .to_string();

    let title = raw
        .get("title")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let agent_mode = raw
        .get("agentMode")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    // Amp `created` is unix millis; many threads omit it.
    let created_at = raw
        .get("created")
        .and_then(|v| v.as_i64())
        .and_then(|ms| Utc.timestamp_millis_opt(ms).single());

    let messages_array = raw
        .get("messages")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default();

    let mut messages: Vec<ShareMessage> = Vec::with_capacity(messages_array.len());
    let mut latest_ts: Option<DateTime<Utc>> = None;
    let mut last_model: Option<String> = None;

    for msg in messages_array {
        let role = msg
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();

        let timestamp = msg
            .get("createdAt")
            .or_else(|| msg.get("timestamp"))
            .and_then(parse_amp_timestamp);
        if let Some(ts) = timestamp {
            latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
        }

        let model = msg
            .get("model")
            .and_then(|v| v.as_str())
            .or_else(|| {
                msg.get("meta")
                    .and_then(|m| m.get("model"))
                    .and_then(|v| v.as_str())
            })
            .map(str::to_string);
        if model.is_some() {
            last_model = model.clone();
        }

        let reasoning = msg
            .get("reasoning")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let content = extract_amp_content(msg);

        messages.push(ShareMessage {
            role,
            timestamp,
            model,
            reasoning,
            content,
        });
    }

    let messages = merge_tool_result_turns(messages);

    let meta = SharePayload::new_meta(false);

    let project = ProjectInfo {
        root: project_root.map(str::to_string),
        name: project_root
            .and_then(|p| std::path::Path::new(p).file_name())
            .and_then(|n| n.to_str())
            .map(str::to_string),
    };

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "amp".to_string(),
        session_id: id,
        project,
        // Amp threads carry no canonical "model" at the top level; we
        // surface `agentMode` (smart/rush/deep/large) as a model proxy when
        // no per-message model is present, since that's what the recipient
        // actually cares about.
        model: last_model.or(agent_mode).or(title),
        created_at,
        // Prefer the latest message timestamp; fall back to `created` so a
        // thread without per-message timestamps still has *some* updated_at.
        updated_at: latest_ts.or(created_at),
        messages,
        meta,
    })
}

/// Pull content blocks out of one amp message. Amp's wire shape varies by
/// turn type (assistant turns may carry structured content arrays, user turns
/// are usually plain strings, tool results live in their own messages); this
/// extractor is tolerant and skips blocks it doesn't recognize.
fn extract_amp_content(msg: &Value) -> Vec<ContentBlock> {
    let mut out: Vec<ContentBlock> = Vec::new();

    if let Some(content) = msg.get("content") {
        match content {
            Value::String(s) if !s.is_empty() => out.push(ContentBlock::Text { text: s.clone() }),
            Value::Array(arr) => {
                for block in arr {
                    if let Some(b) = parse_amp_block(block) {
                        out.push(b);
                    }
                }
            }
            _ => {}
        }
    }

    // Amp sometimes splits tool I/O into siblings of `content`. Fold them in
    // so the recipient sees what the model actually did.
    if let Some(tool_use) = msg.get("toolUse").and_then(|v| v.as_object()) {
        out.push(ContentBlock::ToolCall {
            id: tool_use
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            name: tool_use
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("tool")
                .to_string(),
            arguments: tool_use
                .get("input")
                .or_else(|| tool_use.get("arguments"))
                .cloned()
                .unwrap_or(Value::Null),
        });
    }
    if let Some(tool_result) = msg.get("toolResult").and_then(|v| v.as_object()) {
        let output = tool_result
            .get("content")
            .or_else(|| tool_result.get("output"))
            .map(stringify_block_text)
            .unwrap_or_default();
        let error = tool_result
            .get("error")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        out.push(ContentBlock::ToolResult {
            id: tool_result
                .get("toolUseId")
                .or_else(|| tool_result.get("id"))
                .and_then(|v| v.as_str())
                .map(str::to_string),
            ok: error.is_none(),
            output,
            error,
        });
    }

    out
}

/// Parse one amp content block. Amp's `type` discriminator matches
/// Anthropic's (`text` / `tool_use` / `tool_result` / `thinking`) but the
/// `tool_result` envelope diverges: amp uses `toolUseID` (camelCase) for
/// the linkage and nests output under `run.result.{output,diff,error}`
/// (the per-tool shape varies). We accept both Anthropic and amp shapes
/// so older fixtures and live amp threads both render.
fn parse_amp_block(block: &Value) -> Option<ContentBlock> {
    let kind = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        "text" => block
            .get("text")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| ContentBlock::Text {
                text: s.to_string(),
            }),
        "thinking" | "reasoning" => block
            .get("text")
            .or_else(|| block.get("thinking"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| ContentBlock::Text {
                text: s.to_string(),
            }),
        "tool_use" => Some(ContentBlock::ToolCall {
            id: block.get("id").and_then(|v| v.as_str()).map(str::to_string),
            name: block
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("tool")
                .to_string(),
            arguments: block
                .get("input")
                .or_else(|| block.get("arguments"))
                .cloned()
                .unwrap_or(Value::Null),
        }),
        "tool_result" => Some(parse_amp_tool_result(block)),
        _ => None,
    }
}

/// Extract id + output + error from amp's `tool_result` block, accepting
/// both Anthropic's wire shape (`content`/`tool_use_id`/`is_error`) and
/// amp's native shape (`toolUseID` + `run.result.{output,diff,error,exitCode}`).
fn parse_amp_tool_result(block: &Value) -> ContentBlock {
    let id = block
        .get("tool_use_id")
        .or_else(|| block.get("toolUseID"))
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let run_result = block.get("run").and_then(|r| r.get("result"));

    // Per-tool output: Bash → `output`; edit/create → `diff`; many other
    // tools stuff their payload elsewhere in `result`. Fall back to a
    // JSON dump of `result` so unknown shapes still surface something
    // useful instead of an empty box.
    let amp_output = run_result.and_then(|r| {
        if let Some(s) = r.get("output").and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return Some(s.to_string());
        }
        if let Some(s) = r.get("diff").and_then(|v| v.as_str())
            && !s.is_empty()
        {
            return Some(s.to_string());
        }
        if let Some(obj) = r.as_object()
            && !obj.is_empty()
            && !(obj.len() == 1 && obj.contains_key("error"))
        {
            return Some(serde_json::to_string_pretty(r).unwrap_or_default());
        }
        None
    });

    let anthropic_output = block.get("content").map(stringify_block_text);
    let output = amp_output.or(anthropic_output).unwrap_or_default();

    let amp_error = run_result
        .and_then(|r| r.get("error"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    let exit_code_failed = run_result
        .and_then(|r| r.get("exitCode"))
        .and_then(|v| v.as_i64())
        .is_some_and(|c| c != 0);
    let anthropic_is_error = block
        .get("is_error")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let error = amp_error
        .or_else(|| (exit_code_failed || anthropic_is_error).then(|| output.clone()))
        .filter(|s| !s.is_empty());

    ContentBlock::ToolResult {
        id,
        ok: error.is_none(),
        output,
        error,
    }
}

/// Flatten a `content` field that may be either a plain string or an array
/// of `{type:"text", text:"..."}` blocks (Anthropic-style).
fn stringify_block_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut out = String::new();
            for block in arr {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(t);
                }
            }
            out
        }
        _ => v.to_string(),
    }
}

// ---------------------------------------------------------------------------
// aivo chat extractor
// ---------------------------------------------------------------------------

/// Map a `ChatSessionState` (as persisted by `aivo chat`) onto the share
/// schema. Decryption may fail when the underlying secret is rotated/missing;
/// the error is surfaced rather than silently producing an empty share.
pub fn extract_chat_full(
    state: &ChatSessionState,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let messages = state
        .decrypt_messages()
        .context("failed to decrypt chat session messages")?;

    let created_at = parse_chat_timestamp(&state.created_at);
    let mut latest_ts: Option<DateTime<Utc>> = None;

    let share_messages: Vec<ShareMessage> = messages
        .into_iter()
        .map(|m| {
            let timestamp = m.timestamp.as_deref().and_then(parse_chat_timestamp);
            if let Some(ts) = timestamp {
                latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
            }
            map_chat_message(m, timestamp)
        })
        .collect();

    let project = ProjectInfo {
        root: project_root
            .map(str::to_string)
            .or_else(|| Some(state.cwd.clone())),
        name: project_root
            .or(Some(state.cwd.as_str()))
            .and_then(|p| std::path::Path::new(p).file_name())
            .and_then(|n| n.to_str())
            .map(str::to_string),
    };

    let updated_at = latest_ts
        .or_else(|| parse_chat_timestamp(&state.updated_at))
        .or(created_at);

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "chat".to_string(),
        session_id: state.session_id.clone(),
        project,
        model: Some(state.model.clone()).filter(|s| !s.is_empty()),
        created_at,
        updated_at,
        messages: share_messages,
        meta: SharePayload::new_meta(false),
    })
}

fn map_chat_message(m: StoredChatMessage, timestamp: Option<DateTime<Utc>>) -> ShareMessage {
    let mut content: Vec<ContentBlock> = Vec::new();
    if !m.content.is_empty() {
        content.push(ContentBlock::Text { text: m.content });
    }
    if let Some(attachments) = m.attachments {
        for att in attachments {
            content.push(map_attachment(att));
        }
    }
    ShareMessage {
        role: m.role,
        timestamp,
        // aivo chat stores model at the session level only, never per-turn.
        model: None,
        reasoning: m.reasoning_content,
        content,
    }
}

fn map_attachment(att: MessageAttachment) -> ContentBlock {
    let kind = if att.mime_type.starts_with("image/") {
        "image"
    } else {
        "file"
    }
    .to_string();
    let (sha256, size_bytes) = match &att.storage {
        AttachmentStorage::Inline { data } => {
            // `data` is base64; for v1 we hash the raw base64 string rather
            // than its decoded bytes — the hash is for viewer-side dedup,
            // not forensics, and decoding would require a base64 dep we
            // don't otherwise need.
            (hex_sha256(data.as_bytes()), data.len() as u64)
        }
        AttachmentStorage::FileRef { .. } => {
            // Don't read the user's filesystem at share time; the path is
            // surfaced in `name` and the hash is left blank by design.
            (String::new(), 0)
        }
    };
    ContentBlock::Attachment {
        kind,
        name: Some(att.name),
        sha256,
        size_bytes,
    }
}

fn parse_chat_timestamp(s: &str) -> Option<DateTime<Utc>> {
    if s.is_empty() {
        return None;
    }
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|p| p.with_timezone(&Utc))
}

/// Amp emits timestamps either as RFC3339 strings or unix-millis numbers.
fn parse_amp_timestamp(v: &Value) -> Option<DateTime<Utc>> {
    if let Some(s) = v.as_str()
        && let Ok(parsed) = DateTime::parse_from_rfc3339(s)
    {
        return Some(parsed.with_timezone(&Utc));
    }
    if let Some(ms) = v.as_i64() {
        return Utc.timestamp_millis_opt(ms).single();
    }
    None
}

// ---------------------------------------------------------------------------
// Claude Code extractor
// ---------------------------------------------------------------------------

/// Extract a full SharePayload from a Claude Code JSONL session file.
/// Source path is the `.jsonl` file under `~/.claude/projects/<encoded>/`.
/// Sidechain entries (`isSidechain: true`) are skipped — they're forks of
/// the main thread, not part of the user's primary conversation.
pub async fn extract_claude_full(
    path: &std::path::Path,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let file = fs::File::open(path)
        .await
        .with_context(|| format!("opening claude session {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut latest_ts: Option<DateTime<Utc>> = None;
    let mut messages: Vec<ShareMessage> = Vec::new();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if session_id.is_none()
            && let Some(sid) = v.get("sessionId").and_then(|s| s.as_str())
        {
            session_id = Some(sid.to_string());
        }
        if v.get("isSidechain")
            .and_then(|b| b.as_bool())
            .unwrap_or(false)
        {
            continue;
        }
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind != "user" && kind != "assistant" {
            continue;
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if let Some(ts) = timestamp {
            latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
        }

        let message = match v.get("message") {
            Some(m) => m,
            None => continue,
        };
        if let Some(m) = message.get("model").and_then(|v| v.as_str())
            && !m.is_empty()
        {
            model = Some(m.to_string());
        }

        let content = parse_anthropic_content_array(message.get("content"));
        if content.is_empty() {
            continue;
        }
        messages.push(ShareMessage {
            role: kind.to_string(),
            timestamp,
            model: message
                .get("model")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            reasoning: None,
            content,
        });
    }

    let session_id = session_id.ok_or_else(|| anyhow!("claude session has no sessionId"))?;

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "claude".to_string(),
        session_id,
        project: project_info(project_root),
        model,
        created_at: None,
        updated_at: latest_ts,
        messages: merge_tool_result_turns(messages),
        meta: SharePayload::new_meta(false),
    })
}

/// Anthropic-style content arrays show up in claude (and amp). One block per
/// JSON entry: `text`, `tool_use`, `tool_result`, `thinking`. Strings are
/// also accepted (older sessions inline a string `content`).
fn parse_anthropic_content_array(content: Option<&Value>) -> Vec<ContentBlock> {
    let Some(content) = content else {
        return Vec::new();
    };
    if let Some(s) = content.as_str() {
        if s.is_empty() {
            return Vec::new();
        }
        return vec![ContentBlock::Text {
            text: s.to_string(),
        }];
    }
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    let mut out: Vec<ContentBlock> = Vec::with_capacity(arr.len());
    for block in arr {
        let kind = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match kind {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            "thinking" | "reasoning" => {
                if let Some(text) = block
                    .get("text")
                    .or_else(|| block.get("thinking"))
                    .and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    out.push(ContentBlock::Text {
                        text: text.to_string(),
                    });
                }
            }
            "tool_use" => out.push(ContentBlock::ToolCall {
                id: block.get("id").and_then(|v| v.as_str()).map(str::to_string),
                name: block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool")
                    .to_string(),
                arguments: block
                    .get("input")
                    .or_else(|| block.get("arguments"))
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
            "tool_result" => {
                let output = block
                    .get("content")
                    .map(stringify_block_text)
                    .unwrap_or_default();
                let is_error = block
                    .get("is_error")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                out.push(ContentBlock::ToolResult {
                    id: block
                        .get("tool_use_id")
                        .and_then(|v| v.as_str())
                        .map(str::to_string),
                    ok: !is_error,
                    output: if is_error {
                        String::new()
                    } else {
                        output.clone()
                    },
                    error: if is_error { Some(output) } else { None },
                });
            }
            _ => {}
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Codex extractor
// ---------------------------------------------------------------------------

/// Extract from a Codex rollout JSONL file. The session_meta payload's `cwd`
/// is checked against `project_root` (if provided) so a stray rollout doesn't
/// get attributed to the wrong project. `function_call` / `function_call_output`
/// entries become tool_call / tool_result blocks.
pub async fn extract_codex_full(
    path: &std::path::Path,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let file = fs::File::open(path)
        .await
        .with_context(|| format!("opening codex session {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut latest_ts: Option<DateTime<Utc>> = None;
    let mut messages: Vec<ShareMessage> = Vec::new();
    let mut project_matches = project_root.is_none();
    let mut model: Option<String> = None;
    let mut session_cwd: Option<String> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        if kind == "session_meta"
            && let Some(payload) = v.get("payload")
        {
            if let Some(id) = payload.get("id").and_then(|s| s.as_str()) {
                session_id = Some(id.to_string());
            }
            if let Some(cwd) = payload.get("cwd").and_then(|s| s.as_str()) {
                session_cwd = Some(cwd.to_string());
                if let Some(root) = project_root
                    && paths_match(cwd, root)
                {
                    project_matches = true;
                }
            }
        }

        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if let Some(ts) = timestamp {
            latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
        }

        if kind == "response_item"
            && let Some(payload) = v.get("payload")
        {
            let payload_kind = payload.get("type").and_then(|t| t.as_str()).unwrap_or("");
            match payload_kind {
                "message" => {
                    let role = payload
                        .get("role")
                        .and_then(|s| s.as_str())
                        .unwrap_or("user")
                        .to_string();
                    let content = parse_codex_message_content(payload);
                    if !content.is_empty() {
                        if let Some(m) = payload.get("model").and_then(|v| v.as_str())
                            && !m.is_empty()
                        {
                            model = Some(m.to_string());
                        }
                        messages.push(ShareMessage {
                            role,
                            timestamp,
                            model: payload
                                .get("model")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            reasoning: None,
                            content,
                        });
                    }
                }
                "function_call" => {
                    let arguments = payload
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .and_then(|s| serde_json::from_str::<Value>(s).ok())
                        .or_else(|| payload.get("arguments").cloned())
                        .unwrap_or(Value::Null);
                    messages.push(ShareMessage {
                        role: "assistant".to_string(),
                        timestamp,
                        model: None,
                        reasoning: None,
                        content: vec![ContentBlock::ToolCall {
                            id: payload
                                .get("call_id")
                                .or_else(|| payload.get("id"))
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            name: payload
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("tool")
                                .to_string(),
                            arguments,
                        }],
                    });
                }
                "function_call_output" => {
                    let output = payload
                        .get("output")
                        .map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .unwrap_or_default();
                    messages.push(ShareMessage {
                        role: "tool".to_string(),
                        timestamp,
                        model: None,
                        reasoning: None,
                        content: vec![ContentBlock::ToolResult {
                            id: payload
                                .get("call_id")
                                .or_else(|| payload.get("id"))
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            ok: true,
                            output,
                            error: None,
                        }],
                    });
                }
                _ => {}
            }
        }
    }

    if !project_matches {
        return Err(anyhow!(
            "codex session cwd does not match project root (session cwd: {:?})",
            session_cwd
        ));
    }
    let session_id = session_id.ok_or_else(|| anyhow!("codex session has no id"))?;

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "codex".to_string(),
        session_id,
        project: project_info(project_root.or(session_cwd.as_deref())),
        model,
        created_at: None,
        updated_at: latest_ts,
        messages: merge_tool_result_turns(messages),
        meta: SharePayload::new_meta(false),
    })
}

fn parse_codex_message_content(payload: &Value) -> Vec<ContentBlock> {
    let Some(arr) = payload.get("content").and_then(|c| c.as_array()) else {
        return Vec::new();
    };
    let mut out: Vec<ContentBlock> = Vec::new();
    for block in arr {
        let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if !(kind == "input_text" || kind == "output_text" || kind == "text") {
            continue;
        }
        if let Some(text) = block.get("text").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            out.push(ContentBlock::Text {
                text: text.to_string(),
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gemini extractor
// ---------------------------------------------------------------------------

/// Extract a Gemini session JSON file. Gemini's per-message timestamps are
/// reliable; falls back to top-level `lastUpdated` when missing.
pub async fn extract_gemini_full(
    path: &std::path::Path,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let content = fs::read_to_string(path)
        .await
        .with_context(|| format!("reading gemini session {}", path.display()))?;
    let v: Value = serde_json::from_str(&content)
        .with_context(|| format!("parsing gemini session {}", path.display()))?;

    let session_id = v
        .get("sessionId")
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("gemini session missing sessionId"))?
        .to_string();
    let messages_array = v
        .get("messages")
        .and_then(|m| m.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default();

    let mut messages: Vec<ShareMessage> = Vec::with_capacity(messages_array.len());
    let mut latest_ts: Option<DateTime<Utc>> = None;

    for msg in messages_array {
        let kind = msg.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let role = match kind {
            "user" => "user",
            "gemini" => "assistant",
            _ => continue,
        };
        let text = stringify_gemini_content(msg.get("content"));
        if text.is_empty() {
            continue;
        }
        let timestamp = msg
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if let Some(ts) = timestamp {
            latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
        }
        messages.push(ShareMessage {
            role: role.to_string(),
            timestamp,
            model: None,
            reasoning: None,
            content: vec![ContentBlock::Text { text }],
        });
    }

    if latest_ts.is_none()
        && let Some(ts) = v.get("lastUpdated").and_then(|s| s.as_str())
        && let Ok(parsed) = DateTime::parse_from_rfc3339(ts)
    {
        latest_ts = Some(parsed.with_timezone(&Utc));
    }

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "gemini".to_string(),
        session_id,
        project: project_info(project_root),
        model: None,
        created_at: None,
        updated_at: latest_ts,
        messages,
        meta: SharePayload::new_meta(false),
    })
}

fn stringify_gemini_content(v: Option<&Value>) -> String {
    let Some(v) = v else {
        return String::new();
    };
    match v {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut out = String::new();
            for block in arr {
                if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(t);
                }
            }
            out
        }
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// Pi extractor
// ---------------------------------------------------------------------------

/// Extract from a Pi session JSONL. Pi's lines come in two kinds: `session`
/// (carries the id) and `message` (carries role+content). Tool invocations
/// aren't represented in Pi's JSONL today, so we surface text only.
pub async fn extract_pi_full(
    path: &std::path::Path,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let file = fs::File::open(path)
        .await
        .with_context(|| format!("opening pi session {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();

    let mut session_id: Option<String> = None;
    let mut latest_ts: Option<DateTime<Utc>> = None;
    let mut messages: Vec<ShareMessage> = Vec::new();

    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if kind == "session"
            && let Some(id) = v.get("id").and_then(|s| s.as_str())
        {
            session_id = Some(id.to_string());
        }
        let timestamp = v
            .get("timestamp")
            .and_then(|s| s.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));
        if let Some(ts) = timestamp {
            latest_ts = Some(latest_ts.map_or(ts, |cur| cur.max(ts)));
        }
        if kind != "message" {
            continue;
        }
        let Some(message) = v.get("message") else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(|s| s.as_str())
            .unwrap_or("user")
            .to_string();
        let text = stringify_pi_content(message.get("content"));
        if text.is_empty() {
            continue;
        }
        messages.push(ShareMessage {
            role,
            timestamp,
            model: None,
            reasoning: None,
            content: vec![ContentBlock::Text { text }],
        });
    }

    let session_id = session_id.ok_or_else(|| anyhow!("pi session has no id"))?;
    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "pi".to_string(),
        session_id,
        project: project_info(project_root),
        model: None,
        created_at: None,
        updated_at: latest_ts,
        messages,
        meta: SharePayload::new_meta(false),
    })
}

fn stringify_pi_content(v: Option<&Value>) -> String {
    let Some(arr) = v.and_then(|x| x.as_array()) else {
        return String::new();
    };
    let mut out = String::new();
    for block in arr {
        if block.get("type").and_then(|t| t.as_str()) == Some("text")
            && let Some(t) = block.get("text").and_then(|v| v.as_str())
        {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(t);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// OpenCode extractor
// ---------------------------------------------------------------------------

/// Extract from the OpenCode SQLite database. Sync rusqlite work is scheduled
/// on `spawn_blocking`. The query joins messages → parts ordered chronologically,
/// then maps each `text` part onto a Text block per message.
pub async fn extract_opencode_full(
    db_path: &std::path::Path,
    session_id: &str,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let db_path = db_path.to_path_buf();
    let session_id = session_id.to_string();
    let project_root = project_root.map(str::to_string);

    tokio::task::spawn_blocking(move || opencode_query_one(&db_path, &session_id, project_root))
        .await
        .map_err(|e| anyhow!("opencode extractor task panicked: {e}"))?
}

fn opencode_query_one(
    db_path: &std::path::Path,
    session_id: &str,
    project_root: Option<String>,
) -> Result<SharePayload> {
    let conn =
        rusqlite::Connection::open_with_flags(db_path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .with_context(|| format!("opening opencode db {}", db_path.display()))?;

    // Verify the session exists and pull its updated timestamp.
    let updated_ms: Option<i64> = conn
        .query_row(
            "SELECT time_updated FROM session WHERE id = ?1 LIMIT 1",
            [session_id],
            |row| row.get::<_, i64>(0),
        )
        .ok();
    if updated_ms.is_none() {
        return Err(anyhow!("opencode session '{session_id}' not found"));
    }

    // Walk messages → parts in order. Group parts by message_id into one
    // ShareMessage per row. We materialize as a Vec then group, which is
    // simpler than a per-message subquery.
    let mut stmt = conn.prepare(
        "SELECT m.id,
                json_extract(m.data, '$.role') AS role,
                m.time_created,
                p.data
           FROM message m
           JOIN part p ON p.message_id = m.id
          WHERE m.session_id = ?1
          ORDER BY m.time_created ASC, p.time_created ASC",
    )?;
    let rows = stmt.query_map([session_id], |row| {
        Ok((
            row.get::<_, String>(0)?,                             // message id
            row.get::<_, Option<String>>(1)?.unwrap_or_default(), // role
            row.get::<_, i64>(2)?,                                // time_created (ms)
            row.get::<_, String>(3)?,                             // part.data JSON
        ))
    })?;

    let mut messages: Vec<ShareMessage> = Vec::new();
    let mut current_msg_id: Option<String> = None;
    for row in rows.flatten() {
        let (msg_id, role, time_ms, part_json) = row;
        let part: Value = match serde_json::from_str(&part_json) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let block = match part.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "text" => part
                .get("text")
                .and_then(|t| t.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| ContentBlock::Text {
                    text: s.to_string(),
                }),
            "tool" | "tool_call" | "tool_use" => Some(ContentBlock::ToolCall {
                id: part
                    .get("id")
                    .or_else(|| part.get("call_id"))
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                name: part
                    .get("name")
                    .or_else(|| part.get("tool"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool")
                    .to_string(),
                arguments: part
                    .get("input")
                    .or_else(|| part.get("arguments"))
                    .cloned()
                    .unwrap_or(Value::Null),
            }),
            _ => None,
        };
        let Some(block) = block else { continue };

        let timestamp = chrono::DateTime::<Utc>::from_timestamp_millis(time_ms);
        if Some(&msg_id) != current_msg_id.as_ref() {
            messages.push(ShareMessage {
                role: role.clone(),
                timestamp,
                model: None,
                reasoning: None,
                content: vec![block],
            });
            current_msg_id = Some(msg_id);
        } else if let Some(last) = messages.last_mut() {
            last.content.push(block);
        }
    }

    let updated_at = updated_ms.and_then(chrono::DateTime::<Utc>::from_timestamp_millis);

    Ok(SharePayload {
        schema_version: SHARE_SCHEMA_VERSION.to_string(),
        source_cli: "opencode".to_string(),
        session_id: session_id.to_string(),
        project: project_info(project_root.as_deref()),
        model: None,
        created_at: None,
        updated_at,
        messages,
        meta: SharePayload::new_meta(false),
    })
}

// ---------------------------------------------------------------------------
// Common helpers
// ---------------------------------------------------------------------------

fn project_info(project_root: Option<&str>) -> ProjectInfo {
    ProjectInfo {
        root: project_root.map(str::to_string),
        name: project_root
            .and_then(|p| std::path::Path::new(p).file_name())
            .and_then(|n| n.to_str())
            .map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn text(t: &str) -> ContentBlock {
        ContentBlock::Text { text: t.into() }
    }
    fn tool_call(id: &str) -> ContentBlock {
        ContentBlock::ToolCall {
            id: Some(id.into()),
            name: "Bash".into(),
            arguments: serde_json::Value::Null,
        }
    }
    fn tool_result(id: &str, output: &str) -> ContentBlock {
        ContentBlock::ToolResult {
            id: Some(id.into()),
            ok: true,
            output: output.into(),
            error: None,
        }
    }
    fn msg(role: &str, content: Vec<ContentBlock>) -> ShareMessage {
        ShareMessage {
            role: role.into(),
            timestamp: None,
            model: None,
            reasoning: None,
            content,
        }
    }

    #[test]
    fn merge_tool_result_turns_folds_amp_alternation() {
        // user / asst(tool_use) / user(tool_result) / asst(tool_use) / user(tool_result)
        // collapses to user / asst(tool_use + tool_result) / asst(tool_use + tool_result)
        let input = vec![
            msg("user", vec![text("hi")]),
            msg("assistant", vec![tool_call("a")]),
            msg("user", vec![tool_result("a", "ok-a")]),
            msg("assistant", vec![tool_call("b")]),
            msg("user", vec![tool_result("b", "ok-b")]),
        ];
        let out = merge_tool_result_turns(input);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].role, "user");
        assert_eq!(out[1].role, "assistant");
        assert_eq!(out[1].content.len(), 2);
        assert!(matches!(out[1].content[0], ContentBlock::ToolCall { .. }));
        assert!(matches!(out[1].content[1], ContentBlock::ToolResult { .. }));
        assert_eq!(out[2].content.len(), 2);
    }

    #[test]
    fn merge_tool_result_turns_keeps_mixed_user_turn() {
        // A user message that carries text alongside a tool_result is a real
        // user turn — leave it alone.
        let input = vec![
            msg("assistant", vec![tool_call("a")]),
            msg("user", vec![text("followup"), tool_result("a", "ok")]),
        ];
        let out = merge_tool_result_turns(input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[1].role, "user");
        assert_eq!(out[1].content.len(), 2);
    }

    #[test]
    fn merge_tool_result_turns_preserves_orphan_first_message() {
        // A tool_result with no preceding message survives as-is (rare, but
        // a few amp threads start with one and existing tests rely on it).
        let input = vec![msg("user", vec![tool_result("a", "ok")])];
        let out = merge_tool_result_turns(input);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0].content[0], ContentBlock::ToolResult { .. }));
    }

    #[test]
    fn merge_tool_result_turns_preserves_error_payload() {
        let err_block = ContentBlock::ToolResult {
            id: Some("a".into()),
            ok: false,
            output: String::new(),
            error: Some("boom".into()),
        };
        let input = vec![
            msg("assistant", vec![tool_call("a")]),
            msg("user", vec![err_block]),
        ];
        let out = merge_tool_result_turns(input);
        assert_eq!(out.len(), 1);
        match &out[0].content[1] {
            ContentBlock::ToolResult { ok, error, .. } => {
                assert!(!*ok);
                assert_eq!(error.as_deref(), Some("boom"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn extract_amp_value_preserves_messages_and_models() {
        let raw = json!({
            "v": 1,
            "id": "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0",
            "title": "fix pagination bug",
            "created": 1778211465000i64,
            "agentMode": "smart",
            "messages": [
                {
                    "role": "user",
                    "createdAt": "2026-04-01T10:00:00Z",
                    "content": "Why does the cursor pagination return empty pages?"
                },
                {
                    "role": "assistant",
                    "createdAt": "2026-04-01T10:01:00Z",
                    "model": "claude-sonnet-4-5",
                    "content": [
                        {"type": "text", "text": "Let me check the helper."},
                        {"type": "tool_use", "id": "call_1", "name": "Read",
                         "input": {"path": "handlers/users.go"}}
                    ]
                },
                {
                    "role": "tool",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_1",
                         "content": [{"type": "text", "text": "fn paginate() {...}"}]}
                    ]
                }
            ]
        });

        let payload = extract_amp_value(&raw, Some("/Users/alice/project/work/aivo")).unwrap();
        assert_eq!(payload.source_cli, "amp");
        assert_eq!(payload.session_id, "T-019e05ae-80a5-7718-80ee-ec89cb6fc1c0");
        assert_eq!(payload.project.name.as_deref(), Some("aivo"));
        // Tool-result-only third message folds into the assistant turn so
        // the count matches what the viewer renders.
        assert_eq!(payload.messages.len(), 2);

        // First user turn — plain string content.
        let user = &payload.messages[0];
        assert_eq!(user.role, "user");
        assert_eq!(user.content.len(), 1);
        match &user.content[0] {
            ContentBlock::Text { text } => assert!(text.contains("cursor pagination")),
            other => panic!("expected text block, got {other:?}"),
        }

        // Assistant turn carries text + tool_use + the merged tool_result.
        let asst = &payload.messages[1];
        assert_eq!(asst.role, "assistant");
        assert_eq!(asst.model.as_deref(), Some("claude-sonnet-4-5"));
        assert_eq!(asst.content.len(), 3);
        assert!(matches!(asst.content[0], ContentBlock::Text { .. }));
        match &asst.content[1] {
            ContentBlock::ToolCall { id, name, .. } => {
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(name, "Read");
            }
            other => panic!("expected tool_call, got {other:?}"),
        }
        match &asst.content[2] {
            ContentBlock::ToolResult {
                id,
                ok,
                output,
                error,
            } => {
                assert_eq!(id.as_deref(), Some("call_1"));
                assert!(*ok);
                assert!(output.contains("fn paginate"));
                assert!(error.is_none());
            }
            other => panic!("expected tool_result, got {other:?}"),
        }

        // Top-level model surfaces the assistant's last seen model.
        assert_eq!(payload.model.as_deref(), Some("claude-sonnet-4-5"));

        // updated_at is the latest message timestamp.
        let updated = payload.updated_at.expect("updated_at present");
        assert_eq!(
            updated.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "2026-04-01T10:01:00Z"
        );
    }

    #[test]
    fn extract_amp_value_falls_back_to_agent_mode_when_no_message_model() {
        let raw = json!({
            "id": "T-bare",
            "agentMode": "rush",
            "messages": [
                { "role": "user", "content": "hi" }
            ]
        });
        let payload = extract_amp_value(&raw, None).unwrap();
        assert_eq!(payload.model.as_deref(), Some("rush"));
        assert!(payload.project.root.is_none());
    }

    #[test]
    fn extract_amp_tool_result_unwraps_amp_native_bash_shape() {
        // Real amp threads ship tool_result with `toolUseID` (camelCase)
        // and the output nested under `run.result.output` — not the
        // Anthropic `content`/`tool_use_id`/`is_error` shape the old
        // parser assumed. Both must survive.
        let raw = json!({
            "id": "T-amp-bash",
            "messages": [
                { "role": "user", "content": "run pwd" },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "call_x", "name": "Bash",
                         "input": {"cmd": "pwd"}}
                    ]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "toolUseID": "call_x",
                        "run": {
                            "status": "done",
                            "result": {"exitCode": 0, "output": "/tmp/proj"}
                        }
                    }]
                }
            ]
        });
        let payload = extract_amp_value(&raw, None).unwrap();
        // tool_result merges into the preceding assistant turn — it lives
        // alongside the tool_use at index 1.
        let asst = &payload.messages[1];
        match &asst.content[1] {
            ContentBlock::ToolResult {
                id,
                ok,
                output,
                error,
            } => {
                assert_eq!(id.as_deref(), Some("call_x"));
                assert!(*ok);
                assert_eq!(output, "/tmp/proj");
                assert!(error.is_none());
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn extract_amp_tool_result_unwraps_amp_native_edit_shape() {
        // Edit-tool results live at `run.result.diff` instead of `output`.
        let raw = json!({
            "id": "T-amp-edit",
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "toolUseID": "call_e",
                        "run": {
                            "status": "done",
                            "result": {"diff": "--- a\n+++ b", "lineRange": [1, 2]}
                        }
                    }]
                }
            ]
        });
        let payload = extract_amp_value(&raw, None).unwrap();
        match &payload.messages[0].content[0] {
            ContentBlock::ToolResult { output, ok, .. } => {
                assert!(output.contains("+++ b"));
                assert!(*ok);
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn extract_amp_tool_result_marks_failed_exit_code_as_error() {
        let raw = json!({
            "id": "T-amp-fail",
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "toolUseID": "call_f",
                        "run": {
                            "status": "done",
                            "result": {"exitCode": 1, "output": "permission denied"}
                        }
                    }]
                }
            ]
        });
        let payload = extract_amp_value(&raw, None).unwrap();
        match &payload.messages[0].content[0] {
            ContentBlock::ToolResult {
                ok, output, error, ..
            } => {
                assert!(!*ok);
                assert_eq!(output, "permission denied");
                assert_eq!(error.as_deref(), Some("permission denied"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn extract_amp_tool_result_falls_back_to_json_dump_for_unknown_shape() {
        // Some tools (todoWrite, read-file with summary, ...) put their
        // payload in `result` under keys we don't enumerate. We dump the
        // whole `result` object as a fallback so the viewer never shows
        // an empty box.
        let raw = json!({
            "id": "T-amp-unknown",
            "messages": [
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "toolUseID": "call_u",
                        "run": {
                            "status": "done",
                            "result": {"todos": ["a", "b"], "removed": 1}
                        }
                    }]
                }
            ]
        });
        let payload = extract_amp_value(&raw, None).unwrap();
        match &payload.messages[0].content[0] {
            ContentBlock::ToolResult { output, ok, .. } => {
                assert!(output.contains("todos"));
                assert!(output.contains("removed"));
                assert!(*ok);
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[test]
    fn extract_amp_value_rejects_payload_without_id() {
        let raw = json!({ "messages": [] });
        let err = extract_amp_value(&raw, None).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[tokio::test]
    async fn extract_amp_full_loads_from_disk() {
        let dir = TempDir::new().unwrap();
        let payload = json!({
            "id": "T-aaa",
            "title": "test",
            "messages": [
                { "role": "user", "content": "hello world" }
            ]
        });
        amp_threads::save_thread(dir.path(), &payload)
            .await
            .unwrap();

        let extracted = extract_amp_full(dir.path(), "T-aaa", None).await.unwrap();
        assert_eq!(extracted.session_id, "T-aaa");
        assert_eq!(extracted.messages.len(), 1);
        assert_eq!(extracted.schema_version, SHARE_SCHEMA_VERSION);
    }

    #[tokio::test]
    async fn extract_amp_full_errors_on_missing_thread() {
        let dir = TempDir::new().unwrap();
        let err = extract_amp_full(dir.path(), "T-nope", None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn approximate_chars_sums_text_and_code_and_tool_output() {
        let payload = SharePayload {
            schema_version: SHARE_SCHEMA_VERSION.into(),
            source_cli: "amp".into(),
            session_id: "T-x".into(),
            project: ProjectInfo::default(),
            model: None,
            created_at: None,
            updated_at: None,
            messages: vec![ShareMessage {
                role: "assistant".into(),
                timestamp: None,
                model: None,
                reasoning: None,
                content: vec![
                    ContentBlock::Text {
                        text: "abcd".into(),
                    },
                    ContentBlock::Code {
                        language: Some("rust".into()),
                        text: "let x = 1;".into(),
                    },
                    ContentBlock::ToolResult {
                        id: None,
                        ok: true,
                        output: "ok".into(),
                        error: None,
                    },
                ],
            }],
            meta: SharePayload::new_meta(false),
        };
        // 4 (text) + 10 (code) + 2 (output) = 16
        assert_eq!(payload.approximate_chars(), 16);
    }

    #[test]
    fn extract_chat_full_maps_messages_and_reasoning() {
        use crate::services::session_crypto::encrypt;

        let messages = vec![
            StoredChatMessage {
                role: "user".into(),
                content: "Why does pagination break?".into(),
                reasoning_content: None,
                id: None,
                timestamp: Some("2026-04-01T10:00:00Z".into()),
                attachments: None,
            },
            StoredChatMessage {
                role: "assistant".into(),
                content: "The cursor is unset on first page.".into(),
                reasoning_content: Some("checked the helper".into()),
                id: None,
                timestamp: Some("2026-04-01T10:01:00Z".into()),
                attachments: None,
            },
        ];
        let encrypted = encrypt(&serde_json::to_string(&messages).unwrap()).unwrap();

        let state = ChatSessionState {
            session_id: "chat-abc".into(),
            key_id: "k1".into(),
            base_url: "https://api.example.com".into(),
            cwd: "/Users/alice/project/aivo".into(),
            model: "gpt-4o".into(),
            messages: encrypted,
            updated_at: "2026-04-01T10:01:00Z".into(),
            created_at: "2026-04-01T09:55:00Z".into(),
        };

        let payload = extract_chat_full(&state, None).unwrap();
        assert_eq!(payload.source_cli, "chat");
        assert_eq!(payload.session_id, "chat-abc");
        assert_eq!(payload.model.as_deref(), Some("gpt-4o"));
        assert_eq!(payload.project.name.as_deref(), Some("aivo"));
        assert_eq!(payload.messages.len(), 2);

        // Reasoning passes through.
        assert_eq!(
            payload.messages[1].reasoning.as_deref(),
            Some("checked the helper")
        );

        // updated_at = max message timestamp = 10:01:00Z.
        let updated = payload.updated_at.unwrap();
        assert_eq!(
            updated.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "2026-04-01T10:01:00Z"
        );

        // Roundtrip text content.
        match &payload.messages[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Why does pagination break?"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[test]
    fn extract_chat_full_emits_attachment_blocks() {
        use crate::services::session_crypto::encrypt;

        let messages = vec![StoredChatMessage {
            role: "user".into(),
            content: "look at this".into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: Some(vec![
                MessageAttachment {
                    name: "diagram.png".into(),
                    mime_type: "image/png".into(),
                    storage: AttachmentStorage::Inline {
                        data: "aGVsbG8=".into(),
                    },
                },
                MessageAttachment {
                    name: "/Users/alice/notes.txt".into(),
                    mime_type: "text/plain".into(),
                    storage: AttachmentStorage::FileRef {
                        path: "/Users/alice/notes.txt".into(),
                    },
                },
            ]),
        }];
        let state = ChatSessionState {
            session_id: "s".into(),
            key_id: "k".into(),
            base_url: "u".into(),
            cwd: "/tmp".into(),
            model: "m".into(),
            messages: encrypt(&serde_json::to_string(&messages).unwrap()).unwrap(),
            updated_at: String::new(),
            created_at: String::new(),
        };

        let payload = extract_chat_full(&state, None).unwrap();
        assert_eq!(payload.messages[0].content.len(), 3); // text + 2 attachments
        match &payload.messages[0].content[1] {
            ContentBlock::Attachment {
                kind,
                name,
                sha256,
                size_bytes,
            } => {
                assert_eq!(kind, "image");
                assert_eq!(name.as_deref(), Some("diagram.png"));
                assert!(!sha256.is_empty());
                assert_eq!(*size_bytes, 8);
            }
            other => panic!("expected image attachment, got {other:?}"),
        }
        match &payload.messages[0].content[2] {
            ContentBlock::Attachment {
                kind,
                sha256,
                size_bytes,
                ..
            } => {
                assert_eq!(kind, "file");
                assert!(sha256.is_empty()); // FileRef → blank by design
                assert_eq!(*size_bytes, 0);
            }
            other => panic!("expected file attachment, got {other:?}"),
        }
    }

    #[test]
    fn extract_chat_full_surfaces_decryption_error() {
        let state = ChatSessionState {
            session_id: "s".into(),
            key_id: "k".into(),
            base_url: "u".into(),
            cwd: String::new(),
            model: "m".into(),
            messages: "not-actually-encrypted".into(),
            updated_at: String::new(),
            created_at: String::new(),
        };
        let err = extract_chat_full(&state, None).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("decrypt"));
    }

    #[tokio::test]
    async fn extract_claude_full_preserves_tool_use_and_results() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sess.jsonl");
        let lines = [
            r#"{"type":"user","sessionId":"sess-A","isSidechain":false,"timestamp":"2026-04-01T10:00:00Z","message":{"role":"user","content":"Read handlers/users.go"}}"#,
            // Sidechain — must be skipped.
            r#"{"type":"assistant","sessionId":"sess-A","isSidechain":true,"timestamp":"2026-04-01T10:00:30Z","message":{"role":"assistant","content":[{"type":"text","text":"SHOULD NOT APPEAR"}]}}"#,
            r#"{"type":"assistant","sessionId":"sess-A","isSidechain":false,"timestamp":"2026-04-01T10:01:00Z","message":{"role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"Reading the file."},{"type":"tool_use","id":"call_1","name":"Read","input":{"path":"handlers/users.go"}}]}}"#,
            r#"{"type":"user","sessionId":"sess-A","isSidechain":false,"timestamp":"2026-04-01T10:02:00Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[{"type":"text","text":"file content..."}]}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let payload = extract_claude_full(&path, Some("/Users/alice/project/aivo"))
            .await
            .unwrap();
        assert_eq!(payload.source_cli, "claude");
        assert_eq!(payload.session_id, "sess-A");
        assert_eq!(payload.model.as_deref(), Some("claude-sonnet-4-5"));
        // user + assistant(text + tool_call + folded tool_result). Sidechain
        // is skipped; the tool_result-only user message folds into the prior
        // assistant turn to match the viewer's rendered count.
        assert_eq!(payload.messages.len(), 2);
        for m in &payload.messages {
            for b in &m.content {
                if let ContentBlock::Text { text } = b {
                    assert!(!text.contains("SHOULD NOT APPEAR"));
                }
            }
        }
        let asst = &payload.messages[1];
        assert!(matches!(asst.content[0], ContentBlock::Text { .. }));
        match &asst.content[1] {
            ContentBlock::ToolCall { name, id, .. } => {
                assert_eq!(name, "Read");
                assert_eq!(id.as_deref(), Some("call_1"));
            }
            other => panic!("expected tool_call, got {other:?}"),
        }
        match &asst.content[2] {
            ContentBlock::ToolResult { id, ok, output, .. } => {
                assert_eq!(id.as_deref(), Some("call_1"));
                assert!(*ok);
                assert!(output.contains("file content"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_codex_full_maps_function_calls_and_outputs() {
        let dir = TempDir::new().unwrap();
        let proj = dir.path().join("proj");
        fs::create_dir_all(&proj).await.unwrap();
        let proj_str = proj.to_string_lossy().to_string();
        let proj_json = proj_str.replace('\\', "\\\\");

        let path = dir.path().join("rollout.jsonl");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-04-01T10:00:00Z","payload":{{"id":"codex-X","cwd":"{}","timestamp":"2026-04-01T10:00:00Z"}}}}"#,
                proj_json
            ),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:01:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"List files in src"}]}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:01:30Z","payload":{"type":"function_call","call_id":"fc1","name":"shell","arguments":"{\"cmd\":\"ls src\"}"}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:02:00Z","payload":{"type":"function_call_output","call_id":"fc1","output":"main.rs\nlib.rs"}}"#.to_string(),
            r#"{"type":"response_item","timestamp":"2026-04-01T10:02:30Z","payload":{"type":"message","role":"assistant","model":"gpt-5","content":[{"type":"output_text","text":"Two files."}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let payload = extract_codex_full(&path, Some(&proj_str)).await.unwrap();
        assert_eq!(payload.source_cli, "codex");
        assert_eq!(payload.session_id, "codex-X");
        assert_eq!(payload.model.as_deref(), Some("gpt-5"));
        // user + assistant(function_call + folded function_call_output) +
        // final assistant message. The tool-output turn folds into the
        // tool-call turn so the count matches the viewer.
        assert_eq!(payload.messages.len(), 3);
        let tool_turn = &payload.messages[1];
        match &tool_turn.content[0] {
            ContentBlock::ToolCall {
                name,
                id,
                arguments,
            } => {
                assert_eq!(name, "shell");
                assert_eq!(id.as_deref(), Some("fc1"));
                assert_eq!(arguments["cmd"], "ls src");
            }
            other => panic!("expected tool_call, got {other:?}"),
        }
        match &tool_turn.content[1] {
            ContentBlock::ToolResult { id, output, .. } => {
                assert_eq!(id.as_deref(), Some("fc1"));
                assert!(output.contains("main.rs"));
            }
            other => panic!("expected tool_result, got {other:?}"),
        }
        match &payload.messages[2].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Two files."),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_codex_full_rejects_mismatched_cwd() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rollout.jsonl");
        let lines = [
            r#"{"type":"session_meta","payload":{"id":"codex-Y","cwd":"/elsewhere"}}"#,
            r#"{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();
        let err = extract_codex_full(&path, Some("/not/matching"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("project root"));
    }

    #[tokio::test]
    async fn extract_gemini_full_maps_user_and_assistant_turns() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session-1.json");
        let body = r#"{
            "sessionId": "g-1",
            "projectHash": "abc",
            "lastUpdated": "2026-04-01T10:05:00Z",
            "messages": [
                {"type":"user","timestamp":"2026-04-01T10:00:00Z","content":[{"text":"hello"}]},
                {"type":"gemini","timestamp":"2026-04-01T10:01:00Z","content":"hi there"}
            ]
        }"#;
        fs::write(&path, body).await.unwrap();
        let payload = extract_gemini_full(&path, Some("/Users/alice/work"))
            .await
            .unwrap();
        assert_eq!(payload.session_id, "g-1");
        assert_eq!(payload.messages.len(), 2);
        assert_eq!(payload.messages[0].role, "user");
        assert_eq!(payload.messages[1].role, "assistant");
        let updated = payload.updated_at.unwrap();
        assert_eq!(
            updated.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            "2026-04-01T10:01:00Z"
        );
    }

    #[tokio::test]
    async fn extract_pi_full_pulls_id_and_messages() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("session.jsonl");
        let lines = [
            r#"{"type":"session","id":"pi-z","timestamp":"2026-04-01T10:00:00Z"}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:01:00Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
            r#"{"type":"message","timestamp":"2026-04-01T10:02:00Z","message":{"role":"assistant","content":[{"type":"text","text":"hello!"}]}}"#,
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();
        let payload = extract_pi_full(&path, None).await.unwrap();
        assert_eq!(payload.session_id, "pi-z");
        assert_eq!(payload.messages.len(), 2);
        assert_eq!(payload.messages[0].role, "user");
        if let ContentBlock::Text { text } = &payload.messages[1].content[0] {
            assert_eq!(text, "hello!");
        } else {
            panic!("expected text");
        }
    }

    #[tokio::test]
    async fn extract_opencode_full_groups_parts_per_message() {
        // Build a fake opencode db with one session, two messages, one having
        // two text parts that should fold into the same ShareMessage.
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("opencode.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT, time_created INTEGER);
             INSERT INTO session VALUES ('sess-1', 'proj-1', 1778211465000);
             INSERT INTO message VALUES ('m1', 'sess-1', '{\"role\":\"user\"}', 1000);
             INSERT INTO message VALUES ('m2', 'sess-1', '{\"role\":\"assistant\"}', 2000);
             INSERT INTO part VALUES ('p1', 'm1', '{\"type\":\"text\",\"text\":\"please refactor\"}', 1001);
             INSERT INTO part VALUES ('p2', 'm2', '{\"type\":\"text\",\"text\":\"step one\"}', 2001);
             INSERT INTO part VALUES ('p3', 'm2', '{\"type\":\"text\",\"text\":\"step two\"}', 2002);",
        )
        .unwrap();
        drop(conn);

        let payload = extract_opencode_full(&db_path, "sess-1", Some("/proj"))
            .await
            .unwrap();
        assert_eq!(payload.source_cli, "opencode");
        assert_eq!(payload.session_id, "sess-1");
        assert_eq!(payload.messages.len(), 2);
        // Second message has two text blocks (one per part).
        assert_eq!(payload.messages[1].content.len(), 2);
        match &payload.messages[1].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "step one"),
            other => panic!("expected text, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn extract_opencode_full_errors_on_missing_session() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("opencode.db");
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT, time_updated INTEGER);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT, data TEXT, time_created INTEGER);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT, data TEXT, time_created INTEGER);",
        )
        .unwrap();
        drop(conn);
        let err = extract_opencode_full(&db_path, "missing", None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn content_block_serializes_with_type_tag() {
        let block = ContentBlock::Text { text: "hi".into() };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hi");

        let call = ContentBlock::ToolCall {
            id: Some("a".into()),
            name: "Bash".into(),
            arguments: json!({"cmd": "ls"}),
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["type"], "tool_call");
        assert_eq!(json["name"], "Bash");
    }
}
