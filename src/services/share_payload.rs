//! Normalized share payload schema and per-source full-transcript extractors.
//!
//! `SharePayload` is the lossless JSON representation of one conversation served
//! over the share tunnel; each source has an `extract_*_full` mapping its on-disk
//! shape into this schema (vs. `context_ingest.rs`, which collapses to a summary).

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::services::context_ingest::paths_match;
use crate::services::device_fingerprint::hex_sha256;
use crate::services::session_store::{
    AttachmentStorage, CodeSessionState, MessageAttachment, StoredChatMessage,
};
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Wire schema version; bump on breaking shape changes (the viewer keys off it).
pub const SHARE_SCHEMA_VERSION: &str = "1";

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
    /// Fresh `meta` with `served_at = now`; callers fill `redaction_summary` later.
    pub fn new_meta(live: bool) -> ShareMeta {
        ShareMeta {
            aivo_version: crate::version::VERSION.to_string(),
            redacted: false,
            redaction_summary: None,
            live,
            served_at: Utc::now(),
        }
    }

    /// Approximate total char count across all message content.
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

/// Fold a result-only message (claude emits results as `user`, codex as `tool`)
/// into the preceding tool_use turn, since the viewer renders them inline.
fn merge_tool_result_turns(messages: Vec<ShareMessage>) -> Vec<ShareMessage> {
    let mut out: Vec<ShareMessage> = Vec::with_capacity(messages.len());
    // Chat persists calls/results id-less; mint a shared id per pair as we fold
    // so the viewer can match them (else the call shows "pending" + orphan result).
    let mut synthetic_id: u64 = 0;
    for msg in messages {
        let only_tool_results = !msg.content.is_empty()
            && msg
                .content
                .iter()
                .all(|b| matches!(b, ContentBlock::ToolResult { .. }));
        if only_tool_results && let Some(prev) = out.last_mut() {
            let mut results = msg.content;
            for block in &mut results {
                if let ContentBlock::ToolResult { id: id @ None, .. } = block
                    && let Some(call_id) =
                        next_unpaired_call_id(&mut prev.content, &mut synthetic_id)
                {
                    *id = Some(call_id);
                }
            }
            prev.content.extend(results);
            continue;
        }
        out.push(msg);
    }
    out
}

/// Mint a synthetic id for the next id-less `ToolCall` in `content` and return
/// it, so a freshly-folded `ToolResult` can reference the same id.
fn next_unpaired_call_id(content: &mut [ContentBlock], counter: &mut u64) -> Option<String> {
    for block in content.iter_mut() {
        if let ContentBlock::ToolCall { id: id @ None, .. } = block {
            *counter += 1;
            let new_id = format!("chat-tool-{counter}");
            *id = Some(new_id.clone());
            return Some(new_id);
        }
    }
    None
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
// aivo code extractor
// ---------------------------------------------------------------------------

/// Map a persisted `aivo code` session onto the share schema. Legacy encrypted
/// sessions are decrypted at load time, so a decryption error surfaces from the
/// store rather than here.
pub fn extract_chat_full(
    state: &CodeSessionState,
    project_root: Option<&str>,
) -> Result<SharePayload> {
    let messages = state.messages.clone();

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
    // No-op for a tool-free chat; folds agent tool results inline otherwise.
    let share_messages = merge_tool_result_turns(share_messages);

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
        source_cli: "code".to_string(),
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
    // Decode persisted tool_call/tool_result entries into structured blocks so
    // the viewer renders tools, not raw JSON.
    match m.role.as_str() {
        "tool_call" => return map_chat_tool_call(&m.content, timestamp),
        "tool_result" => return map_chat_tool_result(&m.content, timestamp),
        _ => {}
    }

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
        model: m.model,
        reasoning: m.reasoning_content,
        content,
    }
}

/// Decode a `{"name","args"}` tool_call entry; falls back to text if malformed.
fn map_chat_tool_call(raw: &str, timestamp: Option<DateTime<Utc>>) -> ShareMessage {
    let block = serde_json::from_str::<Value>(raw)
        .ok()
        .and_then(|decoded| {
            let name = decoded.get("name").and_then(Value::as_str)?.to_string();
            let arguments = decoded.get("args").cloned().unwrap_or(Value::Null);
            Some(ContentBlock::ToolCall {
                id: None,
                name,
                arguments,
            })
        })
        .unwrap_or_else(|| ContentBlock::Text {
            text: raw.to_string(),
        });
    ShareMessage {
        role: "assistant".to_string(),
        timestamp,
        model: None,
        reasoning: None,
        content: vec![block],
    }
}

/// Decode a `tool_result` entry (errors are `error: `-prefixed by the bridge).
fn map_chat_tool_result(raw: &str, timestamp: Option<DateTime<Utc>>) -> ShareMessage {
    let block = match raw.strip_prefix("error: ") {
        Some(err) => ContentBlock::ToolResult {
            id: None,
            ok: false,
            output: String::new(),
            error: Some(err.to_string()),
        },
        None => ContentBlock::ToolResult {
            id: None,
            ok: true,
            output: raw.to_string(),
            error: None,
        },
    };
    ShareMessage {
        role: "tool".to_string(),
        timestamp,
        model: None,
        reasoning: None,
        content: vec![block],
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
            // Hash the raw base64 (dedup only, not forensics) to avoid a decode dep.
            (hex_sha256(data.as_bytes()), data.len() as u64)
        }
        AttachmentStorage::FileRef { .. } => {
            // Don't read the filesystem at share time; hash left blank by design.
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

// ---------------------------------------------------------------------------
// Claude Code extractor
// ---------------------------------------------------------------------------

/// Extract from a Claude Code JSONL session. Sidechain entries
/// (`isSidechain: true`) are skipped — forks, not the primary conversation.
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

/// Parse an Anthropic content array (`text`/`tool_use`/`tool_result`/`thinking`);
/// a bare string (older sessions) is accepted too.
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

/// Extract from a Codex rollout JSONL. Rejects the session if its `cwd` doesn't
/// match `project_root`, so a stray rollout isn't attributed to the wrong project.
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

/// Extract a Gemini session JSON; falls back to `lastUpdated` when per-message
/// timestamps are missing.
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

/// Extract from a Pi session JSONL (`session` + `message` lines); text only,
/// as Pi's JSONL has no tool invocations today.
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

/// Extract from the OpenCode SQLite database (rusqlite on `spawn_blocking`).
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

    // Join messages → parts in order; group parts by message_id into one message.
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
    fn merge_tool_result_turns_folds_alternation() {
        // alternating call/result turns collapse into one turn per call.
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
        // A tool_result with no preceding message survives as-is (rare).
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
    fn merge_tool_result_turns_pairs_idless_chat_blocks() {
        // id-less chat call/result should get a shared synthetic id on fold.
        let input = vec![
            msg(
                "assistant",
                vec![ContentBlock::ToolCall {
                    id: None,
                    name: "list_dir".into(),
                    arguments: serde_json::Value::Null,
                }],
            ),
            msg(
                "tool",
                vec![ContentBlock::ToolResult {
                    id: None,
                    ok: true,
                    output: "index.html".into(),
                    error: None,
                }],
            ),
        ];
        let out = merge_tool_result_turns(input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].content.len(), 2);
        let call_id = match &out[0].content[0] {
            ContentBlock::ToolCall { id, .. } => id.clone(),
            other => panic!("expected tool_call, got {other:?}"),
        };
        let result_id = match &out[0].content[1] {
            ContentBlock::ToolResult { id, .. } => id.clone(),
            other => panic!("expected tool_result, got {other:?}"),
        };
        assert!(call_id.is_some(), "call should receive a synthetic id");
        assert_eq!(call_id, result_id, "call and result must share an id");
    }

    #[test]
    fn approximate_chars_sums_text_and_code_and_tool_output() {
        let payload = SharePayload {
            schema_version: SHARE_SCHEMA_VERSION.into(),
            source_cli: "claude".into(),
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
        let messages = vec![
            StoredChatMessage {
                model: None,
                role: "user".into(),
                content: "Why does pagination break?".into(),
                reasoning_content: None,
                id: None,
                timestamp: Some("2026-04-01T10:00:00Z".into()),
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "assistant".into(),
                content: "The cursor is unset on first page.".into(),
                reasoning_content: Some("checked the helper".into()),
                id: None,
                timestamp: Some("2026-04-01T10:01:00Z".into()),
                attachments: None,
            },
        ];

        let state = CodeSessionState {
            session_id: "chat-abc".into(),
            key_id: "k1".into(),
            base_url: "https://api.example.com".into(),
            cwd: "/Users/alice/project/aivo".into(),
            model: "gpt-4o".into(),
            messages,
            engine_messages: None,
            import_fidelity: None,
            plan_state: None,
            updated_at: "2026-04-01T10:01:00Z".into(),
            created_at: "2026-04-01T09:55:00Z".into(),
        };

        let payload = extract_chat_full(&state, None).unwrap();
        assert_eq!(payload.source_cli, "code");
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
        let messages = vec![StoredChatMessage {
            model: None,
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
        let state = CodeSessionState {
            session_id: "s".into(),
            key_id: "k".into(),
            base_url: "u".into(),
            cwd: "/tmp".into(),
            model: "m".into(),
            messages,
            engine_messages: None,
            import_fidelity: None,
            plan_state: None,
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
    fn extract_chat_full_maps_per_message_model() {
        let msg = |model: Option<&str>, role: &str, content: &str| StoredChatMessage {
            model: model.map(str::to_string),
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        };
        let messages = vec![
            msg(None, "user", "q1"),
            msg(Some("model-a"), "assistant", "a1"),
            msg(None, "user", "q2"),
            msg(Some("model-b"), "assistant", "a2"),
        ];
        let state = CodeSessionState {
            session_id: "s".into(),
            key_id: "k".into(),
            base_url: "u".into(),
            cwd: "/tmp".into(),
            model: "model-b".into(),
            messages,
            engine_messages: None,
            import_fidelity: None,
            plan_state: None,
            updated_at: String::new(),
            created_at: String::new(),
        };

        let payload = extract_chat_full(&state, None).unwrap();
        assert_eq!(payload.messages[0].model, None);
        assert_eq!(payload.messages[1].model.as_deref(), Some("model-a"));
        assert_eq!(payload.messages[3].model.as_deref(), Some("model-b"));
    }

    #[test]
    fn extract_chat_full_maps_agent_tool_turns() {
        let messages = vec![
            StoredChatMessage {
                model: None,
                role: "user".into(),
                content: "create out.txt".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "tool_call".into(),
                content: r#"{"name":"write_file","args":{"path":"out.txt"}}"#.into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "tool_result".into(),
                content: "wrote out.txt".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "tool_call".into(),
                content: r#"{"name":"read_file","args":{"path":"missing"}}"#.into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "tool_result".into(),
                content: "error: no such file".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
            StoredChatMessage {
                model: None,
                role: "assistant".into(),
                content: "Done.".into(),
                reasoning_content: None,
                id: None,
                timestamp: None,
                attachments: None,
            },
        ];
        let state = CodeSessionState {
            session_id: "s".into(),
            key_id: "k".into(),
            base_url: "u".into(),
            cwd: "/tmp".into(),
            model: "m".into(),
            messages,
            engine_messages: None,
            import_fidelity: None,
            plan_state: None,
            updated_at: String::new(),
            created_at: String::new(),
        };

        let payload = extract_chat_full(&state, None).unwrap();
        // user, tool_call(+folded result), tool_call(+folded result), assistant.
        assert_eq!(payload.messages.len(), 4);

        // First tool turn: ToolCall + folded successful ToolResult.
        assert_eq!(payload.messages[1].role, "assistant");
        match &payload.messages[1].content[0] {
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                assert_eq!(name, "write_file");
                assert_eq!(arguments["path"], "out.txt");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        match &payload.messages[1].content[1] {
            ContentBlock::ToolResult { ok, output, .. } => {
                assert!(ok);
                assert_eq!(output, "wrote out.txt");
            }
            other => panic!("expected tool result, got {other:?}"),
        }

        // Second tool turn: the `error: ` prefix marks the result as failed.
        match &payload.messages[2].content[1] {
            ContentBlock::ToolResult { ok, error, .. } => {
                assert!(!ok);
                assert_eq!(error.as_deref(), Some("no such file"));
            }
            other => panic!("expected failed tool result, got {other:?}"),
        }
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
