//! Normalized thread type + age constant for the context pipeline.
//! Context is stateless — threads are reconstructed from session files.

use chrono::{DateTime, Utc};

/// Default: threads older than this are filtered out. Override with `--last-days=<N>` or `--all`.
pub const DEFAULT_THREAD_MAX_AGE_DAYS: i64 = 30;

/// A conversational thread: one session summarized into a first user "topic" and
/// last assistant "last_response". In-memory only.
#[derive(Debug, Clone)]
pub struct Thread {
    /// Which CLI produced the session: "claude" | "codex" | "code" | ...
    pub cli: String,
    /// Native session id (Claude UUID, Codex rollout id, aivo code session id).
    pub session_id: String,
    /// Provenance: JSONL path or `log://<session_id>` for chat-from-logs.
    pub source_path: String,
    /// First substantive user message in the session.
    pub topic: String,
    /// Last substantive assistant message.
    pub last_response: String,
    /// Session end timestamp (falls back to file mtime when the source lacks one).
    pub updated_at: DateTime<Utc>,
    /// The cwd this session belongs to, when knowable. None for sources whose
    /// home directory layout obscures cwd (e.g. gemini's sha256-hashed dirs).
    /// Populated best-effort and used by `aivo logs --cwd` filtering and the
    /// listing's cwd column.
    pub cwd: Option<String>,
}
