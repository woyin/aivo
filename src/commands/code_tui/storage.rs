use super::*;
use std::collections::HashMap;

/// Resumable chat sessions, newest first. `Some(dir)` scopes to that cwd;
/// `None` returns all sessions. Sessions whose key was removed stay listed
/// (labeled; resume falls back to the live key) so the list agrees with `aivo logs`.
pub(super) async fn load_resume_snapshots(
    session_store: &SessionStore,
    cwd_filter: Option<&str>,
) -> Result<Vec<SessionPreview>> {
    let by_id: HashMap<String, ApiKey> = session_store
        .get_keys()
        .await?
        .into_iter()
        .map(|key| (key.id.clone(), key))
        .collect();

    let mut sessions: Vec<SessionPreview> = session_store
        .all_chat_sessions()
        .await?
        .into_iter()
        .filter(|entry| cwd_filter.is_none_or(|dir| entry.cwd == dir))
        .map(|entry| {
            let key = by_id.get(&entry.key_id);
            SessionPreview::from_index_entry(entry, key)
        })
        .collect();

    sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(sessions)
}

/// This directory's importable Claude Code / Codex sessions, as picker rows.
/// Scanned once per `/resume` open (headline reads).
pub(super) async fn load_importable_previews(cwd: &str) -> Vec<SessionPreview> {
    crate::services::session_import::list_importable_sessions(std::path::Path::new(cwd))
        .await
        .into_iter()
        .map(SessionPreview::from_importable)
        .collect()
}

/// Resume a picker selection. A foreign row (`preview.origin`) that was already
/// continued in aivo loads its saved fork; one opened for the first time is
/// resumed IN MEMORY from the source transcript (assigned the live key/model) —
/// nothing is persisted until a real turn, so merely viewing a Claude/Codex
/// session never creates an aivo copy. A native row loads directly.
pub(super) async fn load_or_import_resume_session(
    session_store: &SessionStore,
    preview: &SessionPreview,
    key_id: &str,
    model: &str,
) -> std::result::Result<LoadedSession, String> {
    use crate::services::session_import::{ForeignResume, resume_foreign};
    if let Some(origin) = &preview.origin {
        let source_ts = chrono::DateTime::parse_from_rfc3339(&preview.updated_at)
            .ok()
            .map(|ts| ts.with_timezone(&chrono::Utc));
        return match resume_foreign(session_store, origin, source_ts)
            .await
            .map_err(|err| err.to_string())?
        {
            ForeignResume::Fork {
                state,
                source_newer,
            } => {
                let mut loaded = LoadedSession::from_state(state);
                loaded.source_newer = source_newer;
                Ok(loaded)
            }
            // First open → reconstructed in memory; the turn-save persists it.
            ForeignResume::Fresh(transcript) => Ok(LoadedSession {
                key_id: key_id.to_string(),
                session_id: crate::services::session_import::import_session_id(
                    &origin.cli,
                    &origin.foreign_id,
                ),
                raw_model: model.to_string(),
                messages: to_chat_messages(transcript.messages),
                engine_messages: Some(transcript.engine_messages),
                pristine_import: true,
                source_newer: false,
                import_fidelity: Some(transcript.fidelity),
            }),
        };
    }
    load_resume_session(session_store, preview).await
}

pub(super) async fn load_resume_session(
    session_store: &SessionStore,
    preview: &SessionPreview,
) -> std::result::Result<LoadedSession, String> {
    let session = session_store
        .get_code_session(&preview.session_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "Saved session is no longer available".to_string())?;

    Ok(LoadedSession::from_state(session))
}

/// Preview for the highlighted `/resume` row. A foreign (not-yet-imported)
/// session is previewed straight from its source transcript — its aivo id
/// doesn't exist on disk until it's selected — so highlighting it never errors.
pub(super) async fn load_preview_for(
    session_store: &SessionStore,
    preview: &SessionPreview,
    cap: usize,
) -> std::result::Result<(Vec<ChatMessage>, bool), String> {
    if let Some(origin) = &preview.origin
        && session_store
            .get_code_session(&preview.session_id)
            .await
            .ok()
            .flatten()
            .is_none()
    {
        let transcript = crate::services::session_import::convert_foreign(origin)
            .await
            .map_err(|err| err.to_string())?;
        let mut messages = to_chat_messages(transcript.messages);
        let truncated = messages.len() > cap;
        if truncated {
            messages.drain(..messages.len() - cap);
        }
        return Ok((messages, truncated));
    }
    load_session_preview(session_store, &preview.session_id, cap).await
}

/// The last `cap` messages of one session for the `/resume` preview (+ whether
/// older were dropped).
pub(super) async fn load_session_preview(
    session_store: &SessionStore,
    session_id: &str,
    cap: usize,
) -> std::result::Result<(Vec<ChatMessage>, bool), String> {
    let state = session_store
        .get_code_session(session_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "Saved session is no longer available".to_string())?;
    let mut messages = to_chat_messages(state.messages);
    let truncated = messages.len() > cap;
    if truncated {
        let drop = messages.len() - cap;
        messages.drain(..drop);
    }
    Ok((messages, truncated))
}

/// Per-directory recall view: entries typed in `cwd` plus legacy un-attributed
/// ones (empty `cwd`), oldest-first, capped to the last `MAX_DRAFT_HISTORY`.
pub(super) fn draft_history_view(all: &[DraftHistoryEntry], cwd: &str) -> Vec<String> {
    let mut view: Vec<String> = all
        .iter()
        .filter(|entry| entry.cwd == cwd || entry.cwd.is_empty())
        .map(|entry| entry.text.clone())
        .collect();
    let overflow = view.len().saturating_sub(MAX_DRAFT_HISTORY);
    if overflow > 0 {
        view.drain(..overflow);
    }
    view
}

pub(super) fn load_persisted_draft_history() -> Vec<DraftHistoryEntry> {
    let path = draft_history_path();
    load_persisted_draft_history_from_path(&path)
}

pub(super) fn load_persisted_draft_history_from_path(path: &Path) -> Vec<DraftHistoryEntry> {
    let Ok(data) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(plain) = crate::services::session_store::decrypt(&data) else {
        return Vec::new();
    };

    let mut entries: Vec<DraftHistoryEntry> = plain
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            // A non-JSON line is a legacy untagged prompt → empty `cwd`.
            serde_json::from_str::<DraftHistoryEntry>(line).unwrap_or(DraftHistoryEntry {
                cwd: String::new(),
                text: line.to_owned(),
            })
        })
        .collect();
    let overflow = entries.len().saturating_sub(MAX_DRAFT_HISTORY_TOTAL);
    if overflow > 0 {
        entries.drain(..overflow);
    }
    entries
}

pub(super) fn save_persisted_draft_history(history: &[DraftHistoryEntry]) -> io::Result<()> {
    let path = draft_history_path();
    save_persisted_draft_history_to_path(&path, history)
}

pub(super) fn save_persisted_draft_history_to_path(
    path: &Path,
    history: &[DraftHistoryEntry],
) -> io::Result<()> {
    if history.is_empty() {
        return Ok(());
    }

    let joined = history
        .iter()
        .filter_map(|entry| serde_json::to_string(entry).ok())
        .collect::<Vec<_>>()
        .join("\n");
    let encrypted = crate::services::session_store::encrypt(&joined).map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::services::atomic_write::atomic_write_secure_blocking(path, encrypted.as_bytes())
        .map_err(io::Error::other)
}

pub(super) fn draft_history_path() -> PathBuf {
    crate::services::paths::chat_history(&crate::services::paths::config_dir())
}

pub(super) fn restore_cancelled_submission(
    history: &mut Vec<ChatMessage>,
    draft: &mut String,
    draft_attachments: &mut Vec<MessageAttachment>,
    pending_submit: &mut Option<PendingSubmission>,
) {
    if let Some(submitted) = pending_submit.take()
        && draft.is_empty()
    {
        *draft = submitted.content;
        *draft_attachments = submitted.attachments;
    }

    if history.last().is_some_and(|message| message.role == "user") {
        history.pop();
    }
}

/// Only user/assistant prose makes a readable title/preview. Agent turns also
/// store `tool_call` (JSON) and `tool_result` (raw output) entries — skip those
/// so the resume list shows conversation text, not tool noise.
fn is_conversational_role(role: &str) -> bool {
    matches!(role, "user" | "assistant")
}

pub(crate) fn session_title_from_messages(messages: &[ChatMessage], raw_model: &str) -> String {
    let last_user = messages
        .iter()
        .rev()
        .find(|message| message.role == "user" && !message.content.trim().is_empty())
        .map(|message| first_non_empty_line(&message.content));
    let last_attachment = messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(|m| m.attachments.first().map(|a| a.name.clone()));
    let fallback = messages
        .iter()
        .rev()
        .find(|message| is_conversational_role(&message.role) && !message.content.trim().is_empty())
        .map(|message| first_non_empty_line(&message.content));

    last_user
        .or(last_attachment)
        .or(fallback)
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| raw_model.to_string())
}

pub(crate) fn session_preview_text_from_messages(
    messages: &[ChatMessage],
    raw_model: &str,
) -> String {
    let snippets = messages
        .iter()
        .rev()
        .filter(|message| is_conversational_role(&message.role))
        .filter_map(|message| {
            if !message.content.trim().is_empty() {
                Some(collapse_whitespace(&message.content))
            } else {
                message.attachments.first().map(|a| a.name.clone())
            }
        })
        .take(2)
        .collect::<Vec<_>>();

    let joined = snippets
        .into_iter()
        .rev()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");

    if !joined.is_empty() {
        joined
    } else {
        raw_model.to_string()
    }
}

pub(super) fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn plain_text_from_spans(spans: &[Span<'static>]) -> String {
    let mut plain = String::new();
    for span in spans {
        plain.push_str(span.content.as_ref());
    }
    plain
}

pub(super) fn resume_metadata_values(
    preview: &SessionPreview,
    width: u16,
) -> (String, String, Option<String>) {
    const SEPARATOR_LEN: usize = 3;

    let time_value = format_time_ago_short(&preview.updated_at);
    let key_value = preview.key_name.clone();
    let available = usize::from(width.max(1));

    // Foreign (import) rows carry no aivo model — omit that segment.
    if preview.raw_model.is_empty() {
        return (time_value, key_value, None);
    }

    let mut used = display_width(&time_value) + SEPARATOR_LEN + display_width(&key_value);
    let full_model_len = SEPARATOR_LEN + display_width(&preview.raw_model);

    if used + full_model_len <= available {
        return (time_value, key_value, Some(preview.raw_model.clone()));
    }

    used += SEPARATOR_LEN;
    if used >= available {
        return (time_value, key_value, None);
    }

    let model_width = available.saturating_sub(used) as u16;
    (
        time_value,
        key_value,
        Some(truncate_for_width(&preview.raw_model, model_width.max(1))),
    )
}

pub(super) fn push_resume_metadata_segment(
    spans: &mut Vec<Span<'static>>,
    value: String,
    color: Color,
) {
    if !spans.is_empty() {
        spans.push(Span::styled(" · ", Style::default().fg(FAINT())));
    }
    spans.push(Span::styled(value, Style::default().fg(color)));
}

pub(super) fn resume_metadata_spans(preview: &SessionPreview, width: u16) -> Vec<Span<'static>> {
    let (time_value, key_value, model_value) = resume_metadata_values(preview, width);
    let mut spans = Vec::new();
    push_resume_metadata_segment(&mut spans, time_value, MUTED());
    push_resume_metadata_segment(&mut spans, key_value, MUTED());
    if let Some(model_value) = model_value {
        push_resume_metadata_segment(&mut spans, model_value, MUTED());
    }
    spans
}
