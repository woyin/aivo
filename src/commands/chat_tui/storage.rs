use super::*;

pub(super) async fn load_session_snapshots(
    session_store: &SessionStore,
    key: &ApiKey,
    cwd: &str,
) -> Result<Vec<SessionPreview>> {
    Ok(session_store
        .list_chat_sessions(&key.id, &key.base_url, cwd)
        .await?
        .into_iter()
        .map(|entry| SessionPreview::from_index_entry(entry, key))
        .collect())
}

pub(super) async fn load_resume_snapshots(
    session_store: &SessionStore,
    cwd: &str,
) -> Result<Vec<SessionPreview>> {
    let keys = session_store.get_keys().await?;
    let mut sessions = Vec::new();

    for key in keys {
        sessions.extend(load_session_snapshots(session_store, &key, cwd).await?);
    }

    sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    Ok(sessions)
}

pub(super) async fn load_resume_session(
    session_store: &SessionStore,
    preview: &SessionPreview,
) -> std::result::Result<LoadedSession, String> {
    let session = session_store
        .get_chat_session(&preview.session_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "Saved chat is no longer available".to_string())?;

    LoadedSession::from_state(session).map_err(|err| err.to_string())
}

pub(super) fn load_persisted_draft_history() -> Vec<String> {
    let path = draft_history_path();
    load_persisted_draft_history_from_path(&path)
}

pub(super) fn load_persisted_draft_history_from_path(path: &Path) -> Vec<String> {
    let Ok(data) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(plain) = crate::services::session_store::decrypt(&data) else {
        return Vec::new();
    };

    plain
        .lines()
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub(super) fn save_persisted_draft_history(history: &[String]) -> io::Result<()> {
    let path = draft_history_path();
    save_persisted_draft_history_to_path(&path, history)
}

pub(super) fn save_persisted_draft_history_to_path(
    path: &Path,
    history: &[String],
) -> io::Result<()> {
    if history.is_empty() {
        return Ok(());
    }

    let joined = history.join("\n");
    let encrypted = crate::services::session_store::encrypt(&joined).map_err(io::Error::other)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    crate::services::atomic_write::atomic_write_secure_blocking(path, encrypted.as_bytes())
        .map_err(io::Error::other)
}

pub(super) fn draft_history_path() -> PathBuf {
    crate::services::system_env::home_dir()
        .map(|path| path.join(".config").join("aivo").join("chat_history"))
        .unwrap_or_else(|| std::env::temp_dir().join("aivo").join("chat_history"))
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
        .find(|message| !message.content.trim().is_empty())
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
        spans.push(Span::styled(" · ", Style::default().fg(FAINT)));
    }
    spans.push(Span::styled(value, Style::default().fg(color)));
}

pub(super) fn resume_metadata_spans(preview: &SessionPreview, width: u16) -> Vec<Span<'static>> {
    let (time_value, key_value, model_value) = resume_metadata_values(preview, width);
    let mut spans = Vec::new();
    push_resume_metadata_segment(&mut spans, time_value, ACCENT);
    push_resume_metadata_segment(&mut spans, key_value, USER);
    if let Some(model_value) = model_value {
        push_resume_metadata_segment(&mut spans, model_value, ASSISTANT);
    }
    spans
}
