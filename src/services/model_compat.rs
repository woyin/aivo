//! Model picker compatibility annotations.
//!
//! Parallel to `key_compat.rs`: given a model id, tells the picker whether
//! the model is usable for the current command and why not. Pickers render
//! incompatible models as disabled rows with a dim suffix instead of
//! hiding them, so users see the full provider catalog and understand why
//! a given entry isn't selectable.
//!
//! Only the text-chat direction has reliable markers (embedding / TTS /
//! whisper / dall-e / `-image` suffix). The inverse direction — "is this
//! NOT an image model?" — requires guessing at chat-model families and
//! produced more false positives than value, so the image picker shows
//! everything and lets the user decide.

/// Short reason string when `model_id` can't be used from a text-input tool
/// (chat, claude, codex, gemini, pi, opencode), or `None` when it's usable.
/// The string renders as a dim suffix on the disabled picker row.
pub fn text_chat_incompat_reason(model_id: &str) -> Option<&'static str> {
    let lower = model_id.to_ascii_lowercase();
    if lower.contains("embed") {
        return Some("embeddings");
    }
    if lower.starts_with("tts-") {
        return Some("text-to-speech");
    }
    if lower.starts_with("whisper-") {
        return Some("speech-to-text");
    }
    if lower.starts_with("dall-e") || lower.contains("-image") {
        // Catalog evidence beats the name pattern: a `g`-flagged model chats
        // (gemini-2.5-flash-image); only image models with no such evidence stay out.
        if crate::services::model_metadata::model_generates_images(model_id) {
            return None;
        }
        return Some("image generation");
    }
    None
}

/// Precomputes the annotation vector for text-chat pickers. `Some(reason)`
/// entries disable the corresponding row.
pub fn text_chat_annotations(models: &[String]) -> Vec<Option<String>> {
    models
        .iter()
        .map(|m| text_chat_incompat_reason(m).map(String::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disables_image_audio_embedding() {
        assert_eq!(
            text_chat_incompat_reason("dall-e-3"),
            Some("image generation")
        );
        assert_eq!(
            text_chat_incompat_reason("gpt-image-1"),
            Some("image generation")
        );
        assert_eq!(
            text_chat_incompat_reason("grok-2-image"),
            Some("image generation")
        );
        assert_eq!(
            text_chat_incompat_reason("grok-imagine-image"),
            Some("image generation")
        );
        // Catalog-confirmed image-output chat models are exempt from the `-image` name rule.
        assert!(text_chat_incompat_reason("gemini-2.5-flash-image").is_none());
        assert!(text_chat_incompat_reason("google/gemini-2.5-flash-image").is_none());
        assert_eq!(text_chat_incompat_reason("tts-1"), Some("text-to-speech"));
        assert_eq!(
            text_chat_incompat_reason("whisper-1"),
            Some("speech-to-text")
        );
        assert_eq!(
            text_chat_incompat_reason("text-embedding-3-small"),
            Some("embeddings")
        );
    }

    #[test]
    fn enables_chat_models() {
        assert!(text_chat_incompat_reason("gpt-4o").is_none());
        assert!(text_chat_incompat_reason("claude-sonnet-4-6").is_none());
        assert!(text_chat_incompat_reason("gemini-2.0-flash").is_none());
        assert!(text_chat_incompat_reason("o3-mini").is_none());
        // Preserved quirk from is_text_chat_model — audio-preview chat model.
        assert!(text_chat_incompat_reason("gpt-4o-audio-preview").is_none());
    }

    #[test]
    fn annotations_preserve_order() {
        let models = vec![
            "gpt-4o".to_string(),
            "dall-e-3".to_string(),
            "claude-sonnet-4-6".to_string(),
        ];
        let ann = text_chat_annotations(&models);
        assert_eq!(ann.len(), 3);
        assert!(ann[0].is_none());
        assert_eq!(ann[1].as_deref(), Some("image generation"));
        assert!(ann[2].is_none());
    }
}
