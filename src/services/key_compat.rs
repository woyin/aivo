//! Key-compatibility rules for the interactive key picker.
//!
//! Different commands accept different key types: `aivo run claude` can use
//! Claude Code OAuth credentials but not Codex/Gemini OAuth, while `aivo chat`
//! can't use any OAuth credential. `KeyCompatContext` lets the picker annotate
//! incompatible keys with a short reason instead of hiding them, so the user
//! sees *why* a key isn't available and can pick another without leaving the
//! command.

use crate::services::ai_launcher::AIToolType;
use crate::services::cursor_acp::is_cursor_acp_base;
use crate::services::provider_profile::{is_copilot_base, is_ollama_base};
use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
use crate::services::session_store::ApiKey;

/// Describes which command is asking for a key, so the picker can annotate
/// keys that won't work in that context.
#[derive(Debug, Clone, Copy)]
pub enum KeyCompatContext {
    /// No restriction — all keys are selectable.
    None,
    /// `aivo run <tool>` — only the matching OAuth type is allowed; other
    /// OAuth keys are incompatible.
    Tool(AIToolType),
    /// `aivo chat` — all OAuth keys are incompatible.
    Chat,
    /// `aivo image` — OAuth, Copilot, Ollama, and the starter key are all
    /// incompatible (none expose an image-generation endpoint).
    Image,
    /// `aivo audio` — same incompatibility set as image; no provider in
    /// this group exposes a TTS endpoint.
    Audio,
    /// `aivo video` — same incompatibility set as image and audio; no
    /// provider in this group exposes a video-generation endpoint.
    Video,
}

impl KeyCompatContext {
    /// Returns `Some(reason)` if `key` cannot be used under this context, or
    /// `None` if the key is compatible.
    pub fn incompat_reason(&self, key: &ApiKey) -> Option<&'static str> {
        match self {
            KeyCompatContext::None => None,
            KeyCompatContext::Tool(tool) => tool.oauth_incompat_reason(key),
            KeyCompatContext::Chat => key.oauth_run_requirement(),
            KeyCompatContext::Image => image_incompat_reason(key),
            KeyCompatContext::Audio => audio_incompat_reason(key),
            KeyCompatContext::Video => video_incompat_reason(key),
        }
    }

    /// Builds one annotation per key for `FuzzySelect::annotations`.
    pub fn annotations_for(&self, keys: &[ApiKey]) -> Vec<Option<String>> {
        keys.iter()
            .map(|k| self.incompat_reason(k).map(str::to_string))
            .collect()
    }
}

/// Image generation rejects any key that can't hit OpenAI-compatible
/// `/v1/images/generations`:
///   * OAuth bundles (Claude / Codex / Gemini) — no REST image endpoint;
///   * Copilot — no image endpoint;
///   * Ollama — no image endpoint;
///   * aivo-starter — gateway-limited, no image capacity;
///   * Anthropic protocol keys — `image_gen::generate` hard-fails (no image API).
fn image_incompat_reason(key: &ApiKey) -> Option<&'static str> {
    const NO_IMAGE_GEN: &str = "no image generation";
    if key.is_any_oauth()
        || is_copilot_base(&key.base_url)
        || is_cursor_acp_base(&key.base_url)
        || is_ollama_base(&key.base_url)
    {
        return Some(NO_IMAGE_GEN);
    }
    if key.base_url == crate::constants::AIVO_STARTER_SENTINEL {
        return Some("starter key: no image gen");
    }
    match detect_provider_protocol(&key.base_url) {
        ProviderProtocol::Anthropic => Some("Anthropic has no image API"),
        _ => None,
    }
}

/// TTS rejects the same set as image generation: every provider in this
/// group either lacks a `/v1/audio/speech` (or equivalent) endpoint, or
/// the protocol family doesn't expose audio generation at all.
fn audio_incompat_reason(key: &ApiKey) -> Option<&'static str> {
    const NO_TTS: &str = "no TTS";
    if key.is_any_oauth()
        || is_copilot_base(&key.base_url)
        || is_cursor_acp_base(&key.base_url)
        || is_ollama_base(&key.base_url)
    {
        return Some(NO_TTS);
    }
    if key.base_url == crate::constants::AIVO_STARTER_SENTINEL {
        return Some("starter key: no TTS");
    }
    match detect_provider_protocol(&key.base_url) {
        ProviderProtocol::Anthropic => Some("Anthropic has no TTS API"),
        _ => None,
    }
}

/// Video generation rejects the same set as image and audio: no OAuth-bundled
/// provider, Copilot, Ollama, the starter key, or Anthropic protocol exposes
/// a `/v1/videos`-style endpoint.
fn video_incompat_reason(key: &ApiKey) -> Option<&'static str> {
    const NO_VIDEO: &str = "no video generation";
    if key.is_any_oauth()
        || is_copilot_base(&key.base_url)
        || is_cursor_acp_base(&key.base_url)
        || is_ollama_base(&key.base_url)
    {
        return Some(NO_VIDEO);
    }
    if key.base_url == crate::constants::AIVO_STARTER_SENTINEL {
        return Some("starter key: no video gen");
    }
    match detect_provider_protocol(&key.base_url) {
        ProviderProtocol::Anthropic => Some("Anthropic has no video API"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
    use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
    use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;

    fn make_key(name: &str, base_url: &str) -> ApiKey {
        ApiKey::new_with_protocol(
            "id".into(),
            name.into(),
            base_url.into(),
            None,
            "secret".into(),
        )
    }

    #[test]
    fn chat_disables_any_oauth() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);
        let regular = make_key("openrouter", "https://openrouter.ai/api/v1");

        let ctx = KeyCompatContext::Chat;
        assert_eq!(ctx.incompat_reason(&claude), Some("needs `aivo claude`"));
        assert_eq!(ctx.incompat_reason(&codex), Some("needs `aivo codex`"));
        assert_eq!(ctx.incompat_reason(&gemini), Some("needs `aivo gemini`"));
        assert!(ctx.incompat_reason(&regular).is_none());
    }

    #[test]
    fn tool_disables_mismatched_oauth_only() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let regular = make_key("openrouter", "https://openrouter.ai/api/v1");

        let claude_ctx = KeyCompatContext::Tool(AIToolType::Claude);
        assert!(claude_ctx.incompat_reason(&claude).is_none());
        assert!(claude_ctx.incompat_reason(&codex).is_some());
        assert!(claude_ctx.incompat_reason(&regular).is_none());

        let codex_ctx = KeyCompatContext::Tool(AIToolType::Codex);
        assert!(codex_ctx.incompat_reason(&claude).is_some());
        assert!(codex_ctx.incompat_reason(&codex).is_none());
        assert!(codex_ctx.incompat_reason(&regular).is_none());

        let codex_app_ctx = KeyCompatContext::Tool(AIToolType::CodexApp);
        assert!(codex_app_ctx.incompat_reason(&claude).is_some());
        assert!(codex_app_ctx.incompat_reason(&codex).is_none());
        assert!(codex_app_ctx.incompat_reason(&regular).is_none());
    }

    #[test]
    fn opencode_pi_and_amp_disable_all_oauth() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);

        for tool in [AIToolType::Opencode, AIToolType::Pi, AIToolType::Amp] {
            let ctx = KeyCompatContext::Tool(tool);
            assert!(ctx.incompat_reason(&claude).is_some());
            assert!(ctx.incompat_reason(&codex).is_some());
            assert!(ctx.incompat_reason(&gemini).is_some());
        }
    }

    #[test]
    fn none_context_disables_nothing() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        assert!(KeyCompatContext::None.incompat_reason(&claude).is_none());
    }

    #[test]
    fn image_rejects_oauth_keys() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);

        let ctx = KeyCompatContext::Image;
        assert_eq!(ctx.incompat_reason(&claude), Some("no image generation"));
        assert_eq!(ctx.incompat_reason(&codex), Some("no image generation"));
        assert_eq!(ctx.incompat_reason(&gemini), Some("no image generation"));
    }

    #[test]
    fn image_rejects_copilot_and_ollama() {
        let copilot = make_key("copilot", "copilot");
        let cursor = make_key("cursor", "cursor");
        let ollama = make_key("ollama", "ollama");

        let ctx = KeyCompatContext::Image;
        assert_eq!(ctx.incompat_reason(&copilot), Some("no image generation"));
        assert_eq!(ctx.incompat_reason(&cursor), Some("no image generation"));
        assert_eq!(ctx.incompat_reason(&ollama), Some("no image generation"));
    }

    #[test]
    fn cursor_is_chat_and_tool_compatible_but_not_media() {
        let cursor = make_key("cursor", "cursor");
        assert!(KeyCompatContext::Chat.incompat_reason(&cursor).is_none());
        for tool in AIToolType::all() {
            assert!(
                KeyCompatContext::Tool(*tool)
                    .incompat_reason(&cursor)
                    .is_none(),
                "{tool:?}"
            );
        }
        assert_eq!(
            KeyCompatContext::Image.incompat_reason(&cursor),
            Some("no image generation")
        );
        assert_eq!(
            KeyCompatContext::Audio.incompat_reason(&cursor),
            Some("no TTS")
        );
        assert_eq!(
            KeyCompatContext::Video.incompat_reason(&cursor),
            Some("no video generation")
        );
    }

    #[test]
    fn image_rejects_starter_key() {
        let starter = make_key("starter", crate::constants::AIVO_STARTER_SENTINEL);
        let ctx = KeyCompatContext::Image;
        assert_eq!(
            ctx.incompat_reason(&starter),
            Some("starter key: no image gen")
        );
    }

    #[test]
    fn image_rejects_anthropic_protocol_but_accepts_google() {
        let anthropic = make_key("anthropic", "https://api.anthropic.com/v1");
        let google = make_key("google", "https://generativelanguage.googleapis.com/v1beta");

        let ctx = KeyCompatContext::Image;
        assert_eq!(
            ctx.incompat_reason(&anthropic),
            Some("Anthropic has no image API")
        );
        // After Google image generation landed, Google REST keys are compatible.
        assert!(ctx.incompat_reason(&google).is_none());
    }

    #[test]
    fn image_accepts_openai_compatible_keys() {
        let openai = make_key("openai", "https://api.openai.com/v1");
        let openrouter = make_key("openrouter", "https://openrouter.ai/api/v1");
        let xai = make_key("xai", "https://api.x.ai/v1");

        let ctx = KeyCompatContext::Image;
        assert!(ctx.incompat_reason(&openai).is_none());
        assert!(ctx.incompat_reason(&openrouter).is_none());
        assert!(ctx.incompat_reason(&xai).is_none());
    }
}
