//! Key-compatibility rules for the interactive key picker.
//!
//! Different commands accept different key types: `aivo run claude` can use
//! Claude Code OAuth credentials but not Codex/Gemini OAuth, while `aivo chat`
//! can't use any OAuth credential. `KeyCompatContext` lets the picker annotate
//! incompatible keys with a short reason instead of hiding them, so the user
//! sees *why* a key isn't available and can pick another without leaving the
//! command.

use crate::services::ai_launcher::AIToolType;
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
}

impl KeyCompatContext {
    /// Returns `Some(reason)` if `key` cannot be used under this context, or
    /// `None` if the key is compatible.
    pub fn incompat_reason(&self, key: &ApiKey) -> Option<&'static str> {
        match self {
            KeyCompatContext::None => None,
            KeyCompatContext::Tool(tool) => tool.oauth_incompat_reason(key),
            KeyCompatContext::Chat => key.oauth_run_requirement(),
        }
    }

    /// Builds one annotation per key for `FuzzySelect::annotations`.
    pub fn annotations_for(&self, keys: &[ApiKey]) -> Vec<Option<String>> {
        keys.iter()
            .map(|k| self.incompat_reason(k).map(str::to_string))
            .collect()
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
        assert_eq!(
            ctx.incompat_reason(&claude),
            Some("needs `aivo run claude`")
        );
        assert_eq!(ctx.incompat_reason(&codex), Some("needs `aivo run codex`"));
        assert_eq!(
            ctx.incompat_reason(&gemini),
            Some("needs `aivo run gemini`")
        );
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
    }

    #[test]
    fn opencode_and_pi_disable_all_oauth() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);

        for tool in [AIToolType::Opencode, AIToolType::Pi] {
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
}
