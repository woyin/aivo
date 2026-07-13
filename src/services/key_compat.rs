//! Key-compatibility rules for the interactive key picker.
//!
//! Different commands accept different key types: `aivo run claude` can use
//! Claude Code OAuth credentials but not Codex/Gemini OAuth, while `aivo code`
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
    /// `aivo code` — all OAuth keys are incompatible.
    Chat,
    /// A plugin handoff — aivo serves the key over a loopback proxy (the
    /// `endpoint` cap). OAuth keys are native-agent-only. Cursor is available
    /// only to `type: coding-agent` plugins, which use the Cursor ACP router.
    Plugin { allow_cursor: bool },
}

impl KeyCompatContext {
    /// Returns `Some(reason)` if `key` cannot be used under this context, or
    /// `None` if the key is compatible.
    pub fn incompat_reason(&self, key: &ApiKey) -> Option<&'static str> {
        match self {
            KeyCompatContext::None => None,
            KeyCompatContext::Tool(tool) => tool.oauth_incompat_reason(key),
            KeyCompatContext::Chat => key.oauth_run_requirement(),
            KeyCompatContext::Plugin { allow_cursor } => {
                if key.is_cursor_acp() && !allow_cursor {
                    Some("needs coding-agent plugin")
                } else {
                    key.oauth_run_requirement()
                }
            }
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
    fn chat_disables_launch_only_oauth_but_allows_provider_oauth() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);
        let regular = make_key("openrouter", "https://openrouter.ai/api/v1");

        let ctx = KeyCompatContext::Chat;
        assert_eq!(ctx.incompat_reason(&claude), Some("needs `aivo claude`"));
        // Codex is a provider credential — `aivo code` accepts it like grok.
        assert!(ctx.incompat_reason(&codex).is_none());
        assert_eq!(
            ctx.incompat_reason(&gemini),
            Some("Gemini sign-in removed — re-add with an API key")
        );
        assert!(ctx.incompat_reason(&regular).is_none());
    }

    #[test]
    fn tool_disables_mismatched_oauth_only() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let regular = make_key("openrouter", "https://openrouter.ai/api/v1");

        // Codex is a provider credential: usable with any tool.
        let claude_ctx = KeyCompatContext::Tool(AIToolType::Claude);
        assert!(claude_ctx.incompat_reason(&claude).is_none());
        assert!(claude_ctx.incompat_reason(&codex).is_none());
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
    fn opencode_and_pi_disable_launch_only_oauth() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        let codex = make_key("codex", CODEX_OAUTH_SENTINEL);
        let gemini = make_key("gemini", GEMINI_OAUTH_SENTINEL);

        for tool in [AIToolType::Opencode, AIToolType::Pi] {
            let ctx = KeyCompatContext::Tool(tool);
            assert!(ctx.incompat_reason(&claude).is_some());
            assert!(ctx.incompat_reason(&codex).is_none());
            assert!(ctx.incompat_reason(&gemini).is_some());
        }
    }

    #[test]
    fn none_context_disables_nothing() {
        let claude = make_key("claude", CLAUDE_OAUTH_SENTINEL);
        assert!(KeyCompatContext::None.incompat_reason(&claude).is_none());
    }

    #[test]
    fn plugin_context_disables_oauth_and_cursor_when_not_allowed() {
        let ctx = KeyCompatContext::Plugin {
            allow_cursor: false,
        };
        // Claude/Gemini OAuth are native-agent-only; codex (provider) is allowed.
        assert!(
            ctx.incompat_reason(&make_key("cl", CLAUDE_OAUTH_SENTINEL))
                .is_some()
        );
        assert!(
            ctx.incompat_reason(&make_key("cx", CODEX_OAUTH_SENTINEL))
                .is_none()
        );
        assert!(
            ctx.incompat_reason(&make_key("gm", GEMINI_OAUTH_SENTINEL))
                .is_some()
        );
        assert_eq!(
            ctx.incompat_reason(&make_key("cur", "cursor")),
            Some("needs coding-agent plugin")
        );
        // Plain REST keys are fine.
        assert!(
            ctx.incompat_reason(&make_key("o", "https://openrouter.ai/api/v1"))
                .is_none()
        );
    }

    #[test]
    fn plugin_context_can_allow_cursor_for_coding_agents() {
        let ctx = KeyCompatContext::Plugin { allow_cursor: true };
        assert!(ctx.incompat_reason(&make_key("cur", "cursor")).is_none());
        assert!(
            ctx.incompat_reason(&make_key("cl", CLAUDE_OAUTH_SENTINEL))
                .is_some()
        );
    }

    #[test]
    fn cursor_is_chat_and_tool_compatible() {
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
    }
}
