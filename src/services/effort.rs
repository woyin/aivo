//! Cross-protocol normalization of reasoning effort / extended thinking levels.
//!
//! Each provider expresses "how much should the model think" differently:
//!
//! - **OpenAI** (Chat Completions + Responses): `reasoning_effort` /
//!   `reasoning.effort` ∈ {`"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
//!   `"xhigh"`}. GPT-5.1 defaults to `"none"`. GPT-5.4 explicitly disallows
//!   tool calling in Chat Completions when `reasoning_effort` is `"none"`.
//! - **Anthropic** (Messages API): `thinking: { type: "enabled", budget_tokens }`
//!   plus the newer `output_config.effort` ∈ {`"low"`, `"medium"`, `"high"`,
//!   `"max"`}.
//! - **Gemini** (3.x): `generationConfig.thinkingConfig.thinking_level`
//!   ∈ {`"low"`, `"medium"`, `"high"`}. Gemini 2.5 used `thinkingBudget`
//!   (numeric); Gemini 3 rejects that key.
//!
//! Without this module each bridge would either drop the field (current bug)
//! or duplicate ad-hoc mappings (the partial Anthropic→OpenAI mapping in
//! `anthropic_to_openai_router.rs:652` was the existing example: it always
//! collapsed `thinking: enabled` to `reasoning_effort: "high"` regardless of
//! `budget_tokens`, hiding intent from the upstream).

use serde_json::{Value, json};

/// Canonical effort tier used as the bridge's lingua franca.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CanonicalEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Max,
}

impl CanonicalEffort {
    pub(crate) fn from_openai_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "none" => Some(Self::None),
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "xhigh" | "max" => Some(Self::Max),
            _ => None,
        }
    }

    pub(crate) fn from_anthropic_effort_str(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// Approximate canonical tier from an Anthropic `thinking.budget_tokens`
    /// integer. Boundaries chosen to match Anthropic's rough cost / latency
    /// curves: < 2048 ≈ low, < 8192 ≈ medium, < 24576 ≈ high, otherwise max.
    pub(crate) fn from_anthropic_budget_tokens(budget: u64) -> Self {
        if budget == 0 {
            Self::None
        } else if budget < 2048 {
            Self::Low
        } else if budget < 8192 {
            Self::Medium
        } else if budget < 24576 {
            Self::High
        } else {
            Self::Max
        }
    }

    /// Map to OpenAI `reasoning_effort` — the field accepted by both Chat
    /// Completions and Responses APIs. `Max` becomes `"xhigh"` because that
    /// is what GPT-5.4+ recognizes; older models that don't accept `xhigh`
    /// will fall through to provider-specific clamping.
    pub(crate) fn to_openai_effort(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "xhigh",
        }
    }

    /// Map to Anthropic `output_config.effort`. `None` and `Minimal` fold
    /// down to `"low"` because Anthropic has no equivalent below low, and
    /// dropping the field entirely would re-enable Anthropic's default.
    pub(crate) fn to_anthropic_effort(self) -> &'static str {
        match self {
            Self::None | Self::Minimal | Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    /// Approximate `thinking.budget_tokens` for callers that need the
    /// numeric form rather than `output_config.effort`. Matches the
    /// boundaries in [`from_anthropic_budget_tokens`].
    pub(crate) fn to_anthropic_budget_tokens(self) -> Option<u64> {
        match self {
            Self::None => None, // disable thinking
            Self::Minimal | Self::Low => Some(1024),
            Self::Medium => Some(4096),
            Self::High => Some(16384),
            Self::Max => Some(32000),
        }
    }

    /// Map to Gemini 3 `thinking_level`. Gemini has only three tiers, so
    /// `Max` collapses to `high`.
    pub(crate) fn to_gemini_thinking_level(self) -> Option<&'static str> {
        match self {
            Self::None | Self::Minimal => None, // omit to disable thinking
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High | Self::Max => Some("high"),
        }
    }
}

/// Extract a canonical effort from an OpenAI request body. Looks at:
/// 1. `reasoning_effort` (Chat Completions)
/// 2. `reasoning.effort` (Responses API)
pub(crate) fn extract_openai_effort(body: &Value) -> Option<CanonicalEffort> {
    if let Some(value) = body.get("reasoning_effort").and_then(|v| v.as_str())
        && let Some(effort) = CanonicalEffort::from_openai_str(value)
    {
        return Some(effort);
    }
    body.get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|v| v.as_str())
        .and_then(CanonicalEffort::from_openai_str)
}

/// Extract a canonical effort from an Anthropic request body. Looks at
/// `thinking.budget_tokens` first (most precise), then `output_config.effort`.
pub(crate) fn extract_anthropic_effort(body: &Value) -> Option<CanonicalEffort> {
    if let Some(thinking) = body.get("thinking")
        && thinking.get("type").and_then(|t| t.as_str()) == Some("enabled")
    {
        if let Some(budget) = thinking.get("budget_tokens").and_then(|v| v.as_u64()) {
            return Some(CanonicalEffort::from_anthropic_budget_tokens(budget));
        }
        return Some(CanonicalEffort::High);
    }
    body.get("output_config")
        .and_then(|c| c.get("effort"))
        .and_then(|v| v.as_str())
        .and_then(CanonicalEffort::from_anthropic_effort_str)
}

/// True if the OpenAI request would be rejected upstream because GPT-5.4+
/// disallows tool use in Chat Completions when `reasoning_effort` is `"none"`.
/// Returns false for Responses API (uses `reasoning.effort` and lifts the
/// restriction).
pub(crate) fn gpt5_chat_completions_rejects_tools_with_none_reasoning(body: &Value) -> bool {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);
    if !(name_only.starts_with("gpt-5") || name_only.contains("codex")) {
        return false;
    }
    let has_tools = body
        .get("tools")
        .and_then(|t| t.as_array())
        .is_some_and(|t| !t.is_empty());
    if !has_tools {
        return false;
    }
    matches!(extract_openai_effort(body), Some(CanonicalEffort::None))
}

/// Build an Anthropic `thinking` config object from a canonical effort.
/// Returns `None` for `CanonicalEffort::None` (caller should set
/// `thinking: { type: "disabled" }` or omit entirely depending on context).
pub(crate) fn anthropic_thinking_config(effort: CanonicalEffort) -> Option<Value> {
    let budget = effort.to_anthropic_budget_tokens()?;
    Some(json!({
        "type": "enabled",
        "budget_tokens": budget,
    }))
}

/// Build a Gemini `thinkingConfig` from a canonical effort.
pub(crate) fn gemini_thinking_config(effort: CanonicalEffort) -> Option<Value> {
    let level = effort.to_gemini_thinking_level()?;
    Some(json!({ "thinking_level": level }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_through_openai_strings() {
        for tier in [
            CanonicalEffort::None,
            CanonicalEffort::Minimal,
            CanonicalEffort::Low,
            CanonicalEffort::Medium,
            CanonicalEffort::High,
            CanonicalEffort::Max,
        ] {
            let s = tier.to_openai_effort();
            let back = CanonicalEffort::from_openai_str(s).expect("known tier round-trips");
            assert_eq!(back, tier, "round-trip failed for {tier:?} via {s}");
        }
    }

    #[test]
    fn anthropic_budget_to_canonical_boundaries() {
        assert_eq!(
            CanonicalEffort::from_anthropic_budget_tokens(0),
            CanonicalEffort::None
        );
        assert_eq!(
            CanonicalEffort::from_anthropic_budget_tokens(1024),
            CanonicalEffort::Low
        );
        assert_eq!(
            CanonicalEffort::from_anthropic_budget_tokens(4096),
            CanonicalEffort::Medium
        );
        assert_eq!(
            CanonicalEffort::from_anthropic_budget_tokens(16384),
            CanonicalEffort::High
        );
        assert_eq!(
            CanonicalEffort::from_anthropic_budget_tokens(64000),
            CanonicalEffort::Max
        );
    }

    #[test]
    fn extract_openai_effort_prefers_chat_completions_field_over_responses_field() {
        let body = json!({
            "reasoning_effort": "low",
            "reasoning": { "effort": "high" }
        });
        assert_eq!(extract_openai_effort(&body), Some(CanonicalEffort::Low));
    }

    #[test]
    fn extract_openai_effort_falls_back_to_responses_api_field() {
        let body = json!({ "reasoning": { "effort": "xhigh" } });
        assert_eq!(extract_openai_effort(&body), Some(CanonicalEffort::Max));
    }

    #[test]
    fn extract_openai_effort_unknown_value_returns_none() {
        let body = json!({ "reasoning_effort": "ludicrous" });
        assert_eq!(extract_openai_effort(&body), None);
    }

    #[test]
    fn extract_anthropic_effort_uses_budget_tokens_when_thinking_enabled() {
        let body = json!({
            "thinking": { "type": "enabled", "budget_tokens": 8192 }
        });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::High));
    }

    #[test]
    fn extract_anthropic_effort_falls_back_to_output_config_effort() {
        let body = json!({ "output_config": { "effort": "max" } });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::Max));
    }

    #[test]
    fn gpt5_with_tools_and_reasoning_none_is_flagged() {
        let body = json!({
            "model": "gpt-5.4",
            "reasoning_effort": "none",
            "tools": [{"type": "function", "function": {"name": "f"}}]
        });
        assert!(gpt5_chat_completions_rejects_tools_with_none_reasoning(
            &body
        ));
    }

    #[test]
    fn gpt5_with_tools_and_reasoning_low_is_not_flagged() {
        let body = json!({
            "model": "gpt-5.4",
            "reasoning_effort": "low",
            "tools": [{"type": "function", "function": {"name": "f"}}]
        });
        assert!(!gpt5_chat_completions_rejects_tools_with_none_reasoning(
            &body
        ));
    }

    #[test]
    fn non_gpt5_models_are_never_flagged_for_tools_with_none_reasoning() {
        let body = json!({
            "model": "gpt-4o",
            "reasoning_effort": "none",
            "tools": [{"type": "function", "function": {"name": "f"}}]
        });
        assert!(!gpt5_chat_completions_rejects_tools_with_none_reasoning(
            &body
        ));
    }

    #[test]
    fn anthropic_thinking_config_for_none_returns_none() {
        // Caller is responsible for picking the right disabled-form field.
        assert!(anthropic_thinking_config(CanonicalEffort::None).is_none());
    }

    #[test]
    fn gemini_thinking_level_for_minimal_returns_none() {
        // Below "low" Gemini has no equivalent; omit the field to get default.
        assert!(gemini_thinking_config(CanonicalEffort::Minimal).is_none());
    }

    #[test]
    fn gemini_thinking_level_collapses_max_to_high() {
        let cfg = gemini_thinking_config(CanonicalEffort::Max).expect("max maps");
        assert_eq!(cfg["thinking_level"], "high");
    }
}
