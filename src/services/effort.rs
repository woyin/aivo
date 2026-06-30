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
//!   `"xhigh"`, `"max"`}.
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
            // Anthropic accepts `xhigh` as the deep-effort tier on Opus 4.7
            // adaptive thinking; treat it as `max` so callers that bridge
            // to OpenAI emit `reasoning_effort: "xhigh"` instead of
            // collapsing the intent down to `high`.
            "xhigh" | "max" => Some(Self::Max),
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

    /// Map to a Gemini 2.5 `thinkingBudget`. Stays inside every 2.5 variant's
    /// accepted band (flash cap 24576, pro 128–32768). `None`/`Minimal` omit it,
    /// mirroring [`Self::to_gemini_thinking_level`].
    pub(crate) fn to_gemini_thinking_budget(self) -> Option<u64> {
        match self {
            Self::None | Self::Minimal => None,
            Self::Low => Some(4096),
            Self::Medium => Some(8192),
            Self::High => Some(16384),
            Self::Max => Some(24576),
        }
    }
}

/// True for Gemini 3+, which takes `thinkingConfig.thinking_level`; 2.5 and
/// earlier take numeric `thinkingBudget` and 400 on `thinking_level`. Unknown
/// ids → false (the wider-supported budget surface).
pub(crate) fn gemini_uses_thinking_level(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    let Some(rest) = name.strip_prefix("gemini-") else {
        return false;
    };
    let major: u32 = rest
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or(0);
    major >= 3
}

/// Normalize for matching: lowercase, drop `/` and platform prefixes (keep from
/// `claude-`), dots→dashes (so OpenRouter `claude-opus-4.7` == `claude-opus-4-7`).
fn normalize_claude_model(model: &str) -> String {
    let lower = model.to_ascii_lowercase();
    let bare = lower.rsplit('/').next().unwrap_or(&lower);
    let bare = bare.find("claude-").map(|i| &bare[i..]).unwrap_or(bare);
    bare.replace('.', "-")
}

/// `(major, minor)` of a normalized `claude-opus-…` id; a trailing date suffix
/// is ignored (only the first two numeric components are read).
fn claude_opus_version(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("claude-opus-")?;
    let mut nums = rest.split('-').filter_map(|s| s.parse::<u32>().ok());
    Some((nums.next().unwrap_or(0), nums.next().unwrap_or(0)))
}

/// `prefix` match at a component boundary (so `claude-sonnet-4-6` ≠ `…-4-60`).
fn claude_prefix_at_boundary(name: &str, prefix: &str) -> bool {
    name.starts_with(prefix) && matches!(name.as_bytes().get(prefix.len()), None | Some(b'-'))
}

/// Reject `thinking.budget_tokens` (400) → must use adaptive: Fable/Mythos and
/// Opus 4.7+. Single source of truth; `ThinkingNormalizationPatch` delegates here.
pub(crate) fn anthropic_thinking_uses_adaptive(model: &str) -> bool {
    let name = normalize_claude_model(model);
    if name.contains("fable") || name.contains("mythos") {
        return true;
    }
    matches!(claude_opus_version(&name), Some((major, minor)) if major > 4 || (major == 4 && minor >= 7))
}

/// Accept `thinking:{type:"adaptive"}` — Claude 4.6+ (pre-4.6 400s on it).
pub(crate) fn anthropic_supports_adaptive_thinking(model: &str) -> bool {
    let name = normalize_claude_model(model);
    if name.contains("fable") || name.contains("mythos") {
        return true;
    }
    if claude_prefix_at_boundary(&name, "claude-sonnet-4-6") {
        return true;
    }
    matches!(claude_opus_version(&name), Some((major, minor)) if major > 4 || (major == 4 && minor >= 6))
}

/// Accept `output_config.effort` — Fable/Mythos, Sonnet 4.6, Opus 4.5+.
pub(crate) fn anthropic_supports_output_effort(model: &str) -> bool {
    let name = normalize_claude_model(model);
    if name.contains("fable") || name.contains("mythos") {
        return true;
    }
    if claude_prefix_at_boundary(&name, "claude-sonnet-4-6") {
        return true;
    }
    matches!(claude_opus_version(&name), Some((major, minor)) if major > 4 || (major == 4 && minor >= 5))
}

/// Reject `thinking:{type:"disabled"}` (400) → omit instead: Fable/Mythos.
pub(crate) fn anthropic_rejects_disabled_thinking(model: &str) -> bool {
    let name = normalize_claude_model(model);
    name.contains("fable") || name.contains("mythos")
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

/// Extract a canonical effort from an Anthropic request body. For
/// `thinking.type` ∈ {`"enabled"`, `"adaptive"`}, prefers
/// `thinking.budget_tokens` (the precise, manual control), then
/// `output_config.effort` (the primary control for adaptive thinking),
/// then falls back to `High` so an explicit thinking request isn't
/// silently downgraded to Anthropic's default. When thinking is absent or
/// disabled, falls through to `output_config.effort` alone.
pub(crate) fn extract_anthropic_effort(body: &Value) -> Option<CanonicalEffort> {
    let output_config_effort = body
        .get("output_config")
        .and_then(|c| c.get("effort"))
        .and_then(|v| v.as_str())
        .and_then(CanonicalEffort::from_anthropic_effort_str);

    let thinking_type = body
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|t| t.as_str());

    if matches!(thinking_type, Some("enabled" | "adaptive")) {
        if let Some(budget) = body
            .get("thinking")
            .and_then(|t| t.get("budget_tokens"))
            .and_then(|v| v.as_u64())
        {
            return Some(CanonicalEffort::from_anthropic_budget_tokens(budget));
        }
        return output_config_effort.or(Some(CanonicalEffort::High));
    }

    output_config_effort
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

/// Build a Gemini `thinkingConfig` from a canonical effort. `uses_level` picks
/// the surface (see [`gemini_uses_thinking_level`]).
pub(crate) fn gemini_thinking_config(effort: CanonicalEffort, uses_level: bool) -> Option<Value> {
    if uses_level {
        let level = effort.to_gemini_thinking_level()?;
        Some(json!({ "thinking_level": level }))
    } else {
        let budget = effort.to_gemini_thinking_budget()?;
        Some(json!({ "thinkingBudget": budget }))
    }
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
    fn extract_anthropic_effort_uses_output_config_effort_for_adaptive() {
        // Adaptive is controlled by output_config.effort, not budget_tokens.
        let body = json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "low" }
        });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::Low));
    }

    #[test]
    fn extract_anthropic_effort_max_via_adaptive_plus_effort() {
        let body = json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "max" }
        });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::Max));
    }

    #[test]
    fn extract_anthropic_effort_recognizes_xhigh_for_adaptive() {
        // Anthropic accepts `xhigh` for Opus 4.7 adaptive thinking. Without
        // parsing it, we'd fall back to `High` and silently downgrade the
        // user's deep-effort intent when bridging to OpenAI.
        let body = json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": "xhigh" }
        });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::Max));
    }

    #[test]
    fn from_anthropic_effort_str_accepts_xhigh_as_max() {
        assert_eq!(
            CanonicalEffort::from_anthropic_effort_str("xhigh"),
            Some(CanonicalEffort::Max)
        );
        assert_eq!(
            CanonicalEffort::from_anthropic_effort_str("XHIGH"),
            Some(CanonicalEffort::Max)
        );
    }

    #[test]
    fn extract_anthropic_effort_defaults_high_for_adaptive_without_effort() {
        let body = json!({ "thinking": { "type": "adaptive" } });
        assert_eq!(extract_anthropic_effort(&body), Some(CanonicalEffort::High));
    }

    #[test]
    fn extract_anthropic_effort_enabled_without_budget_uses_output_config_effort() {
        // Same precedence change applies to enabled — if no budget, honor
        // the user's effort hint instead of jumping to High.
        let body = json!({
            "thinking": { "type": "enabled" },
            "output_config": { "effort": "medium" }
        });
        assert_eq!(
            extract_anthropic_effort(&body),
            Some(CanonicalEffort::Medium)
        );
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
        assert!(gemini_thinking_config(CanonicalEffort::Minimal, true).is_none());
        assert!(gemini_thinking_config(CanonicalEffort::Minimal, false).is_none());
    }

    #[test]
    fn gemini_thinking_level_collapses_max_to_high() {
        let cfg = gemini_thinking_config(CanonicalEffort::Max, true).expect("max maps");
        assert_eq!(cfg["thinking_level"], "high");
    }

    #[test]
    fn gemini_2_5_gets_numeric_budget_not_level() {
        let cfg = gemini_thinking_config(CanonicalEffort::Medium, false).expect("medium maps");
        assert_eq!(cfg["thinkingBudget"], 8192);
        assert!(cfg.get("thinking_level").is_none());
        let max = gemini_thinking_config(CanonicalEffort::Max, false).expect("max maps");
        assert_eq!(max["thinkingBudget"], 24576);
    }

    #[test]
    fn anthropic_adaptive_thinking_gates_on_version() {
        // Opus 4.7+ and Fable/Mythos reject budget_tokens → adaptive.
        assert!(anthropic_thinking_uses_adaptive("claude-opus-4-8"));
        assert!(anthropic_thinking_uses_adaptive("claude-opus-4-7"));
        assert!(anthropic_thinking_uses_adaptive("claude-fable-5"));
        assert!(anthropic_thinking_uses_adaptive(
            "anthropic/claude-mythos-5"
        ));
        // Older Claude keeps the numeric budget.
        assert!(!anthropic_thinking_uses_adaptive("claude-opus-4-6"));
        assert!(!anthropic_thinking_uses_adaptive("claude-opus-4-5"));
        assert!(!anthropic_thinking_uses_adaptive("claude-sonnet-4-6"));
        assert!(!anthropic_thinking_uses_adaptive("claude-3-5-sonnet"));
        assert!(!anthropic_thinking_uses_adaptive("gpt-5"));
    }

    #[test]
    fn anthropic_adaptive_gate_normalizes_dotted_and_prefixed_ids() {
        assert!(anthropic_thinking_uses_adaptive(
            "anthropic/claude-opus-4.7"
        ));
        assert!(anthropic_thinking_uses_adaptive(
            "us.anthropic.claude-opus-4-8"
        ));
        assert!(anthropic_thinking_uses_adaptive("claude-opus-4-7-20260120"));
        assert!(!anthropic_thinking_uses_adaptive(
            "anthropic/claude-opus-4.6"
        ));
    }

    #[test]
    fn anthropic_supports_adaptive_thinking_covers_4_6_plus() {
        for m in [
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "anthropic/claude-mythos-5",
            "anthropic/claude-opus-4.6",
            "us.anthropic.claude-sonnet-4-6",
            "claude-opus-4-6-20260120",
        ] {
            assert!(anthropic_supports_adaptive_thinking(m), "{m} → adaptive");
        }
        for m in [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet",
            "claude-sonnet-4-60",
            "gpt-5",
        ] {
            assert!(
                !anthropic_supports_adaptive_thinking(m),
                "{m} → not adaptive"
            );
        }
    }

    #[test]
    fn anthropic_supports_output_effort_covers_4_5_plus_and_fable() {
        for m in [
            "claude-opus-4-5",
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "anthropic/claude-mythos-5",
        ] {
            assert!(anthropic_supports_output_effort(m), "{m} → effort");
        }
        for m in ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-3-5-sonnet"] {
            assert!(!anthropic_supports_output_effort(m), "{m} → no effort");
        }
    }

    #[test]
    fn anthropic_rejects_disabled_thinking_only_fable_mythos() {
        assert!(anthropic_rejects_disabled_thinking("claude-fable-5"));
        assert!(anthropic_rejects_disabled_thinking(
            "anthropic/claude-mythos-5"
        ));
        assert!(!anthropic_rejects_disabled_thinking("claude-opus-4-8"));
        assert!(!anthropic_rejects_disabled_thinking("claude-opus-4-7"));
        assert!(!anthropic_rejects_disabled_thinking("claude-haiku-4-5"));
    }

    #[test]
    fn gemini_version_picks_thinking_surface() {
        assert!(gemini_uses_thinking_level("gemini-3-pro-preview"));
        assert!(gemini_uses_thinking_level("google/gemini-3-flash"));
        assert!(!gemini_uses_thinking_level("gemini-2.5-flash"));
        assert!(!gemini_uses_thinking_level("gemini-2.5-pro"));
        assert!(!gemini_uses_thinking_level("models/gemini-2.0-flash"));
        assert!(!gemini_uses_thinking_level("gpt-5"));
    }
}
