//! Shared model name normalization utilities.
//!
//! Different providers expect different model name formats:
//! - OpenRouter: `anthropic/claude-sonnet-4.6` (prefix + dots)
//! - Copilot: `claude-sonnet-4.6` (dots, no prefix)
//! - Anthropic: `claude-sonnet-4-6` (hyphens)
//!
//! This module consolidates the version conversion logic that was previously
//! duplicated across Anthropic router code, copilot_router, and chat.rs.

use crate::services::provider_profile::{is_aivo_starter_base, is_openrouter_base};
use crate::services::provider_protocol::ProviderProtocol;

/// Converts Claude model version separators from hyphens to dots.
///
/// Examples:
/// - `claude-sonnet-4-6` → `claude-sonnet-4.6`
/// - `claude-haiku-4-5` → `claude-haiku-4.5`
/// - `claude-haiku-4-5-20251001` → `claude-haiku-4-5-20251001` (date suffix preserved)
/// - `gpt-4o` → `gpt-4o` (non-Claude models pass through)
pub fn normalize_claude_version(model: &str) -> String {
    if let Some(last_hyphen_pos) = model.rfind('-') {
        let after_last_hyphen = &model[last_hyphen_pos + 1..];

        // Date suffix (8 digits): keep as-is
        if after_last_hyphen.len() == 8 && after_last_hyphen.chars().all(|c| c.is_ascii_digit()) {
            return model.to_string();
        }

        // Version number: convert the separating hyphen to a dot
        if after_last_hyphen
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
            && let Some(second_last_hyphen) = model[..last_hyphen_pos].rfind('-')
            && model[second_last_hyphen + 1..last_hyphen_pos]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit())
        {
            let mut result = model.to_string();
            result.replace_range(last_hyphen_pos..=last_hyphen_pos, ".");
            return result;
        }
    }
    model.to_string()
}

/// Transforms a model name for OpenRouter compatibility.
/// Adds `anthropic/` prefix and normalizes version separators.
///
/// Examples:
/// - `claude-sonnet-4-6` → `anthropic/claude-sonnet-4.6`
/// - `anthropic/claude-sonnet-4.6` → `anthropic/claude-sonnet-4.6` (already prefixed)
/// - `gpt-4o` → `gpt-4o` (non-Claude models pass through)
pub fn transform_model_for_openrouter(model: &str) -> String {
    if !model.starts_with("claude-") || model.starts_with("anthropic/") {
        return model.to_string();
    }
    format!("anthropic/{}", normalize_claude_version(model))
}

/// Lowercase and collapse `.`→`-` so `claude-sonnet-4.6` and `claude-sonnet-4-6`
/// compare equal, while keeping the full `vendor/…/model` path intact.
fn normalize_model_id(model: &str) -> String {
    model.trim().to_ascii_lowercase().replace('.', "-")
}

/// True when `a` and `b` are the same model modulo a leading `vendor/…` path:
/// equal, or one is a *slash-boundary* suffix of the other. Matches `glm-5.2`
/// to `@cf/zai-org/glm-5.2`, but not `qwen/glm-5.2` to `zai-org/glm-5.2` or
/// `gpt-4` to `gpt-4o`. Inputs must be pre-normalized.
fn same_model_modulo_prefix(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (longer, shorter) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    !shorter.is_empty()
        && longer.len() > shorter.len()
        && longer.ends_with(shorter)
        && longer.as_bytes()[longer.len() - shorter.len() - 1] == b'/'
}

/// Resolve `requested` to the exact id the provider advertises in `catalog`
/// (its `/v1/models` list): exact match, else modulo a `vendor/…` prefix and
/// dot-vs-dash. `None` when nothing matches. The data-driven alternative to
/// per-host model-name rules — we send a name the provider actually lists.
pub fn resolve_model_from_catalog(catalog: &[String], requested: &str) -> Option<String> {
    if let Some(hit) = catalog.iter().find(|m| m.as_str() == requested) {
        return Some(hit.clone());
    }
    let want = normalize_model_id(requested);
    catalog
        .iter()
        .find(|m| same_model_modulo_prefix(&normalize_model_id(m), &want))
        .cloned()
}

/// Resolve a model name to what the provider expects: the exact catalog id when
/// one matches (so any `vendor/`-slug gateway works with no per-host rule), else
/// the OpenRouter string transform as a cold-start fallback.
pub fn transform_model_for_provider(
    catalog: Option<&[String]>,
    base_url: &str,
    model: &str,
) -> String {
    if let Some(catalog) = catalog
        && let Some(hit) = resolve_model_from_catalog(catalog, model)
    {
        return hit;
    }
    if is_openrouter_base(base_url) {
        transform_model_for_openrouter(model)
    } else {
        model.to_string()
    }
}

pub fn google_native_model_name(model: &str) -> &str {
    model.strip_prefix("google/").unwrap_or(model)
}

pub fn anthropic_native_model_name(model: &str) -> String {
    let stripped = model.strip_prefix("anthropic/").unwrap_or(model);
    if !stripped.starts_with("claude-") {
        return stripped.to_string();
    }

    if let Some(dot_pos) = stripped.find('.')
        && stripped[..dot_pos]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_digit())
        && stripped[dot_pos + 1..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_digit())
    {
        let mut normalized = stripped.to_string();
        normalized.replace_range(dot_pos..=dot_pos, "-");
        return normalized;
    }

    stripped.to_string()
}

/// Strip the `[<N>m]` UI-hint suffix aivo's env injector appends when
/// `--max-context=<N>m` (or `--<N>m`) is set. The suffix is meaningful to
/// Claude Code's status-bar logic but not to the upstream API; bridges
/// should drop it before forwarding so non-Claude providers don't see an
/// unrecognized model name.
pub fn strip_context_suffix(model: &str) -> &str {
    let Some(s_no_close) = model.strip_suffix(']') else {
        return model;
    };
    let Some(bracket_idx) = s_no_close.rfind('[') else {
        return model;
    };
    let inner = &s_no_close[bracket_idx + 1..];
    let Some(last) = inner.chars().last() else {
        return model;
    };
    if last != 'm' && last != 'M' {
        return model;
    }
    let digits = &inner[..inner.len() - 1];
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return model;
    }
    &model[..bracket_idx]
}

pub fn is_gateway_style_endpoint(base_url: &str) -> bool {
    let lower = base_url.trim().to_ascii_lowercase();
    lower.contains("/endpoint") || lower.contains("gateway")
}

pub fn infer_provider_name_from_model(model: &str) -> Option<String> {
    let trimmed = model.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((provider, _)) = trimmed.split_once('/')
        && !provider.trim().is_empty()
    {
        return Some(provider.trim().to_ascii_lowercase());
    }

    match infer_model_protocol(trimmed) {
        Some(ProviderProtocol::Anthropic) => Some("anthropic".to_string()),
        Some(ProviderProtocol::Google) => Some("google".to_string()),
        Some(ProviderProtocol::Openai) | Some(ProviderProtocol::ResponsesApi) => {
            Some("openai".to_string())
        }
        None => None,
    }
}

/// True when a cross-protocol model must be forwarded verbatim: a multi-vendor
/// gateway routes on the model name, so the wire protocol we speak to it says
/// nothing about which vendors it serves.
pub fn should_preserve_cross_protocol_model(
    base_url: &str,
    model: &str,
    target_protocol: ProviderProtocol,
) -> bool {
    match infer_model_protocol(model) {
        Some(protocol) if model_family(protocol) != model_family(target_protocol) => {
            // The starter is a multi-vendor gateway but its URL misses the name heuristic.
            is_gateway_style_endpoint(base_url) || is_aivo_starter_base(base_url)
        }
        _ => false,
    }
}

/// Converts Claude model names from Anthropic/Claude Code format to Copilot format.
///
/// Claude Code sends names like `claude-sonnet-4-6-20250603` or `claude-sonnet-4-6`.
/// Copilot API expects names like `claude-sonnet-4.6` (dots for minor versions).
///
/// Steps:
///   1. Strip trailing date suffix `-YYYYMMDD`
///   2. Convert `claude-{family}-{major}-{minor}` → `claude-{family}-{major}.{minor}`
pub fn copilot_model_name(model: &str) -> String {
    // Strip trailing -YYYYMMDD date suffix
    let base = if model.len() > 9 {
        let (prefix, suffix) = model.split_at(model.len() - 9);
        if suffix.starts_with('-') && suffix[1..].chars().all(|c| c.is_ascii_digit()) {
            prefix
        } else {
            model
        }
    } else {
        model
    };

    // Convert hyphenated version to dotted: claude-sonnet-4-6 → claude-sonnet-4.6
    // Pattern: claude-{family}-{major}-{minor} where major/minor are digits
    if let Some(stripped) = base.strip_prefix("claude-") {
        let parts: Vec<&str> = stripped.split('-').collect();
        // e.g. ["sonnet", "4", "6"] or ["sonnet", "4"] or ["haiku", "4", "5"]
        if parts.len() >= 3 {
            let family = parts[0]; // sonnet, haiku, opus
            let major = parts[1]; // "4"
            let minor = parts[2]; // "6", "5"
            if major.chars().all(|c| c.is_ascii_digit())
                && minor.chars().all(|c| c.is_ascii_digit())
            {
                // Rejoin any remaining parts (e.g. "-thinking") after the version
                let rest = if parts.len() > 3 {
                    format!("-{}", parts[3..].join("-"))
                } else {
                    String::new()
                };
                return format!("claude-{}-{}.{}{}", family, major, minor, rest);
            }
        }
    }

    base.to_string()
}

pub fn default_model_for_protocol(protocol: ProviderProtocol) -> &'static str {
    match protocol {
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => "gpt-4o",
        ProviderProtocol::Anthropic => "claude-sonnet-4-5",
        ProviderProtocol::Google => "gemini-2.5-pro",
    }
}

pub fn select_model_for_protocol(
    requested_model: Option<&str>,
    explicit_model: Option<&str>,
    target_protocol: ProviderProtocol,
) -> String {
    if let Some(model) = explicit_model.filter(|model| !model.trim().is_empty()) {
        return model.to_string();
    }

    match requested_model.filter(|model| !model.trim().is_empty()) {
        Some(model) => match infer_model_protocol(model) {
            Some(protocol) if model_family(protocol) != model_family(target_protocol) => {
                default_model_for_protocol(target_protocol).to_string()
            }
            _ => model.to_string(),
        },
        None => default_model_for_protocol(target_protocol).to_string(),
    }
}

pub fn select_model_for_provider_attempt(
    catalog: Option<&[String]>,
    base_url: &str,
    requested_model: Option<&str>,
    explicit_model: Option<&str>,
    target_protocol: ProviderProtocol,
) -> String {
    // An explicit model is the user's deliberate choice — respect it verbatim.
    if let Some(model) = explicit_model.filter(|model| !model.trim().is_empty()) {
        return model.to_string();
    }

    let selected = if let Some(model) = requested_model.filter(|model| !model.trim().is_empty())
        && should_preserve_cross_protocol_model(base_url, model, target_protocol)
    {
        model.to_string()
    } else {
        select_model_for_protocol(requested_model, explicit_model, target_protocol)
    };

    // Snap the tool's default name to the exact catalog id when we have one.
    if let Some(catalog) = catalog
        && let Some(hit) = resolve_model_from_catalog(catalog, &selected)
    {
        return hit;
    }
    selected
}

/// Normalize protocol for model comparison — ResponsesApi uses the same models as Openai.
fn model_family(p: ProviderProtocol) -> ProviderProtocol {
    match p {
        ProviderProtocol::ResponsesApi => ProviderProtocol::Openai,
        other => other,
    }
}

/// True for OpenAI chat models (`gpt-*`), excluding the `o*` reasoning series.
pub(crate) fn is_gpt_chat_model_name(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);
    name_only.starts_with("gpt-")
}

/// True for any OpenAI-style model: `gpt-*` chat models or the `o1`/`o3`/`o4`
/// reasoning series. Superset of [`is_gpt_chat_model_name`].
pub(crate) fn is_openai_style_model_name(model: &str) -> bool {
    if is_gpt_chat_model_name(model) {
        return true;
    }
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);
    name_only.starts_with("o1") || name_only.starts_with("o3") || name_only.starts_with("o4")
}

/// True for OpenAI models that reject the legacy `max_tokens` field and require
/// `max_completion_tokens` instead — the o-series reasoning models (o1/o3/o4),
/// the GPT-5 family, and the Codex family.
pub(crate) fn requires_max_completion_tokens(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);
    name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
        || name_only.starts_with("gpt-5")
        || name_only.contains("codex")
}

fn infer_model_protocol(model: &str) -> Option<ProviderProtocol> {
    let lower = model.to_ascii_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);

    if name_only.contains("claude") {
        Some(ProviderProtocol::Anthropic)
    } else if name_only.contains("gemini") {
        Some(ProviderProtocol::Google)
    } else if name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
        || name_only.starts_with("chatgpt")
        || name_only.contains("codex")
    {
        Some(ProviderProtocol::Openai)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_claude_version_basic() {
        assert_eq!(
            normalize_claude_version("claude-sonnet-4-6"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            normalize_claude_version("claude-haiku-4-5"),
            "claude-haiku-4.5"
        );
        assert_eq!(
            normalize_claude_version("claude-opus-4-6"),
            "claude-opus-4.6"
        );
    }

    #[test]
    fn test_normalize_claude_version_date_suffix_preserved() {
        assert_eq!(
            normalize_claude_version("claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_normalize_claude_version_no_change() {
        assert_eq!(normalize_claude_version("gpt-4o"), "gpt-4o");
        assert_eq!(
            normalize_claude_version("claude-sonnet-4"),
            "claude-sonnet-4"
        );
    }

    #[test]
    fn test_transform_model_for_openrouter() {
        assert_eq!(
            transform_model_for_openrouter("claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_openrouter("anthropic/claude-sonnet-4.6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(transform_model_for_openrouter("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_transform_model_for_provider() {
        // No catalog: falls back to the OpenRouter string transform.
        assert_eq!(
            transform_model_for_provider(None, "https://openrouter.ai/api/v1", "claude-sonnet-4-6"),
            "anthropic/claude-sonnet-4.6"
        );
        assert_eq!(
            transform_model_for_provider(None, "https://api.example.com/v1", "claude-sonnet-4-6"),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_catalog_snaps_default_to_provider_slug() {
        // Requesty/Vercel/etc advertise `anthropic/claude-sonnet-4.6`. With the
        // catalog in hand we match the tool's `claude-sonnet-4-6` default to it
        // — no `is_requesty_base`/`is_vercel_base` host rule required.
        let catalog = vec![
            "anthropic/claude-sonnet-4.6".to_string(),
            "openai/gpt-4o".to_string(),
        ];
        assert_eq!(
            transform_model_for_provider(
                Some(&catalog),
                "https://router.requesty.ai/v1",
                "claude-sonnet-4-6"
            ),
            "anthropic/claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_catalog_exact_match_passes_through() {
        let catalog = vec!["claude-sonnet-4-6".to_string()];
        assert_eq!(
            transform_model_for_provider(
                Some(&catalog),
                "https://api.example.com/v1",
                "claude-sonnet-4-6"
            ),
            "claude-sonnet-4-6"
        );
    }

    #[test]
    fn test_catalog_miss_falls_back_to_host_transform() {
        // Model not in the catalog → behave exactly as the no-catalog path.
        let catalog = vec!["openai/gpt-4o".to_string()];
        assert_eq!(
            transform_model_for_provider(
                Some(&catalog),
                "https://openrouter.ai/api/v1",
                "claude-sonnet-4-6"
            ),
            "anthropic/claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_resolve_model_from_catalog_no_false_match() {
        // A different minor version must not match.
        let catalog = vec!["anthropic/claude-sonnet-4.5".to_string()];
        assert_eq!(
            resolve_model_from_catalog(&catalog, "claude-sonnet-4-6"),
            None
        );
    }

    #[test]
    fn test_resolve_multi_segment_id() {
        // Cloudflare-style `@cf/<org>/<model>` ids carry two slashes.
        let catalog = vec!["@cf/zai-org/glm-5.2".to_string()];
        // Exact id passes through.
        assert_eq!(
            resolve_model_from_catalog(&catalog, "@cf/zai-org/glm-5.2").as_deref(),
            Some("@cf/zai-org/glm-5.2")
        );
        // A bare or org-qualified request snaps to the full advertised id.
        assert_eq!(
            resolve_model_from_catalog(&catalog, "glm-5.2").as_deref(),
            Some("@cf/zai-org/glm-5.2")
        );
        assert_eq!(
            resolve_model_from_catalog(&catalog, "zai-org/glm-5-2").as_deref(),
            Some("@cf/zai-org/glm-5.2")
        );
    }

    #[test]
    fn test_resolve_multi_segment_different_org_does_not_match() {
        // Same model name under a different org is a different model.
        let catalog = vec!["zai-org/glm-5.2".to_string()];
        assert_eq!(resolve_model_from_catalog(&catalog, "qwen/glm-5.2"), None);
    }

    #[test]
    fn test_resolve_does_not_partial_token_match() {
        // The slash boundary blocks `gpt-4` matching `gpt-4o`.
        let catalog = vec!["openai/gpt-4o".to_string()];
        assert_eq!(resolve_model_from_catalog(&catalog, "gpt-4"), None);
    }

    #[test]
    fn test_google_native_model_name_strips_provider_prefix() {
        assert_eq!(
            google_native_model_name("google/gemini-2.5-pro"),
            "gemini-2.5-pro"
        );
        assert_eq!(google_native_model_name("gemini-2.5-pro"), "gemini-2.5-pro");
    }

    #[test]
    fn test_strip_context_suffix() {
        assert_eq!(strip_context_suffix("hello[1m]"), "hello");
        assert_eq!(strip_context_suffix("hello[2m]"), "hello");
        assert_eq!(
            strip_context_suffix("deepseek-v4-flash[1m]"),
            "deepseek-v4-flash"
        );
        assert_eq!(strip_context_suffix("hello"), "hello");
        assert_eq!(strip_context_suffix("hello[3m]"), "hello");
        assert_eq!(strip_context_suffix("hello[12m]"), "hello");
        assert_eq!(strip_context_suffix("[1m]"), "");
        assert_eq!(strip_context_suffix("[2m]"), "");
        // Non-context bracketed suffixes left alone.
        assert_eq!(strip_context_suffix("model[v2]"), "model[v2]");
        assert_eq!(strip_context_suffix("model[m]"), "model[m]");
    }

    #[test]
    fn test_anthropic_native_model_name_normalizes_claude_versions() {
        assert_eq!(
            anthropic_native_model_name("anthropic/claude-sonnet-4.6"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            anthropic_native_model_name("claude-haiku-4.5-20251001"),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(anthropic_native_model_name("MiniMax-M1"), "MiniMax-M1");
    }

    #[test]
    fn test_copilot_model_name_strips_date() {
        assert_eq!(
            copilot_model_name("claude-sonnet-4-20250514"),
            "claude-sonnet-4"
        );
        assert_eq!(
            copilot_model_name("claude-sonnet-4-6-20250603"),
            "claude-sonnet-4.6"
        );
        assert_eq!(
            copilot_model_name("claude-haiku-4-5-20250501"),
            "claude-haiku-4.5"
        );
    }

    #[test]
    fn test_copilot_model_name_converts_dots() {
        assert_eq!(copilot_model_name("claude-sonnet-4"), "claude-sonnet-4");
        assert_eq!(copilot_model_name("claude-sonnet-4-6"), "claude-sonnet-4.6");
        assert_eq!(copilot_model_name("gpt-4o"), "gpt-4o");
    }

    #[test]
    fn test_select_model_for_protocol_keeps_provider_native_models() {
        assert_eq!(
            select_model_for_protocol(Some("MiniMax-M1"), None, ProviderProtocol::Anthropic),
            "MiniMax-M1"
        );
        assert_eq!(
            select_model_for_protocol(
                Some("google/gemini-2.5-pro"),
                None,
                ProviderProtocol::Google
            ),
            "google/gemini-2.5-pro"
        );
    }

    #[test]
    fn test_select_model_for_protocol_remaps_cross_protocol_defaults() {
        assert_eq!(
            select_model_for_protocol(Some("gpt-5-codex"), None, ProviderProtocol::Anthropic),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            select_model_for_protocol(Some("claude-sonnet-4-5"), None, ProviderProtocol::Google),
            "gemini-2.5-pro"
        );
        assert_eq!(
            select_model_for_protocol(Some("gemini-2.0-flash"), None, ProviderProtocol::Openai),
            "gpt-4o"
        );
    }

    #[test]
    fn test_should_preserve_cross_protocol_model_for_gateway_endpoint() {
        assert!(should_preserve_cross_protocol_model(
            "https://api.ai.example-gateway.net/endpoint",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
        assert!(should_preserve_cross_protocol_model(
            "http://localhost:3005/endpoint",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
        assert!(is_gateway_style_endpoint("https://ai-gateway.vercel.sh/v1"));
    }

    #[test]
    fn test_should_preserve_cross_protocol_model_for_aivo_starter() {
        assert!(should_preserve_cross_protocol_model(
            crate::constants::AIVO_STARTER_REAL_URL,
            "anthropic/claude-opus-5",
            ProviderProtocol::Openai
        ));
        assert!(should_preserve_cross_protocol_model(
            crate::constants::AIVO_STARTER_SENTINEL,
            "google/gemini-2.5-pro",
            ProviderProtocol::Openai
        ));
    }

    #[test]
    fn test_should_preserve_cross_protocol_model_for_non_openai_targets() {
        assert!(should_preserve_cross_protocol_model(
            "https://api.ai.example-gateway.net/endpoint",
            "anthropic/claude-sonnet-4.5",
            ProviderProtocol::Google
        ));
        assert!(should_preserve_cross_protocol_model(
            "https://api.ai.example-gateway.net/endpoint",
            "google/gemini-2.5-pro",
            ProviderProtocol::Anthropic
        ));
        assert!(should_preserve_cross_protocol_model(
            crate::constants::AIVO_STARTER_REAL_URL,
            "anthropic/claude-opus-5",
            ProviderProtocol::Google
        ));
    }

    #[test]
    fn test_should_not_preserve_cross_protocol_model_for_plain_openai_endpoint() {
        assert!(!should_preserve_cross_protocol_model(
            "https://api.openai.com/v1",
            "claude-sonnet-4-6",
            ProviderProtocol::Openai
        ));
    }

    #[test]
    fn test_infer_provider_name_from_model() {
        assert_eq!(
            infer_provider_name_from_model("claude-sonnet-4-6").as_deref(),
            Some("anthropic")
        );
        assert_eq!(
            infer_provider_name_from_model("moonshot/kimi-k2.5").as_deref(),
            Some("moonshot")
        );
        assert_eq!(infer_provider_name_from_model("").as_deref(), None);
    }

    #[test]
    fn test_select_model_for_protocol_prefers_explicit_model() {
        assert_eq!(
            select_model_for_protocol(
                Some("gpt-5-codex"),
                Some("claude-3-opus"),
                ProviderProtocol::Anthropic
            ),
            "claude-3-opus"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_preserves_cross_protocol_gateway_models() {
        assert_eq!(
            select_model_for_provider_attempt(
                None,
                "https://api.ai.example-gateway.net/endpoint",
                Some("claude-sonnet-4.6"),
                None,
                ProviderProtocol::Openai
            ),
            "claude-sonnet-4.6"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_still_remaps_plain_openai_endpoints() {
        assert_eq!(
            select_model_for_provider_attempt(
                None,
                "https://api.openai.com/v1",
                Some("claude-sonnet-4.6"),
                None,
                ProviderProtocol::Openai
            ),
            "gpt-4o"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_keeps_claude_on_google_route() {
        // Regression: a learned Google route used to swap an explicit Claude
        // selection for gemini-2.5-pro.
        assert_eq!(
            select_model_for_provider_attempt(
                None,
                "https://api.acme.example/endpoint",
                Some("anthropic/claude-sonnet-4.5"),
                None,
                ProviderProtocol::Google
            ),
            "anthropic/claude-sonnet-4.5"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_still_remaps_single_vendor_google_endpoint() {
        assert_eq!(
            select_model_for_provider_attempt(
                None,
                "https://generativelanguage.googleapis.com/v1beta",
                Some("anthropic/claude-sonnet-4.5"),
                None,
                ProviderProtocol::Google
            ),
            "gemini-2.5-pro"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_snaps_to_catalog() {
        // OpenAI-format gateway whose catalog lists the slug form: the tool's
        // default name snaps to the exact id the gateway advertises.
        let catalog = vec!["openai/gpt-4o".to_string()];
        assert_eq!(
            select_model_for_provider_attempt(
                Some(&catalog),
                "https://router.requesty.ai/v1",
                Some("gpt-4o"),
                None,
                ProviderProtocol::Openai
            ),
            "openai/gpt-4o"
        );
    }

    #[test]
    fn test_select_model_for_provider_attempt_explicit_model_not_snapped() {
        // An explicit user choice is respected verbatim, even if the catalog
        // lists a differently-spelled equivalent.
        let catalog = vec!["openai/gpt-4o".to_string()];
        assert_eq!(
            select_model_for_provider_attempt(
                Some(&catalog),
                "https://router.requesty.ai/v1",
                None,
                Some("gpt-4o"),
                ProviderProtocol::Openai
            ),
            "gpt-4o"
        );
    }

    #[test]
    fn test_is_gpt_chat_model_name() {
        assert!(is_gpt_chat_model_name("gpt-4o"));
        assert!(is_gpt_chat_model_name("gpt-5.5"));
        assert!(is_gpt_chat_model_name("openai/gpt-4.1"));
        assert!(is_gpt_chat_model_name("GPT-4o"));

        assert!(!is_gpt_chat_model_name("o1-preview"));
        assert!(!is_gpt_chat_model_name("o4-mini"));
        assert!(!is_gpt_chat_model_name("claude-sonnet-4"));
    }

    #[test]
    fn test_requires_max_completion_tokens() {
        // Reasoning models reject legacy max_tokens.
        assert!(requires_max_completion_tokens("o1-preview"));
        assert!(requires_max_completion_tokens("o3-mini"));
        assert!(requires_max_completion_tokens("o4-mini"));
        assert!(requires_max_completion_tokens("gpt-5"));
        assert!(requires_max_completion_tokens("gpt-5.1"));
        assert!(requires_max_completion_tokens("gpt-5.4"));
        assert!(requires_max_completion_tokens("gpt-5-codex"));
        assert!(requires_max_completion_tokens("gpt-5-pro"));
        assert!(requires_max_completion_tokens("openai/gpt-5"));
        assert!(requires_max_completion_tokens("anthropic/codex-bridge"));

        // Older / non-reasoning models still accept max_tokens.
        assert!(!requires_max_completion_tokens("gpt-4o"));
        assert!(!requires_max_completion_tokens("gpt-4"));
        assert!(!requires_max_completion_tokens("gpt-4-turbo"));
        assert!(!requires_max_completion_tokens("gpt-3.5-turbo"));
        assert!(!requires_max_completion_tokens("claude-sonnet-4-6"));
        assert!(!requires_max_completion_tokens("gemini-2.5-pro"));
    }

    #[test]
    fn test_is_openai_style_model_name() {
        assert!(is_openai_style_model_name("gpt-4o"));
        assert!(is_openai_style_model_name("gpt-5"));
        assert!(is_openai_style_model_name("openai/gpt-4.1"));
        assert!(is_openai_style_model_name("o1-preview"));
        assert!(is_openai_style_model_name("o3-mini"));
        assert!(is_openai_style_model_name("o4-mini"));
        assert!(is_openai_style_model_name("GPT-4o"));

        assert!(!is_openai_style_model_name("claude-sonnet-4"));
        assert!(!is_openai_style_model_name("anthropic/claude-sonnet-4-5"));
        assert!(!is_openai_style_model_name("gemini-2.5-pro"));
        assert!(!is_openai_style_model_name("ollama/llama3"));
    }
}
