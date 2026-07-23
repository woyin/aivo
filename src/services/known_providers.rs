//! Central registry of known AI providers with auto-fill base URLs.
//!
//! Provider data is embedded from `src/data/providers.json` at compile time
//! and parsed once via `LazyLock`. Used by `keys add` for name-based URL
//! auto-detection.

use std::sync::LazyLock;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct KnownProvider {
    pub id: String,
    pub name: String,
    pub base_url: String,
}

/// All known providers, ordered so that more specific ids come first
/// (e.g. "openrouter" before "openai") to avoid substring false-positives.
static KNOWN_PROVIDERS: LazyLock<Vec<KnownProvider>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../data/providers.json"))
        .expect("embedded providers.json must be valid")
});

/// Find a provider whose id matches the input as a substring in either
/// direction (case-insensitive). Matches both "my-openrouter-key" (id
/// contained in input) and "google" (input contained in id).
///
/// Requires at least 3 characters of input — the shortest provider id is
/// 3 chars (e.g. "xai", "poe"), and 1–2 char inputs like "hi" or "ai"
/// produce coincidental substring hits against longer ids (e.g. "hi" in
/// "zhipuai"). Display names are intentionally not matched because they
/// contain descriptive words like "China" or "Gateway" that also produce
/// false positives on short inputs.
pub fn find_by_name_substring(input: &str) -> Option<&KnownProvider> {
    let input_lower = (input.len() >= 3).then(|| input.to_ascii_lowercase())?;
    if let Some(exact) = KNOWN_PROVIDERS
        .iter()
        .find(|p| p.id.eq_ignore_ascii_case(&input_lower))
    {
        return Some(exact);
    }
    matching_providers(input).next()
}

/// Like `find_by_name_substring` but yields every match in declaration order.
/// Used by the interactive picker to hoist *all* matching providers (e.g. an
/// input of "bedrock" matches both `bedrock-mantle` and `bedrock-runtime`).
pub fn find_all_by_name_substring(input: &str) -> Vec<&KnownProvider> {
    matching_providers(input).collect()
}

fn matching_providers(input: &str) -> impl Iterator<Item = &'static KnownProvider> {
    const MIN_LEN: usize = 3;
    let input_lower = (input.len() >= MIN_LEN).then(|| input.to_ascii_lowercase());
    KNOWN_PROVIDERS.iter().filter(move |p| match &input_lower {
        Some(s) => {
            let id_lower = p.id.to_ascii_lowercase();
            id_lower.contains(s) || s.contains(&id_lower)
        }
        None => false,
    })
}

/// Returns all known providers (for the provider picker UI).
pub fn all() -> &'static [KnownProvider] {
    &KNOWN_PROVIDERS
}

/// Suggest a short `keys add` name from a base URL: a matching known provider's
/// id (`api.groq.com` → `groq`), else the host label before the TLD. None for IPs.
pub fn suggest_name_from_url(url: &str) -> Option<String> {
    let host = host_of(url)?;
    if let Some(p) = KNOWN_PROVIDERS
        .iter()
        .find(|p| host_of(&p.base_url).is_some_and(|h| h.eq_ignore_ascii_case(host)))
    {
        return Some(p.id.clone());
    }
    slug_from_host(host)
}

fn host_of(url: &str) -> Option<&str> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or(authority);
    (!host.is_empty()).then_some(host)
}

/// The host label before the TLD (`api.groq.com` → `groq`). None for IP literals.
fn slug_from_host(host: &str) -> Option<String> {
    // Strip :port; leave IPv6 (multiple ':') alone.
    let host = if host.matches(':').count() == 1 {
        host.split(':').next().unwrap_or(host)
    } else {
        host
    };
    let host = host.trim_matches(|c| c == '[' || c == ']');
    if host.is_empty() || host.parse::<std::net::IpAddr>().is_ok() {
        return None;
    }
    let labels: Vec<&str> = host.split('.').filter(|s| !s.is_empty()).collect();
    let label = match labels.len() {
        0 => return None,
        1 => labels[0],
        n => labels[n - 2],
    };
    let slug: String = label
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    (!slug.is_empty()).then_some(slug)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_by_name_substring_matches() {
        let p = find_by_name_substring("my-openrouter-key").unwrap();
        assert_eq!(p.id, "openrouter");

        let p = find_by_name_substring("work_groq").unwrap();
        assert_eq!(p.id, "groq");
    }

    #[test]
    fn find_by_name_substring_no_match() {
        assert!(find_by_name_substring("random").is_none());
        assert!(find_by_name_substring("").is_none());
    }

    #[test]
    fn suggest_name_known_provider_uses_canonical_id() {
        assert_eq!(
            suggest_name_from_url("https://api.groq.com/openai/v1").as_deref(),
            Some("groq")
        );
        assert_eq!(
            suggest_name_from_url("https://openrouter.ai/api/v1").as_deref(),
            Some("openrouter")
        );
        assert_eq!(
            suggest_name_from_url("https://api.openai.com").as_deref(),
            Some("openai")
        );
    }

    #[test]
    fn suggest_name_unknown_host_falls_back_to_label() {
        assert_eq!(
            suggest_name_from_url("https://my-llm.acme.dev/v1").as_deref(),
            Some("acme")
        );
        assert_eq!(
            suggest_name_from_url("http://gateway.internal.corp:8080/v1").as_deref(),
            Some("internal")
        );
        assert_eq!(
            suggest_name_from_url("https://localhost:1234/v1").as_deref(),
            Some("localhost")
        );
    }

    #[test]
    fn suggest_name_rejects_ip_and_garbage() {
        assert_eq!(suggest_name_from_url("http://127.0.0.1:11434/v1"), None);
        assert_eq!(suggest_name_from_url("https://192.168.1.5/v1"), None);
        assert_eq!(suggest_name_from_url("not-a-url"), None);
        assert_eq!(suggest_name_from_url("ftp://example.com"), None);
    }

    #[test]
    fn all_returns_every_provider() {
        let providers = super::all();
        assert!(providers.len() >= 13, "expected at least 13 providers");
        assert!(providers.iter().any(|p| p.id == "openrouter"));
        assert!(providers.iter().any(|p| p.id == "groq"));
    }

    #[test]
    fn substring_match_case_insensitive() {
        let p = find_by_name_substring("My-OpenRouter-Key").unwrap();
        assert_eq!(p.id, "openrouter");

        let p = find_by_name_substring("GROQ_KEY").unwrap();
        assert_eq!(p.id, "groq");
    }

    #[test]
    fn short_input_does_not_match() {
        // Regression: "hi" previously matched "Moonshot AI (China)" because
        // the display name contained "hi" (from "china"), and would also
        // match "zhipuai" by id substring. 1–2 char inputs are never a
        // useful provider hint — require 3+ chars.
        assert!(find_by_name_substring("hi").is_none());
        assert!(find_by_name_substring("ai").is_none());
        assert!(find_by_name_substring("x").is_none());
        assert!(find_by_name_substring("cn").is_none());
    }

    #[test]
    fn descriptive_name_words_do_not_match() {
        // Words that appear only in display names (not ids) should not
        // auto-detect a provider — they're too ambiguous.
        assert!(find_by_name_substring("china").is_none());
        assert!(find_by_name_substring("gateway").is_none());
    }

    #[test]
    fn short_exact_id_still_matches() {
        // 3-char ids are the shortest and must still be detectable.
        let p = find_by_name_substring("xai").unwrap();
        assert_eq!(p.id, "xai");
        let p = find_by_name_substring("poe").unwrap();
        assert_eq!(p.id, "poe");
    }
}
