#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProtocol {
    Openai,
    Anthropic,
    Google,
    ResponsesApi,
}

impl ProviderProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::ResponsesApi => "responses",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "openai" => Some(Self::Openai),
            "anthropic" => Some(Self::Anthropic),
            "google" => Some(Self::Google),
            "responses" => Some(Self::ResponsesApi),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Openai => 0,
            Self::Anthropic => 1,
            Self::Google => 2,
            Self::ResponsesApi => 3,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Anthropic,
            2 => Self::Google,
            3 => Self::ResponsesApi,
            _ => Self::Openai,
        }
    }
}

pub fn normalize_protocol_base(base_url: &str) -> &str {
    let trimmed = base_url.trim_end_matches('/');
    [
        "/v1/messages/count_tokens",
        "/messages/count_tokens",
        "/v1/messages",
        "/messages",
        "/v1/chat/completions",
        "/chat/completions",
        "/v1beta/models",
        "/v1/models",
        "/models",
        "/v1beta",
        "/v1",
    ]
    .into_iter()
    .find_map(|suffix| trimmed.strip_suffix(suffix))
    .filter(|normalized| !normalized.is_empty())
    .unwrap_or(trimmed)
}

pub fn is_anthropic_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    let host = extract_url_host(&normalized);
    host == "api.anthropic.com" || normalized.ends_with("/anthropic")
}

pub fn is_google_endpoint(base_url: &str) -> bool {
    let normalized = normalize_protocol_base(base_url).to_ascii_lowercase();
    let host = extract_url_host(&normalized);
    host == "generativelanguage.googleapis.com"
}

fn extract_url_host(url: &str) -> &str {
    let after_scheme = url.split("://").nth(1).unwrap_or(url);
    let host_port = after_scheme.split('/').next().unwrap_or(after_scheme);
    host_port.split(':').next().unwrap_or(host_port)
}

pub fn detect_provider_protocol(base_url: &str) -> ProviderProtocol {
    if is_anthropic_endpoint(base_url) {
        ProviderProtocol::Anthropic
    } else if is_google_endpoint(base_url) {
        ProviderProtocol::Google
    } else {
        ProviderProtocol::Openai
    }
}

/// Returns true if the HTTP status suggests the endpoint path doesn't exist
/// (wrong protocol), as opposed to auth/model/rate errors.
///
/// 400/422 are deliberately excluded even though some gateways return them
/// for unknown endpoints — they also commonly signal legitimate request
/// validation errors, and misclassifying those as mismatches would mask
/// real user errors and trigger unwanted protocol switches.
pub fn is_protocol_mismatch(status: u16) -> bool {
    matches!(status, 404 | 405 | 415 | 501)
}

/// Returns fallback protocol candidates to try after `current` fails.
/// Google is always included as the last fallback so generic gateways can still
/// auto-switch to Google-native routing if they expose it.
pub fn fallback_protocols(current: ProviderProtocol) -> Vec<ProviderProtocol> {
    [
        ProviderProtocol::Openai,
        ProviderProtocol::ResponsesApi,
        ProviderProtocol::Anthropic,
        ProviderProtocol::Google,
    ]
    .into_iter()
    .filter(|p| *p != current)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_anthropic_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic"),
            ProviderProtocol::Anthropic
        );
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic/v1"),
            ProviderProtocol::Anthropic
        );
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic/v1/messages"),
            ProviderProtocol::Anthropic
        );
        assert_eq!(
            detect_provider_protocol("https://api.minimax.io/anthropic/messages/count_tokens"),
            ProviderProtocol::Anthropic
        );
    }

    #[test]
    fn detects_google_endpoint_variants() {
        assert_eq!(
            detect_provider_protocol("https://generativelanguage.googleapis.com/v1beta"),
            ProviderProtocol::Google
        );
    }

    #[test]
    fn defaults_to_openai_for_other_endpoints() {
        assert_eq!(
            detect_provider_protocol("https://openrouter.ai/api/v1"),
            ProviderProtocol::Openai
        );
    }

    #[test]
    fn is_protocol_mismatch_returns_true_for_404_405_415() {
        assert!(is_protocol_mismatch(404));
        assert!(is_protocol_mismatch(405));
        assert!(is_protocol_mismatch(415));
    }

    #[test]
    fn is_protocol_mismatch_returns_true_for_501() {
        // 501 Not Implemented is the spec-correct code for an unsupported
        // endpoint — some gateways (e.g. routed proxies that recognize the
        // path but can't serve it) return it instead of 404.
        assert!(is_protocol_mismatch(501));
    }

    #[test]
    fn is_protocol_mismatch_returns_false_for_other_codes() {
        assert!(!is_protocol_mismatch(200));
        assert!(!is_protocol_mismatch(401));
        // 400 is ambiguous (could be bad request body, beta header, etc.) —
        // handled separately by the body-inspection paths.
        assert!(!is_protocol_mismatch(400));
        assert!(!is_protocol_mismatch(500));
    }

    #[test]
    fn fallback_protocols_includes_google_for_generic_url() {
        let result = fallback_protocols(ProviderProtocol::Openai);
        assert_eq!(
            result,
            vec![
                ProviderProtocol::ResponsesApi,
                ProviderProtocol::Anthropic,
                ProviderProtocol::Google,
            ]
        );
    }

    #[test]
    fn fallback_protocols_includes_google_for_google_url() {
        let result = fallback_protocols(ProviderProtocol::Openai);
        assert!(result.contains(&ProviderProtocol::Google));
        assert!(result.contains(&ProviderProtocol::Anthropic));
    }
}
