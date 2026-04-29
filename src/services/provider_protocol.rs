#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderProtocol {
    Openai,
    Anthropic,
    Google,
    ResponsesApi,
}

/// Whether protocol paths include the `/v1` (or equivalent) version segment.
/// Some gateways serve, e.g., `/messages` instead of `/v1/messages` — probing
/// both shapes recovers from this without per-provider configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathVariant {
    Default,
    Stripped,
}

// Layout of the active-route AtomicU8:
//   bits 0-2: ProviderProtocol  (mask 0x07 → 0..=3)
//   bit  3:   PathVariant::Stripped flag (0x08)
//   bits 4-7: reserved, must be 0
//
// Pre-existing persisted values (0..=3) decode as
// `(protocol, PathVariant::Default)` since bit 3 is unset.
const PROTOCOL_MASK: u8 = 0x07;
const PATH_STRIPPED_BIT: u8 = 0x08;

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
        match v & PROTOCOL_MASK {
            1 => Self::Anthropic,
            2 => Self::Google,
            3 => Self::ResponsesApi,
            _ => Self::Openai,
        }
    }

    /// Google has a single canonical path shape (`/v1beta/models/...`); other
    /// protocols may also be served without the `/v1` prefix on some gateways.
    pub fn supports_path_variants(self) -> bool {
        !matches!(self, Self::Google)
    }
}

impl PathVariant {
    /// Apply the variant to a default `/v1`-prefixed path. `Stripped` removes a
    /// leading `/v1`; `Default` returns the path unchanged.
    pub fn apply(self, default_path: &str) -> &str {
        if matches!(self, Self::Stripped)
            && let Some(rest) = default_path.strip_prefix("/v1")
            && (rest.is_empty() || rest.starts_with('/'))
        {
            return rest;
        }
        default_path
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Stripped => "stripped",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Some(Self::Default),
            "stripped" => Some(Self::Stripped),
            _ => None,
        }
    }
}

/// Pack `(protocol, path_variant)` into the byte stored in the active-route
/// `AtomicU8`.
pub fn encode_route(protocol: ProviderProtocol, variant: PathVariant) -> u8 {
    let mut byte = protocol.to_u8();
    if matches!(variant, PathVariant::Stripped) {
        byte |= PATH_STRIPPED_BIT;
    }
    byte
}

/// Unpack the active-route byte. Backward compatible with values 0..=3 written
/// before path-variant pinning existed.
pub fn decode_route(byte: u8) -> (ProviderProtocol, PathVariant) {
    let protocol = ProviderProtocol::from_u8(byte);
    let variant = if byte & PATH_STRIPPED_BIT != 0 {
        PathVariant::Stripped
    } else {
        PathVariant::Default
    };
    (protocol, variant)
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

/// Any 4xx/5xx triggers fallback to the next protocol/path-variant candidate.
/// Real upstream errors (genuine 401/429/5xx from the configured route) are
/// preserved by the routers' first-error accumulator and surfaced after
/// exhaustion, rather than being masked by the trailing fallback's response.
pub fn is_protocol_mismatch(status: u16) -> bool {
    status >= 400
}

/// True only for HTTP statuses that genuinely indicate the endpoint path
/// doesn't exist on the upstream — distinct from `is_protocol_mismatch`, which
/// also matches auth/rate/transient errors. Use for "is this path missing?"
/// decisions where misclassifying a 401/429/5xx would persist the wrong
/// route (e.g., disabling native-Anthropic probing because of a bad API key).
pub fn is_endpoint_missing(status: u16) -> bool {
    matches!(status, 404 | 405 | 415 | 501)
}

/// True for statuses where falling back to a different protocol/path cannot
/// help: the upstream answered authoritatively (auth, rate limit, server
/// error) — not "wrong path". Routers should bail out of the fallback loop
/// on these so users see the real error fast instead of paying for 6 more
/// requests that will surface the same rejection.
///
/// 501 (Not Implemented) is excluded from the 5xx range because in practice
/// it signals "this path/method isn't served" — same family as 404/405/415.
pub fn is_terminal_upstream_error(status: u16) -> bool {
    match status {
        401 | 403 | 429 => true,
        501 => false,
        500..=599 => true,
        _ => false,
    }
}

/// True when the response body is a recognizable error envelope from a known
/// LLM API (OpenAI / Anthropic / Google). Combined with a 4xx status, this
/// signals a *semantic* rejection: the upstream parsed our request and
/// answered with its native error shape, which is proof the protocol matches.
/// Switching protocols cannot fix it; routers should bail out of the fallback
/// loop immediately rather than spending 4 more requests on candidates that
/// will return the same rejection in different shapes.
///
/// We require object-shaped error fields with at least one structured key
/// (`type`/`code`/`status`) so generic gateway responses like
/// `{"error":"Upstream request failed"}` — which usually mean "this path
/// didn't reach the real upstream" — keep flowing through the cascade.
pub fn is_request_error_envelope(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let Some(obj) = value.as_object() else {
        return false;
    };

    // Anthropic outer wrapper: { "type": "error", "error": { ... } }
    if obj.get("type").and_then(|t| t.as_str()) == Some("error")
        && obj
            .get("error")
            .and_then(|e| e.as_object())
            .is_some_and(|inner| !inner.is_empty())
    {
        return true;
    }

    // OpenAI / Anthropic-inner / Google: { "error": { "type"|"code"|"status": ... } }
    if let Some(error) = obj.get("error").and_then(|e| e.as_object()) {
        let has_type = error.get("type").and_then(|v| v.as_str()).is_some();
        let has_code = error.get("code").is_some();
        let has_status = error.get("status").and_then(|v| v.as_str()).is_some();
        if has_type || has_code || has_status {
            return true;
        }
    }

    false
}

/// If the response body names a known provider quirk, return a short identifier
/// describing which quirk fired. Used to drive diagnostics and per-key
/// auto-learning without growing the static substring list in
/// `provider_profile.rs::ProviderQuirks::for_base_url`.
///
/// The matching is intentionally narrow: a body saying "unknown field
/// reasoning_content" or "reasoning_content is not allowed" must NOT learn
/// `requires_reasoning_content` — that would teach the wrong thing for
/// providers that explicitly reject the field. We require both the field
/// name and a phrase indicating the upstream wants it preserved.
pub fn quirk_hint_for_error_body(body: &str) -> Option<&'static str> {
    let lower = body.to_ascii_lowercase();

    // Two layers, scanned independently to keep the rules orthogonal so that
    // adding a new rejection or demand phrase below cannot silently shadow
    // another quirk via substring overlap.
    //
    // `rejected` — upstream is *rejecting* the named field. `"not support"`
    // (no 'ed') is a substring of `"not supported"`, so it catches both
    // forms.
    let rejected = [
        "unknown field",
        "not allowed",
        "not support",
        "unrecognized",
        "invalid field",
        "unexpected field",
    ]
    .iter()
    .any(|p| lower.contains(p));
    // `demanded` — upstream wants the named field present.
    let demanded = [
        "must be passed",
        "must be returned",
        "missing",
        "required",
        "must participate",
    ]
    .iter()
    .any(|p| lower.contains(p));

    if lower.contains("reasoning_content") && demanded && !rejected {
        return Some("requires_reasoning_content");
    }
    if lower.contains("cache_control") && rejected {
        return Some("strips_cache_control");
    }
    if lower.contains("tool_choice") && rejected {
        return Some("tool_choice_not_supported");
    }
    if lower.contains("max_tokens") && (lower.contains("exceed") || lower.contains("too large")) {
        return Some("max_tokens_cap");
    }
    None
}

/// Classification of a failed cascade attempt. Combines the three checks
/// every router applies on `AttemptOutcome::Mismatch` so the boilerplate
/// stays in one place — and, when the status is not a 400/422, avoids
/// parsing the body for the envelope and quirk-hint scans entirely.
pub struct AttemptClassification {
    pub is_terminal: bool,
    pub is_semantic_rejection: bool,
    pub quirk_hint: Option<&'static str>,
}

pub fn classify_failed_attempt(status: u16, body: &str) -> AttemptClassification {
    let is_terminal = is_terminal_upstream_error(status);
    let is_semantic_rejection = matches!(status, 400 | 422) && is_request_error_envelope(body);
    let quirk_hint = if is_semantic_rejection {
        quirk_hint_for_error_body(body)
    } else {
        None
    };
    AttemptClassification {
        is_terminal,
        is_semantic_rejection,
        quirk_hint,
    }
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

/// Returns the path-variant candidates for `protocol`. Google ignores variants;
/// other protocols try the active variant first, then the alternative.
pub fn fallback_path_variants(protocol: ProviderProtocol, active: PathVariant) -> Vec<PathVariant> {
    if !protocol.supports_path_variants() {
        return vec![PathVariant::Default];
    }
    let other = match active {
        PathVariant::Default => PathVariant::Stripped,
        PathVariant::Stripped => PathVariant::Default,
    };
    vec![active, other]
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
    fn is_protocol_mismatch_returns_true_for_any_error_status() {
        for status in [400, 401, 403, 404, 405, 415, 422, 429, 500, 501, 502, 503] {
            assert!(is_protocol_mismatch(status), "status {status}");
        }
    }

    #[test]
    fn is_protocol_mismatch_returns_false_for_success_codes() {
        assert!(!is_protocol_mismatch(200));
        assert!(!is_protocol_mismatch(204));
        assert!(!is_protocol_mismatch(301));
        assert!(!is_protocol_mismatch(399));
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

    #[test]
    fn path_variant_apply_strips_v1_prefix() {
        assert_eq!(PathVariant::Stripped.apply("/v1/messages"), "/messages");
        assert_eq!(
            PathVariant::Stripped.apply("/v1/chat/completions"),
            "/chat/completions"
        );
        assert_eq!(PathVariant::Stripped.apply("/v1/responses"), "/responses");
    }

    #[test]
    fn path_variant_default_passes_through() {
        assert_eq!(PathVariant::Default.apply("/v1/messages"), "/v1/messages");
        assert_eq!(
            PathVariant::Default.apply("/v1/chat/completions"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn path_variant_apply_leaves_non_v1_paths_unchanged() {
        // A path that doesn't start with "/v1" is left as-is even when stripped.
        assert_eq!(
            PathVariant::Stripped.apply("/v1beta/models/x:gen"),
            "/v1beta/models/x:gen"
        );
        assert_eq!(PathVariant::Stripped.apply("/messages"), "/messages");
    }

    #[test]
    fn is_endpoint_missing_only_for_path_codes() {
        for status in [404, 405, 415, 501] {
            assert!(is_endpoint_missing(status), "status {status}");
        }
        for status in [200, 301, 400, 401, 403, 422, 429, 500, 502, 503] {
            assert!(!is_endpoint_missing(status), "status {status}");
        }
    }

    #[test]
    fn is_terminal_upstream_error_matches_auth_rate_and_5xx() {
        for status in [401, 403, 429, 500, 502, 503, 504, 505, 511] {
            assert!(is_terminal_upstream_error(status), "status {status}");
        }
    }

    #[test]
    fn is_terminal_upstream_error_excludes_path_mismatch_codes() {
        for status in [200, 301, 400, 404, 405, 415, 422, 501] {
            assert!(!is_terminal_upstream_error(status), "status {status}");
        }
    }

    #[test]
    fn is_request_error_envelope_matches_openai_shape() {
        let body = r#"{"error":{"message":"The reasoning_content in the thinking mode must be passed back to the API.","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}"#;
        assert!(is_request_error_envelope(body));
    }

    #[test]
    fn is_request_error_envelope_matches_anthropic_outer_wrapper() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#;
        assert!(is_request_error_envelope(body));
    }

    #[test]
    fn is_request_error_envelope_matches_google_shape() {
        let body =
            r#"{"error":{"code":400,"message":"Invalid argument","status":"INVALID_ARGUMENT"}}"#;
        assert!(is_request_error_envelope(body));
    }

    #[test]
    fn is_request_error_envelope_rejects_string_error_field() {
        // Generic gateway "this path didn't reach the real upstream" responses.
        // Treating these as authoritative would short-circuit the cascade
        // before we've actually probed the right path.
        assert!(!is_request_error_envelope(
            r#"{"error":"Upstream request failed"}"#
        ));
        assert!(!is_request_error_envelope(r#"{"error":"Not found"}"#));
    }

    #[test]
    fn is_request_error_envelope_rejects_unparseable_or_unrelated_json() {
        assert!(!is_request_error_envelope(""));
        assert!(!is_request_error_envelope("<html>500</html>"));
        assert!(!is_request_error_envelope(
            r#"{"detail":"validation failed"}"#
        ));
        assert!(!is_request_error_envelope(r#"{"error":{}}"#));
        assert!(!is_request_error_envelope(r#"{"type":"error"}"#));
    }

    #[test]
    fn quirk_hint_recognizes_reasoning_content_rejection() {
        let body = r#"{"error":{"message":"The reasoning_content in the thinking mode must be passed back to the API.","type":"invalid_request_error"}}"#;
        assert_eq!(
            quirk_hint_for_error_body(body),
            Some("requires_reasoning_content")
        );
    }

    #[test]
    fn quirk_hint_does_not_mislearn_when_provider_rejects_reasoning_content() {
        // Provider says "we don't accept this field" — must NOT learn
        // `requires_reasoning_content`, that would teach the wrong thing.
        for body in [
            r#"{"error":{"message":"Unknown field reasoning_content","type":"invalid_request_error"}}"#,
            r#"{"error":{"message":"reasoning_content is not allowed","type":"invalid_request_error"}}"#,
            r#"{"error":{"message":"Unrecognized field reasoning_content","type":"invalid_request_error"}}"#,
            r#"{"error":{"message":"reasoning_content is not supported by this model","type":"invalid_request_error"}}"#,
        ] {
            assert_eq!(
                quirk_hint_for_error_body(body),
                None,
                "body must not be learned as `requires_reasoning_content`: {body}"
            );
        }
    }

    #[test]
    fn quirk_hint_does_not_match_bare_field_name_without_directional_phrase() {
        // Just naming the field is not enough — could be a logging diagnostic
        // or unrelated mention. Require explicit "must be passed" /
        // "required" wording.
        let body = r#"{"error":{"message":"reasoning_content","type":"invalid_request_error"}}"#;
        assert_eq!(quirk_hint_for_error_body(body), None);
    }

    #[test]
    fn quirk_hint_recognizes_cache_control_rejection() {
        let body =
            r#"{"error":{"message":"Unknown field cache_control","type":"invalid_request_error"}}"#;
        assert_eq!(
            quirk_hint_for_error_body(body),
            Some("strips_cache_control")
        );
    }

    #[test]
    fn quirk_hint_does_not_match_bare_cache_control_name() {
        let body = r#"{"error":{"message":"cache_control","type":"invalid_request_error"}}"#;
        assert_eq!(quirk_hint_for_error_body(body), None);
    }

    #[test]
    fn quirk_hint_recognizes_tool_choice_rejection() {
        let body = r#"{"error":{"message":"This model does not support tool_choice"}}"#;
        assert_eq!(
            quirk_hint_for_error_body(body),
            Some("tool_choice_not_supported")
        );
    }

    #[test]
    fn quirk_hint_recognizes_max_tokens_rejection() {
        let body = r#"{"error":{"message":"max_tokens exceeds limit"}}"#;
        assert_eq!(quirk_hint_for_error_body(body), Some("max_tokens_cap"));
    }

    #[test]
    fn quirk_hint_returns_none_for_unrecognized_body() {
        assert_eq!(quirk_hint_for_error_body(""), None);
        assert_eq!(
            quirk_hint_for_error_body(r#"{"error":{"message":"something else"}}"#),
            None
        );
    }

    #[test]
    fn route_encoding_round_trip() {
        for proto in [
            ProviderProtocol::Openai,
            ProviderProtocol::Anthropic,
            ProviderProtocol::Google,
            ProviderProtocol::ResponsesApi,
        ] {
            for variant in [PathVariant::Default, PathVariant::Stripped] {
                let byte = encode_route(proto, variant);
                let (p2, v2) = decode_route(byte);
                assert_eq!(p2, proto, "byte {byte}");
                assert_eq!(v2, variant, "byte {byte}");
            }
        }
    }
}
