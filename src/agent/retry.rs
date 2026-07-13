//! Retry timing and error classification for the agent's model calls: the per-turn
//! step budget, transient-vs-terminal error detection, context-overflow detection
//! (recovered by compaction, not retry), and backoff. Pure functions — the loop calls in.

use crate::agent::serve_client;

/// Sanity ceiling for a finite step budget.
pub(crate) const MAX_STEPS_CEILING: usize = 10_000;

/// Per-turn step budget: `0` = no cap (interactive default; repeat-limit and
/// esc-interrupt are the real safeties), else the value capped at [`MAX_STEPS_CEILING`].
pub(crate) fn resolve_max_steps(max_steps: u32) -> usize {
    if max_steps == 0 {
        usize::MAX
    } else {
        (max_steps as usize).min(MAX_STEPS_CEILING)
    }
}

/// Backoff before retry `n`: honor `Retry-After` (capped 30s), else exponential from
/// `AIVO_AGENT_RETRY_BASE_MS`. Mirrors the plain-chat sender.
pub(crate) fn retry_delay(
    attempt: usize,
    retry_after: Option<std::time::Duration>,
) -> std::time::Duration {
    if let Some(d) = retry_after {
        return d.min(std::time::Duration::from_secs(30));
    }
    let base = std::env::var("AIVO_AGENT_RETRY_BASE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600u64);
    std::time::Duration::from_millis(base * (1u64 << attempt.saturating_sub(1)))
}

/// Retryable on a transient status (408/429/5xx), else by message match. Overflow has
/// its own recovery path.
pub(crate) fn error_is_retryable(e: &serve_client::ServeError) -> bool {
    if is_context_overflow_error(&e.message) {
        return false;
    }
    match e.status {
        Some(s) => matches!(s, 408 | 429 | 500 | 502 | 503 | 504),
        None => is_retryable_error(&e.message),
    }
}

/// Whether an LLM/serve error is worth retrying: transient rate-limit / overload
/// / 5xx / network. Overflow (compaction handles it), auth, and bad-request aren't.
pub(crate) fn is_retryable_error(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    if is_context_overflow_error(err) {
        return false;
    }
    // Terminal errors first, so a retryable word ("connection"/"timeout") in the message can't override them. Phrases not bare codes — "400" would match "5400ms".
    const TERMINAL: &[&str] = &[
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid_api_key",
        "bad request",
        "bad_request",
    ];
    if TERMINAL.iter().any(|p| e.contains(p)) {
        return false;
    }
    const PATTERNS: &[&str] = &[
        "429",
        "500",
        "502",
        "503",
        "504",
        "overload",
        "rate limit",
        "rate_limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily",
        "service unavailable",
        "connection",
        "network",
        "fetch failed",
        "stream error",
        "request failed",
        "reset",
        "socket",
        "try again",
    ];
    PATTERNS.iter().any(|p| e.contains(p))
}

/// Short label for a retryable failure's notice — "connection issue" for a 429
/// reads as the user's network being broken, so name the class.
pub(crate) fn retryable_error_label(e: &serve_client::ServeError) -> &'static str {
    match e.status {
        Some(429) => "rate limited",
        Some(408) => "timed out",
        Some(s) if s >= 500 => "server error",
        _ => "connection issue",
    }
}

/// Terminal model-call failure → failure class + fix, then the truncated raw
/// detail. The raw `upstream NNN:` must stay in the text — the headless
/// exit-code classifier (`classify_agent_error`) parses the status from it.
pub(crate) fn terminal_error_notice(e: &serve_client::ServeError) -> String {
    let lower = e.message.to_ascii_lowercase();
    let quota = ["quota", "billing", "credit", "balance"]
        .iter()
        .any(|k| lower.contains(k));
    let advice = if is_context_overflow_error(&e.message) {
        "the conversation no longer fits the model's context window even after compacting — \
         /compact to shrink it further, or /new to start fresh"
    } else {
        match e.status {
            Some(401 | 403) if !quota => {
                "this API key was rejected — check it with `aivo keys ls`, update it with \
                 `aivo keys edit`, or re-run `aivo login`"
            }
            Some(402) => "this key is out of credits — top up, or switch keys",
            _ if quota => {
                "this key is out of quota/credits — top up, or switch key/model with /model"
            }
            Some(429) => {
                "rate limited (retries exhausted) — wait a moment and resend, or switch \
                 key/model with /model"
            }
            Some(404) => "this model isn't available on this key — pick another with /model",
            Some(s) if s >= 500 => {
                "server error (retries exhausted) — usually transient; resend in a moment"
            }
            _ => return format!("LLM error: {e}"),
        }
    };
    // One line: the notice slot renders a single row (the transcript entry wraps).
    let mut raw: String = e.message.replace(['\n', '\r'], " ");
    const MAX_RAW: usize = 240;
    if raw.chars().count() > MAX_RAW {
        raw = raw.chars().take(MAX_RAW - 1).collect::<String>() + "…";
    }
    format!("{advice} ({raw})")
}

/// Provider rejecting the request as over the model's input limit — recoverable by
/// compaction+retry. Wordings vary, hence the phrase list.
pub(crate) fn is_context_overflow_error(err: &str) -> bool {
    let e = err.to_ascii_lowercase();
    const PHRASES: &[&str] = &[
        "maximum allowed input length",
        "maximum input length",
        "context length", // also matches "maximum context length"
        "context_length",
        "context window",
        "maximum context",
        "input length of",
        "too many tokens",
        "prompt is too long",
        "reduce the length",
    ];
    PHRASES.iter().any(|p| e.contains(p))
}

/// Best-effort real token count from an overflow error, for one-shot calibration.
/// Only integers next to a token-context keyword count (so request-ids/timestamps
/// aren't picked); commas stripped; no floor, so small-window models still calibrate.
pub(crate) fn parse_overflow_actual(err: &str) -> Option<u64> {
    // Token-context words only — excludes "request"/"message"/"count" (id contexts).
    const KW: &[&str] = &[
        "token", "length", "input", "context", "exceed", "maximum", "limit", "window", "prompt",
        "allow", "than",
    ];
    // Strip grouping separators so "262,112" reads as one number.
    let norm: String = err
        .chars()
        .filter(|c| *c != ',' && *c != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    let words: Vec<&str> = norm.split_whitespace().collect();
    let kw: Vec<bool> = words
        .iter()
        .map(|w| KW.iter().any(|k| w.contains(k)))
        .collect();
    let mut best: Option<u64> = None;
    for (i, w) in words.iter().enumerate() {
        let digits: String = w.chars().filter(char::is_ascii_digit).collect();
        let Ok(n) = digits.parse::<u64>() else {
            continue; // no digits, or overflows u64
        };
        let near = kw[i] || (i > 0 && kw[i - 1]) || (i + 1 < words.len() && kw[i + 1]);
        if near && best.is_none_or(|b| n > b) {
            best = Some(n);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_max_steps() {
        assert_eq!(resolve_max_steps(0), usize::MAX); // 0 → no cap
        assert_eq!(resolve_max_steps(20), 20);
        assert_eq!(resolve_max_steps(1_000_000), MAX_STEPS_CEILING);
    }

    #[test]
    fn is_retryable_error_classifies() {
        assert!(is_retryable_error(
            "upstream 503 Service Unavailable: overloaded"
        ));
        assert!(is_retryable_error("request failed: connection refused"));
        assert!(is_retryable_error("upstream 429: rate limit exceeded"));
        // Not retryable: auth, bad request, context overflow.
        assert!(!is_retryable_error("upstream 401: invalid api key"));
        assert!(!is_retryable_error(
            "upstream 400: maximum context length exceeded"
        ));
        // Auth/bad-request stay terminal even when the message mentions a retryable word.
        assert!(!is_retryable_error(
            "401 unauthorized: connection token expired"
        ));
        assert!(!is_retryable_error("403 forbidden: network policy blocked"));
        assert!(!is_retryable_error("bad request: malformed timeout field"));
    }

    #[test]
    fn error_is_retryable_trusts_status_over_prose() {
        let err = |msg: &str, status: Option<u16>| serve_client::ServeError {
            message: msg.into(),
            status,
            retry_after: None,
        };
        assert!(error_is_retryable(&err(
            "upstream 429: slow down",
            Some(429)
        )));
        assert!(error_is_retryable(&err("upstream 503", Some(503))));
        assert!(!error_is_retryable(&err(
            "upstream 401: invalid api key",
            Some(401)
        )));
        // Status wins over prose: a 400 mentioning "timeout" is still terminal.
        assert!(!error_is_retryable(&err(
            "bad request: malformed timeout field",
            Some(400)
        )));
        // No status → fall back to the message.
        assert!(error_is_retryable(&err(
            "request failed: connection refused",
            None
        )));
        assert!(!error_is_retryable(&err(
            "context_length_exceeded",
            Some(400)
        )));
    }

    #[test]
    fn terminal_error_notice_classifies_and_keeps_the_raw_status() {
        let err = |msg: &str, status: Option<u16>| serve_client::ServeError {
            message: msg.into(),
            status,
            retry_after: None,
        };
        let auth = terminal_error_notice(&err("upstream 401: {\"error\":\"bad key\"}", Some(401)));
        assert!(auth.contains("aivo keys"), "{auth}");
        assert!(
            auth.contains("upstream 401:"),
            "the headless exit-code classifier reads the raw status: {auth}"
        );
        let quota = terminal_error_notice(&err("upstream 403: insufficient quota", Some(403)));
        assert!(quota.contains("quota"), "{quota}");
        let rate = terminal_error_notice(&err("upstream 429: slow down", Some(429)));
        assert!(rate.contains("rate limited"), "{rate}");
        let overflow = terminal_error_notice(&err(
            "upstream 400: maximum context length exceeded",
            Some(400),
        ));
        assert!(overflow.contains("/new"), "{overflow}");
        let server = terminal_error_notice(&err("upstream 503: overloaded", Some(503)));
        assert!(server.contains("server error"), "{server}");
        // Unclassifiable → the raw dump, unchanged.
        assert_eq!(
            terminal_error_notice(&err("upstream 418: teapot", Some(418))),
            "LLM error: upstream 418: teapot"
        );
        let big = format!("upstream 401: {}", "x".repeat(2000));
        let n = terminal_error_notice(&err(&big, Some(401)));
        assert!(
            n.chars().count() < 400,
            "not truncated: {} chars",
            n.chars().count()
        );
        assert!(!n.contains('\n'));
    }

    #[test]
    fn retryable_error_label_names_the_class() {
        let err = |status: Option<u16>| serve_client::ServeError {
            message: String::new(),
            status,
            retry_after: None,
        };
        assert_eq!(retryable_error_label(&err(Some(429))), "rate limited");
        assert_eq!(retryable_error_label(&err(Some(503))), "server error");
        assert_eq!(retryable_error_label(&err(None)), "connection issue");
    }

    #[test]
    fn retry_delay_honors_and_caps_retry_after() {
        use std::time::Duration;
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(12))),
            Duration::from_secs(12)
        );
        // Capped at 30s.
        assert_eq!(
            retry_delay(1, Some(Duration::from_secs(999))),
            Duration::from_secs(30)
        );
        assert!(retry_delay(1, None) > Duration::ZERO);
    }

    #[test]
    fn context_overflow_error_classified_across_providers() {
        assert!(is_context_overflow_error(
            "upstream 400 Bad Request: token count of 264378 exceeds the maximum allowed input length of 262112 tokens"
        ));
        assert!(is_context_overflow_error(
            "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens"
        ));
        assert!(is_context_overflow_error("error: context_length_exceeded"));
        assert!(!is_context_overflow_error(
            "429 Too Many Requests: rate limit exceeded"
        ));
        assert!(!is_context_overflow_error(
            "401 Unauthorized: invalid api key"
        ));
    }

    #[test]
    fn parse_overflow_actual_reads_the_token_count_not_other_numbers() {
        assert_eq!(
            parse_overflow_actual(
                "264378 exceeds the maximum allowed input length of 262112 tokens"
            ),
            Some(264378)
        );
        assert_eq!(
            parse_overflow_actual(
                "maximum context length is 128000 tokens; your messages resulted in 130000 tokens"
            ),
            Some(130000)
        );
        // A larger id/timestamp isn't next to a token keyword, so it's not picked.
        assert_eq!(
            parse_overflow_actual(
                "request 1719800000000 failed: token count 264378 exceeds the input limit of 262112"
            ),
            Some(264378)
        );
        // Grouped numerals parse; small-window counts have no floor.
        assert_eq!(
            parse_overflow_actual("prompt of 264,378 tokens exceeds the limit of 262,112"),
            Some(264378)
        );
        assert_eq!(
            parse_overflow_actual("maximum context length is 8192 tokens, resulted in 9001 tokens"),
            Some(9001)
        );
        assert_eq!(parse_overflow_actual("model laguna-m.1 returned 400"), None);
        assert_eq!(parse_overflow_actual("no numbers here"), None);
    }

    #[test]
    fn overflow_classifier_makes_error_non_retryable_even_with_transient_wording() {
        // An overflow error carrying a transient token must still be non-retryable.
        for e in [
            "connection to model failed: input exceeds the maximum allowed input length",
            "stream reset: prompt is too long for the context window",
            "request failed: 130000 tokens exceeds the maximum context length",
        ] {
            assert!(
                is_context_overflow_error(e),
                "should classify as overflow: {e}"
            );
            assert!(
                !is_retryable_error(e),
                "overflow must not be treated as a retryable transient: {e}"
            );
        }
        assert!(is_retryable_error("connection reset by peer"));
        assert!(!is_context_overflow_error("connection reset by peer"));
    }
}
