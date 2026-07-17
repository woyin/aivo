//! Token-accounting primitives for the agent engine: heuristic token estimates
//! (character-class BPE approximation, with images counted flat rather than by
//! base64 length), `usage`-object parsing, and the measured/estimate calibration
//! ratio. Pure functions with no engine state — compaction and the loop call in.

use serde_json::{Value, json};

/// Flat per-image token cost — counting the base64 verbatim would blow the budget.
pub(crate) const IMAGE_TOKEN_ESTIMATE: usize = 1_500;

/// Default recent-window size held out of compaction.
pub(crate) const KEEP_RECENT_TOKENS: usize = 20_000;

/// Ceiling on the calibration multiplier — clamps a stray measurement.
pub(crate) const MAX_CALIBRATION: f64 = 2.5;

/// Below this estimate the measured/estimate ratio is too noisy to calibrate from.
pub(crate) const CALIBRATION_MIN_SAMPLE: usize = 2_000;

/// Recent-window size held out of compaction; `AIVO_AGENT_KEEP_RECENT` overrides.
pub(crate) fn keep_recent_tokens() -> usize {
    crate::services::system_env::env_parse("AIVO_AGENT_KEEP_RECENT").unwrap_or(KEEP_RECENT_TOKENS)
}

/// Measured/estimate ratio clamped to [1.0, [`MAX_CALIBRATION`]]; `.max(1)` keeps the division safe.
pub(crate) fn calibration_ratio(measured: u64, estimate: usize) -> f64 {
    (measured as f64 / estimate.max(1) as f64).clamp(1.0, MAX_CALIBRATION)
}

/// Total tokens from a `usage` object (0 if absent). Anthropic's `input_tokens`
/// excludes its cache fields — they still occupy context, so they're added back;
/// OpenAI's `prompt_tokens` already includes them.
pub(crate) fn usage_tokens(usage: &Option<Value>) -> u64 {
    let Some(u) = usage else {
        return 0;
    };
    let num = |k: &str| u.get(k).and_then(|x| x.as_u64());
    if let Some(t) = num("total_tokens") {
        return t;
    }
    let out = num("output_tokens")
        .or_else(|| num("completion_tokens"))
        .unwrap_or(0);
    if let Some(p) = num("prompt_tokens") {
        return p + out;
    }
    num("input_tokens").unwrap_or(0)
        + num("cache_read_input_tokens").unwrap_or(0)
        + num("cache_creation_input_tokens").unwrap_or(0)
        + out
}

/// Flatten a user content value (string or multimodal array) into an array of parts.
pub(crate) fn content_to_parts(v: Value) -> Vec<Value> {
    match v {
        Value::Array(parts) => parts,
        Value::String(s) if s.is_empty() => Vec::new(),
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        other => vec![other],
    }
}

pub(crate) fn is_image_part(part: &Value) -> bool {
    part.get("type").and_then(|t| t.as_str()) == Some("image_url")
}

/// Conservative token estimate over serialized messages (see [`estimate_str_tokens`]).
pub(crate) fn estimate_tokens(messages: &[Value]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

/// Message estimate, but each image part counts as a flat [`IMAGE_TOKEN_ESTIMATE`] — its
/// base64 length would otherwise force needless compaction. Non-image messages are unchanged.
pub(crate) fn estimate_message_tokens(m: &Value) -> usize {
    if let Some(Value::Array(parts)) = m.get("content")
        && parts.iter().any(is_image_part)
    {
        let content: usize = parts
            .iter()
            .map(|p| {
                if is_image_part(p) {
                    IMAGE_TOKEN_ESTIMATE
                } else {
                    serde_json::to_string(p)
                        .map(|s| estimate_str_tokens(&s))
                        .unwrap_or(0)
                }
            })
            .sum();
        return content + 4;
    }
    serde_json::to_string(m)
        .map(|s| estimate_str_tokens(&s))
        .unwrap_or(0)
}

#[derive(Clone, Copy, PartialEq)]
enum CharClass {
    Alpha,
    Digit,
    Space,
    Punct,
    Cjk,
    Other,
}

fn char_class(c: char) -> CharClass {
    if c.is_ascii_alphabetic() {
        CharClass::Alpha
    } else if c.is_ascii_digit() {
        CharClass::Digit
    } else if c.is_whitespace() {
        CharClass::Space
    } else if c.is_ascii() {
        CharClass::Punct
    } else if is_cjk(c) {
        CharClass::Cjk
    } else {
        CharClass::Other
    }
}

/// CJK ideographs, kana, and hangul — scripts BPE tokenizes at ~1 token/char.
fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x30FF      // hiragana + katakana
        | 0x3400..=0x4DBF    // CJK extension A
        | 0x4E00..=0x9FFF    // CJK unified
        | 0xAC00..=0xD7A3    // hangul syllables
        | 0xF900..=0xFAFF    // CJK compatibility
        | 0x20000..=0x2FA1F  // CJK extensions B+
    )
}

/// Approximate BPE rates per run: words ≈1 token (splitting past 8 chars), ~3
/// digits or 4 punctuation chars per token, single spaces merge into the next
/// word, newline runs ≈1, CJK ≈1/char, other non-ASCII ≈2 chars/token.
fn run_tokens(class: CharClass, len: usize, saw_newline: bool) -> usize {
    match class {
        CharClass::Alpha => 1 + (len - 1) / 8,
        CharClass::Digit => len.div_ceil(3),
        CharClass::Space if saw_newline => 1,
        CharClass::Space if len == 1 => 0,
        CharClass::Space => 1 + (len - 2) / 8,
        CharClass::Punct => len.div_ceil(4),
        CharClass::Cjk => len,
        CharClass::Other => len.div_ceil(2),
    }
}

/// Heuristic token estimate: character-class run approximation of BPE. Flat
/// chars/4 overcounts repetitive JSON ~2x and undercounts CJK ~3x.
pub(crate) fn estimate_str_tokens(s: &str) -> usize {
    let mut total = 0usize;
    let mut run: Option<(CharClass, usize, bool)> = None;
    for c in s.chars() {
        let class = char_class(c);
        match &mut run {
            Some((current, len, saw_newline)) if *current == class => {
                *len += 1;
                *saw_newline |= c == '\n';
            }
            _ => {
                if let Some((class, len, saw_newline)) = run {
                    total += run_tokens(class, len, saw_newline);
                }
                run = Some((class, 1, c == '\n'));
            }
        }
    }
    if let Some((class, len, saw_newline)) = run {
        total += run_tokens(class, len, saw_newline);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_counts_image_flat_not_base64_length() {
        // A ~200KB base64 blob would be ~50k "tokens" at chars/4 — must count flat instead.
        let big = "A".repeat(200_000);
        let msg = json!({"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{big}")}},
        ]});
        let est = estimate_tokens(std::slice::from_ref(&msg));
        assert!(est < 3_000, "image bulk inflated the estimate: {est}");
    }

    #[test]
    fn estimate_str_counts_cjk_per_char() {
        assert_eq!(estimate_str_tokens("这是一个中文测试句子"), 10);
    }

    #[test]
    fn estimate_str_prose_tracks_word_count() {
        let est = estimate_str_tokens("The quick brown fox jumps over the lazy dog.");
        assert!((9..=12).contains(&est), "prose ≈ 1 token/word, got {est}");
    }

    #[test]
    fn estimate_str_json_schema_undercuts_flat_chars4() {
        let schema = r#"{"type":"function","function":{"name":"read_file","description":"Read a file from the workspace","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to the file being read"}},"required":["path"]}}}"#;
        let est = estimate_str_tokens(schema);
        let flat = schema.len() / 4;
        assert!(
            est < flat,
            "run classing ({est}) must undercut chars/4 ({flat})"
        );
        assert!(est > flat / 3, "and not collapse toward zero: {est}");
    }

    #[test]
    fn estimate_str_indentation_collapses() {
        assert_eq!(estimate_str_tokens("\n        "), 1);
        assert_eq!(estimate_str_tokens(""), 0);
    }

    #[test]
    fn usage_tokens_handles_both_shapes() {
        assert_eq!(usage_tokens(&Some(json!({"total_tokens": 42}))), 42);
        assert_eq!(
            usage_tokens(&Some(json!({"input_tokens": 10, "output_tokens": 5}))),
            15
        );
        assert_eq!(usage_tokens(&None), 0);
    }

    #[test]
    fn usage_tokens_adds_anthropic_cache_to_exclusive_input() {
        let u = json!({
            "input_tokens": 61, "output_tokens": 32,
            "cache_read_input_tokens": 5_000, "cache_creation_input_tokens": 120,
        });
        assert_eq!(usage_tokens(&Some(u)), 5_213);
        // Inclusive prompt_tokens shape — no double count.
        let u = json!({
            "prompt_tokens": 5_181, "completion_tokens": 32,
            "prompt_tokens_details": {"cached_tokens": 5_000},
        });
        assert_eq!(usage_tokens(&Some(u)), 5_213);
    }
}
