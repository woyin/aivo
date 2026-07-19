//! Token-usage accounting shared by the serve routers and the chat/code
//! senders: one `TokenUsage` type, one merge rule, and every usage extractor.
//!
//! Two extractor families exist ON PURPOSE — they answer different questions:
//!
//! - **Dialect-known** (`extract_openai_usage_update`, `extract_anthropic_usage_update`,
//!   `responses_usage_update`): the caller knows the wire format, so missing
//!   fields stay `None` (mid-stream partials must not clobber earlier counts)
//!   and no cross-dialect guessing happens. An OpenAI body reporting
//!   `input_tokens` + `cache_read_input_tokens` is trusted as-is (cached ⊂
//!   prompt, no add-back).
//! - **Dialect-sniffing** (`extract_usage_from_value`): routers see arbitrary
//!   upstreams, so this guesses the dialect from field shape. The same
//!   `input_tokens` + `cache_read_input_tokens` body reads as Anthropic
//!   (disjoint counts, cache added back into prompt) unless an OpenAI-style
//!   details block says otherwise. It also prefers details-style cached
//!   counts over `cache_read_input_tokens` where both appear, and 0-fills —
//!   an all-zero result means "no usage".
//!
//! Normalized meaning of the fields everywhere: `prompt_tokens` is the
//! INCLUSIVE total input (cache counts folded in); `cache_read_input_tokens`
//! and `cache_creation_input_tokens` are informational subsets.

use serde::Serialize;
use serde_json::Value;

use crate::services::http_utils::{self, parse_token_u64};
use crate::services::openai_models::extract_cached_prompt_tokens;

/// Token counts pulled from a response `usage` block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

impl TokenUsage {
    pub fn is_zero(&self) -> bool {
        *self == TokenUsage::default()
    }

    pub fn total_tokens(self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }

    /// Per-field max. Merges partial usage from successive stream events —
    /// Anthropic reports input in `message_start` and output in `message_delta`,
    /// and providers send cumulative counts, so the max is the final total.
    pub fn merge_max(&mut self, other: &TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.max(other.prompt_tokens);
        self.completion_tokens = self.completion_tokens.max(other.completion_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .max(other.cache_read_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .max(other.cache_creation_input_tokens);
    }
}

/// A partial usage report: `None` = "this event didn't mention the field",
/// which merge must treat differently from an explicit 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsageUpdate {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}

impl TokenUsageUpdate {
    pub fn is_empty(self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cache_read_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
    }

    pub fn zero_filled(self) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens.unwrap_or(0),
            completion_tokens: self.completion_tokens.unwrap_or(0),
            cache_read_input_tokens: self.cache_read_input_tokens.unwrap_or(0),
            cache_creation_input_tokens: self.cache_creation_input_tokens.unwrap_or(0),
        }
    }
}

/// Per-field max: stream counts are cumulative, and a partial Anthropic
/// `message_delta` must not clobber the cache-normalized prompt from `message_start`.
pub fn merge_token_usage(usage: &mut Option<TokenUsage>, update: TokenUsageUpdate) {
    let current = usage.get_or_insert_with(TokenUsage::default);
    if let Some(tokens) = update.prompt_tokens {
        current.prompt_tokens = current.prompt_tokens.max(tokens);
    }
    if let Some(tokens) = update.completion_tokens {
        current.completion_tokens = current.completion_tokens.max(tokens);
    }
    if let Some(tokens) = update.cache_read_input_tokens {
        current.cache_read_input_tokens = current.cache_read_input_tokens.max(tokens);
    }
    if let Some(tokens) = update.cache_creation_input_tokens {
        current.cache_creation_input_tokens = current.cache_creation_input_tokens.max(tokens);
    }
}

pub fn extract_openai_usage(body: &Value) -> Option<TokenUsage> {
    Some(extract_openai_usage_update(body)?.zero_filled())
}

pub fn extract_anthropic_usage(body: &Value) -> Option<TokenUsage> {
    Some(extract_anthropic_usage_update(body)?.zero_filled())
}

pub fn extract_responses_usage(body: &Value) -> Option<TokenUsage> {
    let update = responses_usage_update(body.get("usage")?);
    (!update.is_empty()).then(|| update.zero_filled())
}

pub fn extract_openai_usage_update(body: &Value) -> Option<TokenUsageUpdate> {
    let usage = body.get("usage")?;
    let update = TokenUsageUpdate {
        prompt_tokens: usage
            .get("prompt_tokens")
            .or_else(|| usage.get("input_tokens"))
            .and_then(parse_token_u64),
        completion_tokens: usage
            .get("completion_tokens")
            .or_else(|| usage.get("output_tokens"))
            .and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(parse_token_u64)
            .or_else(|| extract_cached_prompt_tokens(usage)),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    };
    (!update.is_empty()).then_some(update)
}

pub fn extract_anthropic_usage_update(body: &Value) -> Option<TokenUsageUpdate> {
    let usage = body.get("usage")?;
    let raw_input = usage.get("input_tokens").and_then(parse_token_u64);
    let cache_read = usage
        .get("cache_read_input_tokens")
        .and_then(parse_token_u64);
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(parse_token_u64);
    // Normalize: Anthropic's input_tokens excludes cache, so add cache to get total input
    let prompt_tokens = raw_input.map(|it| {
        it.saturating_add(cache_read.unwrap_or(0))
            .saturating_add(cache_creation.unwrap_or(0))
    });
    let update = TokenUsageUpdate {
        prompt_tokens,
        completion_tokens: usage.get("output_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: cache_creation,
    };
    (!update.is_empty()).then_some(update)
}

/// Takes the `usage` node itself (callers dig it out of `response.usage` or
/// the body first).
pub fn responses_usage_update(usage: &Value) -> TokenUsageUpdate {
    TokenUsageUpdate {
        prompt_tokens: usage.get("input_tokens").and_then(parse_token_u64),
        completion_tokens: usage.get("output_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(parse_token_u64)
            .or_else(|| {
                usage
                    .get("input_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(parse_token_u64)
            }),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    }
}

/// Pull a `TokenUsage` out of any provider's response JSON object: OpenAI chat
/// (`usage` with `prompt_tokens`/`completion_tokens`), Responses (`usage` with
/// `input_tokens`/`output_tokens`, or nested under `response`), Anthropic
/// (`usage`, or nested under `message`), or Gemini (`usageMetadata`). Returns
/// `None` when there's no usage or it's all zero.
pub fn extract_usage_from_value(v: &Value) -> Option<TokenUsage> {
    if let Some(u) = v
        .get("usage")
        .or_else(|| v.get("message").and_then(|m| m.get("usage")))
        .or_else(|| v.get("response").and_then(|r| r.get("usage")))
    {
        let num = |k: &str| u.get(k).and_then(parse_token_u64);
        // details/hit-style cached counts are ⊂ the prompt figure; Anthropic-named
        // fields are disjoint from `input_tokens` and get added back.
        let details_cached = extract_cached_prompt_tokens(u).or_else(|| {
            u.get("input_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(parse_token_u64)
        });
        let anthropic_read = num("cache_read_input_tokens");
        let cache_creation = num("cache_creation_input_tokens").unwrap_or(0);
        let prompt = match num("prompt_tokens") {
            Some(p) => p,
            None if details_cached.is_some() => num("input_tokens").unwrap_or(0),
            None => num("input_tokens")
                .unwrap_or(0)
                .saturating_add(anthropic_read.unwrap_or(0))
                .saturating_add(cache_creation),
        };
        let usage = TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: num("completion_tokens")
                .or_else(|| num("output_tokens"))
                .unwrap_or(0),
            cache_read_input_tokens: details_cached.or(anthropic_read).unwrap_or(0),
            cache_creation_input_tokens: cache_creation,
        };
        return (!usage.is_zero()).then_some(usage);
    }
    if let Some(um) = v.get("usageMetadata") {
        let n = |k: &str| um.get(k).and_then(parse_token_u64).unwrap_or(0);
        let usage = TokenUsage {
            prompt_tokens: n("promptTokenCount"),
            completion_tokens: n("candidatesTokenCount"),
            cache_read_input_tokens: n("cachedContentTokenCount"),
            cache_creation_input_tokens: 0,
        };
        return (!usage.is_zero()).then_some(usage);
    }
    None
}

/// Extract token usage from a buffered JSON response body.
pub fn parse_token_usage(body: &[u8]) -> Option<TokenUsage> {
    if let Ok(v) = serde_json::from_slice::<Value>(body) {
        return extract_usage_from_value(&v);
    }
    // Buffered SSE body — the Responses-via-chat path returns
    // text/event-stream even when buffered, so usage rides on `data:` lines
    // instead of a JSON envelope. Without this, those turns account zero.
    let mut sniffer = StreamUsageSniffer::new(true);
    sniffer.observe(body);
    sniffer.observe(b"\n");
    sniffer.finish()
}

/// Accumulates token usage from a forwarded SSE stream by scanning `data:` lines
/// for any provider's usage event. A no-op when `enabled` is false (native
/// launches don't account usage). `finish()` yields the merged per-field max.
pub struct StreamUsageSniffer {
    enabled: bool,
    pending: String,
    usage: TokenUsage,
    seen: bool,
}

impl StreamUsageSniffer {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            pending: String::new(),
            usage: TokenUsage::default(),
            seen: false,
        }
    }

    /// Feed a raw upstream chunk (native provider SSE bytes).
    pub fn observe(&mut self, chunk: &[u8]) {
        if !self.enabled {
            return;
        }
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        // Parse complete lines; keep any trailing partial line buffered. Usage
        // only rides on `data:` lines, so skip everything else.
        while let Some(nl) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=nl).collect();
            let Some(json) = http_utils::sse_data_payload(line.trim()) else {
                continue;
            };
            if json.is_empty() || json == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(json)
                && let Some(u) = extract_usage_from_value(&v)
            {
                self.usage.merge_max(&u);
                self.seen = true;
            }
        }
        // Sniffing is best-effort: a pathological newline-less stream must not
        // grow this buffer without bound, so give up rather than hold it.
        if self.pending.len() > http_utils::MAX_SSE_PENDING_BYTES {
            self.pending = String::new();
            self.enabled = false;
        }
    }

    pub fn finish(self) -> Option<TokenUsage> {
        (self.enabled && self.seen).then_some(self.usage)
    }

    /// True when usage accounting is on — gates request-side `include_usage`
    /// injection so an OpenAI chat stream emits a usage chunk to sniff.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_token_usage_openai_shape() {
        let body = json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 12,
                "prompt_tokens_details": {"cached_tokens": 8}
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens
            ),
            (30, 12, 8)
        );
    }

    #[test]
    fn parse_token_usage_buffered_sse_body() {
        // Responses-via-chat returns text/event-stream even on the buffered
        // path; usage rides on a data: line, not a JSON envelope.
        let body = "event: response.completed\n\
                    data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":21,\"output_tokens\":7}}}\n\n\
                    data: [DONE]\n";
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt_tokens, u.completion_tokens), (21, 7));
    }

    #[test]
    fn parse_token_usage_responses_shape() {
        let body = json!({
            "object": "response",
            "usage": {"input_tokens": 100, "output_tokens": 40}
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt_tokens, u.completion_tokens), (100, 40));
    }

    #[test]
    fn parse_token_usage_anthropic_shape_folds_cache_into_prompt() {
        let body = json!({
            "usage": {
                "input_tokens": 61, "output_tokens": 32,
                "cache_read_input_tokens": 5000, "cache_creation_input_tokens": 120
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens,
                u.cache_creation_input_tokens
            ),
            (5181, 32, 5000, 120)
        );
    }

    #[test]
    fn parse_token_usage_deepseek_hit_tokens_as_cache_read() {
        // DeepSeek without the OpenAI-style details block: hit tokens ⊂ prompt.
        let body = json!({
            "usage": {
                "prompt_tokens": 5000, "completion_tokens": 100,
                "prompt_cache_hit_tokens": 4800, "prompt_cache_miss_tokens": 200
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt_tokens, u.cache_read_input_tokens), (5000, 4800));
    }

    #[test]
    fn parse_token_usage_responses_cached_subset_not_double_added() {
        let body = json!({
            "object": "response",
            "usage": {
                "input_tokens": 1000, "output_tokens": 40,
                "input_tokens_details": {"cached_tokens": 800}
            }
        })
        .to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt_tokens, u.cache_read_input_tokens), (1000, 800));
    }

    #[test]
    fn parse_token_usage_none_when_absent_or_zero() {
        assert!(parse_token_usage(br#"{"choices":[]}"#).is_none());
        assert!(parse_token_usage(b"not json").is_none());
        let zero = json!({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}).to_string();
        assert!(parse_token_usage(zero.as_bytes()).is_none());
    }

    #[test]
    fn parse_token_usage_tolerates_float_and_string_counts() {
        // Some gateways report token counts as floats or strings; the sniffer
        // reads them like the dialect-known extractors do instead of dropping
        // the turn to zero.
        let body = json!({"usage": {"prompt_tokens": 30.0, "completion_tokens": "12"}}).to_string();
        let u = parse_token_usage(body.as_bytes()).unwrap();
        assert_eq!((u.prompt_tokens, u.completion_tokens), (30, 12));
    }

    #[test]
    fn sniffer_guesses_anthropic_but_openai_extractor_trusts_wire() {
        // The documented divergence: same body, different jobs. Bare
        // input_tokens + cache_read_input_tokens sniffs as Anthropic-shaped
        // (disjoint, add-back); the dialect-known OpenAI extractor trusts
        // the wire (cached ⊂ input, no add-back).
        let body = json!({"usage": {"input_tokens": 100, "cache_read_input_tokens": 20}});
        let sniffed = extract_usage_from_value(&body).unwrap();
        assert_eq!(sniffed.prompt_tokens, 120);
        let known = extract_openai_usage(&body).unwrap();
        assert_eq!(known.prompt_tokens, 100);
    }

    #[test]
    fn sniffer_disabled_is_noop() {
        let mut s = StreamUsageSniffer::new(false);
        s.observe(b"data: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n");
        assert!(s.finish().is_none());
    }

    #[test]
    fn sniffer_openai_chat_final_usage_chunk() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        s.observe(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":30,\"completion_tokens\":12,\"prompt_tokens_details\":{\"cached_tokens\":8}}}\n\n");
        s.observe(b"data: [DONE]\n\n");
        let u = s.finish().unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens
            ),
            (30, 12, 8)
        );
    }

    #[test]
    fn sniffer_anthropic_merges_start_and_delta() {
        // Anthropic splits input (message_start) and output (message_delta); its
        // disjoint cache counts fold into the inclusive prompt (100+20+5).
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":20,\"cache_creation_input_tokens\":5,\"output_tokens\":1}}}\n\n");
        s.observe(b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens,
                u.cache_creation_input_tokens
            ),
            (125, 42, 20, 5)
        );
    }

    #[test]
    fn sniffer_responses_completed_event() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":80,\"output_tokens\":25,\"input_tokens_details\":{\"cached_tokens\":10}}}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens
            ),
            (80, 25, 10)
        );
    }

    #[test]
    fn sniffer_gemini_usage_metadata() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"usageMetadata\":{\"promptTokenCount\":70,\"candidatesTokenCount\":18,\"cachedContentTokenCount\":12}}\n\n");
        let u = s.finish().unwrap();
        assert_eq!(
            (
                u.prompt_tokens,
                u.completion_tokens,
                u.cache_read_input_tokens
            ),
            (70, 18, 12)
        );
    }

    #[test]
    fn sniffer_reassembles_usage_line_split_across_chunks() {
        let mut s = StreamUsageSniffer::new(true);
        s.observe(b"data: {\"usage\":{\"prompt_tokens\":11,");
        s.observe(b"\"completion_tokens\":7}}\n");
        let u = s.finish().unwrap();
        assert_eq!((u.prompt_tokens, u.completion_tokens), (11, 7));
    }
}
