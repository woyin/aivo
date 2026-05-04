/**
 * Response parsing for chat: SSE chunk parsing, usage extraction, think-tag
 * handling, and content/delta extraction for OpenAI and Anthropic formats.
 */
use serde::{Deserialize, Serialize};

use crate::services::http_utils::parse_token_u64;

#[derive(Debug, Deserialize)]
pub(crate) struct ChatChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    thinking: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChatResponseChunk {
    Content(String),
    Reasoning(String),
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct AssistantResponse {
    pub content: String,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(crate) struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenUsageUpdate {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
}

impl TokenUsageUpdate {
    pub(crate) fn is_empty(self) -> bool {
        self.prompt_tokens.is_none()
            && self.completion_tokens.is_none()
            && self.cache_read_input_tokens.is_none()
            && self.cache_creation_input_tokens.is_none()
    }
}

impl TokenUsage {
    pub(crate) fn total_tokens(self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

#[derive(Debug, Default)]
pub(crate) struct ChatTurnResult {
    pub content: String,
    pub reasoning_content: Option<String>,
    pub usage: Option<TokenUsage>,
    /// Upstream model echoed by the provider's response. `None` when the
    /// response didn't carry a `model` field (e.g. Google streaming, where
    /// the model is in the URL). Stats prefers this over the user-typed
    /// alias so `aivo/starter` collapses into the upstream `deepseek-v4-flash`
    /// — same key claude-code records.
    pub model: Option<String>,
    /// Raw upstream response body, populated for non-streaming handlers.
    /// Surfaced by `aivo chat --json` so scripts can consume the
    /// provider-native shape.
    pub raw_body: Option<serde_json::Value>,
}

impl ChatTurnResult {
    /// Returns the reported usage, or estimates from text lengths (~4 chars/token).
    pub(crate) fn usage_or_estimate(&self, prompt_text: &str) -> TokenUsage {
        if let Some(usage) = self.usage {
            return usage;
        }
        let prompt_tokens = (prompt_text.len() / 4) as u64;
        let completion_tokens = (self.content.len() / 4) as u64;
        TokenUsage {
            prompt_tokens,
            completion_tokens,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        }
    }
}

// Anthropic response structs

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicDelta>,
}

#[derive(Deserialize)]
struct AnthropicDelta {
    text: Option<String>,
    thinking: Option<String>,
}

pub(crate) fn normalize_reasoning_content(reasoning: String) -> Option<String> {
    if reasoning.trim().is_empty() {
        None
    } else {
        Some(reasoning)
    }
}

pub(crate) fn extract_reasoning_part(part: &serde_json::Value) -> Option<String> {
    part.get("thinking")
        .and_then(|v| v.as_str())
        .or_else(|| part.get("reasoning_content").and_then(|v| v.as_str()))
        .or_else(|| part.get("reasoning").and_then(|v| v.as_str()))
        .or_else(|| part.get("text").and_then(|v| v.as_str()))
        .or_else(|| part.get("content").and_then(|v| v.as_str()))
        .map(ToString::to_string)
}

pub(crate) fn extract_openai_message(body: &serde_json::Value) -> AssistantResponse {
    let message = &body["choices"][0]["message"];
    let mut content_parts = Vec::new();
    let mut reasoning_parts = Vec::new();

    if let Some(reasoning) = message.get("reasoning_content").and_then(|v| v.as_str()) {
        reasoning_parts.push(reasoning.to_string());
    }

    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
        content_parts.push(content.to_string());
    } else if let Some(parts) = message.get("content").and_then(|v| v.as_array()) {
        for part in parts {
            let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if matches!(part_type, "reasoning" | "thinking") {
                if let Some(reasoning) = extract_reasoning_part(part) {
                    reasoning_parts.push(reasoning);
                }
                continue;
            }

            if let Some(text) = part
                .get("text")
                .and_then(|v| v.as_str())
                .or_else(|| part.get("content").and_then(|v| v.as_str()))
            {
                content_parts.push(text.to_string());
            }
        }
    }

    AssistantResponse {
        content: content_parts.concat(),
        reasoning_content: normalize_reasoning_content(reasoning_parts.join("")),
    }
}

pub(crate) fn extract_openai_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let update = extract_openai_usage_update(body)?;
    Some(TokenUsage {
        prompt_tokens: update.prompt_tokens.unwrap_or(0),
        completion_tokens: update.completion_tokens.unwrap_or(0),
        cache_read_input_tokens: update.cache_read_input_tokens.unwrap_or(0),
        cache_creation_input_tokens: update.cache_creation_input_tokens.unwrap_or(0),
    })
}

pub(crate) fn extract_anthropic_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let update = extract_anthropic_usage_update(body)?;
    Some(TokenUsage {
        prompt_tokens: update.prompt_tokens.unwrap_or(0),
        completion_tokens: update.completion_tokens.unwrap_or(0),
        cache_read_input_tokens: update.cache_read_input_tokens.unwrap_or(0),
        cache_creation_input_tokens: update.cache_creation_input_tokens.unwrap_or(0),
    })
}

pub(crate) fn extract_openai_usage_update(body: &serde_json::Value) -> Option<TokenUsageUpdate> {
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
            .or_else(|| {
                usage
                    .get("prompt_tokens_details")
                    .and_then(|details| details.get("cached_tokens"))
                    .and_then(parse_token_u64)
            }),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    };
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

pub(crate) fn extract_anthropic_usage_update(body: &serde_json::Value) -> Option<TokenUsageUpdate> {
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
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

pub(crate) fn parse_openai_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_openai_usage_update(&value)
}

pub(crate) fn parse_anthropic_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_anthropic_usage_update(&value)
}

/// Extract the upstream `model` field from a non-streaming response body.
/// Works uniformly for OpenAI Chat Completions, Anthropic, Copilot, and any
/// other format that exposes `body["model"]` at the top level.
pub(crate) fn extract_response_model(body: &serde_json::Value) -> Option<String> {
    body.get("model")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Run `parser` on `data` and assign to `slot` only if `slot` is still
/// `None`. Streaming handlers call this on every SSE chunk to capture the
/// upstream model from whichever chunk carries it (OpenAI: every chunk;
/// Anthropic: only `message_start`; Responses: `response.created`/
/// `response.completed`), without re-parsing once known.
pub(crate) fn capture_model(
    slot: &mut Option<String>,
    parser: fn(&str) -> Option<String>,
    data: &str,
) {
    if slot.is_none()
        && let Some(m) = parser(data)
    {
        *slot = Some(m);
    }
}

/// Extract `model` from an OpenAI Chat Completions SSE chunk. Each chunk
/// carries `{"id":..., "model":..., "choices":[...]}`, so any non-empty
/// chunk works as a source.
pub(crate) fn parse_openai_model_chunk(data: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    extract_response_model(&value)
}

/// Extract `model` from an Anthropic SSE event. Only `message_start`
/// carries `message.model`; later deltas/usage events don't repeat it.
pub(crate) fn parse_anthropic_model_chunk(data: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    if value.get("type").and_then(|t| t.as_str()) != Some("message_start") {
        return None;
    }
    value
        .get("message")
        .and_then(|m| m.get("model"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Extract the upstream model from a non-streaming Google Gemini body.
/// Gemini uses `modelVersion`; gateway-style endpoints sometimes echo
/// `model` instead, so accept either.
pub(crate) fn extract_google_model(body: &serde_json::Value) -> Option<String> {
    body.get("modelVersion")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .or_else(|| extract_response_model(body))
}

/// Extract `model` from a Responses API SSE event. Both `response.created`
/// and `response.completed` echo `response.model`; either is fine as a
/// source. Other events (text deltas, etc.) don't carry it.
pub(crate) fn parse_responses_model_chunk(data: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(data).ok()?;
    let event_type = value.get("type").and_then(|t| t.as_str())?;
    if event_type != "response.created" && event_type != "response.completed" {
        return None;
    }
    value
        .get("response")
        .and_then(|r| r.get("model"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

pub(crate) fn merge_token_usage(usage: &mut Option<TokenUsage>, update: TokenUsageUpdate) {
    let current = usage.get_or_insert_with(TokenUsage::default);
    if let Some(tokens) = update.prompt_tokens {
        current.prompt_tokens = tokens;
    }
    if let Some(tokens) = update.completion_tokens {
        current.completion_tokens = tokens;
    }
    if let Some(tokens) = update.cache_read_input_tokens {
        current.cache_read_input_tokens = tokens;
    }
    if let Some(tokens) = update.cache_creation_input_tokens {
        current.cache_creation_input_tokens = tokens;
    }
}

/// Parses a single SSE data chunk and extracts either a content or reasoning delta.
pub(crate) fn parse_sse_chunk(data: &str) -> Option<ChatResponseChunk> {
    let chunk: ChatChunk = serde_json::from_str(data).ok()?;
    let delta = &chunk.choices.first()?.delta;
    delta
        .reasoning_content
        .clone()
        .or_else(|| delta.reasoning.clone())
        .or_else(|| delta.thinking.clone())
        .filter(|text| !text.is_empty())
        .map(ChatResponseChunk::Reasoning)
        .or_else(|| {
            delta
                .content
                .clone()
                .filter(|text| !text.is_empty())
                .map(ChatResponseChunk::Content)
        })
}

/// Parses a Responses API SSE data line and returns a text delta if present.
pub(crate) fn parse_responses_chunk(data: &str) -> Option<ChatResponseChunk> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let event_type = value.get("type")?.as_str()?;
    if event_type == "response.output_text.delta" {
        let delta = value.get("delta")?.as_str()?;
        if delta.is_empty() {
            None
        } else {
            Some(ChatResponseChunk::Content(delta.to_string()))
        }
    } else {
        None
    }
}

/// Extracts usage from a Responses API SSE `response.completed` event.
pub(crate) fn parse_responses_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let event_type = value.get("type")?.as_str()?;
    if event_type != "response.completed" {
        return None;
    }
    let usage = value.get("response")?.get("usage")?;
    let update = TokenUsageUpdate {
        prompt_tokens: usage.get("input_tokens").and_then(parse_token_u64),
        completion_tokens: usage.get("output_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(parse_token_u64),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    };
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

/// Extracts the assistant message from a non-streaming Responses API response.
pub(crate) fn extract_responses_message(body: &serde_json::Value) -> AssistantResponse {
    let mut content_parts = Vec::new();
    for output in body
        .get("output")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        if output.get("type").and_then(|v| v.as_str()) == Some("message") {
            for part in output
                .get("content")
                .and_then(|v| v.as_array())
                .into_iter()
                .flatten()
            {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    content_parts.push(text.to_string());
                }
            }
        }
    }
    AssistantResponse {
        content: content_parts.concat(),
        reasoning_content: None,
    }
}

/// Extracts usage from a non-streaming Responses API response.
pub(crate) fn extract_responses_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let usage = body.get("usage")?;
    let update = TokenUsageUpdate {
        prompt_tokens: usage.get("input_tokens").and_then(parse_token_u64),
        completion_tokens: usage.get("output_tokens").and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cache_read_input_tokens")
            .and_then(parse_token_u64),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(parse_token_u64),
    };
    if update.is_empty() {
        None
    } else {
        Some(TokenUsage {
            prompt_tokens: update.prompt_tokens.unwrap_or(0),
            completion_tokens: update.completion_tokens.unwrap_or(0),
            cache_read_input_tokens: update.cache_read_input_tokens.unwrap_or(0),
            cache_creation_input_tokens: update.cache_creation_input_tokens.unwrap_or(0),
        })
    }
}

/// Parses a Google Gemini SSE data line and extracts text content.
pub(crate) fn parse_google_chunk(data: &str) -> Option<ChatResponseChunk> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let candidate = value.get("candidates")?.as_array()?.first()?;
    let parts = candidate.get("content")?.get("parts")?.as_array()?;
    let mut text = String::new();
    for part in parts {
        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
            text.push_str(t);
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(ChatResponseChunk::Content(text))
    }
}

/// Extracts usage from a Google Gemini SSE chunk (usageMetadata may appear in any chunk).
pub(crate) fn parse_google_usage_chunk(data: &str) -> Option<TokenUsageUpdate> {
    let value: serde_json::Value = serde_json::from_str(data).ok()?;
    let usage = value.get("usageMetadata")?;
    let update = TokenUsageUpdate {
        prompt_tokens: usage.get("promptTokenCount").and_then(parse_token_u64),
        completion_tokens: usage.get("candidatesTokenCount").and_then(parse_token_u64),
        cache_read_input_tokens: usage
            .get("cachedContentTokenCount")
            .and_then(parse_token_u64),
        cache_creation_input_tokens: None,
    };
    if update.is_empty() {
        None
    } else {
        Some(update)
    }
}

/// Extracts the assistant message from a non-streaming Google Gemini response.
pub(crate) fn extract_google_message(body: &serde_json::Value) -> AssistantResponse {
    let mut text_parts = Vec::new();
    if let Some(candidates) = body.get("candidates").and_then(|v| v.as_array())
        && let Some(candidate) = candidates.first()
        && let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
    {
        for part in parts {
            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                text_parts.push(text.to_string());
            }
        }
    }
    AssistantResponse {
        content: text_parts.concat(),
        reasoning_content: None,
    }
}

/// Extracts usage from a non-streaming Google Gemini response.
pub(crate) fn extract_google_usage(body: &serde_json::Value) -> Option<TokenUsage> {
    let usage = body.get("usageMetadata")?;
    let prompt = usage
        .get("promptTokenCount")
        .and_then(parse_token_u64)
        .unwrap_or(0);
    let completion = usage
        .get("candidatesTokenCount")
        .and_then(parse_token_u64)
        .unwrap_or(0);
    let cache_read = usage
        .get("cachedContentTokenCount")
        .and_then(parse_token_u64)
        .unwrap_or(0);
    Some(TokenUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cache_read_input_tokens: cache_read,
        cache_creation_input_tokens: 0,
    })
}

/// Parses an Anthropic SSE data line and returns either a text or thinking delta.
pub(crate) fn parse_anthropic_chunk(data: &str) -> Option<ChatResponseChunk> {
    let event: AnthropicStreamEvent = serde_json::from_str(data).ok()?;
    if event.event_type == "content_block_delta" {
        let delta = event.delta?;
        delta
            .thinking
            .filter(|text| !text.is_empty())
            .map(ChatResponseChunk::Reasoning)
            .or_else(|| {
                delta
                    .text
                    .filter(|text| !text.is_empty())
                    .map(ChatResponseChunk::Content)
            })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::http_utils::sse_data_payload;

    #[test]
    fn extract_response_model_reads_top_level_field() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "deepseek-v4-flash",
            "choices": [],
        });
        assert_eq!(
            extract_response_model(&body).as_deref(),
            Some("deepseek-v4-flash")
        );
    }

    #[test]
    fn extract_response_model_returns_none_when_missing_or_blank() {
        let missing = serde_json::json!({"id": "x"});
        assert_eq!(extract_response_model(&missing), None);
        let blank = serde_json::json!({"model": "   "});
        assert_eq!(extract_response_model(&blank), None);
    }

    #[test]
    fn parse_openai_model_chunk_reads_streaming_chunk() {
        let chunk = r#"{"id":"chatcmpl-x","model":"deepseek-v4-flash","choices":[{"delta":{"content":"hi"}}]}"#;
        assert_eq!(
            parse_openai_model_chunk(chunk).as_deref(),
            Some("deepseek-v4-flash")
        );
    }

    #[test]
    fn parse_anthropic_model_chunk_only_message_start() {
        let start = r#"{"type":"message_start","message":{"id":"m","model":"claude-sonnet-4-6"}}"#;
        assert_eq!(
            parse_anthropic_model_chunk(start).as_deref(),
            Some("claude-sonnet-4-6")
        );
        // Other event types must not surface a model name.
        let delta = r#"{"type":"content_block_delta","delta":{"text":"hi"}}"#;
        assert_eq!(parse_anthropic_model_chunk(delta), None);
    }

    #[test]
    fn parse_responses_model_chunk_handles_created_and_completed() {
        let created = r#"{"type":"response.created","response":{"model":"gpt-5"}}"#;
        let completed = r#"{"type":"response.completed","response":{"model":"gpt-5"}}"#;
        let other = r#"{"type":"response.output_text.delta","delta":"x"}"#;
        assert_eq!(
            parse_responses_model_chunk(created).as_deref(),
            Some("gpt-5")
        );
        assert_eq!(
            parse_responses_model_chunk(completed).as_deref(),
            Some("gpt-5")
        );
        assert_eq!(parse_responses_model_chunk(other), None);
    }

    #[test]
    fn test_parse_sse_chunk_with_content() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}"#;
        assert_eq!(
            parse_sse_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
    }

    #[test]
    fn test_parse_sse_chunk_empty_delta() {
        let data = r#"{"id":"chatcmpl-1","choices":[{"delta":{}}]}"#;
        assert_eq!(parse_sse_chunk(data), None);
    }

    #[test]
    fn test_parse_sse_chunk_invalid_json() {
        assert_eq!(parse_sse_chunk("not json"), None);
    }

    #[test]
    fn test_parse_sse_chunk_no_choices() {
        let data = r#"{"id":"chatcmpl-1","choices":[]}"#;
        assert_eq!(parse_sse_chunk(data), None);
    }

    #[test]
    fn test_sse_data_payload_with_optional_space() {
        assert_eq!(
            sse_data_payload(r#"data: {"choices":[]}"#),
            Some(r#"{"choices":[]}"#)
        );
        assert_eq!(
            sse_data_payload(r#"data:{"choices":[]}"#),
            Some(r#"{"choices":[]}"#)
        );
    }

    #[test]
    fn test_extract_openai_message_string_and_parts() {
        let text = serde_json::json!({
            "choices": [{"message": {"content": "hello"}}]
        });
        assert_eq!(
            extract_openai_message(&text),
            AssistantResponse {
                content: "hello".to_string(),
                reasoning_content: None,
            }
        );

        let parts = serde_json::json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type":"text", "text":"hello "},
                        {"type":"text", "text":"world"}
                    ]
                }
            }]
        });
        assert_eq!(
            extract_openai_message(&parts),
            AssistantResponse {
                content: "hello world".to_string(),
                reasoning_content: None,
            }
        );
    }

    #[test]
    fn test_extract_openai_message_reasoning_content() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "answer",
                    "reasoning_content": "step by step"
                }
            }]
        });

        assert_eq!(
            extract_openai_message(&body),
            AssistantResponse {
                content: "answer".to_string(),
                reasoning_content: Some("step by step".to_string()),
            }
        );
    }

    #[test]
    fn test_parse_anthropic_chunk_with_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        assert_eq!(
            parse_anthropic_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
    }

    #[test]
    fn test_parse_anthropic_chunk_with_thinking() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need to inspect files."}}"#;
        assert_eq!(
            parse_anthropic_chunk(data),
            Some(ChatResponseChunk::Reasoning(
                "Need to inspect files.".to_string()
            ))
        );
    }

    #[test]
    fn test_parse_anthropic_chunk_non_delta_event() {
        let data = r#"{"type":"message_start","message":{"id":"msg_1"}}"#;
        assert_eq!(parse_anthropic_chunk(data), None);
    }

    #[test]
    fn test_parse_anthropic_chunk_ping() {
        let data = r#"{"type":"ping"}"#;
        assert_eq!(parse_anthropic_chunk(data), None);
    }

    #[test]
    fn test_parse_anthropic_chunk_invalid_json() {
        assert_eq!(parse_anthropic_chunk("not json"), None);
    }

    #[test]
    fn test_merge_openai_stream_usage_across_chunks() {
        let mut usage = None;
        merge_token_usage(
            &mut usage,
            parse_openai_usage_chunk(r#"{"usage":{"prompt_tokens":24}}"#).unwrap(),
        );
        merge_token_usage(
            &mut usage,
            parse_openai_usage_chunk(r#"{"usage":{"completion_tokens":11}}"#).unwrap(),
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 24,
                completion_tokens: 11,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_openai_usage_accepts_input_output_aliases() {
        let body = serde_json::json!({
            "usage": {
                "input_tokens": 15000,
                "output_tokens": 42
            }
        });

        assert_eq!(
            extract_openai_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 15000,
                completion_tokens: 42,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_merge_anthropic_stream_usage_across_events() {
        let mut usage = None;
        merge_token_usage(
            &mut usage,
            parse_anthropic_usage_chunk(r#"{"usage":{"input_tokens":12}}"#).unwrap(),
        );
        merge_token_usage(
            &mut usage,
            parse_anthropic_usage_chunk(r#"{"usage":{"output_tokens":7}}"#).unwrap(),
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                prompt_tokens: 12,
                completion_tokens: 7,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_openai_usage_accepts_numeric_strings() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": "10",
                "completion_tokens": "5"
            }
        });

        assert_eq!(
            extract_openai_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_anthropic_usage_accepts_numeric_strings() {
        let body = serde_json::json!({
            "usage": {
                "input_tokens": "8",
                "output_tokens": "3",
                "cache_read_input_tokens": "90",
                "cache_creation_input_tokens": "15"
            }
        });

        assert_eq!(
            extract_anthropic_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 113,
                completion_tokens: 3,
                cache_read_input_tokens: 90,
                cache_creation_input_tokens: 15,
            })
        );
    }

    #[test]
    fn test_parse_responses_chunk_text_delta() {
        let data = r#"{"type":"response.output_text.delta","delta":"Hello","item_id":"msg_1","output_index":0,"content_index":0}"#;
        assert_eq!(
            parse_responses_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
    }

    #[test]
    fn test_parse_responses_chunk_non_delta_event() {
        let data = r#"{"type":"response.created","response":{"id":"resp_1"}}"#;
        assert_eq!(parse_responses_chunk(data), None);
    }

    #[test]
    fn test_parse_responses_usage_chunk() {
        let data = r#"{"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#;
        let update = parse_responses_usage_chunk(data).unwrap();
        assert_eq!(update.prompt_tokens, Some(10));
        assert_eq!(update.completion_tokens, Some(5));
    }

    #[test]
    fn test_parse_responses_usage_chunk_non_completed() {
        let data = r#"{"type":"response.output_text.delta","delta":"hi"}"#;
        assert_eq!(parse_responses_usage_chunk(data), None);
    }

    #[test]
    fn test_extract_responses_message() {
        let body = serde_json::json!({
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello world"}]
            }]
        });
        assert_eq!(
            extract_responses_message(&body),
            AssistantResponse {
                content: "Hello world".to_string(),
                reasoning_content: None,
            }
        );
    }

    #[test]
    fn test_extract_responses_usage() {
        let body = serde_json::json!({
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15
            }
        });
        assert_eq!(
            extract_responses_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_extract_openai_usage_reads_cached_tokens_details() {
        let body = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {
                    "cached_tokens": 90
                }
            }
        });

        assert_eq!(
            extract_openai_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 90,
                cache_creation_input_tokens: 0,
            })
        );
    }

    #[test]
    fn test_parse_google_chunk_with_text() {
        let data = r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}"#;
        assert_eq!(
            parse_google_chunk(data),
            Some(ChatResponseChunk::Content("Hello".to_string()))
        );
    }

    #[test]
    fn test_parse_google_chunk_empty_parts() {
        let data = r#"{"candidates":[{"content":{"parts":[],"role":"model"}}]}"#;
        assert_eq!(parse_google_chunk(data), None);
    }

    #[test]
    fn test_parse_google_chunk_no_candidates() {
        let data = r#"{"candidates":[]}"#;
        assert_eq!(parse_google_chunk(data), None);
    }

    #[test]
    fn test_parse_google_chunk_invalid_json() {
        assert_eq!(parse_google_chunk("not json"), None);
    }

    #[test]
    fn test_parse_google_usage_chunk() {
        let data = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}"#;
        let update = parse_google_usage_chunk(data).unwrap();
        assert_eq!(update.prompt_tokens, Some(10));
        assert_eq!(update.completion_tokens, Some(5));
    }

    #[test]
    fn test_parse_google_usage_chunk_with_cache() {
        let data = r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"cachedContentTokenCount":90}}"#;
        let update = parse_google_usage_chunk(data).unwrap();
        assert_eq!(update.cache_read_input_tokens, Some(90));
    }

    #[test]
    fn test_parse_google_usage_chunk_no_metadata() {
        let data = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#;
        assert_eq!(parse_google_usage_chunk(data), None);
    }

    #[test]
    fn test_extract_google_message() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello "}, {"text": "world"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        });
        assert_eq!(
            extract_google_message(&body),
            AssistantResponse {
                content: "Hello world".to_string(),
                reasoning_content: None,
            }
        );
    }

    #[test]
    fn test_extract_google_message_empty() {
        let body = serde_json::json!({"candidates": []});
        assert_eq!(
            extract_google_message(&body),
            AssistantResponse {
                content: String::new(),
                reasoning_content: None,
            }
        );
    }

    #[test]
    fn test_extract_google_usage() {
        let body = serde_json::json!({
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });
        assert_eq!(
            extract_google_usage(&body),
            Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            })
        );
    }
}
