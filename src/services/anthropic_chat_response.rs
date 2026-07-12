use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::services::http_utils;
use crate::services::openai_anthropic_bridge::{
    ANTHROPIC_SERVER_BLOCKS_EXT, ANTHROPIC_THINKING_EXT,
};
use crate::services::openai_models::{
    OpenAIChatChunk, OpenAIChatResponseView, resolve_anthropic_input_and_cache,
};

pub enum UsageValueMode {
    CoerceU64,
    PreserveJson,
}

pub struct OpenAIToAnthropicConfig<'a> {
    pub fallback_id: &'a str,
    pub model: &'a str,
    pub include_created: bool,
    pub usage_value_mode: UsageValueMode,
}

pub fn convert_openai_to_anthropic_message(
    resp: &Value,
    config: &OpenAIToAnthropicConfig<'_>,
) -> Result<Value, serde_json::Error> {
    let response: OpenAIChatResponseView = serde_json::from_value(resp.clone())?;

    let mut content: Vec<Value> = Vec::new();
    let mut final_finish_reason = "stop";

    for (choice_index, choice) in response.choices.iter().enumerate() {
        let finish_reason = choice.finish_reason.as_deref().unwrap_or("stop");

        if finish_reason == "tool_calls" {
            final_finish_reason = "tool_calls";
        } else if final_finish_reason != "tool_calls" {
            final_finish_reason = finish_reason;
        }

        // Restore thinking / redacted_thinking blocks (with signatures) from
        // the OpenAI extension field. They must come first in the Anthropic
        // content array — that's the order Anthropic emitted them, and the
        // signature is validated against that order on continuation turns.
        // The typed `OpenAIChatResponseView` doesn't carry unknown fields, so
        // pull from the raw response.
        if let Some(thinking_blocks) = resp
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(choice_index))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get(ANTHROPIC_THINKING_EXT))
            .and_then(|v| v.as_array())
        {
            for block in thinking_blocks {
                match block.get("type").and_then(|v| v.as_str()) {
                    Some("thinking") | Some("redacted_thinking") => content.push(block.clone()),
                    _ => {}
                }
            }
        }

        if let Some(text) = choice.message.content.as_deref()
            && !text.is_empty()
        {
            content.push(json!({"type": "text", "text": text}));
        }

        if let Some(tool_calls) = &choice.message.tool_calls {
            for (tool_index, tc) in tool_calls.iter().enumerate() {
                let input: Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));

                // An empty id would make the follow-up tool_result's
                // tool_use_id empty too, which strict upstreams reject on the
                // next turn (streaming synthesizes ids for the same reason).
                let id = if tc.id.is_empty() {
                    format!("toolu_{choice_index}_{tool_index}")
                } else {
                    tc.id.clone()
                };
                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": tc.function.name,
                    "input": input,
                }));
            }
        }

        // Restore Anthropic server-tool blocks at the tail of the content
        // array — they're opaque JSON we passed through verbatim.
        if let Some(server_blocks) = resp
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(choice_index))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get(ANTHROPIC_SERVER_BLOCKS_EXT))
            .and_then(|v| v.as_array())
        {
            for block in server_blocks {
                content.push(block.clone());
            }
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    // Lenient upstreams emit tool_calls with finish_reason "stop"; trusting it
    // makes Claude Code render the call but never run it, stalling the turn.
    // Truncated/filtered turns (length, content_filter) must not execute tools.
    let has_tool_use = content
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"));
    let mapped_stop_reason = map_finish_reason(final_finish_reason);
    let stop_reason = if has_tool_use
        && mapped_stop_reason == "end_turn"
        && final_finish_reason != "content_filter"
    {
        "tool_use"
    } else {
        mapped_stop_reason
    };

    let mut anthropic_resp = json!({
        "id": response.id.as_deref().unwrap_or(config.fallback_id),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": config.model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": build_anthropic_usage(resp, &config.usage_value_mode),
    });

    if config.include_created
        && let Some(created) = response.created
    {
        anthropic_resp["created"] = json!(created);
    }

    Ok(anthropic_resp)
}

/// Builds an Anthropic-shape `usage` object from an OpenAI-shape response.
///
/// Without this normalization, Anthropic clients (Claude Code) record
/// `cache_read_input_tokens = 0` for cache-aware OpenAI upstreams, which then
/// undercounts cached tokens in their session logs and downstream stats.
fn build_anthropic_usage(resp: &Value, mode: &UsageValueMode) -> Value {
    let raw_prompt = usage_value(resp, "prompt_tokens", Some("input_tokens"), mode);
    let output = usage_value(resp, "completion_tokens", Some("output_tokens"), mode);
    let usage_obj = resp.get("usage");
    let anthropic_cache_read = usage_obj
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let openai_cached =
        usage_obj.and_then(crate::services::openai_models::extract_cached_prompt_tokens);
    let cache_creation = usage_obj
        .and_then(|u| u.get("cache_creation_input_tokens"))
        .and_then(|v| v.as_u64());

    // PreserveJson mode (Copilot) keeps prompt_tokens stringly-typed; in that
    // case there's no cached info to subtract, so pass the raw Value through.
    let prompt_for_math = raw_prompt.as_u64().unwrap_or(0);
    let (input_tokens, cache_read) = resolve_anthropic_input_and_cache(
        prompt_for_math,
        anthropic_cache_read,
        openai_cached,
        cache_creation.unwrap_or(0),
    );

    let mut usage = json!({
        "input_tokens": if cache_read.is_some() || cache_creation.is_some() {
            json!(input_tokens)
        } else {
            raw_prompt
        },
        "output_tokens": output,
    });
    if let Some(value) = cache_read {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    usage
}

fn map_finish_reason(finish_reason: &str) -> &'static str {
    match finish_reason {
        "stop" => "end_turn",
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "content_filter" => "end_turn",
        _ => "end_turn",
    }
}

fn usage_value(resp: &Value, key: &str, alias: Option<&str>, mode: &UsageValueMode) -> Value {
    let usage = resp.get("usage");
    match mode {
        UsageValueMode::CoerceU64 => json!(
            usage
                .and_then(|u| u.get(key).or_else(|| alias.and_then(|name| u.get(name))))
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        ),
        UsageValueMode::PreserveJson => usage
            .and_then(|u| u.get(key).or_else(|| alias.and_then(|name| u.get(name))))
            .cloned()
            .unwrap_or(json!(0)),
    }
}

pub(crate) fn convert_openai_to_anthropic(response_body: &str, status_code: u16) -> Result<String> {
    if status_code >= 400 {
        return Ok(response_body.to_string());
    }

    let openai_resp: Value = serde_json::from_str(response_body)?;
    let anthropic_resp = convert_openai_to_anthropic_message(
        &openai_resp,
        &OpenAIToAnthropicConfig {
            fallback_id: "msg_default",
            model: openai_resp
                .get("model")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown"),
            include_created: true,
            usage_value_mode: UsageValueMode::CoerceU64,
        },
    )?;

    Ok(anthropic_resp.to_string())
}

#[derive(Default)]
struct StreamToolBlock {
    anthropic_idx: usize,
    id: String,
    name: String,
    opened: bool,
    pending_args: String,
}

fn append_sse_event(output: &mut String, event: &str, data: Value) {
    output.push_str(&http_utils::sse_event(event, &data));
}

fn ensure_message_start(
    output: &mut String,
    started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) {
    if *started {
        return;
    }
    let mut usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": 0
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    append_sse_event(
        output,
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage
            }
        }),
    );
    *started = true;
}

#[allow(clippy::too_many_arguments)]
fn emit_tool_delta(
    output: &mut String,
    block_count: &mut usize,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    openai_idx: usize,
    id: Option<&str>,
    name: Option<&str>,
    args_fragment: Option<&str>,
    saw_tool_use: &mut bool,
) {
    let block = tool_blocks.entry(openai_idx).or_insert_with(|| {
        let idx = *block_count;
        *block_count += 1;
        StreamToolBlock {
            anthropic_idx: idx,
            ..Default::default()
        }
    });

    if let Some(v) = id
        && !v.is_empty()
    {
        block.id = v.to_string();
    }
    if let Some(v) = name
        && !v.is_empty()
    {
        block.name = v.to_string();
    }

    if let Some(fragment) = args_fragment
        && !fragment.is_empty()
    {
        if block.opened {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": fragment
                    }
                }),
            );
        } else {
            block.pending_args.push_str(fragment);
        }
    }

    if !block.opened && !block.name.is_empty() {
        if block.id.is_empty() {
            block.id = http_utils::gen_id("toolu");
        }
        append_sse_event(
            output,
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": block.anthropic_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name
                }
            }),
        );
        block.opened = true;
        *saw_tool_use = true;

        if !block.pending_args.is_empty() {
            append_sse_event(
                output,
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": block.anthropic_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": block.pending_args
                    }
                }),
            );
            block.pending_args.clear();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_stream_message(
    output: &mut String,
    message_started: &mut bool,
    message_id: &str,
    model: &str,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    thinking_block_idx: &mut Option<usize>,
    text_block_idx: &mut Option<usize>,
    tool_blocks: &mut HashMap<usize, StreamToolBlock>,
    stop_reason: &str,
) {
    ensure_message_start(
        output,
        message_started,
        message_id,
        model,
        input_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
    );

    if let Some(idx) = thinking_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({"type": "content_block_stop", "index": idx}),
        );
    }

    if let Some(idx) = text_block_idx.take() {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    let mut ordered_tool_idxs = tool_blocks
        .values()
        .filter(|b| b.opened)
        .map(|b| b.anthropic_idx)
        .collect::<Vec<_>>();
    ordered_tool_idxs.sort_unstable();
    for idx in ordered_tool_idxs {
        append_sse_event(
            output,
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": idx
            }),
        );
    }

    // Repeat input fields: upstream sends usage only in the final chunk, and the SDK merges message_delta.usage into running usage seeded at message_start.
    let mut usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    append_sse_event(
        output,
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": null
            },
            "usage": usage
        }),
    );
    append_sse_event(
        output,
        "message_stop",
        json!({
            "type": "message_stop"
        }),
    );
}

pub(crate) struct OpenAIStreamConverter {
    buf: http_utils::SseLineBuffer,
    message_started: bool,
    finished: bool,
    block_count: usize,
    thinking_block_idx: Option<usize>,
    text_block_idx: Option<usize>,
    tool_blocks: HashMap<usize, StreamToolBlock>,
    message_id: String,
    model: String,
    /// Latest upstream `prompt_tokens`; total-vs-fresh semantics depend on the accompanying cache field.
    raw_prompt_tokens: u64,
    /// Latest OpenAI `cached_tokens`, retained so a later prompt-only chunk still derives correctly.
    openai_cached_tokens: Option<u64>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    pending_stop_reason: Option<&'static str>,
    /// Raw finish_reason was "content_filter" — blocks tool_use promotion.
    finish_was_content_filter: bool,
    saw_tool_use: bool,
}

impl OpenAIStreamConverter {
    pub(crate) fn new(fallback_model: &str) -> Self {
        Self {
            buf: http_utils::SseLineBuffer::new(),
            message_started: false,
            finished: false,
            block_count: 0,
            thinking_block_idx: None,
            text_block_idx: None,
            tool_blocks: HashMap::new(),
            message_id: "msg".to_string(),
            model: fallback_model.to_string(),
            raw_prompt_tokens: 0,
            openai_cached_tokens: None,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            pending_stop_reason: None,
            finish_was_content_filter: false,
            saw_tool_use: false,
        }
    }

    /// Promote end_turn→tool_use when tool_use blocks exist (lenient upstreams finish "stop"); never for length/content_filter — a truncated/filtered call must not execute.
    fn resolve_stop_reason(&self) -> &'static str {
        let fallback = if self.saw_tool_use {
            "tool_use"
        } else {
            "end_turn"
        };
        let resolved = self.pending_stop_reason.unwrap_or(fallback);
        if resolved == "end_turn" && self.saw_tool_use && !self.finish_was_content_filter {
            "tool_use"
        } else {
            resolved
        }
    }

    /// Re-derives input/cache_read from raw inputs; called on every usage update since cache and prompt info can arrive in separate chunks.
    fn recompute_anthropic_input(&mut self) {
        let creation = self.cache_creation_input_tokens.unwrap_or(0);
        let (input, cache_read) = crate::services::openai_models::resolve_anthropic_input_and_cache(
            self.raw_prompt_tokens,
            self.cache_read_input_tokens,
            self.openai_cached_tokens,
            creation,
        );
        self.input_tokens = input;
        if cache_read.is_some() {
            self.cache_read_input_tokens = cache_read;
        }
    }

    pub(crate) fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        let mut output = String::new();
        for line in self.buf.push_chunk(chunk)? {
            self.process_line(&line, &mut output)?;
        }
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        if let Some(tail) = self.buf.take_tail() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished && self.message_started {
            let stop_reason = self.resolve_stop_reason();
            finalize_stream_message(
                &mut output,
                &mut self.message_started,
                &self.message_id,
                &self.model,
                self.input_tokens,
                self.output_tokens,
                self.cache_read_input_tokens,
                self.cache_creation_input_tokens,
                &mut self.thinking_block_idx,
                &mut self.text_block_idx,
                &mut self.tool_blocks,
                stop_reason,
            );
            self.finished = true;
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = line.strip_prefix("data: ") else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let stop_reason = self.resolve_stop_reason();
                finalize_stream_message(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.output_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                    &mut self.thinking_block_idx,
                    &mut self.text_block_idx,
                    &mut self.tool_blocks,
                    stop_reason,
                );
                self.finished = true;
            }
            return Ok(());
        }

        let chunk = match serde_json::from_str::<OpenAIChatChunk>(data) {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };

        if let Some(v) = chunk.id.as_deref()
            && !v.is_empty()
        {
            self.message_id = v.to_string();
        }
        if let Some(v) = chunk.model.as_deref()
            && !v.is_empty()
        {
            self.model = v.to_string();
        }
        if let Some(usage) = chunk.usage {
            if let Some(v) = usage.prompt_tokens {
                self.raw_prompt_tokens = v;
            }
            if let Some(v) = usage.completion_tokens {
                self.output_tokens = v;
            }
            if let Some(v) = usage.cache_read_input_tokens {
                self.cache_read_input_tokens = Some(v);
            }
            if let Some(v) = usage.cache_creation_input_tokens {
                self.cache_creation_input_tokens = Some(v);
            }
            self.openai_cached_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
                .or(usage.prompt_cache_hit_tokens)
                .or(self.openai_cached_tokens);
            self.recompute_anthropic_input();
        }

        for choice in chunk.choices {
            let delta = choice.delta;

            // DeepSeek-reasoner: reasoning_content → thinking blocks
            if let Some(thinking) = delta.reasoning_content.as_deref()
                && !thinking.is_empty()
            {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                if self.thinking_block_idx.is_none() {
                    let idx = self.block_count;
                    self.block_count += 1;
                    self.thinking_block_idx = Some(idx);
                    append_sse_event(
                        output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "thinking",
                                "thinking": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.thinking_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": thinking
                        }
                    }),
                );
            }

            if let Some(text) = delta.content.as_deref()
                && !text.is_empty()
            {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                if let Some(thinking_idx) = self.thinking_block_idx.take() {
                    append_sse_event(
                        output,
                        "content_block_stop",
                        json!({"type": "content_block_stop", "index": thinking_idx}),
                    );
                }
                if self.text_block_idx.is_none() {
                    let idx = self.block_count;
                    self.block_count += 1;
                    self.text_block_idx = Some(idx);
                    append_sse_event(
                        output,
                        "content_block_start",
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "text",
                                "text": ""
                            }
                        }),
                    );
                }
                append_sse_event(
                    output,
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta",
                        "index": self.text_block_idx.unwrap_or(0),
                        "delta": {
                            "type": "text_delta",
                            "text": text
                        }
                    }),
                );
            }

            if let Some(function_call) = delta.function_call {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                emit_tool_delta(
                    output,
                    &mut self.block_count,
                    &mut self.tool_blocks,
                    0,
                    function_call.id.as_deref(),
                    function_call.name.as_deref(),
                    function_call.arguments.as_deref(),
                    &mut self.saw_tool_use,
                );
            }

            if let Some(tool_calls) = delta.tool_calls {
                ensure_message_start(
                    output,
                    &mut self.message_started,
                    &self.message_id,
                    &self.model,
                    self.input_tokens,
                    self.cache_read_input_tokens,
                    self.cache_creation_input_tokens,
                );
                for tc in tool_calls {
                    let openai_idx = tc.index.unwrap_or(0) as usize;
                    emit_tool_delta(
                        output,
                        &mut self.block_count,
                        &mut self.tool_blocks,
                        openai_idx,
                        tc.id.as_deref(),
                        tc.function.as_ref().and_then(|f| f.name.as_deref()),
                        tc.function.as_ref().and_then(|f| f.arguments.as_deref()),
                        &mut self.saw_tool_use,
                    );
                }
            }

            if let Some(finish_reason) = choice.finish_reason.as_deref()
                && !finish_reason.is_empty()
            {
                // Defer message_delta until [DONE]/EOF: some providers (xAI) send usage in a trailing chunk after finish_reason.
                self.pending_stop_reason = Some(map_finish_reason(finish_reason));
                self.finish_was_content_filter = finish_reason == "content_filter";
            }
        }

        Ok(())
    }
}

pub(crate) fn convert_openai_sse_to_anthropic(
    response_body: &str,
    status_code: u16,
) -> Result<String> {
    if status_code >= 400 {
        return Ok(format!("data: {}\n\ndata: [DONE]\n\n", response_body));
    }

    let mut converter = OpenAIStreamConverter::new("claude");
    let mut sse_output = converter.push_bytes(response_body.as_bytes())?;
    sse_output.push_str(&converter.finish()?);
    Ok(sse_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_openai_to_anthropic_message_merges_choices_and_includes_created() {
        let resp = json!({
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Let me check."},
                    "finish_reason": "stop"
                },
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 15
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "gpt-4o",
                include_created: true,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["id"], "chatcmpl-123");
        assert_eq!(result["model"], "gpt-4o");
        assert_eq!(result["created"], 1700000000);
        assert_eq!(result["stop_reason"], "tool_use");
        assert_eq!(result["usage"]["input_tokens"], 12);
        assert_eq!(result["usage"]["output_tokens"], 7);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 90);
        assert_eq!(result["usage"]["cache_creation_input_tokens"], 15);
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Let me check.");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"]["city"], "Paris");
    }

    fn tool_call_resp(finish_reason: &str) -> Value {
        json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{}"}
                    }]
                },
                "finish_reason": finish_reason
            }]
        })
    }

    fn convert_tool_call_resp(finish_reason: &str) -> Value {
        convert_openai_to_anthropic_message(
            &tool_call_resp(finish_reason),
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "gpt-4o",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap()
    }

    #[test]
    fn empty_tool_call_id_gets_synthesized() {
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                        {"id": "", "type": "function", "function": {"name": "b", "arguments": "{}"}}
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "gpt-4o",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();
        let content = result["content"].as_array().unwrap();
        let id0 = content[0]["id"].as_str().unwrap();
        let id1 = content[1]["id"].as_str().unwrap();
        assert!(!id0.is_empty() && !id1.is_empty());
        assert_ne!(id0, id1, "synthesized ids must be unique");
    }

    #[test]
    fn stop_with_tool_calls_promotes_to_tool_use() {
        assert_eq!(convert_tool_call_resp("stop")["stop_reason"], "tool_use");
    }

    #[test]
    fn length_with_tool_calls_keeps_max_tokens() {
        assert_eq!(
            convert_tool_call_resp("length")["stop_reason"],
            "max_tokens"
        );
    }

    #[test]
    fn content_filter_with_tool_calls_keeps_end_turn() {
        assert_eq!(
            convert_tool_call_resp("content_filter")["stop_reason"],
            "end_turn"
        );
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_preserves_usage_json_shape() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": "5", "completion_tokens": 3}
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_copilot",
                model: "claude-sonnet-4",
                include_created: false,
                usage_value_mode: UsageValueMode::PreserveJson,
            },
        )
        .unwrap();

        assert_eq!(result["id"], "msg_copilot");
        assert_eq!(result["model"], "claude-sonnet-4");
        assert_eq!(result["usage"]["input_tokens"], "5");
        assert_eq!(result["usage"]["output_tokens"], 3);
        assert!(result.get("created").is_none());
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_falls_back_for_empty_or_invalid_content() {
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_invalid",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{not-json"}
                    }]
                },
                "finish_reason": "length"
            }]
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "unknown",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["stop_reason"], "max_tokens");
        assert_eq!(result["usage"]["input_tokens"], 0);
        assert_eq!(result["usage"]["output_tokens"], 0);
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["id"], "call_invalid");
        assert_eq!(content[0]["name"], "read_file");
        assert_eq!(content[0]["input"], json!({}));
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_adds_empty_text_when_no_content_blocks_exist() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter"
            }]
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "unknown",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        let content = result["content"].as_array().unwrap();
        assert_eq!(result["stop_reason"], "end_turn");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0], json!({"type": "text", "text": ""}));
    }

    #[test]
    fn anthropic_thinking_blocks_restored_from_extension_field() {
        // When an OpenAI-shaped response carries the `_anthropic_thinking_blocks`
        // extension, the typed converter must lift them back into Anthropic's
        // content array, ahead of text/tool_use, so the model sees the
        // signatures in the same order it originally produced them.
        let resp = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Here it is.",
                    "_anthropic_thinking_blocks": [
                        {"type": "thinking", "thinking": "Reasoning step.", "signature": "SIG_RESP"},
                        {"type": "redacted_thinking", "data": "BLOB_RESP"}
                    ]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_t",
                model: "claude-opus-4-7",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["signature"], "SIG_RESP");
        assert_eq!(content[1]["type"], "redacted_thinking");
        assert_eq!(content[1]["data"], "BLOB_RESP");
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["text"], "Here it is.");
    }

    #[test]
    fn test_convert_openai_to_anthropic_message_malformed_response_returns_error() {
        // A non-object value that can't deserialize into OpenAIChatResponseView
        let resp = json!("not an object");
        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg_default",
                model: "unknown",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        );
        assert!(result.is_err());
    }

    /// Regression: OpenAI-compatible upstreams (zai, DeepSeek, gemma) report
    /// cached input tokens at `usage.prompt_tokens_details.cached_tokens` —
    /// not at the Anthropic-style `cache_read_input_tokens`. Without this
    /// branch, Anthropic clients log `cache_read_input_tokens: 0` and the
    /// cached portion silently disappears from `aivo stats`.
    #[test]
    fn openai_prompt_tokens_details_cached_tokens_become_cache_read_with_fresh_input() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": { "cached_tokens": 800 }
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "aivo/starter",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        // Anthropic semantics: input_tokens excludes cached input.
        assert_eq!(result["usage"]["input_tokens"], 200);
        assert_eq!(result["usage"]["output_tokens"], 50);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 800);
        assert!(result["usage"].get("cache_creation_input_tokens").is_none());
    }

    #[test]
    fn build_anthropic_usage_recognizes_deepseek_cache_hit_tokens() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_cache_hit_tokens": 90
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "deepseek-chat",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        // Anthropic semantics: input_tokens is fresh-only (100 − 90).
        assert_eq!(result["usage"]["input_tokens"], 10);
        assert_eq!(result["usage"]["output_tokens"], 10);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 90);
        assert!(result["usage"].get("cache_creation_input_tokens").is_none());
    }

    #[test]
    fn openai_input_output_token_aliases_become_anthropic_usage() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "input_tokens": 15000,
                "output_tokens": 42
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "grok-4.3",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        assert_eq!(result["usage"]["input_tokens"], 15000);
        assert_eq!(result["usage"]["output_tokens"], 42);
    }

    #[test]
    fn openai_cache_subtracts_creation_too_when_both_present() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1500,
                "completion_tokens": 10,
                "prompt_tokens_details": { "cached_tokens": 800 },
                "cache_creation_input_tokens": 500
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "m",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        assert_eq!(result["usage"]["input_tokens"], 200); // 1500 − 800 − 500
        assert_eq!(result["usage"]["cache_read_input_tokens"], 800);
        assert_eq!(result["usage"]["cache_creation_input_tokens"], 500);
    }

    /// When an Anthropic-bridge upstream re-emits Anthropic-style cache fields
    /// in an OpenAI envelope, `prompt_tokens` is already fresh-only — pass it
    /// through without subtracting.
    #[test]
    fn anthropic_shape_cache_read_passes_through_without_subtraction() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "cache_read_input_tokens": 90
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "m",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        assert_eq!(result["usage"]["input_tokens"], 12);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 90);
    }

    /// When both shapes are present, the Anthropic value wins — the upstream
    /// has already done the math and `prompt_tokens` is fresh-only.
    #[test]
    fn anthropic_cache_read_takes_precedence_over_openai_prompt_tokens_details() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "cache_read_input_tokens": 90,
                "prompt_tokens_details": { "cached_tokens": 999 }
            }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "m",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        assert_eq!(result["usage"]["input_tokens"], 12);
        assert_eq!(result["usage"]["cache_read_input_tokens"], 90);
    }

    #[test]
    fn no_cache_info_emits_only_input_and_output() {
        let resp = json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 100, "completion_tokens": 50 }
        });

        let result = convert_openai_to_anthropic_message(
            &resp,
            &OpenAIToAnthropicConfig {
                fallback_id: "msg",
                model: "m",
                include_created: false,
                usage_value_mode: UsageValueMode::CoerceU64,
            },
        )
        .unwrap();

        let usage = result["usage"].as_object().unwrap();
        assert_eq!(usage.get("input_tokens"), Some(&json!(100)));
        assert_eq!(usage.get("output_tokens"), Some(&json!(50)));
        assert!(usage.get("cache_read_input_tokens").is_none());
        assert!(usage.get("cache_creation_input_tokens").is_none());
    }

    #[test]
    fn test_convert_openai_to_anthropic_uses_response_model_and_created() {
        let openai_resp = r#"{
            "id": "chatcmpl-123",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 90,
                "cache_creation_input_tokens": 15
            }
        }"#;

        let result = convert_openai_to_anthropic(openai_resp, 200).unwrap();
        let parsed: Value = serde_json::from_str(&result).unwrap();

        assert_eq!(parsed["id"], "chatcmpl-123");
        assert_eq!(parsed["model"], "gpt-4");
        assert_eq!(parsed["created"], 1700000000);
        assert_eq!(parsed["usage"]["input_tokens"], 10);
        assert_eq!(parsed["usage"]["output_tokens"], 5);
        assert_eq!(parsed["usage"]["cache_read_input_tokens"], 90);
        assert_eq!(parsed["usage"]["cache_creation_input_tokens"], 15);
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_text() {
        let sse = "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hello \"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":4,\"cache_read_input_tokens\":90,\"cache_creation_input_tokens\":15}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("event: message_start"));
        assert!(result.contains("\"type\":\"text_delta\""));
        assert!(result.contains("\"text\":\"hello \""));
        assert!(result.contains("\"text\":\"world\""));
        assert!(result.contains("\"stop_reason\":\"end_turn\""));
        assert!(result.contains("\"cache_read_input_tokens\":90"));
        assert!(result.contains("\"cache_creation_input_tokens\":15"));
        assert!(result.contains("event: message_stop"));
    }

    /// Regression: usage arrives only in the final chunk; input_tokens must reach message_delta or the status-line percent stays at 0%.
    #[test]
    fn openai_sse_to_anthropic_propagates_input_tokens_via_message_delta() {
        let sse = "data: {\"id\":\"chatcmpl_x\",\"model\":\"deepseek-v4\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_x\",\"model\":\"deepseek-v4\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":32652,\"completion_tokens\":86}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["usage"]["input_tokens"], 32652);
        assert_eq!(payload["usage"]["output_tokens"], 86);
    }

    /// xAI/Grok emit usage as input_tokens/output_tokens; the alias mapping captures them.
    #[test]
    fn openai_sse_to_anthropic_accepts_input_output_token_names() {
        let sse = "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
		data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\
		data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["usage"]["input_tokens"], 15000);
        assert_eq!(payload["usage"]["output_tokens"], 42);
    }

    #[test]
    fn openai_sse_to_anthropic_waits_for_trailing_usage_after_finish_reason() {
        let sse = "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\
data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["delta"]["stop_reason"], "end_turn");
        assert_eq!(payload["usage"]["input_tokens"], 15000);
        assert_eq!(payload["usage"]["output_tokens"], 42);
    }

    /// Regression: cached tokens arrive at prompt_tokens_details.cached_tokens, not cache_read_input_tokens; the converter dropped them, undercounting cached usage.
    #[test]
    fn openai_sse_to_anthropic_extracts_cached_tokens_from_prompt_tokens_details() {
        let sse = "data: {\"id\":\"c\",\"model\":\"zai/glm-4.7-flash\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
data: {\"id\":\"c\",\"model\":\"zai/glm-4.7-flash\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1000,\"completion_tokens\":50,\"prompt_tokens_details\":{\"cached_tokens\":800}}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["usage"]["input_tokens"], 200); // 1000 − 800
        assert_eq!(payload["usage"]["output_tokens"], 50);
        assert_eq!(payload["usage"]["cache_read_input_tokens"], 800);
    }

    /// Both shapes present: the explicit Anthropic value wins and prompt_tokens is already fresh-only — do not subtract again.
    #[test]
    fn openai_sse_to_anthropic_prefers_explicit_cache_read_over_prompt_tokens_details() {
        let sse = "data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"cache_read_input_tokens\":90,\"prompt_tokens_details\":{\"cached_tokens\":999}}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["usage"]["input_tokens"], 12);
        assert_eq!(payload["usage"]["cache_read_input_tokens"], 90);
    }

    #[test]
    fn openai_sse_to_anthropic_extracts_cached_tokens_from_deepseek_cache_hit_field() {
        let sse = "data: {\"id\":\"c\",\"model\":\"deepseek-chat\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\
data: {\"id\":\"c\",\"model\":\"deepseek-chat\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":10,\"prompt_cache_hit_tokens\":90}}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        let delta_section = result
            .split("event: message_delta\n")
            .nth(1)
            .expect("message_delta event present");
        let data_line = delta_section
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("data line after message_delta");
        let payload: Value = serde_json::from_str(data_line.trim_start_matches("data: ")).unwrap();
        assert_eq!(payload["usage"]["input_tokens"], 10);
        assert_eq!(payload["usage"]["output_tokens"], 10);
        assert_eq!(payload["usage"]["cache_read_input_tokens"], 90);
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_split_tool_calls() {
        let sse = "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"list_files\"}}]},\"finish_reason\":null}]}\n\
data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
        let result = convert_openai_sse_to_anthropic(sse, 200).unwrap();
        assert!(result.contains("\"type\":\"tool_use\""));
        assert!(result.contains("\"id\":\"call_1\""));
        assert!(result.contains("\"name\":\"list_files\""));
        assert!(result.contains("\"type\":\"input_json_delta\""));
        assert!(result.contains("\"partial_json\":\"{\\\"path\\\":\\\".\\\"}\""));
        assert!(result.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn test_openai_stream_converter_handles_split_chunks() {
        let mut converter = OpenAIStreamConverter::new("claude");
        let mut output = String::new();

        output.push_str(
            &converter
                .push_bytes(b"data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"hel")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"lo\"},\"finish_reason\":null}]}\n")
                .unwrap(),
        );
        output.push_str(
            &converter
                .push_bytes(b"data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"completion_tokens\":2}}\n")
                .unwrap(),
        );
        output.push_str(&converter.push_bytes(b"data: [DONE]\n").unwrap());
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"text\":\"hello\""));
        assert!(output.contains("\"text\":\" world\""));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert_eq!(output.matches("event: message_stop").count(), 1);
    }

    fn stream_tool_call_turn(finish_reason: &str) -> String {
        let mut converter = OpenAIStreamConverter::new("claude");
        let mut output = String::new();
        output.push_str(
            &converter
                .push_bytes(b"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"ls\",\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}\n")
                .unwrap(),
        );
        let finish = format!(
            "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"{finish_reason}\"}}]}}\n"
        );
        output.push_str(&converter.push_bytes(finish.as_bytes()).unwrap());
        output.push_str(&converter.push_bytes(b"data: [DONE]\n").unwrap());
        output.push_str(&converter.finish().unwrap());
        output
    }

    #[test]
    fn stream_stop_with_tool_calls_promotes_to_tool_use() {
        let output = stream_tool_call_turn("stop");
        assert!(output.contains("\"stop_reason\":\"tool_use\""), "{output}");
    }

    #[test]
    fn stream_length_with_tool_calls_keeps_max_tokens() {
        let output = stream_tool_call_turn("length");
        assert!(
            output.contains("\"stop_reason\":\"max_tokens\""),
            "{output}"
        );
    }

    #[test]
    fn stream_content_filter_with_tool_calls_keeps_end_turn() {
        let output = stream_tool_call_turn("content_filter");
        assert!(output.contains("\"stop_reason\":\"end_turn\""), "{output}");
    }

    #[test]
    fn test_convert_openai_to_anthropic_error_status_passthrough() {
        let error_body = r#"{"error":{"message":"rate limited"}}"#;
        let result = convert_openai_to_anthropic(error_body, 429).unwrap();
        assert!(result.contains("rate limited"));
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_error_status_passthrough() {
        let error_body = r#"{"error":"upstream down"}"#;
        let result = convert_openai_sse_to_anthropic(error_body, 502).unwrap();
        assert!(result.contains("upstream down"));
        assert!(result.contains("data: "));
    }

    #[test]
    fn test_convert_openai_to_anthropic_empty_body() {
        let result = convert_openai_to_anthropic("", 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_openai_to_anthropic_malformed_json() {
        let result = convert_openai_to_anthropic("{not valid}", 200);
        assert!(result.is_err());
    }

    #[test]
    fn test_convert_openai_sse_to_anthropic_empty_sse() {
        let result = convert_openai_sse_to_anthropic("", 200).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_openai_stream_converter_malformed_json_in_data_line() {
        let mut converter = OpenAIStreamConverter::new("claude");
        let output = converter
            .push_bytes(b"data: {invalid json}\ndata: [DONE]\n")
            .unwrap();
        let tail = converter.finish().unwrap();
        let _ = output;
        let _ = tail;
    }

    #[test]
    fn convert_openai_sse_to_anthropic_done_only() {
        let result = convert_openai_sse_to_anthropic("data: [DONE]\n", 200).unwrap();
        assert!(
            result.contains("event: message_start"),
            "must emit message_start"
        );
        assert!(
            result.contains("event: message_stop"),
            "must emit message_stop"
        );
        assert!(
            result.contains("\"stop_reason\":\"end_turn\""),
            "must have a stop_reason"
        );
    }
}
