use serde_json::{Value, json};

use crate::services::openai_anthropic_bridge::{
    ANTHROPIC_SERVER_BLOCKS_EXT, ANTHROPIC_THINKING_EXT,
};
use crate::services::openai_models::{OpenAIChatResponseView, resolve_anthropic_input_and_cache};

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
            for tc in tool_calls {
                let input: Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));

                content.push(json!({
                    "type": "tool_use",
                    "id": tc.id,
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

    let mut anthropic_resp = json!({
        "id": response.id.as_deref().unwrap_or(config.fallback_id),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": config.model,
        "stop_reason": map_finish_reason(final_finish_reason),
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
    let raw_prompt = usage_value(resp, "prompt_tokens", mode);
    let output = usage_value(resp, "completion_tokens", mode);
    let usage_obj = resp.get("usage");
    let anthropic_cache_read = usage_obj
        .and_then(|u| u.get("cache_read_input_tokens"))
        .and_then(|v| v.as_u64());
    let openai_cached = usage_obj
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64());
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

fn usage_value(resp: &Value, key: &str, mode: &UsageValueMode) -> Value {
    match mode {
        UsageValueMode::CoerceU64 => json!(
            resp.get("usage")
                .and_then(|u| u.get(key))
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
        ),
        UsageValueMode::PreserveJson => resp
            .get("usage")
            .and_then(|u| u.get(key))
            .cloned()
            .unwrap_or(json!(0)),
    }
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
}
