//! Shared Anthropic Messages API -> OpenAI Chat Completions request conversion.

use serde_json::{Value, json};

pub type ModelTransform = fn(&str) -> String;

#[derive(Clone, Copy, Debug)]
pub struct AnthropicToOpenAIConfig {
    pub default_model: &'static str,
    pub preserve_stream: bool,
    pub model_transform: Option<ModelTransform>,
    pub include_reasoning_content: bool,
    pub require_non_empty_reasoning_content: bool,
    pub stringify_other_tool_result_content: bool,
    pub fallback_tool_arguments_json: &'static str,
}

pub fn convert_anthropic_to_openai_request(
    body: &Value,
    config: &AnthropicToOpenAIConfig,
) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = body.get("system") {
        let system_text = match system {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            messages.push(json!({"role": "system", "content": system_text}));
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            match msg.get("content") {
                Some(Value::String(text)) => {
                    messages.push(json!({"role": role, "content": text}));
                }
                Some(Value::Array(blocks)) => {
                    convert_content_blocks(blocks, role, &mut messages, config);
                }
                _ => {
                    messages.push(json!({"role": role, "content": ""}));
                }
            }
        }
    }

    let raw_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(config.default_model);
    let model = config
        .model_transform
        .map(|transform| transform(raw_model))
        .unwrap_or_else(|| raw_model.to_string());

    let mut req = json!({
        "model": model,
        "messages": messages,
        "stream": if config.preserve_stream {
            body.get("stream").cloned().unwrap_or(json!(false))
        } else {
            json!(false)
        },
    });

    if let Some(mt) = body.get("max_tokens") {
        req["max_tokens"] = mt.clone();
    }
    if let Some(t) = body.get("temperature") {
        req["temperature"] = t.clone();
    }
    if let Some(tp) = body.get("top_p") {
        req["top_p"] = tp.clone();
    }
    if let Some(ss) = body.get("stop_sequences") {
        req["stop"] = ss.clone();
    }

    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").cloned().unwrap_or_default(),
                        "description": tool.get("description").cloned().unwrap_or(json!("")),
                        "parameters": tool.get("input_schema").cloned().unwrap_or(json!({})),
                    }
                })
            })
            .collect();
        if !openai_tools.is_empty() {
            req["tools"] = Value::Array(openai_tools);
        }
    }

    if let Some(tc) = body.get("tool_choice") {
        // Anthropic disable_parallel_tool_use → OpenAI parallel_tool_calls:false
        if tc.get("disable_parallel_tool_use") == Some(&json!(true)) {
            req["parallel_tool_calls"] = json!(false);
        }
        match tc.get("type").and_then(|t| t.as_str()) {
            Some("auto") => {
                req["tool_choice"] = json!("auto");
            }
            Some("any") => {
                req["tool_choice"] = json!("required");
            }
            Some("tool") => {
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
                }
            }
            _ => {}
        }
    }

    req
}

fn convert_content_blocks(
    blocks: &[Value],
    role: &str,
    messages: &mut Vec<Value>,
    config: &AnthropicToOpenAIConfig,
) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut thinking_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_results: Vec<(String, String)> = Vec::new();

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "thinking" if config.include_reasoning_content => {
                if let Some(thinking) = block
                    .get("thinking")
                    .and_then(|t| t.as_str())
                    .or_else(|| block.get("text").and_then(|t| t.as_str()))
                {
                    thinking_parts.push(thinking.to_string());
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = block.get("input").cloned().unwrap_or(json!({}));

                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(&input)
                            .unwrap_or_else(|_| config.fallback_tool_arguments_json.to_string()),
                    }
                }));
            }
            "tool_result" => {
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let content = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Some(v) if config.stringify_other_tool_result_content => v.to_string(),
                    _ => String::new(),
                };
                tool_results.push((tool_use_id, content));
            }
            _ => {}
        }
    }

    if !tool_results.is_empty() {
        for (tool_use_id, content) in tool_results {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }));
        }
    } else if !tool_calls.is_empty() {
        // Per OpenAI spec, content must be null (not "") when tool_calls is
        // present without text. Strict OpenAI-compatible providers reject the
        // empty-string form.
        let content = if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n"))
        };
        let mut msg = json!({"role": role, "content": content, "tool_calls": tool_calls});
        if config.include_reasoning_content {
            if role == "assistant" && config.require_non_empty_reasoning_content {
                let reasoning_content = if !thinking_parts.is_empty() {
                    thinking_parts.join("\n")
                } else {
                    let text = text_parts.join("\n");
                    if text.is_empty() {
                        " ".to_string()
                    } else {
                        text
                    }
                };
                msg["reasoning_content"] = Value::String(reasoning_content);
            } else if !thinking_parts.is_empty() {
                msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
            }
        }
        messages.push(msg);
    } else {
        let mut msg = json!({"role": role, "content": text_parts.join("\n")});
        if config.include_reasoning_content && !thinking_parts.is_empty() {
            msg["reasoning_content"] = Value::String(thinking_parts.join("\n"));
        }
        messages.push(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AnthropicToOpenAIConfig {
        AnthropicToOpenAIConfig {
            default_model: "test-model",
            preserve_stream: false,
            model_transform: None,
            include_reasoning_content: false,
            require_non_empty_reasoning_content: false,
            stringify_other_tool_result_content: false,
            fallback_tool_arguments_json: "{}",
        }
    }

    #[test]
    fn disable_parallel_tool_use_maps_to_parallel_tool_calls_false() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert_eq!(req["parallel_tool_calls"], json!(false));
        assert_eq!(req["tool_choice"], json!("auto"));
    }

    #[test]
    fn tool_choice_without_disable_parallel_omits_field() {
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"}
        });
        let req = convert_anthropic_to_openai_request(&body, &test_config());
        assert!(req.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn tool_result_image_blocks_are_text_extracted_only() {
        let config = AnthropicToOpenAIConfig {
            stringify_other_tool_result_content: false,
            ..test_config()
        };
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "Screenshot taken"},
                        {"type": "image", "source": {"type": "base64", "data": "abc"}}
                    ]
                }]
            }]
        });
        let req = convert_anthropic_to_openai_request(&body, &config);
        let tool_msg = &req["messages"][0];
        assert_eq!(tool_msg["role"], "tool");
        // Only text content extracted — image block is silently dropped
        assert_eq!(tool_msg["content"], "Screenshot taken");
    }
}
