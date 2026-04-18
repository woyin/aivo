use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::services::http_utils::{current_unix_ts, sse_data_payload};

/// Drains the next complete SSE line from `pending`, returning its text
/// with the trailing `\r` stripped. Returns `None` when no newline is in
/// the buffer. Newlines (0x0A) never appear inside a multi-byte UTF-8
/// sequence, so each complete line is valid UTF-8.
fn drain_sse_line(pending: &mut Vec<u8>) -> Option<String> {
    let pos = pending.iter().position(|&b| b == b'\n')?;
    let line = String::from_utf8_lossy(&pending[..pos]).into_owned();
    pending.drain(..=pos);
    Some(line.trim_end_matches('\r').to_string())
}

/// Decodes any remaining bytes in `pending` as a trailing SSE line and
/// returns the trimmed text if non-empty. Consumes the buffer.
fn drain_sse_tail(pending: &mut Vec<u8>) -> Option<String> {
    if pending.is_empty() {
        return None;
    }
    let tail = String::from_utf8_lossy(pending).into_owned();
    pending.clear();
    let trimmed = tail.trim_end_matches('\r').trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[derive(Default)]
struct AnthropicToolCallState {
    id: String,
    name: String,
}

pub(crate) struct AnthropicToOpenAIStreamConverter {
    pending: Vec<u8>,
    id: String,
    model: String,
    fallback_model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    tool_calls: HashMap<usize, AnthropicToolCallState>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
}

pub(crate) struct GeminiToOpenAIStreamConverter {
    pending: Vec<u8>,
    id: String,
    model: String,
    created: u64,
    role_sent: bool,
    finished: bool,
    saw_tool_call: bool,
    next_tool_index: usize,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read_input_tokens: Option<u64>,
}

impl AnthropicToOpenAIStreamConverter {
    pub(crate) fn new(fallback_model: &str) -> Self {
        Self {
            pending: Vec::new(),
            id: "chatcmpl-aivo".to_string(),
            model: String::new(),
            fallback_model: fallback_model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            tool_calls: HashMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        }
    }

    pub(crate) fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.extend_from_slice(chunk);
        let mut output = String::new();

        while let Some(line) = drain_sse_line(&mut self.pending) {
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        if let Some(tail) = drain_sse_tail(&mut self.pending) {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        match event.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "message_start" => {
                if let Some(message) = event.get("message") {
                    if let Some(id) = message.get("id").and_then(|v| v.as_str())
                        && !id.is_empty()
                    {
                        self.id = id.to_string();
                    }
                    if let Some(model) = message.get("model").and_then(|v| v.as_str())
                        && !model.is_empty()
                    {
                        self.model = model.to_string();
                    }
                    if let Some(usage) = message.get("usage") {
                        if let Some(input_tokens) =
                            usage.get("input_tokens").and_then(|v| v.as_u64())
                        {
                            self.input_tokens = input_tokens;
                        }
                        self.cache_read_input_tokens = usage
                            .get("cache_read_input_tokens")
                            .and_then(|v| v.as_u64());
                        self.cache_creation_input_tokens = usage
                            .get("cache_creation_input_tokens")
                            .and_then(|v| v.as_u64());
                    }
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if event
                    .get("content_block")
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str())
                    == Some("tool_use")
                {
                    let block = event
                        .get("content_block")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .filter(|v| !v.is_empty())
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| format!("call_{index}"));
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    self.tool_calls.insert(
                        index,
                        AnthropicToolCallState {
                            id: id.clone(),
                            name: name.clone(),
                        },
                    );
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        self.model_name(),
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name
                                }
                            }]
                        }),
                        Value::Null,
                        None,
                    ));
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = event.get("delta").cloned().unwrap_or_else(|| json!({}));
                match delta.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                self.model_name(),
                                json!({ "content": text }),
                                Value::Null,
                                None,
                            ));
                        }
                    }
                    "input_json_delta" => {
                        if let Some(partial_json) =
                            delta.get("partial_json").and_then(|v| v.as_str())
                        {
                            let (id, name) = {
                                let tool = self.tool_calls.entry(index).or_default();
                                let id = if tool.id.is_empty() {
                                    format!("call_{index}")
                                } else {
                                    tool.id.clone()
                                };
                                (id, tool.name.clone())
                            };
                            self.emit_role_if_needed(output);
                            output.push_str(&openai_sse_chunk(
                                &self.id,
                                self.created,
                                self.model_name(),
                                json!({
                                    "tool_calls": [{
                                        "index": index,
                                        "id": id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": partial_json
                                        }
                                    }]
                                }),
                                Value::Null,
                                None,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(usage) = event.get("usage") {
                    if let Some(output_tokens) = usage.get("output_tokens").and_then(|v| v.as_u64())
                    {
                        self.output_tokens = output_tokens;
                    }
                    if let Some(cache_read_input_tokens) = usage
                        .get("cache_read_input_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        self.cache_read_input_tokens = Some(cache_read_input_tokens);
                    }
                    if let Some(cache_creation_input_tokens) = usage
                        .get("cache_creation_input_tokens")
                        .and_then(|v| v.as_u64())
                    {
                        self.cache_creation_input_tokens = Some(cache_creation_input_tokens);
                    }
                }
                if let Some(stop_reason) = event
                    .get("delta")
                    .and_then(|v| v.get("stop_reason"))
                    .and_then(|v| v.as_str())
                {
                    self.emit_finish(output, map_anthropic_stop_reason(stop_reason));
                }
            }
            "message_stop" if !self.finished => {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            _ => {}
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            self.model_name(),
            json!({ "role": "assistant" }),
            Value::Null,
            None,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        // Normalize: Anthropic's input_tokens excludes cache, OpenAI's prompt_tokens includes it
        let prompt_tokens = self
            .input_tokens
            .saturating_add(self.cache_read_input_tokens.unwrap_or(0))
            .saturating_add(self.cache_creation_input_tokens.unwrap_or(0));
        let usage = Some(openai_usage_json(
            prompt_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        ));
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            self.model_name(),
            json!({}),
            json!(finish_reason),
            usage,
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }

    fn model_name(&self) -> &str {
        if self.model.is_empty() {
            &self.fallback_model
        } else {
            &self.model
        }
    }
}

impl GeminiToOpenAIStreamConverter {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            pending: Vec::new(),
            id: "chatcmpl-aivo".to_string(),
            model: model.to_string(),
            created: current_unix_ts(),
            role_sent: false,
            finished: false,
            saw_tool_call: false,
            next_tool_index: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            cache_read_input_tokens: None,
        }
    }

    pub(crate) fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.extend_from_slice(chunk);
        let mut output = String::new();

        while let Some(line) = drain_sse_line(&mut self.pending) {
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        if let Some(tail) = drain_sse_tail(&mut self.pending) {
            self.process_line(&tail, &mut output)?;
        }

        if !self.finished {
            let finish_reason = if self.saw_tool_call {
                "tool_calls"
            } else {
                "stop"
            };
            self.emit_finish(&mut output, finish_reason);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.finished {
                let finish_reason = if self.saw_tool_call {
                    "tool_calls"
                } else {
                    "stop"
                };
                self.emit_finish(output, finish_reason);
            }
            return Ok(());
        }

        let event: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        if let Some(id) = event.get("responseId").and_then(|v| v.as_str())
            && !id.is_empty()
        {
            self.id = id.to_string();
        }

        let candidate = event
            .get("candidates")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| json!({}));

        if let Some(usage) = event.get("usageMetadata") {
            if let Some(prompt_tokens) = usage.get("promptTokenCount").and_then(|v| v.as_u64()) {
                self.prompt_tokens = prompt_tokens;
            }
            if let Some(completion_tokens) =
                usage.get("candidatesTokenCount").and_then(|v| v.as_u64())
            {
                self.completion_tokens = completion_tokens;
            }
            if let Some(cache_read_input_tokens) = usage
                .get("cachedContentTokenCount")
                .and_then(|v| v.as_u64())
            {
                self.cache_read_input_tokens = Some(cache_read_input_tokens);
            }
        }

        if let Some(parts) = candidate
            .get("content")
            .and_then(|v| v.get("parts"))
            .and_then(|v| v.as_array())
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(|v| v.as_str())
                    && !text.is_empty()
                {
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({ "content": text }),
                        Value::Null,
                        None,
                    ));
                }

                if let Some(function_call) = part.get("functionCall") {
                    let index = self.next_tool_index;
                    self.next_tool_index += 1;
                    self.saw_tool_call = true;
                    self.emit_role_if_needed(output);
                    output.push_str(&openai_sse_chunk(
                        &self.id,
                        self.created,
                        &self.model,
                        json!({
                            "tool_calls": [{
                                "index": index,
                                "id": function_call
                                    .get("id")
                                    .cloned()
                                    .unwrap_or_else(|| json!(format!("call_{index}"))),
                                "type": "function",
                                "function": {
                                    "name": function_call.get("name").cloned().unwrap_or_else(|| json!("")),
                                    "arguments": serde_json::to_string(
                                        &function_call.get("args").cloned().unwrap_or_else(|| json!({}))
                                    ).unwrap_or_else(|_| "{}".to_string())
                                }
                            }]
                        }),
                        Value::Null,
                        None,
                    ));
                }
            }
        }

        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str())
            && !reason.is_empty()
        {
            self.emit_finish(output, map_gemini_finish_reason(reason, self.saw_tool_call));
        }

        Ok(())
    }

    fn emit_role_if_needed(&mut self, output: &mut String) {
        if self.role_sent {
            return;
        }
        self.role_sent = true;
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({ "role": "assistant" }),
            Value::Null,
            None,
        ));
    }

    fn emit_finish(&mut self, output: &mut String, finish_reason: &str) {
        if self.finished {
            return;
        }
        self.emit_role_if_needed(output);
        let usage = Some(openai_usage_json(
            self.prompt_tokens,
            self.completion_tokens,
            self.cache_read_input_tokens,
            None,
        ));
        output.push_str(&openai_sse_chunk(
            &self.id,
            self.created,
            &self.model,
            json!({}),
            json!(finish_reason),
            usage,
        ));
        output.push_str("data: [DONE]\n\n");
        self.finished = true;
    }
}

fn openai_sse_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Value,
    usage: Option<Value>,
) -> String {
    let mut chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason
        }]
    });
    if let Some(usage) = usage {
        chunk["usage"] = usage;
    }
    format!("data: {}\n\n", chunk)
}

fn openai_usage_json(
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) -> Value {
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    usage
}

fn map_anthropic_stop_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "tool_use" => "tool_calls",
        "max_tokens" => "length",
        _ => "stop",
    }
}

fn map_gemini_finish_reason(finish_reason: &str, saw_tool_call: bool) -> &'static str {
    if saw_tool_call {
        return "tool_calls";
    }

    match finish_reason {
        "MAX_TOKENS" => "length",
        "SAFETY" => "content_filter",
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::{AnthropicToOpenAIStreamConverter, GeminiToOpenAIStreamConverter};

    #[test]
    fn test_anthropic_stream_converter_emits_openai_sse() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let input = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":12,\"cache_creation_input_tokens\":30}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"shell\",\"input\":{}}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7,\"cache_read_input_tokens\":90}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hello\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"toolu_1\""));
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("\"cache_read_input_tokens\":90"));
        assert!(output.contains("\"cache_creation_input_tokens\":30"));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_anthropic_converter_text_only_stop() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let input = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":5}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"content\":\"Hi\""));
        assert!(output.contains("\"finish_reason\":\"stop\""));
        assert!(!output.contains("\"tool_calls\""));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_anthropic_converter_max_tokens_reason() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let input = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_3\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":5}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Truncated\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":10}}\n\n",
        );
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"finish_reason\":\"length\""));
    }

    #[test]
    fn test_anthropic_converter_empty_input() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        let output = converter.push_bytes(b"").unwrap();
        assert!(output.is_empty());
        let finish = converter.finish().unwrap();
        // Should still emit a finish even with no content
        assert!(finish.contains("\"finish_reason\":\"stop\""));
        assert!(finish.contains("data: [DONE]"));
    }

    #[test]
    fn test_anthropic_converter_fallback_model() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("my-fallback");
        // No message_start with model → should use fallback
        let input = "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n";
        let output = converter.push_bytes(input.as_bytes()).unwrap();
        assert!(output.contains("\"model\":\"my-fallback\""));
    }

    #[test]
    fn test_anthropic_converter_ignores_invalid_json() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("model");
        let input = "data: {not valid json}\n\n";
        let output = converter.push_bytes(input.as_bytes()).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn test_gemini_converter_text_only() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2}}\n\n";
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"content\":\"Hello world\""));
        assert!(output.contains("\"finish_reason\":\"stop\""));
        assert!(output.contains("\"prompt_tokens\":5"));
        assert!(output.contains("\"completion_tokens\":2"));
    }

    #[test]
    fn test_gemini_converter_max_tokens() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"cut\"}]},\"finishReason\":\"MAX_TOKENS\"}]}\n\n";
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());
        assert!(output.contains("\"finish_reason\":\"length\""));
    }

    #[test]
    fn test_gemini_converter_safety_filter() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = "data: {\"candidates\":[{\"finishReason\":\"SAFETY\"}]}\n\n";
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());
        assert!(output.contains("\"finish_reason\":\"content_filter\""));
    }

    #[test]
    fn test_gemini_converter_empty_input() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let output = converter.push_bytes(b"").unwrap();
        assert!(output.is_empty());
        let finish = converter.finish().unwrap();
        assert!(finish.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn test_gemini_stream_converter_emits_openai_sse() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        let input = concat!(
            "data: {\"responseId\":\"resp_1\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"id\":\"call_1\",\"name\":\"shell\",\"args\":{\"cmd\":\"ls\"}}}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":3,\"cachedContentTokenCount\":90}}\n\n",
        );

        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        assert!(output.contains("\"role\":\"assistant\""));
        assert!(output.contains("\"content\":\"Hel\""));
        assert!(output.contains("\"content\":\"lo\""));
        assert!(output.contains("\"tool_calls\":[{"));
        assert!(output.contains("\"id\":\"call_1\""));
        assert!(output.contains("\"index\":0"));
        assert!(output.contains("\"name\":\"shell\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
        assert!(output.contains("\"cache_read_input_tokens\":90"));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn test_anthropic_parallel_tool_calls_preserve_nonzero_index() {
        let mut converter = AnthropicToOpenAIStreamConverter::new("claude-sonnet-4-5");
        // Text at index 0, then two tool_use blocks at index 1 and 2
        let input = concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-5\",\"usage\":{\"input_tokens\":10}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Running two tools\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"read_file\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"a.txt\\\"}\"}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_2\",\"name\":\"read_file\",\"input\":{}}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\\\"b.txt\\\"}\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":15}}\n\n",
        );
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        // Verify tool_calls at index 1 and 2 (not 0)
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"id\":\"toolu_1\""));
        assert!(output.contains("\"index\":2"));
        assert!(output.contains("\"id\":\"toolu_2\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
    }

    #[test]
    fn test_gemini_multiple_function_calls_sequential_index() {
        let mut converter = GeminiToOpenAIStreamConverter::new("gemini-2.5-pro");
        // Single candidate with text + two functionCall parts
        let input = concat!(
            "data: {\"responseId\":\"resp_1\",\"candidates\":[{\"content\":{\"parts\":[",
            "{\"text\":\"Running tools\"},",
            "{\"functionCall\":{\"id\":\"call_1\",\"name\":\"read_file\",\"args\":{\"path\":\"a.txt\"}}},",
            "{\"functionCall\":{\"id\":\"call_2\",\"name\":\"write_file\",\"args\":{\"path\":\"b.txt\"}}}",
            "],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":10}}\n\n",
        );
        let mut output = converter.push_bytes(input.as_bytes()).unwrap();
        output.push_str(&converter.finish().unwrap());

        // Verify sequential tool indices 0 and 1
        assert!(output.contains("\"index\":0"));
        assert!(output.contains("\"id\":\"call_1\""));
        assert!(output.contains("\"name\":\"read_file\""));
        assert!(output.contains("\"index\":1"));
        assert!(output.contains("\"id\":\"call_2\""));
        assert!(output.contains("\"name\":\"write_file\""));
        assert!(output.contains("\"finish_reason\":\"tool_calls\""));
    }
}
