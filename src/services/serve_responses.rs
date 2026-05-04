use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::services::http_utils::{current_unix_ts, sse_data_payload};
use crate::services::responses_to_chat_router::convert_chat_response_to_responses_sse;

pub(crate) struct OpenAIToResponsesStreamConverter {
    pending: String,
    response_id: String,
    created_at: u64,
    model: String,
    started: bool,
    completed: bool,
    text_item: Option<ResponsesTextItemState>,
    tool_calls: HashMap<usize, ResponsesToolCallState>,
    next_output_index: usize,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    saw_usage: bool,
}

struct ResponsesTextItemState {
    item_id: String,
    output_index: usize,
    content: String,
}

struct ResponsesToolCallState {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    output_index: usize,
    started: bool,
}

enum ResponsesOutputItem {
    Message {
        item_id: String,
        output_index: usize,
        content: String,
    },
    FunctionCall {
        item_id: String,
        call_id: String,
        name: String,
        arguments: String,
        output_index: usize,
    },
}

static RESPONSES_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl OpenAIToResponsesStreamConverter {
    pub(crate) fn new(original_model: &str) -> Self {
        Self {
            pending: String::new(),
            response_id: next_responses_id("resp"),
            created_at: current_unix_ts(),
            model: original_model.to_string(),
            started: false,
            completed: false,
            text_item: None,
            tool_calls: HashMap::new(),
            next_output_index: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            saw_usage: false,
        }
    }

    pub(crate) fn push_bytes(&mut self, chunk: &[u8]) -> Result<String> {
        self.pending.push_str(&String::from_utf8_lossy(chunk));
        let mut output = String::new();

        while let Some(pos) = self.pending.find('\n') {
            let line = self.pending[..pos].trim_end_matches('\r').to_string();
            self.pending = self.pending[pos + 1..].to_string();
            self.process_line(&line, &mut output)?;
        }

        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        let mut output = String::new();

        let tail = self.pending.trim_end_matches('\r').trim().to_string();
        self.pending.clear();
        if !tail.is_empty() {
            self.process_line(&tail, &mut output)?;
        }

        if !self.completed {
            self.finalize(&mut output);
        }

        Ok(output)
    }

    fn process_line(&mut self, line: &str, output: &mut String) -> Result<()> {
        let Some(data) = sse_data_payload(line) else {
            return Ok(());
        };

        if data == "[DONE]" {
            if !self.completed {
                self.finalize(output);
            }
            return Ok(());
        }

        let chunk: Value = match serde_json::from_str(data) {
            Ok(value) => value,
            Err(_) => return Ok(()),
        };

        if let Some(usage) = chunk.get("usage") {
            self.saw_usage = true;
            if let Some(input_tokens) = usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
                .and_then(|v| v.as_u64())
            {
                self.input_tokens = input_tokens;
            }
            if let Some(output_tokens) = usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
                .and_then(|v| v.as_u64())
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

        if let Some(model) = chunk.get("model").and_then(|v| v.as_str())
            && !model.is_empty()
            && self.model.is_empty()
        {
            self.model = model.to_string();
        }

        let choice = chunk
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .cloned()
            .unwrap_or_else(|| json!({}));
        let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));

        if !delta.is_null() {
            self.ensure_started(output);
        }

        if let Some(text) = delta.get("content").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            self.ensure_text_item(output);
            if let Some(text_item) = self.text_item.as_mut() {
                text_item.content.push_str(text);
                output.push_str(&responses_sse_event(
                    "response.output_text.delta",
                    json!({
                        "type": "response.output_text.delta",
                        "response_id": self.response_id,
                        "item_id": text_item.item_id,
                        "output_index": text_item.output_index,
                        "content_index": 0,
                        "delta": text
                    }),
                ));
            }
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tool_call in tool_calls {
                let index = tool_call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                if !self.tool_calls.contains_key(&index) {
                    let output_index = self.take_output_index();
                    self.tool_calls.insert(
                        index,
                        ResponsesToolCallState {
                            item_id: next_responses_id("fc"),
                            call_id: tool_call
                                .get("id")
                                .and_then(|v| v.as_str())
                                .filter(|v| !v.is_empty())
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("call_{index}")),
                            name: String::new(),
                            arguments: String::new(),
                            output_index,
                            started: false,
                        },
                    );
                }
                let Some(state) = self.tool_calls.get_mut(&index) else {
                    continue;
                };

                if let Some(call_id) = tool_call.get("id").and_then(|v| v.as_str())
                    && !call_id.is_empty()
                {
                    state.call_id = call_id.to_string();
                }
                if let Some(name) = tool_call
                    .get("function")
                    .and_then(|v| v.get("name"))
                    .and_then(|v| v.as_str())
                    && !name.is_empty()
                {
                    state.name = name.to_string();
                }

                if !state.started {
                    output.push_str(&responses_sse_event(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "response_id": self.response_id,
                            "output_index": state.output_index,
                            "item": {
                                "id": state.item_id,
                                "call_id": state.call_id,
                                "type": "function_call",
                                "status": "in_progress",
                                "name": state.name,
                                "arguments": state.arguments
                            }
                        }),
                    ));
                    state.started = true;
                }

                if let Some(arguments) = tool_call
                    .get("function")
                    .and_then(|v| v.get("arguments"))
                    .and_then(|v| v.as_str())
                    && !arguments.is_empty()
                {
                    state.arguments.push_str(arguments);
                    output.push_str(&responses_sse_event(
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "response_id": self.response_id,
                            "output_index": state.output_index,
                            "item_id": state.item_id,
                            "delta": arguments
                        }),
                    ));
                }
            }
        }

        // OpenAI-compatible providers can send a usage-only chunk after
        // finish_reason. Finalize on [DONE]/EOF so those tokens are kept.
        Ok(())
    }

    fn ensure_started(&mut self, output: &mut String) {
        if self.started {
            return;
        }
        self.started = true;
        output.push_str(&responses_sse_event(
            "response.created",
            json!({
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "model": self.model,
                    "created_at": self.created_at,
                    "status": "in_progress",
                    "output": []
                }
            }),
        ));
    }

    fn ensure_text_item(&mut self, output: &mut String) {
        if self.text_item.is_some() {
            return;
        }

        let output_index = self.take_output_index();
        let item_id = next_responses_id("msg");
        output.push_str(&responses_sse_event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": []
                }
            }),
        ));
        output.push_str(&responses_sse_event(
            "response.content_part.added",
            json!({
                "type": "response.content_part.added",
                "response_id": self.response_id,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": ""}
            }),
        ));
        self.text_item = Some(ResponsesTextItemState {
            item_id,
            output_index,
            content: String::new(),
        });
    }

    fn finalize(&mut self, output: &mut String) {
        if self.completed {
            return;
        }

        self.ensure_started(output);

        if self.text_item.is_none() && self.tool_calls.is_empty() {
            self.ensure_text_item(output);
        }

        if let Some(text_item) = self.text_item.as_ref() {
            output.push_str(&responses_sse_event(
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "response_id": self.response_id,
                    "item_id": text_item.item_id,
                    "output_index": text_item.output_index,
                    "content_index": 0,
                    "text": text_item.content
                }),
            ));
            output.push_str(&responses_sse_event(
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": text_item.item_id,
                    "output_index": text_item.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_item.content}
                }),
            ));
            output.push_str(&responses_sse_event(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "response_id": self.response_id,
                    "output_index": text_item.output_index,
                    "item": {
                        "id": text_item.item_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": text_item.content,
                            "annotations": []
                        }]
                    }
                }),
            ));
        }

        let mut tool_indexes: Vec<usize> = self.tool_calls.keys().copied().collect();
        tool_indexes.sort_unstable();
        for index in tool_indexes {
            if let Some(tool_call) = self.tool_calls.get(&index) {
                output.push_str(&responses_sse_event(
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "response_id": self.response_id,
                        "output_index": tool_call.output_index,
                        "item_id": tool_call.item_id,
                        "arguments": tool_call.arguments
                    }),
                ));
                output.push_str(&responses_sse_event(
                    "response.output_item.done",
                    json!({
                        "type": "response.output_item.done",
                        "response_id": self.response_id,
                        "output_index": tool_call.output_index,
                        "item": {
                            "id": tool_call.item_id,
                            "call_id": tool_call.call_id,
                            "type": "function_call",
                            "status": "completed",
                            "name": tool_call.name,
                            "arguments": tool_call.arguments
                        }
                    }),
                ));
            }
        }

        let output_items = self.output_items();
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "model": self.model,
            "created_at": self.created_at,
            "status": "completed",
            "output": output_items
        });
        if self.saw_usage {
            response["usage"] = responses_usage_json(
                self.input_tokens,
                self.output_tokens,
                self.cache_read_input_tokens,
                self.cache_creation_input_tokens,
            );
        }
        output.push_str(&responses_sse_event(
            "response.completed",
            json!({
                "type": "response.completed",
                "response": response
            }),
        ));

        self.completed = true;
    }

    fn take_output_index(&mut self) -> usize {
        let index = self.next_output_index;
        self.next_output_index += 1;
        index
    }

    fn output_items(&self) -> Vec<Value> {
        let mut items = Vec::new();

        if let Some(text_item) = self.text_item.as_ref() {
            items.push(ResponsesOutputItem::Message {
                item_id: text_item.item_id.clone(),
                output_index: text_item.output_index,
                content: text_item.content.clone(),
            });
        }

        let mut tool_items: Vec<ResponsesOutputItem> = self
            .tool_calls
            .values()
            .map(|tool_call| ResponsesOutputItem::FunctionCall {
                item_id: tool_call.item_id.clone(),
                call_id: tool_call.call_id.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
                output_index: tool_call.output_index,
            })
            .collect();
        items.append(&mut tool_items);
        items.sort_by_key(|item| match item {
            ResponsesOutputItem::Message { output_index, .. } => *output_index,
            ResponsesOutputItem::FunctionCall { output_index, .. } => *output_index,
        });

        items
            .into_iter()
            .map(|item| match item {
                ResponsesOutputItem::Message {
                    item_id, content, ..
                } => json!({
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": content,
                        "annotations": []
                    }]
                }),
                ResponsesOutputItem::FunctionCall {
                    item_id,
                    call_id,
                    name,
                    arguments,
                    ..
                } => json!({
                    "id": item_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "status": "completed",
                    "name": name,
                    "arguments": arguments
                }),
            })
            .collect()
    }
}

pub(crate) fn convert_chat_response_to_responses_json(
    chat: &Value,
    original_model: &str,
) -> Result<Value> {
    let sse = convert_chat_response_to_responses_sse(chat, false, original_model);
    extract_completed_response_from_sse(&sse)
        .ok_or_else(|| anyhow::anyhow!("failed to synthesize responses JSON payload"))
}

pub(crate) fn convert_chat_sse_to_responses_sse(
    chat_sse: &str,
    original_model: &str,
) -> Result<String> {
    let mut converter = OpenAIToResponsesStreamConverter::new(original_model);
    let mut output = converter.push_bytes(chat_sse.as_bytes())?;
    output.push_str(&converter.finish()?);
    Ok(output)
}

fn responses_sse_event(event: &str, data: Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

fn next_responses_id(prefix: &str) -> String {
    let count = RESPONSES_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{}_{count}", current_unix_ts())
}

fn responses_usage_json(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
) -> Value {
    let mut usage = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens
    });
    if let Some(value) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(value);
    }
    if let Some(value) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(value);
    }
    usage
}

fn extract_completed_response_from_sse(sse: &str) -> Option<Value> {
    let mut saw_completed_event = false;

    for line in sse.lines() {
        if let Some(event) = line.strip_prefix("event:") {
            saw_completed_event = event.trim() == "response.completed";
            continue;
        }

        if saw_completed_event && let Some(data) = sse_data_payload(line) {
            let payload: Value = serde_json::from_str(data).ok()?;
            return payload.get("response").cloned();
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        convert_chat_response_to_responses_json, convert_chat_sse_to_responses_sse,
        extract_completed_response_from_sse,
    };
    use serde_json::json;

    #[test]
    fn test_convert_chat_response_to_responses_json_text() {
        let chat = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_read_input_tokens": 90
            },
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from responses"}
            }]
        });

        let response = convert_chat_response_to_responses_json(&chat, "gpt-4o").unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["model"], "gpt-4o");
        assert_eq!(response["status"], "completed");
        assert_eq!(response["usage"]["input_tokens"], 10);
        assert_eq!(response["usage"]["output_tokens"], 5);
        assert_eq!(response["usage"]["cache_read_input_tokens"], 90);
        assert_eq!(response["output"][0]["type"], "message");
        assert_eq!(
            response["output"][0]["content"][0]["text"],
            "Hello from responses"
        );
    }

    #[test]
    fn test_convert_chat_response_to_responses_json_tool_call() {
        let chat = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let response = convert_chat_response_to_responses_json(&chat, "gpt-4o").unwrap();

        assert_eq!(response["object"], "response");
        assert_eq!(response["output"][0]["type"], "function_call");
        assert_eq!(response["output"][0]["call_id"], "call_123");
        assert_eq!(response["output"][0]["name"], "shell");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_text() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_1\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"cache_read_input_tokens\":90}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "gpt-4o").unwrap();

        assert!(responses_sse.contains("event: response.created"));
        assert!(responses_sse.contains("event: response.output_text.delta"));
        assert!(responses_sse.contains("\"delta\":\"Hel\""));
        assert!(responses_sse.contains("\"delta\":\"lo\""));
        assert!(responses_sse.contains("\"cache_read_input_tokens\":90"));
        assert!(responses_sse.contains("event: response.completed"));
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_accepts_input_output_usage_aliases() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "grok-4.3").unwrap();
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();

        assert_eq!(response["usage"]["input_tokens"], 15000);
        assert_eq!(response["usage"]["output_tokens"], 42);
        assert_eq!(response["usage"]["total_tokens"], 15042);
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_waits_for_trailing_usage_after_finish_reason() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"chatcmpl_xai\",\"model\":\"grok-4.3\",\"choices\":[],\"usage\":{\"input_tokens\":15000,\"output_tokens\":42}}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "grok-4.3").unwrap();
        let response = extract_completed_response_from_sse(&responses_sse).unwrap();

        assert_eq!(response["usage"]["input_tokens"], 15000);
        assert_eq!(response["usage"]["output_tokens"], 42);
        assert_eq!(response["output"][0]["content"][0]["text"], "hi!");
    }

    #[test]
    fn test_convert_chat_sse_to_responses_sse_tool_call() {
        let chat_sse = concat!(
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_abc\",\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chatcmpl_2\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        let responses_sse = convert_chat_sse_to_responses_sse(chat_sse, "gpt-4o").unwrap();

        assert!(responses_sse.contains("event: response.output_item.added"));
        assert!(responses_sse.contains("event: response.function_call_arguments.delta"));
        assert!(responses_sse.contains("\"call_id\":\"call_abc\""));
        assert!(responses_sse.contains("\"delta\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(responses_sse.contains("event: response.completed"));
    }
}
