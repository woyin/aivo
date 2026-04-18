use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatRequest {
    #[serde(default = "default_openai_model")]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OpenAIChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAIChatTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIChatToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatMessage {
    pub role: String,
    // Serialize even when None so that assistant tool_call messages emit
    // `"content": null`. Strict OpenAI-compatible providers (e.g. Cloudflare
    // Workers AI) require the field to be present.
    #[serde(default)]
    pub content: Option<OpenAIMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIChatToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

impl OpenAIMessageContent {
    fn flatten_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(OpenAIContentPart::text)
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum OpenAIContentPart {
    Text(String),
    Object(OpenAIContentPartObject),
}

impl OpenAIContentPart {
    fn text(&self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text.clone()),
            Self::Object(part) => part.text.clone(),
        }
    }

    fn is_text_only(&self) -> bool {
        match self {
            Self::Text(_) => true,
            Self::Object(part) => {
                part.image_url.is_none() && (part.extra.is_empty() || part.has_only_text_metadata())
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIContentPartObject {
    // Preserve the OpenAI `type` discriminator across deserialize/serialize.
    // Strict providers reject parts without it; flattening to text drops the
    // field, which is fine because the Text variant carries no discriminator.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<OpenAIImageUrl>,
    // Catch-all for content-part kinds we don't have typed fields for
    // (e.g. `input_audio`, `file`). Without this, the typed round-trip in
    // `stringify_message_content` would silently drop their payloads.
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl OpenAIContentPartObject {
    fn has_only_text_metadata(&self) -> bool {
        self.is_text_like_kind() && self.extra.keys().all(|key| is_text_part_metadata_key(key))
    }

    fn is_text_like_kind(&self) -> bool {
        matches!(
            self.kind.as_deref(),
            None | Some("text") | Some("input_text") | Some("output_text")
        )
    }
}

fn is_text_part_metadata_key(key: &str) -> bool {
    matches!(key, "annotations" | "cache_control")
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIImageUrl {
    pub url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAIChatFunctionTool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatFunctionTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum OpenAIChatToolChoice {
    Mode(String),
    Named(OpenAIChatNamedToolChoice),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatNamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAIChatToolChoiceFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatToolChoiceFunction {
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAIChatToolCallFunction,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesRequest {
    pub model: String,
    pub input: Vec<ResponsesRequestInputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIChatToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub(crate) enum ResponsesRequestInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponsesMessageContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: Value },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum ResponsesMessageContent {
    Text(String),
    Parts(Vec<ResponsesMessagePart>),
}

impl ResponsesMessageContent {
    fn text_parts(&self) -> Vec<String> {
        match self {
            Self::Text(text) => vec![text.clone()],
            Self::Parts(parts) => parts.iter().filter_map(|part| part.text.clone()).collect(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesMessagePart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, flatten)]
    pub extra: Map<String, Value>,
}

impl ResponsesMessagePart {
    fn is_effectively_empty(&self) -> bool {
        self.text.as_deref().unwrap_or_default().is_empty() && self.extra.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesReasoning {
    pub effort: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesResponse {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub output: Vec<Value>,
    #[serde(default)]
    pub usage: ResponsesUsage,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message { content: ResponsesMessageContent },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        call_id: Option<String>,
        name: String,
        #[serde(default)]
        arguments: Option<String>,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<OpenAIChatChoice>,
    #[serde(default)]
    pub usage: OpenAIChatUsage,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatResponseView {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<OpenAIChatChoiceView>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChoiceView {
    #[serde(default)]
    pub message: OpenAIChatResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<OpenAIChatChunkUsage>,
    #[serde(default)]
    pub choices: Vec<OpenAIChatChunkChoice>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunkUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunkChoice {
    #[serde(default)]
    pub delta: OpenAIChatChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunkDelta {
    #[serde(default)]
    pub content: Option<String>,
    /// DeepSeek-reasoner thinking content (streamed before `content`)
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub function_call: Option<OpenAIChatChunkFunctionCall>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAIChatChunkToolCall>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunkFunctionCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChunkToolCall {
    #[serde(default)]
    pub index: Option<u64>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAIChatChunkFunctionCall>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatChoice {
    #[serde(default)]
    pub index: u32,
    pub message: OpenAIChatResponseMessage,
    #[serde(default)]
    pub finish_reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatResponseMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek-reasoner / Anthropic thinking content
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIChatToolCall>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub(crate) struct OpenAIChatUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
}

pub(crate) fn stringify_message_content(request: &mut OpenAIChatRequest) {
    for message in &mut request.messages {
        let Some(content) = &message.content else {
            continue;
        };
        if matches!(content, OpenAIMessageContent::Text(_)) {
            continue;
        }
        // Messages carrying images must stay as multimodal arrays —
        // flattening would silently erase the images. Strict providers
        // that reject array content will fail loudly on such requests,
        // which is the correct outcome over losing data.
        if !content_is_pure_text(content) {
            continue;
        }
        message.content = Some(OpenAIMessageContent::Text(content.flatten_text()));
    }
}

fn content_is_pure_text(content: &OpenAIMessageContent) -> bool {
    match content {
        OpenAIMessageContent::Text(_) => true,
        OpenAIMessageContent::Parts(parts) => parts.iter().all(OpenAIContentPart::is_text_only),
    }
}

fn responses_text_kind_for_role(role: &str) -> &'static str {
    if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    }
}

fn convert_openai_content_to_responses(
    content: &OpenAIMessageContent,
    role: &str,
) -> ResponsesMessageContent {
    match content {
        OpenAIMessageContent::Text(text) => {
            if role == "assistant" {
                ResponsesMessageContent::Parts(vec![ResponsesMessagePart {
                    kind: "output_text".to_string(),
                    text: Some(text.clone()),
                    extra: Map::new(),
                }])
            } else {
                ResponsesMessageContent::Text(text.clone())
            }
        }
        OpenAIMessageContent::Parts(parts) => ResponsesMessageContent::Parts(
            parts
                .iter()
                .map(|part| convert_openai_part_to_responses(part, role))
                .collect(),
        ),
    }
}

fn convert_openai_part_to_responses(part: &OpenAIContentPart, role: &str) -> ResponsesMessagePart {
    let text_kind = responses_text_kind_for_role(role).to_string();
    match part {
        OpenAIContentPart::Text(text) => ResponsesMessagePart {
            kind: text_kind,
            text: Some(text.clone()),
            extra: Map::new(),
        },
        OpenAIContentPart::Object(obj) => {
            let mut extra = obj.extra.clone();

            if let Some(image_url) = &obj.image_url {
                extra.insert("image_url".to_string(), json!(image_url.url));
                return ResponsesMessagePart {
                    kind: "input_image".to_string(),
                    text: obj.text.clone(),
                    extra,
                };
            }

            if let Some(file) = extra.remove("file") {
                if let Some(file_obj) = file.as_object() {
                    if let Some(filename) = file_obj.get("filename") {
                        extra.insert("filename".to_string(), filename.clone());
                    }
                    if let Some(file_data) = file_obj.get("file_data") {
                        extra.insert("file_data".to_string(), file_data.clone());
                    }
                } else {
                    extra.insert("file".to_string(), file);
                }
                return ResponsesMessagePart {
                    kind: "input_file".to_string(),
                    text: obj.text.clone(),
                    extra,
                };
            }

            let kind = match obj.kind.as_deref() {
                Some("text") | Some("input_text") | Some("output_text") | None => text_kind,
                Some(kind) => kind.to_string(),
            };

            ResponsesMessagePart {
                kind,
                text: obj.text.clone(),
                extra,
            }
        }
    }
}

fn convert_tool_output(content: Option<&OpenAIMessageContent>) -> Value {
    let Some(content) = content else {
        return json!("");
    };
    if content_is_pure_text(content) {
        return json!(content.flatten_text());
    }
    // `content_is_pure_text` returns true unconditionally for `Text`, so we
    // can only land here with `Parts`.
    let OpenAIMessageContent::Parts(parts) = content else {
        return json!(content.flatten_text());
    };
    serde_json::to_value(parts).unwrap_or_else(|_| json!(content.flatten_text()))
}

fn responses_content_is_empty(content: &ResponsesMessageContent) -> bool {
    match content {
        ResponsesMessageContent::Text(text) => text.is_empty(),
        ResponsesMessageContent::Parts(parts) => {
            parts.iter().all(ResponsesMessagePart::is_effectively_empty)
        }
    }
}

fn default_openai_model() -> String {
    "gpt-4o".to_string()
}

pub(crate) fn convert_chat_to_responses_request(
    openai_req: &OpenAIChatRequest,
) -> ResponsesRequest {
    let mut input = Vec::new();
    let mut instructions: Option<String> = None;
    let mut fc_counter = 0u64;

    for message in &openai_req.messages {
        match message.role.as_str() {
            "system" | "developer" => {
                let text = flatten_openai_message_text(message);
                if !text.is_empty() {
                    match &mut instructions {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(&text);
                        }
                        None => instructions = Some(text),
                    }
                }
            }
            "assistant" => {
                let content = message
                    .content
                    .as_ref()
                    .map(|content| convert_openai_content_to_responses(content, "assistant"));
                if let Some(tool_calls) = &message.tool_calls {
                    if let Some(content) = content
                        .as_ref()
                        .filter(|content| !responses_content_is_empty(content))
                    {
                        input.push(ResponsesRequestInputItem::Message {
                            role: "assistant".to_string(),
                            content: content.clone(),
                        });
                    }
                    for tool_call in tool_calls {
                        fc_counter += 1;
                        input.push(ResponsesRequestInputItem::FunctionCall {
                            id: format!("fc_{fc_counter}"),
                            call_id: tool_call.id.clone(),
                            name: tool_call.function.name.clone(),
                            arguments: tool_call.function.arguments.clone(),
                        });
                    }
                } else {
                    input.push(ResponsesRequestInputItem::Message {
                        role: "assistant".to_string(),
                        content: content.unwrap_or_else(|| ResponsesMessageContent::Parts(vec![])),
                    });
                }
            }
            "tool" => {
                input.push(ResponsesRequestInputItem::FunctionCallOutput {
                    call_id: message.tool_call_id.clone().unwrap_or_default(),
                    output: convert_tool_output(message.content.as_ref()),
                });
            }
            role => {
                input.push(ResponsesRequestInputItem::Message {
                    role: role.to_string(),
                    content: message
                        .content
                        .as_ref()
                        .map(|content| convert_openai_content_to_responses(content, role))
                        .unwrap_or_else(|| ResponsesMessageContent::Text(String::new())),
                });
            }
        }
    }

    ResponsesRequest {
        model: openai_req.model.clone(),
        input,
        instructions,
        max_output_tokens: openai_req.max_tokens.clone(),
        temperature: openai_req.temperature.clone(),
        top_p: openai_req.top_p.clone(),
        tools: openai_req.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| ResponsesTool {
                    kind: tool.kind.clone(),
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                })
                .collect()
        }),
        tool_choice: openai_req.tool_choice.clone(),
        reasoning: openai_req
            .reasoning_effort
            .as_ref()
            .map(|effort| ResponsesReasoning {
                effort: if effort.eq_ignore_ascii_case("xhigh") {
                    "high".to_string()
                } else {
                    effort.clone()
                },
            }),
        stream: openai_req.stream,
    }
}

pub(crate) fn convert_responses_to_chat_response(resp: &ResponsesResponse) -> OpenAIChatResponse {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for item in &resp.output {
        let Ok(item) = serde_json::from_value::<ResponsesOutputItem>(item.clone()) else {
            continue;
        };

        match item {
            ResponsesOutputItem::Message { content } => {
                text_parts.extend(
                    content
                        .text_parts()
                        .into_iter()
                        .filter(|text| !text.is_empty()),
                );
            }
            ResponsesOutputItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } => {
                tool_calls.push(OpenAIChatToolCall {
                    id: call_id.or(id).unwrap_or_default(),
                    kind: "function".to_string(),
                    function: OpenAIChatToolCallFunction {
                        name,
                        arguments: arguments.unwrap_or_else(|| "{}".to_string()),
                    },
                });
            }
        }
    }

    let prompt_tokens = resp.usage.input_tokens.unwrap_or(0);
    let completion_tokens = resp.usage.output_tokens.unwrap_or(0);
    let cache_read_input_tokens = resp.usage.cache_read_input_tokens;
    let cache_creation_input_tokens = resp.usage.cache_creation_input_tokens;
    let content = (!text_parts.is_empty()).then(|| text_parts.join("\n"));
    let tool_calls = (!tool_calls.is_empty()).then_some(tool_calls);
    let finish_reason = if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };

    OpenAIChatResponse {
        id: resp
            .id
            .clone()
            .unwrap_or_else(|| "chatcmpl-aivo".to_string()),
        object: "chat.completion".to_string(),
        created: None,
        model: resp.model.clone().unwrap_or_else(|| "unknown".to_string()),
        choices: vec![OpenAIChatChoice {
            index: 0,
            message: OpenAIChatResponseMessage {
                role: "assistant".to_string(),
                content,
                reasoning_content: None,
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: OpenAIChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
        },
    }
}

fn flatten_openai_message_text(message: &OpenAIChatMessage) -> String {
    message
        .content
        .as_ref()
        .map(OpenAIMessageContent::flatten_text)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_stringify_message_content_flattens_arrays() {
        let mut req = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIMessageContent::Parts(vec![
                        OpenAIContentPart::Object(OpenAIContentPartObject {
                            text: Some("hello".to_string()),
                            ..Default::default()
                        }),
                        OpenAIContentPart::Object(OpenAIContentPartObject {
                            text: Some("world".to_string()),
                            ..Default::default()
                        }),
                    ])),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content: Some(OpenAIMessageContent::Text("already a string".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIMessageContent::Parts(vec![
                        OpenAIContentPart::Text("plain".to_string()),
                        OpenAIContentPart::Text("strings".to_string()),
                    ])),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIMessageContent::Parts(vec![
                        OpenAIContentPart::Object(OpenAIContentPartObject {
                            text: None,
                            ..Default::default()
                        }),
                    ])),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        stringify_message_content(&mut req);

        assert_eq!(
            req.messages[0].content,
            Some(OpenAIMessageContent::Text("hello\nworld".to_string()))
        );
        assert_eq!(
            req.messages[1].content,
            Some(OpenAIMessageContent::Text("already a string".to_string()))
        );
        assert_eq!(
            req.messages[2].content,
            Some(OpenAIMessageContent::Text("plain\nstrings".to_string()))
        );
        assert_eq!(
            req.messages[3].content,
            Some(OpenAIMessageContent::Text(String::new()))
        );
    }

    #[test]
    fn stringify_preserves_messages_containing_image_url_parts() {
        let mut req = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "tool".to_string(),
                    content: Some(OpenAIMessageContent::Parts(vec![
                        OpenAIContentPart::Object(OpenAIContentPartObject {
                            kind: Some("text".to_string()),
                            text: Some("Screenshot taken".to_string()),
                            image_url: None,
                            ..Default::default()
                        }),
                        OpenAIContentPart::Object(OpenAIContentPartObject {
                            kind: Some("image_url".to_string()),
                            text: None,
                            image_url: Some(OpenAIImageUrl {
                                url: "data:image/png;base64,abc".to_string(),
                            }),
                            ..Default::default()
                        }),
                    ])),
                    tool_calls: None,
                    tool_call_id: Some("toolu_1".to_string()),
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIMessageContent::Text("hi".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        stringify_message_content(&mut req);

        match &req.messages[0].content {
            Some(OpenAIMessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                match &parts[1] {
                    OpenAIContentPart::Object(obj) => {
                        assert_eq!(
                            obj.image_url.as_ref().map(|i| i.url.as_str()),
                            Some("data:image/png;base64,abc")
                        );
                    }
                    _ => panic!("expected Object variant with image_url"),
                }
            }
            other => panic!("expected Parts, got {other:?}"),
        }
        assert_eq!(
            req.messages[1].content,
            Some(OpenAIMessageContent::Text("hi".to_string()))
        );
    }

    #[test]
    fn content_part_object_preserves_image_url_through_serde_roundtrip() {
        let raw = json!({
            "type": "image_url",
            "image_url": {"url": "https://example.com/x.png"},
        });
        let part: OpenAIContentPart = serde_json::from_value(raw.clone()).unwrap();
        // Round-trip back to JSON and confirm fields survive.
        let back = serde_json::to_value(&part).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn content_part_object_preserves_unknown_kinds_through_serde_roundtrip() {
        // Unknown content-part kinds (e.g. `input_audio`, `file`) must survive
        // the typed round-trip; the `extra` flatten field is what keeps their
        // payloads from being silently dropped.
        let raw = json!({
            "type": "input_audio",
            "input_audio": {"data": "ZmFrZQ==", "format": "wav"},
        });
        let part: OpenAIContentPart = serde_json::from_value(raw.clone()).unwrap();
        let back = serde_json::to_value(&part).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn stringify_skips_messages_with_unknown_content_kinds() {
        // A message carrying a non-text, non-image kind (audio, file, etc.)
        // must not be flattened to a string — that would lose the payload
        // even though the typed struct no longer does.
        let mut req = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAIChatMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Parts(vec![
                    OpenAIContentPart::Object(
                        serde_json::from_value(json!({
                            "type": "input_audio",
                            "input_audio": {"data": "ZmFrZQ==", "format": "wav"},
                        }))
                        .unwrap(),
                    ),
                ])),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        stringify_message_content(&mut req);

        match &req.messages[0].content {
            Some(OpenAIMessageContent::Parts(parts)) => {
                let serialized = serde_json::to_value(&parts[0]).unwrap();
                assert_eq!(serialized["type"], "input_audio");
                assert_eq!(serialized["input_audio"]["format"], "wav");
            }
            other => panic!("expected Parts preserved, got {other:?}"),
        }
    }

    #[test]
    fn stringify_flattens_text_parts_with_metadata_only_extras() {
        let mut req = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAIChatMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Parts(vec![
                    OpenAIContentPart::Object(
                        serde_json::from_value(json!({
                            "type": "text",
                            "text": "hello",
                            "cache_control": {"type": "ephemeral"},
                        }))
                        .unwrap(),
                    ),
                ])),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        stringify_message_content(&mut req);

        assert_eq!(
            req.messages[0].content,
            Some(OpenAIMessageContent::Text("hello".to_string()))
        );
    }

    #[test]
    fn test_convert_chat_to_responses_request_maps_messages_tools_and_reasoning() {
        let request = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![
                OpenAIChatMessage {
                    role: "system".to_string(),
                    content: Some(OpenAIMessageContent::Text("Be precise.".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "user".to_string(),
                    content: Some(OpenAIMessageContent::Text("List files".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "assistant".to_string(),
                    content: Some(OpenAIMessageContent::Text("Working on it".to_string())),
                    tool_calls: Some(vec![OpenAIChatToolCall {
                        id: "call_1".to_string(),
                        kind: "function".to_string(),
                        function: OpenAIChatToolCallFunction {
                            name: "list_files".to_string(),
                            arguments: "{\"path\":\".\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: None,
                },
                OpenAIChatMessage {
                    role: "tool".to_string(),
                    content: Some(OpenAIMessageContent::Text("file.txt".to_string())),
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                    reasoning_content: None,
                },
            ],
            stream: true,
            max_tokens: Some(json!(128)),
            temperature: Some(json!(0.2)),
            top_p: Some(json!(0.9)),
            tools: Some(vec![OpenAIChatTool {
                kind: "function".to_string(),
                function: OpenAIChatFunctionTool {
                    name: "list_files".to_string(),
                    description: "List files".to_string(),
                    parameters: json!({"type": "object"}),
                },
            }]),
            tool_choice: Some(OpenAIChatToolChoice::Mode("auto".to_string())),
            reasoning_effort: Some("xhigh".to_string()),
            extra: Map::new(),
        };

        let responses = convert_chat_to_responses_request(&request);

        assert_eq!(responses.instructions.as_deref(), Some("Be precise."));
        assert_eq!(responses.max_output_tokens, Some(json!(128)));
        assert_eq!(responses.temperature, Some(json!(0.2)));
        assert_eq!(responses.top_p, Some(json!(0.9)));
        assert_eq!(
            responses.reasoning,
            Some(ResponsesReasoning {
                effort: "high".to_string()
            })
        );
        assert_eq!(responses.tools.as_ref().unwrap()[0].name, "list_files");
        assert_eq!(responses.input.len(), 4);
        assert_eq!(
            responses.input[0],
            ResponsesRequestInputItem::Message {
                role: "user".to_string(),
                content: ResponsesMessageContent::Text("List files".to_string()),
            }
        );
        assert_eq!(
            responses.input[1],
            ResponsesRequestInputItem::Message {
                role: "assistant".to_string(),
                content: ResponsesMessageContent::Parts(vec![ResponsesMessagePart {
                    kind: "output_text".to_string(),
                    text: Some("Working on it".to_string()),
                    extra: Map::new(),
                }]),
            }
        );
        assert_eq!(
            responses.input[3],
            ResponsesRequestInputItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: json!("file.txt"),
            }
        );
    }

    #[test]
    fn convert_chat_to_responses_request_preserves_non_text_tool_output() {
        let request = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAIChatMessage {
                role: "tool".to_string(),
                content: Some(OpenAIMessageContent::Parts(vec![
                    OpenAIContentPart::Object(OpenAIContentPartObject {
                        kind: Some("text".to_string()),
                        text: Some("Screenshot taken".to_string()),
                        ..Default::default()
                    }),
                    OpenAIContentPart::Object(OpenAIContentPartObject {
                        kind: Some("image_url".to_string()),
                        image_url: Some(OpenAIImageUrl {
                            url: "data:image/png;base64,abc".to_string(),
                        }),
                        ..Default::default()
                    }),
                ])),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
                reasoning_content: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        let responses = convert_chat_to_responses_request(&request);

        assert_eq!(
            responses.input[0],
            ResponsesRequestInputItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: json!([
                    {"type": "text", "text": "Screenshot taken"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]),
            }
        );
    }

    #[test]
    fn convert_chat_to_responses_request_maps_image_and_file_parts() {
        let request = OpenAIChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAIChatMessage {
                role: "user".to_string(),
                content: Some(OpenAIMessageContent::Parts(vec![
                    OpenAIContentPart::Object(OpenAIContentPartObject {
                        kind: Some("text".to_string()),
                        text: Some("See attached".to_string()),
                        ..Default::default()
                    }),
                    OpenAIContentPart::Object(OpenAIContentPartObject {
                        kind: Some("image_url".to_string()),
                        image_url: Some(OpenAIImageUrl {
                            url: "https://example.com/x.png".to_string(),
                        }),
                        ..Default::default()
                    }),
                    OpenAIContentPart::Object(
                        serde_json::from_value(json!({
                            "type": "file",
                            "file": {
                                "filename": "report.pdf",
                                "file_data": "data:application/pdf;base64,abc"
                            }
                        }))
                        .unwrap(),
                    ),
                ])),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            extra: Map::new(),
        };

        let responses = convert_chat_to_responses_request(&request);

        assert_eq!(
            responses.input[0],
            ResponsesRequestInputItem::Message {
                role: "user".to_string(),
                content: ResponsesMessageContent::Parts(vec![
                    ResponsesMessagePart {
                        kind: "input_text".to_string(),
                        text: Some("See attached".to_string()),
                        extra: Map::new(),
                    },
                    ResponsesMessagePart {
                        kind: "input_image".to_string(),
                        text: None,
                        extra: {
                            let mut extra = Map::new();
                            extra.insert(
                                "image_url".to_string(),
                                json!("https://example.com/x.png"),
                            );
                            extra
                        },
                    },
                    ResponsesMessagePart {
                        kind: "input_file".to_string(),
                        text: None,
                        extra: {
                            let mut extra = Map::new();
                            extra.insert("filename".to_string(), json!("report.pdf"));
                            extra.insert(
                                "file_data".to_string(),
                                json!("data:application/pdf;base64,abc"),
                            );
                            extra
                        },
                    },
                ]),
            }
        );
    }

    #[test]
    fn test_convert_responses_to_chat_response_preserves_text_and_tool_calls() {
        let response = ResponsesResponse {
            id: Some("resp_123".to_string()),
            model: Some("gpt-4o".to_string()),
            output: vec![
                json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Let me check."}]
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}"
                }),
            ],
            usage: ResponsesUsage {
                input_tokens: Some(12),
                output_tokens: Some(7),
                cache_read_input_tokens: Some(90),
                cache_creation_input_tokens: Some(15),
            },
        };

        let chat = convert_responses_to_chat_response(&response);

        assert_eq!(chat.id, "resp_123");
        assert_eq!(chat.model, "gpt-4o");
        assert_eq!(chat.usage.prompt_tokens, 12);
        assert_eq!(chat.usage.completion_tokens, 7);
        assert_eq!(chat.usage.cache_read_input_tokens, Some(90));
        assert_eq!(chat.usage.cache_creation_input_tokens, Some(15));
        assert_eq!(chat.usage.total_tokens, 19);
        assert_eq!(chat.choices[0].finish_reason, "tool_calls");
        assert_eq!(
            chat.choices[0].message.content.as_deref(),
            Some("Let me check.")
        );
        assert_eq!(
            chat.choices[0].message.tool_calls.as_ref().unwrap()[0].id,
            "call_1"
        );
        assert_eq!(
            chat.choices[0].message.tool_calls.as_ref().unwrap()[0]
                .function
                .name,
            "get_weather"
        );
    }

    #[test]
    fn test_convert_responses_to_chat_response_ignores_non_text_message_parts() {
        let response = ResponsesResponse {
            id: Some("resp_456".to_string()),
            model: Some("gpt-4.1".to_string()),
            output: vec![json!({
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Visible text"},
                    {"type": "refusal", "refusal": "hidden"},
                    {"type": "reasoning", "summary": []}
                ]
            })],
            usage: ResponsesUsage::default(),
        };

        let chat = convert_responses_to_chat_response(&response);

        assert_eq!(
            chat.choices[0].message.content.as_deref(),
            Some("Visible text")
        );
        assert_eq!(chat.choices[0].finish_reason, "stop");
    }
}
