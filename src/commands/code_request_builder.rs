//! Request building for chat: construct HTTP request bodies for OpenAI and
//! Anthropic chat completion APIs, including multimodal attachment encoding.
use anyhow::Result;

use crate::commands::code::is_document_mime;
use crate::services::anthropic_route_pipeline::inject_cache_control_on_last_block;
use crate::services::session_store::{AttachmentStorage, MessageAttachment};

use super::code::ChatMessage;

/// Lowest `reasoning_effort` to send when thinking is off, or `None` for a
/// non-reasoning model that 400s on the field. Catalog-first: the gpt-5 family
/// diverged (5.0 → `minimal`, 5.1+/5.4 → `none`, codex → `low`), so a name guess
/// 400s; the snapshot resolves it, name heuristics cover snapshot-absent models.
fn openai_chat_no_thinking_value(model: &str) -> Option<&'static str> {
    if let Some(limits) = crate::services::model_metadata::snapshot_limits(model) {
        for level in ["none", "minimal", "low"] {
            if limits.reasoning_efforts.iter().any(|e| e.as_str() == level) {
                return Some(level);
            }
        }
        // Reasoning-capable but no off level (e.g. gpt-5-pro) → omit, don't 400.
        if !limits.reasoning_efforts.is_empty() {
            return None;
        }
    }
    let lower = model.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if name.starts_with("gpt-5") || name.contains("codex") {
        Some("minimal")
    } else if name.starts_with("o1") || name.starts_with("o3") || name.starts_with("o4") {
        Some("low")
    } else {
        None
    }
}

fn google_supports_thinking_budget(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.contains("gemini-2.5") || name.contains("gemini-3")
}

fn apply_chat_no_thinking_openai_chat(body: &mut serde_json::Value, model: &str) {
    if let Some(value) = openai_chat_no_thinking_value(model) {
        body["reasoning_effort"] = serde_json::json!(value);
    }
}

fn apply_chat_no_thinking_responses(body: &mut serde_json::Value, model: &str) {
    if let Some(value) = openai_chat_no_thinking_value(model) {
        body["reasoning"] = serde_json::json!({ "effort": value });
    }
}

fn apply_chat_no_thinking_google(body: &mut serde_json::Value, model: &str) {
    if !google_supports_thinking_budget(model) {
        return;
    }
    let root = body.as_object_mut().expect("google body is a JSON object");
    let config = root
        .entry("generationConfig".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(obj) = config.as_object_mut() {
        obj.insert(
            "thinkingConfig".to_string(),
            serde_json::json!({ "thinkingBudget": 0 }),
        );
    }
}

pub(crate) fn format_text_attachment_content(name: &str, content: &str) -> String {
    format!("[Attached file: {name}]\n{content}")
}

pub(crate) fn build_openai_chat_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
    max_tokens: Option<u64>,
) -> Result<serde_json::Value> {
    let mut encoded_messages = Vec::with_capacity(messages.len());
    for message in messages {
        encoded_messages.push(build_openai_message(message)?);
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "stream": stream,
    });
    if let Some(mt) = max_tokens {
        body["max_tokens"] = serde_json::json!(mt);
    }
    if stream {
        body["stream_options"] = serde_json::json!({"include_usage": true});
    }
    apply_chat_no_thinking_openai_chat(&mut body, model);
    Ok(body)
}

/// Returns the inline data for a materialized attachment, or fails if it is still a FileRef.
fn require_inline(attachment: &MessageAttachment) -> Result<&str> {
    match &attachment.storage {
        AttachmentStorage::Inline { data } => Ok(data),
        AttachmentStorage::FileRef { path } => anyhow::bail!(
            "Attachment '{}' is unresolved. Expected inline data before sending.",
            path
        ),
    }
}

pub(crate) fn build_openai_message(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::json!({
            "role": message.role,
            "content": message.content,
        }));
    }

    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", attachment.mime_type, data),
                },
            }));
        } else if is_document_mime(&attachment.mime_type) {
            parts.push(serde_json::json!({
                "type": "file",
                "file": {
                    "filename": attachment.name,
                    "file_data": format!("data:{};base64,{}", attachment.mime_type, data),
                },
            }));
        } else {
            parts.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::json!({
        "role": message.role,
        "content": parts,
    }))
}

pub(crate) fn build_responses_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut input = Vec::new();
    let mut instructions_parts = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                instructions_parts.push(message.content.as_str());
            }
            continue;
        }
        input.push(build_responses_input_item(message)?);
    }

    let mut body = serde_json::json!({
        "model": model,
        "input": input,
        "stream": stream,
    });

    if !instructions_parts.is_empty() {
        body["instructions"] = serde_json::Value::String(instructions_parts.join("\n\n"));
    }

    apply_chat_no_thinking_responses(&mut body, model);
    Ok(body)
}

fn build_responses_input_item(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::json!({
            "type": "message",
            "role": message.role,
            "content": message.content,
        }));
    }

    let mut parts = Vec::new();
    if !message.content.is_empty() {
        parts.push(serde_json::json!({
            "type": "input_text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            parts.push(serde_json::json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", attachment.mime_type, data),
            }));
        } else if is_document_mime(&attachment.mime_type) {
            parts.push(serde_json::json!({
                "type": "input_file",
                "filename": attachment.name,
                "file_data": format!("data:{};base64,{}", attachment.mime_type, data),
            }));
        } else {
            parts.push(serde_json::json!({
                "type": "input_text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::json!({
        "type": "message",
        "role": message.role,
        "content": parts,
    }))
}

pub(crate) fn build_anthropic_request(
    model: &str,
    messages: &[ChatMessage],
    stream: bool,
) -> Result<serde_json::Value> {
    let mut system_parts = Vec::new();
    let mut encoded_messages = Vec::new();

    for message in messages {
        if message.role == "system" {
            if !message.content.is_empty() {
                system_parts.push(message.content.clone());
            }
            continue;
        }

        let role = if message.role == "assistant" {
            "assistant"
        } else {
            "user"
        };
        encoded_messages.push(serde_json::json!({
            "role": role,
            "content": build_anthropic_content(message)?,
        }));
    }

    let mut request = serde_json::json!({
        "model": model,
        "messages": encoded_messages,
        "max_tokens": 8096,
        "stream": stream,
    });
    if !system_parts.is_empty() {
        request["system"] = serde_json::json!([{
            "type": "text",
            "text": system_parts.join("\n\n"),
            "cache_control": {"type": "ephemeral"}
        }]);
    }

    // Add cache_control to the last user message for Anthropic prompt caching
    for msg in encoded_messages.iter_mut().rev() {
        if msg["role"] != "user" {
            continue;
        }
        if let Some(content) = msg.get_mut("content") {
            inject_cache_control_on_last_block(content);
        }
        break;
    }

    request["messages"] = serde_json::json!(encoded_messages);
    // Anthropic defaults to no extended thinking; sending `thinking: disabled`
    // would 400 on models that don't support the field at all (e.g.
    // claude-3-5-sonnet). Leaving it unset is the correct "no thinking".
    Ok(request)
}

fn build_anthropic_content(message: &ChatMessage) -> Result<serde_json::Value> {
    if message.attachments.is_empty() {
        return Ok(serde_json::Value::String(message.content.clone()));
    }

    let mut blocks = Vec::new();
    if !message.content.is_empty() {
        blocks.push(serde_json::json!({
            "type": "text",
            "text": message.content,
        }));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") {
            blocks.push(serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": data,
                },
            }));
        } else if is_document_mime(&attachment.mime_type) {
            blocks.push(serde_json::json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": attachment.mime_type,
                    "data": data,
                },
            }));
        } else {
            blocks.push(serde_json::json!({
                "type": "text",
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(serde_json::Value::Array(blocks))
}

pub(crate) fn build_google_request(
    model: &str,
    messages: &[ChatMessage],
) -> Result<serde_json::Value> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "system" => {
                if !message.content.is_empty() {
                    system_parts.push(message.content.clone());
                }
            }
            "assistant" => {
                let mut parts = Vec::new();
                if !message.content.is_empty() {
                    parts.push(serde_json::json!({"text": message.content}));
                }
                if !parts.is_empty() {
                    contents.push(serde_json::json!({"role": "model", "parts": parts}));
                }
            }
            _ => {
                // "user" or other roles
                let parts = build_google_user_parts(message)?;
                if !parts.is_empty() {
                    contents.push(serde_json::json!({"role": "user", "parts": parts}));
                }
            }
        }
    }

    let mut request = serde_json::json!({"contents": contents});

    if !system_parts.is_empty() {
        request["systemInstruction"] = serde_json::json!({
            "parts": [{"text": system_parts.join("\n\n")}]
        });
    }

    if contents.is_empty() {
        request["contents"] = serde_json::json!([{
            "role": "user",
            "parts": [{"text": ""}]
        }]);
    }

    apply_chat_no_thinking_google(&mut request, model);
    Ok(request)
}

fn build_google_user_parts(message: &ChatMessage) -> Result<Vec<serde_json::Value>> {
    let mut parts = Vec::new();

    if !message.content.is_empty() {
        parts.push(serde_json::json!({"text": message.content}));
    }

    for attachment in &message.attachments {
        let data = require_inline(attachment)?;
        if attachment.mime_type.starts_with("image/") || is_document_mime(&attachment.mime_type) {
            parts.push(serde_json::json!({
                "inlineData": {
                    "mimeType": attachment.mime_type,
                    "data": data,
                }
            }));
        } else {
            parts.push(serde_json::json!({
                "text": format_text_attachment_content(&attachment.name, data),
            }));
        }
    }

    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::AttachmentStorage;

    #[test]
    fn test_openai_chat_disable_uses_catalog_off_level() {
        // gpt-5 family diverged: 5.0 → minimal, 5.1+/5.4 → none, codex → low.
        for (model, expected) in [
            ("gpt-5", "minimal"),
            ("gpt-5-mini", "minimal"),
            ("gpt-5.2", "none"),
            ("gpt-5.4", "none"),
            ("gpt-5-codex", "low"),
        ] {
            let body = build_openai_chat_request(
                model,
                &[ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                }],
                false,
                None,
            )
            .unwrap();
            assert_eq!(body["reasoning_effort"], expected, "model={model}");
        }
    }

    #[test]
    fn test_openai_chat_disable_o_series_uses_low() {
        for model in ["o1-mini", "o3", "o4-mini"] {
            let body = build_openai_chat_request(
                model,
                &[ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                }],
                false,
                None,
            )
            .unwrap();
            assert_eq!(body["reasoning_effort"], "low", "model={model}");
        }
    }

    #[test]
    fn test_openai_chat_omits_reasoning_for_non_reasoning_models() {
        for model in ["gpt-4o", "gpt-4o-mini", "deepseek-chat"] {
            let body = build_openai_chat_request(
                model,
                &[ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                }],
                false,
                None,
            )
            .unwrap();
            assert!(
                body.get("reasoning_effort").is_none(),
                "model={model} should not carry reasoning_effort"
            );
        }
    }

    #[test]
    fn test_responses_disable_only_for_reasoning_models() {
        let gpt5 = build_responses_request(
            "gpt-5.4",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            false,
        )
        .unwrap();
        assert_eq!(gpt5["reasoning"]["effort"], "none"); // gpt-5.4 dropped minimal

        let gpt4o = build_responses_request(
            "gpt-4o",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            false,
        )
        .unwrap();
        assert!(gpt4o.get("reasoning").is_none());
    }

    #[test]
    fn test_anthropic_never_sets_thinking_field() {
        let body = build_anthropic_request(
            "claude-sonnet-4-6",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            false,
        )
        .unwrap();
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn test_google_thinking_budget_only_for_2_5_plus() {
        let g25 = build_google_request(
            "gemini-2.5-pro",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
        )
        .unwrap();
        assert_eq!(
            g25["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            0
        );

        let g15 = build_google_request(
            "gemini-1.5-pro",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
        )
        .unwrap();
        assert!(g15.get("generationConfig").is_none());
    }

    #[test]
    fn test_build_openai_chat_request_encodes_file_and_image_attachments() {
        let request = build_openai_chat_request(
            "gpt-4o",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "Review these".to_string(),
                reasoning_content: None,
                attachments: vec![
                    MessageAttachment {
                        name: "notes.md".to_string(),
                        mime_type: "text/markdown".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "# hello".to_string(),
                        },
                    },
                    MessageAttachment {
                        name: "diagram.png".to_string(),
                        mime_type: "image/png".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "YWJj".to_string(),
                        },
                    },
                ],
            }],
            true,
            None,
        )
        .unwrap();

        let parts = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert!(parts[1]["text"].as_str().unwrap().contains("notes.md"));
        assert_eq!(parts[2]["type"], "image_url");
        assert_eq!(parts[2]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn test_build_openai_chat_request_includes_max_tokens() {
        let with_cap = build_openai_chat_request(
            "deepseek-chat",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            true,
            Some(8192),
        )
        .unwrap();
        assert_eq!(with_cap["max_tokens"], 8192);
        // deepseek-chat is not a reasoning model — reasoning_effort must be
        // omitted to avoid 400s on strict providers.
        assert!(with_cap.get("reasoning_effort").is_none());

        let without_cap = build_openai_chat_request(
            "gpt-4o",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hi".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            true,
            None,
        )
        .unwrap();
        assert!(without_cap.get("max_tokens").is_none());
    }

    #[test]
    fn test_build_anthropic_request_encodes_image_attachment() {
        let request = build_anthropic_request(
            "claude-sonnet-4-5",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: String::new(),
                reasoning_content: None,
                attachments: vec![MessageAttachment {
                    name: "diagram.png".to_string(),
                    mime_type: "image/png".to_string(),
                    storage: AttachmentStorage::Inline {
                        data: "YWJj".to_string(),
                    },
                }],
            }],
            false,
        )
        .unwrap();

        let blocks = request["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "image");
        assert_eq!(blocks[0]["source"]["media_type"], "image/png");
        assert_eq!(blocks[0]["source"]["data"], "YWJj");
    }

    #[test]
    fn test_build_responses_request_basic() {
        let request = build_responses_request(
            "gpt-5.4",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
            true,
        )
        .unwrap();

        assert_eq!(request["model"], "gpt-5.4");
        assert_eq!(request["stream"], true);
        assert_eq!(request["input"][0]["type"], "message");
        assert_eq!(request["input"][0]["role"], "user");
        assert_eq!(request["input"][0]["content"], "hello");
        assert!(request.get("instructions").is_none());
    }

    #[test]
    fn test_build_responses_request_with_system() {
        let request = build_responses_request(
            "gpt-5.4",
            &[
                ChatMessage {
                    model: None,
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
            false,
        )
        .unwrap();

        assert_eq!(request["instructions"], "You are helpful.");
        assert_eq!(request["input"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_build_responses_request_with_attachments() {
        let request = build_responses_request(
            "gpt-5.4",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "Review this".to_string(),
                reasoning_content: None,
                attachments: vec![
                    MessageAttachment {
                        name: "notes.md".to_string(),
                        mime_type: "text/markdown".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "# hello".to_string(),
                        },
                    },
                    MessageAttachment {
                        name: "diagram.png".to_string(),
                        mime_type: "image/png".to_string(),
                        storage: AttachmentStorage::Inline {
                            data: "YWJj".to_string(),
                        },
                    },
                ],
            }],
            true,
        )
        .unwrap();

        let parts = request["input"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "input_text");
        assert!(parts[1]["text"].as_str().unwrap().contains("notes.md"));
        assert_eq!(parts[2]["type"], "input_image");
    }

    #[test]
    fn test_build_google_request_basic() {
        let request = build_google_request(
            "gemini-1.5-flash",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "hello".to_string(),
                reasoning_content: None,
                attachments: vec![],
            }],
        )
        .unwrap();

        let contents = request["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[0]["parts"][0]["text"], "hello");
    }

    #[test]
    fn test_build_google_request_with_system() {
        let request = build_google_request(
            "gemini-1.5-flash",
            &[
                ChatMessage {
                    model: None,
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
        )
        .unwrap();

        assert_eq!(
            request["systemInstruction"]["parts"][0]["text"],
            "You are helpful."
        );
        let contents = request["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
    }

    #[test]
    fn test_build_google_request_with_assistant() {
        let request = build_google_request(
            "gemini-1.5-flash",
            &[
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "hi".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "assistant".to_string(),
                    content: "hello!".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
                ChatMessage {
                    model: None,
                    role: "user".to_string(),
                    content: "thanks".to_string(),
                    reasoning_content: None,
                    attachments: vec![],
                },
            ],
        )
        .unwrap();

        let contents = request["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 3);
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[1]["parts"][0]["text"], "hello!");
    }

    #[test]
    fn test_build_google_request_with_image_attachment() {
        let request = build_google_request(
            "gemini-1.5-flash",
            &[ChatMessage {
                model: None,
                role: "user".to_string(),
                content: "describe this".to_string(),
                reasoning_content: None,
                attachments: vec![MessageAttachment {
                    name: "photo.png".to_string(),
                    mime_type: "image/png".to_string(),
                    storage: AttachmentStorage::Inline {
                        data: "YWJj".to_string(),
                    },
                }],
            }],
        )
        .unwrap();

        let parts = request["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["text"], "describe this");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "YWJj");
    }

    #[test]
    fn test_build_google_request_empty_messages() {
        let request = build_google_request("gemini-1.5-flash", &[]).unwrap();
        let contents = request["contents"].as_array().unwrap();
        assert!(!contents.is_empty());
        assert_eq!(contents[0]["role"], "user");
    }
}
