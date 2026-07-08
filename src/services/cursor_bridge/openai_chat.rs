//! OpenAI `/v1/chat/completions` adapter for cursor-backed tools (opencode, pi).
//! Translates inbound chat-completion requests into cursor-agent ACP prompts
//! and streams cursor's responses back as OpenAI `chat.completion.chunk` SSE.
//! Tool-using turns route through the [`super::mcp`] bridge so client tools
//! surface as MCP HTTP calls instead of being flattened to text.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CURSOR_ACP_SENTINEL, CursorAcpSession};
use crate::services::http_utils::{
    cors_header_block, extract_request_body, http_chunked_response_head_with_extra,
};

use super::mcp::{BridgeEvent, BridgeSession, McpBridge, ToolUseIdStyle};
use super::*;

// === OpenAI chat completions ===

pub(super) async fn handle_openai_chat(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> (u16, Option<String>) {
    match run_openai_chat(socket, state, request).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            // Errors that surface *before* we've sent any bytes get a clean JSON
            // 502; errors after the stream head is on the wire just close the
            // socket (the client sees a truncated SSE stream). Broken pipes
            // get 499 in the log so post-mortems can tell client disconnects
            // apart from real upstream failures.
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

pub(super) async fn run_openai_chat(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value =
        serde_json::from_str(body_str).context("parse OpenAI chat completion request body")?;
    if body.get("messages").and_then(Value::as_array).is_none() {
        return Err(anyhow!("`messages` array is required"));
    }
    let stream_flag = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let requested_model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);

    // Non-streaming resumption: drain any parked call before falling
    // through to the legacy path (see Anthropic equivalent for rationale).
    if !stream_flag && let Some((call_id, output)) = extract_last_openai_tool_message(&body) {
        state
            .mcp_bridge
            .deliver_and_drop_parked(
                &call_id,
                vec![json!({"type": "text", "text": output})],
                false,
            )
            .await;
    }

    if stream_flag && openai_chat_request_uses_tools(&body) {
        return run_openai_chat_bridged(socket, state, body, requested_model).await;
    }

    let image_blocks = extract_openai_image_blocks(&body)?;
    let parsed = ParsedTurn {
        stream_flag,
        requested_model,
        prompt: append_json_output_constraint(
            reduce_openai_request_to_prompt(&body),
            &body,
            !image_blocks.is_empty(),
        ),
        image_blocks,
    };
    if parsed.prompt.trim().is_empty() && parsed.image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }
    run_turn(
        socket,
        state,
        parsed,
        CURSOR_ACP_SENTINEL,
        stream_openai_chat_sse,
        openai_completion_body,
    )
    .await
}

// === OpenAI /chat/completions (opencode + pi) with MCP-bridged tools ===

pub(super) fn openai_chat_request_uses_tools(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty())
}

/// Convert OpenAI chat-completions tools (`{type: "function", function:
/// {name, description, parameters}}`) into the bridge's normalized
/// `{name, description, input_schema}` shape.
pub(super) fn extract_openai_chat_tools_normalized(body: &Value) -> Vec<Value> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(tools.len());
    for tool in tools {
        let inner = tool.get("function").unwrap_or(tool);
        let Some(name) = inner.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = inner
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("");
        let schema = inner
            .get("parameters")
            .or_else(|| inner.get("input_schema"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "object"}));
        out.push(json!({
            "name": name,
            "description": description,
            "input_schema": schema,
        }));
    }
    out
}

/// Find the latest `role: "tool"` message and return its
/// `(tool_call_id, content_string)`. Used by the resumption path.
pub(super) fn extract_last_openai_tool_message(body: &Value) -> Option<(String, String)> {
    let messages = body.get("messages")?.as_array()?;
    for msg in messages.iter().rev() {
        if msg.get("role").and_then(Value::as_str)? != "tool" {
            continue;
        }
        let id = msg.get("tool_call_id").and_then(Value::as_str)?.to_string();
        let content = match msg.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => {
                let mut acc = String::new();
                for p in parts {
                    if let Some(t) = p.get("text").and_then(Value::as_str) {
                        if !acc.is_empty() {
                            acc.push('\n');
                        }
                        acc.push_str(t);
                    }
                }
                acc
            }
            _ => String::new(),
        };
        return Some((id, content));
    }
    None
}

/// Variant of [`reduce_openai_request_to_prompt`] for the bridged route:
/// drops the `Available tools:` header (the MCP server exposes them now)
/// but keeps the message walk so prior tool loops stay in the prompt.
pub(super) fn reduce_openai_request_to_prompt_without_tools(body: &Value) -> String {
    let mut parts = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let text = extract_openai_message_text(msg.get("content"));
        match role {
            "system" | "developer" => {
                if !text.trim().is_empty() {
                    parts.push(format!("System: {text}"));
                }
            }
            "user" => {
                if !text.trim().is_empty() {
                    parts.push(format!("User: {text}"));
                }
            }
            "assistant" => {
                if !text.trim().is_empty() {
                    parts.push(format!("Assistant: {text}"));
                }
                if let Some(tcs) = msg.get("tool_calls").and_then(Value::as_array) {
                    for tc in tcs {
                        let name = tc
                            .get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(Value::as_str)
                            .unwrap_or("tool");
                        let args = tc
                            .get("function")
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .map(parse_loose_json)
                            .unwrap_or(Value::Null);
                        parts.push(format_tool_call_line(name, &args));
                    }
                }
            }
            "tool" => {
                let name = msg
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                parts.push(format_tool_result_block(name, &text));
            }
            other => {
                if !text.trim().is_empty() {
                    parts.push(format!("{other}: {text}"));
                }
            }
        }
    }
    parts.join("\n\n")
}

pub(super) async fn run_openai_chat_bridged(
    socket: &mut TcpStream,
    state: &RouterState,
    body: Value,
    requested_model: Option<String>,
) -> Result<Option<String>> {
    if let Some((call_id, output)) = extract_last_openai_tool_message(&body)
        && let Some(session) = state
            .mcp_bridge
            .resume_with_tool_result(
                &call_id,
                vec![json!({"type": "text", "text": output})],
                // OpenAI's tool messages have no first-class is_error flag;
                // callers communicate failure inline in the content string.
                // Defer interpretation to cursor's model rather than trying
                // to substring-match here.
                false,
            )
            .await
    {
        return run_openai_chat_bridged_resume(socket, state, session, &body, requested_model)
            .await;
    }
    run_openai_chat_bridged_fresh(socket, state, body, requested_model).await
}

pub(super) async fn run_openai_chat_bridged_fresh(
    socket: &mut TcpStream,
    state: &RouterState,
    body: Value,
    requested_model: Option<String>,
) -> Result<Option<String>> {
    let tools = extract_openai_chat_tools_normalized(&body);
    let image_blocks = extract_openai_image_blocks(&body)?;
    let prompt = append_json_output_constraint(
        reduce_openai_request_to_prompt_without_tools(&body),
        &body,
        !image_blocks.is_empty(),
    );
    if prompt.trim().is_empty() && image_blocks.is_empty() {
        return Err(anyhow!("reduced prompt is empty; no user-visible message"));
    }
    let input_tokens = estimate_tokens(&prompt);

    let (bridge_session, mut acp) = if let Some(slot) = take_mcp_prewarmed(state).await {
        McpBridge::take_for_use(&slot.bridge_session, tools).await;
        (slot.bridge_session, slot.acp)
    } else {
        let (bridge_session, mcp_url) = state
            .mcp_bridge
            .open_session(tools, ToolUseIdStyle::OpenAi)
            .await;
        let bridge_id = { bridge_session.lock().await.id.clone() };

        let acp_result = CursorAcpSession::open_with_mcp(
            &state.config.key,
            requested_model.as_deref(),
            &state.config.workspace_cwd,
            Some(&mcp_url),
        )
        .await
        .context("open cursor-agent ACP session with MCP bridge (openai chat)");

        match acp_result {
            Ok(s) => (bridge_session, s),
            Err(e) => {
                state.mcp_bridge.drop_session(&bridge_id).await;
                return Err(e);
            }
        }
    };
    let bridge_id = { bridge_session.lock().await.id.clone() };

    if let Some(model) = &requested_model
        && let Err(e) = acp.set_model(model).await
    {
        state.mcp_bridge.drop_session(&bridge_id).await;
        return Err(e).context("cursor-agent set_model");
    }
    if !image_blocks.is_empty() && !acp.supports_image_prompts() {
        state.mcp_bridge.drop_session(&bridge_id).await;
        return Err(anyhow!(image_capability_error()));
    }

    let response_model = acp
        .model_id()
        .map(str::to_string)
        .or(requested_model.clone())
        .unwrap_or_else(|| CURSOR_ACP_SENTINEL.to_string());

    let blocks = cursor_acp::assemble_prompt_blocks(&prompt, image_blocks);
    let stream = match acp.prompt_with_blocks(blocks).await {
        Ok(s) => s,
        Err(e) => {
            state.mcp_bridge.drop_session(&bridge_id).await;
            return Err(e).context("cursor-agent session/prompt");
        }
    };

    {
        let mut guard = bridge_session.lock().await;
        guard.attach_session(acp, stream);
    }

    stream_bridged_openai_chat_turn(
        socket,
        state,
        bridge_session,
        &bridge_id,
        &response_model,
        input_tokens,
    )
    .await
}

pub(super) async fn run_openai_chat_bridged_resume(
    socket: &mut TcpStream,
    state: &RouterState,
    bridge_session: Arc<tokio::sync::Mutex<BridgeSession>>,
    body: &Value,
    requested_model: Option<String>,
) -> Result<Option<String>> {
    let bridge_id = { bridge_session.lock().await.id.clone() };
    let input_tokens = estimate_tokens(&reduce_openai_request_to_prompt_without_tools(body));
    let response_model = requested_model.unwrap_or_else(|| CURSOR_ACP_SENTINEL.to_string());
    stream_bridged_openai_chat_turn(
        socket,
        state,
        bridge_session,
        &bridge_id,
        &response_model,
        input_tokens,
    )
    .await
}

pub(super) async fn stream_bridged_openai_chat_turn(
    socket: &mut TcpStream,
    state: &RouterState,
    bridge_session: Arc<tokio::sync::Mutex<BridgeSession>>,
    bridge_id: &str,
    response_model: &str,
    input_tokens: u64,
) -> Result<Option<String>> {
    let (acp, mut stream, mut event_rx) = match async {
        let mut guard = bridge_session.lock().await;
        let (acp, stream) = guard.take_active()?;
        let rx = guard.attach_event_sink();
        Ok::<_, anyhow::Error>((acp, stream, rx))
    }
    .await
    {
        Ok(triple) => triple,
        Err(e) => {
            // Race: bridge session is in the sessions map but its ACP
            // session / prompt stream was already taken (or never attached).
            // Tear it down and surface as a 500 instead of panicking.
            state.mcp_bridge.drop_session(bridge_id).await;
            return Err(e).context("bridge session lost its active ACP slot");
        }
    };

    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    if let Err(e) = socket.write_all(head.as_bytes()).await {
        {
            let mut guard = bridge_session.lock().await;
            guard.detach_event_sink();
        }
        drop(acp);
        drop(stream);
        state.mcp_bridge.drop_session(bridge_id).await;
        return Err(e).context("write OpenAI chat SSE head");
    }

    let chat_id = new_chat_completion_id();
    let created = current_unix_timestamp();

    let role_chunk = openai_chunk_frame(
        &chat_id,
        created,
        response_model,
        json!({"role": "assistant"}),
        None,
    );
    let _ = write_sse_chunk(socket, &role_chunk).await;

    let mut finish_reason = "stop".to_string();
    let mut aggregated = String::new();
    let mut parked = false;
    let mut turn_errored = false;
    let mut error_message = String::new();
    // Write failure => client hung up; stop draining cursor.
    let mut client_gone = false;

    // Flag a disconnect on write failure; loop checks it after the `select!`.
    macro_rules! send {
        ($e:expr) => {{
            if $e.await.is_err() {
                client_gone = true;
            }
        }};
    }

    'outer: loop {
        tokio::select! {
            biased;
            ev = event_rx.recv() => {
                match ev {
                    Some(BridgeEvent::ToolCall { tool_use_id, name, arguments }) => {
                        let args_json = arguments.to_string();
                        let open_chunk = openai_chunk_frame(
                            &chat_id,
                            created,
                            response_model,
                            json!({
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tool_use_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": ""},
                                }],
                            }),
                            None,
                        );
                        send!(write_sse_chunk(socket, &open_chunk));
                        let args_chunk = openai_chunk_frame(
                            &chat_id,
                            created,
                            response_model,
                            json!({
                                "tool_calls": [{
                                    "index": 0,
                                    "function": {"arguments": args_json},
                                }],
                            }),
                            None,
                        );
                        send!(write_sse_chunk(socket, &args_chunk));
                        // Only park if the client actually received the call.
                        if client_gone {
                            break 'outer;
                        }
                        finish_reason = "tool_calls".to_string();
                        parked = true;
                        break 'outer;
                    }
                    None => break 'outer,
                }
            }
            ev = stream.next() => {
                match ev {
                    Some(PromptEvent::Update(value)) => {
                        if let Some(text) = extract_agent_text(&value) {
                            aggregated.push_str(text);
                            let chunk = openai_chunk_frame(
                                &chat_id,
                                created,
                                response_model,
                                json!({"content": text}),
                                None,
                            );
                            send!(write_sse_chunk(socket, &chunk));
                        } else if let Some(reasoning) = extract_agent_thought(&value) {
                            // Surface thinking as `reasoning_content` deltas.
                            let chunk = openai_chunk_frame(
                                &chat_id,
                                created,
                                response_model,
                                json!({"reasoning_content": reasoning}),
                                None,
                            );
                            send!(write_sse_chunk(socket, &chunk));
                        } else {
                            send!(write_sse_chunk(socket, SSE_KEEPALIVE));
                        }
                    }
                    Some(PromptEvent::Done(result)) => {
                        match result {
                            Ok(v) => {
                                if finish_reason != "tool_calls" {
                                    finish_reason =
                                        openai_finish_reason(acp_stop_from_result(&v)).to_string();
                                }
                            }
                            Err(e) => {
                                turn_errored = true;
                                error_message = e.to_string();
                            }
                        }
                        break 'outer;
                    }
                    None => break 'outer,
                }
            }
        }
        if client_gone {
            break 'outer;
        }
    }

    // Disconnect or error => stop cursor; only a parked tool turn is kept.
    if (client_gone || turn_errored) && !parked {
        let _ = acp.cancel().await;
    }

    if client_gone {
        // Dead socket — skip close frames, go to teardown.
    } else if turn_errored {
        // Signal failure with an `error` object, not a clean finish + `[DONE]`.
        let _ = write_sse_chunk(socket, &openai_error_chunk(&error_message)).await;
        let _ = write_chunk_terminator(socket).await;
    } else {
        let final_chunk = openai_chunk_frame(
            &chat_id,
            created,
            response_model,
            json!({}),
            Some(finish_reason.as_str()),
        );
        let _ = write_sse_chunk(socket, &final_chunk).await;
        let output_tokens = estimate_tokens(&aggregated);
        let usage_chunk = openai_usage_chunk(
            &chat_id,
            created,
            response_model,
            input_tokens,
            output_tokens,
        );
        let _ = write_sse_chunk(socket, &usage_chunk).await;
        let _ = write_sse_chunk(socket, "data: [DONE]\n\n").await;
        let _ = write_chunk_terminator(socket).await;
    }

    {
        let mut guard = bridge_session.lock().await;
        guard.detach_event_sink();
        if parked && !client_gone {
            guard.return_active(acp, stream);
        } else {
            drop(acp);
            drop(stream);
        }
    }
    if !parked || client_gone {
        state.mcp_bridge.drop_session(bridge_id).await;
    }

    Ok(if aggregated.is_empty() {
        None
    } else {
        Some(aggregated)
    })
}

/// Streams Cursor session/update events into the socket as an OpenAI
/// `chat.completion.chunk` SSE feed, terminating with `data: [DONE]`. Returns
/// the aggregated assistant text so the dispatcher can log it.
pub(super) async fn stream_openai_chat_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let chat_id = new_chat_completion_id();
    let created = current_unix_timestamp();

    // Many OpenAI clients expect a leading `delta.role = "assistant"` chunk.
    let role_chunk =
        openai_chunk_frame(&chat_id, created, model, json!({"role": "assistant"}), None);
    write_sse_chunk(socket, &role_chunk).await?;

    let mut finish_reason = "stop".to_string();
    let mut aggregated = String::new();
    let mut error_message: Option<String> = None;
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    aggregated.push_str(text);
                    let chunk = openai_chunk_frame(
                        &chat_id,
                        created,
                        model,
                        json!({"content": text}),
                        None,
                    );
                    write_sse_chunk(socket, &chunk).await?;
                } else if let Some(reasoning) = extract_agent_thought(&value) {
                    let chunk = openai_chunk_frame(
                        &chat_id,
                        created,
                        model,
                        json!({"reasoning_content": reasoning}),
                        None,
                    );
                    write_sse_chunk(socket, &chunk).await?;
                } else {
                    // Keepalive: cursor can go silent 10+ s on internal work
                    // and pi drops the stream after ~9 s. Comment lines reset
                    // client idle timers without appearing in output.
                    write_sse_chunk(socket, SSE_KEEPALIVE).await?;
                }
            }
            PromptEvent::Done(result) => {
                match result {
                    Ok(v) => {
                        finish_reason = openai_finish_reason(acp_stop_from_result(&v)).to_string()
                    }
                    Err(err) => error_message = Some(err.to_string()),
                }
                break;
            }
        }
    }

    // Emit an `error` object, not a bogus `finish_reason: "error:<code>"`.
    if let Some(message) = error_message {
        write_sse_chunk(socket, &openai_error_chunk(&message)).await?;
        write_chunk_terminator(socket).await?;
        return Ok(aggregated);
    }

    let final_chunk = openai_chunk_frame(
        &chat_id,
        created,
        model,
        json!({}),
        Some(finish_reason.as_str()),
    );
    write_sse_chunk(socket, &final_chunk).await?;

    // Emit a usage-only chunk per OpenAI's stream_options=include_usage
    // convention. Modern SDKs accept it unconditionally; older ones ignore
    // unrecognized fields. Without this, Codex/Pi see prompt_tokens=0 and
    // can't display context-window usage.
    let output_tokens = estimate_tokens(&aggregated);
    let usage_chunk = openai_usage_chunk(&chat_id, created, model, input_tokens, output_tokens);
    write_sse_chunk(socket, &usage_chunk).await?;
    write_sse_chunk(socket, "data: [DONE]\n\n").await?;
    write_chunk_terminator(socket).await?;
    Ok(aggregated)
}

/// Map a normalized ACP stop reason onto OpenAI's `finish_reason` set.
pub(super) fn openai_finish_reason(stop: AcpStop) -> &'static str {
    match stop {
        AcpStop::MaxTokens => "length",
        AcpStop::Refusal => "content_filter",
        AcpStop::EndTurn => "stop",
    }
}

/// A terminal OpenAI streaming `error` frame for a failed turn.
pub(super) fn openai_error_chunk(message: &str) -> String {
    let payload = json!({
        "error": {"message": message, "type": "server_error", "code": null},
    });
    format!("data: {payload}\n\n")
}

pub(super) fn openai_usage_chunk(
    id: &str,
    created: i64,
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
) -> String {
    let payload = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens.saturating_add(completion_tokens),
        },
    });
    format!("data: {payload}\n\n")
}

pub(super) fn openai_completion_body(
    turn: &AggregatedTurn,
    model: &str,
    input_tokens: u64,
) -> Value {
    let mut message = json!({"role": "assistant", "content": turn.content});
    if !turn.reasoning.is_empty() {
        message["reasoning_content"] = Value::String(turn.reasoning.clone());
    }
    let completion_tokens = estimate_tokens(&turn.content);
    json!({
        "id": new_chat_completion_id(),
        "object": "chat.completion",
        "created": current_unix_timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": input_tokens.saturating_add(completion_tokens),
        },
    })
}

pub(super) fn openai_chunk_frame(
    id: &str,
    created: i64,
    model: &str,
    delta: Value,
    finish_reason: Option<&str>,
) -> String {
    let payload = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    });
    format!("data: {payload}\n\n")
}
/// Reduces an OpenAI chat-completions request body to a flat ACP prompt.
/// Preserves tool schemas as a compact "Available tools" header and turns
/// `tool_calls` (assistant) and `role=tool` messages into transcript markers so
/// the downstream agent sees what the original tool was doing.
pub(crate) fn reduce_openai_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_openai_tools_list(tools)
    {
        parts.push(block);
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let text = extract_openai_message_text(msg.get("content"));
        match role {
            "system" | "developer" => {
                if !text.trim().is_empty() {
                    parts.push(format!("System: {text}"));
                }
            }
            "user" => {
                if !text.trim().is_empty() {
                    parts.push(format!("User: {text}"));
                }
            }
            "assistant" => {
                if !text.trim().is_empty() {
                    parts.push(format!("Assistant: {text}"));
                }
                if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        if let Some(line) = format_openai_tool_call(call) {
                            parts.push(line);
                        }
                    }
                }
            }
            "tool" => {
                let name = msg
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| msg.get("tool_call_id").and_then(Value::as_str))
                    .unwrap_or("tool");
                parts.push(format_tool_result_block(name, &text));
            }
            other => {
                if !text.trim().is_empty() {
                    parts.push(format!("{other}: {text}"));
                }
            }
        }
    }
    parts.join("\n\n")
}
