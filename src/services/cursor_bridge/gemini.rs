//! Gemini `generateContent` / `streamGenerateContent` adapter for cursor-cli.
//! Translates Gemini-shaped requests into cursor-agent ACP prompts and
//! streams responses back as `GenerateContentResponse` chunks (no `[DONE]`
//! marker, stream closes on chunk-end). Tool-using turns route through the
//! [`super::mcp`] bridge; `functionResponse` resumption uses name-matching
//! because Gemini's `id` field isn't always populated.

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;

use crate::services::acp_client::PromptEvent;
use crate::services::cursor_acp::{self, CursorAcpSession};
use crate::services::http_utils::{
    cors_header_block, extract_request_body, http_chunked_response_head_with_extra,
};

use super::mcp::{BridgeEvent, BridgeSession, McpBridge, ToolUseIdStyle};
use super::*;

// === Gemini generateContent (gemini-cli) ===

/// Parsed metadata for an incoming Gemini-protocol request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct GeminiGenerate {
    pub(super) model: String,
    pub(super) stream: bool,
}

/// Extracts the model name and stream flag from a Gemini-API path.
///
/// Accepts `/v1beta/models/<model>:generateContent`,
/// `/v1beta/models/<model>:streamGenerateContent`, and the `/v1/models/...` /
/// `/models/...` variants gemini-cli sometimes emits depending on the base URL
/// it was given. Returns `None` for non-Gemini paths so the dispatcher can
/// fall through to the canonical OpenAI/Anthropic/Responses handlers.
pub(super) fn parse_gemini_generate_path(path: &str) -> Option<GeminiGenerate> {
    // Find the `/models/` segment; anything before it is the version prefix.
    let after_models = path
        .strip_prefix("/v1beta/models/")
        .or_else(|| path.strip_prefix("/v1/models/"))
        .or_else(|| path.strip_prefix("/models/"))?;
    let (model, action) = after_models.split_once(':')?;
    if model.is_empty() {
        return None;
    }
    let stream = match action {
        "streamGenerateContent" => true,
        "generateContent" => false,
        _ => return None,
    };
    Some(GeminiGenerate {
        model: model.to_string(),
        stream,
    })
}

pub(super) async fn handle_gemini_generate(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
    generate: &GeminiGenerate,
) -> (u16, Option<String>) {
    match run_gemini_generate(socket, state, request, generate).await {
        Ok(summary) => (200, summary),
        Err(err) => {
            let status = status_for_handler_error(&err);
            let msg = err.to_string();
            let _ = write_json_error(socket, status, &msg).await;
            (status, Some(msg))
        }
    }
}

pub(super) async fn run_gemini_generate(
    socket: &mut TcpStream,
    state: &RouterState,
    request: &str,
    generate: &GeminiGenerate,
) -> Result<Option<String>> {
    let body_str = extract_request_body(request).context("read request body")?;
    let body: Value =
        serde_json::from_str(body_str).context("parse Gemini generateContent request body")?;

    // Non-streaming resumption: drain any parked call before falling
    // through to the legacy path (see Anthropic equivalent for rationale).
    if !generate.stream
        && let Some((_name, id, text, is_error)) = extract_last_gemini_function_response(&body)
    {
        let content = vec![json!({"type": "text", "text": text})];
        if let Some(id) = id.as_deref() {
            state
                .mcp_bridge
                .deliver_and_drop_parked(id, content, is_error)
                .await;
        }
        // For id-less Gemini responses we don't tear down by name in this
        // non-stream path: the by-name fallback can cross-match concurrent
        // sessions ([[reference_cursor_acp_mcp_propagation]]), so when in
        // doubt we'd rather leak one parked call to its 600 s timeout
        // than mis-deliver a tool result to another conversation.
    }

    if generate.stream && gemini_request_uses_tools(&body) {
        return run_gemini_bridged(socket, state, body, generate.model.clone()).await;
    }

    let image_blocks = extract_gemini_image_blocks(&body)?;
    let parsed = ParsedTurn {
        stream_flag: generate.stream,
        requested_model: Some(generate.model.clone()),
        prompt: append_json_output_constraint(
            reduce_gemini_request_to_prompt(&body),
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
        &generate.model,
        stream_gemini_sse,
        gemini_response_body,
    )
    .await
}

// === Gemini generateContent (gemini-cli) with MCP-bridged tools ===

pub(super) fn gemini_request_uses_tools(body: &Value) -> bool {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return false;
    };
    tools.iter().any(|t| {
        t.get("functionDeclarations")
            .and_then(Value::as_array)
            .is_some_and(|d| !d.is_empty())
    })
}

/// Convert Gemini's `tools: [{functionDeclarations: [{name, description,
/// parameters}]}]` shape into the bridge's normalized
/// `{name, description, input_schema}` list. Multiple tool groups in
/// `tools[]` are flattened.
pub(super) fn extract_gemini_tools_normalized(body: &Value) -> Vec<Value> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for tool in tools {
        let Some(decls) = tool.get("functionDeclarations").and_then(Value::as_array) else {
            continue;
        };
        for decl in decls {
            let Some(name) = decl.get("name").and_then(Value::as_str) else {
                continue;
            };
            let description = decl
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            let schema = decl
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            out.push(json!({
                "name": name,
                "description": description,
                "input_schema": schema,
            }));
        }
    }
    out
}

/// Returns `(name, optional id, response_text)` for the latest
/// `functionResponse` part in the request's `contents`. Gemini's old shape
/// has only `name`; newer versions (1.5+) sometimes carry an `id` we can
/// match against our synthetic tool_use_id.
/// Returns `(name, optional id, response_text, is_error)`. The is_error
/// signal is inferred from a top-level `error` key in the structured
/// `response` (a soft convention in Gemini SDK examples; no formal spec).
pub(super) fn extract_last_gemini_function_response(
    body: &Value,
) -> Option<(String, Option<String>, String, bool)> {
    let contents = body.get("contents")?.as_array()?;
    for content in contents.iter().rev() {
        let Some(parts) = content.get("parts").and_then(Value::as_array) else {
            continue;
        };
        for part in parts.iter().rev() {
            let Some(resp) = part.get("functionResponse") else {
                continue;
            };
            let name = resp
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let id = resp.get("id").and_then(Value::as_str).map(str::to_string);
            let response = resp.get("response");
            let is_error = response
                .and_then(|v| v.as_object())
                .is_some_and(|o| o.contains_key("error"));
            let text = match response {
                Some(Value::String(s)) => s.clone(),
                Some(v) => v.to_string(),
                None => String::new(),
            };
            return Some((name, id, text, is_error));
        }
    }
    None
}

pub(super) fn reduce_gemini_request_to_prompt_without_tools(body: &Value) -> String {
    let mut parts = Vec::new();
    let system_text = extract_gemini_system_text(body.get("systemInstruction"));
    if !system_text.trim().is_empty() {
        parts.push(format!("System: {system_text}"));
    }
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let label = match role {
            "model" => "Assistant",
            "user" => "User",
            other => other,
        };
        for entry in flatten_gemini_content_parts(label, content.get("parts")) {
            parts.push(entry);
        }
    }
    parts.join("\n\n")
}

pub(super) async fn run_gemini_bridged(
    socket: &mut TcpStream,
    state: &RouterState,
    body: Value,
    model: String,
) -> Result<Option<String>> {
    if let Some((name, id, text, is_error)) = extract_last_gemini_function_response(&body) {
        let content = vec![json!({"type": "text", "text": text})];
        // When an id is present (Gemini 1.5+), it's authoritative. An
        // id-miss means the request doesn't belong to any current parked
        // call — fall through to fresh path rather than guessing by name
        // across concurrent sessions (which could cross-deliver one user's
        // answer to a different conversation parked on the same tool name).
        let session = match id.as_deref() {
            Some(id) => {
                state
                    .mcp_bridge
                    .resume_with_tool_result(id, content, is_error)
                    .await
            }
            None => {
                state
                    .mcp_bridge
                    .resume_with_tool_result_by_name(&name, content, is_error)
                    .await
            }
        };
        if let Some(session) = session {
            return run_gemini_bridged_resume(socket, state, session, &body, model).await;
        }
    }
    run_gemini_bridged_fresh(socket, state, body, model).await
}

pub(super) async fn run_gemini_bridged_fresh(
    socket: &mut TcpStream,
    state: &RouterState,
    body: Value,
    model: String,
) -> Result<Option<String>> {
    let tools = extract_gemini_tools_normalized(&body);
    let image_blocks = extract_gemini_image_blocks(&body)?;
    let prompt = append_json_output_constraint(
        reduce_gemini_request_to_prompt_without_tools(&body),
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
            .open_session(tools, ToolUseIdStyle::Gemini)
            .await;
        let bridge_id = { bridge_session.lock().await.id.clone() };

        let acp_result = CursorAcpSession::open_with_mcp(
            &state.config.key,
            Some(&model),
            &state.config.workspace_cwd,
            Some(&mcp_url),
        )
        .await
        .context("open cursor-agent ACP session with MCP bridge (gemini)");

        match acp_result {
            Ok(s) => (bridge_session, s),
            Err(e) => {
                state.mcp_bridge.drop_session(&bridge_id).await;
                return Err(e);
            }
        }
    };
    let bridge_id = { bridge_session.lock().await.id.clone() };

    if let Err(e) = acp.set_model(&model).await {
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
        .unwrap_or_else(|| model.clone());

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

    stream_bridged_gemini_turn(
        socket,
        state,
        bridge_session,
        &bridge_id,
        &response_model,
        input_tokens,
    )
    .await
}

pub(super) async fn run_gemini_bridged_resume(
    socket: &mut TcpStream,
    state: &RouterState,
    bridge_session: Arc<tokio::sync::Mutex<BridgeSession>>,
    body: &Value,
    model: String,
) -> Result<Option<String>> {
    let bridge_id = { bridge_session.lock().await.id.clone() };
    let input_tokens = estimate_tokens(&reduce_gemini_request_to_prompt_without_tools(body));
    stream_bridged_gemini_turn(
        socket,
        state,
        bridge_session,
        &bridge_id,
        &model,
        input_tokens,
    )
    .await
}

pub(super) async fn stream_bridged_gemini_turn(
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
        return Err(e).context("write Gemini SSE head");
    }

    let mut aggregated = String::new();
    let mut output_tokens: u64 = 0;
    let mut finish_reason = "STOP";
    let mut parked = false;
    let mut error_message: Option<String> = None;
    // Write failure => client hung up; stop draining cursor.
    let mut client_gone = false;
    // Captured tool-call data for the final frame. We emit it together
    // with finishReason in a single candidate frame so strict gemini-cli
    // parsers (which key dispatch off "functionCall part AND finishReason
    // in the same chunk") see them atomically.
    let mut parked_call: Option<(String, String, Value)> = None;

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
                        finish_reason = "STOP";
                        parked = true;
                        parked_call = Some((tool_use_id, name, arguments));
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
                            output_tokens = output_tokens.saturating_add(estimate_tokens(text));
                            let frame = gemini_stream_text_frame(response_model, text);
                            send!(write_sse_chunk(socket, &frame));
                        } else {
                            send!(write_sse_chunk(socket, SSE_KEEPALIVE));
                        }
                    }
                    Some(PromptEvent::Done(result)) => {
                        match result {
                            Ok(v) => finish_reason = gemini_finish_reason(acp_stop_from_result(&v)),
                            Err(e) => error_message = Some(e.to_string()),
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
    if (client_gone || error_message.is_some()) && !parked {
        let _ = acp.cancel().await;
    }

    if client_gone {
        // Dead socket — skip the final frame, go to teardown.
    } else if let Some(message) = &error_message {
        // Signal failure with an `error` object, not `finishReason: STOP`.
        let _ = write_sse_chunk(socket, &gemini_error_frame(message)).await;
        let _ = write_chunk_terminator(socket).await;
    } else {
        let total_tokens = input_tokens.saturating_add(output_tokens);
        let final_frame = match parked_call.as_ref() {
            Some((tool_use_id, name, arguments)) => gemini_stream_function_call_final_frame(
                response_model,
                tool_use_id,
                name,
                arguments,
                finish_reason,
                input_tokens,
                output_tokens,
                total_tokens,
            ),
            None => gemini_stream_final_frame(
                response_model,
                finish_reason,
                input_tokens,
                output_tokens,
                total_tokens,
            ),
        };
        // A parked turn whose final frame didn't reach the client is dead.
        if write_sse_chunk(socket, &final_frame).await.is_err() {
            client_gone = true;
        }
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

/// Map a normalized ACP stop reason onto Gemini's `finishReason` set.
pub(super) fn gemini_finish_reason(stop: AcpStop) -> &'static str {
    match stop {
        AcpStop::MaxTokens => "MAX_TOKENS",
        AcpStop::Refusal => "SAFETY",
        AcpStop::EndTurn => "STOP",
    }
}

/// A terminal Gemini `error` frame for a mid-stream upstream failure.
pub(super) fn gemini_error_frame(message: &str) -> String {
    let payload = json!({
        "error": {"code": 500, "message": message, "status": "INTERNAL"},
    });
    format!("data: {payload}\n\n")
}

/// One-shot Gemini stream frame carrying the `functionCall` part and the
/// final `finishReason`+`usageMetadata` in the same candidate. Combining
/// the two avoids the split-frame bug where strict gemini-cli parsers
/// only dispatch a tool call when both signals appear atomically.
#[allow(clippy::too_many_arguments)]
pub(super) fn gemini_stream_function_call_final_frame(
    model: &str,
    tool_use_id: &str,
    name: &str,
    args: &Value,
    finish_reason: &str,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
) -> String {
    let payload = json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": tool_use_id,
                        "name": name,
                        "args": args,
                    }
                }],
            },
            "finishReason": finish_reason,
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": total_tokens,
        },
        "modelVersion": model,
    });
    format!("data: {payload}\n\n")
}

/// Reduces a Gemini `GenerateContentRequest` body to a flat ACP prompt.
/// Mirrors `reduce_anthropic_request_to_prompt` in shape: tool schemas become an
/// "Available tools" header, `systemInstruction` becomes a `System:` block, and
/// `functionCall`/`functionResponse` parts become inline transcript markers so
/// multi-turn tool loops survive the round trip.
pub(crate) fn reduce_gemini_request_to_prompt(body: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array)
        && let Some(block) = format_gemini_tools_list(tools)
    {
        parts.push(block);
    }
    let system_text = extract_gemini_system_text(body.get("systemInstruction"));
    if !system_text.trim().is_empty() {
        parts.push(format!("System: {system_text}"));
    }
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return parts.join("\n\n");
    };
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let label = match role {
            "model" => "Assistant",
            "user" => "User",
            other => other,
        };
        for entry in flatten_gemini_content_parts(label, content.get("parts")) {
            parts.push(entry);
        }
    }
    parts.join("\n\n")
}

pub(super) fn extract_gemini_system_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.clone(),
        Value::Object(_) => {
            let parts = value
                .get("parts")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut acc = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !acc.is_empty() {
                        acc.push('\n');
                    }
                    acc.push_str(text);
                }
            }
            acc
        }
        _ => String::new(),
    }
}

pub(super) fn flatten_gemini_content_parts(label: &str, parts: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let Some(parts) = parts.and_then(Value::as_array) else {
        return out;
    };
    let mut buffer = String::new();
    let flush = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.trim().is_empty() {
            out.push(format!("{label}: {buf}"));
        }
        buf.clear();
    };
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            if !buffer.is_empty() {
                buffer.push('\n');
            }
            buffer.push_str(text);
            continue;
        }
        if let Some(call) = part.get("functionCall") {
            flush(&mut buffer, &mut out);
            let name = call.get("name").and_then(Value::as_str).unwrap_or("tool");
            let args = call.get("args").cloned().unwrap_or(Value::Null);
            out.push(format_tool_call_line(name, &args));
            continue;
        }
        if let Some(resp) = part.get("functionResponse") {
            flush(&mut buffer, &mut out);
            let name = resp.get("name").and_then(Value::as_str).unwrap_or("tool");
            let result_text = extract_gemini_function_response_text(resp.get("response"));
            out.push(format_tool_result_block(name, &result_text));
            continue;
        }
        if part.get("inlineData").is_some() || part.get("fileData").is_some() {
            flush(&mut buffer, &mut out);
            out.push("[binary attachment omitted]".to_string());
        }
    }
    flush(&mut buffer, &mut out);
    out
}

pub(super) fn extract_gemini_function_response_text(value: Option<&Value>) -> String {
    let Some(value) = value else {
        return String::new();
    };
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

pub(super) fn format_gemini_tools_list(tools: &[Value]) -> Option<String> {
    let mut lines = Vec::new();
    for tool in tools {
        let Some(declarations) = tool.get("functionDeclarations").and_then(Value::as_array) else {
            continue;
        };
        for decl in declarations {
            let Some(name) = decl.get("name").and_then(Value::as_str) else {
                continue;
            };
            let description = decl
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            lines.push(format_tool_schema_line(name, description));
        }
    }
    finalize_tools_block(lines)
}

/// Builds the non-streaming `GenerateContentResponse` JSON body.
pub(super) fn gemini_response_body(turn: &AggregatedTurn, model: &str, input_tokens: u64) -> Value {
    let completion_tokens = estimate_tokens(&turn.content);
    let total_tokens = input_tokens.saturating_add(completion_tokens);
    let parts = if turn.content.is_empty() {
        json!([{"text": ""}])
    } else {
        json!([{"text": turn.content}])
    };
    json!({
        "candidates": [{
            "content": {"parts": parts, "role": "model"},
            "finishReason": "STOP",
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": total_tokens,
        },
        "modelVersion": model,
    })
}

/// Streams Cursor session updates as a Gemini `streamGenerateContent` SSE feed.
/// Each `data:` line is a partial `GenerateContentResponse`; the final frame
/// carries `finishReason` and `usageMetadata`. Unlike OpenAI's SSE, Gemini
/// streams have no `[DONE]` marker — the stream just ends when the chunked
/// body closes.
pub(super) async fn stream_gemini_sse(
    socket: &mut TcpStream,
    stream: &mut crate::services::acp_client::PromptStream,
    model: &str,
    input_tokens: u64,
) -> Result<String> {
    let head = http_chunked_response_head_with_extra(200, "text/event-stream", cors_header_block());
    socket.write_all(head.as_bytes()).await?;

    let mut aggregated = String::new();
    let mut output_tokens: u64 = 0;
    let mut finish_reason = "STOP";
    let mut error_message: Option<String> = None;
    while let Some(event) = stream.next().await {
        match event {
            PromptEvent::Update(value) => {
                if let Some(text) = extract_agent_text(&value) {
                    aggregated.push_str(text);
                    output_tokens = output_tokens.saturating_add(estimate_tokens(text));
                    let frame = gemini_stream_text_frame(model, text);
                    write_sse_chunk(socket, &frame).await?;
                } else {
                    // Same rationale as the OpenAI handler: cursor can spend
                    // 10+ s on internal work between text deltas. Emit a
                    // comment line to keep idle SDK timers alive.
                    write_sse_chunk(socket, SSE_KEEPALIVE).await?;
                }
            }
            PromptEvent::Done(result) => {
                match result {
                    Ok(v) => finish_reason = gemini_finish_reason(acp_stop_from_result(&v)),
                    Err(e) => error_message = Some(e.to_string()),
                }
                break;
            }
        }
    }
    if let Some(message) = error_message {
        write_sse_chunk(socket, &gemini_error_frame(&message)).await?;
        write_chunk_terminator(socket).await?;
        return Ok(aggregated);
    }
    let total_tokens = input_tokens.saturating_add(output_tokens);
    let final_frame = gemini_stream_final_frame(
        model,
        finish_reason,
        input_tokens,
        output_tokens,
        total_tokens,
    );
    write_sse_chunk(socket, &final_frame).await?;
    write_chunk_terminator(socket).await?;
    Ok(aggregated)
}

pub(super) fn gemini_stream_text_frame(model: &str, text: &str) -> String {
    let payload = json!({
        "candidates": [{
            "content": {"parts": [{"text": text}], "role": "model"},
            "index": 0,
        }],
        "modelVersion": model,
    });
    format!("data: {payload}\n\n")
}

pub(super) fn gemini_stream_final_frame(
    model: &str,
    finish_reason: &str,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
) -> String {
    let payload = json!({
        "candidates": [{
            "content": {"parts": [], "role": "model"},
            "finishReason": finish_reason,
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": total_tokens,
        },
        "modelVersion": model,
    });
    format!("data: {payload}\n\n")
}
