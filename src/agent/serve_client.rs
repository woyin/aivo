//! Client → loopback serve: the `Complete` handler. Streams an OpenAI chat
//! completion and assembles the assistant message (content + tool_calls). This
//! is the client's sole provider I/O — serve translates to the real upstream,
//! so this only ever speaks OpenAI chat. `on_delta` fires per content/reasoning
//! delta for live rendering.

use crate::agent::protocol::{AssistantMessage, ChatRequest, ToolCall};
use futures::StreamExt;
use serde_json::{Value, json};

/// Upper bound on parallel tool calls in one streamed response — guards the
/// index-keyed accumulator against a bogus huge `index` from upstream.
const MAX_TOOL_CALLS: usize = 256;

/// A streamed delta handed to the caller's render callback: the visible answer
/// text, or the model's reasoning/thinking (DeepSeek `reasoning_content`, the
/// generic `reasoning`, or an Anthropic-style `thinking` field). One callback so
/// the caller can keep a single "anything streamed yet?" flag.
pub enum StreamDelta<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

/// A failed provider call. The status and `Retry-After` let the engine decide
/// retryability from the code (not prose) and honor the server's backoff.
#[derive(Debug)]
pub struct ServeError {
    pub message: String,
    /// `None` for a transport failure (no response) or a mid-stream drop.
    pub status: Option<u16>,
    pub retry_after: Option<std::time::Duration>,
}

impl ServeError {
    fn transport(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            status: None,
            retry_after: None,
        }
    }
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// POST `request` to `{base_url}/v1/chat/completions` (the loopback serve),
/// stream the SSE response, and return the assembled assistant message.
pub async fn complete(
    client: &reqwest::Client,
    base_url: &str,
    auth_token: Option<&str>,
    request: &ChatRequest,
    // `+ Send` so the chat TUI can run the engine (and this call) on a spawned task.
    // Fires per content/reasoning delta for live rendering.
    on_delta: &mut (dyn FnMut(StreamDelta) + Send),
) -> Result<AssistantMessage, ServeError> {
    let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let mut body = json!({
        "model": request.model,
        "messages": request.messages,
        "stream": true,
    });
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.clone());
    }
    if let Value::Object(map) = &mut body {
        // Pass-through extras (temperature, tool_choice, …) without clobbering
        // the fields we set above.
        for (k, v) in &request.extra {
            map.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }

    let mut req = client.post(&url).json(&body);
    if let Some(t) = auth_token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .await
        .map_err(|e| ServeError::transport(format!("request failed: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        // `Retry-After` is seconds (the only form providers send).
        let retry_after = resp
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(std::time::Duration::from_secs);
        let text = resp.text().await.unwrap_or_default();
        return Err(ServeError {
            message: format!("upstream {status}: {}", text.trim()),
            status: Some(status.as_u16()),
            retry_after,
        });
    }

    let mut content = String::new();
    let mut tools: Vec<ToolAcc> = Vec::new();
    let mut usage: Option<Value> = None;
    let mut model: Option<String> = None;
    // Accumulate raw BYTES, not lossily-decoded chunks: a multi-byte char (CJK,
    // emoji) can straddle a chunk boundary, and decoding each chunk separately
    // would turn each half into a replacement char. A `\n`-terminated line never
    // splits a char, so we decode only complete lines.
    let mut buf: Vec<u8> = Vec::new();
    let mut done = false;
    let mut truncated = false;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            // A mid-stream drop AFTER a text-only partial reply: keep what
            // already streamed (the user saw it), flagged as truncated —
            // matching the plain-chat sender. But if a tool call was
            // mid-assembly its arguments may be truncated, so bail rather than
            // risk executing a malformed call.
            Err(_) if !content.is_empty() && tools.is_empty() => {
                truncated = true;
                break;
            }
            Err(e) => return Err(ServeError::transport(format!("stream error: {e}"))),
        };
        buf.extend_from_slice(&bytes);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                done = true;
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(u) = v.get("usage")
                && !u.is_null()
            {
                merge_usage(&mut usage, u);
            }
            if model.is_none()
                && let Some(m) = v.get("model").and_then(|x| x.as_str())
                && !m.is_empty()
            {
                model = Some(m.to_string());
            }
            let Some(delta) = v.pointer("/choices/0/delta") else {
                continue;
            };
            // Reasoning/thinking arrives on its own delta fields (no `content`).
            // Stream it for live rendering but DON'T fold it into `content`: it's
            // not part of the assistant's reply sent back on the next turn.
            if let Some(r) = delta
                .get("reasoning_content")
                .and_then(|x| x.as_str())
                .or_else(|| delta.get("reasoning").and_then(|x| x.as_str()))
                .or_else(|| delta.get("thinking").and_then(|x| x.as_str()))
                && !r.is_empty()
            {
                on_delta(StreamDelta::Reasoning(r));
            }
            if let Some(c) = delta.get("content").and_then(|x| x.as_str())
                && !c.is_empty()
            {
                content.push_str(c);
                on_delta(StreamDelta::Text(c));
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(|x| x.as_array()) {
                accumulate_tool_calls(tcs, &mut tools);
            }
        }
        if done {
            break;
        }
    }

    let tool_calls = tools
        .into_iter()
        .enumerate()
        .filter(|(_, a)| !a.name.is_empty())
        .map(|(i, a)| {
            let arguments = repair_tool_arguments(&a.name, &a.args);
            ToolCall {
                id: if a.id.is_empty() {
                    format!("call_{i}")
                } else {
                    a.id
                },
                name: a.name,
                arguments,
            }
        })
        .collect();

    Ok(AssistantMessage {
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        usage,
        truncated,
        model,
    })
}

/// Parse tool-call args; repair a truncated/malformed JSON string before falling
/// back to `{}`. Read-only only — a repaired mutating call could run a wrong command.
fn repair_tool_arguments(name: &str, raw: &str) -> Value {
    if let Ok(v) = serde_json::from_str::<Value>(raw) {
        return v;
    }
    if crate::agent::tools::is_read_only(name)
        && let Some(fixed) = close_truncated_json(raw)
        && let Ok(v) = serde_json::from_str::<Value>(&fixed)
    {
        return v;
    }
    json!({})
}

/// Close a truncated JSON value; `None` if nothing is open (structural, not truncation).
fn close_truncated_json(raw: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for c in raw.chars() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    if !in_string && stack.is_empty() {
        return None;
    }
    let mut out = raw.to_string();
    if in_string {
        out.push('"');
    }
    let trimmed = out.trim_end();
    if trimmed.ends_with(',') {
        out.truncate(trimmed.len() - 1);
    } else if trimmed.ends_with(':') {
        out.push_str("null");
    }
    while let Some(close) = stack.pop() {
        out.push(close);
    }
    Some(out)
}

#[derive(Default)]
struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

/// Merge a streamed `tool_calls` delta array into the per-index accumulators.
/// OpenAI sends `id`/`function.name` once and `function.arguments` as fragments.
/// Some providers (e.g. qwen) re-send the *full* `function.name` on every delta,
/// so the name is assigned (replaced), not appended — otherwise `run_bash` would
/// accumulate into `run_bashrun_bashrun_bash…` and fail tool lookup. Arguments are
/// still genuine fragments and are appended.
fn accumulate_tool_calls(tcs: &[Value], tools: &mut Vec<ToolAcc>) {
    for tc in tcs {
        let idx = tc.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        // Bound the index so a bogus/huge `index` from upstream can't grow the
        // accumulator unboundedly (a 2^31 index would OOM). No real response has
        // anywhere near this many parallel tool calls.
        if idx >= MAX_TOOL_CALLS {
            continue;
        }
        while tools.len() <= idx {
            tools.push(ToolAcc::default());
        }
        let acc = &mut tools[idx];
        if let Some(id) = tc.get("id").and_then(|x| x.as_str())
            && !id.is_empty()
        {
            acc.id = id.to_string();
        }
        if let Some(f) = tc.get("function") {
            if let Some(n) = f.get("name").and_then(|x| x.as_str())
                && !n.is_empty()
            {
                acc.name = n.to_string();
            }
            if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                acc.args.push_str(a);
            }
        }
    }
}

/// Fold a streamed `usage` object into the running one by field-wise max, so a
/// later partial chunk (e.g. an Anthropic-bridged final delta carrying only
/// `output_tokens`) can't wipe an input count and collapse the footer's fill.
fn merge_usage(acc: &mut Option<Value>, incoming: &Value) {
    match acc {
        Some(existing) => merge_numeric_max(existing, incoming),
        None => *acc = Some(incoming.clone()),
    }
    if let Some(existing) = acc {
        floor_total_tokens(existing);
    }
}

/// Deep-merge keeping the max of each numeric leaf; a `null` can't clear a value.
fn merge_numeric_max(acc: &mut Value, incoming: &Value) {
    match (acc, incoming) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                match a.get_mut(k) {
                    Some(slot) => merge_numeric_max(slot, v),
                    None => {
                        a.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (slot @ Value::Number(_), inc @ Value::Number(_)) => {
            if inc.as_f64() > slot.as_f64() {
                *slot = inc.clone();
            }
        }
        (slot, inc) if !inc.is_null() => *slot = inc.clone(),
        _ => {}
    }
}

/// Floor `total_tokens` to the input+output component sum (mirrors `usage_tokens`);
/// never lowers a larger provider total, so its total-first shortcut can't understate.
fn floor_total_tokens(usage: &mut Value) {
    let Some(obj) = usage.as_object_mut() else {
        return;
    };
    let get = |k: &str| obj.get(k).and_then(Value::as_u64).unwrap_or(0);
    let out = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let component = match obj.get("prompt_tokens").and_then(Value::as_u64) {
        Some(prompt) => prompt.saturating_add(out),
        None => get("input_tokens")
            .saturating_add(get("cache_read_input_tokens"))
            .saturating_add(get("cache_creation_input_tokens"))
            .saturating_add(out),
    };
    if component > 0 && component > get("total_tokens") {
        obj.insert("total_tokens".into(), json!(component));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// One-shot raw-HTTP server that returns `sse_body` as a chat stream.
    fn spawn_sse(sse_body: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 8192];
                let _ = sock.read(&mut buf); // drain the request
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    sse_body.len(),
                    sse_body
                );
                let _ = sock.write_all(resp.as_bytes());
                let _ = sock.flush();
            }
        });
        port
    }

    fn req() -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![json!({"role":"user","content":"hi"})],
            tools: vec![],
            extra: serde_json::Map::new(),
        }
    }

    #[tokio::test]
    async fn streams_content_and_split_tool_call() {
        // Content in two deltas; one tool call whose arguments arrive split.
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.txt\\\"}\"}}]}}]}\n\n\
data: [DONE]\n\n";
        let port = spawn_sse(body);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut seen = String::new();
        let msg = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |d| {
                if let StreamDelta::Text(t) = d {
                    seen.push_str(t)
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(seen, "Hello");
        assert_eq!(msg.content.as_deref(), Some("Hello"));
        assert!(!msg.truncated);
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "read_file");
        assert_eq!(msg.tool_calls[0].id, "call_1");
        assert_eq!(msg.tool_calls[0].arguments["path"], "a.txt");
    }

    #[tokio::test]
    async fn captures_upstream_model_from_chunks() {
        let body = "data: {\"model\":\"deepseek-v4-flash\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
data: [DONE]\n\n";
        let port = spawn_sse(body);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let msg = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |_| {},
        )
        .await
        .unwrap();
        assert_eq!(msg.model.as_deref(), Some("deepseek-v4-flash"));
    }

    #[test]
    fn accumulate_tool_calls_bounds_a_huge_index() {
        let mut tools = Vec::new();
        // A bogus huge index must be ignored, not allocated up to.
        accumulate_tool_calls(
            &[json!({"index": 1_000_000_000_u64, "function": {"name": "x"}})],
            &mut tools,
        );
        assert!(
            tools.is_empty(),
            "huge index should be dropped, not allocated"
        );
        // Normal small indices still accumulate.
        accumulate_tool_calls(
            &[
                json!({"index": 0, "id": "a", "function": {"name": "f", "arguments": "{}"}}),
                json!({"index": 1, "id": "b", "function": {"name": "g", "arguments": "[]"}}),
            ],
            &mut tools,
        );
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "f");
        assert_eq!(tools[1].name, "g");
    }

    /// qwen (and some other OpenAI-compatible providers) re-send the *full*
    /// `function.name` on every delta instead of only the first. The name must be
    /// assigned, not appended — otherwise `run_bash` corrupts into
    /// `run_bashrun_bashrun_bash…` and fails tool lookup with "unknown tool".
    #[test]
    fn accumulate_tool_calls_handles_repeated_full_name() {
        let mut tools = Vec::new();
        accumulate_tool_calls(
            &[
                json!({"index": 0, "id": "c1", "function": {"name": "run_bash", "arguments": "{\"cmd\":"}}),
            ],
            &mut tools,
        );
        // Subsequent deltas repeat the whole name and carry argument fragments.
        accumulate_tool_calls(
            &[json!({"index": 0, "function": {"name": "run_bash", "arguments": "\"ls\"}"}})],
            &mut tools,
        );
        accumulate_tool_calls(
            &[json!({"index": 0, "function": {"name": "run_bash"}})],
            &mut tools,
        );
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "run_bash", "name must not duplicate");
        // Genuine argument fragments are still appended into one valid JSON string.
        assert_eq!(tools[0].args, "{\"cmd\":\"ls\"}");
    }

    /// A partial final chunk (only `output_tokens`) must not wipe the input count.
    #[test]
    fn merge_usage_keeps_input_when_final_chunk_is_output_only() {
        let mut usage = None;
        merge_usage(
            &mut usage,
            &json!({
                "prompt_tokens": 118_000,
                "completion_tokens": 1,
                "total_tokens": 118_001,
                "cache_read_input_tokens": 90_000
            }),
        );
        merge_usage(
            &mut usage,
            &json!({ "completion_tokens": 5_000, "total_tokens": 5_000 }),
        );
        let u = usage.unwrap();
        assert_eq!(u["prompt_tokens"], 118_000, "input must survive");
        assert_eq!(u["completion_tokens"], 5_000, "output takes the larger");
        assert_eq!(u["cache_read_input_tokens"], 90_000);
        assert_eq!(u["total_tokens"], 123_000);
        assert_eq!(crate::agent::tokens::usage_tokens(&Some(u)), 123_000);
    }

    #[test]
    fn merge_usage_deep_merges_details_and_ignores_null() {
        let mut usage = None;
        merge_usage(
            &mut usage,
            &json!({
                "prompt_tokens": 1_000,
                "completion_tokens": 10,
                "prompt_tokens_details": { "cached_tokens": 800 }
            }),
        );
        merge_usage(
            &mut usage,
            &json!({
                "prompt_tokens": null,
                "completion_tokens": 40,
                "prompt_tokens_details": { "cached_tokens": 800 }
            }),
        );
        let u = usage.unwrap();
        assert_eq!(u["prompt_tokens"], 1_000, "null must not clear the input");
        assert_eq!(u["completion_tokens"], 40);
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"], 800);
        assert_eq!(u["total_tokens"], 1_040);
    }

    #[test]
    fn merge_usage_single_complete_chunk_is_unchanged() {
        let mut usage = None;
        merge_usage(
            &mut usage,
            &json!({ "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120 }),
        );
        let u = usage.unwrap();
        assert_eq!(u["prompt_tokens"], 100);
        assert_eq!(u["completion_tokens"], 20);
        assert_eq!(u["total_tokens"], 120);
    }

    #[tokio::test]
    async fn complete_merges_usage_across_chunks() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"prompt_tokens\":50000,\"completion_tokens\":1,\"total_tokens\":50001}}\n\n\
data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"completion_tokens\":200,\"total_tokens\":200}}\n\n\
data: [DONE]\n\n";
        let port = spawn_sse(body);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let msg = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |_| {},
        )
        .await
        .unwrap();
        let usage = msg.usage.expect("usage captured");
        assert_eq!(usage["prompt_tokens"], 50_000);
        assert_eq!(usage["completion_tokens"], 200);
        assert_eq!(usage["total_tokens"], 50_200);
    }

    /// A multi-byte char split across two network chunks must be reassembled, not
    /// turned into replacement chars (the bug when each chunk was decoded alone).
    #[tokio::test]
    async fn reassembles_multibyte_char_split_across_chunks() {
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"中文\"}}]}\n\ndata: [DONE]\n\n";
        let body_bytes = body.as_bytes().to_vec();
        // Split inside the first 3-byte char (`中`), so its bytes straddle chunks.
        let split = body.find('中').unwrap() + 1;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut rbuf = [0u8; 8192];
                let _ = sock.read(&mut rbuf);
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body_bytes.len()
                );
                let _ = sock.write_all(header.as_bytes());
                let _ = sock.write_all(&body_bytes[..split]);
                let _ = sock.flush();
                std::thread::sleep(std::time::Duration::from_millis(50));
                let _ = sock.write_all(&body_bytes[split..]);
                let _ = sock.flush();
            }
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut seen = String::new();
        let msg = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |d| {
                if let StreamDelta::Text(t) = d {
                    seen.push_str(t)
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(msg.content.as_deref(), Some("中文"));
        assert_eq!(seen, "中文");
    }

    #[test]
    fn repairs_truncated_tool_arguments() {
        let ro = "read_file";
        assert_eq!(
            repair_tool_arguments(ro, r#"{"path":"a.txt"}"#)["path"],
            "a.txt"
        );
        assert_eq!(
            repair_tool_arguments(ro, r#"{"path":"a.tx"#)["path"],
            "a.tx"
        );
        assert_eq!(repair_tool_arguments(ro, r#"{"n":12"#)["n"], 12);
        assert_eq!(repair_tool_arguments(ro, r#"{"a":1,"#)["a"], 1);
        let v = repair_tool_arguments(ro, r#"{"a":"x","b":"#);
        assert_eq!(v["a"], "x");
        assert!(v["b"].is_null());
        assert_eq!(repair_tool_arguments(ro, r#"{"xs":[1,2"#)["xs"][1], 2);
        assert_eq!(
            repair_tool_arguments(ro, r#"{"q":"a\"b"#)["q"],
            "a\"b".to_string()
        );
        assert_eq!(repair_tool_arguments(ro, "not json at all"), json!({}));
        assert_eq!(repair_tool_arguments(ro, ""), json!({}));
    }

    #[test]
    fn does_not_repair_truncated_args_for_mutating_tools() {
        assert_eq!(
            repair_tool_arguments("run_bash", r#"{"command":"rm -rf /home/u"#),
            json!({})
        );
        assert_eq!(
            repair_tool_arguments("write_file", r#"{"path":"a.tx"#),
            json!({})
        );
        assert_eq!(
            repair_tool_arguments("run_bash", r#"{"command":"echo hi"}"#)["command"],
            "echo hi"
        );
    }

    #[tokio::test]
    async fn surfaces_upstream_error() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let body = "{\"error\":\"nope\"}";
                let resp = format!(
                    "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let err = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |_| {},
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("401"));
        assert_eq!(err.status, Some(401));
    }

    /// A mid-stream drop after a text-only partial reply keeps what streamed
    /// (matching the plain-chat sender) instead of discarding it as an error.
    /// The server claims a larger Content-Length than it sends, then closes —
    /// so reqwest reports an incomplete body (a stream error), not a clean EOF.
    #[tokio::test]
    async fn keeps_partial_text_on_midstream_error() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            if let Ok((mut sock, _)) = listener.accept() {
                let mut rbuf = [0u8; 8192];
                let _ = sock.read(&mut rbuf);
                // Promise 200 more bytes than we actually send.
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len() + 200
                );
                let _ = sock.write_all(header.as_bytes());
                let _ = sock.write_all(body.as_bytes());
                let _ = sock.flush();
                // Let reqwest deliver the partial chunk before we close short.
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let mut seen = String::new();
        let msg = complete(
            &client,
            &format!("http://127.0.0.1:{port}"),
            None,
            &req(),
            &mut |d| {
                if let StreamDelta::Text(t) = d {
                    seen.push_str(t)
                }
            },
        )
        .await
        .unwrap();
        assert_eq!(msg.content.as_deref(), Some("partial"));
        assert_eq!(seen, "partial");
        assert!(msg.tool_calls.is_empty());
        assert!(
            msg.truncated,
            "a kept partial must be flagged so it can't pass for a complete answer"
        );
    }
}
