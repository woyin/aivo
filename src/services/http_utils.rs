//! Shared HTTP utilities for all built-in routers.
//!
//! Provides common functions for reading HTTP requests from raw TCP streams,
//! parsing headers, extracting bodies, and formatting responses.
//! Used by: anthropic_router, anthropic_to_openai_router, copilot_router,
//! responses_to_chat_router, gemini_router.

use anyhow::Result;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

use crate::constants::CONTENT_TYPE_JSON;
use std::future::Future;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::services::copilot_auth::{
    COPILOT_EDITOR_VERSION, COPILOT_INITIATOR_HEADER, COPILOT_INTEGRATION_ID,
    COPILOT_OPENAI_INTENT, CopilotTokenManager,
};

const MAX_REQUEST_HEADER_BYTES: usize = 64 * 1024;
const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
pub enum RequestReadError {
    Io(std::io::Error),
    HeaderTooLarge,
    BodyTooLarge { limit: usize },
    UnsupportedTransferEncoding,
    IncompleteHeaders,
}

impl std::fmt::Display for RequestReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error while reading HTTP request: {err}"),
            Self::HeaderTooLarge => write!(
                f,
                "HTTP request headers exceed {} bytes",
                MAX_REQUEST_HEADER_BYTES
            ),
            Self::BodyTooLarge { limit } => {
                write!(f, "HTTP request body exceeds {limit} bytes")
            }
            Self::UnsupportedTransferEncoding => {
                write!(f, "unsupported HTTP Transfer-Encoding: chunked")
            }
            Self::IncompleteHeaders => write!(f, "incomplete HTTP request headers"),
        }
    }
}

impl std::error::Error for RequestReadError {}

impl From<std::io::Error> for RequestReadError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// Reads a complete HTTP request from a TCP stream: headers + full body (using Content-Length).
pub async fn read_full_request(
    socket: &mut tokio::net::TcpStream,
) -> std::result::Result<Vec<u8>, RequestReadError> {
    use tokio::io::AsyncReadExt;

    let mut buf = Vec::with_capacity(65536); // 64KB initial capacity
    let mut tmp = vec![0u8; 16384]; // 16KB read buffer

    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);

        if let Some(expected_len) = inspect_request_buffer(&mut buf)? {
            while buf.len() < expected_len {
                let remaining = expected_len - buf.len();
                let mut body_buf = vec![0u8; remaining.min(tmp.len())];
                socket.read_exact(&mut body_buf).await?;
                buf.extend_from_slice(&body_buf);
            }
            break;
        }
    }

    if inspect_request_buffer(&mut buf)?.is_none() && !buf.is_empty() {
        return Err(RequestReadError::IncompleteHeaders);
    }

    Ok(buf)
}

/// Binds a router listener to a random localhost port and returns the listener and port.
pub async fn bind_local_listener() -> Result<(tokio::net::TcpListener, u16)> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    Ok((listener, port))
}

/// Builds an authorized POST request for OpenAI-compatible upstreams.
///
/// In Copilot mode, exchanges the GitHub token for a short-lived Copilot token
/// and targets the Copilot API at the path derived from `target_url`.  Otherwise,
/// posts directly to `target_url` with standard bearer auth.
pub async fn authorized_openai_post(
    client: &reqwest::Client,
    target_url: &str,
    api_key: &str,
    copilot_token_manager: Option<&CopilotTokenManager>,
    initiator: Option<&str>,
) -> Result<reqwest::RequestBuilder> {
    if let Some(tm) = copilot_token_manager {
        let (token, api_endpoint) = tm.get_token().await?;
        let copilot_path = copilot_path_from_target(target_url);
        let copilot_url = format!("{}{}", api_endpoint.trim_end_matches('/'), copilot_path);
        Ok(client
            .post(&copilot_url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", CONTENT_TYPE_JSON)
            .header("Editor-Version", COPILOT_EDITOR_VERSION)
            .header("Copilot-Integration-Id", COPILOT_INTEGRATION_ID)
            .header("Openai-Intent", COPILOT_OPENAI_INTENT)
            .header(COPILOT_INITIATOR_HEADER, initiator.unwrap_or("user")))
    } else {
        Ok(client
            .post(target_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", CONTENT_TYPE_JSON))
    }
}

/// Runs a raw TCP HTTP router whose handler returns a complete text HTTP response.
pub async fn run_text_router<State, Handler, Fut>(
    listener: tokio::net::TcpListener,
    state: Arc<State>,
    handler: Handler,
) -> Result<()>
where
    State: Send + Sync + 'static,
    Handler: Fn(String, Arc<State>) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = String> + Send + 'static,
{
    run_router_core(listener, state, move |request, state, mut socket| {
        let handler = handler.clone();
        async move {
            use tokio::io::AsyncWriteExt;
            let response = handler(request, state).await;
            let _ = socket.write_all(response.as_bytes()).await;
        }
    })
    .await
}

/// Finds the end of HTTP headers (the position of the first `\r\n\r\n`).
pub fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

pub fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|line| {
        let (header_name, value) = line.split_once(':')?;
        if header_name.trim().eq_ignore_ascii_case(name) {
            Some(value.trim())
        } else {
            None
        }
    })
}

/// Parses Content-Length from HTTP headers (case-insensitive).
pub fn parse_content_length(headers: &str) -> Option<usize> {
    header_value(headers, "content-length").and_then(|v| v.parse().ok())
}

fn has_chunked_transfer_encoding(headers: &str) -> bool {
    header_value(headers, "transfer-encoding")
        .map(|value| {
            value
                .split(',')
                .any(|encoding| encoding.trim().eq_ignore_ascii_case("chunked"))
        })
        .unwrap_or(false)
}

fn inspect_request_buffer(
    buf: &mut Vec<u8>,
) -> std::result::Result<Option<usize>, RequestReadError> {
    let Some(header_end) = find_header_end(buf) else {
        if buf.len() > MAX_REQUEST_HEADER_BYTES {
            return Err(RequestReadError::HeaderTooLarge);
        }
        return Ok(None);
    };

    let header_bytes = header_end + 4;
    if header_bytes > MAX_REQUEST_HEADER_BYTES {
        return Err(RequestReadError::HeaderTooLarge);
    }

    let headers = String::from_utf8_lossy(&buf[..header_end]);
    if has_chunked_transfer_encoding(&headers) {
        return Err(RequestReadError::UnsupportedTransferEncoding);
    }

    let content_length = parse_content_length(&headers).unwrap_or(0);
    if content_length > MAX_REQUEST_BODY_BYTES {
        return Err(RequestReadError::BodyTooLarge {
            limit: MAX_REQUEST_BODY_BYTES,
        });
    }

    let expected_len = header_bytes + content_length;
    if buf.len() > expected_len {
        buf.truncate(expected_len);
    }

    Ok(Some(expected_len))
}

/// Extracts the HTTP request body (everything after the blank line separator).
/// Returns an error for malformed requests that are missing `\r\n\r\n`.
pub fn extract_request_body(request: &str) -> Result<&str> {
    let pos = request
        .find("\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    Ok(request[pos + 4..].trim_end_matches('\0').trim())
}

/// Extracts request headers that are safe to forward upstream.
///
/// This preserves custom routing metadata sent by tool clients (for example
/// `x-provider`) while excluding hop-by-hop transport headers and headers that
/// the router intentionally manages itself, such as auth and content length.
pub fn extract_passthrough_headers(request: &str) -> Result<HeaderMap> {
    let header_end = find_header_end(request.as_bytes())
        .ok_or_else(|| anyhow::anyhow!("malformed HTTP request: missing header separator"))?;
    let headers = &request[..header_end];
    let mut out = HeaderMap::new();

    for line in headers.lines().skip(1) {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if !should_passthrough_header(name) {
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            continue;
        };
        let Ok(value) = HeaderValue::from_str(value.trim()) else {
            continue;
        };
        out.append(name, value);
    }

    Ok(out)
}

fn should_passthrough_header(name: &str) -> bool {
    let lower = name.trim().to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "host"
            | "connection"
            | "content-length"
            | "content-type"
            | "transfer-encoding"
            | "te"
            | "trailer"
            | "proxy-authorization"
            | "proxy-connection"
            | "authorization"
            | "accept-encoding"
            | "api-key"
            | "x-api-key"
            | "x-goog-api-key"
    ) {
        return false;
    }

    lower.starts_with("x-")
        || lower == "anthropic-version"
        || lower == "anthropic-beta"
        || lower.starts_with("anthropic-")
}

/// Removes the `anthropic-beta` header from a HeaderMap.
/// Used when a provider has been learned to reject beta headers (e.g. Bedrock, Vertex AI).
pub fn strip_beta_headers(headers: &mut HeaderMap) {
    headers.remove("anthropic-beta");
}

/// Returns true if a 400 error response body indicates the provider
/// rejected an `anthropic-beta` header value.
pub fn is_beta_header_rejection(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("invalid beta")
        || lower.contains("anthropic-beta")
        || (lower.contains("unexpected value") && lower.contains("beta"))
}

/// Extracts the HTTP request path from the first line (e.g., "POST /v1/messages HTTP/1.1" → "/v1/messages").
pub fn extract_request_path(request: &str) -> String {
    let first_line = request.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1].to_string()
    } else {
        "/".to_string()
    }
}

/// Returns true when the request is an HTTP POST whose path matches one of `paths`.
pub fn is_post_path(request: &str, paths: &[&str]) -> bool {
    if !request.starts_with("POST ") {
        return false;
    }
    let path = extract_request_path(request);
    let normalized_path = path.split('?').next().unwrap_or(path.as_str());
    paths.contains(&normalized_path)
}

/// Extracts the effective Content-Type from an upstream response.
pub fn response_content_type(response: &reqwest::Response) -> String {
    response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(CONTENT_TYPE_JSON)
        .to_string()
}

/// Returns the standard HTTP reason phrase for common status codes.
fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        301 => "Moved Permanently",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => {
            if status < 300 {
                "OK"
            } else if status < 400 {
                "Redirect"
            } else if status < 500 {
                "Client Error"
            } else {
                "Server Error"
            }
        }
    }
}

/// Returns the pre-formatted CORS header lines (without trailing \r\n\r\n).
pub fn cors_header_block() -> &'static str {
    "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Max-Age: 86400"
}

/// Formats the HTTP response head (status line + headers) without the body.
pub fn http_response_head(status: u16, content_type: &str, content_length: usize) -> String {
    http_response_head_with_extra(status, content_type, content_length, "")
}

/// Formats extra headers as a block to append before the final \r\n\r\n.
fn format_extra_headers(extra: &str) -> String {
    if extra.is_empty() {
        String::new()
    } else {
        format!("\r\n{}", extra)
    }
}

/// Formats the HTTP response head with extra headers injected before the final \r\n\r\n.
pub fn http_response_head_with_extra(
    status: u16,
    content_type: &str,
    content_length: usize,
    extra: &str,
) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close{}\r\n\r\n",
        status,
        reason_phrase(status),
        content_type,
        content_length,
        format_extra_headers(extra)
    )
}

/// Formats the HTTP response head for chunked transfer encoding.
pub fn http_chunked_response_head(status: u16, content_type: &str) -> String {
    http_chunked_response_head_with_extra(status, content_type, "")
}

/// Formats the chunked HTTP response head with extra headers injected.
pub fn http_chunked_response_head_with_extra(
    status: u16,
    content_type: &str,
    extra: &str,
) -> String {
    format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nTransfer-Encoding: chunked\r\nConnection: close{}\r\n\r\n",
        status,
        reason_phrase(status),
        content_type,
        format_extra_headers(extra)
    )
}

/// Formats a single chunk for HTTP chunked transfer encoding.
/// Returns empty vec for empty input.
pub fn format_http_chunk(chunk: &[u8]) -> Vec<u8> {
    if chunk.is_empty() {
        return Vec::new();
    }
    let mut formatted = format!("{:X}\r\n", chunk.len()).into_bytes();
    formatted.extend_from_slice(chunk);
    formatted.extend_from_slice(b"\r\n");
    formatted
}

/// Writes a buffered HTTP response (status + headers + body) to a TCP stream.
pub async fn write_buffered_response(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let headers = http_response_head(status, content_type, body.len());
    socket.write_all(headers.as_bytes()).await?;
    socket.write_all(body).await?;
    Ok(())
}

/// Like `run_text_router`, but passes the TCP socket to the handler so it can
/// stream responses directly (e.g. forwarding upstream SSE chunks in real time).
/// The handler is responsible for writing the full HTTP response to the socket.
pub async fn run_streaming_router<State, Handler, Fut>(
    listener: tokio::net::TcpListener,
    state: Arc<State>,
    handler: Handler,
) -> Result<()>
where
    State: Send + Sync + 'static,
    Handler: Fn(String, Arc<State>, tokio::net::TcpStream) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    run_router_core(listener, state, move |request, state, socket| {
        let handler = handler.clone();
        async move {
            handler(request, state, socket).await;
        }
    })
    .await
}

/// Shared accept-loop core for both text and streaming routers.
async fn run_router_core<State, Handler, Fut>(
    listener: tokio::net::TcpListener,
    state: Arc<State>,
    handler: Handler,
) -> Result<()>
where
    State: Send + Sync + 'static,
    Handler: Fn(String, Arc<State>, tokio::net::TcpStream) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let semaphore = Arc::new(tokio::sync::Semaphore::new(100));

    loop {
        let (mut socket, _) = listener.accept().await?;
        let state = state.clone();
        let handler = handler.clone();
        let permit = match semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => continue,
        };

        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;

            let _permit = permit;
            let read_result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                read_full_request(&mut socket),
            )
            .await;

            let request_bytes = match read_result {
                Ok(Ok(b)) => b,
                Ok(Err(err)) => {
                    let response = http_request_read_error_response(&err);
                    let _ = socket.write_all(response.as_bytes()).await;
                    return;
                }
                Err(_) => {
                    let _ = socket
                        .write_all(http_error_response(408, "Request read timed out").as_bytes())
                        .await;
                    return;
                }
            };
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            handler(request, state, socket).await;
        });
    }
}

/// Streams a reqwest Response as chunked HTTP to a TCP stream.
pub async fn write_streaming_response(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    content_type: &str,
    mut upstream: reqwest::Response,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let headers = http_chunked_response_head(status, content_type);
    socket.write_all(headers.as_bytes()).await?;
    while let Some(chunk) = upstream.chunk().await? {
        let formatted = format_http_chunk(&chunk);
        if !formatted.is_empty() {
            socket.write_all(&formatted).await?;
        }
    }
    socket.write_all(b"0\r\n\r\n").await?;
    Ok(())
}

/// Formats an HTTP response with the correct status line, Content-Type, and body.
pub fn http_response(status: u16, content_type: &str, body: &str) -> String {
    format!(
        "{}{}",
        http_response_head(status, content_type, body.len()),
        body
    )
}

/// Converts a buffered upstream response into a raw HTTP response string.
pub async fn buffered_reqwest_to_http_response(response: reqwest::Response) -> Result<String> {
    let status = response.status().as_u16();
    let content_type = response_content_type(&response);
    let body = response.bytes().await?;
    let body = String::from_utf8_lossy(&body);
    Ok(http_response(status, &content_type, &body))
}

/// Formats a JSON error response with the correct HTTP status line.
pub fn http_json_response(status: u16, body: &str) -> String {
    http_response(status, CONTENT_TYPE_JSON, body)
}

/// Formats a JSON error response body with an error message.
pub fn http_error_response(status: u16, message: &str) -> String {
    let body = serde_json::json!({"error": {"message": message}}).to_string();
    http_response(status, CONTENT_TYPE_JSON, &body)
}

pub fn http_request_read_error_response(error: &RequestReadError) -> String {
    match error {
        RequestReadError::HeaderTooLarge | RequestReadError::BodyTooLarge { .. } => {
            http_error_response(413, &error.to_string())
        }
        RequestReadError::UnsupportedTransferEncoding => {
            http_error_response(400, &error.to_string())
        }
        RequestReadError::IncompleteHeaders => http_error_response(400, &error.to_string()),
        RequestReadError::Io(_) => http_error_response(400, &error.to_string()),
    }
}

/// Extracts the API path from a target URL for Copilot routing.
/// Copilot's API doesn't use `/v1` prefix, so strip it.
///
/// - `"/v1/chat/completions"` → `"/chat/completions"`
/// - `"/v1/responses"` → `"/responses"`
/// - `"https://host/v1/chat/completions"` → `"/chat/completions"`
fn copilot_path_from_target(target_url: &str) -> &str {
    let path = if target_url.starts_with('/') {
        target_url
    } else {
        target_url
            .find("://")
            .and_then(|i| {
                target_url[i + 3..]
                    .find('/')
                    .map(|j| &target_url[i + 3 + j..])
            })
            .unwrap_or("/chat/completions")
    };
    path.strip_prefix("/v1").unwrap_or(path)
}

/// Constructs a target URL, collapsing any path segments that the base URL already
/// includes. If the base path ends with the same N segments that the target path
/// begins with (longest match), those duplicated leading segments are stripped from
/// the path. Handles `/v1`, `/v1beta`, `/anthropic`, `/openai/v1`, etc.
pub fn build_target_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let stripped_path = path.trim_start_matches('/');

    // Walk path segment boundaries left-to-right; for each cumulative prefix
    // (e.g. "v1", then "v1/messages"), check whether `base` already ends with
    // a `/<prefix>` segment boundary. The longest such match is the overlap
    // we strip. Zero allocations, single pass.
    let mut overlap_end = 0usize;
    let bytes = stripped_path.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'/' && base_ends_with_segment(base, &stripped_path[..i]) {
            overlap_end = i + 1;
        }
    }
    if base_ends_with_segment(base, stripped_path) {
        overlap_end = stripped_path.len();
    }

    let trimmed = &stripped_path[overlap_end..];
    if trimmed.is_empty() {
        base.to_string()
    } else {
        format!("{}/{}", base, trimmed)
    }
}

/// True iff `base` ends with `/<seg>` on a segment boundary (i.e. preceded by
/// a `/`). Pure substring check — no allocations.
fn base_ends_with_segment(base: &str, seg: &str) -> bool {
    if seg.is_empty() || !base.ends_with(seg) {
        return false;
    }
    base.len()
        .checked_sub(seg.len() + 1)
        .is_some_and(|idx| base.as_bytes()[idx] == b'/')
}

/// Returns the current Unix timestamp in seconds.
pub fn current_unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parses a JSON value as a `u64`, accepting JSON integers, floats, and numeric strings.
pub fn parse_token_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_f64().map(|f| f as u64))
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

/// Returns the SSE payload for a `data:` line.
/// Accepts both `data: {...}` and `data:{...}`.
pub fn sse_data_payload(line: &str) -> Option<&str> {
    line.strip_prefix("data:").map(str::trim_start)
}

/// Creates a `reqwest::Client` with a configurable overall timeout.
/// If `secs` is 0, no overall timeout is applied.
pub fn router_http_client_with_timeout(secs: u64) -> reqwest::Client {
    let mut builder = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .pool_max_idle_per_host(10)
        .tcp_keepalive(std::time::Duration::from_secs(60));
    if secs > 0 {
        builder = builder.timeout(std::time::Duration::from_secs(secs));
    }
    builder.build().unwrap_or_else(|_| reqwest::Client::new())
}

/// Creates a `reqwest::Client` with connection pooling for router use.
/// Enables keep-alive for connection reuse across requests.
pub fn router_http_client() -> reqwest::Client {
    router_http_client_with_timeout(300)
}

/// Detects `X-Initiator` value from an Anthropic Messages API body.
/// Returns `"user"` for genuine user messages, `"agent"` for tool results / follow-ups.
pub fn copilot_initiator_from_anthropic(body: &Value) -> &'static str {
    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return "user",
    };
    let last = match messages.last() {
        Some(m) => m,
        None => return "user",
    };
    let role = last.get("role").and_then(|r| r.as_str()).unwrap_or("");
    if role != "user" {
        return "agent";
    }
    // Check if the content contains tool_result blocks
    if let Some(content) = last.get("content").and_then(|c| c.as_array()) {
        let has_tool_result = content
            .iter()
            .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"));
        if has_tool_result {
            return "agent";
        }
    }
    "user"
}

/// Detects `X-Initiator` value from an OpenAI Chat Completions body.
/// Returns `"user"` for genuine user messages, `"agent"` for tool/assistant follow-ups.
pub fn copilot_initiator_from_openai(body: &Value) -> &'static str {
    let messages = match body.get("messages").and_then(|m| m.as_array()) {
        Some(m) => m,
        None => return "user",
    };
    let last = match messages.last() {
        Some(m) => m,
        None => return "user",
    };
    let role = last.get("role").and_then(|r| r.as_str()).unwrap_or("");
    match role {
        "user" => "user",
        _ => "agent",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_find_header_end() {
        let buf = b"POST /v1 HTTP/1.1\r\nHost: localhost\r\n\r\nbody";
        assert_eq!(find_header_end(buf), Some(34));
    }

    #[test]
    fn test_find_header_end_none() {
        let buf = b"POST /v1 HTTP/1.1\r\nHost: localhost";
        assert_eq!(find_header_end(buf), None);
    }

    #[test]
    fn test_parse_content_length() {
        let headers = "POST /v1 HTTP/1.1\r\nContent-Length: 42\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), Some(42));
    }

    #[test]
    fn test_parse_content_length_case_insensitive() {
        let headers = "POST /v1 HTTP/1.1\r\ncontent-length: 100\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), Some(100));
    }

    #[test]
    fn test_parse_content_length_missing() {
        let headers = "POST /v1 HTTP/1.1\r\nHost: localhost";
        assert_eq!(parse_content_length(headers), None);
    }

    #[test]
    fn test_has_chunked_transfer_encoding() {
        let headers = "POST /v1 HTTP/1.1\r\nTransfer-Encoding: gzip, chunked\r\nHost: localhost";
        assert!(has_chunked_transfer_encoding(headers));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_large_header() {
        let mut buf = vec![b'a'; MAX_REQUEST_HEADER_BYTES + 1];
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::HeaderTooLarge));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_large_body() {
        let mut buf = format!(
            "POST /v1 HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_REQUEST_BODY_BYTES + 1
        )
        .into_bytes();
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::BodyTooLarge { .. }));
    }

    #[test]
    fn test_inspect_request_buffer_rejects_chunked_requests() {
        let mut buf =
            b"POST /v1 HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n4\r\ntest\r\n".to_vec();
        let err = inspect_request_buffer(&mut buf).unwrap_err();
        assert!(matches!(err, RequestReadError::UnsupportedTransferEncoding));
    }

    #[test]
    fn test_inspect_request_buffer_truncates_to_content_length() {
        let mut buf =
            b"POST /v1 HTTP/1.1\r\nContent-Length: 4\r\n\r\ntestEXTRA_TRAILING_BYTES".to_vec();
        let expected_len = inspect_request_buffer(&mut buf).unwrap().unwrap();
        assert_eq!(expected_len, buf.len());
        assert_eq!(
            std::str::from_utf8(&buf).unwrap(),
            "POST /v1 HTTP/1.1\r\nContent-Length: 4\r\n\r\ntest"
        );
    }

    #[test]
    fn test_extract_request_body() {
        let req =
            "POST /v1/messages HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"key\":\"val\"}";
        assert_eq!(extract_request_body(req).unwrap(), "{\"key\":\"val\"}");
    }

    #[test]
    fn test_extract_request_body_missing_separator() {
        let req = "POST /v1/messages HTTP/1.1";
        assert!(extract_request_body(req).is_err());
    }

    #[test]
    fn test_extract_request_body_short() {
        assert!(extract_request_body("AB").is_err());
    }

    #[test]
    fn test_extract_passthrough_headers_keeps_custom_provider_headers() {
        let req = concat!(
            "POST /v1/messages HTTP/1.1\r\n",
            "Host: localhost:8080\r\n",
            "Authorization: Bearer local-token\r\n",
            "x-api-key: upstream-token\r\n",
            "Content-Type: application/json\r\n",
            "x-provider: anthropic\r\n",
            "x-vercel-ai-gateway-team: team_123\r\n",
            "anthropic-beta: prompt-caching-2024-07-31\r\n",
            "\r\n",
            "{}"
        );

        let headers = extract_passthrough_headers(req).unwrap();
        assert_eq!(
            headers.get("x-provider").and_then(|v| v.to_str().ok()),
            Some("anthropic")
        );
        assert_eq!(
            headers
                .get("x-vercel-ai-gateway-team")
                .and_then(|v| v.to_str().ok()),
            Some("team_123")
        );
        assert_eq!(
            headers.get("anthropic-beta").and_then(|v| v.to_str().ok()),
            Some("prompt-caching-2024-07-31")
        );
        assert!(headers.get("authorization").is_none());
        assert!(headers.get("x-api-key").is_none());
        assert!(headers.get("content-type").is_none());
    }

    #[test]
    fn test_extract_passthrough_headers_requires_header_separator() {
        assert!(extract_passthrough_headers("POST /v1/messages HTTP/1.1").is_err());
    }

    #[test]
    fn test_extract_request_path() {
        let req = "POST /v1/messages HTTP/1.1\r\nHost: localhost";
        assert_eq!(extract_request_path(req), "/v1/messages");
    }

    #[test]
    fn test_extract_request_path_empty() {
        assert_eq!(extract_request_path(""), "/");
    }

    #[test]
    fn test_is_post_path_matches_supported_path() {
        let req = "POST /v1/messages HTTP/1.1\r\nHost: localhost";
        assert!(is_post_path(req, &["/v1/messages", "/messages"]));
    }

    #[test]
    fn test_is_post_path_ignores_query_string() {
        let req = "POST /v1/messages?beta=true HTTP/1.1\r\nHost: localhost";
        assert!(is_post_path(req, &["/v1/messages", "/messages"]));
    }

    #[test]
    fn test_is_post_path_rejects_wrong_method_or_path() {
        let get_req = "GET /v1/messages HTTP/1.1\r\nHost: localhost";
        let other_req = "POST /health HTTP/1.1\r\nHost: localhost";
        assert!(!is_post_path(get_req, &["/v1/messages"]));
        assert!(!is_post_path(other_req, &["/v1/messages"]));
    }

    #[test]
    fn test_reason_phrase() {
        assert_eq!(reason_phrase(200), "OK");
        assert_eq!(reason_phrase(400), "Bad Request");
        assert_eq!(reason_phrase(404), "Not Found");
        assert_eq!(reason_phrase(500), "Internal Server Error");
    }

    #[test]
    fn test_http_response_format() {
        let resp = http_response(200, CONTENT_TYPE_JSON, "{\"ok\":true}");
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Type: application/json"));
        assert!(resp.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn test_http_response_head_format() {
        let head = http_response_head(200, CONTENT_TYPE_JSON, 11);
        assert_eq!(
            head,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn test_http_chunked_response_head_format() {
        let head = http_chunked_response_head(200, "text/event-stream");
        assert_eq!(
            head,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        );
    }

    #[test]
    fn test_http_response_error_status() {
        let resp = http_response(500, CONTENT_TYPE_JSON, "{\"error\":true}");
        assert!(resp.starts_with("HTTP/1.1 500 Internal Server Error\r\n"));
    }

    #[test]
    fn test_http_error_response() {
        let resp = http_error_response(404, "Not found");
        assert!(resp.contains("404 Not Found"));
        assert!(resp.contains("Not found"));
    }

    #[test]
    fn test_http_request_read_error_response_uses_413_for_size_limits() {
        let resp = http_request_read_error_response(&RequestReadError::BodyTooLarge { limit: 123 });
        assert!(resp.starts_with("HTTP/1.1 413 Payload Too Large\r\n"));
    }

    #[test]
    fn test_build_target_url_with_v1() {
        assert_eq!(
            build_target_url("https://api.example.com/v1", "/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_without_v1() {
        assert_eq!(
            build_target_url("https://api.example.com", "/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_trailing_slash() {
        assert_eq!(
            build_target_url("https://example.com/v1/", "/chat/completions"),
            "https://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_collapses_v1beta() {
        assert_eq!(
            build_target_url(
                "https://generativelanguage.googleapis.com/v1beta",
                "/v1beta/models/gemini:generateContent"
            ),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent"
        );
    }

    #[test]
    fn test_build_target_url_collapses_anthropic_namespace() {
        assert_eq!(
            build_target_url("https://api.minimax.io/anthropic", "/anthropic/v1/messages"),
            "https://api.minimax.io/anthropic/v1/messages"
        );
    }

    #[test]
    fn test_build_target_url_collapses_openai_v1_namespace() {
        assert_eq!(
            build_target_url("https://gateway.example/openai/v1", "/v1/chat/completions"),
            "https://gateway.example/openai/v1/chat/completions"
        );
    }

    #[test]
    fn test_build_target_url_collapses_multi_segment_overlap() {
        assert_eq!(
            build_target_url(
                "https://api.example.com/anthropic/v1",
                "/anthropic/v1/messages"
            ),
            "https://api.example.com/anthropic/v1/messages"
        );
    }

    #[test]
    fn test_build_target_url_no_overlap_keeps_path() {
        assert_eq!(
            build_target_url("https://api.example.com", "/v1/messages"),
            "https://api.example.com/v1/messages"
        );
    }

    #[test]
    fn test_build_target_url_preserves_disjoint_segments() {
        assert_eq!(
            build_target_url("https://api.example.com/foo", "/v1/messages"),
            "https://api.example.com/foo/v1/messages"
        );
    }

    #[test]
    fn test_reason_phrase_uncommon_status_codes() {
        assert_eq!(reason_phrase(201), "Created");
        assert_eq!(reason_phrase(204), "No Content");
        assert_eq!(reason_phrase(301), "Moved Permanently");
        assert_eq!(reason_phrase(302), "Found");
        assert_eq!(reason_phrase(304), "Not Modified");
        assert_eq!(reason_phrase(405), "Method Not Allowed");
        assert_eq!(reason_phrase(408), "Request Timeout");
        assert_eq!(reason_phrase(413), "Payload Too Large");
        assert_eq!(reason_phrase(429), "Too Many Requests");
        assert_eq!(reason_phrase(502), "Bad Gateway");
        assert_eq!(reason_phrase(503), "Service Unavailable");
        assert_eq!(reason_phrase(504), "Gateway Timeout");
    }

    #[test]
    fn test_reason_phrase_unknown_ranges() {
        assert_eq!(reason_phrase(299), "OK");
        assert_eq!(reason_phrase(399), "Redirect");
        assert_eq!(reason_phrase(499), "Client Error");
        assert_eq!(reason_phrase(599), "Server Error");
    }

    #[test]
    fn test_http_error_response_json_structure() {
        let resp = http_error_response(422, "Validation failed");
        assert!(resp.contains("422"));
        assert!(resp.contains("Validation failed"));
        assert!(resp.contains("application/json"));
    }

    #[test]
    fn test_http_request_read_error_response_header_too_large() {
        let resp = http_request_read_error_response(&RequestReadError::HeaderTooLarge);
        assert!(resp.starts_with("HTTP/1.1 413"));
    }

    #[test]
    fn test_http_request_read_error_response_unsupported_encoding() {
        let resp = http_request_read_error_response(&RequestReadError::UnsupportedTransferEncoding);
        assert!(resp.starts_with("HTTP/1.1 400"));
        assert!(resp.contains("chunked"));
    }

    #[test]
    fn test_http_request_read_error_response_incomplete_headers() {
        let resp = http_request_read_error_response(&RequestReadError::IncompleteHeaders);
        assert!(resp.starts_with("HTTP/1.1 400"));
    }

    #[test]
    fn test_parse_content_length_invalid_value() {
        let headers = "POST /v1 HTTP/1.1\r\nContent-Length: not_a_number\r\n";
        assert_eq!(parse_content_length(headers), None);
    }

    #[test]
    fn test_sse_data_payload_with_space() {
        assert_eq!(
            sse_data_payload("data: {\"ok\":true}"),
            Some("{\"ok\":true}")
        );
    }

    #[test]
    fn test_sse_data_payload_without_space() {
        assert_eq!(
            sse_data_payload("data:{\"ok\":true}"),
            Some("{\"ok\":true}")
        );
    }

    #[test]
    fn test_sse_data_payload_non_data_line() {
        assert_eq!(sse_data_payload("event: message"), None);
        assert_eq!(sse_data_payload(""), None);
    }

    #[test]
    fn test_parse_token_u64_number() {
        assert_eq!(parse_token_u64(&json!(42)), Some(42));
    }

    #[test]
    fn test_parse_token_u64_string() {
        assert_eq!(parse_token_u64(&json!("100")), Some(100));
    }

    #[test]
    fn test_parse_token_u64_invalid_string() {
        assert_eq!(parse_token_u64(&json!("not_a_number")), None);
    }

    #[test]
    fn test_parse_token_u64_null() {
        assert_eq!(parse_token_u64(&json!(null)), None);
    }

    #[test]
    fn test_extract_request_path_single_word() {
        assert_eq!(extract_request_path("GET"), "/");
    }

    #[test]
    fn test_is_post_path_empty_paths() {
        let req = "POST /v1/messages HTTP/1.1\r\n";
        assert!(!is_post_path(req, &[]));
    }

    #[test]
    fn test_extract_passthrough_headers_no_custom_headers() {
        let req = "POST /v1 HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer token\r\n\r\n{}";
        let headers = extract_passthrough_headers(req).unwrap();
        assert!(headers.is_empty());
    }

    #[test]
    fn test_copilot_initiator_from_anthropic_user_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        assert_eq!(copilot_initiator_from_anthropic(&body), "user");
    }

    #[test]
    fn test_copilot_initiator_from_anthropic_tool_result() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file contents"}]}
            ]
        });
        assert_eq!(copilot_initiator_from_anthropic(&body), "agent");
    }

    #[test]
    fn test_copilot_initiator_from_anthropic_empty_messages() {
        let body = json!({"messages": []});
        assert_eq!(copilot_initiator_from_anthropic(&body), "user");
    }

    #[test]
    fn test_copilot_initiator_from_anthropic_assistant_last() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        });
        assert_eq!(copilot_initiator_from_anthropic(&body), "agent");
    }

    #[test]
    fn test_copilot_initiator_from_openai_user_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        assert_eq!(copilot_initiator_from_openai(&body), "user");
    }

    #[test]
    fn test_copilot_initiator_from_openai_tool_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "result"}
            ]
        });
        assert_eq!(copilot_initiator_from_openai(&body), "agent");
    }

    #[test]
    fn test_copilot_initiator_from_openai_assistant_message() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        });
        assert_eq!(copilot_initiator_from_openai(&body), "agent");
    }

    #[test]
    fn copilot_path_strips_v1_prefix() {
        assert_eq!(
            copilot_path_from_target("/v1/chat/completions"),
            "/chat/completions"
        );
        assert_eq!(copilot_path_from_target("/v1/responses"), "/responses");
    }

    #[test]
    fn copilot_path_preserves_path_without_v1() {
        assert_eq!(
            copilot_path_from_target("/chat/completions"),
            "/chat/completions"
        );
        assert_eq!(copilot_path_from_target("/responses"), "/responses");
    }

    #[test]
    fn copilot_path_extracts_from_full_url() {
        assert_eq!(
            copilot_path_from_target("https://api.example.com/v1/chat/completions"),
            "/chat/completions"
        );
        assert_eq!(
            copilot_path_from_target("https://api.example.com/v1/responses"),
            "/responses"
        );
    }

    #[test]
    fn copilot_path_fallback_for_unexpected_input() {
        assert_eq!(copilot_path_from_target("not-a-url"), "/chat/completions");
    }

    #[test]
    fn test_strip_beta_headers_removes_header() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "anthropic-beta",
            HeaderValue::from_static("prompt-caching-2024-07-31"),
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        strip_beta_headers(&mut headers);
        assert!(headers.get("anthropic-beta").is_none());
        assert!(headers.get("anthropic-version").is_some());
    }

    #[test]
    fn test_strip_beta_headers_noop_without_header() {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        strip_beta_headers(&mut headers);
        assert!(headers.get("anthropic-version").is_some());
    }

    #[test]
    fn test_is_beta_header_rejection_invalid_beta() {
        assert!(is_beta_header_rejection(
            r#"{"error":"Invalid beta: prompt-caching-2024-07-31"}"#
        ));
    }

    #[test]
    fn test_is_beta_header_rejection_anthropic_beta_mentioned() {
        assert!(is_beta_header_rejection(
            r#"{"error":"Unknown header: anthropic-beta is not supported"}"#
        ));
    }

    #[test]
    fn test_is_beta_header_rejection_unexpected_value() {
        assert!(is_beta_header_rejection(
            r#"{"error":"unexpected value in beta field"}"#
        ));
    }

    #[test]
    fn test_is_beta_header_rejection_false_for_unrelated_error() {
        assert!(!is_beta_header_rejection(r#"{"error":"model not found"}"#));
        assert!(!is_beta_header_rejection(
            r#"{"error":"rate limit exceeded"}"#
        ));
    }
}
