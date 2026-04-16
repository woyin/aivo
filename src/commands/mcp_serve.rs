//! `aivo mcp-serve` — stdio MCP server exposing live cross-tool session data.
//!
//! This subcommand is launched by Claude or Codex (via `aivo run --as <name>`)
//! and speaks newline-delimited JSON-RPC 2.0 over stdin/stdout. The tool
//! surface is small and static:
//!
//! - `list_sessions(cli?)` — list recent sessions for the scoped project,
//!   optionally filtered by CLI name. Thin wrapper over `ingest_project`.
//! - `get_session(cli, session_id?, max_turns?)` — verbatim recent turns for
//!   the most-recent (or prefix-matched) session. Thin wrapper over
//!   `resolve_session` in `session_transcript`.
//!
//! Gotcha: stdout is the protocol channel. **Never** write to stdout outside
//! of the framed JSON-RPC responses. Diagnostics go to stderr.
//!
//! The server exits cleanly on stdin EOF — its lifetime is bound to the
//! parent tool's MCP-client process.

use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::cli::McpServeArgs;
use crate::errors::ExitCode;
use crate::services::ai_launcher::AIToolType;
use crate::services::context_ingest::{IngestOptions, ingest_project};
use crate::services::nickname_registry;
use crate::services::session_transcript::resolve_session;
use crate::services::system_env;

/// MCP protocol version this server implements.
const MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// Safety cap on how many turns a single `get_session` call can return.
const MAX_TURNS_CAP: usize = 50;

/// Default for `max_turns` when the caller doesn't specify.
const DEFAULT_MAX_TURNS: usize = 20;

pub struct McpServeCommand;

impl McpServeCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: McpServeArgs) -> ExitCode {
        let cwd = match resolve_cwd(args.cwd) {
            Some(p) => p,
            None => {
                eprintln!("aivo mcp-serve: unable to resolve cwd");
                return ExitCode::UserError;
            }
        };
        eprintln!("aivo mcp-serve: scoping to {}", cwd.display());

        let registry_root = nickname_registry::registry_dir_for_cwd(&cwd);

        // If a nickname was provided, register it and hold the guard for the
        // server's lifetime so the file is cleaned up on exit.
        let _guard = match (&args.nickname, &args.caller_cli, &registry_root) {
            (Some(nick), Some(cli), Some(root)) => {
                eprintln!("aivo mcp-serve: nickname={nick}, caller={cli}");
                match nickname_registry::register(nick, cli, root).await {
                    Ok(guard) => Some(guard),
                    Err(e) => {
                        eprintln!("aivo mcp-serve: failed to register nickname: {e}");
                        None
                    }
                }
            }
            (Some(nick), None, _) => {
                eprintln!("aivo mcp-serve: nickname={nick}, caller=unknown (no --caller-cli)");
                None
            }
            _ => None,
        };

        match run_stdio_loop(&cwd, registry_root.as_deref()).await {
            Ok(()) => ExitCode::Success,
            Err(e) => {
                eprintln!("aivo mcp-serve: fatal error: {e}");
                ExitCode::UserError
            }
        }
    }
}

impl Default for McpServeCommand {
    fn default() -> Self {
        Self::new()
    }
}

fn resolve_cwd(explicit: Option<PathBuf>) -> Option<PathBuf> {
    let raw = explicit.or_else(system_env::current_dir)?;
    Some(std::fs::canonicalize(&raw).unwrap_or(raw))
}

/// Read newline-delimited JSON-RPC from stdin, handle each request, and write
/// responses to stdout. Exits on EOF.
async fn run_stdio_loop(
    project_root: &std::path::Path,
    registry_root: Option<&Path>,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = handle_request(&line, project_root, registry_root).await;
        if let Some(resp) = response {
            let serialized = serde_json::to_string(&resp)?;
            stdout.write_all(serialized.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

/// Dispatch a single JSON-RPC request line and produce its response.
/// Returns `None` for notifications (no reply expected).
async fn handle_request(
    line: &str,
    project_root: &std::path::Path,
    registry_root: Option<&Path>,
) -> Option<Value> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("aivo mcp-serve: parse error on input: {e}");
            return Some(error_response(Value::Null, -32700, "Parse error"));
        }
    };

    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let is_notification = req.get("id").is_none();
    let params = req.get("params").cloned().unwrap_or(Value::Null);

    match method {
        "initialize" => Some(success_response(id, initialize_result())),
        "notifications/initialized" | "initialized" => None,
        "tools/list" => Some(success_response(id, tools_list_result())),
        "tools/call" => {
            let result = tools_call(&params, project_root, registry_root).await;
            Some(success_response(id, result))
        }
        "ping" => Some(success_response(id, json!({}))),
        "shutdown" => Some(success_response(id, Value::Null)),
        _ if is_notification => None,
        _ => Some(error_response(
            id,
            -32601,
            &format!("Method not found: {method}"),
        )),
    }
}

fn initialize_result() -> Value {
    json!({
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": "aivo",
            "version": crate::version::VERSION,
        }
    })
}

fn tools_list_result() -> Value {
    // Single source of truth for the cli enum — keyed off `AIToolType::all()`
    // so a new tool just needs a variant there and a resolver in
    // `session_transcript`.
    let cli_names: Vec<&str> = AIToolType::all().iter().map(|t| t.as_str()).collect();
    let cli_names_human = cli_names
        .iter()
        .map(|s| format!("'{s}'"))
        .collect::<Vec<_>>()
        .join(", ");

    json!({
        "tools": [
            {
                "name": "list_sessions",
                "description": "List recent AI CLI sessions for the current project, newest first. Returns {cli, session_id, topic, updated_at, source_path, nickname} for each. Call this first whenever the user references another tool's work — especially if multiple sessions for the same CLI might be running in this directory (e.g. two Claude windows + one Codex). Use the `topic` (first user message) to disambiguate which session the user means, then pass the chosen `session_id` to get_session. Note: your own session is likely among the results; its topic will mention the current conversation, so skip it. Sessions from named tools (launched with `--as`) include a `nickname` field.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "cli": {
                            "type": "string",
                            "description": format!("Filter to a specific CLI: {cli_names_human}. Omit to list all."),
                            "enum": cli_names.clone(),
                        }
                    }
                }
            },
            {
                "name": "get_session",
                "description": "Fetch a verbatim recent-turns transcript of a peer AI CLI session in the current project. Use when the user references what another tool 'just said' or 'just found' — returns the last N conversational turns so you can act on them.\n\nPreferred: pass the peer's `nickname` (e.g. `nickname=\"coder\"`) — this is unambiguous and auto-excludes your own session. Fallback: use `cli` + `session_id` or `exclude_session_ids` for unnamed sessions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "nickname": {
                            "type": "string",
                            "description": "Peer's nickname (from --as). Use instead of cli + session_id."
                        },
                        "cli": {
                            "type": "string",
                            "description": format!("Which peer CLI to read from: {cli_names_human}."),
                            "enum": cli_names,
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session id prefix. Omit to get the most-recent session."
                        },
                        "exclude_session_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Session id prefixes to skip. Pass your own session_id here for same-CLI peer queries to avoid returning your own transcript."
                        },
                        "max_turns": {
                            "type": "integer",
                            "description": "Maximum turns to return (default 20, max 50).",
                            "minimum": 1,
                            "maximum": 50
                        }
                    }
                }
            }
        ]
    })
}

async fn tools_call(
    params: &Value,
    project_root: &std::path::Path,
    registry_root: Option<&Path>,
) -> Value {
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

    match name {
        "list_sessions" => {
            match handle_list_sessions(&arguments, project_root, registry_root).await {
                Ok(v) => tool_text_result(&v),
                Err(msg) => tool_error_result(&msg),
            }
        }
        "get_session" => match handle_get_session(&arguments, project_root, registry_root).await {
            Ok(v) => tool_text_result(&v),
            Err(msg) => tool_error_result(&msg),
        },
        other => tool_error_result(&format!("Unknown tool: {other}")),
    }
}

async fn handle_list_sessions(
    args: &Value,
    project_root: &std::path::Path,
    registry_root: Option<&Path>,
) -> std::result::Result<Value, String> {
    let cli_filter = args
        .get("cli")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let threads = ingest_project(project_root, IngestOptions::default())
        .await
        .map_err(|e| format!("Failed to enumerate sessions: {e}"))?;

    // Load active nickname entries so we can annotate matching sessions.
    let registry_entries = match registry_root {
        Some(root) => nickname_registry::list_active(root).await,
        None => Vec::new(),
    };

    let items: Vec<Value> = threads
        .into_iter()
        .filter(|t| cli_filter.as_deref().is_none_or(|c| t.cli == c))
        .map(|t| {
            // A session matches a registry entry if the CLI matches and the
            // session was updated at or after the entry's started_at.
            let nickname = registry_entries
                .iter()
                .find(|e| e.cli == t.cli && t.updated_at >= e.started_at)
                .map(|e| Value::String(e.nickname.clone()));

            json!({
                "cli": t.cli,
                "session_id": t.session_id,
                "topic": t.topic,
                "updated_at": t.updated_at.to_rfc3339(),
                "source_path": t.source_path,
                "nickname": nickname.unwrap_or(Value::Null),
            })
        })
        .collect();

    Ok(json!({ "sessions": items }))
}

async fn handle_get_session(
    args: &Value,
    project_root: &std::path::Path,
    registry_root: Option<&Path>,
) -> std::result::Result<Value, String> {
    let nickname = args
        .get("nickname")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty());
    let max_turns = args
        .get("max_turns")
        .and_then(|v| v.as_u64())
        .map(|n| (n as usize).clamp(1, MAX_TURNS_CAP))
        .unwrap_or(DEFAULT_MAX_TURNS);

    // Resolve CLI and started_after from nickname, or fall back to explicit args.
    let (cli, session_id, exclude_session_ids, started_after): (
        String,
        Option<String>,
        Vec<String>,
        Option<DateTime<Utc>>,
    ) = if let Some(nick) = nickname {
        let root = registry_root
            .ok_or_else(|| "Nickname lookup unavailable (no registry root)".to_string())?;
        let entry = nickname_registry::resolve_nickname(nick, root)
            .await
            .ok_or_else(|| {
                format!("No active tool found with nickname '{nick}'. Is it still running?")
            })?;
        (entry.cli.clone(), None, Vec::new(), Some(entry.started_at))
    } else {
        let cli = args.get("cli").and_then(|v| v.as_str()).ok_or_else(|| {
            "Missing required argument: provide either 'nickname' or 'cli'".to_string()
        })?;
        if AIToolType::parse(cli).is_none() {
            return Err(format!(
                "Unsupported cli '{cli}'. Valid values: {}.",
                AIToolType::all()
                    .iter()
                    .map(|t| format!("'{}'", t.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let session_id = args
            .get("session_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let exclude: Vec<String> = args
            .get("exclude_session_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();
        (cli.to_string(), session_id, exclude, None)
    };

    let transcript = resolve_session(
        project_root,
        &cli,
        session_id.as_deref(),
        &exclude_session_ids,
        started_after,
        max_turns,
    )
    .await
    .map_err(|e| format!("Failed to load session: {e}"))?;

    match transcript {
        Some(t) => Ok(serde_json::to_value(&t).unwrap_or_else(|_| json!({}))),
        None => {
            if nickname.is_some() {
                Err(format!(
                    "No {cli} session found for nickname '{}' in this project. The tool may not have written any turns yet.",
                    nickname.unwrap_or("?")
                ))
            } else if !exclude_session_ids.is_empty() {
                Err(format!(
                    "No {cli} session found for this project after excluding {} id(s). If you're looking for a peer of the same CLI, ensure the other tool has written at least one substantive turn.",
                    exclude_session_ids.len()
                ))
            } else {
                Err(format!(
                    "No {cli} session found for this project yet. Ask the user to run aivo {cli} --as <name> in this directory."
                ))
            }
        }
    }
}

fn tool_text_result(value: &Value) -> Value {
    let text = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    json!({
        "content": [
            { "type": "text", "text": text }
        ]
    })
}

fn tool_error_result(message: &str) -> Value {
    json!({
        "isError": true,
        "content": [
            { "type": "text", "text": message }
        ]
    })
}

fn success_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn error_response(id: Value, code: i64, message: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn initialize_returns_protocol_version_and_capabilities() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], MCP_PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], "aivo");
    }

    #[tokio::test]
    async fn tools_list_returns_both_tools() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"list_sessions"));
        assert!(names.contains(&"get_session"));
    }

    #[tokio::test]
    async fn get_session_schema_exposes_exclude_session_ids() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":9,"method":"tools/list"}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        let get_session = tools
            .iter()
            .find(|t| t["name"] == "get_session")
            .expect("get_session tool");
        let props = &get_session["inputSchema"]["properties"];
        assert!(props["exclude_session_ids"].is_object());
        assert_eq!(props["exclude_session_ids"]["type"], "array");
        // Description should mention nickname as the preferred approach.
        let desc = get_session["description"].as_str().unwrap();
        assert!(
            desc.contains("nickname"),
            "get_session description should mention nickname, got: {desc}"
        );
        // Schema should expose the nickname property.
        assert!(props["nickname"].is_object());
        assert_eq!(props["nickname"]["type"], "string");
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":3,"method":"something/weird"}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn notifications_return_no_response() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let resp = handle_request(req, dir.path(), None).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn parse_error_returns_minus_32700() {
        let dir = TempDir::new().unwrap();
        let resp = handle_request("{not valid json", dir.path(), None)
            .await
            .unwrap();
        assert_eq!(resp["error"]["code"], -32700);
    }

    #[tokio::test]
    async fn get_session_without_cli_returns_tool_error() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_session","arguments":{}}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        let result = &resp["result"];
        assert_eq!(result["isError"], true);
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("cli")
        );
    }

    #[tokio::test]
    async fn get_session_with_unsupported_cli_returns_tool_error() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_session","arguments":{"cli":"gemini"}}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        assert_eq!(resp["result"]["isError"], true);
    }

    #[tokio::test]
    async fn get_session_missing_data_returns_friendly_error() {
        // No fixture files — should return isError:true with guidance, not a protocol error.
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"get_session","arguments":{"cli":"claude"}}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        assert_eq!(resp["result"]["isError"], true);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("No claude session found"));
    }

    #[tokio::test]
    async fn list_sessions_with_no_sessions_returns_empty_array() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"list_sessions","arguments":{}}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        // Normal (non-error) tool result with a content text block containing JSON.
        assert!(resp["result"]["isError"].is_null() || resp["result"]["isError"] == false);
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["sessions"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn unknown_tool_name_returns_tool_error() {
        let dir = TempDir::new().unwrap();
        let req = r#"{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"mystery","arguments":{}}}"#;
        let resp = handle_request(req, dir.path(), None).await.unwrap();
        assert_eq!(resp["result"]["isError"], true);
    }
}
