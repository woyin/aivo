//! Headless one-shot agent: `aivo code -e "<task>"` runs the real `AgentEngine`
//! (tools + multi-step loop) to completion and exits. Auto-approves the confirm
//! tier; remote side effects fail closed without `--auto-approve`; catastrophic
//! commands always fail closed.
//!
//! Output is `--output-format`-selected ([`OutputFormat`]):
//! - `text` (default): answer → stdout, tool/step activity → stderr (human prose).
//! - `json`: activity → stderr, one final secret-redacted result document on stdout.
//! - `stream-json`: one secret-redacted JSON event per line on stdout for
//!   editors/automation — each carries `{schemaVersion, type, runId}` plus type-specific
//!   fields (see the `stream_event` call sites for the per-type payloads).
//!
//! Completed runs persist a session (display messages + exact engine transcript), so
//! `--resume last|<id>` continues one headlessly and the TUI's `/resume` can pick it up.

use std::io::Write;
use std::path::Path;

use futures::future::BoxFuture;
use serde_json::{Value, json};

use crate::agent::engine::{AgentEngine, AgentUi, TurnCtx};
use crate::agent::protocol::Decision;
use crate::agent::system_prompt::discover_project_guides;
use crate::errors::ExitCode;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, SessionStore};

/// Whether `key` can drive the in-process agent (not OAuth/copilot/cursor).
pub(crate) fn key_is_agent_capable(key: &ApiKey) -> bool {
    !key.is_any_oauth() && !key.is_cursor_acp() && !key.is_copilot()
}

/// Unattended `-e` backstops (env-overridable, 0 disables) — the TUI relies on esc instead.
const DEFAULT_MAX_STEPS: u32 = 1000;
const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 300_000;

fn env_or<T: std::str::FromStr>(var: &str, default: T) -> T {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn cli_env_or<T: Copy + std::str::FromStr>(cli: Option<T>, var: &str, default: T) -> T {
    cli.unwrap_or_else(|| env_or(var, default))
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OneShotAgentLimits {
    pub(crate) max_steps: Option<u32>,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) max_cost: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_one_shot_agent(
    session_store: &SessionStore,
    cache: &ModelsCache,
    key: &ApiKey,
    model: &str,
    prompt: String,
    injected_context: Option<String>,
    context_window_override: Option<u64>,
    format: OutputFormat,
    limits: OneShotAgentLimits,
    auto_approve: bool,
    resume: Option<String>,
    model_explicit: bool,
) -> anyhow::Result<ExitCode> {
    // Real launch dir (like the TUI's real_cwd), not chat's sandbox.
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".to_string());

    // Resolve resume before building the engine: a resumed session's model wins
    // over the key default (like the TUI's /resume) unless `--model` was explicit.
    let resumed = match resume.as_deref() {
        Some(sel) => Some(resolve_resume_session(session_store, sel, &cwd, &key.id).await?),
        None => None,
    };
    let resumed_model = resumed
        .as_ref()
        .map(|s| s.model.clone())
        .filter(|m| !m.is_empty());
    let effective_model = match resumed_model {
        Some(m) if !model_explicit => m,
        _ => model.to_string(),
    };
    let model: &str = &effective_model;

    let context_window = match context_window_override {
        Some(w) => w,
        None => crate::services::model_metadata::resolve_limits(cache, Some(&key.base_url), model)
            .await
            .context
            .unwrap_or(0),
    }
    .min(u32::MAX as u64) as u32;

    let date = chrono::Local::now().format("%Y-%m-%d").to_string();
    let guides = discover_project_guides(Path::new(&cwd));
    let skills = crate::agent::skills::discover_skills(Path::new(&cwd));
    let max_steps = cli_env_or(limits.max_steps, "AIVO_AGENT_MAX_STEPS", DEFAULT_MAX_STEPS);
    let mut engine = AgentEngine::new(
        &cwd,
        model,
        &date,
        &guides,
        &skills,
        context_window,
        max_steps,
    );
    engine.set_output_budget(cli_env_or(
        limits.max_output_tokens,
        "AIVO_AGENT_MAX_OUTPUT_TOKENS",
        DEFAULT_MAX_OUTPUT_TOKENS,
    ));
    // A cost estimate needs both input and output prices; fail closed otherwise.
    if let Some(usd) = limits.max_cost.filter(|c| *c > 0.0) {
        let pricing = crate::services::model_metadata::model_pricing(model)
            .filter(|p| p.input.is_some() && p.output.is_some())
            .ok_or_else(|| {
                anyhow::anyhow!("--max-cost: no input/output pricing known for model '{model}'")
            })?;
        engine.set_cost_budget(usd, pricing);
    }
    // Unattended run: don't accept an answer that admits it isn't done — nudge to continue.
    engine.set_require_completion();
    // Self-verify: default on but mutation-gated (investigate-only stays fast);
    // explicit `=1` also verifies the unknown starting baseline; `=0` opts out.
    match crate::services::system_env::env_flag("AIVO_AGENT_SELF_CORRECT") {
        Some(true) => engine.set_self_correct(true),
        None => {
            engine.set_self_correct(true);
            engine.set_verified_baseline();
        }
        Some(false) => {}
    }
    if crate::services::provider_profile::is_aivo_starter_base(&key.base_url) {
        engine.set_first_party();
    }
    if let Some(ctx) = injected_context.as_deref() {
        engine.append_system_context(ctx);
    }
    let subagents =
        crate::agent::subagents::discover_subagents(Path::new(&cwd), session_store.config_dir());
    engine.set_subagents(&subagents);
    // Persistent grant store: remembered "always allow"s survive across runs.
    engine.set_grants_path(session_store.config_dir());
    // Background-job table (temp logs — one-shots have no session dir); killed at run end.
    let jobs = crate::agent::jobs::JobTable::new(None);
    engine.set_jobs(jobs.clone());
    // Durable sub-agent reports, same temp-dir arrangement: without this, a long
    // headless run's delegated work gets stubbed away by in-run compaction.
    engine.set_artifacts_dir(
        std::env::temp_dir().join(format!("aivo-artifacts-{}", std::process::id())),
    );
    // LSP diagnostics-after-edit (default on; AIVO_AGENT_LSP=0 opts out).
    engine.maybe_enable_lsp(Path::new(&cwd));
    // User lifecycle hooks (~/.config/aivo/hooks.json).
    engine.set_hooks(std::sync::Arc::new(
        crate::agent::hooks::HookSet::load_default(),
    ));

    // Resume: best fidelity first (exact engine log, else display text). The
    // session was resolved up front (for model restore); replay it into the engine.
    if let Some(state) = &resumed {
        match &state.engine_messages {
            Some(msgs) if !msgs.is_empty() => engine.restore_conversation(msgs.clone()),
            _ => {
                let seed: Vec<serde_json::Value> = state
                    .messages
                    .iter()
                    .filter(|m| m.role == "user" || m.role == "assistant")
                    .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                    .collect();
                engine.restore_conversation(seed);
            }
        }
    }
    let session_id = resumed
        .as_ref()
        .map(|s| s.session_id.clone())
        .unwrap_or_else(crate::commands::code::new_code_session_id);

    // Eval/CI hook: AIVO_AGENT_FAKE_SSE=<script> swaps the provider for a scripted
    // loopback model, so the real loop + real tool execution run deterministically.
    let (base, auth_opt, router_cleanup) = if let Ok(script) = std::env::var("AIVO_AGENT_FAKE_SSE")
    {
        let bodies =
            crate::services::fake_model::load_script(&script).map_err(|e| anyhow::anyhow!(e))?;
        let port = crate::services::fake_model::start(bodies)?;
        (format!("http://127.0.0.1:{port}"), None, None)
    } else {
        use crate::services::serve_router::{ServeRouter, ServeRouterConfig, random_auth_token};
        let auth = random_auth_token();
        let config = ServeRouterConfig::from_key(
            key,
            false,
            300,
            Some(auth.clone()),
            std::collections::HashMap::new(),
        );
        let router = ServeRouter::new(config, key.clone(), session_store.logs())
            .with_usage_accounting(session_store.clone(), "code".to_string())
            .quiet(true);
        let (handle, shutdown, port) = router.start_background_with_addr("127.0.0.1", 0).await?;
        (
            format!("http://127.0.0.1:{port}"),
            Some(auth),
            Some((handle, shutdown)),
        )
    };

    // Loopback-only: bypass any env proxy, which can't reach the serve port (hangs).
    let client = crate::services::http_utils::router_http_client_loopback();
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: auth_opt.as_deref(),
        cwd: Path::new(&cwd),
        yes: true,
        auto_approve_all: auto_approve,
        auto_approve: None,
        review_edits: None,
    };
    let mut ui = HeadlessAgentUi::new(format, session_id.clone());
    ui.run_start(model, &cwd);
    if let Some(warn) = crate::agent::sandbox::confinement_notice() {
        ui.notify(warn);
    }
    let prompt_for_log = prompt.clone();
    let started = std::time::Instant::now();
    let completed = tokio::select! {
        _ = engine.run_turn(&ctx, &mut ui, prompt) => true,
        _ = tokio::signal::ctrl_c() => {
            eprintln!();
            false
        }
    };
    // Unattended run: never leave a background job running past exit; drop its temp logs.
    let _ = jobs.kill_all().await;
    let _ = tokio::fs::remove_dir_all(jobs.logs_root()).await;
    let exit = if !completed {
        ExitCode::ToolExit(130)
    } else {
        match &ui.last_error {
            Some(msg) => classify_agent_error(msg),
            None => ExitCode::Success,
        }
    };
    // Always close the stream so a machine consumer sees a terminal event (an error
    // event is followed by run_end, never left dangling).
    ui.run_end(i64::from(exit.code()));

    // Persist the session so `--resume` can continue it; an interrupted run saves
    // nothing (its announced sessionId simply never materializes).
    if completed {
        let usage = engine.take_turn_usage();
        persist_oneshot_session(
            session_store,
            key,
            model,
            &cwd,
            &session_id,
            resumed.map(|s| s.messages).unwrap_or_default(),
            &prompt_for_log,
            &ui.answer,
            &usage,
        )
        .await;
        let _ = session_store
            .save_agent_messages(&session_id, &engine.export_conversation())
            .await;
        if format == OutputFormat::Text {
            eprintln!("[session {session_id} — continue with --resume {session_id}]");
        }
        log_oneshot_turn(
            session_store,
            key,
            model,
            &cwd,
            &prompt_for_log,
            &ui.answer,
            &usage,
            exit,
            started.elapsed(),
        )
        .await;
    }

    if let Some((handle, shutdown)) = router_cleanup {
        shutdown.notify_one();
        handle.abort();
    }
    Ok(exit)
}

/// Error text → exit code: serve_client's `upstream <status>:` prefix gives the status;
/// a transport failure (no status) falls back to keywords.
fn classify_agent_error(msg: &str) -> ExitCode {
    if let Some(status) = parse_upstream_status(msg) {
        return match status {
            401 | 403 => ExitCode::AuthError,
            408 | 429 | 500..=599 => ExitCode::NetworkError,
            _ => ExitCode::UserError,
        };
    }
    let m = msg.to_ascii_lowercase();
    if [
        "request failed",
        "stream error",
        "connection",
        "timeout",
        "timed out",
        "network",
        "dns",
    ]
    .iter()
    .any(|k| m.contains(k))
    {
        ExitCode::NetworkError
    } else {
        ExitCode::UserError
    }
}

/// Extract `NNN` from a `… upstream NNN: …` message (serve_client's HTTP-error format).
fn parse_upstream_status(msg: &str) -> Option<u16> {
    let rest = msg.split("upstream ").nth(1)?;
    rest.chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// `last` = cwd+key-scoped (matching the TUI); an explicit id resolves globally —
/// a named session is user intent regardless of where it ran. Unknown → hard error.
async fn resolve_resume_session(
    session_store: &SessionStore,
    selector: &str,
    cwd: &str,
    key_id: &str,
) -> anyhow::Result<crate::services::session_store::CodeSessionState> {
    // Bare `--resume` parses to "" — headless has no picker, so it means `last`.
    let id = if selector.is_empty() || selector == "last" {
        session_store
            .list_chat_sessions(key_id, "", cwd)
            .await?
            .first()
            .map(|e| e.session_id.clone())
            .ok_or_else(|| anyhow::anyhow!("no saved session in this directory to resume"))?
    } else {
        selector.to_string()
    };
    session_store
        .get_code_session(&id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("no saved session '{id}' — see `aivo code` → /resume"))
}

/// Best-effort — a failed save must not fail the run whose answer already printed.
#[allow(clippy::too_many_arguments)]
async fn persist_oneshot_session(
    session_store: &SessionStore,
    key: &ApiKey,
    model: &str,
    cwd: &str,
    session_id: &str,
    mut messages: Vec<crate::services::session_store::StoredChatMessage>,
    prompt: &str,
    answer: &str,
    usage: &crate::services::session_store::SessionTokens,
) {
    use crate::services::session_store::StoredChatMessage;
    let now = chrono::Utc::now().to_rfc3339();
    messages.push(StoredChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
        reasoning_content: None,
        id: None,
        timestamp: Some(now.clone()),
        attachments: None,
        model: None,
    });
    messages.push(StoredChatMessage {
        role: "assistant".to_string(),
        content: answer.to_string(),
        reasoning_content: None,
        id: None,
        timestamp: Some(now),
        attachments: None,
        model: Some(model.to_string()),
    });
    let title: String = messages
        .iter()
        .find(|m| m.role == "user")
        .map(|m| {
            m.content
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(120)
                .collect()
        })
        .unwrap_or_default();
    let preview: String = answer
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .chars()
        .take(160)
        .collect();
    let mut tokens = session_store.chat_session_tokens(session_id).await;
    tokens.prompt_tokens += usage.prompt_tokens;
    tokens.completion_tokens += usage.completion_tokens;
    tokens.cache_read_tokens += usage.cache_read_tokens;
    tokens.cache_write_tokens += usage.cache_write_tokens;
    let _ = session_store
        .save_code_session_with_id(
            &key.id,
            &key.base_url,
            cwd,
            session_id,
            model,
            Some(model),
            &messages,
            &title,
            &preview,
            tokens,
        )
        .await;
}

/// Log a completed headless turn (`chat_turn`) so `-e` shows in `aivo logs`. Best-effort.
#[allow(clippy::too_many_arguments)]
async fn log_oneshot_turn(
    session_store: &SessionStore,
    key: &ApiKey,
    model: &str,
    cwd: &str,
    prompt: &str,
    answer: &str,
    usage: &crate::services::session_store::SessionTokens,
    exit: ExitCode,
    elapsed: std::time::Duration,
) {
    let title: String = prompt
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(120)
        .collect();
    let _ = session_store
        .logs()
        .append(crate::services::log_store::LogEvent {
            source: "code".to_string(),
            kind: "code_turn".to_string(),
            key_id: Some(key.id.clone()),
            key_name: Some(key.display_name().to_string()),
            base_url: Some(key.base_url.clone()),
            tool: Some("code".to_string()),
            model: Some(model.to_string()),
            cwd: Some(cwd.to_string()),
            exit_code: Some(i64::from(exit.code())),
            duration_ms: Some(elapsed.as_millis() as i64),
            input_tokens: Some(usage.prompt_tokens as i64),
            output_tokens: Some(usage.completion_tokens as i64),
            cache_read_input_tokens: Some(usage.cache_read_tokens as i64),
            cache_creation_input_tokens: Some(usage.cache_write_tokens as i64),
            title: Some(title),
            body_text: Some(format!("User:\n{prompt}\n\nAssistant:\n{answer}")),
            ..Default::default()
        })
        .await;
}

/// Headless output format for `-e`. `Text` = human prose (answer → stdout, activity
/// → stderr). `Json` = activity → stderr, one final result document on stdout.
/// `StreamJson` = one schema-versioned JSON event per line on stdout,
/// secret-redacted — a stable protocol for editors/automation driving the agent.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Text,
    Json,
    StreamJson,
}

impl OutputFormat {
    /// Parse the `--output-format` value (clap already limits it to the known set).
    pub(crate) fn parse(s: Option<&str>) -> Self {
        match s {
            Some("stream-json") => Self::StreamJson,
            Some("json") => Self::Json,
            _ => Self::Text,
        }
    }
}

/// Bumped on any incompatible change to the event shape below, so a consumer can
/// reject a protocol it doesn't understand.
const STREAM_JSON_SCHEMA_VERSION: u32 = 1;

/// Build one protocol event object: the common envelope (`schemaVersion`, `type`,
/// `runId`) merged with the event-specific `fields`.
fn stream_event(run_id: &str, ev_type: &str, fields: Value) -> Value {
    let mut obj = serde_json::Map::new();
    obj.insert("schemaVersion".into(), json!(STREAM_JSON_SCHEMA_VERSION));
    obj.insert("type".into(), json!(ev_type));
    obj.insert("runId".into(), json!(run_id));
    if let Value::Object(m) = fields {
        obj.extend(m);
    }
    Value::Object(obj)
}

/// A per-run id so a consumer can correlate this run's lines (one run per `-e`).
fn new_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("run_{nanos:x}_{:x}", std::process::id())
}

/// Redact secret-shaped substrings from a model/tool string before it goes on the wire.
fn redact(s: &str) -> String {
    crate::agent::secrets_guard::redact_for_model(s).0
}

/// Redact secrets inside tool args while keeping them structured; falls back to the
/// redacted string if the redaction breaks JSON (it normally doesn't).
fn redact_args(args: &Value) -> Value {
    let (red, n) = crate::agent::secrets_guard::redact_for_model(&args.to_string());
    if n == 0 {
        return args.clone();
    }
    serde_json::from_str(&red).unwrap_or(Value::String(red))
}

/// Text mode: answer → stdout (buffered per step so a stripped tool-call-as-text can't
/// leak), tool/step activity → stderr. Stream-json mode: every callback is a redacted
/// JSON event line on stdout.
struct HeadlessAgentUi {
    format: OutputFormat,
    run_id: String,
    /// Persisted-session id, fixed up front so machine consumers can `--resume` it.
    session_id: String,
    model: String,
    cwd: String,
    seg: String,
    wrote_answer: bool,
    /// Full answer text (flushed segments), kept for the `aivo logs` row and the
    /// stream-json `final` event.
    answer: String,
    /// The last terminal error the engine reported, for exit-code classification.
    last_error: Option<String>,
    /// Footer stats, kept for the single-document `json` result.
    stats: (usize, u64, u64),
}

impl HeadlessAgentUi {
    fn new(format: OutputFormat, session_id: String) -> Self {
        Self {
            format,
            run_id: new_run_id(),
            session_id,
            model: String::new(),
            cwd: String::new(),
            seg: String::new(),
            wrote_answer: false,
            answer: String::new(),
            last_error: None,
            stats: (0, 0, 0),
        }
    }

    /// Write one JSON event line to stdout (stream-json mode only).
    fn emit(&self, ev_type: &str, fields: Value) {
        println!("{}", stream_event(&self.run_id, ev_type, fields));
        let _ = std::io::stdout().flush();
    }

    /// First line of the stream: the run's identity. No-op in text/json mode.
    fn run_start(&mut self, model: &str, cwd: &str) {
        self.model = model.to_string();
        self.cwd = cwd.to_string();
        if self.format == OutputFormat::StreamJson {
            self.emit(
                "run_start",
                json!({ "model": model, "cwd": cwd, "sessionId": self.session_id }),
            );
        }
    }

    /// Terminal output: stream-json emits `run_end` (always, even after an error, so a
    /// machine consumer sees a terminal event); `json` prints its single result document.
    fn run_end(&self, exit_code: i64) {
        match self.format {
            OutputFormat::StreamJson => self.emit("run_end", json!({ "exit": exit_code })),
            OutputFormat::Json => {
                let (steps, tokens, elapsed_secs) = self.stats;
                let doc = stream_event(
                    &self.run_id,
                    "result",
                    json!({
                        "sessionId": self.session_id,
                        "model": self.model,
                        "cwd": self.cwd,
                        "exit": exit_code,
                        "answer": redact(&self.answer),
                        "error": self.last_error.as_deref().map(redact),
                        "steps": steps,
                        "tokens": tokens,
                        "elapsedSecs": elapsed_secs,
                    }),
                );
                println!("{doc}");
                let _ = std::io::stdout().flush();
            }
            OutputFormat::Text => {}
        }
    }

    fn flush_seg(&mut self) {
        if self.seg.is_empty() {
            return;
        }
        match self.format {
            OutputFormat::Text => {
                print!("{}", self.seg);
                let _ = std::io::stdout().flush();
            }
            // Json: stdout is reserved for the final result document.
            OutputFormat::Json => {}
            OutputFormat::StreamJson => {
                self.emit("text", json!({ "text": redact(&self.seg) }));
            }
        }
        self.wrote_answer = true;
        self.answer.push_str(&self.seg);
        self.seg.clear();
    }
}

impl AgentUi for HeadlessAgentUi {
    fn turn_start(&mut self) {
        self.flush_seg();
    }
    fn assistant_text(&mut self, delta: &str) {
        self.seg.push_str(delta);
    }
    fn discard_streamed_segment(&mut self) {
        self.seg.clear();
    }
    fn tool_start(&mut self, name: &str, args: &Value) {
        self.flush_seg();
        match self.format {
            OutputFormat::Text | OutputFormat::Json => {
                eprintln!("⏺ {name} {}", one_line(&args.to_string()));
            }
            OutputFormat::StreamJson => {
                self.emit(
                    "tool_call",
                    json!({ "tool": name, "args": redact_args(args) }),
                );
            }
        }
    }
    fn tool_result(&mut self, name: &str, result: &Result<String, String>) {
        match self.format {
            OutputFormat::Text | OutputFormat::Json => match result {
                Ok(s) => eprintln!("  ⎿ {}", one_line(s)),
                Err(e) => eprintln!("  ✗ {}", one_line(e)),
            },
            OutputFormat::StreamJson => {
                let ev = match result {
                    Ok(s) => json!({ "tool": name, "ok": true, "output": redact(s) }),
                    Err(e) => json!({ "tool": name, "ok": false, "error": redact(e) }),
                };
                self.emit("tool_result", ev);
            }
        }
    }
    fn notify(&mut self, text: &str) {
        match self.format {
            OutputFormat::Text | OutputFormat::Json => eprintln!("{text}"),
            OutputFormat::StreamJson => self.emit("notice", json!({ "text": redact(text) })),
        }
    }
    fn notify_error(&mut self, text: &str) {
        self.last_error = Some(text.to_string());
        match self.format {
            OutputFormat::Text | OutputFormat::Json => eprintln!("{text}"),
            OutputFormat::StreamJson => self.emit("error", json!({ "text": redact(text) })),
        }
    }
    fn footer(
        &mut self,
        _summary: Option<&str>,
        steps: usize,
        tokens: u64,
        _context_tokens: u64,
        elapsed_secs: u64,
    ) {
        self.flush_seg();
        self.stats = (steps, tokens, elapsed_secs);
        match self.format {
            OutputFormat::Text => {
                if self.wrote_answer {
                    println!();
                }
                eprintln!("[{steps} step(s) · {tokens} tok · {elapsed_secs}s]");
            }
            OutputFormat::Json => {
                eprintln!("[{steps} step(s) · {tokens} tok · {elapsed_secs}s]");
            }
            OutputFormat::StreamJson => {
                self.emit(
                    "usage",
                    json!({ "steps": steps, "tokens": tokens, "elapsedSecs": elapsed_secs }),
                );
                self.emit(
                    "final",
                    json!({ "text": redact(&self.answer), "sessionId": self.session_id }),
                );
            }
        }
    }
    fn ask_permission<'a>(
        &'a mut self,
        _tool: &'a str,
        _preview: Option<&'a str>,
    ) -> BoxFuture<'a, Decision> {
        // Only catastrophic commands and remote side effects reach here (ctx.yes
        // auto-approves the rest); no human to confirm, so fail closed.
        Box::pin(async move { Decision::Deny })
    }
}

fn one_line(s: &str) -> String {
    let line = s.lines().next().unwrap_or("");
    const MAX: usize = 160;
    if line.chars().count() > MAX {
        line.chars().take(MAX).collect::<String>() + "…"
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_terminal_error_to_the_exit_code_contract() {
        assert_eq!(
            classify_agent_error("LLM error: upstream 401: invalid api key"),
            ExitCode::AuthError
        );
        assert_eq!(
            classify_agent_error("LLM error: upstream 403: forbidden"),
            ExitCode::AuthError
        );
        assert_eq!(
            classify_agent_error("LLM error: upstream 503: overloaded"),
            ExitCode::NetworkError
        );
        assert_eq!(
            classify_agent_error("LLM error: upstream 429: slow down"),
            ExitCode::NetworkError
        );
        assert_eq!(
            classify_agent_error("LLM error: upstream 400: bad request"),
            ExitCode::UserError
        );
        assert_eq!(
            classify_agent_error("LLM error: request failed: connection refused"),
            ExitCode::NetworkError
        );
    }

    #[test]
    fn parse_upstream_status_extracts_code() {
        assert_eq!(parse_upstream_status("x upstream 429: y"), Some(429));
        assert_eq!(parse_upstream_status("no status here"), None);
    }

    #[test]
    fn output_format_parses_known_values() {
        assert!(matches!(
            OutputFormat::parse(Some("stream-json")),
            OutputFormat::StreamJson
        ));
        assert!(matches!(
            OutputFormat::parse(Some("text")),
            OutputFormat::Text
        ));
        assert!(matches!(
            OutputFormat::parse(Some("json")),
            OutputFormat::Json
        ));
        assert!(matches!(OutputFormat::parse(None), OutputFormat::Text));
    }

    #[test]
    fn stream_event_carries_the_common_envelope_and_merges_fields() {
        let ev = stream_event("run_abc", "tool_call", json!({ "tool": "edit_file" }));
        assert_eq!(ev["schemaVersion"], json!(STREAM_JSON_SCHEMA_VERSION));
        assert_eq!(ev["type"], json!("tool_call"));
        assert_eq!(ev["runId"], json!("run_abc"));
        assert_eq!(ev["tool"], json!("edit_file"));
        // One object per line: serializing must not contain a newline.
        assert!(!ev.to_string().contains('\n'));
    }

    #[test]
    fn redact_args_keeps_clean_args_structured_and_scrubs_secrets() {
        // No secret → returned structurally unchanged.
        let clean = json!({ "path": "src/main.rs" });
        assert_eq!(redact_args(&clean), clean);
        // A secret-shaped value is scrubbed but the result stays valid JSON.
        let secret = json!({ "command": "export TOKEN=sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" });
        let red = redact_args(&secret);
        assert!(red.is_object() || red.is_string());
        assert!(!red.to_string().contains("sk-ant-api03-AAAA"));
    }

    #[test]
    fn cli_limit_overrides_env_and_default() {
        let var = "AIVO_TEST_AGENT_LIMIT_CLI_WINS";
        // SAFETY: this test uses a unique env var name, so it does not race with
        // production env reads or other tests.
        unsafe { std::env::set_var(var, "7") };
        assert_eq!(cli_env_or(Some(42_u32), var, 1000_u32), 42);
        // SAFETY: see set_var note above.
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn env_limit_overrides_default() {
        let var = "AIVO_TEST_AGENT_LIMIT_ENV_WINS";
        // SAFETY: this test uses a unique env var name, so it does not race with
        // production env reads or other tests.
        unsafe { std::env::set_var(var, "7") };
        assert_eq!(cli_env_or(None::<u32>, var, 1000_u32), 7);
        // SAFETY: see set_var note above.
        unsafe { std::env::remove_var(var) };
    }

    #[test]
    fn default_limit_used_when_no_cli_or_env() {
        let var = "AIVO_TEST_AGENT_LIMIT_DEFAULT";
        // SAFETY: this test uses a unique env var name, so it does not race with
        // production env reads or other tests.
        unsafe { std::env::remove_var(var) };
        assert_eq!(cli_env_or(None::<u64>, var, 300_000_u64), 300_000);
    }

    #[test]
    fn cli_limit_preserves_zero() {
        let var = "AIVO_TEST_AGENT_LIMIT_ZERO";
        // SAFETY: this test uses a unique env var name, so it does not race with
        // production env reads or other tests.
        unsafe { std::env::set_var(var, "7") };
        assert_eq!(cli_env_or(Some(0_u64), var, 300_000_u64), 0);
        // SAFETY: see set_var note above.
        unsafe { std::env::remove_var(var) };
    }
}
