//! Headless one-shot agent: `aivo chat -e "<task>"` runs the real `AgentEngine`
//! (tools + multi-step loop) to completion and exits. Answer → stdout, tool/step
//! activity → stderr. Auto-approves mutations; catastrophic commands and remote
//! side effects (deploy/publish/DELETE) fail closed.

use std::io::Write;
use std::path::Path;

use futures::future::BoxFuture;
use serde_json::Value;

use crate::agent::engine::{AgentEngine, AgentUi, TurnCtx, discover_project_guides};
use crate::agent::protocol::Decision;
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

pub(crate) async fn run_one_shot_agent(
    session_store: &SessionStore,
    cache: &ModelsCache,
    key: &ApiKey,
    model: &str,
    prompt: String,
    context_window_override: Option<u64>,
) -> anyhow::Result<ExitCode> {
    // Real launch dir (like the TUI's real_cwd), not chat's sandbox.
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".to_string());

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
    let max_steps = env_or("AIVO_AGENT_MAX_STEPS", DEFAULT_MAX_STEPS);
    let mut engine = AgentEngine::new(
        &cwd,
        model,
        &date,
        &guides,
        &skills,
        context_window,
        max_steps,
    );
    engine.set_output_budget(env_or(
        "AIVO_AGENT_MAX_OUTPUT_TOKENS",
        DEFAULT_MAX_OUTPUT_TOKENS,
    ));
    if crate::services::provider_profile::is_aivo_starter_base(&key.base_url) {
        engine.set_first_party();
    }
    let subagents = crate::agent::subagents::discover_subagents(session_store.config_dir());
    engine.set_subagents(&subagents);

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
        .with_usage_accounting(session_store.clone(), "chat".to_string())
        .quiet(true);
    let (handle, shutdown, port) = router.start_background_with_addr("127.0.0.1", 0).await?;
    let base = format!("http://127.0.0.1:{port}");

    // Loopback-only: bypass any env proxy, which can't reach the serve port (hangs).
    let client = crate::services::http_utils::router_http_client_loopback();
    let ctx = TurnCtx {
        client: &client,
        serve_base: &base,
        auth: Some(&auth),
        cwd: Path::new(&cwd),
        yes: true,
        auto_approve: None,
    };
    let mut ui = HeadlessAgentUi::new();
    let prompt_for_log = prompt.clone();
    let started = std::time::Instant::now();
    let completed = tokio::select! {
        _ = engine.run_turn(&ctx, &mut ui, prompt) => true,
        _ = tokio::signal::ctrl_c() => {
            eprintln!();
            false
        }
    };
    let exit = if !completed {
        ExitCode::ToolExit(130)
    } else {
        match &ui.last_error {
            Some(msg) => classify_agent_error(msg),
            None => ExitCode::Success,
        }
    };

    // No session written — one-shots aren't resumable by design; just log the turn.
    if completed {
        let usage = engine.take_turn_usage();
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

    shutdown.notify_one();
    handle.abort();
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
            source: "chat".to_string(),
            kind: "chat_turn".to_string(),
            key_id: Some(key.id.clone()),
            key_name: Some(key.display_name().to_string()),
            base_url: Some(key.base_url.clone()),
            tool: Some("chat".to_string()),
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

/// Answer → stdout (buffered per step so a stripped tool-call-as-text can't leak);
/// tool/step activity → stderr.
struct HeadlessAgentUi {
    seg: String,
    wrote_answer: bool,
    /// Full answer text (flushed segments), kept for the `aivo logs` row.
    answer: String,
    /// The last terminal error the engine reported, for exit-code classification.
    last_error: Option<String>,
}

impl HeadlessAgentUi {
    fn new() -> Self {
        Self {
            seg: String::new(),
            wrote_answer: false,
            answer: String::new(),
            last_error: None,
        }
    }

    fn flush_seg(&mut self) {
        if self.seg.is_empty() {
            return;
        }
        print!("{}", self.seg);
        let _ = std::io::stdout().flush();
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
        eprintln!("⏺ {name} {}", one_line(&args.to_string()));
    }
    fn tool_result(&mut self, _name: &str, result: &Result<String, String>) {
        match result {
            Ok(s) => eprintln!("  ⎿ {}", one_line(s)),
            Err(e) => eprintln!("  ✗ {}", one_line(e)),
        }
    }
    fn notify(&mut self, text: &str) {
        eprintln!("{text}");
    }
    fn notify_error(&mut self, text: &str) {
        self.last_error = Some(text.to_string());
        eprintln!("{text}");
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
        if self.wrote_answer {
            println!();
        }
        eprintln!("[{steps} step(s) · {tokens} tok · {elapsed_secs}s]");
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
}
