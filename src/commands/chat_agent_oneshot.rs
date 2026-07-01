//! Headless one-shot agent: `aivo chat -e "<task>"` runs the real `AgentEngine`
//! (tools + multi-step loop) to completion and exits. Answer → stdout, tool/step
//! activity → stderr. Auto-approves mutations; catastrophic commands fail closed.

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
    let mut engine = AgentEngine::new(&cwd, model, &date, &guides, &skills, context_window, 0);
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
    let exit = tokio::select! {
        _ = engine.run_turn(&ctx, &mut ui, prompt) => {
            if ui.saw_error { ExitCode::UserError } else { ExitCode::Success }
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!();
            ExitCode::ToolExit(130)
        }
    };

    shutdown.notify_one();
    handle.abort();
    Ok(exit)
}

/// Answer → stdout (buffered per step so a stripped tool-call-as-text can't leak);
/// tool/step activity → stderr.
struct HeadlessAgentUi {
    seg: String,
    wrote_answer: bool,
    saw_error: bool,
}

impl HeadlessAgentUi {
    fn new() -> Self {
        Self {
            seg: String::new(),
            wrote_answer: false,
            saw_error: false,
        }
    }

    fn flush_seg(&mut self) {
        if self.seg.is_empty() {
            return;
        }
        print!("{}", self.seg);
        let _ = std::io::stdout().flush();
        self.wrote_answer = true;
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
        self.saw_error = true;
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
        // Only catastrophic commands reach here (ctx.yes auto-approves the rest); fail closed.
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
