//! RunCommand handler for unified AI tool launching.
use std::collections::HashMap;

use anyhow::Result;
use reqwest::Client;

use crate::commands::code_tui_format::format_time_ago_short_dt;
use crate::commands::models::resolve_model_placeholder;
use crate::commands::{print_launch_preview, trim_to_one_line};
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::context_ingest::{IngestOptions, ingest_project_with_code};
use crate::services::context_render::{RenderedContext, render_single_session};
use crate::services::environment_injector::{ClaudeModelOverrides, ClaudeSlotFlags};
use crate::services::http_utils;
use crate::services::huggingface;
use crate::services::models_cache::ModelsCache;
use crate::services::project_id::Thread;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::system_env;
use crate::style;

use crate::commands::models::ModelOutcome;

/// RunCommand provides a unified interface to launch AI tools
pub struct RunCommand {
    session_store: SessionStore,
    ai_launcher: AILauncher,
    cache: ModelsCache,
}

impl RunCommand {
    pub fn new(session_store: SessionStore, ai_launcher: AILauncher, cache: ModelsCache) -> Self {
        Self {
            session_store,
            ai_launcher,
            cache,
        }
    }

    /// Resolves the model to use when --model flag is provided.
    /// --model <value> → use as-is. --model (no value) → show picker with
    /// the given header. No --model flag → returns `UseDefault` (let the tool
    /// use its own default). The `prompt` lets callers render per-slot
    /// headers like `"Step 2 of 3 — fast model"`.
    #[allow(clippy::too_many_arguments)]
    async fn resolve_model(
        &self,
        client: &Client,
        key: &ApiKey,
        flag_model: Option<String>,
        explicit_model_flag: bool,
        refresh: bool,
        tool: AIToolType,
        prompt: &str,
    ) -> Result<ModelOutcome> {
        crate::commands::models::resolve_model_outcome(
            client,
            key,
            flag_model,
            explicit_model_flag,
            refresh,
            Some(tool),
            &self.cache,
            prompt,
        )
        .await
    }

    /// Walks the per-slot Claude model flags. For each slot: leave unset,
    /// take the explicit value as-is, or open a sequential picker (with a
    /// `Step N of M — <slot>` header) for bare flags. ESC at any picker step
    /// aborts the launch (parity with `-m`).
    async fn resolve_claude_overrides(
        &self,
        client: &Client,
        key: &ApiKey,
        flags: ClaudeSlotFlags,
        refresh: bool,
    ) -> Result<Option<ClaudeModelOverrides>> {
        let slots = [
            ("reasoning model", flags.reasoning),
            ("subagent model", flags.subagent),
            ("haiku family model", flags.haiku),
            ("sonnet family model", flags.sonnet),
            ("opus family model", flags.opus),
        ];
        let total_pickers = slots
            .iter()
            .filter(|(_, v)| matches!(v, Some(s) if s.is_empty()))
            .count();

        let mut resolved: [Option<String>; 5] = [None, None, None, None, None];
        let mut step = 0usize;
        for (idx, (label, value)) in slots.into_iter().enumerate() {
            let prompt = if matches!(value, Some(ref s) if s.is_empty()) {
                step += 1;
                format!("Step {step} of {total_pickers} — {label}")
            } else {
                String::new()
            };
            let outcome = self
                .resolve_model(
                    client,
                    key,
                    value,
                    true,
                    refresh,
                    AIToolType::Claude,
                    &prompt,
                )
                .await?;
            match outcome {
                ModelOutcome::Cancelled => return Ok(None),
                ModelOutcome::Model(m) => resolved[idx] = Some(m),
                ModelOutcome::UseDefault => {}
            }
        }
        let [reasoning, subagent, haiku, sonnet, opus] = resolved;
        Ok(Some(ClaudeModelOverrides {
            reasoning,
            subagent,
            haiku,
            sonnet,
            opus,
            max_context: None,
        }))
    }

    /// Executes the run command with the specified AI tool
    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        explicit_model_flag: bool,
        slots: ClaudeSlotFlags,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
        context_selector: Option<String>,
        max_context: Option<String>,
    ) -> ExitCode {
        match self
            .execute_internal(
                tool,
                args,
                dry_run,
                refresh,
                model,
                explicit_model_flag,
                slots,
                env,
                key_override,
                context_selector,
                max_context,
            )
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_internal(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        explicit_model_flag: bool,
        slots: ClaudeSlotFlags,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
        context_selector: Option<String>,
        max_context: Option<String>,
    ) -> anyhow::Result<ExitCode> {
        let mut model = model;
        let tool = match tool {
            Some(t) => t,
            None => {
                Self::print_help(None);
                return Ok(ExitCode::UserError);
            }
        };

        // Handle help flags
        if tool == "--help" || tool == "-h" {
            Self::print_help(None);
            return Ok(ExitCode::Success);
        }

        // Validate tool
        let ai_tool = match AIToolType::parse(tool) {
            Some(t) => t,
            None => {
                eprintln!(
                    "{} Unknown tool '{}'. Valid tools: chat, claude, codex, codex-app, gemini, opencode, pi.",
                    style::red("Error:"),
                    tool
                );
                eprintln!("Run `aivo run --help` for details.");
                return Ok(ExitCode::UserError);
            }
        };

        // OAuth keys carry serialized tokens only the matching native CLI can
        // consume (shadow CODEX_HOME, CLAUDE_CODE_OAUTH_TOKEN); every other
        // tool would see an unusable JSON blob. Legacy `gemini-oauth` keys are
        // defunct (sign-in removed) and rejected before launch via
        // `oauth_incompat_reason`.
        let mut key_override = key_override;

        // Bare `hf:` opens a picker; rewrite to a concrete ref so the
        // match below treats it like any other `hf:<repo>` value.
        if let Some(m) = model.as_deref()
            && huggingface::is_bare_hf_picker_trigger(m)
        {
            match huggingface::pick_cached_short_ref() {
                Some(short) => model = Some(short),
                None => return Ok(ExitCode::Success),
            }
        }

        // HF takeover: bypass key resolution, spawn a local llama-server,
        // synthesize a loopback key pinned to OpenAI Chat Completions on
        // every tool's protocol (llama-server speaks nothing else).
        let hf_active = match model.as_deref() {
            Some(m) if huggingface::is_hf_or_local_gguf(m) => {
                let hf_ref = huggingface::parse_hf_ref(m)?;
                let port = if dry_run {
                    eprintln!(
                        "  {} {}",
                        style::yellow("Note:"),
                        style::dim(
                            "--dry-run shows a placeholder local port; llama-server is not spawned."
                        )
                    );
                    0
                } else {
                    huggingface::ensure_ready(&hf_ref).await?
                };
                key_override = Some(huggingface::local_takeover_key(&hf_ref, port));
                model = Some(hf_ref.display_model_name());
                true
            }
            _ => false,
        };

        if let Some(ref key) = key_override
            && ai_tool.oauth_incompat_reason(key).is_some()
        {
            let context_phrase = format!("aivo run {}", ai_tool.as_str());
            match crate::commands::keys::swap_incompatible_key(
                &self.session_store,
                key,
                crate::services::key_compat::KeyCompatContext::Tool(ai_tool),
                &context_phrase,
            )
            .await?
            {
                Some(new_key) => key_override = Some(new_key),
                None => return Ok(ExitCode::UserError),
            }
        }

        let client = http_utils::router_http_client();
        let resolved_model = if let Some(ref key) = key_override {
            let outcome = self
                .resolve_model(
                    &client,
                    key,
                    model,
                    explicit_model_flag,
                    refresh,
                    ai_tool,
                    "Select model",
                )
                .await?;
            match outcome {
                ModelOutcome::Cancelled => return Ok(ExitCode::Success),
                ModelOutcome::Model(m) => Some(m),
                ModelOutcome::UseDefault => None,
            }
        } else {
            // key_override is always resolved in main.rs before reaching here; this
            // branch is unreachable in normal operation. Bail defensively rather than
            // silently discarding the picker trigger.
            anyhow::bail!("Internal error: no active key available for model resolution");
        };

        if let Some(ref key) = key_override
            && !hf_active
        {
            let _ = self
                .session_store
                .set_last_selection(key, tool, resolved_model.as_deref())
                .await;
        }

        let mut claude_overrides = match ai_tool {
            AIToolType::Claude if slots.any_set() => {
                let key = key_override
                    .as_ref()
                    .expect("key_override is required (validated above)");
                match self
                    .resolve_claude_overrides(&client, key, slots, refresh)
                    .await?
                {
                    Some(o) => o,
                    None => return Ok(ExitCode::Success),
                }
            }
            AIToolType::Claude => ClaudeModelOverrides::default(),
            _ => {
                // Non-Claude tool: warn once per set slot flag and forget them.
                // Running pickers we'd throw away would be wasted UI.
                warn_slot_flags_ignored(ai_tool, &slots);
                ClaudeModelOverrides::default()
            }
        };

        // `--max-context` is Claude-only — it targets Anthropic's beta-tier
        // context-bar opt-in 1m/2m. Other tools (codex included) get their
        // context windows from the limits cascade; the run entry point
        // rejects the flag for them before dispatch reaches here.
        claude_overrides.max_context = match ai_tool {
            AIToolType::Claude => {
                resolve_max_context(
                    &self.cache,
                    key_override.as_ref().map(|k| k.base_url.as_str()),
                    resolved_model.as_deref(),
                    max_context,
                )
                .await
            }
            _ => None,
        };

        let launch_model = resolve_model_placeholder(resolved_model);

        // `--max-context` / `--1m` is a model-name suffix: aivo writes
        // `<model>[<tag>]` into ANTHROPIC_MODEL, Claude Code parses the suffix
        // off and adds the `anthropic-beta: context-1m-2025-08-07` header.
        // With no model resolved (user picked "(leave it to the tool)", let
        // it persist via `__default__`, or skipped `-m` entirely), there's
        // nothing to attach the suffix to — env vars are never written and
        // the flag silently no-ops. Surface this before Claude takes over the
        // screen, and gate the launch on Enter so the note isn't lost. Check
        // after `resolve_model_placeholder` so the `__default__` sentinel is
        // already collapsed to `None`.
        if matches!(ai_tool, AIToolType::Claude)
            && claude_overrides.max_context.is_some()
            && launch_model.is_none()
        {
            let tag = claude_overrides.max_context.as_deref().unwrap_or("1m");
            eprintln!();
            eprintln!(
                "{} `--{tag}` needs a model id to attach `[{tag}]` to.",
                style::yellow("Note:"),
            );
            eprintln!("  No model was selected, so Claude Code will boot with its built-in");
            eprintln!("  default and {tag} context will NOT be active. Inside the session, run");
            eprintln!("    /model <model-id>[{tag}]   (e.g. /model claude-sonnet-4-6[{tag}])");
            eprintln!("  to enable it.");
            eprintln!();
            use std::io::{IsTerminal, Write};
            if !dry_run && std::io::stderr().is_terminal() && std::io::stdin().is_terminal() {
                eprint!("Press Enter to continue, or Ctrl+C to abort... ");
                let _ = std::io::stderr().flush();
                let mut buf = String::new();
                let _ = std::io::stdin().read_line(&mut buf);
            }
        }

        // Optional context injection: inject exactly one past session.
        let args = if let Some(selector) = context_selector {
            if ai_tool == AIToolType::CodexApp {
                eprintln!(
                    "  {} --context is ignored for codex-app",
                    style::yellow("!")
                );
                args
            } else {
                maybe_inject_context(&self.session_store, ai_tool, args, &selector).await
            }
        } else {
            args
        };

        // `--transform` only changes pi's launch path. On other tools the
        // flag is meaningless — warn and clear the global so `for_pi`
        // never sees it on a subsequent call in the same process.
        if crate::services::transform_mode::is_active() && ai_tool != AIToolType::Pi {
            eprintln!(
                "  {} --transform is ignored for {}",
                style::yellow("!"),
                ai_tool.as_str(),
            );
            crate::services::transform_mode::set_active(false);
        }

        // Launch the AI tool
        let options = LaunchOptions {
            tool: ai_tool,
            args,
            model: launch_model,
            claude_overrides,
            env,
            key_override,
        };

        if dry_run {
            let plan = self.ai_launcher.prepare_launch(&options).await?;
            print_launch_preview(&plan);
            return Ok(ExitCode::Success);
        }

        let exit_code = self.ai_launcher.launch(&options).await?;
        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    /// Shows usage information. When `tool` names a specific CLI, only the
    /// flags that actually apply to that CLI are listed; bare `aivo run --help`
    /// (no tool) shows the union. Each option's visibility is gated on which
    /// tools the run pipeline actually honors:
    ///   - Claude slot flags (`--reasoning-model`, `--{haiku,sonnet,opus}-model`,
    ///     `--subagent-model`) → claude
    ///   - `--max-context`/`--1m`/`--2m` → claude
    ///   - `--relogin` → claude, codex/codex-app, gemini (the OAuth-backed keys)
    ///   - `-c, --context` → every tool (no flat prompt-flag path)
    pub fn print_help(tool: Option<&str>) {
        let generic = tool.is_none();
        let is = |name: &str| generic || tool == Some(name);

        if let Some(t) = tool {
            println!("{} aivo {} [args...]", style::bold("Usage:"), t);
        } else {
            println!("{} aivo run [tool] [args...]", style::bold("Usage:"));
        }
        println!();
        match tool {
            Some("claude") => {
                println!("{}", style::dim("Launch Claude Code with a local API key."))
            }
            Some("codex") => println!("{}", style::dim("Launch Codex with a local API key.")),
            Some("codex-app") => println!(
                "{}",
                style::dim(
                    "Launch Codex Desktop App with a local API key (experimental, macOS only)."
                )
            ),
            Some("gemini") => println!("{}", style::dim("Launch Gemini with a local API key.")),
            Some("opencode") => println!("{}", style::dim("Launch OpenCode with a local API key.")),
            Some("pi") => println!("{}", style::dim("Launch Pi with a local API key.")),
            _ => {
                println!(
                    "{}",
                    style::dim(
                        "Launch an AI coding assistant with local API keys; no tool opens an interactive picker."
                    )
                );
            }
        }
        println!(
            "{}",
            style::dim("All arguments are passed through to the underlying tool.")
        );
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<28}", flag)),
                style::dim(desc)
            );
        };
        let section = |name: &str| {
            println!();
            println!("{}", style::bold(name));
        };
        // Generic help keeps the "Claude only:" prefixes so the union view
        // stays unambiguous; per-tool help drops them since every flag listed
        // already applies to that tool.
        let label = |s: &str| -> String {
            if generic {
                s.to_string()
            } else {
                s.trim_start_matches("Claude only: ")
                    .trim_start_matches("Codex only: ")
                    .to_string()
            }
        };

        section("Model:");
        print_opt("-m, --model <model>", "Specify AI model to use");
        if is("claude") {
            print_opt(
                "--reasoning-model <m>",
                &label("Claude only: override reasoning slot (bare = picker)"),
            );
            print_opt(
                "--subagent-model <m>",
                &label("Claude only: override subagent slot (bare = picker)"),
            );
            print_opt(
                "--haiku|sonnet|opus-model",
                &label("Claude only: what `/model <name>` resolves to (bare = picker)"),
            );
        }

        section("Context:");
        if is("claude") {
            print_opt(
                "--max-context <size>",
                "Larger context window (e.g. 1m, 2m)",
            );
            print_opt("--1m", "Shorthand for --max-context=1m");
            print_opt("--2m", "Shorthand for --max-context=2m");
        } else {
            print_opt(
                "--max-context <size>",
                "Context window for unknown models (e.g. 200k)",
            );
        }
        if tool != Some("codex-app") {
            print_opt("-c, --context[=<id>]", "Inject one past session");
        }

        section("Key & Auth:");
        print_opt("-k, --key <id|name>", "Select API key by ID or name");
        if is("claude") || is("codex") || is("codex-app") {
            let relogin_desc = if generic {
                "Force OAuth re-login (codex/codex-app/claude)"
            } else {
                "Force OAuth re-login for the selected key"
            };
            print_opt("--relogin", relogin_desc);
        }

        section("Run:");
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("--env <k=v>", "Inject environment variable");
        print_opt("--dry-run", "Print the resolved command without launching");
        if is("pi") {
            print_opt(
                "--transparent",
                &label("Pi only: bypass the router (talk natively)"),
            );
        }

        if generic {
            println!();
            println!("{}", style::bold("Commands:"));
            let print_tool = |label: &str, desc: &str| {
                println!(
                    "  {}{}",
                    style::cyan(format!("{:<12}", label)),
                    style::dim(desc)
                );
            };
            for t in AIToolType::all()
                .iter()
                .filter(|t| t.supported_on_current_platform())
            {
                print_tool(t.as_str(), t.description());
            }
        }

        println!();
        println!("{}", style::bold("Examples:"));
        match tool {
            Some("claude") => {
                println!("  {}", style::dim("aivo claude"));
                println!("  {}", style::dim("aivo claude --model claude-sonnet-4.5"));
                println!("  {}", style::dim("aivo claude \"fix the login bug\""));
            }
            Some("codex") => {
                println!("  {}", style::dim("aivo codex"));
                println!("  {}", style::dim("aivo codex -k mykey -m gpt-5"));
                println!("  {}", style::dim("aivo codex \"refactor this function\""));
            }
            Some("codex-app") => {
                println!("  {}", style::dim("aivo codex-app"));
                println!("  {}", style::dim("aivo codex-app -k mykey -m gpt-5 ."));
            }
            Some("gemini") => {
                println!("  {}", style::dim("aivo gemini"));
                println!("  {}", style::dim("aivo gemini -k mykey"));
                println!("  {}", style::dim("aivo gemini \"explain this code\""));
            }
            Some("opencode") => {
                println!("  {}", style::dim("aivo opencode"));
                println!("  {}", style::dim("aivo opencode -k mykey"));
            }
            Some("pi") => {
                println!("  {}", style::dim("aivo pi"));
                println!("  {}", style::dim("aivo pi -k mykey"));
                println!("  {}", style::dim("aivo pi --transparent -k openrouter"));
            }
            _ => {
                println!("  {}", style::dim("aivo run claude"));
                println!(
                    "  {}",
                    style::dim("aivo run claude --model claude-sonnet-4.5")
                );
                println!("  {}", style::dim("aivo claude \"fix the login bug\""));
                println!("  {}", style::dim("aivo codex \"refactor this function\""));
            }
        }
    }
}

/// Emits one stderr line per Claude-only slot flag set when running a
/// non-Claude tool. Forgiving by design — preserves the launch.
fn warn_slot_flags_ignored(tool: AIToolType, slots: &ClaudeSlotFlags) {
    for (set, flag) in [
        (slots.reasoning.is_some(), "--reasoning-model"),
        (slots.subagent.is_some(), "--subagent-model"),
        (slots.haiku.is_some(), "--haiku-model"),
        (slots.sonnet.is_some(), "--sonnet-model"),
        (slots.opus.is_some(), "--opus-model"),
    ] {
        if set {
            eprintln!(
                "  {} {} is ignored for {}",
                style::yellow("!"),
                flag,
                tool.as_str(),
            );
        }
    }
}

/// Outcome of resolving a `--context` selector. `aivo run` skips injection on
/// Cancelled/Unavailable; `aivo code -c` launches on a cancel but bails on
/// `Unavailable` (its `String` is a ready-to-print reason).
pub(crate) enum ContextResolution {
    Selected(Thread),
    Cancelled,
    Unavailable(String),
}

/// Resolve a `--context` selector into one past session: empty → interactive
/// picker; else session-id prefix match. The scan runs under a spinner. Shared
/// with `aivo code -c`.
pub(crate) async fn resolve_context_thread(
    store: &SessionStore,
    selector: &str,
) -> ContextResolution {
    let Some(cwd) = system_env::current_dir() else {
        return ContextResolution::Unavailable("Could not determine the current directory.".into());
    };
    let opts = if selector.trim().is_empty() {
        IngestOptions::default()
    } else {
        // Explicit id: skip the substance filter — `aivo logs` lists
        // short-first-prompt sessions the user may name here.
        IngestOptions {
            include_short_first_user: true,
            ..IngestOptions::unlimited()
        }
    };

    // Spinner covers the scan only — stop it before the picker.
    let (spinning, spinner_handle) = style::start_spinner(Some(" Loading context…"));
    let mut threads = match ingest_project_with_code(store, &cwd, opts).await {
        Ok(t) => t,
        Err(e) => {
            style::stop_spinner(&spinning);
            let _ = spinner_handle.await;
            return ContextResolution::Unavailable(format!("Could not load context: {e}"));
        }
    };
    // The substance filter keeps the picker's working set high-signal, but a
    // project whose only sessions are short ("hello") would show an empty
    // picker right after `aivo logs` listed them. Retry permissive rather
    // than claim there's no context.
    if threads.is_empty() && selector.trim().is_empty() {
        threads = ingest_project_with_code(
            store,
            &cwd,
            IngestOptions {
                include_short_first_user: true,
                ..IngestOptions::default()
            },
        )
        .await
        .unwrap_or_default();
    }
    style::stop_spinner(&spinning);
    let _ = spinner_handle.await;

    match select_thread(&threads, selector) {
        SelectOutcome::Picked(t) => ContextResolution::Selected(t.clone()),
        SelectOutcome::Cancelled => ContextResolution::Cancelled,
        SelectOutcome::Err(msg) => ContextResolution::Unavailable(msg),
    }
}

/// One-line summary of what got injected, for stderr and the TUI startup notice.
pub(crate) fn context_injection_summary(rendered: &RenderedContext, thread: &Thread) -> String {
    let sid_prefix = &thread.session_id[..thread.session_id.len().min(8)];
    format!(
        "injected ~{} tokens from {} session {} ({})",
        rendered.tokens,
        thread.cli,
        sid_prefix,
        format_time_ago_short_dt(thread.updated_at),
    )
}

/// Print the injection summary to stderr (non-TUI paths).
pub(crate) fn announce_context_injection(rendered: &RenderedContext, thread: &Thread) {
    eprintln!(
        "  {} {}",
        style::arrow_symbol(),
        context_injection_summary(rendered, thread),
    );
}

/// Loads context from exactly one past session and injects it into the CLI
/// args. `selector`: empty → most-recent; otherwise prefix-match on session_id.
async fn maybe_inject_context(
    store: &SessionStore,
    tool: AIToolType,
    args: Vec<String>,
    selector: &str,
) -> Vec<String> {
    // Every tool has *some* injection path:
    //   claude, pi  → `--append-system-prompt <text>` (clean, hidden from user)
    //   codex       → prepended to [PROMPT] positional
    //   gemini      → `-i <text>` (prompt-interactive)
    //   opencode    → `--prompt <text>` (TUI launch flag)
    // The non-claude paths are visible to the user as part of the first
    // message; we wrap with a "context only — wait for me" preamble.
    let selected = match resolve_context_thread(store, selector).await {
        ContextResolution::Selected(thread) => thread,
        ContextResolution::Cancelled => {
            eprintln!(
                "  {} context picker cancelled; launching without injection",
                style::dim("›")
            );
            return args;
        }
        ContextResolution::Unavailable(msg) => {
            eprintln!("  {} {}", style::yellow("!"), msg);
            return args;
        }
    };

    let rendered = render_single_session(tool, &selected);
    announce_context_injection(&rendered, &selected);

    match tool {
        // Both claude and pi accept the same `--append-system-prompt <text>` flag.
        AIToolType::Claude | AIToolType::Pi => inject_append_system_prompt(&rendered, args),
        AIToolType::Codex | AIToolType::CodexApp => inject_codex(&rendered, args),
        AIToolType::Gemini => inject_via_flag(&rendered, args, "-i"),
        AIToolType::Opencode => inject_via_flag(&rendered, args, "--prompt"),
    }
}

/// Selection outcome. `Cancelled` distinguishes "user aborted the picker"
/// from a real error — it's treated as a soft skip, not a failure.
enum SelectOutcome<'a> {
    Picked(&'a Thread),
    Cancelled,
    Err(String),
}

fn select_thread<'a>(threads: &'a [Thread], selector: &str) -> SelectOutcome<'a> {
    if threads.is_empty() {
        return SelectOutcome::Err("No context available for this project yet.".to_string());
    }
    let selector = selector.trim();
    if selector.is_empty() {
        // Bare `--context`: interactive picker.
        return pick_interactive(threads);
    }
    let matches: Vec<&Thread> = threads
        .iter()
        .filter(|t| t.session_id.starts_with(selector))
        .collect();
    match matches.len() {
        0 => SelectOutcome::Err(format!(
            "No session matches id prefix '{}' in this project. Run `aivo logs` to see available ids (sessions from other directories don't count).",
            selector
        )),
        1 => SelectOutcome::Picked(matches[0]),
        _ => {
            let options = matches
                .iter()
                .take(5)
                .map(|t| {
                    format!(
                        "{} ({})",
                        &t.session_id[..t.session_id.len().min(12)],
                        t.cli
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            SelectOutcome::Err(format!(
                "Session prefix '{}' is ambiguous ({} matches: {}). Use more characters.",
                selector,
                matches.len(),
                options
            ))
        }
    }
}

fn pick_interactive(threads: &[Thread]) -> SelectOutcome<'_> {
    use crate::tui::FuzzySelect;

    let max_cli_len = threads.iter().map(|t| t.cli.len()).max().unwrap_or(6);
    let items: Vec<String> = threads
        .iter()
        .map(|t| {
            format!(
                "{}  {:<cli_w$}  {:>8}  {}",
                &t.session_id[..t.session_id.len().min(8)],
                t.cli,
                format_time_ago_short_dt(t.updated_at),
                trim_to_one_line(&t.topic, 70),
                cli_w = max_cli_len,
            )
        })
        .collect();

    match FuzzySelect::new()
        .with_prompt("Select a session to inject")
        .items(&items)
        .default(0)
        .interact_opt()
    {
        Ok(Some(idx)) => SelectOutcome::Picked(&threads[idx]),
        Ok(None) | Err(_) => SelectOutcome::Cancelled,
    }
}

/// `--append-system-prompt <rendered>` goes at the front so it applies
/// regardless of what comes next in the passthrough args. Used by both
/// claude and pi (their CLIs accept the identical flag).
fn inject_append_system_prompt(rendered: &RenderedContext, args: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len() + 2);
    out.push("--append-system-prompt".to_string());
    out.push(rendered.text.clone());
    out.extend(args);
    out
}

/// Preamble shown to tools that receive the context as a "user message"
/// (anything that isn't claude/pi). Tells the model to treat what follows as
/// background and wait for the user's actual instruction.
const CONTEXT_PREAMBLE: &str = "The block below is aivo context — auto-extracted from previous AI CLI \
     sessions in this project. Treat it as background. Acknowledge briefly, then wait \
     for the user's next message before taking any action.\n\n";

/// Inject context for tools whose only hook is a `<flag> <text>` arg pair
/// (gemini's `-i`, opencode's `--prompt`). If the user already supplied the
/// same flag, prepend the context to its existing value; otherwise append a
/// new `<flag> <preamble + context>` pair.
fn inject_via_flag(rendered: &RenderedContext, mut args: Vec<String>, flag: &str) -> Vec<String> {
    // Find an existing `<flag> <value>` or `<flag>=<value>` pair.
    if let Some(idx) = args.iter().position(|a| a == flag) {
        // `<flag> <value>` form — value is the next arg.
        if idx + 1 < args.len() {
            let existing = std::mem::take(&mut args[idx + 1]);
            args[idx + 1] = format!("{}\n\n{}", rendered.text, existing);
            return args;
        }
        // Trailing bare flag: supply context as its value.
        args.push(format!("{}{}", CONTEXT_PREAMBLE, rendered.text));
        return args;
    }
    let prefix = format!("{flag}=");
    if let Some(idx) = args.iter().position(|a| a.starts_with(&prefix)) {
        // `<flag>=<value>` form — splice into the same arg.
        let existing = args[idx][prefix.len()..].to_string();
        args[idx] = format!("{prefix}{}\n\n{existing}", rendered.text);
        return args;
    }
    // No existing flag: append the context-only preamble + content.
    args.push(flag.to_string());
    args.push(format!("{}{}", CONTEXT_PREAMBLE, rendered.text));
    args
}

/// Codex has no system-prompt flag (as of 2026-04), so we inject context via
/// the `[PROMPT]` positional. Two cases:
/// - User already provided a prompt → prepend context to that prompt.
/// - No prompt (interactive launch) → pass context alone, wrapped in a
///   "context only" preamble so codex treats it as background and waits for
///   the user's real question instead of trying to act on the context itself.
fn inject_codex(rendered: &RenderedContext, mut args: Vec<String>) -> Vec<String> {
    let last_positional = args.iter().rposition(|a| !a.starts_with('-'));

    match last_positional {
        // A non-dash token after a valueless `--flag` may be that flag's value,
        // not the prompt (`--sandbox read-only`); skip rather than corrupt it.
        Some(idx) if idx > 0 && args[idx - 1].starts_with('-') && !args[idx - 1].contains('=') => {
            eprintln!(
                "  {} skipping context injection: {:?} may be the value of {:?} rather than the prompt; use `{}=<value>` to disambiguate",
                style::yellow("!"),
                args[idx],
                args[idx - 1],
                args[idx - 1],
            );
            args
        }
        Some(idx) => {
            let prompt = std::mem::take(&mut args[idx]);
            args[idx] = format!("{}\n\n{}", rendered.text, prompt);
            args
        }
        None => {
            // Interactive launch — no user prompt. Inject context as the
            // initial prompt but wrap it so codex reads it as background
            // rather than an instruction to act on.
            args.push(format!("{}{}", CONTEXT_PREAMBLE, rendered.text));
            args
        }
    }
}

/// Default `--max-context` based on the resolved model.
/// Precedence: explicit flag → limits cascade (live models-cache, then the
/// embedded models.dev snapshot; ≥2M→2m, ≥1M→1m — see
/// `services::model_metadata`). `aivo/starter` gets no special default: the
/// cascade knows real context windows, so an unresolved model means no tag.
async fn resolve_max_context(
    cache: &ModelsCache,
    base_url: Option<&str>,
    model: Option<&str>,
    explicit: Option<String>,
) -> Option<String> {
    if explicit.is_some() {
        return explicit;
    }
    let ctx = crate::services::model_metadata::resolve_limits(cache, base_url, model?)
        .await
        .context;
    match ctx {
        Some(c) if c >= 2_000_000 => Some("2m".to_string()),
        Some(c) if c >= 1_000_000 => Some("1m".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_append_system_prompt_puts_flag_first() {
        let rendered = RenderedContext {
            text: "<mem>hello</mem>".into(),
            tokens: 10,
        };
        let out = inject_append_system_prompt(
            &rendered,
            vec!["--model".into(), "opus".into(), "prompt".into()],
        );
        assert_eq!(out[0], "--append-system-prompt");
        assert_eq!(out[1], "<mem>hello</mem>");
        assert_eq!(out[2], "--model");
        assert_eq!(out[3], "opus");
        assert_eq!(out[4], "prompt");
    }

    #[test]
    fn inject_codex_prepends_context_to_last_positional() {
        let rendered = RenderedContext {
            text: "MEM".into(),
            tokens: 1,
        };
        let out = inject_codex(
            &rendered,
            vec!["-m".into(), "gpt-5".into(), "fix the bug".into()],
        );
        assert_eq!(out[0], "-m");
        assert_eq!(out[1], "gpt-5");
        assert_eq!(out[2], "MEM\n\nfix the bug");
    }

    #[test]
    fn inject_codex_without_prompt_appends_context_as_initial_prompt() {
        let rendered = RenderedContext {
            text: "CTX".into(),
            tokens: 1,
        };
        let out = inject_codex(&rendered, vec!["--verbose".to_string()]);
        assert_eq!(out[0], "--verbose");
        // Context is appended as a new positional at the end.
        assert_eq!(out.len(), 2);
        assert!(out[1].contains("aivo context"));
        assert!(out[1].ends_with("CTX"));
    }

    #[test]
    fn inject_codex_skips_when_trailing_token_may_be_flag_value() {
        let rendered = RenderedContext {
            text: "CTX".into(),
            tokens: 1,
        };
        let args = vec!["--sandbox".to_string(), "read-only".to_string()];
        let out = inject_codex(&rendered, args.clone());
        assert_eq!(out, args);
    }

    #[test]
    fn inject_codex_equals_form_disambiguates() {
        let rendered = RenderedContext {
            text: "CTX".into(),
            tokens: 1,
        };
        let out = inject_codex(
            &rendered,
            vec!["--sandbox=read-only".to_string(), "fix the bug".to_string()],
        );
        assert_eq!(out[0], "--sandbox=read-only");
        assert_eq!(out[1], "CTX\n\nfix the bug");
    }

    #[test]
    fn inject_via_flag_trailing_bare_flag_gets_context_as_value() {
        let rendered = RenderedContext {
            text: "CTX".into(),
            tokens: 1,
        };
        let out = inject_via_flag(&rendered, vec!["-i".to_string()], "-i");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], "-i");
        assert!(out[1].ends_with("CTX"));
    }

    #[test]
    fn test_valid_ai_tools() {
        assert!(AIToolType::parse("claude").is_some());
        assert!(AIToolType::parse("codex").is_some());
        assert!(AIToolType::parse("gemini").is_some());
        assert!(AIToolType::parse("opencode").is_some());
        assert!(AIToolType::parse("pi").is_some());
    }

    #[test]
    fn test_invalid_ai_tool() {
        assert!(AIToolType::parse("unknown").is_none());
        assert!(AIToolType::parse("").is_none());
        assert!(AIToolType::parse("chatgpt").is_none());
    }

    #[test]
    fn test_ai_tool_type_display_names() {
        // Ensure all tools have valid string representations
        let tools = ["claude", "codex", "codex-app", "gemini", "opencode", "pi"];
        for tool in &tools {
            let parsed = AIToolType::parse(tool).unwrap();
            // Roundtrip: parsing should give a valid tool type
            assert!(
                matches!(
                    parsed,
                    AIToolType::Claude
                        | AIToolType::Codex
                        | AIToolType::CodexApp
                        | AIToolType::Gemini
                        | AIToolType::Opencode
                        | AIToolType::Pi
                ),
                "Tool {} should parse to a valid AIToolType",
                tool
            );
        }
    }

    use crate::services::models_cache::{ModelMetadata, full_catalog_key};
    use std::collections::HashMap;
    use tempfile::TempDir;

    const TEST_BASE_URL: &str = "https://api.example.com";

    /// TempDir is returned so the cache file outlives the test body.
    fn empty_cache() -> (TempDir, ModelsCache) {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = ModelsCache::with_path(dir.path().join("models-cache.json"));
        (dir, cache)
    }

    async fn cache_with_context(model: &str, ctx_tokens: u64) -> (TempDir, ModelsCache) {
        let (dir, cache) = empty_cache();
        let mut metadata = HashMap::new();
        metadata.insert(
            model.to_string(),
            ModelMetadata {
                context_window: Some(ctx_tokens),
                ..Default::default()
            },
        );
        cache
            .set_with_metadata(
                &full_catalog_key(TEST_BASE_URL),
                vec![model.to_string()],
                metadata,
            )
            .await;
        (dir, cache)
    }

    #[tokio::test]
    async fn resolve_max_context_no_blind_default_for_starter_model() {
        // The cascade is the only source now — an unresolved starter model
        // gets no tag instead of the old hardcoded "1m".
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(
                &cache,
                None,
                Some(crate::constants::AIVO_STARTER_MODEL),
                None
            )
            .await,
            None
        );
    }

    #[tokio::test]
    async fn resolve_max_context_starter_model_follows_cache() {
        let (_dir, cache) =
            cache_with_context(crate::constants::AIVO_STARTER_MODEL, 1_000_000).await;
        assert_eq!(
            resolve_max_context(
                &cache,
                Some(TEST_BASE_URL),
                Some(crate::constants::AIVO_STARTER_MODEL),
                None
            )
            .await,
            Some("1m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_user_override_wins_for_starter_model() {
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(
                &cache,
                None,
                Some(crate::constants::AIVO_STARTER_MODEL),
                Some("2m".to_string())
            )
            .await,
            Some("2m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_no_default_for_unknown_short_context_model() {
        // Cache miss + not in the static long-context list → no default.
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(&cache, None, Some("gpt-3.5-turbo"), None).await,
            None
        );
    }

    #[tokio::test]
    async fn resolve_max_context_static_fallback_for_known_1m_claude() {
        // Cache cold → static list still resolves Claude 1M models.
        // Anthropic's /v1/models doesn't expose context_length, so this
        // path is the common case for fresh installs.
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(&cache, None, Some("claude-sonnet-4-6"), None).await,
            Some("1m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_static_fallback_for_grok_4_3() {
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(&cache, Some(TEST_BASE_URL), Some("grok-4.3"), None).await,
            Some("1m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_static_fallback_for_grok_4_fast_2m() {
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(&cache, Some(TEST_BASE_URL), Some("grok-4-fast"), None).await,
            Some("2m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_static_fallback_handles_provider_prefix() {
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(
                &cache,
                Some(TEST_BASE_URL),
                Some("anthropic/claude-opus-4-7"),
                None,
            )
            .await,
            Some("1m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_explicit_wins_over_static_fallback() {
        let (_dir, cache) = empty_cache();
        assert_eq!(
            resolve_max_context(
                &cache,
                None,
                Some("claude-sonnet-4-6"),
                Some("200k".to_string()),
            )
            .await,
            Some("200k".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_no_default_when_model_missing() {
        let (_dir, cache) = empty_cache();
        assert_eq!(resolve_max_context(&cache, None, None, None).await, None);
    }

    #[tokio::test]
    async fn resolve_max_context_uses_cache_for_1m_model() {
        let (_dir, cache) = cache_with_context("gpt-4.1", 1_000_000).await;
        assert_eq!(
            resolve_max_context(&cache, Some(TEST_BASE_URL), Some("gpt-4.1"), None).await,
            Some("1m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_uses_cache_for_2m_model() {
        let (_dir, cache) = cache_with_context("huge-context-model", 2_000_000).await;
        assert_eq!(
            resolve_max_context(
                &cache,
                Some(TEST_BASE_URL),
                Some("huge-context-model"),
                None,
            )
            .await,
            Some("2m".to_string())
        );
    }

    #[tokio::test]
    async fn resolve_max_context_cache_below_1m_returns_none() {
        let (_dir, cache) = cache_with_context("small-context-model", 200_000).await;
        assert_eq!(
            resolve_max_context(
                &cache,
                Some(TEST_BASE_URL),
                Some("small-context-model"),
                None,
            )
            .await,
            None
        );
    }

    #[tokio::test]
    async fn resolve_max_context_explicit_wins_over_cached_metadata() {
        let (_dir, cache) = cache_with_context("gpt-4.1", 1_000_000).await;
        assert_eq!(
            resolve_max_context(
                &cache,
                Some(TEST_BASE_URL),
                Some("gpt-4.1"),
                Some("2m".to_string()),
            )
            .await,
            Some("2m".to_string())
        );
    }
}
