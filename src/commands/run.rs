/**
 * RunCommand handler for unified AI tool launching.
 */
use std::collections::HashMap;

use anyhow::Result;
use reqwest::Client;

use crate::commands::chat_tui_format::format_time_ago_short_dt;
use crate::commands::models::{
    fetch_models_for_select, prompt_model_picker, resolve_model_placeholder,
};
use crate::commands::{print_launch_preview, trim_to_one_line};
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::context_ingest::{IngestOptions, ingest_project};
use crate::services::context_render::{RenderedContext, render_single_session};
use crate::services::http_utils;
use crate::services::models_cache::ModelsCache;
use crate::services::nickname_registry;
use crate::services::project_id::Thread;
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::share_config::{ShareCleanup, maybe_enable_share};
use crate::services::system_env;
use crate::style;

/// Outcome of picker-style model resolution. Distinguishes "user cancelled
/// the picker" (exit success, don't launch) from "no fetchable model list,
/// fall back to the tool's default" (launch anyway, no model flag).
enum ModelOutcome {
    /// User picked a model, or `--model <value>` was passed.
    Model(String),
    /// No `--model` flag, or the picker fetched an empty list. Launch the
    /// tool with its own default model.
    UseDefault,
    /// Picker was shown and the user cancelled (Ctrl-C / Esc). Caller
    /// should exit without launching.
    Cancelled,
}

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
    /// --model <value> → use as-is. --model (no value) → show picker.
    /// No --model flag → returns None (let the tool use its own default).
    /// Returns None when the picker was cancelled or no flag was given.
    async fn resolve_model(
        &self,
        client: &Client,
        key: &ApiKey,
        flag_model: Option<String>,
        explicit_model_flag: bool,
        refresh: bool,
        tool: AIToolType,
    ) -> Result<ModelOutcome> {
        match flag_model {
            // No --model flag → don't override, let the tool use its default
            None => return Ok(ModelOutcome::UseDefault),
            // --model <value> → use it as-is
            Some(ref m) if !m.is_empty() => return Ok(ModelOutcome::Model(m.clone())),
            // --model with no value → show picker
            Some(_) => {}
        }

        let models_list = if refresh {
            crate::commands::models::fetch_models_cached(client, key, &self.cache, true)
                .await
                .unwrap_or_default()
        } else {
            fetch_models_for_select(client, key, &self.cache).await
        };

        if models_list.is_empty() {
            // No fetchable model list (common for providers without a public
            // /v1/models endpoint — e.g. Codex ChatGPT OAuth). Skip the
            // picker and let the tool use its own default rather than
            // blocking the launch. Only explain this when the user
            // explicitly asked for a picker; the implicit picker on first
            // launch falls through silently.
            if explicit_model_flag {
                eprintln!(
                    "  {} {}",
                    style::dim("note:"),
                    crate::commands::NO_MODEL_LIST_HINT
                );
            }
            return Ok(ModelOutcome::UseDefault);
        }

        match prompt_model_picker(models_list, Some(tool)) {
            Some(m) => Ok(ModelOutcome::Model(m)),
            None => Ok(ModelOutcome::Cancelled),
        }
    }

    /// Executes the run command with the specified AI tool
    #[allow(clippy::too_many_arguments)]
    pub async fn execute(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        explicit_model_flag: bool,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
        context_selector: Option<String>,
        as_name: Option<String>,
    ) -> ExitCode {
        match self
            .execute_internal(
                tool,
                args,
                debug,
                dry_run,
                refresh,
                model,
                explicit_model_flag,
                env,
                key_override,
                context_selector,
                as_name,
            )
            .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_internal(
        &self,
        tool: Option<&str>,
        args: Vec<String>,
        debug: bool,
        dry_run: bool,
        refresh: bool,
        model: Option<String>,
        explicit_model_flag: bool,
        env: Option<HashMap<String, String>>,
        key_override: Option<ApiKey>,
        context_selector: Option<String>,
        as_name: Option<String>,
    ) -> anyhow::Result<ExitCode> {
        let tool = match tool {
            Some(t) => t,
            None => {
                Self::print_help();
                return Ok(ExitCode::UserError);
            }
        };

        // Handle help flags
        if tool == "--help" || tool == "-h" {
            Self::print_help();
            return Ok(ExitCode::Success);
        }

        // Validate tool
        let ai_tool = match AIToolType::parse(tool) {
            Some(t) => t,
            None => {
                eprintln!("{} Unknown AI tool '{}'", style::red("Error:"), tool);
                eprintln!();
                eprintln!("Available tools:");
                eprintln!(
                    "  {}    {}",
                    style::cyan("claude"),
                    style::dim("Claude Code")
                );
                eprintln!("  {}     {}", style::cyan("codex"), style::dim("Codex"));
                eprintln!("  {}    {}", style::cyan("gemini"), style::dim("Gemini"));
                eprintln!("  {}  {}", style::cyan("opencode"), style::dim("OpenCode"));
                eprintln!("  {}        {}", style::cyan("pi"), style::dim("Pi"));
                eprintln!();
                eprintln!(
                    "{}",
                    style::dim("Usage: aivo run <tool> [options] [args...]")
                );
                return Ok(ExitCode::UserError);
            }
        };

        // OAuth keys carry serialized tokens only the matching native CLI can
        // consume (shadow CODEX_HOME, CLAUDE_CODE_OAUTH_TOKEN, shadow
        // GEMINI_CLI_HOME); every other tool would see an unusable JSON blob.
        let mut key_override = key_override;
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
                .resolve_model(&client, key, model, explicit_model_flag, refresh, ai_tool)
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

        if let Some(ref key) = key_override {
            let _ = self
                .session_store
                .set_last_selection(key, tool, resolved_model.as_deref())
                .await;
        }

        let launch_model = resolve_model_placeholder(resolved_model);

        // Optional context injection: inject exactly one past session.
        let args = if let Some(selector) = context_selector {
            maybe_inject_context(ai_tool, args, &selector).await
        } else {
            args
        };

        // Cross-tool MCP wiring: always enabled. Each tool gets a nickname
        // (explicit via `--as reviewer`, or auto-derived from the tool name)
        // and an aivo MCP server so peers can call list_sessions / get_session.
        let (args, _share_cleanup) = {
            let cwd = system_env::current_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
            let registry_root = nickname_registry::registry_dir_for_cwd(&cwd);

            // Register: explicit --as name, or auto-name from tool (claude, codex, …)
            let (nickname, registry_guard) = if let Some(ref root) = registry_root {
                if let Some(ref explicit) = as_name {
                    match nickname_registry::register(explicit, ai_tool.as_str(), root).await {
                        Ok(guard) => (explicit.clone(), Some(guard)),
                        Err(e) => {
                            eprintln!(
                                "  {} --as: {}; launching without nickname.",
                                style::yellow("!"),
                                e
                            );
                            (explicit.clone(), None)
                        }
                    }
                } else {
                    match nickname_registry::register_auto(ai_tool.as_str(), root).await {
                        Ok((name, guard)) => (name, Some(guard)),
                        Err(_) => (ai_tool.as_str().to_string(), None),
                    }
                }
            } else {
                (
                    as_name
                        .clone()
                        .unwrap_or_else(|| ai_tool.as_str().to_string()),
                    None,
                )
            };

            let fallback = args.clone();
            match maybe_enable_share(ai_tool, args, &cwd, &nickname).await {
                Ok((new_args, mut cleanup)) => {
                    if let Some(guard) = registry_guard {
                        cleanup.set_registry_guard(guard);
                    }
                    (new_args, cleanup)
                }
                Err(e) => {
                    eprintln!(
                        "  {} MCP wiring failed: {}; launching without cross-tool context.",
                        style::yellow("!"),
                        e
                    );
                    (fallback, ShareCleanup::empty())
                }
            }
        };

        // Launch the AI tool
        let options = LaunchOptions {
            tool: ai_tool,
            args,
            debug,
            model: launch_model,
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

    /// Shows usage information
    pub fn print_help() {
        println!("{} aivo run [tool] [args...]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Launch an AI coding assistant with local API keys.")
        );
        println!(
            "{}",
            style::dim(
                "When no tool is provided, `aivo run` falls back to the saved `start` flow."
            )
        );
        println!(
            "{}",
            style::dim("All arguments are passed through to the underlying tool.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-m, --model <model>", "Specify AI model to use");
        print_opt(
            "-k, --key <id|name>",
            "Select API key by ID or name (-k opens key picker)",
        );
        print_opt("-r, --refresh", "Bypass cache and fetch fresh model list");
        print_opt("--env <k=v>", "Inject environment variable");
        print_opt(
            "-c, --context[=<id>]",
            "Inject one past session (bare = picker; id from `aivo context`)",
        );
        print_opt(
            "--as <name>",
            "Name this tool for cross-tool MCP communication",
        );
        print_opt(
            "--dry-run",
            "Print resolved command and environment without launching",
        );
        println!();
        println!("{}", style::bold("Tools:"));
        let print_tool = |label: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<12}", label)),
                style::dim(desc)
            );
        };
        print_tool("claude", "Claude Code");
        print_tool("codex", "Codex");
        print_tool("gemini", "Gemini");
        print_tool("opencode", "OpenCode");
        print_tool("pi", "Pi");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo run claude"));
        println!(
            "  {}",
            style::dim("aivo run claude --model claude-sonnet-4.5")
        );
        println!("  {}", style::dim("aivo claude \"fix the login bug\""));
        println!("  {}", style::dim("aivo codex \"refactor this function\""));
        println!("  {}", style::dim("aivo gemini \"explain this code\""));
    }
}

/// Loads context from exactly one past session and injects it into the CLI
/// args. `selector`: empty → most-recent; otherwise prefix-match on session_id.
async fn maybe_inject_context(tool: AIToolType, args: Vec<String>, selector: &str) -> Vec<String> {
    // Every tool has *some* injection path:
    //   claude, pi  → `--append-system-prompt <text>` (clean, hidden from user)
    //   codex       → prepended to [PROMPT] positional
    //   gemini      → `-i <text>` (prompt-interactive)
    //   opencode    → `--prompt <text>` (TUI launch flag)
    // The non-claude paths are visible to the user as part of the first
    // message; we wrap with a "context only — wait for me" preamble.

    let cwd = match system_env::current_dir() {
        Some(p) => p,
        None => return args,
    };

    // Default options pick the recent set for the picker. When the user
    // supplied a specific session id, scan everything — they may be reaching
    // for an older session shown by `aivo context -a`.
    let opts = if selector.trim().is_empty() {
        IngestOptions::default()
    } else {
        IngestOptions::unlimited()
    };
    let threads = match ingest_project(&cwd, opts).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  {} Skipping context injection: {}", style::yellow("!"), e);
            return args;
        }
    };

    let selected = match select_thread(&threads, selector) {
        SelectOutcome::Picked(t) => t,
        SelectOutcome::Cancelled => {
            eprintln!(
                "  {} context picker cancelled; launching without injection",
                style::dim("›")
            );
            return args;
        }
        SelectOutcome::Err(msg) => {
            eprintln!("  {} {}", style::yellow("!"), msg);
            return args;
        }
    };

    let rendered = render_single_session(tool, selected);
    let sid_prefix = &selected.session_id[..selected.session_id.len().min(8)];
    eprintln!(
        "  {} injecting {} tokens from {} session {} ({})",
        style::arrow_symbol(),
        rendered.tokens,
        selected.cli,
        sid_prefix,
        format_time_ago_short_dt(selected.updated_at),
    );

    match tool {
        // Both claude and pi accept the same `--append-system-prompt <text>` flag.
        AIToolType::Claude | AIToolType::Pi => inject_append_system_prompt(&rendered, args),
        AIToolType::Codex => inject_codex(&rendered, args),
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
            "No session matches id prefix '{}'. Run `aivo context` to see available ids.",
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
        let tools = ["claude", "codex", "gemini", "opencode", "pi"];
        for tool in &tools {
            let parsed = AIToolType::parse(tool).unwrap();
            // Roundtrip: parsing should give a valid tool type
            assert!(
                matches!(
                    parsed,
                    AIToolType::Claude
                        | AIToolType::Codex
                        | AIToolType::Gemini
                        | AIToolType::Opencode
                        | AIToolType::Pi
                ),
                "Tool {} should parse to a valid AIToolType",
                tool
            );
        }
    }
}
