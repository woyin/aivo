/**
 * Main entry point for the aivo CLI.
 * Initializes services with dependency injection and routes commands to handlers.
 */
use std::process;

use clap::{CommandFactory, Parser};

mod cli;
mod commands;
mod constants;
mod errors;
mod key_resolution;
mod services;
mod style;
mod tui;
mod version;

use cli::{Cli, Commands};
use commands::{
    AliasCommand, ChatCommand, ContextCommand, InfoCommand, KeysCommand, LogsCommand,
    McpServeCommand, ModelsCommand, RunCommand, ServeCommand, ServeParams, StartCommand,
    StartFlowArgs, StatsCommand, UpdateCommand,
};
use errors::ExitCode;
use key_resolution::{KeyLookupMode, KeyResolution, key_or_exit, resolve_key_override};
use services::ai_launcher::AIToolType;
use services::key_compat::KeyCompatContext;
use services::{AILauncher, EnvironmentInjector, SessionStore};

/// Known AI tool names that can be used as shortcut aliases for `run`.
const TOOL_ALIASES: &[&str] = &["claude", "codex", "gemini", "opencode", "pi"];

/// Collects all single-character short flags from the CLI definition,
/// including global flags and all subcommands.
fn collect_known_short_flags() -> Vec<char> {
    let cmd = Cli::command();
    let mut flags: Vec<char> = Vec::new();
    let all_args = cmd
        .get_arguments()
        .chain(cmd.get_subcommands().flat_map(|s| s.get_arguments()));
    for arg in all_args {
        if let Some(c) = arg.get_short()
            && !flags.contains(&c)
        {
            flags.push(c);
        }
    }
    flags
}

/// Refuses to run a `test-fast-crypto`-built binary against real user config.
///
/// `test-fast-crypto` (documented in CLAUDE.md) uses reduced PBKDF2 iterations
/// so the test suite runs fast. A binary built with that feature derives a
/// different encryption key than a normal release binary, so it silently fails
/// to decrypt any real API key stored in `~/.config/aivo/config.json`. Shipping
/// or running such a binary as a CLI is always a mistake. Tests run library
/// code directly in tempdirs and don't hit this guard.
///
/// Override for intentional manual testing: `AIVO_TEST_FAST_CRYPTO_OK=1`.
#[cfg(feature = "test-fast-crypto")]
fn fast_crypto_guard() {
    if std::env::var_os("AIVO_TEST_FAST_CRYPTO_OK").is_some() {
        return;
    }
    eprintln!(
        "{} This aivo binary was built with the `test-fast-crypto` feature,\n       which uses reduced PBKDF2 iterations for fast tests.\n       It cannot decrypt keys encrypted by a normal aivo binary.\n\n       Rebuild without the feature: {}\n       Or, to override intentionally, set {}",
        style::red("error:"),
        style::cyan("cargo build"),
        style::dim("AIVO_TEST_FAST_CRYPTO_OK=1"),
    );
    process::exit(ExitCode::UserError.code());
}

#[cfg(not(feature = "test-fast-crypto"))]
fn fast_crypto_guard() {}

/// Main entry point for the CLI
#[tokio::main(flavor = "current_thread")]
async fn main() {
    fast_crypto_guard();
    let raw_args: Vec<String> = std::env::args().collect();
    let args = Cli::parse_from(rewrite_cli_args(expand_combined_short_flags(raw_args)));

    // Handle --version and subcommand --help early, before any service initialization.
    if args.version {
        print_version();
        process::exit(0);
    }

    if args.help
        && let Some(cmd) = &args.command
    {
        match cmd {
            Commands::Run(_) => RunCommand::print_help(),
            Commands::Keys(_) => KeysCommand::print_help(),
            Commands::Chat(_) => ChatCommand::print_help(),
            Commands::Models(_) => ModelsCommand::print_help(),
            Commands::Serve(_) => ServeCommand::print_help(),
            Commands::Alias(_) => AliasCommand::print_help(),
            Commands::Info(_) => InfoCommand::print_help(),
            Commands::Logs(_) => LogsCommand::print_help(),
            Commands::Stats(_) => StatsCommand::print_help(),
            Commands::Update(_) => UpdateCommand::print_help(),
            Commands::Context(_) => ContextCommand::print_help(),
            Commands::McpServe(_) => {
                eprintln!(
                    "aivo mcp-serve is an internal stdio MCP server launched by Claude/Codex via --as."
                );
                eprintln!(
                    "Usage: aivo mcp-serve --cwd <PATH>  (run by the host tool, not by users)"
                );
            }
        }
        process::exit(0);
    }

    // Initialize services
    let session_store = SessionStore::new();
    let models_cache = services::ModelsCache::new();

    // Ensure the free starter key exists for all users.
    // For new users (no keys), also activate it.
    if let Some((starter, is_new_user)) = session_store.ensure_starter_key().await
        && is_new_user
    {
        let _ = session_store.set_active_key(&starter.id).await;
    }

    if args.help {
        print_help();
        print_active_selection(&session_store).await;
        process::exit(0);
    }

    // Get the command or show help if none provided
    let command = match args.command {
        Some(cmd) => cmd,
        None => {
            print_help();
            print_active_selection(&session_store).await;
            process::exit(0);
        }
    };

    // Route to command handler
    let exit_code = match command {
        Commands::Alias(alias_args) => {
            let command = AliasCommand::new(session_store);
            command.execute(alias_args).await
        }

        Commands::Keys(keys_args) => {
            let command = KeysCommand::new(session_store);
            command.execute(keys_args).await
        }

        Commands::Chat(chat_args) => {
            let key_explicit = chat_args.key.is_some();
            let key_override = key_or_exit(
                resolve_key_override(
                    &session_store,
                    chat_args.key.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                    KeyCompatContext::Chat,
                )
                .await,
            );
            // When -k is used without -m, force model picker (same as run/start)
            let model = if chat_args.model.is_some() {
                resolve_model_alias(&session_store, chat_args.model).await
            } else if key_explicit {
                Some(String::new())
            } else {
                None
            };
            let command = ChatCommand::new(session_store, models_cache.clone());
            command
                .execute(
                    model,
                    chat_args.execute,
                    chat_args.attachments,
                    chat_args.refresh,
                    key_override,
                    chat_args.json,
                )
                .await
        }

        Commands::Run(run_args) => {
            let env_injector = EnvironmentInjector::new();
            let ai_launcher =
                AILauncher::new(session_store.clone(), env_injector, models_cache.clone());

            // Re-extract aivo flags from passthrough args that clap's trailing_var_arg
            // may have swallowed (e.g. `aivo run claude --agent-name foo --model opus`
            // puts --model into args instead of parsing it as an aivo flag).
            let extracted = extract_aivo_flags(
                run_args.model,
                run_args.key,
                run_args.debug,
                run_args.dry_run,
                run_args.refresh,
                run_args.as_name,
                run_args.envs,
                &run_args.args,
            );
            let model = resolve_model_alias(&session_store, extracted.model).await;
            let key_flag = extracted.key_flag;
            let debug = extracted.debug;
            let dry_run = extracted.dry_run;
            let refresh = extracted.refresh;
            let as_name = extracted.as_name;
            // Context selector: prefer clap-parsed value, fall back to passthrough-recovered.
            let context_selector = run_args.context.or(extracted.context);
            let env_strings = extracted.env_strings;
            let remaining_args = extracted.remaining_args;

            if run_args.tool.is_none() {
                if !remaining_args.is_empty() {
                    eprintln!(
                        "{} `aivo run` without a tool does not accept passthrough args",
                        style::red("Error:")
                    );
                    eprintln!(
                        "  {}",
                        style::dim("Use `aivo run <tool> ...` for passthrough flags.")
                    );
                    process::exit(ExitCode::UserError.code());
                }

                let command = StartCommand::new(session_store, ai_launcher, models_cache);
                command
                    .execute(StartFlowArgs {
                        model,
                        key: key_flag,
                        tool: None,
                        debug,
                        dry_run,
                        refresh,
                        yes: false,
                        envs: env_strings,
                    })
                    .await
            } else {
                let command = RunCommand::new(session_store.clone(), ai_launcher, models_cache);

                let key_explicit = key_flag.is_some();
                let compat = run_args
                    .tool
                    .as_deref()
                    .and_then(AIToolType::parse)
                    .map(KeyCompatContext::Tool)
                    .unwrap_or(KeyCompatContext::None);
                let key_override = key_or_exit(
                    resolve_key_override(
                        &session_store,
                        key_flag.as_deref(),
                        KeyLookupMode::RequireActiveOrPrompt,
                        compat,
                    )
                    .await,
                );

                // Resolve model using last selection when no explicit flags given.
                // When -k is used without -m, always force the model picker.
                let model_flag_explicit = model.is_some();
                let model = if model.is_some() {
                    // -m was explicitly provided → use it
                    model
                } else if key_explicit {
                    // -k used without -m → force model picker
                    Some(String::new())
                } else {
                    // No -k, no -m → check last_selection for saved model
                    let last_sel = session_store.get_last_selection().await.ok().flatten();
                    match last_sel {
                        Some(ref sel)
                            if key_override.as_ref().is_some_and(|k| k.id == sel.key_id) =>
                        {
                            // Same key as last time — reuse saved model (could be __default__)
                            sel.model.clone().or(Some(String::new()))
                        }
                        _ => Some(String::new()), // no matching selection → trigger picker
                    }
                };

                let env = if !env_strings.is_empty() {
                    let mut map = std::collections::HashMap::new();
                    for env_str in &env_strings {
                        if let Some((key, value)) = env_str.split_once('=') {
                            map.insert(key.to_string(), value.to_string());
                        } else {
                            eprintln!(
                                "{} Ignoring malformed env value '{}' (expected KEY=VALUE format)",
                                style::yellow("Warning:"),
                                env_str
                            );
                        }
                    }
                    Some(map)
                } else {
                    None
                };

                command
                    .execute(
                        run_args.tool.as_deref(),
                        remaining_args,
                        debug,
                        dry_run,
                        refresh,
                        model,
                        model_flag_explicit,
                        env,
                        key_override,
                        context_selector,
                        as_name,
                    )
                    .await
            }
        }

        Commands::Models(models_args) => {
            let key_override = key_or_exit(
                resolve_key_override(
                    &session_store,
                    models_args.key.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                    KeyCompatContext::None,
                )
                .await,
            );
            let command = ModelsCommand::new(session_store, models_cache);
            command
                .execute(
                    key_override,
                    models_args.refresh,
                    models_args.search,
                    models_args.json,
                )
                .await
        }

        Commands::Serve(serve_args) => {
            let key_override = match resolve_key_override(
                &session_store,
                serve_args.key.as_deref(),
                KeyLookupMode::PreferActiveAllowNone,
                KeyCompatContext::None,
            )
            .await
            {
                Ok(KeyResolution::Selected(key)) => Some(key),
                Ok(KeyResolution::Cancelled) => process::exit(ExitCode::Success.code()),
                Ok(KeyResolution::MissingAuth) => None,
                Err(e) => {
                    eprintln!("{} {}", style::red("Error:"), e);
                    process::exit(ExitCode::UserError.code());
                }
            };
            // Build failover key list when --failover is enabled
            let failover_keys = if serve_args.failover {
                let mut all_keys = session_store.get_keys().await.unwrap_or_default();
                // Decrypt and exclude the primary key
                let primary_id = key_override.as_ref().map(|k| k.id.clone());
                all_keys.retain(|k| primary_id.as_deref() != Some(&k.id) && !k.is_any_oauth());
                all_keys.iter_mut().for_each(|k| {
                    let _ = SessionStore::decrypt_key_secret(k);
                });
                all_keys
            } else {
                Vec::new()
            };
            let command = ServeCommand::new(session_store.logs());
            command
                .execute(ServeParams {
                    port: serve_args.port,
                    host: serve_args.host,
                    key_override,
                    log: serve_args.log,
                    failover_keys,
                    cors: serve_args.cors,
                    timeout: serve_args.timeout,
                    auth_token: serve_args.auth_token,
                })
                .await
        }

        Commands::Info(info_args) => {
            let command = InfoCommand::new(session_store);
            command.execute(info_args.ping, info_args.json).await
        }

        Commands::Logs(logs_args) => {
            let command = LogsCommand::new(session_store);
            command.execute(logs_args).await
        }

        Commands::Stats(stats_args) => {
            let command = StatsCommand::new(session_store);
            command.execute(stats_args).await
        }

        Commands::Context(context_args) => {
            let command = ContextCommand::new();
            command.execute(context_args).await
        }

        Commands::McpServe(mcp_args) => {
            let command = McpServeCommand::new();
            command.execute(mcp_args).await
        }

        Commands::Update(update_args) if update_args.rollback => {
            commands::update::execute_rollback().await
        }

        Commands::Update(update_args) => match UpdateCommand::new() {
            Ok(command) => command.execute(update_args.force).await,
            Err(e) => {
                eprintln!(
                    "{} Failed to initialize update command: {}",
                    style::red("Error:"),
                    e
                );
                ExitCode::UserError
            }
        },
    };

    // Stop Ollama if aivo auto-started it during this session.
    services::ollama::stop_if_we_started();

    process::exit(exit_code.code());
}

/// Expands combined short flags (e.g. `-nar`) into individual flags (`-n`, `-a`, `-r`)
/// when every character after the leading `-` is a known single-character flag.
/// Tokens that don't match are left untouched (e.g. `-p8080`, `-mfoo`, `--model`).
/// Stops processing after a bare `--` separator.
fn expand_combined_short_flags(args: Vec<String>) -> Vec<String> {
    let known = collect_known_short_flags();
    let mut result = Vec::with_capacity(args.len());
    let mut past_separator = false;

    for arg in args {
        if past_separator {
            result.push(arg);
            continue;
        }

        if arg == "--" {
            past_separator = true;
            result.push(arg);
            continue;
        }

        if arg.starts_with('-')
            && !arg.starts_with("--")
            && arg.len() > 2
            && arg[1..].chars().all(|c| known.contains(&c))
        {
            for c in arg[1..].chars() {
                result.push(format!("-{c}"));
            }
        } else {
            result.push(arg);
        }
    }

    result
}

fn rewrite_cli_args(raw_args: Vec<String>) -> Vec<String> {
    if raw_args.len() <= 1 {
        return raw_args;
    }

    if TOOL_ALIASES.contains(&raw_args[1].as_str()) {
        let mut rewritten = vec![raw_args[0].clone(), "run".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        return rewritten;
    }

    if raw_args[1] == "use" {
        let mut rewritten = vec![raw_args[0].clone(), "keys".to_string(), "use".to_string()];
        rewritten.extend_from_slice(&raw_args[2..]);
        return rewritten;
    }

    if raw_args[1] == "ping" {
        let mut rewritten = vec![raw_args[0].clone(), "keys".to_string(), "ping".to_string()];
        rewritten.extend_from_slice(&raw_args[2..]);
        return rewritten;
    }

    if raw_args[1] == "-x" || raw_args[1] == "--execute" {
        let mut rewritten = vec![raw_args[0].clone(), "chat".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        return rewritten;
    }

    raw_args
}

/// Prints help information
fn print_help() {
    println!(
        "{} {} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION)),
        style::dim("— CLI for AI coding assistants")
    );
    println!();
    println!("{} aivo <command> [options]", style::bold("Usage:"));
    println!();
    println!("{}", style::bold("Commands:"));
    let print_cmd = |name: &str, desc: &str| {
        let padded = format!("{:<10}", name);
        println!("  {}{}", style::cyan(&padded), style::dim(desc));
    };
    print_cmd("run", "Launch AI tool, or use the saved start flow");
    print_cmd("chat", "Start the interactive chat TUI");
    print_cmd("serve", "Start a local OpenAI-compatible API server");
    print_cmd("keys", "Manage API keys (use, rm, add, cat, edit)");
    print_cmd("models", "List available models from the active provider");
    print_cmd("alias", "Create, list, or remove model aliases");
    print_cmd("info", "Show system info, keys, tools, and directory state");
    print_cmd("logs", "Show recent local logs from chat, run, and serve");
    print_cmd("stats", "Show usage statistics");
    print_cmd(
        "context",
        "Cross-CLI context — recent activity shared between tools",
    );
    print_cmd("update", "Update to the latest version");
    println!();
    println!("{}", style::bold("Shortcuts:"));
    let print_shortcut = |alias: &str, expansion: &str| {
        let padded = format!("{:<10}", alias);
        println!("  {}{}", style::cyan(&padded), style::dim(expansion));
    };
    print_shortcut("use", "keys use");
    print_shortcut("ping", "keys ping");
    print_shortcut("-x", "chat -x (one-shot; reads stdin when no value)");
    print_shortcut("claude/codex/gemini/opencode/pi", " run <tool>");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo claude -m kimi-k2.5"));
    println!("  {}", style::dim("aivo chat -x \"hello\""));
    println!(
        "  {}",
        style::dim("git diff | aivo -x \"summarize changes\"")
    );
    println!("  {}", style::dim("aivo gemini -k mykey -m minimax-m2.7"));
    println!("  {}", style::dim("aivo info --ping"));
    println!();
    println!("{}", style::bold("Options:"));
    let print_opt = |flag: &str, desc: &str| {
        println!(
            "  {}{}",
            style::cyan(format!("{:<16}", flag)),
            style::dim(desc)
        );
    };
    print_opt("-h, --help", "Display help information");
    print_opt("-v, --version", "Display the current version");
}

/// Prints the active selection (key, tool, model) at the bottom of help output.
async fn print_active_selection(session_store: &SessionStore) {
    let sel = match session_store.get_last_selection().await.ok().flatten() {
        Some(sel) => sel,
        None => return,
    };

    // Load config directly to get display name without triggering PBKDF2 decryption.
    let key_label = session_store
        .load()
        .await
        .ok()
        .and_then(|c| {
            c.api_keys
                .into_iter()
                .find(|k| k.id == sel.key_id)
                .map(|k| k.display_name().to_string())
        })
        .unwrap_or(sel.key_id.clone());
    let model_display = commands::models::model_display_label(sel.model.as_deref());

    println!();
    println!("{}", style::bold("Active key:"));
    println!(
        "  {} {}  {}",
        style::bullet_symbol(),
        key_label,
        style::dim(model_display),
    );
}

/// Prints version information
fn print_version() {
    println!(
        "{} {}",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION))
    );
}

/// Resolves a model alias if the model is a non-empty Some value.
/// Returns the original value unchanged if resolution fails or if it's None/empty (picker).
async fn resolve_model_alias(
    session_store: &SessionStore,
    model: Option<String>,
) -> Option<String> {
    match model {
        Some(ref m) if !m.is_empty() => match session_store.resolve_alias(m).await {
            Ok(resolved) => Some(resolved),
            Err(_) => model,
        },
        other => other,
    }
}

/// Result of extracting aivo-specific flags from clap's trailing passthrough args.
struct ExtractedFlags {
    model: Option<String>,
    key_flag: Option<String>,
    debug: bool,
    dry_run: bool,
    refresh: bool,
    as_name: Option<String>,
    env_strings: Vec<String>,
    remaining_args: Vec<String>,
    /// `None` = flag absent. `Some("")` = bare flag (interactive picker).
    /// `Some("id")` = explicit session id prefix.
    context: Option<String>,
}

/// Extracts aivo-owned flags (`--model`/`-m`, `--key`/`-k`, `--debug`, `--dry-run`, `--refresh`/`-r`, `--env`/`-e`) from
/// the passthrough `args` slice that clap's `trailing_var_arg` may have swallowed.
///
/// Flags already parsed by clap are supplied via `initial_*` parameters so that the
/// function produces a single consistent view regardless of where clap stopped.
#[allow(clippy::too_many_arguments)]
fn extract_aivo_flags(
    initial_model: Option<String>,
    initial_key: Option<String>,
    initial_debug: bool,
    initial_dry_run: bool,
    initial_refresh: bool,
    initial_as_name: Option<String>,
    initial_envs: Vec<String>,
    passthrough_args: &[String],
) -> ExtractedFlags {
    // Clap may have consumed a following flag as the value of -m/-k (e.g. `-m --resume`
    // gives model="--resume"). Detect and undo that by pushing the flag-like value back.
    let mut model = match initial_model {
        Some(m) if m.starts_with('-') => {
            // Will be pushed into remaining_args below via the passthrough loop seed
            // but we need it back in the stream — handled after the loop.
            Some((true, m)) // (is_flag_lookalike, value)
        }
        Some(m) => Some((false, m)),
        None => None,
    };
    let mut key_flag = match initial_key {
        Some(k) if k.starts_with('-') => Some((true, k)),
        Some(k) => Some((false, k)),
        None => None,
    };

    let mut debug = initial_debug;
    let mut dry_run = initial_dry_run;
    let mut refresh = initial_refresh;
    let mut as_name = initial_as_name;
    let mut context: Option<String> = None;
    let mut env_strings = initial_envs;
    let mut remaining_args: Vec<String> = Vec::new();

    // Flush flag-lookalike values back into remaining_args before processing passthrough.
    if let Some((true, ref v)) = model {
        remaining_args.push(v.clone());
        model = Some((false, String::new())); // empty → picker
    }
    if let Some((true, ref v)) = key_flag {
        remaining_args.push(v.clone());
        key_flag = Some((false, String::new()));
    }

    let mut model: Option<String> = model.map(|(_, v)| v);
    let mut key_flag: Option<String> = key_flag.map(|(_, v)| v);

    let mut i = 0;
    while i < passthrough_args.len() {
        let arg = &passthrough_args[i];
        if let Some(value) = arg.strip_prefix("--model=") {
            if !value.is_empty() && model.is_none() {
                model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if (arg == "--model" || arg == "-m") && model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                // --model with no value → trigger interactive picker
                model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--key=") {
            if !value.is_empty() && key_flag.is_none() {
                key_flag = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if (arg == "--key" || arg == "-k") && key_flag.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                key_flag = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                key_flag = Some(String::new());
            }
        } else if arg == "--debug" {
            debug = true;
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--refresh" || arg == "-r" {
            refresh = true;
        } else if arg == "--as" && i + 1 < passthrough_args.len() {
            as_name = Some(passthrough_args[i + 1].clone());
            i += 1;
        } else if let Some(value) = arg.strip_prefix("--as=") {
            if !value.is_empty() {
                as_name = Some(value.to_string());
            }
        } else if let Some(value) = arg
            .strip_prefix("--context=")
            .or_else(|| arg.strip_prefix("-c="))
        {
            if context.is_none() {
                context = Some(value.to_string());
            }
        } else if (arg == "--context" || arg == "-c") && context.is_none() {
            // Bare flag (no value): open the interactive picker.
            context = Some(String::new());
        } else if let Some(value) = arg
            .strip_prefix("--env=")
            .or_else(|| arg.strip_prefix("-e="))
        {
            if !value.is_empty() {
                env_strings.push(value.to_string());
            }
        } else if (arg == "--env" || arg == "-e") && i + 1 < passthrough_args.len() {
            env_strings.push(passthrough_args[i + 1].clone());
            i += 1;
        } else {
            remaining_args.push(arg.clone());
        }
        i += 1;
    }

    ExtractedFlags {
        model,
        key_flag,
        debug,
        dry_run,
        refresh,
        as_name,
        env_strings,
        remaining_args,
        context,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn model_inline_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--model=gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--model", "gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_short_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["-m", "gpt-4o"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn model_no_value_triggers_picker() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--model"]),
        );
        assert_eq!(r.model, Some(String::new()));
    }

    #[test]
    fn model_flag_as_value_corrected() {
        // Clap swallowed `--resume` as the value of -m
        let r = extract_aivo_flags(
            Some("--resume".to_string()),
            None,
            false,
            false,
            false,
            None,
            vec![],
            &[],
        );
        assert_eq!(r.model, Some(String::new())); // picker triggered
        assert_eq!(r.remaining_args, args(&["--resume"]));
    }

    #[test]
    fn model_already_set_passthrough_not_overwritten() {
        // clap parsed --model correctly; a second --model in passthrough should pass through
        let r = extract_aivo_flags(
            Some("gpt-4o".to_string()),
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--model", "other"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["--model", "other"]));
    }

    #[test]
    fn key_inline_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--key=mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--key", "mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_short_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["-k", "mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_flag_as_value_corrected() {
        let r = extract_aivo_flags(
            None,
            Some("--something".to_string()),
            false,
            false,
            false,
            None,
            vec![],
            &[],
        );
        assert_eq!(r.key_flag, Some(String::new()));
        assert_eq!(r.remaining_args, args(&["--something"]));
    }

    #[test]
    fn key_no_value_triggers_picker() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["-k"]),
        );
        assert_eq!(r.key_flag, Some(String::new()));
    }

    #[test]
    fn debug_flag() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--debug", "file.ts"]),
        );
        assert!(r.debug);
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn debug_already_set_preserved() {
        let r = extract_aivo_flags(None, None, true, false, false, None, vec![], &[]);
        assert!(r.debug);
    }

    #[test]
    fn dry_run_flag() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--dry-run"]),
        );
        assert!(r.dry_run);
    }

    #[test]
    fn env_inline_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--env=FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_inline_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["-e=FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--env", "FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_space_form() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["-e", "FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn initial_envs_preserved() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec!["PRE=1".to_string()],
            &args(&["-e", "POST=2"]),
        );
        assert_eq!(r.env_strings, vec!["PRE=1", "POST=2"]);
    }

    #[test]
    fn unknown_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--agent-name", "foo", "--resume"]),
        );
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "--resume"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn mixed_flags() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&[
                "--agent-name",
                "foo",
                "--model",
                "gpt-4o",
                "--debug",
                "file.ts",
            ]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert!(r.debug);
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "file.ts"]));
    }

    #[test]
    fn rewrite_injects_chat_for_top_level_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-x", "hello"])),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_chat_for_long_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "--execute", "hello"])),
            args(&["aivo", "chat", "--execute", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_explicit_chat() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "chat", "-x", "hello"])),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_tool_alias_precedence() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "claude", "--model", "gpt-5"])),
            args(&["aivo", "run", "claude", "--model", "gpt-5"])
        );
    }

    #[test]
    fn prompt_passes_through_extraction() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["fix the login bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn prompt_preserved_with_model_flag() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["--model", "gpt-4o", "fix the login bug"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
    }

    #[test]
    fn multi_word_unquoted_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            None,
            false,
            false,
            false,
            None,
            vec![],
            &args(&["fix", "the", "bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix", "the", "bug"]));
    }

    // --- expand_combined_short_flags tests ---

    #[test]
    fn expand_boolean_flags() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "stats", "-nar"])),
            args(&["aivo", "stats", "-n", "-a", "-r"])
        );
    }

    #[test]
    fn expand_does_not_touch_single_flag() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "-m"])),
            args(&["aivo", "-m"])
        );
    }

    #[test]
    fn expand_does_not_touch_long_flag() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "--model"])),
            args(&["aivo", "--model"])
        );
    }

    #[test]
    fn expand_leaves_unknown_chars_intact() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "chat", "-mfoo"])),
            args(&["aivo", "chat", "-mfoo"])
        );
    }

    #[test]
    fn expand_leaves_digit_value_intact() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "serve", "-p8080"])),
            args(&["aivo", "serve", "-p8080"])
        );
    }

    #[test]
    fn expand_stops_after_separator() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "run", "claude", "--", "-nar"])),
            args(&["aivo", "run", "claude", "--", "-nar"])
        );
    }

    #[test]
    fn expand_xr_full_pipeline() {
        let expanded = expand_combined_short_flags(args(&["aivo", "-xr", "hello"]));
        assert_eq!(expanded, args(&["aivo", "-x", "-r", "hello"]));
        let rewritten = rewrite_cli_args(expanded);
        assert_eq!(rewritten, args(&["aivo", "chat", "-x", "-r", "hello"]));
    }

    #[test]
    fn expand_mr() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "run", "claude", "-mr"])),
            args(&["aivo", "run", "claude", "-m", "-r"])
        );
    }

    #[test]
    fn expand_multiple_groups() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "stats", "-na", "-rs"])),
            args(&["aivo", "stats", "-n", "-a", "-r", "-s"])
        );
    }

    #[test]
    fn expand_preserves_surrounding_args() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "stats", "--search", "gpt", "-na"])),
            args(&["aivo", "stats", "--search", "gpt", "-n", "-a"])
        );
    }

    #[test]
    fn expand_global_hv() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo", "-hv"])),
            args(&["aivo", "-h", "-v"])
        );
    }

    #[test]
    fn expand_no_args() {
        assert_eq!(
            expand_combined_short_flags(args(&["aivo"])),
            args(&["aivo"])
        );
    }

    #[test]
    fn expand_empty() {
        let empty: Vec<String> = vec![];
        assert_eq!(expand_combined_short_flags(empty.clone()), empty);
    }
}
