//! Top-level CLI dispatch.
//!
//! Owns the process lifecycle (`process::exit`) so internal helpers can stay
//! `pub(crate)` instead of leaking onto the lib's public surface.

use std::process;

use clap::Parser;

use crate::cli::{self, Cli, Commands};
use crate::cli_args::{
    extract_aivo_flags, lift_context_suffix, needs_bundle_lookup, resolve_alias_in_memory,
    rewrite_cli_args,
};
use crate::commands::{
    self, AliasCommand, AudioCommand, ChatCommand, ContextCommand, ImageCommand, InfoCommand,
    KeysCommand, LogsCommand, ModelsCommand, RunCommand, ServeCommand, ServeParams, StartCommand,
    StartFlowArgs, StatsCommand, UpdateCommand, VideoCommand,
};
use crate::errors::ExitCode;
use crate::key_resolution::{
    KeyLookupMode, KeyResolution, key_or_exit, resolve_audio_key_override,
    resolve_image_key_override, resolve_key_override, resolve_video_key_override,
};
use crate::services::ai_launcher::AIToolType;
use crate::services::environment_injector::ClaudeSlotFlags;
use crate::services::key_compat::KeyCompatContext;
use crate::services::session_store::{AliasValue, BundleAlias};
use crate::services::{self, AILauncher, EnvironmentInjector, SessionStore};
use crate::{style, version};

/// Refuses to run a `__internal_test_fast_crypto`-built binary against real user config.
///
/// `__internal_test_fast_crypto` (documented in CLAUDE.md) uses reduced PBKDF2 iterations
/// so the test suite runs fast. A binary built with that feature derives a
/// different encryption key than a normal release binary, so it silently fails
/// to decrypt any real API key stored in `~/.config/aivo/config.json`. Shipping
/// or running such a binary as a CLI is always a mistake. Tests run library
/// code directly in tempdirs and don't hit this guard.
///
/// Override for intentional manual testing: `AIVO_TEST_FAST_CRYPTO_OK=1`.
#[cfg(feature = "__internal_test_fast_crypto")]
fn fast_crypto_guard() {
    if std::env::var_os("AIVO_TEST_FAST_CRYPTO_OK").is_some() {
        return;
    }
    eprintln!(
        "{} This aivo binary was built with the `__internal_test_fast_crypto` feature,\n       which uses reduced PBKDF2 iterations for fast tests.\n       It cannot decrypt keys encrypted by a normal aivo binary.\n\n       Rebuild without the feature: {}\n       Or, to override intentionally, set {}",
        style::red("error:"),
        style::cyan("cargo build"),
        style::dim("AIVO_TEST_FAST_CRYPTO_OK=1"),
    );
    process::exit(ExitCode::UserError.code());
}

#[cfg(not(feature = "__internal_test_fast_crypto"))]
fn fast_crypto_guard() {}

/// If `--debug` was passed on the CLI, resolve its path (default vs explicit)
/// and initialize the global HTTP debug logger so subsequent `.send_logged()`
/// calls capture request/response details. Prints the resolved log path to
/// stderr; on open failure, warns and continues without logging.
async fn maybe_init_http_debug(value: &Option<String>) {
    let Some(raw) = value else {
        return;
    };
    let path = if raw.is_empty() {
        services::http_debug::default_log_path()
    } else {
        std::path::PathBuf::from(raw)
    };
    match services::http_debug::init(path).await {
        Ok(p) => eprintln!("[aivo] HTTP debug log → {}", p.display()),
        Err(e) => {
            eprintln!("[aivo] failed to open debug log: {e}; HTTP requests will not be logged")
        }
    }
}

/// CLI entry point. Parses argv, routes to the matching command handler, and
/// exits the process with the resulting code. Never returns.
pub async fn run() -> ! {
    fast_crypto_guard();
    let raw_args: Vec<String> = std::env::args().collect();

    // Initialize services. Bundle aliases need to be loaded *before* CLI
    // parsing because `aivo <bundle>` and `aivo run <bundle>` are expanded by
    // `rewrite_cli_args` ahead of clap.
    let session_store = SessionStore::new();
    let bundle_index = if needs_bundle_lookup(&raw_args) {
        load_bundle_index(&session_store).await
    } else {
        std::collections::HashMap::new()
    };

    let args = Cli::parse_from(rewrite_cli_args(raw_args, &bundle_index));

    // Handle --version and subcommand --help early, before any further service initialization.
    if args.version {
        print_version();
        process::exit(0);
    }

    let models_cache = services::ModelsCache::new();

    if args.help
        && let Some(cmd) = &args.command
    {
        match cmd {
            Commands::Run(run_args) => RunCommand::print_help(run_args.tool.as_deref()),
            Commands::Keys(_) => KeysCommand::print_help(),
            Commands::Chat(_) => ChatCommand::print_help(),
            Commands::Image(_) => {
                ImageCommand::print_help();
                ImageCommand::print_active_selection(&session_store).await;
            }
            Commands::Speak(_) => {
                AudioCommand::print_help();
                AudioCommand::print_active_selection(&session_store).await;
            }
            Commands::Video(_) => {
                VideoCommand::print_help();
                VideoCommand::print_active_selection(&session_store).await;
            }
            Commands::Models(_) => ModelsCommand::print_help(),
            Commands::Serve(_) => ServeCommand::print_help(),
            Commands::Alias(_) => AliasCommand::print_help(),
            Commands::Info(_) => InfoCommand::print_help(),
            Commands::Logs(_) => LogsCommand::print_help(),
            Commands::Stats(_) => StatsCommand::print_help(),
            Commands::Update(_) => UpdateCommand::print_help(),
            Commands::Context(_) => ContextCommand::print_help(),
        }
        process::exit(0);
    }

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
            maybe_init_http_debug(&chat_args.debug).await;
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

        Commands::Image(image_args) => {
            // No prompt → short-circuit before any key resolution so the
            // user gets help + active-selection footer without a forced
            // key picker. Mirrors the bare `aivo` and `aivo image -h` paths.
            if image_args
                .prompt
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .is_none()
            {
                ImageCommand::print_help();
                ImageCommand::print_active_selection(&session_store).await;
                process::exit(ExitCode::Success.code());
            }

            let key_override = key_or_exit(
                resolve_image_key_override(
                    &session_store,
                    image_args.key.as_deref(),
                    KeyLookupMode::RequireActiveOrPrompt,
                    KeyCompatContext::Image,
                )
                .await,
            );
            let initial_key = match key_override {
                Some(k) => k,
                None => {
                    eprintln!(
                        "{} No API key available for image generation.",
                        style::red("Error:")
                    );
                    process::exit(ExitCode::AuthError.code());
                }
            };
            // resolve_key_override only annotates the picker; active-key /
            // last-used / explicit `--key` paths bypass the annotation, so
            // we have to re-check compat here and offer a swap if the
            // resolved key can't actually talk to /v1/images/generations.
            let key = match ensure_compatible_key(
                &session_store,
                initial_key,
                KeyCompatContext::Image,
                "image generation",
            )
            .await
            {
                Some(k) => k,
                None => process::exit(ExitCode::UserError.code()),
            };
            let command = ImageCommand::new(session_store, models_cache.clone());
            command.execute(image_args, key).await
        }

        Commands::Speak(audio_args) => {
            audio_dispatch(&session_store, &models_cache, audio_args).await
        }

        Commands::Video(video_args) => {
            video_dispatch(&session_store, &models_cache, video_args).await
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
                ClaudeSlotFlags {
                    reasoning: run_args.reasoning_model,
                    subagent: run_args.subagent_model,
                    haiku: run_args.haiku_model,
                    sonnet: run_args.sonnet_model,
                    opus: run_args.opus_model,
                },
                run_args.key,
                run_args.debug,
                run_args.dry_run,
                run_args.refresh,
                run_args.relogin,
                run_args.envs,
                // `--1m`/`--2m` shorthands collapsed into max_context up
                // front so every downstream consumer sees a single signal.
                run_args
                    .max_context
                    .or_else(|| run_args.one_m.then(|| "1m".to_string()))
                    .or_else(|| run_args.two_m.then(|| "2m".to_string())),
                &run_args.args,
            );
            // After extract_aivo_flags so `--debug` after the tool name
            // (recovered from passthrough) activates the logger too.
            maybe_init_http_debug(&extracted.debug).await;
            // Resolve aliases for main + 6 slot models against a single
            // in-memory snapshot of the alias map, instead of paying one disk
            // read per call (worst case 7).
            let aliases = session_store.get_aliases().await.unwrap_or_default();
            let resolve = |m: Option<String>| resolve_alias_in_memory(&aliases, m);
            let resolved_model = resolve(extracted.model);
            let slots = ClaudeSlotFlags {
                reasoning: resolve(extracted.slots.reasoning),
                subagent: resolve(extracted.slots.subagent),
                haiku: resolve(extracted.slots.haiku),
                sonnet: resolve(extracted.slots.sonnet),
                opus: resolve(extracted.slots.opus),
            };
            // Normalize `-m foo[1m]`/`-m foo[2m]` (and any alias that expands
            // to one) into the same internal state as `-m foo --1m`/`--2m`.
            // Without this, mixing the two — `-m foo[1m] --1m` — would
            // double-append the suffix and the env injector would emit
            // `foo[1m][1m]`.
            let (model, max_context) = lift_context_suffix(resolved_model, extracted.max_context);
            // Validate --max-context before any picker UI runs (key picker,
            // model picker). Otherwise the user picks a key + model and
            // *then* gets told their flag is invalid.
            if let Some(value) = max_context.as_deref() {
                if !matches!(value, "1m" | "2m") {
                    eprintln!(
                        "{} --max-context only accepts '1m' or '2m' (got {:?}).",
                        style::red("Error:"),
                        value
                    );
                    process::exit(ExitCode::UserError.code());
                }
                let tool_is_claude = run_args
                    .tool
                    .as_deref()
                    .and_then(AIToolType::parse)
                    .is_some_and(|t| matches!(t, AIToolType::Claude));
                if !tool_is_claude {
                    let tool_name = run_args.tool.as_deref().unwrap_or("(none)");
                    eprintln!(
                        "{} --max-context only applies to `aivo run claude`. {} doesn't have a context-bar issue.",
                        style::red("Error:"),
                        tool_name
                    );
                    process::exit(ExitCode::UserError.code());
                }
            }
            let key_flag = extracted.key_flag;
            let dry_run = extracted.dry_run;
            let refresh = extracted.refresh;
            let relogin = extracted.relogin;
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

                // --relogin: drive the OAuth flow up front so the launch sees
                // a fresh credential. Done after key resolution (so the user
                // sees the picker if they didn't pass `-k`) and before any
                // model picker (so a stale token doesn't lead to a model list
                // the user can't actually use).
                let key_override = if relogin {
                    let Some(key) = key_override else {
                        eprintln!(
                            "{} --relogin requires a key — none selected.",
                            style::red("Error:")
                        );
                        process::exit(ExitCode::UserError.code());
                    };
                    match services::oauth_relogin::relogin_key(&session_store, &key).await {
                        Ok(updated) => Some(updated),
                        Err(e) => {
                            eprintln!("{} {e}", style::red("Error:"));
                            process::exit(ExitCode::UserError.code());
                        }
                    }
                } else {
                    key_override
                };

                // Resolve model using last selection when no explicit flags given.
                // When -k is used without -m, normally force the model picker —
                // except under --dry-run, where the user just wants to see what
                // would launch without being forced into an interactive prompt.
                let model_flag_explicit = model.is_some();
                let last_sel_for_key = session_store
                    .get_last_selection()
                    .await
                    .ok()
                    .flatten()
                    .filter(|sel| key_override.as_ref().is_some_and(|k| k.id == sel.key_id));
                // First-run-only fallback: when the user has just installed
                // aivo and hasn't added any of their own keys yet (only the
                // auto-created `aivo-starter` key exists, and it's active),
                // skip the model picker and use the seeded chat model
                // (`aivo/starter` from `ensure_starter_key`). Once the user
                // adds another key, the fallback no longer applies and the
                // picker resumes its normal behavior.
                let persisted_model_for_key =
                    match last_sel_for_key.as_ref().and_then(|sel| sel.model.clone()) {
                        Some(m) => Some(m),
                        None => {
                            if let Some(k) = key_override.as_ref()
                                && services::provider_profile::is_aivo_starter_base(&k.base_url)
                                && session_store
                                    .get_keys()
                                    .await
                                    .map(|keys| keys.len() == 1)
                                    .unwrap_or(false)
                            {
                                session_store.get_chat_model(&k.id).await.ok().flatten()
                            } else {
                                None
                            }
                        }
                    };
                let model = if model.is_some() {
                    model
                } else if dry_run {
                    // --dry-run never opens a picker. Reuse persisted model for
                    // this key if any; otherwise let the tool's default surface
                    // in the dry-run output.
                    persisted_model_for_key
                } else if key_explicit {
                    // -k used without -m → force model picker
                    Some(String::new())
                } else {
                    // Same key as last time → reuse saved model (could be
                    // __default__); otherwise empty string triggers the picker.
                    persisted_model_for_key.or(Some(String::new()))
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
                        dry_run,
                        refresh,
                        model,
                        model_flag_explicit,
                        slots,
                        env,
                        key_override,
                        context_selector,
                        max_context,
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

/// Snapshot of Bundle aliases for `rewrite_cli_args`. Errors during load are
/// swallowed — a startup-time alias read failure must never stop the user
/// from invoking a non-bundle command.
async fn load_bundle_index(
    session_store: &SessionStore,
) -> std::collections::HashMap<String, BundleAlias> {
    session_store
        .list_alias_values()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(k, v)| match v {
            AliasValue::Bundle(b) => Some((k, b)),
            AliasValue::Model(_) => None,
        })
        .collect()
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
    print_cmd("keys", "Manage API keys (use, rm, add, cat, edit)");
    print_cmd("models", "List available models from the active provider");
    print_cmd("chat", "Start the interactive chat TUI");
    print_cmd("serve", "Start a local OpenAI-compatible API server");
    print_cmd("alias", "Create, list, or remove model aliases");
    print_cmd("info", "Show system info, keys, tools, and directory state");
    print_cmd("logs", "Show recent local logs from chat, run, and serve");
    print_cmd("stats", "Show usage statistics");
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
    println!("  {}", style::dim("aivo claude -k aivo"));
    println!("  {}", style::dim("aivo -x \"hello\""));
    println!(
        "  {}",
        style::dim("git diff | aivo -x \"summarize changes\"")
    );
    println!("  {}", style::dim("aivo gemini -k mykey -m minimax-m2.7"));
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

/// Re-checks media-modality key compatibility against a concrete key
/// *after* `resolve_key_override` returns. The picker path already annotates
/// incompatible entries, but the active-key and explicit-`--key` paths skip
/// that, so a user with an OAuth / Copilot / Ollama / Anthropic key can
/// reach the command. Returns the original key if already compatible, or
/// prompts for a compatible one. Returns `None` when the user cancels or
/// no compatible keys exist.
///
/// `label` is the user-visible activity name ("image generation",
/// "audio generation", "video generation") interpolated into the
/// rejection messages.
async fn ensure_compatible_key(
    session_store: &SessionStore,
    key: services::session_store::ApiKey,
    compat: KeyCompatContext,
    label: &str,
) -> Option<services::session_store::ApiKey> {
    use std::io::IsTerminal;

    let reason = match compat.incompat_reason(&key) {
        Some(r) => r,
        None => return Some(key),
    };

    let all_keys = match session_store.get_keys().await {
        Ok(k) => k,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            return None;
        }
    };
    let annotations = compat.annotations_for(&all_keys);
    let has_eligible = annotations.iter().any(Option::is_none);

    if !has_eligible {
        eprintln!(
            "{} Key '{}' can't be used for {} ({}).",
            style::red("Error:"),
            key.display_name(),
            label,
            reason
        );
        eprintln!(
            "  {} Add an OpenAI-compatible key with `aivo keys add`.",
            style::dim("hint:")
        );
        return None;
    }

    if !std::io::stderr().is_terminal() {
        eprintln!(
            "{} Key '{}' can't be used for {} ({}). Pass `--key <id|name>` to pick another.",
            style::red("Error:"),
            key.display_name(),
            label,
            reason
        );
        return None;
    }

    eprintln!(
        "{} Key '{}' can't be used for {} ({}) — pick a compatible key.",
        style::yellow("Note:"),
        key.display_name(),
        label,
        reason
    );
    match commands::keys::prompt_pick_key_without_activation(
        &all_keys,
        &annotations,
        "Select a key",
        0,
    ) {
        Ok(Some(picked)) => Some(picked),
        Ok(None) => None,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            None
        }
    }
}

/// Dispatch for `Commands::Speak`. Resolves the prompt from positional
/// arg / `--file` / piped stdin (in that precedence) before any key or
/// model work — so a no-prompt invocation in an interactive shell prints
/// help instead of triggering a picker.
async fn audio_dispatch(
    session_store: &SessionStore,
    models_cache: &services::ModelsCache,
    audio_args: cli::AudioArgs,
) -> ExitCode {
    // List mode is purely local: no provider call, no key resolution.
    // Route it before we try to resolve a prompt or a key.
    if audio_args.list {
        let command = AudioCommand::new(session_store.clone(), models_cache.clone());
        return command.run_list().await;
    }

    let prompt = match resolve_speak_prompt(&audio_args) {
        Ok(Some(p)) => p,
        Ok(None) => {
            AudioCommand::print_help();
            AudioCommand::print_active_selection(session_store).await;
            process::exit(ExitCode::Success.code());
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            process::exit(ExitCode::UserError.code());
        }
    };

    let key_override = key_or_exit(
        resolve_audio_key_override(
            session_store,
            audio_args.key.as_deref(),
            KeyLookupMode::RequireActiveOrPrompt,
            KeyCompatContext::Audio,
        )
        .await,
    );
    let initial_key = match key_override {
        Some(k) => k,
        None => {
            eprintln!(
                "{} No API key available for audio generation.",
                style::red("Error:")
            );
            process::exit(ExitCode::AuthError.code());
        }
    };
    let key = match ensure_compatible_key(
        session_store,
        initial_key,
        KeyCompatContext::Audio,
        "audio generation",
    )
    .await
    {
        Some(k) => k,
        None => process::exit(ExitCode::UserError.code()),
    };
    let command = AudioCommand::new(session_store.clone(), models_cache.clone());
    command.execute(audio_args, key, prompt).await
}

/// Resolves the speak prompt from (positional, `--file`, piped stdin) in
/// that precedence. `--file -` or `--file` with no value reads stdin
/// explicitly. Returns `Ok(None)` to mean "show help" — i.e. the caller
/// had no positional, no `--file`, and stdin was a TTY or empty.
fn resolve_speak_prompt(args: &cli::AudioArgs) -> anyhow::Result<Option<String>> {
    if let Some(p) = args
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Ok(Some(p.to_string()));
    }
    if let Some(path) = args.file.as_deref() {
        return if commands::audio::is_stdin_file_arg(path) {
            commands::audio::read_prompt_stdin_explicit().map(Some)
        } else {
            commands::audio::read_prompt_file(std::path::Path::new(path)).map(Some)
        };
    }
    match services::stdin_io::read_stdin_if_piped()? {
        Some(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        None => Ok(None),
    }
}

/// Dispatch for `Commands::Video`. Mirrors `audio_dispatch` but does *not*
/// short-circuit on empty prompt when `--job-id` is set — that's the
/// recovery path and a prompt isn't required for it.
async fn video_dispatch(
    session_store: &SessionStore,
    models_cache: &services::ModelsCache,
    video_args: cli::VideoArgs,
) -> ExitCode {
    let has_prompt = video_args
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .is_some();
    if !has_prompt && video_args.job_id.is_none() {
        VideoCommand::print_help();
        VideoCommand::print_active_selection(session_store).await;
        process::exit(ExitCode::Success.code());
    }

    let key_override = key_or_exit(
        resolve_video_key_override(
            session_store,
            video_args.key.as_deref(),
            KeyLookupMode::RequireActiveOrPrompt,
            KeyCompatContext::Video,
        )
        .await,
    );
    let initial_key = match key_override {
        Some(k) => k,
        None => {
            eprintln!(
                "{} No API key available for video generation.",
                style::red("Error:")
            );
            process::exit(ExitCode::AuthError.code());
        }
    };
    let key = match ensure_compatible_key(
        session_store,
        initial_key,
        KeyCompatContext::Video,
        "video generation",
    )
    .await
    {
        Some(k) => k,
        None => process::exit(ExitCode::UserError.code()),
    };
    let command = VideoCommand::new(session_store.clone(), models_cache.clone());
    command.execute(video_args, key).await
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
