//! Top-level CLI dispatch.
//!
//! Owns the process lifecycle (`process::exit`) so internal helpers can stay
//! `pub(crate)` instead of leaking onto the lib's public surface.

use std::process;

use clap::{Command, CommandFactory, Parser};
use serde_json::{Map, Value, json};

use crate::cli::{self, Cli, Commands};
use crate::cli_args::{
    extract_aivo_flags, lift_context_suffix, needs_bundle_lookup, parse_context_token,
    resolve_alias_in_memory, rewrite_cli_args,
};
use crate::commands::{
    self, AliasCommand, AudioCommand, ChatCommand, ImageCommand, InfoCommand, KeysCommand,
    LogsCommand, ModelsCommand, RunCommand, ServeCommand, ServeParams, ShareCommand, StartCommand,
    StartFlowArgs, StatsCommand, UpdateCommand, VideoCommand,
};
use crate::errors::ExitCode;
use crate::key_resolution::{
    KeyLookupMode, KeyResolution, key_or_exit, resolve_audio_key_override,
    resolve_image_key_override, resolve_key_override, resolve_key_override_info,
    resolve_video_key_override,
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
/// Outer-dispatcher predicate for skipping key resolution; the HF flow
/// synthesizes its own loopback `ApiKey`.
fn is_hf_takeover(model: Option<&str>) -> bool {
    use services::huggingface;
    match model {
        Some(m) => huggingface::is_hf_or_local_gguf(m) || huggingface::is_bare_hf_picker_trigger(m),
        None => false,
    }
}

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
    // Must run before any reqwest client is built or `spawn_blocking` is
    // launched — reqwest snapshots proxy env at construction, and env
    // mutation is only race-free while the current-thread runtime is idle.
    services::launch_runtime::ensure_loopback_no_proxy_in_process_env();
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

    if args.help_json {
        print_help_json();
        process::exit(0);
    }

    let models_cache = services::ModelsCache::new();

    if args.help
        && let Some(cmd) = &args.command
    {
        match cmd {
            Commands::Run(run_args) => RunCommand::print_help(run_args.tool.as_deref()),
            Commands::Keys(keys_args) => KeysCommand::print_help(keys_args.action.as_deref()),
            Commands::Chat(_) => ChatCommand::print_help(),
            Commands::Image(_) => {
                ImageCommand::print_help();
                ImageCommand::print_active_selection(&session_store).await;
            }
            Commands::Audio(_) => {
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
            Commands::Logs(logs_args) => LogsCommand::print_help(logs_args.action.as_deref()),
            Commands::Stats(_) => StatsCommand::print_help(),
            Commands::Update(_) => UpdateCommand::print_help(),
            Commands::Amp(amp_args) => {
                crate::commands::AmpCommand::print_help(amp_args.action.as_deref())
            }
            Commands::Hf(_) => crate::commands::hf::HfCommand::print_help(),
            Commands::Share(_) => ShareCommand::print_help(),
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
            // Reject non-HF, non-local-path positionals up-front, matching `aivo serve`.
            if let Some(ref raw) = chat_args.reference
                && !is_hf_takeover(Some(raw.as_str()))
            {
                eprintln!(
                    "{} `aivo chat <REF>` only accepts a HuggingFace ref (`hf:<owner>/<repo>` or `https://huggingface.co/...`) or a local `.gguf` path. Got: {}",
                    style::red("Error:"),
                    raw,
                );
                eprintln!(
                    "  {}",
                    style::dim("For a remote provider, pass `--model <name>` instead."),
                );
                process::exit(ExitCode::UserError.code());
            }
            // Positional lifts into model; explicit -m still wins.
            let model_input = chat_args
                .model
                .clone()
                .or_else(|| chat_args.reference.clone());
            // Expand alias before the HF check so `-m <alias-to-hf-ref>`
            // takes the HF path. `run`'s flow does the same.
            let expanded_model = resolve_model_alias(&session_store, model_input).await;
            let key_override = if is_hf_takeover(expanded_model.as_deref()) {
                None
            } else {
                key_or_exit(
                    resolve_key_override(
                        &session_store,
                        chat_args.key.as_deref(),
                        KeyLookupMode::RequireActiveOrPrompt,
                        KeyCompatContext::Chat,
                    )
                    .await,
                )
            };
            // When -k is used without -m, force model picker (same as run/start)
            let model = if chat_args.model.is_some() || chat_args.reference.is_some() {
                expanded_model
            } else if key_explicit {
                Some(String::new())
            } else {
                None
            };
            let command = ChatCommand::new(session_store, models_cache.clone());
            command
                .execute(
                    model,
                    chat_args.prompt,
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

        Commands::Audio(audio_args) => {
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
                services::environment_injector::AmpModeModels {
                    rush: run_args.rush_model,
                    smart: run_args.smart_model,
                    deep: run_args.deep_model,
                    large: run_args.large_model,
                    disable_tools: run_args.disable_tool,
                    initial_mode: run_args.mode,
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
            services::transform_mode::set_active(run_args.transform || extracted.transform);
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
            // Validate shape (must be `<digits>m`) and tool restriction before
            // any picker UI runs — otherwise the user picks a key + model and
            // *then* gets told their flag is malformed. The tier itself
            // (1m/2m/12m/…) isn't validated; aivo passes whatever the user
            // gave through to Claude as a `[<N>m]` model-name suffix.
            let max_context = if let Some(value) = max_context.as_deref() {
                let Some(canonical) = parse_context_token(value) else {
                    eprintln!(
                        "{} --max-context expects a value like '1m' or '12m' (got {:?}).",
                        style::red("Error:"),
                        value
                    );
                    process::exit(ExitCode::UserError.code());
                };
                let parsed_tool = run_args.tool.as_deref().and_then(AIToolType::parse);
                let supported = parsed_tool
                    .is_some_and(|t| matches!(t, AIToolType::Claude | AIToolType::Codex));
                if !supported {
                    let tool_name = run_args.tool.as_deref().unwrap_or("(none)");
                    if parsed_tool == Some(AIToolType::Amp) {
                        eprintln!(
                            "{} --max-context / --1m / --2m don't apply to `aivo run amp`. Use `--mode large` for amp's built-in 1M-context tier, or `--smart-model` / `--large-model` to swap the upstream model.",
                            style::red("Error:"),
                        );
                    } else {
                        eprintln!(
                            "{} --max-context only applies to `aivo run claude` and `aivo run codex` (got {}).",
                            style::red("Error:"),
                            tool_name
                        );
                    }
                    process::exit(ExitCode::UserError.code());
                }
                Some(canonical)
            } else {
                None
            };
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
                let command =
                    RunCommand::new(session_store.clone(), ai_launcher, models_cache.clone());

                let key_explicit = key_flag.is_some();
                let compat = run_args
                    .tool
                    .as_deref()
                    .and_then(AIToolType::parse)
                    .map(KeyCompatContext::Tool)
                    .unwrap_or(KeyCompatContext::None);
                let key_override = if is_hf_takeover(model.as_deref()) {
                    None
                } else {
                    key_or_exit(
                        resolve_key_override(
                            &session_store,
                            key_flag.as_deref(),
                            KeyLookupMode::RequireActiveOrPrompt,
                            compat,
                        )
                        .await,
                    )
                };

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
                // Drop a persisted starter model that's been removed from the
                // server catalog so the picker reopens with the current list.
                // Skipped when --model was explicit — we trust what the user
                // typed and let upstream surface any mismatch.
                let persisted_model_for_key = if model_flag_explicit {
                    persisted_model_for_key
                } else {
                    match (persisted_model_for_key, key_override.as_ref()) {
                        (Some(m), Some(k))
                            if services::provider_profile::is_aivo_starter_base(&k.base_url) =>
                        {
                            let client = services::http_utils::router_http_client();
                            if commands::models::starter_model_still_available(
                                &client,
                                k,
                                &models_cache,
                                &m,
                            )
                            .await
                            {
                                Some(m)
                            } else {
                                eprintln!(
                                    "{} Model '{}' is no longer available on aivo-starter. Pick another:",
                                    style::yellow("Note:"),
                                    m
                                );
                                None
                            }
                        }
                        (other, _) => other,
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

                let amp_modes = services::environment_injector::AmpModeModels {
                    rush: resolve(extracted.amp_modes.rush),
                    smart: resolve(extracted.amp_modes.smart),
                    deep: resolve(extracted.amp_modes.deep),
                    large: resolve(extracted.amp_modes.large),
                    disable_tools: extracted.amp_modes.disable_tools,
                    initial_mode: extracted.amp_modes.initial_mode,
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
                        amp_modes,
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
                resolve_key_override_info(
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
            // Reject non-HF positionals up-front so a typo doesn't fall
            // through to "no key" mode.
            if let Some(ref raw) = serve_args.reference
                && !is_hf_takeover(Some(raw.as_str()))
            {
                eprintln!(
                    "{} `aivo serve <REF>` only accepts a HuggingFace ref (`hf:<owner>/<repo>` or `https://huggingface.co/...`). Got: {}",
                    style::red("Error:"),
                    raw,
                );
                eprintln!(
                    "  {}",
                    style::dim("For a remote provider, omit the positional and use `-k <key>`."),
                );
                process::exit(ExitCode::UserError.code());
            }
            let hf_mode = serve_args.reference.is_some();
            if hf_mode {
                // Warn before download starts so it lands above the spinner.
                if serve_args.key.is_some() {
                    eprintln!(
                        "  {} -k / --key is ignored when a local model is specified",
                        style::yellow("!"),
                    );
                }
                if serve_args.failover {
                    eprintln!(
                        "  {} --failover is ignored when a local model is specified",
                        style::yellow("!"),
                    );
                }
            }
            let hf_takeover_key = if hf_mode {
                let raw = serve_args.reference.as_deref().unwrap_or("");
                let hf_ref = match services::huggingface::parse_hf_ref(raw) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        process::exit(ExitCode::UserError.code());
                    }
                };
                match services::huggingface::ensure_ready(&hf_ref).await {
                    Ok(port) => {
                        let base_url = services::huggingface::local_openai_base_url(port);
                        Some(services::session_store::ApiKey::new_with_protocol(
                            "aivo-hf-local".to_string(),
                            format!("hf:{}", hf_ref.repo),
                            base_url,
                            None,
                            "huggingface".to_string(),
                        ))
                    }
                    Err(e) => {
                        eprintln!("{} {}", style::red("Error:"), e);
                        // ensure_ready may have stored the child in
                        // SERVER_CHILD before its health-check failed;
                        // `process::exit` skips destructors so we have
                        // to kill the orphan explicitly.
                        services::huggingface::stop_if_we_started();
                        process::exit(ExitCode::UserError.code());
                    }
                }
            } else {
                None
            };

            let key_override = if let Some(key) = hf_takeover_key {
                Some(key)
            } else {
                match resolve_key_override(
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
                }
            };
            let failover_keys = if serve_args.failover && !hf_mode {
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
            // Snapshot aliases at startup; edits made while serve is running
            // require a restart (matches `aivo run`'s behavior).
            let aliases = session_store.get_aliases().await.unwrap_or_default();
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
                    aliases,
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

        Commands::Amp(amp_args) => {
            use crate::commands::AmpCommand;
            let command = AmpCommand::new();
            command.execute(amp_args).await
        }

        Commands::Hf(hf_args) => {
            let command = crate::commands::hf::HfCommand::new();
            command.execute(hf_args).await
        }

        Commands::Share(share_args) => {
            let command = ShareCommand::new(session_store);
            command.execute(share_args).await
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
    // Stop llama-server if aivo auto-started it for a HuggingFace run.
    services::huggingface::stop_if_we_started();

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
        "{} {} — CLI for AI coding assistants",
        style::cyan("aivo"),
        style::dim(format!("v{}", version::VERSION)),
    );
    println!();
    println!("{} aivo <command> [options]", style::bold("Usage:"));
    println!();

    println!("{}", style::bold("Commands:"));
    let print_cmd = |name: &str, desc: &str| {
        println!("  {}  {}", style::cyan(format!("{:<8}", name)), desc);
    };
    print_cmd("run", "Launch an AI tool, or the saved start flow");
    print_cmd("keys", "Manage API keys");
    print_cmd("chat", "Start the interactive chat TUI");
    print_cmd("models", "List available models from the active provider");
    print_cmd("serve", "Start a local OpenAI-compatible API server");
    print_cmd("alias", "Create, list, or remove model aliases");
    print_cmd("hf", "Manage cached HuggingFace GGUF files");
    print_cmd("logs", "Show recent local logs from chat, run, and serve");
    print_cmd("stats", "Show usage statistics");
    print_cmd("update", "Update to the latest version");
    println!();

    println!("{}", style::bold("Shortcuts:"));
    let shortcuts: &[(&str, &str, &str)] = &[
        ("use", "keys use", "aivo keys use --help"),
        ("ping", "keys ping", "aivo keys ping --help"),
        ("share", "logs share", "aivo logs share --help"),
        ("hf:/url", "chat <ref>", "open chat with a local HF model"),
        (
            "<tool>",
            "run <tool>",
            "claude / codex / gemini / opencode / pi / amp",
        ),
    ];
    let expansion_width = shortcuts.iter().map(|(_, e, _)| e.len()).max().unwrap_or(0);
    for (alias, expansion, hint) in shortcuts {
        let alias_col = style::cyan(format!("{alias:<8}"));
        let hint_col = style::dim(format!("({hint})"));
        println!("  {alias_col}  {expansion:<expansion_width$}  {hint_col}");
    }
    println!();

    println!("{}", style::bold("Examples:"));
    for cmd in [
        "aivo \"tell me a short story\"",
        "aivo pi -k openrouter",
        "git diff | aivo \"summarize changes\"",
        "aivo hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    ] {
        println!("  {}", style::dim(cmd));
    }
    println!();

    println!("{}", style::bold("Options:"));
    let print_opt = |flag: &str, desc: &str| {
        println!("  {}  {}", style::bold(format!("{:<13}", flag)), desc);
    };
    print_opt("-h, --help", "Display help information");
    print_opt("--help-json", "Dump the full command tree as JSON");
    print_opt("-v, --version", "Display the current version");
}

/// Prints the active selection (key, tool, model) at the bottom of help output.
///
/// One config load resolves both the last-selection record and the key's
/// display name (no PBKDF2). Skips the stale-record self-healing that
/// `get_last_selection` does — that path will heal on the next command
/// that actually touches selection state.
async fn print_active_selection(session_store: &SessionStore) {
    let Some(config) = session_store.load().await.ok() else {
        return;
    };
    let Some(sel) = config.last_selection.clone() else {
        return;
    };
    let key_entry = config.api_keys.iter().find(|k| k.id == sel.key_id);
    // Treat a missing or moved key as "no active selection" — mirrors
    // `get_last_selection`'s stale check without rewriting the config here.
    if key_entry.is_none_or(|k| k.base_url != sel.base_url) {
        return;
    }
    let key_label = key_entry
        .map(|k| k.display_name().to_string())
        .unwrap_or_else(|| sel.key_id.clone());
    let model_display = commands::models::model_display_label(sel.model.as_deref());
    // HF models bypass the API key entirely (local llama-server with a
    // synthetic loopback key). Showing `key  hf:...` implies a coupling that
    // doesn't exist at runtime, so swap the line out for an HF-specific one.
    let model_is_hf = sel
        .model
        .as_deref()
        .is_some_and(services::huggingface::is_huggingface_ref);

    println!();
    println!("{}", style::bold("Active key:"));
    if model_is_hf {
        println!(
            "  {} {}  {}  {}",
            style::bullet_symbol(),
            style::dim(model_display),
            style::dim("(local, no API key)"),
            style::dim("(change with: aivo use)"),
        );
    } else {
        println!(
            "  {} {}  {}  {}",
            style::bullet_symbol(),
            key_label,
            style::dim(model_display),
            style::dim("(change with: aivo use)"),
        );
    }
}

/// Dump the entire CLI command tree (commands, flags, descriptions, env hints)
/// as JSON on stdout. Intended for AI agents / tooling that needs reliable
/// machine-readable command discovery — human help text is easy to misparse.
fn print_help_json() {
    let cmd = Cli::command();
    let tree = serialize_command(&cmd);
    let payload = json!({
        "name": "aivo",
        "version": version::VERSION,
        "shortcuts": [
            { "alias": "use", "expands_to": ["keys", "use"] },
            { "alias": "ping", "expands_to": ["keys", "ping"] },
            { "alias": "share", "expands_to": ["logs", "share"] },
            { "alias": "-p", "expands_to": ["chat", "-p"] },
            { "alias": "-x", "expands_to": ["chat", "-x"], "deprecated": true, "replaced_by": "-p" },
            { "alias": "<text>", "expands_to": ["chat", "-p", "<text>"], "note": "Any non-subcommand, non-flag top-level arg → one-shot chat prompt" },
            { "alias": "hf:<ref> | http(s)://<url>", "expands_to": ["chat", "<ref>"], "note": "Top-level HF/URL arg → chat with that model" },
            { "alias": "claude", "expands_to": ["run", "claude"] },
            { "alias": "codex", "expands_to": ["run", "codex"] },
            { "alias": "gemini", "expands_to": ["run", "gemini"] },
            { "alias": "opencode", "expands_to": ["run", "opencode"] },
            { "alias": "pi", "expands_to": ["run", "pi"] },
            { "alias": "amp", "expands_to": ["run", "amp"] }
        ],
        "environment": [
            { "name": "AIVO_REDUCE_MOTION", "desc": "Disable chat TUI motion effects (=1)" },
            { "name": "AIVO_PREVIEW", "desc": "Force-disable (=0) or force-enable (=1) terminal image preview" },
            { "name": "AIVO_CHAT_DISABLE_MOUSE", "desc": "Disable mouse capture in chat TUI (=1)" },
            { "name": "AIVO_CHAT_SCROLL_SPEED", "desc": "Lines scrolled per wheel tick in chat TUI (default 3)" },
            { "name": "AIVO_PATH", "desc": "Override the install path detected by `aivo update`" },
            { "name": "AIVO_SHARE_BASE_URL", "desc": "Override the public tunnel endpoint used by `aivo logs share`" },
            { "name": "AIVO_DEBUG", "desc": "Surface upstream HTTP request/response detail in some flows (=1)" }
        ],
        "tree": tree,
    });
    match serde_json::to_string_pretty(&payload) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("failed to serialize help: {e}"),
    }
}

/// Recursively serialize a clap `Command` into JSON.
fn serialize_command(cmd: &Command) -> Value {
    let mut obj = Map::new();
    obj.insert("name".into(), json!(cmd.get_name()));
    if let Some(about) = cmd.get_about() {
        obj.insert("about".into(), json!(about.to_string()));
    }
    if let Some(long) = cmd.get_long_about() {
        obj.insert("long_about".into(), json!(long.to_string()));
    }
    let aliases: Vec<String> = cmd
        .get_visible_aliases()
        .map(|s| s.to_string())
        .chain(cmd.get_all_aliases().map(|s| s.to_string()))
        .collect();
    if !aliases.is_empty() {
        obj.insert("aliases".into(), json!(aliases));
    }
    if cmd.is_hide_set() {
        obj.insert("hidden".into(), json!(true));
    }

    let mut args = Vec::new();
    for arg in cmd.get_arguments() {
        if arg.is_hide_set() {
            continue;
        }
        let mut a = Map::new();
        a.insert("id".into(), json!(arg.get_id().as_str()));
        if let Some(s) = arg.get_short() {
            a.insert("short".into(), json!(format!("-{s}")));
        }
        if let Some(l) = arg.get_long() {
            a.insert("long".into(), json!(format!("--{l}")));
        }
        if let Some(h) = arg.get_help().or_else(|| arg.get_long_help()) {
            a.insert("help".into(), json!(h.to_string()));
        }
        let names: Vec<String> = arg
            .get_value_names()
            .map(|v| v.iter().map(|s| s.to_string()).collect())
            .unwrap_or_default();
        if !names.is_empty() {
            a.insert("value_names".into(), json!(names));
        }
        a.insert("takes_value".into(), json!(arg.get_action().takes_values()));
        a.insert("positional".into(), json!(arg.is_positional()));
        args.push(Value::Object(a));
    }
    if !args.is_empty() {
        obj.insert("args".into(), Value::Array(args));
    }

    let subs: Vec<Value> = cmd.get_subcommands().map(serialize_command).collect();
    if !subs.is_empty() {
        obj.insert("subcommands".into(), Value::Array(subs));
    }
    Value::Object(obj)
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

/// Dispatch for `Commands::Audio`. Resolves the prompt from positional
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

    let prompt = match resolve_audio_prompt(&audio_args) {
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

/// Resolves the audio prompt from (positional, `--file`, piped stdin) in
/// that precedence. `--file -` or `--file` with no value reads stdin
/// explicitly. Returns `Ok(None)` to mean "show help" — i.e. the caller
/// had no positional, no `--file`, and stdin was a TTY or empty.
fn resolve_audio_prompt(args: &cli::AudioArgs) -> anyhow::Result<Option<String>> {
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
