use anyhow::{Context, Result};
use serde_json::json;
use std::collections::HashMap;

use crate::cli_args::context_tag_to_tokens;
use crate::constants::PLACEHOLDER_LOOPBACK_URL;
use crate::services::ai_launcher::AIToolType;
use crate::services::codex_model_map::map_model_for_codex_cli;

pub(crate) struct RuntimeArgs {
    pub(crate) args: Vec<String>,
    pub(crate) codex_model_catalog_path: Option<String>,
}

pub(crate) fn merge_preview_env(
    tool_env: &HashMap<String, String>,
    manual_env: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut merged = tool_env.clone();
    if let Some(manual) = manual_env {
        for (key, value) in manual {
            merged.insert(key.clone(), value.clone());
        }
    }
    merged
}

pub(crate) fn preview_args(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    env: &HashMap<String, String>,
) -> Vec<String> {
    let args = inject_claude_teammate_mode(tool, raw_args);
    if tool == AIToolType::Pi {
        return inject_pi_model(model, &args);
    }
    if tool == AIToolType::Amp {
        let args = inject_amp_no_ide(&args, env);
        let args = inject_amp_dangerously_allow_all(&args, env);
        let args = inject_amp_initial_mode(&args, env);
        return inject_amp_settings_file(&args, env);
    }
    if !tool.is_codex_family() {
        return args;
    }

    let use_responses_router = uses_responses_to_chat_router(env);
    let args = inject_codex_model(model, &args, use_responses_router);
    let args = if should_preview_codex_model_catalog(model, use_responses_router) {
        let mut preview = vec![
            "--config".to_string(),
            "model_catalog_json=\"<temp:aivo-codex-model-catalog.json>\"".to_string(),
        ];
        preview.extend(args);
        preview
    } else {
        args
    };
    preview_codex_provider_config_args(env, args)
}

pub(crate) fn build_preview_notes(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    env: &HashMap<String, String>,
) -> Vec<String> {
    let mut notes = Vec::new();

    if tool == AIToolType::Claude
        && !raw_args
            .iter()
            .any(|arg| arg == "--teammate-mode" || arg.starts_with("--teammate-mode="))
    {
        notes.push("injects `--teammate-mode in-process` for Claude".to_string());
    }

    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_ROUTER"],
        "starts an Anthropic compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_ANTHROPIC_TO_OPENAI_ROUTER"],
        "starts an Anthropic-to-OpenAI compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_COPILOT_ROUTER"],
        "starts a Copilot router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_RESPONSES_TO_CHAT_ROUTER"],
        "starts a Responses-to-Chat router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER"],
        "starts a Copilot-backed Responses-to-Chat router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_GEMINI_ROUTER"],
        "starts a Gemini router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_GEMINI_COPILOT_ROUTER"],
        "starts a Copilot-backed Gemini router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_OPENCODE_ROUTER"],
        "starts an OpenCode compatibility router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_OPENCODE_COPILOT_ROUTER"],
        "starts a Copilot-backed OpenCode router on a random local port",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_SETUP_PI_AGENT_DIR"],
        "writes a temporary Pi agent dir with custom provider config",
    );
    maybe_push_router_note(
        &mut notes,
        env,
        &["AIVO_USE_PI_COPILOT_ROUTER"],
        "starts a Copilot-backed Pi router on a random local port",
    );

    let use_responses_router = uses_responses_to_chat_router(env);
    if tool.is_codex_family()
        && model.is_some()
        && !raw_args.iter().any(|arg| {
            arg == "--model" || arg == "-m" || arg.starts_with("--model=") || arg.starts_with("-m=")
        })
    {
        notes.push("injects `-m <model>` for Codex".to_string());
    }
    if tool.is_codex_family() && should_preview_codex_model_catalog(model, use_responses_router) {
        notes.push("writes a temporary Codex model catalog file at launch time".to_string());
    }
    if tool.is_codex_family() && env.contains_key("OPENAI_BASE_URL") {
        notes.push("injects `--config model_provider=aivo` to bypass codex auth.json".to_string());
    }

    if tool == AIToolType::Pi
        && model.is_some()
        && !raw_args
            .iter()
            .any(|arg| arg == "--model" || arg.starts_with("--model="))
    {
        notes.push("injects `--model <model>` for Pi".to_string());
    }

    if tool == AIToolType::Amp && env.contains_key("AIVO_USE_AMP_BRIDGE") {
        notes.push(
            "starts an Amp bridge on a random local port — stubs the management plane locally \
             (auth/threads/telemetry) and translates LLM calls to the upstream"
                .to_string(),
        );
        if !raw_args.iter().any(|a| a == "--ide" || a == "--no-ide") {
            notes.push(
                "injects `--no-ide` so amp doesn't auto-prepend open IDE file/selection to \
                 messages going to the rerouted upstream (pass `--ide` to opt back in)"
                    .to_string(),
            );
        }
        if amp_runs_non_interactively(raw_args)
            && !raw_args.iter().any(|a| a == "--dangerously-allow-all")
        {
            notes.push(
                "injects `--dangerously-allow-all` because amp is in a non-interactive mode \
                 (`-x` / `--stream-json-input`); without it, tool-approval prompts would hang \
                 the run with no human to answer them"
                    .to_string(),
            );
        }
    }
    if tool == AIToolType::Amp
        && (env.contains_key("AIVO_AMP_INTERNAL_MODEL")
            || env.contains_key("AIVO_AMP_INTERNAL_MODEL_JSON"))
    {
        notes.push(
            "writes a temporary amp settings.json (merged from your ~/.config/amp/settings.json) \
             with the requested `internal.model` override and passes it via `--settings-file`"
                .to_string(),
        );
    }

    notes
}

pub(crate) async fn build_runtime_args(
    tool: AIToolType,
    raw_args: &[String],
    model: Option<&str>,
    codex_app_models: Option<&[String]>,
    env: &HashMap<String, String>,
) -> Result<RuntimeArgs> {
    let args = inject_claude_teammate_mode(tool, raw_args);
    if tool == AIToolType::Pi {
        return Ok(RuntimeArgs {
            args: inject_pi_model(model, &args),
            codex_model_catalog_path: None,
        });
    }
    if tool == AIToolType::Amp {
        let args = inject_amp_no_ide(&args, env);
        let args = inject_amp_dangerously_allow_all(&args, env);
        let args = inject_amp_initial_mode(&args, env);
        return Ok(RuntimeArgs {
            args: inject_amp_settings_file(&args, env),
            codex_model_catalog_path: None,
        });
    }
    if !tool.is_codex_family() {
        return Ok(RuntimeArgs {
            args,
            codex_model_catalog_path: None,
        });
    }

    let use_responses_router = uses_responses_to_chat_router(env);
    let codex_model_catalog_path =
        maybe_write_codex_model_catalog(model, codex_app_models, use_responses_router).await?;
    let args = inject_codex_model(model, &args, use_responses_router);
    let args = inject_codex_model_catalog(codex_model_catalog_path.as_deref(), &args);
    let args = inject_codex_cursor_tui_reasoning(use_responses_router, &args);

    Ok(RuntimeArgs {
        args,
        codex_model_catalog_path,
    })
}

/// Force codex's TUI to render the reasoning panel for cursor-backed
/// turns. The catalog override + `reasoning_summary_format=experimental`
/// makes codex *request* reasoning summaries and parse the incoming
/// `agent_reasoning` events (verified in `~/.codex/sessions/.../rollout-*.jsonl` —
/// the events ARE recorded), but codex's TUI hides them by default for
/// any model whose `hide_agent_reasoning` resolves to true. Explicit
/// override forces the panel visible so cursor's `agent_thought_chunk`
/// stream and the bridge's heartbeat dots show up as visible activity.
/// `show_raw_agent_reasoning` makes raw thoughts render in addition to
/// model-emitted summaries; cursor's composer-* doesn't emit OpenAI-style
/// encrypted reasoning, only thought chunks routed through summary
/// deltas, so the raw flag is what surfaces them.
fn inject_codex_cursor_tui_reasoning(use_router: bool, args: &[String]) -> Vec<String> {
    if !use_router {
        return args.to_vec();
    }
    if args
        .iter()
        .any(|a| a.contains("tui.hide_agent_reasoning") || a.contains("hide_agent_reasoning"))
    {
        return args.to_vec();
    }
    let mut new_args = vec![
        "--config".to_string(),
        "tui.hide_agent_reasoning=false".to_string(),
        "--config".to_string(),
        "tui.show_raw_agent_reasoning=true".to_string(),
    ];
    new_args.extend_from_slice(args);
    new_args
}

/// True when codex's upstream is one of aivo's local routers rather than a
/// real OpenAI-shaped endpoint. In that case the model id is meaningful to
/// the router (it picks the upstream provider/model), so `inject_codex_model`
/// must pass the raw name and `maybe_write_codex_model_catalog` must emit a
/// catalog entry so codex itself accepts the unknown slug. Includes the
/// cursor router because cursor models like `composer-2.5` would otherwise
/// hit `map_model_for_codex_cli`'s fallback and be rewritten to `gpt-4o`.
fn uses_responses_to_chat_router(env: &HashMap<String, String>) -> bool {
    env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_ROUTER")
        || env.contains_key("AIVO_USE_RESPONSES_TO_CHAT_COPILOT_ROUTER")
        || env.contains_key("AIVO_USE_CURSOR_ROUTER")
}

/// Converts Codex `OPENAI_BASE_URL` + `OPENAI_API_KEY` env vars into
/// `--config model_provider` CLI flags so codex uses a custom provider
/// named "aivo" instead of its built-in auth flow.
///
/// Bypasses `~/.codex/auth.json` and avoids the deprecated `OPENAI_BASE_URL`
/// env var warning. Must be called after `prepare_runtime_env` (placeholder
/// URLs resolved) and before `spawn_child`.
pub(crate) fn inject_codex_provider_config(
    env: &mut HashMap<String, String>,
    args: &mut Vec<String>,
) {
    if args.iter().any(|a| a.contains("model_provider")) {
        return;
    }
    let base_url = match env.remove("OPENAI_BASE_URL") {
        Some(url) => url,
        None => return,
    };
    let api_key = match env.remove("OPENAI_API_KEY") {
        Some(key) => key,
        None => {
            env.insert("OPENAI_BASE_URL".to_string(), base_url);
            return;
        }
    };

    env.insert("AIVO_CODEX_API_KEY".to_string(), api_key);

    let escaped_url = base_url.replace('\\', "\\\\").replace('"', "\\\"");
    let mut config_args = vec![
        "--config".to_string(),
        "model_provider=\"aivo\"".to_string(),
        "--config".to_string(),
        "model_providers.aivo.name=\"aivo\"".to_string(),
        "--config".to_string(),
        format!("model_providers.aivo.base_url=\"{}\"", escaped_url),
        "--config".to_string(),
        "model_providers.aivo.env_key=\"AIVO_CODEX_API_KEY\"".to_string(),
    ];
    // Disable the built-in `codex_apps` MCP (OpenAI Connectors registry).
    // When aivo is routing codex to a non-OpenAI provider, the user is not
    // authed with ChatGPT, so codex_apps can't do anything useful — but it
    // still tries to fetch chatgpt.com/backend-api/connectors/directory on
    // startup, which costs 10s of wall-clock time and fails outright
    // without VPN. Disabling removes that tax; users who need apps should
    // run `codex` directly rather than going through aivo.
    if !args.iter().any(|a| a == "apps" || a == "connectors")
        && !args
            .windows(2)
            .any(|w| (w[0] == "--disable" || w[0] == "--enable") && w[1] == "apps")
    {
        config_args.push("--disable".to_string());
        config_args.push("apps".to_string());
    }
    config_args.append(args);
    *args = config_args;
}

/// Append `--config model_context_window=<tokens>` for codex when the user
/// asked for `--max-context=<N>m`. Codex clamps the value against the
/// model's advertised ceiling internally, so passing a high value on a
/// small model is silently a no-op rather than an error. We append (not
/// prepend) so the user's own `--config` flags, if any, parse first and
/// can win on conflict per codex's last-write-wins semantics.
pub(crate) fn inject_codex_max_context(args: &mut Vec<String>, max_context: Option<&str>) {
    let Some(tag) = max_context else {
        return;
    };
    let Some(tokens) = context_tag_to_tokens(tag) else {
        return;
    };
    args.push("--config".to_string());
    args.push(format!("model_context_window={tokens}"));
}

pub(crate) fn inject_codex_max_context_before_args(
    args: &mut Vec<String>,
    max_context: Option<&str>,
) {
    let Some(tag) = max_context else {
        return;
    };
    let Some(tokens) = context_tag_to_tokens(tag) else {
        return;
    };
    let insert_at = codex_global_prefix_len(args);
    args.splice(
        insert_at..insert_at,
        [
            "--config".to_string(),
            format!("model_context_window={tokens}"),
        ],
    );
}

/// Converts Codex CLI args into Codex Desktop App args by inserting the
/// `app` subcommand after leading global options. aivo injects Codex config
/// as top-level flags so the desktop app server receives provider/model
/// overrides without writing them into the user's real `~/.codex/config.toml`.
pub(crate) fn inject_codex_app_subcommand(args: &mut Vec<String>) {
    let insert_at = codex_global_prefix_len(args);
    if args.get(insert_at).is_some_and(|arg| arg == "app") {
        return;
    }
    args.insert(insert_at, "app".to_string());
}

/// Drains the global flag prefix from `args`. Used for codex-app launches:
/// the parent `codex app` invocation's `-c` overrides are NOT propagated to
/// the GUI's spawned app-server child, so we move them into the
/// `CODEX_CLI_PATH` wrapper instead — see `codex_app_wrapper`.
pub(crate) fn drain_codex_global_prefix(args: &mut Vec<String>) -> Vec<String> {
    let end = codex_global_prefix_len(args);
    args.drain(0..end).collect()
}

fn codex_global_prefix_len(args: &[String]) -> usize {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if arg == "--" {
            break;
        }
        if codex_global_flag_takes_value(arg) {
            index += if arg.contains('=') { 1 } else { 2 };
            continue;
        }
        if codex_global_flag_no_value(arg) {
            index += 1;
            continue;
        }
        break;
    }
    index.min(args.len())
}

fn codex_global_flag_takes_value(arg: &str) -> bool {
    matches!(
        arg,
        "-c" | "--config"
            | "-m"
            | "--model"
            | "--model-provider"
            | "--profile"
            | "-s"
            | "--sandbox"
            | "-a"
            | "--ask-for-approval"
            | "-C"
            | "--cd"
            | "--search"
            | "--image"
            | "--enable"
            | "--disable"
    ) || arg.starts_with("--config=")
        || arg.starts_with("-c=")
        || arg.starts_with("--model=")
        || arg.starts_with("-m=")
        || arg.starts_with("--model-provider=")
        || arg.starts_with("--profile=")
        || arg.starts_with("--sandbox=")
        || arg.starts_with("-s=")
        || arg.starts_with("--ask-for-approval=")
        || arg.starts_with("-a=")
        || arg.starts_with("--cd=")
        || arg.starts_with("-C=")
        || arg.starts_with("--search=")
        || arg.starts_with("--image=")
        || arg.starts_with("--enable=")
        || arg.starts_with("--disable=")
}

fn codex_global_flag_no_value(arg: &str) -> bool {
    matches!(
        arg,
        "--oss"
            | "--dangerously-bypass-approvals-and-sandbox"
            | "--skip-git-repo-check"
            | "--full-auto"
            | "--json"
    )
}

/// Rewrites env vars for the dry-run preview so it reflects what codex
/// will actually receive at runtime.
pub(crate) fn rewrite_codex_preview_env(env: &mut HashMap<String, String>) {
    if let Some(api_key) = env.remove("OPENAI_API_KEY") {
        env.insert("AIVO_CODEX_API_KEY".to_string(), api_key);
    }
    if env.remove("AIVO_CODEX_OAUTH_CREDS").is_some() {
        env.insert(
            "CODEX_HOME".to_string(),
            "<temp:aivo-codex-home>".to_string(),
        );
    }
    env.remove("AIVO_CODEX_KEY_ID");
    env.remove("AIVO_CODEX_APP_HOME_KEY");
    env.remove("OPENAI_BASE_URL");
}

/// Rewrites env vars for the dry-run preview so it reflects what amp will
/// actually see at runtime: `AMP_URL` and `AMP_API_KEY` are set by
/// `start_amp_bridge` after binding the bridge port, so they don't show up
/// in the env produced by `for_amp`. The preview adds placeholders here
/// (`http://127.0.0.1:<port>`, `aivo-bridge`) so the user can see at a
/// glance that amp will talk to a localhost bridge — not directly to
/// `AIVO_AMP_UPSTREAM_BASE_URL` like the bare env might suggest.
pub(crate) fn rewrite_amp_preview_env(env: &mut HashMap<String, String>) {
    if env.contains_key("AIVO_USE_AMP_BRIDGE") {
        env.insert("AMP_URL".to_string(), "http://127.0.0.1:<port>".to_string());
        env.insert("AMP_API_KEY".to_string(), "aivo-bridge".to_string());
    }
}

/// Preview-only: prepends model_provider `--config` flags for Codex args
/// without mutating the env map.
fn preview_codex_provider_config_args(
    env: &HashMap<String, String>,
    args: Vec<String>,
) -> Vec<String> {
    let base_url = match env.get("OPENAI_BASE_URL") {
        Some(url) => url.as_str(),
        None => return args,
    };

    let display_url = if base_url == PLACEHOLDER_LOOPBACK_URL {
        "http://127.0.0.1:<port>"
    } else {
        base_url
    };

    let mut prefix = vec![
        "--config".to_string(),
        "model_provider=\"aivo\"".to_string(),
        "--config".to_string(),
        "model_providers.aivo.name=\"aivo\"".to_string(),
        "--config".to_string(),
        format!("model_providers.aivo.base_url=\"{}\"", display_url),
        "--config".to_string(),
        "model_providers.aivo.env_key=\"AIVO_CODEX_API_KEY\"".to_string(),
    ];
    // Mirror the runtime behavior of inject_codex_provider_config: disable
    // the codex_apps MCP to avoid a startup call to chatgpt.com that would
    // hang without VPN and yield nothing useful under aivo's routing.
    if !args
        .windows(2)
        .any(|w| (w[0] == "--disable" || w[0] == "--enable") && w[1] == "apps")
    {
        prefix.push("--disable".to_string());
        prefix.push("apps".to_string());
    }
    prefix.extend(args);
    prefix
}

fn maybe_push_router_note(
    notes: &mut Vec<String>,
    env: &HashMap<String, String>,
    env_keys: &[&str],
    note: &str,
) {
    if env_keys.iter().any(|key| env.contains_key(*key)) {
        notes.push(note.to_string());
    }
}

fn should_preview_codex_model_catalog(model: Option<&str>, uses_non_openai_router: bool) -> bool {
    let model = match model {
        Some(model) if !model.is_empty() => model,
        _ => return false,
    };

    if !uses_non_openai_router {
        return false;
    }

    let model_lower = model.to_lowercase();
    let name_only = model_lower.split('/').next_back().unwrap_or(&model_lower);
    !(name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4"))
}

fn inject_claude_teammate_mode(tool: AIToolType, args: &[String]) -> Vec<String> {
    if tool != AIToolType::Claude {
        return args.to_vec();
    }

    let has_teammate_mode = args
        .iter()
        .any(|a| a == "--teammate-mode" || a.starts_with("--teammate-mode="));
    if has_teammate_mode {
        return args.to_vec();
    }

    let mut new_args = vec!["--teammate-mode".to_string(), "in-process".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

/// Prepends `--settings-file <path>` to amp's args when the bridge will
/// write a merged settings override. Triggered by any of: `--1m`, the
/// per-mode `--rush-model / --smart-model / --deep-model / --large-model`
/// flags, `--disable-tool`, or — in bridge mode — the always-on
/// auto-disable of unsupported tools (web_search/read_web_page/Task).
/// At runtime, the real path is in `AIVO_AMP_SETTINGS_FILE`; for dry-run
/// preview we substitute a `<temp:aivo-amp-settings.json>` placeholder
/// since the path is only known after `start_amp_bridge` runs. Skips if
/// the user already passed `--settings-file` themselves.
fn inject_amp_settings_file(args: &[String], env: &HashMap<String, String>) -> Vec<String> {
    let path = if let Some(p) = env.get("AIVO_AMP_SETTINGS_FILE") {
        p.clone()
    } else if env.contains_key("AIVO_AMP_INTERNAL_MODEL")
        || env.contains_key("AIVO_AMP_INTERNAL_MODEL_JSON")
        || env.contains_key("AIVO_AMP_TOOLS_DISABLE")
    {
        "<temp:aivo-amp-settings.json>".to_string()
    } else {
        return args.to_vec();
    };
    let already_set = args
        .iter()
        .any(|a| a == "--settings-file" || a.starts_with("--settings-file="));
    if already_set {
        return args.to_vec();
    }
    let mut new_args = vec!["--settings-file".to_string(), path];
    new_args.extend_from_slice(args);
    new_args
}

/// Prepends `--no-ide` to amp's args when the bridge is active. Amp's
/// IDE integration (default on) auto-prepends the open IDE file's path
/// and current text selection to every user message — useful when amp
/// is talking to ampcode.com, but a privacy leak when the bridge is
/// rerouting traffic to a third-party upstream (deepseek/openrouter/etc.).
/// Native-amp launches (`AIVO_USE_AMP_BRIDGE` unset) keep the default since
/// the user's data only goes back to Sourcegraph in that case.
///
/// Skipped if the user already passed `--ide` or `--no-ide` themselves —
/// explicit choice wins.
fn inject_amp_no_ide(args: &[String], env: &HashMap<String, String>) -> Vec<String> {
    if !env.contains_key("AIVO_USE_AMP_BRIDGE") {
        return args.to_vec();
    }
    let already_set = args.iter().any(|a| a == "--ide" || a == "--no-ide");
    if already_set {
        return args.to_vec();
    }
    let mut new_args = vec!["--no-ide".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

/// True when amp's args put it in a non-interactive mode that can't surface
/// tool-approval prompts to a human:
/// - `-x` / `--execute "<prompt>"` — one-shot execution
/// - `--stream-json-input` — programmatic JSON-over-stdin
fn amp_runs_non_interactively(args: &[String]) -> bool {
    args.iter().any(|a| {
        a == "-x" || a == "--execute" || a.starts_with("--execute=") || a == "--stream-json-input"
    })
}

/// Prepends `--mode <X>` to amp's args when aivo's `--mode` flag is set
/// (carried via `AIVO_AMP_INITIAL_MODE`). Amp accepts the same flag and
/// pins the thread's initial agent mode; aivo translates because aivo's
/// `-m` is the model flag and would collide with amp's short alias.
///
/// Skipped if the user already passed `--mode` (or `-m` with a recognized
/// mode value) directly to amp — explicit choice wins. Note: the simpler
/// `args.iter().any(|a| a == "-m")` check would misfire because `-m` is
/// also aivo's model short flag; by the time we reach this injection
/// point aivo's flags have been stripped, so any `-m` left in args is
/// destined for amp.
fn inject_amp_initial_mode(args: &[String], env: &HashMap<String, String>) -> Vec<String> {
    let Some(mode) = env.get("AIVO_AMP_INITIAL_MODE") else {
        return args.to_vec();
    };
    let already_set = args
        .iter()
        .any(|a| a == "--mode" || a == "-m" || a.starts_with("--mode="));
    if already_set {
        return args.to_vec();
    }
    let mut new_args = vec!["--mode".to_string(), mode.clone()];
    new_args.extend_from_slice(args);
    new_args
}

/// Prepends `--dangerously-allow-all` to amp's args when the bridge is
/// active AND amp is in a non-interactive mode (`-x`/`--execute`/
/// `--stream-json-input`). Without this, amp blocks on every tool-approval
/// prompt and the one-shot/programmatic call hangs forever — there's no
/// human at the other end to press a key.
///
/// Skipped if the user already passed the flag explicitly, and skipped
/// for native-amp launches (no bridge) so we don't widen permissions on
/// runs talking to ampcode.com itself.
fn inject_amp_dangerously_allow_all(args: &[String], env: &HashMap<String, String>) -> Vec<String> {
    if !env.contains_key("AIVO_USE_AMP_BRIDGE") {
        return args.to_vec();
    }
    if !amp_runs_non_interactively(args) {
        return args.to_vec();
    }
    let already_set = args.iter().any(|a| a == "--dangerously-allow-all");
    if already_set {
        return args.to_vec();
    }
    let mut new_args = vec!["--dangerously-allow-all".to_string()];
    new_args.extend_from_slice(args);
    new_args
}

fn inject_pi_model(model: Option<&str>, args: &[String]) -> Vec<String> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return args.to_vec(),
    };

    let has_model_flag = args
        .iter()
        .any(|a| a == "--model" || a.starts_with("--model="));
    if has_model_flag {
        return args.to_vec();
    }

    // Always prefix model with "aivo/" so pi selects
    // the custom provider from models.json.
    let pi_model = format!("aivo/{model}");

    let mut new_args = vec!["--model".to_string(), pi_model];
    new_args.extend_from_slice(args);
    new_args
}

fn inject_codex_model(model: Option<&str>, args: &[String], use_router: bool) -> Vec<String> {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return args.to_vec(),
    };

    let has_model_flag = args
        .iter()
        .any(|a| a == "--model" || a == "-m" || a.starts_with("--model=") || a.starts_with("-m="));
    if has_model_flag {
        return args.to_vec();
    }

    let codex_model = if use_router {
        model.to_string()
    } else {
        map_model_for_codex_cli(model)
    };
    let mut new_args = vec!["-m".to_string(), codex_model];
    new_args.extend_from_slice(args);
    new_args
}

/// Sets root `model = "<X>"` in the codex config via `-c`. The codex CLI's
/// `-m` flag only seeds the current launch's model; codex-app's GUI picks its
/// per-thread default from the resolved config's `model` field. Without this,
/// the GUI falls back to the bundled default (`gpt-5.5`) and the upstream
/// rejects the request.
pub(crate) fn inject_codex_root_model(args: &mut Vec<String>, model: Option<&str>) {
    let model = match model {
        Some(m) if !m.is_empty() => m,
        _ => return,
    };
    if args.iter().any(|a| a == "model" || a.starts_with("model=")) {
        return;
    }
    let escaped = model.replace('\\', "\\\\").replace('"', "\\\"");
    let insert_at = codex_global_prefix_len(args);
    args.splice(
        insert_at..insert_at,
        ["--config".to_string(), format!("model=\"{}\"", escaped)],
    );
}

/// Best-effort reaper for older `aivo-codex-model-catalog-*.json` tempfiles
/// the codex-app launch path intentionally leaves behind (the GUI references
/// them for the duration of its session, so we can't unlink during cleanup
/// without risking a Ctrl-C'd aivo yanking the file out from under a still-
/// running Codex.app). Deletes files older than 24h that match our prefix.
async fn cleanup_stale_codex_model_catalogs() {
    use std::time::{Duration, SystemTime};
    const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
    let dir = std::env::temp_dir();
    let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
        return;
    };
    let now = SystemTime::now();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.starts_with("aivo-codex-model-catalog-") || !name.ends_with(".json") {
            continue;
        }
        let Ok(meta) = entry.metadata().await else {
            continue;
        };
        let Ok(modified) = meta.modified() else {
            continue;
        };
        if now.duration_since(modified).is_ok_and(|age| age > MAX_AGE) {
            let _ = tokio::fs::remove_file(&path).await;
        }
    }
}

fn inject_codex_model_catalog(path: Option<&str>, args: &[String]) -> Vec<String> {
    let path = match path {
        Some(p) if !p.is_empty() => p,
        _ => return args.to_vec(),
    };

    if args.iter().any(|a| a.contains("model_catalog_json")) {
        return args.to_vec();
    }

    let escaped_path = path.replace('\\', "\\\\").replace('"', "\\\"");
    let mut new_args = vec![
        "--config".to_string(),
        format!("model_catalog_json=\"{}\"", escaped_path),
    ];
    new_args.extend_from_slice(args);
    new_args
}

async fn maybe_write_codex_model_catalog(
    model: Option<&str>,
    codex_app_models: Option<&[String]>,
    uses_non_openai_router: bool,
) -> Result<Option<String>> {
    let slugs = catalog_slugs(model, codex_app_models, uses_non_openai_router);
    if slugs.is_empty() {
        return Ok(None);
    }
    cleanup_stale_codex_model_catalogs().await;

    let catalog_json = build_codex_model_catalog_json(&slugs)?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = format!(
        "aivo-codex-model-catalog-{}-{}.json",
        std::process::id(),
        nonce
    );
    let path = std::env::temp_dir().join(file_name);

    tokio::fs::write(&path, catalog_json)
        .await
        .with_context(|| {
            format!(
                "Failed to write Codex model catalog override at {}",
                path.display()
            )
        })?;

    Ok(Some(path.to_string_lossy().to_string()))
}

/// Determines which model slugs the codex catalog file should contain.
/// Returns empty when no catalog is needed (regular OpenAI-style models hitting
/// codex's built-in catalog).
fn catalog_slugs(
    model: Option<&str>,
    codex_app_models: Option<&[String]>,
    uses_non_openai_router: bool,
) -> Vec<String> {
    // CodexApp: discovered provider models plus the explicit `-m`, for the GUI
    // dropdown. Reject control-byte slugs — they'd break out of the TOML
    // basic-string in `inject_codex_root_model` / catalog JSON.
    if let Some(list) = codex_app_models {
        let mut slugs: Vec<String> = list
            .iter()
            .map(String::as_str)
            .chain(model)
            .filter(|&m| is_safe_codex_slug(m))
            .map(str::to_string)
            .collect();
        if !slugs.is_empty() {
            slugs.sort();
            slugs.dedup();
            // If EVERY discovered slug is OpenAI-shaped, codex's built-in
            // catalog already serves them with correct metadata (272k context,
            // proper reasoning fields, etc.) — overwriting with our slim
            // 128k/freeform entries silently degrades capability. Defer to the
            // bundled catalog in that case; aivo's wrapper still routes
            // requests through our local provider so the user's key is used.
            if slugs.iter().all(|m| is_openai_shaped_slug(m)) {
                return Vec::new();
            }
            return slugs;
        }
    }

    // CLI single-model path: only write when the model is non-OpenAI-shaped
    // and we're behind a non-OpenAI router (else codex's built-in catalog
    // serves the user without aivo interference).
    let model = match model {
        Some(m) if !m.is_empty() && is_safe_codex_slug(m) => m,
        _ => return Vec::new(),
    };
    if !uses_non_openai_router {
        return Vec::new();
    }
    if is_openai_shaped_slug(model) {
        return Vec::new();
    }
    vec![model.to_string()]
}

/// True when the slug's local name (after any `provider/` prefix) starts with
/// an OpenAI family prefix codex's bundled catalog already covers.
fn is_openai_shaped_slug(model: &str) -> bool {
    let lower = model.to_lowercase();
    let name_only = lower.split('/').next_back().unwrap_or(&lower);
    name_only.starts_with("gpt-")
        || name_only.starts_with("o1")
        || name_only.starts_with("o3")
        || name_only.starts_with("o4")
}

/// Rejects model slugs that contain control bytes (NUL, newline, CR, tab).
/// These would otherwise be embedded into TOML basic-strings via
/// `inject_codex_root_model` / `inject_codex_provider_config`, where they
/// terminate the string and let the rest re-target unrelated config keys.
fn is_safe_codex_slug(s: &str) -> bool {
    !s.is_empty() && !s.chars().any(|c| c.is_control())
}

fn build_codex_model_catalog_json(slugs: &[String]) -> Result<String> {
    let models: Vec<_> = slugs
        .iter()
        .enumerate()
        .map(|(i, m)| model_entry(m, i))
        .collect();
    let catalog = json!({ "models": models });
    Ok(serde_json::to_string(&catalog)?)
}

fn model_entry(model: &str, index: usize) -> serde_json::Value {
    // Field set tracks codex 0.133+ `ModelInfo` (protocol/src/openai_models.rs).
    // Missing required fields make codex silently reject the catalog file —
    // codex then falls back to its built-in `models_cache.json` (full of stock
    // OpenAI slugs), so the GUI's model picker ignores aivo's provider entirely.
    //
    // Priority starts at 10 (matches codex's stock catalog band — gpt-5.5 is 9,
    // gpt-5.4 is 16, gpt-5.4-mini is 23) and increases by 10 per entry. Lower
    // values render earlier / more prominently in the picker; the GUI hides
    // entries with priority 0 ("internal") so a non-zero value is required.
    let priority = 10 + (index as i64) * 10;
    json!({
        "slug": model,
        "display_name": model,
        "description": format!("aivo-routed model {}", model),
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Lighter reasoning"},
            {"effort": "medium", "description": "Balanced"},
            {"effort": "high", "description": "Deeper reasoning"}
        ],
        "shell_type": "default",
        "visibility": "list",
        "supported_in_api": true,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "availability_nux": serde_json::Value::Null,
        "upgrade": serde_json::Value::Null,
        "base_instructions": "You are a coding agent.",
        "model_messages": serde_json::Value::Null,
        "supports_reasoning_summaries": false,
        "default_reasoning_summary": "none",
        "support_verbosity": false,
        "default_verbosity": serde_json::Value::Null,
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "tokens", "limit": 100000},
        "supports_parallel_tool_calls": false,
        "supports_image_detail_original": false,
        "context_window": 128000,
        "max_context_window": 128000,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_responses_to_chat_router_recognizes_cursor_router() {
        // Regression: codex + cursor key must take the non-OpenAI router
        // branch so the raw model id (e.g. `composer-2.5`) is preserved and
        // a model_catalog_json override is written. Without this,
        // map_model_for_codex_cli rewrites the slug to `gpt-4o` and codex
        // shows / requests the wrong model.
        let env = HashMap::from([("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string())]);
        assert!(uses_responses_to_chat_router(&env));
    }

    #[tokio::test]
    async fn cursor_router_codex_keeps_model_and_writes_catalog() {
        let env = HashMap::from([("AIVO_USE_CURSOR_ROUTER".to_string(), "1".to_string())]);
        let runtime = build_runtime_args(
            AIToolType::Codex,
            &["prompt".to_string()],
            Some("composer-2.5"),
            None,
            &env,
        )
        .await
        .unwrap();

        let m_idx = runtime
            .args
            .iter()
            .position(|a| a == "-m")
            .expect("-m flag present");
        assert_eq!(runtime.args[m_idx + 1], "composer-2.5");
        assert!(
            runtime.codex_model_catalog_path.is_some(),
            "cursor router branch must emit a model catalog so codex accepts the slug"
        );
        assert!(
            runtime
                .args
                .iter()
                .any(|a| a.starts_with("model_catalog_json=")),
            "model_catalog_json --config flag must be injected"
        );
        if let Some(path) = runtime.codex_model_catalog_path {
            let _ = tokio::fs::remove_file(path).await;
        }
    }

    #[tokio::test]
    async fn codex_app_wraps_global_options_before_app_subcommand() {
        let env = HashMap::from([
            (
                "OPENAI_BASE_URL".to_string(),
                "https://api.openai.com/v1".to_string(),
            ),
            ("OPENAI_API_KEY".to_string(), "sk-test".to_string()),
        ]);
        let mut runtime = build_runtime_args(
            AIToolType::CodexApp,
            &[".".to_string()],
            Some("gpt-5"),
            None,
            &env,
        )
        .await
        .unwrap();

        let mut env_for_provider = env.clone();
        inject_codex_provider_config(&mut env_for_provider, &mut runtime.args);
        inject_codex_max_context_before_args(&mut runtime.args, Some("1m"));
        inject_codex_app_subcommand(&mut runtime.args);

        let app_idx = runtime
            .args
            .iter()
            .position(|arg| arg == "app")
            .expect("app subcommand present");
        let path_idx = runtime.args.iter().position(|arg| arg == ".").unwrap();
        assert!(app_idx < path_idx, "app must come before PATH");
        assert!(
            runtime.args[..app_idx]
                .windows(2)
                .any(|w| w[0] == "--config" && w[1] == "model_context_window=1000000"),
            "max-context config must remain a top-level codex option"
        );
        assert!(
            runtime.args[..app_idx]
                .windows(2)
                .any(|w| w[0] == "--disable" && w[1] == "apps"),
            "provider config should remain before app"
        );
    }

    #[test]
    fn codex_app_subcommand_respects_user_global_flags() {
        let mut args = vec![
            "--profile".to_string(),
            "work".to_string(),
            "--help".to_string(),
        ];
        inject_codex_app_subcommand(&mut args);
        assert_eq!(args, vec!["--profile", "work", "app", "--help"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_for_claude() {
        let args = vec!["--verbose".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(
            result,
            vec!["--teammate-mode", "in-process", "--verbose", "prompt"]
        );
    }

    #[test]
    fn test_inject_claude_teammate_mode_skips_non_claude() {
        let args = vec!["--verbose".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Codex, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Gemini, &args);
        assert_eq!(result, vec!["--verbose"]);

        let result = inject_claude_teammate_mode(AIToolType::Opencode, &args);
        assert_eq!(result, vec!["--verbose"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_respects_user_flag() {
        let args = vec![
            "--teammate-mode".to_string(),
            "split".to_string(),
            "prompt".to_string(),
        ];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "split", "prompt"]);

        let args = vec!["--teammate-mode=split".to_string(), "prompt".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode=split", "prompt"]);
    }

    #[test]
    fn test_inject_claude_teammate_mode_empty_args() {
        let args: Vec<String> = vec![];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(result, vec!["--teammate-mode", "in-process"]);
    }

    #[test]
    fn test_inject_codex_model_injects_when_provided() {
        let model = Some("o4-mini");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "o4-mini", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_original() {
        let model = Some("kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_router_passes_namespaced() {
        let model = Some("moonshot/kimi-k2.5");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, true);
        assert_eq!(result, vec!["-m", "moonshot/kimi-k2.5", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_when_already_specified() {
        let model = Some("o4-mini");
        let args = vec![
            "--model".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["--model", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_shorthand_flag() {
        let model = Some("o4-mini");
        let args = vec![
            "-m".to_string(),
            "gpt-4o".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["-m", "gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_equals_format() {
        let model = Some("o4-mini");
        let args = vec!["--model=gpt-4o".to_string(), "file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["--model=gpt-4o", "file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_empty_model() {
        let model = Some("");
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_skips_none_model() {
        let model: Option<&str> = None;
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model(model, &args, false);
        assert_eq!(result, vec!["file.ts"]);
    }

    #[test]
    fn test_inject_codex_model_catalog_injects_when_path_provided() {
        let args = vec!["file.ts".to_string()];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(
            result,
            vec![
                "--config",
                "model_catalog_json=\"/tmp/catalog.json\"",
                "file.ts"
            ]
        );
    }

    #[test]
    fn test_inject_codex_model_catalog_skips_when_existing_setting_present() {
        let args = vec![
            "--config".to_string(),
            "model_catalog_json=\"/tmp/custom.json\"".to_string(),
            "file.ts".to_string(),
        ];
        let result = inject_codex_model_catalog(Some("/tmp/catalog.json"), &args);
        assert_eq!(result, args);
    }

    #[test]
    fn test_build_codex_model_catalog_json_includes_model_slug() {
        let model = "minimax/minimax-m2.5".to_string();
        let json = build_codex_model_catalog_json(&[model.clone()]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["models"][0]["slug"], model);
        assert_eq!(parsed["models"][0]["display_name"], model);
    }

    #[test]
    fn build_codex_model_catalog_json_uses_shell_type_default() {
        // Pin: codex 0.132+ only accepts `"default"` for `shell_type` in
        // model_catalog_json entries. Anything else (we previously emitted
        // `"shell_command"`) fails the catalog parse with
        // `Error: failed to parse model_catalog_json path '...' as JSON: ...`,
        // codex silently swallows the failure and falls back to its built-in
        // default model — so the user's `-m <picked>` is ignored and the
        // banner shows `gpt-4o` instead of the chosen cursor/openrouter slug.
        let json = build_codex_model_catalog_json(&["composer-2.5".to_string()]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["models"][0]["shell_type"], "default");
    }

    #[test]
    fn build_codex_model_catalog_json_emits_multiple_entries() {
        // CodexApp without -m: catalog should list every discovered model so
        // the GUI dropdown can show them.
        let json = build_codex_model_catalog_json(&[
            "deepseek-chat".to_string(),
            "deepseek-reasoner".to_string(),
        ])
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["models"][0]["slug"], "deepseek-chat");
        assert_eq!(parsed["models"][1]["slug"], "deepseek-reasoner");
    }

    #[test]
    fn catalog_slugs_falls_back_to_single_model_when_no_app_list() {
        let slugs = catalog_slugs(Some("composer-2.5"), None, true);
        assert_eq!(slugs, vec!["composer-2.5"]);
    }

    #[test]
    fn catalog_slugs_uses_app_list_when_present_regardless_of_router() {
        // CodexApp path: catalog gets written even on the OpenAI router so
        // the GUI dropdown is populated with provider's models.
        let app_models = vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()];
        let slugs = catalog_slugs(None, Some(&app_models), false);
        assert_eq!(slugs, vec!["deepseek-chat", "deepseek-reasoner"]);
    }

    #[test]
    fn catalog_slugs_skips_all_openai_shaped_app_list() {
        // When every discovered slug is OpenAI-shaped, codex's bundled catalog
        // already serves them with correct metadata (272k context window,
        // proper reasoning fields, etc.). Overwriting with our slim entries
        // would silently degrade the GUI. catalog_slugs should defer.
        let app_models = vec![
            "gpt-5".to_string(),
            "gpt-5-codex".to_string(),
            "o3-mini".to_string(),
        ];
        let slugs = catalog_slugs(None, Some(&app_models), true);
        assert!(
            slugs.is_empty(),
            "all-OpenAI list should defer to codex's bundled catalog"
        );
    }

    #[test]
    fn catalog_slugs_writes_when_mixed_with_non_openai() {
        let app_models = vec!["gpt-5".to_string(), "deepseek-chat".to_string()];
        let slugs = catalog_slugs(None, Some(&app_models), false);
        assert_eq!(slugs, vec!["deepseek-chat", "gpt-5"]);
    }

    #[test]
    fn catalog_slugs_merges_explicit_model_into_app_list() {
        let app_models = vec!["deepseek-chat".to_string()];
        let slugs = catalog_slugs(Some("my-custom-model"), Some(&app_models), false);
        assert_eq!(slugs, vec!["deepseek-chat", "my-custom-model"]);
    }

    #[test]
    fn catalog_slugs_explicit_model_breaks_all_openai_defer() {
        // Non-OpenAI `-m` forces a catalog even when all discovered slugs are
        // OpenAI-shaped — else the custom model stays invisible.
        let app_models = vec!["gpt-5".to_string(), "o3-mini".to_string()];
        let slugs = catalog_slugs(Some("deepseek-chat"), Some(&app_models), true);
        assert_eq!(slugs, vec!["deepseek-chat", "gpt-5", "o3-mini"]);
    }

    #[test]
    fn catalog_slugs_dedups_explicit_model_already_in_app_list() {
        let app_models = vec!["deepseek-chat".to_string(), "deepseek-reasoner".to_string()];
        let slugs = catalog_slugs(Some("deepseek-chat"), Some(&app_models), false);
        assert_eq!(slugs, vec!["deepseek-chat", "deepseek-reasoner"]);
    }

    #[test]
    fn catalog_slugs_rejects_control_bytes_in_explicit_model() {
        // The chained `-m` is filtered too, not just the discovered list.
        let app_models = vec!["deepseek-chat".to_string()];
        let slugs = catalog_slugs(Some("evil\n[features]\nfoo=true"), Some(&app_models), false);
        assert_eq!(slugs, vec!["deepseek-chat"]);
    }

    #[test]
    fn catalog_slugs_filters_control_byte_slugs() {
        // A buggy /v1/models endpoint returning a slug with a newline must
        // not flow through to TOML formatting.
        let app_models = vec![
            "good-model".to_string(),
            "evil\n[features]\nfoo=true".to_string(),
        ];
        let slugs = catalog_slugs(None, Some(&app_models), false);
        assert_eq!(slugs, vec!["good-model"]);
    }

    #[test]
    fn is_safe_codex_slug_rejects_control_chars() {
        assert!(is_safe_codex_slug("deepseek-v4-flash"));
        assert!(is_safe_codex_slug("provider/model-name"));
        assert!(!is_safe_codex_slug(""));
        assert!(!is_safe_codex_slug("with\nnewline"));
        assert!(!is_safe_codex_slug("with\ttab"));
        assert!(!is_safe_codex_slug("with\0nul"));
    }

    #[test]
    fn claude_prompt_after_teammate_mode() {
        let args = vec!["fix the login bug".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Claude, &args);
        assert_eq!(
            result,
            vec!["--teammate-mode", "in-process", "fix the login bug"]
        );
    }

    #[test]
    fn codex_prompt_after_model_flag() {
        let args = vec!["refactor this function".to_string()];
        let result = inject_codex_model(Some("gpt-4o"), &args, false);
        assert_eq!(result, vec!["-m", "gpt-4o", "refactor this function"]);
    }

    #[test]
    fn pi_prompt_after_model_flag() {
        let args = vec!["explain this code".to_string()];
        let result = inject_pi_model(Some("gpt-4o"), &args);
        assert_eq!(result, vec!["--model", "aivo/gpt-4o", "explain this code"]);
    }

    #[test]
    fn gemini_prompt_passes_through() {
        let args = vec!["explain this code".to_string()];
        let result = inject_claude_teammate_mode(AIToolType::Gemini, &args);
        assert_eq!(result, vec!["explain this code"]);
    }

    #[tokio::test]
    async fn opencode_prompt_passes_through_build_runtime_args() {
        let args = vec!["explain this code".to_string()];
        let env = HashMap::new();
        let result = build_runtime_args(AIToolType::Opencode, &args, None, None, &env)
            .await
            .unwrap();
        assert_eq!(result.args, vec!["explain this code"]);
    }

    #[test]
    fn inject_codex_max_context_appends_config_arg() {
        let mut args = vec!["-m".to_string(), "gpt-5".to_string()];
        inject_codex_max_context(&mut args, Some("1m"));
        assert_eq!(
            args,
            vec!["-m", "gpt-5", "--config", "model_context_window=1000000"]
        );
    }

    #[test]
    fn inject_codex_max_context_handles_multi_digit_tags() {
        let mut args: Vec<String> = vec![];
        inject_codex_max_context(&mut args, Some("12m"));
        assert_eq!(args, vec!["--config", "model_context_window=12000000"]);
    }

    #[test]
    fn inject_codex_max_context_noop_when_unset() {
        let mut args = vec!["existing".to_string()];
        inject_codex_max_context(&mut args, None);
        assert_eq!(args, vec!["existing"]);
    }

    #[test]
    fn inject_codex_max_context_noop_on_malformed_tag() {
        // Defensive: callers should pass canonical `<N>m`, but if junk slips
        // through (e.g. a future code path forgets to validate), we silently
        // skip rather than appending a garbage `--config` value.
        let mut args = vec!["existing".to_string()];
        inject_codex_max_context(&mut args, Some("foo"));
        assert_eq!(args, vec!["existing"]);
    }

    #[test]
    fn test_inject_codex_provider_config_direct_openai() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into()),
            ("OPENAI_API_KEY".into(), "sk-test-key".into()),
        ]);
        let mut args = vec!["-m".into(), "o4-mini".into()];
        inject_codex_provider_config(&mut env, &mut args);

        assert!(!env.contains_key("OPENAI_BASE_URL"));
        assert!(!env.contains_key("OPENAI_API_KEY"));
        assert_eq!(env.get("AIVO_CODEX_API_KEY").unwrap(), "sk-test-key");
        assert_eq!(
            args,
            vec![
                "--config",
                "model_provider=\"aivo\"",
                "--config",
                "model_providers.aivo.name=\"aivo\"",
                "--config",
                "model_providers.aivo.base_url=\"https://api.openai.com/v1\"",
                "--config",
                "model_providers.aivo.env_key=\"AIVO_CODEX_API_KEY\"",
                "--disable",
                "apps",
                "-m",
                "o4-mini",
            ]
        );
    }

    #[test]
    fn test_inject_codex_provider_config_local_router() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "http://127.0.0.1:54321".into()),
            ("OPENAI_API_KEY".into(), "provider-key".into()),
        ]);
        let mut args = vec!["-m".into(), "claude-sonnet-4-6".into()];
        inject_codex_provider_config(&mut env, &mut args);

        assert_eq!(env.get("AIVO_CODEX_API_KEY").unwrap(), "provider-key");
        assert!(args[5].contains("http://127.0.0.1:54321"));
    }

    #[test]
    fn test_inject_codex_provider_config_ollama() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "http://127.0.0.1:12345".into()),
            ("OPENAI_API_KEY".into(), "ollama".into()),
        ]);
        let mut args = vec![];
        inject_codex_provider_config(&mut env, &mut args);

        assert_eq!(env.get("AIVO_CODEX_API_KEY").unwrap(), "ollama");
        assert!(args.contains(&"model_provider=\"aivo\"".to_string()));
    }

    #[test]
    fn test_inject_codex_provider_config_noop_without_base_url() {
        let mut env = HashMap::from([("OPENAI_API_KEY".into(), "sk-key".into())]);
        let mut args = vec!["prompt".into()];
        inject_codex_provider_config(&mut env, &mut args);

        assert_eq!(env.get("OPENAI_API_KEY").unwrap(), "sk-key");
        assert_eq!(args, vec!["prompt"]);
    }

    #[test]
    fn test_inject_codex_provider_config_noop_without_api_key() {
        let mut env =
            HashMap::from([("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into())]);
        let mut args = vec!["prompt".into()];
        inject_codex_provider_config(&mut env, &mut args);

        // base_url should be restored
        assert!(env.contains_key("OPENAI_BASE_URL"));
        assert_eq!(args, vec!["prompt"]);
    }

    #[test]
    fn test_inject_codex_provider_config_skips_if_model_provider_in_args() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into()),
            ("OPENAI_API_KEY".into(), "sk-key".into()),
        ]);
        let mut args = vec![
            "--config".into(),
            "model_provider=\"custom\"".into(),
            "-m".into(),
            "gpt-4o".into(),
        ];
        inject_codex_provider_config(&mut env, &mut args);

        // Should not modify anything
        assert!(env.contains_key("OPENAI_BASE_URL"));
        assert!(env.contains_key("OPENAI_API_KEY"));
        assert!(!env.contains_key("AIVO_CODEX_API_KEY"));
    }

    #[test]
    fn test_inject_codex_provider_config_preserves_existing_args() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into()),
            ("OPENAI_API_KEY".into(), "sk-key".into()),
        ]);
        let mut args = vec![
            "--config".into(),
            "model_catalog_json=\"/tmp/cat.json\"".into(),
            "-m".into(),
            "gpt-4o".into(),
            "fix bug".into(),
        ];
        inject_codex_provider_config(&mut env, &mut args);

        // Config flags + --disable apps prepended, original args at the end
        assert_eq!(args[8], "--disable");
        assert_eq!(args[9], "apps");
        assert_eq!(args[10], "--config");
        assert_eq!(args[11], "model_catalog_json=\"/tmp/cat.json\"");
        assert_eq!(args[12], "-m");
        assert_eq!(args[13], "gpt-4o");
        assert_eq!(args[14], "fix bug");
    }

    #[test]
    fn test_rewrite_codex_preview_env() {
        let mut env = HashMap::from([
            ("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into()),
            ("OPENAI_API_KEY".into(), "sk-key".into()),
            ("CODEX_MODEL".into(), "gpt-4o".into()),
        ]);
        rewrite_codex_preview_env(&mut env);

        assert!(!env.contains_key("OPENAI_BASE_URL"));
        assert!(!env.contains_key("OPENAI_API_KEY"));
        assert_eq!(env.get("AIVO_CODEX_API_KEY").unwrap(), "sk-key");
        assert_eq!(env.get("CODEX_MODEL").unwrap(), "gpt-4o");
    }

    #[test]
    fn test_preview_codex_provider_config_args_with_base_url() {
        let env = HashMap::from([("OPENAI_BASE_URL".into(), "https://api.openai.com/v1".into())]);
        let args = vec!["-m".into(), "gpt-4o".into()];
        let result = preview_codex_provider_config_args(&env, args);

        assert_eq!(result[0], "--config");
        assert_eq!(result[1], "model_provider=\"aivo\"");
        assert!(result[5].contains("https://api.openai.com/v1"));
        assert_eq!(result[8], "--disable");
        assert_eq!(result[9], "apps");
        assert_eq!(result[10], "-m");
        assert_eq!(result[11], "gpt-4o");
    }

    #[test]
    fn test_preview_codex_provider_config_args_placeholder_url() {
        let env = HashMap::from([("OPENAI_BASE_URL".into(), PLACEHOLDER_LOOPBACK_URL.into())]);
        let args = vec!["-m".into(), "model".into()];
        let result = preview_codex_provider_config_args(&env, args);

        assert!(result[5].contains("http://127.0.0.1:<port>"));
    }

    #[test]
    fn test_preview_codex_provider_config_args_noop_without_base_url() {
        let env = HashMap::new();
        let args = vec!["-m".into(), "gpt-4o".into()];
        let result = preview_codex_provider_config_args(&env, args);

        assert_eq!(result, vec!["-m", "gpt-4o"]);
    }

    #[test]
    fn test_inject_amp_no_ide_prepends_when_bridge_active() {
        // Bridge active + user didn't pick a side → prepend `--no-ide` so
        // amp doesn't auto-prefix open-IDE file content to messages going
        // through the bridge to a third-party upstream.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["--mode".into(), "smart".into()];
        let result = inject_amp_no_ide(&args, &env);
        assert_eq!(result, vec!["--no-ide", "--mode", "smart"]);
    }

    #[test]
    fn test_inject_amp_no_ide_skips_for_native_amp() {
        // Native amp (no bridge) → user's data only goes back to
        // Sourcegraph, no leak risk. Leave `--ide` behavior at amp's
        // default rather than silently disabling a useful feature.
        let env = HashMap::new();
        let args = vec!["thread".into(), "list".into()];
        let result = inject_amp_no_ide(&args, &env);
        assert_eq!(result, vec!["thread", "list"]);
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_fires_for_one_shot_under_bridge() {
        // Bridge active + amp invoked non-interactively (`-x "..."`) → there
        // is no human to answer tool-approval prompts, so prepend the flag
        // so amp actually completes instead of hanging.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["-x".into(), "fix the failing test".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert_eq!(
            result,
            vec!["--dangerously-allow-all", "-x", "fix the failing test"]
        );
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_fires_for_stream_json_input() {
        // `--stream-json-input` is amp's programmatic path — same story as
        // `-x`: no human in the loop, so auto-allow.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["--stream-json-input".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert!(
            result
                .first()
                .is_some_and(|a| a == "--dangerously-allow-all")
        );
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_skips_interactive() {
        // No `-x` / `--execute` / `--stream-json-input` → user is interactive
        // and CAN answer tool prompts. Don't widen permissions silently.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["--mode".into(), "smart".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert_eq!(result, vec!["--mode", "smart"]);
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_skips_native_amp() {
        // No bridge → user is talking directly to ampcode.com; they own the
        // permissions story, don't auto-widen.
        let env = HashMap::new();
        let args = vec!["-x".into(), "ship it".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert_eq!(result, vec!["-x", "ship it"]);
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_idempotent() {
        // User already passed the flag → don't double-inject.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["--dangerously-allow-all".into(), "-x".into(), "go".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert_eq!(result, args);
        assert_eq!(
            result
                .iter()
                .filter(|a| *a == "--dangerously-allow-all")
                .count(),
            1
        );
    }

    #[test]
    fn test_inject_amp_dangerously_allow_all_handles_execute_with_equals() {
        // `--execute=hi` (single token) is the same non-interactive mode as
        // `-x hi` / `--execute hi` — must trigger.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let args = vec!["--execute=hi".into()];
        let result = inject_amp_dangerously_allow_all(&args, &env);
        assert!(
            result
                .first()
                .is_some_and(|a| a == "--dangerously-allow-all")
        );
    }

    #[test]
    fn test_inject_amp_no_ide_respects_explicit_user_flag() {
        // User passed `--ide` explicitly even with the bridge active —
        // they've made a deliberate choice; don't override.
        let env = HashMap::from([("AIVO_USE_AMP_BRIDGE".into(), "1".into())]);
        let with_ide = vec!["--ide".into(), "prompt".into()];
        assert_eq!(inject_amp_no_ide(&with_ide, &env), with_ide);

        // Same with explicit `--no-ide` — don't double-inject.
        let with_no_ide = vec!["--no-ide".into(), "prompt".into()];
        let result = inject_amp_no_ide(&with_no_ide, &env);
        assert_eq!(result, vec!["--no-ide", "prompt"]);
        assert_eq!(result.iter().filter(|a| *a == "--no-ide").count(), 1);
    }

    #[test]
    fn test_inject_amp_initial_mode_prepends_when_env_set() {
        let env = HashMap::from([("AIVO_AMP_INITIAL_MODE".into(), "rush".into())]);
        let args = vec!["prompt text".into()];
        let result = inject_amp_initial_mode(&args, &env);
        assert_eq!(result, vec!["--mode", "rush", "prompt text"]);
    }

    #[test]
    fn test_inject_amp_initial_mode_skips_when_user_already_passed_mode() {
        // User typed `aivo amp -- --mode smart`; their explicit pick wins
        // even if `--mode <X>` was set on aivo's side.
        let env = HashMap::from([("AIVO_AMP_INITIAL_MODE".into(), "rush".into())]);
        let args = vec!["--mode".into(), "smart".into()];
        let result = inject_amp_initial_mode(&args, &env);
        assert_eq!(result, vec!["--mode", "smart"]);
        assert_eq!(result.iter().filter(|a| *a == "--mode").count(), 1);

        let args_eq = vec!["--mode=deep".into()];
        let result_eq = inject_amp_initial_mode(&args_eq, &env);
        assert_eq!(result_eq, vec!["--mode=deep"]);
    }

    #[test]
    fn test_inject_amp_initial_mode_noop_when_env_unset() {
        let env = HashMap::new();
        let args = vec!["prompt".into()];
        let result = inject_amp_initial_mode(&args, &env);
        assert_eq!(result, vec!["prompt"]);
    }
}
