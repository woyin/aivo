//! AILauncher service for spawning AI tool processes.
//! Handles process spawning with environment injection and stdio passthrough.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::process::Stdio;
use std::time::Instant;
use tokio::process::Command;
#[cfg(unix)]
use tokio::signal;

use crate::errors::{CLIError, ErrorCategory};
use crate::services::environment_injector::{ClaudeModelOverrides, EnvironmentInjector};
use crate::services::launch_args::{
    build_preview_notes, build_runtime_args, inject_codex_provider_config, inject_codex_root_model,
    merge_preview_env, preview_args, rewrite_codex_preview_env,
};
use crate::services::launch_runtime::{
    cleanup_runtime_artifacts, finalize_codex_oauth, persist_runtime_discoveries,
    prepare_runtime_env, process_pi_sessions, record_launch_state,
};
use crate::services::log_store::{LogEvent, new_log_id};
use crate::services::model_names::{is_gpt_chat_model_name, is_openai_style_model_name};
use crate::services::models_cache::ModelsCache;
use crate::services::native_session_probe::SessionProbe;
use crate::services::ollama;
use crate::services::path_search::{collect_path_dirs, collect_path_dirs_from, find_in_dirs};
use crate::services::provider_profile::{
    is_aivo_starter_base, is_copilot_base, is_direct_openai_base, is_ollama_base,
    provider_profile_for_base_url,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::route_cache::PersistedRoute;
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, OpenAICompatibilityMode, SessionStore,
};

/// Seed a tool's default (`""`) route in-memory so the router starts on the
/// cli-native protocol; per-model learning then writes the durable result.
fn seed_default_route(key: &mut ApiKey, tool: &str, protocol: &str) {
    key.protocol_routes
        .entry(tool.to_string())
        .or_default()
        .entry(String::new())
        .or_insert_with(|| PersistedRoute {
            protocol: protocol.to_string(),
            path_variant: String::new(),
        });
}

/// Supported AI tool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIToolType {
    Claude,
    Codex,
    CodexApp,
    Gemini,
    Opencode,
    Pi,
}

impl AIToolType {
    /// Parses a string into an AIToolType
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claude" => Some(Self::Claude),
            "codex" => Some(Self::Codex),
            "codex-app" => Some(Self::CodexApp),
            "gemini" => Some(Self::Gemini),
            "opencode" => Some(Self::Opencode),
            "pi" => Some(Self::Pi),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::CodexApp => "codex-app",
            Self::Gemini => "gemini",
            Self::Opencode => "opencode",
            Self::Pi => "pi",
        }
    }

    pub fn command_name(&self) -> &'static str {
        match self {
            Self::CodexApp => "codex",
            _ => self.as_str(),
        }
    }

    /// One-line detail shown next to the tool name in the picker and the
    /// generic `aivo run` help.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Claude => "Anthropic's official terminal coding agent.",
            Self::Codex => "OpenAI's official Codex CLI.",
            Self::CodexApp => "OpenAI's official Codex desktop app (experimental).",
            Self::Gemini => "Google's official Gemini CLI.",
            Self::Opencode => "An open-source coding agent.",
            Self::Pi => "A terminal coding agent from the pi-mono toolkit.",
        }
    }

    /// False when the tool cannot run on this OS. Pickers and generic help
    /// hide unsupported tools; an explicit `aivo run codex-app` still gets
    /// the launch-time platform error.
    pub fn supported_on_current_platform(&self) -> bool {
        !matches!(self, Self::CodexApp) || cfg!(any(target_os = "macos", windows))
    }

    /// Best-effort "binary already on this machine" probe for picker hints.
    /// The launch path re-resolves PATH and offers an install, so a false
    /// negative only mislabels a row.
    pub fn looks_installed(&self) -> bool {
        if matches!(self, Self::CodexApp) {
            // No readable bundle dir on Windows; the materialized codex is
            // the cheap synchronous signal there.
            #[cfg(windows)]
            return crate::services::codex_app_wrapper::locate_bundled_codex().is_some();
            #[cfg(not(windows))]
            return crate::services::codex_app_wrapper::locate_codex_app().is_some();
        }
        find_in_dirs(self.command_name(), &collect_path_dirs()).is_some()
            || find_in_dirs(self.command_name(), &self.well_known_install_dirs()).is_some()
    }

    pub fn is_codex_family(&self) -> bool {
        matches!(self, Self::Codex | Self::CodexApp)
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Claude,
            Self::Codex,
            Self::CodexApp,
            Self::Gemini,
            Self::Opencode,
            Self::Pi,
        ]
    }

    /// Returns `Some(reason)` when `key` is an OAuth credential that can't be
    /// used to launch this tool (e.g. a Codex OAuth key for `aivo run claude`),
    /// or `None` when the key is compatible.
    pub fn oauth_incompat_reason(&self, key: &ApiKey) -> Option<&'static str> {
        let matches_tool = (*self == AIToolType::Claude && key.is_claude_oauth())
            || (self.is_codex_family() && key.is_codex_oauth());
        if matches_tool {
            None
        } else {
            key.oauth_run_requirement()
        }
    }

    /// Returns installation instructions for the tool (platform-appropriate).
    pub fn install_hint(&self) -> &'static str {
        #[cfg(unix)]
        match self {
            Self::Claude => "curl -fsSL https://claude.ai/install.sh | bash",
            Self::Codex | Self::CodexApp => "npm install -g @openai/codex",
            Self::Gemini => "npm install -g @google/gemini-cli",
            Self::Opencode => "curl -fsSL https://opencode.ai/install | bash",
            Self::Pi => "npm install -g @earendil-works/pi-coding-agent",
        }
        #[cfg(not(unix))]
        match self {
            Self::Claude => "npm install -g @anthropic-ai/claude-code",
            Self::Codex | Self::CodexApp => "npm install -g @openai/codex",
            Self::Gemini => "npm install -g @google/gemini-cli",
            Self::Opencode => "npm install -g opencode-ai",
            Self::Pi => "npm install -g @earendil-works/pi-coding-agent",
        }
    }

    /// Directories the tool's installer is known to drop its binary into,
    /// outside of the typical `$PATH`. Used as a fallback after a fresh
    /// install when PATH lookup fails (e.g. the Claude installer writes to
    /// `~/.local/bin`, which isn't on PATH for the current shell yet).
    pub fn well_known_install_dirs(&self) -> Vec<std::path::PathBuf> {
        use std::path::PathBuf;
        let mut dirs: Vec<PathBuf> = Vec::new();

        if let Some(home) = crate::services::system_env::home_dir() {
            // Common locations across installers (curl scripts, user-level
            // npm prefixes, bun).
            dirs.push(home.join(".local").join("bin"));
            dirs.push(home.join(".npm-global").join("bin"));
            dirs.push(home.join(".bun").join("bin"));
            // Tool-specific installer paths.
            match self {
                Self::Claude => dirs.push(home.join(".claude").join("local")),
                Self::Opencode => dirs.push(home.join(".opencode").join("bin")),
                _ => {}
            }
            #[cfg(windows)]
            {
                if let Some(appdata) = std::env::var_os("APPDATA") {
                    dirs.push(PathBuf::from(appdata).join("npm"));
                }
            }
        }

        #[cfg(unix)]
        {
            dirs.push(PathBuf::from("/usr/local/bin"));
            dirs.push(PathBuf::from("/opt/homebrew/bin"));
        }

        dirs
    }
}

/// Launch options for AI tools
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub tool: AIToolType,
    pub args: Vec<String>,
    pub model: Option<String>,
    /// Per-slot Claude model overrides (six addressable slots) plus the
    /// shared `max_context` tag. All fields are Claude-only — the slot flags
    /// trigger one stderr warning per slot on other tools and are then
    /// dropped; `max_context` is rejected up-front for non-Claude tools (see
    /// `for_claude_with_overrides` for how Claude consumes it).
    pub claude_overrides: ClaudeModelOverrides,
    pub env: Option<HashMap<String, String>>,
    /// Temporary key override for this launch (does not persist to config)
    pub key_override: Option<ApiKey>,
}

/// Tool configuration including command and environment variables
#[derive(Debug, Clone)]
pub struct ToolConfig {
    pub command: String,
    pub env_vars: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct PreparedLaunch {
    pub tool: AIToolType,
    pub key: ApiKey,
    pub command: String,
    pub model: Option<String>,
    pub args: Vec<String>,
    pub env_vars: HashMap<String, String>,
    pub notes: Vec<String>,
}

/// AILauncher spawns AI tool processes with configured environment and stdio passthrough
#[derive(Debug, Clone)]
pub struct AILauncher {
    session_store: SessionStore,
    env_injector: EnvironmentInjector,
    cache: ModelsCache,
}

impl AILauncher {
    /// Creates a new AILauncher
    pub fn new(
        session_store: SessionStore,
        env_injector: EnvironmentInjector,
        cache: ModelsCache,
    ) -> Self {
        Self {
            session_store,
            env_injector,
            cache,
        }
    }

    /// Spawns an AI tool with configured environment and stdio passthrough
    pub async fn launch(&self, options: &LaunchOptions) -> Result<i32> {
        let mut resolved = self.resolve_launch_context(options, true).await?;

        // Ollama lifecycle: ensure server is running and model is pulled
        if is_ollama_base(&resolved.key.base_url) {
            ollama::ensure_ready().await?;
            if let Some(ref model) = resolved.model {
                ollama::ensure_model(model).await?;
            }
        }

        self.output_key_info(&resolved.key);

        // Preflight: resolve the codex-app GUI launcher and bundled codex
        // once, failing closed with a clear error before any argv surgery.
        #[cfg_attr(
            not(any(target_os = "macos", windows)),
            allow(unused_mut, unused_variables)
        )]
        let mut codex_app_desktop: Option<CodexAppDesktop> = None;
        if options.tool == AIToolType::CodexApp {
            #[cfg(not(any(target_os = "macos", windows)))]
            {
                return Err(CLIError::new(
                    "`aivo codex-app` requires the Codex desktop app, which ships only for macOS and Windows.",
                    ErrorCategory::User,
                    None::<String>,
                    Some("Use `aivo codex` (CLI) instead."),
                )
                .into());
            }
            #[cfg(target_os = "macos")]
            {
                let Some(app) = crate::services::codex_app_wrapper::locate_codex_app() else {
                    return Err(CLIError::new(
                        "Codex desktop app not found. `aivo codex-app` requires it (ChatGPT.app, formerly Codex.app).",
                        ErrorCategory::User,
                        None::<String>,
                        Some("Install it from https://chatgpt.com/codex (or set AIVO_CODEX_APP_PATH to its bundle path)"),
                    )
                    .into());
                };
                // Only reachable via an AIVO_CODEX_APP_PATH override; named
                // locate candidates already require the bundled codex.
                let Some(codex_bin) = crate::services::codex_app_wrapper::bundled_codex_in(&app)
                else {
                    return Err(CLIError::new(
                        "Codex desktop app bundle has no Contents/Resources/codex.",
                        ErrorCategory::User,
                        None::<String>,
                        Some("Check that AIVO_CODEX_APP_PATH points at the real app bundle."),
                    )
                    .into());
                };
                codex_app_desktop = Some(CodexAppDesktop {
                    gui: app,
                    codex_bin,
                });
            }
            #[cfg(windows)]
            {
                let Some(pkg) = windows_codex_package().await else {
                    return Err(CLIError::new(
                        "Codex desktop app not found. `aivo codex-app` requires the Store package (OpenAI.Codex).",
                        ErrorCategory::User,
                        None::<String>,
                        Some("Install it from the Microsoft Store: https://apps.microsoft.com/detail/9plm9xgg6vks"),
                    )
                    .into());
                };
                let Some(gui) = windows_codex_gui_exe(&pkg) else {
                    return Err(CLIError::new(
                        "Codex desktop app package found, but no launchable executable.",
                        ErrorCategory::User,
                        None::<String>,
                        Some("aivo must spawn the app directly to route it; protocol activation would drop the key routing. Please report this with your app version."),
                    )
                    .into());
                };
                let Some(codex_bin) = crate::services::codex_app_wrapper::locate_bundled_codex()
                else {
                    return Err(CLIError::new(
                        "Codex desktop app is installed but its codex runtime isn't materialized yet.",
                        ErrorCategory::User,
                        None::<String>,
                        Some("Launch the desktop app once (it installs its codex runtime), quit it, then re-run `aivo codex-app`."),
                    )
                    .into());
                };
                codex_app_desktop = Some(CodexAppDesktop { gui, codex_bin });
            }
        }

        // Preflight: Codex.app is a singleton on macOS, and `CODEX_CLI_PATH`
        // is captured at GUI launch — `aivo codex-app` against an already-
        // running instance silently fails to route. Prompt the user to
        // restart so the new wrapper/key takes effect.
        if options.tool == AIToolType::CodexApp {
            preflight_codex_app_running().await?;
        }

        let env = self
            .env_injector
            .merge(&resolved.tool_config.env_vars, options.env.as_ref());
        let mut env = env;
        if options.tool == AIToolType::CodexApp {
            env.insert(
                "AIVO_CODEX_APP_HOME_KEY".to_string(),
                resolved.key.id.clone(),
            );
        }
        let mut runtime = prepare_runtime_env(options.tool, env, &self.session_store).await?;

        let mut runtime_args = build_runtime_args(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            resolved.codex_app_models.as_deref(),
            &runtime.env,
            &resolved.tool_config.env_vars,
            &self.cache,
            Some(resolved.key.base_url.as_str()),
        )
        .await?;

        let mut codex_app_wrapper_path: Option<std::path::PathBuf> = None;
        if options.tool.is_codex_family() {
            inject_codex_provider_config(&mut runtime.env, &mut runtime_args.args);
            if options.tool == AIToolType::CodexApp {
                inject_codex_root_model(&mut runtime_args.args, resolved.model.as_deref());
                crate::services::launch_args::inject_codex_app_subcommand(&mut runtime_args.args);
                if let Some(desktop) = codex_app_desktop.as_ref() {
                    codex_app_wrapper_path = install_codex_app_wrapper(
                        &mut runtime.env,
                        &mut runtime_args.args,
                        self.session_store.config_dir(),
                        &resolved.key.id,
                        &desktop.codex_bin,
                    )
                    .await;
                    install_codex_app_models_cache(
                        &runtime.env,
                        runtime_args.codex_model_catalog_path.as_deref(),
                    )
                    .await;
                    // Launch the GUI ourselves, not via `codex app`: stable
                    // codex CLIs (≤0.144.x) still search only `Codex.app`
                    // post-rename and auto-download a duplicate installer,
                    // and upstream's Windows `codex://` protocol activation
                    // drops aivo's env — with it the CODEX_CLI_PATH hook.
                    // macOS mirrors upstream's open_codex_app (`open -a`).
                    let (command, args) = codex_app_launch_invocation(desktop, &runtime_args.args);
                    resolved.tool_config.command = command;
                    runtime_args.args = args;
                }
            }
        }

        // The cursor router marker was left in place through
        // `build_runtime_args` so codex's catalog/model-passthrough branch
        // could detect cursor mode. Strip it now so it doesn't leak to the
        // spawned tool.
        runtime.env.remove("AIVO_USE_CURSOR_ROUTER");

        let event_group_id = new_log_id();
        let cwd = crate::services::system_env::current_dir_string();
        let log_args = runtime_args.args.clone();

        // Check if the tool binary is available on PATH before attempting to spawn.
        // When found, pin `tool_config.command` to the full resolved path so the
        // spawn step picks up the correct extension on Windows — CreateProcessW
        // does not honor PATHEXT for non-.exe files, so a bare `claude` would
        // fail to spawn even when `claude.cmd` is on PATH.
        // Absolute commands (codex-app desktop launchers) spawn as-is: the
        // Windows app-execution alias is a reparse point the executability
        // probe can misjudge.
        if !std::path::Path::new(&resolved.tool_config.command).is_absolute() {
            let path_dirs = collect_path_dirs();
            if let Some(found) = find_in_dirs(&resolved.tool_config.command, &path_dirs) {
                resolved.tool_config.command = found.to_string_lossy().into_owned();
            } else {
                let tool = options.tool;

                let not_installed = || -> Result<()> {
                    eprintln!(
                        "{} '{}' is not installed or not found on PATH.",
                        crate::style::red("Error:"),
                        tool.as_str()
                    );
                    eprintln!();
                    eprintln!(
                        "  {}",
                        crate::style::dim(format!("Install: {}", tool.install_hint()))
                    );
                    anyhow::bail!("tool '{}' not found", tool.as_str());
                };

                if !std::io::stdin().is_terminal() {
                    not_installed()?;
                }

                eprint!(
                    "  {} '{}' is not installed. Install it? [Y/n] ",
                    crate::style::yellow("?"),
                    tool.as_str()
                );
                let _ = std::io::stderr().flush();

                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let trimmed = input.trim();

                if !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("y")
                    || trimmed.eq_ignore_ascii_case("yes"))
                {
                    not_installed()?;
                }

                eprintln!(
                    "  {} Installing {}...",
                    crate::style::arrow_symbol(),
                    tool.as_str()
                );

                #[cfg(unix)]
                let status = Command::new("sh")
                    .arg("-c")
                    .arg(tool.install_hint())
                    .stdin(Stdio::inherit())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .status()
                    .await;

                #[cfg(not(unix))]
                let status = Command::new("cmd")
                    .arg("/C")
                    .arg(tool.install_hint())
                    .stdin(Stdio::inherit())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .status()
                    .await;

                let status = status.context(format!(
                    "Failed to run install command for '{}'",
                    tool.as_str()
                ))?;

                if !status.success() {
                    anyhow::bail!(
                        "Installation of '{}' failed (exit code: {})",
                        tool.as_str(),
                        status.code().unwrap_or(-1)
                    );
                }

                // The installer may have added a new directory to PATH via shell
                // profile. Re-read PATH from a login shell so we pick it up.
                refresh_path_from_shell().await;

                // Use the freshened PATH for the post-install lookup if available,
                // otherwise fall back to the inherited PATH. Avoids mutating global
                // env state.
                let path_dirs = match freshened_path_for_lookup() {
                    Some(fresh) => collect_path_dirs_from(Some(fresh)),
                    None => collect_path_dirs(),
                };
                // Resolve the binary to a full path with extension. Required on
                // Windows so .cmd/.bat npm shims can be spawned via CreateProcessW.
                let resolved_path = find_in_dirs(&resolved.tool_config.command, &path_dirs)
                    .or_else(|| {
                        // PATH still doesn't see the binary (e.g. the installer
                        // wrote to `~/.local/bin` and added an `export PATH=...`
                        // line to a shell profile that this non-login shell hasn't
                        // sourced). Try the installer's well-known drop locations.
                        find_in_dirs(
                            &resolved.tool_config.command,
                            &tool.well_known_install_dirs(),
                        )
                    });
                match resolved_path {
                    Some(found) => {
                        eprintln!(
                            "  {} Found at {}",
                            crate::style::arrow_symbol(),
                            crate::style::dim(found.display().to_string())
                        );
                        resolved.tool_config.command = found.to_string_lossy().into_owned();
                    }
                    None => {
                        eprintln!(
                            "  {} '{}' was installed but not found on PATH. You may need to restart your shell.",
                            crate::style::yellow("!"),
                            tool.as_str()
                        );
                        anyhow::bail!("tool '{}' not found on PATH after install", tool.as_str());
                    }
                }

                eprintln!(
                    "  {} Installed successfully.\n",
                    crate::style::success_symbol()
                );
            }
        }

        let base_event = || LogEvent {
            source: "run".to_string(),
            kind: "tool_launch".to_string(),
            event_group_id: Some(event_group_id.clone()),
            key_id: Some(resolved.key.id.clone()),
            key_name: Some(resolved.key.display_name().to_string()),
            base_url: Some(resolved.key.base_url.clone()),
            tool: Some(options.tool.as_str().to_string()),
            model: resolved.model.clone(),
            cwd: cwd.clone(),
            title: Some(format!(
                "{} {}",
                options.tool.as_str(),
                resolved.key.display_name()
            )),
            ..Default::default()
        };

        let _ = self
            .session_store
            .logs()
            .append(LogEvent {
                phase: Some("started".to_string()),
                body_text: if log_args.is_empty() {
                    None
                } else {
                    Some(log_args.join(" "))
                },
                payload_json: Some(serde_json::json!({
                    "command": resolved.tool_config.command,
                    "args": log_args,
                })),
                ..base_event()
            })
            .await;

        // Snapshot the launched CLI's session dir so we can identify the
        // new session file it produces and link the [run] event to it.
        // Cwd-scoped where the layout supports it, so a heavy user with
        // thousands of historical sessions still pays only ms.
        let probe = SessionProbe::snapshot(options.tool, cwd.as_deref()).await;

        let child_result = self.spawn_child(
            &resolved.tool_config.command,
            &runtime_args.args,
            runtime.env,
            &runtime.env_unset,
        );
        let mut child = match child_result {
            Ok(child) => child,
            Err(err) => {
                let _ = self
                    .session_store
                    .logs()
                    .append(LogEvent {
                        phase: Some("finished".to_string()),
                        exit_code: Some(1),
                        duration_ms: Some(0),
                        body_text: Some(err.to_string()),
                        payload_json: Some(serde_json::json!({
                            "error": err.to_string(),
                        })),
                        ..base_event()
                    })
                    .await;
                return Err(err);
            }
        };

        record_launch_state(
            &self.session_store,
            &resolved.key,
            options.tool,
            resolved.model.as_deref(),
        )
        .await;

        let started_at = Instant::now();
        let result = self.wait_for_process(&mut child).await;

        // The launch child returns once the GUI is open; block on the GUI's
        // lifetime so aivo's local router survives the session.
        if options.tool == AIToolType::CodexApp {
            wait_for_codex_app_gui_exit().await;
            if let Some(path) = codex_app_wrapper_path.as_deref() {
                crate::services::codex_app_wrapper::remove_wrapper(path).await;
            }
        }

        if options.tool == AIToolType::Pi {
            process_pi_sessions(runtime.pi_agent_dir.as_deref()).await;
        }

        persist_runtime_discoveries(
            &self.session_store,
            &resolved.key,
            runtime.route_cache,
            runtime.learned_requires_reasoning,
        )
        .await;

        finalize_codex_oauth(&self.session_store, runtime.codex_oauth_sync).await;

        // The codex model catalog tempfile is referenced by the GUI's wrapper
        // for the duration of the Codex.app session. If we cleaned it up here
        // after a SIGINT-early-return from wait_for_codex_app_gui_exit, the
        // still-running GUI would lose its catalog on the next app-server
        // re-spawn (settings change, crash recovery). Leave it for codex-app
        // launches — `cleanup_stale_codex_catalogs` reaps it on subsequent
        // aivo runs.
        let catalog_to_clean = if options.tool == AIToolType::CodexApp {
            None
        } else {
            runtime_args.codex_model_catalog_path.as_deref()
        };
        cleanup_runtime_artifacts(
            catalog_to_clean,
            runtime_args.claude_settings_pin_path.as_deref(),
            runtime.pi_agent_dir.as_deref(),
        )
        .await;

        let exit_code = result.as_ref().ok().copied();
        let detected_session_id = probe.detect_new().await;
        let _ = self
            .session_store
            .logs()
            .append(LogEvent {
                phase: Some("finished".to_string()),
                exit_code: exit_code.map(i64::from),
                duration_ms: Some(started_at.elapsed().as_millis() as i64),
                session_id: detected_session_id,
                payload_json: Some(serde_json::json!({
                    "command": resolved.tool_config.command,
                    "args": runtime_args.args,
                })),
                ..base_event()
            })
            .await;

        result
    }

    pub async fn prepare_launch(&self, options: &LaunchOptions) -> Result<PreparedLaunch> {
        let resolved = self.resolve_launch_context(options, false).await?;
        let mut env_vars = merge_preview_env(&resolved.tool_config.env_vars, options.env.as_ref());
        let mut args = preview_args(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            &resolved.tool_config.env_vars,
        );
        let mut notes = build_preview_notes(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            &resolved.tool_config.env_vars,
        );

        if options.tool.is_codex_family() {
            rewrite_codex_preview_env(&mut env_vars);
            if options.tool == AIToolType::CodexApp {
                // Match the real launch ordering exactly so --dry-run isn't
                // misleading: root model first, then the `app` subcommand
                // insert. The actual launch then drains the global prefix
                // into a wrapper at CODEX_CLI_PATH (we document that via a
                // note rather than simulating the file write here).
                inject_codex_root_model(&mut args, resolved.model.as_deref());
                env_vars.insert(
                    "CODEX_HOME".to_string(),
                    crate::services::codex_home_shadow::CodexHomeShadow::persistent_path(
                        self.session_store.config_dir(),
                        &resolved.key.id,
                    )
                    .to_string_lossy()
                    .to_string(),
                );
                env_vars.insert(
                    "CODEX_CLI_PATH".to_string(),
                    "<temp:aivo-codex-wrapper>".to_string(),
                );
                crate::services::launch_args::inject_codex_app_subcommand(&mut args);
                notes.push(
                    "installs a per-launch wrapper at $AIVO_CONFIG/codex-app-wrappers/<key>-<pid>-<nanos>.sh (.exe shim on Windows), points CODEX_CLI_PATH at it, and moves the global --config prefix into the wrapper so the desktop app's app-server subprocess inherits the overrides"
                        .to_string(),
                );
                notes.push(
                    "writes a per-key models_cache.json into the shadow CODEX_HOME so the GUI's model picker reflects aivo's discovered models"
                        .to_string(),
                );
            }
        }

        Ok(PreparedLaunch {
            tool: options.tool,
            key: resolved.key,
            command: resolved.tool_config.command,
            model: resolved.model,
            args,
            env_vars,
            notes,
        })
    }

    async fn resolve_launch_context(
        &self,
        options: &LaunchOptions,
        persist: bool,
    ) -> Result<ResolvedLaunchContext> {
        let mut key = match &options.key_override {
            Some(k) => k.clone(),
            None => match self.session_store.get_active_key().await? {
                Some(k) => k,
                None => {
                    return Err(CLIError::new(
                        "No API key configured. Please add a key with 'aivo keys add'.",
                        ErrorCategory::Auth,
                        None::<String>,
                        Some("Run 'aivo keys add' to add an API key"),
                    )
                    .into());
                }
            },
        };

        // One-shot migration of routing fields written under older buggy logic.
        // Always runs before any router/injection step reads `responses_api_supported`.
        crate::services::launch_runtime::migrate_routing_schema_for_key(
            &self.session_store,
            &mut key,
        )
        .await;

        if is_ollama_base(&key.base_url) {
            // Ollama is always OpenAI-compatible; no protocol probing needed.
        } else if options.tool == AIToolType::Claude {
            key = self
                .resolve_claude_protocol(key, options.model.as_deref())
                .await?;
        } else if options.tool.is_codex_family() {
            key = self
                .resolve_codex_mode(key, persist && options.key_override.is_none())
                .await?;
        } else if options.tool == AIToolType::Gemini {
            key = self.resolve_gemini_protocol(key).await?;
        } else if options.tool == AIToolType::Opencode {
            key = self
                .resolve_opencode_mode(key, persist && options.key_override.is_none())
                .await?;
        }

        let (model, opencode_models) = if options.tool == AIToolType::Opencode {
            let (selected_model, discovered_models) = self
                .resolve_opencode_model_config(&key, options.model.as_deref())
                .await?;
            (selected_model, Some(discovered_models))
        } else {
            (options.model.clone(), None)
        };
        // Discover with `-m` too: the picker lists the whole catalog, not just
        // the pinned model. Gated on `persist` so dry-run skips the live fetch.
        let codex_app_models = if options.tool == AIToolType::CodexApp && persist {
            self.discover_codex_app_models(&key).await
        } else {
            None
        };
        // For CodexApp without -m, pick a default so codex doesn't ship an
        // empty model name.
        let model = if model.is_none() {
            codex_app_models
                .as_ref()
                .and_then(|m| pick_default_codex_app_model(m))
        } else {
            model
        };
        // Pi's /model picker only lists what's in its models.json; fetch the
        // provider catalog so it shows the whole list, not just the pinned
        // model. Soft-fails to empty (build_pi_models_json then writes the
        // single pinned entry). Gated on `persist` so dry-run skips the live
        // fetch; cursor keys use an ACP whitelist, not /v1/models.
        let pi_models = if options.tool == AIToolType::Pi && persist && !key.is_cursor_acp() {
            self.fetch_pi_models(&key).await
        } else {
            Vec::new()
        };
        // Per-model limits for the tools that can consume them: pi models.json,
        // opencode `limit`, claude CLAUDE_CODE_MAX_OUTPUT_TOKENS.
        let catalog_ids: Vec<&str> = match options.tool {
            AIToolType::Pi => pi_models.iter().map(String::as_str).collect(),
            AIToolType::Opencode => opencode_models
                .iter()
                .flatten()
                .map(String::as_str)
                .collect(),
            _ => Vec::new(),
        };
        let model_limits = if matches!(
            options.tool,
            AIToolType::Pi | AIToolType::Opencode | AIToolType::Claude
        ) {
            let cache_base = crate::services::model_catalog::model_cache_key_for_key(&key);
            let mut limits = HashMap::new();
            for id in catalog_ids.into_iter().chain(model.as_deref()) {
                if !limits.contains_key(id) {
                    let resolved = crate::services::model_metadata::resolve_limits(
                        &self.cache,
                        Some(&cache_base),
                        id,
                    )
                    .await;
                    limits.insert(id.to_string(), resolved);
                }
            }
            limits
        } else {
            HashMap::new()
        };
        let tool_config = self.get_tool_config(
            options.tool,
            &key,
            model.as_deref(),
            opencode_models.as_deref(),
            &pi_models,
            &model_limits,
            &options.claude_overrides,
        );
        Ok(ResolvedLaunchContext {
            key,
            model,
            codex_app_models,
            tool_config,
        })
    }

    /// Fetches the key's available models for CodexApp's GUI dropdown. Soft
    /// failure: returns `None` so the launch still works against codex's
    /// built-in catalog (the user can pass `-m` explicitly).
    ///
    /// Skipped for keys whose `/v1/models` endpoint isn't meaningful here:
    /// Codex OAuth (uses ChatGPT's hardcoded list, not /v1/models), Ollama
    /// and Copilot sentinels (custom URL schemes handled by their own
    /// runners). Avoids a pointless spinner + guaranteed failure on those.
    async fn discover_codex_app_models(&self, key: &ApiKey) -> Option<Vec<String>> {
        if key.is_codex_oauth() || is_ollama_base(&key.base_url) || is_copilot_base(&key.base_url) {
            return None;
        }
        let cache_key = crate::services::model_catalog::model_cache_key_for_key(key);
        if let Some(models) = self.cache.get(&cache_key).await {
            return Some(sanitize_discovered_slugs(models));
        }
        let client = crate::services::http_utils::router_http_client();
        let (spinning, spinner_handle) = crate::style::start_spinner(Some(" Fetching models..."));
        let result =
            crate::services::model_catalog::fetch_models_cached(&client, key, &self.cache, false)
                .await;
        crate::style::stop_spinner(&spinning);
        let _ = spinner_handle.await;
        result.ok().map(sanitize_discovered_slugs)
    }

    /// Outputs information about which key is being used
    fn output_key_info(&self, key: &ApiKey) {
        use crate::commands::truncate_url_for_display;
        use crate::style;

        eprintln!(
            "  {} Using key: {} {}",
            style::success_symbol(),
            style::cyan(key.display_name()),
            style::dim(format!("({})", truncate_url_for_display(&key.base_url, 50)))
        );
    }

    async fn resolve_claude_protocol(
        &self,
        mut key: ApiKey,
        _model: Option<&str>,
    ) -> Result<ApiKey> {
        let profile = provider_profile_for_base_url(&key.base_url);
        if profile.serve_flags.is_copilot || profile.serve_flags.is_openrouter {
            return Ok(key);
        }
        // Default guess only; the router learns/persists the working route
        // per model after confirming it.
        let proto = preferred_claude_protocol(&key.base_url);
        seed_default_route(&mut key, "claude", proto.as_str());
        Ok(key)
    }

    async fn resolve_codex_mode(&self, mut key: ApiKey, persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        if key.codex_mode.is_none() {
            key.codex_mode = Some(preferred_codex_mode(&key.base_url));
            if persist {
                let _ = self
                    .session_store
                    .set_key_codex_mode(&key.id, key.codex_mode)
                    .await;
            }
        }
        Ok(key)
    }

    async fn resolve_gemini_protocol(&self, mut key: ApiKey) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        // Legacy Gemini OAuth entries (sign-in flow removed) have no REST
        // endpoint, so no router protocol applies. They're rejected before
        // launch via `oauth_incompat_reason`; this guard just avoids seeding a
        // bogus route if such a key reaches protocol resolution another way.
        if key.is_gemini_oauth() {
            return Ok(key);
        }
        // Default guess only — see resolve_claude_protocol.
        let proto = preferred_gemini_protocol(&key.base_url);
        seed_default_route(&mut key, "gemini", proto.as_str());
        Ok(key)
    }

    async fn resolve_opencode_mode(&self, mut key: ApiKey, persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        let preferred = preferred_opencode_mode(&key.base_url);
        let has_stale_direct_pin = is_opencode_zen_base(&key.base_url)
            && key.opencode_mode == Some(OpenAICompatibilityMode::Direct)
            && preferred == OpenAICompatibilityMode::Router;
        if key.opencode_mode.is_none() || has_stale_direct_pin {
            key.opencode_mode = Some(preferred);
            if persist {
                let _ = self
                    .session_store
                    .set_key_opencode_mode(&key.id, key.opencode_mode)
                    .await;
            }
        }
        Ok(key)
    }

    async fn resolve_opencode_model_config(
        &self,
        key: &ApiKey,
        model: Option<&str>,
    ) -> Result<(Option<String>, Vec<String>)> {
        let requested_model = model.map(|m| m.strip_prefix("aivo/").unwrap_or(m).to_string());
        let client = crate::services::http_utils::router_http_client();
        let cache_key = crate::services::model_catalog::model_cache_key_for_key(key);

        // Check cache first — skip the spinner if we get a hit
        let fetch_result = if let Some(cached) = self.cache.get(&cache_key).await {
            Ok(cached)
        } else {
            // Cache miss: show spinner while fetching from network
            let (spinning, spinner_handle) =
                crate::style::start_spinner(Some(" Fetching models..."));

            // bypass_cache=true: we know it's a miss; fetch_models_cached will still write result to cache
            let result = crate::services::model_catalog::fetch_models_cached(
                &client,
                key,
                &self.cache,
                true,
            )
            .await;

            crate::style::stop_spinner(&spinning);
            let _ = spinner_handle.await;

            result
        };

        let mut models = match fetch_result {
            Ok(models) => models,
            Err(e) => {
                if let Some(requested_model) = requested_model.clone() {
                    return Ok((Some(requested_model.clone()), vec![requested_model]));
                }
                return Err(e).with_context(|| {
                    "Unable to determine an OpenCode model from your provider. Pass --model <provider/model>."
                });
            }
        };
        if let Some(requested_model) = requested_model {
            if !models.contains(&requested_model) {
                models.push(requested_model.clone());
            }
            models.sort();
            models.dedup();
            return Ok((Some(requested_model), models));
        }

        models.sort();
        models.dedup();

        // Default to a non-reasoning OpenAI-style model so cost/latency match a
        // typical workhorse session. Reverse iteration over the alphabetically
        // sorted list yields the newest version (gpt-5.5 over gpt-3.5-turbo).
        // Falls back to o-series, then to first alphabetical.
        let selected_model = models
            .iter()
            .rev()
            .find(|m| is_gpt_chat_model_name(m))
            .or_else(|| models.iter().rev().find(|m| is_openai_style_model_name(m)))
            .or_else(|| models.first())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No models returned by provider. Pass --model <provider/model> for opencode."
                )
            })?;
        Ok((Some(selected_model), models))
    }

    /// Fetches the key's available model ids for Pi's `/model` picker. Soft
    /// failure returns empty so the launch still works against the single
    /// pinned model (build_pi_models_json's fallback).
    ///
    /// Reads the `#all` namespace via `fetch_all_models_cached` — the same
    /// source the pre-launch model picker (`run`/`start`) warms. Reading the
    /// plain namespace here would always miss the picker's entry and re-fetch
    /// right after the user selected a model. Cache-first so the spinner only
    /// shows on a genuine miss.
    async fn fetch_pi_models(&self, key: &ApiKey) -> Vec<String> {
        let cache_key = crate::services::model_catalog::full_catalog_cache_key_for_key(key);
        if let Some((ids, metadata)) = self.cache.get_with_metadata(&cache_key).await
            && starter_catalog_window_cached(key, &metadata)
        {
            return ids;
        }
        let client = crate::services::http_utils::router_http_client();
        let (spinning, spinner_handle) = crate::style::start_spinner(Some(" Fetching models..."));
        let result = crate::services::model_catalog::fetch_all_models_cached(
            &client,
            key,
            &self.cache,
            true,
        )
        .await;
        crate::style::stop_spinner(&spinning);
        let _ = spinner_handle.await;
        result.unwrap_or_default()
    }

    /// Gets tool-specific configuration including command and environment variables
    #[allow(clippy::too_many_arguments)]
    fn get_tool_config(
        &self,
        tool: AIToolType,
        key: &ApiKey,
        model: Option<&str>,
        opencode_models: Option<&[String]>,
        pi_models: &[String],
        model_limits: &HashMap<String, crate::services::model_metadata::ResolvedLimits>,
        claude_overrides: &ClaudeModelOverrides,
    ) -> ToolConfig {
        // claude_overrides are non-empty only on the Claude path; for other
        // tools the run command warns or errors up-front and never populates
        // them.
        let env_vars = match tool {
            AIToolType::Claude => {
                let mut env =
                    self.env_injector
                        .for_claude_with_overrides(key, model, claude_overrides);
                // Claude Code's per-request output cap; user `--env` still
                // wins at the runtime merge.
                if let Some(output) = model
                    .and_then(|m| model_limits.get(m))
                    .and_then(|l| l.output)
                {
                    env.entry("CLAUDE_CODE_MAX_OUTPUT_TOKENS".to_string())
                        .or_insert_with(|| output.to_string());
                }
                env
            }
            AIToolType::Codex | AIToolType::CodexApp => self.env_injector.for_codex(key, model),
            AIToolType::Gemini => self.env_injector.for_gemini(key, model),
            AIToolType::Opencode => {
                self.env_injector
                    .for_opencode(key, model, opencode_models, model_limits)
            }
            AIToolType::Pi => self
                .env_injector
                .for_pi(key, model, pi_models, model_limits),
        };

        ToolConfig {
            command: tool.command_name().to_string(),
            env_vars,
        }
    }

    /// Spawns a child process with stdio inheritance and returns its exit code
    fn spawn_child(
        &self,
        command: &str,
        args: &[String],
        env: HashMap<String, String>,
        env_unset: &[String],
    ) -> Result<tokio::process::Child> {
        // On Termux, musl-static aivo bypasses the `termux-exec` LD_PRELOAD
        // shebang shim — so npm-installed CLIs (`#!/usr/bin/env node` …)
        // would fail `execve()` with ENOENT on `/usr/bin/env`. Rewrite the
        // shebang ourselves before spawning.
        let (effective_command, effective_args_owned) =
            match crate::services::termux_exec::rewrite_shebang(command, args) {
                Some((c, a)) => (c, Some(a)),
                None => (command.to_string(), None),
            };
        let effective_args: &[String] = effective_args_owned.as_deref().unwrap_or(args);
        let mut cmd = Command::new(&effective_command);
        cmd.args(effective_args)
            .envs(&env)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        // Drop inherited env vars the injector explicitly marked for removal.
        // Done after `.envs()` so a name appearing in both wins as removed —
        // matters when a caller's env carries a placeholder of the same name.
        for name in env_unset {
            cmd.env_remove(name);
        }
        // If a tool was just installed and pulled a new dir into PATH via
        // shell profile, propagate that PATH to the child. Done here rather
        // than via global set_var so we never race with concurrent tasks.
        apply_refreshed_path(&mut cmd);

        // kill_on_drop ensures the child is sent SIGKILL if aivo panics or
        // is SIGKILL'd before wait_for_process() can forward a graceful signal.
        cmd.kill_on_drop(true);
        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn {}", effective_command))?;
        Ok(child)
    }

    /// Waits for a child process while forwarding signals on Unix.
    #[cfg(unix)]
    async fn wait_for_process(&self, child: &mut tokio::process::Child) -> Result<i32> {
        // Get the child PID for signal forwarding
        let child_id = child.id();

        // Set up signal forwarding
        let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())?;
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())?;

        // Wait for the child to complete, while also listening for signals
        let result = tokio::select! {
            status = child.wait() => {
                status.map(|s| s.code().unwrap_or(1))
            }
            _ = sigint.recv() => {
                // Forward SIGINT to child
                if let Some(id) = child_id {
                    // SAFETY: `kill` does not dereference pointers; pid/signal values are plain integers.
                    let _ = unsafe { libc::kill(id as i32, libc::SIGINT) };
                }
                child.wait().await.map(|s| s.code().unwrap_or(130)) // 128 + SIGINT (2)
            }
            _ = sigterm.recv() => {
                // Forward SIGTERM to child
                if let Some(id) = child_id {
                    // SAFETY: `kill` does not dereference pointers; pid/signal values are plain integers.
                    let _ = unsafe { libc::kill(id as i32, libc::SIGTERM) };
                }
                child.wait().await.map(|s| s.code().unwrap_or(143)) // 128 + SIGTERM (15)
            }
        };

        result.map_err(|e| e.into())
    }

    /// Waits for a child process and returns its exit code (non-Unix)
    #[cfg(not(unix))]
    async fn wait_for_process(&self, child: &mut tokio::process::Child) -> Result<i32> {
        let status = child.wait().await?;
        Ok(status.code().unwrap_or(1))
    }
}

#[derive(Debug, Clone)]
struct ResolvedLaunchContext {
    key: ApiKey,
    model: Option<String>,
    /// Models discovered for the key on a CodexApp launch (an explicit `-m` is
    /// merged in by `catalog_slugs`). Feeds the codex model catalog file via
    /// `build_runtime_args`. `None` on preview and for OAuth/ollama/copilot keys.
    codex_app_models: Option<Vec<String>>,
    tool_config: ToolConfig,
}

/// The snapshot can't know `aivo/starter`, so a starter catalog cached without
/// its window must re-fetch from `/v1/models` or Pi drops to 128k. Scoped to
/// starter so id-only providers don't re-fetch every launch.
fn starter_catalog_window_cached(
    key: &ApiKey,
    metadata: &HashMap<String, crate::services::models_cache::ModelMetadata>,
) -> bool {
    !is_aivo_starter_base(&key.base_url)
        || metadata
            .get(crate::constants::AIVO_STARTER_MODEL)
            .and_then(|m| m.context_window)
            .is_some()
}

/// Drops slugs containing control bytes (NUL, newline, CR, tab) that a buggy
/// or hostile `/v1/models` endpoint might return. Those would break out of the
/// TOML basic-strings we emit via `-c model="..."` and the catalog JSON.
fn sanitize_discovered_slugs(models: Vec<String>) -> Vec<String> {
    models
        .into_iter()
        .filter(|m| !m.is_empty() && !m.chars().any(|c| c.is_control()))
        .collect()
}

/// Picks a sensible default model from the provider's `/v1/models` listing for
/// codex-app's GUI when the user didn't pass `-m`. Mirrors the OpenCode logic:
/// newest GPT-style chat model wins; then any OpenAI-shaped name; else the
/// first alphabetical. Returns `None` only for an empty list.
fn pick_default_codex_app_model(models: &[String]) -> Option<String> {
    let mut sorted = models.to_vec();
    sorted.sort();
    sorted.dedup();
    sorted
        .iter()
        .rev()
        .find(|m| is_gpt_chat_model_name(m))
        .or_else(|| sorted.iter().rev().find(|m| is_openai_style_model_name(m)))
        .or_else(|| sorted.first())
        .cloned()
}

/// Replaces the shadow `CODEX_HOME`'s `models_cache.json` with aivo's models.
/// NOT the picker source — that's the `-c model_catalog_json` override
/// (`inject_codex_model_catalog`); this cache is TTL-gated (~300s) and ignored
/// when the catalog is present. Kept only so the shadow cache doesn't symlink
/// back to the user's stock `~/.codex/models_cache.json`, which we unlink
/// before writing so the write can't follow into the real home dir.
async fn install_codex_app_models_cache(env: &HashMap<String, String>, catalog_path: Option<&str>) {
    let Some(codex_home) = env.get("CODEX_HOME") else {
        return;
    };
    let Some(catalog) = catalog_path else {
        return;
    };
    let catalog_text = match tokio::fs::read_to_string(catalog).await {
        Ok(t) => t,
        Err(e) => {
            warn_codex_app_cache(format!("read catalog {catalog}: {e}"));
            return;
        }
    };
    let parsed: serde_json::Value = match serde_json::from_str(&catalog_text) {
        Ok(v) => v,
        Err(e) => {
            warn_codex_app_cache(format!("parse catalog JSON: {e}"));
            return;
        }
    };
    let models = match parsed.get("models").and_then(|m| m.as_array()) {
        Some(m) if !m.is_empty() => m.clone(),
        _ => {
            warn_codex_app_cache("catalog has no `models`; picker will fall back");
            return;
        }
    };
    // The bundled codex tags its cache with its own version. A mismatch makes
    // codex treat the file as stale and refetch from chatgpt.com — which fails
    // for non-ChatGPT keys, leaving the picker empty. If we can't detect the
    // version (probe timed out, bundle broken), skip the cache write entirely:
    // `model_catalog_json` still serves the slugs via codex's CLI plumbing,
    // and we avoid stamping the cache with a wrong literal that codex would
    // reject anyway.
    let Some(client_version) = detect_bundled_codex_version().await else {
        warn_codex_app_cache(
            "could not detect bundled codex version; leaving shadow models_cache.json alone",
        );
        return;
    };
    let cache = serde_json::json!({
        "fetched_at": chrono::Utc::now().to_rfc3339(),
        "client_version": client_version,
        "etag": "aivo-injected",
        "models": models,
    });
    let body = match serde_json::to_vec(&cache) {
        Ok(b) => b,
        Err(e) => {
            warn_codex_app_cache(format!("serialize models cache: {e}"));
            return;
        }
    };
    let cache_path = std::path::PathBuf::from(codex_home).join("models_cache.json");
    // `link_session_state` may have symlinked this path to ~/.codex/models_cache.json.
    // Unlink before write so the upcoming write can't follow the symlink into
    // the user's real home dir. Tolerate NotFound (no prior file). Any other
    // error must abort: writing through a surviving symlink would corrupt the
    // user's real codex cache — exactly what this whole helper exists to avoid.
    match tokio::fs::remove_file(&cache_path).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            warn_codex_app_cache(format!(
                "could not unlink shadow {} before write: {} \u{2014} skipping to avoid writing through to ~/.codex/",
                cache_path.display(),
                e
            ));
            return;
        }
    }
    if let Err(e) = tokio::fs::write(&cache_path, body).await {
        warn_codex_app_cache(format!("write {}: {}", cache_path.display(), e));
    }
}

fn warn_codex_app_cache(msg: impl AsRef<str>) {
    eprintln!(
        "  {} codex-app picker: {}",
        crate::style::yellow("aivo:"),
        msg.as_ref()
    );
}

/// Probes `<Codex.app>/Contents/Resources/codex --version` and returns the
/// trimmed version string (e.g. `"0.133.0"`). Bounded by a hard timeout so a
/// stuck binary (Gatekeeper translocation, hung sig verification, damaged
/// bundle) can't wedge the whole `aivo codex-app` launch. On failure we return
/// `None` and the caller skips the cache install instead of stamping a stale
/// literal.
async fn detect_bundled_codex_version() -> Option<String> {
    let bin = crate::services::codex_app_wrapper::locate_bundled_codex()?;
    let mut cmd = Command::new(&bin);
    cmd.arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    let probe = cmd.output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(3), probe)
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    // Some CLIs (npm update-notifier, banners) print extra lines on
    // `--version`. Look specifically for "codex-cli X.Y.Z" — the exact format
    // the bundled codex emits — and fall back to any version-shaped token only
    // if that anchor is missing.
    if let Some(version) = text.lines().find_map(parse_codex_version_line) {
        return Some(version);
    }
    text.split_whitespace()
        .find(|t| version_token(t))
        .map(|s| s.to_string())
}

fn parse_codex_version_line(line: &str) -> Option<String> {
    let mut tokens = line.split_whitespace();
    let head = tokens.next()?;
    if !head.eq_ignore_ascii_case("codex-cli") && !head.eq_ignore_ascii_case("codex") {
        return None;
    }
    let v = tokens.next()?;
    if version_token(v) {
        Some(v.to_string())
    } else {
        None
    }
}

fn version_token(t: &str) -> bool {
    let mut chars = t.chars();
    chars.next().is_some_and(|c| c.is_ascii_digit()) && t.contains('.')
}

/// Polls until the Codex.app GUI process has exited. Snapshots the pid(s) of
/// the GUI's Mach-O main on first sighting (so an unrelated process whose
/// argv mentions the `Codex.app/Contents/MacOS/` path — debugger, editor
/// opening it, log tail — can't keep us pinned), then watches only those pids.
///
/// Ctrl-C / SIGTERM offers two-tap confirmation: the first signal prints a
/// hint, the second within 5s sends a graceful `osascript quit` (or SIGTERM
/// fallback) and waits for the GUI to exit. A single accidental Ctrl-C
/// reverts to silent waiting after the prompt timeout.
#[cfg(unix)]
async fn wait_for_codex_app_gui_exit() {
    use std::time::{Duration, Instant};
    use tokio::time::sleep;
    let mut sigint = match signal::unix::signal(signal::unix::SignalKind::interrupt()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut sigterm = match signal::unix::signal(signal::unix::SignalKind::terminate()) {
        Ok(s) => s,
        Err(_) => return,
    };
    let gui_appear_deadline = Instant::now() + Duration::from_secs(30);
    let mut tracked: Vec<i32> = Vec::new();
    loop {
        tokio::select! {
            _ = sigint.recv() => {
                if handle_signal_quit_prompt(&mut sigint, &mut sigterm, &mut tracked).await {
                    return;
                }
            }
            _ = sigterm.recv() => {
                if handle_signal_quit_prompt(&mut sigint, &mut sigterm, &mut tracked).await {
                    return;
                }
            }
            _ = sleep(Duration::from_millis(1500)) => {
                let pids = codex_app_gui_pids().await;
                if tracked.is_empty() {
                    if !pids.is_empty() {
                        tracked = pids;
                    } else if Instant::now() >= gui_appear_deadline {
                        // No GUI sighting within the appearance window — treat
                        // as never-launched and bail rather than blocking
                        // forever.
                        return;
                    }
                    continue;
                }
                // Keep only the pids we first saw that are still alive.
                tracked.retain(|pid| pid_alive(*pid));
                if tracked.is_empty() {
                    return;
                }
            }
        }
    }
}

/// First signal: print the two-tap hint. Second signal within 5s: send a
/// graceful quit to Codex.app and poll the tracked pids until they exit (or
/// up to 8s — Cocoa's "are you sure?" dialog can take a beat).
///
/// Returns `true` when the caller should exit the wait loop (signal-driven
/// quit completed, or user chose to abandon and the GUI is dying), `false` to
/// resume the normal poll loop (single accidental signal, no follow-up).
#[cfg(unix)]
async fn handle_signal_quit_prompt(
    sigint: &mut signal::unix::Signal,
    sigterm: &mut signal::unix::Signal,
    tracked: &mut Vec<i32>,
) -> bool {
    use std::time::Duration;
    use tokio::time::sleep;
    eprintln!(
        "  {} press Ctrl-C again within 5s to quit the Codex app (and aivo). Wait to keep both running.",
        crate::style::yellow("aivo:")
    );
    tokio::select! {
        _ = sigint.recv() => {}
        _ = sigterm.recv() => {}
        _ = sleep(Duration::from_secs(5)) => {
            eprintln!(
                "  {} keeping the app open; resuming normal wait.",
                crate::style::dim("aivo:")
            );
            return false;
        }
    }
    // Confirmed quit. Best-effort graceful shutdown of any tracked pids we
    // know about, plus a generic AppleScript quit that hits whichever Codex
    // instance LaunchServices currently knows about (some launches we tracked
    // a single MacOS/Codex pid, others use the lib's NSWorkspace bundle id).
    eprintln!(
        "  {} quitting the app gracefully...",
        crate::style::yellow("aivo:")
    );
    quit_codex_app(tracked).await;
    // Then poll briefly until the pids drop, so the local router survives
    // until Cocoa actually finishes its quit handlers.
    let quit_deadline = std::time::Instant::now() + Duration::from_secs(8);
    while std::time::Instant::now() < quit_deadline {
        tracked.retain(|pid| pid_alive(*pid));
        if tracked.is_empty() {
            break;
        }
        sleep(Duration::from_millis(400)).await;
    }
    true
}

#[cfg(target_os = "macos")]
const CODEX_APP_QUIT_HINT: &str = "Quit ChatGPT.app manually (Cmd-Q) and re-run `aivo codex-app`.";
#[cfg(not(target_os = "macos"))]
const CODEX_APP_QUIT_HINT: &str =
    "Quit the Codex desktop app manually and re-run `aivo codex-app`.";

/// Detects an existing Codex.app instance and asks the user to restart it.
/// Bails (or auto-aborts on a non-interactive stdin) if the user declines —
/// continuing would silently leave the running GUI on its old wrapper / key.
async fn preflight_codex_app_running() -> Result<()> {
    let pids = codex_app_gui_pids().await;
    if pids.is_empty() {
        return Ok(());
    }

    eprintln!(
        "  {} the Codex app (ChatGPT.app) is already running (pid{} {}). aivo can't route a new key into an existing instance \u{2014} `CODEX_CLI_PATH` is captured at launch.",
        crate::style::yellow("aivo:"),
        if pids.len() == 1 { "" } else { "s" },
        pids.iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );

    if !std::io::stdin().is_terminal() {
        return Err(CLIError::new(
            "The Codex app is already running and stdin is not a TTY \u{2014} cannot prompt to restart.",
            ErrorCategory::User,
            None::<String>,
            Some(CODEX_APP_QUIT_HINT),
        )
        .into());
    }

    eprint!(
        "  {} Quit it and relaunch via aivo? [Y/n] ",
        crate::style::yellow("?")
    );
    let _ = std::io::stderr().flush();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let trimmed = input.trim();
    let consent = trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("y")
        || trimmed.eq_ignore_ascii_case("yes");
    if !consent {
        return Err(CLIError::new(
            "Aborted by user; the Codex app left running with its existing routing.",
            ErrorCategory::User,
            None::<String>,
            None::<String>,
        )
        .into());
    }

    eprintln!(
        "  {} quitting the app gracefully so the new key takes effect...",
        crate::style::yellow("aivo:")
    );
    quit_codex_app(&pids).await;
    // Poll until the tracked pids are gone, with a hard cap so a hung GUI
    // doesn't trap the launch indefinitely.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        // Per-pid liveness, not a rescan — a full scan spawns powershell
        // every 300ms on Windows. Pid reuse within 10s is not a concern.
        if !pids.iter().any(|p| pid_alive(*p)) {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    }
    Err(CLIError::new(
        "The Codex app did not exit within 10s after the quit request.",
        ErrorCategory::User,
        None::<String>,
        Some(CODEX_APP_QUIT_HINT),
    )
    .into())
}

#[cfg(unix)]
async fn quit_codex_app(tracked: &[i32]) {
    // Prefer the AppleScript quit — it lets Cocoa run normal shutdown handlers
    // (including any "are you sure?" dialogs the app's UI might show).
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("osascript")
            // Address by bundle id: the app's display name changed from
            // "Codex" to "ChatGPT" in v26.707, but the id is stable.
            .args(["-e", "tell application id \"com.openai.codex\" to quit"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
    // Then signal-fallback to the tracked pids in case osascript didn't find
    // the app (or we're on a Unix that isn't macOS). SIGTERM lets the process
    // run its own shutdown handlers, unlike SIGKILL.
    for pid in tracked {
        // SAFETY: `kill` takes plain integers; nothing is dereferenced.
        unsafe { libc::kill(*pid, libc::SIGTERM) };
    }
}

/// `taskkill` without `/F` posts WM_CLOSE — graceful; shutdown handlers run.
#[cfg(not(unix))]
async fn quit_codex_app(tracked: &[i32]) {
    for pid in tracked {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
    }
}

#[cfg(not(unix))]
async fn wait_for_codex_app_gui_exit() {
    use std::time::{Duration, Instant};
    use tokio::time::sleep;
    let gui_appear_deadline = Instant::now() + Duration::from_secs(30);
    let mut tracked: Vec<i32> = Vec::new();
    loop {
        sleep(Duration::from_millis(1500)).await;
        if tracked.is_empty() {
            // Sighting needs the full (powershell-backed) process scan; once
            // pids are tracked, per-tick liveness is a native syscall.
            let pids = codex_app_gui_pids().await;
            if !pids.is_empty() {
                tracked = pids;
            } else if Instant::now() >= gui_appear_deadline {
                return;
            }
            continue;
        }
        tracked.retain(|pid| pid_alive(*pid));
        if tracked.is_empty() {
            return;
        }
    }
}

/// Pids of running Codex desktop app GUI processes. Matches only the
/// executable path, never argv: a `tail -f` or `lldb` invocation naming a
/// path inside the bundle would otherwise pin us.
async fn codex_app_gui_pids() -> Vec<i32> {
    #[cfg(unix)]
    {
        // `ps -axo pid,comm` prints `<pid> <executable path>`. `comm` on macOS
        // shows the actual executable backing the process, not argv[0], so we
        // can match Codex.app's Mach-O without false positives.
        let output = Command::new("ps")
            .args(["-axo", "pid=,comm="])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await;
        let Ok(out) = output else {
            return Vec::new();
        };
        let text = String::from_utf8_lossy(&out.stdout);
        let mut pids = Vec::new();
        for line in text.lines() {
            let trimmed = line.trim_start();
            let (pid_str, rest) = match trimmed.split_once(char::is_whitespace) {
                Some(parts) => parts,
                None => continue,
            };
            let exe = rest.trim();
            // Match the bundle dir, not the binary name (Codex→ChatGPT
            // rename; see APP_NAMES). `Contents/Frameworks/` helpers stay
            // out. `ChatGPT.app` alone is ambiguous with the legacy
            // chat-only app — require the codex bundle id; we SIGTERM
            // whatever matches.
            let is_codex_gui = exe.contains("/Codex.app/Contents/MacOS/")
                || (exe.contains("/ChatGPT.app/Contents/MacOS/") && bundle_is_codex(exe));
            if is_codex_gui && let Ok(pid) = pid_str.parse::<i32>() {
                pids.push(pid);
            }
        }
        pids
    }
    #[cfg(not(unix))]
    {
        // Image name alone is ambiguous (the legacy chat-only ChatGPT.exe
        // shares it) — require the exe path inside the OpenAI.Codex package.
        const PS: &str = r#"Get-CimInstance Win32_Process -Filter "Name='ChatGPT.exe' OR Name='Codex.exe'" | Where-Object { $_.ExecutablePath -like '*\WindowsApps\OpenAI.Codex_*' } | Select-Object -ExpandProperty ProcessId"#;
        let output = Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", PS])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await;
        let Ok(out) = output else {
            return Vec::new();
        };
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .filter_map(|line| line.trim().parse::<i32>().ok())
            .collect()
    }
}

/// True when the bundle containing `exe` is the Codex app (`com.openai.codex`)
/// vs the legacy chat-only app (`com.openai.chat`). Byte-scans Info.plist —
/// the ASCII id is stored verbatim in both XML and binary plists.
#[cfg(unix)]
fn bundle_is_codex(exe: &str) -> bool {
    let Some(idx) = exe.find("/Contents/MacOS/") else {
        return false;
    };
    let plist = format!("{}/Contents/Info.plist", &exe[..idx]);
    std::fs::read(plist)
        .map(|bytes| String::from_utf8_lossy(&bytes).contains("com.openai.codex"))
        .unwrap_or(false)
}

fn pid_alive(pid: i32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: kill(pid, 0) doesn't deliver a signal — it only probes that
        // the pid exists and we have permission to signal it. No memory is
        // dereferenced.
        let rc = unsafe { libc::kill(pid, 0) };
        if rc == 0 {
            return true;
        }
        // ESRCH means the pid is gone. EPERM means it exists but we can't
        // signal it — still counts as alive for our "is the GUI running"
        // purposes.
        std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
    #[cfg(windows)]
    {
        pid >= 0 && crate::services::system_env::is_pid_alive(pid as u32)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

/// Codex Store package identity via `Get-AppxPackage` — stable across Codex-
/// and ChatGPT-branded builds.
#[cfg(windows)]
#[derive(Debug, Clone, serde::Deserialize)]
struct WindowsCodexPackage {
    install: String,
    exe: String,
    family: String,
}

/// One PowerShell round trip: install location, the manifest's GUI exe
/// (package-relative), and the family name (app-execution-alias dir).
#[cfg(windows)]
async fn windows_codex_package() -> Option<WindowsCodexPackage> {
    const PROBE: &str = r#"$p = Get-AppxPackage -Name 'OpenAI.Codex*' | Sort-Object -Property Version -Descending | Select-Object -First 1; if ($p) { $a = (Get-AppxPackageManifest -Package $p.PackageFullName).Package.Applications.Application; if ($a -is [array]) { $a = $a[0] }; @{ install = $p.InstallLocation; exe = [string]$a.Executable; family = $p.PackageFamilyName } | ConvertTo-Json -Compress }"#;
    let probe = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", PROBE])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(15), probe)
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let pkg: WindowsCodexPackage = serde_json::from_str(text.trim()).ok()?;
    (!pkg.install.is_empty() && !pkg.exe.is_empty()).then_some(pkg)
}

/// GUI exe to spawn: prefer the app-execution alias (alias launches inherit
/// the caller's env), else the package exe (spawnable despite WindowsApps
/// ACLs). `symlink_metadata` because aliases are appexeclink reparse points
/// some metadata calls reject.
#[cfg(windows)]
fn windows_codex_gui_exe(pkg: &WindowsCodexPackage) -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    let exe_name = Path::new(&pkg.exe).file_name()?;
    if let Some(lad) = std::env::var_os("LOCALAPPDATA") {
        let alias = PathBuf::from(lad)
            .join("Microsoft")
            .join("WindowsApps")
            .join(&pkg.family)
            .join(exe_name);
        if alias.symlink_metadata().is_ok() {
            return Some(alias);
        }
    }
    let direct = Path::new(&pkg.install).join(&pkg.exe);
    direct.symlink_metadata().is_ok().then_some(direct)
}

/// Resolved once at the launch gate: the GUI launcher (macOS: bundle path
/// for `open -a`; Windows: the exe) and the bundled codex the wrapper
/// re-execs.
#[cfg_attr(not(any(target_os = "macos", windows)), allow(dead_code))]
struct CodexAppDesktop {
    gui: std::path::PathBuf,
    codex_bin: std::path::PathBuf,
}

fn codex_app_launch_invocation(
    desktop: &CodexAppDesktop,
    post_drain_args: &[String],
) -> (String, Vec<String>) {
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        let _ = (desktop, post_drain_args);
        unreachable!("codex-app desktop launch is gated to macOS/Windows");
    }
    #[cfg(any(target_os = "macos", windows))]
    {
        let (workspace, dropped) = codex_app_workspace_from_args(post_drain_args);
        if !dropped.is_empty() {
            eprintln!(
                "  {} codex-app: extra args not forwarded to the desktop launch: {}",
                crate::style::yellow("aivo:"),
                dropped.join(" ")
            );
        }
        let url = codex_app_new_thread_url(&workspace);
        #[cfg(target_os = "macos")]
        return (
            "/usr/bin/open".to_string(),
            vec![
                "-a".to_string(),
                desktop.gui.to_string_lossy().into_owned(),
                url,
            ],
        );
        #[cfg(windows)]
        return (desktop.gui.to_string_lossy().into_owned(), vec![url]);
    }
}

/// Splits post-drain argv (`["app", ...rest]`) into the workspace (first
/// non-flag arg after `app`, made absolute — the deep link is resolved by
/// the GUI process, not against our cwd) and args a direct launch can't
/// forward.
#[cfg(any(target_os = "macos", windows))]
fn codex_app_workspace_from_args(args: &[String]) -> (std::path::PathBuf, Vec<String>) {
    let mut workspace = None;
    let mut dropped = Vec::new();
    for arg in args.iter().skip_while(|a| *a != "app").skip(1) {
        if workspace.is_none() && !arg.starts_with('-') {
            workspace = Some(std::path::absolute(arg).unwrap_or_else(|_| arg.into()));
        } else {
            dropped.push(arg.clone());
        }
    }
    let workspace = workspace
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    (workspace, dropped)
}

/// New-thread deep link the desktop app opens on launch — same URL upstream's
/// `codex app` builds (codex_new_thread_url in codex-rs/cli/src/desktop_app/mac.rs).
#[cfg(any(target_os = "macos", windows))]
fn codex_app_new_thread_url(workspace: &std::path::Path) -> String {
    let mut url = url::Url::parse("codex://threads/new").expect("static deep-link base parses");
    url.query_pairs_mut()
        .append_pair("path", &workspace.to_string_lossy());
    url.to_string()
}

/// Moves codex's global `-c/--config/--enable/--disable` prefix into a wrapper
/// shell script and points `CODEX_CLI_PATH` at it. Codex.app spawns its codex
/// app-server subprocess via that env var (see `codex_app_wrapper` module
/// docs); the parent `codex app` call's own `--config` flags are NOT
/// inherited by that subprocess, so we move them where they'll actually be
/// read.
///
/// Installs the per-launch wrapper, stores its path in `CODEX_CLI_PATH`, and
/// returns it so the caller can clean up after the GUI exits.
async fn install_codex_app_wrapper(
    env: &mut HashMap<String, String>,
    args: &mut Vec<String>,
    config_dir: &std::path::Path,
    key_id: &str,
    codex_bin: &std::path::Path,
) -> Option<std::path::PathBuf> {
    use crate::services::codex_app_wrapper;
    let prefix = crate::services::launch_args::drain_codex_global_prefix(args);
    if prefix.is_empty() {
        return None;
    }
    let dir = codex_app_wrapper::wrapper_dir(config_dir);
    codex_app_wrapper::cleanup_stale_wrappers(&dir).await;
    match codex_app_wrapper::write_wrapper(&dir, key_id, codex_bin, &prefix).await {
        Ok(path) => {
            env.insert(
                "CODEX_CLI_PATH".to_string(),
                path.to_string_lossy().into_owned(),
            );
            Some(path)
        }
        Err(e) => {
            eprintln!(
                "  {} could not install codex-app wrapper: {} \u{2014} falling back to parent --config flags",
                crate::style::yellow("aivo:"),
                e
            );
            // Restore args so the parent codex CLI sees the flags. The flags
            // won't reach the GUI child, but the user gets a working CLI
            // launch rather than a silent misconfiguration.
            let mut restored = prefix;
            restored.append(args);
            *args = restored;
            None
        }
    }
}

/// Cached fresh PATH read from a login shell after a tool install. Set once
/// per process by `refresh_path_from_shell`. Read by `apply_refreshed_path` to
/// thread the value into spawned children without mutating global env state
/// (which would race with any future `tokio::spawn` near this code).
static FRESHENED_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Re-read PATH from a login shell and stash it in `FRESHENED_PATH`. Unix
/// installers often append to shell profiles (~/.zshrc, ~/.bashrc); this picks
/// up those changes without requiring a terminal restart. On Windows,
/// installers use npm global bin which is already on PATH, so this is a no-op.
///
/// Caller should subsequently use `freshened_path_for_lookup` to find newly
/// installed binaries and `apply_refreshed_path` when spawning children.
async fn refresh_path_from_shell() {
    #[cfg(not(unix))]
    return;

    #[cfg(unix)]
    if let Ok(output) = Command::new("sh")
        .arg("-lc")
        .arg("echo $PATH")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
    {
        let fresh = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !fresh.is_empty() {
            // OnceLock::set fails (silently) if already populated — earliest
            // refresh wins for a given process, which matches the intent
            // (installers run before launches).
            let _ = FRESHENED_PATH.set(fresh);
        }
    }
}

/// Returns the freshened PATH (if any) as an `OsString` suitable for
/// `collect_path_dirs_from`.
fn freshened_path_for_lookup() -> Option<std::ffi::OsString> {
    FRESHENED_PATH.get().map(std::ffi::OsString::from)
}

/// Merge any freshened install dirs into the child's PATH, else inherit as-is.
fn apply_refreshed_path(cmd: &mut Command) {
    if let Some(fresh) = FRESHENED_PATH.get() {
        let inherited = std::env::var_os("PATH").unwrap_or_default();
        cmd.env("PATH", merge_path_dirs(&inherited, fresh));
    }
}

/// Append `fresh` dirs missing from `inherited`, preserving `inherited`'s
/// order. Clobbering instead would let the login-`sh` PATH front a broken
/// Homebrew `node` over the working one the interactive shell resolves.
fn merge_path_dirs(inherited: &std::ffi::OsStr, fresh: &str) -> std::ffi::OsString {
    let mut dirs: Vec<std::path::PathBuf> = std::env::split_paths(inherited).collect();
    let mut seen: std::collections::HashSet<std::path::PathBuf> = dirs.iter().cloned().collect();
    for dir in std::env::split_paths(fresh) {
        if seen.insert(dir.clone()) {
            dirs.push(dir);
        }
    }
    std::env::join_paths(&dirs).unwrap_or_else(|_| inherited.to_os_string())
}

fn preferred_claude_protocol(base_url: &str) -> ClaudeProviderProtocol {
    let profile = provider_profile_for_base_url(base_url);
    match profile.upstream_protocol_for_cli(ProviderProtocol::Anthropic) {
        ProviderProtocol::Anthropic => ClaudeProviderProtocol::Anthropic,
        ProviderProtocol::Google => ClaudeProviderProtocol::Google,
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => ClaudeProviderProtocol::Openai,
    }
}

fn preferred_codex_mode(base_url: &str) -> OpenAICompatibilityMode {
    if is_direct_openai_base(base_url) {
        OpenAICompatibilityMode::Direct
    } else {
        OpenAICompatibilityMode::Router
    }
}

fn preferred_gemini_protocol(base_url: &str) -> GeminiProviderProtocol {
    let profile = provider_profile_for_base_url(base_url);
    match profile.upstream_protocol_for_cli(ProviderProtocol::Google) {
        ProviderProtocol::Google => GeminiProviderProtocol::Google,
        ProviderProtocol::Anthropic => GeminiProviderProtocol::Anthropic,
        ProviderProtocol::Openai | ProviderProtocol::ResponsesApi => GeminiProviderProtocol::Openai,
    }
}

fn preferred_opencode_mode(base_url: &str) -> OpenAICompatibilityMode {
    if is_direct_openai_base(base_url) {
        return OpenAICompatibilityMode::Direct;
    }
    if is_opencode_zen_base(base_url) {
        return OpenAICompatibilityMode::Router;
    }
    let profile = provider_profile_for_base_url(base_url);
    if profile.default_protocol == ProviderProtocol::Openai {
        // Direct connection works for plain OpenAI-compatible endpoints,
        // but use router if the provider has quirks that need request transformation.
        if profile.quirks.has_quirks() {
            OpenAICompatibilityMode::Router
        } else {
            OpenAICompatibilityMode::Direct
        }
    } else {
        OpenAICompatibilityMode::Router
    }
}

fn is_opencode_zen_base(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return base_url.to_ascii_lowercase().contains("opencode.ai/zen");
    };
    url.host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("opencode.ai"))
        && url.path().starts_with("/zen")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn codex_app_workspace_split_and_deep_link() {
        let args: Vec<String> = ["app", "/tmp/ws dir", "--force", "extra"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (ws, dropped) = codex_app_workspace_from_args(&args);
        assert_eq!(ws, std::path::PathBuf::from("/tmp/ws dir"));
        assert_eq!(dropped, vec!["--force".to_string(), "extra".to_string()]);
        // Space encoded as `+` — same form_urlencoded serialization upstream uses.
        assert_eq!(
            codex_app_new_thread_url(&ws),
            "codex://threads/new?path=%2Ftmp%2Fws+dir"
        );

        let (ws, dropped) = codex_app_workspace_from_args(&["app".to_string()]);
        assert_eq!(ws, std::env::current_dir().unwrap());
        assert!(dropped.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn bundle_is_codex_requires_codex_bundle_id() {
        let tmp = tempfile::tempdir().unwrap();
        let app = tmp.path().join("ChatGPT.app");
        std::fs::create_dir_all(app.join("Contents/MacOS")).unwrap();
        let exe = app.join("Contents/MacOS/ChatGPT");
        let exe = exe.to_string_lossy();
        let plist = app.join("Contents/Info.plist");

        std::fs::write(
            &plist,
            r#"<plist><dict><key>CFBundleIdentifier</key><string>com.openai.codex</string></dict></plist>"#,
        )
        .unwrap();
        assert!(bundle_is_codex(&exe));

        std::fs::write(
            &plist,
            r#"<plist><dict><key>CFBundleIdentifier</key><string>com.openai.chat</string></dict></plist>"#,
        )
        .unwrap();
        assert!(!bundle_is_codex(&exe));

        std::fs::remove_file(&plist).unwrap();
        assert!(!bundle_is_codex(&exe));
        assert!(!bundle_is_codex("/usr/bin/ChatGPT"));
    }

    #[test]
    fn test_ai_tool_type_from_str() {
        assert_eq!(AIToolType::parse("claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("Claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("CLAUDE"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("codex"), Some(AIToolType::Codex));
        assert_eq!(AIToolType::parse("codex-app"), Some(AIToolType::CodexApp));
        assert_eq!(AIToolType::parse("gemini"), Some(AIToolType::Gemini));
        assert_eq!(AIToolType::parse("opencode"), Some(AIToolType::Opencode));
        assert_eq!(AIToolType::parse("pi"), Some(AIToolType::Pi));
        assert_eq!(AIToolType::parse("unknown"), None);
    }

    #[test]
    fn well_known_install_dirs_includes_claude_native_installer_path() {
        // The Claude native installer drops the binary in `~/.local/bin`.
        // If that's not on PATH, the post-install fallback lookup must still
        // find it — this test pins that contract.
        let dirs = AIToolType::Claude.well_known_install_dirs();
        let home = crate::services::system_env::home_dir().expect("HOME set in test env");
        assert!(
            dirs.contains(&home.join(".local").join("bin")),
            "expected ~/.local/bin among Claude fallback dirs, got {dirs:?}"
        );
        assert!(
            dirs.contains(&home.join(".claude").join("local")),
            "expected ~/.claude/local among Claude fallback dirs, got {dirs:?}"
        );
    }

    #[test]
    fn well_known_install_dirs_includes_user_npm_global_for_npm_tools() {
        // Codex/Gemini/Pi install via `npm install -g`. Users who set their
        // npm prefix to `~/.npm-global` (a common rootless pattern) need that
        // dir as a fallback when the installer ran but PATH wasn't refreshed.
        let home = crate::services::system_env::home_dir().expect("HOME set in test env");
        for tool in [
            AIToolType::Codex,
            AIToolType::CodexApp,
            AIToolType::Gemini,
            AIToolType::Pi,
        ] {
            let dirs = tool.well_known_install_dirs();
            assert!(
                dirs.contains(&home.join(".npm-global").join("bin")),
                "{tool:?} fallback dirs missing ~/.npm-global/bin: {dirs:?}"
            );
        }
    }

    #[test]
    fn merge_path_dirs_keeps_inherited_order_and_appends_new() {
        let inherited = std::env::join_paths(["/a", "/b"]).unwrap();
        let fresh = std::env::join_paths(["/b", "/c", "/d"]).unwrap();
        let merged = merge_path_dirs(&inherited, fresh.to_str().unwrap());
        let expected = std::env::join_paths(["/a", "/b", "/c", "/d"]).unwrap();
        assert_eq!(merged, expected);
    }

    #[test]
    fn merge_path_dirs_never_lets_fresh_shadow_inherited_node() {
        // Inherited fronts a working node; fresh fronts brew's. Inherited wins.
        let inherited = std::env::join_paths(["/Users/me/.nvm/v20/bin", "/usr/local/bin"]).unwrap();
        let fresh = std::env::join_paths(["/usr/local/bin", "/usr/bin"]).unwrap();
        let merged = merge_path_dirs(&inherited, fresh.to_str().unwrap());
        let expected =
            std::env::join_paths(["/Users/me/.nvm/v20/bin", "/usr/local/bin", "/usr/bin"]).unwrap();
        assert_eq!(merged, expected);
    }

    #[test]
    fn test_preferred_claude_protocol_for_anthropic_urls() {
        assert_eq!(
            preferred_claude_protocol("https://api.anthropic.com/v1"),
            ClaudeProviderProtocol::Anthropic
        );
        assert_eq!(
            preferred_claude_protocol("https://api.minimax.io/anthropic/v1"),
            ClaudeProviderProtocol::Anthropic
        );
    }

    #[test]
    fn test_preferred_claude_protocol_for_openai_compatible_hosts_picks_anthropic() {
        // Claude Code emits /v1/messages; forward that as-is to any host whose
        // default protocol is OpenAI (known or inferred). Protocol fallback
        // downgrades on 404 and learns the pin for next launch. Saves the
        // multi-hop chain vs. a cold-start against gateway-like hosts.
        for url in [
            "https://api.openai.com/v1",
            "https://ai-gateway.vercel.sh/v1",
            "https://example.com/openai",
        ] {
            assert_eq!(
                preferred_claude_protocol(url),
                ClaudeProviderProtocol::Anthropic,
                "expected Anthropic upstream for {url}"
            );
        }
    }

    #[test]
    fn test_preferred_codex_mode() {
        assert_eq!(
            preferred_codex_mode("https://api.openai.com/v1"),
            OpenAICompatibilityMode::Direct
        );
        assert_eq!(
            preferred_codex_mode("https://openrouter.ai/api/v1"),
            OpenAICompatibilityMode::Router
        );
    }

    #[test]
    fn test_preferred_gemini_protocol_keeps_native_google_host() {
        assert_eq!(
            preferred_gemini_protocol("https://generativelanguage.googleapis.com/v1beta"),
            GeminiProviderProtocol::Google
        );
    }

    #[test]
    fn test_preferred_gemini_protocol_for_openai_compatible_hosts_picks_google() {
        // Gemini CLI's native protocol is Google; forward it as-is for any
        // host whose default protocol is OpenAI. Fallback handles the rest.
        for url in [
            "https://api.openai.com/v1",
            "https://ai-gateway.vercel.sh/v1",
        ] {
            assert_eq!(
                preferred_gemini_protocol(url),
                GeminiProviderProtocol::Google,
                "expected Google upstream for {url}"
            );
        }
    }

    #[test]
    fn preferred_protocols_always_start_with_cli_native_across_hosts() {
        for url in [
            "https://api.anthropic.com",
            "https://generativelanguage.googleapis.com/v1beta",
            "https://api.openai.com/v1",
            "https://api.deepseek.com",
            "aivo-starter",
        ] {
            assert_eq!(
                preferred_claude_protocol(url),
                ClaudeProviderProtocol::Anthropic,
                "claude should start with Anthropic for {url}",
            );
            assert_eq!(
                preferred_gemini_protocol(url),
                GeminiProviderProtocol::Google,
                "gemini should start with Google for {url}",
            );
        }
    }

    #[test]
    fn test_preferred_opencode_mode() {
        assert_eq!(
            preferred_opencode_mode("https://api.openai.com/v1"),
            OpenAICompatibilityMode::Direct
        );
        assert_eq!(
            preferred_opencode_mode("https://openrouter.ai/api/v1"),
            OpenAICompatibilityMode::Direct
        );
        assert_eq!(
            preferred_opencode_mode("https://opencode.ai/zen/v1"),
            OpenAICompatibilityMode::Router
        );
        assert_eq!(
            preferred_opencode_mode("https://opencode.ai/zen/go/v1"),
            OpenAICompatibilityMode::Router
        );
    }

    #[test]
    fn test_install_hint_non_empty_for_all_tools() {
        for tool in AIToolType::all() {
            let hint = tool.install_hint();
            assert!(!hint.is_empty(), "{:?} should have an install hint", tool);
        }
    }

    // ── resolve_*_protocol: in-memory only, disk-write is the caller's job ──

    async fn test_launcher_with_store() -> (SessionStore, AILauncher, tempfile::TempDir) {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.json");
        let store = SessionStore::with_path(config_path);
        let launcher = AILauncher::new(
            store.clone(),
            EnvironmentInjector::new(),
            ModelsCache::new(),
        );
        (store, launcher, temp_dir)
    }

    async fn test_insert_key(store: &SessionStore, base_url: &str) -> ApiKey {
        let id = store
            .add_key_with_protocol("test-key", base_url, None, "sk-test")
            .await
            .unwrap();
        store.get_key_by_id(&id).await.unwrap().unwrap()
    }

    #[tokio::test]
    async fn get_tool_config_sets_claude_max_output_tokens_from_limits() {
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://api.example.com/v1").await;
        let mut limits = HashMap::new();
        limits.insert(
            "deepseek-chat".to_string(),
            crate::services::model_metadata::ResolvedLimits {
                context: Some(128_000),
                output: Some(8_000),
                caps: None,
                reasoning_efforts: Vec::new(),
            },
        );
        let config = launcher.get_tool_config(
            AIToolType::Claude,
            &key,
            Some("deepseek-chat"),
            None,
            &[],
            &limits,
            &ClaudeModelOverrides::default(),
        );
        assert_eq!(
            config.env_vars.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
            Some(&"8000".to_string())
        );

        // Output unknown → the var stays unset so Claude's default applies.
        let config = launcher.get_tool_config(
            AIToolType::Claude,
            &key,
            Some("mystery-model"),
            None,
            &[],
            &limits,
            &ClaudeModelOverrides::default(),
        );
        assert!(
            !config
                .env_vars
                .contains_key("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
        );
    }

    #[tokio::test]
    async fn resolve_claude_protocol_populates_in_memory_only() {
        // resolve_* seeds the default route in-memory; disk stays clean until
        // the route cache's write-behind confirms a route.
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://api.example.com/v1").await;
        assert!(key.protocol_routes.is_empty(), "precondition: no routes");

        let resolved = launcher
            .resolve_claude_protocol(key.clone(), None)
            .await
            .unwrap();

        assert!(
            resolved
                .protocol_routes
                .get("claude")
                .is_some_and(|m| m.contains_key("")),
            "in-memory default route must be populated for this launch"
        );

        let reloaded = store.get_key_by_id(&key.id).await.unwrap().unwrap();
        assert!(
            reloaded.protocol_routes.is_empty(),
            "disk must stay clean — write-behind is the sole writer"
        );
    }

    #[tokio::test]
    async fn resolve_gemini_protocol_populates_in_memory_only() {
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://api.example.com/v1").await;
        assert!(key.protocol_routes.is_empty(), "precondition: no routes");

        let resolved = launcher.resolve_gemini_protocol(key.clone()).await.unwrap();

        assert!(
            resolved
                .protocol_routes
                .get("gemini")
                .is_some_and(|m| m.contains_key("")),
            "in-memory default route must be populated for this launch"
        );

        let reloaded = store.get_key_by_id(&key.id).await.unwrap().unwrap();
        assert!(
            reloaded.protocol_routes.is_empty(),
            "disk must stay clean — write-behind is the sole writer"
        );
    }

    #[tokio::test]
    async fn resolve_opencode_mode_rewrites_stale_opencode_zen_direct_pin() {
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://opencode.ai/zen/go/v1").await;
        store
            .set_key_opencode_mode(&key.id, Some(OpenAICompatibilityMode::Direct))
            .await
            .unwrap();
        let key = store.get_key_by_id(&key.id).await.unwrap().unwrap();

        let resolved = launcher
            .resolve_opencode_mode(key.clone(), true)
            .await
            .unwrap();

        assert_eq!(
            resolved.opencode_mode,
            Some(OpenAICompatibilityMode::Router)
        );
        let reloaded = store.get_key_by_id(&key.id).await.unwrap().unwrap();
        assert_eq!(
            reloaded.opencode_mode,
            Some(OpenAICompatibilityMode::Router)
        );
    }
}
