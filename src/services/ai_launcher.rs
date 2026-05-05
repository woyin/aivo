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
    build_preview_notes, build_runtime_args, inject_codex_provider_config, merge_preview_env,
    preview_args, rewrite_codex_preview_env,
};
use crate::services::launch_runtime::{
    cleanup_runtime_artifacts, finalize_codex_oauth, finalize_gemini_oauth,
    persist_runtime_discoveries, prepare_runtime_env, process_pi_sessions, record_launch_state,
};
use crate::services::log_store::{LogEvent, new_log_id};
use crate::services::model_names::{is_gpt_chat_model_name, is_openai_style_model_name};
use crate::services::models_cache::ModelsCache;
use crate::services::ollama;
use crate::services::path_search::{collect_path_dirs, collect_path_dirs_from, find_in_dirs};
use crate::services::provider_profile::{
    is_copilot_base, is_direct_openai_base, is_ollama_base, provider_profile_for_base_url,
};
use crate::services::provider_protocol::ProviderProtocol;
use crate::services::session_store::{
    ApiKey, ClaudeProviderProtocol, GeminiProviderProtocol, OpenAICompatibilityMode, SessionStore,
};

/// Supported AI tool types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AIToolType {
    Claude,
    Codex,
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
            Self::Gemini => "gemini",
            Self::Opencode => "opencode",
            Self::Pi => "pi",
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Claude,
            Self::Codex,
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
            || (*self == AIToolType::Codex && key.is_codex_oauth())
            || (*self == AIToolType::Gemini && key.is_gemini_oauth());
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
            Self::Codex => "npm install -g @openai/codex",
            Self::Gemini => "npm install -g @google/gemini-cli",
            Self::Opencode => "curl -fsSL https://opencode.ai/install | bash",
            Self::Pi => "npm install -g @mariozechner/pi-coding-agent",
        }
        #[cfg(not(unix))]
        match self {
            Self::Claude => "npm install -g @anthropic-ai/claude-code",
            Self::Codex => "npm install -g @openai/codex",
            Self::Gemini => "npm install -g @google/gemini-cli",
            Self::Opencode => "npm install -g opencode-ai",
            Self::Pi => "npm install -g @mariozechner/pi-coding-agent",
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
            for sub in [".local/bin", ".npm-global/bin", ".bun/bin"] {
                dirs.push(home.join(sub));
            }
            // Tool-specific installer paths.
            match self {
                Self::Claude => dirs.push(home.join(".claude/local")),
                Self::Opencode => dirs.push(home.join(".opencode/bin")),
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
    /// Claude-only per-slot model overrides for the six addressable slots.
    /// Ignored — with a single stderr warning per set slot — when `tool` is
    /// not `Claude`.
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

        let env = self
            .env_injector
            .merge(&resolved.tool_config.env_vars, options.env.as_ref());
        let mut runtime = prepare_runtime_env(options.tool, env, &self.session_store).await?;

        let mut runtime_args = build_runtime_args(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            &runtime.env,
        )
        .await?;

        if options.tool == AIToolType::Codex {
            inject_codex_provider_config(&mut runtime.env, &mut runtime_args.args);
        }

        let event_group_id = new_log_id();
        let cwd = crate::services::system_env::current_dir_string();
        let log_args = runtime_args.args.clone();

        // Check if the tool binary is available on PATH before attempting to spawn.
        // When found, pin `tool_config.command` to the full resolved path so the
        // spawn step picks up the correct extension on Windows — CreateProcessW
        // does not honor PATHEXT for non-.exe files, so a bare `claude` would
        // fail to spawn even when `claude.cmd` is on PATH.
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
            let resolved_path =
                find_in_dirs(&resolved.tool_config.command, &path_dirs).or_else(|| {
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

        let child_result = self.spawn_child(
            &resolved.tool_config.command,
            &runtime_args.args,
            runtime.env,
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

        if options.tool == AIToolType::Pi {
            process_pi_sessions(runtime.pi_agent_dir.as_deref()).await;
        }

        persist_runtime_discoveries(
            &self.session_store,
            options.tool,
            &resolved.key,
            runtime.router_protocol,
            runtime.responses_api_support,
            runtime.request_succeeded,
            runtime.saw_authoritative_response,
            runtime.learned_requires_reasoning,
        )
        .await;

        finalize_codex_oauth(&self.session_store, runtime.codex_oauth_sync).await;
        finalize_gemini_oauth(&self.session_store, runtime.gemini_oauth_sync).await;

        cleanup_runtime_artifacts(
            runtime_args.codex_model_catalog_path.as_deref(),
            runtime.pi_agent_dir.as_deref(),
        )
        .await;

        let exit_code = result.as_ref().ok().copied();
        let _ = self
            .session_store
            .logs()
            .append(LogEvent {
                phase: Some("finished".to_string()),
                exit_code: exit_code.map(i64::from),
                duration_ms: Some(started_at.elapsed().as_millis() as i64),
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
        let args = preview_args(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            &resolved.tool_config.env_vars,
        );
        let notes = build_preview_notes(
            options.tool,
            &options.args,
            resolved.model.as_deref(),
            &resolved.tool_config.env_vars,
        );

        if options.tool == AIToolType::Codex {
            rewrite_codex_preview_env(&mut env_vars);
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
        } else if options.tool == AIToolType::Codex {
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
        let tool_config = self.get_tool_config(
            options.tool,
            &key,
            model.as_deref(),
            opencode_models.as_deref(),
            &options.claude_overrides,
        );
        Ok(ResolvedLaunchContext {
            key,
            model,
            tool_config,
        })
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
        if key.claude_protocol.is_none() {
            // Default guess only; persist_runtime_discoveries writes the pin
            // to disk after the router confirms the working protocol.
            key.claude_protocol = Some(preferred_claude_protocol(&key.base_url));
        }
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
        // OAuth entries are pinned to the native Google endpoint (handled by
        // the shadow GEMINI_CLI_HOME in launch_runtime). No router protocol
        // applies.
        if key.is_gemini_oauth() {
            return Ok(key);
        }
        if key.gemini_protocol.is_none() {
            // Default guess only — see resolve_claude_protocol.
            key.gemini_protocol = Some(preferred_gemini_protocol(&key.base_url));
        }
        Ok(key)
    }

    async fn resolve_opencode_mode(&self, mut key: ApiKey, persist: bool) -> Result<ApiKey> {
        if is_copilot_base(&key.base_url) {
            return Ok(key);
        }
        if key.opencode_mode.is_none() {
            key.opencode_mode = Some(preferred_opencode_mode(&key.base_url));
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

        // Check cache first — skip the spinner if we get a hit
        let fetch_result = if let Some(cached) = self.cache.get(&key.base_url).await {
            Ok(cached)
        } else {
            // Cache miss: show spinner while fetching from network
            let (spinning, spinner_handle) =
                crate::style::start_spinner(Some(" Fetching models..."));

            // bypass_cache=true: we know it's a miss; fetch_models_cached will still write result to cache
            let result =
                crate::commands::models::fetch_models_cached(&client, key, &self.cache, true).await;

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

    /// Gets tool-specific configuration including command and environment variables
    fn get_tool_config(
        &self,
        tool: AIToolType,
        key: &ApiKey,
        model: Option<&str>,
        opencode_models: Option<&[String]>,
        claude_overrides: &ClaudeModelOverrides,
    ) -> ToolConfig {
        // claude_overrides is non-empty only on the Claude path; for other
        // tools the run command warns up-front and never populates it.
        let env_vars = match tool {
            AIToolType::Claude => {
                self.env_injector
                    .for_claude_with_overrides(key, model, claude_overrides)
            }
            AIToolType::Codex => self.env_injector.for_codex(key, model),
            AIToolType::Gemini => self.env_injector.for_gemini(key, model),
            AIToolType::Opencode => self.env_injector.for_opencode(key, model, opencode_models),
            AIToolType::Pi => self.env_injector.for_pi(key, model),
        };

        ToolConfig {
            command: tool.as_str().to_string(),
            env_vars,
        }
    }

    /// Spawns a child process with stdio inheritance and returns its exit code
    fn spawn_child(
        &self,
        command: &str,
        args: &[String],
        env: HashMap<String, String>,
    ) -> Result<tokio::process::Child> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(&env)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        // If a tool was just installed and pulled a new dir into PATH via
        // shell profile, propagate that PATH to the child. Done here rather
        // than via global set_var so we never race with concurrent tasks.
        apply_refreshed_path(&mut cmd);

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn {}", command))?;
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
    tool_config: ToolConfig,
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

/// If a freshened PATH was captured, override the child's PATH with it.
/// Otherwise the child inherits the parent's PATH unchanged.
fn apply_refreshed_path(cmd: &mut Command) {
    if let Some(fresh) = FRESHENED_PATH.get() {
        cmd.env("PATH", fresh);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ai_tool_type_from_str() {
        assert_eq!(AIToolType::parse("claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("Claude"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("CLAUDE"), Some(AIToolType::Claude));
        assert_eq!(AIToolType::parse("codex"), Some(AIToolType::Codex));
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
            dirs.contains(&home.join(".local/bin")),
            "expected ~/.local/bin among Claude fallback dirs, got {dirs:?}"
        );
        assert!(
            dirs.contains(&home.join(".claude/local")),
            "expected ~/.claude/local among Claude fallback dirs, got {dirs:?}"
        );
    }

    #[test]
    fn well_known_install_dirs_includes_user_npm_global_for_npm_tools() {
        // Codex/Gemini/Pi install via `npm install -g`. Users who set their
        // npm prefix to `~/.npm-global` (a common rootless pattern) need that
        // dir as a fallback when the installer ran but PATH wasn't refreshed.
        let home = crate::services::system_env::home_dir().expect("HOME set in test env");
        for tool in [AIToolType::Codex, AIToolType::Gemini, AIToolType::Pi] {
            let dirs = tool.well_known_install_dirs();
            assert!(
                dirs.contains(&home.join(".npm-global/bin")),
                "{tool:?} fallback dirs missing ~/.npm-global/bin: {dirs:?}"
            );
        }
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
    async fn resolve_claude_protocol_populates_in_memory_only() {
        // A generic OpenAI-compatible host with no pin. After resolve_* the
        // returned key must carry the guess in-memory, but the store on disk
        // must stay clean — persist_runtime_discoveries is the sole writer
        // once the router confirms a working protocol.
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://api.example.com/v1").await;
        assert!(key.claude_protocol.is_none(), "precondition: no pin");

        let resolved = launcher
            .resolve_claude_protocol(key.clone(), None)
            .await
            .unwrap();

        assert!(
            resolved.claude_protocol.is_some(),
            "in-memory pin must be populated for this launch"
        );

        let reloaded = store.get_key_by_id(&key.id).await.unwrap().unwrap();
        assert!(
            reloaded.claude_protocol.is_none(),
            "disk pin must stay None — persist_runtime_discoveries is the sole writer"
        );
    }

    #[tokio::test]
    async fn resolve_gemini_protocol_populates_in_memory_only() {
        let (store, launcher, _tmp) = test_launcher_with_store().await;
        let key = test_insert_key(&store, "https://api.example.com/v1").await;
        assert!(key.gemini_protocol.is_none(), "precondition: no pin");

        let resolved = launcher.resolve_gemini_protocol(key.clone()).await.unwrap();

        assert!(
            resolved.gemini_protocol.is_some(),
            "in-memory pin must be populated for this launch"
        );

        let reloaded = store.get_key_by_id(&key.id).await.unwrap().unwrap();
        assert!(
            reloaded.gemini_protocol.is_none(),
            "disk pin must stay None — persist_runtime_discoveries is the sole writer"
        );
    }
}
