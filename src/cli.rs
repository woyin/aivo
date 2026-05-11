/**
 * CLI argument parsing and command routing.
 * Uses clap for argument parsing.
 */
use clap::builder::NonEmptyStringValueParser;
use clap::{Args, Parser, Subcommand};
use std::collections::HashMap;

fn non_empty() -> NonEmptyStringValueParser {
    NonEmptyStringValueParser::new()
}

/// The aivo CLI - unified access to AI coding assistants
#[derive(Parser, Debug)]
#[command(
    name = "aivo",
    about = "CLI tool for unified access to AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi)",
    version = crate::version::VERSION,
    author = "yuanchuan",
    disable_help_flag = true,
    disable_version_flag = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Display help information
    #[arg(short, long, global = true, help = "Display help information")]
    pub help: bool,

    /// Display the current version
    #[arg(short, long, global = true, help = "Display the current version")]
    pub version: bool,
}

/// Available commands for the CLI
#[derive(Subcommand, Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    /// Run AI tools (claude, codex, gemini, opencode, pi, amp) - all args passed through
    Run(RunArgs),

    /// Manage API keys (use <id|name>, rm <id|name>, add, cat, edit)
    Keys(KeysArgs),

    /// List available models from the active provider
    Models(ModelsArgs),

    /// Start the interactive chat TUI
    Chat(ChatArgs),

    /// Serve an OpenAI-compatible API that proxies to the active provider
    Serve(ServeArgs),

    /// Create, list, or remove model aliases
    Alias(AliasArgs),

    /// Show system info, keys, tools, and directory state
    #[command(alias = "ls")]
    Info(InfoArgs),

    /// Show recent local logs from chat, run, and serve
    Logs(LogsArgs),

    /// Show usage statistics (tokens, requests, breakdowns)
    Stats(StatsArgs),

    /// Generate images from a text prompt (OpenAI-compatible providers)
    Image(ImageArgs),

    /// Generate videos from a text prompt (OpenAI-compatible providers, async)
    Video(VideoArgs),

    /// Speak a text prompt aloud (TTS). Saves to `~/.config/aivo/audio/`
    /// by default and reuses the cached file on repeat calls.
    Speak(AudioArgs),

    /// Update the CLI tool to the latest version
    Update(UpdateArgs),

    /// Amp-specific configuration (workspace MCP server trust, etc.)
    Amp(AmpArgs),

    /// Alias for `aivo logs share` — share a session via tunneled viewer URL.
    /// Both forms accept the same flags.
    Share(ShareArgs),
}

/// Arguments for `aivo logs share` (and the hidden top-level `aivo share` alias).
#[derive(Args, Debug, Clone)]
pub struct ShareArgs {
    /// Session id from `aivo logs` (claude / codex / gemini / pi / opencode / chat / amp).
    #[arg(value_name = "SESSION_ID")]
    pub session_id: Option<String>,

    /// Follow ongoing changes; default is a one-time snapshot taken at share time.
    #[arg(long)]
    pub live: bool,

    /// Skip redaction. Default scrubs API keys, OAuth tokens, $HOME paths,
    /// and secret-shaped env values.
    #[arg(long)]
    pub no_redact: bool,

    /// Show sessions from all projects, not just the current directory.
    #[arg(long)]
    pub all: bool,

    /// Open the share URL in the default browser once the link is ready.
    #[arg(long)]
    pub open: bool,

    /// Bind only on 127.0.0.1 — local debugging without the public tunnel.
    #[arg(long, hide = true)]
    pub debug_local_only: bool,
}

/// Arguments for `aivo amp`. Currently scoped to the `trust` subcommand
/// — the workspace MCP approval gate that mirrors `amp mcp approve` for
/// servers the bridge would otherwise auto-load from a repo's
/// `.amp/settings.json`.
#[derive(Args, Debug, Clone)]
pub struct AmpArgs {
    /// Subcommand: `trust` (currently the only option). Bare `aivo amp`
    /// prints help.
    #[arg(value_name = "ACTION")]
    pub action: Option<String>,

    /// Approve every pending workspace MCP server without prompting.
    /// Use only when you've already audited the file by hand.
    #[arg(long)]
    pub all: bool,

    /// List approved MCP servers for the current workspace and exit.
    #[arg(long)]
    pub list: bool,

    /// Revoke approval for a specific server name in the current
    /// workspace.
    #[arg(long, value_name = "NAME")]
    pub revoke: Option<String>,
}

/// Arguments for `aivo alias`
#[derive(Args, Debug, Clone)]
pub struct AliasArgs {
    /// Alias name, `name=model` shorthand, or the `rm` keyword.
    #[arg(value_name = "NAME[=MODEL]")]
    pub assignment: Option<String>,

    /// Trailing tokens. For Model alias positional form (`name model`): one
    /// token. For Bundle aliases (`name <tool> [args...]`): tool plus its
    /// preset args. For `rm <name>`: the name to remove. Empty for listing.
    #[arg(
        value_name = "ARGS",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub rest: Vec<String>,

    /// Remove an alias
    #[arg(long, short)]
    pub rm: bool,

    /// Output alias list as JSON (only affects listing)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the keys command
#[derive(Args, Debug, Clone)]
pub struct KeysArgs {
    /// The action to perform (use, rm, add, cat, edit)
    #[arg(
        value_name = "ACTION",
        help = "Action to perform: use, rm, add, cat, edit"
    )]
    pub action: Option<String>,

    /// Additional arguments for the action (e.g., key ID or name)
    #[arg(value_name = "ARGS", help = "Additional arguments for the action")]
    pub args: Vec<String>,

    /// API key display name for `keys add`
    #[arg(long, value_name = "NAME", value_parser = non_empty())]
    pub name: Option<String>,

    /// Provider base URL for `keys add`
    #[arg(long = "base-url", value_name = "URL", value_parser = non_empty())]
    pub base_url: Option<String>,

    /// Provider API key for `keys add`
    #[arg(long, value_name = "API_KEY", value_parser = non_empty())]
    pub key: Option<String>,

    /// Ping all keys (for `keys ping`)
    #[arg(long)]
    pub all: bool,

    /// List keys with ping status
    #[arg(long)]
    pub ping: bool,

    /// Output key list as JSON (listing only; secret is never included)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the run command
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// The AI tool to run (claude, codex, gemini, opencode, pi, amp)
    #[arg(
        value_name = "TOOL",
        help = "AI tool to run: claude, codex, gemini, opencode, pi, or amp"
    )]
    pub tool: Option<String>,

    /// Specify AI model to use
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Claude only: override the reasoning/thinking model slot
    /// (ANTHROPIC_REASONING_MODEL). Bare flag opens a picker.
    #[arg(long = "reasoning-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub reasoning_model: Option<String>,

    /// Claude only: override the subagent model slot
    /// (CLAUDE_CODE_SUBAGENT_MODEL). Bare flag opens a picker.
    #[arg(long = "subagent-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub subagent_model: Option<String>,

    /// Claude only: override the Haiku family-default slot
    /// (ANTHROPIC_DEFAULT_HAIKU_MODEL) — what Claude's `/model haiku` resolves to.
    /// Bare flag opens a picker.
    #[arg(long = "haiku-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub haiku_model: Option<String>,

    /// Claude only: override the Sonnet family-default slot
    /// (ANTHROPIC_DEFAULT_SONNET_MODEL) — what Claude's `/model sonnet` resolves to.
    /// Bare flag opens a picker.
    #[arg(long = "sonnet-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub sonnet_model: Option<String>,

    /// Claude only: override the Opus family-default slot
    /// (ANTHROPIC_DEFAULT_OPUS_MODEL) — what Claude's `/model opus` resolves to.
    /// Bare flag opens a picker.
    #[arg(long = "opus-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub opus_model: Option<String>,

    /// Amp only: model used by the `rush` agent mode (fast/cheap tier).
    /// Sets `amp.internal.model.rush` in the bridge's settings override.
    /// Bare flag opens a picker.
    #[arg(long = "rush-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub rush_model: Option<String>,

    /// Amp only: model used by the `smart` agent mode (default tier).
    /// Sets `amp.internal.model.smart` in the bridge's settings override.
    /// Bare flag opens a picker.
    #[arg(long = "smart-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub smart_model: Option<String>,

    /// Amp only: model used by the `deep` agent mode (reasoning tier).
    /// Sets `amp.internal.model.deep` in the bridge's settings override.
    /// Bare flag opens a picker.
    #[arg(long = "deep-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub deep_model: Option<String>,

    /// Amp only: model used by the `large` agent mode (long-context tier).
    /// Sets `amp.internal.model.large` in the bridge's settings override.
    /// Bare flag opens a picker.
    #[arg(long = "large-model", value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub large_model: Option<String>,

    /// Amp only: pin the initial agent mode for this thread (rush, smart,
    /// deep, large). Amp locks the mode after the first message lands in
    /// the thread, so this matters for the first-message UX. Bare flag
    /// opens an interactive picker.
    #[arg(long = "mode", value_name = "MODE", num_args = 0..=1, default_missing_value = "")]
    pub mode: Option<String>,

    /// Amp only: strip a tool from amp's request to the upstream
    /// (repeatable). Unknown names are silently ignored by amp.
    #[arg(long = "disable-tool", value_name = "NAME")]
    pub disable_tool: Vec<String>,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Bypass cache and fetch fresh model list for the model picker
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Log all aivo HTTP requests/responses to a JSONL file (default:
    /// ~/.config/aivo/logs/debug-<ts>-<pid>.jsonl). Sensitive headers and
    /// URL query params are redacted.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub debug: Option<String>,

    /// Print the resolved command and environment without launching
    #[arg(long)]
    pub dry_run: bool,

    /// Force a fresh OAuth login for the selected key before launching.
    /// Only applies to OAuth keys (codex / gemini / claude); errors out
    /// on plain API keys. Useful when the stored credential has been
    /// revoked server-side or otherwise can't be refreshed.
    #[arg(long)]
    pub relogin: bool,

    /// Inject cross-CLI context for this launch. Bare flag opens an
    /// interactive picker; `--context=<session-id>` picks a specific session
    /// (prefix match; see `aivo logs --by native` for available ids).
    #[arg(short = 'c', long, value_name = "SESSION_ID", num_args = 0..=1, default_missing_value = "")]
    pub context: Option<String>,

    /// Inject environment variable (KEY=VALUE)
    #[arg(short, long = "env", value_name = "KEY=VALUE")]
    pub envs: Vec<String>,

    /// Opt into a larger context window for the underlying tool. Accepts any
    /// `<N>m` (e.g. `1m`, `2m`, `12m`); aivo only validates shape.
    ///
    /// - For `claude`: appends a `[<size>]` suffix to the model name in every
    ///   default slot env var (`ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`,
    ///   etc.) so Claude Code opts into the matching beta context tier. Per-slot
    ///   overrides (`--haiku-model`, `--sonnet-model`, …) are left verbatim.
    /// - For `codex`: passes `--config model_context_window=<N×1_000_000>` to
    ///   codex, which clamps the value against the model's advertised ceiling.
    #[arg(long = "max-context", value_name = "SIZE")]
    pub max_context: Option<String>,

    /// Shorthand for `--max-context=1m`.
    #[arg(long = "1m")]
    pub one_m: bool,

    /// Shorthand for `--max-context=2m`.
    #[arg(long = "2m")]
    pub two_m: bool,

    /// Additional arguments to pass through to the AI tool
    #[arg(
        value_name = "ARGS",
        help = "Arguments to pass through to the AI tool",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    pub args: Vec<String>,
}

/// Arguments for the models command
#[derive(Args, Debug, Clone)]
pub struct ModelsArgs {
    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Bypass cache and fetch fresh model list from the provider
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Search models by substring
    #[arg(short = 's', long, value_name = "QUERY", value_parser = non_empty())]
    pub search: Option<String>,

    /// Output { provider, is_static, models[] } as JSON (pipe to `jq` to filter)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the serve command
#[derive(Args, Debug, Clone)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short = 'p', long, default_value_t = 24860)]
    pub port: u16,

    /// Host address to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Enable request logging (stdout, or to a file if path given)
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub log: Option<String>,

    /// Enable multi-key failover on 429/5xx errors
    #[arg(long)]
    pub failover: bool,

    /// Enable CORS headers for browser-based clients
    #[arg(long)]
    pub cors: bool,

    /// Upstream request timeout in seconds (0 = no timeout)
    #[arg(long, default_value_t = 300)]
    pub timeout: u64,

    /// Require bearer token (auto-generated if no value given)
    #[arg(long, value_name = "TOKEN", num_args = 0..=1, default_missing_value = "")]
    pub auth_token: Option<String>,
}

/// Arguments for the stats command
#[derive(Args, Debug, Clone)]
pub struct StatsArgs {
    /// Show stats for a specific tool (claude, codex, gemini, opencode, pi, amp, chat)
    #[arg(value_name = "TOOL")]
    pub tool: Option<String>,

    /// Exact numbers instead of human-readable
    #[arg(short = 'n', long)]
    pub numbers: bool,

    /// Search by key, model, or tool name (substring match)
    #[arg(short = 's', long, value_name = "QUERY")]
    pub search: Option<String>,

    /// Bypass cache and re-read all data files
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Show all models (default: top 20, rest grouped as "others")
    #[arg(short = 'a', long)]
    pub all: bool,

    /// Expand the per-model table to input/output/cached/total columns
    #[arg(short = 'd', long)]
    pub detailed: bool,

    /// Show the heaviest native session files for supported tools
    #[arg(long)]
    pub top_sessions: bool,

    /// Output stats as JSON (always uses exact numbers; includes all models)
    #[arg(long)]
    pub json: bool,

    /// Filter to the last N units (e.g. 7d, 24h, 30m, 2w)
    #[arg(long, value_name = "DURATION")]
    pub since: Option<String>,
}

/// Arguments for the update command
#[derive(Args, Debug, Clone)]
pub struct UpdateArgs {
    /// Force update even if installed via a package manager
    #[arg(short, long)]
    pub force: bool,
    /// Restore the previous version from the last update backup
    #[arg(long)]
    pub rollback: bool,
}

/// Arguments for the info command
#[derive(Args, Debug, Clone)]
pub struct InfoArgs {
    /// Ping all keys and show pass/fail summary
    #[arg(long)]
    pub ping: bool,

    /// Output info as JSON (useful for scripting)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the logs command
#[derive(Args, Debug, Clone)]
pub struct LogsArgs {
    /// Action: show, path, or status
    #[arg(value_name = "ACTION")]
    pub action: Option<String>,

    /// Target ID for `aivo logs show <id>`
    #[arg(value_name = "TARGET")]
    pub target: Option<String>,

    /// Maximum number of rows to display
    #[arg(short = 'n', long, default_value_t = 20)]
    pub limit: usize,

    /// Output JSON
    #[arg(long)]
    pub json: bool,

    /// Continuously refresh matching logs
    #[arg(long)]
    pub watch: bool,

    /// Emit newly seen entries as JSONL while watching
    #[arg(long)]
    pub jsonl: bool,

    /// Search title/body text
    #[arg(short = 's', long, value_name = "QUERY", value_parser = non_empty())]
    pub search: Option<String>,

    /// Filter by activity: aivo subcommand (chat, run, serve) or launched tool (claude, codex, ...)
    #[arg(long, value_name = "NAME", value_parser = non_empty())]
    pub by: Option<String>,

    /// Filter by model substring
    #[arg(long, value_name = "MODEL", value_parser = non_empty())]
    pub model: Option<String>,

    /// Filter by saved key ID or name substring
    #[arg(short = 'k', long, value_name = "ID|NAME", value_parser = non_empty())]
    pub key: Option<String>,

    /// Filter by working directory substring. Defaults to the current cwd
    /// when neither `--cwd` nor `--all` is given.
    #[arg(long, value_name = "PATH", value_parser = non_empty())]
    pub cwd: Option<String>,

    /// Show rows from every project, ignoring the implicit current-cwd
    /// filter. Mutually exclusive with `--cwd`.
    #[arg(short = 'a', long, conflicts_with = "cwd")]
    pub all: bool,

    /// Filter events since this ISO-like timestamp/date
    #[arg(long, value_name = "TIME", value_parser = non_empty())]
    pub since: Option<String>,

    /// Filter events until this ISO-like timestamp/date
    #[arg(long, value_name = "TIME", value_parser = non_empty())]
    pub until: Option<String>,

    /// Only show errors (HTTP >= 400 or exit_code != 0)
    #[arg(long)]
    pub errors: bool,

    // `logs share` only — guarded by validate_args() for non-share actions.
    /// `logs share`: follow ongoing changes (default: one-time snapshot).
    #[arg(long)]
    pub live: bool,

    /// `logs share`: skip redaction (default: scrub API keys, OAuth, $HOME, secret env).
    #[arg(long)]
    pub no_redact: bool,

    /// `logs share`: open the share URL in the default browser once ready.
    #[arg(long)]
    pub open: bool,

    /// `logs share`: bind only on 127.0.0.1 — local debugging without the public tunnel.
    #[arg(long, hide = true)]
    pub debug_local_only: bool,
}

/// Arguments for the chat command
#[derive(Args, Debug, Clone)]
pub struct ChatArgs {
    /// Specify AI model to use (remembered across sessions)
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Bypass cache and fetch fresh model list for the model picker
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Send one message and exit; reads stdin when no value given
    #[arg(
        short = 'x',
        long = "execute",
        value_name = "MESSAGE",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub execute: Option<String>,

    /// Print the upstream provider's raw JSON response (requires -x; useful for scripting)
    #[arg(long, requires = "execute")]
    pub json: bool,

    /// Attach a file or image to the next chat message (repeatable)
    #[arg(long = "attach", value_name = "PATH", value_parser = non_empty())]
    pub attachments: Vec<String>,

    /// Log all aivo HTTP requests/responses to a JSONL file (default:
    /// ~/.config/aivo/logs/debug-<ts>-<pid>.jsonl). Sensitive headers and
    /// URL query params are redacted.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub debug: Option<String>,
}

/// Arguments for the image command
#[derive(Args, Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageArgs {
    /// Text prompt for the image. When omitted, `aivo image` prints help
    /// and the active key/model instead of generating anything.
    #[arg(value_name = "PROMPT", value_parser = non_empty())]
    pub prompt: Option<String>,

    /// Image model to use (e.g. gpt-image-1, dall-e-3, grok-2-image)
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Output path: file (`cat.png`), directory (`out/`), or template with
    /// `{n}`/`{ts}`/`{model}` tokens. Default: `./aivo-<timestamp>.png`.
    #[arg(short = 'o', long, value_name = "PATH", value_parser = non_empty())]
    pub output: Option<String>,

    /// Overwrite existing files without prompting
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Image size, e.g. 1024x1024, 1792x1024, 1024x1792
    #[arg(short = 's', long, value_name = "WxH", value_parser = non_empty())]
    pub size: Option<String>,

    /// Quality: standard | hd | high | low (provider-dependent)
    #[arg(short = 'q', long, value_name = "LEVEL", value_parser = non_empty())]
    pub quality: Option<String>,

    /// Bypass cache and fetch fresh model list for the model picker
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Skip download; print the provider URL only (URLs may expire)
    #[arg(long)]
    pub url: bool,

    /// Emit a JSON object with the result (path, bytes, url, model, size)
    #[arg(long)]
    pub json: bool,

    /// Disable inline image preview in supported terminals (Kitty,
    /// Ghostty, WezTerm, Warp, recent iTerm2). Override per-invocation;
    /// `AIVO_PREVIEW=0` disables globally.
    #[arg(long)]
    pub no_preview: bool,
}

/// Arguments for the video command. Video generation is async on every
/// supported provider (Sora 2, Veo 3): we submit a job, poll until done,
/// then download. Use `--job-id` to recover after a Ctrl+C.
#[derive(Args, Debug, Clone, Default, PartialEq, Eq)]
pub struct VideoArgs {
    /// Text prompt. When omitted (and no `--job-id`), `aivo video` prints
    /// help and the active key/model instead of generating anything.
    #[arg(value_name = "PROMPT", value_parser = non_empty())]
    pub prompt: Option<String>,

    /// Video model (e.g. sora-2, veo-3.0-generate-preview)
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Output path: file (`clip.mp4`), directory (`out/`), or template
    /// with `{ts}`/`{model}` tokens. Default: `./aivo-<timestamp>.mp4`.
    #[arg(short = 'o', long, value_name = "PATH", value_parser = non_empty())]
    pub output: Option<String>,

    /// Overwrite existing files without prompting
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Frame size as WxH (e.g. 1280x720, 720x1280) or aspect ratio (16:9)
    #[arg(short = 's', long, value_name = "WxH", value_parser = non_empty())]
    pub size: Option<String>,

    /// Clip length in seconds (provider-dependent bounds, typically 4–20)
    #[arg(long, value_name = "SECS")]
    pub seconds: Option<u32>,

    /// Random seed for reproducibility (provider-dependent)
    #[arg(long, value_name = "SEED")]
    pub seed: Option<u64>,

    /// How long to wait before giving up on the polling loop (default 600s).
    /// On timeout, the job keeps running on the provider — recover with
    /// `--job-id <id>`.
    #[arg(long, value_name = "SECS", default_value_t = 600)]
    pub timeout: u32,

    /// Attach to an existing job and wait for its result instead of
    /// submitting a new one. Use after a previous run was Ctrl+C'd or
    /// timed out — the job ID is printed at submit time.
    #[arg(long, value_name = "ID", value_parser = non_empty())]
    pub job_id: Option<String>,

    /// Bypass cache and fetch fresh model list for the model picker
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Emit a JSON object with the result (path, bytes, model, job_id)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for the audio (TTS) command.
///
/// `Eq` is intentionally not derived because of `Option<f32>` for `speed`;
/// `PartialEq` is sufficient for clap and tests.
#[derive(Args, Debug, Clone, Default, PartialEq)]
pub struct AudioArgs {
    /// Text prompt to read aloud. When omitted, falls back to `--file` or
    /// piped stdin; with none of those, prints help and the active
    /// key/model instead of generating anything.
    #[arg(value_name = "PROMPT", value_parser = non_empty())]
    pub prompt: Option<String>,

    /// Read the prompt text from a file (UTF-8). Omit PATH or pass `-` to
    /// read stdin explicitly. Mutually exclusive with the positional
    /// `<PROMPT>`.
    #[arg(
        short = 'f',
        long = "file",
        value_name = "PATH",
        num_args = 0..=1,
        default_missing_value = "-",
        value_parser = non_empty(),
        conflicts_with = "prompt"
    )]
    pub file: Option<String>,

    /// Audio model (e.g. tts-1, tts-1-hd, gpt-4o-mini-tts)
    #[arg(short, long, value_name = "MODEL", num_args = 0..=1, default_missing_value = "")]
    pub model: Option<String>,

    /// Select API key by ID or name
    #[arg(
        short = 'k',
        long,
        value_name = "ID|NAME",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub key: Option<String>,

    /// Output path: file (`hello.mp3`), directory (`out/`), or template
    /// with `{ts}`/`{model}` tokens. When omitted, the audio is saved
    /// (and cached) under `~/.config/aivo/audio/`.
    #[arg(short = 'o', long, value_name = "PATH", value_parser = non_empty())]
    pub output: Option<String>,

    /// Overwrite any existing artifact: bypass the cache (regenerate)
    /// and overwrite a colliding -o path without prompting.
    #[arg(long)]
    pub overwrite: bool,

    /// Voice (provider-specific: alloy/echo/fable/onyx/nova/shimmer for
    /// OpenAI; Aoede/Charon/Kore/etc. for Gemini TTS)
    #[arg(long, value_name = "VOICE", value_parser = non_empty())]
    pub voice: Option<String>,

    /// Audio format. When unset, the interactive default is `wav` (so we
    /// can stream PCM and start playback before generation finishes); for
    /// `--no-play` or `-o` the buffered fallback defaults to `mp3`.
    /// Provider support varies — OpenAI accepts mp3/wav/opus/aac/flac/pcm;
    /// Gemini's TTS surface is fixed-format and ignores this flag.
    #[arg(
        long,
        value_name = "FORMAT",
        value_parser = ["mp3", "wav", "opus", "aac", "flac", "pcm"],
    )]
    pub format: Option<String>,

    /// Playback speed, typically 0.25–4.0 (provider-dependent)
    #[arg(long, value_name = "SPEED")]
    pub speed: Option<f32>,

    /// Bypass cache and fetch fresh model list for the model picker
    #[arg(short = 'r', long)]
    pub refresh: bool,

    /// Suppress playback. Useful for scripting where `speak` should just
    /// save a file.
    #[arg(long)]
    pub no_play: bool,

    /// List cached TTS entries with a fuzzy picker. Each entry can be
    /// replayed or deleted. Mutually exclusive with a prompt or `--file`.
    #[arg(long = "list", conflicts_with_all = ["prompt", "file"])]
    pub list: bool,

    /// Emit a JSON object with the result (path, bytes, model, voice)
    #[arg(long)]
    pub json: bool,
}

/// Parse environment variable strings in the format KEY=VALUE
pub fn parse_env_vars(env_strings: &[String]) -> HashMap<String, String> {
    let mut env_map = HashMap::new();

    for env_str in env_strings {
        if let Some((key, value)) = env_str.split_once('=') {
            env_map.insert(key.to_string(), value.to_string());
        }
    }

    env_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_env_vars() {
        let env_strings = vec![
            "KEY1=value1".to_string(),
            "KEY2=value2".to_string(),
            "KEY3=nested=value".to_string(),
        ];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.get("KEY1"), Some(&"value1".to_string()));
        assert_eq!(env_map.get("KEY2"), Some(&"value2".to_string()));
        assert_eq!(env_map.get("KEY3"), Some(&"nested=value".to_string()));
    }

    #[test]
    fn test_parse_env_vars_invalid() {
        let env_strings = vec!["NO_EQUALS".to_string(), "VALID=key".to_string()];

        let env_map = parse_env_vars(&env_strings);
        assert_eq!(env_map.len(), 1);
        assert_eq!(env_map.get("VALID"), Some(&"key".to_string()));
    }

    #[test]
    fn test_run_args_debug_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--debug"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            assert_eq!(run_args.debug, Some(String::new()));
            assert!(!run_args.dry_run);
            assert!(!run_args.args.contains(&"--debug".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_without_tool_parses_for_start_fallback() {
        let cli = Cli::try_parse_from(["aivo", "run", "--model", "gpt-5", "--debug"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, None);
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
            assert_eq!(run_args.debug, Some(String::new()));
            assert!(!run_args.dry_run);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_dry_run_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--dry-run"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert!(run_args.dry_run);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_flag() {
        // --model value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // --model=value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model=gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // -m value
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-m", "gpt-5"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
        } else {
            panic!("Expected Run command");
        }

        // --model with no value → triggers picker (Some(""))
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--model"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_env_flag() {
        let cli =
            Cli::try_parse_from(["aivo", "run", "claude", "--env", "FOO=bar", "-e", "BAZ=qux"])
                .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.envs, vec!["FOO=bar", "BAZ=qux"]);
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough() {
        // Arguments not matching aivo flags should be passed through
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--debug",
            "--",
            "--some-tool-flag",
            "value",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.debug, Some(String::new()));
            assert!(run_args.args.contains(&"--some-tool-flag".to_string()));
            assert!(run_args.args.contains(&"value".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_passthrough_claude_teammate_flags() {
        // Real-world usage: claude flags mixed with aivo --model flag.
        // When unknown flags appear before --model, clap's trailing_var_arg swallows
        // --model into args. main.rs re-extracts aivo flags from args at runtime.
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--agent-name",
            "senior-engineer",
            "--team-name",
            "ai-gateway-team",
            "--agent-color",
            "blue",
            "--parent-session-id",
            "df205d21-e955-421c-b2b9-5ff42c900cb6",
            "--agent-type",
            "general-purpose",
            "--dangerously-skip-permissions",
            "--model",
            "claude-opus-4-6",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
            // --model gets swallowed into args by trailing_var_arg; main.rs re-extracts it
            assert!(run_args.args.contains(&"--model".to_string()));
            assert!(run_args.args.contains(&"claude-opus-4-6".to_string()));
            // All unknown flags pass through
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(run_args.args.contains(&"senior-engineer".to_string()));
            assert!(run_args.args.contains(&"--team-name".to_string()));
            assert!(run_args.args.contains(&"ai-gateway-team".to_string()));
            assert!(
                run_args
                    .args
                    .contains(&"--dangerously-skip-permissions".to_string())
            );
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_model_before_unknown_flags() {
        // When --model comes before unknown flags, clap parses it directly
        let cli = Cli::try_parse_from([
            "aivo",
            "run",
            "claude",
            "--model",
            "claude-opus-4-6",
            "--agent-name",
            "senior-engineer",
            "--dangerously-skip-permissions",
        ])
        .unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.model, Some("claude-opus-4-6".to_string()));
            assert!(run_args.args.contains(&"--agent-name".to_string()));
            assert!(
                run_args
                    .args
                    .contains(&"--dangerously-skip-permissions".to_string())
            );
        } else {
            panic!("Expected Run command");
        }
    }

    /// Helper to simulate the alias rewriting done in main.rs
    fn rewrite_alias(args: &[&str]) -> Vec<String> {
        let aliases = ["claude", "codex", "gemini", "opencode", "pi", "amp"];
        let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        if raw.len() > 1 && aliases.contains(&raw[1].as_str()) {
            let mut rewritten = vec![raw[0].clone(), "run".to_string()];
            rewritten.extend_from_slice(&raw[1..]);
            rewritten
        } else {
            raw
        }
    }

    #[test]
    fn test_tool_alias_claude() {
        let args = rewrite_alias(&["aivo", "claude"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("claude".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_codex_with_args() {
        let args = rewrite_alias(&["aivo", "codex", "--model", "o4-mini", "file.ts"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("codex".to_string()));
            assert_eq!(run_args.model, Some("o4-mini".to_string()));
            assert!(run_args.args.contains(&"file.ts".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_gemini_with_debug() {
        let args = rewrite_alias(&["aivo", "gemini", "--debug"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("gemini".to_string()));
            assert_eq!(run_args.debug, Some(String::new()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_tool_alias_opencode() {
        let args = rewrite_alias(&["aivo", "opencode"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("opencode".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_non_alias_not_rewritten() {
        let args = rewrite_alias(&["aivo", "keys"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        assert!(matches!(cli.command, Some(Commands::Keys(_))));
    }

    #[test]
    fn test_models_search_option() {
        let cli = Cli::try_parse_from(["aivo", "models", "-s", "sonnet"]).unwrap();
        if let Some(Commands::Models(models_args)) = cli.command {
            assert_eq!(models_args.search.as_deref(), Some("sonnet"));
        } else {
            panic!("Expected Models command");
        }
    }

    #[test]
    fn test_serve_port_flag() {
        let cli = Cli::try_parse_from(["aivo", "serve", "--port", "8080"]).unwrap();
        if let Some(Commands::Serve(serve_args)) = cli.command {
            assert_eq!(serve_args.port, 8080);
        } else {
            panic!("Expected Serve command");
        }
    }

    /// Helper to simulate the 'use' alias rewriting done in main.rs
    fn rewrite_use_alias(args: &[&str]) -> Vec<String> {
        let raw: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        if raw.len() > 1 && raw[1] == "use" {
            let mut rewritten = vec![raw[0].clone(), "keys".to_string(), "use".to_string()];
            rewritten.extend_from_slice(&raw[2..]);
            rewritten
        } else {
            raw
        }
    }

    #[test]
    fn test_use_alias_with_key_name() {
        let args = rewrite_use_alias(&["aivo", "use", "my-key"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Keys(keys_args)) = cli.command {
            assert_eq!(keys_args.action.as_deref(), Some("use"));
            assert_eq!(keys_args.args, vec!["my-key"]);
        } else {
            panic!("Expected Keys command");
        }
    }

    #[test]
    fn test_use_alias_no_arg() {
        let args = rewrite_use_alias(&["aivo", "use"]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Keys(keys_args)) = cli.command {
            assert_eq!(keys_args.action.as_deref(), Some("use"));
            assert!(keys_args.args.is_empty());
        } else {
            panic!("Expected Keys command");
        }
    }

    #[test]
    fn test_keys_add_flags() {
        let cli = Cli::try_parse_from([
            "aivo",
            "keys",
            "add",
            "--name",
            "openrouter",
            "--base-url",
            "https://openrouter.ai/api/v1",
            "--key",
            "sk-or-v1-test",
        ])
        .unwrap();

        if let Some(Commands::Keys(keys_args)) = cli.command {
            assert_eq!(keys_args.action.as_deref(), Some("add"));
            assert_eq!(keys_args.name.as_deref(), Some("openrouter"));
            assert_eq!(
                keys_args.base_url.as_deref(),
                Some("https://openrouter.ai/api/v1")
            );
            assert_eq!(keys_args.key.as_deref(), Some("sk-or-v1-test"));
            assert!(keys_args.args.is_empty());
        } else {
            panic!("Expected Keys command");
        }
    }

    #[test]
    fn test_chat_command_no_model() {
        let cli = Cli::try_parse_from(["aivo", "chat"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, None);
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_with_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--model", "gpt-4o"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_model_no_value() {
        // --model with no value → triggers picker (Some(""))
        let cli = Cli::try_parse_from(["aivo", "chat", "--model"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_command_with_short_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-m", "claude-sonnet-4-5"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.model, Some("claude-sonnet-4-5".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_run_args_key_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--key", "my-key"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_key_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-k", "a1b2"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("a1b2".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_key_no_value() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "-k"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some(String::new()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_run_args_key_equals_syntax() {
        let cli = Cli::try_parse_from(["aivo", "run", "claude", "--key=my-key"]).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Run command");
        }
    }

    #[test]
    fn test_chat_args_key_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--key", "my-key"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_key_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "a1b2"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("a1b2".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_models_args_key_no_value() {
        let cli = Cli::try_parse_from(["aivo", "models", "-k"]).unwrap();
        if let Some(Commands::Models(models_args)) = cli.command {
            assert_eq!(models_args.key, Some(String::new()));
        } else {
            panic!("Expected Models command");
        }
    }

    #[test]
    fn test_serve_args_key_no_value() {
        let cli = Cli::try_parse_from(["aivo", "serve", "-k"]).unwrap();
        if let Some(Commands::Serve(serve_args)) = cli.command {
            assert_eq!(serve_args.key, Some(String::new()));
        } else {
            panic!("Expected Serve command");
        }
    }

    #[test]
    fn test_chat_args_key_no_value() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some(String::new()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_key_with_model() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "my-key", "-m", "gpt-4o"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_execute_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-x", "hello"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.execute, Some("hello".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_execute_no_value() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-x"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.execute, Some(String::new()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_execute_long_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--execute", "hello world"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.execute, Some("hello world".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_execute_with_model_and_key() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "my-key", "-m", "gpt-4o", "-x", "hi"])
            .unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
            assert_eq!(chat_args.execute, Some("hi".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_empty_key_and_model_force_pickers() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "-m"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some(String::new()));
            assert_eq!(chat_args.model, Some(String::new()));
        } else {
            panic!("Expected Chat command");
        }
    }
}
