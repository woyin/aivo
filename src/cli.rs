//! CLI argument parsing and command routing.
//! Uses clap for argument parsing.
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

    /// Dump the full command tree (commands, flags, descriptions) as JSON.
    /// Intended for AI agents and tooling that needs reliable
    /// machine-readable command discovery.
    #[arg(long = "help-json", global = true, help = "Dump command tree as JSON")]
    pub help_json: bool,

    /// Display the current version
    #[arg(short, long, global = true, help = "Display the current version")]
    pub version: bool,
}

/// Available commands for the CLI
#[derive(Subcommand, Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Commands {
    /// Run AI tools (claude, codex, codex-app, gemini, opencode, pi) - all args passed through
    Run(RunArgs),

    /// Manage API keys (use <id|name>, rm <id|name>, add, cat, edit)
    Keys(KeysArgs),

    /// Manage your aivo account (info, usage, login, logout)
    Account(AccountArgs),

    /// Alias for `aivo account login` (kept for backward compatibility)
    #[command(hide = true)]
    Login(LoginArgs),

    /// Alias for `aivo account logout` (kept for backward compatibility)
    #[command(hide = true)]
    Logout(LogoutArgs),

    /// List available models from the active provider
    Models(ModelsArgs),

    /// Start the interactive chat TUI
    Chat(ChatArgs),

    /// Serve an OpenAI-compatible API that proxies to the active provider
    Serve(ServeArgs),

    /// Create, list, or remove model aliases
    Alias(AliasArgs),

    /// Show system info, keys, tools, and directory state
    #[command(alias = "ls", hide = true)]
    Info(InfoArgs),

    /// Show recent local logs from chat, run, and serve
    Logs(LogsArgs),

    /// Show usage statistics (tokens, requests, breakdowns)
    Stats(StatsArgs),

    /// Update the CLI tool to the latest version
    Update(UpdateArgs),

    /// Inspect or manage cached HuggingFace GGUF files
    Hf(HfArgs),

    /// Install, list, or remove plugins (sibling `aivo-<name>` binaries)
    #[command(alias = "plugin")]
    Plugins(PluginsArgs),

    /// Alias for `aivo logs share` — share a session via tunneled viewer URL.
    /// Both forms accept the same flags.
    Share(ShareArgs),
}

/// Arguments for `aivo login`.
#[derive(Args, Debug, Clone)]
pub struct LoginArgs {
    /// Label for this device in your account's device list
    /// (default: "aivo <version> on <hostname>").
    #[arg(long, value_name = "LABEL", value_parser = non_empty())]
    pub label: Option<String>,
}

/// Arguments for `aivo logout`.
#[derive(Args, Debug, Clone)]
pub struct LogoutArgs {
    /// Skip the confirmation prompt
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Arguments for `aivo account`. No subcommand → defaults to `info`.
#[derive(Args, Debug, Clone)]
pub struct AccountArgs {
    #[command(subcommand)]
    pub command: Option<AccountSubcommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum AccountSubcommand {
    /// Sign in and link this device to your aivo account
    Login(LoginArgs),

    /// Sign out on this device (unlinks it from your account)
    Logout(LogoutArgs),

    /// Show account identity, plan, and linked-device count
    #[command(alias = "status")]
    Info(AccountInfoArgs),

    /// Show usage: requests/tokens, daily caps, and per-model breakdown
    Usage(AccountUsageArgs),

    /// Open your account dashboard in the browser
    Open(AccountOpenArgs),
}

/// Arguments for `aivo account open`.
#[derive(Args, Debug, Clone)]
pub struct AccountOpenArgs {
    /// Print the dashboard URL instead of opening a browser
    #[arg(long)]
    pub print: bool,
}

/// Arguments for `aivo account info`.
#[derive(Args, Debug, Clone)]
pub struct AccountInfoArgs {
    /// Output account info as JSON
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `aivo account usage`.
#[derive(Args, Debug, Clone)]
pub struct AccountUsageArgs {
    /// Output usage as JSON (verbatim gateway shape)
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `aivo logs share` (and the hidden top-level `aivo share` alias).
#[derive(Args, Debug, Clone)]
pub struct ShareArgs {
    /// Session id from `aivo logs` (claude / codex / gemini / pi / opencode / chat).
    #[arg(value_name = "SESSION_ID")]
    pub session_id: Option<String>,

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

/// Arguments for `aivo hf`. No subcommand → defaults to `list`. Real clap
/// subcommands so each verb gets its own help + validation + completions
/// instead of being squeezed into a positional `ACTION` string.
#[derive(Args, Debug, Clone)]
pub struct HfArgs {
    #[command(subcommand)]
    pub command: Option<HfSubcommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum HfSubcommand {
    /// List cached repos with size and last-used time
    #[command(alias = "ls")]
    List(HfListArgs),

    /// Pre-download a GGUF for future runs (no server spawn)
    Pull(HfPullArgs),

    /// Delete cached files for one repo (one quant by default, or `--all`)
    #[command(alias = "remove")]
    Rm(HfRmArgs),

    /// Delete every cached repo
    Clean(HfCleanArgs),
}

#[derive(Args, Debug, Clone)]
pub struct HfListArgs {
    /// Show one row per cached file (filename, size, age) instead of a
    /// repo-aggregated summary.
    #[arg(long)]
    pub verbose: bool,
}

#[derive(Args, Debug, Clone)]
pub struct HfPullArgs {
    /// `hf:<owner>/<repo>[:<quant>]` short ref, a full
    /// `https://huggingface.co/<owner>/<repo>` URL, or a local `.gguf`
    /// path to import into the cache.
    #[arg(value_name = "REF_OR_PATH")]
    pub reference: String,

    /// When importing a local file, override the auto-derived
    /// `<owner>/<repo>` cache name (default: `local/<filename-stem>`).
    /// Ignored for `hf:` / URL refs.
    #[arg(long = "as", value_name = "OWNER/REPO")]
    pub as_repo: Option<String>,

    /// Ignore any cached resolve for this ref and re-resolve from
    /// HuggingFace. Use after a failed pull so a prior (stale or gated)
    /// pick doesn't pin the retry to the same file.
    #[arg(long)]
    pub refresh: bool,
}

#[derive(Args, Debug, Clone)]
pub struct HfRmArgs {
    /// Repo path: `<owner>/<repo>`.
    #[arg(value_name = "REPO")]
    pub repo: String,

    /// Remove only the file matching this quant tag (e.g. `Q5_K_M`).
    /// When omitted and the repo has a single cached file, that file is
    /// removed. When omitted and multiple files are cached, the command
    /// refuses unless `--all` is passed.
    #[arg(long, value_name = "QUANT")]
    pub quant: Option<String>,

    /// Remove every cached file under this repo regardless of quant.
    #[arg(long, conflicts_with = "quant")]
    pub all: bool,

    /// Skip the confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

#[derive(Args, Debug, Clone)]
pub struct HfCleanArgs {
    /// Skip the confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Arguments for `aivo plugins`. No subcommand → defaults to `list`. Plugins
/// are sibling `aivo-<name>` executables; `aivo <name>` runs the matching one.
#[derive(Args, Debug, Clone)]
pub struct PluginsArgs {
    #[command(subcommand)]
    pub command: Option<PluginsSubcommand>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum PluginsSubcommand {
    /// List installed plugins and where each resolves
    #[command(alias = "ls")]
    List,

    /// Install a plugin from a local file or an http(s):// URL
    Install(PluginInstallArgs),

    /// Re-install a plugin from the source it was installed from
    Update(PluginUpdateArgs),

    /// Remove an installed plugin by name
    #[command(name = "rm", alias = "remove")]
    Remove(PluginRemoveArgs),
}

#[derive(Args, Debug, Clone)]
pub struct PluginInstallArgs {
    /// What to install: a local path, an `http(s)://` URL, `github:owner/repo[@tag]`
    /// (or `gh:` / a bare github.com URL), `npm:[@scope/]pkg[@version]`, or `cargo:crate`.
    #[arg(value_name = "SOURCE", value_parser = non_empty())]
    pub source: String,

    /// Plugin name (default: inferred from the source file name). The binary
    /// is stored as `aivo-<name>` and invoked as `aivo <name>`.
    #[arg(long, value_name = "NAME", value_parser = non_empty())]
    pub name: Option<String>,

    /// Overwrite an existing plugin of the same name.
    #[arg(short = 'f', long)]
    pub force: bool,

    /// Skip the consent prompts: grant the manifest's grantable capabilities
    /// and approve the binary's first run (for non-interactive installs).
    #[arg(long)]
    pub trust: bool,
}

#[derive(Args, Debug, Clone)]
pub struct PluginUpdateArgs {
    /// Plugin to update, with or without the `aivo-` prefix. Omit to update
    /// every plugin with a recorded install source.
    #[arg(value_name = "NAME", value_parser = non_empty())]
    pub name: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub struct PluginRemoveArgs {
    /// Plugin name, with or without the `aivo-` prefix (e.g. `amp`).
    #[arg(value_name = "NAME", value_parser = non_empty())]
    pub name: String,

    /// Skip the confirmation prompt.
    #[arg(short = 'y', long)]
    pub yes: bool,
}

/// Arguments for `aivo alias`
#[derive(Args, Debug, Clone)]
pub struct AliasArgs {
    /// Alias name, `name=model` shorthand, or the `rm`/`remove`/`list`/`ls`
    /// keywords.
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
    /// The action to perform (list, use, add, rm, cat, edit, reauth, ping, reset-route, export, import)
    #[arg(
        value_name = "ACTION",
        help = "Action to perform: list, use, add, rm, cat, edit, reauth, ping, reset-route, export, import"
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

    /// Comma-separated key ids for `keys export` (default: all keys)
    #[arg(long, value_name = "IDS", value_delimiter = ',')]
    pub ids: Vec<String>,

    /// Read the export/import password from stdin instead of prompting
    #[arg(long = "password-stdin")]
    pub password_stdin: bool,

    /// On `keys import` conflict, replace the existing key in place
    #[arg(long, conflicts_with = "rename")]
    pub overwrite: bool,

    /// On `keys import` conflict, insert the imported key under a fresh id
    #[arg(long)]
    pub rename: bool,

    /// On `keys export`, include the device-bound aivo-starter key
    /// (filtered out by default; not portable between machines)
    #[arg(long = "include-starter")]
    pub include_starter: bool,

    /// On `keys export`, include OAuth/login sessions (Claude, Codex,
    /// Gemini, Copilot, Cursor login). Off by default — subscription-bound
    /// credentials shouldn't travel silently with an API-key backup.
    #[arg(long = "include-oauth")]
    pub include_oauth: bool,

    /// On `keys export`, overwrite an existing file at the target path
    #[arg(long)]
    pub force: bool,
}

/// Arguments for the run command
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    /// The AI tool to run (claude, codex, codex-app, gemini, opencode, pi)
    #[arg(
        value_name = "TOOL",
        help = "AI tool to run: claude, codex, codex-app, gemini, opencode, or pi"
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

    /// Without a tool name: skip the tool picker and replay the last
    /// selection as-is (no prompts)
    #[arg(short = 'y', long)]
    pub yes: bool,

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

    /// Claude only: opt into a larger context window. Accepts any `<N>m`
    /// (e.g. `1m`, `2m`, `12m`); aivo only validates shape.
    ///
    /// Appends a `[<size>]` suffix to the model name in every default slot
    /// env var (`ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, etc.) so
    /// Claude Code opts into the matching beta context tier. Per-slot
    /// overrides (`--haiku-model`, `--sonnet-model`, …) are left verbatim.
    #[arg(long = "max-context", value_name = "SIZE")]
    pub max_context: Option<String>,

    /// Shorthand for `--max-context=1m`.
    #[arg(long = "1m")]
    pub one_m: bool,

    /// Shorthand for `--max-context=2m`.
    #[arg(long = "2m")]
    pub two_m: bool,

    /// Pi only: route through aivo's responses-to-chat router. This is the
    /// default for `pi` (pass `--transparent` to opt out). Applies
    /// model-name + protocol transforms and normalizes the SSE stream, so
    /// upstreams that emit malformed chunks (e.g. newapi omitting
    /// `finish_reason`) still parse. Same path as `--debug`, no JSONL log.
    #[arg(long)]
    pub transform: bool,

    /// Pi only: opt out of the default transform router and talk to the
    /// upstream natively (transparent passthrough).
    #[arg(long)]
    pub transparent: bool,

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

    /// Optional positional: `hf:<owner>/<repo>[:<quant>]` short ref or
    /// `https://huggingface.co/...` URL. When given, `aivo serve`
    /// spawns a local llama-server for that model and proxies to it
    /// (no API-key picker; `-k` and `--failover` are ignored).
    /// Without it, `aivo serve` behaves as a proxy to the active
    /// provider key, as it always has.
    #[arg(value_name = "REF")]
    pub reference: Option<String>,
}

/// Arguments for the stats command
#[derive(Args, Debug, Clone)]
pub struct StatsArgs {
    /// Filter to one tool: claude, codex, gemini, opencode, pi, chat, or an
    /// installed coding-agent plugin (e.g. omp). Mirrors `aivo logs --by`.
    #[arg(long, value_name = "NAME", value_parser = non_empty())]
    pub by: Option<String>,

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
    /// Refresh model data from models.dev (context windows, capabilities); leaves the binary alone
    #[arg(long)]
    pub sync_model_data: bool,
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
    /// Action: list (default), show, share, or prune
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
    /// `logs share`: skip redaction (default: scrub API keys, OAuth, $HOME, secret env).
    #[arg(long)]
    pub no_redact: bool,

    /// `logs share`: open the share URL in the default browser once ready.
    #[arg(long)]
    pub open: bool,

    /// `logs share`: bind only on 127.0.0.1 — local debugging without the public tunnel.
    #[arg(long, hide = true)]
    pub debug_local_only: bool,

    /// `logs prune`: skip the interactive confirmation and delete immediately.
    #[arg(short = 'f', long)]
    pub force: bool,
}

/// Arguments for the chat command
#[derive(Args, Debug, Clone)]
pub struct ChatArgs {
    /// Optional positional: `hf:<owner>/<repo>[:<quant>]` short ref or
    /// `https://huggingface.co/...` URL. Equivalent to `-m <REF>` with
    /// the local llama-server lifecycle wired up.
    #[arg(value_name = "REF")]
    pub reference: Option<String>,

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

    /// Resume a saved chat: bare opens the session picker, `last` reopens the
    /// most recent chat, or pass a session id to jump straight to it (same as
    /// the in-chat `/resume [query]`).
    #[arg(long, value_name = "last|SESSION_ID", num_args = 0..=1, default_missing_value = "")]
    pub resume: Option<String>,

    /// Send one prompt and exit; reads stdin when no value given (plain single-turn, no tools)
    #[arg(
        short = 'p',
        long = "prompt",
        visible_short_alias = 'x',
        visible_alias = "execute",
        value_name = "PROMPT",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub prompt: Option<String>,

    /// Run the agent (tools + multi-step loop) on one prompt and exit; reads stdin
    /// when no value given. Like -p but agentic. Text-only; conflicts with -p.
    #[arg(
        short = 'e',
        long = "exec",
        value_name = "PROMPT",
        num_args = 0..=1,
        default_missing_value = "",
        conflicts_with = "prompt"
    )]
    pub exec: Option<String>,

    /// Print the upstream provider's raw JSON response (requires -p; useful for scripting)
    #[arg(long, requires = "prompt")]
    pub json: bool,

    /// Attach a file or image to the next chat message (repeatable)
    #[arg(long = "attach", value_name = "PATH", value_parser = non_empty())]
    pub attachments: Vec<String>,

    /// Log all aivo HTTP requests/responses to a JSONL file (default:
    /// ~/.config/aivo/logs/debug-<ts>-<pid>.jsonl). Sensitive headers and
    /// URL query params are redacted.
    #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "")]
    pub debug: Option<String>,

    /// Manually set this model's context window for the session (e.g. 200k,
    /// 128000, 1m). Use for new models aivo doesn't know the window for yet —
    /// it drives compaction and the context-usage stat. Not persisted.
    #[arg(long, value_name = "SIZE")]
    pub max_context: Option<String>,

    /// Shorthand for `--max-context=1m`.
    #[arg(long = "1m")]
    pub one_m: bool,

    /// Shorthand for `--max-context=2m`.
    #[arg(long = "2m")]
    pub two_m: bool,

    /// Print the resolved key, model, and endpoint without connecting
    #[arg(long)]
    pub dry_run: bool,

    /// Publish a live, redacted view of this chat to a viewer URL (shown in the
    /// TUI). Needs a linked account (`aivo login`); toggle in-chat with `/share`.
    #[arg(long)]
    pub share: bool,
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
        let aliases = ["claude", "codex", "codex-app", "gemini", "opencode", "pi"];
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
    fn test_tool_alias_codex_app_with_path() {
        let args = rewrite_alias(&["aivo", "codex-app", "--model", "gpt-5", "."]);
        let cli = Cli::try_parse_from(&args).unwrap();
        if let Some(Commands::Run(run_args)) = cli.command {
            assert_eq!(run_args.tool, Some("codex-app".to_string()));
            assert_eq!(run_args.model, Some("gpt-5".to_string()));
            assert!(run_args.args.contains(&".".to_string()));
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
    fn test_chat_command_resume_variants() {
        // No flag → None (fresh chat).
        let cli = Cli::try_parse_from(["aivo", "chat"]).unwrap();
        let Some(Commands::Chat(args)) = cli.command else {
            panic!("Expected Chat command");
        };
        assert_eq!(args.resume, None);

        // Bare --resume → Some("") (opens the session picker).
        let cli = Cli::try_parse_from(["aivo", "chat", "--resume"]).unwrap();
        let Some(Commands::Chat(args)) = cli.command else {
            panic!("Expected Chat command");
        };
        assert_eq!(args.resume, Some(String::new()));

        // --resume <id> → Some(id) (jumps to that session).
        let cli = Cli::try_parse_from(["aivo", "chat", "--resume", "abc123"]).unwrap();
        let Some(Commands::Chat(args)) = cli.command else {
            panic!("Expected Chat command");
        };
        assert_eq!(args.resume, Some("abc123".to_string()));
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
    fn test_chat_args_prompt_short_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-p", "hello"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.prompt, Some("hello".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_dry_run_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--dry-run"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert!(chat_args.dry_run);
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_dry_run_defaults_off() {
        let cli = Cli::try_parse_from(["aivo", "chat"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert!(!chat_args.dry_run);
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_prompt_no_value() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-p"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.prompt, Some(String::new()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_prompt_long_flag() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--prompt", "hello world"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.prompt, Some("hello world".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_prompt_legacy_short_alias_x() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-x", "hello"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.prompt, Some("hello".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_prompt_legacy_long_alias_execute() {
        let cli = Cli::try_parse_from(["aivo", "chat", "--execute", "hello world"]).unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.prompt, Some("hello world".to_string()));
        } else {
            panic!("Expected Chat command");
        }
    }

    #[test]
    fn test_chat_args_prompt_with_model_and_key() {
        let cli = Cli::try_parse_from(["aivo", "chat", "-k", "my-key", "-m", "gpt-4o", "-p", "hi"])
            .unwrap();
        if let Some(Commands::Chat(chat_args)) = cli.command {
            assert_eq!(chat_args.key, Some("my-key".to_string()));
            assert_eq!(chat_args.model, Some("gpt-4o".to_string()));
            assert_eq!(chat_args.prompt, Some("hi".to_string()));
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

    #[test]
    fn test_account_bare_has_no_subcommand() {
        let cli = Cli::try_parse_from(["aivo", "account"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            assert!(a.command.is_none());
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_info_json() {
        let cli = Cli::try_parse_from(["aivo", "account", "info", "--json"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            assert!(matches!(
                a.command,
                Some(AccountSubcommand::Info(AccountInfoArgs { json: true }))
            ));
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_info_status_alias() {
        let cli = Cli::try_parse_from(["aivo", "account", "status"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            assert!(matches!(a.command, Some(AccountSubcommand::Info(_))));
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_usage_json() {
        let cli = Cli::try_parse_from(["aivo", "account", "usage", "--json"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            assert!(matches!(
                a.command,
                Some(AccountSubcommand::Usage(AccountUsageArgs { json: true }))
            ));
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_open_print_flag() {
        let cli = Cli::try_parse_from(["aivo", "account", "open", "--print"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            assert!(matches!(
                a.command,
                Some(AccountSubcommand::Open(AccountOpenArgs { print: true }))
            ));
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_login_passes_label() {
        let cli = Cli::try_parse_from(["aivo", "account", "login", "--label", "work"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            match a.command {
                Some(AccountSubcommand::Login(login)) => {
                    assert_eq!(login.label.as_deref(), Some("work"));
                }
                _ => panic!("Expected Account login subcommand"),
            }
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_account_logout_yes() {
        let cli = Cli::try_parse_from(["aivo", "account", "logout", "-y"]).unwrap();
        if let Some(Commands::Account(a)) = cli.command {
            match a.command {
                Some(AccountSubcommand::Logout(logout)) => assert!(logout.yes),
                _ => panic!("Expected Account logout subcommand"),
            }
        } else {
            panic!("Expected Account command");
        }
    }

    #[test]
    fn test_top_level_login_logout_still_parse() {
        // Backward-compat aliases must keep working alongside `aivo account`.
        let cli = Cli::try_parse_from(["aivo", "login"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Login(_))));
        let cli = Cli::try_parse_from(["aivo", "logout", "-y"]).unwrap();
        assert!(matches!(cli.command, Some(Commands::Logout(_))));
    }
}
