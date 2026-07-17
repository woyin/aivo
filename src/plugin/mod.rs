//! External-subcommand plugin dispatch: `aivo <name>` / `aivo run <name>` for a
//! name aivo doesn't own runs a sibling `aivo-<name>` binary (git/cargo style).
//! Checked before clap, and only when the sibling exists, so built-ins, tools,
//! and the chat shortcut always win.

mod endpoint;
pub(crate) mod manifest;
pub(crate) mod registry;
pub(crate) mod source;
pub(crate) mod stats;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::constants::{KNOWN_TOOLS, RESERVED_ALIAS_NAMES};
use crate::errors::ExitCode;
use crate::services::path_search::{collect_path_dirs, find_in_dirs, is_executable};
use crate::services::session_store::{BundleAlias, SessionStore};
use crate::style;

pub(crate) const PLUGIN_PREFIX: &str = "aivo-";

/// `AIVO_PLUGIN_PROBE_TIMEOUT_MS` overrides a probe's default deadline
/// (manifest and stats probes alike). The integration suite depends on it both
/// ways: generous so full-suite CPU load can't starve a healthy probe into a
/// flake, tiny so hanging-probe tests are instant.
pub(crate) fn probe_timeout(default: std::time::Duration) -> std::time::Duration {
    crate::services::system_env::env_parse("AIVO_PLUGIN_PROBE_TIMEOUT_MS")
        .map(std::time::Duration::from_millis)
        .unwrap_or(default)
}

/// Well-known aivo plugins with a canonical install source. Invoking one that
/// isn't installed offers to install it on the spot instead of letting clap
/// reject the name as unknown. `(name, install_source)`.
const KNOWN_PLUGINS: &[(&str, &str)] = &[
    ("amp", "github:yuanchuan/aivo-amp"),
    ("copilot", "github:yuanchuan/aivo-copilot"),
    ("grok", "github:yuanchuan/aivo-grok"),
    ("omp", "github:yuanchuan/aivo-omp"),
];

/// Run the matching `aivo-<name>` plugin and return its exit code, or `None` if
/// none applies. Call before `Cli::parse_from` — clap rejects unknown subcommands.
/// Granted plugins (and `coding-agent` types) get a key/endpoint handoff and run
/// accounting via `endpoint::dispatch`; everything else spawns plain. A known
/// plugin (e.g. `amp`, `omp`) that isn't installed is offered an install on
/// the spot (TTY) — accepted installs run the original invocation; declined or
/// non-interactive ones get the `aivo plugins install` hint and a non-zero code.
pub async fn try_dispatch(
    raw_args: &[String],
    bundles: &HashMap<String, BundleAlias>,
    store: &SessionStore,
) -> Option<i32> {
    let (name, plugin_args) = resolve_invocation(raw_args, bundles)?;
    match discover(name) {
        Some(bin) => Some(endpoint::dispatch(name, &bin, plugin_args, store).await),
        // Not installed: offer the install for the well-known plugins;
        // otherwise fall through to clap (typo / unknown name / bare-prompt
        // rewrite).
        None => {
            let source = known_plugin_source(name)?;
            Some(install_and_dispatch(name, source, plugin_args, store).await)
        }
    }
}

/// Install-on-demand for an uninstalled well-known plugin: ask, install from
/// its known source, then run the original invocation. Non-interactive or
/// declined → the `plugins install` hint and a user-error code.
async fn install_and_dispatch(
    name: &str,
    source: &str,
    plugin_args: &[String],
    store: &SessionStore,
) -> i32 {
    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        print_missing_plugin_hint(name, source);
        return ExitCode::UserError.code();
    }
    // Declining needs no follow-up — the prompt already named the source.
    if !prompt_install_plugin(name, source) {
        return ExitCode::UserError.code();
    }
    if let Err(e) = crate::commands::plugins::install_for_dispatch(name, source).await {
        eprintln!("{} {e:#}", style::red("Error:"));
        return crate::errors::exit_code_for_error(&e).code();
    }
    // The install prompt's consent covered execution, so the first-dispatch
    // run gate would be a redundant second ask.
    registry::approve_run(name);
    match discover(name) {
        Some(bin) => endpoint::dispatch(name, &bin, plugin_args, store).await,
        None => {
            print_missing_plugin_hint(name, source);
            ExitCode::UserError.code()
        }
    }
}

/// Offer to install a well-known plugin right now. Caller gates on a TTY;
/// defaults to yes like the native-tool install prompt. Showing the source
/// makes this consent cover the binary's first run.
fn prompt_install_plugin(name: &str, source: &str) -> bool {
    use std::io::Write;
    eprintln!(
        "  {} {} is not installed.",
        style::yellow("?"),
        style::cyan(format!("`{name}`")),
    );
    eprint!("    Install it from {}? [Y/n] ", style::cyan(source));
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    match std::io::stdin().read_line(&mut input) {
        // EOF (^D) is a non-answer — only an actual Enter takes the default.
        Ok(0) | Err(_) => false,
        Ok(_) => matches!(input.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes"),
    }
}

/// The `aivo plugins install` source for a well-known plugin, or `None` if
/// `name` isn't one. Pure lookup.
fn known_plugin_source(name: &str) -> Option<&'static str> {
    KNOWN_PLUGINS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, source)| *source)
}

/// Tell the user the name is an aivo plugin and how to add it.
fn print_missing_plugin_hint(name: &str, source: &str) {
    eprintln!(
        "{} `{name}` is an aivo plugin and isn't installed.",
        style::red("Error:"),
    );
    eprintln!(
        "  Install it:  {}",
        style::cyan(format!("aivo plugins install {source}")),
    );
    eprintln!(
        "  Then run:    {}",
        style::dim(format!("aivo {name} --help")),
    );
}

/// Every installed plugin as `(name, path)` from one search-path sweep; first
/// match per name wins (managed dir → exe dir → `$PATH`), like `discover`.
pub fn installed_plugins() -> Vec<(String, PathBuf)> {
    let mut found: std::collections::BTreeMap<String, PathBuf> = std::collections::BTreeMap::new();
    for dir in search_dirs() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !is_executable(&path) {
                continue;
            }
            // `file_stem` drops the `.exe`/`.cmd` extension on Windows.
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Some(name) = stem.strip_prefix(PLUGIN_PREFIX)
                && !name.is_empty()
            {
                found.entry(name.to_string()).or_insert(path);
            }
        }
    }
    found.into_iter().collect()
}

/// Bare plugin names for `--help` / `--help-json` listings.
pub fn installed_plugin_names() -> Vec<String> {
    installed_plugins()
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

/// `(plugin_name, transcript_format, sessions_dir)` for every installed plugin
/// that declares a transcript source in its manifest — so `aivo share` can read
/// a plugin run with a built-in reader. Plain strings (no type leak into the
/// share resolver); the caller validates the format and expands `~`.
pub fn installed_transcript_sources() -> Vec<(String, String, String)> {
    registry::load()
        .plugins
        .into_iter()
        .filter_map(|(name, rec)| {
            let t = rec.manifest?.transcripts?;
            Some((name, t.format, t.dir))
        })
        .collect()
}

/// Names of installed `type: coding-agent` plugins — the ones whose launches
/// are recorded into `aivo logs`/`aivo stats`. Stats treats these as valid tool
/// names so plugin runs show up alongside native tools.
pub fn coding_agent_plugin_names() -> std::collections::HashSet<String> {
    registry::load()
        .plugins
        .into_iter()
        .filter(|(_, rec)| rec.manifest.as_ref().is_some_and(|m| m.is_coding_agent()))
        .map(|(name, _)| name)
        .collect()
}

/// Curated one-line details for well-known coding-agent plugins, in the same
/// terse style as the native tools'. Manifest descriptions are free-form and
/// often too long for a picker row.
fn known_coding_agent_description(name: &str) -> Option<&'static str> {
    match name {
        "amp" => Some("Sourcegraph's coding agent."),
        "copilot" => Some("GitHub's official terminal coding agent."),
        "grok" => Some("An open-source coding agent for the Grok API."),
        "omp" => Some("Oh My Pi, a terminal coding agent built on pi-mono."),
        _ => None,
    }
}

/// Normalize a free-form manifest description to the picker's house style:
/// first line only, sentence-cased, capped at a word boundary, trailing
/// period.
fn normalize_plugin_description(desc: &str) -> Option<String> {
    const MAX_CHARS: usize = 60;
    let first = desc.lines().next()?.trim();
    let mut chars = first.chars();
    let mut out: String = chars.next()?.to_uppercase().chain(chars).collect();
    if out.chars().count() > MAX_CHARS {
        let head: String = out.chars().take(MAX_CHARS).collect();
        let cut = match head.rfind(' ') {
            Some(i) => &head[..i],
            None => head.as_str(),
        };
        out = format!("{}…", cut.trim_end_matches([' ', ',', ';', ':', '.']));
    } else if !out.ends_with(['.', '!', '?', '…']) {
        out.push('.');
    }
    Some(out)
}

/// Detail text for installed coding-agent plugins, keyed by name — curated
/// for well-known plugins, normalized first manifest line otherwise.
pub fn coding_agent_descriptions() -> HashMap<String, String> {
    registry::load()
        .plugins
        .into_iter()
        .filter_map(|(name, rec)| {
            let manifest = rec.manifest?;
            if !manifest.is_coding_agent() {
                return None;
            }
            let detail = match known_coding_agent_description(&name) {
                Some(curated) => curated.to_string(),
                None => manifest
                    .description
                    .as_deref()
                    .and_then(normalize_plugin_description)
                    .unwrap_or_else(|| "A coding agent plugin.".to_string()),
            };
            Some((name, detail))
        })
        .collect()
}

/// Sorted `type: coding-agent` plugins whose binary is still discoverable —
/// offered alongside native tools in the start flow's tool picker.
pub fn launchable_coding_agents() -> Vec<String> {
    let mut names: Vec<String> = coding_agent_plugin_names()
        .into_iter()
        .filter(|name| discover(name).is_some())
        .collect();
    names.sort();
    names
}

/// Dispatch an installed plugin by name through the standard grant/endpoint
/// path, returning its exit code — `None` when no `aivo-<name>` binary exists.
pub async fn dispatch_installed(name: &str, args: &[String], store: &SessionStore) -> Option<i32> {
    let bin = discover(name)?;
    Some(endpoint::dispatch(name, &bin, args, store).await)
}

/// argv → `(plugin_name, args_after_name)`, or `None` if no plugin applies.
/// `aivo amp …` and `aivo run amp …` both yield the same `aivo-amp …`.
fn resolve_invocation<'a>(
    raw_args: &'a [String],
    bundles: &HashMap<String, BundleAlias>,
) -> Option<(&'a str, &'a [String])> {
    let first = raw_args.get(1)?;
    if first == "run" {
        // `aivo run <name> …` — forward an unknown run-tool to its sibling.
        let name = raw_args.get(2)?;
        return dispatchable(name, bundles).then_some((name.as_str(), &raw_args[3..]));
    }
    dispatchable(first, bundles).then_some((first.as_str(), &raw_args[2..]))
}

/// True when `name` is eligible for plugin dispatch — aivo doesn't own it (not a
/// built-in, tool, chat ref, or user bundle).
fn dispatchable(name: &str, bundles: &HashMap<String, BundleAlias>) -> bool {
    if name.is_empty() || name.starts_with('-') {
        return false;
    }
    if is_reserved_plugin_name(name) {
        return false;
    }
    // Chat refs (mirrors rewrite_cli_args).
    if name.starts_with("hf:") || name.starts_with("http://") || name.starts_with("https://") {
        return false;
    }
    !bundles.contains_key(name)
}

/// `~/.config/aivo/plugins` — the directory `aivo plugins install` manages and
/// the first place discovery looks. `None` when the home dir can't be resolved.
pub fn plugins_dir() -> Option<PathBuf> {
    Some(crate::services::paths::config_dir().join("plugins"))
}

/// Locate `aivo-<name>` across the search path.
pub fn discover(name: &str) -> Option<PathBuf> {
    find_in_dirs(&format!("{PLUGIN_PREFIX}{name}"), &search_dirs())
}

/// Directories searched for `aivo-<name>`, in priority order: the managed
/// plugins dir, then next to the running executable, then every `$PATH` entry.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(p) = plugins_dir() {
        dirs.push(p);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        dirs.push(dir.to_path_buf());
    }
    dirs.extend(collect_path_dirs());
    dirs
}

/// True when `name` is a built-in/shortcut/tool aivo owns — `install` refuses
/// these since the built-in always shadows the plugin.
pub fn is_reserved_plugin_name(name: &str) -> bool {
    name == "help" || RESERVED_ALIAS_NAMES.contains(&name) || KNOWN_TOOLS.contains(&name)
}

/// Infer a plugin name from an install source. Scheme-aware (github:/npm:/cargo:/
/// URL/local) — delegates to the source classifier.
pub fn infer_plugin_name(source: &str) -> Option<String> {
    source::suggested_name(source)
}

/// Spawn the plugin with stdio inherited and wait for it. Spawn-and-wait (not
/// `exec`) so aivo's own cleanup paths still run. Always sets `AIVO_CONFIG_DIR`
/// (+ `AIVO_DEBUG_LOG` under `--debug`); `extra_env` carries the capability-gated
/// key/endpoint handoff (env only — never argv, so secrets don't leak via `ps`).
/// Returns the child's exit code.
async fn exec_plugin(
    bin: &Path,
    args: &[String],
    config_dir: &Path,
    extra_env: &[(String, String)],
) -> i32 {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.args(args);
    cmd.env("AIVO_CONFIG_DIR", config_dir);
    if let Some(log) = debug_log_path(args) {
        cmd.env("AIVO_DEBUG_LOG", log);
    }
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    cmd.kill_on_drop(true);
    match cmd.status().await {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!(
                "{} failed to launch plugin {}: {e}",
                style::red("Error:"),
                bin.display(),
            );
            ExitCode::UserError.code()
        }
    }
}

/// Shared y/N consent prompt for a plugin's grantable capabilities. Callers
/// gate on a TTY; this just asks and reads. Used at install and on first dispatch.
/// First-dispatch gate for a remote-installed plugin aivo has never executed:
/// the manifest probe that follows would be the binary's first run, so confirm
/// before executing anything. TTY-only; callers skip the gate when scripted.
pub(crate) fn prompt_first_run(name: &str, source: &str) -> bool {
    use std::io::Write;
    eprintln!(
        "  {} first run of plugin `{}` (installed from {})",
        style::yellow("?"),
        name,
        source,
    );
    eprintln!(
        "    {}",
        style::dim("it runs unsandboxed with your user's permissions — only run plugins you trust")
    );
    eprint!("  {} run it? [y/N] ", style::yellow("?"));
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

pub(crate) fn prompt_capability_grant(name: &str, caps: &[String]) -> bool {
    use std::io::Write;
    eprintln!(
        "  {} {} requests {} — a per-launch local endpoint for your selected key.",
        style::yellow("?"),
        style::cyan(format!("`{name}`")),
        style::cyan(caps.join(", ")),
    );
    eprint!("    Grant it? [y/N] ");
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// Aivo-level flags pulled out of a coding-agent plugin's argv, so they behave
/// like native tools (`aivo <tool> -k …`) and don't leak to the wrapped tool.
pub(crate) struct AivoFlags {
    /// `Some("")` = bare `-k` (open the picker); `Some(id)` = that key; `None` = absent.
    pub key: Option<String>,
    /// `Some("")` = bare `-m` (open the picker); `Some(model)` = that model; `None` = absent.
    pub model: Option<String>,
    /// Resolved debug log path from `--debug` / `--debug=<path>`.
    pub debug_log: Option<PathBuf>,
    /// `--dry-run` present: preview the resolved key/model/command instead of
    /// launching (mirrors native `aivo run --dry-run`).
    pub dry_run: bool,
    /// `--max-context <SIZE>`: manual context-window size (raw; parsed by caller).
    pub max_context: Option<String>,
    /// argv with aivo-owned flags removed. Tool-specific flags and prompt args
    /// are preserved.
    pub rest: Vec<String>,
}

/// Extract `-k`/`--key` (bare → picker), `-m`/`--model`, `--debug`, and
/// `--dry-run` from a coding-agent plugin's argv, stripping them from `rest`.
/// First occurrence of each wins; a value form consumes the next arg only when
/// it isn't itself a flag.
pub(crate) fn extract_aivo_flags(args: &[String]) -> AivoFlags {
    let mut key: Option<String> = None;
    let mut model: Option<String> = None;
    let mut debug_log: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut max_context: Option<String> = None;
    let mut rest: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if let Some(v) = a.strip_prefix("--key=").or_else(|| a.strip_prefix("-k=")) {
            key.get_or_insert_with(|| v.to_string());
        } else if a == "--key" || a == "-k" {
            if key.is_none() {
                key = Some(match args.get(i + 1) {
                    Some(n) if !n.is_empty() && !n.starts_with('-') => {
                        i += 1;
                        n.clone()
                    }
                    _ => String::new(), // bare `-k` → picker
                });
            }
        } else if let Some(v) = a.strip_prefix("--model=").or_else(|| a.strip_prefix("-m=")) {
            model.get_or_insert_with(|| v.to_string());
        } else if a == "--model" || a == "-m" {
            if model.is_none() {
                model = Some(match args.get(i + 1) {
                    Some(n) if !n.is_empty() && !n.starts_with('-') => {
                        i += 1;
                        n.clone()
                    }
                    _ => String::new(), // bare `-m` → picker
                });
            }
        } else if a == "--debug" {
            debug_log.get_or_insert_with(crate::services::http_debug::default_log_path);
        } else if let Some(rest_path) = a.strip_prefix("--debug=") {
            debug_log.get_or_insert_with(|| debug_path_from_value(rest_path));
        } else if a == "--dry-run" {
            dry_run = true;
        } else if let Some(v) = a.strip_prefix("--max-context=") {
            max_context.get_or_insert_with(|| v.to_string());
        } else if a == "--max-context" {
            if max_context.is_none()
                && let Some(n) = args
                    .get(i + 1)
                    .filter(|n| !n.is_empty() && !n.starts_with('-'))
            {
                max_context = Some(n.clone());
                i += 1;
            }
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    AivoFlags {
        key,
        model,
        debug_log,
        dry_run,
        max_context,
        rest,
    }
}

/// Remove only `-k`/`--key` and `-m`/`--model` (and their separate values) from argv,
/// leaving every other arg — including `--debug`/`--dry-run` and the plugin's own flags — in
/// place. Used for endpoint-granted (non-coding-agent) plugins, where aivo owns key/model
/// selection but the plugin keeps the rest of its argv. Value-consumption mirrors
/// `extract_aivo_flags`: a `-k`/`-m` consumes the next arg only when it isn't itself a flag.
pub(crate) fn strip_key_model_flags(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a.starts_with("--key=")
            || a.starts_with("-k=")
            || a.starts_with("--model=")
            || a.starts_with("-m=")
        {
            // inline value form → drop
        } else if a == "--key" || a == "-k" || a == "--model" || a == "-m" {
            // drop the flag, and a following separate value when it isn't itself a flag
            if let Some(n) = args.get(i + 1)
                && !n.is_empty()
                && !n.starts_with('-')
            {
                i += 1;
            }
        } else {
            out.push(args[i].clone());
        }
        i += 1;
    }
    out
}

/// Resolve the debug-log path to hand a plugin, mirroring aivo's `--debug`
/// handling: `--debug=<path>` uses that path, bare `--debug` uses the shared
/// default. `None` when `--debug` isn't present.
fn debug_log_path(args: &[String]) -> Option<PathBuf> {
    for a in args {
        if a == "--debug" {
            return Some(crate::services::http_debug::default_log_path());
        }
        if let Some(rest) = a.strip_prefix("--debug=") {
            return Some(debug_path_from_value(rest));
        }
    }
    None
}

fn debug_path_from_value(value: &str) -> PathBuf {
    if value.is_empty() {
        crate::services::http_debug::default_log_path()
    } else {
        PathBuf::from(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_description_sentence_cased_with_period() {
        assert_eq!(
            normalize_plugin_description("a coding agent for acme"),
            Some("A coding agent for acme.".to_string())
        );
        // Existing terminal punctuation is kept as-is.
        assert_eq!(
            normalize_plugin_description("Reviews diffs. "),
            Some("Reviews diffs.".to_string())
        );
    }

    #[test]
    fn plugin_description_takes_first_line_and_truncates_on_word_boundary() {
        assert_eq!(
            normalize_plugin_description("Fast agent\nsecond line"),
            Some("Fast agent.".to_string())
        );
        assert_eq!(
            normalize_plugin_description(
                "Run GitHub Copilot CLI on aivo-managed keys, models and endpoints"
            ),
            Some("Run GitHub Copilot CLI on aivo-managed keys, models and…".to_string())
        );
    }

    #[test]
    fn plugin_description_empty_is_none() {
        assert_eq!(normalize_plugin_description(""), None);
        assert_eq!(normalize_plugin_description("  \n x"), None);
    }

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn no_bundles() -> HashMap<String, BundleAlias> {
        HashMap::new()
    }

    #[test]
    fn unknown_top_level_name_dispatches_with_remaining_args() {
        let a = args(&["aivo", "amp", "trust", "list"]);
        let (name, rest) = resolve_invocation(&a, &no_bundles()).unwrap();
        assert_eq!(name, "amp");
        assert_eq!(rest, &["trust".to_string(), "list".to_string()]);
    }

    #[test]
    fn run_form_strips_run_and_name() {
        let a = args(&["aivo", "run", "amp", "-m", "x"]);
        let (name, rest) = resolve_invocation(&a, &no_bundles()).unwrap();
        assert_eq!(name, "amp");
        assert_eq!(rest, &["-m".to_string(), "x".to_string()]);
    }

    #[test]
    fn no_args_or_bare_run_is_not_a_plugin() {
        assert!(resolve_invocation(&args(&["aivo"]), &no_bundles()).is_none());
        assert!(resolve_invocation(&args(&["aivo", "run"]), &no_bundles()).is_none());
    }

    #[test]
    fn builtins_tools_and_flags_are_never_plugins() {
        for name in ["keys", "code", "chat", "serve", "logs", "help"] {
            assert!(
                resolve_invocation(&args(&["aivo", name]), &no_bundles()).is_none(),
                "{name} must not dispatch as a plugin"
            );
        }
        assert!(resolve_invocation(&args(&["aivo", "claude"]), &no_bundles()).is_none());
        assert!(resolve_invocation(&args(&["aivo", "run", "codex"]), &no_bundles()).is_none());
        assert!(resolve_invocation(&args(&["aivo", "--help"]), &no_bundles()).is_none());
        assert!(resolve_invocation(&args(&["aivo", "hf:owner/repo"]), &no_bundles()).is_none());
        assert!(
            resolve_invocation(&args(&["aivo", "https://example.com"]), &no_bundles()).is_none()
        );
    }

    #[test]
    fn user_bundle_wins_over_plugin() {
        let mut bundles = HashMap::new();
        bundles.insert(
            "myflow".to_string(),
            BundleAlias {
                tool: "claude".to_string(),
                args: vec![],
            },
        );
        assert!(resolve_invocation(&args(&["aivo", "myflow"]), &bundles).is_none());
    }

    #[test]
    fn infer_name_from_sources() {
        assert_eq!(infer_plugin_name("aivo-amp").as_deref(), Some("amp"));
        assert_eq!(infer_plugin_name("./bin/aivo-amp").as_deref(), Some("amp"));
        assert_eq!(
            infer_plugin_name("https://x.dev/dl/aivo-amp.exe?v=1").as_deref(),
            Some("amp")
        );
        assert_eq!(
            infer_plugin_name("/usr/local/bin/mytool").as_deref(),
            Some("mytool")
        );
        assert_eq!(infer_plugin_name(""), None);
    }

    #[test]
    fn known_plugins_have_install_sources() {
        // An uninstalled well-known plugin is offered its install source
        // rather than rejected by clap.
        for name in ["amp", "copilot", "grok", "omp"] {
            assert_eq!(
                known_plugin_source(name),
                Some(format!("github:yuanchuan/aivo-{name}")).as_deref(),
            );
            // The offer is only reachable if the name dispatches as a plugin.
            assert!(
                !is_reserved_plugin_name(name),
                "{name} must be dispatchable"
            );
        }
        // Live native tools and genuinely unknown names get no offer.
        assert_eq!(known_plugin_source("claude"), None);
        assert_eq!(known_plugin_source("foobar"), None);
    }

    #[test]
    fn reserved_names_are_rejected() {
        for n in ["keys", "code", "chat", "run", "claude", "help", "plugins"] {
            assert!(is_reserved_plugin_name(n), "{n} should be reserved");
        }
        assert!(!is_reserved_plugin_name("amp"));
    }

    #[test]
    fn debug_log_path_handling() {
        assert!(debug_log_path(&args(&["--debug"])).is_some());
        assert!(debug_log_path(&args(&["--debug="])).is_some());
        assert_eq!(
            debug_log_path(&args(&["--debug=/tmp/x.jsonl"])),
            Some(PathBuf::from("/tmp/x.jsonl"))
        );
        assert!(debug_log_path(&args(&["trust", "list"])).is_none());
    }

    #[test]
    fn extract_aivo_flags_handling() {
        // `-k <id>` + `-m <model>` are pulled out; the rest passes through.
        let f = extract_aivo_flags(&args(&["-k", "work", "-m", "gpt-4o", "-p", "hi"]));
        assert_eq!(f.key.as_deref(), Some("work"));
        assert_eq!(f.model.as_deref(), Some("gpt-4o"));
        assert_eq!(f.rest, args(&["-p", "hi"]));

        // `--key=`/`--model=` forms.
        let f = extract_aivo_flags(&args(&["--key=work", "--model=m1", "file.rs"]));
        assert_eq!(f.key.as_deref(), Some("work"));
        assert_eq!(f.model.as_deref(), Some("m1"));
        assert_eq!(f.rest, args(&["file.rs"]));

        // Bare `-k` / `--key` (no value, or a flag follows) → "" → opens the key picker.
        for a in [&["-k"][..], &["--key"][..], &["-k", "--verbose"][..]] {
            let f = extract_aivo_flags(&args(a));
            assert_eq!(f.key.as_deref(), Some(""), "{a:?}");
        }
        // `-k --verbose`: `--verbose` is not consumed as the value.
        assert_eq!(
            extract_aivo_flags(&args(&["-k", "--verbose"])).rest,
            args(&["--verbose"])
        );

        // Bare `-m` / `--model` mirrors native tools: opens the model picker.
        for a in [&["-m"][..], &["--model"][..], &["-m", "--verbose"][..]] {
            let f = extract_aivo_flags(&args(a));
            assert_eq!(f.model.as_deref(), Some(""), "{a:?}");
        }
        // `-m --verbose`: `--verbose` is not consumed as the value.
        assert_eq!(
            extract_aivo_flags(&args(&["-m", "--verbose"])).rest,
            args(&["--verbose"])
        );
        // Empty inline model is also a picker trigger, matching the key path.
        assert_eq!(
            extract_aivo_flags(&args(&["--model="])).model.as_deref(),
            Some("")
        );

        // `--debug` is aivo-owned for coding-agent plugins: it sets the log
        // path and does not leak to the wrapped tool.
        let f = extract_aivo_flags(&args(&["--debug", "-p", "hello"]));
        assert!(f.debug_log.is_some());
        assert_eq!(f.rest, args(&["-p", "hello"]));
        let f = extract_aivo_flags(&args(&["--debug=/tmp/plugin-debug.jsonl"]));
        assert_eq!(f.debug_log, Some(PathBuf::from("/tmp/plugin-debug.jsonl")));
        assert!(f.rest.is_empty());

        // `--dry-run` is aivo-owned for coding-agent plugins: it sets the flag
        // and is stripped from the argv handed to the wrapped tool.
        let f = extract_aivo_flags(&args(&["--dry-run", "-m", "gpt-4o", "-p", "hi"]));
        assert!(f.dry_run);
        assert_eq!(f.model.as_deref(), Some("gpt-4o"));
        assert_eq!(f.rest, args(&["-p", "hi"]));

        // No aivo flags → nothing extracted, everything passes through.
        let f = extract_aivo_flags(&args(&["-p", "hello", "--thinking"]));
        assert!(f.key.is_none());
        assert!(f.model.is_none());
        assert!(f.debug_log.is_none());
        assert!(!f.dry_run);
        assert_eq!(f.rest, args(&["-p", "hello", "--thinking"]));
    }

    #[test]
    fn strip_key_model_flags_handling() {
        // Separate-value and inline forms are dropped with their values.
        assert_eq!(
            strip_key_model_flags(&args(&["-k", "work", "-m", "gpt-4o", "-p", "hi"])),
            args(&["-p", "hi"])
        );
        assert_eq!(
            strip_key_model_flags(&args(&["--key=work", "--model=m1", "file.rs"])),
            args(&["file.rs"])
        );
        // A bare flag (next arg is itself a flag) drops only the flag.
        assert_eq!(
            strip_key_model_flags(&args(&["-k", "--verbose", "-m"])),
            args(&["--verbose"])
        );
        // `--debug`/`--dry-run` and everything else stay with the plugin.
        assert_eq!(
            strip_key_model_flags(&args(&["--dry-run", "-m", "x", "--debug", "HEAD~1"])),
            args(&["--dry-run", "--debug", "HEAD~1"])
        );
        // No key/model flags → argv unchanged.
        assert_eq!(
            strip_key_model_flags(&args(&["review", "/abs/path", "--all"])),
            args(&["review", "/abs/path", "--all"])
        );
    }
}
