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

/// Run the matching `aivo-<name>` plugin and return its exit code, or `None` if
/// none applies. Call before `Cli::parse_from` — clap rejects unknown subcommands.
/// Granted plugins (and `coding-agent` types) get a key/endpoint handoff and run
/// accounting via `endpoint::dispatch`; everything else spawns plain.
pub async fn try_dispatch(
    raw_args: &[String],
    bundles: &HashMap<String, BundleAlias>,
    store: &SessionStore,
) -> Option<i32> {
    let (name, plugin_args) = resolve_invocation(raw_args, bundles)?;
    let bin = discover(name)?;
    Some(endpoint::dispatch(name, &bin, plugin_args, store).await)
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
        .filter(|(_, rec)| rec.manifest.as_ref().is_some_and(endpoint::is_coding_agent))
        .map(|(name, _)| name)
        .collect()
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
    crate::services::system_env::home_dir().map(|h| h.join(".config").join("aivo").join("plugins"))
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
pub(crate) fn prompt_capability_grant(name: &str, caps: &[String]) -> bool {
    use std::io::Write;
    eprintln!(
        "  {} plugin `{}` requests {}",
        style::yellow("?"),
        name,
        caps.join(", "),
    );
    eprintln!(
        "    {}",
        style::dim(
            "granting hands it a per-launch local endpoint for your selected key — only allow plugins you trust"
        )
    );
    eprint!("  {} grant these capabilities? [y/N] ", style::yellow("?"));
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
    /// argv with aivo-owned flags removed. Tool-specific flags and prompt args
    /// are preserved.
    pub rest: Vec<String>,
}

/// Extract `-k`/`--key` (bare → picker), `-m`/`--model`, and `--debug` from a
/// coding-agent plugin's argv, stripping them from `rest`. First occurrence of
/// each wins; a value form consumes the next arg only when it isn't itself a
/// flag.
pub(crate) fn extract_aivo_flags(args: &[String]) -> AivoFlags {
    let mut key: Option<String> = None;
    let mut model: Option<String> = None;
    let mut debug_log: Option<PathBuf> = None;
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
        } else {
            rest.push(args[i].clone());
        }
        i += 1;
    }
    AivoFlags {
        key,
        model,
        debug_log,
        rest,
    }
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
        for name in ["keys", "chat", "serve", "logs", "help", "image", "audio"] {
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
    fn reserved_names_are_rejected() {
        for n in ["keys", "chat", "run", "claude", "image", "help", "plugins"] {
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

        // No aivo flags → nothing extracted, everything passes through.
        let f = extract_aivo_flags(&args(&["-p", "hello", "--thinking"]));
        assert!(f.key.is_none());
        assert!(f.model.is_none());
        assert!(f.debug_log.is_none());
        assert_eq!(f.rest, args(&["-p", "hello", "--thinking"]));
    }
}
