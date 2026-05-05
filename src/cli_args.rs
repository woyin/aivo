//! aivo-owned flag extraction and bundle expansion that runs *before* clap.
//!
//! Two responsibilities:
//! 1. **`rewrite_cli_args`**: pre-clap argv munging. Expands tool aliases
//!    (`aivo claude` → `aivo run claude`), built-in shortcuts (`aivo use`,
//!    `aivo ping`, `aivo -x`), and Bundle aliases (user's saved
//!    `aivo run <tool> <args...>` macros) so clap sees a normalized form.
//! 2. **`extract_aivo_flags`**: post-clap recovery for flags clap's
//!    `trailing_var_arg` swallowed. Also collapses `[1m]`/`[2m]` suffixes
//!    and `--1m`/`--2m` shorthands into a single `max_context` signal.
//!
//! Pure functions over `Vec<String>` and `HashMap` — no I/O.

use std::collections::{HashMap, HashSet};

use crate::constants::{KNOWN_TOOLS, RESERVED_ALIAS_NAMES};
use crate::services::environment_injector::ClaudeSlotFlags;
use crate::services::session_store::BundleAlias;

pub(crate) fn rewrite_cli_args(
    raw_args: Vec<String>,
    bundles: &HashMap<String, BundleAlias>,
) -> Vec<String> {
    if raw_args.len() <= 1 {
        return raw_args;
    }

    if KNOWN_TOOLS.contains(&raw_args[1].as_str()) {
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

    // `aivo run <bundle> [user-args...]` — expand the bundle, dropping flags
    // the user already supplied so user wins on conflicts.
    if raw_args[1] == "run"
        && raw_args.len() > 2
        && let Some(bundle) = bundles.get(&raw_args[2])
    {
        return expand_bundle(&raw_args[0], bundle, &raw_args[3..]);
    }

    // `aivo <bundle> [user-args...]` — top-level shortcut, expanded if and
    // only if the first arg doesn't collide with a built-in name. Reserved
    // names are validated at `alias add`, so a Bundle entry here is always
    // safe to expand.
    if !raw_args[1].starts_with('-')
        && !RESERVED_ALIAS_NAMES.contains(&raw_args[1].as_str())
        && let Some(bundle) = bundles.get(&raw_args[1])
    {
        return expand_bundle(&raw_args[0], bundle, &raw_args[2..]);
    }

    raw_args
}

/// Builds the final argv for a Bundle launch:
///   `<argv0> run <bundle.tool> <filtered_bundle_args...> <user_args...>`
/// `filtered_bundle_args` is the bundle's stored args minus any flag the user
/// has already typed — bundle args fill in gaps; user wins on conflicts.
fn expand_bundle(argv0: &str, bundle: &BundleAlias, user_args: &[String]) -> Vec<String> {
    let filtered = merge_bundle_with_user(&bundle.args, user_args);
    let mut out = Vec::with_capacity(3 + filtered.len() + user_args.len());
    out.push(argv0.to_string());
    out.push("run".to_string());
    out.push(bundle.tool.clone());
    out.extend(filtered);
    out.extend_from_slice(user_args);
    out
}

/// Filter a bundle's preset args against the user's typed args. Drops any
/// flag (and its value) from the bundle that the user has also supplied —
/// keyed on the *canonical* long name so `-k` cancels bundle's `--key`,
/// `--1m` cancels bundle's `--max-context`, etc.
fn merge_bundle_with_user(bundle_args: &[String], user_args: &[String]) -> Vec<String> {
    let user_flags: HashSet<String> = user_args
        .iter()
        .filter_map(|a| canonical_flag_name(a))
        .collect();

    let mut out: Vec<String> = Vec::with_capacity(bundle_args.len());
    let mut i = 0;
    while i < bundle_args.len() {
        let arg = &bundle_args[i];
        match canonical_flag_name(arg) {
            Some(name) if user_flags.contains(&name) => {
                // Skip the flag itself, plus its value when written as
                // `--flag value` (no `=`) and the next token isn't another flag.
                let consumes_value = !arg.contains('=')
                    && i + 1 < bundle_args.len()
                    && !bundle_args[i + 1].starts_with('-');
                i += if consumes_value { 2 } else { 1 };
            }
            _ => {
                out.push(arg.clone());
                i += 1;
            }
        }
    }
    out
}

/// Canonicalize a flag's name so short and long forms compare equal in
/// bundle-merging. Returns `None` for non-flag args (positional values,
/// prompt text, etc.).
fn canonical_flag_name(arg: &str) -> Option<String> {
    if !arg.starts_with('-') {
        return None;
    }
    let raw = arg.split_once('=').map(|(f, _)| f).unwrap_or(arg);
    let canon = match raw {
        "--1m" | "--2m" => "--max-context",
        "-k" => "--key",
        "-m" => "--model",
        "-r" => "--refresh",
        "-e" => "--env",
        "-c" => "--context",
        s => s,
    };
    Some(canon.to_string())
}

/// Returns true if `raw_args[1]` could plausibly be a Bundle alias name —
/// i.e. it's worth paying for a config read to find out. False for built-in
/// commands, tool shortcuts, and bare flags.
pub(crate) fn needs_bundle_lookup(raw_args: &[String]) -> bool {
    let Some(arg1) = raw_args.get(1) else {
        return false;
    };
    if arg1.starts_with('-') {
        return false;
    }
    if arg1 == "run" {
        return true;
    }
    !RESERVED_ALIAS_NAMES.contains(&arg1.as_str())
}

/// Like `resolve_model_alias` but resolves against a pre-loaded alias map so
/// callers with many lookups (the run command resolves up to 7 model fields)
/// don't pay one disk read per call. Falls back to the input on any error.
pub(crate) fn resolve_alias_in_memory(
    aliases: &HashMap<String, String>,
    model: Option<String>,
) -> Option<String> {
    let m = match model {
        Some(ref m) if !m.is_empty() => m,
        other => return other,
    };
    let mut current = m.to_string();
    let mut seen = HashSet::new();
    while let Some(target) = aliases.get(&current) {
        if !seen.insert(current.clone()) {
            return model; // cycle — return the original input
        }
        current = target.clone();
    }
    Some(current)
}

/// Result of extracting aivo-specific flags from clap's trailing passthrough args.
pub(crate) struct ExtractedFlags {
    pub(crate) model: Option<String>,
    pub(crate) slots: ClaudeSlotFlags,
    pub(crate) key_flag: Option<String>,
    /// `None` = flag absent. `Some("")` = bare `--debug` (default path).
    /// `Some("/path/to.jsonl")` = explicit log path.
    pub(crate) debug: Option<String>,
    pub(crate) dry_run: bool,
    pub(crate) refresh: bool,
    pub(crate) relogin: bool,
    pub(crate) env_strings: Vec<String>,
    pub(crate) remaining_args: Vec<String>,
    /// `None` = flag absent. `Some("")` = bare flag (interactive picker).
    /// `Some("id")` = explicit session id prefix.
    pub(crate) context: Option<String>,
    /// `None` = flag absent. `Some("1m")` = activate the 1M-context spoof.
    pub(crate) max_context: Option<String>,
}

/// Strip a trailing `[1m]` or `[2m]` from the model name and lift it into
/// `max_context`, so `-m foo[1m]`, `-m foo --1m`, and `-m foo --max-context=1m`
/// all collapse into the same internal state (and likewise for 2m). Without
/// this, mixing the suffix and the flag would produce a double `[Nm][Nm]`
/// after fan-out. `max_context` is left alone if it was already set —
/// validation downstream rejects mismatches.
pub(crate) fn lift_context_suffix(
    model: Option<String>,
    max_context: Option<String>,
) -> (Option<String>, Option<String>) {
    let Some(m) = model else {
        return (None, max_context);
    };
    for tag in ["1m", "2m"] {
        let suffix_with_brackets = ["[", tag, "]"].concat();
        if let Some(stripped) = m.strip_suffix(&suffix_with_brackets) {
            let new_max_context = max_context.or_else(|| Some(tag.to_string()));
            return (Some(stripped.to_string()), new_max_context);
        }
    }
    (Some(m), max_context)
}

/// Extracts aivo-owned flags (`--model`/`-m`, `--key`/`-k`, `--debug`, `--dry-run`, `--refresh`/`-r`, `--env`/`-e`) from
/// the passthrough `args` slice that clap's `trailing_var_arg` may have swallowed.
///
/// Flags already parsed by clap are supplied via `initial_*` parameters so that the
/// function produces a single consistent view regardless of where clap stopped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_aivo_flags(
    initial_model: Option<String>,
    initial_slots: ClaudeSlotFlags,
    initial_key: Option<String>,
    initial_debug: Option<String>,
    initial_dry_run: bool,
    initial_refresh: bool,
    initial_relogin: bool,
    initial_envs: Vec<String>,
    initial_max_context: Option<String>,
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
    let mut relogin = initial_relogin;
    let mut context: Option<String> = None;
    let mut max_context: Option<String> = initial_max_context;
    let mut env_strings = initial_envs;
    let ClaudeSlotFlags {
        reasoning: mut reasoning_model,
        subagent: mut subagent_model,
        haiku: mut haiku_model,
        sonnet: mut sonnet_model,
        opus: mut opus_model,
    } = initial_slots;
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
    // Same protection for the per-slot Claude flags: a model name never starts
    // with `-`, so if clap handed us one, the user mistyped a flag (e.g.
    // `--haiku-model --opus-model X`). Push it back to passthrough and treat
    // the slot as bare so the next pass can re-parse it as the intended flag.
    let mut sanitize_slot = |slot: &mut Option<String>| {
        if let Some(ref v) = *slot
            && v.starts_with('-')
        {
            remaining_args.push(v.clone());
            *slot = Some(String::new());
        }
    };
    sanitize_slot(&mut reasoning_model);
    sanitize_slot(&mut subagent_model);
    sanitize_slot(&mut haiku_model);
    sanitize_slot(&mut sonnet_model);
    sanitize_slot(&mut opus_model);

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
            // Bare passthrough --debug → default path (empty string sentinel).
            if debug.is_none() {
                debug = Some(String::new());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if let Some(value) = arg.strip_prefix("--debug=") {
            if debug.is_none() {
                debug = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--dry-run" {
            dry_run = true;
        } else if arg == "--refresh" || arg == "-r" {
            refresh = true;
        } else if arg == "--relogin" {
            relogin = true;
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
        } else if let Some(value) = arg.strip_prefix("--reasoning-model=") {
            if !value.is_empty() && reasoning_model.is_none() {
                reasoning_model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--reasoning-model" && reasoning_model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                reasoning_model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                reasoning_model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--subagent-model=") {
            if !value.is_empty() && subagent_model.is_none() {
                subagent_model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--subagent-model" && subagent_model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                subagent_model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                subagent_model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--haiku-model=") {
            if !value.is_empty() && haiku_model.is_none() {
                haiku_model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--haiku-model" && haiku_model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                haiku_model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                haiku_model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--sonnet-model=") {
            if !value.is_empty() && sonnet_model.is_none() {
                sonnet_model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--sonnet-model" && sonnet_model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                sonnet_model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                sonnet_model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--opus-model=") {
            if !value.is_empty() && opus_model.is_none() {
                opus_model = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--opus-model" && opus_model.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                opus_model = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                opus_model = Some(String::new());
            }
        } else if let Some(value) = arg.strip_prefix("--max-context=") {
            if !value.is_empty() && max_context.is_none() {
                max_context = Some(value.to_string());
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--max-context" && max_context.is_none() {
            if i + 1 < passthrough_args.len() && !passthrough_args[i + 1].starts_with('-') {
                max_context = Some(passthrough_args[i + 1].clone());
                i += 1;
            } else {
                remaining_args.push(arg.clone());
            }
        } else if arg == "--1m" {
            if max_context.is_none() {
                max_context = Some("1m".to_string());
            }
        } else if arg == "--2m" {
            if max_context.is_none() {
                max_context = Some("2m".to_string());
            }
        } else {
            remaining_args.push(arg.clone());
        }
        i += 1;
    }

    ExtractedFlags {
        model,
        slots: ClaudeSlotFlags {
            reasoning: reasoning_model,
            subagent: subagent_model,
            haiku: haiku_model,
            sonnet: sonnet_model,
            opus: opus_model,
        },
        key_flag,
        debug,
        dry_run,
        refresh,
        relogin,
        env_strings,
        remaining_args,
        context,
        max_context,
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
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--model=gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_space_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--model", "gpt-4o", "file.ts"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn model_short_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["-m", "gpt-4o"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn model_no_value_triggers_picker() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--model"]),
        );
        assert_eq!(r.model, Some(String::new()));
    }

    #[test]
    fn model_flag_as_value_corrected() {
        // Clap swallowed `--resume` as the value of -m
        let r = extract_aivo_flags(
            Some("--resume".to_string()),
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
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
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--model", "other"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["--model", "other"]));
    }

    #[test]
    fn key_inline_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--key=mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_space_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--key", "mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_short_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["-k", "mykey"]),
        );
        assert_eq!(r.key_flag, Some("mykey".to_string()));
    }

    #[test]
    fn key_flag_as_value_corrected() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            Some("--something".to_string()),
            None,
            false,
            false,
            false,
            vec![],
            None,
            &[],
        );
        assert_eq!(r.key_flag, Some(String::new()));
        assert_eq!(r.remaining_args, args(&["--something"]));
    }

    #[test]
    fn key_no_value_triggers_picker() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["-k"]),
        );
        assert_eq!(r.key_flag, Some(String::new()));
    }

    #[test]
    fn debug_flag() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--debug", "file.ts"]),
        );
        assert_eq!(r.debug, Some(String::new()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn debug_already_set_preserved() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            Some(String::new()),
            false,
            false,
            false,
            vec![],
            None,
            &[],
        );
        assert_eq!(r.debug, Some(String::new()));
    }

    #[test]
    fn dry_run_flag() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--dry-run"]),
        );
        assert!(r.dry_run);
    }

    #[test]
    fn env_inline_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--env=FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_inline_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["-e=FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_space_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--env", "FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn env_short_space_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["-e", "FOO=bar"]),
        );
        assert_eq!(r.env_strings, vec!["FOO=bar"]);
    }

    #[test]
    fn relogin_flag_in_passthrough_position() {
        // `aivo run codex --relogin` puts --relogin after the tool name, so
        // clap's trailing_var_arg captures it into passthrough. This must
        // round-trip into ExtractedFlags::relogin.
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--relogin"]),
        );
        assert!(r.relogin);
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn relogin_flag_carried_through_when_set_by_clap() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            true, // initial_relogin from clap
            vec![],
            None,
            &args(&[]),
        );
        assert!(r.relogin);
    }

    #[test]
    fn initial_envs_preserved() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec!["PRE=1".to_string()],
            None,
            &args(&["-e", "POST=2"]),
        );
        assert_eq!(r.env_strings, vec!["PRE=1", "POST=2"]);
    }

    #[test]
    fn max_context_inline_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--max-context=1m", "file.ts"]),
        );
        assert_eq!(r.max_context, Some("1m".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn max_context_space_form() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--max-context", "1m"]),
        );
        assert_eq!(r.max_context, Some("1m".to_string()));
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn max_context_initial_preserved() {
        // clap parsed --max-context up front; passthrough stays as-is.
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            Some("1m".to_string()),
            &args(&["file.ts"]),
        );
        assert_eq!(r.max_context, Some("1m".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn lift_context_suffix_strips_and_sets_max_context() {
        let (m, mc) = lift_context_suffix(Some("deepseek[1m]".to_string()), None);
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, Some("1m".to_string()));

        let (m, mc) = lift_context_suffix(Some("deepseek[2m]".to_string()), None);
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, Some("2m".to_string()));
    }

    #[test]
    fn lift_context_suffix_strips_when_max_context_already_set() {
        // Both `-m X[1m]` and `--1m` together: strip the redundant suffix so
        // the env injector doesn't double-append.
        let (m, mc) = lift_context_suffix(Some("deepseek[1m]".to_string()), Some("1m".to_string()));
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, Some("1m".to_string()));
    }

    #[test]
    fn lift_context_suffix_passes_through_when_no_suffix() {
        let (m, mc) = lift_context_suffix(Some("deepseek".to_string()), None);
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, None);
    }

    #[test]
    fn lift_context_suffix_handles_no_model() {
        let (m, mc) = lift_context_suffix(None, Some("1m".to_string()));
        assert_eq!(m, None);
        assert_eq!(mc, Some("1m".to_string()));
    }

    #[test]
    fn one_m_shorthand_resolves_to_max_context() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--1m", "file.ts"]),
        );
        assert_eq!(r.max_context, Some("1m".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn two_m_shorthand_resolves_to_max_context() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--2m", "file.ts"]),
        );
        assert_eq!(r.max_context, Some("2m".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn unknown_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--agent-name", "foo", "--resume"]),
        );
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "--resume"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn mixed_flags() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
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
        assert_eq!(r.debug, Some(String::new()));
        assert_eq!(r.remaining_args, args(&["--agent-name", "foo", "file.ts"]));
    }

    fn no_bundles() -> HashMap<String, BundleAlias> {
        HashMap::new()
    }

    fn one_bundle(name: &str, tool: &str, args: &[&str]) -> HashMap<String, BundleAlias> {
        let mut m = HashMap::new();
        m.insert(
            name.to_string(),
            BundleAlias {
                tool: tool.to_string(),
                args: args.iter().map(|s| s.to_string()).collect(),
            },
        );
        m
    }

    #[test]
    fn rewrite_injects_chat_for_top_level_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-x", "hello"]), &no_bundles()),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_chat_for_long_execute() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "--execute", "hello"]), &no_bundles()),
            args(&["aivo", "chat", "--execute", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_explicit_chat() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "chat", "-x", "hello"]), &no_bundles()),
            args(&["aivo", "chat", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_keeps_tool_alias_precedence() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "claude", "--model", "gpt-5"]), &no_bundles()),
            args(&["aivo", "run", "claude", "--model", "gpt-5"])
        );
    }

    #[test]
    fn rewrite_use_shortcut() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "use", "work"]), &no_bundles()),
            args(&["aivo", "keys", "use", "work"])
        );
    }

    #[test]
    fn rewrite_ping_shortcut() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "ping"]), &no_bundles()),
            args(&["aivo", "keys", "ping"])
        );
    }

    #[test]
    fn bundle_top_level_expands() {
        let bundles = one_bundle("quick", "claude", &["--key", "work", "--model", "fast"]);
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "quick"]), &bundles),
            args(&["aivo", "run", "claude", "--key", "work", "--model", "fast"])
        );
    }

    #[test]
    fn bundle_via_run_expands() {
        let bundles = one_bundle("quick", "claude", &["--key", "work", "--model", "fast"]);
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "run", "quick"]), &bundles),
            args(&["aivo", "run", "claude", "--key", "work", "--model", "fast"])
        );
    }

    #[test]
    fn bundle_user_flag_overrides_bundle() {
        // User's --model wins; bundle's --key still applies.
        let bundles = one_bundle(
            "quick",
            "claude",
            &["--key", "work", "--model", "fast", "--max-context", "1m"],
        );
        let result = rewrite_cli_args(args(&["aivo", "quick", "--model", "other"]), &bundles);
        // Expected: bundle args minus --model fast, then user args
        assert_eq!(
            result,
            args(&[
                "aivo",
                "run",
                "claude",
                "--key",
                "work",
                "--max-context",
                "1m",
                "--model",
                "other"
            ])
        );
    }

    #[test]
    fn bundle_short_flag_cancels_long_in_bundle() {
        let bundles = one_bundle("quick", "claude", &["--key", "work", "--model", "fast"]);
        let result = rewrite_cli_args(args(&["aivo", "quick", "-k", "play"]), &bundles);
        // Bundle's --key was canceled by user's -k
        assert_eq!(
            result,
            args(&["aivo", "run", "claude", "--model", "fast", "-k", "play"])
        );
    }

    #[test]
    fn bundle_one_m_cancels_max_context() {
        let bundles = one_bundle("quick", "claude", &["--max-context", "1m"]);
        let result = rewrite_cli_args(args(&["aivo", "quick", "--1m"]), &bundles);
        assert_eq!(result, args(&["aivo", "run", "claude", "--1m"]));
    }

    #[test]
    fn bundle_inline_flag_cancels_bundle_flag() {
        let bundles = one_bundle("quick", "claude", &["--model", "fast"]);
        let result = rewrite_cli_args(args(&["aivo", "quick", "--model=other"]), &bundles);
        assert_eq!(result, args(&["aivo", "run", "claude", "--model=other"]));
    }

    #[test]
    fn bundle_passes_through_positional_args() {
        let bundles = one_bundle("quick", "claude", &["--key", "work"]);
        let result = rewrite_cli_args(args(&["aivo", "quick", "fix", "the", "bug"]), &bundles);
        assert_eq!(
            result,
            args(&[
                "aivo", "run", "claude", "--key", "work", "fix", "the", "bug"
            ])
        );
    }

    #[test]
    fn bundle_keeps_value_with_separate_flag() {
        // `--debug` in bundle (no value); user passes `--debug=path`.
        // Since user's `--debug=path` exists, bundle's `--debug` is filtered.
        let bundles = one_bundle("dbg", "claude", &["--debug"]);
        let result = rewrite_cli_args(args(&["aivo", "dbg", "--debug=/tmp/x"]), &bundles);
        assert_eq!(result, args(&["aivo", "run", "claude", "--debug=/tmp/x"]));
    }

    #[test]
    fn bundle_named_after_reserved_does_not_expand_top_level() {
        // Even if a Bundle entry exists for "claude" (shouldn't happen — alias
        // creation rejects it — but we're defending against tampering), the
        // top-level path *must* keep the existing tool-alias rewrite. Reserved
        // names are filtered out before the bundle lookup.
        let bundles = one_bundle("claude", "codex", &["--key", "x"]);
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "claude"]), &bundles),
            args(&["aivo", "run", "claude"])
        );
    }

    #[test]
    fn canonical_flag_name_normalizes_short_and_alt_forms() {
        assert_eq!(canonical_flag_name("-k"), Some("--key".to_string()));
        assert_eq!(canonical_flag_name("--key"), Some("--key".to_string()));
        assert_eq!(canonical_flag_name("--key=foo"), Some("--key".to_string()));
        assert_eq!(
            canonical_flag_name("--1m"),
            Some("--max-context".to_string())
        );
        assert_eq!(
            canonical_flag_name("--2m"),
            Some("--max-context".to_string())
        );
        assert_eq!(
            canonical_flag_name("--unknown"),
            Some("--unknown".to_string())
        );
        assert_eq!(canonical_flag_name("positional"), None);
    }

    #[test]
    fn prompt_passes_through_extraction() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["fix the login bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
        assert_eq!(r.model, None);
    }

    #[test]
    fn prompt_preserved_with_model_flag() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["--model", "gpt-4o", "fix the login bug"]),
        );
        assert_eq!(r.model, Some("gpt-4o".to_string()));
        assert_eq!(r.remaining_args, args(&["fix the login bug"]));
    }

    #[test]
    fn multi_word_unquoted_args_pass_through() {
        let r = extract_aivo_flags(
            None,
            ClaudeSlotFlags::default(),
            None,
            None,
            false,
            false,
            false,
            vec![],
            None,
            &args(&["fix", "the", "bug"]),
        );
        assert_eq!(r.remaining_args, args(&["fix", "the", "bug"]));
    }

    #[test]
    fn alias_resolution_detects_cycles() {
        let aliases = HashMap::from([
            ("fast".to_string(), "cheap".to_string()),
            ("cheap".to_string(), "fast".to_string()),
        ]);
        assert_eq!(
            resolve_alias_in_memory(&aliases, Some("fast".to_string())),
            Some("fast".to_string())
        );
    }

    #[test]
    fn alias_resolution_follows_chain() {
        let aliases = HashMap::from([
            ("fast".to_string(), "mid".to_string()),
            ("mid".to_string(), "gpt-4o".to_string()),
        ]);
        assert_eq!(
            resolve_alias_in_memory(&aliases, Some("fast".to_string())),
            Some("gpt-4o".to_string())
        );
    }

    #[test]
    fn alias_resolution_passes_through_unknown() {
        let aliases = HashMap::new();
        assert_eq!(
            resolve_alias_in_memory(&aliases, Some("gpt-4o".to_string())),
            Some("gpt-4o".to_string())
        );
        assert_eq!(resolve_alias_in_memory(&aliases, None), None);
        assert_eq!(
            resolve_alias_in_memory(&aliases, Some(String::new())),
            Some(String::new())
        );
    }

    #[test]
    fn needs_bundle_lookup_rejects_short_argv() {
        assert!(!needs_bundle_lookup(&args(&["aivo"])));
        assert!(!needs_bundle_lookup(&[]));
    }

    #[test]
    fn needs_bundle_lookup_rejects_flags() {
        assert!(!needs_bundle_lookup(&args(&["aivo", "-h"])));
        assert!(!needs_bundle_lookup(&args(&["aivo", "--version"])));
        assert!(!needs_bundle_lookup(&args(&["aivo", "-x", "hello"])));
    }

    #[test]
    fn needs_bundle_lookup_rejects_reserved_subcommands() {
        // `claude`, `keys`, `chat`, etc. are reserved — never trigger the disk read.
        assert!(!needs_bundle_lookup(&args(&["aivo", "claude"])));
        assert!(!needs_bundle_lookup(&args(&["aivo", "keys"])));
        assert!(!needs_bundle_lookup(&args(&["aivo", "chat"])));
    }

    #[test]
    fn needs_bundle_lookup_accepts_run_form() {
        // `aivo run <bundle>` always needs the lookup, even if the bundle name
        // happens to clash with a reserved subcommand later in argv.
        assert!(needs_bundle_lookup(&args(&["aivo", "run"])));
        assert!(needs_bundle_lookup(&args(&["aivo", "run", "quick"])));
    }

    #[test]
    fn needs_bundle_lookup_accepts_unknown_first_word() {
        // Anything that *could* be a user-defined bundle name triggers the lookup.
        assert!(needs_bundle_lookup(&args(&["aivo", "quick"])));
        assert!(needs_bundle_lookup(&args(&["aivo", "myproject"])));
    }
}
