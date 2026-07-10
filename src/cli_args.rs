//! aivo-owned flag extraction and bundle expansion that runs *before* clap.
//!
//! Two responsibilities:
//! 1. **`rewrite_cli_args`**: pre-clap argv munging. Expands tool aliases
//!    (`aivo claude` → `aivo run claude`), built-in shortcuts (`aivo use`,
//!    `aivo ping`, `aivo -x`), and Bundle aliases (user's saved
//!    `aivo run <tool> <args...>` macros) so clap sees a normalized form.
//! 2. **`extract_aivo_flags`**: post-clap recovery for flags clap's
//!    `trailing_var_arg` swallowed. Also collapses `[<N>m]` suffixes and
//!    `--<N>m` shorthands (any digits) into a single `max_context` signal.
//!    Validation that <N> is actually supported (1m/2m today) lives downstream.
//!
//! Pure functions over `Vec<String>` and `HashMap` — no I/O.

use std::collections::{HashMap, HashSet};

use crate::constants::{KNOWN_TOOLS, RESERVED_ALIAS_NAMES};
use crate::services::environment_injector::ClaudeSlotFlags;
use crate::services::huggingface::is_hf_or_local_gguf;
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

    // `aivo code <mcp|skills> …` → the hidden clap command named "code mcp"/"code skills".
    if raw_args[1] == "code"
        && let Some(sub @ ("mcp" | "skills" | "packs")) = raw_args.get(2).map(String::as_str)
    {
        let mut rewritten = vec![raw_args[0].clone(), format!("code {sub}")];
        rewritten.extend_from_slice(&raw_args[3..]);
        return rewritten;
    }

    if matches!(raw_args[1].as_str(), "-p" | "--prompt" | "-x" | "--execute") {
        let mut rewritten = vec![raw_args[0].clone(), "code".to_string()];
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

    // Bare-prompt shortcut. After tool/subcommand/bundle/plugin matches have
    // failed, a top-level non-flag arg is interpreted as input to `code`:
    //   `aivo hf:Qwen/...` / `aivo https://...`      → `aivo code <ref>`
    //     (code's positional REF; opens TUI with that model)
    //   `aivo "tell me a story"` / `aivo 你好` / `aivo hi?` → `aivo code -p <text>`
    //     (one-shot prompt; trailing args pass through to code)
    // A bare `[a-z0-9-]` word is never a prompt: it falls through to clap's
    // "unrecognized subcommand" with did-you-mean (this also keeps reserved
    // names and clap's built-in `help` reachable — all are shaped). The shell
    // strips quotes, so `aivo "hello"` is indistinguishable from `aivo hello`
    // and gets the same treatment; use `-p` to force a one-word prompt.
    // Embedded whitespace (only producible via quoting), uppercase,
    // punctuation, or non-ASCII marks a prompt.
    let first = raw_args[1].as_str();
    if first.starts_with('-') || is_subcommand_shaped(first) {
        return raw_args;
    }
    if first.starts_with("hf:") || first.starts_with("http://") || first.starts_with("https://") {
        let mut rewritten = vec![raw_args[0].clone(), "code".to_string()];
        rewritten.extend_from_slice(&raw_args[1..]);
        return rewritten;
    }
    let mut rewritten = vec![raw_args[0].clone(), "code".to_string(), "-p".to_string()];
    rewritten.extend_from_slice(&raw_args[1..]);
    rewritten
}

/// True when `s` looks like a subcommand name — only `[a-z0-9-]` chars.
/// Gates the bare-prompt rewrite: anything with whitespace, uppercase,
/// punctuation, or non-ASCII can't be a command name and is prompt-shaped.
pub(crate) fn is_subcommand_shaped(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
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
    if parse_context_shorthand(raw).is_some() {
        return Some("--max-context".to_string());
    }
    let canon = match raw {
        "-k" => "--key",
        "-m" => "--model",
        "-r" => "--refresh",
        "-e" => "--env",
        "-c" => "--context",
        s => s,
    };
    Some(canon.to_string())
}

/// Recognize `--<digits>m` / `--<digits>M` as a `--max-context` shorthand.
/// Returns the canonical (lowercased) `<digits>m` value if matched.
/// The set of *supported* values (e.g. only `1m`/`2m`) is enforced downstream
/// in `run.rs` — this helper is just about parser shape.
fn parse_context_shorthand(arg: &str) -> Option<String> {
    parse_context_token(arg.strip_prefix("--")?)
}

/// `<digits>m` or `<digits>M` → `Some("<digits>m")`. Anything else → `None`.
pub(crate) fn parse_context_token(tok: &str) -> Option<String> {
    let last = tok.chars().last()?;
    if last != 'm' && last != 'M' {
        return None;
    }
    let digits = &tok[..tok.len() - 1];
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("{digits}m"))
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
    pub(crate) transform: bool,
    /// Pi only: `--transparent` opts out of the default transform router.
    pub(crate) transparent: bool,
}

/// Strip a trailing `[<digits>m]` from the model name and lift it into
/// `max_context`, so `-m foo[1m]`, `-m foo --1m`, and `-m foo --max-context=1m`
/// all collapse into the same internal state. Without this, mixing the suffix
/// and the flag would produce a double `[Nm][Nm]` after fan-out. `max_context`
/// is left alone if it was already set — validation downstream rejects
/// mismatches and unsupported sizes.
pub(crate) fn lift_context_suffix(
    model: Option<String>,
    max_context: Option<String>,
) -> (Option<String>, Option<String>) {
    let Some(m) = model else {
        return (None, max_context);
    };
    let s_no_close = match m.strip_suffix(']') {
        Some(s) => s,
        None => return (Some(m), max_context),
    };
    let Some(bracket_idx) = s_no_close.rfind('[') else {
        return (Some(m), max_context);
    };
    let inner = &s_no_close[bracket_idx + 1..];
    let Some(tag) = parse_context_token(inner) else {
        return (Some(m), max_context);
    };
    let stripped = m[..bracket_idx].to_string();
    let new_max_context = max_context.or(Some(tag));
    (Some(stripped), new_max_context)
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
    let mut transform = false;
    let mut transparent = false;
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
        } else if let Some(value) = parse_context_shorthand(arg) {
            if max_context.is_none() {
                max_context = Some(value);
            }
        } else if arg == "--transform" {
            transform = true;
        } else if arg == "--transparent" {
            transparent = true;
        } else if model.is_none() && is_hf_or_local_gguf(arg) {
            // Lift positional `hf:`/URL/local-path into `-m`. Explicit `-m` wins.
            model = Some(arg.clone());
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
        transform,
        transparent,
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
    fn transform_flags_parsed_not_passed_through() {
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
            &args(&["--transform", "--transparent"]),
        );
        assert!(r.transform);
        assert!(r.transparent);
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn positional_hf_ref_lifts_to_model() {
        // `aivo codex hf:Qwen/...` should set model and not pass the
        // ref through to the underlying tool.
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
            &args(&["hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"]),
        );
        assert_eq!(
            r.model.as_deref(),
            Some("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        );
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn positional_hf_url_lifts_to_model() {
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
            &args(&["https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"]),
        );
        assert_eq!(
            r.model.as_deref(),
            Some("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        );
        assert!(r.remaining_args.is_empty());
    }

    #[test]
    fn explicit_model_wins_over_positional_hf_ref() {
        // Explicit -m precedes; the HF positional then passes through
        // to the tool unchanged (e.g. as a prompt fragment).
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
            &args(&["hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"]),
        );
        assert_eq!(r.model.as_deref(), Some("gpt-4o"));
        assert_eq!(
            r.remaining_args,
            args(&["hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"])
        );
    }

    #[test]
    fn positional_hf_ref_with_trailing_prompt() {
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
            &args(&["hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF", "tell me about rust"]),
        );
        assert_eq!(
            r.model.as_deref(),
            Some("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")
        );
        assert_eq!(r.remaining_args, args(&["tell me about rust"]));
    }

    #[test]
    fn non_hf_positional_passes_through() {
        // Anything that isn't an HF ref keeps its normal passthrough behavior.
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
            &args(&["file.ts", "do the thing"]),
        );
        assert_eq!(r.model, None);
        assert_eq!(r.remaining_args, args(&["file.ts", "do the thing"]));
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
    fn dynamic_m_shorthand_resolves_to_max_context() {
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
            &args(&["--12m", "file.ts"]),
        );
        assert_eq!(r.max_context, Some("12m".to_string()));
        assert_eq!(r.remaining_args, args(&["file.ts"]));
    }

    #[test]
    fn uppercase_m_shorthand_normalizes_to_lowercase() {
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
            &args(&["--3M"]),
        );
        assert_eq!(r.max_context, Some("3m".to_string()));
    }

    #[test]
    fn non_digit_dash_dash_passes_through() {
        // `--foo` and `--ma` (looks shorthand-ish but isn't) should not be
        // captured as max_context; they fall through to remaining_args.
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
            &args(&["--foo", "--ma", "--m", "--1mb"]),
        );
        assert_eq!(r.max_context, None);
        assert_eq!(r.remaining_args, args(&["--foo", "--ma", "--m", "--1mb"]));
    }

    #[test]
    fn lift_context_suffix_handles_dynamic_sizes() {
        let (m, mc) = lift_context_suffix(Some("deepseek[12m]".to_string()), None);
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, Some("12m".to_string()));

        let (m, mc) = lift_context_suffix(Some("deepseek[3M]".to_string()), None);
        assert_eq!(m, Some("deepseek".to_string()));
        assert_eq!(mc, Some("3m".to_string()));
    }

    #[test]
    fn lift_context_suffix_ignores_non_context_brackets() {
        let (m, mc) = lift_context_suffix(Some("model[v2]".to_string()), None);
        assert_eq!(m, Some("model[v2]".to_string()));
        assert_eq!(mc, None);
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
    fn rewrite_injects_code_for_top_level_prompt() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-p", "hello"]), &no_bundles()),
            args(&["aivo", "code", "-p", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_code_for_long_prompt() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "--prompt", "hello"]), &no_bundles()),
            args(&["aivo", "code", "--prompt", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_code_for_legacy_x_alias() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-x", "hello"]), &no_bundles()),
            args(&["aivo", "code", "-x", "hello"])
        );
    }

    #[test]
    fn rewrite_injects_code_for_legacy_execute_alias() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "--execute", "hello"]), &no_bundles()),
            args(&["aivo", "code", "--execute", "hello"])
        );
    }

    #[test]
    fn rewrite_treats_multiword_top_level_arg_as_prompt() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "hello world"]), &no_bundles()),
            args(&["aivo", "code", "-p", "hello world"])
        );
    }

    #[test]
    fn rewrite_multiword_prompt_preserves_trailing_flags() {
        // `aivo "hi" --model gpt-4o` should let --model pass through to chat.
        assert_eq!(
            rewrite_cli_args(
                args(&["aivo", "hi there", "--model", "gpt-4o"]),
                &no_bundles()
            ),
            args(&["aivo", "code", "-p", "hi there", "--model", "gpt-4o"])
        );
    }

    #[test]
    fn rewrite_treats_hf_ref_as_code_positional() {
        assert_eq!(
            rewrite_cli_args(
                args(&["aivo", "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"]),
                &no_bundles()
            ),
            args(&["aivo", "code", "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"])
        );
    }

    #[test]
    fn rewrite_treats_http_url_as_code_positional() {
        assert_eq!(
            rewrite_cli_args(
                args(&["aivo", "https://huggingface.co/foo/bar"]),
                &no_bundles()
            ),
            args(&["aivo", "code", "https://huggingface.co/foo/bar"])
        );
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "http://example.com/m"]), &no_bundles()),
            args(&["aivo", "code", "http://example.com/m"])
        );
    }

    #[test]
    fn rewrite_bare_word_falls_through_to_clap() {
        // A bare `[a-z0-9-]` word is never a prompt — short tokens, typos
        // (`chta`), and real words (`hello`, `what`) all reach clap's
        // "unrecognized subcommand" with did-you-mean. The shell collapses
        // `aivo "hello"` to the same argv, so the quoted form necessarily
        // behaves identically; `-p` forces a one-word prompt.
        for s in [
            "a", "ab", "hi", "yo", "hello", "what", "runs", "chta", "kyes", "claud", "logz",
            "modls", "gpt-4o",
        ] {
            assert_eq!(
                rewrite_cli_args(args(&["aivo", s]), &no_bundles()),
                args(&["aivo", s]),
                "expected `aivo {s}` to fall through to clap",
            );
        }
    }

    #[test]
    fn rewrite_leaves_reserved_subcommand_unchanged() {
        // `aivo code`, `aivo keys`, etc. must reach clap as real subcommands,
        // not get rewritten to `chat -p chat`. All reserved names are
        // `[a-z0-9-]`, so the bare-word gate covers them.
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "chat"]), &no_bundles()),
            args(&["aivo", "chat"])
        );
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "models"]), &no_bundles()),
            args(&["aivo", "models"])
        );
    }

    #[test]
    fn rewrite_leaves_help_subcommand_unchanged() {
        // `help` is clap-generated and not in RESERVED_ALIAS_NAMES; the
        // bare-word gate keeps it reachable.
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "help"]), &no_bundles()),
            args(&["aivo", "help"])
        );
    }

    #[test]
    fn rewrite_bare_word_with_trailing_args_still_falls_through() {
        // First-token check: even with more args, a bare first word gets
        // rejected — unquoted multi-word prompts must be quoted or use `-p`.
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "a", "b", "c"]), &no_bundles()),
            args(&["aivo", "a", "b", "c"]),
        );
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "tell", "me", "a", "story"]), &no_bundles()),
            args(&["aivo", "tell", "me", "a", "story"]),
        );
    }

    #[test]
    fn rewrite_bare_word_with_explicit_p_still_works() {
        // `-p` is parsed before the bare-prompt branch — the bare-word gate
        // never fires when the user is explicit about intent.
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "-p", "a"]), &no_bundles()),
            args(&["aivo", "code", "-p", "a"]),
        );
    }

    #[test]
    fn rewrite_non_ascii_input_is_a_prompt() {
        // Chinese/Japanese/Korean/emoji/Cyrillic prompts contain no spaces
        // even as full sentences. They bypass the bare-word gate via
        // `is_subcommand_shaped` (which requires `[a-z0-9-]`-only).
        for s in ["你", "你好", "こんにちは", "안녕", "привет", "🎉", "1你"] {
            assert_eq!(
                rewrite_cli_args(args(&["aivo", s]), &no_bundles()),
                args(&["aivo", "code", "-p", s]),
                "expected `aivo {s}` to be treated as a prompt",
            );
        }
    }

    #[test]
    fn rewrite_short_bundle_alias_still_expands() {
        // `aivo a` where `a` is a registered Bundle alias must still
        // expand — the bundle lookup runs before the bare-word gate, so
        // registered aliases win.
        let bundles = one_bundle("a", "claude", &["--key", "work"]);
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "a"]), &bundles),
            args(&["aivo", "run", "claude", "--key", "work"]),
        );
    }

    #[test]
    fn rewrite_punctuated_or_uppercase_input_is_a_prompt() {
        // Anything that isn't `[a-z0-9-]` can't be a command name —
        // punctuation/uppercase marks a prompt, even one word.
        for s in ["hi?", "Hello", "Hi", "1+1"] {
            assert_eq!(
                rewrite_cli_args(args(&["aivo", s]), &no_bundles()),
                args(&["aivo", "code", "-p", s]),
                "expected `aivo {s}` to be treated as a prompt",
            );
        }
    }

    #[test]
    fn rewrite_keeps_explicit_code() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code", "-x", "hello"]), &no_bundles()),
            args(&["aivo", "code", "-x", "hello"])
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
    fn rewrite_code_mcp_to_mcp_command() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code", "mcp"]), &no_bundles()),
            args(&["aivo", "code mcp"])
        );
        assert_eq!(
            rewrite_cli_args(
                args(&["aivo", "code", "mcp", "add", "-p", "npx", "-y", "srv"]),
                &no_bundles()
            ),
            args(&["aivo", "code mcp", "add", "-p", "npx", "-y", "srv"])
        );
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code", "mcp", "--help"]), &no_bundles()),
            args(&["aivo", "code mcp", "--help"])
        );
    }

    #[test]
    fn rewrite_code_skills_to_skills_command() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code", "skills"]), &no_bundles()),
            args(&["aivo", "code skills"])
        );
        assert_eq!(
            rewrite_cli_args(
                args(&["aivo", "code", "skills", "install", "github:a/b", "--all"]),
                &no_bundles()
            ),
            args(&["aivo", "code skills", "install", "github:a/b", "--all"])
        );
    }

    #[test]
    fn rewrite_code_without_mcp_unchanged() {
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code"]), &no_bundles()),
            args(&["aivo", "code"])
        );
        assert_eq!(
            rewrite_cli_args(args(&["aivo", "code", "-p", "mcp"]), &no_bundles()),
            args(&["aivo", "code", "-p", "mcp"])
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
            canonical_flag_name("--12m"),
            Some("--max-context".to_string())
        );
        assert_eq!(
            canonical_flag_name("--3M"),
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
