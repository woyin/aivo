//! Capability-gated handoff for granted plugins. When a plugin declares the
//! `endpoint` capability and the user has consented, aivo serves the active key
//! through a per-launch loopback proxy and hands over only `AIVO_ENDPOINT_URL` +
//! `AIVO_ENDPOINT_TOKEN` — the secret never leaves aivo. The proxy is backed by
//! `ServeRouter`, the responses-capable internal router for OpenAI-protocol keys
//! and Copilot, or the Cursor ACP router for `type: coding-agent` plugins. OAuth
//! credentials are native-agent-only, so they run bare. For `type: coding-agent`
//! plugins the launch is also wrapped in the same stats/logs accounting built-in
//! tools get. Frozen contract: `docs/PLUGIN-PROTOCOL.md`.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::key_resolution::{KeyLookupMode, KeyResolution, resolve_key_override};
use crate::services::key_compat::KeyCompatContext;
use crate::services::log_store::{LogEvent, LogStore, new_log_id};
use crate::services::request_log::RequestLogger;
use crate::services::serve_router::{ServeRouter, ServeRouterConfig, random_auth_token};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::services::usage_stats_store::RunTokenTally;
use crate::style;

use super::manifest::{grantable_capabilities, probe_manifest};
use super::registry::{self, PluginRecord};

/// Dispatch a plugin that may need a key/endpoint handoff or run accounting.
/// Falls back to a plain spawn (today's behavior) when neither applies.
pub(crate) async fn dispatch(name: &str, bin: &Path, args: &[String], store: &SessionStore) -> i32 {
    let config_dir = store.config_dir().to_path_buf();

    let Some(plan) = grant_plan(name, bin).await else {
        eprintln!("  aborted; `{name}` was not run.");
        return crate::errors::ExitCode::UserError.code();
    };
    if plan.caps.is_empty() && !plan.is_coding_agent {
        // Nothing granted and no host behavior — plain dispatch.
        return super::exec_plugin(bin, args, &config_dir, &[]).await;
    }

    // `--help`/`-h` is a pure passthrough: never resolve a key, open a picker,
    // stand up an endpoint, or fail on auth just to print help. For top-level
    // help, prepend a uniform banner documenting the flags aivo intercepts, so
    // thin wrappers surface them the same way a fat plugin's own help does.
    match help_kind(args) {
        HelpKind::None => {}
        kind => {
            // Skip the banner when the plugin self-documents aivo's flags (a fat
            // plugin's own `--help` lists them better) — else it's pure duplication.
            if matches!(kind, HelpKind::TopLevel) && !plan.documents_aivo_flags {
                print_aivo_help_banner(name, &plan);
            }
            return super::exec_plugin(bin, args, &config_dir, &[]).await;
        }
    }

    // aivo owns `-k`/`-m` for any plugin it serves a key to — a coding-agent OR an
    // endpoint-granted plugin of any type. For those it resolves the key/model (picker on a
    // bare `-k`/`-m`, otherwise the value, otherwise the remembered selection) and hands the
    // model over as `AIVO_KEY_MODEL`. A coding-agent additionally owns `--debug`/`--dry-run`
    // and strips all four; an endpoint `tool` plugin keeps its own `--debug`/`--dry-run`, so
    // we strip only `-k`/`-m`. A granted-but-no-endpoint plugin keeps verbatim argv — `-k <id>`
    // still selects a key, but a bare `-k` uses the active key (no picker).
    let manages_km = plan.is_coding_agent || plan.has("endpoint");
    let flags = super::extract_aivo_flags(args);
    // `--max-context` (coding-agent only, like --debug/--dry-run): a manual window
    // for an unknown model, reaching the plugin via AIVO_MODEL_CONTEXT_WINDOW.
    if plan.is_coding_agent
        && let Some(value) = flags.max_context.as_deref()
    {
        match crate::services::model_metadata::parse_context_size(value) {
            Some(tokens) => crate::services::model_metadata::set_context_window_override(tokens),
            None => {
                eprintln!(
                    "{} --max-context expects a size like '200k', '1m', or '128000' (got {value:?}).",
                    style::red("Error:")
                );
                return crate::errors::ExitCode::UserError.code();
            }
        }
    }
    // `--dry-run` previews the resolved plan instead of launching — coding-agent only (it
    // mirrors native `aivo run --dry-run`; other plugins keep their own `--dry-run`).
    let dry_run = plan.is_coding_agent && flags.dry_run;
    let debug_log = if plan.is_coding_agent {
        flags.debug_log.clone()
    } else {
        super::debug_log_path(args)
    };
    let (key_flag, model_flag, mut plugin_args): (Option<String>, Option<String>, Vec<String>) =
        if plan.is_coding_agent {
            (flags.key, flags.model, flags.rest)
        } else if manages_km {
            // Endpoint plugin: aivo owns -k/-m; strip just those, keep --debug/--dry-run.
            (flags.key, flags.model, super::strip_key_model_flags(args))
        } else {
            (flags.key.filter(|s| !s.is_empty()), None, args.to_vec())
        };

    // HF/local-gguf takeover: an `hf:`/gguf model is served by a local
    // llama-server, not the active key. Spawn it, synthesize the loopback key,
    // and serve the model over the endpoint (the wrapped tool can't read `hf:`).
    // An explicit `-m hf:…` takes over for any key-managed plugin — without this
    // the ref would be handed to the real key's endpoint and persisted as a
    // bogus remembered model. The positional form (`aivo <plugin> hf:…`) stays
    // coding-agent only: generic plugins take real paths/refs positionally,
    // which must not be lifted. Mirrors `aivo run`'s takeover.
    let hf = if manages_km {
        match take_hf_takeover(
            model_flag.as_deref(),
            &mut plugin_args,
            plan.is_coding_agent,
            dry_run,
        )
        .await
        {
            Ok(h) => h,
            Err(e) => {
                eprintln!("  {} {e:#}", style::red("Error:"));
                return 1;
            }
        }
    } else {
        None
    };

    // The takeover owns the key + model when active; otherwise resolve as usual.
    let key_was_explicit = hf.is_none() && key_flag.is_some();
    // --dry-run is non-interactive: a bare `-k` (which would open the key picker)
    // falls back to the active/last-used key for the preview.
    let key_flag_for_resolve = match (dry_run, key_flag.as_deref()) {
        (true, Some("")) => None,
        (_, other) => other,
    };
    let key = match &hf {
        Some(h) => Some(h.key.clone()),
        None => resolve_plugin_key(store, key_flag_for_resolve, plan.is_coding_agent).await,
    };
    let serve = key.as_ref().map(plugin_serve);
    // Whether aivo can serve this key to THIS plugin: Cursor goes through the
    // coding-agent-only ACP router, so for any other plugin it is effectively
    // blocked — resolving a model for it would picker-then-refuse.
    let servable = match serve {
        Some(PluginServe::Serve) => true,
        Some(PluginServe::Cursor) => plan.is_coding_agent,
        _ => false,
    };

    let model_flag_was_explicit = model_flag.is_some();
    let model_request = plugin_model_request(model_flag, key_was_explicit);

    // aivo resolves a concrete model for any plugin whose `-k`/`-m` it owns:
    // `-m <v>` is used + remembered, a bare launch reuses the remembered model,
    // a bare `-m` (or `-k` without `-m`) opens the picker once and the choice is
    // remembered — handed over as `AIVO_KEY_MODEL`. Unservable keys (OAuth,
    // Cursor for a non-coding-agent) skip resolution: the plugin runs on its
    // own auth, so a picker would resolve-then-refuse.
    let model = match &hf {
        // The takeover already resolved the served model.
        Some(h) => Some(h.model.clone()),
        None => match key.as_ref() {
            Some(k) if manages_km && servable => {
                resolve_plugin_model(store, k, model_request, model_flag_was_explicit, dry_run)
                    .await
            }
            _ => model_request,
        },
    };

    // An explicit `-m` (value or picker) — or an hf takeover, whose endpoint
    // serves only that model — makes the handoff model binding rather than a
    // remembered fallback; plugins rank `AIVO_KEY_MODEL` above their own model
    // pin env when this is set.
    let model_explicit = model_flag_was_explicit || hf.is_some();

    // --dry-run: report the resolved plan and stop before any side effect —
    // no last-selection write, endpoint, accounting, llama-server, or launch.
    if dry_run {
        print_plugin_dry_run(
            name,
            bin,
            &config_dir,
            key.as_ref(),
            serve,
            plan.has("endpoint"),
            model.as_deref(),
            model_explicit,
            debug_log.as_deref(),
            &plugin_args,
            hf.is_some(),
        );
        return crate::errors::ExitCode::Success.code();
    }

    // Record this launch as the last selection — the same per-key model memory
    // native `aivo run` writes (`commands/run.rs`), so a coding-agent plugin and
    // a native tool agree on the model for a key (`resolve_plugin_model` reads it
    // back via `remembered_plugin_model`). Fires on every launch, not just an
    // explicit `-k`: a *fat* plugin (amp) re-resolves the key from the store, and
    // a bare launch — whose key/model were themselves resolved from this record —
    // writes back idempotently. Skipped on an hf takeover (mirrors run.rs): the
    // synthetic loopback key isn't in the store, so persisting it would poison
    // the record and the next read would prune it. A modelless launch
    // (OAuth/unservable key, or no TTY for the picker) persists only a key
    // *switch*: rewriting the same key's record with `model: None` would erase
    // the remembered model every other launch relies on.
    if manages_km
        && hf.is_none()
        && let Some(k) = key.as_ref()
    {
        let same_key_modelless = model.is_none()
            && store
                .get_last_selection()
                .await
                .ok()
                .flatten()
                .is_some_and(|s| s.key_id == k.id);
        if !same_key_modelless {
            let _ = store.set_last_selection(k, name, model.as_deref()).await;
        }
    }

    // The endpoint is the only handoff: aivo stands up a loopback proxy bound to
    // the key and hands over one OpenAI-compatible URL + bearer — the secret
    // never leaves aivo. `ServeRouter` serves Anthropic/Gemini REST keys and
    // Ollama; OpenAI-protocol REST keys and Copilot use the responses-capable
    // router; Cursor uses the ACP-backed cursor router for coding-agent plugins.
    // OAuth credentials are native-agent-only (no proxy aivo hands to a plugin),
    // so they run bare.
    let mut extra_env: Vec<(String, String)> = Vec::new();
    if let Some(path) = &debug_log {
        extra_env.push((
            "AIVO_DEBUG_LOG".to_string(),
            path.to_string_lossy().to_string(),
        ));
    }
    let mut endpoint: Option<EndpointHandle> = None;
    // Token-accounted REST engines (serve / responses) fold this run's usage into
    // the tally; `finish_accounting` stamps the total onto the run's finished log
    // row so `aivo stats --since` can window it. Cursor/OAuth runs leave it at zero
    // (the endpoint doesn't token-account them), so their finished row stays bare.
    let run_tally = Arc::new(RunTokenTally::default());
    if let (Some(key), Some(serve)) = (key.as_ref(), serve) {
        match serve {
            // OAuth lives only in its own native agent — nothing to hand over;
            // the plugin runs on its own auth. Note it once so an
            // endpoint-granted plugin isn't silently mis-keyed.
            PluginServe::Blocked if plan.has("endpoint") => {
                eprintln!(
                    "  {} key `{}` is a {} credential — usable only in `{}`; `{}` runs on its own auth",
                    style::yellow("!"),
                    key.display_name(),
                    key.oauth_kind_label(),
                    key.oauth_tool_hint(),
                    name,
                );
            }
            PluginServe::Blocked => {}
            PluginServe::Cursor if plan.has("endpoint") => {
                if plan.is_coding_agent {
                    push_model_env(&mut extra_env, key, model.as_deref(), model_explicit).await;
                    maybe_init_http_debug(debug_log.as_deref()).await;
                    apply_endpoint(
                        start_cursor_endpoint(key).await,
                        &mut extra_env,
                        &mut endpoint,
                    );
                } else {
                    eprintln!(
                        "  {} key `{}` is a Cursor credential — usable only by `type: coding-agent` plugins; `{}` runs on its own auth",
                        style::yellow("!"),
                        key.display_name(),
                        name,
                    );
                }
            }
            PluginServe::Cursor => {}
            // The resolved model travels with the handoff (the plugin sends it to
            // the endpoint). OpenAI-protocol keys and Copilot use the
            // responses-capable internal proxy (so gpt-5.x reasoning+tools reach
            // `/v1/responses`, even for a Chat Completions client); others use the
            // serve proxy, which additionally logs each request to the plugin's
            // `--debug` file.
            PluginServe::Serve if plan.has("endpoint") => {
                push_model_env(&mut extra_env, key, model.as_deref(), model_explicit).await;
                let started = if use_responses_router(key) {
                    maybe_init_http_debug(debug_log.as_deref()).await;
                    start_responses_endpoint(key, store, name, run_tally.clone()).await
                } else {
                    let started = start_loopback_endpoint(
                        key,
                        store.logs(),
                        store.clone(),
                        name,
                        debug_log.clone(),
                        run_tally.clone(),
                    )
                    .await;
                    if started.is_ok()
                        && let Some(p) = &debug_log
                    {
                        eprintln!(
                            "  {} {}",
                            style::dim("·"),
                            style::dim(format!("endpoint traffic logged to {}", p.display())),
                        );
                    }
                    started
                };
                apply_endpoint(started, &mut extra_env, &mut endpoint);
            }
            // Servable, but the plugin didn't request `endpoint` — nothing to do.
            _ => {}
        }
    }

    let accounting = if plan.is_coding_agent {
        Some(
            begin_accounting(
                store,
                name,
                key.as_ref(),
                model.as_deref(),
                &plugin_args,
                run_tally.clone(),
            )
            .await,
        )
    } else {
        None
    };

    let code = super::exec_plugin(bin, &plugin_args, &config_dir, &extra_env).await;

    // Duration measures the run itself; the endpoint drain below doesn't count.
    let run_duration = accounting.as_ref().map(|a| a.started.elapsed());
    // Drain the endpoint before reading the tally — an accounting task may still
    // be folding the final response's usage into it.
    if let Some(ep) = endpoint {
        ep.shutdown().await;
    }
    if let (Some(acct), Some(duration)) = (accounting, run_duration) {
        finish_accounting(store, acct, code, duration).await;
    }
    // Stop the llama-server an hf takeover auto-started: the plugin path exits via
    // `process::exit` and never reaches run.rs's global `stop_if_we_started`.
    if hf.is_some() {
        crate::services::huggingface::stop_if_we_started();
    }
    code
}

async fn maybe_init_http_debug(path: Option<&Path>) {
    let Some(path) = path else {
        return;
    };
    match crate::services::http_debug::init(path.to_path_buf()).await {
        Ok(p) => eprintln!("[aivo] HTTP debug log → {}", p.display()),
        Err(e) => {
            eprintln!("[aivo] failed to open debug log: {e}; HTTP requests will not be logged")
        }
    }
}

// ── capability grant resolution ─────────────────────────────────────────────

struct GrantPlan {
    /// Grantable caps the user approved (a subset of what was requested).
    caps: Vec<String>,
    is_coding_agent: bool,
    /// The plugin self-documents aivo's flags → skip the help banner.
    documents_aivo_flags: bool,
}

impl GrantPlan {
    fn has(&self, cap: &str) -> bool {
        self.caps.iter().any(|c| c == cap)
    }
}

/// Resolve what's granted to `name`: read the cached manifest + grants, or
/// (for a managed but manifest-less plugin) lazily probe and seek consent once.
/// `None` means the user declined the first-run gate — don't run the plugin.
async fn grant_plan(name: &str, bin: &Path) -> Option<GrantPlan> {
    let reg = registry::load();
    let Some(rec) = reg.plugins.get(name) else {
        // Unmanaged/PATH plugin — nothing to grant; runs plain.
        return Some(GrantPlan {
            caps: Vec::new(),
            is_coding_agent: false,
            documents_aivo_flags: false,
        });
    };
    // Consent is bound to the bytes it was given for; strip it (and the
    // stale manifest) when the recorded binary drifted from the approved pin.
    let rec = reverify_on_drift(name, rec);
    if let Some(manifest) = &rec.manifest {
        // Already probed — honor the recorded grants, but re-ask (TTY) for
        // requested caps not yet granted: a past decline means "not that
        // time", not forever, else the plugin is stuck cap-less until a
        // reinstall. Accepted grants persist so the prompt doesn't recur.
        let requested = grantable_capabilities(&manifest.capabilities);
        let mut granted = retained_grants(&requested, &rec.granted_caps);
        let missing = missing_grants(&requested, &rec.granted_caps);
        if !missing.is_empty()
            && std::io::stdin().is_terminal()
            && super::prompt_capability_grant(name, &missing)
        {
            granted.extend(missing);
            let mut updated = rec.clone();
            updated.granted_caps = granted.clone();
            updated.approved_checksum = updated.checksum.clone();
            registry::record(name, updated);
        }
        return Some(GrantPlan {
            caps: granted,
            is_coding_agent: manifest.is_coding_agent(),
            documents_aivo_flags: manifest.documents_aivo_flags,
        });
    }
    // Manifest-less managed plugin (e.g. a `npm:`/`github:` install): lazily
    // probe + seek consent once, persisting the grant for next time.
    lazy_probe_and_consent(name, bin, &rec).await
}

/// True when the record's consent was given for different bytes than the ones
/// on record. A missing pin on either side proves nothing and passes (legacy
/// records); local paths are the user's own file and are never re-gated.
fn consent_drifted(rec: &PluginRecord) -> bool {
    if rec.run_approved || !rec.granted_caps.is_empty() {
        let local = matches!(
            super::source::classify(&rec.source),
            Ok(super::source::SourceKind::LocalPath)
        );
        return !local
            && matches!(
                (&rec.approved_checksum, &rec.checksum),
                (Some(approved), Some(current)) if approved != current
            );
    }
    false
}

/// Strip consent (persisted) when the recorded binary no longer matches the
/// bytes it was approved for, so the first-run + capability gates re-ask for
/// the new binary instead of inheriting the old one's approval.
fn reverify_on_drift(name: &str, rec: &PluginRecord) -> PluginRecord {
    if !consent_drifted(rec) {
        return rec.clone();
    }
    eprintln!(
        "  {} `{name}` changed since it was approved — re-verifying",
        style::yellow("!")
    );
    let mut fresh = rec.clone();
    fresh.run_approved = false;
    fresh.granted_caps.clear();
    fresh.approved_checksum = None;
    // The cached manifest describes the approved bytes, not these.
    fresh.manifest = None;
    registry::record(name, fresh.clone());
    fresh
}

/// First-dispatch probe + consent for a managed plugin recorded without a
/// manifest (e.g. installed via `npm:`/`github:`, where install-time probing is
/// skipped). On a successful probe the manifest + approved caps are persisted so
/// neither the probe nor the prompt recurs. Returns `None` when the user
/// declines the first-run gate.
async fn lazy_probe_and_consent(
    name: &str,
    bin: &Path,
    existing: &PluginRecord,
) -> Option<GrantPlan> {
    // First-run gate: the manifest probe below would be this downloaded
    // binary's very first execution. Capability consent governs what aivo
    // hands over, not whether the binary runs — that decision is this one.
    // Fail closed off-TTY: silently executing an unapproved (or changed)
    // binary would turn the re-verify promise into a no-op in CI/cron.
    if !existing.run_approved && existing.granted_caps.is_empty() {
        if !std::io::stdin().is_terminal() {
            eprintln!(
                "  {} plugin `{name}` (from {}) is not approved to run, and this session cannot prompt (no TTY).",
                style::red("✗"),
                existing.source
            );
            eprintln!(
                "    Run {} once from a terminal to approve it, or reinstall with {}.",
                style::cyan(format!("aivo {name}")),
                style::cyan(format!("aivo plugins install {} --trust", existing.source)),
            );
            return None;
        }
        if !super::prompt_first_run(name, &existing.source) {
            return None;
        }
    }
    let Some(manifest) = probe_manifest(bin, name).await else {
        // Persist the run approval so a manifest-less plugin (probe failed or
        // unsupported) isn't re-gated on every dispatch.
        if !existing.run_approved {
            let mut rec = existing.clone();
            rec.run_approved = true;
            rec.approved_checksum = rec.checksum.clone();
            registry::record(name, rec);
        }
        return Some(GrantPlan {
            caps: Vec::new(),
            is_coding_agent: false,
            documents_aivo_flags: false,
        });
    };
    let is_ca = manifest.is_coding_agent();
    let documents_aivo_flags = manifest.documents_aivo_flags;
    let requested = grantable_capabilities(&manifest.capabilities);
    let mut granted = retained_grants(&requested, &existing.granted_caps);
    let missing = missing_grants(&requested, &existing.granted_caps);
    if !missing.is_empty()
        && std::io::stdin().is_terminal()
        && super::prompt_capability_grant(name, &missing)
    {
        granted.extend(missing);
    }
    let mut rec = existing.clone();
    rec.manifest = Some(manifest);
    rec.granted_caps = granted.clone();
    rec.run_approved = true;
    rec.approved_checksum = rec.checksum.clone();
    registry::record(name, rec);
    Some(GrantPlan {
        caps: granted,
        is_coding_agent: is_ca,
        documents_aivo_flags,
    })
}

fn retained_grants(requested: &[String], prior: &[String]) -> Vec<String> {
    requested
        .iter()
        .filter(|c| prior.contains(c))
        .cloned()
        .collect()
}

fn missing_grants(requested: &[String], prior: &[String]) -> Vec<String> {
    requested
        .iter()
        .filter(|c| !prior.contains(c))
        .cloned()
        .collect()
}

/// How a plugin invocation requests help. `Sub` (`aivo amp trust --help`) is a
/// passthrough only; `TopLevel` (`aivo amp --help`) also gets the aivo banner.
enum HelpKind {
    None,
    Sub,
    TopLevel,
}

fn help_kind(args: &[String]) -> HelpKind {
    if !args.iter().any(|a| a == "-h" || a == "--help") {
        HelpKind::None
    } else if matches!(args.first().map(String::as_str), Some("-h" | "--help")) {
        HelpKind::TopLevel
    } else {
        HelpKind::Sub
    }
}

/// One uniform block above a plugin's own `--help` documenting the flags aivo
/// intercepts, so every plugin (fat or thin) surfaces them the same way. Goes to
/// stdout (it's help content) and is flushed before the child writes its own.
fn print_aivo_help_banner(name: &str, plan: &GrantPlan) {
    use std::io::Write;
    println!(
        "{} {}",
        style::cyan(format!("aivo {name}")),
        style::dim("— flags handled by aivo:"),
    );
    let row = |flag: &str, desc: &str| {
        println!(
            "  {}  {}",
            style::bold(format!("{flag:<22}")),
            style::dim(desc)
        );
    };
    if plan.is_coding_agent || plan.has("endpoint") {
        // aivo owns -k/-m for coding-agent and endpoint plugins (picker on a bare flag).
        row(
            "-k, --key [<id|name>]",
            "aivo key to use (bare -k opens the picker)",
        );
        row(
            "-m, --model [<model>]",
            "model to use (bare -m opens the picker)",
        );
        // --debug/--dry-run are aivo-owned only for a coding-agent; an endpoint tool plugin
        // keeps its own.
        if plan.is_coding_agent {
            row("--debug[=<path>]", "log the proxied endpoint traffic");
            row(
                "--dry-run",
                "preview the resolved key/model/command without launching",
            );
        }
    } else {
        row(
            "-k, --key [<id|name>]",
            "aivo key to serve over the endpoint (bare -k opens the picker)",
        );
        row(
            "-m, --model [<model>]",
            "model to use (bare -m opens the picker)",
        );
        row("--debug[=<path>]", "log the proxied endpoint traffic");
    }
    println!();
    println!("{}", style::dim(format!("{name}'s own help:")));
    println!("{}", style::dim("─".repeat(40)));
    let _ = std::io::stdout().flush();
}

fn plugin_model_request(model_flag: Option<String>, key_was_explicit: bool) -> Option<String> {
    match (model_flag, key_was_explicit) {
        (None, true) => Some(String::new()),
        (other, _) => other,
    }
}

/// A stood-up hf/local-gguf takeover: the synthetic loopback key bound to the
/// local llama-server, plus the served model's display name.
struct HfTakeover {
    key: ApiKey,
    model: String,
}

/// Detect and stand up an `hf:`/local-gguf takeover for a key-managed plugin.
/// The ref is an explicit `-m hf:…`, else (under `allow_positional`, coding-agent
/// only) the first positional `hf:`/gguf arg — lifted out of `plugin_args` since
/// the wrapped tool can't interpret it. Spawns the local llama-server and returns
/// the synthetic loopback key + served model. `Ok(None)` when the invocation
/// carries no hf ref. Under `dry_run` the llama-server is not spawned: the
/// synthetic key carries a placeholder port so the preview reflects the takeover
/// without the (slow, stateful) launch.
async fn take_hf_takeover(
    model_flag: Option<&str>,
    plugin_args: &mut Vec<String>,
    allow_positional: bool,
    dry_run: bool,
) -> anyhow::Result<Option<HfTakeover>> {
    use crate::services::huggingface as hf;

    let Some(raw) = take_hf_ref(model_flag, plugin_args, allow_positional) else {
        return Ok(None);
    };

    // Bare `hf:` opens the cached-model picker; resolve it to a concrete ref.
    let raw = if hf::is_bare_hf_picker_trigger(&raw) {
        match hf::pick_cached_short_ref() {
            Some(short) => short,
            None => std::process::exit(0),
        }
    } else {
        raw
    };

    let hf_ref = hf::parse_hf_ref(&raw)?;
    let port = if dry_run {
        eprintln!(
            "  {} {}",
            style::yellow("Note:"),
            style::dim("--dry-run shows a placeholder local port; llama-server is not spawned."),
        );
        0
    } else {
        hf::ensure_ready(&hf_ref).await?
    };
    Ok(Some(HfTakeover {
        key: hf::local_takeover_key(&hf_ref, port),
        model: hf_ref.display_model_name(),
    }))
}

/// Render the coding-agent `--dry-run` preview: the resolved key, model, endpoint
/// router, injected env, and the command aivo would spawn — without standing up
/// an endpoint, spawning a llama-server, recording accounting, or launching the
/// plugin. The env mirrors what `dispatch` would inject (`exec_plugin` always
/// adds `AIVO_CONFIG_DIR`; the endpoint URL/token are shown as placeholders since
/// the loopback port is assigned at bind time).
#[allow(clippy::too_many_arguments)]
fn print_plugin_dry_run(
    name: &str,
    bin: &Path,
    config_dir: &Path,
    key: Option<&ApiKey>,
    serve: Option<PluginServe>,
    endpoint_granted: bool,
    model: Option<&str>,
    model_explicit: bool,
    debug_log: Option<&Path>,
    plugin_args: &[String],
    hf_active: bool,
) {
    use crate::commands::format_shell_command;

    println!("{} {}", style::bold("Plugin:"), style::cyan(name));
    println!(
        "{} {}",
        style::bold("Binary:"),
        style::dim(bin.display().to_string())
    );
    match key {
        Some(k) => println!(
            "{} {} {}",
            style::bold("Key:"),
            style::cyan(k.display_name()),
            style::dim(format!("({})", k.base_url)),
        ),
        None => println!(
            "{} {}",
            style::bold("Key:"),
            style::dim("(none — plugin runs on its own auth)"),
        ),
    }
    println!(
        "{} {}",
        style::bold("Model:"),
        model.unwrap_or("(needs -m)")
    );

    // Build the env aivo would inject and describe the endpoint handoff, reusing
    // the same classification `dispatch` uses so the preview can't drift.
    let mut env: Vec<(String, String)> = vec![(
        "AIVO_CONFIG_DIR".to_string(),
        config_dir.display().to_string(),
    )];
    if let Some(p) = debug_log {
        env.push(("AIVO_DEBUG_LOG".to_string(), p.display().to_string()));
    }
    let endpoint_desc = match (key, serve) {
        (Some(_), Some(PluginServe::Blocked)) => {
            "(OAuth credential — plugin runs on its own auth)".to_string()
        }
        (Some(k), Some(serve)) if serve.is_servable() && endpoint_granted => {
            if let Some(m) = model {
                env.push(("AIVO_KEY_MODEL".to_string(), m.to_string()));
                if model_explicit {
                    env.push(("AIVO_KEY_MODEL_EXPLICIT".to_string(), "1".to_string()));
                }
            }
            env.push((
                "AIVO_ENDPOINT_URL".to_string(),
                "http://127.0.0.1:<port>/v1".to_string(),
            ));
            env.push((
                "AIVO_ENDPOINT_TOKEN".to_string(),
                "<assigned at launch>".to_string(),
            ));
            match serve {
                PluginServe::Cursor => "Cursor ACP router".to_string(),
                PluginServe::Serve if use_responses_router(k) => {
                    "responses-capable proxy".to_string()
                }
                _ => "serve proxy".to_string(),
            }
        }
        (Some(_), Some(_)) => {
            "(plugin lacks the `endpoint` capability — runs on its own auth)".to_string()
        }
        _ => "(no key — plugin runs on its own auth)".to_string(),
    };
    println!("{} {}", style::bold("Endpoint:"), style::dim(endpoint_desc));
    println!(
        "{} {}",
        style::bold("Command:"),
        format_shell_command(&bin.to_string_lossy(), plugin_args),
    );

    println!();
    println!("{}", style::bold("Environment:"));
    for (k, v) in &env {
        println!("  {k}={v}");
    }

    let mut notes: Vec<String> = Vec::new();
    if hf_active {
        notes.push(
            "hf/local-gguf model: a local llama-server is started at launch (skipped here)"
                .to_string(),
        );
    }
    notes.push("--dry-run resolves the key/model but skips the endpoint and launch".to_string());
    println!();
    println!("{}", style::bold("Notes:"));
    for n in &notes {
        println!("  {} {}", style::arrow_symbol(), n);
    }
}

/// The hf/gguf ref for a key-managed plugin invocation, or `None`. An explicit
/// `-m` wins entirely (an `hf:` value takes over; a normal model suppresses any
/// positional lift). The positional lift — the first `hf:`/gguf arg, parity with
/// `aivo run`'s lifting in `cli_args` — runs only under `allow_positional`
/// (coding-agent): the predicate matches any `/`-anchored path, and a generic
/// plugin's positionals are real paths/refs that must reach it untouched.
fn take_hf_ref(
    model_flag: Option<&str>,
    plugin_args: &mut Vec<String>,
    allow_positional: bool,
) -> Option<String> {
    use crate::services::huggingface::is_hf_or_local_gguf;
    match model_flag {
        Some(m) if is_hf_or_local_gguf(m) => Some(m.to_string()),
        Some(_) => None,
        None if allow_positional => {
            let idx = plugin_args.iter().position(|a| is_hf_or_local_gguf(a))?;
            Some(plugin_args.remove(idx))
        }
        None => None,
    }
}

/// Resolve the handoff key from the extracted `-k`/`--key` value: `Some("")`
/// opens the picker, `Some(id)` selects that key, `None` uses the active/last-used
/// key silently (`PreferActiveAllowNone` — no prompt before launch).
async fn resolve_plugin_key(
    store: &SessionStore,
    key_flag: Option<&str>,
    allow_cursor: bool,
) -> Option<ApiKey> {
    match resolve_key_override(
        store,
        key_flag,
        KeyLookupMode::PreferActiveAllowNone,
        // The picker dims CLI-bound OAuth credentials. Cursor is selectable only
        // for coding-agent plugins, which can use the Cursor ACP endpoint.
        KeyCompatContext::Plugin { allow_cursor },
    )
    .await
    {
        Ok(KeyResolution::Selected(k)) => Some(k),
        // The user opened the picker (`-k`) and cancelled → abort the run, like
        // `aivo <tool>` does, rather than silently launching on another key.
        Ok(KeyResolution::Cancelled) => {
            eprintln!("{}", style::dim("Cancelled."));
            std::process::exit(0)
        }
        // No usable key → run bare; the plugin falls back to its own auth.
        Ok(KeyResolution::MissingAuth) => None,
        // A real resolution failure (unknown key id, picker without a terminal,
        // unreadable store) aborts like native `aivo <tool>` — running bare would
        // mislabel it as a missing endpoint grant.
        Err(e) => {
            eprintln!("  {} {e:#}", style::red("Error:"));
            std::process::exit(crate::errors::ExitCode::UserError.code())
        }
    }
}

/// Which loopback proxy (if any) aivo uses to serve a key to a plugin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PluginServe {
    /// Any REST key, starter, Copilot, Ollama — served over a loopback proxy
    /// (`ServeRouter` or the responses-capable router; see `use_responses_router`).
    Serve,
    /// Cursor ACP — served through `CursorModelRouter` for coding-agent plugins.
    Cursor,
    /// OAuth (Claude/Codex/Gemini) — native-agent-only; aivo has no
    /// plugin-servable proxy for them, so they run bare.
    Blocked,
}

impl PluginServe {
    fn is_servable(self) -> bool {
        !matches!(self, PluginServe::Blocked)
    }
}

/// Classify how aivo can serve `key` to a plugin. OAuth is bound to native-agent
/// auth; Cursor goes through the ACP-backed cursor router; everything else aivo
/// proxies over a loopback endpoint.
fn plugin_serve(key: &ApiKey) -> PluginServe {
    if key.is_cursor_acp() {
        PluginServe::Cursor
    } else if key.is_provider_oauth() {
        // Grok/Codex OAuth are servable: aivo injects the token upstream.
        PluginServe::Serve
    } else if key.is_any_oauth() {
        PluginServe::Blocked
    } else {
        PluginServe::Serve
    }
}

/// The remembered model for `key`, preferring the shared `last_selection` (the
/// per-key model native `aivo run` writes — so a coding-agent plugin and a native
/// tool agree on the model for a key) and falling back to the per-key `chat_model`
/// (`aivo code` and an explicit plugin `-m` also write it). The `__default__`
/// sentinel — a native "let the tool choose" pin — isn't a concrete model a plugin
/// can run, so it's skipped in favor of the fallback.
async fn remembered_plugin_model(store: &SessionStore, key: &ApiKey) -> Option<String> {
    if let Some(sel) = store.get_last_selection().await.ok().flatten()
        && sel.key_id == key.id
        && let Some(m) = sel
            .model
            .filter(|m| m.as_str() != crate::constants::MODEL_DEFAULT_PLACEHOLDER)
    {
        return Some(m);
    }
    store.get_code_model(&key.id).await.ok().flatten()
}

/// Resolve the model for a coding-agent plugin. Unlike native tools (which fall
/// back to their own default), such a plugin needs a concrete model, so this
/// yields one whenever possible: explicit `-m <value>` is used; a bare launch
/// reuses the key's remembered model (see `remembered_plugin_model`); otherwise
/// the picker opens — delegating to the shared `resolve_model_outcome`. The picked
/// model is persisted by `dispatch`'s shared last-selection write (plus the
/// `chat_model` written here for `aivo code` interop / fallback). `None` only when
/// nothing resolves (bare launch, nothing saved, no picker) — the plugin then asks
/// the user to pass `-m`. Under `dry_run` nothing is persisted and the picker never
/// opens: a saved model is reused, else `None` (the preview shows `(needs -m)`).
async fn resolve_plugin_model(
    store: &SessionStore,
    key: &ApiKey,
    model_flag: Option<String>,
    explicit_model_flag: bool,
    dry_run: bool,
) -> Option<String> {
    use crate::commands::models::{ModelOutcome, resolve_model_outcome};

    // Explicit `-m <value>`: use it. Mirror it into `chat_model` for chat interop
    // (the shared last-selection write happens in `dispatch`); skipped under
    // --dry-run, which must not mutate saved state.
    if let Some(m) = model_flag.as_deref().filter(|s| !s.is_empty()) {
        if !dry_run {
            let _ = store.set_code_model(&key.id, m).await;
        }
        return Some(m.to_string());
    }
    // Bare launch reuses the remembered model rather than re-prompting. --dry-run
    // also reuses it for a bare `-m` (which would otherwise force the picker),
    // since the preview is non-interactive.
    if (model_flag.is_none() || dry_run)
        && let Some(saved) = remembered_plugin_model(store, key).await
    {
        return Some(saved);
    }
    // --dry-run never opens the picker: report no resolved model instead.
    if dry_run {
        return None;
    }
    // Bare `-m`, or a bare launch with nothing saved: the plugin needs a model, so
    // force the picker (synthesize a bare `-m`) via the shared resolver.
    let client = crate::services::http_utils::router_http_client();
    let cache = crate::services::ModelsCache::new();
    match resolve_model_outcome(
        &client,
        key,
        Some(String::new()),
        explicit_model_flag,
        false,
        None,
        &cache,
        "Select model",
    )
    .await
    {
        Ok(ModelOutcome::Model(m)) => {
            let _ = store.set_code_model(&key.id, &m).await;
            Some(m)
        }
        // Picker cancelled — don't launch (parity with `aivo run`).
        Ok(ModelOutcome::Cancelled) => {
            eprintln!("{}", crate::style::dim("Cancelled."));
            std::process::exit(0)
        }
        // No TTY / empty catalog: leave it to the plugin (it asks the user for -m).
        _ => None,
    }
}

// ── loopback control endpoint (reuses serve) ────────────────────────────────

/// A running per-launch loopback proxy bound to the plugin's key.
pub(crate) struct EndpointHandle {
    pub url: String,
    pub token: String,
    handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    /// `Some` for routers that honor a graceful-shutdown notify (`ServeRouter`);
    /// `None` for the responses router, which has no notify and is
    /// torn down via the abort path in [`EndpointHandle::shutdown`].
    shutdown: Option<Arc<tokio::sync::Notify>>,
    /// When set, protocol routes the router learned during the run are merged
    /// back into the key's config on shutdown — the same `(tool,key,model)`
    /// persistence native launches get, so the cascade isn't re-probed next time.
    persist: Option<RoutePersist>,
}

/// Captures what `persist_runtime_discoveries` needs to write a router's learned
/// routes back to the key after the plugin exits.
struct RoutePersist {
    store: SessionStore,
    key: ApiKey,
    route_cache: Arc<crate::services::route_cache::RouteCache>,
    learned: Arc<std::sync::atomic::AtomicBool>,
}

impl EndpointHandle {
    /// Assemble a handle from a bound `port`. Centralizes the load-bearing `/v1`
    /// base-URL contract: clients (omp, OpenAI SDKs) append `/chat/completions`
    /// etc., so the path must already carry `/v1`.
    fn new(
        port: u16,
        token: String,
        handle: tokio::task::JoinHandle<anyhow::Result<()>>,
        shutdown: Option<Arc<tokio::sync::Notify>>,
    ) -> Self {
        EndpointHandle {
            url: format!("http://127.0.0.1:{port}/v1"),
            token,
            handle,
            shutdown,
            persist: None,
        }
    }

    /// Persist the router's learned routes back to the key on shutdown.
    fn with_persist(mut self, persist: RoutePersist) -> Self {
        self.persist = Some(persist);
        self
    }

    /// Stop the proxy and release the ephemeral port, persisting learned routes
    /// first. The plugin has already exited, so nothing is in flight: a router
    /// with a notify (`ServeRouter`) gets a bounded graceful wind-down; one
    /// without (`ResponsesToChatRouter`) never finishes and is aborted at once
    /// rather than burning the full timeout on every teardown.
    async fn shutdown(self) {
        if let Some(p) = self.persist {
            crate::services::launch_runtime::persist_runtime_discoveries(
                &p.store,
                &p.key,
                Some(p.route_cache),
                Some(p.learned),
            )
            .await;
        }
        match self.shutdown {
            Some(notify) => {
                notify.notify_one();
                let abort = self.handle.abort_handle();
                if tokio::time::timeout(Duration::from_secs(3), self.handle)
                    .await
                    .is_err()
                {
                    abort.abort();
                }
            }
            None => self.handle.abort(),
        }
    }
}

/// Record a started endpoint's env (or warn on failure) — the shared tail of the
/// serve/responses dispatch arms.
/// Advisory limit env vars for the resolved model (limits cascade: live
/// models-cache → embedded snapshot). Emitted only when known; plugins map
/// them onto their CLI's own config instead of guessing. Contract:
/// `docs/PLUGIN-PROTOCOL.md`.
/// Push the resolved-model handoff env: `AIVO_KEY_MODEL`, the binding marker
/// `AIVO_KEY_MODEL_EXPLICIT` (this launch's `-m`/picker/hf — not the remembered
/// fallback), and the advisory limit vars. No-op when no model resolved.
async fn push_model_env(
    extra_env: &mut Vec<(String, String)>,
    key: &ApiKey,
    model: Option<&str>,
    explicit: bool,
) {
    let Some(m) = model else {
        return;
    };
    extra_env.push(("AIVO_KEY_MODEL".to_string(), m.to_string()));
    if explicit {
        extra_env.push(("AIVO_KEY_MODEL_EXPLICIT".to_string(), "1".to_string()));
    }
    extra_env.extend(model_limit_env(key, m).await);
}

async fn model_limit_env(key: &ApiKey, model: &str) -> Vec<(String, String)> {
    let cache = crate::services::models_cache::ModelsCache::new();
    model_limit_env_with_cache(&cache, key, model).await
}

async fn model_limit_env_with_cache(
    cache: &crate::services::models_cache::ModelsCache,
    key: &ApiKey,
    model: &str,
) -> Vec<(String, String)> {
    let cache_base = crate::commands::models::model_cache_key_for_key(key);
    let limits =
        crate::services::model_metadata::resolve_limits(cache, Some(&cache_base), model).await;
    let mut env = Vec::new();
    if let Some(context) = limits.context {
        env.push(("AIVO_MODEL_CONTEXT_WINDOW".to_string(), context.to_string()));
    }
    if let Some(output) = limits.output {
        env.push((
            "AIVO_MODEL_MAX_OUTPUT_TOKENS".to_string(),
            output.to_string(),
        ));
    }
    env
}

fn apply_endpoint(
    started: anyhow::Result<EndpointHandle>,
    extra_env: &mut Vec<(String, String)>,
    endpoint: &mut Option<EndpointHandle>,
) {
    match started {
        Ok(ep) => {
            extra_env.push(("AIVO_ENDPOINT_URL".to_string(), ep.url.clone()));
            extra_env.push(("AIVO_ENDPOINT_TOKEN".to_string(), ep.token.clone()));
            // The plugin only talks to this loopback endpoint (aivo proxies real
            // upstream itself). A system proxy would route the 127.0.0.1 request
            // through it, and plugins that ignore NO_PROXY (e.g. the Node grok
            // client) then fail. Neutralize the proxy vars so it connects direct.
            const PROXY_VARS: &[&str] = &[
                "HTTP_PROXY",
                "http_proxy",
                "HTTPS_PROXY",
                "https_proxy",
                "ALL_PROXY",
                "all_proxy",
            ];
            let proxy_set = PROXY_VARS
                .iter()
                .any(|v| std::env::var(v).is_ok_and(|s| !s.is_empty()));
            if proxy_set {
                for var in PROXY_VARS {
                    extra_env.push(((*var).to_string(), String::new()));
                }
                extra_env.push(("NO_PROXY".to_string(), "*".to_string()));
                extra_env.push(("no_proxy".to_string(), "*".to_string()));
            }
            *endpoint = Some(ep);
        }
        Err(e) => eprintln!(
            "  {} plugin endpoint unavailable: {e:#}",
            style::yellow("!")
        ),
    }
}

/// Start a `serve` proxy bound to `key` on an OS-assigned loopback port, with an
/// ephemeral bearer token. Token usage of buffered 2xx responses is recorded
/// against the key, labeled with `tool`. `debug_log`, when set, logs each
/// proxied request to that file.
async fn start_loopback_endpoint(
    key: &ApiKey,
    log_store: LogStore,
    usage: SessionStore,
    tool: &str,
    debug_log: Option<PathBuf>,
    run_tally: Arc<RunTokenTally>,
) -> anyhow::Result<EndpointHandle> {
    let token = random_auth_token();
    let config = ServeRouterConfig::from_key(key, false, 300, Some(token.clone()), HashMap::new());
    let mut router = ServeRouter::new(config, key.clone(), log_store)
        .with_usage_accounting(usage, tool.to_string())
        .with_run_tally(run_tally);
    if let Some(path) = &debug_log
        && let Some(logger) = RequestLogger::new_with_path(path).await
    {
        router = router.with_logger(Some(logger));
    }
    let (handle, shutdown, port) = router.start_background_with_addr("127.0.0.1", 0).await?;
    Ok(EndpointHandle::new(port, token, handle, Some(shutdown)))
}

/// Start the Cursor ACP compatibility router for a coding-agent plugin. The
/// router speaks the same local OpenAI/Anthropic/Gemini/Responses paths native
/// Cursor-backed tools use, but is bearer-gated for the plugin endpoint.
async fn start_cursor_endpoint(key: &ApiKey) -> anyhow::Result<EndpointHandle> {
    use crate::services::cursor_acp;
    use crate::services::cursor_bridge::{CursorModelRouter, CursorRouterConfig};

    cursor_acp::ensure_cursor_agent_installed()?;
    if cursor_acp::is_legacy_cursor_login_secret(key.key.as_str()) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Run `aivo keys rm {0}` then `aivo keys add cursor` to recreate it as an isolated account.",
            key.id
        );
    }

    // Check OAuth-login auth before handing a dead endpoint to the plugin.
    if cursor_acp::cursor_oauth_shadow_signed_out(key).await {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth {0}` to sign in again.",
            key.id
        );
    }

    let token = random_auth_token();
    let workspace_cwd =
        crate::services::system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
    let router = CursorModelRouter::new(CursorRouterConfig {
        key: key.clone(),
        workspace_cwd,
        models_cache: Some(crate::services::models_cache::ModelsCache::new()),
        // Do not prewarm for plugin endpoints: merely launching a plugin should
        // not open ACP generation sessions before the endpoint is actually used.
        prewarm_count: 0,
        mcp_prewarm_id_style: None,
        expected_token: Some(token.clone()),
    });
    let (port, handle) = router.start_background().await?;
    Ok(EndpointHandle::new(port, token, handle, None))
}

/// The `(tool,key,model)` route-cache namespace for plugin endpoints. Routes are
/// a property of the upstream (key,model), not the plugin, so all plugins share
/// one namespace — distinct from native tools' ("codex"/"opencode"/…).
const PLUGIN_ROUTE_TOOL: &str = "plugin";

/// True when a key should be served through the responses-capable internal proxy
/// (`ResponsesToChatRouter`) rather than `ServeRouter`: OpenAI-protocol REST keys
/// and Copilot, which may need `/v1/responses` (gpt-5.x reasoning + tools). The
/// responses router exchanges Copilot's token itself. Anthropic/Gemini keys keep
/// `ServeRouter`'s family cascade; Ollama keeps `ServeRouter`.
fn use_responses_router(key: &ApiKey) -> bool {
    use crate::services::provider_profile::{is_ollama_base, provider_profile_for_key};
    use crate::services::provider_protocol::ProviderProtocol;
    // The hf-local llama-server speaks only OpenAI Chat Completions (like Ollama),
    // so keep it on ServeRouter — never probe `/v1/responses` against it.
    if key.id == crate::services::huggingface::HF_LOCAL_KEY_ID {
        return false;
    }
    if key.is_copilot() {
        return true;
    }
    !is_ollama_base(&key.base_url)
        && matches!(
            provider_profile_for_key(key).default_protocol,
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
        )
}

/// Start the responses-capable internal proxy (`ResponsesToChatRouter`) bound to
/// `key` — the same engine native codex/opencode/pi use. It speaks `/responses`
/// upstream (or chat, learned per `(plugin,key,model)` and persisted to config on
/// exit), so a plugin driving a gpt-5.x model with reasoning + tools works. The
/// loopback bearer is enforced and buffered token usage is accounted.
async fn start_responses_endpoint(
    key: &ApiKey,
    store: &SessionStore,
    tool: &str,
    run_tally: Arc<RunTokenTally>,
) -> anyhow::Result<EndpointHandle> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::provider_profile::{provider_profile_for_key, resolve_starter_base_url};
    use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let token = random_auth_token();
    let profile = provider_profile_for_key(key);
    // Copilot is served like native codex's copilot router: the token manager
    // exchanges the GitHub token and supplies the upstream base URL, so the
    // config carries no static base/key. Everyone else targets the resolved REST
    // base directly.
    let (base_url, api_key, target_protocol, copilot_token_manager, is_starter) =
        if key.is_copilot() {
            (
                String::new(),
                String::new(),
                ProviderProtocol::Openai,
                Some(Arc::new(CopilotTokenManager::new(
                    key.key.as_str().to_string(),
                ))),
                false,
            )
        } else {
            let base = resolve_starter_base_url(&key.base_url);
            let proto = detect_provider_protocol(&base);
            (
                base,
                key.key.as_str().to_string(),
                proto,
                None,
                profile.serve_flags.is_starter,
            )
        };
    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        target_path_variant: None,
        copilot_token_manager,
        model_prefix: None,
        requires_reasoning_content: key.requires_reasoning_content.unwrap_or(false),
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: key.responses_api_supported,
        is_starter,
        aivo_prefix_models: Vec::new(),
    })
    .with_tool(PLUGIN_ROUTE_TOOL)
    .with_seed_routes(key.routes_for_tool(PLUGIN_ROUTE_TOOL))
    .with_usage_accounting(store.clone(), key.id.clone(), tool.to_string())
    .with_run_tally(run_tally)
    .with_auth_token(token.clone());
    let (port, route_cache, learned, handle) = router.start_background().await?;
    Ok(
        EndpointHandle::new(port, token, handle, None).with_persist(RoutePersist {
            store: store.clone(),
            key: key.clone(),
            route_cache,
            learned,
        }),
    )
}

// ── coding-agent run accounting (mirrors ai_launcher's LogEvent pair) ────────

/// The shared fields of the launch's log rows, captured once so the finished
/// row is the started template with the phase/exit/duration flipped.
struct Accounting {
    base: LogEvent,
    started: Instant,
    /// This run's endpoint token usage, read onto the finished row at the end.
    run_tally: Arc<RunTokenTally>,
}

async fn begin_accounting(
    store: &SessionStore,
    name: &str,
    key: Option<&ApiKey>,
    model: Option<&str>,
    args: &[String],
    run_tally: Arc<RunTokenTally>,
) -> Accounting {
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string());
    let base = LogEvent {
        source: "run".to_string(),
        kind: "tool_launch".to_string(),
        event_group_id: Some(new_log_id()),
        key_id: key.map(|k| k.id.clone()),
        key_name: key.map(|k| k.display_name().to_string()),
        base_url: key.map(|k| k.base_url.clone()),
        tool: Some(name.to_string()),
        model: model.map(|m| m.to_string()),
        cwd,
        title: Some(match key {
            Some(k) => format!("{name} · {}", k.display_name()),
            None => name.to_string(),
        }),
        ..Default::default()
    };

    let _ = store
        .logs()
        .append(LogEvent {
            phase: Some("started".to_string()),
            body_text: Some(args.join(" ")),
            ..base.clone()
        })
        .await;
    if let Some(k) = key {
        let _ = store.record_selection(&k.id, name, model).await;
    }

    Accounting {
        base,
        started: Instant::now(),
        run_tally,
    }
}

async fn finish_accounting(store: &SessionStore, acct: Accounting, code: i32, duration: Duration) {
    // Stamp the run's endpoint token usage onto the finished row (a no-op zero for
    // Cursor/OAuth runs), so a probe-less coding-agent plugin is windowable under
    // `aivo stats --since` — its lifetime per-key counters carry no timestamp.
    let (prompt, completion, cache_read, cache_creation) = acct.run_tally.snapshot();
    let some_positive = |v: u64| (v > 0).then_some(v as i64);
    let _ = store
        .logs()
        .append(LogEvent {
            phase: Some("finished".to_string()),
            exit_code: Some(code as i64),
            duration_ms: Some(duration.as_millis() as i64),
            input_tokens: some_positive(prompt),
            output_tokens: some_positive(completion),
            cache_read_input_tokens: some_positive(cache_read),
            cache_creation_input_tokens: some_positive(cache_creation),
            ..acct.base
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::{
        HelpKind, PluginServe, consent_drifted, help_kind, missing_grants,
        model_limit_env_with_cache, plugin_model_request, plugin_serve, retained_grants,
        take_hf_ref, use_responses_router,
    };
    use crate::plugin::manifest::grantable_capabilities;
    use crate::plugin::registry::PluginRecord;
    use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
    use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
    use crate::services::cursor_acp::CURSOR_ACP_SENTINEL;
    use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;
    use crate::services::huggingface::{HF_LOCAL_KEY_ID, HfModelRef, local_takeover_key};
    use crate::services::session_store::ApiKey;

    fn key(base: &str) -> ApiKey {
        ApiKey::new_with_protocol("id".into(), "n".into(), base.into(), None, "secret".into())
    }

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn consented_rec(source: &str, checksum: Option<&str>, approved: Option<&str>) -> PluginRecord {
        PluginRecord {
            source: source.to_string(),
            checksum: checksum.map(|s| s.to_string()),
            manifest: None,
            installed_at: None,
            granted_caps: vec!["endpoint".to_string()],
            run_approved: true,
            approved_checksum: approved.map(|s| s.to_string()),
        }
    }

    #[test]
    fn consent_drift_detection() {
        // Pin matches → no drift.
        let rec = consented_rec("github:o/aivo-x", Some("sha256:v1"), Some("sha256:v1"));
        assert!(!consent_drifted(&rec));
        // Bytes changed since approval → drift.
        let rec = consented_rec("github:o/aivo-x", Some("sha256:v2"), Some("sha256:v1"));
        assert!(consent_drifted(&rec));
        // Local path is the user's own file — never re-gated.
        let rec = consented_rec("/abs/aivo-x", Some("sha256:v2"), Some("sha256:v1"));
        assert!(!consent_drifted(&rec));
        // Missing pins prove nothing (legacy records) → pass.
        let rec = consented_rec("github:o/aivo-x", Some("sha256:v2"), None);
        assert!(!consent_drifted(&rec));
        let rec = consented_rec("github:o/aivo-x", None, Some("sha256:v1"));
        assert!(!consent_drifted(&rec));
        // No consent to protect → nothing to drift.
        let mut rec = consented_rec("github:o/aivo-x", Some("sha256:v2"), Some("sha256:v1"));
        rec.run_approved = false;
        rec.granted_caps.clear();
        assert!(!consent_drifted(&rec));
    }

    #[tokio::test]
    async fn model_limit_env_emits_known_limits_only() {
        let dir = tempfile::TempDir::new().unwrap();
        let cache = crate::services::models_cache::ModelsCache::with_path(
            dir.path().join("models-cache.json"),
        );
        let k = key("https://api.example.com");
        // Snapshot-known model → both advisory vars.
        let env = model_limit_env_with_cache(&cache, &k, "claude-sonnet-4-6").await;
        assert!(env.contains(&(
            "AIVO_MODEL_CONTEXT_WINDOW".to_string(),
            "1000000".to_string()
        )));
        assert!(env.contains(&(
            "AIVO_MODEL_MAX_OUTPUT_TOKENS".to_string(),
            "64000".to_string()
        )));
        // Unknown model → nothing (plugins keep their own defaults).
        let env = model_limit_env_with_cache(&cache, &k, "totally-unknown-model-xyz").await;
        assert!(env.is_empty());
    }

    #[test]
    fn help_kind_distinguishes_top_level_sub_and_none() {
        // `aivo <plugin> --help` / `-h` → top-level (gets the banner).
        assert!(matches!(help_kind(&argv(&["--help"])), HelpKind::TopLevel));
        assert!(matches!(help_kind(&argv(&["-h"])), HelpKind::TopLevel));
        // `aivo <plugin> <sub> --help` → sub-help (passthrough, no banner).
        assert!(matches!(
            help_kind(&argv(&["trust", "--help"])),
            HelpKind::Sub
        ));
        // No help flag, or `--help` only as a value, is not a help request.
        assert!(matches!(help_kind(&argv(&["-p", "hi"])), HelpKind::None));
        assert!(matches!(
            help_kind(&argv(&["-p", "explain --help"])),
            HelpKind::None
        ));
    }

    #[test]
    fn plugin_serve_picks_the_right_router() {
        // Plain REST keys (and Copilot, which the serve proxy exchanges) go to
        // ServeRouter.
        for base in [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "copilot",
        ] {
            assert_eq!(plugin_serve(&key(base)), PluginServe::Serve, "{base}");
        }
        // Cursor has its own ACP-backed router for coding-agent plugins.
        assert_eq!(plugin_serve(&key(CURSOR_ACP_SENTINEL)), PluginServe::Cursor);
        assert!(plugin_serve(&key(CURSOR_ACP_SENTINEL)).is_servable());
        // Provider OAuth (grok/codex) is servable.
        for base in [
            crate::services::grok_oauth::GROK_OAUTH_SENTINEL,
            CODEX_OAUTH_SENTINEL,
        ] {
            assert_eq!(plugin_serve(&key(base)), PluginServe::Serve, "{base}");
            assert!(plugin_serve(&key(base)).is_servable(), "{base}");
        }
        // Single-CLI OAuth credentials are native-agent-only — blocked.
        for base in [CLAUDE_OAUTH_SENTINEL, GEMINI_OAUTH_SENTINEL] {
            assert_eq!(plugin_serve(&key(base)), PluginServe::Blocked, "{base}");
            assert!(!plugin_serve(&key(base)).is_servable(), "{base}");
        }
    }

    #[test]
    fn responses_router_for_openai_rest_and_copilot() {
        // OpenAI-protocol REST keys + Copilot → responses-capable internal proxy
        // (Copilot's token exchange runs inside that router).
        assert!(use_responses_router(&key("https://api.openai.com/v1")));
        assert!(use_responses_router(&key("https://openrouter.ai/api/v1")));
        assert!(use_responses_router(&key("copilot")));
        // Anthropic-protocol and Ollama keep the serve engine.
        assert!(!use_responses_router(&key("https://api.anthropic.com")));
        assert!(!use_responses_router(&key("ollama")));
    }

    #[test]
    fn lazy_probe_preserves_prior_grants_without_auto_escalation() {
        let capabilities = vec![
            "endpoint".to_string(),
            "endpoint".to_string(),
            "config-read".to_string(),
            "config-write".to_string(),
        ];
        let requested = grantable_capabilities(&capabilities);
        let prior = vec![
            "endpoint".to_string(),
            "config-read".to_string(),
            "old-cap".to_string(),
        ];

        assert_eq!(requested, ["endpoint"]);
        assert_eq!(retained_grants(&requested, &prior), ["endpoint"]);
        assert!(missing_grants(&requested, &prior).is_empty());
    }

    #[test]
    fn explicit_key_without_model_forces_model_picker() {
        assert_eq!(
            plugin_model_request(None, true).as_deref(),
            Some(""),
            "-k without -m must select a model after selecting the key"
        );
        assert_eq!(
            plugin_model_request(Some("gpt-4o".to_string()), true).as_deref(),
            Some("gpt-4o"),
            "an explicit -m value still wins"
        );
        assert!(plugin_model_request(None, false).is_none());
    }

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn take_hf_ref_prefers_explicit_model_flag() {
        // `-m hf:…` is the ref; positional args are left untouched.
        let mut rest = args(&["-p", "hi"]);
        assert_eq!(
            take_hf_ref(Some("hf:owner/repo"), &mut rest, true).as_deref(),
            Some("hf:owner/repo")
        );
        assert_eq!(rest, args(&["-p", "hi"]));
        // A normal `-m` suppresses any positional lift entirely.
        let mut rest = args(&["hf:owner/repo"]);
        assert!(take_hf_ref(Some("gpt-4o"), &mut rest, true).is_none());
        assert_eq!(rest, args(&["hf:owner/repo"]));
    }

    #[test]
    fn take_hf_ref_lifts_positional_out_of_argv() {
        // `aivo <plugin> hf:…` with no `-m`: the ref is lifted out of the argv so
        // the wrapped tool never sees the (uninterpretable) `hf:` token.
        let mut rest = args(&["hf:owner/repo", "explain this"]);
        assert_eq!(
            take_hf_ref(None, &mut rest, true).as_deref(),
            Some("hf:owner/repo")
        );
        assert_eq!(rest, args(&["explain this"]));
        // A `.gguf` path is lifted the same way.
        let mut rest = args(&["./model.gguf"]);
        assert_eq!(
            take_hf_ref(None, &mut rest, true).as_deref(),
            Some("./model.gguf")
        );
        assert!(rest.is_empty());
        // No hf ref anywhere → no takeover, argv unchanged.
        let mut rest = args(&["just", "a", "prompt"]);
        assert!(take_hf_ref(None, &mut rest, true).is_none());
        assert_eq!(rest, args(&["just", "a", "prompt"]));
    }

    #[test]
    fn take_hf_ref_never_lifts_positionals_for_generic_plugins() {
        // A generic plugin's positionals are real paths/refs (`/abs/path`,
        // `./dir`) that match the hf-path predicate — they must reach the
        // plugin untouched.
        let mut rest = args(&["/etc/hosts", "review this"]);
        assert!(take_hf_ref(None, &mut rest, false).is_none());
        assert_eq!(rest, args(&["/etc/hosts", "review this"]));
        // An explicit `-m hf:…` still takes over regardless.
        let mut rest = args(&["./src"]);
        assert_eq!(
            take_hf_ref(Some("hf:owner/repo"), &mut rest, false).as_deref(),
            Some("hf:owner/repo")
        );
        assert_eq!(rest, args(&["./src"]));
    }

    #[test]
    fn hf_local_key_stays_on_serve_router() {
        // The synthetic llama-server key speaks only OpenAI Chat Completions, so it
        // must use ServeRouter (not the responses router that probes /v1/responses).
        let hf_ref = HfModelRef {
            repo: "DevQuasar/google.gemma-4-E2B".to_string(),
            quant: None,
            file: None,
            revision: None,
            local_source: None,
        };
        let hf_key = local_takeover_key(&hf_ref, 12345);
        assert_eq!(hf_key.id, HF_LOCAL_KEY_ID);
        assert_eq!(plugin_serve(&hf_key), PluginServe::Serve);
        assert!(!use_responses_router(&hf_key));
    }

    #[tokio::test]
    async fn remembered_model_prefers_last_selection_then_chat_model() {
        use crate::constants::MODEL_DEFAULT_PLACEHOLDER;
        use crate::services::session_store::SessionStore;
        use tempfile::TempDir;

        let tmp = TempDir::new().unwrap();
        let store = SessionStore::with_path(tmp.path().join("config.json"));
        let id = store
            .add_key_with_protocol("k", "https://api.openai.com/v1", None, "sk-test")
            .await
            .unwrap();
        store.set_active_key(&id).await.unwrap();
        let k = store.get_key_by_id_info(&id).await.unwrap().unwrap();

        // chat_model only → it's the fallback.
        store.set_code_model(&id, "chat-pinned").await.unwrap();
        assert_eq!(
            super::remembered_plugin_model(&store, &k).await.as_deref(),
            Some("chat-pinned"),
        );

        // A last_selection for this key wins over chat_model — this is what makes a
        // coding-agent plugin agree with native tools on the model for a key.
        store
            .set_last_selection(&k, "pi", Some("sel-model"))
            .await
            .unwrap();
        assert_eq!(
            super::remembered_plugin_model(&store, &k).await.as_deref(),
            Some("sel-model"),
        );

        // The native "let the tool choose" sentinel isn't a concrete model a plugin
        // can run, so it's skipped in favor of the chat_model fallback.
        store
            .set_last_selection(&k, "pi", Some(MODEL_DEFAULT_PLACEHOLDER))
            .await
            .unwrap();
        assert_eq!(
            super::remembered_plugin_model(&store, &k).await.as_deref(),
            Some("chat-pinned"),
        );

        // A last_selection for a *different* key doesn't apply to this key — it
        // still falls back to its own chat_model.
        let other_id = store
            .add_key_with_protocol("o", "https://api.openai.com/v1", None, "sk-o")
            .await
            .unwrap();
        let other = store.get_key_by_id_info(&other_id).await.unwrap().unwrap();
        store
            .set_last_selection(&other, "pi", Some("other-model"))
            .await
            .unwrap();
        assert_eq!(
            super::remembered_plugin_model(&store, &k).await.as_deref(),
            Some("chat-pinned"),
        );
    }
}
