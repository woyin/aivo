//! `aivo plugins` — manage sibling-binary plugins (install/update/list/rm).
//! Plugins are `aivo-<name>` executables in `~/.config/aivo/plugins/`; dispatch
//! lives in `crate::plugin`.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::{
    PluginInstallArgs, PluginRemoveArgs, PluginUpdateArgs, PluginsArgs, PluginsSubcommand,
};
use crate::errors::ExitCode;
use crate::plugin::manifest::{PluginManifest, grantable_capabilities, probe_manifest};
use crate::plugin::registry::{self, PluginRecord};
use crate::plugin::source::{self, SourceKind};
use crate::plugin::{
    PLUGIN_PREFIX, discover, infer_plugin_name, installed_plugins, is_reserved_plugin_name,
    plugins_dir, prompt_capability_grant,
};
use crate::services::system_env::collapse_tilde;
use crate::style;
use chrono::Utc;

const INSTALL_HINT: &str =
    "Install one with `aivo plugins install <source>` (path, url, github:/npm:/cargo:).";

#[derive(Default)]
pub struct PluginsCommand;

impl PluginsCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: PluginsArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(PluginsSubcommand::List);
        let result = match cmd {
            PluginsSubcommand::List => list_action(),
            PluginsSubcommand::Install(a) => install_action(a).await,
            PluginsSubcommand::Update(a) => update_action(a).await,
            PluginsSubcommand::Remove(a) => remove_action(a),
        };
        match result {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {:#}", style::red("Error:"), e);
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    pub fn print_help() {
        println!("{} aivo plugins [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage plugins — sibling `aivo-<name>` binaries under ~/.config/aivo/plugins.\n\
                 Once installed, `aivo <name> …` (or `aivo run <name> …`) runs the plugin."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row(
            "list",
            "Show installed plugins and where each resolves (default)",
        );
        row(
            "install <source> [--name N]",
            "Install from a path, URL, github:owner/repo, npm:pkg, or cargo:crate",
        );
        row(
            "update [name]",
            "Re-install from the recorded source (all plugins if no name)",
        );
        row("rm <name> [-y]", "Remove an installed plugin");
        println!();
        println!("{}", style::bold("Examples:"));
        for ex in [
            "aivo plugins",
            "aivo plugins install ./target/release/aivo-amp",
            "aivo plugins install github:owner/aivo-amp",
            "aivo plugins install npm:aivo-foo",
            "aivo plugins install cargo:aivo-bar",
            "aivo plugins update amp",
            "aivo plugins rm amp",
            "aivo amp --help        # run an installed plugin",
        ] {
            println!("  {}", style::dim(ex));
        }
    }
}

fn list_action() -> Result<ExitCode> {
    let plugins = installed_plugins();
    let managed_dir = plugins_dir();
    let managed_dir_display = managed_dir
        .as_ref()
        .map(|d| collapse_tilde(&d.display().to_string()));

    if plugins.is_empty() {
        eprintln!("  {} No plugins installed.", style::dim("·"));
        eprintln!("  {} {}", style::dim("·"), style::dim(INSTALL_HINT));
        if let Some(dir) = &managed_dir_display {
            eprintln!(
                "  {} {}",
                style::dim("·"),
                style::dim(format!("Plugins live in {dir}"))
            );
        }
        return Ok(ExitCode::Success);
    }

    let records = registry::load().plugins;
    let width = plugins
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .min(24);
    // Continuation lines align under the name: "  ● " (4) + name + "  " (2).
    let indent = " ".repeat(width + 6);
    let sep = style::dim("  ·  ");

    println!(
        "{} {}",
        style::bold("Installed plugins"),
        style::dim(format!("({})", plugins.len()))
    );
    println!();

    for (name, path) in &plugins {
        let is_managed = managed_dir
            .as_deref()
            .is_some_and(|d| path.parent() == Some(d));
        let manifest = records.get(name).and_then(|r| r.manifest.as_ref());

        // Resolve each required binary once (PATH scan), reused for both the
        // status bullet and the detail line.
        let reqs: Vec<(&str, bool)> = manifest
            .map(|m| {
                m.requires
                    .iter()
                    .map(|r| (r.bin.as_str(), bin_on_path(&r.bin)))
                    .collect()
            })
            .unwrap_or_default();

        // Bullet encodes readiness: green = ready to run, yellow = installed but
        // a required binary is missing, dim ○ = no manifest (can't tell).
        let bullet = if manifest.is_none() {
            style::empty_bullet_symbol()
        } else if reqs.iter().any(|(_, ok)| !ok) {
            style::yellow("●")
        } else {
            style::bullet_symbol()
        };

        // Line 1: bullet + name + identity (type · version) + provenance tag.
        let mut ident: Vec<String> = Vec::new();
        if let Some(m) = manifest {
            if let Some(kind) = &m.kind {
                ident.push(kind.to_string());
            }
            ident.push(format!("v{}", m.version));
        }
        let tag = if !is_managed {
            " (external)"
        } else if manifest.is_none() {
            " (no manifest)"
        } else {
            ""
        };
        println!(
            "  {} {}  {}{}",
            bullet,
            style::cyan(format!("{name:<width$}")),
            style::dim(ident.join(" · ")),
            style::dim(tag),
        );

        // Description on its own line — tells you what the plugin actually is.
        if let Some(desc) = manifest.and_then(|m| m.description.as_deref())
            && !desc.is_empty()
        {
            println!("{indent}{}", style::dim(desc));
        }

        // Detail line: granted/declared caps, requirement status, and — for
        // external plugins only — where the binary resolves (managed ones all
        // live in the footer dir, so the path would be noise).
        let mut details: Vec<String> = Vec::new();
        if let Some(m) = manifest {
            let grantable = grantable_capabilities(&m.capabilities);
            let granted = records.get(name).map(|r| {
                grantable
                    .iter()
                    .filter(|c| r.granted_caps.contains(c))
                    .cloned()
                    .collect::<Vec<_>>()
            });
            match granted.as_deref() {
                Some(g) if !g.is_empty() => {
                    details.push(style::dim(format!("caps: {}", g.join(", "))))
                }
                _ if !m.capabilities.is_empty() => details.push(style::dim(format!(
                    "requests: {}",
                    m.capabilities.join(", ")
                ))),
                _ => {}
            }
            if !reqs.is_empty() {
                let marked = reqs
                    .iter()
                    .map(|(bin, ok)| {
                        let mark = if *ok {
                            style::green("✓")
                        } else {
                            style::red("✗")
                        };
                        format!("{} {mark}", style::dim(*bin))
                    })
                    .collect::<Vec<_>>()
                    .join(&style::dim(", "));
                details.push(format!("{} {marked}", style::dim("requires")));
            }
        }
        if !is_managed {
            details.push(style::dim(collapse_tilde(&path.display().to_string())));
        }
        if !details.is_empty() {
            println!("{indent}{}", details.join(&sep));
        }
        println!();
    }

    if let Some(dir) = &managed_dir_display {
        println!("  {}", style::dim(format!("Plugins live in {dir}")));
    }
    Ok(ExitCode::Success)
}

/// Install-on-demand entry used by plugin dispatch when a known plugin is
/// invoked uninstalled. Leaner than `plugins install`: the plugin runs next,
/// so the run-hint/path/disclosure lines would be noise — the dispatch that
/// follows probes the manifest and seeks any capability consent.
pub(crate) async fn install_for_dispatch(name: &str, source: &str) -> Result<()> {
    let dir =
        plugins_dir().context("could not resolve the home directory for ~/.config/aivo/plugins")?;
    let prior = registry::load()
        .plugins
        .get(name)
        .map(|r| r.granted_caps.clone())
        .unwrap_or_default();
    let outcome = reinstall_animated(name, source, &dir, false, "Installing", false).await?;
    let granted = resolve_grants(name, outcome.manifest.as_ref(), &prior, false);
    record_install(
        name,
        source,
        outcome.checksum.clone(),
        outcome.manifest.clone(),
        granted,
    );
    eprintln!("  {} Installed plugin `{name}`", style::success_symbol());
    Ok(())
}

async fn install_action(args: PluginInstallArgs) -> Result<ExitCode> {
    let dir =
        plugins_dir().context("could not resolve the home directory for ~/.config/aivo/plugins")?;

    let name = match args.name {
        Some(n) => n,
        None => {
            // Surface a specific scheme error (e.g. `expected github:owner/repo`)
            // ahead of the generic name-inference failure.
            source::classify(&args.source)?;
            infer_plugin_name(&args.source)
                .context("could not infer a plugin name from the source — pass --name <name>")?
        }
    };
    validate_name(&name)?;
    if is_reserved_plugin_name(&name) {
        anyhow::bail!(
            "`{name}` collides with a built-in command or tool, so it would never run as a plugin. Choose a different --name."
        );
    }

    let target = dir.join(source::plugin_filename(&name));
    if target.exists() && !args.force {
        anyhow::bail!(
            "plugin `{name}` is already installed.\n  \
             Re-fetch it in place with `aivo plugins update {name}`, or pass --force to reinstall (e.g. from a different source)."
        );
    }

    // Stable, re-fetchable source (absolute path for local files) for `update`.
    let source = canonical_source(&args.source);
    // Any caps already granted to this name (a force-reinstall preserves them).
    let prior = registry::load()
        .plugins
        .get(&name)
        .map(|r| r.granted_caps.clone())
        .unwrap_or_default();
    let outcome = reinstall_animated(&name, &source, &dir, false, "Installing", false).await?;

    let version = outcome
        .manifest
        .as_ref()
        .map(|m| format!(" v{}", m.version));
    eprintln!(
        "  {} Installed plugin `{}`{} — run it with {}",
        style::success_symbol(),
        name,
        version.unwrap_or_default(),
        style::cyan(format!("aivo {name}")),
    );
    eprintln!(
        "  {} {}",
        style::dim("·"),
        style::dim(outcome.primary.display().to_string())
    );
    // Seek consent for any grantable caps the manifest requests, then
    // persist the grant alongside the record.
    let granted = resolve_grants(&name, outcome.manifest.as_ref(), &prior, args.trust);
    record_install(
        &name,
        &source,
        outcome.checksum.clone(),
        outcome.manifest.clone(),
        granted.clone(),
    );
    print_disclosure(&outcome, &granted);
    ensure_requirements(outcome.manifest.as_ref()).await;
    Ok(ExitCode::Success)
}

async fn update_action(args: PluginUpdateArgs) -> Result<ExitCode> {
    let dir = plugins_dir().context("could not resolve ~/.config/aivo/plugins")?;
    let records = registry::load().plugins;

    let update_all = args.name.is_none();
    let targets: Vec<String> = match args.name {
        Some(n) => vec![n.strip_prefix(PLUGIN_PREFIX).unwrap_or(&n).to_string()],
        None => records.keys().cloned().collect(),
    };
    if targets.is_empty() {
        eprintln!(
            "  {} No plugins with a recorded source to update.",
            style::dim("·")
        );
        eprintln!("  {} {}", style::dim("·"), style::dim(INSTALL_HINT));
        return Ok(ExitCode::Success);
    }

    // Announce the scope of a bulk run so interleaved fetch lines have context.
    if update_all && targets.len() > 1 {
        eprintln!(
            "{}",
            style::dim(format!("Updating {} plugins…", targets.len()))
        );
    }
    // Align names into a column, matching `list`'s layout.
    let width = targets.iter().map(|n| n.len()).max().unwrap_or(0).min(24);

    let (mut updated, mut unchanged, mut failed) = (0usize, 0usize, 0usize);
    for name in &targets {
        let col = style::cyan(format!("{name:<width$}"));
        let Some(rec) = records.get(name) else {
            failed += 1;
            if discover(name).is_some() {
                eprintln!(
                    "  {} {col}  {}",
                    style::red("✗"),
                    style::dim(
                        "no recorded source (installed manually) — reinstall with `aivo plugins install <source>`"
                    )
                );
            } else {
                eprintln!(
                    "  {} {col}  {}",
                    style::red("✗"),
                    style::dim("not installed")
                );
            }
            continue;
        };

        let source = rec.source.clone();
        let prior_checksum = rec.checksum.clone();
        let prior_version = rec.manifest.as_ref().map(|m| m.version.clone());
        let prior_granted = rec.granted_caps.clone();
        // Re-probe only a plugin aivo has already run (manifest cached), so the
        // displayed version/caps refresh in place without executing a never-run
        // remote binary at update time.
        let reprobe = rec.manifest.is_some();

        // A spinner animates the fetch (TTY); the per-plugin result line below is
        // the lasting signal, so the resolve/download chatter stays muted.
        match reinstall_animated(name, &source, &dir, reprobe, "Updating", true).await {
            Ok(outcome) => {
                // Carry the cached manifest forward when a re-probe yields nothing
                // (best-effort probe failed), so update never wipes it.
                let manifest = outcome.manifest.clone().or_else(|| rec.manifest.clone());
                // Preserve prior grants; only a newly-requested cap prompts (TTY).
                let granted = resolve_grants(name, manifest.as_ref(), &prior_granted, false);
                let new_version = manifest.as_ref().map(|m| m.version.clone());
                // Same bytes as before → nothing actually changed. A missing pin
                // (legacy record) is treated as changed, since we can't prove otherwise.
                let changed = match (&prior_checksum, &outcome.checksum) {
                    (Some(a), Some(b)) => a != b,
                    _ => true,
                };
                record_install(name, &source, outcome.checksum.clone(), manifest, granted);

                if changed {
                    updated += 1;
                    eprintln!(
                        "  {} {col}  {}  {}",
                        style::success_symbol(),
                        version_transition(prior_version.as_deref(), new_version.as_deref()),
                        style::dim(collapse_tilde(&source)),
                    );
                } else {
                    unchanged += 1;
                    let v = new_version.map(|v| format!(" · v{v}")).unwrap_or_default();
                    eprintln!(
                        "  {} {col}  {}{}",
                        style::dim("·"),
                        style::dim("up to date"),
                        style::dim(v),
                    );
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!(
                    "  {} {col}  {}",
                    style::red("✗"),
                    style::dim(format!("{e:#}"))
                );
            }
        }
    }

    if targets.len() > 1 {
        let parts: Vec<String> = [
            (updated, "updated"),
            (unchanged, "unchanged"),
            (failed, "failed"),
        ]
        .into_iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, label)| format!("{n} {label}"))
        .collect();
        if !parts.is_empty() {
            eprintln!();
            eprintln!("  {}", style::dim(parts.join(" · ")));
        }
    }

    if failed > 0 {
        Ok(ExitCode::UserError)
    } else {
        Ok(ExitCode::Success)
    }
}

/// Render a plugin's version change for an `update` result line: `vA → vB` when
/// both are known and differ, the new (or only-known) version otherwise, and a
/// plain `updated` when no manifest version is available either side.
fn version_transition(old: Option<&str>, new: Option<&str>) -> String {
    match (old, new) {
        (Some(o), Some(n)) if o != n => {
            format!("v{o} {} {}", style::dim("→"), style::cyan(format!("v{n}")))
        }
        (_, Some(n)) => style::cyan(format!("v{n}")),
        (Some(o), None) => format!("v{o}"),
        (None, None) => style::dim("updated").to_string(),
    }
}

fn remove_action(args: PluginRemoveArgs) -> Result<ExitCode> {
    let dir = plugins_dir().context("could not resolve ~/.config/aivo/plugins")?;
    let name = args
        .name
        .strip_prefix(PLUGIN_PREFIX)
        .unwrap_or(&args.name)
        .to_string();
    // Binary plugins are `aivo-<name>`; an npm plugin's shim may be `.cmd` on Windows.
    let bin = dir.join(source::plugin_filename(&name));
    let target = if bin.exists() {
        bin
    } else {
        dir.join(format!("{PLUGIN_PREFIX}{name}.cmd"))
    };

    if !target.exists() {
        if let Some(found) = discover(&name) {
            anyhow::bail!(
                "`{name}` isn't managed by aivo — it's at {}. Remove it there (e.g. `cargo uninstall`).",
                found.display()
            );
        }
        anyhow::bail!("plugin `{name}` is not installed. See `aivo plugins list`.");
    }

    if !args.yes && !confirm(&format!("Remove plugin `{name}`?"))? {
        return Ok(ExitCode::Success);
    }

    std::fs::remove_file(&target).with_context(|| format!("removing {}", target.display()))?;
    // npm plugins also leave an `aivo-<name>.d/` payload directory.
    let bundle = dir.join(format!("{PLUGIN_PREFIX}{name}.d"));
    if bundle.is_dir() {
        let _ = std::fs::remove_dir_all(&bundle);
    }
    registry::forget(&name);
    eprintln!("  {} Removed plugin `{name}`", style::success_symbol());
    Ok(ExitCode::Success)
}

/// Interactive y/N prompt; bails non-interactively (pass `--yes`).
fn confirm(prompt: &str) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        anyhow::bail!("{prompt} (non-interactive; pass --yes to confirm)");
    }
    eprint!("  {} {prompt} [y/N] ", style::yellow("?"));
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

/// Reject names that can't dispatch: empty, flag-shaped, or containing a path
/// separator (which would escape the plugins dir / break the `aivo-<name>` map).
fn validate_name(name: &str) -> Result<()> {
    if name.starts_with('-') {
        anyhow::bail!("plugin name `{name}` must not start with `-`");
    }
    if name.contains('/') || name.contains('\\') {
        anyhow::bail!("plugin name `{name}` must not contain a path separator");
    }
    Ok(())
}

/// Outcome of materializing a source: where it installed, the integrity pin, and
/// the probed manifest (local installs only). Shared by install and update.
struct InstallOutcome {
    primary: PathBuf,
    checksum: Option<String>,
    manifest: Option<PluginManifest>,
}

/// Resolve the source (local path / URL / `github:` / `npm:` / `cargo:`), install
/// `aivo-<name>` into `dir`, and probe for a manifest. At install the probe runs
/// for **local-path installs only** — aivo doesn't execute a freshly-fetched
/// remote binary just to read its manifest; such plugins are recorded
/// manifest-less and probed lazily on first dispatch (see `crate::plugin::endpoint`).
/// `reprobe` lifts that on `update` for a plugin aivo has already run (its
/// manifest is cached), so the refreshed version/caps are picked up in place —
/// the trust surface is the same as the next dispatch would be.
/// `reinstall` with terminal-appropriate progress feedback: an animated spinner
/// on a TTY — erased when done, leaving only the result line the caller prints —
/// and discrete `·` lines otherwise. The spinner is skipped for instant local
/// copies and for `cargo:` builds (cargo prints its own output, which a spinner
/// would fight). Off-TTY `install` keeps its lines (useful in CI logs); off-TTY
/// `update` stays silent (it reports its own per-plugin result).
async fn reinstall_animated(
    name: &str,
    source: &str,
    dir: &Path,
    reprobe: bool,
    verb: &str,
    is_update: bool,
) -> Result<InstallOutcome> {
    let tty = std::io::stderr().is_terminal();
    let kind = source::classify(source).ok();
    let instant_or_noisy = matches!(
        kind,
        Some(SourceKind::LocalPath) | Some(SourceKind::Cargo { .. })
    );
    let spin = tty && !instant_or_noisy;
    // Mute materialize's own lines while a spinner animates, and off-TTY for
    // `update` (keeping piped output clean); `install` keeps them off-TTY.
    let quiet = spin || (is_update && !tty && !instant_or_noisy);

    if !spin {
        return reinstall(name, source, dir, reprobe, quiet).await;
    }
    let label = format!(" {verb} {name}…");
    let (spinning, handle) = style::start_spinner(Some(label.as_str()));
    let result = reinstall(name, source, dir, reprobe, quiet).await;
    style::stop_spinner(&spinning);
    let _ = handle.await;
    result
}

async fn reinstall(
    name: &str,
    source: &str,
    dir: &Path,
    reprobe: bool,
    quiet: bool,
) -> Result<InstallOutcome> {
    let materialized = source::materialize(source, name, dir, quiet).await?;
    let manifest = if materialized.trusted_local || reprobe {
        probe_manifest(&materialized.primary, name).await
    } else {
        None
    };
    Ok(InstallOutcome {
        primary: materialized.primary,
        checksum: materialized.checksum,
        manifest,
    })
}

/// A stable, re-fetchable form of the install source: scheme strings (`github:`,
/// `npm:`, `cargo:`, URLs) verbatim so `update` re-resolves; local paths made
/// absolute so `update` works regardless of the current directory.
fn canonical_source(source: &str) -> String {
    match source::classify(source) {
        Ok(SourceKind::LocalPath) | Err(_) => std::fs::canonicalize(source)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| source.to_string()),
        _ => source.to_string(),
    }
}

// ── Registry write + install disclosure ────────────────────────────────────

/// Persist provenance (source + checksum + manifest + timestamp + granted caps)
/// so `update` can re-fetch and dispatch knows what to hand over. See
/// `crate::plugin::registry`.
fn record_install(
    name: &str,
    source: &str,
    checksum: Option<String>,
    manifest: Option<PluginManifest>,
    granted_caps: Vec<String>,
) {
    // An install-time manifest probe already executed the binary with the
    // user's explicit consent, so the first-dispatch run gate is satisfied.
    let run_approved = manifest.is_some();
    registry::record(
        name,
        PluginRecord {
            source: source.to_string(),
            checksum,
            manifest,
            installed_at: Some(Utc::now().to_rfc3339()),
            granted_caps,
            run_approved,
        },
    );
}

/// Decide which grantable caps to grant. With `auto_grant` (`--trust`),
/// grants all requested without prompting; otherwise prompts (TTY only) for caps
/// not already approved, never escalating silently. A manifest-less plugin
/// (remote install) keeps its prior grant — the first dispatch probes + prompts
/// instead (see `crate::plugin::endpoint`).
fn resolve_grants(
    name: &str,
    manifest: Option<&PluginManifest>,
    prior: &[String],
    auto_grant: bool,
) -> Vec<String> {
    let Some(m) = manifest else {
        return prior.to_vec();
    };
    let requested = grantable_capabilities(&m.capabilities);
    if requested.is_empty() {
        return Vec::new();
    }
    if requested.iter().all(|c| prior.contains(c)) {
        return requested; // already consented to everything requested
    }
    if auto_grant || (std::io::stdin().is_terminal() && prompt_capability_grant(name, &requested)) {
        requested
    } else {
        // Keep only previously-granted caps still requested; no silent escalation.
        requested
            .into_iter()
            .filter(|c| prior.contains(c))
            .collect()
    }
}

/// True when `bin` resolves on `$PATH`.
fn bin_on_path(bin: &str) -> bool {
    crate::services::path_search::find_in_dirs(
        bin,
        &crate::services::path_search::collect_path_dirs(),
    )
    .is_some()
}

/// After install, check the plugin's declared `requires`: for each missing
/// executable, offer to run its (plugin-authored) install command — the same
/// consent-gated flow native tools get — or print a hint. aivo never invents the
/// command; it only runs what the plugin declared, after showing it.
async fn ensure_requirements(manifest: Option<&PluginManifest>) {
    let Some(m) = manifest else { return };
    for req in &m.requires {
        if bin_on_path(&req.bin) {
            continue;
        }
        eprintln!(
            "  {} this plugin needs `{}`, which isn't on your PATH.",
            style::yellow("!"),
            req.bin,
        );
        let Some(cmd) = &req.install else {
            eprintln!(
                "    {}",
                style::dim(format!("install `{}` and re-run.", req.bin))
            );
            continue;
        };
        eprintln!("    {}", style::dim(format!("install command: {cmd}")));
        // Non-interactive (CI) → just leave the hint; don't run installers blind.
        if !std::io::stdin().is_terminal() {
            continue;
        }
        if confirm(&format!("Run it to install `{}`?", req.bin)).unwrap_or(false) {
            run_install_command(cmd, &req.bin).await;
        }
    }
}

/// Run a plugin-declared install command with inherited stdio (consent already
/// given by the caller).
async fn run_install_command(cmd: &str, bin: &str) {
    eprintln!("  {} Installing `{bin}`...", style::arrow_symbol());
    let mut command = if cfg!(windows) {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    } else {
        let mut c = tokio::process::Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    };
    match command.status().await {
        Ok(s) if s.success() => eprintln!("  {} `{bin}` installed.", style::success_symbol()),
        Ok(_) => eprintln!(
            "  {} install command exited non-zero — install `{bin}` manually.",
            style::yellow("!")
        ),
        Err(e) => eprintln!(
            "  {} couldn't run the install command: {e}",
            style::yellow("!")
        ),
    }
}

/// Surface what a freshly-installed plugin declared, and what was granted.
fn print_disclosure(outcome: &InstallOutcome, granted: &[String]) {
    let Some(m) = &outcome.manifest else {
        eprintln!(
            "  {} {}",
            style::dim("·"),
            style::dim("no manifest — runs as a plain subcommand")
        );
        return;
    };
    let mut bits = vec![format!("v{}", m.version)];
    if let Some(t) = &m.kind {
        bits.push(format!("type: {t}"));
    }
    if !m.roles.is_empty() {
        bits.push(format!("roles: {}", m.roles.join(", ")));
    }
    if !m.capabilities.is_empty() {
        bits.push(format!("requests: {}", m.capabilities.join(", ")));
    }
    eprintln!("  {} {}", style::dim("·"), style::dim(bits.join("  ·  ")));
    if !granted.is_empty() {
        eprintln!(
            "    {}",
            style::dim(format!("granted: {}", granted.join(", ")))
        );
    } else if !grantable_capabilities(&m.capabilities).is_empty() {
        eprintln!(
            "    {}",
            style::dim("no capabilities granted — reinstall interactively to grant")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with_caps(caps: &[&str]) -> PluginManifest {
        PluginManifest {
            name: "amp".to_string(),
            version: "1".to_string(),
            protocol: "1".to_string(),
            description: None,
            kind: None,
            roles: Vec::new(),
            documents_aivo_flags: false,
            capabilities: caps.iter().map(|c| c.to_string()).collect(),
            hooks: Vec::new(),
            homepage: None,
            transcripts: None,
            requires: Vec::new(),
        }
    }

    #[test]
    fn version_transition_renders_each_case() {
        // Both known and differ → an arrowed transition carrying both versions.
        let t = version_transition(Some("1.0.0"), Some("1.1.0"));
        assert!(
            t.contains("1.0.0") && t.contains("1.1.0") && t.contains('→'),
            "{t}"
        );
        // Only the new version known → just the new version.
        assert!(version_transition(None, Some("2.0.0")).contains("2.0.0"));
        // Bytes changed but the version string is identical → show it once, no arrow.
        let same = version_transition(Some("1.0.0"), Some("1.0.0"));
        assert!(same.contains("1.0.0") && !same.contains('→'), "{same}");
        // No version either side → a plain "updated".
        assert!(version_transition(None, None).contains("updated"));
    }

    #[test]
    fn resolve_grants_ignores_reserved_capabilities() {
        let manifest = manifest_with_caps(&["config-read", "endpoint", "config-write"]);
        assert_eq!(
            resolve_grants("amp", Some(&manifest), &[], true),
            ["endpoint"]
        );

        let prior = vec!["config-read".to_string(), "endpoint".to_string()];
        assert_eq!(
            resolve_grants("amp", Some(&manifest), &prior, false),
            ["endpoint"]
        );
    }
}
