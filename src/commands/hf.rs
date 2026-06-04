//! `aivo hf` — inspect and manage cached HuggingFace GGUF files.

use std::io::{IsTerminal, Write};

use anyhow::Result;

use crate::cli::{HfArgs, HfCleanArgs, HfListArgs, HfPullArgs, HfRmArgs, HfSubcommand};
use crate::errors::ExitCode;
use crate::services::huggingface::{
    self, CachedModel, CachedRepo, ensure_cached_refresh, format_modified_ago, format_size,
    is_hf_or_local_gguf, list_cached_models, list_cached_repos, parse_hf_ref, remove_all_cached,
    remove_cached_repo,
};
use crate::style;

#[derive(Default)]
pub struct HfCommand;

impl HfCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: HfArgs) -> ExitCode {
        let cmd = args
            .command
            .unwrap_or(HfSubcommand::List(HfListArgs { verbose: false }));
        let result = match cmd {
            HfSubcommand::List(a) => list_action(a),
            HfSubcommand::Pull(a) => pull_action(a).await,
            HfSubcommand::Rm(a) => rm_action(a),
            HfSubcommand::Clean(a) => clean_action(a),
        };
        match result {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {:#}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    pub fn print_help() {
        println!("{} aivo hf [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage cached HuggingFace GGUF files under ~/.config/aivo/cache/huggingface."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<24}", a)), style::dim(b));
        };
        row(
            "list [--verbose]",
            "Show cached repos (default; or files with --verbose)",
        );
        row(
            "pull <ref|path>",
            "Download a HF GGUF, or import a local `.gguf` into the cache",
        );
        row(
            "rm <repo> [--quant <q>]",
            "Delete one quant (or whole repo with --all)",
        );
        row("clean [-y]", "Delete every cached repo");
        println!();
        println!("{}", style::bold("Examples:"));
        for ex in [
            "aivo hf",
            "aivo hf list --verbose",
            "aivo hf pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "aivo hf pull hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M",
            "aivo hf pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF --refresh",
            "aivo hf pull ~/Downloads/Llama-3.2-3B-Instruct-Q5_K_M.gguf",
            "aivo hf pull ./my-model.gguf --as me/my-model",
            "aivo hf rm bartowski/Llama-3.2-3B-Instruct-GGUF --quant Q5_K_M",
            "aivo hf rm bartowski/Llama-3.2-3B-Instruct-GGUF --all -y",
            "aivo hf clean -y",
        ] {
            println!("  {}", style::dim(ex));
        }
        println!();
        println!(
            "{}",
            style::dim(
                "Gated/private repos (e.g. google/gemma-*) need an HF token: set HF_TOKEN or run `huggingface-cli login`."
            )
        );
    }
}

fn list_action(args: HfListArgs) -> Result<ExitCode> {
    if args.verbose {
        list_verbose()
    } else {
        list_summary()
    }
}

fn list_summary() -> Result<ExitCode> {
    let repos = list_cached_repos();
    if repos.is_empty() {
        print_empty_hint();
        return Ok(ExitCode::Success);
    }

    let repo_w = repos
        .iter()
        .map(|r| r.repo.len())
        .max()
        .unwrap_or(0)
        .min(60);
    let total: u64 = repos.iter().map(|r| r.total_bytes).sum();

    for r in &repos {
        let quant_summary = quant_summary(r);
        let age = format_modified_ago(r.modified);
        println!(
            "  {:<repo_w$}  {:<10}  {:>9}  used {}",
            r.repo,
            quant_summary,
            format_size(r.total_bytes),
            age,
            repo_w = repo_w,
        );
    }
    print_total_footer(repos.len(), total);
    Ok(ExitCode::Success)
}

fn list_verbose() -> Result<ExitCode> {
    let models = list_cached_models();
    if models.is_empty() {
        print_empty_hint();
        return Ok(ExitCode::Success);
    }

    let repo_w = models
        .iter()
        .map(|m| m.repo.len())
        .max()
        .unwrap_or(0)
        .min(60);
    let file_w = models
        .iter()
        .map(|m| m.filename.len())
        .max()
        .unwrap_or(0)
        .min(50);
    // Only render the revision column when at least one entry is non-main.
    let any_non_main = models.iter().any(|m| m.revision != "main");
    let total: u64 = models.iter().map(|m| m.size_bytes).sum();

    let mut seen_repos: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for m in &models {
        seen_repos.insert(m.repo.as_str());
    }

    for m in &models {
        let quant = m.quant().unwrap_or_else(|| "?".into());
        let age = format_modified_ago(m.modified);
        if any_non_main {
            println!(
                "  {:<repo_w$}  {:<file_w$}  {:<8}  @{:<8}  {:>9}  used {}",
                m.repo,
                m.filename,
                quant,
                m.revision,
                format_size(m.size_bytes),
                age,
                repo_w = repo_w,
                file_w = file_w,
            );
        } else {
            println!(
                "  {:<repo_w$}  {:<file_w$}  {:<8}  {:>9}  used {}",
                m.repo,
                m.filename,
                quant,
                format_size(m.size_bytes),
                age,
                repo_w = repo_w,
                file_w = file_w,
            );
        }
    }
    print_total_footer(seen_repos.len(), total);
    Ok(ExitCode::Success)
}

fn print_total_footer(repo_count: usize, total: u64) {
    println!(
        "  {}",
        style::dim(format!(
            "{} model{}, {} total — {}",
            repo_count,
            if repo_count == 1 { "" } else { "s" },
            format_size(total),
            huggingface::cache_root()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        ))
    );
}

fn print_empty_hint() {
    eprintln!("  {} No HuggingFace models cached yet.", style::dim("·"));
    eprintln!(
        "  {} {}",
        style::dim("·"),
        style::dim("Run e.g. `aivo hf pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF`.")
    );
}

/// `Q4_K_M` for a single quant, `Q4_K_M+2` for multiple.
fn quant_summary(r: &CachedRepo) -> String {
    let mut quants: Vec<String> = r.files.iter().filter_map(|f| f.quant()).collect();
    quants.sort();
    quants.dedup();
    match quants.len() {
        0 => "?".into(),
        1 => quants.remove(0),
        n => {
            let first = quants[0].clone();
            format!("{first}+{}", n - 1)
        }
    }
}

async fn pull_action(args: HfPullArgs) -> Result<ExitCode> {
    let reference = args.reference;
    if !is_hf_or_local_gguf(&reference) {
        anyhow::bail!(
            "`{reference}` is not a HuggingFace reference or local `.gguf` path. \
             Pass `hf:<owner>/<repo>[:<quant>]`, a https://huggingface.co/<owner>/<repo> URL, or a path ending in `.gguf`."
        );
    }
    let mut hf_ref = parse_hf_ref(&reference)?;
    if let Some(as_repo) = args.as_repo {
        apply_as_repo(&mut hf_ref, as_repo)?;
    }

    let was_local = hf_ref.is_local();
    let cached = ensure_cached_refresh(&hf_ref, args.refresh).await?;
    let launch_ref = cached.launch_ref();
    if cached.was_cached {
        eprintln!(
            "  {} `{}` is already cached ({}).",
            style::dim("·"),
            cached.repo,
            format_size(cached.size_bytes)
        );
    } else {
        eprintln!(
            "  {} {} `{}` ({}).",
            style::success_symbol(),
            if was_local { "Imported" } else { "Cached" },
            cached.repo,
            format_size(cached.size_bytes),
        );
    }
    eprintln!(
        "  {} Launch with {}",
        style::dim("·"),
        style::cyan(format!("-m {launch_ref}"))
    );
    Ok(ExitCode::Success)
}

fn apply_as_repo(hf_ref: &mut huggingface::HfModelRef, as_repo: String) -> Result<()> {
    if !hf_ref.is_local() {
        anyhow::bail!("`--as` only applies to local file imports, not `hf:` / URL refs");
    }
    let segments: Vec<&str> = as_repo.split('/').collect();
    if segments.len() != 2 || segments[0].is_empty() || segments[1].is_empty() {
        anyhow::bail!("`--as` expects `<owner>/<repo>`, got `{as_repo}`");
    }
    hf_ref.repo = as_repo;
    Ok(())
}

fn rm_action(args: HfRmArgs) -> Result<ExitCode> {
    let repos = list_cached_repos();
    let Some(target) = repos.iter().find(|r| r.repo == args.repo) else {
        anyhow::bail!(
            "No cached files for `{}`. Run `aivo hf list` to see what's cached.",
            args.repo
        );
    };

    if let Some(quant) = &args.quant {
        return rm_one_quant(target, quant, args.yes);
    }
    if args.all {
        return rm_whole_repo(target, args.yes);
    }
    // Auto-delete a single-file repo; refuse with available quants otherwise.
    if target.files.len() == 1 {
        return rm_whole_repo(target, args.yes);
    }

    let mut quants: Vec<String> = target.files.iter().filter_map(|f| f.quant()).collect();
    quants.sort();
    quants.dedup();
    let quants_hint = if quants.is_empty() {
        "(no recognizable quant tags)".to_string()
    } else {
        quants.join(", ")
    };
    anyhow::bail!(
        "`{}` has {} cached file{}: {}. Pass `--quant <q>` to remove one, or `--all` for the whole repo.",
        args.repo,
        target.files.len(),
        if target.files.len() == 1 { "" } else { "s" },
        quants_hint
    );
}

fn rm_one_quant(target: &CachedRepo, quant: &str, yes: bool) -> Result<ExitCode> {
    let upper = quant.to_ascii_uppercase();
    let matches: Vec<&CachedModel> = target
        .files
        .iter()
        .filter(|f| f.quant().as_deref() == Some(upper.as_str()))
        .collect();
    let file = match matches.as_slice() {
        [] => {
            let available: Vec<String> = target.files.iter().filter_map(|f| f.quant()).collect();
            anyhow::bail!(
                "No `{quant}` quant cached for `{}`. Available: {}.",
                target.repo,
                if available.is_empty() {
                    "(none)".to_string()
                } else {
                    available.join(", ")
                }
            );
        }
        [only] => *only,
        many => {
            // Multiple files share the same quant (different revisions
            // or nested paths). Refuse rather than picking one
            // arbitrarily — the user has to disambiguate.
            let listing: Vec<String> = many
                .iter()
                .map(|f| {
                    if f.revision == "main" {
                        format!("    {}", f.filename)
                    } else {
                        format!("    {} @ {}", f.filename, f.revision)
                    }
                })
                .collect();
            anyhow::bail!(
                "`{}` has {} files matching quant `{quant}`:\n{}\nDelete one with `aivo hf rm <repo> --all`, or remove manually under `aivo hf list --verbose`'s reported path.",
                target.repo,
                many.len(),
                listing.join("\n"),
            );
        }
    };

    if !yes
        && !confirm(&format!(
            "Remove {} ({}) from `{}`?",
            file.filename,
            format_size(file.size_bytes),
            target.repo
        ))?
    {
        return Ok(ExitCode::Success);
    }

    let size = file.size_bytes;
    std::fs::remove_file(&file.path)
        .map_err(|e| anyhow::anyhow!("Failed to remove {}: {e}", file.path.display()))?;
    if let Some(parent) = file.path.parent() {
        prune_if_empty(parent);
    }
    eprintln!(
        "  {} Removed {} from `{}` ({})",
        style::success_symbol(),
        file.filename,
        target.repo,
        format_size(size)
    );
    Ok(ExitCode::Success)
}

fn rm_whole_repo(target: &CachedRepo, yes: bool) -> Result<ExitCode> {
    let count = target.files.len();
    let prompt = if count <= 1 {
        format!(
            "Remove cached `{}` ({})?",
            target.repo,
            format_size(target.total_bytes)
        )
    } else {
        format!(
            "Remove all {count} cached files under `{}` ({})?",
            target.repo,
            format_size(target.total_bytes)
        )
    };
    if !yes && !confirm(&prompt)? {
        return Ok(ExitCode::Success);
    }
    let freed = remove_cached_repo(&target.repo)?;
    eprintln!(
        "  {} Removed {} ({})",
        style::success_symbol(),
        target.repo,
        format_size(freed)
    );
    Ok(ExitCode::Success)
}

fn prune_if_empty(dir: &std::path::Path) {
    if let Ok(mut entries) = std::fs::read_dir(dir)
        && entries.next().is_none()
    {
        let _ = std::fs::remove_dir(dir);
    }
}

fn clean_action(args: HfCleanArgs) -> Result<ExitCode> {
    let repos = list_cached_repos();
    if repos.is_empty() {
        eprintln!("  {} Cache is already empty.", style::dim("·"));
        return Ok(ExitCode::Success);
    }
    let total: u64 = repos.iter().map(|r| r.total_bytes).sum();
    let prompt = format!(
        "Remove all {} cached model{} ({})?",
        repos.len(),
        if repos.len() == 1 { "" } else { "s" },
        format_size(total)
    );
    if !args.yes && !confirm(&prompt)? {
        return Ok(ExitCode::Success);
    }
    let freed = remove_all_cached()?;
    eprintln!(
        "  {} Removed {} model{} ({})",
        style::success_symbol(),
        repos.len(),
        if repos.len() == 1 { "" } else { "s" },
        format_size(freed)
    );
    Ok(ExitCode::Success)
}

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
