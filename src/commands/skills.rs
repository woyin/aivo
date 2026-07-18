//! `aivo code skills` — list/install/rm the coding agent's skills from the CLI.
//! Interactive twin: `/skills` inside `aivo code` (same roots, same install dirs).

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Result, anyhow};

use crate::agent::skills;
use crate::cli::{SkillsArgs, SkillsInstallArgs, SkillsNameArgs, SkillsSubcommand};
use crate::errors::ExitCode;
use crate::services::session_store::SessionStore;
use crate::style;

#[derive(Default)]
pub struct SkillsCommand;

impl SkillsCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: SkillsArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(SkillsSubcommand::List);
        let result = match cmd {
            SkillsSubcommand::List => list_action().await,
            SkillsSubcommand::Cat(a) => cat_action(a).await,
            SkillsSubcommand::Install(a) => install_action(a).await,
            SkillsSubcommand::Enable(a) => toggle_action(a, true).await,
            SkillsSubcommand::Disable(a) => toggle_action(a, false).await,
            SkillsSubcommand::Remove(a) => remove_action(a).await,
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
        println!("{} aivo code skills [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage the coding agent's skills. Interactive twin: /skills inside `aivo code`."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row("list", "Show discovered skills (default)");
        row("cat <name>", "Show one skill in full");
        row(
            "install <source>",
            "Install the sole skill, or list when several",
        );
        row(
            "install <source> <name>",
            "Install one skill from the source",
        );
        row(
            "install <source> --all",
            "Install every skill at the source",
        );
        row(
            "install -p …",
            "…into the repo ./.agents/skills (project scope)",
        );
        row("enable|disable <name>", "Turn a skill on/off for the agent");
        row("rm <name>", "Remove a user-scope skill");
        println!();
        println!("{}", style::bold("Sources:"));
        println!(
            "  {}",
            style::dim(
                "github:owner/repo[@ref], a github.com repo or /tree/… folder URL, or a local path"
            )
        );
        println!();
        println!("{}", style::bold("Files:"));
        println!(
            "  {}",
            style::dim("~/.config/aivo/skills     user scope (managed here)")
        );
        println!(
            "  {}",
            style::dim(
                "./.agents/skills          project scope (install -p; rm = delete the folder)"
            )
        );
    }
}

fn cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

async fn list_action() -> Result<ExitCode> {
    let cwd = cwd();
    let found = skills::discover_skills(&cwd);
    if found.is_empty() {
        println!("No skills discovered.");
        println!(
            "{}",
            style::dim("Install one with `aivo code skills install <source>`.")
        );
        return Ok(ExitCode::Success);
    }
    let disabled: HashSet<String> = SessionStore::new()
        .get_disabled_skills()
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
    // Char count, not byte len — `format!` pads by chars.
    let name_w = found
        .iter()
        .map(|s| s.name.chars().count())
        .max()
        .unwrap_or(4);
    // bullet(2) + name + scope(7) + separators → the rest is description room.
    let desc_w = terminal_cols().saturating_sub(name_w + 13).max(20);
    for s in &found {
        let scope = match skills::skill_scope(&s.dir, &cwd) {
            skills::SkillScope::User => "user   ",
            skills::SkillScope::Project => "project",
        };
        let state = if disabled.contains(&s.name) {
            style::empty_bullet_symbol()
        } else {
            style::bullet_symbol()
        };
        println!(
            "{} {}  {}  {}",
            state,
            style::cyan(format!("{:<name_w$}", s.name)),
            style::dim(scope),
            style::dim(fit(&skills::advert_description(&s.description), desc_w)),
        );
    }
    Ok(ExitCode::Success)
}

fn terminal_cols() -> usize {
    crossterm::terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(100)
}

fn fit(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let cut: String = s.chars().take(width.saturating_sub(1)).collect();
    format!("{cut}…")
}

async fn cat_action(args: SkillsNameArgs) -> Result<ExitCode> {
    let cwd = cwd();
    let name = args.name;
    let Some(skill) = skills::discover_skills(&cwd)
        .into_iter()
        .find(|s| s.name == name)
    else {
        eprintln!("No skill named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    let disabled = SessionStore::new()
        .get_disabled_skills()
        .await
        .unwrap_or_default();
    let scope = match skills::skill_scope(&skill.dir, &cwd) {
        skills::SkillScope::User => "user",
        skills::SkillScope::Project => "project",
    };
    println!("Name:         {}", style::cyan(&skill.name));
    println!(
        "Scope:        {scope} ({})",
        style::dim(crate::services::system_env::collapse_tilde(
            &skill.dir.display().to_string()
        ))
    );
    let state = if disabled.contains(&name) {
        style::dim("off").to_string()
    } else {
        "on".to_string()
    };
    println!("State:        {state}");
    if let Some(source) = skills::skill_source(&skill.dir) {
        println!("Source:       {source}");
    }
    println!("Description:  {}", skill.description);
    let body = skill.instructions();
    if !body.trim().is_empty() {
        println!();
        println!("{}", body.trim_end());
    }
    Ok(ExitCode::Success)
}

async fn toggle_action(args: SkillsNameArgs, enable: bool) -> Result<ExitCode> {
    let name = args.name;
    if !skills::discover_skills(&cwd())
        .iter()
        .any(|s| s.name == name)
    {
        eprintln!("No skill named `{name}`.");
        return Ok(ExitCode::UserError);
    }
    SessionStore::new()
        .set_skill_enabled(&name, enable)
        .await
        .map_err(|e| anyhow!("failed to update skill state: {e}"))?;
    println!(
        "{} skill `{name}`",
        if enable { "Enabled" } else { "Disabled" }
    );
    Ok(ExitCode::Success)
}

async fn install_action(args: SkillsInstallArgs) -> Result<ExitCode> {
    let dest_root = if args.project {
        skills::project_skills_dir(&cwd())
    } else {
        skills::user_skills_dir()
    };
    let only = if args.all {
        Some("*".to_string())
    } else {
        args.name
    };
    let outcome = skills::install_or_stage_into(&dest_root, &args.source, only.as_deref(), None)
        .await
        .map_err(|e| anyhow!(e))?;
    match outcome {
        skills::InstallOrStage::Installed(report) => {
            for name in &report.installed {
                println!("Installed `{name}` → {}", dest_root.display());
            }
            for name in &report.updated {
                println!("Updated `{name}` in {}", dest_root.display());
            }
            for name in &report.skipped_existing {
                println!("Skipped `{name}` — already installed");
            }
            Ok(ExitCode::Success)
        }
        skills::InstallOrStage::Pick(staged) => {
            let installed = staged.already_installed_in(&dest_root);
            let name_w = staged
                .skills
                .iter()
                .map(|s| s.name.chars().count())
                .max()
                .unwrap_or(4);
            println!("`{}` has {} skills:", args.source, staged.skills.len());
            for (s, exists) in staged.skills.iter().zip(installed) {
                let marker = if exists {
                    style::dim("  (already installed)").to_string()
                } else {
                    String::new()
                };
                println!(
                    "  {}  {}{marker}",
                    style::cyan(format!("{:<name_w$}", s.name)),
                    style::dim(skills::advert_description(&s.description)),
                );
            }
            println!();
            println!(
                "{}",
                style::dim("Install with `aivo code skills install <source> <name>`, or `--all`.")
            );
            Ok(ExitCode::Success)
        }
    }
}

async fn remove_action(args: SkillsNameArgs) -> Result<ExitCode> {
    let cwd = cwd();
    let name = args.name;
    let Some(skill) = skills::discover_skills(&cwd)
        .into_iter()
        .find(|s| s.name == name)
    else {
        eprintln!("No skill named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    // Match `/skills`: project-tier skills are never deleted by aivo.
    if skills::skill_scope(&skill.dir, &cwd) == skills::SkillScope::Project {
        eprintln!(
            "`{name}` is a project skill ({}) — delete that folder to remove it.",
            skill.dir.display()
        );
        return Ok(ExitCode::UserError);
    }
    skills::remove_skill_dir(&skill.dir).map_err(|e| anyhow!("failed to remove `{name}`: {e}"))?;
    // Clear any leftover disabled flag so a re-install isn't stuck off.
    SessionStore::new()
        .set_skill_enabled(&name, true)
        .await
        .ok();
    println!("Removed skill `{name}`");
    Ok(ExitCode::Success)
}
