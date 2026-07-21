//! `aivo code agents` — list/cat/rm the coding agent's named sub-agents.
//! Interactive twin: `/agents` inside `aivo code`. Creation is conversational
//! (ask the agent to make one), so there is deliberately no `add` verb.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};

use crate::agent::skills::advert_description;
use crate::agent::subagents::{self, Subagent};
use crate::cli::{AgentsArgs, AgentsNameArgs, AgentsSubcommand};
use crate::errors::ExitCode;
use crate::services::session_store::SessionStore;
use crate::style;

#[derive(Default)]
pub struct AgentsCommand;

impl AgentsCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: AgentsArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(AgentsSubcommand::List);
        let result = match cmd {
            AgentsSubcommand::List => list_action(),
            AgentsSubcommand::Cat(a) => cat_action(a),
            AgentsSubcommand::Remove(a) => remove_action(a),
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
        println!("{} aivo code agents [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage the coding agent's named sub-agents. Interactive twin: /agents inside `aivo code`."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row("list", "Show discovered sub-agents (default)");
        row("cat <name>", "Show one sub-agent in full");
        row("rm <name>", "Remove a repo- or user-scope sub-agent");
        println!();
        println!("{}", style::bold("Create one:"));
        println!(
            "  {}",
            style::dim("ask inside `aivo code` — \"make me a code-reviewer subagent\"")
        );
        println!();
        println!("{}", style::bold("Files:"));
        println!(
            "  {}",
            style::dim("./.aivo/agents ./.claude/agents   repo scope (ships with the project)")
        );
        println!(
            "  {}",
            style::dim("~/.config/aivo/agents             user scope (every project)")
        );
        println!(
            "  {}",
            style::dim("<pack>/agents                     pack scope (rm the pack to remove)")
        );
        println!(
            "  {}",
            style::dim("built-in (shadow with a same-named file): explorer, aivo-guide,")
        );
        println!("  {}", style::dim("  verification, advisor, evaluate"));
    }
}

fn cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Where a discovered profile lives, for display and delete eligibility.
fn scope_label(sa: &Subagent, config_dir: &Path) -> &'static str {
    if sa.is_builtin() {
        "builtin"
    } else if sa.repo_local {
        "repo"
    } else if sa.source.starts_with(config_dir.join("agents")) {
        "user"
    } else {
        "pack"
    }
}

fn discover() -> Vec<Subagent> {
    subagents::discover_subagents(&cwd(), SessionStore::new().config_dir())
}

fn list_action() -> Result<ExitCode> {
    let found = discover();
    if found.is_empty() {
        println!("No sub-agents discovered.");
        println!(
            "{}",
            style::dim("Create one inside `aivo code`: \"make me a code-reviewer subagent\".")
        );
        return Ok(ExitCode::Success);
    }
    let config_dir = SessionStore::new().config_dir().to_path_buf();
    let name_w = found
        .iter()
        .map(|s| s.name.chars().count())
        .max()
        .unwrap_or(4);
    // bullet(2) + name + scope(7) + separators → the rest is description room.
    let desc_w = terminal_cols().saturating_sub(name_w + 13).max(20);
    for s in &found {
        println!(
            "{} {}  {}  {}",
            style::bullet_symbol(),
            style::cyan(format!("{:<name_w$}", s.name)),
            style::dim(format!("{:<7}", scope_label(s, &config_dir))),
            style::dim(fit(&advert_description(&s.description), desc_w)),
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

fn cat_action(args: AgentsNameArgs) -> Result<ExitCode> {
    let name = args.name;
    let Some(sa) = discover().into_iter().find(|s| s.name == name) else {
        eprintln!("No sub-agent named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    let config_dir = SessionStore::new().config_dir().to_path_buf();
    println!("Name:         {}", style::cyan(&sa.name));
    let location = if sa.is_builtin() {
        "compiled into aivo".to_string()
    } else {
        crate::services::system_env::collapse_tilde(&sa.source.display().to_string())
    };
    println!(
        "Scope:        {} ({})",
        scope_label(&sa, &config_dir),
        style::dim(location)
    );
    println!(
        "Model:        {}",
        sa.model.as_deref().unwrap_or("(inherits the parent's)")
    );
    let tools = match sa.resolved_tools() {
        Some(t) => t.join(", "),
        None => "all".to_string(),
    };
    println!("Tools:        {tools}");
    if sa.isolation_worktree {
        println!("Isolation:    worktree");
    }
    println!("Description:  {}", sa.description);
    if !sa.body.trim().is_empty() {
        println!();
        println!("{}", sa.body.trim_end());
    }
    Ok(ExitCode::Success)
}

fn remove_action(args: AgentsNameArgs) -> Result<ExitCode> {
    let name = args.name;
    let Some(sa) = discover().into_iter().find(|s| s.name == name) else {
        eprintln!("No sub-agent named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    if sa.is_builtin() {
        return Err(anyhow!(
            "`{name}` is built into aivo — shadow it with your own ~/.config/aivo/agents/{name}.md instead"
        ));
    }
    let config_dir = SessionStore::new().config_dir().to_path_buf();
    if scope_label(&sa, &config_dir) == "pack" {
        return Err(anyhow!(
            "`{name}` ships with an extension pack — remove the pack instead (`aivo code packs`)"
        ));
    }
    std::fs::remove_file(&sa.source)
        .map_err(|e| anyhow!("failed to remove {}: {e}", sa.source.display()))?;
    println!(
        "Removed sub-agent `{name}` ({})",
        style::dim(crate::services::system_env::collapse_tilde(
            &sa.source.display().to_string()
        ))
    );
    Ok(ExitCode::Success)
}
