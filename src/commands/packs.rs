//! `aivo code packs` — list/add/rm agent extension packs (see
//! [`crate::agent::packs`]). `add` shows what the pack ships and asks first.

use anyhow::{Result, anyhow};

use crate::agent::packs;
use crate::cli::{PacksArgs, PacksSubcommand};
use crate::errors::ExitCode;
use crate::style;

#[derive(Default)]
pub struct PacksCommand;

impl PacksCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: PacksArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(PacksSubcommand::List);
        let result = match cmd {
            PacksSubcommand::List => list_action(),
            PacksSubcommand::Add { source, yes } => add_action(&source, yes).await,
            PacksSubcommand::Rm { name } => rm_action(&name),
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
        println!("{} aivo code packs [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage the coding agent's extension packs — skills, sub-agent profiles, \
hooks, and MCP servers installed as one unit (Claude Code plugin layout)."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row("list", "Show installed packs and what each ships (default)");
        row(
            "add <source>",
            "Install from github:owner/repo, URL, or path",
        );
        row("add -y …", "…without the contents confirmation");
        row("rm <name>", "Remove a pack and everything it shipped");
    }
}

fn packs_root() -> Result<std::path::PathBuf> {
    packs::packs_root().ok_or_else(|| anyhow!("cannot resolve the home directory"))
}

fn list_action() -> Result<ExitCode> {
    let installed = packs::installed_packs();
    if installed.is_empty() {
        println!(
            "No packs installed. Add one with {}",
            style::cyan("aivo code packs add <source>")
        );
        return Ok(ExitCode::Success);
    }
    for pack in installed {
        let version = pack
            .version
            .as_deref()
            .map(|v| format!(" v{v}"))
            .unwrap_or_default();
        println!("{}{}", style::cyan(&pack.name), style::dim(version));
        if let Some(desc) = &pack.description {
            println!("  {}", style::dim(desc));
        }
        println!(
            "  {}",
            style::dim(summarize(&packs::scan_contents(&pack.dir)))
        );
    }
    Ok(ExitCode::Success)
}

async fn add_action(source: &str, yes: bool) -> Result<ExitCode> {
    let root = packs_root()?;
    let tree = crate::agent::skills::fetch_source_tree(source, None)
        .await
        .map_err(|e| anyhow!(e))?;
    let result = install_staged(&root, source, &tree.root, yes).await;
    if let Some(tmp) = &tree.cleanup {
        let _ = std::fs::remove_dir_all(tmp);
    }
    result
}

async fn install_staged(
    root: &std::path::Path,
    source: &str,
    staged: &std::path::Path,
    yes: bool,
) -> Result<ExitCode> {
    let contents = packs::scan_contents(staged);
    if contents.is_empty() {
        return Err(anyhow!(
            "`{source}` doesn't look like a pack — no skills/, agents/, hooks/hooks.json, \
or .mcp.json found"
        ));
    }
    let name = derive_pack_name(source, staged)?;
    println!("Pack {} from {}:", style::cyan(&name), style::dim(source));
    print!("{}", describe(&contents));
    if !yes {
        // Consent is for code-executing components; a skills/agents-only pack
        // installs unattended off a TTY (matches the `-y` contract).
        if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            eprint!("Install? [y/N] ");
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            if !matches!(line.trim(), "y" | "Y" | "yes") {
                println!("Cancelled.");
                return Ok(ExitCode::Success);
            }
        } else if contents.executes_code() {
            return Err(anyhow!(
                "not a terminal — re-run with -y to accept the hooks/MCP servers above"
            ));
        }
    }
    let dest = packs::install_tree(root, &name, staged).map_err(|e| anyhow!(e))?;
    println!(
        "{} installed {} → {}",
        style::green("✓"),
        style::cyan(&name),
        style::dim(crate::services::system_env::collapse_tilde(
            &dest.display().to_string()
        ))
    );
    Ok(ExitCode::Success)
}

/// Manifest name, else the source's last path-ish segment.
fn derive_pack_name(source: &str, staged: &std::path::Path) -> Result<String> {
    let manifest_name = packs::manifest_name(staged);
    let fallback = source
        .trim_end_matches('/')
        .rsplit(['/', ':'])
        .next()
        .unwrap_or("")
        .trim_end_matches(".git")
        .to_string();
    let name = manifest_name.filter(|n| !n.is_empty()).unwrap_or(fallback);
    if !crate::agent::subagents::is_valid_name(&name) {
        return Err(anyhow!(
            "cannot derive a usable pack name from `{source}` — add a \
.claude-plugin/plugin.json with a \"name\""
        ));
    }
    Ok(name)
}

fn rm_action(name: &str) -> Result<ExitCode> {
    let root = packs_root()?;
    packs::remove(&root, name).map_err(|e| anyhow!(e))?;
    println!("{} removed {}", style::green("✓"), style::cyan(name));
    Ok(ExitCode::Success)
}

fn summarize(c: &packs::PackContents) -> String {
    let mut parts = Vec::new();
    let count = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
    if !c.skills.is_empty() {
        parts.push(count(c.skills.len(), "skill"));
    }
    if !c.agents.is_empty() {
        parts.push(count(c.agents.len(), "agent"));
    }
    if !c.hook_commands.is_empty() {
        parts.push(count(c.hook_commands.len(), "hook"));
    }
    if !c.mcp_stdio.is_empty() {
        parts.push(count(c.mcp_stdio.len(), "MCP server"));
    }
    if parts.is_empty() {
        "empty".to_string()
    } else {
        parts.join(" · ")
    }
}

/// The full pre-install disclosure; code-executing components are called out.
fn describe(c: &packs::PackContents) -> String {
    let mut out = String::new();
    if !c.skills.is_empty() {
        out.push_str(&format!("  skills: {}\n", c.skills.join(", ")));
    }
    if !c.agents.is_empty() {
        out.push_str(&format!("  agents: {}\n", c.agents.join(", ")));
    }
    for cmd in &c.hook_commands {
        out.push_str(&format!("  hook (runs on agent lifecycle events): {cmd}\n"));
    }
    for (name, cmd) in &c.mcp_stdio {
        out.push_str(&format!("  MCP server (local process) {name}: {cmd}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_and_describe_call_out_code_execution() {
        let c = packs::PackContents {
            skills: vec!["review".into()],
            agents: vec![],
            hook_commands: vec!["fmt.sh".into()],
            mcp_stdio: vec![("db".into(), "npx -y db".into())],
        };
        assert_eq!(summarize(&c), "1 skill · 1 hook · 1 MCP server");
        let d = describe(&c);
        assert!(d.contains("runs on agent lifecycle events"));
        assert!(d.contains("local process"));
    }

    #[test]
    fn pack_name_from_source_fallback() {
        let staged = std::env::temp_dir();
        assert_eq!(
            derive_pack_name("github:owner/my-pack", &staged).unwrap(),
            "my-pack"
        );
        assert_eq!(
            derive_pack_name("https://github.com/o/toolkit.git", &staged).unwrap(),
            "toolkit"
        );
        assert!(derive_pack_name("::", &staged).is_err());
    }
}
