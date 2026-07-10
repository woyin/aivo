//! `aivo code mcp` — list/add/rm the coding agent's MCP servers from the CLI.
//! Interactive twin: `/mcp` inside `aivo code`; both edit the user
//! `~/.config/aivo/mcp.json` and read the repo `.mcp.json` (project scope).

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};

use crate::agent::{mcp, mcp_import};
use crate::cli::{McpAddArgs, McpArgs, McpImportArgs, McpNameArgs, McpRemoveArgs, McpSubcommand};
use crate::errors::ExitCode;
use crate::services::session_store::SessionStore;
use crate::style;

#[derive(Default)]
pub struct McpCommand;

impl McpCommand {
    pub fn new() -> Self {
        Self
    }

    pub async fn execute(&self, args: McpArgs) -> ExitCode {
        let cmd = args.command.unwrap_or(McpSubcommand::List);
        let result = match cmd {
            McpSubcommand::List => list_action().await,
            McpSubcommand::Cat(a) => cat_action(a).await,
            McpSubcommand::Add(a) => add_action(a).await,
            McpSubcommand::Enable(a) => toggle_action(a, true).await,
            McpSubcommand::Disable(a) => toggle_action(a, false).await,
            McpSubcommand::Remove(a) => remove_action(a).await,
            McpSubcommand::Import(a) => import_action(a).await,
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
        println!("{} aivo code mcp [SUBCOMMAND]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Manage the coding agent's MCP servers. Interactive twin: /mcp inside `aivo code`."
            )
        );
        println!();
        println!("{}", style::bold("Subcommands:"));
        let row = |a: &str, b: &str| {
            println!("  {}  {}", style::cyan(format!("{:<26}", a)), style::dim(b));
        };
        row("list", "Show configured servers (default)");
        row("cat <name>", "Show one server's config and state");
        row("add <command> [args…]", "Add a stdio server (name derived)");
        row("add <https://url>", "Add a remote Streamable HTTP server");
        row("add '<json>'", "Add from a pasted mcpServers JSON block");
        row("add -p …", "…into the repo ./.mcp.json (project scope)");
        row(
            "enable|disable <name>",
            "Turn a server on/off for the agent",
        );
        row("rm [-p] <name>", "Remove a server (-p: from ./.mcp.json)");
        row(
            "import [tool] [name]",
            "Copy servers from claude/cursor/gemini/copilot/amp",
        );
        println!();
        println!("{}", style::bold("Files:"));
        println!(
            "  {}",
            style::dim("~/.config/aivo/mcp.json   user scope (managed here)")
        );
        println!(
            "  {}",
            style::dim("./.mcp.json               project scope (add -p / rm -p)")
        );
    }
}

fn cwd() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

async fn list_action() -> Result<ExitCode> {
    let servers = mcp::configured_servers(&cwd());
    if servers.is_empty() {
        println!("No MCP servers configured.");
        println!(
            "{}",
            style::dim("Add one with `aivo code mcp add <command|url|json>`.")
        );
        return Ok(ExitCode::Success);
    }
    let store = SessionStore::new();
    let disabled: HashSet<String> = store
        .get_disabled_mcp_servers()
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
    let tool_optouts = store.get_disabled_mcp_tools().await.unwrap_or_default();
    // Char count, not byte len — `format!` pads by chars.
    let name_w = servers
        .iter()
        .map(|s| s.name.chars().count())
        .max()
        .unwrap_or(4);
    for s in &servers {
        let scope = match s.scope {
            mcp::ServerScope::User => "user   ",
            mcp::ServerScope::Project => "project",
            mcp::ServerScope::Pack => "pack   ",
        };
        // Prefix match on the advertised (sanitized) form; display-only.
        let prefix = mcp::qualified_name(&s.name, "");
        let off_tools = tool_optouts
            .iter()
            .filter(|q| q.starts_with(&prefix))
            .count();
        let is_on = !disabled.contains(&s.name);
        let state = if is_on {
            style::bullet_symbol()
        } else {
            style::empty_bullet_symbol()
        };
        let tools_note = if is_on && off_tools > 0 {
            style::dim(format!(" ({off_tools} tools off)")).to_string()
        } else {
            String::new()
        };
        println!(
            "{} {}  {}  {}{}",
            state,
            style::cyan(format!("{:<name_w$}", s.name)),
            style::dim(scope),
            style::dim(&s.command),
            tools_note,
        );
    }
    Ok(ExitCode::Success)
}

async fn toggle_action(args: McpNameArgs, enable: bool) -> Result<ExitCode> {
    let name = args.name;
    if !mcp::configured_servers(&cwd())
        .iter()
        .any(|s| s.name == name)
    {
        eprintln!("No MCP server named `{name}`.");
        return Ok(ExitCode::UserError);
    }
    SessionStore::new()
        .set_mcp_server_enabled(&name, enable)
        .await
        .map_err(|e| anyhow!("failed to update server state: {e}"))?;
    println!(
        "{} MCP server `{name}`",
        if enable { "Enabled" } else { "Disabled" }
    );
    Ok(ExitCode::Success)
}

async fn cat_action(args: McpNameArgs) -> Result<ExitCode> {
    let cwd = cwd();
    let name = args.name;
    let Some(server) = mcp::configured_servers(&cwd)
        .into_iter()
        .find(|s| s.name == name)
    else {
        eprintln!("No MCP server named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    let store = SessionStore::new();
    let disabled: HashSet<String> = store
        .get_disabled_mcp_servers()
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
    let prefix = mcp::qualified_name(&name, "");
    let off_tools: Vec<String> = store
        .get_disabled_mcp_tools()
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|q| q.strip_prefix(&prefix).map(str::to_string))
        .collect();
    let (scope, path) = match server.scope {
        mcp::ServerScope::User => (
            "user",
            mcp::user_config_path()
                .unwrap_or_default()
                .display()
                .to_string(),
        ),
        mcp::ServerScope::Project => ("project", cwd.join(".mcp.json").display().to_string()),
        mcp::ServerScope::Pack => (
            "pack",
            pack_mcp_path(&name)
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        ),
    };
    println!("Name:       {}", style::cyan(&name));
    println!(
        "Scope:      {scope} ({})",
        style::dim(crate::services::system_env::collapse_tilde(&path))
    );
    println!(
        "Transport:  {}",
        if server.remote { "http" } else { "stdio" }
    );
    let state = if disabled.contains(&name) {
        style::dim("off").to_string()
    } else {
        "on".to_string()
    };
    println!("State:      {state}");
    if !off_tools.is_empty() {
        println!("Tools off:  {}", style::dim(off_tools.join(", ")));
    }
    if let Some(cfg) = raw_server_config(server.scope, &cwd, &name)
        && let Ok(pretty) = serde_json::to_string_pretty(&cfg)
    {
        println!("Config:");
        for line in pretty.lines() {
            println!("  {}", style::dim(line));
        }
    }
    Ok(ExitCode::Success)
}

/// The raw `mcpServers.<name>` JSON from the file that defines the server.
fn raw_server_config(
    scope: mcp::ServerScope,
    cwd: &std::path::Path,
    name: &str,
) -> Option<serde_json::Value> {
    let path = match scope {
        mcp::ServerScope::User => mcp::user_config_path()?,
        mcp::ServerScope::Project => cwd.join(".mcp.json"),
        mcp::ServerScope::Pack => pack_mcp_path(name)?,
    };
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()?
        .get("mcpServers")?
        .get(name)
        .cloned()
}

/// The `.mcp.json` of the installed pack that defines server `name`.
fn pack_mcp_path(name: &str) -> Option<std::path::PathBuf> {
    crate::agent::packs::mcp_dirs()
        .into_iter()
        .map(|d| d.join(".mcp.json"))
        .find(|p| {
            std::fs::read_to_string(p)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v.get("mcpServers")?.get(name).cloned())
                .is_some()
        })
}

/// Split a leading or trailing `-p`/`--project` off the add spec. Only the
/// edges — clap can't own the flag here (`allow_hyphen_values` on the spec
/// would swallow it), and a server's own mid-line `-p` arg must survive.
fn split_project_flag(mut spec: Vec<String>) -> (Vec<String>, bool) {
    let is_flag = |s: &String| s == "-p" || s == "--project";
    if spec.first().is_some_and(is_flag) {
        spec.remove(0);
        (spec, true)
    } else if spec.last().is_some_and(is_flag) {
        spec.pop();
        (spec, true)
    } else {
        (spec, false)
    }
}

async fn add_action(args: McpAddArgs) -> Result<ExitCode> {
    const USAGE: &str = "usage: aivo code mcp add [-p] <command [args…] | https://url | json>";
    let (spec, project) = split_project_flag(args.spec);
    let args = McpAddArgs { spec };
    let mut existing: HashSet<String> = mcp::configured_servers(&cwd())
        .into_iter()
        .map(|s| s.name)
        .collect();
    let store = SessionStore::new();
    let write_value = |name: String, value: serde_json::Value| async move {
        let result = if project {
            mcp::add_project_server_value(&cwd(), &name, &value).await
        } else {
            mcp::add_user_server_value(&name, &value).await
        };
        result.map_err(|e| anyhow!("failed to add `{name}`: {e}"))
    };
    let added_note = if project {
        " → ./.mcp.json (project — commit it to share)"
    } else {
        ""
    };

    // The shell already tokenized argv — multiple arguments are used verbatim
    // as `command args…` (re-joining and re-splitting would mangle args with
    // spaces the user quoted for their shell). A single argument is a JSON
    // block, a bare URL, or a full quoted command line (TUI-style), which
    // shlex splits.
    let (command, cmd_args) = match args.spec.as_slice() {
        [] => bail!("{USAGE}"),
        [single] => {
            let single = single.trim();
            if single.is_empty() {
                bail!("{USAGE}");
            }
            let json_input = if single.starts_with('{') {
                Some(single.to_string())
            } else {
                mcp::bare_url_to_config(single)
            };
            if let Some(json) = json_input {
                let parsed = mcp::parse_mcp_json(&json)
                    .map_err(|e| anyhow!("couldn't parse MCP config: {e}"))?;
                let mut added_stdio = false;
                for (name_opt, value) in parsed {
                    let name = mcp::dedupe_name(
                        name_opt.unwrap_or_else(|| mcp::derive_name_from_value(&value)),
                        &existing,
                    );
                    write_value(name.clone(), value.clone()).await?;
                    // A freshly added server starts enabled, matching the TUI.
                    store.set_mcp_server_enabled(&name, true).await.ok();
                    println!("Added MCP server `{name}`{added_note}");
                    if value.get("url").is_some() {
                        println!(
                            "{}",
                            style::dim(
                                "If it needs OAuth, authorize it inside `aivo code` → /mcp → Ctrl+O."
                            )
                        );
                    }
                    added_stdio |= value.get("command").is_some();
                    existing.insert(name);
                }
                if project && added_stdio {
                    note_project_stdio_consent();
                }
                return Ok(ExitCode::Success);
            }
            mcp::parse_mcp_add_input(single).map_err(|e| anyhow!(e))?
        }
        [command, rest @ ..] => {
            if command.starts_with("http://") || command.starts_with("https://") {
                bail!("unexpected arguments after a URL — {USAGE}");
            }
            (command.clone(), rest.to_vec())
        }
    };

    let name = mcp::dedupe_name(mcp::derive_server_name(&command, &cmd_args), &existing);
    write_value(
        name.clone(),
        serde_json::json!({"command": command, "args": cmd_args}),
    )
    .await?;
    store.set_mcp_server_enabled(&name, true).await.ok();
    println!("Added MCP server `{name}`{added_note}");
    if project {
        note_project_stdio_consent();
    }
    Ok(ExitCode::Success)
}

/// Printed after a project-scope stdio add: the agent gates repo `.mcp.json`
/// stdio servers behind a one-time consent card, so the add isn't mistaken
/// for silent auto-run approval.
fn note_project_stdio_consent() {
    println!(
        "{}",
        style::dim("Project stdio servers ask for a one-time consent when the agent starts.")
    );
}

async fn import_action(args: McpImportArgs) -> Result<ExitCode> {
    let sources = mcp_import::discover();
    let existing: HashSet<String> = mcp::configured_servers(&cwd())
        .into_iter()
        .map(|s| s.name)
        .collect();

    let Some(tool) = args.tool.as_deref().map(str::to_lowercase) else {
        // Listing mode: show everything found and where it came from.
        if sources.is_empty() {
            println!("No MCP servers found in other tools' configs.");
        } else {
            for src in &sources {
                println!(
                    "{}  {}",
                    style::bold(src.tool),
                    style::dim(crate::services::system_env::collapse_tilde(
                        &src.path.display().to_string()
                    )),
                );
                let name_w = src
                    .servers
                    .iter()
                    .map(|s| s.name.chars().count())
                    .max()
                    .unwrap_or(4);
                for s in &src.servers {
                    let marker = if existing.contains(&s.name) {
                        style::dim("  (already configured)").to_string()
                    } else {
                        String::new()
                    };
                    println!(
                        "  {}  {}{marker}",
                        style::cyan(format!("{:<name_w$}", s.name)),
                        style::dim(&s.display),
                    );
                }
            }
            println!();
            println!(
                "{}",
                style::dim(
                    "Import with `aivo code mcp import <tool> [name]`, or `aivo code mcp import all`."
                )
            );
        }
        note_unsupported_toml();
        return Ok(ExitCode::Success);
    };

    if let Some((_, rel)) = mcp_import::UNSUPPORTED_TOML
        .iter()
        .find(|(t, _)| *t == tool)
    {
        eprintln!(
            "{tool} keeps its MCP servers in TOML (~/{rel}), which aivo can't import yet — add them with `aivo code mcp add`."
        );
        return Ok(ExitCode::UserError);
    }
    let selected: Vec<_> = sources
        .iter()
        .filter(|s| tool == "all" || s.tool == tool)
        .collect();
    if selected.is_empty() {
        eprintln!(
            "No importable MCP servers found for `{tool}` (supported: claude, cursor, gemini, copilot, amp, all)."
        );
        return Ok(ExitCode::UserError);
    }

    let store = SessionStore::new();
    let mut existing = existing;
    let mut touched = 0usize;
    for src in selected {
        for s in &src.servers {
            if args.name.as_deref().is_some_and(|only| only != s.name) {
                continue;
            }
            touched += 1;
            if existing.contains(&s.name) {
                println!("Skipped `{}` — already configured", s.name);
                continue;
            }
            mcp::add_user_server_value(&s.name, &s.config)
                .await
                .map_err(|e| anyhow!("failed to import `{}`: {e}", s.name))?;
            // Imported servers start enabled, like any fresh add.
            store.set_mcp_server_enabled(&s.name, true).await.ok();
            existing.insert(s.name.clone());
            println!("Imported `{}` from {}", s.name, src.tool);
        }
    }
    if touched == 0 {
        match args.name {
            Some(name) => eprintln!("No server named `{name}` found in `{tool}`."),
            None => eprintln!("Nothing to import from `{tool}`."),
        }
        return Ok(ExitCode::UserError);
    }
    Ok(ExitCode::Success)
}

/// Mention TOML-config tools that exist on disk but can't be imported, so
/// their absence from the listing isn't mistaken for "no servers".
fn note_unsupported_toml() {
    let Some(home) = crate::services::system_env::home_dir() else {
        return;
    };
    for (tool, rel) in mcp_import::UNSUPPORTED_TOML {
        if home.join(rel).is_file() {
            println!(
                "{}",
                style::dim(format!(
                    "({tool} uses TOML (~/{rel}) — not importable yet; use `aivo code mcp add`)"
                ))
            );
        }
    }
}

async fn remove_action(args: McpRemoveArgs) -> Result<ExitCode> {
    let name = args.name;
    if args.project {
        return match mcp::remove_project_server(&cwd(), &name).await {
            Ok(true) => {
                println!("Removed project MCP server `{name}` from ./.mcp.json");
                Ok(ExitCode::Success)
            }
            Ok(false) => {
                eprintln!("`{name}` is not in ./.mcp.json.");
                Ok(ExitCode::UserError)
            }
            Err(e) => Err(anyhow!("failed to remove `{name}`: {e}")),
        };
    }
    let Some(server) = mcp::configured_servers(&cwd())
        .into_iter()
        .find(|s| s.name == name)
    else {
        eprintln!("No MCP server named `{name}`.");
        return Ok(ExitCode::UserError);
    };
    if server.scope == mcp::ServerScope::Project {
        // The merged view is project-wins, which can shadow a same-named user
        // entry — that one is still ours to remove.
        return match mcp::remove_user_server(&name).await {
            Ok(true) => {
                println!(
                    "Removed user-scope MCP server `{name}` (the project .mcp.json entry with this name still applies)"
                );
                Ok(ExitCode::Success)
            }
            Ok(false) => {
                eprintln!(
                    "`{name}` is defined in this repo's .mcp.json — remove it with `aivo code mcp rm -p {name}`."
                );
                Ok(ExitCode::UserError)
            }
            Err(e) => Err(anyhow!("failed to remove `{name}`: {e}")),
        };
    }
    match mcp::remove_user_server(&name).await {
        Ok(true) => {
            println!("Removed MCP server `{name}`");
            Ok(ExitCode::Success)
        }
        Ok(false) => {
            eprintln!("`{name}` was not in mcp.json.");
            Ok(ExitCode::UserError)
        }
        Err(e) => Err(anyhow!("failed to remove `{name}`: {e}")),
    }
}
