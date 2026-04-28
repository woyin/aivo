/**
 * AliasCommand handler — manage aliases.
 *
 * Two flavors share the same namespace:
 * - Model alias: short name → model name (e.g. "fast" → "claude-haiku-4-5"),
 *   resolved by any command that accepts --model.
 * - Bundle alias: short name → preset launch (tool + args), resolved by
 *   `aivo run <bundle>` and the `aivo <bundle>` shortcut.
 */
use anyhow::Result;

use crate::cli::AliasArgs;
use crate::constants::{KNOWN_TOOLS, RESERVED_ALIAS_NAMES};
use crate::errors::ExitCode;
use crate::services::session_store::{AliasValue, BundleAlias, SessionStore};
use crate::style;

pub struct AliasCommand {
    session_store: SessionStore,
}

impl AliasCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: AliasArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, args: AliasArgs) -> Result<ExitCode> {
        // `rm <name>` keyword form (matches what the help advertises) and
        // `--rm <name>` flag form. The keyword form only triggers when the
        // first positional is *exactly* "rm"; `rm=model` still creates an
        // alias literally named `rm`.
        let rm_keyword = args.assignment.as_deref() == Some("rm");
        if args.rm || rm_keyword {
            if args.json {
                anyhow::bail!("--json only applies to listing aliases");
            }
            let name = if rm_keyword {
                args.rest.first().map(String::as_str)
            } else {
                args.assignment.as_deref()
            };
            return self.remove_alias(name).await;
        }

        // `aivo alias name=model`, `aivo alias name model`, or
        // `aivo alias name <tool> [args...]` (Bundle).
        if let Some(ref assignment) = args.assignment {
            if args.json {
                anyhow::bail!("--json only applies to listing aliases");
            }
            return self.set_alias(assignment, &args.rest).await;
        }

        // `aivo alias` — list all
        self.list_aliases(args.json).await
    }

    async fn list_aliases(&self, json: bool) -> Result<ExitCode> {
        let aliases = self.session_store.list_alias_values().await?;

        if json {
            let mut entries: Vec<_> = aliases.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let payload: serde_json::Map<String, serde_json::Value> = entries
                .into_iter()
                .map(|(name, value)| (name, serde_json::to_value(&value).unwrap_or_default()))
                .collect();
            println!("{}", serde_json::to_string_pretty(&payload)?);
            return Ok(ExitCode::Success);
        }

        if aliases.is_empty() {
            println!("{}", style::dim("No aliases defined."));
            println!();
            println!(
                "{}",
                style::dim("Create a model alias:  aivo alias fast=claude-haiku-4-5")
            );
            println!(
                "{}",
                style::dim(
                    "Create a launch alias: aivo alias quick claude --key work --model fast"
                )
            );
            return Ok(ExitCode::Success);
        }

        let mut entries: Vec<_> = aliases.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Pad raw strings *before* styling — `{:width$}` counts ANSI escape
        // bytes as part of the string length, so applying width to a styled
        // string produces wrong visual alignment.
        let max_name = entries.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        const KIND_WIDTH: usize = 6; // "launch"
        for (name, value) in &entries {
            let kind = match value {
                AliasValue::Model(_) => "model",
                AliasValue::Bundle(_) => "launch",
            };
            let padded_name = format!("{:<width$}", name, width = max_name);
            let padded_kind = format!("{:<width$}", kind, width = KIND_WIDTH);
            println!(
                "{}  {}  {} {}",
                style::cyan(padded_name),
                style::dim(padded_kind),
                style::dim("->"),
                value,
            );
        }
        Ok(ExitCode::Success)
    }

    async fn set_alias(&self, assignment: &str, rest: &[String]) -> Result<ExitCode> {
        // `name=model` shorthand: Model alias only, rejects extra trailing args.
        if let Some((name, model)) = assignment.split_once('=') {
            if !rest.is_empty() {
                eprintln!(
                    "{} `name=model` shorthand cannot mix with trailing args. Use `aivo alias {} {}` instead.",
                    style::red("Error:"),
                    name,
                    rest.join(" ")
                );
                return Ok(ExitCode::UserError);
            }
            return self.set_model_alias(name, model).await;
        }

        let name = assignment;
        let Some(first) = rest.first() else {
            eprintln!(
                "{} Expected:  aivo alias <name>=<model>  |  aivo alias <name> <model>  |  aivo alias <name> <tool> [args...]",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        };

        // Bundle: first trailing token is a known tool name.
        if KNOWN_TOOLS.contains(&first.as_str()) {
            let bundle = BundleAlias {
                tool: first.clone(),
                args: rest[1..].to_vec(),
            };
            return self.set_bundle_alias(name, bundle).await;
        }

        // Otherwise positional Model alias: `name model` (single trailing token).
        if rest.len() == 1 {
            return self.set_model_alias(name, first).await;
        }

        eprintln!(
            "{} To create a launch alias, the first arg after the name must be a tool ({}). Got '{}'.",
            style::red("Error:"),
            KNOWN_TOOLS.join(", "),
            first
        );
        Ok(ExitCode::UserError)
    }

    async fn set_model_alias(&self, name: &str, model: &str) -> Result<ExitCode> {
        if model.is_empty() {
            eprintln!(
                "{} Model alias target must not be empty",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }
        if let Err(msg) = validate_alias_name(name) {
            eprintln!("{} {}", style::red("Error:"), msg);
            return Ok(ExitCode::UserError);
        }

        if name == model {
            eprintln!(
                "{} Alias cannot point to itself: {}",
                style::red("Error:"),
                name
            );
            return Ok(ExitCode::UserError);
        }

        // Cycle check operates only on the Model-alias subgraph; Bundle entries
        // are not chainable through `--model` resolution.
        let mut aliases = self.session_store.get_aliases().await?;
        aliases.insert(name.to_string(), model.to_string());
        if has_cycle(&aliases, name) {
            eprintln!(
                "{} This would create a circular alias chain",
                style::red("Error:")
            );
            return Ok(ExitCode::UserError);
        }

        let prev = self
            .session_store
            .set_alias(name.to_string(), model.to_string())
            .await?;
        report_set(name, &AliasValue::Model(model.to_string()), prev);
        Ok(ExitCode::Success)
    }

    async fn set_bundle_alias(&self, name: &str, bundle: BundleAlias) -> Result<ExitCode> {
        if let Err(msg) = validate_alias_name(name) {
            eprintln!("{} {}", style::red("Error:"), msg);
            return Ok(ExitCode::UserError);
        }
        // Tool already validated by KNOWN_TOOLS check at the call site.
        let prev = self
            .session_store
            .set_bundle(name.to_string(), bundle.clone())
            .await?;
        report_set(name, &AliasValue::Bundle(bundle), prev);
        Ok(ExitCode::Success)
    }

    async fn remove_alias(&self, name: Option<&str>) -> Result<ExitCode> {
        let name = match name {
            Some(n) => n,
            None => {
                eprintln!("{} Expected: aivo alias rm <name>", style::red("Error:"));
                return Ok(ExitCode::UserError);
            }
        };

        match self.session_store.remove_alias(name).await? {
            Some(value) => {
                println!(
                    "Removed {} {} {}",
                    style::cyan(name),
                    style::dim("->"),
                    style::dim(value.to_string())
                );
                Ok(ExitCode::Success)
            }
            None => {
                eprintln!("{} No alias named '{}'", style::red("Error:"), name);
                Ok(ExitCode::UserError)
            }
        }
    }

    pub fn print_help() {
        println!(
            "{} aivo alias [name[=model] | name <tool> [args...]]",
            style::bold("Usage:")
        );
        println!();
        println!(
            "{}",
            style::dim("Create, list, or remove aliases. Two flavors share one namespace:")
        );
        println!(
            "{}",
            style::dim(
                "  model — short name → model name; works wherever -m / --model is accepted."
            )
        );
        println!(
            "{}",
            style::dim("  launch — short name → preset (tool + flags); run via `aivo run <name>`.")
        );
        println!();
        println!("{}", style::bold("Actions:"));
        let print_row = |label: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<22}", label)),
                style::dim(desc)
            );
        };
        print_row("(no args)", "List all aliases");
        print_row("name=model", "Create or update a model alias");
        print_row("name model", "Create or update a model alias (positional)");
        print_row("name <tool> [args...]", "Create or update a launch alias");
        print_row("rm <name>", "Remove an alias of either kind");
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<22}", flag)),
                style::dim(desc)
            );
        };
        print_opt("--json", "Output alias list as JSON (listing only)");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo alias fast=claude-haiku-4-5"));
        println!("  {}", style::dim("aivo alias best claude-sonnet-4-6"));
        println!(
            "  {}",
            style::dim("aivo alias quick claude --key work --model fast --max-context 1m")
        );
        println!("  {}", style::dim("aivo run quick"));
        println!(
            "  {}",
            style::dim("aivo run quick --model other  # override one flag")
        );
        println!("  {}", style::dim("aivo alias rm quick"));
        println!("  {}", style::dim("aivo alias --json"));
    }
}

fn validate_alias_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Alias name must not be empty".to_string());
    }
    if name.starts_with('-') || name.contains('=') || name.contains(char::is_whitespace) {
        return Err(format!(
            "Alias name '{name}' must not start with '-', contain '=', or contain whitespace"
        ));
    }
    if RESERVED_ALIAS_NAMES.contains(&name) {
        return Err(format!(
            "'{name}' is reserved (collides with a built-in command, shortcut, or tool name). Pick a different alias name."
        ));
    }
    Ok(())
}

fn report_set(name: &str, new_value: &AliasValue, prev: Option<AliasValue>) {
    match prev {
        Some(old) => println!(
            "Updated {} {} {} (was {})",
            style::cyan(name),
            style::dim("->"),
            new_value,
            style::dim(old.to_string())
        ),
        None => println!(
            "Created {} {} {}",
            style::cyan(name),
            style::dim("->"),
            new_value
        ),
    }
}

/// Detects cycles in the alias map starting from `start`.
fn has_cycle(aliases: &std::collections::HashMap<String, String>, start: &str) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut current = start;
    while let Some(target) = aliases.get(current) {
        if !seen.insert(current) {
            return true;
        }
        current = target;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn has_cycle_no_cycle() {
        let mut m = HashMap::new();
        m.insert("fast".to_string(), "claude-haiku".to_string());
        m.insert("best".to_string(), "claude-sonnet".to_string());
        assert!(!has_cycle(&m, "fast"));
        assert!(!has_cycle(&m, "best"));
    }

    #[test]
    fn has_cycle_self_reference() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), "a".to_string());
        assert!(has_cycle(&m, "a"));
    }

    #[test]
    fn has_cycle_two_hop() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), "b".to_string());
        m.insert("b".to_string(), "a".to_string());
        assert!(has_cycle(&m, "a"));
        assert!(has_cycle(&m, "b"));
    }

    #[test]
    fn has_cycle_three_hop() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), "b".to_string());
        m.insert("b".to_string(), "c".to_string());
        m.insert("c".to_string(), "a".to_string());
        assert!(has_cycle(&m, "a"));
    }

    #[test]
    fn has_cycle_chain_no_cycle() {
        let mut m = HashMap::new();
        m.insert("a".to_string(), "b".to_string());
        m.insert("b".to_string(), "c".to_string());
        // c doesn't map to anything, so no cycle
        assert!(!has_cycle(&m, "a"));
    }

    #[tokio::test]
    async fn set_and_get_aliases() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        assert!(store.get_aliases().await.unwrap().is_empty());

        store
            .set_alias("fast".to_string(), "claude-haiku".to_string())
            .await
            .unwrap();
        let aliases = store.get_aliases().await.unwrap();
        assert_eq!(aliases.get("fast").unwrap(), "claude-haiku");
    }

    #[tokio::test]
    async fn remove_alias_returns_old_value() {
        use crate::services::session_store::AliasValue;
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        store
            .set_alias("fast".to_string(), "haiku".to_string())
            .await
            .unwrap();
        let removed = store.remove_alias("fast").await.unwrap();
        assert_eq!(removed, Some(AliasValue::Model("haiku".to_string())));

        let removed_again = store.remove_alias("fast").await.unwrap();
        assert_eq!(removed_again, None);
    }

    #[tokio::test]
    async fn resolve_alias_follows_chain() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        store
            .set_alias("quick".to_string(), "fast".to_string())
            .await
            .unwrap();
        store
            .set_alias("fast".to_string(), "claude-haiku".to_string())
            .await
            .unwrap();

        let resolved = store.resolve_alias("quick").await.unwrap();
        assert_eq!(resolved, "claude-haiku");
    }

    #[tokio::test]
    async fn resolve_alias_detects_cycle() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        store
            .set_alias("a".to_string(), "b".to_string())
            .await
            .unwrap();
        store
            .set_alias("b".to_string(), "a".to_string())
            .await
            .unwrap();

        let result = store.resolve_alias("a").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resolve_alias_passthrough_non_alias() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        let resolved = store.resolve_alias("claude-sonnet-4-6").await.unwrap();
        assert_eq!(resolved, "claude-sonnet-4-6");
    }

    #[test]
    fn validate_alias_name_accepts_normal_names() {
        assert!(validate_alias_name("fast").is_ok());
        assert!(validate_alias_name("k25").is_ok());
        assert!(validate_alias_name("my-alias").is_ok());
    }

    #[test]
    fn validate_alias_name_rejects_reserved_names() {
        for reserved in &["claude", "run", "alias", "use", "ping", "ls"] {
            assert!(
                validate_alias_name(reserved).is_err(),
                "expected '{reserved}' to be rejected"
            );
        }
    }

    #[test]
    fn validate_alias_name_rejects_bad_shapes() {
        assert!(validate_alias_name("").is_err());
        assert!(validate_alias_name("--flag").is_err());
        assert!(validate_alias_name("name=value").is_err());
        assert!(validate_alias_name("has space").is_err());
    }

    #[tokio::test]
    async fn set_bundle_alias_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        let bundle = BundleAlias {
            tool: "claude".to_string(),
            args: vec![
                "--key".to_string(),
                "work".to_string(),
                "--model".to_string(),
                "fast".to_string(),
            ],
        };
        store
            .set_bundle("quick".to_string(), bundle.clone())
            .await
            .unwrap();

        let all = store.list_alias_values().await.unwrap();
        assert_eq!(all.get("quick"), Some(&AliasValue::Bundle(bundle)));

        // get_aliases() filters out bundle entries, so model resolution paths
        // never see them.
        let model_aliases = store.get_aliases().await.unwrap();
        assert!(!model_aliases.contains_key("quick"));
    }

    #[tokio::test]
    async fn set_bundle_then_overwrite_with_model_returns_prev_bundle() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(dir.path().join("config.json"));

        let bundle = BundleAlias {
            tool: "claude".to_string(),
            args: vec![],
        };
        store
            .set_bundle("dev".to_string(), bundle.clone())
            .await
            .unwrap();
        let prev = store
            .set_alias("dev".to_string(), "claude-sonnet".to_string())
            .await
            .unwrap();
        assert_eq!(prev, Some(AliasValue::Bundle(bundle)));
    }
}
