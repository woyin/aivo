//! Command handlers module for the aivo CLI.
//! Provides implementations for all CLI commands.

use unicode_width::UnicodeWidthChar;

use crate::services::ai_launcher::PreparedLaunch;
use crate::services::environment_injector::redact_env_value;
use crate::style;

/// Shown (only on explicit picker requests) when the selected key has no
/// fetchable model list. The tool still launches with its own default.
pub(crate) const NO_MODEL_LIST_HINT: &str =
    "No model list available; launching with the tool's default. Use --model <name> to override.";

/// Prints `NO_MODEL_LIST_HINT` to stderr when an explicit picker request
/// can't actually open a picker (no model list, or no TTY).
pub(crate) fn print_no_model_list_hint() {
    eprintln!("  {} {}", style::dim("note:"), NO_MODEL_LIST_HINT);
}

/// Strips trailing slashes and a bare `/v1` suffix from a provider base URL.
pub(crate) fn normalize_base_url(url: &str) -> &str {
    let url = url.trim_end_matches('/');
    url.strip_suffix("/v1").unwrap_or(url)
}

/// Truncates `text` to its first line, then to `max_cols` terminal columns
/// with an ellipsis. Width-aware: CJK chars count as 2 columns so picker
/// rows don't overflow the terminal and wrap to a second row.
pub(crate) fn trim_to_one_line(text: &str, max_cols: usize) -> String {
    let one_line: String = text.lines().next().unwrap_or("").chars().collect();
    let total_cols: usize = one_line
        .chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum();
    if total_cols <= max_cols {
        return one_line;
    }
    let budget = max_cols.saturating_sub(1);
    let mut acc = 0usize;
    let mut truncated = String::new();
    for c in one_line.chars() {
        let w = UnicodeWidthChar::width(c).unwrap_or(0);
        if acc + w > budget {
            break;
        }
        truncated.push(c);
        acc += w;
    }
    truncated.push('…');
    truncated
}

/// Truncates a URL for display while preserving both the prefix and suffix.
pub(crate) fn truncate_url_for_display(url: &str, max_len: usize) -> String {
    let char_count = url.chars().count();
    if char_count <= max_len {
        return url.to_string();
    }
    let keep_suffix = 15.min(max_len / 3);
    let keep_prefix = max_len.saturating_sub(keep_suffix + 1);
    let prefix: String = url.chars().take(keep_prefix).collect();
    let suffix: String = url.chars().skip(char_count - keep_suffix).collect();
    format!("{prefix}…{suffix}")
}

/// Provider-cell label for the first-party aivo key: the plan label (or slug) when
/// known, else the free-tier sentinel. One colour for every plan.
pub(crate) fn starter_provider_label(
    cached_plan: Option<&str>,
    cached_label: Option<&str>,
) -> String {
    match cached_plan.map(str::trim).filter(|s| !s.is_empty()) {
        Some(plan) => cached_label
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(plan)
            .to_string(),
        None => crate::constants::AIVO_STARTER_SENTINEL.to_string(),
    }
}

pub mod account;
pub mod agents;
pub mod alias;
pub mod code;
pub(crate) mod code_agent_oneshot;
pub(crate) mod code_request_builder;
pub(crate) mod code_response_parser;
pub(crate) mod code_tui_format;
pub mod guide;
pub mod hf;
pub mod info;
pub mod keys;
pub(crate) mod keys_ui;
pub mod login;
pub mod logs;
pub mod mcp;
pub mod models;
pub mod packs;
pub mod plugins;
pub mod run;
pub mod serve;
pub mod share;
pub mod skills;
pub mod start;
pub mod stats;
pub mod update;

pub use account::AccountCommand;
pub use alias::AliasCommand;
pub use code::CodeCommand;
pub use info::InfoCommand;
pub use keys::KeysCommand;
pub use login::{LoginCommand, LogoutCommand};
pub use logs::LogsCommand;
pub use models::ModelsCommand;
pub use plugins::PluginsCommand;
pub use run::RunCommand;
pub use serve::{ServeCommand, ServeParams};
pub use share::ShareCommand;
pub use start::{StartCommand, StartFlowArgs};
pub use stats::StatsCommand;
pub use update::UpdateCommand;

pub(crate) fn print_launch_preview(plan: &PreparedLaunch) {
    println!(
        "{} {}",
        style::bold("Tool:"),
        style::cyan(plan.tool.as_str())
    );
    println!(
        "{} {} {}",
        style::bold("Key:"),
        style::cyan(plan.key.display_name()),
        style::dim(format!("({})", plan.key.base_url))
    );
    println!(
        "{} {}",
        style::bold("Model:"),
        plan.model.as_deref().unwrap_or("(tool default)")
    );
    println!(
        "{} {}",
        style::bold("Command:"),
        format_shell_command(&plan.command, &plan.args)
    );
    println!();
    println!("{}", style::bold("Environment:"));
    if plan.env_vars.is_empty() {
        println!("  {}", style::dim("(none)"));
    } else {
        let mut keys: Vec<_> = plan.env_vars.keys().collect();
        keys.sort();
        for key in keys {
            println!("  {}={}", key, redact_env_value(key, &plan.env_vars[key]));
        }
    }

    if !plan.notes.is_empty() {
        println!();
        println!("{}", style::bold("Notes:"));
        for note in &plan.notes {
            println!("  {} {}", style::arrow_symbol(), note);
        }
    }
}

pub(crate) fn format_shell_command(command: &str, args: &[String]) -> String {
    let mut parts = vec![shell_quote(command)];
    parts.extend(args.iter().map(|arg| shell_quote(arg)));
    parts.join(" ")
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/' | '.' | ':' | '='))
    {
        return value.to_string();
    }

    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::{starter_provider_label, truncate_url_for_display};

    #[test]
    fn starter_provider_label_falls_back_to_sentinel() {
        assert_eq!(
            starter_provider_label(None, None),
            crate::constants::AIVO_STARTER_SENTINEL
        );
        assert_eq!(
            starter_provider_label(Some(""), Some("ignored")),
            crate::constants::AIVO_STARTER_SENTINEL
        );
    }

    #[test]
    fn starter_provider_label_uses_plan_slug() {
        assert_eq!(starter_provider_label(Some("aivo-pro"), None), "aivo-pro");
    }

    #[test]
    fn starter_provider_label_prefers_server_label() {
        assert_eq!(
            starter_provider_label(Some("aivo-friend"), Some("Friend")),
            "Friend"
        );
        assert_eq!(
            starter_provider_label(Some("aivo-pro"), Some("  ")),
            "aivo-pro"
        );
    }

    #[test]
    fn truncate_url_for_display_preserves_short_urls() {
        assert_eq!(
            truncate_url_for_display("https://api.example.com/v1", 50),
            "https://api.example.com/v1"
        );
    }

    #[test]
    fn truncate_url_for_display_shortens_long_urls() {
        let url = "https://very-long-provider-host.example.com/path/to/a/deeply/nested/resource/v1";
        let truncated = truncate_url_for_display(url, 32);

        assert_eq!(
            truncated,
            format!("{}…{}", &url[..21], &url[url.len() - 10..])
        );
    }
}
