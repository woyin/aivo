use anyhow::Result;
use std::collections::HashSet;
use std::io::{self, Write};
use std::time::Duration;

use crate::cli::LogsArgs;
use crate::commands::chat::format_time_ago_short;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::log_store::{LogEntry, LogQuery};
use crate::services::system_env;
use crate::style;

const SEPARATOR: &str = "\u{00b7}";

pub struct LogsCommand {
    session_store: SessionStore,
}

impl LogsCommand {
    pub fn new(session_store: SessionStore) -> Self {
        Self { session_store }
    }

    pub async fn execute(&self, args: LogsArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, args: LogsArgs) -> Result<ExitCode> {
        validate_args(&args)?;
        match args.action.as_deref() {
            Some("status") => self.show_status(&args).await,
            Some("show") => self.show_entry(&args).await,
            Some(other) => anyhow::bail!(
                "Unknown subcommand '{other}'. Valid: show <id>, status. Use -s <query> to search."
            ),
            None => self.list_entries(&args).await,
        }
    }

    async fn show_status(&self, args: &LogsArgs) -> Result<ExitCode> {
        ensure_no_target(args, "status")?;
        let status = self.session_store.logs().status().await?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&status)?);
            return Ok(ExitCode::Success);
        }

        println!(
            "{} {} {} {} {} {}",
            style::bold(status.total_entries.to_string()),
            style::dim("events"),
            style::dim(SEPARATOR),
            style::dim(format_bytes(status.file_size_bytes)),
            style::dim(SEPARATOR),
            style::dim(system_env::collapse_tilde(&status.path)),
        );

        if status.counts_by_source.is_empty() {
            println!();
            println!("{}", style::dim("No log entries recorded yet."));
            return Ok(ExitCode::Success);
        }

        // Flatten sources + tools into a single list sorted by count desc.
        // A source with 2+ tools is replaced by its individual tool rows so
        // the breakdown reads as one flat ranking rather than a hierarchy.
        let mut rows: Vec<(String, u64)> = Vec::new();
        for source in status.counts_by_source {
            if source.tools.len() >= 2 {
                for tool in source.tools {
                    rows.push((tool.tool, tool.count));
                }
            } else {
                rows.push((source.source, source.count));
            }
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));

        let name_width = rows.iter().map(|(n, _)| n.len()).max().unwrap_or(0);
        let max_count = rows.iter().map(|(_, c)| *c).max().unwrap_or(0);
        let count_width = max_count.to_string().len();
        // `total` guards `count / total` in print_activity_row against zero.
        let total = status.total_entries.max(1);
        const BAR_WIDTH: usize = 32;

        println!();
        for (name, count) in rows {
            print_activity_row(
                &name,
                count,
                total,
                max_count,
                name_width,
                count_width,
                BAR_WIDTH,
            );
        }
        Ok(ExitCode::Success)
    }

    async fn show_entry(&self, args: &LogsArgs) -> Result<ExitCode> {
        let id = args
            .target
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Usage: aivo logs show <id>"))?;
        let entry = self
            .session_store
            .logs()
            .get_by_reference(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("No log entry with id '{}'", id))?;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&entry)?);
            return Ok(ExitCode::Success);
        }

        print_entry(&entry);
        Ok(ExitCode::Success)
    }

    async fn list_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        if args.target.is_some() {
            anyhow::bail!("Unexpected target without an action. Use `aivo logs show <id>`");
        }
        if args.watch {
            return self.watch_entries(args).await;
        }
        let entries = self.fetch_entries(args).await?;

        if args.json {
            println!("{}", serde_json::to_string_pretty(&entries)?);
            return Ok(ExitCode::Success);
        }

        render_text_entries(entries, args.limit);
        Ok(ExitCode::Success)
    }

    async fn watch_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        const WATCH_INTERVAL: Duration = Duration::from_secs(1);
        let mut seen_ids = HashSet::new();

        loop {
            let entries = self.fetch_entries(args).await?;

            if args.jsonl {
                let mut ordered = entries;
                ordered.reverse();
                for entry in ordered {
                    if seen_ids.insert(entry.id.clone()) {
                        println!("{}", serde_json::to_string(&entry)?);
                    }
                }
                io::stdout().flush()?;
            } else {
                print!("\x1b[2J\x1b[H");
                println!(
                    "{} {}",
                    style::bold("Watching logs"),
                    style::dim("(Ctrl+C to stop)")
                );
                println!();
                render_text_entries(entries, args.limit);
                io::stdout().flush()?;
            }

            tokio::time::sleep(WATCH_INTERVAL).await;
        }
    }

    async fn fetch_entries(&self, args: &LogsArgs) -> Result<Vec<LogEntry>> {
        // Over-fetch to compensate for run event collapsing (start+finish pairs)
        let query_limit = if args.watch {
            args.limit.saturating_mul(5)
        } else if args.json {
            args.limit
        } else {
            args.limit.saturating_mul(3)
        };
        let entries = self
            .session_store
            .logs()
            .list(LogQuery {
                limit: query_limit,
                search: args.search.clone(),
                by: args.by.clone(),
                model: args.model.clone(),
                key_query: args.key.clone(),
                cwd: args.cwd.clone(),
                since: args.since.clone(),
                until: args.until.clone(),
                errors_only: args.errors,
            })
            .await?;
        Ok(entries)
    }

    pub fn print_help() {
        println!("{} aivo logs [COMMAND] [OPTIONS]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Query local SQLite logs for chat, run, and serve activity.")
        );
        println!();
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        println!("{}", style::bold("Commands:"));
        print_opt("(default)", "List recent log entries (newest first)");
        print_opt("show <id>", "Show one entry in detail");
        print_opt("status", "Show entry counts, size, and database path");
        println!();
        println!("{}", style::bold("Filters:"));
        print_opt(
            "-n, --limit <N>",
            "Maximum number of rows to show (default: 20)",
        );
        print_opt("--json", "Output JSON");
        print_opt("--watch", "Continuously refresh matching logs");
        print_opt("--jsonl", "Emit newly seen entries as JSONL while watching");
        print_opt("-s, --search <query>", "Search title/body text");
        print_opt(
            "--by <name>",
            "Filter by chat, run, serve, or tool (claude, codex, gemini, opencode, pi)",
        );
        print_opt("--model <model>", "Filter by model substring");
        print_opt("-k, --key <id|name>", "Filter by saved key ID or name");
        print_opt("--cwd <path>", "Filter by working directory substring");
        print_opt("--since <time>", "Only show entries on or after this time");
        print_opt("--until <time>", "Only show entries on or before this time");
        print_opt("--errors", "Only show HTTP >= 400 or non-zero exit code");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo logs"));
        println!("  {}", style::dim("aivo logs --by chat -n 5"));
        println!("  {}", style::dim("aivo logs --by claude --errors"));
        println!("  {}", style::dim("aivo logs --by run --watch"));
        println!("  {}", style::dim("aivo logs --watch --jsonl"));
        println!("  {}", style::dim("aivo logs show 7m2q8k4v9cpr"));
        println!("  {}", style::dim("aivo logs status"));
    }
}

fn ensure_no_target(args: &LogsArgs, action: &str) -> Result<()> {
    if args.target.is_some() {
        anyhow::bail!("`aivo logs {}` does not take a target", action);
    }
    Ok(())
}

fn validate_args(args: &LogsArgs) -> Result<()> {
    if args.jsonl && !args.watch {
        anyhow::bail!("--jsonl requires --watch");
    }
    if args.json && args.watch {
        anyhow::bail!("--json cannot be combined with --watch; use --jsonl for watch mode");
    }
    if args.json && args.jsonl {
        anyhow::bail!("--json and --jsonl cannot be combined");
    }
    if args.watch && args.action.is_some() {
        anyhow::bail!("--watch is only supported for `aivo logs` list output");
    }
    Ok(())
}

fn render_text_entries(entries: Vec<LogEntry>, limit: usize) {
    if entries.is_empty() {
        println!("{}", style::dim("No log entries found."));
        return;
    }

    for entry in collapse_run_events(entries, limit) {
        print_summary(&entry);
    }
}

fn print_summary(entry: &LogEntry) {
    let display_id = display_id(entry);
    let time_ago = format_time_ago_short(&entry.ts_utc);
    let detail = match entry.source.as_str() {
        "chat" => {
            let title = entry.title.clone().unwrap_or_else(|| "(chat)".to_string());
            let tokens = format_token_summary(entry);
            if tokens.is_empty() {
                title
            } else {
                format!("{title}  {tokens}")
            }
        }
        "run" => {
            let tool = entry.tool.as_deref().unwrap_or("run");
            let model = entry
                .model
                .clone()
                .unwrap_or_else(|| "(tool default)".to_string());
            let state = match entry.phase.as_deref() {
                Some("started") => "running".to_string(),
                _ => entry
                    .exit_code
                    .map(|code| format!("exit={code}"))
                    .unwrap_or_else(|| "exit=?".to_string()),
            };
            let duration = entry
                .duration_ms
                .map(|ms| format!(" ({})", format_duration_ms(ms)))
                .unwrap_or_default();
            format!("{tool} {model} {state}{duration}")
        }
        "serve" => {
            let title = entry.title.clone().unwrap_or_else(|| "request".to_string());
            let status = entry
                .status_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "?".to_string());
            let duration = entry
                .duration_ms
                .map(|ms| format!(" ({})", format_duration_ms(ms)))
                .unwrap_or_default();
            format!("{title} status={status}{duration}")
        }
        _ => entry.title.clone().unwrap_or_else(|| entry.kind.clone()),
    };
    // Column order matches `aivo context`: age first, id second, then detail.
    println!(
        "{} {} {} {}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(display_id),
        style::yellow(format!("[{}]", entry.source)),
        detail
    );
}

fn format_token_summary(entry: &LogEntry) -> String {
    match (entry.input_tokens, entry.output_tokens) {
        (Some(input), Some(output)) if input > 0 || output > 0 => {
            style::dim(format!("({input}\u{2192}{output} tokens)"))
        }
        _ => String::new(),
    }
}

fn print_activity_row(
    name: &str,
    count: u64,
    total: u64,
    max_count: u64,
    name_width: usize,
    count_width: usize,
    bar_width: usize,
) {
    let pct = ((count as f64 / total as f64) * 100.0).round() as u64;
    let row = format!(
        "  {:<name_w$}  {:>count_w$}  {:>2}%  {}",
        name,
        count,
        pct,
        style::bar(count, max_count, bar_width),
        name_w = name_width,
        count_w = count_width,
    );
    println!("{}", style::cyan(row));
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut idx = 0;
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    // Guard against rounding to "1024 X" when a larger unit exists.
    if value >= 1023.95 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    if (value - value.round()).abs() < 0.05 {
        format!("{:.0} {}", value, UNITS[idx])
    } else {
        format!("{:.1} {}", value, UNITS[idx])
    }
}

fn format_duration_ms(ms: i64) -> String {
    let ms = ms.unsigned_abs();
    match ms {
        0..=999 => format!("{ms}ms"),
        1000..=59_999 => format!("{:.1}s", ms as f64 / 1000.0),
        60_000..=3_599_999 => {
            let minutes = ms / 60_000;
            let seconds = (ms % 60_000) / 1000;
            if seconds == 0 {
                format!("{minutes}m")
            } else {
                format!("{minutes}m {seconds}s")
            }
        }
        _ => {
            let hours = ms / 3_600_000;
            let minutes = (ms % 3_600_000) / 60_000;
            if minutes == 0 {
                format!("{hours}h")
            } else {
                format!("{hours}h {minutes}m")
            }
        }
    }
}

fn display_id(entry: &LogEntry) -> &str {
    if entry.source == "run"
        && let Some(group_id) = entry.event_group_id.as_deref()
    {
        return group_id;
    }
    &entry.id
}

fn print_entry(entry: &LogEntry) {
    println!("{} {}", style::bold("id:"), entry.id);
    println!("{} {}", style::bold("time:"), entry.ts_utc);
    println!("{} {}", style::bold("source:"), entry.source);
    println!("{} {}", style::bold("kind:"), entry.kind);
    if let Some(value) = &entry.event_group_id {
        println!("{} {}", style::bold("group:"), value);
    }
    if let Some(value) = &entry.phase {
        println!("{} {}", style::bold("phase:"), value);
    }
    if let Some(value) = &entry.key_name {
        println!("{} {}", style::bold("key:"), value);
    }
    if let Some(value) = &entry.key_id {
        println!("{} {}", style::bold("key id:"), value);
    }
    if let Some(value) = &entry.base_url {
        println!("{} {}", style::bold("base url:"), style::dim(value));
    }
    if let Some(value) = &entry.tool {
        println!("{} {}", style::bold("tool:"), value);
    }
    if let Some(value) = &entry.model {
        println!("{} {}", style::bold("model:"), value);
    }
    if let Some(value) = &entry.cwd {
        println!("{} {}", style::bold("cwd:"), style::dim(value));
    }
    if let Some(value) = &entry.session_id {
        println!("{} {}", style::bold("session:"), value);
    }
    if let Some(value) = entry.status_code {
        println!("{} {}", style::bold("status:"), value);
    }
    if let Some(value) = entry.exit_code {
        println!("{} {}", style::bold("exit code:"), value);
    }
    if let Some(value) = entry.duration_ms {
        println!("{} {}", style::bold("duration:"), format_duration_ms(value));
    }
    if entry.input_tokens.is_some() || entry.output_tokens.is_some() {
        println!(
            "{} input={} output={} cache_read={} cache_write={}",
            style::bold("tokens:"),
            entry.input_tokens.unwrap_or(0),
            entry.output_tokens.unwrap_or(0),
            entry.cache_read_input_tokens.unwrap_or(0),
            entry.cache_creation_input_tokens.unwrap_or(0)
        );
    }
    if let Some(value) = &entry.title {
        println!("{} {}", style::bold("title:"), value);
    }
    if let Some(value) = &entry.body_text {
        println!();
        println!("{}", style::bold("Body:"));
        println!("{}", value);
    }
    if let Some(value) = &entry.payload_json {
        println!();
        println!("{}", style::bold("Payload:"));
        println!(
            "{}",
            serde_json::to_string_pretty(value).unwrap_or_default()
        );
    }
}

fn collapse_run_events(entries: Vec<LogEntry>, limit: usize) -> Vec<LogEntry> {
    let mut seen_groups = HashSet::new();
    let mut collapsed = Vec::new();

    for entry in entries {
        if entry.source == "run"
            && let Some(group_id) = &entry.event_group_id
            && !seen_groups.insert(group_id.clone())
        {
            continue;
        }
        collapsed.push(entry);
        if collapsed.len() >= limit {
            break;
        }
    }

    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> LogsArgs {
        LogsArgs {
            action: None,
            target: None,
            limit: 20,
            json: false,
            watch: false,
            jsonl: false,
            search: None,
            by: None,
            model: None,
            key: None,
            cwd: None,
            since: None,
            until: None,
            errors: false,
        }
    }

    #[test]
    fn validate_args_rejects_jsonl_without_watch() {
        let mut args = base_args();
        args.jsonl = true;
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn validate_args_rejects_json_with_watch() {
        let mut args = base_args();
        args.json = true;
        args.watch = true;
        assert!(validate_args(&args).is_err());
    }

    fn test_entry(id: &str, ts: &str, source: &str) -> LogEntry {
        LogEntry {
            id: id.to_string(),
            ts_utc: ts.to_string(),
            source: source.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn collapse_run_events_prefers_latest_group_event() {
        let entries = vec![
            LogEntry {
                event_group_id: Some("run-1".to_string()),
                phase: Some("finished".to_string()),
                exit_code: Some(0),
                ..test_entry("2", "2026-03-27T12:00:01Z", "run")
            },
            LogEntry {
                event_group_id: Some("run-1".to_string()),
                phase: Some("started".to_string()),
                ..test_entry("1", "2026-03-27T12:00:00Z", "run")
            },
        ];

        let collapsed = collapse_run_events(entries, 20);
        assert_eq!(collapsed.len(), 1);
        assert_eq!(collapsed[0].id, "2");
    }

    #[test]
    fn display_id_prefers_run_group_id() {
        let entry = LogEntry {
            event_group_id: Some("group123".to_string()),
            phase: Some("finished".to_string()),
            exit_code: Some(0),
            ..test_entry("event123", "2026-03-27T12:00:01Z", "run")
        };

        assert_eq!(display_id(&entry), "group123");
    }

    #[test]
    fn format_bytes_ranges() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1 KB");
        assert_eq!(format_bytes(688_128), "672 KB");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1_572_864), "1.5 MB");
        assert_eq!(format_bytes(2_147_483_648), "2 GB");
        // Boundary: would round to "1024 KB" under naive formatting — promote to MB.
        assert_eq!(format_bytes(1_048_575), "1 MB");
    }

    #[test]
    fn format_duration_ms_ranges() {
        assert_eq!(format_duration_ms(0), "0ms");
        assert_eq!(format_duration_ms(500), "500ms");
        assert_eq!(format_duration_ms(1234), "1.2s");
        assert_eq!(format_duration_ms(60_000), "1m");
        assert_eq!(format_duration_ms(90_000), "1m 30s");
        assert_eq!(format_duration_ms(3_600_000), "1h");
        assert_eq!(format_duration_ms(5_400_000), "1h 30m");
    }
}
