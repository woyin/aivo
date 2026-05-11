use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::io::{self, Write};
use std::time::Duration;

use crate::cli::LogsArgs;
use crate::commands::chat::format_time_ago_short;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::amp_threads;
use crate::services::context_ingest::{self, IngestOptions};
use crate::services::id_compact::compact_id;
use crate::services::log_store::{LogEntry, LogQuery};
use crate::services::project_id::Thread;
use crate::services::system_env;
use crate::style;

/// One row in the unified `aivo logs` listing. Mixes `logs.db` events
/// (chat/run/serve), native CLI sessions (claude/codex/gemini/pi/opencode),
/// and amp threads. Each variant carries its own provenance and time, so
/// the merge sort + filter pipeline can treat them uniformly.
#[derive(Debug, Clone)]
enum UnifiedRow {
    // LogEntry is large (~1KB worth of optional fields); box to keep the
    // enum's stack footprint reasonable for `Vec<UnifiedRow>` allocation.
    Log(Box<LogEntry>),
    Native(Thread),
    Amp(AmpRow),
}

#[derive(Debug, Clone)]
struct AmpRow {
    id: String,
    title: Option<String>,
    updated_at: DateTime<Utc>,
    message_count: u64,
}

impl AmpRow {
    fn from_value(v: &Value) -> Option<Self> {
        let id = v.get("id")?.as_str()?.to_string();
        let title = v.get("title").and_then(|x| x.as_str()).map(str::to_string);
        let updated_at = v
            .get("updatedAt")
            .and_then(|x| x.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);
        let message_count = v.get("messageCount").and_then(|x| x.as_u64()).unwrap_or(0);
        Some(Self {
            id,
            title,
            updated_at,
            message_count,
        })
    }
}

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
            Some("share") => self.share_session(args).await,
            Some(other) => anyhow::bail!(
                "Unknown subcommand '{other}'. Valid: show <id>, share [id], status. Use -s <query> to search."
            ),
            None => self.list_entries(&args).await,
        }
    }

    async fn share_session(&self, args: LogsArgs) -> Result<ExitCode> {
        use crate::cli::ShareArgs;
        use crate::commands::ShareCommand;

        let share_args = ShareArgs {
            session_id: args.target,
            live: args.live,
            no_redact: args.no_redact,
            all: args.all,
            open: args.open,
            debug_local_only: args.debug_local_only,
        };
        let cmd = ShareCommand::new(self.session_store.clone());
        Ok(cmd.execute(share_args).await)
    }

    async fn show_status(&self, args: &LogsArgs) -> Result<ExitCode> {
        ensure_no_target(args, "status")?;
        let status = self.session_store.logs().status().await?;

        // Aggregate native CLI + amp counts alongside logs.db.
        let native_counts = native_session_counts().await;
        let amp_count = amp_threads::list_threads(&amp_threads::default_threads_dir(), 10_000)
            .await
            .len() as u64;
        let native_total: u64 = native_counts.iter().map(|(_, c)| *c).sum();
        let grand_total = status.total_entries + native_total + amp_count;

        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "logs_db": status,
                    "native_sessions": native_counts
                        .iter()
                        .map(|(cli, count)| json!({"cli": cli, "count": count}))
                        .collect::<Vec<_>>(),
                    "amp_threads": amp_count,
                    "grand_total": grand_total,
                }))?
            );
            return Ok(ExitCode::Success);
        }

        println!(
            "{} {} {} {} {} {}",
            style::bold(grand_total.to_string()),
            style::dim("rows"),
            style::dim(SEPARATOR),
            style::dim(format_bytes(status.file_size_bytes)),
            style::dim(SEPARATOR),
            style::dim(system_env::collapse_tilde(&status.path)),
        );

        if grand_total == 0 {
            println!();
            println!("{}", style::dim("Nothing recorded yet."));
            return Ok(ExitCode::Success);
        }

        // Flatten the breakdown across all three sources into one ranking.
        // Prefix logs.db rows with their source ("run claude" vs bare
        // "claude") so they're distinguishable from native session counts.
        let mut rows: Vec<(String, u64)> = Vec::new();
        for source in status.counts_by_source {
            if source.tools.len() >= 2 {
                for tool in source.tools {
                    rows.push((format!("{} {}", source.source, tool.tool), tool.count));
                }
            } else {
                rows.push((source.source, source.count));
            }
        }
        for (cli, count) in native_counts {
            rows.push((format!("{cli} sessions"), count));
        }
        if amp_count > 0 {
            rows.push(("amp threads".to_string(), amp_count));
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

        // 1. Try logs.db first with prefix matching — `aivo logs` displays
        //    8-char ids and copy-pasting one needs to find the full row.
        //    For chat/run/serve, the metadata view here is the right output
        //    (don't auto-drill into chat sessions; that's `aivo logs share`'s
        //    job — `show` is the row-level inspector).
        let logs_hits = self.session_store.logs().find_by_id_prefix(id, 5).await?;
        if logs_hits.len() > 1 {
            let summary = logs_hits
                .iter()
                .map(|e| format!("{} [{}]", &e.id, e.source))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "ambiguous logs.db prefix '{}' — matched: {}. Re-run with a longer prefix.",
                id,
                summary
            );
        }
        if let Some(entry) = logs_hits.into_iter().next() {
            if args.json {
                println!("{}", serde_json::to_string_pretty(&entry)?);
            } else {
                print_entry(&entry, &self.session_store).await;
            }
            return Ok(ExitCode::Success);
        }

        // 2. Fall back to native CLI / amp via the share resolver, which
        //    already does prefix-matching across all 7 sources and surfaces
        //    cross-source ambiguity. We don't need the full payload — the
        //    metadata + first/last messages from `Thread`-style summaries
        //    are enough — but reusing one resolver is simpler than carving
        //    out a duplicate.
        use crate::services::share_resolver::{ResolverContext, resolve_session};
        let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let ctx = ResolverContext::from_system(cwd, self.session_store.clone());
        match resolve_session(id, &ctx).await {
            Ok(resolved) => {
                if args.json {
                    println!("{}", serde_json::to_string_pretty(&resolved.payload)?);
                } else {
                    print_share_payload(&resolved.payload);
                }
                Ok(ExitCode::Success)
            }
            Err(e) => Err(anyhow::anyhow!(
                "No log entry, native session, or amp thread matching '{}'.\n  ({})",
                id,
                e
            )),
        }
    }

    async fn list_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        if args.target.is_some() {
            anyhow::bail!("Unexpected target without an action. Use `aivo logs show <id>`");
        }
        if args.watch {
            return self.watch_entries(args).await;
        }

        // Paint a spinner only if the fetch takes longer than `SPINNER_DELAY`
        // — fast invocations stay flicker-free. JSON output skips the
        // spinner entirely so machine-readable callers see clean stdout.
        const SPINNER_DELAY: std::time::Duration = std::time::Duration::from_millis(250);
        let rows = if args.json {
            self.fetch_unified_rows(args).await?
        } else {
            let fetch = self.fetch_unified_rows(args);
            tokio::pin!(fetch);
            tokio::select! {
                r = &mut fetch => r?,
                _ = tokio::time::sleep(SPINNER_DELAY) => {
                    let (spinning, handle) = style::start_spinner(Some(" loading…"));
                    let r = (&mut fetch).await;
                    style::stop_spinner(&spinning);
                    let _ = handle.await;
                    r?
                }
            }
        };

        if args.json {
            println!("{}", serde_json::to_string_pretty(&unified_to_json(&rows))?);
            return Ok(ExitCode::Success);
        }

        render_unified_rows(&rows);
        Ok(ExitCode::Success)
    }

    async fn watch_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        const WATCH_INTERVAL: Duration = Duration::from_secs(1);
        let mut seen_ids: HashSet<String> = HashSet::new();

        loop {
            let rows = self.fetch_unified_rows(args).await?;

            if args.jsonl {
                let mut ordered = rows.clone();
                ordered.reverse();
                for row in ordered {
                    let key = unified_id_key(&row);
                    if seen_ids.insert(key) {
                        println!("{}", serde_json::to_string(&unified_row_to_json(&row))?);
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
                render_unified_rows(&rows);
                io::stdout().flush()?;
            }

            tokio::time::sleep(WATCH_INTERVAL).await;
        }
    }

    /// Pulls rows from all three data sources concurrently, then k-way-merges
    /// the already-newest-first streams to take only `--limit` rows. No full
    /// union, no global sort.
    async fn fetch_unified_rows(&self, args: &LogsArgs) -> Result<Vec<UnifiedRow>> {
        let plan = SourcePlan::from_args(args);
        // Default = current cwd. `--all` opts out, `--cwd <path>` overrides.
        // Expand `.` / `~/` so explicit shorthands work.
        let cwd_filter: Option<String> = if args.all {
            None
        } else if let Some(explicit) = args.cwd.as_deref() {
            Some(expand_cwd_filter(explicit))
        } else {
            system_env::current_dir().map(|p| p.to_string_lossy().to_string())
        };

        // Run the three source fetches concurrently — they touch disjoint
        // backends (sqlite, native session jsonl files, amp's thread dir),
        // so the prior sequential awaits cost us ~3× the slowest source.
        let (log_rows, native_rows, amp_rows) = tokio::try_join!(
            fetch_logs_rows(&self.session_store, args, &plan, cwd_filter.clone()),
            fetch_native_rows(args, &plan, cwd_filter.clone()),
            fetch_amp_rows(args, &plan, cwd_filter.as_deref()),
        )?;

        Ok(merge_unified(log_rows, native_rows, amp_rows, args.limit))
    }

    pub fn print_help(action: Option<&str>) {
        match action {
            Some("show") => print_help_show(),
            Some("share") => print_help_share(),
            Some("status") => print_help_status(),
            _ => print_help_overview(),
        }
    }
}

fn logs_help_row(flag: &str, desc: &str) {
    println!(
        "  {}{}",
        style::cyan(format!("{:<26}", flag)),
        style::dim(desc)
    );
}

fn print_help_overview() {
    println!("{} aivo logs [COMMAND] [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Unified activity feed: aivo's own events (chat, run, serve), native CLI")
    );
    println!(
        "{}",
        style::dim("sessions (claude, codex, gemini, pi, opencode), and amp threads.")
    );
    println!();
    println!("{}", style::bold("Commands:"));
    logs_help_row(
        "(default)",
        "List recent rows from all sources (newest first)",
    );
    logs_help_row(
        "show <id>",
        "Show one row in detail; accepts logs.db ids, native session ids, or T-…",
    );
    logs_help_row(
        "share [id]",
        "Share a session via tunneled viewer URL; omit id to open the picker",
    );
    logs_help_row("status", "Show counts and storage paths across sources");
    println!();
    println!("{}", style::bold("Filters:"));
    logs_help_row("-n, --limit <N>", "Maximum rows after merge (default: 20)");
    logs_help_row(
        "--json",
        "Output JSON (tagged union: log_entry|native_session|amp_thread)",
    );
    logs_help_row(
        "--watch",
        "Continuously refresh (1s poll across all sources)",
    );
    logs_help_row("--jsonl", "Emit newly seen rows as JSONL while watching");
    logs_help_row("-s, --search <query>", "Search title/topic/body text");
    logs_help_row(
        "--by <name>",
        "chat | run | serve | claude | codex | gemini | pi | opencode | amp | native",
    );
    logs_help_row(
        "--model <model>",
        "Filter by model substring (logs.db only)",
    );
    logs_help_row("-k, --key <id|name>", "Filter by saved key (logs.db only)");
    logs_help_row(
        "--cwd <path>",
        "Filter to a specific cwd (default: current cwd)",
    );
    logs_help_row("-a, --all", "Show rows from every project (no cwd filter)");
    logs_help_row("--since <time>", "Only show rows on or after this time");
    logs_help_row("--until <time>", "Only show rows on or before this time");
    logs_help_row(
        "--errors",
        "Only HTTP >= 400 or non-zero exit (logs.db only)",
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!(
        "  {}",
        style::dim("aivo logs                     # current project, newest first")
    );
    println!(
        "  {}",
        style::dim("aivo logs --all               # every project")
    );
    println!(
        "  {}",
        style::dim("aivo logs --by claude         # claude run-events + claude native sessions")
    );
    println!(
        "  {}",
        style::dim("aivo logs --by native         # only native CLI sessions")
    );
    println!(
        "  {}",
        style::dim("aivo logs --by amp            # only amp threads")
    );
    println!(
        "  {}",
        style::dim("aivo logs show 1335c631       # any unique id prefix works")
    );
    println!(
        "  {}",
        style::dim("aivo logs share               # pick a session and share it")
    );
    println!(
        "  {}",
        style::dim("aivo logs share 1335c631      # share by id prefix")
    );
    println!("  {}", style::dim("aivo logs --watch --jsonl"));
}

fn print_help_show() {
    println!("{} aivo logs show <ID>", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Show full detail for a single row: logs.db id, native session id, or T-… thread."
        )
    );
    println!(
        "{}",
        style::dim("Any unique id prefix works (matches the prefix shown in `aivo logs`).")
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo logs show 1335c631"));
    println!("  {}", style::dim("aivo logs show T-abc123"));
}

fn print_help_share() {
    println!("{} aivo logs share [ID] [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim("Share a session via a tunneled viewer URL. Default is a one-shot snapshot")
    );
    println!(
        "{}",
        style::dim("of the session at share time; secrets and $HOME paths are redacted.")
    );
    println!(
        "{}",
        style::dim("Omit the id to open the picker. `aivo share` is an alias for this command.")
    );
    println!();
    println!("{}", style::bold("Options:"));
    logs_help_row(
        "--live",
        "Follow ongoing changes (default: snapshot at share time)",
    );
    logs_help_row(
        "--no-redact",
        "Skip redaction (API keys, OAuth tokens, $HOME, secret-shaped env)",
    );
    logs_help_row(
        "--all",
        "Pick from sessions in every project, not just the current cwd",
    );
    logs_help_row(
        "--open",
        "Open the share URL in the default browser once ready",
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo logs share"));
    println!("  {}", style::dim("aivo logs share 1335c631"));
    println!("  {}", style::dim("aivo logs share --live --open"));
}

fn print_help_status() {
    println!("{} aivo logs status", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Print row counts and storage paths across logs.db, native sessions, and amp threads."
        )
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo logs status"));
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
    let is_share = args.action.as_deref() == Some("share");
    if !is_share && (args.live || args.no_redact || args.open || args.debug_local_only) {
        anyhow::bail!(
            "--live, --no-redact, --open, --debug-local-only only apply to `aivo logs share`"
        );
    }
    Ok(())
}

fn render_unified_rows(rows: &[UnifiedRow]) {
    if rows.is_empty() {
        println!("{}", style::dim("No entries found."));
        return;
    }
    let detail_width = available_detail_width();
    for row in rows {
        match row {
            UnifiedRow::Log(e) => print_summary(e, detail_width),
            UnifiedRow::Native(t) => print_native_summary(t, detail_width),
            UnifiedRow::Amp(a) => print_amp_summary(a, detail_width),
        }
    }
}

/// Detail-column width that keeps each row on a single terminal line.
/// `prefix` covers age (5) + id (8) + bracket (10) + 3 separator spaces = 26;
/// the trailing `+1` leaves headroom for the cursor so terminals that
/// auto-wrap on the *last* column don't push every row to two lines.
/// Clamped to a comfortable reading band so very wide terminals don't
/// produce unscannable 200-char rows.
fn available_detail_width() -> usize {
    const PREFIX: usize = 5 + 1 + ID_COL_WIDTH + 1 + BRACKET_COL_WIDTH + 1;
    let cols = console::Term::stdout().size().1 as usize;
    cols.saturating_sub(PREFIX + 1).clamp(30, 80)
}

/// Width of the id column. 8 chars matches git-style short SHA — enough
/// entropy to avoid collisions across a user's history while keeping the
/// table compact. The resolver matches any unique prefix, so longer ids
/// still work when pasted from other tools.
const ID_COL_WIDTH: usize = 8;
/// Width of the source bracket column, padded for `[opencode]` (10 chars).
/// Keeps detail-column alignment consistent across all sources.
const BRACKET_COL_WIDTH: usize = 10;

fn print_native_summary(t: &Thread, detail_width: usize) {
    let time_ago = format_time_ago_short_dt(t.updated_at);
    let id = compact_id(&t.session_id, ID_COL_WIDTH);
    let topic = trim_to_one_line(&t.topic, detail_width);
    println!(
        "{} {} {} {}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(format!("{:<width$}", id, width = ID_COL_WIDTH)),
        style::magenta(format!(
            "{:<width$}",
            format!("[{}]", t.cli),
            width = BRACKET_COL_WIDTH
        )),
        topic
    );
}

fn print_amp_summary(a: &AmpRow, detail_width: usize) {
    let time_ago = format_time_ago_short_dt(a.updated_at);
    let id = compact_id(&a.id, ID_COL_WIDTH);
    let title = a
        .title
        .clone()
        .unwrap_or_else(|| format!("(amp thread, {} messages)", a.message_count));
    let title = trim_to_one_line(&title, detail_width);
    println!(
        "{} {} {} {}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(format!("{:<width$}", id, width = ID_COL_WIDTH)),
        style::magenta(format!("{:<width$}", "[amp]", width = BRACKET_COL_WIDTH)),
        title
    );
}

/// "5m" / "2d" — for `Thread`/`AmpRow` which have already-parsed timestamps.
fn format_time_ago_short_dt(ts: DateTime<Utc>) -> String {
    format_time_ago_short(&ts.to_rfc3339())
}

/// Collapse every kind of line/whitespace separator into a single space, then
/// truncate to `max_chars` with an ellipsis. Bulletproof against topics with
/// `\r`-only line breaks, Unicode line separators, tabs, or other control
/// chars that would otherwise leak a second line into the table.
pub(crate) fn trim_to_one_line(text: &str, max_chars: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = false;
    for c in text.chars() {
        let is_separator = matches!(
            c,
            '\n' | '\r' | '\t' | '\x0B' | '\x0C' | '\u{2028}' | '\u{2029}'
        ) || c.is_control();
        let ch = if is_separator { ' ' } else { c };
        if ch == ' ' {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    let one_line = out.trim();
    if one_line.chars().count() > max_chars {
        let prefix: String = one_line.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", prefix)
    } else {
        one_line.to_string()
    }
}

fn print_summary(entry: &LogEntry, detail_width: usize) {
    let display_id = display_id(entry);
    let time_ago = format_time_ago_short(&entry.ts_utc);
    // (text, dim_suffix). Trimming runs on the plain text *before* styling
    // so `is_control()`-based whitespace collapse can't strip ANSI escape
    // bytes out of the suffix and leave bare `[2m…[0m` literals on screen.
    let (text, dim_suffix): (String, String) = match entry.source.as_str() {
        "chat" => (
            entry.title.clone().unwrap_or_else(|| "(chat)".to_string()),
            format_token_summary(entry),
        ),
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
            (format!("{tool} {model} {state}{duration}"), String::new())
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
            (format!("{title} status={status}{duration}"), String::new())
        }
        _ => (
            entry.title.clone().unwrap_or_else(|| entry.kind.clone()),
            String::new(),
        ),
    };
    // Reserve space for the suffix when sizing the title — token counts are
    // short and high-signal, so prefer trimming the title over dropping them.
    // Floor at 10 so a pathological wide suffix can't squeeze the title to
    // nothing.
    let text_budget = if dim_suffix.is_empty() {
        detail_width
    } else {
        detail_width
            .saturating_sub(dim_suffix.chars().count() + 1)
            .max(10)
    };
    let text = trim_to_one_line(&text, text_budget);
    let detail = if dim_suffix.is_empty() {
        text
    } else {
        format!("{text} {}", style::dim(dim_suffix))
    };
    // Same column shape as native/amp rows: age (5) · id (8) · bracket (10) · detail.
    // `{:<W.W}` truncates a too-long id to W chars then pads it to W — gives
    // a clean column even when logs.db's full 12-char id is longer than W.
    println!(
        "{} {} {} {}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(format!(
            "{:<width$.width$}",
            display_id,
            width = ID_COL_WIDTH
        )),
        style::yellow(format!(
            "{:<width$}",
            format!("[{}]", entry.source),
            width = BRACKET_COL_WIDTH
        )),
        detail
    );
}

/// Plain (unstyled) token-count fragment — caller applies dim *after*
/// trimming so the escape bytes aren't fed through `trim_to_one_line`.
fn format_token_summary(entry: &LogEntry) -> String {
    match (entry.input_tokens, entry.output_tokens) {
        (Some(input), Some(output)) if input > 0 || output > 0 => {
            format!("({input}\u{2192}{output} tokens)")
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

async fn print_entry(entry: &LogEntry, store: &SessionStore) {
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
    } else if let Some(inferred) = inferred_chat_session(entry, store).await {
        println!(
            "{} {} {}",
            style::bold("session:"),
            inferred,
            style::dim("(inferred)")
        );
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

/// Old `chat` rows (written before `LogEvent.session_id` existed) have no
/// stored linkage to their chat session. Match cwd + key_id and pick the
/// session whose `updated_at` is closest to the event's `ts_utc` — chat
/// sessions are persisted within ~1s of the log row, so the match is reliable
/// when both still exist on disk.
async fn inferred_chat_session(entry: &LogEntry, store: &SessionStore) -> Option<String> {
    if entry.source != "chat" {
        return None;
    }
    let cwd = entry.cwd.as_deref()?;
    let ts = DateTime::parse_from_rfc3339(&entry.ts_utc)
        .ok()?
        .with_timezone(&Utc);
    store
        .find_chat_session_near(cwd, entry.key_id.as_deref(), ts, 60)
        .await
        .ok()
        .flatten()
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

// ---------------------------------------------------------------------------
// Per-source fetchers. Each returns rows already sorted newest-first so the
// k-way merge below can pop the global newest without a full sort.
// ---------------------------------------------------------------------------

async fn fetch_logs_rows(
    store: &SessionStore,
    args: &LogsArgs,
    plan: &SourcePlan,
    cwd_filter: Option<String>,
) -> Result<Vec<LogEntry>> {
    if !plan.include_logs() {
        return Ok(Vec::new());
    }
    // Over-fetch to compensate for run-pair collapsing.
    let query_limit = if args.watch {
        args.limit.saturating_mul(5)
    } else if args.json {
        args.limit
    } else {
        args.limit.saturating_mul(3)
    };
    let entries = store
        .logs()
        .list(LogQuery {
            limit: query_limit,
            search: args.search.clone(),
            by: args.by.clone(),
            model: args.model.clone(),
            key_query: args.key.clone(),
            cwd: cwd_filter,
            since: normalize_time_filter(args.since.as_deref()),
            until: normalize_time_filter(args.until.as_deref()),
            errors_only: args.errors,
        })
        .await?;
    // Run events are emitted as start+finish pairs sharing an event_group_id;
    // collapse here too so the unified listing doesn't show both halves.
    Ok(collapse_run_events(entries, args.limit.saturating_mul(3)))
}

// Perf: counter-intuitively, the project-scoped ingester is slower than global
// on a multi-project machine — codex/gemini walk the same global tree either
// way, but scoped *rejects* every non-matching file (forcing the walk to
// continue past the cap). Global stops at cap quickly. So we always walk
// globally and post-filter by cwd. The 14-day age cap bounds the worst case.
async fn fetch_native_rows(
    args: &LogsArgs,
    plan: &SourcePlan,
    cwd_filter: Option<String>,
) -> Result<Vec<Thread>> {
    if !plan.include_native() {
        return Ok(Vec::new());
    }
    let opts = IngestOptions {
        max_age_days: if args.since.is_some() || args.until.is_some() {
            None // explicit time filter takes over
        } else {
            Some(14) // matches the previous `aivo context` default
        },
        // Push --since down to the ingester so jsonl files whose mtime is
        // older than the cutoff can be skipped without parsing. --until can't
        // be pushed down (mtime is an upper bound on updated_at).
        min_updated_at: args.since.as_deref().and_then(parse_loose_time),
        max_per_source: Some(args.limit.saturating_mul(2).max(50)),
    };
    let mut all = context_ingest::ingest_native_sessions_global(opts).await?;
    all.retain(|t| native_passes_filters(t, args, cwd_filter.as_deref()));
    Ok(all)
}

// Amp threads have no cwd concept. If the user filtered by --cwd, exclude amp
// from the listing entirely (it can't possibly match).
async fn fetch_amp_rows(
    args: &LogsArgs,
    plan: &SourcePlan,
    cwd_filter: Option<&str>,
) -> Result<Vec<AmpRow>> {
    if !plan.include_amp() || cwd_filter.is_some() {
        return Ok(Vec::new());
    }
    let amp_dir = amp_threads::default_threads_dir();
    let raw = amp_threads::list_threads(&amp_dir, args.limit.saturating_mul(2).max(50)).await;
    Ok(raw
        .iter()
        .filter_map(AmpRow::from_value)
        .filter(|a| amp_passes_filters(a, args))
        .collect())
}

/// Three-way merge of newest-first streams. Pops the source with the newest
/// head until `limit` rows are emitted — never materializes the full union.
fn merge_unified(
    logs: Vec<LogEntry>,
    native: Vec<Thread>,
    amp: Vec<AmpRow>,
    limit: usize,
) -> Vec<UnifiedRow> {
    let mut logs = logs.into_iter().peekable();
    let mut native = native.into_iter().peekable();
    let mut amp = amp.into_iter().peekable();
    let mut out: Vec<UnifiedRow> = Vec::with_capacity(limit);

    while out.len() < limit {
        let heads = [
            logs.peek().map(|e| parse_log_ts(&e.ts_utc)),
            native.peek().map(|t| t.updated_at),
            amp.peek().map(|a| a.updated_at),
        ];
        let winner = heads
            .into_iter()
            .enumerate()
            .filter_map(|(i, ts)| ts.map(|t| (t, i)))
            .max_by_key(|(t, _)| *t)
            .map(|(_, i)| i);

        match winner {
            None => break,
            Some(0) => out.push(UnifiedRow::Log(Box::new(logs.next().unwrap()))),
            Some(1) => out.push(UnifiedRow::Native(native.next().unwrap())),
            Some(2) => out.push(UnifiedRow::Amp(amp.next().unwrap())),
            Some(_) => unreachable!(),
        }
    }
    out
}

fn parse_log_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

// ---------------------------------------------------------------------------
// SourcePlan — translates `--by` and the strict-logs.db filters into a
// per-source eligibility set. Centralizes the "which sources can this query
// possibly include" decision so fetch_unified_rows stays readable.
// ---------------------------------------------------------------------------

struct SourcePlan {
    logs: bool,
    native: bool,
    amp: bool,
}

impl SourcePlan {
    fn from_args(args: &LogsArgs) -> Self {
        // Filters that only make sense for logs.db rows force a strict mode.
        let strict_logs = args.errors || args.key.is_some() || args.model.is_some();

        let by = args.by.as_deref().map(str::to_ascii_lowercase);
        let by = by.as_deref();

        // Determine eligibility per source.
        let logs;
        let native;
        let amp;
        match by {
            // Bare logs.db sources.
            Some("chat") | Some("run") | Some("serve") => {
                logs = true;
                native = false;
                amp = false;
            }
            // "amp" is amp-only.
            Some("amp") => {
                logs = false;
                native = false;
                amp = true;
            }
            // "native" = all native CLI sessions, no logs.db, no amp.
            Some("native") => {
                logs = false;
                native = true;
                amp = false;
            }
            // CLI names: matching logs.db `tool` substring + native rows of that cli.
            Some("claude") | Some("codex") | Some("gemini") | Some("opencode") | Some("pi") => {
                logs = true;
                native = !strict_logs;
                amp = false;
            }
            // No --by: all sources, modulo strict-logs.
            None | Some(_) => {
                logs = true;
                native = !strict_logs;
                amp = !strict_logs;
            }
        }

        Self { logs, native, amp }
    }

    fn include_logs(&self) -> bool {
        self.logs
    }
    fn include_native(&self) -> bool {
        self.native
    }
    fn include_amp(&self) -> bool {
        self.amp
    }
}

/// Apply the user-facing filters that aren't already encoded by SourcePlan
/// to a single native `Thread`. `cwd_filter` is pre-expanded (`.` → cwd,
/// `~/` → home). Returns true if the thread should be kept.
fn native_passes_filters(t: &Thread, args: &LogsArgs, cwd_filter: Option<&str>) -> bool {
    // `--by claude` (or other cli name) must match the thread's cli.
    if let Some(by) = args.by.as_deref() {
        let by = by.to_ascii_lowercase();
        match by.as_str() {
            "native" => {} // accepts every native cli
            "claude" | "codex" | "gemini" | "opencode" | "pi" if t.cli != by => {
                return false;
            }
            _ => {}
        }
    }
    if let Some(needle) = cwd_filter {
        let cwd_match = t
            .cwd
            .as_deref()
            .map(|c| c.contains(needle))
            .unwrap_or(false);
        if !cwd_match {
            return false;
        }
    }
    if let Some(needle) = args.search.as_deref() {
        let n = needle.to_ascii_lowercase();
        let hay = format!(
            "{} {} {}",
            t.topic.to_ascii_lowercase(),
            t.last_response.to_ascii_lowercase(),
            t.cli.to_ascii_lowercase()
        );
        if !hay.contains(&n) {
            return false;
        }
    }
    if let Some(since) = args.since.as_deref()
        && let Some(cutoff) = parse_loose_time(since)
        && t.updated_at < cutoff
    {
        return false;
    }
    if let Some(until) = args.until.as_deref()
        && let Some(cutoff) = parse_loose_time(until)
        && t.updated_at > cutoff
    {
        return false;
    }
    true
}

/// Expand `--cwd` shorthand: `.` → current cwd, `~/` → home, otherwise
/// passthrough. Used so users can type `aivo logs --cwd .` without thinking
/// about path semantics.
fn expand_cwd_filter(input: &str) -> String {
    if input == "."
        && let Some(cwd) = system_env::current_dir()
    {
        return cwd.to_string_lossy().to_string();
    }
    if let Some(rest) = input.strip_prefix("~/")
        && let Some(home) = system_env::home_dir()
    {
        return home.join(rest).to_string_lossy().to_string();
    }
    if input == "~"
        && let Some(home) = system_env::home_dir()
    {
        return home.to_string_lossy().to_string();
    }
    input.to_string()
}

fn amp_passes_filters(a: &AmpRow, args: &LogsArgs) -> bool {
    if let Some(needle) = args.search.as_deref() {
        let n = needle.to_ascii_lowercase();
        let title = a.title.as_deref().unwrap_or("").to_ascii_lowercase();
        if !title.contains(&n) && !a.id.to_ascii_lowercase().contains(&n) {
            return false;
        }
    }
    if let Some(since) = args.since.as_deref()
        && let Some(cutoff) = parse_loose_time(since)
        && a.updated_at < cutoff
    {
        return false;
    }
    if let Some(until) = args.until.as_deref()
        && let Some(cutoff) = parse_loose_time(until)
        && a.updated_at > cutoff
    {
        return false;
    }
    true
}

/// Best-effort time parse: tries relative durations (`2h`, `30m`, `1d`, `1w`,
/// `45s`) first, then RFC3339, then a date-only `YYYY-MM-DD` parse. Relative
/// values are interpreted as "ago" so `--since 2h` means "in the last 2 hours".
fn parse_loose_time(s: &str) -> Option<DateTime<Utc>> {
    let trimmed = s.trim();
    if let Some(dur) = parse_relative_duration(trimmed) {
        return Some(Utc::now() - dur);
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(d) = chrono::NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return d.and_hms_opt(0, 0, 0).map(|n| Utc.from_utc_datetime(&n));
    }
    None
}

/// Parse `<int><unit>` where unit ∈ {s, m, h, d, w}. Returns None for any
/// other shape so callers fall through to absolute-time parsers.
fn parse_relative_duration(s: &str) -> Option<chrono::Duration> {
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit())?);
    if num.is_empty() {
        return None;
    }
    let n: i64 = num.parse().ok()?;
    match unit {
        "s" => Some(chrono::Duration::seconds(n)),
        "m" => Some(chrono::Duration::minutes(n)),
        "h" => Some(chrono::Duration::hours(n)),
        "d" => Some(chrono::Duration::days(n)),
        "w" => Some(chrono::Duration::weeks(n)),
        _ => None,
    }
}

/// Normalize a user-supplied `--since`/`--until` value to an RFC3339 string
/// so LogStore's SQL `ts_utc >= ?` comparison works. Leaves unparsable input
/// alone — the SQL string compare will then just match nothing, which beats
/// silently treating `2h` as "everything".
fn normalize_time_filter(raw: Option<&str>) -> Option<String> {
    let raw = raw?;
    parse_loose_time(raw)
        .map(|dt| dt.to_rfc3339())
        .or_else(|| Some(raw.to_string()))
}

// ---------------------------------------------------------------------------
// JSON shape for --json / --jsonl
// ---------------------------------------------------------------------------

fn unified_to_json(rows: &[UnifiedRow]) -> Vec<Value> {
    rows.iter().map(unified_row_to_json).collect()
}

fn unified_row_to_json(row: &UnifiedRow) -> Value {
    match row {
        UnifiedRow::Log(e) => {
            let mut v = serde_json::to_value(e).unwrap_or(Value::Null);
            if let Some(map) = v.as_object_mut() {
                map.insert("kind".to_string(), Value::String("log_entry".into()));
            }
            v
        }
        UnifiedRow::Native(t) => json!({
            "kind": "native_session",
            "cli": t.cli,
            "session_id": t.session_id,
            "source_path": t.source_path,
            "topic": t.topic,
            "last_response": t.last_response,
            "updated_at": t.updated_at.to_rfc3339(),
            "cwd": t.cwd,
        }),
        UnifiedRow::Amp(a) => json!({
            "kind": "amp_thread",
            "id": a.id,
            "title": a.title,
            "updated_at": a.updated_at.to_rfc3339(),
            "message_count": a.message_count,
        }),
    }
}

/// Tally native session counts per CLI for `aivo logs status`. Walks the
/// global ingester with caps lifted; capped at a sensible upper bound so
/// status never spends too long on huge histories.
async fn native_session_counts() -> Vec<(String, u64)> {
    let opts = IngestOptions {
        max_age_days: None,
        min_updated_at: None,
        max_per_source: Some(10_000),
    };
    let threads = context_ingest::ingest_native_sessions_global(opts)
        .await
        .unwrap_or_default();
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for t in threads {
        *counts.entry(t.cli).or_insert(0) += 1;
    }
    let mut out: Vec<(String, u64)> = counts.into_iter().collect();
    out.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    out
}

/// Stable identity per row for watch-mode dedup. Logs.db `run` events
/// collapse on `event_group_id`; otherwise the row's own id is unique.
fn unified_id_key(row: &UnifiedRow) -> String {
    match row {
        UnifiedRow::Log(e) => {
            if e.source == "run"
                && let Some(g) = &e.event_group_id
            {
                format!("log:run:{g}")
            } else {
                format!("log:{}", e.id)
            }
        }
        UnifiedRow::Native(t) => format!("native:{}:{}", t.cli, t.session_id),
        UnifiedRow::Amp(a) => format!("amp:{}", a.id),
    }
}

/// Pretty-print a `SharePayload` returned by `share_resolver` for
/// `aivo logs show <native-or-amp-id>`. Mirrors `print_entry`'s style.
fn print_share_payload(p: &crate::services::share_payload::SharePayload) {
    println!("{} {}", style::bold("source:"), p.source_cli);
    println!("{} {}", style::bold("session:"), p.session_id);
    if let Some(model) = &p.model {
        println!("{} {}", style::bold("model:"), model);
    }
    if let Some(root) = &p.project.root {
        println!("{} {}", style::bold("cwd:"), style::dim(root));
    }
    if let Some(updated) = p.updated_at {
        println!("{} {}", style::bold("updated:"), updated.to_rfc3339());
    }
    println!("{} {}", style::bold("messages:"), p.messages.len());
    if let Some(summary) = &p.meta.redaction_summary
        && !summary.is_empty()
    {
        let s = summary
            .iter()
            .map(|h| format!("{} {}", h.count, h.category))
            .collect::<Vec<_>>()
            .join(", ");
        println!("{} {}", style::bold("redacted:"), style::dim(s));
    }
    println!();
    println!(
        "{}",
        style::dim("(use `aivo logs share <id>` to open a viewer)")
    );
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
            all: false,
            since: None,
            until: None,
            errors: false,
            live: false,
            no_redact: false,
            open: false,
            debug_local_only: false,
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

    fn test_thread(session_id: &str, ts: DateTime<Utc>) -> Thread {
        Thread {
            cli: "claude".into(),
            session_id: session_id.into(),
            source_path: String::new(),
            topic: String::new(),
            last_response: String::new(),
            updated_at: ts,
            cwd: None,
        }
    }

    fn test_amp(id: &str, ts: DateTime<Utc>) -> AmpRow {
        AmpRow {
            id: id.into(),
            title: None,
            updated_at: ts,
            message_count: 0,
        }
    }

    fn unified_key(row: &UnifiedRow) -> String {
        match row {
            UnifiedRow::Log(e) => format!("log:{}", e.id),
            UnifiedRow::Native(t) => format!("native:{}", t.session_id),
            UnifiedRow::Amp(a) => format!("amp:{}", a.id),
        }
    }

    #[test]
    fn merge_unified_interleaves_newest_first_across_sources() {
        let logs = vec![
            test_entry("L1", "2026-05-01T10:00:00Z", "chat"),
            test_entry("L2", "2026-05-01T08:00:00Z", "chat"),
        ];
        let native = vec![
            test_thread("N1", "2026-05-01T09:30:00Z".parse().unwrap()),
            test_thread("N2", "2026-05-01T07:00:00Z".parse().unwrap()),
        ];
        let amp = vec![test_amp("A1", "2026-05-01T09:00:00Z".parse().unwrap())];

        let merged = merge_unified(logs, native, amp, 10);
        let order: Vec<String> = merged.iter().map(unified_key).collect();
        assert_eq!(
            order,
            vec!["log:L1", "native:N1", "amp:A1", "log:L2", "native:N2",]
        );
    }

    #[test]
    fn merge_unified_caps_at_limit() {
        let logs = vec![
            test_entry("L1", "2026-05-01T10:00:00Z", "chat"),
            test_entry("L2", "2026-05-01T09:00:00Z", "chat"),
            test_entry("L3", "2026-05-01T08:00:00Z", "chat"),
        ];
        let native = vec![test_thread("N1", "2026-05-01T07:00:00Z".parse().unwrap())];

        let merged = merge_unified(logs, native, Vec::new(), 2);
        assert_eq!(merged.len(), 2);
        let order: Vec<String> = merged.iter().map(unified_key).collect();
        assert_eq!(order, vec!["log:L1", "log:L2"]);
    }

    #[test]
    fn merge_unified_handles_empty_sources() {
        let merged: Vec<UnifiedRow> = merge_unified(Vec::new(), Vec::new(), Vec::new(), 5);
        assert!(merged.is_empty());
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
