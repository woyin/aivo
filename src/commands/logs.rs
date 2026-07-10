use anyhow::Result;
use chrono::{DateTime, TimeZone, Utc};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::time::Duration;

use crate::cli::LogsArgs;
use crate::commands::code::format_time_ago_short;
use crate::constants::KNOWN_TOOLS;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::context_ingest::{self, IngestOptions};
use crate::services::id_compact::compact_id;
use crate::services::log_store::{LogEntry, LogQuery, RunMeta};
use crate::services::project_id::Thread;
use crate::services::session_store::SessionIndexEntry;
use crate::services::system_env;
use crate::style;
use unicode_width::UnicodeWidthChar;

/// One row in the unified `aivo logs` listing. Mixes `logs.db` events
/// (chat/run/serve) and native CLI sessions (claude/codex/gemini/pi/opencode).
/// Each variant carries its own provenance and time, so the merge sort +
/// filter pipeline can treat them uniformly.
#[derive(Debug, Clone)]
pub(crate) enum UnifiedRow {
    // LogEntry is large (~1KB worth of optional fields); box to keep the
    // enum's stack footprint reasonable for `Vec<UnifiedRow>` allocation.
    Log(Box<LogEntry>),
    Native(Thread),
}

impl UnifiedRow {
    /// Stable identifier suitable for the share resolver. For `Log/run` this
    /// is the event group id (matches what `aivo logs` prints), otherwise the
    /// native session_id / logs row id.
    pub(crate) fn id(&self) -> String {
        match self {
            UnifiedRow::Log(e) => display_id(e).to_string(),
            UnifiedRow::Native(t) => t.session_id.clone(),
        }
    }

    /// `"code" | "run" | "serve" | "claude" | …`. Matches what `aivo
    /// logs` displays in the bracket column.
    pub(crate) fn source_label(&self) -> &str {
        match self {
            UnifiedRow::Log(e) => log_bracket_label(e),
            UnifiedRow::Native(t) => t.cli.as_str(),
        }
    }

    /// True iff this row is a chat event referencing a `session_id` that no
    /// longer has a file on disk — `logs.db` outlives the chat session file
    /// because it records durable events, while session files can be
    /// deleted. Used to tag stale rows in the listing and to short-circuit
    /// share attempts with a friendlier error.
    pub(crate) fn is_orphan_chat(&self, orphan_chat_ids: &HashSet<String>) -> bool {
        match self {
            UnifiedRow::Log(e) if is_code_source(&e.source) => e
                .session_id
                .as_deref()
                .is_some_and(|sid| orphan_chat_ids.contains(sid)),
            _ => false,
        }
    }

    /// Plain-text formatted row matching `aivo logs`'s column shape
    /// (`<age:5> <id:id_width> <bracket:10> <detail>`). No ANSI escapes —
    /// callers (e.g. `FuzzySelect`) apply their own highlighting. Orphan
    /// chat rows (session file deleted) get a `(file deleted)` suffix;
    /// native rows with `[run]` metadata pick up a ` · <key> · exit <N>`
    /// suffix.
    pub(crate) fn picker_label(
        &self,
        id_width: usize,
        detail_width: usize,
        orphan_chat_ids: &HashSet<String>,
        run_meta: &RunMetaIndex,
    ) -> String {
        let (age, id, detail) = match self {
            UnifiedRow::Log(e) => (
                format_time_ago_short(&e.ts_utc),
                display_id(e).to_string(),
                log_row_detail(e),
            ),
            UnifiedRow::Native(t) => (
                format_time_ago_short_dt(t.updated_at),
                compact_id(&t.session_id, id_width),
                t.topic.clone(),
            ),
        };
        let bracket = format!("[{}]", self.source_label());
        let orphan_suffix = if self.is_orphan_chat(orphan_chat_ids) {
            " (file deleted)"
        } else {
            ""
        };
        let run_suffix_plain = match self {
            UnifiedRow::Native(t) => run_meta
                .get(&t.session_id)
                .map(format_run_meta_suffix_plain)
                .unwrap_or_default(),
            _ => String::new(),
        };
        let detail_budget = detail_width
            .saturating_sub(orphan_suffix.chars().count())
            .saturating_sub(run_suffix_plain.chars().count());
        let detail = trim_to_one_line(&detail, detail_budget);
        format!(
            "{:>5} {:<id_w$.id_w$} {:<br_w$} {}{}{}",
            age,
            id,
            bracket,
            detail,
            run_suffix_plain,
            orphan_suffix,
            id_w = id_width,
            br_w = BRACKET_COL_WIDTH,
        )
    }
}

/// Smallest prefix length in `ID_COL_WIDTH..=ID_COL_WIDTH_MAX` that keeps
/// every row's displayed id unique. UUIDv7 session ids share 10+ leading
/// hex chars for same-minute creates, so the 8-char floor isn't always
/// enough; residual collisions past the cap fall through to the picker.
pub(crate) fn min_unique_id_width(rows: &[UnifiedRow]) -> usize {
    if rows.len() < 2 {
        return ID_COL_WIDTH;
    }
    let ids: Vec<String> = rows
        .iter()
        .map(|r| compact_id(&r.id(), ID_COL_WIDTH_MAX))
        .collect();
    for width in ID_COL_WIDTH..=ID_COL_WIDTH_MAX {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut unique = true;
        for id in &ids {
            let take = id
                .char_indices()
                .nth(width)
                .map(|(i, _)| i)
                .unwrap_or(id.len());
            if !seen.insert(&id[..take]) {
                unique = false;
                break;
            }
        }
        if unique {
            return width;
        }
    }
    ID_COL_WIDTH_MAX
}

/// A `[run]` row for a plugin coding-agent — a tool that isn't a native CLI
/// (e.g. omp, amp). These have no native session source, so their run row is
/// their only representation in `aivo logs`. We surface them like first-class
/// agents: tool name in the bracket column, not a generic `[run]`.
fn is_plugin_run(entry: &LogEntry) -> bool {
    entry.source == "run"
        && entry
            .tool
            .as_deref()
            .is_some_and(|t| !KNOWN_TOOLS.contains(&t))
}

/// The built-in agent's logs.db source. `"code"` is written post-rename;
/// `"chat"` is the pre-rename value still on disk in existing users' logs.db.
pub(crate) fn is_code_source(source: &str) -> bool {
    matches!(source, "code" | "chat")
}

/// Bracket-column label for a logs.db row. Plugin coding-agent runs use their
/// tool name (`[omp]`) so they read like native agents; the built-in agent
/// normalizes to `[code]` (legacy `chat` rows included); everything else uses
/// its source (`[run]`, `[serve]`).
fn log_bracket_label(entry: &LogEntry) -> &str {
    if is_plugin_run(entry) {
        entry.tool.as_deref().unwrap_or("run")
    } else if is_code_source(&entry.source) {
        "code"
    } else {
        entry.source.as_str()
    }
}

/// Plain-text detail string for a logs.db row — mirrors the `text` half of
/// `print_summary` (token suffix included for chat rows). Kept here so
/// picker labels stay in sync with `aivo logs` printing.
fn log_row_detail(entry: &LogEntry) -> String {
    match entry.source.as_str() {
        "chat" | "code" => {
            let title = entry.title.clone().unwrap_or_else(|| "(code)".to_string());
            let suffix = format_token_summary(entry);
            if suffix.is_empty() {
                title
            } else {
                format!("{title} {suffix}")
            }
        }
        "run" => {
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
            // Plugin runs put the tool name in the bracket column; native
            // `--by run` rows keep it inline since their bracket is `[run]`.
            if is_plugin_run(entry) {
                format!("{model} {state}{duration}")
            } else {
                let tool = entry.tool.as_deref().unwrap_or("run");
                format!("{tool} {model} {state}{duration}")
            }
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
    }
}

/// Picker-facing column-width helper. Same math as
/// `available_detail_width` so picker labels and `aivo logs` rows line up.
pub(crate) fn picker_detail_width(term_cols: usize, id_width: usize) -> usize {
    let prefix = 5 + 1 + id_width + 1 + BRACKET_COL_WIDTH + 1;
    term_cols.saturating_sub(prefix + 4).clamp(20, 80)
}

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
                crate::errors::exit_code_for_error(&e)
            }
        }
    }

    async fn execute_internal(&self, args: LogsArgs) -> Result<ExitCode> {
        validate_args(&args)?;
        match args.action.as_deref() {
            Some("show") => self.show_entry(&args).await,
            Some("share") => self.share_session(args).await,
            Some("prune") => self.prune_orphans(&args).await,
            None | Some("list" | "ls") => self.list_entries(&args).await,
            Some(other) => anyhow::bail!(
                "Unknown action '{other}'. Valid actions: list, show, share, prune.\nRun `aivo logs --help` for details (use -s <query> to search)."
            ),
        }
    }

    /// `aivo logs prune` — delete chat events in logs.db whose session file
    /// has been removed. Confirms unless `--force` is set.
    async fn prune_orphans(&self, args: &LogsArgs) -> Result<ExitCode> {
        let orphan_ids = compute_orphan_code_ids(&self.session_store).await;
        if orphan_ids.is_empty() {
            println!("{} No orphan code events found.", style::green("✓"));
            return Ok(ExitCode::Success);
        }
        let mut ids: Vec<String> = orphan_ids.into_iter().collect();
        ids.sort();
        println!(
            "Found {} chat session(s) with deleted files:",
            style::bold(ids.len().to_string()),
        );
        for id in &ids {
            let prefix: String = id.chars().take(8).collect();
            println!("  {} {}", style::cyan(prefix), style::dim(id));
        }
        if !args.force {
            print!("Delete the orphan logs.db rows for these sessions? [y/N] ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if !input.trim().eq_ignore_ascii_case("y") {
                println!("{}", style::dim("Cancelled."));
                return Ok(ExitCode::Success);
            }
        }
        let deleted = self
            .session_store
            .logs()
            .delete_code_events_by_session_ids(&ids)
            .await?;
        println!(
            "{} Deleted {} code event(s) from logs.db.",
            style::green("✓"),
            style::bold(deleted.to_string()),
        );
        Ok(ExitCode::Success)
    }

    async fn share_session(&self, args: LogsArgs) -> Result<ExitCode> {
        use crate::cli::ShareArgs;
        use crate::commands::ShareCommand;

        let share_args = ShareArgs {
            session_id: args.target,
            no_redact: args.no_redact,
            all: args.all,
            open: args.open,
            debug_local_only: args.debug_local_only,
        };
        let cmd = ShareCommand::new(self.session_store.clone());
        Ok(cmd.execute(share_args).await)
    }

    async fn show_entry(&self, args: &LogsArgs) -> Result<ExitCode> {
        // No id passed → open the picker, same UX as `aivo logs share`.
        // Honors --all so users can scope to the current project or every
        // project. Clean cancel exits quietly.
        let picked;
        let id = match args.target.as_deref().filter(|s| !s.is_empty()) {
            Some(id) => id,
            None => {
                let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                match crate::services::share_picker::pick_session_id(
                    &self.session_store,
                    &cwd,
                    args.all,
                    "Show which entry?",
                    "aivo logs show <id>",
                )
                .await?
                {
                    Some(id) => {
                        picked = id;
                        picked.as_str()
                    }
                    None => return Ok(ExitCode::Success),
                }
            }
        };

        // logs.db first — don't auto-drill into chat sessions, that's
        // `aivo logs share`'s job; `show` is the row-level inspector.
        let logs_hits = self.session_store.logs().find_by_id_prefix(id, 5).await?;
        if logs_hits.len() > 1 {
            if let Some(entry) = pick_ambiguous_log_hit(id, &logs_hits, args.json).await? {
                if args.json {
                    print_entry_json(&entry, &self.session_store).await?;
                } else {
                    print_entry(&entry, &self.session_store).await;
                }
            }
            return Ok(ExitCode::Success);
        }
        if let Some(entry) = logs_hits.into_iter().next() {
            if args.json {
                print_entry_json(&entry, &self.session_store).await?;
            } else {
                print_entry(&entry, &self.session_store).await;
            }
            return Ok(ExitCode::Success);
        }

        // 2. Fall back to native CLI via the share resolver, which
        //    already does prefix-matching across all sources and surfaces
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
                "No log entry or native session matching '{}'.\n  ({})",
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
        let (rows, run_meta) = if args.json {
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
            println!(
                "{}",
                serde_json::to_string_pretty(&unified_to_json(&rows, &run_meta))?
            );
            return Ok(ExitCode::Success);
        }

        let orphan_chat_ids = compute_orphan_code_ids(&self.session_store).await;
        render_unified_rows(&rows, &orphan_chat_ids, &run_meta);
        Ok(ExitCode::Success)
    }

    async fn watch_entries(&self, args: &LogsArgs) -> Result<ExitCode> {
        const WATCH_INTERVAL: Duration = Duration::from_secs(1);
        let mut seen_ids: HashSet<String> = HashSet::new();

        loop {
            let (rows, run_meta) = self.fetch_unified_rows(args).await?;

            if args.jsonl {
                let mut ordered = rows.clone();
                ordered.reverse();
                for row in ordered {
                    let key = unified_id_key(&row);
                    if seen_ids.insert(key) {
                        println!(
                            "{}",
                            serde_json::to_string(&unified_row_to_json(&row, &run_meta))?
                        );
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
                let orphan_chat_ids = compute_orphan_code_ids(&self.session_store).await;
                render_unified_rows(&rows, &orphan_chat_ids, &run_meta);
                io::stdout().flush()?;
            }

            tokio::time::sleep(WATCH_INTERVAL).await;
        }
    }

    /// Pulls rows from all three data sources concurrently, then k-way-merges
    /// the already-newest-first streams to take only `--limit` rows. Thin
    /// wrapper around the free `fetch_unified_rows` so `aivo share`'s picker
    /// can reuse the same pipeline without going through `LogsCommand`.
    async fn fetch_unified_rows(&self, args: &LogsArgs) -> Result<(Vec<UnifiedRow>, RunMetaIndex)> {
        fetch_unified_rows(&self.session_store, args).await
    }

    pub fn print_help(action: Option<&str>) {
        match action {
            Some("show") => print_help_show(),
            Some("share") => print_help_share(),
            Some("prune") => print_help_prune(),
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
        style::dim(
            "Unified session list — aivo code + native CLI sessions (claude, codex, gemini, pi, opencode). Use --by run / --by serve for launch events."
        )
    );
    println!();
    println!("{}", style::bold("Commands:"));
    logs_help_row("list", "Recent rows, newest first (default)");
    logs_help_row("show [id]", "Show one row in detail (omit id → picker)");
    logs_help_row(
        "share [id]",
        "Share a session via viewer URL (omit id → picker)",
    );
    logs_help_row("prune", "Remove logs.db events whose session file is gone");
    println!();
    println!("{}", style::bold("Filters:"));
    logs_help_row("-n, --limit <N>", "Max rows after merge (default: 20)");
    logs_help_row("--by <name>", "Any source above, or a plugin name");
    logs_help_row("-s, --search <query>", "Search title/topic/body text");
    logs_help_row(
        "-a, --all",
        "Every project (else current cwd; --cwd <path>)",
    );
    logs_help_row("--since / --until <t>", "Bound results by time");
    logs_help_row(
        "--model / -k <v>",
        "Filter by model / saved key (logs.db only)",
    );
    logs_help_row("--errors", "Only HTTP >= 400 or non-zero exit");
    logs_help_row(
        "--json / --watch",
        "JSON output / live refresh (--jsonl to stream)",
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!(
        "  {}",
        style::dim("aivo logs                     # current project, newest first")
    );
    println!(
        "  {}",
        style::dim("aivo logs --by run            # launch events (tool, model, exit code)")
    );
    println!(
        "  {}",
        style::dim("aivo logs show 1335c631       # any unique id prefix works")
    );
}

fn print_help_show() {
    println!("{} aivo logs show [ID]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Show full detail for one row (logs.db id or native session id; any unique prefix). Omit the id to open the picker."
        )
    );
    println!();
    println!("{}", style::bold("Examples:"));
    println!(
        "  {}",
        style::dim("aivo logs show               # pick from this project")
    );
    println!("  {}", style::dim("aivo logs show 1335c631"));
}

fn print_help_share() {
    println!("{} aivo logs share [ID] [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Share a session via a tunneled, live viewer URL (secrets and $HOME redacted). Omit the id to open the picker; `aivo share` is an alias."
        )
    );
    println!();
    println!("{}", style::bold("Options:"));
    logs_help_row(
        "--no-redact",
        "Skip redaction of keys, tokens, $HOME, secrets",
    );
    logs_help_row("--all", "Pick from every project (default: current cwd)");
    logs_help_row("--open", "Open the share URL in the browser when ready");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo logs share"));
    println!("  {}", style::dim("aivo logs share 1335c631"));
}

fn print_help_prune() {
    println!("{} aivo logs prune [OPTIONS]", style::bold("Usage:"));
    println!();
    println!(
        "{}",
        style::dim(
            "Delete logs.db code events whose session file is gone (prompts unless --force). Native session files are not touched."
        )
    );
    println!();
    println!("{}", style::bold("Options:"));
    logs_help_row("-f, --force", "Skip the confirmation prompt and delete");
    println!();
    println!("{}", style::bold("Examples:"));
    println!("  {}", style::dim("aivo logs prune"));
    println!("  {}", style::dim("aivo logs prune --force"));
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
    let is_list = matches!(args.action.as_deref(), None | Some("list" | "ls"));
    if args.watch && !is_list {
        anyhow::bail!("--watch is only supported for `aivo logs` list output");
    }
    let is_share = args.action.as_deref() == Some("share");
    if !is_share && (args.no_redact || args.open || args.debug_local_only) {
        anyhow::bail!("--no-redact, --open, --debug-local-only only apply to `aivo logs share`");
    }
    Ok(())
}

fn render_unified_rows(
    rows: &[UnifiedRow],
    orphan_chat_ids: &HashSet<String>,
    run_meta: &RunMetaIndex,
) {
    if rows.is_empty() {
        println!("{}", style::dim("No entries found."));
        return;
    }
    let id_width = min_unique_id_width(rows);
    let detail_width = available_detail_width(id_width);
    for row in rows {
        let orphan = row.is_orphan_chat(orphan_chat_ids);
        match row {
            UnifiedRow::Log(e) => print_summary(e, id_width, detail_width, orphan),
            UnifiedRow::Native(t) => {
                print_native_summary(t, id_width, detail_width, run_meta.get(&t.session_id))
            }
        }
    }
}

/// Detail-column width that keeps each row on a single terminal line.
/// `prefix` covers age (5) + id (`id_width`) + bracket (10) + 3 separator
/// spaces; the trailing `+1` leaves headroom for the cursor so terminals
/// that auto-wrap on the *last* column don't push every row to two lines.
/// Clamped to a comfortable reading band so very wide terminals don't
/// produce unscannable 200-char rows.
fn available_detail_width(id_width: usize) -> usize {
    let prefix = 5 + 1 + id_width + 1 + BRACKET_COL_WIDTH + 1;
    let cols = console::Term::stdout().size().1 as usize;
    cols.saturating_sub(prefix + 1).clamp(30, 80)
}

/// Floor width for the id column — git-style short SHA.
/// `min_unique_id_width` widens this up to `ID_COL_WIDTH_MAX` when the
/// rows being rendered collide at 8.
const ID_COL_WIDTH: usize = 8;
/// Hard cap on the dynamic id column so the detail field stays readable;
/// residual collisions past this fall through to the picker.
const ID_COL_WIDTH_MAX: usize = 14;
/// Width of the source bracket column, padded for `[opencode]` (10 chars).
/// Keeps detail-column alignment consistent across all sources.
const BRACKET_COL_WIDTH: usize = 10;

fn print_native_summary(
    t: &Thread,
    id_width: usize,
    detail_width: usize,
    run_meta: Option<&RunMeta>,
) {
    let time_ago = format_time_ago_short_dt(t.updated_at);
    let id = compact_id(&t.session_id, id_width);
    let suffix = run_meta.map(format_run_meta_suffix).unwrap_or_default();
    let topic_budget = detail_width.saturating_sub(visible_width(&suffix));
    let topic = trim_to_one_line(&t.topic, topic_budget);
    println!(
        "{} {} {} {}{}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(format!("{:<width$}", id, width = id_width)),
        style::magenta(format!(
            "{:<width$}",
            format!("[{}]", t.cli),
            width = BRACKET_COL_WIDTH
        )),
        topic,
        suffix,
    );
}

/// Trailing tag rendered after a native row's topic when we have aivo-side
/// metadata for it: ` · <key> · exit <N>`. Empty when both fields are
/// missing — leaves the row identical to the un-enriched form.
fn format_run_meta_suffix(meta: &RunMeta) -> String {
    let plain = format_run_meta_suffix_plain(meta);
    if plain.is_empty() {
        String::new()
    } else {
        format!(" {}", style::dim(plain.trim_start()))
    }
}

/// ANSI-free version used by picker labels (FuzzySelect highlights
/// matched ranges itself; embedded escapes break that highlighting).
fn format_run_meta_suffix_plain(meta: &RunMeta) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(name) = meta.key_name.as_deref() {
        parts.push(name.to_string());
    }
    if let Some(code) = meta.exit_code {
        parts.push(format!("exit {code}"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!(" · {}", parts.join(" · "))
    }
}

/// Width of a string with ANSI escapes counted as zero-width — needed when
/// computing how much detail-column budget the suffix consumes.
fn visible_width(s: &str) -> usize {
    let mut out = 0;
    let mut in_esc = false;
    for c in s.chars() {
        if in_esc {
            if c == 'm' {
                in_esc = false;
            }
            continue;
        }
        if c == '\x1b' {
            in_esc = true;
            continue;
        }
        out += 1;
    }
    out
}

/// "5m" / "2d" — for `Thread`, which has already-parsed timestamps.
fn format_time_ago_short_dt(ts: DateTime<Utc>) -> String {
    format_time_ago_short(&ts.to_rfc3339())
}

/// Collapse every kind of line/whitespace separator into a single space, then
/// truncate to `max_cols` terminal columns with an ellipsis. Width-aware:
/// CJK and other East Asian Wide chars count as 2 columns so picker rows
/// containing them don't overflow the terminal and wrap to a second row
/// (which would defeat `clear_last_lines` in the FuzzySelect redraw loop).
pub(crate) fn trim_to_one_line(text: &str, max_cols: usize) -> String {
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
    let total_cols: usize = one_line
        .chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum();
    if total_cols <= max_cols {
        return one_line.to_string();
    }
    let budget = max_cols.saturating_sub(1); // reserve one column for the ellipsis
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

fn print_summary(entry: &LogEntry, id_width: usize, detail_width: usize, is_orphan: bool) {
    let display_id = display_id(entry);
    let time_ago = format_time_ago_short(&entry.ts_utc);
    // (text, dim_suffix). Trimming runs on the plain text *before* styling
    // so `is_control()`-based whitespace collapse can't strip ANSI escape
    // bytes out of the suffix and leave bare `[2m…[0m` literals on screen.
    let (text, dim_suffix): (String, String) = match entry.source.as_str() {
        "chat" | "code" => (
            entry.title.clone().unwrap_or_else(|| "(code)".to_string()),
            format_token_summary(entry),
        ),
        "run" => {
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
            // Plugin runs put the tool name in the bracket column; native
            // `--by run` rows keep it inline since their bracket is `[run]`.
            let text = if is_plugin_run(entry) {
                format!("{model} {state}{duration}")
            } else {
                let tool = entry.tool.as_deref().unwrap_or("run");
                format!("{tool} {model} {state}{duration}")
            };
            (text, String::new())
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
    let detail = if is_orphan {
        format!("{detail} {}", style::dim("(file deleted)"))
    } else {
        detail
    };
    // Same column shape as native rows: age (5) · id (`id_width`) · bracket (10) · detail.
    // `{:<W.W}` truncates a too-long id to W chars then pads it to W — gives
    // a clean column even when logs.db's full 12-char id is longer than W.
    println!(
        "{} {} {} {}",
        style::dim(format!("{:>5}", time_ago)),
        style::cyan(format!("{:<width$.width$}", display_id, width = id_width)),
        style::yellow(format!(
            "{:<width$}",
            format!("[{}]", log_bracket_label(entry)),
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

/// Picker over multiple `find_by_id_prefix` hits. Bails with the legacy
/// listing when JSON mode or non-TTY makes the picker unusable.
async fn pick_ambiguous_log_hit(
    prefix: &str,
    hits: &[LogEntry],
    json_mode: bool,
) -> Result<Option<LogEntry>> {
    use std::io::IsTerminal;
    let interactive = !json_mode && io::stdout().is_terminal() && io::stdin().is_terminal();
    if !interactive {
        let summary = hits
            .iter()
            .map(|e| format!("{} [{}]", &e.id, e.source))
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "ambiguous logs.db prefix '{}' — matched: {}. Re-run with a longer prefix.",
            prefix,
            summary
        );
    }

    let rows: Vec<UnifiedRow> = hits
        .iter()
        .map(|e| UnifiedRow::Log(Box::new(e.clone())))
        .collect();
    let id_width = min_unique_id_width(&rows);
    let detail_width = picker_detail_width(console::Term::stdout().size().1 as usize, id_width);
    let orphan_chat_ids: HashSet<String> = HashSet::new();
    let run_meta: RunMetaIndex = RunMetaIndex::new();
    let labels: Vec<String> = rows
        .iter()
        .map(|r| r.picker_label(id_width, detail_width, &orphan_chat_ids, &run_meta))
        .collect();
    let owned: Vec<LogEntry> = hits.to_vec();
    let prompt = format!("Multiple matches for '{prefix}' — pick one");
    tokio::task::spawn_blocking(move || -> std::io::Result<Option<LogEntry>> {
        let selected = crate::tui::FuzzySelect::new()
            .with_prompt(&prompt)
            .items(&labels)
            .default(0)
            .interact_opt()?;
        Ok(selected.map(|idx| owned[idx].clone()))
    })
    .await
    .map_err(|e| anyhow::anyhow!("picker thread panicked: {e}"))?
    .map_err(|e| anyhow::anyhow!("picker I/O failed: {e}"))
}

fn display_id(entry: &LogEntry) -> &str {
    if entry.source == "run"
        && let Some(group_id) = entry.event_group_id.as_deref()
    {
        return group_id;
    }
    // Chat events: prefer the chat session UUID so a row's id matches what
    // `aivo share`'s resolver expects and what the chat picker / session
    // files use. Falls back to the logs.db row id for legacy events that
    // pre-date the `session_id` linkage column.
    if is_code_source(&entry.source)
        && let Some(sid) = entry.session_id.as_deref()
    {
        return sid;
    }
    &entry.id
}

/// Emit the row's JSON, enriched with the resolved transcript for `run`
/// rows that reference a native session. The row alone is just launch
/// metadata (model, key, cwd, exit code) — the actual conversation
/// lives in the native CLI's session file. Without this, `aivo logs
/// show T-… --json` returned a row with null tokens and no messages.
async fn print_entry_json(entry: &LogEntry, store: &SessionStore) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let ctx = crate::services::share_resolver::ResolverContext::from_system(cwd, store.clone());
    let value = build_entry_json(entry, &ctx).await?;
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

async fn build_entry_json(
    entry: &LogEntry,
    ctx: &crate::services::share_resolver::ResolverContext,
) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(entry)?;
    if entry.source == "run"
        && let Some(session_id) = entry.session_id.as_deref().filter(|s| !s.is_empty())
        && let Some(obj) = value.as_object_mut()
    {
        // Resolver errors are non-fatal — the row metadata is still
        // useful by itself (e.g. for runs whose native session was
        // deleted). Just omit the `session` field in that case.
        if let Ok(resolved) =
            crate::services::share_resolver::resolve_session(session_id, ctx).await
        {
            obj.insert("session".into(), serde_json::to_value(&resolved.payload)?);
        }
    }
    Ok(value)
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
    if !is_code_source(&entry.source) {
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
    let mut entries = store
        .logs()
        .list(LogQuery {
            limit: query_limit,
            search: args.search.clone(),
            by: plan.logs_by(),
            model: args.model.clone(),
            key_query: args.key.clone(),
            cwd: cwd_filter.clone(),
            since: normalize_time_filter(args.since.as_deref()),
            until: normalize_time_filter(args.until.as_deref()),
            errors_only: args.errors,
        })
        .await?;
    // Plugin tools (e.g. amp) aren't a native AIToolType and have no native
    // session source, so the unified view never sees them. Pull their `[run]`
    // rows in — non-native tools only, so native tools aren't duplicated — and
    // re-sort newest-first so the downstream collapse/merge invariants hold.
    if plan.plugin_runs {
        let runs = store
            .logs()
            .list(LogQuery {
                limit: query_limit,
                search: args.search.clone(),
                by: Some("run".to_string()),
                model: args.model.clone(),
                key_query: args.key.clone(),
                cwd: cwd_filter.clone(),
                since: normalize_time_filter(args.since.as_deref()),
                until: normalize_time_filter(args.until.as_deref()),
                errors_only: args.errors,
            })
            .await?;
        entries.extend(
            runs.into_iter()
                .filter(|e| e.tool.as_deref().is_some_and(|t| !KNOWN_TOOLS.contains(&t))),
        );
        // ts_utc is always a UTC rfc3339 stamp written by aivo, so lexical
        // descending order is chronological newest-first.
        entries.sort_by(|a, b| b.ts_utc.cmp(&a.ts_utc));
    }
    // Run events are emitted as start+finish pairs sharing an event_group_id;
    // collapse here too so the unified listing doesn't show both halves.
    // Then collapse chat events by session_id so the session list shows one
    // row per chat conversation instead of one per turn.
    let entries = collapse_run_events(entries, args.limit.saturating_mul(3));
    let mut entries = collapse_chat_sessions(entries);

    // Sessions whose every turn was cancelled/interrupted never log a `chat_turn`
    // row; pull them off disk so the listing matches `aivo code --resume`.
    if plan.includes_code() && !args.errors {
        let logged: HashSet<String> = entries
            .iter()
            .filter(|e| is_code_source(&e.source))
            .filter_map(|e| e.session_id.clone())
            .collect();
        let extra = fetch_unlogged_chat_rows(store, args, cwd_filter.as_deref(), &logged).await;
        if !extra.is_empty() {
            entries.extend(extra);
            entries.sort_by(|a, b| b.ts_utc.cmp(&a.ts_utc));
            entries.truncate(query_limit);
        }
    }
    Ok(entries)
}

/// `[chat]` rows for on-disk sessions absent from `logged_ids`, filtered to
/// match logged-row semantics (cwd/model/key/search/since-until).
async fn fetch_unlogged_chat_rows(
    store: &SessionStore,
    args: &LogsArgs,
    cwd_filter: Option<&str>,
    logged_ids: &HashSet<String>,
) -> Vec<LogEntry> {
    let entries = match store.all_chat_sessions().await {
        Ok(entries) => entries,
        Err(_) => return Vec::new(),
    };
    // key_id → display name for `--key <name>`/search parity with logged rows.
    let key_names: HashMap<String, String> = store
        .get_keys()
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|k| (k.id.clone(), k.display_name().to_string()))
        .collect();
    entries
        .into_iter()
        .filter(|e| !logged_ids.contains(&e.session_id))
        .filter(|e| {
            chat_index_passes_filters(
                e,
                args,
                cwd_filter,
                key_names.get(&e.key_id).map(String::as_str),
            )
        })
        .map(|e| {
            let key_name = key_names.get(&e.key_id).cloned();
            synthesize_chat_log_entry(e, key_name)
        })
        .collect()
}

/// Project a `SessionIndexEntry` onto a `source="code"` `LogEntry` so it renders
/// and merges like a logged code turn (id = session UUID; cumulative tokens).
fn synthesize_chat_log_entry(e: SessionIndexEntry, key_name: Option<String>) -> LogEntry {
    LogEntry {
        id: e.session_id.clone(),
        ts_utc: e.updated_at,
        source: "code".to_string(),
        kind: "code_session".to_string(),
        key_id: Some(e.key_id),
        key_name,
        base_url: Some(e.base_url),
        tool: Some("code".to_string()),
        model: Some(e.model),
        cwd: Some(e.cwd),
        session_id: Some(e.session_id),
        input_tokens: Some(e.prompt_tokens as i64),
        output_tokens: Some(e.completion_tokens as i64),
        cache_read_input_tokens: Some(e.cache_read_tokens as i64),
        cache_creation_input_tokens: Some(e.cache_write_tokens as i64),
        title: Some(e.title),
        body_text: Some(e.preview),
        ..Default::default()
    }
}

/// Rust mirror of the logs.db chat-row filters; `cwd_filter` is pre-canonicalized.
fn chat_index_passes_filters(
    e: &SessionIndexEntry,
    args: &LogsArgs,
    cwd_filter: Option<&str>,
    key_name: Option<&str>,
) -> bool {
    if let Some(needle) = cwd_filter
        && !cwd_is_under(&canonicalize_for_match(&e.cwd), needle)
    {
        return false;
    }
    if let Some(model) = args.model.as_deref() {
        let needle = model.to_ascii_lowercase();
        let hay = format!(
            "{} {}",
            e.model.to_ascii_lowercase(),
            e.billed_model
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase()
        );
        if !hay.contains(&needle) {
            return false;
        }
    }
    if let Some(key) = args.key.as_deref() {
        let needle = key.to_ascii_lowercase();
        let hay = format!(
            "{} {}",
            e.key_id.to_ascii_lowercase(),
            key_name.unwrap_or_default().to_ascii_lowercase()
        );
        if !hay.contains(&needle) {
            return false;
        }
    }
    if let Some(search) = args.search.as_deref() {
        let needle = search.to_ascii_lowercase();
        let hay = format!(
            "{} {} {} {} {} {}",
            e.title.to_ascii_lowercase(),
            e.preview.to_ascii_lowercase(),
            e.model.to_ascii_lowercase(),
            e.key_id.to_ascii_lowercase(),
            key_name.unwrap_or_default().to_ascii_lowercase(),
            e.cwd.to_ascii_lowercase(),
        );
        if !hay.contains(&needle) {
            return false;
        }
    }
    if let Ok(updated) = DateTime::parse_from_rfc3339(&e.updated_at) {
        let updated = updated.with_timezone(&Utc);
        if let Some(since) = args.since.as_deref().and_then(parse_loose_time)
            && updated < since
        {
            return false;
        }
        if let Some(until) = args.until.as_deref().and_then(parse_loose_time)
            && updated > until
        {
            return false;
        }
    }
    true
}

/// Collapse same-session chat events down to one row per `session_id`,
/// keeping the most recent event in each group (entries arrive already
/// sorted newest-first, so first-seen wins). Old chat rows with
/// `session_id: None` (written before the linkage field existed) pass
/// through unchanged — without a key we can't safely group them.
fn collapse_chat_sessions(entries: Vec<LogEntry>) -> Vec<LogEntry> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if is_code_source(&entry.source)
            && let Some(sid) = entry.session_id.as_deref()
            && !seen.insert(sid.to_string())
        {
            continue;
        }
        out.push(entry);
    }
    out
}

// Perf: we always walk globally and post-filter by cwd — project-scoped
// walks are slower on a multi-project machine because codex/gemini hit the
// same global tree either way but reject every non-matching file (forcing
// the walk to continue past the cap). Global stops at cap quickly.
//
// The wrinkle: when a cwd filter IS in effect, capping at limit*2 = 50
// extracts the globally-newest 50 sessions per CLI and *then* drops the
// non-cwd ones. On a multi-project machine that often leaves zero matches
// for the user's actual project even though plenty exist within the 14d
// window. So we lift the cap when cwd is filtered — the 14d age cap still
// bounds the worst case, and cwd usually narrows results to a handful.
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
        max_per_source: if cwd_filter.is_some() {
            None
        } else {
            Some(args.limit.saturating_mul(2).max(50))
        },
        // Listing view: keep `hi`/`ok` sessions visible. `aivo run`'s
        // context picker uses a separate IngestOptions with this off.
        include_short_first_user: true,
    };
    // Only search matching and `--json` read `last_response`; when neither
    // does, the ingester head-parses each session file instead of the full
    // (often multi-MB) transcript.
    let need_last_response = args.search.is_some() || args.json;
    let mut all = context_ingest::ingest_native_sessions_global(opts, need_last_response).await?;
    all.retain(|t| native_passes_filters(t, args, cwd_filter.as_deref()));
    Ok(all)
}

/// Compute the set of chat `session_id`s referenced by logs.db that no
/// longer have a session file on disk. Used to tag stale rows in the
/// listing and to refuse `aivo logs share` with a useful pointer. Returns
/// an empty set on any error — orphan tagging is decoration, not safety.
pub(crate) async fn compute_orphan_code_ids(store: &SessionStore) -> HashSet<String> {
    let log_ids = match store.logs().distinct_code_session_ids().await {
        Ok(ids) => ids,
        Err(_) => return HashSet::new(),
    };
    if log_ids.is_empty() {
        return HashSet::new();
    }
    let disk_ids = store.code_session_ids_on_disk().await;
    log_ids
        .into_iter()
        .filter(|id| !disk_ids.contains(id))
        .collect()
}

/// Lookup keyed on a native session_id, supplying the aivo-side metadata
/// (key name, exit code) recorded by the `[run]` event that produced the
/// session. Empty for callers that don't fetch enrichment.
pub(crate) type RunMetaIndex = HashMap<String, RunMeta>;

/// Free-function counterpart of `LogsCommand::fetch_unified_rows`. Shared
/// with the `aivo share` picker so both surfaces draw from the same merged
/// stream (same ids, same ordering, same filters). Returns the merged rows
/// plus a side-table of `[run]` metadata keyed by the native session_id
/// each launch produced — renderers join on it to show the key/exit on
/// the native row instead of as a separate `[run]` line.
pub(crate) async fn fetch_unified_rows(
    store: &SessionStore,
    args: &LogsArgs,
) -> Result<(Vec<UnifiedRow>, RunMetaIndex)> {
    let plan = SourcePlan::from_args(args);
    let cwd_filter: Option<String> = if args.all {
        None
    } else if let Some(explicit) = args.cwd.as_deref() {
        Some(expand_cwd_filter(explicit))
    } else {
        system_env::current_dir().map(|p| p.to_string_lossy().to_string())
    };
    // Canonicalize once so symlinks/`.` resolve and downstream matchers can
    // do a straight prefix check without each one calling `canonicalize`.
    let cwd_filter = cwd_filter.map(|s| canonicalize_for_match(&s));

    let (log_rows, native_rows) = tokio::try_join!(
        fetch_logs_rows(store, args, &plan, cwd_filter.clone()),
        fetch_native_rows(args, &plan, cwd_filter.clone()),
    )?;

    let rows = merge_unified(log_rows, native_rows, args.limit);
    let run_meta = run_meta_for_native_rows(store, &rows).await;
    Ok((rows, run_meta))
}

/// Pull the `[run]` finished-event metadata for every native session in
/// the merged view in one query, so the renderer can annotate native rows
/// without per-row round trips. Errors are swallowed (the enrichment is
/// decoration, not safety).
async fn run_meta_for_native_rows(store: &SessionStore, rows: &[UnifiedRow]) -> RunMetaIndex {
    let session_ids: Vec<String> = rows
        .iter()
        .filter_map(|r| match r {
            UnifiedRow::Native(t) => Some(t.session_id.clone()),
            _ => None,
        })
        .collect();
    if session_ids.is_empty() {
        return HashMap::new();
    }
    store
        .logs()
        .run_meta_for_sessions(&session_ids)
        .await
        .unwrap_or_default()
}

/// Three-way merge of newest-first streams. Pops the source with the newest
/// head until `limit` rows are emitted — never materializes the full union.
fn merge_unified(logs: Vec<LogEntry>, native: Vec<Thread>, limit: usize) -> Vec<UnifiedRow> {
    let mut logs = logs.into_iter().peekable();
    let mut native = native.into_iter().peekable();
    let mut out: Vec<UnifiedRow> = Vec::with_capacity(limit);

    while out.len() < limit {
        let heads = [
            logs.peek().map(|e| parse_log_ts(&e.ts_utc)),
            native.peek().map(|t| t.updated_at),
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
    /// Whether to query logs.db at all.
    logs: bool,
    /// Effective `LogQuery.by` when `logs` is true. `None` = no SQL-level
    /// source filter; `Some(s)` = restrict to that source/tool. Default
    /// is `Some("code")` so the unified view shows the built-in agent's
    /// *sessions* (legacy `chat`-sourced rows included) but not run/serve
    /// events (those have their own views via explicit `--by run` / `--by serve`).
    logs_by: Option<String>,
    native: bool,
    /// Include `[run]` rows for plugin (non-native) tools in the unified view.
    /// Native tools surface as native session rows, so their run events stay
    /// excluded; a plugin tool (e.g. amp) has no native source, so its run
    /// events are its representation in `aivo logs`.
    plugin_runs: bool,
}

impl SourcePlan {
    fn from_args(args: &LogsArgs) -> Self {
        // Filters that only make sense for logs.db rows force a strict mode:
        // drop native, and open logs.db up to every source so audits
        // see run/serve errors alongside chat ones.
        let strict_logs = args.errors || args.key.is_some() || args.model.is_some();

        let by = args.by.as_deref().map(str::to_ascii_lowercase);
        let by = by.as_deref();

        let logs;
        let logs_by;
        let native;
        let plugin_runs;
        match by {
            // The built-in agent. `chat` is the pre-rename alias; normalize it
            // to `code` so the query shim (`source in ('chat','code')`) picks up
            // both old and new rows either way.
            Some("code") | Some("chat") => {
                logs = true;
                logs_by = Some("code".to_string());
                native = false;
                plugin_runs = false;
            }
            // Explicit logs.db source.
            Some(name @ ("run" | "serve")) => {
                logs = true;
                logs_by = Some(name.to_string());
                native = false;
                plugin_runs = false;
            }
            Some("native") => {
                logs = false;
                logs_by = None;
                native = true;
                plugin_runs = false;
            }
            // CLI names: native sessions of that cli only. `[run]` rows are
            // aivo's launch record for the same session — including them
            // would just duplicate the native row. Users who want the run-
            // event view ask for `--by run` explicitly. Strict-logs callers
            // (errors/key/model) bypass this and get the logs.db rows for
            // that tool.
            Some("claude") | Some("codex") | Some("gemini") | Some("opencode") | Some("pi") => {
                logs = strict_logs;
                logs_by = if strict_logs { args.by.clone() } else { None };
                native = !strict_logs;
                plugin_runs = false;
            }
            // Any other name is a plugin coding-agent (e.g. omp, amp). Plugins
            // have no native session source, so their `[run]` rows (tool=<name>)
            // are their only representation. Scope logs.db to that tool — the
            // `by` filter matches `source = ? or tool like %name%`, so the run
            // rows come through — and drop both native and the broad
            // `plugin_runs` pull (which would re-admit every *other* plugin).
            // The strict-logs filters (errors/key/model) still apply as
            // separate LogQuery fields, so `--by omp --errors` stays scoped.
            Some(plugin) => {
                logs = true;
                logs_by = Some(plugin.to_string());
                native = false;
                plugin_runs = false;
            }
            // No --by: the unified session list. logs.db restricted to chat
            // (collapsed by session_id downstream), native joins in,
            // run/serve stay out of the default view. Strict-logs reopens
            // logs.db to every source for auditing. Plugin tools have no
            // native source, so their `[run]` rows are pulled in (see
            // `plugin_runs`).
            None => {
                logs = true;
                logs_by = if strict_logs {
                    None
                } else {
                    Some("code".to_string())
                };
                native = !strict_logs;
                plugin_runs = !strict_logs;
            }
        }

        Self {
            logs,
            logs_by,
            native,
            plugin_runs,
        }
    }

    fn include_logs(&self) -> bool {
        self.logs
    }
    fn logs_by(&self) -> Option<String> {
        self.logs_by.clone()
    }
    fn include_native(&self) -> bool {
        self.native
    }
    /// Code-agent rows in scope: the default view (`Some("code")`) or a
    /// strict-logs audit opened to every source (`None`). Run/serve/cli/plugin
    /// plans exclude them. Gates whether on-disk-only sessions get synthesized in.
    fn includes_code(&self) -> bool {
        self.logs && matches!(self.logs_by.as_deref(), None | Some("code"))
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
            "claude" | "codex" | "gemini" | "opencode" | "pi" => {
                if t.cli != by {
                    return false;
                }
            }
            // Unknown name = a plugin coding-agent, which has no native
            // session. Reject every native thread so `--by <plugin>` can't
            // surface native sessions (SourcePlan already gates native off
            // for plugin names; this keeps the filter self-consistent).
            _ => return false,
        }
    }
    if let Some(needle) = cwd_filter {
        // Canonicalize the thread's cwd too — sessions recorded a non-
        // canonical path (e.g. `/tmp/hi` on macOS where the real path is
        // `/private/tmp/hi`) still need to match. The filter side is
        // already canonicalized by `fetch_unified_rows`.
        let thread_cwd = t.cwd.as_deref().map(canonicalize_for_match);
        let cwd_match = thread_cwd
            .as_deref()
            .map(|c| cwd_is_under(c, needle))
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

/// True iff `child` is `parent` or one of its descendants. Both inputs are
/// normalized via [`normalize_cwd_for_match`] so backslashes vs forward
/// slashes and case differences (Windows) don't cause spurious mismatches.
/// The strip-then-check-separator shape resists prefix-collision false
/// positives (`/foo/bar` vs `/foo/bar-other`, `/foo/bar` vs
/// `/elsewhere/foo/bar/x`) that a plain `String::contains` would accept.
pub(crate) fn cwd_is_under(child: &str, parent: &str) -> bool {
    let c = normalize_cwd_for_match(child);
    let p = normalize_cwd_for_match(parent);
    if c == p {
        return true;
    }
    // After normalization both sides use '/'.
    c.strip_prefix(&p).is_some_and(|rest| rest.starts_with('/'))
}

/// Normalize a cwd string for matching:
/// - trim trailing path separators (`/` or `\`)
/// - collapse `\` to `/` so Windows paths (`C:\Foo\Bar`) compare with paths
///   recorded forward-slashed elsewhere
/// - lowercase on Windows, where the filesystem is case-insensitive
fn normalize_cwd_for_match(s: &str) -> String {
    let trimmed = s.trim_end_matches(['/', '\\']);
    let unified: String = trimmed
        .chars()
        .map(|c| if c == '\\' { '/' } else { c })
        .collect();
    if cfg!(windows) {
        unified.to_ascii_lowercase()
    } else {
        unified
    }
}

/// Best-effort path canonicalization for cwd matching: resolves symlinks
/// and `..`/`.` when the path exists on disk, otherwise returns the input
/// with its trailing separator trimmed. On Windows, strips the `\\?\`
/// verbatim-namespace prefix that `std::fs::canonicalize` adds — keeping
/// it would create artificial mismatches against stored cwds (which don't
/// carry the prefix). Deleted-cwd sessions still need to be listable, so
/// canonicalize failures fall back to the trimmed input.
pub(crate) fn canonicalize_for_match(input: &str) -> String {
    let trimmed = input.trim_end_matches(['/', '\\']);
    let canonical = std::fs::canonicalize(trimmed)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| trimmed.to_string());
    #[cfg(windows)]
    {
        canonical
            .strip_prefix("\\\\?\\")
            .map(str::to_string)
            .unwrap_or(canonical)
    }
    #[cfg(not(windows))]
    {
        canonical
    }
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

fn unified_to_json(rows: &[UnifiedRow], run_meta: &RunMetaIndex) -> Vec<Value> {
    rows.iter()
        .map(|r| unified_row_to_json(r, run_meta))
        .collect()
}

fn unified_row_to_json(row: &UnifiedRow, run_meta: &RunMetaIndex) -> Value {
    match row {
        UnifiedRow::Log(e) => {
            let mut v = serde_json::to_value(e).unwrap_or(Value::Null);
            if let Some(map) = v.as_object_mut() {
                map.insert("kind".to_string(), Value::String("log_entry".into()));
            }
            v
        }
        UnifiedRow::Native(t) => {
            let mut v = json!({
                "kind": "native_session",
                "cli": t.cli,
                "session_id": t.session_id,
                "source_path": t.source_path,
                "topic": t.topic,
                "last_response": t.last_response,
                "updated_at": t.updated_at.to_rfc3339(),
                "cwd": t.cwd,
            });
            if let Some(meta) = run_meta.get(&t.session_id) {
                let map = v.as_object_mut().unwrap();
                map.insert(
                    "run".to_string(),
                    json!({
                        "key_name": meta.key_name,
                        "exit_code": meta.exit_code,
                    }),
                );
            }
            v
        }
    }
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
    }
}

/// Pretty-print a `SharePayload` returned by `share_resolver` for
/// `aivo logs show <native-id>`. Mirrors `print_entry`'s style.
fn print_share_payload(p: &crate::services::share_payload::SharePayload) {
    use crate::services::share_payload::ContentBlock;

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

    // Aggregate content size — useful for "is this conversation worth
    // sharing?" without spinning up the local server.
    let chars: usize = p.approximate_chars();
    if chars > 0 {
        println!(
            "{} ~{} KB",
            style::bold("size:"),
            (chars / 1024).max(if chars > 0 { 1 } else { 0 }),
        );
    }

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

    // First user turn ─ the "what was this conversation about?" line.
    // Skip turns that are just CLI-harness wrappers (codex injects an
    // `<environment_context>` block, claude can inject `<command-message>`,
    // etc.); we want the user's actual prompt.
    if let Some(text) = p
        .messages
        .iter()
        .filter(|m| m.role == "user")
        .filter_map(|m| first_text(m.content.as_slice()))
        .find(|t| !looks_like_cli_boilerplate(t))
    {
        println!();
        println!("{}", style::bold("First user:"));
        print_preview_block(text);
    }

    // Last assistant turn ─ "where did the conversation end up?". Skip if
    // the only assistant turns are tool-calls (no text content).
    if let Some(last_assistant) = p
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "assistant" && first_text(m.content.as_slice()).is_some())
        && let Some(text) = first_text(last_assistant.content.as_slice())
    {
        println!();
        println!("{}", style::bold("Last assistant:"));
        print_preview_block(text);
    }

    // Surface tool activity at a glance: how many tool calls were made and
    // what tools were used. Doesn't dump payloads — `aivo logs share` is for
    // the full view.
    let tool_calls: Vec<&str> = p
        .messages
        .iter()
        .flat_map(|m| m.content.iter())
        .filter_map(|c| match c {
            ContentBlock::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    if !tool_calls.is_empty() {
        let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for name in &tool_calls {
            *counts.entry(name).or_insert(0) += 1;
        }
        let summary = counts
            .iter()
            .map(|(name, n)| format!("{name}×{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!();
        println!("{} {}", style::bold("tools:"), style::dim(summary));
    }

    println!();
    println!(
        "{}",
        style::dim("(use `aivo logs share <id>` to open a full viewer)")
    );
}

/// First text-like content block from a message, if any. Skips tool
/// calls/results so the preview shows the human-readable side.
fn first_text(blocks: &[crate::services::share_payload::ContentBlock]) -> Option<&str> {
    use crate::services::share_payload::ContentBlock;
    blocks.iter().find_map(|b| match b {
        ContentBlock::Text { text } | ContentBlock::Code { text, .. } => Some(text.as_str()),
        _ => None,
    })
}

/// Heuristic: does `text` look like a CLI-harness injection rather than
/// something the user actually typed? Matches the same markers
/// `pick_first_user_turn` filters during ingest.
fn looks_like_cli_boilerplate(text: &str) -> bool {
    const MARKERS: &[&str] = &[
        "<environment_context>",
        "<command-message>",
        "<command-name>",
        "<local-command-caveat>",
        "<local-command-stdout>",
        "<local-command-stderr>",
        "<system-reminder>",
        "<user_instructions>",
        "<developer_instructions>",
    ];
    let lower = text.trim_start().to_lowercase();
    MARKERS.iter().any(|m| lower.starts_with(m))
}

/// Render a preview block under a "First user:" / "Last assistant:" label.
/// Each line is prefixed with `│ ` so embedded `key: value` content (or our
/// own metadata fields quoted back in a transcript) reads as part of the
/// preview, not as a continuation of the row's metadata block above.
fn print_preview_block(text: &str) {
    let clipped = preview_text(text, 6, 600);
    for line in clipped.split('\n') {
        println!("{} {}", style::dim("│"), line);
    }
}

/// Clip `text` to at most `max_lines` lines and `max_chars`, appending `…`
/// if anything was dropped. Keeps the show output bounded on huge turns.
fn preview_text(text: &str, max_lines: usize, max_chars: usize) -> String {
    let mut out = String::new();
    for (lines_used, line) in text.lines().enumerate() {
        if lines_used >= max_lines {
            out.push('…');
            return out;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        let remaining = max_chars.saturating_sub(out.chars().count());
        if line.chars().count() > remaining {
            out.extend(line.chars().take(remaining.saturating_sub(1)));
            out.push('…');
            return out;
        }
        out.push_str(line);
        if out.chars().count() >= max_chars {
            out.push('…');
            return out;
        }
    }
    out
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
            no_redact: false,
            open: false,
            debug_local_only: false,
            force: false,
        }
    }

    fn chat_entry(session_id: &str, cwd: &str, model: &str, key_id: &str) -> SessionIndexEntry {
        SessionIndexEntry {
            session_id: session_id.to_string(),
            key_id: key_id.to_string(),
            base_url: "https://api.example.com".to_string(),
            cwd: cwd.to_string(),
            model: model.to_string(),
            billed_model: None,
            updated_at: "2026-06-25T07:36:49+00:00".to_string(),
            created_at: "2026-06-25T07:36:06+00:00".to_string(),
            title: "investigate cancel bug".to_string(),
            preview: "Let me examine the agent engine".to_string(),
            prompt_tokens: 0,
            completion_tokens: 3057,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }

    #[test]
    fn synthesized_chat_row_renders_like_a_logged_one() {
        let entry = chat_entry("a3476c91-uuid", "/repo", "fugu", "cpg");
        let row = synthesize_chat_log_entry(entry, Some("sakana".to_string()));
        assert_eq!(row.source, "code");
        // display_id keys on the session UUID, not the row id.
        assert_eq!(display_id(&row), "a3476c91-uuid");
        assert_eq!(log_bracket_label(&row), "code");
        assert_eq!(format_token_summary(&row), "(0\u{2192}3057 tokens)");
        assert!(log_row_detail(&row).starts_with("investigate cancel bug"));
    }

    #[test]
    fn chat_index_filters_match_logged_chat_semantics() {
        let entry = chat_entry("sid", "/repo/aivo", "fugu", "cpg");
        let key_name = Some("sakana");

        assert!(chat_index_passes_filters(
            &entry,
            &base_args(),
            None,
            key_name
        ));

        // cwd scoping: parent matches, sibling-prefix does not.
        assert!(chat_index_passes_filters(
            &entry,
            &base_args(),
            Some("/repo/aivo"),
            key_name
        ));
        assert!(!chat_index_passes_filters(
            &entry,
            &base_args(),
            Some("/repo/other"),
            key_name
        ));

        let mut args = base_args();
        args.model = Some("FUG".to_string());
        assert!(chat_index_passes_filters(&entry, &args, None, key_name));
        args.model = Some("gpt".to_string());
        assert!(!chat_index_passes_filters(&entry, &args, None, key_name));

        let mut args = base_args();
        args.key = Some("sakana".to_string()); // by display name, not id
        assert!(chat_index_passes_filters(&entry, &args, None, key_name));

        let mut args = base_args();
        args.search = Some("cancel".to_string());
        assert!(chat_index_passes_filters(&entry, &args, None, key_name));
        args.search = Some("nomatch".to_string());
        assert!(!chat_index_passes_filters(&entry, &args, None, key_name));

        // updated_at is 2026-06-25.
        let mut args = base_args();
        args.since = Some("2026-06-24".to_string());
        assert!(chat_index_passes_filters(&entry, &args, None, key_name));
        args.since = Some("2026-06-26".to_string());
        assert!(!chat_index_passes_filters(&entry, &args, None, key_name));
    }

    #[test]
    fn source_plan_includes_chat_only_for_chat_scoped_plans() {
        assert!(SourcePlan::from_args(&base_args()).includes_code());
        assert!(SourcePlan::from_args(&logs_args_by(Some("chat"))).includes_code());
        // Strict-logs (audit) reopens logs.db to every source, chat included.
        let mut args = base_args();
        args.model = Some("fugu".to_string());
        assert!(SourcePlan::from_args(&args).includes_code());
        assert!(!SourcePlan::from_args(&logs_args_by(Some("run"))).includes_code());
        assert!(!SourcePlan::from_args(&logs_args_by(Some("serve"))).includes_code());
        assert!(!SourcePlan::from_args(&logs_args_by(Some("claude"))).includes_code());
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

    #[test]
    fn cwd_is_under_handles_prefix_collisions() {
        assert!(cwd_is_under("/foo/bar", "/foo/bar"));
        assert!(cwd_is_under("/foo/bar/sub", "/foo/bar"));
        assert!(cwd_is_under("/foo/bar/sub/deep", "/foo/bar"));
        assert!(cwd_is_under("/foo/bar/", "/foo/bar"));
        assert!(cwd_is_under("/foo/bar", "/foo/bar/"));
        // Sibling/mid-string prefixes must NOT match (regression vs the old
        // `String::contains` matcher).
        assert!(!cwd_is_under("/foo/bar-other", "/foo/bar"));
        assert!(!cwd_is_under("/foo/barother", "/foo/bar"));
        assert!(!cwd_is_under("/elsewhere/foo/bar/x", "/foo/bar"));
        assert!(!cwd_is_under("/other", "/foo/bar"));
    }

    #[test]
    fn cwd_is_under_handles_windows_paths() {
        assert!(cwd_is_under(
            r"C:\Users\alice\project",
            r"C:\Users\alice\project"
        ));
        assert!(cwd_is_under(
            r"C:\Users\alice\project\src",
            r"C:\Users\alice\project"
        ));
        // Mixed separators: real Windows runs produce both.
        assert!(cwd_is_under(
            r"C:\Users\alice\project\src",
            "C:/Users/alice/project"
        ));
        assert!(cwd_is_under(
            "C:/Users/alice/project/src",
            r"C:\Users\alice\project"
        ));
        assert!(cwd_is_under(
            r"C:\Users\alice\project\",
            r"C:\Users\alice\project"
        ));
        assert!(cwd_is_under(
            r"C:\Users\alice\project",
            r"C:\Users\alice\project\"
        ));
        assert!(!cwd_is_under(
            r"C:\Users\alice\project-other",
            r"C:\Users\alice\project"
        ));
    }

    #[cfg(windows)]
    #[test]
    fn cwd_is_under_case_insensitive_on_windows() {
        assert!(cwd_is_under(
            r"C:\USERS\ALICE\Project",
            r"c:\users\alice\project"
        ));
        assert!(cwd_is_under(
            r"c:\users\alice\project\Src",
            r"C:\Users\alice\project"
        ));
    }

    #[tokio::test]
    async fn build_entry_json_skips_session_for_chat_rows() {
        // Only `run` rows get the transcript drill-through; chat rows
        // already carry their content in the row itself.
        use crate::services::share_resolver::ResolverContext;
        let temp = tempfile::TempDir::new().unwrap();
        let store = SessionStore::with_path(temp.path().join("config.json"));
        let ctx = ResolverContext {
            project_root: temp.path().to_path_buf(),
            session_store: store,
            chat_sessions_dir: temp.path().join("sessions"),
            claude_projects_root: temp.path().join("claude_projects"),
            codex_sessions_root: temp.path().join("codex"),
            gemini_tmp_root: temp.path().join("gemini"),
            pi_sessions_root: temp.path().join("pi"),
            opencode_db_path: temp.path().join("opencode.db"),
            plugin_transcripts: std::collections::HashMap::new(),
        };
        let mut entry = test_entry("c1", "2026-05-24T09:00:00Z", "chat");
        entry.session_id = Some("chat-xyz".into());
        let value = build_entry_json(&entry, &ctx).await.unwrap();
        assert!(value.get("session").is_none());
    }

    fn test_entry(id: &str, ts: &str, source: &str) -> LogEntry {
        LogEntry {
            id: id.to_string(),
            ts_utc: ts.to_string(),
            source: source.to_string(),
            ..Default::default()
        }
    }

    /// `LogsArgs` with `--by <name>` and everything else at its default — just
    /// enough to exercise `SourcePlan::from_args`.
    fn logs_args_by(by: Option<&str>) -> LogsArgs {
        LogsArgs {
            action: None,
            target: None,
            limit: 20,
            json: false,
            watch: false,
            jsonl: false,
            search: None,
            by: by.map(str::to_string),
            model: None,
            key: None,
            cwd: None,
            all: false,
            since: None,
            until: None,
            errors: false,
            no_redact: false,
            open: false,
            debug_local_only: false,
            force: false,
        }
    }

    #[test]
    fn source_plan_plugin_name_scopes_to_that_tool() {
        // `--by omp` (a plugin coding-agent, not a native CLI) must scope
        // logs.db to that tool and drop native + the broad plugin_runs pull —
        // otherwise the filter is a no-op that shows every plugin and every
        // native session.
        let plan = SourcePlan::from_args(&logs_args_by(Some("omp")));
        assert!(plan.include_logs());
        assert_eq!(plan.logs_by().as_deref(), Some("omp"));
        assert!(!plan.include_native());
        assert!(!plan.plugin_runs);
    }

    #[test]
    fn source_plan_plugin_name_is_case_insensitive() {
        // `by` is lowercased before matching, so the SQL tool filter (also
        // lowercased) sees a normalized value.
        let plan = SourcePlan::from_args(&logs_args_by(Some("OMP")));
        assert_eq!(plan.logs_by().as_deref(), Some("omp"));
        assert!(!plan.include_native());
    }

    #[test]
    fn source_plan_default_keeps_unified_view() {
        // No --by: chat sessions + native + plugin runs, the unchanged default.
        let plan = SourcePlan::from_args(&logs_args_by(None));
        assert!(plan.include_logs());
        assert_eq!(plan.logs_by().as_deref(), Some("code"));
        assert!(plan.include_native());
        assert!(plan.plugin_runs);
    }

    #[test]
    fn source_plan_native_cli_name_uses_native_source() {
        // Regression guard: a real native CLI name still routes to the native
        // source, not the new plugin arm.
        let plan = SourcePlan::from_args(&logs_args_by(Some("claude")));
        assert!(plan.include_native());
        assert!(!plan.include_logs());
        assert!(!plan.plugin_runs);
    }

    #[test]
    fn native_filter_rejects_threads_for_plugin_name() {
        // A native session must never satisfy `--by <plugin>`.
        let args = logs_args_by(Some("omp"));
        let thread = test_thread("n1", "2026-05-01T10:00:00Z".parse().unwrap());
        assert!(!native_passes_filters(&thread, &args, None));
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
    fn run_meta_suffix_includes_key_and_exit() {
        let suffix = format_run_meta_suffix_plain(&RunMeta {
            key_name: Some("copilot-1".into()),
            exit_code: Some(0),
        });
        assert_eq!(suffix, " · copilot-1 · exit 0");
    }

    #[test]
    fn run_meta_suffix_drops_missing_fields() {
        let suffix = format_run_meta_suffix_plain(&RunMeta {
            key_name: None,
            exit_code: Some(2),
        });
        assert_eq!(suffix, " · exit 2");
    }

    #[test]
    fn run_meta_suffix_empty_when_nothing_known() {
        let suffix = format_run_meta_suffix_plain(&RunMeta::default());
        assert!(suffix.is_empty());
    }

    #[test]
    fn picker_label_appends_run_meta_for_native_row() {
        let row = UnifiedRow::Native(test_thread(
            "abc123",
            DateTime::parse_from_rfc3339("2026-03-27T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        ));
        let mut meta = HashMap::new();
        meta.insert(
            "abc123".to_string(),
            RunMeta {
                key_name: Some("copilot-1".into()),
                exit_code: Some(0),
            },
        );
        let label = row.picker_label(ID_COL_WIDTH, 60, &HashSet::new(), &meta);
        assert!(
            label.contains("· copilot-1 · exit 0"),
            "label missing run-meta suffix: {label}"
        );
    }

    #[test]
    fn min_unique_id_width_widens_for_uuidv7_prefix_collision() {
        // Dash-stripped, both start with `019e47b1` and diverge at index 8.
        let ts: DateTime<Utc> = "2026-05-01T10:00:00Z".parse().unwrap();
        let rows = vec![
            UnifiedRow::Native(test_thread("019e47b1-1a3b-711f-a383-2f1d2cf040e5", ts)),
            UnifiedRow::Native(test_thread("019e47b1-4e7c-767c-894c-3ffde3f26302", ts)),
        ];
        assert_eq!(min_unique_id_width(&rows), 9);
    }

    #[test]
    fn min_unique_id_width_returns_floor_for_distinct_ids() {
        let ts: DateTime<Utc> = "2026-05-01T10:00:00Z".parse().unwrap();
        let rows = vec![
            UnifiedRow::Native(test_thread("aaaaaaaa-1111-2222-3333-444444444444", ts)),
            UnifiedRow::Native(test_thread("bbbbbbbb-1111-2222-3333-444444444444", ts)),
        ];
        assert_eq!(min_unique_id_width(&rows), ID_COL_WIDTH);
    }

    #[test]
    fn min_unique_id_width_caps_when_collision_persists() {
        let ts: DateTime<Utc> = "2026-05-01T10:00:00Z".parse().unwrap();
        let rows = vec![
            UnifiedRow::Native(test_thread(&"a".repeat(40), ts)),
            UnifiedRow::Native(test_thread(&"a".repeat(40), ts)),
        ];
        assert_eq!(min_unique_id_width(&rows), ID_COL_WIDTH_MAX);
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

    fn unified_key(row: &UnifiedRow) -> String {
        match row {
            UnifiedRow::Log(e) => format!("log:{}", e.id),
            UnifiedRow::Native(t) => format!("native:{}", t.session_id),
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

        let merged = merge_unified(logs, native, 10);
        let order: Vec<String> = merged.iter().map(unified_key).collect();
        assert_eq!(order, vec!["log:L1", "native:N1", "log:L2", "native:N2",]);
    }

    #[test]
    fn merge_unified_caps_at_limit() {
        let logs = vec![
            test_entry("L1", "2026-05-01T10:00:00Z", "chat"),
            test_entry("L2", "2026-05-01T09:00:00Z", "chat"),
            test_entry("L3", "2026-05-01T08:00:00Z", "chat"),
        ];
        let native = vec![test_thread("N1", "2026-05-01T07:00:00Z".parse().unwrap())];

        let merged = merge_unified(logs, native, 2);
        assert_eq!(merged.len(), 2);
        let order: Vec<String> = merged.iter().map(unified_key).collect();
        assert_eq!(order, vec!["log:L1", "log:L2"]);
    }

    #[test]
    fn merge_unified_handles_empty_sources() {
        let merged: Vec<UnifiedRow> = merge_unified(Vec::new(), Vec::new(), 5);
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
