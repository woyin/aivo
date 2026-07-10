use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::cli::StatsArgs;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::ai_launcher::AIToolType;
use crate::services::global_stats::{self, normalize_model_for_display};
use crate::services::session_store::{ChatTokenWindow, UsageStats};
use crate::style;

/// By tool `tokens` column placeholder for launch-only tools.
const TOK_NOT_TRACKED: &str = "—";

/// Max rows in a `By model` table before the tail folds into an `others` row.
const MAX_MODEL_ROWS: usize = 20;

/// Per-model totals shown in the `By model` table.
///
/// `input` + `output` is the displayed "tokens" column (fresh I/O).
/// `cache_read` + `cache_write` is the displayed "cached" column.
/// `total` (input + output + cache_read + cache_write) matches what most
/// provider consoles surface as "total tokens".
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct ModelTotals {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl ModelTotals {
    fn tokens(&self) -> u64 {
        self.input.saturating_add(self.output)
    }
    fn cached(&self) -> u64 {
        self.cache_read.saturating_add(self.cache_write)
    }
    fn total(&self) -> u64 {
        self.tokens().saturating_add(self.cached())
    }
    /// Accumulate one report's four token dimensions (saturating).
    fn add(&mut self, input: u64, output: u64, cache_read: u64, cache_write: u64) {
        self.input = self.input.saturating_add(input);
        self.output = self.output.saturating_add(output);
        self.cache_read = self.cache_read.saturating_add(cache_read);
        self.cache_write = self.cache_write.saturating_add(cache_write);
    }
}

pub struct StatsCommand {
    store: SessionStore,
}

impl StatsCommand {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }

    pub async fn execute(&self, args: StatsArgs) -> ExitCode {
        if let Some(ref tool) = args.by {
            return self.show_tool(tool, &args).await;
        }
        self.show(&args).await
    }

    async fn show(&self, args: &StatsArgs) -> ExitCode {
        let stats = match self.store.load_stats().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                return crate::errors::exit_code_for_error(&e);
            }
        };

        let cutoff = match resolve_since(args.since.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{} {e}", style::red("Error:"));
                return ExitCode::UserError;
            }
        };
        let keys = self.store.get_keys().await.unwrap_or_default();
        let key_ids: HashSet<&str> = keys.iter().map(|k| k.id.as_str()).collect();
        let plugins = crate::plugin::coding_agent_plugin_names();
        let aivo_tool_counts = aggregate_tool_counts(&stats, &key_ids, &plugins);

        // Launched, self-reporting coding-agent plugins, counted up front so the
        // `(x/N)` counter spans native tools *and* plugins. `is_native_tool`
        // guards the rare plugin-shadows-a-builtin case (the scan owns those).
        let mut probe_targets: Vec<String> = Vec::new();
        for (tool, &count) in &aivo_tool_counts {
            if tool.as_str() != "code"
                && count > 0
                && !global_stats::is_native_tool(tool)
                && crate::plugin::stats::probes_stats(tool)
            {
                probe_targets.push(tool.clone());
            }
        }
        probe_targets.sort();
        let native_count = global_stats::native_present_count();
        let total_steps = native_count + probe_targets.len();

        let global = global_stats::collect_all(args.refresh, cutoff, probe_targets.len()).await;

        if stats.is_empty() && global.is_empty() {
            if args.json {
                return print_json(&empty_overview_json());
            }
            println!("{}", style::dim("No usage stats recorded yet."));
            return ExitCode::Success;
        }

        let fmt = if args.numbers {
            format_number
        } else {
            format_human
        };

        let mut tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        for (tool, gs) in &global {
            if !is_valid_tool(tool, &plugins) {
                continue;
            }
            tool_tokens.insert(
                tool.clone(),
                ToolTokenSummary {
                    sessions: gs.sessions,
                    input: gs.input_tokens,
                    output: gs.output_tokens,
                    cache_read: gs.cache_read_tokens,
                    cache_write: gs.cache_write_tokens,
                },
            );
        }
        // Tools with launches but no global (event) token source. Prefer the
        // plugin's own `--aivo-stats` report — its per-session `ts` lets us window
        // it by `--since`, the same way the `--by <tool>` view does, so a
        // probe-backed tool (e.g. amp) no longer vanishes from a windowed overview.
        // Without a probe we fall back to recorded per-(tool, model) tokens or a
        // launch-only `—`; both are lifetime-only counters, so under `--since` a
        // probe-less tool falls back to the endpoint tokens stamped on its
        // finished run rows (timestamped), accumulated per model here so the
        // breakdown below stays consistent with the per-tool totals.
        let mut launch_only: HashSet<String> = HashSet::new();
        let mut plugin_window_models: HashMap<String, ModelTotals> = HashMap::new();
        // Tools needing a non-global summary: skip chat, zero-launch, and tools
        // the global (event) source already covers.
        let pending: Vec<(&String, u64)> = aivo_tool_counts
            .iter()
            .filter(|&(tool, &count)| {
                tool != "code"
                    && count > 0
                    && tool_tokens.get(tool).is_none_or(|t| t.total_tokens() == 0)
            })
            .map(|(tool, &count)| (tool, count))
            .collect();
        // Probe concurrently (each spawns the plugin binary, 5s timeout), but
        // drive the stream so the `(x/N)` counter advances as each finishes.
        let probe_progress =
            !probe_targets.is_empty() && std::io::IsTerminal::is_terminal(&std::io::stderr());
        let mut probe_futs: futures::stream::FuturesUnordered<_> = probe_targets
            .iter()
            .map(
                |tool| async move { (tool.clone(), crate::plugin::stats::probe_stats(tool).await) },
            )
            .collect();
        let mut probe_reports: HashMap<String, crate::plugin::stats::PluginStatsReport> =
            HashMap::new();
        let mut probed = 0usize;
        {
            use futures::StreamExt as _;
            while let Some((tool, report)) = probe_futs.next().await {
                probed += 1;
                if probe_progress {
                    global_stats::render_step(native_count + probed, total_steps, &tool);
                }
                if let Some(report) = report {
                    probe_reports.insert(tool, report);
                }
            }
        }
        if probe_progress {
            global_stats::clear_progress_line();
        }
        for &(tool, count) in &pending {
            let report = probe_reports.get(tool.as_str());
            let summary = match (report, cutoff) {
                // Probe present: window per-session `ts` by the cutoff (None =
                // lifetime). Lifetime keeps the launch count when the probe
                // carries no usage rows; a window shows only its in-window sessions.
                (Some(r), _) => {
                    let (cnt, models) = aggregate_plugin_sessions(r, cutoff);
                    let (i, o, cr, cw) = sum_model_totals(&models);
                    let sessions = if cutoff.is_none() && cnt == 0 {
                        count
                    } else {
                        cnt
                    };
                    Some((sessions, i, o, cr, cw))
                }
                // No probe, lifetime: fall back to per-key counters / launch count.
                (None, None) => {
                    let (i, o, cr, cw) =
                        sum_model_totals(&per_tool_model_totals(&stats, tool, &key_ids));
                    Some((count, i, o, cr, cw))
                }
                // No probe under `--since`: lifetime per-key counters can't be
                // windowed, but the endpoint stamps each run's tokens on its
                // finished tool_launch row — read those, windowed. Launch count
                // comes from the started rows in the same window.
                (None, Some(c)) => {
                    let (models, windowed_launches) = self.windowed_run_usage(c, tool).await;
                    let (i, o, cr, cw) = sum_model_totals(&models);
                    for (model, t) in models {
                        plugin_window_models.entry(model).or_default().add(
                            t.input,
                            t.output,
                            t.cache_read,
                            t.cache_write,
                        );
                    }
                    if i + o + cr + cw == 0 && windowed_launches == 0 {
                        None
                    } else {
                        Some((windowed_launches, i, o, cr, cw))
                    }
                }
            };
            let Some((sessions, input, output, cache_read, cache_write)) = summary else {
                continue;
            };
            // Under `--since`, drop a tool with no in-window activity rather than
            // fabricating a row from lifetime launch counts.
            if cutoff.is_some() && sessions == 0 && input + output + cache_read + cache_write == 0 {
                continue;
            }
            if input + output + cache_read + cache_write == 0 {
                launch_only.insert(tool.clone());
            }
            tool_tokens.insert(
                tool.clone(),
                ToolTokenSummary {
                    sessions,
                    input,
                    output,
                    cache_read,
                    cache_write,
                },
            );
        }
        // Chat sessions go through the index (timestamped), not the per-key
        // counters (lifetime-only). One walk yields both count and tokens.
        let (chat_sessions, chat_window) = match cutoff {
            Some(c) => {
                let w = self.store.aggregate_chat_window_since(c).await;
                (w.count, w)
            }
            None => (
                self.store.count_chat_sessions().await,
                ChatTokenWindow::default(),
            ),
        };
        let chat_tokens = chat_tokens_for_summary(&stats, &key_ids, cutoff, &chat_window);
        if chat_tokens.total_tokens() > 0 || chat_sessions > 0 {
            tool_tokens.insert(
                "code".to_string(),
                ToolTokenSummary {
                    sessions: chat_sessions,
                    ..chat_tokens
                },
            );
        }

        let (total_input, total_output, total_cache_read, total_cache_write) = tool_tokens
            .values()
            .fold((0u64, 0u64, 0u64, 0u64), |(i, o, cr, cw), t| {
                (
                    i + t.input,
                    o + t.output,
                    cr + t.cache_read,
                    cw + t.cache_write,
                )
            });
        let total_tokens = total_input.saturating_add(total_output);
        let total_cache = total_cache_read.saturating_add(total_cache_write);
        let show_cache = total_cache > 0;
        let total_sessions: u64 = tool_tokens.values().map(|t| t.sessions).sum();

        let mut aivo_model_usage = aivo_model_usage_for_window(&stats, &key_ids, cutoff);
        for (model, tokens) in &chat_window.per_model {
            let key = global_stats::normalize_model_for_display(model);
            let entry = aivo_model_usage.entry(key).or_default();
            entry.input = entry.input.saturating_add(tokens.prompt_tokens);
            entry.output = entry.output.saturating_add(tokens.completion_tokens);
            entry.cache_read = entry.cache_read.saturating_add(tokens.cache_read_tokens);
            entry.cache_write = entry.cache_write.saturating_add(tokens.cache_write_tokens);
        }
        let mut model_tokens = combine_model_tokens(&global, &aivo_model_usage);
        // Fold in the windowed per-model tokens from probe-less plugin runs (their
        // endpoint usage isn't in `global`/`aivo_model_usage`), so the breakdown
        // matches the per-tool totals computed above.
        for (model, t) in &plugin_window_models {
            model_tokens.entry(model.clone()).or_default().add(
                t.input,
                t.output,
                t.cache_read,
                t.cache_write,
            );
        }
        // Under --since, surface models that were launched in the window even
        // if no upstream usage was recorded — `logs.db` is the table-of-truth
        // for "what did I run", independent of provider-side `usage` fields.
        if let Some(c) = cutoff {
            let run_models = self
                .store
                .logs()
                .aggregate_run_models_since(c, None)
                .await
                .unwrap_or_default();
            for model in run_models.keys() {
                let key = global_stats::normalize_model_for_display(model);
                model_tokens.entry(key).or_default();
            }
        }
        // Under --since, every entry in `model_tokens` represents real activity
        // (either real tokens, or a `tool_launch` row); count them all so the
        // header tally matches the rendered table. Without --since, keep the
        // legacy filter so stale zero-token rows from older parsers don't
        // inflate the count.
        let total_models = if cutoff.is_some() {
            model_tokens.len() as u64
        } else {
            model_tokens
                .values()
                .filter(|m| m.tokens() > 0 || m.cached() > 0)
                .count() as u64
        };

        let omitted_sources: &[&str] = if cutoff.is_some() {
            &["aivo-proxy"]
        } else {
            &[]
        };
        let window = cutoff.and_then(|c| args.since.as_deref().map(|raw| (raw, c)));

        if args.json {
            return print_json(&build_overview_json(
                &tool_tokens,
                &launch_only,
                &model_tokens,
                (total_input, total_output),
                (total_cache_read, total_cache_write),
                total_sessions,
                total_models,
                args.search.as_deref(),
                window,
                omitted_sources,
            ));
        }

        let mut parts = Vec::new();
        if total_tokens > 0 {
            parts.push(format!("{} tokens", colorize_unit(&fmt(total_tokens))));
        }
        if show_cache {
            parts.push(format!("{} cached", colorize_unit(&fmt(total_cache))));
        }
        parts.push(format!("{} sessions", colorize_unit(&fmt(total_sessions))));
        parts.push(format!("{} models", colorize_unit(&fmt(total_models))));
        let header = parts.join(" · ");
        style::print_header(&header);

        if !tool_tokens.is_empty() {
            println!();
            // `None` = launch-only tool (no token attribution) → renders `—`.
            let mut rows: Vec<(&str, u64, Option<u64>)> = tool_tokens
                .iter()
                .map(|(name, t)| {
                    let tokens = if launch_only.contains(name) {
                        None
                    } else {
                        Some(t.total_tokens())
                    };
                    (name.as_str(), t.sessions, tokens)
                })
                .collect();
            rows.sort_by(|a, b| {
                b.2.unwrap_or(0)
                    .cmp(&a.2.unwrap_or(0))
                    .then_with(|| b.1.cmp(&a.1))
            });

            let name_w = rows
                .iter()
                .map(|(n, _, _)| n.len())
                .max()
                .unwrap_or(0)
                .max("By tool".len());
            let ses_w = rows
                .iter()
                .map(|(_, s, _)| fmt(*s).len())
                .max()
                .unwrap_or(0)
                .max("sessions".len());
            let tok_w = rows
                .iter()
                .map(|(_, _, t)| match t {
                    Some(v) => fmt(*v).chars().count(),
                    None => TOK_NOT_TRACKED.chars().count(),
                })
                .max()
                .unwrap_or(0)
                .max("tokens".len());
            let max_tok = rows.iter().filter_map(|(_, _, t)| *t).max().unwrap_or(0);

            // Title row with column headers — pad plain text first, then style
            println!(
                "{} {} {}",
                style::bold(format!("{:<name_w$}", "By tool")),
                style::dim(format!("{:>ses_w$}", "sessions")),
                style::dim(format!("{:>tok_w$}", "tokens")),
            );

            let show_tool_bar = rows.len() > 1;
            for (name, ses, tok) in &rows {
                let pn = format!("{:<width$}", name, width = name_w);
                let ps = colorize_unit(&format!("{:>width$}", fmt(*ses), width = ses_w));
                let pt = match tok {
                    Some(v) => colorize_unit(&format!("{:>width$}", fmt(*v), width = tok_w)),
                    None => style::dim(format!("{:>tok_w$}", TOK_NOT_TRACKED)),
                };
                match (show_tool_bar, tok) {
                    (true, Some(v)) => {
                        println!("{} {} {} {}", style::cyan(&pn), ps, pt, bar(*v, max_tok),);
                    }
                    _ => println!("{} {} {}", style::cyan(&pn), ps, pt),
                }
            }
        }

        render_model_table(&model_tokens, fmt, args);
        render_since_footer(args.since.as_deref(), omitted_sources);

        ExitCode::Success
    }

    async fn show_tool(&self, tool: &str, args: &StatsArgs) -> ExitCode {
        let tool = tool.to_lowercase();
        let plugins = crate::plugin::coding_agent_plugin_names();
        if !is_valid_tool(&tool, &plugins) {
            let mut valid: Vec<String> = ["claude", "codex", "gemini", "opencode", "pi", "code"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            let mut names: Vec<String> = plugins.into_iter().collect();
            names.sort();
            valid.extend(names);
            eprintln!(
                "{} Unknown tool '{}'. Valid tools: {}.",
                style::red("Error:"),
                tool,
                valid.join(", "),
            );
            eprintln!("Run `aivo stats --help` for details.");
            return ExitCode::UserError;
        }
        // codex-app shares ~/.codex/sessions with the codex CLI, so report them
        // as one rather than a launch-only `codex-app` view with no tokens.
        let tool = canonical_stats_tool(&tool).to_string();

        let fmt = if args.numbers {
            format_number
        } else {
            format_human
        };

        let cutoff = match resolve_since(args.since.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("{} {e}", style::red("Error:"));
                return ExitCode::UserError;
            }
        };

        let global = match global_stats::collect(&tool, args.refresh, cutoff).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "{} Failed to read {} data: {}",
                    style::red("Error:"),
                    global_stats::tool_display_name(&tool),
                    e
                );
                None
            }
        };
        // No native usage source (plugin agents, never-launched native tools):
        // use recorded per-(tool, model) tokens if any, else launch counts.
        let Some(gs) = global.filter(|gs| gs.total_tokens() > 0 || gs.sessions > 0) else {
            return self.show_tool_no_global(&tool, args, cutoff, fmt).await;
        };

        let mut view = ToolView {
            count: gs.sessions,
            input_tokens: gs.input_tokens,
            output_tokens: gs.output_tokens,
            cache_read: gs.cache_read_tokens,
            cache_write: gs.cache_write_tokens,
            models: gs
                .models
                .iter()
                .map(|(name, m)| {
                    (
                        name.clone(),
                        ModelTotals {
                            input: m.input_tokens,
                            output: m.output_tokens,
                            cache_read: m.cache_read_tokens,
                            cache_write: m.cache_write_tokens,
                        },
                    )
                })
                .collect(),
        };

        // Surface models launched in the window that recorded no usage (a run
        // with all-zero token accounting) so --since still lists them.
        if let Some(c) = cutoff {
            let run_models = self
                .store
                .logs()
                .aggregate_run_models_since(c, Some(&tool))
                .await
                .unwrap_or_default();
            for model in run_models.keys() {
                let key = global_stats::normalize_model_for_display(model);
                view.models.entry(key).or_default();
            }
        }

        let window = cutoff.and_then(|c| args.since.as_deref().map(|raw| (raw, c)));

        if args.json {
            return print_json(&build_tool_view_json(
                &tool,
                &view,
                "global",
                "sessions",
                args,
                window,
                &[],
            ));
        }

        print_tool_view(&view, "sessions", fmt, args);
        render_since_footer(args.since.as_deref(), &[]);

        ExitCode::Success
    }

    /// Per-tool view for a tool with no native usage source. Precedence:
    /// the plugin's own `--aivo-stats` report (its complete data) → aivo's
    /// recorded per-(tool, model) tokens → launch counts.
    async fn show_tool_no_global(
        &self,
        tool: &str,
        args: &StatsArgs,
        cutoff: Option<chrono::DateTime<chrono::Utc>>,
        fmt: fn(u64) -> String,
    ) -> ExitCode {
        // A coding-agent plugin that implements `--aivo-stats` reports its own
        // usage (its data folder + format) — the authoritative, complete source.
        // It only provides raw per-session data; aivo windows/aggregates below.
        if crate::plugin::stats::probes_stats(tool)
            && let Some(report) = crate::plugin::stats::probe_stats(tool).await
        {
            return render_plugin_report(tool, &report, args, cutoff, fmt);
        }
        // Per-(tool, model) tokens, then launches. Lifetime reads the per-key
        // counters in UsageStats; `--since` reads the timestamped tokens the
        // endpoint stamps on each run's finished row (the per-key counters carry
        // no timestamp), so a probe-less coding-agent plugin is windowable too.
        let (models, launches) = match cutoff {
            None => {
                let stats = self.store.load_stats().await.unwrap_or_default();
                let keys = self.store.get_keys().await.unwrap_or_default();
                let key_ids: HashSet<&str> = keys.iter().map(|k| k.id.as_str()).collect();
                let plugins = crate::plugin::coding_agent_plugin_names();
                let launches = aggregate_tool_counts(&stats, &key_ids, &plugins)
                    .get(tool)
                    .copied()
                    .unwrap_or(0);
                (per_tool_model_totals(&stats, tool, &key_ids), launches)
            }
            Some(c) => self.windowed_run_usage(c, tool).await,
        };
        let (input, output, cache_read, cache_write) = sum_model_totals(&models);
        if input + output + cache_read + cache_write > 0 {
            let view = ToolView {
                count: launches,
                input_tokens: input,
                output_tokens: output,
                cache_read,
                cache_write,
                models,
            };
            if args.json {
                let window = cutoff.and_then(|c| args.since.as_deref().map(|raw| (raw, c)));
                return print_json(&build_tool_view_json(
                    tool,
                    &view,
                    "aivo",
                    "launches",
                    args,
                    window,
                    &[],
                ));
            }
            print_tool_view(&view, "launches", fmt, args);
            render_since_footer(args.since.as_deref(), &[]);
            return ExitCode::Success;
        }
        self.show_tool_launches(tool, args, cutoff, fmt).await
    }

    /// Windowed per-model tokens + launch count for one tool from logs.db run
    /// rows: tokens from the finished rows the endpoint stamps, launches from
    /// the started rows (model-less launches included).
    async fn windowed_run_usage(
        &self,
        cutoff: chrono::DateTime<chrono::Utc>,
        tool: &str,
    ) -> (HashMap<String, ModelTotals>, u64) {
        let logs = self.store.logs();
        let (tokens, launches) = tokio::join!(
            logs.aggregate_run_tokens_since(cutoff, Some(tool)),
            logs.count_runs_since(cutoff, Some(tool)),
        );
        (
            run_model_tokens_to_totals(tokens.unwrap_or_default()),
            launches.unwrap_or_default(),
        )
    }

    /// Launch-count view for tools with no per-tool token attribution: real
    /// counts from logs.db instead of fabricated token totals.
    async fn show_tool_launches(
        &self,
        tool: &str,
        args: &StatsArgs,
        cutoff: Option<chrono::DateTime<chrono::Utc>>,
        fmt: fn(u64) -> String,
    ) -> ExitCode {
        let since = cutoff.unwrap_or_else(epoch_utc);
        let model_counts = self
            .store
            .logs()
            .aggregate_run_models_since(since, Some(tool))
            .await
            .unwrap_or_default();
        let launches: u64 = model_counts.values().sum();

        if args.json {
            return print_json(&build_launch_view_json(
                tool,
                &model_counts,
                launches,
                args,
                cutoff,
            ));
        }

        if launches == 0 {
            println!(
                "{}",
                style::dim(format!(
                    "No launches recorded for {}.",
                    global_stats::tool_display_name(tool)
                ))
            );
            render_since_footer(args.since.as_deref(), &[]);
            return ExitCode::Success;
        }

        print_launch_view(tool, launches, &model_counts, fmt, args);
        render_since_footer(args.since.as_deref(), &[]);
        ExitCode::Success
    }

    pub fn print_help() {
        println!("{} aivo stats [options]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Show usage statistics: token counts, request counts, and breakdowns.")
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt(
            "--by <NAME>",
            "Filter to one tool or plugin (e.g. claude, code)",
        );
        print_opt("-n, --numbers", "Exact numbers instead of human-readable");
        print_opt("-r, --refresh", "Bypass cache and re-read all data files");
        print_opt("-s, --search <QUERY>", "Search by key, model, or tool name");
        print_opt("-a, --all", "Show all models (default: top 20)");
        print_opt(
            "-d, --detailed",
            "Expand per-model to input/output/cached/total",
        );
        print_opt(
            "--since <DURATION>",
            "Filter to the last N units (7d, 24h, 30m, 2w)",
        );
        print_opt("--json", "Output stats as JSON (all models, exact numbers)");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo stats"));
        println!("  {}", style::dim("aivo stats --by claude"));
        println!("  {}", style::dim("aivo stats --since 7d"));
    }
}

fn filter_models<'a>(
    models: impl IntoIterator<Item = (&'a String, &'a ModelTotals)>,
    search: Option<&str>,
    keep_zero_rows: bool,
) -> Vec<(String, ModelTotals)> {
    let needle = search.map(|s| s.to_lowercase());
    let mut rows: Vec<(String, ModelTotals)> = models
        .into_iter()
        .filter(|(_, m)| keep_zero_rows || m.tokens() > 0 || m.cached() > 0)
        .filter(|(name, _)| {
            needle
                .as_ref()
                .is_none_or(|q| name.to_lowercase().contains(q))
        })
        .map(|(name, m)| (name.clone(), *m))
        .collect();
    // Rank by fresh I/O first, then cached as tiebreaker.
    rows.sort_by(|a, b| {
        b.1.tokens()
            .cmp(&a.1.tokens())
            .then_with(|| b.1.cached().cmp(&a.1.cached()))
    });
    rows
}

fn print_json(payload: &Value) -> ExitCode {
    match serde_json::to_string_pretty(payload) {
        Ok(s) => {
            println!("{}", s);
            ExitCode::Success
        }
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            ExitCode::UserError
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_overview_json(
    tool_tokens: &HashMap<String, ToolTokenSummary>,
    launch_only: &HashSet<String>,
    model_tokens: &HashMap<String, ModelTotals>,
    (total_input, total_output): (u64, u64),
    (total_cache_read, total_cache_write): (u64, u64),
    total_sessions: u64,
    total_models: u64,
    search: Option<&str>,
    window: Option<(&str, chrono::DateTime<chrono::Utc>)>,
    omitted_sources: &[&str],
) -> Value {
    let mut tool_rows: Vec<(&String, &ToolTokenSummary)> = tool_tokens.iter().collect();
    tool_rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_tokens()));
    let by_tool: Vec<Value> = tool_rows
        .into_iter()
        .map(|(name, t)| {
            // Launch-only: emit the count + flag, no fabricated token fields.
            if launch_only.contains(name) {
                json!({
                    "name": name,
                    "sessions": t.sessions,
                    "per_tool_tokens_tracked": false,
                })
            } else {
                json!({
                    "name": name,
                    "sessions": t.sessions,
                    "tokens": t.total_tokens(),
                    "input_tokens": t.input,
                    "output_tokens": t.output,
                    "cache_read_tokens": t.cache_read,
                    "cache_write_tokens": t.cache_write,
                    "per_tool_tokens_tracked": true,
                })
            }
        })
        .collect();

    let by_model: Vec<Value> = filter_models(model_tokens, search, window.is_some())
        .into_iter()
        .map(|(name, m)| {
            json!({
                "name": name,
                "tokens": m.tokens(),
                "input_tokens": m.input,
                "output_tokens": m.output,
                "cached_tokens": m.cached(),
                "cache_read_tokens": m.cache_read,
                "cache_write_tokens": m.cache_write,
                "total_tokens": m.total(),
            })
        })
        .collect();

    let mut payload = json!({
        "totals": {
            "tokens": total_input.saturating_add(total_output),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_tokens": total_cache_read.saturating_add(total_cache_write),
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "sessions": total_sessions,
            "models": total_models,
        },
        "by_tool": by_tool,
        "by_model": by_model,
        "omitted_sources": omitted_sources,
    });
    if let Some((raw, cutoff)) = window {
        payload["window"] = json!({
            "since": raw,
            "since_iso": cutoff.to_rfc3339(),
        });
    }
    payload
}

fn build_tool_view_json(
    tool: &str,
    view: &ToolView,
    source: &str,
    count_label: &str,
    args: &StatsArgs,
    window: Option<(&str, chrono::DateTime<chrono::Utc>)>,
    omitted_sources: &[&str],
) -> Value {
    let by_model: Vec<Value> =
        filter_models(&view.models, args.search.as_deref(), window.is_some())
            .into_iter()
            .map(|(name, m)| {
                json!({
                    "name": name,
                    "tokens": m.tokens(),
                    "input_tokens": m.input,
                    "output_tokens": m.output,
                    "cached_tokens": m.cached(),
                    "cache_read_tokens": m.cache_read,
                    "cache_write_tokens": m.cache_write,
                    "total_tokens": m.total(),
                })
            })
            .collect();
    // Match the human view's count: under --since, every entry in
    // `view.models` represents real activity (real tokens or a logged
    // launch), so count them all. Without a window keep the legacy filter.
    let total_models = if window.is_some() {
        view.models.len() as u64
    } else {
        view.models
            .values()
            .filter(|m| m.tokens() > 0 || m.cached() > 0)
            .count() as u64
    };
    let mut payload = json!({
        "tool": tool,
        "source": source,
        "totals": {
            "tokens": view.input_tokens.saturating_add(view.output_tokens),
            "input_tokens": view.input_tokens,
            "output_tokens": view.output_tokens,
            "cache_tokens": view.cache_read.saturating_add(view.cache_write),
            "cache_read_tokens": view.cache_read,
            "cache_write_tokens": view.cache_write,
            "count": view.count,
            "count_label": count_label,
            "models": total_models,
        },
        "by_model": by_model,
        "omitted_sources": omitted_sources,
    });
    if let Some((raw, cutoff)) = window {
        payload["window"] = json!({
            "since": raw,
            "since_iso": cutoff.to_rfc3339(),
        });
    }
    payload
}

/// Aggregate a plugin's per-session report into `(session_count, per-model
/// totals)`, applying the `--since` cutoff **host-side**. Sessions before the
/// cutoff — and, under `--since`, sessions the plugin couldn't timestamp — are
/// dropped. Model names are normalized for display, merging same-named models.
fn aggregate_plugin_sessions(
    report: &crate::plugin::stats::PluginStatsReport,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> (u64, HashMap<String, ModelTotals>) {
    let mut models: HashMap<String, ModelTotals> = HashMap::new();
    let mut count = 0u64;
    for session in &report.sessions {
        if let Some(cut) = cutoff {
            let ts = session
                .ts
                .as_deref()
                .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
                .map(|t| t.with_timezone(&chrono::Utc));
            if ts.is_none_or(|t| t < cut) {
                continue;
            }
        }
        count += 1;
        for m in &session.models {
            models
                .entry(global_stats::normalize_model_for_display(&m.name))
                .or_default()
                .add(
                    m.input_tokens,
                    m.output_tokens,
                    m.cache_read_tokens,
                    m.cache_write_tokens,
                );
        }
    }
    (count, models)
}

/// Render a plugin's `--aivo-stats` report as the token-accurate per-tool view,
/// labeled with its `source`. The plugin only supplies raw per-session data;
/// aivo applies `--since` and aggregation here.
fn render_plugin_report(
    tool: &str,
    report: &crate::plugin::stats::PluginStatsReport,
    args: &StatsArgs,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
    fmt: fn(u64) -> String,
) -> ExitCode {
    let (count, models) = aggregate_plugin_sessions(report, cutoff);
    let (input, output, cache_read, cache_write) = sum_model_totals(&models);
    let view = ToolView {
        count,
        input_tokens: input,
        output_tokens: output,
        cache_read,
        cache_write,
        models,
    };
    let source = report.source.as_deref().unwrap_or("plugin");
    let window = cutoff.and_then(|c| args.since.as_deref().map(|raw| (raw, c)));
    if args.json {
        return print_json(&build_tool_view_json(
            tool,
            &view,
            source,
            "sessions",
            args,
            window,
            &[],
        ));
    }
    print_tool_view(&view, "sessions", fmt, args);
    println!();
    println!("{}", style::dim(format!("source: {source}")));
    render_since_footer(args.since.as_deref(), &[]);
    ExitCode::Success
}

fn empty_overview_json() -> Value {
    json!({
        "totals": {
            "tokens": 0u64,
            "input_tokens": 0u64,
            "output_tokens": 0u64,
            "cache_tokens": 0u64,
            "cache_read_tokens": 0u64,
            "cache_write_tokens": 0u64,
            "sessions": 0u64,
            "models": 0u64,
        },
        "by_tool": Vec::<Value>::new(),
        "by_model": Vec::<Value>::new(),
    })
}

/// JSON twin of `print_launch_view`: no token fields; `per_tool_tokens_tracked:
/// false` tells consumers only launch counts are meaningful.
fn build_launch_view_json(
    tool: &str,
    model_counts: &HashMap<String, u64>,
    launches: u64,
    args: &StatsArgs,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> Value {
    let rows = filter_sort_launch_models(model_counts, args.search.as_deref());
    let by_model: Vec<Value> = rows
        .iter()
        .map(|(name, count)| json!({ "name": name, "launches": count }))
        .collect();
    let mut payload = json!({
        "tool": tool,
        "source": "logs",
        "per_tool_tokens_tracked": false,
        "totals": {
            "launches": launches,
            "models": rows.len() as u64,
        },
        "by_model": by_model,
        "omitted_sources": Vec::<&str>::new(),
    });
    if let (Some(c), Some(raw)) = (cutoff, args.since.as_deref()) {
        payload["window"] = json!({ "since": raw, "since_iso": c.to_rfc3339() });
    }
    payload
}

/// Unix epoch as an "all-time" cutoff for logs.db aggregates.
fn epoch_utc() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::<chrono::Utc>::from_timestamp(0, 0).expect("unix epoch is a valid timestamp")
}

/// Apply the `-s` search to launch-model names and sort by launch count desc
/// (name asc as tiebreak). Shared by the launch view's human and JSON renderers.
fn filter_sort_launch_models(
    model_counts: &HashMap<String, u64>,
    search: Option<&str>,
) -> Vec<(String, u64)> {
    let needle = search.map(|s| s.to_lowercase());
    let mut rows: Vec<(String, u64)> = model_counts
        .iter()
        .filter(|(name, _)| {
            needle
                .as_ref()
                .is_none_or(|q| name.to_lowercase().contains(q))
        })
        .map(|(name, count)| (name.clone(), *count))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows
}

struct ToolTokenSummary {
    sessions: u64,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
}

impl ToolTokenSummary {
    fn total_tokens(&self) -> u64 {
        self.input.saturating_add(self.output)
    }
}

/// Chat token totals for the stats `By tool` row.
///
/// Lifetime path uses per-key counters; windowed path uses pre-aggregated
/// per-session totals from the index — per-key counters are lifetime-only
/// and would over-attribute to the window.
fn chat_tokens_for_summary(
    stats: &UsageStats,
    key_ids: &HashSet<&str>,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
    chat_window: &ChatTokenWindow,
) -> ToolTokenSummary {
    if cutoff.is_some() {
        let total = chat_window.total();
        return ToolTokenSummary {
            sessions: 0,
            input: total.prompt_tokens,
            output: total.completion_tokens,
            cache_read: total.cache_read_tokens,
            cache_write: total.cache_write_tokens,
        };
    }
    tool_token_totals(stats, "code", key_ids)
}

/// Per-model usage from aivo-tracked counters, scoped to the requested
/// window. Lifetime-cumulative under the hood, so under cutoff we return an
/// empty map and the model table renders only window-filtered data — the
/// chat session index covers windowed chat models separately.
fn aivo_model_usage_for_window(
    stats: &UsageStats,
    key_ids: &HashSet<&str>,
    cutoff: Option<chrono::DateTime<chrono::Utc>>,
) -> HashMap<String, ModelTotals> {
    if cutoff.is_some() {
        return HashMap::new();
    }
    aggregate_model_usage(stats, key_ids)
}

/// Sum token totals from aivo-tracked stats for keys that used a given tool.
fn tool_token_totals(stats: &UsageStats, tool: &str, key_ids: &HashSet<&str>) -> ToolTokenSummary {
    let mut input = 0u64;
    let mut output = 0u64;
    let mut cache_read = 0u64;
    let mut cache_write = 0u64;
    for (key_id, entry) in &stats.key_usage {
        if !key_ids.contains(key_id.as_str()) {
            continue;
        }
        // Canonical match so the `code` total also folds the pre-rename `chat`
        // per-tool bucket (see `canonical_stats_tool`).
        if !entry
            .per_tool
            .iter()
            .any(|(t, c)| *c > 0 && canonical_stats_tool(t) == tool)
        {
            continue;
        }
        input += entry.prompt_tokens;
        output += entry.completion_tokens;
        cache_read += entry.cache_read_input_tokens;
        cache_write += entry.cache_creation_input_tokens;
    }
    ToolTokenSummary {
        sessions: 0,
        input,
        output,
        cache_read,
        cache_write,
    }
}

/// Real per-(tool, model) token totals for `tool`, summed across the user's
/// keys. Populated only by aivo-proxied launches (plugins, chat); empty for
/// tools with no recorded per-tool usage. The accurate replacement for the old
/// shared-key over-attribution.
fn per_tool_model_totals(
    stats: &UsageStats,
    tool: &str,
    key_ids: &HashSet<&str>,
) -> HashMap<String, ModelTotals> {
    let mut models: HashMap<String, ModelTotals> = HashMap::new();
    for (key_id, entry) in &stats.key_usage {
        if !key_ids.contains(key_id.as_str()) {
            continue;
        }
        let Some(tool_models) = entry.per_tool_model_usage.get(tool) else {
            continue;
        };
        for (model, mc) in tool_models {
            models
                .entry(normalize_model_for_display(model))
                .or_default()
                .add(
                    mc.prompt_tokens,
                    mc.completion_tokens,
                    mc.cache_read_input_tokens,
                    mc.cache_creation_input_tokens,
                );
        }
    }
    models
}

/// Convert the endpoint's timestamped per-model run tokens (from logs.db) into
/// the display `ModelTotals` map, normalizing model names the same way the
/// lifetime path does.
fn run_model_tokens_to_totals(
    raw: HashMap<String, crate::services::log_store::RunModelTokens>,
) -> HashMap<String, ModelTotals> {
    let mut models: HashMap<String, ModelTotals> = HashMap::new();
    for (model, t) in raw {
        models
            .entry(normalize_model_for_display(&model))
            .or_default()
            .add(t.input, t.output, t.cache_read, t.cache_creation);
    }
    models
}

/// Aggregate `(input, output, cache_read, cache_write)` across a per-model map.
fn sum_model_totals(models: &HashMap<String, ModelTotals>) -> (u64, u64, u64, u64) {
    models.values().fold((0, 0, 0, 0), |(i, o, cr, cw), m| {
        (
            i.saturating_add(m.input),
            o.saturating_add(m.output),
            cr.saturating_add(m.cache_read),
            cw.saturating_add(m.cache_write),
        )
    })
}

/// Token-accurate per-tool view: native (global) usage, or recorded
/// per-(tool, model) usage — both attribute tokens to the tool honestly.
struct ToolView {
    count: u64,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    models: HashMap<String, ModelTotals>,
}

fn print_tool_view(view: &ToolView, count_label: &str, fmt: fn(u64) -> String, args: &StatsArgs) {
    let total_tokens = view.input_tokens.saturating_add(view.output_tokens);
    let total_cache = view.cache_read.saturating_add(view.cache_write);
    let model_count = if args.since.is_some() {
        view.models.len() as u64
    } else {
        view.models
            .values()
            .filter(|m| m.tokens() > 0 || m.cached() > 0)
            .count() as u64
    };

    let mut parts = Vec::new();
    if total_tokens > 0 {
        parts.push(format!("{} tokens", colorize_unit(&fmt(total_tokens))));
    }
    if total_cache > 0 {
        parts.push(format!("{} cached", colorize_unit(&fmt(total_cache))));
    }
    parts.push(format!("{} {count_label}", colorize_unit(&fmt(view.count))));
    parts.push(format!("{} models", colorize_unit(&fmt(model_count))));

    let header = parts.join(" · ");
    style::print_header(&header);

    render_model_table(&view.models, fmt, args);
}

/// Launch-count view: a launch total + a "By model" table of launch counts,
/// with a note that per-tool token usage isn't tracked.
fn print_launch_view(
    tool: &str,
    launches: u64,
    model_counts: &HashMap<String, u64>,
    fmt: fn(u64) -> String,
    args: &StatsArgs,
) {
    let rows = filter_sort_launch_models(model_counts, args.search.as_deref());

    let header = format!(
        "{} launches · {} models",
        colorize_unit(&fmt(launches)),
        colorize_unit(&fmt(rows.len() as u64)),
    );
    style::print_header(&header);
    println!();
    println!(
        "{}",
        style::dim("Per-tool token usage isn't tracked (tokens are recorded per key+model, not")
    );
    println!(
        "{}",
        style::dim(format!(
            "per tool). See `aivo stats` for token totals or `aivo logs --by {tool}` for runs."
        ))
    );

    if rows.is_empty() {
        return;
    }

    // Top-N unless --all; fold the tail into an "others" row so the launch
    // total still reconciles with the table.
    let total = rows.len();
    let max_display = MAX_MODEL_ROWS;
    let truncated = !args.all && total > max_display;
    let display: Vec<(String, u64)> = if truncated {
        let others: u64 = rows[max_display..].iter().map(|(_, c)| *c).sum();
        let mut shown = rows[..max_display].to_vec();
        shown.push((format!("others ({} models)", total - max_display), others));
        shown
    } else {
        rows
    };

    println!();
    let name_w = display
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("By model".len());
    let cnt_w = display
        .iter()
        .map(|(_, c)| fmt(*c).len())
        .max()
        .unwrap_or(0)
        .max("launches".len());
    let max_count = display.iter().map(|(_, c)| *c).max().unwrap_or(0);
    let show_bar = args.search.is_none() && display.len() > 1;

    println!(
        "{} {}",
        style::bold(format!("{:<name_w$}", "By model")),
        style::dim(format!("{:>cnt_w$}", "launches")),
    );
    for (name, count) in &display {
        let pn = style::cyan(format!("{:<name_w$}", name));
        let pc = colorize_unit(&format!("{:>cnt_w$}", fmt(*count)));
        if show_bar {
            println!("{} {} {}", pn, pc, bar(*count, max_count));
        } else {
            println!("{} {}", pn, pc);
        }
    }

    println!();
    let mut hints = Vec::new();
    if truncated {
        hints.push("-a all models");
    }
    hints.push("-n numbers");
    hints.push("-s filter");
    println!("{}", style::dim(hints.join(" · ")));
}

fn render_model_table(
    models: &HashMap<String, ModelTotals>,
    fmt: fn(u64) -> String,
    args: &StatsArgs,
) {
    let searching = args.search.is_some();
    let mut model_rows = filter_models(models, args.search.as_deref(), args.since.is_some());

    if model_rows.is_empty() {
        return;
    }

    println!();

    // Detailed view ranks by total (input+output+cached) so the top rows
    // reflect what the provider console highlights. filter_models sorts by
    // fresh tokens by default, which suits the 2-column view.
    if args.detailed {
        model_rows.sort_by(|a, b| {
            b.1.total()
                .cmp(&a.1.total())
                .then_with(|| b.1.tokens().cmp(&a.1.tokens()))
        });
    }

    let total_model_count = model_rows.len();
    let max_display = MAX_MODEL_ROWS;
    let truncated = !args.all && total_model_count > max_display;

    let display_rows: Vec<(String, ModelTotals)> = if truncated {
        let others_count = total_model_count - max_display;
        let others =
            model_rows[max_display..]
                .iter()
                .fold(ModelTotals::default(), |acc, (_, m)| ModelTotals {
                    input: acc.input.saturating_add(m.input),
                    output: acc.output.saturating_add(m.output),
                    cache_read: acc.cache_read.saturating_add(m.cache_read),
                    cache_write: acc.cache_write.saturating_add(m.cache_write),
                });
        let mut rows: Vec<(String, ModelTotals)> = model_rows[..max_display].to_vec();
        rows.push((format!("others ({} models)", others_count), others));
        rows
    } else {
        model_rows
    };

    let any_cached = display_rows.iter().any(|(_, m)| m.cached() > 0);
    let show_bar = !searching && display_rows.len() > 1;

    if args.detailed {
        render_detailed_model_table(&display_rows, fmt, show_bar);
    } else {
        render_default_model_table(&display_rows, fmt, show_bar, any_cached);
    }

    println!();
    let mut hints = Vec::new();
    if truncated {
        hints.push("-a all models".to_string());
    }
    if !args.detailed {
        hints.push("-d detailed".to_string());
    }
    hints.push("-n numbers".to_string());
    hints.push("-r refresh".to_string());
    hints.push("-s filter".to_string());
    println!("{}", style::dim(hints.join(" · ")));
}

fn render_default_model_table(
    display_rows: &[(String, ModelTotals)],
    fmt: fn(u64) -> String,
    show_bar: bool,
    any_cached: bool,
) {
    let name_w = display_rows
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("By model".len());
    let tok_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.tokens()).len())
        .max()
        .unwrap_or(0)
        .max("tokens".len());
    let cache_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.cached()).len())
        .max()
        .unwrap_or(0)
        .max("cached".len());
    let max_tok = display_rows
        .iter()
        .map(|(_, m)| m.tokens())
        .max()
        .unwrap_or(0);

    if any_cached {
        println!(
            "{} {} {}",
            style::bold(format!("{:<name_w$}", "By model")),
            style::dim(format!("{:>cache_w$}", "cached")),
            style::dim(format!("{:>tok_w$}", "tokens")),
        );
    } else {
        println!(
            "{} {}",
            style::bold(format!("{:<name_w$}", "By model")),
            style::dim(format!("{:>tok_w$}", "tokens")),
        );
    }

    for (name, m) in display_rows {
        let pn = format!("{:<width$}", name, width = name_w);
        let pt = colorize_unit(&format!("{:>width$}", fmt(m.tokens()), width = tok_w));
        if any_cached {
            let pc = colorize_unit(&format!("{:>width$}", fmt(m.cached()), width = cache_w));
            if show_bar {
                println!(
                    "{} {} {} {}",
                    style::cyan(&pn),
                    pc,
                    pt,
                    bar(m.tokens(), max_tok),
                );
            } else {
                println!("{} {} {}", style::cyan(&pn), pc, pt);
            }
        } else if show_bar {
            println!("{} {} {}", style::cyan(&pn), pt, bar(m.tokens(), max_tok),);
        } else {
            println!("{} {}", style::cyan(&pn), pt);
        }
    }
}

fn render_detailed_model_table(
    display_rows: &[(String, ModelTotals)],
    fmt: fn(u64) -> String,
    show_bar: bool,
) {
    let name_w = display_rows
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("By model".len());
    let in_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.input).len())
        .max()
        .unwrap_or(0)
        .max("input".len());
    let out_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.output).len())
        .max()
        .unwrap_or(0)
        .max("output".len());
    let cache_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.cached()).len())
        .max()
        .unwrap_or(0)
        .max("cached".len());
    let total_w = display_rows
        .iter()
        .map(|(_, m)| fmt(m.total()).len())
        .max()
        .unwrap_or(0)
        .max("total".len());
    let max_total = display_rows
        .iter()
        .map(|(_, m)| m.total())
        .max()
        .unwrap_or(0);

    println!(
        "{} {} {} {} {}",
        style::bold(format!("{:<name_w$}", "By model")),
        style::dim(format!("{:>in_w$}", "input")),
        style::dim(format!("{:>out_w$}", "output")),
        style::dim(format!("{:>cache_w$}", "cached")),
        style::dim(format!("{:>total_w$}", "total")),
    );

    for (name, m) in display_rows {
        let pn = format!("{:<width$}", name, width = name_w);
        let pi = colorize_unit(&format!("{:>width$}", fmt(m.input), width = in_w));
        let po = colorize_unit(&format!("{:>width$}", fmt(m.output), width = out_w));
        let pc = colorize_unit(&format!("{:>width$}", fmt(m.cached()), width = cache_w));
        let ptot = colorize_unit(&format!("{:>width$}", fmt(m.total()), width = total_w));
        if show_bar {
            println!(
                "{} {} {} {} {} {}",
                style::cyan(&pn),
                pi,
                po,
                pc,
                ptot,
                bar(m.total(), max_total),
            );
        } else {
            println!("{} {} {} {} {}", style::cyan(&pn), pi, po, pc, ptot);
        }
    }
}

fn render_since_footer(since: Option<&str>, omitted_sources: &[&str]) {
    let Some(raw) = since else {
        return;
    };
    let mut bits = vec![format!("filtered to last {raw}")];
    for src in omitted_sources {
        bits.push(format!("{src} omitted"));
    }
    println!();
    println!("{}", style::dim(bits.join(" · ")));
}

/// Combines per-model usage from native tool data and aivo-tracked data.
///
/// Native (Claude Code, Codex, Gemini, OpenCode, Pi) and aivo-proxy data
/// overlap for any model launched through aivo. To avoid double-counting,
/// native wins when both have an entry for the same model name; aivo-only
/// models (e.g. usage that only flowed through `aivo code` or a non-native
/// integration) are added on top so they show up in the overview rather than
/// being dropped whenever any native data exists.
fn combine_model_tokens(
    global: &HashMap<String, global_stats::GlobalToolStats>,
    aivo_model_usage: &HashMap<String, ModelTotals>,
) -> HashMap<String, ModelTotals> {
    let mut model_tokens: HashMap<String, ModelTotals> = HashMap::new();
    for gs in global.values() {
        for (model, mt) in &gs.models {
            let key = normalize_model_for_display(model);
            let entry = model_tokens.entry(key).or_default();
            entry.input = entry.input.saturating_add(mt.input_tokens);
            entry.output = entry.output.saturating_add(mt.output_tokens);
            entry.cache_read = entry.cache_read.saturating_add(mt.cache_read_tokens);
            entry.cache_write = entry.cache_write.saturating_add(mt.cache_write_tokens);
        }
    }

    for (model, totals) in aivo_model_usage {
        let key = normalize_model_for_display(model);
        if model_tokens.contains_key(&key) {
            // Native already covers this model; skip to avoid double counting.
            continue;
        }
        let entry = model_tokens.entry(key).or_default();
        entry.input = entry.input.saturating_add(totals.input);
        entry.output = entry.output.saturating_add(totals.output);
        entry.cache_read = entry.cache_read.saturating_add(totals.cache_read);
        entry.cache_write = entry.cache_write.saturating_add(totals.cache_write);
    }

    model_tokens
}

/// Aggregates tool counts from per-key data of existing keys.
/// Falls back to global tool_counts when any existing key lacks per-key breakdowns
/// (mixed legacy + new data).
fn aggregate_tool_counts(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
    plugins: &HashSet<String>,
) -> HashMap<String, u64> {
    let mut result: HashMap<String, u64> = HashMap::new();
    let mut all_have_per_key = true;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            if entry.per_tool.is_empty() {
                all_have_per_key = false;
            }
            for (tool, count) in &entry.per_tool {
                if !is_valid_tool(tool, plugins) {
                    continue;
                }
                *result
                    .entry(canonical_stats_tool(tool).to_string())
                    .or_default() += count;
            }
        }
    }
    if !all_have_per_key {
        let mut global: HashMap<String, u64> = HashMap::new();
        for (tool, count) in &stats.tool_counts {
            if !is_valid_tool(tool, plugins) {
                continue;
            }
            *global
                .entry(canonical_stats_tool(tool).to_string())
                .or_default() += *count;
        }
        return global;
    }
    result
}

/// Aggregates per-model usage from per-key data of existing keys, reading the
/// canonical `per_model_usage` field. Loads of legacy data run through
/// `migrate_legacy_per_model` first, so this function never has to know about
/// the old per-model maps.
///
/// Falls back to the global `model_usage` map (a flat `total_tokens` per model
/// with no input/output split) when no key has per-model data — i.e. an
/// install that predates per-key tracking entirely. The fallback assigns the
/// total to `output` so the row total still renders; the detailed view will
/// show `input=0` for those rows, which is a knowingly-imperfect attribution
/// since the legacy data simply doesn't carry the split.
fn aggregate_model_usage(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
) -> HashMap<String, ModelTotals> {
    let mut result: HashMap<String, ModelTotals> = HashMap::new();
    let mut any_per_key = false;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            if !entry.per_model_usage.is_empty() {
                any_per_key = true;
            }
            for (model, mc) in &entry.per_model_usage {
                let m = result.entry(model.clone()).or_default();
                m.input = m.input.saturating_add(mc.prompt_tokens);
                m.output = m.output.saturating_add(mc.completion_tokens);
                m.cache_read = m.cache_read.saturating_add(mc.cache_read_input_tokens);
                m.cache_write = m.cache_write.saturating_add(mc.cache_creation_input_tokens);
            }
        }
    }
    if !any_per_key {
        return stats
            .model_usage
            .iter()
            .map(|(name, c)| {
                (
                    name.clone(),
                    ModelTotals {
                        input: 0,
                        output: c.total_tokens,
                        cache_read: 0,
                        cache_write: 0,
                    },
                )
            })
            .collect();
    }
    result
}

fn is_valid_tool(tool: &str, plugins: &HashSet<String>) -> bool {
    // `chat` accepted as the pre-rename alias of the built-in agent (`code`).
    AIToolType::parse(tool).is_some() || tool == "code" || tool == "chat" || plugins.contains(tool)
}

/// Collapse `codex-app` into `codex` for stats. Codex Desktop App's shadow
/// `CODEX_HOME` symlinks `sessions/` back to the real `~/.codex/`, so its
/// rollouts pile into the same files the `codex` CLI writes — the token and
/// session data is indistinguishable. Without this, codex-app surfaces a
/// launch-only row (tokens `—`) that also double-counts sessions the `codex`
/// row already includes.
fn canonical_stats_tool(tool: &str) -> &str {
    match tool {
        "codex-app" => "codex",
        // Pre-rename built-in agent bucket folds into `code` so historical
        // token/launch counts stay attributed after the `chat`→`code` rename.
        "chat" => "code",
        other => other,
    }
}

fn resolve_since(arg: Option<&str>) -> Result<Option<chrono::DateTime<chrono::Utc>>, String> {
    match arg {
        None => Ok(None),
        Some(raw) => {
            let dur = crate::services::since::parse_since_duration(raw)?;
            let cutoff = chrono::Utc::now()
                .checked_sub_signed(dur)
                .ok_or_else(|| format!("duration '{raw}' is too large"))?;
            Ok(Some(cutoff))
        }
    }
}

/// Shared line meter, same as `aivo account usage`. Already styled — don't re-wrap.
fn bar(value: u64, max_value: u64) -> String {
    style::meter(value, max_value, style::METER_WIDTH)
}

fn format_number(n: u64) -> String {
    if n < 1000 {
        return n.to_string();
    }
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

pub(crate) fn format_human(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    if n < 1_000_000 {
        let val = n as f64 / 1_000.0;
        return if val < 10.0 {
            format!("{:.1}K", val)
        } else {
            format!("{:.0}K", val)
        };
    }
    if n < 1_000_000_000 {
        let val = n as f64 / 1_000_000.0;
        return if val < 10.0 {
            format!("{:.1}M", val)
        } else {
            format!("{:.0}M", val)
        };
    }
    if n < 1_000_000_000_000 {
        let val = n as f64 / 1_000_000_000.0;
        return if val < 10.0 {
            format!("{:.1}B", val)
        } else {
            format!("{:.0}B", val)
        };
    }
    let val = n as f64 / 1_000_000_000_000.0;
    if val < 10.0 {
        format!("{:.1}T", val)
    } else {
        format!("{:.0}T", val)
    }
}

/// Colorize the unit suffix (K/M/B/T) in an already-padded string.
/// Applied at display time so width calculations use plain text.
pub(crate) fn colorize_unit(s: &str) -> String {
    use console::style as csty;
    for (ch, styler) in [
        ('T', csty("T").bold().magenta().to_string()),
        ('B', csty("B").bold().yellow().to_string()),
        ('M', csty("M").bold().green().to_string()),
        ('K', csty("K").bold().blue().to_string()),
    ] {
        if let Some(pos) = s.rfind(ch) {
            return format!("{}{}", &s[..pos], styler);
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_store::UsageCounter;

    #[test]
    fn format_number_small() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
    }

    #[test]
    fn format_number_with_commas() {
        assert_eq!(format_number(1_000), "1,000");
        assert_eq!(format_number(12_345), "12,345");
        assert_eq!(format_number(1_234_567), "1,234,567");
        assert_eq!(format_number(1_000_000_000), "1,000,000,000");
    }

    #[test]
    fn format_human_small() {
        assert_eq!(format_human(0), "0");
        assert_eq!(format_human(42), "42");
        assert_eq!(format_human(999), "999");
    }

    #[test]
    fn format_human_thousands() {
        assert_eq!(format_human(1_000), "1.0K");
        assert_eq!(format_human(1_500), "1.5K");
        assert_eq!(format_human(9_999), "10.0K");
        assert_eq!(format_human(12_345), "12K");
        assert_eq!(format_human(998_660), "999K");
    }

    #[test]
    fn format_human_millions() {
        assert_eq!(format_human(1_000_000), "1.0M");
        assert_eq!(format_human(1_500_000), "1.5M");
        assert_eq!(format_human(12_345_678), "12M");
    }

    #[test]
    fn format_human_billions() {
        assert_eq!(format_human(1_000_000_000), "1.0B");
        assert_eq!(format_human(2_500_000_000), "2.5B");
        assert_eq!(format_human(15_000_000_000), "15B");
    }

    #[test]
    fn format_human_trillions() {
        assert_eq!(format_human(1_000_000_000_000), "1.0T");
        assert_eq!(format_human(2_500_000_000_000), "2.5T");
        assert_eq!(format_human(15_000_000_000_000), "15T");
    }

    fn meter_cells(s: &str) -> (usize, usize) {
        (
            s.chars().filter(|c| *c == '━').count(),
            s.chars().filter(|c| *c == '─').count(),
        )
    }

    #[test]
    fn bar_is_shared_meter_at_meter_width() {
        let w = crate::style::METER_WIDTH;
        // `bar` delegates to the shared line meter at the shared width.
        assert_eq!(meter_cells(&bar(100, 100)), (w, 0)); // full
        assert_eq!(meter_cells(&bar(0, 100)), (0, w)); // empty rail
        let (f, e) = meter_cells(&bar(50, 100)); // half shows both fill and rail
        assert_eq!(f + e, w);
        assert!(f > 0 && e > 0);
    }

    #[test]
    fn bar_tiny_value_shows_one_tick() {
        // Any non-zero value keeps ≥1 fill cell so it stays visible.
        assert_eq!(
            meter_cells(&bar(1, 1000)),
            (1, crate::style::METER_WIDTH - 1)
        );
    }

    #[test]
    fn valid_tool_names() {
        let none: HashSet<String> = HashSet::new();
        assert!(is_valid_tool("claude", &none));
        assert!(is_valid_tool("codex", &none));
        assert!(is_valid_tool("gemini", &none));
        assert!(is_valid_tool("opencode", &none));
        assert!(is_valid_tool("pi", &none));
        assert!(is_valid_tool("chat", &none));
        assert!(!is_valid_tool("unknown", &none));
        assert!(!is_valid_tool("", &none));
        assert!(is_valid_tool("Claude", &none)); // AIToolType::parse is case-insensitive

        // A coding-agent plugin name is valid only when it's installed.
        let plugins: HashSet<String> = ["omp".to_string()].into_iter().collect();
        assert!(is_valid_tool("omp", &plugins));
        assert!(!is_valid_tool("omp", &none));
    }

    #[test]
    fn aggregate_tool_counts_from_per_key() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("claude".to_string(), 5);
        counter.per_tool.insert("codex".to_string(), 3);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("claude"), Some(&5));
        assert_eq!(result.get("codex"), Some(&3));
    }

    #[test]
    fn aggregate_tool_counts_folds_codex_app_into_codex() {
        // codex-app writes to the same ~/.codex/sessions as the codex CLI, so
        // its launches merge into `codex` instead of forming a separate
        // launch-only row that double-counts shared sessions.
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("codex".to_string(), 2);
        counter.per_tool.insert("codex-app".to_string(), 3);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("codex"), Some(&5));
        assert!(!result.contains_key("codex-app"));
    }

    #[test]
    fn aggregate_tool_counts_folds_codex_app_in_global_fallback() {
        // Legacy install: a key with no per_tool data forces the global
        // fallback, which must also fold codex-app into codex.
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("codex".to_string(), 4);
        stats.tool_counts.insert("codex-app".to_string(), 6);
        stats
            .key_usage
            .insert("key1".to_string(), UsageCounter::default());

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("codex"), Some(&10));
        assert!(!result.contains_key("codex-app"));
    }

    #[test]
    fn aggregate_tool_counts_skips_unsupported_tools() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("claude".to_string(), 5);
        counter.per_tool.insert("omp".to_string(), 3);
        counter.per_tool.insert("cursor".to_string(), 1);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("claude"), Some(&5));
        assert!(!result.contains_key("omp"));
        assert!(!result.contains_key("cursor"));
    }

    #[test]
    fn aggregate_tool_counts_includes_installed_coding_agent_plugin() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("claude".to_string(), 5);
        counter.per_tool.insert("omp".to_string(), 3);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let plugins: HashSet<String> = ["omp".to_string()].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &plugins);
        assert_eq!(result.get("claude"), Some(&5));
        assert_eq!(result.get("omp"), Some(&3));
    }

    #[test]
    fn aggregate_tool_counts_global_fallback_also_filters() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 10);
        stats.tool_counts.insert("omp".to_string(), 5);
        stats
            .key_usage
            .insert("key1".to_string(), UsageCounter::default());

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("claude"), Some(&10));
        assert!(!result.contains_key("omp"));
    }

    #[test]
    fn aggregate_tool_counts_falls_back_to_global() {
        let mut stats = UsageStats::default();
        stats.tool_counts.insert("claude".to_string(), 10);
        // Legacy key exists but has no per_tool data
        stats
            .key_usage
            .insert("key1".to_string(), UsageCounter::default());

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("claude"), Some(&10));
    }

    #[test]
    fn aggregate_model_usage_from_per_key() {
        use crate::services::session_store::ModelCounter;

        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_model_usage.insert(
            "gpt-4o".to_string(),
            ModelCounter {
                prompt_tokens: 700,
                completion_tokens: 300,
                cache_read_input_tokens: 800,
                cache_creation_input_tokens: 200,
            },
        );
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_model_usage(&stats, &keys);
        let m = result.get("gpt-4o").unwrap();
        assert_eq!(m.input, 700);
        assert_eq!(m.output, 300);
        assert_eq!(m.tokens(), 1000);
        assert_eq!(m.cached(), 1000);
        assert_eq!(m.cache_read, 800);
        assert_eq!(m.cache_write, 200);
    }

    #[test]
    fn migration_folds_legacy_residue_into_completion() {
        // Key recorded sessions both before (no split) and after (with split).
        // After load_with_migration runs, residue (legacy total minus split) lands in completion.
        let mut counter = UsageCounter::default();
        counter.per_model_tokens.insert("kimi".to_string(), 450); // 150 legacy + 300 new
        counter
            .per_model_prompt_tokens
            .insert("kimi".to_string(), 200);
        counter
            .per_model_completion_tokens
            .insert("kimi".to_string(), 100);
        counter
            .per_model_cache_read_tokens
            .insert("kimi".to_string(), 25);

        let mut stats = UsageStats::default();
        stats.key_usage.insert("key1".to_string(), counter);
        stats.migrate_legacy_per_model();

        let entry = &stats.key_usage["key1"];
        assert!(entry.per_model_tokens.is_empty());
        assert!(entry.per_model_prompt_tokens.is_empty());
        let mc = &entry.per_model_usage["kimi"];
        assert_eq!(mc.prompt_tokens, 200);
        // 100 (split completion) + 150 (residue) = 250
        assert_eq!(mc.completion_tokens, 250);
        assert_eq!(mc.cache_read_input_tokens, 25);
    }

    #[test]
    fn aggregate_model_usage_falls_back_when_no_per_key_data() {
        let mut stats = UsageStats::default();
        // Legacy install: keys exist but no per-model data anywhere
        let c = UsageCounter::default();
        stats.key_usage.insert("legacy_key".to_string(), c);
        // Global model_usage has the full picture
        let global = UsageCounter {
            total_tokens: 500_000,
            ..Default::default()
        };
        stats.model_usage.insert("gpt-4o".to_string(), global);

        let keys: HashSet<&str> = ["legacy_key"].into_iter().collect();
        let result = aggregate_model_usage(&stats, &keys);
        // Legacy global has no per-model cache or prompt/completion split — it lands in output.
        let m = result.get("gpt-4o").unwrap();
        assert_eq!(m.tokens(), 500_000);
        assert_eq!(m.cached(), 0);
    }

    #[test]
    fn combine_model_tokens_uses_aivo_only_without_global_data() {
        let global = HashMap::new();
        let mut aivo = HashMap::new();
        aivo.insert(
            "gpt-4o".to_string(),
            ModelTotals {
                input: 800,
                output: 434,
                cache_read: 5000,
                cache_write: 678,
            },
        );

        let result = combine_model_tokens(&global, &aivo);
        let m = result.get("gpt-4o").unwrap();
        assert_eq!(m.input, 800);
        assert_eq!(m.output, 434);
        assert_eq!(m.tokens(), 1234);
        assert_eq!(m.cached(), 5678);
    }

    #[test]
    fn combine_model_tokens_skips_aivo_when_native_has_same_model() {
        let mut global = HashMap::new();
        global.insert(
            "codex".to_string(),
            global_stats::GlobalToolStats {
                models: HashMap::from([(
                    "gpt-5.4".to_string(),
                    global_stats::ModelTokens {
                        input_tokens: 100,
                        output_tokens: 25,
                        cache_read_tokens: 50,
                        cache_write_tokens: 0,
                    },
                )]),
                ..Default::default()
            },
        );
        let mut aivo = HashMap::new();
        aivo.insert(
            "gpt-5.4".to_string(),
            ModelTotals {
                input: 400,
                output: 100,
                cache_read: 999,
                cache_write: 0,
            },
        );

        let result = combine_model_tokens(&global, &aivo);
        let m = result.get("gpt-5.4").unwrap();
        assert_eq!(m.input, 100);
        assert_eq!(m.output, 25);
        assert_eq!(m.tokens(), 125);
        assert_eq!(m.cached(), 50);
    }

    #[test]
    fn combine_model_tokens_includes_aivo_only_models_alongside_native() {
        // The kimi-via-aivo-chat case: native has Claude data, aivo has kimi,
        // both should appear in the combined map.
        let mut global = HashMap::new();
        global.insert(
            "claude".to_string(),
            global_stats::GlobalToolStats {
                models: HashMap::from([(
                    "claude-opus".to_string(),
                    global_stats::ModelTokens {
                        input_tokens: 100,
                        output_tokens: 50,
                        cache_read_tokens: 0,
                        cache_write_tokens: 0,
                    },
                )]),
                ..Default::default()
            },
        );
        let mut aivo = HashMap::new();
        aivo.insert(
            "kimi-k2.6".to_string(),
            ModelTotals {
                input: 132,
                output: 19,
                cache_read: 2_500,
                cache_write: 0,
            },
        );

        let result = combine_model_tokens(&global, &aivo);
        assert!(result.contains_key("claude-opus"));
        assert!(result.contains_key("kimi-k2.6"));
        assert_eq!(result.get("kimi-k2.6").unwrap().input, 132);
    }

    #[test]
    fn aggregate_tool_counts_falls_back_when_mixed_legacy_and_new() {
        let mut stats = UsageStats::default();
        // Key with per-tool data (new)
        let mut c1 = UsageCounter::default();
        c1.per_tool.insert("claude".to_string(), 5);
        stats.key_usage.insert("new_key".to_string(), c1);
        // Key without per-tool data (legacy)
        let c2 = UsageCounter::default();
        stats.key_usage.insert("legacy_key".to_string(), c2);
        // Global tool_counts has the full picture
        stats.tool_counts.insert("claude".to_string(), 100);

        let keys: HashSet<&str> = ["new_key", "legacy_key"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        // Should fall back to global since legacy_key lacks per-tool data
        assert_eq!(result.get("claude"), Some(&100));
    }

    #[test]
    fn aggregate_excludes_deleted_keys() {
        let mut stats = UsageStats::default();
        let mut c1 = UsageCounter::default();
        c1.per_tool.insert("claude".to_string(), 5);
        stats.key_usage.insert("key1".to_string(), c1);
        let mut c2 = UsageCounter::default();
        c2.per_tool.insert("claude".to_string(), 3);
        stats.key_usage.insert("deleted_key".to_string(), c2);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys, &HashSet::new());
        assert_eq!(result.get("claude"), Some(&5));
    }

    #[test]
    fn overview_json_includes_window_block_when_since_set() {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(7);
        let tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        let model_tokens: HashMap<String, ModelTotals> = HashMap::new();
        let payload = build_overview_json(
            &tool_tokens,
            &HashSet::new(),
            &model_tokens,
            (0, 0),
            (0, 0),
            0,
            0,
            None,
            Some(("7d", cutoff)),
            &["aivo-proxy"],
        );
        let window = payload.get("window").expect("window block");
        assert_eq!(window.get("since").and_then(|v| v.as_str()), Some("7d"));
        assert!(window.get("since_iso").and_then(|v| v.as_str()).is_some());
        let omitted = payload
            .get("omitted_sources")
            .and_then(|v| v.as_array())
            .unwrap();
        assert!(omitted.iter().any(|v| v == "aivo-proxy"));
    }

    #[test]
    fn overview_json_omits_window_block_when_since_unset() {
        let tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        let model_tokens: HashMap<String, ModelTotals> = HashMap::new();
        let payload = build_overview_json(
            &tool_tokens,
            &HashSet::new(),
            &model_tokens,
            (0, 0),
            (0, 0),
            0,
            0,
            None,
            None,
            &[],
        );
        assert!(payload.get("window").is_none());
        let omitted = payload
            .get("omitted_sources")
            .and_then(|v| v.as_array())
            .expect("omitted_sources array");
        assert!(omitted.is_empty());
    }

    #[test]
    fn overview_json_omits_tokens_for_launch_only_tools() {
        let mut tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        tool_tokens.insert(
            "claude".to_string(),
            ToolTokenSummary {
                sessions: 10,
                input: 100,
                output: 50,
                cache_read: 0,
                cache_write: 0,
            },
        );
        tool_tokens.insert(
            "omp".to_string(),
            ToolTokenSummary {
                sessions: 5,
                input: 0,
                output: 0,
                cache_read: 0,
                cache_write: 0,
            },
        );
        let launch_only: HashSet<String> = ["omp".to_string()].into_iter().collect();
        let model_tokens: HashMap<String, ModelTotals> = HashMap::new();
        let payload = build_overview_json(
            &tool_tokens,
            &launch_only,
            &model_tokens,
            (100, 50),
            (0, 0),
            15,
            0,
            None,
            None,
            &[],
        );
        let by_tool = payload["by_tool"].as_array().unwrap();
        let omp = by_tool.iter().find(|t| t["name"] == "omp").unwrap();
        assert_eq!(omp["per_tool_tokens_tracked"], false);
        assert!(
            omp.get("tokens").is_none(),
            "launch-only tool leaked tokens"
        );
        assert_eq!(omp["sessions"], 5);
        let claude = by_tool.iter().find(|t| t["name"] == "claude").unwrap();
        assert_eq!(claude["per_tool_tokens_tracked"], true);
        assert_eq!(claude["tokens"], 150);
    }

    #[test]
    fn tool_view_json_includes_window_block_when_since_set() {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        let view = ToolView {
            count: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            models: HashMap::new(),
        };
        let args = StatsArgs {
            by: Some("claude".to_string()),
            numbers: false,
            refresh: false,
            search: None,
            all: false,
            detailed: false,
            json: true,
            since: Some("24h".to_string()),
        };
        let payload = build_tool_view_json(
            "claude",
            &view,
            "global",
            "sessions",
            &args,
            Some(("24h", cutoff)),
            &[],
        );
        let window = payload.get("window").expect("window block");
        assert_eq!(window.get("since").and_then(|v| v.as_str()), Some("24h"));
        assert!(window.get("since_iso").and_then(|v| v.as_str()).is_some());
        let omitted = payload
            .get("omitted_sources")
            .and_then(|v| v.as_array())
            .expect("omitted_sources array");
        assert!(omitted.is_empty());
    }

    #[test]
    fn tool_view_json_omits_window_block_when_since_unset() {
        let view = ToolView {
            count: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            models: HashMap::new(),
        };
        let args = StatsArgs {
            by: Some("claude".to_string()),
            numbers: false,
            refresh: false,
            search: None,
            all: false,
            detailed: false,
            json: true,
            since: None,
        };
        let payload = build_tool_view_json("claude", &view, "global", "sessions", &args, None, &[]);
        assert!(payload.get("window").is_none());
        let omitted = payload
            .get("omitted_sources")
            .and_then(|v| v.as_array())
            .expect("omitted_sources array");
        assert!(omitted.is_empty());
    }

    fn launch_args(since: Option<&str>) -> StatsArgs {
        StatsArgs {
            by: Some("omp".to_string()),
            numbers: false,
            refresh: false,
            search: None,
            all: false,
            detailed: false,
            json: true,
            since: since.map(str::to_string),
        }
    }

    #[test]
    fn launch_view_json_reports_launches_not_tokens() {
        let mut counts = HashMap::new();
        counts.insert("aivo/starter".to_string(), 21u64);
        counts.insert("gpt-5.4".to_string(), 4u64);
        let payload = build_launch_view_json("omp", &counts, 25, &launch_args(None), None);

        assert_eq!(payload["source"], "logs");
        assert_eq!(payload["per_tool_tokens_tracked"], false);
        assert_eq!(payload["totals"]["launches"], 25);
        assert_eq!(payload["totals"]["models"], 2);
        // No token fields anywhere — that's the whole point of this view.
        assert!(payload["totals"].get("tokens").is_none());
        let serialized = serde_json::to_string(&payload).unwrap();
        assert!(!serialized.contains("\"tokens\""));
        assert!(!serialized.contains("cache_read"));
        let by_model = payload["by_model"].as_array().unwrap();
        assert_eq!(by_model[0]["name"], "aivo/starter");
        assert_eq!(by_model[0]["launches"], 21);
        assert!(by_model[0].get("tokens").is_none());
        assert!(payload.get("window").is_none());
    }

    #[test]
    fn launch_view_json_includes_window_when_since_set() {
        let counts = HashMap::from([("auto".to_string(), 3u64)]);
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
        let payload =
            build_launch_view_json("omp", &counts, 3, &launch_args(Some("24h")), Some(cutoff));
        let window = payload.get("window").expect("window block");
        assert_eq!(window.get("since").and_then(|v| v.as_str()), Some("24h"));
        assert!(window.get("since_iso").and_then(|v| v.as_str()).is_some());
    }

    #[test]
    fn epoch_utc_is_unix_zero() {
        assert_eq!(epoch_utc().timestamp(), 0);
    }

    #[test]
    fn aggregate_plugin_sessions_windows_host_side() {
        use crate::plugin::stats::{ModelStat, PluginStatsReport, SessionStat};
        let model = |name: &str, out: u64| ModelStat {
            name: name.to_string(),
            output_tokens: out,
            ..Default::default()
        };
        let report = PluginStatsReport {
            source: Some("test".into()),
            sessions: vec![
                SessionStat {
                    ts: Some("2026-06-08T01:00:00Z".into()),
                    models: vec![model("deepseek-v4-flash", 400)],
                },
                SessionStat {
                    ts: Some("2026-01-01T00:00:00Z".into()),
                    models: vec![model("deepseek-v4-flash", 99)],
                },
                // No timestamp → excluded under --since, included lifetime.
                SessionStat {
                    ts: None,
                    models: vec![model("gpt-5.4", 7)],
                },
            ],
        };

        // Lifetime: every session.
        let (count, models) = aggregate_plugin_sessions(&report, None);
        assert_eq!(count, 3);
        assert_eq!(models["deepseek-v4-flash"].output, 499);
        assert_eq!(models["gpt-5.4"].output, 7);

        // --since after the Jan session: only the recent timestamped one.
        let cut: chrono::DateTime<chrono::Utc> = "2026-06-01T00:00:00Z".parse().unwrap();
        let (count, models) = aggregate_plugin_sessions(&report, Some(cut));
        assert_eq!(count, 1);
        assert_eq!(models["deepseek-v4-flash"].output, 400);
        assert!(
            !models.contains_key("gpt-5.4"),
            "untimed session excluded under --since"
        );
    }

    #[test]
    fn per_tool_model_totals_scopes_to_one_tool() {
        use crate::services::session_store::{ModelCounter, UsageCounter};
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool_model_usage.insert(
            "amp".to_string(),
            HashMap::from([(
                "gpt-5.4".to_string(),
                ModelCounter {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    cache_read_input_tokens: 10,
                    cache_creation_input_tokens: 5,
                },
            )]),
        );
        counter.per_tool_model_usage.insert(
            "omp".to_string(),
            HashMap::from([(
                "gpt-5.4".to_string(),
                ModelCounter {
                    prompt_tokens: 7,
                    ..Default::default()
                },
            )]),
        );
        stats.key_usage.insert("k1".to_string(), counter);
        let key_ids: HashSet<&str> = ["k1"].into_iter().collect();

        // amp gets exactly its own usage — omp's tokens do NOT leak in.
        let amp = per_tool_model_totals(&stats, "amp", &key_ids);
        assert_eq!(sum_model_totals(&amp), (100, 50, 10, 5));
        assert_eq!(
            sum_model_totals(&per_tool_model_totals(&stats, "omp", &key_ids)).0,
            7
        );
        // A tool with no recorded usage is empty (→ launch-count fallback).
        assert!(per_tool_model_totals(&stats, "copilot", &key_ids).is_empty());
    }

    #[test]
    fn chat_tokens_for_summary_returns_lifetime_without_cutoff() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("chat".to_string(), 5);
        counter.prompt_tokens = 1_000_000;
        counter.completion_tokens = 200_000;
        counter.cache_read_input_tokens = 50_000;
        counter.cache_creation_input_tokens = 10_000;
        stats.key_usage.insert("k1".to_string(), counter);
        let keys: HashSet<&str> = ["k1"].into_iter().collect();

        let result = chat_tokens_for_summary(&stats, &keys, None, &ChatTokenWindow::default());
        assert_eq!(result.input, 1_000_000);
        assert_eq!(result.output, 200_000);
        assert_eq!(result.cache_read, 50_000);
        assert_eq!(result.cache_write, 10_000);
    }

    #[test]
    fn chat_tokens_for_summary_uses_window_under_cutoff() {
        use crate::services::session_store::SessionTokens;

        // Lifetime per-key counters set high — must be ignored under cutoff.
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("chat".to_string(), 5);
        counter.prompt_tokens = 7_839_935;
        counter.completion_tokens = 176_508;
        stats.key_usage.insert("k1".to_string(), counter);
        let keys: HashSet<&str> = ["k1"].into_iter().collect();

        let mut window = ChatTokenWindow::default();
        window.per_model.insert(
            "minimax-m2.7".to_string(),
            SessionTokens {
                prompt_tokens: 12,
                completion_tokens: 34,
                cache_read_tokens: 5,
                cache_write_tokens: 0,
            },
        );

        let cutoff = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        let result = chat_tokens_for_summary(&stats, &keys, cutoff, &window);
        assert_eq!(result.input, 12);
        assert_eq!(result.output, 34);
        assert_eq!(result.cache_read, 5);
        assert_eq!(result.cache_write, 0);
    }

    #[test]
    fn aivo_model_usage_for_window_returns_lifetime_without_cutoff() {
        use crate::services::session_store::ModelCounter;

        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_model_usage.insert(
            "minimax-m2.7".to_string(),
            ModelCounter {
                prompt_tokens: 0,
                completion_tokens: 6_717_791,
                cache_read_input_tokens: 0,
                cache_creation_input_tokens: 0,
            },
        );
        stats.key_usage.insert("k1".to_string(), counter);
        let keys: HashSet<&str> = ["k1"].into_iter().collect();

        let result = aivo_model_usage_for_window(&stats, &keys, None);
        let m = result.get("minimax-m2.7").expect("lifetime model present");
        assert_eq!(m.output, 6_717_791);
    }

    #[test]
    fn aivo_model_usage_for_window_empty_under_cutoff() {
        // Reproduces the "40 models" footgun: every model ever used through
        // aivo would otherwise leak into every --since view.
        use crate::services::session_store::ModelCounter;

        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        for name in ["minimax-m2.7", "deepseek-chat", "claude-sonnet-4.6"] {
            counter.per_model_usage.insert(
                name.to_string(),
                ModelCounter {
                    prompt_tokens: 100,
                    completion_tokens: 200,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                },
            );
        }
        stats.key_usage.insert("k1".to_string(), counter);
        let keys: HashSet<&str> = ["k1"].into_iter().collect();

        let cutoff = Some(chrono::Utc::now() - chrono::Duration::hours(1));
        let result = aivo_model_usage_for_window(&stats, &keys, cutoff);
        assert!(
            result.is_empty(),
            "lifetime models must not leak under cutoff"
        );
    }

    #[test]
    fn filter_models_keeps_zero_rows_under_window() {
        // Models with zero tokens represent runs we know happened (logged in
        // logs.db) but for which no upstream usage was recorded. Under
        // --since the user explicitly asked "what did I do in this window";
        // dropping these would silently hide the answer.
        let mut models = HashMap::new();
        models.insert("grok-4.3".to_string(), ModelTotals::default());
        models.insert(
            "claude-opus-4-7".to_string(),
            ModelTotals {
                input: 100,
                output: 50,
                cache_read: 0,
                cache_write: 0,
            },
        );

        let windowed = filter_models(&models, None, true);
        let names: Vec<&str> = windowed.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"grok-4.3"));
        assert!(names.contains(&"claude-opus-4-7"));

        let lifetime = filter_models(&models, None, false);
        let names: Vec<&str> = lifetime.iter().map(|(n, _)| n.as_str()).collect();
        assert!(!names.contains(&"grok-4.3"));
        assert!(names.contains(&"claude-opus-4-7"));
    }

    #[test]
    fn overview_json_includes_zero_token_model_under_window() {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        let tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        let mut model_tokens: HashMap<String, ModelTotals> = HashMap::new();
        model_tokens.insert("grok-4.3".to_string(), ModelTotals::default());
        let payload = build_overview_json(
            &tool_tokens,
            &HashSet::new(),
            &model_tokens,
            (0, 0),
            (0, 0),
            0,
            1, // total_models pre-computed by show()
            None,
            Some(("1h", cutoff)),
            &[],
        );
        let by_model = payload
            .get("by_model")
            .and_then(|v| v.as_array())
            .expect("by_model array");
        let names: Vec<&str> = by_model
            .iter()
            .filter_map(|m| m.get("name").and_then(|n| n.as_str()))
            .collect();
        assert!(
            names.contains(&"grok-4.3"),
            "zero-token model must surface under --since, got {names:?}"
        );
    }

    #[test]
    fn overview_json_omits_zero_token_model_without_window() {
        let tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        let mut model_tokens: HashMap<String, ModelTotals> = HashMap::new();
        model_tokens.insert("grok-4.3".to_string(), ModelTotals::default());
        let payload = build_overview_json(
            &tool_tokens,
            &HashSet::new(),
            &model_tokens,
            (0, 0),
            (0, 0),
            0,
            0,
            None,
            None,
            &[],
        );
        let by_model = payload
            .get("by_model")
            .and_then(|v| v.as_array())
            .expect("by_model array");
        assert!(
            by_model.is_empty(),
            "lifetime view must keep its zero-row filter"
        );
    }

    #[test]
    fn parse_since_arg_to_cutoff() {
        use chrono::Utc;
        let now = Utc::now();
        let cutoff = resolve_since(Some("7d")).unwrap().unwrap();
        let delta = (now - cutoff).num_seconds();
        // Allow a few seconds of drift between Utc::now() calls.
        assert!((7 * 86400 - 5..=7 * 86400 + 5).contains(&delta));
        assert!(resolve_since(None).unwrap().is_none());
        assert!(resolve_since(Some("garbage")).is_err());
        // Duration that fits in chrono::Duration but overflows Utc::now() - dur
        // must surface as Err, not panic.
        assert!(resolve_since(Some("99999999w")).is_err());
    }
}
