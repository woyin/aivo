use std::collections::{HashMap, HashSet};

use serde_json::{Value, json};

use crate::cli::StatsArgs;
use crate::errors::ExitCode;
use crate::services::SessionStore;
use crate::services::ai_launcher::AIToolType;
use crate::services::global_stats::{self, NativeSessionSummary, normalize_model_for_display};
use crate::services::session_store::{UsageCounter, UsageStats};
use crate::style;

pub struct StatsCommand {
    store: SessionStore,
}

impl StatsCommand {
    pub fn new(store: SessionStore) -> Self {
        Self { store }
    }

    pub async fn execute(&self, args: StatsArgs) -> ExitCode {
        if let Some(ref tool) = args.tool {
            return self.show_tool(tool, &args).await;
        }
        self.show(&args).await
    }

    async fn show(&self, args: &StatsArgs) -> ExitCode {
        let stats = match self.store.load_stats().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                return ExitCode::UserError;
            }
        };

        let global = global_stats::collect_all(args.refresh).await;

        if stats.is_empty() && global.is_empty() {
            if args.json {
                return print_json(&empty_overview_json());
            }
            println!("{}", style::dim("No usage stats recorded yet."));
            return ExitCode::Success;
        }

        let keys = self.store.get_keys().await.unwrap_or_default();
        let fmt = if args.numbers {
            format_number
        } else {
            format_human
        };

        let key_ids: HashSet<&str> = keys.iter().map(|k| k.id.as_str()).collect();

        let mut tool_tokens: HashMap<String, ToolTokenSummary> = HashMap::new();
        let aivo_tool_counts = aggregate_tool_counts(&stats, &key_ids);
        for (tool, gs) in &global {
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
        // When global stats have 0 tokens for a tool, fall back to aivo-tracked data
        for (tool, &count) in &aivo_tool_counts {
            if tool == "chat" || count == 0 {
                continue;
            }
            let dominated_by_global = tool_tokens.get(tool).is_some_and(|t| t.total_tokens() > 0);
            if !dominated_by_global {
                let mut aivo = tool_token_totals(&stats, tool, &key_ids);
                aivo.sessions = count;
                tool_tokens.insert(tool.clone(), aivo);
            }
        }
        // Add chat — use actual session file count, not record_selection count
        // (record_selection is called on every model/key switch, inflating the count)
        let chat_sessions = self.store.count_chat_sessions().await;
        let chat_tokens = tool_token_totals(&stats, "chat", &key_ids);
        if chat_tokens.total_tokens() > 0 || chat_sessions > 0 {
            tool_tokens.insert(
                "chat".to_string(),
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

        let aivo_model_usage = aggregate_model_usage(&stats, &key_ids);
        let model_tokens = combine_model_tokens(&global, &aivo_model_usage);
        let total_models = model_tokens.values().filter(|t| **t > 0).count() as u64;

        if args.json {
            return print_json(&build_overview_json(
                &tool_tokens,
                &model_tokens,
                (total_input, total_output),
                (total_cache_read, total_cache_write),
                total_sessions,
                total_models,
                args.search.as_deref(),
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
        println!(
            "{}",
            style::dim("─".repeat(console::measure_text_width(&header)))
        );
        println!("{}", style::bold(header));

        if !tool_tokens.is_empty() {
            println!();
            let mut rows: Vec<(&str, u64, u64)> = tool_tokens
                .iter()
                .map(|(name, t)| (name.as_str(), t.sessions, t.total_tokens()))
                .collect();
            rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| b.1.cmp(&a.1)));

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
                .map(|(_, _, t)| fmt(*t).len())
                .max()
                .unwrap_or(0)
                .max("tokens".len());
            let max_tok = rows.iter().map(|(_, _, t)| *t).max().unwrap_or(0);

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
                let pt = colorize_unit(&format!("{:>width$}", fmt(*tok), width = tok_w));
                if show_tool_bar {
                    println!(
                        "{} {} {} {}",
                        style::cyan(&pn),
                        ps,
                        pt,
                        style::cyan(bar(*tok, max_tok)),
                    );
                } else {
                    println!("{} {} {}", style::cyan(&pn), ps, pt);
                }
            }
        }

        render_model_table(&model_tokens, fmt, args);

        ExitCode::Success
    }

    async fn show_tool(&self, tool: &str, args: &StatsArgs) -> ExitCode {
        let tool = tool.to_lowercase();
        if !is_valid_tool(&tool) {
            eprintln!(
                "{} Unknown tool '{}'. Valid tools: claude, codex, gemini, opencode, pi, chat",
                style::red("Error:"),
                tool
            );
            return ExitCode::UserError;
        }

        let fmt = if args.numbers {
            format_number
        } else {
            format_human
        };

        let global = match global_stats::collect(&tool, args.refresh).await {
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
        let aivo = self.get_aivo_tool_stats(&tool).await;
        let has_global = global
            .as_ref()
            .is_some_and(|gs| gs.total_tokens() > 0 || gs.sessions > 0);

        if !has_global && aivo.launches == 0 {
            if args.json {
                return print_json(&empty_tool_view_json(&tool));
            }
            println!(
                "{}",
                style::dim(format!(
                    "No stats found for {}.",
                    global_stats::tool_display_name(&tool)
                ))
            );
            return ExitCode::Success;
        }

        let (view, top_sessions) = if let Some(gs) = global
            .as_ref()
            .filter(|gs| gs.total_tokens() > 0 || gs.sessions > 0)
        {
            let view = ToolView {
                source: StatsSource::Global,
                count: gs.sessions,
                input_tokens: gs.input_tokens,
                output_tokens: gs.output_tokens,
                cache_read: gs.cache_read_tokens,
                cache_write: gs.cache_write_tokens,
                models: gs
                    .models
                    .iter()
                    .map(|(name, m)| (name.clone(), m.input_tokens + m.output_tokens))
                    .collect(),
            };
            let top = if args.top_sessions && matches!(tool.as_str(), "codex" | "claude" | "gemini")
            {
                match global_stats::top_sessions(&tool, args.refresh, 10).await {
                    Ok(sessions) => Some(sessions),
                    Err(e) => {
                        eprintln!(
                            "{} Failed to read {} sessions: {}",
                            style::red("Error:"),
                            global_stats::tool_display_name(&tool),
                            e
                        );
                        None
                    }
                }
            } else {
                None
            };
            (view, top)
        } else {
            let view = ToolView {
                source: StatsSource::Aivo,
                count: aivo.launches,
                input_tokens: aivo.prompt_tokens,
                output_tokens: aivo.completion_tokens,
                cache_read: aivo.cache_read,
                cache_write: aivo.cache_write,
                models: aivo.models,
            };
            (view, None)
        };

        if args.json {
            return print_json(&build_tool_view_json(
                &tool,
                &view,
                top_sessions.as_deref(),
                args,
            ));
        }

        print_tool_view(&view, fmt, args);
        if let Some(sessions) = top_sessions {
            render_top_sessions(&sessions, fmt);
        }

        ExitCode::Success
    }

    async fn get_aivo_tool_stats(&self, tool: &str) -> AivoToolStats {
        let stats = match self.store.load_stats().await {
            Ok(s) => s,
            Err(_) => return AivoToolStats::default(),
        };
        let keys = self.store.get_keys().await.unwrap_or_default();
        let key_ids: HashSet<&str> = keys.iter().map(|k| k.id.as_str()).collect();
        let tool_counts = aggregate_tool_counts(&stats, &key_ids);
        let launches = tool_counts.get(tool).copied().unwrap_or(0);
        let totals = tool_token_totals(&stats, tool, &key_ids);

        let mut models: HashMap<String, u64> = HashMap::new();
        for (key_id, entry) in &stats.key_usage {
            if !key_ids.contains(key_id.as_str()) {
                continue;
            }
            if entry.per_tool.get(tool).copied().unwrap_or(0) == 0 {
                continue;
            }
            for (model, &tok) in &entry.per_model_tokens {
                let key = normalize_model_for_display(model);
                *models.entry(key).or_default() += tok;
            }
        }

        AivoToolStats {
            launches,
            prompt_tokens: totals.input,
            completion_tokens: totals.output,
            cache_read: totals.cache_read,
            cache_write: totals.cache_write,
            models,
        }
    }

    pub fn print_help() {
        println!("{} aivo stats [tool] [options]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Show usage statistics: token counts, request counts, and breakdowns.")
        );
        println!();
        println!("{}", style::bold("Arguments:"));
        println!(
            "  {}{}",
            style::cyan(format!("{:<26}", "[tool]")),
            style::dim(
                "Show stats for a specific tool (claude, codex, gemini, opencode, pi, chat)"
            )
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
        print_opt("-n, --numbers", "Exact numbers instead of human-readable");
        print_opt("-r, --refresh", "Bypass cache and re-read all data files");
        print_opt("-s, --search <QUERY>", "Search by key, model, or tool name");
        print_opt("-a, --all", "Show all models (default: top 20)");
        print_opt("--top-sessions", "Show the heaviest native session files");
        print_opt("--json", "Output stats as JSON (all models, exact numbers)");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo stats"));
        println!("  {}", style::dim("aivo stats claude"));
        println!("  {}", style::dim("aivo stats claude -n"));
        println!("  {}", style::dim("aivo stats -n"));
        println!("  {}", style::dim("aivo stats -s openrouter"));
        println!(
            "  {}",
            style::dim("aivo stats --json | jq '.totals.tokens'")
        );
    }
}

fn filter_models<'a>(
    models: impl IntoIterator<Item = (&'a String, &'a u64)>,
    search: Option<&str>,
) -> Vec<(String, u64)> {
    let needle = search.map(|s| s.to_lowercase());
    let mut rows: Vec<(String, u64)> = models
        .into_iter()
        .filter(|(_, tok)| **tok > 0)
        .filter(|(name, _)| {
            needle
                .as_ref()
                .is_none_or(|q| name.to_lowercase().contains(q))
        })
        .map(|(name, tok)| (name.clone(), *tok))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
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

fn build_overview_json(
    tool_tokens: &HashMap<String, ToolTokenSummary>,
    model_tokens: &HashMap<String, u64>,
    (total_input, total_output): (u64, u64),
    (total_cache_read, total_cache_write): (u64, u64),
    total_sessions: u64,
    total_models: u64,
    search: Option<&str>,
) -> Value {
    let mut tool_rows: Vec<(&String, &ToolTokenSummary)> = tool_tokens.iter().collect();
    tool_rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_tokens()));
    let by_tool: Vec<Value> = tool_rows
        .into_iter()
        .map(|(name, t)| {
            json!({
                "name": name,
                "sessions": t.sessions,
                "tokens": t.total_tokens(),
                "input_tokens": t.input,
                "output_tokens": t.output,
                "cache_read_tokens": t.cache_read,
                "cache_write_tokens": t.cache_write,
            })
        })
        .collect();

    let by_model: Vec<Value> = filter_models(model_tokens, search)
        .into_iter()
        .map(|(name, tok)| json!({ "name": name, "tokens": tok }))
        .collect();

    json!({
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
    })
}

fn build_tool_view_json(
    tool: &str,
    view: &ToolView,
    top_sessions: Option<&[NativeSessionSummary]>,
    args: &StatsArgs,
) -> Value {
    let source = match view.source {
        StatsSource::Global => "global",
        StatsSource::Aivo => "aivo",
    };
    let by_model: Vec<Value> = filter_models(&view.models, args.search.as_deref())
        .into_iter()
        .map(|(name, tok)| json!({ "name": name, "tokens": tok }))
        .collect();
    // Match the human view's count: total models with any tokens, ignoring search filter.
    let total_models = view.models.values().filter(|t| **t > 0).count() as u64;
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
            "models": total_models,
        },
        "by_model": by_model,
    });
    // Always emit `top_sessions` when --top-sessions was requested, so consumers
    // see a stable schema (empty array means "tool doesn't expose native sessions").
    if args.top_sessions {
        let sessions_json: Vec<Value> = top_sessions
            .unwrap_or(&[])
            .iter()
            .map(|s| {
                json!({
                    "path": s.path.display().to_string(),
                    "label": session_label(s),
                    "model": s.model,
                    "tokens": s.total_tokens(),
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cache_read_tokens": s.cache_read_tokens,
                    "cache_write_tokens": s.cache_write_tokens,
                })
            })
            .collect();
        payload["top_sessions"] = Value::Array(sessions_json);
    }
    payload
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

fn empty_tool_view_json(tool: &str) -> Value {
    json!({
        "tool": tool,
        "source": Value::Null,
        "totals": {
            "tokens": 0u64,
            "input_tokens": 0u64,
            "output_tokens": 0u64,
            "cache_tokens": 0u64,
            "cache_read_tokens": 0u64,
            "cache_write_tokens": 0u64,
            "count": 0u64,
            "models": 0u64,
        },
        "by_model": Vec::<Value>::new(),
    })
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
        if entry.per_tool.get(tool).copied().unwrap_or(0) == 0 {
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

#[derive(Copy, Clone)]
enum StatsSource {
    Global,
    Aivo,
}

struct ToolView {
    source: StatsSource,
    count: u64,
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    models: HashMap<String, u64>,
}

#[derive(Default)]
struct AivoToolStats {
    launches: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    models: HashMap<String, u64>,
}

fn print_tool_view(view: &ToolView, fmt: fn(u64) -> String, args: &StatsArgs) {
    let total_tokens = view.input_tokens.saturating_add(view.output_tokens);
    let total_cache = view.cache_read.saturating_add(view.cache_write);
    let model_count = view.models.values().filter(|t| **t > 0).count() as u64;

    let count_label = match view.source {
        StatsSource::Global => "sessions",
        StatsSource::Aivo => "launches",
    };

    let mut parts = Vec::new();
    if total_tokens > 0 {
        parts.push(format!("{} tokens", colorize_unit(&fmt(total_tokens))));
    }
    if total_cache > 0 {
        parts.push(format!("{} cached", colorize_unit(&fmt(total_cache))));
    }
    parts.push(format!(
        "{} {}",
        colorize_unit(&fmt(view.count)),
        count_label
    ));
    parts.push(format!("{} models", colorize_unit(&fmt(model_count))));

    let header = parts.join(" · ");
    println!(
        "{}",
        style::dim("─".repeat(console::measure_text_width(&header)))
    );
    println!("{}", style::bold(header));

    render_model_table(&view.models, fmt, args);
}

fn render_model_table(models: &HashMap<String, u64>, fmt: fn(u64) -> String, args: &StatsArgs) {
    let searching = args.search.is_some();
    let model_rows = filter_models(models, args.search.as_deref());

    if model_rows.is_empty() {
        return;
    }

    println!();

    let total_model_count = model_rows.len();
    let max_display = 20;
    let truncated = !args.all && total_model_count > max_display;

    let display_rows: Vec<(String, u64)> = if truncated {
        let others_count = total_model_count - max_display;
        let others_tokens: u64 = model_rows[max_display..].iter().map(|(_, t)| *t).sum();
        let mut rows: Vec<(String, u64)> = model_rows[..max_display].to_vec();
        rows.push((format!("others ({} models)", others_count), others_tokens));
        rows
    } else {
        model_rows
    };

    let name_w = display_rows
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("By model".len());
    let tok_w = display_rows
        .iter()
        .map(|(_, t)| fmt(*t).len())
        .max()
        .unwrap_or(0)
        .max("tokens".len());
    let max_tok = display_rows.iter().map(|(_, t)| *t).max().unwrap_or(0);

    println!(
        "{} {}",
        style::bold(format!("{:<name_w$}", "By model")),
        style::dim(format!("{:>tok_w$}", "tokens")),
    );

    let show_bar = !searching && display_rows.len() > 1;
    for (name, tok) in &display_rows {
        let pn = format!("{:<width$}", name, width = name_w);
        let pt = colorize_unit(&format!("{:>width$}", fmt(*tok), width = tok_w));
        if show_bar {
            println!(
                "{} {} {}",
                style::cyan(&pn),
                pt,
                style::cyan(bar(*tok, max_tok)),
            );
        } else {
            println!("{} {}", style::cyan(&pn), pt);
        }
    }

    println!();
    let mut hints = Vec::new();
    if truncated {
        hints.push("-a all models".to_string());
    }
    hints.push("-n numbers".to_string());
    hints.push("-r refresh".to_string());
    hints.push("-s filter".to_string());
    println!("{}", style::dim(hints.join(" · ")));
}

fn render_top_sessions(sessions: &[NativeSessionSummary], fmt: fn(u64) -> String) {
    if sessions.is_empty() {
        return;
    }

    println!();
    println!("{}", style::bold("Top sessions"));

    let labels: Vec<String> = sessions.iter().map(session_label).collect();
    let label_w = labels
        .iter()
        .map(|label| label.len())
        .max()
        .unwrap_or(0)
        .max("session".len());
    let tok_w = sessions
        .iter()
        .map(|s| fmt(s.total_tokens()).len())
        .max()
        .unwrap_or(0)
        .max("tokens".len());
    let cache_w = sessions
        .iter()
        .map(|s| fmt(s.cache_read_tokens.saturating_add(s.cache_write_tokens)).len())
        .max()
        .unwrap_or(0)
        .max("cached".len());

    println!(
        "{} {} {} {}",
        style::bold(format!("{:<label_w$}", "session")),
        style::dim(format!("{:>tok_w$}", "tokens")),
        style::dim(format!("{:>cache_w$}", "cached")),
        style::dim("model"),
    );

    for (summary, label) in sessions.iter().zip(labels.iter()) {
        let total = colorize_unit(&format!(
            "{:>width$}",
            fmt(summary.total_tokens()),
            width = tok_w
        ));
        let cached = colorize_unit(&format!(
            "{:>width$}",
            fmt(summary
                .cache_read_tokens
                .saturating_add(summary.cache_write_tokens)),
            width = cache_w
        ));
        let model = summary.model.as_deref().unwrap_or("-");
        println!(
            "{} {} {} {}",
            style::cyan(format!("{:<width$}", label, width = label_w)),
            total,
            cached,
            style::dim(model),
        );
    }
}

fn session_label(summary: &NativeSessionSummary) -> String {
    let path = summary.path.to_string_lossy();
    if let Some(pos) = path.find("/sessions/") {
        let relative = &path[pos + "/sessions/".len()..];
        return relative
            .strip_suffix(".jsonl")
            .unwrap_or(relative)
            .to_string();
    }
    let label = summary
        .path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string();
    label.strip_suffix(".jsonl").unwrap_or(&label).to_string()
}

/// Native tool stats and aivo model stats overlap for launched tools, but the
/// aivo store does not keep per-tool token attribution. To avoid inflated
/// totals, prefer native tool models whenever any native data exists.
fn combine_model_tokens(
    global: &HashMap<String, global_stats::GlobalToolStats>,
    aivo_model_usage: &HashMap<String, UsageCounter>,
) -> HashMap<String, u64> {
    let mut model_tokens: HashMap<String, u64> = HashMap::new();
    for gs in global.values() {
        for (model, mt) in &gs.models {
            let key = normalize_model_for_display(model);
            *model_tokens.entry(key).or_default() +=
                mt.input_tokens.saturating_add(mt.output_tokens);
        }
    }

    if model_tokens.is_empty() {
        for (model, counter) in aivo_model_usage {
            let key = normalize_model_for_display(model);
            *model_tokens.entry(key).or_default() += counter.total_tokens;
        }
    }

    model_tokens
}

/// Aggregates tool counts from per-key data of existing keys.
/// Falls back to global tool_counts when any existing key lacks per-key breakdowns
/// (mixed legacy + new data).
fn aggregate_tool_counts(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
) -> HashMap<String, u64> {
    let mut result: HashMap<String, u64> = HashMap::new();
    let mut all_have_per_key = true;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            if entry.per_tool.is_empty() {
                all_have_per_key = false;
            }
            for (tool, count) in &entry.per_tool {
                *result.entry(tool.clone()).or_default() += count;
            }
        }
    }
    if !all_have_per_key {
        return stats.tool_counts.clone();
    }
    result
}

/// Aggregates model usage from per-key data of existing keys.
/// Falls back to global model_usage when any existing key lacks per-key breakdowns
/// (mixed legacy + new data).
fn aggregate_model_usage(
    stats: &UsageStats,
    existing_keys: &HashSet<&str>,
) -> HashMap<String, UsageCounter> {
    let mut result: HashMap<String, UsageCounter> = HashMap::new();
    let mut all_have_per_key = true;
    for (key_id, entry) in &stats.key_usage {
        if existing_keys.contains(key_id.as_str()) {
            if entry.per_model_tokens.is_empty() {
                all_have_per_key = false;
            }
            for (model, tok) in &entry.per_model_tokens {
                result.entry(model.clone()).or_default().total_tokens += tok;
            }
        }
    }
    if !all_have_per_key {
        return stats.model_usage.clone();
    }
    result
}

fn is_valid_tool(tool: &str) -> bool {
    AIToolType::parse(tool).is_some() || tool == "chat"
}

const BAR_MAX: usize = 20;

fn bar(value: u64, max_value: u64) -> String {
    style::bar(value, max_value, BAR_MAX)
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

fn format_human(n: u64) -> String {
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
fn colorize_unit(s: &str) -> String {
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

    #[test]
    fn bar_proportional() {
        assert_eq!(bar(100, 100), "████████████████████");
        assert_eq!(bar(50, 100), "██████████");
        assert_eq!(bar(0, 100), "");
        assert_eq!(bar(0, 0), "");
    }

    #[test]
    fn bar_small_value_shows_sliver() {
        let b = bar(1, 1000);
        assert!(!b.is_empty());
        assert!(b.len() <= 4);
    }

    #[test]
    fn valid_tool_names() {
        assert!(is_valid_tool("claude"));
        assert!(is_valid_tool("codex"));
        assert!(is_valid_tool("gemini"));
        assert!(is_valid_tool("opencode"));
        assert!(is_valid_tool("pi"));
        assert!(is_valid_tool("chat"));
        assert!(!is_valid_tool("unknown"));
        assert!(!is_valid_tool(""));
        assert!(is_valid_tool("Claude")); // AIToolType::parse is case-insensitive
    }

    #[test]
    fn aggregate_tool_counts_from_per_key() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_tool.insert("claude".to_string(), 5);
        counter.per_tool.insert("codex".to_string(), 3);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&5));
        assert_eq!(result.get("codex"), Some(&3));
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
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&10));
    }

    #[test]
    fn aggregate_model_usage_from_per_key() {
        let mut stats = UsageStats::default();
        let mut counter = UsageCounter::default();
        counter.per_model_tokens.insert("gpt-4o".to_string(), 1000);
        stats.key_usage.insert("key1".to_string(), counter);

        let keys: HashSet<&str> = ["key1"].into_iter().collect();
        let result = aggregate_model_usage(&stats, &keys);
        assert_eq!(result.get("gpt-4o").unwrap().total_tokens, 1000);
    }

    #[test]
    fn aggregate_model_usage_falls_back_when_mixed_legacy_and_new() {
        let mut stats = UsageStats::default();
        // Key with per-model data (new)
        let mut c1 = UsageCounter::default();
        c1.per_model_tokens.insert("gpt-4o".to_string(), 1000);
        stats.key_usage.insert("new_key".to_string(), c1);
        // Key without per-model data (legacy)
        let c2 = UsageCounter::default();
        stats.key_usage.insert("legacy_key".to_string(), c2);
        // Global model_usage has the full picture
        let global = UsageCounter {
            total_tokens: 500_000,
            ..Default::default()
        };
        stats.model_usage.insert("gpt-4o".to_string(), global);

        let keys: HashSet<&str> = ["new_key", "legacy_key"].into_iter().collect();
        let result = aggregate_model_usage(&stats, &keys);
        // Should fall back to global since legacy_key lacks per-model data
        assert_eq!(result.get("gpt-4o").unwrap().total_tokens, 500_000);
    }

    #[test]
    fn combine_model_tokens_uses_aivo_only_without_global_data() {
        let global = HashMap::new();
        let mut aivo = HashMap::new();
        aivo.insert(
            "gpt-4o".to_string(),
            UsageCounter {
                total_tokens: 1234,
                ..Default::default()
            },
        );

        let result = combine_model_tokens(&global, &aivo);
        assert_eq!(result.get("gpt-4o"), Some(&1234));
    }

    #[test]
    fn combine_model_tokens_ignores_overlapping_aivo_data_when_global_exists() {
        let mut global = HashMap::new();
        global.insert(
            "codex".to_string(),
            global_stats::GlobalToolStats {
                models: HashMap::from([(
                    "gpt-5.4".to_string(),
                    global_stats::ModelTokens {
                        input_tokens: 100,
                        output_tokens: 25,
                    },
                )]),
                ..Default::default()
            },
        );
        let mut aivo = HashMap::new();
        aivo.insert(
            "gpt-5.4".to_string(),
            UsageCounter {
                total_tokens: 500,
                ..Default::default()
            },
        );

        let result = combine_model_tokens(&global, &aivo);
        assert_eq!(result.get("gpt-5.4"), Some(&125));
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
        let result = aggregate_tool_counts(&stats, &keys);
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
        let result = aggregate_tool_counts(&stats, &keys);
        assert_eq!(result.get("claude"), Some(&5));
    }
}
