use std::path::Path;
use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::commands::code::ChatMessage;
use crate::commands::code_response_parser::TokenUsage;

/// Formats a live elapsed clock for the in-stream status line, scaling the
/// units up as the wait grows so a long turn reads `12m 50s` / `1h 23m` /
/// `2d 3h` instead of an unwieldy raw second count. Seconds are kept at the
/// minute scale (the clock is ticking) but dropped at the hour/day scale to
/// stay compact.
pub(super) fn format_request_elapsed(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    let (days, hours, minutes, seconds) = (
        secs / 86_400,
        (secs % 86_400) / 3_600,
        (secs % 3_600) / 60,
        secs % 60,
    );
    if days > 0 {
        format!("{days}d {hours}h")
    } else if hours > 0 {
        format!("{hours}h {minutes}m")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

pub(super) fn format_token_count(tokens: u64, usage: Option<TokenUsage>) -> String {
    if let Some(usage) = usage {
        let total = usage.total_tokens();
        let label = if total == 1 { "token" } else { "tokens" };
        return format!("{} {}", format_token_count_value(total), label);
    }
    if tokens == 0 {
        "0 tokens".to_string()
    } else {
        let label = if tokens == 1 { "token" } else { "tokens" };
        format!("~{} {}", format_token_count_value(tokens), label)
    }
}

/// One decimal under 10 tok/s, whole numbers above.
pub(super) fn format_tps(tps: f64) -> String {
    if tps < 10.0 {
        format!("{tps:.1} tok/s")
    } else {
        format!("{} tok/s", tps.round() as u64)
    }
}

/// USD figure: two decimals from a cent up, else two significant digits so a
/// fraction-of-a-cent turn still shows.
pub(super) fn format_usd(usd: f64) -> String {
    if usd.is_nan() || usd <= 0.0 || usd >= 0.01 {
        return format!("{:.2}", usd.max(0.0));
    }
    let decimals = ((-usd.log10()).floor() as usize + 2).min(8);
    format!("{usd:.decimals$}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

pub(super) fn format_token_count_value(tokens: u64) -> String {
    if tokens < 1_000 {
        return tokens.to_string();
    }
    // k below ~1M, then M (so a 1,000,000-token window reads "1M", not "1000k").
    // The cutoff sits just under 1M so values that would round up to "1000k"
    // render as "1M" instead.
    if tokens < 999_950 {
        format_scaled(tokens, 1_000, 'k')
    } else {
        format_scaled(tokens, 1_000_000, 'M')
    }
}

/// Format `tokens` as a value in `unit`s with one optional decimal place,
/// rounded to the nearest tenth of a unit (e.g. 1_500_000 in M → "1.5M",
/// 200_000 in k → "200k").
fn format_scaled(tokens: u64, unit: u64, suffix: char) -> String {
    let rounded_tenths = (tokens + unit / 20) / (unit / 10);
    let whole = rounded_tenths / 10;
    let tenths = rounded_tenths % 10;
    if tenths == 0 {
        format!("{whole}{suffix}")
    } else {
        format!("{whole}.{tenths}{suffix}")
    }
}

const ATTACHMENT_OVERHEAD_TOKENS: usize = 16;
const MESSAGE_OVERHEAD_TOKENS: usize = 5;

pub(super) fn estimate_context_tokens(history: &[ChatMessage]) -> u64 {
    use crate::agent::tokens::estimate_str_tokens;
    let total: usize = history
        .iter()
        .map(|m| {
            let attachment_tokens = m
                .attachments
                .iter()
                .map(|a| estimate_str_tokens(&a.name) + ATTACHMENT_OVERHEAD_TOKENS)
                .sum::<usize>();
            estimate_str_tokens(&m.content) + attachment_tokens + MESSAGE_OVERHEAD_TOKENS
        })
        .sum();
    total as u64
}

/// The engine cluster's identity: the model label plus (when it adds anything)
/// the key/host segment. A key named after the model or its `owner/` prefix,
/// or an `hf:owner/repo` ref ending in the model, collapses to one label —
/// the model id alone already carries the provider.
pub(super) fn footer_engine_labels(
    model: &str,
    base_url: &str,
    key_name: &str,
) -> (String, Option<String>) {
    // Prefer the user's key name; fall back to the provider host from the URL.
    let host = if key_name.trim().is_empty() {
        footer_host_label(base_url)
    } else {
        key_name.trim().to_string()
    };
    if host.is_empty() {
        return (model.to_string(), None);
    }
    // Local HF model: key name `hf:owner/repo` already ends in the model, so
    // lead with the ref (which subsumes the model name).
    if host
        .strip_prefix("hf:")
        .is_some_and(|repo| repo.rsplit('/').next() == Some(model))
    {
        return (host, None);
    }
    let is_redundant_key = host.eq_ignore_ascii_case(model)
        || model
            .split_once('/')
            .is_some_and(|(owner, _)| owner.eq_ignore_ascii_case(&host));
    if is_redundant_key {
        return (model.to_string(), None);
    }
    (model.to_string(), Some(host))
}

/// The workspace cluster's cwd text, best-first: the full (home-abbreviated)
/// path with the git branch, the basename with the branch, then the bare
/// basename as the width tightens.
pub(super) fn footer_workspace_candidates(cwd: &str, branch: Option<&str>) -> [String; 3] {
    let cwd_full = footer_cwd_label(cwd);
    let cwd_base = footer_cwd_basename(cwd);
    let branch_suffix = branch
        .filter(|b| !b.is_empty())
        .map(|b| format!(" ({b})"))
        .unwrap_or_default();
    [
        format!("{cwd_full}{branch_suffix}"),
        format!("{cwd_base}{branch_suffix}"),
        cwd_base,
    ]
}

/// The footer's session-id handle (the full id lives in the `/session` overlay):
/// a fork keeps its source tag over an 8-char grip (`pi·019f15e1`), a native id
/// shows its `#`-prefixed prefix. Clicking it opens the overlay.
pub(super) fn footer_session_label(session_id: &str) -> String {
    if let Some((cli, id)) = crate::services::session_import::split_fork_id(session_id) {
        let short: String = id.chars().take(8).collect();
        return format!("{cli}·{short}");
    }
    format!("#{}", session_id.chars().take(8).collect::<String>())
}

pub(super) fn footer_host_label(base_url: &str) -> String {
    if base_url == "copilot" {
        return "copilot".to_string();
    }

    let trimmed = base_url.trim().trim_end_matches('/');
    let without_scheme = trimmed
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(trimmed);
    without_scheme
        .split('/')
        .next()
        .filter(|host| !host.is_empty())
        .unwrap_or(trimmed)
        .to_string()
}

/// The working directory abbreviated with `~` for the home dir, e.g.
/// `~/project/work/aivo` or `/private/tmp/hi`.
fn footer_cwd_label(cwd: &str) -> String {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    if !home.is_empty() {
        if cwd == home {
            return "~".to_string();
        }
        if let Some(rest) = cwd.strip_prefix(&format!("{home}/")) {
            return format!("~/{rest}");
        }
    }
    cwd.to_string()
}

/// Just the final path component (width fallback when the full path won't fit).
fn footer_cwd_basename(cwd: &str) -> String {
    Path::new(cwd)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(cwd)
        .to_string()
}

/// The current git branch for `dir` (walking up to the repo root), or `None`
/// when `dir` isn't inside a git work tree. Reads `.git/HEAD` directly — no
/// subprocess — so it's cheap enough to poll on the footer's refresh throttle.
/// A detached HEAD yields the short commit hash. An empty `dir` is rejected up
/// front so a relative `.git` lookup can't latch onto the process's own repo.
pub(super) fn git_branch_for(dir: &str) -> Option<String> {
    if dir.is_empty() {
        return None;
    }
    let mut cur = Path::new(dir);
    loop {
        let dot_git = cur.join(".git");
        if dot_git.is_dir() {
            return read_head_branch(&dot_git);
        }
        if dot_git.is_file() {
            // A linked worktree / submodule: `.git` is a file `gitdir: <path>`.
            let contents = std::fs::read_to_string(&dot_git).ok()?;
            let target = contents.strip_prefix("gitdir:")?.trim();
            let git_dir = if Path::new(target).is_absolute() {
                std::path::PathBuf::from(target)
            } else {
                cur.join(target)
            };
            return read_head_branch(&git_dir);
        }
        cur = cur.parent()?;
    }
}

/// Parse the branch from a git dir's `HEAD`: `ref: refs/heads/<branch>` → the
/// branch; a raw commit hash (detached HEAD) → its short form.
fn read_head_branch(git_dir: &Path) -> Option<String> {
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    if let Some(branch) = head.strip_prefix("ref: refs/heads/") {
        return (!branch.is_empty()).then(|| branch.to_string());
    }
    if head.len() >= 7 && head.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(head[..7].to_string());
    }
    None
}

pub(super) fn format_session_group_label(updated_at: &str) -> String {
    let parsed = DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Local))
        .ok();
    let Some(parsed) = parsed else {
        return updated_at.to_string();
    };
    let today = Local::now().date_naive();
    if parsed.date_naive() == today {
        "Today".to_string()
    } else {
        parsed.format("%a %b %d %Y").to_string()
    }
}

pub(super) fn format_session_time(updated_at: &str) -> String {
    DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Local).format("%-I:%M %p").to_string())
        .unwrap_or_else(|_| updated_at.to_string())
}

pub(super) fn format_session_match_count(filtered: usize, total: usize) -> String {
    if total == 0 {
        return "0 sessions".to_string();
    }
    if filtered == total {
        return format!("{total} sessions");
    }
    format!("{filtered}/{total}")
}

pub(super) fn format_picker_match_count(filtered: usize, total: usize, noun: &str) -> String {
    if total == 0 {
        return format!("0 {noun}");
    }
    if filtered == total {
        return format!("{total} {noun}");
    }
    format!("{filtered}/{total}")
}

pub(super) fn display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

pub(super) fn truncate_for_display_width(text: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }
    if display_width(text) <= max_width {
        return text.to_string();
    }
    if max_width == 1 {
        return "…".to_string();
    }

    let mut result = String::new();
    let mut used = 0;
    let limit = max_width - 1;
    for ch in text.chars() {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + width > limit {
            break;
        }
        used += width;
        result.push(ch);
    }
    result.push('…');
    result
}

pub fn format_time_ago_short(updated_at: &str) -> String {
    let parsed = DateTime::parse_from_rfc3339(updated_at)
        .map(|value| value.with_timezone(&Utc))
        .ok();
    let Some(parsed) = parsed else {
        return updated_at.to_string();
    };
    format_time_ago_short_dt(parsed)
}

/// Same compact "now/5m/6h/2d/3w/4mo/5y" output as `format_time_ago_short`,
/// but takes a parsed `DateTime` so callers that already have one don't need
/// to round-trip through RFC-3339.
pub fn format_time_ago_short_dt(updated_at: DateTime<Utc>) -> String {
    let seconds = (Utc::now() - updated_at).num_seconds().max(0);
    match seconds {
        0..=59 => "now".to_string(),
        60..=3599 => format!("{}m", seconds / 60),
        3600..=86_399 => format!("{}h", seconds / 3600),
        86_400..=604_799 => format!("{}d", seconds / 86_400),
        604_800..=2_592_000 => format!("{}w", seconds / 604_800),
        2_592_001..=31_535_999 => format!("{}mo", seconds / 2_592_000),
        _ => format!("{}y", seconds / 31_536_000),
    }
}

pub(super) fn truncate_for_width(text: &str, width: u16) -> String {
    truncate_for_display_width(text, usize::from(width))
}

#[cfg(test)]
mod tests {
    use super::{
        display_width, footer_engine_labels, footer_session_label, footer_workspace_candidates,
        format_request_elapsed, format_session_match_count, format_time_ago_short,
        format_token_count, format_token_count_value, format_usd, git_branch_for,
        truncate_for_display_width, truncate_for_width,
    };
    use crate::commands::code::TokenUsage;
    use chrono::{Duration as ChronoDuration, Utc};
    use std::time::Duration;

    #[test]
    fn format_usd_shows_sub_cent_spend() {
        assert_eq!(format_usd(1.234), "1.23");
        assert_eq!(format_usd(0.01), "0.01");
        assert_eq!(format_usd(0.0042), "0.0042");
        assert_eq!(format_usd(0.000065), "0.000065");
        assert_eq!(format_usd(0.009999), "0.01");
        assert_eq!(format_usd(0.0), "0.00");
    }

    #[test]
    fn test_truncate_for_width() {
        assert_eq!(truncate_for_width("hello", 10), "hello");
        assert_eq!(truncate_for_width("hello world", 6), "hello…");
    }

    #[test]
    fn footer_session_label_forms() {
        // A native id shows its short `#` handle.
        assert_eq!(
            footer_session_label("abcdef12-3456-7890-abcd-ef1234567890"),
            "#abcdef12"
        );
        // A fork keeps its source tag over an 8-char handle (new `<cli>-<8 hex>` ids).
        assert_eq!(footer_session_label("claude-a1b2c3d4"), "claude·a1b2c3d4");
        assert_eq!(footer_session_label("codex-deadbeef"), "codex·deadbeef");
        assert_eq!(footer_session_label("pi-019f15e1"), "pi·019f15e1");
        // Legacy `import-` forks (full UUID) clamp the same, not trailed in full.
        assert_eq!(
            footer_session_label("import-pi-019f15e1-a4bd-70e4-abcc-7fd70a5c4ca9"),
            "pi·019f15e1"
        );
    }

    #[test]
    fn test_footer_engine_labels_split_and_dedupe() {
        // Distinct key name: model + host segments.
        assert_eq!(
            footer_engine_labels("gpt-4o", "https://openrouter.ai/api/v1", "my-router"),
            ("gpt-4o".to_string(), Some("my-router".to_string()))
        );
        // Blank key name falls back to the URL host.
        assert_eq!(
            footer_engine_labels("gpt-4o", "https://openrouter.ai/api/v1", "  "),
            ("gpt-4o".to_string(), Some("openrouter.ai".to_string()))
        );
        // A key named after the model's `owner/` prefix would print the same
        // word twice (`aivo/starter · aivo`) — the host segment is dropped.
        assert_eq!(
            footer_engine_labels("aivo/starter", "https://api.getaivo.dev/v1", "aivo"),
            ("aivo/starter".to_string(), None)
        );
        // …or a key named exactly after the model, case-insensitively.
        assert_eq!(
            footer_engine_labels("GPT-4o", "https://api.openai.com/v1", "gpt-4o"),
            ("GPT-4o".to_string(), None)
        );
        // A local HF key (`hf:owner/repo`) ending in the model subsumes it.
        assert_eq!(
            footer_engine_labels(
                "Qwen2.5-0.5B-Instruct-GGUF",
                "http://127.0.0.1:8080/v1",
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"
            ),
            ("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string(), None)
        );
        // A non-matching hf basename is not treated as redundant (both kept).
        assert_eq!(
            footer_engine_labels(
                "some-model",
                "http://127.0.0.1:8080/v1",
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF"
            ),
            (
                "some-model".to_string(),
                Some("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string())
            )
        );
    }

    #[test]
    fn test_footer_workspace_candidates_degrade() {
        // Best-first: full path with branch, basename with branch, bare basename.
        assert_eq!(
            footer_workspace_candidates("/tmp/project", Some("feat/agent")),
            [
                "/tmp/project (feat/agent)".to_string(),
                "project (feat/agent)".to_string(),
                "project".to_string(),
            ]
        );
        // Empty branch → no suffix (same as None / not a repo).
        assert_eq!(
            footer_workspace_candidates("/tmp/project", Some("")),
            [
                "/tmp/project".to_string(),
                "project".to_string(),
                "project".to_string(),
            ]
        );
    }

    #[test]
    fn test_git_branch_for_reads_head() {
        let base = std::env::temp_dir().join(format!("aivo-gitbranch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let git = base.join(".git");
        std::fs::create_dir_all(&git).unwrap();

        // On a branch.
        std::fs::write(git.join("HEAD"), "ref: refs/heads/feat/x\n").unwrap();
        assert_eq!(
            git_branch_for(base.to_str().unwrap()).as_deref(),
            Some("feat/x")
        );
        // A nested subdir resolves up to the repo root.
        let sub = base.join("a/b");
        std::fs::create_dir_all(&sub).unwrap();
        assert_eq!(
            git_branch_for(sub.to_str().unwrap()).as_deref(),
            Some("feat/x")
        );
        // Detached HEAD → short commit hash.
        std::fs::write(
            git.join("HEAD"),
            "0123456789abcdef0123456789abcdef01234567\n",
        )
        .unwrap();
        assert_eq!(
            git_branch_for(base.to_str().unwrap()).as_deref(),
            Some("0123456")
        );
        // An empty path never latches onto the process's own repo.
        assert_eq!(git_branch_for(""), None);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_format_token_count_with_usage_shows_total() {
        assert_eq!(
            format_token_count(
                999,
                Some(TokenUsage {
                    prompt_tokens: 129,
                    completion_tokens: 11,
                    cache_read_input_tokens: 90,
                    cache_creation_input_tokens: 15,
                }),
            ),
            "140 tokens"
        );
        assert_eq!(
            format_token_count(
                5_120,
                Some(TokenUsage {
                    prompt_tokens: 5_000,
                    completion_tokens: 120,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                }),
            ),
            "5.1k tokens"
        );
    }

    #[test]
    fn test_format_token_count_marks_estimates() {
        assert_eq!(format_token_count(0, None), "0 tokens");
        assert_eq!(format_token_count(105, None), "~105 tokens");
        assert_eq!(format_token_count(5_000, None), "~5k tokens");
        assert_eq!(format_token_count(12_345, None), "~12.3k tokens");
    }

    #[test]
    fn test_format_token_count_value_scales_to_m() {
        // k tier (unchanged).
        assert_eq!(format_token_count_value(999), "999");
        assert_eq!(format_token_count_value(200_000), "200k");
        assert_eq!(format_token_count_value(128_000), "128k");
        // M tier — a 1M-token window reads "1M", not "1000k".
        assert_eq!(format_token_count_value(1_000_000), "1M");
        assert_eq!(format_token_count_value(1_500_000), "1.5M");
        assert_eq!(format_token_count_value(2_000_000), "2M");
        // Rollover boundary never shows "1000k".
        assert_eq!(format_token_count_value(999_999), "1M");
    }

    #[test]
    fn test_format_session_match_count() {
        assert_eq!(format_session_match_count(0, 0), "0 sessions");
        assert_eq!(format_session_match_count(4, 4), "4 sessions");
        assert_eq!(format_session_match_count(2, 5), "2/5");
    }

    #[test]
    fn test_truncate_for_display_width_handles_wide_text() {
        let truncated = truncate_for_display_width("你好🙂 hello", 8);
        assert!(display_width(&truncated) <= 8);
        assert!(truncated.ends_with('…'));
    }

    #[test]
    fn test_format_time_ago_short() {
        let updated_at = (Utc::now() - ChronoDuration::minutes(5)).to_rfc3339();
        assert_eq!(format_time_ago_short(&updated_at), "5m");
    }

    #[test]
    fn test_format_request_elapsed() {
        assert_eq!(format_request_elapsed(Duration::from_secs(54)), "54s");
        // Minute scale keeps the ticking seconds.
        assert_eq!(format_request_elapsed(Duration::from_secs(770)), "12m 50s");
        assert_eq!(format_request_elapsed(Duration::from_secs(60)), "1m 0s");
        // Hour/day scale drops the smallest unit to stay compact.
        assert_eq!(format_request_elapsed(Duration::from_secs(3_661)), "1h 1m");
        assert_eq!(format_request_elapsed(Duration::from_secs(90_061)), "1d 1h");
    }
}
