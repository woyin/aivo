use std::path::Path;
use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use ratatui::text::Text;
use ratatui::widgets::{Paragraph, Wrap};
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

const ATTACHMENT_OVERHEAD_CHARS: usize = 64;
const MESSAGE_OVERHEAD_CHARS: usize = 20;

pub(super) fn estimate_context_tokens(history: &[ChatMessage]) -> u64 {
    let total_chars: usize = history
        .iter()
        .map(|m| {
            let attachment_chars = m
                .attachments
                .iter()
                .map(|a| a.name.len() + ATTACHMENT_OVERHEAD_CHARS)
                .sum::<usize>();
            m.role.len() + m.content.len() + attachment_chars + MESSAGE_OVERHEAD_CHARS
        })
        .sum();
    (total_chars / 4) as u64
}

pub(super) fn build_footer_text(
    model: &str,
    base_url: &str,
    key_name: &str,
    cwd: &str,
    branch: Option<&str>,
    width: u16,
) -> String {
    // Prefer the user's key name; fall back to the provider host from the URL.
    let host = if key_name.trim().is_empty() {
        footer_host_label(base_url)
    } else {
        key_name.trim().to_string()
    };
    // Prefer the full (home-abbreviated) path so the agent's working dir is clear;
    // fall back to the basename, then drop the cwd, as width shrinks.
    let cwd_full = footer_cwd_label(cwd);
    let cwd_base = footer_cwd_basename(cwd);
    // When the cwd is a git repo, the branch trails the path as ` (branch)`. It's
    // kept alongside the basename fallback, then dropped as the width tightens.
    let branch_suffix = branch
        .filter(|b| !b.is_empty())
        .map(|b| format!(" ({b})"))
        .unwrap_or_default();
    // Local HF model: key name `hf:owner/repo` already ends in the model, so drop
    // the duplicate and lead with the ref.
    let is_redundant_hf = host
        .strip_prefix("hf:")
        .is_some_and(|repo| repo.rsplit('/').next() == Some(model));
    let lead = if is_redundant_hf {
        host.clone()
    } else {
        format!("{model} · {host}")
    };
    let candidates = [
        format!("{lead} · {cwd_full}{branch_suffix}"),
        format!("{lead} · {cwd_base}{branch_suffix}"),
        format!("{lead} · {cwd_base}"),
        lead.clone(),
        model.to_string(),
    ];

    candidates
        .into_iter()
        // `width` is a terminal-column budget, so pick by display width — a CJK
        // path/model (each glyph 2 columns) would otherwise count as half its
        // real width and be chosen even though it overflows the footer.
        .find(|candidate| display_width(candidate) <= usize::from(width.max(1)))
        .unwrap_or_else(|| truncate_for_width(model, width))
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

#[allow(dead_code)]
pub(super) fn wrapped_text_line_count(text: impl Into<Text<'static>>, width: u16) -> usize {
    if width == 0 {
        return 0;
    }

    Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .line_count(width)
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
        build_footer_text, display_width, format_request_elapsed, format_session_match_count,
        format_time_ago_short, format_token_count, format_token_count_value, git_branch_for,
        truncate_for_display_width, truncate_for_width, wrapped_text_line_count,
    };
    use crate::commands::code::TokenUsage;
    use chrono::{Duration as ChronoDuration, Utc};
    use std::time::Duration;

    #[test]
    fn test_wrapped_text_line_count_uses_ratatui_word_wrap() {
        assert_eq!(wrapped_text_line_count("", 10), 1);
        assert_eq!(wrapped_text_line_count("hello", 10), 1);
        assert_eq!(wrapped_text_line_count("abcdefghij", 5), 2);
        assert_eq!(wrapped_text_line_count("word word word", 8), 3);
    }

    #[test]
    fn test_truncate_for_width() {
        assert_eq!(truncate_for_width("hello", 10), "hello");
        assert_eq!(truncate_for_width("hello world", 6), "hello…");
    }

    #[test]
    fn test_build_footer_text_prefers_whole_segments() {
        // Wide: the full path (so an agent's working dir is unambiguous).
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                None,
                80
            ),
            "gpt-4o · openrouter.ai · /tmp/project"
        );
        // Medium: full path won't fit → fall back to the basename.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                None,
                34
            ),
            "gpt-4o · openrouter.ai · project"
        );
        // Narrow: drop the cwd.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                None,
                22
            ),
            "gpt-4o · openrouter.ai"
        );
        // Tiny: model only.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                None,
                6
            ),
            "gpt-4o"
        );
    }

    #[test]
    fn test_build_footer_text_collapses_redundant_hf_ref() {
        // A local HF model's key name (`hf:owner/repo`) has the model as its
        // basename — the footer collapses the duplicated model into the ref.
        assert_eq!(
            build_footer_text(
                "Qwen2.5-0.5B-Instruct-GGUF",
                "http://127.0.0.1:8080/v1",
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "/tmp/project",
                Some("main"),
                80,
            ),
            "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF · /tmp/project (main)"
        );
        // Tiniest width degrades to the bare model name, not the long ref.
        assert_eq!(
            build_footer_text(
                "Qwen2.5-0.5B-Instruct-GGUF",
                "http://127.0.0.1:8080/v1",
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "/tmp/project",
                Some("main"),
                26,
            ),
            "Qwen2.5-0.5B-Instruct-GGUF"
        );
        // A non-matching basename is not treated as redundant (both kept).
        assert_eq!(
            build_footer_text(
                "some-model",
                "http://127.0.0.1:8080/v1",
                "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "/tmp/project",
                None,
                80,
            ),
            "some-model · hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF · /tmp/project"
        );
    }

    #[test]
    fn test_build_footer_text_appends_git_branch() {
        // Wide: the branch trails the full path as ` (branch)`.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                Some("feat/agent"),
                80
            ),
            "gpt-4o · openrouter.ai · /tmp/project (feat/agent)"
        );
        // The branch is kept alongside the basename when the full path won't fit.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                Some("main"),
                40
            ),
            "gpt-4o · openrouter.ai · project (main)"
        );
        // Tighter: drop the branch before the basename.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                Some("main"),
                35
            ),
            "gpt-4o · openrouter.ai · project"
        );
        // Empty branch → no suffix (same as None / not a repo).
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "",
                "/tmp/project",
                Some(""),
                80
            ),
            "gpt-4o · openrouter.ai · /tmp/project"
        );
    }

    #[test]
    fn test_build_footer_text_prefers_key_name_over_host() {
        // A non-empty key name replaces the URL-derived host.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "my-router",
                "/tmp/project",
                None,
                80
            ),
            "gpt-4o · my-router · /tmp/project"
        );
        // Blank/whitespace name falls back to the host.
        assert_eq!(
            build_footer_text(
                "gpt-4o",
                "https://openrouter.ai/api/v1",
                "  ",
                "/tmp/project",
                None,
                80
            ),
            "gpt-4o · openrouter.ai · /tmp/project"
        );
    }

    #[test]
    fn test_build_footer_text_uses_display_width_for_cjk() {
        // A CJK cwd: each glyph is 2 columns, so the full-path candidate is 34
        // columns wide while only 32 chars. With width=32 a char-count check
        // would wrongly keep it (32 ≤ 32) and overflow the footer; display width
        // must fall back to the basename, which fits.
        let out = build_footer_text(
            "gpt-4o",
            "https://openrouter.ai/api/v1",
            "",
            "/tmp/项目",
            None,
            32,
        );
        assert_eq!(out, "gpt-4o · openrouter.ai · 项目");
        assert!(
            display_width(&out) <= 32,
            "footer overflows {} cols",
            display_width(&out)
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
