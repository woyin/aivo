use super::*;
use std::borrow::Cow;

#[derive(Clone)]
pub(super) struct StyledLine {
    pub(super) line: Line<'static>,
    pub(super) plain: String,
}

pub(super) struct RenderedTranscript {
    /// The logical (unwrapped) lines. Wrapped to the render width via
    /// [`wrap_transcript`] at draw time so our row model matches what ratatui
    /// actually paints (it word-wraps; a char-wrap count under-scrolls).
    pub(super) lines: Vec<StyledLine>,
    pub(super) plain_lines: Vec<String>,
    /// Per logical line: the accent-bar color for the block it belongs to, or
    /// `None` for chrome (intro, inter-block spacing). Aligned with `lines`.
    pub(super) bar_colors: Vec<Option<Color>>,
}

impl RenderedTranscript {
    pub(super) fn new(lines: Vec<StyledLine>, bar_colors: Vec<Option<Color>>) -> Self {
        let plain_lines = lines.iter().map(|l| l.plain.clone()).collect();
        Self {
            lines,
            plain_lines,
            bar_colors,
        }
    }
}

/// Word-wrap one styled line to `width`, preserving per-character styles and
/// returning one [`StyledLine`] per visual row. Mirrors a terminal word-wrap:
/// break on whitespace, hard-break words longer than the width. Rendering these
/// rows with ratatui's wrap OFF makes our row count exactly match the display.
/// A run of like characters (all whitespace or all non-whitespace) used by the
/// word-wrapper.
struct Token {
    is_space: bool,
    chars: Vec<(char, Style)>,
    width: usize,
}

// `cur_w` is a loop accumulator; its final write (a row break on the last token)
// is intentionally not read again.
#[allow(unused_assignments)]
pub(super) fn wrap_styled_line(spans: &[Span<'static>], width: usize) -> Vec<StyledLine> {
    let chars: Vec<(char, Style)> = spans
        .iter()
        .flat_map(|s| {
            let style = s.style;
            s.content.chars().map(move |c| (c, style))
        })
        .collect();
    if width == 0 || chars.is_empty() {
        return vec![styled_line_from_chars(&chars)];
    }
    let cw = |c: char| UnicodeWidthChar::width(c).unwrap_or(0).max(1);

    // Tokenize into alternating whitespace / word runs.
    let mut tokens: Vec<Token> = Vec::new();
    for (c, st) in chars {
        let is_space = c == ' ' || c == '\t';
        let w = cw(c);
        match tokens.last_mut() {
            Some(tok) if tok.is_space == is_space => {
                tok.chars.push((c, st));
                tok.width += w;
            }
            _ => tokens.push(Token {
                is_space,
                chars: vec![(c, st)],
                width: w,
            }),
        }
    }

    let mut rows: Vec<Vec<(char, Style)>> = Vec::new();
    let mut cur: Vec<(char, Style)> = Vec::new();
    let mut cur_w = 0usize;
    for Token {
        is_space,
        chars: buf,
        width: tw,
    } in tokens
    {
        if cur_w + tw <= width {
            cur.extend(buf);
            cur_w += tw;
        } else if is_space {
            // Whitespace that won't fit ends the row; drop it (no leading space).
            rows.push(std::mem::take(&mut cur));
            cur_w = 0;
        } else if tw <= width {
            // Word fits on a fresh row.
            if !cur.is_empty() {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            cur = buf;
            cur_w = tw;
        } else {
            // Word longer than the width: hard-break across rows.
            for (c, st) in buf {
                let w = cw(c);
                if cur_w + w > width && !cur.is_empty() {
                    rows.push(std::mem::take(&mut cur));
                    cur_w = 0;
                }
                cur.push((c, st));
                cur_w += w;
            }
        }
    }
    if !cur.is_empty() || rows.is_empty() {
        rows.push(cur);
    }
    rows.iter().map(|r| styled_line_from_chars(r)).collect()
}

/// Build a [`StyledLine`] from per-character styles, coalescing runs of the same
/// style into spans.
fn styled_line_from_chars(chars: &[(char, Style)]) -> StyledLine {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut plain = String::new();
    let mut cur = String::new();
    let mut cur_style: Option<Style> = None;
    for &(c, style) in chars {
        plain.push(c);
        if cur_style == Some(style) {
            cur.push(c);
        } else {
            if let Some(s) = cur_style {
                spans.push(Span::styled(std::mem::take(&mut cur), s));
            }
            cur.push(c);
            cur_style = Some(style);
        }
    }
    if let Some(s) = cur_style {
        spans.push(Span::styled(cur, s));
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

/// A transcript wrapped to the render width: the `Text` (rendered with wrap OFF),
/// the per-row plain strings (for selection), and per-row bar colors (the gutter).
pub(super) struct WrappedTranscript {
    pub(super) text: Text<'static>,
    pub(super) rows: Vec<String>,
    pub(super) bars: Vec<Option<Color>>,
}

/// Word-wrap all logical lines to `width`, carrying each line's bar color onto
/// every visual row.
pub(super) fn wrap_transcript(
    lines: &[StyledLine],
    bars: &[Option<Color>],
    width: u16,
) -> WrappedTranscript {
    let width = usize::from(width.max(1));
    let mut text_lines: Vec<Line<'static>> = Vec::new();
    let mut rows: Vec<String> = Vec::new();
    let mut row_bars: Vec<Option<Color>> = Vec::new();
    for (idx, sl) in lines.iter().enumerate() {
        let bar = bars.get(idx).copied().flatten();
        for vrow in wrap_styled_line(&sl.line.spans, width) {
            let vrow = fill_trailing_background(vrow, width);
            text_lines.push(vrow.line);
            rows.push(vrow.plain);
            row_bars.push(bar);
        }
    }
    if text_lines.is_empty() {
        text_lines.push(Line::default());
        rows.push(String::new());
        row_bars.push(None);
    }
    WrappedTranscript {
        text: Text::from(text_lines),
        rows,
        bars: row_bars,
    }
}

/// If a wrapped visual row ends in a background-colored span, extend that
/// background across the rest of the row width. This turns the tinted inline-diff
/// lines (see `render_edit_diff`) into contiguous full-width blocks, so a long
/// changed line that wraps still reads as one shaded region instead of a ragged
/// tail at the left margin. Ordinary rows (no trailing background) are untouched,
/// so nothing else in the transcript gains a background.
fn fill_trailing_background(mut row: StyledLine, width: usize) -> StyledLine {
    let Some(bg) = row.line.spans.last().and_then(|span| span.style.bg) else {
        return row;
    };
    let used = usize::from(row_display_width(&row.plain));
    if used >= width {
        return row;
    }
    let pad = " ".repeat(width - used);
    row.plain.push_str(&pad);
    row.line
        .spans
        .push(Span::styled(pad, Style::default().bg(bg)));
    row
}

/// Accent-bar color for a transcript role: user = blue, assistant = magenta
/// (aivo's voice), agent tool steps = cyan, everything else = muted.
pub(super) fn role_bar_color(role: &str) -> Color {
    match role {
        "user" => USER,
        "assistant" => ACCENT,
        "tool_call" | "tool_result" | "local_command" | "plan" => TOOL,
        _ => MUTED,
    }
}

fn wrap_one_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    let mut wrapped = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in line.chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        if current_width > 0 && current_width + ch_width > width {
            wrapped.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }
    wrapped.push(current);
    wrapped
}

pub(super) fn wrap_plain_lines(lines: &[String], width: u16) -> Vec<String> {
    let width = usize::from(width.max(1));
    let mut wrapped = Vec::new();
    for line in lines {
        wrapped.extend(wrap_one_line(line, width));
    }
    if wrapped.is_empty() {
        wrapped.push(String::new());
    }
    wrapped
}

pub(super) fn normalized_selection(
    selection: TranscriptSelection,
) -> (TranscriptPoint, TranscriptPoint) {
    if (selection.anchor.row, selection.anchor.column)
        <= (selection.focus.row, selection.focus.column)
    {
        (selection.anchor, selection.focus)
    } else {
        (selection.focus, selection.anchor)
    }
}

pub(super) fn selected_text_from_rows(
    rows: &[String],
    selection: TranscriptSelection,
) -> Option<String> {
    let (start, end) = normalized_selection(selection);
    if (start.row, start.column) == (end.row, end.column) || start.row >= rows.len() {
        return None;
    }

    let end_row = end.row.min(rows.len().saturating_sub(1));
    let mut selected = Vec::new();
    for (row_index, row) in rows.iter().enumerate().take(end_row + 1).skip(start.row) {
        let text = if start.row == end_row {
            slice_display_columns(row, start.column, end.column)
        } else if row_index == start.row {
            slice_display_columns(row, start.column, u16::MAX)
        } else if row_index == end_row {
            slice_display_columns(row, 0, end.column)
        } else {
            row.clone()
        };
        // Drop wrap-padding spaces so pasted text has no ragged trailing runs.
        selected.push(text.trim_end().to_string());
    }

    Some(selected.join("\n"))
}

/// Display-column width of a visual row (CJK-aware).
pub(super) fn row_display_width(row: &str) -> u16 {
    let mut width = 0usize;
    for ch in row.chars() {
        width += UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
    }
    width.min(u16::MAX as usize) as u16
}

/// Word boundaries (display columns `[start, end)`) around `column` in `row`.
/// Returns `None` when the click lands on whitespace or past the row's end —
/// the caller falls back to a plain caret in that case.
pub(super) fn word_bounds_at(row: &str, column: u16) -> Option<(u16, u16)> {
    let column = usize::from(column);
    // Build (start_col, end_col, is_word) spans per character so we can locate
    // the clicked column and expand over the contiguous word run it sits in.
    let mut spans: Vec<(usize, usize, bool)> = Vec::new();
    let mut col = 0usize;
    for ch in row.chars() {
        let w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        spans.push((col, col + w, !ch.is_whitespace()));
        col += w;
    }
    let hit = spans
        .iter()
        .position(|&(s, e, is_word)| is_word && column >= s && column < e)?;
    let mut start = hit;
    while start > 0 && spans[start - 1].2 {
        start -= 1;
    }
    let mut end = hit;
    while end + 1 < spans.len() && spans[end + 1].2 {
        end += 1;
    }
    let start_col = spans[start].0.min(u16::MAX as usize) as u16;
    let end_col = spans[end].1.min(u16::MAX as usize) as u16;
    Some((start_col, end_col))
}

pub(super) fn slice_display_columns(text: &str, start: u16, end: u16) -> String {
    let start = usize::from(start);
    let end = usize::from(end);
    let mut result = String::new();
    let mut col = 0usize;
    for ch in text.chars() {
        let width = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
        let next = col + width;
        if next > start && col < end {
            result.push(ch);
        }
        col = next;
        if col >= end {
            break;
        }
    }
    result
}

pub(super) fn push_message_spacing(lines: &mut Vec<StyledLine>) {
    if !lines.is_empty() {
        lines.push(blank_line());
    }
}

/// Append a rendered block under a single accent-bar color (`None` = no bar),
/// keeping `bars` in lockstep with `lines`. Trailing blank lines are trimmed so
/// the only gaps between blocks are the explicit (barless) spacing lines — that
/// keeps the accent bar tight to its content.
pub(super) fn push_block(
    lines: &mut Vec<StyledLine>,
    bars: &mut Vec<Option<Color>>,
    mut block: Vec<StyledLine>,
    bar: Option<Color>,
) {
    while block.last().is_some_and(|l| l.plain.trim().is_empty()) {
        block.pop();
    }
    bars.resize(lines.len() + block.len(), bar);
    lines.extend(block);
}

/// The two-row half-block "AIVO" wordmark. Each glyph cell is a single-width
/// box-drawing char, so the art is exactly 13 columns wide and lines up cleanly
/// under any monospace font.
pub(super) const BRAND_WORDMARK: [&str; 2] = ["▄▀█ █ █░█ █▀█", "█▀█ █ ▀▄▀ █▄█"];
/// Welcome-screen tagline shown under the wordmark in the empty state.
pub(super) const BRAND_TAGLINE: &str = "chat · ask anything";

/// The brand wordmark as styled lines, painted in the accent color. Single
/// source of truth for the empty state and the transcript-top intro so both
/// always start at the same column (see `test_intro_column_stable_*`).
pub(super) fn brand_wordmark_lines() -> Vec<StyledLine> {
    BRAND_WORDMARK
        .iter()
        .map(|row| {
            line_plain(
                (*row).to_string(),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

pub(super) fn push_transcript_intro(lines: &mut Vec<StyledLine>) {
    // Mirror the empty-state top gap so the banner keeps its top padding once a
    // message lands — no vertical jump when the transcript takes over.
    for _ in 0..EMPTY_STATE_TOP_GAP {
        lines.push(blank_line());
    }
    // Wordmark + tagline above a live conversation, mirroring the welcome
    // screen so the banner survives the first message instead of being clipped
    // to the bare wordmark. Model / base_url / cwd live in the footer status bar.
    lines.extend(brand_wordmark_lines());
    lines.push(line_plain(
        BRAND_TAGLINE.to_string(),
        Style::default().fg(MUTED),
    ));
}

pub(super) fn should_add_message_spacing(previous_role: Option<&str>, next_role: &str) -> bool {
    let Some(prev) = previous_role else {
        return false;
    };
    if next_role.is_empty() {
        return false;
    }
    // Keep a tool result tight against its call, and keep consecutive tool
    // steps grouped — no blank line within an agent tool sequence (the blank
    // before/after the whole sequence is still kept).
    if next_role == "tool_result" {
        return false;
    }
    let prev_is_tool = prev == "tool_call" || prev == "tool_result";
    if prev_is_tool && next_role == "tool_call" {
        return false;
    }
    true
}

pub(super) fn attachment_kind_label(attachment: &MessageAttachment) -> &'static str {
    if attachment.mime_type.starts_with("image/") {
        "image"
    } else {
        "file"
    }
}

pub(super) fn render_user_attachment_lines(
    lines: &mut Vec<StyledLine>,
    attachments: &[MessageAttachment],
) {
    for attachment in attachments {
        push_styled_line(
            lines,
            format!(
                "  [{}] {}",
                attachment_kind_label(attachment),
                attachment.name
            ),
            Style::default().fg(MUTED),
        );
    }
}

pub(super) fn composer_attachment_lines(attachments: &[MessageAttachment]) -> Vec<Line<'static>> {
    attachments
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            Line::from(vec![
                Span::styled("· ", Style::default().fg(ACCENT)),
                Span::styled(
                    format!(
                        "{}. [{}] {}",
                        index + 1,
                        attachment_kind_label(attachment),
                        attachment.name
                    ),
                    Style::default().fg(MUTED),
                ),
            ])
        })
        .collect()
}

pub(super) fn render_user_message(
    lines: &mut Vec<StyledLine>,
    content: &str,
    attachments: &[MessageAttachment],
) {
    let mut had_line = false;
    for (idx, raw_line) in content.lines().enumerate() {
        let prefix = if idx == 0 { "> " } else { "  " };
        push_styled_line(
            lines,
            format!("{prefix}{raw_line}"),
            Style::default().fg(USER),
        );
        had_line = true;
    }
    if !had_line {
        push_styled_line(lines, "> ", Style::default().fg(USER));
    }
    render_user_attachment_lines(lines, attachments);
}

pub(super) fn extend_without_leading_blank(
    lines: &mut Vec<StyledLine>,
    mut rendered: Vec<StyledLine>,
) {
    while rendered
        .first()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        rendered.remove(0);
    }
    lines.extend(rendered);
}

/// Leading marker of a folded reasoning header (`▸ thinking · N lines`).
pub(super) const THINKING_SUMMARY_PREFIX: &str = "▸ thinking";
/// Leading marker of an expanded reasoning header (`▾ thinking · N lines`).
pub(super) const THINKING_EXPANDED_PREFIX: &str = "▾ thinking";

/// Whether a rendered transcript row is a clickable thinking header (collapsed
/// `▸` or expanded `▾`). The click handler maps a click back to its block; the
/// content rows of an expanded block are indented so they don't match.
pub(super) fn is_thinking_header(row: &str) -> bool {
    let row = row.trim_start();
    row.starts_with(THINKING_SUMMARY_PREFIX) || row.starts_with(THINKING_EXPANDED_PREFIX)
}

/// Human "thinking · X" suffix: a duration when known (`2s`, `1m 5s`), else the
/// line count as a fallback (e.g. cursor turns / resumed history with no timing).
fn thinking_suffix(reasoning: &str, duration_ms: Option<u64>) -> String {
    if let Some(ms) = duration_ms {
        let secs = (ms + 500) / 1000;
        return if secs == 0 {
            "<1s".to_string()
        } else if secs < 60 {
            format!("{secs}s")
        } else if secs % 60 == 0 {
            format!("{}m", secs / 60)
        } else {
            format!("{}m {}s", secs / 60, secs % 60)
        };
    }
    let count = normalized_reasoning_lines(reasoning).len();
    let unit = if count == 1 { "line" } else { "lines" };
    format!("{count} {unit}")
}

fn reasoning_header(prefix: &str, reasoning: &str, duration_ms: Option<u64>) -> StyledLine {
    line_with_plain(vec![Span::styled(
        format!("{prefix} · {}", thinking_suffix(reasoning, duration_ms)),
        Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
    )])
}

/// One-line folded stand-in for a reasoning block. Live and committed thinking
/// both render this way, so the block keeps constant height across the handoff.
pub(super) fn render_reasoning_summary(
    lines: &mut Vec<StyledLine>,
    reasoning: &str,
    duration_ms: Option<u64>,
) {
    lines.push(reasoning_header(
        THINKING_SUMMARY_PREFIX,
        reasoning,
        duration_ms,
    ));
}

/// Expanded form: the `▾ thinking` header above the full reasoning, indented so
/// content rows aren't matched as headers (see `is_thinking_header`).
pub(super) fn render_reasoning_block(
    lines: &mut Vec<StyledLine>,
    reasoning: &str,
    duration_ms: Option<u64>,
) {
    lines.push(reasoning_header(
        THINKING_EXPANDED_PREFIX,
        reasoning,
        duration_ms,
    ));

    let reasoning_lines = normalized_reasoning_lines(reasoning);
    let mut had_line = false;
    for raw_line in reasoning_lines {
        lines.push(line_with_plain(vec![
            Span::styled("  ".to_string(), Style::default()),
            Span::styled(
                raw_line,
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            ),
        ]));
        had_line = true;
    }

    if !had_line {
        push_styled_line(
            lines,
            "  ".to_string(),
            Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
        );
    }
}

pub(super) fn normalized_reasoning_lines(reasoning: &str) -> Vec<String> {
    let mut lines = Vec::new();

    for raw_line in reasoning.lines() {
        let trimmed = raw_line.trim();
        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }

    lines
}

pub(super) fn render_assistant_message(
    lines: &mut Vec<StyledLine>,
    reasoning: Option<&str>,
    content: &str,
    width: u16,
) {
    if let Some(reasoning) = reasoning.filter(|text| !text.trim().is_empty()) {
        render_reasoning_block(lines, reasoning, None);
        if !content.is_empty() {
            push_styled_line(lines, "", Style::default());
        }
    }

    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content, width));
    }
}

/// Reasoning to render alongside an assistant turn: text, fold state, and how
/// long thinking took (`None` → line-count fallback, see `thinking_suffix`).
pub(super) struct ReasoningView<'a> {
    pub(super) text: &'a str,
    pub(super) collapsed: bool,
    pub(super) duration_ms: Option<u64>,
}

/// Push an assistant turn as up to two separately-barred blocks so the muted
/// "Thinking" block carries a different left border (`MUTED`) than the answer
/// (`content_bar`). `push_block` paints one gutter color per block, so a distinct
/// thinking bar requires committing the reasoning as its own block.
pub(super) fn push_assistant_blocks(
    lines: &mut Vec<StyledLine>,
    bars: &mut Vec<Option<Color>>,
    reasoning: Option<ReasoningView<'_>>,
    content: &str,
    width: u16,
    content_bar: Color,
) {
    if let Some(view) = reasoning.filter(|v| !v.text.trim().is_empty()) {
        let mut block = Vec::new();
        // The live tail is always collapsed so it doesn't expand-then-fold mid-turn.
        if view.collapsed {
            render_reasoning_summary(&mut block, view.text, view.duration_ms);
        } else {
            render_reasoning_block(&mut block, view.text, view.duration_ms);
        }
        push_block(lines, bars, block, Some(MUTED));
        if !content.is_empty() {
            // A barless blank separates the muted thinking gutter from the answer's.
            lines.push(blank_line());
            bars.push(None);
        }
    }

    if !content.is_empty() {
        // Reuse the single content renderer (no reasoning — that's its own block).
        let mut block = Vec::new();
        render_assistant_message(&mut block, None, content, width);
        push_block(lines, bars, block, Some(content_bar));
    }
}

pub(super) fn render_pending_status(
    lines: &mut Vec<StyledLine>,
    frame_tick: usize,
    reduce_motion: bool,
    elapsed: Duration,
    activity: &str,
) {
    let spinner = spinner_frame_indexed(frame_tick, reduce_motion);
    let text = format!(
        "{spinner} {activity} ({} • esc to interrupt)",
        format_request_elapsed(elapsed)
    );
    push_styled_line(
        lines,
        text,
        Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
    );
}

/// Footer context-stat color by fullness: a quiet signal that warms toward the
/// model's window limit (compaction territory) before it's hit.
pub(super) fn context_fill_color(pct: u64) -> Color {
    if pct >= 95 {
        ERROR
    } else if pct >= 80 {
        WARNING
    } else {
        MUTED
    }
}

pub(super) fn spinner_frame_indexed(frame_tick: usize, reduce_motion: bool) -> &'static str {
    if reduce_motion {
        return spinner_frame(0);
    }
    spinner_frame(frame_tick / 5)
}

pub(super) fn notice_display(notice: Option<&(Color, String)>) -> Option<(Color, Cow<'_, str>)> {
    notice.map(|(color, text)| {
        let formatted = if *color == ERROR {
            Cow::Owned(format!("Error: {text}"))
        } else {
            Cow::Borrowed(text.as_str())
        };
        (*color, formatted)
    })
}

pub(super) fn rect_contains(area: Rect, point: (u16, u16)) -> bool {
    let (x, y) = point;
    x >= area.x
        && x < area.x.saturating_add(area.width)
        && y >= area.y
        && y < area.y.saturating_add(area.height)
}

pub(super) fn render_notice_line(lines: &mut Vec<StyledLine>, color: Color, text: &str) {
    push_styled_line(lines, text.to_string(), Style::default().fg(color));
}

pub(super) fn render_system_message(
    lines: &mut Vec<StyledLine>,
    role: &str,
    content: &str,
    width: u16,
) {
    push_styled_line(
        lines,
        role.to_string(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    );
    if !content.is_empty() {
        extend_without_leading_blank(lines, render_markdown_lines(content, width));
    }
}

/// Render an agent tool invocation as a Tree-style `→ verb(args)` line. The verb
/// is yellow for mutating tools (write/edit/bash), dim otherwise. The agent
/// bridge stores these as `tool_call` entries whose `content` is JSON
/// `{"name", "args"}` (chat history is a display log; the engine owns the real
/// LLM context).
pub(super) fn render_tool_call(
    lines: &mut Vec<StyledLine>,
    name: &str,
    args: &serde_json::Value,
    result: Option<&str>,
    failed: bool,
    cwd: &str,
) {
    let summary = tool_arg_summary(name, args, cwd);
    let verb_style = if crate::agent::tools::is_mutating(name) {
        Style::default().fg(WARNING)
    } else {
        Style::default().fg(MUTED)
    };
    let mut spans = vec![Span::styled("→ ".to_string(), Style::default().fg(TOOL))];
    if name == "subagent" {
        // A delegated task reads as the task itself — "subagent" is jargon and the
        // arrow already marks it as a step. Show the task in place of the verb.
        let label = if summary.is_empty() {
            "delegated a task".to_string()
        } else {
            summary
        };
        spans.push(Span::styled(label, verb_style));
    } else {
        spans.push(Span::styled(tool_display_name(name), verb_style));
        // cursor reports no target for its tools (only a generic title), so the
        // summary is often empty — skip the `()` rather than render `read_file()`.
        if !summary.is_empty() {
            spans.push(Span::styled(
                format!("({summary})"),
                Style::default().fg(MUTED),
            ));
        }
    }
    lines.push(line_with_plain(spans));
    // For edit tools, show a compact diff of what changed so the user can review
    // the agent's edit without opening the file (no-op for tools without a
    // textual old/new, e.g. cursor edits or write_file).
    render_edit_diff(lines, name, args);
    // Cursor stores its result on the call entry (the in-process agent emits a
    // separate `tool_result` line instead) — surface it as a compact `⎿` line,
    // in the error hue when the tool failed.
    if failed {
        lines.push(line_with_plain(vec![
            Span::styled("  ⎿ ".to_string(), Style::default().fg(ERROR)),
            Span::styled(
                result.unwrap_or("failed").to_string(),
                Style::default().fg(ERROR),
            ),
        ]));
    } else if let Some(result) = result.filter(|r| !r.is_empty()) {
        lines.push(line_with_plain(vec![
            Span::styled("  ⎿ ".to_string(), Style::default().fg(FAINT)),
            Span::styled(result.to_string(), Style::default().fg(FAINT)),
        ]));
    }
}

/// Decode the optional result + failed flag a cursor `tool_call` entry carries
/// after enrichment (the in-process agent leaves both unset — it reports results
/// as separate `tool_result` entries).
pub(super) fn decode_tool_outcome(content: &str) -> (Option<String>, bool) {
    let Some(v) = serde_json::from_str::<serde_json::Value>(content).ok() else {
        return (None, false);
    };
    let result = v.get("result").and_then(|x| x.as_str()).map(str::to_string);
    let failed = v.get("failed").and_then(|x| x.as_bool()).unwrap_or(false);
    (result, failed)
}

/// Decode a `tool_call` history entry's JSON `{name, args}` payload.
pub(super) fn decode_tool_call(content: &str) -> (String, serde_json::Value) {
    let decoded = serde_json::from_str::<serde_json::Value>(content).ok();
    let name = decoded
        .as_ref()
        .and_then(|v| v.get("name"))
        .and_then(|x| x.as_str())
        .unwrap_or("tool")
        .to_string();
    let args = decoded
        .as_ref()
        .and_then(|v| v.get("args"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    (name, args)
}

/// Render a coalesced run of same-kind tool calls as one `→ verb N: a, b…` line.
/// cursor agents explore in many small steps, so a card per call is noise — the
/// run collapses to its count and the distinguishing targets (file basenames,
/// patterns, commands).
pub(super) fn render_tool_call_group(
    lines: &mut Vec<StyledLine>,
    name: &str,
    targets: &[String],
    failed: usize,
) {
    let n = targets.len();
    let head = match name {
        "read_file" => format!("read {n} files"),
        "edit_file" | "multi_edit" => format!("edited {n} files"),
        "delete_file" => format!("deleted {n} files"),
        "grep" | "glob" => format!("searched ×{n}"),
        "run_bash" => format!("ran {n} commands"),
        "web_fetch" => format!("fetched {n} URLs"),
        _ => format!("{} ×{n}", tool_display_name(name)),
    };
    let verb_style = if crate::agent::tools::is_mutating(name) {
        Style::default().fg(WARNING)
    } else {
        Style::default().fg(MUTED)
    };
    let mut spans = vec![
        Span::styled("→ ".to_string(), Style::default().fg(TOOL)),
        Span::styled(head, verb_style),
    ];
    let list = join_targets(targets, 56);
    if !list.is_empty() {
        spans.push(Span::styled(
            format!(": {list}"),
            Style::default().fg(MUTED),
        ));
    }
    if failed > 0 {
        spans.push(Span::styled(
            format!(" · {failed} failed"),
            Style::default().fg(ERROR),
        ));
    }
    lines.push(line_with_plain(spans));
}

/// Friendly verb for a tool call: an `mcp__server__tool` name renders as
/// `server/tool` (the raw double-underscore form is ugly); everything else is
/// shown as-is.
fn tool_display_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("mcp__")
        && let Some((server, tool)) = rest.split_once("__")
    {
        return format!("{server}/{tool}");
    }
    name.to_string()
}

/// The compact target shown for a tool inside a coalesced run: a file's
/// basename, a search pattern, or a command — the bit that differs between
/// sibling calls.
pub(super) fn tool_call_target(name: &str, args: &serde_json::Value) -> String {
    let pick = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match name {
        "read_file" | "edit_file" | "multi_edit" | "delete_file" | "write_file" | "list_dir" => {
            basename(pick("path"))
        }
        "grep" | "glob" => pick("pattern").to_string(),
        "run_bash" => pick("command").to_string(),
        "web_fetch" => pick("url").to_string(),
        _ => String::new(),
    }
}

/// Final path segment (handles both `/` and `\` separators).
fn basename(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .find(|s| !s.is_empty())
        .unwrap_or(path)
        .to_string()
}

/// Join target labels into one line, dropping empties, capped at `max` display
/// columns with a `(+K more)` tail when the rest don't fit.
fn join_targets(targets: &[String], max: usize) -> String {
    let items: Vec<&str> = targets
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let mut out = String::new();
    let mut shown = 0usize;
    for (i, item) in items.iter().enumerate() {
        let piece = if i == 0 {
            (*item).to_string()
        } else {
            format!(", {item}")
        };
        if shown > 0 && cell_width(&out) + cell_width(&piece) > max {
            break;
        }
        out.push_str(&piece);
        shown += 1;
    }
    let remaining = items.len() - shown;
    if remaining > 0 {
        out.push_str(&format!(" (+{remaining} more)"));
    }
    out
}

/// Most output lines a `!cmd` run shows in the transcript; the rest collapse to a
/// "+N more lines" marker. The reader (`run_shell_streaming`) captures somewhat
/// more than this so that count is accurate for moderate output.
pub(super) const MAX_OUTPUT_LINES: usize = 40;

/// How many output lines a finished `!cmd` persists into its `local_command`
/// history entry (and thus the on-disk session). Only the first `MAX_OUTPUT_LINES`
/// ever render, so a small preview keeps the session file from ballooning on a
/// floody command; the true line count rides along as `total_lines` so the
/// "+N more" marker stays honest, and the full output (for `ctrl+o`) lives in
/// `last_local_output`, not here.
pub(super) const MAX_PERSISTED_OUTPUT_LINES: usize = 200;

/// The first `n` lines of `s`, rejoined with `\n` (no trailing newline). Used to
/// bound what a `!cmd` run persists without disturbing its display line count.
pub(super) fn first_lines(s: &str, n: usize) -> String {
    s.lines().take(n).collect::<Vec<_>>().join("\n")
}

/// Render a `!cmd` local shell run: a `! command` header over its output (stdout
/// faint, stderr in the warning hue), with the line COUNT capped so a noisy
/// command can't flood the transcript. Each shown line is rendered in full — the
/// transcript word-wraps long lines onto extra rows (grok's pager shows command
/// output wrapped and scrollable, never per-line truncated), so nothing is clipped
/// at the pane edge. Stored as a `local_command` entry whose `content` is JSON
/// `{"command", "stdout", "stderr", "exit_code"}` (plus optional
/// `running`/`truncated`/`interrupted` flags).
pub(super) fn render_local_command(lines: &mut Vec<StyledLine>, content: &str) {
    let decoded =
        serde_json::from_str::<serde_json::Value>(content).unwrap_or(serde_json::Value::Null);
    let pick = |k: &str| decoded.get(k).and_then(|v| v.as_str()).unwrap_or("");
    let flag = |k: &str| decoded.get(k).and_then(|v| v.as_bool()).unwrap_or(false);
    let command = pick("command");
    let stdout = pick("stdout");
    let stderr = pick("stderr");
    let exit_code = decoded
        .get("exit_code")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    // While running there is no exit code yet; `truncated`/`interrupted` annotate
    // a finished run cut short by the capture caps or by esc.
    let running = flag("running");
    let truncated = flag("truncated");
    let interrupted = flag("interrupted");

    lines.push(line_with_plain(vec![
        Span::styled(
            "! ".to_string(),
            Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
        ),
        Span::styled(command.to_string(), Style::default().fg(TEXT)),
    ]));

    // A committed run stores only a bounded preview of its output but carries the
    // true line count in `total_lines`, so "+N more" reflects everything the
    // command produced (viewable in full via ctrl+o), not just the persisted slice.
    // A live run has no `total_lines` yet — count its (full) streamed output.
    let total = decoded
        .get("total_lines")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .unwrap_or_else(|| stdout.lines().count() + stderr.lines().count());
    let mut shown = 0usize;
    for (text, color) in stdout
        .lines()
        .map(|l| (l, FAINT))
        .chain(stderr.lines().map(|l| (l, WARNING)))
    {
        if shown >= MAX_OUTPUT_LINES {
            break;
        }
        lines.push(line_with_plain(vec![Span::styled(
            format!("  {text}"),
            Style::default().fg(color),
        )]));
        shown += 1;
    }
    if total > shown {
        let suffix = if truncated { ", truncated" } else { "" };
        lines.push(line_with_plain(vec![Span::styled(
            format!("  … (+{} more lines{suffix})", total - shown),
            Style::default().fg(FAINT),
        )]));
    } else if truncated {
        lines.push(line_with_plain(vec![Span::styled(
            "  … (output truncated)".to_string(),
            Style::default().fg(FAINT),
        )]));
    }
    if running {
        lines.push(line_with_plain(vec![Span::styled(
            "  (running…)".to_string(),
            Style::default().fg(FAINT),
        )]));
    } else if total == 0 && exit_code == 0 && !interrupted {
        lines.push(line_with_plain(vec![Span::styled(
            "  (no output)".to_string(),
            Style::default().fg(FAINT),
        )]));
    }
    if interrupted {
        lines.push(line_with_plain(vec![Span::styled(
            "  [interrupted]".to_string(),
            Style::default().fg(ERROR),
        )]));
    } else if !running && !truncated && exit_code != 0 {
        // A truncated run was killed by us (cap/timeout), so its non-zero status is
        // ours, not the command's — the "truncated" note already explains the stop,
        // so don't add a misleading `[exited -1]`.
        lines.push(line_with_plain(vec![Span::styled(
            format!("  [exited {exit_code}]"),
            Style::default().fg(ERROR),
        )]));
    }
}

/// Render an agent tool result as a compact `⎿ summary` line under its call.
pub(super) fn render_tool_result(
    lines: &mut Vec<StyledLine>,
    result: &str,
    cwd: &str,
    tool: Option<&str>,
) {
    lines.push(line_with_plain(vec![
        Span::styled("  ⎿ ".to_string(), Style::default().fg(FAINT)),
        Span::styled(
            tool_result_summary(result, cwd, tool),
            Style::default().fg(FAINT),
        ),
    ]));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DiffTag {
    Equal,
    Del,
    Ins,
}

/// One display row of a grouped diff: an unchanged context line, a removed or
/// added line, or a `⋯` marker standing in for a collapsed run of context.
enum DiffRow<'a> {
    Context(&'a str),
    Del(&'a str),
    Ins(&'a str),
    Gap,
}

/// Line-level LCS diff of `old` vs `new`, so an edit preview marks only the lines
/// that actually changed and shows the rest as context — the way Claude Code
/// previews an edit, rather than blindly flagging every old line removed and every
/// new line added. Removals are emitted before additions within each change run
/// (git order). Falls back to a plain remove-all / add-all list past a size cap so
/// a giant rewrite can't trigger the O(n·m) table.
fn diff_lines<'a>(old: &'a str, new: &'a str) -> Vec<(DiffTag, &'a str)> {
    let a: Vec<&str> = old.lines().collect();
    let b: Vec<&str> = new.lines().collect();
    let (n, m) = (a.len(), b.len());
    if n > 600 || m > 600 {
        let mut ops = Vec::with_capacity(n + m);
        ops.extend(a.iter().map(|l| (DiffTag::Del, *l)));
        ops.extend(b.iter().map(|l| (DiffTag::Ins, *l)));
        return ops;
    }

    // Suffix LCS-length table: lcs[i][j] = LCS length of a[i..], b[j..].
    let mut lcs = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if a[i] == b[j] {
                lcs[i + 1][j + 1] + 1
            } else {
                lcs[i + 1][j].max(lcs[i][j + 1])
            };
        }
    }

    let mut raw: Vec<(DiffTag, &str)> = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            raw.push((DiffTag::Equal, a[i]));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            raw.push((DiffTag::Del, a[i]));
            i += 1;
        } else {
            raw.push((DiffTag::Ins, b[j]));
            j += 1;
        }
    }
    raw.extend(a[i..].iter().map(|l| (DiffTag::Del, *l)));
    raw.extend(b[j..].iter().map(|l| (DiffTag::Ins, *l)));

    // Within each maximal run of changed lines, show removals before additions
    // regardless of how the backtrack interleaved them.
    let mut ops = Vec::with_capacity(raw.len());
    let mut k = 0;
    while k < raw.len() {
        if raw[k].0 == DiffTag::Equal {
            ops.push(raw[k]);
            k += 1;
            continue;
        }
        let start = k;
        while k < raw.len() && raw[k].0 != DiffTag::Equal {
            k += 1;
        }
        ops.extend(raw[start..k].iter().filter(|o| o.0 == DiffTag::Del));
        ops.extend(raw[start..k].iter().filter(|o| o.0 == DiffTag::Ins));
    }
    ops
}

/// Trim a diff to the changed lines plus `context` lines of surrounding context
/// (git's `-U`), collapsing any longer unchanged gap between two kept regions into
/// a single `⋯`. Leading/trailing context beyond the window is dropped silently.
/// Returns empty when nothing changed.
fn group_diff<'a>(ops: &[(DiffTag, &'a str)], context: usize) -> Vec<DiffRow<'a>> {
    if !ops.iter().any(|(t, _)| *t != DiffTag::Equal) {
        return Vec::new();
    }
    let keep: Vec<bool> = (0..ops.len())
        .map(|i| {
            let lo = i.saturating_sub(context);
            let hi = (i + context).min(ops.len() - 1);
            (lo..=hi).any(|j| ops[j].0 != DiffTag::Equal)
        })
        .collect();

    let mut rows = Vec::new();
    let mut prev_kept: Option<usize> = None;
    for (i, keep_it) in keep.iter().enumerate() {
        if !keep_it {
            continue;
        }
        if prev_kept.is_some_and(|p| i > p + 1) {
            rows.push(DiffRow::Gap);
        }
        rows.push(match ops[i].0 {
            DiffTag::Equal => DiffRow::Context(ops[i].1),
            DiffTag::Del => DiffRow::Del(ops[i].1),
            DiffTag::Ins => DiffRow::Ins(ops[i].1),
        });
        prev_kept = Some(i);
    }
    rows
}

/// A compact line diff shown under an edit tool call. Removed lines are tinted
/// red, added lines green — each with a bright `-`/`+` gutter and a full-row
/// background (filled at wrap time, see `fill_trailing_background`) so a wrapped
/// line still reads as one contiguous block. Unchanged lines render as dim context
/// (no gutter, no tint), and only the genuinely changed lines are flagged — a real
/// line diff (see `diff_lines`), not a blanket remove-all/add-all. Capped so a
/// large rewrite can't flood the transcript. Renders nothing for tools that carry
/// no textual diff (cursor edits with empty args, write_file, etc.).
fn render_edit_diff(lines: &mut Vec<StyledLine>, name: &str, args: &serde_json::Value) {
    const MAX_DIFF_LINES: usize = 14;
    const CONTEXT: usize = 3;
    let pick = |v: &serde_json::Value, k: &str| {
        v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string()
    };
    let pairs: Vec<(String, String)> = match name {
        "edit_file" => {
            let (old, new) = (pick(args, "old_string"), pick(args, "new_string"));
            if old.is_empty() && new.is_empty() {
                vec![]
            } else {
                vec![(old, new)]
            }
        }
        "multi_edit" => args
            .get("edits")
            .and_then(|v| v.as_array())
            .map(|edits| {
                edits
                    .iter()
                    .map(|e| (pick(e, "old_string"), pick(e, "new_string")))
                    .collect()
            })
            .unwrap_or_default(),
        _ => return,
    };

    // Diff each edit, separating successive edits in a multi_edit with a `⋯`.
    let mut rows: Vec<DiffRow> = Vec::new();
    for (old, new) in &pairs {
        let ops = diff_lines(old, new);
        let grouped = group_diff(&ops, CONTEXT);
        if grouped.is_empty() {
            continue;
        }
        if !rows.is_empty() {
            rows.push(DiffRow::Gap);
        }
        rows.extend(grouped);
    }
    if rows.is_empty() {
        return;
    }

    // The tint starts at the text-area edge (no outer indent) so the gutter and
    // every wrapped continuation row share one left edge — a long changed line
    // wraps as a single flush block. Context lines align under the gutter.
    let change_line = |sign: char, text: &str, bg, fg, sign_fg| {
        line_with_plain(vec![
            Span::styled(
                format!(" {sign} "),
                Style::default()
                    .fg(sign_fg)
                    .bg(bg)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(text.to_string(), Style::default().fg(fg).bg(bg)),
        ])
    };
    for row in rows.iter().take(MAX_DIFF_LINES) {
        lines.push(match row {
            DiffRow::Context(text) => line_with_plain(vec![Span::styled(
                format!("   {text}"),
                Style::default().fg(MUTED),
            )]),
            DiffRow::Del(text) => change_line('-', text, DIFF_DEL_BG, DIFF_DEL_FG, DIFF_DEL_SIGN),
            DiffRow::Ins(text) => change_line('+', text, DIFF_ADD_BG, DIFF_ADD_FG, DIFF_ADD_SIGN),
            DiffRow::Gap => line_with_plain(vec![Span::styled(
                "   ⋯".to_string(),
                Style::default().fg(FAINT),
            )]),
        });
    }
    if rows.len() > MAX_DIFF_LINES {
        lines.push(line_with_plain(vec![Span::styled(
            format!("    … (+{} more)", rows.len() - MAX_DIFF_LINES),
            Style::default().fg(FAINT),
        )]));
    }
}

/// Render an `update_plan` checklist card: a "Plan N/M done" header over one
/// line per step, each prefixed by a status glyph (done = green ✔, active =
/// teal ▸, pending = muted ○). Completed steps are dimmed and struck through so
/// the eye lands on what's left. `content` is the JSON array of `{step, status}`.
pub(super) fn render_plan(lines: &mut Vec<StyledLine>, content: &str) {
    let items = serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    if items.is_empty() {
        return;
    }
    let done = items
        .iter()
        .filter(|i| i.get("status").and_then(|s| s.as_str()) == Some("completed"))
        .count();
    lines.push(line_with_plain(vec![
        Span::styled(
            "Plan".to_string(),
            Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {done}/{} done", items.len()),
            Style::default().fg(FAINT),
        ),
    ]));
    for item in &items {
        let step = item.get("step").and_then(|v| v.as_str()).unwrap_or("");
        let status = item
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");
        let (glyph, glyph_color, text_style) = match status {
            "completed" => (
                "✔",
                ASSISTANT,
                Style::default()
                    .fg(FAINT)
                    .add_modifier(Modifier::CROSSED_OUT),
            ),
            "in_progress" => (
                "▸",
                TOOL,
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            _ => ("○", MUTED, Style::default().fg(MUTED)),
        };
        lines.push(line_with_plain(vec![
            Span::styled(format!("  {glyph} "), Style::default().fg(glyph_color)),
            Span::styled(step.to_string(), text_style),
        ]));
    }
}

/// Whether a stored plan card (`content` is the JSON `[{step,status}]` array) has
/// at least one step and every step is `completed`. Drives the pinned-panel
/// lifecycle: a finished plan stays pinned until the next user message clears it.
pub(super) fn plan_all_completed(content: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()
        .and_then(|v| v.as_array().cloned())
        .is_some_and(|items| {
            !items.is_empty()
                && items
                    .iter()
                    .all(|i| i.get("status").and_then(|s| s.as_str()) == Some("completed"))
        })
}

/// Display width of a string (sum of per-character terminal widths).
fn cell_width(s: &str) -> usize {
    s.chars()
        .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
        .sum()
}

/// Distribute `budget` display columns across a table's columns. Keeps natural
/// widths when they already fit; otherwise shrinks the widest columns first —
/// down toward each column's longest word, then below it only if still forced —
/// until the total fits. Always returns widths summing to ≤ `budget`, each ≥ 1.
fn fit_column_widths(natural: &[usize], min_word: &[usize], budget: usize) -> Vec<usize> {
    let ncols = natural.len();
    let mut widths: Vec<usize> = natural.to_vec();
    let total: usize = widths.iter().sum();
    if ncols == 0 || total <= budget {
        return widths;
    }

    // Soft floors: avoid shrinking a column below its longest word while another
    // column still has slack — keeps words intact as long as possible. Cap the
    // floor so one giant token can't starve every other column.
    let floor_cap = (budget / ncols).max(1);
    let floors: Vec<usize> = min_word.iter().map(|&w| w.min(floor_cap).max(1)).collect();

    // Shrink the widest column above its floor, one column at a time, until we fit
    // or every column is at its floor; then, if still over, shrink the widest
    // column above a hard minimum of 1 (this hard-breaks words).
    let mut excess = total - budget;
    for floor in [floors.as_slice(), &vec![1usize; ncols]] {
        while excess > 0 {
            let pick = (0..ncols)
                .filter(|&i| widths[i] > floor[i])
                .max_by_key(|&i| widths[i]);
            let Some(i) = pick else { break };
            widths[i] -= 1;
            excess -= 1;
        }
    }
    widths
}

/// Word-wrap a plain table cell to `width` display columns, hard-breaking any
/// word wider than the column. Whitespace runs collapse to a single space.
fn wrap_cell_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for word in text.split_whitespace() {
        let ww = cell_width(word);
        if ww > width {
            // Oversized word: flush the line, then hard-break the word across rows,
            // carrying the trailing partial chunk into the next line.
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            let chunks = wrap_one_line(word, width);
            let split = chunks.len().saturating_sub(1);
            out.extend(chunks[..split].iter().cloned());
            if let Some(tail) = chunks.last() {
                cur = tail.clone();
                cur_w = cell_width(tail);
            }
            continue;
        }
        let sep = usize::from(!cur.is_empty());
        if cur_w + sep + ww > width {
            out.push(std::mem::take(&mut cur));
            cur = word.to_string();
            cur_w = ww;
        } else {
            if sep == 1 {
                cur.push(' ');
                cur_w += 1;
            }
            cur.push_str(word);
            cur_w += ww;
        }
    }
    if !cur.is_empty() || out.is_empty() {
        out.push(cur);
    }
    out
}

/// Condense a sub-agent `task` for the transcript label: strip the second-person
/// instruction preamble models tend to write ("You are reviewing …", "Your task
/// is to …", "Please …") so the label shows the actual subtask, not boilerplate
/// the model wrote to brief the sub-agent. Leaves a task without a preamble as-is.
fn condense_subagent_task(task: &str) -> String {
    let t = task.trim();
    // Longest / most specific first so a generic prefix doesn't pre-empt a fuller
    // one. Matched case-insensitively against the start (all entries are ASCII, so
    // the byte length is a valid slice boundary on the original-cased string).
    const PREAMBLES: &[&str] = &[
        "you have been asked to ",
        "you are going to ",
        "you are tasked with ",
        "i would like you to ",
        "i'd like you to ",
        "i want you to ",
        "i need you to ",
        "your task is to ",
        "your job is to ",
        "your goal is to ",
        "your task: ",
        "you need to ",
        "you should ",
        "you must ",
        "you will ",
        "you are ",
        "you're ",
        "please ",
        "task: ",
    ];
    let lower = t.to_ascii_lowercase();
    for p in PREAMBLES {
        if let Some(rest) = lower.strip_prefix(p)
            && !rest.trim().is_empty()
        {
            return t[p.len()..].trim().to_string();
        }
    }
    t.to_string()
}

/// One-line argument summary for a tool's `→ verb(...)` line (the salient field).
fn tool_arg_summary(name: &str, args: &serde_json::Value, cwd: &str) -> String {
    let pick = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("");
    match name {
        // File paths: relative to cwd, left-truncated so the basename survives.
        "read_file" | "list_dir" | "write_file" | "edit_file" | "multi_edit" | "delete_file" => {
            display_path(pick("path"), cwd)
        }
        "glob" | "grep" => truncate_chars(pick("pattern"), 60),
        "run_bash" => truncate_chars(pick("command"), 60),
        "web_fetch" => truncate_chars(pick("url"), 60),
        "skill" => truncate_chars(pick("name"), 60),
        // "subagent" is jargon — the delegated task is the salient detail and
        // renders as the label itself (see `render_tool_call`), so surface the
        // task directly (preamble stripped) with no `subagent(...)` wrapper.
        "subagent" => truncate_chars(&condense_subagent_task(pick("task")), 72),
        _ => String::new(),
    }
}

/// A file path for the transcript: made relative to the agent's `cwd` (the footer
/// already shows it), then — if still too wide — truncated from the LEFT on a
/// segment boundary so the basename (what distinguishes sibling files) survives.
fn display_path(path: &str, cwd: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let rel = strip_cwd(path, cwd);
    truncate_path_left(&rel, 56)
}

/// Strip a leading `cwd/` so paths under the working dir render relative; a path
/// equal to the cwd becomes `.`; paths elsewhere are returned unchanged.
fn strip_cwd(path: &str, cwd: &str) -> String {
    let cwd = cwd.trim_end_matches(['/', '\\']);
    if cwd.is_empty() {
        return path.to_string();
    }
    if let Some(rest) = path.strip_prefix(cwd) {
        let rest = rest.trim_start_matches(['/', '\\']);
        return if rest.is_empty() {
            ".".to_string()
        } else {
            rest.to_string()
        };
    }
    path.to_string()
}

/// Truncate from the left, keeping whole trailing path segments (and the
/// basename) under `max` columns, with a leading `…/`.
pub(super) fn truncate_path_left(s: &str, max: usize) -> String {
    // `max` is a column budget — measure display width (CJK / wide glyphs are 2
    // columns each), not char count, or a wide path overflows its slot. Mirrors
    // `cell_width` / `truncate_to_width` above.
    if cell_width(s) <= max {
        return s.to_string();
    }
    let segments: Vec<&str> = s.split(['/', '\\']).filter(|p| !p.is_empty()).collect();
    let mut kept = String::new();
    for seg in segments.iter().rev() {
        let candidate = if kept.is_empty() {
            (*seg).to_string()
        } else {
            format!("{seg}/{kept}")
        };
        // +2 leaves room for the leading `…/` (each glyph one column).
        if cell_width(&candidate) + 2 > max {
            break;
        }
        kept = candidate;
    }
    if kept.is_empty() {
        // A single oversized segment (long basename) — hard left-truncate to the
        // rightmost characters that fit in `max - 1` columns (the `…` takes one).
        let budget = max.saturating_sub(1);
        let mut tail: Vec<char> = Vec::new();
        let mut w = 0usize;
        for c in s.chars().rev() {
            let cw = UnicodeWidthChar::width(c).unwrap_or(0);
            if w + cw > budget {
                break;
            }
            tail.push(c);
            w += cw;
        }
        tail.reverse();
        let tail: String = tail.into_iter().collect();
        return format!("…{tail}");
    }
    format!("…/{kept}")
}

/// Compact `⎿` summary: a line count for multi-line output, else the single line
/// (with the agent's cwd stripped so result paths stay short).
/// Strip ANSI escape sequences and other control characters from text about to
/// be shown in the TUI. Captured command output (`git -c color.ui=always …`,
/// `printf '\033[..'`, anything forced to `--color=always`) can carry escape
/// bytes that ratatui mis-measures (zero/unknown width) and paints as garbage;
/// we only ever want the visible characters. Tabs become a single space so words
/// stay separated.
pub(super) fn strip_ansi_and_controls(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                // CSI: ESC [ … final byte in 0x40..=0x7e.
                Some('[') => {
                    chars.next();
                    for d in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … terminated by BEL or ST (ESC \).
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\x07' {
                            break;
                        }
                        if d == '\x1b' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Lone ESC or a 2-byte escape — drop the following byte.
                _ => {
                    chars.next();
                }
            }
            continue;
        }
        if c == '\t' {
            out.push(' ');
        } else if !c.is_control() {
            out.push(c);
        }
    }
    out
}

fn tool_result_summary(s: &str, cwd: &str, tool: Option<&str>) -> String {
    // Multi-line output (read_file's numbered lines, dir listings, command
    // output) collapses to a clean count — the `→ verb(args)` line above already
    // says what ran, so a peek at the noisy first line (line numbers, etc.) adds
    // nothing. Single-line results are the meaningful outcome ("wrote x", "+1 −1").
    let count = s.lines().count();
    // A subagent's result is its written report; a bare line count ("243 lines")
    // says nothing about what it found and can't be told apart from a sibling's.
    // Preview the first non-empty line (with a `+N more` tail) so each subagent's
    // outcome is legible at a glance.
    if tool == Some("subagent") {
        let first = s
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("");
        let preview = truncate_chars(&strip_ansi_and_controls(first), 56);
        let extra = count.saturating_sub(1);
        return if extra > 0 {
            format!("{preview} (+{extra} more)")
        } else {
            preview
        };
    }
    if count > 1 {
        format!("{count} {}", count_unit(tool, count))
    } else {
        let cwd = cwd.trim_end_matches(['/', '\\']);
        let clean = strip_ansi_and_controls(s.trim());
        let stripped = if cwd.is_empty() {
            clean
        } else {
            clean.replace(&format!("{cwd}/"), "")
        };
        truncate_chars(&stripped, 60)
    }
}

/// Unit for a tool's multi-line result count. File listers/searchers count
/// files/entries/matches, not "lines"; everything else (read_file, run_bash
/// output) is line-oriented.
fn count_unit(tool: Option<&str>, count: usize) -> &'static str {
    let one = count == 1;
    match tool {
        Some("list_dir") => {
            if one {
                "entry"
            } else {
                "entries"
            }
        }
        Some("glob") => {
            if one {
                "file"
            } else {
                "files"
            }
        }
        Some("grep") => {
            if one {
                "match"
            } else {
                "matches"
            }
        }
        _ => {
            if one {
                "line"
            } else {
                "lines"
            }
        }
    }
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let t: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{t}…")
}

/// Render markdown to styled transcript lines. `width` is the text-area width
/// tables are laid out to fit (so they never exceed the pane and get sheared by
/// the line word-wrapper); pass 0 for an unconstrained default. Non-table
/// content is wrapped later by [`wrap_transcript`], so `width` only affects tables.
pub(super) fn render_markdown_lines(content: &str, width: u16) -> Vec<StyledLine> {
    let mut options = MdOptions::empty();
    options.insert(MdOptions::ENABLE_STRIKETHROUGH);
    options.insert(MdOptions::ENABLE_TABLES);
    options.insert(MdOptions::ENABLE_TASKLISTS);
    let parser = Parser::new_ext(content, options);
    let mut renderer = MarkdownRenderer::new(width);

    for event in parser {
        renderer.push_event(event);
    }

    renderer.finish()
}

pub(super) struct MarkdownRenderer {
    lines: Vec<StyledLine>,
    current_spans: Vec<Span<'static>>,
    current_plain: String,
    inline_style: InlineStyle,
    heading: Option<HeadingLevel>,
    quote_depth: usize,
    list_stack: Vec<ListState>,
    item_prefix: Option<String>,
    code_block: Option<CodeFence>,
    /// Accumulates a GFM table until its end, then emits aligned columns.
    table: Option<TableAcc>,
    /// Text-area width (display columns) tables are laid out to fit.
    table_width: usize,
}

/// Buffers a markdown table's cells so it can be rendered as aligned columns
/// (the renderer is otherwise streaming, but a table needs all rows to size).
#[derive(Default)]
struct TableAcc {
    rows: Vec<Vec<String>>,
    cur_row: Vec<String>,
    cur_cell: String,
}

impl MarkdownRenderer {
    fn new(width: u16) -> Self {
        // 0 means "unconstrained" — fall back to a comfortable default so a table
        // rendered outside the transcript (e.g. a direct call) still looks sane.
        let table_width = if width == 0 { 80 } else { usize::from(width) };
        Self {
            lines: Vec::new(),
            current_spans: Vec::new(),
            current_plain: String::new(),
            inline_style: InlineStyle::default(),
            heading: None,
            quote_depth: 0,
            list_stack: Vec::new(),
            item_prefix: None,
            code_block: None,
            table: None,
            table_width,
        }
    }

    fn finish(mut self) -> Vec<StyledLine> {
        self.flush_line();
        self.lines
    }

    fn push_event(&mut self, event: MdEvent<'_>) {
        match event {
            MdEvent::Start(tag) => self.start_tag(tag),
            MdEvent::End(tag) => self.end_tag(tag),
            MdEvent::Text(text) => self.push_text(text.as_ref()),
            MdEvent::Code(text) => self.push_inline_code(text.as_ref()),
            MdEvent::SoftBreak | MdEvent::HardBreak => self.flush_line(),
            MdEvent::Rule => {
                self.flush_line();
                self.lines.push(line_plain(
                    "────────────────────────────────".to_string(),
                    Style::default().fg(FAINT),
                ));
            }
            MdEvent::Html(text) | MdEvent::InlineHtml(text) => self.push_text(text.as_ref()),
            MdEvent::FootnoteReference(text) => self.push_text(text.as_ref()),
            MdEvent::TaskListMarker(checked) => {
                self.ensure_prefix();
                let marker = if checked { "☑ " } else { "☐ " };
                self.push_span(marker.to_string(), Style::default().fg(ACCENT));
            }
            _ => {}
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => {}
            Tag::Heading { level, .. } => {
                self.flush_line();
                self.heading = Some(level);
            }
            Tag::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth += 1;
            }
            Tag::List(start) => {
                self.flush_line();
                self.list_stack.push(ListState::new(start));
            }
            Tag::Item => {
                self.flush_pending();
                self.item_prefix = Some(self.next_item_prefix());
            }
            Tag::CodeBlock(kind) => {
                self.flush_line();
                self.code_block = Some(CodeFence::new(kind));
            }
            Tag::Table(_) => {
                self.flush_line();
                self.table = Some(TableAcc::default());
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.cur_row.clear();
                }
            }
            Tag::TableCell => {
                if let Some(table) = &mut self.table {
                    table.cur_cell.clear();
                }
            }
            Tag::Emphasis => self.inline_style.emphasis += 1,
            Tag::Strong => self.inline_style.strong += 1,
            Tag::Strikethrough => self.inline_style.strike += 1,
            Tag::Link { .. } => self.inline_style.link += 1,
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => {
                self.flush_line();
            }
            TagEnd::Heading(_) => {
                self.flush_line();
                self.heading = None;
                self.lines.push(blank_line());
            }
            TagEnd::BlockQuote(_) => {
                self.flush_line();
                self.quote_depth = self.quote_depth.saturating_sub(1);
                self.lines.push(blank_line());
            }
            TagEnd::List(_) => {
                // `flush_pending` (not `flush_line`) so the list's content end
                // doesn't add its own blank on top of the explicit one below —
                // that double-blanked the gap after a list.
                self.flush_pending();
                self.list_stack.pop();
                self.lines.push(blank_line());
            }
            TagEnd::Item => {
                // No `flush_line`: a loose list's item content was already flushed
                // at `End(Paragraph)`, so flushing here would emit a blank between
                // every item. `flush_pending` keeps tight lists working (their text
                // is still pending) without spacing loose ones out.
                self.flush_pending();
                self.item_prefix = None;
            }
            TagEnd::CodeBlock => {
                if let Some(block) = self.code_block.take() {
                    self.emit_code_block(block);
                    self.lines.push(blank_line());
                }
            }
            TagEnd::TableCell => {
                if let Some(table) = &mut self.table {
                    let cell = table.cur_cell.trim().to_string();
                    table.cur_row.push(cell);
                    table.cur_cell.clear();
                }
            }
            TagEnd::TableHead | TagEnd::TableRow => {
                if let Some(table) = &mut self.table {
                    let row = std::mem::take(&mut table.cur_row);
                    table.rows.push(row);
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.emit_table(table);
                    self.lines.push(blank_line());
                }
            }
            TagEnd::Emphasis => {
                self.inline_style.emphasis = self.inline_style.emphasis.saturating_sub(1)
            }
            TagEnd::Strong => self.inline_style.strong = self.inline_style.strong.saturating_sub(1),
            TagEnd::Strikethrough => {
                self.inline_style.strike = self.inline_style.strike.saturating_sub(1)
            }
            TagEnd::Link => self.inline_style.link = self.inline_style.link.saturating_sub(1),
            _ => {}
        }
    }

    fn emit_code_block(&mut self, block: CodeFence) {
        let label = if block.language.is_empty() {
            "code".to_string()
        } else {
            block.language
        };
        self.lines.push(line_plain(
            format!("  {label}"),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));

        let content = if block.content.is_empty() {
            String::new()
        } else {
            block.content
        };
        for raw_line in content.lines() {
            self.lines.push(line_plain(
                format!("  {raw_line}"),
                Style::default().fg(TEXT),
            ));
        }
        if content.is_empty() || content.ends_with('\n') {
            self.lines
                .push(line_plain("  ".to_string(), Style::default().fg(TEXT)));
        }
    }

    /// Render a buffered GFM table as a bordered, width-aware box (Claude-Code
    /// style): columns are sized to fit the available text width, long cells wrap
    /// across multiple lines instead of overflowing, and the whole table never
    /// exceeds the pane — so the transcript word-wrapper can't shear it apart.
    /// A bold header row sits under a top border, a `├─┼─┤` rule splits it from
    /// the body, and a bottom border closes the box.
    fn emit_table(&mut self, table: TableAcc) {
        let rows = table.rows;
        let Some(ncols) = rows.iter().map(Vec::len).max().filter(|n| *n > 0) else {
            return;
        };

        // Natural (unwrapped) width of each column, plus its widest single word —
        // the column can't shrink below that word without hard-breaking it, so the
        // word width is the soft floor the fair-shrink pass tries to respect.
        let mut natural = vec![0usize; ncols];
        let mut min_word = vec![1usize; ncols];
        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                natural[i] = natural[i].max(cell_width(cell));
                let longest = cell.split_whitespace().map(cell_width).max().unwrap_or(0);
                min_word[i] = min_word[i].max(longest.max(1));
            }
        }

        // Box chrome per row: a `│` on each side and between every column, plus one
        // space of padding on each side of every cell. The rest is column content.
        let chrome = (ncols + 1) + 2 * ncols;
        let budget = self.table_width.saturating_sub(chrome).max(ncols);
        let widths = fit_column_widths(&natural, &min_word, budget);

        let border = Style::default().fg(FAINT);
        let rule = |left: &str, mid: &str, right: &str| -> StyledLine {
            let segs: Vec<String> = widths.iter().map(|w| "─".repeat(w + 2)).collect();
            line_plain(format!("{left}{}{right}", segs.join(mid)), border)
        };

        let last = rows.len().saturating_sub(1);
        self.lines.push(rule("┌", "┬", "┐"));
        for (ri, row) in rows.iter().enumerate() {
            let header = ri == 0;
            // Wrap each cell to its column width; the row is as tall as its tallest cell.
            let cells: Vec<Vec<String>> = (0..ncols)
                .map(|i| wrap_cell_text(row.get(i).map(String::as_str).unwrap_or(""), widths[i]))
                .collect();
            let height = cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
            let cell_style = if header {
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            for k in 0..height {
                let mut spans: Vec<Span<'static>> = vec![Span::styled("│", border)];
                for (i, w) in widths.iter().enumerate() {
                    let text = cells[i].get(k).map(String::as_str).unwrap_or("");
                    let mut padded = String::with_capacity(w + 2);
                    padded.push(' ');
                    padded.push_str(text);
                    padded.push_str(&" ".repeat(w.saturating_sub(cell_width(text))));
                    padded.push(' ');
                    spans.push(Span::styled(padded, cell_style));
                    spans.push(Span::styled("│", border));
                }
                self.lines.push(line_with_plain(spans));
            }
            // Header gets a `├─┼─┤` rule only when a body follows; close with the
            // bottom border after the final row.
            if header && last > 0 {
                self.lines.push(rule("├", "┼", "┤"));
            }
            if ri == last {
                self.lines.push(rule("└", "┴", "┘"));
            }
        }
    }

    fn next_item_prefix(&mut self) -> String {
        if let Some(list) = self.list_stack.last_mut() {
            list.take_prefix()
        } else {
            "• ".to_string()
        }
    }

    fn push_text(&mut self, text: &str) {
        if let Some(block) = &mut self.code_block {
            block.content.push_str(text);
            return;
        }
        if let Some(table) = &mut self.table {
            table.cur_cell.push_str(text);
            return;
        }

        for (idx, part) in text.split('\n').enumerate() {
            if idx > 0 {
                self.flush_line();
            }
            if !part.is_empty() {
                self.ensure_prefix();
                self.push_span(part.to_string(), self.current_style());
            }
        }
    }

    fn push_inline_code(&mut self, text: &str) {
        if let Some(table) = &mut self.table {
            table.cur_cell.push_str(text);
            return;
        }
        self.ensure_prefix();
        // No surrounding padding — the markdown source already supplies spacing
        // and the accent color marks the span; padding caused double spaces.
        self.push_span(
            text.to_string(),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        );
    }

    fn current_style(&self) -> Style {
        let mut style = Style::default().fg(TEXT);
        if self.inline_style.emphasis > 0 {
            style = style.add_modifier(Modifier::ITALIC);
        }
        if self.inline_style.strong > 0 {
            style = style.add_modifier(Modifier::BOLD);
        }
        if self.inline_style.strike > 0 {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.inline_style.link > 0 {
            style = style.fg(LINK).add_modifier(Modifier::UNDERLINED);
        }
        if let Some(level) = self.heading {
            style = heading_style(level);
        }
        style
    }

    fn ensure_prefix(&mut self) {
        if !self.current_spans.is_empty() || !self.current_plain.is_empty() {
            return;
        }
        if self.quote_depth > 0 {
            let prefix = format!("{} ", "▎".repeat(self.quote_depth));
            self.push_span(prefix, Style::default().fg(QUOTE));
        }
        if let Some(prefix) = self.item_prefix.take() {
            self.push_span(prefix, Style::default().fg(ACCENT));
        }
    }

    fn push_span(&mut self, text: String, style: Style) {
        self.current_plain.push_str(&text);
        self.current_spans.push(Span::styled(text, style));
    }

    fn flush_line(&mut self) {
        if self.current_spans.is_empty() {
            if !self.lines.last().is_some_and(|line| line.plain.is_empty()) {
                self.lines.push(blank_line());
            }
            self.current_plain.clear();
            return;
        }
        self.flush_pending();
    }

    /// Emit any pending inline content as a line, but — unlike [`flush_line`] —
    /// never push a paragraph-break blank when there's nothing pending. Used at
    /// list-item boundaries so a *loose* list (whose items pulldown wraps in
    /// paragraphs, leaving `End(Item)` to flush empty) renders tight, with no
    /// blank line between items, matching a tight list.
    fn flush_pending(&mut self) {
        if self.current_spans.is_empty() {
            self.current_plain.clear();
            return;
        }
        let line = StyledLine {
            line: Line::from(std::mem::take(&mut self.current_spans)),
            plain: std::mem::take(&mut self.current_plain),
        };
        self.lines.push(line);
    }
}

#[derive(Default)]
pub(super) struct InlineStyle {
    emphasis: usize,
    strong: usize,
    strike: usize,
    link: usize,
}

pub(super) struct ListState {
    next_number: Option<u64>,
}

impl ListState {
    fn new(start: Option<u64>) -> Self {
        Self { next_number: start }
    }

    fn take_prefix(&mut self) -> String {
        match self.next_number {
            Some(number) => {
                self.next_number = Some(number + 1);
                format!("{number}. ")
            }
            None => "• ".to_string(),
        }
    }
}

pub(super) struct CodeFence {
    language: String,
    content: String,
}

impl CodeFence {
    fn new(kind: CodeBlockKind<'_>) -> Self {
        let language = match kind {
            CodeBlockKind::Indented => String::new(),
            CodeBlockKind::Fenced(name) => name.to_string(),
        };
        Self {
            language,
            content: String::new(),
        }
    }
}

pub(super) fn heading_style(level: HeadingLevel) -> Style {
    match level {
        HeadingLevel::H1 => Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
        HeadingLevel::H2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        HeadingLevel::H3 => Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
        _ => Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    }
}

pub(super) fn blank_line() -> StyledLine {
    line_plain(String::new(), Style::default())
}

pub(super) fn line_plain(text: String, style: Style) -> StyledLine {
    StyledLine {
        plain: text.clone(),
        line: Line::from(Span::styled(text, style)),
    }
}

pub(super) fn line_with_plain(spans: Vec<Span<'static>>) -> StyledLine {
    let mut plain = String::new();
    for span in &spans {
        plain.push_str(span.content.as_ref());
    }
    StyledLine {
        line: Line::from(spans),
        plain,
    }
}

pub(super) fn push_styled_line(lines: &mut Vec<StyledLine>, text: impl Into<String>, style: Style) {
    lines.push(line_plain(text.into(), style));
}

/// Collapse runs of blank lines (and drop trailing blanks), dropping each
/// line's parallel bar color in lockstep so the two vectors stay aligned.
pub(super) fn compact_lines_and_bars(lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
    let mut out_lines = Vec::with_capacity(lines.len());
    let mut out_bars = Vec::with_capacity(bars.len());
    // Start `false` so a single intentional leading blank survives — the
    // transcript intro opens with `EMPTY_STATE_TOP_GAP` so the banner keeps its
    // top padding once a message lands. Runs of blanks still collapse below.
    let mut last_was_blank = false;

    for (line, bar) in lines.drain(..).zip(bars.drain(..)) {
        let is_blank = line.plain.trim().is_empty();
        if is_blank && last_was_blank {
            continue;
        }
        last_was_blank = is_blank;
        out_lines.push(line);
        out_bars.push(bar);
    }

    while out_lines
        .last()
        .is_some_and(|line| line.plain.trim().is_empty())
    {
        out_lines.pop();
        out_bars.pop();
    }

    *lines = out_lines;
    *bars = out_bars;
}

#[cfg(test)]
mod render_tests {
    use super::{condense_subagent_task, strip_ansi_and_controls, tool_result_summary};

    #[test]
    fn condense_subagent_task_strips_instruction_preamble() {
        assert_eq!(
            condense_subagent_task("You are reviewing a SvelteKit chat page component"),
            "reviewing a SvelteKit chat page component"
        );
        assert_eq!(
            condense_subagent_task("Your task is to audit the auth flow"),
            "audit the auth flow"
        );
        assert_eq!(
            condense_subagent_task("Please investigate the crash"),
            "investigate the crash"
        );
        // A task with no preamble is left untouched.
        assert_eq!(
            condense_subagent_task("Review the gateway proxy changes"),
            "Review the gateway proxy changes"
        );
        // Never strip down to nothing — a bare preamble stays as-is.
        assert_eq!(condense_subagent_task("You are"), "You are");
    }

    #[test]
    fn strip_ansi_removes_csi_osc_and_controls() {
        // CSI color codes gone, visible text kept.
        assert_eq!(strip_ansi_and_controls("\x1b[32mok\x1b[0m"), "ok");
        // OSC (title) sequence terminated by BEL is dropped.
        assert_eq!(strip_ansi_and_controls("\x1b]0;title\x07done"), "done");
        // Bare control chars dropped; tab becomes a space.
        assert_eq!(strip_ansi_and_controls("a\x07b\tc"), "ab c");
        // Plain text is untouched.
        assert_eq!(
            strip_ansi_and_controls("nothing special"),
            "nothing special"
        );
    }

    #[test]
    fn single_line_summary_is_sanitized() {
        // A single-line colored result renders with no escape bytes leaking in.
        let out = tool_result_summary("\x1b[31mfatal: bad ref\x1b[0m", "", None);
        assert_eq!(out, "fatal: bad ref");
        assert!(!out.contains('\x1b'));
    }
}
