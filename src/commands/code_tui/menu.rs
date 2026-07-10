use super::*;

// Selected-row background: a cool slate — neutral but on the cold side, so it
// reads as a clean raised surface on the dark terminals it sits on (a warm gray
// here turns muddy against a navy/black background).
pub(super) const SELECT_BG: Color = Color::Rgb(58, 62, 76);
// Primary text + chevron on the bar: near-white, bold — crisp on the slate.
pub(super) const SELECT_TEXT: Color = Color::Rgb(242, 243, 246);
// Secondary text on the bar (endpoint URL, session time, toggle description): a
// soft cool gray, clearly readable but under the bold-white primary label.
pub(super) const SELECT_ACCENT: Color = Color::Rgb(196, 200, 210);

fn picker_content_width(width: u16) -> usize {
    usize::from(width.max(1))
        .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
        .max(1)
}

pub(super) fn filter_slash_commands(query: &str) -> Vec<&'static SlashCommandSpec> {
    if query.is_empty() {
        return SLASH_COMMANDS.iter().collect();
    }

    let mut prefix_matches = Vec::new();
    let mut fuzzy_matches = Vec::new();
    for command in SLASH_COMMANDS {
        if command.name.starts_with(query) || is_alias_target(query, command.name) {
            prefix_matches.push(command);
        } else if matches_fuzzy(query, command.name) {
            fuzzy_matches.push(command);
        }
    }
    prefix_matches.extend(fuzzy_matches);
    prefix_matches
}

/// Whether `query` is a prefix of an alias pointing at `command_name` (so `/qu`
/// surfaces `/exit`).
fn is_alias_target(query: &str, command_name: &str) -> bool {
    SLASH_ALIASES
        .iter()
        .any(|(alias, target)| *target == command_name && alias.starts_with(query))
}

/// Filter discovered skill slash commands by `query` (the text after `/`), prefix
/// matches ranked before fuzzy ones — the same ranking as `filter_slash_commands`.
/// Returns clones so the result can outlive the borrow of `commands`.
pub(super) fn filter_skill_commands(commands: &[SkillCommand], query: &str) -> Vec<SkillCommand> {
    if query.is_empty() {
        return commands.to_vec();
    }
    let mut prefix_matches = Vec::new();
    let mut fuzzy_matches = Vec::new();
    for command in commands {
        if command.name.starts_with(query) {
            prefix_matches.push(command.clone());
        } else if matches_fuzzy(query, &command.name) {
            fuzzy_matches.push(command.clone());
        }
    }
    prefix_matches.extend(fuzzy_matches);
    prefix_matches
}

pub(super) fn collect_attach_path_suggestions(cwd: &str, query: &str) -> Vec<PathMenuEntry> {
    let trimmed = query.trim_start();
    let (dir_part, prefix) = match trimmed.rfind('/') {
        Some(index) => (&trimmed[..=index], &trimmed[index + 1..]),
        None => ("", trimmed),
    };

    let dir_path = {
        let expanded = crate::services::system_env::expand_tilde(dir_part);
        if expanded.is_absolute() {
            expanded
        } else {
            Path::new(cwd).join(dir_part)
        }
    };

    let Ok(read_dir) = std::fs::read_dir(&dir_path) else {
        return Vec::new();
    };

    let mut entries = read_dir
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !prefix.is_empty() && !name.starts_with(prefix) && !matches_fuzzy(prefix, &name) {
                return None;
            }
            let file_type = entry.file_type().ok()?;
            let is_dir = file_type.is_dir();
            let suffix = if is_dir { "/" } else { "" };
            let display_name = format!("{name}{suffix}");
            Some(PathMenuEntry {
                label: display_name.clone(),
                is_dir,
                description: if is_dir { "directory" } else { "file" }.to_string(),
                insertion_text: format!("/attach {dir_part}{display_name}"),
            })
        })
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| {
        // Prefix matches rank above fuzzy-only matches, then dirs before files, then alphabetical.
        let a_prefix = a.label.starts_with(prefix);
        let b_prefix = b.label.starts_with(prefix);
        b_prefix
            .cmp(&a_prefix)
            .then_with(|| b.is_dir.cmp(&a.is_dir))
            .then_with(|| a.label.to_lowercase().cmp(&b.label.to_lowercase()))
    });
    entries.truncate(64);
    entries
}

pub(super) fn command_menu_area(
    composer_area: Rect,
    frame_area: Rect,
    item_count: usize,
    preferred_placement: Option<CommandMenuPlacement>,
) -> (Rect, CommandMenuPlacement) {
    let left_offset = composer_area.x.saturating_sub(frame_area.x);
    let max_width = frame_area.width.saturating_sub(left_offset).max(1);
    let min_width = max_width.min(24);
    let width = composer_area.width.min(max_width).max(min_width);
    let row_count = item_count.clamp(1, COMMAND_MENU_MAX_ROWS) as u16;
    let desired_height = row_count.saturating_add(3);
    let above_space = composer_area.y.saturating_sub(frame_area.y);
    let below_space = frame_area
        .y
        .saturating_add(frame_area.height)
        .saturating_sub(composer_area.y.saturating_add(composer_area.height));
    let placement = preferred_placement.unwrap_or({
        if above_space >= desired_height || above_space >= below_space {
            CommandMenuPlacement::Above
        } else {
            CommandMenuPlacement::Below
        }
    });
    // Cap height to the available space in the chosen direction so the menu never
    // overlaps the composer when space is tight.
    let available = match placement {
        CommandMenuPlacement::Above => above_space,
        CommandMenuPlacement::Below => below_space,
    };
    let height = desired_height
        .min(available.max(4))
        .min(frame_area.height.max(4));
    let y = match placement {
        CommandMenuPlacement::Above => composer_area.y.saturating_sub(height).max(frame_area.y),
        CommandMenuPlacement::Below => composer_area.y.saturating_add(composer_area.height),
    };
    (Rect::new(composer_area.x, y, width, height), placement)
}

pub(super) fn command_menu_item_line(
    label: &str,
    description: &str,
    selected: bool,
    width: u16,
    label_column_width: usize,
) -> Line<'static> {
    // The selected-row chevron sits on the dark transcript, not on SELECT_BG, so
    // it uses a visible gray rather than the dim SELECT_WASH color.
    const SELECT_TEXT: Color = MUTED;
    const COLUMN_GAP: usize = 2;

    let content_width = picker_content_width(width);
    let description_width = content_width
        .saturating_sub(label_column_width)
        .saturating_sub(COLUMN_GAP);
    let rendered_label = truncate_for_display_width(label, label_column_width.max(1));
    let rendered_description = if description_width >= 8 {
        truncate_for_display_width(description, description_width)
    } else {
        String::new()
    };
    let label_padding = label_column_width.saturating_sub(display_width(&rendered_label));
    let description_gap = if rendered_description.is_empty() {
        0
    } else {
        COLUMN_GAP
    };
    let plain = format!(
        "{}{}{}{}",
        rendered_label,
        " ".repeat(label_padding),
        " ".repeat(description_gap),
        rendered_description
    );
    let fill_width = content_width.saturating_sub(display_width(&plain));

    let prefix_style = if selected {
        Style::default()
            .fg(SELECT_TEXT)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(FAINT)
    };
    let label_style = if selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD)
    };
    let description_style = if selected {
        Style::default().fg(TEXT)
    } else {
        Style::default().fg(MUTED)
    };

    let mut spans = vec![Span::styled(
        if selected { "> " } else { "  " },
        prefix_style,
    )];
    spans.push(Span::styled(rendered_label, label_style));
    if label_padding > 0 {
        spans.push(Span::styled(" ".repeat(label_padding), description_style));
    }
    if description_gap > 0 {
        spans.push(Span::styled(" ".repeat(description_gap), description_style));
    }
    if !rendered_description.is_empty() {
        spans.push(Span::styled(rendered_description, description_style));
    }
    if fill_width > 0 {
        spans.push(Span::styled(" ".repeat(fill_width), description_style));
    }
    Line::from(spans)
}

pub(super) fn render_command_menu_rows(
    menu: &VisibleCommandMenu,
    width: u16,
) -> Vec<Line<'static>> {
    if menu.entries.is_empty() {
        return vec![Line::from(Span::styled(
            match menu.kind {
                MenuKind::Commands => "No matching command",
                MenuKind::AttachPath => "No matching path",
            },
            Style::default().fg(MUTED),
        ))];
    }

    let selected = menu.selected.unwrap_or(0);
    let start = if menu.entries.len() <= COMMAND_MENU_MAX_ROWS {
        0
    } else {
        selected
            .saturating_sub(COMMAND_MENU_MAX_ROWS / 2)
            .min(menu.entries.len().saturating_sub(COMMAND_MENU_MAX_ROWS))
    };
    let end = (start + COMMAND_MENU_MAX_ROWS).min(menu.entries.len());
    let content_width = picker_content_width(width);
    let labels: Vec<String> = menu.entries[start..end]
        .iter()
        .map(ComposerMenuEntry::label)
        .collect();
    let label_column_width = labels
        .iter()
        .map(|label| display_width(label))
        .max()
        .unwrap_or(0)
        .min(content_width.saturating_sub(8))
        .max(4);
    menu.entries[start..end]
        .iter()
        .zip(labels.iter())
        .enumerate()
        .map(|(index, (entry, label))| {
            command_menu_item_line(
                label,
                entry.description(),
                start + index == selected,
                width,
                label_column_width,
            )
        })
        .collect()
}

/// Append a ` (current)` marker to a picker label for the in-effect option.
/// Shared by the pickers that highlight the active choice (`/effort`).
pub(super) fn picker_current_label(label: String, is_current: bool) -> String {
    if is_current {
        format!("{label}  (current)")
    } else {
        label
    }
}

pub(super) fn picker_kind_noun(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "keys",
        PickerKind::Model { .. } => "models",
        PickerKind::Session => "sessions",
        PickerKind::Rewind => "turns",
        PickerKind::Effort => "levels",
    }
}

pub(super) fn picker_search_placeholder(kind: &PickerKind) -> &'static str {
    match kind {
        PickerKind::Key => "filter key name or endpoint",
        PickerKind::Model { .. } => "filter model names",
        PickerKind::Session => "filter saved sessions",
        PickerKind::Rewind => "filter turns",
        PickerKind::Effort => "filter levels",
    }
}

pub(super) fn key_search_text(key: &ApiKey) -> String {
    format!(
        "{} {} {}",
        key.id,
        key.display_name(),
        footer_host_label(&key.base_url)
    )
}

pub(super) fn key_picker_item_line(key: &ApiKey, selected: bool, width: u16) -> Line<'static> {
    const SEPARATOR: &str = " · ";

    let name = key.display_name().to_string();
    let endpoint = key.base_url.clone();
    let content_width = picker_content_width(width);
    let separator_width = display_width(SEPARATOR);
    let name_width = display_width(&name);
    let max_endpoint_width = content_width.saturating_sub(name_width + separator_width);

    let (rendered_name, rendered_endpoint) = if max_endpoint_width >= 12 {
        (
            name,
            truncate_for_display_width(&endpoint, max_endpoint_width.max(1)),
        )
    } else {
        let combined = format!("{}{}{}", key.display_name(), SEPARATOR, key.base_url);
        (
            truncate_for_display_width(&combined, content_width),
            String::new(),
        )
    };

    let plain = if rendered_endpoint.is_empty() {
        rendered_name.clone()
    } else {
        format!("{rendered_name}{SEPARATOR}{rendered_endpoint}")
    };
    let fill_width = content_width.saturating_sub(display_width(&plain));

    let fill_style = if selected {
        Style::default().bg(SELECT_BG)
    } else {
        Style::default()
    };
    let prefix_style = if selected {
        fill_style.fg(SELECT_TEXT).add_modifier(Modifier::BOLD)
    } else {
        fill_style
    };
    let name_style = if selected {
        Style::default()
            .fg(SELECT_TEXT)
            .bg(SELECT_BG)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
    };
    let endpoint_style = if selected {
        Style::default().fg(SELECT_ACCENT).bg(SELECT_BG)
    } else {
        Style::default().fg(MUTED)
    };

    let mut spans = vec![Span::styled(
        if selected { "> " } else { "  " },
        prefix_style,
    )];
    spans.push(Span::styled(rendered_name, name_style));
    if !rendered_endpoint.is_empty() {
        spans.push(Span::styled(SEPARATOR, endpoint_style));
        spans.push(Span::styled(rendered_endpoint, endpoint_style));
    }
    spans.push(Span::styled(" ".repeat(fill_width), fill_style));
    Line::from(spans)
}

pub(super) fn session_picker_item_lines(
    preview: &SessionPreview,
    selected: bool,
    armed_delete: bool,
    width: u16,
) -> Vec<Line<'static>> {
    const SELECT_TIME: Color = SELECT_ACCENT;
    const DELETE_BG: Color = Color::Rgb(102, 58, 52);
    const DELETE_TEXT: Color = Color::Rgb(255, 240, 230);
    const DELETE_TIME: Color = Color::Rgb(255, 194, 170);

    let content_width = picker_content_width(width);
    // A narrow (split-left) pane can't spare 8 cols for "12:42 PM" — use the "5m" stamp.
    let time = if content_width < 44 {
        format_time_ago_short(&preview.updated_at)
    } else {
        format_session_time(&preview.updated_at)
    };
    let time_width = display_width(&time);
    let preview_width = content_width
        .saturating_sub(time_width.saturating_add(2))
        .max(1);
    let summary = truncate_for_display_width(&preview.preview_text, preview_width);
    let summary_width = display_width(&summary);
    let gap_width = content_width
        .saturating_sub(summary_width + time_width)
        .max(1);

    let (active_bg, active_text, active_time) = if armed_delete {
        (DELETE_BG, DELETE_TEXT, DELETE_TIME)
    } else {
        (SELECT_BG, SELECT_TEXT, SELECT_TIME)
    };

    let line_style = if selected {
        Style::default()
            .fg(active_text)
            .bg(active_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(TEXT)
    };
    let time_style = if selected {
        Style::default()
            .fg(active_time)
            .bg(active_bg)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(MUTED)
    };
    let fill_style = if selected {
        Style::default().bg(active_bg)
    } else {
        Style::default()
    };

    vec![Line::from(vec![
        Span::styled(
            if armed_delete {
                "! "
            } else if selected {
                "> "
            } else {
                "  "
            },
            if selected {
                fill_style.fg(active_time).add_modifier(Modifier::BOLD)
            } else {
                fill_style
            },
        ),
        Span::styled(summary, line_style),
        Span::styled(" ".repeat(gap_width), fill_style),
        Span::styled(time, time_style),
    ])]
}

pub(super) fn render_session_picker_rows(
    picker: &PickerState,
    max_rows: usize,
    width: u16,
) -> (Vec<Line<'static>>, Vec<Option<usize>>) {
    let filtered = picker.filtered_items();
    if filtered.is_empty() || max_rows == 0 {
        let msg = if picker.items.is_empty() {
            "No saved sessions yet"
        } else {
            "No matches"
        };
        return (
            vec![Line::from(Span::styled(msg, Style::default().fg(MUTED)))],
            Vec::new(),
        );
    }

    let mut all_rows: Vec<(Line<'static>, Option<usize>, bool)> = Vec::new();
    let mut previous_group = String::new();
    for (filtered_index, (_, item)) in filtered.iter().enumerate() {
        let PickerValue::Session(preview) = &item.value else {
            continue;
        };
        let group = format_session_group_label(&preview.updated_at);
        if group != previous_group {
            if !all_rows.is_empty() {
                all_rows.push((Line::from(""), None, false));
            }
            all_rows.push((
                Line::from(Span::styled(
                    group.clone(),
                    Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
                )),
                Some(filtered_index),
                false,
            ));
            previous_group = group;
        }
        let selected = filtered_index == picker.selected;
        let armed_delete = selected && picker.delete_is_armed_for_session(preview);
        for line in session_picker_item_lines(preview, selected, armed_delete, width) {
            all_rows.push((line, Some(filtered_index), true));
        }
    }

    if all_rows.len() <= max_rows {
        let (lines, row_map): (Vec<_>, Vec<_>) = all_rows
            .into_iter()
            .map(|(line, index, _)| (line, index))
            .unzip();
        return (lines, row_map);
    }

    let selected_row = all_rows
        .iter()
        .rposition(|(_, index, is_item)| *is_item && *index == Some(picker.selected))
        .unwrap_or(0);
    let mut start = selected_row.saturating_sub(max_rows / 2);
    let mut end = (start + max_rows).min(all_rows.len());
    if end - start < max_rows {
        start = end.saturating_sub(max_rows);
    }
    while start > 0
        && all_rows[start].2
        && all_rows[start - 1].2
        && all_rows[start].1 == all_rows[start - 1].1
    {
        start -= 1;
        end = (start + max_rows).min(all_rows.len());
    }
    if start > 0 && all_rows[start].1.is_some() && all_rows[start - 1].1.is_none() {
        start -= 1;
        end = (start + max_rows).min(all_rows.len());
    }

    let (lines, row_map): (Vec<_>, Vec<_>) = all_rows[start..end]
        .iter()
        .cloned()
        .map(|(line, index, _)| (line, index))
        .unzip();
    (lines, row_map)
}

pub(super) fn picker_entry_lines(
    item: &PickerEntry,
    selected: bool,
    width: u16,
) -> Vec<Line<'static>> {
    match &item.value {
        PickerValue::Session(preview) => session_picker_item_lines(preview, selected, false, width),
        PickerValue::Key(key) => vec![key_picker_item_line(key, selected, width)],
        _ => {
            let content_width = usize::from(width.max(1))
                .saturating_sub(PICKER_ROW_PREFIX_WIDTH)
                .max(1);
            let label = truncate_for_display_width(&item.label, content_width);
            let fill_width = content_width.saturating_sub(display_width(&label));
            let fill_style = if selected {
                Style::default().bg(SELECT_BG)
            } else {
                Style::default()
            };
            let prefix_style = if selected {
                fill_style.fg(SELECT_TEXT).add_modifier(Modifier::BOLD)
            } else {
                fill_style
            };
            let label_style = if selected {
                Style::default()
                    .fg(SELECT_TEXT)
                    .bg(SELECT_BG)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT)
            };
            vec![Line::from(vec![
                Span::styled(if selected { "> " } else { "  " }, prefix_style),
                Span::styled(label, label_style),
                Span::styled(" ".repeat(fill_width), fill_style),
            ])]
        }
    }
}

/// Minimum shell-inner width for the two-pane split modals; narrower stays single-pane.
pub(super) const SPLIT_MIN_INNER_WIDTH: u16 = 76;
const SPLIT_LEFT_PCT: u16 = 40;

pub(super) fn split_capable(area: Rect) -> bool {
    area.width.saturating_sub(4) >= SPLIT_MIN_INNER_WIDTH
}

/// (left list, 3-col rule gutter, right detail) panes of a split overlay body.
pub(super) fn split_columns(body: Rect) -> (Rect, Rect, Rect) {
    let left_w = body.width * SPLIT_LEFT_PCT / 100;
    let rule_w = 3u16.min(body.width.saturating_sub(left_w));
    let right_w = body.width.saturating_sub(left_w + rule_w);
    (
        Rect {
            width: left_w,
            ..body
        },
        Rect {
            x: body.x + left_w,
            width: rule_w,
            ..body
        },
        Rect {
            x: body.x + left_w + rule_w,
            width: right_w,
            ..body
        },
    )
}

/// The overlay rect + whether the split layout fits (else the classic narrow rect).
pub(super) fn split_overlay_area(
    body: Rect,
    wide_pct: u16,
    wide_h_pct: u16,
    narrow_pct: u16,
    narrow_h_pct: u16,
) -> (Rect, bool) {
    let wide = centered_rect(wide_pct, wide_h_pct, body);
    if split_capable(wide) {
        (wide, true)
    } else {
        (centered_rect(narrow_pct, narrow_h_pct, body), false)
    }
}

pub(super) fn centered_rect(width_pct: u16, height_pct: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_pct) / 2),
            Constraint::Percentage(height_pct),
            Constraint::Percentage((100 - height_pct) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_pct) / 2),
            Constraint::Percentage(width_pct),
            Constraint::Percentage((100 - width_pct) / 2),
        ])
        .split(vertical[1])[1]
}

/// Wrap the composer `draft` into visual rows, soft-wrapping each logical line
/// (split on `\n`) at `text_width` display columns. Returns each row's
/// `[start, end)` byte range into `draft`. Every visual row carries the same
/// 2-col left margin in the render (the `> ` prompt on row 0, a hanging indent
/// elsewhere), so callers wrap at `composer_width - COMPOSER_PREFIX_WIDTH`.
///
/// Wraps at word boundaries: an overflowing word moves whole to the next row,
/// its leading space(s) left on the closed row so rows stay contiguous. A word
/// wider than `text_width` falls back to a hard mid-word wrap.
///
/// Always returns at least one row (an empty draft → a single empty row), and a
/// trailing `\n` yields a final empty row so the cursor can rest on the new line.
/// This is the single wrap model shared by rendering, cursor placement, mouse
/// hit-testing, and visual-line cursor movement — so they never disagree.
pub(super) fn composer_visual_rows(draft: &str, text_width: usize) -> Vec<(usize, usize)> {
    let text_width = text_width.max(1);
    let mut rows = Vec::new();
    let mut line_start = 0usize;
    loop {
        let rel_nl = draft[line_start..].find('\n');
        let line_end = rel_nl.map_or(draft.len(), |idx| line_start + idx);
        let mut row_start = line_start;
        let mut col = 0usize;
        let mut pos = line_start;
        // Start of the current word (a break candidate); `None` → hard-wrap mid-word.
        let mut word_start: Option<usize> = None;
        let mut prev_space = false;
        for ch in draft[line_start..line_end].chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            let is_space = ch == ' ' || ch == '\t';
            if !is_space && prev_space {
                word_start = Some(pos);
            }
            if ch_width > 0 && col + ch_width > text_width {
                match word_start.filter(|&ws| ws > row_start) {
                    Some(ws) => {
                        // Move the overflowing word down, re-measuring its consumed part.
                        rows.push((row_start, ws));
                        row_start = ws;
                        col = draft[ws..pos]
                            .chars()
                            .map(|c| UnicodeWidthChar::width(c).unwrap_or(0))
                            .sum();
                    }
                    None => {
                        rows.push((row_start, pos));
                        row_start = pos;
                        col = 0;
                    }
                }
                word_start = None;
            }
            col += ch_width;
            pos += ch.len_utf8();
            prev_space = is_space;
        }
        rows.push((row_start, line_end));
        match rel_nl {
            Some(idx) => line_start += idx + 1,
            None => break,
        }
    }
    rows
}

/// The cursor's `(visual row, display column)` within `rows`. The column counts
/// the 2-col prompt indent every row carries. At a soft-wrap boundary the cursor
/// belongs to the *start* of the next row (the last row whose start ≤ cursor).
pub(super) fn composer_cursor_rowcol(
    draft: &str,
    cursor: usize,
    rows: &[(usize, usize)],
) -> (usize, usize) {
    let cursor = cursor.min(draft.len());
    let row = rows
        .iter()
        .rposition(|(start, _)| *start <= cursor)
        .unwrap_or(0);
    let (start, _) = rows[row.min(rows.len().saturating_sub(1))];
    let mut col = usize::from(COMPOSER_PREFIX_WIDTH);
    for ch in draft[start..cursor.max(start)].chars() {
        col += UnicodeWidthChar::width(ch).unwrap_or(0);
    }
    (row, col)
}

/// Inverse of [`composer_cursor_rowcol`]: the byte offset on visual `row` nearest
/// to display column `target_col` (used by mouse clicks and visual up/down). Lands
/// on the last character boundary at or before `target_col`.
pub(super) fn composer_offset_for_col(
    draft: &str,
    rows: &[(usize, usize)],
    row: usize,
    target_col: usize,
) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let (start, end) = rows[row.min(rows.len() - 1)];
    let prefix = usize::from(COMPOSER_PREFIX_WIDTH);
    if target_col <= prefix {
        return start;
    }
    let mut col = prefix;
    let mut pos = start;
    for ch in draft[start..end].chars() {
        let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
        if col + ch_width > target_col {
            break;
        }
        col += ch_width;
        pos += ch.len_utf8();
    }
    pos
}

/// Cursor `(x, y)` for the composer, in the same hanging-indent wrap model the
/// composer renders with. `y` is the visual row; `x` includes the prompt indent.
pub(super) fn cursor_position(
    text: &str,
    cursor: usize,
    width: u16,
    line_prefix_width: u16,
) -> (u16, u16) {
    let prefix = usize::from(line_prefix_width);
    let text_width = usize::from(width).saturating_sub(prefix).max(1);
    let rows = composer_visual_rows(text, text_width);
    let (row, col) = composer_cursor_rowcol(text, cursor, &rows);
    let x = col.min(usize::from(width).saturating_sub(1).max(prefix));
    (x as u16, row as u16)
}
