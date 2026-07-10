use super::*;

impl CodeTuiApp {
    pub(super) fn render_command_menu(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        menu: &VisibleCommandMenu,
    ) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(FAINT));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 1,
        });
        let footer_height = 1u16;
        let rows_area = Rect {
            x: inner.x,
            y: inner.y,
            width: inner.width,
            height: inner.height.saturating_sub(footer_height),
        };
        let footer_area = Rect::new(
            inner.x,
            inner.y + inner.height.saturating_sub(footer_height),
            inner.width,
            footer_height,
        );

        let lines = render_command_menu_rows(menu, rows_area.width);
        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            rows_area,
        );

        let footer_text = if menu.entries.is_empty() {
            "Esc close · Enter submit"
        } else if menu.kind == MenuKind::AttachPath {
            "Esc close · Enter/Tab insert · ↑/↓ navigate"
        } else {
            "Esc close · Enter run · Tab insert · ↑/↓ navigate"
        };
        frame.render_widget(
            Paragraph::new(footer_text).style(Style::default().fg(MUTED)),
            footer_area,
        );
    }

    pub(super) fn render_picker(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        picker: &PickerState,
        split: bool,
    ) -> OverlayRenderOut {
        if matches!(picker.kind, PickerKind::Session) {
            return self.render_session_picker(frame, area, picker, split);
        }

        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(FAINT));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(6),
                Constraint::Length(1),
            ])
            .split(inner);

        let filtered_count = picker.filtered_items().len();
        let total_count = picker.items.len();
        let status_label = if picker.loading {
            "loading · esc".to_string()
        } else {
            format!(
                "{} · esc",
                format_picker_match_count(
                    filtered_count,
                    total_count,
                    picker_kind_noun(&picker.kind)
                )
            )
        };
        let status_width = display_width(&status_label) as u16;
        let title_width = display_width(picker.title) as u16;
        let middle_padding = chunks[0]
            .width
            .saturating_sub(title_width + status_width)
            .max(1);
        let header = Line::from(vec![
            Span::styled(
                picker.title,
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" ".repeat(usize::from(middle_padding))),
            Span::styled(status_label, Style::default().fg(MUTED)),
        ]);
        frame.render_widget(
            Paragraph::new(header),
            Rect::new(chunks[0].x, chunks[0].y, chunks[0].width, 1),
        );
        let search_line = if picker.query.is_empty() {
            Line::from(vec![
                Span::styled("/ ", Style::default().fg(MUTED)),
                Span::styled(
                    picker_search_placeholder(&picker.kind),
                    Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled("/ ", Style::default().fg(MUTED)),
                Span::styled(
                    picker.query.clone(),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
            ])
        };
        frame.render_widget(
            Paragraph::new(search_line),
            Rect::new(chunks[0].x, chunks[0].y + 1, chunks[0].width, 1),
        );

        if picker.loading {
            frame.render_widget(
                Paragraph::new("Loading available models…").style(Style::default().fg(MUTED)),
                chunks[1],
            );
            return OverlayRenderOut::default();
        }

        let visible = picker.visible_items(usize::from(chunks[1].height));
        let (lines, row_to_filtered_index) = if visible.is_empty() {
            (
                vec![Line::from(Span::styled(
                    "No matches",
                    Style::default().fg(MUTED),
                ))],
                Vec::new(),
            )
        } else {
            let mut lines = Vec::new();
            let mut row_to_filtered_index = Vec::new();

            for (filtered_index, item) in visible {
                let item_lines =
                    picker_entry_lines(item, filtered_index == picker.selected, chunks[1].width);
                row_to_filtered_index.extend(std::iter::repeat_n(filtered_index, item_lines.len()));
                lines.extend(item_lines);
            }

            (lines, row_to_filtered_index)
        };

        self.picker_hitbox = Some(PickerHitbox {
            overlay_area: area,
            list_area: chunks[1],
            row_to_filtered_index: row_to_filtered_index.into_iter().map(Some).collect(),
        });

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new("Type to filter · Up/Down wrap · Enter open · Esc close")
                .style(Style::default().fg(MUTED)),
            chunks[2],
        );
        OverlayRenderOut::default()
    }

    pub(super) fn render_session_picker(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        picker: &PickerState,
        split: bool,
    ) -> OverlayRenderOut {
        // Same chrome as /skills and /mcp: title + count badge in the top border.
        let badge = format!(
            "{} · esc",
            format_session_match_count(picker.filtered_items().len(), picker.items.len())
        );
        let inner = overlay_shell(frame, area, "Sessions", Some((badge, MUTED)));
        if inner.height == 0 {
            return OverlayRenderOut::default();
        }
        // Split: list left, conversation preview right, full-width footer strip.
        let (list_pane, preview_pane) = if split && inner.height > 1 {
            let body = Rect {
                height: inner.height - 1,
                ..inner
            };
            let (left, rule, right) = split_columns(body);
            render_vertical_rule(frame, rule);
            (left, Some(right))
        } else {
            (inner, None)
        };
        // Search row + a breathing gap, mirroring the toggle overlays' body.
        let gap = u16::from(list_pane.height >= 8);
        let chunks = if preview_pane.is_some() {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(gap),
                    Constraint::Min(6),
                ])
                .split(list_pane)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(gap),
                    Constraint::Min(6),
                    Constraint::Length(1),
                ])
                .split(list_pane)
        };
        let footer_rect = if preview_pane.is_some() {
            Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            }
        } else {
            chunks[3]
        };

        frame.render_widget(
            Paragraph::new(search_input_line(
                &picker.query,
                "filter sessions, keys, models",
            )),
            chunks[0],
        );

        let (lines, row_to_filtered_index) =
            render_session_picker_rows(picker, usize::from(chunks[2].height), chunks[2].width);

        self.picker_hitbox = Some(PickerHitbox {
            overlay_area: area,
            list_area: chunks[2],
            row_to_filtered_index,
        });

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            chunks[2],
        );
        let footer_text = if picker.pending_delete.is_some() {
            "Enter or Ctrl+D confirm delete · Esc cancel"
        } else if preview_pane.is_some() {
            "Type to filter · Up/Down wrap · Enter open · PgUp preview · Ctrl+D delete"
        } else {
            "Type to filter · Up/Down wrap · Enter open · Ctrl+D delete"
        };
        frame.render_widget(
            Paragraph::new(footer_text).style(Style::default().fg(MUTED)),
            footer_rect,
        );

        let Some(right) = preview_pane else {
            return OverlayRenderOut::default();
        };
        let selected =
            picker
                .filtered_items()
                .get(picker.selected)
                .and_then(|(_, item)| match &item.value {
                    PickerValue::Session(preview) => Some(preview.clone()),
                    _ => None,
                });
        let Some(preview) = selected else {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "no session selected",
                    Style::default().fg(MUTED),
                )),
                right,
            );
            return OverlayRenderOut {
                detail_area: Some(right),
                ..Default::default()
            };
        };
        // A stale entry (updated_at mismatch) reads as absent; the tick reloads it.
        let entry = self
            .session_preview_cache
            .get(&preview.session_id)
            .filter(|entry| entry.updated_at == preview.updated_at);
        let scroll_up = if picker.preview_scroll_for.as_deref() == Some(preview.session_id.as_str())
        {
            picker.preview_scroll
        } else {
            0
        };
        let clamped = render_session_preview_pane(frame, right, &preview, entry, scroll_up);
        OverlayRenderOut {
            detail_scroll: Some(clamped),
            detail_area: Some(right),
            scroll_for: Some(preview.session_id),
        }
    }

    /// `/help` overlay: an at-a-glance reference of every slash command (grouped
    /// by purpose), the keybindings (grouped: compose / edit / navigate / select /
    /// session), and the literal-text escapes. Discovered skills are intentionally
    /// left out — they live in the dedicated `/skills` overlay. The body is taller
    /// than the box on most terminals, so it scrolls via [`render_detail_lines`];
    /// `scroll` is the caller-held offset and the clamped value is returned to be
    /// written back.
    pub(super) fn render_help_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        scroll: u16,
    ) -> u16 {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(FAINT))
            .title(Span::styled(
                "Help",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });

        let cmd_style = Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD);
        let key_style = Style::default().fg(ACCENT).add_modifier(Modifier::BOLD);
        let section_style = Style::default().fg(MUTED).add_modifier(Modifier::BOLD);
        let group_style = Style::default().fg(FAINT);

        let mut lines: Vec<Line> = Vec::new();

        // Intro — what the surface is, in one line (the footer already notes Esc).
        lines.push(Line::from(Span::styled(
            "Message the model, or type a command. Enter sends.",
            Style::default().fg(TEXT),
        )));

        // --- Slash commands, grouped by purpose. ---
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Slash commands", section_style)));
        // One column width across every command so the groups stay aligned.
        let cmd_col = SLASH_COMMANDS
            .iter()
            .map(|c| display_width(c.help_label))
            .max()
            .unwrap_or(0);
        let mut shown: Vec<&str> = Vec::new();
        for (group, names) in HELP_COMMAND_GROUPS {
            // Drop commands hidden for the active key; skip empty groups whole.
            let visible: Vec<&SlashCommandSpec> = names
                .iter()
                .filter_map(|name| SLASH_COMMANDS.iter().find(|c| c.name == *name))
                .filter(|c| self.slash_command_visible(c.name))
                .collect();
            if visible.is_empty() {
                continue;
            }
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(format!("  {group}"), group_style)));
            for command in visible {
                shown.push(command.name);
                lines.push(help_kv_row(
                    command.help_label,
                    command.description,
                    cmd_col,
                    cmd_style,
                ));
            }
        }
        // Completeness guard: a command missing from every group above (e.g. one
        // added later) still shows here rather than silently vanishing.
        let leftover: Vec<_> = SLASH_COMMANDS
            .iter()
            .filter(|c| !shown.contains(&c.name) && self.slash_command_visible(c.name))
            .collect();
        if !leftover.is_empty() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("  More", group_style)));
            for command in leftover {
                lines.push(help_kv_row(
                    command.help_label,
                    command.description,
                    cmd_col,
                    cmd_style,
                ));
            }
        }

        // --- Keybindings, grouped. ---
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Keybindings", section_style)));
        let key_col = HELP_KEYBINDINGS
            .iter()
            .flat_map(|(_, rows)| rows.iter())
            .map(|(key, _)| display_width(key))
            .max()
            .unwrap_or(0);
        for (group, rows) in HELP_KEYBINDINGS {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(format!("  {group}"), group_style)));
            for (key, desc) in *rows {
                lines.push(help_kv_row(key, desc, key_col, key_style));
            }
        }

        // --- Literal-text escapes. ---
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Text entry", section_style)));
        lines.push(Line::from(""));
        for note in [
            "!cmd runs a shell command locally (output not sent to the model)",
            "//text sends a literal leading slash · !!text a literal !",
        ] {
            lines.push(Line::from(Span::styled(
                format!("  {note}"),
                Style::default().fg(MUTED),
            )));
        }

        render_detail_lines(frame, inner, lines, scroll, "Esc close")
    }

    /// `/context` overlay: fill bar + segment legend + free space, then the injected
    /// `-c` block's full text beneath a divider when one was supplied.
    pub(super) fn render_context_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        report: &crate::agent::engine::ContextReport,
        scroll: u16,
    ) -> u16 {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(FAINT))
            .title(Span::styled(
                "Context",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(shell, area);

        let inner = area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 2,
        });
        let width = usize::from(inner.width).max(1);

        // Anchor the split to the footer's measured fill so the two match.
        let (fill, is_estimate) = self.context_fill();
        let mut report = report.clone();
        if !is_estimate {
            report.rescale(fill);
        }
        let report = &report;

        let window = u64::from(report.context_window);
        let known = window > 0;
        let used = report.used();

        // Injected sits next to the system prompt it's folded into.
        let mut segs: Vec<(String, u64, Color)> = Vec::new();
        segs.push(("System prompt".to_string(), report.system_prompt, ASSISTANT));
        if report.injected_context > 0 {
            segs.push((
                "Injected context".to_string(),
                report.injected_context,
                WARNING,
            ));
        }
        segs.push((format!("Tools · {}", report.tool_count), report.tools, TOOL));
        if report.mcp_tool_count > 0 {
            segs.push((
                format!("MCP tools · {}", report.mcp_tool_count),
                report.mcp_tools,
                USER,
            ));
        }
        segs.push((
            format!("Messages · {}", report.message_count),
            report.messages,
            ACCENT,
        ));

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled(
            self.raw_model.clone(),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        if known {
            let pct = used.saturating_mul(100) / window.max(1);
            lines.push(Line::from(Span::styled(
                format!(
                    "{} / {} · {}%",
                    format_token_count_value(used),
                    format_token_count_value(window),
                    pct
                ),
                Style::default()
                    .fg(context_fill_color(pct))
                    .add_modifier(Modifier::BOLD),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                format!(
                    "{} tokens · context window unknown",
                    format_token_count_value(used)
                ),
                Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
            )));
        }

        lines.push(context_fill_bar(&segs, used, window, width.clamp(8, 52)));
        lines.push(Line::from(""));

        let label_w = segs
            .iter()
            .map(|(l, _, _)| l.len())
            .chain(std::iter::once("Free".len()))
            .max()
            .unwrap_or(12);
        let pct_of = |tokens: u64| -> String {
            if !known {
                return "—".to_string();
            }
            let pct = tokens.saturating_mul(100) / window.max(1);
            if pct == 0 && tokens > 0 {
                "<1%".to_string()
            } else {
                format!("{pct}%")
            }
        };
        for (label, tokens, color) in &segs {
            lines.push(context_legend_row(
                "■",
                label,
                *tokens,
                &pct_of(*tokens),
                *color,
                label_w,
            ));
        }
        if known {
            let free = report.free();
            lines.push(context_legend_row(
                "□",
                "Free",
                free,
                &pct_of(free),
                FAINT,
                label_w,
            ));
        }

        lines.push(Line::from(""));
        let note = if is_estimate {
            "Figures are estimates (chars/4); the model's real fill may differ."
        } else {
            "Total is the last measured prompt; the split is a chars/4 estimate."
        };
        lines.push(Line::from(Span::styled(note, Style::default().fg(FAINT))));

        if let Some(text) = self.injected_context.as_deref() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "─".repeat(width),
                Style::default().fg(FAINT),
            )));
            lines.push(Line::from(Span::styled(
                "Injected context",
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            )));
            if let Some(summary) = &self.injected_context_summary {
                lines.push(Line::from(Span::styled(
                    summary.clone(),
                    Style::default().fg(MUTED),
                )));
            }
            lines.push(Line::from(Span::styled(
                "Background awareness only — the model won't reference this unless it's relevant.",
                Style::default().fg(FAINT),
            )));
            lines.push(Line::from(""));
            for raw in text.split('\n') {
                if raw.trim().is_empty() {
                    lines.push(Line::from(""));
                    continue;
                }
                for wrapped in wrap_chars(raw, width) {
                    lines.push(Line::from(Span::styled(wrapped, Style::default().fg(TEXT))));
                }
            }
        }

        render_detail_lines(frame, inner, lines, scroll, "Esc close")
    }

    /// `/skills` overlay: a checkbox toggle list of the discovered skills. Split
    /// shows the highlighted skill's full text in the right pane; narrow keeps
    /// the Tab drill-in. Chrome shared with `/mcp`.
    pub(super) fn render_skills_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &SkillsOverlay,
        split: bool,
    ) -> OverlayRenderOut {
        // Narrow-only Tab drill-in; the split's right pane replaces it.
        if !split && let Some(item) = state.viewing.and_then(|i| state.items.get(i)) {
            let inner = overlay_shell(frame, area, "Skills", None);
            if inner.height == 0 {
                return OverlayRenderOut {
                    detail_scroll: Some(state.detail_scroll),
                    ..Default::default()
                };
            }
            return OverlayRenderOut {
                detail_scroll: Some(self.render_skill_detail(
                    frame,
                    inner,
                    item,
                    usize::from(inner.width).max(1),
                    state.detail_scroll,
                    "Esc back",
                )),
                ..Default::default()
            };
        }

        let filtered = state.filtered_indices();
        let enabled = state.items.iter().filter(|i| i.enabled).count();
        // The search line doubles as the add-input field while adding and as the
        // delete-confirm prompt while a Ctrl+D is armed.
        let input_line = if let Some(buffer) = &state.adding {
            add_input_line(buffer)
        } else if let Some(name) = state
            .pending_delete
            .and_then(|i| state.items.get(i))
            .map(|i| i.name.as_str())
        {
            Line::from(Span::styled(
                format!("Delete “{name}”?  ^D confirm · Esc cancel"),
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            ))
        } else {
            search_input_line(&state.query, "filter skills")
        };

        let inner = overlay_shell(
            frame,
            area,
            "Skills",
            count_badge(state.adding.is_none(), enabled, state.items.len()),
        );
        if inner.height == 0 {
            return OverlayRenderOut::default();
        }
        // Split: two panes over a full-width footer strip (hints don't fit the left pane).
        let (list_pane, split_panes) = if split && inner.height > 1 {
            let footer_rect = Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            };
            let body = Rect {
                height: inner.height - 1,
                ..inner
            };
            let (left, rule, right) = split_columns(body);
            render_vertical_rule(frame, rule);
            (left, Some((right, footer_rect)))
        } else {
            (inner, None)
        };

        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let footer: Vec<(&str, &str)>;
        if state.adding.is_some() {
            rows.extend([
                Line::from(Span::styled(
                    "name [description], or github:owner/repo · GitHub URL · path to install",
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    "e.g.  changelog Summarize the git log   ·   github:anthropics/skills",
                    Style::default().fg(FAINT),
                )),
                Line::from(Span::styled(
                    "-p / --project installs into ./.agents/skills (shared via the repo)",
                    Style::default().fg(FAINT),
                )),
            ]);
            footer = vec![("Enter", "save"), ("Esc", "cancel")];
        } else if state.items.is_empty() {
            rows.push(Line::from(Span::styled(
                "No skills yet — add one with ^A, or drop a <name>/SKILL.md folder in:",
                Style::default().fg(MUTED),
            )));
            for path in [
                "~/.config/aivo/skills",
                "~/.agents/skills",
                ".agents/skills",
                ".aivo/skills",
            ] {
                rows.push(Line::from(Span::styled(
                    format!("  {path}"),
                    Style::default().fg(FAINT),
                )));
            }
            footer = vec![("^A", "add"), ("Esc", "close")];
        } else if filtered.is_empty() {
            rows.push(Line::from(Span::styled(
                format!("No skills match “{}”", state.query),
                Style::default().fg(MUTED),
            )));
            footer = vec![("Esc", "close")];
        } else {
            let inner_width = usize::from(list_pane.width).saturating_sub(2).max(1);
            for (pos, &i) in filtered.iter().enumerate() {
                let item = &state.items[i];
                // Advert first: the stored description is full text, the row wants one line.
                let desc = truncate_for_display_width(
                    &crate::agent::skills::advert_description(&item.description),
                    toggle_detail_room(inner_width),
                );
                if i == state.selected {
                    // Anchor scrolling to the item's second (description) line so
                    // both of its lines stay visible.
                    selected_pos = rows.len() + 1;
                }
                rows.extend(toggle_list_rows(
                    item.enabled,
                    &item.name,
                    &desc,
                    MUTED,
                    i == state.selected,
                    inner_width,
                ));
                if pos + 1 < filtered.len() {
                    rows.push(Line::from(""));
                }
            }
            footer = vec![
                ("↑↓", "move"),
                ("Space", "toggle"),
                ("Tab", "view"),
                ("^A", "add"),
                ("^D", "remove"),
            ];
        }

        if !self.agent_capable() {
            rows.push(Line::from(""));
            rows.push(Line::from(Span::styled(
                "Skills run with the native agent (plain API keys); this key uses a different backend.",
                Style::default().fg(MUTED),
            )));
        }

        // Animated install-progress row while a background install runs.
        if let Some(progress) = &self.installing_skill {
            let spinner = spinner_frame_indexed(self.frame_tick, self.reduce_motion);
            rows.insert(
                0,
                Line::from(Span::styled(
                    format!("{spinner} {}", progress.status_text()),
                    Style::default().fg(ACCENT),
                )),
            );
            selected_pos += 1; // the inserted row shifts the scroll anchor down one
        }

        let selected_item = state
            .items
            .get(state.selected)
            .filter(|_| state.adding.is_none() && filtered.contains(&state.selected));
        // One-line detail only in narrow — the split's right pane carries the full version.
        let detail = if split_panes.is_none() {
            selected_item.map(|item| (self.skill_detail_text(item), MUTED))
        } else {
            None
        };
        // Split: Tab is gone (the pane is always on) — swap its hint for the scroll key.
        let (body_footer, strip_footer): (Vec<_>, Vec<_>) = match split_panes {
            Some(_) => (
                Vec::new(),
                footer
                    .into_iter()
                    .map(|hint| {
                        if hint == ("Tab", "view") {
                            ("PgDn", "scroll")
                        } else {
                            hint
                        }
                    })
                    .collect(),
            ),
            None => (footer, Vec::new()),
        };

        render_toggle_list_body(
            frame,
            list_pane,
            ToggleListView {
                title: "Skills",
                badge: None,
                input_line,
                rows,
                selected_pos,
                detail,
                footer: body_footer,
            },
        );

        let Some((right, footer_rect)) = split_panes else {
            return OverlayRenderOut::default();
        };
        render_footer_hints(frame, footer_rect, &strip_footer);
        let detail_scroll = match selected_item {
            Some(item) => Some(self.render_skill_detail(
                frame,
                right,
                item,
                usize::from(right.width).max(1),
                state.detail_scroll,
                "",
            )),
            None => {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "no skill selected",
                        Style::default().fg(MUTED),
                    )),
                    right,
                );
                None
            }
        };
        OverlayRenderOut {
            detail_scroll,
            detail_area: Some(right),
            scroll_for: None,
        }
    }

    /// The skill-install picker (empty items = loading). Space marks, Enter
    /// applies, Esc cancels; chrome shared with `/skills`.
    pub(super) fn render_skill_install_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &SkillInstallOverlay,
        split: bool,
    ) -> OverlayRenderOut {
        // Narrow-only Tab drill-in; the split's right pane replaces it.
        if !split && let Some(item) = state.viewing.and_then(|i| state.items.get(i)) {
            let inner = overlay_shell(frame, area, "Install skills", None);
            if inner.height == 0 {
                return OverlayRenderOut {
                    detail_scroll: Some(state.detail_scroll),
                    ..Default::default()
                };
            }
            return OverlayRenderOut {
                detail_scroll: Some(self.render_install_pick_detail(
                    frame,
                    inner,
                    &state.source,
                    item,
                    usize::from(inner.width).max(1),
                    state.detail_scroll,
                    "Esc back",
                )),
                ..Default::default()
            };
        }

        let filtered = state.filtered_indices();
        let marked = state.items.iter().filter(|i| i.checked).count();
        let badge = (!state.items.is_empty()).then(|| {
            let color = if marked > 0 { ASSISTANT } else { MUTED };
            (format!("{marked}/{} marked", state.items.len()), color)
        });
        let inner = overlay_shell(frame, area, "Install skills", badge);
        if inner.height == 0 {
            return OverlayRenderOut::default();
        }
        // Split: two panes over a full-width footer strip, same as `/skills`.
        // Loading stays single-pane — nothing to preview.
        let (list_pane, split_panes) = if split && inner.height > 1 && !state.items.is_empty() {
            let footer_rect = Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            };
            let body = Rect {
                height: inner.height - 1,
                ..inner
            };
            let (left, rule, right) = split_columns(body);
            render_vertical_rule(frame, rule);
            (left, Some((right, footer_rect)))
        } else {
            (inner, None)
        };

        let inner_width = usize::from(list_pane.width).saturating_sub(2).max(1);
        let mut rows: Vec<Line> = Vec::new();
        // Source + destination header, in the loading state too — where an
        // install lands (user vs project) should be visible before Enter.
        if !state.source.is_empty() {
            rows.push(Line::from(Span::styled(
                truncate_for_display_width(&format!("from {}", state.source), inner_width),
                Style::default().fg(FAINT),
            )));
            let dest = if state.project {
                "into ./.agents/skills (project)"
            } else {
                "into ~/.config/aivo/skills (user)"
            };
            rows.push(Line::from(Span::styled(dest, Style::default().fg(FAINT))));
            rows.push(Line::from(""));
        }
        let mut selected_pos = 0usize;
        let footer: Vec<(&str, &str)>;
        if state.items.is_empty() {
            if let Some(progress) = &self.installing_skill {
                let spinner = spinner_frame_indexed(self.frame_tick, self.reduce_motion);
                rows.push(Line::from(Span::styled(
                    format!("{spinner} {}", progress.status_text()),
                    Style::default().fg(ACCENT),
                )));
                rows.push(Line::from(""));
                rows.push(Line::from(Span::styled(
                    "Skills found in the source will appear here to pick from.",
                    Style::default().fg(MUTED),
                )));
            } else {
                rows.push(Line::from(Span::styled(
                    "Nothing to install.",
                    Style::default().fg(MUTED),
                )));
            }
            footer = vec![("Esc", "cancel")];
        } else if filtered.is_empty() {
            rows.push(Line::from(Span::styled(
                format!("No skills match “{}”", state.query),
                Style::default().fg(MUTED),
            )));
            footer = vec![("Esc", "cancel")];
        } else {
            for (pos, &i) in filtered.iter().enumerate() {
                let item = &state.items[i];
                let advert = crate::agent::skills::advert_description(&item.description);
                let (desc, desc_color) = if item.installed && item.checked {
                    (format!("will update · {advert}"), ACCENT)
                } else if item.installed {
                    (format!("installed — Space to update · {advert}"), FAINT)
                } else {
                    (advert, MUTED)
                };
                let desc = truncate_for_display_width(&desc, toggle_detail_room(inner_width));
                if i == state.selected {
                    // Anchor scrolling to the item's second (description) line so
                    // both of its lines stay visible.
                    selected_pos = rows.len() + 1;
                }
                rows.extend(toggle_list_rows(
                    item.checked,
                    &item.name,
                    &desc,
                    desc_color,
                    i == state.selected,
                    inner_width,
                ));
                if pos + 1 < filtered.len() {
                    rows.push(Line::from(""));
                }
            }
            footer = vec![
                ("↑↓", "move"),
                ("Space", "mark"),
                ("Enter", "install"),
                ("Esc", "cancel"),
                ("^A", "all"),
                ("Tab", "view"),
            ];
        }

        let selected_item = state
            .items
            .get(state.selected)
            .filter(|_| filtered.contains(&state.selected));
        // One-line detail only in narrow — the split's right pane carries the full version.
        let detail = if split_panes.is_none() {
            selected_item.map(|item| {
                if item.installed && !item.checked {
                    (
                        "already installed — Space marks it for update".to_string(),
                        FAINT,
                    )
                } else if marked > 0 {
                    (format!("Enter applies the {marked} marked"), MUTED)
                } else {
                    ("Enter installs the highlighted skill".to_string(), MUTED)
                }
            })
        } else {
            None
        };
        let (body_footer, strip_footer): (Vec<_>, Vec<_>) = match split_panes {
            Some(_) => (
                Vec::new(),
                footer
                    .into_iter()
                    .map(|hint| {
                        if hint == ("Tab", "view") {
                            ("PgDn", "scroll")
                        } else {
                            hint
                        }
                    })
                    .collect(),
            ),
            None => (footer, Vec::new()),
        };

        render_toggle_list_body(
            frame,
            list_pane,
            ToggleListView {
                title: "Install skills",
                badge: None,
                input_line: search_input_line(&state.query, "filter skills"),
                rows,
                selected_pos,
                detail,
                footer: body_footer,
            },
        );

        let Some((right, footer_rect)) = split_panes else {
            return OverlayRenderOut::default();
        };
        render_footer_hints(frame, footer_rect, &strip_footer);
        let detail_scroll = match selected_item {
            Some(item) => Some(self.render_install_pick_detail(
                frame,
                right,
                &state.source,
                item,
                usize::from(right.width).max(1),
                state.detail_scroll,
                "",
            )),
            None => {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "no skill selected",
                        Style::default().fg(MUTED),
                    )),
                    right,
                );
                None
            }
        };
        OverlayRenderOut {
            detail_scroll,
            detail_area: Some(right),
            scroll_for: None,
        }
    }

    /// `/config` overlay: a small fixed toggle list of chat preferences, sharing
    /// the `/skills` and `/mcp` chrome via [`render_toggle_list`] so all three feel
    /// the same. No filter/add/remove — the top row is a static heading instead of
    /// a search field.
    pub(super) fn render_config_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &ConfigOverlay,
    ) {
        let on = state
            .items
            .iter()
            .filter(|i| self.config_setting_enabled(i.setting))
            .count();
        let input_line = Line::from(Span::styled(
            "Settings — remembered across sessions",
            Style::default().fg(MUTED),
        ));

        let inner_width = usize::from(area.width).saturating_sub(6).max(1);
        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        for (pos, item) in state.items.iter().enumerate() {
            let desc =
                truncate_for_display_width(item.description, toggle_detail_room(inner_width));
            if pos == state.selected {
                selected_pos = rows.len() + 1;
            }
            rows.extend(toggle_list_rows(
                self.config_setting_enabled(item.setting),
                item.label,
                &desc,
                MUTED,
                pos == state.selected,
                inner_width,
            ));
            if pos + 1 < state.items.len() {
                rows.push(Line::from(""));
            }
        }

        render_toggle_list(
            frame,
            area,
            ToggleListView {
                title: "Config",
                badge: count_badge(true, on, state.items.len()),
                input_line,
                rows,
                selected_pos,
                detail: None,
                footer: vec![("↑↓", "move"), ("Enter/Space", "toggle"), ("Esc", "close")],
            },
        );
    }

    /// The Ctrl+T drill-in: one MCP server's tools as a toggle list. Toggling
    /// only changes what's advertised to the agent; the connection is untouched.
    pub(super) fn render_mcp_tools_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &McpToolsOverlay,
    ) {
        let on = state.items.iter().filter(|i| i.enabled).count();
        let input_line = search_input_line(&state.query, "type to filter tools…");
        let inner_width = usize::from(area.width).saturating_sub(6).max(1);
        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let filtered = state.filtered_indices();
        if filtered.is_empty() {
            rows.push(Line::from(Span::styled(
                if state.items.is_empty() {
                    "This server exposes no tools.".to_string()
                } else {
                    format!("No tools match \"{}\".", state.query)
                },
                Style::default().fg(MUTED),
            )));
        }
        for (pos, &i) in filtered.iter().enumerate() {
            let item = &state.items[i];
            let desc = item
                .description
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let desc = truncate_for_display_width(&desc, toggle_detail_room(inner_width));
            if i == state.selected {
                selected_pos = rows.len() + 1;
            }
            rows.extend(toggle_list_rows(
                item.enabled,
                &item.name,
                &desc,
                MUTED,
                i == state.selected,
                inner_width,
            ));
            if pos + 1 < filtered.len() {
                rows.push(Line::from(""));
            }
        }
        let title = format!("{} tools", state.server);
        render_toggle_list(
            frame,
            area,
            ToggleListView {
                title: &title,
                badge: count_badge(true, on, state.items.len()),
                input_line,
                rows,
                selected_pos,
                detail: None,
                footer: vec![("↑↓", "move"), ("Space", "toggle"), ("Esc", "back")],
            },
        );
    }

    /// The multi-server paste picker: choose which pasted servers to add. An
    /// existing name needs an explicit mark and replaces that entry in place.
    pub(super) fn render_mcp_paste_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &McpPasteOverlay,
    ) {
        let marked = state.items.iter().filter(|i| i.checked).count();
        let input_line = search_input_line(&state.query, "type to filter…");
        let inner_width = usize::from(area.width).saturating_sub(6).max(1);
        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let filtered = state.filtered_indices();
        if filtered.is_empty() {
            rows.push(Line::from(Span::styled(
                format!("No servers match \"{}\".", state.query),
                Style::default().fg(MUTED),
            )));
        }
        for (pos, &i) in filtered.iter().enumerate() {
            let item = &state.items[i];
            let detail = if item.exists {
                format!("configured — Space replaces · {}", item.display)
            } else {
                item.display.clone()
            };
            let detail = truncate_for_display_width(&detail, toggle_detail_room(inner_width));
            if i == state.selected {
                selected_pos = rows.len() + 1;
            }
            rows.extend(toggle_list_rows(
                item.checked,
                &item.name,
                &detail,
                if item.exists { ACCENT } else { MUTED },
                i == state.selected,
                inner_width,
            ));
            if pos + 1 < filtered.len() {
                rows.push(Line::from(""));
            }
        }
        let title = if state.project {
            "Add MCP servers → ./.mcp.json"
        } else {
            "Add MCP servers"
        };
        render_toggle_list(
            frame,
            area,
            ToggleListView {
                title,
                badge: Some((
                    format!("{marked}/{} marked", state.items.len()),
                    if marked > 0 { ACCENT } else { MUTED },
                )),
                input_line,
                rows,
                selected_pos,
                detail: None,
                footer: vec![
                    ("↑↓", "move"),
                    ("Space", "mark"),
                    ("^A", "all"),
                    ("Enter", "add"),
                    ("Esc", "back"),
                ],
            },
        );
    }

    /// One-line detail for the selected `/skills` row: where the skill lives (home
    /// dir abbreviated to `~`) and, for a repo skill, a `project` tag — so the user
    /// knows where to edit it and that `d` won't delete it.
    fn skill_detail_text(&self, item: &SkillToggle) -> String {
        use crate::agent::skills::SkillScope;
        let path = abbreviate_home_path(&item.dir);
        match item.scope {
            SkillScope::Project => format!("project · {path}"),
            SkillScope::User => path,
        }
    }

    /// `/mcp` overlay: a checkbox toggle list of the configured servers, status
    /// colored by health. Split shows the highlighted server's command + tools
    /// in the right pane; narrow keeps the Tab drill-in. Chrome shared with `/skills`.
    pub(super) fn render_mcp_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &McpOverlay,
        split: bool,
    ) -> OverlayRenderOut {
        // Narrow-only Tab drill-in; the split's right pane replaces it.
        if !split && let Some(row) = state.viewing.and_then(|i| state.items.get(i)) {
            let inner = overlay_shell(frame, area, "MCP servers", None);
            if inner.height == 0 {
                return OverlayRenderOut {
                    detail_scroll: Some(state.detail_scroll),
                    ..Default::default()
                };
            }
            return OverlayRenderOut {
                detail_scroll: Some(self.render_mcp_detail(
                    frame,
                    inner,
                    row,
                    usize::from(inner.width).max(1),
                    state.detail_scroll,
                    "Esc back",
                )),
                ..Default::default()
            };
        }

        let filtered = state.filtered_indices();
        let on = state.items.iter().filter(|i| i.enabled).count();
        // The search line doubles as the add-input field while adding and as the
        // delete-confirm prompt while a Ctrl+D is armed (same as /skills).
        let input_line = if let Some(buffer) = &state.adding {
            add_input_line(buffer)
        } else if let Some(name) = state
            .pending_delete
            .and_then(|i| state.items.get(i))
            .map(|i| i.name.as_str())
        {
            Line::from(Span::styled(
                format!("Delete “{name}”?  ^D confirm · Esc cancel"),
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            ))
        } else {
            search_input_line(&state.query, "filter servers")
        };

        let inner = overlay_shell(
            frame,
            area,
            "MCP servers",
            count_badge(state.adding.is_none(), on, state.items.len()),
        );
        if inner.height == 0 {
            return OverlayRenderOut::default();
        }
        // Split: two panes over a full-width footer strip (hints don't fit the left pane).
        let (list_pane, split_panes) = if split && inner.height > 1 {
            let footer_rect = Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            };
            let body = Rect {
                height: inner.height - 1,
                ..inner
            };
            let (left, rule, right) = split_columns(body);
            render_vertical_rule(frame, rule);
            (left, Some((right, footer_rect)))
        } else {
            (inner, None)
        };

        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let footer: Vec<(&str, &str)>;
        if state.adding.is_some() {
            rows.extend([
                Line::from(Span::styled(
                    "command args… or a https:// URL (name derived), or Ctrl+V a JSON block · -p → project .mcp.json",
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    "e.g.  npx -y @modelcontextprotocol/server-filesystem ~",
                    Style::default().fg(FAINT),
                )),
            ]);
            footer = vec![("Enter", "save"), ("Esc", "cancel")];
        } else if state.items.is_empty() {
            rows.push(Line::from(Span::styled(
                "No servers yet — add one with ^A, or drop an \"mcpServers\" entry in:",
                Style::default().fg(MUTED),
            )));
            for path in ["~/.config/aivo/mcp.json", ".mcp.json"] {
                rows.push(Line::from(Span::styled(
                    format!("  {path}"),
                    Style::default().fg(FAINT),
                )));
            }
            footer = vec![("^A", "add"), ("Esc", "close")];
        } else if filtered.is_empty() {
            rows.push(Line::from(Span::styled(
                format!("No servers match “{}”", state.query),
                Style::default().fg(MUTED),
            )));
            footer = vec![("Esc", "close")];
        } else {
            let inner_width = usize::from(list_pane.width).saturating_sub(2).max(1);
            for (pos, &i) in filtered.iter().enumerate() {
                let item = &state.items[i];
                let status =
                    truncate_for_display_width(&item.status, toggle_detail_room(inner_width));
                if i == state.selected {
                    selected_pos = rows.len() + 1;
                }
                rows.extend(toggle_list_rows(
                    item.enabled,
                    &item.name,
                    &status,
                    mcp_health_color(item.health),
                    i == state.selected,
                    inner_width,
                ));
                if pos + 1 < filtered.len() {
                    rows.push(Line::from(""));
                }
            }
            footer = vec![
                ("↑↓", "move"),
                ("Space", "toggle"),
                ("Tab", "view"),
                ("^A", "add"),
                ("^D", "rm"),
                ("^O", "auth"),
                ("^R", "retry"),
                ("^T", "tools"),
            ];
        }

        if !self.agent_capable() {
            rows.push(Line::from(""));
            rows.push(Line::from(Span::styled(
                "MCP tools run with the native agent (plain API keys); this key uses a different backend.",
                Style::default().fg(MUTED),
            )));
        }

        let selected_row = state
            .items
            .get(state.selected)
            .filter(|_| state.adding.is_none() && filtered.contains(&state.selected));
        // One-line detail only in narrow — the split's right pane carries the full version.
        let detail = if split_panes.is_none() {
            selected_row.map(|row| (self.mcp_detail_text(row), mcp_health_color(row.health)))
        } else {
            None
        };
        // Split: Tab is gone (the pane is always on) — swap its hint for the scroll key.
        let (body_footer, strip_footer): (Vec<_>, Vec<_>) = match split_panes {
            Some(_) => (
                Vec::new(),
                footer
                    .into_iter()
                    .map(|hint| {
                        if hint == ("Tab", "view") {
                            ("PgDn", "scroll")
                        } else {
                            hint
                        }
                    })
                    .collect(),
            ),
            None => (footer, Vec::new()),
        };

        render_toggle_list_body(
            frame,
            list_pane,
            ToggleListView {
                title: "MCP servers",
                badge: None,
                input_line,
                rows,
                selected_pos,
                detail,
                footer: body_footer,
            },
        );

        let Some((right, footer_rect)) = split_panes else {
            return OverlayRenderOut::default();
        };
        render_footer_hints(frame, footer_rect, &strip_footer);
        let detail_scroll = match selected_row {
            Some(row) => Some(self.render_mcp_detail(
                frame,
                right,
                row,
                usize::from(right.width).max(1),
                state.detail_scroll,
                "",
            )),
            None => {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "no server selected",
                        Style::default().fg(MUTED),
                    )),
                    right,
                );
                None
            }
        };
        OverlayRenderOut {
            detail_scroll,
            detail_area: Some(right),
            scroll_for: None,
        }
    }

    /// One-line detail for the selected `/mcp` row: its tool names when connected,
    /// the full (un-truncated) error when it failed, the connecting/disabled state
    /// otherwise — plus a scope tag for project servers (which `d` can't remove).
    fn mcp_detail_text(&self, row: &McpServerRow) -> String {
        use crate::agent::mcp::ServerScope;
        let mut parts: Vec<String> = Vec::new();
        if row.scope == ServerScope::Project {
            parts.push("project (.mcp.json)".to_string());
        }
        if !row.enabled {
            parts.push("disabled · Space to enable".to_string());
        } else if let Some(names) = self
            .mcp_client
            .as_ref()
            .and_then(|c| c.tool_names(&row.name))
        {
            parts.push(if names.is_empty() {
                "connected · no tools".to_string()
            } else {
                format!("tools: {}", names.join(", "))
            });
        } else if let Some(err) = self
            .mcp_client
            .as_ref()
            .and_then(|c| c.error_for(&row.name))
        {
            parts.push(format!("failed: {err}"));
        } else {
            parts.push("connecting…".to_string());
        }
        format!("{} · {}", row.name, parts.join(" · "))
    }

    /// `/mcp` detail (split right pane / narrow drill-in): command, status, and
    /// tool list or full error; returns the clamped scroll.
    fn render_mcp_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        row: &McpServerRow,
        width: usize,
        scroll: u16,
        esc_label: &str,
    ) -> u16 {
        use crate::agent::mcp::ServerScope;
        let scope = match row.scope {
            ServerScope::User => "user",
            ServerScope::Project => "project · .mcp.json",
            ServerScope::Pack => "pack (managed by `aivo code packs`)",
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(
                row.name.clone(),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  [{scope}]"), Style::default().fg(MUTED)),
        ])];
        // `row.command` holds the launch command line for stdio servers and the
        // endpoint URL for HTTP ones; the transport kind rides on the row.
        let label = if row.remote { "url" } else { "command" };
        for chunk in wrap_chars(&format!("{label}: {}", row.command), width) {
            lines.push(Line::from(Span::styled(chunk, Style::default().fg(MUTED))));
        }
        lines.push(Line::from(Span::styled(
            truncate_for_display_width(&format!("status: {}", row.status), width),
            Style::default().fg(mcp_health_color(row.health)),
        )));
        lines.push(Line::from(""));
        if let Some(tools) = self
            .mcp_client
            .as_ref()
            .and_then(|c| c.tool_details(&row.name))
        {
            if tools.is_empty() {
                lines.push(Line::from(Span::styled(
                    "Connected, but this server exposes no tools.",
                    Style::default().fg(MUTED),
                )));
            } else {
                let tools: Vec<(&str, &str, bool)> = tools
                    .into_iter()
                    .map(|(t, d)| {
                        let on = !self
                            .disabled_mcp_tools
                            .contains(&crate::agent::mcp::qualified_name(&row.name, t));
                        (t, d, on)
                    })
                    .collect();
                let off = tools.iter().filter(|(_, _, on)| !on).count();
                // Estimate excludes toggled-off tools — they aren't advertised.
                let cost = self
                    .mcp_client
                    .as_ref()
                    .and_then(|c| c.estimated_tokens(&row.name, &self.disabled_mcp_tools))
                    .map(|t| format!(" · ~{} tokens/turn", humanize_count(t)))
                    .unwrap_or_default();
                let counts = if off > 0 {
                    format!("{} on · {off} off", tools.len() - off)
                } else {
                    tools.len().to_string()
                };
                lines.push(Line::from(Span::styled(
                    format!("Tools ({counts}){cost}:"),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                )));
                lines.push(Line::from(""));
                lines.extend(mcp_tool_lines(&tools, width));
            }
        } else if let Some(err) = self
            .mcp_client
            .as_ref()
            .and_then(|c| c.error_for(&row.name))
        {
            lines.push(Line::from(Span::styled(
                "Failed to connect:",
                Style::default().fg(ERROR).add_modifier(Modifier::BOLD),
            )));
            for chunk in wrap_chars(err, width) {
                lines.push(Line::from(Span::styled(chunk, Style::default().fg(ERROR))));
            }
        } else if !row.enabled {
            lines.push(Line::from(Span::styled(
                "Disabled — Space in the list to enable.",
                Style::default().fg(MUTED),
            )));
        } else {
            lines.push(Line::from(Span::styled(
                "Connecting… tools appear here once the handshake completes.",
                Style::default().fg(MUTED),
            )));
        }
        render_detail_lines(frame, area, lines, scroll, esc_label)
    }

    /// `/skills` detail (split right pane / narrow drill-in): location, full
    /// description, and the SKILL.md instructions; returns the clamped scroll.
    fn render_skill_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        item: &SkillToggle,
        width: usize,
        scroll: u16,
        esc_label: &str,
    ) -> u16 {
        use crate::agent::skills::SkillScope;
        let scope = match item.scope {
            SkillScope::User => "user",
            SkillScope::Project => "project",
        };
        self.render_skill_doc_detail(
            frame,
            area,
            &item.name,
            &format!("[{scope}{}]", if item.enabled { "" } else { " · off" }),
            &abbreviate_home_path(&item.dir),
            &item.description,
            &item.body,
            width,
            scroll,
            esc_label,
        )
    }

    /// Picker detail — located by source; the staged dir is a meaningless temp path.
    #[allow(clippy::too_many_arguments)]
    fn render_install_pick_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        source: &str,
        item: &InstallPickItem,
        width: usize,
        scroll: u16,
        esc_label: &str,
    ) -> u16 {
        let tag = if item.installed && item.checked {
            "[marked for update]"
        } else if item.installed {
            "[already installed]"
        } else if item.checked {
            "[marked]"
        } else {
            ""
        };
        self.render_skill_doc_detail(
            frame,
            area,
            &item.name,
            tag,
            &format!("from {source}"),
            &item.description,
            &item.body,
            width,
            scroll,
            esc_label,
        )
    }

    /// Shared layout behind the `/skills` and install-picker detail panes.
    #[allow(clippy::too_many_arguments)]
    fn render_skill_doc_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        name: &str,
        tag: &str,
        location: &str,
        description: &str,
        body: &str,
        width: usize,
        scroll: u16,
        esc_label: &str,
    ) -> u16 {
        let mut title = vec![Span::styled(
            name.to_string(),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )];
        if !tag.is_empty() {
            title.push(Span::styled(format!("  {tag}"), Style::default().fg(MUTED)));
        }
        let mut lines = vec![
            Line::from(title),
            Line::from(Span::styled(
                truncate_for_display_width(location, width),
                Style::default().fg(MUTED),
            )),
            Line::from(""),
        ];
        if !description.is_empty() {
            for chunk in wrap_chars(description, width) {
                lines.push(Line::from(Span::styled(chunk, Style::default().fg(TEXT))));
            }
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            "Instructions:",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        if body.trim().is_empty() {
            lines.push(Line::from(Span::styled(
                "  (empty — edit SKILL.md to add instructions)",
                Style::default().fg(FAINT),
            )));
        } else {
            // The body IS markdown — render it like the transcript, not dim raw lines.
            lines.push(Line::from(""));
            let mut body_lines = render_markdown_lines(body, width as u16);
            while body_lines
                .first()
                .is_some_and(|l| l.plain.trim().is_empty())
            {
                body_lines.remove(0);
            }
            for styled in body_lines {
                for row in wrap_styled_line(&styled.line.spans, width) {
                    lines.push(row.line);
                }
            }
        }
        render_detail_lines(frame, area, lines, scroll, esc_label)
    }
}

/// Columns a toggle item's `[✓] ` checkbox occupies; the description line is
/// indented by the same amount so it aligns under the name.
const TOGGLE_CHECKBOX_WIDTH: usize = 4;

/// Display columns available to a toggle item's description / status line (the
/// full inner width minus the checkbox-aligned indent).
fn toggle_detail_room(inner_width: usize) -> usize {
    inner_width.saturating_sub(TOGGLE_CHECKBOX_WIDTH).max(8)
}

/// A pre-built view of a `/skills` or `/mcp` toggle overlay, handed to
/// [`render_toggle_list`] which owns the shared chrome and layout so the two
/// overlays stay pixel-identical apart from their content.
struct ToggleListView<'a> {
    title: &'a str,
    /// `Some((label, color))` shown right-aligned in the top border (e.g. `3/5 on`).
    badge: Option<(String, Color)>,
    /// The top input row: the search field, the add field, or a delete prompt.
    input_line: Line<'a>,
    /// The list body: styled item rows, empty-state hints, or add hints.
    rows: Vec<Line<'a>>,
    /// Line index within `rows` to keep on screen, so the list scrolls to follow
    /// the selection (each item spans two lines plus a separator).
    selected_pos: usize,
    /// One-line detail for the selected item, shown just above the footer.
    detail: Option<(String, Color)>,
    /// `(key, label)` hints rendered along the footer.
    footer: Vec<(&'a str, &'a str)>,
}

/// Draw the rounded modal frame with a bold title (left) and an optional
/// status badge (right) on the top border, returning the padded inner rect.
fn overlay_shell(
    frame: &mut Frame<'_>,
    area: Rect,
    title: &str,
    badge: Option<(String, Color)>,
) -> Rect {
    frame.render_widget(Clear, area);
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(FAINT))
        .title_top(Line::from(Span::styled(
            format!(" {title} "),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
    if let Some((label, color)) = badge {
        block = block.title_top(
            Line::from(Span::styled(
                format!(" {label} "),
                Style::default().fg(color),
            ))
            .right_aligned(),
        );
    }
    frame.render_widget(block, area);
    area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 2,
    })
}

/// Shell + body for a whole-modal toggle overlay (`/config`); `/skills` and
/// `/mcp` draw their own shell and call [`render_toggle_list_body`] directly.
fn render_toggle_list(frame: &mut Frame<'_>, area: Rect, mut view: ToggleListView) {
    let inner = overlay_shell(frame, area, view.title, view.badge.take());
    render_toggle_list_body(frame, inner, view);
}

/// The toggle overlay's body (input row, list anchored to the selection, detail
/// line, footer) inside an already-drawn shell; `view.title`/`badge` unused here.
fn render_toggle_list_body(frame: &mut Frame<'_>, inner: Rect, view: ToggleListView) {
    if inner.height == 0 {
        return;
    }
    let width = usize::from(inner.width).max(1);

    frame.render_widget(Paragraph::new(view.input_line), Rect { height: 1, ..inner });

    let footer_h = u16::from(!view.footer.is_empty());
    let detail_h = u16::from(view.detail.is_some());
    // A blank row between the input line and the list when there's room to spare.
    let gap = u16::from(inner.height >= 8);
    let list_area = Rect {
        y: inner.y + 1 + gap,
        height: inner.height.saturating_sub(1 + gap + detail_h + footer_h),
        ..inner
    };
    if list_area.height > 0 {
        let view_h = usize::from(list_area.height);
        let offset = view.selected_pos.saturating_sub(view_h.saturating_sub(1));
        frame.render_widget(
            Paragraph::new(Text::from(view.rows)).scroll((offset as u16, 0)),
            list_area,
        );
    }

    if let Some((text, color)) = view.detail {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                truncate_for_display_width(&text, width),
                Style::default().fg(color),
            ))),
            Rect {
                y: inner.y + inner.height - 1 - footer_h,
                height: 1,
                ..inner
            },
        );
    }

    if footer_h == 1 {
        render_footer_hints(
            frame,
            Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            },
            &view.footer,
        );
    }
}

fn render_footer_hints(frame: &mut Frame<'_>, area: Rect, hints: &[(&str, &str)]) {
    frame.render_widget(Paragraph::new(footer_hints(hints)), area);
}

/// The `│` divider between a split overlay's panes, centered in its gutter rect.
fn render_vertical_rule(frame: &mut Frame<'_>, area: Rect) {
    if area.width == 0 {
        return;
    }
    let lines: Vec<Line> = (0..area.height)
        .map(|_| Line::from(Span::styled("│", Style::default().fg(FAINT))))
        .collect();
    frame.render_widget(
        Paragraph::new(Text::from(lines)),
        Rect {
            x: area.x + area.width / 2,
            width: 1,
            ..area
        },
    );
}

/// The two lines for one toggle-list item: a `[✓]`/`[ ]` checkbox plus the bold
/// name on the first line, and the indented detail (a skill's description or a
/// server's status, already truncated to one line) on the second. `detail_color`
/// applies when enabled; a disabled item dims to `FAINT`; the selected item gets
/// a full-width highlight across both lines.
fn toggle_list_rows(
    enabled: bool,
    name: &str,
    detail: &str,
    detail_color: Color,
    selected: bool,
    width: usize,
) -> Vec<Line<'static>> {
    let check = if enabled { "[✓] " } else { "[ ] " };
    let indent = " ".repeat(TOGGLE_CHECKBOX_WIDTH);
    if selected {
        // Keep hierarchy on the bar: bold name, softer description, accent ✓ survives.
        let bar = Style::default().bg(SELECT_BG);
        vec![
            Line::from(vec![
                Span::styled(check, bar.fg(if enabled { ACCENT } else { SELECT_ACCENT })),
                Span::styled(
                    pad_to_width(name, width.saturating_sub(TOGGLE_CHECKBOX_WIDTH)),
                    bar.fg(SELECT_TEXT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                pad_to_width(&format!("{indent}{detail}"), width),
                bar.fg(SELECT_ACCENT),
            )),
        ]
    } else if enabled {
        vec![
            Line::from(vec![
                Span::styled(check, Style::default().fg(ACCENT)),
                // Neutral name: the accent checkmark alone carries "enabled".
                Span::styled(
                    name.to_string(),
                    Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                format!("{indent}{detail}"),
                Style::default().fg(detail_color),
            )),
        ]
    } else {
        // Name a step above the description so a long disabled tail stays scannable.
        vec![
            Line::from(vec![
                Span::styled(check, Style::default().fg(FAINT)),
                Span::styled(name.to_string(), Style::default().fg(MUTED)),
            ]),
            Line::from(Span::styled(
                format!("{indent}{detail}"),
                Style::default().fg(FAINT),
            )),
        ]
    }
}

/// The search input row: an accent `/` prompt then the live query (with a caret),
/// or a muted italic placeholder when the query is empty.
fn search_input_line(query: &str, placeholder: &str) -> Line<'static> {
    if query.is_empty() {
        Line::from(vec![
            Span::styled("/ ", Style::default().fg(MUTED)),
            Span::styled(
                placeholder.to_string(),
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("/ ", Style::default().fg(MUTED)),
            Span::styled(
                query.to_string(),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            ),
            Span::styled("▏", Style::default().fg(ACCENT)),
        ])
    }
}

/// The add-input row shared by both overlays: a `+` prompt, the buffer, a caret.
fn add_input_line(buffer: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            "+ ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(buffer.to_string(), Style::default().fg(TEXT)),
        Span::styled("▏", Style::default().fg(ACCENT)),
    ])
}

/// The right-aligned `N/M on` border badge, or `None` while adding / when empty.
fn count_badge(show: bool, on: usize, total: usize) -> Option<(String, Color)> {
    if !show || total == 0 {
        return None;
    }
    let color = if on > 0 { ASSISTANT } else { MUTED };
    Some((format!("{on}/{total} on"), color))
}

/// Footer key hints: each `key` brightish-bold, its `label` muted, groups spaced.
fn footer_hints(hints: &[(&str, &str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (key, label)) in hints.iter().enumerate() {
        if i > 0 {
            spans.push(Span::raw("   "));
        }
        spans.push(Span::styled(
            key.to_string(),
            Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {label}"),
            Style::default().fg(FAINT),
        ));
    }
    Line::from(spans)
}

/// Slash commands grouped by purpose for the `/help` overlay. Each entry is a
/// section label and the command names (matched against [`SLASH_COMMANDS`])
/// shown under it. Any command not listed here is swept into a trailing "More"
/// group by the renderer, so a newly added command never silently vanishes.
const HELP_COMMAND_GROUPS: &[(&str, &[&str])] = &[
    (
        "Session",
        &[
            "new", "resume", "rewind", "copy", "config", "effort", "share", "help", "exit",
        ],
    ),
    ("Model & key", &["model", "key"]),
    ("Context", &["attach", "detach", "compact", "context"]),
    ("Skills & tools", &["skills", "create-skill", "mcp"]),
    ("Autonomous", &["plan", "goal"]),
    // Shown only on the aivo provider (hidden by `slash_command_visible`).
    ("aivo account", &["login", "usage", "logout"]),
];

/// Keybindings grouped for the `/help` overlay (label, then `(key, action)`
/// rows). Mirrors the live handlers in `key_handler_impl` / `event_loop_impl`.
const HELP_KEYBINDINGS: &[(&str, &[(&str, &str)])] = &[
    (
        "Compose",
        &[
            ("Enter", "send message / run command"),
            ("Ctrl+J", "insert newline"),
            ("Tab", "complete command or path"),
            ("Ctrl+V", "paste clipboard text/image"),
            ("Ctrl+X Ctrl+E", "edit draft in $EDITOR"),
        ],
    ),
    (
        "Edit",
        &[
            ("Home/Ctrl+A", "line start"),
            ("End/Ctrl+E", "line end"),
            ("Ctrl+←/→", "word jump"),
            ("Ctrl+W", "delete word backward"),
            ("Del/Ctrl+D", "delete forward"),
            ("Ctrl+K", "kill to end of line"),
            ("Ctrl+L", "clear prompt"),
        ],
    ),
    (
        "Navigate",
        &[
            ("↑/↓", "menu / history / line nav"),
            ("←/→", "move cursor"),
            ("PgUp/PgDn", "scroll half page"),
            ("Ctrl+↑/↓", "scroll 3 lines"),
            ("Ctrl+Home/End", "jump to top/bottom"),
            ("Click ↓ pill", "jump to latest (when scrolled up)"),
            ("Mouse wheel", "scroll transcript"),
        ],
    ),
    (
        "Select & copy",
        &[
            ("Mouse drag", "select + copy (auto-scrolls at edges)"),
            ("Double/triple-click", "select word / line"),
            ("Click ▸ / ▾", "expand a folded thinking / !cmd block"),
            ("⌥ / Shift + drag", "native select (any text on screen)"),
        ],
    ),
    (
        "Session",
        &[
            ("Ctrl+R", "resume a saved session"),
            ("Shift+Tab", "cycle mode (normal/auto-approve/plan/review)"),
            ("Esc", "cancel / close overlay"),
            ("Ctrl+C", "exit (press twice to confirm)"),
        ],
    ),
];

/// One `label   description` row for the `/help` overlay: the label (padded to
/// `col` display columns) in `label_style`, the description in `TEXT`, the whole
/// row indented two spaces so it nests under its group header.
fn help_kv_row(label: &str, desc: &str, col: usize, label_style: Style) -> Line<'static> {
    let pad = (col + 2).saturating_sub(display_width(label));
    Line::from(vec![
        Span::styled(format!("  {label}"), label_style),
        Span::styled(
            format!("{}{}", " ".repeat(pad), desc),
            Style::default().fg(TEXT),
        ),
    ])
}

/// Right-pane `/resume` preview: title + meta header over the history tail,
/// bottom-anchored; `scroll_up` counts lines up from the bottom, returned clamped.
fn render_session_preview_pane(
    frame: &mut Frame<'_>,
    area: Rect,
    preview: &SessionPreview,
    entry: Option<&PreviewEntry>,
    scroll_up: u16,
) -> u16 {
    if area.width == 0 || area.height == 0 {
        return 0;
    }
    let width = usize::from(area.width);
    let title = if preview.title.trim().is_empty() {
        preview.session_id.as_str()
    } else {
        preview.title.as_str()
    };
    let header = vec![
        Line::from(Span::styled(
            truncate_for_display_width(title, width),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from(resume_metadata_spans(preview, area.width)),
        Line::from(Span::styled("─".repeat(width), Style::default().fg(FAINT))),
    ];
    let header_h = (header.len() as u16).min(area.height);
    frame.render_widget(
        Paragraph::new(Text::from(header)),
        Rect {
            height: header_h,
            ..area
        },
    );
    let body = Rect {
        y: area.y + header_h,
        height: area.height.saturating_sub(header_h),
        ..area
    };
    if body.height == 0 {
        return 0;
    }

    let Some(entry) = entry else {
        frame.render_widget(
            Paragraph::new(Span::styled("Loading preview…", Style::default().fg(MUTED))),
            body,
        );
        return 0;
    };
    if let Some(err) = &entry.error {
        let lines: Vec<Line> = wrap_chars(&format!("Couldn't load session: {err}"), width)
            .into_iter()
            .map(|chunk| Line::from(Span::styled(chunk, Style::default().fg(ERROR))))
            .collect();
        frame.render_widget(Paragraph::new(Text::from(lines)), body);
        return 0;
    }
    if entry.messages.is_empty() {
        frame.render_widget(
            Paragraph::new(Span::styled("No messages yet", Style::default().fg(MUTED))),
            body,
        );
        return 0;
    }

    // Wrap at width-2, then prefix each visual row with its 2-col role gutter.
    let text_width = body.width.saturating_sub(2).max(1);
    let (lines, bars) = session_preview_lines(&entry.messages, text_width, entry.truncated);
    let wrapped = wrap_transcript(&lines, &bars, text_width);
    let rows: Vec<Line> = wrapped
        .text
        .lines
        .into_iter()
        .zip(wrapped.bars.iter())
        .map(|(line, bar)| {
            let mut spans = vec![match bar {
                Some(color) => Span::styled("▎ ", Style::default().fg(*color)),
                None => Span::raw("  "),
            }];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect();

    // Bottom anchor: 0 pins the tail; render_detail_lines' clamp, upside down.
    let total = rows.len();
    let viewport = usize::from(body.height.saturating_sub(1)).max(1);
    let max_up = total.saturating_sub(viewport).min(usize::from(u16::MAX)) as u16;
    let scroll_up = scroll_up.min(max_up);
    let top = max_up - scroll_up;
    frame.render_widget(
        Paragraph::new(Text::from(rows)).scroll((top, 0)),
        Rect {
            height: body.height.saturating_sub(1),
            ..body
        },
    );
    if max_up > 0 {
        let first = usize::from(top) + 1;
        let last = (usize::from(top) + viewport).min(total);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!("{first}–{last}/{total}"),
                    Style::default().fg(MUTED),
                ),
                Span::styled("   PgUp older", Style::default().fg(FAINT)),
            ])),
            Rect {
                y: body.y + body.height - 1,
                height: 1,
                ..body
            },
        );
    }
    scroll_up
}

/// Render a scrollable Enter drill-in panel: `lines` in the body (the last row is
/// a footer with the close hint (`esc_label`), scroll hint, and position),
/// scrolled by `scroll` lines. Returns the scroll offset clamped to the real
/// content height so the caller can write it back — over-scrolling (e.g. `End`)
/// lands exactly at the bottom rather than off the end.
fn render_detail_lines(
    frame: &mut Frame<'_>,
    area: Rect,
    lines: Vec<Line>,
    scroll: u16,
    esc_label: &str,
) -> u16 {
    if area.height == 0 {
        return 0;
    }
    let body_h = usize::from(area.height.saturating_sub(1));
    let total = lines.len();
    let max_scroll = (total.saturating_sub(body_h)) as u16;
    let scroll = scroll.min(max_scroll);

    let footer_line = if max_scroll == 0 {
        Line::from(Span::styled(
            esc_label.to_string(),
            Style::default().fg(MUTED),
        ))
    } else {
        let first = usize::from(scroll) + 1;
        let last = (usize::from(scroll) + body_h).min(total);
        Line::from(vec![
            Span::styled(esc_label.to_string(), Style::default().fg(MUTED)),
            Span::styled("   ↑↓ scroll   ", Style::default().fg(FAINT)),
            Span::styled(
                format!("{first}–{last}/{total}"),
                Style::default().fg(MUTED),
            ),
        ])
    };
    frame.render_widget(
        Paragraph::new(footer_line),
        Rect {
            y: area.y + area.height - 1,
            height: 1,
            ..area
        },
    );
    if body_h == 0 {
        return scroll;
    }
    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((scroll, 0)),
        Rect {
            height: area.height.saturating_sub(1),
            ..area
        },
    );
    scroll
}

/// Greedy word-wrap to `width` display columns (for error text / descriptions in
/// a drill-in). A single word wider than `width` overflows rather than splitting.
pub(super) fn wrap_chars(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut w = 0usize;
    for word in text.split_whitespace() {
        let ww = display_width(word);
        if w > 0 && w + 1 + ww > width {
            out.push(std::mem::take(&mut cur));
            w = 0;
        }
        if w > 0 {
            cur.push(' ');
            w += 1;
        }
        cur.push_str(word);
        w += ww;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Render an MCP server's tool list for the drill-in: each tool as a `•`-bulleted
/// name on its own line (cyan, bold) with its description word-wrapped full-width
/// and indented beneath, a blank line between tools. Stacking name-over-desc
/// (rather than a name column) keeps short names from stranding their text in a
/// far-right gutter and gives long descriptions the whole panel width.
pub(super) fn mcp_tool_lines(tools: &[(&str, &str, bool)], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (i, (tname, tdesc, on)) in tools.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        if *on {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(TOOL)),
                Span::styled(
                    (*tname).to_string(),
                    Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled("  • ", Style::default().fg(FAINT)),
                Span::styled((*tname).to_string(), Style::default().fg(FAINT)),
                Span::styled(" · off", Style::default().fg(FAINT)),
            ]));
        }
        let desc = tdesc.split_whitespace().collect::<Vec<_>>().join(" ");
        for chunk in wrap_chars(&desc, width.saturating_sub(4)) {
            lines.push(Line::from(Span::styled(
                format!("    {chunk}"),
                Style::default().fg(if *on { MUTED } else { FAINT }),
            )));
        }
    }
    lines
}

/// A skill dir with the home prefix abbreviated to `~`, for the detail views.
fn abbreviate_home_path(dir: &std::path::Path) -> String {
    crate::services::system_env::home_dir()
        .and_then(|home| {
            dir.strip_prefix(&home)
                .ok()
                .map(std::path::Path::to_path_buf)
        })
        .map(|rel| format!("~/{}", rel.display()))
        .unwrap_or_else(|| dir.display().to_string())
}

/// Status color for one MCP server's health.
fn mcp_health_color(health: McpHealth) -> Color {
    match health {
        McpHealth::Connected => ASSISTANT,
        McpHealth::Failed => ERROR,
        McpHealth::NeedsAuth => WARNING,
        McpHealth::Idle => MUTED,
        McpHealth::Disabled => FAINT,
    }
}

/// `/context` fill bar: colored blocks per segment (share of `window`, else `used`),
/// faint blocks for the free tail. Non-zero segments get at least one block.
fn context_fill_bar(
    segs: &[(String, u64, Color)],
    used: u64,
    window: u64,
    bar_w: usize,
) -> Line<'static> {
    let total = if window > 0 { window } else { used.max(1) };
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut cells = 0usize;
    for (_, tokens, color) in segs {
        if *tokens == 0 {
            continue;
        }
        let mut c = ((u128::from(*tokens) * bar_w as u128) / u128::from(total)) as usize;
        c = c.max(1).min(bar_w.saturating_sub(cells));
        if c == 0 {
            continue;
        }
        cells += c;
        spans.push(Span::styled("█".repeat(c), Style::default().fg(*color)));
    }
    let free = bar_w.saturating_sub(cells);
    if free > 0 {
        spans.push(Span::styled("░".repeat(free), Style::default().fg(FAINT)));
    }
    Line::from(spans)
}

/// One legend row for `/context`: `marker · label · right-aligned tokens · share`.
fn context_legend_row(
    marker: &str,
    label: &str,
    tokens: u64,
    pct: &str,
    color: Color,
    label_w: usize,
) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ".to_string(), Style::default()),
        Span::styled(format!("{marker} "), Style::default().fg(color)),
        Span::styled(pad_to_width(label, label_w), Style::default().fg(TEXT)),
        Span::styled(
            format!("  {:>7}", format_token_count_value(tokens)),
            Style::default().fg(MUTED),
        ),
        Span::styled(format!("  {pct:>4}"), Style::default().fg(FAINT)),
    ])
}

/// Right-pad `s` with spaces to `width` display columns so a styled row's
/// background fills the overlay width. Never truncates.
fn pad_to_width(s: &str, width: usize) -> String {
    let w = display_width(s);
    if w >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - w))
    }
}
