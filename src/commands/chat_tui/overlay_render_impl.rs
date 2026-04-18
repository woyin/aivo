use super::*;

impl ChatTuiApp {
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
    ) {
        if matches!(picker.kind, PickerKind::Session) {
            self.render_session_picker(frame, area, picker);
            return;
        }

        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
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
                Span::styled("/ ", Style::default().fg(ACCENT)),
                Span::styled(
                    picker_search_placeholder(&picker.kind),
                    Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                ),
            ])
        } else {
            Line::from(vec![
                Span::styled("/ ", Style::default().fg(ACCENT)),
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
            return;
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
    }

    pub(super) fn render_session_picker(
        &mut self,
        frame: &mut Frame<'_>,
        area: Rect,
        picker: &PickerState,
    ) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
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
        let status_label = format!(
            "{} · esc",
            format_session_match_count(filtered_count, total_count)
        );
        let search_placeholder = if picker.query.is_empty() {
            vec![Span::styled(
                "filter chats, keys, models",
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            )]
        } else {
            vec![Span::styled(
                picker.query.clone(),
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
            )]
        };
        let search_width = chunks[0].width.max(1);
        let title_label = "Sessions";
        let esc_width = display_width(&status_label) as u16;
        let title_width = display_width(title_label) as u16;
        let middle_padding = search_width.saturating_sub(title_width + esc_width).max(1);
        let mut header_spans = vec![Span::styled(
            title_label,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )];
        header_spans.push(Span::raw(" ".repeat(usize::from(middle_padding))));
        header_spans.push(Span::styled(status_label, Style::default().fg(MUTED)));
        frame.render_widget(Paragraph::new(Line::from(header_spans)), chunks[0]);

        let search_line = Line::from(
            std::iter::once(Span::styled("/ ", Style::default().fg(ACCENT)))
                .chain(search_placeholder)
                .collect::<Vec<_>>(),
        );
        frame.render_widget(
            Paragraph::new(search_line),
            Rect::new(chunks[0].x, chunks[0].y + 1, chunks[0].width, 1),
        );

        let (lines, row_to_filtered_index) =
            render_session_picker_rows(picker, usize::from(chunks[1].height), chunks[1].width);

        self.picker_hitbox = Some(PickerHitbox {
            overlay_area: area,
            list_area: chunks[1],
            row_to_filtered_index,
        });

        frame.render_widget(
            Paragraph::new(Text::from(lines))
                .style(Style::default().fg(TEXT))
                .wrap(Wrap { trim: false }),
            chunks[1],
        );
        let footer_text = if picker.pending_delete.is_some() {
            "Enter or Ctrl+D confirm delete · Esc cancel"
        } else {
            "Type to filter · Up/Down wrap · Enter open · Ctrl+D delete"
        };
        frame.render_widget(
            Paragraph::new(footer_text).style(Style::default().fg(MUTED)),
            chunks[2],
        );
    }

    pub(super) fn render_help_overlay(&self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(Clear, area);
        let shell = Block::default()
            .borders(Borders::ALL)
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
        let mut lines = vec![
            Line::from(Span::styled(
                "Slash commands",
                Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];
        let max_cmd_width = SLASH_COMMANDS
            .iter()
            .map(|c| display_width(c.help_label))
            .max()
            .unwrap_or(0);
        for command in SLASH_COMMANDS {
            let pad = max_cmd_width + 2 - display_width(command.help_label);
            lines.push(Line::from(vec![
                Span::styled(command.help_label, cmd_style),
                Span::styled(
                    format!("{}{}", " ".repeat(pad), command.description),
                    Style::default().fg(TEXT),
                ),
            ]));
        }

        let keybindings: &[(&str, &str)] = &[
            ("Enter", "send message / run command"),
            ("Ctrl+J", "insert newline"),
            ("Ctrl+V", "paste clipboard text/image"),
            ("↑/↓", "menu / history / line nav"),
            ("←/→", "move cursor"),
            ("Home/Ctrl+A", "line start"),
            ("End/Ctrl+E", "line end"),
            ("Ctrl+←/→", "word jump"),
            ("Ctrl+W", "delete word backward"),
            ("Ctrl+K", "kill to end of line"),
            ("Ctrl+C", "clear prompt / exit when empty"),
            ("Ctrl+L", "clear screen"),
            ("Ctrl+T", "toggle thinking blocks"),
            ("Tab", "complete command or path"),
            ("PgUp/PgDn", "scroll half page"),
            ("Ctrl+↑/↓", "scroll 3 lines"),
            ("Mouse wheel", "scroll 3 lines"),
            ("Shift+mouse", "select and copy text"),
            ("Ctrl+Home/End", "jump to top/bottom"),
            ("Esc", "cancel / close overlay"),
        ];
        let max_key_width = keybindings
            .iter()
            .map(|(key, _)| display_width(key))
            .max()
            .unwrap_or(0);
        lines.extend([
            Line::from(""),
            Line::from(Span::styled(
                "Keybindings",
                Style::default().fg(MUTED).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ]);
        for (key, desc) in keybindings {
            let pad = max_key_width + 2 - display_width(key);
            lines.push(Line::from(vec![
                Span::styled(*key, key_style),
                Span::styled(
                    format!("{}{}", " ".repeat(pad), desc),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        lines.extend([
            Line::from(""),
            Line::from(Span::styled(
                "//message sends a literal leading slash",
                Style::default().fg(MUTED),
            )),
            Line::from(Span::styled(
                "Esc closes this overlay",
                Style::default().fg(MUTED),
            )),
        ]);

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            inner,
        );
    }
}
