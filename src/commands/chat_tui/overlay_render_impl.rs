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
            "Chat with the model, or type a command. Enter sends.",
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
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(format!("  {group}"), group_style)));
            for name in *names {
                if let Some(command) = SLASH_COMMANDS.iter().find(|c| c.name == *name) {
                    shown.push(command.name);
                    lines.push(help_kv_row(
                        command.help_label,
                        command.description,
                        cmd_col,
                        cmd_style,
                    ));
                }
            }
        }
        // Completeness guard: a command missing from every group above (e.g. one
        // added later) still shows here rather than silently vanishing.
        let leftover: Vec<_> = SLASH_COMMANDS
            .iter()
            .filter(|c| !shown.contains(&c.name))
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

    /// `/skills` overlay: a toggle list of the agent skills discovered for the
    /// working dir. A `[✓]`/`[ ]` checkbox shows each skill's enabled state, with
    /// the name on its own line and the (truncated) description below it; Tab
    /// drills in for the full text. The chrome (rounded frame, on-count badge,
    /// search line, footer hints) is shared with `/mcp` via [`render_toggle_list`]
    /// so the two feel the same; the list scrolls to follow the selection.
    /// Returns the clamped detail-scroll offset while a drill-in is open (so the
    /// caller can persist it), else `None`.
    pub(super) fn render_skills_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &SkillsOverlay,
    ) -> Option<u16> {
        // Enter drill-in: full description + body preview for the selected skill.
        if let Some(item) = state.viewing.and_then(|i| state.items.get(i)) {
            let inner = overlay_shell(frame, area, "Skills", None);
            if inner.height == 0 {
                return Some(state.detail_scroll);
            }
            return Some(self.render_skill_detail(
                frame,
                inner,
                item,
                usize::from(inner.width).max(1),
                state.detail_scroll,
            ));
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

        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let footer: Vec<(&str, &str)>;
        if state.adding.is_some() {
            rows.extend([
                Line::from(Span::styled(
                    "name [description], or a github:owner/repo to install",
                    Style::default().fg(MUTED),
                )),
                Line::from(Span::styled(
                    "e.g.  changelog Summarize the git log   ·   github:anthropics/skills",
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
            let inner_width = usize::from(area.width).saturating_sub(6).max(1);
            for (pos, &i) in filtered.iter().enumerate() {
                let item = &state.items[i];
                let desc =
                    truncate_for_display_width(&item.description, toggle_detail_room(inner_width));
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
                ("Enter", "toggle"),
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

        let detail = state
            .items
            .get(state.selected)
            .filter(|_| state.adding.is_none() && filtered.contains(&state.selected))
            .map(|item| (self.skill_detail_text(item), MUTED));

        render_toggle_list(
            frame,
            area,
            ToggleListView {
                title: "Skills",
                badge: count_badge(state.adding.is_none(), enabled, state.items.len()),
                input_line,
                rows,
                selected_pos,
                detail,
                footer,
            },
        );
        None
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
            "Chat settings — remembered across sessions",
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

    /// `/mcp` overlay: a toggle list of the configured MCP servers. A `[✓]`/`[ ]`
    /// checkbox shows the enabled state, with the server name on its own line and
    /// its live status (tool count / failure / connecting, colored by health) on
    /// the line below; Tab drills in for the full tool list. Shares its chrome and
    /// layout with `/skills` via [`render_toggle_list`] so the two feel the same.
    /// Returns the clamped detail-scroll offset while a drill-in is open (so the
    /// caller can persist it), else `None`.
    pub(super) fn render_mcp_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        state: &McpOverlay,
    ) -> Option<u16> {
        // Enter drill-in: full tool list (or error) for the selected server.
        if let Some(row) = state.viewing.and_then(|i| state.items.get(i)) {
            let inner = overlay_shell(frame, area, "MCP servers", None);
            if inner.height == 0 {
                return Some(state.detail_scroll);
            }
            return Some(self.render_mcp_detail(
                frame,
                inner,
                row,
                usize::from(inner.width).max(1),
                state.detail_scroll,
            ));
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

        let mut rows: Vec<Line> = Vec::new();
        let mut selected_pos = 0usize;
        let footer: Vec<(&str, &str)>;
        if state.adding.is_some() {
            rows.extend([
                Line::from(Span::styled(
                    "command args… or a https:// URL (name derived), or Ctrl+V a JSON block",
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
            let inner_width = usize::from(area.width).saturating_sub(6).max(1);
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
                ("Enter", "toggle"),
                ("Tab", "view"),
                ("^A", "add"),
                ("^D", "rm"),
                ("^O", "auth"),
            ];
        }

        if !self.agent_capable() {
            rows.push(Line::from(""));
            rows.push(Line::from(Span::styled(
                "MCP tools run with the native agent (plain API keys); this key uses a different backend.",
                Style::default().fg(MUTED),
            )));
        }

        let detail = state
            .items
            .get(state.selected)
            .filter(|_| state.adding.is_none() && filtered.contains(&state.selected))
            .map(|row| (self.mcp_detail_text(row), mcp_health_color(row.health)));

        render_toggle_list(
            frame,
            area,
            ToggleListView {
                title: "MCP servers",
                badge: count_badge(state.adding.is_none(), on, state.items.len()),
                input_line,
                rows,
                selected_pos,
                detail,
                footer,
            },
        );
        None
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

    /// Enter drill-in for `/mcp`: the server's command, status, and full tool
    /// list (name + wrapped description) when connected, or its full error.
    /// Scrolls by `scroll` lines; returns the clamped scroll offset.
    fn render_mcp_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        row: &McpServerRow,
        width: usize,
        scroll: u16,
    ) -> u16 {
        use crate::agent::mcp::ServerScope;
        let scope = match row.scope {
            ServerScope::User => "user",
            ServerScope::Project => "project · .mcp.json",
        };
        let mut lines = vec![Line::from(vec![
            Span::styled(
                row.name.clone(),
                Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  [{scope}]"), Style::default().fg(MUTED)),
        ])];
        // `row.command` holds the launch command for stdio servers and the
        // endpoint URL for HTTP ones; label it accordingly, wrapping if long.
        let label = if row.command.contains("://") {
            "url"
        } else {
            "command"
        };
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
                let cost = self
                    .mcp_client
                    .as_ref()
                    .and_then(|c| c.estimated_tokens(&row.name))
                    .map(|t| format!(" · ~{} tokens/turn", humanize_count(t)))
                    .unwrap_or_default();
                lines.push(Line::from(Span::styled(
                    format!("Tools ({}){cost}:", tools.len()),
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
        render_detail_lines(frame, area, lines, scroll, "Esc back")
    }

    /// Enter drill-in for `/skills`: the skill's location, full description, and
    /// the complete SKILL.md instructions (word-wrapped, indentation preserved).
    /// Scrolls by `scroll` lines; returns the clamped scroll offset.
    fn render_skill_detail(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        item: &SkillToggle,
        width: usize,
        scroll: u16,
    ) -> u16 {
        use crate::agent::skills::SkillScope;
        let scope = match item.scope {
            SkillScope::User => "user",
            SkillScope::Project => "project",
        };
        let mut lines = vec![
            Line::from(vec![
                Span::styled(
                    item.name.clone(),
                    Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  [{scope}{}]", if item.enabled { "" } else { " · off" }),
                    Style::default().fg(MUTED),
                ),
            ]),
            Line::from(Span::styled(
                truncate_for_display_width(&abbreviate_home_path(&item.dir), width),
                Style::default().fg(MUTED),
            )),
            Line::from(""),
        ];
        if !item.description.is_empty() {
            for chunk in wrap_chars(&item.description, width) {
                lines.push(Line::from(Span::styled(chunk, Style::default().fg(TEXT))));
            }
            lines.push(Line::from(""));
        }
        lines.push(Line::from(Span::styled(
            "Instructions:",
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )));
        if item.body.trim().is_empty() {
            lines.push(Line::from(Span::styled(
                "  (empty — edit SKILL.md to add instructions)",
                Style::default().fg(FAINT),
            )));
        } else {
            // Wrap each source line to the panel width (the leading "  " body
            // indent plus any of the line's own indentation is preserved on wraps)
            // so nothing is cut off the right edge.
            for body_line in item.body.lines() {
                for chunk in wrap_indented(body_line, width.saturating_sub(2)) {
                    lines.push(Line::from(Span::styled(
                        format!("  {chunk}"),
                        Style::default().fg(FAINT),
                    )));
                }
            }
        }
        render_detail_lines(frame, area, lines, scroll, "Esc back")
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

/// Draw the rounded modal frame with an accent title (left) and an optional
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
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
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

/// Render the shared toggle overlay: frame + input row + scrolling list +
/// selected-item detail + footer hints. Both `/skills` and `/mcp` route through
/// here so their chrome, spacing, and scroll behavior stay in lockstep.
fn render_toggle_list(frame: &mut Frame<'_>, area: Rect, view: ToggleListView) {
    let inner = overlay_shell(frame, area, view.title, view.badge);
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
        frame.render_widget(
            Paragraph::new(footer_hints(&view.footer)),
            Rect {
                y: inner.y + inner.height - 1,
                height: 1,
                ..inner
            },
        );
    }
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
        let hl = Style::default().bg(SELECT_BG).fg(SELECT_TEXT);
        vec![
            Line::from(Span::styled(
                pad_to_width(&format!("{check}{name}"), width),
                hl,
            )),
            Line::from(Span::styled(
                pad_to_width(&format!("{indent}{detail}"), width),
                hl,
            )),
        ]
    } else if enabled {
        vec![
            Line::from(vec![
                Span::styled(check, Style::default().fg(ACCENT)),
                Span::styled(
                    name.to_string(),
                    Style::default().fg(ASSISTANT).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                format!("{indent}{detail}"),
                Style::default().fg(detail_color),
            )),
        ]
    } else {
        vec![
            Line::from(Span::styled(
                format!("{check}{name}"),
                Style::default().fg(FAINT),
            )),
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
            Span::styled("/ ", Style::default().fg(ACCENT)),
            Span::styled(
                placeholder.to_string(),
                Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled("/ ", Style::default().fg(ACCENT)),
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
        "Chat",
        &[
            "new", "resume", "rewind", "copy", "config", "effort", "help", "exit",
        ],
    ),
    ("Model & key", &["model", "key"]),
    ("Context", &["attach", "detach"]),
    (
        "Skills & tools",
        &["skills", "create-skill", "mcp", "agent"],
    ),
    ("Autonomous", &["plan", "goal"]),
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
            ("Ctrl+R", "resume a saved chat"),
            ("Shift+Tab", "toggle agent auto-approve"),
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
fn wrap_chars(text: &str, width: usize) -> Vec<String> {
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

/// Word-wrap one source line to `width`, preserving its leading indentation and
/// re-applying that indent to every continuation line (so wrapped list items and
/// nested blocks stay aligned). A blank line stays a single blank line.
fn wrap_indented(line: &str, width: usize) -> Vec<String> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return vec![String::new()];
    }
    let indent = &line[..line.len() - trimmed.len()];
    let avail = width.saturating_sub(display_width(indent)).max(1);
    let chunks = wrap_chars(trimmed, avail);
    if chunks.is_empty() {
        return vec![indent.to_string()];
    }
    chunks
        .into_iter()
        .map(|chunk| format!("{indent}{chunk}"))
        .collect()
}

/// Render an MCP server's tool list for the drill-in: each tool as a `•`-bulleted
/// name on its own line (cyan, bold) with its description word-wrapped full-width
/// and indented beneath, a blank line between tools. Stacking name-over-desc
/// (rather than a name column) keeps short names from stranding their text in a
/// far-right gutter and gives long descriptions the whole panel width.
pub(super) fn mcp_tool_lines(tools: &[(&str, &str)], width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (i, (tname, tdesc)) in tools.iter().enumerate() {
        if i > 0 {
            lines.push(Line::from(""));
        }
        lines.push(Line::from(vec![
            Span::styled("  • ", Style::default().fg(TOOL)),
            Span::styled(
                (*tname).to_string(),
                Style::default().fg(TOOL).add_modifier(Modifier::BOLD),
            ),
        ]));
        let desc = tdesc.split_whitespace().collect::<Vec<_>>().join(" ");
        for chunk in wrap_chars(&desc, width.saturating_sub(4)) {
            lines.push(Line::from(Span::styled(
                format!("    {chunk}"),
                Style::default().fg(MUTED),
            )));
        }
    }
    lines
}

/// Compact a count: `1234` → `1.2k`, `12345` → `12k`, `<1000` verbatim.
pub(super) fn humanize_count(n: usize) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 10_000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format!("{}k", n / 1000)
    }
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
