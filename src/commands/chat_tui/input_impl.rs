use super::*;

impl ChatTuiApp {
    pub(super) fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut pos = self.cursor - 1;
        while pos > 0 && !self.draft.is_char_boundary(pos) {
            pos -= 1;
        }
        self.cursor = pos;
    }

    pub(super) fn cursor_right(&mut self) {
        if self.cursor >= self.draft.len() {
            return;
        }
        let mut pos = self.cursor + 1;
        while pos < self.draft.len() && !self.draft.is_char_boundary(pos) {
            pos += 1;
        }
        self.cursor = pos;
    }

    pub(super) fn cursor_home(&mut self) {
        let before = &self.draft[..self.cursor];
        self.cursor = before.rfind('\n').map(|pos| pos + 1).unwrap_or(0);
    }

    pub(super) fn cursor_end(&mut self) {
        let after = &self.draft[self.cursor..];
        self.cursor = after
            .find('\n')
            .map(|pos| self.cursor + pos)
            .unwrap_or(self.draft.len());
    }

    pub(super) fn cursor_word_left(&mut self) {
        let chars: Vec<(usize, char)> = self.draft[..self.cursor].char_indices().collect();
        let mut i = chars.len();
        while i > 0 && chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        while i > 0 && !chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        self.cursor = chars.get(i).map(|(pos, _)| *pos).unwrap_or(0);
    }

    pub(super) fn cursor_word_right(&mut self) {
        let rest = &self.draft[self.cursor..];
        let chars: Vec<(usize, char)> = rest.char_indices().collect();
        let mut i = 0;
        while i < chars.len() && chars[i].1.is_whitespace() {
            i += 1;
        }
        while i < chars.len() && !chars[i].1.is_whitespace() {
            i += 1;
        }
        if i >= chars.len() {
            self.cursor = self.draft.len();
        } else {
            self.cursor += chars[i].0;
        }
    }

    pub(super) fn cursor_up(&mut self) {
        use unicode_width::UnicodeWidthStr;
        let before = &self.draft[..self.cursor];
        let Some(prev_nl) = before.rfind('\n') else {
            return;
        };
        let col = UnicodeWidthStr::width(&before[prev_nl + 1..]);
        let before_prev = &before[..prev_nl];
        let prev_line_start = before_prev.rfind('\n').map(|pos| pos + 1).unwrap_or(0);
        self.advance_cursor_to_visual_col(prev_line_start, col);
    }

    pub(super) fn cursor_down(&mut self) {
        use unicode_width::UnicodeWidthStr;
        let before = &self.draft[..self.cursor];
        let col = if let Some(prev_nl) = before.rfind('\n') {
            UnicodeWidthStr::width(&before[prev_nl + 1..])
        } else {
            UnicodeWidthStr::width(before)
        };
        let after = &self.draft[self.cursor..];
        let Some(next_nl_offset) = after.find('\n') else {
            return;
        };
        let next_line_start = self.cursor + next_nl_offset + 1;
        self.advance_cursor_to_visual_col(next_line_start, col);
    }

    fn advance_cursor_to_visual_col(&mut self, line_start: usize, target_col: usize) {
        use unicode_width::UnicodeWidthStr;
        self.cursor = line_start;
        let mut acc_width = 0usize;
        while self.cursor < self.draft.len() {
            let rest = &self.draft[self.cursor..];
            if rest.starts_with('\n') {
                break;
            }
            let mut next_end = self.cursor + 1;
            while next_end < self.draft.len() && !self.draft.is_char_boundary(next_end) {
                next_end += 1;
            }
            let segment_width = UnicodeWidthStr::width(&self.draft[self.cursor..next_end]);
            if acc_width + segment_width > target_col {
                break;
            }
            acc_width += segment_width;
            self.cursor = next_end;
        }
    }

    pub(super) fn insert_char_at_cursor(&mut self, ch: char) {
        self.draft.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub(super) fn insert_pasted_text(&mut self, text: &str) {
        self.leave_history_navigation();
        for ch in text.chars() {
            self.insert_char_at_cursor(ch);
        }
        self.sync_command_menu_state();
    }

    pub(super) fn delete_char_before_cursor(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut start = self.cursor - 1;
        while start > 0 && !self.draft.is_char_boundary(start) {
            start -= 1;
        }
        self.draft.remove(start);
        self.cursor = start;
    }

    pub(super) fn delete_char_at_cursor(&mut self) {
        if self.cursor >= self.draft.len() {
            return;
        }
        self.draft.remove(self.cursor);
    }

    pub(super) fn delete_word_backward(&mut self) {
        let old_cursor = self.cursor;
        self.cursor_word_left();
        self.draft.drain(self.cursor..old_cursor);
    }

    pub(super) fn kill_to_end_of_line(&mut self) {
        let after = &self.draft[self.cursor..];
        let end = after
            .find('\n')
            .map(|pos| self.cursor + pos)
            .unwrap_or(self.draft.len());
        if end == self.cursor && end < self.draft.len() {
            self.draft.remove(self.cursor);
        } else {
            self.draft.drain(self.cursor..end);
        }
    }

    pub(super) fn active_command_query(&self) -> Option<&str> {
        if self.overlay.blocks_input()
            || self.is_busy()
            || !self.draft.starts_with('/')
            || self.draft.starts_with("//")
            || self.draft.contains('\n')
            || self.draft.contains(' ')
        {
            return None;
        }
        Some(&self.draft[1..])
    }

    pub(super) fn active_attach_query(&self) -> Option<&str> {
        if self.overlay.blocks_input()
            || self.is_busy()
            || !self.draft.starts_with("/attach ")
            || self.draft.starts_with("//")
            || self.draft.contains('\n')
        {
            return None;
        }
        Some(&self.draft["/attach ".len()..])
    }

    pub(super) fn visible_command_menu(&self) -> Option<VisibleCommandMenu> {
        if self.command_menu.dismissed {
            return None;
        }
        let (kind, entries) = if let Some(query) = self.active_command_query() {
            (
                MenuKind::Commands,
                filter_slash_commands(query)
                    .into_iter()
                    .map(ComposerMenuEntry::Command)
                    .collect::<Vec<_>>(),
            )
        } else if let Some(query) = self.active_attach_query() {
            (
                MenuKind::AttachPath,
                collect_attach_path_suggestions(&self.cwd, query)
                    .into_iter()
                    .map(ComposerMenuEntry::Path)
                    .collect::<Vec<_>>(),
            )
        } else {
            return None;
        };
        let selected = if entries.is_empty() {
            None
        } else {
            Some(
                self.command_menu
                    .selected
                    .min(entries.len().saturating_sub(1)),
            )
        };
        Some(VisibleCommandMenu {
            kind,
            entries,
            selected,
        })
    }

    pub(super) fn sync_command_menu_state(&mut self) {
        let query = if let Some(query) = self.active_command_query() {
            query.to_string()
        } else if let Some(query) = self.active_attach_query() {
            query.to_string()
        } else {
            self.command_menu.reset();
            return;
        };

        if self.command_menu.query != query {
            if self.command_menu.dismissed {
                self.command_menu.placement = None;
            }
            self.command_menu.query = query.clone();
            self.command_menu.selected = 0;
            self.command_menu.dismissed = false;
        }

        let matches = if self.active_command_query().is_some() {
            filter_slash_commands(&query).len()
        } else {
            collect_attach_path_suggestions(&self.cwd, &query).len()
        };
        if matches == 0 {
            self.command_menu.selected = 0;
        } else {
            self.command_menu.selected = self.command_menu.selected.min(matches - 1);
        }
    }

    pub(super) fn select_previous_command(&mut self) {
        let Some(menu) = self.visible_command_menu() else {
            return;
        };
        let Some(selected) = menu.selected else {
            return;
        };
        self.command_menu.selected = if selected == 0 {
            menu.entries.len() - 1
        } else {
            selected - 1
        };
    }

    pub(super) fn select_next_command(&mut self) {
        let Some(menu) = self.visible_command_menu() else {
            return;
        };
        let Some(selected) = menu.selected else {
            return;
        };
        self.command_menu.selected = if selected + 1 >= menu.entries.len() {
            0
        } else {
            selected + 1
        };
    }

    pub(super) fn dismiss_command_menu(&mut self) -> bool {
        if (self.active_command_query().is_none() && self.active_attach_query().is_none())
            || self.command_menu.dismissed
        {
            return false;
        }
        self.command_menu.dismissed = true;
        self.command_menu.placement = None;
        true
    }

    pub(super) fn selected_menu_entry(&self) -> Option<ComposerMenuEntry> {
        let menu = self.visible_command_menu()?;
        let selected = menu.selected?;
        menu.entries.get(selected).cloned()
    }

    pub(super) fn insert_selected_command(&mut self) -> bool {
        let Some(entry) = self.selected_menu_entry() else {
            return false;
        };
        self.command_menu.selected = 0;
        match entry {
            ComposerMenuEntry::Command(command) => {
                self.draft = command.insertion_text();
                self.cursor = self.draft.len();
                self.command_menu.dismissed = true;
                self.command_menu.placement = None;
            }
            ComposerMenuEntry::Path(path) => {
                self.draft = path.insertion_text;
                self.cursor = self.draft.len();
                // Keep the menu open for directories so the user can continue
                // navigating into the selected directory with subsequent Tab presses.
                self.command_menu.dismissed = !path.is_dir;
                // Only reset placement when dismissing — same rule as dismiss_command_menu.
                // When the menu stays open (directory), preserve placement to avoid jumping.
                if !path.is_dir {
                    self.command_menu.placement = None;
                }
            }
        }
        true
    }

    pub(super) async fn execute_selected_command(&mut self) -> Result<bool> {
        let Some(entry) = self.selected_menu_entry() else {
            return Ok(false);
        };
        match entry {
            ComposerMenuEntry::Command(command) => {
                self.draft = command.command_label();
                self.cursor = self.draft.len();
                self.command_menu.reset();
                self.submit_draft().await
            }
            ComposerMenuEntry::Path(path) => {
                self.draft = path.insertion_text;
                self.cursor = self.draft.len();
                self.command_menu.reset();
                if path.is_dir {
                    Ok(false)
                } else {
                    self.submit_draft().await
                }
            }
        }
    }

    pub(super) fn paste_system_clipboard(&mut self) -> Result<()> {
        match read_system_clipboard()? {
            ClipboardPayload::Text(text) => {
                if text.is_empty() {
                    self.notice = Some((MUTED, "Clipboard is empty".to_string()));
                } else {
                    self.insert_pasted_text(&text);
                }
            }
            ClipboardPayload::Attachment(attachment) => {
                let kind = attachment_kind_label(&attachment);
                let name = attachment.name.clone();
                self.draft_attachments.push(attachment);
                self.notice = Some((MUTED, format!("Pasted {kind}: {name}")));
            }
            ClipboardPayload::Empty => {
                self.notice = Some((MUTED, "Clipboard is empty".to_string()));
            }
        }
        Ok(())
    }
}
