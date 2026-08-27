use super::*;

impl CodeTuiApp {
    /// Byte span of each attachment's tag in the draft; `None` when absent
    /// (staged mid-turn, recalled history).
    pub(super) fn attachment_tag_spans(&self) -> Vec<Option<(usize, usize)>> {
        self.draft_attachments
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let tag = attachment_tag(a, i + 1);
                self.draft.find(&tag).map(|s| (s, s + tag.len()))
            })
            .collect()
    }

    fn tag_span_covering(&self, pos: usize) -> Option<(usize, usize)> {
        self.attachment_tag_spans()
            .into_iter()
            .flatten()
            .find(|&(s, e)| s <= pos && pos < e)
    }

    fn tag_span_strictly_inside(&self, pos: usize) -> Option<(usize, usize)> {
        self.attachment_tag_spans()
            .into_iter()
            .flatten()
            .find(|&(s, e)| s < pos && pos < e)
    }

    /// Drop attachments whose tag the edit removed and renumber survivors.
    /// `before` = spans captured pre-edit: only attachments that had a tag then
    /// can be dropped, so tagless staged attachments always survive.
    pub(super) fn reconcile_attachment_tags(&mut self, before: &[Option<(usize, usize)>]) {
        if self.draft_attachments.is_empty() {
            return;
        }
        let mut keep: Vec<(usize, MessageAttachment)> = Vec::new();
        let mut dropped: Vec<MessageAttachment> = Vec::new();
        for (i, att) in std::mem::take(&mut self.draft_attachments)
            .into_iter()
            .enumerate()
        {
            let had_tag = before.get(i).copied().flatten().is_some();
            if had_tag && !self.draft.contains(&attachment_tag(&att, i + 1)) {
                dropped.push(att);
            } else {
                keep.push((i, att));
            }
        }
        if !dropped.is_empty() {
            // Ascending rewrite can't collide: new numbers never exceed old.
            for (new_idx, (old_idx, att)) in keep.iter().enumerate() {
                if new_idx != *old_idx {
                    self.replace_in_draft(
                        &attachment_tag(att, *old_idx + 1),
                        &attachment_tag(att, new_idx + 1),
                    );
                }
            }
            self.notice = Some((MUTED(), removed_attachment_notice(&dropped)));
        }
        self.draft_attachments = keep.into_iter().map(|(_, att)| att).collect();
    }

    fn replace_in_draft(&mut self, old: &str, new: &str) {
        let Some(start) = self.draft.find(old) else {
            return;
        };
        self.draft.replace_range(start..start + old.len(), new);
        if self.cursor >= start + old.len() {
            self.cursor = self.cursor + new.len() - old.len();
        } else if self.cursor > start {
            self.cursor = self.cursor.min(start + new.len());
        }
    }

    pub(super) fn insert_attachment_tag(&mut self, tag: &str) {
        if let Some((_, end)) = self.tag_span_strictly_inside(self.cursor) {
            self.cursor = end;
        }
        let mut text = String::new();
        if self.draft[..self.cursor]
            .chars()
            .next_back()
            .is_some_and(|c| !c.is_whitespace())
        {
            text.push(' ');
        }
        text.push_str(tag);
        if self.draft[self.cursor..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_whitespace())
        {
            text.push(' ');
        }
        self.draft.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    /// Seed tags for attachments the draft doesn't show (`/attach`, CLI `-a`).
    pub(super) fn append_missing_attachment_tags(&mut self) {
        for i in 0..self.draft_attachments.len() {
            let tag = attachment_tag(&self.draft_attachments[i], i + 1);
            if self.draft.contains(&tag) {
                continue;
            }
            let cursor_at_end = self.cursor == self.draft.len();
            if !self.draft.is_empty() && !self.draft.ends_with(char::is_whitespace) {
                self.draft.push(' ');
            }
            self.draft.push_str(&tag);
            self.draft.push(' ');
            if cursor_at_end {
                self.cursor = self.draft.len();
            }
        }
    }

    pub(super) fn cursor_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut pos = self.cursor - 1;
        while pos > 0 && !self.draft.is_char_boundary(pos) {
            pos -= 1;
        }
        if let Some((start, _)) = self.tag_span_strictly_inside(pos) {
            pos = start;
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
        if let Some((_, end)) = self.tag_span_strictly_inside(pos) {
            pos = end;
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

    /// Move the cursor up one *visual* (wrapped) row, keeping its display column
    /// where possible. Returns false when already on the top row, so the caller
    /// can fall back to history recall (Claude-Code style: history at the edge).
    pub(super) fn composer_cursor_up(&mut self) -> bool {
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let (row, col) = composer_cursor_rowcol(&self.draft, self.cursor, &rows);
        if row == 0 {
            return false;
        }
        self.cursor = composer_offset_for_col(&self.draft, &rows, row - 1, col);
        true
    }

    /// Move the cursor down one visual row. Returns false when already on the
    /// bottom row (caller falls back to forward history navigation).
    pub(super) fn composer_cursor_down(&mut self) -> bool {
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let (row, col) = composer_cursor_rowcol(&self.draft, self.cursor, &rows);
        if row + 1 >= rows.len() {
            return false;
        }
        self.cursor = composer_offset_for_col(&self.draft, &rows, row + 1, col);
        true
    }

    /// Up / Ctrl+P: move up a visual row, or — once on the top row — recall the
    /// previous draft from history (Claude-Code style: history at the edge).
    pub(super) fn composer_up_or_history_prev(&mut self) {
        if !self.composer_cursor_up() {
            self.history_prev();
        }
    }

    /// Down / Ctrl+N: move down a visual row, or step forward through history
    /// once on the bottom row.
    pub(super) fn composer_down_or_history_next(&mut self) {
        if !self.composer_cursor_down() {
            self.history_next();
        }
    }

    /// Bare Up: move up a visual row, or at the top edge scroll the transcript
    /// (`swipe_scroll`) or recall the previous draft. Ctrl+P always means history.
    pub(super) fn composer_up_key(&mut self) {
        if self.composer_cursor_up() {
            return;
        }
        if self.swipe_scroll {
            self.scroll_up_lines(self.scroll_speed);
        } else {
            self.history_prev();
        }
    }

    /// Bare Down; mirror of [`Self::composer_up_key`] downward.
    pub(super) fn composer_down_key(&mut self) {
        if self.composer_cursor_down() {
            return;
        }
        if self.swipe_scroll {
            self.scroll_down_lines(self.scroll_speed);
        } else {
            self.history_next();
        }
    }

    pub(super) fn insert_char_at_cursor(&mut self, ch: char) {
        // Typing inside a tag would split it — snap out.
        if let Some((_, end)) = self.tag_span_strictly_inside(self.cursor) {
            self.cursor = end;
        }
        self.draft.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    pub(super) fn insert_pasted_text(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
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
        // Deleting into a tag swallows the whole tag.
        if let Some((s, e)) = self.tag_span_covering(start) {
            let before = self.attachment_tag_spans();
            self.draft.drain(s..e);
            self.cursor = s;
            self.reconcile_attachment_tags(&before);
            return;
        }
        self.draft.remove(start);
        self.cursor = start;
    }

    pub(super) fn delete_char_at_cursor(&mut self) {
        if self.cursor >= self.draft.len() {
            return;
        }
        if let Some((s, e)) = self.tag_span_covering(self.cursor) {
            let before = self.attachment_tag_spans();
            self.draft.drain(s..e);
            self.cursor = s;
            self.reconcile_attachment_tags(&before);
            return;
        }
        self.draft.remove(self.cursor);
    }

    /// Widen `start..end` over any tag it bites into — no half-tag remnants.
    fn expand_range_over_tags(
        &self,
        spans: &[Option<(usize, usize)>],
        mut start: usize,
        mut end: usize,
    ) -> (usize, usize) {
        for &(s, e) in spans.iter().flatten() {
            if s < end && e > start {
                start = start.min(s);
                end = end.max(e);
            }
        }
        (start, end)
    }

    pub(super) fn delete_word_backward(&mut self) {
        let before = self.attachment_tag_spans();
        let old_cursor = self.cursor;
        self.cursor_word_left();
        let (start, end) = self.expand_range_over_tags(&before, self.cursor, old_cursor);
        self.draft.drain(start..end);
        self.cursor = start;
        self.reconcile_attachment_tags(&before);
    }

    pub(super) fn kill_to_end_of_line(&mut self) {
        let before = self.attachment_tag_spans();
        let after = &self.draft[self.cursor..];
        let end = after
            .find('\n')
            .map(|pos| self.cursor + pos)
            .unwrap_or(self.draft.len());
        if end == self.cursor && end < self.draft.len() {
            self.delete_char_at_cursor();
            return;
        }
        let (start, end) = self.expand_range_over_tags(&before, self.cursor, end);
        self.draft.drain(start..end);
        self.cursor = start;
        self.reconcile_attachment_tags(&before);
    }

    pub(super) fn active_command_query(&self) -> Option<&str> {
        if self.overlay.blocks_input()
            || !self.draft.starts_with('/')
            || self.draft.starts_with("//")
            || self.draft.contains('\n')
            || self.draft.contains(' ')
        {
            return None;
        }
        Some(&self.draft[1..])
    }

    /// Inline ghost hint shown right after a bare slash command in the composer
    /// (Claude-Code style), e.g. `/mcp [add … | rm <name>]`. Only while the draft
    /// is exactly the command with no arguments typed yet and the cursor is at the
    /// end — once a real argument is typed, `trim_end()` keeps a space so no
    /// single-token name matches and the hint clears.
    pub(super) fn composer_command_hint(&self) -> Option<&'static str> {
        if self.overlay.blocks_input()
            || self.cursor != self.draft.len()
            || !self.draft.starts_with('/')
            || self.draft.starts_with("//")
            || self.draft.contains('\n')
        {
            return None;
        }
        command_usage_hint(self.draft[1..].trim_end())
    }

    /// The (command, partial path) of a path-completing command being typed.
    pub(super) fn active_path_query(&self) -> Option<(&'static str, &str)> {
        if self.overlay.blocks_input() || self.draft.starts_with("//") || self.draft.contains('\n')
        {
            return None;
        }
        [("attach", "/attach "), ("preview", "/preview ")]
            .into_iter()
            .find_map(|(command, prefix)| {
                self.draft.strip_prefix(prefix).map(|rest| (command, rest))
            })
    }

    /// An `@name` sub-agent mention being typed at the cursor: the byte offset
    /// of the `@` and the partial name after it. Only at a word boundary (start
    /// of draft or after whitespace) so emails and paths don't trigger it, and
    /// only when discovered profiles exist to suggest. Command/attach modes win.
    pub(super) fn active_mention_query(&self) -> Option<(usize, String)> {
        if self.overlay.blocks_input()
            || self.last_subagents.is_empty()
            || self.active_command_query().is_some()
            || self.active_path_query().is_some()
        {
            return None;
        }
        let head = self.draft.get(..self.cursor)?;
        let at = head.rfind('@')?;
        if !head[..at]
            .chars()
            .next_back()
            .is_none_or(char::is_whitespace)
        {
            return None;
        }
        let query = &head[at + 1..];
        if query.chars().any(char::is_whitespace) {
            return None;
        }
        Some((at, query.to_string()))
    }

    /// The `@` menu entries: discovered sub-agent profiles matching the partial
    /// name, prefix matches first (same ranking as the skill-command filter).
    pub(super) fn matching_mention_entries(&self, query: &str) -> Vec<ComposerMenuEntry> {
        let mut prefix = Vec::new();
        let mut fuzzy = Vec::new();
        for sa in &self.last_subagents {
            let entry = ComposerMenuEntry::Agent(AgentMention {
                name: sa.name.clone(),
                description: crate::agent::skills::advert_description(&sa.description),
            });
            if sa.name.starts_with(query) {
                prefix.push(entry);
            } else if matches_fuzzy(query, &sa.name) {
                fuzzy.push(entry);
            }
        }
        prefix.extend(fuzzy);
        prefix
    }

    /// Whether the draft is a `!cmd` local shell command — a single line whose
    /// first non-space char is a lone `!` (not the `!!` literal-`!` escape). Drives
    /// the composer's shell-command highlight; mirrors `prepare_submit_action`'s
    /// shell branch (which trims and treats a multi-line draft as a plain message).
    pub(super) fn draft_is_shell_command(&self) -> bool {
        if self.draft.contains('\n') {
            return false;
        }
        let trimmed = self.draft.trim_start();
        trimmed.starts_with('!') && !trimmed.starts_with("!!")
    }

    /// The `/` menu entries for `query`: built-in commands first, then discovered
    /// skill commands (`/repo-study`). A skill whose name collides with a built-in
    /// is dropped — the built-in wins — so a stray skill can't shadow `/model` etc.
    pub(super) fn matching_command_entries(&self, query: &str) -> Vec<ComposerMenuEntry> {
        let mut entries: Vec<ComposerMenuEntry> = filter_slash_commands(query)
            .into_iter()
            // Account commands are hidden on BYOK keys.
            .filter(|command| self.slash_command_visible(command.name))
            .map(ComposerMenuEntry::Command)
            .collect();
        for skill in filter_skill_commands(&self.skill_commands, query) {
            if !SLASH_COMMANDS.iter().any(|c| c.name == skill.name) {
                entries.push(ComposerMenuEntry::Skill(skill));
            }
        }
        entries
    }

    fn slash_command_opens_inline_menu(&self, command: &SlashCommandSpec) -> bool {
        command.name == "attach"
    }

    fn complete_inline_arg_command(&mut self, command: &SlashCommandSpec) -> bool {
        if !self.slash_command_opens_inline_menu(command) {
            return false;
        }
        self.draft = command.insertion_text();
        self.cursor = self.draft.len();
        self.command_menu.reset();
        self.sync_command_menu_state();
        true
    }

    pub(super) fn visible_command_menu(&self) -> Option<VisibleCommandMenu> {
        // While recalling input history (↑/↓), keep arrows driving history
        // navigation — don't pop the dropdown for a recalled `/command` and let
        // it hijack the very keys used to keep scrolling. The menu returns once
        // the user edits the recalled draft (which leaves history navigation).
        if self.draft_history_index.is_some() {
            return None;
        }
        if self.command_menu.dismissed {
            return None;
        }
        let (kind, entries) = if self.composer_command_hint().is_some() {
            return None;
        } else if let Some(query) = self.active_command_query() {
            (MenuKind::Commands, self.matching_command_entries(query))
        } else if let Some((command, query)) = self.active_path_query() {
            (
                MenuKind::Path,
                // Suggest from the real launch dir (where relative paths actually
                // resolve at queue time), NOT chat's empty sandbox (`self.cwd`).
                collect_path_suggestions(self.persist_cwd(), command, query)
                    .into_iter()
                    .map(ComposerMenuEntry::Path)
                    .collect::<Vec<_>>(),
            )
        } else if let Some((_, query)) = self.active_mention_query() {
            (MenuKind::Mention, self.matching_mention_entries(&query))
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
        let (kind, query) = if let Some(query) = self.active_command_query() {
            (MenuKind::Commands, query.to_string())
        } else if let Some((_, query)) = self.active_path_query() {
            (MenuKind::Path, query.to_string())
        } else if let Some((_, query)) = self.active_mention_query() {
            (MenuKind::Mention, query)
        } else {
            self.command_menu.reset();
            return;
        };

        if self.command_menu.kind != Some(kind) || self.command_menu.query != query {
            if self.command_menu.dismissed {
                self.command_menu.placement = None;
            }
            self.command_menu.kind = Some(kind);
            self.command_menu.query = query.clone();
            self.command_menu.selected = 0;
            self.command_menu.dismissed = false;
        }

        let matches = if self.active_command_query().is_some() {
            self.matching_command_entries(&query).len()
        } else if let Some((command, _)) = self.active_path_query() {
            collect_path_suggestions(self.persist_cwd(), command, &query).len()
        } else {
            self.matching_mention_entries(&query).len()
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
        if (self.active_command_query().is_none()
            && self.active_path_query().is_none()
            && self.active_mention_query().is_none())
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
                if !self.complete_inline_arg_command(command) {
                    self.draft = command.insertion_text();
                    self.cursor = self.draft.len();
                    self.command_menu.dismissed = true;
                    self.command_menu.placement = None;
                }
            }
            ComposerMenuEntry::Skill(skill) => {
                self.draft = skill.insertion_text();
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
                // When the menu stays open (directory), preserve placement to avoid jumping.
                if !path.is_dir {
                    self.command_menu.placement = None;
                }
            }
            ComposerMenuEntry::Agent(agent) => {
                self.insert_mention(&agent.name);
            }
        }
        true
    }

    /// Replace the `@partial` at the cursor with `@name ` — the mention is part
    /// of a message still being composed, so the draft is never submitted here.
    fn insert_mention(&mut self, name: &str) {
        if let Some((at, _)) = self.active_mention_query() {
            let mention = format!("@{name} ");
            self.draft.replace_range(at..self.cursor, &mention);
            self.cursor = at + mention.len();
        }
        self.command_menu.dismissed = true;
        self.command_menu.placement = None;
    }

    pub(super) async fn execute_selected_command(&mut self) -> Result<bool> {
        let Some(entry) = self.selected_menu_entry() else {
            return self.submit_draft().await;
        };
        match entry {
            ComposerMenuEntry::Command(command) => {
                if self.complete_inline_arg_command(command) {
                    return Ok(false);
                }
                self.draft = command.command_label();
                self.cursor = self.draft.len();
                self.command_menu.reset();
                self.submit_draft().await
            }
            ComposerMenuEntry::Skill(skill) => {
                self.draft = skill.command_label();
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
            // Enter completes the mention like Tab — the user still has the
            // actual request to type around it.
            ComposerMenuEntry::Agent(agent) => {
                self.insert_mention(&agent.name);
                Ok(false)
            }
        }
    }

    pub(super) fn paste_system_clipboard(&mut self) -> Result<()> {
        match read_system_clipboard()? {
            ClipboardPayload::Text(text) => {
                if text.is_empty() {
                    self.notice = Some((MUTED(), "Clipboard is empty".to_string()));
                } else {
                    self.insert_pasted_text(&text);
                }
            }
            ClipboardPayload::Attachment(attachment) => {
                let kind = attachment_kind_label(&attachment);
                let name = attachment.name.clone();
                self.leave_history_navigation();
                let tag = attachment_tag(&attachment, self.draft_attachments.len() + 1);
                self.draft_attachments.push(attachment);
                self.insert_attachment_tag(&tag);
                self.notice = Some((
                    MUTED(),
                    format!("Pasted {kind}: {name} — delete {tag} to detach"),
                ));
            }
            ClipboardPayload::Empty => {
                self.notice = Some((MUTED(), "Clipboard is empty".to_string()));
            }
        }
        Ok(())
    }
}

pub(super) fn removed_attachment_notice(dropped: &[MessageAttachment]) -> String {
    match dropped {
        [one] => format!("Removed {}: {}", attachment_kind_label(one), one.name),
        many => format!(
            "Removed {} attachments: {}",
            many.len(),
            many.iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
