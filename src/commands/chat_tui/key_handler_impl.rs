use super::*;

enum OverlayKeyAction {
    NotOpen,
    Handled,
    SubmitPicker(usize),
    DeletePicker(usize),
    ToggleSkill(usize),
    AddSkill(String),
    RemoveSkill(usize),
    ToggleMcpServer(usize),
    AddMcpServer(String),
    RemoveMcpServer(usize),
    AuthorizeMcpServer(usize),
    SignOutMcpServer(usize),
    ToggleConfigSetting(usize),
}

impl ChatTuiApp {
    pub(super) async fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            if self.exit_confirm_pending {
                return Ok(true);
            }
            self.exit_confirm_pending = true;
            self.notice = Some((WARNING, "Press Ctrl+C again to exit".to_string()));
            return Ok(false);
        }

        if self.exit_confirm_pending {
            self.exit_confirm_pending = false;
            self.notice = None;
        }

        if matches!(key.code, KeyCode::Esc) && self.transcript_selection.is_some() {
            self.transcript_selection = None;
            self.transcript_drag_active = false;
            return Ok(false);
        }

        // A project-MCP consent card (a repo's .mcp.json wants to spawn stdio
        // servers) owns the keyboard until decided: y run once, a always (this
        // repo), n/Esc deny. Unrecognized keys are ignored (card stays up).
        if self.pending_mcp_consent.is_some() {
            self.handle_mcp_consent_key(key).await;
            return Ok(false);
        }

        // A pending tool-permission card owns the decision keys (y/a/n), but only
        // while the composer is empty. If the user is mid-composing a queued
        // message those keystrokes belong to that message — letting them decide
        // would corrupt the draft and risk an accidental approval. Unconsumed keys
        // fall through to the editor so composing continues with the card still up.
        if self.agent_permission.is_some() {
            if self.handle_permission_key(key) {
                return Ok(false);
            }
            return self.handle_editor_key(key).await;
        }

        if let Some(should_exit) = self.handle_overlay_key(key).await? {
            return Ok(should_exit);
        }

        if let Some(should_exit) = self.handle_global_key(key).await? {
            return Ok(should_exit);
        }

        // Only block input while a resume is loading. During a turn we let the
        // composer stay live so the user can type/queue the next message; Enter
        // queues it (see submit_draft).
        if self.loading_resume.is_some() {
            return Ok(false);
        }

        if let Some(should_exit) = self.handle_command_menu_key(key).await? {
            return Ok(should_exit);
        }

        self.handle_editor_key(key).await
    }

    /// Resolve a pending tool-permission card; the decision is sent back to the
    /// waiting engine task. Returns `true` if the key was consumed as a decision
    /// or card chord, `false` if it should fall through to the composer (so the
    /// user can keep typing a queued message while the card stays up).
    fn handle_permission_key(&mut self, key: KeyEvent) -> bool {
        use crate::agent::protocol::Decision;
        // Shift+Tab while the card is up: enable auto-approve AND approve this
        // pending request, so "turn on auto-approve" takes effect on the request
        // in front of you — not just the next turn. (A card only shows when
        // auto-approve is off, so the toggle always means "turn on" here.)
        if is_auto_approve_toggle(key) {
            self.set_auto_approve(true);
            if let Some(pending) = self.agent_permission.take() {
                let _ = pending.reply.send(Decision::Allow);
            }
            return true;
        }
        // Esc denies regardless of the composer state — Esc is never message
        // content, so it can't be a stray keystroke from a queued draft.
        if matches!(key.code, KeyCode::Esc) {
            if let Some(pending) = self.agent_permission.take() {
                let _ = pending.reply.send(Decision::Deny);
            }
            return true;
        }
        // The single-letter decision keys only act on an EMPTY composer. While the
        // user is composing a queued message they're part of that text, so let
        // them fall through to the editor — a stray 'y'/'a' must never approve a
        // tool by accident, and the draft must not be clobbered.
        if !self.draft.is_empty() {
            return false;
        }
        let decision = match key.code {
            KeyCode::Char('y' | 'Y') => Decision::Allow,
            KeyCode::Char('a' | 'A') => Decision::AlwaysAllow,
            KeyCode::Char('n' | 'N') => Decision::Deny,
            _ => return false,
        };
        // "Always" on a Cursor card means session-wide auto-approve: cursor's
        // out-of-process tools can't be remembered per (tool,args) the way the
        // in-process engine scopes its own "always", so cursor_acp implements it
        // by flipping the shared live flag on. Route that through set_auto_approve
        // so the bool the badge and exit-persistence read agrees with the atomic —
        // otherwise the composer keeps reading "off" while Cursor silently allows
        // the rest of the session. (Native "always" stays scoped, badge unchanged.)
        if matches!(decision, Decision::AlwaysAllow)
            && self
                .agent_permission
                .as_ref()
                .is_some_and(|p| p.tool == "cursor")
        {
            self.set_auto_approve(true);
        }
        if let Some(pending) = self.agent_permission.take() {
            let _ = pending.reply.send(decision);
        }
        true
    }

    /// Set session auto-approve and mirror it to the shared live flag the running
    /// agent turn reads (native engine + cursor ACP), with a fading toast. A
    /// toast — not a persistent notice — so the confirmation flashes and vanishes
    /// instead of sitting pinned above the input for the rest of the session.
    pub(super) fn set_auto_approve(&mut self, on: bool) {
        self.agent_auto_approve = on;
        self.auto_approve_flag
            .store(on, std::sync::atomic::Ordering::Relaxed);
        self.show_toast(if on {
            "Auto-approve ON — the agent runs tools without asking"
        } else {
            "Auto-approve off — the agent will ask before write/edit/bash"
        });
    }

    async fn handle_overlay_key(&mut self, key: KeyEvent) -> Result<Option<bool>> {
        match self.apply_overlay_key(key) {
            OverlayKeyAction::NotOpen => Ok(None),
            OverlayKeyAction::Handled => Ok(Some(false)),
            OverlayKeyAction::SubmitPicker(selected) => {
                self.activate_picker_selection(selected).await.map(Some)
            }
            OverlayKeyAction::DeletePicker(selected) => {
                self.delete_picker_selection(selected).await.map(Some)
            }
            OverlayKeyAction::ToggleSkill(index) => {
                self.toggle_skill(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::AddSkill(input) => {
                self.submit_skill_add(input).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::RemoveSkill(index) => {
                self.remove_skill(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::ToggleMcpServer(index) => {
                self.toggle_mcp_server(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::AddMcpServer(input) => {
                self.submit_mcp_add(input).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::RemoveMcpServer(index) => {
                self.remove_mcp_server(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::AuthorizeMcpServer(index) => {
                self.authorize_mcp_server(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::SignOutMcpServer(index) => {
                self.sign_out_mcp_server(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::ToggleConfigSetting(index) => {
                self.toggle_config_setting(index).await;
                Ok(Some(false))
            }
        }
    }

    fn apply_overlay_key(&mut self, key: KeyEvent) -> OverlayKeyAction {
        match &mut self.overlay {
            Overlay::Help { .. } => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::F(1)) {
                    self.overlay = Overlay::None;
                } else if let Overlay::Help { scroll } = &mut self.overlay {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    apply_detail_scroll(scroll, key, ctrl);
                }
                OverlayKeyAction::Handled
            }
            Overlay::Config(state) => {
                // A small fixed toggle list: ↑/↓ (or Ctrl+P/N) move, Enter/Space/Tab
                // flip the row, Esc closes. No filter/add/remove.
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Esc => self.overlay = Overlay::None,
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Tab
                        if !state.items.is_empty() =>
                    {
                        return OverlayKeyAction::ToggleConfigSetting(state.selected);
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Output { .. } => {
                // The full `!cmd` output pager: Esc/Enter (or ctrl+o again) close;
                // everything else scrolls the body.
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter)
                    || (ctrl && matches!(key.code, KeyCode::Char('o')))
                {
                    self.overlay = Overlay::None;
                } else if let Overlay::Output { scroll } = &mut self.overlay {
                    apply_detail_scroll(scroll, key, ctrl);
                }
                OverlayKeyAction::Handled
            }
            Overlay::Skills(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Detail drill-in: scroll through the body; Esc/Enter/Tab back out.
                if state.viewing.is_some() {
                    apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Tab) {
                        state.viewing = None;
                        state.detail_scroll = 0;
                    }
                    return OverlayKeyAction::Handled;
                }
                // Add-input mode: keys edit the `name [description]` line.
                if let Some(buffer) = state.adding.as_mut() {
                    match key.code {
                        KeyCode::Esc => state.adding = None,
                        KeyCode::Enter => {
                            let input = std::mem::take(buffer);
                            state.adding = None;
                            return OverlayKeyAction::AddSkill(input);
                        }
                        KeyCode::Backspace => {
                            buffer.pop();
                        }
                        KeyCode::Char(c) if !ctrl => buffer.push(c),
                        _ => {}
                    }
                    return OverlayKeyAction::Handled;
                }
                // List mode: plain keys type into the filter; commands are on
                // Enter (toggle), Tab (view), Ctrl+A (add), Ctrl+D (remove).
                match key.code {
                    KeyCode::Esc if state.pending_delete.is_some() => state.pending_delete = None,
                    KeyCode::Esc if !state.query.is_empty() => {
                        state.query.clear();
                        state.refilter();
                    }
                    KeyCode::Esc => self.overlay = Overlay::None,
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    KeyCode::Enter if state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::ToggleSkill(state.selected);
                    }
                    KeyCode::Tab if state.has_selection() => {
                        state.pending_delete = None;
                        state.viewing = Some(state.selected);
                    }
                    KeyCode::Char('a') if ctrl => {
                        state.pending_delete = None;
                        state.adding = Some(String::new());
                    }
                    // First Ctrl+D arms the delete (folder removal needs a
                    // confirm), a second on the same row carries it out.
                    KeyCode::Char('d') if ctrl && state.has_selection() => {
                        let confirmed = state.arm_or_confirm_delete();
                        if confirmed {
                            return OverlayKeyAction::RemoveSkill(state.selected);
                        }
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Mcp(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Detail drill-in: scroll through the tools; Esc/Enter/Tab back out.
                if state.viewing.is_some() {
                    apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Tab) {
                        state.viewing = None;
                        state.detail_scroll = 0;
                    }
                    return OverlayKeyAction::Handled;
                }
                // Add-input mode: keys edit the `command args…` line, or Ctrl+V
                // pastes a `mcpServers` JSON block (the form READMEs hand you).
                if let Some(buffer) = state.adding.as_mut() {
                    match key.code {
                        KeyCode::Esc => state.adding = None,
                        KeyCode::Enter => {
                            let input = std::mem::take(buffer);
                            state.adding = None;
                            return OverlayKeyAction::AddMcpServer(input);
                        }
                        KeyCode::Backspace => {
                            buffer.pop();
                        }
                        KeyCode::Char('v') if ctrl => {
                            if let Ok(ClipboardPayload::Text(text)) = read_system_clipboard() {
                                buffer.push_str(&text);
                            }
                        }
                        KeyCode::Char(c) if !ctrl => buffer.push(c),
                        _ => {}
                    }
                    return OverlayKeyAction::Handled;
                }
                // List mode: plain keys type into the filter; commands are on
                // Enter (toggle), Tab (view), Ctrl+A (add), Ctrl+D (remove),
                // Ctrl+O (authorize an HTTP server), Ctrl+X (sign out).
                match key.code {
                    KeyCode::Esc if state.pending_delete.is_some() => state.pending_delete = None,
                    KeyCode::Esc if !state.query.is_empty() => {
                        state.query.clear();
                        state.refilter();
                    }
                    KeyCode::Esc => self.overlay = Overlay::None,
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    KeyCode::Enter if state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::ToggleMcpServer(state.selected);
                    }
                    KeyCode::Tab if state.has_selection() => {
                        state.pending_delete = None;
                        state.viewing = Some(state.selected);
                    }
                    KeyCode::Char('a') if ctrl => {
                        state.pending_delete = None;
                        state.adding = Some(String::new());
                    }
                    KeyCode::Char('o') if ctrl && state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::AuthorizeMcpServer(state.selected);
                    }
                    KeyCode::Char('x') if ctrl && state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::SignOutMcpServer(state.selected);
                    }
                    // First Ctrl+D arms the delete (removal edits the user
                    // mcp.json), a second on the same row carries it out — same
                    // two-press confirm as /skills and the resume picker.
                    KeyCode::Char('d') if ctrl && state.has_selection() => {
                        let confirmed = state.arm_or_confirm_delete();
                        if confirmed {
                            return OverlayKeyAction::RemoveMcpServer(state.selected);
                        }
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Picker(picker) => {
                if picker.loading {
                    if matches!(key.code, KeyCode::Esc) {
                        self.overlay = Overlay::None;
                    }
                    return OverlayKeyAction::Handled;
                }

                match key.code {
                    KeyCode::Esc => {
                        self.overlay = Overlay::None;
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Up => {
                        picker.clear_pending_delete();
                        picker.select_prev();
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.select_prev();
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Down => {
                        picker.clear_pending_delete();
                        picker.select_next();
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.select_next();
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Backspace => {
                        picker.clear_pending_delete();
                        picker.query.pop();
                        picker.selected = 0;
                        OverlayKeyAction::Handled
                    }
                    KeyCode::Enter => {
                        if picker.delete_is_armed_for_selected() {
                            OverlayKeyAction::DeletePicker(picker.selected)
                        } else {
                            OverlayKeyAction::SubmitPicker(picker.selected)
                        }
                    }
                    KeyCode::Char('d')
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(picker.kind, PickerKind::Session) =>
                    {
                        if picker.arm_or_confirm_delete() {
                            OverlayKeyAction::DeletePicker(picker.selected)
                        } else {
                            OverlayKeyAction::Handled
                        }
                    }
                    KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                        picker.clear_pending_delete();
                        picker.query.push(ch);
                        picker.selected = 0;
                        OverlayKeyAction::Handled
                    }
                    _ => OverlayKeyAction::Handled,
                }
            }
            Overlay::None => OverlayKeyAction::NotOpen,
        }
    }

    async fn handle_global_key(&mut self, key: KeyEvent) -> Result<Option<bool>> {
        if is_help_shortcut(key) {
            self.open_help_overlay();
            return Ok(Some(false));
        }

        // Shift+Tab toggles session auto-approve for the agent — aligned with
        // Claude Code's Shift+Tab permission-mode switch.
        if is_auto_approve_toggle(key) {
            self.set_auto_approve(!self.agent_auto_approve);
            return Ok(Some(false));
        }

        let handled = match key.code {
            KeyCode::Esc if self.loading_resume.is_some() => {
                self.cancel_resume_load();
                true
            }
            KeyCode::Esc if self.local_command.is_some() => {
                self.interrupt_local_command().await?;
                true
            }
            KeyCode::Esc if self.sending => {
                self.interrupt_inflight_request().await?;
                true
            }
            KeyCode::PageUp => {
                self.scroll_up();
                true
            }
            KeyCode::PageDown => {
                self.scroll_down();
                true
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_up();
                true
            }
            // Ctrl+D is left for the composer's delete-forward (emacs `delete-char`);
            // PageDown / Ctrl+Down scroll the transcript down.
            KeyCode::Up if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_up_lines(3);
                true
            }
            KeyCode::Down if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_down_lines(3);
                true
            }
            KeyCode::Home if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_top();
                true
            }
            KeyCode::End if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_bottom();
                true
            }
            KeyCode::Up if self.sending => {
                self.scroll_up_lines(3);
                true
            }
            KeyCode::Down if self.sending => {
                self.scroll_down_lines(3);
                true
            }
            KeyCode::Char('r')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.open_resume_picker(None).await?;
                true
            }
            KeyCode::Char('m')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.open_model_picker(None, ModelSelectionTarget::CurrentChat, false);
                true
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.open_output_overlay();
                true
            }
            _ => false,
        };

        Ok(handled.then_some(false))
    }

    async fn handle_command_menu_key(&mut self, key: KeyEvent) -> Result<Option<bool>> {
        let command_menu_visible = self.visible_command_menu().is_some();

        if matches!(key.code, KeyCode::Esc) && self.dismiss_command_menu() {
            return Ok(Some(false));
        }

        if matches!(key.code, KeyCode::Char('p'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            self.select_previous_command();
            return Ok(Some(false));
        }

        if matches!(key.code, KeyCode::Char('n'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            self.select_next_command();
            return Ok(Some(false));
        }

        if matches!(key.code, KeyCode::Enter)
            && !key.modifiers.contains(KeyModifiers::CONTROL)
            && command_menu_visible
        {
            return Ok(Some(self.execute_selected_command().await?));
        }

        if matches!(key.code, KeyCode::Tab) && self.insert_selected_command() {
            return Ok(Some(false));
        }

        Ok(None)
    }

    async fn handle_editor_key(&mut self, key: KeyEvent) -> Result<bool> {
        let command_menu_visible = self.visible_command_menu().is_some();

        match key.code {
            KeyCode::Enter if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                return self.submit_draft().await;
            }
            KeyCode::Tab => {}
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.push_newline();
                self.sync_command_menu_state();
            }
            KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if let Err(err) = self.paste_system_clipboard() {
                    self.notice = Some((ERROR, err.to_string()));
                }
            }
            KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.reset_composer();
            }
            KeyCode::Char('w') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.delete_word_backward();
                self.sync_command_menu_state();
            }
            KeyCode::Backspace if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.delete_word_backward();
                self.sync_command_menu_state();
            }
            KeyCode::Char('k') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.kill_to_end_of_line();
                self.sync_command_menu_state();
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_home();
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_end();
            }
            KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_left();
            }
            KeyCode::Char('f') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_right();
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                // Emacs `delete-char`: delete the character under the cursor (same
                // as the Delete key).
                self.delete_char_at_cursor();
                self.sync_command_menu_state();
            }
            KeyCode::Backspace => {
                self.leave_history_navigation();
                self.delete_char_before_cursor();
                self.sync_command_menu_state();
            }
            KeyCode::Delete => {
                self.delete_char_at_cursor();
                self.sync_command_menu_state();
            }
            KeyCode::Left if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_left();
            }
            KeyCode::Right if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor_word_right();
            }
            KeyCode::Left => {
                self.cursor_left();
            }
            KeyCode::Right => {
                self.cursor_right();
            }
            KeyCode::Home => {
                self.cursor_home();
            }
            KeyCode::End => {
                self.cursor_end();
            }
            KeyCode::Up if command_menu_visible => {
                self.select_previous_command();
            }
            KeyCode::Down if command_menu_visible => {
                self.select_next_command();
            }
            // Up / Ctrl+P move the cursor up one visual (wrapped) row; once on the
            // top row they recall the previous draft from history (Claude-Code
            // style: history at the edge). Ctrl+N / Down mirror it downward. A
            // single-line draft has one row, so these collapse to history nav.
            KeyCode::Up => self.composer_up_or_history_prev(),
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer_up_or_history_prev()
            }
            KeyCode::Down => self.composer_down_or_history_next(),
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer_down_or_history_next()
            }
            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.leave_history_navigation();
                self.insert_char_at_cursor(ch);
                self.sync_command_menu_state();
            }
            _ => {}
        }

        Ok(false)
    }
}

/// Adjust a detail drill-in's scroll offset for a nav key (Up/Down, Ctrl+P/N,
/// PageUp/PageDn, Home/End). `End` jumps to `u16::MAX`, which the renderer clamps
/// to the real bottom and writes back — so over-scrolling never strands the view.
fn apply_detail_scroll(scroll: &mut u16, key: KeyEvent, ctrl: bool) {
    *scroll = match key.code {
        KeyCode::Up => scroll.saturating_sub(1),
        KeyCode::Char('p') if ctrl => scroll.saturating_sub(1),
        KeyCode::Down => scroll.saturating_add(1),
        KeyCode::Char('n') if ctrl => scroll.saturating_add(1),
        KeyCode::PageUp => scroll.saturating_sub(DETAIL_PAGE_LINES),
        KeyCode::PageDown => scroll.saturating_add(DETAIL_PAGE_LINES),
        KeyCode::Home => 0,
        KeyCode::End => u16::MAX,
        _ => *scroll,
    };
}
