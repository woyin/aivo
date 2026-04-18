use super::*;

enum OverlayKeyAction {
    NotOpen,
    Handled,
    SubmitPicker(usize),
    DeletePicker(usize),
}

impl ChatTuiApp {
    pub(super) async fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            if matches!(self.overlay, Overlay::None)
                && (!self.draft.is_empty() || !self.draft_attachments.is_empty())
            {
                self.reset_composer();
                return Ok(false);
            }
            return Ok(true);
        }

        if let Some(should_exit) = self.handle_overlay_key(key).await? {
            return Ok(should_exit);
        }

        if let Some(should_exit) = self.handle_global_key(key).await? {
            return Ok(should_exit);
        }

        if self.is_busy() {
            return Ok(false);
        }

        if let Some(should_exit) = self.handle_command_menu_key(key).await? {
            return Ok(should_exit);
        }

        self.handle_editor_key(key).await
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
        }
    }

    fn apply_overlay_key(&mut self, key: KeyEvent) -> OverlayKeyAction {
        match &mut self.overlay {
            Overlay::Help => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::F(1)) {
                    self.overlay = Overlay::None;
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

        let command_menu_shortcuts_active = self.command_menu_shortcuts_active();
        let handled = match key.code {
            KeyCode::Esc if self.loading_resume.is_some() => {
                self.cancel_resume_load();
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
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_down();
                true
            }
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
            KeyCode::Char('p')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none()
                    && !command_menu_shortcuts_active =>
            {
                self.history_prev();
                true
            }
            KeyCode::Char('n')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none()
                    && !command_menu_shortcuts_active =>
            {
                self.history_next();
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
            KeyCode::Char('t')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.loading_resume.is_none() =>
            {
                self.toggle_reasoning_visibility();
                true
            }
            _ => false,
        };

        Ok(handled.then_some(false))
    }

    fn command_menu_shortcuts_active(&self) -> bool {
        (self.active_command_query().is_some() || self.active_attach_query().is_some())
            && !self.command_menu.dismissed
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
                self.pending_clear_screen = true;
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
            KeyCode::Up => {
                if !self.draft.contains('\n') {
                    self.history_prev();
                } else {
                    self.cursor_up();
                }
            }
            KeyCode::Down => {
                if !self.draft.contains('\n') {
                    self.history_next();
                } else {
                    self.cursor_down();
                }
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

    pub(super) fn toggle_reasoning_visibility(&mut self) {
        self.show_reasoning = !self.show_reasoning;
        self.notice = Some((
            MUTED,
            if self.show_reasoning {
                "Thinking blocks shown".to_string()
            } else {
                "Thinking blocks hidden".to_string()
            },
        ));
    }
}
