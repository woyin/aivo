use super::*;

impl ChatTuiApp {
    pub(super) async fn submit_draft(&mut self) -> Result<bool> {
        let action = match self.prepare_submit_action() {
            Ok(action) => action,
            Err(err) => {
                self.notice = Some((ERROR, err.to_string()));
                return Ok(false);
            }
        };
        let Some(action) = action else {
            return Ok(false);
        };

        match action {
            SubmitAction::Send(input) => {
                if let Err(err) = self.send_user_message(input).await {
                    self.notice = Some((ERROR, err.to_string()));
                }
                Ok(false)
            }
            SubmitAction::Command(command) => match self.execute_slash_command(command).await {
                Ok(should_exit) => {
                    self.draft.clear();
                    self.cursor = 0;
                    self.command_menu.reset();
                    self.draft_history_index = None;
                    self.draft_history_stash = None;
                    Ok(should_exit)
                }
                Err(err) => {
                    self.notice = Some((ERROR, err.to_string()));
                    Ok(false)
                }
            },
        }
    }

    pub(super) fn prepare_submit_action(&self) -> Result<Option<SubmitAction>> {
        let trimmed = self.draft.trim();
        if trimmed.is_empty() {
            return if self.draft_attachments.is_empty() {
                Ok(None)
            } else {
                Ok(Some(SubmitAction::Send(String::new())))
            };
        }
        if self.draft.contains('\n') {
            return Ok(Some(SubmitAction::Send(trimmed.to_string())));
        }
        if let Some(escaped) = trimmed.strip_prefix("//") {
            return Ok(Some(SubmitAction::Send(format!("/{escaped}"))));
        }
        if let Some(command) = trimmed.strip_prefix('/') {
            return Ok(Some(SubmitAction::Command(parse_slash_command(command)?)));
        }
        Ok(Some(SubmitAction::Send(trimmed.to_string())))
    }

    pub(super) async fn send_user_message(&mut self, input: String) -> Result<()> {
        let attachments = materialize_attachments(&self.draft_attachments).await?;
        self.record_draft_history(&input);
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.overlay = Overlay::None;
        self.notice = None;
        self.last_usage = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = Some(PendingSubmission {
            content: input.clone(),
            attachments: attachments.clone(),
        });
        self.request_started_at = Some(Instant::now());
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content: input,
            reasoning_content: None,
            attachments,
        });
        trim_history(&mut self.history, MAX_HISTORY_MESSAGES);
        self.sending = true;
        self.follow_output = true;

        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = self.key.clone();
        let model = self.model.clone();
        let history = self.history.clone();
        let copilot_tm = self.copilot_tm.clone();
        let mut format = self.format.clone();

        self.response_task = Some(tokio::spawn(async move {
            let spinning = Arc::new(AtomicBool::new(false));
            let mut parser = ThinkTagParser::new();
            let result = {
                let mut on_chunk = |chunk: ChatResponseChunk| -> Result<()> {
                    match chunk {
                        ChatResponseChunk::Content(text) => {
                            for c in parser.feed(&text) {
                                tx.send(RuntimeEvent::Delta(c)).ok();
                            }
                        }
                        other => {
                            tx.send(RuntimeEvent::Delta(other)).ok();
                        }
                    }
                    Ok(())
                };

                send_message_turn(
                    &client,
                    &key,
                    copilot_tm.as_deref(),
                    &model,
                    &history,
                    &mut format,
                    &spinning,
                    false, // TUI always streams for live rendering
                    &mut on_chunk,
                )
                .await
            };

            for chunk in parser.flush() {
                tx.send(RuntimeEvent::Delta(chunk)).ok();
            }

            let result = result.map_err(|err| err.to_string());

            tx.send(RuntimeEvent::Finished { result, format }).ok();
        }));
        Ok(())
    }

    pub(super) fn queue_attachment(&mut self, path: String) -> Result<()> {
        let attachment = build_pending_attachment(&path)?;
        let name = attachment.name.clone();
        let kind = attachment_kind_label(&attachment);
        self.draft_attachments.push(attachment);
        self.notice = Some((MUTED, format!("Queued {kind}: {name}")));
        Ok(())
    }

    pub(super) fn detach_attachment(&mut self, index: usize) -> Result<()> {
        if index == 0 {
            anyhow::bail!("Usage: /detach <n> where n starts at 1");
        }
        let remove_at = index - 1;
        if remove_at >= self.draft_attachments.len() {
            anyhow::bail!(
                "No queued attachment #{index}. There {} {} queued.",
                if self.draft_attachments.len() == 1 {
                    "is"
                } else {
                    "are"
                },
                self.draft_attachments.len()
            );
        }
        let attachment = self.draft_attachments.remove(remove_at);
        let kind = attachment_kind_label(&attachment);
        self.notice = Some((MUTED, format!("Removed {kind}: {}", attachment.name)));
        Ok(())
    }

    pub(super) fn clear_draft_attachments(&mut self) {
        let count = self.draft_attachments.len();
        self.draft_attachments.clear();
        self.notice = Some((
            MUTED,
            if count == 0 {
                "No queued attachments".to_string()
            } else {
                format!(
                    "Cleared {count} attachment{}",
                    if count == 1 { "" } else { "s" }
                )
            },
        ));
    }

    pub(super) async fn execute_slash_command(&mut self, command: SlashCommand) -> Result<bool> {
        match command {
            SlashCommand::New => {
                self.start_new_chat();
                Ok(false)
            }
            SlashCommand::Exit => Ok(true),
            SlashCommand::Resume(query) => {
                self.open_resume_picker(query).await?;
                Ok(false)
            }
            SlashCommand::Model(query) => {
                let auto_accept_exact = query.is_some();
                self.open_model_picker(query, ModelSelectionTarget::CurrentChat, auto_accept_exact);
                Ok(false)
            }
            SlashCommand::Key(query) => {
                self.open_or_switch_key(query).await?;
                Ok(false)
            }
            SlashCommand::Attach(path) => {
                self.queue_attachment(path)?;
                Ok(false)
            }
            SlashCommand::Detach(index) => {
                self.detach_attachment(index)?;
                Ok(false)
            }
            SlashCommand::Clear => {
                self.clear_draft_attachments();
                Ok(false)
            }
            SlashCommand::Help => {
                self.open_help_overlay();
                Ok(false)
            }
        }
    }

    pub(super) fn push_newline(&mut self) {
        if !self.draft.is_empty() {
            self.leave_history_navigation();
            self.insert_char_at_cursor('\n');
        }
    }

    pub(super) fn reset_composer(&mut self) {
        self.draft.clear();
        self.draft_attachments.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    pub(super) fn start_new_chat(&mut self) {
        self.discard_resume_state();
        self.cancel_inflight_request();
        self.overlay = Overlay::None;
        self.history.clear();
        self.reset_composer();
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.sending = false;
        self.request_started_at = None;
        self.session_id = new_chat_session_id();
        self.format = detect_initial_chat_format(&self.key.base_url);
        self.last_usage = None;
        self.context_tokens = 0;
        self.follow_output = true;
        self.notice = None;
    }

    pub(super) fn cancel_inflight_request(&mut self) {
        if let Some(task) = self.response_task.take() {
            task.abort();
        }
        restore_cancelled_submission(
            &mut self.history,
            &mut self.draft,
            &mut self.draft_attachments,
            &mut self.pending_submit,
        );
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.follow_output = true;
        self.notice = Some((MUTED, "Request cancelled".to_string()));
    }

    pub(super) async fn interrupt_inflight_request(&mut self) -> Result<()> {
        if self.pending_response.is_empty() && self.pending_reasoning.is_empty() {
            self.cancel_inflight_request();
            return Ok(());
        }

        if let Some(task) = self.response_task.take() {
            task.abort();
        }

        let partial = std::mem::take(&mut self.pending_response);
        let reasoning = std::mem::take(&mut self.pending_reasoning);
        self.pending_submit = None;
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
        self.sending = false;
        self.request_started_at = None;
        self.follow_output = true;
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content: partial,
            reasoning_content: normalize_reasoning_content(reasoning),
            attachments: vec![],
        });
        self.context_tokens = estimate_context_tokens(&self.history);
        self.last_usage = None;
        self.persist_history().await?;
        self.notice = Some((MUTED, "Response interrupted".to_string()));
        Ok(())
    }

    pub(super) fn record_draft_history(&mut self, input: &str) {
        if input.is_empty() {
            return;
        }
        self.draft_history.push(input.to_string());
        self.draft_history_index = None;
        self.draft_history_stash = None;
    }

    pub(super) fn history_prev(&mut self) {
        if self.draft_history.is_empty() {
            return;
        }

        let next_index = match self.draft_history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.draft_history_stash = Some(self.draft.clone());
                self.draft_history.len().saturating_sub(1)
            }
        };

        self.draft_history_index = Some(next_index);
        self.draft = self.draft_history[next_index].clone();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn history_next(&mut self) {
        let Some(index) = self.draft_history_index else {
            return;
        };

        if index + 1 < self.draft_history.len() {
            let next_index = index + 1;
            self.draft_history_index = Some(next_index);
            self.draft = self.draft_history[next_index].clone();
            self.cursor = self.draft.len();
            self.sync_command_menu_state();
            return;
        }

        self.draft_history_index = None;
        self.draft = self.draft_history_stash.take().unwrap_or_default();
        self.cursor = self.draft.len();
        self.sync_command_menu_state();
    }

    pub(super) fn leave_history_navigation(&mut self) {
        if self.draft_history_index.is_some() && self.draft_history_stash.is_none() {
            self.draft_history_stash = Some(self.draft.clone());
        }
        self.draft_history_index = None;
    }
}
