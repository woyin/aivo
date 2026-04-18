use super::*;

impl ChatTuiApp {
    pub(super) async fn handle_runtime_events(&mut self) -> Result<()> {
        while let Ok(event) = self.rx.try_recv() {
            self.handle_runtime_event(event).await?;
        }
        Ok(())
    }

    async fn handle_runtime_event(&mut self, event: RuntimeEvent) -> Result<()> {
        match event {
            RuntimeEvent::Delta(delta) => self.apply_runtime_delta(delta),
            RuntimeEvent::Finished { result, format } => {
                self.finish_response(result, format).await?;
            }
            RuntimeEvent::ModelsLoaded(result) => {
                self.apply_loaded_models(result).await?;
            }
            RuntimeEvent::ResumeLoaded { request_id, result } => {
                self.apply_resume_load_result(request_id, result).await?;
            }
        }
        Ok(())
    }

    fn apply_runtime_delta(&mut self, delta: ChatResponseChunk) {
        match delta {
            ChatResponseChunk::Content(text) => self.pending_response.push_str(&text),
            ChatResponseChunk::Reasoning(text) => self.pending_reasoning.push_str(&text),
        }
    }

    async fn finish_response(
        &mut self,
        result: std::result::Result<ChatTurnResult, String>,
        format: ChatFormat,
    ) -> Result<()> {
        self.sending = false;
        self.request_started_at = None;
        self.response_task = None;
        self.format = format;

        match result {
            Ok(turn) => self.finish_successful_response(turn).await?,
            Err(err) => self.finish_failed_response(err),
        }

        Ok(())
    }

    async fn finish_successful_response(&mut self, turn: ChatTurnResult) -> Result<()> {
        let content = if self.pending_response.is_empty() {
            turn.content.clone()
        } else {
            self.pending_response.clone()
        };
        let reasoning_content = if self.pending_reasoning.is_empty() {
            turn.reasoning_content.clone()
        } else {
            Some(self.pending_reasoning.clone())
        };
        self.pending_submit = None;
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content,
            reasoning_content,
            attachments: vec![],
        });

        let prompt_text: String = self
            .history
            .iter()
            .rev()
            .skip(1) // skip the assistant message we just pushed
            .map(|m| m.content.as_str())
            .collect();
        let usage = turn.usage_or_estimate(&prompt_text);
        self.session_store
            .record_tokens(
                &self.key.id,
                Some(&self.raw_model),
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cache_read_input_tokens,
                usage.cache_creation_input_tokens,
            )
            .await?;
        self.context_tokens = if turn.usage.is_some() {
            usage.total_tokens()
        } else {
            estimate_context_tokens(&self.history)
        };
        self.last_usage = turn.usage;

        let assistant_content = self
            .history
            .last()
            .map(|message| message.content.clone())
            .unwrap_or_default();
        let assistant_reasoning = self
            .history
            .last()
            .and_then(|message| message.reasoning_content.clone());
        let user_message = self
            .history
            .iter()
            .rev()
            .skip(1)
            .find(|message| message.role == "user")
            .cloned();
        if let Some(user_message) = user_message {
            let _ = log_chat_turn(
                &self.session_store,
                &self.key,
                &self.raw_model,
                Some(&self.cwd),
                Some(&self.session_id),
                &user_message,
                &assistant_content,
                assistant_reasoning.as_deref(),
                &usage,
            )
            .await;
        }

        self.persist_history().await?;
        self.notice = None;
        Ok(())
    }

    fn finish_failed_response(&mut self, err: String) {
        self.pending_response.clear();
        self.pending_reasoning.clear();
        restore_cancelled_submission(
            &mut self.history,
            &mut self.draft,
            &mut self.draft_attachments,
            &mut self.pending_submit,
        );
        self.notice = Some((ERROR, err));
    }

    async fn apply_loaded_models(
        &mut self,
        result: std::result::Result<Vec<ModelChoice>, String>,
    ) -> Result<()> {
        match result {
            Ok(models) => {
                if let Some(index) = self.populate_model_picker(models) {
                    self.activate_picker_selection(index).await?;
                }
            }
            Err(err) => {
                self.overlay = Overlay::None;
                self.notice = Some((ERROR, err));
            }
        }
        Ok(())
    }

    fn populate_model_picker(&mut self, models: Vec<ModelChoice>) -> Option<usize> {
        let Overlay::Picker(picker) = &mut self.overlay else {
            return None;
        };
        if !matches!(picker.kind, PickerKind::Model { .. }) {
            return None;
        }

        picker.items = models
            .into_iter()
            .map(|m| PickerEntry {
                search_text: m.id.clone(),
                label: m.label,
                value: PickerValue::Model(m.id),
            })
            .collect();
        picker.loading = false;
        picker.selected = 0;
        picker.exact_match_index()
    }

    async fn apply_resume_load_result(
        &mut self,
        request_id: u64,
        result: std::result::Result<LoadedSession, String>,
    ) -> Result<()> {
        let Some(loading) = &self.loading_resume else {
            return Ok(());
        };
        if loading.request_id != request_id {
            return Ok(());
        }

        self.resume_task = None;
        match result {
            Ok(session) => {
                self.apply_loaded_session(session).await?;
                self.loading_resume = None;
                self.resume_restore_state = None;
                self.notice = None;
            }
            Err(err) => {
                self.loading_resume = None;
                if let Some(state) = self.resume_restore_state.take() {
                    self.restore_resume_state(state);
                }
                self.notice = Some((ERROR, err));
            }
        }

        Ok(())
    }

    pub(super) async fn run(&mut self) -> Result<()> {
        let mut terminal = setup_terminal()?;
        let run_result = loop {
            self.frame_tick = self.frame_tick.wrapping_add(1);

            if let Err(err) = self.handle_runtime_events().await {
                break Err(err);
            }

            if self.pending_clear_screen {
                self.pending_clear_screen = false;
                if let Err(err) = terminal.clear() {
                    break Err(err.into());
                }
            }

            if let Err(err) = terminal.draw(|frame| self.render(frame)) {
                break Err(err.into());
            }

            match event::poll(Duration::from_millis(0)) {
                Ok(true) => match event::read() {
                    Ok(event) => {
                        if let Some(should_exit) = self.handle_terminal_event(event).await?
                            && should_exit
                        {
                            break Ok(());
                        }
                    }
                    Err(err) => break Err(err.into()),
                },
                Ok(false) => {}
                Err(err) => break Err(err.into()),
            }

            tokio::time::sleep(Duration::from_millis(16)).await;
        };

        // Abort in-flight tasks and await them so their futures are actually
        // dropped (closing any open HTTP connections) before we return. On the
        // current-thread runtime, `abort()` alone only schedules cancellation;
        // without the explicit `await` the task stays alive until the runtime
        // itself shuts down at process exit.
        let response_task = self.response_task.take();
        let resume_task = self.resume_task.take();
        self.loading_resume = None;
        self.resume_restore_state = None;
        if let Some(task) = response_task {
            task.abort();
            let _ = task.await;
        }
        if let Some(task) = resume_task {
            task.abort();
            let _ = task.await;
        }
        restore_terminal(terminal)?;
        run_result
    }

    async fn handle_terminal_event(&mut self, event: Event) -> Result<Option<bool>> {
        match event {
            Event::Key(key) => Ok(Some(self.handle_key(key).await?)),
            Event::Mouse(mouse) => Ok(Some(self.handle_mouse(mouse).await?)),
            Event::Resize(_, _) => Ok(None),
            Event::Paste(text) => {
                if !self.overlay.blocks_input() && !self.is_busy() {
                    self.insert_pasted_text(&text);
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    pub(super) async fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<bool> {
        if let Some(should_exit) = self.handle_overlay_mouse(mouse).await? {
            return Ok(should_exit);
        }

        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_up_lines(3),
            MouseEventKind::ScrollDown => self.scroll_down_lines(3),
            _ => {}
        }

        Ok(false)
    }

    async fn handle_overlay_mouse(&mut self, mouse: MouseEvent) -> Result<Option<bool>> {
        match (&self.overlay, mouse.kind) {
            (Overlay::Help, _) => Ok(Some(false)),
            (Overlay::Picker(picker), MouseEventKind::ScrollUp) if !picker.loading => {
                if let Overlay::Picker(picker) = &mut self.overlay {
                    picker.select_prev();
                }
                Ok(Some(false))
            }
            (Overlay::Picker(picker), MouseEventKind::ScrollDown) if !picker.loading => {
                if let Overlay::Picker(picker) = &mut self.overlay {
                    picker.select_next();
                }
                Ok(Some(false))
            }
            (Overlay::Picker(picker), MouseEventKind::Down(MouseButton::Left))
                if !picker.loading =>
            {
                self.handle_picker_click(mouse).await
            }
            (Overlay::Picker(_), _) => Ok(Some(false)),
            (Overlay::None, _) => Ok(None),
        }
    }

    async fn handle_picker_click(&mut self, mouse: MouseEvent) -> Result<Option<bool>> {
        let Some(hitbox) = &self.picker_hitbox else {
            return Ok(Some(false));
        };

        let point = (mouse.column, mouse.row);
        if rect_contains(hitbox.list_area, point) {
            let row = usize::from(mouse.row.saturating_sub(hitbox.list_area.y));
            if let Some(Some(filtered_index)) = hitbox.row_to_filtered_index.get(row) {
                return self
                    .activate_picker_selection(*filtered_index)
                    .await
                    .map(Some);
            }
        } else if !rect_contains(hitbox.overlay_area, point) {
            self.overlay = Overlay::None;
        }

        Ok(Some(false))
    }
}
