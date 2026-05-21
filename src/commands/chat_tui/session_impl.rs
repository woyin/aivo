use super::*;
use crate::commands::models::fetch_models_for_select;
use crate::services::session_store::SessionTokens;

impl ChatTuiApp {
    pub(super) fn open_model_picker(
        &mut self,
        query: Option<String>,
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    ) {
        self.prepare_for_model_picker();
        let query = query.unwrap_or_default();
        self.overlay = Overlay::Picker(Box::new(PickerState::loading(
            "Select model",
            query,
            PickerKind::Model {
                target,
                auto_accept_exact,
            },
        )));
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = match self.current_model_picker_key() {
            Some(key) => key,
            None => return,
        };
        let cache = self.cache.clone();

        tokio::spawn(async move {
            let choices = load_model_choices(&client, &key, &cache).await;
            if choices.is_empty() {
                tx.send(RuntimeEvent::ModelsLoaded(Err(
                    "No models available for this provider".to_string(),
                )))
                .ok();
            } else {
                tx.send(RuntimeEvent::ModelsLoaded(Ok(choices))).ok();
            }
        });
    }

    pub(super) fn prepare_for_model_picker(&mut self) {
        if self.sending {
            self.cancel_inflight_request();
        }
    }

    pub(super) async fn apply_model(&mut self, raw_model: String) -> Result<()> {
        self.persist_model_selection(&raw_model).await?;

        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.billed_model = None;
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.notice = None;

        // If we have a live cursor ACP session, switch its model in place so
        // the conversation context is preserved. Drop the session on failure
        // so the next turn opens a fresh one with the new model.
        let drop_session = if let Some(session) = self.cursor_acp_session.as_mut() {
            session.set_model(&raw_model).await.is_err()
        } else {
            false
        };
        if drop_session {
            self.cursor_acp_session = None;
        }

        if !self.history.is_empty() {
            self.persist_history().await?;
        }
        Ok(())
    }

    pub(super) async fn complete_key_switch(
        &mut self,
        key: ApiKey,
        raw_model: String,
    ) -> Result<()> {
        self.key = key;
        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.billed_model = None;
        self.copilot_tm = copilot_token_manager_for_key(&self.key);
        self.persist_model_selection(&raw_model).await?;

        self.start_new_chat();
        Ok(())
    }

    pub(super) async fn open_or_switch_key(&mut self, query: Option<String>) -> Result<()> {
        if let Some(query) = query {
            if let Some(key) = self.resolve_key_exact(&query).await? {
                self.begin_key_switch(key).await?;
                return Ok(());
            }
            self.open_key_picker(Some(query)).await?;
            return Ok(());
        }

        self.open_key_picker(None).await
    }

    pub(super) async fn begin_key_switch(&mut self, mut key: ApiKey) -> Result<()> {
        SessionStore::decrypt_key_secret(&mut key)?;
        if let Some(raw_model) = self.session_store.get_chat_model(&key.id).await? {
            self.complete_key_switch(key, raw_model).await?;
        } else {
            self.overlay = Overlay::None;
            self.open_model_picker(None, ModelSelectionTarget::KeySwitch(key), false);
        }
        Ok(())
    }

    pub(super) async fn open_key_picker(&mut self, query: Option<String>) -> Result<()> {
        let keys = self.session_store.get_keys().await?;
        if keys.is_empty() {
            self.notice = Some((ERROR, "No saved keys".to_string()));
            return Ok(());
        }

        let items = keys
            .into_iter()
            .map(|key| PickerEntry {
                label: format!("{} · {}", key.display_name(), key.base_url),
                search_text: key_search_text(&key),
                value: PickerValue::Key(key),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Keys",
            query.unwrap_or_default(),
            items,
            PickerKind::Key,
        )));
        Ok(())
    }

    pub(super) async fn open_resume_picker(&mut self, query: Option<String>) -> Result<()> {
        let mut sessions = load_resume_snapshots(&self.session_store, &self.cwd).await?;
        if !self.history.is_empty()
            && !sessions
                .iter()
                .any(|session| session.session_id == self.session_id)
        {
            self.persist_history().await?;
            sessions = load_resume_snapshots(&self.session_store, &self.cwd).await?;
        }

        if let Some(query) = &query
            && let Some(snapshot) = sessions.iter().find(|session| session.session_id == *query)
        {
            self.begin_resume_load(snapshot.clone());
            return Ok(());
        }

        let items = sessions
            .into_iter()
            .map(|session| PickerEntry {
                label: session.title.clone(),
                search_text: session.search_text(),
                value: PickerValue::Session(session),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            query.unwrap_or_default(),
            items,
            PickerKind::Session,
        )));
        Ok(())
    }

    pub(super) fn open_help_overlay(&mut self) {
        self.overlay = Overlay::Help;
    }

    pub(super) async fn activate_picker_selection(
        &mut self,
        filtered_index: usize,
    ) -> Result<bool> {
        let (kind, value) = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((original_index, _)) = picker.filtered_items().get(filtered_index).copied()
            else {
                return Ok(false);
            };
            (
                picker.kind.clone(),
                picker.items[original_index].value.clone(),
            )
        };

        self.overlay = Overlay::None;

        match (kind, value) {
            (PickerKind::Model { target, .. }, PickerValue::Model(model)) => match target {
                ModelSelectionTarget::CurrentChat => self.apply_model(model).await?,
                ModelSelectionTarget::KeySwitch(key) => {
                    self.complete_key_switch(key, model).await?
                }
            },
            (PickerKind::Key, PickerValue::Key(key)) => {
                self.begin_key_switch(key).await?;
            }
            (PickerKind::Session, PickerValue::Session(session)) => {
                self.begin_resume_load(session);
            }
            _ => {}
        }

        Ok(false)
    }

    pub(super) async fn delete_picker_selection(&mut self, filtered_index: usize) -> Result<bool> {
        let session = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((_, item)) = picker.filtered_items().get(filtered_index).copied() else {
                return Ok(false);
            };
            match &item.value {
                PickerValue::Session(session) => session.clone(),
                _ => return Ok(false),
            }
        };

        let removed = self
            .session_store
            .delete_chat_session(&session.session_id)
            .await?;
        if !removed {
            self.notice = Some((ERROR, "Saved chat no longer exists".to_string()));
            return Ok(false);
        }

        if let Overlay::Picker(picker) = &mut self.overlay {
            picker.clear_pending_delete();
            picker.items.retain(|item| {
                !matches!(
                    &item.value,
                    PickerValue::Session(existing)
                        if existing.key_id == session.key_id && existing.session_id == session.session_id
                )
            });

            let filtered_len = picker.filtered_items().len();
            if filtered_len == 0 {
                self.overlay = Overlay::None;
                self.notice = Some((MUTED, "Saved chat deleted".to_string()));
                return Ok(false);
            }

            picker.selected = picker.selected.min(filtered_len.saturating_sub(1));
        }

        self.notice = Some((MUTED, "Saved chat deleted".to_string()));
        Ok(false)
    }

    pub(super) async fn resolve_key_exact(&self, query: &str) -> Result<Option<ApiKey>> {
        let keys = self.session_store.get_keys().await?;

        if let Some(key) = keys.iter().find(|key| key.id == query).cloned() {
            return Ok(Some(key));
        }

        let name_matches = keys
            .into_iter()
            .filter(|key| key.name == query)
            .collect::<Vec<_>>();

        if name_matches.len() == 1 {
            Ok(name_matches.into_iter().next())
        } else {
            Ok(None)
        }
    }

    pub(super) fn current_model_picker_key(&self) -> Option<ApiKey> {
        let Overlay::Picker(picker) = &self.overlay else {
            return None;
        };
        match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::CurrentChat,
                ..
            } => Some(self.key.clone()),
            PickerKind::Model {
                target: ModelSelectionTarget::KeySwitch(key),
                ..
            } => Some(key.clone()),
            _ => None,
        }
    }

    pub(super) async fn persist_history(&self) -> Result<()> {
        let stored = to_stored_messages(&self.history);
        let title = session_title_from_messages(&self.history, &self.raw_model);
        let preview = session_preview_text_from_messages(&self.history, &self.raw_model);
        // No TUI-side per-turn token accumulator yet; `aivo stats --since`
        // will under-report TUI chat tokens until that lands.
        self.session_store
            .save_chat_session_with_id(
                &self.key.id,
                &self.key.base_url,
                &self.cwd,
                &self.session_id,
                &self.raw_model,
                self.billed_model.as_deref(),
                &stored,
                &title,
                &preview,
                SessionTokens::default(),
            )
            .await
    }

    pub(super) fn begin_resume_load(&mut self, preview: SessionPreview) {
        self.discard_resume_state();
        self.overlay = Overlay::None;
        if self.sending {
            self.cancel_inflight_request();
        }

        self.resume_restore_state = Some(ResumeRestoreState::capture(self));
        self.clear_for_resume_loading();
        // The new session id will come from storage; drop any live cursor ACP
        // session since cursor doesn't know about the resumed session.
        self.cursor_acp_session = None;
        self.resume_request_id = self.resume_request_id.wrapping_add(1);
        let request_id = self.resume_request_id;
        self.loading_resume = Some(LoadingResume {
            request_id,
            preview: preview.clone(),
        });

        let session_store = self.session_store.clone();
        let tx = self.tx.clone();
        let task = tokio::spawn(async move {
            let result = load_resume_session(&session_store, &preview).await;
            let _ = tx.send(RuntimeEvent::ResumeLoaded { request_id, result });
        });
        self.resume_task = Some(task);
    }

    pub(super) async fn apply_loaded_session(&mut self, session: LoadedSession) -> Result<()> {
        if self.key.id != session.key_id {
            let key = self
                .session_store
                .get_key_by_id(&session.key_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Saved key for this chat is no longer available"))?;
            self.key = key;
            self.copilot_tm = copilot_token_manager_for_key(&self.key);
        }

        self.overlay = Overlay::None;
        self.session_id = session.session_id;
        self.history = session.messages;
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_submit = None;
        self.format = detect_initial_chat_format(&self.key.base_url);
        self.last_usage = None;
        self.context_tokens = estimate_context_tokens(&self.history);
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.raw_model = session.raw_model.clone();
        self.model =
            ChatCommand::transform_model_for_provider(&self.key.base_url, &session.raw_model);
        self.billed_model = None;
        self.persist_model_selection(&session.raw_model).await?;
        Ok(())
    }

    async fn persist_model_selection(&self, raw_model: &str) -> Result<()> {
        self.session_store
            .set_chat_model(&self.key.id, raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "chat", Some(raw_model))
            .await
    }

    pub(super) fn scroll_up(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(step.max(1));
    }

    pub(super) fn scroll_down(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + step.max(1)).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_up_lines(&mut self, lines: usize) {
        let max_scroll = self.max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(lines);
    }

    pub(super) fn scroll_down_lines(&mut self, lines: usize) {
        let max_scroll = self.max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + lines).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_to_top(&mut self) {
        self.transcript_scroll = 0;
        self.follow_output = false;
    }

    pub(super) fn scroll_to_bottom(&mut self) {
        self.transcript_scroll = self.max_scroll();
        self.follow_output = true;
    }

    pub(super) fn max_scroll(&self) -> usize {
        let transcript = self.build_transcript();
        let total = wrap_plain_lines(&transcript.plain_lines, self.transcript_width).len();
        total.saturating_sub(usize::from(self.transcript_view_height))
    }

    pub(super) fn selected_transcript_text(&self) -> Option<String> {
        let selection = self.transcript_selection?;
        let rows = &self.transcript_hitbox.as_ref()?.rows;
        selected_text_from_rows(rows, selection)
    }
}

/// Fetch model metadata for the picker, falling back to cached IDs on error.
/// On a successful detailed fetch we also seed the IDs cache so other commands
/// stay warm.
async fn load_model_choices(
    client: &reqwest::Client,
    key: &ApiKey,
    cache: &crate::services::ModelsCache,
) -> Vec<ModelChoice> {
    match crate::commands::models::fetch_models_detailed(client, key).await {
        Ok(infos) => {
            let ids: Vec<String> = infos.iter().map(|m| m.id.clone()).collect();
            let cache_key = crate::commands::models::model_cache_key_for_key(key);
            cache.set(&cache_key, ids).await;
            infos
                .into_iter()
                .map(|m| ModelChoice {
                    label: crate::commands::models::picker_label(&m),
                    id: m.id,
                })
                .collect()
        }
        Err(_) => fetch_models_for_select(client, key, cache)
            .await
            .into_iter()
            .map(|id| ModelChoice {
                label: id.clone(),
                id,
            })
            .collect(),
    }
}
