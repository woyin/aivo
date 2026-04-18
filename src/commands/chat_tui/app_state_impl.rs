use super::*;

impl ChatTuiApp {
    pub(super) fn persist_draft_history(&self) {
        let _ = save_persisted_draft_history(&self.draft_history);
    }

    pub(super) fn is_busy(&self) -> bool {
        self.sending || self.loading_resume.is_some()
    }

    pub(super) fn should_show_input_cursor(&self) -> bool {
        !self.overlay.blocks_input() && !self.is_busy()
    }

    pub(super) fn abort_resume_task(&mut self) {
        if let Some(task) = self.resume_task.take() {
            task.abort();
        }
    }

    pub(super) fn discard_resume_state(&mut self) {
        self.abort_resume_task();
        self.loading_resume = None;
        self.resume_restore_state = None;
    }

    pub(super) fn restore_resume_state(&mut self, state: ResumeRestoreState) {
        self.key = state.key;
        self.copilot_tm = state.copilot_tm;
        self.raw_model = state.raw_model;
        self.model = state.model;
        self.format = state.format;
        self.history = state.history;
        self.draft = state.draft;
        self.draft_attachments = state.draft_attachments;
        self.cursor = state.cursor;
        self.command_menu = state.command_menu;
        self.draft_history_index = state.draft_history_index;
        self.draft_history_stash = state.draft_history_stash;
        self.session_id = state.session_id;
        self.notice = state.notice;
        self.show_reasoning = state.show_reasoning;
        self.pending_response = state.pending_response;
        self.pending_reasoning = state.pending_reasoning;
        self.pending_submit = state.pending_submit;
        self.last_usage = state.last_usage;
        self.context_tokens = state.context_tokens;
        self.follow_output = state.follow_output;
        self.transcript_scroll = state.transcript_scroll;
        self.loading_resume = None;
        self.resume_restore_state = None;
        self.request_started_at = None;
        self.sending = false;
    }

    pub(super) fn cancel_resume_load(&mut self) {
        self.abort_resume_task();
        self.loading_resume = None;
        if let Some(state) = self.resume_restore_state.take() {
            self.restore_resume_state(state);
        }
        self.notice = Some((MUTED, "Resume cancelled".to_string()));
    }

    pub(super) fn clear_for_resume_loading(&mut self) {
        self.history.clear();
        self.reset_composer();
        self.pending_response.clear();
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.format = detect_initial_chat_format(&self.key.base_url);
        self.last_usage = None;
        self.context_tokens = 0;
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.request_started_at = None;
        self.sending = false;
        self.notice = None;
    }
}
