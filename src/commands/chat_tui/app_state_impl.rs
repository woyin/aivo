use super::*;

impl ChatTuiApp {
    pub(super) fn persist_draft_history(&self) {
        let _ = save_persisted_draft_history(&self.draft_history_all);
    }

    pub(super) fn is_busy(&self) -> bool {
        self.sending || self.loading_resume.is_some() || self.local_command.is_some()
    }

    /// The session id worth advertising on exit so the user can `--resume`
    /// straight back into this conversation. `None` for an untouched chat —
    /// `flush_for_exit` only persists a non-empty history, so an empty one has
    /// nothing to resume and shouldn't be hinted.
    pub(super) fn resumable_session_id(&self) -> Option<&str> {
        (!self.history.is_empty()).then_some(self.session_id.as_str())
    }

    /// Elements that repaint without input — the `sending` spinner/clock, the
    /// copy-toast fade, edge auto-scroll, and the post-copy selection flash.
    /// Gates the redraw cadence and `frame_tick`.
    pub(super) fn is_animating(&self) -> bool {
        self.sending
            || self.local_command.is_some()
            || !self.incoming_buffer.is_empty()
            || self.toast.is_some()
            || self.drag_autoscroll.is_some()
            || self.selection_flash_until.is_some()
    }

    /// Drop a fully-completed plan card so a finished checklist doesn't linger
    /// into the next task. A completed plan stays pinned (as a done marker) until
    /// the user sends their next message, at which point this clears it — see the
    /// pinned plan panel in `render_main`. An unfinished plan is left alone.
    pub(super) fn clear_completed_plan(&mut self) {
        let complete = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "plan")
            .is_some_and(|m| plan_all_completed(&m.content));
        if complete {
            self.history.retain(|m| m.role != "plan");
        }
    }

    /// Drops any active selection and its drag/flash/click state. Called when
    /// the transcript content shifts under the selection (new turn, /new,
    /// resume) so a stale, content-detached highlight can't linger.
    pub(super) fn clear_transcript_selection(&mut self) {
        self.transcript_selection = None;
        self.screen_selection = None;
        self.transcript_drag_active = false;
        self.screen_drag_active = false;
        self.drag_autoscroll = None;
        self.selection_flash_until = None;
        self.last_click = None;
    }

    pub(super) fn should_show_input_cursor(&self) -> bool {
        // Cursor stays live during a turn (type-to-queue), but not while a resume
        // loads or a permission card holds the keyboard.
        !self.overlay.blocks_input()
            && self.loading_resume.is_none()
            && self.agent_permission.is_none()
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
        self.pending_response = state.pending_response;
        self.pending_reasoning = state.pending_reasoning;
        self.pending_submit = state.pending_submit;
        self.last_usage = state.last_usage;
        self.live_usage = None;
        self.context_tokens = state.context_tokens;
        self.context_window = state.context_window;
        self.context_is_estimate = state.context_is_estimate;
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
        self.expanded_thinking.clear();
        self.reasoning_durations.clear();
        self.clear_transcript_selection();
        self.reset_composer();
        self.pending_response.clear();
        self.incoming_buffer.clear();
        self.pending_finish = None;
        self.pending_reasoning.clear();
        self.pending_submit = None;
        self.format = seeded_chat_format(&self.key, &self.raw_model);
        self.last_usage = None;
        self.live_usage = None;
        self.context_tokens = 0;
        self.context_is_estimate = true;
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.request_started_at = None;
        self.sending = false;
        self.notice = None;
    }
}
