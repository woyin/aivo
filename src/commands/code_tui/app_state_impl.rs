use super::*;

impl CodeTuiApp {
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
            || self.installing_skill.is_some()
            || !self.incoming_buffer.is_empty()
            || self.toast.is_some()
            || self.drag_autoscroll.is_some()
            || self.selection_flash_until.is_some()
    }

    /// Advance the welcome-screen tip once its interval elapses; returns `true`
    /// when it changed (the cue to repaint). Only runs on the untouched welcome
    /// screen — an overlay, a draft, or any message pauses it and resets the clock,
    /// so the tip that reappears gets a full interval.
    pub(super) fn tick_welcome_tip(&mut self) -> bool {
        let composing = !self.draft.is_empty() || !self.draft_attachments.is_empty();
        let visible =
            self.is_transcript_empty() && matches!(self.overlay, Overlay::None) && !composing;
        if !visible {
            self.welcome_tip_rotated_at = None;
            return false;
        }
        let now = Instant::now();
        match self.welcome_tip_rotated_at {
            // Start the clock on the first frame; don't swap yet.
            None => {
                self.welcome_tip_rotated_at = Some(now);
                false
            }
            Some(shown_at) if now.duration_since(shown_at) >= WELCOME_TIP_ROTATE_INTERVAL => {
                self.welcome_tip_index = self.welcome_tip_index.wrapping_add(1);
                self.welcome_tip_rotated_at = Some(now);
                true
            }
            Some(_) => false,
        }
    }

    /// At the next user turn, drop a plan card that's fully completed (done marker)
    /// or unstarted (an abandoned proposal). A mid-execution plan is left alone.
    pub(super) fn clear_stale_plan(&mut self) {
        let stale = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "plan")
            .is_some_and(|m| plan_all_completed(&m.content) || plan_unstarted(&m.content));
        if stale {
            self.drop_plan_entries();
        }
    }

    /// Remove the pinned plan card(s), re-keying the index-keyed view maps
    /// (Done-in markers, reasoning durations, expansions, outputs) so later
    /// markers don't slide onto the wrong row.
    pub(super) fn drop_plan_entries(&mut self) {
        let plan_indices: Vec<usize> = self
            .history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == "plan")
            .map(|(i, _)| i)
            .collect();
        // Highest first, so each removal leaves the lower indices untouched.
        for &idx in plan_indices.iter().rev() {
            self.history.remove(idx);
            shift_index_map_after_removal(&mut self.turn_durations, idx);
            shift_index_map_after_removal(&mut self.turn_notes, idx);
            shift_index_map_after_removal(&mut self.reasoning_durations, idx);
            shift_index_map_after_removal(&mut self.local_outputs, idx);
            shift_index_set_after_removal(&mut self.expanded_thinking, idx);
            shift_index_set_after_removal(&mut self.expanded_output, idx);
            shift_index_opt_after_removal(&mut self.plan_card_idx, idx);
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
        // loads, a permission card holds the keyboard, or queue focus is active.
        !self.overlay.blocks_input()
            && self.loading_resume.is_none()
            && self.cards.permission().is_none()
            && self.queue_focus.is_none()
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
        self.jobs.set_logs_root(
            self.session_store
                .session_artifacts_dir(&self.session_id)
                .join("jobs"),
        );
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
        self.notice = Some((MUTED(), "Resume cancelled".to_string()));
    }

    pub(super) fn clear_for_resume_loading(&mut self) {
        self.history.clear();
        self.expanded_thinking.clear();
        self.expanded_output.clear();
        self.local_outputs.clear();
        self.reasoning_durations.clear();
        self.turn_durations.clear();
        self.turn_notes.clear();
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

/// Re-key a history-index map after the entry at `removed` is deleted: drop that
/// key, slide higher keys down by one.
fn shift_index_map_after_removal<V>(map: &mut std::collections::HashMap<usize, V>, removed: usize) {
    *map = std::mem::take(map)
        .into_iter()
        .filter_map(|(k, v)| match k {
            k if k == removed => None,
            k if k > removed => Some((k - 1, v)),
            k => Some((k, v)),
        })
        .collect();
}

/// Optional-scalar twin of [`shift_index_map_after_removal`].
fn shift_index_opt_after_removal(opt: &mut Option<usize>, removed: usize) {
    match *opt {
        Some(i) if i == removed => *opt = None,
        Some(i) if i > removed => *opt = Some(i - 1),
        _ => {}
    }
}

/// Set twin of [`shift_index_map_after_removal`].
fn shift_index_set_after_removal(set: &mut std::collections::HashSet<usize>, removed: usize) {
    *set = std::mem::take(set)
        .into_iter()
        .filter_map(|k| match k {
            k if k == removed => None,
            k if k > removed => Some(k - 1),
            k => Some(k),
        })
        .collect();
}
