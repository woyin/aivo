use super::*;

enum OverlayKeyAction {
    NotOpen,
    Handled,
    SubmitPicker(usize),
    DeletePicker(usize),
    ToggleSkill(usize),
    AddSkill(String),
    RemoveSkill(usize),
    RemoveAgent(usize),
    InstallStagedSkills(Vec<String>),
    CancelSkillInstall,
    ToggleMcpServer(usize),
    AddMcpServer(String),
    RemoveMcpServer(usize),
    AuthorizeMcpServer(usize),
    SignOutMcpServer(usize),
    RetryMcpServers,
    OpenMcpTools(usize),
    ToggleMcpTool(usize),
    ApplyMcpPaste,
    /// `/config` row, direction −1/+1.
    StepConfigSetting(usize, i32),
    CycleConfigSetting(usize),
}

const GOAL_STOP_CONFIRM_NOTICE: &str = "Press Esc again to stop goal mode";
const QUEUE_ROW_GONE_NOTICE: &str = "That message was already picked up by the agent";

impl CodeTuiApp {
    pub(super) async fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        // Ctrl+X Ctrl+E chord: Ctrl+E completes it; any other key cancels and runs normally.
        if std::mem::take(&mut self.pending_ctrl_x)
            && matches!(key.code, KeyCode::Char('e'))
            && key.modifiers.contains(KeyModifiers::CONTROL)
        {
            self.pending_external_edit = true;
            return Ok(false);
        }

        if matches!(key.code, KeyCode::Char('c')) && key.modifiers.contains(KeyModifiers::CONTROL) {
            if self.exit_confirm_pending {
                return Ok(true);
            }
            self.exit_confirm_pending = true;
            // Mid-turn, a reflexive Ctrl+C usually means "stop the agent" — point
            // at esc before the second press tears down the TUI.
            self.notice = Some((
                WARNING(),
                if self.sending {
                    "esc stops the agent's turn — Ctrl+C again exits aivo".to_string()
                } else {
                    "Press Ctrl+C again to exit".to_string()
                },
            ));
            return Ok(false);
        }

        if self.exit_confirm_pending {
            self.exit_confirm_pending = false;
            self.notice = None;
        }

        // Any non-Esc key disarms the goal-stop confirm — the second Esc must be consecutive.
        if self.goal_stop_confirm_pending && !matches!(key.code, KeyCode::Esc) {
            self.goal_stop_confirm_pending = false;
            if matches!(&self.notice, Some((_, msg)) if msg.as_str() == GOAL_STOP_CONFIRM_NOTICE) {
                self.notice = None;
            }
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

        // The `/logout` confirm card likewise owns the keyboard until decided.
        if self.pending_logout.is_some() {
            self.handle_logout_confirm_key(key);
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

        // The `ask_user` card owns nav/selection on an empty composer; keys fall
        // through to the editor once a free-text answer is being typed.
        if self.agent_ask.is_some() {
            if self.handle_ask_user_key(key) {
                return Ok(false);
            }
            return self.handle_editor_key(key).await;
        }

        // The plan-approval card: nav/verdict on an empty composer; typed text
        // becomes keep-planning feedback (Enter submits it).
        if self.agent_plan_approval.is_some() {
            if self.handle_plan_approval_key(key) {
                return Ok(false);
            }
            return self.handle_editor_key(key).await;
        }

        // Same draft-guard as the permission card so a queued message isn't corrupted.
        if self.agent_review.is_some() {
            if self.handle_review_key(key) {
                return Ok(false);
            }
            return self.handle_editor_key(key).await;
        }

        // The `/login` status card consumes Enter/Esc only on an empty composer.
        if self.account_login.is_some() && self.handle_login_card_key(key) {
            return Ok(false);
        }

        if let Some(should_exit) = self.handle_overlay_key(key).await? {
            return Ok(should_exit);
        }

        if let Some(should_exit) = self.handle_queue_focus_key(key) {
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
    pub(super) fn handle_permission_key(&mut self, key: KeyEvent) -> bool {
        use crate::agent::protocol::Decision;
        // Shift+Tab: allow this request AND turn on auto-approve, so the chord
        // takes effect on the card in front of you. In plan mode there's no
        // auto-approve — just allow this one call; the session stays read-only.
        if is_auto_approve_toggle(key) {
            if self.plan_mode {
                if let Some(pending) = self.agent_permission.take() {
                    let _ = pending.reply.send(Decision::Allow);
                }
                self.show_toast("Allowed once — plan mode stays read-only");
                return true;
            }
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

    /// Resolve a key against the `ask_user` card. On an empty composer: ↑/↓
    /// (Ctrl+P/N) move, a digit picks, Enter picks the highlighted, Esc dismisses.
    /// While a free-text answer is being typed, Enter submits it. Returns `true`
    /// when consumed, `false` to hand the key to the editor.
    fn handle_ask_user_key(&mut self, key: KeyEvent) -> bool {
        let Some(ask) = self.agent_ask.as_ref() else {
            return false;
        };
        let len = ask.options.len();
        let allow_free_text = ask.allow_free_text;
        let multi = ask.multi_select;

        // Esc dismisses regardless of composer state — Esc is never message text.
        if matches!(key.code, KeyCode::Esc) {
            self.dismiss_ask_user();
            return true;
        }

        // Mid-composing a free-text answer: Enter submits the draft as the answer;
        // every other key falls through so the user keeps typing.
        if !self.draft.is_empty() {
            if allow_free_text
                && matches!(key.code, KeyCode::Enter)
                && !key.modifiers.contains(KeyModifiers::CONTROL)
            {
                let answer = self.draft.trim().to_string();
                if answer.is_empty() {
                    return false;
                }
                self.record_draft_history(&answer);
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
                self.draft_history_index = None;
                self.draft_history_stash = None;
                self.answer_ask_user(answer);
                return true;
            }
            return false;
        }

        // Empty composer: the card owns navigation + selection.
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up => {
                self.move_ask_selection(-1);
                true
            }
            KeyCode::Char('p') if ctrl => {
                self.move_ask_selection(-1);
                true
            }
            KeyCode::Down => {
                self.move_ask_selection(1);
                true
            }
            KeyCode::Char('n') if ctrl => {
                self.move_ask_selection(1);
                true
            }
            // Multi-select: space toggles the highlighted box.
            KeyCode::Char(' ') if multi => {
                let idx = self.agent_ask.as_ref().map(|a| a.selected).unwrap_or(0);
                self.toggle_ask_check(idx);
                true
            }
            KeyCode::Enter if !ctrl => {
                if multi {
                    self.confirm_ask_multi();
                } else {
                    let idx = self.agent_ask.as_ref().map(|a| a.selected).unwrap_or(0);
                    self.select_ask_option(idx);
                }
                true
            }
            // A 1–9 digit picks that option (single-select) or toggles its box (multi).
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let idx = (c as usize) - ('1' as usize);
                if idx < len {
                    if multi {
                        self.toggle_ask_check(idx);
                    } else {
                        self.select_ask_option(idx);
                    }
                    true
                } else {
                    !allow_free_text
                }
            }
            // Otherwise fall through to the editor when free text is allowed, else
            // the card swallows the key.
            _ => !allow_free_text,
        }
    }

    /// Move the `ask_user` highlight by `delta`, clamped to the option range.
    fn move_ask_selection(&mut self, delta: isize) {
        if let Some(ask) = self.agent_ask.as_mut() {
            let last = ask.options.len().saturating_sub(1);
            ask.selected = (ask.selected as isize + delta).clamp(0, last as isize) as usize;
        }
    }

    /// Pick option `idx` and send its label back to the waiting engine task.
    fn select_ask_option(&mut self, idx: usize) {
        let Some(ask) = self.agent_ask.take() else {
            return;
        };
        let answer = ask
            .options
            .get(idx)
            .map(|o| o.label.clone())
            .unwrap_or_default();
        let _ = ask.reply.send(Ok(answer));
    }

    /// Toggle the checkbox for option `idx` in a multi-select card.
    fn toggle_ask_check(&mut self, idx: usize) {
        if let Some(ask) = self.agent_ask.as_mut()
            && let Some(c) = ask.checked.get_mut(idx)
        {
            *c = !*c;
        }
    }

    /// Confirm a multi-select card: send the checked labels joined by ", " (an empty
    /// selection sends "none" so the model still gets an explicit answer, not a hang).
    fn confirm_ask_multi(&mut self) {
        let Some(ask) = self.agent_ask.take() else {
            return;
        };
        let picked: Vec<String> = ask
            .options
            .iter()
            .zip(ask.checked.iter())
            .filter(|(_, checked)| **checked)
            .map(|(o, _)| o.label.clone())
            .collect();
        let answer = if picked.is_empty() {
            "none".to_string()
        } else {
            picked.join(", ")
        };
        let _ = ask.reply.send(Ok(answer));
    }

    /// Send a free-text answer back to the waiting engine task.
    fn answer_ask_user(&mut self, answer: String) {
        if let Some(ask) = self.agent_ask.take() {
            let _ = ask.reply.send(Ok(answer));
        }
    }

    /// Dismiss the card without answering — the engine gets the stop-don't-decide
    /// directive as the tool result.
    fn dismiss_ask_user(&mut self) {
        if let Some(ask) = self.agent_ask.take() {
            let _ = ask
                .reply
                .send(Err(crate::agent::ask::DISMISSED_DIRECTIVE.to_string()));
        }
    }

    /// Resolve a key against the plan-approval card. Empty composer: ↑/↓ (Ctrl+P/N)
    /// move the highlight, PgUp/PgDn scroll the plan, 1–3 pick, Enter picks the
    /// highlighted, `y` approves. Typed text is keep-planning feedback — Enter
    /// submits it; Esc dismisses (plan mode stays on). `true` when consumed.
    fn handle_plan_approval_key(&mut self, key: KeyEvent) -> bool {
        use crate::agent::protocol::PlanDecision;
        if self.agent_plan_approval.is_none() {
            return false;
        }
        // Esc dismisses regardless of composer state — Esc is never message text.
        if matches!(key.code, KeyCode::Esc) {
            if let Some(pending) = self.agent_plan_approval.take() {
                let _ = pending.reply.send(Err(
                    crate::agent::plan_mode::PLAN_APPROVAL_DISMISSED.to_string()
                ));
            }
            self.show_toast("Plan not approved — still in plan mode");
            return true;
        }
        // Mid-composing feedback: Enter sends it as "keep planning"; every other
        // key falls through so the user keeps typing.
        if !self.draft.is_empty() {
            if matches!(key.code, KeyCode::Enter) && !key.modifiers.contains(KeyModifiers::CONTROL)
            {
                let feedback = self.draft.trim().to_string();
                if feedback.is_empty() {
                    return false;
                }
                self.record_draft_history(&feedback);
                self.draft.clear();
                self.cursor = 0;
                self.command_menu.reset();
                self.draft_history_index = None;
                self.draft_history_stash = None;
                self.resolve_plan_approval(
                    PlanDecision::KeepPlanning {
                        feedback: Some(feedback),
                    },
                    false,
                );
                return true;
            }
            return false;
        }
        // Empty composer: the card owns navigation + the verdict keys.
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up => {
                self.move_plan_approval_selection(-1);
                true
            }
            KeyCode::Char('p') if ctrl => {
                self.move_plan_approval_selection(-1);
                true
            }
            KeyCode::Down => {
                self.move_plan_approval_selection(1);
                true
            }
            KeyCode::Char('n') if ctrl => {
                self.move_plan_approval_selection(1);
                true
            }
            KeyCode::PageUp => {
                self.scroll_plan_approval(-5);
                true
            }
            KeyCode::PageDown => {
                self.scroll_plan_approval(5);
                true
            }
            KeyCode::Enter if !ctrl => {
                let idx = self
                    .agent_plan_approval
                    .as_ref()
                    .map(|p| p.selected)
                    .unwrap_or(0);
                self.pick_plan_approval_option(idx);
                true
            }
            KeyCode::Char(c) if ('1'..='3').contains(&c) => {
                self.pick_plan_approval_option((c as usize) - ('1' as usize));
                true
            }
            // Anything else falls through to the editor so feedback can be typed.
            // (No `y` accelerator: options 1 and 2 both approve — ambiguous.)
            _ => false,
        }
    }

    /// Move the plan-approval highlight by `delta`, clamped to the 3 options.
    fn move_plan_approval_selection(&mut self, delta: isize) {
        if let Some(pending) = self.agent_plan_approval.as_mut() {
            pending.selected = (pending.selected as isize + delta).clamp(0, 2) as usize;
        }
    }

    /// Scroll the plan body by `delta` rows; render clamps and writes back.
    fn scroll_plan_approval(&mut self, delta: isize) {
        if let Some(pending) = self.agent_plan_approval.as_mut() {
            let max = (pending.body.len().saturating_sub(1)) as isize;
            pending.scroll = (pending.scroll as isize + delta).clamp(0, max.max(0)) as u16;
        }
    }

    /// Resolve by option index — approval also picks the exit mode: 0 approve +
    /// auto, 1 approve + review, 2 keep planning. (Discard = Esc then `/plan stop`.)
    pub(super) fn pick_plan_approval_option(&mut self, idx: usize) {
        use crate::agent::protocol::PlanDecision;
        match idx {
            0 => self.resolve_plan_approval(PlanDecision::Approve, true),
            1 => self.resolve_plan_approval(PlanDecision::Approve, false),
            2 => self.resolve_plan_approval(PlanDecision::KeepPlanning { feedback: None }, false),
            _ => {}
        }
    }

    /// Send the verdict to the waiting engine task and flip the TUI mode: approval
    /// exits plan mode into auto or review mode (per `auto_approve`).
    pub(super) fn resolve_plan_approval(
        &mut self,
        decision: crate::agent::protocol::PlanDecision,
        auto_approve: bool,
    ) {
        use crate::agent::protocol::PlanDecision;
        let Some(pending) = self.agent_plan_approval.take() else {
            return;
        };
        match &decision {
            PlanDecision::Approve => {
                self.plan_mode = false;
                self.pending_plan = None;
                self.plan_card_idx = None;
                self.set_auto_quiet(auto_approve);
                self.set_review_quiet(!auto_approve);
                self.show_toast(if auto_approve {
                    "Plan approved — executing with auto-approve"
                } else {
                    "Plan approved — executing; each edit shows a diff to approve"
                });
            }
            PlanDecision::KeepPlanning { feedback } => {
                self.show_toast(if feedback.is_some() {
                    "Feedback sent — still planning"
                } else {
                    "Keeping planning — still read-only"
                });
            }
            // Unreachable from the card (no discard option); kept for completeness.
            PlanDecision::Discard => {
                self.plan_mode = false;
                self.pending_plan = None;
                self.plan_card_idx = None;
                self.plan_exit_pending = true;
            }
        }
        let _ = pending.reply.send(Ok(decision));
    }

    /// Resolve a key against the edit-review card: on an empty composer `y`/Enter
    /// approve, `n` rejects, arrows scroll. Esc always rejects; decision/scroll keys
    /// fall through while a queued message is being typed. `true` when consumed.
    fn handle_review_key(&mut self, key: KeyEvent) -> bool {
        if self.agent_review.is_none() {
            return false;
        }
        // Esc rejects regardless of composer state — Esc is never message text.
        if matches!(key.code, KeyCode::Esc) {
            self.resolve_review(crate::agent::review::ReviewDecision::Reject);
            return true;
        }
        if !self.draft.is_empty() {
            return false;
        }
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Up | KeyCode::PageUp => {
                self.scroll_review(-1);
                true
            }
            KeyCode::Char('p') if ctrl => {
                self.scroll_review(-1);
                true
            }
            KeyCode::Down | KeyCode::PageDown => {
                self.scroll_review(1);
                true
            }
            KeyCode::Char('n') if ctrl => {
                self.scroll_review(1);
                true
            }
            KeyCode::Char('y') | KeyCode::Enter => {
                self.resolve_review(crate::agent::review::ReviewDecision::ApproveAll);
                true
            }
            KeyCode::Char('n') => {
                self.resolve_review(crate::agent::review::ReviewDecision::Reject);
                true
            }
            // The card owns every other key while the composer is empty.
            _ => true,
        }
    }

    /// Scroll the review body by `delta` rows; render clamps and writes back.
    fn scroll_review(&mut self, delta: isize) {
        if let Some(review) = self.agent_review.as_mut() {
            let max = (review.body.len().saturating_sub(1)) as isize;
            review.scroll = (review.scroll as isize + delta).clamp(0, max.max(0)) as u16;
        }
    }

    /// Send the review verdict back to the waiting engine task and close the card.
    fn resolve_review(&mut self, decision: crate::agent::review::ReviewDecision) {
        if let Some(review) = self.agent_review.take() {
            let _ = review.reply.send(decision);
        }
    }

    /// Shift+Tab: cycle normal → auto-approve → plan → review → normal. Plan entry
    /// falls through to review mid-turn (the engine can't be restricted while a
    /// turn holds it) or when the key lacks the agent.
    pub(super) async fn cycle_agent_mode(&mut self) {
        if self.plan_mode {
            self.leave_plan_mode(false).await;
            self.set_review_quiet(true);
            self.show_toast("Review mode — approve each edit");
        } else if self.agent_review_edits {
            self.set_review_quiet(false);
            self.show_toast("Normal mode — risky actions ask first");
        } else if self.agent_auto_approve {
            self.set_auto_quiet(false);
            if !self.sending && self.enter_plan_mode().await {
                self.show_toast("Plan mode — read-only until you approve");
            } else {
                self.set_review_quiet(true);
                self.show_toast("Review mode — approve each edit");
            }
        } else {
            self.set_auto_quiet(true);
            self.show_toast("Auto-approve mode — tools run without asking");
        }
    }

    /// Flip auto-approve (field + live atomic) without a toast.
    pub(super) fn set_auto_quiet(&mut self, on: bool) {
        self.agent_auto_approve = on;
        self.auto_approve_flag
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Flip edit-review (field + live atomic) without a toast.
    pub(super) fn set_review_quiet(&mut self, on: bool) {
        self.agent_review_edits = on;
        self.review_edits_flag
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set session auto-approve and mirror it to the shared live flag the running
    /// agent turn reads (native engine + cursor ACP), with a fading toast. A
    /// toast — not a persistent notice — so the confirmation flashes and vanishes
    /// instead of sitting pinned above the input for the rest of the session.
    pub(super) fn set_auto_approve(&mut self, on: bool) {
        self.set_auto_quiet(on);
        self.show_toast(if on {
            "Auto-approve mode — tools run without asking"
        } else {
            "Normal mode — risky actions ask first"
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
            OverlayKeyAction::RemoveAgent(index) => {
                self.remove_agent(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::InstallStagedSkills(names) => {
                self.install_staged_skills(names).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::CancelSkillInstall => {
                self.cancel_skill_install().await?;
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
            OverlayKeyAction::RetryMcpServers => {
                self.retry_mcp_failed();
                Ok(Some(false))
            }
            OverlayKeyAction::OpenMcpTools(index) => {
                self.open_mcp_tools(index);
                Ok(Some(false))
            }
            OverlayKeyAction::ToggleMcpTool(index) => {
                self.toggle_mcp_tool(index).await?;
                Ok(Some(false))
            }
            OverlayKeyAction::ApplyMcpPaste => {
                self.apply_mcp_paste().await?;
                Ok(Some(false))
            }
            OverlayKeyAction::StepConfigSetting(row, dir) => {
                self.step_config_setting(row, dir).await;
                Ok(Some(false))
            }
            OverlayKeyAction::CycleConfigSetting(row) => {
                self.cycle_config_setting(row).await;
                Ok(Some(false))
            }
        }
    }

    /// Route a bracketed paste into the focused overlay text input (add field,
    /// else filter). Control chars collapse to spaces — the inputs are one line,
    /// and JSON pasted into `/mcp` stays valid. `false` = not consumed.
    pub(super) fn overlay_paste(&mut self, text: &str) -> bool {
        let clean: String = text
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        let clean = clean.trim();
        match &mut self.overlay {
            Overlay::Skills(state) => {
                if let Some(buffer) = state.adding.as_mut() {
                    buffer.push_str(clean);
                } else {
                    state.query.push_str(clean);
                    state.refilter();
                }
                true
            }
            Overlay::Mcp(state) => {
                if let Some(buffer) = state.adding.as_mut() {
                    buffer.push_str(clean);
                } else {
                    state.query.push_str(clean);
                    state.refilter();
                }
                true
            }
            Overlay::Agents(state) => {
                state.query.push_str(clean);
                state.refilter();
                true
            }
            Overlay::SkillInstall(state) => {
                state.query.push_str(clean);
                state.refilter();
                true
            }
            Overlay::McpTools(state) => {
                state.query.push_str(clean);
                state.refilter();
                true
            }
            Overlay::McpPaste(state) => {
                state.query.push_str(clean);
                state.refilter();
                true
            }
            Overlay::Picker(picker) => {
                picker.clear_pending_delete();
                picker.query.push_str(clean);
                picker.selected = 0;
                true
            }
            Overlay::Help { .. }
            | Overlay::Context { .. }
            | Overlay::Session { .. }
            | Overlay::Config(_)
            | Overlay::None => false,
        }
    }

    fn apply_overlay_key(&mut self, key: KeyEvent) -> OverlayKeyAction {
        // Split active last frame: the right pane owns the page keys, Tab drill-in is off.
        let split = self.overlay_detail_area.is_some();
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
            Overlay::Context { .. } => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                    self.overlay = Overlay::None;
                } else if let Overlay::Context { scroll, .. } = &mut self.overlay {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    apply_detail_scroll(scroll, key, ctrl);
                }
                OverlayKeyAction::Handled
            }
            Overlay::Session { .. } => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Enter) {
                    self.overlay = Overlay::None;
                } else if let Overlay::Session { scroll } = &mut self.overlay {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    apply_detail_scroll(scroll, key, ctrl);
                }
                OverlayKeyAction::Handled
            }
            Overlay::Config(state) => {
                // ↑/↓ (Ctrl+P/N) move rows, ←/→ change the value, Enter/Space/Tab
                // advance it (wrapping), Esc closes.
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                let has_rows = !state.items.is_empty();
                match key.code {
                    KeyCode::Esc => self.overlay = Overlay::None,
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    KeyCode::Left if has_rows => {
                        return OverlayKeyAction::StepConfigSetting(state.selected, -1);
                    }
                    KeyCode::Right if has_rows => {
                        return OverlayKeyAction::StepConfigSetting(state.selected, 1);
                    }
                    KeyCode::Enter | KeyCode::Char(' ') | KeyCode::Tab if has_rows => {
                        return OverlayKeyAction::CycleConfigSetting(state.selected);
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Skills(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Narrow-only drill-in: scroll the body; Esc/Enter/Tab back out.
                if state.viewing.is_some() && !split {
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
                    // Space toggles too — a dead key in a fuzzy filter, so it never types.
                    KeyCode::Enter | KeyCode::Char(' ') if state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::ToggleSkill(state.selected);
                    }
                    KeyCode::Tab if state.has_selection() && !split => {
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
                    // Page keys scroll the split's right detail pane.
                    KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End if split => {
                        apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Agents(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Narrow-only drill-in: scroll the body; Esc/Enter/Tab back out.
                if state.viewing.is_some() && !split {
                    apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Tab) {
                        state.viewing = None;
                        state.detail_scroll = 0;
                    }
                    return OverlayKeyAction::Handled;
                }
                // List mode: plain keys type into the filter; no toggle/add —
                // Enter/Tab view (narrow), Ctrl+D removes the file (two-press).
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
                    KeyCode::Enter | KeyCode::Tab if state.has_selection() && !split => {
                        state.pending_delete = None;
                        state.viewing = Some(state.selected);
                    }
                    KeyCode::Char('d') if ctrl && state.has_selection() => {
                        let confirmed = state.arm_or_confirm_delete();
                        if confirmed {
                            return OverlayKeyAction::RemoveAgent(state.selected);
                        }
                    }
                    // Page keys scroll the split's right detail pane.
                    KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End if split => {
                        apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::SkillInstall(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Narrow-only drill-in: scroll the body; Esc/Enter/Tab back out.
                if state.viewing.is_some() && !split {
                    apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    if matches!(key.code, KeyCode::Esc | KeyCode::Enter | KeyCode::Tab) {
                        state.viewing = None;
                        state.detail_scroll = 0;
                    }
                    return OverlayKeyAction::Handled;
                }
                match key.code {
                    KeyCode::Esc if !state.query.is_empty() => {
                        state.query.clear();
                        state.refilter();
                    }
                    KeyCode::Esc => return OverlayKeyAction::CancelSkillInstall,
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    // Space never types (dead in the fuzzy filter); on an
                    // installed row the mark means update-in-place.
                    KeyCode::Char(' ') if state.has_selection() => {
                        if let Some(item) = state.items.get_mut(state.selected) {
                            item.checked = !item.checked;
                        }
                    }
                    KeyCode::Enter => {
                        let names = state.pick_names();
                        if !names.is_empty() {
                            return OverlayKeyAction::InstallStagedSkills(names);
                        }
                    }
                    KeyCode::Char('a') if ctrl => state.toggle_all(),
                    KeyCode::Tab if state.has_selection() && !split => {
                        state.viewing = Some(state.selected);
                    }
                    // Page keys scroll the split's right detail pane.
                    KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End if split => {
                        apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::Mcp(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                // Narrow-only drill-in: scroll the tools; Esc/Enter/Tab back out.
                if state.viewing.is_some() && !split {
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
                    // Space toggles too — a dead key in a fuzzy filter, so it never types.
                    KeyCode::Enter | KeyCode::Char(' ') if state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::ToggleMcpServer(state.selected);
                    }
                    KeyCode::Tab if state.has_selection() && !split => {
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
                    // Retry failed servers without the toggle-off/on dance; live
                    // ones are preserved, so this only reconnects the broken set.
                    KeyCode::Char('r') if ctrl => {
                        state.pending_delete = None;
                        return OverlayKeyAction::RetryMcpServers;
                    }
                    KeyCode::Char('t') if ctrl && state.has_selection() => {
                        state.pending_delete = None;
                        return OverlayKeyAction::OpenMcpTools(state.selected);
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
                    // Page keys scroll the split's right detail pane.
                    KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End if split => {
                        apply_detail_scroll(&mut state.detail_scroll, key, ctrl);
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::McpTools(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Esc if !state.query.is_empty() => {
                        state.query.clear();
                        state.refilter();
                    }
                    // Esc returns to `/mcp`, statuses refreshed so the server
                    // row's `· N off` count reflects the toggles just made.
                    KeyCode::Esc => {
                        let parent = std::mem::take(&mut state.parent);
                        self.overlay = Overlay::Mcp(*parent);
                        self.refresh_mcp_overlay_status();
                    }
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    KeyCode::Enter | KeyCode::Char(' ') if state.has_selection() => {
                        return OverlayKeyAction::ToggleMcpTool(state.selected);
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
                        state.query.push(c);
                        state.refilter();
                    }
                    _ => {}
                }
                OverlayKeyAction::Handled
            }
            Overlay::McpPaste(state) => {
                let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                match key.code {
                    KeyCode::Esc if !state.query.is_empty() => {
                        state.query.clear();
                        state.refilter();
                    }
                    // Esc abandons the paste; back to `/mcp` when it was open.
                    KeyCode::Esc => {
                        self.overlay = match state.parent.take() {
                            Some(parent) => Overlay::Mcp(*parent),
                            None => Overlay::None,
                        };
                    }
                    KeyCode::Up => state.select_prev(),
                    KeyCode::Char('p') if ctrl => state.select_prev(),
                    KeyCode::Down => state.select_next(),
                    KeyCode::Char('n') if ctrl => state.select_next(),
                    // Space marks; on an existing name the mark means replace.
                    KeyCode::Char(' ') if state.has_selection() => {
                        if let Some(item) = state.items.get_mut(state.selected) {
                            item.checked = !item.checked;
                        }
                    }
                    KeyCode::Char('a') if ctrl => state.toggle_all(),
                    KeyCode::Enter => {
                        return OverlayKeyAction::ApplyMcpPaste;
                    }
                    KeyCode::Backspace => {
                        state.query.pop();
                        state.refilter();
                    }
                    KeyCode::Char(c) if !ctrl && c != ' ' => {
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
                    // Page keys scroll the split session picker's preview pane.
                    KeyCode::PageUp | KeyCode::PageDown | KeyCode::Home | KeyCode::End
                        if split && matches!(picker.kind, PickerKind::Session) =>
                    {
                        let sid =
                            picker
                                .filtered_items()
                                .get(picker.selected)
                                .and_then(|(_, item)| match &item.value {
                                    PickerValue::Session(preview) => {
                                        Some(preview.session_id.clone())
                                    }
                                    _ => None,
                                });
                        apply_preview_scroll(&mut picker.preview_scroll, key);
                        picker.preview_scroll_for = sid;
                        OverlayKeyAction::Handled
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

    /// Queue-focus mode over the queued-input panel (↑ on an empty composer
    /// enters it). Sits between the overlay and global handlers: pre-empts the
    /// sending-scroll ↑/↓ and Esc-interrupt branches, yields to modal cards.
    /// `None` = fall through, exiting focus first so typing resumes composing.
    fn handle_queue_focus_key(&mut self, key: KeyEvent) -> Option<bool> {
        let Some(selected) = self.queue_focus else {
            if matches!(key.code, KeyCode::Up)
                && key.modifiers.is_empty()
                && self.draft.is_empty()
                && self.loading_resume.is_none()
            {
                let rows = self.queued_rows();
                if !rows.is_empty() {
                    self.queue_focus = Some(rows.len() - 1);
                    return Some(false);
                }
            }
            return None;
        };

        let rows = self.queued_rows();
        if rows.is_empty() {
            self.queue_focus = None;
            return None;
        }
        let selected = selected.min(rows.len() - 1);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let reorder = key
            .modifiers
            .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT)
            && !ctrl;
        match key.code {
            KeyCode::Up if reorder => {
                if self.queue_row_move(&rows[selected], -1) {
                    self.queue_focus = Some(selected.saturating_sub(1));
                }
            }
            KeyCode::Down if reorder => {
                if self.queue_row_move(&rows[selected], 1) {
                    self.queue_focus = Some(selected + 1);
                }
            }
            KeyCode::Up if key.modifiers.is_empty() => {
                self.queue_focus = Some(selected.saturating_sub(1));
            }
            KeyCode::Char('p') if ctrl => {
                self.queue_focus = Some(selected.saturating_sub(1));
            }
            KeyCode::Down if key.modifiers.is_empty() => {
                if selected + 1 < rows.len() {
                    self.queue_focus = Some(selected + 1);
                } else {
                    self.queue_focus = None;
                }
            }
            KeyCode::Char('n') if ctrl => {
                if selected + 1 < rows.len() {
                    self.queue_focus = Some(selected + 1);
                } else {
                    self.queue_focus = None;
                }
            }
            KeyCode::Enter => match self.queue_row_recall(&rows[selected]) {
                Some(text) => {
                    self.queue_focus = None;
                    self.draft = text;
                    self.cursor = self.draft.len();
                    self.sync_command_menu_state();
                }
                None => {
                    self.notice = Some((MUTED(), QUEUE_ROW_GONE_NOTICE.to_string()));
                }
            },
            KeyCode::Backspace | KeyCode::Delete => {
                if !self.queue_row_remove(&rows[selected]) {
                    self.notice = Some((MUTED(), QUEUE_ROW_GONE_NOTICE.to_string()));
                }
                match self.queued_rows().len() {
                    0 => self.queue_focus = None,
                    n => self.queue_focus = Some(selected.min(n - 1)),
                }
            }
            KeyCode::Char('d') if ctrl => {
                if !self.queue_row_remove(&rows[selected]) {
                    self.notice = Some((MUTED(), QUEUE_ROW_GONE_NOTICE.to_string()));
                }
                match self.queued_rows().len() {
                    0 => self.queue_focus = None,
                    n => self.queue_focus = Some(selected.min(n - 1)),
                }
            }
            KeyCode::Esc => {
                self.queue_focus = None;
            }
            _ => {
                self.queue_focus = None;
                return None;
            }
        }
        Some(false)
    }

    async fn handle_global_key(&mut self, key: KeyEvent) -> Result<Option<bool>> {
        if is_help_shortcut(key) {
            self.open_help_overlay();
            return Ok(Some(false));
        }

        // Shift+Tab cycles the agent mode (normal → auto-approve → plan) —
        // aligned with Claude Code's Shift+Tab permission-mode cycle.
        if is_auto_approve_toggle(key) {
            self.cycle_agent_mode().await;
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
                // In /goal mode one Esc only arms a confirm so a stray Esc can't
                // tear down the loop; a second consecutive Esc interrupts + stops it.
                if self.goal_mode.is_some() && !self.goal_stop_confirm_pending {
                    self.goal_stop_confirm_pending = true;
                    self.notice = Some((WARNING(), GOAL_STOP_CONFIRM_NOTICE.to_string()));
                } else {
                    self.goal_stop_confirm_pending = false;
                    self.interrupt_inflight_request().await?;
                }
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
            // Keyboard path to the `▸ +N lines` fold toggle.
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if !self.toggle_latest_output() {
                    self.notice = Some((MUTED(), "no collapsed output to expand".to_string()));
                }
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
            // While a turn streams, bare ↑/↓ scroll the transcript (swipe-scroll
            // on mobile terminals) — unless the `/` menu is open, which owns them.
            KeyCode::Up if self.sending && self.visible_command_menu().is_none() => {
                self.scroll_up_lines(3);
                true
            }
            KeyCode::Down if self.sending && self.visible_command_menu().is_none() => {
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
            // No Ctrl+M binding — terminals send CR for it, i.e. Enter without the kitty protocol.
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
                    self.notice = Some((ERROR(), err.to_string()));
                }
            }
            KeyCode::Char('l') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.reset_composer();
                // Readline muscle memory: Ctrl+L also redraws the screen.
                self.pending_full_repaint = true;
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
            KeyCode::Char('x') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.pending_ctrl_x = true;
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
            // Bare Up/Down move the caret in a multi-line draft, else scroll or
            // recall history at the edge (see composer_up_key). Ctrl+P/N: history.
            KeyCode::Up => self.composer_up_key(),
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.composer_up_or_history_prev()
            }
            KeyCode::Down => self.composer_down_key(),
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

/// `/resume` preview scroll (lines up from the bottom): PageUp older, PageDown
/// newer, Home oldest (`u16::MAX`, renderer-clamped), End latest.
fn apply_preview_scroll(scroll_up: &mut u16, key: KeyEvent) {
    *scroll_up = match key.code {
        KeyCode::PageUp => scroll_up.saturating_add(DETAIL_PAGE_LINES),
        KeyCode::PageDown => scroll_up.saturating_sub(DETAIL_PAGE_LINES),
        KeyCode::Home => u16::MAX,
        KeyCode::End => 0,
        _ => *scroll_up,
    };
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
