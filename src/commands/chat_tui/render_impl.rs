use super::*;

/// The text width a markdown table is laid out to fit, given the full transcript
/// column width: drop the accent gutter and reserve one column for a possible
/// scrollbar. Reserving the scrollbar unconditionally keeps the table the same
/// width whether or not it actually appears, so a resize can't shear a table that
/// was sized when no scrollbar was showing (and vice-versa).
fn table_layout_width(area_width: u16) -> u16 {
    area_width.saturating_sub(ACCENT_GUTTER_WIDTH + 1)
}

/// The inner content rect of a centered overlay — mirrors the `Margin` every
/// overlay insets its body by, so the screen selection can be confined to it.
fn overlay_content_rect(area: Rect) -> Rect {
    area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 2,
    })
}

/// Draw a `↓ Jump to bottom` pill — dark text on a light chip — centered on the
/// bottom row of `area`. Returns its rect, or `None` when even the short form won't
/// fit; falls back to a compact label on a narrow transcript.
fn render_jump_to_bottom(frame: &mut Frame<'_>, area: Rect) -> Option<Rect> {
    if area.height == 0 {
        return None;
    }
    // A light chip with dark text reads cleanly against the dark transcript.
    let style = Style::default()
        .fg(Color::Rgb(30, 33, 35))
        .bg(Color::Rgb(206, 210, 213));
    let label = [" ↓ Jump to bottom ", " ↓ bottom "]
        .into_iter()
        .find(|l| l.chars().count() as u16 + 2 <= area.width)?;
    let width = label.chars().count() as u16;
    let rect = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height - 1,
        width,
        height: 1,
    };
    frame.render_widget(Paragraph::new(Span::styled(label, style)), rect);
    Some(rect)
}

impl ChatTuiApp {
    pub(super) fn is_transcript_empty(&self) -> bool {
        self.history.is_empty()
            && self.pending_response.is_empty()
            && self.incoming_buffer.is_empty()
            && self.pending_reasoning.is_empty()
            && !self.sending
            && self.local_command.is_none()
    }

    /// The full transcript including the live spinner status line. The body is
    /// memoized across frames (see [`build_transcript_body`]); the spinner is
    /// volatile so it is appended fresh here. Used directly by tests and as the
    /// single source of truth for the cached render path.
    pub(super) fn build_transcript(&self) -> RenderedTranscript {
        let body = self.build_transcript_body();
        let mut lines = body.lines;
        let mut bar_colors = body.bar_colors;
        self.append_spinner_status(&mut lines, &mut bar_colors);
        RenderedTranscript::new(lines, bar_colors)
    }

    /// The transcript body: intro, history, the streamed reply, and any notice —
    /// everything except the per-frame spinner status line. Composed from the
    /// memoized history prefix plus the volatile tail; the result is byte-for-byte
    /// what the single-pass build produced, so tests and `max_scroll` are
    /// unaffected. The render path uses the two pieces separately (cached prefix +
    /// fresh tail) so a growing stream never re-renders the whole history.
    pub(super) fn build_transcript_body(&self) -> RenderedTranscript {
        // `transcript_width` is the last-rendered text-area width (already gutter-
        // and scrollbar-adjusted) — the width tables should fit into. It equals the
        // render path's reserved width whenever a scrollbar is present, which is the
        // only case `max_scroll` (the consumer of this path) actually cares about.
        let text_width = self.transcript_width;
        let body = self.build_transcript_history_body(text_width);
        let (tail_lines, tail_bars) = self.volatile_tail_blocks(text_width);
        if tail_lines.is_empty() {
            return body;
        }
        let mut lines = body.lines;
        let mut bars = body.bar_colors;
        // The prefix is already compacted (no trailing blank) and the tail leads
        // with exactly one spacing blank, so the concatenation is canonical —
        // identical to the old single-pass `compact` over the whole body.
        lines.extend(tail_lines);
        bars.extend(tail_bars);
        RenderedTranscript::new(lines, bars)
    }

    /// The memoizable transcript prefix: intro + committed history, with no
    /// dependence on the live stream or notice. This is the expensive part
    /// (markdown parsing, tool decoding) and what
    /// [`ensure_transcript_cache`](Self::ensure_transcript_cache) caches — so it
    /// is rebuilt at most once per *history* change, never per streamed token.
    pub(super) fn build_transcript_history_body(&self, text_width: u16) -> RenderedTranscript {
        let mut lines = Vec::new();
        // Bar color per logical line, kept in lockstep with `lines`. Chrome
        // (intro, spacing) is `None`; each message block paints its role color.
        let mut bars: Vec<Option<Color>> = Vec::new();
        let mut previous_role: Option<&str> = None;

        if self.is_transcript_empty() {
            push_styled_line(&mut lines, "", Style::default());
            bars.push(None);
            return RenderedTranscript::new(lines, bars);
        }

        push_transcript_intro(&mut lines);
        push_message_spacing(&mut lines);
        bars.resize(lines.len(), None);

        // The agent's working dir — tool paths render relative to it (the footer
        // already shows the cwd, so absolute paths are just noise that hides the
        // basename). Falls back to the chat sandbox when the real dir is unknown.
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.as_str()
        } else {
            self.real_cwd.as_str()
        };

        let mut idx = 0;
        while idx < self.history.len() {
            let message = &self.history[idx];
            // The plan/task list is pinned in its own panel above the composer
            // (see `render_plan_panel`), not rendered inline — so it stays visible
            // instead of scrolling away under later tool calls. Skip it here
            // without touching `previous_role`, so spacing reads as if it weren't
            // present.
            if message.role == "plan" {
                idx += 1;
                continue;
            }
            if should_add_message_spacing(previous_role, message.role.as_str()) {
                push_message_spacing(&mut lines);
                bars.resize(lines.len(), None);
            }
            let mut block = Vec::new();
            let mut advance = 1;
            match message.role.as_str() {
                "user" => {
                    // A skill invocation stores the full expanded SKILL.md as the
                    // user message (the model needs it), but the transcript should
                    // show the compact `/name args` the user actually typed.
                    let label = super::skill_invocation_label(&message.content);
                    let shown = label.as_deref().unwrap_or(&message.content);
                    render_user_message(&mut block, shown, &message.attachments);
                }
                "assistant" => {
                    let reasoning = self
                        .thinking_enabled
                        .then_some(message.reasoning_content.as_deref())
                        .flatten();
                    // Folds to the thinking summary unless this turn's index is in
                    // `expanded_thinking`. Separate barred blocks give the thinking
                    // gutter its own color; `block` stays empty so the generic push
                    // below no-ops.
                    let collapsed = !self.expanded_thinking.contains(&idx);
                    let duration_ms = self.reasoning_durations.get(&idx).copied();
                    push_assistant_blocks(
                        &mut lines,
                        &mut bars,
                        reasoning.map(|text| ReasoningView {
                            text,
                            collapsed,
                            duration_ms,
                        }),
                        &message.content,
                        text_width,
                        role_bar_color("assistant"),
                    );
                }
                "tool_call" => {
                    let (name, args) = decode_tool_call(&message.content);
                    // Coalesce a run of consecutive same-kind calls into one line.
                    // Cursor agents emit no interleaved results, so their calls are
                    // adjacent; the in-process agent's calls are split by results,
                    // so this never merges them. Subagents are the exception: each is
                    // a heavyweight, distinct unit of work (often dispatched in
                    // parallel, so adjacent), not the tiny exploration steps
                    // coalescing is meant to fold — render each on its own line so
                    // its task is visible instead of an opaque `subagent ×N`.
                    let run = if name == "subagent" {
                        1
                    } else {
                        self.tool_call_run_len(idx, &name)
                    };
                    if run >= 2 {
                        let targets: Vec<String> = self.history[idx..idx + run]
                            .iter()
                            .map(|m| {
                                let (n, a) = decode_tool_call(&m.content);
                                let target = tool_call_target(&n, &a);
                                // cursor gives no path/pattern, so show the per-call
                                // result (e.g. `18 matches`) in the detail slot.
                                if target.is_empty() {
                                    decode_tool_outcome(&m.content).0.unwrap_or_default()
                                } else {
                                    target
                                }
                            })
                            .collect();
                        let failed = self.history[idx..idx + run]
                            .iter()
                            .filter(|m| decode_tool_outcome(&m.content).1)
                            .count();
                        render_tool_call_group(&mut block, &name, &targets, failed);
                        advance = run;
                    } else {
                        let (result, failed) = decode_tool_outcome(&message.content);
                        render_tool_call(&mut block, &name, &args, result.as_deref(), failed, cwd);
                    }
                }
                "tool_result" => {
                    // The matching call is the immediately preceding entry (the
                    // in-process agent emits call then result) — its tool name lets
                    // the count read in the right unit (files/entries/matches).
                    let tool = idx
                        .checked_sub(1)
                        .and_then(|i| self.history.get(i))
                        .filter(|m| m.role == "tool_call")
                        .map(|m| decode_tool_call(&m.content).0);
                    render_tool_result(&mut block, &message.content, cwd, tool.as_deref());
                }
                "local_command" => {
                    // Expanded renders the in-memory output (persisted preview after a
                    // resume) in place; folded shows the preview + clickable expander.
                    let view = if self.expanded_output.contains(&idx) {
                        OutputView::Expanded {
                            full: self.local_outputs.get(&idx),
                        }
                    } else {
                        OutputView::Collapsed
                    };
                    render_local_command(&mut block, &message.content, view);
                }
                "plan" => render_plan(&mut block, &message.content),
                other => render_system_message(&mut block, other, &message.content, text_width),
            }
            let bar = role_bar_color(message.role.as_str());
            push_block(&mut lines, &mut bars, block, Some(bar));
            previous_role = Some(message.role.as_str());
            idx += advance;
        }

        compact_lines_and_bars(&mut lines, &mut bars);
        RenderedTranscript::new(lines, bars)
    }

    /// The volatile blocks that follow the committed history — the live streamed
    /// reply and any notice — each with its leading spacing blank. Kept OUT of the
    /// memoized body so a growing stream doesn't re-render and re-wrap the whole
    /// history every frame; composed fresh per frame, like the spinner. Empty in
    /// the empty-state (`build_transcript_history_body` shows neither then).
    ///
    /// The leading blank is unconditional because both blocks always follow
    /// preceding content here: a streamed reply follows the history/intro (and
    /// `should_add_message_spacing(_, "assistant")` is always true), and a notice
    /// always gets its separator — matching the old single-pass spacing exactly.
    pub(super) fn volatile_tail_blocks(
        &self,
        text_width: u16,
    ) -> (Vec<StyledLine>, Vec<Option<Color>>) {
        let mut lines: Vec<StyledLine> = Vec::new();
        let mut bars: Vec<Option<Color>> = Vec::new();
        if self.is_transcript_empty() {
            return (lines, bars);
        }
        // Only render the live tail once the answer has started: during the
        // thinking-only phase the spinner already shows "thinking (Xs ...)", and a
        // folded `▸ thought` line here would double it. The summary is frozen at
        // the answer's start so it matches the committed form (no jump on commit).
        if !self.pending_response.is_empty() {
            let live_reasoning = (self.thinking_enabled && !self.pending_reasoning.is_empty())
                .then_some(self.pending_reasoning.as_str());
            lines.push(blank_line());
            bars.push(None);
            push_assistant_blocks(
                &mut lines,
                &mut bars,
                live_reasoning.map(|text| ReasoningView {
                    text,
                    collapsed: true,
                    duration_ms: self.segment_reasoning_ms(),
                }),
                &self.pending_response,
                text_width,
                ACCENT,
            );
        }
        // A running `!cmd` streams its output here (not into history) so the
        // memoized history body stays put while lines arrive; it's committed to
        // history once it finishes.
        if let Some(run) = &self.local_command {
            lines.push(blank_line());
            bars.push(None);
            let mut block = Vec::new();
            // Serialize only a bounded preview (+ the true total) rather than the
            // whole streamed buffer: a long-running command can accumulate megabytes,
            // and this runs every frame. Only the first MAX_OUTPUT_LINES ever show.
            let total = run.stdout.lines().count() + run.stderr.lines().count();
            let content = serde_json::json!({
                "command": run.command,
                "stdout": first_lines(&run.stdout, MAX_PERSISTED_OUTPUT_LINES),
                "stderr": first_lines(&run.stderr, MAX_PERSISTED_OUTPUT_LINES),
                "total_lines": total,
                "running": true,
            })
            .to_string();
            render_local_command(&mut block, &content, OutputView::Live);
            push_block(&mut lines, &mut bars, block, Some(TOOL));
        }
        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            lines.push(blank_line());
            bars.push(None);
            let mut block = Vec::new();
            render_notice_line(&mut block, color, text.as_ref());
            push_block(&mut lines, &mut bars, block, Some(color));
        }
        (lines, bars)
    }

    /// Length of the run of consecutive `tool_call` entries starting at `start`
    /// that share the tool name `name` (≥1; used to coalesce cursor's tool runs).
    fn tool_call_run_len(&self, start: usize, name: &str) -> usize {
        self.history[start..]
            .iter()
            .take_while(|m| m.role == "tool_call" && decode_tool_call(&m.content).0 == name)
            .count()
    }

    /// The live processing status line — thinking / running a tool / working —
    /// shown while a turn runs, with the spinner glyph + elapsed clock. Returns
    /// `None` when idle. Rebuilt every frame (it animates) and appended after the
    /// cached body so animation never invalidates the cache.
    pub(super) fn spinner_status_line(&self) -> Option<StyledLine> {
        // A model turn and a `!cmd` run never overlap, so at most one drives the
        // spinner; the local command reports its own elapsed clock and activity.
        let (started_at, activity) = if self.sending {
            (self.request_started_at, self.processing_activity())
        } else if let Some(run) = &self.local_command {
            (Some(run.started_at), "running command".to_string())
        } else {
            return None;
        };
        let mut block = Vec::new();
        render_pending_status(
            &mut block,
            self.frame_tick,
            self.reduce_motion,
            started_at
                .map(|started_at| started_at.elapsed())
                .unwrap_or_default(),
            &activity,
        );
        block.into_iter().next()
    }

    /// Appends the live spinner status line (with its leading spacing blank) to a
    /// freshly built body. The body is already compacted (no trailing blank), so
    /// one blank + the spinner keeps "what's happening" pinned to the bottom of
    /// the transcript without a double gap.
    fn append_spinner_status(&self, lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
        let Some(spinner) = self.spinner_status_line() else {
            return;
        };
        if !lines.is_empty() {
            lines.push(blank_line());
            bars.push(None);
        }
        lines.push(spinner);
        bars.push(Some(ACCENT));
    }

    /// A cheap O(1) fingerprint of everything the cached *history body* depends
    /// on: intro + committed history (length + the immutable first/last entries).
    /// History entries are *mostly* append-only, so length plus the endpoints
    /// identifies the body without hashing all of it every frame; the one
    /// exception — in-place enrichment of a cursor tool-call entry — bumps
    /// `transcript_revision`, which is mixed in here so those edits still
    /// invalidate. The streamed reply, notice, and spinner are deliberately
    /// EXCLUDED: they live in the volatile tail (composed fresh each frame), so a
    /// growing stream must not bust this cache and force a full-history re-render.
    /// `is_transcript_empty` (which reads the stream/sending) flips at most once
    /// per turn — when streaming starts/ends — so it stays stable mid-stream.
    fn transcript_body_fp(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.is_transcript_empty().hash(&mut hasher);
        self.transcript_revision.hash(&mut hasher);
        // The committed history renders the folded reasoning summary only while
        // this is on, so a toggle must invalidate the memoized body.
        self.thinking_enabled.hash(&mut hasher);
        self.history.len().hash(&mut hasher);
        if let Some(first) = self.history.first() {
            first.role.hash(&mut hasher);
            first.content.len().hash(&mut hasher);
            first.attachments.len().hash(&mut hasher);
        }
        if let Some(last) = self.history.last() {
            last.role.hash(&mut hasher);
            last.content.len().hash(&mut hasher);
            last.attachments.len().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Rebuilds the cached transcript history body (and its char-wrap height
    /// estimate) only when the history fingerprint or the terminal width changed.
    /// The expensive markdown render and tool decoding happen here, at most once
    /// per *history* change — not on every animation frame, keystroke, or streamed
    /// token (the live reply and notice are the volatile tail, composed outside).
    fn ensure_transcript_cache(&mut self, area_width: u16) {
        let fp = self.transcript_body_fp();
        let fresh = self
            .transcript_cache
            .as_ref()
            .is_some_and(|cache| cache.fp == fp && cache.area_width == area_width);
        if fresh {
            return;
        }
        let body = self.build_transcript_history_body(table_layout_width(area_width));
        let plain_width = area_width.saturating_sub(ACCENT_GUTTER_WIDTH).max(1);
        let plain_prepass = wrap_plain_lines(&body.plain_lines, plain_width).len();
        self.transcript_cache = Some(TranscriptCache {
            fp,
            area_width,
            body,
            plain_prepass,
            styled_width: 0,
            wrapped: None,
        });
    }

    /// Word-wraps the cached body to `text_width`, reusing the previous wrap when
    /// the width is unchanged. Must run after [`ensure_transcript_cache`].
    fn ensure_transcript_wrap(&mut self, text_width: u16) {
        let cache = self
            .transcript_cache
            .as_mut()
            .expect("ensure_transcript_cache runs before ensure_transcript_wrap");
        if cache.wrapped.is_some() && cache.styled_width == text_width {
            return;
        }
        let wrapped = wrap_transcript(&cache.body.lines, &cache.body.bar_colors, text_width);
        cache.styled_width = text_width;
        cache.wrapped = Some(wrapped);
    }

    /// Cheap O(1) fingerprint of the volatile tail's inputs — the streamed reply,
    /// a running `!cmd`'s buffered output, and any notice. All grow append-only
    /// (or are short), so byte lengths + the notice identify the rendered tail
    /// without hashing it. The spinner is deliberately EXCLUDED (it animates every
    /// frame and is wrapped fresh at compose time), so a pure animation tick — the
    /// 60fps redraw that drove the O(reply²) re-parse — hits the cache.
    pub(super) fn volatile_tail_fp(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.is_transcript_empty().hash(&mut hasher);
        self.pending_response.len().hash(&mut hasher);
        // `pending_reasoning.len()` keys the live thinking summary's line-count
        // fallback; `thinking_enabled` gates whether it renders, so a /config
        // toggle mid-turn must invalidate too.
        self.pending_reasoning.len().hash(&mut hasher);
        self.thinking_enabled.hash(&mut hasher);
        match &self.local_command {
            Some(run) => {
                run.command.hash(&mut hasher);
                run.stdout.len().hash(&mut hasher);
                run.stderr.len().hash(&mut hasher);
            }
            None => 0usize.hash(&mut hasher),
        }
        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            text.as_ref().hash(&mut hasher);
            format!("{color:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Renders the volatile tail (markdown → styled lines) only when its
    /// fingerprint or the render width changed — not on every animation frame or
    /// every streamed token of an unchanged reply. The expensive markdown parse +
    /// table layout (the O(reply) cost the spinner redraw used to repeat every
    /// frame) happen here, at most once per *content* change.
    fn ensure_volatile_tail(&mut self, render_width: u16) {
        let fp = self.volatile_tail_fp();
        let fresh = self
            .volatile_tail_cache
            .as_ref()
            .is_some_and(|cache| cache.fp == fp && cache.render_width == render_width);
        if fresh {
            return;
        }
        let (lines, bars) = self.volatile_tail_blocks(render_width);
        self.volatile_tail_cache = Some(VolatileTailCache {
            fp,
            render_width,
            lines,
            bars,
            plain_width: 0,
            prepass: 0,
            styled_width: 0,
            wrapped: None,
        });
    }

    /// Char-wrapped row count of the cached tail at `plain_width`, for the pane-
    /// height prepass — memoized on the width so a streamed reply isn't re-counted
    /// every frame. Must run after [`ensure_volatile_tail`].
    fn volatile_tail_prepass(&mut self, plain_width: u16) -> usize {
        let cache = self
            .volatile_tail_cache
            .as_mut()
            .expect("ensure_volatile_tail runs before volatile_tail_prepass");
        if cache.lines.is_empty() {
            return 0;
        }
        if cache.plain_width == plain_width {
            return cache.prepass;
        }
        let plain: Vec<String> = cache.lines.iter().map(|l| l.plain.clone()).collect();
        let rows = wrap_plain_lines(&plain, plain_width).len();
        cache.plain_width = plain_width;
        cache.prepass = rows;
        rows
    }

    /// Word-wraps the cached tail to `text_width`, reusing the previous wrap when
    /// the width is unchanged. Leaves `wrapped = None` when the tail is empty (so
    /// it contributes no rows). Must run after [`ensure_volatile_tail`].
    fn ensure_volatile_tail_wrap(&mut self, text_width: u16) {
        let cache = self
            .volatile_tail_cache
            .as_mut()
            .expect("ensure_volatile_tail runs before ensure_volatile_tail_wrap");
        if cache.lines.is_empty() {
            cache.wrapped = None;
            cache.styled_width = text_width;
            return;
        }
        if cache.wrapped.is_some() && cache.styled_width == text_width {
            return;
        }
        cache.wrapped = Some(wrap_transcript(&cache.lines, &cache.bars, text_width));
        cache.styled_width = text_width;
    }

    /// Composes the final wrapped transcript for this frame: the cached history
    /// body wrap, the cached volatile-tail wrap (streamed reply + running `!cmd` +
    /// notice), and the freshly wrapped spinner. Only the spinner — two lines that
    /// animate — is wrapped per frame; the history and tail wraps are reused from
    /// their caches, so a growing stream costs O(tail-delta), not O(history+reply)
    /// every frame. The rest are shallow clones.
    fn composed_transcript_rows(
        &self,
        spinner: Option<&StyledLine>,
        text_width: u16,
    ) -> (Text<'static>, Vec<String>, Vec<Option<Color>>) {
        let wrapped = self
            .transcript_cache
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
            .expect("ensure_transcript_wrap runs before composed_transcript_rows");
        let mut text_lines: Vec<Line<'static>> = wrapped.text.lines.clone();
        let mut rows = wrapped.rows.clone();
        let mut bars = wrapped.bars.clone();

        // The volatile tail already carries its own leading spacing blanks (see
        // `volatile_tail_blocks`); reuse its cached wrap instead of re-wrapping the
        // streamed reply every frame. `wrapped` is `None` exactly when the tail is
        // empty, so this adds no spurious rows.
        if let Some(tail) = self
            .volatile_tail_cache
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
        {
            text_lines.extend(tail.text.lines.iter().cloned());
            rows.extend(tail.rows.iter().cloned());
            bars.extend(tail.bars.iter().cloned());
        }

        // The spinner mirrors `append_spinner_status` — a leading blank (whenever
        // anything precedes it, which it always does) then the status line. Wrapped
        // fresh here, never cached, since it animates every frame.
        if let Some(spinner) = spinner {
            let mut tail: Vec<StyledLine> = Vec::new();
            let mut tail_bars: Vec<Option<Color>> = Vec::new();
            if !rows.is_empty() {
                tail.push(blank_line());
                tail_bars.push(None);
            }
            tail.push(spinner.clone());
            tail_bars.push(Some(ACCENT));
            let wrapped_tail = wrap_transcript(&tail, &tail_bars, text_width);
            text_lines.extend(wrapped_tail.text.lines);
            rows.extend(wrapped_tail.rows);
            bars.extend(wrapped_tail.bars);
        }

        (Text::from(text_lines), rows, bars)
    }

    pub(super) fn transcript_intro_lines(&self) -> Vec<String> {
        // Plain-text mirror of the empty-state banner (wordmark + tagline), used
        // to reserve its height. Must stay in lockstep with `render_empty_state`.
        // Model / base_url / cwd live in the footer (the persistent status bar).
        let mut lines: Vec<String> = BRAND_WORDMARK.iter().map(|row| row.to_string()).collect();
        lines.push(BRAND_TAGLINE.to_string());
        lines
    }

    pub(super) fn render(&mut self, frame: &mut Frame<'_>) {
        self.tick_selection_flash();
        self.refresh_git_branch();
        let outer = frame.area();
        self.picker_hitbox = None;
        self.transcript_hitbox = None;
        self.screen_region = None;
        let composer_area = self.render_main(frame, outer);
        if let Some(menu) = self.visible_command_menu() {
            let (area, placement) = command_menu_area(
                composer_area,
                outer,
                menu.entries.len(),
                self.command_menu.placement,
            );
            self.command_menu.placement = Some(placement);
            self.render_command_menu(frame, area, &menu);
        }
        let body = outer;

        match self.overlay.clone() {
            Overlay::Picker(picker) => {
                self.render_picker(frame, centered_rect(68, 72, body), &picker);
            }
            Overlay::Help { scroll } => {
                let area = centered_rect(72, 88, body);
                self.screen_region = Some(overlay_content_rect(area));
                let clamped = self.render_help_overlay(frame, area, scroll);
                if let Overlay::Help { scroll } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Skills(skills) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                let clamped = self.render_skills_overlay(frame, area, &skills);
                if let (Some(c), Overlay::Skills(s)) = (clamped, &mut self.overlay) {
                    s.detail_scroll = c;
                }
            }
            Overlay::Mcp(mcp) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                let clamped = self.render_mcp_overlay(frame, area, &mcp);
                if let (Some(c), Overlay::Mcp(s)) = (clamped, &mut self.overlay) {
                    s.detail_scroll = c;
                }
            }
            Overlay::Config(config) => {
                let area = centered_rect(64, 80, body);
                self.screen_region = Some(overlay_content_rect(area));
                self.render_config_overlay(frame, area, &config);
            }
            Overlay::None => {}
        }

        if self.pending_mcp_consent.is_some() {
            self.render_mcp_consent_card(frame, composer_area, outer);
        } else if self.agent_permission.is_some() {
            self.render_permission_card(frame, composer_area, outer);
        }

        // Snapshot the finished screen so a drag can copy from anywhere on it,
        // then wash the full-screen selection over whatever now sits there.
        self.capture_screen_surface(frame);
        self.render_screen_selection_highlight(frame);

        self.render_toast(frame, outer);
    }

    /// Captures the rendered screen into `screen_surface` for full-screen drag-copy.
    /// Confined to `screen_region` (a modal's content rect) when one is open.
    fn capture_screen_surface(&mut self, frame: &mut Frame<'_>) {
        let full = frame.area();
        let area = self
            .screen_region
            .map(|region| region.intersection(full))
            .unwrap_or(full);
        let buffer = frame.buffer_mut();
        let mut rows = Vec::with_capacity(usize::from(area.height));
        for y in area.y..area.y.saturating_add(area.height) {
            let mut row = String::with_capacity(usize::from(area.width));
            for x in area.x..area.x.saturating_add(area.width) {
                if let Some(cell) = buffer.cell((x, y)) {
                    // Wide-char spacers carry an empty symbol, so the row's display
                    // width stays equal to its column count.
                    row.push_str(cell.symbol());
                }
            }
            rows.push(row);
        }
        self.screen_surface = Some(ScreenSurface { area, rows });
    }

    /// Screen-coordinate twin of `render_transcript_selection_highlight`.
    fn render_screen_selection_highlight(&self, frame: &mut Frame<'_>) {
        let Some(selection) = self
            .screen_selection
            .filter(|selection| !selection.is_empty())
        else {
            return;
        };
        let Some(surface) = &self.screen_surface else {
            return;
        };

        let wash = if self.selection_flash_until.is_some() {
            SELECT_FLASH
        } else {
            SELECT_WARM
        };
        let (start, end) = normalized_selection(selection);
        let area = surface.area;
        let row_end = end.row.min(surface.rows.len().saturating_sub(1));
        if start.row > row_end {
            return;
        }

        let buffer = frame.buffer_mut();
        for row in start.row..=row_end {
            // Clamp to the row's real text (trailing blanks excluded), matching copy.
            let text_width = surface
                .rows
                .get(row)
                .map(|line| row_display_width(line.trim_end()))
                .unwrap_or(0);
            let start_col = if row == start.row { start.column } else { 0 };
            let end_col = if row == end.row {
                end.column
            } else {
                text_width
            };
            let start_col = start_col.min(text_width);
            let end_col = end_col.min(text_width);
            if start_col >= end_col {
                continue;
            }
            let y = area.y.saturating_add(row as u16);
            for column in start_col..end_col {
                if let Some(cell) = buffer.cell_mut((area.x + column, y)) {
                    cell.set_bg(wash);
                }
            }
        }
    }

    /// Card asking whether to spawn a repo's project `.mcp.json` stdio servers
    /// (the local code-execution surface). Lists each server's exact command so
    /// the risk is visible, then color-coded y/a/n keys. Anchored above the
    /// composer like the permission card.
    fn render_mcp_consent_card(
        &self,
        frame: &mut Frame<'_>,
        composer_area: Rect,
        frame_area: Rect,
    ) {
        let Some(prompt) = self.pending_mcp_consent.as_ref() else {
            return;
        };
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);
        // Fixed chrome: 2 borders + heading + note + blank-before-keys + keys = 6.
        // Whatever rows remain list the servers (trimmed if the screen is short).
        let chrome = 6usize;
        let list_budget = usize::from(max_total).saturating_sub(chrome);

        let n = prompt.servers.len();
        let heading = format!(
            "Run {n} MCP server{} from this repo's .mcp.json?",
            if n == 1 { "" } else { "s" }
        );
        let note = "These commands run locally on your machine.";
        let keys = mcp_consent_keys_line();

        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        let mut content_w = display_width(&heading).max(display_width(note)).max(keys_w);
        for (name, cmd) in &prompt.servers {
            content_w = content_w.max(display_width(&format!("{name}  {cmd}")));
        }
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))];
        let shown = n.min(list_budget);
        for (name, cmd) in prompt.servers.iter().take(shown) {
            let room = inner_width.saturating_sub(display_width(name) + 2).max(1);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{name}  "),
                    Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_for_display_width(cmd, room),
                    Style::default().fg(TEXT),
                ),
            ]));
        }
        if shown < n {
            lines.push(Line::from(Span::styled(
                format!("…and {} more", n - shown),
                Style::default().fg(MUTED),
            )));
        }
        lines.push(Line::from(Span::styled(
            note.to_string(),
            Style::default().fg(MUTED),
        )));
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(WARNING))
            .title(Span::styled(
                " mcp servers ",
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    /// Card asking the user to approve a mutating agent tool. Anchored directly
    /// above the composer (where the eye and cursor already are) rather than
    /// floating mid-screen, so the decision sits right next to the input. Shows
    /// the action, a preview (diff / command / path), and color-coded y/a/n keys.
    fn render_permission_card(&self, frame: &mut Frame<'_>, composer_area: Rect, frame_area: Rect) {
        let Some(pending) = self.agent_permission.as_ref() else {
            return;
        };
        let destructive = pending.tool == "run_bash"
            && pending
                .preview
                .as_deref()
                .map(crate::agent::tools::bash_looks_destructive)
                .unwrap_or(false);

        // The card rests just above the composer's divider line (a narrower card
        // would otherwise leave that full-width rule poking out past its right
        // edge). Cap its height to the rows above so it never runs off the top.
        let anchor = composer_area.y.saturating_sub(1);
        let max_total = anchor.saturating_sub(frame_area.y).max(1);

        // Fixed chrome: 2 borders + heading + blank-before-keys + keys (+ the
        // destructive flag line). Whatever rows remain feed the preview block
        // (its own leading blank + the preview lines), bottom-trimmed first so
        // the keys line is always the last thing the user sees.
        let chrome = 5 + usize::from(destructive);
        let preview_budget = usize::from(max_total).saturating_sub(chrome);
        let preview: Vec<&str> = pending
            .preview
            .as_deref()
            .map(|p| p.lines().take(12).collect())
            .unwrap_or_default();
        let preview_take = if preview.is_empty() {
            0
        } else {
            preview.len().min(preview_budget.saturating_sub(1))
        };

        // Size the card to its widest visible line rather than the whole input
        // row, so a short confirm reads as a compact card; never wider than the
        // composer. +4 = 2 borders + 1 col of padding on each side.
        let heading = permission_heading(&pending.tool);
        let keys = permission_keys_line(&pending.tool, !self.draft.is_empty());
        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        let mut content_w = display_width(&heading).max(keys_w);
        if destructive {
            content_w = content_w.max(display_width("⚠ looks destructive"));
        }
        for &raw in preview.iter().take(preview_take) {
            content_w = content_w.max(display_width(raw));
        }
        let max_width = composer_area.width.min(frame_area.width).max(1);
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ))];
        if preview_take > 0 {
            lines.push(Line::from(""));
            for &raw in preview.iter().take(preview_take) {
                let trimmed = raw.trim_start();
                let style = if trimmed.starts_with("+ ") {
                    Style::default().fg(ASSISTANT)
                } else if trimmed.starts_with("- ") {
                    Style::default().fg(ERROR)
                } else if pending.tool == "run_bash" {
                    Style::default().fg(if destructive { WARNING } else { TEXT })
                } else {
                    Style::default().fg(MUTED)
                };
                lines.push(Line::from(Span::styled(
                    truncate_for_display_width(raw, inner_width),
                    style,
                )));
            }
        }
        if destructive {
            lines.push(Line::from(Span::styled(
                "⚠ looks destructive".to_string(),
                Style::default().fg(WARNING).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        lines.push(keys);

        let height = (lines.len() as u16 + 2).min(max_total);
        let y = anchor.saturating_sub(height).max(frame_area.y);
        let card = Rect {
            x: composer_area.x,
            y,
            width,
            height,
        };
        frame.render_widget(Clear, card);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT))
            .title(Span::styled(
                " permission ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(card).inner(ratatui::layout::Margin {
            vertical: 0,
            horizontal: 1,
        });
        frame.render_widget(block, card);
        frame.render_widget(Paragraph::new(Text::from(lines)), inner);
    }

    pub(super) fn render_main(&mut self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        let composer_height = self.composer_height(area.width);
        // Two rows: the status line + the contextual shortcut hint bar.
        let footer_height = 2u16;
        // The pinned plan/task-list panel sits between the transcript and the
        // composer (a faint top rule + the wrapped checklist), so progress stays
        // visible instead of scrolling away under later tool calls. Sized from the
        // current plan, capped so the transcript keeps a usable minimum.
        let plan_lines = self.plan_panel_lines();
        let plan_panel_height =
            self.plan_panel_height(&plan_lines, area, composer_height, footer_height);
        let max_transcript_height = area
            .height
            .saturating_sub(composer_height + footer_height + plan_panel_height)
            .max(1);
        let is_empty = self.is_transcript_empty();
        // Memoize the heavy history body build + wrap AND the volatile tail
        // (streamed reply + running !cmd + notice); only the per-frame spinner is
        // rebuilt fresh (see `ensure_transcript_cache` / `ensure_volatile_tail`).
        self.ensure_transcript_cache(area.width);
        // Render the volatile tail at most once per content change — its markdown
        // parse + wrap are reused across animation frames of an unchanged reply.
        self.ensure_volatile_tail(table_layout_width(area.width));
        let spinner = self.spinner_status_line();
        let plain_width = area.width.saturating_sub(ACCENT_GUTTER_WIDTH).max(1);
        // The volatile tail's char-wrap height, sized like the body's estimate so
        // the pane grows to fit the streamed reply (which left the cached body).
        let volatile_prepass = self.volatile_tail_prepass(plain_width);
        // The spinner's leading blank + its (short) status line, sized the same
        // way the body's char-wrap height estimate is, so the pane height matches.
        let spinner_prepass = spinner
            .as_ref()
            .map(|line| wrap_plain_lines(&[String::new(), line.plain.clone()], plain_width).len())
            .unwrap_or(0);
        let prepass_rows = self
            .transcript_cache
            .as_ref()
            .map(|cache| cache.plain_prepass)
            .unwrap_or(1)
            + volatile_prepass
            + spinner_prepass;
        let min_transcript_height = self
            .empty_state_height(area.width.max(1))
            .clamp(1, max_transcript_height);
        let transcript_height = if is_empty {
            min_transcript_height
        } else {
            (prepass_rows as u16).clamp(min_transcript_height, max_transcript_height)
        };
        let stack_height = transcript_height
            .saturating_add(plan_panel_height)
            .saturating_add(composer_height)
            .saturating_add(footer_height)
            .min(area.height.max(1));

        let stack = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(stack_height), Constraint::Min(0)])
            .split(area);
        // transcript / [plan panel] / composer / footer — the plan row is present
        // only when there's a plan, so the indices below shift accordingly.
        let mut constraints = vec![Constraint::Length(transcript_height)];
        if plan_panel_height > 0 {
            constraints.push(Constraint::Length(plan_panel_height));
        }
        constraints.push(Constraint::Length(composer_height));
        constraints.push(Constraint::Length(footer_height));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(stack[0]);

        let transcript_area = chunks[0];
        let mut chunk_idx = 1usize;
        let plan_panel_area = if plan_panel_height > 0 {
            let a = chunks[chunk_idx];
            chunk_idx += 1;
            Some(a)
        } else {
            None
        };
        let composer_outer = chunks[chunk_idx];
        let footer_outer = chunks[chunk_idx + 1];
        let transcript_total_lines = prepass_rows;
        // The composer reserves its own blank spacing row above the divider (see
        // the composer layout below), so the transcript fills its whole area here
        // — no extra bottom padding carved from within, in overflow or otherwise.
        let transcript_view_area = transcript_area;
        let view_height = transcript_view_area.height.max(1);
        let needs_scrollbar = transcript_total_lines > usize::from(view_height);
        let transcript_content_area = Rect {
            x: transcript_view_area.x,
            y: transcript_view_area.y,
            width: if needs_scrollbar {
                transcript_view_area.width.saturating_sub(1).max(1)
            } else {
                transcript_view_area.width
            },
            height: transcript_view_area.height,
        };
        // Reserve a left gutter for per-role accent bars. Content wraps and
        // renders into the inset text area; the bars are painted separately so
        // they never bleed into copied/selected text.
        let transcript_text_area = Rect {
            x: transcript_content_area
                .x
                .saturating_add(ACCENT_GUTTER_WIDTH),
            y: transcript_content_area.y,
            width: transcript_content_area
                .width
                .saturating_sub(ACCENT_GUTTER_WIDTH)
                .max(1),
            height: transcript_content_area.height,
        };
        // Word-wrap ourselves to the text width so our row model (scroll, gutter,
        // selection) exactly matches the rendered rows; we render with wrap OFF.
        // The body and volatile-tail wraps are cached; only the spinner is wrapped
        // per frame.
        self.ensure_transcript_wrap(transcript_text_area.width);
        self.ensure_volatile_tail_wrap(transcript_text_area.width);
        let (wrapped_text, visual_rows, visual_bars) =
            self.composed_transcript_rows(spinner.as_ref(), transcript_text_area.width);
        let transcript_total_lines = visual_rows.len();
        self.transcript_width = transcript_text_area.width.max(1);
        self.transcript_view_height = view_height;
        let max_scroll = transcript_total_lines.saturating_sub(usize::from(view_height));
        // Cache the exact value so the scroll handlers don't rebuild the whole
        // transcript per wheel event (see `effective_max_scroll`).
        self.last_max_scroll = Some(max_scroll);
        if self.follow_output {
            self.transcript_scroll = max_scroll;
        } else {
            self.transcript_scroll = self.transcript_scroll.min(max_scroll);
        }
        self.transcript_hitbox = Some(TranscriptHitbox {
            area: transcript_text_area,
            first_row: self.transcript_scroll,
            // Last use of `visual_rows`; move it in rather than re-cloning the
            // whole-transcript row vector on every repaint.
            rows: visual_rows,
        });

        frame.render_widget(Clear, chunks[0]);

        if is_empty {
            // Inset by the accent gutter so the brand banner sits at the same
            // column as the transcript content does once a message arrives — without
            // this the banner jumps 2 cols right when the first message lands.
            self.render_empty_state(frame, transcript_text_area);
            self.jump_to_bottom_hit = None;
        } else {
            // Pre-wrapped above → render with wrap OFF so rendered rows match.
            // ratatui's `Paragraph` does NOT virtualize: `.scroll((y, 0))` still
            // runs EVERY line through its reflow/LineComposer and writes cells
            // before discarding the scrolled-past rows, so a full-transcript
            // Paragraph costs O(total rows) per frame. While a turn streams, the
            // spinner forces a ~60fps repaint, and on the single-thread runtime
            // that O(n) draw starves the streaming task — a long session makes a
            // subagent crawl. Slice to the visible window and render at scroll 0
            // → O(visible rows). The full row model (gutter, selection, scrollbar,
            // hitbox) below is unchanged, so geometry stays exact.
            let view_start = self.transcript_scroll.min(wrapped_text.lines.len());
            let view_end = view_start
                .saturating_add(usize::from(transcript_text_area.height))
                .min(wrapped_text.lines.len());
            let visible_text = Text::from(wrapped_text.lines[view_start..view_end].to_vec());
            let transcript_widget = Paragraph::new(visible_text).style(Style::default().fg(TEXT));
            frame.render_widget(transcript_widget, transcript_text_area);
            self.paint_accent_gutter(
                frame,
                transcript_content_area.x,
                transcript_text_area.y,
                transcript_text_area.height,
                &visual_bars,
            );
            self.render_transcript_selection_highlight(frame, transcript_text_area);
            let total_lines = transcript_total_lines;
            if total_lines > usize::from(view_height) {
                let mut scrollbar_state =
                    ScrollbarState::new(total_lines.saturating_sub(usize::from(view_height)))
                        .position(self.transcript_scroll);
                let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                    .thumb_style(Style::default().fg(FAINT))
                    .track_style(Style::default().fg(Color::Rgb(50, 54, 56)))
                    .begin_symbol(None)
                    .end_symbol(None);
                frame.render_stateful_widget(scrollbar, transcript_view_area, &mut scrollbar_state);
            }
            // Clickable jump-to-bottom pill (like Ctrl+End), only while scrolled up.
            self.jump_to_bottom_hit = if self.transcript_scroll < max_scroll {
                render_jump_to_bottom(frame, transcript_view_area)
            } else {
                None
            };
        }

        if let Some(plan_panel_area) = plan_panel_area {
            self.render_plan_panel(frame, plan_panel_area, &plan_lines);
        }

        // A blank spacing row, then the divider rule, then the input. The blank
        // row gives the prompt one line of breathing room above its divider —
        // matching the single blank line the transcript leaves between blocks —
        // instead of the rule pressing directly against the last message.
        let composer_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
            ])
            .split(composer_outer);

        frame.render_widget(Clear, composer_chunks[0]);
        frame.render_widget(
            Paragraph::new(self.composer_rule_line(composer_chunks[1].width.max(1))),
            composer_chunks[1],
        );

        let composer_area = composer_chunks[2];
        // Record the area + scroll the draft so the cursor row stays on-screen,
        // then render the (pre-wrapped) visible rows. We wrap ourselves into
        // hanging-indent rows and render with wrap OFF, so rendering, cursor
        // placement, and mouse hit-testing all share one geometry.
        self.composer_text_area = Some(composer_area);
        self.update_composer_scroll(composer_area);
        let composer = Paragraph::new(self.render_composer_text());
        frame.render_widget(composer, composer_area);

        if self.should_show_input_cursor()
            && let Some((cursor_x, cursor_y)) = self.composer_cursor_screen(composer_area)
        {
            frame.set_cursor_position((cursor_x, cursor_y));
        }

        // Footer block: status line on top, contextual hint bar at the bottom.
        let footer_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(footer_outer);
        self.render_footer(frame, footer_rows[0]);
        self.render_hint_bar(frame, footer_rows[1]);
        composer_area
    }

    /// The divider rule above the composer. It always carries the auto-approve
    /// mode badge, inset near the right end — amber "on" when active, a faint
    /// "⇧⇥ … off" (naming the toggle key) when not — so the mode and how to
    /// change it are always discoverable. It gets this fixed home above the input
    /// rather than the hint bar, which drops its right-hand items on narrow
    /// terminals.
    pub(super) fn composer_rule_line(&self, width: u16) -> Line<'static> {
        let width = usize::from(width);
        // In `!cmd` shell mode the prompt's top line picks up the accent hue,
        // matching the accent-tinted draft text so shell mode reads at a glance.
        let rule_style = if self.draft_is_shell_command() {
            Style::default().fg(ACCENT)
        } else {
            Style::default().fg(FAINT)
        };
        let (badge, badge_style) = if self.agent_auto_approve {
            ("⚡ auto-approve: on", Style::default().fg(WARNING))
        } else {
            ("⇧⇥ auto-approve: off", Style::default().fg(MUTED))
        };
        // Left title on the rule. While recalling input history, show
        // `History {pos}/{total}` — the newest entry reads as total/total and
        // counts down as you scroll further back — preceded by two rule dashes
        // so it reads as a titled divider (matching the recall affordance).
        // Otherwise, a live `/goal` step indicator pinned to the very left so an
        // unattended loop is always visible (not just in a transient notice).
        // The two never coincide in one frame: history recall is a foreground
        // composer action.
        let (left_text, left_style, left_lead) = if let Some(index) = self.draft_history_index {
            (
                format!(" History {}/{} ", index + 1, self.draft_history.len()),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                2usize,
            )
        } else if let Some(goal) = self.goal_mode.as_ref() {
            (
                format!(" ◎ goal {}/{} ", goal.iteration, goal.max),
                Style::default().fg(ACCENT),
                0usize,
            )
        } else {
            (String::new(), Style::default(), 0usize)
        };
        let trailing = 2usize;
        // Badge cell width including one space of padding on each side.
        let badge_w = display_width(badge) + 2;
        let left_w = if left_text.is_empty() {
            0
        } else {
            left_lead + display_width(&left_text)
        };
        if width <= left_w + badge_w + trailing + 2 {
            // Too narrow to inset both — keep the safety-critical auto-approve badge.
            return Line::from(Span::styled(badge.to_string(), badge_style));
        }
        let fill = width - left_w - badge_w - trailing;
        let mut spans = Vec::with_capacity(5);
        if !left_text.is_empty() {
            if left_lead > 0 {
                spans.push(Span::styled("─".repeat(left_lead), rule_style));
            }
            spans.push(Span::styled(left_text, left_style));
        }
        spans.push(Span::styled("─".repeat(fill), rule_style));
        spans.push(Span::styled(format!(" {badge} "), badge_style));
        spans.push(Span::styled("─".repeat(trailing), rule_style));
        Line::from(spans)
    }

    /// The pinned plan/task-list panel's content lines (the `Plan N/M done`
    /// header plus one line per step), or empty when there's no plan. Built fresh
    /// each frame — it's small, and the plan changes rarely.
    fn plan_panel_lines(&self) -> Vec<StyledLine> {
        let Some(content) = self
            .history
            .iter()
            .rev()
            .find(|m| m.role == "plan")
            .map(|m| m.content.as_str())
        else {
            return Vec::new();
        };
        let mut lines = Vec::new();
        render_plan(&mut lines, content);
        lines
    }

    /// Rows the pinned plan panel will occupy (0 when there's no plan): a top rule
    /// plus the wrapped checklist, capped so the transcript keeps a usable minimum
    /// and a long plan can't dominate the screen (it scrolls instead).
    fn plan_panel_height(
        &self,
        lines: &[StyledLine],
        area: Rect,
        composer_height: u16,
        footer_height: u16,
    ) -> u16 {
        if lines.is_empty() {
            return 0;
        }
        let body_width = area.width.saturating_sub(2).max(1);
        let plain: Vec<String> = lines.iter().map(|l| l.plain.clone()).collect();
        let content_rows = wrap_plain_lines(&plain, body_width).len() as u16;
        let reserved = composer_height
            .saturating_add(footer_height)
            .saturating_add(PLAN_PANEL_MIN_TRANSCRIPT);
        let max_body = area
            .height
            .saturating_sub(reserved)
            .min(area.height / 3)
            .max(1);
        content_rows.clamp(1, max_body).saturating_add(1) // + top rule
    }

    /// Paint the pinned plan panel: a faint top rule (fencing it off from the
    /// transcript, mirroring the composer's divider) over the wrapped checklist.
    /// When the plan overflows the panel, scroll so the active (`in_progress`)
    /// step stays on screen.
    fn render_plan_panel(&self, frame: &mut Frame<'_>, area: Rect, lines: &[StyledLine]) {
        if area.height == 0 || lines.is_empty() {
            return;
        }
        frame.render_widget(Clear, area);
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "─".repeat(usize::from(area.width.max(1))),
                Style::default().fg(FAINT),
            ))),
            Rect {
                x: area.x,
                y: area.y,
                width: area.width,
                height: 1,
            },
        );
        let body = Rect {
            x: area.x.saturating_add(1),
            y: area.y.saturating_add(1),
            width: area.width.saturating_sub(2).max(1),
            height: area.height.saturating_sub(1),
        };
        if body.height == 0 {
            return;
        }
        let bars = vec![None; lines.len()];
        let wrapped = wrap_transcript(lines, &bars, body.width);
        // Keep the in-progress step visible when the plan is taller than the panel.
        let scroll = wrapped
            .rows
            .iter()
            .position(|r| r.contains('▸'))
            .filter(|&i| i >= usize::from(body.height))
            .map(|i| (i + 1 - usize::from(body.height)) as u16)
            .unwrap_or(0);
        frame.render_widget(Paragraph::new(wrapped.text).scroll((scroll, 0)), body);
    }

    pub(super) fn empty_state_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(1).max(1);
        let mut height = if let Some(loading) = &self.loading_resume {
            let mut rows = vec![
                "Loading saved chat…".to_string(),
                loading.preview.title.clone(),
                plain_text_from_spans(&resume_metadata_spans(
                    &loading.preview,
                    content_width.saturating_sub(1).max(1),
                )),
                self.display_cwd().to_string(),
            ];
            rows.extend(self.notice_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        } else {
            let mut rows = self.transcript_intro_lines();
            rows.extend(self.notice_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        };
        height = height
            .saturating_add(EMPTY_STATE_TOP_GAP)
            .saturating_add(EMPTY_STATE_BOTTOM_GAP);
        height.max(1)
    }

    fn notice_plain_lines(&self, width: u16) -> Vec<String> {
        notice_display(self.notice.as_ref())
            .map(|(_, text)| {
                let mut lines = vec![String::new()];
                lines.extend(wrap_plain_lines(&[text.into_owned()], width));
                lines
            })
            .unwrap_or_default()
    }

    /// Paint a `▌` accent bar in the reserved gutter column for each visible
    /// transcript row, colored by the role of the block that owns that row.
    fn paint_accent_gutter(
        &self,
        frame: &mut Frame<'_>,
        gutter_x: u16,
        top_y: u16,
        view_height: u16,
        bars: &[Option<Color>],
    ) {
        let buffer = frame.buffer_mut();
        for offset in 0..view_height {
            let row_index = self.transcript_scroll + usize::from(offset);
            let Some(Some(color)) = bars.get(row_index).copied() else {
                continue;
            };
            if let Some(cell) = buffer.cell_mut((gutter_x, top_y.saturating_add(offset))) {
                cell.set_symbol("▌");
                cell.set_fg(color);
            }
        }
    }

    /// Auto-clears the selection once the post-copy flash window elapses, so a
    /// just-copied selection briefly lights up then disappears (amp-style).
    pub(super) fn tick_selection_flash(&mut self) {
        if let Some(until) = self.selection_flash_until
            && Instant::now() >= until
        {
            self.selection_flash_until = None;
            self.transcript_selection = None;
            self.screen_selection = None;
        }
    }

    fn render_transcript_selection_highlight(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(selection) = self
            .transcript_selection
            .filter(|selection| !selection.is_empty())
        else {
            return;
        };
        let Some(hitbox) = &self.transcript_hitbox else {
            return;
        };

        let wash = if self.selection_flash_until.is_some() {
            SELECT_FLASH
        } else {
            SELECT_WARM
        };
        let (start, end) = normalized_selection(selection);
        let visible_start = hitbox.first_row;
        let visible_end = visible_start.saturating_add(usize::from(area.height));
        let row_start = start.row.max(visible_start);
        let row_end = end.row.min(visible_end.saturating_sub(1));
        if row_start > row_end {
            return;
        }

        let buffer = frame.buffer_mut();
        for row in row_start..=row_end {
            let local_y = row.saturating_sub(visible_start) as u16;
            // Clamp the wash to the row's real text so we never paint the blank
            // cells past a line's end — keeps the highlight matching what copy
            // actually yields (trailing space is trimmed on copy).
            let text_width = hitbox
                .rows
                .get(row)
                .map(|line| row_display_width(line))
                .unwrap_or(0);
            let start_col = if row == start.row { start.column } else { 0 };
            let end_col = if row == end.row {
                end.column
            } else {
                text_width
            };
            let start_col = start_col.min(area.width);
            let end_col = end_col.min(text_width).min(area.width);
            if start_col >= end_col {
                continue;
            }

            for column in start_col..end_col {
                if let Some(cell) = buffer.cell_mut((area.x + column, area.y + local_y)) {
                    cell.set_bg(wash);
                }
            }
        }
    }

    pub(super) fn render_toast(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let Some(toast) = self.toast.clone() else {
            return;
        };
        let now = Instant::now();
        if now >= toast.expires_at {
            self.toast = None;
            return;
        }

        let text_width = display_width(&toast.text).min(usize::from(area.width));
        let toast_width = (text_width as u16).saturating_add(4).min(area.width.max(1));
        // Anchor bottom-right: float on the last transcript row, just above the
        // composer rule, so the confirmation appears near where the user acted
        // without clobbering the divider, composer, or footer. Falls back to the
        // top edge before the first layout records the composer area.
        let anchor_y = self
            .composer_text_area
            .map(|c| c.y.saturating_sub(2))
            .unwrap_or(area.y);
        let toast_area = Rect {
            x: area
                .x
                .saturating_add(area.width.saturating_sub(toast_width)),
            y: anchor_y,
            width: toast_width,
            height: 1,
        };
        let color = if now.duration_since(toast.created_at) >= TOAST_FADE_AFTER {
            FAINT
        } else {
            ACCENT
        };
        frame.render_widget(Clear, toast_area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::raw(" "),
                Span::styled(&toast.text, Style::default().fg(color)),
                Span::raw(" "),
            ]))
            .style(Style::default().bg(Color::Rgb(24, 26, 27))),
            toast_area,
        );
    }

    pub(super) fn render_empty_state(&self, frame: &mut Frame<'_>, area: Rect) {
        let content_area = Rect {
            x: area.x,
            y: area.y.saturating_add(EMPTY_STATE_TOP_GAP),
            width: area.width,
            height: area
                .height
                .saturating_sub(EMPTY_STATE_TOP_GAP)
                .saturating_sub(EMPTY_STATE_BOTTOM_GAP),
        };

        let lines = if let Some(loading) = &self.loading_resume {
            vec![
                Line::from(vec![
                    Span::styled(
                        "Loading",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " saved chat…",
                        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    loading.preview.title.clone(),
                    Style::default().fg(TEXT),
                )),
                Line::from(resume_metadata_spans(
                    &loading.preview,
                    area.width.max(1).saturating_sub(2),
                )),
                Line::from(Span::styled(self.display_cwd(), Style::default().fg(FAINT))),
            ]
        } else {
            // Two-row wordmark + a muted tagline. Mirrors `transcript_intro_lines`
            // (height) and shares the wordmark with `push_transcript_intro`.
            let mut lines: Vec<Line<'static>> = brand_wordmark_lines()
                .into_iter()
                .map(|sl| sl.line)
                .collect();
            lines.push(Line::from(Span::styled(
                BRAND_TAGLINE,
                Style::default().fg(MUTED),
            )));
            lines
        };

        let mut lines = lines;
        if let Some((color, text)) = notice_display(self.notice.as_ref()) {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                text.into_owned(),
                Style::default().fg(color),
            )));
        }

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            content_area,
        );
    }

    pub(super) fn render_composer_text(&self) -> Text<'static> {
        let prompt = if self.draft_history_index.is_some() {
            Span::styled(
                "^ ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else if self.draft_is_shell_command() {
            // `!cmd` shell mode tints the prompt itself in the accent hue too.
            Span::styled(
                "> ",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("> ", Style::default().fg(USER).add_modifier(Modifier::BOLD))
        };
        let mut lines = composer_attachment_lines(&self.draft_attachments);
        if self.draft.is_empty() {
            let placeholder = if self.loading_resume.is_some() {
                Span::styled("Resume loading…", Style::default().fg(FAINT))
            } else if self.sending {
                Span::styled(
                    " Type to queue your next message…",
                    Style::default().fg(FAINT),
                )
            } else {
                Span::styled(" Ask anything · / for commands", Style::default().fg(FAINT))
            };
            lines.push(Line::from(vec![prompt, placeholder]));
            return Text::from(lines);
        }

        // Ghost hint trailing a bare slash command (Claude-Code style), e.g.
        // `> /mcp [add … | rm <name>]`. Only set when the draft is a single line.
        let ghost = self.composer_command_hint();
        // A `!cmd` draft is tinted in the accent hue, so shell mode reads at a
        // glance — the prompt and its top divider pick up the same accent color.
        let draft_color = if self.draft_is_shell_command() {
            ACCENT
        } else {
            TEXT
        };
        // Pre-wrapped visual rows: row 0 gets the `> ` prompt; every other row
        // gets a matching 2-col hanging indent so wrapped text aligns under the
        // first character. Rendered with wrap OFF (see `render_main`).
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let last = rows.len().saturating_sub(1);
        for (index, &(start, end)) in rows.iter().enumerate().skip(self.composer_scroll) {
            let prefix = if index == 0 {
                prompt.clone()
            } else {
                Span::raw("  ")
            };
            let mut spans = vec![
                prefix,
                Span::styled(
                    self.draft[start..end].to_string(),
                    Style::default().fg(draft_color),
                ),
            ];
            if index == last
                && let Some(hint) = ghost
            {
                spans.push(Span::styled(format!(" {hint}"), Style::default().fg(FAINT)));
            }
            lines.push(Line::from(spans));
        }

        Text::from(lines)
    }

    /// Scroll the draft within the composer so the cursor's visual row stays
    /// inside the visible window. Attachment lines sit above the draft and aren't
    /// scrolled; only the draft rows scroll. Recomputed each render.
    pub(super) fn update_composer_scroll(&mut self, area: Rect) {
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let attach = self.draft_attachments.len();
        let visible = usize::from(area.height).saturating_sub(attach).max(1);
        let (cursor_row, _) = composer_cursor_rowcol(&self.draft, self.cursor, &rows);
        if cursor_row < self.composer_scroll {
            self.composer_scroll = cursor_row;
        } else if cursor_row >= self.composer_scroll + visible {
            self.composer_scroll = cursor_row + 1 - visible;
        }
        self.composer_scroll = self.composer_scroll.min(rows.len().saturating_sub(visible));
    }

    /// Absolute terminal `(x, y)` for the input cursor, in the composer's
    /// hanging-indent wrap model, accounting for attachment rows and scroll.
    /// `None` when the cursor row is scrolled out of view.
    pub(super) fn composer_cursor_screen(&self, area: Rect) -> Option<(u16, u16)> {
        let (x_rel, row) = cursor_position(
            &self.draft,
            self.cursor,
            area.width.max(1),
            COMPOSER_PREFIX_WIDTH,
        );
        let row = usize::from(row);
        if row < self.composer_scroll {
            return None;
        }
        let attach = self.draft_attachments.len() as u16;
        let y = area.y + attach + (row - self.composer_scroll) as u16;
        let x = area.x + x_rel;
        let max_x = area.x + area.width.saturating_sub(1);
        let max_y = area.y + area.height.saturating_sub(1);
        Some((x.min(max_x), y.min(max_y)))
    }

    pub(super) fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        let (right_label, right_color) = self.footer_status_label();
        let right_label_width = display_width(&right_label) as u16;
        let left_width = if right_label_width == 0 {
            area.width
        } else {
            area.width.saturating_sub(right_label_width + 1)
        };
        let left_text = build_footer_text(
            &self.raw_model,
            &self.key.base_url,
            self.display_cwd(),
            self.git_branch.as_deref(),
            self.active_agent.as_deref(),
            left_width,
        );
        let left_len = display_width(&left_text) as u16;
        let pad = left_width.saturating_sub(left_len);
        let mut spans = vec![Span::styled(left_text, Style::default().fg(MUTED))];
        if right_label_width > 0 {
            spans.push(Span::raw(" ".repeat(usize::from(pad) + 1)));
            spans.push(Span::styled(right_label, Style::default().fg(right_color)));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Contextual shortcut bar: the most relevant keys for the current state on
    /// the left, plus the queued-message indicator on the right.
    pub(super) fn render_hint_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let left = self.hint_key_spans();
        let right = self.hint_indicator_spans();
        let span_w = |spans: &[Span]| -> usize {
            spans
                .iter()
                .map(|s| display_width(s.content.as_ref()))
                .sum()
        };
        let total = usize::from(area.width);
        let left_w = span_w(&left);
        let right_w = span_w(&right);
        let mut spans = left;
        if right_w > 0 && total > left_w + right_w + 1 {
            spans.push(Span::raw(" ".repeat(total - left_w - right_w)));
            spans.extend(right);
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// The left-hand key hints, chosen by the current state.
    fn hint_key_spans(&self) -> Vec<Span<'static>> {
        let pairs: Vec<(&str, &str)> = if self.pending_mcp_consent.is_some() {
            vec![("y", "run"), ("a", "always"), ("n", "deny")]
        } else if self.agent_permission.is_some() {
            vec![("y", "allow"), ("n", "deny"), ("a", "always")]
        } else if self.sending {
            vec![("esc", "interrupt"), ("type", "queue next")]
        } else {
            let mut idle = vec![("/", "commands"), ("↑", "history")];
            idle.push(("^C", "exit"));
            idle
        };
        let mut spans = Vec::new();
        for (idx, (keycap, label)) in pairs.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::styled("   ".to_string(), Style::default().fg(FAINT)));
            }
            spans.push(Span::styled(keycap.to_string(), Style::default().fg(MUTED)));
            spans.push(Span::styled(
                format!(" {label}"),
                Style::default().fg(FAINT),
            ));
        }
        spans
    }

    /// Right-hand state indicators. Just the queued-message dot now — the
    /// auto-approve mode badge lives on the composer rule (see
    /// [`Self::composer_rule_line`]) so a permission-bypass mode can't be dropped
    /// when a narrow terminal squeezes this bar.
    fn hint_indicator_spans(&self) -> Vec<Span<'static>> {
        let mut spans: Vec<Span<'static>> = Vec::new();
        // Reasoning effort — the same effective level the engine sends, so they
        // can't disagree. Shown only when thinking is on for a thinking-capable
        // model; hidden when thinking is off (the engine isn't reasoning then).
        if self.thinking_enabled
            && self.model_supports_thinking
            && let Some(level) = self.effective_reasoning_effort()
        {
            spans.push(Span::styled(
                format!("effort: {level}"),
                Style::default().fg(TOOL),
            ));
        }
        let queued = self.queued_messages.len();
        if queued > 0 {
            if !spans.is_empty() {
                spans.push(Span::styled("   ".to_string(), Style::default().fg(FAINT)));
            }
            spans.push(Span::styled(
                format!("● {queued} queued"),
                Style::default().fg(ACCENT),
            ));
        }
        spans
    }

    /// What the turn is doing right now, derived from the live transcript so no
    /// extra state is needed: streaming a reply, running a tool, or waiting on
    /// the model. Drives the footer status and the in-stream spinner.
    pub(super) fn processing_activity(&self) -> String {
        if !self.pending_response.is_empty() || !self.incoming_buffer.is_empty() {
            return "working".to_string();
        }
        if let Some(last) = self.history.last()
            && last.role == "tool_call"
        {
            let verb = serde_json::from_str::<serde_json::Value>(&last.content)
                .ok()
                .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
                .unwrap_or_else(|| "tool".to_string());
            // "subagent" is jargon — surface the delegation as a plain verb.
            if verb == "subagent" {
                return "delegating".to_string();
            }
            return format!("running {verb}");
        }
        "thinking".to_string()
    }

    /// Tokens to show in the footer fill right now, and whether the figure is a
    /// chars/4 estimate rather than provider-measured. During an in-flight turn we
    /// prefer the live measured usage (Anthropic streams it from `message_start`);
    /// until that lands we grow a chars/4 estimate of the transcript plus the text
    /// streamed so far, so the fill still moves for providers that only report
    /// usage at the end of the turn. Idle: the last turn's measured total.
    fn context_fill(&self) -> (u64, bool) {
        if self.sending {
            if let Some(usage) = self.live_usage {
                return (usage.total_tokens(), false);
            }
            // No measured usage yet this turn: grow from the best known baseline —
            // the prior turn's fill, or the transcript estimate when larger (a fresh
            // chat with no prior turn) — plus the text streamed so far. Taking the
            // max avoids the footer dropping at turn start when the prior fill was a
            // measured total (which exceeds the chars/4 transcript estimate).
            let streamed = (self.pending_response.len()
                + self.incoming_buffer.len()
                + self.pending_reasoning.len()) as u64
                / 4;
            let baseline = self
                .context_tokens
                .max(estimate_context_tokens(&self.history));
            return (baseline + streamed, true);
        }
        match self.last_usage {
            Some(usage) => (usage.total_tokens(), false),
            None => (self.context_tokens, self.context_is_estimate),
        }
    }

    pub(super) fn footer_status_label(&self) -> (String, Color) {
        let (used, is_estimate) = self.context_fill();
        if self.context_window == 0 {
            // No known window: just the count. Use the live measurement while a
            // turn is in flight so the figure tracks the stream, else the last
            // turn's; a `None` makes `format_token_count` flag it `~` (estimate).
            let usage = if self.sending {
                self.live_usage
            } else {
                self.last_usage
            };
            return (format_token_count(used, usage), MUTED);
        }
        let pct = (used.saturating_mul(100) / self.context_window).min(100);
        // Mark estimate-only figures (cursor ACP / agents without reported usage):
        // aivo's tracked transcript is a fraction of the model's real context, so
        // the number understates the true fill — `~` flags it as approximate.
        let approx = if is_estimate && used > 0 { "~" } else { "" };
        // A non-zero fill that rounds down to 0% reads as `<1%` rather than a flat
        // `0%` (common for cursor, whose real context we can't measure and only
        // estimate from the transcript).
        let pct_label = if pct == 0 && used > 0 {
            "<1%".to_string()
        } else {
            format!("{approx}{pct}%")
        };
        let label = format!(
            "{approx}{} / {} · {pct_label}",
            format_token_count_value(used),
            format_token_count_value(self.context_window),
        );
        (label, context_fill_color(pct))
    }

    pub(super) fn composer_height(&self, width: u16) -> u16 {
        // Count wrapped *visual* rows, not logical lines, so a long line that
        // wraps grows the box (and keeps the cursor on-screen) instead of being
        // clipped. The clamp caps growth at 7 text rows; longer drafts scroll
        // within the box (see `composer_scroll`).
        let draft_rows = if self.draft.is_empty() {
            1
        } else {
            let text_width = usize::from(width)
                .saturating_sub(usize::from(COMPOSER_PREFIX_WIDTH))
                .max(1);
            composer_visual_rows(&self.draft, text_width).len()
        };
        let lines = (draft_rows + self.draft_attachments.len()) as u16;
        // +3 reserves the leading blank spacing row, the divider rule, and one
        // trailing row below the input; the rest is draft text (capped, then it
        // scrolls within the box).
        (lines + 3).clamp(4, 10)
    }

    /// Wrap width available to the composer's draft text (the rendered composer
    /// width minus the per-row prompt indent). Falls back to a sane default
    /// before the first render has recorded the area.
    pub(super) fn composer_text_width(&self) -> usize {
        self.composer_text_area
            .map(|area| usize::from(area.width))
            .unwrap_or(80)
            .saturating_sub(usize::from(COMPOSER_PREFIX_WIDTH))
            .max(1)
    }
}

/// A human-readable question for the permission card heading. Known mutating
/// tools get a plain-language phrase; anything else (e.g. an MCP tool) falls
/// back to its raw name.
fn permission_heading(tool: &str) -> String {
    match tool {
        "run_bash" => "Run a command?".to_string(),
        "run_bash_unsandboxed" => "Run outside the workspace sandbox?".to_string(),
        "cursor" => "Allow Cursor to run this?".to_string(),
        "write_file" => "Write a file?".to_string(),
        "edit_file" | "multi_edit" => "Edit a file?".to_string(),
        other => format!("Allow {other}?"),
    }
}

/// The project-MCP consent choices row: run once / always (this repo) / deny,
/// color-coded like the permission card's traffic light.
fn mcp_consent_keys_line() -> Line<'static> {
    let keycap = |key: &str, color: Color| {
        Span::styled(
            key.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    Line::from(vec![
        keycap("y", ASSISTANT),
        label(" run once"),
        gap(),
        keycap("a", WARNING),
        label(" always (this repo)"),
        gap(),
        keycap("n", ERROR),
        label(" deny"),
    ])
}

/// The choices row: color-coded keycaps reading like a traffic light —
/// green allow, amber always (it arms auto-approve), red deny. `tool` selects
/// the "always" scope wording (a Cursor card's "always" is session-wide, unlike
/// the native engine's, which is scoped to this one command/path), and
/// `composing` swaps in a hint when a queued-message draft is in progress —
/// there the letter keys type into the draft instead of deciding (see
/// `handle_permission_key`).
fn permission_keys_line(tool: &str, composing: bool) -> Line<'static> {
    let keycap = |key: &str, color: Color| {
        Span::styled(
            key.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT));
    if composing {
        // A draft is in progress, so y/a/n flow into the message, not the card.
        // Only Esc (deny) and Shift+Tab (allow + auto-approve) act on the card.
        return Line::from(vec![
            keycap("⇧⇥", WARNING),
            label(" allow"),
            gap(),
            keycap("esc", ERROR),
            label(" deny"),
            gap(),
            label("y/a/n type into your message"),
        ]);
    }
    // Cursor's "always" turns on auto-approve for the rest of the session (its
    // out-of-process tools can't be remembered per-action), so spell that out;
    // the native engine's "always" is scoped to this command/path and reads as
    // expected without a qualifier.
    let always_label = if tool == "cursor" {
        " always (this session)"
    } else {
        " always"
    };
    Line::from(vec![
        keycap("y", ASSISTANT),
        label(" allow once"),
        gap(),
        keycap("a", WARNING),
        label(always_label),
        gap(),
        keycap("n", ERROR),
        label(" deny"),
    ])
}
