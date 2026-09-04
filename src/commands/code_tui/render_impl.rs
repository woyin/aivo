use super::*;

/// Trailing tool entries of a run left visible outside its step fold (~4 steps
/// with separate results) — the recent context worth keeping at a glance.
const FOLD_KEEP_TAIL: usize = 8;
/// Minimum entries a step fold must cover — folding a handful of rows behind a
/// marker costs more indirection than it saves.
const FOLD_MIN: usize = 12;

/// Width the transcript body is built to fit (tables, pre-wrapped turn blocks):
/// full column minus left gutter and right margin. Must equal the paint-time text
/// area's width, or pre-wrapped rows get re-broken flush at the gutter.
fn table_layout_width(area_width: u16) -> u16 {
    area_width.saturating_sub(ACCENT_GUTTER_WIDTH + TRANSCRIPT_RIGHT_MARGIN)
}

/// True when a line can't merge backward across the blank line before it —
/// indented continuations and list items (loose lists span blanks) can.
fn is_safe_block_start(line: &str) -> bool {
    if line.starts_with(' ') || line.starts_with('\t') {
        return false;
    }
    if matches!(line, "-" | "*" | "+")
        || line.starts_with("- ")
        || line.starts_with("* ")
        || line.starts_with("+ ")
    {
        return false;
    }
    let digits = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0
        && matches!(line[digits..].chars().next(), Some('.') | Some(')'))
        && matches!(line[digits + 1..].chars().next(), Some(' ') | None)
    {
        return false;
    }
    true
}

/// Largest markdown-safe settle boundary in `src` at or after `from` (itself a
/// boundary): the start of a non-blank [`is_safe_block_start`] line after a
/// blank line, outside code fences and blank-spanning raw-HTML blocks. The
/// candidate line itself stays live, so the suffix always has content.
fn settled_reply_boundary(src: &str, from: usize) -> usize {
    let mut best = from;
    let mut fence: Option<(char, usize)> = None;
    let mut html_close: Option<&'static str> = None;
    let mut prev_blank = false;
    let mut pos = from;
    for line in src[from..].split_inclusive('\n') {
        // Never judge the unterminated live edge: "2" may grow into "2. item"
        // and continue a loose list, invalidating a latched boundary.
        if !line.ends_with('\n') {
            break;
        }
        let content = line.trim_end_matches(['\n', '\r']);
        let trimmed = content.trim();
        if let Some(closer) = html_close {
            if content.contains(closer) {
                html_close = None;
            }
            prev_blank = trimmed.is_empty();
            pos += line.len();
            continue;
        }
        let fence_len = trimmed
            .chars()
            .take_while(|&c| c == '`' || c == '~')
            .count();
        match fence {
            Some((ch, len)) => {
                // Closing fence: same char, at least as long, nothing after.
                if fence_len >= len && trimmed.chars().all(|c| c == ch) {
                    fence = None;
                }
            }
            None => {
                if prev_blank && !trimmed.is_empty() && is_safe_block_start(content) {
                    best = pos;
                }
                if fence_len >= 3 {
                    let ch = trimmed.chars().next().unwrap();
                    if trimmed[..fence_len].chars().all(|c| c == ch) {
                        fence = Some((ch, fence_len));
                    }
                } else {
                    let lower = trimmed.to_ascii_lowercase();
                    for (open, close) in [
                        ("<pre", "</pre>"),
                        ("<script", "</script>"),
                        ("<style", "</style>"),
                        ("<!--", "-->"),
                    ] {
                        if lower.starts_with(open) && !lower.contains(close) {
                            html_close = Some(close);
                            break;
                        }
                    }
                }
            }
        }
        prev_blank = trimmed.is_empty();
        pos += line.len();
    }
    best
}

/// Replace every control-char cell (tab, ESC, …) with a space, keeping its
/// style — a raw `\t` (unicode-width 1) desyncs the terminal's cell grid. Run on
/// the finished frame so no widget can poison the grid, whatever its source.
fn scrub_control_cells(buffer: &mut ratatui::buffer::Buffer) {
    for cell in &mut buffer.content {
        if cell.symbol().chars().any(char::is_control) {
            cell.set_symbol(" ");
        }
    }
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
    let style = Style::default().fg(palette().jump_fg).bg(palette().jump_bg);
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

impl CodeTuiApp {
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
        // adjusted) — the width tables should fit into.
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

    /// Committed history length for rendering: hides the trailing `tool_call`
    /// run while any of it is in flight — the status line (and batch rows)
    /// names the work instead. Cursor resolves entries out of order, so the
    /// cards wait for the whole run.
    pub(super) fn committed_render_len(&self) -> usize {
        let mut render_len = self.history.len();
        if self.sending {
            let mut start = render_len;
            while start > 0 && self.history[start - 1].role == "tool_call" {
                start -= 1;
            }
            let live = self.history[start..render_len].iter().any(|m| {
                let (result, failed) = decode_tool_outcome(&m.content);
                result.is_none() && !failed
            });
            if live {
                render_len = start;
            }
        }
        render_len
    }

    /// `(start, len)` history spans that fold to one `▸ N earlier steps` row:
    /// within each maximal run of consecutive tool entries, everything except
    /// the trailing [`FOLD_KEEP_TAIL`] entries — kept only when the foldable
    /// prefix is itself long enough ([`FOLD_MIN`]) to be worth the indirection.
    /// Pure over (history, render_len) so render and click mapping agree.
    pub(super) fn step_folds(&self, render_len: usize) -> Vec<(usize, usize)> {
        // A plan payload renders as a card, never inside a fold.
        let foldable = |m: &ChatMessage| match m.role.as_str() {
            "tool_result" => true,
            "tool_call" => decode_tool_name(&m.content) != "exit_plan_mode",
            _ => false,
        };
        let mut folds = Vec::new();
        let mut i = 0;
        while i < render_len {
            if !foldable(&self.history[i]) {
                i += 1;
                continue;
            }
            let start = i;
            while i < render_len && foldable(&self.history[i]) {
                i += 1;
            }
            let run_len = i - start;
            if run_len <= FOLD_KEEP_TAIL {
                continue;
            }
            let mut fold_len = run_len - FOLD_KEEP_TAIL;
            // Never split a call from its trailing result(s): grow the fold
            // until the first visible entry is a call (or the run ends).
            while start + fold_len < i && self.history[start + fold_len].role == "tool_result" {
                fold_len += 1;
            }
            if fold_len >= FOLD_MIN {
                folds.push((start, fold_len));
            }
        }
        folds
    }

    /// `(steps, tool counts sorted by frequency, failures)` for a fold's span.
    /// One JSON parse per entry: the name and a cursor-style `failed` come off
    /// the same value; in-process failures come off the following `tool_result`.
    fn fold_summary(&self, start: usize, len: usize) -> (usize, Vec<(String, usize)>, usize) {
        let mut steps = 0;
        let mut failed = 0;
        let mut counts: Vec<(String, usize)> = Vec::new();
        let mut last_tool: Option<String> = None;
        for m in &self.history[start..start + len] {
            match m.role.as_str() {
                "tool_call" => {
                    let v = serde_json::from_str::<serde_json::Value>(&m.content)
                        .unwrap_or(serde_json::Value::Null);
                    let name =
                        canonical_tool_name(v.get("name").and_then(|x| x.as_str()).unwrap_or(""))
                            .to_string();
                    steps += 1;
                    let disp = tool_display_name(&name);
                    match counts.iter_mut().find(|(n, _)| *n == disp) {
                        Some((_, c)) => *c += 1,
                        None => counts.push((disp, 1)),
                    }
                    if v.get("failed").and_then(|x| x.as_bool()).unwrap_or(false) {
                        failed += 1;
                    }
                    last_tool = Some(name);
                }
                "tool_result" if result_failed(&m.content, last_tool.as_deref()) => {
                    failed += 1;
                }
                _ => {}
            }
        }
        counts.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        (steps, counts, failed)
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
        // Last stamped assistant model; unstamped (pre-feature) turns don't reset it.
        let mut previous_model: Option<&str> = None;

        if self.is_transcript_empty() {
            push_styled_line(&mut lines, "", Style::default());
            bars.push(None);
            return RenderedTranscript::new(lines, bars);
        }

        // The welcome header (banner + tip) sits one column right of the messages:
        // build it apart, indent it, then append.
        let mut header = Vec::new();
        push_transcript_intro(&mut header, text_width.saturating_sub(HEADER_LEFT_INSET));
        // Tip stays pinned above the conversation; frozen once non-empty, so safe
        // to memoize.
        header.extend(self.welcome_status_lines());
        lines.extend(header.into_iter().map(|line| {
            if line.plain.is_empty() {
                line
            } else {
                indent_styled_line(line, usize::from(HEADER_LEFT_INSET))
            }
        }));
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

        // In-process emits separate `tool_result` lines (which carry per-call
        // targets); cursor enriches the call in place. With separate results, a
        // coalesced call line drops its target list to avoid repeating it.
        let separate_results = self.history.iter().any(|m| m.role == "tool_result");

        let render_len = self.committed_render_len();

        let (inline_result, consumed_results, split_calls) =
            self.parallel_batch_pairing(render_len);

        // The click handler recomputes this list for its ordinal map.
        let folds = self.step_folds(render_len);

        // One preview block per image PER TURN, whichever mention renders
        // first (see `push_preview_once`); reset at each user message so
        // "show it again" in a later turn renders again.
        let mut previewed: std::collections::HashSet<u64> = std::collections::HashSet::new();

        // The previous lone tool call's `→` row — the row an adjacent result
        // merges onto instead of spending a `⎿` line.
        let mut merge_anchor: Option<usize> = None;
        // A path the USER named previews under the ANSWER: anchored to the request
        // it lands above the reasoning still looking for it.
        let mut deferred_mention: Option<usize> = None;
        let mut idx = 0;
        while idx < render_len {
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
            // Turn ended with no assistant message (error, interrupt).
            if message.role == "user"
                && let Some(d_idx) = deferred_mention.take()
            {
                self.push_deferred_mention(
                    &mut lines,
                    &mut bars,
                    &mut previewed,
                    d_idx,
                    text_width,
                );
            }
            if should_add_message_spacing(previous_role, message.role.as_str()) {
                push_message_spacing(&mut lines);
                bars.resize(lines.len(), None);
            }
            // Model-switch divider; mirrors the share viewer's phase divider.
            if message.role == "assistant"
                && let Some(model) = message.model.as_deref()
            {
                if previous_model.is_some_and(|prev| prev != model) {
                    push_styled_line(
                        &mut lines,
                        format!("  model → {model}"),
                        Style::default().fg(MUTED()).add_modifier(Modifier::ITALIC),
                    );
                    bars.push(None);
                    push_styled_line(&mut lines, String::new(), Style::default());
                    bars.push(None);
                }
                previous_model = Some(model);
            }
            // A fold either collapses here to one summary row, or — expanded —
            // keeps a `▾` header above its normally-rendered rows.
            let fold_here = folds.iter().find(|&&(s, _)| s == idx).map(|&(_, l)| l);
            let collapsed_fold = fold_here.filter(|_| !self.expanded_step_folds.contains(&idx));
            if let Some(fold_len) = fold_here
                && collapsed_fold.is_none()
            {
                let (steps, _, _) = self.fold_summary(idx, fold_len);
                let header = line_with_plain(vec![Span::styled(
                    step_fold_header(steps),
                    Style::default().fg(MUTED()),
                )]);
                lines.push(indent_styled_line(header, usize::from(SUB_BLOCK_INDENT)));
                bars.push(Some(role_bar_color(message.role.as_str())));
            }
            let mut block = Vec::new();
            let mut advance = 1;
            let mut anchor_call = false;
            match message.role.as_str() {
                _ if collapsed_fold.is_some() => {
                    let fold_len = collapsed_fold.unwrap();
                    let (steps, counts, failed) = self.fold_summary(idx, fold_len);
                    render_step_fold(&mut block, steps, &counts, failed);
                    advance = fold_len;
                }
                "user" => {
                    previewed.clear();
                    // A skill invocation stores the full expanded SKILL.md as the
                    // user message (the model needs it), but the transcript should
                    // show the compact `/name args` the user actually typed.
                    let label = super::skill_invocation_label(&message.content);
                    let shown = label.as_deref().unwrap_or(&message.content);
                    render_user_message(&mut block, shown, &message.attachments, text_width);
                    self.push_attachment_preview_lines(
                        &mut block,
                        &mut previewed,
                        idx,
                        &message.content,
                        &message.attachments,
                        text_width,
                    );
                    deferred_mention = Some(idx);
                }
                "assistant" => {
                    let reasoning = self
                        .thinking_enabled
                        .then_some(message.reasoning_content.as_deref())
                        .flatten();
                    // Windowed by default; expanded only for turns the user clicked.
                    let expanded = self.expanded_thinking.contains(&idx);
                    let view = reasoning.map(|text| ReasoningView { text, expanded });
                    if self.plan_card_idx == Some(idx) {
                        push_plan_card(
                            &mut lines,
                            &mut bars,
                            view,
                            &message.content,
                            text_width,
                            Some("approve on the card · /plan go [guidance] · /plan exit to leave"),
                        );
                    } else {
                        push_assistant_blocks(
                            &mut lines,
                            &mut bars,
                            view,
                            &message.content,
                            text_width,
                            role_bar_color("assistant"),
                            true,
                        );
                        let mut extra = Vec::new();
                        if let Some(d_idx) = deferred_mention.take() {
                            self.push_text_image_preview_lines(
                                &mut extra,
                                &mut previewed,
                                d_idx,
                                &self.history[d_idx].content,
                                text_width,
                            );
                        }
                        self.push_text_image_preview_lines(
                            &mut extra,
                            &mut previewed,
                            idx,
                            &message.content,
                            text_width,
                        );
                        push_block(
                            &mut lines,
                            &mut bars,
                            extra,
                            Some(role_bar_color("assistant")),
                        );
                    }
                }
                "tool_call" => {
                    let (name, args) = decode_tool_call(&message.content);
                    // Render the plan payload as a card, not an opaque tool row.
                    if name == "exit_plan_mode" {
                        let plan = args.get("plan").and_then(|v| v.as_str()).unwrap_or("");
                        push_plan_card(&mut lines, &mut bars, None, plan, text_width, None);
                        previous_role = Some(message.role.as_str());
                        merge_anchor = None;
                        idx += 1;
                        continue;
                    }
                    // Coalesce adjacent same-verb calls into one line. Exceptions stay
                    // split: subagents (never an opaque `subagent ×N`) and mixed-batch
                    // calls, whose result inlines under them (`split_calls`).
                    let run = if name == "subagent" || split_calls.contains(&idx) {
                        1
                    } else {
                        self.tool_call_run_len(idx, &name)
                    };
                    // Don't coalesce into the hidden in-flight tail.
                    let run = run.min(render_len - idx);
                    // Nor across a fold start — every fold must render its own
                    // marker row, or the click ordinal → fold mapping shifts.
                    let run = match folds.iter().map(|&(s, _)| s).find(|&s| s > idx) {
                        Some(s) => run.min(s - idx),
                        None => run,
                    };
                    if run >= 2 {
                        let targets: Vec<String> = self.history[idx..idx + run]
                            .iter()
                            .map(|m| {
                                let (n, a) = decode_tool_call(&m.content);
                                let target = tool_call_target_display(&n, &a, cwd);
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
                        let header_targets: &[String] =
                            if separate_results { &[] } else { &targets };
                        render_tool_call_group(&mut block, &name, run, header_targets, failed);
                        advance = run;
                    } else {
                        let (result, failed) = decode_tool_outcome(&message.content);
                        let line_starts = decode_line_starts(&message.content);
                        let old_content = decode_old_content(&message.content);
                        let call_row = block.len();
                        render_tool_call(
                            &mut block,
                            &name,
                            &args,
                            result.as_deref(),
                            failed,
                            cwd,
                            &line_starts,
                            old_content.as_deref(),
                        );
                        if let Some(&res_idx) = inline_result.get(&idx) {
                            let content = &self.history[res_idx].content;
                            let expanded = self.expanded_output.contains(&res_idx);
                            let (spans, res_failed) = tool_result_spans(
                                content,
                                cwd,
                                Some(name.as_str()),
                                None,
                                expanded,
                                true,
                            );
                            merge_result_onto(&mut block[call_row], spans, res_failed);
                            render_tool_result_body(
                                &mut block,
                                content,
                                Some(name.as_str()),
                                expanded,
                            );
                            self.push_saved_image_preview_lines(
                                &mut block,
                                &mut previewed,
                                res_idx,
                                content,
                                Some(name.as_str()),
                                text_width,
                            );
                        } else {
                            anchor_call = true;
                        }
                        self.push_tool_image_preview_lines(
                            &mut block,
                            &mut previewed,
                            idx,
                            &message.content,
                            text_width,
                        );
                    }
                }
                // Drawn under its call; empty block still carries a `✻ Done in` marker.
                "tool_result" if consumed_results.contains(&idx) => {}
                "tool_result" => {
                    // `tool` fixes the unit (files/entries/matches); a detached
                    // call's target tags the result (see `tool_result_source`).
                    let (tool, label, detached) = match self.tool_result_source(idx, cwd) {
                        Some((name, target, detached)) => (
                            Some(name),
                            detached.then_some(target).filter(|t| !t.is_empty()),
                            detached,
                        ),
                        None => (None, None, true),
                    };
                    let expanded = self.expanded_output.contains(&idx);
                    match merge_anchor.take().filter(|_| !detached) {
                        // Ride the call row directly above: `→ verb(args) ▸ +N lines`.
                        Some(anchor) if !message.content.trim().is_empty() => {
                            let (spans, res_failed) = tool_result_spans(
                                &message.content,
                                cwd,
                                tool.as_deref(),
                                None,
                                expanded,
                                true,
                            );
                            merge_result_onto(&mut lines[anchor], spans, res_failed);
                            render_tool_result_body(
                                &mut block,
                                &message.content,
                                tool.as_deref(),
                                expanded,
                            );
                        }
                        Some(_) => {}
                        None => render_tool_result(
                            &mut block,
                            &message.content,
                            cwd,
                            tool.as_deref(),
                            label.as_deref(),
                            expanded,
                        ),
                    }
                    self.push_saved_image_preview_lines(
                        &mut block,
                        &mut previewed,
                        idx,
                        &message.content,
                        tool.as_deref(),
                        text_width,
                    );
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
                "error" => render_error_message(&mut block, &message.content),
                other => render_system_message(
                    &mut block,
                    other,
                    &message.content,
                    text_width.saturating_sub(SUB_BLOCK_INDENT),
                ),
            }
            // User `> ` turns and `◆ ` replies are main blocks; everything else in
            // `block` (tool calls, shell, plan/error/system notes) nests under them.
            let block = if message.role == "user" {
                block
            } else {
                indent_sub_block(block)
            };
            let bar = role_bar_color(message.role.as_str());
            let first_block_row = lines.len();
            push_block(&mut lines, &mut bars, block, Some(bar));
            merge_anchor = anchor_call.then_some(first_block_row);
            // The `✻ Done in …` marker for a turn stamped on its last entry (which
            // may sit inside a coalesced block, so scan the block's index range).
            if let Some((i, &ms)) =
                (idx..idx + advance).find_map(|i| self.turn_durations.get(&i).map(|ms| (i, ms)))
            {
                push_styled_line(&mut lines, String::new(), Style::default());
                bars.push(None);
                // Trailing per-turn tokens/cost note, when the finish recorded one.
                let note = self
                    .turn_notes
                    .get(&i)
                    .map(|n| format!(" · {n}"))
                    .unwrap_or_default();
                push_styled_line(
                    &mut lines,
                    format!(
                        "  ✻ Done in {}{note}",
                        format_request_elapsed(std::time::Duration::from_millis(ms))
                    ),
                    Style::default().fg(MUTED()).add_modifier(Modifier::ITALIC),
                );
                bars.push(None);
            }
            previous_role = Some(message.role.as_str());
            idx += advance;
        }
        // Reply not in yet (mid-turn, or an in-flight tool run hidden by
        // `committed_render_len`): render at the tail, else naming a path shows
        // nothing for the whole turn.
        if let Some(d_idx) = deferred_mention.take() {
            self.push_deferred_mention(&mut lines, &mut bars, &mut previewed, d_idx, text_width);
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
        if self.pending_response.is_empty() {
            // Thinking-only phase: stream the reasoning as a rolling window so the
            // user watches it think (the spinner carries elapsed/tokens).
            if self.thinking_enabled && reasoning_is_substantive(&self.pending_reasoning) {
                lines.push(blank_line());
                bars.push(None);
                let mut block = Vec::new();
                render_reasoning_window(
                    &mut block,
                    &self.pending_reasoning,
                    text_width.saturating_sub(SUB_BLOCK_INDENT),
                );
                push_block(&mut lines, &mut bars, indent_sub_block(block), None);
            }
        } else {
            // Answer started: show the same window above the streaming reply.
            let live_reasoning = (self.thinking_enabled
                && reasoning_is_substantive(&self.pending_reasoning))
            .then_some(self.pending_reasoning.as_str());
            lines.push(blank_line());
            bars.push(None);
            push_assistant_blocks(
                &mut lines,
                &mut bars,
                live_reasoning.map(|text| ReasoningView {
                    text,
                    expanded: false,
                }),
                &self.pending_response,
                text_width,
                ACCENT(),
                true,
            );
        }
        self.push_live_command_block(&mut lines, &mut bars);
        self.push_notice_block(&mut lines, &mut bars);
        (lines, bars)
    }

    /// A running `!cmd`'s live preview. Streams here (not into history) so the
    /// memoized body stays put; committed to history once it finishes. Bounded
    /// preview only — this runs per content change and a long command can
    /// buffer megabytes; only the first MAX_OUTPUT_LINES ever show.
    fn push_live_command_block(&self, lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
        let Some(run) = &self.local_command else {
            return;
        };
        lines.push(blank_line());
        bars.push(None);
        let total = run.stdout.lines().count() + run.stderr.lines().count();
        let content = serde_json::json!({
            "command": run.command,
            "stdout": first_lines(&run.stdout, MAX_PERSISTED_OUTPUT_LINES),
            "stderr": first_lines(&run.stderr, MAX_PERSISTED_OUTPUT_LINES),
            "total_lines": total,
            "running": true,
        })
        .to_string();
        let mut block = Vec::new();
        render_local_command(&mut block, &content, OutputView::Live);
        push_block(lines, bars, indent_sub_block(block), Some(SHELL()));
    }

    /// A terminal agent error lands twice — the transient notice and a durable
    /// `error` transcript entry. While that entry is the transcript's last
    /// message the notice would repeat it directly beneath, so it stays hidden.
    fn notice_repeats_last_error(&self) -> bool {
        match (&self.notice, self.history.last()) {
            (Some((color, text)), Some(last)) => {
                *color == ERROR() && last.role == "error" && last.content == *text
            }
            _ => false,
        }
    }

    fn push_notice_block(&self, lines: &mut Vec<StyledLine>, bars: &mut Vec<Option<Color>>) {
        let Some((color, _)) = notice_display(self.notice.as_ref()) else {
            return;
        };
        if self.notice_repeats_last_error() {
            return;
        }
        lines.push(blank_line());
        bars.push(None);
        let mut block = Vec::new();
        if let Some(spans) = notice_spans(self.notice.as_ref()) {
            block.push(line_with_plain(spans));
        }
        push_block(lines, bars, indent_sub_block(block), Some(color));
    }

    /// Pairs each `tool_call` in a *mixed* parallel batch with its result:
    /// (`call → result`, `results drawn under a call`, `calls to render split`). A
    /// *pure* batch coalesces into a `verb ×N` header instead, so it's skipped —
    /// interleaving its results would desync the history-ordered click-to-expand map.
    fn parallel_batch_pairing(
        &self,
        render_len: usize,
    ) -> (
        std::collections::HashMap<usize, usize>,
        std::collections::HashSet<usize>,
        std::collections::HashSet<usize>,
    ) {
        let mut inline = std::collections::HashMap::new();
        let mut consumed = std::collections::HashSet::new();
        let mut split = std::collections::HashSet::new();
        let role = |i: usize| self.history[i].role.as_str();
        let mut i = 0;
        while i < render_len {
            if role(i) != "tool_call" {
                i += 1;
                continue;
            }
            let call_start = i;
            while i < render_len && role(i) == "tool_call" {
                i += 1;
            }
            let res_start = i;
            while i < render_len && role(i) == "tool_result" {
                i += 1;
            }
            let calls = res_start - call_start;
            let results = i - res_start;
            // Pure → coalesce + clump (standalone arm); mixed → split & inline.
            let group0 =
                tool_group_key(&decode_tool_call(&self.history[call_start].content).0).to_string();
            let pure = (call_start + 1..res_start)
                .all(|k| group0 == tool_group_key(&decode_tool_call(&self.history[k].content).0));
            if calls < 2 || results == 0 || pure {
                continue;
            }
            for j in 0..calls {
                split.insert(call_start + j);
            }
            for j in 0..calls.min(results) {
                inline.insert(call_start + j, res_start + j);
                consumed.insert(res_start + j);
            }
        }
        (inline, consumed, split)
    }

    /// Length of the run of consecutive `tool_call` entries starting at `start`
    /// that share `name`'s coalescing verb (≥1; see `tool_group_key`).
    fn tool_call_run_len(&self, start: usize, name: &str) -> usize {
        let key = tool_group_key(name);
        self.history[start..]
            .iter()
            .take_while(|m| {
                m.role == "tool_call" && tool_group_key(&decode_tool_call(&m.content).0) == key
            })
            .count()
    }

    /// The `(tool name, target, detached)` for the `tool_result` at `idx`. Results
    /// are emitted after the whole batch in call order, so the j-th result pairs
    /// with the j-th call in the preceding call run (not `idx-1`). `detached` —
    /// the call isn't immediately before the result — means the result carries its
    /// own target, since no adjacent call line shows it.
    fn tool_result_source(&self, idx: usize, cwd: &str) -> Option<(String, String, bool)> {
        // Offset of this result within its contiguous run of results.
        let mut res_start = idx;
        while res_start > 0 && self.history[res_start - 1].role == "tool_result" {
            res_start -= 1;
        }
        let j = idx - res_start;
        // The matching calls are the contiguous tool_call run just before them.
        let mut call_start = res_start;
        while call_start > 0 && self.history[call_start - 1].role == "tool_call" {
            call_start -= 1;
        }
        let call_idx = call_start + j;
        if call_idx >= res_start {
            return None;
        }
        let (name, args) = decode_tool_call(&self.history[call_idx].content);
        let detached = call_idx + 1 != idx;
        Some((
            name.clone(),
            tool_call_target_display(&name, &args, cwd),
            detached,
        ))
    }

    /// The live status line (spinner + activity + elapsed + this turn's tokens),
    /// or `None` when idle. Rebuilt per frame and appended after the cached body
    /// so animation never invalidates the cache.
    pub(super) fn spinner_status_line(&self) -> Option<StyledLine> {
        // A background skill install drives the spinner when no turn or `!cmd`
        // owns it. Suppressed while a skills modal is open — its row narrates,
        // and the status must show in exactly one place.
        if !self.sending
            && self.local_command.is_none()
            && !matches!(self.overlay, Overlay::Skills(_) | Overlay::SkillInstall(_))
            && let Some(progress) = &self.installing_skill
        {
            let mut block = Vec::new();
            render_pending_status(
                &mut block,
                self.frame_tick,
                self.reduce_motion,
                progress.started.elapsed(),
                None,
                &progress.status_text(),
                "",
            );
            return block.into_iter().next();
        }
        let started_at = if self.sending {
            self.request_started_at
        } else if let Some(run) = &self.local_command {
            Some(run.started_at)
        } else {
            return None;
        };
        // Throttled label; fall back to the live one before the first tick.
        let activity = match &self.render_cache.status_display {
            Some((label, _)) => label.clone(),
            None => self.desired_status(),
        };
        // Steps · tokens (measured, else ~chars/4; 0 omitted) · queued count.
        let tail = if self.sending {
            let mut parts: Vec<String> = Vec::new();
            // >1 only — a single step would just restate the action label.
            if self.turn_steps > 1 {
                parts.push(format!("{} steps", self.turn_steps));
            }
            // Measured rounds + chars/4 of the stream since, so the count keeps
            // ticking through a long thought.
            let unmeasured = crate::agent::tokens::chars_to_tokens(
                self.turn_stream_chars
                    .saturating_sub(self.turn_stream_chars_measured),
            );
            let used = self.turn_output_tokens + unmeasured;
            if used > 0 {
                let approx = if unmeasured > 0 { "~" } else { "" };
                parts.push(format!("{approx}{} tokens", format_token_count_value(used)));
            }
            let queued = self.queued_input_count();
            if queued > 0 {
                parts.push(format!("{queued} queued"));
            }
            parts.join(" · ")
        } else {
            String::new()
        };
        // A named tool step is timed by its own runtime, not the whole turn's —
        // else a fast read reads "20m" when a sibling subagent dominates the turn.
        let (elapsed, deadline) = if self.current_action_label().is_some() {
            self.last_tool_action
                .as_ref()
                .map(|(_, since, budget)| {
                    (since.elapsed(), budget.map(std::time::Duration::from_secs))
                })
                .unwrap_or_default()
        } else if activity == "waiting"
            && let Some(last) = self.last_stream_activity
        {
            // The stall label times the stall — "waiting (6m)" next to the
            // whole-turn clock would read as a six-minute hang.
            (last.elapsed(), None)
        } else {
            (
                started_at
                    .map(|started_at| started_at.elapsed())
                    .unwrap_or_default(),
                None,
            )
        };
        let mut block = Vec::new();
        render_pending_status(
            &mut block,
            self.frame_tick,
            self.reduce_motion,
            elapsed,
            deadline,
            &activity,
            &tail,
        );
        block.into_iter().next()
    }

    /// Input typed mid-turn still waiting to run: steering + follow-ups + commands.
    pub(super) fn queued_input_count(&self) -> usize {
        let steering = self
            .steering_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        steering + self.queued_messages.len() + self.queued_commands.len()
    }

    /// The status label right now, pre-throttle: a decision card names the wait,
    /// a tool step names itself, else "Working"/"Thinking" (or a stall's "waiting").
    pub(super) fn desired_status(&self) -> String {
        if !self.sending {
            return "running command".to_string();
        }
        // Blocked on the user — "running rm -rf build (38s)" would imply the
        // command is already executing.
        if self.cards.permission().is_some() {
            return "waiting for your approval".to_string();
        }
        if self.cards.ask().is_some() {
            return "waiting for your answer".to_string();
        }
        if self.cards.plan_approval().is_some() {
            return "waiting for plan approval".to_string();
        }
        // A parallel sub-agent batch owns the headline while its rows are live.
        if !self.subagent_rows.is_empty() {
            let done = self
                .subagent_rows
                .iter()
                .filter(|r| r.done.is_some())
                .count();
            let total = self.subagent_rows.len();
            return format!("running {total} sub-agents ({done}/{total} done)");
        }
        // A bridged parallel batch: count the trailing run rather than naming
        // only the newest call.
        if let Some((start, total, done)) = self.trailing_tool_batch()
            && total >= 2
        {
            let all_delegates = self.history[start..].iter().all(|m| {
                super::render::canonical_tool_name(&decode_tool_call(&m.content).0) == "subagent"
            });
            let noun = if all_delegates {
                "sub-agents"
            } else {
                "parallel steps"
            };
            return format!("running {total} {noun} ({done}/{total} done)");
        }
        if let Some(action) = self.current_action_label() {
            return action;
        }
        // Streaming/retrying → "Working", else "Thinking" — unless silent long
        // enough to look like a stall.
        const STALL_AFTER: std::time::Duration = std::time::Duration::from_secs(10);
        if let Some(last) = self.last_stream_activity
            && last.elapsed() >= STALL_AFTER
        {
            return "waiting".to_string();
        }
        if self.retrying || !self.pending_response.is_empty() || !self.incoming_buffer.is_empty() {
            return "Working".to_string();
        }
        "Thinking".to_string()
    }

    /// Advance the throttled status label (once per loop iteration): adopt the
    /// new label only after the current one has shown for `STATUS_MIN_DURATION`,
    /// so fast steps don't flicker. Called per frame.
    pub(super) fn tick_status_throttle(&mut self) {
        if !self.sending && self.local_command.is_none() {
            self.render_cache.status_display = None;
            return;
        }
        let desired = self.desired_status();
        match &self.render_cache.status_display {
            // Unchanged — keep the original timestamp so it can still age out.
            Some((label, _)) if *label == desired => {}
            Some((_, since)) if since.elapsed() < STATUS_MIN_DURATION => {}
            _ => self.render_cache.status_display = Some((desired, Instant::now())),
        }
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
        // No accent bar — the status line is chrome, not a message.
        bars.push(None);
        for row in self.subagent_status_rows() {
            lines.push(row);
            bars.push(None);
        }
        for row in self.tool_output_tail_rows(self.transcript_width) {
            lines.push(row);
            bars.push(None);
        }
    }

    /// Live parallel-batch rows, styled like the spinner they sit under. Empty
    /// when idle so an interrupted batch can't leave ghost rows. Without sink
    /// rows (a bridged engine), a trailing parallel run derives its rows from
    /// the history entries instead.
    fn subagent_status_rows(&self) -> Vec<StyledLine> {
        if !self.sending {
            return Vec::new();
        }
        let style = Style::default().fg(MUTED()).add_modifier(Modifier::ITALIC);
        if !self.subagent_rows.is_empty() {
            return self
                .subagent_rows
                .iter()
                .map(|row| line_plain(super::render::subagent_row_text(row), style))
                .collect();
        }
        let Some((start, total, _)) = self.trailing_tool_batch() else {
            return Vec::new();
        };
        if total < 2 {
            return Vec::new();
        }
        let cwd = if self.real_cwd.is_empty() {
            self.cwd.as_str()
        } else {
            self.real_cwd.as_str()
        };
        self.history[start..]
            .iter()
            .map(|m| {
                let (name, args) = decode_tool_call(&m.content);
                let outcome = decode_tool_outcome(&m.content);
                line_plain(
                    super::render::parallel_call_row_text(&name, &args, outcome, cwd),
                    style,
                )
            })
            .collect()
    }

    /// Live `run_bash` tail rows under the spinner; empty when idle so an
    /// interrupted turn can't leave ghost output. The view is bottom-pinned
    /// while streaming, so the block height must stay frame-stable: completed
    /// lines and the in-flight partial share one window, and each row is
    /// clamped to the live width so it never wraps.
    fn tool_output_tail_rows(&self, text_width: u16) -> Vec<StyledLine> {
        if !self.sending {
            return Vec::new();
        }
        let style = Style::default().fg(MUTED());
        let max_cols = usize::from(text_width)
            .saturating_sub(super::render::TOOL_TAIL_INDENT_COLS)
            .min(super::render::TOOL_TAIL_MAX_COLS);
        let partial = self.tool_output_partial.trim_end();
        let mut rows: Vec<&str> = self.tool_output_tail.iter().map(String::as_str).collect();
        if !partial.trim().is_empty() {
            rows.push(partial);
        }
        let skip = rows.len().saturating_sub(super::render::STREAM_TAIL_LINES);
        rows.into_iter()
            .skip(skip)
            .map(|line| line_plain(super::render::tool_tail_row_text(line, max_cols), style))
            .collect()
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
        // The in-flight card hide depends on `sending`, so a flip must rebuild.
        self.sending.hash(&mut hasher);
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
            .render_cache
            .transcript
            .as_ref()
            .is_some_and(|cache| cache.fp == fp && cache.area_width == area_width);
        if fresh {
            return;
        }
        // New images referenced by this rebuild start their (async) preview
        // prep here; completion bumps `transcript_revision`, re-entering once.
        self.queue_missing_previews();
        let body = self.build_transcript_history_body(table_layout_width(area_width));
        let plain_width = table_layout_width(area_width).max(1);
        let plain_prepass = wrap_plain_lines(&body.plain_lines, plain_width).len();
        self.render_cache.transcript = Some(TranscriptCache {
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
            .render_cache
            .transcript
            .as_mut()
            .expect("ensure_transcript_cache runs before ensure_transcript_wrap");
        if cache.wrapped.is_some() && cache.styled_width == text_width {
            return;
        }
        let wrapped = wrap_transcript(&cache.body.lines, &cache.body.bar_colors, text_width);
        cache.styled_width = text_width;
        cache.wrapped = Some(wrapped);
    }

    /// Cheap O(1) fingerprint of every volatile-tail input EXCEPT the streamed
    /// reply's length — the reply is handled by the settled/live split, so a
    /// streamed token re-renders only the live remainder while any change here
    /// resets all sections. The spinner is excluded (wrapped fresh per frame),
    /// so a pure animation tick hits the cache.
    pub(super) fn volatile_tail_fp(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.is_transcript_empty().hash(&mut hasher);
        // The empty↔non-empty flip changes the head's shape; growth is NOT hashed.
        self.pending_response.is_empty().hash(&mut hasher);
        // A /config thinking toggle mid-turn must invalidate too.
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
        // History isn't hashed here, and the notice's visibility depends on it.
        self.notice_repeats_last_error().hash(&mut hasher);
        hasher.finish()
    }

    /// The tail lines BEFORE the streamed reply's content: the outer spacing
    /// blank and the live reasoning window. Mirrors `volatile_tail_blocks`.
    fn build_tail_head(&self, width: u16) -> TailSection {
        let substantive =
            self.thinking_enabled && reasoning_is_substantive(&self.pending_reasoning);
        let answer_started = !self.pending_response.is_empty();
        if self.is_transcript_empty() || (!substantive && !answer_started) {
            return TailSection::empty();
        }
        let mut lines: Vec<StyledLine> = vec![blank_line()];
        let mut bars: Vec<Option<Color>> = vec![None];
        if substantive {
            let mut block = Vec::new();
            render_reasoning_window(
                &mut block,
                &self.pending_reasoning,
                width.saturating_sub(SUB_BLOCK_INDENT),
            );
            push_block(&mut lines, &mut bars, indent_sub_block(block), None);
            if answer_started {
                lines.push(blank_line());
                bars.push(None);
            }
        }
        TailSection::new(lines, bars)
    }

    /// The live remainder of the tail: the reply suffix past the settled
    /// boundary, then the `!cmd` preview and notice. Mirrors `volatile_tail_blocks`.
    fn build_tail_live(
        &self,
        width: u16,
        settled_src: usize,
        settled_marked: bool,
        prev_ends_blank: bool,
    ) -> TailSection {
        let mut lines: Vec<StyledLine> = Vec::new();
        let mut bars: Vec<Option<Color>> = Vec::new();
        if self.is_transcript_empty() {
            return TailSection::empty();
        }
        if !self.pending_response.is_empty() {
            let (part, _) = render_reply_part(
                &self.pending_response[settled_src..],
                width,
                settled_marked,
                settled_src == 0,
                false,
                prev_ends_blank,
            );
            // `push_block`'s trailing-blank trim is correct here — this is the
            // reply's true end; settled chunks keep theirs (interior).
            push_block(&mut lines, &mut bars, part, Some(ACCENT()));
        }
        self.push_live_command_block(&mut lines, &mut bars);
        self.push_notice_block(&mut lines, &mut bars);
        TailSection::new(lines, bars)
    }

    /// Renders the volatile tail incrementally: a non-reply input (or width)
    /// change resets all sections; a grown reply renders only the newly settled
    /// chunk and the live suffix; an unchanged tail returns untouched.
    fn ensure_volatile_tail(&mut self, render_width: u16) {
        let fp = self.volatile_tail_fp();
        let reply_len = self.pending_response.len();
        if let Some(cache) = self.render_cache.volatile_tail.as_ref()
            && cache.fp == fp
            && cache.render_width == render_width
            && reply_len >= cache.reply_len
        {
            if reply_len == cache.reply_len {
                return;
            }
        } else {
            let head = self.build_tail_head(render_width);
            self.render_cache.volatile_tail = Some(VolatileTailCache {
                fp,
                render_width,
                reply_len: 0,
                settled_src: 0,
                settled_marked: false,
                head,
                settled: Vec::new(),
                live: TailSection::empty(),
                plain_width: 0,
                styled_width: 0,
            });
        }
        let ends_blank = |section: &TailSection| {
            section
                .lines
                .last()
                .is_some_and(|l| l.plain.trim().is_empty())
        };
        let (settled_src, settled_marked, prev_ends_blank) = {
            let cache = self.render_cache.volatile_tail.as_ref().unwrap();
            (
                cache.settled_src,
                cache.settled_marked,
                cache.settled.last().is_some_and(&ends_blank),
            )
        };
        let boundary = settled_reply_boundary(&self.pending_response, settled_src);
        let advanced = (boundary > settled_src).then(|| {
            let (chunk, marked) = render_reply_part(
                &self.pending_response[settled_src..boundary],
                render_width,
                settled_marked,
                settled_src == 0,
                true,
                prev_ends_blank,
            );
            // No trailing trim: the chunk's separator blank is interior.
            let chunk_bars = vec![Some(ACCENT()); chunk.len()];
            (TailSection::new(chunk, chunk_bars), marked)
        });
        let (new_marked, live_prev_blank) = match &advanced {
            Some((section, marked)) => (*marked, ends_blank(section)),
            None => (settled_marked, prev_ends_blank),
        };
        let live = self.build_tail_live(render_width, boundary, new_marked, live_prev_blank);
        let cache = self.render_cache.volatile_tail.as_mut().unwrap();
        if let Some((section, marked)) = advanced {
            cache.settled.push(section);
            cache.settled_src = boundary;
            cache.settled_marked = marked;
        }
        cache.live = live;
        cache.reply_len = reply_len;
    }

    /// Char-wrapped row count of the tail at `plain_width` for the pane-height
    /// prepass, memoized per section. Must run after [`ensure_volatile_tail`].
    fn volatile_tail_prepass(&mut self, plain_width: u16) -> usize {
        let cache = self
            .render_cache
            .volatile_tail
            .as_mut()
            .expect("ensure_volatile_tail runs before volatile_tail_prepass");
        if cache.plain_width != plain_width {
            cache.plain_width = plain_width;
            for section in cache.sections_mut() {
                section.prepass = None;
            }
        }
        let mut total = 0usize;
        for section in cache.sections_mut() {
            total += *section.prepass.get_or_insert_with(|| {
                if section.lines.is_empty() {
                    0
                } else {
                    let plain: Vec<String> =
                        section.lines.iter().map(|l| l.plain.clone()).collect();
                    wrap_plain_lines(&plain, plain_width).len()
                }
            });
        }
        total
    }

    /// Word-wraps only the sections that changed (all of them after a width
    /// change); empty sections stay `None` so they contribute no rows. Must run
    /// after [`ensure_volatile_tail`].
    fn ensure_volatile_tail_wrap(&mut self, text_width: u16) {
        let cache = self
            .render_cache
            .volatile_tail
            .as_mut()
            .expect("ensure_volatile_tail runs before ensure_volatile_tail_wrap");
        if cache.styled_width != text_width {
            cache.styled_width = text_width;
            for section in cache.sections_mut() {
                section.wrapped = None;
            }
        }
        for section in cache.sections_mut() {
            if section.wrapped.is_none() && !section.lines.is_empty() {
                section.wrapped = Some(wrap_transcript(&section.lines, &section.bars, text_width));
            }
        }
    }

    /// The tail sections' cached wraps in display order; valid after
    /// [`ensure_volatile_tail_wrap`].
    fn volatile_tail_parts(&self) -> impl Iterator<Item = &WrappedTranscript> {
        self.render_cache
            .volatile_tail
            .iter()
            .flat_map(|cache| cache.sections())
            .filter_map(|section| section.wrapped.as_ref())
    }

    /// The spinner block (leading blank + status + live sub-agent/tool rows),
    /// mirroring `append_spinner_status`. It animates, so this is the one wrap
    /// paid per frame — a handful of lines.
    fn wrapped_spinner_block(
        &self,
        spinner: &StyledLine,
        base_nonempty: bool,
        text_width: u16,
    ) -> WrappedTranscript {
        let mut tail: Vec<StyledLine> = Vec::new();
        let mut tail_bars: Vec<Option<Color>> = Vec::new();
        if base_nonempty {
            tail.push(blank_line());
            tail_bars.push(None);
        }
        tail.push(spinner.clone());
        tail_bars.push(None);
        for row in self.subagent_status_rows() {
            tail.push(row);
            tail_bars.push(None);
        }
        for row in self.tool_output_tail_rows(text_width) {
            tail.push(row);
            tail_bars.push(None);
        }
        wrap_transcript(&tail, &tail_bars, text_width)
    }

    pub(super) fn transcript_intro_lines(&self, width: u16) -> Vec<String> {
        // Plain mirror of the banner for height reservation; derived from the
        // styled builder to stay in lockstep with `render_empty_state`.
        brand_wordmark_lines(width)
            .into_iter()
            .map(|line| line.plain)
            .collect()
    }

    /// Records a non-picker overlay's rects for input routing: the inner
    /// content confines the screen selection; the full box is the click-away
    /// dismiss boundary.
    fn set_overlay_regions(&mut self, area: Rect) {
        self.render_cache.screen_region = Some(overlay_content_rect(area));
        self.render_cache.overlay_hitbox = Some(area);
    }

    pub(super) fn render(&mut self, frame: &mut Frame<'_>) {
        self.tick_selection_flash();
        self.refresh_git_branch();
        self.check_live_share_health();
        let outer = frame.area();
        if let Some(canvas) = palette().canvas {
            frame.render_widget(Block::default().style(Style::default().bg(canvas)), outer);
        }
        self.picker_hitbox = None;
        self.scrollbar_hit = None;
        self.transcript_hitbox = None;
        self.render_cache.screen_region = None;
        self.overlay_detail_area = None;
        self.render_cache.overlay_hitbox = None;
        let composer_area = self.render_main(frame, outer);
        // Surfaces drawn over the transcript this frame, for image suppression.
        let mut covering: Vec<Rect> = Vec::new();
        if let Some(menu) = self.visible_command_menu() {
            let (area, placement) = command_menu_area(
                composer_area,
                outer,
                menu.entries.len(),
                self.command_menu.placement,
            );
            self.command_menu.placement = Some(placement);
            self.render_command_menu(frame, area, &menu);
            covering.push(area);
        }
        let body = outer;

        match self.overlay.clone() {
            Overlay::Picker(picker) => {
                // Only the session picker gets the split (preview) layout.
                let (area, split) = match picker.kind {
                    PickerKind::Session => split_overlay_area(body, 86, 80, 68, 72),
                    _ => (centered_rect(68, 72, body), false),
                };
                let out = self.render_picker(frame, area, &picker, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Picker(p) = &mut self.overlay {
                    if let Some(start) = out.list_scroll {
                        p.scroll_top = start;
                        p.scroll_selected = p.selected;
                    }
                    if let Some(clamped) = out.detail_scroll {
                        p.preview_scroll = clamped;
                        p.preview_scroll_for = out.scroll_for;
                    }
                }
            }
            Overlay::Help { scroll } => {
                let area = centered_rect(72, 88, body);
                self.set_overlay_regions(area);
                let (clamped, bar) = self.render_help_overlay(frame, area, scroll);
                self.scrollbar_hit = bar;
                if let Overlay::Help { scroll } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Context { report, scroll } => {
                let area = centered_rect(72, 88, body);
                self.set_overlay_regions(area);
                let (clamped, bar) = self.render_context_overlay(frame, area, &report, scroll);
                self.scrollbar_hit = bar;
                if let Overlay::Context { scroll, .. } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Session { scroll } => {
                let area = centered_rect(64, 60, body);
                self.set_overlay_regions(area);
                let (clamped, bar) = self.render_session_overlay(frame, area, scroll);
                self.scrollbar_hit = bar;
                if let Overlay::Session { scroll } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Btw { scroll, follow } => {
                let (area, lines) = self.btw_overlay_layout(body);
                self.set_overlay_regions(area);
                let (clamped, bar) = self.render_btw_overlay(frame, area, lines, scroll, follow);
                self.scrollbar_hit = bar;
                if let Overlay::Btw { scroll, .. } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Share { scroll } => {
                let area = centered_rect_fixed(64, 9, body);
                self.set_overlay_regions(area);
                let (clamped, bar) = self.render_share_overlay(frame, area, scroll);
                self.scrollbar_hit = bar;
                if let Overlay::Share { scroll } = &mut self.overlay {
                    *scroll = clamped;
                }
            }
            Overlay::Skills(skills) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.set_overlay_regions(area);
                let out = self.render_skills_overlay(frame, area, &skills, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Skills(s) = &mut self.overlay {
                    if let Some(l) = out.list_scroll {
                        s.list_scroll = l;
                        s.scroll_selected = s.selected;
                    }
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    // Canonicalize a drill-in that a resize carried into split mode.
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::Agents(agents) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.set_overlay_regions(area);
                let out = self.render_agents_overlay(frame, area, &agents, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Agents(s) = &mut self.overlay {
                    if let Some(l) = out.list_scroll {
                        s.list_scroll = l;
                        s.scroll_selected = s.selected;
                    }
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::SkillInstall(pick) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.set_overlay_regions(area);
                let out = self.render_skill_install_overlay(frame, area, &pick, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                if let Overlay::SkillInstall(s) = &mut self.overlay {
                    if let Some(l) = out.list_scroll {
                        s.list_scroll = l;
                        s.scroll_selected = s.selected;
                    }
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::Mcp(mcp) => {
                let (area, split) = split_overlay_area(body, 84, 80, 64, 80);
                self.set_overlay_regions(area);
                let out = self.render_mcp_overlay(frame, area, &mcp, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                if let Overlay::Mcp(s) = &mut self.overlay {
                    if let Some(l) = out.list_scroll {
                        s.list_scroll = l;
                        s.scroll_selected = s.selected;
                    }
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                    if split {
                        s.viewing = None;
                    }
                }
            }
            Overlay::McpTools(tools) => {
                let area = centered_rect(64, 80, body);
                self.set_overlay_regions(area);
                let (list_scroll, scrollbar) = self.render_mcp_tools_overlay(frame, area, &tools);
                self.scrollbar_hit = scrollbar;
                if let Overlay::McpTools(s) = &mut self.overlay {
                    s.list_scroll = list_scroll;
                    s.scroll_selected = s.selected;
                }
            }
            Overlay::McpPaste(paste) => {
                let area = centered_rect(64, 80, body);
                self.set_overlay_regions(area);
                let (list_scroll, scrollbar) = self.render_mcp_paste_overlay(frame, area, &paste);
                self.scrollbar_hit = scrollbar;
                if let Overlay::McpPaste(s) = &mut self.overlay {
                    s.list_scroll = list_scroll;
                    s.scroll_selected = s.selected;
                }
            }
            Overlay::Config(config) => {
                let (area, split) = split_overlay_area(body, 84, 80, 72, 82);
                self.set_overlay_regions(area);
                let out = self.render_config_overlay(frame, area, &config, split);
                self.scrollbar_hit = out.scrollbar;
                self.overlay_detail_area = out.detail_area;
                self.picker_hitbox = out.list_area.map(|list_area| PickerHitbox {
                    overlay_area: area,
                    list_area,
                    row_to_filtered_index: out.list_row_index,
                    segment_hits: out.segment_hits,
                });
                if let Overlay::Config(s) = &mut self.overlay {
                    if let Some(l) = out.list_scroll {
                        s.list_scroll = l;
                        s.scroll_selected = s.selected;
                    }
                    if let Some(c) = out.detail_scroll {
                        s.detail_scroll = c;
                    }
                }
            }
            Overlay::None => {}
        }

        // Cursor-addressed images float above cells, so a surface drawn over
        // the transcript suppresses the placements it covers for the frame —
        // the post-draw flush then deletes/erases the stale ones. Placements
        // it doesn't touch stay put: the bottom-anchored command menu must not
        // blank an image at the top of the screen. Virtual placements are
        // exempt: covering cells clips them naturally.
        if !self.inline_images.caps.virtual_placement() {
            let overlay_rect = self
                .render_cache
                .overlay_hitbox
                .or_else(|| self.picker_hitbox.as_ref().map(|h| h.overlay_area));
            if !matches!(self.overlay, Overlay::None) && overlay_rect.is_none() {
                // An overlay that recorded no geometry: hide everything.
                self.inline_images.desired.clear();
            } else {
                covering.extend(overlay_rect);
                self.inline_images
                    .desired
                    .retain(|p| !covering.iter().any(|c| c.intersects(p.rect())));
            }
        }

        // Snapshot the finished screen so a drag can copy from anywhere on it,
        // then wash the full-screen selection over whatever now sits there.
        self.capture_screen_surface(frame);
        self.render_screen_selection_highlight(frame);

        self.render_toast(frame, outer);
        scrub_control_cells(frame.buffer_mut());
        self.mark_sixel_clear_cells(frame.buffer_mut());
    }

    /// Captures the rendered screen into `screen_surface` for full-screen drag-copy.
    /// Confined to `screen_region` (a modal's content rect) when one is open.
    fn capture_screen_surface(&mut self, frame: &mut Frame<'_>) {
        let full = frame.area();
        let area = self
            .render_cache
            .screen_region
            .map(|region| region.intersection(full))
            .unwrap_or(full);
        let buffer = frame.buffer_mut();
        let mut rows = Vec::with_capacity(usize::from(area.height));
        for y in area.y..area.y.saturating_add(area.height) {
            let mut row = String::with_capacity(usize::from(area.width));
            // Cells behind a wide glyph reset to " " — skip them or CJK copy gains spaces.
            let mut spacer_cells = 0u16;
            for x in area.x..area.x.saturating_add(area.width) {
                if let Some(cell) = buffer.cell((x, y)) {
                    if spacer_cells > 0 {
                        spacer_cells -= 1;
                        continue;
                    }
                    // Image placeholder cells copy as blanks, not U+10EEEE noise.
                    if cell
                        .symbol()
                        .starts_with(crate::services::terminal_graphics::PLACEHOLDER_CHAR)
                    {
                        row.push(' ');
                        continue;
                    }
                    row.push_str(cell.symbol());
                    spacer_cells = row_display_width(cell.symbol()).saturating_sub(1);
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
            SELECT_FLASH()
        } else {
            SELECT_WASH()
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

    /// Builds whichever decision card is up, in the key handlers' precedence
    /// order. `max_total` includes the card's two border rows.
    fn build_slot_card(&self, max_width: u16, max_total: u16) -> Option<SlotCard> {
        if self.cards.mcp_consent.is_some() {
            self.build_mcp_consent_card(max_width, max_total)
        } else if self.account.pending_logout.is_some() {
            self.build_logout_confirm_card(max_width)
        } else if self.cards.permission().is_some() {
            self.build_permission_card(max_width, max_total)
        } else if self.cards.ask().is_some() {
            self.build_ask_user_card(max_width, max_total)
        } else if self.cards.plan_approval().is_some() {
            self.build_plan_approval_card(max_width, max_total)
        } else if self.account.login.is_some() {
            // Last: passive status — decision cards win the slot.
            self.build_login_card(max_width)
        } else {
            None
        }
    }

    /// Card asking whether to spawn a repo's project `.mcp.json` stdio servers
    /// (the local code-execution surface). Lists each server's exact command so
    /// the risk is visible, then color-coded y/a/n keys.
    fn build_mcp_consent_card(&self, max_width: u16, max_total: u16) -> Option<SlotCard> {
        let prompt = self.cards.mcp_consent.as_ref()?;
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
        let width = (content_w as u16)
            .saturating_add(4)
            .clamp(1, max_width.max(1));
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
        ))];
        let shown = n.min(list_budget);
        for (name, cmd) in prompt.servers.iter().take(shown) {
            let room = inner_width.saturating_sub(display_width(name) + 2).max(1);
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{name}  "),
                    Style::default().fg(WARNING()).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_for_display_width(cmd, room),
                    Style::default().fg(TEXT()),
                ),
            ]));
        }
        if shown < n {
            lines.push(Line::from(Span::styled(
                format!("…and {} more", n - shown),
                Style::default().fg(MUTED()),
            )));
        }
        lines.push(Line::from(Span::styled(
            note.to_string(),
            Style::default().fg(MUTED()),
        )));
        lines.push(Line::from(""));
        lines.push(keys);

        Some(SlotCard {
            title: " mcp servers ",
            border: WARNING(),
            width,
            lines,
            scroll: None,
        })
    }

    /// The `/logout` y/n confirm card (owns the keyboard, like MCP consent).
    fn build_logout_confirm_card(&self, max_width: u16) -> Option<SlotCard> {
        let account = self.account.pending_logout.as_ref()?;
        let lines = vec![
            Line::from(vec![
                Span::styled("Unlink this device from ", Style::default().fg(TEXT())),
                Span::styled(
                    account.clone(),
                    Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
                ),
                Span::styled("?", Style::default().fg(TEXT())),
            ]),
            Line::from(Span::styled(
                "This device drops to the free tier until you sign in again.",
                Style::default().fg(MUTED()),
            )),
            Line::from(""),
            account_keys_line(&[("y", ASSISTANT(), "sign out"), ("n", ERROR(), "cancel")]),
        ];
        Some(build_account_card(
            " sign out of aivo ",
            WARNING(),
            lines,
            max_width,
        ))
    }

    /// The `/login` status card: code + URL + waiting state. Passive — it never
    /// owns the keyboard (see `handle_login_card_key`), so typing stays live.
    fn build_login_card(&self, max_width: u16) -> Option<SlotCard> {
        let card = self.account.login.as_ref()?;
        let lines = vec![
            Line::from(vec![
                Span::styled(
                    "Confirm this code in your browser:  ",
                    Style::default().fg(TEXT()),
                ),
                Span::styled(
                    card.user_code.clone(),
                    Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                card.open_url.clone(),
                Style::default().fg(LINK()),
            )),
            Line::from(Span::styled(
                "Waiting for approval…",
                Style::default().fg(MUTED()),
            )),
            Line::from(""),
            account_keys_line(&[
                ("Enter", ASSISTANT(), "open browser"),
                ("Esc", ERROR(), "cancel"),
            ]),
        ];
        Some(build_account_card(
            " sign in to aivo ",
            ACCENT(),
            lines,
            max_width,
        ))
    }

    /// Card asking the user to approve a mutating agent tool. Sits directly
    /// above the composer (where the eye and cursor already are) rather than
    /// floating mid-screen, so the decision sits right next to the input. Shows
    /// the action, a preview (diff / command / path), and color-coded y/a/n keys.
    fn build_permission_card(&self, max_width: u16, max_total: u16) -> Option<SlotCard> {
        let pending = self.cards.permission()?;
        // Flag a run_bash card: locally destructive, or a remote mutation.
        // Destructive wins the label when both apply.
        let cmd = if pending.tool == "run_bash" {
            pending.preview.as_deref()
        } else {
            None
        };
        let flag_line = if cmd.is_some_and(crate::agent::tools::bash_looks_destructive) {
            Some("⚠ looks destructive")
        } else if cmd.is_some_and(crate::agent::tools::bash_mutates_remote) {
            Some("⚠ remote side effect")
        } else {
            None
        };
        let destructive = flag_line.is_some();

        // Fixed chrome: 2 borders + heading + blank-before-keys + keys (+ the
        // destructive flag line). Whatever rows remain feed the preview block
        // (its own leading blank + the preview lines), bottom-trimmed first so
        // the keys line is always the last thing the user sees.
        let chrome = 5 + usize::from(destructive);
        let preview_budget = usize::from(max_total).saturating_sub(chrome);
        // Expand tabs up front so width sizing and the cell grid agree.
        let preview: Vec<String> = pending
            .preview
            .as_deref()
            .map(|p| {
                p.lines()
                    .take(12)
                    .map(|l| expand_tabs(l).into_owned())
                    .collect()
            })
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
        let keys = permission_keys_line(&pending.tool, pending.once_only, !self.draft.is_empty());
        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        let mut content_w = display_width(&heading).max(keys_w);
        if let Some(flag) = flag_line {
            content_w = content_w.max(display_width(flag));
        }
        for raw in preview.iter().take(preview_take) {
            content_w = content_w.max(display_width(raw));
        }
        let width = (content_w as u16)
            .saturating_add(4)
            .clamp(1, max_width.max(1));
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        let mut lines: Vec<Line<'static>> = vec![Line::from(Span::styled(
            heading,
            Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
        ))];
        if preview_take > 0 {
            lines.push(Line::from(""));
            for raw in preview.iter().take(preview_take) {
                let trimmed = raw.trim_start();
                let style = if trimmed.starts_with("+ ") {
                    Style::default().fg(ASSISTANT())
                } else if trimmed.starts_with("- ") {
                    Style::default().fg(ERROR())
                } else if pending.tool == "run_bash" {
                    Style::default().fg(if destructive { WARNING() } else { TEXT() })
                } else {
                    Style::default().fg(MUTED())
                };
                lines.push(Line::from(Span::styled(
                    truncate_for_display_width(raw, inner_width),
                    style,
                )));
            }
        }
        if let Some(flag) = flag_line {
            lines.push(Line::from(Span::styled(
                flag.to_string(),
                Style::default().fg(WARNING()).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        lines.push(keys);

        Some(SlotCard {
            title: " permission ",
            border: ACCENT(),
            width,
            lines,
            scroll: None,
        })
    }

    /// The `ask_user` card: question, numbered pick-list (`❯` = highlighted), key
    /// hint. Sits above the composer like the permission card, clamped to the
    /// slot's budget.
    fn build_ask_user_card(&self, max_width: u16, max_total: u16) -> Option<SlotCard> {
        let ask = self.cards.ask()?;
        let max_width = max_width.max(1);
        let inner_cap = usize::from(max_width.saturating_sub(4)).max(1);

        // Question wraps to at most 3 lines; options render as "N. label — desc".
        let mut q_lines = super::overlay_render_impl::wrap_chars(&ask.question, inner_cap);
        q_lines.truncate(3);
        let opt_plain: Vec<String> = ask
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| match &o.description {
                Some(d) => format!("{}. {} — {}", i + 1, o.label, d),
                None => format!("{}. {}", i + 1, o.label),
            })
            .collect();

        // Size to the widest visible line (question / option+marker / keys).
        let keys = ask_user_keys_line(ask.allow_free_text, ask.multi_select);
        let keys_w: usize = keys
            .spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();
        // Multi-select prefixes each option with a "[✓] " checkbox.
        let box_w = if ask.multi_select { 4 } else { 0 };
        let mut content_w = keys_w;
        for l in &q_lines {
            content_w = content_w.max(display_width(l));
        }
        for s in &opt_plain {
            content_w = content_w.max(display_width(s) + 2 + box_w);
        }
        let width = (content_w as u16).saturating_add(4).clamp(1, max_width);
        let inner_width = usize::from(width.saturating_sub(4)).max(1);

        // Assemble lines; trim the option list from the bottom if the card would
        // overrun the space above the composer (keys/question stay visible).
        let mut lines: Vec<Line<'static>> = Vec::new();
        for l in &q_lines {
            lines.push(Line::from(Span::styled(
                truncate_for_display_width(l, inner_width),
                Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
            )));
        }
        lines.push(Line::from(""));
        // Fixed chrome after the options: blank + keys + 2 borders.
        let chrome_after = 3usize;
        let option_budget = usize::from(max_total)
            .saturating_sub(lines.len() + chrome_after)
            .max(1);

        // Long descriptions wrap onto rows hanging under the label; the compact
        // one-row form is the fallback when the wrapped list won't fit.
        const MAX_OPT_ROWS: usize = 3;
        let mut groups: Vec<Vec<Line<'static>>> = Vec::with_capacity(ask.options.len());
        let mut compact: Vec<Line<'static>> = Vec::with_capacity(ask.options.len());
        for (i, opt) in ask.options.iter().enumerate() {
            let selected = i == ask.selected;
            let marker_style = if selected {
                Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(FAINT())
            };
            let label_style = if selected {
                Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT())
            };
            let mut prefix = vec![Span::styled(
                if selected { "❯ " } else { "  " },
                marker_style,
            )];
            if ask.multi_select {
                let checked = ask.checked.get(i).copied().unwrap_or(false);
                let box_style = if checked {
                    Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(FAINT())
                };
                prefix.push(Span::styled(
                    if checked { "[✓] " } else { "[ ] " },
                    box_style,
                ));
            }
            prefix.push(Span::styled(
                format!("{}. ", i + 1),
                Style::default().fg(MUTED()),
            ));
            let prefix_w: usize = prefix
                .iter()
                .map(|s| display_width(s.content.as_ref()))
                .sum();
            let wrap_w = inner_width.saturating_sub(prefix_w).max(1);

            let mut body = vec![Span::styled(opt.label.clone(), label_style)];
            if let Some(desc) = &opt.description {
                body.push(Span::styled(
                    format!(" — {desc}"),
                    Style::default().fg(FAINT()),
                ));
            }
            let rows = super::render::wrap_styled_line(&body, wrap_w);
            let capped = rows.len() > MAX_OPT_ROWS;
            let mut opt_lines: Vec<Line<'static>> =
                Vec::with_capacity(rows.len().min(MAX_OPT_ROWS));
            for (r, row) in rows.into_iter().take(MAX_OPT_ROWS).enumerate() {
                let mut spans = if r == 0 {
                    prefix.clone()
                } else {
                    vec![Span::raw(" ".repeat(prefix_w))]
                };
                if capped && r == MAX_OPT_ROWS - 1 {
                    // re-truncate so the ellipsis stays inside the width
                    spans.push(Span::styled(
                        truncate_for_display_width(&format!("{}…", row.plain), wrap_w),
                        Style::default().fg(FAINT()),
                    ));
                } else {
                    spans.extend(row.line.spans);
                }
                opt_lines.push(Line::from(spans));
            }
            groups.push(opt_lines);

            let label = truncate_for_display_width(&opt.label, wrap_w);
            let used = display_width(&label) + prefix_w;
            let mut spans = prefix;
            spans.push(Span::styled(label, label_style));
            if let Some(desc) = opt
                .description
                .as_deref()
                .filter(|_| used + 3 < inner_width)
            {
                let room = inner_width - used - 3;
                spans.push(Span::styled(
                    format!(" — {}", truncate_for_display_width(desc, room)),
                    Style::default().fg(FAINT()),
                ));
            }
            compact.push(Line::from(spans));
        }

        let wrapped_total: usize = groups.iter().map(Vec::len).sum();
        if wrapped_total <= option_budget {
            lines.extend(groups.into_iter().flatten());
        } else {
            let shown = ask.options.len().min(option_budget);
            lines.extend(compact.into_iter().take(shown));
            if shown < ask.options.len() {
                lines.push(Line::from(Span::styled(
                    format!("  …{} more", ask.options.len() - shown),
                    Style::default().fg(FAINT()),
                )));
            }
        }
        lines.push(Line::from(""));
        lines.push(keys);

        Some(SlotCard {
            title: " question ",
            border: ACCENT(),
            width,
            lines,
            scroll: None,
        })
    }

    /// The plan-approval card (`exit_plan_mode`): heading, the scrollable rendered
    /// plan, and the three verdicts. Sits above the composer; carries the clamped
    /// scroll for write-back.
    fn build_plan_approval_card(&self, max_width: u16, max_total: u16) -> Option<SlotCard> {
        let pending = self.cards.plan_approval()?;
        let max_width = max_width.max(1);
        let inner_width = usize::from(max_width.saturating_sub(4)).max(1);

        // heading + blank + (plan…) + blank + 3 options + blank + keys + 2 borders;
        // one more row is reserved for the "+N more" marker when the plan overflows.
        let chrome = 10u16;
        let body_budget = usize::from(max_total.saturating_sub(chrome)).max(1);
        let overflow = pending.body.len() > body_budget;
        let scroll =
            usize::from(pending.scroll).min(pending.body.len().saturating_sub(body_budget));
        let visible = pending.body.len().saturating_sub(scroll).min(body_budget);

        let mut lines: Vec<Line<'static>> = Vec::new();
        lines.push(Line::from(Span::styled(
            truncate_for_display_width("Implementation plan — ready for review", inner_width),
            Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for line in pending.body.iter().skip(scroll).take(visible) {
            lines.push(line.clone());
        }
        let remaining = pending.body.len().saturating_sub(scroll + visible);
        if remaining > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … +{remaining} more (PgUp/PgDn scroll)"),
                Style::default().fg(FAINT()),
            )));
        } else if overflow {
            lines.push(Line::from(Span::styled(
                "  … end of plan",
                Style::default().fg(FAINT()),
            )));
        }
        lines.push(Line::from(""));
        const OPTIONS: [&str; 3] = [
            "Approve — execute with auto-approve",
            "Approve — review each edit first",
            "Keep planning — type feedback below",
        ];
        for (i, opt) in OPTIONS.iter().enumerate() {
            let selected = i == pending.selected;
            let (marker_style, label_style) = if selected {
                (
                    Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
                    Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
                )
            } else {
                (Style::default().fg(FAINT()), Style::default().fg(TEXT()))
            };
            lines.push(Line::from(vec![
                Span::styled(if selected { "❯ " } else { "  " }, marker_style),
                Span::styled(format!("{}. ", i + 1), Style::default().fg(MUTED())),
                Span::styled(
                    truncate_for_display_width(opt, inner_width.saturating_sub(5)),
                    label_style,
                ),
            ]));
        }
        lines.push(Line::from(""));
        lines.push(plan_approval_keys_line());

        Some(SlotCard {
            title: " plan ",
            border: ACCENT(),
            width: max_width,
            lines,
            scroll: Some(u16::try_from(scroll).unwrap_or(u16::MAX)),
        })
    }

    pub(super) fn render_main(&mut self, frame: &mut Frame<'_>, area: Rect) -> Rect {
        // One shared left + top margin so markers don't hug the edge; hitboxes
        // derive from the inset area, so mouse mapping follows automatically.
        let area = Rect {
            x: area.x.saturating_add(APP_LEFT_MARGIN),
            y: area.y.saturating_add(APP_TOP_MARGIN),
            width: area.width.saturating_sub(APP_LEFT_MARGIN),
            height: area.height.saturating_sub(APP_TOP_MARGIN),
        };
        // The pane takes a full-height right column; everything else lays out
        // in the left column.
        let (area, preview_pane_area) = self.split_preview_pane(area);
        let composer_height = self.composer_height(area.width);
        // The footer is a single fixed row: just the status line. No hint bar, so
        // the layout never shifts up or down as turns start and finish.
        let footer_height = 1u16;
        // The pinned plan/task-list panel sits between the transcript and the
        // composer (a faint top rule + the wrapped checklist), so progress stays
        // visible instead of scrolling away under later tool calls. Sized from the
        // current plan, capped so the transcript keeps a usable minimum.
        let plan_lines = self.plan_panel_lines();
        let plan_panel_height =
            self.plan_panel_height(&plan_lines, area, composer_height, footer_height);
        // Clamp queue focus each frame — the engine or a turn-end drain may
        // have emptied the rows it selects since the last event.
        let queue_rows = self.queued_rows();
        match (&mut self.queue_focus, queue_rows.len()) {
            (focus @ Some(_), 0) => *focus = None,
            (Some(sel), n) => *sel = (*sel).min(n - 1),
            (None, _) => {}
        }
        let queue_lines = self.queued_panel_lines(&queue_rows, area.width);
        let queue_panel_height = self.queued_panel_height(
            &queue_lines,
            area,
            composer_height,
            footer_height,
            plan_panel_height,
        );
        // The active decision card gets its own slot so the transcript shrinks
        // instead of being painted over. Cap: leave CARD_MIN_TRANSCRIPT rows
        // when the screen allows, but never take the last row — a Length
        // overflow would clip the composer/footer.
        let card_avail = area.height.saturating_sub(
            composer_height + footer_height + plan_panel_height + queue_panel_height,
        );
        let card_cap = card_avail
            .saturating_sub(CARD_MIN_TRANSCRIPT)
            .max(CARD_MIN_HEIGHT.min(card_avail.saturating_sub(1)))
            .max(1);
        let card_max_width = area.width.saturating_sub(2).max(1);
        let slot_card = self.build_slot_card(card_max_width, card_cap);
        if let Some(clamped) = slot_card.as_ref().and_then(|card| card.scroll)
            && let Some(p) = self.cards.plan_approval_mut()
        {
            p.scroll = clamped;
        }
        let card_slot_height = slot_card
            .as_ref()
            .map(|card| (card.lines.len() as u16).saturating_add(2).min(card_cap))
            .unwrap_or(0);
        let max_transcript_height = area
            .height
            .saturating_sub(
                composer_height
                    + footer_height
                    + plan_panel_height
                    + queue_panel_height
                    + card_slot_height,
            )
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
        // Spinner blank + status line + any live sub-agent rows, sized the same
        // way the body's char-wrap height estimate is, so the pane height matches.
        let spinner_prepass = spinner
            .as_ref()
            .map(|line| {
                let mut plain = vec![String::new(), line.plain.clone()];
                plain.extend(self.subagent_status_rows().into_iter().map(|r| r.plain));
                wrap_plain_lines(&plain, plain_width).len()
            })
            .unwrap_or(0);
        let prepass_rows = self
            .render_cache
            .transcript
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
            .saturating_add(queue_panel_height)
            .saturating_add(card_slot_height)
            .saturating_add(composer_height)
            .saturating_add(footer_height)
            .min(area.height.max(1));

        let stack = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(stack_height), Constraint::Min(0)])
            .split(area);
        // transcript / [plan panel] / [queue panel] / [card slot] / composer /
        // footer — the plan, queue, and card rows are present only when
        // non-empty, so the indices below shift accordingly.
        let mut constraints = vec![Constraint::Length(transcript_height)];
        if plan_panel_height > 0 {
            constraints.push(Constraint::Length(plan_panel_height));
        }
        if queue_panel_height > 0 {
            constraints.push(Constraint::Length(queue_panel_height));
        }
        if card_slot_height > 0 {
            constraints.push(Constraint::Length(card_slot_height));
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
        let queue_panel_area = if queue_panel_height > 0 {
            let a = chunks[chunk_idx];
            chunk_idx += 1;
            Some(a)
        } else {
            None
        };
        let card_slot_area = if card_slot_height > 0 {
            let a = chunks[chunk_idx];
            chunk_idx += 1;
            Some(a)
        } else {
            None
        };
        let composer_outer = chunks[chunk_idx];
        let footer_outer = chunks[chunk_idx + 1];
        // The composer reserves its own blank spacing row above the divider (see
        // the composer layout below), so the transcript fills its whole area here
        // — no extra bottom padding carved from within, in overflow or otherwise.
        let transcript_view_area = transcript_area;
        let view_height = transcript_view_area.height.max(1);
        // The transcript uses its full width — no column reserved for a scrollbar.
        let transcript_content_area = transcript_view_area;
        // Inset text area: a fixed left margin (formerly the accent-bar gutter) plus
        // a right margin so wrapped prose never touches the terminal edge.
        let transcript_text_area = Rect {
            x: transcript_content_area
                .x
                .saturating_add(ACCENT_GUTTER_WIDTH),
            y: transcript_content_area.y,
            width: transcript_content_area
                .width
                .saturating_sub(ACCENT_GUTTER_WIDTH + TRANSCRIPT_RIGHT_MARGIN)
                .max(1),
            height: transcript_content_area.height,
        };
        // Word-wrap ourselves to the text width so our row model (scroll, gutter,
        // selection) exactly matches the rendered rows; we render with wrap OFF.
        // The composed transcript is never materialized — counts, hitbox, and the
        // visible window read the cached segment wraps in place, so a repaint
        // costs O(visible rows), not O(history + reply).
        self.ensure_transcript_wrap(transcript_text_area.width);
        self.ensure_volatile_tail_wrap(transcript_text_area.width);
        let body_len = self
            .render_cache
            .transcript
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
            .expect("ensure_transcript_wrap runs before composition")
            .rows
            .len();
        let tail_len: usize = self.volatile_tail_parts().map(|part| part.rows.len()).sum();
        let spinner_wrap = spinner.as_ref().map(|line| {
            self.wrapped_spinner_block(line, body_len + tail_len > 0, transcript_text_area.width)
        });
        let transcript_total_lines =
            body_len + tail_len + spinner_wrap.as_ref().map_or(0, |sw| sw.rows.len());
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
        // Display-order row segments, `Arc`-shared with the caches.
        let mut segments: Vec<std::sync::Arc<Vec<String>>> = Vec::new();
        if let Some(body) = self
            .render_cache
            .transcript
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
        {
            segments.push(std::sync::Arc::clone(&body.rows));
        }
        for part in self.volatile_tail_parts() {
            segments.push(std::sync::Arc::clone(&part.rows));
        }
        if let Some(sw) = &spinner_wrap {
            segments.push(std::sync::Arc::clone(&sw.rows));
        }
        self.transcript_hitbox = Some(TranscriptHitbox {
            area: transcript_text_area,
            first_row: self.transcript_scroll,
            segments,
        });

        self.collect_desired_inline_images(transcript_text_area);

        clear_to_canvas(frame, chunks[0]);

        if is_empty {
            // Inset by the accent gutter so the brand banner sits at the same
            // column as the transcript content does once a message arrives — without
            // this the banner jumps 2 cols right when the first message lands.
            self.render_empty_state(frame, transcript_text_area);
            self.jump_to_bottom_hit = None;
        } else {
            // Pre-wrapped above → render with wrap OFF so rendered rows match.
            // ratatui's `Paragraph` does NOT virtualize (`.scroll` still lays
            // out every line), so stitch the visible window from the cached
            // segment wraps and render at scroll 0 — O(visible rows), and the
            // row model above keeps geometry exact.
            let view_start = self.transcript_scroll.min(transcript_total_lines);
            let view_end = view_start
                .saturating_add(usize::from(transcript_text_area.height))
                .min(transcript_total_lines);
            let mut visible_lines: Vec<Line<'static>> =
                Vec::with_capacity(view_end.saturating_sub(view_start));
            {
                let body = self
                    .render_cache
                    .transcript
                    .as_ref()
                    .and_then(|cache| cache.wrapped.as_ref());
                let mut parts: Vec<&[Line<'static>]> = Vec::new();
                if let Some(w) = body {
                    parts.push(w.text.lines.as_slice());
                }
                for part in self.volatile_tail_parts() {
                    parts.push(part.text.lines.as_slice());
                }
                if let Some(w) = &spinner_wrap {
                    parts.push(w.text.lines.as_slice());
                }
                let mut skip = view_start;
                let mut take = view_end - view_start;
                for part in parts {
                    if take == 0 {
                        break;
                    }
                    if skip >= part.len() {
                        skip -= part.len();
                        continue;
                    }
                    let end = (skip + take).min(part.len());
                    visible_lines.extend(part[skip..end].iter().cloned());
                    take -= end - skip;
                    skip = 0;
                }
            }
            let visible_text = Text::from(visible_lines);
            let transcript_widget = Paragraph::new(visible_text).style(Style::default().fg(TEXT()));
            frame.render_widget(transcript_widget, transcript_text_area);
            self.render_transcript_selection_highlight(frame, transcript_text_area);
            // Clickable jump-to-bottom pill (like Ctrl+End), only while scrolled up.
            self.jump_to_bottom_hit = if self.transcript_scroll < max_scroll {
                render_jump_to_bottom(frame, transcript_view_area)
            } else {
                None
            };
        }

        // After `collect_desired_inline_images` (it clears `desired`), so the
        // pane's placement survives into the post-draw flush.
        self.preview_close_hits.clear();
        if let Some(pane_area) = preview_pane_area {
            self.render_preview_pane(frame, pane_area);
        }

        if let Some(plan_panel_area) = plan_panel_area {
            self.render_plan_panel(frame, plan_panel_area, &plan_lines);
        }

        if let Some(queue_panel_area) = queue_panel_area {
            self.render_queued_panel(frame, queue_panel_area, &queue_lines);
        }

        if let (Some(slot), Some(card)) = (card_slot_area, slot_card) {
            render_slot_card(frame, slot, card);
        }

        // Leave one breathing row, then enclose the input in a rounded box.
        let composer_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(1)])
            .split(composer_outer);

        clear_to_canvas(frame, composer_chunks[0]);
        let composer_box_area = composer_chunks[1];
        clear_to_canvas(frame, composer_box_area);
        let composer_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(self.composer_rule_style());
        let composer_area = composer_block.inner(composer_box_area);
        frame.render_widget(composer_block, composer_box_area);

        // Paint the live mode/history badges into the top border while leaving
        // the rounded corner cells intact.
        let title_area = Rect {
            x: composer_box_area.x.saturating_add(1),
            y: composer_box_area.y,
            width: composer_box_area.width.saturating_sub(2),
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(self.composer_rule_line(title_area.width.max(1))),
            title_area,
        );

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

        self.render_footer(frame, footer_outer);
        composer_area
    }

    /// Hue shared by the composer's rounded border: amber shell, lime plan, blue
    /// ask, and a quiet ordinary input.
    pub(super) fn composer_rule_style(&self) -> Style {
        if self.draft_is_shell_command() {
            Style::default().fg(SHELL())
        } else if self.plan_mode || self.cursor_plan_mode {
            Style::default().fg(ACCENT())
        } else if self.ask_mode {
            Style::default().fg(INFO())
        } else {
            Style::default().fg(FAINT())
        }
    }

    /// The divider rule above the composer, always carrying the auto-approve
    /// badge inset near the right end so the mode and its toggle key stay
    /// discoverable (the hint bar drops right-hand items when narrow).
    pub(super) fn composer_rule_line(&self, width: u16) -> Line<'static> {
        let width = usize::from(width);
        let plan_mode = self.plan_mode || self.cursor_plan_mode;
        let rule_style = self.composer_rule_style();
        // The mode badge — one slot, since the modes are exclusive.
        let (badge, badge_style) = if plan_mode {
            (
                "◇ plan",
                Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
            )
        } else if self.ask_mode {
            (
                "◆ ask",
                Style::default().fg(INFO()).add_modifier(Modifier::BOLD),
            )
        } else if self.agent_auto_approve {
            ("↯ auto-approve", Style::default().fg(WARNING()))
        } else {
            ("default", Style::default().fg(MUTED()))
        };
        const CYCLE_HINT: &str = " (Shift+Tab)";
        // Left title: `History {pos}/{total}` while recalling input (counts down
        // as you scroll back), else a live `/goal` step indicator so an
        // unattended loop stays visible. Never both in one frame.
        let (left_text, left_style, left_lead) = if let Some(index) = self.draft_history_index {
            (
                format!(" History {}/{} ", index + 1, self.draft_history.len()),
                Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
                2usize,
            )
        } else if let Some(goal) = self.goal_mode.as_ref() {
            (
                format!(" ◎ goal {}/{} ", goal.iteration, goal.max),
                Style::default().fg(ACCENT()),
                0usize,
            )
        } else {
            (String::new(), Style::default(), 0usize)
        };
        // Left-cluster badge for running jobs.
        let jobs_running = self.jobs_running;
        let jobs_text = if jobs_running > 0 {
            let s = if jobs_running == 1 { "" } else { "s" };
            format!(" ✦ {jobs_running} job{s} ")
        } else {
            String::new()
        };
        let jobs_w = display_width(&jobs_text);
        let trailing = 2usize;
        let left_w = if left_text.is_empty() {
            0
        } else {
            left_lead + display_width(&left_text)
        };
        // The keybinding hint drops first on a narrow terminal — the badge alone
        // still names the mode — and again when the left cluster crowds it.
        let mut hint = if width >= 60 { CYCLE_HINT } else { "" };
        let badge_w = |hint: &str| display_width(badge) + display_width(hint) + 2;
        if !hint.is_empty() && width <= left_w + jobs_w + badge_w(hint) + trailing + 2 {
            hint = "";
        }
        if width <= left_w + jobs_w + badge_w(hint) + trailing + 2 {
            // Too narrow to inset it all — keep just the mode badge.
            return Line::from(Span::styled(badge.to_string(), badge_style));
        }
        let fill = width - left_w - jobs_w - badge_w(hint) - trailing;
        let mut spans = Vec::with_capacity(8);
        if !left_text.is_empty() {
            if left_lead > 0 {
                spans.push(Span::styled("─".repeat(left_lead), rule_style));
            }
            spans.push(Span::styled(left_text, left_style));
        }
        if !jobs_text.is_empty() {
            spans.push(Span::styled(jobs_text, Style::default().fg(MUTED())));
        }
        spans.push(Span::styled("─".repeat(fill), rule_style));
        spans.push(Span::styled(format!(" {badge}"), badge_style));
        if !hint.is_empty() {
            spans.push(Span::styled(hint.to_string(), Style::default().fg(FAINT())));
        }
        spans.push(Span::raw(" "));
        spans.push(Span::styled("─".repeat(trailing), rule_style));
        Line::from(spans)
    }

    /// The pinned plan/task-list panel's content lines (the `Tasks N/M done`
    /// header plus one line per step), or empty when there's no plan or the plan
    /// is fully done. Built fresh each frame — it's small, and the plan changes
    /// rarely.
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
        // A finished plan is hidden (clutter, and reads as false "done" on error).
        if plan_all_completed(content) {
            return Vec::new();
        }
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
        // + blank, top rule, blank — the rule breathes on both sides.
        content_rows.clamp(1, max_body).saturating_add(3)
    }

    /// Paint the pinned plan panel: a faint top rule (fencing it off from the
    /// transcript, mirroring the composer's divider) over the wrapped checklist.
    /// When the plan overflows the panel, scroll so the active (`in_progress`)
    /// step stays on screen.
    fn render_plan_panel(&self, frame: &mut Frame<'_>, area: Rect, lines: &[StyledLine]) {
        if area.height == 0 || lines.is_empty() {
            return;
        }
        clear_to_canvas(frame, area);
        // A blank row above and below the rule so it doesn't crowd the transcript
        // or the panel header.
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "─".repeat(usize::from(area.width.max(1))),
                Style::default().fg(FAINT()),
            ))),
            Rect {
                x: area.x,
                y: area
                    .y
                    .saturating_add(1)
                    .min(area.bottom().saturating_sub(1)),
                width: area.width,
                height: 1,
            },
        );
        let body = Rect {
            x: area.x.saturating_add(1),
            y: area.y.saturating_add(3),
            width: area.width.saturating_sub(2).max(1),
            height: area.height.saturating_sub(3),
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

    /// Queued-input panel lines: a blank spacer, one line per item (windowed
    /// around the selection with `… +k` indicators), a hint line while focused.
    fn queued_panel_lines(&self, rows: &[QueuedRow], width: u16) -> Vec<Line<'static>> {
        if rows.is_empty() {
            return Vec::new();
        }
        let selected = self.queue_focus;
        let start = selected
            .map(|sel| sel.saturating_sub(QUEUE_PANEL_MAX_ROWS - 1))
            .unwrap_or(0)
            .min(rows.len().saturating_sub(QUEUE_PANEL_MAX_ROWS));
        let end = (start + QUEUE_PANEL_MAX_ROWS).min(rows.len());
        let mut lines = vec![Line::default()];
        if start > 0 {
            lines.push(Line::from(Span::styled(
                format!("  … +{start} earlier"),
                Style::default().fg(FAINT()),
            )));
        }
        for (i, row) in rows.iter().enumerate().take(end).skip(start) {
            let is_selected = selected == Some(i);
            let marker = if is_selected { "▸ " } else { "  " };
            let prefix = match row.segment {
                QueueSegment::Steering => "» ",
                QueueSegment::Command => "",
                QueueSegment::Message => "· ",
            };
            let room = usize::from(width).saturating_sub(3 + prefix.chars().count());
            let (marker_style, text_style) = if is_selected {
                (Style::default().fg(ACCENT()), Style::default().fg(TEXT()))
            } else {
                (Style::default().fg(MUTED()), Style::default().fg(MUTED()))
            };
            lines.push(Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(
                    format!("{prefix}{}", truncate_for_display_width(&row.display, room)),
                    text_style,
                ),
            ]));
        }
        if end < rows.len() {
            lines.push(Line::from(Span::styled(
                format!("  … +{} more", rows.len() - end),
                Style::default().fg(FAINT()),
            )));
        }
        if selected.is_some() {
            lines.push(Line::from(Span::styled(
                "  Enter edit · Ctrl+D remove · Alt+↑↓ move · Esc back",
                Style::default().fg(FAINT()),
            )));
        }
        lines
    }

    /// Panel height, clamped so the transcript keeps a usable minimum.
    fn queued_panel_height(
        &self,
        lines: &[Line<'static>],
        area: Rect,
        composer_height: u16,
        footer_height: u16,
        plan_panel_height: u16,
    ) -> u16 {
        if lines.is_empty() {
            return 0;
        }
        let reserved = composer_height
            .saturating_add(footer_height)
            .saturating_add(plan_panel_height)
            .saturating_add(PLAN_PANEL_MIN_TRANSCRIPT);
        (lines.len() as u16).min(area.height.saturating_sub(reserved))
    }

    fn render_queued_panel(&self, frame: &mut Frame<'_>, area: Rect, lines: &[Line<'static>]) {
        if area.height == 0 || lines.is_empty() {
            return;
        }
        clear_to_canvas(frame, area);
        frame.render_widget(Paragraph::new(Text::from(lines.to_vec())), area);
    }

    pub(super) fn empty_state_height(&self, width: u16) -> u16 {
        let content_width = width.saturating_sub(1).max(1);
        let mut height = if let Some(loading) = &self.loading_resume {
            let mut rows = vec![
                "Loading saved session…".to_string(),
                loading.preview.title.clone(),
                plain_text_from_spans(&resume_metadata_spans(
                    &loading.preview,
                    content_width.saturating_sub(1).max(1),
                )),
                self.display_cwd().to_string(),
            ];
            rows.extend(self.notice_plain_lines(content_width));
            rows.extend(self.spinner_status_plain_lines(content_width));
            wrap_plain_lines(&rows, content_width).len() as u16
        } else {
            // Measure at the column `render_empty_state` draws into; any wider
            // undercounts wrapped rows and clips the tip on narrow terminals.
            let empty_content_width = width
                .saturating_sub(ACCENT_GUTTER_WIDTH + TRANSCRIPT_RIGHT_MARGIN + HEADER_LEFT_INSET)
                .max(1);
            let mut rows = self.transcript_intro_lines(empty_content_width);
            // Reserve the chip + tip height too, matching `render_empty_state`.
            rows.extend(self.welcome_status_lines().into_iter().map(|sl| sl.plain));
            rows.extend(self.notice_plain_lines(empty_content_width));
            rows.extend(self.spinner_status_plain_lines(empty_content_width));
            wrap_plain_lines(&rows, empty_content_width).len() as u16
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

    /// Height-side twin of the spinner line `render_empty_state` appends.
    fn spinner_status_plain_lines(&self, width: u16) -> Vec<String> {
        self.spinner_status_line()
            .map(|line| {
                let mut lines = vec![String::new()];
                lines.extend(wrap_plain_lines(&[line.plain], width));
                lines
            })
            .unwrap_or_default()
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
            SELECT_FLASH()
        } else {
            SELECT_WASH()
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
            let text_width = hitbox.row(row).map(row_display_width).unwrap_or(0);
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
        let (toast_area, fade_window) = match toast.anchor {
            // Last transcript row, above the composer rule, near where the user acted.
            ToastAnchor::Corner => (
                Rect {
                    x: area
                        .x
                        .saturating_add(area.width.saturating_sub(toast_width)),
                    y: self
                        .composer_text_area
                        .map(|c| c.y.saturating_sub(2))
                        .unwrap_or(area.y),
                    width: toast_width,
                    height: 1,
                },
                TOAST_DURATION - TOAST_FADE_AFTER,
            ),
            // Mid-transcript chip: 3 rows where there's room, 1 otherwise.
            ToastAnchor::Center => {
                let region = self
                    .composer_text_area
                    .map(|c| c.y.saturating_sub(1))
                    .unwrap_or(area.bottom())
                    .saturating_sub(area.y);
                let height = if region >= 5 { 3 } else { 1 };
                (
                    Rect {
                        x: area
                            .x
                            .saturating_add(area.width.saturating_sub(toast_width) / 2),
                        y: area.y.saturating_add(region.saturating_sub(height) / 2),
                        width: toast_width,
                        height,
                    },
                    CENTER_TOAST_DURATION - CENTER_TOAST_FADE_AFTER,
                )
            }
        };
        // Cells have no alpha: fading = blending the text into the pill bg.
        let remaining = toast.expires_at.duration_since(now).as_secs_f32();
        let t = 1.0 - (remaining / fade_window.as_secs_f32()).min(1.0);
        let color = fade_color(ACCENT(), palette().toast_bg, t);
        let mut lines = vec![Line::default(); usize::from(toast_area.height / 2)];
        lines.push(
            Line::from(Span::styled(&toast.text, Style::default().fg(color)))
                .alignment(ratatui::layout::Alignment::Center),
        );
        clear_to_canvas(frame, toast_area);
        frame.render_widget(
            Paragraph::new(lines).style(Style::default().bg(palette().toast_bg)),
            toast_area,
        );
    }

    pub(super) fn render_empty_state(&self, frame: &mut Frame<'_>, area: Rect) {
        // `area` is the gutter-inset transcript text area; adding the header
        // inset lands on the same column as the intro banner in
        // `build_transcript_history_body`, so the banner doesn't jump once a
        // message lands.
        let content_area = Rect {
            x: area.x.saturating_add(HEADER_LEFT_INSET),
            y: area.y.saturating_add(EMPTY_STATE_TOP_GAP),
            width: area.width.saturating_sub(HEADER_LEFT_INSET),
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
                        Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " saved session…",
                        Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::styled(
                    loading.preview.title.clone(),
                    Style::default().fg(TEXT()),
                )),
                Line::from(resume_metadata_spans(
                    &loading.preview,
                    area.width.max(1).saturating_sub(2),
                )),
                Line::from(Span::styled(
                    self.display_cwd(),
                    Style::default().fg(FAINT()),
                )),
            ]
        } else {
            // Pick full/narrow at the banner's real render width; the wider
            // `area.width` would let the wordmark wrap and push the tip offscreen.
            brand_wordmark_lines(content_area.width)
                .into_iter()
                .map(|sl| sl.line)
                .collect()
        };

        let mut lines = lines;
        // Chip + tip on the fresh welcome only, never the resume-loading screen.
        if self.loading_resume.is_none() {
            lines.extend(self.welcome_status_lines().into_iter().map(|sl| sl.line));
        }
        if let Some(spans) = notice_spans(self.notice.as_ref()) {
            lines.push(Line::from(""));
            lines.push(Line::from(spans));
        }
        // The empty state replaces the transcript's spinner tail, so a fetch on
        // a fresh chat with no overlay open must narrate here.
        if let Some(spinner) = self.spinner_status_line() {
            lines.push(Line::from(""));
            lines.push(spinner.line);
        }

        frame.render_widget(
            Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false }),
            content_area,
        );
    }

    /// The `N skills · M MCP` capability chip, or `None` when neither is configured.
    pub(super) fn welcome_capabilities_label(&self) -> Option<String> {
        let skills = self.skill_commands.len();
        let mcp = self.mcp_configured_count;
        let mut parts: Vec<String> = Vec::new();
        if skills > 0 {
            let noun = if skills == 1 { "skill" } else { "skills" };
            parts.push(format!("{skills} {noun}"));
        }
        if mcp > 0 {
            parts.push(format!("{mcp} MCP"));
        }
        parts.extend(self.inline_images.caps.protocol.label().map(str::to_string));
        (!parts.is_empty()).then(|| parts.join(" · "))
    }

    /// Blank spacer, optional capability chip, then the rotating tip. Shared by
    /// the empty state, the transcript intro, and `empty_state_height` (kept in
    /// lockstep, measuring the same lines).
    pub(super) fn welcome_status_lines(&self) -> Vec<StyledLine> {
        let mut lines = vec![blank_line()];
        // A fork of another agent's session carries a provenance line, so it's clear
        // this thread began in Claude/Codex/Pi (its id also stays `import-<cli>-…`).
        if let Some(source) = crate::services::session_import::import_source_label(&self.session_id)
        {
            let fidelity = self
                .import_fidelity
                .as_ref()
                .map(|f| format!(" · fidelity {}", f.tier().label()))
                .unwrap_or_default();
            lines.push(line_with_plain(vec![
                Span::styled("↳ ", Style::default().fg(ACCENT())),
                Span::styled(
                    format!("Forked from a {source} session{fidelity} · /session for details"),
                    Style::default().fg(MUTED()),
                ),
            ]));
        }
        if let Some(chip) = self.welcome_capabilities_label() {
            lines.push(line_with_plain(vec![
                Span::styled("◈  ", Style::default().fg(ACCENT())),
                Span::styled(chip, Style::default().fg(MUTED())),
            ]));
        }
        let tip = self.current_welcome_tip();
        lines.push(line_with_plain(vec![
            // MUTED hint (up from FAINT) so the tip reads on dim terminals.
            Span::styled("✻ Tip  ", Style::default().fg(ACCENT())),
            Span::styled(tip.to_string(), Style::default().fg(MUTED())),
        ]));
        lines
    }

    pub(super) fn render_composer_text(&self) -> Text<'static> {
        let prompt = Span::styled(
            format!("{} ", self.composer_prompt_glyph()),
            self.composer_prompt_style(),
        );
        let mut lines = Vec::new();
        if self.draft.is_empty() {
            let placeholder = if self.loading_resume.is_some() {
                Span::styled("Resume loading…", Style::default().fg(FAINT()))
            } else if self.sending {
                Span::styled(
                    "Type to queue your next message…",
                    Style::default().fg(FAINT()),
                )
            } else if self.plan_mode || self.cursor_plan_mode {
                Span::styled(
                    "Describe what to plan · read-only until approved",
                    Style::default().fg(FAINT()),
                )
            } else if self.ask_mode {
                Span::styled(
                    "Ask anything · concepts, docs, code, the web",
                    Style::default().fg(FAINT()),
                )
            } else {
                Span::styled(
                    "Ask, plan, or build · / for commands",
                    Style::default().fg(FAINT()),
                )
            };
            lines.push(Line::from(vec![prompt, placeholder]));
            return Text::from(lines);
        }

        // Ghost hint trailing a bare slash command (Claude-Code style), e.g.
        // `> /mcp [add … | rm <name>]`. Only set when the draft is a single line.
        let ghost = self.composer_command_hint();
        // The prompt and border carry shell state; long command text remains calm
        // and readable instead of becoming a saturated block.
        let draft_color = TEXT();
        // Row 0 carries the in-box `❯ ` prompt; wrapped rows get a same-width
        // hanging indent so text aligns under the first character.
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let last = rows.len().saturating_sub(1);
        // Tags and mentions render accented so they read as objects.
        let mut tag_spans: Vec<(usize, usize)> =
            self.attachment_tag_spans().into_iter().flatten().collect();
        tag_spans.extend(
            mention_tokens(&self.draft)
                .into_iter()
                .map(|t| (t.start, t.end)),
        );
        tag_spans.sort_unstable();
        for (index, &(start, end)) in rows.iter().enumerate().skip(self.composer_scroll) {
            let prefix = if index == 0 {
                prompt.clone()
            } else {
                Span::raw("  ")
            };
            let mut spans = vec![prefix];
            let mut pos = start;
            for &(ts, te) in &tag_spans {
                let (ts, te) = (ts.max(pos), te.min(end));
                if ts >= te {
                    continue;
                }
                if pos < ts {
                    spans.push(Span::styled(
                        self.draft[pos..ts].to_string(),
                        Style::default().fg(draft_color),
                    ));
                }
                spans.push(Span::styled(
                    self.draft[ts..te].to_string(),
                    Style::default().fg(ACCENT()),
                ));
                pos = te;
            }
            if pos < end {
                spans.push(Span::styled(
                    self.draft[pos..end].to_string(),
                    Style::default().fg(draft_color),
                ));
            }
            if index == last
                && let Some(hint) = ghost
            {
                spans.push(Span::styled(
                    format!(" {hint}"),
                    Style::default().fg(FAINT()),
                ));
            }
            lines.push(Line::from(spans));
        }

        Text::from(lines)
    }

    fn composer_prompt_glyph(&self) -> &'static str {
        if self.draft_history_index.is_some() {
            "^"
        } else {
            "❯"
        }
    }

    fn composer_prompt_style(&self) -> Style {
        let color = if self.draft_history_index.is_some() {
            ACCENT()
        } else if self.draft_is_shell_command() {
            SHELL()
        } else {
            ACCENT()
        };
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    }

    /// Scroll the draft within the composer so the cursor's visual row stays
    /// inside the visible window. Recomputed each render.
    pub(super) fn update_composer_scroll(&mut self, area: Rect) {
        let rows = composer_visual_rows(&self.draft, self.composer_text_width());
        let visible = usize::from(area.height).max(1);
        let (cursor_row, _) = composer_cursor_rowcol(&self.draft, self.cursor, &rows);
        if cursor_row < self.composer_scroll {
            self.composer_scroll = cursor_row;
        } else if cursor_row >= self.composer_scroll + visible {
            self.composer_scroll = cursor_row + 1 - visible;
        }
        self.composer_scroll = self.composer_scroll.min(rows.len().saturating_sub(visible));
    }

    /// Absolute terminal `(x, y)` for the input cursor, in the composer's
    /// hanging-indent wrap model, accounting for scroll. `None` when the cursor
    /// row is scrolled out of view.
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
        let y = area.y + (row - self.composer_scroll) as u16;
        let x = area.x + x_rel;
        let max_x = area.x + area.width.saturating_sub(1);
        let max_y = area.y + area.height.saturating_sub(1);
        Some((x.min(max_x), y.min(max_y)))
    }

    pub(super) fn render_footer(&mut self, frame: &mut Frame<'_>, area: Rect) {
        // One column of inset on each side so the row's padding reads symmetric.
        let area = Rect {
            x: area.x.saturating_add(1),
            width: area.width.saturating_sub(2),
            ..area
        };
        // Cleared so a frame that omits the id/badge leaves no stale click
        // target; re-armed below when shown.
        self.session_id_hit = None;
        self.share_badge_hit = None;
        self.effort_badge_hit = None;
        let width = usize::from(area.width);
        let glue_w = 3usize; // " · "

        // Right cluster — the engine: model, key/host, MCP health, effort, then
        // the context meter anchoring the corner (the one element that warms
        // toward the limit, so its warning color sits at the edge).
        let (meter_label, meter_color) = self.footer_status_label();
        let (model_label, host_label) =
            footer_engine_labels(&self.raw_model, &self.key.base_url, &self.key.name);
        // Tail right of the model: width-gated so the model and meter win when narrow.
        let mut tail: Vec<Span<'static>> = Vec::new();
        if area.width >= 70
            && let Some((mcp_label, mcp_color)) = self.footer_mcp_label()
        {
            tail.push(Span::styled(" · ", Style::default().fg(FAINT())));
            tail.push(Span::styled(mcp_label, Style::default().fg(mcp_color)));
        }
        // Taken at the push site so a span reorder can't drift the click target.
        let mut effort_rel_in_tail = None;
        if let Some(effort) = self.footer_effort_label() {
            // A static setting: FAINT, a step under the meter it introduces.
            tail.push(Span::styled(" · ", Style::default().fg(FAINT())));
            // Cursor's tier is part of the model id — /config can't change it,
            // so that badge stays inert.
            if self.cursor_effort_label.is_none() {
                let rel: usize = tail.iter().map(|s| display_width(s.content.as_ref())).sum();
                effort_rel_in_tail = Some((rel, display_width(&effort)));
            }
            tail.push(Span::styled(effort, Style::default().fg(FAINT())));
        }
        if area.width >= 70 {
            for label in [self.footer_tps_label(), self.footer_cache_label()]
                .into_iter()
                .flatten()
            {
                tail.push(Span::styled(" · ", Style::default().fg(FAINT())));
                tail.push(Span::styled(label, Style::default().fg(FAINT())));
            }
        }
        tail.push(Span::styled(" · ", Style::default().fg(FAINT())));
        tail.push(Span::styled(meter_label, Style::default().fg(meter_color)));
        let tail_w: usize = tail.iter().map(|s| display_width(s.content.as_ref())).sum();
        let live = self.share.handle.is_some();
        let plain_chat = !self.agent_tools_enabled;
        let live_badge_w = display_width(LIVE_BADGE);
        let badge_w = if live { live_badge_w + glue_w } else { 0 }
            + if plain_chat {
                display_width(PLAIN_CODE_BADGE) + glue_w
            } else {
                0
            };
        // The host segment is the first thing dropped, then the model itself
        // truncates, so the meter never leaves the corner.
        let host = host_label.filter(|h| {
            display_width(&model_label) + badge_w + glue_w + display_width(h) + tail_w <= width
        });
        let host_w = host
            .as_ref()
            .map(|h| glue_w + display_width(h))
            .unwrap_or(0);
        let model_shown = truncate_for_display_width(
            &model_label,
            width.saturating_sub(badge_w + host_w + tail_w).max(1),
        );
        let mut right_spans: Vec<Span<'static>> = Vec::new();
        right_spans.push(Span::styled(model_shown, Style::default().fg(MUTED())));
        // Badges sit right after the model they qualify.
        let mut live_badge_rel = 0usize;
        if live {
            right_spans.push(Span::styled(" · ", Style::default().fg(FAINT())));
            // Taken at the push site so a span reorder can't drift the click target.
            live_badge_rel = right_spans
                .iter()
                .map(|s| display_width(s.content.as_ref()))
                .sum();
            right_spans.push(Span::styled(LIVE_BADGE, Style::default().fg(LIVE())));
        }
        if plain_chat {
            right_spans.push(Span::styled(" · ", Style::default().fg(FAINT())));
            right_spans.push(Span::styled(PLAIN_CODE_BADGE, Style::default().fg(MUTED())));
        }
        if let Some(host) = host {
            right_spans.push(Span::styled(" · ", Style::default().fg(FAINT())));
            right_spans.push(Span::styled(host, Style::default().fg(FAINT())));
        }
        right_spans.extend(tail);
        let right_w: usize = right_spans
            .iter()
            .map(|s| display_width(s.content.as_ref()))
            .sum();

        // Left cluster — the workspace: cwd (branch) degrading to basename, plus
        // the clickable session-id handle behind a width gate. Cedes the row to
        // the engine side when narrow.
        let left_budget = width.saturating_sub(right_w + 1);
        // A fork keeps its source in view (`claude·a1b2c3d4`); a native id shows
        // its short handle (`#3f2a1b4c`). Click it for the full detail overlay.
        let id_label = (area.width >= 90 && !self.session_id.is_empty())
            .then(|| footer_session_label(&self.session_id));
        let id_w = id_label
            .as_ref()
            .map(|id| glue_w + display_width(id))
            .unwrap_or(0);
        let candidates =
            footer_workspace_candidates(self.display_cwd(), self.git_branch.as_deref());
        // Keep the id only while a cwd candidate leaves room for it.
        let chosen = candidates
            .iter()
            .find(|c| display_width(c) + id_w <= left_budget)
            .map(|c| (c.clone(), id_label.is_some()))
            .or_else(|| {
                candidates
                    .iter()
                    .find(|c| display_width(c) <= left_budget)
                    .map(|c| (c.clone(), false))
            });
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut left_w = 0usize;
        if let Some((cwd_text, show_id)) = chosen {
            if !cwd_text.is_empty() {
                left_w = display_width(&cwd_text);
                spans.push(Span::styled(cwd_text, Style::default().fg(MUTED())));
            }
            if show_id && let Some(id) = id_label.as_ref() {
                if left_w > 0 {
                    spans.push(Span::styled(" · ", Style::default().fg(FAINT())));
                    left_w += glue_w;
                }
                self.session_id_hit = Some(Rect {
                    x: area.x + left_w as u16,
                    y: area.y,
                    width: display_width(id) as u16,
                    height: area.height.max(1),
                });
                spans.push(Span::styled(id.clone(), Style::default().fg(FAINT())));
                left_w += display_width(id);
            }
        }
        spans.push(Span::raw(
            " ".repeat(width.saturating_sub(left_w + right_w)),
        ));
        spans.extend(right_spans);
        if live {
            self.share_badge_hit = Some(Rect {
                x: area.x + (width.saturating_sub(right_w) + live_badge_rel) as u16,
                y: area.y,
                width: live_badge_w as u16,
                height: area.height.max(1),
            });
        }
        if let Some((rel, effort_w)) = effort_rel_in_tail {
            let tail_origin = width.saturating_sub(tail_w.saturating_sub(rel));
            self.effort_badge_hit = Some(Rect {
                x: area.x + tail_origin as u16,
                y: area.y,
                width: effort_w as u16,
                height: area.height.max(1),
            });
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Effort tier for the status line: Cursor's from the model id, else the
    /// engine's level while thinking is on, else the off pill's word — or the
    /// level the engine substitutes for off (nothing on models that can't
    /// think at all).
    pub(super) fn footer_effort_label(&self) -> Option<String> {
        if let Some(label) = self.cursor_effort_label.as_deref() {
            Some(label.to_string())
        } else if !self.model_supports_thinking {
            None
        } else if self.thinking_enabled {
            self.effective_reasoning_effort()
        } else if self.thinking_off_unavailable() {
            // Off unrepresentable (tools forbid it): the engine substitutes the
            // lowest real level — show that, never a false "thinking off".
            self.model_reasoning_efforts
                .iter()
                .find(|l| !crate::services::model_metadata::effort_level_is_off(l))
                .cloned()
        } else {
            match crate::services::model_metadata::thinking_off_wire(
                &self.model,
                &self.model_reasoning_efforts,
            ) {
                // Off maps to a plain level here (o-series `low`): show what ships.
                (Some(level), _)
                    if !crate::services::model_metadata::effort_level_is_off(level) =>
                {
                    Some(level.to_string())
                }
                // Same word as the scale's pill — the badge sits in the level slot.
                _ => Some("off".to_string()),
            }
        }
    }

    /// Mid-turn: live throughput over the decision-wait-corrected turn clock
    /// (tool runtime dilutes on purpose), quiet the first second. Idle: the
    /// last turn's frozen figure.
    pub(super) fn footer_tps_label(&self) -> Option<String> {
        if !self.footer_tps_enabled {
            return None;
        }
        if self.sending {
            let started = self.request_started_at?;
            let elapsed = started.elapsed().as_secs_f64();
            let unmeasured = crate::agent::tokens::chars_to_tokens(
                self.turn_stream_chars
                    .saturating_sub(self.turn_stream_chars_measured),
            );
            let tokens = self.turn_output_tokens + unmeasured;
            if elapsed < 1.0 || tokens == 0 {
                return None;
            }
            return Some(format_tps(tokens as f64 / elapsed));
        }
        self.last_turn_tps.map(format_tps)
    }

    pub(super) fn footer_cache_label(&self) -> Option<String> {
        if !self.footer_cache_enabled {
            return None;
        }
        self.last_cache_hit_pct.map(|pct| format!("cache:{pct}%"))
    }

    /// Aggregate MCP health for the status line, or `None` with no configured
    /// servers. Quiet MUTED when healthy — only trouble gets warmth.
    pub(super) fn footer_mcp_label(&self) -> Option<(String, Color)> {
        let n = self.mcp_configured_count;
        if n == 0 {
            return None;
        }
        if let Some(client) = &self.mcp_client {
            if client.any_dead() || !client.errors().is_empty() {
                return Some((format!("mcp:{n}!"), ERROR()));
            }
            if client.any_needs_auth() {
                return Some((format!("mcp:{n}!"), WARNING()));
            }
            return Some((format!("mcp:{n}"), MUTED()));
        }
        if self.mcp_connecting {
            return Some((format!("mcp:{n}…"), FAINT()));
        }
        Some((format!("mcp:{n}"), FAINT()))
    }

    /// Present-tense label for the in-flight tool step (e.g. `running grep`), or
    /// `None`. Uses the same in-flight test that hides the tool's card (trailing
    /// `tool_call` run with a pending result), so the status and the card never
    /// both show.
    pub(super) fn current_action_label(&self) -> Option<String> {
        self.trailing_tool_batch()?;
        self.last_tool_action
            .as_ref()
            .map(|(label, _, _)| label.clone())
    }

    /// The trailing contiguous `tool_call` run while the model is between
    /// replies: `(start index, total, resolved)`. `None` when idle, once reply
    /// text is streaming, or when every call has its outcome — cursor resolves
    /// entries in place, so a batch stays live until its last member resolves.
    pub(super) fn trailing_tool_batch(&self) -> Option<(usize, usize, usize)> {
        if !self.sending || !self.pending_response.is_empty() || !self.incoming_buffer.is_empty() {
            return None;
        }
        let mut start = self.history.len();
        while start > 0 && self.history[start - 1].role == "tool_call" {
            start -= 1;
        }
        let total = self.history.len() - start;
        let done = self.history[start..]
            .iter()
            .filter(|m| {
                let (result, failed) = decode_tool_outcome(&m.content);
                result.is_some() || failed
            })
            .count();
        (done < total).then_some((start, total, done))
    }

    /// Tokens to show in the footer fill right now, and whether the figure is a
    /// chars/4 estimate rather than provider-measured. During an in-flight turn we
    /// prefer the live measured usage (Anthropic streams it from `message_start`);
    /// until that lands we grow a chars/4 estimate of the transcript plus the text
    /// streamed so far, so the fill still moves for providers that only report
    /// usage at the end of the turn. Idle: the last turn's measured total.
    pub(super) fn context_fill(&self) -> (u64, bool) {
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
            return (format_token_count(used, usage), MUTED());
        }
        // Fresh session: show the window size, not an empty `0 / 1M` meter.
        if used == 0 {
            return (
                format!("{} context", format_token_count_value(self.context_window)),
                MUTED(),
            );
        }
        // Percent isn't shown (the used/window pair already implies it) but still
        // drives the meter color.
        let pct = (used.saturating_mul(100) / self.context_window).min(100);
        // Mark estimate-only figures (cursor ACP / agents without reported usage):
        // aivo's tracked transcript is a fraction of the model's real context, so
        // the number understates the true fill — `~` flags it as approximate.
        let approx = if is_estimate && used > 0 { "~" } else { "" };
        let label = format!(
            "{approx}{}/{}{}",
            format_token_count_value(used),
            format_token_count_value(self.context_window),
            self.session_cost_label(),
        );
        (label, context_fill_color(pct))
    }

    /// ` · ~$X.XX` session-spend suffix; empty when toggled off (`/config`) or
    /// without any recorded spend.
    /// Always `~`: snapshot list prices × parsed usage is an estimate, not a bill.
    pub(super) fn session_cost_label(&self) -> String {
        if !self.footer_price_enabled || self.session_cost_usd <= 0.0 {
            return String::new();
        }
        format!(" · ~${}", format_usd(self.session_cost_usd))
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
                .saturating_sub(2)
                .saturating_sub(usize::from(COMPOSER_PREFIX_WIDTH))
                .max(1);
            composer_visual_rows(&self.draft, text_width).len()
        };
        // +3 reserves the leading blank spacing row and the box's two border
        // rows; the rest is draft text (capped, then it scrolls within the box).
        (draft_rows as u16 + 3).clamp(4, 10)
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
        "write_outside_workspace" => "Write outside the workspace?".to_string(),
        "add_write_root" => "Add a writable root?".to_string(),
        "cursor" => "Allow Cursor to run this?".to_string(),
        "write_file" => "Write a file?".to_string(),
        "edit_file" | "multi_edit" => "Edit a file?".to_string(),
        other => format!("Allow {other}?"),
    }
}

/// A decision card built for the reserved slot between the panels and the
/// composer; `scroll` carries a scrollable card's clamp for write-back.
struct SlotCard {
    title: &'static str,
    border: Color,
    width: u16,
    lines: Vec<Line<'static>>,
    scroll: Option<u16>,
}

fn build_account_card(
    title: &'static str,
    border: Color,
    lines: Vec<Line<'static>>,
    max_width: u16,
) -> SlotCard {
    let content_w = lines.iter().map(Line::width).max().unwrap_or(0);
    let width = (content_w as u16)
        .saturating_add(4)
        .clamp(1, max_width.max(1));
    SlotCard {
        title,
        border,
        width,
        lines,
        scroll: None,
    }
}

/// Paints a built card into its slot. When even the builder-trimmed card
/// overflows, spacer rows go first, then rows shed from the top — the
/// key-hint row is the last thing to disappear.
fn render_slot_card(frame: &mut Frame<'_>, slot: Rect, card: SlotCard) {
    if slot.height == 0 || slot.width == 0 {
        return;
    }
    clear_to_canvas(frame, slot);
    let mut lines = card.lines;
    let inner_h = usize::from(slot.height.saturating_sub(2)).max(1);
    if lines.len() > inner_h {
        lines.retain(|l| l.width() > 0);
    }
    if lines.len() > inner_h {
        let drop = lines.len() - inner_h;
        lines.drain(..drop);
    }
    let height = (lines.len() as u16 + 2).min(slot.height);
    let width = card.width.min(slot.width.saturating_sub(1)).max(1);
    let rect = Rect {
        x: slot.x.saturating_add(1),
        y: slot.bottom().saturating_sub(height),
        width,
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(card.border))
        .title(Span::styled(
            card.title,
            Style::default()
                .fg(card.border)
                .add_modifier(Modifier::BOLD),
        ));
    let inner = block.inner(rect).inner(ratatui::layout::Margin {
        vertical: 0,
        horizontal: 1,
    });
    frame.render_widget(block, rect);
    frame.render_widget(Paragraph::new(Text::from(lines)), inner);
}

/// Color-coded `key label` hint row for the account cards.
fn account_keys_line(keys: &[(&'static str, Color, &'static str)]) -> Line<'static> {
    let mut spans = Vec::new();
    for (i, (key, color, label)) in keys.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(
                "    ".to_string(),
                Style::default().fg(FAINT()),
            ));
        }
        spans.push(Span::styled(
            key.to_string(),
            Style::default().fg(*color).add_modifier(Modifier::BOLD),
        ));
        spans.push(Span::styled(
            format!(" {label}"),
            Style::default().fg(MUTED()),
        ));
    }
    Line::from(spans)
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
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED()));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT()));
    Line::from(vec![
        keycap("y", ASSISTANT()),
        label(" run once"),
        gap(),
        keycap("a", WARNING()),
        label(" always (this repo)"),
        gap(),
        keycap("n", ERROR()),
        label(" deny"),
    ])
}

/// The choices row: color-coded keycaps reading like a traffic light —
/// green allow, amber always (it arms auto-approve), red deny. `tool` selects
/// the "always" scope wording (a Cursor card's "always" is session-wide, unlike
/// the native engine's, which is scoped to this one command/path), `once_only`
/// drops "always" entirely (never remembered), and `composing` swaps in a hint
/// when a queued-message draft is in progress — there the letter keys type into
/// the draft instead of deciding (see `handle_permission_key`).
fn permission_keys_line(tool: &str, once_only: bool, composing: bool) -> Line<'static> {
    let keycap = |key: &str, color: Color| {
        Span::styled(
            key.to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED()));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT()));
    if composing {
        // A draft is in progress, so y/a/n flow into the message, not the card.
        // Only Esc (deny) and Shift+Tab (allow + auto-approve) act on the card.
        return Line::from(vec![
            keycap("⇧⇥", WARNING()),
            label(" allow"),
            gap(),
            keycap("Esc", ERROR()),
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
    let mut spans = vec![keycap("y", ASSISTANT()), label(" allow once"), gap()];
    if !once_only {
        spans.push(keycap("a", WARNING()));
        spans.push(label(always_label));
        spans.push(gap());
    }
    spans.push(keycap("n", ERROR()));
    spans.push(label(" deny"));
    Line::from(spans)
}

/// The `ask_user` card's key-hint row: "space toggle · ↵ confirm" in multi-select,
/// otherwise "↵ select" (with a "type your own" note when free text is allowed).
fn ask_user_keys_line(allow_free_text: bool, multi_select: bool) -> Line<'static> {
    let keycap = |key: &str| {
        Span::styled(
            key.to_string(),
            Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED()));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT()));
    let mut spans = vec![keycap("↑↓"), label(" move")];
    if multi_select {
        spans.push(gap());
        spans.push(keycap("Space"));
        spans.push(label(" toggle"));
        spans.push(gap());
        spans.push(keycap("↵"));
        spans.push(label(" confirm"));
    } else {
        spans.push(gap());
        spans.push(keycap("↵"));
        spans.push(label(" select"));
        if allow_free_text {
            spans.push(gap());
            spans.push(label("type your own"));
        }
    }
    spans.push(gap());
    spans.push(keycap("Esc"));
    spans.push(label(" dismiss"));
    Line::from(spans)
}

/// The plan-approval card's key-hint row.
fn plan_approval_keys_line() -> Line<'static> {
    let keycap = |key: &str| {
        Span::styled(
            key.to_string(),
            Style::default().fg(ACCENT()).add_modifier(Modifier::BOLD),
        )
    };
    let label = |text: &str| Span::styled(text.to_string(), Style::default().fg(MUTED()));
    let gap = || Span::styled("    ".to_string(), Style::default().fg(FAINT()));
    Line::from(vec![
        keycap("↑↓"),
        label(" choose"),
        gap(),
        keycap("↵"),
        label(" confirm"),
        gap(),
        keycap("⇞⇟"),
        label(" scroll"),
        gap(),
        label("type feedback"),
        gap(),
        keycap("Esc"),
        label(" dismiss"),
    ])
}

#[cfg(test)]
mod render_impl_tests {
    use super::scrub_control_cells;
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;

    #[test]
    fn scrub_replaces_control_cells_with_spaces() {
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 1));
        buf.cell_mut((0, 0)).unwrap().set_symbol("a");
        buf.cell_mut((1, 0)).unwrap().set_symbol("\t");
        buf.cell_mut((2, 0)).unwrap().set_symbol("\u{1b}");
        buf.cell_mut((3, 0)).unwrap().set_symbol("界");
        scrub_control_cells(&mut buf);
        assert_eq!(buf.cell((0, 0)).unwrap().symbol(), "a");
        assert_eq!(buf.cell((1, 0)).unwrap().symbol(), " ");
        assert_eq!(buf.cell((2, 0)).unwrap().symbol(), " ");
        // Non-control symbols (incl. wide chars) are untouched.
        assert_eq!(buf.cell((3, 0)).unwrap().symbol(), "界");
    }
}
