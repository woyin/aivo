use super::super::*;
use super::helpers::*;

#[test]
fn test_finished_turn_renders_done_marker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "fix it".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.turn_durations.insert(1, 404_000); // stamped on the last entry; 6m 44s
    app.transcript_revision = app.transcript_revision.wrapping_add(1);

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✶ Done in 6m 44s"), "{plain}");

    // No recorded duration → no marker.
    app.turn_durations.clear();
    app.transcript_revision = app.transcript_revision.wrapping_add(1);
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(!plain.contains("Done in"), "{plain}");
}

#[test]
fn test_footer_status_label_is_token_count_without_window() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_tokens = 5_120; // window unknown (0) → plain count

    // Idle and while sending: the footer is the token count — processing status
    // lives in the transcript, not the footer corner.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "~5.1k tokens");
    assert_eq!(color, MUTED());

    app.sending = true;
    assert_eq!(app.footer_status_label().0, "~5.1k tokens");
}

#[test]
fn test_footer_status_label_shows_window_on_pristine_session() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 1_000_000;
    app.context_tokens = 0; // no turn yet → no fill to gauge

    // The welcome screen shows the window size, not an empty `0/1M` meter.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "1M context");
    assert_eq!(color, MUTED());

    // The first tokens flip it to the live gauge.
    app.context_tokens = 50_000;
    app.context_is_estimate = false;
    assert_eq!(app.footer_status_label().0, "50k/1M");
}

#[test]
fn test_footer_status_label_shows_context_utilization() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;
    app.context_is_estimate = false; // a provider-measured fill

    // used/window, quiet until it nears the limit.
    let (label, color) = app.footer_status_label();
    assert_eq!(label, "10k/200k");
    assert_eq!(color, MUTED());

    // Warms toward the window limit (compaction territory).
    app.context_tokens = 170_000; // 85%
    assert_eq!(app.footer_status_label().1, WARNING());
    app.context_tokens = 195_000; // 97%
    assert_eq!(app.footer_status_label().1, ERROR());

    // A measured last-turn total wins over the chars/4 estimate.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k/200k");
}

#[test]
fn test_footer_status_label_marks_estimates() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 10_000;

    // cursor ACP / agents without reported usage: the chars/4 transcript figure
    // understates the model's real context, so flag it with `~`.
    app.context_is_estimate = true;
    app.last_usage = None;
    assert_eq!(app.footer_status_label().0, "~10k/200k");

    // A provider-measured last-turn total is exact even if the estimate flag
    // lingers from a prior turn — no tilde.
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 40_000,
        completion_tokens: 0,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "40k/200k");
}

#[test]
fn test_footer_shows_plain_chat_badge_when_agent_tools_off() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn footer_text(app: &mut CodeTuiApp) -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 1)).unwrap();
        terminal
            .draw(|frame| app.render_footer(frame, frame.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().to_string())
            .collect()
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // On (default): no badge.
    app.agent_tools_enabled = true;
    assert!(!footer_text(&mut app).contains("plain chat"));

    // Off: the badge marks plain-chat mode in the footer.
    app.agent_tools_enabled = false;
    assert!(footer_text(&mut app).contains("plain chat"));
}

#[test]
fn test_footer_mcp_label_reflects_aggregate_health() {
    use crate::agent::mcp::McpClient;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.mcp_configured_count = 0;
    assert!(app.footer_mcp_label().is_none());

    app.mcp_configured_count = 2;
    app.mcp_connecting = true;
    assert_eq!(app.footer_mcp_label().unwrap().0, "mcp:2…");

    app.mcp_connecting = false;
    app.mcp_client = Some(std::sync::Arc::new(McpClient::with_state_for_tests(
        Vec::new(),
        std::collections::HashSet::new(),
    )));
    assert_eq!(
        app.footer_mcp_label().unwrap(),
        ("mcp:2".to_string(), MUTED())
    );

    app.mcp_client = Some(std::sync::Arc::new(McpClient::with_state_for_tests(
        vec![("db".to_string(), "spawn failed".to_string())],
        std::collections::HashSet::new(),
    )));
    assert_eq!(
        app.footer_mcp_label().unwrap(),
        ("mcp:2!".to_string(), ERROR())
    );

    // OAuth-pending with nothing failed is WARNING, not ERROR.
    app.mcp_client = Some(std::sync::Arc::new(McpClient::with_state_for_tests(
        Vec::new(),
        std::collections::HashSet::from(["gh".to_string()]),
    )));
    assert_eq!(
        app.footer_mcp_label().unwrap(),
        ("mcp:2!".to_string(), WARNING())
    );
}

#[test]
fn test_footer_shows_short_session_id_only_on_wide_terminals() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn footer_text(app: &mut CodeTuiApp, width: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, 1)).unwrap();
        terminal
            .draw(|frame| app.render_footer(frame, frame.area()))
            .unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.width)
            .map(|x| buf.cell((x, 0)).unwrap().symbol().to_string())
            .collect()
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "abcdef12-3456-7890-abcd-ef1234567890".to_string();

    assert!(footer_text(&mut app, 100).contains("#abcdef12"));
    assert!(!footer_text(&mut app, 80).contains("#abcdef12"));
}

#[test]
fn test_welcome_status_lines_include_static_essentials_hint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let plain: Vec<String> = app
        .welcome_status_lines()
        .into_iter()
        .map(|sl| sl.plain)
        .collect();
    assert!(
        plain
            .iter()
            .any(|l| l.contains("/help commands") && l.contains("Shift+Tab modes")),
        "essentials hint missing from welcome: {plain:?}"
    );
}

#[test]
fn test_footer_fork_id_is_labeled_and_clickable() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.session_id = "import-claude-a1b2c3d4".to_string();

    let mut terminal = Terminal::new(TestBackend::new(100, 1)).unwrap();
    terminal
        .draw(|frame| app.render_footer(frame, frame.area()))
        .unwrap();
    let buf = terminal.backend().buffer();
    let row: String = (0..buf.area.width)
        .map(|x| buf.cell((x, 0)).unwrap().symbol().to_string())
        .collect();
    // The fork's source stays in view; the `import-` noise is gone.
    assert!(row.contains("claude·a1b2c3d4"), "footer row: {row:?}");
    assert!(!row.contains("import-"), "footer row: {row:?}");
    // The id is recorded as a click target for the detail overlay.
    let hit = app.session_id_hit.expect("session id click rect recorded");
    // The label the click rect covers actually holds the id (not the meter).
    let covered: String = (hit.x..hit.x + hit.width)
        .map(|x| buf.cell((x, hit.y)).unwrap().symbol().to_string())
        .collect();
    assert_eq!(covered, "claude·a1b2c3d4", "click rect misaligned: {row:?}");
}

#[test]
fn test_footer_status_label_updates_live_during_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.context_tokens = 40_000; // prior turn's measured fill
    app.context_is_estimate = false;

    // Turn in flight, no measured usage yet: the footer holds the prior fill as a
    // baseline (no drop at turn start) and grows it as text streams in, flagged `~`.
    app.sending = true;
    app.pending_response = "x".repeat(4_000); // ~1k tokens streamed so far
    assert_eq!(app.footer_status_label().0, "~41k/200k");

    // Provider-measured usage arrives mid-stream (Anthropic message_start/_delta):
    // the live figure replaces the estimate immediately, no `~`.
    app.live_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_000,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52k/200k");

    // Turn ends: the fold into last_usage keeps the measured total on the footer.
    app.sending = false;
    app.live_usage = None;
    app.last_usage = Some(TokenUsage {
        prompt_tokens: 50_000,
        completion_tokens: 2_500,
        ..Default::default()
    });
    assert_eq!(app.footer_status_label().0, "52.5k/200k");
}

#[test]
fn test_agent_context_drives_footer_live() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.context_window = 200_000;
    app.sending = true;

    // Pre-usage request estimate (system prompt + tool schemas + conversation):
    // counts the real request the engine sends, not just the visible transcript.
    // Flagged `~` since it is a chars/4 estimate.
    app.apply_agent_context(50_000, false);
    assert_eq!(app.footer_status_label().0, "~50k/200k");

    // The step's measured total arrives → exact figure, no `~`, and streamed text
    // is not re-added on top (it is already in the measured completion).
    app.pending_response = "x".repeat(8_000); // would add ~2k if double-counted
    app.apply_agent_context(60_000, true);
    assert_eq!(app.footer_status_label().0, "60k/200k");
}

#[test]
fn test_current_action_label_reflects_phase() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // No tool in flight → pure model compute, only the Thinking heartbeat.
    assert_eq!(app.current_action_label(), None);
    // A tool call in flight → name the step, present-tense, with its target.
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "ls"}),
        vec![],
        None,
    );
    assert_eq!(app.current_action_label().as_deref(), Some("running ls"));
    // Streamed tokens arriving → the step is over, heartbeat only.
    app.pending_response = "partial".to_string();
    assert_eq!(app.current_action_label(), None);
}

#[test]
fn test_tool_action_label_is_present_tense_with_target() {
    use super::super::render::tool_action_label;
    assert_eq!(
        tool_action_label("read_file", &serde_json::json!({"path": "src/main.rs"}), ""),
        "reading main.rs"
    );
    assert_eq!(
        tool_action_label("grep", &serde_json::json!({"pattern": "parse_expr"}), ""),
        "searching parse_expr"
    );
    assert_eq!(
        tool_action_label("edit_file", &serde_json::json!({"path": "a/b.rs"}), ""),
        "editing b.rs"
    );
    // MCP / unknown tools: named, no target to show.
    assert_eq!(
        tool_action_label("mcp__linear__create_issue", &serde_json::json!({}), ""),
        "running linear/create_issue"
    );
}

#[test]
fn test_tool_action_label_caps_long_command() {
    use super::super::render::tool_action_label;
    let long = "cd /private/tmp/test/markdown-preview && node server.js > /tmp/server.log & \
                sleep 2 && curl -s http://localhost:3000/ && kill $SERVER_PID";
    let label = tool_action_label("run_bash", &serde_json::json!({ "command": long }), "");
    // Capped to one line: short, ellipsized, single line.
    assert!(
        label.starts_with("running cd /private/tmp"),
        "label: {label:?}"
    );
    assert!(label.ends_with('…'), "ellipsized: {label:?}");
    assert!(label.chars().count() <= 50, "kept short: {label:?}");
    assert!(!label.contains('\n'), "single line: {label:?}");
}

#[test]
fn test_current_action_shows_inline_on_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "parse_expr"}),
        vec![],
        None,
    );
    // The action replaces "Thinking" on the SAME single status line (no extra
    // line), so the layout never shifts as steps come and go.
    let status = app
        .build_transcript()
        .plain_lines
        .into_iter()
        .find(|l| l.contains("searching parse_expr"))
        .expect("action shown inline on the status line");
    // It's the live status line (carries the elapsed clock), not a tool card.
    assert!(status.contains('('), "on the status line: {status:?}");
}

#[test]
fn test_subagent_activity_drives_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // Parent's `subagent` call in flight (no result) — the case that used to
    // freeze the status line on the parent's last tool for the whole delegation.
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"agent": "code-reviewer", "task": "review mcp.rs"}),
        vec![],
        None,
    );
    app.apply_subagent_activity(
        "code-reviewer".to_string(),
        "grep".to_string(),
        serde_json::json!({"pattern": "fn"}),
        7,
    );
    let status = app
        .build_transcript()
        .plain_lines
        .into_iter()
        .find(|l| l.contains("code-reviewer: searching"))
        .expect("nested sub-agent activity shown on the status line");
    assert!(
        status.contains("step 7"),
        "carries the child's step count: {status:?}"
    );
    assert!(
        status.contains('↳'),
        "marked as nested delegation: {status:?}"
    );
}

#[test]
fn test_parallel_subagent_rows_render_under_status_line() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // The batch's `subagent` calls are in flight (no results yet).
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"label": "audit auth flow", "task": "…"}),
        vec![],
        None,
    );
    app.apply_subagent_begin(vec!["audit auth flow".to_string(), String::new()]);
    // The unnamed delegate is numbered so its row stays distinguishable.
    assert_eq!(app.subagent_rows[1].name, "sub-agent 2");
    app.apply_subagent_slot(
        0,
        "audit auth flow".to_string(),
        "grep".to_string(),
        serde_json::json!({"pattern": "session"}),
        3,
    );
    app.apply_subagent_denied(1, "run_bash".to_string());
    app.apply_subagent_done(1, true, 8, 1200);
    // Headline counts completions; per-delegate rows sit under it.
    assert_eq!(app.desired_status(), "running 2 sub-agents (1 done)");
    let lines = app.build_transcript().plain_lines;
    let running = lines
        .iter()
        .find(|l| l.contains("audit auth flow — searching session"))
        .expect("live row shows the delegate's current action");
    assert!(running.contains("step 3"), "carries steps: {running:?}");
    let done = lines
        .iter()
        .find(|l| l.contains("✓ sub-agent 2 — done"))
        .expect("finished row flips to ✓ with stats");
    assert!(done.contains("1.2k tokens"), "carries tokens: {done:?}");
    // A denied gated call surfaces as a warning notice, not a silent deny.
    let (_, notice) = app.notice.clone().expect("denial sets a notice");
    assert!(
        notice.contains("run_bash") && notice.contains("auto-denied"),
        "explains the deny: {notice:?}"
    );
    // Slot events past the row list are ignored (defensive).
    app.apply_subagent_slot(9, String::new(), String::new(), serde_json::Value::Null, 1);
    // Batch end retires the rows and hands the headline back.
    app.subagent_rows.clear();
    assert_ne!(app.desired_status(), "running 2 sub-agents (1 done)");
}

#[test]
fn subagent_done_attributes_only_discovered_profiles() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let profile = |name: &str| crate::agent::subagents::Subagent {
        name: name.to_string(),
        description: String::new(),
        model: None,
        tools: None,
        body: String::new(),
        isolation_worktree: false,
        repo_local: false,
        source: std::path::PathBuf::new(),
    };
    app.last_subagents = vec![profile("code-reviewer")];
    app.apply_subagent_begin(vec!["code-reviewer".to_string(), "sub-agent 2".to_string()]);
    // A row named after a discovered profile is attributed to it.
    assert_eq!(
        app.apply_subagent_done(0, true, 6, 900),
        Some("code-reviewer".to_string())
    );
    // A generic/labeled delegate is not (it would pollute per-agent stats).
    assert_eq!(app.apply_subagent_done(1, true, 4, 200), None);
    // An out-of-range slot is a defensive no-op.
    assert_eq!(app.apply_subagent_done(9, true, 1, 1), None);
}

#[test]
fn test_cursor_task_notice_renames_generic_batch_rows() {
    // A `cursor/task` notice arrives as an args-only AgentToolUpdate.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    for id in ["1", "2"] {
        app.apply_agent_tool_call(
            Some(id.to_string()),
            "subagent".to_string(),
            serde_json::json!({"label": "Subagent task"}),
            vec![],
            None,
        );
    }
    app.apply_agent_tool_update(
        "2".to_string(),
        Some(serde_json::json!({"label": "Audit the auth flow", "agent": "explore"})),
        None,
        false,
    );
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("↳ explore — Audit the auth flow"),
        "row renamed by the notice: {plain:?}"
    );
    assert!(
        plain.contains("↳ Subagent task"),
        "sibling untouched: {plain:?}"
    );
    // Enrichment also refreshes the one-line status label.
    assert_eq!(
        app.current_action_label().as_deref(),
        Some("delegating: Audit the auth flow")
    );
}

#[test]
fn test_status_tail_shows_turn_output_tokens() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.pending_response = "x".repeat(4_000); // ~1k tokens streamed
    // A live ~-flagged estimate, alongside the always-present interrupt hint —
    // stopping a runaway turn must be discoverable mid-turn.
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("tokens"), "estimate shown: {plain:?}");
    assert!(
        plain.contains("esc to interrupt"),
        "esc hint in the agent-turn status: {plain:?}"
    );
    // The engine reports the turn's cumulative generated tokens → exact (no ~),
    // distinct from the prompt-dominated context total (which stays in the footer).
    app.turn_output_tokens = 512;
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("512 tokens"), "turn output shown: {plain:?}");
    assert!(!plain.contains("~512"), "measured, not estimate: {plain:?}");
}

#[test]
fn test_status_tail_counts_queued_input() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.queued_messages.push("follow-up one".to_string());
    app.queued_messages.push("follow-up two".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("2 queued"), "queued chip missing: {plain:?}");
}

#[test]
fn test_desired_status_names_decision_waits_and_stalls() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    // A permission card means blocked on the user — the status must not read as
    // the (possibly destructive) command already executing.
    let (reply, _rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: Some("rm -rf build".to_string()),
        reply,
    });
    assert_eq!(app.desired_status(), "waiting for your approval");
    app.agent_permission = None;

    // A silent stream reads as a stall, not an ever-ticking "Thinking".
    app.last_stream_activity =
        std::time::Instant::now().checked_sub(std::time::Duration::from_secs(15));
    assert_eq!(app.desired_status(), "waiting");
    app.last_stream_activity = Some(std::time::Instant::now());
    assert_eq!(app.desired_status(), "Thinking");
}

#[test]
fn test_decision_wait_freezes_step_and_turn_clocks() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    let t0 = std::time::Instant::now();
    app.last_tool_action = Some(("running rm -rf build".to_string(), t0, None));
    app.request_started_at = Some(t0);
    let (reply, _rx) = tokio::sync::oneshot::channel();
    app.agent_permission = Some(PendingPermission {
        tool: "run_bash".to_string(),
        preview: None,
        reply,
    });
    // Two ticks a real interval apart: the waiting span must be pushed out of
    // both clocks, so their effective elapsed stays near zero.
    app.tick_decision_wait();
    std::thread::sleep(std::time::Duration::from_millis(30));
    app.tick_decision_wait();
    let (_, since, _) = app.last_tool_action.as_ref().unwrap();
    assert!(
        since.elapsed() < std::time::Duration::from_millis(15),
        "tool clock ran during the wait: {:?}",
        since.elapsed()
    );
    assert!(
        app.request_started_at.unwrap().elapsed() < std::time::Duration::from_millis(15),
        "turn clock ran during the wait"
    );
    // Card resolved → clocks run again from here.
    app.agent_permission = None;
    app.tick_decision_wait();
    assert!(app.wait_tick.is_none());
}

#[test]
fn test_bash_status_clock_shows_timeout_budget() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "run_bash".to_string(),
        serde_json::json!({"command": "cargo test", "timeout": 300}),
        vec![],
        None,
    );
    let (_, _, budget) = app.last_tool_action.as_ref().unwrap();
    assert_eq!(*budget, Some(300));
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains(" / "),
        "deadline missing from clock: {plain:?}"
    );
    // Non-bash steps carry no budget.
    app.apply_agent_tool_call(
        None,
        "read_file".to_string(),
        serde_json::json!({"path": "x.rs"}),
        vec![],
        None,
    );
    assert_eq!(app.last_tool_action.as_ref().unwrap().2, None);
}

#[test]
fn test_done_marker_appends_turn_note() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "done".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let idx = app.history.len() - 1;
    app.turn_durations.insert(idx, 42_000);
    app.turn_notes.insert(idx, "3.1k tokens".to_string());
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("✶ Done in 42s · 3.1k tokens"), "{plain}");
}

#[tokio::test]
async fn test_agent_error_notice_uses_error_color() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // An engine error (notify_error) lands in the error hue, not the neutral one.
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: 429".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, Some((ERROR(), "LLM error: 429".to_string())));
    // An ordinary notice stays neutral.
    app.tx
        .send(RuntimeEvent::AgentNotice("compacting context…".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.notice,
        Some((MUTED(), "compacting context…".to_string()))
    );
}

#[tokio::test]
async fn test_agent_error_persists_in_transcript() {
    // Beyond the transient notice, the error must land as a durable transcript entry.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentError(
            "the provider rejected this API key (upstream 401: bad key)".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    let last = app.history.last().expect("error entry committed");
    assert_eq!(last.role, "error");
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        plain.contains("✗ the provider rejected this API key"),
        "{plain}"
    );
    // Never seeded back to the model on an engine rebuild.
    assert!(
        super::super::runtime_impl::agent_seed_turns(&app.history).is_empty(),
        "error entries must stay display-only"
    );
}

#[tokio::test]
async fn test_done_marker_skipped_on_errored_turn() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    app.sending = true;
    app.request_started_at =
        std::time::Instant::now().checked_sub(std::time::Duration::from_secs(3));
    // The turn errors, then finishes → no `Done in …` marker (would misrepresent).
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: 429".to_string()))
        .unwrap();
    app.tx
        .send(RuntimeEvent::AgentFinished {
            steps: 1,
            tokens: 0,
            context_tokens: 0,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(
        app.turn_durations.is_empty(),
        "no Done marker on error: {:?}",
        app.turn_durations
    );
}

#[tokio::test]
async fn test_sandbox_escalation_notice_clears_on_next_output() {
    use crate::agent::engine::SANDBOX_ESCALATION_NOTICE;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            SANDBOX_ESCALATION_NOTICE.to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.notice,
        Some((MUTED(), SANDBOX_ESCALATION_NOTICE.to_string()))
    );
    app.tx
        .send(RuntimeEvent::AgentToolCall {
            id: None,
            name: "read_file".to_string(),
            args: serde_json::json!({"path": "x"}),
            line_starts: vec![],
            old_content: None,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "next tool clears the ack");

    // Streamed prose clears it too.
    app.tx
        .send(RuntimeEvent::AgentNotice(
            SANDBOX_ESCALATION_NOTICE.to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "done".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "new prose clears the ack");

    // An unrelated notice in the same slot is left alone.
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "Queued — sends later".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::AgentToolCall {
            id: None,
            name: "read_file".to_string(),
            args: serde_json::json!({"path": "y"}),
            line_starts: vec![],
            old_content: None,
        })
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(
        app.notice,
        Some((MUTED(), "Queued — sends later".to_string())),
        "unrelated notice survives"
    );
}

#[tokio::test]
async fn test_retry_notice_clears_on_recovery_not_stuck() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "connection issue — retrying (2/3)…".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.retrying, "retry flag set");
    assert!(app.notice.is_some());

    // Output resumes → recovered → notice + flag clear, not stuck.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "ok".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert_eq!(app.notice, None, "retry notice cleared on recovery");
    assert!(!app.retrying);

    // A real error notice must NOT be cleared by later output.
    app.tx
        .send(RuntimeEvent::AgentError("LLM error: boom".to_string()))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "x".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.notice.is_some(), "unrelated error notice survives");
}

#[test]
fn test_connection_retry_status_reads_working() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    assert_eq!(app.desired_status(), "Thinking");
    app.retrying = true;
    assert_eq!(app.desired_status(), "Working");
}

#[tokio::test]
async fn test_retry_notice_sets_retrying_then_progress_clears_it() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.tx
        .send(RuntimeEvent::AgentNotice(
            "connection issue — retrying (2/3)…".to_string(),
        ))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(app.retrying, "retry notice sets the flag");
    // A streamed chunk = recovery → clears it.
    app.tx
        .send(RuntimeEvent::Delta(ChatResponseChunk::Content(
            "hi".to_string(),
        )))
        .unwrap();
    app.handle_runtime_events().await.unwrap();
    assert!(!app.retrying, "progress clears the flag");
}

#[test]
fn test_status_label_throttled_to_min_duration() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    // First tick adopts the current label.
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("Thinking")
    );

    // A tool starts, but "Thinking" was just shown → it must hold its second.
    app.apply_agent_tool_call(
        None,
        "grep".to_string(),
        serde_json::json!({"pattern": "foo"}),
        vec![],
        None,
    );
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("Thinking"),
        "must hold the prior label for its minimum second"
    );

    // Backdate the display past the minimum → the next tick may switch.
    let old = std::time::Instant::now()
        .checked_sub(STATUS_MIN_DURATION + std::time::Duration::from_millis(50))
        .expect("instant in range");
    app.status_display = Some(("Thinking".to_string(), old));
    app.tick_status_throttle();
    assert_eq!(
        app.status_display.as_ref().map(|(s, _)| s.as_str()),
        Some("searching foo"),
        "switches once the prior label has had its second"
    );
}

#[test]
fn test_intro_column_stable_from_empty_to_message() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn aivo_col(app: &mut CodeTuiApp) -> u16 {
        let mut terminal = Terminal::new(TestBackend::new(48, 16)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer();
        let area = buf.area;
        // The half-block wordmark's top-left glyphs are "▄▀█" — find that run.
        for y in area.y..area.y + area.height {
            for x in area.x..area.x + area.width.saturating_sub(2) {
                if buf.cell((x, y)).unwrap().symbol() == "▄"
                    && buf.cell((x + 1, y)).unwrap().symbol() == "▀"
                    && buf.cell((x + 2, y)).unwrap().symbol() == "█"
                {
                    return x;
                }
            }
        }
        panic!("AIVO banner not found");
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let empty_col = aivo_col(&mut app);

    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: "hi".to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    let message_col = aivo_col(&mut app);

    assert_eq!(
        empty_col, message_col,
        "AIVO Chat banner shifts columns between the empty and message states"
    );
}

#[test]
fn test_transcript_intro_is_brand_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.raw_model = "claude-sonnet-4".to_string();
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/tmp/project".to_string();

    // Wide column: full "aivo code" mark, version trailing the baseline.
    let version = format!("  v{}", crate::version::VERSION);
    assert_eq!(
        app.transcript_intro_lines(80),
        vec![
            "▄▀█ █ █░█ █▀█  █▀▀ █▀█ █▀▄ █▀█".to_string(),
            format!("█▀█ █ ▀▄▀ █▄█  █▄▄ █▄█ █▄▀ █▄▄{version}"),
        ]
    );
}

#[test]
fn test_transcript_intro_narrow_falls_back_to_aivo() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);

    // Too slim for the full mark and the version: bare "aivo", no wrap.
    assert_eq!(
        app.transcript_intro_lines(20),
        vec!["▄▀█ █ █░█ █▀█".to_string(), "█▀█ █ ▀▄▀ █▄█".to_string()]
    );
    // Exactly the mark width keeps the full mark (version needs more room).
    assert_eq!(
        app.transcript_intro_lines(30)[0],
        "▄▀█ █ █░█ █▀█  █▀▀ █▀█ █▀▄ █▀█"
    );
    let version = format!("v{}", crate::version::VERSION);
    assert!(
        app.transcript_intro_lines(80)[1].ends_with(&version),
        "version should trail the wide banner"
    );
    assert!(
        !app.transcript_intro_lines(20)
            .iter()
            .any(|l| l.contains(&version)),
        "version should be dropped when the column is too narrow"
    );
}

#[test]
fn test_welcome_shows_capability_chip_and_tip() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![
        skill_command("repo-study", "Study a repo"),
        skill_command("release", "Cut a release"),
    ];
    app.mcp_configured_count = 1;
    app.welcome_tip_index = 0; // "start a line with ! to run a shell command"

    let (screen, _) = render_full_screen(&mut app, 90, 24);
    assert!(
        screen.contains("2 skills · 1 MCP"),
        "missing chip:\n{screen}"
    );
    assert!(
        screen.contains("✶ Tip"),
        "missing tip label + glyph:\n{screen}"
    );
    assert!(
        screen.contains(WELCOME_TIPS[0]),
        "missing tip text:\n{screen}"
    );
    // The bare "Ready" filler is gone.
    assert!(!screen.contains("Ready"), "stale Ready line:\n{screen}");
}

#[test]
fn test_welcome_tip_rotates_on_interval_only_while_visible() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx); // empty transcript, no overlay
    app.welcome_tip_index = 0;

    // First frame just starts the clock — no swap yet.
    assert!(!app.tick_welcome_tip(), "swapped on the first frame");
    assert_eq!(app.welcome_tip_index, 0);
    assert!(app.welcome_tip_rotated_at.is_some(), "clock not started");

    // Before the interval elapses, still no swap.
    assert!(!app.tick_welcome_tip());
    assert_eq!(app.welcome_tip_index, 0);

    // Backdate past the interval → the next tick advances and re-clocks.
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(app.tick_welcome_tip(), "no swap after the interval");
    assert_eq!(app.welcome_tip_index, 1);

    // An overlay pauses rotation and resets the clock.
    app.overlay = Overlay::Help { scroll: 0 };
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(!app.tick_welcome_tip(), "must not rotate under an overlay");
    assert_eq!(app.welcome_tip_index, 1, "index frozen while covered");
    assert!(
        app.welcome_tip_rotated_at.is_none(),
        "clock reset while covered"
    );

    // A draft pauses it too; clearing it restarts the clock.
    app.overlay = Overlay::None;
    app.draft = "hello".to_string();
    app.welcome_tip_rotated_at =
        Some(std::time::Instant::now() - WELCOME_TIP_ROTATE_INTERVAL - Duration::from_secs(1));
    assert!(!app.tick_welcome_tip(), "must not rotate while composing");
    assert_eq!(app.welcome_tip_index, 1, "index frozen while composing");
    assert!(
        app.welcome_tip_rotated_at.is_none(),
        "clock reset while composing"
    );
    app.draft.clear();
    assert!(
        !app.tick_welcome_tip(),
        "cleared draft restarts the clock, no swap"
    );
    assert!(
        app.welcome_tip_rotated_at.is_some(),
        "clock restarted after clear"
    );
}

#[test]
fn test_welcome_chip_omits_zero_and_pluralizes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Nothing configured → no chip, but the tip still shows.
    assert_eq!(app.welcome_capabilities_label(), None);

    // One skill, no MCP → singular, MCP segment omitted.
    app.skill_commands = vec![skill_command("repo-study", "Study a repo")];
    assert_eq!(app.welcome_capabilities_label().as_deref(), Some("1 skill"));

    // MCP only → skills segment omitted.
    app.skill_commands.clear();
    app.mcp_configured_count = 3;
    assert_eq!(app.welcome_capabilities_label().as_deref(), Some("3 MCP"));
}

#[test]
fn test_notice_display_prefixes_errors_only() {
    let error = (ERROR(), "boom".to_string());
    let info = (MUTED(), "ok".to_string());

    let displayed = notice_display(Some(&error)).unwrap();
    assert_eq!(displayed.0, ERROR());
    assert_eq!(displayed.1.as_ref(), "Error: boom");

    let displayed = notice_display(Some(&info)).unwrap();
    assert_eq!(displayed.0, MUTED());
    assert_eq!(displayed.1.as_ref(), "ok");

    assert!(notice_display(None).is_none());
}

#[test]
fn test_footer_is_single_status_row() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // Render the app, returning just the footer row — a single status line now
    // (there is no hint bar), pinned to the terminal's bottom row.
    fn footer_text(configure: impl Fn(&mut CodeTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        // A long transcript so the footer sits on the terminal's bottom rows.
        app.history.push(ChatMessage {
            model: None,
            role: "assistant".to_string(),
            content: (0..40)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            reasoning_content: None,
            attachments: vec![],
        });
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        // The footer is a single row (the status line) at the terminal's bottom.
        let mut text = String::new();
        for x in 0..80u16 {
            text.push_str(buf[(x, 11)].symbol());
        }
        text
    }

    // The whole rendered screen as one string, for state shown outside the bottom
    // hint-bar row (e.g. the composer-rule mode badge).
    fn full_screen(configure: impl Fn(&mut CodeTuiApp)) -> String {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut app = make_test_app(tx, rx);
        configure(&mut app);
        let mut terminal = Terminal::new(TestBackend::new(80, 12)).unwrap();
        terminal
            .draw(|frame| {
                app.render_main(frame, frame.area());
            })
            .unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut screen = String::new();
        for y in 0..12u16 {
            for x in 0..80u16 {
                screen.push_str(buf[(x, y)].symbol());
            }
            screen.push('\n');
        }
        screen
    }

    // The footer is a single row: the status line (context meter) in every state.
    let idle = footer_text(|_| {});
    assert!(idle.contains("tokens"), "idle status line: {idle:?}");
    // No hint bar — no contextual key hints ever leak into the footer.
    assert!(!idle.contains("send"), "no idle key hints: {idle:?}");
    assert!(
        !footer_text(|a| a.sending = true).contains("interrupt"),
        "no sending hint bar — footer stays status-only"
    );
    // The mode badge rides the composer rule, not the footer, and every mode is
    // shown so the current state + its cycle key stay discoverable.
    // Matched by each badge's unique glyph: a wide glyph splits across buffer
    // cells (breaking adjacent-text matches) and the rotating welcome tip can
    // contain the bare mode words.
    assert!(
        full_screen(|a| a.agent_auto_approve = true).contains('⚡'),
        "auto badge on composer rule"
    );
    assert!(
        full_screen(|a| a.agent_auto_approve = false).contains("normal (shift+tab)"),
        "normal mode shown on composer rule (discoverable)"
    );
    assert!(
        full_screen(|a| a.agent_review_edits = true).contains('✎'),
        "review badge on composer rule"
    );
    assert!(
        !footer_text(|a| a.agent_auto_approve = true).contains('⚡'),
        "the mode badge is not in the footer"
    );
    // Effort tier (bare value) shows on the status line only when thinking is on.
    assert!(
        footer_text(|a| {
            a.thinking_enabled = true;
            a.model_supports_thinking = true;
            a.model_reasoning_efforts = vec!["high".to_string()];
        })
        .contains("high"),
        "effort tier shown when thinking on"
    );
    assert!(
        !footer_text(|a| {
            a.thinking_enabled = false;
            a.model_supports_thinking = true;
            a.model_reasoning_efforts = vec!["high".to_string()];
        })
        .contains("high"),
        "effort tier hidden when thinking off"
    );
}

#[test]
fn test_inline_status_stays_in_transcript_across_phases() {
    // The processing status renders in the transcript in every phase, so its
    // position never jumps to the footer when the reply starts streaming.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;

    // Model-compute phase: the Thinking heartbeat shows, no streamed text yet —
    // and no round tokens generated, so no "0 tokens" noise.
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("Thinking"), "compute phase: {plain:?}");
    assert!(!plain.contains("tokens"), "no token tail at 0: {plain:?}");

    // Streaming the reply reads "Working"; once output flows the tail shows it.
    app.pending_response = "streaming the answer".to_string();
    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(plain.contains("streaming the answer"));
    assert!(plain.contains("Working"), "streaming phase: {plain:?}");
    assert!(plain.contains("tokens"));
}

#[test]
fn test_display_cwd_shows_real_dir_for_agent_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // make_test_app uses a plain API key → agent-capable → show the real dir.
    assert!(app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
}

#[test]
fn test_display_cwd_shows_real_dir_for_cursor_keys() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.key = ApiKey::new_with_protocol(
        "cursor".to_string(),
        String::new(),
        "cursor".to_string(),
        None,
        String::new(),
    );
    app.cwd = "/sandbox".to_string();
    app.real_cwd = "/home/me/project".to_string();
    // cursor-agent now runs as a real agent in the launch dir (not the in-process
    // engine, so still not `agent_capable`), so the footer shows the real dir.
    assert!(app.key.is_cursor_acp());
    assert!(!app.agent_capable());
    assert_eq!(app.display_cwd(), "/home/me/project");
}

#[test]
fn test_subagents_render_individually_not_coalesced() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // Two subagents dispatched in parallel arrive as adjacent tool_calls. Unlike
    // the many tiny cursor steps coalescing is meant to fold, each subagent is a
    // distinct unit of work and must keep its task visible — never an opaque
    // `subagent ×2`.
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Review the chat API endpoint for gaps"}),
        vec![],
        None,
    );
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"agent": "reviewer", "task": "Audit the auth flow"}),
        vec![],
        None,
    );
    app.apply_agent_tool_result("## Findings\nfirst\nsecond".to_string());

    let plain = app.build_transcript().plain_lines.join("\n");
    assert!(
        !plain.contains("subagent ×"),
        "subagents must not coalesce: {plain}"
    );
    // The word "subagent" is jargon and must not appear — each delegated task is
    // shown directly after the arrow, with no `subagent(...)` wrapper.
    assert!(
        !plain.contains("subagent"),
        "the word 'subagent' must not be shown: {plain}"
    );
    assert!(
        plain.contains("→ Review the chat API endpoint for gaps"),
        "first delegated task missing: {plain}"
    );
    // A named delegation leads with the profile name so the row attributes it.
    assert!(
        plain.contains("→ reviewer — Audit the auth flow"),
        "second delegated task missing: {plain}"
    );
    // The result previews the report's first line after the fold toggle — not a
    // bare count alone that says nothing about what the subagent found.
    assert!(
        plain.contains("▸ +3 lines · ## Findings"),
        "subagent result preview missing: {plain}"
    );
}

#[test]
fn test_delegating_spinner_label_drops_subagent_word() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.sending = true;
    app.apply_agent_tool_call(
        None,
        "subagent".to_string(),
        serde_json::json!({"task": "Audit the auth flow"}),
        vec![],
        None,
    );
    let activity = app.current_action_label().unwrap_or_default();
    assert_eq!(
        activity, "delegating",
        "action line should read 'delegating'"
    );
    assert!(
        !activity.contains("subagent"),
        "action line leaked 'subagent'"
    );
}

#[test]
fn test_footer_shows_share_badge_only_when_sharing() {
    use crate::services::share_live::LiveShareHandle;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // No share → no badge.
    let (screen, _) = render_full_screen(&mut app, 80, 12);
    assert!(
        !screen.contains("● sharing"),
        "share badge shown without an active share:\n{screen}"
    );

    // Active share → the `● sharing` badge appears in the footer.
    app.share.handle = Some(LiveShareHandle::for_test(
        "https://s.getaivo.dev/v.html?t=ab",
    ));
    let (screen, _) = render_full_screen(&mut app, 80, 12);
    assert!(
        screen.contains("● sharing"),
        "no share badge in footer while sharing:\n{screen}"
    );
}

/// A running background job shows a `✦ N job(s)` badge in the composer rule.
#[cfg(unix)]
#[tokio::test]
async fn jobs_badge_shows_running_count() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let cwd = std::env::temp_dir();
    app.jobs.spawn("sleep 30", &cwd).unwrap();
    // The event loop caches this each tick; refresh it directly for the test.
    app.jobs_running = app.jobs.running_count();
    let line = app.composer_rule_line(120);
    let plain = plain_text_from_spans(&line.spans);
    assert!(plain.contains("✦ 1 job"), "badge missing: {plain}");
    let _ = app.jobs.kill_all().await;
}

/// With no running jobs, the composer rule carries no jobs badge (guards width math).
#[test]
fn jobs_badge_absent_when_idle() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let app = make_test_app(tx, rx);
    let line = app.composer_rule_line(120);
    let plain = plain_text_from_spans(&line.spans);
    assert!(!plain.contains("✦"), "no jobs → no badge: {plain}");
}

/// `/new` re-roots NEW background-job logs under the fresh session's artifacts dir.
#[cfg(unix)]
#[tokio::test]
async fn new_session_reroots_job_logs() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.start_new_chat();
    let new_id = app.session_id.clone();
    assert!(!new_id.is_empty(), "a new session id was minted");
    let out = app.jobs.spawn("echo hi", &std::env::temp_dir()).unwrap();
    assert!(
        out.contains(&new_id),
        "job log should be under the new session dir: {out}"
    );
    let _ = app.jobs.kill_all().await;
}

#[test]
fn test_humanize_count() {
    use super::super::shared::humanize_count;
    assert_eq!(humanize_count(0), "0");
    assert_eq!(humanize_count(999), "999");
    assert_eq!(humanize_count(1234), "1.2k");
    assert_eq!(humanize_count(12345), "12k");
}
