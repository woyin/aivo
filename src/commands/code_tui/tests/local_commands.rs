use super::super::*;
use super::helpers::*;

#[test]
fn test_prepare_submit_action_bang_runs_shell() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!ls -la".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Shell(cmd)) if cmd == "ls -la"
    ));
}

#[test]
fn test_prepare_submit_action_double_bang_escapes_to_literal() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!!important".to_string();

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(text)) if text == "!important"
    ));
}

#[test]
fn test_prepare_submit_action_bare_bang_errors() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft = "!   ".to_string();

    assert!(app.prepare_submit_action().is_err());
}

#[test]
fn test_prepare_submit_action_interactive_bang_is_refused() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    for draft in ["!vim notes.txt", "!make && vim Cargo.toml", "!top"] {
        app.draft = draft.to_string();
        // `SubmitAction` isn't `Debug`, so match rather than `unwrap_err`.
        match app.prepare_submit_action() {
            Err(err) => {
                let err = err.to_string();
                assert!(
                    err.contains("separate terminal") || err.contains("ps aux"),
                    "{draft}: {err}"
                );
            }
            Ok(_) => panic!("{draft} should be refused"),
        }
    }
    // A non-interactive command in the same family still runs.
    app.draft = "!git add src/".to_string();
    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Shell(cmd)) if cmd == "git add src/"
    ));
    // `tail -f`/`watch` stream live under the PTY (esc stops them), so `!cmd` runs
    // them even though the agent's `run_bash` refuses them.
    for draft in ["!tail -f server.log", "!watch ls"] {
        app.draft = draft.to_string();
        assert!(
            matches!(
                app.prepare_submit_action().unwrap(),
                Some(SubmitAction::Shell(_))
            ),
            "{draft} should run under !cmd"
        );
    }
}

// Unix-only: drives the `!cmd` PTY with POSIX commands (`printf`) that the
// Windows shell (PowerShell) doesn't provide, and a PTY read that blocks until
// the child exits would stall the tokio runtime drop on Windows. The `!cmd`
// feature itself is cross-platform; only this Unix-command assertion is gated.
#[cfg(unix)]
#[tokio::test]
async fn test_run_local_command_is_display_only() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf hi".to_string());
    run_local_command_to_completion(&mut app).await;

    // A transcript step is recorded for display once the run finishes.
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"command\":\"printf hi\""));
    assert!(step.content.contains("hi"));

    // It is purely local: the `local_command` role is excluded from the model
    // context (only user/assistant turns are sent), so nothing reaches the server.
    let sent_to_model = app
        .history
        .iter()
        .filter(|m| m.role == "user" || m.role == "assistant")
        .count();
    assert_eq!(sent_to_model, 0);
}

// Unix-only: see `test_run_local_command_is_display_only` — POSIX `printf`
// through the PTY isn't portable to the Windows shell.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("printf 'a\\nb\\nc\\n'".to_string());
    // The run is live (not yet in history) right after starting.
    assert!(app.local_command.is_some());
    assert!(app.history.is_empty());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once, with the streamed output and a zero exit code.
    assert!(app.local_command.is_none());
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[cfg(unix)]
#[tokio::test]
async fn test_local_command_neutralizes_pager() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // A pager-spawning command (`git diff`) would launch `less` under the PTY and
    // hang waiting for a keypress we never send. The spawn env disables pagers, so
    // the child sees PAGER/GIT_PAGER=cat — verify that reaches it through the PTY.
    app.start_local_command("echo \"p=[$PAGER] g=[$GIT_PAGER]\"".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert!(
        step.content.contains("p=[cat] g=[cat]"),
        "pager env not injected into the PTY child: {}",
        step.content
    );
}

// Unix-only: `yes` (the infinite-flood command this caps) has no PowerShell
// equivalent, and the PTY drive is Unix-oriented like the sibling tests above.
#[cfg(unix)]
#[tokio::test]
async fn test_local_command_caps_huge_output() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // `yes` floods forever — the reader must cap, kill it, and still finish.
    app.start_local_command("yes aivo".to_string());
    run_local_command_to_completion(&mut app).await;

    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains("\"truncated\":true"));
    let stdout = serde_json::from_str::<serde_json::Value>(&step.content)
        .unwrap()
        .get("stdout")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .lines()
        .count();
    // Bounded by the capture cap, not unbounded.
    assert!(stdout <= 1000, "captured {stdout} lines, expected ≤ 1000");

    // A run we killed at the cap must NOT render a scary `[exited -1]` — the
    // "truncated" note explains the stop; the SIGKILL status is ours, not `yes`'s.
    let mut block = Vec::new();
    render_local_command(&mut block, &step.content, OutputView::Collapsed);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !rendered.contains("[exited"),
        "truncated run should not show an exit code:\n{rendered}"
    );
    assert!(
        rendered.contains("truncated"),
        "truncated run should show the truncated note:\n{rendered}"
    );
}

#[tokio::test]
async fn test_local_command_full_output_kept_for_inline_expand() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // 250 output lines: past the 40-line display cap AND the persisted preview.
    let full: String = (1..=250).map(|i| format!("{i}\n")).collect();
    let total =
        app.record_local_output("seq 250".to_string(), full, String::new(), 0, false, false);
    assert_eq!(total, 250);
    let idx = app.history.len() - 1;

    // The committed transcript entry keeps only a bounded preview…
    let step = app.history.last().expect("history entry");
    let decoded: serde_json::Value = serde_json::from_str(&step.content).unwrap();
    let persisted = decoded["stdout"].as_str().unwrap().lines().count();
    assert!(
        persisted <= MAX_PERSISTED_OUTPUT_LINES,
        "persisted {persisted} lines, expected ≤ {MAX_PERSISTED_OUTPUT_LINES}"
    );
    // …but records the TRUE total, so the transcript's "+N more" stays honest.
    assert_eq!(decoded["total_lines"].as_u64(), Some(250));

    // The full output is retained in memory keyed by the entry's history index (the
    // source an expanded block renders from), never persisted into history.
    let kept = app.local_outputs.get(&idx).expect("full output retained");
    assert_eq!(kept.stdout.lines().count(), 250);
}

#[test]
fn test_render_local_command_marker_counts_total_lines() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let _app = make_test_app(tx, rx);
    // A committed entry stores a small preview but the true total in `total_lines`.
    let content = serde_json::json!({
        "command": "find .",
        "stdout": "a\nb\nc\n",
        "stderr": "",
        "exit_code": 0,
        "total_lines": 41243,
    })
    .to_string();
    let mut block = Vec::new();
    render_local_command(&mut block, &content, OutputView::Collapsed);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    // Marker reflects the true total (41243 − 3 shown), not the 3 persisted lines.
    assert!(
        rendered.contains("+41240 more lines"),
        "marker should count the true total:\n{rendered}"
    );
}

#[test]
fn test_local_command_long_line_wraps_in_full() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long = "./target/release/.fingerprint/typenum-2abcdef0123456789-very-long/lib-typenum.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let width: u16 = 58;
    let body = app.build_transcript_history_body(width);
    let wrapped = wrap_transcript(&body.lines, &body.bar_colors, width);

    // No row overflows the width (so ratatui's wrap-OFF render can't clip)…
    for row in &wrapped.rows {
        assert!(
            row_display_width(row) <= width,
            "row exceeds width: {row:?}"
        );
    }
    // …the long path is shown IN FULL — wrapped onto an extra row, not truncated.
    // Stitch every row (the path has no spaces, so dropping spaces rejoins it).
    let stitched: String = wrapped.rows.iter().map(|r| r.replace(' ', "")).collect();
    assert!(
        stitched.contains(long),
        "long path not shown in full across wrapped rows:\n{stitched}"
    );
    assert!(
        !stitched.contains('…'),
        "output should not be per-line truncated with an ellipsis"
    );
    let path_rows = wrapped
        .rows
        .iter()
        .filter(|r| r.contains("target") || r.contains("typenum"))
        .count();
    assert!(
        path_rows >= 2,
        "the long line should wrap onto multiple rows"
    );
}

#[test]
fn test_render_main_local_command_no_clip() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let long =
        "./target/release/.fingerprint/typenum-2ebc5dae76d28bAAAAAA/lib-typenum-bbbbbbbbbbbb.json";
    let content = serde_json::json!({
        "command": "find .",
        "stdout": format!("{long}\n"),
        "stderr": "",
        "exit_code": 0,
    })
    .to_string();
    app.history.push(ChatMessage {
        model: None,
        role: "local_command".to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    });

    let (w, h) = (60u16, 20u16);
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();

    // Through the FULL render pipeline (cache → wrap → slice → paint), the path is
    // shown in full — wrapped across rows, never truncated (`…`) nor edge-clipped
    // (which would drop the tail). The path has no spaces, so stitching the whole
    // screen with spaces and the accent-bar glyph removed rejoins it intact.
    let mut screen = String::new();
    for y in 0..h {
        for x in 0..w {
            screen.push_str(buf[(x, y)].symbol());
        }
    }
    let compact: String = screen
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '▌')
        .collect();
    assert!(
        compact.contains(long),
        "path not shown in full (clipped/truncated):\n{screen}"
    );
    assert!(
        !screen.contains('…'),
        "output should wrap in full, not truncate with an ellipsis:\n{screen}"
    );
}

/// Drive a started `!cmd` to completion: drain runtime events until the run
/// commits to history (clearing `local_command`). Bounded so a `!cmd` that never
/// finishes (the Windows ConPTY hang this guards against) fails the test instead of
/// hanging the runner forever.
#[cfg(any(unix, windows))]
async fn run_local_command_to_completion(app: &mut CodeTuiApp) {
    for _ in 0..5000 {
        app.handle_runtime_events().await.unwrap();
        if app.local_command.is_none() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    panic!("local command did not finish in time");
}

// Windows counterpart of `test_local_command_streams_then_commits`: `!cmd` runs
// through plain pipes (not ConPTY), whose output EOFs when the child exits — so the
// run must stream output AND commit (clear `local_command`). The bug this guards
// against left every `!cmd` stuck at "running…" forever. Uses a PowerShell command
// since `bare_shell` spawns PowerShell on Windows.
#[cfg(windows)]
#[tokio::test]
async fn test_local_command_pipe_streams_then_commits() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.start_local_command("Write-Output a; Write-Output b; Write-Output c".to_string());
    assert!(app.local_command.is_some());

    run_local_command_to_completion(&mut app).await;

    // Committed exactly once (run finished), with the streamed output and exit 0.
    assert!(app.local_command.is_none(), "the run must finish, not hang");
    let step = app.history.last().expect("history entry");
    assert_eq!(step.role, "local_command");
    assert!(step.content.contains('a') && step.content.contains('c'));
    assert!(step.content.contains("\"exit_code\":0"));
}

#[test]
fn test_prepare_submit_action_allows_attachment_only_message() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.draft_attachments.push(MessageAttachment {
        name: "notes.md".to_string(),
        mime_type: "text/markdown".to_string(),
        storage: AttachmentStorage::FileRef {
            path: "./notes.md".to_string(),
        },
    });

    assert!(matches!(
        app.prepare_submit_action().unwrap(),
        Some(SubmitAction::Send(input)) if input.is_empty()
    ));
}

/// A `find .`-sized capture (way past the inline cap) expands to at most
/// `MAX_EXPANDED_OUTPUT_LINES` rendered lines — bounding the O(lines) re-wrap so the
/// UI can't freeze — and notes the remainder instead of rendering it.
#[tokio::test]
async fn output_expand_caps_huge_capture_and_notes_remainder() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // 10_000 lines captured this session (held in memory, but bounded by the cap).
    let full: String = (1..=10_000).map(|i| format!("L{i:05}\n")).collect();
    app.record_local_output("find .".to_string(), full, String::new(), 0, false, false);
    let idx = app.history.len() - 1;

    // Memory retention is bounded by the inline cap, not the full 10k capture.
    let kept = app.local_outputs.get(&idx).expect("output retained");
    assert!(
        kept.stdout.lines().count() <= MAX_EXPANDED_OUTPUT_LINES,
        "retained {} lines, expected ≤ {MAX_EXPANDED_OUTPUT_LINES}",
        kept.stdout.lines().count()
    );

    app.expanded_output.insert(idx);
    let mut block = Vec::new();
    render_local_command(
        &mut block,
        &app.history[idx].content,
        OutputView::Expanded {
            full: app.local_outputs.get(&idx),
        },
    );
    // The `! command` header + at most the cap of output rows + the overflow note +
    // the `▾ collapse` toggle — never 10k rows.
    let output_rows = block.iter().filter(|l| l.plain.starts_with("  L")).count();
    assert_eq!(output_rows, MAX_EXPANDED_OUTPUT_LINES);
    let rendered: String = block
        .iter()
        .map(|l| l.plain.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        rendered.contains(&format!(
            "+{} more lines",
            10_000 - MAX_EXPANDED_OUTPUT_LINES
        )),
        "overflow note counts the un-rendered remainder:\n{rendered}"
    );
    assert!(rendered.contains("too long to show inline"));
    assert!(
        !rendered.contains("L10000"),
        "the tail is not rendered inline"
    );
}

#[test]
fn test_render_output_line_collapses_spinner_frames() {
    // The spinner writes "\r{dim frame} Fetching models..." per frame (no newline
    // between frames), then "\r\x1b[2K" + the result; the PTY hands it over as one line.
    let frame = |glyph: &str| format!("\r\u{1b}[2m{glyph}\u{1b}[0m Fetching models...");
    let mut raw = String::new();
    for glyph in ["⠋", "⠙", "⠹", "⠸", "⠼"] {
        raw.push_str(&frame(glyph));
    }
    raw.push_str("\r\u{1b}[2K✓ 2 models via aivo-starter\r"); // trailing \r = PTY's \r\n
    assert_eq!(render_output_line(&raw), "✓ 2 models via aivo-starter");
}

#[test]
fn test_render_output_line_plain_line_unchanged() {
    // Normal lines pass through; the trailing \r (PTY's \r\n) and colour SGR drop out.
    assert_eq!(render_output_line("hello world\r"), "hello world");
    assert_eq!(render_output_line("  \u{1b}[32mOK\u{1b}[0m"), "  OK");
    assert_eq!(render_output_line("plain"), "plain");
}

#[test]
fn test_render_output_line_progress_bar_keeps_final_state() {
    // A progress bar redraws in place with bare \r; only the last state should show.
    let raw = "[#   ] 25%\r[##  ] 50%\r[####] 100%";
    assert_eq!(render_output_line(raw), "[####] 100%");
}

#[test]
fn test_render_output_line_erase_in_line_drops_stale_tail() {
    // Erase-to-end (`\x1b[K`) must drop the stale tail when a shorter string overwrites.
    assert_eq!(render_output_line("longlabel\rhi\u{1b}[K"), "hi");
    // `\x1b[2K` clears the whole line.
    assert_eq!(render_output_line("spam\r\u{1b}[2Kdone"), "done");
}

#[cfg(unix)]
#[test]
fn test_pty_run_collapses_carriage_return_overwrites() {
    // End-to-end through the real PTY reader: a command that redraws one line with
    // bare \r (like the spinner) must commit only its final state, not every frame.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let shell =
        spawn_local_shell("printf '111\\r222\\r333\\n'", &std::env::temp_dir()).expect("spawn pty");
    run_local_to_completion(shell, tx);

    let mut lines = Vec::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            RuntimeEvent::LocalCommandLine { line, .. } => lines.push(line),
            RuntimeEvent::LocalCommandDone {
                exit_code,
                truncated,
            } => {
                assert_eq!(exit_code, 0, "printf should exit 0");
                assert!(!truncated, "a tiny run is not truncated");
            }
            _ => {}
        }
    }
    assert_eq!(
        lines,
        vec!["333".to_string()],
        "carriage-return overwrites collapse to the final state"
    );
}
