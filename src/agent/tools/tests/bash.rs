use super::super::*;
use super::helpers::*;
use serde_json::json;

#[tokio::test]
async fn run_bash_captures_output_and_exit() {
    let dir = tmp();
    let ok = run_bash(&json!({"command":"echo hi"}), &dir).await.unwrap();
    assert!(ok.contains("hi"));
    let bad = run_bash(&json!({"command":"exit 3"}), &dir).await.unwrap();
    assert!(bad.contains("[exit 3]"));
}

/// The first chunk arrives before the command completes; result unchanged.
#[cfg(unix)]
#[tokio::test]
async fn run_bash_streams_progress_before_completion() {
    let dir = tmp();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let args = json!({"command":"echo first; sleep 2; echo second"});
    let fut = run_bash_confined(&args, &dir, Some(tx));
    let mut fut = std::pin::pin!(fut);
    let first = tokio::select! {
        _ = &mut fut => panic!("command finished before any chunk streamed"),
        chunk = rx.recv() => chunk.expect("a live chunk"),
    };
    assert!(first.contains("first"));
    let out = fut.await.result.unwrap();
    assert!(out.contains("first") && out.contains("second"));
}

#[cfg(unix)]
#[tokio::test]
async fn run_bash_spills_full_output_past_the_cap() {
    let dir = tmp();
    let out = run_bash(&json!({"command":"seq 1 3000"}), &dir)
        .await
        .unwrap();
    assert!(out.contains("truncated") && out.contains("3000"));
    let path = out
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("[full output: "))
        .and_then(|l| l.strip_suffix(']'))
        .expect("spill note present");
    let full = std::fs::read_to_string(path).unwrap();
    assert!(full.starts_with("1\n") && full.contains("\n3000"));
    let _ = std::fs::remove_file(path);
}

/// The seatbelt sandbox lets a command write inside the workspace but blocks
/// a write to the home root (not on the allowlist). Skipped when the sandbox
/// is disabled in the environment.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn sandbox_confines_writes_to_workspace() {
    if !crate::agent::sandbox::active() {
        return;
    }
    let dir = tmp();
    // In-workspace write succeeds.
    run_bash(&json!({"command":"echo hi > inside.txt"}), &dir)
        .await
        .unwrap();
    assert!(
        dir.join("inside.txt").exists(),
        "in-workspace write blocked"
    );

    // A write to a file directly in $HOME (only specific subdirs are allowed)
    // is denied — the file never appears and the model sees the EPERM hint.
    let home = crate::services::system_env::home_dir().unwrap();
    let outside = home.join(format!("aivo_sbx_test_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let out = run_bash(
        &json!({"command": format!("echo hi > '{}'", outside.display())}),
        &dir,
    )
    .await
    .unwrap();
    let existed = outside.exists();
    let _ = std::fs::remove_file(&outside);
    assert!(!existed, "out-of-workspace write was NOT blocked: {out}");
    assert!(out.contains("workspace"), "missing sandbox hint: {out}");
}

/// `run_bash_confined` flags a sandbox-blocked out-of-workspace write (and
/// emits the confinement hint), while `run_bash_unconfined` runs the same
/// command with no confinement — so the write lands and no hint appears.
/// This is the load-bearing split behind the engine's escalation flow.
#[cfg(target_os = "macos")]
#[tokio::test]
async fn confined_flags_block_then_unconfined_succeeds() {
    if !crate::agent::sandbox::active() {
        return;
    }
    let dir = tmp();
    let home = crate::services::system_env::home_dir().unwrap();
    let outside = home.join(format!("aivo_unconf_test_{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let cmd = json!({ "command": format!("echo hi > '{}'", outside.display()) });

    // Confined: blocked, flagged, file absent, hint present.
    let confined = run_bash_confined(&cmd, &dir, None).await;
    assert!(
        confined.sandbox_blocked,
        "out-of-workspace write was not flagged as blocked"
    );
    assert!(!outside.exists(), "confined write escaped the sandbox");
    assert!(confined.result.unwrap().contains("write-sandbox"));

    // Unconfined: same command, write lands, no sandbox hint.
    let out = run_bash_unconfined(&cmd, &dir, None).await.unwrap();
    let existed = outside.exists();
    let _ = std::fs::remove_file(&outside);
    assert!(existed, "unconfined write was still blocked");
    assert!(
        !out.contains("write-sandbox"),
        "unconfined output carried a sandbox hint: {out}"
    );
}

#[tokio::test]
async fn run_bash_times_out() {
    let dir = tmp();
    let err = run_bash(&json!({"command":"sleep 5","timeout":1}), &dir)
        .await
        .unwrap_err();
    assert!(err.contains("timed out"));
}

/// Sharing the TUI's pgroup lets a descendant steal /dev/tty (interface freeze).
#[cfg(unix)]
#[tokio::test]
async fn run_bash_runs_in_its_own_process_group() {
    let dir = tmp();
    // Unconfined: macOS seatbelt denies `ps`; the spawn builder is shared.
    let out = run_bash_unconfined(&json!({"command":"ps -o pgid= -p $$"}), &dir, None)
        .await
        .unwrap();
    let child_pgid: i32 = out
        .lines()
        .next()
        .and_then(|l| l.trim().parse().ok())
        .unwrap_or_else(|| panic!("unparseable pgid output: {out:?}"));
    // SAFETY: getpgrp cannot fail.
    let own_pgid = unsafe { libc::getpgrp() };
    assert_ne!(
        child_pgid, own_pgid,
        "run_bash child shares the caller's process group"
    );
}

/// Timeout must reach grandchildren, not just the direct shell.
#[cfg(unix)]
#[tokio::test]
async fn run_bash_timeout_kills_grandchildren() {
    let dir = tmp();
    let pid_file = dir.join("orphan.pid");
    let cmd = format!("sleep 30 & echo $! > '{}'; wait", pid_file.display());
    let err = run_bash(&json!({"command": cmd, "timeout": 1}), &dir)
        .await
        .unwrap_err();
    assert!(err.contains("timed out"));
    let pid: i32 = std::fs::read_to_string(&pid_file)
        .expect("grandchild pid file")
        .trim()
        .parse()
        .expect("grandchild pid");
    // SIGKILL is immediate but reaping isn't; poll until gone or zombie.
    for _ in 0..40 {
        // SAFETY: signal 0 only probes liveness.
        if unsafe { libc::kill(pid, 0) } != 0 {
            return;
        }
        let stat = std::process::Command::new("ps")
            .args(["-o", "stat=", "-p", &pid.to_string()])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        if stat.is_empty() || stat.starts_with('Z') {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("grandchild {pid} survived the timeout tree-kill");
}

#[test]
fn interactive_commands_are_refused_argument_aware() {
    // refuse:
    for cmd in [
        "vim file.txt",
        "nano",
        "/usr/bin/emacs notes.md",
        "ssh prod",
        "ssh -p 2222 user@host",
        "sudo apt update",
        "top",
        "htop",
        "watch ls",
        "tail -f server.log",
        "git rebase -i HEAD~3",
        "git add -p",
        "docker run -it ubuntu bash",
        "kubectl exec -it pod -- sh",
        "make && vim Cargo.toml",
    ] {
        assert!(
            interactive_block_reason(cmd).is_some(),
            "should refuse: {cmd}"
        );
    }
    // allow:
    for cmd in [
        "ssh prod 'systemctl status'",
        "ssh -p 22 host uptime",
        "sudo -n systemctl restart x",
        "tail -n 100 server.log",
        "top -b -n1",
        "git commit -m \"msg\"",
        "git add src/",
        "docker build -t x .",
        "cargo build --release",
        "python script.py",
        "psql -c 'select 1'",
        "ls | grep foo",
        "echo \"ssh prod\"",
    ] {
        assert!(
            interactive_block_reason(cmd).is_none(),
            "should allow: {cmd}"
        );
    }
}

#[tokio::test]
async fn run_bash_refuses_interactive_command_before_spawning() {
    let dir = tmp();
    let err = run_bash(&json!({"command":"vim notes.txt"}), &dir)
        .await
        .unwrap_err();
    assert!(err.contains("interactive editor"), "got: {err}");
}

#[test]
fn blocker_messages_are_audience_specific() {
    assert_eq!(
        interactive_block_reason("vim x"),
        Some(InteractiveBlocker::Editor("vim".into()))
    );
    let editor = InteractiveBlocker::Editor("vim".into());
    assert!(editor.agent_message().contains("write/edit tools"));
    assert!(editor.user_message().contains("separate terminal"));
    assert_ne!(editor.agent_message(), editor.user_message());
    // `!cmd` phrasing never says "ask the user" (the human IS the user).
    for cmd in ["ssh prod", "sudo apt update", "git rebase -i HEAD~2"] {
        let msg = interactive_block_reason(cmd).unwrap().user_message();
        assert!(!msg.contains("ask the user"), "{cmd}: {msg}");
    }
}

#[test]
fn tail_and_watch_stream_under_bang_but_agent_refuses() {
    // The agent refuses them (it can't watch the stream or press esc); `!cmd`
    // streams them live, so only its refusal is relaxed.
    for cmd in ["tail -f log", "watch ls"] {
        assert!(
            !interactive_block_reason(cmd).unwrap().blocks_bang_cmd(),
            "{cmd}"
        );
    }
    for cmd in ["vim x", "top", "ssh host", "docker run -it ubuntu bash"] {
        assert!(
            interactive_block_reason(cmd).unwrap().blocks_bang_cmd(),
            "{cmd}"
        );
    }
}
