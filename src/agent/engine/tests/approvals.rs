use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn auto_approve_enabled_tracks_static_flag_and_live_toggle() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let client = reqwest::Client::new();
    let cwd = std::path::Path::new(".");
    let ctx = |yes, flag| TurnCtx {
        client: &client,
        serve_base: "",
        auth: None,
        cwd,
        yes,
        auto_approve_all: false,
        auto_approve: flag,
        review_edits: None,
    };
    assert!(ctx(true, None).auto_approve_enabled());
    assert!(!ctx(false, None).auto_approve_enabled());
    // The live flag flips the SAME ctx: a mid-turn Shift+Tab is seen by the running turn.
    let flag = AtomicBool::new(false);
    let live = ctx(false, Some(&flag));
    assert!(!live.auto_approve_enabled());
    flag.store(true, Ordering::Relaxed);
    assert!(
        live.auto_approve_enabled(),
        "a mid-turn toggle is seen live"
    );
}

/// `rm -rf /` prompts even with auto-approve on; the mock denies so it never runs.
#[tokio::test]
async fn catastrophic_command_prompts_even_under_auto_approve() {
    let dir = tmp();
    let bash = tool_call_sse("run_bash", json!({ "command": "rm -rf /" }));
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi {
        deny: true, // never let a real `rm -rf /` execute
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("clean up".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.ask_tools, vec!["run_bash"]);
}

/// A remote mutation prompts even under `yes` (headless `-e` baseline):
/// only auto-approve mode waives the gate. The mock denies, so nothing runs.
#[tokio::test]
async fn remote_side_effect_prompts_under_yes_without_auto_approve_mode() {
    let dir = tmp();
    let bash = tool_call_sse(
        "run_bash",
        json!({ "command": "gh repo delete acme/prod --yes" }),
    );
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // turn_ctx sets yes:true (auto-approve); deny so the delete never fires.
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("remove the prod repo".into()),
        &mut ui,
    )
    .await;

    // Prompted despite auto-approve (deny keeps the delete from running).
    assert_eq!(ui.ask_tools, vec!["run_bash"]);
}

/// "Always allow" at the remote gate grants the command *family*, so a deploy
/// loop isn't re-prompted as arguments churn — while a sibling verb still asks.
/// `flyctl` is absent on every test runner, so approved calls just exit 127.
#[tokio::test]
async fn always_allow_on_remote_mutation_covers_the_family() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![
        tool_call_sse(
            "run_bash",
            json!({ "command": "cd . && flyctl deploy --app one" }),
        ),
        tool_call_sse(
            "run_bash",
            json!({ "command": "flyctl deploy --app two --strategy canary" }),
        ),
        tool_call_sse(
            "run_bash",
            json!({ "command": "flyctl rollback --app one" }),
        ),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("ship it".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.asks, 2, "expected deploy once + rollback once");
    assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
}

/// Auto-approve mode (static `--auto-approve`) waives the remote gate, but the
/// catastrophic floor still prompts (and the mock denies it).
#[tokio::test]
async fn auto_approve_mode_waives_remote_gate_but_not_catastrophic() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![
        tool_call_sse("run_bash", json!({ "command": "flyctl deploy --now" })),
        tool_call_sse("run_bash", json!({ "command": "rm -rf /" })),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi {
        deny: true, // only the catastrophic call reaches a prompt — deny it
        ..Default::default()
    };
    let ctx = TurnCtx {
        auto_approve_all: true,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("ship it".into()), &mut ui).await;

    // The remote mutation ran unprompted; only `rm -rf /` asked.
    assert_eq!(ui.ask_tools, vec!["run_bash"]);
    assert_eq!(ui.asks, 1);
}

/// The live toggle is the same auto-approve mode: remote runs unprompted.
#[tokio::test]
async fn live_toggle_waives_remote_gate() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![
        tool_call_sse("run_bash", json!({ "command": "flyctl deploy --now" })),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    let toggle = std::sync::atomic::AtomicBool::new(true);
    let ctx = TurnCtx {
        yes: false,
        auto_approve: Some(&toggle),
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("ship it".into()), &mut ui).await;

    assert_eq!(ui.asks, 0, "auto-approve mode must not prompt for remote");
    assert_eq!(ui.tools, vec!["run_bash"]);
}

/// Contrast: a workspace-local `rm -rf ./build` isn't in the floor, so auto-approve waives it (path absent → no-op).
#[tokio::test]
async fn auto_approve_waives_workspace_local_destructive() {
    let dir = tmp();
    let bash = tool_call_sse(
        "run_bash",
        json!({ "command": "rm -rf ./build_does_not_exist" }),
    );
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("clean build dir".into()),
        &mut ui,
    )
    .await;

    assert_eq!(ui.asks, 0);
    assert!(ui.tools.contains(&"run_bash".to_string()));
}

/// A denied dangerous tool (destructive bash) doesn't run; the refusal feeds back and the next turn converges.
#[tokio::test]
async fn denied_dangerous_tool_does_not_run() {
    let dir = tmp();
    let sentinel = dir.join("RAN");
    // `rm -rf` makes this dangerous → gated; if it ran it would touch RAN.
    let cmd = format!("rm -rf zzz_absent && touch {}", sentinel.display());
    let bash = tool_call_sse("run_bash", json!({ "command": cmd }));
    let port = spawn_sse_sequence(vec![bash, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

    assert_eq!(ui.tools, vec!["run_bash"]);
    assert!(!sentinel.exists(), "denied command still ran");
}

/// "Always allow" remembers the exact command, not the whole tool — a different
/// destructive command prompts again. Unix-only (uses `rm -rf … && touch …`); the logic is platform-agnostic.
#[cfg(unix)]
#[tokio::test]
async fn always_allow_is_scoped_to_the_exact_command() {
    let dir = tmp();
    let (sa, sb) = (dir.join("RAN_A"), dir.join("RAN_B"));
    let cmd_a = format!("rm -rf zzz_a && touch {}", sa.display());
    let cmd_b = format!("rm -rf zzz_b && touch {}", sb.display());
    // Steps in one turn: A, A again, a different B, then text.
    let port = spawn_sse_sequence(vec![
        tool_call_sse("run_bash", json!({ "command": cmd_a })),
        tool_call_sse("run_bash", json!({ "command": cmd_a })),
        tool_call_sse("run_bash", json!({ "command": cmd_b })),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

    // A prompts once (repeat reuses the scope); B is different → two asks total.
    assert_eq!(ui.asks, 2, "expected A once + B once");
    assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
    assert!(sa.exists(), "command A did not run");
    assert!(sb.exists(), "command B did not run");
}

/// "Always allow" on a subcommand-style command (`git reset`) covers the whole
/// family for the session, so a varied re-run isn't re-prompted — while a
/// different subcommand still asks. (Commands are no-ops in a non-repo tmp dir.)
#[tokio::test]
async fn always_allow_broadens_a_subcommand_family_but_not_siblings() {
    let dir = tmp();
    // All destructive (so they prompt); `git` is a subcommand tool (so it broadens).
    let port = spawn_sse_sequence(vec![
        tool_call_sse("run_bash", json!({ "command": "git reset --hard HEAD~1" })),
        tool_call_sse("run_bash", json!({ "command": "git reset --hard HEAD~2" })),
        tool_call_sse("run_bash", json!({ "command": "git clean -fd" })),
        FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    let mut ui = CapturingUi {
        always_allow: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("clean up".into()), &mut ui).await;

    // `git reset` approved once covers the second reset; `git clean` is a different
    // subcommand → one more ask. Two asks total, all three commands attempted.
    assert_eq!(ui.asks, 2, "reset once (family reused) + clean once");
    assert_eq!(ui.tools, vec!["run_bash", "run_bash", "run_bash"]);
}

#[test]
fn write_clobbers_unread_only_flags_blind_overwrites() {
    let dir = tmp();
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    // A new file is not a clobber.
    assert!(!engine.write_clobbers_unread("write_file", &json!({"path":"new.txt"}), &dir));
    // An existing file the model never touched IS a blind clobber.
    std::fs::write(dir.join("exists.txt"), "old").unwrap();
    assert!(engine.write_clobbers_unread("write_file", &json!({"path":"exists.txt"}), &dir));
    // Once read, overwriting it is fine.
    engine.record_touched_file("read_file", &json!({"path":"exists.txt"}));
    assert!(!engine.write_clobbers_unread("write_file", &json!({"path":"exists.txt"}), &dir));
    // edit_file / multi_edit are never blind (they read to match).
    assert!(!engine.write_clobbers_unread("edit_file", &json!({"path":"exists.txt"}), &dir));
}

/// A `write_file` overwriting a pre-existing unread file is gated; denying leaves it intact.
#[tokio::test]
async fn blind_overwrite_of_existing_file_is_gated() {
    let dir = tmp();
    std::fs::write(dir.join("precious.txt"), "USER DATA").unwrap();
    let sse = tool_call_sse(
        "write_file",
        json!({"path":"precious.txt","content":"CLOBBERED"}),
    );
    let port = spawn_sse_sequence(vec![sse, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("overwrite it".into()), &mut ui).await;

    assert_eq!(ui.asks, 1, "a blind overwrite should prompt");
    assert_eq!(
        std::fs::read_to_string(dir.join("precious.txt")).unwrap(),
        "USER DATA",
        "denied overwrite must leave the file untouched"
    );
}

/// A safe mutating tool (in-project write) runs WITHOUT a prompt even when the UI would deny — only dangerous actions are gated.
#[tokio::test]
async fn safe_tool_runs_without_prompt() {
    let dir = tmp();
    let port = spawn_sse_sequence(vec![WRITE_TOOL_SSE.to_string(), FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(
        &dir.display().to_string(),
        "m",
        "2026-01-01",
        &[],
        &[],
        0,
        0,
    );
    // deny=true would block anything that asked — but a safe write never asks.
    let mut ui = CapturingUi {
        deny: true,
        ..Default::default()
    };
    let ctx = TurnCtx {
        yes: false,
        auto_approve_all: false,
        ..turn_ctx(&client, &base, &dir)
    };
    run_session(&mut engine, &ctx, Some("write out.txt".into()), &mut ui).await;

    assert_eq!(ui.tools, vec!["write_file"]);
    assert!(dir.join("out.txt").exists(), "safe write was blocked");
}
