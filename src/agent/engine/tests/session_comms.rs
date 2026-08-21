use super::super::*;
use crate::services::session_mail::SessionMail;
use serde_json::json;

fn mail_pair(base: &std::path::Path) -> (SessionMail, SessionMail) {
    let a = SessionMail::new(base, "aaaa-sess-1");
    let b = SessionMail::new(base, "bbbb-sess-2");
    a.register(Some("/repo/a".into()), Some("model-a".into()))
        .unwrap();
    b.register(Some("/repo/b".into()), Some("model-b".into()))
        .unwrap();
    (a, b)
}

fn engine_with_mail(mail: SessionMail) -> AgentEngine {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.set_session_mail(mail);
    e
}

#[test]
fn set_session_mail_advertises_both_tools_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let count = |e: &AgentEngine, name: &str| {
        e.tools_openai
            .iter()
            .filter(|t| t["function"]["name"] == name)
            .count()
    };
    assert_eq!(count(&e, "send_session"), 0);
    e.set_session_mail(SessionMail::new(dir.path(), "s1"));
    e.set_session_mail(SessionMail::new(dir.path(), "s1"));
    assert_eq!(count(&e, "list_sessions"), 1);
    assert_eq!(count(&e, "send_session"), 1);
}

#[test]
fn list_sessions_shows_peers_not_self() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _b) = mail_pair(dir.path());
    let e = engine_with_mail(a);
    let out = e.list_sessions_result().unwrap();
    assert!(out.contains("yours: aaaa-ses"), "{out}");
    assert!(out.contains("bbbb-ses"), "{out}");
    assert!(out.contains("model-b"), "{out}");
    assert!(
        !out.contains("- aaaa-ses"),
        "own session isn't a peer row: {out}"
    );
}

#[tokio::test]
async fn send_session_fire_and_forget_delivers() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    let out = e
        .send_session(
            &json!({"target": "bbbb", "text": "hello over there"}),
            &mut ui,
        )
        .await
        .unwrap();
    assert!(out.contains("Delivered to session bbbb-ses"), "{out}");
    let got = b.claim_next().unwrap();
    assert_eq!(got.text, "hello over there");
    assert_eq!(got.from, "aaaa-sess-1");
}

/// Regression: `reply_to: ""` framed a first contact as a reply, telling the
/// receiver no answer was needed — so the sender's wait always timed out.
#[tokio::test]
async fn send_session_ignores_an_empty_reply_to() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    e.send_session(
        &json!({"target": "bbbb", "text": "round 1?", "reply_to": "  "}),
        &mut ui,
    )
    .await
    .unwrap();
    let got = b.claim_next().unwrap();
    assert_eq!(got.reply_to, None);
    assert!(
        !got.transcript_display().contains("reply from"),
        "a first message isn't a reply: {}",
        got.transcript_display()
    );
    assert!(!got.agent_frame().contains("no further reply"));
}

#[tokio::test]
async fn send_session_wait_marks_the_message_as_awaited() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    // Stringly-typed `wait` on purpose — models send that too.
    let waiter = tokio::spawn(async move {
        e.send_session(
            &json!({
                "target": "bbbb", "text": "answer me", "wait": "true", "timeout_ms": 10_000
            }),
            &mut ui,
        )
        .await
    });
    let got = loop {
        if let Some(msg) = b.claim_next() {
            break msg;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    };
    assert!(got.awaiting_reply, "the sender is blocked on this message");
    assert!(got.agent_frame().contains("BLOCKED waiting"));
    b.send("aaaa-sess-1", "ok", Some(&got.id), None).unwrap();
    assert!(waiter.await.unwrap().unwrap().contains("ok"));
}

/// A blocks on B while B is already blocked on a question to A — A's wait
/// must return B's question instead of sitting out the timeout.
#[tokio::test]
async fn send_session_wait_breaks_a_mutual_wait_by_handing_over_the_question() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let question_id = b
        .send_awaiting_reply("aaaa-sess-1", "which port?", None, None)
        .unwrap();
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    let out = e
        .send_session(
            &json!({
                "target": "bbbb", "text": "review this please",
                "wait": true, "timeout_ms": 60_000
            }),
            &mut ui,
        )
        .await
        .unwrap();
    assert!(out.contains("blocked waiting for YOUR answer"), "{out}");
    assert!(out.contains("which port?"), "{out}");
    assert!(
        out.contains("target=\"bbbb-ses\""),
        "the handed-over frame must carry reply addressing: {out}"
    );
    // A's own message is untouched — B answers it in a later turn.
    let pending = b.claim_next().unwrap();
    assert_eq!(pending.text, "review this please");
    assert_eq!(
        ui.mail_rows,
        vec!["✉ from session bbbb-ses\nwhich port?"],
        "the handed-over question surfaces as a ✉ row too"
    );
    // Its sender is blocked → a full (backstop-enforced) obligation.
    let ob = e.reply_obligation.clone().unwrap();
    assert_eq!(
        (ob.peer.as_str(), ob.msg_id.as_str(), ob.blocking),
        ("bbbb-sess-2", question_id.as_str(), true)
    );
}

#[tokio::test]
async fn send_session_wait_gets_the_reply() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    // Simulated peer: claims the question, replies referencing its id.
    let replier = tokio::spawn(async move {
        loop {
            if let Some(q) = b.claim_next() {
                let id = b
                    .send("aaaa-sess-1", "the answer is 42", Some(&q.id), None)
                    .unwrap();
                break (b, id);
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    });
    let out = e
        .send_session(
            &json!({
                "target": "bbbb", "text": "what is the answer?",
                "wait": true, "timeout_ms": 10_000
            }),
            &mut ui,
        )
        .await
        .unwrap();
    let (b, reply_id) = replier.await.unwrap();
    assert!(out.contains("Reply from session bbbb-ses"), "{out}");
    assert!(out.contains("the answer is 42"), "{out}");
    assert!(
        out.contains("message id: "),
        "the next round's reply_to needs the reply's own id: {out}"
    );
    assert_eq!(
        ui.mail_rows,
        vec!["✉ reply from session bbbb-ses\nthe answer is 42"],
        "the reply must surface as a ✉ transcript row, not only a folded tool result"
    );
    // The reply arms a soft obligation, so the next round threads to it even
    // without an explicit reply_to.
    let ob = e.reply_obligation.clone().unwrap();
    assert_eq!(
        (ob.peer.as_str(), ob.msg_id.as_str(), ob.blocking),
        ("bbbb-sess-2", reply_id.as_str(), false)
    );
    e.send_session(&json!({"target": "bbbb", "text": "round 2"}), &mut ui)
        .await
        .unwrap();
    assert_eq!(
        b.claim_next().unwrap().reply_to.as_deref(),
        Some(reply_id.as_str())
    );
}

#[tokio::test]
async fn send_session_wait_times_out_with_delivered_note() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    let err = e
        .send_session(
            &json!({
                "target": "bbbb", "text": "anyone home?",
                "wait": true, "timeout_ms": 5_000
            }),
            &mut ui,
        )
        .await
        .unwrap_err();
    assert!(err.contains("no reply"), "{err}");
    assert!(err.contains("WAS delivered"), "{err}");
}

/// The obligation forces the matching `reply_to`; only the first send consumes it.
#[tokio::test]
async fn obligation_forces_reply_to_on_sends_to_the_blocked_peer() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    e.set_reply_obligation(Some(ReplyObligation {
        peer: "bbbb-sess-2".into(),
        msg_id: "q-123".into(),
        blocking: true,
    }));
    e.send_session(
        &json!({"target": "bbbb", "text": "the answer", "reply_to": "made-up"}),
        &mut ui,
    )
    .await
    .unwrap();
    let reply = b.claim_next().unwrap();
    assert_eq!(reply.reply_to.as_deref(), Some("q-123"));
    e.send_session(
        &json!({"target": "bbbb", "text": "one more thing"}),
        &mut ui,
    )
    .await
    .unwrap();
    assert_eq!(b.claim_next().unwrap().reply_to, None);
}

/// A send to a third session must neither borrow nor consume the obligation.
#[tokio::test]
async fn obligation_ignores_sends_to_other_peers() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let c = SessionMail::new(dir.path(), "cccc-sess-3");
    c.register(None, None).unwrap();
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    e.set_reply_obligation(Some(ReplyObligation {
        peer: "cccc-sess-3".into(),
        msg_id: "q-999".into(),
        blocking: true,
    }));
    e.send_session(&json!({"target": "bbbb", "text": "unrelated"}), &mut ui)
        .await
        .unwrap();
    assert_eq!(b.claim_next().unwrap().reply_to, None);
    assert!(e.reply_obligation.is_some(), "obligation must survive");
}

/// A model that answers only in the transcript still unblocks the waiter.
#[tokio::test]
async fn unanswered_obligation_auto_delivers_the_turns_final_text() {
    let dir = super::helpers::tmp();
    let mail_dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(mail_dir.path());
    let port = super::helpers::spawn_sse_sequence(vec![super::helpers::FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_session_mail(a);
    engine.set_reply_obligation(Some(ReplyObligation {
        peer: "bbbb-sess-2".into(),
        msg_id: "q-123".into(),
        blocking: true,
    }));
    let mut ui = super::helpers::CapturingUi::default();
    run_session(
        &mut engine,
        &super::helpers::turn_ctx(&client, &base, &dir),
        Some("[mail frame]".into()),
        &mut ui,
    )
    .await;

    let reply = b.claim_next().unwrap();
    assert_eq!(reply.text, "done", "the turn's final text is the reply");
    assert_eq!(reply.reply_to.as_deref(), Some("q-123"));
    assert!(engine.reply_obligation.is_none());
    assert!(
        ui.notices.iter().any(|n| n.contains("auto-delivered")),
        "{:?}",
        ui.notices
    );
}

/// A fulfilled obligation must not get a duplicate reply from the backstop.
#[tokio::test]
async fn fulfilled_obligation_skips_the_backstop() {
    let dir = super::helpers::tmp();
    let mail_dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(mail_dir.path());
    let port = super::helpers::spawn_sse_sequence(vec![
        super::helpers::tool_call_sse(
            "send_session",
            json!({"target": "bbbb", "text": "the answer"}),
        ),
        super::helpers::FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_session_mail(a);
    engine.set_reply_obligation(Some(ReplyObligation {
        peer: "bbbb-sess-2".into(),
        msg_id: "q-123".into(),
        blocking: true,
    }));
    let mut ui = super::helpers::CapturingUi::default();
    run_session(
        &mut engine,
        &super::helpers::turn_ctx(&client, &base, &dir),
        Some("[mail frame]".into()),
        &mut ui,
    )
    .await;

    let reply = b.claim_next().unwrap();
    assert_eq!(reply.text, "the answer");
    assert_eq!(reply.reply_to.as_deref(), Some("q-123"), "id auto-filled");
    assert!(b.claim_next().is_none(), "no duplicate from the backstop");
    assert!(!ui.notices.iter().any(|n| n.contains("auto-delivered")));
}

/// Answered only in the transcript → one nudge; the follow-up send is threaded.
#[tokio::test]
async fn non_blocking_obligation_nudges_then_threads_the_send() {
    let dir = super::helpers::tmp();
    let mail_dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(mail_dir.path());
    let port = super::helpers::spawn_sse_sequence(vec![
        super::helpers::FINAL_TEXT_SSE.to_string(),
        super::helpers::tool_call_sse(
            "send_session",
            json!({"target": "bbbb", "text": "round 1 case"}),
        ),
        super::helpers::FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_session_mail(a);
    engine.set_reply_obligation(Some(ReplyObligation {
        peer: "bbbb-sess-2".into(),
        msg_id: "q-123".into(),
        blocking: false,
    }));
    let mut ui = super::helpers::CapturingUi::default();
    run_session(
        &mut engine,
        &super::helpers::turn_ctx(&client, &base, &dir),
        Some("[mail frame]".into()),
        &mut ui,
    )
    .await;

    let sent = b.claim_next().unwrap();
    assert_eq!(sent.text, "round 1 case");
    assert_eq!(
        sent.reply_to.as_deref(),
        Some("q-123"),
        "threaded by the obligation"
    );
    assert!(b.claim_next().is_none());
    assert!(
        engine.messages.iter().any(|m| m["content"]
            .as_str()
            .is_some_and(|c| c.contains("does NOT reach"))),
        "the nudge must be in the conversation"
    );
}

/// The nudge is advisory — a model that declines to send is respected.
#[tokio::test]
async fn non_blocking_obligation_never_force_delivers() {
    let dir = super::helpers::tmp();
    let mail_dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(mail_dir.path());
    let port = super::helpers::spawn_sse_sequence(vec![
        super::helpers::FINAL_TEXT_SSE.to_string(),
        super::helpers::FINAL_TEXT_SSE.to_string(),
    ]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.set_session_mail(a);
    engine.set_reply_obligation(Some(ReplyObligation {
        peer: "bbbb-sess-2".into(),
        msg_id: "q-123".into(),
        blocking: false,
    }));
    let mut ui = super::helpers::CapturingUi::default();
    run_session(
        &mut engine,
        &super::helpers::turn_ctx(&client, &base, &dir),
        Some("[mail frame]".into()),
        &mut ui,
    )
    .await;

    assert!(b.claim_next().is_none(), "nothing may be auto-sent");
    assert!(engine.reply_obligation.is_none(), "cleared at turn end");
    assert!(!ui.notices.iter().any(|n| n.contains("auto-delivered")));
}

#[tokio::test]
async fn send_session_unknown_target_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _b) = mail_pair(dir.path());
    let mut e = engine_with_mail(a);
    let mut ui = super::helpers::CapturingUi::default();
    let err = e
        .send_session(&json!({"target": "zzzz", "text": "x"}), &mut ui)
        .await
        .unwrap_err();
    assert!(err.contains("no open session"), "{err}");
    // Without a mailbox the tools refuse cleanly.
    let bare = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    assert!(bare.list_sessions_result().is_err());
}
