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
    let e = engine_with_mail(a);
    let out = e
        .send_session(&json!({"target": "bbbb", "text": "hello over there"}))
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
    let e = engine_with_mail(a);
    e.send_session(&json!({"target": "bbbb", "text": "round 1?", "reply_to": "  "}))
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
    let e = engine_with_mail(a);
    // Stringly-typed `wait` on purpose — models send that too.
    let waiter = tokio::spawn(async move {
        e.send_session(&json!({
            "target": "bbbb", "text": "answer me", "wait": "true", "timeout_ms": 10_000
        }))
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
    b.send_awaiting_reply("aaaa-sess-1", "which port?", None, None)
        .unwrap();
    let e = engine_with_mail(a);
    let out = e
        .send_session(&json!({
            "target": "bbbb", "text": "review this please",
            "wait": true, "timeout_ms": 60_000
        }))
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
}

#[tokio::test]
async fn send_session_wait_gets_the_reply() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = mail_pair(dir.path());
    let e = engine_with_mail(a);
    // Simulated peer: claims the question, replies referencing its id.
    let replier = tokio::spawn(async move {
        loop {
            if let Some(q) = b.claim_next() {
                b.send("aaaa-sess-1", "the answer is 42", Some(&q.id), None)
                    .unwrap();
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    });
    let out = e
        .send_session(&json!({
            "target": "bbbb", "text": "what is the answer?",
            "wait": true, "timeout_ms": 10_000
        }))
        .await
        .unwrap();
    replier.await.unwrap();
    assert!(out.contains("Reply from session bbbb-ses"), "{out}");
    assert!(out.contains("the answer is 42"), "{out}");
    assert!(
        out.contains("message id: "),
        "the next round's reply_to needs the reply's own id: {out}"
    );
}

#[tokio::test]
async fn send_session_wait_times_out_with_delivered_note() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _b) = mail_pair(dir.path());
    let e = engine_with_mail(a);
    let err = e
        .send_session(&json!({
            "target": "bbbb", "text": "anyone home?",
            "wait": true, "timeout_ms": 5_000
        }))
        .await
        .unwrap_err();
    assert!(err.contains("no reply"), "{err}");
    assert!(err.contains("WAS delivered"), "{err}");
}

#[tokio::test]
async fn send_session_unknown_target_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (a, _b) = mail_pair(dir.path());
    let e = engine_with_mail(a);
    let err = e
        .send_session(&json!({"target": "zzzz", "text": "x"}))
        .await
        .unwrap_err();
    assert!(err.contains("no open session"), "{err}");
    // Without a mailbox the tools refuse cleanly.
    let bare = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    assert!(bare.list_sessions_result().is_err());
}
