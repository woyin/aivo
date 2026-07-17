use super::super::*;
use super::helpers::*;
use crate::agent::request::content_str;
use serde_json::json;

#[test]
fn push_user_content_never_makes_consecutive_user_turns() {
    let dir = tmp();
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    engine.push_user_content(Value::String("first".into()));
    engine.push_user_content(json!([
        {"type": "text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]));
    let users: Vec<_> = engine
        .messages
        .iter()
        .filter(|m| m["role"] == "user")
        .collect();
    assert_eq!(
        users.len(),
        1,
        "the image turn must fold into the trailing user turn"
    );
    let parts = users[0]["content"].as_array().unwrap();
    assert!(parts.iter().any(|p| p["type"] == "image_url"));
    assert!(
        parts
            .iter()
            .any(|p| p.get("text").and_then(|t| t.as_str()) == Some("first"))
    );
}

#[test]
fn reset_keeps_only_the_system_prompt() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.messages.push(json!({"role":"user","content":"hi"}));
    engine
        .messages
        .push(json!({"role":"assistant","content":"yo"}));
    engine.reset();
    assert_eq!(engine.messages.len(), 1);
    assert_eq!(role(&engine.messages[0]), "system");
}

#[test]
fn seed_history_carries_user_and_assistant_only() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.seed_history(vec![
        ("user".to_string(), "hi".to_string()),
        ("assistant".to_string(), "hello".to_string()),
        ("tool_call".to_string(), "{}".to_string()), // dropped
        ("tool_result".to_string(), "x".to_string()), // dropped
        ("user".to_string(), "next".to_string()),
    ]);
    // system + user + assistant + user (tool entries skipped)
    assert_eq!(engine.messages.len(), 4);
    assert_eq!(role(&engine.messages[0]), "system");
    assert_eq!(role(&engine.messages[1]), "user");
    assert_eq!(content_str(&engine.messages[1]), "hi");
    assert_eq!(role(&engine.messages[2]), "assistant");
    assert_eq!(role(&engine.messages[3]), "user");
    assert_eq!(content_str(&engine.messages[3]), "next");
}

/// `push_text_turn` merges into a preceding same-role plain-text message (never
/// two consecutive same-role turns — Anthropic 400s). Different roles / tool_call assistants aren't merged.
#[test]
fn push_text_turn_merges_consecutive_same_role() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.push_text_turn("user", "first".to_string());
    engine.push_text_turn("user", "second".to_string()); // merges into "first"
    engine.push_text_turn("assistant", "reply".to_string());
    engine.push_text_turn("assistant", "more".to_string()); // merges into "reply"
    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert_eq!(roles, vec!["system", "user", "assistant"]);
    assert_eq!(content_str(&engine.messages[1]), "first\n\nsecond");
    assert_eq!(content_str(&engine.messages[2]), "reply\n\nmore");

    // A tool_calls-bearing assistant is not a plain-text turn → never merged.
    let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e2.messages.push(json!({
        "role":"assistant",
        "tool_calls":[{"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]
    }));
    e2.push_text_turn("assistant", "text".to_string());
    assert_eq!(
        e2.messages.len(),
        3,
        "must not merge into a tool_calls assistant"
    );
}

/// Seeding a history with two adjacent user turns (cancelled + next) must not reproduce them as consecutive user messages.
#[test]
fn seed_history_merges_adjacent_user_turns() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.seed_history(vec![
        ("user".to_string(), "cancelled task".to_string()),
        ("user".to_string(), "real task".to_string()),
    ]);
    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert!(
        !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
        "consecutive user after seeding: {roles:?}"
    );
    assert_eq!(
        content_str(&engine.messages[1]),
        "cancelled task\n\nreal task"
    );
}

/// Seeding drops leading assistant turns so the conversation opens with a user message (Anthropic rejects assistant-first).
#[test]
fn seed_history_drops_leading_assistant_for_user_first() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.seed_history(vec![
        ("assistant".to_string(), "(mid-exchange reply)".to_string()),
        ("user".to_string(), "real question".to_string()),
        ("assistant".to_string(), "answer".to_string()),
    ]);
    // system + user + assistant — the leading assistant turn was dropped.
    assert_eq!(engine.messages.len(), 3);
    assert_eq!(role(&engine.messages[1]), "user");
    assert_eq!(content_str(&engine.messages[1]), "real question");
    assert_eq!(role(&engine.messages[2]), "assistant");

    // All-assistant history seeds nothing (a following user turn opens it).
    let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e2.seed_history(vec![("assistant".to_string(), "orphan".to_string())]);
    assert_eq!(e2.messages.len(), 1); // system only
}

#[test]
fn first_party_branding_is_opt_in_idempotent_and_durable() {
    let mut e = AgentEngine::new("/tmp", "aivo/starter", "", &[], &[], 0, 0);
    // Off by default: the base prompt never names the model/provider (BYOK stays honest).
    assert!(!system_content(&e).contains("aivo's own assistant"));

    e.set_first_party();
    let branded = system_content(&e);
    assert!(branded.contains("aivo's own assistant"));
    assert!(branded.contains("aivo models"));
    // Must mutate in place, not push — `restore_conversation` no-ops unless `messages.len() == 1`.
    assert_eq!(e.messages.len(), 1);

    // Idempotent: a rebuild/resume re-runs it; a double call doesn't duplicate.
    e.set_first_party();
    assert_eq!(
        system_content(&e).matches("aivo's own assistant").count(),
        1
    );

    // Survives `reset()` (which keeps only the system message).
    e.reset();
    assert!(system_content(&e).contains("aivo's own assistant"));
}

#[test]
fn confirm_before_build_is_opt_in_idempotent_and_durable() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    // Off by default (headless/-e and sub-agents must build without asking).
    assert!(!system_content(&e).contains("before you BUILD something substantial"));

    e.set_confirm_before_build();
    let gated = system_content(&e);
    assert!(gated.contains("before you BUILD something substantial"));
    // Carve-outs: small edits pass through, go-aheads skip, refinements re-ask.
    assert!(gated.contains("small single-file edits"));
    assert!(gated.contains("work autonomously"));
    assert!(gated.contains("plan REVISION, not approval"));
    // Mutate in place (single-system-message invariant).
    assert_eq!(e.messages.len(), 1);

    // Idempotent: a rebuild/resume re-runs it; a double call doesn't duplicate.
    e.set_confirm_before_build();
    assert_eq!(
        system_content(&e)
            .matches("before you BUILD something substantial")
            .count(),
        1
    );

    // Survives `reset()` (keeps only the system message).
    e.reset();
    assert!(system_content(&e).contains("before you BUILD something substantial"));
}

#[test]
fn record_touched_file_dedups_orders_and_filters_tools() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.record_touched_file("read_file", &json!({"path":"a.rs"}));
    engine.record_touched_file("read_file", &json!({"path":"a.rs"})); // dup
    engine.record_touched_file("write_file", &json!({"path":"b.rs"}));
    engine.record_touched_file("run_bash", &json!({"command":"ls"})); // not a file tool
    engine.record_touched_file("grep", &json!({"path":"c.rs"})); // tracked? no
    assert_eq!(
        engine.touched_files,
        vec!["a.rs".to_string(), "b.rs".to_string()]
    );
}

#[test]
fn record_touched_file_caps_and_evicts_oldest() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    for i in 0..(MAX_TOUCHED_FILES + 5) {
        engine.record_touched_file("read_file", &json!({ "path": format!("f{i}.rs") }));
    }
    assert_eq!(engine.touched_files.len(), MAX_TOUCHED_FILES);
    assert!(!engine.touched_files.contains(&"f0.rs".to_string())); // oldest evicted
    assert!(
        engine
            .touched_files
            .contains(&format!("f{}.rs", MAX_TOUCHED_FILES + 4))
    );
}

/// A dangling-tool_calls assistant (interrupted mid-tool) is repaired before the next turn: each unanswered call id gets a synthetic result.
#[test]
fn repair_interrupted_tail_answers_dangling_calls() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine
        .messages
        .push(json!({"role":"user","content":"do it"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[
            {"id":"c1","type":"function","function":{"name":"run_bash","arguments":"{}"}},
            {"id":"c2","type":"function","function":{"name":"read_file","arguments":"{}"}}
        ]
    }));
    // Only the first call's result made it in before the interrupt.
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));

    engine.repair_interrupted_tail();

    // c2 now has a result, sitting in the contiguous tool run after the call.
    let tool_ids: Vec<&str> = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .filter_map(|m| m["tool_call_id"].as_str())
        .collect();
    assert_eq!(tool_ids, vec!["c1", "c2"]);
    // A short assistant turn caps the synthesized results so the next user turn alternates (bare user after them → 2nd consecutive user, Anthropic 400).
    let last = engine.messages.last().unwrap();
    assert_eq!(role(last), "assistant");
    assert_eq!(last["content"], "[interrupted]");

    // Idempotent: a fully-answered + capped tail is left untouched.
    let len = engine.messages.len();
    engine.repair_interrupted_tail();
    assert_eq!(engine.messages.len(), len);

    // With a real next turn appended, the synthetic assistant sits between the results and the user, so roles alternate.
    engine
        .messages
        .push(json!({"role":"user","content":"next"}));
    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert_eq!(roles.last(), Some(&"user"));
    assert_eq!(
        roles[roles.len() - 2],
        "assistant",
        "tool results must be capped by an assistant before the next user: {roles:?}"
    );
}

/// Repaired-tail invariant: no assistant `tool_calls` is left without a matching `tool` result for every call id in the following run.
#[test]
fn repair_interrupted_tail_leaves_no_unanswered_tool_use() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine
        .messages
        .push(json!({"role":"user","content":"do it"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[
            {"id":"a","type":"function","function":{"name":"read_file","arguments":"{}"}},
            {"id":"b","type":"function","function":{"name":"glob","arguments":"{}"}},
            {"id":"c","type":"function","function":{"name":"grep","arguments":"{}"}}
        ]
    }));
    // None of the three results landed before the interrupt.
    engine.repair_interrupted_tail();

    // Every assistant-with-tool_calls is fully answered: each call id appears in
    // the contiguous `tool` run immediately following the assistant.
    for (idx, m) in engine.messages.iter().enumerate() {
        let Some(calls) = m
            .get("tool_calls")
            .and_then(|t| t.as_array())
            .filter(|a| !a.is_empty())
        else {
            continue;
        };
        let answered: HashSet<&str> = engine.messages[idx + 1..]
            .iter()
            .take_while(|m| role(m) == "tool")
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        for call in calls {
            let id = call["id"].as_str().unwrap();
            assert!(
                answered.contains(id),
                "dangling tool_use {id} left unanswered: {:?}",
                engine.messages
            );
        }
    }
}

/// A clean transcript (every tool_use answered AND capped by a following assistant) must be left byte-for-byte unchanged.
#[test]
fn repair_interrupted_tail_leaves_clean_transcript_unchanged() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine
        .messages
        .push(json!({"role":"user","content":"do it"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[
            {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}
        ]
    }));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));
    // An assistant already follows the result → already capped, so the alternation guard must NOT add a second cap.
    engine
        .messages
        .push(json!({"role":"assistant","content":"all done"}));

    let before = engine.messages.clone();
    engine.repair_interrupted_tail();
    assert_eq!(
        engine.messages, before,
        "clean (answered + capped) transcript was modified"
    );
}

/// Esc before anything streamed un-sends the turn from the engine too: the TUI
/// returned the text to the composer, so a resend must not merge with the stale
/// copy ("hello" + "hi" → "hello\n\nhi" on the wire).
#[test]
fn unsend_last_user_turn_removes_fresh_turn_and_checkpoint() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.begin_user_turn(Value::String("hello".into()), "hello".to_string());
    assert_eq!(engine.messages.len(), 2);
    assert_eq!(engine.checkpoints.len(), 1);

    engine.unsend_last_user_turn();
    assert_eq!(engine.messages.len(), 1, "the bare user turn is popped");
    assert_eq!(engine.checkpoints.len(), 0, "its checkpoint goes with it");

    // The resend carries only the new text — the cancelled copy is gone.
    engine.begin_user_turn(Value::String("hi".into()), "hi".to_string());
    assert_eq!(content_str(&engine.messages[1]), "hi");

    // A second un-send without a new turn is a no-op (record already consumed).
    engine
        .messages
        .push(json!({"role":"assistant","content":"yo"}));
    engine.unsend_last_user_turn();
    engine.unsend_last_user_turn();
    assert_eq!(engine.messages.len(), 3);
}

/// A turn merged into a prior interrupted user turn un-sends only its own text:
/// the prior tail (still shown in the transcript) is restored verbatim, and the
/// reused checkpoint stays.
#[test]
fn unsend_last_user_turn_restores_merged_prior_tail() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    // Turn 1 interrupted after partial output: its user message stays in the engine.
    engine.begin_user_turn(Value::String("hello".into()), "hello".to_string());
    engine.begin_user_turn(Value::String("hi".into()), "hi".to_string());
    assert_eq!(content_str(&engine.messages[1]), "hello\n\nhi");
    assert_eq!(engine.checkpoints.len(), 1);

    engine.unsend_last_user_turn();
    assert_eq!(content_str(&engine.messages[1]), "hello");
    assert_eq!(engine.checkpoints.len(), 1, "turn 1's checkpoint is kept");
}

/// Once anything was recorded after the opening user message, un-send must not
/// touch the transcript (the interrupt path commits the partial turn instead).
#[test]
fn unsend_last_user_turn_noops_after_any_reply() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.begin_user_turn(Value::String("hello".into()), "hello".to_string());
    engine
        .messages
        .push(json!({"role":"assistant","content":"partial"}));

    engine.unsend_last_user_turn();
    assert_eq!(engine.messages.len(), 3, "nothing removed");
    assert_eq!(content_str(&engine.messages[1]), "hello");
    assert_eq!(engine.checkpoints.len(), 1);
}

// ── named specialist sub-agents ─────────────────────────────────────────

/// Durable resume round trip: `export_conversation` drops the system prompt but keeps
/// exact tool-call/result pairing, and `restore_conversation` rebuilds it after a fresh
/// system prompt. Restore is a no-op once non-fresh.
#[test]
fn export_then_restore_round_trips_tool_history() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.messages
        .push(json!({"role": "user", "content": "read it"}));
    e.messages.push(json!({
        "role": "assistant",
        "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}
        }]
    }));
    e.messages
        .push(json!({"role": "tool", "tool_call_id": "call_1", "content": "FILE BODY"}));
    e.messages
        .push(json!({"role": "assistant", "content": "done"}));

    let convo = e.export_conversation();
    assert_eq!(convo.len(), 4, "system prompt is excluded");
    assert_eq!(convo[1]["tool_calls"][0]["id"], "call_1");

    let mut restored = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    restored.restore_conversation(convo.clone());
    assert_eq!(restored.messages.len(), 5, "fresh system prompt + 4 turns");
    // Tool-call id and its matching result survive exactly (the lost-on-resume bug).
    assert_eq!(restored.messages[2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(restored.messages[3]["tool_call_id"], "call_1");
    assert_eq!(restored.messages[3]["content"], "FILE BODY");

    // Restoring into a non-fresh engine is a no-op (guards double-restore).
    restored.restore_conversation(convo);
    assert_eq!(restored.messages.len(), 5);
}

// --- /rewind: tree checkpoints ---
// (Git file-revert is covered exhaustively in `agent::checkpoint`; these
// exercise the engine's truncation + mapping + store wiring.)
