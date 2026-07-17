use super::super::*;
use super::helpers::*;
use crate::agent::compaction::{
    PINNED_MAX_TOKENS, SUMMARY_SYSTEM_PROMPT, SUMMARY_UPDATE_SYSTEM_PROMPT, TOOL_RESULT_CLEARED,
    find_cut,
};
use crate::agent::request::content_str;
use crate::agent::tokens::estimate_str_tokens;
use serde_json::json;

#[test]
fn find_cut_lands_on_user_boundary() {
    let m = |role: &str, content: &str| json!({"role": role, "content": content});
    let messages = vec![
        m("system", "sys"),
        m("user", "turn1"),
        m("assistant", "a1"),
        m("tool", "t1"),
        m("user", "turn2"),
        m("assistant", "a2"),
    ];
    let cut = find_cut(&messages, 1);
    assert_eq!(cut, 4);
    assert_eq!(role(&messages[cut]), "user");
}

/// Compaction folds the summary INTO the first kept user turn (not before it) so
/// roles keep alternating — Anthropic 400s otherwise, bricking the agent post-compaction.
#[test]
fn apply_compaction_folds_summary_and_keeps_roles_alternating() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    // [system, user, assistant, tool, user(=cut), assistant]
    engine
        .messages
        .push(json!({"role":"user","content":"first task"}));
    engine
        .messages
        .push(json!({"role":"assistant","content":"working"}));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"c1","content":"result"}));
    engine
        .messages
        .push(json!({"role":"user","content":"second task"}));
    engine
        .messages
        .push(json!({"role":"assistant","content":"done"}));

    engine.apply_compaction(4, "did the early work");

    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert!(
        !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
        "compaction left consecutive user messages: {roles:?}"
    );
    // Summary folded into the (former) messages[4] user turn, now at index 1.
    assert_eq!(role(&engine.messages[1]), "user");
    let folded = content_str(&engine.messages[1]);
    assert!(
        folded.contains("did the early work") && folded.contains("second task"),
        "summary not folded into the kept user turn: {folded}"
    );
    // …and its assistant reply still follows it (alternation intact).
    assert_eq!(role(&engine.messages[2]), "assistant");
}

/// The pinned working set survives a compaction verbatim, folded into the SAME kept user turn so alternation holds even with a non-empty block.
#[test]
fn pinned_plan_and_files_survive_compaction() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.plan = plan::parse_plan(&json!({"plan":[
        {"step":"scan code","status":"completed"},
        {"step":"write fix","status":"in_progress"}
    ]}))
    .unwrap();
    engine.touched_files = vec!["src/a.rs".to_string(), "src/b.rs".to_string()];
    // [system, user, assistant, tool, user(=cut), assistant]
    engine
        .messages
        .push(json!({"role":"user","content":"first task"}));
    engine
        .messages
        .push(json!({"role":"assistant","content":"working"}));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"c1","content":"result"}));
    engine
        .messages
        .push(json!({"role":"user","content":"second task"}));
    engine
        .messages
        .push(json!({"role":"assistant","content":"done"}));

    engine.apply_compaction(4, "summary body");

    let folded = content_str(&engine.messages[1]);
    assert!(folded.contains("summary body"), "{folded}");
    assert!(folded.contains("## Pinned Plan"), "{folded}");
    assert!(folded.contains("scan code") && folded.contains("write fix"));
    assert!(folded.contains("## Files touched"));
    assert!(folded.contains("src/a.rs") && folded.contains("src/b.rs"));
    // Alternation intact with a non-empty pinned block.
    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert!(
        !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
        "consecutive user after pinned compaction: {roles:?}"
    );
    assert_eq!(role(&engine.messages[2]), "assistant");
}

/// An empty working set folds byte-identically to the pre-pinning behavior
/// (no `## Pinned …` sections leak in) — guards the existing-test invariant.
#[test]
fn apply_compaction_without_working_set_adds_no_pinned_sections() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine
        .messages
        .push(json!({"role":"user","content":"only task"}));
    engine.apply_compaction(1, "sum");
    let folded = content_str(&engine.messages[1]);
    assert!(!folded.contains("## Pinned Plan"));
    assert!(!folded.contains("## Files touched"));
}

/// Compaction preserves tool_use↔tool_result pairing in the KEPT region: every
/// surviving `tool` follows an assistant `tool_calls` naming its id, and no orphan
/// tool heads the kept history (a leading tool result also 400s strict providers).
#[test]
fn apply_compaction_preserves_tool_pairing_across_cut() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    // [system, user, assistant(call x1), tool(x1), user(=cut), assistant(call y1), tool(y1)]
    engine
        .messages
        .push(json!({"role":"user","content":"first task"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[{"id":"x1","type":"function","function":{"name":"read_file","arguments":"{}"}}]
    }));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"x1","content":"early result"}));
    engine
        .messages
        .push(json!({"role":"user","content":"second task"}));
    engine.messages.push(json!({
        "role":"assistant",
        "tool_calls":[{"id":"y1","type":"function","function":{"name":"grep","arguments":"{}"}}]
    }));
    engine
        .messages
        .push(json!({"role":"tool","tool_call_id":"y1","content":"late result"}));

    // Cut at the second user turn (index 4): everything before is summarized away.
    let cut = find_cut(&engine.messages, 1);
    assert_eq!(cut, 4, "cut should land on the second user boundary");
    engine.apply_compaction(cut, "summary of the early work");

    // The early pair (x1) is gone; the kept pair (y1) survives intact.
    let ids: Vec<&str> = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .filter_map(|m| m["tool_call_id"].as_str())
        .collect();
    assert_eq!(ids, vec!["y1"], "only the kept tool result should remain");

    // No orphan tool result: every surviving `tool` follows an assistant whose
    // `tool_calls` names its id.
    for (i, m) in engine.messages.iter().enumerate() {
        if role(m) != "tool" {
            continue;
        }
        let id = m["tool_call_id"].as_str().unwrap();
        let prev = &engine.messages[i - 1];
        assert_eq!(
            role(prev),
            "assistant",
            "tool result not preceded by assistant"
        );
        let names: Vec<&str> = prev["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["id"].as_str())
            .collect();
        assert!(names.contains(&id), "tool result {id} has no matching call");
    }
    // First kept message after the system prompt is the folded user turn, never
    // an orphan tool/assistant — alternation holds from the very top.
    assert_eq!(role(&engine.messages[0]), "system");
    assert_eq!(role(&engine.messages[1]), "user");
    let roles: Vec<&str> = engine.messages.iter().map(role).collect();
    assert!(
        !roles.windows(2).any(|w| w[0] == "user" && w[1] == "user"),
        "compaction left consecutive user messages: {roles:?}"
    );
}

#[test]
fn build_summary_request_carries_prior_summary() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    // No prior summary → fresh prompt, transcript verbatim.
    let r1 = engine.build_summary_request("TRANSCRIPT");
    assert_eq!(r1.messages[0]["content"], json!(SUMMARY_SYSTEM_PROMPT));
    assert_eq!(r1.messages[1]["content"], json!("TRANSCRIPT"));
    // Carry-forward: prior summary set → update prompt + prior summary in user.
    engine.last_summary = Some("PRIOR".to_string());
    let r2 = engine.build_summary_request("NEWEVENTS");
    assert_eq!(
        r2.messages[0]["content"],
        json!(SUMMARY_UPDATE_SYSTEM_PROMPT)
    );
    let user = r2.messages[1]["content"].as_str().unwrap();
    assert!(
        user.contains("PRIOR") && user.contains("NEWEVENTS"),
        "{user}"
    );
}

#[test]
fn pinned_block_token_cap_trims_files_keeps_plan() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.plan =
        plan::parse_plan(&json!({"plan":[{"step":"keep me","status":"pending"}]})).unwrap();
    // Far more files than fit under PINNED_MAX_TOKENS (set directly to bypass MAX_TOUCHED_FILES).
    engine.touched_files = (0..600)
        .map(|i| format!("src/very/long/path/segment/file_{i}.rs"))
        .collect();
    let block = engine.render_pinned_block();
    assert!(
        estimate_str_tokens(&block) <= PINNED_MAX_TOKENS,
        "pinned block over cap: {} tokens",
        estimate_str_tokens(&block)
    );
    assert!(block.contains("keep me"), "plan must be kept whole");
    // Most-recent file kept, oldest trimmed.
    assert!(block.contains("file_599.rs"));
    assert!(!block.contains("file_0.rs"));
}

#[test]
fn reset_clears_compaction_working_set() {
    let mut engine = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    engine.last_summary = Some("stale".to_string());
    engine.plan = plan::parse_plan(&json!({"plan":[{"step":"x","status":"pending"}]})).unwrap();
    engine.touched_files = vec!["a.rs".to_string()];
    engine.notes = vec![notes::Note {
        id: None,
        text: "a finding".to_string(),
    }];
    engine.reset();
    assert!(engine.last_summary.is_none());
    assert!(engine.plan.is_empty());
    assert!(engine.touched_files.is_empty());
    assert!(engine.notes.is_empty());
}

/// The cheap pass stubs bulky OLD tool outputs (before the keep window), leaving recent ones + their ids intact; idempotent.
#[test]
fn clear_stale_tool_results_clears_only_old_bulky_outputs() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let big = "x".repeat(10_000); // ~1.25k tokens
    // [0 system, 1 user, 2 assistant(call), 3 tool BIG(old), 4 user, 5 tool BIG(recent), 6 asst]
    e.messages.push(json!({"role":"user","content":"go"}));
    e.messages.push(json!({"role":"assistant","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c1","content": big.clone()}));
    e.messages.push(json!({"role":"user","content":"more"}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c2","content": big.clone()}));
    e.messages.push(json!({"role":"assistant","content":"ok"}));

    let cut = 4; // messages[4] is the second user turn → [1..4] is "old"
    assert!(e.stale_tool_result_savings(cut) > 1000);
    e.clear_stale_tool_results(cut);
    assert_eq!(e.messages[3]["content"], TOOL_RESULT_CLEARED);
    assert_eq!(e.messages[3]["tool_call_id"], "c1"); // pairing intact
    assert_eq!(
        e.messages[5]["content"].as_str().unwrap().len(),
        10_000,
        "recent tool output untouched"
    );
    assert_eq!(e.stale_tool_result_savings(cut), 0, "idempotent");
}

/// Clearing a bulky sub-agent tool result keeps its artifact-pointer line so the
/// parent can re-read the report; the pass is idempotent.
#[test]
fn clear_stale_keeps_artifact_pointer() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let pointer = format!("{ARTIFACT_POINTER_PREFIX}/tmp/sub-001-x.md — re-read it]");
    let big = format!("{}\n\n{pointer}", "x".repeat(5000));
    e.messages.push(json!({"role":"user","content":"go"}));
    e.messages.push(json!({"role":"assistant","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"subagent","arguments":"{}"}}]}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c1","content": big}));
    e.messages.push(json!({"role":"user","content":"more"}));

    let cut = 4;
    e.clear_stale_tool_results(cut);
    let content = e.messages[3]["content"].as_str().unwrap().to_string();
    assert_eq!(
        content,
        format!("{TOOL_RESULT_CLEARED}\n{pointer}"),
        "stub should retain the pointer line"
    );
    // Second pass is a no-op (stub + pointer < TOOL_RESULT_CLEAR_MIN).
    e.clear_stale_tool_results(cut);
    assert_eq!(e.messages[3]["content"].as_str().unwrap(), content);
}

/// Clearing keeps the REAL (trailing) pointer, not a decoy line in the answer body.
#[test]
fn clear_stale_keeps_trailing_pointer_not_a_decoy_body_line() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    let decoy = format!("{ARTIFACT_POINTER_PREFIX}/decoy.md] (quoted inside the answer)");
    let real = format!("{ARTIFACT_POINTER_PREFIX}/tmp/sub-001-x.md — re-read it]");
    let big = format!(
        "{}\n{decoy}\n{}\n\n{real}",
        "x".repeat(2000),
        "y".repeat(2000)
    );
    e.messages.push(json!({"role":"user","content":"go"}));
    e.messages.push(json!({"role":"assistant","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"subagent","arguments":"{}"}}]}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c1","content": big}));
    e.messages.push(json!({"role":"user","content":"more"}));

    e.clear_stale_tool_results(4);
    assert_eq!(
        e.messages[3]["content"].as_str().unwrap(),
        format!("{TOOL_RESULT_CLEARED}\n{real}"),
        "the trailing real pointer must win over the decoy"
    );
}

/// A bulky tool result without a pointer clears to the plain stub (regression guard).
#[test]
fn clear_stale_without_pointer_unchanged() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.messages.push(json!({"role":"user","content":"go"}));
    e.messages.push(json!({"role":"assistant","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c1","content": "y".repeat(5000)}));
    e.messages.push(json!({"role":"user","content":"more"}));

    e.clear_stale_tool_results(4);
    assert_eq!(e.messages[3]["content"], TOOL_RESULT_CLEARED);
}

/// `take_note` content rides into a compaction via the pinned block; the cap trims files before notes (notes kept, plan whole).
#[test]
fn notes_pin_into_compaction_block() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.plan = plan::parse_plan(&json!({"plan":[{"step":"keep me","status":"pending"}]})).unwrap();
    e.notes = vec![
        notes::Note {
            id: None,
            text: "decided on X".to_string(),
        },
        notes::Note {
            id: Some("dead-end".to_string()),
            text: "Y 500s — avoid".to_string(),
        },
    ];
    e.touched_files = (0..600).map(|i| format!("src/seg/file_{i}.rs")).collect();
    let block = e.render_pinned_block();
    assert!(estimate_str_tokens(&block) <= PINNED_MAX_TOKENS);
    assert!(block.contains("## Notes"));
    assert!(block.contains("decided on X"));
    assert!(
        block.contains("- (dead-end) Y 500s"),
        "id rendered for update targeting"
    );
    assert!(block.contains("keep me"), "plan kept whole");
    assert!(!block.contains("file_0.rs"), "files trimmed before notes");
}

/// Restore re-derives the working set (plan, notes, touched files) from the message log — the stateless-reducer property.
#[test]
fn restore_rebuilds_working_set_from_log() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.messages.push(json!({"role":"user","content":"do it"}));
    e.messages.push(json!({"role":"assistant","tool_calls":[
        {"id":"c1","type":"function","function":{"name":"update_plan",
         "arguments":"{\"plan\":[{\"step\":\"a\",\"status\":\"completed\"},{\"step\":\"b\",\"status\":\"in_progress\"}]}"}},
        {"id":"c2","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"src/x.rs\"}"}},
        {"id":"c3","type":"function","function":{"name":"take_note","arguments":"{\"note\":\"x uses async\"}"}}
    ]}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c1","content":"ok"}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c2","content":"FILE"}));
    e.messages
        .push(json!({"role":"tool","tool_call_id":"c3","content":"Noted (1 saved)."}));
    e.messages
        .push(json!({"role":"assistant","content":"done"}));
    let convo = e.export_conversation();

    let mut restored = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    restored.restore_conversation(convo);
    assert_eq!(restored.plan.len(), 2);
    assert_eq!(restored.plan[1].status, plan::PlanStatus::InProgress);
    assert_eq!(restored.touched_files, vec!["src/x.rs".to_string()]);
    assert_eq!(restored.notes.len(), 1);
    assert_eq!(restored.notes[0].text, "x uses async");
}

/// The three merge outcomes surface distinct tool-result confirmations, and a
/// resumed transcript reproduces the merged notes (live/resume parity).
#[tokio::test]
async fn take_note_merge_outcomes_and_resume_parity() {
    let dir = tmp();
    let calls = batch_tool_call_sse(&[
        (
            "c1",
            "take_note",
            json!({"note": "use sessions", "id": "auth"}),
        ), // Added
        ("c2", "take_note", json!({"note": "a finding"})), // Added
        ("c3", "take_note", json!({"note": "use JWT", "id": "auth"})), // Updated(auth)
        ("c4", "take_note", json!({"note": "A   FINDING"})), // Refreshed (dup)
    ]);
    let port = spawn_sse_sequence(vec![calls, FINAL_TEXT_SSE.to_string()]);
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let base = format!("http://127.0.0.1:{port}");
    let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    let mut ui = CapturingUi::default();
    run_session(
        &mut engine,
        &turn_ctx(&client, &base, &dir),
        Some("note stuff".into()),
        &mut ui,
    )
    .await;

    // Each call's confirmation reflects its outcome, in order.
    let tool_results: Vec<String> = engine
        .messages
        .iter()
        .filter(|m| role(m) == "tool")
        .filter_map(|m| {
            m.get("content")
                .and_then(|c| c.as_str())
                .map(str::to_string)
        })
        .collect();
    assert_eq!(
        tool_results,
        vec![
            "Noted (1 saved).".to_string(),
            "Noted (2 saved).".to_string(),
            "Updated note 'auth'.".to_string(),
            "Already noted (refreshed).".to_string(),
        ]
    );
    // Two notes remain: the id-updated one and the deduped finding.
    assert_eq!(engine.notes.len(), 2);
    assert_eq!(engine.notes[0].text, "use JWT");
    assert_eq!(engine.notes[0].id.as_deref(), Some("auth"));
    assert_eq!(engine.notes[1].text, "a finding");

    // Resume from the same transcript reproduces the merged notes.
    let convo = engine.export_conversation();
    let mut restored = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
    restored.restore_conversation(convo);
    assert_eq!(restored.notes.len(), 2);
    assert_eq!(restored.notes[0].text, "use JWT");
    assert_eq!(restored.notes[1].text, "a finding");
}
