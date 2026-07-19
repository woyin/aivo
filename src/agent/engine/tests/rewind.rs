use super::super::*;
use super::helpers::*;
use serde_json::json;

fn rewind_engine(dir: &Path) -> AgentEngine {
    AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0)
}

#[tokio::test]
async fn rewind_to_truncates_messages_and_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = rewind_engine(dir.path());
    // [system]; then two turns each adding a user + assistant message.
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "a".into(),
        tree: None,
        changed: None,
        seg_tree: None,
    });
    engine
        .messages
        .push(json!({"role": "user", "content": "a"}));
    engine
        .messages
        .push(json!({"role": "assistant", "content": "b"}));
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "c".into(),
        tree: None,
        changed: None,
        seg_tree: None,
    });
    engine
        .messages
        .push(json!({"role": "user", "content": "c"}));
    engine
        .messages
        .push(json!({"role": "assistant", "content": "d"}));

    let outcome = engine.rewind_to(1).await;
    // Truncated back to the start of turn 1 (system + turn 0's two messages).
    assert_eq!(engine.messages.len(), 3);
    let targets = engine.rewind_targets();
    assert_eq!(targets.len(), 1);
    // No tree on these checkpoints → conversation-only, nothing reverted.
    assert!(!targets[0].1);
    assert_eq!((outcome.restored, outcome.deleted), (0, 0));
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn rewind_targets_report_prompts_and_revertibility() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = rewind_engine(dir.path());
    // Two turns restored on resume carry no checkpoints (conversation-only).
    engine.restore_conversation(vec![
        json!({"role": "user", "content": "a"}),
        json!({"role": "assistant", "content": "b"}),
    ]);
    // A live turn (user "c") with a tree snapshot, then one (user "e") without.
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "c".into(),
        tree: Some("abc".into()),
        changed: None,
        seg_tree: None,
    });
    engine
        .messages
        .push(json!({"role": "user", "content": "c"}));
    engine
        .messages
        .push(json!({"role": "assistant", "content": "d"}));
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "e".into(),
        tree: None,
        changed: None,
        seg_tree: None,
    });
    engine
        .messages
        .push(json!({"role": "user", "content": "e"}));

    // Targets carry the opening prompt + revertibility, in order.
    let targets = engine.rewind_targets();
    assert_eq!(
        targets,
        vec![("c".to_string(), true), ("e".to_string(), false)]
    );
}

#[tokio::test]
async fn compaction_rebases_checkpoint_indices() {
    let dir = tempfile::tempdir().unwrap();
    let mut engine = rewind_engine(dir.path());
    // Three turns: [system, u0, a0, u1, a1, u2, a2] with a checkpoint opening
    // each user turn (indices 1, 3, 5).
    for (u, a) in [("u0", "a0"), ("u1", "a1"), ("u2", "a2")] {
        engine.checkpoints.push(Checkpoint {
            msg_index: engine.messages.len(),
            prompt: u.into(),
            tree: None,
            changed: None,
            seg_tree: None,
        });
        engine.messages.push(json!({"role": "user", "content": u}));
        engine
            .messages
            .push(json!({"role": "assistant", "content": a}));
    }
    assert_eq!(
        engine
            .checkpoints
            .iter()
            .map(|c| c.msg_index)
            .collect::<Vec<_>>(),
        vec![1, 3, 5]
    );

    // Fold the first turn (cut lands on u1 at index 3).
    engine.apply_compaction(3, "S");

    // u0's checkpoint dropped; survivors shifted down to stay valid (not stale).
    // messages is now [system, "S\n\nu1", a1, u2, a2].
    assert_eq!(
        engine
            .checkpoints
            .iter()
            .map(|c| c.msg_index)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    // Verbatim "u1", not the folded "S\n\nu1" at messages[1].
    assert_eq!(engine.rewind_targets()[0].0, "u1");
    assert_eq!(engine.rewind_targets()[1].0, "u2");

    // Rewinding to the last turn truncates at the correct (rebased) index.
    engine.rewind_to(1).await;
    assert_eq!(engine.messages.len(), 3);
    assert_eq!(role(engine.messages.last().unwrap()), "assistant");
    assert_eq!(engine.messages[2]["content"], "a1");
}

#[tokio::test]
async fn rewind_target_survives_interrupted_turn_merge() {
    // A resend after an interrupt merges into the trailing `user` ("first\n\nsecond");
    // the stored prompt must stay "first" so the turn keeps file revert, not conversation-only.
    let dir = tempfile::tempdir().unwrap();
    let mut engine = rewind_engine(dir.path());
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "first".into(),
        tree: Some("abc".into()),
        changed: None,
        seg_tree: None,
    });
    engine.push_text_turn("user", "first".into());
    engine.push_text_turn("user", "second".into());
    assert_eq!(
        engine.messages.last().unwrap()["content"],
        "first\n\nsecond"
    );

    let targets = engine.rewind_targets();
    assert_eq!(targets, vec![("first".to_string(), true)]);
}

#[tokio::test]
async fn rewind_reverts_files_through_the_engine() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("a.txt"), "v0").unwrap();
    let mut engine = rewind_engine(p);
    engine.enable_rewind_checkpoints(&p.display().to_string());
    let tree = {
        let store = engine.checkpoint_store.as_mut().unwrap();
        if !store.git_available().await {
            return; // git missing → skip
        }
        store.snapshot().await
    };
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "go".into(),
        tree,
        changed: None,
        seg_tree: None,
    });
    // Simulate the turn: rename + edit (the case byte-snapshots couldn't revert).
    std::fs::rename(p.join("a.txt"), p.join("b.txt")).unwrap();
    std::fs::write(p.join("b.txt"), "v1").unwrap();

    let outcome = engine.rewind_to(0).await;
    assert_eq!(std::fs::read_to_string(p.join("a.txt")).unwrap(), "v0");
    assert!(!p.join("b.txt").exists(), "renamed file removed");
    assert!(outcome.restored >= 1);
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn interrupt_record_shields_user_edits_from_rewind() {
    // Hand-edits made after the cancel-time record must survive a later rewind.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("agent.txt"), "a0").unwrap();
    std::fs::write(p.join("mine.txt"), "m0").unwrap();
    let mut engine = rewind_engine(p);
    engine.enable_rewind_checkpoints(&p.display().to_string());
    let tree = {
        let store = engine.checkpoint_store.as_mut().unwrap();
        if !store.git_available().await {
            return; // git missing → skip
        }
        store.snapshot().await
    };
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "go".into(),
        tree,
        changed: None,
        seg_tree: None,
    });
    // Agent edit + cancel-time record, then a post-interrupt hand-edit.
    std::fs::write(p.join("agent.txt"), "a1").unwrap();
    engine.record_turn_changes().await;
    std::fs::write(p.join("mine.txt"), "m-edited").unwrap();

    let outcome = engine.rewind_to(0).await;
    assert_eq!(std::fs::read_to_string(p.join("agent.txt")).unwrap(), "a0");
    assert_eq!(
        std::fs::read_to_string(p.join("mine.txt")).unwrap(),
        "m-edited",
        "the user's post-interrupt edit must survive the rewind"
    );
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn resumed_segment_diffs_from_its_own_base() {
    // Segments union across an interrupt+resend, minus the user's idle-gap edit.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("agent.txt"), "a0").unwrap();
    std::fs::write(p.join("mine.txt"), "m0").unwrap();
    let mut engine = rewind_engine(p);
    engine.enable_rewind_checkpoints(&p.display().to_string());
    let tree = {
        let store = engine.checkpoint_store.as_mut().unwrap();
        if !store.git_available().await {
            return; // git missing → skip
        }
        store.snapshot().await
    };
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "go".into(),
        tree,
        changed: None,
        seg_tree: None,
    });
    // Segment 1 (agent edit + interrupt record), idle-gap user edit, then
    // segment 2 from a fresh seg base (agent create + turn-end record).
    std::fs::write(p.join("agent.txt"), "a1").unwrap();
    engine.record_turn_changes().await;
    std::fs::write(p.join("mine.txt"), "m-edited").unwrap();
    let seg = engine.checkpoint_store.as_mut().unwrap().snapshot().await;
    engine.checkpoints.last_mut().unwrap().seg_tree = seg;
    std::fs::write(p.join("new.txt"), "n").unwrap();
    engine.record_turn_changes().await;

    let outcome = engine.rewind_to(0).await;
    assert_eq!(std::fs::read_to_string(p.join("agent.txt")).unwrap(), "a0");
    assert!(!p.join("new.txt").exists(), "segment-2 create removed");
    assert_eq!(
        std::fs::read_to_string(p.join("mine.txt")).unwrap(),
        "m-edited",
        "the idle-gap hand-edit is not part of either segment"
    );
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn rewind_to_read_only_turn_restores_from_the_next_tree() {
    // Rewinding to a tree-less (read-only) turn must still revert later edits.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("a.txt"), "v0").unwrap();
    let mut engine = rewind_engine(p);
    engine.enable_rewind_checkpoints(&p.display().to_string());
    if !engine
        .checkpoint_store
        .as_mut()
        .unwrap()
        .git_available()
        .await
    {
        return; // git missing → skip
    }
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "look".into(),
        tree: None,
        changed: Some(Vec::new()),
        seg_tree: None,
    });
    let tree = engine.checkpoint_store.as_mut().unwrap().snapshot().await;
    engine.checkpoints.push(Checkpoint {
        msg_index: engine.messages.len(),
        prompt: "edit".into(),
        tree,
        changed: None,
        seg_tree: None,
    });
    std::fs::write(p.join("a.txt"), "v1").unwrap();
    engine.record_turn_changes().await;

    // Both turns read as file-revertible: the read-only one borrows the later tree.
    let targets = engine.rewind_targets();
    assert_eq!(
        targets.iter().map(|t| t.1).collect::<Vec<_>>(),
        [true, true]
    );

    let outcome = engine.rewind_to(0).await;
    assert_eq!(
        std::fs::read_to_string(p.join("a.txt")).unwrap(),
        "v0",
        "the later turn's edit reverts even from a read-only target"
    );
    assert_eq!(outcome.restored, 1);
    assert!(outcome.error.is_none());
}

#[tokio::test]
async fn model_switch_transplant_keeps_file_revert() {
    // export→restore preserves message indices, so transplanted checkpoints stay valid.
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("a.txt"), "v0").unwrap();
    let mut old_engine = rewind_engine(p);
    old_engine.enable_rewind_checkpoints(&p.display().to_string());
    let tree = {
        let store = old_engine.checkpoint_store.as_mut().unwrap();
        if !store.git_available().await {
            return; // git missing → skip
        }
        store.snapshot().await
    };
    old_engine.checkpoints.push(Checkpoint {
        msg_index: old_engine.messages.len(),
        prompt: "go".into(),
        tree,
        changed: None,
        seg_tree: None,
    });
    old_engine
        .messages
        .push(json!({"role": "user", "content": "go"}));
    old_engine
        .messages
        .push(json!({"role": "assistant", "content": "done"}));
    std::fs::write(p.join("a.txt"), "v1").unwrap();

    let conversation = old_engine.export_conversation();
    let (store, checkpoints) = old_engine.take_rewind_state();
    let mut new_engine = rewind_engine(p);
    new_engine.enable_rewind_checkpoints(&p.display().to_string());
    new_engine.restore_conversation(conversation);
    new_engine.adopt_rewind_state(store, checkpoints);

    let targets = new_engine.rewind_targets();
    assert_eq!(targets, vec![("go".to_string(), true)]);
    let outcome = new_engine.rewind_to(0).await;
    assert_eq!(std::fs::read_to_string(p.join("a.txt")).unwrap(), "v0");
    assert_eq!(outcome.restored, 1);
    assert!(outcome.error.is_none());
    assert_eq!(new_engine.messages.len(), 1);
}

#[tokio::test]
async fn lazy_checkpoint_snapshots_only_before_a_mutating_tool() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    std::fs::write(p.join("a.txt"), "v0").unwrap();
    let mut engine = rewind_engine(p);
    engine.enable_rewind_checkpoints(&p.display().to_string());
    if !engine
        .checkpoint_store
        .as_mut()
        .unwrap()
        .git_available()
        .await
    {
        return; // git missing → skip
    }
    let client = reqwest::Client::new();
    let ctx = turn_ctx(&client, "http://127.0.0.1", p);
    let mut ui = CapturingUi::default();
    // Stands in for the turn-start checkpoint (tree filled lazily, if at all).
    engine.checkpoints.push(Checkpoint {
        msg_index: 0,
        prompt: "go".into(),
        tree: None,
        changed: None,
        seg_tree: None,
    });

    // A read-only batch must NOT snapshot.
    let read = vec![ToolCall {
        id: "1".into(),
        name: "read_file".into(),
        arguments: json!({ "path": "a.txt" }),
    }];
    engine.execute_tool_batch(&ctx, &mut ui, &read).await;
    assert!(
        engine.checkpoints.last().unwrap().tree.is_none(),
        "read-only turn pays no snapshot"
    );

    // A mutating batch snapshots the pre-edit tree first.
    let write = vec![ToolCall {
        id: "2".into(),
        name: "write_file".into(),
        arguments: json!({ "path": "a.txt", "content": "v1" }),
    }];
    engine.execute_tool_batch(&ctx, &mut ui, &write).await;
    assert!(
        engine.checkpoints.last().unwrap().tree.is_some(),
        "snapshot taken before a mutating tool"
    );
}
