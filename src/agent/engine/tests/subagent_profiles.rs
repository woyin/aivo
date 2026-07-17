use super::super::*;
use super::helpers::*;
use serde_json::json;

/// With no subagents the tool stays generic — no `agent` field, no listing.
#[test]
fn subagent_tool_spec_is_generic_without_profiles() {
    let spec = subagent_tool_spec(&[]);
    assert_eq!(spec.name, "subagent");
    assert!(spec.parameters["properties"].get("agent").is_none());
    assert!(spec.parameters["properties"].get("label").is_some());
    assert_eq!(spec.parameters["required"], json!(["task"]));
}

#[test]
fn subagent_display_name_prefers_label_and_cc_names() {
    assert_eq!(
        subagent_display_name(&json!({"label": "audit auth", "agent": "reviewer"})),
        "audit auth"
    );
    // Claude Code arg names are accepted as fallbacks.
    assert_eq!(
        subagent_display_name(&json!({"description": "deep dive", "subagent_type": "explore"})),
        "deep dive"
    );
    assert_eq!(
        subagent_display_name(&json!({"subagent_type": "explore"})),
        "explore"
    );
    assert_eq!(subagent_display_name(&json!({"task": "long text"})), "");
}

/// With profiles, the tool advertises them via an `agent` enum.
#[test]
fn subagent_tool_spec_enumerates_named_profiles() {
    let subs = vec![
        subagent("reviewer", None, None),
        subagent("researcher", None, None),
    ];
    let spec = subagent_tool_spec(&subs);
    let enumv = &spec.parameters["properties"]["agent"]["enum"];
    assert_eq!(enumv, &json!(["reviewer", "researcher"]));
    assert!(spec.description.contains("named specialist"));
}

/// `set_subagents` swaps in the enum-bearing tool and advertises the names in
/// the system prompt (progressive disclosure — body is NOT inlined).
#[test]
fn set_subagents_wires_tool_and_prompt() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.set_subagents(&[subagent("reviewer", None, None)]);
    // The subagent tool now carries the enum.
    let sub_tool = e
        .tools_openai
        .iter()
        .find(|t| t["function"]["name"] == "subagent")
        .unwrap();
    assert_eq!(
        sub_tool["function"]["parameters"]["properties"]["agent"]["enum"],
        json!(["reviewer"])
    );
    // exactly one `subagent` tool (the generic one was replaced, not duplicated).
    assert_eq!(
        tool_names(&e).iter().filter(|n| *n == "subagent").count(),
        1
    );
    // System prompt lists the name + one-liner, not the full body.
    let sys = system_content(&e);
    assert!(sys.contains("- reviewer: the reviewer specialist"));
    assert!(!sys.contains("Follow the reviewer playbook"));
    // Empty set is a no-op (no agent field).
    let mut e2 = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e2.set_subagents(&[]);
    let sub_tool2 = e2
        .tools_openai
        .iter()
        .find(|t| t["function"]["name"] == "subagent")
        .unwrap();
    assert!(
        sub_tool2["function"]["parameters"]["properties"]
            .get("agent")
            .is_none()
    );
}

/// Delegation-time profile resolution: with an agents dir configured, a
/// profile written AFTER engine build (or edited since) resolves fresh from
/// disk; without one, the build-time snapshot answers.
#[test]
fn resolve_profile_prefers_disk_over_snapshot() {
    let cwd = tempfile::tempdir().unwrap();
    let cfg = tempfile::tempdir().unwrap();

    // Snapshot-only engine: unknown dir → falls back to set_subagents.
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.set_subagents(&[subagent("reviewer", None, None)]);
    let p = e.resolve_profile(cwd.path(), "reviewer").unwrap();
    assert!(p.body.contains("reviewer playbook"), "snapshot body");

    // With an agents dir, a file authored after build wins over the snapshot…
    e.set_agents_dir(cfg.path());
    let dir = cwd.path().join(".aivo/agents");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("reviewer.md"),
        "---\nname: reviewer\ndescription: fresh\n---\nFRESH BODY v2\n",
    )
    .unwrap();
    let p = e.resolve_profile(cwd.path(), "reviewer").unwrap();
    assert_eq!(p.body, "FRESH BODY v2", "disk beats snapshot");
    // …and a brand-new name (created mid-turn) is delegatable immediately.
    std::fs::write(
        dir.join("tester.md"),
        "---\nname: tester\ndescription: t\n---\nTEST BODY\n",
    )
    .unwrap();
    assert!(e.resolve_profile(cwd.path(), "tester").is_some());
    // Unknown names still miss (→ generic fallback with a note).
    assert!(e.resolve_profile(cwd.path(), "ghost").is_none());
}

/// A profile's body folds into the system prompt; a `tools` allow-list restricts the built-ins (keeping `update_plan`).
#[test]
fn apply_profile_folds_role_and_scopes_tools() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.drop_subagent_tool(); // mirror a real sub-engine
    let before = tool_names(&e);
    assert!(before.contains(&"write_file".to_string()));
    assert!(before.contains(&"run_bash".to_string()));

    e.apply_profile(&subagent("reviewer", None, Some(vec!["read_file", "grep"])));

    // Role instructions are appended verbatim.
    let sys = system_content(&e);
    assert!(sys.contains("## Your role: reviewer"));
    assert!(sys.contains("Follow the reviewer playbook"));

    // Scoped to the allow-list (+ update_plan); writes/bash are gone.
    let after = tool_names(&e);
    assert!(after.contains(&"read_file".to_string()));
    assert!(after.contains(&"grep".to_string()));
    assert!(after.contains(&"update_plan".to_string()));
    assert!(!after.contains(&"write_file".to_string()));
    assert!(!after.contains(&"run_bash".to_string()));
}

/// On a gpt-5 engine, an authored `Edit` scope grants `apply_patch`.
#[test]
fn apply_profile_edit_scope_grants_apply_patch_on_gpt5() {
    let mut e = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
    e.drop_subagent_tool();
    assert!(tool_names(&e).contains(&"apply_patch".to_string()));
    e.apply_profile(&subagent(
        "editor",
        None,
        Some(vec!["read_file", "edit_file"]),
    ));
    let after = tool_names(&e);
    assert!(
        after.contains(&"apply_patch".to_string()),
        "lost editor on gpt-5"
    );
    assert!(after.contains(&"read_file".to_string()));
    assert!(!after.contains(&"run_bash".to_string()));
}

/// Reverse of the gpt-5 case: an `apply_patch` scope grants `edit_file` — the edit family is one class, so scoping is symmetric.
#[test]
fn apply_profile_apply_patch_scope_grants_edit_file_off_codex() {
    let mut e = AgentEngine::new("/tmp", "claude-sonnet-4-6", "", &[], &[], 0, 0);
    e.drop_subagent_tool();
    assert!(tool_names(&e).contains(&"edit_file".to_string()));
    e.apply_profile(&subagent(
        "patcher",
        None,
        Some(vec!["read_file", "apply_patch"]),
    ));
    let after = tool_names(&e);
    assert!(
        after.contains(&"edit_file".to_string()),
        "lost editor off codex"
    );
    assert!(after.contains(&"read_file".to_string()));
    assert!(!after.contains(&"run_bash".to_string()));
}

/// A profile with no `tools` scope leaves the toolset untouched.
#[test]
fn apply_profile_without_scope_keeps_all_tools() {
    let mut e = AgentEngine::new("/tmp", "m", "", &[], &[], 0, 0);
    e.drop_subagent_tool();
    let before = tool_names(&e);
    e.apply_profile(&subagent("helper", None, None));
    assert_eq!(tool_names(&e), before);
}
