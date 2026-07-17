use super::super::*;
use super::helpers::*;
use serde_json::json;

#[test]
fn native_search_supported_is_conservative() {
    assert!(native_search_supported("claude-opus-4"));
    assert!(native_search_supported("anthropic/claude-3.5-sonnet"));
    // Gemini 400s on native-search + function-calling and the agent always
    // sends function tools, so it keeps the hosted tool (B/C).
    assert!(!native_search_supported("gemini-2.5-pro"));
    assert!(!native_search_supported("google/gemini-2.5-flash"));
    // Everything else keeps the hosted web_search tool (B/C).
    assert!(!native_search_supported("deepseek-v4-flash"));
    assert!(!native_search_supported("gpt-5"));
    assert!(!native_search_supported("qwen3-max"));
    assert!(!native_search_supported("llama-3.3-70b"));
}

#[test]
fn specs_cover_all_tools() {
    let names: Vec<String> = tool_specs().into_iter().map(|s| s.name).collect();
    assert_eq!(names.len(), 10);
    for n in [
        "read_file",
        "list_dir",
        "glob",
        "grep",
        "write_file",
        "edit_file",
        "multi_edit",
        "web_fetch",
        "web_search",
        "run_bash",
    ] {
        assert!(names.iter().any(|x| x == n), "missing {n}");
    }
}

#[test]
fn apply_patch_routing_by_model() {
    for m in ["gpt-5", "openai/gpt-5-codex", "codex-mini", "gpt-4.1-mini"] {
        assert!(uses_apply_patch(m), "{m} should use apply_patch");
        let names: Vec<String> = tool_specs_for(m).into_iter().map(|s| s.name).collect();
        assert!(
            names.iter().any(|n| n == "apply_patch"),
            "{m} missing apply_patch"
        );
        assert!(
            !names.iter().any(|n| n == "edit_file"),
            "{m} kept edit_file"
        );
        assert!(
            !names.iter().any(|n| n == "multi_edit"),
            "{m} kept multi_edit"
        );
    }
    for m in [
        "claude-sonnet-4-6",
        "gpt-4o",
        "anthropic/claude-opus-4-8",
        "gemini-2.5-pro",
    ] {
        assert!(!uses_apply_patch(m), "{m} should not use apply_patch");
        let names: Vec<String> = tool_specs_for(m).into_iter().map(|s| s.name).collect();
        assert!(names.iter().any(|n| n == "edit_file"));
        assert!(
            !names.iter().any(|n| n == "apply_patch"),
            "{m} got apply_patch"
        );
    }
}

/// `execute` must route `apply_patch` (the advertised tool for GPT-5/Codex) to
/// the V4A applier, not to `edit_file` — the normalize table once collapsed the
/// two, which errored on the missing `path` arg and broke editing for those
/// models. Also covers dispatch through an alias.
#[tokio::test]
async fn execute_routes_apply_patch_not_to_edit_file() {
    for name in ["apply_patch", "applyPatch"] {
        let dir = tmp();
        let patch = "*** Begin Patch\n*** Add File: made.txt\n+hi\n*** End Patch";
        execute(name, &json!({ "input": patch }), &dir)
            .await
            .unwrap_or_else(|e| panic!("{name} should apply a patch, got: {e}"));
        assert_eq!(
            std::fs::read_to_string(dir.join("made.txt"))
                .unwrap()
                .trim(),
            "hi",
            "{name} did not write the patched file"
        );
    }
}

#[test]
fn unknown_tool_in_preview_is_none() {
    assert!(preview("read_file", &json!({"path":"x"})).is_none());
    assert!(preview("run_bash", &json!({"command":"ls"})).is_some());
    assert!(preview("multi_edit", &json!({"path":"x","edits":[{}]})).is_some());
}
