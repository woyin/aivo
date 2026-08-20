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

/// For every model variant, any tool name a description mentions must itself be
/// advertised to that model — "use X instead" must never point at an absent tool.
#[test]
fn descriptions_only_reference_tools_advertised_to_the_same_model() {
    let all_names: Vec<String> = tool_specs()
        .into_iter()
        .map(|s| s.name)
        .chain(std::iter::once("apply_patch".to_string()))
        .collect();
    // Word-boundary mention, so `ripgrep` doesn't count as naming `grep`.
    let mentions = |desc: &str, name: &str| {
        let is_word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
        desc.match_indices(name).any(|(i, _)| {
            !is_word(desc[..i].chars().next_back())
                && !is_word(desc[i + name.len()..].chars().next())
        })
    };
    for m in [
        "gpt-5",
        "openai/gpt-5-codex",
        "gpt-4.1-mini",
        "claude-sonnet-4-6",
        "gemini-2.5-pro",
        "deepseek-v4",
    ] {
        let specs = tool_specs_for(m);
        let advertised: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        for s in &specs {
            for name in &all_names {
                if mentions(&s.description, name) && !advertised.contains(&name.as_str()) {
                    panic!(
                        "{m}: `{}` description references `{name}`, which this model \
doesn't have — reroute the cross-reference in tool_specs_for",
                        s.name
                    );
                }
            }
        }
    }
    // Rerouting, not deletion: swap models point at apply_patch, others keep the pointer.
    let desc_for = |m: &str| {
        tool_specs_for(m)
            .into_iter()
            .find(|s| s.name == "write_file")
            .unwrap()
            .description
    };
    assert!(desc_for("gpt-5").contains("apply_patch"));
    assert!(desc_for("claude-sonnet-4-6").contains("edit_file/multi_edit"));
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
