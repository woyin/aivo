//! The one table of built-in tool traits. Every classification the engine used
//! to hand-keep in scattered `matches!` lists — read-only, parallel-safe,
//! mutating, file-writing, plan-hidden, path-carrying — is one row here; the
//! predicates in `safety`/`file_tracker` query it. Adding a tool without a
//! row fails the spec/registry contract test.

/// Classification of one built-in tool. Flags default to the SAFE side (an
/// unknown tool is not read-only, not parallel, not a write), so a missing row
/// degrades to prompts and snapshots, never to silent trust.
pub struct ToolTraits {
    pub name: &'static str,
    /// Never touches the workspace — skips the `/rewind` snapshot.
    pub read_only: bool,
    /// Side-effect-free and independently runnable within one batch.
    pub parallel_safe: bool,
    /// Mutates the workspace via the client (permission-gated before execute).
    pub mutating: bool,
    /// File-writing tool: write escalation, LSP fold, read-only refusal.
    pub file_write: bool,
    /// Stripped from the advertised schema in plan mode.
    pub plan_hidden: bool,
    /// Carries workspace path arguments worth tracking (reads + writes).
    pub path_tracked: bool,
    /// Dispatched by `tools::execute` (names the error message's available list).
    pub workspace_exec: bool,
}

const BASE: ToolTraits = ToolTraits {
    name: "",
    read_only: false,
    parallel_safe: false,
    mutating: false,
    file_write: false,
    plan_hidden: false,
    path_tracked: false,
    workspace_exec: false,
};

macro_rules! tool {
    ($name:literal $(, $flag:ident)* $(,)?) => {
        ToolTraits { name: $name, $($flag: true,)* ..BASE }
    };
}

/// One row per built-in tool the engine can advertise, in spec order.
#[rustfmt::skip]
pub static TOOL_TRAITS: &[ToolTraits] = &[
    tool!("read_file", read_only, parallel_safe, path_tracked, workspace_exec),
    tool!("list_dir", read_only, workspace_exec),
    tool!("glob", read_only, parallel_safe, workspace_exec),
    tool!("grep", read_only, parallel_safe, workspace_exec),
    tool!("web_fetch", read_only, parallel_safe, workspace_exec),
    tool!("web_search", read_only, parallel_safe, workspace_exec),
    tool!("write_file", mutating, file_write, plan_hidden, path_tracked, workspace_exec),
    tool!("edit_file", mutating, file_write, plan_hidden, path_tracked, workspace_exec),
    tool!("multi_edit", mutating, file_write, plan_hidden, path_tracked, workspace_exec),
    tool!("apply_patch", mutating, file_write, plan_hidden, path_tracked, workspace_exec),
    tool!("run_bash", mutating, workspace_exec),
    // Engine-handled tools (never reach `tools::execute`).
    tool!("update_plan"),
    tool!("finish_turn"),
    tool!("skill"),
    tool!("take_note"),
    // A sub-engine isn't read-only; image generation writes a file + bills a call.
    tool!("subagent", plan_hidden),
    tool!("generate_image", plan_hidden),
    // Display only — no workspace touch.
    tool!("preview", read_only),
    // Session controls / prompts / job+schema queries — engine state only.
    tool!("switch_model", read_only),
    tool!("switch_key", read_only),
    tool!("set_effort", read_only),
    tool!("ask_user", read_only),
    tool!("check_job", read_only),
    tool!("search_tools", read_only),
    tool!("exit_plan_mode"),
    tool!("list_sessions"),
    tool!("send_session"),
];

/// The row for `name`; `None` for unknown/external tools (callers treat that as
/// the conservative default on every flag).
pub fn traits(name: &str) -> Option<&'static ToolTraits> {
    TOOL_TRAITS.iter().find(|t| t.name == name)
}

/// `, `-joined names `tools::execute` dispatches — the "available:" error list.
pub fn workspace_exec_names() -> String {
    TOOL_TRAITS
        .iter()
        .filter(|t| t.workspace_exec)
        .map(|t| t.name)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every advertised spec has a registry row (a new tool must be classified),
    /// under both edit-format variants.
    #[test]
    fn every_spec_name_has_a_registry_row() {
        for model in ["claude-sonnet-5", "gpt-5.2"] {
            for spec in crate::agent::tools::tool_specs_for(model) {
                assert!(
                    traits(&spec.name).is_some(),
                    "tool `{}` is advertised but has no ToolTraits row",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn error_list_names_every_executable_tool() {
        let list = workspace_exec_names();
        assert!(list.contains("apply_patch"), "{list}");
        assert!(list.contains("run_bash"), "{list}");
    }
}
