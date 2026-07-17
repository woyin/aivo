//! Tool JSON-schema specs and per-model routing.

use super::*;

/// OpenAI function specs for the locally-executed tools — sent with each chat
/// request (the engine appends `skill`/`update_plan`, which it handles itself).
pub fn tool_specs() -> Vec<ToolSpec> {
    vec![
        spec(
            "read_file",
            "Read a file's contents with line numbers. Use offset/limit to page large files.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to cwd or absolute)"},
                    "offset": {"type": "integer", "description": "1-based starting line (default 1)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default 2000)"}
                },
                "required": ["path"]
            }),
        ),
        spec(
            "list_dir",
            "List the entries of a directory (directories shown with a trailing /).",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default current dir)"}
                }
            }),
        ),
        spec(
            "glob",
            "Find files by glob pattern. Supports *, ?, and **/ for recursive matching (e.g. **/*.rs).",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern matched against paths relative to `path`"},
                    "path": {"type": "string", "description": "Root directory to search (default current dir)"}
                },
                "required": ["pattern"]
            }),
        ),
        spec(
            "grep",
            "Search file contents for a pattern (regex via ripgrep when available). Returns path:line:text. Set `context` to also show N lines around each match (like grep -C) — see a match's surrounding code in one call instead of a follow-up read_file.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "File or directory to search (default current dir)"},
                    "context": {"type": "integer", "description": "Lines of context to show around each match (default 0)"}
                },
                "required": ["pattern"]
            }),
        ),
        spec(
            "write_file",
            "Write (create or overwrite) a file with the given content. Creates parent directories.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
        ),
        spec(
            "edit_file",
            "Replace an exact string in a file with a new string. By default old_string must match exactly once (errors if missing or ambiguous); set replace_all to replace every occurrence.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match (default false)."}
                },
                "required": ["path", "old_string", "new_string"]
            }),
        ),
        spec(
            "multi_edit",
            "Apply several edits to one file in a single call. Edits run in order, each against the result of the previous one; if any edit fails to match, none are applied (the file is left untouched). Prefer this over repeated edit_file calls on the same file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "description": "Edits applied sequentially. Each replaces old_string with new_string (unique match unless replace_all).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {"type": "boolean"}
                            },
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        ),
        spec(
            "web_fetch",
            "Fetch a public http(s) URL and return its content as readable text (HTML is reduced to text). Read-only GET; for APIs, custom headers, or POST, use run_bash with curl. Private/loopback/link-local addresses (localhost, RFC1918, cloud metadata) are refused.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http:// or https:// URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Cap on returned characters (default 30000)"}
                },
                "required": ["url"]
            }),
        ),
        spec(
            "web_search",
            "Search the web and return ranked results (title, URL, snippet). Use it to find current or external information, then call web_fetch on a result URL to read that page.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "max_results": {"type": "integer", "description": "Max results to return (default 8, max 20)"}
                },
                "required": ["query"]
            }),
        ),
        spec(
            "run_bash",
            "Run a shell command in the working directory. Each call is a fresh shell (cd does not persist). Runs non-interactively with no TTY: interactive programs (editors, `ssh`/`sudo` prompts, TUIs) are refused, and long-running ones are killed at the timeout — use non-interactive flags. To start a server, watcher, or anything long-running, pass `background: true`: the command is detached and this returns immediately with a job id and log file; poll or stop it with `check_job`.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds before the command is killed (default 120, max 600). Ignored with background."},
                    "background": {"type": "boolean", "description": "Run detached in the background and return a job id + log file immediately (for servers/watchers). Poll or stop it with check_job."}
                },
                "required": ["command"]
            }),
        ),
    ]
}

/// Built-in specs for `model`: GPT-5/Codex get `apply_patch` instead of
/// `edit_file`/`multi_edit` (never both — they'd mix edit formats).
pub fn tool_specs_for(model: &str) -> Vec<ToolSpec> {
    let mut specs = tool_specs();
    if uses_apply_patch(model) {
        specs.retain(|s| s.name != "edit_file" && s.name != "multi_edit");
        specs.push(apply_patch_spec());
    }
    specs
}

/// Models that emit V4A `apply_patch` fluently (and botch exact-string edits).
pub fn uses_apply_patch(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.contains("codex") || name.starts_with("gpt-5") || name.starts_with("gpt-4.1")
}

/// Providers whose native `{type:"web_search"}` server tool can coexist with
/// the agent's function tools. Anthropic can; Gemini 400s on the mix, so it
/// uses the hosted tool. Name-based — no `/v1/models` flag advertises this.
pub(super) fn native_search_supported(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.starts_with("claude") || lower.contains("anthropic")
}

/// Layer A: hand search to the provider instead of the local tool. Conservative —
/// the bridge drops an untranslatable server tool, so unknown models keep
/// `web_search`. `AIVO_AGENT_NATIVE_SEARCH=0` forces the hosted path.
pub fn native_web_search_enabled(model: &str) -> bool {
    !matches!(
        std::env::var("AIVO_AGENT_NATIVE_SEARCH").as_deref(),
        Ok("0") | Ok("false")
    ) && native_search_supported(model)
}

pub(super) fn apply_patch_spec() -> ToolSpec {
    spec(
        "apply_patch",
        "Create, edit, rename, or delete files with a V4A patch (pass the whole patch as `input`). \
Format:\n\
*** Begin Patch\n\
*** Update File: path/to/file\n\
@@ optional_anchor_line\n\
 unchanged context line\n\
-removed line\n\
+added line\n\
*** Add File: path/to/new\n\
+every line of the new file, each prefixed with +\n\
*** Delete File: path/to/old\n\
*** End Patch\n\
Update hunks use NO line numbers: include a few unchanged context lines (each prefixed with a single space) around every change so the hunk can be located, and prefix removed lines with `-` and added lines with `+`. Add `*** Move to: path` on the line after an `*** Update File:` header to rename. One patch may touch several files.",
        json!({
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The full V4A patch, from '*** Begin Patch' to '*** End Patch'."}
            },
            "required": ["input"]
        }),
    )
}

pub(super) fn spec(name: &str, description: &str, parameters: Value) -> ToolSpec {
    ToolSpec {
        name: name.to_string(),
        description: description.to_string(),
        parameters,
    }
}
