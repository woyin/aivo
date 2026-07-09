//! User lifecycle hooks (`~/.config/aivo/hooks.json`, Claude Code-shaped): PreToolUse
//! (exit 2 vetoes the call), PostToolUse (output folded into the tool result), Stop
//! (exit 2 refuses the stop, stderr = guidance). JSON payload on stdin. User-authored →
//! unsandboxed; failures/timeouts fail OPEN — the permission tiers stay the security floor.

use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

/// Overrun → abandoned (fail-open), so a hung script can't wedge a turn.
const DEFAULT_TIMEOUT_SECS: u64 = 60;
/// Byte caps on hook output / payload fields, so context can't be flooded.
const OUTPUT_CAP: usize = 16 * 1024;
const PAYLOAD_FIELD_CAP: usize = 8 * 1024;

#[derive(Deserialize)]
struct HooksFile {
    #[serde(default)]
    hooks: HooksByEvent,
}

#[derive(Deserialize, Default)]
struct HooksByEvent {
    #[serde(rename = "PreToolUse", default)]
    pre_tool_use: Vec<HookRule>,
    #[serde(rename = "PostToolUse", default)]
    post_tool_use: Vec<HookRule>,
    #[serde(rename = "Stop", default)]
    stop: Vec<HookRule>,
}

/// `matcher`: `""`/`"*"` = all tools, else `|`-separated exact names (Stop ignores it).
#[derive(Deserialize, Clone)]
pub struct HookRule {
    #[serde(default)]
    matcher: String,
    #[serde(default)]
    hooks: Vec<HookCmd>,
}

#[derive(Deserialize, Clone)]
pub struct HookCmd {
    command: String,
    #[serde(default)]
    timeout: Option<u64>,
}

#[derive(Default)]
pub struct HookSet {
    pre_tool_use: Vec<HookRule>,
    post_tool_use: Vec<HookRule>,
    stop: Vec<HookRule>,
}

/// `Block` carries the script's stderr (reason/guidance).
enum HookVerdict {
    Pass(Option<String>),
    Block(String),
}

impl HookSet {
    /// Missing/malformed → empty set (a broken config must not brick the agent).
    /// Project-scope hooks are deliberately not read: repo commands = RCE-on-open.
    pub fn load_default() -> Self {
        let Some(home) = crate::services::system_env::home_dir() else {
            return Self::default();
        };
        Self::load_from(&home.join(".config/aivo/hooks.json"))
    }

    pub fn load_from(path: &Path) -> Self {
        let Ok(raw) = std::fs::read_to_string(path) else {
            return Self::default();
        };
        let Ok(file) = serde_json::from_str::<HooksFile>(&raw) else {
            return Self::default();
        };
        Self {
            pre_tool_use: file.hooks.pre_tool_use,
            post_tool_use: file.hooks.post_tool_use,
            stop: file.hooks.stop,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pre_tool_use.is_empty() && self.post_tool_use.is_empty() && self.stop.is_empty()
    }

    pub fn has_post(&self) -> bool {
        !self.post_tool_use.is_empty()
    }

    /// First veto (exit 2) wins; `None` = allowed (permission tiers still apply).
    pub async fn pre_tool_use_deny(&self, tool: &str, args: &Value, cwd: &Path) -> Option<String> {
        let payload = json!({
            "event": "PreToolUse",
            "tool": tool,
            "args": args,
            "cwd": cwd.display().to_string(),
        });
        for rule in matching(&self.pre_tool_use, tool) {
            for cmd in &rule.hooks {
                if let HookVerdict::Block(reason) = run_hook(cmd, &payload, cwd).await {
                    return Some(reason);
                }
            }
        }
        None
    }

    /// What matching hooks want folded into the tool result: stdout, or exit-2 stderr.
    pub async fn post_tool_use(
        &self,
        tool: &str,
        args: &Value,
        result: &Result<String, String>,
        cwd: &Path,
    ) -> Option<String> {
        let (ok, output) = match result {
            Ok(s) => (true, s),
            Err(e) => (false, e),
        };
        let payload = json!({
            "event": "PostToolUse",
            "tool": tool,
            "args": args,
            "ok": ok,
            "output": cap(output, PAYLOAD_FIELD_CAP),
            "cwd": cwd.display().to_string(),
        });
        let mut extra = Vec::new();
        for rule in matching(&self.post_tool_use, tool) {
            for cmd in &rule.hooks {
                match run_hook(cmd, &payload, cwd).await {
                    HookVerdict::Pass(Some(out)) => extra.push(out),
                    HookVerdict::Block(feedback) => extra.push(feedback),
                    HookVerdict::Pass(None) => {}
                }
            }
        }
        (!extra.is_empty()).then(|| extra.join("\n"))
    }

    /// First refusal (exit 2) returns its guidance; the engine feeds it back (bounded).
    pub async fn stop_guidance(&self, answer: &str, cwd: &Path) -> Option<String> {
        let payload = json!({
            "event": "Stop",
            "answer": cap(answer, PAYLOAD_FIELD_CAP),
            "cwd": cwd.display().to_string(),
        });
        for rule in &self.stop {
            for cmd in &rule.hooks {
                if let HookVerdict::Block(guidance) = run_hook(cmd, &payload, cwd).await {
                    return Some(guidance);
                }
            }
        }
        None
    }
}

fn matching<'a>(rules: &'a [HookRule], tool: &'a str) -> impl Iterator<Item = &'a HookRule> {
    rules.iter().filter(move |r| {
        let m = r.matcher.trim();
        m.is_empty() || m == "*" || m.split('|').any(|t| t.trim() == tool)
    })
}

fn cap(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    // Back off so the cap can't split a multibyte char.
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Exit 2 → Block(stderr); exit 0 → Pass(stdout); anything else (spawn failure,
/// timeout, other exits) → Pass(None) — fail-open by design.
async fn run_hook(cmd: &HookCmd, payload: &Value, cwd: &Path) -> HookVerdict {
    let inv = crate::agent::sandbox::bare_shell(&cmd.command);
    let mut command = tokio::process::Command::new(&inv.program);
    command
        .args(&inv.args)
        .current_dir(cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    let Ok(mut child) = command.spawn() else {
        return HookVerdict::Pass(None);
    };
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(payload.to_string().as_bytes()).await;
        // Dropped here → EOF, so a hook reading stdin to end can't deadlock.
    }
    let timeout = Duration::from_secs(cmd.timeout.unwrap_or(DEFAULT_TIMEOUT_SECS).max(1));
    let out = match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(out)) => out,
        _ => return HookVerdict::Pass(None),
    };
    match out.status.code() {
        Some(0) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let trimmed = stdout.trim();
            if trimmed.is_empty() {
                HookVerdict::Pass(None)
            } else {
                HookVerdict::Pass(Some(cap(trimmed, OUTPUT_CAP).to_string()))
            }
        }
        Some(2) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let msg = stderr.trim();
            let msg = if msg.is_empty() {
                "(no reason given)"
            } else {
                msg
            };
            HookVerdict::Block(cap(msg, OUTPUT_CAP).to_string())
        }
        _ => HookVerdict::Pass(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-hooks-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn set(json_str: &str) -> HookSet {
        let dir = tmp();
        let path = dir.join("hooks.json");
        std::fs::write(&path, json_str).unwrap();
        HookSet::load_from(&path)
    }

    #[test]
    fn missing_or_malformed_config_loads_empty() {
        assert!(HookSet::load_from(Path::new("/nonexistent/hooks.json")).is_empty());
        assert!(set("{not json").is_empty());
        assert!(set("{}").is_empty());
    }

    #[test]
    fn matcher_selects_all_exact_and_pipe_lists() {
        let rules = vec![
            HookRule {
                matcher: "*".into(),
                hooks: vec![],
            },
            HookRule {
                matcher: "run_bash|write_file".into(),
                hooks: vec![],
            },
            HookRule {
                matcher: "edit_file".into(),
                hooks: vec![],
            },
        ];
        assert_eq!(matching(&rules, "run_bash").count(), 2);
        assert_eq!(matching(&rules, "edit_file").count(), 2);
        assert_eq!(matching(&rules, "read_file").count(), 1); // only "*"
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pre_tool_use_exit_2_vetoes_with_stderr_reason() {
        let hooks = set(r#"{"hooks":{"PreToolUse":[{"matcher":"run_bash","hooks":[
                {"command":"echo not on my watch >&2; exit 2"}]}]}}"#);
        let cwd = tmp();
        let deny = hooks
            .pre_tool_use_deny("run_bash", &json!({"command":"rm x"}), &cwd)
            .await;
        assert_eq!(deny.as_deref(), Some("not on my watch"));
        // Non-matching tool: allowed.
        assert!(
            hooks
                .pre_tool_use_deny("read_file", &json!({}), &cwd)
                .await
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn pre_tool_use_reads_the_json_payload_on_stdin() {
        // The hook vetoes only when the payload names the tool it dislikes —
        // proving stdin carries the structured payload.
        let hooks = set(r#"{"hooks":{"PreToolUse":[{"matcher":"*","hooks":[
                {"command":"grep -q '\"tool\":\"write_file\"' && { echo no writes >&2; exit 2; } || exit 0"}]}]}}"#);
        let cwd = tmp();
        assert!(
            hooks
                .pre_tool_use_deny("write_file", &json!({}), &cwd)
                .await
                .is_some()
        );
        assert!(
            hooks
                .pre_tool_use_deny("read_file", &json!({}), &cwd)
                .await
                .is_none()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn post_tool_use_folds_stdout_and_exit2_stderr() {
        let hooks = set(r#"{"hooks":{"PostToolUse":[
                {"matcher":"write_file","hooks":[{"command":"echo lint: looks fine"}]},
                {"matcher":"write_file","hooks":[{"command":"echo style nit >&2; exit 2"}]}]}}"#);
        let cwd = tmp();
        let extra = hooks
            .post_tool_use("write_file", &json!({}), &Ok("done".into()), &cwd)
            .await
            .unwrap();
        assert!(extra.contains("lint: looks fine"));
        assert!(extra.contains("style nit"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stop_exit_2_returns_guidance() {
        let hooks = set(
            r#"{"hooks":{"Stop":[{"hooks":[{"command":"echo run the tests first >&2; exit 2"}]}]}}"#,
        );
        let cwd = tmp();
        assert_eq!(
            hooks.stop_guidance("all done", &cwd).await.as_deref(),
            Some("run the tests first")
        );
        let allow = set(r#"{"hooks":{"Stop":[{"hooks":[{"command":"exit 0"}]}]}}"#);
        assert!(allow.stop_guidance("all done", &cwd).await.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failures_and_timeouts_fail_open() {
        let hooks = set(r#"{"hooks":{"PreToolUse":[{"matcher":"*","hooks":[
                {"command":"exit 1"},
                {"command":"/definitely/not/a/binary"},
                {"command":"sleep 30", "timeout": 1}]}]}}"#);
        let cwd = tmp();
        assert!(
            hooks
                .pre_tool_use_deny("run_bash", &json!({}), &cwd)
                .await
                .is_none()
        );
    }
}
