//! Post-edit self-verification: detect the project's validator, run it at declared-done,
//! feed failures back so the run can't finish red. Default on for headless `-e`
//! (`AIVO_AGENT_SELF_CORRECT=0` opts out); opt-in (`=1`) for interactive turns, where a
//! surprise full-suite run would stall a watched turn.
//!
//! Detection is best-effort and conservative: a recognized validator or nothing. Only
//! the agent's declared-done moment triggers a run, so it isn't run after every edit.

use std::path::Path;
use std::time::Duration;

/// A detected project validator: a human label + the argv to run in the workspace.
#[derive(Clone)]
pub struct Validator {
    pub label: String,
    command: Vec<String>,
}

impl Validator {
    fn new(label: &str, command: &[&str]) -> Self {
        Self {
            label: label.to_string(),
            command: command.iter().map(|s| (*s).to_string()).collect(),
        }
    }
}

/// A validator that overruns this is treated as inconclusive (Ok), not a failure —
/// better to accept the answer than to loop the agent on a hanging suite.
const VERIFY_TIMEOUT: Duration = Duration::from_secs(120);

/// The project's primary validator, or `None` if the workspace isn't recognized.
/// Ordered cheapest / most-explicit first so a repo's own entrypoint wins over a
/// heavier language default.
pub fn detect(cwd: &Path) -> Option<Validator> {
    if cwd.join("run_tests.sh").is_file() {
        return Some(Validator::new("run_tests.sh", &["sh", "run_tests.sh"]));
    }
    if makefile_has_target(cwd, "test") {
        return Some(Validator::new("make test", &["make", "test"]));
    }
    if makefile_has_target(cwd, "check") {
        return Some(Validator::new("make check", &["make", "check"]));
    }
    if package_json_has_test(cwd) {
        return Some(Validator::new("npm test", &["npm", "test", "--silent"]));
    }
    if cwd.join("Cargo.toml").is_file() {
        return Some(Validator::new("cargo test", &["cargo", "test"]));
    }
    if cwd.join("go.mod").is_file() {
        return Some(Validator::new("go test", &["go", "test", "./..."]));
    }
    if cwd.join("pytest.ini").is_file() || pyproject_has_pytest(cwd) {
        return Some(Validator::new("pytest", &["pytest", "-q"]));
    }
    None
}

/// Run `v` in `cwd`. `Ok(())` when it passes (or is inconclusive: can't launch, or
/// times out — never block the agent on those); `Err(summary)` with a short failure
/// tail when it fails, for feeding back to the model.
/// Takes the `Validator` by value (not `&Validator`): a borrow held across the await
/// below would make this future's Send-ness higher-ranked, breaking callers that fan
/// it out concurrently (`buffer_unordered`) or `tokio::spawn` it. The caller clones.
pub async fn run(v: Validator, cwd: &Path) -> Result<(), String> {
    let mut cmd = tokio::process::Command::new(&v.command[0]);
    cmd.args(&v.command[1..])
        .current_dir(cwd)
        .stdin(std::process::Stdio::null());
    let output = match tokio::time::timeout(VERIFY_TIMEOUT, cmd.output()).await {
        Ok(Ok(o)) => o,
        // Missing tool / spawn error / timeout → inconclusive, don't fail the run.
        Ok(Err(_)) | Err(_) => return Ok(()),
    };
    if output.status.success() {
        return Ok(());
    }
    Err(summarize_failure(&v.label, &output.stdout, &output.stderr))
}

/// Build a compact failure message: the label + the last few non-empty output lines
/// (tests print the reason near the end), capped so it can't blow up the context.
fn summarize_failure(label: &str, stdout: &[u8], stderr: &[u8]) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for chunk in [stderr, stdout] {
        let text = std::str::from_utf8(chunk).unwrap_or("");
        for line in text.lines() {
            if !line.trim().is_empty() {
                lines.push(line);
            }
        }
    }
    let tail: Vec<&str> = lines.iter().rev().take(12).rev().copied().collect();
    let mut body = tail.join("\n");
    if body.len() > 2000 {
        body.truncate(2000);
        body.push_str("\n… (truncated)");
    }
    if body.is_empty() {
        body = "(no output)".to_string();
    }
    format!("`{label}` failed:\n{body}")
}

/// Whether a Makefile in `cwd` declares a `<target>:` rule.
fn makefile_has_target(cwd: &Path, target: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(cwd.join("Makefile")) else {
        return false;
    };
    text.lines().any(|l| {
        let l = l.trim_start();
        l.strip_prefix(target)
            .is_some_and(|rest| rest.trim_start().starts_with(':'))
    })
}

/// Whether `package.json` defines a real `scripts.test` (not npm's default stub).
fn package_json_has_test(cwd: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(cwd.join("package.json")) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    v.get("scripts")
        .and_then(|s| s.get("test"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| !t.contains("no test specified"))
}

/// Whether `pyproject.toml` configures pytest.
fn pyproject_has_pytest(cwd: &Path) -> bool {
    std::fs::read_to_string(cwd.join("pyproject.toml"))
        .map(|t| t.contains("[tool.pytest"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp() -> std::path::PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-verify-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn detect_prefers_explicit_entrypoints_and_reads_makefile_targets() {
        let d = tmp();
        assert!(detect(&d).is_none()); // empty workspace → nothing

        std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
        assert_eq!(detect(&d).unwrap().label, "cargo test");

        // A run_tests.sh wins over the Cargo default.
        std::fs::write(d.join("run_tests.sh"), "exit 0").unwrap();
        assert_eq!(detect(&d).unwrap().label, "run_tests.sh");
    }

    #[test]
    fn makefile_target_detection_is_precise() {
        let d = tmp();
        std::fs::write(d.join("Makefile"), "build:\n\tcc x.c\ntest:\n\techo ok\n").unwrap();
        assert!(makefile_has_target(&d, "test"));
        assert!(makefile_has_target(&d, "build"));
        assert!(!makefile_has_target(&d, "lint"));
        // `testfoo:` must not match `test`.
        std::fs::write(d.join("Makefile"), "testfoo:\n\techo no\n").unwrap();
        assert!(!makefile_has_target(&d, "test"));
    }

    #[test]
    fn package_json_test_ignores_the_npm_default_stub() {
        let d = tmp();
        std::fs::write(
            d.join("package.json"),
            r#"{"scripts":{"test":"echo \"Error: no test specified\" && exit 1"}}"#,
        )
        .unwrap();
        assert!(!package_json_has_test(&d));
        std::fs::write(
            d.join("package.json"),
            r#"{"scripts":{"test":"vitest run"}}"#,
        )
        .unwrap();
        assert!(package_json_has_test(&d));
    }

    #[test]
    fn summarize_keeps_the_failing_tail_and_caps_size() {
        let out = summarize_failure(
            "make test",
            b"line1\n\nline2\n",
            b"boom: assertion failed\n",
        );
        assert!(out.starts_with("`make test` failed:"));
        assert!(out.contains("boom: assertion failed"));
        assert!(out.contains("line2"));
        let big = vec![b'x'; 5000];
        assert!(summarize_failure("t", &big, b"").len() < 2100);
    }
}
