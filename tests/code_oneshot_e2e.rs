//! Black-box tests for `aivo code -e`: the real binary and real tool execution,
//! driven end-to-end by the scripted fake model (`AIVO_AGENT_FAKE_SSE`).

mod support;

use std::path::PathBuf;
use std::process::{Command, Output};

use serde_json::Value;
use tempfile::TempDir;

fn aivo_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_aivo") {
        return PathBuf::from(path);
    }

    let mut path = std::env::current_exe().expect("current test exe");
    path.pop(); // test binary name
    if path.ends_with("deps") {
        path.pop();
    }
    path.push(if cfg!(windows) { "aivo.exe" } else { "aivo" });
    path
}

/// Hermetic install: temp HOME + config dir, a throwaway key (never used — the
/// fake model replaces the provider), and a temp project dir for the tools.
struct ExecEnv {
    home: TempDir,
    proj: TempDir,
    config: PathBuf,
}

impl ExecEnv {
    fn new() -> Self {
        let home = TempDir::new().unwrap();
        let proj = TempDir::new().unwrap();
        let config = home.path().join("aivo-config");
        let env = Self { home, proj, config };
        let out = env
            .cmd()
            .args([
                "keys",
                "add",
                "--name",
                "e2e",
                "--base-url",
                "https://api.openai.com/v1",
                "--key",
                "sk-e2e-test",
            ])
            .output()
            .expect("spawn aivo keys add");
        assert!(
            out.status.success(),
            "keys add failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        env
    }

    fn cmd(&self) -> Command {
        let mut cmd = Command::new(aivo_bin());
        cmd.env("HOME", self.home.path())
            .env("USERPROFILE", self.home.path())
            .env("AIVO_CONFIG_DIR", &self.config)
            .env("AIVO_TEST_FAST_CRYPTO_OK", "1")
            .env("NO_COLOR", "1")
            .current_dir(self.proj.path());
        cmd
    }

    fn code_exec(&self, script: &str, task: &str, extra: &[&str]) -> Output {
        self.code_exec_model(script, task, "gpt-4o", extra)
    }

    /// Self-correct and LSP are off so the script's turn sequence is exact.
    fn code_exec_model(&self, script: &str, task: &str, model: &str, extra: &[&str]) -> Output {
        let script_path = self.home.path().join("fake-script.json");
        std::fs::write(&script_path, script).unwrap();
        let mut cmd = self.cmd();
        cmd.env("AIVO_AGENT_FAKE_SSE", &script_path)
            .env("AIVO_FAKE_CAPTURE", self.home.path().join("capture.jsonl"))
            .env("AIVO_AGENT_SELF_CORRECT", "0")
            .env("AIVO_AGENT_LSP", "0")
            .args(["code", "-e", task, "--model", model])
            .args(extra);
        cmd.output().expect("spawn aivo code -e")
    }

    fn session_file(&self, id: &str) -> PathBuf {
        self.config.join("sessions").join(format!("{id}.json"))
    }

    fn first_capture(&self) -> Value {
        let raw = std::fs::read_to_string(self.home.path().join("capture.jsonl")).unwrap();
        serde_json::from_str(raw.lines().next().expect("captured request")).unwrap()
    }
}

fn last_user_content(req: &Value) -> &Value {
    let msg = req["messages"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find(|m| m["role"] == "user")
        .expect("user message");
    &msg["content"]
}

fn stdout_str(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout utf8")
}

fn stderr_str(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr utf8")
}

const WRITE_SCRIPT: &str = r#"[
  {"tools": [{"name": "write_file", "args": {"path": "hello.txt", "content": "hi\n"}}]},
  {"text": "wrote the file"}
]"#;

const OK_SCHEMA: &str =
    r#"{"type":"object","required":["ok"],"properties":{"ok":{"type":"boolean"}}}"#;

#[test]
fn exec_text_runs_tools_writes_files_and_persists_a_resumable_session() {
    let env = ExecEnv::new();
    let out = env.code_exec(WRITE_SCRIPT, "write hello.txt", &[]);
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));

    let written = std::fs::read_to_string(env.proj.path().join("hello.txt")).unwrap();
    assert_eq!(written, "hi\n");
    assert!(stdout_str(&out).contains("wrote the file"));
    let err = stderr_str(&out);
    assert!(err.contains("write_file"), "stderr:\n{err}");
    let hint = err
        .lines()
        .find(|l| l.contains("continue with --resume"))
        .unwrap_or_else(|| panic!("no resume hint in stderr:\n{err}"));
    let id = hint
        .split_whitespace()
        .last()
        .unwrap()
        .trim_end_matches(']');
    let session: Value =
        serde_json::from_str(&std::fs::read_to_string(env.session_file(id)).unwrap()).unwrap();
    assert_eq!(session["messages"].as_array().unwrap().len(), 2);
    assert!(
        !session["engineMessages"].as_array().unwrap().is_empty(),
        "engine transcript must persist for --resume fidelity"
    );
}

#[test]
fn exec_json_emits_one_result_document() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        WRITE_SCRIPT,
        "write hello.txt",
        &["--output-format", "json"],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));

    let doc: Value = serde_json::from_str(stdout_str(&out).trim()).unwrap_or_else(|e| {
        panic!(
            "stdout must be one JSON document ({e}):\n{}",
            stdout_str(&out)
        )
    });
    assert_eq!(doc["type"], "result");
    assert_eq!(doc["exit"], 0);
    assert_eq!(doc["sessionSaved"], true);
    assert!(doc["answer"].as_str().unwrap().contains("wrote the file"));
    assert!(!doc["sessionId"].as_str().unwrap().is_empty());
}

#[test]
fn exec_stream_json_emits_the_full_event_envelope() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        WRITE_SCRIPT,
        "write hello.txt",
        &["--output-format", "stream-json"],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));

    let events: Vec<Value> = stdout_str(&out)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad event line ({e}): {l}")))
        .collect();
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types.first(), Some(&"run_start"));
    assert_eq!(types.last(), Some(&"run_end"));
    for needed in ["tool_call", "tool_result", "usage", "final"] {
        assert!(types.contains(&needed), "missing {needed} in {types:?}");
    }
    let run_id = events[0]["runId"].as_str().unwrap();
    for e in &events {
        assert_eq!(e["schemaVersion"], 1);
        assert_eq!(e["runId"], run_id);
    }
    let tool = events.iter().find(|e| e["type"] == "tool_call").unwrap();
    assert_eq!(tool["tool"], "write_file");
    let result = events.iter().find(|e| e["type"] == "tool_result").unwrap();
    assert_eq!(result["ok"], true);
    let end = events.last().unwrap();
    assert_eq!(end["exit"], 0);
}

#[test]
fn exec_resume_continues_the_same_session() {
    let env = ExecEnv::new();
    let first = env.code_exec(
        r#"[{"text": "first answer"}]"#,
        "start",
        &["--output-format", "json"],
    );
    assert!(first.status.success(), "stderr:\n{}", stderr_str(&first));
    let doc: Value = serde_json::from_str(stdout_str(&first).trim()).unwrap();
    let id = doc["sessionId"].as_str().unwrap().to_string();

    let second = env.code_exec(
        r#"[{"text": "second answer"}]"#,
        "follow up",
        &["--output-format", "json", "--resume", &id],
    );
    assert!(second.status.success(), "stderr:\n{}", stderr_str(&second));
    let doc2: Value = serde_json::from_str(stdout_str(&second).trim()).unwrap();
    assert_eq!(doc2["sessionId"], id.as_str(), "resume must keep the id");

    let session: Value =
        serde_json::from_str(&std::fs::read_to_string(env.session_file(&id)).unwrap()).unwrap();
    let messages = session["messages"].as_array().unwrap();
    assert_eq!(
        messages.len(),
        4,
        "two turns = user/assistant × 2: {messages:?}"
    );
    assert!(
        messages[3]["content"]
            .as_str()
            .unwrap()
            .contains("second answer")
    );
}

#[test]
fn json_schema_conforming_answer_passes_through() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        r#"[{"text": "{\"ok\": true}"}]"#,
        "emit ok",
        &["--json-schema", OK_SCHEMA],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));
    let answer: Value = serde_json::from_str(stdout_str(&out).trim()).unwrap();
    assert_eq!(answer, serde_json::json!({"ok": true}));
}

#[test]
fn json_schema_invalid_answer_gets_a_corrective_retry() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        r#"[
            {"text": "here is your answer: ok=true"},
            {"text": "{\"ok\": true}"}
        ]"#,
        "emit ok",
        &["--json-schema", OK_SCHEMA],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));
    let stdout = stdout_str(&out);
    let answer: Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout must be the corrected JSON ({e}):\n{stdout}"));
    assert_eq!(answer, serde_json::json!({"ok": true}));
    assert!(!stdout.contains("here is your answer"));
    let err = stderr_str(&out);
    assert!(
        err.contains("asking the model to correct it"),
        "stderr:\n{err}"
    );
}

#[test]
fn json_schema_never_valid_fails_the_run_with_validation_errors() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        r#"[{"text": "nope"}, {"text": "still nope"}, {"text": "nope again"}]"#,
        "emit ok",
        &["--json-schema", OK_SCHEMA, "--output-format", "stream-json"],
    );
    assert_eq!(out.status.code(), Some(1), "stderr:\n{}", stderr_str(&out));

    let events: Vec<Value> = stdout_str(&out)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let violations: Vec<&Value> = events
        .iter()
        .filter(|e| e["type"] == "validation_error")
        .collect();
    // Initial attempt + two corrective turns, then the terminal failure.
    assert_eq!(violations.len(), 3, "events: {events:?}");
    assert_eq!(violations[0]["retrying"], true);
    assert_eq!(violations[2]["retrying"], false);
    let end = events.last().unwrap();
    assert_eq!(end["type"], "run_end");
    assert_eq!(end["exit"], 1);
}

const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn write_png(dir: &std::path::Path) -> PathBuf {
    use base64::Engine as _;
    let path = dir.join("shot.png");
    std::fs::write(
        &path,
        base64::engine::general_purpose::STANDARD
            .decode(PNG_1X1)
            .unwrap(),
    )
    .unwrap();
    path
}

#[test]
fn exec_attach_text_file_rides_the_opening_message() {
    let env = ExecEnv::new();
    let notes = env.proj.path().join("notes.txt");
    std::fs::write(&notes, "the deploy failed at step 7\n").unwrap();
    let out = env.code_exec(
        r#"[{"text": "diagnosed"}]"#,
        "read the attached log",
        &[
            "--attach",
            notes.to_str().unwrap(),
            "--output-format",
            "json",
        ],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));

    let req = env.first_capture();
    let parts = last_user_content(&req)
        .as_array()
        .expect("multimodal parts");
    assert!(
        parts.iter().any(|p| p["type"] == "text"
            && p["text"]
                .as_str()
                .unwrap()
                .contains("the deploy failed at step 7")),
        "attachment text must reach the wire: {parts:?}"
    );
    assert!(
        parts
            .iter()
            .any(|p| p["type"] == "text" && p["text"].as_str().unwrap().contains("attached log")),
        "prompt text must reach the wire: {parts:?}"
    );

    let doc: Value = serde_json::from_str(stdout_str(&out).trim()).unwrap();
    let id = doc["sessionId"].as_str().unwrap();
    let session: Value =
        serde_json::from_str(&std::fs::read_to_string(env.session_file(id)).unwrap()).unwrap();
    assert_eq!(
        session["messages"][0]["attachments"][0]["name"],
        "notes.txt"
    );
}

#[test]
fn exec_attach_image_becomes_an_image_part() {
    let env = ExecEnv::new();
    let img = write_png(env.proj.path());
    let out = env.code_exec(
        r#"[{"text": "described"}]"#,
        "describe the screenshot",
        &["--attach", img.to_str().unwrap()],
    );
    assert!(out.status.success(), "stderr:\n{}", stderr_str(&out));

    let req = env.first_capture();
    let parts = last_user_content(&req)
        .as_array()
        .expect("multimodal parts");
    assert!(
        parts.iter().any(|p| p["type"] == "image_url"
            && p["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/")),
        "image must reach the wire as a data URL: {parts:?}"
    );
}

#[test]
fn exec_attach_image_rejected_for_a_known_non_vision_model() {
    let env = ExecEnv::new();
    let img = write_png(env.proj.path());
    let out = env.code_exec_model(
        r#"[{"text": "unreachable"}]"#,
        "describe the screenshot",
        "deepseek-chat",
        &["--attach", img.to_str().unwrap()],
    );
    assert!(!out.status.success());
    assert!(
        stderr_str(&out).contains("can't read images"),
        "stderr:\n{}",
        stderr_str(&out)
    );
}

#[cfg(unix)]
#[test]
fn persist_failure_warns_and_withholds_the_resume_hint() {
    use std::os::unix::fs::PermissionsExt;

    let env = ExecEnv::new();
    let sessions = env.config.join("sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    std::fs::set_permissions(&sessions, std::fs::Permissions::from_mode(0o555)).unwrap();

    let text = env.code_exec(r#"[{"text": "the answer"}]"#, "answer me", &[]);
    let json = env.code_exec(
        r#"[{"text": "the answer"}]"#,
        "answer me",
        &["--output-format", "json"],
    );
    // Restore before asserting so TempDir cleanup always works.
    std::fs::set_permissions(&sessions, std::fs::Permissions::from_mode(0o755)).unwrap();

    assert!(text.status.success(), "stderr:\n{}", stderr_str(&text));
    assert!(stdout_str(&text).contains("the answer"));
    let err = stderr_str(&text);
    assert!(err.contains("failed to save session"), "stderr:\n{err}");
    assert!(
        !err.contains("continue with --resume"),
        "must not advertise an unresumable session:\n{err}"
    );

    assert!(json.status.success(), "stderr:\n{}", stderr_str(&json));
    let doc: Value = serde_json::from_str(stdout_str(&json).trim()).unwrap();
    assert_eq!(doc["exit"], 0);
    assert_eq!(doc["sessionSaved"], false);
}

#[test]
fn exec_step_limit_exits_1_with_a_typed_stop_reason() {
    let env = ExecEnv::new();
    let out = env.code_exec(
        r#"[
            {"tools": [{"name": "run_bash", "args": {"command": "echo step1"}}]},
            {"tools": [{"name": "run_bash", "args": {"command": "echo step2"}}]},
            {"text": "done"}
        ]"#,
        "loop forever",
        &["--output-format", "json", "--max-steps", "1"],
    );
    assert_eq!(out.status.code(), Some(1), "stderr:\n{}", stderr_str(&out));

    let doc: Value = serde_json::from_str(stdout_str(&out).trim()).unwrap();
    assert_eq!(doc["exit"], 1);
    assert_eq!(doc["stopReason"], "stepLimit");
    assert_eq!(doc["error"], Value::Null, "a stop is not an engine error");
    let err = stderr_str(&out);
    assert!(err.contains("run stopped early"), "stderr:\n{err}");
}

#[test]
fn exec_output_budget_stop_emits_a_stopped_event_and_exits_1() {
    let env = ExecEnv::new();
    // Scripted usage pushes the turn past --max-output-tokens on step 1.
    let out = env.code_exec(
        r#"[
            {"tools": [{"name": "run_bash", "args": {"command": "echo hi"}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 500}},
            {"text": "done"}
        ]"#,
        "burn the budget",
        &[
            "--output-format",
            "stream-json",
            "--max-output-tokens",
            "100",
        ],
    );
    assert_eq!(out.status.code(), Some(1), "stderr:\n{}", stderr_str(&out));

    let events: Vec<Value> = stdout_str(&out)
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad event line ({e}): {l}")))
        .collect();
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    let stopped = types
        .iter()
        .position(|t| *t == "stopped")
        .expect("stopped event");
    assert_eq!(events[stopped]["reason"], "outputBudget");
    let fin = types
        .iter()
        .position(|t| *t == "final")
        .expect("final event");
    assert!(stopped < fin, "stopped must precede final: {types:?}");
    assert_eq!(events.last().unwrap()["exit"], 1);
}
