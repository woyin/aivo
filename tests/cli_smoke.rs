use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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

fn aivo(home: &TempDir) -> Command {
    let mut cmd = Command::new(aivo_bin());
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("AIVO_TEST_FAST_CRYPTO_OK", "1")
        .env("NO_COLOR", "1");
    cmd
}

fn run_ok(home: &TempDir, args: &[&str]) -> String {
    let output = aivo(home)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn aivo {args:?}: {e}"));
    assert!(
        output.status.success(),
        "aivo {args:?} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).expect("stdout utf8")
}

#[test]
fn version_and_help_work_without_config() {
    let home = TempDir::new().unwrap();

    let version = run_ok(&home, &["--version"]);
    assert!(version.contains(env!("CARGO_PKG_VERSION")));

    let help = run_ok(&home, &["--help"]);
    assert!(help.contains("aivo"));
    assert!(help.contains("keys"));
    assert!(help.contains("run"));
}

#[test]
fn key_lifecycle_and_dry_run_use_real_binary() {
    let home = TempDir::new().unwrap();

    let added = run_ok(
        &home,
        &[
            "keys",
            "add",
            "--name",
            "smoke-key",
            "--base-url",
            "https://api.openai.com/v1",
            "--key",
            "sk-smoke-test",
        ],
    );
    assert!(added.contains("Added and activated key"));

    let list = run_ok(&home, &["keys", "--json"]);
    let keys: Vec<Value> = serde_json::from_str(&list).expect("keys json");
    assert!(keys.iter().any(|k| k["name"] == "smoke-key"));
    assert!(
        keys.iter().all(|k| k.get("key").is_none()),
        "list json must not expose secrets: {keys:?}"
    );

    let cat = run_ok(&home, &["keys", "cat", "smoke-key"]);
    assert!(cat.contains("Name:"));
    assert!(cat.contains("smoke-key"));
    assert!(cat.contains("sk-smoke-test"));

    let dry_run = run_ok(
        &home,
        &[
            "run",
            "codex",
            "--key",
            "smoke-key",
            "--model",
            "gpt-5",
            "--dry-run",
        ],
    );
    assert!(dry_run.contains("codex"));
    assert!(dry_run.contains("gpt-5"));

    let mut rm = aivo(&home)
        .args(["keys", "rm", "smoke-key"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn aivo keys rm");
    rm.stdin
        .as_mut()
        .expect("rm stdin")
        .write_all(b"y\n")
        .expect("write rm confirmation");
    let output = rm.wait_with_output().expect("wait rm");
    assert!(
        output.status.success(),
        "aivo keys rm failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let list_after = run_ok(&home, &["keys", "--json"]);
    let keys_after: Vec<Value> = serde_json::from_str(&list_after).expect("keys json after rm");
    assert!(!keys_after.iter().any(|k| k["name"] == "smoke-key"));
}
