mod support;

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

// Only the `#[cfg(unix)]` fake-binary tests use this; unused on Windows.
#[cfg(unix)]
fn prepend_path(cmd: &mut Command, dir: &std::path::Path) {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(existing) = std::env::var_os("PATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    cmd.env("PATH", std::env::join_paths(paths).expect("join PATH"));
}

#[cfg(unix)]
fn fake_cursor_agent(dir: &TempDir) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let path = dir.path().join("cursor-agent");
    std::fs::write(
        &path,
        r#"#!/bin/sh
if [ "$1" = "models" ]; then
  printf '%s\n' 'composer-2.5' 'gpt-5'
  exit 0
fi
if [ "$1" = "--list-models" ]; then
  printf '%s\n' 'fallback-model'
  exit 0
fi
if [ "$1" = "status" ]; then
  printf '%s\n' '{"authenticated":true}'
  exit 0
fi
exit 2
"#,
    )
    .expect("write fake cursor-agent");
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

fn run_ok(home: &TempDir, args: &[&str]) -> String {
    run_ok_env(home, &[], args)
}

fn run_ok_env(home: &TempDir, envs: &[(&str, &str)], args: &[&str]) -> String {
    let mut cmd = aivo(home);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let output = cmd
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

#[cfg(unix)]
#[test]
fn cursor_key_and_models_use_cursor_agent() {
    let home = TempDir::new().unwrap();
    let fake_bin = TempDir::new().unwrap();
    let _agent = fake_cursor_agent(&fake_bin);

    let mut add = aivo(&home);
    prepend_path(&mut add, fake_bin.path());
    let output = add
        .args(["keys", "add", "cursor", "--key", "sk-cursor-smoke"])
        .output()
        .expect("aivo keys add cursor");
    assert!(
        output.status.success(),
        "aivo keys add cursor failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    let list = run_ok(&home, &["keys", "--json"]);
    let keys: Vec<Value> = serde_json::from_str(&list).expect("keys json");
    let cursor = keys
        .iter()
        .find(|k| k["name"] == "cursor")
        .expect("cursor key");
    assert_eq!(cursor["base_url"], "cursor");

    let mut models = aivo(&home);
    prepend_path(&mut models, fake_bin.path());
    let output = models
        .args(["models", "-k", "cursor", "--json", "--refresh"])
        .output()
        .expect("aivo models cursor");
    assert!(
        output.status.success(),
        "aivo models cursor failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let payload: Value = serde_json::from_slice(&output.stdout).expect("models json");
    assert_eq!(payload["provider"], "cursor");
    assert_eq!(payload["models"][0]["id"], "composer-2.5");
    assert_eq!(payload["models"][1]["id"], "gpt-5");
}

#[cfg(unix)]
fn fake_plugin(dir: &TempDir, name: &str, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let path = dir.path().join(name);
    std::fs::write(&path, body).expect("write fake plugin");
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

/// Echoes its argv + the handoff env vars, then exits 7 so we can assert the
/// child's exit code propagates through aivo.
#[cfg(unix)]
const PLUGIN_ECHO: &str = "#!/bin/sh\n\
echo \"ARGS: $*\"\n\
echo \"CONFIG: $AIVO_CONFIG_DIR\"\n\
echo \"DEBUGLOG: $AIVO_DEBUG_LOG\"\n\
exit 7\n";

#[cfg(unix)]
#[test]
fn plugin_dispatch_forwards_args_env_and_exit_code() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    fake_plugin(&bin, "aivo-foo", PLUGIN_ECHO);

    let mut cmd = aivo(&home);
    prepend_path(&mut cmd, bin.path());
    let output = cmd
        .args(["foo", "trust", "list"])
        .output()
        .expect("spawn aivo foo");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(output.status.code(), Some(7), "stdout:\n{stdout}");
    assert!(stdout.contains("ARGS: trust list"), "stdout:\n{stdout}");
    assert!(
        stdout.contains(".config/aivo"),
        "config dir must be handed to the plugin\nstdout:\n{stdout}"
    );
    // No --debug → AIVO_DEBUG_LOG unset.
    assert!(!stdout.contains(".jsonl"), "stdout:\n{stdout}");
}

#[cfg(unix)]
#[test]
fn plugin_dispatch_via_run_form_and_debug_handoff() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    fake_plugin(&bin, "aivo-foo", PLUGIN_ECHO);

    // `aivo run foo …` forwards to the same sibling, stripping `run foo`.
    let mut run_form = aivo(&home);
    prepend_path(&mut run_form, bin.path());
    let output = run_form
        .args(["run", "foo", "--debug", "x"])
        .output()
        .expect("spawn aivo run foo");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert_eq!(output.status.code(), Some(7), "stdout:\n{stdout}");
    assert!(stdout.contains("ARGS: --debug x"), "stdout:\n{stdout}");
    // `--debug` in the passthrough args → AIVO_DEBUG_LOG resolved for the plugin.
    assert!(
        stdout.contains(".jsonl"),
        "--debug must yield a debug-log path\nstdout:\n{stdout}"
    );
}

#[cfg(unix)]
#[test]
fn plugin_never_shadows_a_builtin_command() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    // A plugin colliding with the built-in `keys` must be ignored.
    fake_plugin(&bin, "aivo-keys", "#!/bin/sh\necho PLUGIN_RAN\nexit 99\n");

    let mut cmd = aivo(&home);
    prepend_path(&mut cmd, bin.path());
    let output = cmd
        .args(["keys", "--json"])
        .output()
        .expect("spawn aivo keys");
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "stdout:\n{stdout}");
    assert!(
        !stdout.contains("PLUGIN_RAN"),
        "built-in `keys` must win over an aivo-keys plugin\nstdout:\n{stdout}"
    );
    // Built-in keys --json emits a JSON array.
    serde_json::from_str::<Vec<Value>>(&stdout).expect("keys json");
}

#[cfg(unix)]
#[test]
fn plugins_install_makes_it_discoverable_then_remove() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(
        &src,
        "aivo-greet",
        "#!/bin/sh\necho \"greet: $*\"\nexit 0\n",
    );

    // Install from a local path → lands in ~/.config/aivo/plugins (name inferred).
    run_ok(&home, &["plugins", "install", plugin_src.to_str().unwrap()]);

    // It shows up in `plugins list`.
    let list = run_ok(&home, &["plugins", "list"]);
    assert!(list.contains("greet"), "list:\n{list}");

    // It dispatches WITHOUT being on PATH — the managed dir is searched first.
    let greet = run_ok(&home, &["greet", "hi", "there"]);
    assert!(greet.contains("greet: hi there"), "greet:\n{greet}");

    // Remove it (-y: the test runs non-interactively); it leaves the listing.
    run_ok(&home, &["plugins", "remove", "greet", "-y"]);
    let after = run_ok(&home, &["plugins", "list"]);
    assert!(!after.contains("greet"), "after:\n{after}");
}

#[cfg(unix)]
#[test]
fn plugins_remove_without_yes_is_non_interactive_safe() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(&src, "aivo-greet", "#!/bin/sh\nexit 0\n");
    run_ok(&home, &["plugins", "install", plugin_src.to_str().unwrap()]);

    // No -y and no TTY → refuse rather than delete; the plugin survives.
    let output = aivo(&home)
        .args(["plugins", "remove", "greet"])
        .output()
        .expect("spawn aivo plugins remove");
    assert!(
        !output.status.success(),
        "must refuse without --yes when non-interactive"
    );
    assert!(run_ok(&home, &["plugins", "list"]).contains("greet"));
}

#[cfg(unix)]
#[test]
fn plugins_install_rejects_a_reserved_name() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(&src, "aivo-x", "#!/bin/sh\nexit 0\n");

    let output = aivo(&home)
        .args([
            "plugins",
            "install",
            plugin_src.to_str().unwrap(),
            "--name",
            "keys",
        ])
        .output()
        .expect("spawn aivo plugins install");
    assert!(
        !output.status.success(),
        "installing under a built-in name must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("built-in"), "stderr:\n{stderr}");
}

#[cfg(unix)]
#[test]
fn plugins_update_reinstalls_from_recorded_source() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();

    // Install v1 from a local path; aivo records the source.
    fake_plugin(&src, "aivo-greet", "#!/bin/sh\necho v1\n");
    let plugin_src = src.path().join("aivo-greet");
    run_ok(&home, &["plugins", "install", plugin_src.to_str().unwrap()]);
    assert!(run_ok(&home, &["greet"]).contains("v1"));

    // Change the source on disk, then `update` re-fetches from it.
    fake_plugin(&src, "aivo-greet", "#!/bin/sh\necho v2\n");
    run_ok(&home, &["plugins", "update", "greet"]);
    assert!(
        run_ok(&home, &["greet"]).contains("v2"),
        "update should pick up the rebuilt source"
    );
}

/// Self-describes via `--aivo-manifest` (first arg), else dispatches normally.
#[cfg(unix)]
const MANIFEST_PLUGIN: &str = r#"#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  printf '%s\n' '{"name":"widget","version":"0.3.0","protocol":"1","roles":["subcommand"],"capabilities":["endpoint"],"description":"demo"}'
  exit 0
fi
echo "widget ran: $*"
exit 0
"#;

#[cfg(unix)]
#[test]
fn plugins_install_captures_manifest_and_surfaces_it() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(&src, "aivo-widget", MANIFEST_PLUGIN);

    // Generous probe deadline: under full-suite CPU load the default 2s starved
    // the child and flaked this test (the probe is best-effort by design).
    run_ok_env(
        &home,
        &[("AIVO_PLUGIN_PROBE_TIMEOUT_MS", "30000")],
        &["plugins", "install", plugin_src.to_str().unwrap()],
    );

    // The registry captured the probed manifest + an integrity pin.
    let registry_path = home.path().join(".config/aivo/plugins/.registry.json");
    let registry: Value =
        serde_json::from_str(&std::fs::read_to_string(&registry_path).expect("registry file"))
            .expect("registry json");
    let rec = &registry["plugins"]["widget"];
    assert_eq!(rec["manifest"]["version"], "0.3.0");
    assert_eq!(rec["manifest"]["capabilities"][0], "endpoint");
    assert!(
        rec["checksum"]
            .as_str()
            .unwrap_or_default()
            .starts_with("sha256:"),
        "expected a sha256 pin, got:\n{rec}"
    );

    // `plugins list` shows the version + description from the manifest.
    let list = run_ok(&home, &["plugins", "list"]);
    assert!(list.contains("v0.3.0"), "list:\n{list}");
    assert!(list.contains("demo"), "list:\n{list}");

    // `--help-json` enriches the plugin entry from the registry.
    let help: Value = serde_json::from_str(&run_ok(&home, &["--help-json"])).expect("help json");
    let entry = help["plugins"]
        .as_array()
        .expect("plugins array")
        .iter()
        .find(|p| p["name"] == "widget")
        .expect("widget in help-json plugins");
    assert_eq!(entry["version"], "0.3.0");
    assert_eq!(entry["roles"][0], "subcommand");
    assert_eq!(entry["capabilities"][0], "endpoint");
}

#[cfg(unix)]
fn registry_json(home: &TempDir) -> Value {
    let path = home.path().join(".config/aivo/plugins/.registry.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("registry file"))
        .expect("registry json")
}

/// Hangs on the manifest probe; the host's timeout must reap it and install anyway.
#[cfg(unix)]
const SLOW_MANIFEST_PLUGIN: &str = r#"#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  sleep 30
fi
echo "slow ran"
"#;

#[cfg(unix)]
#[test]
fn plugins_install_survives_a_hanging_manifest_probe() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(&src, "aivo-slow", SLOW_MANIFEST_PLUGIN);

    // Must return (the probe is deadline-bounded), installing without a manifest.
    // Tiny deadline: the hang is the point, so don't sit out the default 2s.
    run_ok_env(
        &home,
        &[("AIVO_PLUGIN_PROBE_TIMEOUT_MS", "300")],
        &["plugins", "install", plugin_src.to_str().unwrap()],
    );
    assert!(
        registry_json(&home)["plugins"]["slow"]["manifest"].is_null(),
        "a hung probe must yield no manifest"
    );
    // It still dispatches normally.
    assert!(run_ok(&home, &["slow"]).contains("slow ran"));
}

/// Prints junk and exits non-zero on the probe → no manifest, but still installs.
#[cfg(unix)]
const FAILING_MANIFEST_PLUGIN: &str = r#"#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  echo "not a manifest"
  exit 1
fi
echo "ran fine"
"#;

#[cfg(unix)]
#[test]
fn plugins_install_records_no_manifest_on_probe_failure() {
    let home = TempDir::new().unwrap();
    let src = TempDir::new().unwrap();
    let plugin_src = fake_plugin(&src, "aivo-grumpy", FAILING_MANIFEST_PLUGIN);

    run_ok(&home, &["plugins", "install", plugin_src.to_str().unwrap()]);
    let rec = &registry_json(&home)["plugins"]["grumpy"];
    assert!(
        rec["manifest"].is_null(),
        "a non-zero probe exit must yield no manifest"
    );
    assert!(
        rec["checksum"]
            .as_str()
            .unwrap_or_default()
            .starts_with("sha256:"),
        "but the integrity pin is still recorded"
    );
}
