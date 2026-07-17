//! Smart-source plugin install (`github:` / `npm:` / smarter errors), exercised
//! end-to-end against an in-process mock HTTP server. The `aivo` binary runs as
//! a child pointed at the mock via `AIVO_GITHUB_API` / `AIVO_NPM_REGISTRY`.
#![cfg(unix)]

mod support;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::thread;

use serde_json::Value;
use tempfile::TempDir;

fn aivo_bin() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_aivo") {
        return PathBuf::from(path);
    }
    let mut path = std::env::current_exe().expect("current test exe");
    path.pop();
    if path.ends_with("deps") {
        path.pop();
    }
    path.push("aivo");
    path
}

/// One mock route: requests whose path starts with `prefix` get this response.
struct Route {
    prefix: String,
    content_type: String,
    body: Vec<u8>,
}

fn route(prefix: &str, content_type: &str, body: Vec<u8>) -> Route {
    Route {
        prefix: prefix.to_string(),
        content_type: content_type.to_string(),
        body,
    }
}

/// Bind a loopback port, then serve `routes` from a background thread (one
/// response per connection, `Connection: close`). The thread is abandoned at
/// test exit. Returns `http://127.0.0.1:<port>`.
fn serve(listener: TcpListener, routes: Vec<Route>) {
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            if reader.read_line(&mut request_line).is_err() {
                continue;
            }
            // Drain headers up to the blank line.
            loop {
                let mut h = String::new();
                match reader.read_line(&mut h) {
                    Ok(0) => break,
                    Ok(_) if h == "\r\n" || h == "\n" => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            let path = request_line.split_whitespace().nth(1).unwrap_or("/");
            let resp = routes.iter().find(|r| path.starts_with(&r.prefix));
            let (status, ctype, body): (u16, &str, &[u8]) = match resp {
                Some(r) => (200, &r.content_type, &r.body),
                None => (404, "text/plain", b"not found"),
            };
            let header = format!(
                "HTTP/1.1 {status} OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body);
            let _ = stream.flush();
        }
    });
}

fn bind() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, format!("http://127.0.0.1:{port}"))
}

/// `aivo` child wired to a temp HOME and no proxy (so it reaches the mock).
fn aivo(home: &TempDir) -> Command {
    let mut cmd = Command::new(aivo_bin());
    cmd.env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("AIVO_TEST_FAST_CRYPTO_OK", "1")
        // Generous probe deadline: under full-suite CPU load the default 2s
        // starves a healthy manifest probe and flakes the capture assertions
        // (same guard as cli_smoke.rs).
        .env("AIVO_PLUGIN_PROBE_TIMEOUT_MS", "30000")
        .env("NO_COLOR", "1")
        .env("HTTP_PROXY", "")
        .env("HTTPS_PROXY", "")
        .env("NO_PROXY", "127.0.0.1,localhost");
    cmd
}

fn tar_czf(workdir: &std::path::Path, entry: &str) -> Vec<u8> {
    let out = workdir.join("out.tgz");
    let status = Command::new("tar")
        .arg("-czf")
        .arg(&out)
        .arg("-C")
        .arg(workdir)
        .arg(entry)
        .status()
        .expect("run tar");
    assert!(status.success(), "tar -czf failed");
    std::fs::read(&out).unwrap()
}

fn registry(home: &TempDir) -> Value {
    let path = home.path().join(".config/aivo/plugins/.registry.json");
    serde_json::from_str(&std::fs::read_to_string(&path).expect("registry file")).unwrap()
}

#[test]
fn github_install_resolves_release_asset() {
    // A release tarball wrapping an `aivo-widget` executable.
    let work = TempDir::new().unwrap();
    let exe = work.path().join("aivo-widget");
    std::fs::write(&exe, b"#!/bin/sh\necho 'widget ran'\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    let tgz = tar_czf(work.path(), "aivo-widget");

    let (listener, base) = bind();
    // Sole asset → accepted regardless of host triple.
    let release = format!(
        r#"{{"tag_name":"v1","assets":[{{"name":"aivo-widget.tar.gz","browser_download_url":"{base}/dl/aivo-widget.tar.gz","size":{}}}]}}"#,
        tgz.len()
    );
    serve(
        listener,
        vec![
            route(
                "/repos/o/aivo-widget/releases/latest",
                "application/json",
                release.into_bytes(),
            ),
            route("/dl/aivo-widget.tar.gz", "application/octet-stream", tgz),
        ],
    );

    let home = TempDir::new().unwrap();
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "install", "github:o/aivo-widget", "--trust"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "install failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let installed = home.path().join(".config/aivo/plugins/aivo-widget");
    assert!(installed.exists(), "aivo-widget should be installed");
    // The scheme string is stored verbatim so `update` re-resolves.
    assert_eq!(
        registry(&home)["plugins"]["widget"]["source"],
        "github:o/aivo-widget"
    );

    // It dispatches (`--trust` approved the first run; this session has no TTY).
    let run = aivo(&home).args(["widget"]).output().unwrap();
    assert!(String::from_utf8_lossy(&run.stdout).contains("widget ran"));
}

#[test]
fn github_install_falls_back_to_node_source_tarball() {
    // A GitHub source tarball: single `owner-repo-<sha>/` wrapper holding an
    // npm-style Node package (package.json `bin` + script).
    let work = TempDir::new().unwrap();
    let tree = work.path().join("o-aivo-hello-abc1234");
    std::fs::create_dir_all(tree.join("bin")).unwrap();
    std::fs::write(
        tree.join("package.json"),
        r#"{"name":"aivo-hello","version":"1.0.0","bin":{"aivo-hello":"bin/aivo-hello"}}"#,
    )
    .unwrap();
    std::fs::write(
        tree.join("bin/aivo-hello"),
        "#!/usr/bin/env node\nconsole.log('hello ran');\n",
    )
    .unwrap();
    let tgz = tar_czf(work.path(), "o-aivo-hello-abc1234");

    let (listener, base) = bind();
    // No binary assets → the installer must fall back to tarball_url.
    let release = format!(
        r#"{{"tag_name":"v1","assets":[],"tarball_url":"{base}/tarball/o/aivo-hello/v1"}}"#
    );
    serve(
        listener,
        vec![
            route(
                "/repos/o/aivo-hello/releases/latest",
                "application/json",
                release.into_bytes(),
            ),
            route("/tarball/o/aivo-hello/v1", "application/x-gzip", tgz),
        ],
    );

    let home = TempDir::new().unwrap();
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "install", "github:o/aivo-hello", "--trust"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "install failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let shim = home.path().join(".config/aivo/plugins/aivo-hello");
    let shim_text = std::fs::read_to_string(&shim).expect("shim file");
    assert!(
        shim_text.contains("node"),
        "shim must invoke node: {shim_text}"
    );
    assert!(
        home.path()
            .join(".config/aivo/plugins/aivo-hello.d/bin/aivo-hello")
            .exists(),
        "the source tree should be extracted alongside the shim"
    );
    assert_eq!(
        registry(&home)["plugins"]["hello"]["source"],
        "github:o/aivo-hello"
    );

    // It dispatches.
    let run = aivo(&home).args(["hello"]).output().unwrap();
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("hello ran"),
        "dispatch failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
}

#[test]
fn github_install_without_assets_or_node_package_keeps_asset_error() {
    // Source tarball that is NOT a Node package (no package.json) → the
    // fallback must decline and the no-asset error must survive.
    let work = TempDir::new().unwrap();
    let tree = work.path().join("o-aivo-rusty-abc1234");
    std::fs::create_dir_all(&tree).unwrap();
    std::fs::write(tree.join("README.md"), "a rust repo\n").unwrap();
    let tgz = tar_czf(work.path(), "o-aivo-rusty-abc1234");

    let (listener, base) = bind();
    let release = format!(
        r#"{{"tag_name":"v1","assets":[],"tarball_url":"{base}/tarball/o/aivo-rusty/v1"}}"#
    );
    serve(
        listener,
        vec![
            route(
                "/repos/o/aivo-rusty/releases/latest",
                "application/json",
                release.into_bytes(),
            ),
            route("/tarball/o/aivo-rusty/v1", "application/x-gzip", tgz),
        ],
    );

    let home = TempDir::new().unwrap();
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "install", "github:o/aivo-rusty"])
        .output()
        .expect("spawn aivo");
    assert!(!out.status.success(), "install must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no release asset"),
        "stderr should keep the no-asset error:\n{stderr}"
    );
    assert!(
        !home
            .path()
            .join(".config/aivo/plugins/aivo-rusty.d")
            .exists(),
        "the probed bundle must be cleaned up"
    );
}

#[test]
fn npm_install_extracts_and_writes_a_node_shim() {
    // npm tarballs wrap everything in `package/`.
    let work = TempDir::new().unwrap();
    let pkg = work.path().join("package");
    std::fs::create_dir(&pkg).unwrap();
    std::fs::write(
        pkg.join("package.json"),
        r#"{"name":"aivo-foo","version":"1.0.0","bin":{"aivo-foo":"cli.js"}}"#,
    )
    .unwrap();
    std::fs::write(
        pkg.join("cli.js"),
        "#!/usr/bin/env node\nconsole.log('foo');\n",
    )
    .unwrap();
    let tgz = tar_czf(work.path(), "package");

    let (listener, base) = bind();
    let meta = format!(
        r#"{{"dist-tags":{{"latest":"1.0.0"}},"versions":{{"1.0.0":{{"dist":{{"tarball":"{base}/tgz/aivo-foo.tgz"}},"bin":{{"aivo-foo":"cli.js"}}}}}}}}"#
    );
    serve(
        listener,
        vec![
            route("/aivo-foo", "application/json", meta.into_bytes()),
            route("/tgz/aivo-foo.tgz", "application/octet-stream", tgz),
        ],
    );

    let home = TempDir::new().unwrap();
    let out = aivo(&home)
        .env("AIVO_NPM_REGISTRY", &base)
        .args(["plugins", "install", "npm:aivo-foo"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "npm install failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let shim = home.path().join(".config/aivo/plugins/aivo-foo");
    let shim_text = std::fs::read_to_string(&shim).expect("shim file");
    assert!(
        shim_text.contains("node"),
        "shim must invoke node: {shim_text}"
    );
    assert!(
        home.path()
            .join(".config/aivo/plugins/aivo-foo.d/cli.js")
            .exists(),
        "the npm payload should be extracted alongside the shim"
    );
    assert_eq!(registry(&home)["plugins"]["foo"]["source"], "npm:aivo-foo");
}

#[test]
fn direct_url_returning_html_is_rejected() {
    let (listener, base) = bind();
    serve(
        listener,
        vec![route(
            "/repo",
            "text/html",
            b"<!DOCTYPE html><html><body>a repo page</body></html>".to_vec(),
        )],
    );

    let home = TempDir::new().unwrap();
    let out = aivo(&home)
        .args([
            "plugins",
            "install",
            &format!("{base}/repo"),
            "--name",
            "broken",
        ])
        .output()
        .expect("spawn aivo");

    assert!(!out.status.success(), "installing an HTML page must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("HTML page"),
        "stderr should explain the HTML mistake:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(
        !home
            .path()
            .join(".config/aivo/plugins/aivo-broken")
            .exists(),
        "no broken binary should be written",
    );
}

/// A manifest-capable widget binary tagged with `version`; the differing
/// version string also makes v1 and v2 differ byte-for-byte.
fn widget_release_tgz(work: &TempDir, version: &str) -> Vec<u8> {
    let exe = work.path().join("aivo-widget");
    std::fs::write(
        &exe,
        format!(
            "#!/bin/sh\nif [ \"$1\" = \"--aivo-manifest\" ]; then\n  printf '%s' '{{\"name\":\"widget\",\"version\":\"{version}\",\"protocol\":\"1\",\"roles\":[\"subcommand\"],\"capabilities\":[\"endpoint\"],\"description\":\"demo\"}}'\n  exit 0\nfi\necho \"widget {version}\"\n"
        ),
    )
    .unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    tar_czf(work.path(), "aivo-widget")
}

fn widget_routes(base: &str, tgz: Vec<u8>) -> Vec<Route> {
    let release = format!(
        r#"{{"tag_name":"v1","assets":[{{"name":"aivo-widget.tar.gz","browser_download_url":"{base}/dl/aivo-widget.tar.gz","size":{}}}]}}"#,
        tgz.len()
    );
    vec![
        route(
            "/repos/o/aivo-widget/releases/latest",
            "application/json",
            release.into_bytes(),
        ),
        route("/dl/aivo-widget.tar.gz", "application/octet-stream", tgz),
    ]
}

/// A remote binary that changes underneath an already-approved plugin must
/// have its consent cleared on `update`, so the next dispatch re-prompts the
/// first-run + capability gates instead of inheriting the old approval.
#[test]
fn update_resets_consent_when_remote_binary_changes() {
    let home = TempDir::new().unwrap();

    // Install v1, then run it once so it becomes run-approved with a captured
    // manifest (the state a trojaned update would otherwise inherit).
    let work1 = TempDir::new().unwrap();
    let (l1, base1) = bind();
    serve(
        l1,
        widget_routes(&base1, widget_release_tgz(&work1, "1.0.0")),
    );
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base1)
        .args(["plugins", "install", "github:o/aivo-widget", "--trust"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "install failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = aivo(&home).args(["widget"]).output().expect("spawn aivo");
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("widget 1.0.0"),
        "first run failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );

    let before = registry(&home);
    let rec = &before["plugins"]["widget"];
    assert_eq!(
        rec["run_approved"], true,
        "plugin should be run-approved after first run: {rec}"
    );
    assert_eq!(
        rec["manifest"]["version"], "1.0.0",
        "manifest should be captured: {rec}"
    );
    let checksum_v1 = rec["checksum"].as_str().expect("checksum pin").to_string();

    // A different binary now sits at the same source. `update` must NOT inherit
    // the old approval.
    let work2 = TempDir::new().unwrap();
    let (l2, base2) = bind();
    serve(
        l2,
        widget_routes(&base2, widget_release_tgz(&work2, "2.0.0")),
    );
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base2)
        .args(["plugins", "update", "widget"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "update failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("binary changed"),
        "update should announce the binary change:\n{}",
        String::from_utf8_lossy(&out.stderr),
    );

    let after = registry(&home);
    let rec = &after["plugins"]["widget"];
    // Consent reset: manifest dropped, run-approval cleared, caps cleared, new pin.
    assert!(
        rec["manifest"].is_null(),
        "manifest must be dropped on change: {rec}"
    );
    assert!(
        !rec["run_approved"].as_bool().unwrap_or(false),
        "run approval must be cleared on change: {rec}"
    );
    assert!(
        rec["granted_caps"].as_array().is_none_or(|a| a.is_empty()),
        "granted caps must be cleared on change: {rec}"
    );
    assert_ne!(
        rec["checksum"].as_str(),
        Some(checksum_v1.as_str()),
        "checksum pin must advance to the new binary: {rec}"
    );
    assert!(
        rec["approved_checksum"].is_null(),
        "approval pin must be cleared on change: {rec}"
    );
}

/// Off-TTY, an unapproved remote plugin must be refused with guidance — not
/// silently executed and permanently approved (the old fail-open).
#[test]
fn non_tty_dispatch_refuses_unapproved_plugin() {
    let home = TempDir::new().unwrap();
    let work = TempDir::new().unwrap();
    let (l, base) = bind();
    serve(l, widget_routes(&base, widget_release_tgz(&work, "1.0.0")));

    // Plain install (no --trust): nothing has approved execution yet.
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "install", "github:o/aivo-widget"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "install failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let run = aivo(&home).args(["widget"]).output().expect("spawn aivo");
    assert!(!run.status.success(), "dispatch must refuse to run");
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("widget 1.0.0"),
        "the binary must not have executed"
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("not approved") && stderr.contains("--trust"),
        "refusal should explain how to approve:\n{stderr}"
    );
    // Crucially, the refusal must not have persisted an approval.
    let rec = &registry(&home)["plugins"]["widget"];
    assert!(
        !rec["run_approved"].as_bool().unwrap_or(false),
        "refusal must not approve: {rec}"
    );
}

/// `install --force` over an already-consented plugin must not carry the old
/// binary's consent onto different remote bytes (the same trojan-update hole
/// `plugins update` closes).
#[test]
fn force_reinstall_does_not_inherit_consent_for_changed_binary() {
    let home = TempDir::new().unwrap();

    // v1: trusted install + one run to capture manifest and bind approval.
    let work1 = TempDir::new().unwrap();
    let (l1, base1) = bind();
    serve(
        l1,
        widget_routes(&base1, widget_release_tgz(&work1, "1.0.0")),
    );
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base1)
        .args(["plugins", "install", "github:o/aivo-widget", "--trust"])
        .output()
        .expect("spawn aivo");
    assert!(out.status.success());
    let run = aivo(&home).args(["widget"]).output().unwrap();
    assert!(String::from_utf8_lossy(&run.stdout).contains("widget 1.0.0"));

    // Simulate granted caps (a TTY would have prompted at first dispatch).
    let reg_path = home.path().join(".config/aivo/plugins/.registry.json");
    let mut reg: Value =
        serde_json::from_str(&std::fs::read_to_string(&reg_path).unwrap()).unwrap();
    reg["plugins"]["widget"]["granted_caps"] = serde_json::json!(["endpoint"]);
    std::fs::write(&reg_path, serde_json::to_string(&reg).unwrap()).unwrap();

    // Different bytes now sit at the source; the user force-reinstalls.
    let work2 = TempDir::new().unwrap();
    let (l2, base2) = bind();
    serve(
        l2,
        widget_routes(&base2, widget_release_tgz(&work2, "2.0.0")),
    );
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base2)
        .args(["plugins", "install", "github:o/aivo-widget", "--force"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "force reinstall failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Consent must not have carried onto the new bytes...
    let rec = &registry(&home)["plugins"]["widget"];
    assert!(
        rec["granted_caps"].as_array().is_none_or(|a| a.is_empty()),
        "caps must not survive a byte change: {rec}"
    );
    assert!(
        !rec["run_approved"].as_bool().unwrap_or(false),
        "run approval must not survive a byte change: {rec}"
    );
    // ...so a non-TTY dispatch of the new binary is refused.
    let run = aivo(&home).args(["widget"]).output().unwrap();
    assert!(!run.status.success(), "changed binary must be re-gated");
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("widget 2.0.0"),
        "the changed binary must not have executed"
    );
}

/// A no-op `plugins update` (identical bytes) must keep the dispatch-time run
/// approval — amp-style manifest-less plugins were re-gated on every update.
#[test]
fn noop_update_preserves_dispatch_approval() {
    let home = TempDir::new().unwrap();
    let work = TempDir::new().unwrap();

    // A binary that does NOT answer --aivo-manifest (amp-style): the probe
    // fails, so approval lives only on the record, not via a manifest.
    let exe = work.path().join("aivo-widget");
    std::fs::write(&exe, "#!/bin/sh\necho \"widget ran\"\n").unwrap();
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    let tgz = tar_czf(work.path(), "aivo-widget");

    let (l, base) = bind();
    serve(l, widget_routes(&base, tgz));

    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "install", "github:o/aivo-widget", "--trust"])
        .output()
        .expect("spawn aivo");
    assert!(out.status.success());
    let run = aivo(&home).args(["widget"]).output().unwrap();
    assert!(String::from_utf8_lossy(&run.stdout).contains("widget ran"));
    assert_eq!(registry(&home)["plugins"]["widget"]["run_approved"], true);

    // Same bytes re-fetched: approval must survive.
    let out = aivo(&home)
        .env("AIVO_GITHUB_API", &base)
        .args(["plugins", "update", "widget"])
        .output()
        .expect("spawn aivo");
    assert!(
        out.status.success(),
        "update failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let rec = &registry(&home)["plugins"]["widget"];
    assert_eq!(
        rec["run_approved"], true,
        "a no-op update must not revoke the run approval: {rec}"
    );
    let run = aivo(&home).args(["widget"]).output().unwrap();
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("widget ran"),
        "plugin must still dispatch after a no-op update"
    );
}
