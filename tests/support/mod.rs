//! Process-wide test sandbox, compiled into every test binary (`mod support;`)
//! and into the lib's unit-test build (`#[path]` in lib.rs). Points HOME at a
//! quarantine dir before main() so a test that forgets per-case isolation can
//! never read or write the real user environment.
//!
//! The quarantine lives under the *real* home, NOT `temp_dir()`: the agent
//! write-sandbox allowlists /tmp and $TMPDIR, and its enforcement tests need a
//! HOME that writes are still blocked in (see e.g. `sandbox_confines_writes_to_
//! workspace`). `tests/sandbox_linux.rs` omits this module for the same reason
//! it exists — its assertions are about the real HOME's placement.

/// Pre-main and therefore single-threaded: the env mutation is race-free.
#[ctor::ctor(unsafe)]
fn sandbox_process_env() {
    let real_home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let root = real_home.join(".aivo-test-home");
    sweep_stale_quarantines(&root);
    let home = root.join(std::process::id().to_string());
    // PID reuse could resurface state from an earlier run; start clean.
    let _ = std::fs::remove_dir_all(&home);
    std::fs::create_dir_all(&home).expect("create sandbox HOME");
    // SAFETY: no other threads exist before main().
    unsafe {
        std::env::set_var("HOME", &home);
        std::env::set_var("USERPROFILE", &home);
        // Would otherwise leak into spawned aivo children that set their own
        // per-test HOME, collapsing their isolation onto one shared config dir.
        std::env::remove_var("AIVO_CONFIG_DIR");
    }
}

/// Quarantines from runs older than a day are dead weight (no test binary
/// lives that long); mtime keeps this liveness-free and safe under
/// concurrent binaries, which are by definition recent.
fn sweep_stale_quarantines(root: &std::path::Path) {
    const STALE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STALE);
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}

/// The sandbox only protects binaries that compile this module in — a new
/// tests/*.rs that forgets `mod support;` silently runs unprotected. Enforced
/// here so the check needs no CI wiring; runs (cheaply) in every binary.
#[test]
fn every_integration_test_binary_includes_the_sandbox() {
    // Opted out by design: its Landlock assertions are about the real HOME's
    // placement outside the sandbox allowlist.
    const EXEMPT: &[&str] = &["sandbox_linux"];
    let tests_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for entry in std::fs::read_dir(&tests_dir)
        .expect("read tests/")
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        if EXEMPT.contains(&name.as_ref()) {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read test source");
        assert!(
            src.contains("mod support;"),
            "tests/{name}.rs must declare `mod support;` (HOME sandbox) or join the exemption list in tests/support/mod.rs"
        );
    }
}
