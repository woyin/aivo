//! File-staleness guard. Records a content baseline the moment the agent reads a
//! file, and refuses a later `edit_file`/`write_file`/`multi_edit`/`apply_patch`
//! whose target changed on disk since — so an unattended run can't silently clobber
//! an edit made (by the user, a formatter, a concurrent process) between its read and
//! its write. Conservative and fail-closed; see `FileTracker::stale_block`.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// What we remember about a file the moment the agent last saw it. Change is decided
/// on content (`len` + hash), not mtime — a bare `touch` with identical bytes is not a
/// clobber risk and shouldn't force a re-read.
#[derive(Clone, PartialEq)]
struct Baseline {
    /// Full file size — also catches a change past the hashed prefix (see `snapshot`).
    len: u64,
    /// Non-cryptographic content hash (fixed-seed SipHash) — only needs to detect change.
    hash: u64,
}

/// Per-session map of files the agent has read/written → their last-seen baseline.
#[derive(Default)]
pub(crate) struct FileTracker {
    seen: HashMap<PathBuf, Baseline>,
}

impl FileTracker {
    /// Record (or refresh) the baseline for the file `name`+`args` touched, if it's a
    /// single-file read/write tool. Call after a *successful* read (anchors the baseline
    /// a future edit is checked against) or write (so our own change isn't later seen as
    /// an external clobber). A path that can't be snapshotted is left untracked.
    pub(crate) fn record(&mut self, name: &str, args: &Value, cwd: &Path) {
        for p in tracked_paths(name, args) {
            let key = key(cwd, &p);
            if let Some(b) = snapshot(&key) {
                self.seen.insert(key, b);
            }
        }
    }

    /// If a mutating tool targets a file we hold a baseline for and it changed on disk
    /// since, return a fail-closed message telling the model to re-read first. `None`
    /// when there's no baseline, the file is unchanged, or it can't be read now (don't
    /// block on a stat error). Only meaningful for the mutating file tools; other tools
    /// carry no tracked path and return `None`.
    pub(crate) fn stale_block(&self, name: &str, args: &Value, cwd: &Path) -> Option<String> {
        for p in tracked_paths(name, args) {
            let key = key(cwd, &p);
            let Some(base) = self.seen.get(&key) else {
                continue;
            };
            let Some(now) = snapshot(&key) else { continue };
            if now != *base {
                return Some(format!(
                    "`{p}` changed on disk since you last read it — re-read it before \
                     editing so you don't overwrite that change."
                ));
            }
        }
        None
    }
}

/// The path(s) a read/write tool touches — mirrors the engine's `record_touched_file`
/// extraction so tracking and the touched-files list stay in step. Shared with the
/// grant store so "which files does this tool touch" has one definition.
pub(crate) fn tracked_paths(name: &str, args: &Value) -> Vec<String> {
    if !crate::agent::tools::registry::traits(name).is_some_and(|t| t.path_tracked) {
        return Vec::new();
    }
    // `apply_patch` carries its targets in the V4A body; the rest one `path` arg.
    if name == "apply_patch" {
        return args
            .get("input")
            .and_then(Value::as_str)
            .map(crate::agent::apply_patch::target_paths)
            .unwrap_or_default();
    }
    args.get("path")
        .and_then(Value::as_str)
        .map(|p| vec![p.to_string()])
        .unwrap_or_default()
}

/// The mutating file tools — those whose `tracked_paths` should be re-checked after a
/// write. Excludes `read_file`, which carries a path but changes nothing.
pub(crate) fn is_write_tool(name: &str) -> bool {
    crate::agent::tools::registry::traits(name).is_some_and(|t| t.file_write)
}

/// Stable key for a workspace path: `~`/cwd-resolved (matching the tools' own
/// resolution), canonicalized when the file exists so `./f` and `f` collapse to one
/// entry. Falls back to the resolved path when canonicalization fails (e.g. not yet
/// created) — record and check use the same function, so keys still agree.
fn key(cwd: &Path, p: &str) -> PathBuf {
    let resolved = crate::agent::tools::resolve(cwd, p);
    std::fs::canonicalize(&resolved).unwrap_or(resolved)
}

/// Stat + content-hash a file; `None` if it can't be read as a regular file
/// (missing / directory / permission).
fn snapshot(path: &Path) -> Option<Baseline> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() {
        return None;
    }
    // Bound the hash read the way `read_file` bounds its own — editing a multi-GB file
    // shouldn't OOM on every call. `len` (full size) still catches changes past the cap.
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .ok()?
        .take(crate::agent::tools::MAX_READ_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    Some(Baseline {
        len: meta.len(),
        hash: h.finish(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-ftrack-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn edit(path: &str) -> Value {
        json!({ "path": path, "old_string": "a", "new_string": "b" })
    }

    #[test]
    fn unread_file_is_never_blocked() {
        let d = tmp();
        std::fs::write(d.join("f.txt"), "hello").unwrap();
        let t = FileTracker::default();
        // No baseline was recorded → editing is allowed (staleness is about *changed
        // since last read*, not read-before-write).
        assert!(t.stale_block("edit_file", &edit("f.txt"), &d).is_none());
    }

    #[test]
    fn unchanged_file_passes_changed_file_is_blocked() {
        let d = tmp();
        std::fs::write(d.join("f.txt"), "hello").unwrap();
        let mut t = FileTracker::default();
        t.record("read_file", &json!({ "path": "f.txt" }), &d);

        // Unchanged since the read → no block.
        assert!(t.stale_block("edit_file", &edit("f.txt"), &d).is_none());

        // External change → blocked with a re-read message.
        std::fs::write(d.join("f.txt"), "hello world").unwrap();
        let msg = t.stale_block("edit_file", &edit("f.txt"), &d).unwrap();
        assert!(msg.contains("changed on disk"));
        assert!(msg.contains("f.txt"));
    }

    #[test]
    fn refreshing_the_baseline_clears_the_block() {
        let d = tmp();
        std::fs::write(d.join("f.txt"), "one").unwrap();
        let mut t = FileTracker::default();
        t.record("read_file", &json!({ "path": "f.txt" }), &d);
        std::fs::write(d.join("f.txt"), "two").unwrap();
        assert!(t.stale_block("edit_file", &edit("f.txt"), &d).is_some());
        // Recording after our own write (or a re-read) re-anchors → no longer stale.
        t.record("edit_file", &edit("f.txt"), &d);
        assert!(t.stale_block("edit_file", &edit("f.txt"), &d).is_none());
    }

    #[test]
    fn same_content_written_back_is_not_a_change() {
        let d = tmp();
        std::fs::write(d.join("f.txt"), "same").unwrap();
        let mut t = FileTracker::default();
        t.record("read_file", &json!({ "path": "f.txt" }), &d);
        // Rewrite identical bytes (a touch): mtime moves but content doesn't → no block.
        std::fs::write(d.join("f.txt"), "same").unwrap();
        assert!(t.stale_block("edit_file", &edit("f.txt"), &d).is_none());
    }

    #[test]
    fn deleted_target_does_not_block() {
        let d = tmp();
        std::fs::write(d.join("f.txt"), "hi").unwrap();
        let mut t = FileTracker::default();
        t.record("read_file", &json!({ "path": "f.txt" }), &d);
        std::fs::remove_file(d.join("f.txt")).unwrap();
        // Can't snapshot now → don't block (a write_file recreating it is fine).
        assert!(
            t.stale_block("write_file", &json!({ "path": "f.txt" }), &d)
                .is_none()
        );
    }

    #[test]
    fn non_file_tools_carry_no_tracked_path() {
        assert!(tracked_paths("run_bash", &json!({ "command": "ls" })).is_empty());
        assert!(tracked_paths("grep", &json!({ "pattern": "x" })).is_empty());
        assert_eq!(tracked_paths("edit_file", &edit("a.rs")), vec!["a.rs"]);
    }
}
