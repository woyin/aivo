//! Scoped permission grants — generalizes the engine's session "always allow" set
//! from exact-match to *scoped* (exact / command-prefix / directory / tool), with an
//! optional schema-versioned on-disk tier.
//!
//! A grant is created when the user answers "always" at a permission prompt; a
//! conservative policy ([`classify`]) picks its scope and whether it persists, and
//! never broadens or persists a *dangerous* action. Grants are allow-only (a "no"
//! creates nothing), so there's no allow/deny precedence — any covering grant permits.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::file_tracker::{is_write_tool, tracked_paths};

/// How broadly a grant applies. Ordered narrowest-ish first; matching is any-covers.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value")]
enum GrantScope {
    /// Exactly this [`exact_key`] — the narrowest scope (one command / one path set).
    Exact(String),
    /// A `run_bash` command sharing this `program subcommand` prefix (e.g. `git push`).
    CommandPrefix(String),
    /// A remote-mutation family (`az repos pr update`); matched only by
    /// [`GrantStore::covers_remote`], never the generic walk.
    RemoteCmd(String),
    /// Any write whose target resolves under this directory.
    Dir(PathBuf),
    /// Any call to this tool by name (e.g. a trusted MCP tool `mcp__server__x`).
    Tool(String),
}

/// Session + optional persistent grants. `path` set ⇒ persistent grants are mirrored
/// to that file; unset ⇒ session-only (nothing is written to disk).
#[derive(Default)]
pub(crate) struct GrantStore {
    session: Vec<GrantScope>,
    persistent: Vec<GrantScope>,
    path: Option<PathBuf>,
}

impl GrantStore {
    /// A store backed by `path` (e.g. `<config>/grants.json`): loads any persistent
    /// grants already there. Missing/unreadable/foreign-schema file ⇒ empty (grants are
    /// additive, so failing open here only means "ask again", never "allow more").
    pub(crate) fn load(path: PathBuf) -> Self {
        let persistent = crate::services::json_store::load_optional::<GrantsFile>(&path)
            .filter(|f| f.schema_version == SCHEMA_VERSION)
            .map(|f| f.grants)
            .unwrap_or_default();
        Self {
            session: Vec::new(),
            persistent,
            path: Some(path),
        }
    }

    /// Whether a prior grant already permits this tool call (checked before prompting).
    pub(crate) fn covers(&self, name: &str, args: &Value, cwd: &Path) -> bool {
        let exact = exact_key(name, args);
        self.all()
            .any(|g| scope_matches(g, name, args, cwd, &exact))
    }

    /// Record an "always allow" for this call under the policy-chosen scope; persists
    /// if the policy says so. Idempotent (a covering grant isn't duplicated).
    pub(crate) fn remember(&mut self, name: &str, args: &Value, cwd: &Path) {
        if self.covers(name, args, cwd) {
            return;
        }
        let (scope, persist) = classify(name, args, cwd);
        if persist {
            self.persistent.push(scope);
            self.save();
        } else {
            self.session.push(scope);
        }
    }

    /// Every remote-mutation family of a command already granted; empty never covers.
    pub(crate) fn covers_remote(&self, prefixes: &[String]) -> bool {
        !prefixes.is_empty()
            && prefixes.iter().all(|p| {
                self.all()
                    .any(|g| matches!(g, GrantScope::RemoteCmd(k) if k == p))
            })
    }

    /// "Always allow" for remote-mutation families. Session-only — an outward
    /// write is never pre-approved in a later session.
    pub(crate) fn remember_remote(&mut self, prefixes: &[String]) {
        for p in prefixes {
            if !self.covers_remote(std::slice::from_ref(p)) {
                self.session.push(GrantScope::RemoteCmd(p.clone()));
            }
        }
    }

    /// Direct exact-key check for a bespoke permission (e.g. sandbox escalation, whose
    /// key isn't a tool call). Session-scoped.
    pub(crate) fn covers_key(&self, key: &str) -> bool {
        self.all()
            .any(|g| matches!(g, GrantScope::Exact(k) if k == key))
    }

    /// Remember a bespoke exact key for the session.
    pub(crate) fn remember_key(&mut self, key: String) {
        if !self.covers_key(&key) {
            self.session.push(GrantScope::Exact(key));
        }
    }

    fn all(&self) -> impl Iterator<Item = &GrantScope> {
        self.session.iter().chain(self.persistent.iter())
    }

    /// Best-effort mirror of the persistent tier to disk (grants aren't secrets, so
    /// plaintext JSON). A write failure is silent — the in-memory grant still holds.
    fn save(&self) {
        let Some(path) = &self.path else { return };
        let file = GrantsFile {
            schema_version: SCHEMA_VERSION,
            grants: self.persistent.clone(),
        };
        let _ = crate::services::json_store::save_blocking(path, &file);
    }
}

/// Bumped on any incompatible change to the on-disk grant shape.
const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct GrantsFile {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    grants: Vec<GrantScope>,
}

/// The narrowest ("exact") key an approval is remembered under — `run_bash` on the
/// command, file writes on the path(s), else the tool name. NUL/US separators avoid
/// name↔value and path↔path collisions.
pub(crate) fn exact_key(name: &str, args: &Value) -> String {
    let arg = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim();
    match name {
        "run_bash" => format!("run_bash\u{0}{}", arg("command")),
        "write_file" | "edit_file" | "multi_edit" => format!("{name}\u{0}{}", arg("path")),
        "apply_patch" => format!(
            "apply_patch\u{0}{}",
            crate::agent::apply_patch::target_paths(arg("input")).join("\u{1}")
        ),
        _ => name.to_string(),
    }
}

/// Choose the grant scope + whether to persist for an approved call. See the module
/// docs for the rationale; the invariant is *never broaden or persist a dangerous act*.
fn classify(name: &str, args: &Value, cwd: &Path) -> (GrantScope, bool) {
    // Untrusted external / MCP tool: trust the specific tool, and persist it — the
    // one grant safe to remember across sessions (it's not destructive by nature).
    if name.starts_with("mcp__") {
        return (GrantScope::Tool(name.to_string()), true);
    }
    if name == "run_bash" {
        let cmd = args
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if let Some(prefix) = command_prefix(cmd) {
            return (GrantScope::CommandPrefix(prefix), false);
        }
        return (GrantScope::Exact(exact_key(name, args)), false);
    }
    if is_write_tool(name) {
        // A write only prompts when it leaves cwd or would clobber an unread file.
        // For an out-of-cwd target, grant its directory (a session write-root); keep
        // an in-cwd clobber approval exact so it can't silently widen to the whole dir.
        if let Some(dir) = out_of_cwd_dir(name, args, cwd) {
            return (GrantScope::Dir(dir), false);
        }
    }
    (GrantScope::Exact(exact_key(name, args)), false)
}

fn scope_matches(scope: &GrantScope, name: &str, args: &Value, cwd: &Path, exact: &str) -> bool {
    match scope {
        GrantScope::Exact(k) => k == exact,
        GrantScope::Tool(t) => t == name,
        // covers_remote only — all-families semantics, not any-match.
        GrantScope::RemoteCmd(_) => false,
        GrantScope::CommandPrefix(p) => {
            name == "run_bash"
                && command_prefix(
                    args.get("command")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim(),
                )
                .as_deref()
                    == Some(p.as_str())
        }
        GrantScope::Dir(d) => {
            is_write_tool(name)
                && resolved_targets(name, args, cwd)
                    .iter()
                    .any(|p| p.starts_with(d))
        }
    }
}

/// Subcommand-style programs whose first two tokens (`git push`) form a safe grant
/// family. Bare file-op commands (`rm`, `mv`, `dd`) are deliberately excluded so
/// approving one never widens to the whole program.
const SUBCOMMAND_TOOLS: &[&str] = &[
    "git",
    "cargo",
    "npm",
    "pnpm",
    "yarn",
    "go",
    "docker",
    "kubectl",
    "make",
    "pip",
    "pip3",
    "cmake",
    "bundle",
    "dotnet",
    "terraform",
    "gh",
    "poetry",
    "az",
    "aws",
    "gcloud",
];

/// `program subcommand` for a subcommand-style command, else `None` (→ stay exact).
/// Returns `None` when the second token is a flag/absent, so there's no subcommand to
/// broaden to.
fn command_prefix(command: &str) -> Option<String> {
    let mut it = command.split_whitespace();
    let prog = it.next()?;
    if !SUBCOMMAND_TOOLS.contains(&prog) {
        return None;
    }
    let sub = it.next()?;
    if sub.starts_with('-') {
        return None;
    }
    Some(format!("{prog} {sub}"))
}

/// A write tool's target paths, resolved to absolute (via the same `~`/cwd resolution
/// the tools use) so directory containment is comparable.
fn resolved_targets(name: &str, args: &Value, cwd: &Path) -> Vec<PathBuf> {
    tracked_paths(name, args)
        .iter()
        .map(|p| crate::agent::tools::resolve(cwd, p))
        .collect()
}

/// The parent directory of the first target that resolves outside `cwd`, if any.
fn out_of_cwd_dir(name: &str, args: &Value, cwd: &Path) -> Option<PathBuf> {
    let cwd_abs = std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf());
    for t in resolved_targets(name, args, cwd) {
        if !t.starts_with(&cwd_abs) {
            return t.parent().map(Path::to_path_buf);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn bash(cmd: &str) -> Value {
        json!({ "command": cmd })
    }

    #[test]
    fn exact_key_scopes_to_command_and_path() {
        // Different commands / paths → different keys; whitespace-trimmed.
        assert_ne!(
            exact_key("run_bash", &bash("rm -rf build")),
            exact_key("run_bash", &bash("rm -rf /")),
        );
        assert_eq!(
            exact_key("run_bash", &bash("  cargo test ")),
            exact_key("run_bash", &bash("cargo test")),
        );
        assert_ne!(
            exact_key("edit_file", &json!({ "path": "a.rs" })),
            exact_key("edit_file", &json!({ "path": "b.rs" })),
        );
    }

    #[test]
    fn command_prefix_only_broadens_subcommand_tools() {
        assert_eq!(command_prefix("git push origin x"), Some("git push".into()));
        assert_eq!(
            command_prefix("cargo build --release"),
            Some("cargo build".into())
        );
        // Bare file-op commands never broaden.
        assert_eq!(command_prefix("rm -rf build"), None);
        assert_eq!(command_prefix("dd if=/dev/zero of=x"), None);
        // A flag where the subcommand should be → no broadening.
        assert_eq!(command_prefix("git --version"), None);
        // Unknown program → exact only.
        assert_eq!(command_prefix("./deploy.sh prod"), None);
    }

    #[test]
    fn granting_a_git_subcommand_covers_the_family_but_not_siblings() {
        let cwd = std::env::temp_dir();
        let mut g = GrantStore::default();
        assert!(!g.covers("run_bash", &bash("git push origin main"), &cwd));
        g.remember("run_bash", &bash("git push origin main"), &cwd);
        // Same family, different flags/args → covered without a new prompt.
        assert!(g.covers("run_bash", &bash("git push --force-with-lease"), &cwd));
        // A different subcommand is NOT covered.
        assert!(!g.covers("run_bash", &bash("git reset --hard"), &cwd));
    }

    #[test]
    fn bare_destructive_command_grant_stays_exact() {
        let cwd = std::env::temp_dir();
        let mut g = GrantStore::default();
        g.remember("run_bash", &bash("rm -rf build"), &cwd);
        // Approving one rm must NOT allow a different rm.
        assert!(g.covers("run_bash", &bash("rm -rf build"), &cwd));
        assert!(!g.covers("run_bash", &bash("rm -rf other"), &cwd));
    }

    #[test]
    fn out_of_cwd_write_grants_the_target_directory_for_the_session() {
        let cwd = tmp();
        let outside = tmp(); // a sibling dir, not under cwd
        let a = json!({ "path": outside.join("a.txt").to_string_lossy(), "content": "x" });
        let b = json!({ "path": outside.join("b.txt").to_string_lossy(), "content": "y" });
        let mut g = GrantStore::default();
        assert!(!g.covers("write_file", &a, &cwd));
        g.remember("write_file", &a, &cwd);
        // A sibling file in the same external dir is now covered.
        assert!(g.covers("write_file", &b, &cwd));
    }

    #[test]
    fn mcp_tool_grant_persists_across_a_reload() {
        let dir = tmp();
        let path = dir.join("grants.json");
        let cwd = tmp();
        let call = json!({});
        {
            let mut g = GrantStore::load(path.clone());
            g.remember("mcp__server__do_thing", &call, &cwd);
            assert!(g.covers("mcp__server__do_thing", &call, &cwd));
        }
        // A fresh store from the same file still covers it (persisted to disk).
        let g2 = GrantStore::load(path);
        assert!(g2.covers("mcp__server__do_thing", &call, &cwd));
        assert!(!g2.covers("mcp__server__other", &call, &cwd));
    }

    #[test]
    fn session_grants_are_not_written_to_disk() {
        let dir = tmp();
        let path = dir.join("grants.json");
        let cwd = tmp();
        let mut g = GrantStore::load(path.clone());
        g.remember("run_bash", &bash("git push origin main"), &cwd);
        // A command-prefix grant is session-only → the file is never created.
        assert!(!path.exists());
    }

    #[test]
    fn remote_grant_covers_the_family_and_only_the_family() {
        let mut g = GrantStore::default();
        let update = vec!["az repos pr update".to_string()];
        assert!(!g.covers_remote(&update));
        g.remember_remote(&update);
        assert!(g.covers_remote(&update));
        // A sibling verb is a different grant.
        assert!(!g.covers_remote(&["az repos pr delete".to_string()]));
        // Multi-family commands need every family granted, and empty never covers.
        let both = vec!["az repos pr update".to_string(), "gh pr merge".to_string()];
        assert!(!g.covers_remote(&both));
        g.remember_remote(&both);
        assert!(g.covers_remote(&both));
        assert!(!g.covers_remote(&[]));
    }

    #[test]
    fn remote_grant_never_leaks_into_generic_coverage_or_disk() {
        let dir = tmp();
        let path = dir.join("grants.json");
        let cwd = tmp();
        let mut g = GrantStore::load(path.clone());
        g.remember_remote(&["az repos pr update".to_string()]);
        // A remote grant must never approve a dangerous command via the generic walk.
        assert!(!g.covers(
            "run_bash",
            &bash("az repos pr update --id 7 && rm -rf /"),
            &cwd
        ));
        // Session-only: nothing reaches disk.
        assert!(!path.exists());
    }

    #[test]
    fn cloud_clis_broaden_to_subcommand_prefix() {
        assert_eq!(
            command_prefix("az repos pr update --id 7"),
            Some("az repos".into())
        );
        assert_eq!(command_prefix("aws s3 cp a s3://b"), Some("aws s3".into()));
        assert_eq!(
            command_prefix("gcloud compute instances list"),
            Some("gcloud compute".into())
        );
    }

    #[test]
    fn bespoke_exact_key_roundtrips_for_escalation() {
        let mut g = GrantStore::default();
        let key = "run_bash_unsandboxed\u{0}make install".to_string();
        assert!(!g.covers_key(&key));
        g.remember_key(key.clone());
        assert!(g.covers_key(&key));
        assert!(!g.covers_key("run_bash_unsandboxed\u{0}make uninstall"));
    }

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let id = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("aivo-grant-{}-{}", std::process::id(), id));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::canonicalize(&dir).unwrap()
    }
}
