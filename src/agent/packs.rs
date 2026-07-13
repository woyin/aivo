//! Agent extension packs: skills + sub-agent profiles + hooks + MCP servers as one
//! installable unit, in Claude Code's plugin layout (so CC plugins install
//! unchanged). Installed under `~/.config/aivo/packs/<name>`; components join
//! normal discovery at the LOWEST precedence. Install is the consent moment —
//! hooks and stdio MCP servers execute code, so `add` lists them and asks first.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

#[derive(Clone, Debug)]
pub struct Pack {
    pub name: String,
    pub dir: PathBuf,
    pub description: Option<String>,
    pub version: Option<String>,
}

/// What a pack tree ships, for the install consent display and `packs list`.
#[derive(Debug, Default)]
pub struct PackContents {
    pub skills: Vec<String>,
    pub agents: Vec<String>,
    /// Hook commands (code that will run on agent lifecycle events).
    pub hook_commands: Vec<String>,
    /// `(name, command)` stdio MCP servers (local child processes).
    pub mcp_stdio: Vec<(String, String)>,
}

impl PackContents {
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
            && self.agents.is_empty()
            && self.hook_commands.is_empty()
            && self.mcp_stdio.is_empty()
    }

    /// Ships components that execute code (the consent-worthy ones).
    pub fn executes_code(&self) -> bool {
        !self.hook_commands.is_empty() || !self.mcp_stdio.is_empty()
    }
}

pub fn packs_root() -> Option<PathBuf> {
    Some(crate::services::paths::config_dir().join("packs"))
}

/// Parsed-pack cache keyed by (root, root mtime); install/remove touch a
/// direct child of the root, so its mtime bump invalidates the entry.
static PACKS_CACHE: Mutex<Vec<(PathBuf, SystemTime, Vec<Pack>)>> = Mutex::new(Vec::new());

pub fn installed_packs() -> Vec<Pack> {
    packs_root()
        .map(|r| installed_packs_cached(&r))
        .unwrap_or_default()
}

fn installed_packs_cached(root: &Path) -> Vec<Pack> {
    let Ok(mtime) = std::fs::metadata(root).and_then(|m| m.modified()) else {
        return installed_packs_at(root);
    };
    {
        let cache = PACKS_CACHE.lock().unwrap();
        if let Some((_, _, packs)) = cache.iter().find(|(r, t, _)| r == root && *t == mtime) {
            return packs.clone();
        }
    }
    let packs = installed_packs_at(root);
    let mut cache = PACKS_CACHE.lock().unwrap();
    cache.retain(|(r, _, _)| r != root);
    cache.push((root.to_path_buf(), mtime, packs.clone()));
    packs
}

pub fn installed_packs_at(root: &Path) -> Vec<Pack> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.iter().map(|d| load_pack(d)).collect()
}

/// Manifest fields, else the dir name — a manifest-less tree still works.
fn load_pack(dir: &Path) -> Pack {
    let dir_name = dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let manifest: Option<serde_json::Value> =
        std::fs::read_to_string(dir.join(".claude-plugin/plugin.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok());
    let field = |k: &str| {
        manifest
            .as_ref()
            .and_then(|m| m.get(k))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };
    Pack {
        name: field("name").unwrap_or(dir_name),
        dir: dir.to_path_buf(),
        description: field("description"),
        version: field("version"),
    }
}

/// The manifest's `name` field, if present and non-empty. `None` (not the dir
/// name) when absent, so the caller falls back to the source-derived name rather
/// than the temp staging dir.
pub fn manifest_name(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join(".claude-plugin/plugin.json")).ok()?;
    let manifest: serde_json::Value = serde_json::from_str(&text).ok()?;
    manifest
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .filter(|n| !n.is_empty())
}

/// Existing `skills/` dirs across installed packs, for skill discovery.
pub fn skills_roots() -> Vec<PathBuf> {
    component_dirs("skills")
}

/// Existing `agents/` dirs across installed packs, for sub-agent discovery.
pub fn agents_roots() -> Vec<PathBuf> {
    component_dirs("agents")
}

/// Existing `hooks/hooks.json` files across installed packs.
pub fn hooks_files() -> Vec<PathBuf> {
    installed_packs()
        .iter()
        .map(|p| p.dir.join("hooks/hooks.json"))
        .filter(|p| p.is_file())
        .collect()
}

/// Installed pack dirs holding a `.mcp.json` (read like a project config).
pub fn mcp_dirs() -> Vec<PathBuf> {
    installed_packs()
        .into_iter()
        .map(|p| p.dir)
        .filter(|d| d.join(".mcp.json").is_file())
        .collect()
}

fn component_dirs(sub: &str) -> Vec<PathBuf> {
    installed_packs()
        .iter()
        .map(|p| p.dir.join(sub))
        .filter(|p| p.is_dir())
        .collect()
}

/// Scan a pack tree (installed or staged) for what it ships.
pub fn scan_contents(dir: &Path) -> PackContents {
    let skills = subdir_names(&dir.join("skills"), |p| p.join("SKILL.md").is_file());
    let agents = crate::agent::subagents::profile_names(&dir.join("agents"));
    let hook_commands =
        crate::agent::hooks::HookSet::load_from(&dir.join("hooks/hooks.json")).commands();
    let mcp_stdio = crate::agent::mcp::project_stdio_servers(dir);
    PackContents {
        skills,
        agents,
        hook_commands,
        mcp_stdio,
    }
}

fn subdir_names(root: &Path, keep: impl Fn(&Path) -> bool) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir() && keep(p))
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    names
}

/// Copy a pack tree into `root/<name>`; Err when already installed — no silent
/// overwrite of consented content.
pub fn install_tree(root: &Path, name: &str, src: &Path) -> Result<PathBuf, String> {
    if !crate::agent::subagents::is_valid_name(name) {
        return Err(format!("`{name}` isn't a usable pack name ([A-Za-z0-9_-])"));
    }
    let dest = root.join(name);
    if dest.exists() {
        return Err(format!(
            "pack `{name}` is already installed (aivo code packs rm {name} first)"
        ));
    }
    std::fs::create_dir_all(root).map_err(|e| format!("create packs dir: {e}"))?;
    crate::agent::skills::copy_dir_all(src, &dest).map_err(|e| {
        let _ = std::fs::remove_dir_all(&dest);
        format!("copy pack: {e}")
    })?;
    Ok(dest)
}

pub fn remove(root: &Path, name: &str) -> Result<(), String> {
    // Validate before joining — a `../` name would escape the root into an
    // arbitrary `remove_dir_all` (install validates the same way).
    if !crate::agent::subagents::is_valid_name(name) {
        return Err(format!("`{name}` isn't a valid pack name ([A-Za-z0-9_-])"));
    }
    let dir = root.join(name);
    if !dir.is_dir() {
        let known: Vec<String> = installed_packs_at(root)
            .into_iter()
            .map(|p| p.name)
            .collect();
        return Err(if known.is_empty() {
            format!("no pack `{name}` — none are installed")
        } else {
            format!("no pack `{name}`. Installed: {}", known.join(", "))
        });
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("remove pack `{name}`: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "aivo-packs-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: PathBuf, contents: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    fn sample_pack(dir: &Path) {
        write(
            dir.join(".claude-plugin/plugin.json"),
            r#"{"name":"toolkit","description":"a test pack","version":"1.0.0"}"#,
        );
        write(
            dir.join("skills/review/SKILL.md"),
            "---\nname: review\ndescription: reviews\n---\nbody\n",
        );
        write(
            dir.join("agents/tester.md"),
            "---\nname: tester\ndescription: runs tests\n---\nbody\n",
        );
        write(
            dir.join("hooks/hooks.json"),
            r#"{"hooks":{"PostToolUse":[{"matcher":"write_file","hooks":[{"command":"fmt.sh"}]}]}}"#,
        );
        write(
            dir.join(".mcp.json"),
            r#"{"mcpServers":{"db":{"command":"npx","args":["-y","db-server"]}}}"#,
        );
    }

    #[test]
    fn scan_lists_every_component_kind() {
        let dir = tmp();
        sample_pack(&dir);
        let c = scan_contents(&dir);
        assert_eq!(c.skills, vec!["review"]);
        assert_eq!(c.agents, vec!["tester"]);
        assert_eq!(c.hook_commands, vec!["fmt.sh"]);
        assert_eq!(c.mcp_stdio.len(), 1, "{:?}", c.mcp_stdio);
        assert!(c.executes_code());
        assert!(scan_contents(&tmp()).is_empty());
    }

    #[test]
    fn install_list_remove_roundtrip() {
        let root = tmp();
        let staged = tmp();
        sample_pack(&staged);
        let dest = install_tree(&root, "toolkit", &staged).unwrap();
        assert!(dest.join("skills/review/SKILL.md").is_file());
        // Manifest name + metadata surface in the listing.
        let packs = installed_packs_at(&root);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].name, "toolkit");
        assert_eq!(packs[0].version.as_deref(), Some("1.0.0"));
        // No silent overwrite of already-consented content.
        assert!(install_tree(&root, "toolkit", &staged).is_err());
        remove(&root, "toolkit").unwrap();
        assert!(installed_packs_at(&root).is_empty());
        assert!(remove(&root, "toolkit").is_err());
    }

    #[test]
    fn manifest_less_dir_still_loads_by_dir_name() {
        let root = tmp();
        write(
            root.join("bare/skills/x/SKILL.md"),
            "---\nname: x\ndescription: d\n---\nb\n",
        );
        let packs = installed_packs_at(&root);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].name, "bare");
        assert!(packs[0].description.is_none());
    }

    #[test]
    fn remove_rejects_path_traversal_name() {
        let root = tmp().join("packs");
        std::fs::create_dir_all(&root).unwrap();
        let victim = root.parent().unwrap().join("victim");
        std::fs::create_dir_all(&victim).unwrap();
        assert!(remove(&root, "../victim").is_err());
        assert!(
            victim.is_dir(),
            "traversal name must not delete outside the packs root"
        );
    }

    #[test]
    fn installed_packs_cache_serves_unchanged_root_and_refreshes_on_change() {
        let root = tmp();
        write(
            root.join("alpha/.claude-plugin/plugin.json"),
            r#"{"name":"alpha-name"}"#,
        );
        let packs = installed_packs_cached(&root);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].name, "alpha-name");
        // In-place manifest edit: root mtime unchanged → cached parse served.
        write(
            root.join("alpha/.claude-plugin/plugin.json"),
            r#"{"name":"renamed"}"#,
        );
        assert_eq!(installed_packs_cached(&root)[0].name, "alpha-name");
        // Install bumps root mtime → whole set re-parsed.
        write(
            root.join("beta/.claude-plugin/plugin.json"),
            r#"{"name":"beta-name"}"#,
        );
        let packs = installed_packs_cached(&root);
        let names: Vec<_> = packs.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["renamed", "beta-name"]);
        std::fs::remove_dir_all(root.join("beta")).unwrap();
        assert_eq!(installed_packs_cached(&root).len(), 1);
    }

    #[test]
    fn manifest_name_needs_an_actual_name_field() {
        let dir = tmp();
        // Manifest present but no `name` → None.
        write(
            dir.join(".claude-plugin/plugin.json"),
            r#"{"version":"1.0.0"}"#,
        );
        assert_eq!(manifest_name(&dir), None);
        write(
            dir.join(".claude-plugin/plugin.json"),
            r#"{"name":"toolkit"}"#,
        );
        assert_eq!(manifest_name(&dir).as_deref(), Some("toolkit"));
    }
}
