//! Skill discovery for the agent, using the portable `SKILL.md` format. A skill
//! is a folder holding a `SKILL.md` with YAML frontmatter (`name`,
//! `description`) and a body of instructions. Discovery covers aivo's own dirs
//! (`<project>/.aivo/skills`, `~/.config/aivo/skills`), the tool-neutral Agent
//! Skills location (`<project>/.agents/skills`, `~/.agents/skills`) that the
//! ecosystem shares (agentskills.io; Gemini CLI and Vercel's `skills` CLI
//! populate it), AND Claude Code's `.claude/skills` (project + user) — so an
//! existing library of `~/.claude/skills/*/SKILL.md` works in `aivo chat`
//! unchanged. Only the
//! names + (first-sentence) descriptions go in the system prompt; the `skill`
//! tool loads a body on demand (progressive disclosure), and the model reads
//! bundled files in the dir via its file tools.

use crate::agent::protocol::ToolSpec;
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub dir: PathBuf,
}

/// Discover skills, project dir before user dir (so a repo can shadow a personal
/// skill of the same name), and within each tier the tool-neutral `.agents/skills`
/// before aivo's own dir before Claude Code's `.claude/skills` — matching the
/// precedence Gemini CLI gives `.agents`. Each `<root>/<name>/SKILL.md` is one
/// skill; the first occurrence of a name wins.
pub fn discover_skills(cwd: &Path) -> Vec<Skill> {
    let mut roots = vec![
        cwd.join(".agents").join("skills"),
        cwd.join(".aivo").join("skills"),
        cwd.join(".claude").join("skills"),
    ];
    if let Some(home) = crate::services::system_env::home_dir() {
        roots.push(home.join(".agents").join("skills"));
        roots.push(home.join(".config").join("aivo").join("skills"));
        roots.push(home.join(".claude").join("skills"));
    }
    discover_from_roots(&roots)
}

/// The built-in **create-skill** instructions, compiled into the binary and
/// driven by the first-class `/create-skill` slash command (NOT a discovered
/// skill — it has no folder and never appears in `/skills`). Parsed from the
/// embedded `SKILL.md` so the name/body live in one editable place; `dir` is
/// empty because there is nothing on disk.
pub fn create_skill_builtin() -> Skill {
    const SRC: &str = include_str!("builtin_skills/create-skill.md");
    let (front, body) = split_frontmatter(SRC);
    let name = front
        .as_ref()
        .and_then(|f| field(f, "name"))
        .unwrap_or_else(|| "create-skill".to_string());
    let description = front
        .as_ref()
        .and_then(|f| field(f, "description"))
        .unwrap_or_else(|| first_non_empty_line(body));
    Skill {
        name,
        description,
        body: body.trim().to_string(),
        dir: PathBuf::new(),
    }
}

/// Which tier a discovered skill lives in. The `/skills` overlay only scaffolds
/// and deletes `User` skills (under `~/.config/aivo/skills`, the dir aivo owns); a
/// `Project` skill is shown and toggleable but never deleted from a chat — that
/// covers a repo's `.agents/.aivo/.claude` skills AND the user's Claude Code
/// library (`~/.claude/skills`), since deleting either from aivo would surprise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillScope {
    User,
    Project,
}

/// The user-global skills dir the `/skills` overlay scaffolds into.
pub fn user_skills_dir() -> Option<PathBuf> {
    crate::services::system_env::home_dir().map(|h| h.join(".config/aivo/skills"))
}

/// Classify a discovered skill's dir: `User` (deletable from chat) only when it
/// lives under a root aivo manages itself; everything else is `Project`
/// (discovered + usable, but protected from deletion). A repo's `.agents/.aivo/
/// .claude` skills and the user's Claude Code library (`~/.claude/skills`) are all
/// `Project` — aivo never deletes skills it didn't create.
pub fn skill_scope(dir: &Path, cwd: &Path) -> SkillScope {
    let mut protected = vec![
        cwd.join(".agents").join("skills"),
        cwd.join(".aivo").join("skills"),
        cwd.join(".claude").join("skills"),
    ];
    if let Some(home) = crate::services::system_env::home_dir() {
        // Claude Code's library is discovered and usable, but belongs to Claude
        // Code, not aivo — never deletable via the `/skills` overlay.
        protected.push(home.join(".claude").join("skills"));
    }
    if protected.iter().any(|root| dir.starts_with(root)) {
        SkillScope::Project
    } else {
        SkillScope::User
    }
}

/// A usable skill name (and folder name): non-empty, `[A-Za-z0-9_-]` only.
pub fn is_valid_skill_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Scaffold a new user skill at `~/.config/aivo/skills/<name>/SKILL.md` from a
/// template (frontmatter + a placeholder body) for the user to fill in. Returns
/// the created `SKILL.md` path. `Err` on an invalid name, a missing home dir, or
/// a name that already exists (we never overwrite an existing skill).
pub fn scaffold_skill(name: &str, description: &str) -> Result<PathBuf, String> {
    let root = user_skills_dir().ok_or("no home directory")?;
    scaffold_skill_at(&root, name, description)
}

/// Inner with the root dir injected, so a test can scaffold into a tempdir
/// instead of the real `~/.config/aivo/skills`.
fn scaffold_skill_at(root: &Path, name: &str, description: &str) -> Result<PathBuf, String> {
    if !is_valid_skill_name(name) {
        return Err("Skill name must be letters, digits, '-' or '_'".to_string());
    }
    let dir = root.join(name);
    if dir.exists() {
        return Err(format!("a skill named `{name}` already exists"));
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let desc = {
        let trimmed = description.trim();
        if trimmed.is_empty() {
            "One-line summary of what this skill does and when to use it."
        } else {
            trimmed
        }
    };
    let template = format!(
        "---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n\nWrite the step-by-step instructions the agent should follow when this skill is invoked. Bundle any scripts or resources alongside this file.\n"
    );
    let path = dir.join("SKILL.md");
    std::fs::write(&path, template).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Delete a discovered skill's folder. Guarded: the dir must actually hold a
/// `SKILL.md`, so a mis-resolved path can't recursively delete an unrelated
/// folder. Powers the `/skills` overlay's `d` / `/skills rm`.
pub fn remove_skill_dir(dir: &Path) -> Result<(), String> {
    if !dir.join("SKILL.md").is_file() {
        return Err(format!("{} is not a skill folder", dir.display()));
    }
    std::fs::remove_dir_all(dir).map_err(|e| format!("remove {}: {e}", dir.display()))
}

// ── install from an online / local source ───────────────────────────────────
//
// Follows the Agent Skills `skills/*/SKILL.md` convention shared by `npx skills`
// and `gh skill install`: scan a fetched source tree for skill folders (root
// SKILL.md + `skills/<name>` flat + `skills/<cat>/<name>` catalog under the
// standard containers), then COPY the chosen folder(s) into the user skills dir.

/// What `install_from_source` resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallOutcome {
    /// Skills copied in, by display name.
    Installed(Vec<String>),
    /// The source held several skills and none was picked — names to choose from
    /// (re-run with `<source> <name>` or `<source> *`).
    Ambiguous(Vec<String>),
}

/// Scan a source tree for installable skills, following the `skills/*/SKILL.md`
/// convention: a root `SKILL.md`, plus skill folders one level deep
/// (`skills/<name>/SKILL.md`) and one catalog level (`skills/<cat>/<name>/`)
/// under `skills/`, `.agents/skills/`, `.claude/skills/`, `.github/skills/`.
/// Deduped by skill name (first wins), sorted for determinism.
pub fn discover_installable(root: &Path) -> Vec<Skill> {
    let mut found: Vec<Skill> = Vec::new();
    try_push_skill(root, &mut found); // the source root itself may be a skill
    for container in [
        "skills",
        ".agents/skills",
        ".claude/skills",
        ".github/skills",
    ] {
        let Ok(entries) = std::fs::read_dir(root.join(container)) else {
            continue;
        };
        let mut subdirs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subdirs.sort();
        for sub in subdirs {
            if load_skill(&sub).is_some() {
                try_push_skill(&sub, &mut found); // flat: <container>/<name>/SKILL.md
            } else if let Ok(inner) = std::fs::read_dir(&sub) {
                // catalog: <container>/<category>/<name>/SKILL.md
                let mut innerdirs: Vec<PathBuf> = inner
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .collect();
                innerdirs.sort();
                for id in innerdirs {
                    try_push_skill(&id, &mut found);
                }
            }
        }
    }
    found
}

fn try_push_skill(dir: &Path, found: &mut Vec<Skill>) {
    if let Some(skill) = load_skill(dir)
        && !found.iter().any(|e| e.name == skill.name)
    {
        found.push(skill);
    }
}

/// Fetch + discover + copy skills from `source` (a `github:owner/repo[@ref]`,
/// `gh:`, bare `github.com/owner/repo`, or local path) into `~/.config/aivo/skills`.
/// `only` filters: `Some("*")` = all, `Some(name)` = just that one, `None` =
/// install the sole skill or report `Ambiguous` when there are several.
pub async fn install_from_source(
    source: &str,
    only: Option<&str>,
) -> Result<InstallOutcome, String> {
    let dest_root = user_skills_dir().ok_or("no home directory")?;
    install_from_source_into(&dest_root, source, only).await
}

/// Inner with the destination root injected, so a test installs into a tempdir
/// instead of the real `~/.config/aivo/skills`.
async fn install_from_source_into(
    dest_root: &Path,
    source: &str,
    only: Option<&str>,
) -> Result<InstallOutcome, String> {
    let tree = fetch_source_tree(source).await?;
    let result = (|| {
        let mut skills = discover_installable(&tree.root);
        if skills.is_empty() {
            return Err(format!(
                "no SKILL.md found in `{source}` (looked at the root and skills/ folders)"
            ));
        }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        match only {
            Some("*") => {}
            Some(name) => {
                skills.retain(|s| s.name == name);
                if skills.is_empty() {
                    return Err(format!("no skill named `{name}` in `{source}`"));
                }
            }
            None if skills.len() > 1 => {
                return Ok(InstallOutcome::Ambiguous(
                    skills.into_iter().map(|s| s.name).collect(),
                ));
            }
            None => {}
        }
        // Symlink-escape guard: a candidate skill dir must canonicalize (through
        // every symlink in its chain) to a path inside the fetched tree, else an
        // untrusted repo's `skills/x -> /etc` would copy the link target's real
        // files into the user's skills dir. Like `archive::find_executable`.
        let root_canon = std::fs::canonicalize(&tree.root).unwrap_or_else(|_| tree.root.clone());
        let mut installed = Vec::new();
        for skill in &skills {
            let folder = skill_folder_name(&skill.name);
            if folder.is_empty() {
                return Err(format!("skill `{}` has no usable folder name", skill.name));
            }
            let escapes = std::fs::canonicalize(&skill.dir)
                .map(|c| !c.starts_with(&root_canon))
                .unwrap_or(true);
            if escapes {
                return Err(format!(
                    "skill `{}` resolves outside the source tree (symlink escape) — refusing to install",
                    skill.name
                ));
            }
            let dest = dest_root.join(&folder);
            if dest.exists() {
                return Err(format!("a skill named `{folder}` already exists"));
            }
            copy_dir_all(&skill.dir, &dest)
                .map_err(|e| format!("copying `{}`: {e}", skill.name))?;
            installed.push(skill.name.clone());
        }
        Ok(InstallOutcome::Installed(installed))
    })();
    if let Some(tmp) = &tree.cleanup {
        let _ = std::fs::remove_dir_all(tmp);
    }
    result
}

/// A resolved source tree on disk: `root` is where to scan; `cleanup` is a temp
/// dir to delete afterward (None for a local path the user owns).
struct SourceTree {
    root: PathBuf,
    cleanup: Option<PathBuf>,
}

async fn fetch_source_tree(source: &str) -> Result<SourceTree, String> {
    use crate::plugin::source::{SourceKind, classify};
    match classify(source).map_err(|e| e.to_string())? {
        SourceKind::LocalPath => {
            let path = PathBuf::from(shellexpand_tilde(source));
            if !path.is_dir() {
                return Err(format!("`{source}` is not a directory"));
            }
            Ok(SourceTree {
                root: path,
                cleanup: None,
            })
        }
        SourceKind::GitHub { owner, repo, tag } => {
            let tmp = std::env::temp_dir().join(format!(
                "aivo-skill-install-{}-{}",
                std::process::id(),
                next_install_seq()
            ));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).map_err(|e| format!("create temp dir: {e}"))?;
            let tgz = tmp.join("source.tar.gz");
            // Stream the tarball straight to disk with a hard byte cap so a giant
            // (or malicious) repo can't spike memory by being read whole into RAM.
            download_github_tarball(&owner, &repo, tag.as_deref(), &tgz)
                .await
                .inspect_err(|_| {
                    let _ = std::fs::remove_dir_all(&tmp);
                })?;
            crate::services::archive::extract_archive(
                &tgz,
                &tmp,
                crate::services::archive::ArchiveKind::TarGz,
            )
            .map_err(|e| {
                let _ = std::fs::remove_dir_all(&tmp);
                format!("extract tarball: {e}")
            })?;
            let _ = std::fs::remove_file(&tgz);
            // Guard against a compression bomb: tiny on the wire, huge on disk.
            if let Err(e) = enforce_extracted_caps(&tmp) {
                let _ = std::fs::remove_dir_all(&tmp);
                return Err(e);
            }
            // GitHub wraps everything in one `owner-repo-<sha>/` folder.
            let _ = crate::services::archive::flatten_single_subdir(&tmp);
            Ok(SourceTree {
                root: tmp.clone(),
                cleanup: Some(tmp),
            })
        }
        SourceKind::DirectUrl | SourceKind::Npm { .. } | SourceKind::Cargo { .. } => Err(format!(
            "`{source}`: skills install from a github:owner/repo, a github.com URL, or a local path"
        )),
    }
}

/// Hard cap on a downloaded skill tarball (compressed). Skills are a few KB of
/// Markdown; 50 MiB is generous and bounds both memory and disk against a giant
/// or malicious repo. The download aborts the moment the body crosses it.
const MAX_SKILL_TARBALL_BYTES: u64 = 50 * 1024 * 1024;
/// Cap on the EXTRACTED tree, so a compression bomb (tiny on the wire, enormous
/// on disk) is rejected before we scan/copy it.
const MAX_SKILL_EXTRACTED_BYTES: u64 = 200 * 1024 * 1024;
const MAX_SKILL_EXTRACTED_ENTRIES: usize = 50_000;

/// Stream a GitHub repo tarball to `dest`, enforcing `MAX_SKILL_TARBALL_BYTES`
/// both up front (via `Content-Length`, when the server sends it) and while
/// reading (chunk by chunk, so a missing/lying length can't get past us). On any
/// error the partial file is removed.
async fn download_github_tarball(
    owner: &str,
    repo: &str,
    gref: Option<&str>,
    dest: &Path,
) -> Result<(), String> {
    use std::io::Write;
    let base =
        std::env::var("AIVO_GITHUB_API").unwrap_or_else(|_| "https://api.github.com".to_string());
    let url = match gref {
        Some(r) => format!("{base}/repos/{owner}/{repo}/tarball/{r}"),
        None => format!("{base}/repos/{owner}/{repo}/tarball"),
    };
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .user_agent(concat!("aivo/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("requesting {url}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("downloading {owner}/{repo}: {e}"))?;
    if let Some(len) = resp.content_length()
        && len > MAX_SKILL_TARBALL_BYTES
    {
        return Err(too_big_msg(owner, repo, len));
    }
    let mut file =
        std::fs::File::create(dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let mut total: u64 = 0;
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                total += chunk.len() as u64;
                if total > MAX_SKILL_TARBALL_BYTES {
                    drop(file);
                    let _ = std::fs::remove_file(dest);
                    return Err(too_big_msg(owner, repo, total));
                }
                if let Err(e) = file.write_all(&chunk) {
                    drop(file);
                    let _ = std::fs::remove_file(dest);
                    return Err(format!("write tarball: {e}"));
                }
            }
            Ok(None) => break,
            Err(e) => {
                drop(file);
                let _ = std::fs::remove_file(dest);
                return Err(format!("reading {owner}/{repo} tarball: {e}"));
            }
        }
    }
    Ok(())
}

fn too_big_msg(owner: &str, repo: &str, bytes: u64) -> String {
    format!(
        "{owner}/{repo} tarball is {} MiB, over the {} MiB skill-install limit",
        bytes / (1024 * 1024),
        MAX_SKILL_TARBALL_BYTES / (1024 * 1024),
    )
}

/// Reject an extracted tree that exceeds the configured size/entry caps (a
/// decompression bomb). See `enforce_caps`.
fn enforce_extracted_caps(root: &Path) -> Result<(), String> {
    enforce_caps(root, MAX_SKILL_EXTRACTED_BYTES, MAX_SKILL_EXTRACTED_ENTRIES)
}

/// Walk `root`, summing regular-file bytes and counting entries, and bail the
/// instant either cap is crossed — so the walk itself stays bounded even on a
/// bomb. Symlinks are counted but not followed (their target is out of scope).
fn enforce_caps(root: &Path, max_bytes: u64, max_entries: usize) -> Result<(), String> {
    let mut total_bytes: u64 = 0;
    let mut entries: usize = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let read = std::fs::read_dir(&dir)
            .map_err(|e| format!("scan extracted tree {}: {e}", dir.display()))?;
        for entry in read {
            let entry = entry.map_err(|e| format!("scan extracted tree: {e}"))?;
            entries += 1;
            if entries > max_entries {
                return Err(format!(
                    "downloaded skill archive has over {max_entries} files — refusing to install"
                ));
            }
            let meta = entry
                .metadata()
                .map_err(|e| format!("stat {}: {e}", entry.path().display()))?;
            if meta.is_dir() {
                stack.push(entry.path());
            } else if meta.is_file() {
                total_bytes += meta.len();
                if total_bytes > max_bytes {
                    return Err(format!(
                        "downloaded skill archive is over {} MiB unpacked — refusing to install",
                        max_bytes / (1024 * 1024)
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Sanitize a skill's (possibly free-form) frontmatter name into a filesystem-
/// safe destination folder: `[A-Za-z0-9_-]`, other runs collapsed to `-`.
fn skill_folder_name(name: &str) -> String {
    let mapped: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    mapped.trim_matches('-').to_string()
}

/// Recursive copy of a skill folder, skipping VCS / build junk so a skill that
/// lives at a repo root doesn't drag `.git`/`node_modules` along.
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    // Never follow a symlinked source dir — `read_dir` would walk the link
    // target. Nested links are skipped per-entry below; this guards the top level.
    if std::fs::symlink_metadata(src)?.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "refusing to copy a symlinked directory",
        ));
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(".git" | "node_modules" | "target" | ".DS_Store")
        ) {
            continue;
        }
        let ty = entry.file_type()?;
        let to = dst.join(&name);
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), &to)?;
        }
        // symlinks and other types are skipped (don't follow links out of tree)
    }
    Ok(())
}

fn shellexpand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/")
        && let Some(home) = crate::services::system_env::home_dir()
    {
        return home.join(rest).to_string_lossy().into_owned();
    }
    p.to_string()
}

fn next_install_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Collect skills from `roots` in order, first name winning on collision.
fn discover_from_roots(roots: &[PathBuf]) -> Vec<Skill> {
    let mut skills: Vec<Skill> = Vec::new();
    for root in roots {
        for skill in read_root(root) {
            if !skills.iter().any(|existing| existing.name == skill.name) {
                skills.push(skill);
            }
        }
    }
    skills
}

/// Parse every `<root>/<name>/SKILL.md` under one root (alphabetical, so
/// discovery is deterministic). Missing/unreadable roots yield nothing.
fn read_root(root: &Path) -> Vec<Skill> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.iter().filter_map(|dir| load_skill(dir)).collect()
}

/// Load one skill folder; `None` if it has no readable `SKILL.md`.
fn load_skill(dir: &Path) -> Option<Skill> {
    let text = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let dir_name = dir.file_name()?.to_string_lossy().into_owned();
    let (front, body) = split_frontmatter(&text);
    let name = front
        .as_ref()
        .and_then(|f| field(f, "name"))
        .unwrap_or(dir_name);
    let description = front
        .as_ref()
        .and_then(|f| field(f, "description"))
        .unwrap_or_else(|| first_non_empty_line(body));
    Some(Skill {
        name,
        description,
        body: body.trim().to_string(),
        dir: dir.to_path_buf(),
    })
}

/// Split a leading `---`…`---` YAML block from the body. Returns `(frontmatter,
/// body)`; the frontmatter is `None` when the file doesn't open with `---`.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let rest = match text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    {
        Some(rest) => rest,
        None => return (None, text),
    };
    // Find the closing fence at the start of a line.
    for marker in ["\n---\n", "\n---\r\n", "\r\n---\r\n"] {
        if let Some(idx) = rest.find(marker) {
            return (Some(&rest[..idx]), &rest[idx + marker.len()..]);
        }
    }
    // Trailing `---` with no following newline (file ends at the fence).
    if let Some(stripped) = rest.strip_suffix("\n---") {
        return (Some(stripped), "");
    }
    (None, text)
}

/// Pull a `key: value` from a minimal frontmatter block. Handles an inline value
/// (quote-stripped) AND YAML block scalars — `key: >` (folded, newlines → spaces)
/// and `key: |` (literal, newlines kept), whose value is the following
/// more-indented lines. Claude Code skills routinely write a long `description: >`
/// across several lines, so without this the description would parse as just `>`.
fn field(frontmatter: &str, key: &str) -> Option<String> {
    let lines: Vec<&str> = frontmatter.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let indent = line.len() - line.trim_start().len();
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix(key) else {
            continue;
        };
        let Some(value) = rest.trim_start().strip_prefix(':') else {
            continue;
        };
        let value = value.trim();

        // Block scalar: `>` folds lines into spaces, `|` keeps newlines. Chomping
        // / indentation indicators (`>-`, `|+`, …) are tolerated — we just read
        // the indented body that follows.
        if value.starts_with('>') || value.starts_with('|') {
            let fold = value.starts_with('>');
            let mut collected: Vec<String> = Vec::new();
            for next in &lines[i + 1..] {
                if next.trim().is_empty() {
                    collected.push(String::new());
                    continue;
                }
                let next_indent = next.len() - next.trim_start().len();
                if next_indent <= indent {
                    break; // dedented back to a sibling key — block is done
                }
                collected.push(next.trim().to_string());
            }
            let joined = if fold {
                collected
                    .join(" ")
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                collected.join("\n").trim().to_string()
            };
            return (!joined.is_empty()).then_some(joined);
        }

        let unquoted = value
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            .unwrap_or(value);
        if !unquoted.is_empty() {
            return Some(unquoted.to_string());
        }
    }
    None
}

fn first_non_empty_line(body: &str) -> String {
    body.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

/// The `skill` tool, offered only when skills are discovered. The `name` enum
/// constrains it to known skills so the model can't invent one.
pub fn skill_tool_spec(skills: &[Skill]) -> ToolSpec {
    let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    ToolSpec {
        name: "skill".to_string(),
        description:
            "Load a skill's full instructions on demand, then follow them. Call this when \
the user's request matches a skill listed in the system prompt."
                .to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": names,
                    "description": "Name of the skill to load."
                }
            },
            "required": ["name"]
        }),
    }
}

/// One-line advert for the system prompt (and the `/skills` overlay): the
/// description's first sentence, hard capped so a long (600+ char) skill
/// description — common in ecosystem skills — can't bloat every turn. The full
/// body still loads on demand via `skill`.
pub(crate) fn advert_description(description: &str) -> String {
    const MAX: usize = 160;
    let one_line: String = description.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some((i, _)) = one_line.match_indices(". ").find(|&(i, _)| i < MAX) {
        return one_line[..=i].to_string(); // include the period, drop the rest
    }
    if one_line.chars().count() <= MAX {
        return one_line;
    }
    let truncated: String = one_line.chars().take(MAX).collect();
    let cut = truncated
        .rsplit_once(' ')
        .map(|(h, _)| h)
        .unwrap_or(&truncated);
    format!("{cut}…")
}

/// The system-prompt block that advertises available skills (names + one-line
/// descriptions). Empty when there are no skills.
pub fn skills_prompt_section(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut list = String::new();
    for skill in skills {
        list.push_str(&format!(
            "\n- {}: {}",
            skill.name,
            advert_description(&skill.description)
        ));
    }
    format!(
        "\n\nYou have skills — pre-written instructions for specific tasks. When a request matches \
one, call the `skill` tool with its name to load the full instructions, then follow them. The \
skill's folder may hold scripts or resources you can read/run with your file and shell tools. \
Available skills:{list}"
    )
}

/// Resolve a `skill` tool call to the loaded instructions, or an error naming the
/// available skills. The dir is surfaced so the model can find bundled files.
pub fn load_skill_result(skills: &[Skill], name: &str) -> Result<String, String> {
    match skills.iter().find(|s| s.name == name) {
        Some(skill) => Ok(format!(
            "Skill: {}\nFolder: {}\n\n{}",
            skill.name,
            skill.dir.display(),
            skill.body
        )),
        None => {
            let available: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
            Err(format!(
                "unknown skill `{name}` (available: {})",
                available.join(", ")
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "aivo-skills-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(root: &Path, folder: &str, contents: &str) {
        let dir = root.join(folder);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("SKILL.md"), contents).unwrap();
    }

    // Tests target explicit roots (a tempdir) rather than `discover_skills`,
    // which also reads the real `~/.config/aivo/skills` and would pollute assertions.
    #[test]
    fn discovers_and_parses_frontmatter() {
        let root = tmp();
        write_skill(
            &root,
            "pdf",
            "---\nname: pdf-filler\ndescription: \"Fill PDF forms\"\n---\nStep 1. Do the thing.\n",
        );

        let skills = discover_from_roots(&[root]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "pdf-filler");
        assert_eq!(skills[0].description, "Fill PDF forms");
        assert_eq!(skills[0].body, "Step 1. Do the thing.");
    }

    #[test]
    fn parses_folded_and_literal_block_scalar_descriptions() {
        // A folded `>` description (as Claude Code skills write it) collapses to a
        // clean one-liner, and the body after the frontmatter is unaffected.
        let root = tmp();
        write_skill(
            &root,
            "repo-study",
            "---\nname: repo-study\ndescription: >\n  MANUAL TRIGGER ONLY: invoke only\n  when the user types it.\n  Study a repo.\n---\nDo the study.\n",
        );
        let skills = discover_from_roots(std::slice::from_ref(&root));
        let s = skills.iter().find(|s| s.name == "repo-study").unwrap();
        assert_eq!(
            s.description,
            "MANUAL TRIGGER ONLY: invoke only when the user types it. Study a repo."
        );
        assert_eq!(s.body, "Do the study.");

        // A literal `|` description keeps its line breaks; a following key still
        // parses (the block ends when indentation returns to the key level).
        let root2 = tmp();
        write_skill(
            &root2,
            "lit",
            "---\ndescription: |\n  line one\n  line two\nname: lit\n---\nBody.\n",
        );
        let skills2 = discover_from_roots(std::slice::from_ref(&root2));
        let s2 = &skills2[0];
        assert_eq!(s2.name, "lit");
        assert_eq!(s2.description, "line one\nline two");
    }

    #[test]
    fn falls_back_to_folder_name_and_first_line() {
        let root = tmp();
        write_skill(
            &root,
            "no-front",
            "Just instructions, no frontmatter.\nmore.\n",
        );

        let skills = discover_from_roots(&[root]);
        assert_eq!(skills[0].name, "no-front");
        assert_eq!(skills[0].description, "Just instructions, no frontmatter.");
    }

    #[test]
    fn earlier_root_shadows_same_name() {
        let project = tmp();
        let user = tmp();
        write_skill(
            &project,
            "dup",
            "---\nname: dup\ndescription: from project\n---\nA\n",
        );
        write_skill(
            &user,
            "dup",
            "---\nname: dup\ndescription: from user\n---\nB\n",
        );

        let skills = discover_from_roots(&[project, user]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "from project");
    }

    #[test]
    fn load_result_and_unknown_error() {
        let skills = vec![Skill {
            name: "demo".to_string(),
            description: "d".to_string(),
            body: "BODY".to_string(),
            dir: PathBuf::from("/tmp/demo"),
        }];
        let ok = load_skill_result(&skills, "demo").unwrap();
        assert!(ok.contains("BODY"));
        assert!(ok.contains("/tmp/demo"));

        let err = load_skill_result(&skills, "nope").unwrap_err();
        assert!(err.contains("unknown skill"));
        assert!(err.contains("demo"));
    }

    #[test]
    fn advert_description_truncates_for_the_prompt() {
        // First sentence wins when present.
        assert_eq!(
            advert_description(
                "Adapt designs across screens. Ensures consistency everywhere and keeps going."
            ),
            "Adapt designs across screens."
        );
        // No sentence break → hard cap with an ellipsis on a word boundary.
        let long = "word ".repeat(80); // ~400 chars, no period
        let out = advert_description(&long);
        assert!(
            out.chars().count() <= 161,
            "too long: {}",
            out.chars().count()
        );
        assert!(out.ends_with('…'));
        // Short, no period → whitespace-collapsed but otherwise unchanged.
        assert_eq!(advert_description("just  a   skill"), "just a skill");
    }

    #[test]
    fn prompt_section_uses_truncated_descriptions() {
        let skills = vec![Skill {
            name: "x".to_string(),
            description: format!("Short summary. {}", "verbose ".repeat(60)),
            body: String::new(),
            dir: PathBuf::new(),
        }];
        let section = skills_prompt_section(&skills);
        assert!(section.contains("- x: Short summary."));
        assert!(!section.contains("verbose verbose"));
    }

    #[test]
    fn discovery_covers_agents_and_aivo_roots_with_agents_first() {
        // `.agents/skills` shadows `.aivo/skills` within a tier (first wins).
        let agents = tmp();
        let aivo = tmp();
        write_skill(
            &agents,
            "dup",
            "---\nname: dup\ndescription: from .agents\n---\nA\n",
        );
        write_skill(
            &aivo,
            "dup",
            "---\nname: dup\ndescription: from .aivo\n---\nB\n",
        );
        write_skill(
            &aivo,
            "solo",
            "---\nname: solo\ndescription: only here\n---\nC\n",
        );

        let skills = discover_from_roots(&[agents, aivo]);
        let dup = skills.iter().find(|s| s.name == "dup").unwrap();
        assert_eq!(dup.description, "from .agents");
        assert!(skills.iter().any(|s| s.name == "solo"));
    }

    #[test]
    fn tool_spec_enumerates_names() {
        let skills = vec![Skill {
            name: "alpha".to_string(),
            description: "a".to_string(),
            body: String::new(),
            dir: PathBuf::new(),
        }];
        let spec = skill_tool_spec(&skills);
        assert_eq!(spec.name, "skill");
        assert_eq!(spec.parameters["properties"]["name"]["enum"][0], "alpha");
    }

    /// Scaffold writes a discoverable SKILL.md (parses back to the given name +
    /// description), refuses a duplicate, and the result round-trips through
    /// discovery + removal. Targets a tempdir, never the real skills dir.
    #[test]
    fn scaffold_then_remove_round_trip() {
        let root = tmp();
        let path = scaffold_skill_at(&root, "changelog", "Summarize the git log").unwrap();
        assert!(path.ends_with("SKILL.md"));

        // The scaffold is a real, discoverable skill with our name + description.
        let skills = discover_from_roots(std::slice::from_ref(&root));
        let skill = skills.iter().find(|s| s.name == "changelog").unwrap();
        assert_eq!(skill.description, "Summarize the git log");
        assert!(!skill.body.is_empty(), "template body should not be empty");

        // A second scaffold of the same name refuses rather than clobbering.
        assert!(
            scaffold_skill_at(&root, "changelog", "other").is_err(),
            "duplicate name must error"
        );
        // Invalid (folder-unsafe) names are rejected.
        assert!(scaffold_skill_at(&root, "bad name", "x").is_err());

        // Remove deletes the folder; a second remove (gone) errors.
        remove_skill_dir(&skill.dir).unwrap();
        assert!(!skill.dir.exists(), "folder not removed");
        assert!(
            remove_skill_dir(&skill.dir).is_err(),
            "removing a vanished skill errors"
        );
        // The guard refuses a dir without a SKILL.md (can't nuke a random folder).
        let plain = tmp();
        assert!(
            remove_skill_dir(&plain).is_err(),
            "non-skill dir is guarded"
        );
    }

    #[test]
    fn skill_scope_classifies_project_vs_user() {
        let cwd = Path::new("/work/repo");
        assert_eq!(
            skill_scope(Path::new("/work/repo/.agents/skills/foo"), cwd),
            SkillScope::Project
        );
        assert_eq!(
            skill_scope(Path::new("/work/repo/.aivo/skills/bar"), cwd),
            SkillScope::Project
        );
        assert_eq!(
            skill_scope(Path::new("/home/me/.config/aivo/skills/baz"), cwd),
            SkillScope::User
        );
        // A project's `.claude/skills` is protected (repo-owned).
        assert_eq!(
            skill_scope(Path::new("/work/repo/.claude/skills/qux"), cwd),
            SkillScope::Project
        );
        // The user's Claude Code library is discovered but never deletable.
        if let Some(home) = crate::services::system_env::home_dir() {
            assert_eq!(
                skill_scope(&home.join(".claude/skills/repo-study"), cwd),
                SkillScope::Project
            );
        }
    }

    /// The embedded `create-skill` parses into a usable Skill (name from
    /// frontmatter, non-empty body, empty dir) that powers the `/create-skill`
    /// command. It is NOT auto-injected into discovery — discovery returns only
    /// what's on disk (asserted against empty roots, so the real `~/.claude` on
    /// the test machine can't influence the result).
    #[test]
    fn create_skill_builtin_parses_and_is_not_injected() {
        let sc = create_skill_builtin();
        assert_eq!(sc.name, "create-skill");
        assert!(!sc.description.is_empty());
        assert!(!sc.body.is_empty());
        assert!(sc.dir.as_os_str().is_empty(), "no folder on disk");
        // The advert (first sentence) stays within the prompt cap.
        assert!(advert_description(&sc.description).len() <= 161);

        // Discovery over controlled (empty) roots surfaces nothing — proving the
        // built-in command is not folded into skill discovery.
        assert!(
            discover_from_roots(&[]).is_empty(),
            "no built-in is injected into discovery"
        );
    }

    #[test]
    fn skill_folder_name_sanitizes() {
        assert_eq!(skill_folder_name("My Skill!"), "My-Skill");
        assert_eq!(skill_folder_name("ok_name-1"), "ok_name-1");
        assert_eq!(skill_folder_name("--edge--"), "edge");
    }

    /// `discover_installable` finds a root SKILL.md, a flat `skills/<name>`, and a
    /// catalog `skills/<cat>/<name>` — the `skills/*/SKILL.md` convention.
    #[test]
    fn discover_installable_covers_root_flat_and_catalog() {
        let root = tmp();
        std::fs::write(
            root.join("SKILL.md"),
            "---\nname: root-skill\ndescription: d\n---\nBody.\n",
        )
        .unwrap();
        write_skill(
            &root.join("skills"),
            "flat",
            "---\nname: flat-skill\ndescription: d\n---\nBody.\n",
        );
        write_skill(
            &root.join("skills").join("category"),
            "deep",
            "---\nname: deep-skill\ndescription: d\n---\nBody.\n",
        );
        let names: Vec<String> = discover_installable(&root)
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert!(names.contains(&"root-skill".to_string()), "{names:?}");
        assert!(names.contains(&"flat-skill".to_string()), "{names:?}");
        assert!(names.contains(&"deep-skill".to_string()), "{names:?}");
    }

    /// End-to-end install from a LOCAL source into a tempdir (no network): the
    /// sole-skill, ambiguous-many, pick-one, install-all, and collision cases.
    #[tokio::test]
    async fn install_from_local_source_paths() {
        // A single-skill source (root SKILL.md) installs straight away.
        let solo_src = tmp();
        std::fs::write(
            solo_src.join("SKILL.md"),
            "---\nname: solo\ndescription: d\n---\nBody.\n",
        )
        .unwrap();
        let dest = tmp();
        let out = install_from_source_into(&dest, solo_src.to_str().unwrap(), None)
            .await
            .unwrap();
        assert_eq!(out, InstallOutcome::Installed(vec!["solo".to_string()]));
        assert!(dest.join("solo").join("SKILL.md").is_file());

        // A multi-skill source with no filter is Ambiguous (installs nothing).
        let pack = tmp();
        write_skill(
            &pack.join("skills"),
            "alpha",
            "---\nname: alpha\ndescription: d\n---\nB\n",
        );
        write_skill(
            &pack.join("skills"),
            "beta",
            "---\nname: beta\ndescription: d\n---\nB\n",
        );
        let dest2 = tmp();
        let out = install_from_source_into(&dest2, pack.to_str().unwrap(), None)
            .await
            .unwrap();
        assert!(
            matches!(&out, InstallOutcome::Ambiguous(n) if n.len() == 2),
            "{out:?}"
        );
        assert!(
            !dest2.join("alpha").exists(),
            "ambiguous must install nothing"
        );

        // Picking one installs just it; a second pick collides.
        let out = install_from_source_into(&dest2, pack.to_str().unwrap(), Some("alpha"))
            .await
            .unwrap();
        assert_eq!(out, InstallOutcome::Installed(vec!["alpha".to_string()]));
        assert!(dest2.join("alpha").join("SKILL.md").is_file());
        assert!(!dest2.join("beta").exists());
        let err = install_from_source_into(&dest2, pack.to_str().unwrap(), Some("alpha"))
            .await
            .unwrap_err();
        assert!(err.contains("already exists"), "{err}");

        // `*` installs all.
        let dest3 = tmp();
        let out = install_from_source_into(&dest3, pack.to_str().unwrap(), Some("*"))
            .await
            .unwrap();
        assert!(
            matches!(&out, InstallOutcome::Installed(n) if n.len() == 2),
            "{out:?}"
        );
    }

    #[tokio::test]
    async fn install_from_local_source_errors() {
        // A directory with no SKILL.md anywhere.
        let empty = tmp();
        let err = install_from_source_into(&tmp(), empty.to_str().unwrap(), None)
            .await
            .unwrap_err();
        assert!(err.contains("no SKILL.md"), "{err}");
        // A path that isn't a directory.
        let err = install_from_source_into(&tmp(), "/no/such/aivo/skill/dir", None)
            .await
            .unwrap_err();
        assert!(err.contains("not a directory"), "{err}");
    }

    /// The extracted-tree guard rejects an archive that busts the entry cap or
    /// the byte cap (zip/tar-bomb shapes) and accepts an ordinary small tree.
    /// Uses tiny caps via `enforce_caps` so the test stays cheap.
    #[test]
    fn enforce_caps_rejects_bombs() {
        let dir = tmp();
        std::fs::write(dir.join("a"), b"hello").unwrap();
        std::fs::write(dir.join("b"), b"world").unwrap();
        // Comfortably under both caps.
        assert!(enforce_caps(&dir, 1024, 100).is_ok());
        // Entry cap busted (3 entries here: a, b, and the nested dir below none).
        let err = enforce_caps(&dir, 1024, 1).unwrap_err();
        assert!(err.contains("files"), "{err}");
        // Byte cap busted.
        let err = enforce_caps(&dir, 4, 100).unwrap_err();
        assert!(err.contains("unpacked"), "{err}");

        // The production wrapper passes a normal tree.
        assert!(enforce_extracted_caps(&dir).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // `enforce_caps` must count symlinks but never follow them: a crafted tarball
    // could otherwise make the scan walk out of the extracted tree or loop forever
    // through a symlink cycle. `DirEntry::metadata()` is the non-following
    // (`symlink_metadata`) variant, so a symlink is neither a dir (not recursed)
    // nor a file (its target's bytes aren't counted).
    #[cfg(unix)]
    #[test]
    fn enforce_caps_does_not_follow_symlinks() {
        use std::os::unix::fs::symlink;
        let root = tmp();
        std::fs::write(root.join("real.txt"), b"ten bytes!").unwrap();

        // A fat file living OUTSIDE the extracted tree — counting it would mean
        // the scan escaped via a link.
        let outside = tmp();
        std::fs::write(outside.join("big.bin"), vec![0u8; 5000]).unwrap();
        symlink(outside.join("big.bin"), root.join("link_to_outside_file")).unwrap();
        symlink(&outside, root.join("link_to_outside_dir")).unwrap();
        // A self-referential link — following it would spin forever.
        symlink(&root, root.join("loop")).unwrap();

        // Byte cap of 1000 sits below the 5000-byte target but above the 10 real
        // bytes: it stays Ok only because the link's target is not counted, and
        // the call returns at all only because the loop is not followed.
        assert!(enforce_caps(&root, 1000, 100).is_ok());

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // An untrusted source whose skill folder is a SYMLINK out of the tree
    // (`skills/evil -> /secrets`) must be refused, not followed-and-copied.
    #[cfg(unix)]
    #[tokio::test]
    async fn install_refuses_symlinked_skill_dir_escape() {
        use std::os::unix::fs::symlink;

        // A dir OUTSIDE the tree holding a valid skill plus a secret to exfiltrate.
        let outside = tmp();
        std::fs::write(
            outside.join("SKILL.md"),
            "---\nname: evil\ndescription: d\n---\nBody.\n",
        )
        .unwrap();
        std::fs::write(outside.join("secret.txt"), b"stolen").unwrap();

        let src = tmp();
        std::fs::create_dir_all(src.join("skills")).unwrap();
        symlink(&outside, src.join("skills").join("evil")).unwrap();

        // Discovery still finds it (a symlinked dir reports `is_dir()`), so the
        // guard — not discovery — is what must stop the install.
        assert!(
            discover_installable(&src).iter().any(|s| s.name == "evil"),
            "symlinked skill should be discovered so the guard is exercised"
        );

        let dest = tmp();
        let err = install_from_source_into(&dest, src.to_str().unwrap(), Some("evil"))
            .await
            .unwrap_err();
        assert!(
            err.contains("symlink escape") || err.contains("outside"),
            "{err}"
        );
        // Nothing was copied — not even the secret.
        assert!(!dest.join("evil").exists(), "must not install the escape");
        assert!(!dest.join("evil").join("secret.txt").exists());

        // Defense-in-depth: copy_dir_all refuses a symlinked source directly.
        let direct = install_dir_copy_rejects_symlink(&src.join("skills").join("evil"));
        assert!(direct, "copy_dir_all must refuse a symlinked source dir");

        let _ = std::fs::remove_dir_all(&outside);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[cfg(unix)]
    fn install_dir_copy_rejects_symlink(src: &Path) -> bool {
        copy_dir_all(src, &tmp().join("out")).is_err()
    }
}
