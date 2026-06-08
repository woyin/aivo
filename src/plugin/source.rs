//! Smart install sources for `aivo plugins install`. `classify` parses a source
//! string into a `SourceKind`; `materialize` fetches/builds the artifact and
//! installs `aivo-<name>` into the managed dir. Supported forms:
//!
//!   local path · `http(s)://…/binary` · `github:owner/repo[@tag]` / `gh:` /
//!   bare `github.com/owner/repo` · `npm:[@scope/]pkg[@version]` · `cargo:crate[@version]`
//!
//! Network bases are overridable for tests / private mirrors: `AIVO_GITHUB_API`,
//! `AIVO_NPM_REGISTRY`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::services::archive::{self, ArchiveKind};
use crate::services::device_fingerprint::hex_sha256;
use crate::services::http_utils;
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::style;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceKind {
    LocalPath,
    DirectUrl,
    GitHub {
        owner: String,
        repo: String,
        tag: Option<String>,
    },
    Npm {
        pkg: String,
        version: Option<String>,
    },
    Cargo {
        krate: String,
        version: Option<String>,
    },
}

pub(crate) struct Materialized {
    pub primary: PathBuf,
    pub checksum: Option<String>,
    /// True only for a local path the user pointed at — the one case aivo
    /// executes the artifact at install (to probe `--aivo-manifest`). Remote
    /// artifacts (url/github/npm/cargo) aren't run at install time; their
    /// manifest is probed lazily on first dispatch instead.
    pub trusted_local: bool,
}

// ── classification ──────────────────────────────────────────────────────────

/// Parse a source string. A recognized scheme prefix that's malformed errors;
/// anything unrecognized is treated as a local path.
pub(crate) fn classify(source: &str) -> Result<SourceKind> {
    let s = source.trim();
    if let Some(rest) = s.strip_prefix("github:").or_else(|| s.strip_prefix("gh:")) {
        let (owner, repo, tag) = parse_owner_repo(rest)
            .with_context(|| format!("invalid `{source}` (expected github:owner/repo[@tag])"))?;
        return Ok(SourceKind::GitHub { owner, repo, tag });
    }
    if let Some(rest) = s.strip_prefix("npm:") {
        let (pkg, version) = split_scoped_version(rest);
        if pkg.is_empty() {
            anyhow::bail!("invalid `{source}` (expected npm:[@scope/]pkg[@version])");
        }
        return Ok(SourceKind::Npm { pkg, version });
    }
    if let Some(rest) = s.strip_prefix("cargo:") {
        let (krate, version) = split_scoped_version(rest);
        if krate.is_empty() {
            anyhow::bail!("invalid `{source}` (expected cargo:crate[@version])");
        }
        return Ok(SourceKind::Cargo { krate, version });
    }
    if let Some(gh) = parse_bare_github_url(s) {
        return Ok(gh);
    }
    if is_url(s) {
        return Ok(SourceKind::DirectUrl);
    }
    Ok(SourceKind::LocalPath)
}

/// Suggested plugin name for a source (scheme/version/scope/`aivo-` stripped).
pub(crate) fn suggested_name(source: &str) -> Option<String> {
    let raw = match classify(source).ok()? {
        SourceKind::GitHub { repo, .. } => repo,
        SourceKind::Npm { pkg, .. } => pkg.rsplit('/').next().unwrap_or(&pkg).to_string(),
        SourceKind::Cargo { krate, .. } => krate,
        SourceKind::LocalPath | SourceKind::DirectUrl => basename(source)?,
    };
    let name = raw.strip_prefix("aivo-").unwrap_or(&raw).trim();
    (!name.is_empty()).then(|| name.to_string())
}

fn parse_owner_repo(s: &str) -> Option<(String, String, Option<String>)> {
    let (path, tag) = match s.split_once('@') {
        Some((p, t)) if !t.is_empty() => (p, Some(t.to_string())),
        _ => (s, None),
    };
    let (owner, repo) = path.split_once('/')?;
    let repo = repo.strip_suffix(".git").unwrap_or(repo);
    if owner.is_empty() || repo.is_empty() || repo.contains('/') {
        return None;
    }
    Some((owner.to_string(), repo.to_string(), tag))
}

/// Split `[@scope/]pkg[@version]`. The version `@` is the LAST one and never the
/// scope's leading `@`.
fn split_scoped_version(s: &str) -> (String, Option<String>) {
    if let Some(idx) = s.rfind('@')
        && idx > 0
    {
        let ver = &s[idx + 1..];
        if !ver.is_empty() {
            return (s[..idx].to_string(), Some(ver.to_string()));
        }
    }
    (s.to_string(), None)
}

/// A bare `https://github.com/owner/repo` (exactly two path segments) → GitHub.
/// A deeper URL (a release asset, blob, raw, …) is left as a DirectUrl.
fn parse_bare_github_url(s: &str) -> Option<SourceKind> {
    let rest = s
        .strip_prefix("https://github.com/")
        .or_else(|| s.strip_prefix("http://github.com/"))?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let segs: Vec<&str> = rest.split('/').collect();
    if segs.len() != 2 || segs[0].is_empty() || segs[1].is_empty() {
        return None;
    }
    Some(SourceKind::GitHub {
        owner: segs[0].to_string(),
        repo: segs[1].strip_suffix(".git").unwrap_or(segs[1]).to_string(),
        tag: None,
    })
}

/// Final path/url segment minus query/frag and file extension.
fn basename(source: &str) -> Option<String> {
    let last = source.rsplit(['/', '\\']).next().unwrap_or(source);
    let last = last.split(['?', '#']).next().unwrap_or(last);
    let stem = Path::new(last)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(last);
    (!stem.is_empty()).then(|| stem.to_string())
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

// ── GitHub asset matching (pure) ────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct NoAssetMatch {
    pub available: Vec<String>,
}

fn os_aliases(os: &str) -> &'static [&'static str] {
    match os {
        "macos" => &["apple-darwin", "darwin", "macos", "apple", "osx"],
        "linux" => &["linux"],
        "windows" => &["pc-windows", "windows", "win64", "win32"],
        _ => &[],
    }
}

fn arch_aliases(arch: &str) -> &'static [&'static str] {
    match arch {
        "x86_64" => &["x86_64", "x86-64", "amd64", "x64"],
        "aarch64" => &["aarch64", "arm64"],
        _ => &[],
    }
}

/// Arch tokens for every arch *other* than `arch`, so `pick_asset` can reject an
/// asset that explicitly targets a different CPU while still accepting one that
/// names no arch at all (an OS-only or universal build).
fn foreign_arch_aliases(arch: &str) -> Vec<&'static str> {
    ["x86_64", "aarch64"]
        .iter()
        .filter(|known| **known != arch)
        .flat_map(|known| arch_aliases(known).iter().copied())
        .collect()
}

fn target_triples(os: &str, arch: &str) -> &'static [&'static str] {
    match (os, arch) {
        ("macos", "x86_64") => &["x86_64-apple-darwin"],
        ("macos", "aarch64") => &["aarch64-apple-darwin"],
        ("linux", "x86_64") => &["x86_64-unknown-linux-gnu", "x86_64-unknown-linux-musl"],
        ("linux", "aarch64") => &["aarch64-unknown-linux-gnu", "aarch64-unknown-linux-musl"],
        ("windows", "x86_64") => &["x86_64-pc-windows-msvc", "x86_64-pc-windows-gnu"],
        ("windows", "aarch64") => &["aarch64-pc-windows-msvc"],
        _ => &[],
    }
}

/// Name tokens that mark a build as arch-independent (a macOS universal binary,
/// a `noarch` package), so `pick_asset` accepts it on any CPU.
const ARCH_AGNOSTIC_MARKERS: &[&str] = &["universal", "noarch"];

fn asset_score(name_lc: &str, os: &str, arch: &str, prefer_musl: bool) -> i32 {
    let mut score = 0;
    if target_triples(os, arch).iter().any(|t| name_lc.contains(t)) {
        score += 100;
    }
    // An asset naming the host arch outranks an arch-agnostic (universal/noarch)
    // one, so the exact build wins when both are published.
    if arch_aliases(arch).iter().any(|t| name_lc.contains(t)) {
        score += 30;
    }
    if os == "linux" {
        let (strong, weak) = if prefer_musl {
            ("musl", "gnu")
        } else {
            ("gnu", "musl")
        };
        if name_lc.contains(strong) {
            score += 10;
        } else if name_lc.contains(weak) {
            score += 2;
        }
    }
    if os == "windows" {
        if name_lc.contains("msvc") {
            score += 5;
        }
        if name_lc.ends_with(".zip") || name_lc.ends_with(".exe") {
            score += 10;
        }
    }
    // Prefer a ready-to-run binary over an archive (no extract needed).
    if archive::archive_kind_for(name_lc).is_none() && !name_lc.ends_with(".exe") {
        score += 20;
    }
    score
}

/// Pick the release asset best matching `(os, arch)`. A sole asset is accepted
/// even without tokens; otherwise an asset must carry an OS and an arch token.
pub(crate) fn pick_asset<'a>(
    names: &[&'a str],
    os: &str,
    arch: &str,
    prefer_musl: bool,
) -> std::result::Result<&'a str, NoAssetMatch> {
    let os_al = os_aliases(os);
    let arch_al = arch_aliases(arch);
    let foreign_arch = foreign_arch_aliases(arch);
    // Score every asset once (lowercasing each name once). Eligible = names the
    // host OS and either names the host arch, is explicitly arch-agnostic
    // (universal/noarch), or names no *other* arch — the last clause accepts an
    // OS-only asset (e.g. `tool-darwin.tar.gz`) while still rejecting a build
    // that targets a different CPU.
    let mut scored: Vec<(bool, i32, &'a str)> = names
        .iter()
        .map(|n| {
            let lc = n.to_ascii_lowercase();
            let has_os = os_al.iter().any(|a| lc.contains(a));
            let has_host_arch = arch_al.iter().any(|a| lc.contains(a));
            let arch_agnostic = ARCH_AGNOSTIC_MARKERS.iter().any(|m| lc.contains(m));
            let names_foreign_arch = foreign_arch.iter().any(|a| lc.contains(a));
            let eligible = has_os && (has_host_arch || arch_agnostic || !names_foreign_arch);
            (eligible, asset_score(&lc, os, arch, prefer_musl), *n)
        })
        .collect();

    // Fall back to a sole asset; otherwise require an eligible match.
    let sole = names.len() == 1;
    if !sole && !scored.iter().any(|(e, _, _)| *e) {
        return Err(NoAssetMatch {
            available: names.iter().map(|s| s.to_string()).collect(),
        });
    }
    scored.retain(|(e, _, _)| *e || sole);

    // Best first: higher score, then shorter name, then lexicographically smaller.
    scored.sort_by(|(_, sa, a), (_, sb, b)| {
        sb.cmp(sa)
            .then_with(|| a.len().cmp(&b.len()))
            .then_with(|| a.cmp(b))
    });
    Ok(scored[0].2)
}

// ── npm helpers (pure) ──────────────────────────────────────────────────────

fn npm_metadata_url(base: &str, pkg: &str) -> String {
    let base = base.trim_end_matches('/');
    match pkg.strip_prefix('@') {
        // @scope/name → @scope%2fname (only that slash is encoded)
        Some(rest) => format!("{base}/@{}", rest.replacen('/', "%2f", 1)),
        None => format!("{base}/{pkg}"),
    }
}

/// Resolve a package's `bin` to a single script path for `aivo-<name>`.
fn resolve_npm_bin(bin: &serde_json::Value, name: &str) -> Result<String> {
    match bin {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Object(map) => {
            for key in [format!("aivo-{name}"), name.to_string()] {
                if let Some(v) = map.get(&key).and_then(|v| v.as_str()) {
                    return Ok(v.to_string());
                }
            }
            if map.len() == 1
                && let Some(v) = map.values().next().and_then(|v| v.as_str())
            {
                return Ok(v.to_string());
            }
            anyhow::bail!(
                "npm package `bin` has multiple commands and none is `aivo-{name}`: {}",
                map.keys().cloned().collect::<Vec<_>>().join(", "),
            )
        }
        _ => anyhow::bail!("npm package declares no `bin` to run"),
    }
}

fn unix_shim(bin_abs: &Path) -> String {
    format!("#!/bin/sh\nexec node \"{}\" \"$@\"\n", bin_abs.display())
}

fn windows_cmd(bin_abs: &Path) -> String {
    format!("@node \"{}\" %*\r\n", bin_abs.display())
}

// ── content sniffing (pure) ─────────────────────────────────────────────────

fn looks_like_html(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(512)];
    let lower = String::from_utf8_lossy(head)
        .trim_start()
        .to_ascii_lowercase();
    lower.starts_with("<!doctype html") || lower.starts_with("<html") || lower.starts_with("<?xml")
}

// ── materialization ─────────────────────────────────────────────────────────

struct Body {
    bytes: Vec<u8>,
    content_type: Option<String>,
}

/// `quiet` mutes the resolve/asset/download progress lines (used by `update`,
/// which re-fetches known plugins and reports its own per-plugin result);
/// warnings and errors are always shown.
pub(crate) async fn materialize(
    source: &str,
    name: &str,
    dir: &Path,
    quiet: bool,
) -> Result<Materialized> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    match classify(source)? {
        SourceKind::LocalPath => materialize_local(source, name, dir),
        SourceKind::DirectUrl => materialize_url(source, name, dir, quiet).await,
        SourceKind::GitHub { owner, repo, tag } => {
            materialize_github(&owner, &repo, tag.as_deref(), name, dir, quiet).await
        }
        SourceKind::Npm { pkg, version } => {
            materialize_npm(&pkg, version.as_deref(), name, dir, quiet).await
        }
        SourceKind::Cargo { krate, version } => {
            materialize_cargo(&krate, version.as_deref(), name, dir, quiet).await
        }
    }
}

fn materialize_local(source: &str, name: &str, dir: &Path) -> Result<Materialized> {
    let path = Path::new(source);
    let meta =
        std::fs::metadata(path).with_context(|| format!("reading local source `{source}`"))?;
    if !meta.is_file() {
        anyhow::bail!("`{source}` is not a file");
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading `{source}`"))?;
    let target = dir.join(plugin_filename(name));
    write_executable(&target, &bytes)?;
    Ok(Materialized {
        primary: target,
        checksum: Some(sha(&bytes)),
        trusted_local: true,
    })
}

async fn materialize_url(url: &str, name: &str, dir: &Path, quiet: bool) -> Result<Materialized> {
    let body = download(url, quiet).await?;
    let html_ct = body
        .content_type
        .as_deref()
        .is_some_and(|ct| ct.contains("text/html"));
    if html_ct || looks_like_html(&body.bytes) {
        anyhow::bail!(
            "`{url}` returned an HTML page, not a binary.\n  \
             For a GitHub repo use `github:owner/repo`; for a release asset, link directly to the asset file."
        );
    }
    let checksum = sha(&body.bytes);
    let primary = install_payload(&body.bytes, basename(url).as_deref(), name, dir)?;
    Ok(Materialized {
        primary,
        checksum: Some(checksum),
        trusted_local: false,
    })
}

/// Best-effort: does the host run musl libc (Alpine, etc.)? A musl-built aivo
/// always does; a glibc build probes at runtime so it still prefers a musl asset
/// on a musl host. Decides the gnu↔musl tiebreak in `asset_score`.
fn host_prefers_musl() -> bool {
    // A musl-built binary only runs on a musl-capable host.
    cfg!(target_env = "musl") || musl_host_runtime()
}

#[cfg(target_os = "linux")]
fn musl_host_runtime() -> bool {
    // Alpine's marker, or the musl dynamic loader at `/lib/ld-musl-<arch>.so.*`.
    Path::new("/etc/alpine-release").exists()
        || std::fs::read_dir("/lib").is_ok_and(|entries| {
            entries
                .flatten()
                .any(|e| e.file_name().to_string_lossy().starts_with("ld-musl-"))
        })
}

#[cfg(not(target_os = "linux"))]
fn musl_host_runtime() -> bool {
    false
}

async fn materialize_github(
    owner: &str,
    repo: &str,
    tag: Option<&str>,
    name: &str,
    dir: &Path,
    quiet: bool,
) -> Result<Materialized> {
    let base = env_base("AIVO_GITHUB_API", "https://api.github.com");
    let url = match tag {
        Some(t) => format!("{base}/repos/{owner}/{repo}/releases/tags/{t}"),
        None => format!("{base}/repos/{owner}/{repo}/releases/latest"),
    };
    if !quiet {
        eprintln!(
            "  {} Resolving {} release…",
            style::dim("·"),
            style::cyan(format!("{owner}/{repo}"))
        );
    }
    let client = http_client(30)?;
    let resp = client
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .with_context(|| format!("querying {url}"))?;
    match resp.status() {
        reqwest::StatusCode::NOT_FOUND => match tag {
            Some(t) => anyhow::bail!("no release tagged `{t}` in {owner}/{repo}"),
            None => {
                anyhow::bail!("{owner}/{repo} has no published releases (or the repo is private)")
            }
        },
        reqwest::StatusCode::FORBIDDEN => anyhow::bail!(
            "GitHub API rate limit hit (unauthenticated). Try again later, or install from a direct release-asset URL."
        ),
        _ => {}
    }
    let resp = resp
        .error_for_status()
        .with_context(|| format!("GitHub API {url}"))?;
    let manifest: serde_json::Value = resp.json().await.context("parsing GitHub release JSON")?;
    let assets = manifest
        .get("assets")
        .and_then(|v| v.as_array())
        .context("GitHub release has no `assets`")?;
    let names: Vec<&str> = assets
        .iter()
        .filter_map(|a| a.get("name").and_then(|v| v.as_str()))
        .collect();
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let picked = pick_asset(&names, os, arch, host_prefers_musl()).map_err(|e| {
        anyhow::anyhow!(
            "no release asset for {os}/{arch} in {owner}/{repo}.\n  Available: {}.\n  \
             Publish an asset whose name contains the OS and arch (e.g. `aivo-{name}-{arch}-…`).",
            e.available.join(", "),
        )
    })?;
    let asset_url = assets
        .iter()
        .find(|a| a.get("name").and_then(|v| v.as_str()) == Some(picked))
        .and_then(|a| a.get("browser_download_url"))
        .and_then(|v| v.as_str())
        .context("chosen asset has no download URL")?;
    if !quiet {
        eprintln!("  {} {picked}", style::dim("·"));
    }
    let body = download(asset_url, quiet).await?;
    let checksum = sha(&body.bytes);
    let primary = install_payload(&body.bytes, Some(picked), name, dir)?;
    Ok(Materialized {
        primary,
        checksum: Some(checksum),
        trusted_local: false,
    })
}

async fn materialize_npm(
    pkg: &str,
    version: Option<&str>,
    name: &str,
    dir: &Path,
    quiet: bool,
) -> Result<Materialized> {
    let base = env_base("AIVO_NPM_REGISTRY", "https://registry.npmjs.org");
    let meta_url = npm_metadata_url(&base, pkg);
    let client = http_client(30)?;
    let meta: serde_json::Value = client
        .get(&meta_url)
        .send()
        .await
        .with_context(|| format!("querying {meta_url}"))?
        .error_for_status()
        .with_context(|| format!("npm metadata for `{pkg}`"))?
        .json()
        .await
        .context("parsing npm metadata")?;

    let ver = match version {
        Some(v) => v.to_string(),
        None => meta
            .get("dist-tags")
            .and_then(|t| t.get("latest"))
            .and_then(|v| v.as_str())
            .context("npm metadata missing dist-tags.latest")?
            .to_string(),
    };
    let vmeta = meta
        .get("versions")
        .and_then(|vs| vs.get(&ver))
        .with_context(|| format!("npm package `{pkg}` has no version `{ver}`"))?;
    let tarball = vmeta
        .get("dist")
        .and_then(|d| d.get("tarball"))
        .and_then(|v| v.as_str())
        .context("npm version missing dist.tarball")?;
    if !quiet {
        eprintln!("  {} {pkg}@{ver}", style::dim("·"));
    }
    let body = download(tarball, quiet).await?;
    let checksum = sha(&body.bytes);

    let bundle = dir.join(format!("aivo-{name}.d"));
    let _ = std::fs::remove_dir_all(&bundle);
    std::fs::create_dir_all(&bundle).with_context(|| format!("creating {}", bundle.display()))?;
    let tgz = bundle.join(".pkg.tgz");
    std::fs::write(&tgz, &body.bytes).context("writing npm tarball")?;
    archive::extract_archive(&tgz, &bundle, ArchiveKind::TarGz)?;
    let _ = std::fs::remove_file(&tgz);
    archive::flatten_single_subdir(&bundle)?; // npm tarballs wrap in `package/`

    let bin_value = match vmeta.get("bin") {
        Some(b) if !b.is_null() => b.clone(),
        _ => {
            let pj = std::fs::read(bundle.join("package.json"))
                .context("npm package has no package.json")?;
            serde_json::from_slice::<serde_json::Value>(&pj)
                .context("parsing package.json")?
                .get("bin")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        }
    };
    let bin_rel = resolve_npm_bin(&bin_value, name)?;
    let bin_abs = bundle.join(bin_rel.trim_start_matches("./"));
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(&bin_abs) {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = meta.permissions();
        perms.set_mode(0o755);
        let _ = std::fs::set_permissions(&bin_abs, perms);
    }

    if find_in_dirs("node", &collect_path_dirs()).is_none() {
        eprintln!(
            "  {} {}",
            style::yellow("!"),
            style::dim(format!(
                "node was not found on PATH; `aivo {name}` needs Node.js to run"
            ))
        );
    }

    let primary = write_shim(name, &bin_abs, dir)?;
    Ok(Materialized {
        primary,
        checksum: Some(checksum),
        trusted_local: false,
    })
}

async fn materialize_cargo(
    krate: &str,
    version: Option<&str>,
    name: &str,
    dir: &Path,
    quiet: bool,
) -> Result<Materialized> {
    if find_in_dirs("cargo", &collect_path_dirs()).is_none() {
        anyhow::bail!(
            "`cargo` is not on PATH. Install Rust (https://rustup.rs) to use `cargo:` sources."
        );
    }
    let tmp = dir.join(format!(".aivo-{name}.cargo"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let _cleanup = CleanupDir(tmp.clone());
    if !quiet {
        eprintln!(
            "  {} cargo install {krate}{}…",
            style::dim("·"),
            version.map(|v| format!("@{v}")).unwrap_or_default()
        );
    }
    let mut cmd = tokio::process::Command::new("cargo");
    cmd.arg("install").arg(krate);
    if let Some(v) = version {
        cmd.arg("--version").arg(v);
    }
    cmd.arg("--root").arg(&tmp).arg("--force");
    let status = cmd.status().await.context("running cargo install")?;
    if !status.success() {
        anyhow::bail!("`cargo install {krate}` failed");
    }
    let picked = archive::find_executable(&tmp.join("bin"), name)?;
    let bytes = std::fs::read(&picked).context("reading the built binary")?;
    let target = dir.join(plugin_filename(name));
    write_executable(&target, &bytes)?;
    Ok(Materialized {
        primary: target,
        checksum: Some(sha(&bytes)),
        trusted_local: false,
    })
}

/// Install a downloaded payload (raw binary or archive) as `aivo-<name>`,
/// returning the installed path. `filename` hints the archive type.
fn install_payload(
    bytes: &[u8],
    filename: Option<&str>,
    name: &str,
    dir: &Path,
) -> Result<PathBuf> {
    if let Some(kind) = filename.and_then(archive::archive_kind_for) {
        let work = dir.join(format!(".aivo-{name}.unpack"));
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(&work).with_context(|| format!("creating {}", work.display()))?;
        let _cleanup = CleanupDir(work.clone());
        let archive_path = work.join("artifact");
        std::fs::write(&archive_path, bytes).context("writing the archive")?;
        archive::extract_archive(&archive_path, &work, kind)?;
        let _ = std::fs::remove_file(&archive_path);
        let found = archive::find_executable(&work, name)?;
        let bin = std::fs::read(&found).context("reading the unpacked binary")?;
        let target = dir.join(plugin_filename(name));
        write_executable(&target, &bin)?;
        Ok(target)
    } else {
        let target = dir.join(plugin_filename(name));
        write_executable(&target, bytes)?;
        Ok(target)
    }
}

/// aivo's shared HTTP client (proxy / IPv4 / Termux-DNS handling) with the
/// `aivo-cli` User-Agent as a default header.
fn http_client(timeout_secs: u64) -> Result<reqwest::Client> {
    http_utils::aivo_http_client_builder()
        .timeout(Duration::from_secs(timeout_secs))
        .user_agent("aivo-cli")
        .build()
        .context("building HTTP client")
}

/// A network base from `var`, else `default` (trailing slash trimmed). Lets
/// tests and private mirrors (GitHub Enterprise, a custom npm registry) redirect.
fn env_base(var: &str, default: &str) -> String {
    std::env::var(var)
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.trim_end_matches('/').to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Removes a directory when dropped — straight-line cleanup for the temp work
/// dirs used during extraction / cargo builds, on success and error alike.
struct CleanupDir(PathBuf);

impl Drop for CleanupDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn download(url: &str, quiet: bool) -> Result<Body> {
    if !quiet {
        eprintln!("  {} Downloading {url}", style::dim("·"));
    }
    let client = http_client(180)?;
    let resp = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("requesting {url}"))?
        .error_for_status()
        .with_context(|| format!("downloading {url}"))?;
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let bytes = resp
        .bytes()
        .await
        .context("reading download body")?
        .to_vec();
    Ok(Body {
        bytes,
        content_type,
    })
}

fn sha(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_sha256(bytes))
}

// ── filesystem helpers (moved from commands::plugins) ───────────────────────

/// On-disk filename for a binary plugin: `aivo-<name>` (`.exe` on Windows).
pub(crate) fn plugin_filename(name: &str) -> String {
    if cfg!(windows) {
        format!("aivo-{name}.exe")
    } else {
        format!("aivo-{name}")
    }
}

fn write_shim(name: &str, bin_abs: &Path, dir: &Path) -> Result<PathBuf> {
    if cfg!(windows) {
        let path = dir.join(format!("aivo-{name}.cmd"));
        std::fs::write(&path, windows_cmd(bin_abs))
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(path)
    } else {
        let path = dir.join(format!("aivo-{name}"));
        std::fs::write(&path, unix_shim(bin_abs))
            .with_context(|| format!("writing {}", path.display()))?;
        set_executable(&path)?;
        Ok(path)
    }
}

/// Write via a temp dotfile + rename so a partial write never leaves a
/// half-written (or discoverable `aivo-*`) binary. Sets +x on Unix.
fn write_executable(target: &Path, bytes: &[u8]) -> Result<()> {
    let file_name = target
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = target.with_file_name(format!(".{file_name}.tmp"));
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    set_executable(&tmp)?;
    std::fs::rename(&tmp, target).with_context(|| format!("installing {}", target.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_schemes() {
        assert_eq!(
            classify("github:o/aivo-amp").unwrap(),
            SourceKind::GitHub {
                owner: "o".into(),
                repo: "aivo-amp".into(),
                tag: None
            }
        );
        assert_eq!(
            classify("github:o/repo@v1.2.3").unwrap(),
            SourceKind::GitHub {
                owner: "o".into(),
                repo: "repo".into(),
                tag: Some("v1.2.3".into())
            }
        );
        assert!(matches!(
            classify("gh:o/r").unwrap(),
            SourceKind::GitHub { .. }
        ));
        assert!(matches!(
            classify("https://github.com/o/r").unwrap(),
            SourceKind::GitHub { .. }
        ));
        // a deeper github URL is a direct asset, not a repo
        assert_eq!(
            classify("https://github.com/o/r/releases/download/v1/aivo-amp").unwrap(),
            SourceKind::DirectUrl
        );
        assert_eq!(
            classify("npm:@scope/aivo-amp@1.0.0").unwrap(),
            SourceKind::Npm {
                pkg: "@scope/aivo-amp".into(),
                version: Some("1.0.0".into())
            }
        );
        assert_eq!(
            classify("npm:foo").unwrap(),
            SourceKind::Npm {
                pkg: "foo".into(),
                version: None
            }
        );
        assert_eq!(
            classify("cargo:ripgrep@13").unwrap(),
            SourceKind::Cargo {
                krate: "ripgrep".into(),
                version: Some("13".into())
            }
        );
        assert_eq!(classify("./bin/aivo-amp").unwrap(), SourceKind::LocalPath);
        assert_eq!(
            classify("https://x.dev/dl/aivo-amp").unwrap(),
            SourceKind::DirectUrl
        );
        assert!(classify("github:nope").is_err());
    }

    #[test]
    fn suggested_names() {
        assert_eq!(suggested_name("github:o/aivo-amp").as_deref(), Some("amp"));
        assert_eq!(suggested_name("github:o/widget").as_deref(), Some("widget"));
        assert_eq!(
            suggested_name("npm:@acme/aivo-foo@1").as_deref(),
            Some("foo")
        );
        assert_eq!(suggested_name("cargo:aivo-bar").as_deref(), Some("bar"));
        assert_eq!(
            suggested_name("/usr/local/bin/mytool").as_deref(),
            Some("mytool")
        );
        assert_eq!(
            suggested_name("https://x.dev/dl/aivo-amp.exe?v=1").as_deref(),
            Some("amp")
        );
    }

    #[test]
    fn picks_asset_for_host() {
        let macos = &[
            "app-x86_64-apple-darwin.tar.gz",
            "app-aarch64-apple-darwin.tar.gz",
            "app-x86_64-unknown-linux-gnu.tar.gz",
        ];
        assert_eq!(
            pick_asset(macos, "macos", "aarch64", false).unwrap(),
            "app-aarch64-apple-darwin.tar.gz"
        );
        // darwin must not be matched as windows ("win" ⊂ "darwin")
        assert!(pick_asset(macos, "windows", "x86_64", false).is_err());

        let linux = &[
            "app_Linux_x86_64.tar.gz",
            "app-x86_64-unknown-linux-gnu.tar.gz",
            "app-x86_64-unknown-linux-musl.tar.gz",
        ];
        assert_eq!(
            pick_asset(linux, "linux", "x86_64", false).unwrap(),
            "app-x86_64-unknown-linux-gnu.tar.gz"
        );
        assert_eq!(
            pick_asset(linux, "linux", "x86_64", true).unwrap(),
            "app-x86_64-unknown-linux-musl.tar.gz"
        );

        let win = &["app-windows-x64.zip", "app-x86_64-pc-windows-msvc.zip"];
        assert_eq!(
            pick_asset(win, "windows", "x86_64", false).unwrap(),
            "app-x86_64-pc-windows-msvc.zip"
        );

        // sole asset accepted even without tokens
        assert_eq!(
            pick_asset(&["aivo-amp"], "linux", "x86_64", false).unwrap(),
            "aivo-amp"
        );
        // nothing matches → Err with the list
        assert!(pick_asset(&["a.tar.gz", "b.zip"], "linux", "x86_64", false).is_err());
        // a raw binary outscores an equivalent archive
        let raw = &["app-linux-x64", "app-linux-x64.tar.gz"];
        assert_eq!(
            pick_asset(raw, "linux", "x86_64", false).unwrap(),
            "app-linux-x64"
        );

        // arch-agnostic fallback: a universal build matches when there's no
        // exact-arch asset, but an exact-arch asset still wins when both exist.
        let mac_uni = &[
            "app-macos-universal.tar.gz",
            "app-x86_64-apple-darwin.tar.gz",
        ];
        assert_eq!(
            pick_asset(mac_uni, "macos", "aarch64", false).unwrap(),
            "app-macos-universal.tar.gz"
        );
        assert_eq!(
            pick_asset(mac_uni, "macos", "x86_64", false).unwrap(),
            "app-x86_64-apple-darwin.tar.gz"
        );

        // an OS-only asset (no arch token) is accepted as a fallback…
        let osonly = &["tool-linux.tar.gz", "tool-darwin.tar.gz"];
        assert_eq!(
            pick_asset(osonly, "linux", "x86_64", false).unwrap(),
            "tool-linux.tar.gz"
        );
        // …but an asset that names a *different* arch is still rejected.
        assert!(
            pick_asset(
                &["tool-linux-arm64.tar.gz", "notes.txt"],
                "linux",
                "x86_64",
                false
            )
            .is_err()
        );
    }

    #[test]
    fn npm_bin_resolution() {
        assert_eq!(
            resolve_npm_bin(&serde_json::json!("./cli.js"), "foo").unwrap(),
            "./cli.js"
        );
        assert_eq!(
            resolve_npm_bin(
                &serde_json::json!({"aivo-foo": "a.js", "other": "b.js"}),
                "foo"
            )
            .unwrap(),
            "a.js"
        );
        assert_eq!(
            resolve_npm_bin(&serde_json::json!({"only": "x.js"}), "foo").unwrap(),
            "x.js"
        );
        assert!(resolve_npm_bin(&serde_json::json!({"a": "1", "b": "2"}), "foo").is_err());
    }

    #[test]
    fn npm_url_encoding() {
        assert_eq!(
            npm_metadata_url("https://r/", "@scope/pkg"),
            "https://r/@scope%2fpkg"
        );
        assert_eq!(npm_metadata_url("https://r", "pkg"), "https://r/pkg");
    }

    #[test]
    fn shim_text_is_exact() {
        assert_eq!(
            unix_shim(Path::new("/x/cli.js")),
            "#!/bin/sh\nexec node \"/x/cli.js\" \"$@\"\n"
        );
        assert_eq!(
            windows_cmd(Path::new("C:/x/cli.js")),
            "@node \"C:/x/cli.js\" %*\r\n"
        );
    }

    #[test]
    fn html_detection() {
        assert!(looks_like_html(b"<!DOCTYPE html><html>..."));
        assert!(looks_like_html(b"  \n<html>"));
        assert!(!looks_like_html(b"\x7fELF\x02\x01"));
        assert!(!looks_like_html(b"#!/bin/sh"));
    }
}
