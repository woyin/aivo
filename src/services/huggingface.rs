//! HuggingFace direct-run: resolve a model ref, download the GGUF, and
//! spawn a local `llama-server` for it.

use std::collections::VecDeque;
use std::io::Write;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::services::archive::{ArchiveKind, extract_archive, flatten_single_subdir};
use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::system_env;

static SERVER_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();

const DEFAULT_QUANT: &str = "Q4_K_M";
const HF_URL_PREFIX: &str = "https://huggingface.co/";
const HF_SHORT_PREFIX: &str = "hf:";

/// Synthetic key id for the local llama-server takeover, shared by `aivo run`,
/// `aivo serve`, and plugin dispatch so all three build — and recognize — the
/// same loopback key.
pub const HF_LOCAL_KEY_ID: &str = "aivo-hf-local";

static SPAWN_INFO: OnceLock<SpawnInfo> = OnceLock::new();

/// Spawn-time facts about this process's llama-server, recorded so the
/// limits cascade advertises what the server actually runs instead of a
/// snapshot guess (`model_metadata::resolve_limits` consults this).
#[derive(Debug, Clone)]
pub struct SpawnInfo {
    pub port: u16,
    /// The `-c` the server runs with (or the user's explicit override).
    pub context: u64,
    /// True when an mmproj sidecar was loaded — the server takes images.
    pub image_input: bool,
    /// Display model name tools address the server by.
    pub model_name: String,
}

pub fn spawn_info() -> Option<&'static SpawnInfo> {
    SPAWN_INFO.get()
}

fn child_slot() -> &'static Mutex<Option<Child>> {
    SERVER_CHILD.get_or_init(|| Mutex::new(None))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfModelRef {
    pub repo: String,
    pub quant: Option<String>,
    /// Repo-relative path. May contain `/` for nested files (e.g.
    /// `subdir/model.gguf`); the cache flattens this to a single segment.
    pub file: Option<String>,
    /// `None` collapses to `main`.
    pub revision: Option<String>,
    /// `Some` for local `.gguf` paths; `None` for `hf:` / URL refs.
    /// Read via `is_local()`; `ensure_cached` dispatches on it.
    pub(crate) local_source: Option<PathBuf>,
}

impl HfModelRef {
    pub fn display_model_name(&self) -> String {
        self.repo
            .rsplit('/')
            .next()
            .unwrap_or(&self.repo)
            .to_string()
    }

    pub fn is_local(&self) -> bool {
        self.local_source.is_some()
    }
}

pub fn is_huggingface_ref(model: &str) -> bool {
    model.starts_with(HF_URL_PREFIX) || model.starts_with(HF_SHORT_PREFIX)
}

/// True for arguments that should be resolved as a filesystem path
/// rather than an `hf:` / URL ref. Anchored by either a path-like
/// prefix (`/`, `./`, `../`, `~/`, `~`, or `C:\`-style on Windows) or
/// a `.gguf` suffix.
pub fn looks_like_local_gguf_path(s: &str) -> bool {
    if s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("../")
        || s.starts_with("~/")
        || s == "~"
    {
        return true;
    }
    #[cfg(windows)]
    {
        if s.starts_with(".\\") || s.starts_with("..\\") || s.contains(":\\") {
            return true;
        }
    }
    is_gguf_name(s)
}

/// True for any string that should route through the HF/llama-server
/// flow — `hf:` short refs, `https://huggingface.co/...` URLs, and
/// local `.gguf` paths.
pub fn is_hf_or_local_gguf(s: &str) -> bool {
    is_huggingface_ref(s) || looks_like_local_gguf_path(s)
}

pub fn parse_hf_ref(input: &str) -> Result<HfModelRef> {
    // HF-form check wins over `looks_like_local_gguf_path` because a
    // URL like `.../resolve/main/foo.gguf` matches both predicates.
    if let Some(rest) = input.strip_prefix(HF_SHORT_PREFIX) {
        return parse_hf_short(rest, input);
    }
    if input.starts_with(HF_URL_PREFIX) {
        return parse_hf_url(input);
    }
    if looks_like_local_gguf_path(input) {
        return parse_local_path(input);
    }
    parse_hf_url(input)
}

/// Build an HF ref pointing at a user-supplied `.gguf` on disk. The
/// returned `repo` defaults to `local/<filename-stem-minus-quant>`;
/// callers (e.g. `aivo hf pull --as`) can mutate it before `ensure_cached`.
fn parse_local_path(input: &str) -> Result<HfModelRef> {
    let path = system_env::expand_tilde(input);
    if !path.exists() {
        anyhow::bail!("No such file: {}", path.display());
    }
    let basename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("Path has no filename: {}", path.display()))?;
    if !is_gguf_name(basename) {
        anyhow::bail!(
            "Only `.gguf` files can be imported (got `{basename}`). \
             Pi-style consolidated models or safetensors aren't supported."
        );
    }
    let (derived_stem, quant) = split_repo_and_quant(basename);
    let basename = basename.to_string();
    let repo = format!("local/{derived_stem}");
    Ok(HfModelRef {
        repo,
        quant,
        file: Some(basename),
        revision: None,
        local_source: Some(path),
    })
}

fn parse_hf_short(rest: &str, original: &str) -> Result<HfModelRef> {
    if rest.is_empty() {
        anyhow::bail!("`hf:` is missing the <owner>/<repo> part: {original}");
    }

    // Direct file form maps cleanly onto the URL form.
    if rest.contains("/blob/") || rest.contains("/resolve/") {
        return parse_hf_url(&format!("{HF_URL_PREFIX}{rest}"));
    }

    // `:` can appear only as the quant separator (no `/` after).
    let (repo_path, quant) = match rest.rsplit_once(':') {
        Some((repo, q)) if !q.is_empty() && !q.contains('/') => (repo, Some(q.to_string())),
        _ => (rest, None),
    };

    let segments: Vec<&str> = repo_path.split('/').collect();
    if segments.len() != 2 || segments[0].is_empty() || segments[1].is_empty() {
        anyhow::bail!("Expected `hf:<owner>/<repo>[:<quant>]`, got: {original}");
    }

    Ok(HfModelRef {
        repo: repo_path.to_string(),
        quant,
        file: None,
        revision: None,
        local_source: None,
    })
}

/// Accepted forms (all under `https://huggingface.co/`):
/// `<owner>/<repo>` (bare), `<owner>/<repo>/blob|resolve/<branch>/<file>.gguf`.
fn parse_hf_url(url: &str) -> Result<HfModelRef> {
    let rest = url
        .strip_prefix(HF_URL_PREFIX)
        .with_context(|| format!("Not a huggingface.co URL: {url}"))?;
    let trimmed = rest.trim_end_matches('/');
    if trimmed.is_empty() {
        anyhow::bail!("HuggingFace URL is missing the <owner>/<repo> path: {url}");
    }

    let segments: Vec<&str> = trimmed.split('/').collect();
    if segments.len() < 2 {
        anyhow::bail!("HuggingFace URL must include both <owner> and <repo>: {url}");
    }

    let owner = segments[0];
    let repo = segments[1];
    if owner.is_empty() || repo.is_empty() {
        anyhow::bail!("HuggingFace URL has an empty owner or repo segment: {url}");
    }
    let repo_path = format!("{owner}/{repo}");

    if segments.len() == 2 {
        return Ok(HfModelRef {
            repo: repo_path,
            quant: None,
            file: None,
            revision: None,
            local_source: None,
        });
    }

    // Beyond `<owner>/<repo>` we only understand /blob/ and /resolve/ URLs that
    // point at a concrete .gguf file. Anything else (tree/, commits/, …) gets
    // rejected so the user notices instead of getting a bad llama-server arg.
    let kind = segments[2];
    if !matches!(kind, "blob" | "resolve") {
        anyhow::bail!(
            "Unsupported HuggingFace URL path. Expected `<owner>/<repo>` or `.../blob|resolve/<branch>/<file>.gguf`, got: {url}"
        );
    }
    if segments.len() < 5 {
        anyhow::bail!("HuggingFace file URL must include both <revision> and <file>: {url}");
    }
    let revision = segments[3];
    let path_segments = &segments[4..];
    let file_path = path_segments.join("/");
    let basename = path_segments
        .last()
        .copied()
        .filter(|s| !s.is_empty())
        .with_context(|| format!("HuggingFace file URL is missing the filename: {url}"))?;
    if !is_gguf_name(basename) {
        anyhow::bail!(
            "Only GGUF files are supported for direct HuggingFace runs (got `{basename}`). \
             v1 doesn't auto-convert safetensors."
        );
    }

    Ok(HfModelRef {
        repo: repo_path,
        quant: quant_from_filename(basename),
        file: Some(file_path),
        revision: (!revision.is_empty() && revision != "main").then(|| revision.to_string()),
        local_source: None,
    })
}

/// Allocation-free `.gguf`/`.GGUF` suffix check; byte-wise so a non-ASCII
/// tail can't split a char boundary and panic.
fn is_gguf_name(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() >= 5 && b[b.len() - 5..].eq_ignore_ascii_case(b".gguf")
}

/// Splits `Model-Q5_K_M.gguf` → (`Model`, Some(`Q5_K_M`)). Handles both
/// `-` and `.` separators. Returns `(stem, None)` when no quant tag is
/// recognized.
fn split_repo_and_quant(filename: &str) -> (&str, Option<String>) {
    // Strip via `is_gguf_name` so any case mix (`.Gguf`) that passes the gguf
    // gates is also stripped here; the suffix is ASCII, so len-5 is a boundary.
    let stem = if is_gguf_name(filename) {
        &filename[..filename.len() - 5]
    } else {
        filename
    };
    let Some(idx) = stem.rfind(['-', '.']) else {
        return (stem, None);
    };
    let upper = stem[idx + 1..].to_ascii_uppercase();
    if upper.starts_with('Q') || upper.starts_with("IQ") || upper == "F16" || upper == "BF16" {
        (&stem[..idx], Some(upper))
    } else {
        (stem, None)
    }
}

pub fn quant_from_filename(file: &str) -> Option<String> {
    split_repo_and_quant(file).1
}

pub fn detect_binary() -> Option<PathBuf> {
    use crate::services::path_search::{collect_path_dirs, find_in_dirs};

    if let Some(p) = find_in_dirs("llama-server", &collect_path_dirs()) {
        return Some(p);
    }
    find_in_dirs("llama-server", &well_known_install_dirs())
}

fn well_known_install_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(home) = system_env::home_dir() {
        dirs.push(home.join(".aivo").join("bin"));
        dirs.push(opt_install_dir_under(&home));
    }
    #[cfg(unix)]
    {
        dirs.push(PathBuf::from("/opt/homebrew/bin"));
        dirs.push(PathBuf::from("/usr/local/bin"));
    }
    dirs
}

fn opt_install_dir_under(home: &Path) -> PathBuf {
    home.join(".aivo").join("opt").join("llama.cpp")
}

fn opt_install_dir() -> Option<PathBuf> {
    system_env::home_dir().as_deref().map(opt_install_dir_under)
}

/// On macOS prefer brew; otherwise (and as a fallback on mac without brew) offer
/// to download the latest prebuilt llama.cpp release into `~/.aivo/opt/llama.cpp`.
/// Non-interactive callers get a hint that names the exact release asset.
pub async fn ensure_installed() -> Result<PathBuf> {
    if let Some(p) = detect_binary() {
        return Ok(p);
    }

    eprintln!(
        "  {} llama-server is not installed.",
        crate::style::yellow("?")
    );

    use std::io::IsTerminal;
    let interactive = std::io::stdin().is_terminal();

    #[cfg(target_os = "macos")]
    {
        if which_brew().is_some()
            && interactive
            && prompt_yes("  ? Install via `brew install llama.cpp`? [Y/n] ")?
        {
            run_brew_install()?;
            if let Some(p) = detect_binary() {
                return Ok(p);
            }
        }
    }

    if let Some(target) = release_asset_target() {
        let dest = opt_install_dir().context("Cannot resolve home directory")?;
        if interactive {
            let msg = format!(
                "  {} Download latest llama.cpp release for {} into {}? [Y/n] ",
                crate::style::yellow("?"),
                target.label,
                dest.display(),
            );
            if prompt_yes(&msg)? {
                install_from_release(&target, &dest).await?;
                if let Some(p) = detect_binary() {
                    return Ok(p);
                }
                anyhow::bail!(
                    "llama-server downloaded but not found at {}. Please file a bug.",
                    dest.display(),
                );
            }
        }
    }

    let hint = manual_install_hint();
    anyhow::bail!(
        "llama-server is required to run HuggingFace models directly.\n  Install: {hint}"
    );
}

fn prompt_yes(msg: &str) -> Result<bool> {
    eprint!("{msg}");
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(
        input.trim().to_ascii_lowercase().as_str(),
        "" | "y" | "yes"
    ))
}

fn manual_install_hint() -> String {
    #[cfg(target_os = "macos")]
    {
        "brew install llama.cpp".to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Some(target) = release_asset_target() {
            format!(
                "download the `*{}` asset from https://github.com/ggml-org/llama.cpp/releases/latest \
                 and extract it into ~/.aivo/opt/llama.cpp/",
                target.asset_suffix,
            )
        } else {
            "see https://github.com/ggml-org/llama.cpp/releases for prebuilt binaries".to_string()
        }
    }
}

#[cfg(target_os = "macos")]
fn which_brew() -> Option<PathBuf> {
    use crate::services::path_search::{collect_path_dirs, find_in_dirs};
    find_in_dirs("brew", &collect_path_dirs())
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/brew")).filter(|p| p.exists()))
        .or_else(|| Some(PathBuf::from("/usr/local/bin/brew")).filter(|p| p.exists()))
}

#[cfg(target_os = "macos")]
fn run_brew_install() -> Result<()> {
    eprintln!(
        "  {} Installing llama.cpp via Homebrew...",
        crate::style::arrow_symbol()
    );
    let status = Command::new("brew")
        .arg("install")
        .arg("llama.cpp")
        .status()
        .context("Failed to invoke `brew install llama.cpp`")?;
    if !status.success() {
        anyhow::bail!(
            "`brew install llama.cpp` failed (exit {}). Try running it manually.",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ReleaseTarget {
    /// Substring that must terminate a release asset's filename
    /// (e.g. `-bin-ubuntu-x64.tar.gz`). The leading `llama-bXXXX` tag varies.
    asset_suffix: &'static str,
    label: &'static str,
    kind: ArchiveKind,
}

/// Maps the current `(OS, ARCH)` to a llama.cpp prebuilt release asset.
/// Returns `None` on unsupported platforms (e.g. freebsd, 32-bit).
fn release_asset_target() -> Option<ReleaseTarget> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let (asset_suffix, label, kind) = match (os, arch) {
        ("linux", "x86_64") => ("-bin-ubuntu-x64.tar.gz", "linux x64", ArchiveKind::TarGz),
        ("linux", "aarch64") => (
            "-bin-ubuntu-arm64.tar.gz",
            "linux arm64",
            ArchiveKind::TarGz,
        ),
        ("macos", "x86_64") => ("-bin-macos-x64.tar.gz", "macOS x64", ArchiveKind::TarGz),
        ("macos", "aarch64") => ("-bin-macos-arm64.tar.gz", "macOS arm64", ArchiveKind::TarGz),
        ("windows", "x86_64") => ("-bin-win-cpu-x64.zip", "Windows x64", ArchiveKind::Zip),
        ("windows", "aarch64") => ("-bin-win-cpu-arm64.zip", "Windows arm64", ArchiveKind::Zip),
        _ => return None,
    };
    Some(ReleaseTarget {
        asset_suffix,
        label,
        kind,
    })
}

const LLAMA_CPP_RELEASES_API: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest";

async fn install_from_release(target: &ReleaseTarget, dest_dir: &Path) -> Result<()> {
    eprintln!(
        "  {} Fetching latest llama.cpp release...",
        crate::style::arrow_symbol()
    );

    let client = http_utils::router_http_client_with_timeout(30);
    let resp = client
        .get(LLAMA_CPP_RELEASES_API)
        .header("User-Agent", "aivo-cli")
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .context("Failed to query GitHub for the latest llama.cpp release")?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "GitHub release manifest request failed: HTTP {}",
            resp.status()
        );
    }
    let manifest: serde_json::Value = resp
        .json()
        .await
        .context("Failed to parse llama.cpp release manifest as JSON")?;

    let assets = manifest
        .get("assets")
        .and_then(|v| v.as_array())
        .context("Release manifest is missing the `assets` array")?;

    let asset = assets
        .iter()
        .find_map(|a| {
            let name = a.get("name")?.as_str()?;
            let url = a.get("browser_download_url")?.as_str()?;
            let size = a.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
            if name.ends_with(target.asset_suffix) {
                Some((name.to_string(), url.to_string(), size))
            } else {
                None
            }
        })
        .with_context(|| {
            format!(
                "No release asset matched `*{}`. The llama.cpp release format may have changed; \
                 install manually from https://github.com/ggml-org/llama.cpp/releases",
                target.asset_suffix,
            )
        })?;

    let (asset_name, asset_url, asset_size) = asset;
    eprintln!(
        "  {} {} ({})",
        crate::style::arrow_symbol(),
        asset_name,
        human_size(asset_size),
    );

    let tmp_dir = std::env::temp_dir();
    let _ = std::fs::create_dir_all(&tmp_dir);
    let archive_path = tmp_dir.join(format!("aivo-{}-{}", std::process::id(), asset_name));

    let download_result = download_release_archive(&asset_url, &archive_path, asset_size).await;
    if let Err(e) = download_result {
        let _ = std::fs::remove_file(&archive_path);
        return Err(e);
    }

    // Wipe any prior install so a stale binary doesn't shadow the new libs
    // (`$ORIGIN` RPATH means `llama-server` only loads from its own dir).
    if dest_dir.exists() {
        std::fs::remove_dir_all(dest_dir)
            .with_context(|| format!("Failed to clear {}", dest_dir.display()))?;
    }
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("Failed to create {}", dest_dir.display()))?;

    eprintln!(
        "  {} Extracting into {}",
        crate::style::arrow_symbol(),
        dest_dir.display()
    );
    let extract_result = extract_archive(&archive_path, dest_dir, target.kind)
        .and_then(|()| flatten_single_subdir(dest_dir));
    let _ = std::fs::remove_file(&archive_path);
    extract_result?;

    Ok(())
}

async fn download_release_archive(url: &str, dest: &Path, expected_size: u64) -> Result<()> {
    let client = http_utils::router_http_streaming_client(60);
    let resp = client
        .get(url)
        .header("User-Agent", "aivo-cli")
        .send()
        .await
        .with_context(|| format!("GET {url} failed"))?;
    if !resp.status().is_success() {
        anyhow::bail!("Download of {url} returned HTTP {}", resp.status());
    }

    let total = resp.content_length().unwrap_or(expected_size);
    let mut file = tokio::fs::File::create(dest)
        .await
        .with_context(|| format!("Failed to create {}", dest.display()))?;
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_render = Instant::now();
    render_download_progress(0, total);

    while let Some(chunk) = stream.next().await {
        let chunk = chunk
            .with_context(|| format!("Network error downloading {url} (peer reset or stalled)"))?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("Failed to write {}", dest.display()))?;
        downloaded += chunk.len() as u64;
        if last_render.elapsed() >= Duration::from_millis(150) {
            render_download_progress(downloaded, total);
            last_render = Instant::now();
        }
    }
    file.flush().await?;
    drop(file);
    render_download_progress(downloaded, total);
    eprintln!();
    Ok(())
}

fn render_download_progress(downloaded: u64, total: u64) {
    if total > 0 {
        let pct = (downloaded as f64 / total as f64) * 100.0;
        eprint!(
            "\r  {} {}/{} ({:.0}%)   ",
            crate::style::dim("Downloading:"),
            human_size(downloaded),
            human_size(total),
            pct,
        );
    } else {
        eprint!(
            "\r  {} {}   ",
            crate::style::dim("Downloading:"),
            human_size(downloaded),
        );
    }
    let _ = std::io::stderr().flush();
}

fn alloc_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0").context("Failed to allocate a local port")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

fn local_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(timeout)
        .build()
        .unwrap_or_default()
}

async fn check_health(client: &reqwest::Client, port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    client
        .get(&url)
        .send_logged()
        .await
        .is_ok_and(|r| r.status().is_success())
}

pub fn local_openai_base_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/v1")
}

/// Build the synthetic loopback `ApiKey` for an already-spawned llama-server on
/// `port` serving `hf_ref`. The local server is OpenAI Chat Completions-only for
/// every consumer, so claude/gemini are pinned off their native protocols (codex
/// is already OpenAI-native) to skip the wasted first-attempt probe, and the
/// codex/opencode routers are forced into OpenAI mode. Shared by `aivo run` and
/// plugin dispatch so both reach the local server identically.
pub fn local_takeover_key(
    hf_ref: &HfModelRef,
    port: u16,
) -> crate::services::session_store::ApiKey {
    use crate::services::session_store::{ApiKey, OpenAICompatibilityMode};
    let mut k = ApiKey::new_with_protocol(
        HF_LOCAL_KEY_ID.to_string(),
        format!("hf:{}", hf_ref.repo),
        local_openai_base_url(port),
        None,
        "huggingface".to_string(),
    );
    for tool in ["claude", "gemini"] {
        k.protocol_routes
            .entry(tool.to_string())
            .or_default()
            .insert(
                String::new(),
                crate::services::route_cache::PersistedRoute {
                    protocol: "openai".to_string(),
                    path_variant: String::new(),
                },
            );
    }
    k.codex_mode = Some(OpenAICompatibilityMode::Router);
    k.opencode_mode = Some(OpenAICompatibilityMode::Router);
    k
}

/// Cache-first: skips the HF tree API call when the file is already on
/// disk. Separate from [`ensure_ready`] so `aivo hf pull` can populate
/// the cache without spawning anything. For local-path refs, imports
/// from disk instead of downloading.
pub async fn ensure_cached(model: &HfModelRef) -> Result<CachedFile> {
    ensure_cached_refresh(model, false).await
}

/// Like [`ensure_cached`], but `refresh` skips the cached resolve so a prior
/// (stale or gated) pick can't pin re-runs to the same file. The on-disk file
/// is still honored — a fully cached `.gguf` is never re-downloaded.
pub async fn ensure_cached_refresh(model: &HfModelRef, refresh: bool) -> Result<CachedFile> {
    if let Some(src) = &model.local_source {
        return import_into_cache(src, &model.repo);
    }
    if let Some(cached) = lookup_cached(model) {
        return Ok(cached);
    }
    eprintln!(
        "  {} Resolving {} on HuggingFace…",
        crate::style::dim("⟳"),
        crate::style::cyan(&model.repo)
    );
    let resolved = resolve_gguf_file(model, refresh).await?;
    eprintln!(
        "  {} Found {} ({})",
        crate::style::dim("·"),
        resolved.filename,
        human_size(resolved.size_bytes)
    );

    // Cache key uses the *resolved* repo (post mirror-picker) so calling
    // by upstream or by mirror lands in the same cache entry.
    let cache_path = local_cache_path(&resolved.repo, &resolved.revision, &resolved.filename)?;
    let was_cached = matches!(
        tokio::fs::metadata(&cache_path).await,
        Ok(m) if m.len() == resolved.size_bytes
    );

    if was_cached {
        eprintln!(
            "  {} Already cached at {}",
            crate::style::dim("·"),
            crate::style::dim(cache_path.display().to_string())
        );
    } else {
        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create cache dir {}", parent.display()))?;
        }
        download_with_progress(&resolved, &cache_path).await?;
        eprintln!(
            "  {} Downloaded {} ({})",
            crate::style::success_symbol(),
            resolved.filename,
            human_size(resolved.size_bytes)
        );
    }

    // Cache the resolve only now — a failed download (e.g. 401 on a gated
    // repo) must never pin re-runs to the same unreachable file.
    save_cached_metadata(model, &resolved);

    // Fail-open: a missing projector or draft never blocks the main model.
    for sc in &resolved.sidecars {
        if let Err(e) = ensure_sidecar_cached(&resolved, sc).await {
            eprintln!(
                "  {} Skipping companion file {}: {e:#}",
                crate::style::yellow("!"),
                sc.filename
            );
        }
    }

    Ok(CachedFile {
        repo: resolved.repo,
        revision: resolved.revision,
        filename: resolved.filename,
        size_bytes: resolved.size_bytes,
        path: cache_path,
        was_cached,
    })
}

/// Downloads one companion file next to the main model, skipping when the
/// on-disk size already matches (the same idempotence rule as the main
/// model's download).
async fn ensure_sidecar_cached(resolved: &ResolvedGgufFile, sc: &SidecarMeta) -> Result<()> {
    let path = local_cache_path(&resolved.repo, &resolved.revision, &sc.filename)?;
    // Tree API occasionally returns size 0 for LFS pointers.
    let size = if sc.size_bytes > 0 {
        sc.size_bytes
    } else {
        head_content_length(&sc.download_url).await?
    };
    if matches!(tokio::fs::metadata(&path).await, Ok(m) if m.len() == size) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create cache dir {}", parent.display()))?;
    }
    eprintln!(
        "  {} Found companion {} ({})",
        crate::style::dim("·"),
        sc.filename,
        human_size(size)
    );
    let as_resolved = ResolvedGgufFile {
        repo: resolved.repo.clone(),
        revision: resolved.revision.clone(),
        filename: sc.filename.clone(),
        download_url: sc.download_url.clone(),
        size_bytes: size,
        sidecars: Vec::new(),
    };
    download_with_progress(&as_resolved, &path).await?;
    eprintln!(
        "  {} Downloaded {} ({})",
        crate::style::success_symbol(),
        sc.filename,
        human_size(size)
    );
    Ok(())
}

/// The local-source arm of [`ensure_cached`]. Hardlinks the source
/// into the cache under `<repo>/<basename>`; falls back to a copy
/// across filesystems. Refuses to overwrite a different file of the
/// same name (likely a separate model) with a clear `aivo hf rm` hint.
fn import_into_cache(src: &Path, repo: &str) -> Result<CachedFile> {
    let basename = src
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("Path has no filename: {}", src.display()))?
        .to_string();
    let meta =
        std::fs::metadata(src).with_context(|| format!("Cannot stat `{}`", src.display()))?;
    if !meta.is_file() {
        anyhow::bail!("`{}` is not a regular file", src.display());
    }
    let size_bytes = meta.len();

    let dest = local_cache_path(repo, "main", &basename)?;
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create cache dir {}", parent.display()))?;
    }

    let already_there = match std::fs::metadata(&dest) {
        Ok(m) if m.len() == size_bytes => true,
        Ok(m) => {
            anyhow::bail!(
                "A different `{basename}` ({} bytes) already exists at {}. \
                 Remove it first with `aivo hf rm {repo} --all -y`.",
                m.len(),
                dest.display()
            );
        }
        Err(_) => false,
    };

    if !already_there {
        // Same-filesystem hardlink is instant and disk-free. Falls back
        // to a regular copy across filesystems or on filesystems that
        // reject cross-directory links (e.g. some FUSE mounts).
        if std::fs::hard_link(src, &dest).is_err() {
            std::fs::copy(src, &dest).with_context(|| {
                format!("Failed to copy {} → {}", src.display(), dest.display())
            })?;
        }
    }

    Ok(CachedFile {
        repo: repo.to_string(),
        revision: "main".to_string(),
        filename: basename,
        size_bytes,
        path: dest,
        was_cached: already_there,
    })
}

/// Synchronous, silent local-cache probe; the fast path inside
/// `ensure_cached`/`ensure_ready`.
pub fn lookup_cached(model: &HfModelRef) -> Option<CachedFile> {
    let revision = model.revision.as_deref().unwrap_or("main");

    if let Some(file) = &model.file {
        let path = local_cache_path(&model.repo, revision, file).ok()?;
        let meta = std::fs::metadata(&path).ok()?;
        return Some(CachedFile {
            repo: model.repo.clone(),
            revision: revision.to_string(),
            filename: file.clone(),
            size_bytes: meta.len(),
            path,
            was_cached: true,
        });
    }

    // A bare ref can't express a revision, so cached non-main entries
    // (the `@<rev>__` prefixed files) must not satisfy it.
    let repo_dir = cache_root()?.join(model.repo.replace('/', "__"));
    if !repo_dir.is_dir() {
        return None;
    }
    let entries: Vec<_> = std::fs::read_dir(&repo_dir)
        .ok()?
        .flatten()
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            !name.starts_with('@') && is_gguf_name(&name) && !is_sidecar_filename(&name)
        })
        .collect();
    if entries.is_empty() {
        return None;
    }

    let requested = model.quant.as_deref().unwrap_or(DEFAULT_QUANT);
    let explicit = model.quant.is_some();
    let entries_as_tree: Vec<TreeEntry> = entries
        .iter()
        .map(|e| TreeEntry {
            path: e.file_name().to_string_lossy().into_owned(),
            size: e.metadata().map(|m| m.len()).unwrap_or(0),
            kind: "file".to_string(),
        })
        .collect();
    let refs: Vec<&TreeEntry> = entries_as_tree.iter().collect();
    let matched = find_matching_gguf(&refs, &requested.to_ascii_uppercase(), explicit)?;

    let path = repo_dir.join(&matched.path);
    let meta = std::fs::metadata(&path).ok()?;
    Some(CachedFile {
        repo: model.repo.clone(),
        revision: "main".to_string(),
        filename: matched.path.clone(),
        size_bytes: meta.len(),
        path,
        was_cached: true,
    })
}

pub struct CachedFile {
    /// Resolved repo (post mirror-picker; may differ from input).
    pub repo: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    pub path: PathBuf,
    pub was_cached: bool,
}

impl CachedFile {
    /// `hf:<repo>[:<quant>]` for the common case; falls back to the
    /// full URL when revision or nested path would make the short
    /// form re-resolve to a different file.
    pub fn launch_ref(&self) -> String {
        let nested = self.filename.contains('/');
        let non_main = self.revision != "main" && !self.revision.is_empty();
        if non_main || nested {
            return format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repo, self.revision, self.filename
            );
        }
        match quant_from_filename(&self.filename) {
            Some(q) => format!("hf:{}:{q}", self.repo),
            None => format!("hf:{}", self.repo),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct GgufMeta {
    architecture: Option<String>,
    /// `{arch}.context_length` — the model's training context.
    context_length: Option<u64>,
}

/// Reads `general.architecture` and `{arch}.context_length` from a GGUF
/// v2/v3 header. Best-effort: any parse failure returns whatever was
/// captured so far — the probe must never block a legitimate model.
fn read_gguf_meta(path: &Path) -> GgufMeta {
    let Ok(f) = std::fs::File::open(path) else {
        return GgufMeta::default();
    };
    let mut r = std::io::BufReader::new(f);
    let mut budget: usize = 2 * 1024 * 1024;
    read_gguf_meta_from(&mut r, &mut budget)
}

fn read_gguf_meta_from<R: std::io::Read>(r: &mut R, budget: &mut usize) -> GgufMeta {
    let mut meta = GgufMeta::default();
    let _ = scan_gguf_header(r, budget, &mut meta);
    meta
}

fn scan_gguf_header<R: std::io::Read>(
    r: &mut R,
    budget: &mut usize,
    meta: &mut GgufMeta,
) -> Option<()> {
    fn take<R: std::io::Read>(r: &mut R, buf: &mut [u8], budget: &mut usize) -> Option<()> {
        if buf.len() > *budget {
            return None;
        }
        r.read_exact(buf).ok()?;
        *budget -= buf.len();
        Some(())
    }
    fn u32_le<R: std::io::Read>(r: &mut R, budget: &mut usize) -> Option<u32> {
        let mut b = [0u8; 4];
        take(r, &mut b, budget)?;
        Some(u32::from_le_bytes(b))
    }
    fn u64_le<R: std::io::Read>(r: &mut R, budget: &mut usize) -> Option<u64> {
        let mut b = [0u8; 8];
        take(r, &mut b, budget)?;
        Some(u64::from_le_bytes(b))
    }
    fn gguf_string<R: std::io::Read>(r: &mut R, budget: &mut usize) -> Option<String> {
        let len = u64_le(r, budget)? as usize;
        if len > 64 * 1024 {
            return None;
        }
        let mut buf = vec![0u8; len];
        take(r, &mut buf, budget)?;
        String::from_utf8(buf).ok()
    }
    fn skip_value<R: std::io::Read>(r: &mut R, ty: u32, budget: &mut usize) -> Option<()> {
        let scalar = match ty {
            0 | 1 | 7 => Some(1usize),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        if let Some(n) = scalar {
            let mut tmp = vec![0u8; n];
            return take(r, &mut tmp, budget);
        }
        if ty == 8 {
            gguf_string(r, budget)?;
            return Some(());
        }
        if ty == 9 {
            let elem_ty = u32_le(r, budget)?;
            let len = u64_le(r, budget)? as usize;
            if len > 1 << 24 {
                return None;
            }
            for _ in 0..len {
                skip_value(r, elem_ty, budget)?;
            }
            return Some(());
        }
        None
    }

    let mut magic = [0u8; 4];
    take(r, &mut magic, budget)?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = u32_le(r, budget)?;
    if version < 2 {
        return None;
    }
    let _tensor_count = u64_le(r, budget)?;
    let kv_count = u64_le(r, budget)?;

    for _ in 0..kv_count.min(4096) {
        let key = gguf_string(r, budget)?;
        let ty = u32_le(r, budget)?;
        if key == "general.architecture" && ty == 8 {
            meta.architecture = Some(gguf_string(r, budget)?);
        } else if is_context_length_key(&key, meta.architecture.as_deref()) {
            // Read aligned, then validate — a bogus value must not
            // desynchronize the stream.
            let value = match ty {
                4 => Some(u64::from(u32_le(r, budget)?)),
                5 => {
                    let v = u32_le(r, budget)? as i32;
                    (v >= 0).then_some(v as u64)
                }
                10 => Some(u64_le(r, budget)?),
                11 => {
                    let v = u64_le(r, budget)? as i64;
                    (v >= 0).then_some(v as u64)
                }
                _ => {
                    skip_value(r, ty, budget)?;
                    None
                }
            };
            if let Some(v) = value
                && v > 0
            {
                meta.context_length = Some(v);
            }
        } else {
            skip_value(r, ty, budget)?;
        }
        if meta.architecture.is_some() && meta.context_length.is_some() {
            return Some(());
        }
    }
    Some(())
}

/// `general.architecture` is the first kv llama.cpp writes, so the exact
/// `{arch}.context_length` match almost always applies; the suffix match
/// covers headers where the arch key hasn't been seen yet.
fn is_context_length_key(key: &str, arch: Option<&str>) -> bool {
    match arch {
        Some(a) => {
            key.len() == a.len() + ".context_length".len()
                && key.starts_with(a)
                && key.ends_with(".context_length")
        }
        None => key.ends_with(".context_length"),
    }
}

/// Returns a short user-facing label when `arch` is a llama.cpp architecture
/// that can't drive `/v1/chat/completions` (encoder-only families and
/// classifier heads). `None` means the arch is either generative or unknown
/// — we let the unknown case through rather than guess wrong.
fn non_chat_arch_label(arch: &str) -> Option<&'static str> {
    match arch {
        "bert" | "roberta" | "xlm-roberta" => Some("BERT/RoBERTa encoder"),
        "nomic-bert" | "nomic-bert-moe" => Some("Nomic-BERT encoder"),
        "jina-bert-v2" => Some("Jina-BERT encoder"),
        "t5encoder" => Some("T5 encoder-only"),
        _ => None,
    }
}

fn ensure_arch_is_chat_capable(meta: &GgufMeta) -> Result<()> {
    let Some(arch) = meta.architecture.as_deref() else {
        return Ok(());
    };
    let Some(label) = non_chat_arch_label(arch) else {
        return Ok(());
    };
    anyhow::bail!(
        "This GGUF declares architecture `{arch}` ({label}). Encoder-only models don't \
         compute next-token logits, so llama-server's /v1/chat/completions can't generate \
         from them — chat requests fail with HTTP 500 `the current context does not logits \
         computation. skipping`. aivo only routes chat traffic to llama-server, so this \
         model can't be used here.\n  \
         Pick a generative GGUF instead (HuggingFace `text-generation` — Llama, Qwen, \
         Mistral, Gemma, …)."
    );
}

pub async fn ensure_ready(model: &HfModelRef) -> Result<u16> {
    if let Some(port) = SERVER_PORT.get().copied() {
        return Ok(port);
    }

    let bin = ensure_installed().await?;
    // Cache hit suppresses the "Starting / ready" status pair to keep
    // one-shot `aivo -p hello -m hf:…` output uncluttered.
    let cache_hit = lookup_cached(model).is_some();
    let cached = ensure_cached(model).await?;
    let cache_path = cached.path;
    let meta = read_gguf_meta(&cache_path);
    ensure_arch_is_chat_capable(&meta)?;

    let user_args = user_llama_args()?;
    let (user_owns_ctx, user_ctx) = user_ctx_directive(&user_args);
    let (ctx_flag, advertised_ctx) =
        resolve_ctx(user_owns_ctx, user_ctx, env_ctx(), meta.context_length);

    // Sidecars only pair with main-revision files — a pinned revision's
    // projector/draft may not match what's on disk for `main`.
    let sidecars = if cached.revision == "main" {
        discover_sidecars(&cache_path)
    } else {
        Sidecars::default()
    };
    let mmproj = (!env_disabled("AIVO_LLAMA_MMPROJ"))
        .then_some(sidecars.mmproj)
        .flatten();
    let mut draft = (!env_disabled("AIVO_LLAMA_DRAFT"))
        .then_some(sidecars.mtp_draft)
        .flatten();

    // Some GGUFs ship templates that use filters (e.g. `tojson`) that
    // llama.cpp's embedded minijinja can't parse; the retry overrides with
    // chatml — first-class in llama.cpp and tool-capable — instead of
    // `--no-jinja` (which kills tool routing). Draft flags fail open the
    // same way so a llama.cpp predating `--spec-type` keeps working.
    let mut chatml = false;
    let port = loop {
        let args = build_spawn_args(
            ctx_flag,
            mmproj.as_deref(),
            draft.as_deref(),
            chatml,
            &user_args,
        );
        match try_spawn_and_warmup(&bin, &cache_path, cache_hit, &args, ctx_flag).await? {
            WarmupOutcome::Ready(p) => break p,
            WarmupOutcome::JinjaFailed { stderr_tail } => {
                if chatml {
                    stop_if_we_started();
                    anyhow::bail!(
                        "llama-server failed even with --chat-template chatml:\n--- last stderr ---\n{stderr_tail}"
                    )
                }
                eprintln!(
                    "  {} Model's embedded jinja chat template failed to parse; \
                     retrying with --chat-template chatml",
                    crate::style::yellow("!"),
                );
                stop_if_we_started();
                chatml = true;
            }
            WarmupOutcome::DraftRejected { stderr_tail } => {
                if draft.is_none() {
                    stop_if_we_started();
                    anyhow::bail!(
                        "llama-server exited during warmup:\n--- last stderr ---\n{stderr_tail}"
                    )
                }
                eprintln!(
                    "  {} This llama-server doesn't support MTP draft flags; \
                     retrying without speculative decoding (update llama.cpp to enable it)",
                    crate::style::yellow("!"),
                );
                stop_if_we_started();
                draft = None;
            }
        }
    };
    let _ = SPAWN_INFO.set(SpawnInfo {
        port,
        context: advertised_ctx,
        image_input: mmproj.is_some(),
        model_name: model.display_model_name(),
    });
    let _ = SERVER_PORT.set(port);
    Ok(port)
}

/// Assembly order: aivo's flags first, user passthrough last — llama-server
/// parses last-one-wins, so `AIVO_LLAMA_ARGS` can override anything aivo
/// chose. The chatml rescue template stays after user args: it only runs
/// once the previous template (the user's included) failed to parse.
fn build_spawn_args(
    ctx_flag: Option<u64>,
    mmproj: Option<&Path>,
    draft: Option<&Path>,
    chatml: bool,
    user_args: &[String],
) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    if let Some(n) = ctx_flag {
        args.push("-c".into());
        args.push(n.to_string());
    }
    if let Some(p) = mmproj {
        args.push("--mmproj".into());
        args.push(p.to_string_lossy().into_owned());
    }
    if let Some(p) = draft {
        args.push("--model-draft".into());
        args.push(p.to_string_lossy().into_owned());
        args.push("--spec-type".into());
        args.push("draft-mtp".into());
        args.push("--spec-draft-n-max".into());
        // The MTP sweet spot measured on Apple Silicon; llama.cpp adapts
        // downward when drafts miss.
        args.push("3".into());
    }
    // `--jinja` keeps `tools:` routing alive — llama-server returns 500
    // ("tools param requires --jinja") for any tool request without it.
    args.push("--jinja".into());
    args.extend(user_args.iter().cloned());
    if chatml {
        args.push("--chat-template".into());
        args.push("chatml".into());
    }
    args
}

enum WarmupOutcome {
    Ready(u16),
    JinjaFailed { stderr_tail: String },
    DraftRejected { stderr_tail: String },
}

fn ngl_for_spawn() -> Option<u32> {
    resolved_ngl(
        std::env::var("AIVO_GPU").ok().as_deref(),
        std::env::var("AIVO_LLAMA_NGL").ok().as_deref(),
        cfg!(target_os = "macos"),
    )
}

/// `-ngl 99` means "offload as many layers as fit"; llama.cpp clamps.
/// Harmless on CPU-only builds (prints one warning, then proceeds).
fn resolved_ngl(gpu_env: Option<&str>, ngl_env: Option<&str>, default_on: bool) -> Option<u32> {
    if let Some(v) = ngl_env
        && let Ok(n) = v.trim().parse::<u32>()
    {
        return Some(n);
    }
    if matches!(gpu_env, Some(v) if v.trim().eq_ignore_ascii_case("cpu")) {
        return None;
    }
    default_on.then_some(99)
}

/// Hard cap on the default `-c` — bounds KV-cache memory; the blog-class
/// coding-agent workload is comfortable here and `AIVO_LLAMA_CTX` lifts it.
const CTX_CAP: u64 = 65_536;
/// Default when the GGUF header doesn't reveal the training context.
const CTX_UNKNOWN_DEFAULT: u64 = 32_768;

/// Decides the server context. Returns `(flag aivo passes, context
/// advertised to tools)` — the two must agree or tools overrun the server.
/// Precedence: user's own `-c`/`--ctx-size` (aivo passes nothing) →
/// `AIVO_LLAMA_CTX` → `min(training ctx, 65536)` → 32768.
fn resolve_ctx(
    user_owns: bool,
    user_value: Option<u64>,
    env_value: Option<u64>,
    training: Option<u64>,
) -> (Option<u64>, u64) {
    if user_owns {
        // `-c 0` tells llama-server to use the model's training context.
        let advertised = match user_value {
            Some(n) if n > 0 => n,
            _ => training.unwrap_or(CTX_UNKNOWN_DEFAULT),
        };
        return (None, advertised);
    }
    if let Some(n) = env_value {
        return (Some(n), n);
    }
    let n = training
        .map(|t| t.min(CTX_CAP))
        .unwrap_or(CTX_UNKNOWN_DEFAULT);
    (Some(n), n)
}

fn env_ctx() -> Option<u64> {
    let raw = std::env::var("AIVO_LLAMA_CTX").ok()?;
    match raw.trim().parse::<u64>() {
        Ok(n) if n > 0 => Some(n),
        _ => {
            eprintln!(
                "  {} Ignoring AIVO_LLAMA_CTX=`{raw}` (expected a positive integer)",
                crate::style::yellow("!"),
            );
            None
        }
    }
}

/// `AIVO_LLAMA_MMPROJ=off` / `AIVO_LLAMA_DRAFT=off` style opt-outs.
fn env_disabled(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "off" | "0" | "false" | "no"
        )
    })
}

/// Flags aivo owns outright; user values for these are dropped with a
/// warning since overriding them would detach aivo from its own server.
const OWNED_LLAMA_FLAGS: &[&str] = &["-m", "--model", "--port", "--host"];

fn user_llama_args() -> Result<Vec<String>> {
    let raw = std::env::var("AIVO_LLAMA_ARGS").unwrap_or_default();
    let (args, stripped) = parse_user_llama_args(&raw)?;
    for flag in stripped {
        eprintln!(
            "  {} Ignoring `{flag}` from AIVO_LLAMA_ARGS — aivo owns the model path and server binding",
            crate::style::yellow("!"),
        );
    }
    Ok(args)
}

/// Shell-style split of `AIVO_LLAMA_ARGS` → (kept tokens, stripped owned
/// flags). `~/` paths are expanded; quoting follows POSIX rules (on Windows,
/// quote backslash paths or use forward slashes).
fn parse_user_llama_args(raw: &str) -> Result<(Vec<String>, Vec<String>)> {
    if raw.trim().is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let tokens = shlex::split(raw).ok_or_else(|| {
        anyhow::anyhow!("AIVO_LLAMA_ARGS could not be parsed (unbalanced quote?): {raw}")
    })?;
    let mut kept = Vec::with_capacity(tokens.len());
    let mut stripped = Vec::new();
    let mut skip_next = false;
    for tok in tokens {
        if skip_next {
            skip_next = false;
            continue;
        }
        let flag = tok.split('=').next().unwrap_or(&tok);
        if OWNED_LLAMA_FLAGS.contains(&flag) {
            skip_next = !tok.contains('=');
            stripped.push(flag.to_string());
            continue;
        }
        if tok == "~" || tok.starts_with("~/") {
            kept.push(
                system_env::expand_tilde(&tok)
                    .to_string_lossy()
                    .into_owned(),
            );
        } else {
            kept.push(tok);
        }
    }
    Ok((kept, stripped))
}

/// Whether user args contain `-c`/`--ctx-size`, and its value when it
/// parses. Doesn't consume the value token — it stays in the passthrough.
fn user_ctx_directive(args: &[String]) -> (bool, Option<u64>) {
    let owns = args.iter().any(|a| {
        a == "-c" || a == "--ctx-size" || a.starts_with("--ctx-size=") || a.starts_with("-c=")
    });
    let value = flag_value(args, &["-c", "--ctx-size"]).and_then(|v| v.parse::<u64>().ok());
    (owns, value)
}

#[derive(Debug, Default, Clone)]
struct Sidecars {
    mmproj: Option<PathBuf>,
    mtp_draft: Option<PathBuf>,
}

/// Last path segment with the cache's `__` flattening and `@rev__` prefix
/// stripped, so tree paths and on-disk names classify identically.
fn sidecar_base(name: &str) -> &str {
    let s = name.rsplit('/').next().unwrap_or(name);
    s.rsplit("__").next().unwrap_or(s)
}

fn is_mmproj_name(name: &str) -> bool {
    let b = sidecar_base(name);
    is_gguf_name(b) && b.len() >= 6 && b.as_bytes()[..6].eq_ignore_ascii_case(b"mmproj")
}

/// Unsloth's MTP draft convention: `<model>-<quant>-MTP.gguf`.
fn is_mtp_draft_name(name: &str) -> bool {
    let b = sidecar_base(name);
    if !is_gguf_name(b) {
        return false;
    }
    let stem = &b[..b.len() - 5];
    stem.len() >= 4 && stem.as_bytes()[stem.len() - 4..].eq_ignore_ascii_case(b"-mtp")
}

/// True for files that ride along with a main model rather than being one
/// (multimodal projectors, MTP draft models).
pub fn is_sidecar_filename(name: &str) -> bool {
    is_mmproj_name(name) || is_mtp_draft_name(name)
}

/// Short display label for `aivo hf list`.
pub fn sidecar_label(name: &str) -> Option<&'static str> {
    if is_mmproj_name(name) {
        Some("mmproj")
    } else if is_mtp_draft_name(name) {
        Some("draft")
    } else {
        None
    }
}

/// Projector precision preference; quality first, llama.cpp loads all three.
const MMPROJ_PRECISION_ORDER: &[&str] = &["BF16", "F16", "F32"];

fn pick_mmproj_idx(candidates: &[&str]) -> Option<usize> {
    for tag in MMPROJ_PRECISION_ORDER {
        if let Some(i) = candidates
            .iter()
            .position(|c| c.to_ascii_uppercase().contains(tag))
        {
            return Some(i);
        }
    }
    (!candidates.is_empty()).then_some(0)
}

/// MTP drafts must pair with their exact base model; with several
/// candidates there's no safe pick, so only a lone file qualifies.
fn pick_mtp_idx(candidates: &[&str]) -> Option<usize> {
    (candidates.len() == 1).then_some(0)
}

/// Sidecars living next to the cached main model. Disk is the source of
/// truth at spawn time: files appear here when a tree resolve (or
/// `aivo hf pull`) downloaded them, and pre-feature caches simply have none.
fn discover_sidecars(main_path: &Path) -> Sidecars {
    let Some(dir) = main_path.parent() else {
        return Sidecars::default();
    };
    let main_name = main_path.file_name().map(|n| n.to_os_string());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Sidecars::default();
    };
    let names: Vec<String> = entries
        .flatten()
        .filter(|e| Some(e.file_name()) != main_name)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        // Only main-revision files — `@rev__` entries belong to a pin.
        .filter(|n| !n.starts_with('@') && is_gguf_name(n))
        .collect();
    let mmproj: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| is_mmproj_name(n))
        .collect();
    let drafts: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| is_mtp_draft_name(n))
        .collect();
    Sidecars {
        mmproj: pick_mmproj_idx(&mmproj).map(|i| dir.join(mmproj[i])),
        mtp_draft: pick_mtp_idx(&drafts).map(|i| dir.join(drafts[i])),
    }
}

async fn try_spawn_and_warmup(
    bin: &Path,
    cache_path: &Path,
    cache_hit: bool,
    extra_args: &[String],
    ctx_for_hint: Option<u64>,
) -> Result<WarmupOutcome> {
    let port = alloc_free_port()?;
    let ngl = ngl_for_spawn();
    if !cache_hit {
        let mut notes: Vec<String> = Vec::new();
        if let Some(n) = ngl {
            notes.push(format!("GPU offload: {n} layers"));
        }
        if let Some(c) = flag_value(extra_args, &["-c", "--ctx-size"]) {
            notes.push(format!("ctx {c}"));
        }
        if extra_args.iter().any(|a| a == "--mmproj") {
            notes.push("vision".to_string());
        }
        if extra_args.iter().any(|a| a == "--model-draft") {
            notes.push("MTP draft".to_string());
        }
        let suffix = if notes.is_empty() {
            String::new()
        } else {
            format!(" ({})", notes.join(", "))
        };
        eprintln!(
            "  {} Starting llama-server on port {}{}",
            crate::style::dim("⟳"),
            port,
            suffix,
        );
    }

    let mut cmd = Command::new(bin);
    cmd.arg("-m")
        .arg(cache_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1");
    if let Some(n) = ngl {
        cmd.arg("--n-gpu-layers").arg(n.to_string());
    }
    for a in extra_args {
        cmd.arg(a);
    }
    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to spawn llama-server at {}", bin.display()))?;

    // Drain stderr into a bounded ring so the tail survives until the
    // warmup loop decides whether to surface it.
    let stderr = child.stderr.take().expect("stderr was piped");
    let drain = start_stderr_drain(stderr);

    // Per-process ownership: each aivo kills its own child unconditionally
    // on graceful exit. A cross-process refcount would leak servers when
    // a concurrent aivo is still running.
    if let Ok(mut slot) = child_slot().lock() {
        *slot = Some(child);
    }

    // Animated spinner with elapsed-time label. llama-server's own stdout
    // is still nulled (verbose progress), but stderr is now mirrored on
    // failure paths so the user can see real errors instead of timing out.
    let label = Arc::new(Mutex::new(spinner_label(0)));
    let (spinning, handle) = crate::style::start_spinner_with_label(label.clone());

    let started = Instant::now();
    let outcome = wait_until_healthy(port, label, started, &drain).await;

    crate::style::stop_spinner(&spinning);
    let _ = handle.await;

    match outcome {
        WaitOutcome::Healthy => {
            if !cache_hit || started.elapsed().as_secs() >= 3 {
                eprintln!(
                    "  {} llama-server ready ({}s)",
                    crate::style::success_symbol(),
                    started.elapsed().as_secs()
                );
            }
            Ok(WarmupOutcome::Ready(port))
        }
        WaitOutcome::ChildExited => {
            let tail = drain.snapshot();
            if is_jinja_template_error(&tail) {
                Ok(WarmupOutcome::JinjaFailed { stderr_tail: tail })
            } else if extra_args.iter().any(|a| a == "--model-draft") && is_draft_flag_error(&tail)
            {
                Ok(WarmupOutcome::DraftRejected { stderr_tail: tail })
            } else {
                stop_if_we_started();
                anyhow::bail!(
                    "llama-server exited during warmup on port {port}.\n--- last stderr ---\n{tail}{}",
                    ctx_hint(ctx_for_hint)
                )
            }
        }
        WaitOutcome::Timeout => {
            let tail = drain.snapshot();
            stop_if_we_started();
            anyhow::bail!(
                "llama-server did not become ready within 10 minutes on port {port}.\n\
                 --- last stderr ---\n{tail}{}",
                ctx_hint(ctx_for_hint)
            )
        }
    }
}

/// First value following any of `flags` (`-c 65536` or `--ctx-size=65536`).
fn flag_value<'a>(args: &'a [String], flags: &[&str]) -> Option<&'a str> {
    let mut hit = None;
    let mut iter = args.iter().peekable();
    while let Some(tok) = iter.next() {
        match tok.split_once('=') {
            Some((f, v)) if flags.contains(&f) => hit = Some(v),
            _ if flags.contains(&tok.as_str()) => {
                if let Some(v) = iter.peek() {
                    hit = Some(v.as_str());
                }
            }
            _ => {}
        }
    }
    hit
}

fn ctx_hint(ctx_flag: Option<u64>) -> String {
    match ctx_flag {
        Some(n) => format!(
            "\nThe server was started with -c {n}; if the failure is memory-related, \
             retry with a smaller context, e.g. AIVO_LLAMA_CTX=16384."
        ),
        None => String::new(),
    }
}

/// True when the stderr tail is an argument-parsing rejection of the MTP
/// draft flags (llama.cpp predating `--spec-type`), as opposed to a real
/// load failure with the draft model itself.
fn is_draft_flag_error(stderr_tail: &str) -> bool {
    let mentions_flag = stderr_tail.contains("--spec-type")
        || stderr_tail.contains("--spec-draft-n-max")
        || stderr_tail.contains("--model-draft");
    let arg_error = stderr_tail.contains("invalid argument")
        || stderr_tail.contains("unknown argument")
        || stderr_tail.contains("unrecognized argument")
        || stderr_tail.contains("error while handling argument");
    mentions_flag && arg_error
}

enum WaitOutcome {
    Healthy,
    ChildExited,
    Timeout,
}

/// 10-minute health-poll budget accommodates first-run model loads.
/// Returns early if the stderr drain reports the child has exited.
async fn wait_until_healthy(
    port: u16,
    label: Arc<Mutex<String>>,
    started: Instant,
    drain: &StderrDrain,
) -> WaitOutcome {
    let client = local_client(Duration::from_secs(2));
    for _ in 0..600 {
        if check_health(&client, port).await {
            return WaitOutcome::Healthy;
        }
        if drain.exited() {
            return WaitOutcome::ChildExited;
        }
        if let Ok(mut s) = label.lock() {
            *s = spinner_label(started.elapsed().as_secs());
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    WaitOutcome::Timeout
}

/// Captures the last `STDERR_TAIL_LINES` lines from llama-server's stderr
/// and signals when the stream ends (which on llama-server coincides with
/// process exit).
struct StderrDrain {
    tail: Arc<Mutex<VecDeque<String>>>,
    exited: Arc<AtomicBool>,
}

const STDERR_TAIL_LINES: usize = 80;

impl StderrDrain {
    fn exited(&self) -> bool {
        self.exited.load(Ordering::Relaxed)
    }

    fn snapshot(&self) -> String {
        match self.tail.lock() {
            Ok(t) => t.iter().cloned().collect::<Vec<_>>().join("\n"),
            Err(_) => String::new(),
        }
    }
}

fn start_stderr_drain(stderr: ChildStderr) -> StderrDrain {
    let tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
    let exited = Arc::new(AtomicBool::new(false));
    let tail_writer = tail.clone();
    let exited_writer = exited.clone();
    std::thread::spawn(move || {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(mut t) = tail_writer.lock() {
                if t.len() == STDERR_TAIL_LINES {
                    t.pop_front();
                }
                t.push_back(line);
            }
        }
        exited_writer.store(true, Ordering::Relaxed);
    });
    StderrDrain { tail, exited }
}

fn is_jinja_template_error(stderr_tail: &str) -> bool {
    stderr_tail.contains("chat template parsing error")
        || (stderr_tail.contains("--no-jinja") && stderr_tail.contains("template"))
}

fn spinner_label(elapsed_secs: u64) -> String {
    format!(" Warming up llama-server… ({elapsed_secs}s)")
}

/// `repo` is the post-mirror-picker repo, not the user's input.
struct ResolvedGgufFile {
    repo: String,
    revision: String,
    filename: String,
    download_url: String,
    size_bytes: u64,
    /// Companion files (mmproj projector, MTP draft) found in the same
    /// tree; downloaded fail-open after the main model.
    sidecars: Vec<SidecarMeta>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SidecarMeta {
    filename: String,
    download_url: String,
    size_bytes: u64,
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(rename = "type", default)]
    kind: String,
}

async fn resolve_gguf_file(model: &HfModelRef, refresh: bool) -> Result<ResolvedGgufFile> {
    if !refresh && let Some(cached) = load_cached_metadata(model) {
        return Ok(cached);
    }
    // Saved by the caller after a successful download, not here — so a
    // download failure leaves no cached resolve to stick to on retry.
    resolve_gguf_file_uncached(model).await
}

async fn resolve_gguf_file_uncached(model: &HfModelRef) -> Result<ResolvedGgufFile> {
    let mut current = model.clone();
    loop {
        let revision = current.revision.as_deref().unwrap_or("main");
        if let Some(file) = &current.file {
            let url = format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                current.repo, revision, file
            );
            let size = head_content_length(&url).await?;
            // Direct-file refs skip the tree API, so no sidecar discovery;
            // previously pulled sidecars still pair up via the disk scan.
            return Ok(ResolvedGgufFile {
                repo: current.repo.clone(),
                revision: revision.to_string(),
                filename: file.clone(),
                download_url: url,
                size_bytes: size,
                sidecars: Vec::new(),
            });
        }

        let quant = current.quant.as_deref().unwrap_or(DEFAULT_QUANT);
        let quant_upper = quant.to_ascii_uppercase();
        let tree_url = format!(
            "https://huggingface.co/api/models/{}/tree/{}",
            current.repo, revision
        );
        let client = http_utils::router_http_client();
        let resp = with_hf_auth(client.get(&tree_url))
            .send_logged()
            .await
            .with_context(|| format!("Failed to query HuggingFace tree API at {tree_url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            if matches!(
                status,
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
            ) {
                anyhow::bail!(
                    "HuggingFace tree API returned HTTP {status} for {} — gated or private repo.\n  · {}",
                    current.repo,
                    gated_repo_hint(),
                );
            }
            anyhow::bail!(
                "HuggingFace tree API returned HTTP {status} for {}",
                current.repo
            );
        }
        let entries: Vec<TreeEntry> = resp
            .json()
            .await
            .context("Failed to decode HuggingFace tree API response")?;

        let gguf: Vec<&TreeEntry> = entries
            .iter()
            .filter(|e| e.kind == "file" && is_gguf_name(&e.path))
            .collect();
        if gguf.is_empty() {
            let basename = current
                .repo
                .rsplit('/')
                .next()
                .unwrap_or(&current.repo)
                .to_string();
            let suggestions = search_gguf_mirrors(&basename).await;
            if let Some(chosen) = prompt_pick_mirror(&current.repo, &suggestions) {
                current.repo = chosen;
                current.file = None;
                eprintln!(
                    "  {} Resolving {} on HuggingFace…",
                    crate::style::dim("⟳"),
                    crate::style::cyan(&current.repo)
                );
                continue;
            }
            anyhow::bail!(
                "Repo `{}` has no GGUF files — it's likely the original safetensors release.\n  {}",
                current.repo,
                format_no_gguf_hint(&basename, &suggestions)
            );
        }
        // Skip `*-of-*.gguf` split parts — loading one in isolation would
        // give incomplete weights, and llama.cpp's split-load isn't wired up.
        let singles: Vec<&TreeEntry> = gguf
            .iter()
            .copied()
            .filter(|e| !e.path.to_ascii_lowercase().contains("-of-"))
            .collect();
        if singles.is_empty() {
            anyhow::bail!(
                "Repo `{}` only contains split-GGUF files (e.g. `*-00001-of-00003.gguf`), which v1 doesn't load yet.",
                current.repo
            );
        }
        // Sidecars never compete for the main slot — `mmproj-BF16.gguf`
        // would otherwise win an explicit `:BF16` quant request.
        let (sidecar_entries, mains): (Vec<&TreeEntry>, Vec<&TreeEntry>) = singles
            .iter()
            .copied()
            .partition(|e| is_sidecar_filename(&e.path));
        let explicit_quant = current.quant.is_some();
        let matched =
            find_matching_gguf(&mains, &quant_upper, explicit_quant).ok_or_else(|| {
                let available: Vec<String> = mains.iter().map(|e| e.path.clone()).collect();
                anyhow::anyhow!(
                    "No GGUF file matching quant `{quant}` in `{}`. Available files: {}",
                    current.repo,
                    available.join(", ")
                )
            })?;
        if let Some(picked) = quant_tag_of(&matched.path)
            && picked != quant_upper
        {
            eprintln!(
                "  {} Using {} (requested {quant} not available in this repo)",
                crate::style::dim("·"),
                picked
            );
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            current.repo, revision, matched.path
        );
        // Tree API occasionally returns size 0 for LFS pointers.
        let size = if matched.size > 0 {
            matched.size
        } else {
            head_content_length(&url).await?
        };
        return Ok(ResolvedGgufFile {
            repo: current.repo.clone(),
            revision: revision.to_string(),
            filename: matched.path.clone(),
            download_url: url,
            size_bytes: size,
            sidecars: pick_tree_sidecars(&sidecar_entries, &current.repo, revision),
        });
    }
}

/// The mmproj (best precision) and MTP draft (only when unambiguous) a
/// fresh tree resolve should bring along with the main model.
fn pick_tree_sidecars(entries: &[&TreeEntry], repo: &str, revision: &str) -> Vec<SidecarMeta> {
    let as_meta = |e: &TreeEntry| SidecarMeta {
        filename: e.path.clone(),
        download_url: format!(
            "https://huggingface.co/{repo}/resolve/{revision}/{}",
            e.path
        ),
        size_bytes: e.size,
    };
    let mmproj: Vec<&str> = entries
        .iter()
        .map(|e| e.path.as_str())
        .filter(|p| is_mmproj_name(p))
        .collect();
    let drafts: Vec<&str> = entries
        .iter()
        .map(|e| e.path.as_str())
        .filter(|p| is_mtp_draft_name(p))
        .collect();
    let mut picked = Vec::new();
    if let Some(i) = pick_mmproj_idx(&mmproj)
        && let Some(e) = entries.iter().find(|e| e.path == mmproj[i])
    {
        picked.push(as_meta(e));
    }
    if let Some(i) = pick_mtp_idx(&drafts)
        && let Some(e) = entries.iter().find(|e| e.path == drafts[i])
    {
        picked.push(as_meta(e));
    }
    picked
}

/// Returns `None` on empty suggestions, non-TTY, or user dismissal.
fn prompt_pick_mirror(original_repo: &str, suggestions: &[String]) -> Option<String> {
    use std::io::IsTerminal;
    if suggestions.is_empty() {
        return None;
    }
    if !std::io::stderr().is_terminal() || !std::io::stdin().is_terminal() {
        return None;
    }

    eprintln!(
        "  {} `{}` has no GGUF files — likely the original safetensors release.",
        crate::style::yellow("!"),
        original_repo
    );
    eprintln!(
        "  {} {}",
        crate::style::dim("·"),
        crate::style::dim("Pick a community GGUF mirror (Esc to cancel):")
    );

    use crate::tui::FuzzySelect;
    match FuzzySelect::new()
        .with_prompt("GGUF mirror")
        .items(suggestions)
        .default(0)
        .interact_opt()
    {
        Ok(Some(idx)) => suggestions.get(idx).cloned(),
        Ok(None) | Err(_) => None,
    }
}

fn format_no_gguf_hint(basename: &str, suggestions: &[String]) -> String {
    if suggestions.is_empty() {
        format!(
            "Search huggingface.co for `{basename}-GGUF` (community converters: bartowski, lmstudio-community, TheBloke)."
        )
    } else {
        let lines: Vec<String> = suggestions
            .iter()
            .map(|s| format!("    aivo claude --model hf:{s}"))
            .collect();
        format!("Try one of:\n{}", lines.join("\n"))
    }
}

/// Ordered "best general-purpose size/quality first".
const QUANT_FALLBACK_ORDER: &[&str] = &[
    "Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q4_0", "Q5_0", "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_S",
    "Q2_K", "IQ4_NL", "IQ4_XS", "IQ3_M", "IQ3_XS", "IQ2_M", "F16", "BF16",
];

/// Explicit user-supplied quant: exact match only.
/// Implicit (default Q4_K_M): walk fallback chain, then accept a single
/// file as last resort.
fn find_matching_gguf<'a>(
    candidates: &[&'a TreeEntry],
    requested_upper: &str,
    explicit: bool,
) -> Option<&'a TreeEntry> {
    if let Some(hit) = find_by_quant_tag(candidates, requested_upper) {
        return Some(hit);
    }
    if !explicit {
        for fallback in QUANT_FALLBACK_ORDER {
            if let Some(hit) = find_by_quant_tag(candidates, fallback) {
                return Some(hit);
            }
        }
        if candidates.len() == 1 {
            return Some(candidates[0]);
        }
    }
    None
}

fn find_by_quant_tag<'a>(candidates: &[&'a TreeEntry], tag_upper: &str) -> Option<&'a TreeEntry> {
    candidates
        .iter()
        .find(|e| {
            let upper = e.path.to_ascii_uppercase();
            upper.contains(&format!("-{tag_upper}.GGUF"))
                || upper.contains(&format!(".{tag_upper}.GGUF"))
        })
        .copied()
}

fn quant_tag_of(filename: &str) -> Option<String> {
    quant_from_filename(filename)
}

/// How many mirror suggestions to surface in the picker (it's type-to-filter,
/// so a handful is plenty without crowding the terminal).
const MIRROR_SUGGESTION_LIMIT: usize = 8;

/// Best-effort GGUF mirror search for the picker. Returns empty on any error so
/// the error path renders even when the network is flaky.
///
/// Uses HuggingFace's default *relevance* ranking, NOT `sort=downloads`: a
/// just-released model's GGUF mirrors all have ~0 downloads, so download-sorting
/// buries the correct-version converts under older same-family repos (e.g. a
/// `gemma-4` query returned only `gemma-3` repos). Trusted converters are then
/// floated to the top of the relevance order.
async fn search_gguf_mirrors(basename: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct SearchHit {
        id: String,
        #[serde(default)]
        tags: Vec<String>,
    }

    let url = format!(
        "https://huggingface.co/api/models?search={}&limit=20",
        urlencode(&format!("{basename} GGUF"))
    );
    let client = http_utils::router_http_client();
    let Ok(resp) = with_hf_auth(client.get(&url)).send_logged().await else {
        return Vec::new();
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let Ok(hits) = resp.json::<Vec<SearchHit>>().await else {
        return Vec::new();
    };
    // HF's `gguf` tag coverage is patchy for older uploads — fall back
    // to a name-substring match.
    let ids: Vec<String> = hits
        .into_iter()
        .filter(|h| {
            h.tags.iter().any(|t| t.eq_ignore_ascii_case("gguf"))
                || h.id.to_ascii_uppercase().contains("GGUF")
        })
        .map(|h| h.id)
        .collect();
    rank_gguf_mirrors(ids)
}

/// Stable-promotes well-known converters ahead of noisier community uploads
/// (preserving relevance order within each group), then truncates. Never
/// excludes anything — only reorders.
fn rank_gguf_mirrors(mut ids: Vec<String>) -> Vec<String> {
    ids.sort_by_key(|id| !is_trusted_gguf_converter(id));
    ids.truncate(MIRROR_SUGGESTION_LIMIT);
    ids
}

/// Owners that reliably publish faithful GGUF conversions. Used only to order
/// the mirror picker.
fn is_trusted_gguf_converter(id: &str) -> bool {
    const TRUSTED: &[&str] = &[
        "unsloth",
        "ggml-org",
        "bartowski",
        "lmstudio-community",
        "google",
        "TheBloke",
    ];
    let owner = id.split('/').next().unwrap_or("");
    TRUSTED.iter().any(|t| owner.eq_ignore_ascii_case(t))
}

/// Minimal percent-encoder for query strings; avoids a percent-encode dep.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '~') {
            out.push(ch);
        } else {
            let mut buf = [0u8; 4];
            for b in ch.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

/// Time-To-Live for cached tree-API/HEAD results. `main` is mutable, so we
/// don't want infinite reuse; 4h matches the user's tolerance for staleness
/// (a re-uploaded GGUF would mismatch `size_bytes` at download time and
/// invalidate naturally via the existing `m.len() == resolved.size_bytes`
/// check in `ensure_cached`).
const METADATA_TTL_SECS: u64 = 4 * 60 * 60;

#[derive(Serialize, Deserialize)]
struct CachedMetadata {
    cached_at_unix: u64,
    repo: String,
    revision: String,
    filename: String,
    download_url: String,
    size_bytes: u64,
    #[serde(default)]
    sidecars: Vec<SidecarMeta>,
}

fn metadata_cache_path(model: &HfModelRef) -> Option<PathBuf> {
    let root = cache_root()?;
    // Key on the *input* ref so a mirror pick made in a prior session is
    // remembered. `_` stands in for `None` to keep the filename unambiguous.
    let key = format!(
        "{}__{}__{}__{}.json",
        model.repo.replace('/', "__"),
        model.revision.as_deref().unwrap_or("main"),
        model
            .file
            .as_deref()
            .map(|f| f.replace('/', "__"))
            .unwrap_or_else(|| "_".to_string()),
        model.quant.as_deref().unwrap_or("_"),
    );
    Some(root.join("_meta").join(key))
}

fn load_cached_metadata(model: &HfModelRef) -> Option<ResolvedGgufFile> {
    if std::env::var("AIVO_HF_NO_META_CACHE").is_ok() {
        return None;
    }
    let path = metadata_cache_path(model)?;
    let data = std::fs::read_to_string(&path).ok()?;
    let entry: CachedMetadata = serde_json::from_str(&data).ok()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    if now.saturating_sub(entry.cached_at_unix) > METADATA_TTL_SECS {
        return None;
    }
    Some(ResolvedGgufFile {
        repo: entry.repo,
        revision: entry.revision,
        filename: entry.filename,
        download_url: entry.download_url,
        size_bytes: entry.size_bytes,
        sidecars: entry.sidecars,
    })
}

fn save_cached_metadata(model: &HfModelRef, resolved: &ResolvedGgufFile) {
    let Some(path) = metadata_cache_path(model) else {
        return;
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let entry = CachedMetadata {
        cached_at_unix: now,
        repo: resolved.repo.clone(),
        revision: resolved.revision.clone(),
        filename: resolved.filename.clone(),
        download_url: resolved.download_url.clone(),
        size_bytes: resolved.size_bytes,
        sidecars: resolved.sidecars.clone(),
    };
    let Ok(json) = serde_json::to_string(&entry) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, json);
}

/// HuggingFace env vars holding a bearer token, in precedence order.
const HF_TOKEN_ENV_VARS: &[&str] = &["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"];

/// Token for gated/private repos. Env vars win; otherwise the CLI token file
/// (`huggingface-cli login` writes `$HF_HOME/token`). `None` when unset.
fn hf_token() -> Option<String> {
    hf_token_from_env(|k| std::env::var(k).ok())
        .or_else(|| read_hf_token_file(hf_token_file_path()))
}

fn hf_token_from_env(lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    HF_TOKEN_ENV_VARS.iter().find_map(|&var| {
        lookup(var)
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

fn hf_token_file_path() -> Option<PathBuf> {
    hf_token_file_path_from(
        std::env::var("HF_TOKEN_PATH").ok(),
        std::env::var("HF_HOME").ok(),
        system_env::home_dir(),
    )
}

/// `$HF_TOKEN_PATH` → `$HF_HOME/token` → `~/.cache/huggingface/token`.
fn hf_token_file_path_from(
    token_path: Option<String>,
    hf_home: Option<String>,
    home: Option<PathBuf>,
) -> Option<PathBuf> {
    if let Some(p) = token_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return Some(PathBuf::from(p));
    }
    if let Some(h) = hf_home.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return Some(Path::new(h).join("token"));
    }
    home.map(|h| h.join(".cache").join("huggingface").join("token"))
}

fn read_hf_token_file(path: Option<PathBuf>) -> Option<String> {
    let raw = std::fs::read_to_string(path?).ok()?;
    let token = raw.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// Attaches `Authorization: Bearer <token>` when a token is configured.
/// reqwest strips this header on the cross-host redirect to HF's CDN, so the
/// signed download URL never receives a redundant bearer.
fn with_hf_auth(req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match hf_token() {
        Some(token) => req.bearer_auth(token),
        None => req,
    }
}

/// Trailing hint for gated/private 401/403s, tailored to whether a token
/// was already sent.
fn gated_repo_hint() -> &'static str {
    if hf_token().is_some() {
        "An HF token was sent but access was denied — accept the model's license on its \
         HuggingFace page (gated repos need a one-time click), or the token lacks access."
    } else {
        "Set a token for gated/private repos — `export HF_TOKEN=hf_…` or `huggingface-cli \
         login` — then accept the model's license on its page."
    }
}

async fn head_content_length(url: &str) -> Result<u64> {
    let client = http_utils::router_http_client();
    let resp = with_hf_auth(client.head(url))
        .send_logged()
        .await
        .with_context(|| format!("HEAD {url} failed"))?;
    if !resp.status().is_success() && !resp.status().is_redirection() {
        anyhow::bail!("HEAD {url} returned HTTP {}", resp.status());
    }
    resp.headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| anyhow::anyhow!("HEAD {url} did not return a Content-Length header"))
}

/// Atomic via `.partial` rename. Interrupted runs resume via HTTP Range
/// against the existing `.partial`; if the server doesn't honor `Range`
/// (returns `200` instead of `206`) we restart from zero.
async fn download_with_progress(file: &ResolvedGgufFile, dest: &Path) -> Result<()> {
    let label_text = format!(" 0% · 0B/{}", human_size(file.size_bytes));
    let label = Arc::new(Mutex::new(label_text));
    let (spinning, handle) = crate::style::start_spinner_with_label(label.clone());
    let started = Instant::now();

    let result = stream_to_file(file, dest, label.clone(), started).await;

    crate::style::stop_spinner(&spinning);
    let _ = handle.await;
    result
}

async fn stream_to_file(
    file: &ResolvedGgufFile,
    dest: &Path,
    label: Arc<Mutex<String>>,
    started: Instant,
) -> Result<()> {
    let tmp = dest.with_extension("partial");

    // Resume if a `.partial` from a prior run is shorter than the expected size.
    // A `.partial` that's already >= total is suspect (size changed upstream,
    // or a prior run finished writing but failed to rename) — start fresh.
    let resume_offset = match tokio::fs::metadata(&tmp).await {
        Ok(meta) if meta.is_file() => {
            let len = meta.len();
            if file.size_bytes > 0 && len >= file.size_bytes {
                let _ = tokio::fs::remove_file(&tmp).await;
                0
            } else {
                len
            }
        }
        _ => 0,
    };

    let client = http_utils::router_http_streaming_client(60);
    let mut req = with_hf_auth(client.get(&file.download_url));
    if resume_offset > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
    }
    let resp = req
        .send_logged()
        .await
        .with_context(|| format!("GET {} failed", file.download_url))?;
    if !resp.status().is_success() {
        let status = resp.status();
        if matches!(
            status,
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) {
            anyhow::bail!(
                "Download of {} returned HTTP {status} — gated or private repo \
                 (e.g. `google/gemma-*`).\n  · {}\n  \
                 · Or pull an ungated community GGUF mirror — a `bartowski/…-GGUF`, \
                 `unsloth/…-GGUF`, or `lmstudio-community/…-GGUF` repo.\n  \
                 · Re-run with `--refresh` to drop the cached pick and re-resolve.",
                file.filename,
                gated_repo_hint(),
            );
        }
        anyhow::bail!("Download of {} returned HTTP {status}", file.filename);
    }

    // A 206 body may only be appended at the offset the server actually starts
    // at — not blindly at the partial's end. Some caching/forward proxies honor
    // `Range` but realign the start *down* to a block boundary, so the response
    // re-sends bytes we already have. Appending that overlap duplicates it and
    // the file ends up larger than the real size (then loads as garbage). Read
    // the response's own `Content-Range` start and truncate the partial back to
    // it before appending. Anything we can't place cleanly (200 OK, a missing
    // header, or a start past our resume point) restarts the whole download.
    let resume_start = if resume_offset > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT
    {
        match content_range_start(resp.headers()) {
            Some(s) if s <= resume_offset => Some(s),
            _ => None,
        }
    } else {
        None
    };
    let starting_offset = resume_start.unwrap_or(0);
    let mut out = if let Some(s) = resume_start {
        // Drop any bytes past the server's range start (normally a no-op, since
        // `s == resume_offset`), then append the body from there.
        let f = tokio::fs::OpenOptions::new()
            .write(true)
            .open(&tmp)
            .await
            .with_context(|| format!("Failed to open {} for resume", tmp.display()))?;
        f.set_len(s)
            .await
            .with_context(|| format!("Failed to truncate {} to {s}", tmp.display()))?;
        drop(f);
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&tmp)
            .await
            .with_context(|| format!("Failed to open {} for append", tmp.display()))?
    } else {
        tokio::fs::File::create(&tmp)
            .await
            .with_context(|| format!("Failed to create {}", tmp.display()))?
    };

    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = starting_offset;
    let mut session_downloaded: u64 = 0;
    let mut last_label = Instant::now();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| {
            format!(
                "Network error during model download after {} (stalled for >60s, or peer reset)",
                human_size(downloaded)
            )
        })?;
        downloaded += chunk.len() as u64;
        session_downloaded += chunk.len() as u64;
        out.write_all(&chunk)
            .await
            .with_context(|| format!("Failed to write {}", tmp.display()))?;
        if last_label.elapsed() >= Duration::from_millis(200) {
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            let bytes_per_sec = session_downloaded as f64 / elapsed;
            let speed_mb_s = bytes_per_sec / (1024.0 * 1024.0);
            let pct = if file.size_bytes > 0 {
                (downloaded as f64 / file.size_bytes as f64 * 100.0) as u32
            } else {
                0
            };
            let eta = if file.size_bytes > downloaded && bytes_per_sec > 0.0 {
                let remaining = (file.size_bytes - downloaded) as f64 / bytes_per_sec;
                format!(" · {}", format_eta(remaining as u64))
            } else {
                String::new()
            };
            if let Ok(mut s) = label.lock() {
                *s = format!(
                    " {pct}% · {}/{} · {speed_mb_s:.1}MB/s{eta}",
                    human_size(downloaded),
                    human_size(file.size_bytes)
                );
            }
            last_label = Instant::now();
        }
    }
    out.flush()
        .await
        .with_context(|| format!("Failed to flush {}", tmp.display()))?;
    drop(out);

    // Integrity gate: the assembled file must be exactly the expected size
    // before it reaches the cache. A short file (interrupted) or an over-long
    // one (a proxy that realigned a resume range and duplicated bytes) is
    // corrupt — a GGUF that loads but emits garbage tokens. Drop the partial so
    // the next run starts clean instead of resuming from corruption.
    if file.size_bytes > 0 && downloaded != file.size_bytes {
        let _ = tokio::fs::remove_file(&tmp).await;
        anyhow::bail!(
            "Downloaded {} is {} but expected {} — the transfer was corrupted \
             (often a flaky network or proxy). Re-run to download it again.",
            file.filename,
            human_size(downloaded),
            human_size(file.size_bytes),
        );
    }

    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("Failed to rename {} → {}", tmp.display(), dest.display()))?;
    Ok(())
}

/// Start byte of a `Content-Range: bytes START-END/TOTAL` response header
/// (e.g. `bytes 1024-4095/4096` → `1024`). `None` if absent or unparseable.
fn content_range_start(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let raw = headers.get(reqwest::header::CONTENT_RANGE)?.to_str().ok()?;
    raw.trim()
        .strip_prefix("bytes")?
        .trim_start()
        .split('-')
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Segments rather than a literal `"a/b/c"` so Windows `PathBuf::join`
/// produces backslash-separated paths instead of mixing `/` and `\`.
const HF_CACHE_SEGMENTS: &[&str] = &[".config", "aivo", "cache", "huggingface"];
const LEGACY_HF_CACHE_SEGMENTS: &[&str] = &[".aivo", "cache", "huggingface"];

/// One-shot migration from the pre-0.23 cache location (`~/.aivo/cache/huggingface`)
/// to the new one under `~/.config/aivo/`. Skips silently if the new directory
/// already exists, so a downgrade-then-upgrade won't clobber recent downloads.
fn migrate_legacy_cache_once() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let Some(home) = system_env::home_dir() else {
            return;
        };
        let old = system_env::join_segments(&home, LEGACY_HF_CACHE_SEGMENTS);
        let new = system_env::join_segments(&home, HF_CACHE_SEGMENTS);
        if !old.is_dir() || new.exists() {
            return;
        }
        if let Some(parent) = new.parent()
            && std::fs::create_dir_all(parent).is_err()
        {
            return;
        }
        if std::fs::rename(&old, &new).is_ok() {
            eprintln!(
                "  {} Moved HuggingFace cache: {} → {}",
                crate::style::success_symbol(),
                old.display(),
                new.display()
            );
            let _ = std::fs::remove_dir(home.join(".aivo").join("cache"));
            let _ = std::fs::remove_dir(home.join(".aivo"));
        }
    });
}

/// On-disk encoding:
/// - `<owner>__<repo>/<flat-file>` for main revision
/// - `<owner>__<repo>/@<rev>__<flat-file>` otherwise
///
/// `<flat-file>` replaces `/` with `__` so the layout stays one level deep.
fn local_cache_path(repo: &str, revision: &str, filename: &str) -> Result<PathBuf> {
    migrate_legacy_cache_once();
    let home = system_env::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory for model cache"))?;
    let sanitized_repo = repo.replace('/', "__");
    let flat = filename.replace('/', "__");
    let on_disk = if revision == "main" || revision.is_empty() {
        flat
    } else {
        format!("@{revision}__{flat}")
    };
    Ok(system_env::join_segments(&home, HF_CACHE_SEGMENTS)
        .join(sanitized_repo)
        .join(on_disk))
}

pub fn cache_root() -> Option<PathBuf> {
    migrate_legacy_cache_once();
    system_env::home_dir().map(|h| system_env::join_segments(&h, HF_CACHE_SEGMENTS))
}

#[derive(Debug, Clone)]
pub struct CachedModel {
    pub repo: String,
    pub revision: String,
    pub filename: String,
    pub size_bytes: u64,
    pub modified: Option<std::time::SystemTime>,
    pub path: PathBuf,
}

impl CachedModel {
    pub fn quant(&self) -> Option<String> {
        quant_from_filename(&self.filename)
    }

    pub fn launch_ref(&self) -> String {
        let nested = self.filename.contains('/');
        let non_main = self.revision != "main" && !self.revision.is_empty();
        if non_main || nested {
            return format!(
                "https://huggingface.co/{}/resolve/{}/{}",
                self.repo, self.revision, self.filename
            );
        }
        match self.quant() {
            Some(q) => format!("hf:{}:{q}", self.repo),
            None => format!("hf:{}", self.repo),
        }
    }
}

pub fn list_cached_models() -> Vec<CachedModel> {
    let Some(root) = cache_root() else {
        return Vec::new();
    };
    let Ok(repo_dirs) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut models = Vec::new();
    for repo_entry in repo_dirs.flatten() {
        let Ok(ft) = repo_entry.file_type() else {
            continue;
        };
        if !ft.is_dir() {
            continue;
        }
        let dir_name = repo_entry.file_name().to_string_lossy().into_owned();
        let Some(repo) = unsanitize_repo(&dir_name) else {
            continue;
        };
        let Ok(files) = std::fs::read_dir(repo_entry.path()) else {
            continue;
        };
        for file_entry in files.flatten() {
            let on_disk = file_entry.file_name().to_string_lossy().into_owned();
            if !is_gguf_name(&on_disk) {
                continue;
            }
            let (revision, logical_filename) = parse_on_disk_filename(&on_disk);
            let Ok(meta) = file_entry.metadata() else {
                continue;
            };
            models.push(CachedModel {
                repo: repo.clone(),
                revision,
                filename: logical_filename,
                size_bytes: meta.len(),
                modified: meta.modified().ok(),
                path: file_entry.path(),
            });
        }
    }
    models.sort_by_key(|m| std::cmp::Reverse(m.modified));
    models
}

/// Sums file sizes across all quants in a repo. Sorted most-recently-used.
pub fn list_cached_repos() -> Vec<CachedRepo> {
    use std::collections::HashMap;
    let mut by_repo: HashMap<String, Vec<CachedModel>> = HashMap::new();
    for m in list_cached_models() {
        by_repo.entry(m.repo.clone()).or_default().push(m);
    }
    let mut repos: Vec<CachedRepo> = by_repo
        .into_iter()
        .map(|(repo, files)| {
            let total_bytes = files.iter().map(|f| f.size_bytes).sum();
            let modified = files.iter().filter_map(|f| f.modified).max();
            // A sidecar must never become the repo's launchable face —
            // its launch_ref would point llama-server at the projector.
            let primary = files
                .iter()
                .filter(|f| !is_sidecar_filename(&f.filename))
                .max_by_key(|f| f.modified)
                .or_else(|| files.iter().max_by_key(|f| f.modified))
                .expect("repo entry implies at least one file")
                .clone();
            CachedRepo {
                repo,
                total_bytes,
                modified,
                primary,
                files,
            }
        })
        .collect();
    repos.sort_by_key(|r| std::cmp::Reverse(r.modified));
    repos
}

#[derive(Debug, Clone)]
pub struct CachedRepo {
    pub repo: String,
    pub total_bytes: u64,
    pub modified: Option<std::time::SystemTime>,
    /// Most-recently-modified file in the repo; drives the picker label.
    pub primary: CachedModel,
    pub files: Vec<CachedModel>,
}

fn unsanitize_repo(dir_name: &str) -> Option<String> {
    // HF repo names may contain `__`, so split on the *first* one only.
    let (owner, repo) = dir_name.split_once("__")?;
    if owner.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{owner}/{repo}"))
}

/// We don't un-flatten `__` back to `/` for nested filenames — a real
/// GGUF name containing literal `__` would be misread and produce a
/// broken URL on re-resolve. Listings show the on-disk name verbatim.
fn parse_on_disk_filename(on_disk: &str) -> (String, String) {
    match on_disk.strip_prefix('@') {
        Some(after_at) => match after_at.split_once("__") {
            Some((rev, rest)) if !rev.is_empty() && !rest.is_empty() => {
                (rev.to_string(), rest.to_string())
            }
            _ => ("main".to_string(), on_disk.to_string()),
        },
        None => ("main".to_string(), on_disk.to_string()),
    }
}

pub fn remove_cached_repo(repo: &str) -> Result<u64> {
    let root = cache_root().ok_or_else(|| anyhow::anyhow!("Cannot resolve cache root"))?;
    let dir = root.join(repo.replace('/', "__"));
    if !dir.exists() {
        anyhow::bail!("No cached files for `{repo}` under {}", root.display());
    }
    // Guard against `repo = "../../etc/passwd"` and the like.
    if !dir.starts_with(&root) {
        anyhow::bail!("Refusing to delete outside cache root: {}", dir.display());
    }
    let freed = dir_size(&dir);
    std::fs::remove_dir_all(&dir).with_context(|| format!("Failed to remove {}", dir.display()))?;
    Ok(freed)
}

pub fn remove_all_cached() -> Result<u64> {
    let Some(root) = cache_root() else {
        return Ok(0);
    };
    if !root.exists() {
        return Ok(0);
    }
    let freed = dir_size(&root);
    std::fs::remove_dir_all(&root)
        .with_context(|| format!("Failed to remove {}", root.display()))?;
    Ok(freed)
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            total += dir_size(&entry.path());
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

pub fn format_size(b: u64) -> String {
    human_size(b)
}

pub fn cached_model_as_short_ref(m: &CachedModel) -> String {
    m.launch_ref()
}

/// Bare `hf:` (or `hf:   `) triggers the cached-models picker.
pub fn is_bare_hf_picker_trigger(model: &str) -> bool {
    model
        .strip_prefix(HF_SHORT_PREFIX)
        .is_some_and(|rest| rest.trim().is_empty())
}

/// Opens a TUI picker over cached HuggingFace models. Returns the chosen
/// model as an `hf:<repo>[:<quant>]` short ref. Returns `None` on Esc,
/// non-TTY, or empty cache (with a helpful message in each case).
pub fn pick_cached_short_ref() -> Option<String> {
    let repos = list_cached_repos();
    if repos.is_empty() {
        eprintln!(
            "  {} No HuggingFace models cached yet.",
            crate::style::yellow("!")
        );
        eprintln!(
            "  {} {}",
            crate::style::dim("·"),
            crate::style::dim(
                "Run e.g. `aivo claude -m hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF` to download one."
            )
        );
        return None;
    }

    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() || !std::io::stdin().is_terminal() {
        eprintln!(
            "  {} Bare `hf:` needs a TTY for the picker. Try `aivo hf list` for the cached set.",
            crate::style::yellow("!")
        );
        return None;
    }

    let repo_width = repos
        .iter()
        .map(|r| r.repo.len())
        .max()
        .unwrap_or(0)
        .min(60);
    let items: Vec<String> = repos
        .iter()
        .map(|r| {
            let quant = r.primary.quant().unwrap_or_else(|| "?".into());
            let age = format_modified_ago(r.modified);
            format!(
                "{:<repo_width$}  {:<9}  {:>9}  used {}",
                r.repo,
                quant,
                human_size(r.total_bytes),
                age,
                repo_width = repo_width
            )
        })
        .collect();

    use crate::tui::FuzzySelect;
    let idx = FuzzySelect::new()
        .with_prompt("Cached HuggingFace models")
        .items(&items)
        .default(0)
        .interact_opt()
        .ok()
        .flatten()?;
    Some(cached_model_as_short_ref(&repos[idx].primary))
}

pub fn format_modified_ago(t: Option<std::time::SystemTime>) -> String {
    let Some(t) = t else {
        return "?".to_string();
    };
    let now = std::time::SystemTime::now();
    let secs = now.duration_since(t).ok().map(|d| d.as_secs()).unwrap_or(0);
    match secs {
        0..=59 => "now".into(),
        60..=3599 => format!("{}m ago", secs / 60),
        3600..=86_399 => format!("{}h ago", secs / 3600),
        86_400..=604_799 => format!("{}d ago", secs / 86_400),
        604_800..=2_592_000 => format!("{}w ago", secs / 604_800),
        2_592_001..=31_535_999 => format!("{}mo ago", secs / 2_592_000),
        _ => format!("{}y ago", secs / 31_536_000),
    }
}

fn format_eta(secs: u64) -> String {
    match secs {
        0..=59 => format!("{secs}s"),
        60..=3599 => format!("{}m {}s", secs / 60, secs % 60),
        3600..=86_399 => format!("{}h {}m", secs / 3600, (secs % 3600) / 60),
        _ => format!("{}d {}h", secs / 86_400, (secs % 86_400) / 3600),
    }
}

/// Scales up to GB.
pub(crate) fn human_size(b: u64) -> String {
    const K: f64 = 1024.0;
    let bf = b as f64;
    if bf < K {
        format!("{b}B")
    } else if bf < K * K {
        format!("{:.1}KB", bf / K)
    } else if bf < K * K * K {
        format!("{:.1}MB", bf / (K * K))
    } else {
        format!("{:.2}GB", bf / (K * K * K))
    }
}

/// Safe to call multiple times; no-op when no server was started.
pub fn stop_if_we_started() {
    if let Ok(mut slot) = child_slot().lock()
        && let Some(mut child) = slot.take()
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers_with_content_range(value: &str) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(reqwest::header::CONTENT_RANGE, value.parse().unwrap());
        h
    }

    #[test]
    fn content_range_start_parses_resume_offset() {
        // Normal CDN resume: range begins exactly where we asked.
        assert_eq!(
            content_range_start(&headers_with_content_range("bytes 1024-4095/4096")),
            Some(1024)
        );
        // A proxy that realigned the start down to a block boundary.
        assert_eq!(
            content_range_start(&headers_with_content_range("bytes 0-4095/4096")),
            Some(0)
        );
        // Unsatisfied/unknown total still yields the start.
        assert_eq!(
            content_range_start(&headers_with_content_range("bytes 512-1023/*")),
            Some(512)
        );
        // Missing or malformed headers are ignored (caller restarts).
        assert_eq!(
            content_range_start(&reqwest::header::HeaderMap::new()),
            None
        );
        assert_eq!(
            content_range_start(&headers_with_content_range("bytes */4096")),
            None
        );
    }

    #[test]
    fn is_huggingface_ref_accepts_both_forms() {
        assert!(is_huggingface_ref("https://huggingface.co/bartowski/x"));
        assert!(is_huggingface_ref("hf:bartowski/x"));
        assert!(is_huggingface_ref("hf:bartowski/x:Q5_K_M"));
        assert!(!is_huggingface_ref("http://huggingface.co/bartowski/x"));
        assert!(!is_huggingface_ref("https://hf.co/bartowski/x"));
        assert!(!is_huggingface_ref("HF:owner/repo"), "case-sensitive");
        assert!(!is_huggingface_ref("gpt-4o"));
        assert!(!is_huggingface_ref(""));
    }

    #[test]
    fn parse_hf_short_bare_repo() {
        let r = parse_hf_ref("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF").unwrap();
        assert_eq!(r.repo, "Qwen/Qwen2.5-0.5B-Instruct-GGUF");
        assert_eq!(r.quant, None);
        assert_eq!(r.file, None);
    }

    #[test]
    fn parse_hf_short_with_quant() {
        let r = parse_hf_ref("hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M").unwrap();
        assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q5_K_M"));
        assert_eq!(r.file, None);
    }

    #[test]
    fn parse_hf_short_with_file_path() {
        // Direct file via the short prefix — equivalent to the URL form.
        let r = parse_hf_ref("hf:owner/repo/resolve/main/model-Q8_0.gguf").unwrap();
        assert_eq!(r.repo, "owner/repo");
        assert_eq!(r.quant.as_deref(), Some("Q8_0"));
        assert_eq!(r.file.as_deref(), Some("model-Q8_0.gguf"));
    }

    #[test]
    fn parse_hf_short_rejects_malformed() {
        assert!(parse_hf_ref("hf:").is_err());
        assert!(parse_hf_ref("hf:only-owner").is_err());
        assert!(parse_hf_ref("hf:owner/").is_err());
        assert!(parse_hf_ref("hf:/repo").is_err());
        // Three-segment repo path with no /blob|/resolve is rejected (not
        // a known shape, would silently mis-route otherwise).
        assert!(parse_hf_ref("hf:owner/repo/extra").is_err());
    }

    #[test]
    fn parse_bare_repo_uses_default_quant() {
        let r =
            parse_hf_url("https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF").unwrap();
        assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(r.quant, None);
        assert_eq!(r.file, None);
        assert_eq!(r.revision, None);
    }

    #[test]
    fn parse_bare_repo_trailing_slash() {
        let r = parse_hf_url("https://huggingface.co/owner/repo/").unwrap();
        assert_eq!(r.repo, "owner/repo");
        assert_eq!(r.quant, None);
        assert_eq!(r.file, None);
        assert_eq!(r.revision, None);
    }

    #[test]
    fn parse_blob_url_extracts_quant_and_file() {
        let r = parse_hf_url(
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/blob/main/Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        )
        .unwrap();
        assert_eq!(r.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(r.quant.as_deref(), Some("Q5_K_M"));
        assert_eq!(r.file.as_deref(), Some("Llama-3.2-3B-Instruct-Q5_K_M.gguf"));
        // Main revision normalizes to None so the cache keeps existing layout.
        assert_eq!(r.revision, None);
    }

    #[test]
    fn parse_resolve_url_extracts_quant_and_file() {
        let r =
            parse_hf_url("https://huggingface.co/owner/repo/resolve/main/model-Q8_0.gguf").unwrap();
        assert_eq!(r.repo, "owner/repo");
        assert_eq!(r.quant.as_deref(), Some("Q8_0"));
        assert_eq!(r.file.as_deref(), Some("model-Q8_0.gguf"));
        assert_eq!(r.revision, None);
    }

    #[test]
    fn parse_url_preserves_non_main_revision() {
        let r =
            parse_hf_url("https://huggingface.co/owner/repo/blob/v1.0/model-Q4_K_M.gguf").unwrap();
        assert_eq!(r.repo, "owner/repo");
        assert_eq!(r.revision.as_deref(), Some("v1.0"));
        assert_eq!(r.file.as_deref(), Some("model-Q4_K_M.gguf"));
    }

    #[test]
    fn parse_url_preserves_nested_file_path() {
        // Nested GGUF paths (e.g. quants/ subdirectory) must round-trip
        // verbatim — previously we kept only the basename and built a
        // /resolve/main/{basename} URL, 404ing on the real upstream.
        let r = parse_hf_url(
            "https://huggingface.co/owner/repo/resolve/main/quants/v2/model-Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(r.repo, "owner/repo");
        assert_eq!(r.file.as_deref(), Some("quants/v2/model-Q4_K_M.gguf"));
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_url_combines_revision_and_nested_path() {
        let r = parse_hf_url("https://huggingface.co/owner/repo/blob/dev/subdir/model-Q5_K_M.gguf")
            .unwrap();
        assert_eq!(r.revision.as_deref(), Some("dev"));
        assert_eq!(r.file.as_deref(), Some("subdir/model-Q5_K_M.gguf"));
    }

    #[test]
    fn parse_url_rejects_blob_with_no_file() {
        // `/blob/main` alone, with no file segment, should fail rather
        // than treat `main` as the filename.
        assert!(parse_hf_url("https://huggingface.co/owner/repo/blob/main").is_err());
    }

    #[test]
    fn parse_thebloke_style_dot_separator() {
        // TheBloke-style filenames use a `.` before the quant tag.
        let r = parse_hf_ref(
            "https://huggingface.co/owner/repo/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        )
        .unwrap();
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn parse_iq_quant() {
        let r = parse_hf_url("https://huggingface.co/owner/repo/resolve/main/model-IQ3_M.gguf")
            .unwrap();
        assert_eq!(r.quant.as_deref(), Some("IQ3_M"));
        assert_eq!(r.file.as_deref(), Some("model-IQ3_M.gguf"));
    }

    #[test]
    fn parse_unknown_quant_falls_back_to_default() {
        // Filename doesn't follow the `-<QUANT>.gguf` convention.
        let r = parse_hf_url("https://huggingface.co/owner/repo/resolve/main/model.gguf").unwrap();
        assert_eq!(r.quant, None);
        assert_eq!(r.file.as_deref(), Some("model.gguf"));
    }

    #[test]
    fn parse_rejects_non_gguf_file() {
        let err = parse_hf_url(
            "https://huggingface.co/openbmb/MiniCPM-V-4.6/resolve/main/model.safetensors",
        )
        .unwrap_err();
        assert!(err.to_string().contains("GGUF"));
    }

    #[test]
    fn parse_rejects_non_hf_url() {
        assert!(parse_hf_url("https://example.com/owner/repo").is_err());
    }

    #[test]
    fn parse_rejects_missing_repo() {
        assert!(parse_hf_url("https://huggingface.co/onlyowner").is_err());
        assert!(parse_hf_url("https://huggingface.co/").is_err());
        assert!(parse_hf_url("https://huggingface.co").is_err());
    }

    #[test]
    fn parse_rejects_tree_or_commits_paths() {
        assert!(
            parse_hf_url("https://huggingface.co/owner/repo/tree/main").is_err(),
            "tree/ paths don't identify a file"
        );
        assert!(parse_hf_url("https://huggingface.co/owner/repo/commits/main").is_err());
    }

    #[test]
    fn display_model_name_is_repo_basename() {
        let r = HfModelRef {
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF".to_string(),
            quant: None,
            file: None,
            revision: None,
            local_source: None,
        };
        assert_eq!(r.display_model_name(), "Llama-3.2-3B-Instruct-GGUF");
    }

    #[test]
    fn local_openai_base_url_formats_correctly() {
        assert_eq!(local_openai_base_url(48721), "http://127.0.0.1:48721/v1");
    }

    #[test]
    fn resolved_ngl_defaults_to_99_on_mac() {
        assert_eq!(resolved_ngl(None, None, true), Some(99));
    }

    #[test]
    fn resolved_ngl_defaults_off_elsewhere() {
        assert_eq!(resolved_ngl(None, None, false), None);
    }

    #[test]
    fn resolved_ngl_gpu_cpu_disables_default() {
        assert_eq!(resolved_ngl(Some("cpu"), None, true), None);
        assert_eq!(resolved_ngl(Some("CPU"), None, true), None);
    }

    #[test]
    fn resolved_ngl_explicit_override_wins_on_any_platform() {
        assert_eq!(resolved_ngl(None, Some("32"), false), Some(32));
        assert_eq!(resolved_ngl(Some("cpu"), Some("32"), true), Some(32));
        assert_eq!(resolved_ngl(None, Some("0"), true), Some(0));
        assert_eq!(resolved_ngl(None, Some("  16 "), false), Some(16));
    }

    #[test]
    fn resolved_ngl_malformed_override_falls_back_to_default() {
        assert_eq!(resolved_ngl(None, Some("nope"), true), Some(99));
        assert_eq!(resolved_ngl(None, Some(""), true), Some(99));
        assert_eq!(resolved_ngl(Some("cpu"), Some("nope"), true), None);
    }

    #[test]
    fn jinja_template_error_detected_from_real_llama_server_output() {
        let tail = "srv          init: init: chat template parsing error: Unable to generate parser for this template. Automatic parser generation failed:\n\
                    While executing FilterExpression at line 6, column 86 in source:\n\
                    Error: Unknown (built-in) filter 'tojson' for type Undefined (hint: 'tools')\n\
                    srv          init: init: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n\
                    main: exiting due to model loading error";
        assert!(is_jinja_template_error(tail));
    }

    #[test]
    fn jinja_template_error_not_triggered_by_unrelated_failures() {
        assert!(!is_jinja_template_error(
            "llama_model_load: error loading model: failed to mmap"
        ));
        assert!(!is_jinja_template_error(""));
    }

    #[test]
    fn local_cache_path_keeps_main_revision_at_existing_layout() {
        // Backwards-compatible: existing caches at .../<repo>/<file>.gguf
        // continue to resolve to the same path when revision is main / empty.
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!("owner__repo{sep}x.gguf");
        let main_path = local_cache_path("owner/repo", "main", "x.gguf").unwrap();
        let empty_path = local_cache_path("owner/repo", "", "x.gguf").unwrap();
        assert!(main_path.to_string_lossy().ends_with(&expected));
        assert!(empty_path.to_string_lossy().ends_with(&expected));
    }

    #[test]
    fn local_cache_path_isolates_non_main_revision() {
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!("owner__repo{sep}@v1.0__x.gguf");
        let p = local_cache_path("owner/repo", "v1.0", "x.gguf").unwrap();
        assert!(
            p.to_string_lossy().ends_with(&expected),
            "got {}",
            p.display()
        );
    }

    #[test]
    fn local_cache_path_flattens_nested_filename() {
        let sep = std::path::MAIN_SEPARATOR;
        let expected = format!("owner__repo{sep}subdir__x.gguf");
        let p = local_cache_path("owner/repo", "main", "subdir/x.gguf").unwrap();
        assert!(
            p.to_string_lossy().ends_with(&expected),
            "got {}",
            p.display()
        );
    }

    fn entry(path: &str) -> TreeEntry {
        TreeEntry {
            path: path.to_string(),
            size: 1,
            kind: "file".to_string(),
        }
    }

    #[test]
    fn find_matching_gguf_prefers_exact_quant() {
        let owned = [
            entry("model-Q4_K_M.gguf"),
            entry("model-Q5_K_M.gguf"),
            entry("model-Q8_0.gguf"),
        ];
        let refs: Vec<&TreeEntry> = owned.iter().collect();
        let hit = find_matching_gguf(&refs, "Q5_K_M", false).unwrap();
        assert_eq!(hit.path, "model-Q5_K_M.gguf");
    }

    #[test]
    fn find_matching_gguf_falls_back_when_default_missing() {
        // Requested Q4_K_M (default), repo only has Q8_0 → pick Q8_0.
        let owned = [entry("llama-3.2-1b-instruct-q8_0.gguf")];
        let refs: Vec<&TreeEntry> = owned.iter().collect();
        let hit = find_matching_gguf(&refs, "Q4_K_M", false).unwrap();
        assert_eq!(hit.path, "llama-3.2-1b-instruct-q8_0.gguf");
    }

    #[test]
    fn find_matching_gguf_explicit_quant_does_not_fall_back() {
        // User explicitly asked for Q5_K_M but the repo only has Q8_0 →
        // refuse to substitute. The error message lets them pick another
        // repo or tag.
        let owned = [entry("llama-3.2-1b-instruct-q8_0.gguf")];
        let refs: Vec<&TreeEntry> = owned.iter().collect();
        assert!(find_matching_gguf(&refs, "Q5_K_M", true).is_none());
    }

    #[test]
    fn find_matching_gguf_dot_separator() {
        // TheBloke-style naming uses a `.` before the quant tag instead of `-`.
        let owned = [entry("mistral-7b-instruct-v0.2.Q4_K_M.gguf")];
        let refs: Vec<&TreeEntry> = owned.iter().collect();
        let hit = find_matching_gguf(&refs, "Q4_K_M", false).unwrap();
        assert_eq!(hit.path, "mistral-7b-instruct-v0.2.Q4_K_M.gguf");
    }

    #[test]
    fn find_matching_gguf_single_file_last_resort() {
        // Repo has one GGUF with a non-standard name. Implicit-quant call
        // should still pick it; explicit call should not.
        let owned = [entry("custom-model.gguf")];
        let refs: Vec<&TreeEntry> = owned.iter().collect();
        assert!(find_matching_gguf(&refs, "Q4_K_M", false).is_some());
        assert!(find_matching_gguf(&refs, "Q4_K_M", true).is_none());
    }

    #[test]
    fn urlencode_handles_space_and_safe_chars() {
        assert_eq!(urlencode("Llama-3.1 GGUF"), "Llama-3.1%20GGUF");
        assert_eq!(urlencode("a.b_c-d~e"), "a.b_c-d~e");
        assert_eq!(urlencode("/"), "%2F");
    }

    #[test]
    fn release_asset_target_matches_host_platform() {
        // We can only assert what the current host build target is. A failing
        // suffix here means the llama.cpp release naming has drifted.
        let target = release_asset_target().expect("host platform should be supported");
        assert!(
            target.asset_suffix.starts_with("-bin-"),
            "asset suffix must be anchored on `-bin-`: {}",
            target.asset_suffix,
        );
        match target.kind {
            ArchiveKind::TarGz => assert!(target.asset_suffix.ends_with(".tar.gz")),
            ArchiveKind::Zip => assert!(target.asset_suffix.ends_with(".zip")),
        }
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn manual_install_hint_names_the_platform_asset() {
        let target = release_asset_target().expect("host platform should be supported");
        let hint = manual_install_hint();
        assert!(
            hint.contains(target.asset_suffix),
            "manual hint should name the actual asset suffix; got: {hint}",
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn manual_install_hint_recommends_brew_on_mac() {
        assert!(manual_install_hint().contains("brew install llama.cpp"));
    }

    #[test]
    fn release_asset_suffix_disambiguates_kleidiai_variant() {
        // `bin-macos-arm64.tar.gz` must not also match `bin-macos-arm64-kleidiai.tar.gz`.
        let kleidiai = "llama-b9294-bin-macos-arm64-kleidiai.tar.gz";
        let plain = "llama-b9294-bin-macos-arm64.tar.gz";
        let suffix = "-bin-macos-arm64.tar.gz";
        assert!(plain.ends_with(suffix));
        assert!(!kleidiai.ends_with(suffix));
    }

    #[test]
    fn human_size_scales_through_gb() {
        assert_eq!(human_size(500), "500B");
        assert_eq!(human_size(1536), "1.5KB");
        assert_eq!(human_size(5 * 1024 * 1024), "5.0MB");
        // 1.91 GB — a typical Q4_K_M quant of a 3B model
        assert_eq!(human_size(2_050_000_000), "1.91GB");
    }

    #[test]
    fn format_eta_covers_all_buckets() {
        assert_eq!(format_eta(0), "0s");
        assert_eq!(format_eta(45), "45s");
        assert_eq!(format_eta(60), "1m 0s");
        assert_eq!(format_eta(125), "2m 5s");
        assert_eq!(format_eta(3600), "1h 0m");
        assert_eq!(format_eta(3725), "1h 2m");
        assert_eq!(format_eta(90_061), "1d 1h");
    }

    #[test]
    fn parse_on_disk_filename_round_trips_main() {
        let (rev, file) = parse_on_disk_filename("model-Q4_K_M.gguf");
        assert_eq!(rev, "main");
        assert_eq!(file, "model-Q4_K_M.gguf");
    }

    #[test]
    fn parse_on_disk_filename_extracts_revision() {
        let (rev, file) = parse_on_disk_filename("@v1.0__model-Q4_K_M.gguf");
        assert_eq!(rev, "v1.0");
        assert_eq!(file, "model-Q4_K_M.gguf");
    }

    #[test]
    fn parse_on_disk_filename_handles_malformed_at_prefix() {
        // No `__` separator after `@` → treat the whole thing as filename.
        let (rev, file) = parse_on_disk_filename("@weird-name.gguf");
        assert_eq!(rev, "main");
        assert_eq!(file, "@weird-name.gguf");
    }

    #[test]
    fn cached_model_launch_ref_short_for_main() {
        let m = CachedModel {
            repo: "owner/repo".into(),
            revision: "main".into(),
            filename: "model-Q4_K_M.gguf".into(),
            size_bytes: 0,
            modified: None,
            path: PathBuf::new(),
        };
        assert_eq!(m.launch_ref(), "hf:owner/repo:Q4_K_M");
    }

    #[test]
    fn cached_model_launch_ref_full_url_for_non_main() {
        let m = CachedModel {
            repo: "owner/repo".into(),
            revision: "v1.0".into(),
            filename: "model-Q4_K_M.gguf".into(),
            size_bytes: 0,
            modified: None,
            path: PathBuf::new(),
        };
        assert_eq!(
            m.launch_ref(),
            "https://huggingface.co/owner/repo/resolve/v1.0/model-Q4_K_M.gguf"
        );
    }

    #[test]
    fn cached_model_launch_ref_full_url_for_nested_path() {
        let m = CachedModel {
            repo: "owner/repo".into(),
            revision: "main".into(),
            filename: "subdir/model-Q4_K_M.gguf".into(),
            size_bytes: 0,
            modified: None,
            path: PathBuf::new(),
        };
        assert_eq!(
            m.launch_ref(),
            "https://huggingface.co/owner/repo/resolve/main/subdir/model-Q4_K_M.gguf"
        );
    }

    #[test]
    fn split_repo_and_quant_extracts_dash_suffix() {
        let (stem, q) = split_repo_and_quant("Llama-3.2-3B-Instruct-Q5_K_M.gguf");
        assert_eq!(stem, "Llama-3.2-3B-Instruct");
        assert_eq!(q.as_deref(), Some("Q5_K_M"));
    }

    #[test]
    fn split_repo_and_quant_extracts_dot_suffix() {
        let (stem, q) = split_repo_and_quant("Model.Q4_K_M.gguf");
        assert_eq!(stem, "Model");
        assert_eq!(q.as_deref(), Some("Q4_K_M"));
    }

    #[test]
    fn split_repo_and_quant_no_quant_tag() {
        let (stem, q) = split_repo_and_quant("custom-model.gguf");
        assert_eq!(stem, "custom-model");
        assert_eq!(q, None);
    }

    #[test]
    fn split_repo_and_quant_accepts_iq_and_f16() {
        assert_eq!(
            split_repo_and_quant("Phi-IQ3_M.gguf").1.as_deref(),
            Some("IQ3_M")
        );
        assert_eq!(
            split_repo_and_quant("Tiny-F16.gguf").1.as_deref(),
            Some("F16")
        );
    }

    #[test]
    fn looks_like_local_gguf_path_recognizes_anchors() {
        assert!(looks_like_local_gguf_path("/abs/path.gguf"));
        assert!(looks_like_local_gguf_path("./rel/model.gguf"));
        assert!(looks_like_local_gguf_path("../up.gguf"));
        assert!(looks_like_local_gguf_path("~/in-home.gguf"));
        assert!(looks_like_local_gguf_path("bare-file.gguf"));
        assert!(!looks_like_local_gguf_path("hf:owner/repo"));
        assert!(!looks_like_local_gguf_path("https://huggingface.co/x/y"));
        assert!(!looks_like_local_gguf_path("gpt-4o"));
        assert!(!looks_like_local_gguf_path(""));
    }

    #[test]
    fn is_gguf_name_is_total_over_non_ascii() {
        // len-5 lands mid-char here; a str-slice implementation panics.
        assert!(!is_gguf_name("总结一下这个项目"));
        assert!(!is_gguf_name("你好吗"));
        assert!(is_gguf_name("模型.gguf"));
    }

    #[test]
    fn split_repo_and_quant_strips_any_suffix_case() {
        assert_eq!(
            split_repo_and_quant("Model-Q5_K_M.Gguf"),
            ("Model", Some("Q5_K_M".to_string()))
        );
        assert_eq!(
            split_repo_and_quant("Model-Q5_K_M.GGUF"),
            ("Model", Some("Q5_K_M".to_string()))
        );
        assert_eq!(
            split_repo_and_quant("no-quant-here"),
            ("no-quant-here", None)
        );
    }

    #[test]
    fn parse_hf_ref_errors_on_missing_local_file() {
        let result = parse_hf_ref("/tmp/aivo-definitely-does-not-exist.gguf");
        let err = result.err().unwrap();
        assert!(err.to_string().contains("No such file"), "got {err}");
    }

    #[test]
    fn parse_hf_ref_rejects_non_gguf_local_file() {
        let tmp = std::env::temp_dir().join("aivo-parse-not-gguf.bin");
        std::fs::write(&tmp, b"x").unwrap();
        let result = parse_hf_ref(tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Only `.gguf`"), "got {err}");
    }

    #[test]
    fn parse_hf_ref_local_path_derives_repo_and_quant() {
        let tmp = std::env::temp_dir().join("aivo-parse-Q4_K_M.gguf");
        std::fs::write(&tmp, b"x").unwrap();
        let result = parse_hf_ref(tmp.to_str().unwrap());
        let _ = std::fs::remove_file(&tmp);
        let r = result.unwrap();
        assert_eq!(r.repo, "local/aivo-parse");
        assert_eq!(r.quant.as_deref(), Some("Q4_K_M"));
        assert_eq!(r.local_source.as_deref(), Some(tmp.as_path()));
    }

    #[test]
    fn parse_hf_ref_hf_short_has_no_local_source() {
        let r = parse_hf_ref("hf:owner/repo").unwrap();
        assert!(r.local_source.is_none());
    }

    #[test]
    fn local_cache_path_is_under_aivo_cache() {
        let sep = std::path::MAIN_SEPARATOR;
        let dir_segment = format!(
            ".config{sep}aivo{sep}cache{sep}huggingface{sep}bartowski__Llama-3.2-3B-Instruct-GGUF{sep}x.gguf"
        );
        let p = local_cache_path("bartowski/Llama-3.2-3B-Instruct-GGUF", "main", "x.gguf").unwrap();
        let s = p.to_string_lossy();
        assert!(s.contains(&dir_segment), "got {s}");
    }

    fn build_gguf_header(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes()); // version
        out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, ty, val) in kvs {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(val);
        }
        out
    }

    fn gguf_string_val(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn read_gguf_meta_pulls_arch_after_skipping_other_kvs() {
        // u32 value (type 4), then a string array (type 9 of type 8, len 2),
        // then the architecture string — exercises scalar + array skips.
        let mut array_val = Vec::new();
        array_val.extend_from_slice(&8u32.to_le_bytes()); // element type = string
        array_val.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
        array_val.extend(gguf_string_val("hello"));
        array_val.extend(gguf_string_val("world"));

        let bytes = build_gguf_header(&[
            (
                "general.quantization_version",
                4,
                2u32.to_le_bytes().to_vec(),
            ),
            ("tokenizer.ggml.tokens", 9, array_val),
            ("general.architecture", 8, gguf_string_val("bert")),
        ]);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        let meta = read_gguf_meta_from(&mut cursor, &mut budget);
        assert_eq!(meta.architecture.as_deref(), Some("bert"));
        assert_eq!(meta.context_length, None);
    }

    #[test]
    fn read_gguf_meta_returns_empty_on_bad_magic() {
        let bytes = b"NOPE\x03\x00\x00\x00".to_vec();
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        assert_eq!(
            read_gguf_meta_from(&mut cursor, &mut budget),
            GgufMeta::default()
        );
    }

    #[test]
    fn read_gguf_meta_captures_u32_context_length() {
        let bytes = build_gguf_header(&[
            ("general.architecture", 8, gguf_string_val("llama")),
            ("llama.vocab_size", 4, 32_000u32.to_le_bytes().to_vec()),
            ("llama.context_length", 4, 131_072u32.to_le_bytes().to_vec()),
        ]);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        let meta = read_gguf_meta_from(&mut cursor, &mut budget);
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, Some(131_072));
    }

    #[test]
    fn read_gguf_meta_captures_u64_context_length() {
        let bytes = build_gguf_header(&[
            ("general.architecture", 8, gguf_string_val("qwen2")),
            ("qwen2.context_length", 10, 32_768u64.to_le_bytes().to_vec()),
        ]);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        assert_eq!(
            read_gguf_meta_from(&mut cursor, &mut budget).context_length,
            Some(32_768)
        );
    }

    #[test]
    fn read_gguf_meta_ignores_foreign_context_length_once_arch_known() {
        // `clip.context_length` must not satisfy a `llama` model's lookup.
        let bytes = build_gguf_header(&[
            ("general.architecture", 8, gguf_string_val("llama")),
            ("clip.context_length", 4, 77u32.to_le_bytes().to_vec()),
            ("llama.context_length", 4, 8_192u32.to_le_bytes().to_vec()),
        ]);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        assert_eq!(
            read_gguf_meta_from(&mut cursor, &mut budget).context_length,
            Some(8_192)
        );
    }

    #[test]
    fn read_gguf_meta_accepts_context_length_before_arch() {
        let bytes = build_gguf_header(&[
            ("llama.context_length", 4, 4_096u32.to_le_bytes().to_vec()),
            ("general.architecture", 8, gguf_string_val("llama")),
        ]);
        let mut cursor = std::io::Cursor::new(bytes);
        let mut budget = 2 * 1024 * 1024;
        let meta = read_gguf_meta_from(&mut cursor, &mut budget);
        assert_eq!(meta.context_length, Some(4_096));
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
    }

    #[test]
    fn resolve_ctx_user_flag_wins_and_suppresses_aivos() {
        // Explicit user value: aivo passes nothing, advertises the value.
        assert_eq!(
            resolve_ctx(true, Some(8_192), Some(99_999), Some(131_072)),
            (None, 8_192)
        );
        // `-c 0` = model training context.
        assert_eq!(
            resolve_ctx(true, Some(0), None, Some(131_072)),
            (None, 131_072)
        );
        // `-c 0` with unreadable header falls to the unknown default.
        assert_eq!(resolve_ctx(true, Some(0), None, None), (None, 32_768));
    }

    #[test]
    fn resolve_ctx_env_beats_default() {
        assert_eq!(
            resolve_ctx(false, None, Some(131_072), Some(8_192)),
            (Some(131_072), 131_072)
        );
    }

    #[test]
    fn resolve_ctx_defaults_to_capped_training_ctx() {
        // Big training context clamps to the cap.
        assert_eq!(
            resolve_ctx(false, None, None, Some(1_048_576)),
            (Some(65_536), 65_536)
        );
        // Small training context is used as-is (never stretched).
        assert_eq!(
            resolve_ctx(false, None, None, Some(8_192)),
            (Some(8_192), 8_192)
        );
        // Unknown header → bounded default.
        assert_eq!(resolve_ctx(false, None, None, None), (Some(32_768), 32_768));
    }

    #[test]
    fn parse_user_llama_args_splits_and_strips_owned_flags() {
        let (kept, stripped) =
            parse_user_llama_args("--temp 0.6 --port 9999 -m /x.gguf --host=0.0.0.0 -fa on")
                .unwrap();
        assert_eq!(kept, ["--temp", "0.6", "-fa", "on"]);
        assert_eq!(stripped, ["--port", "-m", "--host"]);
    }

    #[test]
    fn parse_user_llama_args_handles_quotes_and_empty() {
        let (kept, _) = parse_user_llama_args(r#"--chat-template "a b c""#).unwrap();
        assert_eq!(kept, ["--chat-template", "a b c"]);
        assert_eq!(parse_user_llama_args("").unwrap().0, Vec::<String>::new());
        assert_eq!(
            parse_user_llama_args("   ").unwrap().0,
            Vec::<String>::new()
        );
    }

    #[test]
    fn parse_user_llama_args_rejects_unbalanced_quote() {
        let err = parse_user_llama_args(r#"--temp "0.6"#).unwrap_err();
        assert!(err.to_string().contains("AIVO_LLAMA_ARGS"), "got {err}");
    }

    #[test]
    fn user_ctx_directive_detects_both_spellings() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            user_ctx_directive(&args(&["-c", "4096"])),
            (true, Some(4_096))
        );
        assert_eq!(
            user_ctx_directive(&args(&["--ctx-size=16384"])),
            (true, Some(16_384))
        );
        assert_eq!(user_ctx_directive(&args(&["--temp", "1"])), (false, None));
        // Owns it even when the value doesn't parse — aivo must not add
        // a second `-c`.
        assert_eq!(user_ctx_directive(&args(&["-c", "lots"])), (true, None));
    }

    #[test]
    fn flag_value_takes_last_occurrence() {
        let args: Vec<String> = ["-c", "1024", "--ctx-size=2048"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(flag_value(&args, &["-c", "--ctx-size"]), Some("2048"));
    }

    #[test]
    fn sidecar_classification() {
        assert!(is_mmproj_name("mmproj-BF16.gguf"));
        assert!(is_mmproj_name("mmproj-model-f16.gguf"));
        // Cache-flattened and revision-prefixed forms classify the same.
        assert!(is_mmproj_name("@abc123__mmproj-BF16.gguf"));
        assert!(is_mmproj_name("subdir/mmproj-F32.gguf"));
        assert!(is_mtp_draft_name("gemma-4-26B-A4B-it-Q8_0-MTP.gguf"));
        assert!(is_mtp_draft_name("model-q4-mtp.GGUF"));
        assert!(!is_mmproj_name("gemma-4-26B-A4B-it-Q4_K_XL.gguf"));
        assert!(!is_mtp_draft_name("gemma-4-26B-A4B-it-Q4_K_XL.gguf"));
        // `-MTP` must be a suffix of the stem, not a substring.
        assert!(!is_mtp_draft_name("MTP-bench-Q4_K_M.gguf"));
        assert!(!is_sidecar_filename("Llama-3.2-3B-Instruct-Q5_K_M.gguf"));
    }

    #[test]
    fn mmproj_pick_prefers_precision_order() {
        assert_eq!(
            pick_mmproj_idx(&["mmproj-F32.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"]),
            Some(2)
        );
        assert_eq!(pick_mmproj_idx(&["mmproj-weird.gguf"]), Some(0));
        assert_eq!(pick_mmproj_idx(&[]), None);
    }

    #[test]
    fn mtp_pick_requires_single_candidate() {
        assert_eq!(pick_mtp_idx(&["a-MTP.gguf"]), Some(0));
        assert_eq!(pick_mtp_idx(&["a-MTP.gguf", "b-MTP.gguf"]), None);
        assert_eq!(pick_mtp_idx(&[]), None);
    }

    #[test]
    fn draft_flag_error_requires_flag_mention_and_arg_error() {
        assert!(is_draft_flag_error(
            "error: invalid argument: --spec-type\nusage: llama-server ..."
        ));
        assert!(is_draft_flag_error("unknown argument: --model-draft"));
        // A real draft-model load failure is NOT an argument error.
        assert!(!is_draft_flag_error(
            "failed to load model '--model-draft path': tensor mismatch"
        ));
        assert!(!is_draft_flag_error("invalid argument: --frobnicate"));
    }

    #[test]
    fn build_spawn_args_orders_aivo_then_user_then_rescue_template() {
        let user: Vec<String> = ["--temp", "0.6"].iter().map(|s| s.to_string()).collect();
        let args = build_spawn_args(
            Some(65_536),
            Some(Path::new("/cache/mmproj-BF16.gguf")),
            Some(Path::new("/cache/m-Q8_0-MTP.gguf")),
            true,
            &user,
        );
        let joined = args.join(" ");
        assert!(joined.starts_with("-c 65536 --mmproj"), "got {joined}");
        assert!(
            joined.contains(
                "--model-draft /cache/m-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 3"
            ),
            "got {joined}"
        );
        // User args after aivo's flags; chatml rescue last so it wins.
        assert!(
            joined.ends_with("--jinja --temp 0.6 --chat-template chatml"),
            "got {joined}"
        );
    }

    #[test]
    fn build_spawn_args_minimal_is_jinja_only() {
        assert_eq!(build_spawn_args(None, None, None, false, &[]), ["--jinja"]);
    }

    #[test]
    fn discover_sidecars_pairs_main_revision_files_only() {
        let dir = tempfile::tempdir().unwrap();
        let touch = |name: &str| std::fs::write(dir.path().join(name), b"x").unwrap();
        touch("model-Q4_K_M.gguf");
        touch("mmproj-F16.gguf");
        touch("mmproj-BF16.gguf");
        touch("model-Q8_0-MTP.gguf");
        touch("@deadbeef__mmproj-F32.gguf"); // pinned revision: ignored
        touch("notes.txt");

        let found = discover_sidecars(&dir.path().join("model-Q4_K_M.gguf"));
        assert_eq!(
            found.mmproj.as_deref(),
            Some(dir.path().join("mmproj-BF16.gguf").as_path())
        );
        assert_eq!(
            found.mtp_draft.as_deref(),
            Some(dir.path().join("model-Q8_0-MTP.gguf").as_path())
        );
    }

    #[test]
    fn discover_sidecars_skips_ambiguous_drafts_and_main_itself() {
        let dir = tempfile::tempdir().unwrap();
        let touch = |name: &str| std::fs::write(dir.path().join(name), b"x").unwrap();
        touch("model-Q4_K_M.gguf");
        touch("model-Q8_0-MTP.gguf");
        touch("model-BF16-MTP.gguf");

        let found = discover_sidecars(&dir.path().join("model-Q4_K_M.gguf"));
        assert_eq!(found.mmproj, None);
        assert_eq!(found.mtp_draft, None, "two MTP candidates → no safe pick");
    }

    #[test]
    fn non_chat_arch_label_flags_encoder_families() {
        assert!(non_chat_arch_label("bert").is_some());
        assert!(non_chat_arch_label("roberta").is_some());
        assert!(non_chat_arch_label("xlm-roberta").is_some());
        assert!(non_chat_arch_label("nomic-bert").is_some());
        assert!(non_chat_arch_label("jina-bert-v2").is_some());
        assert!(non_chat_arch_label("t5encoder").is_some());
        // Generative families must pass through.
        assert!(non_chat_arch_label("llama").is_none());
        assert!(non_chat_arch_label("qwen2").is_none());
        assert!(non_chat_arch_label("mistral").is_none());
        // Unknown arches default to "let it through" so new generative
        // models aren't blocked.
        assert!(non_chat_arch_label("brand-new-arch").is_none());
    }

    #[test]
    fn ensure_arch_is_chat_capable_rejects_bert_gguf() {
        let bytes = build_gguf_header(&[("general.architecture", 8, gguf_string_val("bert"))]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        let err = ensure_arch_is_chat_capable(&read_gguf_meta(tmp.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("`bert`"), "got: {msg}");
        assert!(msg.contains("logits computation"), "got: {msg}");
    }

    #[test]
    fn ensure_arch_is_chat_capable_passes_llama_gguf() {
        let bytes = build_gguf_header(&[("general.architecture", 8, gguf_string_val("llama"))]);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &bytes).unwrap();
        ensure_arch_is_chat_capable(&read_gguf_meta(tmp.path())).unwrap();
    }

    #[test]
    fn hf_token_from_env_precedence_and_trim() {
        let lookup = |k: &str| match k {
            "HF_TOKEN" => Some("  hf_primary  ".to_string()),
            "HUGGING_FACE_HUB_TOKEN" => Some("hf_secondary".to_string()),
            _ => None,
        };
        assert_eq!(hf_token_from_env(lookup), Some("hf_primary".to_string()));
    }

    #[test]
    fn hf_token_from_env_skips_blank_to_next_var() {
        let lookup = |k: &str| match k {
            "HF_TOKEN" => Some("   ".to_string()),
            "HUGGING_FACE_HUB_TOKEN" => Some("hf_secondary".to_string()),
            _ => None,
        };
        assert_eq!(hf_token_from_env(lookup), Some("hf_secondary".to_string()));
    }

    #[test]
    fn hf_token_from_env_none_when_all_unset() {
        assert_eq!(hf_token_from_env(|_| None), None);
    }

    #[test]
    fn hf_token_file_path_prefers_token_path_then_home() {
        let home = Some(PathBuf::from("/home/u"));
        assert_eq!(
            hf_token_file_path_from(Some("/x/tok".into()), Some("/y".into()), home.clone()),
            Some(PathBuf::from("/x/tok"))
        );
        assert_eq!(
            hf_token_file_path_from(Some("  ".into()), Some("/y".into()), home.clone()),
            Some(PathBuf::from("/y/token"))
        );
        assert_eq!(
            hf_token_file_path_from(None, None, home),
            Some(PathBuf::from("/home/u/.cache/huggingface/token"))
        );
        assert_eq!(hf_token_file_path_from(None, None, None), None);
    }

    #[test]
    fn read_hf_token_file_trims_and_rejects_empty() {
        assert_eq!(read_hf_token_file(None), None);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "  hf_fromfile\n").unwrap();
        assert_eq!(
            read_hf_token_file(Some(tmp.path().to_path_buf())),
            Some("hf_fromfile".to_string())
        );
        let blank = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(blank.path(), "\n  \n").unwrap();
        assert_eq!(read_hf_token_file(Some(blank.path().to_path_buf())), None);
    }

    #[test]
    fn is_trusted_gguf_converter_matches_owner_only() {
        assert!(is_trusted_gguf_converter("unsloth/gemma-4-12b-it-GGUF"));
        assert!(is_trusted_gguf_converter("bartowski/whatever-GGUF"));
        assert!(is_trusted_gguf_converter("LMStudio-Community/x")); // case-insensitive
        assert!(!is_trusted_gguf_converter(
            "llmfan46/gemma-3-uncensored-GGUF"
        ));
        // "unsloth" must be the owner, not a substring of the repo name.
        assert!(!is_trusted_gguf_converter("someone/unsloth-clone-GGUF"));
    }

    #[test]
    fn rank_gguf_mirrors_floats_trusted_then_truncates() {
        // Mimics the relevance-ordered, download-blind result for a fresh model:
        // trusted converters interleaved with noise; order preserved per group.
        let ids = vec![
            "Dampfinchen/gemma-3-12b-it-qat-q4_0-gguf-small-fix".to_string(),
            "unsloth/gemma-4-12b-it-GGUF".to_string(),
            "llmfan46/gemma-3-12b-it-heretic-GGUF".to_string(),
            "bartowski/gemma-4-12B-it-GGUF".to_string(),
        ];
        let ranked = rank_gguf_mirrors(ids);
        assert_eq!(
            ranked,
            vec![
                "unsloth/gemma-4-12b-it-GGUF".to_string(),
                "bartowski/gemma-4-12B-it-GGUF".to_string(),
                "Dampfinchen/gemma-3-12b-it-qat-q4_0-gguf-small-fix".to_string(),
                "llmfan46/gemma-3-12b-it-heretic-GGUF".to_string(),
            ]
        );
    }

    #[test]
    fn rank_gguf_mirrors_truncates_to_limit() {
        let ids: Vec<String> = (0..20).map(|i| format!("owner{i}/m-GGUF")).collect();
        assert_eq!(rank_gguf_mirrors(ids).len(), MIRROR_SUGGESTION_LIMIT);
    }
}
