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

use crate::services::http_debug::LoggedSend;
use crate::services::http_utils;
use crate::services::system_env;

static SERVER_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();

const DEFAULT_QUANT: &str = "Q4_K_M";
const HF_URL_PREFIX: &str = "https://huggingface.co/";
const HF_SHORT_PREFIX: &str = "hf:";

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

/// Allocation-free `.gguf`/`.GGUF` suffix check.
fn is_gguf_name(s: &str) -> bool {
    s.len() >= 5 && s[s.len() - 5..].eq_ignore_ascii_case(".gguf")
}

/// Splits `Model-Q5_K_M.gguf` → (`Model`, Some(`Q5_K_M`)). Handles both
/// `-` and `.` separators. Returns `(stem, None)` when no quant tag is
/// recognized.
fn split_repo_and_quant(filename: &str) -> (&str, Option<String>) {
    let stem = filename
        .strip_suffix(".gguf")
        .or_else(|| filename.strip_suffix(".GGUF"))
        .unwrap_or(filename);
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
    }
    #[cfg(unix)]
    {
        dirs.push(PathBuf::from("/opt/homebrew/bin"));
        dirs.push(PathBuf::from("/usr/local/bin"));
    }
    dirs
}

/// macOS auto-installs via brew; other platforms print a manual hint.
pub async fn ensure_installed() -> Result<PathBuf> {
    if let Some(p) = detect_binary() {
        return Ok(p);
    }

    eprintln!(
        "  {} llama-server is not installed.",
        crate::style::yellow("?")
    );

    let hint = manual_install_hint();
    #[cfg(target_os = "macos")]
    let can_auto = which_brew().is_some();
    #[cfg(not(target_os = "macos"))]
    let can_auto = false;

    if !can_auto {
        anyhow::bail!(
            "llama-server is required to run HuggingFace models directly.\n  Install: {hint}"
        );
    }

    use std::io::IsTerminal;
    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "llama-server is required to run HuggingFace models directly.\n  Install: {hint}"
        );
    }

    eprint!(
        "  {} Install via `brew install llama.cpp`? [Y/n] ",
        crate::style::yellow("?")
    );
    let _ = std::io::stderr().flush();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    if !matches!(input.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes") {
        anyhow::bail!("Install: {hint}");
    }

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

    detect_binary().ok_or_else(|| {
        anyhow::anyhow!(
            "llama-server was installed but not found on PATH. You may need to restart your shell."
        )
    })
}

fn manual_install_hint() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "brew install llama.cpp"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "see https://github.com/ggerganov/llama.cpp/releases for prebuilt binaries"
    }
    #[cfg(windows)]
    {
        "download llama-server.exe from https://github.com/ggerganov/llama.cpp/releases"
    }
}

#[cfg(target_os = "macos")]
fn which_brew() -> Option<PathBuf> {
    use crate::services::path_search::{collect_path_dirs, find_in_dirs};
    find_in_dirs("brew", &collect_path_dirs())
        .or_else(|| Some(PathBuf::from("/opt/homebrew/bin/brew")).filter(|p| p.exists()))
        .or_else(|| Some(PathBuf::from("/usr/local/bin/brew")).filter(|p| p.exists()))
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

/// Cache-first: skips the HF tree API call when the file is already on
/// disk. Separate from [`ensure_ready`] so `aivo hf pull` can populate
/// the cache without spawning anything. For local-path refs, imports
/// from disk instead of downloading.
pub async fn ensure_cached(model: &HfModelRef) -> Result<CachedFile> {
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
    let resolved = resolve_gguf_file(model).await?;
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

    Ok(CachedFile {
        repo: resolved.repo,
        revision: resolved.revision,
        filename: resolved.filename,
        size_bytes: resolved.size_bytes,
        path: cache_path,
        was_cached,
    })
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
            !name.starts_with('@') && is_gguf_name(&name)
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

    let port = match try_spawn_and_warmup(&bin, &cache_path, cache_hit, &[]).await? {
        WarmupOutcome::Ready(p) => p,
        WarmupOutcome::JinjaFailed { .. } => {
            // Some GGUFs ship Jinja chat templates that use filters
            // (e.g. `tojson`) that llama.cpp's embedded minijinja can't
            // parse. llama-server's own hint is to retry with --no-jinja.
            eprintln!(
                "  {} Model's embedded jinja chat template failed to parse; \
                 retrying with --no-jinja (tool-call template fidelity may be reduced)",
                crate::style::yellow("!"),
            );
            stop_if_we_started();
            match try_spawn_and_warmup(&bin, &cache_path, cache_hit, &["--no-jinja"]).await? {
                WarmupOutcome::Ready(p) => p,
                WarmupOutcome::JinjaFailed { stderr_tail } => {
                    stop_if_we_started();
                    anyhow::bail!(
                        "llama-server failed even with --no-jinja:\n--- last stderr ---\n{stderr_tail}"
                    )
                }
            }
        }
    };
    let _ = SERVER_PORT.set(port);
    Ok(port)
}

enum WarmupOutcome {
    Ready(u16),
    JinjaFailed { stderr_tail: String },
}

async fn try_spawn_and_warmup(
    bin: &Path,
    cache_path: &Path,
    cache_hit: bool,
    extra_args: &[&str],
) -> Result<WarmupOutcome> {
    let port = alloc_free_port()?;
    if !cache_hit {
        eprintln!(
            "  {} Starting llama-server on port {}",
            crate::style::dim("⟳"),
            port
        );
    }

    let mut cmd = Command::new(bin);
    cmd.arg("-m")
        .arg(cache_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1");
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
            } else {
                stop_if_we_started();
                anyhow::bail!(
                    "llama-server exited during warmup on port {port}.\n--- last stderr ---\n{tail}"
                )
            }
        }
        WaitOutcome::Timeout => {
            let tail = drain.snapshot();
            stop_if_we_started();
            anyhow::bail!(
                "llama-server did not become ready within 10 minutes on port {port}.\n\
                 --- last stderr ---\n{tail}"
            )
        }
    }
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
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(rename = "type", default)]
    kind: String,
}

async fn resolve_gguf_file(model: &HfModelRef) -> Result<ResolvedGgufFile> {
    if let Some(cached) = load_cached_metadata(model) {
        return Ok(cached);
    }
    let resolved = resolve_gguf_file_uncached(model).await?;
    save_cached_metadata(model, &resolved);
    Ok(resolved)
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
            return Ok(ResolvedGgufFile {
                repo: current.repo.clone(),
                revision: revision.to_string(),
                filename: file.clone(),
                download_url: url,
                size_bytes: size,
            });
        }

        let quant = current.quant.as_deref().unwrap_or(DEFAULT_QUANT);
        let quant_upper = quant.to_ascii_uppercase();
        let tree_url = format!(
            "https://huggingface.co/api/models/{}/tree/{}",
            current.repo, revision
        );
        let client = http_utils::router_http_client();
        let resp = client
            .get(&tree_url)
            .send_logged()
            .await
            .with_context(|| format!("Failed to query HuggingFace tree API at {tree_url}"))?;
        if !resp.status().is_success() {
            anyhow::bail!(
                "HuggingFace tree API returned HTTP {} for {}",
                resp.status(),
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
        let explicit_quant = current.quant.is_some();
        let matched =
            find_matching_gguf(&singles, &quant_upper, explicit_quant).ok_or_else(|| {
                let available: Vec<String> = singles.iter().map(|e| e.path.clone()).collect();
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
        });
    }
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

/// Best-effort top-3 GGUF mirror search. Returns empty on any error so
/// the error path renders even when the network is flaky.
async fn search_gguf_mirrors(basename: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct SearchHit {
        id: String,
        #[serde(default)]
        tags: Vec<String>,
    }

    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit=10",
        urlencode(&format!("{basename} GGUF"))
    );
    let client = http_utils::router_http_client();
    let Ok(resp) = client.get(&url).send_logged().await else {
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
    hits.into_iter()
        .filter(|h| {
            h.tags.iter().any(|t| t.eq_ignore_ascii_case("gguf"))
                || h.id.to_ascii_uppercase().contains("GGUF")
        })
        .take(3)
        .map(|h| h.id)
        .collect()
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
    };
    let Ok(json) = serde_json::to_string(&entry) else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, json);
}

async fn head_content_length(url: &str) -> Result<u64> {
    let client = http_utils::router_http_client();
    let resp = client
        .head(url)
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
    let mut req = client.get(&file.download_url);
    if resume_offset > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
    }
    let resp = req
        .send_logged()
        .await
        .with_context(|| format!("GET {} failed", file.download_url))?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "Download of {} returned HTTP {}",
            file.filename,
            resp.status()
        );
    }

    // Server honored `Range` → append. Anything else (incl. 200 OK when we
    // asked for a range) means we can't trust the partial; truncate and restart.
    let resuming = resume_offset > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
    let starting_offset = if resuming { resume_offset } else { 0 };
    let mut out = if resuming {
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

    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("Failed to rename {} → {}", tmp.display(), dest.display()))?;
    Ok(())
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
    let mut by_repo: HashMap<String, CachedRepo> = HashMap::new();
    for m in list_cached_models() {
        let entry = by_repo.entry(m.repo.clone()).or_insert_with(|| CachedRepo {
            repo: m.repo.clone(),
            total_bytes: 0,
            modified: None,
            primary: m.clone(),
            files: Vec::new(),
        });
        entry.total_bytes += m.size_bytes;
        if entry.modified < m.modified {
            entry.modified = m.modified;
            entry.primary = m.clone();
        }
        entry.files.push(m);
    }
    let mut repos: Vec<CachedRepo> = by_repo.into_values().collect();
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

/// Scales up to GB (unlike `media_io::human_bytes`, which stops at MB).
fn human_size(b: u64) -> String {
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
}
