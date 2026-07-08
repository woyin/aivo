//! UpdateCommand handler for CLI self-update functionality.
use std::env;
#[cfg(not(windows))]
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use reqwest::Client;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::errors::ExitCode;
#[cfg(not(windows))]
use crate::services::path_search::{collect_path_dirs, find_in_dirs};
use crate::style;

const DOWNLOAD_BASE: &str = "https://getaivo.dev/dl";

/// Source for `--sync-model-data`: refreshes model metadata between releases.
const MODELS_DEV_API: &str = "https://models.dev/api.json";

/// Download ceiling (the file is ~2 MB today); bounds memory if the host misbehaves.
const MODELS_DEV_MAX_BYTES: u64 = 32 * 1024 * 1024;

/// Trust anchor for self-update: every download is verified against this Ed25519
/// key, whose secret lives only in CI and signs each release. Authenticity check
/// — SHA-256 only guards against corruption.
const MINISIGN_PUBKEY: &str = "RWTXF3LNcIUx6667XJo3zslNJQPdcNqMagE/Qp7AQUZTQ2BoghNzgwd7";

pub struct UpdateCommand {
    client: Client,
    /// Separate client for binary downloads: no total deadline, only a
    /// per-read timeout so slow-but-progressing connections finish.
    download_client: Client,
}

impl UpdateCommand {
    pub fn print_help() {
        println!("{} aivo update [OPTIONS]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim(
                "Update the CLI tool to the latest version (delegates to Homebrew when installed that way)."
            )
        );
        println!();
        println!("{}", style::bold("Options:"));
        let print_opt = |flag: &str, desc: &str| {
            println!(
                "  {}{}",
                style::cyan(format!("{:<26}", flag)),
                style::dim(desc)
            );
        };
        print_opt("-f, --force", "Force update even if via a package manager");
        print_opt("--rollback", "Restore the previous version (last backup)");
        print_opt("--sync-model-data", "Refresh model data from models.dev");
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo update"));
        println!("  {}", style::dim("aivo update --rollback"));
        println!("  {}", style::dim("aivo update --sync-model-data"));
    }

    pub fn new() -> Result<Self> {
        let client = crate::services::http_utils::aivo_http_client_builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("Failed to create HTTP client")?;

        let download_client = crate::services::http_utils::aivo_http_client_builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .read_timeout(std::time::Duration::from_secs(60))
            .build()
            .context("Failed to create download HTTP client")?;

        Ok(Self {
            client,
            download_client,
        })
    }

    /// Executes the update command (binary only — model data is refreshed
    /// separately via `aivo update --sync-model-data`).
    pub async fn execute(&self, force: bool, elevated: bool) -> ExitCode {
        match self.execute_internal(force, elevated).await {
            Ok(code) => code,
            Err(e) => {
                self.handle_error(e);
                ExitCode::UserError
            }
        }
    }

    /// Refreshes the model-limits snapshot from live models.dev, leaving the
    /// binary untouched (`aivo update --sync-model-data`). Returns non-zero on
    /// failure since the refresh is the user's explicit request.
    pub async fn execute_sync_model_data(&self) -> ExitCode {
        println!(
            "{} Refreshing model data from models.dev...",
            style::arrow_symbol()
        );
        match self.refresh_model_data().await {
            Ok(count) => {
                println!(
                    "  {} {}",
                    style::dim("Model data:"),
                    style::green(format!("updated ({count} models)"))
                );
                ExitCode::Success
            }
            Err(e) => {
                self.handle_error(e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, force: bool, elevated: bool) -> Result<ExitCode> {
        if !force {
            let install_path = get_install_path()?;
            if let Some(manager) = detect_managed_install(&install_path) {
                match manager.kind {
                    PackageManager::Homebrew => {
                        return Ok(self.update_via_homebrew());
                    }
                    PackageManager::Cargo => {
                        eprintln!(
                            "{} aivo was installed via {}.",
                            style::yellow("Warning:"),
                            manager.name
                        );
                        eprintln!(
                            "  Self-update would bypass {} and may cause issues.",
                            manager.name
                        );
                        eprintln!();
                        eprintln!(
                            "  {} {}",
                            style::dim("Update with:"),
                            style::green(manager.upgrade_command)
                        );
                        eprintln!(
                            "  {} {}",
                            style::dim("Force self-update:"),
                            style::green("aivo update --force")
                        );
                        return Ok(ExitCode::UserError);
                    }
                }
            }
        }

        // The elevated re-run inherits the parent's preamble output.
        if !elevated {
            println!("{} Checking for updates...", style::arrow_symbol());
        }

        let current_version = crate::version::VERSION;
        let latest_version = self.get_latest_version().await?;

        if !self.is_newer_version(&latest_version, current_version) {
            println!(
                "{} Already up to date {}",
                style::success_symbol(),
                style::dim(format!("({})", current_version))
            );
            return Ok(ExitCode::Success);
        }

        if !elevated {
            println!("  Current: {}", style::dim(current_version));
            println!("  Latest:  {}", style::green(&latest_version));
        }

        // Fail before any download when the binary can't be replaced in place;
        // on a root-owned dir, re-run under sudo instead of aborting.
        let install_path = get_install_path()?;
        if let Err(e) = check_install_dir_writable(&install_path) {
            if !elevated
                && is_permission_denied(&e)
                && let Some(code) = reexec_with_sudo(&install_path, force)
            {
                return Ok(code);
            }
            return Err(e);
        }

        let binary_name = get_binary_name()?;
        let base_url = format!("{}/v{}", DOWNLOAD_BASE, latest_version);
        let binary_url = format!("{}/{}", base_url, binary_name);
        let sha256_url = format!("{}/{}.sha256", base_url, binary_name);
        let minisig_url = format!("{}/{}.minisig", base_url, binary_name);

        let expected_sha256 = self.fetch_sha256(&sha256_url, &binary_name).await?;
        let signature = self.fetch_text(&minisig_url).await.context(
            "Failed to fetch the update's signature (.minisig) — refusing to install unverified",
        )?;

        println!("{} Downloading update...", style::arrow_symbol());
        self.install_update(&binary_url, &expected_sha256, &signature, &latest_version)
            .await?;

        println!(
            "{} Updated to version {}",
            style::success_symbol(),
            latest_version
        );

        Ok(ExitCode::Success)
    }

    /// Fetches and transforms `models.dev/api.json`, writing the snapshot to the
    /// override path the limits cascade overlays. Returns the model count.
    async fn refresh_model_data(&self) -> Result<usize> {
        let bytes = self
            .fetch_capped(MODELS_DEV_API, MODELS_DEV_MAX_BYTES)
            .await
            .context("Failed to fetch models.dev")?;
        let api_json = String::from_utf8(bytes).context("models.dev returned invalid UTF-8")?;
        let (snapshot_json, count) = crate::services::model_data_sync::transform(&api_json)
            .context("Failed to transform models.dev data")?;

        let path = crate::services::model_metadata::override_model_limits_path()
            .context("Could not resolve the aivo config directory")?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        // Temp + rename so an interrupted write can't leave a truncated file.
        let tmp = path.with_file_name(format!("model_limits.json.{}.tmp", std::process::id()));
        tokio::fs::write(&tmp, snapshot_json.as_bytes())
            .await
            .with_context(|| format!("Failed to write {}", tmp.display()))?;
        if let Err(e) = tokio::fs::rename(&tmp, &path).await {
            tokio::fs::remove_file(&tmp).await.ok();
            return Err(e).with_context(|| format!("Failed to replace {}", path.display()));
        }
        Ok(count)
    }

    /// Downloads `url` into memory, enforcing a byte cap (via `Content-Length`
    /// up front, then while streaming).
    async fn fetch_capped(&self, url: &str, cap: u64) -> Result<Vec<u8>> {
        let mut response = self
            .client
            .get(url)
            .header("User-Agent", "aivo-cli")
            .send()
            .await
            .context("Request failed")?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("HTTP {}", response.status()));
        }
        if let Some(len) = response.content_length()
            && len > cap
        {
            return Err(anyhow::anyhow!("response too large ({len} bytes)"));
        }
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = response.chunk().await.context("Error reading response")? {
            if buf.len() as u64 + chunk.len() as u64 > cap {
                return Err(anyhow::anyhow!("response exceeded {cap} bytes"));
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(buf)
    }

    /// Fetches the latest version string from the R2-backed `/dl/latest` endpoint.
    async fn get_latest_version(&self) -> Result<String> {
        let url = format!("{}/latest", DOWNLOAD_BASE);
        let response = self
            .client
            .get(&url)
            .header("User-Agent", "aivo-cli")
            .send()
            .await
            .context("Failed to fetch latest version")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!(
                "Failed to fetch latest version: HTTP {}",
                status
            ));
        }

        let text = response
            .text()
            .await
            .context("Failed to read latest version response")?;
        let version = text.trim().trim_start_matches('v').to_string();
        if version.is_empty() {
            return Err(anyhow::anyhow!("Empty latest version response"));
        }
        if !version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        {
            return Err(anyhow::anyhow!("Invalid latest version: {}", version));
        }
        Ok(version)
    }

    async fn fetch_sha256(&self, url: &str, binary_name: &str) -> Result<String> {
        let response = self
            .client
            .get(url)
            .header("User-Agent", "aivo-cli")
            .send()
            .await
            .context("Failed to fetch checksum")?;

        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("Checksum download failed: HTTP {}", status));
        }

        let text = response
            .text()
            .await
            .context("Failed to read checksum response")?;
        parse_checksum_text(&text, binary_name)
            .ok_or_else(|| anyhow::anyhow!("Could not parse checksum for {}", binary_name))
    }

    /// Fetches a small text resource (e.g. the `.minisig` signature).
    async fn fetch_text(&self, url: &str) -> Result<String> {
        let response = self
            .client
            .get(url)
            .header("User-Agent", "aivo-cli")
            .send()
            .await
            .context("Request failed")?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow::anyhow!("HTTP {}", status));
        }
        response
            .text()
            .await
            .context("Failed to read response body")
    }

    async fn install_update(
        &self,
        download_url: &str,
        expected_sha256: &str,
        signature: &str,
        expected_version: &str,
    ) -> Result<()> {
        let exec_path = get_install_path()?;
        let tmp_path = exec_path.with_extension("tmp");

        let actual_sha256 = match self.download_to_file(download_url, &tmp_path).await {
            Ok(sha) => sha,
            Err(err) => {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(err).context("Failed to download update");
            }
        };

        // Authenticity first: a valid signature over the downloaded bytes proves
        // the release came from the holder of the CI secret key, not just that
        // the bytes match a hash the same host served.
        let bytes = match tokio::fs::read(&tmp_path).await {
            Ok(b) => b,
            Err(err) => {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(err).context("Failed to re-read downloaded update for verification");
            }
        };
        if let Err(err) = verify_minisign(MINISIGN_PUBKEY, &bytes, signature) {
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(err).context("Signature verification failed — refusing to install");
        }
        println!(
            "  {} {}",
            style::dim("Signature (Ed25519):"),
            style::green("verified")
        );

        if actual_sha256 != expected_sha256 {
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(anyhow::anyhow!(
                "Checksum verification failed for downloaded update"
            ));
        }
        println!(
            "  {} {}",
            style::dim("Checksum (SHA-256):"),
            style::green("verified")
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = std::fs::Permissions::from_mode(0o755);
            if let Err(e) = tokio::fs::set_permissions(&tmp_path, permissions).await {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(e.into());
            }
        }

        // Back up current binary before replacing (rename is O(1) on same filesystem)
        let backup_path = backup_path_for(&exec_path);
        let has_backup =
            exec_path.exists() && tokio::fs::rename(&exec_path, &backup_path).await.is_ok();
        if exec_path.exists() && !has_backup {
            eprintln!(
                "  {} Could not back up current binary",
                style::yellow("Warning:")
            );
        }

        if let Err(e) = tokio::fs::rename(&tmp_path, &exec_path).await {
            if has_backup {
                tokio::fs::rename(&backup_path, &exec_path).await.ok();
            }
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(e).with_context(|| format!("Failed to replace binary at {:?}", exec_path));
        }

        // Smoke test: the new binary must run AND report the advertised version.
        // The signature only proves "signed by us at some point" — without this
        // check a compromised host could replay an old, genuinely-signed release.
        let smoke_failure = match run_version_check(&exec_path) {
            Err(reason) => Some(format!("{}", reason)),
            Ok(reported) if !version_output_matches(&reported, expected_version) => Some(format!(
                "binary reports \"{}\", expected version {}",
                reported, expected_version
            )),
            Ok(_) => None,
        };
        if let Some(reason) = smoke_failure {
            eprintln!(
                "  {} New binary failed smoke test, rolling back...",
                style::red("Error:")
            );
            eprintln!("  {}", style::dim(reason));
            if has_backup {
                match tokio::fs::rename(&backup_path, &exec_path).await {
                    Ok(()) => {
                        return Err(anyhow::anyhow!(
                            "Updated binary failed to run. Previous version has been restored."
                        ));
                    }
                    Err(e) => {
                        return Err(anyhow::anyhow!(
                            "Updated binary failed to run and rollback failed: {}. \
                             Manual restore: cp {} {}",
                            e,
                            backup_path.display(),
                            exec_path.display()
                        ));
                    }
                }
            }
            return Err(anyhow::anyhow!(
                "Updated binary failed to run and no backup was available."
            ));
        }

        println!("  {} {}", style::dim("Installed to:"), exec_path.display());
        if has_backup {
            println!(
                "  {} {}",
                style::dim("Backup saved:"),
                backup_path.display()
            );
            println!(
                "  {} {}",
                style::dim("Rollback with:"),
                style::green("aivo update --rollback")
            );
        }

        Ok(())
    }

    /// Streams a binary download into `tmp_path`, computes its SHA-256 on the
    /// fly, and renders progress. Uses `download_client` (no total deadline,
    /// only a per-read stall timeout) so slow connections that keep delivering
    /// bytes finish instead of being killed mid-stream.
    async fn download_to_file(&self, url: &str, tmp_path: &Path) -> Result<String> {
        let mut response = self
            .download_client
            .get(url)
            .header("User-Agent", "aivo-cli")
            .send()
            .await
            .context("Failed to start download")?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Download failed: HTTP {}",
                response.status()
            ));
        }

        let total_size = response.content_length().unwrap_or(0);

        let mut hasher = Sha256::new();
        let mut downloaded: u64 = 0;
        let mut file = tokio::fs::File::create(tmp_path)
            .await
            .with_context(|| format!("Failed to create temporary file at {:?}", tmp_path))?;

        while let Some(chunk) = response
            .chunk()
            .await
            .context("Error reading download stream")?
        {
            hasher.update(&chunk);
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write to temporary file at {:?}", tmp_path))?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let mb = downloaded as f64 / 1024.0 / 1024.0;
                let total_mb = total_size as f64 / 1024.0 / 1024.0;
                let percent = (downloaded as f64 / total_size as f64) * 100.0;
                eprint!(
                    "\r  {} {:.1}/{:.1} MB ({:.0}%)",
                    style::dim("Downloading:"),
                    mb,
                    total_mb,
                    percent
                );
            }
        }
        file.flush().await?;
        drop(file); // Close write FD before rename/exec to avoid ETXTBSY on Linux
        if total_size > 0 {
            eprintln!(); // newline after progress
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn is_newer_version(&self, latest: &str, current: &str) -> bool {
        crate::services::update_check::is_newer_version(latest, current)
    }

    fn handle_error(&self, error: anyhow::Error) {
        eprintln!("{} {:#}", style::red("Error:"), error);
        eprintln!();
        if is_permission_denied(&error) {
            #[cfg(not(windows))]
            if resolve_command_path("sudo").is_some() {
                // Absolute path: sudoers `secure_path` often omits
                // /usr/local/bin, so bare `sudo aivo` is "command not found".
                let exe = get_install_path()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "aivo".to_string());
                eprintln!(
                    "{} Re-run with elevated permissions:",
                    style::yellow("Suggestion:")
                );
                eprintln!("  {}", style::green(format!("sudo {exe} update")));
            } else {
                eprintln!(
                    "{} Reinstall to a directory you can write to:",
                    style::yellow("Suggestion:")
                );
                eprintln!(
                    "  {}",
                    style::green(
                        "curl -fsSL https://getaivo.dev/install.sh | AIVO_INSTALL_DIR=\"$HOME/.local/bin\" bash"
                    )
                );
            }
            #[cfg(windows)]
            eprintln!(
                "{} Re-run {} from an elevated (Administrator) terminal.",
                style::yellow("Suggestion:"),
                style::green("aivo update")
            );
        } else {
            eprintln!(
                "{} Check your internet connection and try again.",
                style::yellow("Suggestion:")
            );
        }
    }

    fn update_via_homebrew(&self) -> ExitCode {
        println!("{} Updating via Homebrew...", style::arrow_symbol());

        // Run brew update first to fetch latest formulas (ignore errors)
        let _ = Command::new("brew").args(["update", "--quiet"]).status();

        // Then upgrade aivo (--overwrite to handle symlink conflicts)
        println!("{} Upgrading aivo...", style::arrow_symbol());
        match Command::new("brew")
            .args(["upgrade", "--overwrite", "aivo"])
            .status()
        {
            Ok(status) if status.success() => ExitCode::Success,
            Ok(_) => ExitCode::Success,
            Err(e) => {
                eprintln!("{} brew upgrade failed: {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }
}

/// Verifies `data` against a detached minisign `signature` using `pubkey_b64`
/// (the base64 line of an `aivo.pub`). Any parse or verification failure is an
/// error — the caller treats that as "do not install".
fn verify_minisign(pubkey_b64: &str, data: &[u8], signature: &str) -> Result<()> {
    use minisign_verify::{PublicKey, Signature};

    let public_key = PublicKey::from_base64(pubkey_b64.trim()).map_err(|e| {
        anyhow::anyhow!(
            "this aivo build has no valid update-signing public key ({e}); \
             reinstall from https://getaivo.dev or your package manager"
        )
    })?;
    let signature = Signature::decode(signature.trim())
        .map_err(|e| anyhow::anyhow!("malformed signature: {e}"))?;
    public_key
        .verify(data, &signature, false)
        .map_err(|e| anyhow::anyhow!("signature does not match the downloaded binary: {e}"))
}

fn normalize_sha256(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.len() != 64 || !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

fn parse_checksum_text(text: &str, binary_name: &str) -> Option<String> {
    let mut fallback_hash: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some((left, right)) = line.split_once(" = ")
            && left.starts_with("SHA256 (")
            && left.ends_with(')')
            && (left.contains(binary_name) || binary_name.is_empty())
            && let Some(hash) = normalize_sha256(right)
        {
            return Some(hash);
        }

        let mut parts = line.split_whitespace();
        if let Some(first) = parts.next()
            && let Some(hash) = normalize_sha256(first)
        {
            let remainder = line[first.len()..].trim_start();
            let cleaned_remainder = remainder.trim_start_matches('*').trim_start();
            if cleaned_remainder.is_empty() {
                fallback_hash = Some(hash);
            } else if cleaned_remainder.ends_with(binary_name) || cleaned_remainder == binary_name {
                return Some(hash);
            }
        }
    }

    fallback_hash
}

fn get_binary_name() -> Result<String> {
    let platform = env::consts::OS;
    let arch = env::consts::ARCH;

    let name = match (platform, arch) {
        ("macos", "aarch64") => "aivo-darwin-arm64",
        ("macos", "x86_64") => "aivo-darwin-x64",
        ("linux", "aarch64") => "aivo-linux-arm64",
        ("linux", "x86_64") => "aivo-linux-x64",
        ("windows", "x86_64") => "aivo-windows-x64.exe",
        ("windows", "aarch64") => "aivo-windows-arm64.exe",
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported platform: {}-{}",
                platform,
                arch
            ));
        }
    };

    Ok(name.to_string())
}

/// Restores the previous binary from the backup created during the last update.
/// Free function — no HTTP client needed for a local-only operation.
pub async fn execute_rollback() -> ExitCode {
    let exec_path = match get_install_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{} {}", style::red("Error:"), e);
            return ExitCode::UserError;
        }
    };
    let backup_path = backup_path_for(&exec_path);

    match tokio::fs::rename(&backup_path, &exec_path).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "{} No backup found at {}",
                style::red("Error:"),
                backup_path.display()
            );
            eprintln!(
                "  {}",
                style::dim("A backup is created automatically during `aivo update`.")
            );
            return ExitCode::UserError;
        }
        Err(e) => {
            eprintln!("{} Failed to restore backup: {}", style::red("Error:"), e);
            eprintln!(
                "  {} cp {} {}",
                style::dim("You can restore manually:"),
                backup_path.display(),
                exec_path.display()
            );
            return ExitCode::UserError;
        }
    }

    let version = run_version_check(&exec_path).ok().unwrap_or_default();
    println!(
        "{} Rolled back to previous version{}",
        style::success_symbol(),
        if version.is_empty() {
            String::new()
        } else {
            format!(" ({})", version)
        }
    );

    ExitCode::Success
}

fn run_version_check(exec_path: &Path) -> Result<String> {
    let output = Command::new(exec_path)
        .arg("--version")
        .output()
        .context("Failed to execute binary")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.trim().is_empty() {
            anyhow::bail!("Exit code: {}", output.status);
        }
        anyhow::bail!("Exit code: {}, stderr: {}", output.status, stderr.trim());
    }

    let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
    anyhow::ensure!(!version.is_empty(), "Binary produced no version output");
    Ok(version)
}

/// True when `--version` output (e.g. "aivo v0.28.2") names `expected`.
fn version_output_matches(output: &str, expected: &str) -> bool {
    let expected = expected.trim_start_matches('v');
    output
        .split_whitespace()
        .any(|tok| tok.trim_start_matches('v') == expected)
}

fn backup_path_for(exec_path: &Path) -> PathBuf {
    exec_path.with_extension("previous")
}

fn get_install_path() -> Result<PathBuf> {
    if let Ok(path) = env::var("AIVO_PATH") {
        return Ok(PathBuf::from(path));
    }
    let current_exe = env::current_exe()?;
    Ok(current_exe)
}

/// Pre-flight: replacing the binary needs write access to its directory no
/// matter where the download is staged, so prove it by opening the actual
/// `aivo.tmp` staging path (also sweeping up a stale one) before any bytes
/// are downloaded — e.g. a root-owned /usr/local/bin fails here, actionably.
fn check_install_dir_writable(exec_path: &Path) -> Result<()> {
    let probe = exec_path.with_extension("tmp");
    match std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&probe)
    {
        Ok(file) => {
            drop(file);
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(e) => {
            let dir = exec_path.parent().unwrap_or_else(|| Path::new("."));
            Err(anyhow::Error::new(e).context(format!("No write access to {}", dir.display())))
        }
    }
}

/// True when any cause in the chain is an EACCES-style I/O error.
fn is_permission_denied(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied)
    })
}

#[cfg(not(windows))]
fn resolve_command_path(program: &str) -> Option<PathBuf> {
    let dirs = collect_path_dirs();
    find_in_dirs(program, &dirs)
}

/// Elevation gate, split out for testing. `AIVO_PATH` vetoes because sudo's env
/// reset would drop it, sending the child at the wrong binary. `sudo` presence
/// is checked lazily by the caller so a failed gate skips the PATH scan.
#[cfg(not(windows))]
fn should_attempt_elevation(is_root: bool, has_aivo_path: bool, is_tty: bool) -> bool {
    !is_root && !has_aivo_path && is_tty
}

/// Re-run `aivo update` under `sudo` for a root-owned install dir, mirroring
/// install.sh's `sudo mv`. `Some(code)` when it elevated, `None` when it can't —
/// the caller then falls back to the permission-denied error. `!is_root` doubles
/// as the loop guard: the elevated child is root and can't re-enter here.
#[cfg(not(windows))]
fn reexec_with_sudo(exe: &Path, force: bool) -> Option<ExitCode> {
    if !should_attempt_elevation(
        unsafe { libc::geteuid() } == 0,
        env::var_os("AIVO_PATH").is_some(),
        std::io::stdin().is_terminal(),
    ) {
        return None;
    }
    let sudo = resolve_command_path("sudo")?;
    let dir = exe.parent().unwrap_or_else(|| Path::new("."));
    println!(
        "  {}",
        style::dim(format!("needs sudo to write {}", dir.display()))
    );

    match Command::new(sudo)
        .args(sudo_update_args(exe, force))
        .status()
    {
        // The child is another aivo process; its exit code already IS an ExitCode.
        Ok(status) => Some(match status.code() {
            Some(0) => ExitCode::Success,
            Some(n) => ExitCode::ToolExit(n),
            None => ExitCode::UserError,
        }),
        Err(e) => {
            eprintln!(
                "{} Failed to elevate with sudo: {}",
                style::red("Error:"),
                e
            );
            Some(ExitCode::UserError)
        }
    }
}

#[cfg(windows)]
fn reexec_with_sudo(_exe: &Path, _force: bool) -> Option<ExitCode> {
    None
}

/// `--sudo-elevated` marks the re-run: suppresses the duplicate preamble and
/// blocks re-elevation.
#[cfg(not(windows))]
fn sudo_update_args(exe: &Path, force: bool) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;
    let mut argv: Vec<OsString> = vec![exe.into(), "update".into(), "--sudo-elevated".into()];
    if force {
        argv.push("--force".into());
    }
    argv
}

fn normalize_install_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("//?/")
        .to_ascii_lowercase()
}

enum PackageManager {
    Homebrew,
    Cargo,
}

struct ManagedInstall {
    kind: PackageManager,
    name: &'static str,
    upgrade_command: &'static str,
}

/// Detects whether the binary is managed by a package manager that should update
/// itself (Homebrew, Cargo), based on its path. Returns None for a direct download,
/// an npm install (which self-updates natively — `npm install -g` can't replace the
/// in-use global binary on Windows), or when AIVO_PATH is set.
fn detect_managed_install(install_path: &Path) -> Option<ManagedInstall> {
    // If AIVO_PATH is explicitly set, user opted into this path — skip detection
    if env::var("AIVO_PATH").is_ok() {
        return None;
    }

    let path_str = normalize_install_path(install_path);

    // An npm install self-updates natively, so it's not a "managed" install. This
    // must come before the Homebrew check: an npm global install under Homebrew's
    // node prefix (e.g. /opt/homebrew/lib/node_modules/@yuanchuan/aivo/...) matches
    // the `/homebrew/` substring below and would otherwise misroute to `brew upgrade`.
    if path_str.contains("/node_modules/") {
        return None;
    }

    // Homebrew: /opt/homebrew/Cellar/..., /usr/local/Cellar/..., /home/linuxbrew/.linuxbrew/Cellar/...
    if path_str.contains("/cellar/") || path_str.contains("/homebrew/") {
        return Some(ManagedInstall {
            kind: PackageManager::Homebrew,
            name: "Homebrew",
            upgrade_command: "brew upgrade aivo",
        });
    }

    // Cargo: ~/.cargo/bin/aivo
    if path_str.contains("/.cargo/bin/") {
        return Some(ManagedInstall {
            kind: PackageManager::Cargo,
            name: "Cargo",
            upgrade_command: "cargo install aivo",
        });
    }

    None
}

/// Upgrade command for the current install: `brew`/`cargo` when detected, else `aivo update`.
pub(crate) fn upgrade_command_for_current_install() -> &'static str {
    get_install_path()
        .ok()
        .and_then(|p| detect_managed_install(&p))
        .map(|m| m.upgrade_command)
        .unwrap_or("aivo update")
}

/// The clean npm launcher shim (`bin/aivo.js`), embedded at build time so the
/// native binary can repair a stale one in place. Kept byte-identical to the
/// shipped file via `include_str!`. The clean shim just execs this binary for
/// every subcommand — no npm.
#[cfg(any(windows, test))]
const CLEAN_NPM_SHIM: &str = include_str!("../../npm/bin/aivo.js");

/// True if `content` is a pre-0.31.1 npm shim that hijacks `aivo update` into
/// `npm install -g` on Windows. That shim pulls in `lib/update`'s
/// `shouldDelegateWindowsNpmUpdate`; the clean shim has no such marker.
#[cfg(any(windows, test))]
fn shim_is_stale(content: &str) -> bool {
    content.contains("shouldDelegateWindowsNpmUpdate")
}

/// Heal a pre-0.31.1 npm launcher shim that intercepts `aivo update` and runs
/// `npm install -g` (Windows only). The old `bin/aivo.js` can't replace itself —
/// npm can't overwrite the in-use file — and bare `aivo update` never reaches
/// this binary, so it loops through npm forever. But every OTHER command
/// (`aivo update --force`, `aivo --version`, `aivo code`, …) DOES run this
/// binary, so we rewrite the stale shim to the clean one from here. Best-effort
/// and silent: any failure just leaves `aivo update --force` as the escape hatch.
#[cfg(windows)]
pub(crate) fn repair_npm_shim() {
    if let Ok(exe) = env::current_exe() {
        repair_npm_shim_at(&exe);
    }
}

/// Testable core of [`repair_npm_shim`]: given the running binary's path, rewrite
/// a stale sibling `bin/aivo.js` to the clean shim. Split out so it can be tested
/// against a fake package layout without depending on `current_exe()`.
#[cfg(any(windows, test))]
fn repair_npm_shim_at(exe: &Path) {
    // Only touch an actual npm install of our package.
    if !normalize_install_path(exe).contains("/node_modules/") {
        return;
    }
    // Package layout: `<pkg>/native/aivo.exe` alongside `<pkg>/bin/aivo.js`.
    let Some(shim) = exe
        .parent()
        .and_then(|native_dir| native_dir.parent())
        .map(|pkg_root| pkg_root.join("bin").join("aivo.js"))
    else {
        return;
    };
    let Ok(current) = std::fs::read_to_string(&shim) else {
        return;
    };
    if !shim_is_stale(&current) {
        return;
    }
    // Write a sibling temp then rename over the shim, so a crash mid-write can't
    // leave a truncated `aivo.js` that fails to launch.
    let tmp = shim.with_file_name(format!("aivo.js.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, CLEAN_NPM_SHIM).is_ok() && std::fs::rename(&tmp, &shim).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A self-contained minisign test vector (generated offline; not the
    // production key). Signature is over the exact bytes of TEST_PAYLOAD.
    const TEST_PUBKEY: &str = "RWQGB+jqLf9L8qddcLvjpHkTbNnCoF959l7GeTTqTUCCiGiFIuxEOQ+F";
    const TEST_PAYLOAD: &[u8] = b"aivo-update-test-payload";
    const TEST_SIG: &str = "untrusted comment: signature from minisign secret key\n\
RUQGB+jqLf9L8gqLIGqP9w+H8OnkiCS9gNcFNLx4zujoZKUsLJGVVSfgVLqC4HoozPwBh0JN4uRbI2S+b7PfgopEw8LigYxcdw0=\n\
trusted comment: timestamp:1781191999\tfile:msg.bin\thashed\n\
Pi5pASxJ8C5JIeBSzqSS09rJdnjExlwHgQeJ1MRy0Q5oZAhtB+TFk65XQbkSwv8hbpGICsVCjCq/3cmuWTyQCA==\n";

    #[test]
    fn verify_minisign_accepts_valid_signature() {
        assert!(verify_minisign(TEST_PUBKEY, TEST_PAYLOAD, TEST_SIG).is_ok());
    }

    #[test]
    fn verify_minisign_rejects_tampered_payload() {
        let mut tampered = TEST_PAYLOAD.to_vec();
        tampered[0] ^= 0x01;
        assert!(verify_minisign(TEST_PUBKEY, &tampered, TEST_SIG).is_err());
    }

    #[test]
    fn verify_minisign_rejects_wrong_key() {
        // A real, different minisign key must not verify this signature.
        let other = "RWS3ZL6im4hk6qKNxqj4pM0CUc/N/kpnr8Q6stytqeRMJKgaeUeIEhu2";
        assert!(verify_minisign(other, TEST_PAYLOAD, TEST_SIG).is_err());
    }

    #[test]
    fn production_pubkey_parses_and_rejects_foreign_signature() {
        // The embedded production key must be a well-formed minisign key (so it
        // can verify real releases) yet reject a signature made by any other key.
        assert!(
            minisign_verify::PublicKey::from_base64(MINISIGN_PUBKEY).is_ok(),
            "embedded MINISIGN_PUBKEY must be a valid minisign public key"
        );
        assert!(verify_minisign(MINISIGN_PUBKEY, TEST_PAYLOAD, TEST_SIG).is_err());
    }

    #[test]
    fn verify_minisign_rejects_garbage_signature() {
        assert!(verify_minisign(TEST_PUBKEY, TEST_PAYLOAD, "not a signature").is_err());
    }

    #[test]
    fn test_version_output_matches() {
        assert!(version_output_matches("aivo v0.28.2", "0.28.2"));
        assert!(version_output_matches("aivo 0.28.2", "0.28.2"));
        assert!(version_output_matches("0.28.2", "0.28.2"));
        assert!(version_output_matches("aivo v0.28.2", "v0.28.2"));
        assert!(!version_output_matches("aivo v0.20.0", "0.28.2"));
        assert!(!version_output_matches("aivo v0.28.21", "0.28.2"));
        assert!(!version_output_matches("", "0.28.2"));
    }

    #[test]
    fn test_is_newer_version() {
        let cmd = UpdateCommand::new().unwrap();

        assert!(cmd.is_newer_version("1.1.0", "1.0.0"));
        assert!(cmd.is_newer_version("2.0.0", "1.0.0"));
        assert!(cmd.is_newer_version("1.0.1", "1.0.0"));
        assert!(!cmd.is_newer_version("1.0.0", "1.0.0"));
        assert!(!cmd.is_newer_version("0.9.0", "1.0.0"));
        assert!(!cmd.is_newer_version("1.0.0", "1.0.1"));
    }

    #[test]
    fn test_parse_version() {
        let cmd = UpdateCommand::new().unwrap();

        assert!(cmd.is_newer_version("v1.1.0", "v1.0.0"));
        assert!(cmd.is_newer_version("1.1.0", "v1.0.0"));
    }

    #[test]
    fn test_prerelease_version() {
        let cmd = UpdateCommand::new().unwrap();

        assert!(cmd.is_newer_version("2.0.0", "2.0.0-rc1"));
        assert!(cmd.is_newer_version("2.0.0", "2.0.0-beta.1"));
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0"));
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0-rc1"));
        // Higher version wins regardless of pre-release.
        assert!(cmd.is_newer_version("2.1.0-rc1", "2.0.0"));
        assert!(cmd.is_newer_version("2.1.0", "2.0.0-rc1"));
    }

    #[test]
    fn test_parse_checksum_text_variants() {
        let artifact = "aivo-darwin-arm64";
        let plain = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n";
        assert_eq!(
            parse_checksum_text(plain, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );

        let with_name =
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  aivo-darwin-arm64";
        assert_eq!(
            parse_checksum_text(with_name, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );

        let bsd = "SHA256 (aivo-darwin-arm64) = 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        assert_eq!(
            parse_checksum_text(bsd, artifact),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string())
        );
    }

    // npm installs self-update natively (not via `npm install -g`, which can't
    // replace the in-use global binary on Windows), so npm is NOT a managed install:
    // detection returns None and `update` takes the native download path.
    #[test]
    fn test_detect_npm_global() {
        let path = Path::new("/opt/homebrew/lib/node_modules/@yuanchuan/aivo/native/aivo");
        assert!(detect_managed_install(path).is_none());
    }

    #[test]
    fn test_detect_npm_nvm() {
        let path = Path::new(
            "/Users/user/.nvm/versions/node/v22.0.0/lib/node_modules/@yuanchuan/aivo/native/aivo",
        );
        assert!(detect_managed_install(path).is_none());
    }

    #[test]
    fn test_detect_npm_windows_path() {
        let path = Path::new(
            r"C:\Users\user\AppData\Roaming\npm\node_modules\@yuanchuan\aivo\native\aivo.exe",
        );
        assert!(detect_managed_install(path).is_none());
    }

    #[test]
    fn test_detect_homebrew_cellar_arm() {
        let path = Path::new("/opt/homebrew/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "Homebrew");
        assert_eq!(m.upgrade_command, "brew upgrade aivo");
    }

    #[test]
    fn test_detect_homebrew_cellar_intel() {
        let path = Path::new("/usr/local/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_homebrew_linuxbrew() {
        let path = Path::new("/home/linuxbrew/.linuxbrew/Cellar/aivo/0.4.3/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_homebrew_opt_path() {
        let path = Path::new("/opt/homebrew/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Homebrew");
    }

    #[test]
    fn test_detect_cargo_install() {
        let path = Path::new("/Users/user/.cargo/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "Cargo");
        assert_eq!(m.upgrade_command, "cargo install aivo");
    }

    #[test]
    fn test_detect_cargo_windows_path() {
        let path = Path::new(r"C:\Users\user\.cargo\bin\aivo.exe");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "Cargo");
    }

    #[test]
    fn test_detect_direct_download() {
        let path = Path::new("/usr/local/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_none());
    }

    #[test]
    fn clean_npm_shim_is_not_stale() {
        // The embedded clean shim must never trip the staleness check, or repair
        // would rewrite it on every startup.
        assert!(!shim_is_stale(CLEAN_NPM_SHIM));
        // Sanity: it really is the exec-native shim.
        assert!(CLEAN_NPM_SHIM.contains("getInstalledBinaryPath"));
    }

    #[test]
    fn old_intercepting_shim_is_stale() {
        let old = r#"const { shouldDelegateWindowsNpmUpdate } = require("../lib/update");"#;
        assert!(shim_is_stale(old));
        // A clean shim body (no npm-delegation marker) is left alone.
        assert!(!shim_is_stale(
            r#"const { getInstalledBinaryPath } = require("../lib/paths");"#
        ));
    }

    #[test]
    fn repair_rewrites_only_a_stale_npm_shim() {
        let dir = tempfile::tempdir().unwrap();
        // Fake npm layout: <pkg>/native/aivo.exe alongside <pkg>/bin/aivo.js.
        let pkg = dir.path().join("node_modules/@yuanchuan/aivo");
        std::fs::create_dir_all(pkg.join("bin")).unwrap();
        std::fs::create_dir_all(pkg.join("native")).unwrap();
        let shim = pkg.join("bin").join("aivo.js");
        let exe = pkg.join("native").join("aivo.exe");
        std::fs::write(&exe, b"binary").unwrap();

        // A pre-0.31.1 intercepting shim is rewritten to the clean one.
        std::fs::write(
            &shim,
            "require(\"../lib/update\").shouldDelegateWindowsNpmUpdate();",
        )
        .unwrap();
        repair_npm_shim_at(&exe);
        assert_eq!(std::fs::read_to_string(&shim).unwrap(), CLEAN_NPM_SHIM);

        // Idempotent: a second pass leaves the now-clean shim untouched.
        repair_npm_shim_at(&exe);
        assert_eq!(std::fs::read_to_string(&shim).unwrap(), CLEAN_NPM_SHIM);

        // No temp file left behind in bin/.
        let leftover = std::fs::read_dir(pkg.join("bin"))
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| e.file_name().to_string_lossy().contains(".tmp"));
        assert!(!leftover, "repair left a temp file behind");
    }

    #[test]
    fn repair_ignores_non_npm_install() {
        let dir = tempfile::tempdir().unwrap();
        // A direct install (no node_modules in the path).
        std::fs::create_dir_all(dir.path().join("bin")).unwrap();
        std::fs::create_dir_all(dir.path().join("native")).unwrap();
        let shim = dir.path().join("bin").join("aivo.js");
        let exe = dir.path().join("native").join("aivo");
        std::fs::write(&exe, b"binary").unwrap();
        let stale = "shouldDelegateWindowsNpmUpdate marker";
        std::fs::write(&shim, stale).unwrap();

        repair_npm_shim_at(&exe);

        // Untouched — not an npm install, so not ours to heal.
        assert_eq!(std::fs::read_to_string(&shim).unwrap(), stale);
    }

    #[test]
    fn test_detect_custom_path() {
        let path = Path::new("/home/user/bin/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_none());
    }

    #[test]
    fn test_normalize_install_path_strips_verbatim_prefix() {
        let path = Path::new(
            r"\\?\C:\Users\User\AppData\Roaming\npm\node_modules\@yuanchuan\aivo\aivo.exe",
        );
        assert_eq!(
            normalize_install_path(path),
            "c:/users/user/appdata/roaming/npm/node_modules/@yuanchuan/aivo/aivo.exe"
        );
    }

    #[test]
    fn is_permission_denied_detects_io_cause_through_context() {
        let err = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "denied",
        ))
        .context("Failed to create temporary file");
        assert!(is_permission_denied(&err));

        assert!(!is_permission_denied(&anyhow::anyhow!("HTTP 404")));
        let other_io =
            anyhow::Error::new(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"));
        assert!(!is_permission_denied(&other_io));
    }

    #[cfg(not(windows))]
    #[test]
    fn elevation_gate_requires_all_conditions() {
        assert!(should_attempt_elevation(false, false, true));
        assert!(!should_attempt_elevation(true, false, true)); // already root
        assert!(!should_attempt_elevation(false, true, true)); // AIVO_PATH override
        assert!(!should_attempt_elevation(false, false, false)); // no terminal
    }

    #[cfg(not(windows))]
    #[test]
    fn sudo_argv_reruns_update_with_marker() {
        let exe = Path::new("/usr/local/bin/aivo");
        assert_eq!(
            sudo_update_args(exe, false),
            vec![
                std::ffi::OsString::from("/usr/local/bin/aivo"),
                "update".into(),
                "--sudo-elevated".into(),
            ]
        );
        assert_eq!(
            sudo_update_args(exe, true),
            vec![
                std::ffi::OsString::from("/usr/local/bin/aivo"),
                "update".into(),
                "--sudo-elevated".into(),
                "--force".into(),
            ]
        );
    }

    #[test]
    fn preflight_accepts_writable_dir_and_cleans_probe() {
        let dir = tempfile::tempdir().unwrap();
        let exec_path = dir.path().join("aivo");
        check_install_dir_writable(&exec_path).unwrap();
        assert!(!exec_path.with_extension("tmp").exists());
    }

    #[test]
    fn preflight_removes_stale_staging_file() {
        let dir = tempfile::tempdir().unwrap();
        let exec_path = dir.path().join("aivo");
        let stale = exec_path.with_extension("tmp");
        std::fs::write(&stale, b"stale").unwrap();
        check_install_dir_writable(&exec_path).unwrap();
        assert!(!stale.exists());
    }

    #[cfg(unix)]
    #[test]
    fn preflight_rejects_unwritable_dir() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let exec_path = dir.path().join("aivo");
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

        let result = check_install_dir_writable(&exec_path);

        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();

        // Root bypasses directory permissions; only assert when the probe failed.
        if let Err(err) = result {
            assert!(is_permission_denied(&err), "unexpected error: {err:#}");
        }
    }

    #[test]
    fn test_backup_path_unix() {
        let exec = Path::new("/usr/local/bin/aivo");
        assert_eq!(
            backup_path_for(exec),
            PathBuf::from("/usr/local/bin/aivo.previous")
        );
    }

    #[test]
    fn test_backup_path_windows() {
        let exec = Path::new(r"C:\Users\user\bin\aivo.exe");
        assert_eq!(
            backup_path_for(exec),
            PathBuf::from(r"C:\Users\user\bin\aivo.previous")
        );
    }
}
