/**
 * UpdateCommand handler for CLI self-update functionality.
 */
use std::env;
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
const NPM_UPDATE_COMMAND: &str = "npm install -g @yuanchuan/aivo@latest";
#[cfg(not(windows))]
const NPM_UPDATE_ARGS: [&str; 3] = ["install", "-g", "@yuanchuan/aivo@latest"];

pub struct UpdateCommand {
    client: Client,
    /// Separate client for binary downloads: no total deadline, only a
    /// per-read timeout so slow-but-progressing connections finish.
    download_client: Client,
}

impl UpdateCommand {
    /// Shows usage information for the update command
    pub fn print_help() {
        println!("{} aivo update [OPTIONS]", style::bold("Usage:"));
        println!();
        println!(
            "{}",
            style::dim("Update the CLI tool to the latest version.")
        );
        println!(
            "{}",
            style::dim("Delegates to Homebrew or npm when installed via those package managers.")
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
        print_opt(
            "-f, --force",
            "Force update even if installed via a package manager",
        );
        print_opt(
            "--rollback",
            "Restore the previous version from the last update backup",
        );
        println!();
        println!("{}", style::bold("Examples:"));
        println!("  {}", style::dim("aivo update"));
        println!("  {}", style::dim("aivo update --force"));
        println!("  {}", style::dim("aivo update --rollback"));
    }

    /// Creates a new UpdateCommand instance
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

    /// Executes the update command
    pub async fn execute(&self, force: bool) -> ExitCode {
        match self.execute_internal(force).await {
            Ok(code) => code,
            Err(e) => {
                self.handle_error(e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, force: bool) -> Result<ExitCode> {
        // Check for package-manager-managed installations
        if !force {
            let install_path = get_install_path()?;
            if let Some(manager) = detect_managed_install(&install_path) {
                match manager.kind {
                    PackageManager::Homebrew => {
                        return Ok(self.update_via_homebrew());
                    }
                    PackageManager::Npm => {
                        return self.update_via_npm(&manager);
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

        println!("{} Checking for updates...", style::arrow_symbol());

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

        println!("  Current: {}", style::dim(current_version));
        println!("  Latest:  {}", style::green(&latest_version));

        let binary_name = get_binary_name()?;
        let base_url = format!("{}/v{}", DOWNLOAD_BASE, latest_version);
        let binary_url = format!("{}/{}", base_url, binary_name);
        let sha256_url = format!("{}/{}.sha256", base_url, binary_name);

        let expected_sha256 = self.fetch_sha256(&sha256_url, &binary_name).await?;

        println!("{} Downloading update...", style::arrow_symbol());
        self.install_update(&binary_url, &expected_sha256).await?;

        println!(
            "{} Updated to version {}",
            style::success_symbol(),
            latest_version
        );

        Ok(ExitCode::Success)
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

    /// Downloads and installs the update
    async fn install_update(&self, download_url: &str, expected_sha256: &str) -> Result<()> {
        let exec_path = get_install_path()?;
        let tmp_path = exec_path.with_extension("tmp");

        let actual_sha256 = match self.download_to_file(download_url, &tmp_path).await {
            Ok(sha) => sha,
            Err(err) => {
                tokio::fs::remove_file(&tmp_path).await.ok();
                return Err(err).context("Failed to download update");
            }
        };

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

        // Make executable (Unix only)
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

        // Replace with the new binary
        if let Err(e) = tokio::fs::rename(&tmp_path, &exec_path).await {
            // Restore backup if the replace failed
            if has_backup {
                tokio::fs::rename(&backup_path, &exec_path).await.ok();
            }
            tokio::fs::remove_file(&tmp_path).await.ok();
            return Err(e).with_context(|| format!("Failed to replace binary at {:?}", exec_path));
        }

        // Smoke test: verify the new binary can run
        if let Err(reason) = run_version_check(&exec_path) {
            eprintln!(
                "  {} New binary failed smoke test, rolling back...",
                style::red("Error:")
            );
            eprintln!("  {}", style::dim(format!("{}", reason)));
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

    /// Compares two semantic version strings.
    /// Strips pre-release suffixes (e.g. -rc1, -beta.1) before comparing.
    /// A pre-release version is considered older than its release counterpart.
    fn is_newer_version(&self, latest: &str, current: &str) -> bool {
        let parse_version = |version: &str| -> (Vec<u32>, bool) {
            let cleaned = version.trim_start_matches('v');
            // Split off pre-release suffix at the first hyphen
            let (version_str, has_prerelease) = match cleaned.split_once('-') {
                Some((v, _)) => (v, true),
                None => (cleaned, false),
            };
            let parts = version_str
                .split('.')
                .filter_map(|part| part.parse::<u32>().ok())
                .collect();
            (parts, has_prerelease)
        };

        let (latest_parts, latest_pre) = parse_version(latest);
        let (current_parts, current_pre) = parse_version(current);

        let max_len = latest_parts.len().max(current_parts.len());

        for i in 0..max_len {
            let latest_part = latest_parts.get(i).copied().unwrap_or(0);
            let current_part = current_parts.get(i).copied().unwrap_or(0);

            if latest_part > current_part {
                return true;
            }
            if latest_part < current_part {
                return false;
            }
        }

        // Same numeric version: release is newer than pre-release
        // e.g. "2.0.0" is newer than "2.0.0-rc1"
        if current_pre && !latest_pre {
            return true;
        }

        false
    }

    /// Handles errors
    fn handle_error(&self, error: anyhow::Error) {
        eprintln!("{} {:#}", style::red("Error:"), error);
        eprintln!();
        eprintln!(
            "{} Check your internet connection and try again.",
            style::yellow("Suggestion:")
        );
    }

    /// Delegates update to Homebrew
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

    fn update_via_npm(&self, manager: &ManagedInstall) -> Result<ExitCode> {
        #[cfg(windows)]
        {
            eprintln!(
                "{} Windows npm installs are updated by the npm shim, not by aivo.exe directly.",
                style::yellow("Warning:")
            );
            eprintln!("  {} {}", style::dim("Run:"), style::green("aivo update"));
            eprintln!(
                "  {} {}",
                style::dim("Or repair with:"),
                style::green(manager.upgrade_command)
            );
            Ok(ExitCode::UserError)
        }

        #[cfg(not(windows))]
        let npm_path = resolve_command_path("npm").ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find npm on PATH. Run this command manually: {}",
                manager.upgrade_command
            )
        })?;

        #[cfg(not(windows))]
        {
            println!("{} Updating via npm...", style::arrow_symbol());
            println!(
                "  {} {}",
                style::dim("Running:"),
                style::green(manager.upgrade_command)
            );

            let status = Command::new(&npm_path)
                .args(NPM_UPDATE_ARGS)
                .status()
                .with_context(|| format!("Failed to launch npm at {}", npm_path.display()))?;

            Ok(if status.success() {
                ExitCode::Success
            } else {
                ExitCode::UserError
            })
        }
    }
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

/// Gets the expected binary asset name for the current platform/arch
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

#[cfg(not(windows))]
fn resolve_command_path(program: &str) -> Option<PathBuf> {
    let dirs = collect_path_dirs();
    find_in_dirs(program, &dirs)
}

fn normalize_install_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "/")
        .trim_start_matches("//?/")
        .to_ascii_lowercase()
}

/// Detected package manager type
enum PackageManager {
    Homebrew,
    Cargo,
    Npm,
}

/// Information about a detected package manager
struct ManagedInstall {
    kind: PackageManager,
    name: &'static str,
    upgrade_command: &'static str,
}

/// Detects whether the binary is managed by a package manager based on its path.
/// Returns None if the binary appears to be a direct download or AIVO_PATH is set.
fn detect_managed_install(install_path: &Path) -> Option<ManagedInstall> {
    // If AIVO_PATH is explicitly set, user opted into this path — skip detection
    if env::var("AIVO_PATH").is_ok() {
        return None;
    }

    let path_str = normalize_install_path(install_path);

    // npm: .../node_modules/@yuanchuan/aivo/...
    if path_str.contains("/node_modules/") {
        return Some(ManagedInstall {
            kind: PackageManager::Npm,
            name: "npm",
            upgrade_command: NPM_UPDATE_COMMAND,
        });
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

#[cfg(test)]
mod tests {
    use super::*;

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

        // Release is newer than same-version pre-release
        assert!(cmd.is_newer_version("2.0.0", "2.0.0-rc1"));
        assert!(cmd.is_newer_version("2.0.0", "2.0.0-beta.1"));

        // Pre-release is not newer than its release
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0"));

        // Same pre-release versions are not newer
        assert!(!cmd.is_newer_version("2.0.0-rc1", "2.0.0-rc1"));

        // Higher version is still newer regardless of pre-release
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

    #[test]
    fn test_detect_npm_global() {
        let path = Path::new("/opt/homebrew/lib/node_modules/@yuanchuan/aivo/native/aivo");
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "npm");
        assert_eq!(m.upgrade_command, NPM_UPDATE_COMMAND);
    }

    #[test]
    fn test_detect_npm_nvm() {
        let path = Path::new(
            "/Users/user/.nvm/versions/node/v22.0.0/lib/node_modules/@yuanchuan/aivo/native/aivo",
        );
        let result = detect_managed_install(path);
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "npm");
    }

    #[test]
    fn test_detect_npm_windows_path() {
        let path = Path::new(
            r"C:\Users\user\AppData\Roaming\npm\node_modules\@yuanchuan\aivo\native\aivo.exe",
        );
        let result = detect_managed_install(path);
        assert!(result.is_some());
        let m = result.unwrap();
        assert_eq!(m.name, "npm");
        assert_eq!(m.upgrade_command, NPM_UPDATE_COMMAND);
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
