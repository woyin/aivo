//! Cross-platform "open this URL in the user's default browser".
//!
//! Best-effort only; the caller must be prepared to fall back to printing
//! the URL and asking the user to paste the callback URL back.

use std::process::{Command, Stdio};

#[cfg(target_os = "macos")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    Command::new("open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(target_os = "linux")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    // Try xdg-open first, then sensible-browser. Either failing means the
    // environment lacks a desktop integration; the caller will prompt for a
    // manual paste.
    if let Ok(child) = Command::new("xdg-open")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        drop(child);
        return Ok(());
    }
    Command::new("sensible-browser")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(target_os = "windows")]
pub fn open_url(url: &str) -> std::io::Result<()> {
    // NOT `cmd /C start`: cmd reparses the command line, so a URL's `&` becomes a
    // command separator (truncating every OAuth authorize URL after the first
    // query param) and `%xx` percent-encodings get env-expanded. `explorer.exe`
    // receives argv directly and hands the whole URL to the default protocol
    // handler. It exits 1 even on success, so the ignored status is fine.
    Command::new("explorer")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map(|_| ())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
pub fn open_url(_url: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no known browser launcher for this platform",
    ))
}
