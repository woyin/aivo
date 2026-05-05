//! Cross-platform symlink helpers used by the codex and gemini home
//! shadows. Each shadow seeds a temp dir with the user's auth files, then
//! symlinks user-state files (chat history, memory, settings) from the
//! real `~/.codex/` or `~/.gemini/` so the spawned CLI behaves like a
//! normal launch and writes persist back across runs.
//!
//! On Unix we use `tokio::fs::symlink` (POSIX symlinks). On Windows we
//! use `std::os::windows::fs::symlink_{dir,file}` via `spawn_blocking`,
//! which silently fails without dev-mode/admin — callers tolerate that.

use std::path::Path;

#[cfg(unix)]
pub async fn symlink_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    tokio::fs::symlink(src, dst).await
}

#[cfg(unix)]
pub async fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    tokio::fs::symlink(src, dst).await
}

#[cfg(windows)]
pub async fn symlink_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    let src = src.to_path_buf();
    let dst = dst.to_path_buf();
    tokio::task::spawn_blocking(move || std::os::windows::fs::symlink_dir(&src, &dst))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
}

#[cfg(windows)]
pub async fn symlink_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    let src = src.to_path_buf();
    let dst = dst.to_path_buf();
    tokio::task::spawn_blocking(move || std::os::windows::fs::symlink_file(&src, &dst))
        .await
        .map_err(|e| std::io::Error::other(e.to_string()))?
}

#[cfg(not(any(unix, windows)))]
pub async fn symlink_dir(_src: &Path, _dst: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("symlink unsupported"))
}

#[cfg(not(any(unix, windows)))]
pub async fn symlink_file(_src: &Path, _dst: &Path) -> std::io::Result<()> {
    Err(std::io::Error::other("symlink unsupported"))
}
