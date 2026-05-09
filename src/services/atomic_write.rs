use anyhow::{Context, Result};
use std::path::Path;

/// Atomically writes `data` to `final_path` with mode `0o600` on Unix.
///
/// Writes to a sibling `*.tmp` file created with restrictive permissions from
/// the start (no world-readable window), fsyncs, then renames into place.
/// Callers must ensure the parent directory exists.
///
/// Windows: the temp file inherits NTFS ACLs from the parent directory (which
/// for files under `~/.config/aivo` is user-only by default), and is opened
/// with `share_mode(0)` so no other process can read it while it's open.
pub(crate) async fn atomic_write_secure(final_path: &Path, data: Vec<u8>) -> Result<()> {
    let final_path_owned = final_path.to_path_buf();
    tokio::task::spawn_blocking(move || atomic_write_secure_blocking(&final_path_owned, &data))
        .await
        .context("Join error writing temp file")?
}

/// Synchronous variant of [`atomic_write_secure`] for callers that aren't in
/// async context. Same semantics: temp-file write with restrictive permissions,
/// fsync, then rename.
pub(crate) fn atomic_write_secure_blocking(final_path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    let tmp_path = final_path.with_extension("json.tmp");

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        opts.share_mode(0);
    }

    {
        let mut file = opts
            .open(&tmp_path)
            .with_context(|| format!("Failed to open temp file: {:?}", tmp_path))?;
        file.write_all(data)
            .with_context(|| format!("Failed to write temp file: {:?}", tmp_path))?;
        file.sync_all()
            .with_context(|| format!("Failed to sync temp file: {:?}", tmp_path))?;
    }

    std::fs::rename(&tmp_path, final_path)
        .with_context(|| format!("Failed to rename temp file to {:?}", final_path))?;

    Ok(())
}
