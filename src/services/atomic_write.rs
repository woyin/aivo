use anyhow::{Context, Result};
use std::path::Path;

/// Atomically writes `data` to `final_path` with mode `0o600` on Unix.
///
/// Uses an `O_CREAT | O_EXCL` temp file with a random suffix so a hostile
/// symlink at a predictable temp path can't be hijacked, then renames over
/// `final_path` (the rename replaces a symlink at the destination without
/// following it).
pub(crate) async fn atomic_write_secure(final_path: &Path, data: Vec<u8>) -> Result<()> {
    let final_path_owned = final_path.to_path_buf();
    tokio::task::spawn_blocking(move || atomic_write_secure_blocking(&final_path_owned, &data))
        .await
        .context("Join error writing temp file")?
}

pub(crate) fn atomic_write_secure_blocking(final_path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;

    let parent = final_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    let mut tmp = tempfile::Builder::new()
        .prefix(".aivo-tmp-")
        .tempfile_in(parent)
        .with_context(|| format!("Failed to create temp file in {:?}", parent))?;

    // Belt-and-braces against a future tempfile-crate default widening perms.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("Failed to set 0600 on {:?}", tmp.path()))?;
    }

    tmp.as_file_mut()
        .write_all(data)
        .with_context(|| format!("Failed to write temp file: {:?}", tmp.path()))?;
    tmp.as_file()
        .sync_all()
        .with_context(|| format!("Failed to sync temp file: {:?}", tmp.path()))?;

    tmp.persist(final_path).map_err(|e| {
        anyhow::anyhow!(
            "Failed to rename temp file to {:?}: {}",
            final_path,
            e.error
        )
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn roundtrip_writes_data() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("config.json");
        atomic_write_secure_blocking(&target, b"hello").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
    }

    #[test]
    fn overwrites_existing_regular_file() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("config.json");
        std::fs::write(&target, b"old").unwrap();
        atomic_write_secure_blocking(&target, b"new").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
    }

    #[cfg(unix)]
    #[test]
    fn rename_replaces_symlink_without_following_it() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("config.json");
        let sentinel = dir.path().join("sentinel");
        std::fs::write(&sentinel, b"do not clobber").unwrap();
        std::os::unix::fs::symlink(&sentinel, &target).unwrap();

        atomic_write_secure_blocking(&target, b"new content").unwrap();

        assert_eq!(std::fs::read(&sentinel).unwrap(), b"do not clobber");
        let meta = std::fs::symlink_metadata(&target).unwrap();
        assert!(!meta.file_type().is_symlink(), "symlink should be replaced");
        assert_eq!(std::fs::read(&target).unwrap(), b"new content");
    }

    #[cfg(unix)]
    #[test]
    fn written_file_is_mode_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("config.json");
        atomic_write_secure_blocking(&target, b"x").unwrap();
        let mode = std::fs::metadata(&target).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
