use super::super::*;

pub(super) fn tmp() -> PathBuf {
    crate::test_sandbox::tmp("aivo-agent-tools")
}

#[cfg(unix)]
pub(super) fn mkfifo(path: &Path) {
    let c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o644) }, 0);
}
