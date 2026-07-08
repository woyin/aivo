use std::path::{Path, PathBuf};

/// Best-effort user home directory lookup using standard environment variables.
pub fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
            .or_else(|| {
                let drive = std::env::var_os("HOMEDRIVE")?;
                let path = std::env::var_os("HOMEPATH")?;
                Some(PathBuf::from(format!(
                    "{}{}",
                    drive.to_string_lossy(),
                    path.to_string_lossy()
                )))
            })
    }

    #[cfg(not(windows))]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Best-effort current username lookup.
/// Tries the USER/USERNAME environment variable first, then falls back to the
/// OS user database via libc on Unix (unaffected by sudo USER overrides or
/// environments where USER is unset).
pub fn username() -> Option<String> {
    #[cfg(windows)]
    {
        std::env::var("USERNAME").ok().filter(|s| !s.is_empty())
    }

    #[cfg(not(windows))]
    {
        if let Ok(user) = std::env::var("USER")
            && !user.is_empty()
        {
            return Some(user);
        }

        // Fall back to OS user database so key derivation remains consistent
        // when USER is unset (CI, containers, sudo -i, etc.).
        #[cfg(unix)]
        // SAFETY: getpwuid returns a pointer to static thread-local storage valid
        // until the next getpwuid call on this thread. We copy pw_name immediately.
        unsafe {
            let uid = libc::getuid();
            let passwd = libc::getpwuid(uid);
            if !passwd.is_null() {
                let name = std::ffi::CStr::from_ptr((*passwd).pw_name);
                if let Ok(s) = name.to_str()
                    && !s.is_empty()
                {
                    return Some(s.to_string());
                }
            }
        }

        None
    }
}

/// Parse an on/off env flag; `None` means unset/empty (caller's default applies).
/// The one truthiness rule — hand-rolled per-site checks used to drift.
pub fn env_flag(var: &str) -> Option<bool> {
    let v = std::env::var(var).ok()?;
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    Some(!matches!(
        v.to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    ))
}

/// Returns true if a process with `pid` is still alive on this system. Used
/// to prune stale registry entries and detect orphaned helper processes.
pub fn is_pid_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // SAFETY: `kill(pid, 0)` sends no signal; it only checks whether the
        // process exists and we have permission to signal it.
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, WaitForSingleObject,
        };

        // SAFETY: OpenProcess/WaitForSingleObject/CloseHandle take integer or
        // handle values only; no memory is dereferenced here. Returned handles
        // are always closed before returning.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                // Open failure is almost always "no such process" for PIDs we
                // own; treat it as dead so stale entries get pruned instead
                // of latching on.
                return false;
            }
            // 0 timeout: returns WAIT_OBJECT_0 once the process has exited,
            // WAIT_TIMEOUT while it is still alive.
            let alive = WaitForSingleObject(handle, 0) != WAIT_OBJECT_0;
            CloseHandle(handle);
            alive
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

/// Expands a leading `~` to the user's home directory.
/// Returns the path unchanged (as a `PathBuf`) if expansion is not needed or not possible.
pub fn expand_tilde(path: &str) -> PathBuf {
    expand_tilde_with_home(path, home_dir().as_deref())
}

/// Joins each segment onto `base` with `PathBuf::push`, so the result
/// uses the platform separator. Plain `base.join("a/b/c")` preserves
/// the embedded `/` on Windows and produces mixed-separator paths.
pub fn join_segments(base: &Path, segments: &[&str]) -> PathBuf {
    let mut p = base.to_path_buf();
    for seg in segments {
        p.push(seg);
    }
    p
}

/// Replaces a leading home directory with `~` for display purposes.
/// Returns the path unchanged if it doesn't start with the user's home.
pub fn collapse_tilde(path: &str) -> String {
    collapse_tilde_with_home(path, home_dir().as_deref())
}

/// Best-effort current working directory lookup with canonicalization when possible.
pub fn current_dir() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    std::fs::canonicalize(&cwd).ok().or(Some(cwd))
}

pub fn current_dir_string() -> Option<String> {
    current_dir().map(|path| path.to_string_lossy().to_string())
}

/// Returns a hardware-specific machine identifier.
/// - macOS: IOPlatformUUID
/// - Linux: /etc/machine-id
/// - Windows: HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid
pub fn machine_id() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()?;
        parse_macos_platform_uuid(&String::from_utf8_lossy(&output.stdout))
    }

    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/etc/machine-id")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("reg")
            .args([
                "query",
                r"HKLM\SOFTWARE\Microsoft\Cryptography",
                "/v",
                "MachineGuid",
            ])
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("MachineGuid") {
                // Format: MachineGuid    REG_SZ    XXXXXXXX-...
                if let Some(guid) = line.split_whitespace().last() {
                    let guid = guid.trim().to_string();
                    if !guid.is_empty() {
                        return Some(guid);
                    }
                }
            }
        }
        None
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn parse_macos_platform_uuid(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let pos = line.find("IOPlatformUUID")?;
        let rest = &line[pos..];
        // Skip past `"IOPlatformUUID" = "` to find the value
        let eq_pos = rest.find('=')?;
        let after_eq = &rest[eq_pos + 1..];
        let open_quote = after_eq.find('"')?;
        let value_start = open_quote + 1;
        let close_quote = after_eq[value_start..].find('"')?;
        let uuid = after_eq[value_start..value_start + close_quote]
            .trim()
            .to_string();
        (!uuid.is_empty()).then_some(uuid)
    })
}

/// Legacy parser preserved for v3 key derivation backward compatibility.
/// The original parser incorrectly extracted `=` instead of the UUID value.
#[cfg(target_os = "macos")]
fn parse_macos_platform_uuid_legacy(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let pos = line.find("IOPlatformUUID")?;
        let start = line[pos..].find('"').map(|i| pos + i + 1)?;
        let end = line[start..].find('"').map(|i| start + i)?;
        let uuid = line[start..end].trim().to_string();
        (!uuid.is_empty()).then_some(uuid)
    })
}

/// Legacy machine ID using the buggy UUID parser, preserved for v3 key derivation.
pub fn machine_id_legacy() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
            .ok()?;
        parse_macos_platform_uuid_legacy(&String::from_utf8_lossy(&output.stdout))
    }

    #[cfg(not(target_os = "macos"))]
    {
        machine_id()
    }
}

fn expand_tilde_with_home(path: &str, home: Option<&Path>) -> PathBuf {
    if path == "~" {
        return home
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = home
    {
        return rest
            .split('/')
            .filter(|s| !s.is_empty())
            .fold(home.to_path_buf(), |mut p, s| {
                p.push(s);
                p
            });
    }
    PathBuf::from(path)
}

fn collapse_tilde_with_home(path: &str, home: Option<&Path>) -> String {
    if let Some(home) = home {
        let home_str = home.to_string_lossy();
        if !home_str.is_empty()
            && let Some(rest) = path.strip_prefix(&*home_str)
        {
            if rest.is_empty() {
                return "~".to_string();
            }
            // Guard against matching `/home/userfoo` when home is `/home/user`.
            if rest.starts_with('/') || rest.starts_with('\\') {
                return format!("~{rest}");
            }
        }
    }
    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_replaces_home_prefix() {
        let home = Path::new("/tmp/example-home");
        let expanded = expand_tilde_with_home("~/config/aivo", Some(home));
        let expected: PathBuf = home.join("config").join("aivo");
        assert_eq!(expanded, expected);
        assert_eq!(expand_tilde_with_home("~", Some(home)), home);
    }

    #[test]
    fn collapse_tilde_replaces_home_prefix() {
        let home = Path::new("/tmp/example-home");
        assert_eq!(
            collapse_tilde_with_home("/tmp/example-home/config/aivo", Some(home)),
            "~/config/aivo"
        );
        assert_eq!(
            collapse_tilde_with_home("/tmp/example-home", Some(home)),
            "~"
        );
    }

    #[test]
    fn collapse_tilde_does_not_match_sibling_paths() {
        // e.g. home = /home/user, path = /home/user2 — must NOT become ~2
        let home = Path::new("/home/user");
        assert_eq!(
            collapse_tilde_with_home("/home/user2/foo", Some(home)),
            "/home/user2/foo"
        );
    }

    #[test]
    fn collapse_tilde_leaves_non_home_paths_unchanged() {
        assert_eq!(
            collapse_tilde_with_home("/var/tmp/aivo", Some(Path::new("/tmp/home"))),
            "/var/tmp/aivo"
        );
        assert_eq!(
            collapse_tilde_with_home("/var/tmp/aivo", None),
            "/var/tmp/aivo"
        );
    }

    #[test]
    fn expand_tilde_leaves_non_home_paths_unchanged() {
        assert_eq!(
            expand_tilde_with_home("/var/tmp/aivo", Some(Path::new("/tmp/home"))),
            PathBuf::from("/var/tmp/aivo")
        );
        assert_eq!(
            expand_tilde_with_home("~/docs", None),
            PathBuf::from("~/docs")
        );
    }

    #[test]
    fn current_dir_string_returns_non_empty_path() {
        let cwd = current_dir_string().expect("cwd should be available");
        assert!(!cwd.is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_macos_platform_uuid_extracts_value() {
        let output = r#"    "IOPlatformUUID" = "12345678-1234-1234-1234-123456789ABC""#;
        assert_eq!(
            parse_macos_platform_uuid(output).as_deref(),
            Some("12345678-1234-1234-1234-123456789ABC")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_macos_platform_uuid_rejects_blank_values() {
        let output = r#"    "IOPlatformUUID" = """#;
        assert_eq!(parse_macos_platform_uuid(output), None);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parse_macos_platform_uuid_legacy_returns_equals() {
        // Legacy parser preserved for v3 backward compatibility — returns "="
        let output = r#"    "IOPlatformUUID" = "12345678-1234-1234-1234-123456789ABC""#;
        assert_eq!(
            parse_macos_platform_uuid_legacy(output).as_deref(),
            Some("=")
        );
    }
}
