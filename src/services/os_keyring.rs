//! OS-keyring custody of the v5 encryption master secret.
//! Backends: macOS `/usr/bin/security`, Linux `secret-tool`, Windows
//! Credential Manager. Lookups are process-cached; creation is verified
//! by read-back so concurrent first-runs converge on one secret.

use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(not(test))]
const SERVICE: &str = "aivo";
#[cfg(not(test))]
const ACCOUNT: &str = "master-secret";
pub const SECRET_LEN: usize = 32;

/// Write-side default: v5 stays opt-in (AIVO_KEYCHAIN=1) until the
/// read-capable release has been out long enough to make downgrades safe.
#[cfg(not(test))]
const DEFAULT_ENABLED: bool = false;

#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct MasterSecret([u8; SECRET_LEN]);

impl MasterSecret {
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }
}

#[cfg(not(test))]
#[derive(Clone)]
enum Lookup {
    Found(MasterSecret),
    Absent,
    Unavailable,
}

/// Gate for *writing* v5 values. Reads of existing v5 values ignore this.
pub fn keyring_enabled() -> bool {
    #[cfg(test)]
    return test_state::enabled();
    #[cfg(not(test))]
    match std::env::var("AIVO_KEYCHAIN").ok().as_deref() {
        Some("1") | Some("true") => true,
        Some("0") | Some("false") => false,
        _ => DEFAULT_ENABLED,
    }
}

/// Read-only lookup; never creates the secret. Used by decrypt.
pub fn master_secret() -> Option<MasterSecret> {
    #[cfg(test)]
    return test_state::secret();
    #[cfg(not(test))]
    {
        let mut cache = lock_cache();
        if cache.is_none() {
            *cache = Some(backend_lookup());
        }
        match cache.as_ref() {
            Some(Lookup::Found(secret)) => Some(secret.clone()),
            _ => None,
        }
    }
}

/// Lookup that creates the secret on first use. Used by encrypt/migration;
/// callers hold the config lock, which serializes competing first writes.
pub fn ensure_master_secret() -> Option<MasterSecret> {
    #[cfg(test)]
    return test_state::secret();
    #[cfg(not(test))]
    {
        let mut cache = lock_cache();
        if cache.is_none() {
            *cache = Some(backend_lookup());
        }
        match cache.as_ref() {
            Some(Lookup::Found(secret)) => return Some(secret.clone()),
            Some(Lookup::Absent) => {}
            _ => return None,
        }
        use rand::RngCore;
        let mut fresh = [0u8; SECRET_LEN];
        rand::thread_rng().fill_bytes(&mut fresh);
        backend_store(&fresh);
        fresh.zeroize();
        // The read-back is the source of truth: if another writer raced us,
        // everyone converges on whatever the keyring actually holds.
        let outcome = backend_lookup();
        let secret = match &outcome {
            Lookup::Found(secret) => Some(secret.clone()),
            _ => None,
        };
        *cache = Some(outcome);
        secret
    }
}

#[cfg(not(test))]
fn lock_cache() -> std::sync::MutexGuard<'static, Option<Lookup>> {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Option<Lookup>>> = OnceLock::new();
    CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(not(test))]
fn backend_lookup() -> Lookup {
    #[cfg(feature = "__internal_test_fast_crypto")]
    if let Ok(hex) = std::env::var("AIVO_TEST_MASTER_SECRET") {
        return match decode_secret_hex(&hex) {
            Some(bytes) => Lookup::Found(MasterSecret(bytes)),
            None => Lookup::Unavailable,
        };
    }
    platform::lookup()
}

#[cfg(not(test))]
fn backend_store(secret: &[u8; SECRET_LEN]) {
    #[cfg(feature = "__internal_test_fast_crypto")]
    if std::env::var("AIVO_TEST_MASTER_SECRET").is_ok() {
        return;
    }
    platform::store(&encode_hex(secret));
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_secret_hex(s: &str) -> Option<[u8; SECRET_LEN]> {
    let s = s.trim();
    if s.len() != SECRET_LEN * 2 {
        return None;
    }
    let mut out = [0u8; SECRET_LEN];
    for (i, chunk) in s.as_bytes().chunks_exact(2).enumerate() {
        let hi = (chunk[0] as char).to_digit(16)?;
        let lo = (chunk[1] as char).to_digit(16)?;
        out[i] = (hi as u8) << 4 | lo as u8;
    }
    Some(out)
}

/// Watchdog wrapper for keyring helper processes: `security`/`secret-tool`
/// can block forever waiting on auth UI in sessionless contexts (SSH, cron),
/// which must degrade to v4, not hang the launch.
#[cfg(all(not(test), any(target_os = "macos", target_os = "linux")))]
mod subprocess {
    use std::io::{Read, Write};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const WATCHDOG: Duration = Duration::from_secs(10);

    pub(super) struct RunOutput {
        pub code: Option<i32>,
        pub success: bool,
        pub stdout: String,
        pub stderr: String,
    }

    pub(super) fn run(mut cmd: Command, stdin_data: Option<&str>) -> Option<RunOutput> {
        cmd.stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        let mut child = cmd.spawn().ok()?;
        if let Some(data) = stdin_data
            && let Some(mut stdin) = child.stdin.take()
        {
            let _ = stdin.write_all(data.as_bytes());
        }
        let deadline = Instant::now() + WATCHDOG;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let mut stdout = String::new();
                    let mut stderr = String::new();
                    if let Some(mut s) = child.stdout.take() {
                        let _ = s.read_to_string(&mut stdout);
                    }
                    if let Some(mut s) = child.stderr.take() {
                        let _ = s.read_to_string(&mut stderr);
                    }
                    return Some(RunOutput {
                        code: status.code(),
                        success: status.success(),
                        stdout,
                        stderr,
                    });
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                Err(_) => return None,
            }
        }
    }
}

#[cfg(all(not(test), target_os = "macos"))]
mod platform {
    use super::subprocess::run;
    use super::{ACCOUNT, Lookup, MasterSecret, SERVICE, decode_secret_hex};
    use std::process::Command;

    pub(super) fn lookup() -> Lookup {
        let mut cmd = Command::new("/usr/bin/security");
        cmd.args(["find-generic-password", "-s", SERVICE, "-a", ACCOUNT, "-w"]);
        let Some(output) = run(cmd, None) else {
            return Lookup::Unavailable;
        };
        if output.success {
            return match decode_secret_hex(&output.stdout) {
                Some(bytes) => Lookup::Found(MasterSecret(bytes)),
                // Malformed item: never report Absent, or a store would clobber it.
                None => Lookup::Unavailable,
            };
        }
        // errSecItemNotFound; anything else (e.g. locked keychain) is Unavailable.
        if output.code == Some(44) || output.stderr.contains("could not be found") {
            Lookup::Absent
        } else {
            Lookup::Unavailable
        }
    }

    pub(super) fn store(hex: &str) {
        // `security -i` keeps the secret out of argv; no -U so the first
        // writer wins and racing creators converge via read-back.
        let mut cmd = Command::new("/usr/bin/security");
        cmd.arg("-i");
        let script = format!(
            "add-generic-password -s \"{SERVICE}\" -a \"{ACCOUNT}\" -w \"{hex}\" -T \"/usr/bin/security\"\n"
        );
        let _ = run(cmd, Some(&script));
    }
}

#[cfg(all(not(test), target_os = "linux"))]
mod platform {
    use super::subprocess::run;
    use super::{ACCOUNT, Lookup, MasterSecret, SERVICE, decode_secret_hex};
    use std::process::Command;

    pub(super) fn lookup() -> Lookup {
        let mut cmd = Command::new("secret-tool");
        cmd.args(["lookup", "service", SERVICE, "account", ACCOUNT]);
        let Some(output) = run(cmd, None) else {
            // secret-tool missing, hung, or no DBus session: stay on v4.
            return Lookup::Unavailable;
        };
        if output.success {
            return match decode_secret_hex(&output.stdout) {
                Some(bytes) => Lookup::Found(MasterSecret(bytes)),
                None => Lookup::Unavailable,
            };
        }
        if output.code == Some(1) && output.stderr.trim().is_empty() {
            Lookup::Absent
        } else {
            Lookup::Unavailable
        }
    }

    pub(super) fn store(hex: &str) {
        let mut cmd = Command::new("secret-tool");
        cmd.args([
            "store",
            "--label",
            "aivo master secret",
            "service",
            SERVICE,
            "account",
            ACCOUNT,
        ]);
        let _ = run(cmd, Some(hex));
    }
}

#[cfg(all(not(test), target_os = "windows"))]
mod platform {
    use super::{Lookup, MasterSecret, decode_secret_hex};
    use windows_sys::Win32::Foundation::{ERROR_NOT_FOUND, FILETIME, GetLastError};
    use windows_sys::Win32::Security::Credentials::{
        CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC, CREDENTIALW, CredFree, CredReadW, CredWriteW,
    };

    const TARGET: &str = "aivo/master-secret";

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub(super) fn lookup() -> Lookup {
        let target = wide(TARGET);
        let mut credential: *mut CREDENTIALW = std::ptr::null_mut();
        let ok = unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) };
        if ok == 0 {
            return if unsafe { GetLastError() } == ERROR_NOT_FOUND {
                Lookup::Absent
            } else {
                Lookup::Unavailable
            };
        }
        let result = unsafe {
            let blob = std::slice::from_raw_parts(
                (*credential).CredentialBlob,
                (*credential).CredentialBlobSize as usize,
            );
            decode_secret_hex(&String::from_utf8_lossy(blob))
        };
        unsafe { CredFree(credential as *mut core::ffi::c_void) };
        match result {
            Some(bytes) => Lookup::Found(MasterSecret(bytes)),
            None => Lookup::Unavailable,
        }
    }

    pub(super) fn store(hex: &str) {
        let target = wide(TARGET);
        let username = wide("aivo");
        let mut blob = hex.as_bytes().to_vec();
        let credential = CREDENTIALW {
            Flags: 0,
            Type: CRED_TYPE_GENERIC,
            TargetName: target.as_ptr() as *mut u16,
            Comment: std::ptr::null_mut(),
            LastWritten: FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            },
            CredentialBlobSize: blob.len() as u32,
            CredentialBlob: blob.as_mut_ptr(),
            Persist: CRED_PERSIST_LOCAL_MACHINE,
            AttributeCount: 0,
            Attributes: std::ptr::null_mut(),
            TargetAlias: std::ptr::null_mut(),
            UserName: username.as_ptr() as *mut u16,
        };
        unsafe { CredWriteW(&credential, 0) };
    }
}

#[cfg(all(
    not(test),
    not(any(target_os = "macos", target_os = "linux", target_os = "windows"))
))]
mod platform {
    use super::Lookup;

    pub(super) fn lookup() -> Lookup {
        Lookup::Unavailable
    }

    pub(super) fn store(_hex: &str) {}
}

/// Unit tests never touch the real OS keyring: state is thread-local so
/// parallel tests stay isolated.
#[cfg(test)]
pub(crate) mod test_state {
    use super::{MasterSecret, SECRET_LEN};
    use std::cell::RefCell;

    thread_local! {
        static STATE: RefCell<(bool, Option<[u8; SECRET_LEN]>)> = const { RefCell::new((false, None)) };
    }

    pub fn set(enabled: bool, secret: Option<[u8; SECRET_LEN]>) {
        STATE.with(|s| *s.borrow_mut() = (enabled, secret));
    }

    pub(super) fn enabled() -> bool {
        STATE.with(|s| s.borrow().0)
    }

    pub(super) fn secret() -> Option<MasterSecret> {
        STATE.with(|s| s.borrow().1.map(MasterSecret))
    }
}

#[cfg(test)]
mod tests {
    use super::{SECRET_LEN, decode_secret_hex, encode_hex};

    #[test]
    fn test_hex_roundtrip() {
        let bytes: Vec<u8> = (0..SECRET_LEN as u8).collect();
        let hex = encode_hex(&bytes);
        assert_eq!(hex.len(), SECRET_LEN * 2);
        let decoded = decode_secret_hex(&hex).unwrap();
        assert_eq!(decoded.as_slice(), bytes.as_slice());
    }

    #[test]
    fn test_decode_rejects_bad_input() {
        assert!(decode_secret_hex("").is_none());
        assert!(decode_secret_hex("zz").is_none());
        assert!(decode_secret_hex(&"a".repeat(SECRET_LEN * 2 - 1)).is_none());
        assert!(decode_secret_hex(&"g".repeat(SECRET_LEN * 2)).is_none());
        let with_newline = format!("{}\n", "ab".repeat(SECRET_LEN));
        assert!(decode_secret_hex(&with_newline).is_some());
    }
}
