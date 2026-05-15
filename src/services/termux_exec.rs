//! Termux exec shim: rewrites shebang lines that point at non-existent
//! Linux paths (`/usr/bin/env`, `/bin/sh`, …) so script-based child
//! processes resolve under Termux's `$PREFIX`. Termux ships `termux-exec`
//! as an `LD_PRELOAD` shim, but aivo is a musl-static binary that bypasses
//! it — `execve()` of an npm shim therefore hits the kernel directly and
//! fails with ENOENT on the missing interpreter.

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// True when running inside Termux. Detected via `$PREFIX` pointing at a
/// `com.termux` rootless tree.
pub fn is_termux() -> bool {
    match std::env::var("PREFIX") {
        Ok(p) => p.contains("/com.termux/") && Path::new(&p).join("bin").is_dir(),
        Err(_) => false,
    }
}

/// If `command` resolves to a shebang script whose interpreter is a path
/// that doesn't exist on Termux (e.g. `/usr/bin/env`), return a rewritten
/// `(interpreter, args)` pair that invokes `$PREFIX/bin/<basename>`
/// instead. Returns `None` when no rewrite is needed or possible.
pub fn rewrite_shebang(command: &str, args: &[String]) -> Option<(String, Vec<String>)> {
    if !is_termux() {
        return None;
    }
    let prefix = std::env::var_os("PREFIX")?;
    let prefix_bin = PathBuf::from(prefix).join("bin");
    rewrite_with_prefix_bin(&prefix_bin, command, args)
}

fn rewrite_with_prefix_bin(
    prefix_bin: &Path,
    command: &str,
    args: &[String],
) -> Option<(String, Vec<String>)> {
    let script_path = resolve_command_path(command, prefix_bin)?;
    let canonical = std::fs::canonicalize(&script_path).ok()?;
    let (interp, extra) = read_shebang(&canonical)?;
    let interp_path = Path::new(&interp);
    if !is_typical_linux_bin_path(interp_path) {
        return None;
    }
    let basename = interp_path.file_name()?.to_str()?;
    let mapped = prefix_bin.join(basename);
    if !mapped.exists() {
        return None;
    }
    let mut new_args: Vec<String> = Vec::with_capacity(2 + args.len());
    if let Some(s) = extra
        && !s.is_empty()
    {
        new_args.push(s);
    }
    // Pass the original (possibly-symlink) script path, matching what
    // termux-exec does — keeps `__filename` / `$0` stable for consumers.
    new_args.push(script_path.to_string_lossy().into_owned());
    new_args.extend(args.iter().cloned());
    Some((mapped.to_string_lossy().into_owned(), new_args))
}

/// Whether `p` is one of the standard FHS bin dirs that don't exist on
/// Android/Termux. Matches the set `termux-exec` rewrites.
fn is_typical_linux_bin_path(p: &Path) -> bool {
    matches!(
        p.parent().and_then(|d| d.to_str()),
        Some("/bin" | "/usr/bin" | "/usr/local/bin" | "/sbin" | "/usr/sbin")
    )
}

fn resolve_command_path(command: &str, prefix_bin: &Path) -> Option<PathBuf> {
    if command.contains('/') {
        let p = PathBuf::from(command);
        if p.exists() {
            return Some(p);
        }
        return None;
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(command);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    let fallback = prefix_bin.join(command);
    if fallback.is_file() {
        Some(fallback)
    } else {
        None
    }
}

fn read_shebang(path: &Path) -> Option<(String, Option<String>)> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let line = line.trim_end_matches(['\r', '\n']);
    let stripped = line.strip_prefix("#!")?.trim_start();
    // The Linux kernel passes everything after the first whitespace as a
    // single argument; `env -S` does its own re-splitting. Mirror that.
    match stripped.split_once(char::is_whitespace) {
        Some((interp, rest)) => Some((interp.to_string(), Some(rest.trim().to_string()))),
        None => Some((stripped.to_string(), None)),
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    fn mk_script(dir: &Path, name: &str, shebang: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, "{shebang}").unwrap();
        writeln!(f, "echo hi").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn mk_fake_bin(prefix_bin: &Path, name: &str) {
        fs::create_dir_all(prefix_bin).unwrap();
        let p = prefix_bin.join(name);
        fs::write(&p, b"#!stub\n").unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn remaps_usr_bin_env_to_prefix_bin() {
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        mk_fake_bin(&prefix_bin, "env");
        let script = mk_script(tmp.path(), "pi", "#!/usr/bin/env node");
        let (cmd, args) =
            rewrite_with_prefix_bin(&prefix_bin, script.to_str().unwrap(), &["--v".into()])
                .expect("rewrite expected");
        assert_eq!(cmd, prefix_bin.join("env").to_string_lossy());
        assert_eq!(
            args,
            vec![
                "node".to_string(),
                script.to_string_lossy().into_owned(),
                "--v".to_string(),
            ]
        );
    }

    #[test]
    fn keeps_env_s_args_as_single_kernel_arg() {
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        mk_fake_bin(&prefix_bin, "env");
        let script = mk_script(tmp.path(), "cli", "#!/usr/bin/env -S node --no-warnings");
        let (_, args) =
            rewrite_with_prefix_bin(&prefix_bin, script.to_str().unwrap(), &[]).unwrap();
        assert_eq!(args[0], "-S node --no-warnings");
        assert_eq!(args[1], script.to_string_lossy());
    }

    #[test]
    fn follows_symlink_to_read_target_shebang() {
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        mk_fake_bin(&prefix_bin, "env");
        let real = mk_script(tmp.path(), "cli.js", "#!/usr/bin/env node");
        let link = tmp.path().join("pi");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let (_, args) = rewrite_with_prefix_bin(&prefix_bin, link.to_str().unwrap(), &[]).unwrap();
        // Symlink path preserved as the script arg, not the canonical target.
        assert_eq!(args[1], link.to_string_lossy());
    }

    #[test]
    fn skips_termux_aware_shebangs() {
        // A script whose shebang already points inside `$PREFIX/bin` (i.e.
        // the user already fixed it, or Termux's own packages installed it)
        // must not be rewritten — it would be redundant and could mangle
        // unusual interpreter paths.
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        mk_fake_bin(&prefix_bin, "sh");
        let prefixed_shebang = format!("#!{}", prefix_bin.join("sh").display());
        let script = mk_script(tmp.path(), "ok", &prefixed_shebang);
        assert!(rewrite_with_prefix_bin(&prefix_bin, script.to_str().unwrap(), &[]).is_none());
    }

    #[test]
    fn skips_non_shebang_files() {
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        let bin = tmp.path().join("native");
        fs::write(&bin, b"\x7fELF not really a binary").unwrap();
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(rewrite_with_prefix_bin(&prefix_bin, bin.to_str().unwrap(), &[]).is_none());
    }

    #[test]
    fn skips_when_prefix_bin_has_no_matching_interpreter() {
        // Shebang points at a typical FHS path (`/usr/bin/python3`) but
        // there's no `$PREFIX/bin/python3` to rewrite to — we can't help,
        // so return None and let the caller's spawn surface the real error
        // rather than silently rerouting to a different interpreter.
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        fs::create_dir_all(&prefix_bin).unwrap();
        let script = mk_script(tmp.path(), "x", "#!/usr/bin/python3");
        assert!(rewrite_with_prefix_bin(&prefix_bin, script.to_str().unwrap(), &[]).is_none());
    }

    #[test]
    fn skips_non_fhs_shebang_paths() {
        // `/opt/...` and other non-FHS bin dirs are left alone — Termux
        // users who write custom paths know what they're doing.
        let tmp = TempDir::new().unwrap();
        let prefix_bin = tmp.path().join("usr/bin");
        mk_fake_bin(&prefix_bin, "node");
        let script = mk_script(tmp.path(), "x", "#!/opt/weird/node");
        assert!(rewrite_with_prefix_bin(&prefix_bin, script.to_str().unwrap(), &[]).is_none());
    }
}
