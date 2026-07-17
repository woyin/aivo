//! Safety classification: mutating/read-only/dangerous/catastrophic gates
//! and bash command analysis backing them.

use super::*;

/// Side-effecting tools the client must permission-gate before `execute`.
pub fn is_mutating(name: &str) -> bool {
    matches!(
        name,
        "write_file" | "edit_file" | "multi_edit" | "apply_patch" | "run_bash"
    )
}

/// Built-in tools that only read (filesystem or network) and share no mutable
/// state, so several can run concurrently within one tool-call batch. A
/// deliberate allowlist: writes and `run_bash` mutate the workspace; plan /
/// skill / subagent and external (MCP) tools mutate the engine or need ordered
/// permission handling, so they stay sequential even though they aren't here.
pub fn is_parallel_safe(name: &str) -> bool {
    matches!(
        name,
        "read_file" | "glob" | "grep" | "web_fetch" | "web_search"
    )
}

/// Whether a tool only reads — never touches the workspace. A conservative
/// allowlist: a missing read-only tool just costs a redundant `/rewind` snapshot,
/// but classifying a mutating tool here would lose its rewind point. Kept distinct
/// from [`is_parallel_safe`] on purpose — that answers a concurrency question, and
/// e.g. `list_dir` is read-only here yet not parallel-run there.
pub fn is_read_only(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_dir"
            | "glob"
            | "grep"
            | "web_fetch"
            | "web_search"
            // Session controls — change session state, not the workspace.
            | "switch_model"
            | "set_effort"
            // Interactive prompt — reads the user's answer, touches nothing.
            | "ask_user"
            // Job control on a process the agent itself started; never touches files.
            | "check_job"
            // Loads deferred MCP schemas — engine state only, never the workspace.
            | "search_tools"
    )
}

/// Whether a tool call warrants a confirmation prompt. Only genuinely risky
/// actions are gated — a destructive shell command, or a write/edit that leaves
/// the working directory. Ordinary in-project edits and benign commands
/// (`cargo test`, `ls`, `git status`, …) run without interruption.
pub fn is_dangerous(name: &str, args: &Value, cwd: &Path) -> bool {
    match name {
        "run_bash" => args
            .get("command")
            .and_then(|c| c.as_str())
            .map(bash_looks_destructive)
            .unwrap_or(false),
        "write_file" | "edit_file" | "multi_edit" => args
            .get("path")
            .and_then(|p| p.as_str())
            .map(|p| path_escapes_cwd(p, cwd))
            .unwrap_or(false),
        // A patch may touch many files; gate it if *any* target leaves the cwd.
        "apply_patch" => args
            .get("input")
            .and_then(|p| p.as_str())
            .map(|p| {
                crate::agent::apply_patch::target_paths(p)
                    .iter()
                    .any(|t| path_escapes_cwd(t, cwd))
            })
            .unwrap_or(false),
        _ => false,
    }
}

/// A hard floor under [`is_dangerous`]: an unrecoverable `run_bash` command (see
/// [`bash_is_catastrophic`]) the engine confirms even under auto-approve.
pub fn is_catastrophic(name: &str, args: &Value) -> bool {
    name == "run_bash"
        && args
            .get("command")
            .and_then(|c| c.as_str())
            .map(bash_is_catastrophic)
            .unwrap_or(false)
}

/// A `run_bash` command that mutates remote/cloud/API state (see
/// [`bash_mutates_remote`]); the engine confirms it even under auto-approve.
pub fn is_remote_side_effect(name: &str, args: &Value) -> bool {
    name == "run_bash"
        && args
            .get("command")
            .and_then(|c| c.as_str())
            .map(bash_mutates_remote)
            .unwrap_or(false)
}

/// Plan-mode allowlist: `true` only for provably read-only inspection commands,
/// which run without the per-call confirmation. Fail-closed (worst case: prompt).
pub fn is_readonly_command(args: &Value) -> bool {
    args.get("command")
        .and_then(|c| c.as_str())
        .map(bash_is_readonly)
        .unwrap_or(false)
}

/// Every segment must be a known inspection binary with no write-capable syntax
/// (substitution, non-pseudo-device redirect). Quotes are NOT parsed — a quoted
/// `$(…)`/`>` fails closed (a prompt), never a false pass.
pub(super) fn bash_is_readonly(cmd: &str) -> bool {
    let cmd = cmd.trim();
    if cmd.is_empty()
        || cmd.contains("$(")
        || cmd.contains('`')
        || cmd.contains("<(")
        || cmd.contains(">(")
        || has_file_write_redirect(cmd)
    {
        return false;
    }
    // Drop fd-dups (`2>&1`, `>&2`, …) before the walk: their `&` would otherwise
    // read as a control operator and mint a bogus `1`/`2` segment.
    let scrubbed = cmd
        .replace("2>&1", "")
        .replace("1>&2", "")
        .replace(">&1", "")
        .replace(">&2", "");
    let mut saw_command = false;
    for seg in scrubbed.split(['\n', ';', '|', '&']) {
        let tokens: Vec<&str> = seg.split_whitespace().collect();
        let Some(&cmd0) = tokens.first() else {
            continue;
        };
        saw_command = true;
        // An env-var prefix (`FOO=bar cmd`) hides the real command.
        if cmd0.contains('=') {
            return false;
        }
        let base = cmd0.rsplit('/').next().unwrap_or(cmd0);
        let ok = match base {
            "cd" | "ls" | "pwd" | "cat" | "head" | "tail" | "wc" | "file" | "stat" | "du"
            | "df" | "which" | "date" | "echo" | "printf" | "tree" | "realpath" | "dirname"
            | "basename" | "uname" | "type" | "grep" | "egrep" | "fgrep" | "rg" | "jq" | "diff"
            | "cmp" | "cut" | "uniq" | "tr" | "nl" | "column" | "strings" | "true" => true,
            // `sort -o file` writes; plain sort only prints.
            "sort" => !tokens
                .iter()
                .any(|t| *t == "-o" || t.starts_with("--output")),
            // `-delete` removes matches; `-exec` family runs arbitrary commands;
            // `-fprint*` writes files.
            "find" => !tokens.iter().any(|t| {
                matches!(*t, "-delete" | "-exec" | "-execdir" | "-ok" | "-okdir")
                    || t.starts_with("-fprint")
            }),
            "git" => git_subcommand_is_readonly(&tokens),
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    saw_command
}

/// Any `>`/`>>` output redirect whose target is not a safe `/dev/` pseudo-device
/// or a bare fd dup (`2>&1`). Input redirects (`<`, heredocs) don't write.
pub(super) fn has_file_write_redirect(cmd: &str) -> bool {
    let mut search = cmd;
    while let Some(pos) = search.find('>') {
        let mut rest = &search[pos + 1..];
        if let Some(stripped) = rest.strip_prefix('>') {
            rest = stripped;
        }
        search = rest;
        let target = rest.trim_start();
        // An fd dup (`2>&1`, `>&2`) writes nothing to disk.
        if target.starts_with('&') {
            continue;
        }
        let target: String = target.chars().take_while(|c| !c.is_whitespace()).collect();
        let dev_name = target.strip_prefix("/dev/").map(|dev| {
            dev.chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect::<String>()
        });
        if !dev_name.is_some_and(|name| SAFE_DEVICES.contains(&name.as_str())) {
            return true;
        }
    }
    false
}

/// `git <sub>` where `<sub>` only reads; an unrecognized global flag fails closed.
pub(super) fn git_subcommand_is_readonly(tokens: &[&str]) -> bool {
    let mut it = tokens.iter().skip(1);
    while let Some(&t) = it.next() {
        match t {
            // `-c key=value` executes: `core.fsmonitor` runs on `status`, `core.pager` on output.
            "-c" => return false,
            "-C" => {
                it.next();
            }
            "--no-pager" | "-P" => {}
            _ if t.starts_with('-') => return false,
            _ => {
                return matches!(
                    t,
                    "status"
                        | "log"
                        | "diff"
                        | "show"
                        | "blame"
                        | "shortlog"
                        | "describe"
                        | "rev-parse"
                        | "ls-files"
                        | "ls-tree"
                        | "cat-file"
                        | "grep"
                        | "reflog"
                        | "show-ref"
                        | "count-objects"
                );
            }
        }
    }
    false // bare `git` — nothing to judge
}

/// True when a path resolves outside the working directory (absolute elsewhere,
/// `..` traversal, or **through a symlink** that points out of the project) —
/// editing outside the project is worth confirming.
///
/// A purely lexical check follows a symlink blindly: `repo/link/file` where
/// `repo/link -> /outside` normalizes to `repo/link/file`, which *looks*
/// in-project, yet the write lands at `/outside/file`. So we resolve symlinks by
/// canonicalizing the workspace root and the target's closest existing ancestor
/// (the file itself may not exist yet) before comparing.
pub(super) fn path_escapes_cwd(path: &str, cwd: &Path) -> bool {
    // `--add-dir` roots are part of the workspace too.
    path_escapes_roots(path, cwd, crate::agent::sandbox::extra_write_roots())
}

pub(super) fn path_escapes_roots(path: &str, cwd: &Path, extra: &[PathBuf]) -> bool {
    let target = canonicalize_existing_ancestor(&resolve(cwd, path));
    if target.starts_with(canonicalize_existing_ancestor(cwd)) {
        return false;
    }
    !extra
        .iter()
        .any(|root| target.starts_with(canonicalize_existing_ancestor(root)))
}

/// Resolve `path` as far as the filesystem allows: canonicalize the longest
/// existing prefix (following every symlink in it), then re-attach the remaining
/// not-yet-created components and collapse any `.`/`..` lexically. A symlink can
/// only exist within the existing prefix, so this catches a link that escapes
/// the workspace while still working for a brand-new file. Falls back to a
/// lexical normalize when nothing canonicalizes (so `..` traversal is still
/// rejected).
pub(super) fn canonicalize_existing_ancestor(path: &Path) -> PathBuf {
    let comps: Vec<Component> = path.components().collect();
    for split in (0..=comps.len()).rev() {
        let mut prefix = PathBuf::new();
        for comp in &comps[..split] {
            prefix.push(comp.as_os_str());
        }
        if let Ok(canon) = prefix.canonicalize() {
            let mut out = canon;
            for comp in &comps[split..] {
                out.push(comp.as_os_str());
            }
            return lexical_normalize(&out);
        }
    }
    lexical_normalize(path)
}

/// Collapse `.`/`..` components lexically (no filesystem access, so it works for
/// not-yet-created files).
pub(super) fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Interpreters that execute their *program* from piped stdin (`curl … | sh`)
/// — the classic remote-code-execution shape. Matched only when the piped
/// invocation reads code from stdin (bare, `-`, `-c`, or `-s`); an interpreter
/// given a script file or `-m module` is just consuming data and is left alone.
pub(super) const INTERPRETERS: &[&str] = &[
    "sh", "bash", "zsh", "fish", "dash", "ksh", "python", "python2", "python3", "node", "nodejs",
    "ruby", "perl", "php", "pwsh",
];

/// `/dev/` entries harmless to write to; anything else is a real device.
pub(super) const SAFE_DEVICES: &[&str] = &[
    "null", "zero", "stdin", "stdout", "stderr", "tty", "fd", "full", "random", "urandom",
];

/// Detects destructive shell commands that should be confirmed before running
/// (and highlighted ⚠ in the card). Best-effort and advisory — a heuristic, not
/// a sandbox: it tokenizes per simple-command (so flag order / extra spaces
/// don't defeat it, and `cargo add` is no longer mistaken for `dd`), but a
/// determined command can still slip past. The real guard is the user's eyes.
pub fn bash_looks_destructive(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();

    // Piping fetched bytes into a shell/interpreter that runs them as code.
    if pipes_into_interpreter(&lower) {
        return true;
    }

    // Inspect the leading command of each segment between control operators.
    for seg in lower.split(['\n', ';', '|', '&']) {
        let tokens: Vec<&str> = seg.split_whitespace().collect();
        let Some(&cmd0) = tokens.first() else {
            continue;
        };
        let base = cmd0.rsplit('/').next().unwrap_or(cmd0); // strip a leading path
        // An inline-code interpreter (`sh -c '…'`, `python -c '…'`, `perl -e '…'`)
        // hides its real command inside a quoted argument the per-command walk
        // can't reach — it only reads the leading `sh`/`python`. Pull that program
        // out and scan it on its own.
        if INTERPRETERS.contains(&base)
            && interpreter_inline_code(seg).is_some_and(|inner| bash_looks_destructive(&inner))
        {
            return true;
        }
        let flagged = match base {
            "rm" => has_short_or_long(&tokens, &['r', 'f'], &["recursive", "force"]),
            "mkfs" | "shred" | "dd" => true,
            "chmod" | "chown" | "chgrp" => {
                has_short_or_long(&tokens, &['r'], &["recursive"]) || tokens.contains(&"-R")
            }
            "sudo" | "doas" | "su" => true,
            "git" => git_is_destructive(&tokens),
            // `-delete` removes matches; `-exec`/`-execdir` run an arbitrary
            // command per match (`find . -exec rm {} \;` is the classic deleter
            // that `-delete` alone misses).
            "find" => tokens
                .iter()
                .any(|t| matches!(*t, "-delete" | "-exec" | "-execdir")),
            _ => false,
        };
        if flagged {
            return true;
        }
    }

    // Residual patterns the per-command walk doesn't structurally cover.
    const SIGNALS: &[&str] = &[":(){", "truncate -s 0"];
    if SIGNALS.iter().any(|s| lower.contains(s)) {
        return true;
    }

    // Redirecting onto a raw device (`> /dev/sda`) clobbers a disk; redirecting
    // to `/dev/null` and the other pseudo-devices (`2>/dev/null`, `>/dev/stderr`,
    // …) is routine and must not prompt.
    redirects_to_real_device(&lower)
}

/// The un-waivable core under [`bash_looks_destructive`]: commands that are
/// unrecoverable or system-wide. Deliberately FAR narrower — a workspace-local
/// `rm -rf ./build` must stay out, or unattended (`/goal`, `-y`) runs break.
pub(super) fn bash_is_catastrophic(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();

    // Fork bomb, in either canonical spacing.
    if lower
        .split_whitespace()
        .collect::<String>()
        .contains(":(){:|:&};:")
    {
        return true;
    }

    for seg in lower.split(['\n', ';', '|', '&']) {
        let all: Vec<&str> = seg.split_whitespace().collect();
        let tokens = effective_command(&all); // see-through `sudo`/`env`/`nice`
        let Some(&cmd0) = tokens.first() else {
            continue;
        };
        let base = cmd0.rsplit('/').next().unwrap_or(cmd0);
        // `sh -c 'rm -rf /'` hides the real command in a quoted arg — rescan it.
        if INTERPRETERS.contains(&base)
            && interpreter_inline_code(seg).is_some_and(|inner| bash_is_catastrophic(&inner))
        {
            return true;
        }
        let hit = match base {
            "rm" => {
                has_short_or_long(tokens, &['r'], &["recursive"])
                    && tokens.iter().skip(1).any(|t| is_root_or_home_target(t))
            }
            b if b == "mkfs" || b.starts_with("mkfs.") => true,
            "dd" => tokens
                .iter()
                .any(|t| t.strip_prefix("of=").is_some_and(is_raw_device_path)),
            "chmod" | "chown" | "chgrp" => {
                has_short_or_long(tokens, &['r'], &["recursive"])
                    && tokens
                        .iter()
                        .skip(1)
                        .any(|t| strip_matching_quotes(t) == "/")
            }
            "shutdown" | "reboot" | "halt" | "poweroff" => true,
            "init" => matches!(tokens.get(1), Some(&"0") | Some(&"6")),
            _ => false,
        };
        if hit || windows_seg_is_catastrophic(tokens) {
            return true;
        }
    }

    redirects_to_real_device(&lower) // `cat img > /dev/sda`
}

/// Windows half of [`bash_is_catastrophic`]: `run_bash` shells through PowerShell,
/// which the POSIX walk misses. Tokens are lowercased (cmd/PowerShell are
/// case-insensitive).
pub(super) fn windows_seg_is_catastrophic(tokens: &[&str]) -> bool {
    let Some(&cmd0) = tokens.first() else {
        return false;
    };
    let base = cmd0.rsplit(['\\', '/']).next().unwrap_or(cmd0);
    let base = base
        .strip_suffix(".exe")
        .or_else(|| base.strip_suffix(".com"))
        .unwrap_or(base);
    let args = &tokens[1..];
    match base {
        "format-volume" | "clear-disk" | "stop-computer" | "restart-computer" => true,
        "format" => args.iter().any(|a| is_windows_root_target(a)),
        "cipher" => args.iter().any(|a| a.starts_with("/w")), // wipe free space
        // Recursive delete of a root — aliases ri/rm/del/erase/rd/rmdir all map to
        // Remove-Item; recurse is `/s` (cmd) or `-recurse`/`-r` (PowerShell).
        "remove-item" | "ri" | "rm" | "del" | "erase" | "rd" | "rmdir" => {
            let recursive = args
                .iter()
                .any(|a| *a == "/s" || *a == "-r" || a.starts_with("-rec"));
            recursive && args.iter().any(|a| is_windows_root_target(a))
        }
        _ => false,
    }
}

/// A Windows drive/home/system root (`C:\`, `\`, `~`, `$env:`/`%…%`) whose
/// recursive deletion is unrecoverable. A subpath is not matched.
pub(super) fn is_windows_root_target(arg: &str) -> bool {
    let arg = strip_matching_quotes(arg);
    let trimmed = arg.trim_end_matches(['*', '\\', '/']);
    // "<letter>:" drive root.
    if let [letter, b':'] = trimmed.as_bytes()
        && letter.is_ascii_alphabetic()
    {
        return true;
    }
    // Root of the current drive (`\`, `/`) — but not a bare `*`.
    if trimmed.is_empty() && (arg.starts_with('\\') || arg.starts_with('/')) {
        return true;
    }
    matches!(
        trimmed,
        "~" | "$home"
            | "${home}"
            | "$env:systemdrive"
            | "$env:userprofile"
            | "$env:homedrive"
            | "$env:systemroot"
            | "$env:windir"
            | "%systemdrive%"
            | "%userprofile%"
            | "%homedrive%"
            | "%systemroot%"
            | "%windir%"
    )
}

/// Strip a leading privilege/env/scheduling wrapper so `sudo rm -rf /` and
/// `env X=1 rm -rf /` classify as `rm -rf /`. Best-effort.
pub(super) fn effective_command<'a>(tokens: &'a [&'a str]) -> &'a [&'a str] {
    let mut rest = tokens;
    loop {
        let Some((&head, tail)) = rest.split_first() else {
            return rest;
        };
        match head.rsplit('/').next().unwrap_or(head) {
            "sudo" | "doas" => {
                rest = tail;
                while let Some((&t, tl)) = rest.split_first() {
                    let Some(flag) = t.strip_prefix('-').filter(|s| !s.is_empty()) else {
                        break; // first non-option token is the wrapped command
                    };
                    rest = tl;
                    if t == "--" {
                        break;
                    }
                    // -u/-g/-p/-C/-h/-r/-t (and --long forms) take an argument.
                    if matches!(
                        flag.chars().last(),
                        Some('u' | 'g' | 'p' | 'c' | 'h' | 'r' | 't')
                    ) {
                        rest = rest.split_first().map_or(rest, |(_, tl)| tl);
                    }
                }
            }
            "env" => {
                rest = tail;
                while let Some((&t, tl)) = rest.split_first() {
                    if !t.starts_with('-') && t.contains('=') {
                        rest = tl;
                    } else {
                        break;
                    }
                }
            }
            "nice" | "nohup" | "stdbuf" | "ionice" | "time" => rest = tail,
            _ => return rest,
        }
    }
}

/// An `rm` target whose recursive deletion is unrecoverable — `/`, `~`, `$HOME`,
/// or the whole cwd (`.`), with or without a trailing `/` or `/*`. A workspace
/// subpath (`./build`, `~/Documents`) is not matched. `arg` arrives lowercased.
pub(super) fn is_root_or_home_target(arg: &str) -> bool {
    let arg = strip_matching_quotes(arg);
    let base = arg.strip_suffix("/*").unwrap_or(arg).trim_end_matches('/');
    if arg.starts_with('/') && base.is_empty() {
        return true; // "/", "//", "/*"
    }
    matches!(base, "~" | "$home" | "${home}" | ".")
}

/// Strip one layer of matching surrounding quotes (`"$HOME"` → `$HOME`).
pub(super) fn strip_matching_quotes(arg: &str) -> &str {
    let bytes = arg.as_bytes();
    match (bytes.first(), bytes.last()) {
        (Some(b'"'), Some(b'"')) | (Some(b'\''), Some(b'\'')) if bytes.len() >= 2 => {
            &arg[1..arg.len() - 1]
        }
        _ => arg,
    }
}

/// A real `/dev/` block device (`/dev/sda`) vs. a harmless pseudo-device.
pub(super) fn is_raw_device_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("/dev/") else {
        return false;
    };
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    !name.is_empty() && !SAFE_DEVICES.contains(&name.as_str())
}

/// True when the command redirects output (`>`/`>>`, optionally with a leading fd
/// like `2>`) onto a `/dev/` entry that is NOT a harmless pseudo-device. Writing
/// to `/dev/sda` overwrites a disk; `2>/dev/null` is everyday noise-suppression.
pub(super) fn redirects_to_real_device(cmd: &str) -> bool {
    let mut search = cmd;
    while let Some(pos) = search.find("/dev/") {
        // Only a write redirection counts: the bytes just before `/dev/` must end
        // in `>` (covers `>`, `>>`, `2>`, `&>`, and a space-separated `> /dev/…`).
        let is_redirect = search[..pos].trim_end().ends_with('>');
        if is_redirect {
            let name: String = search[pos + "/dev/".len()..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !SAFE_DEVICES.contains(&name.as_str()) {
                return true;
            }
        }
        search = &search[pos + "/dev/".len()..];
    }
    false
}

/// True when a later pipeline stage is an interpreter reading its program from
/// stdin (bare, `-`, `-c`, or `-s`). `cat x | python -m json.tool` (data, not
/// code) and `… | grep foo` are deliberately not flagged.
pub(super) fn pipes_into_interpreter(cmd: &str) -> bool {
    cmd.split('|').skip(1).any(|seg| {
        let mut words = seg.split_whitespace();
        let Some(w) = words.next() else {
            return false;
        };
        let base = w.rsplit('/').next().unwrap_or(w);
        if !INTERPRETERS.contains(&base) {
            return false;
        }
        let rest: Vec<&str> = words.collect();
        rest.is_empty() || rest.iter().any(|a| matches!(*a, "-" | "-c" | "-s"))
    })
}

// --- remote side-effect classifier ---
//
// Flags `run_bash` commands that mutate remote/cloud/API state (deploy, publish,
// DELETE) the workspace card can't undo. Best-effort like [`bash_looks_destructive`];
// biased to leave reads alone and under-flag rather than nag.

/// Split a command line into tokens, honoring single/double quotes and stripping
/// them, so `sh -c 'rm -rf x'` yields `["sh", "-c", "rm -rf x"]`. Best-effort: an
/// unmatched quote just runs to the end of the string.
pub(super) fn shell_split(cmd: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut started = false;
    let mut in_single = false;
    let mut in_double = false;
    for c in cmd.chars() {
        match c {
            '\'' if !in_double => {
                in_single = !in_single;
                started = true;
            }
            '"' if !in_single => {
                in_double = !in_double;
                started = true;
            }
            c if c.is_whitespace() && !in_single && !in_double => {
                if started {
                    tokens.push(std::mem::take(&mut cur));
                    started = false;
                }
            }
            c => {
                cur.push(c);
                started = true;
            }
        }
    }
    if started {
        tokens.push(cur);
    }
    tokens
}

/// If `seg` is an interpreter invoked with an inline program — `sh -c '…'`,
/// `python -c '…'`, `perl -e '…'`, `node --eval '…'`, `pwsh -Command '…'` —
/// return that program so the caller can re-scan it. The per-command walk
/// otherwise only sees the leading `sh`/`python` and waves the wrapper through.
pub(super) fn interpreter_inline_code(seg: &str) -> Option<String> {
    let tokens = shell_split(seg);
    let first = tokens.first()?;
    let base = first.rsplit('/').next().unwrap_or(first);
    if !INTERPRETERS.contains(&base) {
        return None;
    }
    let mut rest = tokens.iter().skip(1);
    while let Some(tok) = rest.next() {
        // `-c` (sh/bash/zsh/python/node), `-e`/`--eval` (perl/ruby/node),
        // `-command` (pwsh, already lowercased) all introduce inline code as the
        // following argument.
        let Some(flag) = tok.strip_prefix('-') else {
            continue;
        };
        let flag = flag.trim_start_matches('-');
        if matches!(flag, "c" | "e" | "eval" | "command") {
            return rest.next().cloned();
        }
    }
    None
}

/// True when any non-leading token is a combined short flag containing one of
/// `shorts` (e.g. `-rf`) or a `--long` flag in `longs`.
pub(super) fn has_short_or_long(tokens: &[&str], shorts: &[char], longs: &[&str]) -> bool {
    tokens.iter().skip(1).any(|t| {
        if let Some(long) = t.strip_prefix("--") {
            longs.contains(&long)
        } else if let Some(short) = t.strip_prefix('-').filter(|s| !s.starts_with('-')) {
            short.chars().any(|c| shorts.contains(&c))
        } else {
            false
        }
    })
}

/// Git subcommands worth confirming: anything that rewrites history, touches a
/// remote, or discards working-tree state. Read-only and routine commands
/// (`status`, `log`, `checkout -b`, `reset` without `--hard`) pass through.
pub(super) fn git_is_destructive(tokens: &[&str]) -> bool {
    // Find the real subcommand by skipping git's *global* options. Several of
    // them take a separate argument (`-C <path>`, `-c <name>=val`,
    // `--git-dir <path>`, `--work-tree <path>`, …); consuming that argument is
    // what stops `git -C . reset --hard` from mistaking the `.` for the
    // subcommand. (The command was lowercased upstream, so `-C` arrives as `-c` —
    // both forms take an argument, so collapsing them here is harmless.)
    let mut it = tokens.iter().skip(1).copied();
    let mut sub: Option<&str> = None;
    let mut rest: Vec<&str> = Vec::new();
    while let Some(a) = it.next() {
        if matches!(
            a,
            "-c" | "--git-dir" | "--work-tree" | "--namespace" | "--super-prefix" | "--config-env"
        ) {
            it.next(); // skip the option's argument so it can't pose as the subcommand
            continue;
        }
        if a.starts_with('-') {
            continue; // argument-less global flag (`-p`, `--no-pager`, `--bare`, …)
        }
        sub = Some(a);
        rest = it.collect(); // everything after the subcommand holds its flags
        break;
    }
    let Some(sub) = sub else {
        return false;
    };
    let has = |flag: &str| rest.contains(&flag);
    match sub {
        "push" | "commit" | "restore" => true,
        "reset" => has("--hard"),
        "clean" => rest
            .iter()
            .any(|t| t.starts_with("-f") || *t == "-d" || *t == "-x" || *t == "--force"),
        "checkout" => has("--") || has("-f") || has("--force"),
        "branch" => has("-d") || has("--delete"),
        "rebase" => !(has("--abort") || has("--continue") || has("--skip") || has("--quit")),
        _ => false,
    }
}
