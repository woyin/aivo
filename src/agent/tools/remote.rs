//! Remote side-effect detection: which bash commands write to the outside
//! world (deploy/publish/API mutations) and the prefix families for grants.

use super::*;

pub(super) fn is_mutating_http_method(m: &str) -> bool {
    ["POST", "PUT", "PATCH", "DELETE"]
        .iter()
        .any(|v| m.eq_ignore_ascii_case(v))
}

/// Cloud/infra-CLI subcommand verbs that write remote state; read verbs
/// (get/list/describe/…) are absent so routine queries don't prompt.
pub(super) const EXACT_REMOTE_VERBS: &[&str] = &[
    "create",
    "delete",
    "remove",
    "rm",
    "destroy",
    "update",
    "modify",
    "edit",
    "patch",
    "put",
    "set",
    "add",
    "apply",
    "install",
    "uninstall",
    "upgrade",
    "rollback",
    "deploy",
    "publish",
    "unpublish",
    "release",
    "push",
    "upload",
    "send",
    "register",
    "deregister",
    "attach",
    "detach",
    "associate",
    "disassociate",
    "enable",
    "disable",
    "import",
    "restore",
    "cancel",
    "purge",
    "drop",
    "scale",
    "provision",
    "mb",
    "rb",
    "up",
    "revoke",
    "grant",
    "merge",
    "close",
    "reopen",
    "rename",
    "transfer",
    "fork",
    "dispatch",
    "rerun",
    "sync",
    "promote",
    "cordon",
    "uncordon",
    "drain",
    "evict",
    "annotate",
    "label",
    "expose",
    "autoscale",
    "start",
    "stop",
    "restart",
    "reboot",
    "resume",
    "pause",
    "undo",
    "taint",
    "untaint",
    "deprecate",
    "yank",
];

/// AWS-style `verb-noun` prefixes (`delete-object`, `run-instances`). Only used on
/// a hyphenated token, so a bare word like `run` (`gh run list`) never trips it.
pub(super) const DASH_REMOTE_VERBS: &[&str] = &[
    "create",
    "delete",
    "remove",
    "update",
    "modify",
    "put",
    "set",
    "add",
    "terminate",
    "run",
    "reboot",
    "start",
    "stop",
    "reset",
    "send",
    "publish",
    "deploy",
    "register",
    "deregister",
    "attach",
    "detach",
    "associate",
    "disassociate",
    "enable",
    "disable",
    "import",
    "restore",
    "cancel",
    "purge",
    "drop",
    "revoke",
    "grant",
    "promote",
    "copy",
    "move",
    "replace",
    "apply",
    "restart",
    "resume",
    "scale",
    "tag",
    "untag",
    "authorize",
    "allocate",
    "provision",
    "rebuild",
    "redeploy",
    "rollback",
    "upgrade",
    "downgrade",
];

/// Lowercased subcommand path up to and including the first `verb` hit — the
/// grantable family (`az repos pr update`). Stops at the first flag so a flag
/// value (`--title "delete old"`) can't trip it.
pub(super) fn verb_path(base: &str, args: &[&str], verb: impl Fn(&str) -> bool) -> Option<String> {
    let mut path = base.to_string();
    for &tok in args {
        if tok.starts_with('-') {
            break;
        }
        let t = tok.to_ascii_lowercase();
        path.push(' ');
        path.push_str(&t);
        if verb(&t) {
            return Some(path);
        }
    }
    None
}

/// Exact mutating verb or AWS-style `verb-noun` prefix — dash form only on a
/// hyphenated token, so a bare `run` (`gh run list`) never trips it.
pub(super) fn cloud_verb(t: &str) -> bool {
    if EXACT_REMOTE_VERBS.contains(&t) {
        return true;
    }
    t.split_once('-')
        .is_some_and(|(verb, rest)| !rest.is_empty() && DASH_REMOTE_VERBS.contains(&verb))
}

/// `curl` mutates on a mutating method (`-X POST`) or a request body (`-d`, `-F`,
/// `-T`, `--json`) without an explicit GET/HEAD. Case-sensitive: `-F` form ≠ `-f`
/// fail, `-T` upload ≠ `-t`.
pub(super) fn curl_is_mutating(args: &[&str]) -> bool {
    let mut method_mutates = false;
    let mut method_readonly = false;
    let mut has_body = false;
    let mut it = args.iter().peekable();
    while let Some(&a) = it.next() {
        let method = if a == "-X" || a == "--request" {
            it.next().copied()
        } else {
            a.strip_prefix("-X").filter(|m| !m.is_empty())
        };
        if let Some(m) = method {
            if is_mutating_http_method(m) {
                method_mutates = true;
            } else if m.eq_ignore_ascii_case("GET") || m.eq_ignore_ascii_case("HEAD") {
                method_readonly = true;
            }
            continue;
        }
        // `-G`/`--get` sends any `-d` data as a GET query string, not a body.
        if a == "-G" || a == "--get" {
            method_readonly = true;
        }
        if a == "-F"
            || a == "--form"
            || a == "-T"
            || a == "--upload-file"
            || a == "-d"
            || a == "--json"
            || a.starts_with("--data")
        {
            has_body = true;
        }
    }
    method_mutates || (has_body && !method_readonly)
}

/// `wget` mutates only with an explicit non-GET method or a POST/body payload;
/// its default (and the common case) is a read-only download.
pub(super) fn wget_is_mutating(args: &[&str]) -> bool {
    let mut it = args.iter().peekable();
    while let Some(&a) = it.next() {
        if a == "--method" {
            if it.peek().is_some_and(|m| is_mutating_http_method(m)) {
                return true;
            }
        } else if let Some(m) = a.strip_prefix("--method=") {
            if is_mutating_http_method(m) {
                return true;
            }
        } else if a.starts_with("--post-data")
            || a.starts_with("--post-file")
            || a.starts_with("--body-data")
            || a.starts_with("--body-file")
        {
            return true;
        }
    }
    false
}

/// HTTPie (`http`/`https`/`xh`): a leading positional METHOD token that mutates.
pub(super) fn httpie_is_mutating(args: &[&str]) -> bool {
    args.iter()
        .take_while(|a| !a.starts_with('-'))
        .any(|a| is_mutating_http_method(a))
}

/// `gh api …` defaults to GET; a mutating `-X/--method` or a field flag (which
/// forces a non-GET request) makes it mutate.
pub(super) fn gh_api_mutates(args: &[&str]) -> bool {
    if args.first().is_none_or(|a| !a.eq_ignore_ascii_case("api")) {
        return false;
    }
    let mut it = args.iter().peekable();
    while let Some(&a) = it.next() {
        if a == "-X" || a == "--method" {
            if it.peek().is_some_and(|m| is_mutating_http_method(m)) {
                return true;
            }
        } else if let Some(m) = a.strip_prefix("--method=") {
            if is_mutating_http_method(m) {
                return true;
            }
        } else if matches!(a, "-f" | "-F" | "--field" | "--raw-field" | "--input")
            || a.starts_with("--field=")
            || a.starts_with("--raw-field=")
        {
            return true;
        }
    }
    false
}

/// Read-only; mutating with a grantable verb path; or mutating with none
/// (curl bodies, `gh api -f`) — exact-grant only.
pub(super) enum RemoteSeg {
    Read,
    Opaque,
    Verb(String),
}

/// Classify one already-split simple command by its program.
pub(super) fn remote_segment(base: &str, args: &[&str]) -> RemoteSeg {
    use RemoteSeg::{Opaque, Read, Verb};
    let flagged = |mutates: bool| if mutates { Opaque } else { Read };
    let by_verbs = |verbs: &[&str]| match verb_path(base, args, |t| verbs.contains(&t)) {
        Some(p) => Verb(p),
        None => Read,
    };
    match base {
        "curl" => flagged(curl_is_mutating(args)),
        "wget" => flagged(wget_is_mutating(args)),
        "http" | "https" | "xh" | "xhs" => flagged(httpie_is_mutating(args)),
        "gh" | "glab" => match verb_path(base, args, cloud_verb) {
            Some(p) => Verb(p),
            None => flagged(gh_api_mutates(args)),
        },
        "aws" | "gcloud" | "gsutil" | "az" | "oci" | "doctl" | "ibmcloud" | "kubectl" | "oc"
        | "terraform" | "tofu" | "terragrunt" | "pulumi" | "flux" | "argocd" | "eksctl" => {
            match verb_path(base, args, cloud_verb) {
                Some(p) => Verb(p),
                None => Read,
            }
        }
        // Helm's `create`/`repo add` are local, so list its remote verbs explicitly.
        "helm" => by_verbs(&[
            "install",
            "upgrade",
            "uninstall",
            "delete",
            "rollback",
            "push",
        ]),
        // Deploy / hosting CLIs push to prod.
        "vercel" | "netlify" | "flyctl" | "fly" | "railway" | "heroku" | "wrangler"
        | "supabase" | "firebase" | "eb" | "serverless" | "sls" | "now" | "surge" | "amplify"
        | "convex" | "render" => by_verbs(&[
            "deploy", "publish", "release", "up", "promote", "rollback", "ship", "push",
        ]),
        // Container / package registries: publishing is public + hard to retract.
        "docker" | "podman" | "nerdctl" => by_verbs(&["push"]),
        "npm" | "pnpm" | "yarn" | "bun" => by_verbs(&["publish", "unpublish", "deprecate"]),
        "cargo" => by_verbs(&["publish", "yank"]),
        "gem" => by_verbs(&["push", "yank"]),
        "twine" => by_verbs(&["upload", "register"]),
        _ => Read,
    }
}

/// Powers [`is_remote_side_effect`] and the card's ⚠ label. Kept case-sensitive
/// (unlike the destructive walk) because `curl -F` (form) ≠ `-f` (fail).
pub fn bash_mutates_remote(cmd: &str) -> bool {
    for seg in cmd.split(['\n', ';', '|', '&']) {
        let all: Vec<&str> = seg.split_whitespace().collect();
        let tokens = effective_command(&all); // see-through sudo/env/nice
        let Some(&cmd0) = tokens.first() else {
            continue;
        };
        let base = cmd0.rsplit('/').next().unwrap_or(cmd0).to_ascii_lowercase();
        // `sh -c 'curl -X POST …'` hides the real command in a quoted arg — rescan it.
        if INTERPRETERS.contains(&base.as_str())
            && interpreter_inline_code(seg).is_some_and(|inner| bash_mutates_remote(&inner))
        {
            return true;
        }
        if !matches!(remote_segment(&base, &tokens[1..]), RemoteSeg::Read) {
            return true;
        }
    }
    false
}

/// Grantable family prefixes of every remote-mutating segment, or empty when any
/// mutation has no verb path — callers then fall back to an exact grant.
pub fn remote_mutation_prefixes(cmd: &str) -> Vec<String> {
    let mut out = Vec::new();
    if !collect_remote_prefixes(cmd, &mut out) {
        return Vec::new();
    }
    out.sort();
    out.dedup();
    out
}

/// `false` = an opaque mutation somewhere; no prefix set can represent the command.
pub(super) fn collect_remote_prefixes(cmd: &str, out: &mut Vec<String>) -> bool {
    for seg in cmd.split(['\n', ';', '|', '&']) {
        let all: Vec<&str> = seg.split_whitespace().collect();
        let tokens = effective_command(&all);
        let Some(&cmd0) = tokens.first() else {
            continue;
        };
        let base = cmd0.rsplit('/').next().unwrap_or(cmd0).to_ascii_lowercase();
        if INTERPRETERS.contains(&base.as_str())
            && let Some(inner) = interpreter_inline_code(seg)
            && !collect_remote_prefixes(&inner, out)
        {
            return false;
        }
        match remote_segment(&base, &tokens[1..]) {
            RemoteSeg::Read => {}
            RemoteSeg::Opaque => return false,
            RemoteSeg::Verb(p) => out.push(p),
        }
    }
    true
}
