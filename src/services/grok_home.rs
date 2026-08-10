//! The managed grok home: `<config_dir>/grok-home`, one stable home so grok
//! keeps sessions, folder trust, and memory across launches. Only the leader
//! socket is per-launch — a shared socket name hung grok <=0.2.9x. A user-set
//! `GROK_HOME` opts out entirely. The path matches the aivo-grok plugin's
//! (<=v0.5.0), so plugin-era sessions and trust must keep resolving here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::services::atomic_write::atomic_write_secure_blocking;

pub const TRUST_FILE: &str = "trusted_folders.toml";
const MIGRATION_MARKER: &str = ".aivo-migrated-v1";
const RETENTION: Duration = Duration::from_secs(7 * 24 * 60 * 60);

pub fn grok_home_dir(config_dir: &Path) -> PathBuf {
    config_dir.join("grok-home")
}

/// Pre-0.5.0 plugin throwaway homes, migrated into the stable one and reaped.
pub fn legacy_homes_base(config_dir: &Path) -> PathBuf {
    config_dir.join("grok-homes")
}

pub fn sessions_dir(home: &Path) -> PathBuf {
    home.join("sessions")
}

pub fn trust_store(home: &Path) -> PathBuf {
    home.join(TRUST_FILE)
}

/// Session roots, most specific first: user `GROK_HOME` (else `~/.grok`),
/// managed home, unreaped legacy homes. Canonical-deduped so a symlinked
/// `GROK_HOME` can't make every session resolve twice.
pub fn session_roots(config_dir: &Path, user_grok_home: Option<&Path>) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    match user_grok_home {
        Some(h) => roots.push(sessions_dir(h)),
        None => {
            if let Some(home) = crate::services::system_env::home_dir() {
                roots.push(sessions_dir(&home.join(".grok")));
            }
        }
    }
    roots.push(sessions_dir(&grok_home_dir(config_dir)));
    let base = legacy_homes_base(config_dir);
    if let Ok(rd) = std::fs::read_dir(&base) {
        for entry in rd.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                roots.push(sessions_dir(&entry.path()));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|r| seen.insert(std::fs::canonicalize(r).unwrap_or_else(|_| r.clone())));
    roots
}

/// The one definition of "which grok homes count" for the ingest, probe, and
/// share paths; honors a `GROK_HOME` env override.
pub fn session_roots_from_system() -> Vec<PathBuf> {
    let user_home = std::env::var("GROK_HOME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from);
    session_roots(&crate::services::paths::config_dir(), user_home.as_deref())
}

/// Unique per launch so an opted-in leader can't be inherited by the next run.
pub fn launch_leader_socket(home: &Path) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    home.join(format!(
        "leader-{}-{}-{}.sock",
        std::process::id(),
        millis,
        seq
    ))
}

/// JS `encodeURIComponent` semantics (grok's session-dir naming). Its
/// unreserved set is wider than RFC 3986's, so `percent_codec` would
/// over-encode `!*'()` and miss dirs grok wrote.
pub fn encode_cwd_dir(cwd: &str) -> String {
    let mut out = String::with_capacity(cwd.len() * 3);
    for b in cwd.bytes() {
        match b {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => out.push(b as char),
            _ => {
                use std::fmt::Write;
                let _ = write!(out, "%{:02X}", b);
            }
        }
    }
    out
}

// ── TOML tables ───────────────────────────────────────────────────────────
//
// Parses just far enough to replace ONE table; everything else — grok's own
// settings, fields we don't model — round-trips verbatim.

#[derive(Debug, Clone, PartialEq)]
pub struct TomlTable {
    pub header: String,
    pub body: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TomlTables {
    /// Top-level keys grok wrote before any table.
    pub preamble: Vec<String>,
    /// Insertion-ordered; `set` replaces in place.
    pub tables: Vec<TomlTable>,
}

impl TomlTables {
    pub fn get(&self, header: &str) -> Option<&TomlTable> {
        let key = header_key_parts(header);
        self.tables
            .iter()
            .find(|t| header_key_parts(&t.header) == key)
    }

    pub fn set(&mut self, table: TomlTable) {
        let key = header_key_parts(&table.header);
        match self
            .tables
            .iter_mut()
            .find(|t| header_key_parts(&t.header) == key)
        {
            Some(slot) => *slot = table,
            None => self.tables.push(table),
        }
    }
}

/// Header key parts with quoting resolved: grok's config rewriter drops
/// quotes from bare-key-safe keys, so raw-string header matching would append
/// a duplicate table that grok's real TOML parser rejects.
fn header_key_parts(header: &str) -> Vec<String> {
    let inner = header.trim().trim_start_matches('[').trim_end_matches(']');
    let mut parts = Vec::new();
    let mut cur = String::new();
    let mut chars = inner.chars();
    let mut in_quotes = false;
    while let Some(c) = chars.next() {
        match c {
            '"' => in_quotes = !in_quotes,
            '\\' if in_quotes => {
                if let Some(escaped) = chars.next() {
                    cur.push(escaped);
                }
            }
            '.' if !in_quotes => {
                parts.push(std::mem::take(&mut cur).trim().to_string());
                continue;
            }
            _ => cur.push(c),
        }
    }
    parts.push(cur.trim().to_string());
    parts
}

fn is_table_header(line: &str) -> bool {
    let t = line.trim();
    t.len() >= 2 && t.starts_with('[') && t.ends_with(']') && !t[1..t.len() - 1].contains(']')
}

pub fn parse_toml_tables(text: &str) -> TomlTables {
    let mut out = TomlTables::default();
    let mut cur: Option<TomlTable> = None;
    for raw in text.split('\n') {
        let line = raw.trim_end();
        if is_table_header(line) {
            if let Some(done) = cur.take() {
                out.set(done);
            }
            cur = Some(TomlTable {
                header: line.trim().to_string(),
                body: Vec::new(),
            });
        } else if line.trim().is_empty() {
            // blank — re-added on serialize
        } else if let Some(t) = cur.as_mut() {
            t.body.push(line.to_string());
        } else {
            out.preamble.push(line.to_string());
        }
    }
    if let Some(done) = cur.take() {
        out.set(done);
    }
    out
}

pub fn serialize_toml_tables(parsed: &TomlTables) -> String {
    let mut out: Vec<String> = parsed.preamble.clone();
    for t in &parsed.tables {
        if !out.is_empty() {
            out.push(String::new());
        }
        out.push(t.header.clone());
        out.extend(t.body.iter().cloned());
    }
    out.push(String::new());
    out.join("\n")
}

fn read_toml_text(file: &Path) -> String {
    std::fs::read_to_string(file).unwrap_or_default()
}

fn read_toml_tables(file: &Path) -> TomlTables {
    parse_toml_tables(&read_toml_text(file))
}

/// The raw text of a scalar field in a table body, e.g. `trusted = true` → `true`.
fn table_field<'a>(table: &'a TomlTable, name: &str) -> Option<&'a str> {
    for line in &table.body {
        if let Some(eq) = line.find('=')
            && eq > 0
            && line[..eq].trim() == name
        {
            return Some(line[eq + 1..].trim());
        }
    }
    None
}

/// Apply `mutate`; true once the file holds the result. The no-op check is
/// against the RAW text, so a file parsing normalizes (e.g. duplicate-spelling
/// tables collapsed) is healed by the write. Racing launches may lose an
/// update (atomic rename, never torn); the next launch re-applies.
pub fn update_toml_tables(file: &Path, mutate: impl FnOnce(&mut TomlTables)) -> bool {
    let raw = read_toml_text(file);
    let mut parsed = parse_toml_tables(&raw);
    mutate(&mut parsed);
    let text = serialize_toml_tables(&parsed);
    text == raw || atomic_write_secure_blocking(file, text.as_bytes()).is_ok()
}

// ── model-limit pinning ───────────────────────────────────────────────────

/// grok mis-reads aivo's `/v1/models` limit fields and guesses, so
/// auto-compaction fires at the wrong point; a `[model."<id>"]` table is
/// grok's highest-priority limits source.
fn model_limits_table(
    model: &str,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
) -> Option<TomlTable> {
    let mut body = Vec::new();
    if let Some(ctx) = context_window {
        body.push(format!("context_window = {ctx}"));
    }
    if let Some(max) = max_output_tokens {
        body.push(format!("max_completion_tokens = {max}"));
    }
    if body.is_empty() || model.is_empty() {
        return None;
    }
    // TOML basic-string escape for the quoted model-id key.
    let id = model.replace('\\', "\\\\").replace('"', "\\\"");
    Some(TomlTable {
        header: format!("[model.\"{id}\"]"),
        body,
    })
}

/// config.toml also holds grok's own settings — replace only our table.
pub fn pin_model_limits(
    home: &Path,
    model: &str,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
) -> bool {
    let Some(table) = model_limits_table(model, context_window, max_output_tokens) else {
        return false;
    };
    update_toml_tables(&home.join("config.toml"), |tables| tables.set(table))
}

// ── one-time migration off the plugin's throwaway homes ───────────────────

/// Newest decision per folder wins, so a later untrust beats an earlier trust.
fn merge_trust_stores(home: &Path, files: &[PathBuf]) -> bool {
    let incoming: Vec<TomlTable> = files
        .iter()
        .flat_map(|f| read_toml_tables(f).tables)
        .collect();
    if incoming.is_empty() {
        return false;
    }
    let decided_at = |t: &TomlTable| -> f64 {
        table_field(t, "decided_at")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    };
    update_toml_tables(&trust_store(home), |tables| {
        for table in incoming {
            let newer = tables
                .get(&table.header)
                .is_none_or(|prev| decided_at(&table) >= decided_at(prev));
            if newer {
                tables.set(table);
            }
        }
    })
}

/// `<model>-<pid>-<ts>-<n>`: a live pid still owns that home — leave its
/// sessions alone. The `<model>` prefix is required; bare `<a>-<b>-<c>`
/// names never came from the plugin.
fn legacy_home_is_live(name: &str) -> bool {
    let mut tail = name.rsplit('-');
    let (Some(n), Some(ts), Some(pid), Some(_prefix)) =
        (tail.next(), tail.next(), tail.next(), tail.next())
    else {
        return false;
    };
    if n.parse::<u64>().is_err() || ts.parse::<u64>().is_err() {
        return false;
    }
    let Ok(pid) = pid.parse::<u32>() else {
        return false;
    };
    crate::services::system_env::is_pid_alive(pid)
}

fn read_dir_names(dir: &Path, dirs_only: bool) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.flatten()
        .filter(|e| !dirs_only || e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect()
}

fn migrate_sessions(legacy_home: &Path, home: &Path) -> usize {
    let mut moved = 0;
    let legacy_sessions = sessions_dir(legacy_home);
    for cwd in read_dir_names(&legacy_sessions, true) {
        let src = legacy_sessions.join(&cwd);
        let dest = sessions_dir(home).join(&cwd);
        for name in read_dir_names(&src, true) {
            if dest.join(&name).exists() {
                continue; // session ids are unique
            }
            if std::fs::create_dir_all(&dest).is_ok()
                && std::fs::rename(src.join(&name), dest.join(&name)).is_ok()
            {
                moved += 1;
            }
            // rename failure: cross-device or a live writer — session_roots
            // still finds it in place
        }
    }
    moved
}

/// Fold the plugin's pre-0.5.0 homes (and v0.4.0's trust store) into the
/// stable home. Returns sessions moved, or None when it already ran.
pub fn migrate_legacy_homes(config_dir: &Path) -> Option<usize> {
    let home = grok_home_dir(config_dir);
    if home.join(MIGRATION_MARKER).exists() {
        return None;
    }
    let base = legacy_homes_base(config_dir);
    let legacy = read_dir_names(&base, true);
    let shared_dir = config_dir.join("grok-shared"); // plugin v0.4.0's store
    let mut stores = vec![shared_dir.join(TRUST_FILE)];
    stores.extend(legacy.iter().map(|n| base.join(n).join(TRUST_FILE)));
    if merge_trust_stores(&home, &stores) {
        let _ = std::fs::remove_dir_all(&shared_dir); // nothing reads it now
    }
    let mut moved = 0;
    let mut deferred = 0;
    for name in &legacy {
        if legacy_home_is_live(name) {
            deferred += 1;
        } else {
            moved += migrate_sessions(&base.join(name), &home);
        }
    }
    // A deferred (live) home must be retried next launch.
    if deferred == 0 {
        let stamp = format!("{}\n", chrono::Utc::now().to_rfc3339());
        let _ = atomic_write_secure_blocking(&home.join(MIGRATION_MARKER), stamp.as_bytes());
    }
    Some(moved)
}

// ── GC ────────────────────────────────────────────────────────────────────

/// Best-effort sweep: legacy homes (kept a week so `aivo share` can still
/// read a recent run) and orphaned leader sockets. Never touches the managed
/// home's own content.
pub fn gc_stale_state(config_dir: &Path, now: SystemTime) {
    let stale = |p: &Path| -> bool {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| now.duration_since(mtime).ok())
            .map(|age| age > RETENTION)
            .unwrap_or(false)
    };
    let base = legacy_homes_base(config_dir);
    for name in read_dir_names(&base, true) {
        let dir = base.join(&name);
        if stale(&dir) {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
    // Once the last legacy home goes, later sweeps cost one ENOENT.
    let _ = std::fs::remove_dir(&base);
    let home = grok_home_dir(config_dir);
    for name in read_dir_names(&home, false) {
        if name.starts_with("leader-") && name.ends_with(".sock") {
            let sock = home.join(&name);
            if stale(&sock) {
                let _ = std::fs::remove_file(&sock);
            }
        }
    }
}

/// Pre-launch home preparation, all best-effort: create the home, fold in any
/// plugin-era legacy state, pin the active model's limits, sweep dead state.
pub fn prepare_managed_home(
    config_dir: &Path,
    model: Option<&str>,
    context_window: Option<u64>,
    max_output_tokens: Option<u64>,
) {
    let home = grok_home_dir(config_dir);
    if std::fs::create_dir_all(&home).is_err() {
        return;
    }
    migrate_legacy_homes(config_dir);
    if let Some(model) = model {
        pin_model_limits(&home, model, context_window, max_output_tokens);
    }
    gc_stale_state(config_dir, SystemTime::now());
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn parse_serialize_roundtrips_unknown_content() {
        let text =
            "top_key = 1\n\n[ui]\nscreen_mode = \"minimal\"\n\n[model.\"a\"]\ncontext_window = 5\n";
        let parsed = parse_toml_tables(text);
        assert_eq!(parsed.preamble, vec!["top_key = 1"]);
        assert_eq!(parsed.tables.len(), 2);
        assert_eq!(serialize_toml_tables(&parsed), text);
    }

    #[test]
    fn pin_model_limits_is_idempotent_and_preserves_foreign_tables() {
        let dir = TempDir::new().unwrap();
        let home = dir.path();
        write(
            &home.join("config.toml"),
            "[ui]\nscreen_mode = \"minimal\"\n",
        );
        assert!(pin_model_limits(home, "m1", Some(100), Some(10)));
        let first = std::fs::read_to_string(home.join("config.toml")).unwrap();
        assert!(first.contains("[ui]"));
        assert!(first.contains("[model.\"m1\"]"));
        assert!(first.contains("context_window = 100"));
        assert!(first.contains("max_completion_tokens = 10"));

        let mtime_before = std::fs::metadata(home.join("config.toml"))
            .unwrap()
            .modified()
            .unwrap();
        assert!(pin_model_limits(home, "m1", Some(100), Some(10)));
        let mtime_after = std::fs::metadata(home.join("config.toml"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(mtime_before, mtime_after, "unchanged pin must not rewrite");
    }

    #[test]
    fn pin_model_limits_matches_groks_unquoted_header_spelling() {
        // grok's own config rewriter drops quotes from bare-key-safe model
        // ids; the pin must replace that table, not append a duplicate that
        // grok's real TOML parser rejects ("duplicate key").
        let dir = TempDir::new().unwrap();
        let home = dir.path();
        write(
            &home.join("config.toml"),
            "[model.deepseek-v4-flash]\ncontext_window = 1\n\n[ui]\nyolo = false\n",
        );
        assert!(pin_model_limits(home, "deepseek-v4-flash", Some(2), None));
        let text = std::fs::read_to_string(home.join("config.toml")).unwrap();
        assert_eq!(
            text.matches("deepseek-v4-flash").count(),
            1,
            "one table, not a quoted duplicate: {text}"
        );
        assert!(text.contains("context_window = 2"));
        assert!(text.contains("[ui]"));
    }

    #[test]
    fn update_toml_tables_heals_a_duplicated_file() {
        // A file already broken by the pre-fix duplicate append: parsing
        // collapses the spellings (last body wins) and the no-op check
        // against the raw text forces the healing rewrite.
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("config.toml");
        write(
            &file,
            "[model.m1]\ncontext_window = 1\n\n[ui]\nyolo = false\n\n[model.\"m1\"]\ncontext_window = 2\n",
        );
        assert!(update_toml_tables(&file, |_| {}));
        let text = std::fs::read_to_string(&file).unwrap();
        assert_eq!(text.matches("m1").count(), 1, "{text}");
        assert!(
            text.contains("context_window = 2"),
            "last body wins: {text}"
        );
        assert!(text.contains("[ui]"));
    }

    #[test]
    fn header_key_parts_resolves_quoting() {
        assert_eq!(
            header_key_parts("[model.\"deepseek-v4-flash\"]"),
            header_key_parts("[model.deepseek-v4-flash]"),
        );
        assert_eq!(
            header_key_parts("[model.\"we\\\"ird\\\\id\"]"),
            vec!["model", "we\"ird\\id"],
        );
        // A quoted dot is part of the key, not a separator.
        assert_eq!(
            header_key_parts("[model.\"grok-4.5\"]"),
            vec!["model", "grok-4.5"],
        );
        assert_ne!(header_key_parts("[model.a]"), header_key_parts("[model.b]"),);
    }

    #[test]
    fn pin_model_limits_escapes_exotic_ids_and_skips_without_limits() {
        let dir = TempDir::new().unwrap();
        assert!(!pin_model_limits(dir.path(), "m", None, None));
        assert!(pin_model_limits(dir.path(), "we\"ird\\id", Some(1), None));
        let text = std::fs::read_to_string(dir.path().join("config.toml")).unwrap();
        assert!(text.contains("[model.\"we\\\"ird\\\\id\"]"));
    }

    #[test]
    fn merge_trust_stores_newest_decision_wins() {
        let dir = TempDir::new().unwrap();
        let home = dir.path().join("home");
        let old = dir.path().join("old.toml");
        let new = dir.path().join("new.toml");
        write(&old, "[folders.\"/a\"]\ntrusted = true\ndecided_at = 100\n");
        write(
            &new,
            "[folders.\"/a\"]\ntrusted = false\ndecided_at = 200\n",
        );
        assert!(merge_trust_stores(&home, &[old, new]));
        let text = std::fs::read_to_string(trust_store(&home)).unwrap();
        assert!(text.contains("trusted = false"));
        assert!(!text.contains("trusted = true"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(trust_store(&home))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode & 0o077, 0, "trust store must stay private");
        }
    }

    #[test]
    fn migrate_legacy_homes_moves_sessions_once_and_defers_live_pids() {
        let dir = TempDir::new().unwrap();
        let config = dir.path();
        let home = grok_home_dir(config);
        std::fs::create_dir_all(&home).unwrap();

        // Dead-pid legacy home migrates; own-pid home is deferred.
        let dead = legacy_homes_base(config).join("m-999999-1-1");
        write(
            &dead
                .join("sessions")
                .join("%2Ftmp%2Fx")
                .join("uuid-1")
                .join("chat_history.jsonl"),
            "{}\n",
        );
        let live_name = format!("m-{}-1-1", std::process::id());
        let live = legacy_homes_base(config).join(&live_name);
        write(
            &live
                .join("sessions")
                .join("%2Ftmp%2Fx")
                .join("uuid-2")
                .join("chat_history.jsonl"),
            "{}\n",
        );

        assert_eq!(migrate_legacy_homes(config), Some(1));
        assert!(
            sessions_dir(&home)
                .join("%2Ftmp%2Fx")
                .join("uuid-1")
                .exists()
        );
        assert!(
            !home.join(MIGRATION_MARKER).exists(),
            "live home defers the marker"
        );

        // With the live home gone, the retry completes and stamps the marker.
        std::fs::remove_dir_all(&live).unwrap();
        assert_eq!(migrate_legacy_homes(config), Some(0));
        assert!(home.join(MIGRATION_MARKER).exists());
        assert_eq!(
            migrate_legacy_homes(config),
            None,
            "marker makes it one-shot"
        );
    }

    #[test]
    fn gc_reaps_stale_legacy_homes_and_sockets_but_not_the_managed_home() {
        let dir = TempDir::new().unwrap();
        let config = dir.path();
        let home = grok_home_dir(config);
        write(&home.join("config.toml"), "x = 1\n");
        write(&home.join("leader-1-1-1.sock"), "");
        let legacy = legacy_homes_base(config).join("m-1-1-1");
        std::fs::create_dir_all(&legacy).unwrap();

        // "Stale" clock: pretend now is 8 days after the files were written.
        let future = SystemTime::now() + Duration::from_secs(8 * 24 * 60 * 60);
        gc_stale_state(config, future);
        assert!(!legacy.exists());
        assert!(!legacy_homes_base(config).exists());
        assert!(!home.join("leader-1-1-1.sock").exists());
        assert!(
            home.join("config.toml").exists(),
            "managed home content survives"
        );

        // Fresh files survive a sweep at the real time.
        write(&home.join("leader-2-2-2.sock"), "");
        gc_stale_state(config, SystemTime::now());
        assert!(home.join("leader-2-2-2.sock").exists());
    }

    #[test]
    fn leader_socket_is_unique_per_call() {
        let home = PathBuf::from("/h");
        assert_ne!(launch_leader_socket(&home), launch_leader_socket(&home));
    }

    #[test]
    fn encode_cwd_dir_matches_encode_uri_component() {
        assert_eq!(encode_cwd_dir("/private/tmp/x"), "%2Fprivate%2Ftmp%2Fx");
        assert_eq!(encode_cwd_dir("/a b"), "%2Fa%20b");
        // encodeURIComponent leaves these bare; RFC 3986 encoders don't.
        assert_eq!(encode_cwd_dir("/a(1)!'*"), "%2Fa(1)!'*");
        assert_eq!(encode_cwd_dir("/家"), "%2F%E5%AE%B6");
    }

    #[test]
    fn session_roots_covers_user_managed_and_legacy_homes() {
        let dir = TempDir::new().unwrap();
        let config = dir.path();
        let legacy = legacy_homes_base(config).join("m-1-1-1");
        std::fs::create_dir_all(&legacy).unwrap();
        let user_home = dir.path().join("pinned");
        let roots = session_roots(config, Some(&user_home));
        assert_eq!(
            roots,
            vec![
                sessions_dir(&user_home),
                sessions_dir(&grok_home_dir(config)),
                sessions_dir(&legacy),
            ]
        );
    }
}
