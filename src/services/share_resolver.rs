//! Find which source owns a given session id and produce its `SharePayload`.
//!
//! aivo can see six distinct conversation sources (aivo chat,
//! claude / codex / gemini / pi / opencode). The user passes a single session
//! id to `aivo logs share`; this resolver fans out across the sources to figure
//! out which one it belongs to, then runs the source-specific extractor.
//!
//! Cross-source collisions are surfaced as a hard error so a user with the
//! same id in two stores has to disambiguate. Per-source resolution accepts
//! unique id *prefixes* (matching how `aivo logs` truncates ids on display):
//! logs.db rows via `SessionStore::find_by_id_prefix`, file-backed CLIs via
//! `id_prefix_matches` over the session directory. Ambiguous prefixes are
//! reported back to the user with the candidates that matched.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde_json::Value;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, BufReader};

use chrono::{DateTime, Utc};

use crate::constants::KNOWN_TOOLS;
use crate::services::context_ingest::{
    self, IngestOptions, encode_claude_dir, gemini_matching_session_files, pi_session_dir,
};
use crate::services::log_store::LogEntry;
use crate::services::project_id::Thread;
use crate::services::session_store::SessionStore;
use crate::services::share_payload::{
    SharePayload, extract_chat_full, extract_claude_full, extract_codex_full, extract_gemini_full,
    extract_opencode_full, extract_pi_full,
};
use crate::services::system_env;

/// Inputs the resolver needs that aren't on the user's command line. All
/// filesystem roots are explicit so tests can inject temp paths instead of
/// having to mutate `$HOME`.
pub struct ResolverContext {
    pub project_root: PathBuf,
    pub session_store: SessionStore,
    pub chat_sessions_dir: PathBuf,
    pub claude_projects_root: PathBuf,
    pub codex_sessions_root: PathBuf,
    pub gemini_tmp_root: PathBuf,
    pub pi_sessions_root: PathBuf,
    pub opencode_db_path: PathBuf,
}

impl ResolverContext {
    /// Build a context that reads the standard system locations. `project_root`
    /// is the cwd to scope native CLI lookups against. `session_store` is the
    /// already-initialized SessionStore from the dispatcher.
    pub fn from_system(project_root: PathBuf, session_store: SessionStore) -> Self {
        let home = system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let aivo_config_dir = home.join(".config").join("aivo");
        Self {
            project_root,
            session_store,
            chat_sessions_dir: aivo_config_dir.join("sessions"),
            claude_projects_root: home.join(".claude").join("projects"),
            codex_sessions_root: home.join(".codex").join("sessions"),
            gemini_tmp_root: home.join(".gemini").join("tmp"),
            pi_sessions_root: home.join(".pi").join("agent").join("sessions"),
            opencode_db_path: home
                .join(".local")
                .join("share")
                .join("opencode")
                .join("opencode.db"),
        }
    }
}

/// Result of a successful resolve. The full payload is held in memory; the
/// caller redacts and serves it.
#[derive(Debug)]
pub struct ResolvedSession {
    pub payload: SharePayload,
}

#[derive(Debug, Clone)]
struct Match {
    source: &'static str,
    full_id: String,
}

/// Find the source for `session_id` (or any unique prefix) and produce its
/// full payload. Within-source and cross-source ambiguity both surface as
/// hard errors listing every match.
pub async fn resolve_session(session_id: &str, ctx: &ResolverContext) -> Result<ResolvedSession> {
    if session_id.is_empty() {
        return Err(anyhow!("session id is empty"));
    }

    // Probe logs.db first. Its ids are aivo-internal short alphanumerics
    // (12-char base32-ish) and don't overlap with UUID-style native ids,
    // so a hit here is decisive. The three logs.db sources
    // each map differently:
    //
    //   chat  → has a session_id linkage; share that chat session.
    //   run   → no transcript of its own (the transcript lives in the
    //           native session aivo launched). Tell the user to share
    //           that row's id instead.
    //   serve → a single HTTP request. Not a conversation. Refuse.
    let logs_hits = ctx
        .session_store
        .logs()
        .find_by_id_prefix(session_id, 5)
        .await?;
    if logs_hits.len() > 1 {
        let summary = logs_hits
            .iter()
            .map(|e| format!("{} [{}]", &e.id, e.source))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!(
            "ambiguous logs.db prefix '{}' — matched: {}. Re-run with a longer prefix.",
            session_id,
            summary
        ));
    }
    if let Some(entry) = logs_hits.into_iter().next() {
        match entry.source.as_str() {
            "chat" => {
                // Old chat rows (written before LogEvent.session_id existed)
                // have no stored linkage. Recover it the same way `aivo logs
                // show` displays it: closest chat session by cwd + key_id +
                // ts_utc.
                let chat_id = match entry.session_id.clone() {
                    Some(id) => id,
                    None => infer_chat_session_id(&entry, ctx).await?,
                };
                let state = ctx
                    .session_store
                    .get_chat_session(&chat_id)
                    .await?
                    .ok_or_else(|| {
                        anyhow!(
                            "chat session '{}' was deleted (logs.db still has its events but the session file is gone). Run `aivo logs prune` to remove orphan chat events.",
                            chat_id
                        )
                    })?;
                let payload = extract_chat_full(&state, ctx.project_root.to_str())?;
                return Ok(ResolvedSession { payload });
            }
            "run" => {
                // Resolve the run event to whichever native session aivo
                // launched: same tool, same cwd, mtime closest to ts_utc.
                // Heuristic but reliable in practice — `aivo run X` produces
                // exactly one native session, and we already know which X.
                let mut payload = resolve_run_event(&entry, ctx).await?;
                // Prefer logs.db's model so `share` matches `logs show
                // --json` (and surfaces the cursor-routed model, where the
                // session file only knows a generic mode name).
                if let Some(m) = entry.model.as_deref().filter(|s| !s.is_empty()) {
                    payload.model = Some(m.to_string());
                }
                return Ok(ResolvedSession { payload });
            }
            "serve" => {
                return Err(anyhow!(
                    "'{}' is a 'serve' event (one HTTP request) — not a shareable conversation.",
                    session_id
                ));
            }
            _ => { /* fall through to native fanout */ }
        }
    }

    // Each find_* returns 0+ matches. Prefix matching means the same input
    // can hit several files inside one source, so within-source collisions
    // count too — flatten everything before deciding.
    let project_root_str = ctx.project_root.to_string_lossy().to_string();
    let (chat_hits, claude_hits, codex_hits, gemini_hits, pi_hits, opencode_hits) = tokio::join!(
        find_chat(&ctx.session_store, &ctx.chat_sessions_dir, session_id),
        find_claude(&ctx.claude_projects_root, &ctx.project_root, session_id),
        find_codex(&ctx.codex_sessions_root, session_id),
        find_gemini(&ctx.gemini_tmp_root, &project_root_str, session_id),
        find_pi(&ctx.pi_sessions_root, &project_root_str, session_id),
        find_opencode(&ctx.opencode_db_path, session_id),
    );

    let mut hits: Vec<Match> = Vec::new();
    hits.extend(chat_hits.unwrap_or_default());
    hits.extend(claude_hits);
    hits.extend(codex_hits);
    hits.extend(gemini_hits);
    hits.extend(pi_hits);
    hits.extend(opencode_hits);

    if hits.len() > 1 {
        let summary = hits
            .iter()
            .map(|h| format!("{} ({})", h.source, h.full_id))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!(
            "ambiguous session id '{}' — matched: {}. Re-run with a longer prefix.",
            session_id,
            summary
        ));
    }

    let Some(hit) = hits.into_iter().next() else {
        return Err(anyhow!(
            "Session id '{}' not found.\nRun `aivo logs --all` to see available sessions across all projects.",
            session_id
        ));
    };

    // For file-backed CLI sources, `Match::full_id` carries the resolved
    // file path. For chat / opencode it carries the actual session id
    // (loaded via the source's own lookup).
    //
    // Per-source cwd: derive from the source itself, not from the caller's
    // cwd, so cross-project resolves (which the global listing makes easy)
    // show the *session's* project rather than the user's terminal cwd.
    let p_root = ctx.project_root.to_str();
    let payload = match hit.source {
        "chat" => {
            let state = ctx
                .session_store
                .get_chat_session(&hit.full_id)
                .await?
                .ok_or_else(|| anyhow!("chat session '{}' vanished during resolve", hit.full_id))?;
            extract_chat_full(&state, p_root)?
        }
        "claude" => {
            let path = PathBuf::from(&hit.full_id);
            let cwd = decode_native_parent_cwd(&path);
            extract_claude_full(&path, cwd.as_deref()).await?
        }
        "codex" => {
            // extract_codex_full reads session_meta.cwd from the file itself
            // when project_root is None. Pass None here so cross-project
            // resolves don't get rejected by the cwd-match guard.
            extract_codex_full(&PathBuf::from(&hit.full_id), None).await?
        }
        "gemini" => extract_gemini_full(&PathBuf::from(&hit.full_id), p_root).await?,
        "pi" => {
            let path = PathBuf::from(&hit.full_id);
            let cwd = decode_pi_parent_cwd(&path);
            extract_pi_full(&path, cwd.as_deref()).await?
        }
        "opencode" => extract_opencode_full(&ctx.opencode_db_path, &hit.full_id, p_root).await?,
        other => return Err(anyhow!("internal error: unknown source '{other}'")),
    };

    Ok(ResolvedSession { payload })
}

/// Reverse Claude Code's encoded-dir convention. `~/.claude/projects/-Users-alice-foo/<id>.jsonl`
/// → `/Users/alice/foo`. Lossy when the original cwd contained literal hyphens;
/// acceptable for display.
fn decode_native_parent_cwd(path: &Path) -> Option<String> {
    let parent = path.parent()?.file_name()?.to_str()?;
    Some(parent.replace('-', "/"))
}

/// Reverse Pi's `--<dashes>--` per-cwd dir name.
/// `~/.pi/agent/sessions/--Users-alice-foo--/<id>.jsonl` → `/Users/alice/foo`.
fn decode_pi_parent_cwd(path: &Path) -> Option<String> {
    let parent = path.parent()?.file_name()?.to_str()?;
    let inner = parent.strip_prefix("--")?.strip_suffix("--")?;
    Some(format!("/{}", inner.replace('-', "/")))
}

// ---------------------------------------------------------------------------
// Per-source probes
// ---------------------------------------------------------------------------
//
// Each `find_*` returns `Some(Match)` when `session_id` resolves to that
// source. For file-backed sources, `Match::full_id` carries the resolved
// file path (the resolver upgrades it back to a `PathBuf` when extracting),
// because that lets each find function decide whether the id lives in a
// filename or inside a file. For `chat` and `opencode`, `full_id` carries
// the actual session id (the loader looks it up by id, not by path).
//
// This conflation of "id vs path" inside `Match::full_id` is intentional —
// it keeps the `Match` type tiny and avoids bifurcating the resolver loop.

/// Scan a directory for files whose stem matches `prefix` (after both sides
/// have their dashes stripped, so `aivo logs`'s display ids — which strip
/// dashes from UUIDs to fit a fixed column width — copy-paste back in cleanly).
/// Returns full paths, one per match.
async fn find_files_by_stem_prefix(dir: &Path, prefix: &str, extension: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(mut rd) = fs::read_dir(dir).await else {
        return out;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some(extension) {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if id_prefix_matches(stem, prefix) {
            out.push(path);
        }
    }
    out
}

/// True iff `candidate` (a full session id) starts with `user_input` (a
/// possibly-truncated, possibly-undashed prefix). Both sides have dashes
/// stripped before comparison so a UUID like `1335c631-9147-4e2d-…` matches
/// both `1335c631` and the dashes-removed display form `1335c6319147`.
fn id_prefix_matches(candidate: &str, user_input: &str) -> bool {
    if user_input.is_empty() {
        return false;
    }
    if candidate.starts_with(user_input) {
        return true;
    }
    let cand: String = candidate.chars().filter(|c| *c != '-').collect();
    let inp: String = user_input.chars().filter(|c| *c != '-').collect();
    !inp.is_empty() && cand.starts_with(&inp)
}

async fn find_chat(store: &SessionStore, chat_dir: &Path, session_id: &str) -> Result<Vec<Match>> {
    // Two paths: an exact id is cheap to probe via `get_chat_session`; a
    // prefix needs a directory scan because the SessionStore index is
    // keyed by full session id.
    if let Some(_state) = store.get_chat_session(session_id).await? {
        return Ok(vec![Match {
            source: "chat",
            full_id: session_id.to_string(),
        }]);
    }
    let matches: Vec<Match> = find_files_by_stem_prefix(chat_dir, session_id, "json")
        .await
        .into_iter()
        .filter_map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str())?;
            // `index.json` lives in the same dir; skip it.
            if stem == "index" {
                return None;
            }
            Some(Match {
                source: "chat",
                full_id: stem.to_string(),
            })
        })
        .collect();
    Ok(matches)
}

async fn find_claude(claude_root: &Path, project_root: &Path, session_id: &str) -> Vec<Match> {
    // Try the current project's encoded dir first — it's the cheap path and
    // the common case.
    let canonical = std::fs::canonicalize(project_root)
        .unwrap_or_else(|_| project_root.to_path_buf())
        .to_string_lossy()
        .to_string();
    let project_dir = claude_root.join(encode_claude_dir(&canonical));
    let mut hits: Vec<Match> = find_files_by_stem_prefix(&project_dir, session_id, "jsonl")
        .await
        .into_iter()
        .map(|p| Match {
            source: "claude",
            full_id: p.to_string_lossy().to_string(),
        })
        .collect();
    if !hits.is_empty() {
        return hits;
    }

    // Fall back to a global scan — `aivo logs` lists native sessions across
    // every project, so the resolver has to be willing to follow the user
    // out of cwd to keep "see-it / show-it" parity.
    let Ok(mut rd) = fs::read_dir(claude_root).await else {
        return hits;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        for path in find_files_by_stem_prefix(&dir, session_id, "jsonl").await {
            hits.push(Match {
                source: "claude",
                full_id: path.to_string_lossy().to_string(),
            });
        }
    }
    hits
}

async fn find_codex(codex_root: &Path, session_id: &str) -> Vec<Match> {
    if !codex_root.exists() {
        return Vec::new();
    }
    // Codex stores rollouts as `rollout-<ts>-<uuid>.jsonl` under a YYYY/MM/DD
    // tree. We walk every file whose name contains the prefix (cheap enough
    // — codex only writes a handful per day) and verify session_meta inside.
    let mut out: Vec<Match> = Vec::new();
    let mut dirs = vec![codex_root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(mut rd) = fs::read_dir(&dir).await else {
            continue;
        };
        while let Ok(Some(entry)) = rd.next_entry().await {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            // The codex filename embeds the session id with dashes intact
            // (rollout-…-019e5d69-fea7-…). Strip dashes from both sides
            // before substring-matching so a display id pasted without
            // dashes (`019e5d69fe`) still gates in the right files.
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let name_no_dashes: String = name.chars().filter(|c| *c != '-').collect();
            let input_no_dashes: String = session_id.chars().filter(|c| *c != '-').collect();
            if input_no_dashes.is_empty() || !name_no_dashes.contains(&input_no_dashes) {
                continue;
            }
            if codex_session_id_matches_prefix(&path, session_id).await {
                out.push(Match {
                    source: "codex",
                    full_id: path.to_string_lossy().to_string(),
                });
            }
        }
    }
    out
}

async fn codex_session_id_matches_prefix(path: &Path, session_id_prefix: &str) -> bool {
    // The user gave an explicit id; cwd-scoping it would re-introduce the
    // "see-it-can't-show-it" mismatch since `aivo logs` lists codex sessions
    // globally. The id itself is unique enough.
    let Ok(file) = fs::File::open(path).await else {
        return false;
    };
    let mut lines = BufReader::new(file).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("type").and_then(|t| t.as_str()) == Some("session_meta")
            && let Some(payload) = v.get("payload")
        {
            return payload
                .get("id")
                .and_then(|s| s.as_str())
                .is_some_and(|id| id_prefix_matches(id, session_id_prefix));
        }
    }
    false
}

async fn find_gemini(_gemini_root: &Path, project_root: &str, session_id: &str) -> Vec<Match> {
    let mut out: Vec<Match> = Vec::new();
    let candidates = gemini_matching_session_files(project_root).await;
    for path in candidates {
        if let Ok(content) = fs::read_to_string(&path).await
            && let Ok(v) = serde_json::from_str::<Value>(&content)
            && v.get("sessionId")
                .and_then(|s| s.as_str())
                .is_some_and(|id| id_prefix_matches(id, session_id))
        {
            out.push(Match {
                source: "gemini",
                full_id: path.to_string_lossy().to_string(),
            });
        }
    }
    out
}

async fn find_pi(pi_root: &Path, project_root: &str, session_id: &str) -> Vec<Match> {
    // Cheap path: probe the cwd-encoded dir for the current project.
    if let Some(session_dir) = pi_session_dir(project_root) {
        let local: Vec<Match> = find_files_by_stem_prefix(&session_dir, session_id, "jsonl")
            .await
            .into_iter()
            .map(|p| Match {
                source: "pi",
                full_id: p.to_string_lossy().to_string(),
            })
            .collect();
        if !local.is_empty() {
            return local;
        }
    }

    // Global fallback — same rationale as find_claude.
    let mut hits: Vec<Match> = Vec::new();
    let Ok(mut rd) = fs::read_dir(pi_root).await else {
        return hits;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        for path in find_files_by_stem_prefix(&dir, session_id, "jsonl").await {
            hits.push(Match {
                source: "pi",
                full_id: path.to_string_lossy().to_string(),
            });
        }
    }
    hits
}

async fn find_opencode(db_path: &Path, session_id: &str) -> Vec<Match> {
    if fs::metadata(db_path).await.is_err() {
        return Vec::new();
    }
    let db_path = db_path.to_path_buf();
    let probe_prefix = session_id.to_string();
    let matches: Vec<String> = tokio::task::spawn_blocking(move || -> Vec<String> {
        let conn = match rusqlite::Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        ) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        // Match either the dashed input or its dashes-stripped form against
        // both the raw id and the dashes-stripped id, so the listing's
        // compact display id round-trips back to the right session.
        let stmt = conn.prepare(
            "SELECT id FROM session
              WHERE id LIKE ?1
                 OR id LIKE ?2
                 OR replace(id, '-', '') LIKE ?2
              ORDER BY id LIMIT 5",
        );
        let mut stmt = match stmt {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let dashed_pattern = format!("{probe_prefix}%");
        let probe_no_dashes: String = probe_prefix.chars().filter(|c| *c != '-').collect();
        let nodash_pattern = format!("{probe_no_dashes}%");
        let rows = stmt.query_map([&dashed_pattern, &nodash_pattern], |row| {
            row.get::<_, String>(0)
        });
        match rows {
            Ok(it) => it.flatten().collect(),
            Err(_) => Vec::new(),
        }
    })
    .await
    .unwrap_or_default();
    matches
        .into_iter()
        .map(|id| Match {
            source: "opencode",
            full_id: id,
        })
        .collect()
}

/// Resolve a logs.db `run` event to the native session aivo launched for it.
/// Strategy: take the event's tool + cwd + ts_utc; among native sessions of
/// that tool in that cwd, pick the one whose `updated_at` is closest to
/// the run's start timestamp (within a generous window).
async fn resolve_run_event(entry: &LogEntry, ctx: &ResolverContext) -> Result<SharePayload> {
    let tool = entry
        .tool
        .as_deref()
        .ok_or_else(|| anyhow!("run event has no tool field"))?;
    // Plugin tools (not a native AIToolType) run out-of-process via their own
    // binary; aivo records the launch in `aivo logs` but never sees the
    // transcript, so there's nothing to share. Fail with a clear reason rather
    // than the misleading "session deleted / lives elsewhere" native message.
    if !KNOWN_TOOLS.contains(&tool) {
        return Err(anyhow!(
            "'{tool}' is a plugin tool — aivo logs its launches but doesn't store a transcript, so this run can't be shared"
        ));
    }
    let run_cwd: String = entry
        .cwd
        .clone()
        .unwrap_or_else(|| ctx.project_root.to_string_lossy().to_string());
    let run_ts = parse_log_timestamp(&entry.ts_utc);

    // Native CLIs: project-scoped enumeration, then closest-mtime match within
    // the matching cli.
    //
    // Every native CLI uses a dedicated enumerator that skips the
    // substantive-content filter `ingest_project`'s extractors apply (turns
    // must be ≥ MIN_TURN_CHARS, no `<environment_context>`/`<turn_aborted>`
    // markers). That filter is right for AI-context injection but wrong here
    // — short prompts like `claude -p 'say hi'`, CJK queries like
    // `今天成都的天气` (7 chars), or a session that's still mid-conversation
    // would otherwise be silently un-shareable, and the resolver would fall
    // back to whichever older session in the same cwd happens to have
    // substantive turns. An unknown tool name falls back to `ingest_project`
    // (which won't return anything for unknown tools, yielding a clean "no
    // session found" error).
    let run_path = std::path::Path::new(&run_cwd);
    let threads: Vec<Thread> = match tool {
        "claude" => context_ingest::list_claude_sessions_for_cwd(run_path).await,
        // codex-app launches the `codex` binary, so its rollouts land in the
        // same ~/.codex/sessions tree (via the shadow CODEX_HOME symlink) and
        // are indistinguishable from plain codex sessions.
        "codex" | "codex-app" => {
            context_ingest::list_codex_sessions_for_cwd(&ctx.codex_sessions_root, run_path).await
        }
        "pi" => context_ingest::list_pi_sessions_for_cwd(&ctx.pi_sessions_root, run_path).await,
        "gemini" => {
            context_ingest::list_gemini_sessions_for_cwd(&ctx.gemini_tmp_root, run_path).await
        }
        "opencode" => {
            context_ingest::list_opencode_sessions_for_cwd(&ctx.opencode_db_path, run_path).await
        }
        _ => {
            let opts = IngestOptions {
                max_age_days: None,
                min_updated_at: None,
                max_per_source: Some(50),
                // Run-event fallback resolves short sessions too — the
                // resolver looks up the native session that aivo launched,
                // and that session might have started with a short prompt.
                include_short_first_user: true,
            };
            context_ingest::ingest_project(run_path, opts)
                .await?
                .into_iter()
                .filter(|t| t.cli == tool)
                .collect()
        }
    };
    let mut candidates: Vec<&Thread> = threads.iter().collect();
    if candidates.is_empty() {
        return Err(anyhow!(
            "no native {tool} session found in {run_cwd} — the run may have been deleted, or its session file lives elsewhere"
        ));
    }
    candidates.sort_by_key(|t| {
        run_ts
            .map(|rt| (t.updated_at - rt).num_seconds().abs())
            .unwrap_or(0)
    });
    let closest = candidates[0];
    extract_thread_full(closest, ctx).await
}

async fn infer_chat_session_id(entry: &LogEntry, ctx: &ResolverContext) -> Result<String> {
    let cwd = entry.cwd.as_deref().ok_or_else(|| {
        anyhow!(
            "chat event '{}' has no session_id linkage and no cwd to infer one",
            entry.id
        )
    })?;
    let ts = parse_log_timestamp(&entry.ts_utc).ok_or_else(|| {
        anyhow!(
            "chat event '{}' has no session_id linkage and an unparseable ts_utc",
            entry.id
        )
    })?;
    ctx.session_store
        .find_chat_session_near(cwd, entry.key_id.as_deref(), ts, 60)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "chat event '{}' has no session_id linkage, and no chat session in {} matched within 60s of the event",
                entry.id,
                cwd
            )
        })
}

/// Re-run the per-cli extractor on a `Thread` (which only carries summary
/// data) to produce a full `SharePayload`. Mirrors the dispatch in
/// `resolve_session` but driven from a Thread rather than a Match.
async fn extract_thread_full(t: &Thread, ctx: &ResolverContext) -> Result<SharePayload> {
    let cwd = t.cwd.as_deref();
    match t.cli.as_str() {
        "claude" => extract_claude_full(Path::new(&t.source_path), cwd).await,
        "codex" => extract_codex_full(Path::new(&t.source_path), None).await,
        "gemini" => extract_gemini_full(Path::new(&t.source_path), cwd).await,
        "pi" => extract_pi_full(Path::new(&t.source_path), cwd).await,
        "opencode" => extract_opencode_full(&ctx.opencode_db_path, &t.session_id, cwd).await,
        other => Err(anyhow!("unexpected cli '{other}' for run-event resolve")),
    }
}

fn parse_log_timestamp(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::session_crypto::encrypt;
    use crate::services::session_store::{SessionTokens, StoredChatMessage};
    use tempfile::TempDir;

    fn ctx_with_tempdirs(temp: &TempDir, project_root: PathBuf) -> ResolverContext {
        let store = SessionStore::with_path(temp.path().join("config.json"));
        ResolverContext {
            project_root,
            session_store: store,
            chat_sessions_dir: temp.path().join("sessions"),
            claude_projects_root: temp.path().join("claude_projects"),
            codex_sessions_root: temp.path().join("codex"),
            gemini_tmp_root: temp.path().join("gemini"),
            pi_sessions_root: temp.path().join("pi"),
            opencode_db_path: temp.path().join("opencode.db"),
        }
    }

    #[tokio::test]
    async fn resolve_returns_not_found_for_unknown_id() {
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());
        let err = resolve_session("definitely-nope-xyz", &ctx)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn resolve_chat_session_through_session_store() {
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());

        // Persist one chat session via the SessionStore so get_chat_session finds it.
        let messages = vec![StoredChatMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }];
        let encrypted = encrypt(&serde_json::to_string(&messages).unwrap()).unwrap();
        let _ = encrypted; // sanity: ensures crypto module is importable

        // SessionStore exposes save_chat_session_with_id for the same purpose.
        ctx.session_store
            .save_chat_session_with_id(
                "kid",
                "https://api.example.com",
                "/tmp",
                "chat-xyz",
                "gpt-4o",
                None,
                &messages,
                "title",
                "preview",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        let resolved = resolve_session("chat-xyz", &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "chat");
        assert_eq!(resolved.payload.session_id, "chat-xyz");
    }

    #[tokio::test]
    async fn resolve_chat_event_without_session_id_infers_via_cwd_and_ts() {
        // Old `chat` rows have no session_id linkage. The resolver should
        // back-fill it from the chat session in the same cwd whose
        // updated_at is closest to the event's ts_utc.
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());

        let messages = vec![StoredChatMessage {
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }];
        ctx.session_store
            .save_chat_session_with_id(
                "kid",
                "https://api.example.com",
                "/tmp/proj",
                "chat-orphan",
                "gpt-4o",
                None,
                &messages,
                "title",
                "preview",
                SessionTokens::default(),
            )
            .await
            .unwrap();

        // Insert a chat log row that references the same cwd/key but leaves
        // session_id blank — the shape produced by aivo before the linkage
        // column existed. Resolver should still find chat-orphan.
        let event_id = ctx
            .session_store
            .logs()
            .append(crate::services::log_store::LogEvent {
                source: "chat".into(),
                kind: "chat_turn".into(),
                key_id: Some("kid".into()),
                key_name: Some("test".into()),
                base_url: Some("https://api.example.com".into()),
                tool: Some("chat".into()),
                model: Some("gpt-4o".into()),
                cwd: Some("/tmp/proj".into()),
                session_id: None,
                ..Default::default()
            })
            .await
            .unwrap();

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "chat");
        assert_eq!(resolved.payload.session_id, "chat-orphan");
    }

    #[tokio::test]
    async fn resolve_chat_event_without_session_id_errors_when_no_match() {
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());

        let event_id = ctx
            .session_store
            .logs()
            .append(crate::services::log_store::LogEvent {
                source: "chat".into(),
                kind: "chat_turn".into(),
                cwd: Some("/tmp/no-sessions-here".into()),
                session_id: None,
                ..Default::default()
            })
            .await
            .unwrap();

        let err = resolve_session(&event_id, &ctx).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no chat session in /tmp/no-sessions-here"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn resolve_rejects_empty_id() {
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());
        let err = resolve_session("", &ctx).await.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn resolve_codex_via_dashes_stripped_uuid_prefix() {
        // Regression: codex filenames embed the session UUID with dashes
        // (`…-019e5d69-fea7-…`). When a user copies a dashes-stripped display
        // id from `aivo logs` (`019e5d69fe`), the substring sits across a
        // dash in the filename and the file gate must still admit it.
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().to_path_buf();
        let ctx = ctx_with_tempdirs(&temp, project_root.clone());
        let day_dir = ctx.codex_sessions_root.join("2026").join("05").join("25");
        fs::create_dir_all(&day_dir).await.unwrap();

        let full_id = "019e5d69-fea7-72b2-a794-7da3a44485a6";
        let path =
            day_dir.join("rollout-2026-05-25T12-34-48-019e5d69-fea7-72b2-a794-7da3a44485a6.jsonl");
        let proj_json = project_root.to_string_lossy().replace('\\', "\\\\");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-05-25T12:34:48Z","payload":{{"id":"{}","cwd":"{}","timestamp":"2026-05-25T12:34:48Z"}}}}"#,
                full_id, proj_json
            ),
            r#"{"type":"response_item","timestamp":"2026-05-25T12:35:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let resolved = resolve_session("019e5d69fe", &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "codex");
        assert_eq!(resolved.payload.session_id, full_id);
    }

    #[tokio::test]
    async fn resolve_codex_app_run_event_finds_codex_rollout() {
        // Regression: `aivo run codex-app` logs the run event with tool
        // "codex-app", but it launches the `codex` binary, so its rollout
        // lands in ~/.codex/sessions like any codex session. The run-event
        // resolver must route "codex-app" to the codex enumerator instead of
        // erroring with "no native codex-app session found".
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().to_path_buf();
        let ctx = ctx_with_tempdirs(&temp, project_root.clone());
        let day_dir = ctx.codex_sessions_root.join("2026").join("05").join("28");
        fs::create_dir_all(&day_dir).await.unwrap();

        let full_id = "019e71a0-1111-72b2-a794-7da3a44485a6";
        let path =
            day_dir.join("rollout-2026-05-28T10-00-00-019e71a0-1111-72b2-a794-7da3a44485a6.jsonl");
        let proj_json = project_root.to_string_lossy().replace('\\', "\\\\");
        let lines = [
            format!(
                r#"{{"type":"session_meta","timestamp":"2026-05-28T10:00:00Z","payload":{{"id":"{}","cwd":"{}","timestamp":"2026-05-28T10:00:00Z"}}}}"#,
                full_id, proj_json
            ),
            r#"{"type":"response_item","timestamp":"2026-05-28T10:00:05Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}"#.to_string(),
        ];
        fs::write(&path, lines.join("\n")).await.unwrap();

        let event_id = ctx
            .session_store
            .logs()
            .append(crate::services::log_store::LogEvent {
                source: "run".into(),
                kind: "tool_launch".into(),
                tool: Some("codex-app".into()),
                cwd: Some(project_root.to_string_lossy().to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "codex");
        assert_eq!(resolved.payload.session_id, full_id);
    }
}
