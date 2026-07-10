//! Find which source owns a given session id and produce its `SharePayload`.
//!
//! aivo can see six distinct conversation sources (aivo code,
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

use std::collections::HashMap;
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

/// A plugin tool's transcript source: a built-in `format` aivo reads from `dir`,
/// or `format == "native"` (the plugin emits its own via `bin
/// --aivo-export-transcript`). Lets `aivo share` resolve a plugin run.
pub struct PluginTranscript {
    pub format: String,
    pub dir: PathBuf,
    /// The `aivo-<name>` binary for `native` export; `None` if not located.
    pub bin: Option<PathBuf>,
}

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
    /// Plugin tools that declared a transcript source (tool name → source).
    /// Populated by the caller, which has plugin-registry access.
    pub plugin_transcripts: HashMap<String, PluginTranscript>,
}

impl ResolverContext {
    /// Build a context that reads the standard system locations. `project_root`
    /// is the cwd to scope native CLI lookups against. `session_store` is the
    /// already-initialized SessionStore from the dispatcher.
    pub fn from_system(project_root: PathBuf, session_store: SessionStore) -> Self {
        let home = system_env::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let aivo_config_dir = crate::services::paths::config_dir();
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
            plugin_transcripts: HashMap::new(),
        }
    }

    /// Attach the plugin transcript sources resolved by the caller (which has
    /// registry access — keeps this module free of a `plugin` dependency).
    pub fn with_plugin_transcripts(mut self, map: HashMap<String, PluginTranscript>) -> Self {
        self.plugin_transcripts = map;
        self
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
            // `code` is the built-in agent post-rename; `chat` is the pre-rename
            // source still on disk in existing users' logs.db.
            "chat" | "code" => {
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
                    .get_code_session(&chat_id)
                    .await?
                    .ok_or_else(|| {
                        anyhow!(
                            "code session '{}' was deleted (logs.db still has its events but the session file is gone). Run `aivo logs prune` to remove orphan code events.",
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
        "code" => {
            let state = ctx
                .session_store
                .get_code_session(&hit.full_id)
                .await?
                .ok_or_else(|| anyhow!("code session '{}' vanished during resolve", hit.full_id))?;
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
    // Two paths: an exact id is cheap to probe via `get_code_session`; a
    // prefix needs a directory scan because the SessionStore index is
    // keyed by full session id.
    if let Some(_state) = store.get_code_session(session_id).await? {
        return Ok(vec![Match {
            source: "code",
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
                source: "code",
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

/// Session formats `aivo share` can read, so a plugin declaring one of these is
/// shareable. The `match` in `resolve_run_event` dispatches each to its reader.
const SHAREABLE_PLUGIN_FORMATS: &[&str] = &["pi", "codex", "gemini", "opencode"];

/// Sentinel `format` for a plugin that emits its own transcript (via
/// `--aivo-export-transcript`) instead of using a built-in reader.
const NATIVE_TRANSCRIPT_FORMAT: &str = "native";

/// The sessions root for a run: the plugin's declared dir, or the native default.
fn reader_root<'a>(plugin_src: Option<&'a PluginTranscript>, default: &'a Path) -> &'a Path {
    plugin_src.map_or(default, |s| s.dir.as_path())
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
    // Plugin tools (not a native AIToolType) run out-of-process and store their
    // own transcripts; aivo only sees the launch. A plugin can still be shared
    // if its manifest declares a transcript source in a format aivo reads — then
    // we reuse that built-in reader pointed at the plugin's sessions dir.
    // Otherwise there's nothing to share: fail with a clear reason.
    let is_native = KNOWN_TOOLS.contains(&tool);
    let plugin_src = if is_native {
        None
    } else {
        ctx.plugin_transcripts.get(tool)
    };
    let run_cwd: String = entry
        .cwd
        .clone()
        .unwrap_or_else(|| ctx.project_root.to_string_lossy().to_string());
    let run_ts = parse_log_timestamp(&entry.ts_utc);
    if !is_native {
        match plugin_src {
            None => {
                return Err(anyhow!(
                    "'{tool}' is a plugin tool — aivo logs its launches but doesn't store a transcript, so this run can't be shared"
                ));
            }
            // `native`: the plugin emits its own transcript for this run.
            Some(src) if src.format == NATIVE_TRANSCRIPT_FORMAT => {
                return export_native_plugin_transcript(
                    src,
                    tool,
                    &run_cwd,
                    &entry.ts_utc,
                    ctx.session_store.config_dir(),
                )
                .await;
            }
            Some(src) if !SHAREABLE_PLUGIN_FORMATS.contains(&src.format.as_str()) => {
                return Err(anyhow!(
                    "'{tool}' declares transcript format '{}', which aivo can't read — shareable plugin formats: {}",
                    src.format,
                    SHAREABLE_PLUGIN_FORMATS.join(", ")
                ));
            }
            _ => {}
        }
    }

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
    // A plugin run reads its declared format from its own sessions dir; a native
    // run uses the tool name + the ctx roots. The format determines the reader,
    // which stamps `cli` on the Thread, so the downstream extractor is correct
    // regardless of the plugin's name.
    let fmt = plugin_src.map(|s| s.format.as_str()).unwrap_or(tool);
    let threads: Vec<Thread> = match fmt {
        "claude" => context_ingest::list_claude_sessions_for_cwd(run_path).await,
        // codex-app launches the `codex` binary, so its rollouts land in the
        // same ~/.codex/sessions tree (via the shadow CODEX_HOME symlink) and
        // are indistinguishable from plain codex sessions.
        "codex" | "codex-app" => {
            let root = reader_root(plugin_src, &ctx.codex_sessions_root);
            context_ingest::list_codex_sessions_for_cwd(root, run_path).await
        }
        "pi" => {
            let root = reader_root(plugin_src, &ctx.pi_sessions_root);
            context_ingest::list_pi_sessions_for_cwd(root, run_path).await
        }
        "gemini" => {
            let root = reader_root(plugin_src, &ctx.gemini_tmp_root);
            context_ingest::list_gemini_sessions_for_cwd(root, run_path).await
        }
        "opencode" => {
            let root = reader_root(plugin_src, &ctx.opencode_db_path);
            context_ingest::list_opencode_sessions_for_cwd(root, run_path).await
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
    extract_thread_full(closest).await
}

/// Run `aivo-<name> --aivo-export-transcript --cwd <cwd> --ts <ts_utc>` and parse
/// the one `SharePayload` JSON it prints. Best-effort with a hard timeout —
/// missing binary, non-zero exit, timeout, or bad output all become a share error.
async fn export_native_plugin_transcript(
    src: &PluginTranscript,
    tool: &str,
    run_cwd: &str,
    run_ts_utc: &str,
    config_dir: &Path,
) -> Result<SharePayload> {
    let bin = src.bin.as_deref().ok_or_else(|| {
        anyhow!("'{tool}' declares a native transcript but aivo couldn't locate its binary")
    })?;
    let mut cmd = tokio::process::Command::new(bin);
    cmd.arg("--aivo-export-transcript")
        .arg("--cwd")
        .arg(run_cwd)
        .arg("--ts")
        .arg(run_ts_utc)
        // Plugin reads its own store; same config dir dispatch hands children.
        .env("AIVO_CONFIG_DIR", config_dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);
    let child = cmd
        .spawn()
        .map_err(|e| anyhow!("launching '{tool}' transcript export failed: {e}"))?;
    let out = tokio::time::timeout(std::time::Duration::from_secs(10), child.wait_with_output())
        .await
        .map_err(|_| anyhow!("'{tool}' transcript export timed out"))?
        .map_err(|e| anyhow!("'{tool}' transcript export failed: {e}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "'{tool}' transcript export exited with {}",
            out.status
        ));
    }
    serde_json::from_slice::<SharePayload>(&out.stdout)
        .map_err(|e| anyhow!("'{tool}' produced an invalid transcript: {e}"))
}

async fn infer_chat_session_id(entry: &LogEntry, ctx: &ResolverContext) -> Result<String> {
    let cwd = entry.cwd.as_deref().ok_or_else(|| {
        anyhow!(
            "code event '{}' has no session_id linkage and no cwd to infer one",
            entry.id
        )
    })?;
    let ts = parse_log_timestamp(&entry.ts_utc).ok_or_else(|| {
        anyhow!(
            "code event '{}' has no session_id linkage and an unparseable ts_utc",
            entry.id
        )
    })?;
    ctx.session_store
        .find_chat_session_near(cwd, entry.key_id.as_deref(), ts, 60)
        .await?
        .ok_or_else(|| {
            anyhow!(
                "code event '{}' has no session_id linkage, and no code session in {} matched within 60s of the event",
                entry.id,
                cwd
            )
        })
}

/// Re-run the per-cli extractor on a `Thread` (which only carries summary
/// data) to produce a full `SharePayload`. Mirrors the dispatch in
/// `resolve_session` but driven from a Thread rather than a Match.
async fn extract_thread_full(t: &Thread) -> Result<SharePayload> {
    let cwd = t.cwd.as_deref();
    match t.cli.as_str() {
        "claude" => extract_claude_full(Path::new(&t.source_path), cwd).await,
        "codex" => extract_codex_full(Path::new(&t.source_path), None).await,
        "gemini" => extract_gemini_full(Path::new(&t.source_path), cwd).await,
        "pi" => extract_pi_full(Path::new(&t.source_path), cwd).await,
        "opencode" => extract_opencode_full(Path::new(&t.source_path), &t.session_id, cwd).await,
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
            plugin_transcripts: HashMap::new(),
        }
    }

    /// Append a `run` tool_launch event and return its id.
    async fn append_run_event(ctx: &ResolverContext, tool: &str, cwd: &str) -> String {
        ctx.session_store
            .logs()
            .append(crate::services::log_store::LogEvent {
                source: "run".into(),
                kind: "tool_launch".into(),
                tool: Some(tool.into()),
                cwd: Some(cwd.into()),
                ..Default::default()
            })
            .await
            .unwrap()
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

        // Persist one chat session via the SessionStore so get_code_session finds it.
        let messages = vec![StoredChatMessage {
            model: None,
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }];
        // SessionStore exposes save_code_session_with_id for the same purpose.
        ctx.session_store
            .save_code_session_with_id(
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
        assert_eq!(resolved.payload.source_cli, "code");
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
            model: None,
            role: "user".into(),
            content: "hi".into(),
            reasoning_content: None,
            id: None,
            timestamp: None,
            attachments: None,
        }];
        ctx.session_store
            .save_code_session_with_id(
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
        assert_eq!(resolved.payload.source_cli, "code");
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
            msg.contains("no code session in /tmp/no-sessions-here"),
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

        let event_id = append_run_event(&ctx, "codex-app", &project_root.to_string_lossy()).await;

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "codex");
        assert_eq!(resolved.payload.session_id, full_id);
    }

    // Pi's encoded-cwd dir name (`--<path>--`) is Unix-path-shaped; on Windows a
    // canonicalized `\\?\C:\…` path yields a directory name with `:`/backslashes
    // that can't be created. Gated like the pi tests in `context_ingest`.
    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_plugin_run_event_via_declared_pi_transcript() {
        // A plugin tool (`omp`) that declares a pi-format transcript source is
        // shareable: the run event routes to the pi reader pointed at the
        // plugin's own sessions dir, and extracts via the pi pipeline.
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let canonical = fs::canonicalize(&project_root).await.unwrap();
        let canonical_str = canonical.to_string_lossy().to_string();

        // omp's sessions live under a pi-format tree at a custom root.
        let omp_root = temp.path().join("omp-sessions");
        let encoded = format!("--{}--", canonical_str.trim_matches('/').replace('/', "-"));
        let session_dir = omp_root.join(encoded);
        fs::create_dir_all(&session_dir).await.unwrap();
        let full_id = "019e9b4a-5627-7000-ae0c-4854c916a807";
        // omp uses pi's record schema: message `content` is an array of parts.
        let lines = [
            format!(
                r#"{{"type":"session","id":"{full_id}","cwd":"{canonical_str}","timestamp":"2026-06-06T04:56:40.000Z","version":"1"}}"#
            ),
            r#"{"type":"message","timestamp":"2026-06-06T04:56:41.000Z","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#.to_string(),
            r#"{"type":"message","timestamp":"2026-06-06T04:56:42.000Z","message":{"role":"assistant","content":[{"type":"text","text":"hello"}]}}"#.to_string(),
        ];
        fs::write(
            session_dir.join(format!("2026-06-06T04-56-40-743Z_{full_id}.jsonl")),
            lines.join("\n"),
        )
        .await
        .unwrap();

        let mut ctx = ctx_with_tempdirs(&temp, canonical.clone());
        ctx.plugin_transcripts.insert(
            "omp".to_string(),
            PluginTranscript {
                format: "pi".to_string(),
                dir: omp_root,
                bin: None,
            },
        );

        let event_id = append_run_event(&ctx, "omp", &canonical_str).await;

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "pi");
        assert_eq!(resolved.payload.session_id, full_id);
        assert_eq!(resolved.payload.messages.len(), 2);
    }

    #[tokio::test]
    async fn resolve_plugin_run_event_via_declared_opencode_transcript() {
        // OpenCode plugin transcripts are a single DB file. The run-event
        // resolver must extract from the same plugin DB it used for enumeration,
        // not from the native OpenCode DB in the resolver context.
        let temp = TempDir::new().unwrap();
        let project_root = temp.path().join("proj");
        fs::create_dir_all(&project_root).await.unwrap();
        let canonical = fs::canonicalize(&project_root).await.unwrap();
        let canonical_str = canonical.to_string_lossy().to_string();

        let plugin_db = temp.path().join("omp-opencode.db");
        let full_id = "oc-plugin-1";
        let conn = rusqlite::Connection::open(&plugin_db).unwrap();
        conn.execute_batch(
            "CREATE TABLE project (id TEXT PRIMARY KEY, worktree TEXT NOT NULL);
             CREATE TABLE session (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, time_updated INTEGER NOT NULL);
             CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, data TEXT NOT NULL, time_created INTEGER NOT NULL);
             CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, data TEXT NOT NULL, time_created INTEGER NOT NULL);",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO project (id, worktree) VALUES ('p1', ?1)",
            [&canonical_str],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO session (id, project_id, time_updated) VALUES (?1, 'p1', 1778211465000)",
            [full_id],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message (id, session_id, data, time_created) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "m1",
                full_id,
                serde_json::json!({"role": "user"}).to_string(),
                1000_i64
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO message (id, session_id, data, time_created) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "m2",
                full_id,
                serde_json::json!({"role": "assistant"}).to_string(),
                2000_i64
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO part (id, message_id, data, time_created) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "p1",
                "m1",
                serde_json::json!({"type": "text", "text": "plugin question"}).to_string(),
                1001_i64
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO part (id, message_id, data, time_created) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                "p2",
                "m2",
                serde_json::json!({"type": "text", "text": "plugin answer"}).to_string(),
                2001_i64
            ],
        )
        .unwrap();
        drop(conn);

        let mut ctx = ctx_with_tempdirs(&temp, canonical.clone());
        ctx.plugin_transcripts.insert(
            "omp".to_string(),
            PluginTranscript {
                format: "opencode".to_string(),
                dir: plugin_db,
                bin: None,
            },
        );

        let event_id = append_run_event(&ctx, "omp", &canonical_str).await;

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "opencode");
        assert_eq!(resolved.payload.session_id, full_id);
        assert_eq!(resolved.payload.messages.len(), 2);
    }

    #[tokio::test]
    async fn resolve_plugin_run_event_without_transcript_source_is_unshareable() {
        let temp = TempDir::new().unwrap();
        let ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());
        let event_id = append_run_event(&ctx, "mystery", &temp.path().to_string_lossy()).await;

        let err = resolve_session(&event_id, &ctx).await.unwrap_err();
        assert!(err.to_string().contains("doesn't store a transcript"));
    }

    /// A `native`-format plugin run is resolved by invoking the plugin's
    /// `--aivo-export-transcript` subcommand and parsing the `SharePayload` it
    /// prints — so the share carries the plugin's own `source_cli`, not a
    /// borrowed format label. Uses a fake plugin binary (a shell script).
    #[cfg(unix)]
    #[tokio::test]
    async fn resolve_native_plugin_run_event_invokes_export_subcommand() {
        use std::os::unix::fs::PermissionsExt;
        let temp = TempDir::new().unwrap();
        let bin = temp.path().join("aivo-myplugin");
        let payload_json = r#"{"schema_version":"1","source_cli":"myplugin","session_id":"T-native-1","project":{},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]},{"role":"assistant","content":[{"type":"text","text":"hello"}]}],"meta":{"aivo_version":"test","redacted":false,"live":false,"served_at":"2026-06-09T00:00:00Z"}}"#;
        fs::write(
            &bin,
            format!("#!/bin/sh\ncat <<'JSON'\n{payload_json}\nJSON\n"),
        )
        .await
        .unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();

        let mut ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());
        ctx.plugin_transcripts.insert(
            "myplugin".to_string(),
            PluginTranscript {
                format: "native".to_string(),
                dir: PathBuf::new(),
                bin: Some(bin),
            },
        );
        let event_id = append_run_event(&ctx, "myplugin", &temp.path().to_string_lossy()).await;

        let resolved = resolve_session(&event_id, &ctx).await.unwrap();
        assert_eq!(resolved.payload.source_cli, "myplugin");
        assert_eq!(resolved.payload.session_id, "T-native-1");
        assert_eq!(resolved.payload.messages.len(), 2);
    }

    /// A `native` plugin whose binary can't be located fails with a clear error
    /// rather than silently falling through to the native-CLI enumeration.
    #[tokio::test]
    async fn resolve_native_plugin_without_binary_errors() {
        let temp = TempDir::new().unwrap();
        let mut ctx = ctx_with_tempdirs(&temp, temp.path().to_path_buf());
        ctx.plugin_transcripts.insert(
            "myplugin".to_string(),
            PluginTranscript {
                format: "native".to_string(),
                dir: PathBuf::new(),
                bin: None,
            },
        );
        let event_id = append_run_event(&ctx, "myplugin", &temp.path().to_string_lossy()).await;
        let err = resolve_session(&event_id, &ctx).await.unwrap_err();
        assert!(err.to_string().contains("couldn't locate its binary"));
    }
}
