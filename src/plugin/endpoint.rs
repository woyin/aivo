//! Capability-gated handoff for granted plugins. When a plugin declares the
//! `endpoint` capability and the user has consented, aivo serves the active key
//! through a per-launch loopback proxy and hands over only `AIVO_ENDPOINT_URL` +
//! `AIVO_ENDPOINT_TOKEN` — the secret never leaves aivo. The proxy is backed by
//! `ServeRouter`, the responses-capable internal router for OpenAI-protocol keys
//! and Copilot, or the Cursor ACP router for `type: coding-agent` plugins. OAuth
//! credentials are native-agent-only, so they run bare. For `type: coding-agent`
//! plugins the launch is also wrapped in the same stats/logs accounting built-in
//! tools get. Frozen contract: `docs/PLUGIN-PROTOCOL.md`.

use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::key_resolution::{KeyLookupMode, KeyResolution, resolve_key_override};
use crate::services::key_compat::KeyCompatContext;
use crate::services::log_store::{LogEvent, LogStore, new_log_id};
use crate::services::request_log::RequestLogger;
use crate::services::serve_router::{ServeRouter, ServeRouterConfig, random_auth_token};
use crate::services::session_store::{ApiKey, SessionStore};
use crate::style;

use super::manifest::{PluginManifest, grantable_capabilities, probe_manifest};
use super::registry::{self, PluginRecord};

/// The only plugin `type` with host behavior today: stats/logs wrapping.
const CODING_AGENT_TYPE: &str = "coding-agent";

/// Dispatch a plugin that may need a key/endpoint handoff or run accounting.
/// Falls back to a plain spawn (today's behavior) when neither applies.
pub(crate) async fn dispatch(name: &str, bin: &Path, args: &[String], store: &SessionStore) -> i32 {
    let config_dir = store.config_dir().to_path_buf();

    let plan = grant_plan(name, bin).await;
    if plan.caps.is_empty() && !plan.is_coding_agent {
        // Nothing granted and no host behavior — plain dispatch.
        return super::exec_plugin(bin, args, &config_dir, &[]).await;
    }

    // `--help`/`-h` is a pure passthrough: never resolve a key, open a picker,
    // stand up an endpoint, or fail on auth just to print help. For top-level
    // help, prepend a uniform banner documenting the flags aivo intercepts, so
    // thin wrappers surface them the same way a fat plugin's own help does.
    match help_kind(args) {
        HelpKind::None => {}
        kind => {
            // Skip the banner when the plugin self-documents aivo's flags (a fat
            // plugin's own `--help` lists them better) — else it's pure duplication.
            if matches!(kind, HelpKind::TopLevel) && !plan.documents_aivo_flags {
                print_aivo_help_banner(name, &plan);
            }
            return super::exec_plugin(bin, args, &config_dir, &[]).await;
        }
    }

    // Coding-agent plugins behave like native tools: aivo owns `-k`/`-m` and
    // `--debug`, strips them from the argv (so they don't reach the wrapped
    // tool), opens the picker on a bare `-k`, and resolves the model. Other
    // granted plugins keep verbatim argv — `-k <id>` still selects a key, but
    // nothing is stripped and a bare `-k` uses the active key (no surprise picker
    // before launch).
    let flags = super::extract_aivo_flags(args);
    let debug_log = if plan.is_coding_agent {
        flags.debug_log.clone()
    } else {
        super::debug_log_path(args)
    };
    let (key_flag, model_flag, plugin_args): (Option<String>, Option<String>, Vec<String>) =
        if plan.is_coding_agent {
            (flags.key, flags.model, flags.rest)
        } else {
            (flags.key.filter(|s| !s.is_empty()), None, args.to_vec())
        };

    let key_was_explicit = key_flag.is_some();
    let key = resolve_plugin_key(store, key_flag.as_deref(), plan.is_coding_agent).await;
    let serve = key.as_ref().map(plugin_serve);
    let servable = serve.is_some_and(PluginServe::is_servable);

    let model_flag_was_explicit = model_flag.is_some();
    let model_request = plugin_model_request(model_flag, key_was_explicit);

    // A coding-agent plugin needs a concrete model (unlike native tools, which
    // fall back to their own default), so resolve one: `-m <v>` is used +
    // remembered, a bare launch reuses the remembered model, `-k` without `-m`
    // forces the model picker after key selection, otherwise the picker opens
    // once and the choice is remembered. OAuth is Blocked; Cursor is servable
    // only through the coding-agent Cursor router.
    let model = match key.as_ref() {
        Some(k) if plan.is_coding_agent && servable => {
            resolve_plugin_model(store, k, model_request, model_flag_was_explicit).await
        }
        _ => model_request,
    };

    // A coding-agent plugin launched with an explicit `-k` records that key as
    // the last selection, exactly like native `aivo run -k`. A *thin* plugin
    // drives the endpoint/handoff from `key` directly, but a *fat* plugin (amp)
    // re-resolves the key from the store itself — without this it never sees the
    // `-k` we just stripped from its argv. Gated on an explicit `-k` so a bare
    // launch doesn't clobber the user's last selection.
    if plan.is_coding_agent
        && key_was_explicit
        && let Some(k) = key.as_ref()
    {
        let _ = store.set_last_selection(k, name, model.as_deref()).await;
    }

    // The endpoint is the only handoff: aivo stands up a loopback proxy bound to
    // the key and hands over one OpenAI-compatible URL + bearer — the secret
    // never leaves aivo. `ServeRouter` serves Anthropic/Gemini REST keys and
    // Ollama; OpenAI-protocol REST keys and Copilot use the responses-capable
    // router; Cursor uses the ACP-backed cursor router for coding-agent plugins.
    // OAuth credentials are native-agent-only (no proxy aivo hands to a plugin),
    // so they run bare.
    let mut extra_env: Vec<(String, String)> = Vec::new();
    if let Some(path) = &debug_log {
        extra_env.push((
            "AIVO_DEBUG_LOG".to_string(),
            path.to_string_lossy().to_string(),
        ));
    }
    let mut endpoint: Option<EndpointHandle> = None;
    if let (Some(key), Some(serve)) = (key.as_ref(), serve) {
        match serve {
            // OAuth lives only in its own native agent — nothing to hand over;
            // the plugin runs on its own auth. Note it once so an
            // endpoint-granted plugin isn't silently mis-keyed.
            PluginServe::Blocked if plan.has("endpoint") => {
                eprintln!(
                    "  {} key `{}` is a {} credential — usable only in `{}`; `{}` runs on its own auth",
                    style::yellow("!"),
                    key.display_name(),
                    key.oauth_kind_label(),
                    key.oauth_tool_hint(),
                    name,
                );
            }
            PluginServe::Blocked => {}
            PluginServe::Cursor if plan.has("endpoint") => {
                if plan.is_coding_agent {
                    if let Some(m) = &model {
                        extra_env.push(("AIVO_KEY_MODEL".to_string(), m.clone()));
                    }
                    maybe_init_http_debug(debug_log.as_deref()).await;
                    apply_endpoint(
                        start_cursor_endpoint(key).await,
                        &mut extra_env,
                        &mut endpoint,
                    );
                } else {
                    eprintln!(
                        "  {} key `{}` is a Cursor credential — usable only by `type: coding-agent` plugins; `{}` runs on its own auth",
                        style::yellow("!"),
                        key.display_name(),
                        name,
                    );
                }
            }
            PluginServe::Cursor => {}
            // The resolved model travels with the handoff (the plugin sends it to
            // the endpoint). OpenAI-protocol keys and Copilot use the
            // responses-capable internal proxy (so gpt-5.x reasoning+tools reach
            // `/v1/responses`, even for a Chat Completions client); others use the
            // serve proxy, which additionally logs each request to the plugin's
            // `--debug` file.
            PluginServe::Serve if plan.has("endpoint") => {
                if let Some(m) = &model {
                    extra_env.push(("AIVO_KEY_MODEL".to_string(), m.clone()));
                }
                let started = if use_responses_router(key) {
                    maybe_init_http_debug(debug_log.as_deref()).await;
                    start_responses_endpoint(key, store, name).await
                } else {
                    let started = start_loopback_endpoint(
                        key,
                        store.logs(),
                        store.clone(),
                        name,
                        debug_log.clone(),
                    )
                    .await;
                    if started.is_ok()
                        && let Some(p) = &debug_log
                    {
                        eprintln!(
                            "  {} {}",
                            style::dim("·"),
                            style::dim(format!("endpoint traffic logged to {}", p.display())),
                        );
                    }
                    started
                };
                apply_endpoint(started, &mut extra_env, &mut endpoint);
            }
            // Servable, but the plugin didn't request `endpoint` — nothing to do.
            _ => {}
        }
    }

    let accounting = if plan.is_coding_agent {
        Some(begin_accounting(store, name, key.as_ref(), model.as_deref(), &plugin_args).await)
    } else {
        None
    };

    let code = super::exec_plugin(bin, &plugin_args, &config_dir, &extra_env).await;

    if let Some(acct) = accounting {
        finish_accounting(store, acct, code).await;
    }
    if let Some(ep) = endpoint {
        ep.shutdown().await;
    }
    code
}

async fn maybe_init_http_debug(path: Option<&Path>) {
    let Some(path) = path else {
        return;
    };
    match crate::services::http_debug::init(path.to_path_buf()).await {
        Ok(p) => eprintln!("[aivo] HTTP debug log → {}", p.display()),
        Err(e) => {
            eprintln!("[aivo] failed to open debug log: {e}; HTTP requests will not be logged")
        }
    }
}

// ── capability grant resolution ─────────────────────────────────────────────

struct GrantPlan {
    /// Grantable caps the user approved (a subset of what was requested).
    caps: Vec<String>,
    is_coding_agent: bool,
    /// The plugin self-documents aivo's flags → skip the help banner.
    documents_aivo_flags: bool,
}

impl GrantPlan {
    fn has(&self, cap: &str) -> bool {
        self.caps.iter().any(|c| c == cap)
    }
}

pub(crate) fn is_coding_agent(m: &PluginManifest) -> bool {
    m.kind.as_deref() == Some(CODING_AGENT_TYPE)
}

/// Resolve what's granted to `name`: read the cached manifest + grants, or
/// (for a managed but manifest-less plugin) lazily probe and seek consent once.
async fn grant_plan(name: &str, bin: &Path) -> GrantPlan {
    let reg = registry::load();
    match reg.plugins.get(name) {
        // Already probed at install — honor the recorded grants.
        Some(rec) if rec.manifest.is_some() => {
            let manifest = rec.manifest.as_ref().unwrap();
            let requested = grantable_capabilities(&manifest.capabilities);
            GrantPlan {
                caps: retained_grants(&requested, &rec.granted_caps),
                is_coding_agent: is_coding_agent(manifest),
                documents_aivo_flags: manifest.documents_aivo_flags,
            }
        }
        // Manifest-less managed plugin (e.g. a `npm:`/`github:` install): lazily
        // probe + seek consent once, persisting the grant for next time.
        Some(rec) => lazy_probe_and_consent(name, bin, rec).await,
        // Unmanaged/PATH plugin — nothing to grant; runs plain.
        None => GrantPlan {
            caps: Vec::new(),
            is_coding_agent: false,
            documents_aivo_flags: false,
        },
    }
}

/// First-dispatch probe + consent for a managed plugin recorded without a
/// manifest (e.g. installed via `npm:`/`github:`, where install-time probing is
/// skipped). On a successful probe the manifest + approved caps are persisted so
/// neither the probe nor the prompt recurs.
async fn lazy_probe_and_consent(name: &str, bin: &Path, existing: &PluginRecord) -> GrantPlan {
    let Some(manifest) = probe_manifest(bin, name).await else {
        return GrantPlan {
            caps: Vec::new(),
            is_coding_agent: false,
            documents_aivo_flags: false,
        };
    };
    let is_ca = is_coding_agent(&manifest);
    let documents_aivo_flags = manifest.documents_aivo_flags;
    let requested = grantable_capabilities(&manifest.capabilities);
    let mut granted = retained_grants(&requested, &existing.granted_caps);
    let missing = missing_grants(&requested, &existing.granted_caps);
    if !missing.is_empty()
        && std::io::stdin().is_terminal()
        && super::prompt_capability_grant(name, &missing)
    {
        granted.extend(missing);
    }
    let mut rec = existing.clone();
    rec.manifest = Some(manifest);
    rec.granted_caps = granted.clone();
    registry::record(name, rec);
    GrantPlan {
        caps: granted,
        is_coding_agent: is_ca,
        documents_aivo_flags,
    }
}

fn retained_grants(requested: &[String], prior: &[String]) -> Vec<String> {
    requested
        .iter()
        .filter(|c| prior.contains(c))
        .cloned()
        .collect()
}

fn missing_grants(requested: &[String], prior: &[String]) -> Vec<String> {
    requested
        .iter()
        .filter(|c| !prior.contains(c))
        .cloned()
        .collect()
}

/// How a plugin invocation requests help. `Sub` (`aivo amp trust --help`) is a
/// passthrough only; `TopLevel` (`aivo amp --help`) also gets the aivo banner.
enum HelpKind {
    None,
    Sub,
    TopLevel,
}

fn help_kind(args: &[String]) -> HelpKind {
    if !args.iter().any(|a| a == "-h" || a == "--help") {
        HelpKind::None
    } else if matches!(args.first().map(String::as_str), Some("-h" | "--help")) {
        HelpKind::TopLevel
    } else {
        HelpKind::Sub
    }
}

/// One uniform block above a plugin's own `--help` documenting the flags aivo
/// intercepts, so every plugin (fat or thin) surfaces them the same way. Goes to
/// stdout (it's help content) and is flushed before the child writes its own.
fn print_aivo_help_banner(name: &str, plan: &GrantPlan) {
    use std::io::Write;
    println!(
        "{} {}",
        style::cyan(format!("aivo {name}")),
        style::dim("— flags handled by aivo:"),
    );
    let row = |flag: &str, desc: &str| {
        println!(
            "  {}  {}",
            style::bold(format!("{flag:<22}")),
            style::dim(desc)
        );
    };
    if plan.is_coding_agent {
        row(
            "-k, --key [<id|name>]",
            "aivo key to use (bare -k opens the picker)",
        );
        row(
            "-m, --model [<model>]",
            "model to use (bare -m opens the picker)",
        );
        row("--debug[=<path>]", "log the proxied endpoint traffic");
    } else {
        row("-k, --key <id|name>", "aivo key to serve over the endpoint");
        row("--debug[=<path>]", "log the proxied endpoint traffic");
    }
    println!();
    println!("{}", style::dim(format!("{name}'s own help:")));
    println!("{}", style::dim("─".repeat(40)));
    let _ = std::io::stdout().flush();
}

fn plugin_model_request(model_flag: Option<String>, key_was_explicit: bool) -> Option<String> {
    match (model_flag, key_was_explicit) {
        (None, true) => Some(String::new()),
        (other, _) => other,
    }
}

/// Resolve the handoff key from the extracted `-k`/`--key` value: `Some("")`
/// opens the picker, `Some(id)` selects that key, `None` uses the active/last-used
/// key silently (`PreferActiveAllowNone` — no prompt before launch).
async fn resolve_plugin_key(
    store: &SessionStore,
    key_flag: Option<&str>,
    allow_cursor: bool,
) -> Option<ApiKey> {
    match resolve_key_override(
        store,
        key_flag,
        KeyLookupMode::PreferActiveAllowNone,
        // The picker dims CLI-bound OAuth credentials. Cursor is selectable only
        // for coding-agent plugins, which can use the Cursor ACP endpoint.
        KeyCompatContext::Plugin { allow_cursor },
    )
    .await
    {
        Ok(KeyResolution::Selected(k)) => Some(k),
        // The user opened the picker (`-k`) and cancelled → abort the run, like
        // `aivo <tool>` does, rather than silently launching on another key.
        Ok(KeyResolution::Cancelled) => std::process::exit(0),
        // No usable key / error → run bare; the plugin falls back to its own auth.
        _ => None,
    }
}

/// Which loopback proxy (if any) aivo uses to serve a key to a plugin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PluginServe {
    /// Any REST key, starter, Copilot, Ollama — served over a loopback proxy
    /// (`ServeRouter` or the responses-capable router; see `use_responses_router`).
    Serve,
    /// Cursor ACP — served through `CursorModelRouter` for coding-agent plugins.
    Cursor,
    /// OAuth (Claude/Codex/Gemini) — native-agent-only; aivo has no
    /// plugin-servable proxy for them, so they run bare.
    Blocked,
}

impl PluginServe {
    fn is_servable(self) -> bool {
        !matches!(self, PluginServe::Blocked)
    }
}

/// Classify how aivo can serve `key` to a plugin. OAuth is bound to native-agent
/// auth; Cursor goes through the ACP-backed cursor router; everything else aivo
/// proxies over a loopback endpoint.
fn plugin_serve(key: &ApiKey) -> PluginServe {
    if key.is_cursor_acp() {
        PluginServe::Cursor
    } else if key.is_any_oauth() {
        PluginServe::Blocked
    } else {
        PluginServe::Serve
    }
}

/// Resolve the model for a coding-agent plugin. Unlike native tools (which fall
/// back to their own default), such a plugin needs a concrete model, so this
/// yields one whenever possible: explicit `-m <value>` is used + remembered; a
/// bare launch reuses the key's remembered model; otherwise the picker opens —
/// delegating to the shared `resolve_model_outcome` — and the pick is remembered.
/// `None` only when nothing resolves (bare launch, nothing saved, no picker) — the
/// plugin then asks the user to pass `-m`.
async fn resolve_plugin_model(
    store: &SessionStore,
    key: &ApiKey,
    model_flag: Option<String>,
    explicit_model_flag: bool,
) -> Option<String> {
    use crate::commands::models::{ModelOutcome, resolve_model_outcome};

    // Explicit `-m <value>`: use it and remember it for next time.
    if let Some(m) = model_flag.as_deref().filter(|s| !s.is_empty()) {
        let _ = store.set_chat_model(&key.id, m).await;
        return Some(m.to_string());
    }
    // Bare launch: reuse the remembered model rather than re-prompting.
    if model_flag.is_none()
        && let Some(saved) = store.get_chat_model(&key.id).await.ok().flatten()
    {
        return Some(saved);
    }
    // Bare `-m`, or a bare launch with nothing saved: the plugin needs a model, so
    // force the picker (synthesize a bare `-m`) via the shared resolver.
    let client = crate::services::http_utils::router_http_client();
    let cache = crate::services::ModelsCache::new();
    match resolve_model_outcome(
        &client,
        key,
        Some(String::new()),
        explicit_model_flag,
        false,
        None,
        &cache,
        "Select model",
    )
    .await
    {
        Ok(ModelOutcome::Model(m)) => {
            let _ = store.set_chat_model(&key.id, &m).await;
            Some(m)
        }
        // Picker cancelled — don't launch (parity with `aivo run`).
        Ok(ModelOutcome::Cancelled) => std::process::exit(0),
        // No TTY / empty catalog: leave it to the plugin (it asks the user for -m).
        _ => None,
    }
}

// ── loopback control endpoint (reuses serve) ────────────────────────────────

/// A running per-launch loopback proxy bound to the plugin's key.
pub(crate) struct EndpointHandle {
    pub url: String,
    pub token: String,
    handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    /// `Some` for routers that honor a graceful-shutdown notify (`ServeRouter`);
    /// `None` for the responses router, which has no notify and is
    /// torn down via the abort path in [`EndpointHandle::shutdown`].
    shutdown: Option<Arc<tokio::sync::Notify>>,
    /// When set, protocol routes the router learned during the run are merged
    /// back into the key's config on shutdown — the same `(tool,key,model)`
    /// persistence native launches get, so the cascade isn't re-probed next time.
    persist: Option<RoutePersist>,
}

/// Captures what `persist_runtime_discoveries` needs to write a router's learned
/// routes back to the key after the plugin exits.
struct RoutePersist {
    store: SessionStore,
    key: ApiKey,
    route_cache: Arc<crate::services::route_cache::RouteCache>,
    learned: Arc<std::sync::atomic::AtomicBool>,
}

impl EndpointHandle {
    /// Assemble a handle from a bound `port`. Centralizes the load-bearing `/v1`
    /// base-URL contract: clients (omp, OpenAI SDKs) append `/chat/completions`
    /// etc., so the path must already carry `/v1`.
    fn new(
        port: u16,
        token: String,
        handle: tokio::task::JoinHandle<anyhow::Result<()>>,
        shutdown: Option<Arc<tokio::sync::Notify>>,
    ) -> Self {
        EndpointHandle {
            url: format!("http://127.0.0.1:{port}/v1"),
            token,
            handle,
            shutdown,
            persist: None,
        }
    }

    /// Persist the router's learned routes back to the key on shutdown.
    fn with_persist(mut self, persist: RoutePersist) -> Self {
        self.persist = Some(persist);
        self
    }

    /// Stop the proxy and drain its task (bounded), so the ephemeral port is
    /// released even when the plugin exits abruptly. First persists any learned
    /// routes (after the run, before teardown). A router with a notify
    /// (`ServeRouter`) exits gracefully within the timeout; one without
    /// (`ResponsesToChatRouter`) is aborted on timeout.
    async fn shutdown(self) {
        if let Some(p) = self.persist {
            crate::services::launch_runtime::persist_runtime_discoveries(
                &p.store,
                &p.key,
                Some(p.route_cache),
                Some(p.learned),
            )
            .await;
        }
        if let Some(notify) = &self.shutdown {
            notify.notify_one();
        }
        let abort = self.handle.abort_handle();
        if tokio::time::timeout(Duration::from_secs(3), self.handle)
            .await
            .is_err()
        {
            abort.abort();
        }
    }
}

/// Record a started endpoint's env (or warn on failure) — the shared tail of the
/// serve/responses dispatch arms.
fn apply_endpoint(
    started: anyhow::Result<EndpointHandle>,
    extra_env: &mut Vec<(String, String)>,
    endpoint: &mut Option<EndpointHandle>,
) {
    match started {
        Ok(ep) => {
            extra_env.push(("AIVO_ENDPOINT_URL".to_string(), ep.url.clone()));
            extra_env.push(("AIVO_ENDPOINT_TOKEN".to_string(), ep.token.clone()));
            *endpoint = Some(ep);
        }
        Err(e) => eprintln!(
            "  {} plugin endpoint unavailable: {e:#}",
            style::yellow("!")
        ),
    }
}

/// Start a `serve` proxy bound to `key` on an OS-assigned loopback port, with an
/// ephemeral bearer token. Token usage of buffered 2xx responses is recorded
/// against the key, labeled with `tool`. `debug_log`, when set, logs each
/// proxied request to that file.
async fn start_loopback_endpoint(
    key: &ApiKey,
    log_store: LogStore,
    usage: SessionStore,
    tool: &str,
    debug_log: Option<PathBuf>,
) -> anyhow::Result<EndpointHandle> {
    let token = random_auth_token();
    let config = ServeRouterConfig::from_key(key, false, 300, Some(token.clone()), HashMap::new());
    let mut router = ServeRouter::new(config, key.clone(), log_store)
        .with_usage_accounting(usage, tool.to_string());
    if let Some(path) = &debug_log
        && let Some(logger) = RequestLogger::new_with_path(path).await
    {
        router = router.with_logger(Some(logger));
    }
    let (handle, shutdown, port) = router.start_background_with_addr("127.0.0.1", 0).await?;
    Ok(EndpointHandle::new(port, token, handle, Some(shutdown)))
}

/// Start the Cursor ACP compatibility router for a coding-agent plugin. The
/// router speaks the same local OpenAI/Anthropic/Gemini/Responses paths native
/// Cursor-backed tools use, but is bearer-gated for the plugin endpoint.
async fn start_cursor_endpoint(key: &ApiKey) -> anyhow::Result<EndpointHandle> {
    use crate::services::cursor_acp;
    use crate::services::cursor_bridge::{CursorModelRouter, CursorRouterConfig};

    cursor_acp::ensure_cursor_agent_installed()?;
    if cursor_acp::is_legacy_cursor_login_secret(key.key.as_str()) {
        anyhow::bail!(
            "This cursor key predates per-account isolation. Run `aivo keys rm {0}` then `aivo keys add cursor` to recreate it as an isolated account.",
            key.id
        );
    }

    // Check OAuth-login auth before handing a dead endpoint to the plugin.
    if cursor_acp::cursor_oauth_shadow_signed_out(key).await {
        anyhow::bail!(
            "Cursor is not logged in for this key. Run `aivo keys reauth {0}` to sign in again.",
            key.id
        );
    }

    let token = random_auth_token();
    let workspace_cwd =
        crate::services::system_env::current_dir_string().unwrap_or_else(|| ".".to_string());
    let router = CursorModelRouter::new(CursorRouterConfig {
        key: key.clone(),
        workspace_cwd,
        models_cache: Some(crate::services::models_cache::ModelsCache::new()),
        // Do not prewarm for plugin endpoints: merely launching a plugin should
        // not open ACP generation sessions before the endpoint is actually used.
        prewarm_count: 0,
        mcp_prewarm_id_style: None,
        expected_token: Some(token.clone()),
    });
    let (port, handle) = router.start_background().await?;
    Ok(EndpointHandle::new(port, token, handle, None))
}

/// The `(tool,key,model)` route-cache namespace for plugin endpoints. Routes are
/// a property of the upstream (key,model), not the plugin, so all plugins share
/// one namespace — distinct from native tools' ("codex"/"opencode"/…).
const PLUGIN_ROUTE_TOOL: &str = "plugin";

/// True when a key should be served through the responses-capable internal proxy
/// (`ResponsesToChatRouter`) rather than `ServeRouter`: OpenAI-protocol REST keys
/// and Copilot, which may need `/v1/responses` (gpt-5.x reasoning + tools). The
/// responses router exchanges Copilot's token itself. Anthropic/Gemini keys keep
/// `ServeRouter`'s family cascade; Ollama keeps `ServeRouter`.
fn use_responses_router(key: &ApiKey) -> bool {
    use crate::services::provider_profile::{is_ollama_base, provider_profile_for_key};
    use crate::services::provider_protocol::ProviderProtocol;
    if key.is_copilot() {
        return true;
    }
    !is_ollama_base(&key.base_url)
        && matches!(
            provider_profile_for_key(key).default_protocol,
            ProviderProtocol::Openai | ProviderProtocol::ResponsesApi
        )
}

/// Start the responses-capable internal proxy (`ResponsesToChatRouter`) bound to
/// `key` — the same engine native codex/opencode/pi use. It speaks `/responses`
/// upstream (or chat, learned per `(plugin,key,model)` and persisted to config on
/// exit), so a plugin driving a gpt-5.x model with reasoning + tools works. The
/// loopback bearer is enforced and buffered token usage is accounted.
async fn start_responses_endpoint(
    key: &ApiKey,
    store: &SessionStore,
    tool: &str,
) -> anyhow::Result<EndpointHandle> {
    use crate::services::copilot_auth::CopilotTokenManager;
    use crate::services::provider_profile::{provider_profile_for_key, resolve_starter_base_url};
    use crate::services::provider_protocol::{ProviderProtocol, detect_provider_protocol};
    use crate::services::{ResponsesToChatRouter, ResponsesToChatRouterConfig};

    let token = random_auth_token();
    let profile = provider_profile_for_key(key);
    // Copilot is served like native codex's copilot router: the token manager
    // exchanges the GitHub token and supplies the upstream base URL, so the
    // config carries no static base/key. Everyone else targets the resolved REST
    // base directly.
    let (base_url, api_key, target_protocol, copilot_token_manager, is_starter) =
        if key.is_copilot() {
            (
                String::new(),
                String::new(),
                ProviderProtocol::Openai,
                Some(Arc::new(CopilotTokenManager::new(
                    key.key.as_str().to_string(),
                ))),
                false,
            )
        } else {
            let base = resolve_starter_base_url(&key.base_url);
            let proto = detect_provider_protocol(&base);
            (
                base,
                key.key.as_str().to_string(),
                proto,
                None,
                profile.serve_flags.is_starter,
            )
        };
    let router = ResponsesToChatRouter::new(ResponsesToChatRouterConfig {
        target_base_url: base_url,
        api_key,
        target_protocol,
        target_path_variant: None,
        copilot_token_manager,
        model_prefix: None,
        requires_reasoning_content: key.requires_reasoning_content.unwrap_or(false),
        actual_model: None,
        max_tokens_cap: None,
        responses_api_supported: key.responses_api_supported,
        is_starter,
        aivo_prefix_models: Vec::new(),
    })
    .with_tool(PLUGIN_ROUTE_TOOL)
    .with_seed_routes(key.routes_for_tool(PLUGIN_ROUTE_TOOL))
    .with_usage_accounting(store.clone(), key.id.clone(), tool.to_string())
    .with_auth_token(token.clone());
    let (port, route_cache, learned, handle) = router.start_background().await?;
    Ok(
        EndpointHandle::new(port, token, handle, None).with_persist(RoutePersist {
            store: store.clone(),
            key: key.clone(),
            route_cache,
            learned,
        }),
    )
}

// ── coding-agent run accounting (mirrors ai_launcher's LogEvent pair) ────────

/// The shared fields of the launch's log rows, captured once so the finished
/// row is the started template with the phase/exit/duration flipped.
struct Accounting {
    base: LogEvent,
    started: Instant,
}

async fn begin_accounting(
    store: &SessionStore,
    name: &str,
    key: Option<&ApiKey>,
    model: Option<&str>,
    args: &[String],
) -> Accounting {
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.display().to_string());
    let base = LogEvent {
        source: "run".to_string(),
        kind: "tool_launch".to_string(),
        event_group_id: Some(new_log_id()),
        key_id: key.map(|k| k.id.clone()),
        key_name: key.map(|k| k.display_name().to_string()),
        base_url: key.map(|k| k.base_url.clone()),
        tool: Some(name.to_string()),
        model: model.map(|m| m.to_string()),
        cwd,
        title: Some(match key {
            Some(k) => format!("{name} · {}", k.display_name()),
            None => name.to_string(),
        }),
        ..Default::default()
    };

    let _ = store
        .logs()
        .append(LogEvent {
            phase: Some("started".to_string()),
            body_text: Some(args.join(" ")),
            ..base.clone()
        })
        .await;
    if let Some(k) = key {
        let _ = store.record_selection(&k.id, name, model).await;
    }

    Accounting {
        base,
        started: Instant::now(),
    }
}

async fn finish_accounting(store: &SessionStore, acct: Accounting, code: i32) {
    let _ = store
        .logs()
        .append(LogEvent {
            phase: Some("finished".to_string()),
            exit_code: Some(code as i64),
            duration_ms: Some(acct.started.elapsed().as_millis() as i64),
            ..acct.base
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::{
        HelpKind, PluginServe, help_kind, missing_grants, plugin_model_request, plugin_serve,
        retained_grants, use_responses_router,
    };
    use crate::plugin::manifest::grantable_capabilities;
    use crate::services::claude_oauth::CLAUDE_OAUTH_SENTINEL;
    use crate::services::codex_oauth::CODEX_OAUTH_SENTINEL;
    use crate::services::cursor_acp::CURSOR_ACP_SENTINEL;
    use crate::services::gemini_oauth::GEMINI_OAUTH_SENTINEL;
    use crate::services::session_store::ApiKey;

    fn key(base: &str) -> ApiKey {
        ApiKey::new_with_protocol("id".into(), "n".into(), base.into(), None, "secret".into())
    }

    fn argv(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn help_kind_distinguishes_top_level_sub_and_none() {
        // `aivo <plugin> --help` / `-h` → top-level (gets the banner).
        assert!(matches!(help_kind(&argv(&["--help"])), HelpKind::TopLevel));
        assert!(matches!(help_kind(&argv(&["-h"])), HelpKind::TopLevel));
        // `aivo <plugin> <sub> --help` → sub-help (passthrough, no banner).
        assert!(matches!(
            help_kind(&argv(&["trust", "--help"])),
            HelpKind::Sub
        ));
        // No help flag, or `--help` only as a value, is not a help request.
        assert!(matches!(help_kind(&argv(&["-p", "hi"])), HelpKind::None));
        assert!(matches!(
            help_kind(&argv(&["-p", "explain --help"])),
            HelpKind::None
        ));
    }

    #[test]
    fn plugin_serve_picks_the_right_router() {
        // Plain REST keys (and Copilot, which the serve proxy exchanges) go to
        // ServeRouter.
        for base in [
            "https://api.openai.com/v1",
            "https://api.anthropic.com",
            "copilot",
        ] {
            assert_eq!(plugin_serve(&key(base)), PluginServe::Serve, "{base}");
        }
        // Cursor has its own ACP-backed router for coding-agent plugins.
        assert_eq!(plugin_serve(&key(CURSOR_ACP_SENTINEL)), PluginServe::Cursor);
        assert!(plugin_serve(&key(CURSOR_ACP_SENTINEL)).is_servable());
        // OAuth credentials are native-agent-only — blocked (no plugin proxy).
        for base in [
            CLAUDE_OAUTH_SENTINEL,
            CODEX_OAUTH_SENTINEL,
            GEMINI_OAUTH_SENTINEL,
        ] {
            assert_eq!(plugin_serve(&key(base)), PluginServe::Blocked, "{base}");
            assert!(!plugin_serve(&key(base)).is_servable(), "{base}");
        }
    }

    #[test]
    fn responses_router_for_openai_rest_and_copilot() {
        // OpenAI-protocol REST keys + Copilot → responses-capable internal proxy
        // (Copilot's token exchange runs inside that router).
        assert!(use_responses_router(&key("https://api.openai.com/v1")));
        assert!(use_responses_router(&key("https://openrouter.ai/api/v1")));
        assert!(use_responses_router(&key("copilot")));
        // Anthropic-protocol and Ollama keep the serve engine.
        assert!(!use_responses_router(&key("https://api.anthropic.com")));
        assert!(!use_responses_router(&key("ollama")));
    }

    #[test]
    fn lazy_probe_preserves_prior_grants_without_auto_escalation() {
        let capabilities = vec![
            "endpoint".to_string(),
            "endpoint".to_string(),
            "config-read".to_string(),
            "config-write".to_string(),
        ];
        let requested = grantable_capabilities(&capabilities);
        let prior = vec![
            "endpoint".to_string(),
            "config-read".to_string(),
            "old-cap".to_string(),
        ];

        assert_eq!(requested, ["endpoint"]);
        assert_eq!(retained_grants(&requested, &prior), ["endpoint"]);
        assert!(missing_grants(&requested, &prior).is_empty());
    }

    #[test]
    fn explicit_key_without_model_forces_model_picker() {
        assert_eq!(
            plugin_model_request(None, true).as_deref(),
            Some(""),
            "-k without -m must select a model after selecting the key"
        );
        assert_eq!(
            plugin_model_request(Some("gpt-4o".to_string()), true).as_deref(),
            Some("gpt-4o"),
            "an explicit -m value still wins"
        );
        assert!(plugin_model_request(None, false).is_none());
    }
}
