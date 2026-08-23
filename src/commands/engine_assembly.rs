//! Shared `AgentEngine` assembly for the two hosts (chat TUI and `-e` oneshot).
//! Everything both hosts wire identically — guides, skills, subagents, grants,
//! jobs, artifacts, session mail, LSP, hooks — is built here once; host-specific
//! setters (TUI: rewind/session-context/confirm-before-build; oneshot: budgets/
//! require-completion/verified-baseline) stay at the call sites, so drift means
//! editing a profile field, not keeping two hand-copied blocks in sync.

use std::path::{Path, PathBuf};

use crate::agent::engine::AgentEngine;
use crate::agent::mcp::{self, McpClient};
use crate::services::session_store::SessionStore;

pub struct EngineAssembly<'a> {
    pub session_store: &'a SessionStore,
    pub cwd: &'a str,
    pub model: &'a str,
    pub base_url: &'a str,
    pub context_window: u32,
    /// 0 = unbounded (the TUI's esc is the brake); oneshot passes its budget.
    pub max_steps: u32,
    pub injected_context: Option<&'a str>,
    pub jobs: std::sync::Arc<crate::agent::jobs::JobTable>,
    pub artifacts_dir: PathBuf,
    pub session_id: &'a str,
}

impl EngineAssembly<'_> {
    /// Build the engine with everything host-independent applied. Returns the
    /// session mail alongside so a headless host can register presence on it.
    pub async fn build(self) -> (AgentEngine, crate::services::session_mail::SessionMail) {
        let cwd_path = Path::new(self.cwd);
        let date = chrono::Local::now().format("%Y-%m-%d").to_string();
        let guides = crate::agent::system_prompt::discover_project_guides(cwd_path);
        // Discovered skills minus `/skills`-disabled, plus the create-agent builtin.
        let disabled: std::collections::HashSet<String> = self
            .session_store
            .get_disabled_skills()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        let skills = crate::agent::skills::engine_skills(cwd_path, &disabled);
        let mut engine = AgentEngine::new(
            self.cwd,
            self.model,
            &date,
            &guides,
            &skills,
            self.context_window,
            self.max_steps,
        );
        // The bundled aivo-starter provider is first-party: brand the agent so it
        // presents as aivo's assistant instead of disclosing the upstream model.
        // BYOK keys stay honest (no branding).
        if crate::services::provider_profile::is_aivo_starter_base(self.base_url) {
            engine.set_first_party();
        }
        if let Some(ctx) = self.injected_context {
            engine.append_system_context(ctx);
        }
        // Named specialist sub-agents (project `.aivo/agents`/`.claude/agents`,
        // then `~/.config/aivo/agents`); the model delegates via `agent`.
        let subagents =
            crate::agent::subagents::discover_subagents(cwd_path, self.session_store.config_dir());
        engine.set_subagents(&subagents);
        // Delegations re-resolve profiles from disk, so one authored or edited
        // mid-run delegates correctly even before the advert refreshes.
        engine.set_agents_dir(self.session_store.config_dir());
        // Persistent grant store so "always allow"s survive across sessions.
        engine.set_grants_path(self.session_store.config_dir());
        engine.set_jobs(self.jobs);
        // Durable sub-agent reports — without this, delegated work gets stubbed
        // away by in-run compaction.
        engine.set_artifacts_dir(self.artifacts_dir);
        let mail = crate::services::session_mail::SessionMail::new(
            self.session_store.config_dir(),
            self.session_id,
        );
        engine.set_session_mail(mail.clone());
        // LSP diagnostics-after-edit (default on; AIVO_AGENT_LSP=0 opts out).
        engine.maybe_enable_lsp(cwd_path);
        // User lifecycle hooks (~/.config/aivo/hooks.json).
        engine.set_hooks(std::sync::Arc::new(
            crate::agent::hooks::HookSet::load_default(),
        ));
        (engine, mail)
    }
}

/// Headless MCP connect for `-e`: same servers and `/mcp` opt-outs as the TUI,
/// but with no way to show a consent card, project `.mcp.json` servers are held
/// back unless a PRIOR interactive session stored an approval for this exact
/// server set — fail closed. Returns `None` when nothing is configured or connects.
pub async fn headless_external_tools(
    store: &SessionStore,
    cwd: &str,
) -> Option<std::sync::Arc<dyn crate::agent::engine::ExternalTools>> {
    let mut held: std::collections::HashSet<String> = store
        .get_disabled_mcp_servers()
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
    let gated = mcp::project_gated_servers(Path::new(cwd));
    if !gated.is_empty() {
        let dir_key = mcp::canonical_dir_key(cwd);
        let digest = mcp::project_mcp_digest(&gated);
        if !store.get_project_mcp_approved(&dir_key, &digest).await {
            held.extend(gated.into_iter().map(|(name, _)| name));
        }
    }
    let client = McpClient::connect_enabled_with_progress(Path::new(cwd), &held, |_, _| {}).await;
    if !client.has_tools() {
        return None;
    }
    let client = std::sync::Arc::new(client);
    let disabled_tools: std::collections::HashSet<String> = store
        .get_disabled_mcp_tools()
        .await
        .unwrap_or_default()
        .into_iter()
        .collect();
    Some(if disabled_tools.is_empty() {
        client
    } else {
        std::sync::Arc::new(crate::agent::mcp::FilteredTools::new(
            client,
            disabled_tools,
        ))
    })
}
