use super::*;
use crate::commands::models::fetch_models_for_select;

impl ChatTuiApp {
    pub(super) fn open_model_picker(
        &mut self,
        query: Option<String>,
        target: ModelSelectionTarget,
        auto_accept_exact: bool,
    ) {
        self.prepare_for_model_picker();
        let query = query.unwrap_or_default();
        self.overlay = Overlay::Picker(Box::new(PickerState::loading(
            "Select model",
            query,
            PickerKind::Model {
                target,
                auto_accept_exact,
            },
        )));
        let tx = self.tx.clone();
        let client = self.client.clone();
        let key = match self.current_model_picker_key() {
            Some(key) => key,
            None => return,
        };
        let cache = self.cache.clone();

        tokio::spawn(async move {
            let choices = load_model_choices(&client, &key, &cache).await;
            if choices.is_empty() {
                tx.send(RuntimeEvent::ModelsLoaded(Err(
                    "No models available for this provider".to_string(),
                )))
                .ok();
            } else {
                tx.send(RuntimeEvent::ModelsLoaded(Ok(choices))).ok();
            }
        });
    }

    pub(super) fn prepare_for_model_picker(&mut self) {
        if self.sending {
            self.cancel_inflight_request();
        }
    }

    /// Resolve and cache the active model's context window for the footer
    /// utilization stat. Cheap (in-memory cache) — call after any model/key
    /// change. 0 when the model isn't in the live catalog or snapshot.
    pub(super) async fn refresh_context_window(&mut self) {
        let limits = crate::services::model_metadata::resolve_limits(
            &self.cache,
            Some(&self.key.base_url),
            &self.model,
        )
        .await;
        // A `--max-context` override wins over the resolved window.
        self.context_window = self.context_window_override.or(limits.context).unwrap_or(0);
        // Valid `/effort` levels: live catalog (e.g. aivo/starter) or snapshot.
        self.model_reasoning_efforts = limits.reasoning_efforts.clone();
        // Reasoning-capable per the snapshot, or implied by advertised levels.
        self.model_supports_thinking =
            limits.caps.is_some_and(|c| c.reasoning) || !self.model_reasoning_efforts.is_empty();
        // Snapshot vision support (None when absent); gates the pre-flight refusal.
        self.model_image_input = limits.caps.map(|c| c.image_input);
        // This model's remembered effort, dropped if no longer a valid level.
        self.reasoning_effort = match self
            .session_store
            .get_chat_reasoning_effort(&self.model)
            .await
        {
            Some(level) if self.model_reasoning_efforts.contains(&level) => Some(level),
            _ => None,
        };
    }

    /// `/effort [level]`: bare opens a picker of the model's reasoning levels,
    /// `<level>` sets it directly. No-op (with a notice) for models that expose
    /// no levels.
    pub(super) async fn run_effort_command(&mut self, arg: Option<String>) {
        if self.model_reasoning_efforts.is_empty() {
            self.notice = Some((
                MUTED,
                format!("{} has no reasoning-effort levels", self.model),
            ));
            return;
        }
        match arg.map(|s| s.trim().to_ascii_lowercase()) {
            Some(level) if self.model_reasoning_efforts.contains(&level) => {
                self.apply_reasoning_effort(level).await;
            }
            Some(level) => {
                self.notice = Some((
                    ERROR,
                    format!(
                        "unknown effort '{level}' (choose: {})",
                        self.model_reasoning_efforts.join(", ")
                    ),
                ));
            }
            None => self.open_effort_picker(),
        }
    }

    fn open_effort_picker(&mut self) {
        let current = self.reasoning_effort.clone();
        let items: Vec<PickerEntry> = self
            .model_reasoning_efforts
            .iter()
            .map(|level| PickerEntry {
                label: picker_current_label(level.clone(), current.as_deref() == Some(level)),
                search_text: level.clone(),
                value: PickerValue::Effort(level.clone()),
            })
            .collect();
        let selected = current
            .as_deref()
            .and_then(|c| self.model_reasoning_efforts.iter().position(|l| l == c))
            .unwrap_or(0);
        let mut state =
            PickerState::ready("Reasoning effort", String::new(), items, PickerKind::Effort);
        state.selected = selected;
        self.overlay = Overlay::Picker(Box::new(state));
    }

    /// Set the reasoning effort: remember it (per-model) and persist it. The
    /// engine picks it up at the start of the next turn (carried in like the
    /// context window), so we never lock the engine here — that would block the
    /// event loop on an in-flight turn's guard. Choosing an effort implies you
    /// want the model to think, so this also turns thinking on if it was off.
    pub(super) async fn apply_reasoning_effort(&mut self, level: String) {
        self.reasoning_effort = Some(level.clone());
        let _ = self
            .session_store
            .set_chat_reasoning_effort(&self.model, Some(&level))
            .await;
        if !self.thinking_enabled {
            self.set_thinking_enabled(true).await;
        }
        self.notice = Some((MUTED, format!("reasoning effort: {level}")));
    }

    /// The effort the engine will request: the user's `/effort` choice, else —
    /// for a model with levels — the default (env or `medium`) if valid, else the
    /// first level; `None` when the model has no levels. Keeps the footer badge in
    /// step with what's sent and lets catalog-only models reason by default.
    pub(super) fn effective_reasoning_effort(&self) -> Option<String> {
        if let Some(level) = &self.reasoning_effort {
            return Some(level.clone());
        }
        if self.model_reasoning_efforts.is_empty() {
            return None;
        }
        let default = crate::agent::engine::default_reasoning_effort_level();
        if self.model_reasoning_efforts.contains(&default) {
            Some(default)
        } else {
            self.model_reasoning_efforts.first().cloned()
        }
    }

    pub(super) async fn apply_model(&mut self, raw_model: String) -> Result<()> {
        self.persist_model_selection(&raw_model).await?;

        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.refresh_context_window().await;
        // Routes are per-model — re-seed the format for the new model.
        self.format = seeded_chat_format(&self.key, &raw_model);
        self.billed_model = None;
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.notice = None;

        // If we have a live cursor ACP session, switch its model in place so
        // the conversation context is preserved. Drop the session on failure
        // so the next turn opens a fresh one with the new model.
        let drop_session = if let Some(session) = self.cursor_acp_session.as_mut() {
            session.set_model(&raw_model).await.is_err()
        } else {
            false
        };
        if drop_session {
            self.cursor_acp_session = None;
        }

        if !self.history.is_empty() {
            self.persist_history().await?;
        }
        Ok(())
    }

    pub(super) async fn complete_key_switch(
        &mut self,
        key: ApiKey,
        raw_model: String,
    ) -> Result<()> {
        self.key = key;
        self.raw_model = raw_model.clone();
        self.model = ChatCommand::transform_model_for_provider(&self.key.base_url, &raw_model);
        self.billed_model = None;
        self.copilot_tm = copilot_token_manager_for_key(&self.key);
        self.persist_model_selection(&raw_model).await?;
        self.refresh_context_window().await;

        self.start_new_chat();
        Ok(())
    }

    pub(super) async fn open_or_switch_key(&mut self, query: Option<String>) -> Result<()> {
        if let Some(query) = query {
            if let Some(key) = self.resolve_key_exact(&query).await? {
                self.begin_key_switch(key).await?;
                return Ok(());
            }
            self.open_key_picker(Some(query)).await?;
            return Ok(());
        }

        self.open_key_picker(None).await
    }

    pub(super) async fn begin_key_switch(&mut self, mut key: ApiKey) -> Result<()> {
        SessionStore::decrypt_key_secret(&mut key)?;
        if let Some(raw_model) = self.session_store.get_chat_model(&key.id).await? {
            self.complete_key_switch(key, raw_model).await?;
        } else {
            self.overlay = Overlay::None;
            self.open_model_picker(None, ModelSelectionTarget::KeySwitch(key), false);
        }
        Ok(())
    }

    pub(super) async fn open_key_picker(&mut self, query: Option<String>) -> Result<()> {
        let keys = self.session_store.get_keys().await?;
        if keys.is_empty() {
            self.notice = Some((ERROR, "No saved keys".to_string()));
            return Ok(());
        }

        let items = keys
            .into_iter()
            .map(|key| PickerEntry {
                label: format!("{} · {}", key.display_name(), key.base_url),
                search_text: key_search_text(&key),
                value: PickerValue::Key(key),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Keys",
            query.unwrap_or_default(),
            items,
            PickerKind::Key,
        )));
        Ok(())
    }

    pub(super) async fn open_resume_picker(&mut self, query: Option<String>) -> Result<()> {
        let mut sessions = load_resume_snapshots(&self.session_store).await?;
        if !self.history.is_empty()
            && !sessions
                .iter()
                .any(|session| session.session_id == self.session_id)
        {
            self.persist_history().await?;
            sessions = load_resume_snapshots(&self.session_store).await?;
        }

        // `--resume last` / `/resume last`: jump to the most recent session with
        // no picker. In-session the current chat was just persisted above and now
        // sorts newest, so skip it to land on the previous one; from a fresh
        // launch the newest IS the chat you just left.
        if query.as_deref() == Some("last") {
            let pick = if self.history.is_empty() {
                sessions.first()
            } else {
                sessions
                    .iter()
                    .find(|session| session.session_id != self.session_id)
            };
            match pick {
                Some(snapshot) => self.begin_resume_load(snapshot.clone()),
                None => self.notice = Some((MUTED, "No saved chat to resume".to_string())),
            }
            return Ok(());
        }

        if let Some(query) = &query
            && let Some(snapshot) = sessions.iter().find(|session| session.session_id == *query)
        {
            self.begin_resume_load(snapshot.clone());
            return Ok(());
        }

        let items = sessions
            .into_iter()
            .map(|session| PickerEntry {
                label: session.title.clone(),
                search_text: session.search_text(),
                value: PickerValue::Session(session),
            })
            .collect();

        self.overlay = Overlay::Picker(Box::new(PickerState::ready(
            "Sessions",
            query.unwrap_or_default(),
            items,
            PickerKind::Session,
        )));
        Ok(())
    }

    pub(super) fn open_help_overlay(&mut self) {
        self.overlay = Overlay::Help { scroll: 0 };
    }

    /// `/config`: a small toggle list of chat preferences, seeded from the live
    /// state. The list is fixed (no filter/add/remove) and each row flips on
    /// Enter/Space.
    pub(super) fn open_config_overlay(&mut self) {
        // Description varies by model capability — the toggle still persists for the
        // next thinking-capable model even when this one can't think.
        let thinking_desc = if self.model_supports_thinking {
            "let the model reason before answering (shown folded)"
        } else {
            "let the model reason (this model has no thinking)"
        };
        let items = vec![
            ConfigToggle {
                setting: ConfigSetting::Thinking,
                label: "Thinking",
                description: thinking_desc,
            },
            ConfigToggle {
                setting: ConfigSetting::AutoApprove,
                label: "Auto-approve tools",
                description: "run write/edit/bash without asking (Shift+Tab)",
            },
        ];
        self.overlay = Overlay::Config(ConfigOverlay { items, selected: 0 });
    }

    /// Whether `setting` is currently on — the single source of truth the `/config`
    /// renderer reads, so a row's checkbox can never drift from the live flag.
    pub(super) fn config_setting_enabled(&self, setting: ConfigSetting) -> bool {
        match setting {
            ConfigSetting::Thinking => self.thinking_enabled,
            ConfigSetting::AutoApprove => self.agent_auto_approve,
        }
    }

    /// Flip the preference for the `/config` row at `index`, applying it live and
    /// persisting it (best-effort). The renderer derives each row's state from the
    /// live flag, so there's no per-row copy to write back.
    pub(super) async fn toggle_config_setting(&mut self, index: usize) {
        let Some(setting) = (match &self.overlay {
            Overlay::Config(state) => state.items.get(index).map(|item| item.setting),
            _ => None,
        }) else {
            return;
        };
        match setting {
            ConfigSetting::Thinking => self.set_thinking_enabled(!self.thinking_enabled).await,
            // Reuse the shared setter so the live atomic + toast stay in lockstep.
            ConfigSetting::AutoApprove => self.set_auto_approve(!self.agent_auto_approve),
        }
    }

    /// Set the thinking on/off flag and persist it (best-effort). Both transcript
    /// fingerprints (history body and volatile tail) key on `thinking_enabled`, so
    /// the flip invalidates the memoized render on its own — no revision bump needed.
    /// The engine reads the flag at the start of the next turn, so a mid-session
    /// toggle applies next turn.
    pub(super) async fn set_thinking_enabled(&mut self, on: bool) {
        if self.thinking_enabled == on {
            return;
        }
        self.thinking_enabled = on;
        self.show_toast(if on { "Thinking on" } else { "Thinking off" });
        let _ = self.session_store.set_chat_thinking_enabled(on).await;
    }

    /// `/skills`: discover the agent skills available for the working dir and show
    /// them in a toggle overlay, seeded with each skill's persisted enabled state.
    /// Discovery is on-demand (cheap dir reads) so the list reflects skills added
    /// since launch.
    pub(super) async fn open_skills_overlay(&mut self) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let cwd_path = std::path::Path::new(&cwd);
        let disabled: std::collections::HashSet<String> = self
            .session_store
            .get_disabled_skills()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect();
        let mut items: Vec<SkillToggle> = crate::agent::skills::discover_skills(cwd_path)
            .into_iter()
            .map(|skill| SkillToggle {
                enabled: !disabled.contains(&skill.name),
                description: crate::agent::skills::advert_description(&skill.description),
                scope: crate::agent::skills::skill_scope(&skill.dir, cwd_path),
                dir: skill.dir,
                name: skill.name,
                body: skill.body,
            })
            .collect();
        sort_skill_rows(&mut items);
        // Keep the `/`-menu skill commands in sync with the freshly discovered set
        // (add/install/remove all reopen this overlay, so it's the convergence point).
        self.skill_commands = enabled_skill_commands(&items);
        self.overlay = Overlay::Skills(SkillsOverlay {
            items,
            selected: 0,
            query: String::new(),
            adding: None,
            pending_delete: None,
            viewing: None,
            detail_scroll: 0,
        });
        Ok(())
    }

    /// `/skills` from the composer: bare opens the overlay; `add <name>
    /// [description]` scaffolds a new skill and `remove|rm <name>` deletes one,
    /// without opening the overlay first.
    pub(super) async fn run_skills_command(&mut self, arg: Option<String>) -> Result<()> {
        let Some(arg) = arg.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
            return self.open_skills_overlay().await;
        };
        let (verb, rest) = match arg.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (arg.as_str(), ""),
        };
        match verb {
            "add" => self.submit_skill_add(rest.to_string()).await,
            "remove" | "rm" if rest.is_empty() => {
                self.notice = Some((ERROR, "Usage: /skills rm <name>".to_string()));
                Ok(())
            }
            "remove" | "rm" => self.remove_skill_named(rest).await,
            _ => {
                self.notice = Some((
                    ERROR,
                    "Usage: /skills [add <name>|<github:owner/repo> …] [rm <name>]".to_string(),
                ));
                Ok(())
            }
        }
    }

    /// Flip the enabled state of the skill at `index` in the open `/skills`
    /// overlay, persist it, and drop the engine so the next turn rebuilds with
    /// the new skill set.
    pub(super) async fn toggle_skill(&mut self, index: usize) -> Result<()> {
        let Some((name, enabled)) = (match &mut self.overlay {
            Overlay::Skills(state) => state.items.get_mut(index).map(|item| {
                item.enabled = !item.enabled;
                (item.name.clone(), item.enabled)
            }),
            _ => None,
        }) else {
            return Ok(());
        };
        self.session_store.set_skill_enabled(&name, enabled).await?;
        self.agent_engine = None;
        // A disabled skill drops out of the `/` menu; an enabled one returns.
        if let Overlay::Skills(state) = &self.overlay {
            self.skill_commands = enabled_skill_commands(&state.items);
        }
        Ok(())
    }

    /// Handle the `/skills` add-input. A first token that isn't a bare skill name
    /// (a `github:owner/repo`, a github.com URL, or a local path) is INSTALLED
    /// from that source (the rest is an optional `<skill-name>` filter or `*`);
    /// otherwise `name [description…]` SCAFFOLDS a template under
    /// `~/.config/aivo/skills`. Either way the overlay reopens.
    pub(super) async fn submit_skill_add(&mut self, input: String) -> Result<()> {
        let input = input.trim();
        let (first, rest) = match input.split_once(char::is_whitespace) {
            Some((a, b)) => (a, b.trim()),
            None => (input, ""),
        };
        // A scaffold name is `[A-Za-z0-9_-]`; anything else (a `/`, `:`, `.`, URL)
        // is an install source.
        if !first.is_empty() && !crate::agent::skills::is_valid_skill_name(first) {
            let only = (!rest.is_empty()).then(|| rest.to_string());
            return self
                .install_skill_from_source(first.to_string(), only)
                .await;
        }

        let (name, description) = match parse_skill_add_input(input) {
            Ok(parsed) => parsed,
            Err(msg) => {
                self.notice = Some((ERROR, msg));
                return Ok(());
            }
        };
        match crate::agent::skills::scaffold_skill(&name, &description) {
            Ok(path) => {
                // A freshly scaffolded skill starts enabled, clearing any stale
                // disabled flag left by a same-name skill removed earlier.
                self.session_store.set_skill_enabled(&name, true).await.ok();
                self.agent_engine = None;
                self.notice = Some((
                    MUTED,
                    format!("Created skill `{name}` — edit {}", path.display()),
                ));
            }
            Err(e) => {
                self.notice = Some((ERROR, format!("Failed to add skill: {e}")));
                return Ok(());
            }
        }
        self.open_skills_overlay().await
    }

    /// Install skill(s) from an online/local source into `~/.config/aivo/skills`,
    /// following the `skills/*/SKILL.md` convention. `only` is an optional skill-
    /// name filter (`*` = all). A multi-skill source with no filter lists the
    /// names so the user can re-run with one (or `*`).
    pub(super) async fn install_skill_from_source(
        &mut self,
        source: String,
        only: Option<String>,
    ) -> Result<()> {
        use crate::agent::skills::InstallOutcome;
        match crate::agent::skills::install_from_source(&source, only.as_deref()).await {
            Ok(InstallOutcome::Installed(names)) if names.is_empty() => {
                self.notice = Some((WARNING, format!("No skills found in `{source}`")));
            }
            Ok(InstallOutcome::Installed(names)) => {
                for name in &names {
                    self.session_store.set_skill_enabled(name, true).await.ok();
                }
                self.agent_engine = None;
                let label = if names.len() == 1 { "skill" } else { "skills" };
                self.notice = Some((MUTED, format!("Installed {label}: {}", names.join(", "))));
            }
            Ok(InstallOutcome::Ambiguous(names)) => {
                self.notice = Some((
                    WARNING,
                    format!(
                        "`{source}` has {} skills: {}. Re-run `/skills add {source} <name>` (or `*` for all)",
                        names.len(),
                        names.join(", ")
                    ),
                ));
            }
            Err(e) => self.notice = Some((ERROR, format!("Failed to install skill: {e}"))),
        }
        self.open_skills_overlay().await
    }

    /// Delete the skill at `index` (the overlay's confirmed `d`); resolves its
    /// name/dir/scope from the row then defers to the shared remover.
    pub(super) async fn remove_skill(&mut self, index: usize) -> Result<()> {
        let Some((name, dir, scope)) = (match &self.overlay {
            Overlay::Skills(state) => state
                .items
                .get(index)
                .map(|i| (i.name.clone(), i.dir.clone(), i.scope)),
            _ => None,
        }) else {
            return Ok(());
        };
        self.remove_scoped_skill(&name, &dir, scope).await
    }

    /// Delete a skill by name (`/skills rm <name>`); resolves its dir+scope from
    /// discovery, erroring if there's no such skill.
    pub(super) async fn remove_skill_named(&mut self, name: &str) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let cwd_path = std::path::Path::new(&cwd);
        let Some(skill) = crate::agent::skills::discover_skills(cwd_path)
            .into_iter()
            .find(|s| s.name == name)
        else {
            self.notice = Some((MUTED, format!("No skill named `{name}`")));
            return Ok(());
        };
        let scope = crate::agent::skills::skill_scope(&skill.dir, cwd_path);
        self.remove_scoped_skill(name, &skill.dir, scope).await
    }

    /// Shared removal: a `Project` skill (in the repo's `.agents`/`.aivo` skills)
    /// is left alone with a notice — that's the repo's to manage; a `User` skill
    /// has its folder deleted and the overlay refreshed so it disappears.
    async fn remove_scoped_skill(
        &mut self,
        name: &str,
        dir: &std::path::Path,
        scope: crate::agent::skills::SkillScope,
    ) -> Result<()> {
        if scope == crate::agent::skills::SkillScope::Project {
            self.notice = Some((
                WARNING,
                format!(
                    "`{name}` is a project skill ({}) — delete that folder to remove it",
                    dir.display()
                ),
            ));
            return Ok(());
        }
        match crate::agent::skills::remove_skill_dir(dir) {
            Ok(()) => {
                // Clear any leftover disabled flag so a re-add isn't stuck off.
                self.session_store.set_skill_enabled(name, true).await.ok();
                self.agent_engine = None;
                self.notice = Some((MUTED, format!("Removed skill `{name}`")));
            }
            Err(e) => self.notice = Some((ERROR, format!("Failed to remove skill: {e}"))),
        }
        self.open_skills_overlay().await
    }

    /// The MCP servers NOT to connect because the user explicitly turned them off
    /// in `/mcp` (`disabled_mcp_servers`). This is the *base* opt-out set; project
    /// `.mcp.json` STDIO servers are additionally held back until the user grants
    /// consent (see [`Self::connect_mcp_with_consent`]) since they run local code.
    pub(super) async fn effective_disabled_mcp_servers(&self) -> std::collections::HashSet<String> {
        self.session_store
            .get_disabled_mcp_servers()
            .await
            .unwrap_or_default()
            .into_iter()
            .collect()
    }

    /// Connect MCP, gating a repo's project `.mcp.json` STDIO servers behind a
    /// one-time consent — they would spawn arbitrary local commands the moment a
    /// connect runs. User-scope servers (`~/.config/aivo/mcp.json`) and HTTP
    /// (`url`) project servers connect freely (no local code execution). Until the
    /// user approves (once / always-for-this-repo), project stdio servers are held
    /// back from the connect; a consent card surfaces the exact commands.
    pub(super) async fn connect_mcp_with_consent(
        &mut self,
        cwd: String,
        base_disabled: std::collections::HashSet<String>,
    ) {
        let stdio = crate::agent::mcp::project_stdio_servers(std::path::Path::new(&cwd));
        if stdio.is_empty() {
            self.start_mcp_connect(cwd, base_disabled);
            return;
        }
        // Seed the session decision from the persistent per-repo allow-list once —
        // but only if the stored approval matches the CURRENT server set (digest),
        // so a changed `.mcp.json` re-prompts instead of silently reusing consent.
        if self.project_mcp_consent == ProjectMcpConsent::Unknown {
            let dir_key = canonical_dir_key(&cwd);
            let digest = project_mcp_digest(&stdio);
            if self
                .session_store
                .get_project_mcp_approved(&dir_key, &digest)
                .await
            {
                self.project_mcp_consent = ProjectMcpConsent::Allowed;
            }
        }
        if self.project_mcp_consent == ProjectMcpConsent::Allowed {
            self.start_mcp_connect(cwd, base_disabled);
            return;
        }
        // Unknown or Denied → hold the project stdio servers back from the connect
        // (user + HTTP project servers still come up).
        let mut held = base_disabled.clone();
        held.extend(stdio.iter().map(|(name, _)| name.clone()));
        // Unknown → surface the consent card (unless one is already up).
        if self.project_mcp_consent == ProjectMcpConsent::Unknown
            && self.pending_mcp_consent.is_none()
        {
            self.pending_mcp_consent = Some(McpConsentPrompt {
                servers: stdio,
                cwd: cwd.clone(),
                base_disabled,
            });
        }
        self.start_mcp_connect(cwd, held);
    }

    /// Resolve the project-MCP consent card. `y` runs the servers once (this
    /// session), `a` also remembers the approval for this repo, `n`/`Esc` denies
    /// (they stay held back). On approval the cached client/engine are dropped so
    /// the reconnect includes the previously-held project stdio servers.
    pub(super) async fn handle_mcp_consent_key(&mut self, key: KeyEvent) {
        let always = matches!(key.code, KeyCode::Char('a' | 'A'));
        let allow = always || matches!(key.code, KeyCode::Char('y' | 'Y'));
        let deny = matches!(key.code, KeyCode::Char('n' | 'N') | KeyCode::Esc);
        if !allow && !deny {
            return; // unrecognized key: leave the card up
        }
        let Some(prompt) = self.pending_mcp_consent.take() else {
            return;
        };
        if deny {
            self.project_mcp_consent = ProjectMcpConsent::Denied;
            self.show_toast("Project MCP servers not started");
            return;
        }
        self.project_mcp_consent = ProjectMcpConsent::Allowed;
        if always {
            let dir_key = canonical_dir_key(&prompt.cwd);
            let digest = project_mcp_digest(&prompt.servers);
            if let Err(e) = self
                .session_store
                .set_project_mcp_approved(&dir_key, &digest)
                .await
            {
                self.notice = Some((ERROR, format!("Couldn't remember the approval: {e}")));
            }
        }
        // Drop the cached client/engine so the reconnect picks up the project
        // stdio servers held back on the first connect.
        self.reset_mcp_after_config_change();
        self.start_mcp_connect(prompt.cwd, prompt.base_disabled);
        self.refresh_mcp_overlay_status();
        self.show_toast(if always {
            "Project MCP servers approved for this repo"
        } else {
            "Project MCP servers started for this session"
        });
    }

    /// `/mcp`: list the configured MCP servers with their live connection status
    /// (a snapshot of the current client) and per-server enabled state. Kicks off
    /// a background connect when nothing is cached yet so a reopened overlay shows
    /// real status instead of a perpetual "not connected".
    pub(super) async fn open_mcp_overlay(&mut self) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let disabled = self.effective_disabled_mcp_servers().await;
        // Kick the connect off first so the snapshot below reads "connecting…".
        // Project stdio servers are gated (consent card) the same as on a turn.
        self.connect_mcp_with_consent(cwd.clone(), disabled.clone())
            .await;
        let mut items: Vec<McpServerRow> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|srv| {
                    let enabled = !disabled.contains(&srv.name);
                    let (status, health) = self.mcp_server_status(&srv.name, enabled);
                    McpServerRow {
                        name: srv.name,
                        status,
                        health,
                        enabled,
                        scope: srv.scope,
                        command: srv.command,
                    }
                })
                .collect();
        sort_mcp_rows(&mut items);
        self.overlay = Overlay::Mcp(McpOverlay {
            items,
            selected: 0,
            query: String::new(),
            adding: None,
            pending_delete: None,
            viewing: None,
            detail_scroll: 0,
        });
        Ok(())
    }

    /// Parse the `/mcp` add-input (`command [args…]`, shell-quoted), DERIVE the
    /// server name from the command (the explicit name was redundant), write the
    /// server to the user `mcp.json`, and refresh the overlay so the agent picks
    /// it up. A parse problem stays in the overlay as a notice.
    pub(super) async fn submit_mcp_add(&mut self, input: String) -> Result<()> {
        // A pasted `mcpServers` JSON block (Ctrl+V in the add field) — the form
        // every README hands you — is parsed directly; the name comes from the
        // JSON key (env and extra fields preserved).
        let trimmed = input.trim();
        if trimmed.starts_with('{') {
            return self.submit_mcp_add_json(input).await;
        }
        // A bare http(s) URL is a remote Streamable HTTP server — wrap it as a
        // `{url}` config (no JSON typing needed) and route through the same path,
        // so naming, dedup, and auto-authorize are shared.
        if let Some(json) = bare_url_to_config(trimmed) {
            return self.submit_mcp_add_json(json).await;
        }
        let (command, args) = match parse_mcp_add_input(&input) {
            Ok(parsed) => parsed,
            Err(msg) => {
                self.notice = Some((ERROR, msg));
                return Ok(());
            }
        };
        // Derive a name from the command and de-duplicate against existing servers
        // (user + project), so two `filesystem` servers become `filesystem`/`-2`.
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let existing: std::collections::HashSet<String> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|s| s.name)
                .collect();
        let name = dedupe_name(
            crate::agent::mcp::derive_server_name(&command, &args),
            &existing,
        );
        if let Err(e) = crate::agent::mcp::add_user_server(&name, &command, &args).await {
            self.notice = Some((ERROR, format!("Failed to add MCP server: {e}")));
            return Ok(());
        }
        // A freshly added server starts enabled, even if a same-name one had been
        // disabled before.
        self.session_store
            .set_mcp_server_enabled(&name, true)
            .await
            .ok();
        self.notice = Some((MUTED, format!("Added MCP server `{name}`")));
        self.reset_mcp_after_config_change();
        self.open_mcp_overlay().await
    }

    /// Add server(s) from a pasted `mcpServers` JSON block, preserving each
    /// entry's `env`/extra fields and taking the name from the JSON key (or
    /// deriving it for a bare `{command,…}`), de-duplicating against existing.
    async fn submit_mcp_add_json(&mut self, input: String) -> Result<()> {
        let parsed = match crate::agent::mcp::parse_mcp_json(&input) {
            Ok(p) => p,
            Err(e) => {
                self.notice = Some((ERROR, format!("Couldn't parse MCP config: {e}")));
                return Ok(());
            }
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let mut existing: std::collections::HashSet<String> =
            crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
                .into_iter()
                .map(|s| s.name)
                .collect();
        let mut added = Vec::new();
        for (name_opt, value) in parsed {
            let name = dedupe_name(
                name_opt.unwrap_or_else(|| crate::agent::mcp::derive_name_from_value(&value)),
                &existing,
            );
            if let Err(e) = crate::agent::mcp::add_user_server_value(&name, &value).await {
                self.notice = Some((ERROR, format!("Failed to add `{name}`: {e}")));
                self.reset_mcp_after_config_change();
                return self.open_mcp_overlay().await;
            }
            self.session_store
                .set_mcp_server_enabled(&name, true)
                .await
                .ok();
            // A url server may need OAuth — queue it to auto-authorize if its
            // connect comes back 401, so the user needn't press Ctrl+O.
            if let Some(url) = value.get("url").and_then(|u| u.as_str()) {
                self.pending_mcp_auth.insert(name.clone(), url.to_string());
            }
            existing.insert(name.clone());
            added.push(name);
        }
        let label = if added.len() == 1 {
            "MCP server"
        } else {
            "MCP servers"
        };
        self.notice = Some((MUTED, format!("Added {label}: {}", added.join(", "))));
        self.reset_mcp_after_config_change();
        self.open_mcp_overlay().await
    }

    /// `/mcp` from the composer: bare opens the overlay; `add <command> [args…]`
    /// (name derived) and `remove|rm <name>` manage servers without opening it.
    pub(super) async fn run_mcp_command(&mut self, arg: Option<String>) -> Result<()> {
        let Some(arg) = arg.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
            return self.open_mcp_overlay().await;
        };
        let (verb, rest) = match arg.split_once(char::is_whitespace) {
            Some((verb, rest)) => (verb, rest.trim()),
            None => (arg.as_str(), ""),
        };
        match verb {
            "add" => self.submit_mcp_add(rest.to_string()).await,
            "remove" | "rm" if rest.is_empty() => {
                self.notice = Some((ERROR, "Usage: /mcp rm <name>".to_string()));
                Ok(())
            }
            "remove" | "rm" => self.remove_mcp_server_named(rest).await,
            _ => {
                self.notice = Some((
                    ERROR,
                    "Usage: /mcp [add <command> …] [rm <name>]".to_string(),
                ));
                Ok(())
            }
        }
    }

    /// Remove the server at `index` (the overlay's `d`); resolves the name+scope
    /// from the row then defers to the shared remover.
    pub(super) async fn remove_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, scope)) = (match &self.overlay {
            Overlay::Mcp(state) => state.items.get(index).map(|i| (i.name.clone(), i.scope)),
            _ => None,
        }) else {
            return Ok(());
        };
        self.remove_scoped_mcp_server(&name, scope).await
    }

    /// Remove a server by name (`/mcp rm <name>`); resolves its scope from config.
    pub(super) async fn remove_mcp_server_named(&mut self, name: &str) -> Result<()> {
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        let Some(scope) = crate::agent::mcp::configured_servers(std::path::Path::new(&cwd))
            .into_iter()
            .find(|s| s.name == name)
            .map(|s| s.scope)
        else {
            self.notice = Some((MUTED, format!("No MCP server named `{name}`")));
            return Ok(());
        };
        self.remove_scoped_mcp_server(name, scope).await
    }

    /// Shared removal: a `Project` server (from a repo `.mcp.json`) is left alone
    /// with a notice — that file is the repo's to edit; a `User` server is removed
    /// from the user `mcp.json` and the overlay refreshed so its tools drop.
    async fn remove_scoped_mcp_server(
        &mut self,
        name: &str,
        scope: crate::agent::mcp::ServerScope,
    ) -> Result<()> {
        if scope == crate::agent::mcp::ServerScope::Project {
            self.notice = Some((
                WARNING,
                format!("`{name}` is defined in .mcp.json — edit that file to remove it"),
            ));
            return Ok(());
        }
        match crate::agent::mcp::remove_user_server(name).await {
            Ok(true) => {
                // Clear any leftover disabled flag so a re-add isn't stuck off.
                self.session_store
                    .set_mcp_server_enabled(name, true)
                    .await
                    .ok();
                self.notice = Some((MUTED, format!("Removed MCP server `{name}`")));
                self.reset_mcp_after_config_change();
            }
            Ok(false) => self.notice = Some((MUTED, format!("`{name}` was not in mcp.json"))),
            Err(e) => self.notice = Some((ERROR, format!("Failed to remove MCP server: {e}"))),
        }
        self.open_mcp_overlay().await
    }

    /// After the configured server set changes, invalidate any in-flight connect
    /// and drop the cached client + engine so the next turn (or overlay reopen)
    /// reconnects with the new set.
    pub(super) fn reset_mcp_after_config_change(&mut self) {
        self.mcp_connect_gen = self.mcp_connect_gen.wrapping_add(1);
        self.mcp_client = None;
        self.mcp_connecting = false;
        self.mcp_connect_progress.clear();
        self.agent_engine = None;
    }

    /// Status + health for one server, read from the current client snapshot.
    /// A disabled server (user- or project-scoped) reads "off".
    fn mcp_server_status(&self, name: &str, enabled: bool) -> (String, McpHealth) {
        if !enabled {
            return ("off".to_string(), McpHealth::Disabled);
        }
        if let Some(client) = &self.mcp_client {
            if let Some(n) = client.tool_count(name) {
                let plural = if n == 1 { "" } else { "s" };
                return (format!("{n} tool{plural}"), McpHealth::Connected);
            }
            // A 401 isn't a hard failure — the server just needs OAuth.
            if client.needs_auth(name) {
                return ("needs authorization".to_string(), McpHealth::NeedsAuth);
            }
            if let Some(err) = client.error_for(name) {
                return (format!("failed: {err}"), McpHealth::Failed);
            }
        }
        // Mid-connect: this server's handshake may have already resolved even
        // though the whole set (and `mcp_client`) hasn't — show its real status
        // instead of a blanket "connecting…".
        if let Some((status, health)) = self.mcp_connect_progress.get(name) {
            return (status.clone(), *health);
        }
        if self.mcp_connecting {
            return ("connecting…".to_string(), McpHealth::Idle);
        }
        ("not connected".to_string(), McpHealth::Idle)
    }

    /// Flip the enabled state of the server at `index` in the open `/mcp` overlay,
    /// persist it, and drop + re-establish the MCP connection for the new server
    /// set. Reconnecting live (rather than deferring to the next turn) means the
    /// overlay updates from "connecting…" to the real tool count / failure while
    /// it's still open, so the toggle has visible feedback.
    pub(super) async fn toggle_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, enabled)) = (match &mut self.overlay {
            Overlay::Mcp(state) => state.items.get_mut(index).map(|item| {
                item.enabled = !item.enabled;
                (item.name.clone(), item.enabled)
            }),
            _ => None,
        }) else {
            return Ok(());
        };
        // Both user- and project-scoped servers use the global opt-out list; a
        // project `.mcp.json` server is enabled by default (the user owns their
        // own repo) and toggling it off just adds it to the opt-outs.
        self.session_store
            .set_mcp_server_enabled(&name, enabled)
            .await?;
        // Reuse the live client so the servers that *aren't* being toggled keep
        // their connection and status — only the toggled one changes — instead of
        // tearing down and reconnecting the whole set.
        self.reconnect_mcp_preserving_for_overlay();
        Ok(())
    }

    /// Kick a fresh background connect for the server set shown in the open `/mcp`
    /// overlay (derived from each row's enabled flag) and repaint every row to its
    /// interim status, so a toggle shows "connecting…"/"off" immediately and the
    /// real status lands when `McpConnected` resolves. A no-op when the overlay
    /// isn't the MCP one.
    pub(super) fn restart_mcp_connect_for_overlay(&mut self) {
        let disabled: std::collections::HashSet<String> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .filter(|i| !i.enabled)
                .map(|i| i.name.clone())
                .collect(),
            _ => return,
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        self.start_mcp_connect(cwd, disabled);
        self.refresh_mcp_overlay_status();
    }

    /// Recompute each `/mcp` row's status + health from the current client
    /// snapshot. Called when a background connect resolves so an open overlay
    /// updates live instead of stalling on a stale "connecting…" until reopened.
    pub(super) fn refresh_mcp_overlay_status(&mut self) {
        // Snapshot (name, enabled) first so the immutable status lookups don't
        // overlap the later mutable borrow of the overlay.
        let rows: Vec<(String, bool)> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .map(|i| (i.name.clone(), i.enabled))
                .collect(),
            _ => return,
        };
        let statuses: Vec<(String, McpHealth)> = rows
            .iter()
            .map(|(name, enabled)| self.mcp_server_status(name, *enabled))
            .collect();
        if let Overlay::Mcp(state) = &mut self.overlay {
            for (item, (status, health)) in state.items.iter_mut().zip(statuses) {
                item.status = status;
                item.health = health;
            }
        }
    }

    /// Start a background MCP connect, idempotently: a no-op when a client is
    /// already cached or a connect is in flight. The result returns as
    /// `RuntimeEvent::McpConnected`. Shared by the first agent turn and `/mcp`.
    pub(super) fn start_mcp_connect(
        &mut self,
        cwd: String,
        disabled: std::collections::HashSet<String>,
    ) {
        if self.mcp_client.is_some() || self.mcp_connecting {
            return;
        }
        self.mcp_connecting = true;
        self.mcp_connect_progress.clear();
        self.spawn_mcp_connect(cwd, disabled, self.mcp_connect_gen, None);
    }

    /// Reconnect for a changed server set in the open `/mcp` overlay (a toggle),
    /// reusing the live client so a server that's still enabled keeps its
    /// connection and its displayed status instead of every row flashing
    /// "connecting…". Only the changed server actually (re)connects. `mcp_client`
    /// is deliberately kept (not nulled) so status stays live during the connect;
    /// the engine is dropped so the next turn rebuilds with the new tool set.
    pub(super) fn reconnect_mcp_preserving_for_overlay(&mut self) {
        let disabled: std::collections::HashSet<String> = match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .iter()
                .filter(|i| !i.enabled)
                .map(|i| i.name.clone())
                .collect(),
            _ => return,
        };
        let cwd = if self.real_cwd.is_empty() {
            ".".to_string()
        } else {
            self.real_cwd.clone()
        };
        self.mcp_connect_gen = self.mcp_connect_gen.wrapping_add(1);
        self.mcp_connecting = true;
        self.mcp_connect_progress.clear();
        self.agent_engine = None;
        self.spawn_mcp_connect(cwd, disabled, self.mcp_connect_gen, self.mcp_client.clone());
        self.refresh_mcp_overlay_status();
    }

    /// Spawn the background connect task: report each server's status as its
    /// handshake resolves (`McpServerProgress`) and the assembled client at the
    /// end (`McpConnected`), both tagged with `generation` so a stale result is
    /// dropped. When `reuse` is given, still-enabled servers already live there
    /// are carried over instead of being reconnected.
    fn spawn_mcp_connect(
        &self,
        cwd: String,
        disabled: std::collections::HashSet<String>,
        generation: u64,
        reuse: Option<std::sync::Arc<crate::agent::mcp::McpClient>>,
    ) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let progress_tx = tx.clone();
            // Report each server's status as its handshake resolves so an open
            // `/mcp` overlay flips that row immediately, rather than every row
            // sitting on "connecting…" until the slowest server finishes.
            let report = move |name, status| {
                let (status, health) = mcp_status_from_connect(status);
                progress_tx
                    .send(RuntimeEvent::McpServerProgress {
                        name,
                        status,
                        health,
                        generation,
                    })
                    .ok();
            };
            let path = std::path::Path::new(&cwd);
            let client = std::sync::Arc::new(match &reuse {
                Some(prev) => {
                    prev.reconnect_enabled_with_progress(path, &disabled, report)
                        .await
                }
                None => {
                    crate::agent::mcp::McpClient::connect_enabled_with_progress(
                        path, &disabled, report,
                    )
                    .await
                }
            });
            tx.send(RuntimeEvent::McpConnected { client, generation })
                .ok();
        });
    }

    /// Start the interactive OAuth flow for the HTTP server at `index` in the open
    /// `/mcp` overlay. Runs on a background task (so the TUI keeps painting): it
    /// auto-opens the browser, emits the authorize URL as a notice, and reports
    /// the outcome via `RuntimeEvent::McpAuthorized`. A stdio server is rejected —
    /// OAuth only applies to `url` servers.
    pub(super) async fn authorize_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some((name, target)) = (match &self.overlay {
            Overlay::Mcp(state) => state
                .items
                .get(index)
                .map(|i| (i.name.clone(), i.command.clone())),
            _ => None,
        }) else {
            return Ok(());
        };
        if !target.contains("://") {
            self.notice = Some((
                WARNING,
                format!("`{name}` is a stdio server — OAuth applies only to HTTP (url) servers"),
            ));
            return Ok(());
        }
        self.start_mcp_authorize(name, target);
        Ok(())
    }

    /// Spawn the interactive OAuth flow for an HTTP MCP server `(name, url)` on a
    /// background task (the TUI keeps painting): it auto-opens the browser, emits
    /// the authorize URL as a notice, and reports the outcome via
    /// `RuntimeEvent::McpAuthorized`. Shared by the explicit Ctrl+O action and the
    /// auto-authorize that follows adding a server that turns out to need OAuth.
    pub(super) fn start_mcp_authorize(&mut self, name: String, url: String) {
        self.notice = Some((
            MUTED,
            format!("Authorizing `{name}` — a browser window should open…"),
        ));
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let url_tx = tx.clone();
            let result = crate::services::mcp_oauth::authorize(&url, None, |authorize_url| {
                url_tx
                    .send(RuntimeEvent::McpAuthorizeUrl {
                        url: authorize_url.to_string(),
                    })
                    .ok();
            })
            .await;
            tx.send(RuntimeEvent::McpAuthorized {
                name,
                result: result.map_err(|e| e.to_string()),
            })
            .ok();
        });
    }

    /// Clear the stored OAuth credential for the server at `index` and reconnect,
    /// so an authorized HTTP server drops back to "needs authorization".
    pub(super) async fn sign_out_mcp_server(&mut self, index: usize) -> Result<()> {
        let Some(name) = (match &self.overlay {
            Overlay::Mcp(state) => state.items.get(index).map(|i| i.name.clone()),
            _ => None,
        }) else {
            return Ok(());
        };
        match crate::services::mcp_token_store::remove(&name).await {
            Ok(true) => {
                self.notice = Some((MUTED, format!("Signed out of `{name}`")));
                self.reset_mcp_after_config_change();
                self.restart_mcp_connect_for_overlay();
            }
            Ok(false) => {
                self.notice = Some((MUTED, format!("`{name}` had no stored credentials")));
            }
            Err(e) => {
                self.notice = Some((ERROR, format!("Failed to sign out of `{name}`: {e}")));
            }
        }
        Ok(())
    }

    pub(super) async fn activate_picker_selection(
        &mut self,
        filtered_index: usize,
    ) -> Result<bool> {
        let (kind, value) = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((original_index, _)) = picker.filtered_items().get(filtered_index).copied()
            else {
                return Ok(false);
            };
            (
                picker.kind.clone(),
                picker.items[original_index].value.clone(),
            )
        };

        self.overlay = Overlay::None;

        match (kind, value) {
            (PickerKind::Model { target, .. }, PickerValue::Model(model)) => match target {
                ModelSelectionTarget::CurrentChat => self.apply_model(model).await?,
                ModelSelectionTarget::KeySwitch(key) => {
                    self.complete_key_switch(key, model).await?
                }
            },
            (PickerKind::Key, PickerValue::Key(key)) => {
                self.begin_key_switch(key).await?;
            }
            (PickerKind::Session, PickerValue::Session(session)) => {
                self.begin_resume_load(session);
            }
            (
                PickerKind::Rewind,
                PickerValue::RewindTurn {
                    history_index,
                    ordinal,
                },
            ) => {
                self.rewind_to_turn(history_index, ordinal).await?;
            }
            (PickerKind::Effort, PickerValue::Effort(level)) => {
                self.apply_reasoning_effort(level).await;
            }
            (PickerKind::Agent, PickerValue::Agent(name)) => {
                self.apply_agent_selection(name).await;
            }
            _ => {}
        }

        Ok(false)
    }

    pub(super) async fn delete_picker_selection(&mut self, filtered_index: usize) -> Result<bool> {
        let session = {
            let Overlay::Picker(picker) = &self.overlay else {
                return Ok(false);
            };
            let Some((_, item)) = picker.filtered_items().get(filtered_index).copied() else {
                return Ok(false);
            };
            match &item.value {
                PickerValue::Session(session) => session.clone(),
                _ => return Ok(false),
            }
        };

        let removed = self
            .session_store
            .delete_chat_session(&session.session_id)
            .await?;
        if !removed {
            self.notice = Some((ERROR, "Saved chat no longer exists".to_string()));
            return Ok(false);
        }

        if let Overlay::Picker(picker) = &mut self.overlay {
            picker.clear_pending_delete();
            picker.items.retain(|item| {
                !matches!(
                    &item.value,
                    PickerValue::Session(existing)
                        if existing.key_id == session.key_id && existing.session_id == session.session_id
                )
            });

            let filtered_len = picker.filtered_items().len();
            if filtered_len == 0 {
                self.overlay = Overlay::None;
                self.notice = Some((MUTED, "Saved chat deleted".to_string()));
                return Ok(false);
            }

            picker.selected = picker.selected.min(filtered_len.saturating_sub(1));
        }

        self.notice = Some((MUTED, "Saved chat deleted".to_string()));
        Ok(false)
    }

    pub(super) async fn resolve_key_exact(&self, query: &str) -> Result<Option<ApiKey>> {
        let keys = self.session_store.get_keys().await?;

        if let Some(key) = keys.iter().find(|key| key.id == query).cloned() {
            return Ok(Some(key));
        }

        let name_matches = keys
            .into_iter()
            .filter(|key| key.name == query)
            .collect::<Vec<_>>();

        if name_matches.len() == 1 {
            Ok(name_matches.into_iter().next())
        } else {
            Ok(None)
        }
    }

    pub(super) fn current_model_picker_key(&self) -> Option<ApiKey> {
        let Overlay::Picker(picker) = &self.overlay else {
            return None;
        };
        match &picker.kind {
            PickerKind::Model {
                target: ModelSelectionTarget::CurrentChat,
                ..
            } => Some(self.key.clone()),
            PickerKind::Model {
                target: ModelSelectionTarget::KeySwitch(key),
                ..
            } => Some(key.clone()),
            _ => None,
        }
    }

    pub(super) async fn persist_history(&self) -> Result<()> {
        let stored = to_stored_messages(&self.history);
        let title = session_title_from_messages(&self.history, &self.raw_model);
        let preview = session_preview_text_from_messages(&self.history, &self.raw_model);
        // `session_tokens` is the running cumulative for this session (each turn
        // folds in the provider-measured split); the index entry stores the total
        // so `aivo stats --since` can attribute windowed chat usage per model.
        self.session_store
            .save_chat_session_with_id(
                &self.key.id,
                &self.key.base_url,
                self.persist_cwd(),
                &self.session_id,
                &self.raw_model,
                self.billed_model.as_deref(),
                &stored,
                &title,
                &preview,
                self.session_tokens,
            )
            .await?;
        // Durable resume: also persist the agent engine's exact conversation
        // (assistant tool_calls + tool results with ids) so a resume restores tool
        // history verbatim. Best-effort and non-blocking: `try_lock` skips the
        // export when a turn is mid-flight (the text save above preserved the prior
        // blob), and a non-agent chat has no engine. Done after the text save so
        // the session file exists for `save_agent_messages` to update.
        if let Some(session) = &self.agent_engine
            && let Ok(engine) = session.engine.try_lock()
        {
            let conversation = engine.export_conversation();
            drop(engine);
            // Persist even when empty: a `/rewind` to the first turn empties the
            // engine, and storing that (which clears the blob — see
            // `save_agent_messages`) keeps resume from restoring the stale
            // pre-rewind transcript. `try_lock` already skips the mid-turn case.
            let _ = self
                .session_store
                .save_agent_messages(&self.session_id, &conversation)
                .await;
        }
        Ok(())
    }

    pub(super) fn begin_resume_load(&mut self, preview: SessionPreview) {
        self.discard_resume_state();
        self.overlay = Overlay::None;
        if self.sending {
            self.cancel_inflight_request();
        }

        self.resume_restore_state = Some(ResumeRestoreState::capture(self));
        self.clear_for_resume_loading();
        // The new session id will come from storage; drop any live cursor ACP
        // session since cursor doesn't know about the resumed session.
        self.cursor_acp_session = None;
        self.resume_request_id = self.resume_request_id.wrapping_add(1);
        let request_id = self.resume_request_id;
        self.loading_resume = Some(LoadingResume {
            request_id,
            preview: preview.clone(),
        });

        let session_store = self.session_store.clone();
        let tx = self.tx.clone();
        let task = tokio::spawn(async move {
            let result = load_resume_session(&session_store, &preview).await;
            let _ = tx.send(RuntimeEvent::ResumeLoaded { request_id, result });
        });
        self.resume_task = Some(task);
    }

    pub(super) async fn apply_loaded_session(&mut self, session: LoadedSession) -> Result<()> {
        if self.key.id != session.key_id {
            let key = self
                .session_store
                .get_key_by_id(&session.key_id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Saved key for this chat is no longer available"))?;
            self.key = key;
            self.copilot_tm = copilot_token_manager_for_key(&self.key);
        }

        self.overlay = Overlay::None;
        // Drop any live agent engine/serve/permission so the resumed
        // conversation re-seeds its context from `session.messages` on the next
        // turn. Reusing a same-key/model engine would continue the PREVIOUS
        // session's thread (`/new` and key/model switches reset it the same way).
        self.agent_engine = None;
        self.agent_permission = None;
        self.stop_agent_serve();
        self.session_id = session.session_id;
        // Re-seed the running token total from the stored entry so further turns
        // accumulate on top of it (the index save overwrites with the cumulative).
        self.session_tokens = self
            .session_store
            .chat_session_tokens(&self.session_id)
            .await;
        self.history = session.messages;
        self.expanded_thinking.clear();
        self.expanded_output.clear();
        self.local_outputs.clear();
        self.reasoning_durations.clear();
        // Restore the exact agent transcript (tool calls + results with ids) into
        // the next engine build instead of the lossy text seed. `None` for
        // non-agent or pre-feature sessions → falls back to the text seed.
        self.pending_agent_messages = session.engine_messages;
        self.draft.clear();
        self.cursor = 0;
        self.command_menu.reset();
        self.draft_history_index = None;
        self.draft_history_stash = None;
        self.pending_response.clear();
        self.pending_submit = None;
        self.format = seeded_chat_format(&self.key, &session.raw_model);
        self.last_usage = None;
        self.context_tokens = estimate_context_tokens(&self.history);
        self.follow_output = true;
        self.transcript_scroll = 0;
        self.raw_model = session.raw_model.clone();
        self.model =
            ChatCommand::transform_model_for_provider(&self.key.base_url, &session.raw_model);
        self.billed_model = None;
        self.refresh_context_window().await;
        self.persist_model_selection(&session.raw_model).await?;
        Ok(())
    }

    async fn persist_model_selection(&self, raw_model: &str) -> Result<()> {
        self.session_store
            .set_chat_model(&self.key.id, raw_model)
            .await?;
        self.session_store
            .record_selection(&self.key.id, "chat", Some(raw_model))
            .await?;
        self.update_last_selection(raw_model).await;
        Ok(())
    }

    /// Mirror launch-time behavior: keep the global "selected key & model" in
    /// sync when the user switches key/model mid-session (`/key`, `/model`,
    /// resume) so `aivo run`/`aivo start`/`aivo info` recall what chat is
    /// actually using — not just the key/model it launched with.
    ///
    /// Preserves the existing launchable tool (so `aivo run` with no tool still
    /// recalls the last *launchable* tool, not "chat"), skips the ephemeral HF
    /// synthetic key, and writes the *stored* key's canonical `base_url`: the
    /// live `self.key` may carry a sentinel resolved to a real URL (ollama,
    /// aivo-starter), and a resolved URL would fail `get_last_selection`'s
    /// validity check against the saved record and be pruned as stale.
    /// Best-effort — a failed write never aborts the switch.
    async fn update_last_selection(&self, raw_model: &str) {
        if self.key.id == crate::services::huggingface::HF_LOCAL_KEY_ID {
            return;
        }
        let Ok(Some(stored_key)) = self.session_store.get_key_by_id(&self.key.id).await else {
            return;
        };
        let existing_tool = self
            .session_store
            .get_last_selection()
            .await
            .ok()
            .flatten()
            .map(|sel| sel.tool);
        let _ = self
            .session_store
            .set_last_selection(
                &stored_key,
                existing_tool.as_deref().unwrap_or("chat"),
                Some(raw_model),
            )
            .await;
    }

    pub(super) fn scroll_up(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.effective_max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(step.max(1));
    }

    pub(super) fn scroll_down(&mut self) {
        let step = usize::from(self.transcript_view_height.max(4) / 2);
        let max_scroll = self.effective_max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + step.max(1)).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_up_lines(&mut self, lines: usize) {
        let max_scroll = self.effective_max_scroll();
        if self.follow_output {
            self.transcript_scroll = max_scroll;
            self.follow_output = false;
        }
        self.transcript_scroll = self.transcript_scroll.saturating_sub(lines);
    }

    pub(super) fn scroll_down_lines(&mut self, lines: usize) {
        let max_scroll = self.effective_max_scroll();
        self.follow_output = false;
        self.transcript_scroll = (self.transcript_scroll + lines).min(max_scroll);
        if self.transcript_scroll >= max_scroll {
            self.follow_output = true;
        }
    }

    pub(super) fn scroll_to_top(&mut self) {
        self.transcript_scroll = 0;
        self.follow_output = false;
    }

    pub(super) fn scroll_to_bottom(&mut self) {
        self.transcript_scroll = self.effective_max_scroll();
        self.follow_output = true;
    }

    /// Max scroll for the hot wheel/key handlers: reuse the value the last
    /// render computed when it's available (cheap — no rebuild), else recompute.
    /// The render re-clamps `transcript_scroll` every pass, so a one-frame-stale
    /// value only ever causes a transient that the next frame corrects.
    pub(super) fn effective_max_scroll(&self) -> usize {
        self.last_max_scroll.unwrap_or_else(|| self.max_scroll())
    }

    pub(super) fn max_scroll(&self) -> usize {
        let transcript = self.build_transcript();
        // Word-wrap to match the render's row count (char-wrap under-counts).
        let wrapped = wrap_transcript(
            &transcript.lines,
            &transcript.bar_colors,
            self.transcript_width,
        );
        wrapped
            .rows
            .len()
            .saturating_sub(usize::from(self.transcript_view_height))
    }

    pub(super) fn selected_transcript_text(&self) -> Option<String> {
        let selection = self.transcript_selection?;
        let rows = &self.transcript_hitbox.as_ref()?.rows;
        selected_text_from_rows(rows, selection)
    }

    /// Text under the full-screen selection, from the captured [`ScreenSurface`].
    pub(super) fn selected_screen_text(&self) -> Option<String> {
        let selection = self.screen_selection?;
        let rows = &self.screen_surface.as_ref()?.rows;
        selected_text_from_rows(rows, selection)
    }

    /// Text under whichever selection is live — transcript or screen.
    pub(super) fn selected_any_text(&self) -> Option<String> {
        self.selected_transcript_text()
            .or_else(|| self.selected_screen_text())
    }
}

/// Parse a `/mcp` add line into `(command, args)`, shell-splitting so quoted
/// args/paths survive. The server name is derived from the command (see
/// `mcp::derive_server_name`), not typed. `Err` is a user-facing message.
/// If `input` is a bare http(s) URL (a remote Streamable HTTP server), the `{url}`
/// JSON config to add for it; `None` for anything else (a `{…}` block or a
/// command line, handled on their own paths).
pub(super) fn bare_url_to_config(input: &str) -> Option<String> {
    let t = input.trim();
    if t.starts_with("http://") || t.starts_with("https://") {
        let url = t.split_whitespace().next().unwrap_or(t);
        Some(serde_json::json!({ "url": url }).to_string())
    } else {
        None
    }
}

pub(super) fn parse_mcp_add_input(
    input: &str,
) -> std::result::Result<(String, Vec<String>), String> {
    let tokens = shlex::split(input.trim()).unwrap_or_default();
    let Some((command, args)) = tokens.split_first() else {
        return Err(
            "Usage: <command> [args…]  (e.g. npx -y @modelcontextprotocol/server-filesystem ~)"
                .to_string(),
        );
    };
    Ok((command.clone(), args.to_vec()))
}

/// Append `-2`, `-3`, … to `base` until it doesn't collide with an existing name.
/// Stable key for the per-repo project-MCP allow-list: the canonicalized cwd
/// (symlinks resolved, so the same repo reached two ways shares one decision),
/// falling back to the raw path when it can't be canonicalized.
pub(super) fn canonical_dir_key(cwd: &str) -> String {
    std::fs::canonicalize(cwd)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| cwd.to_string())
}

fn dedupe_name(base: String, existing: &std::collections::HashSet<String>) -> String {
    if !existing.contains(&base) {
        return base;
    }
    (2..)
        .map(|n| format!("{base}-{n}"))
        .find(|candidate| !existing.contains(candidate))
        .unwrap_or(base)
}

/// Parse a `/skills add` line into `(name, description)`: the first whitespace-
/// delimited token is the (folder-safe) name, the rest is a free-text one-line
/// description (empty is fine — a placeholder is templated in). `Err` is a
/// user-facing message.
pub(super) fn parse_skill_add_input(input: &str) -> std::result::Result<(String, String), String> {
    let input = input.trim();
    let (name, description) = match input.split_once(char::is_whitespace) {
        Some((name, rest)) => (name, rest.trim()),
        None => (input, ""),
    };
    if !crate::agent::skills::is_valid_skill_name(name) {
        return Err(
            "Usage: <name> [description] — name is letters, digits, '-' or '_'".to_string(),
        );
    }
    Ok((name.to_string(), description.to_string()))
}

/// Map a connect-time per-server outcome to the overlay's `(status, health)`
/// display pair — the same vocabulary `mcp_server_status` uses once the full
/// client lands, so an incrementally-updated row reads identically to a finished
/// one.
pub(super) fn mcp_status_from_connect(
    status: crate::agent::mcp::ServerConnectStatus,
) -> (String, McpHealth) {
    use crate::agent::mcp::ServerConnectStatus;
    match status {
        ServerConnectStatus::Connected { tools } => {
            let plural = if tools == 1 { "" } else { "s" };
            (format!("{tools} tool{plural}"), McpHealth::Connected)
        }
        ServerConnectStatus::NeedsAuth => ("needs authorization".to_string(), McpHealth::NeedsAuth),
        ServerConnectStatus::Failed(msg) => (format!("failed: {msg}"), McpHealth::Failed),
    }
}

/// Order `/mcp` rows problems-first: failed servers float to the top (act on
/// them), then connected/connecting, with disabled sunk to the bottom;
/// alphabetical within each group. Applied once at open (not on the live status
/// refresh) so rows don't jump under the cursor mid-session.
pub(super) fn sort_mcp_rows(rows: &mut [McpServerRow]) {
    rows.sort_by(|a, b| {
        mcp_health_rank(a.health)
            .cmp(&mcp_health_rank(b.health))
            .then_with(|| a.name.cmp(&b.name))
    });
}

fn mcp_health_rank(health: McpHealth) -> u8 {
    match health {
        McpHealth::Failed => 0,
        // Actionable (authorize with Ctrl+O) — surface near the top.
        McpHealth::NeedsAuth => 1,
        McpHealth::Connected => 2,
        McpHealth::Idle => 3,
        McpHealth::Disabled => 4,
    }
}

/// Order `/skills` rows: enabled first, disabled sunk to the bottom; alphabetical
/// within each group.
pub(super) fn sort_skill_rows(rows: &mut [SkillToggle]) {
    rows.sort_by(|a, b| b.enabled.cmp(&a.enabled).then_with(|| a.name.cmp(&b.name)));
}

/// The enabled skills from an open `/skills` overlay, as `/`-menu slash commands.
pub(super) fn enabled_skill_commands(rows: &[SkillToggle]) -> Vec<SkillCommand> {
    rows.iter()
        .filter(|i| i.enabled)
        .map(|i| SkillCommand {
            name: i.name.clone(),
            description: i.description.clone(),
        })
        .collect()
}

/// Fetch model metadata for the picker, falling back to cached IDs on error.
/// On a successful detailed fetch we also seed the IDs cache so other commands
/// stay warm.
async fn load_model_choices(
    client: &reqwest::Client,
    key: &ApiKey,
    cache: &crate::services::ModelsCache,
) -> Vec<ModelChoice> {
    match crate::commands::models::fetch_models_detailed(client, key).await {
        Ok(infos) => {
            let ids: Vec<String> = infos.iter().map(|m| m.id.clone()).collect();
            let cache_key = crate::commands::models::model_cache_key_for_key(key);
            cache.set(&cache_key, ids).await;
            infos
                .into_iter()
                .map(|m| ModelChoice {
                    label: crate::commands::models::picker_label(&m),
                    id: m.id,
                })
                .collect()
        }
        Err(_) => fetch_models_for_select(client, key, cache)
            .await
            .into_iter()
            .map(|id| ModelChoice {
                label: id.clone(),
                id,
            })
            .collect(),
    }
}
