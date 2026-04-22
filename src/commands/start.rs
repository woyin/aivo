use anyhow::Result;
use console::{Key, Term};

use crate::cli::parse_env_vars;
use crate::commands::keys::prompt_pick_key_without_activation;
use crate::commands::models::{
    fetch_models_for_select, model_display_label, prompt_model_picker, resolve_model_placeholder,
};
use crate::commands::print_launch_preview;
use crate::errors::ExitCode;
use crate::services::ai_launcher::{AILauncher, AIToolType, LaunchOptions};
use crate::services::http_utils;
use crate::services::models_cache::ModelsCache;
use crate::services::session_store::{ApiKey, LastSelection, SessionStore};
use crate::style;
use crate::tui::FuzzySelect;

#[derive(Debug, Clone)]
pub struct StartFlowArgs {
    pub model: Option<String>,
    pub key: Option<String>,
    pub tool: Option<String>,
    pub debug: bool,
    pub dry_run: bool,
    pub refresh: bool,
    pub yes: bool,
    pub envs: Vec<String>,
}

struct Resolved<T> {
    value: T,
    interactive: bool,
}

pub struct StartCommand {
    session_store: SessionStore,
    ai_launcher: AILauncher,
    cache: ModelsCache,
}

impl StartCommand {
    pub fn new(session_store: SessionStore, ai_launcher: AILauncher, cache: ModelsCache) -> Self {
        Self {
            session_store,
            ai_launcher,
            cache,
        }
    }

    pub async fn execute(&self, args: StartFlowArgs) -> ExitCode {
        match self.execute_internal(args).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("{} {}", style::red("Error:"), e);
                ExitCode::UserError
            }
        }
    }

    async fn execute_internal(&self, args: StartFlowArgs) -> Result<ExitCode> {
        // Use global last selection for defaults
        let last_sel = self.session_store.get_last_selection().await?;

        if last_sel.is_none() {
            eprintln!(
                "{}",
                style::dim("No saved selection yet. I'll help you pick one.")
            );
        }

        let key_explicit = args.key.is_some();

        // Resolve tool first (uses last selection for default)
        let tool = self.resolve_tool(args.tool.as_deref(), last_sel.as_ref())?;

        let key = self
            .resolve_key(args.key.as_deref(), last_sel.as_ref())
            .await?;

        // Determine model: if -k was explicit, force picker; otherwise use last selection
        let model_arg = if args.model.is_some() {
            args.model
        } else if key_explicit {
            // -k used without -m → force model picker
            Some(String::new())
        } else {
            // No -k, no -m → check last selection
            match last_sel.as_ref() {
                Some(sel) if sel.key_id == key.value.id => {
                    sel.model.clone().or(Some(String::new()))
                }
                _ => None, // will trigger picker below
            }
        };

        let model = self
            .resolve_model(model_arg, last_sel.as_ref(), &key, args.refresh, tool.value)
            .await?;

        let _ = self
            .session_store
            .set_last_selection(&key.value, tool.value.as_str(), model.value.as_deref())
            .await;

        let launch_model = resolve_model_placeholder(model.value.clone());

        let env = parse_env_vars(&args.envs);
        let skip_confirm =
            last_sel.is_some() || (key.interactive && tool.interactive && model.interactive);

        if args.dry_run {
            let plan = self
                .ai_launcher
                .prepare_launch(&LaunchOptions {
                    tool: tool.value,
                    args: Vec::new(),
                    debug: args.debug,
                    model: launch_model,
                    env: (!env.is_empty()).then_some(env),
                    key_override: Some(key.value),
                })
                .await?;
            print_launch_preview(&plan);
            return Ok(ExitCode::Success);
        }

        if !args.yes {
            let provider = super::truncate_url_for_display(&key.value.base_url, 50);
            eprintln!(
                "{}{}",
                style::cyan(tool.value.as_str()),
                style::dim(format!(
                    " · {} · {}",
                    provider,
                    model_display_label(model.value.as_deref())
                ))
            );
            if !skip_confirm && !confirm("Run?")? {
                return Ok(ExitCode::Success);
            }
        }

        let exit_code = self
            .ai_launcher
            .launch(&LaunchOptions {
                tool: tool.value,
                args: Vec::new(),
                debug: args.debug,
                model: launch_model,
                env: (!env.is_empty()).then_some(env),
                key_override: Some(key.value),
            })
            .await?;

        Ok(match exit_code {
            0 => ExitCode::Success,
            n => ExitCode::ToolExit(n),
        })
    }

    async fn resolve_key(
        &self,
        key_arg: Option<&str>,
        last_sel: Option<&LastSelection>,
    ) -> Result<Resolved<ApiKey>> {
        if matches!(key_arg, Some("")) {
            return self.prompt_select_key(last_sel).await;
        }

        if let Some(key_id_or_name) = key_arg {
            let matches = self
                .session_store
                .find_keys_by_id_or_name(key_id_or_name)
                .await?;
            let key = match matches.len() {
                0 | 1 => {
                    self.session_store
                        .resolve_key_by_id_or_name(key_id_or_name)
                        .await?
                }
                _ => {
                    use std::io::IsTerminal;
                    if !std::io::stderr().is_terminal() {
                        self.session_store
                            .resolve_key_by_id_or_name(key_id_or_name)
                            .await?
                    } else {
                        eprintln!(
                            "{} Multiple keys match {}:",
                            crate::style::yellow("Note:"),
                            crate::style::cyan(key_id_or_name)
                        );
                        let prompt = format!("Select key '{}'", key_id_or_name);
                        match prompt_pick_key_without_activation(&matches, &[], &prompt, 0)? {
                            Some(key) => key,
                            None => anyhow::bail!("Cancelled."),
                        }
                    }
                }
            };
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        // Try last selection
        if let Some(sel) = last_sel
            && let Some(key) = self.session_store.get_key_by_id(&sel.key_id).await?
        {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        // Fallback to active key
        if let Some(key) = self.session_store.get_active_key().await? {
            return Ok(Resolved {
                value: key,
                interactive: false,
            });
        }

        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => anyhow::bail!("No API key configured. Run 'aivo keys add' first."),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => match prompt_pick_key_without_activation(&keys, &[], "Select key", 0)? {
                Some(key) => Ok(Resolved {
                    value: key,
                    interactive: true,
                }),
                None => Err(anyhow::anyhow!("Cancelled")),
            },
        }
    }

    async fn prompt_select_key(
        &self,
        last_sel: Option<&LastSelection>,
    ) -> Result<Resolved<ApiKey>> {
        let keys = self.session_store.get_keys().await?;
        match keys.len() {
            0 => anyhow::bail!("No API key configured. Run 'aivo keys add' first."),
            1 => {
                let mut key = keys[0].clone();
                SessionStore::decrypt_key_secret(&mut key)?;
                Ok(Resolved {
                    value: key,
                    interactive: false,
                })
            }
            _ => {
                let active_key_id = self
                    .session_store
                    .get_active_key_info()
                    .await?
                    .map(|active_key| active_key.id);
                let default_idx = last_sel
                    .and_then(|sel| keys.iter().position(|key| key.id == sel.key_id))
                    .or_else(|| {
                        active_key_id
                            .as_ref()
                            .and_then(|active_id| keys.iter().position(|key| &key.id == active_id))
                    })
                    .unwrap_or(0);
                match prompt_pick_key_without_activation(&keys, &[], "Select key", default_idx)? {
                    Some(key) => Ok(Resolved {
                        value: key,
                        interactive: true,
                    }),
                    None => Err(anyhow::anyhow!("Cancelled")),
                }
            }
        }
    }

    fn resolve_tool(
        &self,
        tool_arg: Option<&str>,
        last_sel: Option<&LastSelection>,
    ) -> Result<Resolved<AIToolType>> {
        if let Some(tool) = tool_arg {
            return Ok(Resolved {
                value: AIToolType::parse(tool)
                    .ok_or_else(|| anyhow::anyhow!("Unknown AI tool '{}'", tool))?,
                interactive: false,
            });
        }

        if let Some(sel) = last_sel
            && let Some(tool) = AIToolType::parse(&sel.tool)
        {
            return Ok(Resolved {
                value: tool,
                interactive: false,
            });
        }

        let tools = AIToolType::all();
        let items = tools
            .iter()
            .map(|t| t.as_str().to_string())
            .collect::<Vec<_>>();
        let selected = FuzzySelect::new()
            .with_prompt("Select tool")
            .items(&items)
            .default(0)
            .interact_opt()
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("Cancelled"))?;
        Ok(Resolved {
            value: tools[selected],
            interactive: true,
        })
    }

    async fn resolve_model(
        &self,
        model_arg: Option<String>,
        last_sel: Option<&LastSelection>,
        key: &Resolved<ApiKey>,
        refresh: bool,
        tool: AIToolType,
    ) -> Result<Resolved<Option<String>>> {
        // Only use last_sel model when the key matches
        let matching_sel = last_sel.filter(|sel| sel.key_id == key.value.id);
        let explicit_picker = model_arg.as_ref().is_some_and(|value| value.is_empty());
        let should_prompt = explicit_picker || (model_arg.is_none() && matching_sel.is_none());

        if should_prompt {
            return self
                .prompt_select_model(&key.value, refresh, tool, explicit_picker)
                .await;
        }

        match model_arg {
            Some(value) => Ok(Resolved {
                value: Some(value),
                interactive: false,
            }),
            None => Ok(Resolved {
                value: matching_sel.and_then(|sel| sel.model.clone()),
                interactive: false,
            }),
        }
    }

    async fn prompt_select_model(
        &self,
        key: &ApiKey,
        refresh: bool,
        tool: AIToolType,
        explicit_picker: bool,
    ) -> Result<Resolved<Option<String>>> {
        let client = http_utils::router_http_client();
        let models = if refresh {
            crate::commands::models::fetch_models_cached(&client, key, &self.cache, true)
                .await
                .unwrap_or_default()
        } else {
            fetch_models_for_select(&client, key, &self.cache).await
        };
        if models.is_empty() {
            // No fetchable model list (common for providers without a public
            // /v1/models endpoint — e.g. Codex ChatGPT OAuth). Skip the
            // picker and let the tool use its own default rather than
            // blocking the launch. Only explain this when the user
            // explicitly asked for a picker; on the implicit "no prior
            // selection" path the launch just proceeds silently.
            if explicit_picker {
                eprintln!(
                    "  {} {}",
                    style::dim("note:"),
                    crate::commands::NO_MODEL_LIST_HINT
                );
            }
            return Ok(Resolved {
                value: None,
                interactive: false,
            });
        }

        match prompt_model_picker(models, Some(tool)) {
            Some(selected) => Ok(Resolved {
                value: Some(selected),
                interactive: true,
            }),
            None => Err(anyhow::anyhow!("Cancelled")),
        }
    }
}

fn confirm(prompt: &str) -> std::io::Result<bool> {
    let term = Term::stdout();
    term.write_str(&format!("{prompt} [Y/n] "))?;

    loop {
        match term.read_key()? {
            Key::Enter | Key::Char('y') | Key::Char('Y') => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Running..."))?;
                return Ok(true);
            }
            Key::Char('n') | Key::Char('N') | Key::Escape => {
                term.write_str("\r\x1b[2K")?;
                term.write_line(&style::dim("Cancelled."))?;
                return Ok(false);
            }
            _ => {}
        }
    }
}
