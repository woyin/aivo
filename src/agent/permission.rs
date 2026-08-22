//! The one permission ladder: every gated action is a [`PermissionAction`] fed to
//! [`AgentEngine::resolve_permission`], which owns the auto-approve waivers, grant
//! lookup/remember, and the once-only hard floor. Call sites decide *which* gate
//! applies (and in what order); this module decides *how* a gate resolves.

use serde_json::Value;

use crate::agent::engine::{AgentEngine, AgentUi, TurnCtx};
use crate::agent::protocol::Decision;
use crate::agent::tools;

/// One gated action. Tiers differ in what bypasses the prompt and what an
/// "always allow" is remembered as.
pub(crate) enum PermissionAction<'a> {
    /// Hard floor (catastrophic command, plan-mode bash, protected-path escalation):
    /// force-ask every call; Allow and AlwaysAllow both run it once, nothing is
    /// bypassed and nothing remembered.
    Once {
        ask_name: &'a str,
        preview: Option<String>,
    },
    /// Remote mutation: only auto-approve MODE waives it (checked by the caller
    /// when classifying); grant = exact tool call ∨ every command family;
    /// AlwaysAllow remembers the families (exact call when none parse).
    Remote {
        name: &'a str,
        args: &'a Value,
        families: &'a [String],
    },
    /// Risky call (destructive / blind overwrite / secret read / untrusted tool):
    /// `-y`/auto-approve bypass, tool-call grant, AlwaysAllow remembers.
    Confirm { name: &'a str, args: &'a Value },
    /// Bespoke escalation (sandbox / out-of-workspace write): `-y`/auto-approve
    /// bypass, session-scoped exact key.
    Escalated {
        ask_name: &'a str,
        key: String,
        preview: String,
    },
}

/// How a gate resolved. `Covered` ran without a prompt, so a caller stacking
/// gates (remote → confirm) knows to keep going; `Approved` consumed the one
/// prompt a call gets.
#[derive(PartialEq)]
pub(crate) enum Resolution {
    Covered,
    Approved,
    Denied,
}

impl Resolution {
    pub(crate) fn allowed(&self) -> bool {
        *self != Resolution::Denied
    }
}

impl AgentEngine {
    pub(crate) async fn resolve_permission(
        &mut self,
        ctx: &TurnCtx<'_>,
        ui: &mut dyn AgentUi,
        action: PermissionAction<'_>,
    ) -> Resolution {
        match action {
            PermissionAction::Once { ask_name, preview } => {
                match ui.ask_permission(ask_name, preview.as_deref(), true).await {
                    Decision::Deny => Resolution::Denied,
                    _ => Resolution::Approved,
                }
            }
            PermissionAction::Remote {
                name,
                args,
                families,
            } => {
                if self.grants.covers(name, args, ctx.cwd) || self.grants.covers_remote(families) {
                    return Resolution::Covered;
                }
                let preview = tools::preview(name, args);
                match ui.ask_permission(name, preview.as_deref(), false).await {
                    Decision::Allow => Resolution::Approved,
                    Decision::AlwaysAllow => {
                        if families.is_empty() {
                            self.grants.remember(name, args, ctx.cwd);
                        } else {
                            self.grants.remember_remote(families);
                        }
                        Resolution::Approved
                    }
                    Decision::Deny => Resolution::Denied,
                }
            }
            PermissionAction::Confirm { name, args } => {
                if ctx.auto_approve_enabled() || self.grants.covers(name, args, ctx.cwd) {
                    return Resolution::Covered;
                }
                let preview = tools::preview(name, args);
                match ui.ask_permission(name, preview.as_deref(), false).await {
                    Decision::Allow => Resolution::Approved,
                    Decision::AlwaysAllow => {
                        self.grants.remember(name, args, ctx.cwd);
                        Resolution::Approved
                    }
                    Decision::Deny => Resolution::Denied,
                }
            }
            PermissionAction::Escalated {
                ask_name,
                key,
                preview,
            } => {
                if ctx.auto_approve_enabled() || self.grants.covers_key(&key) {
                    return Resolution::Covered;
                }
                match ui.ask_permission(ask_name, Some(&preview), false).await {
                    Decision::Allow => Resolution::Approved,
                    Decision::AlwaysAllow => {
                        self.grants.remember_key(key);
                        Resolution::Approved
                    }
                    Decision::Deny => Resolution::Denied,
                }
            }
        }
    }
}

/// The session key an approved escalation is remembered under — one constructor
/// (NUL separator, cf. `grant_store::exact_key`) so the two escalation sites
/// can't drift apart in spelling.
pub(crate) fn escalation_key(ask_name: &str, detail: &str) -> String {
    format!("{ask_name}\u{0}{detail}")
}
