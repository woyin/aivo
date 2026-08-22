//! Table-driven snapshot of the unified permission ladder (`resolve_permission`):
//! tier × auto-approve × grant × decision. Guards the refactored ladder against
//! behavioral drift; the end-to-end paths live in `approvals.rs`.

use super::super::*;
use super::helpers::*;
use crate::agent::permission::{PermissionAction, Resolution, escalation_key};
use serde_json::json;

struct Case {
    name: &'static str,
    tier: Tier,
    auto_approve: bool,
    pre_granted: bool,
    // CapturingUi decision when prompted: (always_allow, deny).
    always_allow: bool,
    deny: bool,
    expect_allowed: bool,
    expect_asks: usize,
    // A re-resolve of the same action must not prompt again (a grant was remembered).
    expect_remembered: bool,
}

#[derive(Clone, Copy)]
enum Tier {
    Once,
    Remote,
    Confirm,
    Escalated,
}

fn action<'a>(
    tier: Tier,
    args: &'a serde_json::Value,
    families: &'a [String],
) -> PermissionAction<'a> {
    match tier {
        Tier::Once => PermissionAction::Once {
            ask_name: "run_bash",
            preview: None,
        },
        Tier::Remote => PermissionAction::Remote {
            name: "run_bash",
            args,
            families,
        },
        Tier::Confirm => PermissionAction::Confirm {
            name: "run_bash",
            args,
        },
        Tier::Escalated => PermissionAction::Escalated {
            ask_name: "run_bash_unsandboxed",
            key: escalation_key("run_bash_unsandboxed", "flyctl deploy --now"),
            preview: "p".into(),
        },
    }
}

#[tokio::test]
async fn permission_ladder_behavior_table() {
    let families = vec!["flyctl deploy".to_string()];
    let cases = [
        // The hard floor: never bypassed, never remembered — auto-approve still asks,
        // and AlwaysAllow does not stick.
        Case {
            name: "once/auto-approve still asks",
            tier: Tier::Once,
            auto_approve: true,
            pre_granted: false,
            always_allow: false,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: false,
        },
        Case {
            name: "once/always-allow not remembered",
            tier: Tier::Once,
            auto_approve: false,
            pre_granted: false,
            always_allow: true,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: false,
        },
        Case {
            name: "once/deny",
            tier: Tier::Once,
            auto_approve: false,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: false,
            expect_asks: 1,
            expect_remembered: false,
        },
        // Remote: -y does NOT bypass (the caller only waives it in auto-approve MODE);
        // AlwaysAllow remembers the family.
        Case {
            name: "remote/yes still asks",
            tier: Tier::Remote,
            auto_approve: true,
            pre_granted: false,
            always_allow: false,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: false,
        },
        Case {
            name: "remote/family grant covers",
            tier: Tier::Remote,
            auto_approve: false,
            pre_granted: true,
            always_allow: false,
            deny: false,
            expect_allowed: true,
            expect_asks: 0,
            expect_remembered: true,
        },
        Case {
            name: "remote/always-allow remembers family",
            tier: Tier::Remote,
            auto_approve: false,
            pre_granted: false,
            always_allow: true,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: true,
        },
        Case {
            name: "remote/deny",
            tier: Tier::Remote,
            auto_approve: false,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: false,
            expect_asks: 1,
            expect_remembered: false,
        },
        // Confirm: -y/auto bypasses (without remembering); AlwaysAllow remembers the exact call.
        Case {
            name: "confirm/auto bypasses",
            tier: Tier::Confirm,
            auto_approve: true,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: true,
            expect_asks: 0,
            expect_remembered: false,
        },
        Case {
            name: "confirm/allow once",
            tier: Tier::Confirm,
            auto_approve: false,
            pre_granted: false,
            always_allow: false,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: false,
        },
        Case {
            name: "confirm/always-allow remembers",
            tier: Tier::Confirm,
            auto_approve: false,
            pre_granted: false,
            always_allow: true,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: true,
        },
        Case {
            name: "confirm/deny",
            tier: Tier::Confirm,
            auto_approve: false,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: false,
            expect_asks: 1,
            expect_remembered: false,
        },
        // Escalated: -y/auto bypasses (without remembering); AlwaysAllow remembers the session key.
        Case {
            name: "escalated/auto bypasses",
            tier: Tier::Escalated,
            auto_approve: true,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: true,
            expect_asks: 0,
            expect_remembered: false,
        },
        Case {
            name: "escalated/always-allow remembers key",
            tier: Tier::Escalated,
            auto_approve: false,
            pre_granted: false,
            always_allow: true,
            deny: false,
            expect_allowed: true,
            expect_asks: 1,
            expect_remembered: true,
        },
        Case {
            name: "escalated/deny",
            tier: Tier::Escalated,
            auto_approve: false,
            pre_granted: false,
            always_allow: false,
            deny: true,
            expect_allowed: false,
            expect_asks: 1,
            expect_remembered: false,
        },
    ];

    let client = reqwest::Client::new();
    let dir = tmp();
    let args = json!({ "command": "flyctl deploy --now" });
    for case in cases {
        let mut engine = AgentEngine::new(&dir.display().to_string(), "m", "", &[], &[], 0, 0);
        if case.pre_granted {
            match case.tier {
                Tier::Remote => engine.grants.remember_remote(&families),
                Tier::Confirm => engine.grants.remember(
                    "run_bash",
                    &json!({"command":"flyctl deploy --now"}),
                    &dir,
                ),
                Tier::Escalated => engine.grants.remember_key(escalation_key(
                    "run_bash_unsandboxed",
                    "flyctl deploy --now",
                )),
                Tier::Once => {}
            }
        }
        let mut ui = CapturingUi {
            always_allow: case.always_allow,
            deny: case.deny,
            ..Default::default()
        };
        let ctx = TurnCtx {
            yes: case.auto_approve,
            ..turn_ctx(&client, "", &dir)
        };
        let allowed = engine
            .resolve_permission(&ctx, &mut ui, action(case.tier, &args, &families))
            .await
            .allowed();
        assert_eq!(allowed, case.expect_allowed, "[{}] allowed", case.name);
        assert_eq!(ui.asks, case.expect_asks, "[{}] asks", case.name);

        // Remembered = the same action re-resolves silently WITHOUT auto-approve.
        let mut ui2 = CapturingUi::default();
        let ctx2 = TurnCtx {
            yes: false,
            ..turn_ctx(&client, "", &dir)
        };
        let second = engine
            .resolve_permission(&ctx2, &mut ui2, action(case.tier, &args, &families))
            .await;
        let silently_covered = ui2.asks == 0 && second == Resolution::Covered;
        assert_eq!(
            silently_covered, case.expect_remembered,
            "[{}] remembered (asks on re-resolve: {})",
            case.name, ui2.asks
        );
    }
}
