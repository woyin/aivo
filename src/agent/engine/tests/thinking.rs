use super::super::*;

#[test]
fn default_reasoning_effort_gates_on_model_capability() {
    // Reasoning-capable models (snapshot `r` flag) get an effort to send…
    for model in ["o3", "gpt-5", "claude-sonnet-4-5", "gemini-2.5-pro"] {
        assert_eq!(
            default_reasoning_effort(model).as_deref(),
            Some("medium"),
            "model={model} should request reasoning"
        );
    }
    // …non-reasoning models and unknown ids never send it (would 400 strict providers).
    // claude-3-haiku, not 3.5-sonnet: the latter's folded key now unions to a
    // reasoning-capable upstream listing after the dash alias was dropped.
    for model in [
        "gpt-4o",
        "claude-3-haiku",
        "definitely-not-a-real-model-xyz",
    ] {
        assert_eq!(
            default_reasoning_effort(model),
            None,
            "model={model} must not request reasoning"
        );
    }
}

#[test]
fn thinking_request_tracks_capability_when_enabled() {
    // Reasoning-capable model: the level is always requested; `/effort` changes it.
    let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    assert_eq!(engine.thinking_request(), (Some("medium"), false));
    engine.set_reasoning_effort("high".into());
    assert_eq!(engine.thinking_request(), (Some("high"), false));

    // Non-reasoning model: never requested.
    let plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
    assert_eq!(plain.thinking_request(), (None, false));
}

#[test]
fn thinking_request_clamps_effort_to_catalog() {
    // A level carried across a model switch that this model's catalog doesn't
    // advertise is omitted (sending it would 400), not forwarded verbatim.
    let mut engine = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    engine.set_reasoning_effort("xhigh".into());
    engine.set_reasoning_efforts(vec!["low".into(), "medium".into(), "high".into()]);
    assert_eq!(engine.thinking_request(), (None, false));

    engine.set_reasoning_effort("high".into());
    assert_eq!(engine.thinking_request(), (Some("high"), false));

    // No catalog → nothing to clamp against; the level passes through.
    engine.set_reasoning_efforts(Vec::new());
    engine.set_reasoning_effort("custom".into());
    assert_eq!(engine.thinking_request(), (Some("custom"), false));
}

#[test]
fn thinking_request_disables_per_provider_disable_form() {
    // gpt-5 / o-series reject `"none"` alongside tools and reject `thinking` → family effort floor.
    let mut g5 = AgentEngine::new("/tmp", "gpt-5", "", &[], &[], 0, 0);
    g5.set_thinking_enabled(false);
    assert_eq!(g5.thinking_request(), (Some("minimal"), false));
    let mut o = AgentEngine::new("/tmp", "o3", "", &[], &[], 0, 0);
    o.set_thinking_enabled(false);
    assert_eq!(o.thinking_request(), (Some("low"), false));

    // A catalog that lists `none` → send it (a real effort-level off).
    let mut has_none = AgentEngine::new("/tmp", "deepseek-reasoner", "", &[], &[], 0, 0);
    has_none.set_reasoning_efforts(vec!["none".into(), "low".into(), "high".into()]);
    has_none.set_thinking_enabled(false);
    assert_eq!(has_none.thinking_request(), (Some("none"), false));

    // gpt-5.4 lists `none` but not `minimal` → catalog wins (c5d6b17 regression).
    let mut g54 = AgentEngine::new("/tmp", "gpt-5.4", "", &[], &[], 0, 0);
    g54.set_reasoning_efforts(
        ["none", "low", "medium", "high", "xhigh"]
            .map(String::from)
            .to_vec(),
    );
    g54.set_thinking_enabled(false);
    assert_eq!(g54.thinking_request(), (Some("none"), false));

    // codex advertises only low/medium/high → its `low` floor, not `minimal`.
    let mut codex = AgentEngine::new("/tmp", "gpt-5-codex", "", &[], &[], 0, 0);
    codex.set_reasoning_efforts(["low", "medium", "high"].map(String::from).to_vec());
    codex.set_thinking_enabled(false);
    assert_eq!(codex.thinking_request(), (Some("low"), false));

    // Effort scale with no off (aivo/starter, snapshot-absent): emit the `thinking` disable field, not an invalid `"none"` effort.
    let mut alias = AgentEngine::new("/tmp", "aivo/starter", "", &[], &[], 0, 0);
    assert!(
        !alias.reasoning_capable,
        "alias is absent from the snapshot"
    );
    alias.set_reasoning_efforts(vec![
        "low".into(),
        "medium".into(),
        "high".into(),
        "xhigh".into(),
        "max".into(),
    ]);
    alias.set_thinking_enabled(false);
    assert_eq!(alias.thinking_request(), (None, true));

    // Snapshot-known Anthropic model (no none/minimal): the `thinking` field, carried by the bridge.
    let mut claude = AgentEngine::new("/tmp", "claude-sonnet-4-5", "", &[], &[], 0, 0);
    claude.set_thinking_enabled(false);
    assert_eq!(claude.thinking_request(), (None, true));

    // Genuinely non-reasoning model with no catalog level: stay silent.
    let mut plain = AgentEngine::new("/tmp", "gpt-4o", "", &[], &[], 0, 0);
    plain.set_thinking_enabled(false);
    assert_eq!(plain.thinking_request(), (None, false));
}
