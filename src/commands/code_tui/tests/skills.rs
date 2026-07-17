use super::super::*;
use super::helpers::*;

#[test]
fn test_parse_slash_skills() {
    assert_eq!(
        parse_slash_command("skills").unwrap(),
        SlashCommand::Skills(None)
    );
    assert_eq!(
        parse_slash_command("skills add fs Helper").unwrap(),
        SlashCommand::Skills(Some("add fs Helper".to_string()))
    );
    // `/skills` is advertised in the command menu + help listing.
    assert!(SLASH_COMMANDS.iter().any(|c| c.name == "skills"));
}

#[test]
fn test_filter_skill_commands_ranks_prefix_before_fuzzy() {
    let commands = vec![
        skill_command("repo-study", "Study a repo"),
        skill_command("review", "Review a PR"),
        skill_command("deep-research", "Research a topic"),
    ];
    // Empty query returns everything, order preserved.
    assert_eq!(filter_skill_commands(&commands, "").len(), 3);
    // Prefix match wins; the fuzzy `re…h` (deep-research) still shows, but after.
    let names: Vec<String> = filter_skill_commands(&commands, "re")
        .into_iter()
        .map(|c| c.name)
        .collect();
    assert_eq!(names.first().map(String::as_str), Some("repo-study"));
    assert!(names.contains(&"review".to_string()));
    // A query matching nothing yields nothing.
    assert!(filter_skill_commands(&commands, "zzz").is_empty());
}

#[test]
fn test_resolve_slash_command_skill_vs_builtin_vs_typo() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![skill_command("repo-study", "Study a repo")];

    // A discovered skill resolves to the Skill variant, with trailing args.
    assert_eq!(
        app.resolve_slash_command("repo-study https://x/y").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: Some("https://x/y".to_string()),
        }
    );
    // No args → None.
    assert_eq!(
        app.resolve_slash_command("repo-study").unwrap(),
        SlashCommand::Skill {
            name: "repo-study".to_string(),
            argument: None,
        }
    );
    // A built-in always wins, even if a same-named skill exists.
    app.skill_commands.push(skill_command("model", "shadow"));
    assert_eq!(
        app.resolve_slash_command("model gpt").unwrap(),
        SlashCommand::Model(Some("gpt".to_string()))
    );
    // An unknown name (not a built-in, not a skill) still errors.
    let err = app.resolve_slash_command("nope").unwrap_err().to_string();
    assert!(err.contains("Unknown command"), "{err}");
}

#[test]
fn test_matching_command_entries_includes_skills_after_builtins() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.skill_commands = vec![
        skill_command("repo-study", "Study a repo"),
        // A skill colliding with the built-in `/model` must be dropped.
        skill_command("model", "shadow"),
    ];
    // No built-in starts with "repo", so the skill is the sole match.
    let entries = app.matching_command_entries("repo");
    let labels: Vec<String> = entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(labels, vec!["/repo-study".to_string()]);

    // `/model` resolves to the built-in only — the colliding skill never appears.
    let model_entries = app.matching_command_entries("model");
    let model_labels: Vec<String> = model_entries.iter().map(ComposerMenuEntry::label).collect();
    assert_eq!(
        model_labels.iter().filter(|l| *l == "/model").count(),
        1,
        "a skill must not duplicate or shadow a built-in command"
    );
}

#[tokio::test]
async fn test_refresh_skill_commands_discovers_and_respects_disabled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A project skill under the app's working dir, with a unique name so real
    // home-dir skills can't collide with the assertions.
    let proj = std::env::temp_dir().join(format!(
        "aivo-skillcmd-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let name = "zz-unique-skill-cmd";
    let dir = proj.join(".aivo").join("skills").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: A unique test skill.\n---\nBody.\n"),
    )
    .unwrap();
    app.real_cwd = proj.to_string_lossy().into_owned();

    app.refresh_skill_commands().await;
    let found = app.skill_commands.iter().find(|c| c.name == name);
    assert!(found.is_some(), "discovered skill should become a command");
    assert_eq!(found.unwrap().description, "A unique test skill.");

    // Disabling it in `/skills` drops it from the command set.
    app.session_store
        .set_skill_enabled(name, false)
        .await
        .unwrap();
    app.refresh_skill_commands().await;
    assert!(
        !app.skill_commands.iter().any(|c| c.name == name),
        "a disabled skill must not be offered as a command"
    );

    let _ = std::fs::remove_dir_all(&proj);
}

#[test]
fn test_expand_skill_invocation_arguments_and_fallback() {
    use crate::agent::skills::Skill;
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // `$ARGUMENTS` is substituted in place.
    assert_eq!(
        super::super::runtime_impl::expand_skill_invocation(&placeholder, Some("the repo")),
        "Study the repo now."
    );

    let plain = Skill {
        name: "repo-study".to_string(),
        description: "d".to_string(),
        body: "Follow these steps.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    // No placeholder + args → directive + body + appended input.
    let out = super::super::runtime_impl::expand_skill_invocation(&plain, Some("https://x/y"));
    assert!(out.contains("Use the \"repo-study\" skill"), "{out}");
    assert!(out.contains("Follow these steps."), "{out}");
    assert!(out.ends_with("Input: https://x/y"), "{out}");
    // No placeholder, no args → no trailing input line.
    let bare = super::super::runtime_impl::expand_skill_invocation(&plain, None);
    assert!(bare.contains("Follow these steps."));
    assert!(!bare.contains("Input:"));
}

/// The display/log recognizer recovers the compact `/name args` the user typed
/// from an expanded invocation, round-tripping with the producer, and declines
/// ordinary messages and `$ARGUMENTS`-style skills (which leave no wrapper).
#[test]
fn test_skill_invocation_label_recovers_typed_command() {
    use super::super::runtime_impl::{expand_skill_invocation, skill_invocation_label};
    use crate::agent::skills::Skill;
    let skill = Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "Search Baidu.\n\nUse the bundled script.".to_string(),
        dir: std::path::PathBuf::new(),
    };

    // With args (incl. CJK) → `/name args`.
    let expanded = expand_skill_invocation(&skill, Some("歌曲"));
    assert_eq!(
        skill_invocation_label(&expanded).as_deref(),
        Some("/baidu-search 歌曲")
    );
    // No args → bare `/name`.
    let bare = expand_skill_invocation(&skill, None);
    assert_eq!(
        skill_invocation_label(&bare).as_deref(),
        Some("/baidu-search")
    );

    // A body that itself contains an `Input:`-style line must not be mistaken for
    // args when none were passed (multi-line tail ⇒ no args).
    let trappy = Skill {
        name: "trap".to_string(),
        description: "d".to_string(),
        body: "Step.\n\nInput: foo\nmore body".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&trappy, None)).as_deref(),
        Some("/trap")
    );

    // Ordinary user text and a `$ARGUMENTS` skill (no wrapper) → None.
    assert_eq!(skill_invocation_label("just a normal question"), None);
    let placeholder = Skill {
        name: "echo".to_string(),
        description: "d".to_string(),
        body: "Study $ARGUMENTS now.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    assert_eq!(
        skill_invocation_label(&expand_skill_invocation(&placeholder, Some("x"))),
        None
    );
}

/// The transcript shows the compact `/name args` for a skill turn, not the whole
/// inlined SKILL.md body (the verbosity reported in the field).
#[tokio::test]
async fn test_skill_turn_renders_compact_not_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let skill = crate::agent::skills::Skill {
        name: "baidu-search".to_string(),
        description: "d".to_string(),
        body: "SEARCH_BAIDU_BODY_MARKER do the thing.".to_string(),
        dir: std::path::PathBuf::new(),
    };
    let expanded = super::super::runtime_impl::expand_skill_invocation(&skill, Some("歌曲"));
    app.history.push(ChatMessage {
        model: None,
        role: "user".to_string(),
        content: expanded,
        reasoning_content: None,
        attachments: vec![],
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The compact command shows (the test backend pads wide CJK glyphs with an
    // inter-cell space, so assert on the ASCII command + each arg glyph rather
    // than the exact contiguous string).
    assert!(
        screen.contains("/baidu-search"),
        "compact label missing:\n{screen}"
    );
    assert!(
        screen.contains('歌') && screen.contains('曲'),
        "skill arg missing from the turn:\n{screen}"
    );
    assert!(
        !screen.contains("SEARCH_BAIDU_BODY_MARKER"),
        "the inlined body leaked into the transcript:\n{screen}"
    );
}

#[test]
fn test_skills_overlay_renders_toggle_list() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("Skills"), "missing title:\n{screen}");
    assert!(screen.contains("brandkit"), "missing skill name:\n{screen}");
    assert!(
        screen.contains("[✓]"),
        "missing enabled checkbox:\n{screen}"
    );
    assert!(
        screen.contains("[ ]"),
        "missing disabled checkbox:\n{screen}"
    );
    // The name and its description render on separate lines.
    assert!(
        screen.contains("Premium brand-kit"),
        "missing description line:\n{screen}"
    );
    // The on-count badge sits in the top border (1 of 2 on).
    assert!(screen.contains("1/2 on"), "missing count:\n{screen}");
    // Search placeholder up top, controls along the footer.
    assert!(
        screen.contains("filter skills") && screen.contains("toggle"),
        "missing controls:\n{screen}"
    );
}

/// The detail line shows the selected skill's location, and tags a project
/// skill (which `d` can't delete). Selecting the project-scoped row surfaces
/// the `project` marker that appears nowhere else on screen.
#[test]
fn test_skills_overlay_detail_line_shows_scope_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.selected = 1; // "critique", a project skill
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("project"),
        "detail line should tag a project-scoped skill:\n{screen}"
    );
    assert!(
        screen.contains("skills/critique"),
        "detail line should show the skill's path:\n{screen}"
    );
}

#[tokio::test]
async fn test_toggle_skill_persists_and_resets_engine() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Toggle "brandkit" (index 0) from enabled → disabled.
    app.toggle_skill(0).await.unwrap();

    if let Overlay::Skills(state) = &app.overlay {
        assert!(!state.items[0].enabled, "in-overlay state did not flip");
    } else {
        panic!("skills overlay vanished");
    }
    // Persisted to the store, and the engine is dropped so the next turn rebuilds.
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert_eq!(disabled, vec!["brandkit".to_string()]);
    assert!(app.agent_engine.is_none(), "engine not reset after toggle");

    // Toggling back removes it from the disabled set (idempotent enable).
    app.toggle_skill(0).await.unwrap();
    assert!(
        app.session_store
            .get_disabled_skills()
            .await
            .unwrap()
            .is_empty()
    );
}

/// A name hit must outrank rows whose long description merely subsequence-
/// matches, and typing re-anchors the selection to that top hit.
#[test]
fn test_skills_filter_ranks_name_matches_first() {
    use crate::agent::skills::SkillScope;
    let skill = |name: &str, description: &str| SkillToggle {
        name: name.to_string(),
        description: description.to_string(),
        enabled: false,
        dir: std::path::PathBuf::from("/tmp/x"),
        scope: SkillScope::User,
        body: String::new(),
    };
    let mut overlay = SkillsOverlay {
        // "big old dear" contains b-o-l-d-e-r as a subsequence.
        items: vec![
            skill("alpha", "big old dear"),
            skill("bolder", "amplify designs"),
        ],
        selected: 0,
        query: String::new(),
        adding: None,
        pending_delete: None,
        viewing: None,
        detail_scroll: 0,
    };

    overlay.query = "bolder".to_string();
    overlay.refilter();

    assert_eq!(
        overlay.filtered_indices(),
        vec![1, 0],
        "name match must rank above a description-only match"
    );
    assert_eq!(overlay.selected, 1, "typing re-anchors to the top hit");
}

/// Space is a dead key in a fuzzy filter, so it toggles instead of typing.
#[tokio::test]
async fn test_space_toggles_without_entering_filter() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(s) = &app.overlay {
        assert!(
            !s.items[0].enabled,
            "Space should toggle the selected skill"
        );
        assert!(s.query.is_empty(), "Space must never enter the filter");
    } else {
        panic!("skills overlay vanished");
    }

    // With no visible selection (filter matches nothing) Space is a no-op.
    if let Overlay::Skills(s) = &mut app.overlay {
        s.query = "zzz".to_string();
        s.refilter();
    }
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(s) = &app.overlay {
        assert_eq!(s.query, "zzz");
    }

    app.overlay = Overlay::Mcp(mcp_overlay_fixture());
    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Mcp(s) = &app.overlay {
        assert!(
            !s.items[0].enabled,
            "Space should toggle the selected server"
        );
        assert!(s.query.is_empty());
    } else {
        panic!("mcp overlay vanished");
    }
}

/// Discovery leaves `Skill::body` empty (lazy) and the advert truncates the
/// description — the overlay must load/keep both in full for the detail pane.
#[tokio::test]
async fn test_open_skills_overlay_loads_full_body_and_description() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let dir = std::env::temp_dir().join(format!("aivo-skill-detail-{}", std::process::id()));
    let skill_dir = dir.join(".agents").join("skills").join("fulltext");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: fulltext\ndescription: First sentence. Second sentence the advert would drop.\n---\nLine one of instructions.\nLine two of instructions.\n",
    )
    .unwrap();
    app.real_cwd = dir.to_string_lossy().into_owned();

    app.open_skills_overlay().await.unwrap();
    let Overlay::Skills(state) = &app.overlay else {
        panic!("skills overlay did not open");
    };
    let item = state
        .items
        .iter()
        .find(|i| i.name == "fulltext")
        .expect("skill discovered");
    assert!(
        item.description.contains("Second sentence"),
        "full description should survive into the overlay: {}",
        item.description
    );
    assert!(
        item.body.contains("Line two of instructions"),
        "SKILL.md body should be read at open time: {:?}",
        item.body
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_parse_skill_add_input() {
    use super::super::session_impl::parse_skill_add_input;
    // First token is the name; the rest is a free-text description.
    let (name, desc) = parse_skill_add_input("changelog Summarize the git log").unwrap();
    assert_eq!(name, "changelog");
    assert_eq!(desc, "Summarize the git log");
    // A bare name (no description) is fine — a placeholder is templated in.
    assert_eq!(
        parse_skill_add_input("solo").unwrap(),
        ("solo".to_string(), String::new())
    );
    // The first token is the name (so a name can't contain a space) and the
    // remainder is the description.
    let (name, desc) = parse_skill_add_input("multi word description here").unwrap();
    assert_eq!(name, "multi");
    assert_eq!(desc, "word description here");
    // A folder-unsafe single-token name is rejected.
    assert!(parse_skill_add_input("a.b desc").is_err(), "dot in name");
    assert!(parse_skill_add_input("").is_err(), "empty");
}

#[test]
fn test_skill_add_success_notice_includes_advert_and_warnings() {
    use super::super::session_impl::skill_add_success_notice;
    let path = std::path::Path::new("/tmp/aivo-test/skills/deploy/SKILL.md");

    let notice = skill_add_success_notice("deploy", "Deploy safely", path);
    assert!(notice.contains("Created skill `deploy`"), "{notice}");
    assert!(notice.contains("Advert: Deploy safely"), "{notice}");
    assert!(!notice.contains("Warning:"), "{notice}");

    let multi = skill_add_success_notice(
        "deploy",
        "Deploy safely. Use when release or rollback cues appear.",
        path,
    );
    assert!(multi.contains("Advert: Deploy safely."), "{multi}");
    assert!(
        multi.contains("only first sentence is advertised"),
        "{multi}"
    );

    let blank = skill_add_success_notice("deploy", "", path);
    assert!(blank.contains("Advert: One-line summary"), "{blank}");
    assert!(blank.contains("replace placeholder description"), "{blank}");
}

#[tokio::test]
async fn test_skills_command_dispatch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.run_skills_command(None).await.unwrap();
    assert!(
        matches!(app.overlay, Overlay::Skills(_)),
        "bare /skills opens overlay"
    );

    // An unknown verb is a usage error.
    app.run_skills_command(Some("frobnicate".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` with no name is a usage error.
    app.run_skills_command(Some("rm".to_string()))
        .await
        .unwrap();
    assert!(app.notice.as_ref().unwrap().1.contains("Usage"));

    // `rm` of a non-existent skill → "No skill" notice, no deletion.
    app.run_skills_command(Some("rm __aivo_no_such_skill__".to_string()))
        .await
        .unwrap();
    assert!(
        app.notice.as_ref().unwrap().1.contains("No skill"),
        "notice: {}",
        app.notice.as_ref().unwrap().1
    );
}

/// `/create-skill` is a first-class built-in command (in `SLASH_COMMANDS` and
/// `/help`), parses with an optional intent argument, and dispatches the embedded
/// create-skill instructions as a turn — shown compactly in the transcript.
#[tokio::test]
async fn test_create_skill_is_a_builtin_command() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    // It's registered as a built-in command, so `/help` and the `/` menu list it.
    assert!(
        SLASH_COMMANDS.iter().any(|c| c.name == "create-skill"),
        "create-skill must be a built-in command"
    );
    // Parses bare and with an intent argument.
    assert_eq!(
        parse_slash_command("create-skill").unwrap(),
        SlashCommand::CreateSkill(None)
    );
    assert_eq!(
        parse_slash_command("create-skill a git-diff summarizer").unwrap(),
        SlashCommand::CreateSkill(Some("a git-diff summarizer".to_string()))
    );

    // Running it queues/sends the embedded instructions and shows the compact
    // command (not the inlined body) in the transcript.
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.run_create_skill_command(Some("a git-diff summarizer".to_string()))
        .await
        .unwrap();
    let user = app
        .history
        .iter()
        .find(|m| m.role == "user")
        .expect("a user turn was dispatched");
    assert!(
        user.content.contains("create-skill") && user.content.contains("git-diff summarizer"),
        "the model receives the expanded instructions + intent"
    );
    // Regression: the body documents the literal `$ARGUMENTS` token, which must
    // survive verbatim — the command must NOT run it through `$ARGUMENTS`
    // substitution and splice the user's intent into the documentation.
    assert!(
        user.content.contains("$ARGUMENTS"),
        "the `$ARGUMENTS` doc token must not be substituted away"
    );

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("/create-skill"),
        "transcript should show the compact command:\n{screen}"
    );
}

#[tokio::test]
async fn test_skills_add_routes_source_not_scaffold() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // A token that isn't a bare skill name (has `/` or `.`) routes to install-
    // from-source, not scaffold. A bad local path surfaces an install error
    // (and never scaffolds a literal `./…`-named skill).
    app.submit_skill_add("./aivo_no_such_skill_dir_zzz".to_string())
        .await
        .unwrap();
    // Install runs on a background task now; drain its `SkillInstalled` outcome.
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if app.installing_skill.is_none() && app.notice.is_some() {
            break;
        }
        tokio::task::yield_now().await;
    }
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(
        notice.contains("Failed to install") || notice.contains("not a directory"),
        "expected an install error, got: {notice}"
    );
}

/// A local two-skill install source (`skills/alpha`, `skills/beta`) in a tempdir.
fn write_skill_pack() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let pack = std::env::temp_dir().join(format!(
        "aivo-tui-skill-pack-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    for (name, desc) in [("alpha", "First skill."), ("beta", "Second skill.")] {
        let dir = pack.join("skills").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {desc}\n---\nBody of {name}.\n"),
        )
        .unwrap();
    }
    pack
}

/// Phase verb + live size readout; the size precedes the source so a clipped
/// URL never hides it.
#[test]
fn test_skill_install_progress_status_text() {
    let progress = SkillInstallProgress::new("github:o/r".to_string(), "Fetching");
    assert_eq!(progress.status_text(), "Fetching github:o/r…");
    progress
        .bytes
        .store(2_621_440, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(progress.status_text(), "Fetching (2.5MB) github:o/r…");
    let copy = SkillInstallProgress::new("github:o/r".to_string(), "Installing");
    assert_eq!(copy.status_text(), "Installing github:o/r…");
}

/// The `/skills` overlay carries the progress row; the transcript line is
/// suppressed while it does.
#[test]
fn test_skills_overlay_shows_fetch_progress_row() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let progress = SkillInstallProgress::new("github:anthropics/skills".to_string(), "Fetching");
    progress
        .bytes
        .store(1_048_576, std::sync::atomic::Ordering::Relaxed);
    app.installing_skill = Some(progress);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Fetching (1.0MB)"),
        "missing progress row with size:\n{screen}"
    );
    assert!(
        app.spinner_status_line().is_none(),
        "status line must be suppressed while /skills shows the progress row"
    );

    app.overlay = Overlay::None;
    let line = app
        .spinner_status_line()
        .expect("status line while fetching");
    let text = plain_text_from_spans(&line.line.spans);
    assert!(
        text.contains("Fetching (1.0MB) github:anthropics/skills"),
        "status line: {text:?}"
    );
}

/// Picker chrome: title, marked badge, source line, checkbox rows, footer.
#[test]
fn test_skill_install_overlay_renders_picker() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:anthropics/skills".to_string(),
        project: false,
        items: vec![
            InstallPickItem {
                name: "alpha".to_string(),
                description: "First skill.".to_string(),
                body: "Body.".to_string(),
                checked: true,
                installed: false,
            },
            InstallPickItem {
                name: "beta".to_string(),
                description: "Second skill.".to_string(),
                body: "Body.".to_string(),
                checked: false,
                installed: true,
            },
        ],
        selected: 0,
        query: String::new(),
        viewing: None,
        detail_scroll: 0,
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Install skills"),
        "missing title:\n{screen}"
    );
    assert!(
        screen.contains("from github:anthropics/skills"),
        "missing source line:\n{screen}"
    );
    assert!(
        screen.contains("into ~/.config/aivo/skills (user)"),
        "missing destination line:\n{screen}"
    );
    assert!(screen.contains("alpha"), "missing skill row:\n{screen}");
    assert!(
        screen.contains("First skill."),
        "missing description:\n{screen}"
    );
    assert!(screen.contains("1/2 marked"), "missing badge:\n{screen}");
    assert!(
        screen.contains("installed — Space to update"),
        "missing installed/update note:\n{screen}"
    );
    // Footer clips in narrow terminals; the mark/install/Esc trio comes first.
    assert!(
        screen.contains("mark") && screen.contains("install") && screen.contains("Esc"),
        "missing footer controls:\n{screen}"
    );
    assert!(
        screen.contains("Enter applies the 1 marked"),
        "missing marked-count detail:\n{screen}"
    );
}

/// The loading state narrates the fetch; Esc from it returns to the composer.
#[tokio::test]
async fn test_skill_install_loading_state_renders_and_esc_closes() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.installing_skill = Some(SkillInstallProgress::new(
        "github:anthropics/skills".to_string(),
        "Fetching",
    ));
    app.overlay = Overlay::SkillInstall(SkillInstallOverlay {
        source: "github:anthropics/skills".to_string(),
        ..Default::default()
    });

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Install skills"),
        "missing modal title:\n{screen}"
    );
    assert!(
        screen.contains("Fetching github:anthropics/skills"),
        "missing loading row:\n{screen}"
    );
    assert!(
        screen.contains("will appear here"),
        "missing loading hint:\n{screen}"
    );
    assert!(
        app.spinner_status_line().is_none(),
        "transcript line must stay quiet while the modal narrates"
    );

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        matches!(app.overlay, Overlay::None),
        "Esc on loading must close the modal, not open /skills"
    );
}

/// A mark on an installed row is an update ask.
#[test]
fn test_skill_install_picker_marks_installed_for_update() {
    let mut state = SkillInstallOverlay {
        source: "github:o/r".to_string(),
        project: false,
        items: vec![
            InstallPickItem {
                name: "fresh".to_string(),
                description: String::new(),
                body: String::new(),
                checked: false,
                installed: false,
            },
            InstallPickItem {
                name: "have".to_string(),
                description: String::new(),
                body: String::new(),
                checked: false,
                installed: true,
            },
        ],
        selected: 1,
        query: String::new(),
        viewing: None,
        detail_scroll: 0,
    };
    // Enter's fallback never implicitly updates an installed row.
    assert!(state.pick_names().is_empty());
    state.items[1].checked = true;
    assert_eq!(state.pick_names(), ["have"]);
    // Mark-all targets only the not-yet-installed rows.
    state.items[1].checked = false;
    state.toggle_all();
    assert!(state.items[0].checked && !state.items[1].checked);
}

/// Multi-skill source: loading modal at once, picker when staged, marking keys,
/// Esc discards the stage.
#[tokio::test]
async fn test_skills_install_multi_source_opens_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(pack.display().to_string())
        .await
        .unwrap();
    // Loading modal opens at once — never the installed-skills list.
    assert!(
        matches!(&app.overlay, Overlay::SkillInstall(s) if s.items.is_empty()),
        "submit must open the install modal right away, before the fetch completes"
    );
    assert!(
        app.installing_skill.is_some(),
        "progress state set from the first frame"
    );
    // The loading modal is already SkillInstall — wait for the staged items.
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if matches!(&app.overlay, Overlay::SkillInstall(s) if !s.items.is_empty()) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let Overlay::SkillInstall(state) = &app.overlay else {
        panic!(
            "multi-skill source must open the install picker: {:?}",
            app.notice
        );
    };
    assert!(
        !state.items.is_empty(),
        "picker never populated: {:?}",
        app.notice
    );
    let names: Vec<&str> = state.items.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(names, ["alpha", "beta"]);
    assert!(state.items.iter().all(|i| !i.checked), "nothing pre-marked");
    assert_eq!(state.source, pack.display().to_string());
    assert!(
        app.staged_skill_install.is_some(),
        "the fetched tree stays staged for the pick"
    );
    assert!(app.installing_skill.is_none(), "spinner cleared");
    assert_eq!(state.items[0].description, "First skill.");
    assert_eq!(state.items[0].body, "Body of alpha.");
    // Nothing marked: Enter targets the highlighted row.
    assert_eq!(state.pick_names(), ["alpha"]);

    app.handle_key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert!(state.items[0].checked, "Space marks");
        assert_eq!(state.pick_names(), ["alpha"]);
    } else {
        panic!("picker vanished on Space");
    }
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    if let Overlay::SkillInstall(state) = &app.overlay {
        assert!(
            state.items.iter().all(|i| i.checked),
            "Ctrl+A marks all when any is unmarked"
        );
        assert_eq!(state.pick_names(), ["alpha", "beta"]);
    } else {
        panic!("picker vanished on Ctrl+A");
    }

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        app.staged_skill_install.is_none(),
        "Esc must discard the staged tree"
    );
    assert!(
        matches!(app.overlay, Overlay::Skills(_)),
        "Esc falls back to the /skills overlay"
    );
    // A local source is never deleted by the discard.
    assert!(pack.join("skills/alpha/SKILL.md").is_file());
    let _ = std::fs::remove_dir_all(&pack);
}

/// `-p` rides the whole install pipeline: flag parse → staged pick → overlay
/// marked as a project install (nothing is written until names are picked).
#[tokio::test]
async fn test_skills_install_project_flag_reaches_picker() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(format!("-p {}", pack.display()))
        .await
        .unwrap();
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if matches!(&app.overlay, Overlay::SkillInstall(s) if !s.items.is_empty()) {
            break;
        }
        tokio::task::yield_now().await;
    }
    let Overlay::SkillInstall(state) = &app.overlay else {
        panic!("picker must open: {:?}", app.notice);
    };
    assert!(state.project, "picker must carry the -p destination");
    assert_eq!(state.source, pack.display().to_string(), "flag stripped");
    assert!(
        matches!(app.staged_skill_install, Some((_, true))),
        "staged pick must remember the project destination"
    );

    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    let _ = std::fs::remove_dir_all(&pack);
}

/// Notice wording per report shape.
#[test]
fn test_install_report_notice_wording() {
    use super::super::session_impl::install_report_notice;
    use crate::agent::skills::InstallReport;
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec![],
            skipped_existing: vec![],
        },
    );
    assert_eq!(msg, "Installed skill: a");
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into(), "b".into()],
            updated: vec![],
            skipped_existing: vec!["c".into()],
        },
    );
    assert_eq!(msg, "Installed skills: a, b (already installed: c)");
    let (color, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec![],
            updated: vec![],
            skipped_existing: vec!["c".into()],
        },
    );
    assert_eq!(msg, "Already installed: c");
    assert_eq!(color, WARNING());
    let (_, msg) = install_report_notice(
        "src",
        false,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec!["u".into(), "v".into()],
            skipped_existing: vec![],
        },
    );
    assert_eq!(msg, "Installed skill: a · Updated skills: u, v");
    // `-p/--project`: destination and the untrusted caveat are spelled out.
    let (_, msg) = install_report_notice(
        "src",
        true,
        &InstallReport {
            installed: vec!["a".into()],
            updated: vec![],
            skipped_existing: vec![],
        },
    );
    assert!(
        msg.starts_with("Installed skill: a → ./.agents/skills"),
        "{msg}"
    );
    assert!(msg.contains("untrusted"), "{msg}");
    let (color, _) = install_report_notice("src", false, &InstallReport::default());
    assert_eq!(color, WARNING());
}

/// A flag-like token other than `-p/--project` is rejected up front — no
/// download for the install path, no folder named `--foo` for the scaffold path.
#[tokio::test]
async fn test_skill_add_rejects_unknown_options_before_fetch() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    app.submit_skill_add("--foo github:o/r".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `--foo`")),
        "unknown leading flag must error: {:?}",
        app.notice
    );
    assert!(app.installing_skill.is_none(), "no fetch may start");

    // A bad filter used to surface only after the whole source was downloaded.
    app.submit_skill_add("github:o/r --bogus".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `--bogus`")),
        "unknown filter flag must error before the fetch: {:?}",
        app.notice
    );
    assert!(app.installing_skill.is_none(), "no fetch may start");

    // Mid-line `-p` is guided to the edges rather than silently treated as a name.
    app.submit_skill_add("github:o/r -p alpha".to_string())
        .await
        .unwrap();
    assert!(
        matches!(&app.notice, Some((_, msg)) if msg.contains("Unknown option `-p`")),
        "mid-line -p must point at the start/end rule: {:?}",
        app.notice
    );
}

/// `-p`/`--project` is recognized at either edge of the add line, and only there.
#[test]
fn test_split_project_flag() {
    use super::super::session_impl::split_project_flag;
    assert_eq!(
        split_project_flag("-p github:o/r"),
        ("github:o/r".to_string(), true)
    );
    assert_eq!(
        split_project_flag("github:o/r --project"),
        ("github:o/r".to_string(), true)
    );
    assert_eq!(split_project_flag("--project"), (String::new(), true));
    assert_eq!(
        split_project_flag("github:o/r"),
        ("github:o/r".to_string(), false)
    );
    // Mid-line `-p` belongs to the description, not the flag.
    assert_eq!(
        split_project_flag("deploy pass -p to the deploy script"),
        ("deploy pass -p to the deploy script".to_string(), false)
    );
    // `-project` (one dash) is not the flag and must survive intact.
    assert_eq!(
        split_project_flag("-project x"),
        ("-project x".to_string(), false)
    );
}

/// An unrelated open overlay is not replaced; the stage drops with a hint.
#[tokio::test]
async fn test_skills_install_pick_defers_to_open_overlay() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let pack = write_skill_pack();

    app.submit_skill_add(pack.display().to_string())
        .await
        .unwrap();
    app.overlay = Overlay::Help { scroll: 0 };
    for _ in 0..1000 {
        app.handle_runtime_events().await.unwrap();
        if app.installing_skill.is_none() && app.notice.is_some() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "an unrelated overlay is not replaced"
    );
    assert!(app.staged_skill_install.is_none(), "stage is discarded");
    let notice = &app.notice.as_ref().unwrap().1;
    assert!(notice.contains("has 2 skills"), "{notice}");
    let _ = std::fs::remove_dir_all(&pack);
}

#[tokio::test]
async fn test_skills_add_mode_key_flow() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // Ctrl+A enters add mode; typed chars accrue; Esc cancels without scaffolding.
    app.handle_key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    for c in ['f', 's'] {
        app.handle_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE))
            .await
            .unwrap();
    }
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.adding.as_deref(), Some("fs"));
    } else {
        panic!("skills overlay vanished");
    }
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.adding.is_none(), "Esc should cancel add mode");
    } else {
        panic!("Esc in add mode must not close the overlay");
    }
}

#[tokio::test]
async fn test_skills_delete_arms_then_cancels() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // First Ctrl+D arms the delete (no removal yet); the overlay stays open.
    app.handle_key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.pending_delete, Some(0), "first Ctrl+D should arm");
    } else {
        panic!("skills overlay vanished on arm");
    }
    // Esc cancels the arm rather than closing the overlay.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.pending_delete.is_none(), "Esc should disarm");
    } else {
        panic!("Esc on an armed delete must not close the overlay");
    }
}

#[test]
fn test_sort_skill_rows_enabled_first() {
    use super::super::session_impl::sort_skill_rows;
    use crate::agent::skills::SkillScope;
    let row = |name: &str, enabled| SkillToggle {
        name: name.to_string(),
        description: String::new(),
        enabled,
        dir: std::path::PathBuf::from(name),
        scope: SkillScope::User,
        body: String::new(),
    };
    let mut rows = vec![
        row("zoff", false),
        row("able", true),
        row("aoff", false),
        row("baker", true),
    ];
    sort_skill_rows(&mut rows);
    let order: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    // Enabled first (alphabetical), disabled at the bottom (alphabetical).
    assert_eq!(order, vec!["able", "baker", "aoff", "zoff"]);
}

#[tokio::test]
async fn test_skills_filter_narrows_and_enter_toggles() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // brandkit(on), critique(off)

    // Typing 'c' filters to "critique" (index 1); the selection re-anchors to it.
    app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert_eq!(state.query, "c");
        assert_eq!(
            state.filtered_indices(),
            vec![1],
            "only critique matches 'c'"
        );
        assert_eq!(state.selected, 1, "selection re-anchored to the match");
    } else {
        panic!("overlay vanished");
    }
    // Enter toggles the matched skill (critique off → on).
    app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE))
        .await
        .unwrap();
    let disabled = app.session_store.get_disabled_skills().await.unwrap();
    assert!(
        !disabled.contains(&"critique".to_string()),
        "Enter should have toggled critique on"
    );
    // Backspace clears the one-char filter.
    app.handle_key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE))
        .await
        .unwrap();
    if let Overlay::Skills(state) = &app.overlay {
        assert!(state.query.is_empty(), "Backspace should clear the filter");
    }
}

#[test]
fn test_skills_overlay_renders_add_field() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some("changelog".to_string());
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    // The `+` prompt marks the add field; the typed buffer and the install hint
    // both show, and the footer offers save/cancel.
    assert!(screen.contains("+ "), "missing add prompt:\n{screen}");
    assert!(
        screen.contains("changelog"),
        "missing typed input:\n{screen}"
    );
    assert!(
        screen.contains("github:owner/repo"),
        "missing add hint:\n{screen}"
    );
    assert!(
        screen.contains("Enter save"),
        "missing save footer:\n{screen}"
    );
}

#[tokio::test]
async fn test_skills_overlay_wheel_scrolls_like_arrows() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    // List mode: the wheel moves the selection (the list follows it).
    app.overlay = Overlay::Skills(skills_overlay_fixture()); // 2 items, selected 0
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 1));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0));

    // Drill-in: the wheel scrolls the body, leaving the selection put.
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 3 && s.selected == 0));
    app.handle_mouse(wheel(MouseEventKind::ScrollUp))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == 0));

    // Add-input mode: the wheel is ignored (no selection move, no scroll).
    let mut overlay = skills_overlay_fixture();
    overlay.adding = Some(String::new());
    app.overlay = Overlay::Skills(overlay);
    app.handle_mouse(wheel(MouseEventKind::ScrollDown))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 0 && s.detail_scroll == 0));
}

#[test]
fn test_skills_overlay_splits_on_wide_terminal() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);

    assert!(screen.contains("filter skills"), "missing list:\n{screen}");
    assert!(
        screen.contains("Instructions:") && screen.contains("Render the boards"),
        "missing right-pane detail:\n{screen}"
    );
    assert!(
        app.overlay_detail_area.is_some(),
        "split should record the detail pane rect"
    );
}

#[test]
fn test_skills_overlay_narrow_keeps_single_pane_and_drill_in() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        !screen.contains("Instructions:"),
        "narrow list mode must not show the detail pane:\n{screen}"
    );
    assert!(app.overlay_detail_area.is_none());

    if let Overlay::Skills(state) = &mut app.overlay {
        state.viewing = Some(0);
    }
    terminal.draw(|frame| app.render(frame)).unwrap();
    let screen = test_screen(&terminal);
    assert!(
        screen.contains("Instructions:") && screen.contains("esc back"),
        "narrow drill-in should render full-modal:\n{screen}"
    );
}

#[tokio::test]
async fn test_skills_split_page_keys_scroll_detail_and_tab_disabled() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.overlay = Overlay::Skills(skills_overlay_fixture());

    // A draw arms `overlay_detail_area` (the split-active signal for keys).
    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    assert!(app.overlay_detail_area.is_some());

    app.handle_key(KeyEvent::new(KeyCode::PageDown, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(
        matches!(&app.overlay, Overlay::Skills(s) if s.detail_scroll == DETAIL_PAGE_LINES),
        "PageDown should scroll the detail pane"
    );

    app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.viewing.is_none()));

    app.handle_key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE))
        .await
        .unwrap();
    assert!(matches!(&app.overlay, Overlay::Skills(s) if s.selected == 1 && s.detail_scroll == 0));
}

#[test]
fn test_skill_drill_in_renders_body_and_path() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.viewing = Some(0); // "brandkit", body "Step 1. Render the boards."
    app.overlay = Overlay::Skills(overlay);

    let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..24u16 {
        for x in 0..80u16 {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(
        screen.contains("Instructions:"),
        "detail missing body header:\n{screen}"
    );
    assert!(
        screen.contains("Render the boards"),
        "detail missing the SKILL.md body:\n{screen}"
    );
    assert!(
        screen.contains("esc back"),
        "detail missing back hint:\n{screen}"
    );
}

/// A long SKILL.md body is scrollable in the drill-in: the top hides later lines,
/// End reveals the last line (clamped by the renderer), and Esc resets the scroll.
#[tokio::test]
async fn test_skill_drill_in_scrolls_long_body() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let mut overlay = skills_overlay_fixture();
    overlay.items[0].body = (1..=60)
        .map(|i| format!("Line number {i} of the instructions"))
        .collect::<Vec<_>>()
        .join("\n");
    overlay.viewing = Some(0);
    app.overlay = Overlay::Skills(overlay);

    let render_screen = |app: &mut CodeTuiApp| -> String {
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();
        terminal.draw(|frame| app.render(frame)).unwrap();
        let buf = terminal.backend().buffer().clone();
        let mut s = String::new();
        for y in 0..20u16 {
            for x in 0..80u16 {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    };

    let top = render_screen(&mut app);
    assert!(
        top.contains("Line number 1 of"),
        "top hides first line:\n{top}"
    );
    assert!(
        !top.contains("Line number 60 of"),
        "last line should be off-screen at the top:\n{top}"
    );
    assert!(
        top.contains("scroll"),
        "a scrollable body shows the scroll hint:\n{top}"
    );

    // End jumps to the bottom (the renderer clamps the offset).
    app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::NONE))
        .await
        .unwrap();
    let bottom = render_screen(&mut app);
    assert!(
        bottom.contains("Line number 60 of"),
        "End reveals the last line:\n{bottom}"
    );
    match &app.overlay {
        Overlay::Skills(s) => assert!(s.detail_scroll > 0, "scroll offset advanced"),
        _ => panic!("overlay vanished"),
    }

    // Esc backs out of the drill-in and resets the scroll.
    app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE))
        .await
        .unwrap();
    match &app.overlay {
        Overlay::Skills(s) => {
            assert!(s.viewing.is_none(), "Esc leaves the drill-in");
            assert_eq!(s.detail_scroll, 0, "scroll resets on back-out");
        }
        _ => panic!("overlay vanished"),
    }
}
