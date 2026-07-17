use super::super::event_loop_impl::{
    EscReassembly, EscStep, FragStep, osc_reply_frag_step, parse_sgr_scroll, sgr_mouse_frag_step,
};
use super::super::*;
use super::helpers::*;

#[test]
fn test_sgr_mouse_frag_step_classifies_fragment() {
    // Valid growing prefixes of `[<{params}`.
    assert!(matches!(sgr_mouse_frag_step("["), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<"), FragStep::Continue));
    assert!(matches!(sgr_mouse_frag_step("[<6"), FragStep::Continue));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23"),
        FragStep::Continue
    ));
    // Complete reports (press `M` / release `m`).
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23M"),
        FragStep::Final
    ));
    assert!(matches!(
        sgr_mouse_frag_step("[<64;56;23m"),
        FragStep::Final
    ));
    // Not SGR mouse: `[` not followed by `<`, empty params, stray char.
    assert!(matches!(sgr_mouse_frag_step("[h"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<M"), FragStep::Invalid));
    assert!(matches!(sgr_mouse_frag_step("[<6x"), FragStep::Invalid));
}

#[test]
fn test_parse_sgr_scroll_only_wheel_buttons() {
    let up = parse_sgr_scroll("[<64;56;23M").unwrap();
    assert!(matches!(up.kind, MouseEventKind::ScrollUp));
    assert_eq!((up.column, up.row), (55, 22)); // SGR is 1-based, MouseEvent 0-based
    let down = parse_sgr_scroll("[<65;1;1m").unwrap();
    assert!(matches!(down.kind, MouseEventKind::ScrollDown));
    assert_eq!((down.column, down.row), (0, 0));
    assert!(parse_sgr_scroll("[<0;5;5M").is_none()); // left-button press, not a wheel
    assert!(parse_sgr_scroll("[<64;5M").is_none()); // missing the row param
    assert!(parse_sgr_scroll("[<64;5;5;5M").is_none()); // an extra param
}

// A fast mouse-wheel report (`\x1b[<65;…M`) that crossterm splits at its ESC must
// not leak its tail into the composer nor spuriously close the open overlay — the
// bare Esc is withheld, the tail swallowed, and the scroll re-synthesized.
#[tokio::test]
async fn test_split_mouse_report_is_reassembled_not_leaked() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    // The leading ESC arrives alone; it is held, not acted on.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "Esc closed overlay early"
    );

    // The tail `[<65;56;23M` (wheel-down) follows as literal chars in the burst.
    for c in "[<65;56;23M".chars() {
        let step = app
            .step_esc_reassembly(
                &mut esc,
                Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
            )
            .await
            .unwrap();
        assert!(
            matches!(step, EscStep::Consumed),
            "char {c:?} leaked through"
        );
    }

    // Nothing typed, overlay still open, and the wheel-down scrolled the body.
    assert_eq!(app.draft, "", "mouse tail leaked into composer");
    match app.overlay {
        Overlay::Help { scroll } => assert_eq!(scroll, 3, "re-synthesized scroll missing"),
        _ => panic!("help overlay was spuriously closed by the split ESC"),
    }
}

#[tokio::test]
async fn test_lone_esc_still_closes_overlay_at_burst_end() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert!(matches!(app.overlay, Overlay::Help { .. }));

    // Burst ends with the Esc still held: it was real, so flushing closes help.
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert!(matches!(app.overlay, Overlay::None));
}

#[tokio::test]
async fn test_esc_then_non_mouse_text_is_replayed_losslessly() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // No overlay: characters reach the composer.

    let mut esc = EscReassembly::Idle;
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Char('['), KeyModifiers::NONE)),
    )
    .await
    .unwrap();
    // `h` breaks the SGR-mouse shape, so the held `[` and this `h` are real text.
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert_eq!(app.draft, "[h", "non-mouse run after Esc was not replayed");
}

#[test]
fn test_osc_reply_frag_step_classifies_fragment() {
    // Valid growing prefixes of `]{10|11};{payload}`.
    assert!(matches!(osc_reply_frag_step("]"), FragStep::Continue));
    assert!(matches!(osc_reply_frag_step("]1"), FragStep::Continue));
    assert!(matches!(osc_reply_frag_step("]10;"), FragStep::Continue));
    assert!(matches!(
        osc_reply_frag_step("]11;rgb:2885/2a19/353a"),
        FragStep::Continue
    ));
    assert!(matches!(
        osc_reply_frag_step("]10;#ffffff"),
        FragStep::Continue
    ));
    // Not a color reply: wrong code, payload char outside the reply charset.
    assert!(matches!(osc_reply_frag_step("]2"), FragStep::Invalid));
    assert!(matches!(osc_reply_frag_step("]10x"), FragStep::Invalid));
    assert!(matches!(osc_reply_frag_step("]10; "), FragStep::Invalid));
    // Runaway length cap.
    let long = format!("]10;{}", "a".repeat(70));
    assert!(matches!(osc_reply_frag_step(&long), FragStep::Invalid));
}

// OSC 10/11 color replies a terminal delivers late (their querier gone, slow
// SSH) surface as Alt+`]` plus literal chars; both must be swallowed, not typed.
#[tokio::test]
async fn test_leaked_osc_color_reports_are_swallowed() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let mut esc = EscReassembly::Idle;
    for body in ["10;rgb:eb8b/eb8b/eb8b", "11;rgb:2885/2a19/353a"] {
        // `ESC ]` coalesced in one read → Alt+`]`.
        let step = app
            .step_esc_reassembly(
                &mut esc,
                Event::Key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::ALT)),
            )
            .await
            .unwrap();
        assert!(matches!(step, EscStep::Consumed));
        for c in body.chars() {
            let step = app
                .step_esc_reassembly(
                    &mut esc,
                    Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
                )
                .await
                .unwrap();
            assert!(
                matches!(step, EscStep::Consumed),
                "char {c:?} leaked through"
            );
        }
        // The ST terminator `ESC \` coalesced → Alt+`\`.
        let step = app
            .step_esc_reassembly(
                &mut esc,
                Event::Key(KeyEvent::new(KeyCode::Char('\\'), KeyModifiers::ALT)),
            )
            .await
            .unwrap();
        assert!(matches!(step, EscStep::Consumed));
    }
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert_eq!(app.draft, "", "OSC color reply leaked into composer");
}

// The same reply with its ESCs split into separate events must also be
// swallowed, without acting on either Esc.
#[tokio::test]
async fn test_split_esc_osc_reply_swallowed() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.open_help_overlay();

    let mut esc = EscReassembly::Idle;
    let mut feed = Vec::new();
    feed.push(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    feed.extend(
        "]11;rgb:1111/2222/3333"
            .chars()
            .map(|c| KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
    );
    feed.push(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
    feed.push(KeyEvent::new(KeyCode::Char('\\'), KeyModifiers::NONE));
    for key in feed {
        let step = app
            .step_esc_reassembly(&mut esc, Event::Key(key))
            .await
            .unwrap();
        assert!(matches!(step, EscStep::Consumed));
    }
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert_eq!(app.draft, "", "split OSC reply leaked into composer");
    assert!(
        matches!(app.overlay, Overlay::Help { .. }),
        "a swallowed report's Esc closed the overlay"
    );
}

// A run that breaks the reply grammar is real input and must replay as text.
#[tokio::test]
async fn test_alt_bracket_non_osc_replays_as_text() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let mut esc = EscReassembly::Idle;
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::ALT)),
    )
    .await
    .unwrap();
    let step = app
        .step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    assert!(matches!(step, EscStep::Consumed));
    assert_eq!(app.draft, "]x", "broken-grammar run was not replayed");
}

// A confirmed reply prefix cut off at burst end is dropped at flush, not typed.
#[tokio::test]
async fn test_truncated_osc_reply_dropped_at_flush() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);

    let mut esc = EscReassembly::Idle;
    app.step_esc_reassembly(
        &mut esc,
        Event::Key(KeyEvent::new(KeyCode::Char(']'), KeyModifiers::ALT)),
    )
    .await
    .unwrap();
    for c in "10;rgb:eb".chars() {
        app.step_esc_reassembly(
            &mut esc,
            Event::Key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)),
        )
        .await
        .unwrap();
    }
    assert!(!app.flush_esc_reassembly(esc).await.unwrap());
    assert_eq!(app.draft, "", "truncated reply leaked at flush");
}
