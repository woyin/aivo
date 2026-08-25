//! Sixel placement diff: vanished placements queue localized cell clears.

use super::super::inline_images::{PlacedImage, PreviewSlot, pin_id};
use super::super::{ChatMessage, Overlay};
use super::helpers::*;
use crate::services::terminal_graphics::{EncodedPreview, GraphicsCaps, PixelFormat, Protocol};
use ratatui::buffer::CellDiffOption;
use ratatui::layout::{Position, Rect};
use std::sync::Arc;

fn sixel_caps() -> GraphicsCaps {
    GraphicsCaps {
        protocol: Protocol::Sixel,
        tmux: true,
        cell_px: (8, 16),
    }
}

fn placement(y: u16) -> PlacedImage {
    PlacedImage {
        key: u64::from(y) + 1,
        x: 2,
        y,
        cols: 10,
        rows: 4,
    }
}

#[tokio::test]
async fn sixel_removal_queues_partial_clear_not_full_repaint() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    let kept = placement(0);
    let gone = placement(6);
    app.inline_images.placed = vec![kept, gone];
    app.inline_images.desired = vec![kept];

    let mut out = Vec::new();
    assert!(app.flush_inline_images(&mut out), "extra frame needed");
    assert!(out.is_empty(), "no escapes on the blank-image frame");
    assert!(!app.pending_full_repaint);
    assert_eq!(app.inline_images.placed, vec![kept]);
    assert_eq!(app.inline_images.pending_clears, vec![gone]);
}

#[tokio::test]
async fn sixel_move_clears_old_rect_and_forgets_placement() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    let old = placement(0);
    let moved = PlacedImage { y: 6, ..old };
    app.inline_images.placed = vec![old];
    app.inline_images.desired = vec![moved];

    let mut out = Vec::new();
    assert!(app.flush_inline_images(&mut out));
    assert!(app.inline_images.placed.is_empty());
    assert_eq!(app.inline_images.pending_clears, vec![old]);
}

#[tokio::test]
async fn mark_sixel_clear_cells_marks_rect_and_clamps() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    // Rect exceeds the 20×12 buffer: a resize between queue and mark.
    app.inline_images.pending_clears = vec![PlacedImage {
        key: 9,
        x: 18,
        y: 10,
        cols: 5,
        rows: 4,
    }];
    let mut buf = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 20, 12));
    app.mark_sixel_clear_cells(&mut buf);

    assert_eq!(
        buf[Position::new(18, 10)].diff_option,
        CellDiffOption::AlwaysUpdate
    );
    assert_eq!(
        buf[Position::new(19, 11)].diff_option,
        CellDiffOption::AlwaysUpdate
    );
    assert_eq!(buf[Position::new(17, 10)].diff_option, CellDiffOption::None);
    assert!(app.inline_images.pending_clears.is_empty(), "queue drained");
}

/// A covering surface only suppresses the placements it overlaps: the
/// bottom-anchored command menu must not blank an image at the top of the
/// screen, while a centered modal over the image still hides it.
#[tokio::test]
async fn covering_surfaces_hide_only_overlapped_placements() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    let content = "shot.png";
    app.history.push(ChatMessage {
        model: None,
        role: "assistant".to_string(),
        content: content.to_string(),
        reasoning_content: None,
        attachments: vec![],
    });
    // Filler pushes the composer (and the menu anchored to it) to the bottom,
    // clear of the image — the composer floats up under short transcripts.
    for i in 0..6 {
        app.history.push(ChatMessage {
            model: None,
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("filler {i}"),
            reasoning_content: None,
            attachments: vec![],
        });
    }
    let key = 42;
    app.inline_images
        .pinned
        .insert(pin_id(0, content, "shot.png"), key);
    app.inline_images.previews.insert(
        key,
        PreviewSlot::Ready(Arc::new(EncodedPreview {
            format: PixelFormat::Png,
            px_w: 400,
            px_h: 200,
            payload_b64: String::new(),
            thumb: None,
            content_hash: 7,
        })),
    );

    render_full_screen(&mut app, 80, 40);
    assert_eq!(app.inline_images.desired.len(), 1, "image placed when idle");
    let image_rect = app.inline_images.desired[0].rect();
    assert!(
        image_rect.y < 20,
        "test premise: image sits in the top half"
    );

    app.draft = "/".to_string();
    app.cursor = 1;
    app.sync_command_menu_state();
    render_full_screen(&mut app, 80, 40);
    assert_eq!(
        app.inline_images.desired.len(),
        1,
        "menu that doesn't touch the image must not blank it"
    );

    app.draft.clear();
    app.cursor = 0;
    app.sync_command_menu_state();
    app.overlay = Overlay::Help { scroll: 0 };
    render_full_screen(&mut app, 80, 40);
    assert!(
        app.inline_images.desired.is_empty(),
        "modal over the image suppresses its placement"
    );
}

#[tokio::test]
async fn full_repaint_subsumes_queued_sixel_clears() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    app.inline_images.placed = vec![placement(0)];
    app.inline_images.pending_clears = vec![placement(6)];

    app.note_cells_repainted();
    assert!(app.inline_images.placed.is_empty());
    assert!(app.inline_images.pending_clears.is_empty());
}
