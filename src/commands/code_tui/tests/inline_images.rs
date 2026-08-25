//! Sixel placement diff: vanished placements queue localized cell clears.

use super::super::inline_images::PlacedImage;
use super::helpers::*;
use crate::services::terminal_graphics::{GraphicsCaps, Protocol};
use ratatui::buffer::CellDiffOption;
use ratatui::layout::{Position, Rect};

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
