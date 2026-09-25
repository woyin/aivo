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
        grid_rows: 4,
    }
}

fn ready_preview(app: &mut super::super::CodeTuiApp, key: u64, w: u32, h: u32) {
    app.inline_images.previews.insert(
        key,
        PreviewSlot::Ready(Arc::new(EncodedPreview {
            format: PixelFormat::Rgb,
            px_w: w,
            px_h: h,
            payload_b64: String::new(),
            thumb: Some(crate::services::terminal_graphics::Thumb {
                rgb: vec![0x40; (w * h * 3) as usize],
                w,
                h,
            }),
            content_hash: key,
        })),
    );
}

/// The raster must mirror tmux's own cell maths (`sixel_size_in_cells`, 3.5a).
#[tokio::test]
async fn sixel_raster_matches_the_placement_cell_grid() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = GraphicsCaps {
        cell_px: (17, 32),
        ..sixel_caps()
    };
    let want = PlacedImage {
        key: 5,
        x: 1,
        y: 1,
        cols: 46,
        rows: 8,
        grid_rows: 8,
    };
    ready_preview(&mut app, want.key, 400, 400);
    app.inline_images.desired = vec![want];

    let mut out = Vec::new();
    assert!(!app.flush_inline_images(&mut out), "no extra frame");
    let seq = String::from_utf8(out).expect("escapes are ascii");
    let start = seq.find("\"1;1;").expect("sixel raster attributes") + 5;
    // `"1;1;W;H#color…` — the palette entry follows H directly.
    let dims: Vec<u32> = seq[start..]
        .split(';')
        .take(2)
        .map(|part| part.split('#').next().unwrap_or(part).parse().unwrap())
        .collect();
    assert_eq!(dims, [782, 252], "46x8 cells at 17x32");
    assert_eq!(dims[0].div_ceil(17), 46, "tmux column count");
    assert_eq!((dims[1].div_ceil(6) * 6).div_ceil(32), 8, "no spill row");
    assert_eq!(app.inline_images.placed, vec![want]);
}

#[tokio::test]
async fn sixel_scroll_movement_holds_placements_until_settled() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();

    assert!(!app.sixel_scroll_hold());
    app.transcript_scroll = 5;
    assert!(app.sixel_scroll_hold());
    assert!(!app.tick_image_scroll_settle(), "still inside the debounce");
    app.inline_images.scroll_settle =
        Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
    assert!(app.tick_image_scroll_settle());
    assert!(!app.sixel_scroll_hold(), "settled scroll no longer holds");

    // Kitty-virtual scrolls placeholders as cells — never held.
    app.inline_images.caps = GraphicsCaps {
        protocol: Protocol::KittyVirtual,
        tmux: false,
        cell_px: (8, 16),
    };
    app.transcript_scroll = 9;
    assert!(!app.sixel_scroll_hold());
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
        grid_rows: 4,
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
        id: None,
        timestamp: None,
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
            id: None,
            timestamp: None,
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

#[tokio::test]
async fn cursor_addressed_flush_restores_the_cursor() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    let want = placement(1);
    ready_preview(&mut app, want.key, 40, 40);
    app.inline_images.desired = vec![want];

    let mut out = Vec::new();
    app.flush_inline_images(&mut out);
    let seq = String::from_utf8(out).expect("escapes are ascii");
    assert!(
        seq.starts_with("\x1b7\x1b["),
        "save before the CUP: {seq:?}"
    );
    assert!(seq.ends_with("\x1b\\\x1b8"), "restore after the image");

    app.inline_images.desired = vec![want];
    let mut out = Vec::new();
    app.flush_inline_images(&mut out);
    assert!(out.is_empty());
}

fn two_tone_preview(app: &mut super::super::CodeTuiApp, key: u64, w: u32, h: u32) {
    let mut rgb = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for _ in 0..w {
            rgb.extend_from_slice(if y < h / 2 {
                &[255, 0, 0]
            } else {
                &[0, 0, 255]
            });
        }
    }
    app.inline_images.previews.insert(
        key,
        PreviewSlot::Ready(Arc::new(EncodedPreview {
            format: PixelFormat::Rgb,
            px_w: w,
            px_h: h,
            payload_b64: String::new(),
            thumb: Some(crate::services::terminal_graphics::Thumb { rgb, w, h }),
            content_hash: key,
        })),
    );
}

fn sixel_paints(seq: &str, color: u32) -> bool {
    let tag = format!("#{color}");
    seq.match_indices(&tag).any(|(i, _)| {
        seq[i + tag.len()..]
            .chars()
            .next()
            .is_some_and(|c| c != ';' && !c.is_ascii_digit())
    })
}

#[tokio::test]
async fn bottom_clipped_sixel_crops_instead_of_squashing() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = sixel_caps();
    two_tone_preview(&mut app, 5, 64, 64);
    app.inline_images.desired = vec![PlacedImage {
        key: 5,
        x: 0,
        y: 0,
        cols: 8,
        rows: 2,
        grid_rows: 4,
    }];

    let mut out = Vec::new();
    app.flush_inline_images(&mut out);
    let seq = String::from_utf8(out).expect("escapes are ascii");
    assert!(seq.contains("\"1;1;64;30"), "2 of 4 rows at 8x16, banded");
    // pure red = 210, pure blue = 5.
    assert!(sixel_paints(&seq, 210), "top slice painted");
    assert!(
        !sixel_paints(&seq, 5),
        "bottom half cropped, not squashed in"
    );
}

#[tokio::test]
async fn bottom_clipped_kitty_keeps_the_full_grid() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    let clipped = PlacedImage {
        key: 5,
        x: 0,
        y: 0,
        cols: 8,
        rows: 2,
        grid_rows: 4,
    };

    app.inline_images.caps = GraphicsCaps {
        protocol: Protocol::KittyVirtual,
        tmux: false,
        cell_px: (8, 16),
    };
    ready_preview(&mut app, 5, 64, 64);
    app.inline_images.desired = vec![clipped];
    let mut out = Vec::new();
    app.flush_inline_images(&mut out);
    let seq = String::from_utf8(out).expect("escapes are ascii");
    assert!(seq.contains("U=1,i="), "{seq:?}");
    assert!(
        seq.contains("c=8,r=4"),
        "placement grid = full image: {seq:?}"
    );

    app.inline_images.caps.protocol = Protocol::KittyClassic;
    app.reset_inline_image_terminal_state();
    app.inline_images.desired = vec![clipped];
    let mut out = Vec::new();
    app.flush_inline_images(&mut out);
    let seq = String::from_utf8(out).expect("escapes are ascii");
    assert!(
        seq.contains("c=8,r=2,y=0,h=32,C=1"),
        "top half of 64px: {seq:?}"
    );
}

#[tokio::test]
async fn render_places_bottom_clipped_preview_with_its_full_grid() {
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
        id: None,
        timestamp: None,
    });
    for i in 0..30 {
        app.history.push(ChatMessage {
            model: None,
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("filler {i}"),
            reasoning_content: None,
            attachments: vec![],
            id: None,
            timestamp: None,
        });
    }
    let key = 42;
    app.inline_images
        .pinned
        .insert(pin_id(0, content, "shot.png"), key);
    ready_preview(&mut app, key, 400, 400);

    let clipped = (8..40).find_map(|h| {
        app.follow_output = false;
        app.transcript_scroll = 0;
        render_full_screen(&mut app, 80, h);
        app.inline_images
            .desired
            .first()
            .copied()
            .filter(|p| p.rows < p.grid_rows)
    });
    let placed = clipped.expect("some height cuts through the preview");
    let full = app
        .render_cache
        .transcript
        .as_ref()
        .and_then(|c| c.wrapped.as_ref())
        .and_then(|w| w.image_rows.first().map(|&(_, a)| a.rows))
        .expect("anchored");
    assert_eq!(placed.grid_rows, full);
}
