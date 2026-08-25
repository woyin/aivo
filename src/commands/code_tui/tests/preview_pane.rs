//! Side preview pane: pin/close, mtime re-key, and the layout split.

use super::super::ChatMessage;
use super::super::preview_pane::{MIN_PREVIEW_SPLIT_COLS, PreviewPaneTarget};
use super::helpers::*;
use crate::services::terminal_graphics::{GraphicsCaps, Protocol};
use ratatui::layout::Rect;
use std::time::{Duration, Instant};

fn virtual_caps() -> GraphicsCaps {
    GraphicsCaps {
        protocol: Protocol::KittyVirtual,
        tmux: false,
        cell_px: (8, 16),
    }
}

/// A tiny valid PNG (1×1) so classification-by-content also passes.
const PNG_1X1: &[u8] = &[
    0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D, b'I', b'H', b'D', b'R', 0, 0, 0,
    1, 0, 0, 0, 1, 8, 2, 0, 0, 0, 0x90, 0x77, 0x53, 0xDE,
];

#[tokio::test]
async fn preview_command_pins_and_closes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-test");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("logo.png");
    std::fs::write(&png, PNG_1X1).unwrap();

    app.run_preview_command(Some(png.display().to_string()));
    let pane = app.preview_pane.as_ref().expect("pane pinned");
    assert!(matches!(pane.target, PreviewPaneTarget::File { .. }));
    assert_eq!(pane.display, "logo.png");
    let key = pane.key.expect("first poll keyed the file");
    assert!(
        app.inline_images.previews.contains_key(&key),
        "prep job registered"
    );

    app.run_preview_command(Some("off".to_string()));
    assert!(app.preview_pane.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("closed"));
}

#[tokio::test]
async fn preview_command_rejects_bad_targets() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();

    app.run_preview_command(Some("no-such-file.png".to_string()));
    assert!(app.preview_pane.is_none());

    let dir = crate::test_sandbox::tmp("aivo-preview-test");
    std::fs::create_dir_all(&dir).unwrap();
    let rs = dir.join("main.rs");
    std::fs::write(&rs, "fn main() {}").unwrap();
    app.run_preview_command(Some(rs.display().to_string()));
    assert!(app.preview_pane.is_none());
    assert!(app.notice.as_ref().unwrap().1.contains("not previewable"));

    app.inline_images.caps = GraphicsCaps::default();
    let png = dir.join("a.png");
    std::fs::write(&png, PNG_1X1).unwrap();
    app.run_preview_command(Some(png.display().to_string()));
    assert!(app.preview_pane.is_none());
}

#[tokio::test]
async fn tick_rekeys_when_the_file_changes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-test");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("live.png");
    std::fs::write(&png, PNG_1X1).unwrap();
    app.run_preview_command(Some(png.display().to_string()));
    let first = app.preview_pane.as_ref().unwrap().key.unwrap();

    app.preview_pane.as_mut().unwrap().last_poll = Instant::now() - Duration::from_secs(2);
    assert!(!app.tick_preview_pane());
    assert_eq!(app.preview_pane.as_ref().unwrap().key, Some(first));

    let mut grown = PNG_1X1.to_vec();
    grown.push(0);
    std::fs::write(&png, grown).unwrap();
    app.preview_pane.as_mut().unwrap().last_poll = Instant::now() - Duration::from_secs(2);
    assert!(app.tick_preview_pane());
    let second = app.preview_pane.as_ref().unwrap().key.unwrap();
    assert_ne!(first, second);
    assert!(app.inline_images.previews.contains_key(&second));
}

#[tokio::test]
async fn agent_preview_tool_call_pins_and_closes() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-test");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("tool.png");
    std::fs::write(&png, PNG_1X1).unwrap();

    app.apply_preview_tool_call(&serde_json::json!({ "target": png.display().to_string() }));
    assert!(app.preview_pane.is_some());

    app.apply_preview_tool_call(&serde_json::json!({ "close": true }));
    assert!(app.preview_pane.is_none());

    app.apply_preview_tool_call(&serde_json::json!({ "target": "missing.svg" }));
    assert!(app.preview_pane.is_none());
}

#[tokio::test]
async fn fuzzy_query_resolves_kind_and_recency() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-fuzzy");
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    app.real_cwd = dir.display().to_string();

    // Backdate the older so "current html file" must pick the newer.
    std::fs::write(dir.join("old.html"), "<html/>").unwrap();
    let old = std::fs::File::options()
        .write(true)
        .open(dir.join("old.html"))
        .unwrap();
    old.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1))
        .unwrap();
    std::fs::write(dir.join("index.html"), "<html/>").unwrap();
    std::fs::write(dir.join("assets").join("logo.svg"), "<svg/>").unwrap();

    app.run_preview_command(Some("current html file".to_string()));
    assert_eq!(app.preview_pane.as_ref().unwrap().display, "index.html");
    assert!(app.notice.as_ref().unwrap().1.contains("index.html"));

    app.run_preview_command(Some("logo".to_string()));
    assert_eq!(app.preview_pane.as_ref().unwrap().display, "logo.svg");

    app.run_preview_command(Some("nonexistent gadget".to_string()));
    assert_eq!(app.preview_pane.as_ref().unwrap().display, "logo.svg");
    assert!(
        app.notice
            .as_ref()
            .unwrap()
            .1
            .contains("nonexistent gadget")
    );
}

#[tokio::test]
async fn reload_rerenders_unchanged_file_under_fresh_key() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-reload");
    std::fs::create_dir_all(&dir).unwrap();
    let html = dir.join("doodle.html");
    std::fs::write(&html, "<html/>").unwrap();
    app.run_preview_command(Some(html.display().to_string()));
    let first = app.preview_pane.as_ref().unwrap().key.unwrap();

    app.run_preview_command(Some("reload".to_string()));
    let second = app.preview_pane.as_ref().unwrap().key.unwrap();
    assert_ne!(first, second, "reload must not reuse the cached render");
    assert!(app.inline_images.previews.contains_key(&second));
    assert!(app.notice.as_ref().unwrap().1.contains("re-rendering"));

    // Re-previewing the SAME target is a reload in disguise.
    app.apply_preview_tool_call(&serde_json::json!({ "target": html.display().to_string() }));
    let third = app.preview_pane.as_ref().unwrap().key.unwrap();
    assert_ne!(second, third);

    app.apply_preview_tool_call(&serde_json::json!({ "reload": true }));
    let fourth = app.preview_pane.as_ref().unwrap().key.unwrap();
    assert_ne!(third, fourth);
    app.run_preview_command(Some("off".to_string()));
    app.run_preview_command(Some("reload".to_string()));
    assert!(app.notice.as_ref().unwrap().1.contains("no preview pane"));
}

#[test]
fn fuzzy_resolver_skips_hidden_and_heavy_dirs() {
    use super::super::preview_pane::resolve_fuzzy_preview_target;
    let dir = crate::test_sandbox::tmp("aivo-preview-fuzzy-skip");
    for sub in [".git", "target", "node_modules"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
        std::fs::write(dir.join(sub).join("buried.html"), "<html/>").unwrap();
    }
    assert_eq!(resolve_fuzzy_preview_target(&dir, "html"), None);
    // Kindless, nameless queries never guess.
    std::fs::write(dir.join("real.html"), "<html/>").unwrap();
    assert_eq!(resolve_fuzzy_preview_target(&dir, "the current"), None);
    assert_eq!(
        resolve_fuzzy_preview_target(&dir, "html"),
        Some(dir.join("real.html"))
    );
}

#[tokio::test]
async fn pane_renders_header_image_cells_and_hint() {
    use crate::services::terminal_graphics::{EncodedPreview, PLACEHOLDER_CHAR, PixelFormat};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use std::sync::Arc;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-render");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("logo.png");
    std::fs::write(&png, PNG_1X1).unwrap();
    app.run_preview_command(Some(png.display().to_string()));
    let key = app.preview_pane.as_ref().unwrap().key.unwrap();
    // Swap the pending prep for a ready preview so the image path renders.
    app.inline_images.previews.insert(
        key,
        super::super::inline_images::PreviewSlot::Ready(Arc::new(EncodedPreview {
            format: PixelFormat::Png,
            px_w: 100,
            px_h: 80,
            payload_b64: "AAAA".into(),
            thumb: None,
            content_hash: 1,
        })),
    );

    let (w, h) = (140u16, 30u16);
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let buf = terminal.backend().buffer().clone();
    let mut screen = String::new();
    for y in 0..h {
        for x in 0..w {
            screen.push_str(buf[(x, y)].symbol());
        }
        screen.push('\n');
    }
    assert!(screen.contains("logo.png"), "pane title shown");
    assert!(
        screen.contains("image · 100×80 · live"),
        "metadata row shown"
    );
    assert!(
        screen.contains("click ✕ or /preview off"),
        "dismiss hint shown"
    );
    assert_eq!(
        app.preview_close_hits.len(),
        2,
        "header button and hint ✕ both registered"
    );
    let close_hit = app.preview_close_hits[0];
    assert_eq!(
        buf[(close_hit.x + close_hit.width - 1, close_hit.y)].symbol(),
        "✕",
        "close glyph drawn at the hit box's right edge"
    );
    let hint_hit = app.preview_close_hits[1];
    assert_eq!(
        buf[(hint_hit.x + 1, hint_hit.y)].symbol(),
        "✕",
        "hint hit box centers on the hint's ✕ glyph"
    );
    assert!(
        screen.contains(PLACEHOLDER_CHAR),
        "placeholder cells painted for the virtual placement"
    );
    assert!(
        app.inline_images.desired.iter().any(|p| p.key == key),
        "pane placement registered for the flush"
    );
    // Even on the empty welcome screen the divider must reach the composer.
    let deep_divider =
        (h.saturating_sub(8)..h).any(|y| (0..w).any(|x| buf[(x, y)].symbol() == "│"));
    assert!(
        deep_divider,
        "pane divider extends to the lower screen rows"
    );
}

#[tokio::test]
async fn clicking_the_close_button_dismisses_the_pane() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-close-click");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("logo.png");
    std::fs::write(&png, PNG_1X1).unwrap();
    app.run_preview_command(Some(png.display().to_string()));

    let mut terminal = Terminal::new(TestBackend::new(140, 30)).unwrap();
    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    let hit = *app
        .preview_close_hits
        .first()
        .expect("close button registered");

    app.handle_mouse(left_click(hit.x.saturating_sub(4), hit.y))
        .await
        .unwrap();
    assert!(app.preview_pane.is_some(), "miss keeps the pane");

    app.handle_mouse(left_click(hit.x + hit.width - 1, hit.y))
        .await
        .unwrap();
    assert!(app.preview_pane.is_none(), "click on ✕ closes the pane");

    terminal
        .draw(|frame| {
            app.render_main(frame, frame.area());
        })
        .unwrap();
    assert!(app.preview_close_hits.is_empty());
}

#[tokio::test]
async fn list_dir_results_never_queue_image_previews() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-listdir");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a.png"), PNG_1X1).unwrap();
    std::fs::write(dir.join("b.png"), PNG_1X1).unwrap();
    app.real_cwd = dir.display().to_string();
    let listing = "a.png\nb.png\n".to_string();
    let entry = |role: &str, content: String| ChatMessage {
        model: None,
        role: role.to_string(),
        content,
        reasoning_content: None,
        attachments: vec![],
    };

    app.history.push(entry(
        "tool_call",
        serde_json::json!({"name": "list_dir", "args": {"path": "."}}).to_string(),
    ));
    app.history.push(entry("tool_result", listing.clone()));
    app.queue_missing_previews();
    assert!(
        app.inline_images.previews.is_empty(),
        "list_dir results must not preview images"
    );

    app.history.clear();
    app.history.push(entry(
        "tool_call",
        serde_json::json!({"name": "glob", "args": {"pattern": "*.png"}}).to_string(),
    ));
    app.history.push(entry("tool_result", listing));
    app.queue_missing_previews();
    assert!(
        !app.inline_images.previews.is_empty(),
        "non-listing tools keep result previews"
    );
}

#[test]
fn pane_grid_fits_the_box() {
    use super::super::inline_images::pane_preview_grid;
    let (cols, rows) = pane_preview_grid(1280, 800, 40, 10, Protocol::KittyVirtual).unwrap();
    assert!(cols <= 40 && rows <= 10, "{cols}x{rows}");
    assert!(pane_preview_grid(100, 100, 3, 10, Protocol::KittyVirtual).is_none());
    assert!(pane_preview_grid(100, 100, 40, 1, Protocol::KittyVirtual).is_none());
    assert!(pane_preview_grid(0, 100, 40, 10, Protocol::KittyVirtual).is_none());
}

#[tokio::test]
async fn split_carves_the_pane_only_when_wide_enough() {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = make_test_app(tx, rx);
    app.inline_images.caps = virtual_caps();
    let dir = crate::test_sandbox::tmp("aivo-preview-test");
    std::fs::create_dir_all(&dir).unwrap();
    let png = dir.join("split.png");
    std::fs::write(&png, PNG_1X1).unwrap();

    let wide = Rect::new(0, 0, 140, 30);
    assert_eq!(app.split_preview_pane(wide), (wide, None));

    app.run_preview_command(Some(png.display().to_string()));
    let (left, pane) = app.split_preview_pane(wide);
    let pane = pane.expect("wide window splits");
    assert_eq!(left.width + pane.width, wide.width);
    assert_eq!(pane.x, left.x + left.width);
    assert_eq!(pane.height, wide.height);
    assert!(pane.width >= 30 && pane.width <= 64, "{}", pane.width);

    let narrow = Rect::new(0, 0, MIN_PREVIEW_SPLIT_COLS - 1, 30);
    assert_eq!(app.split_preview_pane(narrow), (narrow, None));
    assert!(app.preview_pane.is_some());
}
