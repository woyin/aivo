//! Side preview pane: `/preview` or the agent's `preview` tool pins an
//! image/SVG/HTML file (or URL) right of the transcript. File targets re-key
//! on (path, len, mtime) each poll, so a save re-renders.

use super::*;
use crate::services::svg_raster::{self, PreviewTargetKind};
use crate::services::terminal_graphics::{self, Protocol, image_id};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Below this width no pane is carved out; the pin waits (state kept).
pub(super) const MIN_PREVIEW_SPLIT_COLS: u16 = 88;
/// Pane share of the transcript row, clamped so neither side starves.
const PANE_MIN_COLS: u16 = 30;
const PANE_MAX_COLS: u16 = 64;
/// File re-stat cadence; HTML re-renders shell out to a browser, so stay coarse.
const PANE_POLL: Duration = Duration::from_millis(500);

pub(super) enum PreviewPaneTarget {
    File {
        path: PathBuf,
        kind: PreviewTargetKind,
    },
    /// Fetched once at pin time; the URL itself lives in `display`.
    Url,
}

pub(super) struct PreviewPane {
    pub(super) target: PreviewPaneTarget,
    /// Short label for the pane title (file name, or the URL itself).
    pub(super) display: String,
    /// Content key of the latest prep; `None` until the first poll resolves.
    pub(super) key: Option<u64>,
    pub(super) last_poll: Instant,
    /// Bumped by reload: mixed into the content key so an unchanged file
    /// re-renders under a FRESH kitty image id (retransmit-same-id gap).
    pub(super) reload_gen: u64,
}

/// Gen 0 keeps the raw key so pane and transcript share one cache entry.
fn reload_key(raw: u64, generation: u64) -> u64 {
    raw ^ generation.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl CodeTuiApp {
    pub(super) fn run_preview_command(&mut self, arg: Option<String>) {
        let arg = arg.map(|a| a.trim().to_string()).filter(|a| !a.is_empty());
        match arg.as_deref() {
            None => {
                if self.close_preview_pane() {
                    self.notice = Some((MUTED(), "preview pane closed".to_string()));
                } else {
                    self.notice = Some((MUTED(), "Usage: /preview <path|url|off>".to_string()));
                }
            }
            Some("off") | Some("close") => {
                let message = if self.close_preview_pane() {
                    "preview pane closed"
                } else {
                    "no preview pane open"
                };
                self.notice = Some((MUTED(), message.to_string()));
            }
            Some("reload") | Some("refresh") => {
                let message = if self.reload_preview_pane() {
                    "re-rendering the preview"
                } else {
                    "no preview pane open"
                };
                self.notice = Some((MUTED(), message.to_string()));
            }
            Some(target) => match self.pin_preview_command(target) {
                Ok(label) => {
                    self.notice =
                        Some((MUTED(), format!("previewing {label} — /preview off closes")));
                }
                Err(error) => self.notice = Some((ERROR(), error)),
            },
        }
    }

    /// Exact path first, then a fuzzy workspace search ("current html file", "logo").
    fn pin_preview_command(&mut self, raw: &str) -> Result<String, String> {
        let primary = match self.pin_preview_target(raw) {
            Ok(label) => return Ok(label),
            Err(e) => e,
        };
        let base = self.preview_base().to_string();
        let Some(found) = resolve_fuzzy_preview_target(Path::new(&base), raw) else {
            return Err(primary);
        };
        let label = self.pin_preview_target(&found.display().to_string())?;
        // Show what the query resolved to — never silently pin a guess.
        Ok(found
            .strip_prefix(&base)
            .map(|p| p.display().to_string())
            .unwrap_or(label))
    }

    /// Pins `raw` (path or http(s) URL); returns the display label. Validation
    /// mirrors the engine's `preview_call`, so an accepted tool call pins here.
    pub(super) fn pin_preview_target(&mut self, raw: &str) -> Result<String, String> {
        if !self.inline_images.caps.enabled() {
            return Err(
                "no terminal graphics detected — set AIVO_PREVIEW=1 to force half-block previews"
                    .to_string(),
            );
        }
        if raw.starts_with("http://") || raw.starts_with("https://") {
            let key = hash_url(raw);
            self.preview_pane = Some(PreviewPane {
                target: PreviewPaneTarget::Url,
                display: raw.to_string(),
                key: Some(key),
                last_poll: Instant::now(),
                reload_gen: 0,
            });
            self.spawn_pane_url_preview(key, raw.to_string());
            return Ok(raw.to_string());
        }
        let path = resolve_in(self.preview_base(), raw);
        let meta = std::fs::metadata(&path).map_err(|e| format!("{raw}: {e}"))?;
        if !meta.is_file() {
            return Err(format!("{raw} is not a file"));
        }
        let Some(kind) = svg_raster::classify_preview_target(&path) else {
            return Err(format!(
                "{raw}: not previewable — supported: PNG/JPEG, SVG, HTML"
            ));
        };
        let display = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| raw.to_string());
        self.preview_pane = Some(PreviewPane {
            target: PreviewPaneTarget::File { path, kind },
            display: display.clone(),
            key: None,
            // Backdated so the tick below re-stats immediately.
            last_poll: Instant::now() - PANE_POLL,
            reload_gen: 0,
        });
        self.tick_preview_pane();
        Ok(display)
    }

    pub(super) fn close_preview_pane(&mut self) -> bool {
        self.preview_pane.take().is_some()
    }

    /// Validation matches the engine's `preview_call` — a rejected call opens no pane.
    pub(super) fn apply_preview_tool_call(&mut self, args: &serde_json::Value) {
        if args.get("close").and_then(|v| v.as_bool()).unwrap_or(false) {
            self.close_preview_pane();
            return;
        }
        let reload = args
            .get("reload")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let target = args
            .get("target")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty());
        // Re-pinning the already-shown target is a reload request in disguise.
        let same_target = matches!((target, self.preview_pane.as_ref()), (Some(t), Some(pane))
            if pane.display == t || matches!(&pane.target, PreviewPaneTarget::File { path, .. }
                if resolve_in(self.preview_base(), t) == *path));
        if let Some(target) = target
            && !same_target
        {
            let _ = self.pin_preview_target(target);
        }
        if reload || same_target {
            self.reload_preview_pane();
        }
    }

    /// Fresh render of an unchanged target — randomized pages differ every load.
    pub(super) fn reload_preview_pane(&mut self) -> bool {
        let Some(pane) = self.preview_pane.as_mut() else {
            return false;
        };
        pane.reload_gen = pane.reload_gen.wrapping_add(1);
        pane.key = None;
        match &pane.target {
            PreviewPaneTarget::File { .. } => {
                pane.last_poll = Instant::now() - PANE_POLL;
                self.tick_preview_pane();
            }
            PreviewPaneTarget::Url => {
                let url = pane.display.clone();
                let key = reload_key(hash_url(&url), pane.reload_gen);
                pane.key = Some(key);
                self.spawn_pane_url_preview(key, url);
            }
        }
        true
    }

    /// Re-stat a pinned file at the poll cadence; a changed key spawns a fresh
    /// prep. `true` = redraw-worthy change.
    pub(super) fn tick_preview_pane(&mut self) -> bool {
        let Some(pane) = self.preview_pane.as_mut() else {
            return false;
        };
        let PreviewPaneTarget::File { path, kind } = &pane.target else {
            return false;
        };
        if pane.last_poll.elapsed() < PANE_POLL {
            return false;
        }
        pane.last_poll = Instant::now();
        // A vanished/mid-write file keeps the last good render.
        let Some(key) = file_key(path).map(|raw| reload_key(raw, pane.reload_gen)) else {
            return false;
        };
        if pane.key == Some(key) {
            return false;
        }
        pane.key = Some(key);
        let (path, kind) = (path.clone(), *kind);
        self.spawn_pane_file_preview(key, path, kind);
        true
    }

    fn spawn_pane_file_preview(&mut self, key: u64, path: PathBuf, kind: PreviewTargetKind) {
        if self.inline_images.previews.contains_key(&key) {
            return; // transcript machinery already prepped this exact state
        }
        self.inline_images
            .previews
            .insert(key, PreviewSlot::Pending);
        let tx = self.tx.clone();
        tokio::task::spawn_blocking(move || {
            let preview = match kind {
                PreviewTargetKind::Html => svg_raster::rasterize_page(&path.display().to_string())
                    .and_then(|png| terminal_graphics::prepare_preview(&png)),
                _ => prepare_preview_source(PreviewSource::File(path)),
            }
            .map(Box::new);
            let _ = tx.send(RuntimeEvent::ImagePreviewReady { key, preview });
        });
    }

    /// Image-looking URLs ride the existing fetch path; anything else is
    /// screenshot as a page.
    fn spawn_pane_url_preview(&mut self, key: u64, url: String) {
        if has_image_extension(&url) {
            self.spawn_url_preview(key, url);
            return;
        }
        if self.inline_images.previews.contains_key(&key) {
            return;
        }
        self.inline_images
            .previews
            .insert(key, PreviewSlot::Pending);
        let tx = self.tx.clone();
        tokio::task::spawn_blocking(move || {
            let preview = svg_raster::rasterize_page(&url)
                .and_then(|png| terminal_graphics::prepare_preview(&png))
                .map(Box::new);
            let _ = tx.send(RuntimeEvent::ImagePreviewReady { key, preview });
        });
    }

    /// The pin survives a too-narrow window; it just doesn't render.
    pub(super) fn split_preview_pane(&self, area: Rect) -> (Rect, Option<Rect>) {
        if self.preview_pane.is_none() || area.width < MIN_PREVIEW_SPLIT_COLS {
            return (area, None);
        }
        let pane_width = (area.width * 2 / 5).clamp(PANE_MIN_COLS, PANE_MAX_COLS);
        let left_width = area.width - pane_width;
        let left = Rect {
            width: left_width,
            ..area
        };
        let pane = Rect {
            x: area.x.saturating_add(left_width),
            width: pane_width,
            ..area
        };
        (left, Some(pane))
    }

    /// Prep in flight — the spinner needs repaints without input (`is_animating`).
    pub(super) fn preview_pane_loading(&self) -> bool {
        self.preview_pane.as_ref().is_some_and(|pane| {
            !matches!(
                pane.key.map(|key| self.inline_images.previews.get(&key)),
                Some(Some(PreviewSlot::Ready(_) | PreviewSlot::Failed))
            )
        })
    }

    /// Draws the pane and registers the image placement for the post-draw
    /// flush. Must run AFTER `collect_desired_inline_images` (which clears
    /// `desired`).
    pub(super) fn render_preview_pane(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let Some(pane) = &self.preview_pane else {
            return;
        };
        // Outside the left-column stack, so it clears its own canvas region.
        clear_to_canvas(frame, area);
        let block = Block::default()
            .borders(Borders::LEFT)
            .border_style(Style::default().fg(FAINT()));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        if inner.width < 6 || inner.height < 3 {
            return;
        }
        let content = Rect {
            x: inner.x.saturating_add(1),
            width: inner.width.saturating_sub(1),
            ..inner
        };
        let row = |y_offset: u16| Rect {
            y: content.y.saturating_add(y_offset),
            height: 1,
            ..content
        };
        let title_row = row(0);
        let close = Rect {
            x: title_row.x + title_row.width.saturating_sub(1),
            width: 1,
            ..title_row
        };
        frame.render_widget(
            Paragraph::new(Span::styled(
                middle_truncate(&pane.display, usize::from(content.width.saturating_sub(3))),
                Style::default().fg(TEXT()).add_modifier(Modifier::BOLD),
            )),
            title_row,
        );
        frame.render_widget(
            Paragraph::new(Span::styled("✕", Style::default().fg(MUTED()))),
            close,
        );
        // Generous hit box: the glyph cell plus the two cells left of it.
        self.preview_close_hits.push(Rect {
            x: close.x.saturating_sub(2),
            width: 3,
            ..close
        });

        let slot = pane.key.map(|key| self.inline_images.previews.get(&key));
        let ready = match slot {
            Some(Some(PreviewSlot::Ready(preview))) => Some(Arc::clone(preview)),
            _ => None,
        };
        let kind_label = match &pane.target {
            PreviewPaneTarget::File { kind, .. } => match kind {
                PreviewTargetKind::Image => "image",
                PreviewTargetKind::Svg => "svg",
                PreviewTargetKind::Html => "html",
            },
            PreviewPaneTarget::Url => "url",
        };
        let mut meta = vec![Span::styled(kind_label, Style::default().fg(MUTED()))];
        if let Some(preview) = &ready {
            meta.push(Span::styled(
                format!(" · {}×{}", preview.px_w, preview.px_h),
                Style::default().fg(MUTED()),
            ));
            if matches!(pane.target, PreviewPaneTarget::File { .. }) {
                meta.push(Span::styled(" · live", Style::default().fg(FAINT())));
            }
        } else if matches!(slot, Some(Some(PreviewSlot::Failed))) {
            meta.push(Span::styled(
                " · no preview available",
                Style::default().fg(MUTED()),
            ));
        } else {
            meta.push(Span::styled(
                format!(
                    " · {} rendering…",
                    spinner_frame_indexed(self.frame_tick, self.reduce_motion)
                ),
                Style::default().fg(MUTED()),
            ));
        }
        if inner.height >= 5 {
            frame.render_widget(Paragraph::new(Line::from(meta)), row(1));
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "─".repeat(usize::from(content.width)),
                    Style::default().fg(FAINT()),
                )),
                row(2),
            );
        }
        let Some(preview) = ready else {
            return;
        };
        let Some(key) = pane.key else { return };
        let protocol = self.inline_images.caps.protocol;
        let header_rows: u16 = if inner.height >= 5 { 4 } else { 2 };
        let mut box_area = Rect {
            y: content.y.saturating_add(header_rows),
            height: inner.height.saturating_sub(header_rows),
            ..content
        };
        let hint_row = (box_area.height >= 8).then(|| {
            box_area.height -= 2;
            Rect {
                y: box_area.y.saturating_add(box_area.height).saturating_add(1),
                height: 1,
                ..content
            }
        });
        let Some((cols, rows)) = pane_preview_grid(
            preview.px_w,
            preview.px_h,
            box_area.width,
            box_area.height,
            protocol,
        ) else {
            return;
        };
        let image_area = Rect {
            x: box_area.x + (box_area.width.saturating_sub(cols)) / 2,
            y: box_area.y + (box_area.height.saturating_sub(rows)) / 2,
            width: cols,
            height: rows,
        };
        if let Some(hint_row) = hint_row {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "click ✕ or /preview off",
                    Style::default().fg(FAINT()),
                )),
                hint_row,
            );
            // The hint's own ✕ ("click " puts it at offset 6) must close too.
            if hint_row.width > 6 {
                self.preview_close_hits.push(Rect {
                    x: hint_row.x + 5,
                    width: 3,
                    ..hint_row
                });
            }
        }
        let lines: Vec<Line<'static>> = match protocol {
            Protocol::KittyVirtual => {
                let (r, g, b) = terminal_graphics::placeholder_fg(image_id(key));
                (0..rows)
                    .map(|row| {
                        Line::from(Span::styled(
                            terminal_graphics::placeholder_row(row, cols),
                            Style::default().fg(Color::Rgb(r, g, b)),
                        ))
                    })
                    .collect()
            }
            Protocol::HalfBlocks => {
                let Some(thumb) = &preview.thumb else { return };
                let grid = crate::services::image_optimize::resample_rgb_exact(
                    &thumb.rgb,
                    thumb.w,
                    thumb.h,
                    u32::from(cols),
                    u32::from(rows) * 2,
                );
                (0..rows)
                    .map(|row| half_block_row(&grid, row, cols))
                    .collect()
            }
            // Classic/sixel pixels float over blank cells placed post-draw.
            _ => Vec::new(),
        };
        if !lines.is_empty() {
            frame.render_widget(Paragraph::new(Text::from(lines)), image_area);
        }
        self.inline_images.desired.push(PlacedImage {
            key,
            // Virtual composites on the placeholder cells; x/y only matter
            // for cursor-addressed modes.
            x: if protocol == Protocol::KittyVirtual {
                0
            } else {
                image_area.x
            },
            y: if protocol == Protocol::KittyVirtual {
                0
            } else {
                image_area.y
            },
            cols,
            rows,
        });
    }
}

/// Words that describe *which* file without naming it; the mtime tie-break
/// already covers "most recent".
const FUZZY_STOPWORDS: &[&str] = &[
    "current", "latest", "recent", "newest", "last", "the", "this", "my", "a", "file",
];

/// Kind words filter by extension, stopwords drop, every remaining word must
/// appear in the file name; newest mtime wins. Bounded walk (depth 4, ~4k
/// entries), no file reads.
pub(super) fn resolve_fuzzy_preview_target(base: &Path, query: &str) -> Option<PathBuf> {
    use PreviewTargetKind::*;
    let mut kinds: Vec<PreviewTargetKind> = Vec::new();
    let mut name_terms: Vec<String> = Vec::new();
    for token in query.to_lowercase().split_whitespace() {
        match token {
            "html" | "htm" | "page" | "webpage" | "website" => kinds.push(Html),
            "svg" | "vector" => kinds.push(Svg),
            "png" | "jpg" | "jpeg" | "image" | "img" | "picture" | "photo" | "screenshot" => {
                kinds.push(Image)
            }
            t if FUZZY_STOPWORDS.contains(&t) => {}
            // A path-ish term matches by its final segment ("assets/logo.svg").
            t => name_terms.push(t.rsplit('/').next().unwrap_or(t).to_string()),
        }
    }
    if kinds.is_empty() && name_terms.is_empty() {
        return None;
    }
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let mut stack = vec![(base.to_path_buf(), 0usize)];
    let mut visited = 0usize;
    while let Some((dir, depth)) = stack.pop() {
        let Ok(read_dir) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in read_dir.flatten() {
            visited += 1;
            if visited > 4000 {
                return best.map(|(_, path)| path);
            }
            let name = entry.file_name().to_string_lossy().to_lowercase();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                let skip = name.starts_with('.')
                    || matches!(
                        name.as_str(),
                        "target" | "node_modules" | "dist" | "build" | "vendor"
                    );
                if depth < 4 && !skip {
                    stack.push((entry.path(), depth + 1));
                }
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let kind = match Path::new(&name).extension().and_then(|e| e.to_str()) {
                Some("png" | "jpg" | "jpeg") => Image,
                Some("svg") => Svg,
                Some("html" | "htm") => Html,
                _ => continue,
            };
            if !kinds.is_empty() && !kinds.contains(&kind) {
                continue;
            }
            if !name_terms.iter().all(|term| name.contains(term.as_str())) {
                continue;
            }
            let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
                continue;
            };
            if best.as_ref().is_none_or(|(t, _)| modified > *t) {
                best = Some((modified, entry.path()));
            }
        }
    }
    best.map(|(_, path)| path)
}

/// Display-width-naive but safe on char boundaries.
fn middle_truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max || max < 8 {
        return s.to_string();
    }
    let head = (max - 1) / 2;
    let tail = max - 1 - head;
    let mut out: String = chars[..head].iter().collect();
    out.push('…');
    out.extend(chars[chars.len() - tail..].iter());
    out
}
