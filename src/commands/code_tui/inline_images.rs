//! Inline image previews in the transcript (Kitty graphics protocol).
//!
//! The transcript build reserves invisible rows (ZWSP so blank-collapse and
//! wrap leave them alone) under image attachments, MCP-saved images, and
//! agent-edited SVGs; `render_main` maps the anchors that fall inside the
//! visible window to screen cells, and the event loop flushes place/delete
//! escapes after each `terminal.draw`, inside the synchronized update.

use super::*;
use crate::services::session_store::AttachmentStorage;
use crate::services::terminal_graphics::{
    self, EncodedPreview, GraphicsCaps, PLACEHOLDER_CHAR, Protocol, image_id,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_PREVIEW_COLS: u16 = 46;
const MAX_PREVIEW_ROWS: u16 = 12;
/// Gap between previews packed onto the same band.
const PREVIEW_PACK_GAP: u16 = 2;
/// Half-blocks get a bigger cell budget: every cell is worth only 1×2 pixels,
/// so the grid IS the resolution — a square image at 12 rows would be a
/// 24×24-pixel picture.
const MAX_HALF_BLOCK_COLS: u16 = 64;
const MAX_HALF_BLOCK_ROWS: u16 = 16;
/// Terminal cells are ~twice as tall as wide; sizes the row count without a
/// pixel-size query (see the no-probe rule in `terminal_graphics`).
const CELL_WIDTH_OVER_HEIGHT: f64 = 0.5;
/// Rough px-per-column floor so tiny icons don't blow up to banner size.
const MIN_SOURCE_PX_PER_COL: u32 = 10;
/// Images kept transmitted in the terminal before LRU eviction.
const MAX_TRANSMITTED: usize = 24;
/// Debounce for resize re-transmit — a live drag is a resize-event burst.
const RESIZE_SETTLE: std::time::Duration = std::time::Duration::from_millis(250);
/// Debounce for sixel scroll re-placement — a wheel gesture is a scroll burst.
const SCROLL_SETTLE: std::time::Duration = std::time::Duration::from_millis(200);
const MAX_SOURCE_BYTES: u64 = 20 * 1024 * 1024;

/// The reserved transcript row under an image: a zero-width space — invisible
/// and zero-width, but not `trim`-blank, so `compact_lines_and_bars` keeps it
/// and the wrapper passes it through as exactly one row.
const RESERVED_ROW: &str = "\u{200B}";

pub(super) enum PreviewSlot {
    Pending,
    Ready(Arc<EncodedPreview>),
    Failed,
}

/// Where a local preview's bytes come from, resolved in `spawn_blocking`
/// (URLs go through the async fetch in `spawn_url_preview` instead).
pub(super) enum PreviewSource {
    InlineB64(String),
    File(PathBuf),
    /// Extracted inline `<svg>` markup.
    Raw(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlacedImage {
    pub(super) key: u64,
    pub(super) x: u16,
    pub(super) y: u16,
    pub(super) cols: u16,
    pub(super) rows: u16,
    pub(super) grid_rows: u16,
}

#[derive(Default)]
pub(super) struct InlineImageState {
    pub(super) caps: GraphicsCaps,
    /// Content-keyed prepared previews; `Pending` while a job runs.
    pub(super) previews: HashMap<u64, PreviewSlot>,
    /// Mention identity → the preview key resolved when that mention was
    /// first seen. Pins each transcript block to the file's state AT THAT
    /// TIME: a later edit renders as a NEW block under the edit, instead of
    /// silently rewriting every earlier preview of the same path.
    pub(super) pinned: HashMap<u64, u64>,
    /// Keys whose data the terminal currently holds → the virtual-placement
    /// grid sent with them (so a size change re-creates the placement).
    pub(super) transmitted: HashMap<u64, (u16, u16)>,
    pub(super) transmit_order: VecDeque<u64>,
    /// Sixel mode: encoded per (key, cols, grid_rows, shown rows), re-emitted
    /// on every placement change.
    pub(super) sixel_cache: HashMap<(u64, u16, u16, u16), Arc<String>>,
    /// Classic mode only: placements live on the terminal after the last flush.
    pub(super) placed: Vec<PlacedImage>,
    /// What this frame's render wants on screen; virtual mode uses it purely
    /// as the transmit trigger, classic mode diffs it against `placed`.
    pub(super) desired: Vec<PlacedImage>,
    /// Sixel mode: vanished placements whose cells the next render must
    /// force-rewrite to erase the stale pixels.
    pub(super) pending_clears: Vec<PlacedImage>,
    pub(super) resize_settle: Option<std::time::Instant>,
    /// Sixel settle clock: a "move" is erase + full payload re-emission per
    /// wheel tick (through tmux), so transcript placements are withheld until
    /// the scroll rests.
    pub(super) scroll_settle: Option<std::time::Instant>,
    pub(super) last_scroll: usize,
}

impl PlacedImage {
    pub(super) fn rect(&self) -> Rect {
        Rect::new(self.x, self.y, self.cols, self.rows)
    }
}

/// Placement ids derive from the screen row: the same image visible twice
/// (or after a scroll) gets distinct, reproducible placements.
fn placement_id(p: &PlacedImage) -> u32 {
    u32::from(p.y) + 1
}

fn write_restoring_cursor(out: &mut impl std::io::Write, seq: &str) {
    if seq.is_empty() {
        return;
    }
    let _ = out.write_all(b"\x1b7");
    let _ = out.write_all(seq.as_bytes());
    let _ = out.write_all(b"\x1b8");
}

/// `DefaultHasher` over a domain tag plus caller-fed fields.
fn keyed_hash(
    tag: &str,
    write: impl FnOnce(&mut std::collections::hash_map::DefaultHasher),
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut hasher);
    write(&mut hasher);
    hasher.finish()
}

/// Length plus a 4 KiB prefix — cheap enough to run per rebuild on multi-MB
/// pasted payloads, and length makes prefix collisions unlikely.
pub(super) fn hash_inline(data: &str) -> u64 {
    keyed_hash("inline", |h| {
        data.len().hash(h);
        data.as_bytes()[..data.len().min(4096)].hash(h);
    })
}

/// File previews key on (path, len, mtime), so an agent re-editing an SVG
/// yields a fresh key — and therefore a fresh render — automatically.
pub(super) fn file_key(path: &Path) -> Option<u64> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() == 0 || meta.len() > MAX_SOURCE_BYTES {
        return None;
    }
    Some(keyed_hash("file", |h| {
        path.hash(h);
        meta.len().hash(h);
        if let Ok(modified) = meta.modified()
            && let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            since.as_nanos().hash(h);
        }
    }))
}

/// The image path a write/edit/generate_image tool call touches (unresolved).
/// Callers must wait for evidence the call executed, or the pre-write state
/// gets pinned. `read_file` deliberately previews nothing — that's the preview
/// pane's job.
pub(super) fn file_tool_image_target(content: &str) -> Option<String> {
    let (name, args) = decode_tool_call(content);
    let name = canonical_tool_name(&name);
    if !matches!(
        name,
        "write_file" | "edit_file" | "multi_edit" | "generate_image"
    ) {
        return None;
    }
    let path = args.get("path").and_then(|v| v.as_str())?;
    has_image_extension(path).then(|| path.to_string())
}

/// Identity of one image mention: which history entry, roughly what it said,
/// and which path — the pin under which the resolved preview key is frozen.
pub(super) fn pin_id(idx: usize, content: &str, path: &str) -> u64 {
    keyed_hash("pin", |h| {
        idx.hash(h);
        content.len().hash(h);
        content.as_bytes()[..content.len().min(64)].hash(h);
        path.hash(h);
    })
}

/// Writes a pasted attachment's bytes to a stable temp file so the OS viewer
/// can open something; content-keyed name, so repeat clicks reuse it.
fn materialize_inline(key: u64, data_b64: &str, mime: &str) -> Option<PathBuf> {
    use base64::Engine as _;
    let ext = match mime {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/svg+xml" => "svg",
        _ => "png",
    };
    let path = std::env::temp_dir().join(format!("aivo-preview-{key:016x}.{ext}"));
    if !path.exists() {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data_b64.as_bytes())
            .ok()?;
        std::fs::write(&path, bytes).ok()?;
    }
    Some(path)
}

/// Resolution matches the agent's own tool-path rules (absolute kept,
/// `~` expanded, else joined onto the working dir).
pub(super) fn resolve_in(base: &str, path: &str) -> PathBuf {
    crate::agent::tools::resolve(Path::new(base), path)
}

pub(super) fn has_image_extension(path: &str) -> bool {
    let bytes = path.as_bytes();
    [".svg", ".png", ".jpg", ".jpeg"].iter().any(|ext| {
        bytes.len() >= ext.len()
            && bytes[bytes.len() - ext.len()..].eq_ignore_ascii_case(ext.as_bytes())
    })
}

fn image_stem(path: &str) -> String {
    let lower = path.to_ascii_lowercase();
    for ext in [".svg", ".png", ".jpg", ".jpeg"] {
        if let Some(stem) = lower.strip_suffix(ext) {
            return stem.to_string();
        }
    }
    lower
}

/// `foo.svg` + `foo.png` in one message is one picture (source + render) —
/// content-hash dedup can't see that (different pixels). First mention wins.
fn dedup_same_stem(paths: Vec<String>) -> Vec<String> {
    let mut seen: Vec<String> = Vec::new();
    paths
        .into_iter()
        .filter(|path| {
            let stem = image_stem(path);
            if seen.contains(&stem) {
                false
            } else {
                seen.push(stem);
                true
            }
        })
        .collect()
}

/// Key for a URL preview — content-unknown until fetched, so the URL string
/// itself is the identity (first fetch wins, like a pin).
pub(super) fn hash_url(url: &str) -> u64 {
    keyed_hash("url", |h| url.hash(h))
}

/// Distinct punctuation-trimmed whitespace tokens accepted by `keep`, capped.
fn image_tokens(text: &str, cap: usize, keep: impl Fn(&str) -> bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let push = |token: &str, out: &mut Vec<String>| {
        if token.is_empty() || !keep(token) || out.iter().any(|t| t == token) {
            return;
        }
        out.push(token.to_string());
    };
    // `![alt](path.png)`: the `](` sits mid-token, so whitespace splitting alone
    // never recovers the path — the way models usually present an image.
    for target in markdown_link_targets(text) {
        push(target, &mut out);
        if out.len() == cap {
            return out;
        }
    }
    for token in text.split(is_token_boundary) {
        let token = token.trim_matches(|c: char| "\"'`()[]{}<>,;:!?*".contains(c));
        // Markdown leftovers pass the extension gate and burn a slot on a path
        // that cannot exist; the pass above already had them.
        if token.contains("](") {
            continue;
        }
        push(token, &mut out);
        if out.len() == cap {
            break;
        }
    }
    out
}

/// Whitespace plus CJK punctuation, which sets flush against the word
/// (`路径：a.svg`, `a.svg。`) and so never splits on whitespace alone.
fn is_token_boundary(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '。' | '，'
                | '、'
                | '：'
                | '；'
                | '！'
                | '？'
                | '（'
                | '）'
                | '【'
                | '】'
                | '「'
                | '」'
                | '『'
                | '』'
                | '《'
                | '》'
                | '〈'
                | '〉'
                | '…'
        )
}

/// The `path` in `](path)`, minus an optional `"title"` and `<>` wrapping.
fn markdown_link_targets(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for (i, _) in text.match_indices("](") {
        let rest = &text[i + "](".len()..];
        let Some(end) = rest.find(')') else {
            continue;
        };
        let target = rest[..end].trim();
        let target = target.split_whitespace().next().unwrap_or(""); // drop a "Title"
        let target = target.trim_start_matches('<').trim_end_matches('>');
        if !target.is_empty() {
            out.push(target);
        }
    }
    out
}

/// Image URLs mentioned in text — an image-generation tool often returns ONLY
/// a URL, so the mention is the whole hook. Extension-gated on the path
/// (query/fragment stripped), http(s) only, capped at 2 per message.
fn text_image_urls(text: &str) -> Vec<String> {
    image_tokens(text, 2, |token| {
        (token.starts_with("https://") || token.starts_with("http://"))
            && has_image_extension(token.split(['?', '#']).next().unwrap_or(token))
    })
}

/// Bare-path previews per result, so a broad glob over a screenshots dir
/// doesn't turn the transcript into a gallery.
const MAX_RESULT_PATH_PREVIEWS: usize = 3;

/// Image-path tokens in free text ("show octagon.svg") — the agent often
/// answers such prompts from context with no tool call at all, so the message
/// itself is the only hook. Stat-gating (at queue/push time) keeps ordinary
/// prose mentioning filenames quiet.
fn text_image_paths(text: &str) -> Vec<String> {
    dedup_same_stem(image_tokens(
        text,
        MAX_RESULT_PATH_PREVIEWS,
        has_image_extension,
    ))
}

const MAX_INLINE_SVG_BLOCKS: usize = 2;

const MAX_INLINE_SVG_BYTES: usize = 512 * 1024;

/// Inline `<svg>…</svg>` in message text (fenced or bare). Models often paste
/// markup with no file path; content-addressed like pasted attachments.
fn inline_svg_blocks(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = text;
    while out.len() < MAX_INLINE_SVG_BLOCKS {
        let Some(start) = rest.find("<svg") else {
            break;
        };
        let after = &rest[start..];
        // `<svg>` / `<svg ` / `<svg\n` — not `<svgfoo>`. A false hit only
        // skips the tag, so a real block right after it still extracts.
        if !after[4..]
            .chars()
            .next()
            .is_some_and(|c| c == '>' || c.is_whitespace())
        {
            rest = &after[4..];
            continue;
        }
        let Some(end) = after.find("</svg>") else {
            break;
        };
        let close = end + "</svg>".len();
        if close <= MAX_INLINE_SVG_BYTES {
            let block = &after[..close];
            // External refs would beacon via the browser rasterizer rungs;
            // rasterize_svg re-checks, this just skips doomed work early.
            if !out.iter().any(|b| b == block)
                && !crate::services::svg_raster::svg_has_external_refs(block)
            {
                out.push(block.to_string());
            }
        }
        rest = &after[close..];
    }
    out
}

/// A tool whose deliverable IS an image URL answers in a line or two;
/// incidental image links ride in big payloads (API JSON, scraped HTML) that
/// shouldn't become a gallery.
const MAX_RESULT_URL_SCAN_BYTES: usize = 2048;

fn result_image_urls(result: &str) -> Vec<String> {
    if result.len() > MAX_RESULT_URL_SCAN_BYTES {
        return Vec::new();
    }
    text_image_urls(result)
}

/// Strips a leading bullet and ONE trailing parenthetical —
/// `- /out/cat.png (412.3 KB, image/png)`, the shape MCP image tools answer in.
/// Narrow on purpose: prose that merely names a file still previews nothing.
fn undecorate_path_line(line: &str) -> &str {
    let line = line.trim_start_matches(['-', '*', '•', '+']).trim_start();
    let line = match line.rfind(" (") {
        Some(i) if line.ends_with(')') => &line[..i],
        _ => line,
    };
    line.trim()
}

/// Image paths a tool result points at: aivo's `[image saved:]` trailer, plus
/// bare image-path lines — a glob/ls answering "show me x.png" ends there
/// without any read/write call to hook. Display-only, best-effort.
fn result_image_paths(result: &str) -> Vec<String> {
    let mut paths = saved_image_paths(result);
    let lines: Vec<&str> = result
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut decorated = 0usize;
    let bare: Vec<&str> = lines
        .iter()
        .filter_map(|l| {
            let stripped = undecorate_path_line(l);
            if stripped.is_empty() || !has_image_extension(stripped) {
                return None;
            }
            if stripped != *l {
                decorated += 1;
            }
            Some(stripped)
        })
        .collect();
    // Decorated lines (bullets, sizes) = the tool ANSWERING with images, so a
    // minority share still previews; plain path lines only count when the
    // whole result is image paths (a mixed listing stays quiet).
    let deliberate = if decorated > 0 {
        images_are_the_answer(bare.len(), lines.len())
    } else {
        !bare.is_empty() && bare.len() == lines.len() && lines.len() <= MAX_RESULT_PREVIEW_LINES
    };
    if bare.len() <= MAX_RESULT_PATH_PREVIEWS && deliberate {
        paths.extend(bare.iter().map(|s| s.to_string()));
    }
    paths.dedup();
    dedup_same_stem(paths)
}

pub(super) fn tool_call_name(content: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(content)
        .ok()?
        .get("name")?
        .as_str()
        .map(str::to_string)
}

/// `list_dir` navigates — images its listings name are never the answer.
/// Web results quote untrusted pages: badges are noise, and auto-GETting
/// page-chosen URLs would let a fetched page beacon.
const NO_RESULT_PREVIEW_TOOLS: &[&str] = &["list_dir", "web_fetch", "web_search"];

fn images_are_the_answer(image_lines: usize, total_lines: usize) -> bool {
    image_lines > 0
        && total_lines <= MAX_RESULT_PREVIEW_LINES
        && image_lines * MIN_IMAGE_LINE_SHARE_DENOM >= total_lines
}

/// Lines a result can hold and still be "about" the image it names.
const MAX_RESULT_PREVIEW_LINES: usize = 8;

const MIN_IMAGE_LINE_SHARE_DENOM: usize = 4;

/// Blocking tail of preview prep: rasterize-if-svg, decode, downscale, encode.
fn prepare_bytes(bytes: Vec<u8>) -> Option<EncodedPreview> {
    let bytes = if crate::services::svg_raster::looks_like_svg(&bytes) {
        crate::services::svg_raster::rasterize_svg(&bytes)?
    } else {
        bytes
    };
    terminal_graphics::prepare_preview(&bytes)
}

/// Blocking half of preview prep for local sources.
pub(super) fn prepare_preview_source(source: PreviewSource) -> Option<EncodedPreview> {
    use base64::Engine as _;
    let bytes = match source {
        PreviewSource::InlineB64(data) => base64::engine::general_purpose::STANDARD
            .decode(data.as_bytes())
            .ok()?,
        PreviewSource::File(path) => {
            let meta = std::fs::metadata(&path).ok()?;
            if meta.len() > MAX_SOURCE_BYTES {
                return None;
            }
            std::fs::read(&path).ok()?
        }
        PreviewSource::Raw(bytes) => bytes,
    };
    prepare_bytes(bytes)
}

fn preview_caps(protocol: Protocol) -> (u16, u16) {
    if protocol == Protocol::HalfBlocks {
        (MAX_HALF_BLOCK_COLS, MAX_HALF_BLOCK_ROWS)
    } else {
        (MAX_PREVIEW_COLS, MAX_PREVIEW_ROWS)
    }
}

fn preview_col_budget(width: u16, protocol: Protocol) -> u16 {
    preview_caps(protocol).0.min(width.saturating_sub(6))
}

pub(super) fn preview_grid(
    px_w: u32,
    px_h: u32,
    width: u16,
    protocol: Protocol,
) -> Option<(u16, u16)> {
    pane_preview_grid(
        px_w,
        px_h,
        preview_col_budget(width, protocol),
        preview_caps(protocol).1,
        protocol,
    )
}

fn half_block_pixels(preview: &EncodedPreview, cols: u16, rows: u16) -> Option<Vec<u8>> {
    let thumb = preview.thumb.as_ref()?;
    Some(crate::services::image_optimize::resample_rgb_exact(
        &thumb.rgb,
        thumb.w,
        thumb.h,
        u32::from(cols),
        u32::from(rows) * 2,
    ))
}

/// One cell-row of a preview: placeholders, half-blocks, or a reserved ZWSP.
fn preview_row_line(
    key: u64,
    row: u16,
    cols: u16,
    protocol: Protocol,
    pixel_grid: Option<&[u8]>,
) -> Line<'static> {
    match protocol {
        Protocol::KittyVirtual => {
            let (r, g, b) = terminal_graphics::placeholder_fg(image_id(key));
            Line::from(Span::styled(
                terminal_graphics::placeholder_row(row, cols),
                Style::default().fg(Color::Rgb(r, g, b)),
            ))
        }
        Protocol::HalfBlocks => half_block_row(pixel_grid.unwrap_or(&[]), row, cols),
        _ => Line::from(Span::raw(RESERVED_ROW)),
    }
}

fn extend_preview_row(
    line: &mut Line<'static>,
    pad_cols: u16,
    key: u64,
    row: u16,
    cols: u16,
    protocol: Protocol,
    pixel_grid: Option<&[u8]>,
) {
    if pad_cols > 0 {
        line.spans
            .push(Span::raw(" ".repeat(usize::from(pad_cols))));
    }
    line.spans
        .extend(preview_row_line(key, row, cols, protocol, pixel_grid).spans);
}

fn band_row_width(anchors: &[ImageAnchor], row: u16) -> u16 {
    anchors
        .iter()
        .filter(|a| row < a.rows)
        .map(|a| a.col_offset.saturating_add(a.cols))
        .max()
        .unwrap_or(0)
}

fn reserved_preview_line(line: Line<'static>, images: Vec<ImageAnchor>) -> StyledLine {
    StyledLine {
        line,
        plain: RESERVED_ROW.to_string(),
        images,
        no_wrap: true,
    }
}

fn push_preview_rows(
    lines: &mut Vec<StyledLine>,
    key: u64,
    preview: &EncodedPreview,
    width: u16,
    protocol: Protocol,
) {
    let Some((cols, rows)) = preview_grid(preview.px_w, preview.px_h, width, protocol) else {
        return;
    };
    let pixel_grid = match protocol {
        Protocol::HalfBlocks => match half_block_pixels(preview, cols, rows) {
            Some(grid) => Some(grid),
            None => return,
        },
        _ => None,
    };
    for row in 0..rows {
        lines.push(reserved_preview_line(
            preview_row_line(key, row, cols, protocol, pixel_grid.as_deref()),
            if row == 0 {
                vec![ImageAnchor {
                    key,
                    cols,
                    rows,
                    col_offset: 0,
                }]
            } else {
                Vec::new()
            },
        ));
    }
}

fn last_preview_band(lines: &[StyledLine]) -> Option<(usize, usize)> {
    let last = lines.last()?;
    if last.plain != RESERVED_ROW {
        return None;
    }
    let mut start = lines.len() - 1;
    while start > 0 && lines[start - 1].plain == RESERVED_ROW {
        start -= 1;
    }
    if lines[start].images.is_empty() {
        return None;
    }
    Some((start, lines.len()))
}

fn try_pack_preview(
    lines: &mut Vec<StyledLine>,
    key: u64,
    preview: &EncodedPreview,
    width: u16,
    protocol: Protocol,
) -> bool {
    let Some((band_start, band_end)) = last_preview_band(lines) else {
        return false;
    };
    let used = band_row_width(&lines[band_start].images, 0);
    let remain = match preview_col_budget(width, protocol)
        .checked_sub(used.saturating_add(PREVIEW_PACK_GAP))
    {
        Some(remain) if remain >= 4 => remain,
        _ => return false,
    };
    let Some((cols, rows)) = pane_preview_grid(
        preview.px_w,
        preview.px_h,
        remain,
        preview_caps(protocol).1,
        protocol,
    ) else {
        return false;
    };
    let pixel_grid = match protocol {
        Protocol::HalfBlocks => match half_block_pixels(preview, cols, rows) {
            Some(grid) => Some(grid),
            None => return false,
        },
        _ => None,
    };
    let col_offset = used.saturating_add(PREVIEW_PACK_GAP);
    let band_len = band_end - band_start;
    let placed = lines[band_start].images.clone();
    for row in 0..rows {
        let pad = col_offset.saturating_sub(band_row_width(&placed, row));
        let idx = band_start + usize::from(row);
        if usize::from(row) < band_len {
            extend_preview_row(
                &mut lines[idx].line,
                pad,
                key,
                row,
                cols,
                protocol,
                pixel_grid.as_deref(),
            );
        } else {
            let mut line = Line::default();
            extend_preview_row(
                &mut line,
                pad,
                key,
                row,
                cols,
                protocol,
                pixel_grid.as_deref(),
            );
            lines.push(reserved_preview_line(line, Vec::new()));
        }
    }
    lines[band_start].images.push(ImageAnchor {
        key,
        cols,
        rows,
        col_offset,
    });
    true
}

/// Clip `r` to the window. `None` when the top is off-screen — a clipped
/// placement would index the wrong slice.
fn image_place_rows(row: usize, rows: u16, view_start: usize, view_rows: usize) -> Option<u16> {
    if row < view_start {
        return None;
    }
    let view_end = view_start.saturating_add(view_rows);
    if row >= view_end {
        return None;
    }
    let fit = view_end - row;
    let shown = usize::from(rows).min(fit);
    u16::try_from(shown).ok().filter(|&n| n > 0)
}

fn suppress_partial_image_placeholders(
    visible: &mut [Line<'static>],
    image_rows: &[(usize, ImageAnchor)],
    view_start: usize,
    view_rows: usize,
    body_rows: usize,
) {
    let view_end = view_start.saturating_add(view_rows);
    for &(row, anchor) in image_rows {
        if image_place_rows(row, anchor.rows, view_start, view_rows).is_some() {
            continue;
        }
        let end = row.saturating_add(usize::from(anchor.rows));
        let from = row.max(view_start);
        let to = end.min(view_end).min(body_rows);
        for abs in from..to {
            let vis_idx = abs.saturating_sub(view_start);
            let Some(line) = visible.get_mut(vis_idx) else {
                continue;
            };
            blank_virtual_placeholders(line, anchor.key);
        }
    }
}

fn blank_virtual_placeholders(line: &mut Line<'static>, key: u64) {
    let (r, g, b) = terminal_graphics::placeholder_fg(image_id(key));
    let target = Color::Rgb(r, g, b);
    for span in &mut line.spans {
        if span.style.fg == Some(target) {
            let cols = span
                .content
                .chars()
                .filter(|&c| c == PLACEHOLDER_CHAR)
                .count();
            if cols > 0 {
                *span = Span::raw(" ".repeat(cols));
            }
        }
    }
}

pub(super) fn pane_preview_grid(
    px_w: u32,
    px_h: u32,
    max_cols: u16,
    max_rows: u16,
    protocol: Protocol,
) -> Option<(u16, u16)> {
    if px_w == 0 || px_h == 0 || max_cols < 4 || max_rows < 2 {
        return None;
    }
    let px_per_col = if protocol == Protocol::HalfBlocks {
        1
    } else {
        MIN_SOURCE_PX_PER_COL
    };
    let source_cap = u16::try_from((px_w / px_per_col).max(4)).unwrap_or(u16::MAX);
    let aspect = f64::from(px_h) / f64::from(px_w);
    let mut cols = max_cols.min(source_cap);
    let mut rows = (aspect * f64::from(cols) * CELL_WIDTH_OVER_HEIGHT)
        .round()
        .max(1.0) as u16;
    if rows > max_rows {
        rows = max_rows;
        cols = (f64::from(rows) / CELL_WIDTH_OVER_HEIGHT / aspect)
            .round()
            .max(1.0) as u16;
        cols = cols.min(max_cols);
    }
    Some((cols, rows.max(1)))
}

/// One cell row of `▀` half-blocks from a cols-wide RGB grid: fg = the cell's
/// upper pixel, bg = the lower.
pub(super) fn half_block_row(grid: &[u8], row: u16, cols: u16) -> Line<'static> {
    let pixel = |x: u32, y: u32| -> Color {
        let i = ((y * u32::from(cols) + x) * 3) as usize;
        match grid.get(i..i + 3) {
            Some(p) => Color::Rgb(p[0], p[1], p[2]),
            None => Color::Reset,
        }
    };
    let spans = (0..u32::from(cols))
        .map(|col| {
            Span::styled(
                "▀",
                Style::default()
                    .fg(pixel(col, u32::from(row) * 2))
                    .bg(pixel(col, u32::from(row) * 2 + 1)),
            )
        })
        .collect::<Vec<_>>();
    Line::from(spans)
}

impl CodeTuiApp {
    /// Base dir for tool-relative paths — the agent's real working dir, like
    /// the transcript's path display.
    pub(super) fn preview_base(&self) -> &str {
        if self.real_cwd.is_empty() {
            &self.cwd
        } else {
            &self.real_cwd
        }
    }

    pub(super) fn ready_preview(&self, key: u64) -> Option<Arc<EncodedPreview>> {
        match self.inline_images.previews.get(&key) {
            Some(PreviewSlot::Ready(preview)) => Some(Arc::clone(preview)),
            _ => None,
        }
    }

    /// Preview key for a mention, frozen on first resolution (see `pinned`).
    fn lookup_pinned(&self, idx: usize, content: &str, path: &str) -> Option<u64> {
        self.inline_images
            .pinned
            .get(&pin_id(idx, content, path))
            .copied()
    }

    /// Spawns prep jobs for every previewable image the transcript mentions
    /// that hasn't been seen yet, pinning each mention to the file state it
    /// resolves to NOW. Runs on transcript rebuild, so completion (which
    /// bumps `transcript_revision`) can't loop it.
    pub(super) fn queue_missing_previews(&mut self) {
        if !self.inline_images.caps.enabled() {
            return;
        }
        let base = self.preview_base().to_string();
        // (mention pin, resolved path) — resolved to preview keys below.
        let mut pin_wants: Vec<(u64, PathBuf)> = Vec::new();
        let mut inline_wants: Vec<(u64, String)> = Vec::new();
        // (content key, svg markup) — pasted in a message, no file to stat.
        let mut inline_svg_wants: Vec<(u64, String)> = Vec::new();
        // (mention pin, url) — keyed on the URL string, fetched async.
        let mut url_wants: Vec<(u64, String)> = Vec::new();
        for (idx, message) in self.history.iter().enumerate() {
            let content = &message.content;
            let want_path = |pin_wants: &mut Vec<(u64, PathBuf)>, path: &str| {
                pin_wants.push((pin_id(idx, content, path), resolve_in(&base, path)));
            };
            let want_urls = |url_wants: &mut Vec<(u64, String)>, urls: Vec<String>| {
                for url in urls {
                    url_wants.push((pin_id(idx, content, &url), url));
                }
            };
            match message.role.as_str() {
                "user" => {
                    for attachment in &message.attachments {
                        if !attachment.is_image() {
                            continue;
                        }
                        match &attachment.storage {
                            AttachmentStorage::Inline { data } => {
                                // Hash before cloning: a multi-MB payload is
                                // only copied while its prep is outstanding.
                                let key = hash_inline(data);
                                if !self.inline_images.previews.contains_key(&key) {
                                    inline_wants.push((key, data.clone()));
                                }
                            }
                            AttachmentStorage::FileRef { path } => {
                                want_path(&mut pin_wants, path);
                            }
                        }
                    }
                    for path in text_image_paths(content) {
                        want_path(&mut pin_wants, &path);
                    }
                    want_urls(&mut url_wants, text_image_urls(content));
                }
                "assistant" => {
                    for path in text_image_paths(content) {
                        want_path(&mut pin_wants, &path);
                    }
                    want_urls(&mut url_wants, text_image_urls(content));
                    for block in inline_svg_blocks(content) {
                        let key = hash_inline(&block);
                        if !self.inline_images.previews.contains_key(&key) {
                            inline_svg_wants.push((key, block));
                        }
                    }
                }
                "tool_call" => {
                    if let Some(path) = file_tool_image_target(content) {
                        // A write/edit stats post-execution state: wait for
                        // evidence the call ran (a following entry or an
                        // inlined result), or the pre-write file gets pinned.
                        let ran = idx + 1 < self.history.len()
                            || decode_tool_outcome(content).0.is_some();
                        if ran {
                            want_path(&mut pin_wants, &path);
                        }
                    }
                }
                "tool_result" => {
                    let listing = idx
                        .checked_sub(1)
                        .and_then(|i| self.history.get(i))
                        .filter(|m| m.role == "tool_call")
                        .and_then(|m| tool_call_name(&m.content))
                        .is_some_and(|n| NO_RESULT_PREVIEW_TOOLS.contains(&n.as_str()));
                    if listing {
                        continue;
                    }
                    for path in result_image_paths(content) {
                        want_path(&mut pin_wants, &path);
                    }
                    want_urls(&mut url_wants, result_image_urls(content));
                }
                _ => {}
            }
        }
        let tx = self.tx.clone();
        let spawn = move |images: &mut InlineImageState, key: u64, source: PreviewSource| {
            if images.previews.contains_key(&key) {
                return;
            }
            images.previews.insert(key, PreviewSlot::Pending);
            let tx = tx.clone();
            tokio::task::spawn_blocking(move || {
                let preview = prepare_preview_source(source).map(Box::new);
                let _ = tx.send(RuntimeEvent::ImagePreviewReady { key, preview });
            });
        };
        for (pin, url) in url_wants {
            let key = *self
                .inline_images
                .pinned
                .entry(pin)
                .or_insert_with(|| hash_url(&url));
            self.spawn_url_preview(key, url);
        }
        for (pin, path) in pin_wants {
            let key = match self.inline_images.pinned.get(&pin) {
                Some(&key) => key,
                None => {
                    let Some(key) = file_key(&path) else { continue };
                    self.inline_images.pinned.insert(pin, key);
                    key
                }
            };
            spawn(&mut self.inline_images, key, PreviewSource::File(path));
        }
        for (key, data) in inline_wants {
            spawn(&mut self.inline_images, key, PreviewSource::InlineB64(data));
        }
        for (key, block) in inline_svg_wants {
            spawn(
                &mut self.inline_images,
                key,
                PreviewSource::Raw(block.into_bytes()),
            );
        }
    }

    /// Downloads an image URL and preps it. Display-only, size-capped, and
    /// timed out; a failure just leaves the block unrendered.
    pub(super) fn spawn_url_preview(&mut self, key: u64, url: String) {
        if self.inline_images.previews.contains_key(&key) {
            return;
        }
        self.inline_images
            .previews
            .insert(key, PreviewSlot::Pending);
        let client = self.client.clone();
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let fetch = async {
                let response = client
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(20))
                    .send()
                    .await
                    .ok()?;
                if !response.status().is_success() {
                    return None;
                }
                let bytes = response.bytes().await.ok()?;
                if bytes.len() as u64 > MAX_SOURCE_BYTES {
                    return None;
                }
                tokio::task::spawn_blocking(move || prepare_bytes(bytes.to_vec()))
                    .await
                    .ok()
                    .flatten()
            };
            let preview = fetch.await.map(Box::new);
            let _ = tx.send(RuntimeEvent::ImagePreviewReady { key, preview });
        });
    }

    /// Emits one preview block unless `seen` already holds the picture — a
    /// file mentioned, then read, then globbed must render once per turn.
    /// Dedup is by CONTENT hash, not source key: a generated image previewed
    /// from its URL and again from its downloaded file is one picture.
    fn push_preview_once(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        key: u64,
        width: u16,
    ) {
        let Some(preview) = self.ready_preview(key) else {
            return;
        };
        if !seen.insert(preview.content_hash) {
            return;
        }
        let protocol = self.inline_images.caps.protocol;
        if try_pack_preview(lines, key, &preview, width, protocol) {
            return;
        }
        // Back-to-back blocks read as one glued picture.
        if lines.last().is_some_and(|l| l.plain == RESERVED_ROW) {
            push_styled_line(lines, "", Style::default());
        }
        push_preview_rows(lines, key, &preview, width, protocol);
    }

    pub(super) fn push_attachment_preview_lines(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        idx: usize,
        content: &str,
        attachments: &[MessageAttachment],
        width: u16,
    ) {
        if !self.inline_images.caps.enabled() {
            return;
        }
        for attachment in attachments {
            if !attachment.is_image() {
                continue;
            }
            let key = match &attachment.storage {
                // Inline data is immutable — content-addressed, no pin needed.
                AttachmentStorage::Inline { data } => Some(hash_inline(data)),
                AttachmentStorage::FileRef { path } => self.lookup_pinned(idx, content, path),
            };
            if let Some(key) = key {
                self.push_preview_once(lines, seen, key, width);
            }
        }
    }

    /// Previews for the given path/URL mentions, keyed by the pins
    /// `queue_missing_previews` resolved.
    fn push_mention_preview_lines(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        idx: usize,
        content: &str,
        mentions: Vec<String>,
        width: u16,
    ) {
        if !self.inline_images.caps.enabled() {
            return;
        }
        for mention in mentions {
            if let Some(key) = self.lookup_pinned(idx, content, &mention) {
                self.push_preview_once(lines, seen, key, width);
            }
        }
    }

    /// A user message's deferred path mention as its own block; the pin identity
    /// stays with the user message, only the position moves.
    pub(super) fn push_deferred_mention(
        &self,
        lines: &mut Vec<StyledLine>,
        bars: &mut Vec<Option<Color>>,
        seen: &mut HashSet<u64>,
        idx: usize,
        width: u16,
    ) {
        let Some(content) = self.history.get(idx).map(|m| m.content.clone()) else {
            return;
        };
        let mut extra = Vec::new();
        self.push_text_image_preview_lines(&mut extra, seen, idx, &content, width);
        super::push_block(lines, bars, extra, Some(super::role_bar_color("user")));
    }

    /// Previews for image paths mentioned in a user or assistant message —
    /// "show octagon.svg" is often answered from context with no tool call,
    /// so the message text is the only anchor.
    pub(super) fn push_text_image_preview_lines(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        idx: usize,
        content: &str,
        width: u16,
    ) {
        let mut mentions = text_image_paths(content);
        mentions.extend(text_image_urls(content));
        self.push_mention_preview_lines(lines, seen, idx, content, mentions, width);
        if !self.inline_images.caps.enabled() {
            return;
        }
        // Failed rasterization surfaces like the tool path — the block only
        // exists because the model produced an image.
        for block in inline_svg_blocks(content) {
            let key = hash_inline(&block);
            match self.inline_images.previews.get(&key) {
                Some(PreviewSlot::Ready(_)) => self.push_preview_once(lines, seen, key, width),
                Some(PreviewSlot::Failed) => push_styled_line(
                    lines,
                    "  ⚠ image preview failed (svg needs rsvg-convert or qlmanage)",
                    Style::default().fg(FAINT()),
                ),
                _ => {}
            }
        }
    }

    pub(super) fn push_tool_image_preview_lines(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        idx: usize,
        tool_call_content: &str,
        width: u16,
    ) {
        if !self.inline_images.caps.enabled() {
            return;
        }
        let Some(path) = file_tool_image_target(tool_call_content) else {
            return;
        };
        let Some(key) = self.lookup_pinned(idx, tool_call_content, &path) else {
            return;
        };
        match self.inline_images.previews.get(&key) {
            Some(PreviewSlot::Ready(_)) => self.push_preview_once(lines, seen, key, width),
            // Surface SVG failures (rasterizer missing, parse error) instead
            // of silently showing nothing — this row only exists because an
            // image file was touched, so it never nags on ordinary tool calls.
            Some(PreviewSlot::Failed) => push_styled_line(
                lines,
                "  ⚠ image preview failed (svg needs rsvg-convert or qlmanage)",
                Style::default().fg(FAINT()),
            ),
            _ => {}
        }
    }

    pub(super) fn push_saved_image_preview_lines(
        &self,
        lines: &mut Vec<StyledLine>,
        seen: &mut HashSet<u64>,
        idx: usize,
        result: &str,
        tool: Option<&str>,
        width: u16,
    ) {
        if tool.is_some_and(|t| NO_RESULT_PREVIEW_TOOLS.contains(&t)) {
            return;
        }
        let mut mentions = result_image_paths(result);
        mentions.extend(result_image_urls(result));
        self.push_mention_preview_lines(lines, seen, idx, result, mentions, width);
    }

    /// Ensures `key`'s data is transmitted, appending to `seq`; returns false
    /// while the preview isn't ready. Evicts LRU images beyond the cap, never
    /// one still wanted this frame.
    fn ensure_transmitted(
        &mut self,
        seq: &mut String,
        key: u64,
        wanted: &[PlacedImage],
        tmux: bool,
    ) -> bool {
        if self.inline_images.transmitted.contains_key(&key) {
            return true;
        }
        let Some(preview) = self.ready_preview(key) else {
            return false;
        };
        terminal_graphics::transmit(seq, image_id(key), &preview, tmux);
        self.inline_images.transmitted.insert(key, (0, 0));
        self.inline_images.transmit_order.push_back(key);
        while self.inline_images.transmit_order.len() > MAX_TRANSMITTED {
            let Some(old) = self.inline_images.transmit_order.pop_front() else {
                break;
            };
            if wanted.iter().any(|d| d.key == old) {
                self.inline_images.transmit_order.push_back(old);
                break;
            }
            self.inline_images.transmitted.remove(&old);
            terminal_graphics::delete_image(seq, image_id(old), tmux);
        }
        true
    }

    fn sixel_for(
        &mut self,
        key: u64,
        cols: u16,
        grid_rows: u16,
        shown: u16,
    ) -> Option<Arc<String>> {
        let cache_key = (key, cols, grid_rows, shown);
        if let Some(data) = self.inline_images.sixel_cache.get(&cache_key) {
            return Some(Arc::clone(data));
        }
        let preview = self.ready_preview(key)?;
        let thumb = preview.thumb.as_ref()?;
        let (cell_w, cell_h) = self.inline_images.caps.cell_px;
        let px_w = u32::from(cols) * u32::from(cell_w);
        let px_h = u32::from(grid_rows) * u32::from(cell_h);
        let mut rgb = crate::services::image_optimize::resample_rgb_exact(
            &thumb.rgb, thumb.w, thumb.h, px_w, px_h,
        );
        // tmux rounds a sixel's height up to 6-px bands — unaligned rasters spill a row.
        let shown_h = u32::from(shown.min(grid_rows)) * u32::from(cell_h);
        let shown_h = (shown_h / 6 * 6).max(shown_h.min(6));
        rgb.truncate((px_w * shown_h * 3) as usize);
        let data = Arc::new(terminal_graphics::sixel_encode(&rgb, px_w, shown_h));
        self.inline_images
            .sixel_cache
            .insert(cache_key, Arc::clone(&data));
        Some(data)
    }

    /// Emits whatever the terminal still needs for this frame's `desired` set
    /// (which each render rebuilds). Called after `terminal.draw`, still inside
    /// the synchronized update, so images and cells land as one visual frame.
    /// Returns true when another frame is needed (see `flush_sixel`).
    ///
    /// Virtual mode sends data + a virtual placement once per image (and again
    /// on a grid-size change); position comes from the placeholder cells the
    /// frame itself painted. Classic mode diffs cursor-addressed placements
    /// against what's on the terminal.
    pub(super) fn flush_inline_images(&mut self, out: &mut impl std::io::Write) -> bool {
        if !self.inline_images.caps.emits_escapes() {
            return false;
        }
        if self.inline_images.caps.protocol == Protocol::Sixel {
            return self.flush_sixel(out);
        }
        let tmux = self.inline_images.caps.tmux;
        let desired = std::mem::take(&mut self.inline_images.desired);
        let mut seq = String::new();
        if self.inline_images.caps.virtual_placement() {
            for want in &desired {
                if !self.ensure_transmitted(&mut seq, want.key, &desired, tmux) {
                    continue;
                }
                let grid = (want.cols, want.grid_rows);
                if self.inline_images.transmitted.get(&want.key) != Some(&grid) {
                    terminal_graphics::create_virtual_placement(
                        &mut seq,
                        image_id(want.key),
                        want.cols,
                        want.grid_rows,
                        tmux,
                    );
                    self.inline_images.transmitted.insert(want.key, grid);
                }
            }
        } else if desired != self.inline_images.placed {
            let old_placed = std::mem::take(&mut self.inline_images.placed);
            for old in &old_placed {
                if !desired.contains(old) {
                    terminal_graphics::delete_placement(
                        &mut seq,
                        image_id(old.key),
                        placement_id(old),
                        tmux,
                    );
                }
            }
            let mut new_placed = Vec::with_capacity(desired.len());
            for want in &desired {
                if old_placed.contains(want) {
                    new_placed.push(*want);
                    continue;
                }
                if !self.ensure_transmitted(&mut seq, want.key, &desired, tmux) {
                    continue;
                }
                let src_h = (want.rows < want.grid_rows)
                    .then(|| self.ready_preview(want.key))
                    .flatten()
                    .map(|p| (p.px_h * u32::from(want.rows) / u32::from(want.grid_rows)).max(1));
                terminal_graphics::place(
                    &mut seq,
                    image_id(want.key),
                    placement_id(want),
                    want.x,
                    want.y,
                    want.cols,
                    want.rows,
                    src_h,
                    tmux,
                );
                new_placed.push(*want);
            }
            self.inline_images.placed = new_placed;
        }
        if self.inline_images.caps.virtual_placement() {
            if !seq.is_empty() {
                let _ = out.write_all(seq.as_bytes());
            }
        } else {
            write_restoring_cursor(out, &seq);
        }
        false
    }

    /// Sixel placement diff: emit new images at their cursor address (tmux
    /// composes them as pane content). Sixel has no delete command — when
    /// anything disappears or moves, DON'T draw this frame: queue the stale
    /// rects for a cell rewrite that erases the pixels; the flush after it
    /// re-places only what's missing. Returns true for that extra frame.
    fn flush_sixel(&mut self, out: &mut impl std::io::Write) -> bool {
        let desired = std::mem::take(&mut self.inline_images.desired);
        if desired == self.inline_images.placed {
            return false;
        }
        let removed: Vec<PlacedImage> = self
            .inline_images
            .placed
            .iter()
            .filter(|old| !desired.contains(old))
            .copied()
            .collect();
        if !removed.is_empty() {
            self.inline_images
                .placed
                .retain(|old| desired.contains(old));
            self.inline_images.pending_clears.extend(removed);
            return true;
        }
        let mut seq = String::new();
        let mut new_placed = Vec::with_capacity(desired.len());
        for want in &desired {
            if self.inline_images.placed.contains(want) {
                new_placed.push(*want);
                continue;
            }
            let Some(data) = self.sixel_for(want.key, want.cols, want.grid_rows, want.rows) else {
                continue;
            };
            terminal_graphics::sixel_place(&mut seq, want.x, want.y, &data);
            new_placed.push(*want);
        }
        self.inline_images.placed = new_placed;
        write_restoring_cursor(out, &seq);
        false
    }

    /// The original behind a preview key: the pinned file for path mentions,
    /// the URL itself for remote previews (the browser shows full quality),
    /// or a temp-materialized copy for inline (pasted) attachments. Walks the
    /// history instead of keeping a reverse map — clicks are rare.
    fn find_open_target(&self, key: u64) -> Option<String> {
        let base = self.preview_base().to_string();
        for (idx, message) in self.history.iter().enumerate() {
            let content = &message.content;
            let check = |path: &str| -> Option<String> {
                (self.inline_images.pinned.get(&pin_id(idx, content, path)) == Some(&key))
                    .then(|| resolve_in(&base, path).to_string_lossy().to_string())
            };
            let check_all =
                |paths: Vec<String>| -> Option<String> { paths.iter().find_map(|p| check(p)) };
            let check_urls = |urls: Vec<String>| -> Option<String> {
                urls.into_iter().find(|url| {
                    self.inline_images.pinned.get(&pin_id(idx, content, url)) == Some(&key)
                })
            };
            let found = match message.role.as_str() {
                "user" => {
                    for attachment in &message.attachments {
                        if !attachment.is_image() {
                            continue;
                        }
                        match &attachment.storage {
                            AttachmentStorage::Inline { data } if hash_inline(data) == key => {
                                return materialize_inline(key, data, &attachment.mime_type)
                                    .map(|p| p.to_string_lossy().to_string());
                            }
                            AttachmentStorage::FileRef { path } => {
                                if let Some(p) = check(path) {
                                    return Some(p);
                                }
                            }
                            _ => {}
                        }
                    }
                    check_all(text_image_paths(content))
                        .or_else(|| check_urls(text_image_urls(content)))
                }
                "assistant" => check_all(text_image_paths(content))
                    .or_else(|| check_urls(text_image_urls(content))),
                "tool_call" => file_tool_image_target(content).and_then(|p| check(&p)),
                "tool_result" => check_all(result_image_paths(content))
                    .or_else(|| check_urls(result_image_urls(content))),
                _ => None,
            };
            if found.is_some() {
                return found;
            }
        }
        None
    }

    /// Opens the image under a clicked transcript cell in the OS viewer.
    /// Returns true when the click landed on the picture's column extent —
    /// claimed even on failure, so such a click never starts a stray
    /// drag-select; the blank space right of the image stays ordinary
    /// transcript.
    pub(super) fn open_image_at(&mut self, row: usize, column: u16) -> bool {
        if !self.inline_images.caps.enabled() {
            return false;
        }
        let Some(anchor) = self
            .render_cache
            .transcript
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
            .and_then(|wrapped| {
                wrapped
                    .image_rows
                    .iter()
                    .find(|&&(r, a)| {
                        if row < r || row >= r + usize::from(a.rows) {
                            return false;
                        }
                        // `col_offset` is band-relative; `+ SUB_BLOCK_INDENT` is
                        // the usual nest slop so indented and flush bands both hit.
                        let start = a.col_offset;
                        let end = start
                            .saturating_add(a.cols)
                            .saturating_add(SUB_BLOCK_INDENT);
                        column >= start && column < end
                    })
                    .map(|&(_, a)| a)
            })
        else {
            return false;
        };
        let Some(target) = self.find_open_target(anchor.key) else {
            self.show_toast("image source unavailable");
            return true;
        };
        let name = basename(&target);
        match crate::services::browser_open::open_url(&target) {
            Ok(()) => self.show_toast(format!("↗ opened {name}")),
            Err(err) => self.show_toast(format!("open failed: {err}")),
        }
        true
    }

    pub(super) fn collect_desired_inline_images(&mut self, area: Rect) {
        self.inline_images.desired.clear();
        if !self.inline_images.caps.enabled() {
            return;
        }
        // Mid-scroll sixel: leave `desired` empty so the flush erases the
        // anchors once; the pane is exempt — pushed later at a fixed spot.
        if self.sixel_scroll_hold() {
            return;
        }
        let Some(body) = self
            .render_cache
            .transcript
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
        else {
            return;
        };
        let virtual_placement = self.inline_images.caps.virtual_placement();
        let cell_painted = self.inline_images.caps.protocol == Protocol::HalfBlocks;
        let view_start = self.transcript_scroll;
        let view_rows = usize::from(area.height);
        for &(row, anchor) in &body.image_rows {
            let end = row + usize::from(anchor.rows);
            let indent = SUB_BLOCK_INDENT.saturating_add(anchor.col_offset);
            let cols = anchor.cols.min(area.width.saturating_sub(indent));
            if cols < 4 {
                continue;
            }
            let on_screen = end > view_start && row < view_start + view_rows;
            let rows = if cell_painted {
                if !on_screen {
                    continue;
                }
                anchor.rows
            } else {
                match image_place_rows(row, anchor.rows, view_start, view_rows) {
                    Some(rows) => rows,
                    None => continue,
                }
            };
            let place = if virtual_placement {
                Some((0, 0, anchor.cols))
            } else {
                Some((
                    area.x.saturating_add(indent),
                    area.y.saturating_add((row - view_start) as u16),
                    cols,
                ))
            };
            if let Some((x, y, cols)) = place {
                self.inline_images.desired.push(PlacedImage {
                    key: anchor.key,
                    x,
                    y,
                    cols,
                    rows,
                    grid_rows: anchor.rows,
                });
            }
        }
    }

    pub(super) fn suppress_partial_inline_images(
        &self,
        visible: &mut [Line<'static>],
        view_start: usize,
        view_rows: usize,
    ) {
        if !self.inline_images.caps.virtual_placement() {
            return;
        }
        let Some(body) = self
            .render_cache
            .transcript
            .as_ref()
            .and_then(|cache| cache.wrapped.as_ref())
        else {
            return;
        };
        suppress_partial_image_placeholders(
            visible,
            &body.image_rows,
            view_start,
            view_rows,
            body.rows.len(),
        );
    }

    /// Sixel lives in tmux's pane-content layer: a full cell repaint (self-
    /// heal, settle, resize) erases it. Forgetting `placed` makes the next
    /// flush re-emit whatever is visible; kitty modes keep their images
    /// across cell redraws, so they're untouched.
    pub(super) fn note_cells_repainted(&mut self) {
        if self.inline_images.caps.protocol == Protocol::Sixel {
            self.inline_images.placed.clear();
            self.inline_images.pending_clears.clear();
        }
    }

    /// `AlwaysUpdate` bypasses the unchanged-cell diff shortcut, so the cells
    /// under a vanished placement are re-emitted even when blank — erasing the
    /// pixels without a full-screen repaint. (The next frame re-emits the rect
    /// once more: `diff_option` participates in cell equality. Harmless.)
    pub(super) fn mark_sixel_clear_cells(&mut self, buffer: &mut ratatui::buffer::Buffer) {
        for stale in std::mem::take(&mut self.inline_images.pending_clears) {
            for y in stale.y..stale.y.saturating_add(stale.rows) {
                for x in stale.x..stale.x.saturating_add(stale.cols) {
                    if let Some(cell) = buffer.cell_mut(ratatui::layout::Position::new(x, y)) {
                        cell.set_diff_option(ratatui::buffer::CellDiffOption::AlwaysUpdate);
                    }
                }
            }
        }
    }

    /// Terminals may drop transmitted image data on a window resize; the
    /// repainted placeholder cells then composite nothing.
    pub(super) fn note_image_resize(&mut self) {
        if self.inline_images.caps.emits_escapes() {
            self.inline_images.resize_settle = Some(std::time::Instant::now());
        }
    }

    /// Sixel only: `true` = the scroll is moving, withhold transcript placements.
    pub(super) fn sixel_scroll_hold(&mut self) -> bool {
        if self.inline_images.caps.protocol != Protocol::Sixel {
            return false;
        }
        if self.transcript_scroll != self.inline_images.last_scroll {
            self.inline_images.last_scroll = self.transcript_scroll;
            self.inline_images.scroll_settle = Some(std::time::Instant::now());
        }
        self.inline_images
            .scroll_settle
            .is_some_and(|at| at.elapsed() < SCROLL_SETTLE)
    }

    /// The scroll rested — repaint once so the withheld placements return.
    pub(super) fn tick_image_scroll_settle(&mut self) -> bool {
        let settled = self
            .inline_images
            .scroll_settle
            .is_some_and(|at| at.elapsed() >= SCROLL_SETTLE);
        if !settled {
            return false;
        }
        self.inline_images.scroll_settle = None;
        true
    }

    /// Resize settled: forget terminal-side state so the flush re-sends.
    pub(super) fn tick_image_resize_settle(&mut self) -> bool {
        let settled = self
            .inline_images
            .resize_settle
            .is_some_and(|at| at.elapsed() >= RESIZE_SETTLE);
        if !settled {
            return false;
        }
        self.inline_images.resize_settle = None;
        self.reset_inline_image_terminal_state();
        self.pending_full_repaint = true;
        true
    }

    /// Forget the terminal-side state after something else owned the terminal
    /// (external $EDITOR); data and placements must be re-sent from scratch.
    pub(super) fn reset_inline_image_terminal_state(&mut self) {
        self.inline_images.placed.clear();
        self.inline_images.pending_clears.clear();
        self.inline_images.transmitted.clear();
        self.inline_images.transmit_order.clear();
    }

    /// Deferred half of the `/config` preview disable: delete the kitty-held
    /// images while the caps still permit the escapes, then drop the caps.
    pub(super) fn finish_inline_image_disable(&mut self, out: &mut impl std::io::Write) {
        if !std::mem::take(&mut self.pending_inline_image_cleanup) {
            return;
        }
        self.cleanup_inline_images(out);
        self.inline_images.caps = GraphicsCaps::default();
    }

    /// Frees every transmitted image before leaving the alternate screen.
    pub(super) fn cleanup_inline_images(&mut self, out: &mut impl std::io::Write) {
        if !self.inline_images.caps.kitty() || self.inline_images.transmitted.is_empty() {
            return;
        }
        let tmux = self.inline_images.caps.tmux;
        let mut seq = String::new();
        for (key, _) in self.inline_images.transmitted.drain() {
            terminal_graphics::delete_image(&mut seq, image_id(key), tmux);
        }
        self.inline_images.transmit_order.clear();
        self.inline_images.placed.clear();
        let _ = out.write_all(seq.as_bytes());
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::terminal_graphics::PixelFormat;

    #[test]
    fn grid_preserves_landscape_aspect() {
        // 2:1 landscape at 80 wide: 46 cols → 12 rows fits under the kitty cap.
        assert_eq!(
            preview_grid(1000, 500, 80, Protocol::KittyVirtual),
            Some((46, 12))
        );
        // Half-blocks trade screen space for resolution: bigger box.
        assert_eq!(
            preview_grid(1000, 500, 80, Protocol::HalfBlocks),
            Some((64, 16))
        );
    }

    #[test]
    fn grid_caps_portrait_by_rows() {
        let (cols, rows) = preview_grid(500, 1000, 80, Protocol::KittyVirtual).unwrap();
        assert_eq!(rows, MAX_PREVIEW_ROWS);
        // Box aspect must track the image: cols ≈ rows / 0.5 / 2 = 12.
        assert_eq!(cols, 12);
    }

    #[test]
    fn file_tool_image_target_matches_image_tools_only() {
        let write =
            serde_json::json!({"name": "write_file", "args": {"path": "pic.svg"}}).to_string();
        assert_eq!(file_tool_image_target(&write), Some("pic.svg".to_string()));
        let read =
            serde_json::json!({"name": "read_file", "args": {"path": "a/b.png"}}).to_string();
        assert!(file_tool_image_target(&read).is_none());
        let rs = serde_json::json!({"name": "read_file", "args": {"path": "main.rs"}}).to_string();
        assert!(file_tool_image_target(&rs).is_none());
        let grep = serde_json::json!({"name": "grep", "args": {"pattern": "x.svg"}}).to_string();
        assert!(file_tool_image_target(&grep).is_none());
        let generated = serde_json::json!({
            "name": "generate_image",
            "args": {"path": "icon.png", "prompt": "app icon"}
        })
        .to_string();
        assert_eq!(
            file_tool_image_target(&generated),
            Some("icon.png".to_string())
        );
    }

    /// Whitespace tokenization alone yields `rain](assets/cat.png)` — an
    /// impossible filename the stat gate drops, silently.
    #[test]
    fn text_image_paths_reads_markdown_image_syntax() {
        assert_eq!(
            text_image_paths("Here it is:\n![Cat in the rain](assets/cat-in-the-rain.png)"),
            vec!["assets/cat-in-the-rain.png"]
        );
        assert_eq!(
            text_image_paths("![cat](assets/cat.png)"),
            vec!["assets/cat.png"]
        );
        // Title / angle-bracket forms.
        assert_eq!(
            text_image_paths("![a](shots/a.png \"My title\")"),
            vec!["shots/a.png"]
        );
        assert_eq!(text_image_paths("[link](<b.jpeg>)"), vec!["b.jpeg"]);
        // Markdown + the same bare path dedup.
        assert_eq!(
            text_image_paths("![a](x.png) and also x.png"),
            vec!["x.png"]
        );
        // The extension gate still rules.
        assert!(text_image_paths("[docs](https://example.com/page)").is_empty());
    }

    #[test]
    fn text_image_urls_read_markdown_targets() {
        assert_eq!(
            text_image_urls("![remote](https://cdn.example/a.png)"),
            vec!["https://cdn.example/a.png"]
        );
    }

    /// Chinese sets punctuation flush against the path.
    #[test]
    fn text_image_paths_survives_cjk_punctuation() {
        assert_eq!(
            text_image_paths("SVG 已生成，路径：triangles-circles.svg"),
            vec!["triangles-circles.svg"]
        );
        assert_eq!(
            text_image_paths("就是这张。源文件在 triangles-circles.svg。"),
            vec!["triangles-circles.svg"]
        );
        assert_eq!(text_image_paths("生成了 cat.png，请查看"), vec!["cat.png"]);
        assert_eq!(
            text_image_paths("图片（shots/a.png）已保存"),
            vec!["shots/a.png"]
        );
    }

    #[test]
    fn same_stem_mentions_preview_once() {
        assert_eq!(
            text_image_paths("grid.svg is a 12×12 grid; preview: grid.png"),
            vec!["grid.svg"]
        );
        assert_eq!(
            text_image_paths("rendered out/pic.PNG from out/pic.svg"),
            vec!["out/pic.PNG"]
        );
        // Same basename in different dirs = different pictures.
        assert_eq!(
            text_image_paths("light/logo.png vs dark/logo.png"),
            vec!["light/logo.png", "dark/logo.png"]
        );
        assert_eq!(
            result_image_paths("a/one.svg\na/one.png\n"),
            vec!["a/one.svg".to_string()]
        );
    }

    #[test]
    fn text_image_paths_extracts_mentions() {
        assert_eq!(text_image_paths("show octagon.svg"), vec!["octagon.svg"]);
        assert_eq!(
            text_image_paths("render `shots/a.png` and (b.jpeg)!"),
            vec!["shots/a.png", "b.jpeg"]
        );
        // Dedup + cap; plain prose stays quiet.
        assert_eq!(text_image_paths("a.png a.png").len(), 1);
        assert_eq!(text_image_paths("w.png x.png y.png z.png").len(), 3);
        assert!(text_image_paths("no images mentioned here").is_empty());
    }

    #[test]
    fn inline_svg_blocks_extracts_fenced_and_bare_markup() {
        let bare = "here you go\n<svg width=\"10\"><circle r=\"4\"/></svg>\ndone";
        assert_eq!(
            inline_svg_blocks(bare),
            vec!["<svg width=\"10\"><circle r=\"4\"/></svg>"]
        );
        let fenced = "```svg\n<svg>\n<rect/>\n</svg>\n```";
        assert_eq!(inline_svg_blocks(fenced), vec!["<svg>\n<rect/>\n</svg>"]);
        // `<svgfoo>` is not an svg tag; unterminated markup waits (streaming).
        assert!(inline_svg_blocks("<svgfoo></svgfoo> <svgbar></svg>").is_empty());
        // A false `<svg…` hit must not swallow the real block after it.
        assert_eq!(
            inline_svg_blocks("<svgfoo> then <svg><rect/></svg>"),
            vec!["<svg><rect/></svg>"]
        );
        assert!(inline_svg_blocks("<svg width=\"10\"><circle").is_empty());
        let twice = "<svg><rect/></svg> and again <svg><rect/></svg>";
        assert_eq!(inline_svg_blocks(twice).len(), 1);
        let many = "<svg>1</svg><svg>2</svg><svg>3</svg>";
        assert_eq!(inline_svg_blocks(many).len(), MAX_INLINE_SVG_BLOCKS);
        assert!(inline_svg_blocks("plain prose, no markup").is_empty());
        // External refs never reach the rasterizer (qlmanage could fetch them).
        assert!(inline_svg_blocks("<svg><image href=\"http://evil/t.png\"/></svg>").is_empty());
    }

    #[test]
    fn text_image_urls_extracts_capped_image_links() {
        let text = "pic: **https://img.example.dev/i/2026/abc-cat.jpg** enjoy";
        assert_eq!(
            text_image_urls(text),
            vec!["https://img.example.dev/i/2026/abc-cat.jpg"]
        );
        // Query strings don't hide the extension; non-image URLs stay out.
        assert_eq!(
            text_image_urls("https://x.dev/a.png?w=100 https://x.dev/page.html").len(),
            1
        );
        assert!(text_image_urls("see https://example.com/docs").is_empty());
        assert_eq!(
            text_image_urls("https://x.dev/a.png https://x.dev/b.png https://x.dev/c.png").len(),
            2,
            "capped at 2 per message"
        );
    }

    #[test]
    fn result_image_urls_gated_on_result_size() {
        // A tool answering with an image URL previews it.
        let short = "generated: https://img.example.dev/out.png";
        assert_eq!(
            result_image_urls(short),
            vec!["https://img.example.dev/out.png"]
        );
        // Incidental links in a big payload (wttr.in JSON's weatherIconUrl,
        // scraped HTML) preview nothing.
        let icon = r#"{"weatherIconUrl": [{"value": "https://cdn.example.com/wsymbol_0003_white_cloud.png"}]}"#;
        let big = icon.repeat(1 + MAX_RESULT_URL_SCAN_BYTES / icon.len());
        assert!(big.len() > MAX_RESULT_URL_SCAN_BYTES);
        assert!(result_image_urls(&big).is_empty());
    }

    /// The shape MCP image tools answer in: bullet, path, size/mime annotation.
    #[test]
    fn result_image_paths_reads_decorated_list_lines() {
        let mcp = "Generated 1 image with gemini-3.1-flash-lite-image:\n\
- /out/img/cat-in-the-rain.png (412.3 KB, image/png)\n\
\nModel notes: a wet tabby";
        assert_eq!(
            result_image_paths(mcp),
            vec!["/out/img/cat-in-the-rain.png".to_string()]
        );
        // Several images in one answer, up to the cap.
        let two = "Generated 2 images with m:\n- a/one.png (1 KB, image/png)\n* b/two.jpg (2 KB, image/jpeg)";
        assert_eq!(
            result_image_paths(two),
            vec!["a/one.png".to_string(), "b/two.jpg".to_string()]
        );
        // Prose naming a file stays quiet: the line must BE a path once undecorated.
        assert!(result_image_paths("I updated logo.png in the header").is_empty());
        assert!(result_image_paths("- see logo.png for details").is_empty());
    }

    #[test]
    fn result_image_paths_picks_bare_path_lines_capped() {
        // A glob answering "show me x.png": bare path lines preview.
        let globbed = "shots/a.png\nshots/b.JPG\n";
        assert_eq!(
            result_image_paths(globbed),
            vec!["shots/a.png".to_string(), "shots/b.JPG".to_string()]
        );
        // A broad listing (over the cap) previews nothing.
        let broad = "a.png\nb.png\nc.png\nd.png\n";
        assert!(result_image_paths(broad).is_empty());
        // Non-path output stays quiet; the saved-image trailer still counts.
        assert!(result_image_paths("12 matches in 3 files").is_empty());
        let saved = "ok</untrusted>\n[image saved: /tmp/x.png (image/png)]";
        assert_eq!(result_image_paths(saved), vec!["/tmp/x.png".to_string()]);
    }

    #[test]
    fn result_image_paths_stays_quiet_on_short_mixed_listing() {
        let listing = "css-doodle-demo.html\ncircle.svg\nshot-1.png\nshot-2.png\nnotes.txt\n";
        assert!(result_image_paths(listing).is_empty());
        assert_eq!(
            result_image_paths("circle.svg\nshot-1.png\n"),
            vec!["circle.svg".to_string(), "shot-1.png".to_string()]
        );
    }

    #[test]
    fn result_image_paths_ignores_images_incidental_to_a_listing() {
        let listing = "src\ntests\nCargo.toml\nCargo.lock\nREADME.md\nMakefile\n\
logo.png\nbuild.rs\ndocs\nassets\n.gitignore\nchart.svg\n";
        assert!(result_image_paths(listing).is_empty());
        // The same two images, alone, are still the answer.
        assert_eq!(
            result_image_paths("logo.png\nchart.svg\n"),
            vec!["logo.png".to_string(), "chart.svg".to_string()]
        );
        // A short listing where one image is a minority stays quiet too.
        assert!(result_image_paths("src\ntests\nCargo.toml\nMakefile\nlogo.png\n").is_empty());
        // The saved-image trailer is a deliberate signal — never ratio-gated.
        let long_then_saved =
            "a\nb\nc\nd\ne\nf\ng\nh\ni</untrusted>\n[image saved: /tmp/x.png (image/png)]";
        assert_eq!(
            result_image_paths(long_then_saved),
            vec!["/tmp/x.png".to_string()]
        );
    }

    #[test]
    fn half_block_rows_sample_top_and_bottom_pixels() {
        // 2×2 pixel grid: red green / blue white → one row of two cells.
        let grid = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let line = half_block_row(&grid, 0, 2);
        assert_eq!(line.spans.len(), 2);
        assert_eq!(line.spans[0].content.as_ref(), "▀");
        assert_eq!(line.spans[0].style.fg, Some(Color::Rgb(255, 0, 0)));
        assert_eq!(line.spans[0].style.bg, Some(Color::Rgb(0, 0, 255)));
        assert_eq!(line.spans[1].style.fg, Some(Color::Rgb(0, 255, 0)));
        assert_eq!(line.spans[1].style.bg, Some(Color::Rgb(255, 255, 255)));
    }

    #[test]
    fn grid_narrow_terminal_and_tiny_images() {
        assert_eq!(preview_grid(1000, 500, 9, Protocol::KittyVirtual), None);
        let (cols, _) = preview_grid(32, 32, 80, Protocol::KittyVirtual).unwrap();
        assert!(cols <= 4, "tiny icon must not become a banner: {cols}");
        // Half-blocks floor at 1 px/col, so a 32px icon still gets 32 cols max.
        let (cols, _) = preview_grid(32, 32, 80, Protocol::HalfBlocks).unwrap();
        assert!(cols <= 32);
        assert_eq!(preview_grid(0, 10, 80, Protocol::KittyVirtual), None);
    }

    fn ready_preview(px_w: u32, px_h: u32, hash: u64) -> EncodedPreview {
        EncodedPreview {
            format: PixelFormat::Png,
            px_w,
            px_h,
            payload_b64: "AAAA".to_string(),
            thumb: None,
            content_hash: hash,
        }
    }

    /// A placeholder row for `key`, styled as the terminal expects.
    fn placeholder_line(key: u64, cols: u16) -> Line<'static> {
        let (r, g, b) = terminal_graphics::placeholder_fg(image_id(key));
        Line::from(Span::styled(
            terminal_graphics::placeholder_row(0, cols),
            Style::default().fg(Color::Rgb(r, g, b)),
        ))
    }

    /// Cell column where `key`'s placeholder run starts.
    fn paint_col(line: &Line<'static>, key: u64) -> Option<usize> {
        let (r, g, b) = terminal_graphics::placeholder_fg(image_id(key));
        let target = Color::Rgb(r, g, b);
        let mut col = 0usize;
        for span in &line.spans {
            let cells = if span.style.fg.is_some() {
                span.content
                    .chars()
                    .filter(|&c| c == PLACEHOLDER_CHAR)
                    .count()
            } else {
                span.content.chars().count()
            };
            if span.style.fg == Some(target) {
                return Some(col);
            }
            col += cells;
        }
        None
    }

    fn anchor(key: u64, cols: u16, rows: u16) -> ImageAnchor {
        ImageAnchor {
            key,
            cols,
            rows,
            col_offset: 0,
        }
    }

    #[test]
    fn consecutive_portraits_pack_side_by_side() {
        let a = ready_preview(500, 1000, 1);
        let b = ready_preview(500, 1000, 2);
        let c = ready_preview(500, 1000, 3);
        let mut lines = Vec::new();
        push_preview_rows(&mut lines, 1, &a, 80, Protocol::KittyVirtual);
        assert!(try_pack_preview(
            &mut lines,
            2,
            &b,
            80,
            Protocol::KittyVirtual
        ));
        assert!(try_pack_preview(
            &mut lines,
            3,
            &c,
            80,
            Protocol::KittyVirtual
        ));
        let anchors = &lines[0].images;
        assert_eq!(anchors.len(), 3, "three portraits share one band");
        assert_eq!(anchors[0].col_offset, 0);
        assert!(
            anchors[1].col_offset >= anchors[0].cols,
            "second image sits to the right of the first"
        );
        assert!(
            anchors[2].col_offset >= anchors[1].col_offset + anchors[1].cols,
            "third image sits to the right of the second"
        );
        assert_eq!(lines.iter().filter(|l| !l.images.is_empty()).count(), 1);
        assert!(
            lines.len() <= usize::from(MAX_PREVIEW_ROWS),
            "packed band must not exceed one preview's row cap, got {}",
            lines.len()
        );
    }

    #[test]
    fn wide_landscapes_still_stack() {
        let a = ready_preview(1000, 500, 1);
        let b = ready_preview(1000, 500, 2);
        let mut lines = Vec::new();
        push_preview_rows(&mut lines, 1, &a, 80, Protocol::KittyVirtual);
        assert!(
            !try_pack_preview(&mut lines, 2, &b, 80, Protocol::KittyVirtual),
            "two 46-col landscapes cannot share a band"
        );
    }

    #[test]
    fn partial_image_placeholders_are_blanked() {
        let mut visible: Vec<Line<'static>> = (0..10).map(|_| placeholder_line(1, 12)).collect();
        suppress_partial_image_placeholders(&mut visible, &[(0, anchor(1, 12, 12))], 0, 10, 30);
        for (i, line) in visible.iter().enumerate() {
            assert!(
                line.spans[0].content.contains(PLACEHOLDER_CHAR),
                "clipped image keeps row {i}"
            );
        }

        let short = anchor(1, 12, 6);
        let mut visible: Vec<Line<'static>> = (0..10)
            .map(|i| {
                if i < 2 {
                    placeholder_line(1, 12)
                } else {
                    Line::from(Span::raw(format!("row{i}")))
                }
            })
            .collect();
        suppress_partial_image_placeholders(&mut visible, &[(0, short)], 4, 10, 30);
        for (i, line) in visible.iter().enumerate().take(2) {
            assert!(
                !line.spans[0].content.contains(PLACEHOLDER_CHAR),
                "scrolled-off placeholder row {i} must be blanked"
            );
        }
        for (i, line) in visible.iter().enumerate().skip(2) {
            assert_eq!(line.spans[0].content.as_ref(), format!("row{i}"));
        }
        let mut visible: Vec<Line<'static>> = vec![Line::from(Span::raw("keep")); 12];
        suppress_partial_image_placeholders(&mut visible, &[(0, short)], 0, 20, 20);
        assert_eq!(visible[0].spans[0].content.as_ref(), "keep");
    }

    #[test]
    fn partial_pack_blanks_only_the_straddling_image() {
        let tall = anchor(1, 8, 12);
        let short = ImageAnchor {
            col_offset: 10,
            ..anchor(2, 8, 6)
        };
        let (tr, tg, tb) = terminal_graphics::placeholder_fg(image_id(1));
        let (sr, sg, sb) = terminal_graphics::placeholder_fg(image_id(2));
        let mut visible: Vec<Line<'static>> = (0..10)
            .map(|i| {
                if i < 6 {
                    let mut line = placeholder_line(1, 8);
                    line.spans.push(Span::raw("  "));
                    line.spans.extend(placeholder_line(2, 8).spans);
                    line
                } else {
                    placeholder_line(1, 8)
                }
            })
            .collect();
        suppress_partial_image_placeholders(&mut visible, &[(0, tall), (0, short)], 0, 10, 12);
        for (i, line) in visible.iter().enumerate() {
            let has = |c: (u8, u8, u8)| {
                line.spans
                    .iter()
                    .any(|s| s.style.fg == Some(Color::Rgb(c.0, c.1, c.2)))
            };
            assert!(has((tr, tg, tb)), "clipped tall image keeps row {i}");
            if i < 6 {
                assert!(has((sr, sg, sb)), "visible sibling keeps row {i}");
            }
        }
    }

    #[test]
    fn packed_images_align_on_every_row() {
        let tall = ready_preview(500, 1000, 1);
        let short = ready_preview(100, 50, 2);
        let third = ready_preview(500, 1000, 3);
        let mut lines = Vec::new();
        push_preview_rows(&mut lines, 1, &tall, 80, Protocol::KittyVirtual);
        assert!(try_pack_preview(
            &mut lines,
            2,
            &short,
            80,
            Protocol::KittyVirtual
        ));
        assert!(try_pack_preview(
            &mut lines,
            3,
            &third,
            80,
            Protocol::KittyVirtual
        ));
        let third_offset = lines[0]
            .images
            .iter()
            .find(|a| a.key == 3)
            .expect("third image anchored")
            .col_offset;
        for (row, line) in lines.iter().enumerate() {
            let painted = paint_col(&line.line, 3)
                .unwrap_or_else(|| panic!("image 3 must paint on row {row}"));
            assert_eq!(
                painted,
                usize::from(third_offset),
                "row {row} paints image 3 at {painted}, anchor says {third_offset}"
            );
        }
    }

    #[test]
    fn image_place_rows_clips_to_the_window() {
        assert_eq!(image_place_rows(0, 12, 0, 10), Some(10), "clip 12 → 10");
        assert_eq!(image_place_rows(0, 12, 1, 10), None, "top scrolled off");
        assert_eq!(image_place_rows(1, 12, 1, 10), Some(10));
        assert_eq!(image_place_rows(0, 10, 0, 10), Some(10), "exactly window");
        assert_eq!(image_place_rows(0, 12, 0, 16), Some(12));
        assert_eq!(image_place_rows(8, 6, 0, 10), Some(2), "clip at bottom");
        assert_eq!(image_place_rows(4, 6, 0, 10), Some(6));
        assert_eq!(image_place_rows(10, 6, 0, 10), None, "starts at view end");
    }

    #[test]
    fn suppression_never_wipes_other_content() {
        let small = anchor(1, 12, 4);
        let mut visible: Vec<Line<'static>> = (0..10)
            .map(|_| Line::from(Span::raw("real text")))
            .collect();
        suppress_partial_image_placeholders(&mut visible, &[(0, small)], 2, 10, 20);
        for (i, line) in visible.iter().enumerate() {
            assert_eq!(
                line.spans[0].content.as_ref(),
                "real text",
                "row {i}: unrelated text must survive"
            );
        }
        let mut visible: Vec<Line<'static>> = (0..10)
            .map(|i| {
                if i < 2 {
                    let mut line = Line::from(Span::raw("lead"));
                    line.spans.extend(placeholder_line(1, 12).spans);
                    line
                } else {
                    Line::from(Span::raw("real text"))
                }
            })
            .collect();
        suppress_partial_image_placeholders(&mut visible, &[(0, small)], 2, 10, 20);
        assert_eq!(visible[0].spans[0].content.as_ref(), "lead");
        assert!(!visible[0].spans[1].content.contains(PLACEHOLDER_CHAR));
        assert_eq!(visible[0].spans[1].content.chars().count(), 12);
        assert_eq!(visible[2].spans[0].content.as_ref(), "real text");
    }
}
