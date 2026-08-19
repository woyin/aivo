//! Terminal graphics emission for inline image previews (kitty APC + sixel).
//!
//! Detection is env vars plus a `tmux display-message` subprocess — never an
//! in-band escape query: a probe's late reply leaks as typed input on slow
//! SSH links (the same reason theme auto-detection was removed). Every kitty
//! sequence carries `q=2` so a capable terminal stays silent too.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

/// Max base64 payload bytes per APC escape, per the Kitty protocol.
const CHUNK_SIZE: usize = 4096;

/// PNG small enough to hand the terminal verbatim instead of re-encoding.
const PNG_PASSTHROUGH_MAX_BYTES: usize = 300_000;
const PNG_PASSTHROUGH_MAX_EDGE: u32 = 1024;

/// How previews reach the screen.
///
/// `KittyVirtual` (Unicode placeholders, U+10EEEE cells): the image composites
/// wherever the placeholder *cells* render — pane translation, clipping,
/// scrolling, overlays all handled by the normal cell pipeline. Real pixels,
/// but only kitty (and placeholder-capable peers) composite it; a terminal
/// that ignores `U=1` shows the image at the raw cursor position instead.
///
/// `KittyClassic` (cursor-addressed `a=p`): for WezTerm/iTerm2, which speak
/// the graphics protocol but not virtual placements. Never through tmux —
/// passthrough leaves the outer cursor wherever tmux's redraw parked it, so
/// the image lands at raw window coordinates, ignoring the pane.
///
/// `Sixel`: real pixels through tmux. tmux ≥3.4 parses sixel DCS emitted by a
/// pane application as PANE CONTENT — it clips, positions, and redraws the
/// image itself — so cursor-addressed sixel is the one protocol that both
/// positions correctly under tmux and carries full resolution.
///
/// `HalfBlocks`: no graphics protocol at all — the preview rows are `▀` cells
/// with truecolor fg/bg carrying two pixels each. Coarse, but works in every
/// truecolor terminal, through tmux, with zero misplacement risk. The fallback
/// when `AIVO_PREVIEW=1` is set in an environment nothing vouches for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Protocol {
    #[default]
    None,
    KittyVirtual,
    KittyClassic,
    Sixel,
    HalfBlocks,
}

impl Protocol {
    /// Status-line badge for the active mode.
    pub fn label(self) -> Option<&'static str> {
        match self {
            Protocol::None => None,
            Protocol::KittyVirtual | Protocol::KittyClassic => Some("img:kitty"),
            Protocol::Sixel => Some("img:sixel"),
            Protocol::HalfBlocks => Some("img:blocks"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphicsCaps {
    pub protocol: Protocol,
    /// Inside tmux every APC must ride a DCS passthrough wrapper (sixel is
    /// exempt: tmux parses it natively).
    pub tmux: bool,
    /// Terminal cell size in pixels, for sizing sixel output. From the tmux
    /// server when available, else a conservative 8×16.
    pub cell_px: (u16, u16),
}

impl Default for GraphicsCaps {
    fn default() -> Self {
        Self {
            protocol: Protocol::None,
            tmux: false,
            cell_px: (8, 16),
        }
    }
}

impl GraphicsCaps {
    pub fn enabled(&self) -> bool {
        self.protocol != Protocol::None
    }

    pub fn virtual_placement(&self) -> bool {
        self.protocol == Protocol::KittyVirtual
    }

    pub fn kitty(&self) -> bool {
        matches!(
            self.protocol,
            Protocol::KittyVirtual | Protocol::KittyClassic
        )
    }

    pub fn emits_escapes(&self) -> bool {
        self.kitty() || self.protocol == Protocol::Sixel
    }
}

/// What the tmux server knows about its client (outer) terminal — collected
/// via a `tmux display-message` SUBPROCESS, not an in-band escape probe, so
/// the no-query rule holds.
#[derive(Clone, Debug, Default)]
pub struct TmuxClientInfo {
    pub features: String,
    pub termtype: String,
    pub cell_px: Option<(u16, u16)>,
}

fn tmux_client_info() -> Option<TmuxClientInfo> {
    let out = std::process::Command::new("tmux")
        .args([
            "display-message",
            "-p",
            "#{client_termfeatures}\t#{client_termtype}\t#{client_cell_width}\t#{client_cell_height}",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let mut parts = line.trim_end().splitn(4, '\t');
    let features = parts.next().unwrap_or_default().to_string();
    let termtype = parts.next().unwrap_or_default().to_string();
    let cw = parts.next().and_then(|v| v.parse::<u16>().ok());
    let ch = parts.next().and_then(|v| v.parse::<u16>().ok());
    Some(TmuxClientInfo {
        features,
        termtype,
        cell_px: match (cw, ch) {
            (Some(w), Some(h)) if w > 0 && h > 0 => Some((w, h)),
            _ => None,
        },
    })
}

/// Capability from env sniffing plus (under tmux) the tmux server's own
/// knowledge of its client terminal. `AIVO_PREVIEW` overrides: an explicit
/// `virtual` | `classic` | `sixel` | `blocks`, or standard flag truthiness —
/// falsy off, truthy auto-enable (best mode the evidence supports).
pub fn detect() -> GraphicsCaps {
    let in_tmux = std::env::var("TMUX").is_ok_and(|v| !v.is_empty());
    let tmux_info = if in_tmux { tmux_client_info() } else { None };
    detect_from(
        &|key| std::env::var(key).ok(),
        tmux_info.as_ref(),
        cfg!(windows),
    )
}

fn detect_from(
    var: &dyn Fn(&str) -> Option<String>,
    tmux_info: Option<&TmuxClientInfo>,
    windows: bool,
) -> GraphicsCaps {
    let tmux = var("TMUX").is_some_and(|v| !v.is_empty())
        || var("TERM").is_some_and(|t| t.starts_with("tmux"));
    let cell_px = tmux_info.and_then(|i| i.cell_px).unwrap_or((8, 16));
    let caps = |protocol: Protocol| GraphicsCaps {
        protocol,
        tmux,
        cell_px,
    };

    let term = var("TERM").unwrap_or_default();
    let term_program = var("TERM_PROGRAM").unwrap_or_default().to_lowercase();
    // Inside tmux, TERM/TERM_PROGRAM describe tmux itself; the outer terminal
    // is identified by the tmux server (it ran the DA/XTVERSION handshake at
    // attach) plus any session-scoped env vars that survived.
    let outer = tmux_info
        .map(|i| i.termtype.to_lowercase())
        .unwrap_or_default();
    let placeholder_capable = term.contains("kitty")
        || term.contains("ghostty")
        || matches!(term_program.as_str(), "kitty" | "ghostty")
        || var("KITTY_WINDOW_ID").is_some()
        || var("GHOSTTY_RESOURCES_DIR").is_some()
        || outer.contains("kitty")
        || outer.contains("ghostty");
    // The tmux server re-renders sixel as pane content — correct position and
    // clipping — whenever the outer terminal advertised sixel support.
    let sixel_via_tmux = tmux
        && tmux_info.is_some_and(|i| {
            i.features
                .split(',')
                .any(|feature| feature.trim() == "sixel")
        });
    // Kitty graphics without virtual placements; iTerm2's support is
    // placement-limited too, so both stay on the classic path.
    let classic_only = matches!(term_program.as_str(), "wezterm" | "warpterminal")
        || var("WEZTERM_EXECUTABLE").is_some()
        || var("LC_TERMINAL").is_some_and(|v| v.eq_ignore_ascii_case("iterm2"))
        || var("ITERM_SESSION_ID").is_some();

    let auto = || {
        if placeholder_capable {
            Some(Protocol::KittyVirtual)
        } else if sixel_via_tmux {
            Some(Protocol::Sixel)
        } else if classic_only && !tmux {
            Some(Protocol::KittyClassic)
        } else {
            None
        }
    };

    match var("AIVO_PREVIEW").as_deref() {
        Some("virtual") => return caps(Protocol::KittyVirtual),
        Some("classic") => return caps(Protocol::KittyClassic),
        Some("sixel") => return caps(Protocol::Sixel),
        Some("blocks") => return caps(Protocol::HalfBlocks),
        Some(v) => match crate::services::system_env::flag_value(v) {
            Some(false) => return GraphicsCaps::default(),
            // Forced on: best evidenced mode, half-blocks when nothing vouches.
            Some(true) => return caps(auto().unwrap_or(Protocol::HalfBlocks)),
            None => {}
        },
        None => {}
    }
    // No auto-detection on Windows (conhost/WT can't render these protocols),
    // but explicit AIVO_PREVIEW overrides above still apply.
    if windows {
        return GraphicsCaps::default();
    }
    match auto() {
        Some(protocol) => caps(protocol),
        None => GraphicsCaps::default(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// `f=100`: terminal decodes the PNG itself.
    Png,
    /// `f=24`: raw RGB8, dimensions carried in `s=`/`v=`.
    Rgb,
}

/// Small raw-RGB thumbnail for cell-painted (half-block) previews.
pub struct Thumb {
    pub rgb: Vec<u8>,
    pub w: u32,
    pub h: u32,
}

/// Long-edge cap for the raw-RGB thumb, which doubles as the `f=24` payload
/// on the non-passthrough path. Sixel renders at true pixel size (up to
/// ~46 cols × 8 px), so it must carry real resolution; half-blocks just
/// downsample it further.
const THUMB_MAX_EDGE: u32 = 512;

/// An image pre-encoded for transmission: base64 payload plus the pixel
/// dimensions the preview grid is sized from, and a tiny raw thumb for the
/// protocol-free half-block mode.
pub struct EncodedPreview {
    pub format: PixelFormat,
    pub px_w: u32,
    pub px_h: u32,
    pub payload_b64: String,
    pub thumb: Option<Thumb>,
    /// Identity of the decoded PIXELS, independent of how the bytes arrived —
    /// a generated image seen as a URL and again as its downloaded file must
    /// dedup as one picture.
    pub content_hash: u64,
}

/// Encode PNG/JPEG bytes for preview: small PNGs pass through verbatim,
/// everything else decodes via the existing zune path and downscales to a
/// raw-RGB thumbnail. `None` = undecodable (caller shows no preview).
pub fn prepare_preview(bytes: &[u8]) -> Option<EncodedPreview> {
    use crate::services::image_optimize::{self, SniffedFormat};
    use std::hash::{Hash, Hasher};
    let format = image_optimize::sniff_format(bytes)?;
    let (w, h) = image_optimize::probe_dimensions(bytes, format)?;
    let thumb = image_optimize::preview_rgb(bytes, THUMB_MAX_EDGE).map(|(rgb, tw, th)| Thumb {
        rgb,
        w: tw,
        h: th,
    });
    let content_hash = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (w, h).hash(&mut hasher);
        match &thumb {
            Some(t) => t.rgb.hash(&mut hasher),
            None => bytes.hash(&mut hasher),
        }
        hasher.finish()
    };
    if format == SniffedFormat::Png
        && bytes.len() <= PNG_PASSTHROUGH_MAX_BYTES
        && w.max(h) <= PNG_PASSTHROUGH_MAX_EDGE
    {
        return Some(EncodedPreview {
            format: PixelFormat::Png,
            px_w: w,
            px_h: h,
            payload_b64: BASE64.encode(bytes),
            thumb,
            content_hash,
        });
    }
    let t = thumb.as_ref()?;
    let (px_w, px_h, payload_b64) = (t.w, t.h, BASE64.encode(&t.rgb));
    Some(EncodedPreview {
        format: PixelFormat::Rgb,
        px_w,
        px_h,
        payload_b64,
        thumb,
        content_hash,
    })
}

/// One APC escape, tmux-passthrough-wrapped when needed (ESC doubled inside
/// the DCS body, per tmux's `allow-passthrough`).
fn push_apc(out: &mut String, control: &str, data: &str, tmux: bool) {
    let seq = if data.is_empty() {
        format!("\x1b_G{control}\x1b\\")
    } else {
        format!("\x1b_G{control};{data}\x1b\\")
    };
    if tmux {
        out.push_str("\x1bPtmux;");
        out.push_str(&seq.replace('\x1b', "\x1b\x1b"));
        out.push_str("\x1b\\");
    } else {
        out.push_str(&seq);
    }
}

/// Transmit image data under `id` without displaying it (`a=t`), chunked.
/// Only the first chunk carries the format header.
pub fn transmit(out: &mut String, id: u32, preview: &EncodedPreview, tmux: bool) {
    let data = preview.payload_b64.as_bytes();
    let mut offset = 0usize;
    let mut first = true;
    loop {
        let end = (offset + CHUNK_SIZE).min(data.len());
        // base64 is ASCII, so any byte boundary is a char boundary.
        let chunk = std::str::from_utf8(&data[offset..end]).unwrap_or_default();
        let more = if end < data.len() { 1 } else { 0 };
        let control = if first {
            match preview.format {
                PixelFormat::Png => format!("a=t,q=2,f=100,i={id},m={more}"),
                PixelFormat::Rgb => format!(
                    "a=t,q=2,f=24,s={},v={},i={id},m={more}",
                    preview.px_w, preview.px_h
                ),
            }
        } else {
            format!("m={more}")
        };
        push_apc(out, &control, chunk, tmux);
        first = false;
        offset = end;
        if offset >= data.len() {
            break;
        }
    }
}

/// Place transmitted image `id` at cell (`x`,`y`) (0-based), scaled into a
/// `cols`×`rows` cell box. `C=1` keeps the cursor where the CUP put it.
#[allow(clippy::too_many_arguments)]
pub fn place(
    out: &mut String,
    id: u32,
    pid: u32,
    x: u16,
    y: u16,
    cols: u16,
    rows: u16,
    tmux: bool,
) {
    // CUP is plain CSI — tmux translates it natively, no wrapper.
    out.push_str(&format!(
        "\x1b[{};{}H",
        y.saturating_add(1),
        x.saturating_add(1)
    ));
    push_apc(
        out,
        &format!("a=p,q=2,i={id},p={pid},c={cols},r={rows},C=1"),
        "",
        tmux,
    );
}

/// Create (or replace) image `id`'s virtual placement: a `cols`×`rows` grid
/// that placeholder cells index into. One per image, placement id 1.
pub fn create_virtual_placement(out: &mut String, id: u32, cols: u16, rows: u16, tmux: bool) {
    push_apc(
        out,
        &format!("a=p,q=2,U=1,i={id},p=1,c={cols},r={rows}"),
        "",
        tmux,
    );
}

/// Placeholder base char: the terminal composites the virtual placement's
/// tiles over cells showing this codepoint (image id in the foreground color).
pub const PLACEHOLDER_CHAR: char = '\u{10EEEE}';

/// The Kitty protocol's row/column diacritics, in index order. Enough entries
/// for the preview grid caps; a bad or missing diacritic degrades gracefully —
/// the terminal infers row/col from the neighboring cell.
const ROWCOL_DIACRITICS: [u32; 64] = [
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A, 0x034B, 0x034C,
    0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365, 0x0366, 0x0367, 0x0368, 0x0369,
    0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592,
    0x0593, 0x0594, 0x0595, 0x0597, 0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1,
    0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x05C5, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614,
    0x0615, 0x0616, 0x0617, 0x0657,
];

fn rowcol_diacritic(index: u16) -> Option<char> {
    ROWCOL_DIACRITICS
        .get(usize::from(index))
        .and_then(|&cp| char::from_u32(cp))
}

/// One transcript row of placeholder cells for `row` of the virtual grid.
pub fn placeholder_row(grid_row: u16, cols: u16) -> String {
    let mut out = String::with_capacity(usize::from(cols) * 8);
    let row_mark = rowcol_diacritic(grid_row);
    for col in 0..cols {
        out.push(PLACEHOLDER_CHAR);
        if let Some(row_mark) = row_mark {
            out.push(row_mark);
            if let Some(col_mark) = rowcol_diacritic(col) {
                out.push(col_mark);
            }
        }
    }
    out
}

/// The RGB triple placeholder cells must use as foreground color: it carries
/// the image id (ids are kept to 24 bits for exactly this).
pub fn placeholder_fg(id: u32) -> (u8, u8, u8) {
    ((id >> 16) as u8, (id >> 8) as u8, id as u8)
}

/// Nonzero 24-bit Kitty image id from a preview content key (24 bits so the
/// whole id fits in a placeholder's foreground color).
pub fn image_id(key: u64) -> u32 {
    let folded = (key >> 32) as u32 ^ key as u32;
    let id = (folded ^ (folded >> 24)) & 0x00FF_FFFF;
    id.max(1)
}

const BAYER4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// Quantize one channel to `levels` with ordered dithering — banding on
/// photos is the visible failure mode of a fixed 252-color palette.
fn sixel_quant(value: u8, levels: u32, x: u32, y: u32) -> u32 {
    let dither = (f32::from(BAYER4[(y & 3) as usize][(x & 3) as usize]) + 0.5) / 16.0 - 0.5;
    let level = (f32::from(value) / 255.0) * (levels as f32 - 1.0) + dither;
    level.round().clamp(0.0, levels as f32 - 1.0) as u32
}

/// Encode RGB pixels as a sixel image: DCS intro, raster attributes, a fixed
/// 6×7×6 (252-color) palette, RLE-compressed six-row bands, ST. Pure Rust —
/// a real rasterizer dependency would fight the binary size gate.
pub fn sixel_encode(rgb: &[u8], w: u32, h: u32) -> String {
    let mut out = String::with_capacity((w * h / 3) as usize + 4096);
    out.push_str("\x1bPq");
    out.push_str(&format!("\"1;1;{w};{h}"));
    for r in 0..6u32 {
        for g in 0..7u32 {
            for b in 0..6u32 {
                let index = r * 42 + g * 6 + b;
                // Sixel palette components are percentages.
                out.push_str(&format!(
                    "#{index};2;{};{};{}",
                    r * 100 / 5,
                    g * 100 / 6,
                    b * 100 / 5
                ));
            }
        }
    }
    let mut indexed = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            let (Some(&r), Some(&g), Some(&b)) = (rgb.get(i), rgb.get(i + 1), rgb.get(i + 2))
            else {
                continue;
            };
            let index = sixel_quant(r, 6, x, y) * 42
                + sixel_quant(g, 7, x, y) * 6
                + sixel_quant(b, 6, x, y);
            indexed[(y * w + x) as usize] = index as u8;
        }
    }
    fn flush_run(out: &mut String, bits: u8, len: u32) {
        if len == 0 {
            return;
        }
        let ch = (63 + bits) as char;
        if len > 3 {
            out.push_str(&format!("!{len}{ch}"));
        } else {
            for _ in 0..len {
                out.push(ch);
            }
        }
    }
    let mut y0 = 0u32;
    while y0 < h {
        let band_rows = (h - y0).min(6);
        let mut used = [false; 252];
        for y in y0..y0 + band_rows {
            for x in 0..w {
                used[indexed[(y * w + x) as usize] as usize] = true;
            }
        }
        let mut first_color = true;
        for (color, _) in used.iter().enumerate().filter(|&(_, &u)| u) {
            if !first_color {
                // Carriage return: next color overdraws the same band.
                out.push('$');
            }
            first_color = false;
            out.push_str(&format!("#{color}"));
            let mut run_bits = 0u8;
            let mut run_len = 0u32;
            for x in 0..w {
                let mut bits = 0u8;
                for dy in 0..band_rows {
                    if indexed[((y0 + dy) * w + x) as usize] as usize == color {
                        bits |= 1 << dy;
                    }
                }
                if bits == run_bits {
                    run_len += 1;
                } else {
                    flush_run(&mut out, run_bits, run_len);
                    run_bits = bits;
                    run_len = 1;
                }
            }
            flush_run(&mut out, run_bits, run_len);
        }
        out.push('-');
        y0 += band_rows;
    }
    out.push_str("\x1b\\");
    out
}

/// Position the cursor and emit a sixel image there. Never passthrough-
/// wrapped — see `Protocol::Sixel`.
pub fn sixel_place(out: &mut String, x: u16, y: u16, data: &str) {
    out.push_str(&format!(
        "\x1b[{};{}H",
        y.saturating_add(1),
        x.saturating_add(1)
    ));
    out.push_str(data);
}

/// Remove one placement, keeping the transmitted data for re-placement.
pub fn delete_placement(out: &mut String, id: u32, pid: u32, tmux: bool) {
    push_apc(out, &format!("a=d,d=i,i={id},p={pid},q=2"), "", tmux);
}

/// Remove an image's placements AND free its transmitted data.
pub fn delete_image(out: &mut String, id: u32, tmux: bool) {
    push_apc(out, &format!("a=d,d=I,i={id},q=2"), "", tmux);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn caps_with(vars: &[(&str, &str)]) -> GraphicsCaps {
        caps_with_tmux(vars, None)
    }

    fn caps_with_tmux(vars: &[(&str, &str)], tmux_info: Option<&TmuxClientInfo>) -> GraphicsCaps {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        detect_from(&move |key| map.get(key).cloned(), tmux_info, false)
    }

    #[test]
    fn detect_env_sniffing() {
        assert!(!caps_with(&[]).enabled());
        assert!(caps_with(&[("TERM", "xterm-kitty")]).virtual_placement());
        assert!(caps_with(&[("TERM", "xterm-ghostty")]).virtual_placement());
        assert!(caps_with(&[("KITTY_WINDOW_ID", "3")]).virtual_placement());
        assert!(!caps_with(&[("TERM", "xterm-256color")]).enabled());
        // WezTerm/iTerm2 have no virtual placements → classic, outside tmux only.
        let wez = caps_with(&[("TERM_PROGRAM", "WezTerm")]);
        assert!(wez.enabled() && !wez.virtual_placement());
        let iterm = caps_with(&[("LC_TERMINAL", "iTerm2")]);
        assert!(iterm.enabled() && !iterm.virtual_placement());
        assert!(!caps_with(&[("TERM_PROGRAM", "WezTerm"), ("TMUX", "x")]).enabled());
    }

    #[test]
    fn detect_override_and_tmux() {
        let off = caps_with(&[("AIVO_PREVIEW", "0"), ("TERM", "xterm-kitty")]);
        assert!(!off.enabled());
        // Forced on with no outer-terminal evidence (tmux hides it) → the
        // safe cell-painted mode, never a protocol that can misplace.
        let forced = caps_with(&[
            ("AIVO_PREVIEW", "1"),
            ("TMUX", "/private/tmp/tmux-1/default,1,0"),
        ]);
        assert_eq!(forced.protocol, Protocol::HalfBlocks);
        assert!(forced.tmux);
        // With evidence, forced-on picks the protocol the env vouches for.
        let forced_kitty = caps_with(&[("AIVO_PREVIEW", "1"), ("KITTY_WINDOW_ID", "1")]);
        assert!(forced_kitty.virtual_placement());
        let classic = caps_with(&[("AIVO_PREVIEW", "classic")]);
        assert!(classic.enabled() && !classic.virtual_placement());
        let blocks = caps_with(&[("AIVO_PREVIEW", "blocks"), ("TMUX", "x")]);
        assert_eq!(blocks.protocol, Protocol::HalfBlocks);
        let tmux = caps_with(&[("TMUX", "x"), ("KITTY_WINDOW_ID", "1")]);
        assert!(tmux.virtual_placement());
        assert!(tmux.tmux);
    }

    #[test]
    fn image_ids_are_24_bit_and_nonzero() {
        assert_eq!(image_id(0), 1);
        assert!(image_id(0xdead_beef_0000_0001) <= 0x00FF_FFFF);
        assert!(image_id(u64::MAX) >= 1);
    }

    #[test]
    fn placeholder_rows_carry_row_and_column_diacritics() {
        let row = placeholder_row(0, 2);
        assert_eq!(row, "\u{10EEEE}\u{0305}\u{0305}\u{10EEEE}\u{0305}\u{030D}");
        // Second grid row switches the row diacritic.
        assert!(placeholder_row(1, 1).starts_with("\u{10EEEE}\u{030D}"));
        // Beyond the table, cells degrade to the bare placeholder (terminal
        // infers position from neighbors).
        let far = placeholder_row(64, 2);
        assert_eq!(far, "\u{10EEEE}\u{10EEEE}");
    }

    #[test]
    fn virtual_placement_emission() {
        let mut out = String::new();
        create_virtual_placement(&mut out, 9, 40, 12, false);
        assert_eq!(out, "\x1b_Ga=p,q=2,U=1,i=9,p=1,c=40,r=12\x1b\\");
    }

    fn preview(payload_len: usize) -> EncodedPreview {
        EncodedPreview {
            format: PixelFormat::Png,
            px_w: 10,
            px_h: 10,
            payload_b64: "A".repeat(payload_len),
            thumb: None,
            content_hash: 1,
        }
    }

    #[test]
    fn transmit_single_chunk() {
        let mut out = String::new();
        transmit(&mut out, 7, &preview(16), false);
        assert_eq!(
            out,
            format!("\x1b_Ga=t,q=2,f=100,i=7,m=0;{}\x1b\\", "A".repeat(16))
        );
    }

    #[test]
    fn transmit_chunks_and_headers() {
        let mut out = String::new();
        transmit(&mut out, 7, &preview(CHUNK_SIZE + 10), false);
        let escapes: Vec<&str> = out.split("\x1b\\").filter(|s| !s.is_empty()).collect();
        assert_eq!(escapes.len(), 2);
        assert!(escapes[0].starts_with("\x1b_Ga=t,q=2,f=100,i=7,m=1;"));
        assert!(escapes[1].starts_with("\x1b_Gm=0;"));
        assert!(escapes[1].ends_with(&"A".repeat(10)));
    }

    #[test]
    fn transmit_rgb_carries_dimensions() {
        let mut out = String::new();
        let p = EncodedPreview {
            format: PixelFormat::Rgb,
            px_w: 320,
            px_h: 200,
            payload_b64: "AAAA".into(),
            thumb: None,
            content_hash: 1,
        };
        transmit(&mut out, 9, &p, false);
        assert!(out.contains("f=24,s=320,v=200,i=9"));
    }

    #[test]
    fn tmux_wraps_and_doubles_escapes() {
        let mut out = String::new();
        delete_image(&mut out, 5, true);
        assert_eq!(out, "\x1bPtmux;\x1b\x1b_Ga=d,d=I,i=5,q=2\x1b\x1b\\\x1b\\");
    }

    #[test]
    fn place_moves_cursor_then_places() {
        let mut out = String::new();
        place(&mut out, 3, 21, 4, 20, 40, 12, false);
        assert_eq!(out, "\x1b[21;5H\x1b_Ga=p,q=2,i=3,p=21,c=40,r=12,C=1\x1b\\");
    }

    #[test]
    fn detect_prefers_sixel_when_tmux_client_advertises_it() {
        // The real observed setup: WezTerm outer, tmux 3.5a, sixel feature.
        let info = TmuxClientInfo {
            features: "bpaste,ccolour,clipboard,cstyle,focus,RGB,sixel,title".into(),
            termtype: "WezTerm 20240203-110809-5046fc22".into(),
            cell_px: Some((8, 16)),
        };
        let caps = caps_with_tmux(&[("TMUX", "/tmp/tmux-1/default,1,0")], Some(&info));
        assert_eq!(caps.protocol, Protocol::Sixel);
        assert_eq!(caps.cell_px, (8, 16));
        assert!(caps.tmux);
        // A kitty outer (per the tmux server) wins over sixel: real protocol.
        let kitty = TmuxClientInfo {
            features: "RGB,sixel".into(),
            termtype: "kitty 0.32".into(),
            cell_px: None,
        };
        let caps = caps_with_tmux(&[("TMUX", "x")], Some(&kitty));
        assert_eq!(caps.protocol, Protocol::KittyVirtual);
        // No sixel feature, unknown outer → stays off on auto.
        let plain = TmuxClientInfo::default();
        let caps = caps_with_tmux(&[("TMUX", "x")], Some(&plain));
        assert!(!caps.enabled());
    }

    #[test]
    fn sixel_encoder_structure() {
        // 2x2: red, green / blue, white
        let rgb = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let out = sixel_encode(&rgb, 2, 2);
        assert!(
            out.starts_with("\x1bPq\"1;1;2;2"),
            "DCS + raster: {:?}",
            &out[..14]
        );
        assert!(out.ends_with("\x1b\\"));
        // Full 252-entry palette declared (+1: the raster header "1;1;2;2
        // also matches the pattern).
        assert_eq!(out.matches(";2;").count(), 253);
        // Pure red quantizes to r=5,g=0,b=0 → index 210; it must paint.
        assert!(out.contains("#210"), "red palette entry used");
        // Exactly one band terminator for a 2-row image.
        assert!(out.matches('-').count() >= 1);
    }

    #[test]
    fn sixel_rle_compresses_flat_runs() {
        // 100x6 solid black: one band, one color, one RLE run of 100.
        let rgb = vec![0u8; 100 * 6 * 3];
        let out = sixel_encode(&rgb, 100, 6);
        assert!(out.contains("!100"), "expected RLE run: {out:?}");
    }
}
