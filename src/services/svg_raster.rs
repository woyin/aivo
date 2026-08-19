//! Best-effort SVG → PNG rasterization via external tools.
//!
//! A pure-Rust rasterizer (resvg + tiny-skia) would cost ~2 MiB against the
//! 10 MiB musl size gate, so previews shell out instead: `rsvg-convert`
//! where installed, then macOS `qlmanage`. No tool → no preview, silently.

use std::process::Command;

const RASTER_EDGE: &str = "1024";

/// Sniffs SVG by content, so extensionless attachments still qualify.
pub fn looks_like_svg(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    let Ok(text) = std::str::from_utf8(head) else {
        return false;
    };
    let trimmed = text.trim_start();
    trimmed.starts_with("<svg") || (trimmed.starts_with("<?xml") && text.contains("<svg"))
}

pub fn rasterize_svg(bytes: &[u8]) -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().ok()?;
    let src = dir.path().join("preview.svg");
    std::fs::write(&src, bytes).ok()?;

    let out = dir.path().join("preview.png");
    let rsvg = Command::new("rsvg-convert")
        .args([
            "-w",
            RASTER_EDGE,
            "-h",
            RASTER_EDGE,
            "--keep-aspect-ratio",
            "-o",
        ])
        .arg(&out)
        .arg(&src)
        .output();
    if rsvg.is_ok_and(|o| o.status.success())
        && let Ok(png) = std::fs::read(&out)
    {
        return Some(png);
    }

    #[cfg(target_os = "macos")]
    {
        // qlmanage names its thumbnail `<input-file-name>.png` inside `-o dir`.
        let ql = Command::new("qlmanage")
            .args(["-t", "-s", RASTER_EDGE, "-o"])
            .arg(dir.path())
            .arg(&src)
            .output();
        if ql.is_ok_and(|o| o.status.success())
            && let Ok(png) = std::fs::read(dir.path().join("preview.svg.png"))
        {
            return Some(png);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniffs_svg_content() {
        assert!(looks_like_svg(
            b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>"
        ));
        assert!(looks_like_svg(
            b"<?xml version=\"1.0\"?>\n<svg width=\"10\"></svg>"
        ));
        assert!(looks_like_svg(b"  \n\t<svg>"));
        assert!(!looks_like_svg(b"<html><body></body></html>"));
        assert!(!looks_like_svg(&[0x89, b'P', b'N', b'G']));
        assert!(!looks_like_svg(b""));
    }
}
