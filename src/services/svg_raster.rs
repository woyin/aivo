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

/// Conservative external-reference sniff for untrusted markup (inline
/// assistant SVG). rsvg-convert blocks remote loads, but qlmanage is
/// WebKit-backed and may fetch — a prompt-injected `<image href>` would be a
/// zero-click beacon. Allowlists the *target* rather than blocklisting schemes:
/// a literal-needle filter (`href="http`) waves through the entity-escaped
/// (`href="&#104;ttp:`), unquoted and spaced spellings WebKit still fetches.
pub fn svg_has_external_refs(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    const MARKUP: &[&str] = &["<script", "<foreignobject", "<iframe"];
    if MARKUP.iter().any(|n| lower.contains(n)) {
        return true;
    }
    // `xlink:href` ends in `href`, so both spellings fall out of one needle.
    for attr in ["href", "src"] {
        for (pos, _) in lower.match_indices(attr) {
            // No `=` — prose, or an attribute that merely ends in the needle.
            let Some(value) = lower[pos + attr.len()..].trim_start().strip_prefix('=') else {
                continue;
            };
            if target_is_external(value) {
                return true;
            }
        }
    }
    lower
        .match_indices("url(")
        .any(|(pos, _)| target_is_external(&lower[pos + "url(".len()..]))
}

/// Safe only when local (`#fragment`), inline (`data:`), or empty — anything
/// else may hit the network, so an unrecognized target fails closed.
fn target_is_external(value: &str) -> bool {
    let value = value.trim_start();
    let (quoted, value) = match value.strip_prefix(['"', '\'']) {
        Some(rest) => (true, rest.trim_start()),
        None => (false, value),
    };
    // `href=""`, `url()` — nothing to load.
    if value.is_empty()
        || value.starts_with([')', '>'])
        || (quoted && value.starts_with(['"', '\'']))
    {
        return false;
    }
    !(value.starts_with('#') || value.starts_with("data:"))
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
    fn flags_external_refs_only() {
        assert!(svg_has_external_refs(
            "<svg><image href=\"http://evil/track.png\"/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><use xlink:href='https://evil/x.svg#i'/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><style>.a{fill:url(http://evil/p)}</style></svg>"
        ));
        assert!(svg_has_external_refs("<svg><script>1</script></svg>"));
        assert!(svg_has_external_refs("<svg><foreignObject/></svg>"));
        // The xmlns declaration and local refs are not external loads.
        assert!(!svg_has_external_refs(
            "<svg xmlns=\"http://www.w3.org/2000/svg\"><use href=\"#a\"/>\
<image href=\"data:image/png;base64,aGk=\"/><circle r=\"4\"/></svg>"
        ));
    }

    /// The spellings a literal-needle filter waved through, which WebKit fetches.
    #[test]
    fn flags_obfuscated_external_refs() {
        assert!(svg_has_external_refs(
            "<svg><image href=\"&#104;ttp://evil/track.png\"/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><image href=http://evil/track.png /></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><image href = \"http://evil/track.png\"/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><image href=\"HTTP://EVIL/track.png\"/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><image href=\"/local/secret.png\"/></svg>"
        ));
        assert!(svg_has_external_refs(
            "<svg><style>.a{fill:url( 'http://evil/p' )}</style></svg>"
        ));
        // Empty refs load nothing; the word "href" in prose isn't a ref.
        assert!(!svg_has_external_refs(
            "<svg><image href=\"\"/><text>href and src</text><rect fill=\"url(#g)\"/></svg>"
        ));
    }

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
