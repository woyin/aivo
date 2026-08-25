//! Best-effort SVG → PNG rasterization via external tools.
//!
//! A pure-Rust rasterizer (resvg + tiny-skia) would cost ~2 MiB against the
//! 10 MiB musl size gate, so previews shell out instead: rsvg-convert, resvg,
//! macOS qlmanage, then chrome-family headless (Edge ships with Windows).
//! No tool → no preview, silently.

use std::path::{Path, PathBuf};
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

/// Sniffs an HTML document by content.
pub fn looks_like_html(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    let Ok(text) = std::str::from_utf8(head) else {
        return false;
    };
    let lower = text.trim_start().to_ascii_lowercase();
    lower.starts_with("<!doctype html") || lower.starts_with("<html") || lower.contains("<html")
}

/// What the side preview pane can render; shared by the engine's `preview`
/// vetting and the TUI pane's pipeline pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewTargetKind {
    Image,
    Svg,
    Html,
}

/// Extension first, content sniff for extensionless files; `None` = not renderable.
pub fn classify_preview_target(path: &Path) -> Option<PreviewTargetKind> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("png" | "jpg" | "jpeg") => return Some(PreviewTargetKind::Image),
        Some("svg") => return Some(PreviewTargetKind::Svg),
        Some("html" | "htm") => return Some(PreviewTargetKind::Html),
        _ => {}
    }
    let mut head = vec![0u8; 4096];
    let n = std::fs::File::open(path)
        .and_then(|mut f| std::io::Read::read(&mut f, &mut head))
        .ok()?;
    head.truncate(n);
    if crate::services::image_optimize::sniff_format(&head).is_some() {
        Some(PreviewTargetKind::Image)
    } else if looks_like_svg(&head) {
        Some(PreviewTargetKind::Svg)
    } else if looks_like_html(&head) {
        Some(PreviewTargetKind::Html)
    } else {
        None
    }
}

/// Conservative external-reference sniff. The browser rungs (qlmanage,
/// chrome) fetch external refs — a prompt-injected `<image href>` would be a
/// zero-click beacon — so they only see markup this clears. Allowlists the
/// *target* rather than blocklisting schemes: a literal-needle filter
/// (`href="http`) waves through the entity-escaped (`href="&#104;ttp:`),
/// unquoted and spaced spellings browsers still fetch.
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
    // Kill switch: the whole cascade shells out to host tools, so results are
    // host-dependent — and the browser rungs under a quarantined $HOME pop
    // macOS "keychain cannot be found" dialogs. The test sandbox sets this.
    if std::env::var("AIVO_NO_SVG_RASTER").is_ok_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
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

    // -w only: with both -w and -h resvg stretches.
    let resvg = Command::new("resvg")
        .args(["-w", RASTER_EDGE])
        .arg(&src)
        .arg(&out)
        .output();
    if resvg.is_ok_and(|o| o.status.success())
        && let Ok(png) = std::fs::read(&out)
    {
        return Some(png);
    }

    // The rungs below fetch external refs — fail closed.
    if !std::str::from_utf8(bytes).is_ok_and(|text| !svg_has_external_refs(text)) {
        return None;
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

    // A failed rung above may have left a partial `out`.
    let _ = std::fs::remove_file(&out);
    let profile = dir.path().join("chrome-profile");
    let edge: u32 = RASTER_EDGE.parse().unwrap_or(1024);
    for browser in chrome_candidates() {
        if try_chrome_screenshot(&browser, &profile, src.as_os_str(), &out, (edge, edge))
            && let Ok(png) = std::fs::read(&out)
        {
            return Some(png);
        }
    }

    None
}

/// Page (HTML file or http(s) URL) → PNG viewport screenshot, browser-only.
/// Deliberately no external-ref gate: a page preview is an explicit request,
/// and pages fetch subresources by design.
pub fn rasterize_page(target: &str) -> Option<Vec<u8>> {
    // Same kill switch as the SVG cascade (the test sandbox sets it).
    if std::env::var("AIVO_NO_SVG_RASTER").is_ok_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
    let dir = tempfile::tempdir().ok()?;
    let out = dir.path().join("page.png");
    let profile = dir.path().join("chrome-profile");
    for browser in chrome_candidates() {
        if try_chrome_screenshot(
            &browser,
            &profile,
            std::ffi::OsStr::new(target),
            &out,
            PAGE_VIEWPORT,
        ) && let Ok(png) = std::fs::read(&out)
        {
            return Some(png);
        }
    }
    None
}

/// 1024 wide = the terminal PNG-passthrough edge limit, so screenshots reach
/// kitty full-res instead of the 512px re-encoded thumb.
const PAGE_VIEWPORT: (u32, u32) = (1024, 640);

/// Most-preferred first; missing candidates fail the spawn instantly.
fn chrome_candidates() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "macos")]
    candidates.extend(
        [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        .map(PathBuf::from),
    );
    #[cfg(windows)]
    for (base, rel) in [
        ("ProgramFiles", r"Google\Chrome\Application\chrome.exe"),
        ("ProgramFiles(x86)", r"Google\Chrome\Application\chrome.exe"),
        ("LOCALAPPDATA", r"Google\Chrome\Application\chrome.exe"),
        // Edge installs under x86 even on 64-bit Windows.
        (
            "ProgramFiles(x86)",
            r"Microsoft\Edge\Application\msedge.exe",
        ),
        ("ProgramFiles", r"Microsoft\Edge\Application\msedge.exe"),
    ] {
        if let Ok(base) = std::env::var(base) {
            candidates.push(Path::new(&base).join(rel));
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    candidates.extend(
        [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
        ]
        .map(PathBuf::from),
    );
    candidates
}

const CHROME_DEADLINE: std::time::Duration = std::time::Duration::from_secs(15);
const CHROME_POLL: std::time::Duration = std::time::Duration::from_millis(120);

/// Modern Chrome writes the screenshot but the process lingers (the
/// self-exiting mode moved to chrome-headless-shell), so never wait for
/// exit: poll for the artifact, then kill.
fn try_chrome_screenshot(
    browser: &Path,
    profile: &Path,
    target: &std::ffi::OsStr,
    out: &Path,
    (w, h): (u32, u32),
) -> bool {
    use std::process::Stdio;
    // Null stdio: the crashpad child inherits pipes and outlives the kill.
    let Ok(mut child) = Command::new(browser)
        .arg("--headless")
        .arg(format!("--screenshot={}", out.display()))
        .arg(format!("--window-size={w},{h}"))
        // Fresh profile, or a running browser steals the invocation.
        .arg(format!("--user-data-dir={}", profile.display()))
        .args([
            "--default-background-color=00000000",
            "--no-first-run",
            "--hide-scrollbars",
            "--disable-gpu",
            // Screenshots need no Safe Storage; without this a missing login
            // keychain (fake $HOME) pops a blocking macOS dialog.
            "--use-mock-keychain",
        ])
        .arg(target)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };

    let deadline = std::time::Instant::now() + CHROME_DEADLINE;
    let mut last_len = 0;
    let mut ok = false;
    while std::time::Instant::now() < deadline {
        std::thread::sleep(CHROME_POLL);
        if let Ok(Some(status)) = child.try_wait() {
            ok = status.success() && out.exists();
            break;
        }
        match std::fs::metadata(out).map(|m| m.len()) {
            Ok(len) if len > 0 && len == last_len => {
                ok = true;
                break;
            }
            Ok(len) => last_len = len,
            Err(_) => {}
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    ok
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
    #[ignore = "spawns a real browser"]
    fn chrome_rung_screenshots() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("preview.svg");
        std::fs::write(
            &src,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\">\
<rect width=\"10\" height=\"10\" fill=\"#4a6\"/></svg>",
        )
        .unwrap();
        let out = dir.path().join("preview.png");
        let profile = dir.path().join("chrome-profile");
        let ok = chrome_candidates()
            .iter()
            .any(|b| try_chrome_screenshot(b, &profile, src.as_os_str(), &out, (1024, 1024)));
        assert!(ok, "no chrome-family browser produced a screenshot");
        let png = std::fs::read(&out).unwrap();
        assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn sniffs_html_content() {
        assert!(looks_like_html(b"<!DOCTYPE html><html><body/></html>"));
        assert!(looks_like_html(b"  <html lang=\"en\">"));
        assert!(looks_like_html(b"<!-- x --><html>"));
        assert!(!looks_like_html(b"<svg xmlns=\"x\"/>"));
        assert!(!looks_like_html(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn classifies_preview_targets() {
        use PreviewTargetKind::*;
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.path().join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        };
        assert_eq!(classify_preview_target(&write("a.png", b"")), Some(Image));
        assert_eq!(classify_preview_target(&write("a.svg", b"")), Some(Svg));
        assert_eq!(classify_preview_target(&write("a.html", b"")), Some(Html));
        assert_eq!(
            classify_preview_target(&write("img", &[0x89, b'P', b'N', b'G', 0, 0])),
            Some(Image)
        );
        assert_eq!(
            classify_preview_target(&write("vector", b"<svg xmlns=\"x\"/>")),
            Some(Svg)
        );
        assert_eq!(
            classify_preview_target(&write("page", b"<!doctype html><html/>")),
            Some(Html)
        );
        assert_eq!(
            classify_preview_target(&write("code.rs", b"fn main() {}")),
            None
        );
        assert_eq!(
            classify_preview_target(&dir.path().join("missing.txt")),
            None
        );
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
