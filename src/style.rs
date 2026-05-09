/**
 * Terminal styling utility using the console crate.
 * Provides cross-platform styling with ANSI fallback support.
 */
use console::style;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task::JoinHandle;

const BRAILLE_SPINNER_FRAMES: [&str; 10] = [
    "\u{280b}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283c}", "\u{2834}", "\u{2826}", "\u{2827}",
    "\u{2807}", "\u{280f}",
];

/// Supported style names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StyleName {
    Bold,
    Dim,
    Red,
    Green,
    Yellow,
    Blue,
    Cyan,
    #[allow(dead_code)]
    Magenta,
}

/// Styles text using the console crate.
/// On Windows without ANSI support, console handles the fallback automatically.
pub fn style_text(style_name: StyleName, text: impl AsRef<str>) -> String {
    let text = text.as_ref();

    match style_name {
        StyleName::Bold => style(text).bold().to_string(),
        StyleName::Dim => style(text).dim().to_string(),
        StyleName::Red => style(text).red().to_string(),
        StyleName::Green => style(text).green().to_string(),
        StyleName::Yellow => style(text).yellow().to_string(),
        StyleName::Blue => style(text).blue().to_string(),
        StyleName::Cyan => style(text).cyan().to_string(),
        StyleName::Magenta => style(text).magenta().to_string(),
    }
}

/// Convenience function to style text as cyan (commonly used in the CLI).
pub fn cyan(text: impl AsRef<str>) -> String {
    style_text(StyleName::Cyan, text)
}

/// Convenience function to style text as green (for success).
pub fn green(text: impl AsRef<str>) -> String {
    style_text(StyleName::Green, text)
}

/// Convenience function to style text as red (for errors).
pub fn red(text: impl AsRef<str>) -> String {
    style_text(StyleName::Red, text)
}

/// Convenience function to style text as yellow (for warnings/notes).
pub fn yellow(text: impl AsRef<str>) -> String {
    style_text(StyleName::Yellow, text)
}

/// Convenience function to style text as dim (for secondary information).
pub fn dim(text: impl AsRef<str>) -> String {
    style_text(StyleName::Dim, text)
}

/// Renders text in a true 256-color dark gray. More muted than `dim`, which is
/// just reduced brightness of the default foreground. Use for disabled or
/// deeply de-emphasized content.
pub fn gray(text: impl AsRef<str>) -> String {
    style(text.as_ref()).color256(240).to_string()
}

/// Convenience function to style text as bold.
pub fn bold(text: impl AsRef<str>) -> String {
    style_text(StyleName::Bold, text)
}

/// Convenience function to style text as blue.
pub fn blue(text: impl AsRef<str>) -> String {
    style_text(StyleName::Blue, text)
}

/// Convenience function to style text as magenta/purple.
pub fn magenta(text: impl AsRef<str>) -> String {
    style_text(StyleName::Magenta, text)
}

/// Convenience function for the "✓" success symbol.
pub fn success_symbol() -> String {
    green("✓")
}

/// Convenience function for the "→" arrow symbol.
pub fn arrow_symbol() -> String {
    cyan("→")
}

/// Convenience function for the "●" bullet symbol.
pub fn bullet_symbol() -> String {
    green("●")
}

/// Convenience function for the "○" empty bullet symbol.
pub fn empty_bullet_symbol() -> String {
    dim("○")
}

pub fn spinner_frame(index: usize) -> &'static str {
    BRAILLE_SPINNER_FRAMES[index % BRAILLE_SPINNER_FRAMES.len()]
}

/// Renders a horizontal bar using block-element glyphs with eighth-block resolution.
/// `value` is drawn proportionally to `max_value`, up to `width` full-block characters.
/// Non-zero values render at least the thinnest partial-block glyph.
pub fn bar(value: u64, max_value: u64, width: usize) -> String {
    if max_value == 0 || value == 0 || width == 0 {
        return String::new();
    }
    let eighths = ((value as f64 / max_value as f64) * (width * 8) as f64).round() as usize;
    let eighths = eighths.min(width * 8);
    let full = eighths / 8;
    let frac = eighths % 8;
    let mut s = "\u{2588}".repeat(full);
    if let Some(ch) = [
        "", "\u{258f}", "\u{258e}", "\u{258d}", "\u{258c}", "\u{258b}", "\u{258a}", "\u{2589}",
    ][frac]
        .chars()
        .next()
    {
        s.push(ch);
    }
    if s.is_empty() {
        s.push('\u{258f}');
    }
    s
}

/// Starts a braille spinner on stderr. Returns the flag and join handle.
/// Pass an optional label to display after the spinner character.
pub fn start_spinner(label: Option<&str>) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let spinning = Arc::new(AtomicBool::new(true));
    let spinning_clone = spinning.clone();
    let label = label.map(str::to_owned).unwrap_or_default();
    let first_frame = spinner_frame(0);

    // Paint the first frame synchronously so short operations still show feedback.
    eprint!("\r{}{}", dim(first_frame), label);
    let _ = io::stderr().flush();

    let handle = tokio::task::spawn_blocking(move || {
        let mut i = 1;
        while spinning_clone.load(Ordering::Relaxed) {
            eprint!("\r{}{}", dim(spinner_frame(i)), label);
            let _ = io::stderr().flush();
            std::thread::sleep(std::time::Duration::from_millis(80));
            i += 1;
        }
    });
    (spinning, handle)
}

/// Stops the spinner and clears its character from the line.
pub fn stop_spinner(spinning: &Arc<AtomicBool>) {
    if spinning.swap(false, Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        eprint!("\r \r");
        let _ = io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_text() {
        let styled = style_text(StyleName::Cyan, "test");
        assert!(!styled.is_empty());
        assert!(styled.contains("test"));
    }

    #[test]
    fn test_convenience_functions() {
        assert!(!cyan("test").is_empty());
        assert!(!green("test").is_empty());
        assert!(!red("test").is_empty());
        assert!(!yellow("test").is_empty());
        assert!(!dim("test").is_empty());
        assert!(!bold("test").is_empty());
        assert!(!blue("test").is_empty());
    }

    #[test]
    fn test_symbols() {
        assert!(!success_symbol().is_empty());
        assert!(!arrow_symbol().is_empty());
        assert!(!bullet_symbol().is_empty());
        assert!(!empty_bullet_symbol().is_empty());
    }

    #[test]
    fn bar_renders_full_width_at_max() {
        assert_eq!(bar(100, 100, 20), "\u{2588}".repeat(20));
    }

    #[test]
    fn bar_renders_half_width_at_half() {
        assert_eq!(bar(50, 100, 20), "\u{2588}".repeat(10));
    }

    #[test]
    fn bar_returns_empty_for_zero_inputs() {
        assert_eq!(bar(0, 100, 20), "");
        assert_eq!(bar(10, 0, 20), "");
        assert_eq!(bar(10, 100, 0), "");
    }

    #[test]
    fn bar_shows_minimum_partial_block_for_tiny_values() {
        let b = bar(1, 1000, 20);
        assert!(!b.is_empty());
    }

    #[test]
    fn bar_never_exceeds_requested_width_under_rounding() {
        // Float rounding could otherwise push eighths above width * 8.
        let b = bar(u64::MAX, u64::MAX, 32);
        assert_eq!(b.chars().count(), 32);
    }
}
