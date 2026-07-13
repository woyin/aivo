//! Terminal styling utility using the console crate.
//! Provides cross-platform styling with ANSI fallback support.
use console::style;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;

/// Whether stderr is an interactive terminal. Spinners and other
/// in-place progress writers are no-ops when this is false so piped
/// output (`aivo logs 2>&1 | head`) stays clean.
fn stderr_is_tty() -> bool {
    io::stderr().is_terminal()
}

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

/// Renders text as a reverse-video "keycap" so a key to press stands out.
pub fn keycap(text: impl AsRef<str>) -> String {
    style(text.as_ref()).reverse().bold().to_string()
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

/// Section header: a dim rule sized to `text`, then `text` in bold.
pub fn print_header(text: &str) {
    println!("{}", dim("─".repeat(console::measure_text_width(text))));
    println!("{}", bold(text));
}

/// Shared bar/gauge width for `aivo stats` and `aivo account usage`.
pub const METER_WIDTH: usize = 14;

/// Fixed-`width` line meter: `value`/`max_value` as a cyan `━` rule, the rest a
/// muted `─` rail. Only a full ratio fills every cell; any partial keeps ≥1 rail
/// cell (99% never reads full) and any non-zero keeps ≥1 fill cell (so a tiny
/// value stays visible rather than vanishing).
pub fn meter(value: u64, max_value: u64, width: usize) -> String {
    let frac = if max_value == 0 {
        0.0
    } else {
        (value as f64 / max_value as f64).clamp(0.0, 1.0)
    };
    let filled = if frac >= 1.0 {
        width
    } else if frac > 0.0 {
        // `min().max()` (not `clamp`) so a width of 1 can't panic on min > max.
        ((frac * width as f64).round() as usize)
            .min(width.saturating_sub(1))
            .max(1)
    } else {
        0
    };
    let empty = width - filled;
    format!("{}{}", cyan("━".repeat(filled)), gray("─".repeat(empty)))
}

/// Starts a braille spinner on stderr. Returns the flag and join handle.
/// Pass an optional label to display after the spinner character.
pub fn start_spinner(label: Option<&str>) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let spinning = Arc::new(AtomicBool::new(true));
    let spinning_clone = spinning.clone();
    let label = label.map(str::to_owned).unwrap_or_default();

    if !stderr_is_tty() {
        let handle = tokio::task::spawn_blocking(move || {
            while spinning_clone.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(80));
            }
        });
        return (spinning, handle);
    }

    let first_frame = spinner_frame(0);
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

/// Like `start_spinner` but reads the label from a shared `Mutex<String>`
/// on every frame, so callers can update progress text in flight (e.g.
/// "(2/3) loading…") without restarting the spinner. The line is fully
/// erased between frames so a shrinking label leaves no trailing chars.
pub fn start_spinner_with_label(label: Arc<Mutex<String>>) -> (Arc<AtomicBool>, JoinHandle<()>) {
    let spinning = Arc::new(AtomicBool::new(true));
    let spinning_clone = spinning.clone();

    if !stderr_is_tty() {
        let handle = tokio::task::spawn_blocking(move || {
            while spinning_clone.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(80));
            }
        });
        return (spinning, handle);
    }

    let first_label = label.lock().map(|s| s.clone()).unwrap_or_default();
    let first_frame = spinner_frame(0);
    eprint!("\r\x1b[2K{}{}", dim(first_frame), first_label);
    let _ = io::stderr().flush();

    let handle = tokio::task::spawn_blocking(move || {
        let mut i = 1;
        while spinning_clone.load(Ordering::Relaxed) {
            let text = label.lock().map(|s| s.clone()).unwrap_or_default();
            eprint!("\r\x1b[2K{}{}", dim(spinner_frame(i)), text);
            let _ = io::stderr().flush();
            std::thread::sleep(std::time::Duration::from_millis(80));
            i += 1;
        }
    });
    (spinning, handle)
}

/// Stops the spinner and erases the entire line (so any label printed
/// alongside the spinner glyph is cleared too — `\r\x1b[2K` is "carriage
/// return + erase entire line", which works on every terminal where the
/// rest of `style::*` already assumes ANSI support).
pub fn stop_spinner(spinning: &Arc<AtomicBool>) {
    if spinning.swap(false, Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if stderr_is_tty() {
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts (filled `━`, rail `─`) cells, ignoring any surrounding styling.
    fn cells(s: &str) -> (usize, usize) {
        (
            s.chars().filter(|c| *c == '━').count(),
            s.chars().filter(|c| *c == '─').count(),
        )
    }

    #[test]
    fn meter_full_only_at_max() {
        assert_eq!(cells(&meter(100, 100, 20)), (20, 0));
    }

    #[test]
    fn meter_empty_at_zero_value() {
        assert_eq!(cells(&meter(0, 100, 20)), (0, 20));
    }

    #[test]
    fn meter_below_max_always_leaves_a_rail_cell() {
        // 99% must never render as full — it keeps ≥1 rail cell.
        assert_eq!(cells(&meter(99, 100, 20)), (19, 1));
        // Even 99.9% (rounds to 20) is held to width-1.
        assert_eq!(cells(&meter(999, 1000, 20)), (19, 1));
    }

    #[test]
    fn meter_tiny_nonzero_shows_one_tick() {
        // Any non-zero value keeps ≥1 fill cell so it stays visible.
        assert_eq!(cells(&meter(1, 1_000_000, 20)), (1, 19));
    }

    #[test]
    fn meter_no_max_is_all_rail() {
        assert_eq!(cells(&meter(50, 0, 20)), (0, 20));
    }

    #[test]
    fn meter_width_is_total_cells() {
        for value in [0u64, 1, 37, 99, 100] {
            let (f, e) = cells(&meter(value, 100, 20));
            assert_eq!(f + e, 20, "value={value} must fill exactly the width");
        }
    }
}
