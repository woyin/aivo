//! Strip ANSI escape sequences from terminal output. Good enough for
//! Node/Ink CLIs; not a general-purpose VT parser. One escape grammar lives
//! here — callers pick a [`ControlPolicy`] for the non-escape control bytes.

use std::borrow::Cow;

/// How [`scrub`] treats control characters outside escape sequences.
#[derive(Clone, Copy, PartialEq)]
pub enum ControlPolicy {
    /// Keep them all (newlines, tabs, CR, …) — escapes-only stripping.
    Keep,
    /// Drop every control character; tabs become single spaces.
    Drop,
    /// Like `Drop`, but newlines survive (for multi-line text).
    DropExceptNewlines,
}

/// Strips CSI, OSC, and 2-byte ANSI escapes.
pub fn strip_ansi(s: &str) -> String {
    scrub(s, ControlPolicy::Keep).into_owned()
}

/// Strip escape sequences and apply `policy` to remaining control bytes.
/// Returns the input borrowed when nothing needs removing. An unterminated
/// CSI/OSC is consumed to end-of-input, so a truncated escape can never leak
/// live bytes into the output.
pub fn scrub(s: &str, policy: ControlPolicy) -> Cow<'_, str> {
    let needs_scrub = |c: char| match policy {
        ControlPolicy::Keep => c == '\x1b',
        ControlPolicy::Drop => c.is_control(),
        ControlPolicy::DropExceptNewlines => c.is_control() && c != '\n',
    };
    let Some(start) = s.find(needs_scrub) else {
        return Cow::Borrowed(s);
    };

    let mut out = String::with_capacity(s.len());
    out.push_str(&s[..start]);
    let mut chars = s[start..].chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek() {
                // CSI: ESC [ … final byte in 0x40..=0x7e.
                Some('[') => {
                    chars.next();
                    for d in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            break;
                        }
                    }
                }
                // OSC: ESC ] … terminated by BEL or ST (ESC \).
                Some(']') => {
                    chars.next();
                    while let Some(d) = chars.next() {
                        if d == '\x07' {
                            break;
                        }
                        if d == '\x1b' {
                            if chars.peek() == Some(&'\\') {
                                chars.next();
                            }
                            break;
                        }
                    }
                }
                // Lone ESC or a 2-byte escape — drop the following byte.
                _ => {
                    chars.next();
                }
            }
            continue;
        }
        match policy {
            ControlPolicy::Keep => out.push(c),
            _ if c == '\t' => out.push(' '),
            ControlPolicy::DropExceptNewlines if c == '\n' => out.push(c),
            _ if c.is_control() => {}
            _ => out.push(c),
        }
    }
    Cow::Owned(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_ansi_removes_escapes_and_keeps_controls() {
        assert_eq!(strip_ansi("a\x1b[31mb\x1b[0m\tc\nd"), "ab\tc\nd");
        assert_eq!(strip_ansi("t\x1b]0;title\x07x"), "tx");
    }

    #[test]
    fn scrub_drop_except_newlines() {
        let mixed = "line one\x1b[31m red\x1b[0m\nline\ttwo\x1b]0;title\x07 end\r";
        assert_eq!(
            scrub(mixed, ControlPolicy::DropExceptNewlines),
            "line one red\nline two end"
        );
        // Clean text takes the borrowed fast path.
        assert!(matches!(
            scrub("plain\ntext", ControlPolicy::DropExceptNewlines),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn scrub_drop_flattens_to_one_line() {
        assert_eq!(scrub("a\nb\tc\rd", ControlPolicy::Drop), "ab cd");
    }

    #[test]
    fn scrub_consumes_truncated_escape_without_leaking() {
        assert_eq!(scrub("ok\x1b[31", ControlPolicy::Drop), "ok");
        assert_eq!(scrub("ok\x1b]0;half", ControlPolicy::Drop), "ok");
    }
}
