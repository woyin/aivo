use console::{Key, Term};

impl Default for FuzzySelect {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FuzzySelect {
    prompt: String,
    items: Vec<String>,
    /// Optional per-item annotation. `Some(s)` marks the item as disabled and
    /// renders `s` as a dim suffix; `None` leaves it selectable. When this vec
    /// is empty or shorter than `items`, missing entries are treated as
    /// enabled.
    annotations: Vec<Option<String>>,
    default: usize,
}

impl FuzzySelect {
    pub fn new() -> Self {
        Self {
            prompt: "Select".to_string(),
            items: Vec::new(),
            annotations: Vec::new(),
            default: 0,
        }
    }

    pub fn with_prompt(mut self, prompt: &str) -> Self {
        self.prompt = prompt.to_string();
        self
    }

    pub fn items(mut self, items: &[String]) -> Self {
        self.items = items.to_vec();
        self
    }

    /// Per-item annotations. `annotations[i] = Some(reason)` disables the item
    /// and renders `reason` as a dim suffix; `None` (or a shorter vec) leaves
    /// the item selectable.
    pub fn annotations(mut self, annotations: Vec<Option<String>>) -> Self {
        self.annotations = annotations;
        self
    }

    pub fn default(mut self, default: usize) -> Self {
        self.default = default;
        self
    }

    fn is_disabled(&self, idx: usize) -> bool {
        self.annotations
            .get(idx)
            .map(Option::is_some)
            .unwrap_or(false)
    }

    fn annotation(&self, idx: usize) -> Option<&str> {
        self.annotations.get(idx).and_then(Option::as_deref)
    }

    pub fn interact_opt(self) -> std::io::Result<Option<usize>> {
        let term = Term::stderr();
        term.hide_cursor()?;

        let mut query = String::new();
        let mut selection = self.default.min(self.items.len().saturating_sub(1));
        // If the default index lands on a disabled item, nudge forward to the
        // first enabled one so the picker opens on a usable row. Falls back to
        // the original default if every item is disabled.
        if self.is_disabled(selection)
            && let Some(first_enabled) = (0..self.items.len()).find(|&i| !self.is_disabled(i))
        {
            selection = first_enabled;
        }
        let mut page_start = 0;
        let page_size = 10;

        loop {
            let mut filtered: Vec<(usize, &String)> = self
                .items
                .iter()
                .enumerate()
                .filter(|(_, item)| matches_fuzzy(&query, item))
                .collect();

            if !query.is_empty() {
                // `sort_by_cached_key` scores each item once and is stable, so
                // equal-score items keep their original insertion order.
                filtered.sort_by_cached_key(|(_, item)| score_match(&query, item));
            }

            let count = filtered.len();

            if selection >= count {
                selection = count.saturating_sub(1);
            }

            if count > 0 && self.is_disabled(filtered[selection].0) {
                selection =
                    next_enabled_filtered(&filtered, selection, true, |i| self.is_disabled(i));
            }

            if selection < page_start {
                page_start = selection;
            } else if selection >= page_start + page_size {
                page_start = selection.saturating_sub(page_size).saturating_add(1);
            }

            // Pin the window to the bottom when selection is near the end so
            // trailing disabled rows stay visible — navigation skips them, so
            // `selection` alone can't pull `page_start` far enough to show
            // them (the user would otherwise see "↓ N more below" forever).
            if count > page_size && count.saturating_sub(selection) < page_size {
                page_start = count.saturating_sub(page_size);
            }

            if page_start > count.saturating_sub(1) {
                page_start = count.saturating_sub(1);
            }

            let end_idx = (page_start + page_size).min(count);

            let term_width = term.size().1 as usize;

            let hint = if query.is_empty() && count > page_size {
                format!(" {}", crate::style::dim("(type to filter)"))
            } else {
                String::new()
            };
            let prompt_line = format!("{}: {}{}", crate::style::bold(&self.prompt), query, hint);
            term.write_line(&truncate_to_width(&prompt_line, term_width))?;

            let items_drawn = if count == 0 {
                term.write_line(&format!("  {}", crate::style::dim("(no matches)")))?;
                1
            } else {
                let mut lines = 0;
                if page_start > 0 {
                    let above = page_start;
                    term.write_line(&format!(
                        "  {}",
                        crate::style::dim(format!("↑ {} more above", above))
                    ))?;
                    lines += 1;
                }
                for (i, (orig_idx, item)) in
                    filtered.iter().enumerate().take(end_idx).skip(page_start)
                {
                    let is_selected = i == selection;
                    let disabled = self.is_disabled(*orig_idx);
                    let symbol = if is_selected {
                        crate::style::cyan(">")
                    } else {
                        " ".to_string()
                    };
                    let styled_item = if disabled {
                        // Strip ANSI baked into `item` by callers (e.g.
                        // `format_key_choice` paints the short-id cyan and the
                        // base-url dim) so the outer yellow applies
                        // uniformly instead of getting overridden mid-string.
                        crate::style::yellow(console::strip_ansi_codes(item))
                    } else if is_selected {
                        crate::style::cyan(item)
                    } else {
                        crate::style::dim(item)
                    };
                    let suffix = self
                        .annotation(*orig_idx)
                        .map(|reason| format!("  {}", crate::style::dim(format!("({reason})"))))
                        .unwrap_or_default();
                    let line = format!("{} {}{}", symbol, styled_item, suffix);
                    term.write_line(&truncate_to_width(&line, term_width))?;
                    lines += 1;
                }
                if end_idx < count {
                    let below = count - end_idx;
                    term.write_line(&format!(
                        "  {}",
                        crate::style::dim(format!("↓ {} more below", below))
                    ))?;
                    lines += 1;
                }
                lines
            };

            let key = match term.read_key_raw() {
                Ok(key) => key,
                Err(e) => {
                    let _ = term.clear_last_lines(1 + items_drawn);
                    let _ = term.show_cursor();
                    return Err(e);
                }
            };

            // Clear drawn lines before next iteration or exit
            term.clear_last_lines(1 + items_drawn)?;

            match key {
                key if is_previous_key(&key) && count > 0 => {
                    selection =
                        next_enabled_filtered(&filtered, selection, false, |i| self.is_disabled(i));
                }
                key if is_next_key(&key) && count > 0 => {
                    selection =
                        next_enabled_filtered(&filtered, selection, true, |i| self.is_disabled(i));
                }
                Key::Enter => {
                    if count > 0 {
                        let orig_idx = filtered[selection].0;
                        if self.is_disabled(orig_idx) {
                            // Safety net — only reachable if every filtered
                            // row is disabled, in which case there's nothing
                            // to select. Ignore the keypress.
                            continue;
                        }
                        term.show_cursor()?;
                        return Ok(Some(orig_idx));
                    }
                    term.show_cursor()?;
                    return Ok(None);
                }
                Key::Escape | Key::CtrlC => {
                    term.show_cursor()?;
                    return Ok(None);
                }
                Key::Backspace if !query.is_empty() => {
                    query.pop();
                    selection = 0;
                    page_start = 0;
                }
                Key::Char(c) if !c.is_control() => {
                    query.push(c);
                    selection = 0;
                    page_start = 0;
                }
                _ => {}
            }
        }
    }
}

/// Advances `selection` within `filtered` until it lands on an enabled row,
/// skipping any disabled rows. The search wraps; if every row in `filtered`
/// is disabled, `selection` is returned unchanged. `forward` controls whether
/// the scan moves Down (`true`) or Up (`false`).
///
/// `selection` and the return value are both indices into `filtered`;
/// `is_disabled` receives the original item index (`filtered[i].0`).
fn next_enabled_filtered(
    filtered: &[(usize, &String)],
    selection: usize,
    forward: bool,
    is_disabled: impl Fn(usize) -> bool,
) -> usize {
    let count = filtered.len();
    if count == 0 {
        return 0;
    }
    let start = selection.min(count - 1);
    for step in 1..=count {
        let idx = if forward {
            (start + step) % count
        } else {
            (start + count - (step % count)) % count
        };
        if !is_disabled(filtered[idx].0) {
            return idx;
        }
    }
    start
}

fn is_previous_key(key: &Key) -> bool {
    matches!(key, Key::ArrowUp | Key::Char('\x10')) || matches_application_arrow(key, 'A')
}

fn is_next_key(key: &Key) -> bool {
    matches!(key, Key::ArrowDown | Key::Char('\x0e')) || matches_application_arrow(key, 'B')
}

fn matches_application_arrow(key: &Key, direction: char) -> bool {
    matches!(key, Key::UnknownEscSeq(seq) if seq.as_slice() == ['O', direction])
}

/// Truncate a string to fit within terminal width, accounting for ANSI escape codes.
/// ANSI sequences are not visible characters, so we track visible width separately.
fn truncate_to_width(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let mut visible = 0;
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            result.push(c);
        } else if in_escape {
            result.push(c);
            if c.is_ascii_alphabetic() {
                in_escape = false;
            }
        } else {
            if visible >= width {
                break;
            }
            result.push(c);
            visible += 1;
        }
    }
    result
}

/// Ranks a matching target against a query. Lower is better.
///
/// Assumes `matches_fuzzy(query, target)` is true (i.e. already passed filter).
/// Tuple components, in priority order:
///   0. rank: 0 = case-insensitive prefix, 1 = case-insensitive substring,
///            2 = subsequence-only.
///   1. position: byte index of the match start (earlier wins). For
///      subsequence matches, the byte index of the first query char in target.
///   2. target length: shorter target wins for otherwise-equal matches, so
///      exact-length hits float above longer strings with the query embedded.
pub(crate) fn score_match(query: &str, target: &str) -> (u8, usize, usize) {
    // ASCII case-folding to stay consistent with `matches_fuzzy`, which uses
    // `eq_ignore_ascii_case`. Picker content (provider names, URLs, model IDs)
    // is ASCII-dominant, and `target.len()` byte length is a fine tiebreak.
    let q_lower = query.to_ascii_lowercase();
    let t_lower = target.to_ascii_lowercase();

    if t_lower.starts_with(&q_lower) {
        return (0, 0, target.len());
    }
    if let Some(pos) = t_lower.find(&q_lower) {
        return (1, pos, target.len());
    }
    // Invariant: `matches_fuzzy(query, target)` was true, so the first query
    // char must appear somewhere in target.
    let first_q = q_lower
        .chars()
        .next()
        .expect("score_match called with empty query; guarded by !query.is_empty()");
    let pos = t_lower
        .char_indices()
        .find(|(_, c)| *c == first_q)
        .map(|(i, _)| i)
        .expect("matches_fuzzy guarantees the first query char is present");
    (2, pos, target.len())
}

pub(crate) fn matches_fuzzy(query: &str, target: &str) -> bool {
    let mut q_chars = query.chars();
    let mut current_q_char = match q_chars.next() {
        Some(c) => c,
        None => return true,
    };

    for c in target.chars() {
        if c.eq_ignore_ascii_case(&current_q_char) {
            current_q_char = match q_chars.next() {
                Some(next) => next,
                None => return true, // All query chars found
            };
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::{is_next_key, is_previous_key, next_enabled_filtered, score_match};
    use console::Key;

    fn filtered_fixture(items: &[&'static str]) -> Vec<(usize, &'static String)> {
        // `next_enabled_filtered` only reads the original index, but the
        // signature expects `&String` — leak short static strings to get
        // stable references for the test harness.
        items
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let owned: &'static String = Box::leak(Box::new((*s).to_string()));
                (i, owned)
            })
            .collect()
    }

    #[test]
    fn next_enabled_skips_disabled_going_forward() {
        let filtered = filtered_fixture(&["a", "b", "c", "d"]);
        // Items 1 and 2 disabled — Down from 0 should land on 3.
        let disabled = |i: usize| i == 1 || i == 2;
        assert_eq!(next_enabled_filtered(&filtered, 0, true, disabled), 3);
    }

    #[test]
    fn next_enabled_skips_disabled_going_backward() {
        let filtered = filtered_fixture(&["a", "b", "c", "d"]);
        // Items 1 and 2 disabled — Up from 3 should land on 0.
        let disabled = |i: usize| i == 1 || i == 2;
        assert_eq!(next_enabled_filtered(&filtered, 3, false, disabled), 0);
    }

    #[test]
    fn next_enabled_wraps_around() {
        let filtered = filtered_fixture(&["a", "b", "c"]);
        // Only item 0 enabled — Down from 0 should wrap back to 0.
        let only_first_enabled = |i: usize| i != 0;
        assert_eq!(
            next_enabled_filtered(&filtered, 0, true, only_first_enabled),
            0
        );
        // Up from 0 should also return 0.
        assert_eq!(
            next_enabled_filtered(&filtered, 0, false, only_first_enabled),
            0
        );
    }

    #[test]
    fn next_enabled_keeps_selection_when_all_disabled() {
        let filtered = filtered_fixture(&["a", "b", "c"]);
        let all_disabled = |_: usize| true;
        assert_eq!(next_enabled_filtered(&filtered, 1, true, all_disabled), 1);
        assert_eq!(next_enabled_filtered(&filtered, 1, false, all_disabled), 1);
    }

    #[test]
    fn next_enabled_handles_empty_slice() {
        let filtered: Vec<(usize, &String)> = Vec::new();
        assert_eq!(next_enabled_filtered(&filtered, 0, true, |_| false), 0);
    }

    #[test]
    fn recognizes_application_cursor_mode_arrows() {
        assert!(is_previous_key(&Key::UnknownEscSeq(vec!['O', 'A'])));
        assert!(is_next_key(&Key::UnknownEscSeq(vec!['O', 'B'])));
    }

    #[test]
    fn recognizes_standard_navigation_shortcuts() {
        assert!(is_previous_key(&Key::ArrowUp));
        assert!(is_previous_key(&Key::Char('\x10')));
        assert!(is_next_key(&Key::ArrowDown));
        assert!(is_next_key(&Key::Char('\x0e')));
    }

    #[test]
    fn score_prefers_prefix_over_substring_over_subsequence() {
        // "openai" prefix-matches the literal "openai" provider label, contains-matches
        // "groq …/openai/v1", and only subsequence-matches "OpenRouter …openrouter.ai".
        let prefix = score_match("openai", "openai   https://api.openai.com");
        let substr = score_match("openai", "groq     https://api.groq.com/openai/v1");
        let subseq = score_match("openai", "OpenRouter https://openrouter.ai/api/v1");
        assert_eq!(prefix.0, 0);
        assert_eq!(substr.0, 1);
        assert_eq!(subseq.0, 2);
        assert!(prefix < substr);
        assert!(substr < subseq);
    }

    #[test]
    fn score_prefers_earlier_substring_match() {
        let early = score_match("api", "api-server  https://example.com");
        let late = score_match("api", "Zhipu AI    https://open.bigmodel.cn/api/v4");
        // Both are substring matches (rank 1), earlier position wins.
        assert_eq!(early.0, 0); // actually starts_with → rank 0, even better
        assert!(early < late);

        // Two substring matches, neither prefix.
        let a = score_match("foo", "bar foo baz");
        let b = score_match("foo", "bar baz foo");
        assert_eq!(a.0, 1);
        assert_eq!(b.0, 1);
        assert!(a < b);
    }

    #[test]
    fn score_is_case_insensitive() {
        let upper = score_match("OPENAI", "openai   https://api.openai.com");
        let lower = score_match("openai", "OPENAI   https://api.openai.com");
        assert_eq!(upper.0, 0);
        assert_eq!(lower.0, 0);
    }

    #[test]
    fn score_tiebreaks_by_length_for_equal_rank_and_position() {
        let short = score_match("foo", "foo");
        let long = score_match("foo", "foobar baz qux");
        assert_eq!(short.0, 0);
        assert_eq!(long.0, 0);
        assert!(short < long);
    }

    #[test]
    fn stable_sort_preserves_original_order_within_ties() {
        // Two items with identical scores: sort_by_cached_key is stable.
        let items = ["openai one".to_string(), "openai two".to_string()];
        let mut filtered: Vec<(usize, &String)> = items.iter().enumerate().collect();
        filtered.sort_by_cached_key(|(_, item)| score_match("openai", item));
        assert_eq!(filtered[0].0, 0);
        assert_eq!(filtered[1].0, 1);
    }
}
