/// Claude Code's form; a profile can't collide with a same-named path.
pub(super) const AGENT_MENTION_PREFIX: &str = "agent-";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct MentionToken {
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FileMention {
    pub(super) token: MentionToken,
    /// Absolute, so the send-time read can't depend on the process cwd.
    pub(super) path: String,
}

const TRAILING_PUNCTUATION: [char; 12] =
    ['.', ',', ';', ':', '!', '?', ')', ']', '}', '>', '\'', '"'];

/// Start of draft or after whitespace, so emails and `user@host` don't count.
fn at_word_boundary(draft: &str, at: usize) -> bool {
    draft[..at]
        .chars()
        .next_back()
        .is_none_or(char::is_whitespace)
}

/// An unclosed quote is still being typed, not a token.
pub(super) fn mention_tokens(draft: &str) -> Vec<MentionToken> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = draft[search_from..].find('@') {
        let at = search_from + rel;
        let rest = &draft[at + 1..];
        if !at_word_boundary(draft, at) {
            search_from = at + 1;
            continue;
        }
        if let Some(quoted) = rest.strip_prefix('"') {
            match quoted.find(['"', '\n']) {
                Some(close) if quoted.as_bytes()[close] == b'"' && close > 0 => {
                    let end = at + 2 + close + 1;
                    out.push(MentionToken {
                        start: at,
                        end,
                        text: quoted[..close].to_string(),
                    });
                    search_from = end;
                }
                _ => search_from = at + 1,
            }
            continue;
        }
        let len = rest.find(char::is_whitespace).unwrap_or(rest.len());
        if len == 0 {
            search_from = at + 1;
            continue;
        }
        out.push(MentionToken {
            start: at,
            end: at + 1 + len,
            text: rest[..len].to_string(),
        });
        search_from = at + 1 + len;
    }
    out
}

/// The `@` offset and partial after it; a leading quote is stripped so
/// `@"my dir/fo` keeps completing.
pub(super) fn mention_query_at(head: &str) -> Option<(usize, String)> {
    let at = head.rfind('@')?;
    if !at_word_boundary(head, at) {
        return None;
    }
    let rest = &head[at + 1..];
    let partial = match rest.strip_prefix('"') {
        Some(quoted) => {
            if quoted.contains(['"', '\n']) {
                return None;
            }
            quoted
        }
        None => {
            if rest.chars().any(char::is_whitespace) {
                return None;
            }
            rest
        }
    };
    Some((at, partial.to_string()))
}

/// Trailing sentence punctuation is tolerated (`@README.md,`); a file named
/// twice attaches once.
pub(super) fn resolve_file_mentions(
    draft: &str,
    cwd: &str,
    is_agent_handle: impl Fn(&str) -> bool,
) -> Vec<FileMention> {
    let mut out: Vec<FileMention> = Vec::new();
    for token in mention_tokens(draft) {
        if is_agent_handle(&token.text) {
            continue;
        }
        let quoted = draft[token.start..].starts_with("@\"");
        let mut candidate = token.text.as_str();
        loop {
            if let Some(path) = regular_file(cwd, candidate) {
                if !out.iter().any(|m| m.path == path) {
                    let end = token.start + 1 + usize::from(quoted) * 2 + candidate.len();
                    out.push(FileMention {
                        token: MentionToken {
                            start: token.start,
                            end,
                            text: candidate.to_string(),
                        },
                        path,
                    });
                }
                break;
            }
            // One char at a time so `@a.b.` keeps `a.b`.
            match candidate.char_indices().next_back() {
                Some((i, c)) if !quoted && i > 0 && TRAILING_PUNCTUATION.contains(&c) => {
                    candidate = &candidate[..i];
                }
                _ => break,
            }
        }
    }
    out
}

fn regular_file(cwd: &str, path: &str) -> Option<String> {
    let expanded = crate::services::system_env::expand_tilde(path);
    let full = if expanded.is_absolute() {
        expanded
    } else {
        std::path::Path::new(cwd).join(expanded)
    };
    std::fs::metadata(&full)
        .is_ok_and(|m| m.is_file())
        .then(|| full.to_string_lossy().into_owned())
}

/// A directory keeps its quote open so the picker can descend.
pub(super) fn mention_text_for_path(path: &str, is_dir: bool) -> String {
    if path.chars().any(char::is_whitespace) {
        if is_dir {
            format!("@\"{path}")
        } else {
            format!("@\"{path}\"")
        }
    } else {
        format!("@{path}")
    }
}

pub(super) fn mention_draft_for_paths(paths: &[String]) -> String {
    paths
        .iter()
        .map(|p| format!("{} ", mention_text_for_path(p, false)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(draft: &str) -> Vec<(String, String)> {
        mention_tokens(draft)
            .into_iter()
            .map(|t| (draft[t.start..t.end].to_string(), t.text))
            .collect()
    }

    fn pair(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    #[test]
    fn tokens_respect_word_boundaries_and_quotes() {
        assert_eq!(
            spans("fix @src/main.rs and @\"my file.md\" now"),
            vec![
                pair("@src/main.rs", "src/main.rs"),
                pair("@\"my file.md\"", "my file.md")
            ]
        );
        assert!(spans("mail a@b.c or @ alone").is_empty());
        assert!(spans("see @\"half open").is_empty());
        assert_eq!(spans("@x"), vec![pair("@x", "x")]);
    }

    #[test]
    fn query_tracks_the_token_under_the_cursor() {
        assert_eq!(mention_query_at("use @co"), Some((4, "co".to_string())));
        assert_eq!(mention_query_at("@"), Some((0, String::new())));
        assert_eq!(
            mention_query_at("@\"my dir/fo"),
            Some((0, "my dir/fo".to_string()))
        );
        assert_eq!(mention_query_at("@\"done\""), None);
        assert_eq!(mention_query_at("@explorer go"), None);
        assert_eq!(mention_query_at("a@b"), None);
    }

    #[test]
    fn resolves_only_existing_files_and_tolerates_punctuation() {
        let dir = crate::test_sandbox::tmp("aivo-mentions");
        std::fs::write(dir.join("a.b"), "x").unwrap();
        std::fs::create_dir_all(dir.join("sub dir")).unwrap();
        std::fs::write(dir.join("sub dir").join("f.txt"), "y").unwrap();
        let cwd = dir.to_string_lossy().into_owned();
        let draft =
            "read @a.b. then @\"sub dir/f.txt\", skip @missing.rs @sub, @a.b @agent-explorer";
        let found = resolve_file_mentions(draft, &cwd, |t| t.starts_with(AGENT_MENTION_PREFIX));
        let typed: Vec<&str> = found.iter().map(|m| m.token.text.as_str()).collect();
        assert_eq!(
            typed,
            vec!["a.b", "sub dir/f.txt"],
            "dirs, missing files, agents and repeats skipped"
        );
        assert_eq!(found[0].path, dir.join("a.b").to_string_lossy());
        assert_eq!(&draft[found[0].token.start..found[0].token.end], "@a.b");
        assert_eq!(
            &draft[found[1].token.start..found[1].token.end],
            "@\"sub dir/f.txt\""
        );
    }

    #[test]
    fn path_text_quotes_whitespace_and_keeps_dirs_open() {
        assert_eq!(mention_text_for_path("src/", true), "@src/");
        assert_eq!(mention_text_for_path("my dir/", true), "@\"my dir/");
        assert_eq!(
            mention_text_for_path("my file.md", false),
            "@\"my file.md\""
        );
        assert_eq!(
            mention_draft_for_paths(&["a.png".into(), "b c.md".into()]),
            "@a.png @\"b c.md\" "
        );
    }
}
