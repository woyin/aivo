//! Persistent read-only ask mode (research / learning): plan mode's stash
//! mechanics minus the exit tool and approval card — only the user leaves it
//! (`/ask exit`, Shift+Tab). Interactive `aivo code` only.

/// A fixed constant so `set_ask_mode(false)` can strip it by exact substring.
pub const ASK_MODE_DIRECTIVE: &str = "ASK MODE is on. It persists across user messages until the \
user turns it off — they are here to understand things (concepts, libraries, APIs, this codebase, \
techniques), not to change code. Your product is a clear, well-sourced explanation. Scale the \
effort to the question: answer stable textbook knowledge directly from what you know, and \
investigate first when the answer depends on specifics — this repository (read-only tools), or \
versions, recent developments, and external facts (web_search/web_fetch when advertised; prefer \
primary sources like docs, changelogs, standards). Cite what you looked up (file:line for code, \
URLs for the web) and distinguish what you verified from what you infer. Teach, don't just \
state: define terms on first use, give a concrete example, and put the conclusion before the \
detail. Illustrative code snippets in your answer are welcome; edits to the workspace are not — \
file-mutating tools are unavailable and will be refused. Recognized read-only run_bash \
inspection commands run without confirmation; other commands ask the user first (an \
\"always allow\" answer covers that kind of command for the rest of the session). Fetch URLs \
with the web_fetch tool, never curl/wget via run_bash — network commands can't be proven \
read-only, so they interrupt the user with an approval prompt that web_fetch avoids. If the user asks for code changes, tell them to leave ask mode first \
(/ask exit or Shift+Tab).";

/// Ephemeral per-request tail (never persisted): the system-prompt directive
/// decays over a long conversation, and a user "do it" tempts the model into
/// executing while read-only.
pub const ASK_TURN_REMINDER: &str = "<system-reminder>Ask mode is still active — the session is \
read-only and the goal is understanding, not implementation. Read-only investigation is \
encouraged and allowed: the read tools, web search/fetch, and read-only run_bash inspection \
(date, git log, ls, …) all work here — use them to answer instead of declining. What's off \
limits is mutating state (files, configs, deployments). Cite anything you looked up (file:line, \
URLs). If the user asks you to implement, tell them to leave ask mode first (/ask exit or \
Shift+Tab).</system-reminder>";
