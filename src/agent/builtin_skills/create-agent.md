---
name: create-agent
description: Create or improve an aivo subagent, a specialist the main agent delegates to, whenever the user wants to make, author, or scaffold a subagent or agent role.
---

# Create a Subagent

This is the guided workflow for building named specialist subagents and iteratively improving them. A subagent is a role the main agent can hand a self-contained task to — "Software Architect", "Code Writer", "Code Reviewer", "Test Writer" — each with its own instructions, and optionally its own model and tool scope. Your job is to figure out where the user is and move them forward: sometimes that's "I want a code-reviewer subagent" (start from scratch), sometimes "here's a draft, sharpen it" (go to review/iterate).

## What a subagent is

A subagent is a single `<name>.md` file with YAML frontmatter and a Markdown body that becomes its system prompt. aivo discovers them from these roots (project shadows user; first name wins):

- Project: `./.aivo/agents`, `./.claude/agents`  (ship with the repo)
- User: `~/.config/aivo/agents`  (available in every project — the dir aivo owns)
- Packs: `agents/` inside installed extension packs
- Built-in: `explorer` (read-only exploration), `aivo-guide` (aivo docs expert), `verification` (adversarial PASS/FAIL checks), `advisor` (read-only second opinion), and `evaluate` (code review) ship inside aivo, at the lowest precedence — creating a same-named file replaces them, and that's the supported way to customize one (start from `aivo code agents cat <name>`); warn before authoring a new agent under one of these names, since it silently shadows the built-in

The main agent's `subagent` tool gains an `agent` parameter naming which specialist to run; delegating to a name loads that profile's instructions (and model/tools) for the sub-run. Discovery is progressive-disclosure: only each subagent's **name + first sentence of its description** ride in the system prompt, so the main agent knows what it can delegate to without paying for every full body.

Profiles are resolved fresh at delegation time, so a subagent you create is usable **immediately** — no restart (its one-line advert in the system prompt refreshes next turn). aivo also reads Claude Code's `.claude/agents/` verbatim, so an existing fleet works as-is.

```
~/.config/aivo/agents/code-reviewer.md   (one file = one subagent)
```

## Step 1 — Capture intent

Understand what the user wants before writing anything. The current conversation may already point at the role to capture ("I keep asking you to review diffs — make that a subagent") — mine it. Then confirm:

1. What **task** should this specialist own end to end?
2. **When** should the main agent delegate to it — what kind of request?
3. Does it need a **different model** (e.g. a stronger one for architecture, a cheaper one for mechanical work)?
4. Should its **tools be scoped** (e.g. a reviewer that reads and greps but never writes)?
5. Should it run in an **isolated worktree** (parallel file-mutating work that shouldn't touch the tree)?

Come back with a proposal rather than offloading every decision. Default to global (`~/.config/aivo/agents`) unless the role is specific to this repo — then write it into `./.aivo/agents` so it ships with the project.

## Step 2 — Write the `<name>.md`

Create the file with your file tools (`write_file`). Fill in:

- **name** — the identifier and filename: letters, digits, `-`, `_`. This is the value the main agent passes as `agent` to delegate.
- **description** — the single most important field: it's the *entire* advert the main agent sees when deciding whether to delegate. aivo advertises only the **first sentence** (cut at the first `". "`, ~160 chars), so write ONE sentence that says what the specialist does AND when to hand work to it. Keep trigger cues inside that first sentence (commas/dashes, not new sentences) — anything after the first period never reaches the main agent. Lean slightly pushy; models under-delegate.
- **model** *(optional)* — a model id to run this specialist on. Omit to inherit the parent's model (Claude Code's `model: inherit` is honored the same way). Use a full id the user's provider resolves — bare shorthands like `sonnet` are passed through verbatim and may fail. Use for "have a stronger model do the architecture" or "run the mechanical pass on something cheap".
- **tools** *(optional)* — an allow-list scoping the specialist's toolset. Accepts aivo names (`read_file`, `run_bash`, `write_file`, `edit_file`, `grep`) or Claude Code names (`Read`, `Bash`, `Write`, `Edit`, `Grep`) — unknown names are ignored, never stripping to zero. Use an inline list: `tools: [read_file, grep, run_bash]` or `tools: read_file, grep`. A block-style YAML list is treated as unscoped, so keep it inline. Omit for full tools. A no-edit reviewer is the classic case: give it read/grep — plus bash only if it must run tests, since bash can still mutate files — and no write/edit.
- **isolation** *(optional)* — `isolation: worktree` runs the specialist in a disposable git worktree snapshot of HEAD, so parallel file-mutating subagents don't collide; an unchanged worktree is auto-removed, a changed one is kept with apply/cleanup instructions in the result.
- **the body** — the specialist's actual instructions, shaped by the writing guidance below.

### Writing style

- Prefer the imperative ("Flag every public API without a doc comment", not "you should look for").
- **Explain the why.** A rule the specialist understands beats one it obeys blindly. Reserve hard MUSTs/NEVERs for the few places a mistake is genuinely costly.
- Give it a clear **definition of done** and an **output shape** — a subagent returns a report to the main agent, so say what that report should contain (findings list, a patch, a file path it wrote).
- Keep it focused on the one role. A subagent runs across many prompts you haven't seen; don't overfit to today's example.

### A minimal example

```
~/.config/aivo/agents/code-reviewer.md

---
name: code-reviewer
description: Review a diff or file for correctness bugs, security issues, and unclear code, whenever the user wants a code review or a second pass before committing.
model: claude-opus-4-8
tools: [read_file, grep, run_bash]
---

# Code Reviewer

You review code changes for correctness and clarity. You do not edit files — you report.

1. Read the diff (`git diff`, or the files named in the task).
2. Flag, in priority order: correctness bugs, security issues, then clarity/maintainability. Skip nits unless nothing else is wrong.
3. For each finding give `file:line`, what's wrong, and the fix. If it's clean, say so plainly.
```

## Step 3 — Test it

After the draft, delegate a realistic task to the new subagent and read what comes back. If you have the `subagent` tool, call it with `agent: "<name>"` and a prompt a real user would give — profiles are resolved fresh at delegation, so the file you just wrote is delegatable right away. If the result notes `no profile named …`, it ran a generic sub-agent (the output proves nothing about the profile): the filename or `name:` frontmatter is wrong (letters, digits, `-`, `_`) or the file is in the wrong dir — fix and re-delegate.

Read the **report it returns**, not just that it ran: did it stay in role, respect its tool scope, and produce the output shape you asked for? A reviewer that starts editing files, or an architect that returns vague prose, is a signal to tighten the body.

## Step 4 — Improve, then repeat

1. **Generalize from the feedback** — fix the underlying instruction, not the one example.
2. **Keep it lean** — delete instructions that aren't pulling their weight.
3. **Explain the why** — terse feedback usually points at a real need; transmit that understanding into the body.

Apply, re-delegate, get feedback again. Stop when the user is happy or you've stopped making meaningful progress.

## Step 5 — Sharpen the description (delegation triggering)

The description decides whether the main agent ever delegates, so give it a dedicated pass. Pressure-test with requests that **should** route to this specialist and near-misses that **should not**. Tune that single advertised sentence (remember only up to the first `". "` is shown) until the right requests trigger delegation without false ones leaking in.

## Don't surprise the user

A subagent must not contain anything that could compromise the user's system, and its behavior should match what its description claims. Decline requests to build a subagent meant to enable unauthorized access or data exfiltration. Note that repo-local subagents (`./.aivo/agents`, `./.claude/agents`) are treated as untrusted content by the main agent, since a repo can ship them.

## When you're done

Tell the user where the subagent lives and how it's used: the main agent will delegate to it automatically when a request matches its description, and can be told explicitly ("use the code-reviewer subagent"). To make a global one repo-specific (or vice versa), move the file between `~/.config/aivo/agents/` and the repo's `./.aivo/agents/`.
