---
name: create-skill
description: Create a new aivo skill, or improve an existing one, end to end. Use whenever the user wants to make, author, scaffold, or design a skill, turn a workflow into a reusable skill, write or fix a SKILL.md, or sharpen a skill's description so it triggers reliably. Reach for this even when the user only says "make a skill for X" without naming SKILL.md.
---

# Create a Skill

This is the guided workflow for building new aivo skills and iteratively improving them. Your job is to figure out where the user is in the process and jump in to move them forward — sometimes that's "I want a skill for X" (start from scratch), sometimes it's "here's a draft, help me sharpen it" (go straight to review/iterate).

## What a skill is

A skill is a folder holding a `SKILL.md` with YAML frontmatter (`name`, `description`) and a Markdown body of instructions. aivo discovers skills from these roots (project shadows user; first name wins):

- Project: `./.agents/skills`, `./.aivo/skills`, `./.claude/skills`
- User: `~/.agents/skills`, `~/.config/aivo/skills`, `~/.claude/skills`

`~/.config/aivo/skills/<name>/` is the dir aivo owns and the one `/skills add` scaffolds into; `/skills add -p` (or `--project`) targets the repo's `./.agents/skills` instead — for skills meant to ship with the project (note: the agent advertises repo-local skills as untrusted content). Discovery is on-demand, so a skill you create is usable immediately — no restart: the user can run it with `/<name>` right away, it shows up in `/skills`, and the model starts offering it (via the `skill` tool) from the next turn on.

```
skill-name/
├── SKILL.md            (required: frontmatter + instructions)
└── (optional bundles)
    ├── scripts/        executable helpers for deterministic/repetitive work
    ├── references/     docs the model reads into context as needed
    └── assets/         templates/icons/fonts used in output
```

## Step 1 — Capture intent

Understand what the user wants before writing anything. The current conversation may already contain the workflow to capture ("turn this into a skill") — mine it for the tools used, the step sequence, the corrections the user made, and the input/output formats. Then confirm:

1. What should this skill let the agent **do**?
2. **When** should it trigger — what user phrases and contexts?
3. What's the expected **output** (format, files, side effects)?
4. Does it need bundled scripts/references, or is it pure instructions?

Ask about edge cases and example inputs proactively. Come back with context rather than offloading every decision onto the user. Match your language to their fluency — briefly define a term if you're unsure they'll know it.

## Step 2 — Write the SKILL.md

Create the folder and file. The fastest path is to scaffold a template and edit it: tell the user to run `/skills add <name> [one-line description]` (this writes `~/.config/aivo/skills/<name>/SKILL.md`; add `-p` to write into the repo's `./.agents/skills` for a skill that ships with the project), or just write the folder yourself with your file tools. If it isn't obvious whether the skill is personal or belongs to this repo, ask — default to personal, and to project when the workflow being captured is specific to this codebase. Then fill in:

- **name** — the identifier (and folder name): letters, digits, `-`, `_`.
- **description** — the single most important field, and your *entire* triggering budget. aivo advertises only the **first sentence** of the description to the model (it cuts at the first `". "`, capped at ~160 chars); that one sentence is all the model sees when deciding whether to reach for the skill. So write ONE sentence that states what the skill does AND its key when-to-use cues. Critically: don't write "Does X. Use when Y" — aivo drops everything after that first period, so "Use when Y" would never reach the model. Keep the trigger cues inside that first sentence (commas/dashes, not new sentences). Models tend to *under*-trigger, so lean slightly pushy: not "Build a dashboard," but "Build a fast internal-metrics dashboard whenever the user wants to show dashboards, data viz, or any company metrics — even without the word 'dashboard'."
- **the body** — the actual instructions (shaped by the writing guidance below).
- **arguments** — a skill runs as `/<name>` or `/<name> some text`. By default that trailing text is appended to your instructions as a final `Input: …` line. To place it mid-instruction instead, put the literal token `$ARGUMENTS` in the body and it's substituted in (and your body becomes the whole prompt, with no wrapper) — e.g. `Study the repository at $ARGUMENTS and summarize its architecture.`

### Progressive disclosure

Skills load in three levels — design for it:

1. **name + description** — always in context (~the advert). Keep it tight.
2. **SKILL.md body** — loaded when the skill triggers. Aim under ~500 lines.
3. **bundled resources** — read or executed only as needed (unlimited; a script can run without being loaded into context).

If the body approaches 500 lines, add a layer: move detail into `references/*.md` and point to it from SKILL.md ("read `references/aws.md` when deploying to AWS"). For a reference file over ~300 lines, give it a table of contents.

### Writing style

- Prefer the imperative ("Extract the columns", not "you should extract").
- **Explain the why.** Today's models have good theory of mind; a rule they understand beats a rule they obey blindly. If you catch yourself stacking ALL-CAPS MUSTs and NEVERs or rigid scaffolding, that's a yellow flag — reframe as the reasoning behind the constraint. Reserve hard imperatives for the few places a mistake is genuinely costly.
- Keep it general, not overfit to one example. A skill is meant to run thousands of times across prompts you haven't seen.
- Include a couple of concrete examples (input → output) where a format matters.

### Don't surprise the user

A skill must not contain malware, exploit code, or anything that could compromise the user's system, and its behavior should match what its description claims. Decline requests to build deceptive skills or skills meant to enable unauthorized access or data exfiltration. (Benign roleplay-style skills are fine.)

### A minimal example

A complete, well-shaped skill — note the single-sentence description that carries the whole trigger, and the lean imperative body:

```
~/.config/aivo/skills/changelog/SKILL.md

---
name: changelog
description: Summarize git commits since the last tag into release notes whenever the user asks for a changelog, release notes, or 'what changed'.
---

# Changelog

1. Find the last tag: `git describe --tags --abbrev=0` (fall back to the first commit if there are no tags).
2. Collect subjects: `git log <tag>..HEAD --no-merges --pretty=format:'%s'`.
3. Group under **Features / Fixes / Other** by conventional-commit prefix; write tight, user-facing bullets that explain impact, not mechanics.
```

## Step 3 — Test it

After the draft, write 2–3 realistic test prompts — the kind of thing a real user would actually type, not abstract one-liners. Show them to the user: "Here are a few cases I'd like to try — look right, or want to add any?" Then run them.

How to run depends on what's available:

- If you have the `subagent` tool, spawn the test prompts in parallel — and for a *new* skill, optionally run each prompt a second time with no skill access as a baseline, so you can see what the skill is actually adding. For an *improvement*, snapshot the old skill first and use it as the baseline.
- Otherwise, just run each prompt yourself, one at a time, following the skill's own instructions. You wrote it and you're running it, so this is a sanity check rather than a rigorous eval — the user's review is what makes it real.

Read the **transcripts**, not just the final outputs. If every run independently wrote the same helper script or took the same multi-step detour, that's a strong signal: bundle the script in `scripts/` and have the skill call it, or cut the instruction that sent the model down the detour.

## Step 4 — Improve, then repeat

This is the heart of the loop. With the user's feedback in hand:

1. **Generalize from the feedback** — fix the underlying issue, not the one example. If a problem is stubborn, try a different framing or metaphor rather than piling on constraints.
2. **Keep it lean** — delete instructions that aren't pulling their weight. Length costs attention on every invocation.
3. **Explain the why** (again) — terse user feedback usually points at a real need; understand it, then transmit that understanding into the instructions.

Apply the improvements, rerun the test prompts, get feedback again. Stop when the user is happy, the outputs are consistently good, or you've stopped making meaningful progress. Your thinking time is cheap here — draft a revision, look at it with fresh eyes, and improve it before you rerun.

## Step 5 — Sharpen the description (triggering)

The description decides whether the skill ever fires, so it's worth a dedicated pass. After the skill works, pressure-test triggering with a mix of realistic queries that **should** trigger it and near-miss queries that **should not** (queries that share keywords but actually need something else — those are the valuable negatives). Tune that single advertised sentence (remember aivo only shows the model up to the first `". "`) until the should-trigger cases fire without the near-misses leaking in. Show the user the before/after.

Note on aivo specifically: a skill can also be invoked deterministically as `/<name>`, so even a description that under-triggers is still reachable by hand — but a good description is what makes the model reach for it on its own, which is the whole point.

## When you're done

Tell the user where the skill lives and how to use it: the model will load it automatically when the description matches, or they can type `/<name>` to run it directly. They can enable/disable or remove it from the `/skills` overlay. If they want it available in another project, copy the folder into that repo's `./.agents/skills/` (or keep it under `~/.config/aivo/skills/` to have it everywhere).
