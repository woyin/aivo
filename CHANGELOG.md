# Changelog

## v0.36.1

- feat(code): widen thinking window to 4 lines and use ✓ for multi-select (dc05f87)
- feat(code): speed up the typewriter reveal animation (a9f98fb)
- fix(code): un-send a still-pending message on Esc (d9c5cf9)
- feat(code): plan-approval, multi-select ask_user, and edit-review gate (b882976)
- feat(code): add ask_user tool and rework thinking display (f213ec6)

## v0.36.0

The `chat` command is now `code` — the rename signals the agent's growing role.
Copilot can drive the native agent (with `/responses` fallback), the agent engine
got a major refactor with stronger safety guards, self-verification, a grant store,
and LSP integration. Resume is now session-local so it can't override the key's
default model, and the grid renderer handles tabs and control cells cleanly so
diffs never ghost.

- feat(code): let Copilot drive the native agent, with /responses fallback (607f0d5)
- feat(code): confirm model switch with a "Now using <model>" notice (a9ecbcc)
- feat(agent): refactor agent engine with addn safety, self-verify, grant store, LSP etc. (4ed4aa2)
- refactor: rename `chat` command to `code` (fa4638e)
- fix(code): keep resume session-local so it can't reset the key's default model (8a11b6f)
- fix(chat): expand tabs and scrub control cells so diffs don't ghost the grid (719772c)

## v0.35.0

The chat agent takes center stage. `aivo chat` is now fully the native agent:
it knows its own live model and effort, can switch them mid-session
(`switch_model`/`set_effort`), and reads an embedded `aivo guide` to answer
questions about itself. A new `-e/--exec` flag runs the agent headlessly on a
single prompt, `/compact` compacts context on demand, and a `/config` master
switch turns every agent tool off for plain chat. Under auto-approve the agent
now proposes a plan and confirms before large builds, streams sub-agent
activity to the parent status line, and caps the plan checklist at five items.
On the safety side, `run_bash` gates remote side-effect commands, project
skills are treated as untrusted, and a code-review pass hardened the agent core
(retries, vision, path safety). Breaking: live-sharing is unified under a
single `share` concept — the `--live` flag and `/live` command are removed with
no alias.

- feat(chat): agent knows its live model/effort, adds switch_model/set_effort, reads embedded `aivo guide` (5ce6a33)
- feat(chat): run the agent headlessly on one prompt with `-e/--exec` (113c266)
- feat(chat): `/compact` command to compact context on demand (a077785)
- feat(chat): `/config` "Agent tools" master switch — plain chat when off (1706f92)
- feat(agent): propose a plan and confirm before large builds in interactive chat (38b567f)
- feat(chat): stream sub-agent activity to the parent status line (946b6c7)
- feat(chat): cap plan checklist at 5 and hide finished plans (e2da253)
- feat(chat): scroll transcript on mobile swipe (bare Up/Down under Termux) (c1dda2d)
- feat(agent): add opt-in `context` param to grep (grep -C across all tiers) (ef77156)
- feat(agent): gate remote side-effect commands in run_bash (1aaeff4)
- feat(agent): treat project skills as untrusted (21bb799)
- feat(keys): color aivo key name magenta, plan cell dim (56f1136)
- feat(chat): color the `!cmd` shell feature magenta throughout (34cf298)
- refactor(share)!: unify live-sharing under `share` — `--live`/`/live` → `--share`/`/share`, `aivo share` always live (5f5b5fb)
- refactor(chat): remove top-level `--agent` flag and `/agent` command (855e6e1)
- refactor(cli): shorten per-command `--help` output (a3275bb)
- refactor(chat): remove the transcript scrollbar, reclaim its column (fd3ed9d)
- refactor(agent): trim engine.rs comments and system prompt (7d0c140)
- fix(agent): harden agent core per code review (safety, retries, vision) (34bfe25)
- fix(chat): clear stale plan card and stuck retry notice (b5429fb)
- fix(chat): calibrate token estimate so compaction beats the hard input limit (8e00475)
- fix(agent): keep prior context when compacting a resumed single-long-turn (aad54e4)
- fix(agent): normalize tool-name aliases throughout dispatch (shell→run_bash, apply→apply_patch, …) (d933689)
- fix(agent): don't hang the chat agent on interactive commands run headless (dea194f)
- fix(chat): refuse interactive `!cmd` up front instead of hanging on it (c67e716)
- chore: remove padding from `aivo hf` list output (ec9666b)

## v0.34.0

Account and sharing lead this release. `aivo chat` can now live-share a
session as it happens via the `--live` flag or the in-chat `/live`
command.

- feat(chat): live-share via `--live` flag and `/live` command (ef05049)
- feat(keys): show first-party key's plan in `aivo keys`/`info` (d8e650c)
- feat(account): show estimated cost in `aivo account usage` (77d2ddb)
- feat(account): prefer server plan_label over raw slug in plan displays (f25dac6)
- feat(update): nudge when a newer version is available (a6338cd)
- feat(login): send device OS + arch in device-auth request (55c333f)
- perf(chat): make starter model validation non-blocking on launch (533ba91)
- fix(chat): remember chat as the last tool picked in `aivo run` (ed63ec8)
- fix(agent): expand ~ to $HOME in file-tool path resolver (ed777d0)
- docs(help): trim ping shortcut, refresh <tool> list (f1b4363)

## v0.33.2

More chat-agent hardening. The agent now repairs malformed tool calls (with
prefix-drift telemetry) and detects leaked tool-call markup more tightly,
keeping it out of the scrollback. Edit cards gain word-level diff highlighting
and line numbers, the composer wraps input at word boundaries, and `!cmd`
output collapses carriage-return overwrites. Rounding it out: Gemini
`web_search` config and thinking-budget fixes, aligned adaptive-thinking gating
across the bridge and native paths, id-less share tool-call pairing, and an
`anyhow` bump for RUSTSEC-2026-0190.

- feat(agent): repair malformed tool calls + prefix-drift telemetry (e8b6120)
- feat(chat): word-level diff highlighting + line numbers in edit cards (8420e55)
- fix(agent): tighten leaked-call detection + drop leaked markup from scrollback (1acb555)
- fix(agent): Gemini web_search config and thinking budget (94ffabf)
- fix(thinking): align adaptive-thinking gating across bridge and native paths (d8664b4)
- fix(chat): wrap composer input at word boundaries (1cb30de)
- fix(chat): collapse carriage-return overwrites in !cmd output (c089272)
- fix(share): pair id-less chat tool calls with their results (0d6f933)
- chore: bump anyhow to 1.0.103 (RUSTSEC-2026-0190) (9115e88)

## v0.33.1

A trio of fixes for the chat agent and Pi. No-thinking turns now read their
`reasoning_effort` from the model catalog instead of a hardcoded default, the
sandbox re-run notice clears on the next agent output, and Pi re-fetches
`/v1/models` when the starter catalog was cached without a context window.

- fix(chat): pick no-thinking reasoning_effort from the model catalog (31a9d93)
- fix(chat): clear sandbox re-run notice on next agent output (68f82b1)
- fix(pi): re-fetch /v1/models when starter catalog cached without window (bec5876)

## v0.33.0

The chat agent gets a batch of capability and safety upgrades. A hosted
`web_search` tool now runs through the aivo gateway (toggle it in `/config`), a
new `/plan` command kicks off deep planning before you build, and gpt-5/codex
models route their edits through the V4A `apply_patch` editor. Under
auto-approve, the agent now hard-confirms catastrophic shell commands, verifies
its own writes, and surfaces context-window drift. Rounding it out: Cursor
effort tiers, lazier skill loading, and a handful of chat/stats/pi fixes.

- feat(agent): hosted web_search via aivo gateway + /config toggle (4ea693c)
- feat(chat): add /plan deep-planning command (5488c8d)
- feat(agent): route gpt-5/codex models to V4A apply_patch editor (22980d4)
- feat(agent): verify-after-write sessions + surface context-window drift (782f6b5)
- feat(agent): hard-confirm catastrophic shell commands under auto-approve (babcd2e)
- feat(chat): show Cursor effort tier + resolve its context window (2d310bf)
- perf(agent): load skill bodies lazily, not at discovery (7f953b7)
- fix(pi): inject real context window by caching catalog metadata (1b3c158)
- fix(agent): notify on empty-response convergence instead of silent "Done" (ed81178)
- fix(stats): count Gemini sessions from new .jsonl chat logs (3c07bb0)
- fix(chat): run /skills add install off the event loop so the TUI doesn't freeze (514aaec)
- fix(chat): send gpt-5 no-thinking as reasoning_effort=minimal, not none (c5d6b17)
- fix(account): show tokens only for models (abbf3ba)
- chore: remove dead code and unused hmac dependency (30af2a6)

## v0.32.2

New `aivo account` command surfaces your linked-device info and usage with a
shared bar meter, and folds in `login`/`logout`/`open`. The `aivo login` flow
gets two reliability fixes: the verification URL now shows on screen with the
code pre-filled, and Ctrl+C during the device poll restores terminal echo.

- feat(account): add `aivo account` (info/usage/open/login/logout) + shared bar meter (9df19f5)
- fix(login): show the code-prefilled verification URL on screen (43d0f20)
- fix(login): catch Ctrl+C during device poll to restore terminal echo (bbb0b91)
- fix(login): scope echo guard to avoid Windows-only drop_non_drop clippy lint (03059f7)

## v0.32.1

Polish for the new `aivo login` flow: the device-link screen is redesigned, the
code is checked the moment you submit it, and pressing Enter is now optional. The
chat exit hint loses a stray blank line above it.

- fix(login): redesign device-link screen; stop Enter echo duplicating spinner (904f4a1)
- fix(login): poll immediately, make Enter optional (821d831)
- fix(chat): drop blank line above the exit resume hint (d9eb5e5)
- test(agent): make /rewind probe-timeout test deterministic (3b8e14c)

## v0.32.0

`aivo login`/`logout` link this device to your account, and sharing now requires
a linked device. The chat TUI gets a warm-night brand recolor, a reworked live
status line with a token counter, and a more legible agent timeline.

- feat(login): add `aivo login`/`logout` (60e3525)
- feat(share): gate sharing on a linked device (569ff58)
- feat(logs): list resumable chat sessions with no logged turn (4c5c274)
- feat(chat): recolor TUI to warm-night brand palette (7c46bac)
- feat(chat): rework live status line + token counter (ba680a4)
- feat(chat): improve agent timeline legibility (15943b6)
- fix(chat): don't restore cancelled message into composer on ESC (a13bad4)

## v0.31.5

`/rewind` no longer stalls when chat is launched from your home directory, and
the Gemini bridge answers the unsupported Interactions API with a clean 501
instead of a confusing failure.

- feat(gemini): return 501 UNIMPLEMENTED for Interactions API paths (02147ce)
- fix(chat): stop `/rewind` checkpoint from walking the whole tree in `~` (52f3dc9)

## v0.31.4

The thinking display is redesigned around a folded summary you click to expand,
and chat gains named agent profiles with a `/agent` picker plus an `/effort`
control for reasoning depth.

- feat(chat): redesign thinking display — folded summary, click to expand, true disable (eabcd12)
- feat(chat): polish the thinking block — past-tense title, hanging indent, dim gutter (065e96b)
- feat(chat): scope agent profiles to `~/.config/aivo/agents` + add `/agent` picker (7ad732f)
- feat(chat): add `/effort` reasoning-effort control (8edf8b6)
- feat(chat): fold `!cmd` output inline, click to expand, drop ctrl+o modal (44afd09)
- feat(chat): add a clickable jump-to-bottom pill to the transcript (e4a8a01)
- feat(chat): handle image inputs gracefully on non-vision models (18948b9)
- feat(chat): remember the agent path's negotiated protocol across turns (6b7c0f3)
- feat(chat): re-harvest stale catalog on launch so server-side edits propagate (881fc4e)
- feat(chat): accept `--1m`/`--2m` shorthands for `--max-context` (781cc64)
- feat(chat): extend `--max-context` to run tools and plugins (be409a0)
- feat(agent): tighten the action-bias system prompt wording (a413010)
- feat(data): add Sakana AI and 14 more providers (942ea8c)
- fix(chat): persist `/skills` + `/mcp` toggles in chat-prefs.json, not config.json (6c420be)
- fix(chat): keep `/rewind` file revert across resend merges and compaction (5abbc82)
- fix(chat): don't retry structured 400s on the Responses API (171a385)
- fix(chat): compact unknown-window models and stop dropping transcript (1296f01)
- perf(chat): repaint promptly on input and cache max-scroll for hot scrolling (20cbfb7)
- docs(run): describe the tool picker accurately in help (c0e73b4)
- ci(canary): run every 3 days instead of daily (f391ffd)

## v0.31.3

`/rewind` is now reliable and surgical: it reverts only the files the agent
changed and never restores the wrong tree.

- feat(chat): render model thinking, add `/config` + Ctrl+T toggle, request reasoning (564977e)
- feat(chat): extend drag-to-copy to the whole screen, not just the transcript (0623ff7)
- feat(debug): log `phase=cancelled` when a request is dropped before send() resolves (bd8d6af)
- fix(chat): make `/rewind` reliable and surgical (6545a48)
- fix(chat): keep `/goal` machine text out of input-history recall (d65d31b)
- fix(chat): default mouse capture off under Termux so taps keep toggling the keyboard (1feeeab)
- fix(models): treat per-million `/v1/models` pricing as-is; add publicai provider (5a60c52)
- chore(chat): improve model listing layout (00a617c)

## v0.31.2

Model data now refreshes without a release: `aivo update --sync-model-data`
regenerates the embedded model-limits snapshot from live models.dev.

- feat(update): add `--sync-model-data` to refresh model data from models.dev (4cd7051)
- feat(models): resolve model names against the provider catalog (a6bf370)
- feat(chat): scope input-history recall by launch directory (9df2a4e)
- feat(chat): record `/commands` in input history and dedupe consecutive entries (cb60860)
- feat(chat): cap input history at 100 and title the composer rule with position (614fc56)
- feat(stats): span the scan progress counter across native tools and plugins (0407f8f)
- fix(models): let an override win per folded id when overlaying refreshed limits (9ded5fd)
- fix(update): self-heal the stale npm launcher shim on Windows (d7f0424)
- fix(chat): run `!cmd` through plain pipes on Windows, not ConPTY (80555d9)
- chore(chat): simplify `chat --help` output (6de5134)
- chore(chat): remove the over-broad-workspace launch warning (6867500)

## v0.31.1

- feat(gemini): remove Google OAuth sign-in for the gemini CLI (8d3b793)
- fix(chat): unblock `!cmd` on Windows ConPTY by closing the PTY to end the read (3e30ffe)
- fix(update): self-update npm installs natively instead of via npm (755be07)

## v0.31.0

`aivo chat` is now a native, in-process coding agent. It reads and edits
files, runs commands behind permission prompts, and drives a full chat TUI
directly inside aivo — sharing the same engine, tools, and protocol as the
rest of the CLI instead of shelling out to an external agent.

- feat(chat): native in-process coding agent (74c1b27)
- fix(chat): cross-platform shell + clipboard for the chat agent (718812f)
- fix(pi): symlink `~/.pi/agent/npm` into the temp `PI_CODING_AGENT_DIR` (30e6600)

## v0.30.1

Local llama-server runs (`hf:` refs and local `.gguf` files) now configure
themselves from the model: the real context length is read from the GGUF
header, an `mmproj-*.gguf` sidecar is auto-detected, and `AIVO_LLAMA_ARGS`
is appended last for overrides. The bare `aivo run` picker also gained an
aligned description per tool and hides the macOS-only `codex-app` on other
platforms.

- feat(hf): auto-configure llama-server from the model (6e9f3ed)
- feat(run): describe tools in the picker and hide codex-app off macOS (82f8c9b)
- fix(update): fail before download when the install dir is unwritable (3d65c7c)
- fix(clippy): keep codex-app platform check matches!-free (510659d)

## v0.30.0

OS-keyring custody is now the default: API keys are encrypted under a master
secret held in the OS keychain/keyring (macOS Keychain, Linux Secret Service,
Windows Credential Manager). `AIVO_KEYCHAIN=0` opts out; machines without a
usable keyring keep the previous encryption automatically.

**Cross-machine copies**: once a store is keyring-backed, copying
`~/.config/aivo/config.json` to another machine no longer carries decryptable
keys — the master secret stays in the source machine's keychain. Use
`aivo keys export <file>` / `aivo keys import <file>` to move keys between
machines. The aivo-amp plugin must be ≥ v0.1.7 to read keyring-backed stores.

- feat(keys): enable OS-keyring custody by default (3f17626)
- feat(keys): migrate the whole store to current encryption on add/import (9681fa6)
- feat(ci): nightly canary smoke-testing latest agent CLIs against starter (9553f22)
- feat(ci): add pi to the nightly canary matrix (cd74fd9)
- feat(ci): add tool-call round-trip layer to canary (deb720a)
- feat(ci): canary protocol-transform layer via local OpenAI fake (23daf3d)
- feat(ci): cross-protocol tool round-trip in canary transform layer (199b026)
- fix(keys): harden keyring custody after a master-secret lockout (cca6a22)
- fix(serve): remove ordering-dependent unwraps from streaming converters (cc6ef52)
- chore(run): mark codex-app as experimental (e44637e)
- chore(data): re-sync model limits from models.dev (acb80f6)

## v0.29.0

Security hardening release.

- feat(keys): OS-keyring-backed v5 encryption, opt-in via AIVO_KEYCHAIN=1 (2d30fff)
- feat(router): bearer-gate native loopback routers with a per-launch token (c3c3700)
- feat(plugins): re-consent when an update changes a remote plugin's binary (e264cfa)
- feat(update): verify self-update downloads with an embedded minisign key (3ac11c9)
- fix: harden self-update verification and Windows keyring edges (f36ec10)
- fix(launch): pin claude env via a 0600 settings file (c0c9de1)
- fix(plugins): bind consent to binary identity, fail closed off-TTY (88719e1)
- fix(keys): create the keyring master secret only under the config lock (f698f22)
- chore(data): re-sync model limits from models.dev (513adc9)

## v0.28.1

- feat(plugins): aivo owns -k/-m for all endpoint-granted plugins (fec42af)
- feat(plugins): closed plugin type vocabulary (coding-agent | tool | media) (1b3f96b)
- feat(plugins): advisory AIVO_KEY_MODEL hint for endpoint plugins (c66a31f)

## v0.28.0

- feat(cli): consistent list/ls and rm/remove verbs across commands (2dcd0c9)
- feat(cli): bare lowercase words are subcommands, never prompts (66033bf)
- feat(keys): clear the cached model list on reset-route (ed3f191)
- feat(plugins): offer install-on-demand when a known plugin isn't installed (90480d5)
- feat(plugins): install Node plugins from GitHub via source-tarball fallback (8fa19f2)
- feat(run): plugin-aware tool picker, always shown with replay default (7fa9760)
- feat(plugins): advisory model-limit env vars in the endpoint handoff (0982e27)
- feat(serve): emit model limits in loopback /v1/models responses (dc5f114)
- feat(codex): real context and published reasoning levels in the model catalog (005dcbe)
- feat(metadata): embedded models.dev limits snapshot with canonical lookup cascade (afea486)
- feat(plugins): first-run confirmation before executing remote installs (cb8f80b)
- feat(stats): windowable per-run token stats for plugin endpoints (c6ce13c)
- fix(pi): real default model and resolved limits in generated models.json (94d0c89)
- fix(http): parse K/M display strings in token-limit fields (e2b8d4d)
- fix(bridges): drop sampling params for models that reject them (6cfd47d)
- fix(bridge): emit parallel tool results as one anthropic user message (5760d1f)
- fix(hardening): SSE buffer caps, atomic stats cache, 0700 config dir (ea98a1d)
- fix(errors): honor the documented exit-code contract at command boundaries (ee332e4)
- fix(gemini): accept nullable/union tool-param types (2d87913)
- refactor(routers): unify cascade fallback policy across all routers (ed02559)
- refactor(injector): extract routing decision policy into router_selection (7b6793d)
- test(routers): add fake-provider e2e harness for cascade behavior (7d3eda5)
- chore(data): re-sync model limits from models.dev (b63da77)
- docs(readme): supported coding agents as a table with built-in/plugin type (4eea92d)


## v0.27.1

- feat(plugins): native transcript export for `aivo share` (a46a772)
- feat(plugins): `--dry-run` preview + unified model memory for coding-agent plugins (4effa54)
- feat(plugins): serve `hf:`/local-gguf models to coding-agent plugins (f029895)
- feat(plugins): guide users to install the amp plugin instead of erroring (623cac8)
- feat(plugins): report real update outcomes with a fetch spinner (31b988a)
- docs(plugins): reconcile the protocol doc with shipped endpoint behavior (5ca103c)


## v0.27.0

- feat(plugins): typed plugins + capability-gated key/endpoint handoff (33734f6)
- feat(stats): filter by tool with --by, dropping the positional (af2b2f0)
- fix(responses): merge split assistant turn into one Chat message (446e370)
- fix(router): learn requires_reasoning_content from the streaming bail (5144df3)
- fix(pi): install maintained @earendil-works/pi-coding-agent (b407cad)
- test(share): gate Unix-only pi transcript test on cfg(unix) (873b26a)


## v0.26.0

### Features

- Plugins: external-subcommand plugins (`aivo <x>` → `aivo-<x>` sibling) with managed `install`/`update`/`remove`.
- Plugins: surface plugin-tool runs in `aivo logs` and clean up the share message.
- `aivo cursor`: honor JSON-output requests on the cursor bridge.
- `aivo hf`: authenticate to HuggingFace for gated/private GGUF repos (`HF_TOKEN` and friends).

### Fixes

- `aivo hf`: rank GGUF mirrors by relevance instead of download count.
- `aivo hf`: don't pin a failed pull to its source; add `--refresh` to re-resolve.
- `aivo pi`: create a writable temp bin dir on first-time use.
- `aivo claude`: pin aivo-overwritten env via `--settings` so `settings.json` can't shadow the injected routing/model.


## v0.25.1

- Per-`(tool, key, model)` protocol routes: each tool/model learns and persists its own wire format, seeded from tool-native priors.
- `aivo chat`: persist the learned protocol route per `(key, model)` so a warm start skips the failed-probe round-trip.
- `aivo pi`: route through aivo's transform router by default; pass `--transparent` to talk to the upstream natively.
- Merge install dirs into the child `PATH` instead of clobbering it.


## v0.25.0

### Changes

- Drop support for the Sourcegraph Amp CLI.

### Fixes

- `aivo chat` now shares the launcher's protocol fallback, so the cascade keeps
  probing other wire formats when a provider rejects one as unsupported (e.g. OpenCode Zen).


## v0.24.0

### Features

- `aivo keys`: in-place line editing and paste-safe input for the add/edit flows.
- `aivo pi`: list the full provider catalog in pi's `/model` picker.

### Fixes

- `aivo keys`: show the active key in the bare `aivo` footer when `last_selection` is empty.
- Harden token-usage math and id slicing against malformed input.

### Changes

- Redraw the chat TUI only on state change.
- Match the keys edit flow to add's stepped layout.
- Remove the image/audio/video media commands (archived on the `media` branch).


## v0.23.6

- fix(codex-app): stream the responses path to Codex instead of buffering
- fix(codex-app): stream live, fail fast on bad model, fix empty transcripts
- fix(router): hoist stray role:system messages into top-level system


## v0.23.5

- `aivo codex-app`: add partial support for launching Codex App, macOS only.

## v0.23.4

- `aivo stats`: hide unsupported tools from the by-tool table; drop the `--top-sessions` flag.
- `aivo audio`: reject `hf:` refs with a friendly error.
- `aivo hf`: reject encoder-only GGUFs before spawning `llama-server`.
- Provider bridge: recognize DeepSeek's `prompt_cache_hit_tokens` in usage accounting.

## v0.23.3

- `aivo hf`: default to Metal GPU offload on macOS; keep tool routing alive when the jinja template fails.
- `aivo cursor`: surface in-flight status in the codex/claude TUIs via the cursor bridge.
- `aivo logs share`: match codex sessions by dashes-stripped UUID prefix.

## v0.23.2

- CLI: reject short opaque tokens instead of treating them as prompts.
- `aivo starter`: sign requests with a per-device HMAC.
- Termux: fix device ID detection.

## v0.23.1

- Fix logs and imporove cursor support.

## v0.23.0

## Features

- New provider: **Cursor** with invisible ACP integration. Run `aivo cursor` to drive cursor-agent as a normal chat/code session; auth and config are sandboxed per key.
- New provider: **Command Code gateway** (Cohere).
- `aivo hf`: auto-install `llama-server` from upstream release on Linux/Windows (previously macOS-only).
- `aivo pi --transform`: force aivo's router for Pi launches.
- `aivo chat`: suppress thinking output without breaking non-reasoning models; sandbox cursor cwd and reject tool execution.

## Fixes

- Windows: fix build regression.
- `aivo logs`: prevent UUIDv7 prefix collisions; show picker on ambiguous `aivo logs show <prefix>`; ingest gemini-cli's new jsonl session format.
- `aivo keys export`: show help instead of error when called with no path.
- CLI: polish command UX (TTY spinners, help drill-down, errors).

## Internal

- CI: add cargo-deny supply-chain gate, pin release toolchain to 1.95.0, add `[lints]` table, enforce fmt.
- Patch `rustls-webpki` and `rand` for RUSTSEC advisories.
- Convert JSDoc-style doc comments to rustdoc syntax.

---

## v0.22.1

## Breaking

- Removed `aivo logs status` subcommand.

## Improvements

- `aivo hf`: accept local `.gguf` paths anywhere an `hf:` ref is accepted.
- `aivo hf`: resume interrupted GGUF downloads via HTTP Range requests.
- `aivo update` (npm path): show download progress and the resolved version on Windows.
- `aivo hf`: surface `llama-server` stderr on launch failure and auto-retry on Jinja chat-template errors.
- Windows: build all home-relative paths (HF cache, config dirs) with platform separators.

---

## v0.22.0

## Features

- Run HuggingFace GGUFs locally: `aivo claude|chat|serve hf:owner/repo`, plus `aivo hf <list|pull|rm|clean>` for the cache.
- Bare prompt shortcut: `aivo "hello"` runs `aivo chat -p "hello"`. One-shot flag renamed to `-p`/`--prompt`.
- `aivo claude`: routed model now shows up in Claude Code's `/model` picker.
- `aivo keys export`: filters out OAuth and Copilot entries by default.

---

## v0.21.6

## Features

- `aivo keys export` / `aivo keys import`: portable password-encrypted backups of stored keys.

## Fixes

- `aivo claude`: honor `-m`/`--1m` for OAuth keys, properly unset conflicting `ANTHROPIC_*` auth vars, and set the beta-header policy per connection mode.

---

## v0.21.5

## Fixes

- Termux: plug a hickory-backed DNS resolver into all outbound HTTP clients. aivo's release binary is static-musl and reads `/etc/resolv.conf` literally — which Termux doesn't populate — so every lookup was failing with `EAI_AGAIN`. The resolver reads `$PREFIX/etc/resolv.conf` if present and falls back to Cloudflare + Google.

---

## v0.21.4

## Features

- `aivo logs show`: open the picker when called without an id (mirrors `aivo logs share`).

## Fixes

- HTTP: honor system proxies (`HTTPS_PROXY` / `HTTP_PROXY` / `NO_PROXY`) for all outbound clients.
- HTTP: drop the Termux IPv4-only default (let happy-eyeballs pick v6 where it's faster); keep `AIVO_HTTP_IPV4_ONLY=1` as opt-in. Surface the underlying error chain on `aivo -x` and `aivo update`.
- TUI: keep the `FuzzySelect` redraw clean when rows contain CJK / wide characters (share, logs, run pickers).

---

## v0.21.3

## Features

- `aivo serve`: resolve model aliases in incoming requests; aliases also listed in `/v1/models`.

## Fixes

- Termux: unbreak `aivo run` / `aivo chat`.
- HTTP: extend Termux IPv4-bind to all outbound clients.
- Router: unbreak strict OpenAI-compat providers (e.g. Cloudflare Workers AI).

## Improvements

- `aivo logs` / `aivo share list`: refactor shared listing logic.
- CLI: rename hidden `speak` (TTS) subcommand to `audio`.

---

## v0.21.2

## Features

- `aivo-starter`: detect removed models before launch and re-prompt for a replacement.

## Improvements

- `aivo share`: replace the fullscreen TUI picker with an inline picker.

## Fixes

- Tests: gate Unix-shaped `pi` cwd encoding tests on `cfg(unix)`.

## Polish

- Services: route OpenRouter base-URL checks through a shared `is_openrouter_base` helper.

---

## v0.21.1

## Fixes

- `aivo share`: resolve native-CLI runs whose turns are all short.
- `aivo share`: resolve claude runs whose turns are all short.

## Improvements

- `aivo share`: stream `/state` via chunked ND-JSON for end-to-end token-level live mode.

---

## v0.21.0

## Features

- `aivo share`: new command to share an aivo session via a tunneled web viewer.

## Fixes
- Windows: terminate the ollama server on shutdown and harden secure file writes.
- Help: hide `info`, `image`, `video`, and `speak` from the top-level help to reduce noise.

---

## v0.20.2

## Fixes

- Router: normalize Claude Code 2.x thinking config per model.
- Router: forward the originating tool client's `User-Agent` to upstream providers instead of reqwest's default.
- `aivo pi`: preserve user customization (packages, MCP servers, rules, themes, settings) by symlinking the real `~/.pi/agent/` state instead of overwriting it with a fresh temp dir.
- Gemini: preserve `reasoning_content` across deepseek-thinking turns.
- `aivo models`: respect the model cache instead of refetching on every invocation.
- `aivo models`: always show the spinner during fetch.
- `aivo models`: surface a friendly message ("No models found...", "Invalid API key...") on fetch failure instead of dumping raw HTML.

## Performance

- `aivo models`: faster model display, with table output batched into a single stdout write.

## Polish

- TUI: tidy disabled rows in the fuzzy picker.
- Providers: drop the MiniMax hardcoded model list (now served via `/v1/models`) and the `anthropic_path_prefix` quirk.


---

## v0.20.1

## Features

- `aivo amp`: `--mode <smart|rush|large|deep>` flag, with the model picker now honoring per-mode model overrides.

## Fixes

- `aivo amp`: per-mode model flags take precedence over `-m`.
- `aivo amp`: honor per-mode models without invoking the picker.


---

## v0.20.0

## Features

- `aivo amp`: Amp (Sourcegraph) is now a first-class supported tool, alongside `claude`, `codex`, `gemini`, `opencode`, and `pi`. Per-mode model overrides, `--max-context=1m`, and an on-the-fly bridge that routes the LLM plane to your configured upstream while stubbing management traffic locally for privacy. `aivo amp trust` gates workspace MCP servers so a hostile checkout can't auto-launch them.

## Fixes

- Router: pass through the user-agent header from tool clients.


---

## v0.19.20

## Features

- `aivo image`: inline preview in supported terminals.
- `aivo run` (codex): `--max-context` now drives Codex via `model_context_window`.
- Keys: editing a Bedrock entry reuses the add flow's region picker.

## Fixes

- Router: bail on 429 instead of probing fallback paths.
- CLI: `--<N>m` accepted as a generic max-context shorthand.
- Refactor: split `main.rs` into `cli_args` + `run` modules.


---

## v0.19.19

## Fixes

- Audio: preserve external playback on `--no-default-features` builds.


---

## v0.19.18

## Improvements

- `aivo speak` replaces `aivo audio`: cached TTS, file/stdin input, streaming playback, `--list` picker.
- `aivo run --relogin`: refresh expired Codex/Gemini/Claude OAuth keys in place.
- Keys: Amazon Bedrock added as a known provider.
- Codex/Gemini OAuth shadows: expose skills, plugins, rules, and the real home state; auto-relogin when refresh tokens are invalidated.
- Claude: scoped the non-essential-traffic disable env.
- TUI: fuzzy picker handles multiline paste and trims outer whitespace from filter queries.


---

## v0.19.17

## Fixes

- xAI / OpenAI-compatible providers: usage stats now flow through correctly. Some providers (xAI/Grok especially) emit `input_tokens` / `output_tokens` instead of `prompt_tokens` / `completion_tokens`; the router and bridges now accept both aliases and preserve them end-to-end through the Anthropic bridge so Claude Code sees real token counts instead of zeros.

## Internal

- Tests: `claude` setup-token spawn retries on transient `Other` errors to de-flake CI.


---

## v0.19.16

## Features

- `aivo run`: auto-detect 1m / 2m context windows so users don't have to pass the flag.
- Claude gateway: `/v1/models` endpoint backs the model picker and accepts `x-api-key` auth alongside bearer tokens.
- Stats: per-model cache breakdown with a new `-d` / `--detailed` view (input / output / cached / total), and `--since` now counts one-shot `aivo -x` chat turns with per-session token totals.

## Fixes

- Stats `--since`: window is now actually windowed. Suppresses lifetime aivo-proxy and per-key counters from leaking into the cutoff, applies the cutoff to event timestamps, surfaces every model the user launched in-window (even when the upstream withheld usage), and records chat tokens under the upstream model name to match `claude-code`.
- Pi: reuse session history when relaunching under `aivo pi`.


---

## v0.19.15

## Features

- New media-generation subcommands `aivo video`, `aivo audio`, and `aivo speak` join the existing `aivo image`, sharing a `services::media_io` module for output path parsing, overwrite policy, atomic writes, and error extraction. Hidden from `--help` for now while the surface stabilizes.

## Internal

- CI: scope concurrency cancellation to `ci.yml` and run the test workflow on `main` pushes / PRs so the matrix validates before tagging.


---

## v0.19.14

## Fixes

- Launcher: skip extension-less PATH entries when looking up tool binaries on Windows — npm drops both `claude.cmd` and a bash-style `claude` (no extension) into `%APPDATA%\npm`; the lookup matched the unspawnable bash shim first. `aivo claude` / `aivo codex` / `aivo gemini` now resolve to the `.cmd` shim and spawn correctly even when the tool was already installed

(v0.19.12 and v0.19.13 carried the same fix but failed CI — v0.19.12 on a `PATHEXT` casing mismatch in tests, v0.19.13 on a `clippy::needless_return` lint that only fires on the newer Rust shipped to `windows-latest`. No binaries were published for either.)


---

## v0.19.11

## Fixes

- Launcher: pin the resolved binary path before spawning so npm `.cmd` shims (`claude.cmd`, `codex.cmd`, etc.) launch on Windows — `CreateProcessW` does not honor PATHEXT for non-`.exe` files, so spawning the bare `claude`/`codex` name failed after install


---

## v0.19.10

## Fixes

- Chat TUI / `keys add` secret prompt: filter Windows key Release events so typed characters aren't doubled — crossterm emits both Press and Release on Windows for every keystroke; we now only process Press


---

## v0.19.9

## Fixes

- Build: statically link the MSVC C runtime for Windows targets so `aivo.exe` no longer depends on `VCRUNTIME140.dll` — fixes silent load failure (`STATUS_DLL_NOT_FOUND`) on Windows ARM64 systems without the Visual C++ Redistributable installed, where `aivo --version` printed nothing


---

## v0.19.8

## Improvements

- Bridge: propagate cached input tokens from OpenAI-shape upstreams — Claude Code now records `cache_read_input_tokens` correctly so cached usage shows up in `aivo stats` instead of being silently dropped
- Bridge: coalesce parallel function_calls into one assistant message
- Run: skip model picker on first run when only the starter key exists
- Launcher: fall back to installer drop dirs when PATH lookup misses
- MCP/router: pin nickname session resolution; trust proven route on attempt 0
- Removed cross-tool MCP peer awareness
- Build: add `win32-arm64` target


---

## v0.19.7

## Improvements

- `aivo update` and `install.sh` now fetch binaries from `getaivo.dev` (R2) instead of GitHub Releases — faster in regions with poor GitHub connectivity
- Fixed text selection and session list in chat TUI
- Bridge: filter Anthropic server-side tools (`web_search_*`, `code_execution_*`, etc.)
- Bridge: propagate `input_tokens` via `message_delta` on OpenAI streams — fixes Claude Code status-line percent stuck at 0%
- Router: apply learned `requires_reasoning_content` to in-flight requests, eliminating per-request 400 + retry until relaunch
- MCP: stabilize session discovery and nickname mapping
- Removed Chat TUI thinking toggle and think-tag detection


---

## v0.19.6

## Fixes

- Fix handling `reasoning_content` in deepseek learn the config
- Reorder `--help` command listing (keys, models, chat, serve, image, stats)


---

## v0.19.5

## Improvements

- Extended `alias` support to launch tool, not only model
- Added Poolside provider preset
- Improved `--max-context` flag parsing to handle more formats and edge cases
- Force non-streamed upstream for Inception Mercury when tools are in use
- Fixed `alias rm <name>` parsing


---

## v0.19.4

## Improvements

- Fallback: persistent learned protocol/path routing with smarter Responses-API detection
- Bridge: forward image/file parts and dropped sampling params; safer `tool_use` IDs, stop-reason mapping, and `cache_control` handling
- Gemini: persist `thoughtSignature` across restarts; group parallel tool responses into one user turn
- Claude: `--1m`/`--2m` flags (and `--max-context=1m|2m`) append the canonical `[Nm]` suffix to set max context window
- `aivo keys reset-route <name>` to clear learned routing; `--debug` rewritten as JSONL HTTP logger


---

## v0.19.3

## Improvements

- `aivo stats --since DURATION`: time-windowed reports (e.g. `--since 7d`, `--since 24h`)
- Claude Code: per-task model overrides for individual slots
- CLI: drop custom short-flag pre-expander, rely on clap POSIX bundling
- Gemini: handle `thoughtSignature` for gemini-3


---

## v0.19.2

## Features

- `aivo image`: experimental image generation command

## Fixes

- Router: try native `/v1/messages` when `target_protocol` is anthropic; respect learned `PathVariant` on fast paths
- Router: prefer CLI-native protocol and force aivo-starter through the router
- Fix Linux ETXTBSY flake in Claude `setup-token` spawn tests
- `aivo chat --json` now prints the provider's raw response body instead of aivo's envelope
- `aivo run`: skip the model picker on non-TTY and under `--dry-run`
- `aivo chat` no longer injects `max_tokens: 8192` for DeepSeek / aivo-starter; matches `curl`
- `opencode`: default to an OpenAI-style model instead of `claude-sonnet`

---

## v0.19.1

## UX

- Improve key pickers and status prints for `--as`.

## Bug Fixes

- Preserve Codex trust settings across launches
- Reject for disable OAuth keys for `models` and `serve` instead of error

---

## v0.19.0

## Features

- Multi-account OAuth for Codex, Gemini and Claude Code

## Bug Fixes

- Align `ctx`, `out`, and price columns in `aivo models`

## Refactors

- Drop reserved-name shortcuts in `aivo keys`; preselect in the picker instead
- Rework `Ctrl+C` / `Ctrl+L` handling and drop the `/clear` command

---

## v0.18.1

- Fix build on windows.

---

## v0.18.0

## Features

- **Cross-tool MCP communication via `--as <name>`**: Run a tool under a custom identity so peers can query it live
- **`aivo context` and `--context` injection**: Export recent session context as Markdown and inject it into any launched CLI for cross-tool handoffs
- **Copilot premium-request multiplier**: Surfaced in `aivo models` so you can see which Copilot models cost more
- **Chat TUI keybinding swap**: `Ctrl+C` and `Ctrl+L` swapped to match other agents

## Bug Fixes

- Bypass HTTP proxy for the loopback router when launching CLIs
- Cancel in-flight chat requests when the user exits
- Close tool-calling parity gaps in the serve bridge layer
- Plug correctness gaps across crypto, bridges, and TUI
- Tighten bridge-layer fragility across serve, launch, and `cache_control`
- Make Windows first-class in PID liveness checks and Pi binary reuse
- Preserve `tool_result` images through Anthropic-to-OpenAI conversion and the typed OpenAI round-trip
- Extend multimodal preservation through the Responses API and recover sniffed `media_type`
- Close post-review gaps across serve auth, `tool_result` images, and the OpenAI round-trip
- Surface unreadable session files in debug builds during `--context` ingestion
- Align `aivo stats` Claude totals with Claude Code's `/stats` UI
- Prevent update download from timing out on slow connections

---

## v0.17.0

## Features

- **Improved the ux of adding keys**: Interactive provider picker backed by the full known-provider catalog
- **Added JSON output via `--json`**: Enables scripting and `| jq` pipelines

## Refactors

- Streamline `logs` command and redesign the status output

---

## v0.16.2

## Bug Fixes

- Normalize `input_tokens` to fresh-only across all stats parsers for consistent accounting
- Exclude subagent sidechain files from Claude session count
- Add Claude Code env vars for timeouts, attribution, and subagent model
- Handle Ollama pull errors instead of silently succeeding
- Refresh PATH from login shell after tool install
- Move `--version` and `--help` before service init and improve smoke test diagnostics

---

## v0.16.1

## Bug Fixes

- Fix `ETXTBSY` on Linux during self-update: close write file descriptor before smoke test

---

## v0.16.0

## Features

- **Google Gemini native API**: Direct support for `generativelanguage.googleapis.com` as a provider across all tools
- **Open aivo-starter model list**: aivo-starter users can now access the full model catalog
- **Enriched `aivo models`**: Show context window, max output tokens, and pricing from the provider API
- **Prompt to install missing tools**: Interactively offer to install tool binaries when not found on PATH, with cross-platform support
- **R2 download mirror**: GitHub-first binary downloads with Cloudflare R2 fallback for faster installs and updates

## Bug Fixes

- Clear `last_selection` on key add so the newly added key is actually used
- Resolve aivo-starter sentinel URL in serve router and model picker
- Normalize Pi stats token counting to include cached input tokens
- Reduce GitHub download timeout and deduplicate mirror fallback messages
- Fix mirror fallback for self-update flow

---

## v0.15.0

## Features

- **aivo-starter**: Zero-config provider — start using aivo without any API key setup
- **Update rollback**: Automatically roll back failed updates; added config migration tests and CI clippy gate
- **Local session logging**: SQLite-backed `aivo logs` command for browsing session history
- **Native top session view**: Opt-in `aivo stats --top` for a live session overview
- **Combined short flags**: Support Unix-style combined flags like `-xr`, `-nar`
- **Ollama lifecycle management**: Auto-stop Ollama on exit using PID-file refcount for safe concurrent instances
- **DeepSeek reasoning streaming**: Stream `reasoning_content` through routers for DeepSeek-reasoner models
- **Conditional default model option**: Only show "default model" in the picker when the selected tool supports it

## Bug Fixes

- Cap `max_tokens` for aivo-starter and DeepSeek in chat requests
- Remove last two production `unwrap()` calls for safer error handling
- Fix device auth for starter provider across all tools
- Hide default model option in chat mode since it requires a concrete model
- Support Responses API-only Copilot models (e.g. gpt-5.4) for Claude and Gemini
- Resolve tilde paths and add PDF/binary support for chat attachments
- Remove tool name from active key display, show only key and model

## Performance

- Avoid PBKDF2 decryption when displaying active selection label
- Warm model cache in background after adding API key

## Refactors

- Redesign key/model selection: per-directory → global last-selection
- Replace `sqlite3` CLI with `rusqlite` for OpenCode stats reading
- Route OpenCode through router for providers with quirks

---

## v0.14.5

Major update with stats aggregation, better tool support

## Improvements

- Global stats aggregation across all AI tools (Claude, Codex,
  Gemini, OpenCode, Pi).
- Mask API key input with asterisks during entry
- Show install guide when a tool is not found on PATH
- Support Pi tool with Copilot subscription
- Rename `ls` command to `info` (keep `ls` as alias)
- Embed provider registry as JSON with table-driven tests
- Remove redundant token stats recording from run tool
- Bump GitHub Actions to v5 for Node.js 24 compatibility

## Fixes

- Remove custom User-Agent headers from API requests
- Use Codex `model_provider` config to bypass `auth.json` and
  `OPENAI_BASE_URL` deprecation
- Wire `--refresh` flag through run command for model picker
  cache bypass
- Auto-strip `anthropic-beta` headers for Bedrock/Vertex providers


---

## v0.14.4

Stability hardening. Fixed panics from char-boundary slicing
and API response handling. Switched Linux builds to musl
targets for better portability.

---

## v0.14.3

Added Responses API fallback for models that require the
/v1/responses endpoint. Fixed /attach command autocomplete.

---

## v0.14.2

Bug fixes and CI improvements.
