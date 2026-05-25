# Changelog

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
