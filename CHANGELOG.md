# Changelog

## v0.19.5

### Improvements

- Extended `alias` support to launch tool, not only model
- Added Poolside provider preset
- Improved `--max-context` flag parsing to handle more formats and edge cases
- Force non-streamed upstream for Inception Mercury when tools are in use
- Fixed `alias rm <name>` parsing


## v0.19.4

### Improvements

- Fallback: persistent learned protocol/path routing with smarter Responses-API detection
- Bridge: forward image/file parts and dropped sampling params; safer `tool_use` IDs, stop-reason mapping, and `cache_control` handling
- Gemini: persist `thoughtSignature` across restarts; group parallel tool responses into one user turn
- Claude: `--1m`/`--2m` flags (and `--max-context=1m|2m`) append the canonical `[Nm]` suffix to set max context window
- `aivo keys reset-route <name>` to clear learned routing; `--debug` rewritten as JSONL HTTP logger


## v0.19.3

### Improvements

- `aivo stats --since DURATION`: time-windowed reports (e.g. `--since 7d`, `--since 24h`)
- Claude Code: per-task model overrides for individual slots
- CLI: drop custom short-flag pre-expander, rely on clap POSIX bundling
- Gemini: handle `thoughtSignature` for gemini-3


## v0.19.2

### Features

- `aivo image`: experimental image generation command

### Fixes

- Router: try native `/v1/messages` when `target_protocol` is anthropic; respect learned `PathVariant` on fast paths
- Router: prefer CLI-native protocol and force aivo-starter through the router
- Fix Linux ETXTBSY flake in Claude `setup-token` spawn tests
- `aivo chat --json` now prints the provider's raw response body instead of aivo's envelope
- `aivo run`: skip the model picker on non-TTY and under `--dry-run`
- `aivo chat` no longer injects `max_tokens: 8192` for DeepSeek / aivo-starter; matches `curl`
- `opencode`: default to an OpenAI-style model instead of `claude-sonnet`

## v0.19.1

### UX

- Improve key pickers and status prints for `--as`.

### Bug Fixes

- Preserve Codex trust settings across launches
- Reject for disable OAuth keys for `models` and `serve` instead of error

## v0.19.0

### Features

- Multi-account OAuth for Codex, Gemini and Claude Code

### Bug Fixes

- Align `ctx`, `out`, and price columns in `aivo models`

### Refactors

- Drop reserved-name shortcuts in `aivo keys`; preselect in the picker instead
- Rework `Ctrl+C` / `Ctrl+L` handling and drop the `/clear` command

## v0.18.1

- Fix build on windows.

## v0.18.0

### Features

- **Cross-tool MCP communication via `--as <name>`**: Run a tool under a custom identity so peers can query it live
- **`aivo context` and `--context` injection**: Export recent session context as Markdown and inject it into any launched CLI for cross-tool handoffs
- **Copilot premium-request multiplier**: Surfaced in `aivo models` so you can see which Copilot models cost more
- **Chat TUI keybinding swap**: `Ctrl+C` and `Ctrl+L` swapped to match other agents

### Bug Fixes

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

## v0.17.0

### Features

- **Improved the ux of adding keys**: Interactive provider picker backed by the full known-provider catalog
- **Added JSON output via `--json`**: Enables scripting and `| jq` pipelines

### Refactors

- Streamline `logs` command and redesign the status output

## v0.16.2

### Bug Fixes

- Normalize `input_tokens` to fresh-only across all stats parsers for consistent accounting
- Exclude subagent sidechain files from Claude session count
- Add Claude Code env vars for timeouts, attribution, and subagent model
- Handle Ollama pull errors instead of silently succeeding
- Refresh PATH from login shell after tool install
- Move `--version` and `--help` before service init and improve smoke test diagnostics

## v0.16.1

### Bug Fixes

- Fix `ETXTBSY` on Linux during self-update: close write file descriptor before smoke test

## v0.16.0

### Features

- **Google Gemini native API**: Direct support for `generativelanguage.googleapis.com` as a provider across all tools
- **Open aivo-starter model list**: aivo-starter users can now access the full model catalog
- **Enriched `aivo models`**: Show context window, max output tokens, and pricing from the provider API
- **Prompt to install missing tools**: Interactively offer to install tool binaries when not found on PATH, with cross-platform support
- **R2 download mirror**: GitHub-first binary downloads with Cloudflare R2 fallback for faster installs and updates

### Bug Fixes

- Clear `last_selection` on key add so the newly added key is actually used
- Resolve aivo-starter sentinel URL in serve router and model picker
- Normalize Pi stats token counting to include cached input tokens
- Reduce GitHub download timeout and deduplicate mirror fallback messages
- Fix mirror fallback for self-update flow

## v0.15.0

### Features

- **aivo-starter**: Zero-config provider — start using aivo without any API key setup
- **Update rollback**: Automatically roll back failed updates; added config migration tests and CI clippy gate
- **Local session logging**: SQLite-backed `aivo logs` command for browsing session history
- **Native top session view**: Opt-in `aivo stats --top` for a live session overview
- **Combined short flags**: Support Unix-style combined flags like `-xr`, `-nar`
- **Ollama lifecycle management**: Auto-stop Ollama on exit using PID-file refcount for safe concurrent instances
- **DeepSeek reasoning streaming**: Stream `reasoning_content` through routers for DeepSeek-reasoner models
- **Conditional default model option**: Only show "default model" in the picker when the selected tool supports it

### Bug Fixes

- Cap `max_tokens` for aivo-starter and DeepSeek in chat requests
- Remove last two production `unwrap()` calls for safer error handling
- Fix device auth for starter provider across all tools
- Hide default model option in chat mode since it requires a concrete model
- Support Responses API-only Copilot models (e.g. gpt-5.4) for Claude and Gemini
- Resolve tilde paths and add PDF/binary support for chat attachments
- Remove tool name from active key display, show only key and model

### Performance

- Avoid PBKDF2 decryption when displaying active selection label
- Warm model cache in background after adding API key

### Refactors

- Redesign key/model selection: per-directory → global last-selection
- Replace `sqlite3` CLI with `rusqlite` for OpenCode stats reading
- Route OpenCode through router for providers with quirks

## v0.14.5

Major update with stats aggregation, better tool support

### Improvements

- Global stats aggregation across all AI tools (Claude, Codex,
  Gemini, OpenCode, Pi).
- Mask API key input with asterisks during entry
- Show install guide when a tool is not found on PATH
- Support Pi tool with Copilot subscription
- Rename `ls` command to `info` (keep `ls` as alias)
- Embed provider registry as JSON with table-driven tests
- Remove redundant token stats recording from run tool
- Bump GitHub Actions to v5 for Node.js 24 compatibility

### Fixes

- Remove custom User-Agent headers from API requests
- Use Codex `model_provider` config to bypass `auth.json` and
  `OPENAI_BASE_URL` deprecation
- Wire `--refresh` flag through run command for model picker
  cache bypass
- Auto-strip `anthropic-beta` headers for Bedrock/Vertex providers


## v0.14.4

Stability hardening. Fixed panics from char-boundary slicing
and API response handling. Switched Linux builds to musl
targets for better portability.

## v0.14.3

Added Responses API fallback for models that require the
/v1/responses endpoint. Fixed /attach command autocomplete.

## v0.14.2

Bug fixes and CI improvements.
