# CLAUDE.md

`aivo` is a Rust CLI providing unified access to multiple AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi) with local API key management. Supports OpenAI-compatible providers, GitHub Copilot, OpenRouter, Ollama, native APIs, and OAuth accounts.

## Build & Test

```bash
cargo build                                        # Debug build
cargo test --features __internal_test_fast_crypto  # All tests (reduced PBKDF2 iterations)
cargo clippy && cargo fmt                          # Required clean before committing
```

- After code changes, `cargo build && cargo install --path . --debug` before testing the binary — never test a stale build.
- Tests are hermetic: a pre-main sandbox (`tests/support/mod.rs`) points `$HOME` at `~/.aivo-test-home/<pid>`, so tests can never touch the real config.

## Git & Release

- Squash merge to main; never commit unless asked.
- Release: bump version in `Cargo.toml`, fmt/clippy/test, stage only `Cargo.toml` + `Cargo.lock` + `CHANGELOG.md` (never `git add -A`), commit `chore: release vX.Y.Z`, push main, wait for CI green on **all three runners** (`#[cfg(windows)]` code is invisible to Linux/macOS clippy), then tag. **Never tag before CI is green** — a failed release can't be re-cut on the same tag.

## Architecture

`src/main.rs` → `src/commands/*` (one module per subcommand: `run`, `start`, `code`, `keys`, `serve`, …) → `src/services/*` (key/session/stats stores, process launching, provider routers, wire-format bridges Anthropic ⇄ OpenAI ⇄ Gemini, OAuth flows).

Keys are AES-256-GCM encrypted in `config.json` under `$AIVO_CONFIG_DIR` (default `~/.config/aivo`). Sentinel `base_url` values `"copilot"`/`"ollama"` mark special provider types. Exit codes: 0 success, 1 user error, 2 network, 3 auth.

## Conventions

- Match existing CLI help text formatting exactly; verify interactive-UI edge cases (keyboard handling, empty input, single item, long strings).
- Comments: concise, why-only. Don't comment obvious code.
- Restate the question in fully concrete terms, making every implicit detail explicit. Then answer.

## Anti-goals

Settled constraints — don't re-add or work around. Greppable ones are enforced by `tests/contracts.rs`; extend its lists when adding here.

- No hardcoded provider/model data — derive from `/v1/models` or the models.dev sync.
- Never `terminal.clear()` in the code TUI; no progress bars or emoji in TUI output.
- `/review`, `/vision`, and `/detach` commands are removed; no OSC 11 theme auto-detection (default dark).
- Never modify a launched coding agent's own config files — aivo injects env vars and override dirs instead.
- Help stays hand-rolled (`print_help*`), never clap-generated.
- One agent engine serves the code TUI and oneshot — no second execution path for the same feature.
- Releases ship via R2 + Homebrew; no GitHub Releases (`gh release list` is empty by design).
