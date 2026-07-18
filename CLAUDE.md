# CLAUDE.md

## Project Overview

`aivo` is a Rust CLI providing unified access to multiple AI coding assistants (Claude, Codex, Gemini, OpenCode, Pi) with local API key management and secure storage. Supports OpenAI-compatible providers, GitHub Copilot, OpenRouter, Ollama, native APIs, and OAuth accounts.

> [!IMPORTANT]
> **Rebuild before testing**: after code changes, run `cargo build && cargo install --path . --debug` before testing the binary — never test a stale build. Use `--release` only for final pre-release verification.

## Build & Test

```bash
cargo build                             # Debug build (~1s incremental)
cargo test --features __internal_test_fast_crypto  # All tests (~3000; reduced PBKDF2 iterations)
cargo test -- test_name                 # Single test
cargo clippy                            # Lint (fix all warnings before committing)
cargo fmt                               # Format (run before committing)
```

A `Makefile` wraps these: `make test|build|clippy|install|release`.

Tests are hermetic: a pre-main sandbox (`tests/support/mod.rs`, included by every test binary and the lib) points `$HOME` at `~/.aivo-test-home/<pid>`, so tests can never touch the real config. `tests/sandbox_linux.rs` deliberately opts out. Per-case isolation still uses `with_path` + tempdirs; the sandbox is the safety net.

## Git Conventions

- Always squash merge to main: `git merge --squash <branch> && git commit`
- Never commit automatically — only when asked.

## Release Process

> [!IMPORTANT]
> **Tag only after CI is green on main.** `ci.yml` runs the test matrix on every `main` push. Tagging before tests pass burns the version number — a failed release can't be re-cut on the same tag, and any `chore: release vX.Y.Z` commit becomes a zombie. Push main, wait for the green check, then tag.

1. Bump version in `Cargo.toml` and `npm/package.json` — never tag without bumping.
2. `cargo fmt`, `cargo clippy -- -D warnings`, `cargo test`.
3. `cargo build --release && cargo install --path .` to verify.
4. Stage exactly the release files (`Cargo.toml`, `Cargo.lock`, `npm/package.json`; never `git add -A`), commit `chore: release vX.Y.Z`, push main.
5. Wait for CI to pass on **all three runners** (Linux, macOS, Windows) — `#[cfg(windows)]` code is invisible to Linux/macOS clippy. `gh run watch $(gh run list --workflow=ci.yml --branch=main --limit=1 --json databaseId --jq '.[0].databaseId') --exit-status`
6. `git tag vX.Y.Z && git push origin vX.Y.Z` (triggers `release.yml`).

## CLI / UX Conventions

Match existing CLI help text formatting exactly (alignment, spacing, bracket style). Help output is hand-rolled (`print_help*` fns), not clap-generated. When implementing interactive UI, verify: keyboard handling (arrows, Ctrl+P/N, ESC, Ctrl+C), selection pre-selection, column alignment, and edge cases (empty input, single item, long strings).

## Architecture

```
src/main.rs → src/commands/* → SessionStore → EnvironmentInjector → AILauncher
```

- **`src/`**: entry point, CLI parsing, error handling, TUI components, styling
- **`src/commands/`**: one module per subcommand — `run`, `start` (interactive picker), `code` (coding-agent TUI + one-shot), `keys`, `serve`, `mcp`, `agents`, `plugins`, `logs`, `stats`, …
- **`src/services/`**: key/session/stats stores, process launching, provider routers and wire-format bridges (Anthropic ⇄ OpenAI ⇄ Gemini, Copilot, Ollama), OAuth flows, model-name transforms, HTTP utilities

**Data model**: `ApiKey` (`id`, `name`, `base_url`, `key`, `created_at`) AES-256-GCM encrypted in `config.json` under the config dir — `$AIVO_CONFIG_DIR`, default `~/.config/aivo`, with `state/ secrets/ cache/ logs/ run/` subdirs. Sentinel `base_url` values `"copilot"` and `"ollama"` identify special provider types.

**Cross-platform**: platform-specific code gated behind `cfg(unix)` / `cfg(windows)`.

**Exit codes**: 0 = success, 1 = user error, 2 = network, 3 = auth.

## Instructions

* Restate the question in fully concrete terms, making every implicit detail explicit. Then answer.
* Comments: clean, concise, why-only. Don't comment obvious code.
