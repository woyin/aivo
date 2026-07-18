[![aivo](https://getaivo.dev/banner.webp)](https://getaivo.dev)

Aivo `/ˈeɪ.voʊ/` is a lightweight CLI that runs coding agents — Claude Code, Codex, Gemini,
OpenCode, Pi, and more — keeps your keys encrypted, and ships starter models that need no API key.

![CI](https://github.com/yuanchuan/aivo/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/tag/yuanchuan/aivo?label=release&color=brightgreen)
![MSRV](https://img.shields.io/badge/rustc-1.97+-orange.svg)
![Binary size](https://img.shields.io/badge/binary-%3C10MB-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

[**Docs**](https://getaivo.dev) · [Install](#install) · [Quick Start](#quick-start) · [Coding agents](#coding-agents) · [Commands](#commands)

---

## Install

Install script (macOS, Linux):

```bash
curl -fsSL https://getaivo.dev/install.sh | bash
```

Homebrew:

```bash
brew install yuanchuan/tap/aivo
```

PowerShell (Windows):

```powershell
irm https://getaivo.dev/install.ps1 | iex
```

## Quick Start

The built-in `aivo/starter` provider activates on first run, so no key is required to try it:

```bash
aivo "tell me a short story"
aivo claude
```

Add a key to access more models:

```bash
aivo keys add                                # interactive picker
aivo claude
aivo claude --model moonshotai/kimi-k2.5     # pin a model
```

## Coding agents

| Command | Agent | Type |
| ------- | ----- | ---- |
| `claude` | [Claude Code](https://github.com/anthropics/claude-code) | built-in |
| `codex` | [Codex](https://github.com/openai/codex) | built-in |
| `codex-app` | [Codex.app](https://github.com/openai/codex) desktop (macOS only, experimental) | built-in |
| `gemini` | [Gemini CLI](https://github.com/google-gemini/gemini-cli) | built-in |
| `opencode` | [OpenCode](https://github.com/anomalyco/opencode) | built-in |
| `pi` | [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent) | built-in |
| `amp` | [Amp](https://ampcode.com) | plugin |
| `omp` | [oh-my-pi](https://github.com/can1357/oh-my-pi) | plugin |
| `copilot` | [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/copilot-cli) | plugin |
| `grok` | [Grok CLI](https://x.ai/cli) | plugin |

```bash
aivo claude                                  # launch with active key
aivo claude "fix the login bug"              # pass-through args
aivo claude -m moonshotai/kimi-k2.5          # pin a model (bare -m opens picker)
aivo codex -k openrouter                     # use a specific saved key
aivo pi --dry-run                            # preview command + env, don't launch
aivo opencode --debug                        # JSONL log of upstream HTTP traffic
```

Without a tool name, `aivo run` opens the tool picker — native agents and installed coding-agent plugins.

## Commands

| Command | What it does |
| ------- | ------------ |
| [`keys`](#keys) | Manage saved API keys (encrypted) |
| [`account`](#account) | Link a device, check plan & usage |
| [`models`](#models) | List models from the active provider |
| [`code`](#code) | Built-in terminal coding agent |
| [`hf`](#hf) | Run open-weight GGUF models locally |
| [`serve`](#serve) | Local OpenAI-compatible endpoint |
| [`alias`](#alias) | Shortcuts for models and launch presets |
| [`logs`](#logs) | Unified activity feed across tools |
| [`stats`](#stats) | Token usage across every tool |
| [`update`](#update) | Update aivo to the latest version |
| [`plugins`](#plugins) | Add custom `aivo-<name>` commands |

### keys

Manage saved API keys. Stored AES-256-GCM encrypted in the user config directory.

```bash
aivo keys                                    # list
aivo keys add                                # interactive picker
aivo keys use openrouter                     # switch active key
aivo keys cat | edit | rm <name>
```

One-liner.

```bash
aivo keys add --name groq --base-url https://api.groq.com/openai/v1 --key sk-xxx
```

#### Export & import

Move keys between machines via a password-encrypted file:

```bash
aivo keys export ~/bak.keys     # prompts for password
aivo keys import ~/bak.keys     # same password on the other machine
aivo keys import https://example.com/bak.keys   # or from a URL

# non-interactive with password on stdin
aivo keys export ~/bak.keys --password-stdin <<< "my secret password"
```

### account

Link this device to your [getaivo.dev](https://getaivo.dev) account to unlock higher
`aivo/starter` limits, then check your plan and usage.

```bash
aivo account login/logout
aivo account usage                           # requests/tokens, daily caps, per-model
aivo account usage --json                    # machine-readable usage
aivo account login --label "work laptop"     # name this device in your account
aivo account open                            # open your dashboard in the browser
```

### models

List models from the active provider. Cached for one hour.

```bash
aivo models
aivo models --refresh                        # bypass cache
aivo models -s sonnet                        # filter by substring
aivo models --json | jq '.models[].id'
```

### code

`aivo code` is the built-in coding agent in your terminal.

![aivo](https://getaivo.dev/aivo-chat.webp)

```bash
aivo code                                    # full-screen agent TUI
aivo code -m gpt-4o                          # pick a model (remembered per key)
aivo code --attach README.md                 # attach a file for the agent to read
```

One-shot mode with `-p`:

```bash
aivo -p "Summarize this repo"                # same, via the explicit flag
git diff | aivo -p "Write a commit message"  # piped stdin appended as context
cat error.log | aivo -p                      # stdin alone becomes the prompt
```

Headless agent mode with `-e/--exec` runs tools and exits. Limit long unattended
runs with `--max-steps <N>` or `--max-output-tokens <N>` (0 disables each limit).

```bash
aivo code -e "make the failing test pass"
aivo code -e "fix lint" --max-steps 50 --max-output-tokens 20000
```

### hf

Run open-weight GGUF models locally, it fetches and caches them from HuggingFace repositories.

```bash
aivo code hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo https://huggingface.co/allenai/Olmo-3-1025-7B              # full URL also works
aivo code hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M        # pin a specific quant
aivo pi -m hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF                   # any tool that accepts -m
```

The `hf:` form is accepted anywhere a model is — `-m`, code's positional `REF`, and as a bare top-level arg (which rewrites to `aivo code hf:...`). Full HuggingFace URLs (`https://huggingface.co/...`) work the same way.

The local `llama-server` is configured from the model, no setup required: it runs at the model's real context window (capped at 65536), and if the repo ships an `mmproj-*.gguf` projector or a `*-MTP.gguf` draft model, those are pulled and wired up automatically — enabling image input and speculative decoding respectively. Tune or opt out with environment variables:

```bash
AIVO_LLAMA_CTX=16384      # override the context size (e.g. on a low-RAM machine)
AIVO_LLAMA_ARGS='--temp 0.1'  # pass extra llama-server flags (override aivo's)
AIVO_LLAMA_MMPROJ=off     # skip the auto-detected vision projector
AIVO_LLAMA_DRAFT=off      # skip the auto-detected speculative-decoding draft model
AIVO_LLAMA_NGL=20         # GPU layers to offload (AIVO_GPU=cpu disables GPU)
```

Manage the cached GGUF files (under `~/.config/aivo/cache/huggingface`):

```bash
aivo hf                                       # list cached repos
aivo hf list --verbose                        # show individual files
aivo hf pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo hf rm <repo> --quant Q5_K_M              # delete one quant
aivo hf rm <repo> --all -y                    # delete whole repo
aivo hf clean -y                              # wipe everything
```

### serve

Expose the active provider as a local OpenAI-compatible endpoint.

```bash
aivo serve                                   # http://127.0.0.1:24860
aivo serve -p 8080 --host 0.0.0.0
aivo serve --failover                        # retry across keys on 429/5xx
aivo serve --cors                            # enable CORS for browser clients
aivo serve --auth-token                      # require bearer token (auto-generated)
aivo serve --log /tmp/requests.jsonl
```

### alias

Short names for models or launch presets. Both share one namespace.

```bash
aivo alias                                   # list
aivo alias fast=claude-haiku-4-5             # model alias
aivo alias quick claude --key work -m fast   # launch alias

aivo claude -m fast                          # use anywhere `-m` is accepted
aivo quick                                   # invoke launch alias directly
aivo quick -k personal                       # explicit flags override the preset

aivo alias rm fast                           # remove (works for both kinds)
```

Names that collide with built-in subcommands or tool names are rejected.

### logs

Unified activity feed across aivo's own events (`code`, `run`, `serve`) and native CLI sessions (`claude`, `codex`, `gemini`, `pi`, `opencode`). Defaults to the current project's cwd; use `-a` for every project.

```bash
aivo logs                                    # current cwd, newest first
aivo logs -a                                 # all projects
aivo logs show <id>                          # logs.db id or native session id

aivo logs --by claude -n 5                   # claude run-events + native sessions
aivo logs --by native                        # only native CLI sessions
aivo logs -s "rate limit" --since 7d --errors
aivo logs --watch --jsonl                    # live tail as JSONL
```

Share a session via a tunneled viewer URL:

```bash
aivo logs share                              # interactive picker
aivo logs share <id>                         # share by id prefix
```

### stats

Aggregates token counts from aivo code, Claude Code, Codex, Gemini, OpenCode, and Pi by reading each tool's native data files.

```bash
aivo stats
aivo stats --by claude --since 7d            # one tool, recent window
aivo stats --by omp                          # a coding-agent plugin
aivo stats -s openrouter -n                  # filter, exact numbers
aivo stats --json | jq '.totals.tokens'
```

### update

Update to the latest version. Delegates to Homebrew or npm when installed by those package managers.

```bash
aivo update
aivo update --force                          # force even if pkg-managed
aivo update --rollback                       # restore previous backup
aivo update --sync-model-data                # sync model metadata
```

### plugins

Add a top-level command — a standalone `aivo-<name>` executable, in any language, that aivo runs
as `aivo <name>`. Plugins run with your privileges; install only ones you trust. Full contract:
[docs/PLUGIN-PROTOCOL.md](docs/PLUGIN-PROTOCOL.md).

```bash
aivo plugins install ./aivo-amp              # local file or http(s) URL
aivo plugins install github:owner/aivo-amp   # GitHub release (OS/arch asset)
aivo plugins install npm:aivo-foo            # npm package (node shim)
aivo plugins install cargo:aivo-bar          # cargo install from crates.io
aivo amp --help                              # runs the sibling aivo-amp
aivo plugins list                            # installed, with version/roles/caps
aivo plugins update amp                      # re-fetch / re-resolve the recorded source
aivo plugins rm amp
```

## License

[MIT](LICENSE)
