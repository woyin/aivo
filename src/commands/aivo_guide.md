# aivo — complete usage guide

aivo is a CLI that connects your existing coding agents (and its own built-in agent) to whatever
model you want, with local encrypted API-key management. It ships free `aivo/starter` models, so
you can use it with no key. Config lives in `~/.config/aivo/` — API keys are AES-256-GCM encrypted
in `config.json`.

Machine-readable command structure: `aivo --help-json`. Per-command help: `aivo <command> --help`.

## Global usage

```
aivo <command> [options]
```

Global options: `-h/--help`, `--help-json` (full command tree as JSON), `-v/--version`.

**Shortcuts** (bare forms that expand to a subcommand):

| Shortcut | Expands to | Notes |
| --- | --- | --- |
| `aivo <tool>` | `aivo run <tool>` | `claude`, `codex`, `gemini`, `opencode`, `pi`, `grok`, … |
| `aivo <prompt>` | `aivo chat -p <prompt>` | a bare string / piped stdin runs a one-shot chat |
| `aivo hf:… ` / `aivo https://…` | `aivo chat <ref>` | open chat on a local HuggingFace GGUF |
| `aivo use` | `aivo keys use` | switch active key |
| `aivo ping` | `aivo keys ping` | health-check keys |
| `aivo share` | `aivo logs share` | share a session |

## Providers & keys — `aivo keys`

A key is a saved provider credential (id, name, base URL, secret). The **active** key is what
commands use by default. Any OpenAI-compatible base URL works; native Anthropic/Gemini, GitHub
Copilot, and Ollama are also supported (the sentinel base URLs `copilot` and `ollama` select those
provider types). `aivo/starter` is the bundled first-party provider — no key required.

```bash
aivo keys                       # list all keys (active marked) — same as `aivo keys list`
aivo keys use <id|name>         # activate a key
aivo keys add [name]            # add a key (interactive picker for provider type)
aivo keys cat <id|name>         # show a key's details
aivo keys edit <id|name>        # edit a key
aivo keys rm <id|name>          # remove a key
aivo keys reauth <id|name>      # OAuth re-login or rotate an API key
aivo keys ping [id|name]        # health-check keys (also `aivo ping`)
aivo keys reset-route <id|name> # clear cached provider routing for a key
```

One-liner add (non-interactive):

```bash
aivo keys add --name groq --base-url https://api.groq.com/openai/v1 --key sk-xxx
```

Move keys between machines (password-encrypted file):

```bash
aivo keys export <file>         # prompts for a password
aivo keys import <file>         # or a URL; same password on the other machine
aivo keys export <file> --password-stdin <<< "my password"   # non-interactive
```

## Account — `aivo account`

Link this device to your [getaivo.dev](https://getaivo.dev) account for higher `aivo/starter`
limits, then check plan and usage.

```bash
aivo account                    # identity, plan, linked-device count (same as `info`)
aivo account usage              # requests/tokens, daily caps, per-model  (--json for machine form)
aivo account login              # sign in + link this device  (--label "work laptop" to name it)
aivo account logout             # sign out + unlink this device
aivo account open               # open your dashboard in the browser
```

`login`/`logout` are interactive — run them yourself in a terminal, not headless.

## Models & aliases

```bash
aivo models                     # list models from the active provider (cached ~1h)
aivo models -s sonnet           # -s/--search filters by substring
aivo models -r                  # -r/--refresh bypasses the cache
aivo models --json | jq '.models[].id'
aivo models -k <id|name>        # a specific key's provider
```

Aliases (`aivo alias`) — model names and launch presets share one namespace:

```bash
aivo alias                                  # list (--json for machine form)
aivo alias fast=claude-haiku-4-5            # model alias → use with -m/--model anywhere
aivo alias quick claude -k work -m fast     # launch preset → run via `aivo run quick`
aivo run quick                              # invoke the launch alias (flags override the preset)
aivo alias rm fast                          # remove either kind
```

## Launching coding agents — `aivo run`

`aivo <tool>` (or `aivo run <tool>`) launches a coding agent pointed at your active key/model; no
tool name opens a picker. All extra args pass straight through to the tool.

Built-in tools: `claude` (Claude Code), `codex`, `codex-app` (desktop, experimental, macOS),
`gemini`, `opencode`, `pi`. Plugin tools (install via `aivo plugins`): `amp`, `omp` (oh-my-pi),
`copilot`, `grok`.

Key flags:

```
-m, --model <model>        model to use (bare -m opens a picker)
-k, --key <id|name>        use a specific saved key
-r, --refresh              bypass the model-list cache
--max-context <size>       larger context window (e.g. 1m, 2m);  --1m / --2m shorthands
-c, --context[=<id>]       inject one past session as context
--env <k=v>                inject an environment variable
--relogin                  force OAuth re-login (codex / codex-app / claude)
--dry-run                  print the resolved command + env without launching
--transparent              pi only: bypass the router, talk to the model natively
```

Claude-only model-slot overrides: `--reasoning-model`, `--subagent-model`, and
`--haiku|--sonnet|--opus-model` (what `/model <name>` resolves to; bare = picker).

```bash
aivo claude                                  # launch with the active key
aivo claude "fix the login bug"              # pass-through args
aivo claude -m moonshotai/kimi-k2.5          # pin a model
aivo codex -k openrouter                     # a specific saved key
aivo pi --dry-run                            # preview the command + env, don't launch
```

## The built-in agent — `aivo chat`

`aivo chat` is aivo's own coding agent in your terminal (full-screen TUI). It reads/edits your
project and runs shell commands, prompting for risky actions.

```
aivo chat                     # interactive agent TUI
-m, --model <model>           pick a model (remembered per key; bare = picker)
-k, --key <id|name>           API key by id/name (bare = picker)
-p, --prompt [prompt]         one prompt, plain reply, exit (no tools)
-e, --exec   [prompt]         one prompt, run the full agent (tools), exit
-r, --refresh                 refresh the model list (skip cache)
--resume [last|id]            resume a saved chat
--share                       share this chat live (needs `aivo login`)
--attach <path>               attach a file or image
--json                        raw provider JSON (with -p)
--max-context <size>          override the context window (e.g. 200k)
--dry-run                     show the resolved key/model/endpoint, don't connect
```

```bash
aivo -p "Summarize this repo"                # bare string → one-shot plain reply
git diff | aivo -p "Write a commit message"  # piped stdin appended as context
aivo chat -e "make the failing test pass"    # one-shot agent run
```

### Inside the chat TUI

Type `/help` for the full list. Slash commands:

- Session: `/new`, `/resume [query]`, `/rewind` (undo edits), `/copy [n]`, `/config`, `/share [stop]`, `/help`, `/exit`
- Model & key: `/model [name]`, `/key [id|name]`, `/effort [level]`
- Context: `/attach <path>`, `/detach <n>`, `/compact [fast]`
- Skills & tools: `/skills`, `/create-skill`, `/mcp`
- Autonomous: `/plan <objective>`, `/goal <objective>`

Other input: `!cmd` runs a local shell command; `//` / `!!` escape to literal text.

Keys: `Enter` send · `Ctrl+J` newline · `Tab` complete · `Ctrl+V` paste text/image ·
`Shift+Tab` toggle auto-approve · `Ctrl+R` resume · `Ctrl+M` model picker · `Ctrl+O` pager for a
`!cmd` · `Esc` cancel/close · `Ctrl+C` twice to exit.

`/config` toggles: Thinking, Auto-approve tools, aivo web search, Agent tools (off = plain chat,
no tools). The agent can also change the live model/effort itself when you ask (it calls its
`switch_model` / `set_effort` tools); a key change it hands back to you via `/key`.

## Local models — `hf:` and `aivo hf`

Run open-weight GGUF models locally — fetched/cached from HuggingFace and served by a local
`llama-server`, zero setup. The `hf:` form works anywhere a model is accepted (`-m`, chat's
positional ref, or a bare top-level arg); full `https://huggingface.co/…` URLs work too.

```bash
aivo chat hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo https://huggingface.co/allenai/Olmo-3-1025-7B
aivo chat hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M   # pin a quant
aivo pi -m hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF              # any tool that accepts -m
```

Manage the cache (`~/.config/aivo/cache/huggingface`):

```bash
aivo hf                          # list cached repos  (list --verbose for files)
aivo hf pull <ref|path>          # download a GGUF, or import a local .gguf (--as name)
aivo hf rm <repo> --quant Q5_K_M # delete one quant   (--all -y for the whole repo)
aivo hf clean -y                 # wipe every cached repo
```

llama-server auto-runs at the model's real context window (capped at 65536); an `mmproj-*.gguf`
projector (vision) or `*-MTP.gguf` draft (speculative decoding) in the repo is wired up
automatically. Tune with env vars (see Environment).

## Serve an OpenAI-compatible API — `aivo serve`

Expose the active provider (or a local `hf:` model) as a local endpoint.

```
aivo serve                     # http://127.0.0.1:24860
-p, --port <PORT>              port (default 24860)
--host <ADDR>                  bind address (default 127.0.0.1)
-k, --key <id|name>            which key to proxy
--failover                     retry across keys on 429/5xx
--cors                         CORS headers for browser clients
--auth-token [TOKEN]           require a bearer token (auto-generated if omitted)
--timeout <SECS>               upstream timeout (default 300)
--log [PATH]                   log requests as JSONL (stdout or PATH)
```

```bash
aivo serve --host 0.0.0.0 -p 8080
aivo serve hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF   # serve a local model
```

## Logs & sharing — `aivo logs`

Unified session list across aivo chat + native CLI sessions (claude, codex, gemini, pi,
opencode); `--by run`/`--by serve` show launch events. Defaults to the current project's cwd.

```
aivo logs                      # recent rows, newest first  (list)
aivo logs show [id]            # one row in detail (omit id → picker)
aivo logs share [id]           # share via a tunneled viewer URL (omit id → picker)
aivo logs prune                # drop logs.db events whose session file is gone
```

Filters: `-n/--limit`, `--by <source|plugin>`, `-s/--search`, `-a/--all` (or `--cwd <path>`),
`--since`/`--until`, `--model`, `-k`, `--errors`, `--json`, `--watch` (`--jsonl` to stream).

Sharing (`aivo logs share`, alias `aivo share`) creates an ephemeral, tunneled viewer URL that
dies when the process exits. Redacts keys/tokens/`$HOME`/secrets by default (`--no-redact` to
skip); `--all` picks from every project, `--open` opens the browser.

## Usage stats — `aivo stats`

Aggregates token/request counts from aivo chat and every launched agent by reading each tool's
native data files.

```
aivo stats                     # totals + top models
--by <name>                    one tool or plugin (claude, chat, omp, …)
--since <7d|24h|30m|2w>        recent window
-s, --search <query>           filter by key / model / tool
-d, --detailed                 per-model input/output/cached/total
-a, --all                      all models (default: top 20)
-n, --numbers                  exact numbers
--json                         machine-readable
```

## Info — `aivo info`

```bash
aivo info                      # system info, keys, tools, directory state
aivo info --ping               # ping all keys, pass/fail summary
aivo info --ping --json | jq '.keys[] | select(.ping.ok==false)'
```

## Plugins — `aivo plugins`

A plugin is a sibling `aivo-<name>` executable (any language) that aivo runs as `aivo <name> …`.
Plugins run with your privileges — install only ones you trust.

```bash
aivo plugins                                 # list installed (version / roles / caps)
aivo plugins install ./aivo-amp              # local path
aivo plugins install github:owner/aivo-amp   # GitHub release asset (OS/arch)
aivo plugins install npm:aivo-foo            # npm package (node shim)
aivo plugins install cargo:aivo-bar          # cargo install from crates.io
aivo plugins install <url> --name amp        # http(s) URL, custom name
aivo plugins update [name]                   # re-install from the recorded source
aivo plugins rm amp -y
```

## Update — `aivo update`

```bash
aivo update                    # update to the latest (delegates to Homebrew/npm if managed)
aivo update --force            # force even if package-managed
aivo update --rollback         # restore the previous version
aivo update --sync-model-data  # refresh model metadata from models.dev
```

## Environment variables

- `AIVO_LLAMA_CTX` — override local llama-server context size (e.g. `16384`)
- `AIVO_LLAMA_ARGS` — extra `llama-server` flags (overrides aivo's)
- `AIVO_LLAMA_MMPROJ=off` — skip the auto-detected vision projector
- `AIVO_LLAMA_DRAFT=off` — skip the auto-detected speculative-decoding draft model
- `AIVO_LLAMA_NGL=<n>` — GPU layers to offload (`AIVO_GPU=cpu` disables GPU)
- `AIVO_NO_UPDATE_NOTICE=1` (or `CI`) — suppress the update-available notice
- `NO_PROXY=127.0.0.1,localhost` — set when an `http_proxy` is configured, so `aivo serve` /
  local llama endpoints aren't proxied

## Exit codes

`0` success · `1` user error · `2` network · `3` auth.

Full docs: https://getaivo.dev
