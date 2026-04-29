# aivo

A CLI for managing API keys and running Claude Code, Codex, Gemini, OpenCode, and Pi across providers.

## What it does

- Stores multiple provider API keys, encrypted at rest.
- Runs `claude`, `codex`, `gemini`, `opencode`, and `pi` against any saved key.
- Includes a chat TUI and a one-shot `-x` mode.
- Can expose the active provider as a local OpenAI-compatible server.

## Install

Homebrew:

```bash
brew install yuanchuan/tap/aivo
```

Install script:

```bash
curl -fsSL https://getaivo.dev/install.sh | bash
```

Via npm (recommended for Windows users):

```bash
npm install -g @yuanchuan/aivo
```

Or download a binary from [GitHub Releases](https://github.com/yuanchuan/aivo/releases).


## Quick Start

aivo ships with a free built-in provider (`aivo/starter`) that activates on first run — no API key needed:

```bash
aivo -x hello
aivo claude
```

Add your own provider key for access to more models:

```bash
# 1) Add a provider key (OpenRouter, Vercel AI Gateway, etc.)
aivo keys add

# 2) Launch your tool
aivo claude

# 3) Optionally pin a model
aivo claude --model moonshotai/kimi-k2.5
```

Use your GitHub Copilot subscription.

```bash
aivo keys add        # pick "GitHub Copilot" from the provider list
aivo claude
```

Use local models via Ollama.

```bash
aivo keys add        # pick "Ollama" from the provider list

# auto-pulls the model if not present
aivo claude --model llama3.2
```

## Commands

| Command | Description |
| ------- | ----------- |
| [run](#run) | Launch an AI tool (claude, codex, gemini, opencode, pi) |
| [keys](#keys) | Manage API keys (add, use, rm, cat, edit, ping) |
| [models](#models) | List available models from the active provider |
| [alias](#alias) | Create short names for models |
| [chat](#chat) | Interactive chat TUI or one-shot `-x` mode |
| [image](#image) | Generate images from a text prompt |
| [serve](#serve) | Local OpenAI-compatible API server |
| [info](#info) | Show system info, keys, tools, and directory state |
| [logs](#logs) | Query local SQLite logs for chat, run, and serve |
| [stats](#stats) | Show usage statistics |
| [context](#context) | Show recent cross-CLI activity for this project |
| [update](#update) | Update to the latest version |

## run

Launch an AI tool with the active provider key. All extra arguments are passed through to the underlying tool.

Supported tools:

- `claude` [Claude Code](https://github.com/anthropics/claude-code)
- `codex` [Codex](https://github.com/openai/codex)
- `gemini` [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `opencode` [OpenCode](https://github.com/anomalyco/opencode)
- `pi` [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)

The `run` keyword is optional — tool names work directly as shortcuts, so `aivo claude` is equivalent to `aivo run claude`.

```bash
aivo run claude
aivo claude "fix the login bug"
aivo claude --dangerously-skip-permissions
aivo claude --resume 16354407-050e-4447-a068-4db222ff841
```

#### `--model, -m`

Pick a model for one run, or omit the value to open the model picker:

```bash
aivo claude --model moonshotai/kimi-k2.5
aivo claude --model                      # opens model picker
aivo claude -m                           # short form
```

#### `--key, -k`

Select a saved key by ID or name:

```bash
aivo claude --key openrouter
aivo claude --key copilot
aivo claude --key                        # opens key picker
```

#### `--refresh, -r`

Bypass cache and fetch a fresh model list for the picker:

```bash
aivo claude -r
```

#### `--dry-run`

Preview the resolved command and environment without launching:

```bash
aivo claude --dry-run
```

#### `--env, -e`

Inject extra environment variables into the child process:

```bash
aivo claude --env BASH_DEFAULT_TIMEOUT_MS=60000
```

#### Claude per-slot model overrides

Pin a different model to one of Claude Code's named slots without touching the others. Bare flag opens the model picker:

```bash
aivo claude --reasoning-model claude-opus-4-6
aivo claude --subagent-model claude-haiku-4-5
aivo claude --haiku-model    claude-haiku-4-5
aivo claude --sonnet-model   claude-sonnet-4-6
aivo claude --opus-model                       # picker
```

#### `--1m` / `--2m` / `--max-context`

Claude only. Append the canonical `[1m]`/`[2m]` suffix to the resolved model so Claude Code uses the 1M or 2M context window:

```bash
aivo claude --1m
aivo claude --max-context=2m
aivo claude -m claude-sonnet-4-6 --1m          # equivalent to -m 'claude-sonnet-4-6[1m]'
```

#### `--debug`

JSONL HTTP logger. Records every upstream request/response from this launch to a file (default path printed at startup, or pass an explicit path):

```bash
aivo claude --debug
aivo claude --debug=/tmp/aivo-http.jsonl
```

#### `--context, -c`

Inject a past session from another CLI as background context for this launch. Bridges cross-tool handoffs that each tool's native `--resume` can't span — e.g. pick up in Claude where Codex left off:

```bash
aivo claude --context                    # opens session picker
aivo claude --context=abc123             # specific session (prefix match)
```

Use `aivo context` to see available session IDs.

#### `--as <name>`

Give this launch a nickname so other tools in the same directory can query its live session by name instead of juggling session IDs:

```bash
aivo claude --as reviewer
aivo codex --as coder
```

Cross-tool MCP is enabled by default — each tool auto-registers under its CLI name (`claude`, `codex`, etc.), incrementing on collision (`claude-2`, `claude-3`). Use `--as` only to override. Claude and Codex can call each other via `list_sessions` / `get_session`; Pi, Gemini, and OpenCode are read-only peers (queryable, but they can't query others).

#### `aivo run`

Without a tool name, `aivo run` opens the interactive start flow and remembers your last key + tool selection. The next `aivo run` skips the picker and launches that tool directly.

```bash
aivo run
```

## keys

Manage saved API keys. Keys are stored locally and encrypted in the user config directory.

```bash
aivo keys                                # list all keys
aivo keys --ping                         # list with live ping status
aivo keys --json                         # machine-readable list (secret excluded)
```

#### `keys add`

Add a new provider key. Interactive by default, or pass `--name`, `--base-url`, and `--key` for scripted setup:

```bash
aivo keys add
aivo keys add --name openrouter --base-url https://openrouter.ai/api/v1 --key sk-xxx
aivo keys add --name groq --base-url https://api.groq.com/openai/v1 --key sk-xxx
aivo keys add --name deepseek --base-url https://api.deepseek.com/v1 --key sk-xxx
```

Any endpoint that speaks a supported protocol can be saved — you are not limited to the providers above.

Running `aivo keys add` with no flags opens an interactive picker that covers the built-in OAuth flows:

- **GitHub Copilot** — uses your Copilot subscription via OAuth device flow
- **OpenAI Codex (ChatGPT)** — browser login, multi-account
- **Claude Code (Anthropic)** — browser login, multi-account
- **Gemini (Google)** — browser login, multi-account
- **Ollama** — connects to a local Ollama instance (auto-starts if needed)
- **aivo starter** — free built-in provider (auto-created on first run, re-add if removed)

Typing a matching name as the label (e.g. `aivo keys add codex`) pre-focuses the picker on that row, so it's still a one-keypress confirm.

```bash
aivo keys add                  # open the picker
aivo keys add codex            # picker with Codex pre-focused
aivo keys add aivo-starter     # non-interactive re-add
```

#### `keys use`

Switch the active key by name or ID:

```bash
aivo keys use openrouter
aivo keys use                            # opens key picker
aivo use openrouter                      # shortcut
```

#### `keys cat`

Print the decrypted key details:

```bash
aivo keys cat
aivo keys cat openrouter
```

#### `keys edit`

Edit a saved key interactively:

```bash
aivo keys edit
aivo keys edit openrouter
```

#### `keys rm`

Remove a saved key:

```bash
aivo keys rm openrouter
```

#### `keys ping`

Health-check the active key, or all keys:

```bash
aivo keys ping
aivo keys ping --all
aivo ping                                # shortcut
```

## models

List models available from the active provider. Model lists are cached for one hour.

```bash
aivo models
```

#### `--refresh, -r`

Bypass the cache and fetch a fresh model list:

```bash
aivo models --refresh
```

#### `--key, -k`

List models for a different saved key:

```bash
aivo models --key openrouter
```

#### `--search, -s`

Filter models by substring:

```bash
aivo models -s sonnet
```

#### `--json`

Output the model list as JSON:

```bash
aivo models --json | jq '.models[].id'
```

## chat

`aivo chat` starts the full-screen chat UI.

```bash
aivo chat
```

#### `--model, -m`

Specify or change the chat model. Omit the value to open the model picker. The selected model is remembered per saved key.

```bash
aivo chat --model gpt-4o
aivo chat -m claude-sonnet-4-5
aivo chat --model                        # opens model picker
```

#### `--key, -k`

Use a different saved key for this chat session:

```bash
aivo chat --key openrouter
aivo chat -k                             # opens key picker
```

#### `--execute, -x`

Send a single prompt and exit. When `-x` has a message, piped stdin is appended as context. When `-x` has no message, the entire stdin becomes the prompt.

```bash
aivo chat -x "Summarize this repository"
git diff | aivo -x "Write a one-line commit message"
cat error.log | aivo -x
aivo -x                                 # type interactively, Ctrl-D to send
```

`aivo -x` is a shortcut for `aivo chat -x`.

#### `--attach`

Attach text files or images to the next message (repeatable):

```bash
aivo chat --attach README.md --attach screenshot.png
```

#### `--refresh, -r`

Bypass the model cache when opening the model picker:

```bash
aivo chat -r
```

#### `--json`

With `-x`, print the provider's raw response body (same shape as `curl`).

```bash
aivo chat -x "hello" --json | jq -r '.choices[0].message.content'
```

#### Slash commands

Inside the chat TUI:

| Command | Description |
| ------- | ----------- |
| `/new` | Start a fresh chat with the current key and model |
| `/resume [query]` | Resume a saved chat from this directory |
| `/model [name]` | Switch the current chat model |
| `/key [id\|name]` | Switch to another saved key for this chat |
| `/attach <path>` | Attach a text file or image to the next message |
| `/detach <n>` | Remove one queued attachment by number |
| `/help` | Open command help |
| `/exit` | Leave chat |
| `//message` | Send a literal leading slash |

## image

Generate images from a text prompt against the active provider's image API (e.g. `gpt-image-1`, `dall-e-3`, Gemini image models). Experimental.

```bash
aivo image "a red panda in space"
aivo image "logo sketch" -m dall-e-3 -o logo.png
```

#### Common flags

```bash
aivo image "..." --model gpt-image-1
aivo image "..." --key openrouter
aivo image "..." --output ./out/{ts}-{model}.png   # path or template
aivo image "..." --size 1792x1024 --quality hd
aivo image "..." --url                              # print provider URL, skip download
aivo image "..." --json                             # machine-readable
```

## serve

`aivo serve` exposes the active provider as a local OpenAI-compatible endpoint, for scripts and tools that already speak the OpenAI API.

```bash
aivo serve                               # http://127.0.0.1:24860
```

#### `--port, -p`

Listen on a custom port (default: 24860):

```bash
aivo serve --port 8080
aivo serve -p 8080
```

#### `--host`

Bind to a specific address (default: 127.0.0.1):

```bash
aivo serve --host 0.0.0.0               # expose on all interfaces
```

#### `--key, -k`

Use a different saved key:

```bash
aivo serve --key openrouter
aivo serve -k                            # opens key picker
```

#### `--log`

Enable request logging. Logs to stdout by default, or to a file if a path is given:

```bash
aivo serve --log | jq .                  # JSONL to stdout
aivo serve --log /tmp/requests.jsonl     # JSONL to file
```

#### `--failover`

Enable multi-key failover on 429/5xx errors. Automatically retries with other saved keys:

```bash
aivo serve --failover
```

#### `--cors`

Enable CORS headers for browser-based clients:

```bash
aivo serve --cors
```

#### `--timeout`

Upstream request timeout in seconds (default: 300, 0 = no timeout):

```bash
aivo serve --timeout 60
```

#### `--auth-token`

Require a bearer token. Auto-generated if no value given:

```bash
aivo serve --auth-token                  # auto-generated token
aivo serve --auth-token my-secret        # specific token
```

## alias

Create short names. Two flavors share one namespace:

- **Model alias** — short name → model name, accepted anywhere `-m`/`--model` works.
- **Launch alias** — short name → preset (tool + flags), invoked via `aivo run <name>` or just `aivo <name>`.

```bash
aivo alias                                                 # list all aliases
```

#### Model aliases

```bash
aivo alias fast=claude-haiku-4-5
aivo alias best claude-sonnet-4-6                          # positional form
```

Use anywhere a model name is accepted:

```bash
aivo claude -m fast
aivo chat -m best
```

#### Launch aliases

When the first arg after the name is a known tool (`claude`, `codex`, `gemini`, `opencode`, `pi`), the alias becomes a launch preset:

```bash
aivo alias quick claude --key work --model fast --max-context 1m
aivo alias dev   codex  --key openrouter --model claude-sonnet-4-6
```

Run them:

```bash
aivo run quick                    # full form
aivo quick                        # top-level shortcut
```

Override individual flags by re-typing them on the command line — explicit user flags win over the bundle's preset. `-k`/`--key`, `-m`/`--model`, and `--1m`/`--2m`/`--max-context` are recognized as equivalent for the override:

```bash
aivo run quick --model other      # bundle's --model swapped out, --key still applies
aivo quick -k personal            # bundle's --key overridden via short form
```

A launch alias's `--model` can itself reference a model alias — `quick --model fast` resolves through `fast` to `claude-haiku-4-5`.

#### Remove an alias

`rm` works for both kinds:

```bash
aivo alias rm fast
aivo alias rm quick
```

#### `--json`

Output the alias list as JSON. Model entries are JSON strings; launch entries are `{"tool": ..., "args": [...]}` objects:

```bash
aivo alias --json
```

#### Reserved names

Alias names that collide with built-in subcommands, shortcut keywords (`use`, `ping`), or AI tool names (`claude`, `codex`, etc.) are rejected at definition time so they don't shadow `aivo <name>` dispatch.

## info

Show an overview of saved keys, installed tools, the last remembered tool/model selection, and the cached model count for the active key. (`ls` is accepted as an alias.)

```bash
aivo info
```

#### `--ping`

Also health-check all keys:

```bash
aivo info --ping
```

#### `--json`

Output info as JSON (combines with `--ping`):

```bash
aivo info --json
aivo info --ping --json | jq '.keys[] | select(.ping.ok==false)'
```

## logs

Query the local SQLite log database used by aivo chat, run, and serve. Chat logs include turn content and token usage. `run` logs record launch metadata only. `serve` logs record request metadata only.

`aivo logs` prints entries newest-first.

```bash
aivo logs
```

#### `show <id>`

Show one entry in detail:

```bash
aivo logs show 7m2q8k4v9cpr
```

#### `status`

Show entry counts, database size, and path:

```bash
aivo logs status
```

#### Filters

```bash
aivo logs --by chat -n 5
aivo logs --by claude --errors
aivo logs -s "rate limit"
aivo logs --model sonnet
aivo logs --key openrouter
aivo logs --cwd /path/to/project
aivo logs --since "2025-01-01" --until "2025-02-01"
aivo logs --json
```

#### Live watch

Poll and refresh matching logs continuously:

```bash
aivo logs --by run --watch
aivo logs --watch --jsonl
```

## stats

Show usage statistics across all tools. Aggregates token counts from aivo chat, Claude Code, Codex, Gemini, OpenCode, and Pi by reading each tool's native data files. Per-file caching makes subsequent runs fast.

```bash
aivo stats
```

#### Positional argument

Show stats for a single tool:

```bash
aivo stats claude
aivo stats chat
```

#### `--numbers, -n`

Show exact numbers instead of human-readable approximations:

```bash
aivo stats -n
```

#### `--search, -s`

Filter by key, model, or tool name:

```bash
aivo stats -s openrouter
```

#### `--refresh, -r`

Bypass cache and re-read all data files:

```bash
aivo stats -r
```

#### `--all, -a`

Show all models (default: top 20, rest grouped as "others"):

```bash
aivo stats -a
```

#### `--top-sessions`

Show the heaviest native session files:

```bash
aivo stats --top-sessions
```

#### `--since <DURATION>`

Filter to a recent time window. Accepts `Nm`, `Nh`, `Nd`, `Nw`:

```bash
aivo stats --since 7d
aivo stats claude --since 24h
aivo stats --since 2w
```

#### `--json`

Output stats as JSON (all models, exact numbers):

```bash
aivo stats --json | jq '.totals.tokens'
```

## context

Show recent cross-CLI activity for the current project. Sessions are derived on demand from each tool's native storage (Claude, Codex, Gemini, Pi, OpenCode) — aivo keeps no duplicate state.

```bash
aivo context
```

Pair with `aivo run <tool> --context` to inject one of these sessions into your next launch.

#### `--all, -a`

Show all sessions, bypassing the default 14-day age cap:

```bash
aivo context --all
```

#### `--last-days <N>`

Override the default age cap:

```bash
aivo context --last-days 30
```

#### `--json`

Dump every available thread as JSON:

```bash
aivo context --json | jq '.threads'
```

## update

Update to the latest version. Delegates to Homebrew or npm when installed by those package managers.

```bash
aivo update
```

#### `--force`

Force update even if installed via a package manager:

```bash
aivo update --force
```

#### `--rollback`

Restore the previous version from the last update backup:

```bash
aivo update --rollback
```

## Development

```bash
make build
make build-debug
make check
make test
make clippy
make build-release
```

## License

MIT
