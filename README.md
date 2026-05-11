[![aivo](https://getaivo.dev/banner.webp)](https://getaivo.dev)

> Aivo is a command-line tool that connects your coding agent to almost any model. It also includes built-in models out of the box — no API keys, no signup.

## Docs

https://getaivo.dev


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

Npm

```bash
npm install -g @yuanchuan/aivo
```

## Quick Start

The built-in `aivo/starter` provider activates on first run, so no key is required to try it:

```bash
aivo -x hello
aivo claude
```

Add a key to access more models:

```bash
aivo keys add                                # interactive picker
aivo claude
aivo claude --model moonshotai/kimi-k2.5     # pin a model
```

## run

Launch an AI tool with the active provider key. The `run` keyword is optional: `aivo claude` is equivalent to `aivo run claude`. Extra arguments are passed through.

Supported tools:

- `claude` [Claude Code](https://github.com/anthropics/claude-code)
- `codex` [Codex](https://github.com/openai/codex)
- `gemini` [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `opencode` [OpenCode](https://github.com/anomalyco/opencode)
- `pi` [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)
- `amp` [Amp](https://ampcode.com)

```bash
aivo claude                                  # launch with active key
aivo claude "fix the login bug"              # pass-through args
aivo claude -m moonshotai/kimi-k2.5          # pin a model (bare -m opens picker)
aivo claude -k openrouter                    # use a specific saved key
aivo claude --1m                             # Claude only: 1M context window
aivo claude --dry-run                        # preview command + env, don't launch
aivo claude --debug                          # JSONL log of upstream HTTP traffic
```

Pin a different model to one of Claude Code's named slots:

```bash
aivo claude --opus-model=deepseek-v4-pro --sonnet-model=deepseek-v4-flash
```

Without a tool name, `aivo run` opens the interactive start flow and remembers the last selection.

## keys

Manage saved API keys. Stored AES-256-GCM encrypted in the user config directory.

```bash
aivo keys                                    # list
aivo keys add                                # interactive picker (OAuth flows + custom URLs)
aivo keys add --name groq --base-url https://api.groq.com/openai/v1 --key sk-xxx
aivo keys use openrouter                     # switch active key (or just `aivo use openrouter`)
aivo keys cat | edit | rm <name>
aivo keys ping --all                         # health-check all keys
```

Any endpoint implementing a supported protocol can be saved.

## models

List models from the active provider. Cached for one hour.

```bash
aivo models
aivo models --refresh                        # bypass cache
aivo models -s sonnet                        # filter by substring
aivo models --json | jq '.models[].id'
```

## chat

Interactive chat TUI, or one-shot `-x` mode for scripting and pipelines.

```bash
aivo chat                                    # full-screen TUI
aivo chat -m gpt-4o                          # pick a model (remembered per key)
aivo chat --attach README.md --attach screenshot.png

aivo -x "Summarize this repo"                # one-shot (shortcut for `aivo chat -x`)
git diff | aivo -x "Write a commit message"  # piped stdin appended as context
cat error.log | aivo -x                      # stdin alone becomes the prompt
aivo -x "hi" --json | jq -r '.choices[0].message.content'
```

Slash commands inside the TUI:

| Command | Description |
| ------- | ----------- |
| `/new` | Start a fresh chat |
| `/resume [query]` | Resume a saved chat from this directory |
| `/model [name]` | Switch the chat model |
| `/key [id\|name]` | Switch saved key |
| `/attach <path>` | Attach a text file or image |
| `/detach <n>` | Remove a queued attachment |
| `/help` · `/exit` | Help · Quit |
| `//message` | Send a literal leading slash |

## serve

Expose the active provider as a local OpenAI-compatible endpoint.

```bash
aivo serve                                   # http://127.0.0.1:24860
aivo serve -p 8080 --host 0.0.0.0
aivo serve --failover                        # retry across keys on 429/5xx
aivo serve --cors                            # enable CORS for browser clients
aivo serve --auth-token                      # require bearer token (auto-generated)
aivo serve --log /tmp/requests.jsonl
```

## alias

Short names for models or launch presets. Both share one namespace.

```bash
aivo alias                                   # list
aivo alias fast=claude-haiku-4-5             # model alias
aivo alias quick claude --key work -m fast --1m   # launch alias

aivo claude -m fast                          # use anywhere `-m` is accepted
aivo quick                                   # invoke launch alias directly
aivo quick -k personal                       # explicit flags override the preset

aivo alias rm fast                           # remove (works for both kinds)
```

Names that collide with built-in subcommands or tool names are rejected.

## logs

Query the local SQLite log database used by `chat`, `run`, and `serve`. Chat logs include turn content and token usage. `run` and `serve` log metadata only.

```bash
aivo logs                                    # newest first
aivo logs show <id>                          # one entry in detail
aivo logs status                             # counts, db size, path

aivo logs --by chat -n 5
aivo logs --by claude --errors
aivo logs -s "rate limit" --since 7d
aivo logs --by run --watch                   # live tail
```

## stats

Aggregates token counts from aivo chat, Claude Code, Codex, Gemini, OpenCode, and Pi by reading each tool's native data files.


```bash
aivo stats
aivo stats claude --since 7d                 # one tool, recent window
aivo stats -s openrouter -n                  # filter, exact numbers
aivo stats --top-sessions                    # heaviest native session files
aivo stats --json | jq '.totals.tokens'
```

## update

Update to the latest version. Delegates to Homebrew or npm when installed by those package managers.

```bash
aivo update
aivo update --force                          # force even if pkg-managed
aivo update --rollback                       # restore previous backup
```

## License

MIT
