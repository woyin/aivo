[![aivo](https://getaivo.dev/banner.webp)](https://getaivo.dev)

> Aivo is a command-line tool that connects your existing coding agent to the model you want.
> It includes starter models to get you going — no API key required.


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
aivo "tell me a short story"
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
- `codex-app` [Codex.app](https://github.com/openai/codex) desktop (macOS only)
- `gemini` [Gemini CLI](https://github.com/google-gemini/gemini-cli)
- `opencode` [OpenCode](https://github.com/anomalyco/opencode)
- `pi` [Pi Coding Agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent)

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

### Export & import

Move keys between machines via a password-encrypted file:

```bash
aivo keys export ~/keys.aivo     # prompts for password
aivo keys import ~/keys.aivo     # same password on the other machine
aivo keys import https://example.com/keys.aivo   # or from a URL

# non-interactive with password on stdin
aivo keys export ~/keys.aivo --password-stdin <<< "my secret password"
```

## models

List models from the active provider. Cached for one hour.

```bash
aivo models
aivo models --refresh                        # bypass cache
aivo models -s sonnet                        # filter by substring
aivo models --json | jq '.models[].id'
```

## chat

Interactive chat TUI, or one-shot `-p` mode for scripting and pipelines.

```bash
aivo chat                                    # full-screen TUI
aivo chat -m gpt-4o                          # pick a model (remembered per key)
aivo chat --attach README.md --attach screenshot.png

aivo "Summarize this repo"                   # bare quoted prompt → one-shot chat
aivo -p "Summarize this repo"                # same, via the explicit flag
git diff | aivo -p "Write a commit message"  # piped stdin appended as context
cat error.log | aivo -p                      # stdin alone becomes the prompt
aivo -p "hi" --json | jq -r '.choices[0].message.content'
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

## hf

Run open-weight GGUF models locally, it fetches and caches them from HuggingFace repositories.

```bash
aivo chat hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo https://huggingface.co/allenai/Olmo-3-1025-7B              # full URL also works
aivo chat hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M        # pin a specific quant
aivo pi -m hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF                   # any tool that accepts -m
```

The `hf:` form is accepted anywhere a model is — `-m`, chat's positional `REF`, and as a bare top-level arg (which rewrites to `aivo chat hf:...`). Full HuggingFace URLs (`https://huggingface.co/...`) work the same way.

Manage the cached GGUF files (under `~/.config/aivo/cache/huggingface`):

```bash
aivo hf                                       # list cached repos
aivo hf list --verbose                        # show individual files
aivo hf pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo hf rm <repo> --quant Q5_K_M              # delete one quant
aivo hf rm <repo> --all -y                    # delete whole repo
aivo hf clean -y                              # wipe everything
```

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

Unified activity feed across aivo's own events (`chat`, `run`, `serve`) and native CLI sessions (`claude`, `codex`, `gemini`, `pi`, `opencode`). Defaults to the current project's cwd; use `-a` for every project.

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

## stats

Aggregates token counts from aivo chat, Claude Code, Codex, Gemini, OpenCode, and Pi by reading each tool's native data files.


```bash
aivo stats
aivo stats --by claude --since 7d            # one tool, recent window
aivo stats --by omp                          # a coding-agent plugin
aivo stats -s openrouter -n                  # filter, exact numbers
aivo stats --json | jq '.totals.tokens'
```

## update

Update to the latest version. Delegates to Homebrew or npm when installed by those package managers.

```bash
aivo update
aivo update --force                          # force even if pkg-managed
aivo update --rollback                       # restore previous backup
```

## plugins

Add a top-level command — a standalone `aivo-<name>` executable, in any language, that aivo runs
as `aivo <name>`. Plugins run with your privileges; install only ones you trust. Full contract:
[docs/PLUGIN-PROTOCOL.md](docs/PLUGIN-PROTOCOL.md).

```bash
aivo plugins install ./aivo-amp              # local file or http(s) URL
aivo plugins install github:owner/aivo-amp   # GitHub release (picks your OS/arch asset)
aivo plugins install npm:aivo-foo            # npm package (node shim)
aivo plugins install cargo:aivo-bar          # cargo install from crates.io
aivo amp --help                              # runs the sibling aivo-amp
aivo plugins list                            # installed, with version/roles/caps
aivo plugins update amp                      # re-fetch / re-resolve the recorded source
aivo plugins remove amp
```

## License

MIT
