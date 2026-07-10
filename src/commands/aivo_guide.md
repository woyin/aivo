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
| `aivo <prompt>` | `aivo code -p <prompt>` | a bare string / piped stdin runs a one-shot prompt |
| `aivo hf:… ` / `aivo https://…` | `aivo code <ref>` | open code on a local HuggingFace GGUF |
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

## The built-in agent — `aivo code`

`aivo code` is aivo's own coding agent in your terminal (full-screen TUI). It reads/edits your
project and runs shell commands, prompting for risky actions.

```
aivo code                     # interactive agent TUI
aivo code "<text>"            # TUI with the text sent as the first message
-m, --model <model>           pick a model (remembered per key; bare = picker)
-k, --key <id|name>           API key by id/name (bare = picker)
-p, --prompt [prompt]         one prompt, plain reply, exit (no tools)
-e, --exec   [prompt]         one prompt, run the full agent (tools), exit
--max-steps <N>               max -e agent steps (0 disables)
--max-output-tokens <N>       max -e output tokens (0 disables)
--max-cost <USD>              max estimated -e spend (needs known model pricing)
--add-dir <dir>               extra writable workspace root (repeatable) — writes
                              there skip the out-of-workspace confirm and stay
                              inside the sandbox confinement
-r, --refresh                 refresh the model list (skip cache)
--resume [last|id]            resume a saved session (TUI and -e; -e runs persist too)
--share                       share this session live (needs `aivo login`)
-c, --context[=<id>]          inject one past AI CLI session as context (bare = picker)
--attach <path>               attach a file or image
--json                        raw provider JSON (with -p)
--output-format <fmt>         -e output: text (default), json (one final result
                              document), or stream-json (one event per line)
--max-context <size>          override the context window (e.g. 200k)
--dry-run                     show the resolved key/model/endpoint, don't connect
--auto-approve                start in auto-approve mode: everything runs without
                              a prompt, remote mutations (deploy/publish/PR merge)
                              included; catastrophic commands still confirm. With
                              -e this is how unattended runs get remote rights.
```

```bash
aivo -p "Summarize this repo"                # bare string → one-shot plain reply
git diff | aivo -p "Write a commit message"  # piped stdin appended as context
aivo code -e "make the failing test pass"    # one-shot agent run
aivo code -e "fix lint" --max-steps 50       # override headless agent limits
aivo code -e "audit deps" --output-format json | jq .answer   # scriptable result
aivo code -e "now fix what you found" --resume last           # continue that run
```

Headless runs verify by default: when the agent edited files and declares done, the project's
validator (`run_tests.sh`, `make test`, `npm test`, `cargo test`, …) runs and failures are fed
back for a fix. `AIVO_AGENT_SELF_CORRECT=0` opts out.

### Inside the code TUI

Type `/help` for the full list. Slash commands:

- Session: `/new`, `/resume [query]`, `/rewind` (undo edits), `/copy [n]`, `/config`, `/share [stop]`, `/help`, `/exit`
- Model & key: `/model [name]`, `/key [id|name]`, `/effort [level]`
- Context: `/attach <path>`, `/detach <n>`, `/compact [fast]`
- Skills & tools: `/skills`, `/create-skill`, `/mcp` (CLI twins: `aivo code skills`, `aivo code mcp`)
- Autonomous: `/plan <objective>`, `/goal <objective>`

Other input: `!cmd` runs a local shell command; `//` / `!!` escape to literal text.

Keys: `Enter` send · `Ctrl+J` newline · `Tab` complete · `Ctrl+V` paste text/image ·
`Ctrl+X Ctrl+E` edit draft in $EDITOR · `Shift+Tab` cycle mode (normal/auto-approve/plan/review) · `Ctrl+R` resume ·
`Ctrl+O` pager for a `!cmd` · `Esc` cancel/close · `Ctrl+C` twice to exit.

`/config` toggles: Thinking, Auto-approve tools, aivo web search, Agent tools (off = plain chat,
no tools). The agent can also change the live model/effort itself when you ask (it calls its
`switch_model` / `set_effort` tools); a key change it hands back to you via `/key`.

### Skills & MCP servers — `aivo code skills`, `aivo code mcp`

CLI twins of `/skills` and `/mcp` for scripts and dotfiles; toggles are shared with the TUI.

**Skills** are folders holding a `SKILL.md` (portable Agent Skills format) that the agent loads on
demand. Discovered from the repo (`.agents/skills`, `.aivo/skills`, `.claude/skills`) and user
dirs (`~/.agents/skills`, `~/.config/aivo/skills`, `~/.claude/skills`) — an existing Claude Code
skill library works unchanged.

```bash
aivo code skills                    # list discovered skills (scope + on/off)
aivo code skills cat <name>         # one skill in full: state, source, instructions
aivo code skills install <source>   # github:owner/repo[@ref], github.com /tree/… URL, or local path
aivo code skills install <source> <name>    # just one skill from a multi-skill source
aivo code skills install <source> --all     # every skill found (existing names skip)
aivo code skills install -p <source>        # into the repo ./.agents/skills (project scope)
aivo code skills enable <name>      # enable/disable for the agent (aliases: on/off)
aivo code skills rm <name>          # remove a user-scope skill (project skills: delete the folder)
```

**MCP servers** (stdio or Streamable HTTP, with OAuth) give the agent external tools. User scope
lives in `~/.config/aivo/mcp.json`; a repo `.mcp.json` adds project scope. `${VAR}` /
`${VAR:-default}` in a config expand from the environment at connect time.

```bash
aivo code mcp                       # list servers (scope + on/off + per-tool opt-outs)
aivo code mcp cat <name>            # one server: transport, state, raw JSON config
aivo code mcp add npx -y <pkg>      # stdio server (name derived from the command)
aivo code mcp add https://…         # remote Streamable HTTP server
aivo code mcp add '<json>'          # paste an mcpServers JSON block
aivo code mcp add -p …              # into the repo ./.mcp.json (project scope)
aivo code mcp enable <name>         # enable/disable for the agent (aliases: on/off)
aivo code mcp rm [-p] <name>        # remove a server (-p: from ./.mcp.json)
aivo code mcp import [tool] [name]  # copy servers from claude/cursor/gemini/copilot/amp configs
```

Per-tool toggles within a connected server live in the TUI (`/mcp`, `Ctrl+T`).

### Extension packs — `aivo code packs`

One installable unit bundling skills, sub-agent profiles, hooks, and MCP servers — the
Claude Code plugin layout (`.claude-plugin/plugin.json` + `skills/` + `agents/` +
`hooks/hooks.json` + `.mcp.json`), so existing Claude Code plugins install unchanged.
Installed under `~/.config/aivo/packs/<name>`; components join normal discovery at the
lowest precedence (project and user files shadow them). Installing is the consent
moment: `add` lists everything the pack ships — hooks and stdio MCP servers execute
code — and asks before copying (`-y` skips; required off a TTY).

```bash
aivo code packs                     # list installed packs and what each ships
aivo code packs add github:o/pack   # or a github.com (tree) URL, or a local path
aivo code packs rm <name>           # remove the pack and everything it shipped
```

### Hooks — `~/.config/aivo/hooks.json`

User-authored shell commands the agent runs at lifecycle points (config shape mirrors Claude
Code's `hooks` block; user-scope only — repo-provided hook commands would be code execution
on open). Each hook receives a JSON payload on stdin; exit `0` passes, exit `2` blocks with
stderr as the reason; other failures and timeouts are ignored (fail-open — the built-in
permission tiers remain the security floor).

- `PreToolUse` — before a tool call; exit 2 vetoes it (`matcher`: `*` or `run_bash|write_file`)
- `PostToolUse` — after a tool call; stdout (or exit-2 stderr) is folded into the tool result
- `Stop` — when the agent declares done; exit 2 sends stderr back as guidance and continues

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "run_bash",
        "hooks": [{ "command": "my-guard.sh", "timeout": 10 }] }
    ],
    "Stop": [
      { "hooks": [{ "command": "check-todos.sh" }] }
    ]
  }
}
```

## Local models — `hf:` and `aivo hf`

Run open-weight GGUF models locally — fetched/cached from HuggingFace and served by a local
`llama-server`, zero setup. The `hf:` form works anywhere a model is accepted (`-m`, code's
positional ref, or a bare top-level arg); full `https://huggingface.co/…` URLs work too.

```bash
aivo code hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
aivo https://huggingface.co/allenai/Olmo-3-1025-7B
aivo code hf:bartowski/Llama-3.2-3B-Instruct-GGUF:Q5_K_M   # pin a quant
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

Unified session list across aivo code + native CLI sessions (claude, codex, gemini, pi,
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

Aggregates token/request counts from aivo code and every launched agent by reading each tool's
native data files.

```
aivo stats                     # totals + top models
--by <name>                    one tool or plugin (claude, code, omp, …)
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
- `AIVO_AGENT_LSP=0` — disable LSP diagnostics-after-edit in the code agent (default on)
- `AIVO_AGENT_SELF_CORRECT` — post-edit verification: default on for `-e` (`0` disables);
  `1` also enables it in interactive turns
- `AIVO_NO_UPDATE_NOTICE=1` (or `CI`) — suppress the update-available notice
- `NO_PROXY=127.0.0.1,localhost` — set when an `http_proxy` is configured, so `aivo serve` /
  local llama endpoints aren't proxied

## Exit codes

`0` success · `1` user error · `2` network · `3` auth.

Full docs: https://getaivo.dev
