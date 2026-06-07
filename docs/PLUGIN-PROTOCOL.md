# aivo plugin protocol

**Protocol version `1`.**

A plugin is a standalone executable `aivo-<name>`, in **any language**, that aivo runs as
`aivo <name>`. The only hard requirement is that the file exists and is executable; metadata, the
key handoff, and transcript sharing are all opt-in via a [manifest](#manifest).

```bash
aivo plugins install ./aivo-hello   # add (path | URL | github: | npm: | cargo:)
aivo hello --flag                   # runs `aivo-hello --flag`
aivo plugins list                   # installed plugins + version / type / caps
```

## Dispatch & discovery

`aivo <name>` and `aivo run <name>` resolve to `aivo-<name>` **only when aivo doesn't own `name`** —
built-ins, known tools, chat refs (`hf:`, `http(s)://`), and user bundles always win. The check
runs before clap parses argv.

Lookup is **first-match-wins** across three directories, in order:

1. `~/.config/aivo/plugins/` — managed by `aivo plugins install`
2. the directory of the `aivo` binary
3. each `$PATH` entry

On Windows the file is `aivo-<name>.exe` (or `.cmd`); the extension is stripped to derive `<name>`.

### Environment

aivo spawns the plugin with stdio inherited (spawn-and-wait, not `exec`, so aivo's own cleanup still
runs) and sets:

| Var | When | Value |
|---|---|---|
| `AIVO_CONFIG_DIR` | always | aivo's config dir. The key store is `<dir>/config.json`, **AES-256-GCM encrypted** — read keys through the [endpoint handoff](#endpoint-handoff), never this file. |
| `AIVO_DEBUG_LOG` | `--debug` in argv | log path (`--debug=<path>` overrides the default) |
| `AIVO_ENDPOINT_*`, `AIVO_KEY_MODEL` | `endpoint` granted | see [Endpoint handoff](#endpoint-handoff) |

Secrets are passed **as env only, never argv**, so they never appear in `ps`.

## Manifest

Run as `aivo-<name> --aivo-manifest`, a conforming plugin prints **one JSON object** to stdout and
exits `0` — no network, no child processes. aivo sets `AIVO_MANIFEST_PROBE=1` for this call.

**Probe contract:**

- timeout **2s** (the process is killed on timeout; stdin/stderr are `/dev/null`, stdout is captured)
- exit code must be `0`
- stdout is parsed as JSON — the whole trimmed output first, else the last non-empty line (tolerates
  leading log noise)
- honored only if it parses **and** `protocol == "1"`; any other `protocol` is treated as *no
  manifest* (the plugin still runs, its declared metadata ignored)
- a `name` that differs from the installed name warns but is non-fatal — the installed name wins

A failed probe is never an install error; the plugin is just recorded without metadata.

```jsonc
{
  "name":         "amp",            // required — must match the installed name
  "version":      "0.1.0",          // required
  "protocol":     "1",              // required — must equal the host's version
  "description":  "…",              // optional
  "type":         "coding-agent",   // optional — see Type
  "roles":        ["subcommand"],   // optional — "subcommand" today ("hook" reserved)
  "documents_aivo_flags": true,     // optional — my --help already lists -k/-m/--debug; skip aivo's banner
  "capabilities": ["endpoint"],     // optional — requested; granted on consent
  "transcripts":  { "format": "pi", "dir": "~/.omp/agent/sessions" },                 // optional
  "requires":     [ { "bin": "omp", "install": "curl -fsSL https://omp.sh/install | sh" } ], // optional
  "homepage":     "…"               // optional
}
```

Unknown fields are ignored (forward-compatible). A manifest carries **three orthogonal axes**:
`roles` (how aivo runs it), `type` (what it is), and `capabilities` (what it's granted). Any
combination is valid.

A complete plugin, in shell:

```sh
#!/bin/sh
if [ "$1" = "--aivo-manifest" ]; then
  printf '%s\n' '{"name":"hello","version":"1.0.0","protocol":"1","roles":["subcommand"]}'
  exit 0
fi
echo "hello: $*"
```

### Type

`type` is an **open vocabulary** for what the plugin *is* (`coding-agent`, `media`,
`code-review`, …); aivo validates nothing. Today only **`coding-agent`** carries host behavior:

- aivo owns `-k`/`--key` and `-m`/`--model` in the plugin's argv, strips them before launch (so they
  don't reach the wrapped tool), opens the key/model picker on a bare flag, and resolves a concrete
  model — passed as `AIVO_KEY_MODEL`.
- the launch is wrapped in the same run accounting native tools get: a `started`/`finished` row pair
  in `aivo logs` / `aivo stats` (launch count, duration, exit code).

Other types are recorded and shown but otherwise inert. (Token accounting at the endpoint is
independent of `type` — see [Endpoint handoff](#endpoint-handoff).)

## Stats

A plugin that declares the **`stats`** capability implements a second probe:

```
aivo-<name> --aivo-stats --json
```

It reads its **own** session/usage data — every agent stores tokens differently, in a different
folder — and prints **one** `aivo.stats/v1` JSON object: a **timestamped, per-session** list of
per-model token usage. **The plugin only provides data; aivo owns all filtering.** aivo applies
`--since` windowing, model-name normalization, and aggregation host-side — consistently with native
tools — so the plugin never reimplements them. aivo pulls this on demand for `aivo stats --by <name>`
and the `By tool` overview, **preferring it over aivo's own endpoint accounting** (the plugin's data
is the complete, authoritative view of that agent). Best-effort and capability-gated: a missing flag,
timeout (5s), non-zero exit, or wrong schema → aivo falls back to its endpoint token accounting, then
to launch counts. `stats` is **disclosure-only** (the plugin only reads its own data) — no consent
grant, like `--aivo-manifest`.

```jsonc
{
  "schema":   "aivo.stats/v1",
  "tool":     "amp",
  "source":   "aivo-routed amp threads (~/.config/aivo/amp-threads)",  // shown as provenance
  "sessions": [
    {
      "ts":     "2026-06-08T01:23:45Z",   // RFC3339; omit if you can't place the session in time
      "models": [
        { "name": "deepseek-v4-flash", "input_tokens": 9, "output_tokens": 410,
          "cache_read_tokens": 0, "cache_write_tokens": 0 }
      ]
    }
  ]
}
```

Emit **one entry per session** with its `ts` so aivo can window it (`--since`); a session with no
`ts` is counted in lifetime views but dropped under `--since`. All token fields default to `0` — omit
dimensions you don't track; sum a session's turns into its per-model entries. The `source` string is
shown to the user, so make it descriptive — and scope the data to **aivo-routed** runs where you can,
since these totals sit under aivo's own usage stats.

## Help

`aivo <name> --help` / `-h` is a **pure passthrough**: aivo never resolves a key, opens a picker,
stands up an endpoint, or fails on auth to print help. aivo forwards `--help` verbatim, and for a
top-level request first prints a uniform banner documenting the flags it intercepts
(`-k`/`-m`/`--debug` for a `coding-agent`) — **unless** the plugin sets
`documents_aivo_flags: true`, in which case the banner is skipped (its own help already covers them).
Sub-help (`aivo <name> <sub> --help`) never gets the banner.

A conforming plugin **must answer `-h`/`--help` standalone** — no endpoint, no resolved model, no
network — exactly like [`--aivo-manifest`](#manifest); aivo hands no `AIVO_ENDPOINT_*` on this path.

What the plugin prints is its **own concise, aivo-aware help** (what `aivo <name>` does, the
`-k`/`-m`/`--debug` it honors, its subcommands) — **not** the wrapped tool's full `--help`, which is
long and reachable directly as `<tool> --help`. A plugin that self-documents this way sets
`documents_aivo_flags: true` so aivo omits its banner; one that prints nothing aivo-aware leaves it
off and lets aivo's banner stand in.

## Capabilities & consent

Capability vocabulary: `endpoint`, `config-read`, `config-write`, `spawn`, `stats`, `hook:<event>`.
**`endpoint` is the only grantable capability in v1**; the rest are disclosure/reserved — parsed and
stored verbatim, never granted or injected. `stats` opts the plugin into the
[`--aivo-stats`](#stats) usage probe (read-only, no grant).

`endpoint` hands the plugin real power, so it is consent-gated:

| Install kind | When consent is asked |
|---|---|
| local path | at install (manifest is probed then) |
| `npm:` / `github:` / URL / `cargo:` | on **first dispatch** — probed lazily, then prompted once, and the result persisted |

Decline → the plugin still runs, just without the handoff. The approved set is stored as
`granted_caps` in the [registry](#registry) and is the **only** thing aivo injects; an update never
silently escalates to a newly-requested grantable cap (it re-asks). `aivo plugins install --trust`
grants the requested grantable caps without prompting — local-path installs only (the only form
whose manifest is known at install time).

## Endpoint handoff

A granted plugin gets a **per-launch loopback proxy** bound to one resolved key. aivo picks the
active key (or one named by `-k`/`--key <id>` in the plugin's argv) and injects:

| Var | Value |
|---|---|
| `AIVO_ENDPOINT_URL` | `http://127.0.0.1:<port>/v1` (OS-assigned loopback port) |
| `AIVO_ENDPOINT_TOKEN` | a random per-launch bearer token |
| `AIVO_KEY_MODEL` | the resolved model, for `coding-agent` plugins |

Point the client's base URL at `AIVO_ENDPOINT_URL` and send `Authorization: Bearer
$AIVO_ENDPOINT_TOKEN` (the proxy also accepts `x-api-key`; missing/wrong → `401`). **The upstream
secret never leaves aivo** — there is no raw-key env. The proxy starts at launch and is torn down on
exit.

### Inbound

The endpoint is a **protocol bridge**, not a passthrough. It accepts the two OpenAI **inbound**
protocols — **Chat Completions** and the **Responses API** — and translates each to whatever protocol
the resolved key's upstream speaks: OpenAI Chat Completions, OpenAI Responses, **Anthropic Messages**,
or **Gemini** (see [Engine selection](#engine-selection)). It is **format-transparent**, routing on
the request *body shape* rather than the path: a Chat Completions request returns Chat Completions, a
Responses-API request (`input` array) returns Responses-API, and the upstream protocol, cross-family
fallback, and any `/responses` escalation all stay hidden from the client. Anthropic `/v1/messages`
and Gemini `generateContent` are **not** accepted inbound.

The **portable surface**, served identically by both engines — target this:

| Route | Method | Notes |
|---|---|---|
| `/v1/models`, `/models` | `GET` | twin list: OpenAI `data` **and** codex `models` arrays, so strict Codex clients read the same endpoint |
| `/v1/chat/completions` | `POST` | the safe lowest common denominator across every handoff-able key |
| `/v1/responses`, `/responses` | `POST` | |

Outside that set the two engines diverge:

- **serve engine** (Anthropic / Gemini / Ollama) — a fixed route table that also serves
  `POST /v1/embeddings` and `GET /health`. Wrong method on a POST route → `405`; **any other path →
  `404`**.
- **responses engine** (OpenAI-protocol REST + Copilot) — also accepts the bare `/chat/completions`;
  **any other path is forwarded verbatim to the upstream** (a transparent reverse proxy), so e.g.
  `/v1/embeddings` works only if the upstream itself provides it.

### Engine selection

aivo translates the request to the key's wire protocol and, on rejection, **falls back across wire
families** (chat-completions ↔ Anthropic-messages ↔ Gemini). Three engines back the proxy:

| Key | Engine | Notes |
|---|---|---|
| OpenAI-protocol REST (incl. free starter) + Copilot | responses-capable router | the engine native codex uses. **Escalates to upstream `/v1/responses`** when the model requires it (gpt-5.x reasoning + tools) — even for a **Chat Completions** client, converting the reply (streaming included) back to Chat Completions — else uses chat. Learns the working route per `(plugin, key, model)` and persists it to the key config on exit. Copilot's token exchange runs inside this router |
| Anthropic / Gemini REST | `aivo serve` | wire-family cascade |
| Ollama | `aivo serve` | |
| Cursor, for `type: coding-agent` plugins | Cursor ACP router | translates local HTTP requests into `cursor-agent acp` prompts. Requires `cursor-agent` installed and the selected Cursor key authenticated. Generic plugin types cannot use this handoff |
| OAuth (Claude/Codex/Gemini) | — | **not handoff-able** — runs bare (below) |

Set `"stream": true` and the response is streamed back incrementally as SSE — including when the
request escalated to `/v1/responses` upstream (that upstream stream is relayed chunk-by-chunk, not
buffered into one blob). A request without `stream` is buffered. The proxy binds exactly one key —
**no cross-key failover** (that's an `aivo serve` feature).

For the REST engines, token usage is recorded against the key only for **buffered
(non-streaming) 2xx** responses; **streaming responses are uncounted** (no body to read usage from
mid-stream). Cursor-router usage is not token-accounted by the endpoint. Under
`aivo <plugin> --debug` the proxy logs each proxied request to `AIVO_DEBUG_LOG`.

### Not handoff-able

OAuth keys (Claude/Codex/Gemini) live only inside their native agent — injected as native
credentials, with no provider REST endpoint to proxy. For OAuth, aivo prints a one-line note and the
plugin runs on its own auth. No active key → same.

Cursor is handoff-able only for `type: coding-agent` plugins. Those launches use aivo's
bearer-gated Cursor ACP router instead of handing over Cursor credentials or spawning
`cursor-agent` from plugin code. Other plugin types still run on their own auth when a Cursor key is
selected.

## Transcripts (sharing)

aivo records *that* a plugin ran (`aivo logs`/`aivo stats`), never the conversation — that lives in
the agent's own session store. A plugin whose sessions use a format aivo already reads can opt in:

```jsonc
"transcripts": { "format": "pi", "dir": "~/.omp/agent/sessions" }
```

`format` ∈ `pi` | `codex` | `gemini` | `opencode`. For `pi`/`codex`/`gemini`, `dir` is the sessions
root (leading `~` expanded); for `opencode`, `dir` is the path to the `opencode.db` SQLite file.
`aivo share <run-id>` points the matching built-in reader at `dir`, matches the run by cwd + time,
and emits the transcript. An undeclared or unknown format stays un-shareable — there's no generic
reader.

## Requirements

A plugin that wraps another binary declares it, getting the **same install UX as a native tool**
without the core knowing the tool:

```jsonc
"requires": [ { "bin": "omp", "install": "curl -fsSL https://omp.sh/install | sh" } ]
```

At install aivo checks each `bin` on `$PATH`; if one is missing it shows the **plugin-authored**
`install` command and (interactively) runs it on consent — never a command aivo invented. `install`
omitted → aivo only reports the gap; `aivo plugins list` marks it `(missing)`. A plugin should still
check its own dependency at run time as a fallback.

## Registry

Provenance the bare binary can't carry lives in `~/.config/aivo/plugins/.registry.json` (a dotfile,
skipped by discovery):

```jsonc
{
  "version": 1,
  "plugins": {
    "amp": {
      "source":       "/abs/path-or-url",       // required — for `update`
      "checksum":     "sha256:9f86d0…",         // of the installed bytes at install time
      "manifest":     { /* … */ },              // cached probe; absent if not self-described
      "installed_at": "2026-06-04T05:48:30Z",   // RFC3339
      "granted_caps": ["endpoint"]              // consented caps; omitted when none
    }
  }
}
```

Reads never touch disk. Writes (install/update/remove) are atomic and `0600`. The older source-only
`.sources.json` is migrated automatically; a corrupt registry is moved aside to
`.registry.json.corrupt`, not silently wiped. Verifying the checksum on every run, and signing, are
future work.

## Install sources

`aivo plugins install <source>` accepts:

| Form | Example | Notes |
|---|---|---|
| local path | `./aivo-foo` | used as-is — **the only form probed for a manifest at install** |
| direct URL | `https://host/dl/aivo-foo` | downloads the binary, or unpacks a `.tar.gz`/`.zip` |
| GitHub | `github:owner/repo[@tag]`, `gh:owner/repo`, `https://github.com/owner/repo` | resolves the release, picks the asset matching your OS+arch |
| npm | `npm:pkg`, `npm:@scope/pkg@1.2.0` | fetches the tarball + writes a `node` shim — needs Node.js |
| cargo | `cargo:crate[@version]` | `cargo install` — builds from source, needs a Rust toolchain |

An asset/archive must contain an executable named `aivo-<name>` (or a single executable).
`aivo plugins update` re-runs the recorded source: a bare `github:owner/repo` re-resolves to the
latest release; a pinned `@tag`/`@version` stays put. `AIVO_GITHUB_API` / `AIVO_NPM_REGISTRY`
override the endpoints (GitHub Enterprise, private mirrors). Remote installs are recorded **without**
a manifest — the install-time probe is local-only.

## Security

A plugin runs with **your full privileges** — aivo can't sandbox it. Treat `aivo plugins install`
like `npm i -g`: install only what you trust. aivo's guarantees are **consent + provenance**
(capability consent before any secret handoff, the `sha256` pin), not containment. The grant scopes a
plugin to only the endpoint env its caps cover, but once running it can do anything you can.

## Reserved for later

Specified so the contract stays stable; not yet implemented:

- **`hook` role** — observe/transform launches and routing over JSON-RPC (the `hook:<event>` caps and
  the manifest `hooks` array)
- **`config-read` / `config-write`** — scoped config access
- **streaming token accounting** at the endpoint (today only buffered responses are counted)
- **transcript export** — a subcommand so a plugin with a *novel* session format (one aivo can't
  read) can emit its own transcript for `aivo share`
- **signing + a discovery index**

OAuth keys stay **deliberately out of scope** for the endpoint (no provider REST endpoint to proxy);
serving them would mean puppeting the official CLI as a headless backend — a separate per-CLI
component with its own consent, not a tweak to this endpoint.

## Commands

```bash
aivo plugins list
aivo plugins install <source> [--name N] [--force] [--trust]   # path | url | github:/gh: | npm: | cargo:
aivo plugins update [name]    # re-fetch / re-resolve from the recorded source
aivo plugins remove <name>
```
