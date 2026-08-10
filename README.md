[![aivo](https://getaivo.dev/banner.webp)](https://getaivo.dev)

Aivo `/ˈeɪ.voʊ/` lets you use Claude Code, Codex, Gemini, OpenCode, Pi, Grok, and other coding agents with the model and provider you choose. It also includes Aivo Code, a built-in terminal coding agent.

![CI](https://github.com/yuanchuan/aivo/actions/workflows/ci.yml/badge.svg)
![Release](https://img.shields.io/github/v/tag/yuanchuan/aivo?label=release&color=brightgreen)
![MSRV](https://img.shields.io/badge/rustc-1.97+-orange.svg)
![Binary size](https://img.shields.io/badge/binary-%3C10MB-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

[**Documentation**](https://getaivo.dev) · [Install](#install) · [Quick start](#quick-start) · [Aivo Code](#aivo-code)

---

## Why Aivo?

- Run popular coding agents through one CLI.
- Use any model, from hosted APIs and AI gateways to local GGUF files.
- Manage API keys locally with encrypted storage.
- Work directly in the terminal with Aivo Code.
- Keep a unified view of sessions, logs, and token usage.

## Install

**macOS and Linux**

```bash
curl -fsSL https://getaivo.dev/install.sh | bash
```

**Homebrew**

```bash
brew install yuanchuan/tap/aivo
```

**Windows PowerShell**

```powershell
irm https://getaivo.dev/install.ps1 | iex
```

## Quick start

The built-in `aivo/starter` provider works on first run, so you can try Aivo without an API key:

```bash
aivo "tell me a short story"
aivo claude
```

Add a provider to access more models:

```bash
aivo keys add
aivo claude --model moonshotai/kimi-k2.5
```

Launch other supported agents the same way:

```bash
aivo codex
aivo gemini
aivo opencode
aivo pi
aivo grok
```

## Aivo Code

Aivo Code is the built-in terminal coding agent, with session tools, skills, MCP servers, and support for remote or local models.

[![Aivo Code](https://getaivo.dev/aivo-chat.webp)](https://getaivo.dev)

```bash
aivo code
aivo code vercel::zai/glm-5.2
aivo code hf:lmstudio-community/Olmo-3-1025-7B-GGUF
aivo code -e "今天成都的天气"
```

## Learn more

See [getaivo.dev](https://getaivo.dev) for commands, providers, models, configuration, plugins, and advanced usage.

## License

[MIT](LICENSE)
