//! Wires `aivo run --as <name>` into Claude/Codex by registering an aivo MCP
//! server in each tool. **No persistent config is ever written.** Everything
//! is ephemeral and scoped to this one launch.
//!
//! ## Per-tool strategy
//!
//! - **Claude**: writes a JSON config to a `tempfile::NamedTempFile` and
//!   prepends `--mcp-config <path>`. The file is auto-deleted when the
//!   returned `ShareCleanup` drops at process exit.
//!
//! - **Codex**: uses the `-c / --config key=value` override flag (TOML
//!   syntax) to inject the aivo MCP server into codex's config at launch
//!   time. `~/.codex/config.toml` is **not** modified — codex reads it
//!   normally, then applies our overrides on top. Verified against
//!   codex-cli 0.120.0.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tempfile::NamedTempFile;

use crate::services::ai_launcher::AIToolType;
use crate::services::nickname_registry;
use crate::style;

/// Codex startup timeout (seconds) we register for the aivo MCP entry.
/// Read: `mcp-serve` boots in ~100ms on a warm build; 10s leaves headroom
/// for cold starts and slow disks without turning user-facing failures
/// (e.g. a missing binary) into a long wait.
const CODEX_STARTUP_TIMEOUT_SEC: i64 = 10;

/// RAII guard holding ephemeral resources for the child tool's lifetime.
/// Dropping deletes temp files and nickname registrations.
pub struct ShareCleanup {
    claude_mcp_config: Option<NamedTempFile>,
    _registry_guard: Option<nickname_registry::RegistryGuard>,
}

impl ShareCleanup {
    pub fn empty() -> Self {
        Self {
            claude_mcp_config: None,
            _registry_guard: None,
        }
    }

    pub fn set_registry_guard(&mut self, guard: nickname_registry::RegistryGuard) {
        self._registry_guard = Some(guard);
    }
}

impl Drop for ShareCleanup {
    fn drop(&mut self) {
        // `NamedTempFile`'s default drop silently swallows removal errors.
        // Call `close()` explicitly so a stuck temp config (permission
        // change, already-unlinked, etc.) is surfaced in debug builds —
        // release builds still stay quiet to avoid noisy user output.
        // (Sibling helper: `context_ingest::warn_unreadable_session` does
        // essentially the same debug-only path+error log; consolidate if a
        // third site appears.)
        if let Some(tempfile) = self.claude_mcp_config.take() {
            // `close()` consumes `tempfile`, so capture the path up front.
            let path = tempfile.path().to_path_buf();
            if let Err(err) = tempfile.close() {
                #[cfg(debug_assertions)]
                eprintln!(
                    "aivo: failed to remove temp MCP config {}: {err}",
                    path.display()
                );
                #[cfg(not(debug_assertions))]
                let _ = (path, err);
            }
        }
    }
}

/// If `share` is enabled, produce the args needed to expose aivo's MCP
/// server to the tool for this single launch. Returns `(new_args, cleanup)`.
/// The cleanup guard holds ephemeral resources (e.g. Claude's temp config
/// file) that must outlive the spawned tool.
pub async fn maybe_enable_share(
    tool: AIToolType,
    args: Vec<String>,
    cwd: &Path,
    nickname: &str,
) -> Result<(Vec<String>, ShareCleanup)> {
    let aivo_exe = resolve_aivo_exe()?;

    match tool {
        AIToolType::Claude => enable_share_claude(args, cwd, &aivo_exe, nickname),
        AIToolType::Codex => Ok((
            enable_share_codex(args, cwd, &aivo_exe, nickname),
            ShareCleanup::empty(),
        )),
        // Pi / Gemini / OpenCode can't be MCP clients from aivo (no
        // ephemeral injection path — see `session_transcript` module doc).
        // Their sessions are still *readable* by a peer claude/codex via the
        // nickname registered in `run.rs`, so we skip wiring but print an
        // accurate status line.
        AIToolType::Pi | AIToolType::Gemini | AIToolType::Opencode => {
            eprintln!(
                "  {} {}: nickname '{}' is registered; peer claude/codex can read this session via MCP. {} cannot call MCP servers from aivo (no ephemeral injection path).",
                style::arrow_symbol(),
                tool.as_str(),
                nickname,
                tool.as_str()
            );
            Ok((args, ShareCleanup::empty()))
        }
    }
}

/// Resolve the absolute, canonical path to the current aivo binary. Fails
/// fast if the binary is missing so we don't hand a stale pointer to the
/// child tool.
fn resolve_aivo_exe() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("cannot locate current aivo binary")?;
    let canonical = match std::fs::canonicalize(&exe) {
        Ok(c) => c,
        Err(_) => exe.clone(),
    };
    if !canonical.exists() {
        bail!(
            "aivo binary at {} is missing; refusing to enable --as",
            canonical.display()
        );
    }
    Ok(canonical)
}

/// Claude wiring: write an ephemeral `--mcp-config <file>` JSON.
fn enable_share_claude(
    args: Vec<String>,
    cwd: &Path,
    aivo_exe: &Path,
    nickname: &str,
) -> Result<(Vec<String>, ShareCleanup)> {
    let temp = NamedTempFile::with_prefix("aivo-mcp-")
        .context("failed to create temp MCP config for Claude")?;
    let canonical_cwd = canonicalize_cwd(cwd);
    let config = serde_json::json!({
        "mcpServers": {
            "aivo": {
                "command": aivo_exe.to_string_lossy(),
                "args": [
                    "mcp-serve",
                    "--cwd", canonical_cwd.to_string_lossy(),
                    "--nickname", nickname,
                    "--caller-cli", "claude",
                ],
            }
        }
    });
    serde_json::to_writer(temp.as_file(), &config)
        .context("failed to write temp MCP config for Claude")?;
    temp.as_file()
        .sync_all()
        .context("failed to flush temp MCP config")?;

    let mut new_args = Vec::with_capacity(args.len() + 2);
    new_args.push("--mcp-config".to_string());
    new_args.push(temp.path().to_string_lossy().to_string());
    new_args.extend(args);

    eprintln!(
        "  {} --as: injected ephemeral MCP config for Claude",
        style::arrow_symbol()
    );

    Ok((
        new_args,
        ShareCleanup {
            claude_mcp_config: Some(temp),
            _registry_guard: None,
        },
    ))
}

/// Codex wiring: prepend `-c mcp_servers.aivo.<field>=<value>` overrides so
/// codex picks up the aivo MCP entry at startup without touching
/// `~/.codex/config.toml`. Values are TOML-encoded per codex's `-c` spec.
fn enable_share_codex(
    args: Vec<String>,
    cwd: &Path,
    aivo_exe: &Path,
    nickname: &str,
) -> Vec<String> {
    let canonical_cwd = canonicalize_cwd(cwd);
    let aivo_str = aivo_exe.to_string_lossy().to_string();
    let cwd_str = canonical_cwd.to_string_lossy().to_string();

    // Three separate dotted-path overrides. Each -c value is parsed as a TOML
    // literal; strings need explicit quotes, arrays use TOML array syntax,
    // integers are bare.
    let overrides = [
        build_override("mcp_servers.aivo.command", &toml_string(&aivo_str)),
        build_override(
            "mcp_servers.aivo.args",
            &format!(
                "[{}, {}, {}, {}, {}, {}, {}]",
                toml_string("mcp-serve"),
                toml_string("--cwd"),
                toml_string(&cwd_str),
                toml_string("--nickname"),
                toml_string(nickname),
                toml_string("--caller-cli"),
                toml_string("codex"),
            ),
        ),
        build_override(
            "mcp_servers.aivo.startup_timeout_sec",
            &CODEX_STARTUP_TIMEOUT_SEC.to_string(),
        ),
    ];

    let mut new_args = Vec::with_capacity(args.len() + overrides.len() * 2);
    for (flag, value) in &overrides {
        new_args.push(flag.clone());
        new_args.push(value.clone());
    }
    new_args.extend(args);

    eprintln!(
        "  {} --as: injected MCP server via codex -c overrides (config.toml untouched)",
        style::arrow_symbol()
    );

    new_args
}

/// Build a `-c` override pair `("-c", "path=value")`. Single arg style
/// (`-c path=value`) works identically and is slightly easier to read in
/// process listings, so we keep the key and value together.
fn build_override(path: &str, toml_value: &str) -> (String, String) {
    ("-c".to_string(), format!("{path}={toml_value}"))
}

/// Encode a string as a TOML basic-string literal (double-quoted, with
/// escapes). Good enough for paths — no embedded newlines or control
/// chars expected here.
fn toml_string(s: &str) -> String {
    let escaped: String = s
        .chars()
        .flat_map(|c| match c {
            '\\' => vec!['\\', '\\'],
            '"' => vec!['\\', '"'],
            other => vec![other],
        })
        .collect();
    format!("\"{escaped}\"")
}

fn canonicalize_cwd(cwd: &Path) -> PathBuf {
    std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_string_escapes_quotes_and_backslashes() {
        assert_eq!(toml_string("/a/b/c"), "\"/a/b/c\"");
        assert_eq!(toml_string(r#"weird"path"#), r#""weird\"path""#);
        assert_eq!(toml_string(r"a\b"), r#""a\\b""#);
    }

    #[test]
    fn build_override_pairs_flag_with_dotted_key() {
        let (flag, value) = build_override("mcp_servers.aivo.command", "\"/bin/aivo\"");
        assert_eq!(flag, "-c");
        assert_eq!(value, "mcp_servers.aivo.command=\"/bin/aivo\"");
    }

    #[test]
    fn codex_share_injects_three_overrides_before_user_args() {
        let aivo = PathBuf::from("/opt/aivo");
        let cwd = PathBuf::from("/tmp");
        let user_args = vec!["-m".to_string(), "gpt-5".to_string()];
        let out = enable_share_codex(user_args.clone(), &cwd, &aivo, "editor");

        // 3 overrides × 2 tokens each = 6, plus 2 user args = 8 total.
        assert_eq!(out.len(), 8);
        // Overrides come first so they are parsed before user flags.
        assert_eq!(out[0], "-c");
        assert!(out[1].starts_with("mcp_servers.aivo.command="));
        assert_eq!(out[2], "-c");
        assert!(out[3].starts_with("mcp_servers.aivo.args="));
        // Verify --nickname and --caller-cli are in the args override.
        assert!(out[3].contains("--nickname"));
        assert!(out[3].contains("editor"));
        assert!(out[3].contains("--caller-cli"));
        assert!(out[3].contains("codex"));
        assert_eq!(out[4], "-c");
        assert!(out[5].starts_with("mcp_servers.aivo.startup_timeout_sec="));
        // User args preserved at the end.
        assert_eq!(out[6], "-m");
        assert_eq!(out[7], "gpt-5");
    }

    #[test]
    fn codex_share_embeds_canonical_cwd_and_aivo_path() {
        let aivo = PathBuf::from("/opt/aivo/bin/aivo");
        let cwd = PathBuf::from("/workspace/proj");
        let out = enable_share_codex(vec![], &cwd, &aivo, "reviewer");
        let joined = out.join(" ");
        assert!(joined.contains(r#"mcp_servers.aivo.command="/opt/aivo/bin/aivo""#));
        assert!(joined.contains(r#""mcp-serve""#));
        assert!(joined.contains(r#""--cwd""#));
        assert!(joined.contains(r#""--nickname""#));
        assert!(joined.contains(r#""reviewer""#));
        assert!(joined.contains(r#""--caller-cli""#));
        assert!(joined.contains("mcp_servers.aivo.startup_timeout_sec=10"));
    }

    #[tokio::test]
    async fn claude_share_prepends_mcp_config_flag() {
        let tmp_exe = tempfile::NamedTempFile::with_prefix("fake-aivo").unwrap();
        std::fs::write(tmp_exe.path(), b"#!/bin/sh\n").unwrap();
        let aivo = tmp_exe.path().to_path_buf();
        let cwd = tempfile::tempdir().unwrap();
        let (args, _cleanup) = enable_share_claude(
            vec!["--model".into(), "opus".into()],
            cwd.path(),
            &aivo,
            "planner",
        )
        .unwrap();
        assert_eq!(args[0], "--mcp-config");
        // The generated temp file is the second token and should be readable JSON.
        let written = std::fs::read_to_string(&args[1]).unwrap();
        assert!(written.contains("\"mcpServers\""));
        assert!(written.contains("\"mcp-serve\""));
        // Verify --nickname and --caller-cli are in the MCP config.
        assert!(written.contains("--nickname"));
        assert!(written.contains("planner"));
        assert!(written.contains("--caller-cli"));
        assert!(written.contains("claude"));
        assert_eq!(args[2], "--model");
        assert_eq!(args[3], "opus");
    }
}
