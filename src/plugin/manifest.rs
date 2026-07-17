//! Plugin self-description: the `--aivo-manifest` probe + the manifest schema.
//! A conforming `aivo-<name>` prints one JSON manifest on `--aivo-manifest` and
//! exits 0; legacy/non-conforming plugins fail the probe and are recorded without
//! a manifest. Frozen contract: `docs/PLUGIN-PROTOCOL.md`.

use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::style;

/// Plugin protocol version this host speaks; bumped only on a breaking change.
pub(crate) const PROTOCOL_VERSION: &str = "1";

/// Grace period for a plugin to print its manifest before the probe gives up.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// A plugin's self-description, captured at install/update and cached verbatim in
/// `.registry.json`. Unknown fields are ignored (forward-compatible). In
/// protocol v1, `endpoint` is the only capability with host behavior; the rest
/// are disclosure/reserved vocabulary for future protocol revisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PluginManifest {
    pub name: String,
    pub version: String,
    /// Protocol the plugin targets; must equal `PROTOCOL_VERSION` to be honored.
    pub protocol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// What the plugin *is* — see [`PluginKind`]. Orthogonal to `roles` (how
    /// aivo runs it) and `capabilities` (what it's granted). Absent for
    /// plugins that don't self-classify.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<PluginKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub roles: Vec<String>,
    /// True when the plugin's own `--help` already documents aivo's injected
    /// flags (`-k`/`-m`/`--debug`), so aivo omits its help banner. Default
    /// false: aivo prepends the banner — the only place those flags appear for a
    /// thin wrapper whose `--help` is the wrapped tool's.
    #[serde(default, skip_serializing_if = "is_false")]
    pub documents_aivo_flags: bool,
    /// Requested capabilities. Only `endpoint` is grantable in protocol v1;
    /// reserved/disclosure caps are stored verbatim but ignored at launch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Reserved (P2 hooks); stored verbatim, not acted on in v1.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hooks: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// Where the plugin's session transcripts live + their format, so
    /// `aivo share` can read a plugin run with a built-in reader (the plugin
    /// stores its own transcript; aivo never sees it otherwise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcripts: Option<TranscriptSource>,
    /// External executables the plugin needs on PATH (e.g. the agent it wraps).
    /// aivo checks these at install and offers to run their `install` command —
    /// the same consent-gated flow native tools get. The command is authored by
    /// the plugin (only it knows how to install its dependency).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<Requirement>,
}

/// Plugin `type` vocabulary, closed in protocol v1. Only `CodingAgent` has
/// host behavior (argv ownership, run accounting, stats probe); `Media` is
/// reserved. Unrecognized values land in `Other` — kept verbatim so a future
/// additive type never invalidates a manifest — and warn at probe time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum PluginKind {
    CodingAgent,
    Tool,
    Media,
    #[serde(untagged)]
    Other(String),
}

impl PluginKind {
    pub(crate) fn as_str(&self) -> &str {
        match self {
            PluginKind::CodingAgent => "coding-agent",
            PluginKind::Tool => "tool",
            PluginKind::Media => "media",
            PluginKind::Other(t) => t,
        }
    }
}

impl std::fmt::Display for PluginKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PluginManifest {
    pub(crate) fn is_coding_agent(&self) -> bool {
        matches!(self.kind, Some(PluginKind::CodingAgent))
    }
}

/// A plugin's transcript source for `aivo share`: either a built-in `format`
/// (`pi`/`codex`/`gemini`/`opencode`) aivo reads from `dir`, or `format:
/// "native"` — the plugin emits its own transcript via `--aivo-export-transcript`
/// (no `dir`), keeping its own `source_cli`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TranscriptSource {
    pub format: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dir: String,
}

/// An executable the plugin needs on PATH, with an optional install command
/// aivo shows + runs on consent (`None` → aivo only reports it's missing).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Requirement {
    pub bin: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install: Option<String>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Capabilities that hand the plugin real power, so they require explicit
/// consent before aivo grants them. In protocol v1 this is only `endpoint`: a
/// routed loopback proxy bound to the selected key. Other capability names are
/// parsed for disclosure/forward-compatibility but are not granted.
pub(crate) fn is_grantable_cap(cap: &str) -> bool {
    matches!(cap, "endpoint")
}

/// Normalize the subset of a manifest's capability list that can actually be
/// granted by this host version. Keeps manifest order but removes duplicates and
/// reserved/unknown vocabulary.
pub(crate) fn grantable_capabilities(capabilities: &[String]) -> Vec<String> {
    let mut grantable = Vec::new();
    for cap in capabilities {
        if is_grantable_cap(cap) && !grantable.contains(cap) {
            grantable.push(cap.clone());
        }
    }
    grantable
}

/// Parse a manifest from a plugin's `--aivo-manifest` stdout. Tolerant of leading
/// log noise: tries the whole trimmed output, then the last non-empty line. Yields
/// `None` unless the JSON parses *and* declares a supported protocol (an
/// unsupported protocol is treated as "no manifest" — the plugin still dispatches).
/// A name mismatch warns but is not fatal; the on-disk name stays authoritative.
pub(crate) fn parse_manifest(stdout: &str, expected_name: &str) -> Option<PluginManifest> {
    let manifest = serde_json::from_str::<PluginManifest>(stdout.trim())
        .ok()
        .or_else(|| {
            let last = stdout.lines().rev().find(|l| !l.trim().is_empty())?;
            serde_json::from_str::<PluginManifest>(last.trim()).ok()
        })?;

    if manifest.protocol != PROTOCOL_VERSION {
        eprintln!(
            "  {} plugin manifest targets protocol `{}` (this aivo speaks `{}`) — ignoring its declared roles/capabilities",
            style::yellow("!"),
            manifest.protocol,
            PROTOCOL_VERSION,
        );
        return None;
    }
    if let Some(PluginKind::Other(t)) = &manifest.kind {
        eprintln!(
            "  {} plugin type `{}` is not recognized (coding-agent, tool, media) — recorded but inert",
            style::yellow("!"),
            t,
        );
    }
    if manifest.name != expected_name {
        eprintln!(
            "  {} plugin manifest name `{}` differs from the installed name `{}` — keeping `{}`",
            style::yellow("!"),
            manifest.name,
            expected_name,
            expected_name,
        );
    }
    Some(manifest)
}

/// Run `bin --aivo-manifest` and parse its output. Best-effort: any spawn error,
/// timeout, non-zero exit, or unparseable output yields `None`, and the plugin is
/// then recorded without a manifest — a failed probe is never an install error.
pub(crate) async fn probe_manifest(bin: &Path, name: &str) -> Option<PluginManifest> {
    let mut cmd = tokio::process::Command::new(bin);
    cmd.arg("--aivo-manifest")
        .env("AIVO_MANIFEST_PROBE", "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true);

    // On timeout the future is dropped → kill_on_drop reaps the child.
    let output = tokio::time::timeout(super::probe_timeout(PROBE_TIMEOUT), cmd.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    parse_manifest(&String::from_utf8_lossy(&output.stdout), name)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"{"name":"amp","version":"0.1.0","protocol":"1",
        "roles":["subcommand"],"capabilities":["endpoint","spawn"]}"#;

    #[test]
    fn valid_manifest_parses() {
        let m = parse_manifest(VALID, "amp").expect("should parse");
        assert_eq!(m.version, "0.1.0");
        assert_eq!(m.roles, ["subcommand"]);
        assert_eq!(m.capabilities, ["endpoint", "spawn"]);
    }

    #[test]
    fn unsupported_protocol_is_rejected() {
        let other = r#"{"name":"x","version":"1","protocol":"2"}"#;
        assert!(parse_manifest(other, "x").is_none());
    }

    #[test]
    fn garbage_is_none() {
        assert!(parse_manifest("not json at all", "x").is_none());
        assert!(parse_manifest("", "x").is_none());
        // Valid JSON, wrong shape (missing required fields) → None.
        assert!(parse_manifest(r#"{"hello":"world"}"#, "x").is_none());
    }

    #[test]
    fn name_mismatch_warns_but_parses() {
        let m = parse_manifest(VALID, "renamed").expect("name mismatch is non-fatal");
        assert_eq!(m.name, "amp");
    }

    #[test]
    fn manifest_after_log_noise_uses_last_line() {
        let noisy = format!("loading config...\nready\n{}", VALID.replace('\n', " "));
        let m = parse_manifest(&noisy, "amp").expect("last-line fallback");
        assert_eq!(m.version, "0.1.0");
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let extra = r#"{"name":"a","version":"1","protocol":"1","futureField":42}"#;
        assert!(parse_manifest(extra, "a").is_some());
    }

    #[test]
    fn type_field_parses_and_defaults_to_none() {
        let typed = r#"{"name":"omp","version":"1","protocol":"1","type":"coding-agent"}"#;
        let m = parse_manifest(typed, "omp").unwrap();
        assert_eq!(m.kind, Some(PluginKind::CodingAgent));
        assert!(m.is_coding_agent());
        // A manifest without `type` round-trips to None, not an error.
        assert!(parse_manifest(VALID, "amp").unwrap().kind.is_none());
    }

    #[test]
    fn known_types_round_trip_kebab_case() {
        for (raw, kind) in [
            ("coding-agent", PluginKind::CodingAgent),
            ("tool", PluginKind::Tool),
            ("media", PluginKind::Media),
        ] {
            let json = format!(r#"{{"name":"x","version":"1","protocol":"1","type":"{raw}"}}"#);
            let m = parse_manifest(&json, "x").unwrap();
            assert_eq!(m.kind, Some(kind));
            assert!(
                serde_json::to_string(&m)
                    .unwrap()
                    .contains(&format!("\"type\":\"{raw}\""))
            );
        }
    }

    #[test]
    fn unknown_type_round_trips_verbatim_and_is_omitted_when_absent() {
        let m = parse_manifest(
            r#"{"name":"x","version":"1","protocol":"1","type":"tts"}"#,
            "x",
        )
        .unwrap();
        assert_eq!(m.kind, Some(PluginKind::Other("tts".to_string())));
        assert!(!m.is_coding_agent());
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"type\":\"tts\""));
        // Absent type serializes away entirely (skip_serializing_if).
        let bare = parse_manifest(r#"{"name":"x","version":"1","protocol":"1"}"#, "x").unwrap();
        assert!(!serde_json::to_string(&bare).unwrap().contains("type"));
    }

    #[test]
    fn transcripts_field_parses() {
        let m = parse_manifest(
            r#"{"name":"omp","version":"1","protocol":"1",
                "transcripts":{"format":"pi","dir":"~/.omp/agent/sessions"}}"#,
            "omp",
        )
        .unwrap();
        let t = m.transcripts.unwrap();
        assert_eq!(t.format, "pi");
        assert_eq!(t.dir, "~/.omp/agent/sessions");
        // Absent → None (and omitted on re-serialize).
        let bare = parse_manifest(r#"{"name":"x","version":"1","protocol":"1"}"#, "x").unwrap();
        assert!(bare.transcripts.is_none());
        assert!(
            !serde_json::to_string(&bare)
                .unwrap()
                .contains("transcripts")
        );
    }

    #[test]
    fn requires_field_parses() {
        let m = parse_manifest(
            r#"{"name":"omp","version":"1","protocol":"1",
                "requires":[{"bin":"omp","install":"curl x | sh"},{"bin":"node"}]}"#,
            "omp",
        )
        .unwrap();
        assert_eq!(m.requires.len(), 2);
        assert_eq!(m.requires[0].bin, "omp");
        assert_eq!(m.requires[0].install.as_deref(), Some("curl x | sh"));
        assert_eq!(m.requires[1].bin, "node");
        assert!(m.requires[1].install.is_none());
        // Absent → empty, omitted on re-serialize.
        let bare = parse_manifest(r#"{"name":"x","version":"1","protocol":"1"}"#, "x").unwrap();
        assert!(bare.requires.is_empty());
        assert!(!serde_json::to_string(&bare).unwrap().contains("requires"));
    }

    #[test]
    fn grantable_caps_classified() {
        assert!(is_grantable_cap("endpoint"), "endpoint should gate consent");
        // These are parsed for disclosure/forward-compatibility but never
        // granted in protocol v1.
        for c in [
            "config-read",
            "config-write",
            "spawn",
            "hook:launch.pre",
            "unknown-key-handoff",
            "",
            "unknown",
        ] {
            assert!(!is_grantable_cap(c), "{c} must not gate consent");
        }
    }

    #[test]
    fn grantable_capabilities_filters_reserved_and_dedupes() {
        let caps = vec![
            "config-read".to_string(),
            "endpoint".to_string(),
            "endpoint".to_string(),
            "unknown-key-handoff".to_string(),
            "config-write".to_string(),
        ];
        assert_eq!(grantable_capabilities(&caps), ["endpoint"]);
    }
}
