//! Plugin-provided usage stats: the `--aivo-stats` probe + `aivo.stats/v1`
//! schema. A coding-agent plugin knows its own data folder and format, so it —
//! not aivo core — reads its sessions and emits them as normalized, **per-session
//! timestamped** records. aivo pulls on demand (`aivo-<name> --aivo-stats
//! --json`) and owns all filtering/aggregation: `--since` windowing, model-name
//! normalization, and totals are applied host-side, consistently with native
//! tools. The plugin only provides data. Best-effort: coding-agent plugins are
//! probed automatically (other types opt in via the `stats` cap), and aivo falls
//! back to its own proxy accounting / launch counts when a report is absent.

use std::time::Duration;

use serde::Deserialize;

const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const SCHEMA: &str = "aivo.stats/v1";

/// A plugin's raw usage report: one entry per session, each timestamped so aivo
/// can window it. Aggregation and `--since` filtering happen host-side.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PluginStatsReport {
    /// Human-readable provenance shown to the user (e.g. "aivo-routed amp threads").
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub sessions: Vec<SessionStat>,
}

/// One session's per-model token usage plus its timestamp (RFC3339). `ts` is
/// `None` for sessions a plugin can't place in time — aivo then can't window them.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct SessionStat {
    #[serde(default)]
    pub ts: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelStat>,
}

/// Per-model token usage within a session. Token fields default to 0 so a
/// plugin can omit dimensions it doesn't track.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ModelStat {
    pub name: String,
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

/// True when aivo should pull `name`'s own `--aivo-stats` report. A
/// **coding-agent** plugin reports usage by nature, so it's probed automatically
/// (no `stats` cap needed); any other plugin type opts in by declaring the
/// `stats` capability. The probe is best-effort (`probe_stats`), so a plugin that
/// doesn't actually implement `--aivo-stats` just yields no report. Either way
/// it's disclosure-only (the plugin reads its own data), so no consent grant is
/// required.
pub(crate) fn probes_stats(name: &str) -> bool {
    super::registry::load()
        .plugins
        .get(name)
        .and_then(|r| r.manifest.as_ref())
        .is_some_and(|m| m.is_coding_agent() || m.capabilities.iter().any(|c| c == "stats"))
}

/// Run `aivo-<name> --aivo-stats --json` and parse its report. Best-effort: a
/// missing binary, spawn error, timeout, non-zero exit, or unparseable /
/// wrong-schema output yields `None` so the caller falls back. aivo passes no
/// filters — it windows/aggregates the returned sessions itself.
pub(crate) async fn probe_stats(name: &str) -> Option<PluginStatsReport> {
    let bin = super::discover(name)?;
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("--aivo-stats")
        .arg("--json")
        .env("AIVO_STATS_PROBE", "1")
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
    parse_report(&String::from_utf8_lossy(&output.stdout))
}

/// Parse one `aivo.stats/v1` object from probe stdout. Tolerant of leading
/// banner lines (slices to the outermost `{`…`}`); rejects other schemas.
fn parse_report(stdout: &str) -> Option<PluginStatsReport> {
    let start = stdout.find('{')?;
    let end = stdout.rfind('}')?;
    let value: serde_json::Value = serde_json::from_str(stdout.get(start..=end)?).ok()?;
    if value.get("schema").and_then(|s| s.as_str()) != Some(SCHEMA) {
        return None;
    }
    serde_json::from_value(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_report_reads_v1_sessions_and_ignores_banner() {
        let out = r#"[amp] reading threads…
{"schema":"aivo.stats/v1","source":"aivo-routed amp threads",
 "sessions":[
   {"ts":"2026-06-08T01:00:00Z","models":[
     {"name":"deepseek-v4-flash","output_tokens":410000,"input_tokens":9}]},
   {"models":[{"name":"gpt-5.4","output_tokens":60000}]}
 ]}"#;
        let r = parse_report(out).expect("parses");
        assert_eq!(r.source.as_deref(), Some("aivo-routed amp threads"));
        assert_eq!(r.sessions.len(), 2);
        assert_eq!(r.sessions[0].ts.as_deref(), Some("2026-06-08T01:00:00Z"));
        assert_eq!(r.sessions[0].models[0].name, "deepseek-v4-flash");
        assert_eq!(r.sessions[0].models[0].output_tokens, 410000);
        assert!(r.sessions[1].ts.is_none());
    }

    #[test]
    fn parse_report_rejects_wrong_schema() {
        assert!(parse_report(r#"{"schema":"other/v9","sessions":[]}"#).is_none());
        assert!(parse_report("not json").is_none());
        assert!(parse_report(r#"{"sessions":[]}"#).is_none());
    }
}
