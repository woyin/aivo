//! Regenerates the model-limits snapshot from live `models.dev/api.json` for
//! `aivo update --sync-model-data`. Rust port of `sync_model_limits.py`; output
//! is the `{"models": {…}}` shape `model_metadata::SnapshotFile` reads.
//!
//! Row per normalized id: `[context|null, output|null, "flags", "efforts"]`.
//! flags: `t`=tool_call, `r`=reasoning, `a`=attachment, `i`=image input,
//! `g`=image output,
//! `f`=rejects temperature, `d`=deprecated.
//!
//! Keep the winner/normalization rules in lockstep with the Python generator —
//! drift makes the override disagree with the embedded snapshot it replaces.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const GENERATED_NOTE: &str = "by aivo update from live models.dev/api.json — do not hand-edit";

/// First-party vendors whose listings qualify without cross-provider
/// corroboration. Mirrors `AUTHORITATIVE` in `sync_model_limits.py`.
const AUTHORITATIVE: &[&str] = &[
    "alibaba",
    "alibaba-cn",
    "alibaba-token-plan",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "google-vertex",
    "inception",
    "llama",
    "minimax",
    "minimax-cn",
    "mistral",
    "moonshotai",
    "moonshotai-cn",
    "nvidia",
    "openai",
    "perplexity",
    "poolside",
    "stepfun",
    "stepfun-ai",
    "upstage",
    "xai",
    "xiaomi",
    "zai",
    "zhipuai",
];

fn is_authoritative(provider: &str) -> bool {
    AUTHORITATIVE.contains(&provider)
}

// ── models.dev api.json schema (only the fields we read) ───────────────────

#[derive(Deserialize)]
struct ApiProvider {
    #[serde(default)]
    models: BTreeMap<String, ApiModel>,
}

#[derive(Deserialize)]
struct ApiModel {
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    attachment: bool,
    /// `Some(false)` means the model rejects the `temperature` param.
    #[serde(default)]
    temperature: Option<bool>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    modalities: Modalities,
    #[serde(default)]
    limit: Limit,
    #[serde(default)]
    reasoning_options: Vec<ReasoningOption>,
}

#[derive(Deserialize, Default)]
struct Modalities {
    #[serde(default, deserialize_with = "lenient_string_vec")]
    input: Vec<String>,
    /// `None` = field absent/null (keep); `Some(list)` without "text" = prune.
    #[serde(default, deserialize_with = "lenient_opt_string_vec")]
    output: Option<Vec<String>>,
}

/// Keeps only string elements — models.dev arrays sometimes carry `null`.
fn strings_only(values: Vec<serde_json::Value>) -> Vec<String> {
    values
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        })
        .collect()
}

fn lenient_string_vec<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<String>, D::Error> {
    let raw: Option<Vec<serde_json::Value>> = Option::deserialize(d)?;
    Ok(strings_only(raw.unwrap_or_default()))
}

fn lenient_opt_string_vec<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<Vec<String>>, D::Error> {
    let raw: Option<Vec<serde_json::Value>> = Option::deserialize(d)?;
    Ok(raw.map(strings_only))
}

#[derive(Deserialize, Default)]
struct Limit {
    #[serde(default)]
    context: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
}

#[derive(Deserialize)]
struct ReasoningOption {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default, deserialize_with = "lenient_string_vec")]
    values: Vec<String>,
}

// ── transform ──────────────────────────────────────────────────────────────

type Row = (Option<u64>, Option<u64>, String, String);

#[derive(Serialize)]
struct OverrideFile {
    #[serde(rename = "_generated")]
    generated: &'static str,
    models: BTreeMap<String, Row>,
}

/// One provider's observation of a single (normalized) model id.
struct Obs {
    provider: String,
    ctx: Option<u64>,
    out: Option<u64>,
    flags: String,
    temp: Option<bool>,
    deprecated: bool,
    efforts: Option<Vec<String>>,
}

/// Parses `models.dev/api.json` and returns `(snapshot_json, model_count)`.
pub fn transform(api_json: &str) -> Result<(String, usize)> {
    let data: BTreeMap<String, ApiProvider> =
        serde_json::from_str(api_json).context("models.dev api.json is not the expected shape")?;
    let observations = collect(&data);
    let models: BTreeMap<String, Row> = observations
        .iter()
        .map(|(id, rows)| (id.clone(), pick_winner(rows)))
        .collect();
    let count = models.len();
    if count == 0 {
        anyhow::bail!("models.dev returned no usable models");
    }
    let json = serde_json::to_string_pretty(&OverrideFile {
        generated: GENERATED_NOTE,
        models,
    })
    .context("Failed to serialize model snapshot")?;
    Ok((json, count))
}

/// Groups per-provider observations by normalized model id, pruning entries
/// with no limits and models whose `modalities.output` lacks `"text"`.
fn collect(data: &BTreeMap<String, ApiProvider>) -> BTreeMap<String, Vec<Obs>> {
    let mut observations: BTreeMap<String, Vec<Obs>> = BTreeMap::new();
    for (provider_id, provider) in data {
        for (model_id, entry) in &provider.models {
            if let Some(out) = &entry.modalities.output
                && !out.iter().any(|m| m == "text")
            {
                continue; // non-text output (embeddings / image / audio gen)
            }
            let ctx = entry.limit.context.and_then(pos_u64);
            let out = entry.limit.output.and_then(pos_u64);
            if ctx.is_none() && out.is_none() {
                continue; // nothing to record
            }
            observations
                .entry(normalize(model_id))
                .or_default()
                .push(Obs {
                    provider: provider_id.clone(),
                    ctx,
                    out,
                    flags: flags_for(entry),
                    temp: entry.temperature,
                    deprecated: entry.status.as_deref() == Some("deprecated"),
                    efforts: effort_values(entry),
                });
        }
    }
    observations
}

/// lowercase, last `/`-segment, `:variant` stripped, trailing `-YYYYMMDD`
/// stripped. MUST match `model_metadata::snapshot_limits`' lookup folding.
fn normalize(model_id: &str) -> String {
    let lower = model_id.trim().to_ascii_lowercase();
    let after_slash = lower.rsplit('/').next().unwrap_or(&lower);
    let seg = match after_slash.split(':').next() {
        Some(s) if !s.is_empty() => s,
        _ => after_slash,
    };
    if let Some((stem, suffix)) = seg.rsplit_once('-')
        && suffix.len() == 8
        && suffix.bytes().all(|b| b.is_ascii_digit())
    {
        return stem.to_string();
    }
    seg.to_string()
}

fn pos_u64(v: f64) -> Option<u64> {
    (v > 0.0).then_some(v as u64)
}

fn flags_for(e: &ApiModel) -> String {
    let mut f = String::new();
    if e.tool_call {
        f.push('t');
    }
    if e.reasoning {
        f.push('r');
    }
    if e.attachment {
        f.push('a');
    }
    if e.modalities.input.iter().any(|m| m == "image") {
        f.push('i');
    }
    if e.modalities
        .output
        .as_ref()
        .is_some_and(|o| o.iter().any(|m| m == "image"))
    {
        f.push('g');
    }
    f
}

/// Values of the first `effort`-type reasoning option, or `None`.
fn effort_values(e: &ApiModel) -> Option<Vec<String>> {
    for opt in &e.reasoning_options {
        if opt.kind.as_deref() == Some("effort") {
            return (!opt.values.is_empty()).then(|| opt.values.clone());
        }
    }
    None
}

/// OR of `t/r/a/i` flag chars across observations, emitted in canonical order.
/// `g` is NOT OR'd — see `pick_winner`: it takes a vendor verdict like `r/f/d`.
fn or_flags(rows: &[Obs]) -> String {
    let set: HashSet<char> = rows.iter().flat_map(|r| r.flags.chars()).collect();
    "trai".chars().filter(|c| set.contains(c)).collect()
}

/// First-party vendor's value when one reports, else the strict majority of
/// reported values (a tie at the top, or nothing reported, yields `None`).
fn vendor_verdict<T, F>(rows: &[Obs], getter: F) -> Option<T>
where
    T: Eq + Hash + Clone,
    F: Fn(&Obs) -> Option<T>,
{
    let first_party: Vec<T> = rows
        .iter()
        .filter(|r| is_authoritative(&r.provider))
        .filter_map(&getter)
        .collect();
    let pool: Vec<T> = if first_party.is_empty() {
        rows.iter().filter_map(&getter).collect()
    } else {
        first_party
    };
    strict_mode(&pool)
}

/// The single most-frequent value, or `None` when the top is tied (no clear
/// winner) or the pool is empty.
fn strict_mode<T: Eq + Hash + Clone>(values: &[T]) -> Option<T> {
    if values.is_empty() {
        return None;
    }
    let mut counts: HashMap<&T, usize> = HashMap::new();
    for v in values {
        *counts.entry(v).or_default() += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    let mut at_max = counts.iter().filter(|&(_, &c)| c == max).map(|(&v, _)| v);
    let winner = at_max.next().cloned();
    if at_max.next().is_some() {
        None // tie at the top
    } else {
        winner
    }
}

/// Most-frequent value, breaking ties toward the smallest. Used for output
/// tokens, where a tie should not discard the data.
fn modal_min(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v).or_default() += 1;
    }
    let max = counts.values().copied().max().unwrap_or(0);
    counts
        .iter()
        .filter(|&(_, &c)| c == max)
        .map(|(&v, _)| v)
        .min()
}

/// Distinct `(provider, value)` pairs collapsed to their values, so one
/// aggregator listing many variants of the same id counts once per value.
fn distinct_provider_values(rows: &[&Obs], get: impl Fn(&Obs) -> Option<u64>) -> Vec<u64> {
    let pairs: HashSet<(String, u64)> = rows
        .iter()
        .filter_map(|r| get(r).map(|v| (r.provider.clone(), v)))
        .collect();
    pairs.into_iter().map(|(_, v)| v).collect()
}

fn pick_winner(rows: &[Obs]) -> Row {
    // Context: max among values corroborated by >=2 providers or from a
    // first-party vendor, else plain max. Counted per provider so one
    // aggregator's typo (a 20M grok) can't outvote the field.
    let ctx_pairs: HashSet<(String, u64)> = rows
        .iter()
        .filter_map(|r| r.ctx.map(|c| (r.provider.clone(), c)))
        .collect();
    let mut ctx_counts: HashMap<u64, usize> = HashMap::new();
    let mut authoritative_ctx: HashSet<u64> = HashSet::new();
    for (provider, c) in &ctx_pairs {
        *ctx_counts.entry(*c).or_default() += 1;
        if is_authoritative(provider) {
            authoritative_ctx.insert(*c);
        }
    }
    let qualified: Vec<u64> = ctx_counts
        .iter()
        .filter(|&(c, &n)| n >= 2 || authoritative_ctx.contains(c))
        .map(|(&c, _)| c)
        .collect();
    let ctx = qualified
        .into_iter()
        .max()
        .or_else(|| ctx_counts.keys().copied().max());

    // Output: modal among the chosen-context entries; fall back to all rows.
    let pool: Vec<&Obs> = match ctx {
        Some(c) => rows.iter().filter(|r| r.ctx == Some(c)).collect(),
        None => rows.iter().collect(),
    };
    let mut out_vals = distinct_provider_values(&pool, |r| r.out);
    if out_vals.is_empty() {
        let all: Vec<&Obs> = rows.iter().collect();
        out_vals = distinct_provider_values(&all, |r| r.out);
    }
    let out = modal_min(&out_vals);

    let mut flags = or_flags(rows);
    // `g` drives the generate_image picker/tool, so a lone noisy aggregator
    // (neon branded the whole gpt-5 family image-emitting) must not set it —
    // vendor-verdict-else-majority, like `r/f/d`.
    if vendor_verdict(rows, |r| Some(r.flags.contains('g'))) == Some(true) {
        flags.push('g');
    }
    if vendor_verdict(rows, |r| r.temp) == Some(false) {
        flags.push('f');
    }
    if vendor_verdict(rows, |r| Some(r.deprecated)) == Some(true) {
        flags.push('d');
    }
    let efforts = vendor_verdict(rows, |r| r.efforts.clone()).unwrap_or_default();
    (ctx, out, flags, efforts.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    // A small api.json exercising normalization-merge, authoritative-context
    // selection over a stale aggregator, the temperature/efforts vendor
    // verdicts, and both prune paths.
    const FIXTURE: &str = r#"{
      "openai": {
        "name": "OpenAI",
        "models": {
          "gpt-x": {
            "tool_call": true, "reasoning": true, "temperature": false,
            "modalities": {"input": ["text","image"], "output": ["text"]},
            "limit": {"context": 400000, "output": 128000},
            "reasoning_options": [{"type": "effort", "values": ["low","medium","high"]}]
          }
        }
      },
      "aggregator": {
        "name": "Agg",
        "models": {
          "openai/gpt-x-20240101": {
            "tool_call": true, "temperature": true,
            "modalities": {"input": ["text"], "output": ["text"]},
            "limit": {"context": 128000, "output": 64000}
          }
        }
      },
      "embeddings": {
        "name": "E",
        "models": {
          "embed-1": {
            "modalities": {"input": ["text"], "output": ["embedding"]},
            "limit": {"context": 8192}
          }
        }
      },
      "broken": {
        "name": "B",
        "models": {
          "no-limits": {"modalities": {"output": ["text"]}}
        }
      }
    }"#;

    fn rows_of(json: &str) -> Value {
        serde_json::from_str::<Value>(json).unwrap()["models"].clone()
    }

    #[test]
    fn transform_merges_normalizes_prunes_and_picks_winner() {
        let (json, count) = transform(FIXTURE).unwrap();
        assert_eq!(count, 1, "only gpt-x survives pruning");
        let models = rows_of(&json);
        // embed-1 (non-text output) and no-limits (no limits) are pruned.
        assert!(models.get("embed-1").is_none());
        assert!(models.get("no-limits").is_none());
        // Dated + provider-prefixed listing merged into the plain id.
        let row = models.get("gpt-x").expect("gpt-x present");
        // [context, output, flags, efforts]
        assert_eq!(row[0], 400000, "authoritative ctx beats stale aggregator");
        assert_eq!(row[1], 128000, "output modal among chosen-ctx entries");
        assert_eq!(
            row[2], "trif",
            "t|r|i OR'd, +f from vendor temperature=false"
        );
        assert_eq!(row[3], "low,medium,high", "vendor effort levels");
    }

    #[test]
    fn corroboration_kills_single_provider_typo() {
        // Three non-authoritative providers; two agree on 200k, one lists a 20M
        // typo. The corroborated value wins.
        let api = r#"{
          "a": {"models": {"m": {"limit": {"context": 200000, "output": 8192},
                                 "modalities": {"output": ["text"]}}}},
          "b": {"models": {"m": {"limit": {"context": 200000, "output": 8192},
                                 "modalities": {"output": ["text"]}}}},
          "c": {"models": {"m": {"limit": {"context": 20000000, "output": 8192},
                                 "modalities": {"output": ["text"]}}}}
        }"#;
        let (json, _) = transform(api).unwrap();
        assert_eq!(rows_of(&json)["m"][0], 200000);
    }

    #[test]
    fn plain_max_when_nothing_qualifies() {
        // Two distinct non-authoritative single-provider values, neither
        // corroborated: fall back to plain max.
        let api = r#"{
          "a": {"models": {"m": {"limit": {"context": 100000},
                                 "modalities": {"output": ["text"]}}}},
          "b": {"models": {"m": {"limit": {"context": 250000},
                                 "modalities": {"output": ["text"]}}}}
        }"#;
        let (json, _) = transform(api).unwrap();
        assert_eq!(rows_of(&json)["m"][0], 250000);
    }

    #[test]
    fn deprecated_uses_vendor_verdict() {
        // First-party vendor marks it active; a lone aggregator says deprecated.
        // Vendor wins -> no `d` flag.
        let api = r#"{
          "anthropic": {"models": {"m": {"status": "active",
                                 "limit": {"context": 200000},
                                 "modalities": {"output": ["text"]}}}},
          "groq": {"models": {"m": {"status": "deprecated",
                                 "limit": {"context": 200000},
                                 "modalities": {"output": ["text"]}}}}
        }"#;
        let (json, _) = transform(api).unwrap();
        assert_eq!(rows_of(&json)["m"][2], "");
    }

    #[test]
    fn image_output_uses_vendor_verdict() {
        // A lone aggregator brands the model image-emitting (neon did this to
        // the whole gpt-5 family); the vendor says text-only. Vendor wins → no
        // `g`. The inverse — vendor confirms image output — sets it.
        let api = r#"{
          "openai": {"models": {"m": {"limit": {"context": 400000},
                                 "modalities": {"output": ["text"]}}}},
          "neon": {"models": {"m": {"limit": {"context": 400000},
                                 "modalities": {"output": ["text", "image"]}}}},
          "google": {"models": {"pic": {"limit": {"context": 32768},
                                 "modalities": {"output": ["text", "image"]}}}}
        }"#;
        let (json, _) = transform(api).unwrap();
        assert_eq!(rows_of(&json)["m"][2], "", "lone aggregator can't set g");
        assert_eq!(rows_of(&json)["pic"][2], "g", "vendor-confirmed g sticks");
    }

    #[test]
    fn empty_source_is_an_error() {
        assert!(transform("{}").is_err());
    }

    #[test]
    fn output_parses_back_into_snapshot_shape() {
        // The override file must read as the same shape model_metadata expects.
        let (json, _) = transform(FIXTURE).unwrap();
        let parsed: Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["models"]["gpt-x"].is_array());
        assert!(parsed["_generated"].is_string());
    }
}
